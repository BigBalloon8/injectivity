from __future__ import annotations

import math
import torch


def null_space(A, rcond=None):
    u, s, vh = torch.linalg.svd(A)
    
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = torch.finfo(s.dtype).eps * max(M, N)
    s = torch.nan_to_num(s, 0)
    tol = torch.amax(s) * rcond
    num = torch.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q


def project_kernel_into_safe_region(
    M: torch.Tensor,
    A: torch.Tensor,
    gamma_lower: float,
    safety_factor: float = 0.99,
    tolerance: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Project ker(M) into a certified safe gap ball around col(A)^perp.

    Parameters
    ----------
    M:
        Matrix of shape (n, m). It should have full row rank n.

    A:
        Matrix of shape (m, n). It should have full column rank n.

    gamma_lower:
        A certified lower bound on gamma_0(A, c).
        It must satisfy 0 < gamma_lower <= gamma_0.

    safety_factor:
        Number strictly between 0 and 1. The target gap is
        safety_factor * gamma_lower.

    tolerance:
        Numerical tolerance.

    Returns
    -------
    M_safe:
        A matrix whose kernel lies strictly inside the certified safe region.

    U_safe:
        An orthonormal basis for ker(M_safe), with shape (m, m-n).

    diagnostics:
        Original and final gap information.
    """
    if not isinstance(M, torch.Tensor):
        M = torch.tensor(M)
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A)

    if M.ndim != 2 or A.ndim != 2:
        raise ValueError("M and A must both be matrices.")

    n, m = M.shape

    if A.shape != (m, n):
        raise ValueError(
            f"Expected A to have shape {(m, n)}, but got {A.shape}."
        )

    if torch.linalg.matrix_rank(A, tol=tolerance) != n:
        raise ValueError("A must have full column rank.")

    if torch.linalg.matrix_rank(M, tol=tolerance) != n:
        raise ValueError(
            "M must have full row rank. Otherwise dim ker(M) is larger "
            "than dim col(A)^perp, and their projector gap is 1."
        )

    if not 0.0 < gamma_lower <= 1.0:
        raise ValueError("gamma_lower must lie in (0, 1].")

    if not 0.0 < safety_factor < 1.0:
        raise ValueError("safety_factor must lie strictly between 0 and 1.")

    # Orthonormal bases for K = ker(M) and K0 = ker(A^T).
    U = null_space(M, rcond=tolerance)
    U0 = null_space(A.T, rcond=tolerance)

    if U.shape != U0.shape:
        raise RuntimeError(
            f"Kernel dimensions disagree: {U.shape[1]} versus {U0.shape[1]}."
        )

    # Principal-angle decomposition:
    #
    # U0.T @ U = L diag(cos(theta_i)) R.T
    L, cos_theta, Rt = torch.linalg.svd(
        U0.T @ U,
        full_matrices=False,
    )

    cos_theta = torch.clip(cos_theta, 0.0, 1.0)
    theta = torch.arccos(cos_theta)

    R = Rt.T

    # Corresponding principal vectors.
    P = U0 @ L       # Principal vectors in K0
    Q = U @ R        # Principal vectors in K

    target_gap = torch.tensor(safety_factor * gamma_lower).to(M.device)
    theta_cap = torch.arcsin(target_gap)

    Q_safe = torch.empty_like(Q)

    for i, angle in enumerate(theta):
        # This direction is already safe.
        if angle <= theta_cap:
            Q_safe[:, i] = Q[:, i]
            continue

        sin_angle = torch.sin(angle)

        if sin_angle <= tolerance:
            # This should not occur when angle > theta_cap >= 0,
            # but it protects against numerical problems.
            Q_safe[:, i] = P[:, i]
            continue

        # Unit direction orthogonal to K0 in the principal two-plane.
        residual = (
            Q[:, i] - torch.cos(angle) * P[:, i]
        ) / sin_angle

        Q_safe[:, i] = (
            torch.cos(theta_cap) * P[:, i]
            + torch.sin(theta_cap) * residual
        )

    # Symmetric orthonormalisation to correct floating-point drift.
    gram = Q_safe.T @ Q_safe
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)

    if torch.min(eigenvalues) <= tolerance:
        raise RuntimeError("Projected kernel basis became numerically singular.")

    inverse_sqrt_gram = (
        eigenvectors
        @ torch.diag(1.0 / torch.sqrt(eigenvalues))
        @ eigenvectors.T
    )

    U_safe = Q_safe @ inverse_sqrt_gram

    P0 = U0 @ U0.T
    P_original = U @ U.T
    P_safe = U_safe @ U_safe.T

    original_gap = torch.linalg.norm(P_original - P0, ord=2)
    final_gap = torch.linalg.norm(P_safe - P0, ord=2)

    # Orthogonal projection of each row of M onto ker(U_safe.T).
    M_safe = M @ (torch.eye(m) - P_safe)

    final_rank = torch.linalg.matrix_rank(M_safe, tol=tolerance)

    if final_rank != n:
        raise RuntimeError(
            "The projected matrix lost row rank numerically. "
            "Use the orthonormal-row construction described below."
        )

    annihilation_error = torch.linalg.norm(M_safe @ U_safe, ord=2)

    diagnostics = {
        "principal_angles_radians": theta,
        "principal_angles_degrees": torch.rad2deg(theta),
        "angle_cap_radians": theta_cap,
        "angle_cap_degrees": torch.rad2deg(theta_cap),
        "original_gap": original_gap,
        "target_gap": target_gap,
        "final_gap": final_gap,
        "gamma_lower": gamma_lower,
        "matrix_change_frobenius": torch.linalg.norm(M_safe - M, ord="fro"),
        "annihilation_error": annihilation_error,
        "final_rank": final_rank,
    }

    return M_safe, U_safe, diagnostics


class CustomAdam(torch.optim.Optimizer):
    """Adam optimizer with projection.
 
    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr: learning rate (default: 1e-3).
        betas: coefficients for computing running averages of gradient and its
            square (default: (0.9, 0.999)).
        eps: term added to the denominator for numerical stability (default: 1e-8).
        weight_decay: weight decay coefficient (default: 0).
        decoupled_wd: if True, apply decoupled weight decay (AdamW). If False,
            add weight_decay * param to the gradient (classic L2). (default: False)
        amsgrad: whether to use the AMSGrad variant (default: False).
    """
 
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        decoupled_wd=False,
        amsgrad=False,
        gamma=0.3
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if 0 < gamma <= 1.0:
            raise ValueError(f"Invalid gamma value: {gamma}")

        
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_wd=decoupled_wd,
            amsgrad=amsgrad,
            gamma=gamma
        )
        super().__init__(params, defaults)
 
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.
 
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            decoupled_wd = group["decoupled_wd"]
            amsgrad = group["amsgrad"]
            gamma = group["gamma"]
 
            for p in group["params"]:
                if p.grad is None:
                    continue
 
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "CustomAdam does not support sparse gradients; "
                        "consider SparseAdam instead."
                    )
 
                state = self.state[p]
 
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values (1st moment)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values (2nd moment)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
 
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
 
                # Weight decay
                if weight_decay != 0:
                    if decoupled_wd:
                        # AdamW: decay the weights directly, decoupled from the gradient
                        p.mul_(1 - lr * weight_decay)
                    else:
                        # Classic L2: fold decay into the gradient
                        grad = grad.add(p, alpha=weight_decay)
 
                state["step"] += 1
                step = state["step"]
 
                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
 
                # Bias corrections
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
 
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    # Maintain the running max of the second moment
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
 
                step_size = lr / bias_correction1
 
                # Parameter update: p <- p - step_size * exp_avg / denom
                p.addcdiv_(exp_avg, denom, value=-step_size)
            
            if "up_proj" in group.keys():
                up_proj = p
                
 
        return loss


if __name__ == "__main__":
    import torch
    from analysis import find_kway_collisions
    from logger import Logger
    A = torch.randn(16, 4)
    b = torch.randn(16)
    B = torch.randn(4, 16)
    
    logger = Logger("projection_test", "./logs/projection.log")
    
    gammas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    
    #logger.log(f"Base: {find_kway_collisions(A, b, B, pairs=True, cache=False)}")
    
    for gamma in gammas:
        B_safe, K_safe_basis, info = project_kernel_into_safe_region(
            M=B,
            A=A,
            gamma_lower=gamma,
            safety_factor=0.95,
        )

        collisions = find_kway_collisions(A, b, torch.tensor(B_safe), pairs=True, cache=False)
        logger.log(f"g={gamma}: {collisions}")
        
        