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

@torch.compile
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
    M_safe = M  - M@P_safe

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


@torch.compile
def project_kernel_into_safe_region_smart(
    M: torch.Tensor,
    A: torch.Tensor,
    gamma_lower: float,
    safety_factor: float = 0.99,
    tolerance: float = 1e-10,
    alpha: float = 1.0,
    beta: float = 1.0
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

    P = (torch.eye(m) - P_safe)

    # Orthogonal projection of each row of M onto ker(U_safe.T).
    M_safe = M@P + (alpha/(alpha + beta))*B@(torch.eye(m)-P) #M  - M@P_safe

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


@torch.no_grad()
def nearest_with_kernel_angle_bound_fast(
    A, B, alpha, *,
    n_random_inits=0,       # extra random feasible starts, batched with the two default ones
    max_iter=1000,
    tol=None,               # stop when the projected-gradient norm drops below tol * its initial value
                            #   default 1e-6 (float64) / 1e-4 (float32)
    ftol=None,              # ...or when the loss decrease over the last 20 iterations is below
                            #   ftol * total decrease so far; default 1e-9 (float64) / 1e-6 (float32)
    dtype=torch.float64,    # working precision (float32 is ~2x faster on GPU, slightly less accurate)
    nonmonotone=10,         # GLL nonmonotone window for the line search (1 = monotone Armijo)
    seed=0,
    return_info=False,
    verbose=False,
):
    """
    A: (m, n) full column rank, B: (n, m), alpha in [0, pi/2).
    Returns B' (n, m) minimising ||B' - B||_F s.t. all principal angles between ker(B')
    and col(A)^perp are <= alpha.  Runs on the device of A and B.
    """
    device, out_dtype = B.device, B.dtype
    if tol is None:
        tol = 1e-6 if dtype == torch.float64 else 1e-4
    if ftol is None:
        ftol = 1e-9 if dtype == torch.float64 else 1e-6
    A = A.to(dtype=dtype)
    scale = B.norm()
    B = B.to(dtype=dtype) / scale               # normalise: problem is homogeneous in B
    m, n = A.shape
    t = math.tan(alpha)
    eye = torch.eye(n, device=device, dtype=dtype)

    # ------------------------------------------------------------ one-time O(m n^2) setup
    Q, _ = torch.linalg.qr(A)                    # orthonormal basis of col(A)
    K0 = B @ Q                                   # n x n
    N = B.T - Q @ K0.T                           # (I - P_A) B^T,  m x n
    UN, _ = torch.linalg.qr(N)                   # orthonormal basis of col(N)
    K1 = B @ UN                                  # n x n
    normB2 = (B * B).sum()

    # ------------------------------------------------------------ batched helpers, Y: (r, n, n)
    def project(Y):
        """Clip singular values of each Y[i] at t via eigh of Y^T Y (done in float64)."""
        if t == 0.0:
            return torch.zeros_like(Y)
        Yd = Y.double()
        lam, V = torch.linalg.eigh(Yd.transpose(-1, -2) @ Yd)
        sig = lam.clamp(min=0).sqrt()
        need = sig[:, -1] > t
        if not bool(need.any()):
            return Y
        f = torch.where(sig > t, t / sig.clamp(min=1e-300), torch.ones_like(sig))
        Yc = ((Yd @ V * f.unsqueeze(-2)) @ V.transpose(-1, -2)).to(Y.dtype)
        return torch.where(need[:, None, None], Yc, Y)

    def evaluate(Y, need_grad=True):
        K = K0 + K1 @ Y
        G = eye + Y.transpose(-1, -2) @ Y
        Lc = torch.linalg.cholesky(G)
        H = K.transpose(-1, -2) @ K
        GinvH = torch.cholesky_solve(H, Lc)
        loss = normB2 - GinvH.diagonal(dim1=-2, dim2=-1).sum(-1)
        if not need_grad:
            return loss, None
        X = K1.T @ K - Y @ GinvH
        grad = -2.0 * torch.cholesky_solve(X.transpose(-1, -2), Lc).transpose(-1, -2)
        return loss, grad

    # ------------------------------------------------------------ initial points (batched)
    inits = [torch.zeros(n, n, device=device, dtype=dtype)]           # col(A) itself
    try:                                                              # clipped principal angles of row(B)
        inits.insert(0, torch.linalg.solve(K0, K1).T)                 # Y_B = K1^T K0^{-T}
    except RuntimeError:
        pass
    if n_random_inits > 0:
        gen = torch.Generator(device=device).manual_seed(seed)
        R = torch.randn(n_random_inits, n, n, device=device, dtype=dtype, generator=gen)
        inits.extend(list(t * R / math.sqrt(n)))
    Y = project(torch.stack(inits))
    r = Y.shape[0]

    # ------------------------------------------------------------ spectral projected gradient
    loss, grad = evaluate(Y)
    gnorm0 = grad.flatten(1).norm(dim=-1).clamp(min=1e-300)
    eta = (t + 1e-3) / gnorm0                    # first trial step of size ~ tan(alpha)
    eps_l = 10 * torch.finfo(dtype).eps          # tolerance for round-off in the Armijo test
    hist = loss[None].repeat(nonmonotone, 1)     # loss history for GLL nonmonotone test
    active = torch.ones(r, dtype=torch.bool, device=device)
    res0, it_done, history = None, max_iter, [loss.clone()]
    for it in range(max_iter):
        ref = hist.max(0).values                 # nonmonotone reference value
        accepted = ~active
        Y_new, loss_new = Y.clone(), loss.clone()
        for _ in range(60):
            Yt = project(Y - eta[:, None, None] * grad)
            Lt, _ = evaluate(Yt, need_grad=False)
            dec = (grad * (Y - Yt)).flatten(1).sum(-1)
            ok = (Lt <= ref - 1e-4 * dec + eps_l * ref.abs()) | (eta < 1e-30)
            newly = ok & ~accepted
            Y_new[newly], loss_new[newly] = Yt[newly], Lt[newly]
            accepted |= ok
            if bool(accepted.all()):
                break
            eta = torch.where(accepted, eta, 0.5 * eta)

        res = (Y_new - Y).flatten(1).norm(dim=-1) / eta          # gradient-mapping norm
        if res0 is None:
            res0 = res.clamp(min=1e-300)
        active &= res > tol * res0
        if len(history) > 20:
            recent = history[-20] - loss_new
            total = history[0] - loss_new
            active &= recent > ftol * total
        loss_chk, grad_new = evaluate(Y_new)
        # Barzilai-Borwein step for the next iteration
        s = (Y_new - Y).flatten(1); yv = (grad_new - grad).flatten(1)
        sy, ss = (s * yv).sum(-1), (s * s).sum(-1)
        eta = torch.where(sy > 0, ss / sy.clamp(min=1e-300), 2.0 * eta).clamp(1e-12, 1e12)
        Y, loss, grad = Y_new, loss_chk, grad_new
        hist = torch.cat([hist[1:], loss[None]])
        history.append(loss.clone())
        if verbose and it % 20 == 0:
            print(f"it {it:4d}  loss {[f'{v:.6g}' for v in loss.tolist()]}  "
                  f"rel-res {[f'{v:.1e}' for v in (res / res0).tolist()]}")
        if not bool(active.any()):
            it_done = it + 1
            break

    # ------------------------------------------------------------ assemble the best solution
    best = int(torch.argmin(loss))
    Yb = Y[best]
    K = K0 + K1 @ Yb
    G = eye + Yb.T @ Yb
    W = Q + UN @ Yb                                          # m x n
    Bp = (torch.linalg.solve(G, K.T).T @ W.T) * scale        # K G^{-1} W^T, un-normalised
    Bp = Bp.to(out_dtype)
    if return_info:
        sig = torch.linalg.svdvals(Yb.double())
        info = {
            "loss": (loss[best] * scale**2).item(),                  # ||B' - B||_F^2
            "losses_per_init": (loss * scale**2).tolist(),
            "iterations": it_done,
            "loss_history": (torch.stack(history) * scale**2).cpu(),
            "angles_deg": torch.rad2deg(torch.atan(sig)).flip(0),    # row(B') vs col(A) (= nonzero kernel angles)
            "max_angle_deg": math.degrees(math.atan(sig.max().item())),
        }
        return Bp, info
    return Bp

if __name__ == "__main__":
    import torch
    from analysis import find_kway_collisions
    from logger import Logger
    A = torch.randn(16, 4).cuda()
    b = torch.randn(16).cuda()
    B = torch.randn(4, 16).cuda()
    
    logger = Logger("projection_test_smart", "./logs/projection.log")
    
    gammas = [0.01, 0.05, 0.1, 0.2]
    alphas = [1]
    betas = [0.0, 0.11, 0.33, 1.0, 3.0, 9.0, 100.0] 
    
    #logger.log(f"Base: {find_kway_collisions(A, b, B, pairs=True, cache=False)}")
    
    for gamma in gammas:
        B_safe = nearest_with_kernel_angle_bound_fast(
            A,
            B,
            alpha=gamma,
            max_iter=20,
            verbose=True
        )
        B_safe= B_safe.cuda()
        collisions = find_kway_collisions(A, b, B_safe, pairs=True, cache=False, alpha_check=0.01, verbose=False)
        logger.log(f"g={gamma}: {collisions} (||B-B'||_F={torch.norm(B-B_safe, p="fro"):.4f}, ||B||_F={torch.norm(B,p="fro"):.4f})")

