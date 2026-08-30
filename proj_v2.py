"""
Given A (m x n, full column rank), B (n x m) and an angle alpha, find B' minimising
||B' - B||_F such that every principal angle between ker(B') and col(A)^perp is <= alpha.

Equivalently (taking orthogonal complements preserves principal angles): every principal
angle between row(B') and col(A) is <= alpha.  For a fixed row space S the closest matrix
is B P_S, so we optimise over S.  Writing S = col(Q + Q_perp T) with Q, Q_perp orthonormal
bases of col(A) and col(A)^perp, the tangents of the principal angles are the singular
values of T, so the feasible set is the spectral-norm ball ||T||_2 <= tan(alpha).
We run projected gradient descent with backtracking on

    L(T) = ||B (I - P_S)||_F^2,     P_S = W (W^T W)^{-1} W^T,   W = Q + Q_perp T.
"""

import math
import torch


# ----------------------------------------------------------------------------- helpers
def _bases(A):
    """Orthonormal bases Q of col(A) and Qp of col(A)^perp via a full QR."""
    m, n = A.shape
    Qfull, _ = torch.linalg.qr(A, mode="complete")
    return Qfull[:, :n], Qfull[:, n:]


def _clip_spectral(T, t):
    """Projection onto {T : ||T||_2 <= t}: clip singular values."""
    if T.numel() == 0:
        return T
    U, s, Vh = torch.linalg.svd(T, full_matrices=False)
    return (U * s.clamp(max=t)) @ Vh


def _project_B(B, Q, Qp, T):
    """B' = B P_S for S = col(Q + Qp T).  Returns (loss, B', W, G)."""
    n = Q.shape[1]
    W = Q + Qp @ T
    G = torch.eye(n, dtype=B.dtype, device=B.device) + T.T @ T
    BW = B @ W                                    # n x n
    Bp = torch.linalg.solve(G, BW.T).T @ W.T      # B W G^{-1} W^T
    loss = ((B - Bp) ** 2).sum()
    return loss, Bp, W, G


def _grad(B, Q, Qp, T):
    """Analytic gradient  dL/dT = -2 Qp^T (B - B')^T B W G^{-1}."""
    loss, Bp, W, G = _project_B(B, Q, Qp, T)
    M = (B - Bp).T @ B @ W                        # m x n
    grad = -2.0 * Qp.T @ torch.linalg.solve(G, M.T).T
    return grad, loss, Bp


def principal_angles(X, Y):
    """Principal angles (radians, ascending) between col(X) and col(Y)."""
    Qx, _ = torch.linalg.qr(X)
    Qy, _ = torch.linalg.qr(Y)
    s = torch.linalg.svdvals(Qx.T @ Qy)
    return torch.arccos(s.clamp(-1.0, 1.0)).flip(0)


def kernel_angles(A, Bp, rtol=1e-10):
    """Principal angles between ker(B') and col(A)^perp, computed directly (independent
    of the T-parametrisation) so it can be used as a check."""
    m, n = A.shape
    _, Qp = _bases(A)
    # null space of B' via SVD
    U, s, Vh = torch.linalg.svd(Bp, full_matrices=True)
    rank = int((s > rtol * s.max()).sum()) if s.numel() else 0
    K = Vh[rank:].T                               # m x (m - rank)
    return principal_angles(K, Qp)


# ----------------------------------------------------------------------------- main
def nearest_with_kernel_angle_bound(A, B, alpha, iters=2000, restarts=3, tol=1e-12,
                                    seed=0, return_info=False):
    """
    Args
        A      : (m, n) tensor with m > n and full column rank
        B      : (n, m) tensor
        alpha  : max allowed principal angle in radians (0 <= alpha < pi/2)
    Returns
        B'     : (n, m) tensor, and optionally an info dict
    """
    A = A.to(torch.float64)
    B = B.to(torch.float64)
    m, n = A.shape
    Q, Qp = _bases(A)
    t = math.tan(alpha)
    gen = torch.Generator().manual_seed(seed)

    # ---- initial points
    inits = []
    # (a) clip the principal angles of row(B) at alpha
    C = Q.T @ B.T
    if torch.linalg.matrix_rank(C) == n:
        T_B = torch.linalg.solve(C.T, (Qp.T @ B.T).T).T  # (Qp^T B^T) C^{-1}
        inits.append(_clip_spectral(T_B, t))
    # (b) start from col(A) itself
    inits.append(torch.zeros(m - n, n, dtype=A.dtype))
    # (c) random feasible points
    for _ in range(max(0, restarts - len(inits))):
        R = torch.randn(m - n, n, generator=gen, dtype=A.dtype)
        inits.append(_clip_spectral(R, t))

    best = None
    for T in inits:
        eta = 1.0
        g, loss, Bp = _grad(B, Q, Qp, T)
        for _ in range(iters):
            # projected gradient step with backtracking (Armijo)
            while True:
                T_new = _clip_spectral(T - eta * g, t)
                loss_new, Bp_new, _, _ = _project_B(B, Q, Qp, T_new)
                dec = (g * (T - T_new)).sum()
                if loss_new <= loss - 1e-4 * dec or eta < 1e-16:
                    break
                eta *= 0.5
            step = (T_new - T).norm()
            T, loss, Bp = T_new, loss_new, Bp_new
            g, _, _ = _grad(B, Q, Qp, T)
            eta = min(eta * 2.0, 1e6)
            if step < tol * (1.0 + T.norm()):
                break
        if best is None or loss < best[0]:
            best = (loss, Bp, T)

    loss, Bp, T = best
    if return_info:
        return Bp, {"loss": loss.item(), "T": T,
                    "angles": kernel_angles(A, Bp)}
    return Bp


# ----------------------------------------------------------------------------- tests
def _rand_problem(m, n, gen):
    A = torch.randn(m, n, generator=gen, dtype=torch.float64)
    B  = A.T + 0.1*torch.randn(n, m, generator=gen, dtype=torch.float64)
    return A, B


def test_alpha_zero_matches_projection():
    """alpha = 0  =>  B' = B P_A."""
    gen = torch.Generator().manual_seed(1)
    A, B = _rand_problem(7, 3, gen)
    Bp = nearest_with_kernel_angle_bound(A, B, 0.0)
    P_A = A @ torch.linalg.solve(A.T @ A, A.T)
    assert torch.allclose(Bp, B @ P_A, atol=1e-8), (Bp - B @ P_A).abs().max()


def test_feasible_B_is_fixed_point():
    """If B already satisfies the constraint, B' = B."""
    gen = torch.Generator().manual_seed(2)
    A, B = _rand_problem(8, 3, gen)
    theta_max = kernel_angles(A, B).max().item()
    alpha = theta_max + 0.05
    Bp = nearest_with_kernel_angle_bound(A, B, alpha)
    assert torch.allclose(Bp, B, atol=1e-8), (Bp - B).abs().max()


def test_gradient_matches_finite_differences():
    gen = torch.Generator().manual_seed(3)
    A, B = _rand_problem(6, 2, gen)
    Q, Qp = _bases(A)
    T = torch.randn(4, 2, generator=gen, dtype=torch.float64)
    g, _, _ = _grad(B, Q, Qp, T)
    T_ad = T.clone().requires_grad_(True)
    loss, _, _, _ = _project_B(B, Q, Qp, T_ad)
    loss.backward()
    assert torch.allclose(g, T_ad.grad, atol=1e-8), (g - T_ad.grad).abs().max()


def test_n_equals_one_closed_form():
    """n = 1: projection of the row vector b onto a circular cone of half-angle alpha
    around +-a, i.e. rotate b toward a until angle alpha, scale by cos(phi - alpha)."""
    gen = torch.Generator().manual_seed(4)
    A, B = _rand_problem(6, 1, gen)
    a = A[:, 0] / A[:, 0].norm()
    b = B[0]
    alpha = 0.3
    # angle to the line spanned by a
    ca = torch.dot(a, b) / b.norm()
    if ca < 0:
        a = -a
        ca = -ca
    phi = torch.arccos(ca.clamp(-1, 1))
    if phi <= alpha:
        b_star = b
    else:
        b_perp = b - torch.dot(a, b) * a
        u = b_perp / b_perp.norm()
        direction = math.cos(alpha) * a + math.sin(alpha) * u
        b_star = b.norm() * torch.cos(phi - alpha) * direction
    Bp = nearest_with_kernel_angle_bound(A, B, alpha)
    assert torch.allclose(Bp[0], b_star, atol=1e-8), (Bp[0] - b_star).abs().max()


def test_tiny_case_matches_brute_force():
    """m=3, n=2: T is 1x2, so the feasible set is a disc of radius tan(alpha).  Grid it."""
    gen = torch.Generator().manual_seed(5)
    A, B = _rand_problem(3, 2, gen)
    alpha = 0.4
    Q, Qp = _bases(A)
    t = math.tan(alpha)
    Bp, info = nearest_with_kernel_angle_bound(A, B, alpha, return_info=True)
    best = float("inf")
    for r in torch.linspace(0, t, 120):
        for th in torch.linspace(0, 2 * math.pi, 240):
            T = torch.tensor([[r * math.cos(th), r * math.sin(th)]], dtype=torch.float64)
            l, _, _, _ = _project_B(B, Q, Qp, T)
            best = min(best, l.item())
    assert info["loss"] <= best + 1e-6, (info["loss"], best)


def test_general_case_properties():
    gen = torch.Generator().manual_seed(6)
    for (m, n) in [(6, 2), (10, 4), (12, 5)]:
        A, B = _rand_problem(m, n, gen)
        prev = None
        for alpha in [0.1, 0.3, 0.6, 1.0]:
            Bp, info = nearest_with_kernel_angle_bound(A, B, alpha, return_info=True)
            # constraint holds (checked independently of the parametrisation)
            assert info["angles"].max() <= alpha + 1e-7, (info["angles"].max(), alpha)
            # never worse than the trivially feasible choice B P_A
            P_A = A @ torch.linalg.solve(A.T @ A, A.T)
            assert info["loss"] <= ((B - B @ P_A) ** 2).sum() + 1e-9
            # never worse than the "clip the principal angles" heuristic
            Q, Qp = _bases(A)
            T_B = torch.linalg.solve((Q.T @ B.T).T, (Qp.T @ B.T).T).T
            l_clip, _, _, _ = _project_B(B, Q, Qp, _clip_spectral(T_B, math.tan(alpha)))
            assert info["loss"] <= l_clip + 1e-9
            # distance is non-increasing in alpha
            if prev is not None:
                assert info["loss"] <= prev + 1e-9
            prev = info["loss"]
            # B' has full row rank n and its rows lie in a subspace at angle <= alpha
            assert torch.linalg.matrix_rank(Bp) == n


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in tests:
        f()
        print(f"{f.__name__:45s} ok")

    # small demo
    gen = torch.Generator().manual_seed(0)
    A, B = _rand_problem(768*4, 768, gen)
    alpha = math.radians(5)
    print("\ndemo: m=8, n=3, alpha=5 deg")
    print("angles(ker B, col(A)^perp) before [deg]:",
          [round(x, 2) for x in torch.rad2deg(kernel_angles(A, B)).tolist()])
    Bp, info = nearest_with_kernel_angle_bound(A, B, alpha, return_info=True)
    print("angles(ker B', col(A)^perp) after  [deg]:",
          [round(x, 2) for x in torch.rad2deg(info["angles"]).tolist()])
    print(f"||B'-B||_F = {math.sqrt(info['loss']):.4f}   "
          f"(||B P_A - B||_F = {(B - B @ A @ torch.linalg.solve(A.T @ A, A.T)).norm():.4f})")
