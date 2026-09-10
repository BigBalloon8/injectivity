"""
Fast version of nearest_with_kernel_angle_bound (batched, GPU-ready).

Reduction used (proof in the accompanying note):  with  N = (I - P_A) B^T  (m x n) and
U_N an orthonormal basis of col(N), the optimal Q_perp T can be written U_N Y with Y n x n.
A component of Q_perp T orthogonal to col(N) is invisible to B (B U = 0 for such U), only
enlarges G = I + T^T T (raising the loss) and only enlarges ||T||_2 (never helping
feasibility), so it is never present at the optimum.  Hence

    min_{||Y||_2 <= tan(alpha)}  L(Y) = ||B||_F^2 - tr( (I + Y^T Y)^{-1} K^T K ),
    K = K0 + K1 Y,   K0 = B Q,   K1 = B U_N   (all n x n),

    grad L(Y) = -2 ( K1^T K - Y G^{-1} K^T K ) G^{-1},      G = I + Y^T Y,

and every iteration touches only n x n matrices.  The m x n objects (Q, U_N) are used once
at setup (two thin QRs) and once at the end to form  B' = K G^{-1} (Q + U_N Y)^T.

Solver: spectral projected gradient (Barzilai-Borwein step + nonmonotone Armijo safeguard).
Projection onto the spectral-norm ball is done via eigh of Y^T Y (batched, no SVD).
All initial points run as one batch.  B is normalised to unit Frobenius norm internally.
"""

import math
import torch


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