import numpy as np
import time
from tqdm import tqdm

# ---------------- core pieces (same math as the solver) ----------------

def bases(A, tol=1e-12):
    U, s, _ = np.linalg.svd(A, full_matrices=True)
    r = int((s > tol * s[0]).sum())
    return U[:, :r], U[:, r:]

def spec_proj(T, tau):
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    return (U * np.minimum(s, tau)) @ Vt

def f_grad(T, G, H):
    R = G + H @ T
    Psi = np.linalg.inv(np.eye(T.shape[1]) + T.T @ T)
    RP = R @ Psi
    f = float(np.trace(R.T @ RP))
    g = 2.0 * (H.T @ RP - T @ (Psi @ (R.T @ RP)))
    return f, g

def clamped_T(Ahat, Q, B, alpha, tol=1e-10):
    n = Ahat.shape[1]
    _, _, VBt = np.linalg.svd(B, full_matrices=False)
    V = VBt[:n].T
    Uc, c, Zt = np.linalg.svd(Ahat.T @ V)
    c = np.clip(c, 0.0, 1.0)
    theta = np.arccos(c)
    Wq = Q.T @ V @ Zt.T
    s = np.sqrt(np.maximum(0.0, 1.0 - c ** 2))
    Qd = np.zeros_like(Wq)
    for i in range(n):
        if s[i] > tol:
            Qd[:, i] = Wq[:, i] / s[i]
    th = np.minimum(theta, alpha)
    S = Ahat @ (Uc * np.cos(th)) + Q @ (Qd * np.sin(th))
    return (Q.T @ S) @ np.linalg.inv(Ahat.T @ S)

# ------- PG ascent instrumented: returns per-iteration objective trace -------

def pg_trace(T0, G, H, tau, iters=4000, gtol=1e-13):
    """Projected gradient ascent. Stationarity measured by the gradient mapping
    norm ||T - Pi(T + eta grad)|| / eta. Returns (T, f, history)."""
    T = spec_proj(T0, tau)
    f, g = f_grad(T, G, H)
    step = 1.0
    hist = [f]
    for k in range(iters):
        ok = False
        for _ in range(60):
            Tn = spec_proj(T + step * g, tau)
            fn, gn = f_grad(Tn, G, H)
            if fn >= f:
                ok = True
                break
            step *= 0.5
        if not ok:
            break
        gmap = np.linalg.norm(Tn - T) / step
        T, f, g = Tn, fn, gn
        hist.append(f)
        if gmap < gtol:
            break
        step *= 1.5
    return T, f, np.array(hist)

def iters_to(hist, fstar, rtol):
    """First index at which the relative objective gap drops below rtol."""
    gap = (fstar - hist) / max(1.0, abs(fstar))
    idx = np.where(gap <= rtol)[0]
    return int(idx[0]) if len(idx) else None

# ---------------- instance generation ----------------

def make_instance(m, n, cond, rg):
    A = rg.standard_normal((m, n))
    B = rg.standard_normal((n, m))
    if cond > 1:
        U, s, Vt = np.linalg.svd(B, full_matrices=False)
        s = np.logspace(0, -np.log10(cond), n)
        B = (U * s) @ Vt
    return A, B

def reference(G, H, tau, mn, n, rg, n_rand=4):
    """High-accuracy reference optimum via many restarts."""
    best = -np.inf
    for _ in range(n_rand):
        R = rg.standard_normal((mn, n))
        R *= tau * rg.uniform(0, 1) / max(np.linalg.norm(R, 2), 1e-30)
        _, f, _ = pg_trace(R, G, H, tau, iters=4000)
        best = max(best, f)
    return best

# ---------------- experiment ----------------

def study(sizes, alphas, conds, reps=5, seed=0):
    rows = []
    for (m, n) in sizes:
        for alpha in tqdm(alphas):
            for cond in conds:
                rec = dict(m=m, n=n, alpha=alpha, cond=cond,
                           it6=[], it10=[], it6_zero=[], it6_rand=[],
                           restart_wins=0, t=[])
                for r in range(reps):
                    rg = np.random.default_rng(hash((m, n, r, int(alpha*1e3), cond)) % 2**31)
                    A, B = make_instance(m, n, cond, rg)
                    Ah, Q = bases(A)
                    tau = np.tan(alpha)
                    G, H = B @ Ah, B @ Q

                    fstar = reference(G, H, tau, m - n, n, rg)

                    t0 = time.time()
                    Tc = clamped_T(Ah, Q, B, alpha)
                    _, fc, hc = pg_trace(Tc, G, H, tau)
                    rec["t"].append(time.time() - t0)
                    fstar = max(fstar, fc)

                    _, fz, hz = pg_trace(np.zeros((m - n, n)), G, H, tau)
                    R = rg.standard_normal((m - n, n))
                    R *= tau * 0.5 / np.linalg.norm(R, 2)
                    _, fr, hr = pg_trace(R, G, H, tau)

                    rec["it6"].append(iters_to(hc, fstar, 1e-6))
                    rec["it10"].append(iters_to(hc, fstar, 1e-10))
                    rec["it6_zero"].append(iters_to(hz, fstar, 1e-6))
                    rec["it6_rand"].append(iters_to(hr, fstar, 1e-6))
                    if max(fz, fr) > fc + 1e-9 * max(1, abs(fc)):
                        rec["restart_wins"] += 1
                rows.append(rec)
    return rows

def summarize(rows):
    def fmt(v):
        v = [x for x in v if x is not None]
        return f"{int(np.median(v)):4d}/{int(np.max(v)):4d}" if v else "  --"
    print(f"{'m':>4} {'n':>3} {'alpha':>6} {'cond':>6} | "
          f"{'clamp 1e-6':>11} {'clamp 1e-10':>11} {'zero 1e-6':>11} {'rand 1e-6':>11} | "
          f"{'ms':>6} {'restart wins':>12}")
    print("-" * 104)
    for r in rows:
        print(f"{r['m']:>4} {r['n']:>3} {r['alpha']:>6.2f} {r['cond']:>6.0f} | "
              f"{fmt(r['it6']):>11} {fmt(r['it10']):>11} {fmt(r['it6_zero']):>11} "
              f"{fmt(r['it6_rand']):>11} | "
              f"{1000*np.median(r['t']):>6.1f} {r['restart_wins']:>12}")
    print("\n(cells are median/max iterations over reps)")

print("=== iterations to reach relative objective gap, by init ===")
rows = study(sizes=[(20, 5), (50, 10), (100, 20)],
             alphas=[0.05, 0.25, 0.60, 1.20],
             conds=[1, 1000],
             reps=3)
summarize(rows)
