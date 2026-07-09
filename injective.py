import torch
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
from scipy.linalg import null_space

# ----------------------------------------------------------------------
# 1. Enumerate the realized activation patterns (linear regions).
#    Reuses the vertex-enumeration idea: each region <-> a sign vector of
#    (A_i x + b_i). Returns an (R, m) int8 tensor of +1/-1 sign vectors.
# ----------------------------------------------------------------------
def region_patterns(A, b, device, chunk=4096, tol=1e-9):
    A = A.to(device, torch.float64); b = b.to(device, torch.float64)
    m, n = A.shape
    subs = torch.tensor(list(combinations(range(m), n)), device=device, dtype=torch.long)
    two_n = 1 << n
    gb = (torch.arange(two_n, device=device).unsqueeze(1) >> torch.arange(n, device=device)) & 1
    grid = (gb * 2 - 1).to(torch.int8)
    seen = None
    for s0 in range(0, subs.shape[0], chunk):
        idx = subs[s0:s0 + chunk]
        As, bs = A[idx], b[idx]
        keep = torch.linalg.det(As).abs() > tol
        idx, As, bs = idx[keep], As[keep], bs[keep]
        if idx.shape[0] == 0:
            continue
        v = torch.linalg.solve(As, -bs.unsqueeze(-1)).squeeze(-1)
        S = torch.sign(v @ A.T + b).to(torch.int8)
        c = idx.shape[0]
        cell = S.unsqueeze(1).repeat(1, two_n, 1).reshape(c * two_n, m)
        rows = torch.arange(c, device=device).repeat_interleave(two_n)
        cols = idx[rows]
        vals = grid.unsqueeze(0).expand(c, -1, -1).reshape(c * two_n, n)
        cell.scatter_(1, cols, vals)
        cell = torch.unique(cell, dim=0)
        seen = cell if seen is None else torch.unique(torch.cat([seen, cell]), dim=0)
    return seen  # (R, m) in {-1,+1}


# ----------------------------------------------------------------------
# 2. Per-region affine pieces  M_r = B D_r A,  c_r = B D_r b   (batched).
#    D_r = diag(pattern > 0).  Also the within-region rank test.
# ----------------------------------------------------------------------
def region_maps(A, b, B, patterns):
    A = A.double(); b = b.double(); B = B.double()
    act = (patterns > 0).double()                       # (R, m)  activation masks
    DA = act.unsqueeze(-1) * A                           # (R, m, n)  = D_r A
    M = torch.einsum('ij,rjk->rik', B, DA)               # (R, n, n)  = B D_r A
    c = (act * b) @ B.T                                  # (R, n)     = B D_r b
    # within-region injective iff rank(B D_r A) == rank(D_r A)
    rank_DA = torch.linalg.matrix_rank(DA)
    rank_M  = torch.linalg.matrix_rank(M)
    within_ok = (rank_M == rank_DA)                      # (R,) bool
    return M, c, DA, within_ok


# ----------------------------------------------------------------------
# 3. Between-region LP: does a DISTINCT collapse exist for a pair (r, r')?
#    vars z = [x, x'] in R^{2n}.
#      region r  :  -diag(sig_r) A x  <=  diag(sig_r) b
#      region r' :  -diag(sig_r') A x' <= diag(sig_r') b
#      collapse  :  M_r x - M_r' x' = c_r' - c_r
#    distinctness: maximize +/- w.(y - y'), y=D_r(Ax+b), y'=D_r'(Ax'+b).
#    A genuine collapse exists iff the objective can leave 0.
# ----------------------------------------------------------------------
def distinct_collapse(A, b, B, sig_r, sig_p, M_r, M_p, c_r, c_p, w, eps=1e-6, cap=10.0):
    m, n = A.shape
    Sr = np.diag(sig_r); Sp = np.diag(sig_p)
    # inequalities (region membership)
    A_ub = np.block([[-Sr @ A, np.zeros((m, n))],
                     [np.zeros((m, n)), -Sp @ A]])
    b_ub = np.concatenate([Sr @ b, Sp @ b])
    # equalities (collapse)
    A_eq = np.hstack([M_r, -M_p])
    b_eq = c_p - c_r
    # objective g.z + k  where g,k come from w.(y - y')
    Dr = (sig_r > 0).astype(float); Dp = (sig_p > 0).astype(float)
    g = np.concatenate([w * Dr @ A, -(w * Dp) @ A])      # length 2n
    k = float(w @ (Dr * b) - w @ (Dp * b))
    # box the objective so the LP stays bounded:  |g.z| <= cap
    A_ub2 = np.vstack([A_ub, g, -g])
    b_ub2 = np.concatenate([b_ub, [cap, cap]])
    bnds = [(None, None)] * (2 * n)
    # LP1: maximize g.z  ->  t_max = max + k
    r1 = linprog(-g, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
    if r1.success and (-r1.fun + k) > eps:
        return True, r1.x
    # LP2: minimize g.z  ->  t_min = min + k
    r2 = linprog(g, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
    if r2.success and (r2.fun + k) < -eps:
        return True, r2.x
    return False, None


def is_injective(A, b, B, device=None, verbose=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    A = A.to(device); b = b.to(device); B = B.to(device)
    m, n = A.shape

    patterns = region_patterns(A, b, device)
    R = patterns.shape[0]
    M, c, DA, within_ok = region_maps(A, b, B, patterns)
    if verbose:
        print(f"regions={R}, within-region violations={int((~within_ok).sum())}")

    # move small things to numpy for the LP loop
    P  = patterns.cpu().numpy().astype(float)             # (R, m) sign vectors
    Mn = M.cpu().numpy(); cn = c.cpu().numpy()
    An = A.cpu().numpy(); bn = b.cpu().numpy(); Bn = B.cpu().numpy()

    if not bool(within_ok.all()):
        return False, ("within-region", int((~within_ok).nonzero()[0][0]))

    rng = np.random.default_rng(0)
    w = rng.standard_normal(m)
    for i, j in combinations(range(R), 2):
        hit, z = distinct_collapse(An, bn, Bn, P[i], P[j], Mn[i], Mn[j], cn[i], cn[j], w)
        if hit:
            if verbose:
                print(f"  collapse between regions {i} and {j}")
            return False, ("between", i, j, z)
    return True, None


def matrix_from_kernel(K, tol=1e-10):
    # Check K has full column rank
    if np.linalg.matrix_rank(K, tol) != K.shape[1]:
        raise ValueError("Columns of K must be linearly independent.")

    # Left null space of K^T = orthogonal complement of ker(B)
    B = null_space(K.T).T
    return B



if __name__ == "__main__":
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for (m, n) in [(6, 2), (8, 2), (7, 3)]:
        A = torch.randn(m, n, dtype=torch.float64)
        b = torch.randn(m, dtype=torch.float64)
        B = torch.randn(n, m, dtype=torch.float64)
        B = torch.tensor(matrix_from_kernel(null_space(A.T)))
        print(f"\n=== m={m}, n={n} ===")
        inj, cert = is_injective(A, b, B, device=device)
        print(f"injective on ReLU range: {inj}")
        if not inj:
            print(f"  certificate: {cert[:3]}")