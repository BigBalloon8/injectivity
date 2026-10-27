import torch
from tqdm import tqdm
from itertools import combinations, product, permutations
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import null_space
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from functools import partial
import os
from collections import defaultdict
from logger import Logger
from functools import partial
import random
from math import comb

#from cuopt_linprog import cuopt_batch_linprog

# ----------------------------------------------------------------------
# 1. Enumerate the realized activation patterns (linear regions).
#    Reuses the vertex-enumeration idea: each region <-> a sign vector of
#    (A_i x + b_i). Returns an (R, m) int8 tensor of +1/-1 sign vectors.
# ----------------------------------------------------------------------
@torch.compile
def region_patterns(A, b, device, chunk=4096, tol=1e-9):
    A = A.to(device, torch.float64); b = b.to(device, torch.float64)
    m, n = A.shape
    subs = torch.tensor(list(combinations(range(m), n)), device=device, dtype=torch.long)
    two_n = 2**n
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
@torch.compile
def region_maps(A, b, B, patterns):
    A = A.double(); b = b.double(); B = B.double()
    act = (patterns > 0).double()                        # (R, m)  activation masks
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
    #Sr = np.diag(sig_r); Sp = np.diag(sig_p)
    # inequalities (region membership)
    A_ub = np.block([[-np.expand_dims(sig_r,1) * A, np.zeros((m, n))],
                     [np.zeros((m, n)), -np.expand_dims(sig_p,1) * A]])
    b_ub = np.concatenate([sig_r * b, sig_p * b])
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
    bnds = [(None, None)] * (2*n)
    # LP1: maximize g.z  ->  t_max = max + k
    r1 = linprog(-g, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
    if r1.success and (-r1.fun + k) > eps:
        return True, r1.x
    # LP2: minimize g.z  ->  t_min = min + k
    r2 = linprog(g, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
    if r2.success and (r2.fun + k) < -eps:
        return True, r2.x
    return False, None


@partial(torch.vmap, in_dims=(None, None, 0, 0, 0, 0, 0, 0, None, None))
def batch_inputs(A, b, sig_r, sig_p, M_r, M_p, c_r, c_p, w, cap):
    A_ub = torch.block_diag(-sig_r.unsqueeze(1)*A, -sig_p.unsqueeze(1)*A)
    b_ub = torch.concat([sig_r * b, sig_p * b])
    
    A_eq = torch.hstack([M_r, -M_p])
    b_eq = c_p - c_r
    Dr = (sig_r > 0).to(float); Dp = (sig_p > 0).to(float)
    g = torch.concatenate([w * Dr @ A, -(w * Dp) @ A])      # length 2n
    k = w @ (Dr * b) - w @ (Dp * b)
    
    A_ub2 = torch.vstack([A_ub, g, -g])
    b_ub2 = torch.concatenate([b_ub, cap]) 
    return g, A_ub2, b_ub2, A_eq, b_eq, k


def distinct_colapse_process(indexs, An, bn, Bn, P, Mn, cn, w, verbose=True):
    i,j = indexs
    m, n = An.shape
    hit, z = distinct_collapse(An, bn, Bn, P[i], P[j], Mn[i], Mn[j], cn[i], cn[j], w)
    if hit:
        if verbose:
            print(f"x: {z}")
            print(f"  collapse between regions {i} and {j}")
            print("ReLU(Ax + b)")
            print((np.maximum(An@z[:n] + bn, 0)), "\n",(np.maximum(An@z[n:] + bn, 0)))
            print("sign")
            print(np.sign(np.maximum(An@z[:n] + bn, 0)), "\n", np.sign(np.maximum(An@z[n:] + bn, 0)))
            print("B(ReLU(Ax + b))")
            print(Bn@(np.maximum(An@z[:n] + bn, 0)), "\n",Bn@(np.maximum(An@z[n:] + bn, 0)))
            print("")
        return (i,j,z)




# ----------------------------------------------------------------------
# k-way distinct collapse:
#   exists x_1..x_k, x_i in region r_i, with
#     B f(x_1) = ... = B f(x_k)          (all images equal)
#     y_i = D_i (A x_i + b) pairwise distinct
#
# Variables z = [x_1, ..., x_k, t]  (k*n + 1).
# Distinctness via one random w: order the projections w.y_i with gap >= t,
# enumerate the k! orderings, succeed if any ordering reaches t > eps.
# ----------------------------------------------------------------------
def kway_collapse(A, b, sigs, Ms, cs, w, eps=1e-6, cap=10.0):
    m, n = A.shape
    k = len(sigs)
    nv = k * n + 1                       # variables: k points + margin t

    # --- region membership inequalities: -S_i A x_i <= S_i b ---
    A_ub_rows, b_ub_rows = [], []
    for i, sig in enumerate(sigs):
        S = np.diag(sig)
        row = np.zeros((m, nv))
        row[:, i * n:(i + 1) * n] = -S @ A
        A_ub_rows.append(row)
        b_ub_rows.append(S @ b)

    # --- equal-image equalities: M_1 x_1 - M_i x_i = c_i - c_1 ---
    A_eq_rows, b_eq_rows = [], []
    for i in range(1, k):
        row = np.zeros((n, nv))
        row[:, 0:n] = Ms[0]
        row[:, i * n:(i + 1) * n] = -Ms[i]
        A_eq_rows.append(row)
        b_eq_rows.append(cs[i] - cs[0])
    A_eq = np.vstack(A_eq_rows)
    b_eq = np.concatenate(b_eq_rows)

    # --- projections  w . y_i = g_i . x_i + k_i ---
    gs, ks = [], []
    for sig in sigs:
        d = (sig > 0).astype(float)
        gs.append((w * d) @ A)           # length n
        ks.append(float(w @ (d * b)))

    base_A_ub = np.vstack(A_ub_rows)
    base_b_ub = np.concatenate(b_ub_rows)
    bounds = [(None, None)] * (k * n) + [(None, cap)]
    c_obj = np.zeros(nv); c_obj[-1] = -1.0          # maximize t

    # --- try each ordering of the projections ---
    for perm in permutations(range(k)):
        rows, rhs = [], []
        for j in range(k - 1):
            a, bnext = perm[j], perm[j + 1]
            # w.y_a - w.y_b + t <= 0   =>  g_a x_a - g_b x_b + t <= k_b - k_a
            row = np.zeros(nv)
            row[a * n:(a + 1) * n] = gs[a]
            row[bnext * n:(bnext + 1) * n] -= gs[bnext]
            row[-1] = 1.0
            rows.append(row)
            rhs.append(ks[bnext] - ks[a])
        A_ub = np.vstack([base_A_ub] + rows)
        b_ub = np.concatenate([base_b_ub, np.array(rhs)])
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.success and (-res.fun) > eps:
            xs = [res.x[i * n:(i + 1) * n] for i in range(k)]
            return True, xs
    return False, None

def kway_collapse_process(tup, An, bn, P, Mn, cn, w):
    ok, xs = kway_collapse(An, bn,
                               [P[t] for t in tup],
                               [Mn[t] for t in tup],
                               [cn[t] for t in tup], w)
    if ok:
        return tup
    

def find_supersets(tuples):
    if not tuples:
        return []

    k = len(next(iter(tuples)))              # k = n-1
    present = {tuple(sorted(t)) for t in tuples}   # canonicalize + O(1) lookup

    # group by prefix = first k-1 elements
    groups = defaultdict(list)
    for t in present:
        groups[t[:-1]].append(t[-1])

    results = []
    for prefix, lasts in groups.items():
        lasts.sort()
        for i in range(len(lasts)):
            for j in range(i + 1, len(lasts)):
                cand = prefix + (lasts[i], lasts[j])   # stays sorted
                # all n subsets of size k must be present
                if all(sub in present for sub in combinations(cand, k)):
                    results.append(cand)
    return results

def find_n_pairs_collapsing(A,b,B, patterns=None, logger=None, alpha_check=0.1, device=None, verbose=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    A = A.to(device); b = b.to(device); B = B.to(device)
    m, n = A.shape

    if patterns is None:
        patterns = region_patterns(A, b, device)
    
    R = patterns.shape[0]
    M, c, DA, within_ok = region_maps(A, b, B, patterns)
    P = patterns.cpu().numpy().astype(float)
    Mn, cn = M.cpu().numpy(), c.cpu().numpy()
    An, bn, Bn = A.cpu().numpy(), b.cpu().numpy(), B.cpu().numpy()

    rng = np.random.default_rng(0)
    w = rng.standard_normal(m)

    # 1. pairwise collision graph (prune)
    hits = process_map(partial(distinct_colapse_process, An=An, bn=bn, Bn=Bn, P=P, Mn=Mn, cn=cn, w=w, verbose=False), 
                    random.sample(list(combinations(range(R), 2)), k=int(alpha_check*comb(R, 2))), 
                    max_workers=8, 
                    chunksize=16*32,
                    disable = not verbose
            )
    hits = [(h[0],h[1]) for h in hits if h is not None]

    return len(hits), R


# ======================================================================
# Batched LP solver in PyTorch (dense two-phase primal simplex)
# ----------------------------------------------------------------------
# Solves a batch of independent, same-shaped LPs
#       min  c.z   s.t.  A_ub z <= b_ub,   A_eq z = b_eq,   z free
# with tensor ops only, so tens of thousands of tiny LPs run at once.
#
# Standard form: z = u - v (u, v >= 0), one slack per inequality row and
# one artificial per row.  Rows with negative rhs are flipped so the
# starting basis (slack where possible, else artificial) is feasible.
#   Phase 1: min sum(artificials)   ->  feasible iff the optimum is ~0
#   Phase 2: min c.z from the phase-1 basis.  Artificial columns never
#            (re-)enter; any still basic at level 0 are pivoted out first.
# Pricing: Dantzig (most negative reduced cost) for the first
# `bland_after` pivots, then Bland's rule, which cannot cycle.
# Finished LPs are written back and the working set is compacted, so the
# slow tail of the batch does not pay for the whole batch.
#
# Tableau per LP: rows 0..Mr-1 = constraints, row Mr = objective
# (reduced costs; its rhs entry holds -objective).
# Columns: [u (N) | v (N) | slacks (m1) | artificials (Mr) | rhs].
# ======================================================================
LP_OPTIMAL, LP_ITER_LIMIT, LP_INFEASIBLE, LP_UNBOUNDED = 0, 1, 2, 3   # same codes as scipy


def _lp_tableau(A_ub, b_ub, A_eq, b_eq):
    Bt, m1, N = A_ub.shape
    Mr = m1 + A_eq.shape[1]
    art0 = 2 * N + m1                                    # first artificial column
    C = art0 + Mr
    dev, dt = A_ub.device, A_ub.dtype
    A = torch.cat([A_ub, A_eq], 1)                       # (Bt, Mr, N)
    rhs = torch.cat([b_ub, b_eq], 1)                     # (Bt, Mr)
    sgn = 1.0 - 2.0 * (rhs < 0).to(dt)                   # flip rows with rhs < 0
    As = A * sgn.unsqueeze(-1)
    T = torch.zeros(Bt, Mr + 1, C + 1, device=dev, dtype=dt)
    T[:, :Mr, :N] = As
    T[:, :Mr, N:2 * N] = -As
    i1 = torch.arange(m1, device=dev)
    iM = torch.arange(Mr, device=dev)
    T[:, i1, 2 * N + i1] = sgn[:, :m1]                   # slacks
    T[:, iM, art0 + iM] = 1.0                            # artificials
    T[:, :Mr, C] = rhs * sgn                             # rhs >= 0
    basis = (art0 + iM).expand(Bt, Mr).clone()
    basis[:, :m1] = torch.where(sgn[:, :m1] > 0, 2 * N + i1, basis[:, :m1])
    return T, basis, art0


def _lp_set_objective(T, basis, cost):
    """Write the reduced costs of `cost` (B, C) for the current basis into the objective row."""
    Mr = basis.shape[1]
    cB = cost.gather(1, basis)                                         # (B, Mr)
    T[:, Mr, :-1] = cost - torch.bmm(cB.unsqueeze(1), T[:, :Mr, :-1]).squeeze(1)
    T[:, Mr, -1] = -(cB * T[:, :Mr, -1]).sum(1)


def _lp_pivot(T, basis, row, col, go):
    """One batched pivot on (row, col). LPs with go == False are left untouched."""
    n, R1, _ = T.shape
    ar = torch.arange(n, device=T.device)
    prow = T[ar, row]                                     # (n, C+1)
    pcol = T[ar, :, col]                                  # (n, R1)
    piv = torch.where(go, prow[ar, col], torch.ones_like(prow[:, 0]))
    prow = prow / piv.unsqueeze(1)
    pcol = pcol * go.unsqueeze(1).to(T.dtype)
    T.baddbmm_(pcol.unsqueeze(2), prow.unsqueeze(1), alpha=-1.0)      # T -= pcol ⊗ prow
    g = ar[go]
    T[g, row[go]] = prow[go]
    basis[g, row[go]] = col[go]
    T[:, :R1 - 1, -1].clamp_(min=0.0)                     # basic values are >= 0; drop round-off


def _lp_simplex(T, basis, active, art0, phase2, max_iter, bland_after, tol):
    """Primal simplex, in place, on the LPs flagged in `active`.
    Returns (status, n_pivots) with status in {LP_OPTIMAL, LP_UNBOUNDED, LP_ITER_LIMIT}."""
    Bt, R1, _ = T.shape
    Mr = R1 - 1
    dev = T.device
    status = torch.full((Bt,), LP_OPTIMAL, dtype=torch.int8, device=dev)
    idx = active.nonzero().squeeze(1)                     # global ids of the working set
    if idx.numel() == 0:
        return status, 0
    Tw, bw = T[idx], basis[idx]
    live = torch.ones(idx.numel(), dtype=torch.bool, device=dev)
    cols = torch.arange(art0, device=dev)
    big = torch.iinfo(torch.long).max
    it = 0
    while True:
        n = idx.numel()
        ar = torch.arange(n, device=dev)
        d = Tw[:, Mr, :art0]                              # artificial columns never enter
        neg = d < -tol
        if it < bland_after:
            col = d.argmin(1)                             # Dantzig
        else:
            col = torch.where(neg, cols, art0).argmin(1)  # Bland: lowest improving index
        a = Tw[ar, :Mr, col]                              # entering column (n, Mr)
        ratio = torch.where(a > tol, Tw[:, :Mr, -1] / a, torch.inf)
        if phase2:  # an artificial still basic (at level 0) is pivoted out before anything else
            ratio = torch.where((bw >= art0) & (a.abs() > tol), -1.0, ratio)
        rmin = ratio.min(1).values
        tie = ratio <= (rmin + tol * (1 + rmin.abs())).unsqueeze(1)
        row = torch.where(tie, bw, big).argmin(1)         # ties -> lowest basic index (Bland)
        improving = live & neg.any(1)
        unb = improving & torch.isinf(rmin)
        go = improving & ~unb
        status[idx[unb]] = LP_UNBOUNDED
        _lp_pivot(Tw, bw, row, col, go)
        it += 1

        fin = live & ~go                                  # optimal or unbounded this round
        if fin.any():
            T[idx[fin]], basis[idx[fin]] = Tw[fin], bw[fin]
            live &= go
        n_live = int(live.sum())
        if n_live == 0:
            return status, it
        if it >= max_iter:
            T[idx[live]], basis[idx[live]] = Tw[live], bw[live]
            status[idx[live]] = LP_ITER_LIMIT
            return status, it
        if n_live < 0.5 * n:                              # compact the working set
            Tw, bw, idx = Tw[live], bw[live], idx[live]
            live = torch.ones(n_live, dtype=torch.bool, device=dev)


def _lp_iter_defaults(T, max_iter, bland_after):
    size = T.shape[1] + T.shape[2]
    return (50 * size if max_iter is None else max_iter,
            2 * size if bland_after is None else bland_after)


def _lp_phase1(A_ub, b_ub, A_eq, b_eq, max_iter=None, bland_after=None, tol=1e-9, feas_tol=1e-9):
    """Build the tableau and find a feasible basis. Status: LP_OPTIMAL (= feasible),
    LP_INFEASIBLE or LP_ITER_LIMIT."""
    T, basis, art0 = _lp_tableau(A_ub, b_ub, A_eq, b_eq)
    max_iter, bland_after = _lp_iter_defaults(T, max_iter, bland_after)
    scale = 1.0 + T[:, :-1, -1].amax(1)
    cost = torch.zeros(T.shape[0], T.shape[2] - 1, dtype=T.dtype, device=T.device)
    cost[:, art0:] = 1.0
    _lp_set_objective(T, basis, cost)
    active = torch.ones(T.shape[0], dtype=torch.bool, device=T.device)
    st, _ = _lp_simplex(T, basis, active, art0, False, max_iter, bland_after, tol)
    feasible = -T[:, -1, -1] <= feas_tol * scale             # sum of artificials ~ 0
    status = torch.full_like(st, LP_INFEASIBLE)
    status[st == LP_ITER_LIMIT] = LP_ITER_LIMIT
    status[feasible] = LP_OPTIMAL
    # artificials left in the basis are (numerically) at level 0: make that exact
    T[:, :-1, -1] = torch.where(basis >= art0, 0.0, T[:, :-1, -1])
    return T, basis, art0, status


def _lp_phase2(T, basis, art0, c, active, max_iter=None, bland_after=None, tol=1e-9):
    """Minimise c.z (c: (B, N)) from the current feasible basis, for the LPs in `active`."""
    max_iter, bland_after = _lp_iter_defaults(T, max_iter, bland_after)
    N = c.shape[1]
    cost = torch.zeros(T.shape[0], T.shape[2] - 1, dtype=T.dtype, device=T.device)
    cost[:, :N] = c
    cost[:, N:2 * N] = -c
    _lp_set_objective(T, basis, cost)
    st, _ = _lp_simplex(T, basis, active, art0, True, max_iter, bland_after, tol)
    return st


def _lp_solution(T, basis, N):
    x = torch.zeros(T.shape[0], T.shape[2] - 1, dtype=T.dtype, device=T.device)
    x.scatter_(1, basis, T[:, :-1, -1])
    return x[:, :N] - x[:, N:2 * N]


def batched_linprog(c, A_ub, b_ub, A_eq=None, b_eq=None, **kw):
    """Batched  min c.z  s.t.  A_ub z <= b_ub,  A_eq z = b_eq,  z free (no bounds).
    Shapes: c (B,N), A_ub (B,m1,N), b_ub (B,m1), A_eq (B,m2,N), b_eq (B,m2).
    Returns z (B,N), fun (B,), status (B,) with scipy's codes:
    0 optimal, 1 iteration limit, 2 infeasible, 3 unbounded."""
    if A_eq is None:
        A_eq = A_ub.new_zeros(A_ub.shape[0], 0, A_ub.shape[2])
        b_eq = b_ub.new_zeros(b_ub.shape[0], 0)
    p2 = {k: v for k, v in kw.items() if k != "feas_tol"}
    T, basis, art0, status = _lp_phase1(A_ub, b_ub, A_eq, b_eq, **kw)
    st2 = _lp_phase2(T, basis, art0, c, status == LP_OPTIMAL, **p2)
    status = torch.where(status == LP_OPTIMAL, st2, status)
    z = _lp_solution(T, basis, c.shape[1])
    return z, (c * z).sum(1), status


# ----------------------------------------------------------------------
# Batched version of distinct_collapse: one LP pair per region pair.
#   region r  :  -diag(sig_r) A x  <=  diag(sig_r) b
#   region r' :  -diag(sig_r') A x' <= diag(sig_r') b
#   cap       :  |g.z| <= cap
#   collapse  :  M_r x - M_r' x' = c_r' - c_r
#   t = w.(y - y') = g.z + k ;  hit iff t can leave [-eps, eps].
# Phase 1 is shared by the max and min LPs, and if the first feasible
# point already has |t| > eps the pair is decided without any phase 2.
# ----------------------------------------------------------------------
def _collapse_lp_data(A, b, Pt, M, c, I, J, w, cap):
    m, n = A.shape
    P = I.numel()
    sr, sp = Pt[I], Pt[J]                                 # (P, m) in {-1, +1}
    G = A.new_zeros(P, 2 * m + 2, 2 * n)
    G[:, :m, :n] = -sr.unsqueeze(-1) * A
    G[:, m:2 * m, n:] = -sp.unsqueeze(-1) * A
    wr, wp = w * (sr > 0), w * (sp > 0)                   # w ⊙ diag(D_r), w ⊙ diag(D_r')
    g = torch.cat([wr @ A, -(wp @ A)], 1)                 # (P, 2n)
    k = (wr - wp) @ b                                     # (P,)
    G[:, 2 * m] = g
    G[:, 2 * m + 1] = -g
    h = torch.cat([sr * b, sp * b, A.new_full((P, 2), cap)], 1)
    E = torch.cat([M[I], -M[J]], 2)                       # (P, n, 2n)
    e = c[J] - c[I]
    return G, h, E, e, g, k


def distinct_collapse_batch(A, b, Pt, M, c, I, J, w, eps=1e-6, cap=10.0, **lp_kw):
    """Torch version of distinct_collapse for all pairs (I[p], J[p]) at once.
    Returns hit (P,) bool, witness z = [x, x'] (P, 2n) (meaningful where hit),
    and unresolved (P,) bool for LPs that ran into the iteration limit."""
    G, h, E, e, g, k = _collapse_lp_data(A, b, Pt, M, c, I, J, w, cap)
    p2 = {key: v for key, v in lp_kw.items() if key != "feas_tol"}
    T, basis, art0, st = _lp_phase1(G, h, E, e, **lp_kw)
    unresolved = st == LP_ITER_LIMIT
    N = g.shape[1]
    z = _lp_solution(T, basis, N)
    t = (z * g).sum(1) + k
    hit = (st == LP_OPTIMAL) & (t.abs() > eps)            # first feasible point already distinct
    wit = z
    todo = (st == LP_OPTIMAL) & ~hit
    for sense in (-1.0, 1.0):                             # maximise t, then minimise t (warm start)
        if not bool(todo.any()):
            break
        st2 = _lp_phase2(T, basis, art0, sense * g, todo, **p2)
        z = _lp_solution(T, basis, N)
        t = (z * g).sum(1) + k
        new = todo & (-sense * t > eps)
        unresolved |= todo & ~new & (st2 == LP_ITER_LIMIT)
        hit |= new
        wit = torch.where(new.unsqueeze(1), z, wit)
        todo &= ~new
    return hit, wit, unresolved


def _witness_violation(A, b, B, Pt, I, J, z):
    """Max violation of region membership / equal images at the witness, relative to its scale."""
    n = A.shape[1]
    ur = z[:, :n] @ A.T + b
    up = z[:, n:] @ A.T + b
    reg = torch.maximum((-Pt[I] * ur).amax(1), (-Pt[J] * up).amax(1)).clamp(min=0)
    img = ((torch.relu(ur) - torch.relu(up)) @ B.T).abs().amax(1)
    scale = 1.0 + torch.maximum(ur.abs().amax(1), up.abs().amax(1))
    return torch.maximum(reg, img) / scale


def _unrank_pairs(lin, R):
    """Linear index into combinations(range(R), 2) -> (i, j), vectorised."""
    lin = lin.to(torch.int64)
    a = 2 * R - 1
    i = torch.floor((a - torch.sqrt((a * a - 8 * lin).to(torch.float64))) / 2).to(torch.int64)
    S = lambda i: i * (2 * R - i - 1) // 2                # number of pairs before row i
    i = torch.where(S(i) > lin, i - 1, i)
    i = torch.where(S(i + 1) <= lin, i + 1, i)
    return i, lin - S(i) + i + 1


def find_n_pairs_collapsing_torch(A, b, B, patterns=None, logger=None, alpha_check=0.1, device=None,
                                  verbose=True, pairs=None, seed=None, eps=1e-6, cap=10.0,
                                  chunk_bytes=2**29, verify_tol=1e-7, return_hits=False, **lp_kw):
    """Same as find_n_pairs_collapsing, but every pair LP is solved in batched PyTorch.

    pairs       : optional (P, 2) region-index pairs to test instead of a random sample
    seed        : seed for the pair sample (None -> random, like random.sample)
    chunk_bytes : memory budget for one batch of tableaux
    return_hits : also return (I, J, z) for the colliding pairs and their witnesses
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float64
    A = A.to(device, dt); b = b.to(device, dt); B = B.to(device, dt)
    m, n = A.shape

    if patterns is None:
        patterns = region_patterns(A, b, device)
    patterns = patterns.to(device)
    R = patterns.shape[0]
    M, c, DA, within_ok = region_maps(A, b, B, patterns)
    M, c = M.to(dt), c.to(dt)
    Pt = patterns.to(dt)
    w = torch.as_tensor(np.random.default_rng(0).standard_normal(m), dtype=dt, device=device)  # same w as before

    # 1. which pairs: alpha_check of all C(R, 2), sampled without replacement
    if pairs is None:
        n_all = R * (R - 1) // 2
        n_take = int(alpha_check * n_all)
        gen = torch.Generator()
        gen.manual_seed(seed) if seed is not None else gen.seed()
        lin = torch.randperm(n_all, generator=gen)[:n_take].sort().values if n_take < n_all \
            else torch.arange(n_all)
        I, J = _unrank_pairs(lin, R)
    else:
        pairs = torch.as_tensor(pairs, dtype=torch.long)
        I, J = pairs[:, 0], pairs[:, 1]
    I, J = I.to(device), J.to(device)
    P = I.numel()

    # 2. batched LPs, chunked so a batch of tableaux fits in chunk_bytes
    rows = 2 * m + 2 + n + 1
    cols = 4 * n + 2 * m + 2 + (rows - 1) + 1
    chunk = max(1, chunk_bytes // (3 * rows * cols * 8))  # tableau + working copy + temporaries
    hit_I, hit_J, hit_z = [], [], []
    n_unres = n_bad = 0
    for s in tqdm(range(0, P, chunk), disable=not verbose, desc="pair LPs"):
        Ic, Jc = I[s:s + chunk], J[s:s + chunk]
        hit, z, unres = distinct_collapse_batch(A, b, Pt, M, c, Ic, Jc, w, eps=eps, cap=cap, **lp_kw)
        ok = _witness_violation(A, b, B, Pt, Ic, Jc, z) <= verify_tol
        n_bad += int((hit & ~ok).sum())
        n_unres += int(unres.sum())
        hit &= ok
        hit_I.append(Ic[hit]); hit_J.append(Jc[hit]); hit_z.append(z[hit])
    hit_I, hit_J, hit_z = torch.cat(hit_I), torch.cat(hit_J), torch.cat(hit_z)
    n_hits = hit_I.numel()

    msg = f"regions={R}, colliding pairs={n_hits} of {P} checked"
    if n_unres or n_bad:
        msg += f" (iteration limit: {n_unres}, rejected witnesses: {n_bad})"
    if verbose:
        print(msg)
    if logger is not None:
        logger.log(msg)
    if return_hits:
        return n_hits, R, (hit_I.cpu(), hit_J.cpu(), hit_z.cpu())
    return n_hits, R


        

def find_kway_collisions(A, b, B, logger=None, alpha_check=0.1, device=None, verbose=True, cache=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    file_loaded = False
    if os.path.exists("cache.pt") and cache:
        save_data = torch.load("cache.pt", weights_only=False, map_location=torch.device(device))
        if torch.all(A.cpu()==save_data["A"].cpu()) and torch.all(B.cpu() == save_data["B"].cpu()) and  torch.all(b.cpu()==save_data["b"].cpu()):
            A = A.to(device); b = b.to(device); B = B.to(device)
            patterns = save_data["patterns"]
            hits = save_data["hits"]
            M = save_data["M"]
            c = save_data["c"]
            w = save_data["w"]
            Mn, cn = M.cpu().numpy(), c.cpu().numpy()
            An, bn, Bn = A.cpu().numpy(), b.cpu().numpy(), B.cpu().numpy()
            R = patterns.shape[0]
            P = patterns.cpu().numpy().astype(float)
            adj = np.zeros((R, R), dtype=bool)
            file_loaded=True

    if not file_loaded or not cache:
        A = A.to(device); b = b.to(device); B = B.to(device)
        m, n = A.shape

        patterns = region_patterns(A, b, device)
        R = patterns.shape[0]
        M, c, DA, within_ok = region_maps(A, b, B, patterns)
        P = patterns.cpu().numpy().astype(float)
        Mn, cn = M.cpu().numpy(), c.cpu().numpy()
        An, bn, Bn = A.cpu().numpy(), b.cpu().numpy(), B.cpu().numpy()

        rng = np.random.default_rng(0)
        w = rng.standard_normal(m)

        # 1. pairwise collision graph (prune)
        adj = np.zeros((R, R), dtype=bool)
        hits = process_map(partial(distinct_colapse_process, An=An, bn=bn, Bn=Bn, P=P, Mn=Mn, cn=cn, w=w, verbose=False), 
                        random.sample(list(combinations(range(R), 2)), k=int(alpha_check*comb(R, 2))), 
                        max_workers=8, 
                        chunksize=16*32,
                        disable = not verbose
        )

        save_data = {
            "A": A, "b": b, "B": B,
            "patterns": patterns,
            "hits": hits,
            "M": M,
            "c":c,
            "w": w
        }
        torch.save(save_data, "cache.pt")

    for hit in hits:
        if hit is not None:
            i, j, _ = hit
            adj[i, j] = adj[j, i] = True

    n_edges = int(adj.sum() // 2)
    if verbose:
        print(f"regions={R}, colliding pairs={n_edges} of {R*(R-1)//2}")
    if logger is not None:
        logger.log(f"regions={R}, colliding pairs={n_edges} of {R*(R-1)//2}")

    # 2. k-cliques of the graph -> joint LP
    hits = [(h[0],h[1]) for h in hits if h is not None]
    if logger is not None:
        logger.log(f"{len(hits[0])}: {len(hits)}")
    while len(hits) > 0:
        tuples = find_supersets(hits)
        hits = process_map(partial(kway_collapse_process, An=An, bn=bn, P=P, Mn=Mn, cn=cn, w=w),
                        tuples,
                        max_workers=8, 
                        chunksize=12*16
                )
        hits = [h for h in hits if h is not None]
        print(f"{len(hits[0])}: {len(hits)}")
        if logger is not None:
            logger.log(f"{len(hits[0])}: {len(hits)}")
        break

    # for tup in combinations(range(R), k):
    #     if not all(adj[a, bb] for a, bb in combinations(tup, 2)):
    #         continue
    #     ok, xs = kway_collapse(An, bn,
    #                            [P[t] for t in tup],
    #                            [Mn[t] for t in tup],
    #                            [cn[t] for t in tup], w)
    #     if ok:
    #         hits.append((tup, xs))
    #         if verbose:
    #             print(f"  {k}-way collision on regions {tup}")
    #         if len(hits) >= max_hits:
    #             break
    # return hits

def verify(A, b, B, xs, tol=1e-8):
    """Check a witness: all B-images equal, all ReLU outputs pairwise distinct."""
    ys = [torch.relu(A @ torch.as_tensor(x) + b) for x in xs]
    outs = [B @ y for y in ys]
    img_spread = max((outs[0] - o).norm().item() for o in outs[1:])
    min_ydist = min((ys[i] - ys[j]).norm().item()
                    for i, j in combinations(range(len(ys)), 2))
    return img_spread, min_ydist


def is_injective(A, b, B, device=None, verbose=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    A = A.to(float); b = b.to(float); B = B.to(float)
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
    
    pairs = torch.tensor(list(combinations(range(R), 2)))
    print(pairs.shape)
    chunk_size = 1024*8
    wt = torch.tensor(w).to(device)
    cap = torch.tensor([10, 10]).to(float).to(device)
    for i in tqdm(range(0, pairs.shape[0], chunk_size)):
        rs = pairs[i:i + chunk_size, 0]
        ps = pairs[i:i + chunk_size, 1]
        g, A_ub2, b_ub2, A_eq, b_eq, k = batch_inputs(A, b, patterns[rs], patterns[ps], M[rs], M[ps], c[rs], c[ps], wt, cap)
        cuopt_batch_linprog(g, A_ub2, b_ub2, A_eq, b_eq, method="dual_simplex", mode="stacked")
        

    hits = process_map(partial(distinct_colapse_process, An=An, bn=bn, Bn=Bn, P=P, Mn=Mn, cn=cn, w=w), 
                       list(combinations(range(R), 2)), 
                       max_workers=16, 
                       chunksize=1024
    )
    hits = list([hit for hit in hits if hit is not None]) 
    print(len(hits))

def matrix_from_kernel(K, tol=1e-10):
    # Check K has full column rank
    if np.linalg.matrix_rank(K, tol) != K.shape[1]:
        raise ValueError("Columns of K must be linearly independent.")

    # Left null space of K^T = orthogonal complement of ker(B)
    B = null_space(K.T).T
    return B


def main():
    seed=4
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        map_loc = torch.device("cpu")
    else:
        map_loc = None
    # m, n = 6, 2
    # A = torch.randn(m, n, dtype=torch.float64)
    # b = torch.randn(m, dtype=torch.float64)
    # B = torch.randn(n, m, dtype=torch.float64)

    # for k in (3, 4):
    #     print(f"\n=== k = {k} (m={m}, n={n}) ===")
    #     hits = find_kway_collisions(A, b, B, k=k, max_hits=2)
    #     if not hits:
    #         print("no k-way collisions")
    #     for tup, xs in hits:
    #         spread, dmin = verify(A, b, B, xs)
    #         print(f"  witness on {tup}: max image spread = {spread:.2e}, "
    #               f"min pairwise ||y_i - y_j|| = {dmin:.3f}")
    random = True
    if random:
        up_proj = torch.randn(16, 4)
        up_proj_b = torch.randn(16)
        down_proj = torch.randn(4,16)
        log_file = f"./logs/analysis_random.log"
        logger = Logger(f"Analysis({seed})", log_file)
    else:
        model_dir = "./models/"
        model_file = "modular_{'p': 29, 'op': 'add'}_4_16.pt"
        filename = model_dir + model_file
        log_file = f"./logs/analysis_{model_file}.log"
        logger = Logger("Analysis", log_file)
        state_dict = torch.load(filename, map_location=map_loc)
        if any(["layers.1" in k for k in state_dict.keys()]):
            max_layer = max([i for i in range(100) if any([f"layers.{i}" in k for k in state_dict.keys()])])
            idx = int(input(f"Select Layer (0-{max_layer}): "))
        else:
            idx = 0
        
        try: # compiled
            up_proj = state_dict[f"_orig_mod.layers.{idx}.ffn.l1.weight"]
            up_proj_b = state_dict[f"_orig_mod.layers.{idx}.ffn.l1.bias"]
            down_proj = state_dict[f"_orig_mod.layers.{idx}.ffn.l2.weight"]
        except KeyError: # not compiled
            up_proj = state_dict[f"layers.{idx}.ffn.l1.weight"]
            up_proj_b = state_dict[f"layers.{idx}.ffn.l1.bias"]
            down_proj = state_dict[f"layers.{idx}.ffn.l2.weight"]
    
    down_proj = 2*up_proj.T 

    # is_injective(up_proj, up_proj_b, down_proj, device="cuda")

    find_kway_collisions(up_proj, up_proj_b, down_proj, logger)
    #inj, cert = is_injective(up_proj, up_proj_b, down_proj, device="cuda")
    # print(f"injective on ReLU range: {inj}")
    # if not inj:
    #     print(f"  certificate: {cert}")

if __name__ == "__main__":
    main()