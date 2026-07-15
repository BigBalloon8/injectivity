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

from cuopt_linprog import cuopt_batch_linprog

# ----------------------------------------------------------------------
# 1. Enumerate the realized activation patterns (linear regions).
#    Reuses the vertex-enumeration idea: each region <-> a sign vector of
#    (A_i x + b_i). Returns an (R, m) int8 tensor of +1/-1 sign vectors.
# ----------------------------------------------------------------------
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



def kway_collapse_process(tup, An, bn, P, Mn, cn, w):
    ok, xs = kway_collapse(An, bn,
                               [P[t] for t in tup],
                               [Mn[t] for t in tup],
                               [cn[t] for t in tup], w)
    if ok:
        return tup

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



def find_kway_collisions(A, b, B, logger, alpha_check=0.1, device=None, verbose=True, pairs=False, cache=True):
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
                        list(combinations(range(R), 2)), 
                        max_workers=8, 
                        random.sample(list(combinations(range(R), 2)), k=int(alpha_check*comb(R, 2))), 
                        max_workers=24, 
                        chunksize=16*32
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
    
    if pairs: 
        hits = [(h[0],h[1]) for h in hits if h is not None]
        return len(hits)

    for hit in hits:
        if hit is not None:
            i, j, _ = hit
            adj[i, j] = adj[j, i] = True

    n_edges = int(adj.sum() // 2)
    if verbose:
        print(f"regions={R}, colliding pairs={n_edges} of {R*(R-1)//2}")
    logger.log(f"regions={R}, colliding pairs={n_edges} of {R*(R-1)//2}")

    # 2. k-cliques of the graph -> joint LP
    hits = [(h[0],h[1]) for h in hits if h is not None]
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
    
    # down_proj = torch.tensor(matrix_from_kernel(null_space(up_proj.T)))

    # is_injective(up_proj, up_proj_b, down_proj, device="cuda")

    find_kway_collisions(up_proj, up_proj_b, down_proj, logger)
    #inj, cert = is_injective(up_proj, up_proj_b, down_proj, device="cuda")
    # print(f"injective on ReLU range: {inj}")
    # if not inj:
    #     print(f"  certificate: {cert}")

if __name__ == "__main__":
    main()