"""
cuopt_linprog — scipy.optimize.linprog-style batched LP solving via a custom
PyTorch C++ op wrapping NVIDIA cuOpt (https://github.com/NVIDIA/cuopt).

    from cuopt_linprog import cuopt_batch_linprog

    res = cuopt_batch_linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                              bounds=(0, 1), mode="stacked")
    res.x, res.fun, res.status, res.success   # batched, scipy status codes

Two batching strategies (`mode=`):

* "loop" (default): each LP in the batch is one cuOptSolve call, looped in
  C++ with the GIL released. Per-problem statuses (infeasible/unbounded
  lanes are individually identified by cuOpt's own detection), and each
  problem is solved to full tolerance independently. Per-solve overhead is
  a few ms, so this suits batches of moderately sized LPs.

* "stacked": all B problems are fused into ONE block-diagonal sparse LP
  and solved in a single cuOpt call. Since the problems share no variables
  or constraints, the fused optimum decomposes exactly into the per-problem
  optima. This gives PDLP/barrier one large sparse problem — the regime
  GPU LP solvers are built for — and amortizes all launch overhead, making
  it the right choice for thousands of small LPs. Trade-offs: termination
  criteria apply to the fused problem (per-constraint residual mode is
  enabled to keep per-problem accuracy honest), the reported status is
  global (if ANY lane is infeasible, the whole fused LP is infeasible and
  no lane is solved — rerun in "loop" mode to identify culprits), and one
  hard lane can slow the collective solve.

Unlike a torch-native solver, data marshals through host memory: the cuOpt
C API ingests host arrays and manages its own GPU transfer. GPU-resident
input tensors are copied to CPU (this is cheap next to the solve).

Requires: the compiled extension (see load_extension / setup.py) and
libcuopt. cuOpt itself requires a CUDA GPU; see the cuOpt docs for system
requirements.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch

__all__ = ["cuopt_batch_linprog", "load_extension", "CuOptLinprogResult"]

_STATUS_MESSAGES = {
    0: "Optimization terminated successfully",
    1: "Iteration or time limit reached",
    2: "Problem appears to be infeasible",
    3: "Problem appears to be unbounded",
    4: "Numerical difficulties encountered",
}

_METHODS = {"concurrent": 0, "pdlp": 1, "dual_simplex": 2, "barrier": 3}


def load_extension(name: str = "cuopt_lp_op",
                   sources: Optional[Sequence[str]] = None,
                   cuopt_root: Optional[str] = None,
                   extra_include_paths: Optional[Sequence[str]] = None,
                   extra_ldflags: Optional[Sequence[str]] = None,
                   verbose: bool = False):
    """JIT-build and load the C++ op (alternative to `pip install .`).

    Looks for cuOpt under `cuopt_root` (or $CUOPT_ROOT), expecting
    include/cuopt/linear_programming/cuopt_c.h and lib/libcuopt.so.
    """
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    if sources is None:
        sources = [os.path.join(here, "cuopt_lp_op.cpp")]
    inc, ld = list(extra_include_paths or []), list(extra_ldflags or [])
    root = cuopt_root or os.environ.get("CUOPT_ROOT")
    if root:
        inc.append(os.path.join(root, "include"))
        ld += [f"-L{os.path.join(root, 'lib')}",
               f"-Wl,-rpath,{os.path.join(root, 'lib')}"]
    ld.append("-lcuopt")
    return load(name=name, sources=list(sources), extra_include_paths=inc,
                extra_ldflags=ld, verbose=verbose)


def _ensure_op_loaded():
    if not hasattr(torch.ops.cuopt_lp, "solve_batch") or \
            torch.ops.cuopt_lp.solve_batch is None:
        try:
            import cuopt_lp_op  # noqa: F401  (installed via setup.py)
        except ImportError:
            load_extension()
    return torch.ops.cuopt_lp.solve_batch


@dataclass
class CuOptLinprogResult:
    """Batched scipy.optimize.OptimizeResult analogue (cuOpt backend)."""

    x: torch.Tensor        # (*batch, n)
    fun: torch.Tensor      # (*batch,)
    slack: Optional[torch.Tensor]   # (*batch, m_ub) b_ub - A_ub @ x
    con: Optional[torch.Tensor]     # (*batch, m_eq) b_eq - A_eq @ x
    status: torch.Tensor   # (*batch,) scipy codes 0..4
    success: torch.Tensor  # (*batch,) bool
    ineqlin_marginals: Optional[torch.Tensor]  # duals of A_ub rows
    eqlin_marginals: Optional[torch.Tensor]    # duals of A_eq rows
    solve_time: torch.Tensor  # (*batch,) seconds reported by cuOpt
    message: str

    def __repr__(self) -> str:
        return (f"CuOptLinprogResult(batch_shape={tuple(self.status.shape)}, "
                f"n={self.x.shape[-1]}, "
                f"success={int(self.success.sum())}/{self.status.numel()}, "
                f"message={self.message!r})")


# ---------------------------------------------------------------------------
# input handling (scipy conventions)
# ---------------------------------------------------------------------------

def _parse_bounds(bounds, n):
    inf = math.inf

    def norm(p):
        l, u = p
        return (-inf if l is None else float(l),
                inf if u is None else float(u))

    if bounds is None:
        pairs = [(0.0, inf)] * n
    else:
        bl = list(bounds)
        if len(bl) == 2 and not isinstance(bl[0], (tuple, list)):
            pairs = [norm(bl)] * n
        elif len(bl) == n:
            pairs = [norm(p) for p in bl]
        else:
            raise ValueError(f"bounds must be None, one (lb, ub) pair, or a "
                             f"sequence of {n} pairs")
    lb = torch.tensor([p[0] for p in pairs], dtype=torch.float64)
    ub = torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    if (lb > ub).any():
        raise ValueError("bound lb > ub for some variable")
    return lb, ub


def _to_f64_cpu(v, name):
    if v is None:
        return None
    t = torch.as_tensor(v, dtype=torch.float64).cpu()
    if not (torch.isfinite(t) | torch.isinf(t)).all():
        raise ValueError(f"{name} contains nan entries")
    return t


def _broadcast_batch(entries):
    """entries: list of (tensor_or_None, trailing_dims). Returns
    (batch_shape, tensors flattened to (B, ...))."""
    shapes = [t.shape[:t.dim() - nd] for t, nd in entries if t is not None]
    batch_shape = torch.broadcast_shapes(*shapes) if shapes else ()
    B = 1
    for s in batch_shape:
        B *= s
    out = []
    for t, nd in entries:
        if t is None:
            out.append(None)
            continue
        trailing = t.shape[t.dim() - nd:]
        if len(batch_shape):
            t = t.expand(*batch_shape, *trailing)
        out.append(t.reshape(B, *trailing))
    return batch_shape, out


def _dense_to_csr(A):
    """(B, m, n) or (m, n) dense -> shared CSR pattern + (B, nnz) values.

    For batched A the pattern is the union of nonzeros across the batch,
    so all problems share one sparsity structure (zeros stored where a
    lane lacks the entry)."""
    batched = A.dim() == 3
    mask = (A != 0).any(0) if batched else (A != 0)
    m, n = mask.shape
    rows, cols = mask.nonzero(as_tuple=True)
    nnz = rows.numel()
    row_counts = torch.bincount(rows, minlength=m)
    row_off = torch.zeros(m + 1, dtype=torch.int32)
    row_off[1:] = torch.cumsum(row_counts, 0).to(torch.int32)
    col_idx = cols.to(torch.int32)
    if batched:
        values = A[:, rows, cols].contiguous()          # (B, nnz)
    else:
        values = A[rows, cols].contiguous().unsqueeze(0)  # (1, nnz)
    return row_off, col_idx, values, nnz


def _hstack_csr(row_off_list, col_idx_list, values_list, ncols_list):
    """Horizontally concatenate CSR blocks that share the same row count."""
    m = row_off_list[0].numel() - 1
    parts_ro = torch.stack([ro.to(torch.int64) for ro in row_off_list])
    row_off = parts_ro.sum(0)
    col_chunks, val_chunks = [], []
    col_base = 0
    for ro, ci, vals, nc in zip(row_off_list, col_idx_list, values_list,
                                ncols_list):
        col_chunks.append((ci.to(torch.int64) + col_base, ro))
        val_chunks.append(vals)
        col_base += nc
    # interleave per row
    out_cols, out_vals = [], []
    B = values_list[0].shape[0]
    for r in range(m):
        for (ci, ro), vals in zip(col_chunks, val_chunks):
            s, e = int(ro[r]), int(ro[r + 1])
            out_cols.append(ci[s:e])
            out_vals.append(vals[:, s:e])
    col_idx = (torch.cat(out_cols) if out_cols
               else torch.zeros(0, dtype=torch.int64)).to(torch.int32)
    values = (torch.cat(out_vals, dim=1) if out_vals
              else torch.zeros(B, 0, dtype=torch.float64))
    return row_off.to(torch.int32), col_idx, values


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def cuopt_batch_linprog(
    c: torch.Tensor,
    A_ub: Optional[torch.Tensor] = None,
    b_ub: Optional[torch.Tensor] = None,
    A_eq: Optional[torch.Tensor] = None,
    b_eq: Optional[torch.Tensor] = None,
    bounds: Union[None, Tuple, Sequence[Tuple]] = None,
    *,
    mode: str = "loop",
    method: str = "concurrent",
    tol: float = 1e-8,
    time_limit: float = 0.0,
) -> CuOptLinprogResult:
    """Batch-solve LPs with cuOpt through the custom op.

    Arguments follow scipy.optimize.linprog: c (*batch, n); A_ub/b_ub for
    A_ub @ x <= b_ub; A_eq/b_eq for equalities; scipy-style `bounds`
    (default all variables in [0, inf)). A_ub/A_eq may be dense (m, n),
    dense batched (*batch, m, n), or an unbatched torch.sparse_csr tensor
    shared across the batch.

    mode: "loop" (one cuOpt solve per problem, per-problem statuses) or
    "stacked" (one fused block-diagonal solve; see module docstring for
    trade-offs). method: "concurrent" (default), "pdlp", "dual_simplex",
    or "barrier". tol: relative primal/dual/gap tolerance. time_limit:
    seconds per solve (0 = cuOpt default).
    """
    if (A_ub is None) != (b_ub is None):
        raise ValueError("A_ub and b_ub must be supplied together")
    if (A_eq is None) != (b_eq is None):
        raise ValueError("A_eq and b_eq must be supplied together")
    if mode not in ("loop", "stacked"):
        raise ValueError("mode must be 'loop' or 'stacked'")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}")

    op = _ensure_op_loaded()
    out_device = c.device if isinstance(c, torch.Tensor) else "cpu"

    # sparse CSR input: keep pattern, treat as shared dense-equivalent
    def prep_A(A):
        if A is None:
            return None
        if isinstance(A, torch.Tensor) and A.layout == torch.sparse_csr:
            ro = A.crow_indices().to(torch.int32).cpu()
            ci = A.col_indices().to(torch.int32).cpu()
            vals = A.values().to(torch.float64).cpu().unsqueeze(0)
            return ("csr", ro, ci, vals, A.shape)
        return ("dense", _to_f64_cpu(A, "A"))

    c = _to_f64_cpu(c, "c")
    b_ub = _to_f64_cpu(b_ub, "b_ub")
    b_eq = _to_f64_cpu(b_eq, "b_eq")
    Au, Ae = prep_A(A_ub), prep_A(A_eq)

    n = c.shape[-1]
    m_ub = 0 if b_ub is None else b_ub.shape[-1]
    m_eq = 0 if b_eq is None else b_eq.shape[-1]
    lb, ub = _parse_bounds(bounds, n)

    # ---- broadcast batch dims over c, b_ub, b_eq and dense batched A ------
    dense_Au = Au[1] if (Au and Au[0] == "dense") else None
    dense_Ae = Ae[1] if (Ae and Ae[0] == "dense") else None
    batch_shape, (c, b_ub, b_eq, dAu, dAe) = _broadcast_batch([
        (c, 1), (b_ub, 1), (b_eq, 1),
        (dense_Au, 2) if dense_Au is not None and dense_Au.dim() == 3
        else (None, 2),
        (dense_Ae, 2) if dense_Ae is not None and dense_Ae.dim() == 3
        else (None, 2),
    ])
    B = c.shape[0]

    # ---- CSR assembly: rows = [A_eq; A_ub] ---------------------------------
    def block_csr(entry, dense_batched, m_rows):
        if entry is None:
            return (torch.zeros(m_rows + 1, dtype=torch.int32),
                    torch.zeros(0, dtype=torch.int32),
                    torch.zeros(1, 0, dtype=torch.float64))
        if entry[0] == "csr":
            _, ro, ci, vals, shp = entry
            if shp[1] != n:
                raise ValueError("A has wrong number of columns")
            return ro, ci, vals
        A = dense_batched if dense_batched is not None else entry[1]
        if A.shape[-1] != n:
            raise ValueError("A has wrong number of columns")
        ro, ci, vals, _ = _dense_to_csr(A)
        return ro, ci, vals

    ro_eq, ci_eq, va_eq = block_csr(Ae, dAe, m_eq)
    ro_ub, ci_ub, va_ub = block_csr(Au, dAu, m_ub)

    m = m_eq + m_ub
    if m == 0:
        raise ValueError("at least one constraint row is required")

    nnz_eq, nnz_ub = ci_eq.numel(), ci_ub.numel()
    row_off = torch.cat([ro_eq.to(torch.int64),
                         ro_ub.to(torch.int64)[1:] + nnz_eq]).to(torch.int32)
    col_idx = torch.cat([ci_eq, ci_ub])

    def tile_vals(v):
        return v.expand(B, v.shape[1]) if v.shape[0] == 1 else v
    values = torch.cat([tile_vals(va_eq), tile_vals(va_ub)], dim=1)

    inf = math.inf
    con_lb = torch.empty(B, m, dtype=torch.float64)
    con_ub_t = torch.empty(B, m, dtype=torch.float64)
    if m_eq:
        con_lb[:, :m_eq] = b_eq
        con_ub_t[:, :m_eq] = b_eq
    if m_ub:
        con_lb[:, m_eq:] = -inf
        con_ub_t[:, m_eq:] = b_ub
    var_lb = lb.unsqueeze(0).expand(B, n)
    var_ub = ub.unsqueeze(0).expand(B, n)

    # ---- solve --------------------------------------------------------------
    if mode == "loop":
        x, fun, status, duals, _rc, stime = op(
            c.contiguous(), row_off, col_idx, values.contiguous(),
            con_lb.contiguous(), con_ub_t.contiguous(),
            var_lb.contiguous(), var_ub.contiguous(),
            tol, time_limit, _METHODS[method], False)
    else:  # stacked: one block-diagonal LP, B copies along the diagonal
        nnz = col_idx.numel()
        big_ro = (row_off.to(torch.int64).unsqueeze(0)[:, 1:]
                  + nnz * torch.arange(B).unsqueeze(1)).reshape(-1)
        big_ro = torch.cat([torch.zeros(1, dtype=torch.int64), big_ro])
        big_ci = (col_idx.to(torch.int64).unsqueeze(0)
                  + n * torch.arange(B).unsqueeze(1)).reshape(-1)
        total_n, total_m = B * n, B * m
        if total_n > 2**31 - 1 or big_ci.numel() > 2**31 - 1:
            raise ValueError("stacked problem exceeds 32-bit index range; "
                             "use mode='loop' or split the batch")
        x, fun, status, duals, _rc, stime = op(
            c.reshape(1, total_n).contiguous(),
            big_ro.to(torch.int32), big_ci.to(torch.int32),
            values.reshape(1, -1).contiguous(),
            con_lb.reshape(1, total_m).contiguous(),
            con_ub_t.reshape(1, total_m).contiguous(),
            var_lb.reshape(1, total_n).contiguous(),
            var_ub.reshape(1, total_n).contiguous(),
            tol, time_limit, _METHODS[method], True)
        x = x.reshape(B, n)
        duals = duals.reshape(B, m)
        fun = (c * x).sum(-1)
        status = status.expand(B).clone()   # global status for every lane
        stime = stime.expand(B).clone()

    # ---- scipy-style outputs -------------------------------------------------
    slack = con_out = ineq_marg = eq_marg = None
    valid = torch.isfinite(x).all(-1)
    if m_ub and Au is not None:
        if Au[0] == "csr" or dAu is None:
            A_dense = (Au[1] if Au[0] == "dense" else None)
            if A_dense is None:  # sparse shared
                Ax = torch.zeros(B, m_ub, dtype=torch.float64)
                _, ro, ci, vals, _ = Au
                Ad = torch.sparse_csr_tensor(
                    ro.to(torch.int64), ci.to(torch.int64), vals[0],
                    size=(m_ub, n)).to_dense()
                Ax = torch.where(valid.unsqueeze(-1),
                                 torch.nan_to_num(x) @ Ad.T, torch.nan)
            else:
                Ax = torch.where(valid.unsqueeze(-1),
                                 torch.nan_to_num(x) @ A_dense.T, torch.nan)
        else:
            Ax = torch.einsum("bmn,bn->bm", dAu, torch.nan_to_num(x))
            Ax = torch.where(valid.unsqueeze(-1), Ax, torch.nan)
        slack = b_ub - Ax
        ineq_marg = duals[:, m_eq:]
    if m_eq and Ae is not None:
        if Ae[0] == "dense" and dAe is not None:
            Ax = torch.einsum("bmn,bn->bm", dAe, torch.nan_to_num(x))
        elif Ae[0] == "dense":
            Ax = torch.nan_to_num(x) @ Ae[1].T
        else:
            _, ro, ci, vals, _ = Ae
            Ad = torch.sparse_csr_tensor(ro.to(torch.int64),
                                         ci.to(torch.int64), vals[0],
                                         size=(m_eq, n)).to_dense()
            Ax = torch.nan_to_num(x) @ Ad.T
        con_out = b_eq - torch.where(valid.unsqueeze(-1), Ax, torch.nan)
        eq_marg = duals[:, :m_eq]

    counts = torch.bincount(status.clamp(0, 4), minlength=5)
    message = "; ".join(f"{int(counts[k])}x status {k} "
                        f"({_STATUS_MESSAGES[k]})"
                        for k in range(5) if counts[k] > 0)
    if mode == "stacked" and int(status[0]) in (2, 3, 4):
        message += (" [stacked mode: status is for the fused problem; rerun "
                    "with mode='loop' to identify which lanes are at fault]")

    def shape_back(v, nd):
        if v is None:
            return None
        v = v.to(out_device)
        return v.reshape(*batch_shape, *v.shape[1:1 + nd]) if batch_shape \
            else v.squeeze(0)

    return CuOptLinprogResult(
        x=shape_back(x, 1), fun=shape_back(fun, 0),
        slack=shape_back(slack, 1), con=shape_back(con_out, 1),
        status=shape_back(status, 0), success=shape_back(status == 0, 0),
        ineqlin_marginals=shape_back(ineq_marg, 1),
        eqlin_marginals=shape_back(eq_marg, 1),
        solve_time=shape_back(stime, 0), message=message,
    )
