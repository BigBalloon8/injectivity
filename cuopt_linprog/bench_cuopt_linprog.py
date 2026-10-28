#!/usr/bin/env python
"""Speed benchmarks for torch-cuopt-lp (cuopt_batch_linprog / the C++ op).

    # against real cuOpt: the numbers that matter
    CUOPT_ROOT=/path/to/cuopt python bench_cuopt_linprog.py --baseline

    # against the CPU testing stub: checks the harness runs, nothing more
    CUOPT_ROOT=testing_stub LD_LIBRARY_PATH=testing_stub/lib \\
        python bench_cuopt_linprog.py --preset quick --tol 1e-4

    # wrapper-only overhead: needs neither cuOpt nor the compiled extension
    python bench_cuopt_linprog.py --suite overhead

Suites (choose with --suite, repeatable; default is all, run in this order):

  overhead  Python wrapper cost with the solver swapped for a no-op op.
            Isolates input checks, CSR assembly, block-diagonal stacking and
            slack/con post-processing from anything cuOpt does.
  cold      Extension load (import, or JIT build if not installed), first
            solve (library + CUDA context init) and second solve.
  batch     loop vs stacked as the batch of small LPs grows: the README's
            "stacked wins for thousands of small LPs" crossover.
  size      loop vs stacked as each LP grows, at a fixed batch.
  methods   concurrent / pdlp / dual_simplex / barrier.
  inputs    The same LPs through each input path: dense shared A, sparse
            CSR A, per-variable bounds list, dense batched A, equality rows,
            CUDA-resident tensors (when a GPU is visible).

End-to-end rows split each call's wall time into
  py       the Python wrapper: everything outside the C++ op
  marshal  inside the op but not counted by cuOpt as solve time: problem
           construction, settings, host<->device transfer, copy-out
  solver   cuOpt's own reported solve_time (summed over lanes in loop mode)
and give the fraction of lanes that came back optimal ("ok"). A fast row
with ok < 100% is not a fast solve. --baseline adds scipy's HiGHS looped over
the same batch, "speedup" = scipy time / our time (> 1 means faster than
scipy), and the max relative objective error against scipy.

Regression checks: --save run.json writes the results; a later run with
--compare run.json prints new/old ratios and, with --fail-on-regression,
exits 1 if any case got slower than --threshold (cold-start rows are shown
but not gated: they depend on filesystem and JIT-cache state).
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import datetime
import json
import math
import os
import platform
import statistics
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

import cuopt_linprog as cl

# torch warns that sparse CSR is beta / invariant checks are off; not news.
warnings.filterwarnings("ignore", message=r".*[Ss]parse.*", category=UserWarning)

F64 = torch.float64
NAN = float("nan")
ALL_SUITES = ("overhead", "cold", "batch", "size", "methods", "inputs")
E2E_SUITES = ALL_SUITES[1:]
BOX = (0.0, 3.0)   # variable bounds of every generated LP: keeps it bounded

# sizes: (m, n, sparse). Sparse problems have `nnz_per_row` entries per row.
PRESETS: Dict[str, dict] = {
    # small enough for the CPU testing stub; about a minute there
    "quick": dict(
        batch_mn=(20, 10), batch_sizes=[1, 16, 128],
        size_batch=4,
        sizes=[(10, 10, False), (40, 40, False), (100, 100, False),
               (400, 400, True)],
        methods_case=(16, 20, 10), inputs_case=(16, 20, 10),
        overhead_batch=[1, 256, 4096], overhead_sparse_n=[1000, 2000, 4000],
        overhead_bounds_n=[10_000, 100_000],
        loop_max_batch=4096, nnz_per_row=5),
    # meant for a GPU box with real cuOpt
    "default": dict(
        batch_mn=(20, 10), batch_sizes=[1, 8, 64, 512, 4096],
        size_batch=16,
        sizes=[(10, 10, False), (100, 100, False), (500, 500, False),
               (2000, 2000, True), (8000, 8000, True)],
        methods_case=(64, 50, 50), inputs_case=(64, 50, 50),
        overhead_batch=[1, 64, 512, 4096, 32768],
        overhead_sparse_n=[1000, 4000, 12000],
        overhead_bounds_n=[10_000, 100_000, 1_000_000],
        loop_max_batch=4096, nnz_per_row=5),
    "full": dict(
        batch_mn=(20, 10), batch_sizes=[1, 8, 64, 512, 4096, 16384, 65536],
        size_batch=16,
        sizes=[(10, 10, False), (100, 100, False), (500, 500, False),
               (1000, 1000, False), (2000, 2000, True), (8000, 8000, True),
               (12000, 12000, True)],
        methods_case=(256, 100, 100), inputs_case=(256, 100, 100),
        overhead_batch=[1, 64, 512, 4096, 32768, 131072],
        overhead_sparse_n=[1000, 4000, 16000],
        overhead_bounds_n=[10_000, 100_000, 1_000_000],
        loop_max_batch=4096, nnz_per_row=5),
}


# ---------------------------------------------------------------------------
# problem generation
# ---------------------------------------------------------------------------

@dataclass
class LP:
    """A batch of B LPs (n variables, m constraint rows each) plus the
    keyword arguments that solve it. `key` identifies the underlying LPs, so
    input-path variants of the same problems share one scipy reference."""
    kwargs: dict
    B: int
    m: int
    n: int
    nnz: int            # nonzeros of one LP's constraint matrix
    key: str
    device: str = "cpu"

    def solve(self, **opts):
        return cl.cuopt_batch_linprog(**self.kwargs, **opts)


def _nnz(A: torch.Tensor) -> int:
    if A.layout == torch.sparse_csr:
        return int(A.values().numel())
    return int(A.shape[-2] * A.shape[-1])   # randn: no exact zeros


def make_lp(B: int, m: int, n: int, *, sparse: bool = False,
            nnz_per_row: int = 5, batched_A: bool = False, m_eq: int = 0,
            seed: int = 0) -> LP:
    """Random LPs that are feasible (an interior x0 satisfies every row) and
    bounded (box bounds [0, 3]). `m` inequality rows plus `m_eq` equality
    rows. Dense A is shared across the batch unless `batched_A`; sparse A is
    a shared CSR matrix with `nnz_per_row` entries per row."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.rand(B, n, generator=g, dtype=F64) * 1.5 + 0.5   # (0.5, 2)

    def matrix(rows: int) -> torch.Tensor:
        if sparse:
            k = min(nnz_per_row, n)
            r = torch.arange(rows).repeat_interleave(k)
            col = torch.randint(0, n, (rows * k,), generator=g)
            v = torch.randn(rows * k, generator=g, dtype=F64)
            return torch.sparse_coo_tensor(torch.stack([r, col]), v,
                                           (rows, n)).coalesce().to_sparse_csr()
        shape = (B, rows, n) if batched_A else (rows, n)
        return torch.randn(*shape, generator=g, dtype=F64)

    def times_x0(A: torch.Tensor) -> torch.Tensor:
        if A.layout == torch.sparse_csr:
            return (A @ x0.T).T
        if A.dim() == 3:
            return torch.einsum("bmn,bn->bm", A, x0)
        return x0 @ A.T

    A_ub = matrix(m)
    kw = dict(c=torch.randn(B, n, generator=g, dtype=F64), bounds=BOX,
              A_ub=A_ub,
              b_ub=times_x0(A_ub) + torch.rand(B, m, generator=g, dtype=F64)
              * 0.9 + 0.1)
    nnz = _nnz(A_ub)
    if m_eq:
        A_eq = matrix(m_eq)
        kw.update(A_eq=A_eq, b_eq=times_x0(A_eq))
        nnz += _nnz(A_eq)
    key = (f"B{B}-m{m}-eq{m_eq}-n{n}-{'sp' if sparse else 'de'}"
           f"{'-bA' if batched_A else ''}-s{seed}")
    return LP(kw, B, m + m_eq, n, nnz, key)


def as_csr(lp: LP) -> LP:
    """Same LPs, shared dense A matrices passed as torch.sparse_csr."""
    kw = dict(lp.kwargs)
    for name in ("A_ub", "A_eq"):
        if kw.get(name) is not None and kw[name].layout == torch.strided:
            kw[name] = kw[name].to_sparse_csr()
    return dataclasses.replace(lp, kwargs=kw)


def with_bounds_list(lp: LP) -> LP:
    """Same LPs, bounds given as one (lb, ub) pair per variable."""
    kw = dict(lp.kwargs, bounds=[BOX] * lp.n)
    return dataclasses.replace(lp, kwargs=kw)


def to_device(lp: LP, device: str) -> LP:
    kw = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
          for k, v in lp.kwargs.items()}
    return dataclasses.replace(lp, kwargs=kw, device=device)


def as_equalities(lp: LP) -> LP:
    """Move the A_ub rows to A_eq (shape-only: for null-op overhead runs,
    where feasibility is irrelevant)."""
    kw = dict(lp.kwargs)
    kw["A_eq"], kw["b_eq"] = kw.pop("A_ub"), kw.pop("b_ub")
    return dataclasses.replace(lp, kwargs=kw, key=lp.key + "-aseq")


# ---------------------------------------------------------------------------
# timing machinery
# ---------------------------------------------------------------------------

def measure(fn: Callable, *, warmup: int = 1, repeat: int = 5,
            budget: float = 10.0, sync: bool = False):
    """Call fn `warmup` times untimed, then up to `repeat` timed times,
    stopping early once `budget` seconds of timed runs have passed (always
    at least one). Returns (list of seconds, result of the last call)."""
    result = None
    for _ in range(warmup):
        result = fn()
    if sync:
        torch.cuda.synchronize()
    samples: List[float] = []
    start = time.perf_counter()
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        result = fn()
        if sync:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
        if time.perf_counter() - start >= budget:
            break
    return samples, result


class OpProbe:
    """Stand-in for torch.ops.cuopt_lp.solve_batch that forwards to `op` and
    records, per call, the seconds spent inside the op and the solve time
    cuOpt reported (summed over the lanes of that call)."""

    def __init__(self, op):
        self.op = op
        self.op_s: List[float] = []
        self.solver_s: List[float] = []

    def __call__(self, *args):
        t0 = time.perf_counter()
        out = self.op(*args)
        self.op_s.append(time.perf_counter() - t0)
        self.solver_s.append(float(out[5].sum()))
        return out


def null_op(c, row_off, col_idx, values, con_lb, *_):
    """Same signature and outputs as the real op (every lane 'optimal' at
    x = 0) but solves nothing: what's left is pure wrapper cost."""
    B, n = c.shape
    m = con_lb.shape[1]
    z = lambda *s: torch.zeros(*s, dtype=F64)  # noqa: E731
    return [z(B, n), z(B), torch.zeros(B, dtype=torch.long), z(B, m), z(B, n),
            z(B)]


@contextlib.contextmanager
def use_op(op):
    """Make cuopt_batch_linprog call `op` instead of the compiled extension."""
    orig = cl._ensure_op_loaded
    cl._ensure_op_loaded = lambda: op
    try:
        yield op
    finally:
        cl._ensure_op_loaded = orig


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def backend_info() -> dict:
    """Which libcuopt the process has loaded, and whether it's the CPU
    testing stub (whose timings say nothing about cuOpt). Set
    CUOPT_BENCH_STUB=0/1 to override the detection."""
    path = None
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcuopt" in line:
                    path = line.split()[-1]
                    break
    except OSError:
        pass
    stub: Optional[bool] = None
    if path:
        stub = "testing_stub" in path
        if not stub:
            try:   # the stub implements only what the extension calls
                stub = not hasattr(ctypes.CDLL(path), "cuOptCreateProblem")
            except OSError:
                pass
    override = os.environ.get("CUOPT_BENCH_STUB")
    if override in ("0", "1"):
        stub = override == "1"
    return {"libcuopt": path, "stub": stub}


def environment() -> dict:
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(), "gpu": gpu, **backend_info(),
    }


# ---------------------------------------------------------------------------
# scipy reference
# ---------------------------------------------------------------------------

def _scipy_inputs(lp: LP) -> List[dict]:
    """Per-lane numpy/scipy arguments (built once, outside the timing)."""
    import scipy.sparse as sp

    kw = lp.kwargs

    def conv(t, i, nd):
        if t is None:
            return None
        if t.layout == torch.sparse_csr:
            t = t.cpu()
            return sp.csr_matrix((t.values().numpy(), t.col_indices().numpy(),
                                  t.crow_indices().numpy()),
                                 shape=tuple(t.shape))
        t = t.detach().cpu()
        return (t[i] if t.dim() > nd else t).numpy()

    shared = {k: conv(kw.get(k), 0, 2) for k in ("A_ub", "A_eq")
              if kw.get(k) is not None and (kw[k].layout == torch.sparse_csr
                                            or kw[k].dim() == 2)}
    lanes = []
    for i in range(lp.B):
        lane = dict(c=conv(kw["c"], i, 1), bounds=kw["bounds"])
        for A, b in (("A_ub", "b_ub"), ("A_eq", "b_eq")):
            if kw.get(A) is not None:
                lane[A] = shared[A] if A in shared else conv(kw[A], i, 2)
                lane[b] = conv(kw[b], i, 1)
        lanes.append(lane)
    return lanes


def scipy_solve(lanes: List[dict]):
    import numpy as np
    from scipy.optimize import linprog

    fun = np.full(len(lanes), np.nan)
    status = np.zeros(len(lanes), dtype=int)
    for i, lane in enumerate(lanes):
        r = linprog(method="highs", **lane)
        status[i] = r.status
        if r.status == 0:
            fun[i] = r.fun
    return fun, status


# ---------------------------------------------------------------------------
# records and output
# ---------------------------------------------------------------------------

@dataclass
class Record:
    suite: str
    case: str
    B: int = 0
    m: int = 0
    n: int = 0
    nnz: int = 0
    mode: str = ""
    method: str = ""
    runs: int = 0
    median_s: float = NAN
    min_s: float = NAN
    py_s: float = NAN
    marshal_s: float = NAN
    solver_s: float = NAN
    ok: float = NAN          # fraction of lanes with status 0
    scipy_s: float = NAN
    rel_err: float = NAN     # max |fun - scipy| / (1 + |scipy|)
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.case}"


def fmt_t(s: float) -> str:
    if s is None or math.isnan(s):
        return "-"
    if s < 1e-3:
        return f"{s * 1e6:.0f}µs"
    if s < 1:
        return f"{s * 1e3:.2f}ms"
    return f"{s:.2f}s"


def fmt_rate(r: float) -> str:
    if math.isnan(r):
        return "-"
    for unit, k in (("M", 1e6), ("k", 1e3)):
        if r >= k:
            return f"{r / k:.1f}{unit}"
    return f"{r:.1f}"


def pct(part: float, total: float) -> str:
    if math.isnan(part) or not total or math.isnan(total):
        return "-"
    return f"{100 * max(part, 0.0) / total:.0f}%"


SUITE_TITLES = {
    "overhead": "Python wrapper overhead, solver replaced by a no-op",
    "cold": "cold start",
    "batch": "loop vs stacked as the batch grows",
    "size": "loop vs stacked as each LP grows",
    "methods": "solver methods",
    "inputs": "input paths, same LPs",
}


class Printer:
    def __init__(self, baseline: bool):
        self.baseline = baseline

    def header(self, suite: str, subtitle: str = ""):
        title = SUITE_TITLES[suite] + (f" ({subtitle})" if subtitle else "")
        print(f"\n== {suite}: {title} ==")
        if suite == "overhead":
            cols = (f"{'case':<40}{'median':>10}{'min':>10}{'per LP':>10}"
                    f"{'nnz':>9}")
        elif suite == "cold":
            cols = f"{'case':<40}{'time':>10}"
        else:
            cols = (f"{'case':<28}{'median':>10}{'min':>10}{'LP/s':>9}"
                    f"{'py':>6}{'marsh':>7}{'solver':>7}{'ok':>6}")
            if self.baseline:
                cols += f"{'scipy':>10}{'speedup':>9}{'rel.err':>9}"
        print(cols)
        print("-" * len(cols))

    def row(self, r: Record):
        if r.runs == 0:
            print(f"{r.case:<28}  skipped: {r.note}")
            return
        if r.suite == "overhead":
            per = r.median_s / max(r.B, 1)
            print(f"{r.case:<40}{fmt_t(r.median_s):>10}{fmt_t(r.min_s):>10}"
                  f"{fmt_t(per):>10}{r.nnz:>9}")
            return
        if r.suite == "cold":
            print(f"{r.case:<40}{fmt_t(r.median_s):>10}")
            return
        parts = (r.py_s, r.marshal_s, r.solver_s)
        total = sum(max(p, 0.0) for p in parts)
        ok = "-" if math.isnan(r.ok) else f"{100 * r.ok:.0f}%"
        flag = "" if (math.isnan(r.ok) or r.ok == 1) else "  <- not all optimal"
        line = (f"{r.case:<28}{fmt_t(r.median_s):>10}{fmt_t(r.min_s):>10}"
                f"{fmt_rate(r.B / r.median_s):>9}"
                f"{pct(r.py_s, total):>6}{pct(r.marshal_s, total):>7}"
                f"{pct(r.solver_s, total):>7}{ok:>6}")
        if self.baseline:
            speed = (f"{r.scipy_s / r.median_s:.1f}x"
                     if not math.isnan(r.scipy_s) else "-")
            err = "-" if math.isnan(r.rel_err) else f"{r.rel_err:.0e}"
            line += f"{fmt_t(r.scipy_s):>10}{speed:>9}{err:>9}"
        print(line + flag + (f"  ({r.note})" if r.note else ""))


# ---------------------------------------------------------------------------
# the suites
# ---------------------------------------------------------------------------

class Bench:
    def __init__(self, args, preset: dict):
        self.args = args
        self.p = preset
        self.modes = args.modes
        self.loop_max_batch = (args.loop_max_batch
                               if args.loop_max_batch is not None
                               else preset["loop_max_batch"])
        self.op = None
        self.load_s = NAN
        self._scipy: Dict[str, Tuple[float, object, object]] = {}

    # -- single measurements ------------------------------------------------

    def _measure(self, fn, *, sync=False, repeat=None):
        a = self.args
        return measure(fn, warmup=a.warmup,
                       repeat=a.repeat if repeat is None else repeat,
                       budget=a.budget, sync=sync)

    def wrapper_only(self, case: str, lp: LP, mode: str) -> Record:
        with use_op(null_op):
            samples, _ = self._measure(lambda: lp.solve(mode=mode),
                                       repeat=self.args.repeat * 4)
        return Record("overhead", case, lp.B, lp.m, lp.n, lp.nnz, mode=mode,
                      runs=len(samples), median_s=statistics.median(samples),
                      min_s=min(samples))

    def e2e(self, suite: str, case: str, lp: LP, mode: str,
            method: Optional[str] = None) -> Record:
        method = method or self.args.method
        if mode == "loop" and lp.B > self.loop_max_batch:
            return Record(suite, case, lp.B, lp.m, lp.n, lp.nnz, mode=mode,
                          method=method,
                          note=f"B > --loop-max-batch {self.loop_max_batch}")
        opts = dict(mode=mode, method=method, tol=self.args.tol,
                    time_limit=self.args.time_limit)
        probe = OpProbe(self.op)
        with use_op(probe):
            samples, res = self._measure(lambda: lp.solve(**opts),
                                         sync=lp.device != "cpu")
        k = len(samples)
        op_s, sol_s = probe.op_s[-k:], probe.solver_s[-k:]
        med = statistics.median
        rec = Record(
            suite, case, lp.B, lp.m, lp.n, lp.nnz, mode=mode, method=method,
            runs=k, median_s=med(samples), min_s=min(samples),
            py_s=med(w - o for w, o in zip(samples, op_s)),
            marshal_s=med(o - s for o, s in zip(op_s, sol_s)),
            solver_s=med(sol_s),
            ok=float(res.success.reshape(-1).float().mean()))
        if self.args.baseline:
            self._compare_to_scipy(rec, lp, res)
        return rec

    def _compare_to_scipy(self, rec: Record, lp: LP, res):
        if lp.key not in self._scipy:
            lanes = _scipy_inputs(lp)
            samples, (fun, status) = measure(
                lambda: scipy_solve(lanes), warmup=0,
                repeat=min(3, self.args.repeat), budget=self.args.budget)
            self._scipy[lp.key] = (statistics.median(samples), fun, status)
        t, fun, status = self._scipy[lp.key]
        rec.scipy_s = t
        mine = res.fun.reshape(-1).detach().cpu().numpy()
        good = (res.status.reshape(-1).cpu().numpy() == 0) & (status == 0)
        if good.any():
            rec.rel_err = float((abs(mine[good] - fun[good])
                                 / (1 + abs(fun[good]))).max())

    # -- suites -------------------------------------------------------------

    def suite_overhead(self) -> Iterator[Record]:
        p = self.p
        m, n = p["batch_mn"]
        for mode in self.modes:
            for B in p["overhead_batch"]:
                yield self.wrapper_only(f"{mode:<8}dense shared A  B={B}",
                                        make_lp(B, m, n), mode)
        for mode in self.modes:
            for B in p["overhead_batch"][1:]:
                yield self.wrapper_only(f"{mode:<8}dense batched A B={B}",
                                        make_lp(B, m, n, batched_A=True), mode)
        # Fixed nonzeros per row: nnz grows like n, the dense m*n like n^2.
        # CSR prep and post-processing don't depend on the mode, so these
        # rows use only the first selected mode.
        mode = self.modes[0]
        tag = f"{mode:<8}"
        for n_sp in p["overhead_sparse_n"]:
            lp = make_lp(8, n_sp // 2, n_sp, sparse=True,
                         nnz_per_row=p["nnz_per_row"])
            yield self.wrapper_only(f"{tag}sparse A_ub B=8 n={n_sp}", lp,
                                    mode)
            yield self.wrapper_only(f"{tag}sparse A_eq B=8 n={n_sp}",
                                    as_equalities(lp), mode)
        # one LP with many variables: cost of bounds parsing
        for n_b in p["overhead_bounds_n"]:
            lp = make_lp(1, 1, n_b, sparse=True, nnz_per_row=5)
            yield self.wrapper_only(f"{tag}bounds pair  B=1 n={n_b}", lp,
                                    mode)
            yield self.wrapper_only(f"{tag}bounds list  B=1 n={n_b}",
                                    with_bounds_list(lp), mode)

    def suite_cold(self) -> Iterator[Record]:
        # main() times the load before any suite runs, so it is truly first
        yield Record("cold", "load extension (import or JIT build)", runs=1,
                     median_s=self.load_s)
        lp = make_lp(1, 4, 3)
        for label in ("first solve (library/CUDA init)", "second solve"):
            t0 = time.perf_counter()
            lp.solve(mode=self.modes[0], tol=self.args.tol,
                     method=self.args.method)
            yield Record("cold", label, lp.B, lp.m, lp.n, lp.nnz, runs=1,
                         median_s=time.perf_counter() - t0)

    def suite_batch(self) -> Iterator[Record]:
        m, n = self.p["batch_mn"]
        for B in self.p["batch_sizes"]:
            lp = make_lp(B, m, n, seed=B)
            for mode in self.modes:
                yield self.e2e("batch", f"{mode:<8}B={B}", lp, mode)

    def suite_size(self) -> Iterator[Record]:
        B = self.p["size_batch"]
        for m, n, sparse in self.p["sizes"]:
            lp = make_lp(B, m, n, sparse=sparse, seed=n,
                         nnz_per_row=self.p["nnz_per_row"])
            label = f"{'sparse' if sparse else 'dense'} {m}x{n}"
            for mode in self.modes:
                yield self.e2e("size", f"{mode:<8}{label}", lp, mode)

    def suite_methods(self) -> Iterator[Record]:
        B, m, n = self.p["methods_case"]
        lp = make_lp(B, m, n, seed=2)
        for method in cl._METHODS:
            for mode in self.modes:
                yield self.e2e("methods", f"{mode:<8}{method}", lp, mode,
                               method)

    def suite_inputs(self) -> Iterator[Record]:
        B, m, n = self.p["inputs_case"]
        base = make_lp(B, m, n, seed=1)
        variants = [("dense shared A", base),
                    ("sparse CSR A", as_csr(base)),
                    ("bounds list", with_bounds_list(base)),
                    ("dense batched A", make_lp(B, m, n, batched_A=True,
                                                seed=1)),
                    ("+ equality rows", make_lp(B, m - m // 4, n,
                                                m_eq=m // 4, seed=1))]
        if torch.cuda.is_available():
            variants.append(("CUDA tensors", to_device(base, "cuda")))
        for label, lp in variants:
            for mode in self.modes:
                yield self.e2e("inputs", f"{mode:<8}{label}", lp, mode)

    def subtitle(self, suite: str) -> str:
        p = self.p
        if suite == "batch":
            m, n = p["batch_mn"]
            return f"m={m}, n={n}, dense A"
        if suite == "size":
            return f"B={p['size_batch']}"
        if suite in ("methods", "inputs"):
            B, m, n = p[f"{suite}_case"]
            return f"B={B}, m={m}, n={n}"
        return ""


# ---------------------------------------------------------------------------
# save / compare
# ---------------------------------------------------------------------------

def _clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in d.items()}


def save(path: str, env: dict, args, records: Sequence[Record]):
    data = {"env": env, "args": vars(args),
            "records": [_clean(asdict(r)) for r in records]}
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"\nsaved {len(records)} results to {path}")


def compare(records: Sequence[Record], env: dict, path: str,
            threshold: float, stat: str) -> List[str]:
    """Print new/old ratios against a saved run; return the regressed keys."""
    with open(path) as f:
        old = json.load(f)
    print(f"\n== compare with {path} ({stat}, threshold "
          f"{100 * threshold:.0f}%) ==")
    for field in ("libcuopt", "stub", "gpu", "torch"):
        if old["env"].get(field) != env.get(field):
            print(f"  note: {field} differs: {old['env'].get(field)!r} -> "
                  f"{env.get(field)!r}")
    old_by = {f"{r['suite']}/{r['case']}": r for r in old["records"]}
    regressed: List[str] = []
    print(f"{'case':<50}{'old':>10}{'new':>10}{'ratio':>8}")
    for r in records:
        o = old_by.get(r.key)
        new_v = getattr(r, stat)
        if not o or o.get(stat) is None or r.runs == 0 or math.isnan(new_v):
            continue
        ratio = new_v / o[stat]
        flag = ""
        if ratio > 1 + threshold:
            flag = "  SLOWER" + (" (not gated)" if r.suite == "cold" else "")
            if r.suite != "cold":
                regressed.append(r.key)
        elif ratio < 1 / (1 + threshold):
            flag = "  faster"
        print(f"{r.key:<50}{fmt_t(o[stat]):>10}{fmt_t(new_v):>10}"
              f"{ratio:>7.2f}x{flag}")
    missing = set(old_by) - {r.key for r in records}
    if missing:
        print(f"  ({len(missing)} cases in the old run were not run now)")
    print(f"{len(regressed)} regression(s)")
    return regressed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _modes(s: str) -> List[str]:
    modes = [m.strip() for m in s.split(",") if m.strip()]
    bad = [m for m in modes if m not in ("loop", "stacked")]
    if bad or not modes:
        raise argparse.ArgumentTypeError(
            f"expected 'loop', 'stacked' or 'loop,stacked', got {s!r}")
    return modes


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Speed benchmarks for torch-cuopt-lp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="default",
                    help="problem sizes (quick suits the CPU stub)")
    ap.add_argument("--suite", action="append", choices=ALL_SUITES,
                    help="suite to run (repeatable; default all)")
    ap.add_argument("--modes", type=_modes, default=["loop", "stacked"],
                    help="comma-separated batching modes to run, e.g. "
                         "'--modes stacked' to skip loop mode entirely "
                         "(default: loop,stacked)")
    ap.add_argument("--method", default="concurrent", choices=sorted(cl._METHODS),
                    help="cuOpt method for every suite except 'methods'")
    ap.add_argument("--tol", type=float, default=1e-8,
                    help="solver tolerance (library default 1e-8; the stub "
                         "wants ~1e-4)")
    ap.add_argument("--time-limit", type=float, default=0.0,
                    help="per-solve time limit in seconds (0 = cuOpt default)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5,
                    help="max timed runs per case")
    ap.add_argument("--budget", type=float, default=10.0,
                    help="stop repeating a case after this many seconds")
    ap.add_argument("--loop-max-batch", type=int, default=None,
                    help="skip loop-mode cases with a larger batch")
    ap.add_argument("--baseline", action="store_true",
                    help="also time scipy HiGHS on each batch and report "
                         "objective error against it")
    ap.add_argument("--save", metavar="JSON", help="write results here")
    ap.add_argument("--compare", metavar="JSON",
                    help="compare against an earlier --save")
    ap.add_argument("--compare-stat", choices=("min_s", "median_s"),
                    default="min_s",
                    help="statistic to compare (min is the least noisy)")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="slowdown fraction counted as a regression")
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="exit 1 if --compare finds a regression")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    suites = [s for s in ALL_SUITES if s in (args.suite or ALL_SUITES)]
    bench = Bench(args, PRESETS[args.preset])
    printer = Printer(args.baseline)

    env = environment()
    print(f"torch-cuopt-lp benchmarks  preset={args.preset}  "
          f"suites={','.join(suites)}  tol={args.tol:g}  method={args.method}")
    print(f"python {env['python']}, torch {env['torch']}, "
          f"{env['cpu_count']} CPUs, GPU: {env['gpu'] or 'none'}")

    if any(s in E2E_SUITES for s in suites):
        t0 = time.perf_counter()
        try:
            bench.op = cl._ensure_op_loaded()
        except Exception as e:  # build/link problems surface here
            print(f"\ncould not load the cuopt_lp extension: {e}\n"
                  "Install it (CUOPT_ROOT=... pip install .) or set "
                  "CUOPT_ROOT for a JIT build (plus LD_LIBRARY_PATH for the "
                  "stub). '--suite overhead' runs without it.",
                  file=sys.stderr)
            return 2
        bench.load_s = time.perf_counter() - t0
        env.update(backend_info())
        print(f"libcuopt: {env['libcuopt'] or 'unknown'}")
        if env["stub"]:
            print("  !! TESTING STUB: a CPU PDHG toy. These timings check the "
                  "harness only and say nothing about cuOpt.")

    records: List[Record] = []
    for suite in suites:
        printer.header(suite, bench.subtitle(suite))
        for rec in getattr(bench, f"suite_{suite}")():
            printer.row(rec)
            records.append(rec)
            sys.stdout.flush()

    if args.save:
        save(args.save, env, args, records)
    if args.compare:
        regressed = compare(records, env, args.compare, args.threshold,
                            args.compare_stat)
        if regressed and args.fail_on_regression:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())