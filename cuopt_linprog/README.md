# torch-cuopt-lp — batched LP solving in PyTorch via NVIDIA cuOpt

A custom PyTorch C++ operator (`torch.ops.cuopt_lp.solve_batch`) wrapping
NVIDIA cuOpt's LP solver (Apache-2.0, https://github.com/NVIDIA/cuopt),
plus a `scipy.optimize.linprog`-style Python front end.

```python
from cuopt_linprog import cuopt_batch_linprog

res = cuopt_batch_linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=(0, 1), mode="stacked")
res.x        # (*batch, n)
res.fun      # (*batch,)
res.status   # scipy codes: 0 optimal, 1 limit, 2 infeasible, 3 unbounded, 4 numerical
res.slack, res.con, res.ineqlin_marginals, res.eqlin_marginals, res.solve_time
```

c / b_ub / b_eq (and optionally dense A_ub / A_eq) may carry broadcastable
leading batch dimensions. A matrices may be dense `(m, n)`, dense batched
`(*batch, m, n)`, or an unbatched `torch.sparse_csr` tensor. cuOpt handles
general variable bounds and ranged constraints natively, so no
standard-form conversion happens anywhere.

## Files

| file | purpose |
|---|---|
| `cuopt_lp_op.cpp` | the C++ op: tensors -> cuOpt CSR ranged problems -> `cuOptSolve` loop (GIL released), statuses/duals/reduced costs back as tensors |
| `cuopt_linprog.py` | scipy-style wrapper, CSR assembly, loop/stacked batching, extension loader |
| `setup.py` | `CUOPT_ROOT=/path/to/cuopt pip install .` |
| `test_cuopt_linprog.py` | validation against scipy (HiGHS) |
| `testing_stub/` | a fake `libcuopt` (tiny CPU PDHG solver + header) so the extension can be compiled and numerically tested on machines without a GPU. **Never deploy against this.** |

## Batching strategies

cuOpt's C API is per-problem, so batch throughput comes from one of:

- **`mode="loop"`** — one `cuOptSolve` per problem, looped in C++.
  Per-problem statuses (cuOpt's own rigorous infeasibility/unboundedness
  detection identifies bad lanes individually) and full per-problem
  tolerances. Per-solve overhead is a few ms, so this suits dozens to
  hundreds of medium LPs, or whenever you need reliable per-lane statuses.

- **`mode="stacked"`** — all B problems fused into one block-diagonal
  sparse LP, solved in a single call. Since the problems share nothing,
  the fused optimum decomposes exactly into per-problem optima, and the
  GPU sees one large sparse LP — the regime PDLP/barrier are built for.
  Best for thousands of small LPs. Caveats: termination criteria apply to
  the fused problem (per-constraint residual mode is enabled to keep
  per-lane accuracy honest, but very heterogeneous objective scales can
  still see uneven relative accuracy); the returned status is global — if
  any lane is infeasible the fused LP is infeasible and *no* lane gets
  solved (the result message tells you to rerun in loop mode to find the
  culprits); one hard lane slows the collective solve.

A pragmatic pattern: solve with `mode="stacked"`, and if the fused status
is not optimal, fall back to `mode="loop"` for diagnosis.

## Building against real cuOpt

1. Install cuOpt (needs an NVIDIA GPU; see cuOpt's system requirements):
   `pip install nvidia-cuopt-cu12`, conda, or a source build.
2. Point at it and build:
   ```bash
   CUOPT_ROOT=/path/containing/include+lib pip install .
   ```
   or JIT-build at import time via `cuopt_linprog.load_extension()`.
3. Run the tests (tighten `RTOL` in the test file to ~1e-7 — the loose
   default accommodates the low-accuracy testing stub).

The extension assumes a cuOpt build with 64-bit floats and 32-bit ints
(the default) and checks this at runtime.

## Notes and limitations

- Data marshals through host memory: cuOpt's C API ingests host arrays
  and manages GPU transfer itself, so GPU-resident torch tensors are
  copied to CPU first. This is not a zero-copy GPU pipeline — the win is
  cuOpt's solver quality and its GPU-parallel solve, not tensor residency.
- `method=` selects cuOpt's algorithm: `"concurrent"` (default: PDLP +
  GPU barrier + CPU dual simplex race), `"pdlp"`, `"barrier"`,
  `"dual_simplex"`.
- Not differentiable (LP solutions are piecewise-constant in c anyway).
- LPs only — no integer variables (cuOpt's MILP solver could be wired the
  same way if needed).
- In stacked mode, dimensions of the fused problem must fit cuOpt's
  32-bit index range; the wrapper checks and tells you to split the batch.
- This code was validated end-to-end against scipy through the testing
  stub; the cuOpt calls follow the documented 26.x C API but have not
  been executed against a real GPU build here — expect at most minor
  header-version adjustments.
