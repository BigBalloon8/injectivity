"""End-to-end validation of the cuopt_lp custom op + wrapper against scipy.

Run against the testing stub (CPU PDHG) when no GPU/cuOpt is available:

    CUOPT_ROOT=testing_stub LD_LIBRARY_PATH=testing_stub/lib \
        python test_cuopt_linprog.py

Against real cuOpt, just point CUOPT_ROOT at the installation. The stub
solves to lower accuracy than real cuOpt, so tolerances here are loose;
tighten to ~1e-7 when running against the real library.
"""
import numpy as np
import torch
from scipy.optimize import linprog

from cuopt_linprog import cuopt_batch_linprog

rng = np.random.default_rng(0)
RTOL = 2e-4          # stub PDHG accuracy; use ~1e-7 with real cuOpt
FEAS = 1e-4


def t64(a):
    return None if a is None else torch.as_tensor(a, dtype=torch.float64)


def check(res, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
          label="", rtol=RTOL):
    B = c.shape[0]
    x, fun = res.x.numpy(), res.fun.numpy()
    for i in range(B):
        Ai = A_ub if A_ub is None or A_ub.ndim == 2 else A_ub[i]
        bi = None if b_ub is None else (b_ub if b_ub.ndim == 1 else b_ub[i])
        Aei = A_eq if A_eq is None or A_eq.ndim == 2 else A_eq[i]
        bei = None if b_eq is None else (b_eq if b_eq.ndim == 1 else b_eq[i])
        ref = linprog(c[i], Ai, bi, Aei, bei, bounds, method="highs")
        assert ref.status == 0, f"[{label}] scipy failed lane {i}"
        assert bool(res.success[i]), \
            f"[{label}] lane {i} status={int(res.status[i])}"
        err = abs(fun[i] - ref.fun) / (1 + abs(ref.fun))
        assert err < rtol, f"[{label}] lane {i}: {fun[i]:.8g} vs scipy " \
                           f"{ref.fun:.8g} (rel {err:.1e})"
        if Ai is not None:
            assert (Ai @ x[i] - bi).max() < FEAS, f"[{label}] ineq viol {i}"
        if Aei is not None:
            assert np.abs(Aei @ x[i] - bei).max() < FEAS, \
                f"[{label}] eq viol {i}"
    print(f"  [{label}] {B}/{B} match scipy")


def feasible_lp(B, m, n, batched_A=False):
    A = rng.standard_normal((B, m, n) if batched_A else (m, n))
    x0 = rng.uniform(0.5, 2, (B, n))
    if batched_A:
        b = np.einsum("bmn,bn->bm", A, x0) + rng.uniform(0.1, 1, (B, m))
    else:
        b = x0 @ A.T + rng.uniform(0.1, 1, (B, m))
    return rng.standard_normal((B, n)), A, b


for mode in ("loop", "stacked"):
    print(f"===== mode = {mode} =====")

    print(f"== 1. box-bounded, shared A ==")
    c, A, b = feasible_lp(12, 8, 6)
    res = cuopt_batch_linprog(t64(c), A_ub=t64(A), b_ub=t64(b),
                              bounds=(0, 3), mode=mode, tol=1e-6)
    check(res, c, A, b, bounds=(0, 3), label=f"{mode}/box", rtol=RTOL if mode == "loop" else 5e-3)

    print(f"== 2. mixed bounds + equalities ==")
    n = 5
    bounds = [(0, None), (-4, 4), (None, 3), (None, None), (1, None)]
    A = rng.standard_normal((7, n))
    x0 = rng.uniform(0.2, 1.5, (8, n))
    x0[:, 4] = rng.uniform(1.1, 1.8, 8)   # respect lb = 1
    b = x0 @ A.T + rng.uniform(0.1, 1, (8, 7))
    Ae = rng.standard_normal((2, n))
    be = x0 @ Ae.T
    c = rng.standard_normal((8, n))
    res = cuopt_batch_linprog(t64(c), t64(A), t64(b), t64(Ae), t64(be),
                              bounds=bounds, mode=mode, tol=1e-6)
    check(res, c, A, b, Ae, be, bounds=bounds, label=f"{mode}/mixed", rtol=RTOL if mode == "loop" else 5e-3)

    print(f"== 3. batched A (different matrix per problem) ==")
    c, A, b = feasible_lp(6, 7, 5, batched_A=True)
    res = cuopt_batch_linprog(t64(c), A_ub=t64(A), b_ub=t64(b),
                              bounds=(0, 2.5), mode=mode, tol=1e-6)
    check(res, c, A, b, bounds=(0, 2.5), label=f"{mode}/batched-A", rtol=RTOL if mode == "loop" else 5e-3)

    print(f"== 4. sparse CSR shared A ==")
    dense = rng.standard_normal((9, 6)) * (rng.random((9, 6)) < 0.5)
    dense[np.abs(dense).sum(1) == 0, 0] = 1.0   # no empty rows
    x0 = rng.uniform(0.5, 2, (5, 6))
    b = x0 @ dense.T + rng.uniform(0.1, 1, (5, 9))
    c = rng.standard_normal((5, 6))
    A_csr = t64(dense).to_sparse_csr()
    res = cuopt_batch_linprog(t64(c), A_ub=A_csr, b_ub=t64(b),
                              bounds=(0, 3), mode=mode, tol=1e-6)
    check(res, c, dense, b, bounds=(0, 3), label=f"{mode}/sparse", rtol=RTOL if mode == "loop" else 5e-3)

    print(f"== 5. slack/con/marginal fields ==")
    assert res.slack is not None and res.slack.shape == (5, 9)
    assert res.slack.min() > -FEAS
    assert res.ineqlin_marginals.shape == (5, 9)
    assert res.solve_time.shape == (5,)
    print("  fields verified")

print("===== per-problem statuses (loop mode only) =====")
print("== 6. infeasible lanes identified individually ==")
c = np.array([[1.0, 1.0]] * 3)
A = np.array([[1.0, 0.0]])
b = np.array([[2.0], [-1.0], [5.0]])       # lane 1 infeasible (x0 >= 0)
res = cuopt_batch_linprog(t64(c), A_ub=t64(A), b_ub=t64(b), mode="loop")
assert bool(res.success[0]) and bool(res.success[2])
assert not bool(res.success[1]) and int(res.status[1]) in (2, 4)
print(f"  statuses: {res.status.tolist()}")

print("== 7. unbounded flagged ==")
c = np.array([[-1.0, 0.0]] * 2)
A = np.array([[0.0, 1.0]])
b = np.array([[1.0]] * 2)
res = cuopt_batch_linprog(t64(c), A_ub=t64(A), b_ub=t64(b), mode="loop")
assert not bool(res.success.any())
print(f"  statuses: {res.status.tolist()}")

print("== 8. stacked mode reports fused status with guidance ==")
c = np.array([[1.0, 1.0]] * 3)
b = np.array([[2.0], [-1.0], [5.0]])
res = cuopt_batch_linprog(t64(c), A_ub=t64(np.array([[1.0, 0.0]])),
                          b_ub=t64(b), mode="stacked")
assert not bool(res.success.any())          # global infeasibility
assert "loop" in res.message
print(f"  statuses: {res.status.tolist()} — {res.message}")

print("== 9. multi-dim batch shape ==")
c, A, b = feasible_lp(12, 6, 4)
res = cuopt_batch_linprog(t64(c).reshape(3, 4, 4), A_ub=t64(A),
                          b_ub=t64(b).reshape(3, 4, 6), bounds=(0, 3),
                          mode="loop")
assert res.x.shape == (3, 4, 4) and res.fun.shape == (3, 4)
assert bool(res.success.all())
print("  shapes verified")

print("== 10. unbatched scipy-doc example ==")
c1 = np.array([-1.0, 4.0])
A1 = np.array([[-3.0, 1.0], [1.0, 2.0]])
b1 = np.array([6.0, 4.0])
res = cuopt_batch_linprog(t64(c1), A_ub=t64(A1), b_ub=t64(b1),
                          bounds=[(None, None), (-3, None)])
ref = linprog(c1, A1, b1, bounds=[(None, None), (-3, None)], method="highs")
assert res.x.shape == (2,) and abs(float(res.fun) - ref.fun) < 1e-3
print(f"  fun={float(res.fun):.6f} vs scipy {ref.fun:.6f}")

print("\nALL TESTS PASSED")
