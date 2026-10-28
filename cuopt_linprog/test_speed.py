"""Speed and scaling tests for torch-cuopt-lp.

    # wrapper-only tests; run anywhere, no cuOpt or compiled extension needed
    python -m pytest test_speed.py -v

    # + GIL test, against the CPU testing stub
    CUOPT_ROOT=testing_stub LD_LIBRARY_PATH=testing_stub/lib \\
        python -m pytest test_speed.py -v

    # + the README's performance claims, against real cuOpt
    CUOPT_ROOT=/path/to/cuopt python -m pytest test_speed.py -v

Most tests check how time *scales* (the ratio between a small and a large
input) rather than absolute times, so they hold across machines: an
accidental O(n^2) shows up as a ratio far above linear. Knobs for slow or
noisy machines:

    CUOPT_SPEED_SLACK          allowed factor over linear scaling (default 2.5)
    CUOPT_SPEED_MAX_US_PER_LP  wrapper budget per LP at large batch (default 20)

To skip everything that runs loop mode (every test id containing "loop";
the rest use stacked mode only):

    python -m pytest test_speed.py -k "not loop"

For timing tables and regression tracking use bench_cuopt_linprog.py.
"""
import os
import threading
import time

import pytest
import torch

import cuopt_linprog as cl
from bench_cuopt_linprog import (OpProbe, as_equalities, backend_info,
                                 make_lp, measure, null_op, use_op)

pytestmark = pytest.mark.filterwarnings("ignore:.*[Ss]parse.*:UserWarning")

SLACK = float(os.environ.get("CUOPT_SPEED_SLACK", "2.5"))
MAX_US_PER_LP = float(os.environ.get("CUOPT_SPEED_MAX_US_PER_LP", "20"))


def best_of(fn, repeat=5, budget=30.0):
    samples, result = measure(fn, warmup=1, repeat=repeat, budget=budget)
    return min(samples), result


def wrapper_seconds(lp, mode="stacked"):
    """Best-of-5 wall time of cuopt_batch_linprog with the solver no-op'd."""
    with use_op(null_op):
        return best_of(lambda: lp.solve(mode=mode))[0]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def op():
    try:
        return cl._ensure_op_loaded()
    except Exception as e:  # not built / libcuopt not found
        pytest.skip(f"cuopt_lp extension not available ({type(e).__name__}); "
                    "set CUOPT_ROOT or pip install .")


@pytest.fixture(scope="module")
def real_cuopt(op):
    info = backend_info()
    if info["stub"] is not False:
        pytest.skip(f"needs real cuOpt (loaded: {info['libcuopt']}, "
                    f"stub={info['stub']})")
    return op


# ---------------------------------------------------------------------------
# Python wrapper cost (solver replaced by a no-op)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["loop", "stacked"])
def test_wrapper_overhead_linear_in_batch(mode):
    small, large = 2048, 16384
    t_small = wrapper_seconds(make_lp(small, 20, 10), mode)
    t_large = wrapper_seconds(make_lp(large, 20, 10), mode)
    limit = (large / small) * SLACK
    assert t_large / t_small < limit, (
        f"{mode}: wrapper time grew {t_large / t_small:.1f}x for "
        f"{large // small}x the batch (limit {limit:.0f}x): "
        f"{t_small * 1e3:.2f}ms -> {t_large * 1e3:.2f}ms")


@pytest.mark.parametrize("mode", ["loop", "stacked"])
def test_wrapper_overhead_per_lp(mode):
    B = 16384
    us = wrapper_seconds(make_lp(B, 20, 10), mode) / B * 1e6
    assert us < MAX_US_PER_LP, (
        f"{mode}: {us:.1f}µs of wrapper time per LP at B={B} "
        f"(budget {MAX_US_PER_LP}µs, CUOPT_SPEED_MAX_US_PER_LP)")


@pytest.mark.parametrize("batched_A", [False, True],
                         ids=["shared_A", "batched_A"])
def test_stacked_wrapper_close_to_loop(batched_A):
    """Block-diagonal stacking is index arithmetic; it shouldn't cost more
    than a small multiple of the loop-mode preparation."""
    lp = make_lp(8192, 20, 10, batched_A=batched_A)
    t_loop = wrapper_seconds(lp, "loop")
    t_stacked = wrapper_seconds(lp, "stacked")
    assert t_stacked < 3 * SLACK * t_loop, (
        f"stacked prep {t_stacked * 1e3:.2f}ms vs loop {t_loop * 1e3:.2f}ms")


@pytest.mark.parametrize("rows", ["A_ub", "A_eq"])
def test_sparse_overhead_scales_with_nnz(rows):
    """Sparse LPs with 5 nonzeros per row, m = n/2. Going from n=1000 to
    n=8000 is 8x the nonzeros but 64x the dense m*n, so a wrapper that
    stays sparse grows ~8x and one that densifies A grows ~64x."""
    def lp(n):
        p = make_lp(4, n // 2, n, sparse=True, nnz_per_row=5)
        return as_equalities(p) if rows == "A_eq" else p

    t_small, t_large = wrapper_seconds(lp(1000)), wrapper_seconds(lp(8000))
    limit = 8 * SLACK
    assert t_large / t_small < limit, (
        f"sparse {rows}: wrapper time grew {t_large / t_small:.0f}x for 8x "
        f"the nonzeros (limit {limit:.0f}x; 64x would mean O(m*n)): "
        f"{t_small * 1e3:.1f}ms -> {t_large * 1e3:.1f}ms")


# ---------------------------------------------------------------------------
# the compiled op (stub or real cuOpt)
# ---------------------------------------------------------------------------

def _counter_rate(occupy_main_thread):
    """Increments/second a Python thread manages while the main thread runs
    `occupy_main_thread`."""
    count = 0
    stop = threading.Event()

    def spin():
        nonlocal count
        while not stop.is_set():
            count += 1

    th = threading.Thread(target=spin)
    th.start()
    t0 = time.perf_counter()
    occupy_main_thread()
    elapsed = time.perf_counter() - t0
    stop.set()
    th.join()
    return count / elapsed


def test_solve_releases_gil(op):
    """The README says the solve loop runs with the GIL released. If it
    didn't, another Python thread would stall for the whole solve."""
    opts = dict(mode="stacked", method="pdlp", tol=1e-4)
    B = 4
    while True:   # grow the batch until one call takes >= 0.3 s
        lp = make_lp(B, 40, 20)
        t0 = time.perf_counter()
        lp.solve(**opts)
        dt = time.perf_counter() - t0
        if dt >= 0.3 or B >= 8192:
            break
        B *= 2
    idle = _counter_rate(lambda: time.sleep(dt))
    busy = _counter_rate(lambda: lp.solve(**opts))
    assert busy > 0.15 * idle, (
        f"a Python thread ran at {100 * busy / idle:.0f}% of its idle rate "
        f"during a {dt:.2f}s solve: the GIL looks held")


# ---------------------------------------------------------------------------
# performance claims that only mean something on real cuOpt + GPU
# ---------------------------------------------------------------------------

def test_stacked_beats_loop_for_many_small_lps(real_cuopt):
    """README: stacked mode is the right choice for thousands of small LPs."""
    lp = make_lp(1024, 20, 10)
    best = {}
    for mode in ("loop", "stacked"):
        best[mode], res = best_of(lambda: lp.solve(mode=mode), repeat=3,
                                  budget=60)
        assert bool(res.success.all()), f"{mode}: {res.message}"
    assert best["stacked"] < best["loop"], (
        f"B=1024 small LPs: stacked {best['stacked']:.3f}s vs loop "
        f"{best['loop']:.3f}s")


def test_wrapper_small_next_to_solve(real_cuopt):
    """For medium LPs the Python side should be noise next to cuOpt."""
    lp = make_lp(64, 200, 200)
    probe = OpProbe(real_cuopt)
    with use_op(probe):
        samples, res = measure(lambda: lp.solve(mode="stacked"), warmup=1,
                               repeat=3, budget=60)
    assert bool(res.success.all()), res.message
    k = len(samples)
    py = [w - o for w, o in zip(samples, probe.op_s[-k:])]
    share = min(py) / min(samples)
    assert share < 0.10, f"Python wrapper is {100 * share:.0f}% of wall time"


def test_cuda_inputs_add_little(real_cuopt):
    """GPU-resident inputs are copied to host; that copy should be small."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible to torch")
    from bench_cuopt_linprog import to_device
    lp = make_lp(64, 50, 50)
    lp_gpu = to_device(lp, "cuda")
    t_cpu = best_of(lambda: lp.solve(mode="stacked"), repeat=3)[0]

    def gpu_call():
        r = lp_gpu.solve(mode="stacked")
        torch.cuda.synchronize()
        return r

    t_gpu = best_of(gpu_call, repeat=3)[0]
    assert t_gpu < 1.25 * t_cpu + 0.005, (
        f"CUDA inputs {t_gpu * 1e3:.1f}ms vs CPU inputs {t_cpu * 1e3:.1f}ms")