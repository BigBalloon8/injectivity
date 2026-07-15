"""
test_gpu_execution.py — verify the cuOpt-backed solver actually runs on the GPU.

cuOpt's C API is host-side (you hand it host arrays and it manages device
transfer internally), so there is no tensor .device to inspect. The reliable
way to verify GPU execution is to watch the GPU itself while a solve runs:

  1. this process must acquire GPU memory / a CUDA compute context during
     the solve (it holds none beforehand), and
  2. GPU utilization must rise above idle while the solve is in flight, and
  3. the answers must still match scipy.

The test forces method="pdlp" and then method="barrier". This matters:
cuOpt's default "concurrent" mode races PDLP (GPU), barrier (GPU), and dual
simplex (CPU), and for small LPs the CPU simplex often wins the race — a
perfectly correct result that never touched the GPU. Forcing a GPU method
removes that ambiguity. It also catches accidentally linking the CPU
testing stub, which will fail check (1) immediately.

Run:  python test_gpu_execution.py
Requires an NVIDIA GPU with driver + NVML (bundled with the driver;
`pip install nvidia-ml-py` for the Python bindings, or nvidia-smi is used
as a fallback).
"""

import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from cuopt_linprog import cuopt_batch_linprog

PID = os.getpid()
SOLVE_SECONDS = 3.0        # keep solving at least this long while sampling
UTIL_THRESHOLD = 5         # % GPU utilization that must be observed
RTOL = 1e-6                # objective agreement vs scipy


# ---------------------------------------------------------------------------
# GPU monitoring: NVML if available, nvidia-smi fallback
# ---------------------------------------------------------------------------

class GpuMonitor:
    """Samples (our-PID-has-GPU-memory, utilization%) in a background thread."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.pid_seen = False
        self.pid_mem_bytes = 0
        self.max_util = 0
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        self._handles = []
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(count)]
        except Exception:
            self._nvml = None   # fall back to nvidia-smi

    @staticmethod
    def gpu_present():
        try:
            import pynvml
            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetCount() > 0
        except Exception:
            pass
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                 text=True, timeout=10)
            return out.returncode == 0 and "GPU" in out.stdout
        except Exception:
            return False

    def _sample_nvml(self):
        pynvml = self._nvml
        for h in self._handles:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                self.max_util = max(self.max_util, int(util))
            except Exception:
                pass
            for getter in ("nvmlDeviceGetComputeRunningProcesses",
                           "nvmlDeviceGetGraphicsRunningProcesses"):
                try:
                    for p in getattr(pynvml, getter)(h):
                        if p.pid == PID:
                            self.pid_seen = True
                            if p.usedGpuMemory:
                                self.pid_mem_bytes = max(self.pid_mem_bytes,
                                                         int(p.usedGpuMemory))
                except Exception:
                    pass

    def _sample_smi(self):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            for line in out.stdout.strip().splitlines():
                parts = [s.strip() for s in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit() \
                        and int(parts[0]) == PID:
                    self.pid_seen = True
                    if parts[1].isdigit():
                        self.pid_mem_bytes = max(self.pid_mem_bytes,
                                                 int(parts[1]) * 1024 * 1024)
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            for line in out.stdout.strip().splitlines():
                if line.strip().isdigit():
                    self.max_util = max(self.max_util, int(line.strip()))
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            if self._nvml is not None:
                self._sample_nvml()
            else:
                self._sample_smi()
            time.sleep(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# workload: big enough that the GPU solve is clearly observable
# ---------------------------------------------------------------------------

def make_workload(seed=0, B=512, m=40, n=30):
    """A stacked batch of feasible box-bounded LPs -> one fused sparse LP
    with ~B*m*n/2 nonzeros, sized so PDLP runs for an observable interval."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.5)
    A[np.abs(A).sum(1) == 0, 0] = 1.0
    x0 = rng.uniform(0.5, 2.0, (B, n))
    b = x0 @ A.T + rng.uniform(0.1, 1.0, (B, m))
    c = rng.standard_normal((B, n))
    return c, A, b


def reference_funs(c, A, b, bounds, k=8):
    """scipy answers for the first k lanes (spot check, not all B)."""
    from scipy.optimize import linprog
    refs = []
    for i in range(k):
        r = linprog(c[i], A, b[i], bounds=bounds, method="highs")
        assert r.status == 0
        refs.append(r.fun)
    return np.array(refs)


def main():
    if not GpuMonitor.gpu_present():
        print("SKIP: no NVIDIA GPU / driver detected on this machine.\n"
              "This test must run on the GPU box where cuOpt is installed.")
        sys.exit(0)

    c, A, b = make_workload()
    bounds = (0, 3)
    tc = torch.as_tensor(c, dtype=torch.float64)
    tA = torch.as_tensor(A, dtype=torch.float64)
    tb = torch.as_tensor(b, dtype=torch.float64)
    refs = reference_funs(c, A, b, bounds)

    failures = []
    for method in ("pdlp", "barrier"):
        print(f"== method={method!r}: solving under GPU monitoring ==")
        with GpuMonitor() as mon:
            t0 = time.perf_counter()
            n_solves = 0
            res = None
            # keep the GPU busy long enough to be sampled reliably
            while time.perf_counter() - t0 < SOLVE_SECONDS or n_solves == 0:
                res = cuopt_batch_linprog(
                    tc, A_ub=tA, b_ub=tb, bounds=bounds,
                    mode="stacked", method=method, tol=1e-8)
                n_solves += 1
            elapsed = time.perf_counter() - t0
        print(f"   {n_solves} stacked solve(s) of {c.shape[0]} LPs "
              f"in {elapsed:.1f}s "
              f"(cuOpt-reported solve time {float(res.solve_time[0]):.3f}s)")

        # 1. correctness spot-check vs scipy
        funs = res.fun.numpy()
        ok = bool(res.success.all())
        err = np.abs(funs[:len(refs)] - refs) / (1 + np.abs(refs))
        print(f"   success: {int(res.success.sum())}/{len(funs)}, "
              f"max rel err vs scipy (first {len(refs)}): {err.max():.2e}")
        if not ok or err.max() > RTOL:
            failures.append(f"{method}: wrong/failed solutions")

        # 2. this process must have held a CUDA compute context
        mem_mb = mon.pid_mem_bytes / 1e6
        print(f"   PID {PID} seen on GPU: {mon.pid_seen} "
              f"(process GPU memory observed: {mem_mb:.0f} MB)")
        if not mon.pid_seen:
            failures.append(
                f"{method}: this process never appeared on the GPU — the "
                "solve ran on CPU. Check that libcuopt is the real GPU "
                "build (NOT testing_stub) and that a compatible CUDA "
                "driver is installed.")

        # 3. the GPU must have actually done work
        print(f"   peak GPU utilization sampled: {mon.max_util}%")
        if mon.max_util < UTIL_THRESHOLD:
            failures.append(
                f"{method}: GPU utilization never exceeded "
                f"{UTIL_THRESHOLD}% during the solve. If the machine is "
                "otherwise busy/idle-capped, inspect manually with "
                "`watch nvidia-smi` while rerunning.")

    print()
    if failures:
        print("GPU EXECUTION TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("GPU EXECUTION TEST PASSED: cuOpt solves ran on the GPU with "
          "correct results.")


if __name__ == "__main__":
    main()