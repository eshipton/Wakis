"""Per-step cost of the Wakis time loop: wall time, GPU utilization and a
per-component breakdown.

  python bench.py --case cpml3d --gpu [--fused 1] [--steps 500] [--out x.json]
  python bench.py --case adjoint2d --gpu [--fused 1]
  nsys profile --capture-range=cudaProfilerApi -o prof python bench.py ... --nsys

Measurements (after `--warmup` steps):
  ms_per_step      unsynchronized loop (what a user sees), one sync at the end
  gpu_util_pct     NVML utilization sampled every 20 ms during that loop
  breakdown_ms     same loop with a device sync after every component
                   (sources / step / post-steps / monitors): wall time of each
                   component when serialized (host + device)
  --nsys           cudaProfilerStart/Stop around `--steps` unsynced steps, for
                   kernel and launch counts (nsys stats)
"""

import argparse
import json
import os
import pathlib
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cases  # noqa: E402


def _sync(use_gpu):
    if use_gpu:
        import cupy as cp

        cp.cuda.Device().synchronize()


class _Util:
    """Sample NVML GPU utilization in a background thread."""

    def __init__(self, use_gpu):
        self.samples, self.on = [], use_gpu
        if use_gpu:
            import pynvml

            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nv = pynvml

    def __enter__(self):
        self.stop = False
        if self.on:
            self.th = threading.Thread(target=self._run, daemon=True)
            self.th.start()
        return self

    def _run(self):
        while not self.stop:
            self.samples.append(self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu)
            time.sleep(0.02)

    def __exit__(self, *a):
        self.stop = True
        if self.on:
            self.th.join()

    def mean(self):
        # The first samples cover the time before the loop
        s = self.samples[2:] or self.samples
        return float(np.mean(s)) if s else None


def loop(solver, sources, monitors, n0, n1, sync=None):
    """emsolve's loop body for steps n0..n1-1; with `sync`, time each part."""
    post = [s.post_step for s in sources if hasattr(s, "post_step")]
    acc = {"sources": 0.0, "step": 0.0, "post": 0.0, "monitors": 0.0}
    for n in range(n0, n1):
        t = n * solver.dt
        parts = (
            ("sources", lambda: [s.update(solver, t) for s in sources]),
            ("step", solver.one_step),
            ("post", lambda: [p(solver, t) for p in post]),
            ("monitors", lambda: [m.update(solver, t) for m in monitors]),
        )
        for key, fn in parts:
            if sync is None:
                fn()
            else:
                t0 = time.perf_counter()
                fn()
                sync()
                acc[key] += time.perf_counter() - t0
    return acc


EMSOLVE = False


def run_case(name, use_gpu, fused, steps, warmup, nsys):
    t0 = time.time()
    solver, sources, monitors = cases.CASES[name](use_gpu, fused)
    setup = time.time() - t0
    n = 0
    loop(solver, sources, monitors, n, n + warmup)
    n += warmup
    _sync(use_gpu)
    if nsys:
        import cupy as cp

        cp.cuda.profiler.start()
        if EMSOLVE:
            solver.emsolve(steps, source=sources, monitors=monitors)
        else:
            loop(solver, sources, monitors, n, n + steps)
        _sync(use_gpu)
        cp.cuda.profiler.stop()
        return {"case": name, "nsys_steps": steps}
    with _Util(use_gpu) as util:
        t0 = time.perf_counter()
        if EMSOLVE:
            # emsolve restarts at t = 0; fused solvers replay CUDA graphs
            solver.emsolve(steps, source=sources, monitors=monitors)
        else:
            loop(solver, sources, monitors, n, n + steps)
        _sync(use_gpu)
        wall = time.perf_counter() - t0
    n += steps
    nb = max(steps // 4, 20)
    acc = loop(solver, sources, monitors, n, n + nb, sync=lambda: _sync(use_gpu))
    E = solver.E.array
    E = E.get() if hasattr(E, "get") else E
    return {
        "case": name,
        "backend": "gpu" if use_gpu else "cpu",
        "fused": fused,
        "emsolve": EMSOLVE,
        "cells": int(solver.N),
        "steps": steps,
        "setup_s": setup,
        "ms_per_step": 1e3 * wall / steps,
        "gpu_util_pct": util.mean(),
        "breakdown_ms": {k: 1e3 * v / nb for k, v in acc.items()},
        "checksum_E": float(np.sum(np.abs(E))),
    }


def run_adjoint(use_gpu, fused, nsys):
    sim = cases.adjoint2d(use_gpu, fused)
    sim.output_shapes()
    rho = np.random.default_rng(0).uniform(size=sim.design.shape)
    out = sim.forward(rho)  # warm-up (kernel compilation)
    G = {k: {c: np.ones_like(v) for c, v in d.items()} for k, d in out.items()}
    sim.vjp(G)
    if nsys:
        import cupy as cp

        cp.cuda.profiler.start()
    with _Util(use_gpu) as uf:
        t0 = time.perf_counter()
        out = sim.forward(rho)
        _sync(use_gpu)
        t_fwd = time.perf_counter() - t0
    with _Util(use_gpu) as ub:
        t0 = time.perf_counter()
        grad = sim.vjp(G)
        _sync(use_gpu)
        t_adj = time.perf_counter() - t0
    if nsys:
        cp.cuda.profiler.stop()
    return {
        "case": "adjoint2d",
        "backend": "gpu" if use_gpu else "cpu",
        "fused": fused,
        "Nt": sim.Nt,
        "forward_ms_per_step": 1e3 * t_fwd / sim.Nt,
        "adjoint_ms_per_step": 1e3 * t_adj / sim.Nt,
        "gpu_util_pct_forward": uf.mean(),
        "gpu_util_pct_adjoint": ub.mean(),
        "out_checksum": float(sum(np.abs(v).sum() for d in out.values() for v in d.values())),
        "grad": np.asarray(grad).ravel().tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--fused", type=int, default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--nsys", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emsolve", action="store_true",
                    help="time solver.emsolve (CUDA graphs when fused)")
    a = ap.parse_args()
    global EMSOLVE
    EMSOLVE = a.emsolve
    fused = None if a.fused is None else bool(a.fused)
    if a.case == "adjoint2d":
        res = run_adjoint(a.gpu, fused, a.nsys)
    else:
        res = run_case(a.case, a.gpu, fused, a.steps, a.warmup, a.nsys)
    res["omp_threads"] = os.environ.get("OMP_NUM_THREADS")
    print(json.dumps({k: v for k, v in res.items() if k != "grad"}, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
