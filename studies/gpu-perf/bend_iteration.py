"""Case (f): one forward + adjoint (value_and_grad) iteration of the invrs-gym
full ceviche waveguide bend with the invrs-wakis driver (10 nm grid, r = 1,
'ceviche' port estimator, 6 wavelengths, CPML, mode source, DFT readouts,
exact adjoint) on the GPU.

  python bend_iteration.py [--fused 0|1] [--out res.json] [--nsys]

`--fused 1` makes every SolverFIT3D of the driver fused (the driver itself
is unchanged). Reports the wall time of a warm value_and_grad, the number of
time steps and the GPU utilization; with --fused 0 and 1 the losses and
gradients are compared by compare_bend.py.
"""

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench import _Util  # noqa: E402

NAME = "ceviche_waveguide_bend"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nsys", action="store_true")
    ap.add_argument("--run-time", type=float, default=None)
    a = ap.parse_args()

    import jax
    import wakis
    from invrs_gym import challenges

    import invrs_wakis.ceviche as cev
    from invrs_wakis.ceviche import SimOptions, wakis_ceviche_challenge

    from invrs_wakis import _design

    # invrs-wakis is pinned to an older Wakis; develop's AdjointSimulation also
    # asks designs for a dispersive-tensor network (none for pixel designs)
    for name in ("network_of", "dispersive_group"):
        if not hasattr(_design.PixelDesign, name):
            setattr(_design.PixelDesign, name, lambda self, solver: None)

    if a.fused:

        class FusedSolver(wakis.SolverFIT3D):
            def __init__(self, *args, **kw):
                kw.setdefault("fused", True)
                super().__init__(*args, **kw)

        cev.wakis.SolverFIT3D = FusedSolver

    opts = dict(resolution=1, port_estimator="ceviche", use_gpu=True)
    if a.run_time is not None:
        opts["run_time"] = a.run_time
    ch, _ = wakis_ceviche_challenge(NAME, SimOptions(**opts))
    gym = challenges.BY_NAME[NAME]()
    p = gym.component.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    x0 = np.clip(np.asarray(p.array, float) * 0.8 + 0.1
                 + 0.05 * rng.standard_normal(np.shape(p.array)), 0, 1)  # fmt: skip

    def loss(x):
        return ch.loss(ch.component.response(dataclasses.replace(p, array=x))[0])

    vg = jax.value_and_grad(loss)
    t0 = time.time()
    vg(x0)  # warm-up: simulation set-up, JAX tracing, kernel compilation
    warm = time.time() - t0
    sims = list(ch.component._sims.values())
    if a.nsys:
        import cupy as cp

        cp.cuda.profiler.start()
    with _Util(True) as util:
        t0 = time.time()
        value, grad = vg(x0)
        wall = time.time() - t0
    if a.nsys:
        cp.cuda.profiler.stop()
    Nt = int(sims[0].Nt)
    res = {
        "case": "bend",
        "fused": bool(a.fused),
        "n_sims": len(sims),
        "Nt": Nt,
        "cells": int(sims[0].geometry.make_grid().Nx * sims[0].geometry.make_grid().Ny),
        "first_call_s": warm,
        "value_and_grad_s": wall,
        "ms_per_step": 1e3 * wall / (2 * Nt * len(sims)),
        "gpu_util_pct": util.mean(),
        "loss": float(value),
    }
    print(json.dumps(res, indent=1))
    if a.out:
        res["grad"] = np.asarray(grad).ravel().tolist()
        pathlib.Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
