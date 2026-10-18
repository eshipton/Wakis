"""
Profile the forward and adjoint time loops of an `AdjointSimulation`.

Runs `forward` and `vjp` of the benchmark case of `bench_timestep.py` (PEC or
CPML box of n^3 cells, current source, DFT and flux monitor planes, 0.2 m
design cube) under cProfile and prints the most expensive functions. On the
GPU, CUDA kernels run asynchronously, so cProfile attributes their time to
the first call that synchronizes; `--gpu` therefore also prints the
per-call time of the main kernels measured with `cupyx.profiler.benchmark`.

Usage::

    python benchmarks/profile_step.py --n 64 --steps 50 [--gpu] [--cpml]
"""

import argparse
import cProfile
import pstats

import numpy as np
from bench_timestep import _make_solver, _setup

from wakis.inverse import AdjointSimulation, MaterialGrid


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpml", action="store_true")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    source, monitors = _setup(args.n)
    design = MaterialGrid(center=(0, 0, 0), size=(0.2, 0.2, 0.2), shape=(5, 5, 5))
    sim = AdjointSimulation(
        lambda: _make_solver(args.n, args.gpu, False, args.cpml),
        [source], monitors, design, args.steps,
    )  # fmt: skip
    G = {"flux": {k: np.ones(s) for k, s in sim.output_shapes()["flux"].items()}}
    rho = np.full(design.shape, 0.5)

    for name, call in (
        ("forward", lambda: sim.forward(rho)),
        ("vjp", lambda: sim.vjp(G)),
    ):
        profile = cProfile.Profile()
        profile.enable()
        call()
        if args.gpu:
            import cupy as cp

            cp.cuda.Device().synchronize()
        profile.disable()
        print(f"===== {name}: n = {args.n}, {args.steps} steps")
        pstats.Stats(profile).sort_stats("tottime").print_stats(args.top)

    if args.gpu:
        from cupyx.profiler import benchmark

        solver = sim._solver
        E, H = solver.E.array, solver.H.array
        A_csr = solver.tDsiDmuiDaC.tocsr()
        for label, fn in (
            ("A @ E (csc)", lambda: solver.tDsiDmuiDaC @ E),
            ("B @ H (csc)", lambda: solver.itDaiDepsDstC @ H),
            ("A @ E (csr)", lambda: A_csr @ E),
            ("axpy H -= dt*x", lambda: H.__isub__(solver.dt * E)),
        ):
            print(label, benchmark(fn, n_repeat=20))


if __name__ == "__main__":
    main()
