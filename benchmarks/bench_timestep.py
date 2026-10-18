"""
Forward and adjoint time per step and memory of `SolverFIT3D`.

A PEC box of n^3 cells (1 m side, vacuum) driven by a `CurrentSource` with a
`DFTMonitor` plane and a `FluxMonitor` plane (3 frequencies each), for each
size n:

* forward: seconds per step of the `emsolve` loop (source update, one_step,
  monitor updates) and of the bare `one_step`;
* adjoint: seconds per `AdjointStepper.step` (bare transposed step) and per
  step of a full `AdjointSimulation` (forward with stored design increments,
  then `vjp` with monitor adjoint sources), for a 0.2 m design cube;
* memory: resident host memory (CPU, summed over MPI ranks) or CuPy pool
  memory (GPU) after building the solver and after the adjoint run.

Usage (prints one JSON line per size on rank 0)::

    python benchmarks/bench_timestep.py --sizes 64 128 192
    python benchmarks/bench_timestep.py --sizes 64 128 192 --gpu
    mpiexec -n 4 python benchmarks/bench_timestep.py --sizes 128 --mpi
"""

import argparse
import json
import os
import time

import numpy as np

import wakis
from wakis.inverse import AdjointSimulation, AdjointStepper, MaterialGrid
from wakis.sources import CurrentSource, GaussianSource


def _comm(use_mpi):
    if not use_mpi:
        return None
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _rss():
    """Current resident set size of this process [bytes] (Linux; else peak)."""
    if not os.path.exists("/proc/self/statm"):
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # macOS: bytes
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _memory(use_gpu, comm):
    """Host RSS summed over ranks, or the CuPy pool bytes in use [GB]."""
    if use_gpu:
        import cupy as cp

        cp.cuda.Device().synchronize()
        return cp.get_default_memory_pool().used_bytes() / 1e9
    rss = _rss()
    return (comm.allreduce(rss) if comm else rss) / 1e9


def _sync(use_gpu, comm):
    if use_gpu:
        import cupy as cp

        cp.cuda.Device().synchronize()
    if comm:
        comm.Barrier()


def _timed(fn, n, use_gpu, comm):
    """Seconds per call of fn, slowest rank."""
    _sync(use_gpu, comm)
    t0 = time.perf_counter()
    for i in range(n):
        fn(i)
    _sync(use_gpu, comm)
    dt = (time.perf_counter() - t0) / n
    if comm is None:
        return dt
    from mpi4py import MPI

    return comm.allreduce(dt, op=MPI.MAX)


def _make_solver(n, use_gpu, use_mpi, cpml):
    kw = {"use_mpi": True} if use_mpi else {}
    grid = wakis.GridFIT3D(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, n, n, n, verbose=0, **kw)
    bc = ["cpml"] * 3 if cpml else ["pec"] * 3
    return wakis.SolverFIT3D(
        grid, bc_low=bc, bc_high=bc, n_pml=8, use_gpu=use_gpu, verbose=0, **kw
    )


def _setup(n):
    f0 = 0.3e9 * n / 64  # ~16 cells per wavelength at every size
    source = CurrentSource(
        GaussianSource(f0, f0 / 3), component="z", center=(-0.1, 0, 0)
    )
    freqs = [0.9 * f0, f0, 1.1 * f0]
    monitors = {
        "plane": wakis.DFTMonitor(
            freqs, components=["Ez", "Hy"], center=(0.2, 0, 0), size=(0, 0.8, 0.8)
        ),  # fmt: skip
        "flux": wakis.FluxMonitor(freqs, center=(0.3, 0, 0), size=(0, 0.8, 0.8)),
    }
    return source, monitors


def bench(n, steps, use_gpu, use_mpi, cpml, full, adjoint=True):
    comm = _comm(use_mpi)
    rank = comm.Get_rank() if comm else 0
    cells = n**3
    out = {"n": n, "cells": cells, "gpu": use_gpu, "cpml": cpml,
           "ranks": comm.Get_size() if comm else 1}  # fmt: skip

    t0 = time.perf_counter()
    solver = _make_solver(n, use_gpu, use_mpi, cpml)
    out["build_s"] = time.perf_counter() - t0
    out["mem_solver_GB"] = _memory(use_gpu, comm)

    source, monitors = _setup(n)
    mons = list(monitors.values())

    def loop_step(i):
        t = i * solver.dt
        source.update(solver, t)
        solver.one_step()
        for m in mons:
            m.update(solver, t)

    loop_step(0)  # first-step initialization
    out["fwd_loop_s"] = _timed(loop_step, steps, use_gpu, comm)
    out["fwd_step_s"] = _timed(lambda i: solver.one_step(), steps, use_gpu, comm)
    out["Mcells_s_fwd"] = cells / out["fwd_loop_s"] / 1e6
    out["Mcells_s_step"] = cells / out["fwd_step_s"] / 1e6
    if not adjoint:
        if rank == 0:
            print(json.dumps(out), flush=True)
        return out

    stepper = AdjointStepper(solver)
    lam = stepper.zero_state()
    rng = np.random.default_rng(rank)
    lam["E"][:] = stepper.xp.asarray(rng.standard_normal(lam["E"].shape))
    state = {"lam": lam}

    def adj_step(i):
        state["lam"] = stepper.step(state["lam"])

    out["adj_step_s"] = _timed(adj_step, steps, use_gpu, comm)
    out["mem_adjoint_GB"] = _memory(use_gpu, comm)
    del stepper, state, lam, solver

    if full:
        design = MaterialGrid(center=(0, 0, 0), size=(0.2, 0.2, 0.2), shape=(5, 5, 5))
        build = []

        def make_solver():
            t0 = time.perf_counter()
            solver = _make_solver(n, use_gpu, use_mpi, cpml)
            build.append(time.perf_counter() - t0)
            return solver

        source, monitors = _setup(n)  # fresh (uninitialized) copies
        sim = AdjointSimulation(make_solver, [source], monitors, design, steps)
        rho = np.full(design.shape, 0.5)
        _sync(use_gpu, comm)
        t0 = time.perf_counter()
        sim.forward(rho)
        _sync(use_gpu, comm)
        t_fwd = time.perf_counter() - t0 - build[-1]
        G = {"flux": {k: np.ones(s) for k, s in sim.output_shapes()["flux"].items()}}
        _sync(use_gpu, comm)
        t0 = time.perf_counter()
        sim.vjp(G)
        _sync(use_gpu, comm)
        t_vjp = time.perf_counter() - t0
        out["sim_fwd_step_s"] = t_fwd / steps
        out["sim_vjp_step_s"] = t_vjp / steps
        out["design_edges"] = int(
            comm.allreduce(design.flat_idx.size) if comm else design.flat_idx.size
        )
        out["mem_sim_GB"] = _memory(use_gpu, comm)

    out["adj_over_fwd"] = out["adj_step_s"] / out["fwd_step_s"]
    if full:
        out["sim_vjp_over_fwd"] = out["sim_vjp_step_s"] / out["sim_fwd_step_s"]
    if rank == 0:
        print(json.dumps(out), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--cpml", action="store_true")
    parser.add_argument("--no-full", action="store_true", help="skip AdjointSimulation")
    parser.add_argument("--no-adjoint", action="store_true", help="forward only")
    args = parser.parse_args()
    for n in args.sizes:
        bench(n, args.steps, args.gpu, args.mpi, args.cpml, not args.no_full,
              not args.no_adjoint)  # fmt: skip


if __name__ == "__main__":
    main()
