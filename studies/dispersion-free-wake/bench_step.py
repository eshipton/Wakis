"""
Cost per time step of the Yee and TE/TM steps on the same grid.

PEC box (dx = dy = 2 dz), fields randomised, 10 warm-up steps then the
median of `--steps` timed `one_step` calls (GPU: synchronised). Both schemes
use the same sparse backend (SciPy on CPU without sparse_dot_mkl, cuSPARSE
on the GPU). Writes one JSON line per (scheme, size) to --out.

Usage: python bench_step.py --sizes 64x64x256 128x128x512 [--gpu] --out f.jsonl
"""

import argparse
import json
import time

import numpy as np

import wakis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--cpml", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.gpu:
        import cupy as cp
    for size in args.sizes:
        nx, ny, nz = (int(v) for v in size.split("x"))
        dz = 1e-3
        for scheme in ("yee", "tete"):
            grid = wakis.GridFIT3D(0, nx * 2 * dz, 0, ny * 2 * dz, 0, nz * dz,
                                   nx, ny, nz, verbose=0)  # fmt: skip
            bz = "cpml" if args.cpml else "pec"
            s = wakis.SolverFIT3D(grid, bc_low=["pec", "pec", bz],
                                  bc_high=["pec", "pec", bz], scheme=scheme,
                                  use_gpu=args.gpu, verbose=0, cfln=0.99)  # fmt: skip
            s.one_step()
            xp = s.E.xp
            rng = np.random.default_rng(0)
            s.E.fromarray(xp.asarray(rng.standard_normal(s.E.array.size) * 1e-3))
            for _ in range(10):
                s.one_step()
            ts = []
            for _ in range(args.steps):
                if args.gpu:
                    cp.cuda.Device().synchronize()
                t0 = time.perf_counter()
                s.one_step()
                if args.gpu:
                    cp.cuda.Device().synchronize()
                ts.append(time.perf_counter() - t0)
            rec = dict(scheme=scheme, shape=[nx, ny, nz], cells=nx * ny * nz,
                       gpu=args.gpu, cpml=args.cpml, dt=float(s.dt),
                       t_step=float(np.median(ts)),
                       t_step_min=float(np.min(ts)))  # fmt: skip
            print(json.dumps(rec), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            del s, grid
            if args.gpu:
                cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
