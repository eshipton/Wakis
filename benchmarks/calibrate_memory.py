"""
Measure the peak memory of Wakis simulations versus grid size to calibrate
`wakis.estimate`. Each point runs in a fresh subprocess (clean peak RSS).

Usage (on a cloud VM, not a laptop):
    python benchmarks/calibrate_memory.py --out mem_calibration.json [--gpu]
"""

import argparse
import json
import subprocess
import sys

CHILD = r"""
import json, resource, sys, time
import numpy as np
import wakis
N, bc, dtype, gpu, steps = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1", int(sys.argv[5])
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
grid = wakis.GridFIT3D(0, 1, 0, 1, 0, 1, N, N, N, verbose=0)
rss_grid = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
solver = wakis.SolverFIT3D(grid, bc_low=[bc] * 3, bc_high=[bc] * 3, use_gpu=gpu,
                           dtype=getattr(np, dtype), verbose=0)
t = time.time()
for _ in range(steps):
    solver.one_step()
dt_step = (time.time() - t) / steps
gpu_bytes = 0
if gpu:
    import cupy
    gpu_bytes = cupy.get_default_memory_pool().total_bytes()
print(json.dumps(dict(N=N, cells=grid.Nx * grid.Ny * grid.Nz, bc=bc, dtype=dtype, gpu=gpu,
                      rss0_kb=rss0, rss_grid_kb=rss_grid,
                      peak_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      gpu_bytes=gpu_bytes, step_s=dt_step)))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--sizes", default="16,32,48,64,96,128")
    a = ap.parse_args()
    rows = []
    for gpu in [False, True] if a.gpu else [False]:
        for bc in ("pec", "cpml"):
            for dtype in ("float64", "float32"):
                for N in map(int, a.sizes.split(",")):
                    out = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            CHILD,
                            str(N),
                            bc,
                            dtype,
                            "1" if gpu else "0",
                            "10",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    line = [x for x in out.stdout.splitlines() if x.startswith("{")]
                    if not line:
                        print(
                            "failed", gpu, bc, dtype, N, out.stderr[-400:], flush=True
                        )
                        continue
                    r = json.loads(line[-1])
                    rows.append(r)
                    print(json.dumps(r), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
