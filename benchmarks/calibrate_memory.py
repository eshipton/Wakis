"""
Measure peak memory and run time of Wakis simulations versus problem size to
calibrate `wakis.estimate`. Each point runs in a fresh subprocess (clean peak
RSS, ``ru_maxrss``).

Suites:

* ``plain``: vacuum box of N^3 cells, PEC or CPML walls, float64/float32,
  CPU and (``--gpu``) GPU (the original calibration);
* ``conformal``: dielectric sphere (eps_r = 4, radius 0.3 m in a 1 m box) with
  ``conformal='all'`` and ``'tensor'``, and a PEC sphere with
  ``conformal='pec'``; peak RSS after the setup and after the steps, setup
  time, and the number of interface cells (cells holding an E edge whose
  dual cell or length is cut by the surface);
* ``dispersive``: vacuum box with a dispersive box (one Drude pole plus
  Lorentz poles) filling a fraction of the domain
  (``add_dispersive_material``), modes ``fraction:n_poles``;
* ``unstructured``: `ball_mesh` + `UnstructuredFIT3D` (time stepping).

Step times are means over warmed-up steps (3 steps first). Usage (cloud VM)::

    python benchmarks/calibrate_memory.py --suite plain --out plain.json [--gpu]
    python benchmarks/calibrate_memory.py --suite conformal --sizes 32,64,96 --out conf.json
"""

import argparse
import json
import subprocess
import sys

CHILD = r"""
import json, resource, sys, time
import numpy as np
import wakis

suite, N, bc, dtype, gpu, steps, mode = (sys.argv[1], int(sys.argv[2]), sys.argv[3],
    sys.argv[4], sys.argv[5] == "1", int(sys.argv[6]), sys.argv[7])

def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # bytes

def rss_now():
    with open("/proc/self/statm") as f:  # Linux: current resident set
        return int(f.read().split()[1]) * resource.getpagesize()

def gpu_bytes():
    if not gpu:
        return 0
    import cupy
    return int(cupy.get_default_memory_pool().total_bytes())

def timed_steps(step, n_steps):
    for _ in range(3):
        step()
    if gpu:
        import cupy
        cupy.cuda.Device().synchronize()
    t = time.perf_counter()
    for _ in range(n_steps):
        step()
    if gpu:
        cupy.cuda.Device().synchronize()
    return (time.perf_counter() - t) / n_steps

out = dict(suite=suite, N=N, bc=bc, dtype=dtype, gpu=gpu, mode=mode)
rss0 = rss()
if suite == "unstructured":
    from wakis.unstructured import UnstructuredFIT3D, ball_mesh
    t0 = time.perf_counter()
    mesh = ball_mesh([0.5, 1.0], 1.0 / N)
    solver = UnstructuredFIT3D(mesh, materials={0: [2.0, 1.0], 1: [1.0, 1.0]})
    setup_s = time.perf_counter() - t0
    peak_setup = rss()
    solver.add_point_dipole((0.1, 0.05, 0.02), (0, 0, 1), lambda t: np.sin(1e9 * t))
    step_s = timed_steps(solver.one_step, steps)
    out.update(cells=int(mesh.n_cells), setup_s=setup_s, step_s=step_s,
               rss0=rss0, peak_setup=peak_setup, peak=rss())
    print(json.dumps(out))
    sys.exit()

kw = {}
if suite == "conformal":
    import pyvista as pv
    c = (0.5071, 0.4957, 0.5029)
    sphere = pv.Sphere(radius=0.3, center=c, theta_resolution=120, phi_resolution=120)
    if mode == "pec":
        kw = dict(stl_solids={"s": sphere}, stl_materials={"s": "pec"}, conformal="pec")
    else:
        kw = dict(stl_solids={"s": sphere}, stl_materials={"s": [4.0, 1.0, 0.0]},
                  conformal=mode)
t0 = time.perf_counter()
grid = wakis.GridFIT3D(0, 1, 0, 1, 0, 1, N, N, N, verbose=0, **kw)
rss_grid = rss()
t1 = time.perf_counter()
solver = wakis.SolverFIT3D(grid, bc_low=[bc] * 3, bc_high=[bc] * 3, use_gpu=gpu,
                           use_stl=bool(kw), dtype=getattr(np, dtype), verbose=0)
t2 = time.perf_counter()
peak_setup = rss()
gpu_setup = gpu_bytes()
interface = 0
if suite == "conformal":
    geo = next(iter(grid._conformal_geometry.values()))
    cut = np.zeros(geo.shape, dtype=bool)
    for c in range(3):
        for f in geo.fractions.values():
            cut |= (f[c] > 1e-9) & (f[c] < 1 - 1e-9)
        L = solver.grid.L.to_matrix("xyz"[c])
        part = (geo.pec_length[c] > 1e-9 * L) & (geo.pec_length[c] < (1 - 1e-9) * L)
        cut |= part
    interface = int(cut.sum())
disp_edges = 0
if suite == "dispersive":
    from wakis.dispersive import Drude, DispersiveMaterial, Lorentz
    frac, n_poles = (float(v) for v in mode.split(":"))
    side = frac ** (1 / 3)
    poles = [Drude(2 * np.pi * 1e9, 1e8)] + [
        Lorentz(0.5, 2 * np.pi * (2 + k) * 1e9, 1e8) for k in range(int(n_poles) - 1)
    ]
    mat = DispersiveMaterial(eps_inf=1.0, poles=poles)
    solver.add_dispersive_material(mat, center=(0.5, 0.5, 0.5), size=(side,) * 3)
    disp_edges = int(sum(g["idx_host"].size for g in solver._dispersive_groups))
step_s = timed_steps(solver.one_step, steps)
out.update(cells=int(grid.Nx * grid.Ny * grid.Nz), grid_s=t1 - t0, solver_s=t2 - t1,
           setup_s=t2 - t0, step_s=step_s, rss0=rss0, rss_grid=rss_grid, rss_now=rss_now(),
           peak_setup=peak_setup, peak=rss(), gpu_setup=gpu_setup, gpu_bytes=gpu_bytes(),
           interface_cells=interface, dispersive_edges=disp_edges,
           conformal_dt=float(solver.dt))
print(json.dumps(out))
"""


def run(args):
    out = subprocess.run(
        [sys.executable, "-c", CHILD, *map(str, args)], capture_output=True, text=True
    )
    line = [x for x in out.stdout.splitlines() if x.startswith("{")]
    if not line:
        print("failed", args, out.stderr[-600:], flush=True)
        return None
    r = json.loads(line[-1])
    print(json.dumps(r), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="plain",
                    choices=["plain", "conformal", "dispersive", "unstructured"])  # fmt: skip
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--sizes", default="16,32,48,64,96,128")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--modes", default=None, help="conformal modes / Drude fractions")
    a = ap.parse_args()
    sizes = list(map(int, a.sizes.split(",")))
    rows = []
    backends = [False, True] if a.gpu else [False]
    for gpu in backends:
        g = "1" if gpu else "0"
        if a.suite == "plain":
            for bc in ("pec", "cpml"):
                for dtype in ("float64", "float32"):
                    for N in sizes:
                        rows.append(run(["plain", N, bc, dtype, g, a.steps, "-"]))
        elif a.suite == "conformal":
            for mode in (a.modes or "pec,all,tensor").split(","):
                for N in sizes:
                    rows.append(
                        run(["conformal", N, "pec", "float64", g, a.steps, mode])
                    )
        elif a.suite == "dispersive":
            for frac in (a.modes or "0.1:1,0.5:1,0.5:3").split(","):
                for N in sizes:
                    rows.append(
                        run(["dispersive", N, "pec", "float64", g, a.steps, frac])
                    )
        elif a.suite == "unstructured" and not gpu:
            for N in sizes:
                rows.append(
                    run(["unstructured", N, "pec", "float64", "0", a.steps, "-"])
                )
    json.dump([r for r in rows if r], open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
