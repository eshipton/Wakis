"""
Time and peak memory of the conformal setup (grid + `SolverFIT3D`
construction) versus grid size, with a phase breakdown from cProfile.

Cases (all in a 0.2 m box, n^3 cells):

* ``sphere_pec``: vacuum sphere cavity in a PEC background, ``conformal='all'``;
* ``sphere_tensor``: dielectric sphere (eps_r = 4) in vacuum, ``conformal='tensor'``;
* ``box``: rotated PEC box, ``conformal='pec'``;
* ``mixed``: tilted dielectric cylinder, PEC ball and a thin sheet,
  ``conformal='all'``.

Each point runs in a fresh subprocess (clean peak RSS). Usage (cloud VM)::

    python benchmarks/bench_conformal_setup.py --cases sphere_tensor --sizes 64,100,128
    python benchmarks/bench_conformal_setup.py --cases mixed --sizes 100 --profile 30
"""

import argparse
import json
import subprocess
import sys

CHILD = r"""
import cProfile, json, pstats, resource, sys, time
import numpy as np
import pyvista as pv
import wakis
from wakis.conformal import rectangle_sheet

case, n, top, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
h = 0.2 / n
lo, hi = -0.1, 0.1
res = 200

def geometry(case):
    if case == "sphere_pec":
        s = pv.Sphere(radius=0.083, center=(0.0071, -0.0043, 0.0029),
                      theta_resolution=res, phi_resolution=res)
        return dict(stl_solids={"cav": s}, stl_materials={"cav": "vacuum"},
                    conformal="all"), "pec"
    if case == "sphere_tensor":
        s = pv.Sphere(radius=0.06, center=(0.0071, -0.0043, 0.0029),
                      theta_resolution=res, phi_resolution=res)
        return dict(stl_solids={"core": s}, stl_materials={"core": [4.0, 1.0, 0.0]},
                    conformal="tensor"), [1.0, 1.0, 0.0]
    if case == "box":
        b = pv.Box(bounds=(-0.05, 0.06, -0.04, 0.05, -0.06, 0.03)).triangulate()
        b = b.rotate_z(17.0, inplace=False).rotate_x(11.0, inplace=False)
        return dict(stl_solids={"box": b}, stl_materials={"box": "pec"},
                    conformal="pec"), [1.0, 1.0, 0.0]
    if case == "mixed":
        ball = pv.Sphere(radius=0.043, center=(0.021, -0.013, 0.008),
                         theta_resolution=res, phi_resolution=res)
        rod = pv.Cylinder(center=(-0.05, 0.03, 0), direction=(1, 1, 0.3),
                          radius=0.02, height=0.07, resolution=res).triangulate()
        zs = lo + h * round((-0.05 - lo) / h)  # sheet on a grid node
        sheet = rectangle_sheet((0.0, 0.0, zs), (0.083, 0.061, 0.0))
        return dict(stl_solids={"ball": ball, "rod": rod},
                    stl_materials={"ball": "pec", "rod": [3.0, 1.0, 0.0]},
                    sheets={"plate": sheet}, conformal="all"), [1.0, 1.0, 0.0]
    raise ValueError(case)

kw, bg = geometry(case)
prof = cProfile.Profile()
t0 = time.perf_counter()
prof.enable()
grid = wakis.GridFIT3D(lo, hi, lo, hi, lo, hi, n, n, n, verbose=0, **kw)
t1 = time.perf_counter()
solver = wakis.SolverFIT3D(grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, bg=bg,
                           use_stl=True, verbose=0)
t2 = time.perf_counter()
prof.disable()
rss_setup = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
for _ in range(2):
    solver.one_step()
t3 = time.perf_counter()
for _ in range(steps):
    solver.one_step()
step_s = (time.perf_counter() - t3) / max(steps, 1)
st = pstats.Stats(prof)
phases = {}
names = {"ConformalGeometry.__init__": ("conformal.py", "__init__"),
         "_dual_fractions": ("conformal.py", "_dual_fractions"),
         "_metrics": ("conformal.py", "_metrics"),
         "_interface_normals": ("conformal.py", "_interface_normals"),
         "_redistribution": ("conformal.py", "_redistribution"),
         "_material_tensors": ("conformal.py", "_material_tensors"),
         "_sheet_edge_factors": ("conformal.py", "_sheet_edge_factors"),
         "apply_conformal_materials": ("conformal.py", "apply_conformal_materials"),
         "apply_conformal_timestep": ("conformal.py", "apply_conformal_timestep"),
         "max_stable_dt(conformal)": ("conformal.py", "max_stable_dt"),
         "max_stable_dt(tensor)": ("tensor_averaging.py", "max_stable_dt"),
         "TensorOperator.assemble": ("tensor_averaging.py", "assemble"),
         "_apply_stl_materials": ("solverFIT3D.py", "_apply_stl_materials")}
for key, (fname, func) in names.items():
    for (f, line, fn), stat in st.stats.items():
        if f.endswith(fname) and fn == func:
            if fname == "conformal.py" and func == "__init__" and line > 420:
                continue
            phases[key] = phases.get(key, 0.0) + stat[3]
if top:
    st.sort_stats("cumulative").print_stats(top)
    st.sort_stats("tottime").print_stats(top)
print(json.dumps(dict(case=case, n=n, cells=grid.Nx * grid.Ny * grid.Nz,
                      grid_s=t1 - t0, solver_s=t2 - t1, setup_s=t2 - t0,
                      phases=phases, peak_setup_kb=rss_setup,
                      peak_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      step_s=step_s, dt=float(solver.dt))))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="sphere_pec,sphere_tensor,box,mixed")
    ap.add_argument("--sizes", default="32,64,100")
    ap.add_argument("--profile", type=int, default=0, help="print top-N functions")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for case in a.cases.split(","):
        for n in map(int, a.sizes.split(",")):
            out = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    CHILD,
                    case,
                    str(n),
                    str(a.profile),
                    str(a.steps),
                ],
                capture_output=True,
                text=True,
            )
            if a.profile:
                print(out.stdout)
            line = [x for x in out.stdout.splitlines() if x.startswith("{")]
            if not line:
                print("failed", case, n, out.stderr[-2000:], flush=True)
                continue
            r = json.loads(line[-1])
            rows.append(r)
            print(json.dumps(r), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
