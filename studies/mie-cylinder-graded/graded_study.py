"""
Graded meshes for the Mie cylinder (n = 2, a = 1 um, x = 2..10): can triangles
reach the same accuracy with fewer unknowns than the Cartesian grid?

Arms (target edge length h_in = lambda_min,diel / res inside the cylinder):

* ``tri-graded``       -- `graded_disk_mesh`: h_in for r < a + 2 h_in, then
  geometric growth (ratio 1.2 per h) to h_out = n h_in (equal cells per
  *local* wavelength), Voronoi Hodge; same geometry as the uniform study
  (circles 1.9/2.2/2.5 um, first-order ABC at 2.7 um).
* ``tri-graded-compact`` -- as above with circles 1.25/1.4/1.55 um and the
  ABC at 1.7 um (the scattered-field source needs no TF/SF box).
* ``cart-graded``      -- `SolverFIT3D`, conformal 'all', rectilinear grid
  with h_in on |x|, |y| < a + 2 h_in, growth 1.2 to h_out = n h_in; same
  geometry as the uniform study (TF/SF half width 1.25 um, circles
  1.9/2.2/2.5 um, CPML from 2.7 um).
* ``cart-graded-compact`` -- TF/SF half width 1.15 um, circles
  1.72/1.86/2.0 um, CPML from 2.12 um.
* ``tri-uniform`` / ``cart-uniform`` -- the uniform arms of
  ``studies/mie-cylinder-mesh`` (Voronoi; conformal 'all'), re-run here only
  for uncontended timings.

Extraction and error metric as in ``studies/mie-cylinder-mesh``
(`wakis.cylinder_harmonics`, S-matrix de-embedding).

Usage::

    python graded_study.py list                       # task table
    python graded_study.py run --shard I --nshards N --outdir DIR
    python graded_study.py plot                       # figures + summary

One task per shard (Cloud Batch, one run per small machine: uncontended
timings). Local runs only under ``bin/memguard``.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mie-cylinder-mesh"))
import mie_mesh_study as S  # noqa: E402

C0 = S.C0
A, N_CYL = S.A, S.N_CYL
RES = [10, 14, 20, 28, 40]
GROWTH = 1.2
GEOM = {
    "std": dict(radii=[1.9e-6, 2.2e-6, 2.5e-6], r_out=2.7e-6, tf_half=1.25e-6),
    "compact-tri": dict(radii=[1.25e-6, 1.4e-6, 1.55e-6], r_out=1.7e-6, tf_half=None),
    "compact-cart": dict(
        radii=[1.72e-6, 1.86e-6, 2.0e-6], r_out=2.12e-6, tf_half=1.15e-6
    ),
}
ARMS = {
    "tri-graded": ("tri", True, "std"),
    "tri-graded-compact": ("tri", True, "compact-tri"),
    "cart-graded": ("cart", True, "std"),
    "cart-graded-compact": ("cart", True, "compact-cart"),
    "tri-uniform": ("tri", False, "std"),
    "cart-uniform": ("cart", False, "std"),
}


def tasks():
    out = []
    for res in RES[::-1]:
        for arm in (
            "tri-graded",
            "tri-graded-compact",
            "cart-graded",
            "cart-graded-compact",
        ):
            for pol in ("TM", "TE"):
                out.append((arm, pol, res))
    for res in (20, 28):
        for arm in ("tri-uniform", "cart-uniform"):
            for pol in ("TM", "TE"):
                out.append((arm, pol, res))
    return out


def size_field(h_in, graded):
    if not graded:
        return lambda r: h_in
    band, h_out = 2 * h_in, N_CYL * h_in

    def size(r):
        d = r - A - band
        return h_in if d <= 0 else min(h_out, h_in * GROWTH ** (d / h_in))

    return size


def axis_nodes(h_in, graded, r_out, n_pml):
    """Symmetric node coordinates: fine core (h_in on |x| < a + 2 h_in),
    geometric growth to h_out = n h_in landing on r_out, then n_pml CPML
    cells. Uniform: h_in up to ceil(r_out / h_in) cells (as the uniform study)."""
    if graded:
        size = size_field(h_in, True)
        core = h_in * np.arange(int(np.ceil((A + 2 * h_in) / h_in - 1e-9)) + 1)
        xs = [core[-1]]
        while xs[-1] < r_out:
            xs.append(xs[-1] + size(xs[-1]))
        xs = np.array(xs)
        xs = core[-1] + (xs - core[-1]) * (r_out - core[-1]) / (xs[-1] - core[-1])
        x = np.unique(np.concatenate([[0.0], core, xs]))
    else:
        x = h_in * np.arange(int(np.ceil(r_out / h_in - 1e-9)) + 1)
    hp = x[-1] - x[-2]
    x = np.concatenate([x, x[-1] + hp * np.arange(1, n_pml + 1)])
    return np.concatenate([-x[:0:-1], x])


# --------------------------------------------------------------------------
def run_tri(graded, geom, pol, res, hodge="voronoi"):
    from wakis.cylinder_harmonics import circle_points
    from wakis.unstructured import UnstructuredFIT2D, disk_mesh, graded_disk_mesh

    g = GEOM[geom]
    S.RADII = np.array(g["radii"])
    t0 = time.time()
    h = S.LAM_MIN_DIEL / res
    if graded:
        mesh = graded_disk_mesh([A, g["r_out"]], size_field(h, True))
    else:
        mesh = disk_mesh([A, g["r_out"]], h)
    solver = UnstructuredFIT2D(
        mesh, {0: [N_CYL**2, 1.0]}, polarization=pol, bc="abc", hodge=hodge, cfl=S.CFL
    )
    src = S.pulse()
    solver.add_plane_wave(src)
    pts = np.vstack([circle_points(r, S.M_PTS)[0] for r in S.RADII])
    P = S.tri_interp(solver, pts, max(h, N_CYL * h if graded else h))
    t_setup = time.time() - t0
    dt = solver.dt
    shift = 0.0 if pol == "TM" else -0.5 * dt
    acc = np.zeros((len(S.FREQS), len(pts)), complex)
    n_max = int(S.T_RUN / dt)
    times = (np.arange(n_max) + 1) * dt + shift
    t1 = time.time()
    for n in range(n_max):
        solver.one_step()
        acc += np.outer(np.exp(-2j * np.pi * S.FREQS * times[n]), P @ solver.u) * dt
        if n % 2000 == 0 and not np.all(np.isfinite(solver.u)):
            raise FloatingPointError("unstable")
    t_run = time.time() - t1
    u_inc0 = solver.incident_field(np.zeros((1, 2)), times)
    u0 = (np.exp(-2j * np.pi * np.outer(S.FREQS, times)) @ u_inc0) * dt
    out = S.extract(acc, u0, pol)
    out.update(
        dt=dt, steps=n_max, nodes=int(solver.free.size),
        dofs=int(solver.free.size + mesh.n_edges), cells=int(mesh.n_cells),
        setup_s=t_setup, run_s=t_run, h=h,
    )  # fmt: skip
    return out


def run_cart(graded, geom, pol, res):
    import pyvista as pv

    import wakis
    from wakis.cylinder_harmonics import circle_points

    g = GEOM[geom]
    S.RADII = np.array(g["radii"])
    t0 = time.time()
    h = S.LAM_MIN_DIEL / res
    xn = axis_nodes(h, graded, g["r_out"], S.N_PML)
    dz = 10 * h
    cyl = pv.Cylinder(radius=A, height=10e-6, direction=(0, 0, 1), resolution=720)
    grid = wakis.GridFIT3D(
        x=xn, y=xn.copy(), z=np.array([-dz / 2, dz / 2]),
        stl_solids={"cyl": cyl.triangulate()}, stl_materials={"cyl": [N_CYL**2, 1.0]},
        conformal="all", verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml", "cpml", "periodic"], bc_high=["cpml", "cpml", "periodic"],
        n_pml=S.N_PML, use_stl=True, cfln=S.CFL, verbose=0,
    )  # fmt: skip
    src = wakis.TFSFSource(
        S.pulse(), size=(2 * g["tf_half"], 2 * g["tf_half"], None), direction="+x",
        polarization={"TM": "z", "TE": "y"}[pol], amplitude=1.0,
    )  # fmt: skip
    src.prepare(solver)
    gx, gy = np.asarray(grid.x), np.asarray(grid.y)
    Nx, Ny, N = solver.Nx, solver.Ny, solver.N
    if pol == "TM":
        xs, ys, field = gx[:Nx], gy[:Ny], "E"
    else:
        xs, ys, field = (
            0.5 * (gx[:Nx] + gx[1 : Nx + 1]),
            0.5 * (gy[:Ny] + gy[1 : Ny + 1]),
            "H",
        )
    off = 2 * N
    pts = np.vstack([circle_points(r, S.M_PTS)[0] for r in S.RADII])
    P = S.bilinear(xs, ys, pts, Nx)
    i0 = int(np.argmin(np.abs(gx[:Nx])))
    assert abs(gx[i0]) < 1e-9 * h
    t_setup = time.time() - t0
    dt = solver.dt
    shift = 0.0 if pol == "TM" else -0.5 * dt
    acc = np.zeros((len(S.FREQS), len(pts)), complex)
    line, times = [], []

    def cb(s, t):
        ts = t + dt + shift
        a = (s.E if field == "E" else s.H).array[off : off + Nx * Ny]
        acc[:] += np.outer(np.exp(-2j * np.pi * S.FREQS * ts), P @ a) * dt
        times.append(t + dt)
        line.append(float(np.asarray(src._e)[i0 + src._offset]))

    n_max = int(S.T_RUN / dt)
    t1 = time.time()
    solver.emsolve(n_max, source=[src], callback=cb)
    t_run = time.time() - t1
    if not np.all(np.isfinite(acc)):
        raise FloatingPointError("unstable")
    times, line = np.array(times), np.array(line)
    u0 = (np.exp(-2j * np.pi * np.outer(S.FREQS, times)) @ line) * dt
    if pol == "TE":
        u0 = u0 / np.sqrt(S.MU0 / S.EPS0)
    out = S.extract(acc, u0, pol)
    n2d = (len(xn) - 1) ** 2
    out.update(
        dt=dt, steps=int(solver.Nt), nodes=int(n2d), dofs=int(3 * n2d), cells=int(n2d),
        setup_s=t_setup, run_s=t_run, h=h, h_max=float(np.diff(xn).max()),
    )  # fmt: skip
    return out


def run_task(arm, pol, res):
    kind, graded, geom = ARMS[arm]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    try:
        if kind == "tri":
            out = run_tri(graded, geom, pol, res)
        else:
            out = run_cart(graded, geom, pol, res)
    except Exception as e:
        import traceback

        out = dict(error=repr(e), tb=traceback.format_exc())
    out.update(arm=arm, pol=pol, res=res, wall_s=time.time() - t0, nproc=os.cpu_count())
    try:
        import platform

        out["cpu"] = platform.processor() or platform.machine()
        with open("/proc/cpuinfo") as f:
            out["cpu"] = next(
                line.split(":")[1].strip() for line in f if "model name" in line
            )
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "plot"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=str(HERE / "data"))
    ap.add_argument("--t-run", type=float, default=1000.0, help="fs")
    ap.add_argument("--only", nargs="*", help="arm:pol:res tasks (local tests)")
    a = ap.parse_args()
    S.T_RUN = a.t_run * 1e-15
    tl = tasks()
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    if a.mode == "plot":
        from plot_graded import main_plot

        main_plot()
        return
    if a.only:
        mine = [
            (x.split(":")[0], x.split(":")[1], int(x.split(":")[2])) for x in a.only
        ]
    else:
        mine = tl[a.shard :: a.nshards]
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for arm, pol, res in mine:
        out = run_task(arm, pol, res)
        name = Path(a.outdir) / f"{arm}_{pol}_{res}.json"
        name.write_text(json.dumps(out))
        msg = (
            out.get("error")
            or f"rms {np.sqrt(np.mean((np.array(out['q']) / np.array(out['q_ref']) - 1) ** 2)):.4f}"
        )
        print(
            arm,
            pol,
            res,
            msg,
            f"dofs {out.get('dofs')} steps {out.get('steps')} run {out.get('run_s', 0):.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
