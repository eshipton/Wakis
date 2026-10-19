"""
Polygonal FIT for the Mie cylinder (n = 2, a = 1 um, x = 2..10): hybrid
square/triangle meshes and Voronoi-primal meshes vs the graded baselines of
``studies/mie-cylinder-graded``.

Arms (h_in = lambda_min,diel / res, h_out = n h_in, growth 1.2 as the
graded study):

* ``hybrid``          -- `hybrid_circles_mesh`: h_out squares in the bulk,
  h_in squares inside the cylinder, conforming Delaunay triangles only in
  the band around r = a; diagonal Hodge everywhere. Square domain
  |x|, |y| < 2.7 um (circles 1.9/2.2/2.5 um), split-field PML of 10 square
  cells outside it (same layout as the cart-graded arm: CPML from 2.7 um).
* ``hybrid-compact``  -- same, |x|, |y| < 1.7 um, circles 1.25/1.4/1.55 um.
* ``hybrid-disk``     -- the hybrid mesh on the disk r < 2.7 um of the
  triangle arms (thin h_out triangle annulus at the boundary, first-order
  ABC): same domain as ``tri-graded``, so the comparison isolates the mesh.
* ``hybrid-disk-compact`` -- ABC at 1.7 um (as ``tri-graded-compact``).
* ``voronoi``         -- `voronoi_disk_mesh`: Voronoi cells of staggered
  generator rings (spacing s = sqrt(3) h(r), i.e. Voronoi edges ~ h),
  mirrored generator pairs at r = a, first-order ABC at r = 2.7 um.
* ``voronoi-compact`` -- ABC at 1.7 um, circles 1.25/1.4/1.55 um.

Extraction/error metric identical to the baselines
(`wakis.cylinder_harmonics`, S-matrix de-embedding, 1000 fs window).

Usage::

    python polyfit_study.py list
    python polyfit_study.py run --shard I --nshards N --outdir DIR
    python polyfit_study.py run --only hybrid:TM:10 --outdir /tmp/x   # local smoke
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

A, N_CYL = S.A, S.N_CYL
RES = [10, 14, 20, 28, 40]
GROWTH = 1.2
N_PML = 10
GEOM = {
    "std": dict(radii=[1.9e-6, 2.2e-6, 2.5e-6], r_out=2.7e-6),
    "compact": dict(radii=[1.25e-6, 1.4e-6, 1.55e-6], r_out=1.7e-6),
}
ARMS = {
    "hybrid": ("hybrid", "std"),
    "hybrid-compact": ("hybrid", "compact"),
    "hybrid-disk": ("hybrid-disk", "std"),
    "hybrid-disk-compact": ("hybrid-disk", "compact"),
    "voronoi": ("voronoi", "std"),
    "voronoi-compact": ("voronoi", "compact"),
}
VOR_COLLAPSE = float(os.environ.get("VOR_COLLAPSE", 0.1))
VOR_LLOYD = int(os.environ.get("VOR_LLOYD", 10))


def tasks():
    out = []
    for res in RES[::-1]:
        for arm in ARMS:
            for pol in ("TM", "TE"):
                out.append((arm, pol, res))
    return out


def build(kind, geom, res):
    from wakis.unstructured.polymeshers import hybrid_circles_mesh, voronoi_disk_mesh

    g = GEOM[geom]
    h = S.LAM_MIN_DIEL / res
    if kind == "hybrid":
        mesh, info = hybrid_circles_mesh(
            [(0.0, 0.0, A)], h, N_CYL * h, g["r_out"], growth=GROWTH, pml_cells=N_PML
        )
        L = info["half_width"]
        pml = dict(x=(-L, L), y=(-L, L), thickness=N_PML * N_CYL * h, order=3, R0=1e-6)
        kw = dict(bc="pec", pml=pml, hodge="orthogonal")
    elif kind == "hybrid-disk":
        mesh, info = hybrid_circles_mesh(
            [(0.0, 0.0, A)], h, N_CYL * h, g["r_out"], growth=GROWTH,
            outer_radius=g["r_out"],
        )  # fmt: skip
        kw = dict(bc="abc", hodge="orthogonal")
    else:
        band = 2 * h
        s_in = np.sqrt(3) * h

        def size(r):
            d = max(abs(r - A) - band, 0.0)
            return s_in * min(N_CYL, GROWTH ** (d / h)) if r > A else s_in

        mesh = voronoi_disk_mesh([A, g["r_out"]], size, collapse=VOR_COLLAPSE, lloyd=VOR_LLOYD)
        info = {}
        kw = dict(bc="abc", hodge="orthogonal", ortho_tol=np.inf)
    return mesh, info, kw, h


def run(arm, pol, res, t_run):
    from wakis.cylinder_harmonics import circle_points
    from wakis.unstructured.polysolver2d import PolygonFIT2D

    kind, geom = ARMS[arm]
    S.RADII = np.array(GEOM[geom]["radii"])
    t0 = time.time()
    mesh, info, kw, h = build(kind, geom, res)
    t_mesh = time.time() - t0
    solver = PolygonFIT2D(mesh, {0: [N_CYL**2, 1.0]}, polarization=pol, cfl=S.CFL, **kw)
    solver.add_plane_wave(S.pulse())
    pts = np.vstack([circle_points(r, S.M_PTS)[0] for r in S.RADII])
    P = solver._interp(pts)
    t_setup = time.time() - t0
    dt = solver.dt
    shift = 0.0 if pol == "TM" else -0.5 * dt
    acc = np.zeros((len(S.FREQS), len(pts)), complex)
    n_max = int(t_run / dt)
    times = (np.arange(n_max) + 1) * dt + shift
    t1 = time.time()
    for n in range(n_max):
        solver.one_step()
        acc += np.outer(np.exp(-2j * np.pi * S.FREQS * times[n]), P @ solver.u) * dt
        if n % 2000 == 0 and not np.all(np.isfinite(solver.u)):
            raise FloatingPointError("unstable")
    t_run_s = time.time() - t1
    # pure field-update timing (no DFT), 200 steps
    t2 = time.time()
    n_t = min(200, n_max)
    for _ in range(n_t):
        solver.one_step()
    t_step = (time.time() - t2) / n_t
    u_inc0 = solver.incident_field(np.zeros((1, 2)), times)
    u0 = (np.exp(-2j * np.pi * np.outer(S.FREQS, times)) @ u_inc0) * dt
    out = S.extract(acc, u0, pol)
    d = mesh.diagnostics()
    out.update(
        dt=dt, steps=n_max, nodes=int(solver.free.size), dofs=solver.n_unknowns,
        cells=int(mesh.n_cells), cells_by_size=d["cells_by_size"],
        max_defect=d["max_orthogonality_defect"], setup_s=t_setup, mesh_s=t_mesh,
        run_s=t_run_s, step_s=t_step, h=h, info=info,
        area_err=float(mesh.cell_area[mesh.cell_tags == 0].sum() / (np.pi * A**2) - 1),
    )  # fmt: skip
    return out


def run_task(arm, pol, res, t_run):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    try:
        out = run(arm, pol, res, t_run)
    except Exception as e:
        import traceback

        out = dict(error=repr(e), tb=traceback.format_exc())
    out.update(arm=arm, pol=pol, res=res, wall_s=time.time() - t0, nproc=os.cpu_count())
    try:
        with open("/proc/cpuinfo") as f:
            out["cpu"] = next(line.split(":")[1].strip() for line in f if "model name" in line)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=str(HERE / "data"))
    ap.add_argument("--t-run", type=float, default=1000.0, help="fs")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    tl = tasks()
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    mine = (
        [(x.split(":")[0], x.split(":")[1], int(x.split(":")[2])) for x in a.only]
        if a.only
        else tl[a.shard :: a.nshards]
    )
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for arm, pol, res in mine:
        out = run_task(arm, pol, res, a.t_run * 1e-15)
        (Path(a.outdir) / f"{arm}_{pol}_{res}.json").write_text(json.dumps(out))
        if "error" in out:
            print(arm, pol, res, out["error"], out["tb"], flush=True)
            continue
        q, qr = np.array(out["q"]), np.array(out["q_ref"])
        print(
            arm, pol, res, f"rms {np.sqrt(np.mean((q / qr - 1) ** 2)):.4f}",
            f"dofs {out['dofs']} dt {out['dt'] * 1e18:.1f} as steps {out['steps']}",
            f"run {out['run_s']:.0f} s step {out['step_s'] / out['dofs'] * 1e9:.1f} ns/dof",
            f"refl {max(out['refl']):.3f}", flush=True,
        )  # fmt: skip


if __name__ == "__main__":
    main()
