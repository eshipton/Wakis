"""
Subgridded Cartesian FIT for the Mie cylinder (n = 2, a = 1 um, x = 2..10),
the shared mesh benchmark B1, against the graded baselines of
``studies/mie-cylinder-graded``.

Mesh: a uniform coarse grid with h_out = n h_in (equal cells per *local*
wavelength, as the graded arms) and one 2:1 refined level on the coarse cells
within a + 2 h_in of the axis (the same fine band as the graded arms), so
h_in = lambda_min,diel / res inside the cylinder. Solver: `SubgridFIT2D`
(symmetric hanging-node interface, sub-pixel averaging, split-field PML on
the coarse grid, scattered-field plane wave: no TF/SF box).

Arms:

* ``sg``          -- circles 1.9/2.2/2.5 um, PML from 2.7 um (as the
  ``*-graded`` arms);
* ``sg-compact``  -- circles 1.25/1.4/1.55 um, PML from 1.7 um (as
  ``tri-graded-compact``; no TF/SF box limits it);
* ``sg-lts`` / ``sg-compact-lts`` -- the same with two-level local time
  stepping (p = 2, `wakis.unstructured.lts`): the fine *vacuum* band takes
  two sub-steps, the dielectric and the coarse grid one.

Extraction and error metric identical to the graded study
(`wakis.cylinder_harmonics`, S-matrix de-embedding, RMS of Q_sca/Q_exact - 1).

Usage::

    python subgrid_study.py list
    python subgrid_study.py run --shard I --nshards N --outdir DIR
    python subgrid_study.py run --only sg-compact:TM:10      # local smoke test
    python subgrid_study.py plot
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
N_PML = 10
GEOM = {
    "std": dict(radii=[1.9e-6, 2.2e-6, 2.5e-6], r_out=2.7e-6),
    "compact": dict(radii=[1.25e-6, 1.4e-6, 1.55e-6], r_out=1.7e-6),
}
ARMS = {
    "sg": ("std", None),
    "sg-compact": ("compact", None),
    "sg-lts": ("std", 2),
    "sg-compact-lts": ("compact", 2),
}


def tasks():
    out = []
    for res in RES[::-1]:
        for arm in ARMS:
            for pol in ("TM", "TE"):
                out.append((arm, pol, res))
    return out


def build(geom, pol, res, lts=None, band=2.0, n_sub=8, material=None):
    from wakis.subgrid import Circle, Geometry, QuadMesh, SubgridFIT2D

    g = GEOM[geom]
    h = S.LAM_MIN_DIEL / res
    H = N_CYL * h  # coarse step: 2:1 ratio for n = 2
    m_in = int(np.ceil(g["r_out"] / H - 1e-9))
    M = m_in + N_PML
    xc = H * np.arange(-M, M + 1)
    r_ref = A + band * h

    def refine(level, x0, x1, y0, y1):
        dx = np.maximum(np.maximum(x0, -x1), 0)  # distance of the cell to 0
        dy = np.maximum(np.maximum(y0, -y1), 0)
        return np.hypot(dx, dy) < r_ref

    mesh = QuadMesh(xc, xc.copy(), ratio=2, max_level=1, refine=refine)
    geo = Geometry([Circle((0, 0), A, N_CYL**2 if material is None else material)])
    box = (-m_in * H, m_in * H, -m_in * H, m_in * H)
    solver = SubgridFIT2D(mesh, geo, polarization=pol, pml=dict(box=box),
                          cfl=S.CFL, n_sub=n_sub, lts=lts)  # fmt: skip
    return mesh, solver, h


def run_sg(geom, pol, res, lts=None, material=None):
    from wakis.cylinder_harmonics import circle_points

    g = GEOM[geom]
    S.RADII = np.array(g["radii"])
    t0 = time.time()
    mesh, solver, h = build(geom, pol, res, lts, material=material)
    solver.add_plane_wave(S.pulse())
    pts = np.vstack([circle_points(r, S.M_PTS)[0] for r in S.RADII])
    P = solver.interpolation(pts)
    t_setup = time.time() - t0
    dt = solver.dt
    n_max = int(S.T_RUN / dt)
    times = (np.arange(n_max) + 1) * dt  # u^{n+1} after step n (TM and TE)
    acc = np.zeros((len(S.FREQS), len(pts)), complex)
    t1 = time.time()
    for n in range(n_max):
        solver.one_step()
        acc += np.outer(np.exp(-2j * np.pi * S.FREQS * times[n]), P @ solver.u) * dt
        if n % 2000 == 0 and not np.all(np.isfinite(solver.u)):
            raise FloatingPointError("unstable")
    t_run = time.time() - t1
    u_inc0 = solver.incident(np.zeros((1, 2)), times)[:, 0]
    u0 = (np.exp(-2j * np.pi * np.outer(S.FREQS, times)) @ u_inc0) * dt
    out = S.extract(acc, u0, pol)
    if material is not None and hasattr(material, "epsilon"):
        from wakis.analytic import cylinder_efficiencies

        m_f = np.sqrt(material.epsilon(S.FREQS))
        out["q_ref"] = [float(cylinder_efficiencies(m, x, pol)[1])
                        for m, x in zip(m_f, S.X)]  # fmt: skip
    lc = mesh.level_counts()
    out.update(
        dt=dt, steps=n_max, nodes=int(solver.n_dof), dofs=int(solver.n_unknowns),
        cells=int(mesh.n_cells), cells_coarse=int(lc[0]), cells_fine=int(lc[1]),
        setup_s=t_setup, run_s=t_run, h=h, dt_global=float(solver.cfl * solver.dt_max),
        lts=(None if solver.lts_info is None else dict(
            fine_fraction=float(solver.lts_info["fine_fraction"]),
            buffer=int(solver.lts_info["buffer"]), rho=float(solver.lts_info["rho"]),
            layers=int(solver.lts_info["layers"]))),
    )  # fmt: skip
    return out


def run_task(arm, pol, res):
    geom, lts = ARMS[arm]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    try:
        out = run_sg(geom, pol, res, lts)
    except Exception as e:
        import traceback

        out = dict(error=repr(e), tb=traceback.format_exc())
    out.update(arm=arm, pol=pol, res=res, wall_s=time.time() - t0, nproc=os.cpu_count())
    try:
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
        from plot_subgrid import main_plot

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
        if "error" in out:
            msg = out["error"] + "\n" + out["tb"]
        else:
            q = np.array(out["q"]) / np.array(out["q_ref"]) - 1
            msg = f"rms {np.sqrt(np.mean(q**2)):.4f} max refl {max(out['refl']):.3f}"
        print(arm, pol, res, msg,
              f"dofs {out.get('dofs')} steps {out.get('steps')} "
              f"run {out.get('run_s', 0):.0f} s", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
