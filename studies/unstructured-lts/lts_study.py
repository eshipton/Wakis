"""
Local time stepping (LTS) on the graded Mie-cylinder meshes: same accuracy as
global leapfrog, fewer steps?

Problem, meshes, extraction and error metric are those of
``studies/mie-cylinder-graded`` (dielectric cylinder n = 2, a = 1 um,
x = 2..10, scattered-field plane wave, de-embedded cylindrical-harmonic
extraction, 1000 fs window). Arms:

* ``tri-graded`` / ``tri-graded-compact`` (mode ``global``): the graded
  arms of that study, re-run here so that both modes share the batch;
* the same meshes with ``lts=2`` (mode ``lts``): stabilized two-level
  leapfrog LTS, coarse step ~2x the global one.

Timing: ``step_s`` is the field update only (``one_step``), ``mon_s`` the
DFT accumulation on the extraction circles (one per step as well), ``run_s``
their sum; one run per n2-standard-2 machine (uncontended).

Usage::

    python lts_study.py list
    python lts_study.py run --shard I --nshards N --outdir DIR
    python lts_study.py plot
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mie-cylinder-graded"))
sys.path.insert(0, str(HERE.parent / "mie-cylinder-mesh"))
import graded_study as GS  # noqa: E402

S = GS.S
RES = [10, 14, 20, 28, 40]
ARMS = {"tri-graded": "std", "tri-graded-compact": "compact-tri"}


def tasks():
    out = []
    for res in RES[::-1]:
        for arm in ARMS:
            for pol in ("TM", "TE"):
                for mode in ("lts", "global"):
                    out.append((arm, pol, res, mode))
    return out


def build(arm, pol, res, mode):
    from wakis.unstructured import UnstructuredFIT2D, graded_disk_mesh

    g = GS.GEOM[ARMS[arm]]
    S.RADII = np.array(g["radii"])
    h = S.LAM_MIN_DIEL / res
    mesh = graded_disk_mesh([GS.A, g["r_out"]], GS.size_field(h, True))
    solver = UnstructuredFIT2D(
        mesh, {0: [GS.N_CYL**2, 1.0]}, polarization=pol, bc="abc", cfl=S.CFL,
        lts=2 if mode == "lts" else None,
    )  # fmt: skip
    return mesh, solver, h


def run(arm, pol, res, mode):
    from wakis.cylinder_harmonics import circle_points

    t0 = time.time()
    mesh, solver, h = build(arm, pol, res, mode)
    solver.add_plane_wave(S.pulse())
    pts = np.vstack([circle_points(r, S.M_PTS)[0] for r in S.RADII])
    P = S.tri_interp(solver, pts, 2 * h)
    t_setup = time.time() - t0
    dt = solver.dt
    shift = 0.0 if pol == "TM" else -0.5 * dt
    acc = np.zeros((len(S.FREQS), len(pts)), complex)
    n_max = int(S.T_RUN / dt)
    times = (np.arange(n_max) + 1) * dt + shift
    t_step = t_mon = 0.0
    for n in range(n_max):
        ta = time.perf_counter()
        solver.one_step()
        tb = time.perf_counter()
        acc += np.outer(np.exp(-2j * np.pi * S.FREQS * times[n]), P @ solver.u) * dt
        t_mon += time.perf_counter() - tb
        t_step += tb - ta
        if n % 2000 == 0 and not np.all(np.isfinite(solver.u)):
            raise FloatingPointError("unstable")
    u_inc0 = solver.incident_field(np.zeros((1, 2)), times)
    u0 = (np.exp(-2j * np.pi * np.outer(S.FREQS, times)) @ u_inc0) * dt
    out = S.extract(acc, u0, pol)
    out.update(
        dt=dt, steps=n_max, nodes=int(solver.free.size),
        dofs=int(solver.free.size + mesh.n_edges), cells=int(mesh.n_cells),
        setup_s=t_setup, step_s=t_step, mon_s=t_mon, run_s=t_step + t_mon, h=h,
    )  # fmt: skip
    if solver.lts is not None:
        L = solver.lts
        info = solver.lts_info
        out.update(
            fine=int(L.F.size), buffer=int(L.C.size), rho=float(info["rho"]),
            layers=int(info["layers"]), nnz_AF=int(L.A_F.nnz),
            history=[[float(a), float(b), float(c), d] for a, b, c, d in info["history"]],
        )  # fmt: skip
    return out


def run_task(arm, pol, res, mode):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    try:
        out = run(arm, pol, res, mode)
    except Exception as e:
        import traceback

        out = dict(error=repr(e), tb=traceback.format_exc())
    out.update(arm=arm, pol=pol, res=res, mode=mode, wall_s=time.time() - t0)
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
    ap.add_argument("--only", nargs="*", help="arm:pol:res:mode (local tests)")
    a = ap.parse_args()
    S.T_RUN = a.t_run * 1e-15
    tl = tasks()
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    if a.mode == "plot":
        from plot_lts import main_plot

        main_plot()
        return
    if a.only:
        mine = [tuple(x.split(":")) for x in a.only]
        mine = [(x[0], x[1], int(x[2]), x[3]) for x in mine]
    else:
        mine = tl[a.shard :: a.nshards]
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for t in mine:
        out = run_task(*t)
        name = Path(a.outdir) / ("_".join(map(str, t)) + ".json")
        name.write_text(json.dumps(out))
        msg = out.get("error") or (
            "rms %.4f" % np.sqrt(np.mean((np.array(out["q"]) / out["q_ref"] - 1) ** 2))
        )
        print(*t, msg, "steps", out.get("steps"), "run %.1f s" % out.get("run_s", 0),
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
