"""
High-order schemes on the shared Mie-cylinder benchmark (n = 2, a = 1 um,
x = 2..10, TM and TE): error vs unknowns / work / time against the best
second-order baselines (graded Cartesian conformal FIT, graded triangle FIT,
``../mie-cylinder-graded``).

Arms
----
* ``dg{p}``           -- `wakis.highorder.DGTD2D`, degree p, curved elements,
  box domain with a stretched-coordinate PML (circles 1.4/1.6/1.8 um, PML
  from 2.0 to 2.6 um).
* ``dg{p}-compact``   -- same scheme on the disk of tri-graded-compact
  (circles 1.25/1.4/1.55 um, first-order Silver-Mueller ABC on a curved
  r = 1.7 um boundary; the ABC reflections are de-embedded exactly as for
  the triangle FIT).
* ``dg{p}-straight``  -- dg{p}-compact with straight-sided (polygonal)
  interface elements (isolates the geometric interface error).
* ``fd24`` / ``fd22`` / ``fd24rk`` -- `wakis.highorder.FDTD24` on a uniform
  grid: (2,4) staggered scheme, Yee (2,2), and (2,4) with 4th-order Yoshida
  time stepping; subpixel (Hwang-Cangellaris type) averaging of eps.

``res`` = elements (DG) or cells (FD) per minimum wavelength in the
dielectric (lambda_d = 0.314 um at x = 10): h_in = lambda_d / res. DG meshes
are graded like the baselines: h_in for r < a + h_in, growing to n h_in.

Usage::

    python ho_study.py list
    python ho_study.py run --shard I --nshards N --outdir DIR
    python ho_study.py run --only dg4-compact:TM:2 --t-run 300   # smoke
    python ho_study.py plot

Local runs only under ``bin/memguard``.
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

GEOM = {
    "box": dict(radii=[1.4e-6, 1.6e-6, 1.8e-6], half=2.6e-6, pml=0.6e-6),
    "compact": dict(radii=[1.25e-6, 1.4e-6, 1.55e-6], r_out=1.7e-6),
}
DG_RES = {2: [3, 4, 6, 8, 11], 3: [2, 3, 4, 6, 8], 4: [1.5, 2, 3, 4, 6], 5: [1.2, 1.5, 2, 3, 4]}
FD_RES = [10, 14, 20, 28, 40, 56]


def arm_spec(arm):
    """('dg', p, geom, curved) or ('fd', order, yoshida)."""
    if arm.startswith("dg"):
        parts = arm[2:].split("-")
        p = int(parts[0])
        geom = "box"
        curved = True
        if "compact" in parts:
            geom = "compact"
        if "straight" in parts:
            geom, curved = "compact", False
        return ("dg", p, geom, curved)
    base, _, cfl = arm.partition("-c")
    return ("fd", {"fd22": 2, "fd24": 4, "fd24rk": 4}[base], base == "fd24rk",
            float(cfl) / 100 if cfl else 0.95)


PROBE = [("dg4", "TM", 2), ("dg4", "TE", 2), ("dg3", "TM", 2), ("dg4-compact", "TE", 2),
         ("dg4-compact", "TE", 3), ("dg5-compact", "TM", 1.5), ("dg2-compact", "TM", 4),
         ("fd24", "TM", 14), ("fd24", "TM", 20), ("fd24", "TM", 28), ("fd24-c30", "TM", 20),
         ("fd24rk", "TM", 14), ("fd24rk", "TE", 14), ("fd22", "TM", 20),
         ("dg4-straight", "TE", 2), ("dg3-compact", "TM", 3)]  # fmt: skip


LARGE = dict(
    dg3=[2, 3, 4, 6], dg4=[1.5, 2, 3, 4], dg5=[1.5, 2, 3],
    fd22=[10, 14, 20, 28, 40], fd24=[8, 11, 16, 22, 32], fd24rk=[8, 11, 16, 22, 32],
)  # fmt: skip

# benchmark case: 'b1' (shared Mie benchmark) or 'large' (showcase: lossy
# n = 2 cylinder, x = 20..30, sigma = 5000 S/m so that Im n ~ 0.02 and every
# resonance has Q < ~50: the finite time window then converges and the
# error measures the scheme, not the ring-down truncation)
CASE = dict(name="b1", sigma=0.0)


def set_case(name):
    """Patch the shared-benchmark globals of `mie_mesh_study` for a case."""
    CASE["name"] = name
    if name != "large":
        return
    CASE["sigma"] = 5000.0
    S.X = np.linspace(20.0, 30.0, 41)
    S.FREQS = S.X * C0 / (2 * np.pi * A)
    S.F_CENTER = 0.5 * (S.FREQS[0] + S.FREQS[-1])
    S.F_WIDTH = 0.25 * (S.FREQS[-1] - S.FREQS[0])
    S.N_MAX = 52
    S.LAM_MIN_DIEL = 2 * np.pi * A / (S.X.max() * N_CYL)
    GEOM["compact"] = dict(radii=[1.15e-6, 1.25e-6, 1.35e-6], r_out=1.45e-6)
    GEOM["box"] = dict(radii=[1.15e-6, 1.25e-6, 1.35e-6], half=1.75e-6, pml=0.3e-6)

    def series(pol, x=None):
        from scipy.constants import epsilon_0

        from wakis.analytic import cylinder_coefficients, cylinder_efficiencies

        x = np.atleast_1d(S.X if x is None else x)
        w = x * C0 / A
        m = np.sqrt(N_CYL**2 - 1j * CASE["sigma"] / (w * epsilon_0))  # exp(+iwt)
        q = np.array([cylinder_efficiencies(mm, xx, pol)[1] for mm, xx in zip(m, x)])
        c = np.array([cylinder_coefficients(mm, xx, pol, n_terms=S.N_MAX)
                      for mm, xx in zip(m, x)])  # fmt: skip
        return q, c.conj()  # extraction works in exp(-iwt) (Bohren-Huffman)

    S.series = series


def dg_material():
    if CASE["sigma"]:
        from wakis.dispersive import DispersiveMaterial

        return DispersiveMaterial(N_CYL**2, sigma=CASE["sigma"])
    return N_CYL**2


def tasks(name="main"):
    if name == "probe":
        return list(PROBE)
    if name == "large":
        out = []
        for arm, rs in LARGE.items():
            for res in rs:
                for pol in ("TM", "TE"):
                    out.append((arm + ("-compact" if arm.startswith("dg") else ""), pol, res))
        return out
    out = []
    for p, rs in DG_RES.items():
        for res in rs:
            for pol in ("TM", "TE"):
                out.append((f"dg{p}-compact", pol, res))
                if p in (3, 4):
                    out.append((f"dg{p}", pol, res))
                if p == 4:
                    out.append((f"dg{p}-straight", pol, res))
    for arm in ("fd22", "fd24", "fd24rk"):
        for res in FD_RES:
            for pol in ("TM", "TE"):
                out.append((arm, pol, res))
    for res in (14, 20, 28):  # temporal vs spatial error of the (2,4) leapfrog
        for pol in ("TM", "TE"):
            out.append(("fd24-c30", pol, res))
    # longest first (Batch packs them evenly)
    return out


def dg_size(h_in):
    def size(P):
        r = np.hypot(P[:, 0], P[:, 1])
        d = np.maximum(r - A - h_in, 0)
        return np.minimum(N_CYL * h_in, h_in * (1 + 0.25 * d / h_in))

    return size


def run_dg(p, geom, curved, pol, res, cfl=0.8):
    from wakis.cylinder_harmonics import circle_points
    from wakis.highorder import DGTD2D, circles_mesh

    g = GEOM[geom]
    S.RADII = np.array(g["radii"])
    t0 = time.time()
    h = S.LAM_MIN_DIEL / res
    if geom == "box":
        H = g["half"]
        mesh = circles_mesh([(0, 0, A)], dg_size(h), box=(-H, H, -H, H))
        solver = DGTD2D(
            mesh, p, {1: dg_material()}, polarization=pol, bc="abc",
            pml=dict(thickness=g["pml"], R0=1e-6, order=3), curved=curved, cfl=cfl,
        )  # fmt: skip
    else:
        mesh = circles_mesh([(0, 0, A)], dg_size(h), outer_circle=(0, 0, g["r_out"]))
        solver = DGTD2D(
            mesh, p, {1: dg_material()}, polarization=pol, bc="abc", curved=curved, cfl=cfl
        )
    solver.add_plane_wave(S.pulse())
    pts = np.vstack([circle_points(r, S.M_PTS)[0] for r in S.RADII])
    P = solver.interp_matrix(pts)
    t_setup = time.time() - t0
    dt = solver.dt
    n_max = int(S.T_RUN / dt)
    times = (np.arange(n_max) + 1) * dt
    acc = np.zeros((len(S.FREQS), len(pts)), complex)
    kern = np.exp(-2j * np.pi * np.outer(times, S.FREQS))  # (n, f)
    t1 = time.time()
    for n in range(n_max):
        solver.one_step()
        acc += np.outer(kern[n], P @ solver.u.ravel()) * dt
        if n % 2000 == 0 and not np.all(np.isfinite(solver.X)):
            raise FloatingPointError("unstable")
    t_run = time.time() - t1
    if not np.all(np.isfinite(acc)):
        raise FloatingPointError("unstable")
    u_inc0 = solver.incident_field(np.zeros((1, 2)), times)[:, 0]
    u0 = (kern.T @ u_inc0) * dt
    out = S.extract(acc, u0, pol)
    out.update(
        dt=dt, steps=n_max, dofs=int(solver.n_dofs), state=int(solver.n_state),
        cells=int(solver.K), curved=int(solver.n_curved), p=p, h=h,
        rhs_evals=5 * n_max, setup_s=t_setup, run_s=t_run,
    )  # fmt: skip
    return out


def run_fd(order, rk, pol, res, cfl=0.95):
    from wakis.cylinder_harmonics import circle_points
    from wakis.highorder.fdtd24 import FDTD24

    g = GEOM["box"]
    S.RADII = np.array(g["radii"])
    t0 = time.time()
    h = S.LAM_MIN_DIEL / res
    solver = FDTD24(
        h=h, half_width=g["half"] - g["pml"], n_pml=max(10, int(np.ceil(g["pml"] / h))),
        cylinders=[(0.0, 0.0, A, N_CYL**2, CASE["sigma"])], polarization=pol, order=order,
        time_scheme="lsrk4" if rk else "leapfrog", cfl=cfl,
    )  # fmt: skip
    solver.add_plane_wave(S.pulse())
    pts = np.vstack([circle_points(r, S.M_PTS)[0] for r in S.RADII])
    P = solver.interp_matrix(pts)
    t_setup = time.time() - t0
    dt = solver.dt
    n_max = int(S.T_RUN / dt)
    times = (np.arange(n_max) + 1) * dt + solver.sample_shift
    acc = np.zeros((len(S.FREQS), len(pts)), complex)
    kern = np.exp(-2j * np.pi * np.outer(times, S.FREQS))
    t1 = time.time()
    for n in range(n_max):
        solver.one_step()
        acc += np.outer(kern[n], P @ solver.scalar_field().ravel()) * dt
        if n % 2000 == 0 and not np.all(np.isfinite(solver.scalar_field())):
            raise FloatingPointError("unstable")
    t_run = time.time() - t1
    u_inc0 = solver.incident_field(np.zeros((1, 2)), times)[:, 0]
    u0 = (kern.T @ u_inc0) * dt
    out = S.extract(acc, u0, pol)
    out.update(
        dt=dt, steps=n_max, dofs=int(solver.n_dofs), cells=int(solver.n_cells),
        h=h, setup_s=t_setup, run_s=t_run, order=order, rk=rk,
        rhs_evals=(5 if rk else 1) * n_max,
    )  # fmt: skip
    return out


def run_task(arm, pol, res):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    spec = arm_spec(arm)
    t0 = time.time()
    try:
        if spec[0] == "dg":
            out = run_dg(spec[1], spec[2], spec[3], pol, res)
        else:
            out = run_fd(spec[1], spec[2], pol, res, spec[3])
    except Exception as e:
        import traceback

        out = dict(error=repr(e), tb=traceback.format_exc())
    out.update(arm=arm, pol=pol, res=res, wall_s=time.time() - t0, nproc=os.cpu_count())
    try:
        with open("/proc/cpuinfo") as f:
            out["cpu"] = next(l.split(":")[1].strip() for l in f if "model name" in l)
    except Exception:
        pass
    return out


def rms(out):
    q, qr = np.array(out["q"]), np.array(out["q_ref"])
    return float(np.sqrt(np.mean((q / qr - 1) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "plot"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=str(HERE / "data"))
    ap.add_argument("--t-run", type=float, default=1000.0, help="fs")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--tasklist", default="main")
    a = ap.parse_args()
    S.T_RUN = a.t_run * 1e-15
    if a.tasklist == "large":
        set_case("large")
    tl = tasks(a.tasklist)
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    if a.mode == "plot":
        from plot_ho import main_plot

        main_plot()
        return
    if a.only:
        mine = [(x.split(":")[0], x.split(":")[1], float(x.split(":")[2])) for x in a.only]
    else:
        mine = tl[a.shard :: a.nshards]
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for arm, pol, res in mine:
        out = run_task(arm, pol, res)
        (Path(a.outdir) / f"{arm}_{pol}_{res:g}.json").write_text(json.dumps(out))
        msg = out.get("error") or f"rms {rms(out):.5f}"
        print(arm, pol, res, msg, f"dofs {out.get('dofs')} steps {out.get('steps')} "
              f"run {out.get('run_s', 0):.0f} s", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
