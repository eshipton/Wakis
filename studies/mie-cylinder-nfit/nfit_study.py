"""
Mie cylinder (n = 2, a = 1 um, x = 2..10) on body-fitted O-grids with
time-domain curvilinear FIT (`wakis.curvilinear.CurvilinearFIT2D`): how many
unknowns / how much work to reach 2 % and 1 % RMS error in Q_sca, against the
graded Cartesian and graded triangle baselines of ``../mie-cylinder-graded``.

Arms (h_in = lambda_min,diel / res; n_theta = 4 ceil(2 pi a / (4 h_in))):

* ``ogrid``        -- butterfly core (r < 0.9 a, nonorthogonal corner Hodge)
  + polar rings (diagonal Hodge): h_in for r < a + 2 h_in, growth 1.2 to
  h_out = n h_in; extraction rings 1.15 / 1.3 / 1.45 um; radial PML from
  1.55 um, N_PML cells of h_out, PEC behind.
* ``ogrid-std``    -- the same with the baseline geometry (rings 1.9 / 2.2 /
  2.5 um, PML from 2.7 um).
* ``ogrid-corner`` -- as ``ogrid`` but the nonorthogonal corner Hodge on all
  cells inside the PML (straight chords instead of exact arcs): isolates
  the effect of the Hodge choice.
* ``multiblock``   -- core + polar rings to 1.55 um, a circle-to-square
  transition block (nonorthogonal), a Cartesian outer block (uniform cells)
  with a Cartesian PML.

Extraction: the scattered field on the node rings (exact radii, uniform
angles) is resampled to 256 angles by FFT (trigonometric interpolation) and
passed to the shared harmonic fit and S-matrix de-embedding
(`wakis.cylinder_harmonics`, ``mie_mesh_study.extract``).

Usage::

    python nfit_study.py list
    python nfit_study.py run --shard I --nshards N --outdir DIR
    python nfit_study.py run --only ogrid:TM:10      # local smoke test
    python nfit_study.py plot

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
RES = [10, 14, 20, 28, 40]
GROWTH = 1.2
N_PML = 10
R_CORE = 0.9  # core radius / a
GEOM = {
    "compact": dict(radii=[1.15e-6, 1.3e-6, 1.45e-6], r_pml=1.55e-6),
    "std": dict(radii=[1.9e-6, 2.2e-6, 2.5e-6], r_pml=2.7e-6),
}
ARMS = {
    "ogrid": dict(geom="compact", hodge="polar"),
    "ogrid-std": dict(geom="std", hodge="polar"),
    "ogrid-corner": dict(geom="compact", hodge="corner"),
    "multiblock": dict(geom="compact", hodge="multiblock"),
}


def tasks(arms=None):
    out = []
    for res in RES[::-1]:
        for arm in arms or ARMS:
            for pol in ("TM", "TE"):
                out.append((arm, pol, res))
    return out


def size_field(h_in):
    band, h_out = 2 * h_in, N_CYL * h_in

    def size(r):
        if r < A - 1e-12:
            return h_in
        d = r - A - band
        return h_in if d <= 0 else min(h_out, h_in * GROWTH ** (d / h_in))

    return size


def n_theta_for(h_in):
    return int(4 * np.ceil(2 * np.pi * A / h_in / 4))


def build(arm, res, shell=None):
    """Mesh, PML spec and extraction radii of an arm.

    shell : (thickness, eps) of a coating on the cylinder (showcase), or None.
    """
    from wakis.curvilinear import ogrid_cartesian_mesh, ogrid_mesh, radial_nodes

    cfg = ARMS[arm]
    g = GEOM[cfg["geom"]]
    h = S.LAM_MIN_DIEL / res
    h_out = N_CYL * h
    size = size_field(h)
    nt = n_theta_for(h)
    interfaces = [A]
    breaks = [R_CORE * A, A]
    if shell is not None:
        t_sh = shell[0]
        interfaces = [A, A + t_sh]
        breaks = [R_CORE * A, A, A + t_sh]
        def size(r):  # noqa: F811
            from shell_study import shell_size

            return shell_size(r, h)

    if cfg["hodge"] == "multiblock":
        r_b = g["r_pml"]
        radii = radial_nodes(breaks + list(g["radii"]) + [r_b], size)
        S_half = r_b + 2 * h_out
        hc = 2 * S_half / (nt // 4)
        n_mid = 4
        n_out = n_mid + N_PML
        x_outer = S_half + hc * np.arange(1, n_out + 1)
        mesh = ogrid_cartesian_mesh(radii, nt, S_half, x_outer, m_tr_out=2,
                                    interfaces=interfaces)  # fmt: skip
        x0 = S_half + n_mid * hc
        pml = dict(kind="cartesian", r0=x0, r1=x_outer[-1])
    else:
        r0 = g["r_pml"]
        r1 = r0 + N_PML * h_out
        radii = radial_nodes(breaks + list(g["radii"]) + [r0, r1], lambda r: (
            h_out if r >= r0 - 1e-12 else size(r)))  # fmt: skip
        polar = True if cfg["hodge"] == "polar" else r0
        mesh = ogrid_mesh(radii, nt, interfaces=interfaces, polar=polar)
        pml = dict(kind="radial", r0=r0, r1=r1)
    return mesh, pml, np.asarray(g["radii"]), h


def ring_resampler(mesh, solver, radius, m_pts):
    """(free indices, function u_ring -> 256 samples) by FFT resampling."""
    idx, th = mesh.ring(radius)
    loc = np.searchsorted(solver.free, idx)
    assert np.all(solver.free[loc] == idx)
    nr = len(idx)
    dth = np.diff(th)
    assert np.allclose(dth, 2 * np.pi / nr, rtol=1e-6), "ring not uniform"
    th0 = th[0]
    k = np.fft.fftfreq(nr, 1.0 / nr)
    phase_out = np.exp(1j * np.outer(2 * np.pi * np.arange(m_pts) / m_pts - th0, k))
    # keep |k| < nr/2 (drop the Nyquist term for even nr)
    keep = np.abs(k) < nr / 2
    Tm = (phase_out[:, keep] @ np.fft.fft(np.eye(nr), axis=0)[keep]) / nr
    if os.environ.get("NFIT_INTERP") == "linear":  # check: linear in angle
        tq = np.mod(2 * np.pi * np.arange(m_pts) / m_pts - th0, 2 * np.pi)
        j = np.floor(tq / (2 * np.pi / nr)).astype(int)
        f = tq / (2 * np.pi / nr) - j
        Tm = np.zeros((m_pts, nr))
        Tm[np.arange(m_pts), j % nr] += 1 - f
        Tm[np.arange(m_pts), (j + 1) % nr] += f
    return loc, Tm.real.copy()


def run_ogrid(arm, pol, res, materials=None, shell=None, freqs=None, keep_time=False):
    from wakis.curvilinear import CurvilinearFIT2D

    t0 = time.time()
    mesh, pml, radii, h = build(arm, res, shell=shell)
    S.RADII = radii
    if materials is None:
        materials = {1: [N_CYL**2, 1.0]}
    solver = CurvilinearFIT2D(mesh, materials, polarization=pol, pml=pml, cfl=S.CFL)
    src = S.pulse()
    solver.add_plane_wave(src)
    rings = [ring_resampler(mesh, solver, r, S.M_PTS) for r in radii]
    t_setup = time.time() - t0
    dt = solver.dt
    shift = 0.0 if pol == "TM" else -0.5 * dt
    freqs = S.FREQS if freqs is None else freqs
    n_max = int(S.T_RUN / dt)
    times = (np.arange(n_max) + 1) * dt + shift
    # DFT on the ring nodes (then resample: linear, so it commutes)
    locs = np.concatenate([r[0] for r in rings])
    acc = np.zeros((len(freqs), locs.size), complex)
    ker = np.exp(-2j * np.pi * np.outer(times, freqs))  # (n, f)
    t1 = time.time()
    chunk = 256
    buf = np.zeros((chunk, locs.size))
    for n in range(n_max):
        solver.one_step()
        buf[n % chunk] = solver.u[locs]
        if n % chunk == chunk - 1 or n == n_max - 1:
            n0 = n - n % chunk
            acc += ker[n0 : n + 1].T @ buf[: n - n0 + 1] * dt
        if n % 2000 == 0 and not np.all(np.isfinite(solver.u)):
            raise FloatingPointError("unstable")
    t_run = time.time() - t1
    parts, o = [], 0
    for loc, Tm in rings:
        parts.append(acc[:, o : o + loc.size] @ Tm.T)
        o += loc.size
    dft = np.concatenate(parts, axis=1)
    u_inc0 = solver.incident_field(np.zeros((1, 2)), times)
    u0 = (np.exp(-2j * np.pi * np.outer(freqs, times)) @ u_inc0) * dt
    out = dict(dft=dft, u0=u0) if keep_time else S.extract(dft, u0, pol)
    out.update(
        dt=dt, steps=n_max, nodes=int(solver.free.size), dofs=solver.n_unknowns,
        cells=int(mesh.n_cells), setup_s=t_setup, run_s=t_run, h=h,
        n_theta=n_theta_for(h), courant=float(solver.courant),
        cdt_over_h=float(dt * C0 / h), quality=mesh.quality(),
    )  # fmt: skip
    return out


def run_task(arm, pol, res):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    try:
        out = run_ogrid(arm, pol, res)
    except Exception as e:
        import traceback

        out = dict(error=repr(e), tb=traceback.format_exc())
    out.update(arm=arm, pol=pol, res=res, wall_s=time.time() - t0, nproc=os.cpu_count())
    try:
        with open("/proc/cpuinfo") as f:
            out["cpu"] = next(line.split(":")[1].strip() for line in f if "model name" in line)
    except Exception:
        import platform

        out["cpu"] = platform.processor() or platform.machine()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "plot"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=str(HERE / "data"))
    ap.add_argument("--t-run", type=float, default=1000.0, help="fs")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--only", nargs="*", help="arm:pol:res tasks")
    a = ap.parse_args()
    S.T_RUN = a.t_run * 1e-15
    tl = tasks(a.arms)
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    if a.mode == "plot":
        from plot_nfit import main_plot

        main_plot()
        return
    if a.only:
        mine = [(x.split(":")[0], x.split(":")[1], int(x.split(":")[2])) for x in a.only]
    else:
        mine = tl[a.shard :: a.nshards]
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for arm, pol, res in mine:
        out = run_task(arm, pol, res)
        (Path(a.outdir) / f"{arm}_{pol}_{res}.json").write_text(json.dumps(out))
        msg = out.get("error") or "rms {:.4f}".format(
            np.sqrt(np.mean((np.array(out["q"]) / np.array(out["q_ref"]) - 1) ** 2)))  # fmt: skip
        print(arm, pol, res, msg, f"dofs {out.get('dofs')} steps {out.get('steps')} "
              f"run {out.get('run_s', 0):.0f} s c*dt/h {out.get('cdt_over_h', 0):.3f}",
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
