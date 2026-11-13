"""
3D Cartesian subgridding study (`wakis.subgrid3d`, milestone 1).

Three parts, all independent runs packed onto every core (multiprocessing):

A. Plane-wave reflection of a planar coarse/fine interface (exact
   time-discrete steady state, `bloch_reflection_3d`) vs coarse cells per
   wavelength N and angle, s and p, 2:1 and 3:1; the measurement floor
   (no patch), the exact 1D formula at normal incidence and the 2D TM
   result (`wakis.subgrid.interface.bloch_reflection`).
B. PEC-cavity eigenfrequencies (6 lowest TE+TM modes, analytic) vs h for a
   uniform coarse grid (== SolverFIT3D's operator), an off-centre 2:1 and
   3:1 patch, a nested 2:1+2:1 patch and the 'fraction' mutation.
C. Cost at equal accuracy: PEC box 1.0 x 0.8 x 0.6 m with a dielectric cube
   eps_r = 9 (0.3 m side). Lowest resonance from a time-domain run
   (current pulse, probe, Hann-window DFT peak, mapped back to the
   semi-discrete frequency). Arms: uniform grid h everywhere (the
   SolverFIT3D discretisation) and a coarse grid H = r h with an r:1 patch
   over the cube plus 0.1 m. Reference: Richardson extrapolation of the
   uniform arms. Wall-clock per step is measured serially afterwards for
   SolverFIT3D.one_step (uniform grid, CPU, SciPy CSR, 1 thread) and
   SubgridFIT3D.one_step on the same machine.

Usage: python studies/subgrid3d/run_study.py [--quick] [--parts ABC]
Writes studies/subgrid3d/results.json.
"""

import argparse
import json
import os
import platform
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.constants import c as C0  # noqa: E402

from wakis.subgrid3d import SubgridFIT3D, SubgridMesh3D  # noqa: E402
from wakis.subgrid3d.interface import bloch_reflection_3d, reflection_1d_exact  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ part A
def task_reflection(args):
    N, theta, ratio, pol, kind = args
    t0 = time.time()
    if kind == "floor":
        R = bloch_reflection_3d(N, theta, ratio, pol, refine=False)
    elif kind == "2d":
        from wakis.subgrid.interface import bloch_reflection

        R = bloch_reflection(N, theta, ratio, "TM")
    elif kind == "3d_eqsrc":
        R = bloch_reflection_3d(N, theta, ratio, pol, bloch_source=False)
    elif kind in ("fraction", "missing_transverse", "inject"):
        R = bloch_reflection_3d(N, theta, ratio, pol, mutate=kind)
    else:
        R = bloch_reflection_3d(N, theta, ratio, pol)
    return {"N": N, "theta": theta, "ratio": ratio, "pol": pol, "kind": kind, "R": R,
            "R1d": reflection_1d_exact(N, ratio) if theta == 0 else None, "s": time.time() - t0}  # fmt: skip


def jobs_reflection(quick):
    Ns = [10, 20] if quick else [8, 10, 14, 20, 28, 40]
    th = [0, 30, 60] if quick else [0, 15, 30, 45, 60, 75]
    jobs = []
    for r in (2, 3):
        for pol in ("s", "p"):
            for N in Ns:
                for t in th:
                    jobs.append((N, t, r, pol, "3d"))
                    if r == 2:
                        jobs.append((N, t, r, pol, "floor"))
        for N in Ns:
            for t in th:
                if t > 0:
                    jobs.append((N, t, r, "s", "2d"))
                    jobs.append((N, t, r, "s", "3d_eqsrc"))
    for kind in ("fraction", "missing_transverse", "inject"):
        for N in Ns:
            jobs.append((N, 60, 2, "s", kind))
    return jobs


# ------------------------------------------------------------------ part B
A_CAV, B_CAV, D_CAV = 1.0, 0.75, 0.5
PATCH = ((0.25, 0.625), (0.25, 0.5), (0.125, 0.375))


def exact_modes(k):
    out = []
    for i in range(10):
        for j in range(10):
            for m in range(10):
                nz = (i > 0) + (j > 0) + (m > 0)
                if nz >= 2:
                    f = C0 / 2 * np.sqrt((i / A_CAV) ** 2 + (j / B_CAV) ** 2 + (m / D_CAV) ** 2)
                    out += [f] * (2 if nz == 3 else 1)
    return np.sort(out)[:k]


def task_cavity(args):
    N, arm = args
    t0 = time.time()
    m = SubgridMesh3D(np.linspace(0, A_CAV, N + 1), np.linspace(0, B_CAV, 3 * N // 4 + 1),
                      np.linspace(0, D_CAV, N // 2 + 1))  # fmt: skip
    mutate = None
    if arm in ("r2", "r3", "nested", "fraction"):
        p = m.add_box(PATCH, ratio=3 if arm == "r3" else 2)
        if arm == "nested":
            q = 1.0 / N
            m.add_box(tuple((lo + q, hi - q) for lo, hi in PATCH), ratio=2, parent=p)
        if arm == "fraction":
            mutate = "fraction"
    s = SubgridFIT3D(m, dt=1.0, mutate=mutate)
    ex = exact_modes(6)
    fr = s.eigenfrequencies(6, ex[0])
    return {"N": N, "arm": arm, "h": A_CAV / N, "f": fr.tolist(), "exact": ex.tolist(),
            "err": np.abs(fr / ex - 1).tolist(), "n_free": int(s.n_free),
            "n_faces": int(s.n_faces), "s": time.time() - t0}  # fmt: skip


def jobs_cavity(quick):
    Ns = [8, 16] if quick else [8, 12, 16, 24, 32]
    jobs = []
    for N in Ns:
        for arm in ("uniform", "r2", "r3", "nested", "fraction"):
            if arm == "nested" and N < 16:
                continue  # inner box needs one coarse cell of margin
            jobs.append((N, arm))
    return jobs


# ------------------------------------------------------------------ part C
BOX = (1.0, 0.8, 0.6)
CUBE = ((0.3, 0.6), (0.2, 0.5), (0.2, 0.5))
PATCH_C = ((0.2, 0.7), (0.1, 0.6), (0.1, 0.6))
EPS_CUBE = 9.0
N_PERIODS = 300


def eps_cube(X, Y, Z):
    c = CUBE
    inside = (X > c[0][0]) & (X < c[0][1]) & (Y > c[1][0]) & (Y < c[1][1]) & (Z > c[2][0]) & (Z < c[2][1])
    return np.where(inside, EPS_CUBE, 1.0)


def loaded_solver(h_coarse, ratio, cfl=0.95):
    axes = [np.linspace(0, L, int(round(L / h_coarse)) + 1) for L in BOX]
    m = SubgridMesh3D(*axes)
    if ratio > 1:
        m.add_box(PATCH_C, ratio=ratio)
    return SubgridFIT3D(m, eps_r=eps_cube, cfl=cfl)


def peak_frequency(v, dt, f_guess):
    """Hann-window DFT peak (zero-padded FFT + parabolic) near f_guess [Hz]."""
    n = len(v)
    pad = 16 * n
    X = np.abs(np.fft.rfft(np.hanning(n) * v, n=pad))
    f = np.fft.rfftfreq(pad, dt)
    sel = np.flatnonzero(np.abs(f / f_guess - 1) < 0.15)
    i = sel[np.argmax(X[sel])]
    y0, y1, y2 = X[i - 1], X[i], X[i + 1]
    return f[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (f[1] - f[0])


def task_loaded(args):
    h_coarse, ratio, f_guess = args
    t0 = time.time()
    s = loaded_solver(h_coarse, ratio)
    setup = time.time() - t0
    f0 = f_guess
    tau = 2.0 / f0
    # two off-symmetric sources inside the cube, one probe inside, one outside
    for p, comp in (((0.41, 0.33, 0.37), 2), ((0.47, 0.29, 0.31), 0)):
        s.add_current(p, comp, lambda t, tau=tau: np.exp(-(((t - 4 * tau) / tau) ** 2)) * np.sin(2 * np.pi * f0 * t))
    s.add_probe((0.37, 0.41, 0.33), 2)
    s.add_probe((0.52, 0.27, 0.44), 1)
    n_steps = int(np.ceil(N_PERIODS / f_guess / s.dt))
    t1 = time.time()
    s.run(n_steps)
    run_s = time.time() - t1
    out = []
    for p in s.probes:
        ftd = peak_frequency(np.array(p["values"]), s.dt, f_guess)
        out.append(2 * np.sin(np.pi * ftd * s.dt) / s.dt / (2 * np.pi))  # semi-discrete
    return {"h_coarse": h_coarse, "ratio": ratio, "h_fine": h_coarse / max(ratio, 1),
            "n_free": int(s.n_free), "n_faces": int(s.n_faces),
            "unknowns": int(s.n_free + s.n_faces), "dt": s.dt, "dt_max": s.dt_max,
            "n_steps": n_steps, "f_semi_probes": out, "setup_s": setup, "run_s": run_s,
            "s_per_step": run_s / n_steps}  # fmt: skip


def jobs_loaded(quick, f_guess):
    if quick:
        uni = [0.1, 0.05]
        sg = [(0.1, 3), (0.1, 2)]
    else:
        uni = [0.1, 0.05, 0.1 / 3, 0.025, 0.02, 0.05 / 3, 0.0125]
        sg = [(0.1, 3), (0.05, 3), (0.1 / 3, 3), (0.1, 2), (0.05, 2), (0.1 / 3, 2), (0.025, 2)]
    return [(h, 1, f_guess) for h in uni] + [(h, r, f_guess) for h, r in sg]


def guess_f1():
    s = loaded_solver(0.05, 1)
    return float(s.eigenfrequencies(2, 1.0e8)[0]), s.eigenfrequencies(4, 1.0e8).tolist()


def timing_serial(h_list, n_steps=60):
    """Per-step wall clock on one core: SolverFIT3D vs SubgridFIT3D (uniform)."""
    from wakis import GridFIT3D, SolverFIT3D

    out = []
    for h in h_list:
        axes = [np.linspace(0, L, int(round(L / h)) + 1) for L in BOX]
        g = GridFIT3D(x=axes[0], y=axes[1], z=axes[2], verbose=0)
        sv = SolverFIT3D(g, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0, cfln=0.95)
        rng = np.random.default_rng(0)
        sv.E.fromarray(rng.standard_normal(sv.E.toarray().shape))
        sv.one_step()
        t0 = time.time()
        for _ in range(n_steps):
            sv.one_step()
        t_wk = (time.time() - t0) / n_steps
        s = loaded_solver(h, 1)
        s.e[:] = rng.standard_normal(s.n_free)
        s.one_step()
        t0 = time.time()
        for _ in range(n_steps):
            s.one_step()
        t_sg = (time.time() - t0) / n_steps
        out.append({"h": h, "cells": int(np.prod([len(a) - 1 for a in axes])),
                    "solverfit3d_s_per_step": t_wk, "subgrid3d_uniform_s_per_step": t_sg,
                    "solverfit3d_dt": float(sv.dt), "subgrid3d_dt": s.dt,
                    "subgrid3d_unknowns": int(s.n_free + s.n_faces),
                    "solverfit3d_unknowns_incl_ghosts": int(2 * 3 * sv.N)})  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="ABC")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--nproc", type=int, default=os.cpu_count())
    a = ap.parse_args()
    res = {"meta": {"host": platform.node(), "cpus": os.cpu_count(), "quick": a.quick,
                    "numpy": np.__version__, "started": time.ctime()}}  # fmt: skip
    if os.path.exists(a.out):
        with open(a.out) as f:
            old = json.load(f)
        res.update({k: v for k, v in old.items() if k != "meta"})
    nproc = a.nproc
    t0 = time.time()
    with Pool(nproc) as pool:
        if "A" in a.parts:
            res["reflection"] = pool.map(task_reflection, jobs_reflection(a.quick), chunksize=1)
            print(f"A done {time.time() - t0:.0f} s", flush=True)
        if "B" in a.parts:
            res["cavity"] = pool.map(task_cavity, jobs_cavity(a.quick), chunksize=1)
            print(f"B done {time.time() - t0:.0f} s", flush=True)
        if "C" in a.parts:
            f1, f4 = guess_f1()
            jobs = sorted(jobs_loaded(a.quick, f1), key=lambda j: j[0])  # largest last? small h first
            res["loaded_guess"] = {"f1_h0.05": f1, "f_lowest4_h0.05": f4}
            res["loaded"] = pool.map(task_loaded, jobs, chunksize=1)
            print(f"C done {time.time() - t0:.0f} s", flush=True)
    if "C" in a.parts:
        hs = [0.1, 0.05] if a.quick else [0.1, 0.05, 0.1 / 3, 0.025, 0.02]
        res["timing"] = timing_serial(hs)
        print(f"timing done {time.time() - t0:.0f} s", flush=True)
    res["meta"]["finished"] = time.ctime()
    res["meta"]["wall_s"] = time.time() - t0
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
