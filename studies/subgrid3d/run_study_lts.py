"""
3D subgridding, wave 2: local time stepping, CPML and GPU block kernels.

Parts (each one Cloud Batch job; independent runs packed onto every core):

D. Cost at equal accuracy, the wave-1 problem: PEC box 1.0 x 0.8 x 0.6 m,
   eps_r = 9 cube [0.3,0.6]x[0.2,0.5]x[0.2,0.5] m, patch = cube + 0.1 m.
   Lowest resonance f1 from a time-domain run (two current pulses, two
   probes, 300 periods, Hann DFT peak). Arms, each at 0.95 of its own
   stability limit:
     uniform  - uniform grid h (the SolverFIT3D discretisation), Lanczos dt_max;
     global   - wave-1 SubgridFIT3D, coarse H = r h + r:1 patch, global dt
                (Lanczos dt_max of the composite = set by the fine cells);
     lts      - the same mesh with multi-level LTS (SubgridLTSFIT3D),
                dt_max by bisection on the LTS stability criterion.
E. The same with a small feature: eps_r = 9 cube of side 0.1 m at
   [0.45,0.55]x[0.35,0.45]x[0.25,0.35], patch = cube + 0.05 m (the fine
   patch is 0.8 % of the box volume), 200 periods.
F. CPML: probe signal against a 3.7x larger closed reference run (time
   gated), for layer thickness 4-12 cells, with no patch, a 2:1 patch one
   coarse cell from the layer (global dt and LTS), and a patch through the
   layer (CFS shift); late-time growth vs the CFS shift for a patch inside
   the layer.
G. GPU (L4): per-step wall clock of SolverFIT3D (use_gpu, CSR and fused
   kernels) on the uniform arms, and SubgridLTSFIT3D (block kernels, CuPy)
   on the global and LTS arms of D; GPU vs CPU state after 200 steps.

Usage: python studies/subgrid3d/run_study_lts.py --parts D [--quick] [--out F]
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

from wakis.subgrid3d import SubgridLTSFIT3D, SubgridMesh3D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

BOX = (1.0, 0.8, 0.6)
CASES = {
    "big": dict(cube=((0.3, 0.6), (0.2, 0.5), (0.2, 0.5)),
                patch=((0.2, 0.7), (0.1, 0.6), (0.1, 0.6)), periods=300,
                src=(((0.41, 0.33, 0.37), 2), ((0.47, 0.29, 0.31), 0)),
                probes=(((0.37, 0.41, 0.33), 2), ((0.52, 0.27, 0.44), 1))),
    "small": dict(cube=((0.45, 0.55), (0.35, 0.45), (0.25, 0.35)),
                  patch=((0.4, 0.6), (0.3, 0.5), (0.2, 0.4)), periods=200,
                  src=(((0.47, 0.41, 0.31), 2), ((0.52, 0.38, 0.29), 0)),
                  probes=(((0.43, 0.42, 0.33), 2), ((0.53, 0.37, 0.27), 1))),
}  # fmt: skip
EPS_CUBE = 9.0


def eps_fn(case):
    c = CASES[case]["cube"]

    def eps(X, Y, Z):
        inside = ((X > c[0][0]) & (X < c[0][1]) & (Y > c[1][0]) & (Y < c[1][1])
                  & (Z > c[2][0]) & (Z < c[2][1]))  # fmt: skip
        return np.where(inside, EPS_CUBE, 1.0)

    return eps


def make_solver(case, kind, h_coarse, ratio, dt=None, **kw):
    axes = [np.linspace(0, L, int(round(L / h_coarse)) + 1) for L in BOX]
    m = SubgridMesh3D(*axes)
    if kind != "uniform":
        m.add_box(CASES[case]["patch"], ratio=ratio)
    return SubgridLTSFIT3D(m, eps_r=eps_fn(case), lts=(kind == "lts"), dt=dt, cfl=0.95, **kw)


def peak_frequency(v, dt, f_guess):
    n = len(v)
    pad = 16 * n
    X = np.abs(np.fft.rfft(np.hanning(n) * v, n=pad))
    f = np.fft.rfftfreq(pad, dt)
    sel = np.flatnonzero(np.abs(f / f_guess - 1) < 0.15)
    i = sel[np.argmax(X[sel])]
    y0, y1, y2 = X[i - 1], X[i], X[i + 1]
    return f[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (f[1] - f[0])


def guess_f1(case):
    s = make_solver(case, "uniform", 0.05, 1, dt=1.0)
    return float(s.eigenfrequencies(1, 1.0e8)[0])


def task_cost(args):
    case, kind, h_coarse, ratio, f_guess = args
    cfg = CASES[case]
    t0 = time.time()
    s = make_solver(case, kind, h_coarse, ratio)
    setup = time.time() - t0
    tau = 2.0 / f_guess
    for p, comp in cfg["src"]:
        s.add_current(p, comp, lambda t, tau=tau: np.exp(-(((t - 4 * tau) / tau) ** 2))
                      * np.sin(2 * np.pi * f_guess * t))  # fmt: skip
    for p, comp in cfg["probes"]:
        s.add_probe(p, comp)
    n_steps = int(np.ceil(cfg["periods"] / f_guess / s.dt))
    t1 = time.time()
    s.run(n_steps)
    run_s = time.time() - t1
    f_td = [peak_frequency(np.array(p["values"]), s.dt, f_guess) for p in s.probes]
    f_semi = [2 * np.sin(np.pi * f * s.dt) / s.dt / (2 * np.pi) for f in f_td]
    unknowns = int(s.n_free + s.n_faces)
    wps = s.lts.work_per_step() if s.lts is not None else unknowns
    out = {"case": case, "kind": kind, "h_coarse": h_coarse, "ratio": ratio,
           "h_fine": h_coarse / ratio, "n_free": int(s.n_free), "n_faces": int(s.n_faces),
           "unknowns": unknowns, "dt": s.dt, "dt_global_max": s.dt_max,
           "n_steps": n_steps, "work": float(wps) * n_steps, "work_per_step": int(wps),
           "f_td_probes": f_td, "f_semi_probes": f_semi, "setup_s": setup,
           "run_s": run_s, "s_per_step_pool": run_s / n_steps}  # fmt: skip
    if s.lts is not None:
        info = dict(s.lts_info)
        info["bisection"] = [[float(a) for a in row] for row in info["bisection"]]
        out["lts"] = info
        out["rho"] = float(s.stability_ratio())
    return out


def jobs_cost(case, quick, f_guess):
    if case == "big":
        uni = [0.1, 0.05] if quick else [0.1, 0.05, 0.1 / 3, 0.025, 0.02, 0.05 / 3, 0.0125]
        sg = [(0.1, 2), (0.1, 3)] if quick else [(0.1, 3), (0.05, 3), (0.1 / 3, 3), (0.1, 2),
                                                 (0.05, 2), (0.1 / 3, 2), (0.025, 2)]  # fmt: skip
    else:
        uni = [0.05, 0.025] if quick else [0.05, 0.025, 1 / 60, 0.0125, 0.01, 1 / 120]
        sg = [(0.05, 2)] if quick else [(0.05, 2), (0.025, 2), (0.0125, 2), (0.05, 3),
                                        (0.025, 3)]  # fmt: skip
    jobs = [(case, "uniform", h, 1, f_guess) for h in uni]
    for h, r in sg:
        jobs += [(case, "global", h, r, f_guess), (case, "lts", h, r, f_guess)]
    # largest first (longest runs start first)
    return sorted(jobs, key=lambda j: j[2] / j[3])


def timing_serial(results, n_steps=40):
    """Per-step wall clock on one core, same VM, after the parallel phase."""
    from wakis import GridFIT3D, SolverFIT3D

    out = []
    rng = np.random.default_rng(0)
    for r in results:
        s = make_solver(r["case"], r["kind"], r["h_coarse"], r["ratio"], dt=r["dt"])
        s.e[:] = rng.standard_normal(s.n_free)
        s.one_step()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            s.one_step()
        row = {"case": r["case"], "kind": r["kind"], "h_coarse": r["h_coarse"],
               "ratio": r["ratio"], "s_per_step": (time.perf_counter() - t0) / n_steps}  # fmt: skip
        if r["kind"] == "uniform":
            axes = [np.linspace(0, L, int(round(L / r["h_coarse"])) + 1) for L in BOX]
            g = GridFIT3D(x=axes[0], y=axes[1], z=axes[2], verbose=0)
            sv = SolverFIT3D(g, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0, cfln=0.95)
            sv.E.fromarray(rng.standard_normal(sv.E.toarray().shape))
            sv.one_step()
            t0 = time.perf_counter()
            for _ in range(n_steps):
                sv.one_step()
            row["solverfit3d_s_per_step"] = (time.perf_counter() - t0) / n_steps
        out.append(row)
        print("timing", row, flush=True)
    return out


# ------------------------------------------------------------------ part F
H_F = 0.04


def _box_f(n_int, n_pml, n_ext, patch, lts, dt, pml_kw=None, through=False):
    """Interior n_int cells + n_pml layer cells each side (+ n_ext cells of
    closed padding for the reference)."""
    n = n_int + 2 * n_pml + 2 * n_ext
    x = (np.arange(n + 1) - (n_pml + n_ext)) * H_F  # interior starts at 0
    m = SubgridMesh3D(x, x, x)
    L = n_int * H_F
    if patch:
        if through:
            m.add_box(((0.5 * L, L + (n_pml - 1) * H_F), (0.4 * L, 0.6 * L), (0.4 * L, 0.6 * L)),
                      ratio=2)  # fmt: skip
        else:
            m.add_box(((0.5 * L, L - H_F), (0.4 * L, 0.6 * L), (0.4 * L, 0.6 * L)), ratio=2)
    pml = None
    if n_pml and n_ext == 0:
        pml = dict(thickness=[(n_pml * H_F, n_pml * H_F)] * 3, R0=1e-6)
        pml.update(pml_kw or {})
    return SubgridLTSFIT3D(m, lts=lts, dt=dt, pml=pml)


def task_cpml(args):
    n_pml, variant = args
    n_int = 20
    L = n_int * H_F
    lts = variant == "patch_lts"
    patch = variant != "nopatch"
    through = variant == "through"
    pml_kw = dict(allow_patches=True, alpha_hz=1.0e9) if through else None
    # common dt: 0.95 of the limit of the open run's mesh
    s = _box_f(n_int, n_pml, 0, patch, lts, None, pml_kw, through)
    dt = s.dt
    f0 = 1.0e9
    tau = 0.5 / f0
    src_pt = (0.3 * L, 0.5 * L, 0.5 * L)
    prb = [(L - 2.5 * H_F, 0.45 * L, 0.55 * L), (0.5 * L + 0.5 * H_F, 0.5 * L, 0.5 * L)]
    t_end = 8.0e-9
    sigs = []
    for ext in (0, 3 * n_int // 2 + n_pml):
        s = s if ext == 0 else _box_f(n_int, n_pml, ext, patch, lts, dt, pml_kw, through)
        s.reset()
        s.sources, s.probes = [], []
        s.add_current(src_pt, 2, lambda t: -(t - 4 * tau) / tau * np.exp(-(((t - 4 * tau) / tau) ** 2)))
        for p in prb:
            s.add_probe(p, 2)
        s.run(int(t_end / dt))
        sigs.append(np.array([p["values"] for p in s.probes]))
    err = np.abs(sigs[0] - sigs[1]).max(axis=1) / np.abs(sigs[1]).max(axis=1)
    return {"n_pml": n_pml, "variant": variant, "dt": dt, "err_probes": err.tolist(),
            "err": float(err.max()), "t": (np.arange(sigs[0].shape[1]) * dt).tolist(),
            "sig_open": sigs[0][0].tolist(), "sig_ref": sigs[1][0].tolist()}  # fmt: skip


def task_growth(alpha_hz, n_steps=20000):
    n_int, n_pml = 20, 6
    s = _box_f(n_int, n_pml, 0, True, False, None,
               dict(allow_patches=True, alpha_hz=alpha_hz), through=True)  # fmt: skip
    rng = np.random.default_rng(0)
    s.e[:] = rng.standard_normal(s.n_free)
    a0 = np.abs(s.e).max()
    hist = []
    with np.errstate(all="ignore"):
        for k in range(n_steps):
            s.one_step()
            if k % 200 == 199:
                a = float(np.abs(s.e).max() / a0)
                hist.append(a)
                if not np.isfinite(a) or a > 1e12:
                    break
    return {"alpha_hz": alpha_hz, "dt": s.dt, "every": 200, "max_e": hist}


# ------------------------------------------------------------------ part G
def gpu_timing(res_d, n_steps=200):
    import cupy as cp

    from wakis import GridFIT3D, SolverFIT3D

    out = []
    rng = np.random.default_rng(0)
    for r in res_d:
        row = {"kind": r["kind"], "h_coarse": r["h_coarse"], "ratio": r["ratio"]}
        if r["kind"] == "uniform":
            axes = [np.linspace(0, L, int(round(L / r["h_coarse"])) + 1) for L in BOX]
            for fused in (False, True):
                g = GridFIT3D(x=axes[0], y=axes[1], z=axes[2], verbose=0)
                sv = SolverFIT3D(g, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0,
                                 cfln=0.95, use_gpu=True, fused=fused)  # fmt: skip
                for _ in range(20):
                    sv.one_step()
                cp.cuda.Device().synchronize()
                t0 = time.perf_counter()
                for _ in range(n_steps):
                    sv.one_step()
                cp.cuda.Device().synchronize()
                row["solverfit3d_fused" if fused else "solverfit3d"] = (
                    time.perf_counter() - t0
                ) / n_steps
                row["solverfit3d_dt"] = float(sv.dt)
                del sv, g
        for kernels in ("block", "csr"):
            s = make_solver("big", r["kind"], r["h_coarse"], r["ratio"], dt=r["dt"],
                            kernels=kernels, backend="cupy")  # fmt: skip
            s.e[:] = cp.asarray(rng.standard_normal(s.n_free))
            for _ in range(10):
                s.one_step()
            s.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_steps):
                s.one_step()
            s.synchronize()
            row[f"subgrid_{kernels}"] = (time.perf_counter() - t0) / n_steps
            del s
        cp.get_default_memory_pool().free_all_blocks()
        out.append(row)
        print("gpu", row, flush=True)
    return out


def gpu_vs_cpu(n_steps=200):
    import cupy as cp

    s_c = make_solver("big", "lts", 0.05, 2)
    s_g = make_solver("big", "lts", 0.05, 2, dt=s_c.dt, kernels="block", backend="cupy")
    e0 = np.random.default_rng(1).standard_normal(s_c.n_free)
    s_c.e[:] = e0
    s_g.e[:] = cp.asarray(e0)
    s_c.run(n_steps)
    s_g.run(n_steps)
    err = float(np.abs(s_g.e_host() - s_c.e).max() / np.abs(s_c.e).max())
    rho = float(min(s_c.stability_ratio(), 0.999))
    mK = int(np.diff(s_c.K.indptr).max())
    mKH = int(np.diff(s_c.KH.tocsr().indptr).max())
    bound = n_steps * np.finfo(float).eps / 2 * (mK + mKH) * np.sqrt((1 + rho) / (1 - rho))
    return {"n_steps": n_steps, "rel_err": err, "bound": float(bound), "rho": rho,
            "mK": mK, "mKH": mKH, "n_free": int(s_c.n_free)}  # fmt: skip


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="D")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nproc", type=int, default=os.cpu_count())
    ap.add_argument("--results-d", default=os.path.join(HERE, "results_D.json"))
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, f"results_{a.parts}.json")
    res = {"meta": {"host": platform.node(), "cpus": os.cpu_count(), "quick": a.quick,
                    "numpy": np.__version__, "started": time.ctime(), "parts": a.parts}}  # fmt: skip
    t0 = time.time()
    if "D" in a.parts or "E" in a.parts:
        case = "big" if "D" in a.parts else "small"
        f1 = guess_f1(case)
        res["f1_guess"] = f1
        jobs = jobs_cost(case, a.quick, f1)
        with Pool(a.nproc) as pool:
            res["cost"] = pool.map(task_cost, jobs, chunksize=1)
        print(f"cost done {time.time() - t0:.0f} s", flush=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        res["timing"] = timing_serial(res["cost"])
    if "F" in a.parts:
        thick = [4, 6] if a.quick else [4, 6, 8, 12]
        jobs = [(n, v) for n in thick for v in ("nopatch", "patch_global", "patch_lts", "through")]
        with Pool(a.nproc) as pool:
            res["cpml"] = pool.map(task_cpml, jobs, chunksize=1)
            alphas = [0.0, 1e9] if a.quick else [0.0, 1e8, 3e8, 1e9, 3e9]
            res["growth"] = pool.map(task_growth, alphas, chunksize=1)
    if "G" in a.parts:
        with open(a.results_d) as f:
            rd = json.load(f)["cost"]
        if a.quick:
            rd = [r for r in rd if r["h_coarse"] >= 0.05]
        res["gpu_vs_cpu"] = gpu_vs_cpu()
        print("gpu_vs_cpu", res["gpu_vs_cpu"], flush=True)
        res["gpu"] = gpu_timing(rd)
    res["meta"]["finished"] = time.ctime()
    res["meta"]["wall_s"] = time.time() - t0
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
