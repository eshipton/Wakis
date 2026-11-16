"""
Study script for 3D tetrahedral local time stepping (`wakis.unstructured.lts3d`).

Parts:
- `preflight`: memory estimates via `wakis.unstructured.estimate.estimate_memory`
- `bounds`: element stability bound distributions (`dt_T / dt_global`) and predicted work curves
- `accuracy`: spatial (`eigenmodes`) and time-domain (`harminv`) resonance error across cases/arms
- `timing`: single-process interleaved wall-time measurements (5 repetitions, median and min-max)
- `stability`: `rho` and level margins vs `dt / dt_global`, plus energy trajectories (stable, mutated, unstable)

With `--quick`, runs a lightweight smoke configuration on tiny meshes and writes
`results/*_quick.json` without touching full study outputs.
"""

import argparse
import json
import os
import platform
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy
from scipy.constants import c as C0

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CAD_DIR = os.path.join(ROOT, "studies", "unstructured-cadmesh")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wakis.resonances import harminv
from wakis.unstructured import UnstructuredFIT3D
from wakis.unstructured.estimate import estimate_memory
from wakis.unstructured.improve import improve_dt
from wakis.unstructured.lts3d import (
    TetLTS,
    element_levels,
    element_stability,
    global_limit_ratio,
    is_stable,
    lts_dt_max,
    modified_energy,
    theta_factor,
)
from wakis.unstructured.meshers import box_mesh

BOX_A, BOX_B, BOX_C = 1.0, 0.75, 0.5
F_TM110 = 0.5 * C0 * float(np.sqrt(1.0 / BOX_A**2 + 1.0 / BOX_B**2))


def _env_metadata():
    """Collect CPU model, library versions, and thread environment variables."""
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if "model name" in line:
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    threads = {
        k: os.environ.get(k, "")
        for k in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "cpu_model": cpu,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "threads": threads,
    }


def _dump(name, payload, quick=False):
    os.makedirs(RES, exist_ok=True)
    suffix = "_quick" if quick else ""
    path = os.path.join(RES, f"{name}{suffix}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path


def _graded_xs(a, h):
    """Geometrically graded 1D node array on [0, a] with minimum spacing h/4 at x = 0."""
    n = max(4, round(a / h) + 1)
    s = np.linspace(0.0, 1.0, n)
    # Choose power alpha so first interval is ~ h / 4
    alpha = max(1.0, np.log(0.25 * h / a) / np.log(1.0 / (n - 1)))
    return a * (s**alpha)


def build_case_mesh(case, size_param, seed=1, use_improve=False):
    """
    Build mesh and return `(mesh, materials, f_analytic_hz, f_target_hz, probe_pt)`.
    """
    if case in ("C1_graded_box", "C2_uniform_box"):
        h = float(size_param)
        ny = max(4, round(BOX_B / h) + 1)
        nz = max(4, round(BOX_C / h) + 1)
        ys = np.linspace(0.0, BOX_B, ny)
        zs = np.linspace(0.0, BOX_C, nz)
        if case == "C1_graded_box":
            xs = _graded_xs(BOX_A, h)
        else:
            nx = max(4, round(BOX_A / h) + 1)
            xs = np.linspace(0.0, BOX_A, nx)
        m = box_mesh(xs, ys, zs, jitter=0.2, seed=seed)
        if use_improve:
            m, _ = improve_dt(m, n_sweeps=6)
        return m, None, F_TM110, 0.85 * F_TM110, [[0.48, 0.36, 0.24]]

    if CAD_DIR not in sys.path:
        sys.path.insert(0, CAD_DIR)
    from cadmesh_study import make_mesh, pillbox_exact

    cpw = int(size_param)
    opt = "none" if case == "C3_pillbox_delaunay" else "gmsh"
    imp = {"n_sweeps": 10} if use_improve else None
    m, info = make_mesh("pillbox", cpw, optimize=opt, improve=imp)
    f_tm010 = float(pillbox_exact()[0])
    return m, info.materials, f_tm010, 0.7e9, [[0.032, 0.021, 0.048]]


def _has_gmsh():
    try:
        import gmsh  # noqa: F401

        return True
    except ImportError:
        return False


def run_preflight(quick=False):
    """Estimate memory for all configured ladders before allocating."""
    sizes_box = [0.22, 0.16] if quick else [0.16, 0.12, 0.09, 0.07]
    cpws = [10] if quick else [10, 15, 20, 25]
    records = []
    for case in ("C1_graded_box", "C2_uniform_box"):
        for h in sizes_box:
            n_est = int(6.0 * (BOX_A * BOX_B * BOX_C) / (h**3))
            if case == "C1_graded_box":
                n_est = int(1.8 * n_est)
            mem_td = estimate_memory(n_est, dim=3, eigen=False)
            mem_eig = estimate_memory(n_est, dim=3, eigen=True)
            records.append(
                {
                    "case": case,
                    "size": h,
                    "n_tets_est": n_est,
                    "mem_td_bytes": mem_td,
                    "mem_eig_bytes": mem_eig,
                }
            )
    for case in ("C3_pillbox_delaunay", "C4_pillbox_gmsh"):
        for cpw in cpws:
            n_est = int(1.2 * (cpw**3))
            records.append(
                {
                    "case": case,
                    "size": cpw,
                    "n_tets_est": n_est,
                    "mem_td_bytes": estimate_memory(n_est, dim=3, eigen=False),
                    "mem_eig_bytes": estimate_memory(n_est, dim=3, eigen=True),
                }
            )
    out = {"env": _env_metadata(), "preflight": records}
    _dump("preflight", out, quick=quick)
    return out


def run_bounds(quick=False):
    """Part `bounds`: element dt_T distributions and predicted work ratios vs coarse dt."""
    h = 0.22 if quick else 0.11
    cases = [("C1_graded_box", h, 1), ("C2_uniform_box", h, 1)]
    if not quick and _has_gmsh():
        cases.append(("C3_pillbox_delaunay", 15, 1))

    items = []
    mults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.8]
    for case, sz, seed in cases:
        for hodge in ("voronoi", "whitney"):
            m, mats, _, _, _ = build_case_mesh(case, sz, seed=seed)
            assert estimate_memory(m, eigen=False) < 2.0e9
            s = UnstructuredFIT3D(m, mats, hodge=hodge)
            st = element_stability(s)
            dt_g = float(s.dt_max)
            dt_t_ratio = (st["dt_T"] / dt_g).tolist()

            # Level partition snapshot at 2.2 * dt_global for [2, 2]
            ratios = [2, 2]
            nus = [0.05, 0.10]
            ind = "element" if hodge == "voronoi" else "edge_rate"
            lvl = element_levels(s, 2.2 * dt_g, ratios, nus, indicator=ind, stab=st)
            lts_22 = TetLTS(s, 2.2 * dt_g, lvl, ratios, nu=nus)
            cell_lvl = np.max(
                np.where(
                    np.isin(m.cell_edges, s.free),
                    lvl[
                        np.searchsorted(
                            s.free, np.clip(m.cell_edges, s.free[0], s.free[-1])
                        )
                    ],
                    0,
                ),
                axis=1,
            )

            work_G = float(
                TetLTS(
                    s, 0.95 * dt_g, np.zeros(s.free.size, dtype=int), []
                ).work_per_step()["work_per_s"]
            )
            work_curve = []
            for mult in mults:
                dt_c = mult * 0.95 * dt_g
                r_c = [2] if mult <= 1.95 else [2, 2]
                n_c = [0.05 * lv for lv in range(1, len(r_c) + 1)]
                lv_c = element_levels(s, dt_c, r_c, n_c, indicator=ind, stab=st)
                op_c = TetLTS(s, dt_c, lv_c, r_c, nu=n_c)
                wps = op_c.work_per_step()["work_per_s"]
                work_curve.append(
                    {
                        "dt_over_dt_global": float(dt_c / dt_g),
                        "ratios": r_c,
                        "f1_frac": float(np.mean(lv_c >= 1)),
                        "f2_frac": float(np.mean(lv_c >= 2)),
                        "work_ratio_vs_G": float(work_G / wps),
                    }
                )

            items.append(
                {
                    "case": case,
                    "size": sz,
                    "seed": seed,
                    "hodge": hodge,
                    "n_tets": int(m.n_cells),
                    "n_free": int(s.free.size),
                    "n_massless": int(s.massless.size),
                    "dt_global": dt_g,
                    "tightness": float(st["tightness"]),
                    "theta_2": float(theta_factor([2], [0.05])),
                    "theta_22": float(theta_factor([2, 2], [0.05, 0.10])),
                    "dt_T_over_dt_global": dt_t_ratio,
                    "summary_22": lts_22.summary(),
                    "cell_levels_22": cell_lvl.tolist(),
                    "work_curve": work_curve,
                }
            )
    out = {"env": _env_metadata(), "bounds": items}
    _dump("bounds", out, quick=quick)
    return out


def _lts_mode_near(lts, lam_target, x_ref):
    from scipy.sparse.linalg import LinearOperator, eigsh, splu

    S0, M = lts.stiffness()
    S0 = 0.5 * (S0 + S0.T)
    lu = splu((S0 - lam_target * M).tocsc())
    n = lts.n
    op_inv = LinearOperator((n, n), matvec=lu.solve, dtype=float)
    v0 = np.random.default_rng(7).standard_normal(n)
    vals, vecs = eigsh(
        S0,
        k=min(4, n - 2),
        M=M,
        sigma=lam_target,
        which="LM",
        OPinv=op_inv,
        v0=v0,
        tol=1e-10,
    )
    overlaps = [abs(float(vecs[:, i] @ (M @ x_ref))) for i in range(vals.size)]
    return float(np.real(vals[int(np.argmax(overlaps))]))


def _eval_arm_task(args):
    """Run one (case, size, seed, arm) accuracy and cost evaluation."""
    case, sz, seed, arm, quick = args
    use_imp = arm in ("G-imp", "L-imp")
    m, mats, f_analytic, f_target, probe_pt = build_case_mesh(
        case, sz, seed=seed, use_improve=use_imp
    )
    assert estimate_memory(m, eigen=True) < 2.0e9

    hodge = "whitney" if arm.startswith(("GW", "LW")) else "voronoi"
    ml_tol = 1e-2
    if arm == "G-ml3e-2":
        ml_tol = 3e-2
    elif arm == "G-ml1e-1":
        ml_tol = 1e-1

    s_base = UnstructuredFIT3D(m, mats, hodge=hodge, massless_tol=ml_tol)
    dt_global = float(s_base.dt_max)

    f_modes, e_modes = s_base.eigenmodes(n_modes=4, f_target=f_target)
    idx = int(np.argmin(np.abs(f_modes - f_analytic)))
    f_semi = float(f_modes[idx])
    err_semi = float(abs(f_semi / f_analytic - 1.0))
    x_mode = e_modes[s_base.free, idx]
    lam_semi = (2.0 * np.pi * f_semi) ** 2

    t_setup0 = time.perf_counter()
    if arm in ("G", "G-imp", "G-ml3e-2", "G-ml1e-1", "GW"):
        s = UnstructuredFIT3D(m, mats, hodge=hodge, massless_tol=ml_tol, cfl=0.95)
        setup_s = {"bounds": 0.0, "certificate": 0.0, "bisection": 0.0}
        f_pred = float(np.arcsin(0.5 * s.dt * np.sqrt(lam_semi)) / (np.pi * s.dt))
    elif arm in ("L", "L-imp"):
        s = UnstructuredFIT3D(
            m, mats, hodge=hodge, massless_tol=ml_tol, lts={"p": 2, "dt": "auto"}
        )
        setup_s = s.lts_info["setup_time_s"]
        lam_0 = (
            _lts_mode_near(s.lts, lam_semi, x_mode) if m.n_cells < 4000 else lam_semi
        )
        f_pred = float(np.arcsin(0.5 * s.dt * np.sqrt(lam_0)) / (np.pi * s.dt))
    elif arm.startswith("L-dt"):
        mult = float(arm.replace("L-dt", ""))
        ratios = [2] if mult <= 1.9 else [2, 2]
        nus = [0.05 * lv for lv in range(1, len(ratios) + 1)]
        dt_req = min(
            mult * 0.95 * dt_global, 0.95 * global_limit_ratio(ratios, nus) * dt_global
        )
        s = UnstructuredFIT3D(
            m,
            mats,
            hodge=hodge,
            massless_tol=ml_tol,
            lts={"ratios": ratios, "dt": dt_req},
        )
        setup_s = s.lts_info["setup_time_s"]
        lam_0 = (
            _lts_mode_near(s.lts, lam_semi, x_mode) if m.n_cells < 4000 else lam_semi
        )
        f_pred = float(np.arcsin(0.5 * s.dt * np.sqrt(lam_0)) / (np.pi * s.dt))
    elif arm in ("LW-element", "LW-edge_rate"):
        ind = "element" if arm == "LW-element" else "edge_rate"
        s = UnstructuredFIT3D(
            m,
            mats,
            hodge="whitney",
            lts={"p": 2, "dt": "auto", "indicator": ind},
        )
        setup_s = s.lts_info["setup_time_s"]
        lam_0 = (
            _lts_mode_near(s.lts, lam_semi, x_mode) if m.n_cells < 4000 else lam_semi
        )
        f_pred = float(np.arcsin(0.5 * s.dt * np.sqrt(lam_0)) / (np.pi * s.dt))
    else:
        raise ValueError(f"Unknown arm: {arm}")
    t_setup_total = time.perf_counter() - t_setup0

    n_periods = 42.0 if not quick else 40.0
    n_steps = max(20, int(np.ceil(n_periods / (f_analytic * s.dt))))
    pr = s.add_probe(probe_pt, component=2)
    s.u[:] = x_mode
    t0 = time.perf_counter()
    s.run(n_steps)
    wall = time.perf_counter() - t0
    sim_ns = float(n_steps * s.dt * 1e9)
    wall_per_ns = float(wall / sim_ns)

    sig = s.probe_signal(pr)[:, 0]
    res = harminv(
        sig, s.dt, fmin=0.75 * f_analytic, fmax=1.25 * f_analytic, method="pencil"
    )
    if len(res.frequency):
        f_td = float(res.frequency[np.argmin(np.abs(res.frequency - f_analytic))])
    else:
        f_td = float("nan")
    err_td = float(abs(f_td / f_analytic - 1.0))

    if s.lts is not None:
        summ = s.lts.summary()
        work_per_ns = float(summ["work"]["work_per_s"] * 1e-9)
        rho = float(s.lts_info["rho"])
        margins = [float(x) for x in s.lts_info["margins"]]
        ratios_used = summ["ratios"]
        F_sizes = summ["F_sizes"]
        C_sizes = summ["C_sizes"]
        rows_nnz = summ["rows_nnz"]
    else:
        op0 = TetLTS(s, s.dt, np.zeros(s.free.size, dtype=int), [])
        work_per_ns = float(op0.work_per_step()["work_per_s"] * 1e-9)
        rho = float(s.courant)
        margins = []
        ratios_used = []
        F_sizes = []
        C_sizes = []
        rows_nnz = []

    return {
        "case": case,
        "size": sz,
        "seed": seed,
        "arm": arm,
        "hodge": hodge,
        "n_tets": int(m.n_cells),
        "n_free": int(s.free.size),
        "n_massless": int(s.massless.size),
        "dt_global": dt_global,
        "dt_lts": float(s.dt),
        "ratio": float(s.dt / (0.95 * dt_global)),
        "ratios": ratios_used,
        "F_sizes": F_sizes,
        "C_sizes": C_sizes,
        "rows_nnz": rows_nnz,
        "rho": rho,
        "margins": margins,
        "setup_time_s": setup_s,
        "setup_total_s": float(t_setup_total),
        "f_analytic": float(f_analytic),
        "f_semi": f_semi,
        "f_pred": f_pred,
        "f_td": f_td,
        "err_semi": err_semi,
        "err_td": err_td,
        "work_per_ns": work_per_ns,
        "wall_per_ns": wall_per_ns,
        "n_steps": int(n_steps),
    }


def _equal_error_gain(records, case, base_arm, target_arm):
    """
    Piecewise-linear log-log interpolation of `wall_per_ns` vs `err_td`
    inside the error overlap only (no extrapolation).
    """
    seeds = sorted({r["seed"] for r in records if r["case"] == case})
    seed_gains = []
    for sd in seeds:
        b = [
            r
            for r in records
            if r["case"] == case
            and r["seed"] == sd
            and r["arm"] == base_arm
            and np.isfinite(r["err_td"])
            and r["err_td"] > 0
        ]
        t = [
            r
            for r in records
            if r["case"] == case
            and r["seed"] == sd
            and r["arm"] == target_arm
            and np.isfinite(r["err_td"])
            and r["err_td"] > 0
        ]
        if len(b) < 2 or len(t) < 2:
            continue
        b.sort(key=lambda r: r["err_td"])
        t.sort(key=lambda r: r["err_td"])
        eb = np.log10([r["err_td"] for r in b])
        wb = np.log10([r["wall_per_ns"] for r in b])
        et = np.log10([r["err_td"] for r in t])
        wt = np.log10([r["wall_per_ns"] for r in t])
        lo = max(eb.min(), et.min())
        hi = min(eb.max(), et.max())
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, 5)
        w_b_interp = np.interp(grid, eb, wb)
        w_t_interp = np.interp(grid, et, wt)
        ratios = 10.0 ** (w_b_interp - w_t_interp)
        seed_gains.append(float(np.median(ratios)))
    if not seed_gains:
        return {"status": "NO_OVERLAP", "median_gain": None, "range": None}
    lo_g, hi_g = float(np.min(seed_gains)), float(np.max(seed_gains))
    med_g = float(np.median(seed_gains))
    status = (
        "UNRESOLVED" if (lo_g <= 1.0 <= hi_g) else ("GAIN" if med_g > 1.0 else "LOSS")
    )
    return {"status": status, "median_gain": med_g, "range": [lo_g, hi_g]}


def run_accuracy(quick=False, n_procs=1):
    """Part `accuracy`: evaluate accuracy and work across cases and arms."""
    if quick:
        sizes_box = [0.24, 0.18]
        seeds = [1]
        arms_box = ["G", "G-imp", "G-ml3e-2", "L", "L-dt1.5", "GW", "LW-edge_rate"]
    else:
        sizes_box = [0.16, 0.12, 0.09, 0.07]
        seeds = [1, 2, 3]
        arms_box = [
            "G",
            "G-imp",
            "G-ml3e-2",
            "G-ml1e-1",
            "L",
            "L-dt1.25",
            "L-dt1.5",
            "L-dt2.0",
            "L-dt3.0",
            "L-imp",
            "GW",
            "LW-element",
            "LW-edge_rate",
        ]

    tasks = []
    for case in ("C1_graded_box", "C2_uniform_box"):
        for sd in seeds:
            for sz in sizes_box:
                for arm in arms_box:
                    tasks.append((case, sz, sd, arm, quick))

    if not quick and _has_gmsh():
        cpws = [10, 15, 20, 25]
        for cpw in cpws:
            for arm in arms_box:
                tasks.append(("C3_pillbox_delaunay", cpw, 1, arm, quick))
            for arm in ("GW", "LW-element", "LW-edge_rate"):
                tasks.append(("C4_pillbox_gmsh", cpw, 1, arm, quick))

    if n_procs > 1 and not quick:
        with Pool(n_procs) as pool:
            records = pool.map(_eval_arm_task, tasks)
    else:
        records = [_eval_arm_task(t) for t in tasks]

    # Attach break-even simulated time [ns] against arm G on the same (case, size, seed)
    g_map = {(r["case"], r["size"], r["seed"]): r for r in records if r["arm"] == "G"}
    for r in records:
        g_ref = g_map.get((r["case"], r["size"], r["seed"]))
        if g_ref is not None:
            r["work_ratio_vs_G"] = float(g_ref["work_per_ns"] / r["work_per_ns"])
            r["wall_ratio_vs_G"] = float(g_ref["wall_per_ns"] / r["wall_per_ns"])
            saving = g_ref["wall_per_ns"] - r["wall_per_ns"]
            setup_tot = sum(r["setup_time_s"].values())
            r["break_even_ns"] = (
                float(setup_tot / saving) if saving > 0 else float("inf")
            )
        else:
            r["work_ratio_vs_G"] = None
            r["wall_ratio_vs_G"] = None
            r["break_even_ns"] = None

    comparisons = {}
    for case in sorted({r["case"] for r in records}):
        comparisons[case] = {
            "L_vs_G": _equal_error_gain(records, case, "G", "L"),
            "L_vs_G_imp": _equal_error_gain(records, case, "G-imp", "L"),
            "G_imp_vs_G": _equal_error_gain(records, case, "G", "G-imp"),
            "LW_edge_rate_vs_GW": _equal_error_gain(
                records, case, "GW", "LW-edge_rate"
            ),
        }

    out = {
        "env": _env_metadata(),
        "records": records,
        "comparisons": comparisons,
    }
    _dump("accuracy", out, quick=quick)
    return out


def run_timing(quick=False):
    """Part `timing`: interleaved single-process wall-time measurements (5 repetitions)."""
    h = 0.20 if quick else 0.10
    n_steps = 40 if quick else 150
    arms = ["G", "G-imp", "G-ml3e-2", "L", "L-dt1.5", "L-dt2.0", "GW", "LW-edge_rate"]
    m_raw, mats, _, _, _ = build_case_mesh(
        "C1_graded_box", h, seed=1, use_improve=False
    )
    m_imp, _, _, _, _ = build_case_mesh("C1_graded_box", h, seed=1, use_improve=True)
    dt_g = float(UnstructuredFIT3D(m_raw, mats, hodge="voronoi").dt_max)

    solvers = {}
    for arm in arms:
        mesh_use = m_imp if arm == "G-imp" else m_raw
        hodge = "whitney" if arm.startswith(("GW", "LW")) else "voronoi"
        ml_tol = 3e-2 if arm == "G-ml3e-2" else 1e-2
        if arm in ("G", "G-imp", "G-ml3e-2", "GW"):
            solvers[arm] = UnstructuredFIT3D(
                mesh_use, mats, hodge=hodge, massless_tol=ml_tol, cfl=0.95
            )
        elif arm == "L":
            solvers[arm] = UnstructuredFIT3D(
                mesh_use,
                mats,
                hodge=hodge,
                massless_tol=ml_tol,
                lts={"p": 2, "dt": "auto"},
            )
        elif arm.startswith("L-dt"):
            mult = float(arm.replace("L-dt", ""))
            ratios = [2] if mult <= 1.9 else [2, 2]
            solvers[arm] = UnstructuredFIT3D(
                mesh_use,
                mats,
                hodge=hodge,
                massless_tol=ml_tol,
                lts={"ratios": ratios, "dt": mult * 0.95 * dt_g},
            )
        elif arm == "LW-edge_rate":
            solvers[arm] = UnstructuredFIT3D(
                mesh_use,
                mats,
                hodge="whitney",
                lts={"p": 2, "dt": "auto", "indicator": "edge_rate"},
            )

    samples = {arm: [] for arm in arms}
    rng = np.random.default_rng(0)
    for _ in range(5):
        for arm in arms:
            s = solvers[arm]
            s.reset()
            s.u[:] = 1e-6 * rng.standard_normal(s.free.size)
            t0 = time.perf_counter()
            s.run(n_steps)
            el = time.perf_counter() - t0
            samples[arm].append(float(el / (n_steps * s.dt * 1e9)))

    rows = []
    g_med = float(np.median(samples["G"]))
    for arm in arms:
        s = solvers[arm]
        arr = np.asarray(samples[arm], dtype=float)
        med = float(np.median(arr))
        rows.append(
            {
                "case": "C1_graded_box",
                "size": h,
                "arm": arm,
                "dt_s": float(s.dt),
                "wall_per_ns_median": med,
                "wall_per_ns_min": float(arr.min()),
                "wall_per_ns_max": float(arr.max()),
                "speedup_vs_G": float(g_med / med),
            }
        )
    out = {"env": _env_metadata(), "timing": rows}
    _dump("timing", out, quick=quick)
    return out


def run_stability(quick=False):
    """Part `stability`: rho/margins vs dt/dt_global and energy conservation/instability curves."""
    h = 0.20 if quick else 0.14
    m, mats, _, _, _ = build_case_mesh("C1_graded_box", h, seed=3)
    s0 = UnstructuredFIT3D(m, mats, hodge="voronoi")
    ratios = [2]
    nus = [0.05]
    pred_ratio = float(global_limit_ratio(ratios, nus))
    lvl = element_levels(s0, 1.05 * pred_ratio * s0.dt_max, ratios, nus)

    sweep = []
    factors = np.linspace(0.6, 2.25, 12 if quick else 25)
    for fac in factors:
        dt_c = float(fac * s0.dt_max)
        lts_c = TetLTS(s0, dt_c, lvl, ratios, nu=nus)
        ok, rho, marg = is_stable(lts_c, tol=1e-6)
        sweep.append(
            {
                "dt_over_dt_global": float(fac),
                "rho": float(rho),
                "max_margin": float(max(marg, default=0.0)),
                "is_stable": bool(ok),
            }
        )

    dt_lts, _ = lts_dt_max(
        s0, 2.2 * s0.dt_max, rtol=1e-3, edge_level=lvl, ratios=ratios, nu=nus
    )
    n_steps = 200 if quick else 1000
    rng = np.random.default_rng(4)
    u_init = 1e-3 * rng.standard_normal(s0.free.size)

    def _trace(dt_use, mutate=None, max_steps=n_steps):
        s_run = UnstructuredFIT3D(
            m,
            mats,
            hodge="voronoi",
            dt=dt_use,
            lts={
                "ratios": ratios,
                "edge_level": lvl,
                "mutate": mutate,
                "_unsafe": True,
            },
        )
        s_run.u[:] = u_init
        u0_norm = float(np.linalg.norm(u_init))
        E0 = None
        drifts = []
        norms = []
        for _ in range(max_steps):
            up = s_run.u.copy()
            s_run.one_step()
            E, ek, ep = modified_energy(s_run, up, s_run.u, return_terms=True)
            if E0 is None:
                E0 = E
            denom = max(abs(E0), abs(ek) + abs(ep), 1e-30)
            drifts.append(float(abs(E - E0) / denom))
            gn = float(np.linalg.norm(s_run.u) / u0_norm)
            norms.append(gn)
            if gn > 1e5:
                break
        return {"drift": drifts, "norm_ratio": norms}

    out = {
        "env": _env_metadata(),
        "pred_global_limit_ratio": pred_ratio,
        "meas_dt_lts_over_dt_global": float(dt_lts / s0.dt_max),
        "sweep": sweep,
        "traces": {
            "stable": _trace(0.95 * dt_lts, mutate=None),
            "mut_interface": _trace(0.5 * s0.dt_max, mutate="interface"),
            "mut_drop_buffer": _trace(0.5 * s0.dt_max, mutate="drop_buffer"),
            "unstable": _trace(1.05 * dt_lts, mutate=None, max_steps=60),
        },
    }
    _dump("stability", out, quick=quick)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=["all", "preflight", "bounds", "accuracy", "timing", "stability"],
        default="all",
    )
    parser.add_argument("--quick", action="store_true", help="Run tiny smoke checks")
    parser.add_argument("--procs", type=int, default=1, help="Worker processes")
    args = parser.parse_args()

    summary = {"env": _env_metadata(), "quick": bool(args.quick)}
    if args.part in ("all", "preflight"):
        summary["preflight"] = run_preflight(quick=args.quick)["preflight"]
    if args.part in ("all", "bounds"):
        summary["bounds"] = run_bounds(quick=args.quick)["bounds"]
    if args.part in ("all", "accuracy"):
        acc = run_accuracy(quick=args.quick, n_procs=args.procs)
        summary["comparisons"] = acc["comparisons"]
    if args.part in ("all", "timing"):
        summary["timing"] = run_timing(quick=args.quick)["timing"]
    if args.part in ("all", "stability"):
        stab = run_stability(quick=args.quick)
        summary["stability"] = {
            "pred_global_limit_ratio": stab["pred_global_limit_ratio"],
            "meas_dt_lts_over_dt_global": stab["meas_dt_lts_over_dt_global"],
        }
    _dump("summary", summary, quick=args.quick)
    print(
        f"run_study ({'quick' if args.quick else 'full'}, part={args.part}) complete."
    )


if __name__ == "__main__":
    main()
