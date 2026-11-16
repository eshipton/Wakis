# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Adaptive tetrahedral FIT eigenmode study (`studies/unstructured-adapt/run_adapt_study.py`).

Usage
-----
    python studies/unstructured-adapt/run_adapt_study.py \
        --arm {box,uniform,residual,zz,random0,random1,random2,geometric,theta03,theta07,all} \
        [--quick] --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
import scipy
from scipy.constants import c as C0

from wakis.unstructured.adapt import (
    _P1_TET_MASS,
    adapt_eigen,
    kuhn_mesh,
    l_prism_mesh,
    refine_uniform,
    residual_indicator,
    voronoi_export,
    whitney_fields,
    zz_curl_indicator,
)
from wakis.unstructured.analytic import l_shape_dirichlet_eigenvalue
from wakis.unstructured.estimate import estimate_memory
from wakis.unstructured.solver3d import UnstructuredFIT3D

ALL_SINGLE_ARMS = (
    "box",
    "uniform",
    "residual",
    "zz",
    "random0",
    "random1",
    "random2",
    "geometric",
    "theta03",
    "theta07",
)

PREDICTIONS = {
    "P1": (
        "uniform refinement: |k_h^2 - k^2| prop N^(-4/9) asymptotically "
        "(|||e|||^2 ~ h^(4/3) from H ~ r^(-1/3); N prop h^-3); "
        "pre-asymptotic slopes between -4/9 and -2/3."
    ),
    "P2": (
        "Dorfler-adaptive (residual or ZZ): prop N^(-2/3) asymptotically. "
        "Equidistribution count: per-element error^2 ~ h^5 r^(2 alpha - 4), "
        "alpha = 2/3; h(r) ~ delta^(1/5) r^((4 - 2 alpha)/5); "
        "N ~ int h^-3 r dr is finite iff alpha > 1/3, so isotropic "
        "refinement of this edge singularity recovers the smooth rate."
    ),
    "P3": (
        "at equal N the uniform/adaptive error ratio grows like N^(2/9) "
        "(about 2.1 after a 30x growth from equal error)."
    ),
    "P4": (
        "random marking with the adaptive run's per-iteration counts "
        "behaves like uniform refinement and loses to Dorfler."
    ),
    "P5": (
        "a-priori geometric marking (distance to the re-entrant edge, same "
        "counts) is close to Dorfler on this geometry; the estimator's "
        "value is not needing to know where the singularity is."
    ),
    "P6": (
        "explicit time domain: the adapted mesh's Voronoi dt_max is smaller "
        "than that of the uniform mesh of equal eigenvalue error (dt prop "
        "h_min); report N/dt for both (adaptivity is not free in the time "
        "domain; LTS exists)."
    ),
    "P7": (
        "the residual effectivity eta/|||e||| on the box (§5 E3) is constant "
        "to within the level-ratio band across similar Kuhn meshes; "
        "eta_ZZ / ||curl e|| is close to 1 there."
    ),
}

# Deterministic n_marked counts of the residual Dorfler (theta=0.5) arm on
# l_prism_mesh(5, 3) when running random/geometric arms standalone before
# adapt_residual.json has been merged.
DEFAULT_RESIDUAL_MARK_COUNTS = [
    172,
    181,
    335,
    473,
    808,
    1309,
    2209,
    3753,
    6040,
    9914,
    16000,
    25000,
    40000,
]


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _total_ram_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (ValueError, OSError, AttributeError):
        pass
    return int(8e9)


def build_meta(quick: bool) -> dict:
    ram = _total_ram_bytes()
    budget = int(2e9) if quick else int(0.6 * ram)
    preflight = {
        str(n): float(estimate_memory(n, dim=3, eigen=True))
        for n in (1350, 10800, 40320, 172800)
    }
    return {
        "git_sha": _git_sha(),
        "machine_node": platform.node(),
        "machine_arch": platform.machine(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "ram_total_bytes": ram,
        "memory_budget_bytes": budget,
        "quick": bool(quick),
        "preflight_estimates_bytes": preflight,
    }


def compute_mps_reference() -> dict:
    lam24, sig24 = l_shape_dirichlet_eigenvalue(
        n_terms=24, n_boundary=160, n_interior=80, seed=0
    )
    lam32, sig32 = l_shape_dirichlet_eigenvalue(
        n_terms=32, n_boundary=160, n_interior=80, seed=0
    )
    f_ref_hz = float(np.sqrt(lam24) * C0 / (2.0 * np.pi))
    return {
        "n_terms": 24,
        "k2_per_m2": float(lam24),
        "sigma_min": float(sig24),
        "f_ref_hz": f_ref_hz,
        "n_terms_check": 32,
        "k2_check_per_m2": float(lam32),
        "sigma_min_check": float(sig32),
        "rel_diff_24_vs_32": float(abs(lam24 - lam32) / lam32),
    }


def _duffy_tet_rule(n_1d: int = 6):
    pts, wts = np.polynomial.legendre.leggauss(n_1d)
    t = 0.5 * (pts + 1.0)
    w = 0.5 * wts
    t1, t2, t3 = np.meshgrid(t, t, t, indexing="ij")
    w1, w2, w3 = np.meshgrid(w, w, w, indexing="ij")
    t1, t2, t3 = t1.ravel(), t2.ravel(), t3.ravel()
    w1, w2, w3 = w1.ravel(), w2.ravel(), w3.ravel()

    l1 = t1
    l2 = (1.0 - t1) * t2
    l3 = (1.0 - t1) * (1.0 - t2) * t3
    l0 = 1.0 - l1 - l2 - l3
    bary = np.column_stack([l0, l1, l2, l3])
    weights = w1 * w2 * w3 * ((1.0 - t1) ** 2) * (1.0 - t2) * 6.0
    return bary, weights


def _box_exact_errors(mesh, e_raw, a: float = 0.7, b: float = 0.5, d: float = 0.3):
    U_raw, B_raw = whitney_fields(mesh, e_raw)
    u_l2_T = mesh.cell_volume * np.einsum("nik,ij,njk->n", U_raw, _P1_TET_MASS, U_raw)
    inv_norm = 1.0 / float(np.sqrt(np.sum(u_l2_T)))
    U_vert = U_raw * inv_norm
    B_cell = B_raw * inv_norm
    bary, qw = _duffy_tet_rule(6)
    x_q = np.einsum("qk,nki->nqi", bary, mesh.nodes[mesh.cells])
    u_h_q = np.einsum("qk,nki->nqi", bary, U_vert)

    scale = 2.0 / np.sqrt(a * b * d)
    kx = np.pi / a
    ky = np.pi / b
    k2_exact = float(kx**2 + ky**2)

    sx = np.sin(kx * x_q[..., 0])
    cx = np.cos(kx * x_q[..., 0])
    sy = np.sin(ky * x_q[..., 1])
    cy = np.cos(ky * x_q[..., 1])

    u_ex_z = scale * sx * sy
    vol_w = mesh.cell_volume[:, None] * qw[None, :]
    overlap = float(np.sum(vol_w * u_ex_z * u_h_q[..., 2]))
    sgn = 1.0 if overlap >= 0.0 else -1.0
    u_ex_z *= sgn

    curl_ex_x = sgn * scale * ky * sx * cy
    curl_ex_y = -sgn * scale * kx * cx * sy

    du_x = -u_h_q[..., 0]
    du_y = -u_h_q[..., 1]
    du_z = u_ex_z - u_h_q[..., 2]
    err_u_l2_sq = float(np.sum(vol_w * (du_x**2 + du_y**2 + du_z**2)))

    dc_x = curl_ex_x - B_cell[:, None, 0]
    dc_y = curl_ex_y - B_cell[:, None, 1]
    dc_z = -B_cell[:, None, 2]
    err_curl_l2_sq = float(np.sum(vol_w * (dc_x**2 + dc_y**2 + dc_z**2)))

    err_energy_sq = err_curl_l2_sq + k2_exact * err_u_l2_sq
    return err_u_l2_sq, err_curl_l2_sq, err_energy_sq, k2_exact


def run_box_arm(quick: bool, memory_budget_bytes: int) -> dict:
    a, b, d = 0.7, 0.5, 0.3
    k2_exact = float((np.pi / a) ** 2 + (np.pi / b) ** 2)
    f_exact = float(np.sqrt(k2_exact) * C0 / (2.0 * np.pi))
    base_mesh = kuhn_mesh((7, 5, 3), 0.1)
    levels_gen = (0, 3) if quick else (0, 3, 6)

    levels = []
    cur_mesh = base_mesh
    cur_gen = 0
    for idx, gen in enumerate(levels_gen):
        if gen > cur_gen:
            cur_mesh = refine_uniform(cur_mesh, generations=gen - cur_gen)
            cur_gen = gen
        if cur_mesh.n_cells > 20000:
            est = estimate_memory(cur_mesh, eigen=True)
            if est > memory_budget_bytes:
                print(
                    f"[box L{idx}] skipping n_cells={cur_mesh.n_cells}: "
                    f"estimated {est:.3e} B > budget {memory_budget_bytes:.3e} B"
                )
                break
        t0 = time.perf_counter()
        solver = UnstructuredFIT3D(cur_mesh, bc="pec", hodge="whitney", stability=False)
        freqs, modes = solver.eigenmodes(n_modes=3, f_target=0.95 * f_exact)
        t_solve = time.perf_counter() - t0

        f0 = float(freqs[0])
        f1 = float(freqs[1])
        kh2 = float((2.0 * np.pi * f0 / C0) ** 2)
        e0 = modes[:, 0]

        eta2_res = residual_indicator(cur_mesh, e0, kh2, bc="pec")
        eta2_zz = zz_curl_indicator(cur_mesh, e0)
        eta2_res_sum = float(np.sum(eta2_res))
        eta2_zz_sum = float(np.sum(eta2_zz))

        _, err_curl_sq, err_energy_sq, _ = _box_exact_errors(cur_mesh, e0, a, b, d)
        theta_res = float(np.sqrt(eta2_res_sum / err_energy_sq))
        theta_zz = float(np.sqrt(eta2_zz_sum / err_curl_sq))

        levels.append(
            {
                "level": f"L{idx}",
                "generations": int(gen),
                "n_cells": int(cur_mesh.n_cells),
                "n_dofs": int(solver.free.size),
                "h_max_m": float(np.max(cur_mesh.edge_length)),
                "f0_hz": f0,
                "f1_hz": f1,
                "k2_per_m2": kh2,
                "k2_exact_per_m2": k2_exact,
                "rel_err_signed": float((kh2 - k2_exact) / k2_exact),
                "abs_err_k2_per_m2": float(abs(kh2 - k2_exact)),
                "eta2_res_per_m2": eta2_res_sum,
                "eta_res_per_m": float(np.sqrt(eta2_res_sum)),
                "err_energy_sq_per_m2": err_energy_sq,
                "err_energy_per_m": float(np.sqrt(err_energy_sq)),
                "theta_res": theta_res,
                "eta2_zz_per_m2": eta2_zz_sum,
                "eta_zz_per_m": float(np.sqrt(eta2_zz_sum)),
                "err_curl_sq_per_m2": err_curl_sq,
                "err_curl_per_m": float(np.sqrt(err_curl_sq)),
                "theta_zz": theta_zz,
                "t_solve_s": float(t_solve),
            }
        )

    ratios = {}
    for i in range(1, len(levels)):
        prev_l = levels[i - 1]
        curr_l = levels[i]
        key = f"{curr_l['level']}_over_{prev_l['level']}"
        ratios[key] = {
            "theta_res_ratio": float(curr_l["theta_res"] / prev_l["theta_res"]),
            "theta_zz_ratio": float(curr_l["theta_zz"] / prev_l["theta_zz"]),
        }

    return {
        "arm": "box",
        "levels": levels,
        "level_ratios": ratios,
    }


def fit_log_slope(history: list[dict], i_start: int, i_end: int) -> dict:
    """
    Least-squares fit of log10(|rel_err_signed|) vs log10(n_dofs) on history[i_start:i_end+1].

    Flags any sign change in `rel_err_signed` and refuses to fit across a sign change.
    """
    sub = history[i_start : i_end + 1]
    signs = [np.sign(h.get("rel_err_signed", 0.0)) for h in sub]
    nonzero_signs = [s for s in signs if s != 0]
    sign_change = len(set(nonzero_signs)) > 1

    # Detect all sign change pairs across the full history
    full_signs = [np.sign(h.get("rel_err_signed", 0.0)) for h in history]
    sign_flip_iters = [
        i for i in range(1, len(full_signs)) if full_signs[i] * full_signs[i - 1] < 0
    ]

    if sign_change:
        # Trim to the contiguous tail with the final sign
        last_sign = full_signs[i_end]
        tail_start = i_end
        while tail_start > i_start and full_signs[tail_start - 1] == last_sign:
            tail_start -= 1
        if i_end - tail_start < 1:
            return {
                "slope": None,
                "slope_se": None,
                "intercept": None,
                "index_range": [int(i_start), int(i_end)],
                "dof_range": [
                    int(history[i_start]["n_dofs"]),
                    int(history[i_end]["n_dofs"]),
                ],
                "sign_change_in_requested_range": True,
                "sign_change_iterations": sign_flip_iters,
                "note": "Sign change in rel_err_signed; fewer than 2 same-sign tail points.",
            }
        i_start = tail_start
        sub = history[i_start : i_end + 1]

    x = np.log10([float(h["n_dofs"]) for h in sub])
    y = np.log10([max(abs(float(h["rel_err_signed"])), 1e-30) for h in sub])
    n = len(x)
    if n < 2:
        return {
            "slope": None,
            "slope_se": None,
            "intercept": None,
            "index_range": [int(i_start), int(i_end)],
            "dof_range": [int(sub[0]["n_dofs"]), int(sub[-1]["n_dofs"])],
            "sign_change_in_requested_range": bool(sign_change),
            "sign_change_iterations": sign_flip_iters,
        }

    xm = float(np.mean(x))
    ym = float(np.mean(y))
    sxx = float(np.sum((x - xm) ** 2))
    slope = float(np.sum((x - xm) * (y - ym)) / sxx)
    intercept = float(ym - slope * xm)
    if n > 2 and sxx > 0.0:
        resid = y - (slope * x + intercept)
        s2 = float(np.sum(resid**2) / (n - 2))
        slope_se = float(np.sqrt(s2 / sxx))
    else:
        slope_se = 0.0

    return {
        "slope": slope,
        "slope_se": slope_se,
        "intercept": intercept,
        "index_range": [int(i_start), int(i_end)],
        "dof_range": [int(sub[0]["n_dofs"]), int(sub[-1]["n_dofs"])],
        "sign_change_in_requested_range": bool(sign_change),
        "sign_change_iterations": sign_flip_iters,
    }


def _enrich_history(history: list[dict]) -> list[dict]:
    cum_t = 0.0
    out = []
    for i, h in enumerate(history):
        d = dict(h)
        d["iteration"] = i
        cum_t += float(d["t_total_s"])
        d["cum_wall_s"] = cum_t
        if "rel_err_signed" in d:
            d["abs_rel_err"] = abs(float(d["rel_err_signed"]))
            d["eigenvalue_effectivity"] = float(
                d["eta2_per_m2"] / max(abs(d["k2_per_m2"] * d["rel_err_signed"]), 1e-30)
            )
        out.append(d)
    return out


def _summarize_voronoi(mesh, rel_err_signed: float, n_dofs: int) -> dict:
    _, rep = voronoi_export(mesh, bc="pec")
    dt_max = rep["dt_max_s"]
    dt_rate = rep["dt_rate_s"]
    has_voronoi_dt = bool(np.isfinite(dt_max))
    dt_used = float(dt_max) if has_voronoi_dt else float(dt_rate)
    return {
        "n_cells": int(mesh.n_cells),
        "n_dofs": int(n_dofs),
        "rel_err_signed": float(rel_err_signed),
        "abs_rel_err": abs(float(rel_err_signed)),
        "negative_faces_before": int(rep["negative_faces_before"]),
        "negative_faces_after": int(rep["negative_faces_after"]),
        "dt_max_s": float(dt_max) if has_voronoi_dt else None,
        "dt_rate_s": float(dt_rate),
        "dt_used_s": dt_used,
        "dt_source": "voronoi_solver" if has_voronoi_dt else "improve_dt_rate",
        "n_dofs_per_dt_s": float(n_dofs / dt_used) if dt_used > 0 else None,
        "error": rep["error"],
    }


def _extract_top_face_data(mesh, e_final, f_final_hz, z_top: float = 0.6) -> dict:
    """Extract the z = z_top boundary triangulation and per-triangle h_T, eta_T."""
    kh2 = float((2.0 * np.pi * f_final_hz / C0) ** 2)
    eta2_T = residual_indicator(mesh, e_final, kh2, bc="pec")
    eta_T = np.sqrt(np.maximum(eta2_T, 0.0))

    pairs = mesh.cells[:, adapt_loc_edges()]
    d_vec = mesh.nodes[pairs[..., 1]] - mesh.nodes[pairs[..., 0]]
    h_T = np.sqrt(np.max(np.einsum("nci,nci->nc", d_vec, d_vec), axis=1))

    bfaces = mesh.boundary_faces
    f_nodes = mesh.faces[bfaces]
    f_z = mesh.nodes[f_nodes, 2]
    on_top = np.all(np.abs(f_z - z_top) < 1e-9, axis=1)
    top_faces = bfaces[on_top]
    top_tris_global = mesh.faces[top_faces]
    bcell = np.zeros(mesh.n_faces, dtype=np.int64)
    bcell[mesh.cell_faces.ravel()] = np.repeat(
        np.arange(mesh.n_cells, dtype=np.int64), 4
    )
    parent_cells = bcell[top_faces]

    u_nodes, inv = np.unique(top_tris_global.ravel(), return_inverse=True)
    xy = np.round(mesh.nodes[u_nodes, :2], 6).tolist()
    tris = inv.reshape(-1, 3).tolist()
    h_top = [float(f"{val:.6g}") for val in h_T[parent_cells]]
    eta_top = [float(f"{val:.6g}") for val in eta_T[parent_cells]]

    return {
        "z_m": float(z_top),
        "n_triangles": len(top_faces),
        "xy_m": xy,
        "triangles": tris,
        "h_T_m": h_top,
        "eta_T_per_m": eta_top,
    }


def adapt_loc_edges():
    from wakis.unstructured.adapt import _LOC_E

    return _LOC_E


def _load_or_default_mark_counts(outdir: str, quick: bool) -> list[int]:
    suffix = "_quick" if quick else ""
    res_path = os.path.join(outdir, f"adapt_residual{suffix}.json")
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as fh:
            data = json.load(fh)
        hist = data.get("arms", {}).get("residual", {}).get("history", [])
        counts = [int(h["n_marked"]) for h in hist if int(h.get("n_marked", 0)) > 0]
        if counts:
            return counts
    return list(DEFAULT_RESIDUAL_MARK_COUNTS)


def run_l_prism_arm(
    arm: str,
    quick: bool,
    k2_ref: float,
    memory_budget_bytes: int,
    outdir: str,
) -> dict:
    f_ref = float(np.sqrt(k2_ref) * C0 / (2.0 * np.pi))
    f_target = 0.9 * f_ref
    mesh0 = l_prism_mesh(5, 3, height_m=0.6)

    # Target DOFs: 3 uniform generations in --quick (12,054 DOFs, 10,800 cells),
    # or 2.5e5 DOFs in full mode.
    max_dofs_adapt = 12054 if quick else 250000
    max_iter_adapt = 12 if quick else 25

    extra = {}
    if arm == "uniform":
        n_gen = 3 if quick else 7
        # Track meshes at each generation to run voronoi_export and repeat timing
        gen_meshes = []

        def _cb(it, mesh, _e_vec, h):
            gen_meshes.append((it, mesh, dict(h)))

        res = adapt_eigen(
            mesh0,
            bc="pec",
            f_target_hz=f_target,
            mode=0,
            marking="uniform",
            max_iter=n_gen + 1,
            max_memory_bytes=memory_budget_bytes,
            k2_ref_per_m2=k2_ref,
            callback=_cb,
        )
        # Repeat finest-level eigenmode solve 3 times for Rule 8 timing spread
        finest_mesh = res.mesh
        repeats = []
        for _ in range(3):
            t0 = time.perf_counter()
            s_rep = UnstructuredFIT3D(
                finest_mesh, bc="pec", hodge="whitney", stability=False
            )
            s_rep.eigenmodes(n_modes=2, f_target=f_target)
            repeats.append(float(time.perf_counter() - t0))
        rep_mean = float(np.mean(repeats))
        rep_se = float(np.std(repeats, ddof=1) / np.sqrt(len(repeats)))
        extra["final_solve_repeats_s"] = repeats
        extra["final_solve_mean_s"] = rep_mean
        extra["final_solve_se_s"] = rep_se

        # Voronoi export summary per uniform generation (for P6 closest-error match)
        vor_by_gen = []
        for it, m_g, h_g in gen_meshes:
            if m_g.n_cells <= 50000:
                vor_by_gen.append(
                    {
                        "generation": int(it),
                        **_summarize_voronoi(
                            m_g, float(h_g["rel_err_signed"]), int(h_g["n_dofs"])
                        ),
                    }
                )
        extra["voronoi_by_generation"] = vor_by_gen

    elif arm == "residual":
        res = adapt_eigen(
            mesh0,
            bc="pec",
            f_target_hz=f_target,
            mode=0,
            estimator="residual",
            marking="dorfler",
            theta=0.5,
            max_iter=max_iter_adapt,
            max_dofs=max_dofs_adapt,
            max_memory_bytes=memory_budget_bytes,
            k2_ref_per_m2=k2_ref,
        )
        last_h = res.history[-1]
        if res.mesh.n_cells <= 60000:
            extra["voronoi_export"] = _summarize_voronoi(
                res.mesh, float(last_h["rel_err_signed"]), int(last_h["n_dofs"])
            )
        extra["top_face"] = _extract_top_face_data(res.mesh, res.e, res.f_hz)

    elif arm == "zz":
        res = adapt_eigen(
            mesh0,
            bc="pec",
            f_target_hz=f_target,
            mode=0,
            estimator="zz",
            marking="dorfler",
            theta=0.5,
            max_iter=max_iter_adapt,
            max_dofs=max_dofs_adapt,
            max_memory_bytes=memory_budget_bytes,
            k2_ref_per_m2=k2_ref,
        )

    elif arm in ("random0", "random1", "random2"):
        seed = int(arm[-1])
        counts = _load_or_default_mark_counts(outdir, quick)
        res = adapt_eigen(
            mesh0,
            bc="pec",
            f_target_hz=f_target,
            mode=0,
            estimator="residual",
            marking="random",
            mark_counts=counts,
            seed=seed,
            max_iter=min(max_iter_adapt, len(counts) + 1),
            max_dofs=max_dofs_adapt,
            max_memory_bytes=memory_budget_bytes,
            k2_ref_per_m2=k2_ref,
        )

    elif arm == "geometric":
        counts = _load_or_default_mark_counts(outdir, quick)

        def _geom_mark(mesh, eta2, it):
            k = int(counts[it]) if it < len(counts) else max(1, int(0.1 * mesh.n_cells))
            k = min(k, mesh.n_cells)
            cen = mesh.nodes[mesh.cells].mean(axis=1)
            r2 = cen[:, 0] ** 2 + cen[:, 1] ** 2
            return np.argsort(r2, kind="stable")[:k]

        res = adapt_eigen(
            mesh0,
            bc="pec",
            f_target_hz=f_target,
            mode=0,
            estimator="residual",
            marking=_geom_mark,
            max_iter=min(max_iter_adapt, len(counts) + 1),
            max_dofs=max_dofs_adapt,
            max_memory_bytes=memory_budget_bytes,
            k2_ref_per_m2=k2_ref,
        )

    elif arm in ("theta03", "theta07"):
        theta_val = 0.3 if arm == "theta03" else 0.7
        res = adapt_eigen(
            mesh0,
            bc="pec",
            f_target_hz=f_target,
            mode=0,
            estimator="residual",
            marking="dorfler",
            theta=theta_val,
            max_iter=max_iter_adapt,
            max_dofs=max_dofs_adapt,
            max_memory_bytes=memory_budget_bytes,
            k2_ref_per_m2=k2_ref,
        )
    else:
        raise ValueError(f"Unknown arm: {arm!r}")

    hist = _enrich_history(res.history)
    n_it = len(hist)
    if arm == "uniform":
        i0 = min(2, max(0, n_it - 2))
        slope_info = fit_log_slope(hist, i0, n_it - 1)
    else:
        i0 = max(0, n_it // 2)
        slope_info = fit_log_slope(hist, i0, n_it - 1)

    return {
        "arm": arm,
        "stop_reason": res.stop_reason,
        "f_final_hz": float(res.f_hz),
        "n_cells_final": int(res.mesh.n_cells),
        "n_dofs_final": int(hist[-1]["n_dofs"]),
        "history": hist,
        "slope_fit": slope_info,
        "sign_change_detected": bool(len(slope_info["sign_change_iterations"]) > 0),
        "sign_change_iterations": slope_info["sign_change_iterations"],
        **extra,
    }


def _write_arm_json(
    outdir: str,
    arm_name: str,
    quick: bool,
    meta: dict,
    mps_ref: dict,
    arms_dict: dict,
) -> str:
    os.makedirs(outdir, exist_ok=True)
    suffix = "_quick" if quick else ""
    out_path = os.path.join(outdir, f"adapt_{arm_name}{suffix}.json")
    payload = {
        "meta": meta,
        "mps_reference": mps_ref,
        "predictions": PREDICTIONS,
        "arms": arms_dict,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[run_adapt_study] wrote {out_path}")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run adaptive tetrahedral FIT eigenmode study arms."
    )
    parser.add_argument(
        "--arm",
        required=True,
        choices=(*ALL_SINGLE_ARMS, "all"),
        help="Study arm to run (or 'all').",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick small-scale study (< 3 min, ~1e4 cells).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for JSON results.",
    )
    args = parser.parse_args(argv)

    meta = build_meta(args.quick)
    mps_ref = compute_mps_reference()
    k2_ref = float(mps_ref["k2_per_m2"])
    budget = int(meta["memory_budget_bytes"])

    if args.arm == "all":
        # Run residual before random/geometric so its exact mark_counts are on disk
        order = (
            "box",
            "uniform",
            "residual",
            "zz",
            "random0",
            "random1",
            "random2",
            "geometric",
            "theta03",
            "theta07",
        )
        all_arms = {}
        for arm_name in order:
            if arm_name == "box":
                arm_res = run_box_arm(args.quick, budget)
            else:
                arm_res = run_l_prism_arm(
                    arm_name, args.quick, k2_ref, budget, args.out
                )
            all_arms[arm_name] = arm_res
            _write_arm_json(
                args.out, arm_name, args.quick, meta, mps_ref, {arm_name: arm_res}
            )
        _write_arm_json(args.out, "all", args.quick, meta, mps_ref, all_arms)
    elif args.arm == "box":
        arm_res = run_box_arm(args.quick, budget)
        _write_arm_json(args.out, "box", args.quick, meta, mps_ref, {"box": arm_res})
    else:
        arm_res = run_l_prism_arm(args.arm, args.quick, k2_ref, budget, args.out)
        _write_arm_json(
            args.out, args.arm, args.quick, meta, mps_ref, {args.arm: arm_res}
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
