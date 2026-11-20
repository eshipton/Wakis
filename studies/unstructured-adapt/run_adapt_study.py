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
        [--quick] [--procs N] --out DIR
"""

from __future__ import annotations

import argparse
import gc
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
from wakis.unstructured.analytic import (
    l_shape_dirichlet_eigenvalue,
    l_shape_dirichlet_field,
)
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
    300,
    535,
    759,
    1180,
    1755,
    2489,
    4038,
    5400,
    7994,
    11505,
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
    # Respect the 2.0 GB per-process RSS cap (memguard -m 2000).
    budget = int(2e9)
    preflight = {
        str(n): float(estimate_memory(n, dim=3, eigen=True))
        for n in (1350, 10800, 40320, 43200, 86400)
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


def compute_mps_reference() -> tuple[dict, np.ndarray]:
    lam24, sig24, _ = l_shape_dirichlet_eigenvalue(n_terms=24, seed=0)
    lam32, sig32, coeffs32 = l_shape_dirichlet_eigenvalue(n_terms=32, seed=0)
    f_ref_hz = float(np.sqrt(lam32) * C0 / (2.0 * np.pi))
    f_ref_24_hz = float(np.sqrt(lam24) * C0 / (2.0 * np.pi))
    ref_dict = {
        "n_terms": 32,
        "k2_per_m2": float(lam32),
        "sigma_min": float(sig32),
        "c0": float(coeffs32[0]),
        "f_ref_hz": f_ref_hz,
        "n_terms_24": 24,
        "k2_24_per_m2": float(lam24),
        "sigma_min_24": float(sig24),
        "f_ref_24_hz": f_ref_24_hz,
        "n_terms_check": 32,
        "k2_check_per_m2": float(lam32),
        "sigma_min_check": float(sig32),
        "rel_diff_24_vs_32": float(abs(lam24 - lam32) / lam32),
    }
    return ref_dict, coeffs32


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


_DUFFY_6 = _duffy_tet_rule(6)
_MPS_CELL_CACHE: dict[bytes, tuple[float, np.ndarray, float, float, float]] = {}


def _l_prism_mps_errors(
    mesh, e, lam: float, coeffs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-tetrahedron ``||curl(u - u_h)||_T^2`` and ``k^2 ||u - u_h||_T^2`` [1/m^2]
    on the L-prism against the MPS reference field ``u = psi(x, y) z_hat`` and
    ``curl u = (dpsi/dy, -dpsi/dx, 0)`` using 6x6x6 Duffy Gauss quadrature.

    Since ``psi`` and ``grad psi`` depend only on ``(x, y)``, unique 4x2 cell
    ``(x, y)`` vertex configurations are deduplicated and cached across levels
    and arms with identical floating-point results to element-by-element quadrature.
    """
    U, B = whitney_fields(mesh, e)
    u_l2_T = mesh.cell_volume * np.einsum("nik,ij,njk->n", U, _P1_TET_MASS, U)
    inv_norm_uh = 1.0 / np.sqrt(float(np.sum(u_l2_T)))
    U = U * inv_norm_uh
    B = B * inv_norm_uh
    u_l2_T = u_l2_T * (inv_norm_uh**2)

    bary, W = _DUFFY_6
    nc = mesh.n_cells
    X_xy_all = np.round(mesh.nodes[mesh.cells, :2], 14)
    X_flat = np.ascontiguousarray(X_xy_all.reshape(nc, 8))
    u_flat, inv = np.unique(X_flat, axis=0, return_inverse=True)
    nu = len(u_flat)

    q_psi2 = np.empty(nu, dtype=float)
    q_psi_bary = np.empty((nu, 4), dtype=float)
    q_g2 = np.empty(nu, dtype=float)
    q_gx = np.empty(nu, dtype=float)
    q_gy = np.empty(nu, dtype=float)

    missing_idx = []
    missing_keys = []
    for u_i in range(nu):
        key = u_flat[u_i].tobytes()
        cached = _MPS_CELL_CACHE.get(key)
        if cached is not None:
            q_psi2[u_i], q_psi_bary[u_i], q_g2[u_i], q_gx[u_i], q_gy[u_i] = cached
        else:
            missing_idx.append(u_i)
            missing_keys.append(key)

    if missing_idx:
        miss_arr = np.asarray(missing_idx, dtype=np.int64)
        u_xy_miss = u_flat[miss_arr].reshape(-1, 4, 2)
        n_miss = len(miss_arr)
        for i0 in range(0, n_miss, 256):
            sl = slice(i0, min(n_miss, i0 + 256))
            pts_xy = np.einsum("qk,nki->nqi", bary, u_xy_miss[sl])
            psi, gx, gy = l_shape_dirichlet_field(
                pts_xy[..., 0], pts_xy[..., 1], lam, coeffs
            )
            c_psi2 = np.sum(W[None, :] * (psi**2), axis=1)
            c_psi_bary = np.einsum("q,nq,qk->nk", W, psi, bary)
            c_g2 = np.sum(W[None, :] * (gx**2 + gy**2), axis=1)
            c_gx = np.sum(W[None, :] * gx, axis=1)
            c_gy = np.sum(W[None, :] * gy, axis=1)
            for j, u_i in enumerate(miss_arr[sl]):
                val = (
                    float(c_psi2[j]),
                    c_psi_bary[j].copy(),
                    float(c_g2[j]),
                    float(c_gx[j]),
                    float(c_gy[j]),
                )
                _MPS_CELL_CACHE[missing_keys[i0 + j]] = val
                q_psi2[u_i], q_psi_bary[u_i], q_g2[u_i], q_gx[u_i], q_gy[u_i] = val

    vol = mesh.cell_volume
    I_psi2 = vol * q_psi2[inv]
    I_psi_uh = vol * np.einsum("nk,nk->n", q_psi_bary[inv], U[:, :, 2])
    I_g2 = vol * q_g2[inv]
    I_curl = vol * (q_gy[inv] * B[:, 0] - q_gx[inv] * B[:, 1])
    I_B2 = vol * np.sum(B**2, axis=1)

    sgn = 1.0 if float(np.sum(I_psi_uh)) >= 0.0 else -1.0
    alpha = sgn / np.sqrt(float(np.sum(I_psi2)))
    cell_curl_e2 = (alpha**2) * I_g2 - 2.0 * alpha * I_curl + I_B2
    cell_k2_e2 = float(lam) * ((alpha**2) * I_psi2 - 2.0 * alpha * I_psi_uh + u_l2_T)
    return cell_curl_e2, cell_k2_e2


def _box_exact_errors(mesh, e_raw, a: float = 0.7, b: float = 0.5, d: float = 0.3):
    U_raw, B_raw = whitney_fields(mesh, e_raw)
    u_l2_T = mesh.cell_volume * np.einsum("nik,ij,njk->n", U_raw, _P1_TET_MASS, U_raw)
    inv_norm = 1.0 / float(np.sqrt(np.sum(u_l2_T)))
    U_vert = U_raw * inv_norm
    B_cell = B_raw * inv_norm
    bary, qw = _DUFFY_6
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

        err_u_l2_sq, err_curl_sq, err_energy_sq, _ = _box_exact_errors(
            cur_mesh, e0, a, b, d
        )
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
                "err_u_l2_sq_per_m2": float(k2_exact * err_u_l2_sq),
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
        del solver, modes
        gc.collect()

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
    signs = [np.sign(h["rel_err_signed"]) for h in sub]
    nonzero_signs = [s for s in signs if s != 0]
    sign_change = len(set(nonzero_signs)) > 1

    full_signs = [np.sign(h["rel_err_signed"]) for h in history]
    sign_flip_iters = [
        i for i in range(1, len(full_signs)) if full_signs[i] * full_signs[i - 1] < 0
    ]

    if sign_change:
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


def fit_positive_log_slope(
    history: list[dict], key: str, i_start: int, i_end: int
) -> dict:
    """Least-squares fit of log10(history[i][key]) vs log10(n_dofs) on [i_start, i_end]."""
    sub = history[i_start : i_end + 1]
    x = np.log10([float(h["n_dofs"]) for h in sub])
    y = np.log10([max(float(h[key]), 1e-30) for h in sub])
    n = len(x)
    if n < 2:
        return {
            "key": key,
            "slope": None,
            "slope_se": None,
            "intercept": None,
            "index_range": [int(i_start), int(i_end)],
            "dof_range": [int(sub[0]["n_dofs"]), int(sub[-1]["n_dofs"])],
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
        "key": key,
        "slope": slope,
        "slope_se": slope_se,
        "intercept": intercept,
        "index_range": [int(i_start), int(i_end)],
        "dof_range": [int(sub[0]["n_dofs"]), int(sub[-1]["n_dofs"])],
    }


def _enrich_history(history: list[dict], k2_ref: float) -> list[dict]:
    cum_t = 0.0
    out = []
    for i, h in enumerate(history):
        d = dict(h)
        d["iteration"] = i
        cum_t += float(d["t_total_s"])
        d["cum_wall_s"] = cum_t
        if "rel_err_signed" in d:
            d["abs_rel_err"] = abs(float(d["rel_err_signed"]))
            d["abs_err_k2_per_m2"] = abs(float(d["k2_per_m2"]) - float(k2_ref))
            d["eigenvalue_effectivity"] = float(
                d["eta2_per_m2"]
                / max(abs(float(d["k2_per_m2"]) - float(k2_ref)), 1e-30)
            )
        if "energy_err2_per_m2" in d and d["energy_err2_per_m2"] > 0.0:
            d["theta_res"] = float(np.sqrt(d["eta2_per_m2"] / d["energy_err2_per_m2"]))
        if "curl_err2_per_m2" in d and d["curl_err2_per_m2"] > 0.0:
            d["theta_zz"] = float(np.sqrt(d["eta2_zz_per_m2"] / d["curl_err2_per_m2"]))
        out.append(d)
    return out


def _summarize_voronoi(
    mesh,
    rel_err_signed: float,
    n_dofs: int,
    n_sweeps: int = 2,
    max_flips: int | None = None,
) -> dict:
    kw = {"bc": "pec", "n_sweeps": n_sweeps}
    if max_flips is not None:
        kw["max_flips"] = max_flips
    _, rep = voronoi_export(mesh, **kw)
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


def adapt_loc_edges():
    from wakis.unstructured.adapt import _LOC_E

    return _LOC_E


def _extract_cut_plane_data(
    mesh,
    e_vec: np.ndarray,
    f_hz: float,
    v0_m3: float,
    a0_m2: float = 0.02,
    z_cut: float = 0.2,
    n_dofs_3d: int | None = None,
) -> dict:
    """
    Extract horizontal plane triangulation at ``z = z_cut`` [m] together with
    per-triangle parent tetrahedron bisection level ``level_T``, 2D face
    bisection level ``level_F``, diameter ``h_T_m``, and residual indicator
    ``eta_T_per_m``.
    """
    kh2 = float((2.0 * np.pi * f_hz / C0) ** 2)
    eta2_T = residual_indicator(mesh, e_vec, kh2, bc="pec")
    eta_T = np.sqrt(np.maximum(eta2_T, 0.0))

    pairs = mesh.cells[:, adapt_loc_edges()]
    d_vec = mesh.nodes[pairs[..., 1]] - mesh.nodes[pairs[..., 0]]
    h_T = np.sqrt(np.max(np.einsum("nci,nci->nc", d_vec, d_vec), axis=1))

    level_3d = np.rint(np.log2(v0_m3 / mesh.cell_volume)).astype(int)
    u_lvl, u_cnt = np.unique(level_3d, return_counts=True)
    level_hist_3d = {int(k): int(v) for k, v in zip(u_lvl, u_cnt)}

    f_z = mesh.nodes[mesh.faces, 2]
    on_cut = np.all(np.abs(f_z - z_cut) < 1e-9, axis=1)
    cut_faces = np.flatnonzero(on_cut)
    cut_tris_global = mesh.faces[cut_faces]

    # Map each face on z = z_cut to an adjacent tetrahedron (preferring the
    # side with centroid z > z_cut for interior cuts so each cut triangle has a
    # unique side-consistent parent tetrahedron).
    cell_z_cen = mesh.nodes[mesh.cells, 2].mean(axis=1)
    f_to_cell = np.zeros(mesh.n_faces, dtype=np.int64)
    all_c = np.repeat(np.arange(mesh.n_cells, dtype=np.int64), 4)
    all_f = mesh.cell_faces.ravel()
    f_to_cell[all_f] = all_c
    upper_mask = cell_z_cen[all_c] > z_cut
    f_to_cell[all_f[upper_mask]] = all_c[upper_mask]
    parent_cells = f_to_cell[cut_faces]

    level_2d = np.rint(np.log2(a0_m2 / mesh.face_area[cut_faces])).astype(int)

    u_nodes, inv = np.unique(cut_tris_global.ravel(), return_inverse=True)
    xy = np.round(mesh.nodes[u_nodes, :2], 6).tolist()
    tris = inv.reshape(-1, 3).tolist()
    h_cut = [float(f"{val:.6g}") for val in h_T[parent_cells]]
    eta_cut = [float(f"{val:.6g}") for val in eta_T[parent_cells]]
    lvl_t_cut = [int(val) for val in level_3d[parent_cells]]
    lvl_f_cut = [int(val) for val in level_2d]

    return {
        "z_m": float(z_cut),
        "n_cells_3d": int(mesh.n_cells),
        "n_dofs_3d": int(n_dofs_3d) if n_dofs_3d is not None else None,
        "n_triangles": len(cut_faces),
        "level_min_3d": int(level_3d.min()),
        "level_max_3d": int(level_3d.max()),
        "level_min_cut": int(min(lvl_t_cut)) if lvl_t_cut else 0,
        "level_max_cut": int(max(lvl_t_cut)) if lvl_t_cut else 0,
        "cell_level_counts_3d": level_hist_3d,
        "xy_m": xy,
        "triangles": tris,
        "level_T": lvl_t_cut,
        "level_F": lvl_f_cut,
        "h_T_m": h_cut,
        "eta_T_per_m": eta_cut,
    }


def _load_or_default_mark_counts(outdir: str, quick: bool) -> list[int]:
    suffix = "_quick" if quick else ""
    res_path = os.path.join(outdir, f"adapt_residual{suffix}.json")
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as fh:
            data = json.load(fh)
        hist = data.get("arms", {}).get("residual", {}).get("history", [])
        counts = [int(h["n_marked"]) for h in hist if int(h["n_marked"]) > 0]
        if counts:
            return counts
    return list(DEFAULT_RESIDUAL_MARK_COUNTS)


def run_l_prism_arm(
    arm: str,
    quick: bool,
    k2_ref: float,
    coeffs_ref: np.ndarray,
    memory_budget_bytes: int,
    outdir: str,
) -> dict:
    f_ref = float(np.sqrt(k2_ref) * C0 / (2.0 * np.pi))
    f_target = 0.8 * f_ref
    mesh0 = l_prism_mesh(5, 3, height_m=0.6)
    v0_m3 = float(mesh0.cell_volume[0])

    # Target DOFs: 3 uniform generations in --quick (12,054 DOFs, 10,800 cells),
    # or 5 uniform generations (49,006 DOFs, 43,200 cells) under the 2.0 GB
    # memory cap (since uniform generation 6 has 86,400 cells and requires
    # > 3.1 GB RSS / 5.1 GB estimated SuperLU memory).
    max_dofs_adapt = 12054 if quick else 49006
    max_iter_adapt = 12 if quick else 25

    initial_state: dict = {}

    def _make_callback(store_gen_meshes: list | None = None, capture_it0: bool = False):
        def _cb(it, mesh, e_vec, rec):
            cell_curl_e2, cell_k2_e2 = _l_prism_mps_errors(
                mesh, e_vec, k2_ref, coeffs_ref
            )
            curl_e2 = float(np.sum(cell_curl_e2))
            k2_e2 = float(np.sum(cell_k2_e2))
            rec["curl_err2_per_m2"] = curl_e2
            rec["k2_l2_err2_per_m2"] = k2_e2
            rec["energy_err2_per_m2"] = curl_e2 + k2_e2
            if store_gen_meshes is not None:
                store_gen_meshes.append((it, mesh, dict(rec)))
            if capture_it0 and it == 0:
                initial_state["mesh"] = mesh
                initial_state["e"] = e_vec.copy()
                initial_state["f_hz"] = float(rec["f_hz"])
                initial_state["n_dofs"] = int(rec["n_dofs"])

        return _cb

    extra = {}
    if arm == "uniform":
        n_gen = 3 if quick else 5
        gen_meshes: list = []

        res = adapt_eigen(
            mesh0,
            bc="pec",
            f_target_hz=f_target,
            mode=0,
            marking="uniform",
            max_iter=n_gen + 1,
            max_memory_bytes=memory_budget_bytes,
            k2_ref_per_m2=k2_ref,
            callback=_make_callback(store_gen_meshes=gen_meshes),
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
            del s_rep
            gc.collect()
        rep_mean = float(np.mean(repeats))
        rep_se = float(np.std(repeats, ddof=1) / np.sqrt(len(repeats)))
        extra["final_solve_repeats_s"] = repeats
        extra["final_solve_mean_s"] = rep_mean
        extra["final_solve_se_s"] = rep_se

        # Voronoi export summary per uniform generation (for P6 closest-error match)
        vor_by_gen = []
        for it, m_g, h_g in gen_meshes:
            if m_g.n_cells <= 25000:
                vor_by_gen.append(
                    {
                        "generation": int(it),
                        **_summarize_voronoi(
                            m_g,
                            float(h_g["rel_err_signed"]),
                            int(h_g["n_dofs"]),
                            n_sweeps=2,
                        ),
                    }
                )
            elif m_g.n_cells <= 50000:
                vor_by_gen.append(
                    {
                        "generation": int(it),
                        **_summarize_voronoi(
                            m_g,
                            float(h_g["rel_err_signed"]),
                            int(h_g["n_dofs"]),
                            n_sweeps=1,
                            max_flips=2000,
                        ),
                    }
                )
            gc.collect()
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
            callback=_make_callback(capture_it0=True),
        )
        last_h = res.history[-1]
        if res.mesh.n_cells <= 60000:
            extra["voronoi_export"] = _summarize_voronoi(
                res.mesh,
                float(last_h["rel_err_signed"]),
                int(last_h["n_dofs"]),
                n_sweeps=2,
            )
        extra["initial_mesh_cut"] = _extract_cut_plane_data(
            initial_state["mesh"],
            initial_state["e"],
            initial_state["f_hz"],
            v0_m3=v0_m3,
            z_cut=0.2,
            n_dofs_3d=initial_state["n_dofs"],
        )
        extra["adapted_mesh_cut"] = _extract_cut_plane_data(
            res.mesh,
            res.e,
            res.f_hz,
            v0_m3=v0_m3,
            z_cut=0.2,
            n_dofs_3d=int(last_h["n_dofs"]),
        )
        extra["top_face"] = _extract_cut_plane_data(
            res.mesh,
            res.e,
            res.f_hz,
            v0_m3=v0_m3,
            z_cut=0.6,
            n_dofs_3d=int(last_h["n_dofs"]),
        )

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
            callback=_make_callback(),
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
            callback=_make_callback(),
        )

    elif arm == "geometric":
        counts = _load_or_default_mark_counts(outdir, quick)

        def _geom_mark(mesh, _eta2, it):
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
            callback=_make_callback(),
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
            callback=_make_callback(),
        )
    else:
        raise ValueError(f"Unknown arm: {arm!r}")

    hist = _enrich_history(res.history, k2_ref)
    n_it = len(hist)
    if arm == "uniform":
        i0 = min(2, max(0, n_it - 2))
    else:
        i0 = max(0, n_it // 2)

    slope_info = fit_log_slope(hist, i0, n_it - 1)
    slope_energy = fit_positive_log_slope(hist, "energy_err2_per_m2", i0, n_it - 1)
    slope_eta2 = fit_positive_log_slope(hist, "eta2_per_m2", i0, n_it - 1)
    slope_eta2_zz = fit_positive_log_slope(hist, "eta2_zz_per_m2", i0, n_it - 1)
    slope_curl = fit_positive_log_slope(hist, "curl_err2_per_m2", i0, n_it - 1)

    if arm == "uniform" and n_it >= 6:
        extra["kuhn_window_slopes"] = {
            f"gen_{g}_to_{g + 3}": {
                "eig": fit_log_slope(hist, g, g + 3)["slope"],
                "energy": fit_positive_log_slope(hist, "energy_err2_per_m2", g, g + 3)[
                    "slope"
                ],
                "eta2": fit_positive_log_slope(hist, "eta2_per_m2", g, g + 3)["slope"],
            }
            for g in (0, 1, 2)
        }

    stop_reason_str = str(res.stop_reason)
    del res
    gc.collect()

    return {
        "arm": arm,
        "stop_reason": stop_reason_str,
        "f_final_hz": float(hist[-1]["f_hz"]),
        "n_cells_final": int(hist[-1]["n_cells"]),
        "n_dofs_final": int(hist[-1]["n_dofs"]),
        "history": hist,
        "slope_fit": slope_info,
        "slope_fit_energy": slope_energy,
        "slope_fit_eta2": slope_eta2,
        "slope_fit_eta2_zz": slope_eta2_zz,
        "slope_fit_curl": slope_curl,
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
        "--procs",
        type=int,
        default=1,
        help="Worker count hint (default 1 to stay within 2.0 GB memguard cap).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for JSON results.",
    )
    args = parser.parse_args(argv)

    meta = build_meta(args.quick)
    mps_ref, coeffs_ref = compute_mps_reference()
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
            t_a0 = time.perf_counter()
            if arm_name == "box":
                arm_res = run_box_arm(args.quick, budget)
            else:
                arm_res = run_l_prism_arm(
                    arm_name, args.quick, k2_ref, coeffs_ref, budget, args.out
                )
            all_arms[arm_name] = arm_res
            _write_arm_json(
                args.out, arm_name, args.quick, meta, mps_ref, {arm_name: arm_res}
            )
            print(
                f"[run_adapt_study] finished arm={arm_name} in "
                f"{time.perf_counter() - t_a0:.2f}s"
            )
        _write_arm_json(args.out, "all", args.quick, meta, mps_ref, all_arms)
    elif args.arm == "box":
        arm_res = run_box_arm(args.quick, budget)
        _write_arm_json(args.out, "box", args.quick, meta, mps_ref, {"box": arm_res})
    else:
        arm_res = run_l_prism_arm(
            args.arm, args.quick, k2_ref, coeffs_ref, budget, args.out
        )
        _write_arm_json(
            args.out, args.arm, args.quick, meta, mps_ref, {args.arm: arm_res}
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
