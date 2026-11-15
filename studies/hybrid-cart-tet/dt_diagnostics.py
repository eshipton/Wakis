# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Diagnose why partial lumping (`mass="mixed"`) shortens `dt` on the
`test_delaunay` mesh (`tests/test_109_hybrid_time_domain.py` fixture
`meshes['delaunay']`).

For `mass='consistent'` and `mass='mixed'`, computes:
1. The top eigenvector `x` of the generalized problem `K^T W K x = lambda M x`
   via `_gen_lanczos_max(..., return_eigenvector=True)` in `wakis/hybrid/solver.py`,
   reporting `lambda_max` and `dt_max = 2 / sqrt(lambda_max)`.
2. The fraction of the M-weighted energy of `x` on lumped edges (`L`) vs
   coupled edges (`C`): `sum_{e in L} x_e (M x)_e / (x^T M x)` and the same
   for `C`.
3. For the lumped edges `L` and coupled edges `C`: the dual measure over
   squared edge length (`dual_area / L_e^2`), effective dual length over edge
   length (`sqrt(max(dual_area, 0)) / L_e`), and the ratio `D_V[e] / M1[e, e]`
   against the consistent diagonal, with min, median, max.
4. Sweep over `lump_min_ratio in {None, 0.25, 0.5, 1.0}` via
   `HybridFIT3D(..., mass='mixed', mass_lump_min_ratio=...)`, printing the
   lumped fraction and `dt_max` for each threshold.

Prints one JSON object to stdout and writes no files.
"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from scipy.constants import epsilon_0 as EPS0

from wakis.hybrid import HybridFIT3D, delaunay_box_mesh, mixed_partition
from wakis.hybrid.solver import _gen_lanczos_max
from wakis.unstructured.solver3d import whitney_mass_matrices

H = 0.1
XN = np.arange(0, 1.0 + 1e-9, H)
YN = np.arange(0, 0.8 + 1e-9, H)
ZN = np.arange(0, 0.6 + 1e-9, H)
BOX = ((3, 7), (2, 6), (2, 4))
LINES = (XN[3:8], YN[2:7], ZN[2:5])
LUMP_MIN_RATIOS = (None, 0.25, 0.5, 1.0)


def _summary_stats(arr):
    arr = np.asarray(arr, float)
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def _eig_energy_fractions(s, x, lump_mask_tet):
    Mx = s.M @ x
    denom = float(x @ Mx)
    ew = x * Mx

    lump_global = np.zeros(s.nEc + s.nEt, bool)
    lump_global[s.nEc :] = lump_mask_tet
    is_lump_free = lump_global[s.free_edges]

    is_msolver_coupled = np.zeros(s.n_free, bool)
    is_msolver_coupled[s.msolver.S] = True
    # Partition the active (lumped + coupled) support cleanly into L and C = S \ L
    is_coupled_free = is_msolver_coupled & ~is_lump_free

    frac_lumped = float(ew[is_lump_free].sum() / denom)
    frac_coupled = float(ew[is_coupled_free].sum() / denom)
    return frac_lumped, frac_coupled


def run_dt_diagnostics():
    tm = delaunay_box_mesh(*LINES, h=0.06, r=2, seed=0)
    ones = np.ones(tm.n_cells, float)
    M1_cons, _ = whitney_mass_matrices(tm, ones, ones, ones)
    m1_diag = M1_cons.diagonal()

    lump_mask_default, _ = mixed_partition(tm)
    coupled_mask_default = ~lump_mask_default

    dual_area = tm.dual_measure(ones)
    edge_len = tm.edge_length
    dv_all = EPS0 * dual_area / edge_len
    ratio_all = dv_all / m1_diag
    dual_area_over_len2 = dual_area / (edge_len**2)
    dual_len_over_edge_len = np.sqrt(np.maximum(dual_area, 0.0)) / edge_len

    lumped_ratio = _summary_stats(ratio_all[lump_mask_default])
    coupled_ratio = _summary_stats(ratio_all[coupled_mask_default])
    lumped_dual_len = _summary_stats(dual_len_over_edge_len[lump_mask_default])
    coupled_dual_len = _summary_stats(dual_len_over_edge_len[coupled_mask_default])
    lumped_dual_area = _summary_stats(dual_area_over_len2[lump_mask_default])
    coupled_dual_area = _summary_stats(dual_area_over_len2[coupled_mask_default])

    by_mass = {}
    for mass in ("consistent", "mixed"):
        s = HybridFIT3D(
            XN,
            YN,
            ZN,
            box=BOX,
            tet_mesh=tm,
            mass=mass,
            mass_solver="chebyshev",
        )
        lam_max, x = _gen_lanczos_max(s.K, s.W, s.M, s.msolver, return_eigenvector=True)
        dt_max = float(2.0 / np.sqrt(lam_max))
        frac_lumped, frac_coupled = _eig_energy_fractions(s, x, lump_mask_default)
        by_mass[mass] = {
            "mass": mass,
            "lambda_max": float(lam_max),
            "dt_max": dt_max,
            "dt": float(s.dt),
            "top_eig_fraction_lumped": frac_lumped,
            "top_eig_fraction_coupled": frac_coupled,
            "lumped_ratio_min": lumped_ratio["min"],
            "lumped_ratio_median": lumped_ratio["median"],
            "lumped_ratio_max": lumped_ratio["max"],
            "coupled_ratio_min": coupled_ratio["min"],
            "coupled_ratio_median": coupled_ratio["median"],
            "coupled_ratio_max": coupled_ratio["max"],
            "lumped_dual_over_len_min": lumped_dual_len["min"],
            "lumped_dual_over_len_median": lumped_dual_len["median"],
            "lumped_dual_over_len_max": lumped_dual_len["max"],
            "coupled_dual_over_len_min": coupled_dual_len["min"],
            "coupled_dual_over_len_median": coupled_dual_len["median"],
            "coupled_dual_over_len_max": coupled_dual_len["max"],
            "lumped_dual_area_over_len2_min": lumped_dual_area["min"],
            "lumped_dual_area_over_len2_median": lumped_dual_area["median"],
            "lumped_dual_area_over_len2_max": lumped_dual_area["max"],
            "coupled_dual_area_over_len2_min": coupled_dual_area["min"],
            "coupled_dual_area_over_len2_median": coupled_dual_area["median"],
            "coupled_dual_area_over_len2_max": coupled_dual_area["max"],
        }

    dt_max_cons = by_mass["consistent"]["dt_max"]
    threshold_sweep = {}
    for r_min in LUMP_MIN_RATIOS:
        key = "None" if r_min is None else str(r_min)
        s_r = HybridFIT3D(
            XN,
            YN,
            ZN,
            box=BOX,
            tet_mesh=tm,
            mass="mixed",
            mass_lump_min_ratio=r_min,
            mass_solver="chebyshev",
        )
        lam_r, x_r = _gen_lanczos_max(
            s_r.K, s_r.W, s_r.M, s_r.msolver, return_eigenvector=True
        )
        dt_max_r = float(2.0 / np.sqrt(lam_r))
        frac_l_r, frac_c_r = _eig_energy_fractions(s_r, x_r, s_r.lump_mask)
        n_lumped = int(s_r.lump_mask.sum())
        r_stats = _summary_stats(ratio_all[s_r.lump_mask])
        threshold_sweep[key] = {
            "lump_min_ratio": None if r_min is None else float(r_min),
            "n_tet_edges_lumped": n_lumped,
            "lump_fraction": float(n_lumped / tm.n_edges),
            "coupled_block": int(s_r.msolver.S.size),
            "lambda_max": float(lam_r),
            "dt_max": dt_max_r,
            "dt": float(s_r.dt),
            "dt_max_over_consistent": float(dt_max_r / dt_max_cons),
            "top_eig_fraction_lumped": frac_l_r,
            "top_eig_fraction_coupled": frac_c_r,
            "lumped_ratio_min": r_stats["min"],
            "lumped_ratio_median": r_stats["median"],
            "lumped_ratio_max": r_stats["max"],
        }

    mix = by_mass["mixed"]
    return {
        "mesh": "test_delaunay",
        "n_tets": int(tm.n_cells),
        "n_tet_edges": int(tm.n_edges),
        "n_tet_edges_lumped": int(lump_mask_default.sum()),
        "lump_fraction": float(lump_mask_default.sum() / tm.n_edges),
        "lambda_max": mix["lambda_max"],
        "dt_max": mix["dt_max"],
        "top_eig_fraction_lumped": mix["top_eig_fraction_lumped"],
        "top_eig_fraction_coupled": mix["top_eig_fraction_coupled"],
        "lumped_ratio_min": mix["lumped_ratio_min"],
        "lumped_ratio_median": mix["lumped_ratio_median"],
        "lumped_ratio_max": mix["lumped_ratio_max"],
        "coupled_ratio_min": mix["coupled_ratio_min"],
        "coupled_ratio_median": mix["coupled_ratio_median"],
        "coupled_ratio_max": mix["coupled_ratio_max"],
        "lumped_dual_over_len_min": mix["lumped_dual_over_len_min"],
        "lumped_dual_over_len_median": mix["lumped_dual_over_len_median"],
        "lumped_dual_over_len_max": mix["lumped_dual_over_len_max"],
        "coupled_dual_over_len_min": mix["coupled_dual_over_len_min"],
        "coupled_dual_over_len_median": mix["coupled_dual_over_len_median"],
        "coupled_dual_over_len_max": mix["coupled_dual_over_len_max"],
        "consistent": by_mass["consistent"],
        "mixed": by_mass["mixed"],
        "lump_min_ratio_sweep": threshold_sweep,
    }


def main():
    out = run_dt_diagnostics()
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
