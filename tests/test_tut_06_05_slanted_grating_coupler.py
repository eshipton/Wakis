# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Physics, RCWA cross-check, energy conservation, conformal convergence, and
mutation tests for Tutorial 6.5: Slanted and Blazed In-Coupler Gratings
(06-05_slanted_grating_coupler).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.constants import c

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "06_periodic_structures"
    / "06-05_slanted_grating_coupler"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    DEFAULT_DEPTH_M,
    N_RIDGE,
    N_SUBSTRATE,
    PERIOD_INCOUPLER_M,
    WANG_DEPTH_M,
    WANG_FILL_FACTOR,
    WANG_SLANT_DEG,
    WL_DESIGN_M,
    bragg_slant_angle_deg,
    build_slanted_floquet_cell,
    rcwa_multilayer_1d,
    run_single_grating_orders,
)
from wakis.floquet import (  # noqa: E402
    _run_single_Solver,
    grating_angles,
    rcwa_1d_grating,
)


def test_multilayer_rcwa_matches_single_layer_rcwa_and_conserves_energy():
    """Verify multi-layer RCWA matches wakis.floquet.rcwa_1d_grating for binary gratings and conserves energy."""
    f0 = c / WL_DESIGN_M
    orders = (-2, -1, 0, 1, 2)
    fill = 0.50

    # Construct matching 1D profile for wakis.floquet.rcwa_1d_grating
    n_x = 1024
    x_grid = (np.arange(n_x) + 0.5) / n_x * PERIOD_INCOUPLER_M
    u = np.mod(x_grid / PERIOD_INCOUPLER_M + 0.5 * fill, 1.0)
    eps_prof_1layer = np.where(u < fill, N_RIDGE**2, 1.0)[::-1].copy()

    for pol in ("TE", "TM"):
        for th_deg in (0.0, 10.0):
            th_rad = np.deg2rad(th_deg)
            rc_1 = rcwa_1d_grating(
                [f0],
                th_rad,
                PERIOD_INCOUPLER_M,
                DEFAULT_DEPTH_M,
                eps_prof_1layer,
                eps_below=1.0,
                eps_above=N_SUBSTRATE**2,
                pol=pol,
                n_harmonics=12,
                orders=orders,
            )
            rc_m = rcwa_multilayer_1d(
                [f0],
                th_rad,
                PERIOD_INCOUPLER_M,
                DEFAULT_DEPTH_M,
                profile_type="binary",
                slant_deg=0.0,
                fill_factor=fill,
                eps_ridge=N_RIDGE**2,
                eps_groove=1.0,
                eps_below=1.0,
                eps_above=N_SUBSTRATE**2,
                pol=pol,
                n_harmonics=12,
                orders=orders,
            )
            np.testing.assert_allclose(rc_m["T_orders"], rc_1["T_orders"], atol=1e-12)
            np.testing.assert_allclose(rc_m["R_orders"], rc_1["R_orders"], atol=1e-12)
            assert float(rc_m["R_total"][0] + rc_m["T_total"][0]) == pytest.approx(
                1.0, abs=1e-12
            )

    # Energy conservation for slanted and blazed multi-layer RCWA
    for ptype, s_deg in (("slanted", 25.0), ("blazed", 0.0)):
        rc_p = rcwa_multilayer_1d(
            [f0],
            np.deg2rad(8.0),
            PERIOD_INCOUPLER_M,
            DEFAULT_DEPTH_M,
            profile_type=ptype,
            slant_deg=s_deg,
            fill_factor=0.60,
            orders=orders,
        )
        assert float(rc_p["R_total"][0] + rc_p["T_total"][0]) == pytest.approx(
            1.0, abs=1e-12
        )


def test_mutation_slant_zero_removes_plus_minus_one_asymmetry():
    """
    Mutation test:
    1. A slanted grating (alpha = +25 deg) at normal incidence produces strong
       asymmetry between the m = -1 and m = +1 orders (T_{-1} > 0.70,
       T_{-1} / T_{+1} > 10) in both Wakis 2D FIT and RCWA.
    2. Mutating slant angle alpha -> 0 deg (binary grating) removes the
       asymmetry between the +/-1 orders (|T_{-1} - T_{+1}| < 1e-3 in Wakis
       and < 1e-14 in RCWA).
    3. Mutating slant angle alpha -> -25 deg reverses the asymmetry so that
       T_{+1} > 0.70 and T_{+1} / T_{-1} > 10.
    """
    f0 = c / WL_DESIGN_M
    orders = (-2, -1, 0, 1, 2)
    im_m1 = orders.index(-1)
    im_p1 = orders.index(+1)

    # 1. Slanted alpha = +25 deg
    cell_pos, minfo = build_slanted_floquet_cell(
        period=PERIOD_INCOUPLER_M,
        height=DEFAULT_DEPTH_M,
        wavelength_min=WL_DESIGN_M,
        profile_type="slanted",
        slant_deg=25.0,
        fill_factor=0.55,
        orders_x=orders,
        cpw=24,
        run_time_cycles=24.0,
    )
    assert minfo["cpw_x"] >= 20.0
    assert minfo["cpw_z"] >= 20.0
    assert minfo["cells_per_min_feature_x"] >= 4.0

    res_pos = run_single_grating_orders(cell_pos, f0, theta_inc_rad=0.0, polarization="TE")
    rc_pos = rcwa_multilayer_1d(
        [f0],
        0.0,
        PERIOD_INCOUPLER_M,
        DEFAULT_DEPTH_M,
        profile_type="slanted",
        slant_deg=25.0,
        fill_factor=0.55,
        orders=orders,
    )
    t_m1_pos = float(res_pos["T_orders"][im_m1])
    t_p1_pos = float(res_pos["T_orders"][im_p1])
    assert t_m1_pos > 0.70
    assert t_m1_pos / max(t_p1_pos, 1e-12) > 10.0
    assert abs(t_m1_pos - float(rc_pos["T_orders"][0, im_m1])) < 1.2e-2
    assert float(res_pos["energy_total"]) == pytest.approx(1.0, abs=6e-3)

    # 2. Mutation: slant_deg -> 0.0 removes +/-1 asymmetry completely
    cell_zero, _ = build_slanted_floquet_cell(
        period=PERIOD_INCOUPLER_M,
        height=DEFAULT_DEPTH_M,
        wavelength_min=WL_DESIGN_M,
        profile_type="slanted",
        slant_deg=0.0,
        fill_factor=0.55,
        orders_x=orders,
        cpw=24,
        run_time_cycles=24.0,
    )
    res_zero = run_single_grating_orders(cell_zero, f0, theta_inc_rad=0.0, polarization="TE")
    rc_zero = rcwa_multilayer_1d(
        [f0],
        0.0,
        PERIOD_INCOUPLER_M,
        DEFAULT_DEPTH_M,
        profile_type="slanted",
        slant_deg=0.0,
        fill_factor=0.55,
        orders=orders,
    )
    assert abs(float(rc_zero["T_orders"][0, im_m1] - rc_zero["T_orders"][0, im_p1])) < 1e-14
    assert abs(float(res_zero["T_orders"][im_m1] - res_zero["T_orders"][im_p1])) < 1e-3
    assert float(res_zero["T_orders"][im_m1] / res_zero["T_orders"][im_p1]) == pytest.approx(
        1.0, abs=5e-3
    )

    # 3. Mutation: slant_deg -> -25.0 flips the asymmetry to m = +1
    rc_neg = rcwa_multilayer_1d(
        [f0],
        0.0,
        PERIOD_INCOUPLER_M,
        DEFAULT_DEPTH_M,
        profile_type="slanted",
        slant_deg=-25.0,
        fill_factor=0.55,
        orders=orders,
    )
    assert float(rc_neg["T_orders"][0, im_p1]) == pytest.approx(
        float(rc_pos["T_orders"][0, im_m1]), rel=1e-12
    )
    assert float(rc_neg["T_orders"][0, im_m1]) == pytest.approx(
        float(rc_pos["T_orders"][0, im_p1]), rel=1e-12
    )


def test_grating_angles_exact_and_blazed_coupler_agreement():
    """Verify exact grating angles and Wakis vs RCWA agreement on sawtooth blazed grating."""
    f0 = c / WL_DESIGN_M
    orders = (-2, -1, 0, 1, 2)
    th_inc = np.deg2rad(8.0)

    cell_blz, _ = build_slanted_floquet_cell(
        period=PERIOD_INCOUPLER_M,
        height=DEFAULT_DEPTH_M,
        wavelength_min=WL_DESIGN_M,
        profile_type="blazed",
        slant_deg=0.0,
        fill_factor=0.85,
        orders_x=orders,
        cpw=24,
        run_time_cycles=24.0,
    )
    k_par = float((2.0 * np.pi / WL_DESIGN_M) * np.sin(th_inc))
    pb, pa = _run_single_Solver(
        cell_blz, k_par=k_par, phi=0.0, polarization="TE", frequencies=[f0]
    )

    # Exact grating angles check on both reflection and transmission monitors
    th_refl_mon, _ = pb.monitor.order_angles()
    th_tran_mon, _ = pa.monitor.order_angles()
    ang_refl_port = th_refl_mon[0, :, 0]
    ang_tran_port = th_tran_mon[0, :, 0]
    ang_refl_exact = grating_angles(
        f0, [th_inc], PERIOD_INCOUPLER_M, pb.orders_b, n_inc=1.0, n_out=1.0
    )[0]
    ang_tran_exact = grating_angles(
        f0, [th_inc], PERIOD_INCOUPLER_M, pa.orders_b, n_inc=1.0, n_out=N_SUBSTRATE
    )[0]
    val_r = np.isfinite(ang_refl_exact)
    val_t = np.isfinite(ang_tran_exact)
    np.testing.assert_allclose(
        ang_refl_port[val_r], np.abs(ang_refl_exact[val_r]), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        ang_tran_port[val_t], np.abs(ang_tran_exact[val_t]), rtol=1e-12, atol=1e-12
    )
    assert np.all(np.isnan(ang_refl_port[~val_r]))
    assert np.all(np.isnan(ang_tran_port[~val_t]))

    # Bragg slant angle formula check
    alpha_b = bragg_slant_angle_deg(0.0, WL_DESIGN_M, PERIOD_INCOUPLER_M, 1.0, N_SUBSTRATE, -1)
    assert alpha_b == pytest.approx(24.48257, abs=1e-3)


def test_saved_tutorial_data_and_conformal_vs_staircase_convergence():
    """Verify saved Tutorial 6.5 data/, figures/, and Kottke conformal vs staircase accuracy."""
    data_dir = TUTORIAL_DIR / "data"
    fig_dir = TUTORIAL_DIR / "figures"

    summary = json.loads((data_dir / "summary.json").read_text())
    assert "slant_angle_and_profile_comparison" in summary
    assert "depth_and_fill_factor_sweeps" in summary
    assert "oblique_incidence_and_broadband_sweep" in summary
    assert "staircase_vs_conformal_convergence" in summary

    expected_figs = [
        "01_slanted_binary_blazed_schematic_and_mesh.png",
        "02_efficiency_vs_slant_and_profile_comparison.png",
        "03_depth_and_fill_factor_maps.png",
        "04_oblique_incidence_and_broadband_sweeps.png",
        "05_staircase_vs_conformal_and_nearfields.png",
        "06_slanted_vs_binary_wavefront_movie.gif",
    ]
    for fn in expected_figs:
        fpath = fig_dir / fn
        assert fpath.is_file() and fpath.stat().st_size > 10_000
    assert (fig_dir / "06_slanted_vs_binary_wavefront_movie.gif").stat().st_size < 5 * 1024 * 1024

    # Task 0 checks: binary symmetry vs slanted selectivity > 10x and Wakis vs RCWA agreement
    t0 = summary["slant_angle_and_profile_comparison"]
    assert t0["binary_symmetry_diff_at_0"] < 2e-3
    assert t0["peak_slanted_t_m1_wakis"] > 0.70
    assert t0["max_slant_sweep_err_m1"] < 1.5e-2
    assert t0["max_energy_residual"] < 8e-3
    assert t0["profiles"]["slanted"]["asymmetry_ratio_m1_over_p1"] > 10.0
    assert t0["profiles"]["blazed"]["asymmetry_ratio_m1_over_p1"] > 8.0
    assert t0["profiles"]["binary"]["asymmetry_ratio_m1_over_p1"] == pytest.approx(1.0, abs=0.03)

    # Task 1 checks: depth and fill factor sweeps agree with multi-layer RCWA
    t1 = summary["depth_and_fill_factor_sweeps"]
    assert t1["max_depth_sweep_err_m1"] < 1.5e-2
    assert t1["max_fill_sweep_err_m1"] < 1.5e-2
    assert t1["peak_2d_map_t_m1"] > 0.70

    # Task 2 checks: oblique sweeps and broadband AngleSweep resample vs shifted mutation
    t2 = summary["oblique_incidence_and_broadband_sweep"]
    for p_name in ("binary", "blazed", "slanted"):
        assert t2["profiles_oblique"][p_name]["max_abs_err_vs_rcwa"] < 1.5e-2
        assert t2["profiles_oblique"][p_name]["max_energy_residual"] < 8e-3
    assert t2["broadband_8deg_max_err"] < 1.5e-2
    assert t2["broadband_8deg_shifted_err"] > 2.5 * t2["broadband_8deg_max_err"]

    # Task 3 checks: Kottke conformal subpixel smoothing outperforms staircase meshing
    t3 = summary["staircase_vs_conformal_convergence"]
    assert t3["kottke_finest_rms"] < t3["stair_finest_rms"]
    assert t3["err_kottke_rms"][-1] < t3["err_kottke_rms"][0]
    assert t3["kottke_improvement_factor"] > 1.3


@pytest.mark.skipif(
    importlib.util.find_spec("fmmax") is None,
    reason="fmmax is optional and not installed in this environment",
)
def test_optional_fmmax_cross_check():
    """Optional cross-check of rcwa_multilayer_1d against fmmax when installed."""
    import fmmax  # noqa: F401

    rc = rcwa_multilayer_1d(
        [c / WL_DESIGN_M],
        0.0,
        PERIOD_INCOUPLER_M,
        DEFAULT_DEPTH_M,
        profile_type="slanted",
        slant_deg=25.0,
        fill_factor=0.55,
        orders=(-1, 0, 1),
    )
    assert float(rc["R_total"][0] + rc["T_total"][0]) == pytest.approx(1.0, abs=1e-12)
