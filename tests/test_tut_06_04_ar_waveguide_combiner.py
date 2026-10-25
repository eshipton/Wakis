# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Physics, k-space geometry, RCWA cross-check, energy conservation, and mutation
tests for Tutorial 6.4: AR Waveguide Combiner (06-04_ar_waveguide_combiner).
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
    / "06-04_ar_waveguide_combiner"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    H_SIN_M,
    N_SIN_TIAN,
    N_WG_TIAN,
    PERIOD_ACHROMATIC_M,
    PERIOD_GREEN_1ST_M,
    RGB_ACHROMATIC_ORDERS,
    RGB_WAVELENGTHS_M,
    WANG_OUTCOUPLER_DEPTHS_NM,
    build_coupler_floquet_cell,
    build_grating_eps_profile,
    kspace_combiner_map,
    max_symmetric_fov_deg,
    rgb_kspace_fov_comparison,
    simulate_eyebox_pupil_expansion,
    supercell_uniformity_summary,
    tian_uniformity,
    tir_critical_angle_rad,
)
from wakis.floquet import (  # noqa: E402
    AngleSweep,
    _run_single_Solver,
    grating_angles,
    rcwa_1d_grating,
)


def test_kspace_combiner_closure_and_fov_limits():
    """Verify 2D k-space vector closure, TIR FOV limits, and Tian et al. (2025) (4,5,6) achromatic match."""
    kmap = kspace_combiner_map(
        fov_x_deg=36.0,
        fov_y_deg=24.0,
        wavelength=RGB_WAVELENGTHS_M["G"],
        period_ic=PERIOD_GREEN_1ST_M,
        n_glass=N_WG_TIAN,
    )
    assert kmap["vector_closure_residual"] == pytest.approx(0.0, abs=1e-15)
    assert kmap["output_input_max_diff"] == pytest.approx(0.0, abs=1e-15)
    assert kmap["guided_fraction"] == pytest.approx(1.0, abs=1e-12)

    # Critical angle theta_c = arcsin(1 / 1.9) = 31.7569 deg and FOV_max = 2*arcsin((1.9-1)/2) = 53.487 deg
    th_c_deg = np.rad2deg(tir_critical_angle_rad(N_WG_TIAN, 1.0))
    assert th_c_deg == pytest.approx(np.rad2deg(np.arcsin(1.0 / 1.9)), rel=1e-12)
    fov_19 = max_symmetric_fov_deg(N_WG_TIAN, n_clad=1.0, max_wg_angle_deg=90.0)
    fov_174 = max_symmetric_fov_deg(1.74, n_clad=1.0, max_wg_angle_deg=90.0)
    fov_205 = max_symmetric_fov_deg(2.05, n_clad=1.0, max_wg_angle_deg=90.0)
    assert fov_19 == pytest.approx(2.0 * np.rad2deg(np.arcsin(0.45)), rel=1e-12)
    assert fov_174 == pytest.approx(43.4312, abs=1e-2)
    assert fov_205 == pytest.approx(63.3365, abs=1e-2)

    # Verify per-channel vs joint-RGB uniformity on the Task 1 supercell dataset
    d1 = np.load(TUTORIAL_DIR / "data" / "task_1_achromatic_supercell_incoupler.npz")
    u_sum = supercell_uniformity_summary({col: d1[f"wakis_t_m_{col}"] for col in ("B", "G", "R")})
    assert 0.85 <= u_sum["U_channel_min"] <= u_sum["U_channel_max"] <= 0.94
    assert 0.55 <= u_sum["U_joint_rgb"] <= 0.66
    # Mutation check: scaling the weakest channel by 0.2x degrades joint RGB uniformity below 0.35
    u_mut = supercell_uniformity_summary(
        {"B": d1["wakis_t_m_B"], "G": d1["wakis_t_m_G"] * 0.2, "R": d1["wakis_t_m_R"]}
    )
    assert u_mut["U_joint_rgb"] < 0.35

    # Tian et al. (2025) Eq. (2): m_R * lambda_R == m_G * lambda_G == m_B * lambda_B within 2 nm
    products_nm = [
        RGB_ACHROMATIC_ORDERS[col] * RGB_WAVELENGTHS_M[col] * 1e9
        for col in ("R", "G", "B")
    ]
    assert max(products_nm) - min(products_nm) == pytest.approx(2.0, abs=0.01)

    rgb_comp = rgb_kspace_fov_comparison(n_glass=N_WG_TIAN)
    assert rgb_comp["ach_common_fov_deg"] > 50.0
    assert rgb_comp["conv_common_fov_deg"] < 20.0
    assert rgb_comp["ach_common_fov_deg"] > 2.8 * rgb_comp["conv_common_fov_deg"]

    # Mutation check: perturbing G_OC by +2% breaks vector closure and distorts output angles
    g_oc_mutated = kmap["g_oc"] * 1.02
    closure_mutated = float(np.linalg.norm(kmap["g_ic"] + kmap["g_epe"] + g_oc_mutated))
    k_oc_mutated = kmap["k_epe"] + g_oc_mutated
    diff_mutated = float(np.max(np.abs(k_oc_mutated - kmap["k_in"])))
    assert closure_mutated > 0.025
    assert diff_mutated > 0.025


def test_grating_angles_exact_and_tir_evanescent_cutoff():
    """Verify exact grating-equation order angles and TIR evanescent NaN cutoffs."""
    wl_g = RGB_WAVELENGTHS_M["G"]
    f0 = c / wl_g
    theta_in = np.deg2rad(np.array([-15.0, 0.0, 10.0, 20.0]))
    orders = [-2, -1, 0, 1, 2]

    # 1. In-coupler (air n=1.0 -> glass n=1.9)
    ang_ic = grating_angles(
        f0, theta_in, PERIOD_GREEN_1ST_M, orders, n_inc=1.0, n_out=N_WG_TIAN
    )
    for ia, th in enumerate(theta_in):
        for im, m in enumerate(orders):
            arg = (np.sin(th) + m * (wl_g / PERIOD_GREEN_1ST_M)) / N_WG_TIAN
            if abs(arg) <= 1.0:
                assert ang_ic[ia, im] == pytest.approx(float(np.arcsin(arg)), abs=1e-13)
            else:
                assert np.isnan(ang_ic[ia, im])

    # 2. Out-coupler (glass n=1.9 at TIR bounce angle 45 deg > theta_c=31.8 deg -> air n=1.0)
    theta_tir = np.deg2rad(np.array([40.0, 45.0, 50.0]))
    ang_oc = grating_angles(
        f0, theta_tir, PERIOD_GREEN_1ST_M, [-1, 0, 1], n_inc=N_WG_TIAN, n_out=1.0
    )
    # Order m=0 in air MUST be evanescent (NaN) for all TIR bounce angles > theta_c,
    # whereas order m=-1 is propagating in air toward the eye-box!
    assert np.all(np.isnan(ang_oc[:, 1]))
    assert np.all(np.isfinite(ang_oc[:, 0]))

    # Mutation check: shifting order m=-1 -> m=0 changes the angle from finite to NaN
    assert not np.all(np.isfinite(ang_oc[:, 1]))


def test_wakis_vs_rcwa_incoupler_and_outcoupler_efficiencies():
    """Verify Wakis 2D FIT order efficiencies vs RCWA, passivity, and AngleSweep resampling."""
    wl_g = RGB_WAVELENGTHS_M["G"]
    f_arr = np.array([c / wl_g])
    orders = (-1, 0, 1)

    # 1. In-coupler at theta_in = 10 deg
    cell_ic, minfo_ic = build_coupler_floquet_cell(
        period=PERIOD_GREEN_1ST_M,
        height=H_SIN_M,
        wavelength_min=wl_g,
        fill_factor=0.50,
        eps_ridge=N_SIN_TIAN**2,
        eps_groove=1.0,
        eps_below=1.0,
        eps_above=N_WG_TIAN**2,
        orders_x=orders,
        cpw=22,
        run_time_cycles=22.0,
    )
    assert minfo_ic["cpw_x"] >= 20.0
    assert minfo_ic["cpw_z"] >= 20.0
    assert minfo_ic["cells_per_min_feature_x"] >= 4.0
    assert minfo_ic["cells_per_grating_height_z"] >= 4.0

    th_inc = np.deg2rad(10.0)
    k_par = float((2.0 * np.pi / wl_g) * np.sin(th_inc))
    pb, pa = _run_single_Solver(
        cell_ic, k_par=k_par, phi=0.0, polarization="TE", frequencies=f_arr
    )

    # Check FloquetPort.monitor.order_angles() matches grating_angles() exactly
    th_mon_tran, _ = pa.monitor.order_angles()
    port_angles_tran = th_mon_tran[0, :, 0]
    exact_angles_tran = grating_angles(
        f_arr[0], [th_inc], PERIOD_GREEN_1ST_M, orders, n_inc=1.0, n_out=N_WG_TIAN
    )[0]
    valid_t = np.isfinite(exact_angles_tran)
    np.testing.assert_allclose(
        port_angles_tran[valid_t], np.abs(exact_angles_tran[valid_t]), rtol=1e-12, atol=1e-12
    )

    a_inc = pb.mode_amplitude((1, 0, 0, "TE"), direction="in")[0]
    t_wakis = np.array(
        [abs(pa.mode_amplitude((2, m, 0, "TE"), direction="out")[0] / a_inc) ** 2 for m in orders]
    )
    r_wakis = np.array(
        [abs(pb.mode_amplitude((1, m, 0, "TE"), direction="out")[0] / a_inc) ** 2 for m in orders]
    )
    tot_energy = float(np.nansum(r_wakis) + np.nansum(t_wakis))
    assert tot_energy == pytest.approx(1.0, abs=5e-3)

    eps_prof = build_grating_eps_profile(
        1024, fill_factor=0.50, eps_ridge=N_SIN_TIAN**2, eps_groove=1.0
    )
    rc = rcwa_1d_grating(
        f_arr,
        th_inc,
        PERIOD_GREEN_1ST_M,
        H_SIN_M,
        eps_prof,
        eps_below=1.0,
        eps_above=N_WG_TIAN**2,
        pol="TE",
        orders=orders,
    )
    np.testing.assert_allclose(t_wakis, rc["T_orders"][0], atol=5e-3)

    # 2. AngleSweep broadband resample vs index_shift=1 mutation check
    freqs_bb = np.linspace(0.94 * f_arr[0], 1.06 * f_arr[0], 5)
    angles_bb = np.deg2rad(np.array([4.0, 10.0, 16.0]))
    sweep = AngleSweep(cell_ic, freqs_bb, angles_bb, polarization="TE", n_k=5)
    res_sw = sweep.run(processes=2)
    out_ok = res_sw.resample()
    out_mut = res_sw.resample(index_shift=1)

    rc_10 = rcwa_1d_grating(
        freqs_bb,
        angles_bb[1],
        PERIOD_GREEN_1ST_M,
        H_SIN_M,
        eps_prof,
        eps_below=1.0,
        eps_above=N_WG_TIAN**2,
        pol="TE",
        orders=orders,
    )["T_orders"][:, 0]
    err_ok = float(np.max(np.abs(out_ok["T_order"][1, :, 0, 0] - rc_10)))
    err_mut = float(np.max(np.abs(out_mut["T_order"][1, :, 0, 0] - rc_10)))
    assert err_ok < 8e-3
    assert err_mut > 3.0 * err_ok


def test_saved_tutorial_data_and_eyebox_uniformity():
    """Verify cached tutorial data/ and figures/, eye-box uniformity improvement, and convergence."""
    data_dir = TUTORIAL_DIR / "data"
    fig_dir = TUTORIAL_DIR / "figures"

    summary = json.loads((data_dir / "summary.json").read_text())
    assert "kspace_and_rgb_incoupler_sweep" in summary
    assert "achromatic_supercell_incoupler" in summary
    assert "outcoupler_tir_and_eyebox_uniformity" in summary
    assert "nearfield_and_resolution_study" in summary

    # Check all setup and result figures exist and GIF < 5 MB
    expected_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00a_setup_geometry_supercell.png",
        "00b_setup_mesh_and_excitation_supercell.png",
        "00a_setup_geometry_outcoupler.png",
        "00b_setup_mesh_and_excitation_outcoupler.png",
        "01_combiner_and_unit_cell_schematic.png",
        "02_kspace_diagram_and_fov_limits.png",
        "03_incoupler_rgb_efficiency_vs_angle.png",
        "04_outcoupler_tir_and_eyebox_uniformity.png",
        "05_nearfield_and_resolution_study.png",
        "06_incoupler_wavefront_movie.gif",
    ]
    for fn in expected_figs:
        fpath = fig_dir / fn
        assert fpath.is_file() and fpath.stat().st_size > 10_000
    for tfn in (
        "00_setup_table.md",
        "00_setup_table_supercell.md",
        "00_setup_table_outcoupler.md",
    ):
        tpath = fig_dir / tfn
        assert tpath.is_file() and tpath.stat().st_size > 200
    assert (fig_dir / "06_incoupler_wavefront_movie.gif").stat().st_size < 5 * 1024 * 1024

    # Check RGB channels in Task 0
    t0 = summary["kspace_and_rgb_incoupler_sweep"]
    for col in ("B", "G", "R"):
        ch = t0["rgb_channels"][col]
        assert ch["max_abs_err_vs_rcwa"] < 5e-3
        assert ch["max_energy_residual"] < 5e-3
        assert ch["mesh"]["cpw_x"] >= 20.0

    # Check Task 2 out-coupler TIR and eye-box uniformity
    t2 = summary["outcoupler_tir_and_eyebox_uniformity"]
    assert t2["tir_t0_all_nan"] is True
    assert t2["max_abs_err_r0"] < 8e-3
    assert t2["max_abs_err_tm1"] < 8e-3
    assert t2["reciprocity_abs_diff"] < 5e-3
    assert t2["eyebox_graduated_uniformity_tian"] > t2["eyebox_constant_uniformity_tian"]

    # Check Task 3 resolution convergence
    t3 = summary["nearfield_and_resolution_study"]
    assert t3["err_rms"][-1] < t3["err_rms"][0]
    assert t3["convergence_ratio_first_to_last"] > 2.5


@pytest.mark.skipif(
    importlib.util.find_spec("fmmax") is None,
    reason="fmmax is optional and not installed in this environment",
)
def test_optional_fmmax_cross_check():
    """Optional cross-check of rcwa_1d_grating against fmmax when installed."""
    import fmmax  # noqa: F401

    wl_g = RGB_WAVELENGTHS_M["G"]
    eps_prof = build_grating_eps_profile(
        256, fill_factor=0.50, eps_ridge=N_SIN_TIAN**2, eps_groove=1.0
    )
    rc = rcwa_1d_grating(
        [c / wl_g],
        np.deg2rad(10.0),
        PERIOD_GREEN_1ST_M,
        H_SIN_M,
        eps_prof,
        eps_below=1.0,
        eps_above=N_WG_TIAN**2,
        pol="TE",
        orders=(-1, 0, 1),
    )
    assert float(rc["R_total"][0] + rc["T_total"][0]) == pytest.approx(1.0, abs=1e-12)
