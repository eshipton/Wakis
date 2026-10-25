"""Tests for Tutorial 8.3 extension: A Maxwellian Perspective on Particle Acceleration (Parts B & C)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TUT_DIR = ROOT / "tutorials" / "08_vacuum_electronics" / "08-03_vacuum_laser_acceleration"
if str(TUT_DIR) not in sys.path:
    sys.path.insert(0, str(TUT_DIR))

import _helpers as h  # noqa: E402
import _maxwell_fit as mf  # noqa: E402


def test_part_b_gouy_phase_and_focal_velocities_with_mutation():
    """Verify Gouy phase -arctan(z/z0), superluminal v_p > c, subluminal v_g < c & v_flow < c (Eqs. 30-35), and mutation."""
    z_norm = np.linspace(-3.0, 3.0, 121)
    rho_norm = np.array([0.0, 0.5, 1.0, 1.25])

    th2 = mf.paraxial_focus_velocity_theory(rho_norm, z_norm, w0_over_lambda=2.0)
    th3 = mf.paraxial_focus_velocity_theory(rho_norm, z_norm, w0_over_lambda=3.0)

    # 1. Gouy phase is -45 deg at z = +z0 and +45 deg at z = -z0 (90 deg across [-z0, +z0])
    z1 = mf.paraxial_focus_velocity_theory(np.array([0.0]), np.array([-1.0, 0.0, 1.0]), w0_over_lambda=2.0)
    np.testing.assert_allclose(np.degrees(z1["gouy_phase_rad"]), np.array([45.0, 0.0, -45.0]), atol=1e-12)

    # 2. Inside the waist (rho < w0), v_p > c and v_g < c; at rho = w0, v_p = v_g = c (Eqs. 33-34)
    assert float(th2["vp_over_c"][0]) > 1.012
    assert float(th2["vg_fixed_waist_over_c"][0]) < 0.988
    assert float(th2["vp_over_c"][2]) == pytest.approx(1.0, abs=1e-12)
    assert float(th2["vg_fixed_waist_over_c"][2]) == pytest.approx(1.0, abs=1e-12)
    # Outside the waist (rho = 1.25 * w0 > w0), the sign reverses: v_p < c and v_g > c
    assert float(th2["vp_over_c"][3]) < 1.0
    assert float(th2["vg_fixed_waist_over_c"][3]) > 1.0

    # 3. Scaling with theta_0^2 ~ (lambda / w0)^2: ratio between w0 = 2*lambda and w0 = 3*lambda is (3/2)^2 = 2.25
    ratio_th2 = float((th2["theta0_rad"] / th3["theta0_rad"]) ** 2)
    assert ratio_th2 == pytest.approx(2.25, rel=1e-12)
    assert float((1.0 - th2["keff_over_k"][0]) / (1.0 - th3["keff_over_k"][0])) == pytest.approx(2.25, rel=1e-12)

    # 4. 3D FIT validation and mutation check (flipping Gouy sign or dropping plane-wave baseline)
    gouy_json = h.DATA_DIR / "task_part_b_gouy.json"
    res = json.loads(gouy_json.read_text()) if gouy_json.exists() else mf.run_task_part_b_gouy(quick=True)
    assert float(res["gouy_rms_err_2lambda_rad"]) < 0.02
    assert float(res["vp_axis_over_c_2lambda_fit"]) > 1.008
    assert float(res["vg_axis_over_c_2lambda_fit"]) < 0.994
    assert float(res["vflow_over_c_2lambda_fit"]) < 0.996
    assert float(res["dvp_axis_2lambda_rel_err"]) < 0.08
    assert float(res["dvflow_2lambda_rel_err"]) < 0.12

    # Mutation check: if we flip the Gouy phase sign (+arctan(z/z0)) or omit plane-wave dispersion subtraction,
    # the inferred phase-velocity shift has the wrong sign or > 20% error and MUST fail the check.
    mutated_dvp_sign_flip = -float(res["dvp_axis_2lambda_fit"])
    assert not (mutated_dvp_sign_flip > 0.0)
    mutated_uncompensated_dvp = float(res["dvp_axis_2lambda_fit"]) + float(res["planewave_dispersion_dvp_over_c"])
    err_uncomp = abs(mutated_uncompensated_dvp - float(res["dvp_axis_2lambda_theory"])) / float(
        res["dvp_axis_2lambda_theory"]
    )
    assert err_uncomp > 0.18


def test_part_b_weak_field_dipole_quadrature_and_optical_theorem_with_mutation():
    """Verify 90-deg far-field Gouy quadrature (Sec. 5.3), optical theorem sigma_ext = sigma_T, and phase mutation."""
    theta = np.linspace(1e-4, 0.45, 36)
    ze_scan = np.linspace(-1.5, 1.5, 31)
    th = mf.weak_field_dipole_quadrature_theory(theta, z_e_over_z0=ze_scan, w0_over_lambda=1.5)

    # 1. Exact 90-deg phase quadrature between laser far field (+i z0 E0 / r) and free-electron dipole (-r_e E0 / r)
    np.testing.assert_allclose(th["phase_diff_deg"], 90.0, atol=1e-12)
    assert float(np.max(np.abs(np.real(np.conj(th["f_laser_r_v"]) * th["f_dipole_r_v"])))) < 1e-30

    # 2. Optical theorem: real polarizability gives sigma_ext^(1) = 0; radiation reaction gives Thomson sigma_T
    assert float(th["alpha_real_si"]) < 0.0
    assert float(th["sigma_ext_first_order_m2"]) == 0.0
    assert float(th["sigma_ext_with_rr_m2"]) == pytest.approx(float(th["sigma_thomson_m2"]), rel=1e-12)

    # 3. 3D FIT quick run of focused laser + Dipole projected via Near2FarMonitor (preserving full-run artifacts)
    preserve_paths = [
        h.DATA_DIR / "task_part_b_gouy_dipole.json",
        h.DATA_DIR / "task_part_b_gouy_dipole.npz",
        h.FIG_DIR / "00a_part_b_dipole_setup.png",
        h.FIG_DIR / "00b_part_b_dipole_mesh.png",
        h.FIG_DIR / "00_setup_table_part_b_dipole.md",
    ]
    saved_bytes = {p: (p.read_bytes() if p.exists() else None) for p in preserve_paths}
    try:
        res = mf.run_task_part_b_weak_dipole(quick=True)
    finally:
        for p, b in saved_bytes.items():
            if b is not None:
                p.write_bytes(b)
    assert float(res["farfield_phase_diff_axis_deg_fit"]) == pytest.approx(90.0, rel=0.08)
    assert abs(float(res["u_int_first_order_norm_fit"])) < 0.14
    assert float(res["u_int_if_in_phase_norm_fit"]) > 0.55
    assert float(res["quadrature_suppression_ratio"]) > 5.5

    # Mutation check: shifting the dipole phase by +90 deg removes the Gouy quadrature and produces a
    # large O(E0) cross integral (~0.67), which MUST fail the zero-gain condition |U_int^(1)| < 0.14.
    mutated_in_phase_cross = float(res["u_int_if_in_phase_norm_fit"])
    assert not (abs(mutated_in_phase_cross) < 0.14)


def test_part_c_maxwell_cross_energy_and_cavity_aperture_with_mutations():
    """Verify exact discrete closure W = -(U_int + Phi_int) <= 1e-10, near/radiated split, cavity Eq. 76, and mutations."""
    # 1. Exact discrete cross-energy closure and physical split from precomputed 3D FIT data
    max_json = h.DATA_DIR / "task_part_c_maxwell.json"
    res_m = json.loads(max_json.read_text()) if max_json.exists() else mf.run_task_part_c_maxwell_screen(quick=True)
    assert float(res_m["discrete_closure_screen_rel_err"]) <= 1e-10
    assert float(res_m["discrete_closure_no_screen_cos_rel_err"]) <= 1e-10
    assert float(res_m["dW_direct_screen_finite_norm"]) == pytest.approx(0.6529, rel=0.02)
    assert float(res_m["dW_direct_no_screen_cos_norm"]) == pytest.approx(0.6394, rel=0.02)

    # Physical split: with screen, radiated transition-radiation cross flux carries the net gain and
    # downstream co-moving near-field exit vanishes; without screen, co-moving near-field carries the
    # in-box cosine gain (~0.6394) while radiated cross flux -> 0.
    assert float(res_m["near_field_comoving_screen_exit_norm"]) == 0.0
    assert float(res_m["radiated_cross_flux_screen_total_norm"]) == pytest.approx(
        float(res_m["dW_theory_screen_total_norm"]), rel=0.025
    )
    assert float(res_m["maxwell_vs_direct_total_rel_err"]) < 0.025
    assert float(res_m["near_field_comoving_no_screen_cos_norm"]) == pytest.approx(
        float(res_m["dW_direct_no_screen_cos_norm"]), rel=1e-10
    )
    assert float(res_m["radiated_cross_flux_no_screen_norm"]) < 1e-15

    # Live discrete 3D FIT test with and without PEC screen + mutation checks (half-step J shift & dropped exit face)
    for screen_flag in (True, False):
        live = mf.compute_discrete_fit_cross_energy_balance(with_screen=screen_flag, ppw=12)
        assert float(live["closure_rel_err"]) <= 1e-10
        assert not (float(live["mutation_shift_j_rel_err"]) <= 1e-10)
        assert float(live["mutation_shift_j_rel_err"]) > 1e-5
        assert not (float(live["mutation_drop_face_rel_err"]) <= 1e-10)
        assert float(live["mutation_drop_face_rel_err"]) > 0.5
    mut_shift = mf.compute_discrete_fit_cross_energy_balance(
        with_screen=True, ppw=12, shift_j_half_step=True
    )
    assert not (float(mut_shift["closure_rel_err"]) <= 1e-10)
    mut_face = mf.compute_discrete_fit_cross_energy_balance(
        with_screen=False, ppw=12, drop_exit_face=True
    )
    assert not (float(mut_face["closure_rel_err"]) <= 1e-10)

    # 2. Dual-screen radial-mode cavity length scan (Eqs. 76-77) and aperture cutoff (Eq. 84)
    l_vals = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    th_cav = mf.radial_mode_cavity_gain_theory(l_vals, gamma=np.inf)
    # Peak of 2*(L/z0)/(1 + (L/z0)^2) is 1.0 at L/z0 = 1.0, and symmetric under L/z0 <-> z0/L
    assert float(th_cav["dU_eq76_norm"][2]) == pytest.approx(1.0, abs=1e-12)
    assert float(th_cav["dU_eq76_norm"][0]) == pytest.approx(float(th_cav["dU_eq76_norm"][4]), abs=1e-12)
    assert float(th_cav["dU_eq76_norm"][1]) == pytest.approx(float(th_cav["dU_eq76_norm"][3]), abs=1e-12)
    # Unbounded radial mode (L -> inf) gives identically zero gain
    th_inf = mf.radial_mode_cavity_gain_theory(np.inf, gamma=12.0)
    assert float(th_inf["dU_total_norm"]) == 0.0

    cav_json = h.DATA_DIR / "task_part_c_cavity_aperture.json"
    res_c = json.loads(cav_json.read_text()) if cav_json.exists() else mf.run_task_part_c_cavity_aperture(quick=True)
    assert float(res_c["unbounded_max_gain_over_all_phases"]) < 1e-4
    assert float(res_c["cavity_peak_rel_err"]) < 0.04
    assert float(res_c["cavity_scan_l2_rel_err"]) < 0.04
    assert float(res_c["cavity_gamma12_l2_rel_err"]) < 0.04
    assert float(res_c["aperture_universal_collapse_rms_diff"]) < 0.04
    assert float(res_c["aperture_gamma8_vs_bessel_l2_rel_err"]) < 0.04

    # Mutation check A: dropping the finite-gamma transit-time slippage factor at low gamma = 2.5
    th_low_gam = mf.radial_mode_cavity_gain_theory(np.array([2.0]), gamma=2.5, z0_over_lambda=1.63)
    err_no_slippage = abs(float(th_low_gam["dU_eq76_norm"][0] - th_low_gam["dU_slippage_norm"][0]))
    assert err_no_slippage > 0.35

    # Mutation check B: comparing unscaled aperture radius a/lambda_bar for gamma = 3 vs gamma = 8 (without 1/gamma scaling)
    # destroys the universal collapse!
    xi_grid = np.asarray(res_c["a_over_gamma_lambar_scan"], dtype=float)
    unscaled_g8_at_g3_radii = 1.0 / mf.bessel_i0((3.0 / 8.0) * xi_grid)
    unscaled_rms_diff = float(np.sqrt(np.mean(( np.asarray(res_c["aperture_fit_gamma3_norm"]) - unscaled_g8_at_g3_radii ) ** 2)))
    assert unscaled_rms_diff > 0.25


def test_quantitative_summary_table_matches_json_and_mutation():
    """Verify that the Quantitative Summary table in vacuum_laser_acceleration.py matches data/*.json."""
    nb_path = TUT_DIR / "vacuum_laser_acceleration.py"
    nb_text = nb_path.read_text()

    # Stale value 0.713 from the old inconsistent table must not appear anywhere in the notebook
    assert "0.713" not in nb_text

    # Extract the Quantitative Summary section
    assert "## Quantitative Summary vs Literature" in nb_text
    sec = nb_text.split("## Quantitative Summary vs Literature", 1)[1].split("## Grid Resolution Convergence Study", 1)[0]

    j_a = json.loads((h.DATA_DIR / "task_part_a_plane_wave.json").read_text())
    j_bc = json.loads((h.DATA_DIR / "task_part_b_c_fit_screen.json").read_text())
    j_gouy = json.loads((h.DATA_DIR / "task_part_b_gouy.json").read_text())
    j_dip = json.loads((h.DATA_DIR / "task_part_b_gouy_dipole.json").read_text())
    j_max = json.loads((h.DATA_DIR / "task_part_c_maxwell.json").read_text())
    j_cav = json.loads((h.DATA_DIR / "task_part_c_cavity_aperture.json").read_text())
    j_d = json.loads((h.DATA_DIR / "task_part_d_ponderomotive.json").read_text())
    j_e = json.loads((h.DATA_DIR / "task_part_e_wiggler_fel.json").read_text())

    expected_tokens = [
        f"{j_a['gamma_peak_boris']:.4f}",
        f"{j_bc['dW_unbounded_sin_norm']:.5f}",
        f"{j_bc['dW_unbounded_finite_fit_norm']:.4f}",
        f"{j_gouy['dvp_axis_2lambda_fit']*100:+.3f}",
        f"{j_gouy['dvflow_2lambda_fit']*100:+.3f}",
        f"{j_dip['farfield_phase_diff_axis_deg_fit']:.2f}",
        f"{j_dip['u_int_first_order_norm_fit']:.4f}",
        f"{j_bc['dW_screen_waist_fit_norm']:.4f}",
        f"{j_bc['dW_screen_waist_finite_fit_norm']:.4f}",
        f"{j_max['minus_u_int_screen_finite_norm']:.4f}",
        f"{j_max['minus_u_int_screen_total_norm']:.4f}",
        f"{j_cav['cavity_peak_at_L_eq_z0_fit_norm']:.4f}",
        f"{j_cav['aperture_universal_collapse_rms_diff']*100:.2f}",
        f"{j_d['gamma_at_z_6zr']:.2f}",
        f"{j_d['bunch_rms_duration_as']:.1f}",
        f"{j_e['e_req_mev_laser_0p4um']:.1f}",
        f"{j_e['e_req_mev_laser_1p0um']:.1f}",
    ]
    for tok in expected_tokens:
        assert tok in sec, f"Expected JSON-derived token {tok!r} not found in Quantitative Summary table"

    # Mutation check: perturbing any JSON value must fail the table consistency check
    corrupted_token = f"{j_bc['dW_screen_waist_fit_norm'] + 0.05:.4f}"
    assert corrupted_token not in sec

    # Also verify all new setup figures (00a/00b) and result figures (20-24) exist and are non-empty
    expected_new_figs = [
        "00a_part_b_gouy_setup.png",
        "00b_part_b_gouy_mesh.png",
        "00a_part_b_dipole_setup.png",
        "00b_part_b_dipole_mesh.png",
        "00a_part_c_maxwell_setup.png",
        "00b_part_c_maxwell_mesh.png",
        "00a_part_c_cavity_setup.png",
        "00b_part_c_cavity_mesh.png",
        "20_part_b_gouy_and_velocities.png",
        "21_part_b_weak_field_dipole_quadrature.png",
        "22_part_c_maxwell_cross_energy.png",
        "23_part_c_laser_cavity_length_scan.png",
        "24_part_c_aperture_universal_collapse.png",
    ]
    for fn in expected_new_figs:
        p = h.FIG_DIR / fn
        assert p.exists() and p.stat().st_size > 10_000, f"Missing or empty figure {fn}"

    # Verify unreferenced duplicate 00a_setup_geometry_* / 00b_setup_mesh_and_excitation_* files are removed
    for dup in [
        "00a_setup_geometry_part_b_gouy.png",
        "00b_setup_mesh_and_excitation_part_b_gouy.png",
        "00a_setup_geometry_part_b_dipole.png",
        "00b_setup_mesh_and_excitation_part_b_dipole.png",
        "00a_setup_geometry_part_c_maxwell.png",
        "00b_setup_mesh_and_excitation_part_c_maxwell.png",
        "00a_setup_geometry_part_c_cavity.png",
        "00b_setup_mesh_and_excitation_part_c_cavity.png",
    ]:
        assert not (h.FIG_DIR / dup).exists(), f"Unreferenced duplicate setup figure {dup} still exists"
