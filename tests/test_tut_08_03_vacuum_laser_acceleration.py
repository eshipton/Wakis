"""Tests for Tutorial 8.3: Laser acceleration of electrons in vacuum (after Zolotorev and colleagues)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as c_light

ROOT = Path(__file__).resolve().parents[1]
TUT_DIR = ROOT / "tutorials" / "08_vacuum_electronics" / "08-03_vacuum_laser_acceleration"
if str(TUT_DIR) not in sys.path:
    sys.path.insert(0, str(TUT_DIR))

import _helpers as h  # noqa: E402


def test_plane_wave_zero_net_gain_and_pusher_convergence():
    """Verify plane-wave figure-of-eight orbit, zero net gain, Boris/Vay O(dt^2) convergence, and B=0 mutation."""
    lambda_m = 0.8e-6
    k0 = 2.0 * np.pi / lambda_m
    omega0 = k0 * c_light
    t_opt = lambda_m / c_light
    a0 = 2.2
    sigma_eta = 2.5 * (2.0 * np.pi)

    eta = np.linspace(-4.8 * sigma_eta, 4.8 * sigma_eta, 24001)
    ex = h.plane_wave_exact_solution(
        eta, a0=a0, omega0_rad_s=omega0, sigma_eta_rad=sigma_eta, carrier_phase_rad=0.5 * np.pi
    )

    # Peak intra-pulse energy gamma_max = 1 + a0^2 / 2 and zero net energy after pulse passes
    assert float(np.max(ex["gamma"])) == pytest.approx(1.0 + 0.5 * a0**2, rel=1e-4)
    assert float(abs(ex["gamma"][-1] - 1.0)) < 1e-9

    # Closed figure-of-eight lemniscate 16 * (k0 * z_tilde)^2 = (k0 * x)^2 * (a0^2 - (k0 * x)^2)
    f8 = h.figure_of_eight_orbit(np.linspace(0.0, 2.0 * np.pi, 201), a0=a0, k0_rad_m=k0)
    np.testing.assert_allclose(f8["lemniscate_residual"], 0.0, atol=1e-13)

    # Boris and Vay pusher validation + second-order O(dt^2) convergence
    def _field_fn(r_m: np.ndarray, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        eta_val = omega0 * (t_s - r_m[2] / c_light)
        pw = h.plane_wave_vector_potential_and_fields(
            eta_val,
            a0=a0,
            omega0_rad_s=omega0,
            sigma_eta_rad=sigma_eta,
            carrier_phase_rad=0.5 * np.pi,
        )
        return np.array([float(pw["Ex"]), 0.0, 0.0]), np.array([0.0, float(pw["By"]), 0.0])

    t_min, t_max = float(ex["t_s"][0]), float(ex["t_s"][-1])
    r0 = np.array([float(ex["x_m"][0]), 0.0, float(ex["z_m"][0])])
    u0 = np.array([float(ex["ux"][0]), 0.0, float(ex["uz"][0])])

    n_coarse = int(np.ceil((t_max - t_min) / (t_opt / 45)))
    n_fine = int(np.ceil((t_max - t_min) / (t_opt / 180)))
    t_coarse = np.linspace(t_min, t_max, n_coarse)
    t_fine = np.linspace(t_min, t_max, n_fine)

    b_coarse = h.push_boris_relativistic(r0, u0, t_coarse, _field_fn)
    b_fine = h.push_boris_relativistic(r0, u0, t_fine, _field_fn)
    v_coarse = h.push_vay_relativistic(r0, u0, t_coarse, _field_fn)
    v_fine = h.push_vay_relativistic(r0, u0, t_fine, _field_fn)

    g_ex_c = np.interp(t_coarse, ex["t_s"], ex["gamma"])
    g_ex_f = np.interp(t_fine, ex["t_s"], ex["gamma"])
    err_b_c = float(np.linalg.norm(b_coarse["gamma"] - g_ex_c) / np.linalg.norm(g_ex_c))
    err_b_f = float(np.linalg.norm(b_fine["gamma"] - g_ex_f) / np.linalg.norm(g_ex_f))
    err_v_c = float(np.linalg.norm(v_coarse["gamma"] - g_ex_c) / np.linalg.norm(g_ex_c))
    err_v_f = float(np.linalg.norm(v_fine["gamma"] - g_ex_f) / np.linalg.norm(g_ex_f))

    # 4x smaller dt -> ~16x smaller L2 error (second-order accuracy)
    assert err_b_f < 2e-3
    assert err_v_f < 2.5e-3
    assert (err_b_c / err_b_f) > 13.0
    assert (err_v_c / err_v_f) > 13.0
    # Zero net energy gain after the pulse passes
    assert abs(float(b_fine["gamma"][-1]) - 1.0) < 1e-6
    assert abs(float(v_fine["gamma"][-1]) - 1.0) < 1e-6

    # Mutation check: removing magnetic field B -> 0 eliminates the v x B longitudinal ponderomotive force
    def _mutated_no_b_field(r_m: np.ndarray, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        e_vec, _ = _field_fn(r_m, t_s)
        return e_vec, np.zeros(3)

    b_mut = h.push_boris_relativistic(r0, u0, t_fine, _mutated_no_b_field)
    assert float(np.max(np.abs(b_mut["u"][:, 2] - u0[2]))) < 1e-12
    assert float(np.max(np.abs(b_mut["u"][:, 2]))) < 1e-8
    assert float(np.max(b_mut["gamma"])) < 0.78 * float(np.max(b_fine["gamma"]))


def test_lawson_woodward_unterminated_vs_screen_and_mutation():
    """Verify 3D FIT screen-terminated energy gain vs Lawson-Woodward zero when screen is removed."""
    # 1. Analytical total-derivative identity: screen at waist z_s = 0 vs unbounded z_s -> +inf
    s_opt = 1.1209
    p_laser = 1.0e9
    th_screen = h.crossed_gaussian_energy_gain_theory(0.0, s_param=s_opt, p_laser_w=p_laser)
    th_unbounded = h.crossed_gaussian_energy_gain_theory(np.inf, s_param=s_opt, p_laser_w=p_laser)
    expected_opt_norm = (2.0 / np.sqrt(np.pi)) * (1.0 - np.exp(-(s_opt**2))) / s_opt
    assert float(th_screen["dW_norm"]) == pytest.approx(expected_opt_norm, rel=1e-12)
    assert float(th_screen["dW_norm"]) == pytest.approx(0.720101, rel=1e-4)
    assert float(th_unbounded["dW_norm"]) < 1e-15

    # 2. 3D Wakis FIT simulation with conformal PEC screen at z_s = 0 vs mutation (remove screen -> dW -> 0)
    w0_m = 0.95 * h.DEFAULT_LAMBDA_L_M
    fit_unb = h.run_single_fit_gaussian_beam(
        w0_m=w0_m,
        lambda_m=h.DEFAULT_LAMBDA_L_M,
        p_laser_w=p_laser,
        ppw_z=12,
        ppw_xy_core=10,
        n_pml=6,
        z_s_m=None,
    )
    fit_scr = h.run_single_fit_gaussian_beam(
        w0_m=w0_m,
        lambda_m=h.DEFAULT_LAMBDA_L_M,
        p_laser_w=p_laser,
        ppw_z=12,
        ppw_xy_core=10,
        n_pml=6,
        z_s_m=0.0,
    )
    tr_scr = h.evaluate_fit_trajectory_gain(
        fit_scr, s_param=s_opt, ref_unbounded_res=fit_unb, include_far_field_tails=True
    )
    tr_unb = h.evaluate_fit_trajectory_gain(
        fit_unb, s_param=s_opt, ref_unbounded_res=fit_unb, include_far_field_tails=True
    )

    # With PEC screen at waist: matches analytical optimum 0.7201 within coarse-grid tolerance
    assert float(tr_scr["dW_total_norm"]) == pytest.approx(expected_opt_norm, rel=0.04)
    # Mutation (remove PEC screen): net gain collapses by an order of magnitude toward Lawson-Woodward zero
    assert float(abs(tr_unb["dW_total_sin_norm"])) < 0.12 * float(tr_scr["dW_total_norm"])


def test_polarization_cosine_law_and_crossing_angle_optimization():
    """Verify Delta W proportional to cos(phi_pol) and maximum at optimal crossing angle s_opt ~ 1.12."""
    phi_deg = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
    phi_rad = np.radians(phi_deg)
    th_pol = h.crossed_gaussian_energy_gain_theory(
        0.0, s_param=1.1209, p_laser_w=1.0e9, phi_pol_rad=phi_rad
    )
    dw_0 = float(th_pol["dW_sin_ev"][0])
    np.testing.assert_allclose(th_pol["dW_sin_ev"], dw_0 * np.cos(phi_rad), rtol=1e-12, atol=1e-9)
    # At s-polarization (phi_pol = 90 deg), electric field is orthogonal to crossing plane -> zero gain
    assert abs(float(th_pol["dW_max_ev"][3])) < 1e-9

    # Crossing angle scan: s_opt = 1.1209 strictly exceeds sub-optimal s = 0.4 and s = 2.2
    s_grid = np.array([0.4, 1.1209, 2.2])
    th_s = h.crossed_gaussian_energy_gain_theory(0.0, s_param=s_grid, p_laser_w=1.0e9)
    assert float(th_s["dW_norm"][1]) > float(th_s["dW_norm"][0]) * 1.25
    assert float(th_s["dW_norm"][1]) > float(th_s["dW_norm"][2]) * 1.38


def test_ponderomotive_scaling_and_crater_focusing_with_mutation():
    """Verify gamma_opt = 1 + a^2/2, Z_{R,opt} = a^2 c tau / 2, ODE peak at kappa=1, and crater blowout mutation."""
    tau_s = 12.0e-15
    a_vals = np.array([2.0, 4.0, 6.0, 8.0])
    opt = h.ponderomotive_optimal_parameters(a_vals, tau_s=tau_s)
    np.testing.assert_allclose(opt["gamma_opt"], 1.0 + 0.5 * a_vals**2, rtol=1e-12)
    np.testing.assert_allclose(opt["z_r_opt_m"], 0.5 * (a_vals**2) * c_light * tau_s, rtol=1e-12)
    # Scaling ratio Z_{R,opt}(a=8) / Z_{R,opt}(a=2) == (8/2)^2 == 16
    assert float(opt["z_r_opt_m"][3] / opt["z_r_opt_m"][0]) == pytest.approx(16.0, rel=1e-12)

    # On-axis relativistic ponderomotive ODE peaks at kappa = Z_R / Z_{R,opt} = 1.0
    kappas = np.array([0.4, 0.7, 1.0, 1.4, 2.0, 2.8])
    ode = h.solve_ponderomotive_on_axis_ode(kappas, a0=7.4)
    idx_max = int(np.argmax(ode["gamma_final"]))
    assert float(kappas[idx_max]) == pytest.approx(1.0, abs=1e-12)
    assert float(ode["gamma_over_gamma_opt"][idx_max]) == pytest.approx(0.967, rel=0.02)

    # Crater profile rim at r_rim / sigma = sqrt(2 * (1 - 1 / alpha_mode)) = 0.6324555 for alpha_mode = 1.25
    crater = h.crater_intensity_profile(np.array([0.0, 0.4, 0.6324555, 1.2]), alpha_mode=1.25)
    assert float(crater["r_rim_over_sigma"]) == pytest.approx(np.sqrt(0.4), rel=1e-6)
    # Inside crater (r/sigma = 0.4 < r_rim), dI/dr > 0 (restoring inward ponderomotive force)
    assert float(crater["di_tot_d_r_norm"][1]) > 0.0
    # Outside crater rim (r/sigma = 1.2 > r_rim), dI/dr < 0 (outward defocusing force)
    assert float(crater["di_tot_d_r_norm"][3]) < 0.0

    # 3D bunch push + mutation check: crater mode (alpha = 1.25) vs single Gaussian (alpha = 0.0)
    b_crater = h.push_ponderomotive_bunch_3d(a0=7.4, tau_s=tau_s, alpha_mode=1.25, n_particles=120)
    b_gauss = h.push_ponderomotive_bunch_3d(a0=7.4, tau_s=tau_s, alpha_mode=0.0, n_particles=120)
    max_x_crater = float(np.max(np.abs(b_crater["x_m"])))
    max_x_gauss = float(np.max(np.abs(b_gauss["x_m"])))
    assert max_x_crater < 0.25e-6
    assert max_x_gauss > 50.0 * max_x_crater
    assert 35.0 <= float(b_crater["sigma_t_as"]) <= 75.0


def test_laser_wiggler_resonance_and_fermi_brightness():
    """Verify head-on laser wiggler resonance and Fermi brightness B_F scaling (Zolotorev 2002)."""
    # 10 keV X-rays: optical laser wiggler (0.4-1.0 um) needs 15-28 MeV vs ~13.7 GeV for 3 cm static undulator
    req_0p4 = h.required_electron_energy_for_xray(10.0e3, 0.4e-6, a0_wiggler=1.0)
    req_1p0 = h.required_electron_energy_for_xray(10.0e3, 1.0e-6, a0_wiggler=1.0)
    req_stat = h.required_electron_energy_for_xray(
        10.0e3, 3.0e-2, a0_wiggler=2.5, is_static_undulator=True
    )
    assert 15.0 <= float(req_0p4["energy_req_mev"]) <= 19.0
    assert 25.0 <= float(req_1p0["energy_req_mev"]) <= 29.0
    assert float(req_stat["energy_req_mev"]) > 1.1e4

    # Fermi brightness B_F = N * lambda_c_bar^3 / (eps_x * eps_y * eps_z) ~ 0.96e-6 for Zolotorev (2002)
    fel_hi = h.fermi_brightness_and_fel_gain(
        n_electrons=1.0e5, eps_inv_x_m=2.0e-9, eps_inv_y_m=2.0e-9, eps_inv_z_m=1.5e-9
    )
    fel_lo = h.fermi_brightness_and_fel_gain(
        n_electrons=0.1, eps_inv_x_m=2.0e-9, eps_inv_y_m=2.0e-9, eps_inv_z_m=1.5e-9
    )
    assert float(fel_hi["b_f"]) == pytest.approx(0.9595e-6, rel=1e-2)
    assert 80.0 <= float(fel_hi["m_a_periods"]) <= 160.0
    assert float(fel_lo["m_a_periods"] / fel_hi["m_a_periods"]) == pytest.approx(1.0e6, rel=1e-6)


def test_precomputed_data_and_figures_exist():
    """Verify that all precomputed task data files, static PNGs, and GIFs (< 5 MB) exist."""
    for task_name in h.TASKS:
        npz_p = h.DATA_DIR / f"task_{task_name}.npz"
        json_p = h.DATA_DIR / f"task_{task_name}.json"
        assert npz_p.exists() and npz_p.stat().st_size > 100
        assert json_p.exists() and json_p.stat().st_size > 20
        meta = json.loads(json_p.read_text())
        assert meta["task"] == task_name

    expected_pngs = [
        "01_setup_and_mesh_schematic.png",
        "02_part_a_plane_wave_orbits.png",
        "03_part_b_paraxial_and_lawson_woodward.png",
        "04_part_c_screen_polarization_and_angle.png",
        "05_part_d_ponderomotive_scaling_and_crater.png",
        "06_part_d_attosecond_phase_space.png",
        "07_part_e_laser_wiggler_and_brightness.png",
        "08_fit_resolution_convergence.png",
    ]
    for png_name in expected_pngs:
        p = h.FIG_DIR / png_name
        assert p.exists() and p.stat().st_size > 10_000

    expected_gifs = [
        "plane_wave_figure_of_eight.gif",
        "fit_beam_screen_reflection.gif",
        "ponderomotive_crater_bunching.gif",
    ]
    for gif_name in expected_gifs:
        p = h.FIG_DIR / gif_name
        assert p.exists()
        assert 10_000 < p.stat().st_size < 5 * 1024 * 1024
