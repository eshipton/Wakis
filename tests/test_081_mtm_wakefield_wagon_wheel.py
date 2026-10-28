"""Tests for Tutorial 08-08: Reversed-Cherenkov wakefields in a wagon-wheel metamaterial structure.

Validates a model inspired by X. Lu et al., Phys. Rev. Lett. 122, 014801 (2019), across all six parts:
1. Wagon-wheel metamaterial geometry, empty-pipe TM01 cutoff, and upstream coaxial port reflectometry.
2. 3D Bloch unit-cell cold test (negative group velocity, synchronous crossing near 11.42 GHz, r/Q) and 3-level mesh convergence.
3. Finite-structure wakefield length scaling, 25 MW single-bunch RF power, and reversed-Cherenkov (-z) vs forward-control (+z) Poynting directionality.
4. 3D FIT two-bunch linear superposition (to double-precision round-off), 4*cos^2(Delta_psi/2) phase scan, and 85 nC / 80 MW comparison.
5. Analytic backward-wave pulse duration and output power derivation vs 3D FIT wakefield simulation.
6. Unified control-volume Poynting energy balance and Quantitative Summary table-to-JSON consistency.

Every physical identity includes an explicit mutation check that alters the code or
parameters under test and asserts that the validation check then fails.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = REPO_ROOT / "tutorials" / "08_vacuum_electronics" / "08-08_mtm_wakefield_wagon_wheel"
DATA_DIR = Path(os.environ.get("WAKIS_T88_DATA_DIR", str(TUTORIAL_DIR / "data")))
FIGURES_DIR = Path(os.environ.get("WAKIS_T88_FIGURES_DIR", str(TUTORIAL_DIR / "figures")))

if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    C0,
    WagonWheelMTMSpec,
    _build_length_scaling_dict,
    aggregate_all_tutorial_data,
    analytic_backward_wave_loss_per_meter_j_per_m,
    analytic_backward_wave_power_w,
    analytic_backward_wave_pulse_length_s,
    analytic_sws_traveling_wave_impedance_ohm,
    build_dummy_all_json_dicts,
    circular_waveguide_tm01_cutoff_hz,
    compute_three_level_gci,
    derive_pre_asymptotic_scaling_slopes,
    evaluate_bloch_unit_cell,
    format_gci_line,
    gaussian_bunch_form_factor,
    measure_empty_waveguide_tm01_cutoff_fit,
    measure_filter_step_rise_time_ns,
    measure_upstream_port_reflection,
    reflection_induced_energy_error_bound,
    reflection_induced_peak_error_bound,
    relativistic_beam_kinematics,
    render_quantitative_summary_markdown,
    run_finite_mtm_wakefield,
    verify_two_bunch_fit_superposition,
)


def _check_cutoff_agreement(fc_eval_ghz: float, fc_ref_ghz: float, rtol: float = 0.005) -> bool:
    """Return True if evaluated TM01 cutoff agrees with reference within ``rtol``."""
    return abs(fc_eval_ghz - fc_ref_ghz) / fc_ref_ghz <= rtol


def _check_port_matching(gamma_sync: float, err_bound: float, expected_err_bound: float) -> bool:
    """Return True if upstream port reflection meets |Gamma| < 0.25 (RL > 12 dB, VSWR < 5/3) and error bound matches formula."""
    return (gamma_sync < 0.25) and (abs(err_bound - expected_err_bound) / expected_err_bound < 1e-6)


def _check_backward_cold_test(res: dict[str, float], fc_tm01_ghz: float) -> bool:
    """Return True if unit-cell Bloch mode is a sub-cutoff backward wave with negative energy velocity."""
    f_ghz = float(res["f_mode_ghz"])
    ve_c = float(res["ve_over_c"])
    rq_kohm = float(res["r_over_q_kohm_per_m"])
    return (
        (9.5 <= f_ghz < fc_tm01_ghz)
        and (ve_c < -0.05)
        and (15.0 <= rq_kohm <= 28.0)
    )


def _check_wake_directionality_and_scaling(
    upstream_fraction: float,
    slope_re_z_pk: float,
    slope_fwhm: float,
    slope_tp: float,
    slope_p_up: float,
    slope_bounds: dict[str, float],
    gamma_tap: float,
    a_w_fit: float,
    a_w_analytic: float,
    tol_a_w: float,
    gamma_pb_max: float | None = None,
) -> bool:
    """Return True if upstream fraction, short-N slopes, and affine loss slope a_W lie within derived bounds."""
    # 1. Upstream directionality bound over the drained ring-down window:
    #    A short-N broadband backward-wave packet spanning the [9.8, 13.0] GHz passband experiences
    #    (i) direct backward-to-forward scattering at the downstream taper during beam exit,
    #    (ii) upstream port reflection followed by forward transmission during the second cavity transit
    #         (each bounded across the passband by max_f |Gamma_tap(f)|^2), and
    #    (iii) synchronous transition scattering |Gamma_tap(f_s)|^2 at the taper boundary,
    #    so E_up / E_rad >= 1 - 2*|Gamma_max|^2 - |Gamma_tap(f_s)|^2.
    gamma_dir = gamma_tap if gamma_pb_max is None else gamma_pb_max
    min_upstream_frac = 1.0 - 2.0 * (gamma_dir**2) - (gamma_tap**2)
    # 2. First-principles pre-asymptotic slope intervals from derive_pre_asymptotic_scaling_slopes:
    #    t_p,eff(L) = (L + L_tap,eff)*(1/|v_g| + 1/u_0) + tau_filt with L_tap,eff in [0, 8 mm]
    #    and causal filter envelope offset tau_filt.
    # 3. Cycle-averaged peak power slope bound from single-pass port power reflection |Gamma|^2:
    #    Over L_max / L_min = 24 mm / 12 mm = 2, cavity standing-wave power variation <= 2*|Gamma|^2
    #    bounds |alpha_P| <= 2*|Gamma|^2 / ln(2).
    max_slope_p = 2.0 * (gamma_tap**2) / float(np.log(24.0 / 12.0))
    # 4. Affine passband beam-work slope a_W vs cold-test analytic loss per unit length
    #    a_W,analytic = q^2 * k_L * Phi^2 / (1 - v_g / u_0) (M3):
    rel_err_a_w = abs(float(a_w_fit) - float(a_w_analytic)) / max(abs(float(a_w_analytic)), 1e-30)
    return (
        (upstream_fraction >= min_upstream_frac)
        and (slope_bounds["slope_z_min"] <= slope_re_z_pk <= slope_bounds["slope_z_max"])
        and (slope_bounds["slope_fwhm_min"] <= slope_fwhm <= slope_bounds["slope_fwhm_max"])
        and (slope_bounds["slope_tp_min"] <= slope_tp <= slope_bounds["slope_tp_max"])
        and (abs(slope_p_up) <= max_slope_p)
        and (rel_err_a_w < float(tol_a_w))
    )


def _check_two_bunch_superposition_and_phase(
    rel_l2_err_e: float,
    in_phase_power_ratio: float,
    out_of_phase_power_ratio: float,
    ds_in_mm: float,
    ds_out_mm: float,
    s_ov_max_mm: float,
    gamma_tap: float,
) -> bool:
    """Return True if 3D FIT superposition holds to round-off and passband wake obeys finite-drain interference bounds."""
    # 1. Exact linearity of discrete 3D conformal FIT curl updates requires relative L2 superposition
    #    error < 1e-12 (governed by IEEE-754 float64 accumulation over ~100 steps).
    # 2. Finite-length envelope-shift bounds over the overlap window s <= s_ov_max_mm:
    #    In a short N=8 structure, the single-bunch wake envelope decays linearly with s over the
    #    evaluation window [0, s_ov_max_mm]. Shifting bunch 2 by Delta_s introduces a fractional
    #    amplitude mismatch Delta_A / A_0 <= Delta_s / s_ov_max_mm between the two wakes:
    #    - In phase (Delta_psi = 2*pi, Delta_s = ds_in_mm): (2 - ds_in_mm/s_ov_max_mm)^2 <= R_in <= 4*(1 + |Gamma|^2).
    #    - Out of phase (Delta_psi = 3*pi, Delta_s = ds_out_mm): envelope mismatch (ds_out_mm/s_ov_max_mm)^2
    #      plus port-reflection standing-wave floor |Gamma|^2 bounds R_out <= (ds_out_mm/s_ov_max_mm)^2 + |Gamma|^2.
    in_phase_min = (2.0 - float(ds_in_mm) / float(s_ov_max_mm)) ** 2
    in_phase_max = 4.0 * (1.0 + float(gamma_tap) ** 2)
    out_of_phase_max = (float(ds_out_mm) / float(s_ov_max_mm)) ** 2 + (float(gamma_tap) ** 2)
    return (
        (rel_l2_err_e < 1e-12)
        and (in_phase_min <= in_phase_power_ratio <= in_phase_max)
        and (0.0 <= out_of_phase_power_ratio <= out_of_phase_max)
    )


def _check_analytic_power_and_energy_balance(
    p_analytic_mw: float,
    p_fit_mw: float,
    tp_fit_all_n_ns: list[float],
    tp_eq1_with_tapers_all_n_ns: list[float],
    pulse_end_all_n_ns: list[float],
    t_end_window_all_n_ns: list[float],
    rf_period_all_n_ns: list[float],
    dt_filter_tol_ns: float,
    upstream_rad_fraction: float,
    power_refl_bound: float,
    dir_refl_bound: float,
    rel_err_closure_all_n: list[float],
    u_stored_over_w_all_n: list[float],
    u_stored_fit_mj: float,
    w_loss_bunch_mj: float,
    e_rad_live_mj: float,
    closure_rtol: float = 1e-3,
) -> bool:
    """Return True if analytic power matches 3D FIT within 2*|Gamma|^2, unclipped FIT pulse durations match tapered Eq. (1) within the filter rise time, U_bp(T_end)/W_beam,bp < 0.02 across all N, and control-volume energy balance closes."""
    # 1. Derived two-port reflection power tolerance:
    #    power_refl_bound = reflection_induced_energy_error_bound(gamma_tap) = |Gamma(f_s)|^2 per end coupler;
    #    for the symmetric finite cavity bounded by upstream and downstream 2-cell tapers, the combined
    #    two-port reflection and causal bandpass insertion bound is 2*|Gamma(f_s)|^2.
    # 2. Derived pulse-duration tolerance and non-clipping window condition:
    #    Each finite N-period structure has two 2-cell adiabatic tapers (L_taper = 4*p = 8 mm), so the
    #    total radiating slow-wave length is (N + 4)*p with Eq. (1) group-drain time t_p,Eq1((N+4)*p).
    #    The measured 20% envelope end must also occur strictly earlier than T_end - T_RF (more than
    #    one RF period before the end of the simulation window) so that it never measures the window.
    # 3. Window drainage condition (M1):
    #    With wake_len chosen from estimate_drainage_wake_length_m(N) for dispersive passband transit
    #    at |v_g,min| = 0.052c plus Fabry-Perot ring-down with |Gamma_tap| = 0.22, the residual passband
    #    stored energy fraction at T_end must satisfy U_bp(T_end) / W_beam,bp < 0.02 across N = 6, 8, 12.
    # 4. Control-volume Poynting energy-balance closure tolerance (S3):
    #    On the unified control volume [z_up, z_dn], discrete FIT Poynting conservation holds to
    #    O((omega_s*dt)^2) from second-order leapfrog time-averaging H^n = 0.5*(H^{n-1/2} + H^{n+1/2}),
    #    whose amplitude error factor is 1 - cos(omega_s*dt/2) ~ (1/8)*(omega_s*dt)^2 (or (1/4)*(omega_s*dt)^2
    #    in quadratic energy/flux products). For f_s ~ 11.42 GHz and dt <= 406 fs (CFL = 0.85 on h_min >= 0.25 mm),
    #    (omega_s*dt)^2 <= (2*pi*11.42e9*406e-15)^2 = 8.5e-4, giving a derived upper bound closure_rtol = 1e-3.
    n_ok = (
        len(tp_fit_all_n_ns)
        == len(tp_eq1_with_tapers_all_n_ns)
        == len(pulse_end_all_n_ns)
        == len(t_end_window_all_n_ns)
        == len(rf_period_all_n_ns)
        == len(u_stored_over_w_all_n)
        == 3
    )
    tp_unclipped_and_matches = n_ok and all(
        (tp_fit > 0.0)
        and ((t_end_win - t_end_20) > t_rf)
        and (abs(tp_fit - tp_eq1) <= dt_filter_tol_ns)
        for tp_fit, tp_eq1, t_end_20, t_end_win, t_rf in zip(
            tp_fit_all_n_ns,
            tp_eq1_with_tapers_all_n_ns,
            pulse_end_all_n_ns,
            t_end_window_all_n_ns,
            rf_period_all_n_ns,
        )
    )
    cv_closure_all_n_ok = len(rel_err_closure_all_n) == 3 and all(
        0.0 <= err < closure_rtol for err in rel_err_closure_all_n
    )
    drainage_all_n_ok = len(u_stored_over_w_all_n) == 3 and all(
        0.0 < ratio < 0.02 for ratio in u_stored_over_w_all_n
    )
    live_u_over_w = u_stored_fit_mj / max(w_loss_bunch_mj, 1e-30)
    live_rel_err_closure = abs(e_rad_live_mj + u_stored_fit_mj - w_loss_bunch_mj) / max(w_loss_bunch_mj, 1e-30)
    return (
        (abs(p_analytic_mw - p_fit_mw) / p_fit_mw <= 2.0 * power_refl_bound)
        and tp_unclipped_and_matches
        and (upstream_rad_fraction >= 1.0 - 2.0 * dir_refl_bound - power_refl_bound)
        and cv_closure_all_n_ok
        and drainage_all_n_ok
        and (0.0 < live_u_over_w < 0.02)
        and (live_rel_err_closure < closure_rtol)
    )


def _verify_summary_table_against_json(md_text: str, sm_dict: dict[str, float]) -> None:
    """Parse the 7-column Quantitative Summary table in ``md_text`` and assert every keyed row matches ``sm_dict``."""
    row_pattern = re.compile(
        r"^\|\s*(?P<desc>[^|]+?)\s*\|\s*(?P<sym>[^|]+?)\s*\|\s*(?P<bench>[^|]+?)\s*\|\s*"
        r"(?P<res>[^|]+?)\s*\|\s*(?P<conv>[^|]+?)\s*\|\s*(?P<err>[^|]+?)\s*\|\s*`(?P<key>[a-zA-Z0-9_]+)`\s*\|$"
    )
    matched_rows = 0
    for line in md_text.splitlines():
        raw = line.strip()
        if raw.startswith("# "):
            raw = raw[2:].strip()
        m = row_pattern.match(raw)
        if not m:
            continue
        key = m.group("key")
        assert key in sm_dict, f"Summary table references unknown JSON key: {key}"
        bench_cell = m.group("bench")
        conv_cell = m.group("conv")
        # M2: The summary table must not put the n_xy = 48 value in the "Paper / Theory Benchmark" column;
        # it has a dedicated "Mesh Convergence / GCI" column.
        assert "48" not in bench_cell, f"Benchmark column must not contain n_xy=48 value: {bench_cell}"
        assert len(conv_cell.strip()) > 0, f"Convergence column must not be empty for {key}"
        json_val = float(sm_dict[key])
        res_cell = m.group("res")
        num_match = re.search(r"`([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)`", res_cell)
        assert num_match is not None, f"Could not parse backticked number in table cell: {res_cell}"
        table_val = float(num_match.group(1))
        # For k_loss in units of 10^14 V/(C*m), scale json_val by 1e-14
        if abs(json_val) > 1e12 and abs(table_val) < 100.0:
            json_val = json_val * 1e-14
        rel_diff = abs(table_val - json_val) / max(abs(json_val), 1e-30)
        assert rel_diff < 5e-3, f"Table mismatch for {key}: table={table_val} vs json={json_val} (rel={rel_diff:.4e})"
        matched_rows += 1
    assert matched_rows >= 15, f"Expected at least 15 keyed rows in Quantitative Summary table, found {matched_rows}"


def test_geometry_spec_and_tm01_cutoff_with_mutations() -> None:
    """Verify wagon-wheel MTM geometry, 65 MeV kinematics, and 3D FIT empty-pipe TM01 cutoff (with mutations)."""
    spec = WagonWheelMTMSpec()
    gamma, beta, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    assert abs(gamma - 128.2018) < 0.01
    assert 0.99996 < beta < 1.0

    fc_analytic_ghz = circular_waveguide_tm01_cutoff_hz(spec.waveguide_radius_m) * 1e-9
    fc_fit_quick_ghz = measure_empty_waveguide_tm01_cutoff_fit(spec.waveguide_radius_m, quick=True)
    assert _check_cutoff_agreement(fc_fit_quick_ghz, fc_analytic_ghz, rtol=0.005)

    t1 = json.loads((DATA_DIR / "task_01_geometry_and_port_reflection.json").read_text(encoding="utf-8"))
    assert len(t1["paper_parameters"]) >= 10
    assert len(t1["assumed_parameters"]) >= 5
    for item in t1["paper_parameters"]:
        assert "line" in item["ref_lines"]
    for item in t1["assumed_parameters"]:
        assert len(item["reason"]) > 15

    # Mutation check: perturbing waveguide radius R_wg -> 1.20 * R_wg shifts TM01 cutoff by -16.7% and must fail
    fc_mutated_ghz = circular_waveguide_tm01_cutoff_hz(1.20 * spec.waveguide_radius_m) * 1e-9
    assert not _check_cutoff_agreement(fc_mutated_ghz, fc_analytic_ghz, rtol=0.005)


def test_port_reflection_and_standing_wave_error_bound_with_mutations() -> None:
    """Verify upstream coaxial port reflection |Gamma(f_s)| and passive abrupt-pipe reflection < 1 (with mutations)."""
    t1 = json.loads((DATA_DIR / "task_01_geometry_and_port_reflection.json").read_text(encoding="utf-8"))
    pr = t1["port_reflection"]

    # Standard microwave port-matching criterion: return loss RL = -20*log10(|Gamma|) > 12.04 dB
    # (|Gamma| < 0.25, VSWR < 5/3), corresponding to power transmission 1 - |Gamma|^2 > 0.9375.
    gamma_tap = float(pr["gamma_tapered_at_sync"])
    gamma_abr = float(pr["gamma_abrupt_at_sync"])
    assert gamma_tap < 0.25
    assert 0.80 < gamma_abr < 1.0
    assert float(pr["power_transmission_tapered_at_sync"]) > (1.0 - 0.25**2)

    # Live quick port-reflectometry check (Item 7): passive abrupt cutoff pipe reflection is strictly < 1 without clipping
    pr_live = measure_upstream_port_reflection(n_xy=32, dz_mm=0.50, quick=True)
    assert float(pr_live["gamma_tapered_at_sync"]) < 0.25
    assert 0.80 < float(pr_live["gamma_abrupt_raw_at_sync"]) < 1.0
    assert pr_live["gamma_abrupt_is_clipped"] is False
    assert np.all(np.asarray(pr_live["gamma_abrupt_empty_pipe_raw"], dtype=float) < 1.0)

    err_true = float(reflection_induced_peak_error_bound(gamma_tap, mutate_drop_factor_two=False))
    expected_err = float(pr["rel_error_bound_tapered"])
    assert _check_port_matching(gamma_tap, err_true, expected_err)

    # Mutation 1: dropping the factor of 2 on the cross-term in reflection_induced_peak_error_bound must fail
    err_mut = float(reflection_induced_peak_error_bound(gamma_tap, mutate_drop_factor_two=True))
    assert not _check_port_matching(gamma_tap, err_mut, expected_err)

    # Mutation 2: using the un-tapered abrupt empty circular pipe reflection must fail port matching
    assert not _check_port_matching(gamma_abr, err_true, expected_err)


def test_bloch_cold_test_negative_group_velocity_and_mesh_convergence_with_mutations() -> None:
    """Verify 3D Bloch cold-test backward dispersion, FFT fallback, v_E < 0, r/Q, and 3-level convergence (with live mutations)."""
    spec = WagonWheelMTMSpec()
    fc_tm01_ghz = circular_waveguide_tm01_cutoff_hz(spec.waveguide_radius_m) * 1e-9

    # 1. Live quick 3D Bloch unit-cell run on uniform nf=4 mesh (regression test for Item 14)
    res_uniform_quick = evaluate_bloch_unit_cell(
        phi_deg=27.43,
        n_feat=4,
        use_graded_mesh=False,
        forward_control=False,
        compute_eigenmode_metrics=True,
        quick=True,
        spec=spec,
    )
    assert _check_backward_cold_test(res_uniform_quick, fc_tm01_ghz)

    # 2. Explicit test of the ResonanceMonitor.signal() FFT fallback path when harminv returns no poles (Item 14)
    res_fft_fallback = evaluate_bloch_unit_cell(
        phi_deg=27.43,
        n_feat=4,
        use_graded_mesh=False,
        forward_control=False,
        compute_eigenmode_metrics=False,
        quick=True,
        force_fft_fallback=True,
        spec=spec,
    )
    assert 9.5 <= float(res_fft_fallback["f_mode_ghz"]) < fc_tm01_ghz

    # 3. Verify converged 3-level mesh study in task_02_cold_test_and_convergence.json
    t2 = json.loads((DATA_DIR / "task_02_cold_test_and_convergence.json").read_text(encoding="utf-8"))
    wb = t2["wakis_finest_nf8_benchmarks"]
    cu = t2["convergence_uniform"]
    cg = t2["convergence_graded"]

    assert _check_backward_cold_test(
        {
            "f_mode_ghz": wb["f_sync_ghz"],
            "ve_over_c": wb["ve_dft_over_c"],
            "r_over_q_kohm_per_m": wb["r_over_q_kohm_per_m"],
        },
        fc_tm01_ghz,
    )
    assert abs(float(wb["u_elec_over_u_mag"]) - 1.0) < 0.01

    # Grid resolution rules: >= 20 cells/wavelength and >= 4 cells across 1 mm feature at all 3 levels
    for key in ("nf4", "nf6", "nf8"):
        lv = t2["uniform_levels"][key]
        assert float(lv["cells_per_wavelength_lambda0"]) >= 20.0
        assert float(lv["cells_across_1mm_feature"]) >= 4.0

    # Monotone 3-level convergence of shunt impedance r/Q, loss factor k_L, and conformal |v_E|/c;
    # on production 44-cycle ring-down runs (rel_change_nf8_vs_nf6 < 0.002), f_s and uniform |v_E|/c are also monotone
    assert cu["r_over_q_kohm_per_m"]["is_monotone"] is True
    assert cu["k_loss_v_per_c_m"]["is_monotone"] is True
    assert cg["ve_over_c"]["is_monotone"] is True
    assert cg["r_over_q_kohm_per_m"]["is_monotone"] is True
    assert cg["k_loss_v_per_c_m"]["is_monotone"] is True
    if float(cu["f_mode_ghz"]["rel_change_nf8_vs_nf6"]) < 0.002:
        assert cu["f_mode_ghz"]["is_monotone"] is True
        assert cu["ve_over_c"]["is_monotone"] is True
        assert cg["f_mode_ghz"]["is_monotone"] is True

    # Thesis R_wg = 8.1 mm Bloch variant (Item 12): brackets paper's 11.42 GHz within 1%
    v81 = t2["variant_rwg_8p1mm_thesis"]
    assert abs(float(v81["tm01_cutoff_analytic_ghz"]) - 14.1657) < 0.002
    assert 11.20 < float(v81["nf6"]["f_mode_ghz"]) < 11.42
    assert -0.175 < float(v81["nf6"]["ve_over_c"]) < -0.150
    assert 19.0 < float(v81["nf6"]["r_over_q_kohm_per_m"]) < 24.0

    # Live Mutation 1: flipping the sign of k_bloch (+k_z -> -k_z) in evaluate_bloch_unit_cell reverses
    # the energy velocity v_E > 0 (+0.16c instead of -0.16c) and must fail _check_backward_cold_test
    res_mut_flip_k = evaluate_bloch_unit_cell(
        phi_deg=27.43,
        n_feat=4,
        use_graded_mesh=True,
        forward_control=False,
        compute_eigenmode_metrics=True,
        quick=True,
        mutate_flip_k_bloch_sign=True,
        spec=spec,
    )
    assert float(res_mut_flip_k["ve_over_c"]) > 0.05
    assert not _check_backward_cold_test(res_mut_flip_k, fc_tm01_ghz)

    # Live Mutation 2: removing the inductive radial spokes (forward_control=True) in a live unit-cell run
    # shifts the mode above the 14.34 GHz empty-pipe cutoff with positive velocity, failing _check_backward_cold_test
    res_mut_no_spokes = evaluate_bloch_unit_cell(
        phi_deg=27.43,
        n_feat=4,
        use_graded_mesh=True,
        forward_control=True,
        compute_eigenmode_metrics=True,
        quick=True,
        spec=spec,
    )
    assert float(res_mut_no_spokes["f_mode_ghz"]) > fc_tm01_ghz
    assert not _check_backward_cold_test(res_mut_no_spokes, fc_tm01_ghz)
    with pytest.raises(KeyError):
        _check_backward_cold_test({"f_mode_ghz": 11.42}, fc_tm01_ghz)


def test_wakefield_length_scaling_and_reversed_cherenkov_directionality_with_mutations() -> None:
    """Verify pre-asymptotic short-N scaling and upstream (-z) reversed-Cherenkov directionality (with mutations)."""
    spec = WagonWheelMTMSpec()
    _, _, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    t1 = json.loads((DATA_DIR / "task_01_geometry_and_port_reflection.json").read_text(encoding="utf-8"))
    t2 = json.loads((DATA_DIR / "task_02_cold_test_and_convergence.json").read_text(encoding="utf-8"))
    t3 = json.loads((DATA_DIR / "task_03_wakefield_and_directionality.json").read_text(encoding="utf-8"))
    ls = t3["length_scaling"]
    dc = t3["directionality_comparison_N8"]
    mc = t3["n8_transverse_mesh_convergence"]

    # Grid rule & 3-level transverse mesh convergence (M2):
    # Production runs use n_xy in [48, 64, 96] (3, 4, 6 cells across 1 mm spokes; [28, 36, 48] in --quick)
    assert len(mc["levels_n_xy"]) == 3
    assert len(mc["h_xy_mm"]) == 3
    is_quick_t3 = int(mc["levels_n_xy"][1]) < 64
    assert mc["levels_n_xy"] == ([28, 36, 48] if is_quick_t3 else [48, 64, 96])
    for k in ("6", "8", "12"):
        rn = t3["runs_mtm"][k]
        if is_quick_t3:
            assert int(rn["n_xy"]) >= 36
        else:
            assert int(rn["n_xy"]) >= 64
            assert float(rn["cells_across_1mm_spoke"]) >= 4.0
        assert float(rn["u_stored_final_fit_mj"]) > 0.0

    z1 = np.load(DATA_DIR / "task_01_port_reflection_spectra.npz")
    gamma_tap = float(t1["port_reflection"]["gamma_tapered_at_sync"])
    gamma_pb_max = float(np.max(np.asarray(z1["gamma_tapered_coax"], dtype=float)))

    # 3-level GCI verification (MF1 + MF3):
    # (1) Recompute compute_three_level_gci from stored values and h_xy_mm and compare every field with JSON to 1e-12;
    # (2) Assert recorded convergence_class equals the class implied by R = e32 / e21
    #     (monotone if 0 < R < 1, oscillatory if R < 0, divergent if R >= 1; no monotone requirement);
    # (3) Assert u_fine_rel >= max_i |f_i - f_96| / |f_96| for W, P_up, E_up;
    # (4) Synthetic unit tests on quadratic monotone data and oscillatory data (with mutation forcing Fs = 1.25).
    h_arr = [float(x) for x in mc["h_xy_mm"]]
    r12 = h_arr[0] / h_arr[1]
    for obs_key, gci_obs in mc["gci"].items():
        vals_obs = [float(x) for x in gci_obs["values"]]
        recomp = compute_three_level_gci(vals_obs, h_arr, p_formal=1.0)
        assert set(recomp.keys()) == set(gci_obs.keys()), f"Key mismatch for {obs_key}"
        for k_field, v_rec in recomp.items():
            v_json = gci_obs[k_field]
            if isinstance(v_rec, list):
                assert np.allclose(v_rec, v_json, rtol=0.0, atol=1e-12), f"{obs_key}.{k_field} list mismatch"
            elif isinstance(v_rec, (bool, str)) or v_rec is None:
                assert v_rec == v_json, f"{obs_key}.{k_field}: {v_rec} != {v_json}"
            else:
                assert abs(float(v_rec) - float(v_json)) <= 1e-12 * max(abs(float(v_rec)), 1.0), (
                    f"{obs_key}.{k_field}: {v_rec} != {v_json}"
                )

        e21 = vals_obs[1] - vals_obs[0]
        e32 = vals_obs[2] - vals_obs[1]
        r_ratio = float(gci_obs["R_ratio"])
        assert abs(r_ratio - (e32 / e21)) <= 1e-12
        expected_class = (
            "oscillatory"
            if r_ratio < 0.0
            else ("monotone" if (0.0 < r_ratio < 1.0) else "divergent")
        )
        assert gci_obs["convergence_class"] == expected_class
        if expected_class != "monotone":
            assert gci_obs["observed_order"] is None
            assert gci_obs["richardson_estimate"] is None
            assert float(gci_obs["fs_used"]) == 3.0
            assert float(gci_obs["p_used"]) == 1.0

    for env_key in ("w_beam_passband_cv_mj", "p_up_cycle_avg_max_mw", "e_up_mj"):
        gci_env = mc["gci"][env_key]
        vals_env = [float(x) for x in gci_env["values"]]
        f_fine = vals_env[2]
        max_rel_dev = max(abs(v_i - f_fine) / max(abs(f_fine), 1e-30) for v_i in vals_env)
        assert float(gci_env["u_fine_rel"]) >= max_rel_dev - 1e-12

    # Synthetic unit test (MF3.4a): f = 1 + 0.3 * h^2 on h = (1/3, 1/4, 1/6) recovers p = 2 to 1e-8
    # with Fs = 1.25 when p_formal = 2 is passed.
    h_syn = [1.0 / 3.0, 1.0 / 4.0, 1.0 / 6.0]
    f_syn_quad = [1.0 + 0.3 * (h**2) for h in h_syn]
    gci_syn_quad = compute_three_level_gci(f_syn_quad, h_syn, p_formal=2.0)
    assert gci_syn_quad["convergence_class"] == "monotone"
    assert gci_syn_quad["observed_order"] is not None
    assert abs(float(gci_syn_quad["observed_order"]) - 2.0) < 1e-8
    assert float(gci_syn_quad["fs_used"]) == 1.25
    assert abs(float(gci_syn_quad["richardson_estimate"]) - 1.0) < 1e-8

    # Synthetic unit test (MF3.4b): an oscillatory triple gives observed_order None, fs_used 3, richardson None,
    # and satisfies u_fine_rel >= max_i |f_i - f_fine| / |f_fine|; mutation forcing Fs = 1.25 must fail both.
    f_syn_osc = [1.020, 1.000, 1.005]
    gci_syn_osc = compute_three_level_gci(f_syn_osc, h_syn, p_formal=1.0)
    assert gci_syn_osc["convergence_class"] == "oscillatory"
    assert gci_syn_osc["observed_order"] is None
    assert gci_syn_osc["richardson_estimate"] is None
    assert float(gci_syn_osc["fs_used"]) == 3.0
    max_dev_osc = max(abs(v_i - f_syn_osc[2]) / abs(f_syn_osc[2]) for v_i in f_syn_osc)
    assert float(gci_syn_osc["u_fine_rel"]) >= max_dev_osc

    gci_syn_osc_mut = compute_three_level_gci(
        f_syn_osc, h_syn, p_formal=1.0, mutate_force_fs_1p25=True
    )
    assert float(gci_syn_osc_mut["fs_used"]) != 3.0
    assert float(gci_syn_osc_mut["u_fine_rel"]) < max_dev_osc

    # Mutation check on compute_three_level_gci: swapping coarse and medium levels on monotone data breaks monotonicity
    gci_mut_swap = compute_three_level_gci(
        f_syn_quad, h_syn, p_formal=2.0, mutate_swap_coarse_med=True
    )
    assert gci_mut_swap["is_monotone"] is False

    z_pk_arr = np.asarray(ls["re_z_pk_ohm"], dtype=float)
    fwhm_arr = np.asarray(ls["fwhm_ghz"], dtype=float)
    e_up_arr = np.asarray(ls["e_up_mj"], dtype=float)
    w_bp_arr = np.asarray(ls["w_beam_passband_cv_mj"], dtype=float)
    tp_arr = np.asarray(ls["pulse_duration_20pct_ns"], dtype=float)
    assert z_pk_arr[0] < z_pk_arr[1] < z_pk_arr[2]
    assert fwhm_arr[0] > fwhm_arr[1] > fwhm_arr[2]
    assert e_up_arr[0] < e_up_arr[1] < e_up_arr[2]
    assert w_bp_arr[0] < w_bp_arr[1] < w_bp_arr[2]
    assert tp_arr[0] < tp_arr[1] < tp_arr[2]
    assert ls["is_asymptotic_long_structure_regime"] is False

    # Affine length scaling (M3): E_up(L) = a_E * L + b_E and W_beam,bp,cv(L) = a_W * L + b_W
    assert float(ls["affine_a_w_j_per_m"]) > 0.0
    assert float(ls["affine_b_w_mj"]) > 0.0
    assert float(ls["affine_a_e_up_j_per_m"]) > 0.0
    assert float(ls["affine_b_e_up_mj"]) > 0.0
    assert float(ls["affine_a_k_loss_v_per_c_m"]) > 0.0

    # SHOULD 3: Passband integrated excess int Re Z_FIT - int Re Z_sinc^2 and per-period slopes over N in {6, 8, 12}
    n_arr_f = np.asarray(t3["n_periods_list"], dtype=float)
    int_fit_arr = np.asarray(ls["passband_integrated_re_z_fit_ohm_ghz"], dtype=float)
    int_s2_arr = np.asarray(ls["passband_integrated_re_z_cold_sinc2_ohm_ghz"], dtype=float)
    int_exc_arr = np.asarray(ls["passband_integrated_re_z_excess_ohm_ghz"], dtype=float)
    assert np.allclose(int_exc_arr, int_fit_arr - int_s2_arr, rtol=0.0, atol=1e-12)
    sl_fit, ic_fit = [float(x) for x in np.polyfit(n_arr_f, int_fit_arr, 1)]
    sl_s2, ic_s2 = [float(x) for x in np.polyfit(n_arr_f, int_s2_arr, 1)]
    assert abs(float(ls["passband_integrated_re_z_fit_slope_ohm_ghz_per_period"]) - sl_fit) < 1e-12
    assert abs(float(ls["passband_integrated_re_z_fit_intercept_ohm_ghz"]) - ic_fit) < 1e-12
    assert abs(float(ls["passband_integrated_re_z_cold_sinc2_slope_ohm_ghz_per_period"]) - sl_s2) < 1e-12
    assert abs(float(ls["passband_integrated_re_z_cold_sinc2_intercept_ohm_ghz"]) - ic_s2) < 1e-12
    assert (
        abs(
            float(ls["rel_diff_passband_integrated_re_z_slope_fit_vs_cold_sinc2"])
            - abs(sl_fit - sl_s2) / abs(sl_s2)
        )
        < 1e-12
    )

    # Derive pre-asymptotic short-N slope bounds from first principles:
    # t_p,eff(L) = (L + L_tap,eff)*(1/|v_g| + 1/u_0) + 1/Delta_f_bp with L_tap,eff in [0, 8 mm]
    # and Delta_f_bp = spec.passband_high_hz - spec.passband_low_hz over L in {12, 16, 24} mm.
    wb_cold = t2["wakis_finest_nf8_benchmarks"]
    vg_wak_m_s = float(wb_cold["ve_dft_over_c"]) * C0
    L_m_list = [float(x) * 1e-3 for x in ls["active_length_mm"]]
    slope_bounds = derive_pre_asymptotic_scaling_slopes(
        L_m_list,
        vg_wak_m_s,
        u0,
        taper_length_total_m=4.0 * spec.period_m,
        passband_bw_hz=spec.passband_high_hz - spec.passband_low_hz,
    )

    # Compare affine passband beam-work slope a_W against cold-test analytic loss per unit length
    # a_W,analytic = q^2 * k_L * Phi^2 / (1 - v_g / u_0) computed from JSON (MF4):
    # Tolerance derivation:
    # Because N = 6, 8, 12 are simulated at n_xy = 48 and n_xy = 64 (refinement ratio r_12 = (1/3)/(1/4) = 4/3),
    # we bound the transverse-mesh discretization error of the affine slope a_W directly from the two-level
    # Roache GCI of the slope itself with conservative safety factor Fs = 3 and formal order p = 1:
    #   GCI_2level(a_W) = 3 * |a_W(n_xy=64) - a_W(n_xy=48)| / (|a_W(n_xy=64)| * (r_12^1 - 1)).
    # Adding the cold-test unit-cell discretization uncertainty GCI_fine(k_L, cold) and the single-pass cavity
    # power reflection |Gamma_tap(f_s)|^2 gives:
    #   tol_a_w = GCI_2level(a_W) + GCI_fine(k_L, cold) + |Gamma_tap(f_s)|^2.
    a_w_fit = float(ls["affine_a_w_j_per_m"])
    a_w_fit_48 = float(t3["length_scaling_nxy48"]["affine_a_w_j_per_m"])
    a_w_analytic = analytic_backward_wave_loss_per_meter_j_per_m(
        spec.single_bunch_charge_c,
        float(wb_cold["k_loss_v_per_c_m"]),
        vg_wak_m_s,
        spec.bunch_length_rms_m,
        float(wb_cold["f_sync_ghz"]) * 1e9,
        u0,
        mutate_flip_vg_sign=False,
    )
    assert abs(a_w_analytic - float(ls["cold_analytic_loss_per_m_j_per_m"])) < 1e-9
    gci_a_w_two_level = 3.0 * abs(a_w_fit - a_w_fit_48) / (max(abs(a_w_fit), 1e-30) * ((r12**1.0) - 1.0))
    gci_kl_cold = float(t2["convergence_uniform"]["k_loss_v_per_c_m"]["gci_fine_rel"])
    tol_a_w = gci_a_w_two_level + gci_kl_cold + (gamma_tap**2)
    assert abs(gci_a_w_two_level - float(ls["gci_a_w_two_level_rel"])) < 1e-12
    assert abs(tol_a_w - float(ls["a_w_uncertainty_rel"])) < 1e-12

    up_frac_mtm = float(dc["mtm_backward_upstream_fraction"])
    slope_z = float(ls["slope_re_z_pk_vs_L_fit"])
    slope_fwhm = float(ls["slope_fwhm_vs_L_fit"])
    slope_tp = float(ls["slope_tp_vs_L_fit"])
    slope_p = float(ls["slope_p_up_vs_L_fit"])

    assert _check_wake_directionality_and_scaling(
        up_frac_mtm,
        slope_z,
        slope_fwhm,
        slope_tp,
        slope_p,
        slope_bounds,
        gamma_tap,
        a_w_fit,
        a_w_analytic,
        tol_a_w,
        gamma_pb_max,
    )

    # Analytical sinc^2 impedance peak ratio between L = 24 mm and L = 12 mm is (24/12)^2 = 4.0 (slope = 2.0)
    z_th_12 = float(
        analytic_sws_traveling_wave_impedance_ohm(
            np.array([11.42e9]), 11.42e9, 3.767e14, 0.012, u0, -0.158 * C0
        )[0]
    )
    z_th_24 = float(
        analytic_sws_traveling_wave_impedance_ohm(
            np.array([11.42e9]), 11.42e9, 3.767e14, 0.024, u0, -0.158 * C0
        )[0]
    )
    slope_th = float(np.log(z_th_24 / z_th_12) / np.log(0.024 / 0.012))
    assert abs(slope_th - 2.0) < 1e-12

    # Mutation 1: linear-in-L impedance mutation (mutate_linear_length=True) gives slope = 1.0 (< slope_z_min) and must fail
    z_mut_12 = float(
        analytic_sws_traveling_wave_impedance_ohm(
            np.array([11.42e9]), 11.42e9, 3.767e14, 0.012, u0, -0.158 * C0, mutate_linear_length=True
        )[0]
    )
    z_mut_24 = float(
        analytic_sws_traveling_wave_impedance_ohm(
            np.array([11.42e9]), 11.42e9, 3.767e14, 0.024, u0, -0.158 * C0, mutate_linear_length=True
        )[0]
    )
    slope_mut_z = float(np.log(z_mut_24 / z_mut_12) / np.log(0.024 / 0.012))
    assert not _check_wake_directionality_and_scaling(
        up_frac_mtm,
        slope_mut_z,
        slope_fwhm,
        slope_tp,
        slope_p,
        slope_bounds,
        gamma_tap,
        a_w_fit,
        a_w_analytic,
        tol_a_w,
        gamma_pb_max,
    )

    # Mutation 2: forward-wave disk-loaded control has upstream fraction < 1 - 2*|Gamma_max|^2 and must fail
    up_frac_fwd_control = 1.0 - float(dc["control_forward_downstream_fraction"])
    assert not _check_wake_directionality_and_scaling(
        up_frac_fwd_control,
        slope_z,
        slope_fwhm,
        slope_tp,
        slope_p,
        slope_bounds,
        gamma_tap,
        a_w_fit,
        a_w_analytic,
        tol_a_w,
        gamma_pb_max,
    )

    # Mutation 3 (M3): flipping the slippage factor to (1 + v_g/u_0) inside analytic_backward_wave_loss_per_meter_j_per_m
    # increases the predicted analytic loss slope by (1 + |v_g|/u_0)/(1 - |v_g|/u_0) ~ 1.38 and must fail tol_a_w
    a_w_mut_flip_vg = analytic_backward_wave_loss_per_meter_j_per_m(
        spec.single_bunch_charge_c,
        float(wb_cold["k_loss_v_per_c_m"]),
        vg_wak_m_s,
        spec.bunch_length_rms_m,
        float(wb_cold["f_sync_ghz"]) * 1e9,
        u0,
        mutate_flip_vg_sign=True,
    )
    assert not _check_wake_directionality_and_scaling(
        up_frac_mtm,
        slope_z,
        slope_fwhm,
        slope_tp,
        slope_p,
        slope_bounds,
        gamma_tap,
        a_w_fit,
        a_w_mut_flip_vg,
        tol_a_w,
        gamma_pb_max,
    )


def test_two_bunch_fit_linear_superposition_and_phase_scaling_with_mutations() -> None:
    """Verify 3D FIT two-bunch superposition to machine round-off and passband wake interference (with mutations)."""
    # Live quick 3D conformal FIT two-bunch superposition check (including live +0.5*dt bunch-2 shift mutation)
    sup_live = verify_two_bunch_fit_superposition(n_periods=4, n_steps=90, n_xy=16, dz_mm=0.5)
    assert sup_live["rel_l2_error_E_3d"] < 1e-13
    assert sup_live["rel_l2_error_H_3d"] < 1e-13

    t1 = json.loads((DATA_DIR / "task_01_geometry_and_port_reflection.json").read_text(encoding="utf-8"))
    t4 = json.loads((DATA_DIR / "task_04_two_bunch_superposition.json").read_text(encoding="utf-8"))
    sup = t4["fit_3d_superposition_verification"]
    eq_b = t4["equal_bunch_superposition"]
    gamma_tap = float(t1["port_reflection"]["gamma_tapered_at_sync"])
    ds_in_mm = float(eq_b["in_phase_spacing_mm"])
    ds_out_mm = float(eq_b["out_of_phase_spacing_mm"])
    s_ov_max_mm = 90.0

    assert t4["n8_can_overlap_at_experiment_spacing"] is False
    assert float(t4["n8_pulse_duration_20pct_ps"]) < float(t4["linac_period_ps"])
    # M5 + MF7: exact 1/1.3 GHz spacing is m = T_b * f_s = 11.42 / 1.3 ~ 8.7846 cycles (282.5 deg mod 360 deg),
    # while 9 / f_s = 0.788 ns corresponds to a 1.27 GHz bunch train (Lu thesis lines 2340-2341, 2365).
    expected_m_1p3 = 11.42 / 1.3
    expected_phase_mod = (expected_m_1p3 % 1.0) * 360.0
    assert abs(float(t4["harmonic_number_m_at_1p3ghz"]) - expected_m_1p3) < 1e-6
    assert abs(float(t4["phase_mod_360_deg_at_1p3ghz"]) - expected_phase_mod) < 1e-6
    assert abs(float(t4["ninth_harmonic_bunch_spacing_ns"]) - (9.0 / 11.42)) < 1e-6
    assert abs(float(t4["ninth_harmonic_bunch_train_freq_ghz"]) - (11.42 / 9.0)) < 1e-6
    assert "m = 9 cycles" not in str(t4["bunch_spacing_limitation_note"])
    assert "UV" not in str(t4["bunch_spacing_limitation_note"])
    assert "9th harmonic" in str(t4["bunch_spacing_limitation_note"])

    assert _check_two_bunch_superposition_and_phase(
        float(sup["rel_l2_error_E_3d"]),
        float(eq_b["in_phase_power_ratio_over_single"]),
        float(eq_b["out_of_phase_power_ratio_over_single"]),
        ds_in_mm,
        ds_out_mm,
        s_ov_max_mm,
        gamma_tap,
    )

    # Mutation 1: perturbing bunch 2 injection time by +0.5*dt in live 3D FIT breaks superposition (~2e-2 >> 1e-12)
    assert not _check_two_bunch_superposition_and_phase(
        float(sup_live["mutated_half_dt_shift_rel_l2_error_E"]),
        float(eq_b["in_phase_power_ratio_over_single"]),
        float(eq_b["out_of_phase_power_ratio_over_single"]),
        ds_in_mm,
        ds_out_mm,
        s_ov_max_mm,
        gamma_tap,
    )

    # Mutation 2: linear amplitude scaling (2*|cos(Delta_psi/2)| -> 2.0 in phase instead of quadratic Q^2 scaling) must fail
    z4 = np.load(DATA_DIR / "task_04_two_bunch_traces.npz")
    mut_in_phase_ratio = float(z4["power_ratio_mutated_linear"][0])
    assert not _check_two_bunch_superposition_and_phase(
        float(sup["rel_l2_error_E_3d"]),
        mut_in_phase_ratio,
        float(eq_b["out_of_phase_power_ratio_over_single"]),
        ds_in_mm,
        ds_out_mm,
        s_ov_max_mm,
        gamma_tap,
    )


def test_analytic_backward_wave_power_and_energy_balance_with_mutations() -> None:
    """Verify backward-wave analytical power Eq. (2), FIT pulse duration vs Eq. (1) across N=6/8/12, and control-volume energy balance."""
    spec = WagonWheelMTMSpec()
    _, _, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    t1 = json.loads((DATA_DIR / "task_01_geometry_and_port_reflection.json").read_text(encoding="utf-8"))
    t3 = json.loads((DATA_DIR / "task_03_wakefield_and_directionality.json").read_text(encoding="utf-8"))
    t5 = json.loads((DATA_DIR / "task_05_analytic_power_and_energy_balance.json").read_text(encoding="utf-8"))
    wa = t5["wakis_cold_test_analytic"]
    wc = t5["wakefield_vs_analytic_comparison"]
    eb = t5["energy_balance_N8"]
    eb_all = t5["energy_balance_all_N"]

    # Verify PEC ohmic justification is computed in code (2*Q_0/omega_s >> t_p)
    pj = t1["pec_ohmic_justification"]
    assert float(pj["ohmic_field_decay_time_2q_over_omega_ns"]) > 25.0
    assert float(pj["ratio_field_decay_over_pulse_n40"]) > 10.0

    p_wak_mw = float(wa["single_bunch_45nc_power_mw"])
    p_fit_mw = float(wc["wakis_3d_fit_power_N8_mw"])
    vg_wak_m_s = float(wa["vg_over_c"]) * C0

    # Live 3D FIT control-volume Poynting conservation verification (Items 1, 2, 3, 5, 6; Round 5 M1, S1, S2, S3, N2):
    # Verify that on the unified control volume [z_up, z_dn]:
    # (a) Unfiltered discrete FIT Poynting theorem closes to < 1e-3 (derived from O((omega_s*dt)^2), S3);
    # (b) Causally bandpass-filtered Poynting theorem |E_rad + U(T_end) - W_beam| / W_beam < 1e-3;
    # (c) Smooth-pipe reference run's own unfiltered control-volume closure < 1e-3 and U_ref,bp(T_end)/W_beam,bp < 1e-3 (S1, N2);
    # (d) 20% pulse envelope ends more than 1 RF period before T_end and U_bp(T_end)/W_beam,bp < 0.02 (M1);
    # (e) Re Z_||(f) is not clipped at zero (negative sidelobes preserved, Item 5);
    # (f) fmax >= 100 GHz so |F(fmax)|^2 < 1e-3 while |F(25 GHz)|^2 > 0.5 (Item 6).
    r_live = run_finite_mtm_wakefield(n_periods=4, n_xy=28, dz_mm=0.50, quick=False)
    u_st_live_mj = float(r_live["u_stored_final_fit_mj"])
    w_loss_live_mj = float(r_live["w_loss_passband_mj"])
    e_rad_live_mj = float(r_live["e_rad_total_mj"])
    assert float(np.min(r_live["re_z_ohm"])) < 0.0
    assert float(r_live["fmax_impedance_ghz"]) >= 100.0
    assert float(r_live["form_factor_sq_at_fmax"]) < 1e-3
    assert float(r_live["form_factor_sq_at_25ghz"]) > 0.5
    assert float(r_live["w_loss_wake_fullband_active_mj"]) > float(r_live["w_loss_wake_passband_active_mj"]) > 0.0
    assert 0.0 < float(r_live["out_of_band_wake_loss_fraction_active"]) < 1.0
    # For the MTM run, the dominant synchronous mode f_s = 11.42 GHz has second-order leapfrog time-averaging
    # error O((omega_s * dt)^2) < 1e-3 (S3). For the smooth-pipe reference run (radius r_b = 3.0 mm),
    # all propagating modes lie above the smooth-pipe TM01 cutoff f_c,pipe = 38.25 GHz, scaling the
    # unfiltered leapfrog time-averaging bound by (f_c,pipe / f_s)^2 = (38.25 / 11.42)^2 ≈ 11.2.
    ref_closure_rtol = ((38.25 / 11.42) ** 2) * 1e-3
    assert float(r_live["rel_error_unfiltered_cv_closure"]) < 1e-3
    assert float(r_live["rel_error_total_accounted_vs_bunch_loss"]) < 1e-3
    assert float(r_live["rel_error_ref_unfiltered_cv_closure"]) < ref_closure_rtol
    assert (
        abs(
            (float(r_live["w_beam_unfiltered_cv_mj"]) - float(r_live["w_beam_scat_fullband_cv_mj"]))
            - float(r_live["w_ref_unfiltered_cv_mj"])
        )
        < 1e-6
    )
    assert abs(float(r_live["u_stored_ref_passband_cv_mj"])) / max(w_loss_live_mj, 1e-30) < 1e-3
    assert 0.0 < float(r_live["u_stored_over_w_beam_passband"]) < 0.02
    assert r_live["pulse_clips_window"] is False
    assert (float(r_live["t_end_window_ns"]) - float(r_live["pulse_end_20pct_ns"])) > float(r_live["rf_period_ns"])

    # Non-tautological pulse-length and non-clipping window check across N in {6, 8, 12} (S2):
    # Compare 3D FIT measured 20%-envelope pulse duration against Eq. (1) evaluated over the
    # active + two 2-cell taper length (N + 4)*p, within the numerically measured 10-90% step rise time
    # of the 3rd-order causal Butterworth field filter [9.8, 13.0] GHz from h_bp plus half an RF cycle:
    #   dt_filter_tol_ns = tau_rise,10-90(h_bp) + 0.5 / f_s.
    n_list = [6, 8, 12]
    n_taper_cells_total = 4  # two 2-cell adiabatic tapers (build_adiabatic_coax_mtm_solids default n_taper_cells=2)
    f_sync_wak_ghz = float(wa["f_sync_ghz"])
    dt_n8_s = float(t3["runs_mtm"]["8"]["dt_fs"]) * 1e-15
    tau_rise_10_90_ns = measure_filter_step_rise_time_ns(
        dt_s=dt_n8_s,
        f_lo_hz=9.8e9,
        f_hi_hz=13.0e9,
        order=3,
    )
    dt_filter_tol_ns = float(tau_rise_10_90_ns + 0.5 / f_sync_wak_ghz)
    tp_fit_all_n_ns = [float(t3["runs_mtm"][str(n)]["pulse_duration_20pct_ns"]) for n in n_list]
    pulse_end_all_n_ns = [float(t3["runs_mtm"][str(n)]["pulse_end_20pct_ns"]) for n in n_list]
    t_end_window_all_n_ns = [float(t3["runs_mtm"][str(n)]["t_end_window_ns"]) for n in n_list]
    rf_period_all_n_ns = [float(t3["runs_mtm"][str(n)]["rf_period_ns"]) for n in n_list]
    u_stored_over_w_all_n = [float(eb_all[str(n)]["u_stored_over_w_beam_passband"]) for n in n_list]
    for n in n_list:
        rn_check = t3["runs_mtm"][str(n)]
        ebn_check = eb_all[str(n)]
        assert rn_check["pulse_clips_window"] is False
        assert (float(rn_check["t_end_window_ns"]) - float(rn_check["pulse_end_20pct_ns"])) > float(
            rn_check["rf_period_ns"]
        )
        assert 0.0 < float(ebn_check["u_stored_over_w_beam_passband"]) < 0.02
        # S1 + N2: verify reference smooth-pipe run closure and negligible passband stored energy at every N
        assert float(ebn_check["rel_error_ref_unfiltered_cv_closure"]) < ref_closure_rtol
        assert (
            abs(
                (float(ebn_check["w_beam_unfiltered_cv_mj"]) - float(ebn_check["w_beam_scat_fullband_cv_mj"]))
                - float(ebn_check["w_ref_unfiltered_cv_mj"])
            )
            < 1e-6
        )
        assert (
            abs(float(ebn_check["u_stored_ref_passband_cv_mj"]))
            / max(float(ebn_check["bunch_energy_loss_wake_passband_mj"]), 1e-30)
            < 1e-3
        )
    tp_eq1_tapered_all_n_ns = [
        analytic_backward_wave_pulse_length_s((n + n_taper_cells_total) * spec.period_m, vg_wak_m_s, u0) * 1e9
        for n in n_list
    ]

    z1 = np.load(DATA_DIR / "task_01_port_reflection_spectra.npz")
    gamma_tap = float(t1["port_reflection"]["gamma_tapered_at_sync"])
    gamma_pb_max = float(np.max(np.asarray(z1["gamma_tapered_coax"], dtype=float)))
    power_refl_bound = float(reflection_induced_energy_error_bound(gamma_tap))
    dir_refl_bound = float(reflection_induced_energy_error_bound(gamma_pb_max))
    assert abs(power_refl_bound - (gamma_tap**2)) < 1e-12
    # Live mutation check on reflection_induced_energy_error_bound:
    # Replacing |Gamma|^2 with linear |Gamma| inflates the single-reflection energy bound by 1/|Gamma| and must fail
    energy_refl_bound_mut = float(reflection_induced_energy_error_bound(gamma_tap, mutate_use_linear_gamma=True))
    assert abs(energy_refl_bound_mut - (gamma_tap**2)) > 0.10

    e_up_n8 = float(eb["upstream_radiated_poynting_mj"])
    e_rad_n8 = float(eb["total_radiated_poynting_mj"])
    up_rad_frac_n8 = e_up_n8 / e_rad_n8

    # Verify control-volume Poynting closure |E_rad + U - W_beam| / W_beam < 1e-3 across all N in {6, 8, 12} (S3):
    rel_err_total_all_n = [
        float(eb_all[str(n)]["rel_error_total_accounted_vs_bunch_loss"]) for n in n_list
    ]
    rel_err_unfiltered_all_n = [
        float(eb_all[str(n)]["rel_error_unfiltered_cv_closure"]) for n in n_list
    ]
    assert len(rel_err_total_all_n) == 3
    for err_bp, err_unf in zip(rel_err_total_all_n, rel_err_unfiltered_all_n):
        assert err_bp < 1e-3
        assert err_unf < 1e-3

    assert _check_analytic_power_and_energy_balance(
        p_wak_mw,
        p_fit_mw,
        tp_fit_all_n_ns,
        tp_eq1_tapered_all_n_ns,
        pulse_end_all_n_ns,
        t_end_window_all_n_ns,
        rf_period_all_n_ns,
        dt_filter_tol_ns,
        up_rad_frac_n8,
        power_refl_bound,
        dir_refl_bound,
        rel_err_total_all_n,
        u_stored_over_w_all_n,
        u_st_live_mj,
        w_loss_live_mj,
        e_rad_live_mj,
    )

    # Mutation 1: dropping the backward-wave slippage factor (1 - v_g/u_0)^-2 overpredicts power and fails
    p_mut_noslip_mw = (
        analytic_backward_wave_power_w(
            spec.single_bunch_charge_c,
            float(wa["k_loss_v_per_c_m"]),
            vg_wak_m_s,
            spec.bunch_length_rms_m,
            float(wa["f_sync_ghz"]) * 1e9,
            u0,
            mutate_drop_slippage_factor=True,
        )
        * 1e-6
    )
    # SH4 consistency check vs paper's measured "up to 25 MW" at N = 40 (PRL line 50):
    # |P96 - 25| / 25 <= U_P + 2 * |Gamma|^2, and both mutated powers (no slippage, flipped v_g) lie outside that band.
    p_fit_nxy96_mw = float(wc["wakis_3d_fit_power_N8_nxy96_mw"])
    u_p_fine_rel = float(wc["p_up_u_fine_nxy96_rel"])
    band_sh4 = u_p_fine_rel + 2.0 * power_refl_bound
    p_mut_flip_vg_mw = (
        analytic_backward_wave_power_w(
            spec.single_bunch_charge_c,
            float(wa["k_loss_v_per_c_m"]),
            vg_wak_m_s,
            spec.bunch_length_rms_m,
            float(wa["f_sync_ghz"]) * 1e9,
            u0,
            mutate_flip_vg_sign=True,
        )
        * 1e-6
    )
    assert abs(p_fit_nxy96_mw - 25.0) / 25.0 <= band_sh4
    assert abs(p_mut_noslip_mw - 25.0) / 25.0 > band_sh4
    assert abs(p_mut_flip_vg_mw - 25.0) / 25.0 > band_sh4

    assert not _check_analytic_power_and_energy_balance(
        p_mut_noslip_mw,
        p_fit_mw,
        tp_fit_all_n_ns,
        tp_eq1_tapered_all_n_ns,
        pulse_end_all_n_ns,
        t_end_window_all_n_ns,
        rf_period_all_n_ns,
        dt_filter_tol_ns,
        up_rad_frac_n8,
        power_refl_bound,
        dir_refl_bound,
        rel_err_total_all_n,
        u_stored_over_w_all_n,
        u_st_live_mj,
        w_loss_live_mj,
        e_rad_live_mj,
    )

    # Mutation 2 (S2): flipping the sign of v_g -> +|v_g| (forward-wave sign error in Eq. 1) fails pulse length check at EVERY N
    tp_mut_flip_all_n_ns = [
        analytic_backward_wave_pulse_length_s(
            (n + n_taper_cells_total) * spec.period_m,
            vg_wak_m_s,
            u0,
            mutate_flip_vg_sign=True,
        )
        * 1e9
        for n in n_list
    ]
    for tp_fit_n, tp_mut_n in zip(tp_fit_all_n_ns, tp_mut_flip_all_n_ns):
        assert abs(tp_fit_n - tp_mut_n) > dt_filter_tol_ns
    assert not _check_analytic_power_and_energy_balance(
        p_wak_mw,
        p_fit_mw,
        tp_fit_all_n_ns,
        tp_mut_flip_all_n_ns,
        pulse_end_all_n_ns,
        t_end_window_all_n_ns,
        rf_period_all_n_ns,
        dt_filter_tol_ns,
        up_rad_frac_n8,
        power_refl_bound,
        dir_refl_bound,
        rel_err_total_all_n,
        u_stored_over_w_all_n,
        u_st_live_mj,
        w_loss_live_mj,
        e_rad_live_mj,
    )

    # Mutation 3: omitting the upstream (-z) Poynting flux leaves only downstream leakage (< 1 - 2*|Gamma_pb|^2) and must fail
    e_dn_n8 = float(eb["downstream_radiated_poynting_mj"])
    up_rad_frac_mut_no_up = e_dn_n8 / e_rad_n8
    assert not _check_analytic_power_and_energy_balance(
        p_wak_mw,
        p_fit_mw,
        tp_fit_all_n_ns,
        tp_eq1_tapered_all_n_ns,
        pulse_end_all_n_ns,
        t_end_window_all_n_ns,
        rf_period_all_n_ns,
        dt_filter_tol_ns,
        up_rad_frac_mut_no_up,
        power_refl_bound,
        dir_refl_bound,
        rel_err_total_all_n,
        u_stored_over_w_all_n,
        u_st_live_mj,
        w_loss_live_mj,
        e_rad_live_mj,
    )

    # Mutation 4 (Item 2): shrinking the beam-work z-range back to the active cells [z_act_start, z_act_end]
    # while keeping the port flux planes at [z_up, z_dn] violates control-volume closure (> 2%) and must fail
    r_mut_shrunk = run_finite_mtm_wakefield(
        n_periods=4, n_xy=28, dz_mm=0.50, quick=False, mutate_shrink_beam_work_to_active=True
    )
    w_loss_mut_shrunk_mj = float(r_mut_shrunk["w_loss_passband_mj"])
    assert float(r_mut_shrunk["rel_error_total_accounted_vs_bunch_loss"]) >= 0.02
    rel_err_shrunk_all_n = [
        float(eb_all[str(n)]["mutated_shrunk_active_beam_work_rel_error"]) for n in n_list
    ]
    assert not _check_analytic_power_and_energy_balance(
        p_wak_mw,
        p_fit_mw,
        tp_fit_all_n_ns,
        tp_eq1_tapered_all_n_ns,
        pulse_end_all_n_ns,
        t_end_window_all_n_ns,
        rf_period_all_n_ns,
        dt_filter_tol_ns,
        up_rad_frac_n8,
        power_refl_bound,
        dir_refl_bound,
        rel_err_shrunk_all_n,
        u_stored_over_w_all_n,
        u_st_live_mj,
        w_loss_mut_shrunk_mj,
        e_rad_live_mj,
    )

    # Mutation 5 (M1): truncating the wake window (mutate_truncate_wake_window=True) so the 20%
    # envelope end clips within one RF period of T_end AND residual passband stored energy
    # U_bp(T_end) / W_beam,bp >= 0.02 violates the M1 drainage bound and must fail
    r_mut_trunc = run_finite_mtm_wakefield(
        n_periods=4, n_xy=28, dz_mm=0.50, quick=False, mutate_truncate_wake_window=True
    )
    assert r_mut_trunc["pulse_clips_window"] is True
    assert float(r_mut_trunc["u_stored_over_w_beam_passband"]) >= 0.02
    assert (float(r_mut_trunc["t_end_window_ns"]) - float(r_mut_trunc["pulse_end_20pct_ns"])) <= float(
        r_mut_trunc["rf_period_ns"]
    )
    assert not _check_analytic_power_and_energy_balance(
        p_wak_mw,
        p_fit_mw,
        [float(r_mut_trunc["pulse_duration_20pct_ns"])] * 3,
        tp_eq1_tapered_all_n_ns,
        [float(r_mut_trunc["pulse_end_20pct_ns"])] * 3,
        [float(r_mut_trunc["t_end_window_ns"])] * 3,
        [float(r_mut_trunc["rf_period_ns"])] * 3,
        dt_filter_tol_ns,
        up_rad_frac_n8,
        power_refl_bound,
        dir_refl_bound,
        rel_err_total_all_n,
        [float(r_mut_trunc["u_stored_over_w_beam_passband"])] * 3,
        float(r_mut_trunc["u_stored_final_fit_mj"]),
        float(r_mut_trunc["w_loss_passband_mj"]),
        float(r_mut_trunc["e_rad_total_mj"]),
    )

    # Mutation 6: unfiltered broadband stored energy (mutate_unfiltered_stored_energy=True)
    # violates passband control-volume closure (> 2%) and must fail
    r_mut_unfilt = run_finite_mtm_wakefield(
        n_periods=4, n_xy=28, dz_mm=0.50, quick=False, mutate_unfiltered_stored_energy=True
    )
    u_st_mut_unfilt_mj = float(r_mut_unfilt["u_stored_final_fit_mj"])
    assert float(r_mut_unfilt["rel_error_total_accounted_vs_bunch_loss"]) >= 0.02
    assert not _check_analytic_power_and_energy_balance(
        p_wak_mw,
        p_fit_mw,
        tp_fit_all_n_ns,
        tp_eq1_tapered_all_n_ns,
        pulse_end_all_n_ns,
        t_end_window_all_n_ns,
        rf_period_all_n_ns,
        dt_filter_tol_ns,
        up_rad_frac_n8,
        power_refl_bound,
        dir_refl_bound,
        rel_err_total_all_n,
        u_stored_over_w_all_n,
        u_st_mut_unfilt_mj,
        w_loss_live_mj,
        e_rad_live_mj,
    )


def test_quantitative_summary_table_and_notebook_sync_match_json_with_mutations() -> None:
    """Verify rendered Quantitative Summary table matches summary_metrics.json, figures exist, and bare 'PIC' is absent."""
    py_path = TUTORIAL_DIR / "mtm_wakefield_wagon_wheel.py"
    ipynb_path = TUTORIAL_DIR / "mtm_wakefield_wagon_wheel.ipynb"
    sm_path = DATA_DIR / "summary_metrics.json"

    assert py_path.is_file()
    assert ipynb_path.is_file()
    assert sm_path.is_file()

    py_text = py_path.read_text(encoding="utf-8")
    ipynb_text = ipynb_path.read_text(encoding="utf-8")
    sm_dict = json.loads(sm_path.read_text(encoding="utf-8"))
    t2 = json.loads((DATA_DIR / "task_02_cold_test_and_convergence.json").read_text(encoding="utf-8"))
    t3 = json.loads((DATA_DIR / "task_03_wakefield_and_directionality.json").read_text(encoding="utf-8"))
    t4 = json.loads((DATA_DIR / "task_04_two_bunch_superposition.json").read_text(encoding="utf-8"))
    t5 = json.loads((DATA_DIR / "task_05_analytic_power_and_energy_balance.json").read_text(encoding="utf-8"))

    rendered_table_md = render_quantitative_summary_markdown(sm_dict, t2, t3, t4, t5)

    # 1. Verify every row of rendered Quantitative Summary table matches summary_metrics.json
    _verify_summary_table_against_json(rendered_table_md, sm_dict)

    # Mutation check on the table verifier: corrupting a metric in sm_dict by +10% must raise AssertionError
    sm_corrupted = dict(sm_dict)
    sm_corrupted["single_bunch_45nc_power_fit_mw"] = 1.10 * float(sm_dict["single_bunch_45nc_power_fit_mw"])
    with pytest.raises(AssertionError):
        _verify_summary_table_against_json(rendered_table_md, sm_corrupted)

    # 2. Verify all required setup and result figures (including animated GIF) exist and are non-empty
    required_figures = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00_setup_table.md",
        "01_wagon_wheel_geometry_and_port_reflection.png",
        "02_bloch_dispersion_and_eigenmode_fields.png",
        "03_cold_test_mesh_convergence.png",
        "04_wake_potential_and_impedance_scaling.png",
        "05_reversed_cherenkov_poynting_flux.png",
        "06_reversed_cherenkov_wave_packet.png",
        "07_reversed_cherenkov_wave_packet.gif",
        "08_two_bunch_coherent_superposition.png",
        "09_analytic_power_and_energy_balance.png",
    ]
    for fname in required_figures:
        fpath = FIGURES_DIR / fname
        assert fpath.is_file() and fpath.stat().st_size > 500, f"Missing or empty figure: {fpath}"

    # 3. Hard rule: never write the bare abbreviation "PIC" in any tutorial file
    bare_pic_re = re.compile(r"\bPIC\b")
    for check_file in (
        py_path,
        ipynb_path,
        TUTORIAL_DIR / "_helpers.py",
        TUTORIAL_DIR / "generate_data.py",
        TUTORIAL_DIR / "README.md",
    ):
        txt = check_file.read_text(encoding="utf-8")
        assert bare_pic_re.search(txt) is None, f"Found forbidden bare abbreviation 'PIC' in {check_file}"

    # 4. Round-5 text and reference integrity checks (M1, M4, M6, M7, S1, S3, S6, S7, S9):
    readme_text = (TUTORIAL_DIR / "README.md").read_text(encoding="utf-8")
    helpers_text = (TUTORIAL_DIR / "_helpers.py").read_text(encoding="utf-8")
    assert "drains well before" not in py_text
    assert "drains completely" not in helpers_text
    assert "over 90%" not in py_text
    assert "> 90%" not in rendered_table_md
    assert "> 80%" not in rendered_table_md
    assert "frequencies above 35 GHz are exponentially suppressed" not in py_text
    assert "150 nC" not in py_text
    assert "150\\;\\mathrm{MV/m}" not in py_text
    assert "linear-theory companion" not in py_text
    assert "linear-theory companion" not in readme_text
    assert (
        "Modeling of the interaction of a volumetric metallic metamaterial structure with a relativistic electron beam"
        in py_text
    )
    assert "Wangler" in py_text
    assert "github/eshipton/Wakis" in py_text
    assert "github/eshipton/Wakis" in readme_text
    assert "github/ImanolZ/wakis" not in py_text
    assert "github/ImanolZ/wakis" not in readme_text
    assert "outside the `9.5–13.5 GHz`" in py_text
    assert "higher-order waveguide modes" not in py_text
    assert "memguard" not in py_text
    assert "memguard" not in helpers_text
    assert ".get(" not in py_text
    assert "including a reference-run artifact" in py_text
    assert "k_L^2" not in py_text
    assert "3.95e14" not in helpers_text
    assert "-0.1595 * C0" not in helpers_text
    assert "UV" not in py_text
    assert "UV" not in helpers_text
    assert "injection/exit transient across the CPML" not in py_text
    assert "injection/exit transient across the CPML" not in helpers_text
    assert "unexplained" in py_text
    # M1(d): Every N's coarse (n_xy = 48) passband control-volume closure < 1e-3 (truncation + numerical quadrature bound)
    # and late-time passband stored energy U_bp / W_bp < 0.02 (2% late-time drainage bound), confirming that the
    # late-time unfiltered energy growth on n_xy = 48 (where unfiltered closure > 1e-3 for all N = 6, 8, 12) leaves
    # the 9.5–13.5 GHz wake passband clean:
    for n_key in ("6", "8", "12"):
        assert float(t5["energy_balance_all_N"][n_key]["nxy48_rel_error_unfiltered_cv_closure"]) > 1e-3
        assert float(t3["runs_mtm_nxy48"][n_key]["rel_error_total_accounted_vs_bunch_loss"]) < 1e-3
        assert 0.0 < float(t3["runs_mtm_nxy48"][n_key]["u_stored_over_w_beam_passband"]) < 0.02
    assert float(t3["length_scaling"]["n8_re_z_pk_over_cold_sinc2_ratio"]) > 1.0
    assert "attributed to Fabry" not in py_text
    assert "Standing-wave modulated" not in helpers_text
    assert "oscillatory, $|R| > 1$ (amplitude not decreasing)" in py_text
    for k_gt1 in ("w_beam_passband_cv_mj", "e_up_coax_mj", "port_gamma_tapered_at_sync"):
        assert "oscillatory, |R| > 1 (amplitude not decreasing)" in format_gci_line(
            t3["n8_transverse_mesh_convergence"]["gci"][k_gt1]
        )
    for k_lt1 in ("p_up_cycle_avg_max_mw", "e_up_mj"):
        g_line = format_gci_line(t3["n8_transverse_mesh_convergence"]["gci"][k_lt1])
        assert "class=oscillatory," in g_line
        assert "|R| > 1" not in g_line
    assert "only falls below $10^{-3}$ above $100" not in py_text
    assert "the annular TEM port transmission across all three mesh levels" not in py_text
    assert abs(float(t5["wakis_cold_test_analytic"]["f_form_factor_sq_1e3_ghz"]) - 104.66) < 0.5
    assert "gamma_tap: float = 0.22" not in helpers_text
    assert "vg_min_over_c: float = 0.052" not in helpers_text
    assert "compensated" not in py_text
    assert "compensates" not in py_text
    assert "PRL line 52" not in py_text
    assert "PRL lines 52" not in py_text
    assert "51–52" not in py_text
    # Round-8 Notes 6, 7, 8, 9 checks:
    assert "3 \\times 10^{-5}" not in py_text
    assert "within 0.2%" not in py_text
    assert "within 0.2 %" not in py_text
    assert "% from unity" in py_text
    assert "withdraw" not in py_text
    assert "cannot account for the" in py_text
    assert "dz_mm = 1.0 / 4.0" not in helpers_text
    assert "f_s_paper_ghz = 11.42" not in helpers_text
    assert "f_sync_hz = 11.42e9" not in helpers_text
    assert 'disc_b["cells_across_feature"] or 4.0' not in helpers_text
    assert 'disc_b["min_feature_m"] or 1.0e-3' not in helpers_text


def _check_nxy48_length_scaling_spectra_not_reused(
    ls_64: dict[str, object],
    ls_48: dict[str, object],
) -> bool:
    """Return True if n_xy=64 has valid integrated spectra while n_xy=48 has null/absent spectrum-integral keys (not reusing n_xy=64)."""
    int_64 = ls_64.get("passband_integrated_re_z_fit_ohm_ghz")
    int_48 = ls_48.get("passband_integrated_re_z_fit_ohm_ghz")
    if not isinstance(int_64, list) or len(int_64) != 3 or not all(float(x) > 0.0 for x in int_64):
        return False
    if int_48 == int_64 or int_48 is not None:
        return False
    null_keys_48 = (
        "passband_integrated_re_z_fit_ohm_ghz",
        "passband_integrated_re_z_cold_sinc2_ohm_ghz",
        "passband_integrated_re_z_excess_ohm_ghz",
        "passband_integrated_re_z_fit_slope_ohm_ghz_per_period",
        "passband_integrated_re_z_fit_intercept_ohm_ghz",
        "passband_integrated_re_z_cold_sinc2_slope_ohm_ghz_per_period",
        "passband_integrated_re_z_cold_sinc2_intercept_ohm_ghz",
        "rel_diff_passband_integrated_re_z_slope_fit_vs_cold_sinc2",
        "passband_integrated_re_z_fit_over_cold_sinc2_ratio",
        "n8_passband_integrated_re_z_fit_over_cold_sinc2_ratio",
        "effective_width_fit_ghz",
        "effective_width_cold_sinc2_ghz",
        "effective_width_fit_over_cold_sinc2_ratio",
        "n8_effective_width_fit_over_cold_sinc2_ratio",
    )
    for k_null in null_keys_48:
        if ls_48.get(k_null) is not None:
            return False
        if ls_64.get(k_null) is None:
            return False
    reason_48 = ls_48.get("passband_integrated_re_z_missing_reason")
    if not isinstance(reason_48, str) or len(reason_48) < 15:
        return False
    if ls_64.get("passband_integrated_re_z_missing_reason") is not None:
        return False
    return True


def _check_json_key_sets_match(
    disk_dicts: dict[str, dict[str, object]],
    gen_dicts: dict[str, dict[str, object]],
) -> bool:
    """Return True if every data/*.json top-level and section key set equals the committed generator output."""
    if set(disk_dicts.keys()) != set(gen_dicts.keys()):
        return False

    for fname, d_obj in disk_dicts.items():
        g_obj = gen_dicts[fname]
        if set(d_obj.keys()) != set(g_obj.keys()):
            return False
        for k, d_sub in d_obj.items():
            g_sub = g_obj.get(k)
            if isinstance(d_sub, dict) and isinstance(g_sub, dict):
                if set(d_sub.keys()) != set(g_sub.keys()):
                    return False
    return True


def test_json_schema_matches_committed_generators_with_mutations(tmp_path: Path) -> None:
    """Verify the key set of every data/*.json equals the key set produced by the committed generator functions."""
    gen_dicts = build_dummy_all_json_dicts(tmp_path / "clean")
    disk_dicts = {
        p.name: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(DATA_DIR.glob("*.json"))
    }
    assert _check_json_key_sets_match(disk_dicts, gen_dicts)

    # Explicitly verify the 5 previously writer-less keys (Item 4) and Step 4 keys (Items 12, 13, 14) are produced
    t1_gen = gen_dicts["task_01_geometry_and_port_reflection.json"]
    t2_gen = gen_dicts["task_02_cold_test_and_convergence.json"]
    t3_gen = gen_dicts["task_03_wakefield_and_directionality.json"]
    t4_gen = gen_dicts["task_04_two_bunch_superposition.json"]
    assert t1_gen["port_reflection"]["n_xy_port"] >= 36
    assert t1_gen["port_reflection"]["n_fft_port"] >= 65536
    assert "gamma_tapered_raw_at_sync" in t1_gen["port_reflection"]
    assert "gamma_abrupt_unclipped_at_sync" in t1_gen["port_reflection"]
    assert "variant_rwg_8p1mm_thesis" in t2_gen
    assert "nf8" in t2_gen["variant_rwg_8p1mm_thesis"]
    assert t2_gen["variant_rwg_8p1mm_thesis"]["primary_radius_mm"] == 8.0
    assert "primary_radius_justification" in t2_gen["variant_rwg_8p1mm_thesis"]
    assert t4_gen["fit_3d_superposition_verification"]["n_xy"] >= 64
    for req_t3_key in (
        "n8_transverse_mesh_convergence",
        "runs_mtm_nxy48",
        "runs_mtm_nxy96_N8",
        "forward_control_N8_nxy48",
        "length_scaling_nxy48",
    ):
        assert req_t3_key in t3_gen
    for req_ls_key in (
        "re_z_pk_over_cold_sinc2_ratio",
        "passband_integrated_re_z_fit_over_cold_sinc2_ratio",
        "n8_passband_integrated_re_z_fit_over_cold_sinc2_ratio",
        "effective_width_fit_over_cold_sinc2_ratio",
        "n8_effective_width_fit_over_cold_sinc2_ratio",
    ):
        assert req_ls_key in t3_gen["length_scaling"]
    assert _check_nxy48_length_scaling_spectra_not_reused(
        t3_gen["length_scaling"],
        t3_gen["length_scaling_nxy48"],
    )
    assert _check_nxy48_length_scaling_spectra_not_reused(
        disk_dicts["task_03_wakefield_and_directionality.json"]["length_scaling"],
        disk_dicts["task_03_wakefield_and_directionality.json"]["length_scaling_nxy48"],
    )
    # Live mutation check (MUST 1): reusing n_xy=64 z3_traces when building length_scaling_nxy48 must fail
    agg_mut_reuse48_dir = tmp_path / "mut_reuse_nxy64_traces"
    aggregate_all_tutorial_data(
        agg_mut_reuse48_dir,
        figures_dir=None,
        source_data_dir=DATA_DIR,
        mutate_reuse_nxy64_traces_for_nxy48=True,
    )
    t3_mut_reuse48 = json.loads(
        (agg_mut_reuse48_dir / "task_03_wakefield_and_directionality.json").read_text(encoding="utf-8")
    )
    assert not _check_nxy48_length_scaling_spectra_not_reused(
        t3_mut_reuse48["length_scaling"],
        t3_mut_reuse48["length_scaling_nxy48"],
    )
    # SHOULD 2: _build_length_scaling_dict must raise KeyError (no silent analytic sinc^2 fallback) when spectra are missing
    with pytest.raises(KeyError, match="refusing silent analytic"):
        _build_length_scaling_dict(
            t3_gen["runs_mtm_nxy48"],
            [6, 8, 12],
            gamma_tap_note=0.22,
            cold_k_loss_per_m=3.7e14,
            cold_vg_m_s=-0.158 * C0,
            cold_f_sync_hz=11.42e9,
            z3_traces=None,
            allow_missing_spectra=False,
        )
    # Note 9: _build_length_scaling_dict must raise ValueError if spec.paper_vg_over_c is missing/non-negative
    with pytest.raises(ValueError, match="paper_vg_over_c"):
        _build_length_scaling_dict(
            t3_gen["runs_mtm_nxy48"],
            [6, 8, 12],
            gamma_tap_note=0.22,
            cold_k_loss_per_m=3.7e14,
            cold_vg_m_s=-0.158 * C0,
            cold_f_sync_hz=11.42e9,
            z3_traces=None,
            allow_missing_spectra=True,
            spec=WagonWheelMTMSpec(paper_vg_over_c=0.158),
        )

    # Verify dead keys and duplicate *_with_fit_stored_* keys (Item 10) are absent from committed generator output
    t5_gen = gen_dicts["task_05_analytic_power_and_energy_balance.json"]
    for dead_key in (
        "residual_stored_energy_analytic_estimate_mj",
        "e_accounted_with_fit_stored_mj",
        "total_accounted_em_energy_with_fit_stored_mj",
        "rel_error_total_accounted_fit_vs_bunch_loss",
    ):
        assert dead_key not in t5_gen["energy_balance_N8"]
        assert dead_key not in t3_gen["runs_mtm"]["8"]

    # Live mutation check: dropping 'n8_transverse_mesh_convergence' or 'variant_rwg_8p1mm_thesis'
    # in build_dummy_all_json_dicts must fail _check_json_key_sets_match
    mut_dicts = build_dummy_all_json_dicts(
        tmp_path / "mutated",
        mutate_drop_key="n8_transverse_mesh_convergence",
    )
    assert not _check_json_key_sets_match(disk_dicts, mut_dicts)


_FROZEN_R5_RAW_SHA256: dict[str, str] = {
    "t1.port_reflection": "89afc3bd73d08ee24d672debd6211b2af4b347a40ff24b6b771ce6cf621fcf16",
    "t2.empty_waveguide_tm01_cutoff_fit_ghz": "e4ea2953c742023ba99792a3d816609ff98e488628921188ef27f280dfc87961",
    "t2.dispersion.raw": "1cfeaa3ff1cac11c8da9b3ec046e709c791bf75e38b338a39ac47fa6d3f4339d",
    "t2.uniform_levels.nf4": "db439ee75d3cf47ff3fd224242abe17bf0f27c5a7f801c004bb472871c96d8a4",
    "t2.uniform_levels.nf6": "523e52710da6d94ec3bed7038c49d102271d1fb5668f61aea5ee7d47067bd460",
    "t2.uniform_levels.nf8": "5a984ef8ac2428b89f4b65b7c51b286ab7826199c362147148aaf0831100f2f2",
    "t2.graded_levels.nf4": "ef6d31980736f45c2325fdeb72ab8ef298264df725bad38d82b494a7756632b2",
    "t2.graded_levels.nf6": "9a46328fc62c6d6b156d550cbe41d0c294dcc0221c082d9650ab67975e24b66d",
    "t2.graded_levels.nf8": "ab23ee53d1886fa0c4bd5e00dad6f91ddbaf79cfef0c9c9039aaed588faa559a",
    "t2.variant_rwg_8p1mm_thesis.nf4": "3e2e49861eb80f1f0fb2aecb67234bafd378b8c32a67b5e573c53e95c0f80faf",
    "t2.variant_rwg_8p1mm_thesis.nf6": "abd62decce1a9d5f001eeb4babfe0f5b5c7af6e4b24983e6b902a2619b812e4c",
    "t2.variant_rwg_8p1mm_thesis.nf8": "d06fc07ab993cee7e2e73141981f654e1311b8b6c2d2782f1f8a153c2f8b10e2",
    "t3.runs_mtm.6": "7c8ef6bfc841bacb5d070f1feda19bf32481e93ce3f528285ed6985b067c52ae",
    "t3.runs_mtm.8": "ba4cf8c6b968b82591223a54530a001bf8b2b001d050ee6269b4413dd8ce9914",
    "t3.runs_mtm.12": "b4fb77b57ed6baa9b9236092e803e7d449291f8acbf75f6beda832a2b43dd74d",
    "t3.runs_mtm_nxy48.6": "c1703ae6dd9c0240b9f5f263b2be160e363f97fe356b10735a549f588df5d7fe",
    "t3.runs_mtm_nxy48.8": "4a9bd27d44facd6f48d74e6daae3067fe33b7e9bb62e197062d41688571ce899",
    "t3.runs_mtm_nxy48.12": "d4f8543de941d5ff8ac8cb93da82f4094de4b3131f3e027797426644f4d5793f",
    "t3.runs_mtm_nxy96_N8": "69d59aa4ff7fd0ad95b7bec80fc722ed334147087490718cc5f8d5dd8c58f938",
    "t3.forward_control_N8": "c115082657e7c113e7e26240fb66ebe1ceddfa20e720c85fd59efe5e15290462",
    "t3.forward_control_N8_nxy48": "668e56a69b301b422fdec5ef01a4fe8d19a578c61e1fa112e9e6f4db6795ebf4",
    "t3.port_gamma_tapered_at_sync_values": "b24f23858b90f9c36890b0d45344a2298bbc0b32d5bc54156e4d5ecfd7bae66d",
    "t4.fit_3d_superposition_verification": "1f8e57ffbfd8092968b85b2d6dc94000fb4cb7ef96b67b878a155aebed913ac5",
}


def _check_raw_per_run_sha256(target_dir: Path) -> bool:
    """Return True if every raw per-run sub-dict in target_dir matches the frozen r5 SHA-256 hash."""
    import hashlib

    def _sha(obj: object) -> str:
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    t1 = json.loads((target_dir / "task_01_geometry_and_port_reflection.json").read_text(encoding="utf-8"))
    t2 = json.loads((target_dir / "task_02_cold_test_and_convergence.json").read_text(encoding="utf-8"))
    t3 = json.loads((target_dir / "task_03_wakefield_and_directionality.json").read_text(encoding="utf-8"))
    t4 = json.loads((target_dir / "task_04_two_bunch_superposition.json").read_text(encoding="utf-8"))

    actual = {
        "t1.port_reflection": _sha(t1["port_reflection"]),
        "t2.empty_waveguide_tm01_cutoff_fit_ghz": _sha(t2["empty_waveguide_tm01_cutoff_fit_ghz"]),
        "t2.dispersion.raw": _sha(
            {
                k: t2["dispersion"][k]
                for k in (
                    "phi_deg",
                    "f_mode_ghz",
                    "wall_time_s",
                    "forward_control_phi_deg",
                    "forward_control_f_ghz",
                )
            }
        ),
        "t2.uniform_levels.nf4": _sha(t2["uniform_levels"]["nf4"]),
        "t2.uniform_levels.nf6": _sha(t2["uniform_levels"]["nf6"]),
        "t2.uniform_levels.nf8": _sha(t2["uniform_levels"]["nf8"]),
        "t2.graded_levels.nf4": _sha(t2["graded_levels"]["nf4"]),
        "t2.graded_levels.nf6": _sha(t2["graded_levels"]["nf6"]),
        "t2.graded_levels.nf8": _sha(t2["graded_levels"]["nf8"]),
        "t2.variant_rwg_8p1mm_thesis.nf4": _sha(t2["variant_rwg_8p1mm_thesis"]["nf4"]),
        "t2.variant_rwg_8p1mm_thesis.nf6": _sha(t2["variant_rwg_8p1mm_thesis"]["nf6"]),
        "t2.variant_rwg_8p1mm_thesis.nf8": _sha(t2["variant_rwg_8p1mm_thesis"]["nf8"]),
        "t3.runs_mtm.6": _sha(t3["runs_mtm"]["6"]),
        "t3.runs_mtm.8": _sha(t3["runs_mtm"]["8"]),
        "t3.runs_mtm.12": _sha(t3["runs_mtm"]["12"]),
        "t3.runs_mtm_nxy48.6": _sha(t3["runs_mtm_nxy48"]["6"]),
        "t3.runs_mtm_nxy48.8": _sha(t3["runs_mtm_nxy48"]["8"]),
        "t3.runs_mtm_nxy48.12": _sha(t3["runs_mtm_nxy48"]["12"]),
        "t3.runs_mtm_nxy96_N8": _sha(t3["runs_mtm_nxy96_N8"]),
        "t3.forward_control_N8": _sha(t3["forward_control_N8"]),
        "t3.forward_control_N8_nxy48": _sha(t3["forward_control_N8_nxy48"]),
        "t3.port_gamma_tapered_at_sync_values": _sha(
            t3["n8_transverse_mesh_convergence"]["port_gamma_tapered_at_sync_values"]
        ),
        "t4.fit_3d_superposition_verification": _sha(t4["fit_3d_superposition_verification"]),
    }
    return actual == _FROZEN_R5_RAW_SHA256


def _check_simulation_derived_values_bit_for_bit(orig_dir: Path, agg_dir: Path) -> bool:
    """Return True if every simulation-derived (non-aggregate) value in agg_dir matches orig_dir and frozen r5 SHA-256 hashes."""
    if not _check_raw_per_run_sha256(orig_dir) or not _check_raw_per_run_sha256(agg_dir):
        return False
    t1_o = json.loads((orig_dir / "task_01_geometry_and_port_reflection.json").read_text(encoding="utf-8"))
    t1_a = json.loads((agg_dir / "task_01_geometry_and_port_reflection.json").read_text(encoding="utf-8"))
    if t1_o["port_reflection"] != t1_a["port_reflection"]:
        return False

    t2_o = json.loads((orig_dir / "task_02_cold_test_and_convergence.json").read_text(encoding="utf-8"))
    t2_a = json.loads((agg_dir / "task_02_cold_test_and_convergence.json").read_text(encoding="utf-8"))
    for k2 in ("empty_waveguide_tm01_cutoff_fit_ghz", "uniform_levels", "graded_levels"):
        if t2_o[k2] != t2_a[k2]:
            return False
    for kd in ("phi_deg", "f_mode_ghz", "forward_control_phi_deg", "forward_control_f_ghz"):
        if t2_o["dispersion"][kd] != t2_a["dispersion"][kd]:
            return False
    for kv in ("nf4", "nf6", "nf8"):
        if t2_o["variant_rwg_8p1mm_thesis"][kv] != t2_a["variant_rwg_8p1mm_thesis"][kv]:
            return False

    t3_o = json.loads((orig_dir / "task_03_wakefield_and_directionality.json").read_text(encoding="utf-8"))
    t3_a = json.loads((agg_dir / "task_03_wakefield_and_directionality.json").read_text(encoding="utf-8"))
    for k3 in (
        "n_periods_list",
        "runs_mtm",
        "runs_mtm_nxy48",
        "runs_mtm_nxy96_N8",
        "forward_control_N8",
        "forward_control_N8_nxy48",
    ):
        if t3_o[k3] != t3_a[k3]:
            return False
    if (
        t3_o["n8_transverse_mesh_convergence"]["port_gamma_tapered_at_sync_values"]
        != t3_a["n8_transverse_mesh_convergence"]["port_gamma_tapered_at_sync_values"]
    ):
        return False

    t4_o = json.loads((orig_dir / "task_04_two_bunch_superposition.json").read_text(encoding="utf-8"))
    t4_a = json.loads((agg_dir / "task_04_two_bunch_superposition.json").read_text(encoding="utf-8"))
    if t4_o["fit_3d_superposition_verification"] != t4_a["fit_3d_superposition_verification"]:
        return False

    sm_o = json.loads((orig_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    sm_a = json.loads((agg_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    for k_sm in (
        "tm01_cutoff_empty_fit_ghz",
        "port_gamma_tapered_coax_at_sync",
        "cold_f_sync_nf4_ghz",
        "cold_f_sync_nf6_ghz",
        "cold_f_sync_nf8_ghz",
        "wake_f_pk_N8_ghz",
        "wake_re_z_pk_N8_ohm",
        "single_bunch_45nc_power_fit_mw",
        "single_bunch_45nc_power_fit_nxy96_mw",
        "bwd_mtm_upstream_energy_mj",
        "fwd_control_downstream_energy_mj",
        "two_bunch_superposition_rel_l2_error_E",
        "wake_n8_nxy48_p_up_mw",
        "wake_n8_nxy64_p_up_mw",
        "wake_n8_nxy96_p_up_mw",
    ):
        if sm_o[k_sm] != sm_a[k_sm]:
            return False
    return True


def test_aggregate_only_reproduces_simulation_values_bit_for_bit_with_mutations(tmp_path: Path) -> None:
    """Verify --aggregate-only runs zero FIT simulations and reproduces all simulation-derived values bit-for-bit."""
    from unittest.mock import patch
    import _helpers as h_mod
    from wakis import SolverFIT3D

    agg_dir = tmp_path / "agg_clean"
    with (
        patch.object(SolverFIT3D, "__init__", side_effect=RuntimeError("No FIT SolverFIT3D.__init__ allowed")),
        patch.object(SolverFIT3D, "_one_step", side_effect=RuntimeError("No FIT SolverFIT3D._one_step allowed")),
        patch.object(
            SolverFIT3D, "one_step", side_effect=RuntimeError("No FIT SolverFIT3D.one_step allowed"), create=True
        ),
        patch.object(h_mod, "measure_upstream_port_reflection", side_effect=RuntimeError("No FIT allowed")),
        patch.object(h_mod, "measure_empty_waveguide_tm01_cutoff_fit", side_effect=RuntimeError("No FIT allowed")),
        patch.object(h_mod, "evaluate_bloch_unit_cell", side_effect=RuntimeError("No FIT allowed")),
        patch.object(h_mod, "run_finite_mtm_wakefield", side_effect=RuntimeError("No FIT allowed")),
        patch.object(h_mod, "verify_two_bunch_fit_superposition", side_effect=RuntimeError("No FIT allowed")),
    ):
        aggregate_all_tutorial_data(agg_dir, figures_dir=None, source_data_dir=DATA_DIR)

    assert _check_raw_per_run_sha256(DATA_DIR)
    assert _check_raw_per_run_sha256(agg_dir)
    assert _check_simulation_derived_values_bit_for_bit(DATA_DIR, agg_dir)

    # Mutation check: corrupting raw per-run values inside aggregate_all_tutorial_data must fail bit-for-bit and SHA-256 checks
    agg_mut_dir = tmp_path / "agg_mutated"
    aggregate_all_tutorial_data(
        agg_mut_dir,
        figures_dir=None,
        source_data_dir=DATA_DIR,
        mutate_corrupt_raw_run=True,
    )
    assert not _check_raw_per_run_sha256(agg_mut_dir)
    assert not _check_simulation_derived_values_bit_for_bit(DATA_DIR, agg_mut_dir)


