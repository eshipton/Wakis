"""Tests for Tutorial 8.5: An electron bunch passing through a slow-wave structure (beam-wave impedance)."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = REPO_ROOT / "tutorials" / "08_vacuum_electronics" / "08-05_bunch_through_sws"
SCRIPT_PATH = TUTORIAL_DIR / "bunch_through_sws.py"
NOTEBOOK_PATH = TUTORIAL_DIR / "bunch_through_sws.ipynb"
DRIVER_PATH = TUTORIAL_DIR / "generate_data.py"
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as hlp  # noqa: E402


# ---------------------------------------------------------------------------
# Assertion helpers (used in both positive and code-path mutation checks)
# ---------------------------------------------------------------------------


def _assert_sinc2_scaling_and_width(
    lengths_m: np.ndarray,
    peaks_ohm: np.ndarray,
    fwhms_hz: np.ndarray,
    beta_n_rad_m: float,
    k_pierce_ohm: float,
    u0_m_s: float,
    vg_m_s: float,
    slope_tol: float = 0.025,
    rel_tol: float = 0.06,
) -> None:
    """Assert L^2 peak scaling, 1/L width scaling, and walk-off FWHM agreement."""
    slope_pk, _ = np.polyfit(np.log(lengths_m), np.log(peaks_ohm), 1)
    slope_w, _ = np.polyfit(np.log(lengths_m), np.log(fwhms_hz), 1)
    assert abs(slope_pk - 2.0) < slope_tol, f"Peak log-log slope {slope_pk:.4f} != 2.0"
    assert abs(slope_w + 1.0) < slope_tol, f"FWHM log-log slope {slope_w:.4f} != -1.0"

    expected_peaks = 0.25 * (beta_n_rad_m**2) * k_pierce_ohm * (lengths_m**2)
    rel_err_pk = np.max(np.abs(peaks_ohm - expected_peaks) / expected_peaks)
    assert rel_err_pk < rel_tol, f"Peak Re Z_|| relative error {rel_err_pk:.4f} exceeds {rel_tol}"

    expected_fwhms = np.array(
        [hlp.analytic_fwhm_hz(length_m=float(L), u0_m_s=u0_m_s, vg_m_s=vg_m_s) for L in lengths_m]
    )
    rel_err_w = np.max(np.abs(fwhms_hz - expected_fwhms) / expected_fwhms)
    assert rel_err_w < rel_tol, f"FWHM relative error {rel_err_w:.4f} exceeds {rel_tol}"


def _assert_pierce_k_agreement(
    k_extracted_ohm: float,
    k_reference_ohm: float,
    rel_tol: float = 0.05,
) -> None:
    """Assert that beam-extracted Pierce impedance agrees with cold-test reference."""
    rel_err = abs(k_extracted_ohm - k_reference_ohm) / k_reference_ohm
    assert rel_err < rel_tol, (
        f"Extracted K_n = {k_extracted_ohm:.4f} Ohm differs from reference "
        f"{k_reference_ohm:.4f} Ohm by {rel_err:.4f} (> {rel_tol})"
    )


def _assert_stopping_power_agreement(
    dw_dz_val: float,
    dw_dz_ref: float,
    rel_tol: float = 0.05,
) -> None:
    """Assert that infinite-structure stopping power dW/dz matches analytical limit."""
    rel_err = abs(dw_dz_val - dw_dz_ref) / dw_dz_ref
    assert rel_err < rel_tol, (
        f"Stopping power {dw_dz_val:.4e} J/m differs from reference {dw_dz_ref:.4e} J/m "
        f"by {rel_err:.4f} (> {rel_tol})"
    )


def _assert_poynting_direction_and_balance(
    bwd_energy_in_j: float,
    bwd_energy_out_j: float,
    bwd_w_loss_j: float,
    fwd_energy_in_j: float,
    fwd_energy_out_j: float,
    fwd_w_loss_j: float,
    balance_rel_tol: float = 0.08,
) -> None:
    """Assert upstream/downstream Poynting direction and full-section energy balance closure."""
    assert bwd_energy_in_j < 0.0, f"Backward-wave upstream energy E_in = {bwd_energy_in_j} must be < 0"
    assert abs(bwd_energy_in_j) > 3.0 * abs(bwd_energy_out_j), (
        "Backward-wave upstream energy must dominate downstream energy"
    )
    assert fwd_energy_out_j > 0.0, f"Forward-wave downstream energy E_out = {fwd_energy_out_j} must be > 0"
    assert abs(fwd_energy_out_j) > 3.0 * abs(fwd_energy_in_j), (
        "Forward-wave downstream energy must dominate upstream energy"
    )

    bwd_rad = bwd_energy_out_j - bwd_energy_in_j
    fwd_rad = fwd_energy_out_j - fwd_energy_in_j
    err_bwd = abs(bwd_rad - bwd_w_loss_j) / bwd_w_loss_j
    err_fwd = abs(fwd_rad - fwd_w_loss_j) / fwd_w_loss_j
    assert err_bwd < balance_rel_tol, (
        f"Backward energy balance error {err_bwd:.4f} (E_rad={bwd_rad:.3e} vs W_loss={bwd_w_loss_j:.3e})"
    )
    assert err_fwd < balance_rel_tol, (
        f"Forward energy balance error {err_fwd:.4f} (E_rad={fwd_rad:.3e} vs W_loss={fwd_w_loss_j:.3e})"
    )


def _assert_detuned_frequency_shift(
    df_measured_ghz: float,
    df_expected_ghz: float,
    rel_tol: float = 0.08,
) -> None:
    """Assert that velocity detuning shifts the synchronous peak frequency by predicted df_s."""
    assert np.sign(df_measured_ghz) == np.sign(df_expected_ghz), (
        f"Detuned frequency shift {df_measured_ghz:+.3f} GHz has wrong sign vs {df_expected_ghz:+.3f} GHz"
    )
    rel_err = abs(df_measured_ghz - df_expected_ghz) / abs(df_expected_ghz)
    assert rel_err < rel_tol, (
        f"Detuned shift {df_measured_ghz:+.3f} GHz differs from {df_expected_ghz:+.3f} GHz by {rel_err:.4f}"
    )


def _assert_standing_wave_cavity_identity(
    rq_circuit_ohm: float,
    omega_m_rad_s: float,
    vg_m_s: float,
    length_m: float,
    tw_peak_re_z_ohm: float,
    rel_tol: float = 0.01,
) -> None:
    """Assert that mode area A_m = (pi/2) * omega_m * (R/Q)_circuit divided by FSR recovers Re Z_||^TW."""
    area_m = 0.5 * np.pi * omega_m_rad_s * rq_circuit_ohm
    fsr_rad_s = np.pi * abs(vg_m_s) / length_m
    reconstructed_tw_z = area_m / fsr_rad_s
    rel_err = abs(reconstructed_tw_z - tw_peak_re_z_ohm) / tw_peak_re_z_ohm
    assert rel_err < rel_tol, (
        f"Reconstructed TW peak {reconstructed_tw_z:.2f} Ohm differs from {tw_peak_re_z_ohm:.2f} Ohm by {rel_err:.4f}"
    )


def _assert_summary_table_matches_json(
    table_markdown: str,
    summary_dict: dict[str, float],
) -> None:
    """Assert that the Quantitative Summary markdown table contains the exact formatted JSON values."""
    expected_tokens = [
        f"{summary_dict['termination_f_sync_ghz']:.2f}",
        f"{summary_dict['termination_gamma_tapered_at_sync']:.4f}",
        f"{summary_dict['termination_gamma_abrupt_at_sync']:.4f}",
        f"{summary_dict['length_scaling_peak_slope_fit']:.4f}",
        f"{summary_dict['length_scaling_fwhm_slope_fit']:.4f}",
        f"{summary_dict['n20_f_pk_ghz']:.2f}",
        f"{summary_dict['n20_re_z_pk_ohm']:.1f}",
        f"{summary_dict['n20_k_on_axis_beam_ohm']:.3f}",
        f"{summary_dict['conv_k_axis_nf4_ohm']:.3f}",
        f"{summary_dict['conv_k_axis_nf6_ohm']:.3f}",
        f"{summary_dict['conv_k_axis_nf8_ohm']:.3f}",
        f"{summary_dict['conv_k_axis_band_min_ohm']:.3f}",
        f"{summary_dict['conv_k_axis_band_max_ohm']:.3f}",
        f"{summary_dict['n20_k_sheet_avg_7node_mean_ohm']:.3f}",
        f"{summary_dict['n20_k_sheet_avg_trapz_ohm']:.3f}",
        f"{summary_dict['n20_dw_dz_unattenuated_j_per_m']*1e9:.3f}",
        f"{summary_dict['fwd_k0_at_sync_y50um_fit_ohm']:.3f}",
        f"{summary_dict['fwd_re_z_pk_fit_ohm']:.1f}",
        f"{summary_dict['bwd_w_loss_between_ports_pj']:.3f}",
        f"{summary_dict['fwd_w_loss_between_ports_pj']:.3f}",
        f"{summary_dict['bwd_poynting_energy_in_pj']:.3f}",
        f"{summary_dict['fwd_poynting_energy_out_pj']:.3f}",
        f"{summary_dict['detuning_minus_5pct_df_fit_ghz']:.2f}",
        f"{summary_dict['cavity_m6_f_fit_ghz']:.2f}",
        f"{summary_dict['cavity_m6_rq_circuit_fit_ohm']:.1f}",
        f"{summary_dict['marosi_3d_re_z_pk_ohm']*1e-3:.2f}",
        f"{summary_dict['marosi_3d_k_at_pk_ohm']:.2f}",
    ]
    for tok in expected_tokens:
        assert tok in table_markdown, f"Expected JSON metric token '{tok}' missing from summary table"


# ---------------------------------------------------------------------------
# 1. Analytical Identities & Code-Path Mutation Failure Tests
# ---------------------------------------------------------------------------


def test_no_mutation_flags_in_production_helpers() -> None:
    """Ensure production helper functions in _helpers.py contain no artificial mutation flags."""
    forbidden_prefixes = ("drop_", "flip_", "mutate_")
    for name, func in inspect.getmembers(hlp, inspect.isfunction):
        sig = inspect.signature(func)
        for param_name in sig.parameters:
            assert not param_name.startswith(forbidden_prefixes), (
                f"Production helper {name} has forbidden mutation parameter '{param_name}'"
            )


def test_lorentz_reciprocity_sinc2_and_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Re Z_||(w) = (1/4) beta_n^2 K_n L^2 sinc^2(Delta_beta L / 2) and code-path mutation failures."""
    cold = hlp.load_cold_test_benchmarks()
    l2 = cold["levels_2d"]["L2"]
    u0 = cold["u0_2d_m_s"]
    f_s = l2["f_sync_bwd_hz"]
    beta_m1 = l2["beta_m1_rad_m"]
    k_ax = l2["k_m1_on_axis_ohm"]
    vg = l2["vg_bwd_m_s"]

    lengths = np.array([5, 10, 20, 40], dtype=float) * 0.5e-3
    f_grid = np.linspace(90.0e9, 140.0e9, 4001)

    def _eval_sweep() -> tuple[np.ndarray, np.ndarray]:
        peaks_list = []
        fwhms_list = []
        for L in lengths:
            rz = hlp.analytic_sws_beam_impedance(
                f_grid,
                f_sync_hz=f_s,
                beta_n_rad_m=beta_m1,
                k_pierce_ohm=k_ax,
                length_m=float(L),
                u0_m_s=u0,
                vg_m_s=vg,
            )
            pk = float(np.max(rz))
            peaks_list.append(pk)
            above = f_grid[rz >= 0.5 * pk]
            fwhms_list.append(float(above[-1] - above[0]))
        return np.array(peaks_list), np.array(fwhms_list)

    peaks, fwhms = _eval_sweep()
    _assert_sinc2_scaling_and_width(
        lengths, peaks, fwhms, beta_m1, k_ax, u0, vg, slope_tol=2e-3, rel_tol=1e-2
    )

    # Mutation 1: mutate analytic_sws_beam_impedance to drop the 1/4 Lorentz-reciprocity factor -> must fail
    orig_imp = hlp.analytic_sws_beam_impedance
    with monkeypatch.context() as mp:
        mp.setattr(hlp, "analytic_sws_beam_impedance", lambda *a, **kw: 4.0 * orig_imp(*a, **kw))
        peaks_mut, fwhms_mut = _eval_sweep()
        with pytest.raises(AssertionError):
            _assert_sinc2_scaling_and_width(lengths, peaks_mut, fwhms_mut, beta_m1, k_ax, u0, vg)

    # Mutation 2: mutate analytic_sws_beam_impedance to flip the sign of backward v_g -> must fail
    with monkeypatch.context() as mp:
        mp.setattr(
            hlp,
            "analytic_sws_beam_impedance",
            lambda f_hz, **kw: orig_imp(f_hz, **{**kw, "vg_m_s": -float(kw["vg_m_s"])}),
        )
        peaks_vg_mut, fwhms_vg_mut = _eval_sweep()
        with pytest.raises(AssertionError):
            _assert_sinc2_scaling_and_width(lengths, peaks_vg_mut, fwhms_vg_mut, beta_m1, k_ax, u0, vg)


def test_pierce_k_extraction_stopping_power_and_form_factor_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify K_n inversion, node/continuum sheet averaging, stopping power dW/dz, and code-path mutations."""
    cold = hlp.load_cold_test_benchmarks()
    l1 = cold["levels_2d"]["L1"]
    l2 = cold["levels_2d"]["L2"]
    u0 = cold["u0_2d_m_s"]
    f_s = l2["f_sync_bwd_hz"]
    beta_m1 = l2["beta_m1_rad_m"]
    k_ax = l2["k_m1_on_axis_ohm"]
    vg = l2["vg_bwd_m_s"]
    L = 10.0e-3
    q = 1.0e-12

    z_pk = 0.25 * (beta_m1**2) * k_ax * (L**2)
    k_rec = hlp.extract_pierce_k_from_peak_z(z_pk, beta_n_rad_m=beta_m1, length_m=L)
    _assert_pierce_k_agreement(k_rec, k_ax, rel_tol=1e-12)

    # Discrete 7-node sheet-beam averaging factor must match Tutorial 6.2 L1 node ratio within 1%
    eta_cont = hlp.sheet_beam_averaging_factor(beta_m1, f_s, 0.15e-3)
    eta_nodes_l1 = hlp.sheet_beam_node_averaging_factor(beta_m1, f_s, 0.15e-3, 25.0e-6)
    assert eta_nodes_l1 > eta_cont > 1.10
    _assert_pierce_k_agreement(eta_nodes_l1, l1["k_m1_ratio_beam_over_axis"], rel_tol=0.01)

    # Mutation: mutating extract_pierce_k_from_peak_z to drop factor 4 must fail
    orig_ext = hlp.extract_pierce_k_from_peak_z
    with monkeypatch.context() as mp:
        mp.setattr(hlp, "extract_pierce_k_from_peak_z", lambda *a, **kw: 0.25 * orig_ext(*a, **kw))
        k_mut = hlp.extract_pierce_k_from_peak_z(z_pk, beta_n_rad_m=beta_m1, length_m=L)
        with pytest.raises(AssertionError):
            _assert_pierce_k_agreement(k_mut, k_ax)

    # Infinite-structure stopping power dW/dz = q^2 * beta_n^2 * K_n / (2 * |1/u0 - 1/v_g|)
    dw_dz_th = hlp.analytic_stopping_power_j_per_m(
        q_coulomb=q, beta_n_rad_m=beta_m1, k_pierce_ohm=k_ax, u0_m_s=u0, vg_m_s=vg
    )
    f_dense = np.linspace(50.0e9, 180.0e9, 30001)
    rz_dense = hlp.analytic_sws_beam_impedance(
        f_dense,
        f_sync_hz=f_s,
        beta_n_rad_m=beta_m1,
        k_pierce_ohm=k_ax,
        length_m=L,
        u0_m_s=u0,
        vg_m_s=vg,
    )
    k_loss_unatt = hlp.compute_loss_factor_v_per_pc(f_dense, rz_dense, 0.0, u0)
    dw_dz_num = (k_loss_unatt * 1e12) * (q**2) / L
    _assert_stopping_power_agreement(dw_dz_num, dw_dz_th, rel_tol=0.015)

    # Mutations on analytic_stopping_power_j_per_m and compute_loss_factor_v_per_pc -> must fail
    orig_sp = hlp.analytic_stopping_power_j_per_m
    with monkeypatch.context() as mp:
        mp.setattr(hlp, "analytic_stopping_power_j_per_m", lambda **kw: 2.0 * orig_sp(**kw))
        with pytest.raises(AssertionError):
            _assert_stopping_power_agreement(
                hlp.analytic_stopping_power_j_per_m(
                    q_coulomb=q, beta_n_rad_m=beta_m1, k_pierce_ohm=k_ax, u0_m_s=u0, vg_m_s=vg
                ),
                dw_dz_th,
            )

    orig_kloss = hlp.compute_loss_factor_v_per_pc
    with monkeypatch.context() as mp:
        mp.setattr(hlp, "compute_loss_factor_v_per_pc", lambda *a, **kw: np.pi * orig_kloss(*a, **kw))
        k_loss_mut = hlp.compute_loss_factor_v_per_pc(f_dense, rz_dense, 0.0, u0)
        with pytest.raises(AssertionError):
            _assert_stopping_power_agreement((k_loss_mut * 1e12) * (q**2) / L, dw_dz_th)


def test_reflection_bound_detuning_and_standing_wave_rq_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify port reflection bound, velocity detuning shift, and standing-wave (R/Q)_m to K_n identity."""
    cold = hlp.load_cold_test_benchmarks()
    l1 = cold["levels_2d"]["L1"]
    u0 = cold["u0_2d_m_s"]
    f_s = l1["f_sync_bwd_hz"]
    beta_m1 = l1["beta_m1_rad_m"]
    k_ax = l1["k_m1_on_axis_ohm"]
    vg = l1["vg_bwd_m_s"]
    L = 5.0e-3

    # 1. Port reflection error bound 2|Gamma|/(1 - |Gamma|^2)
    b_exact = float(hlp.reflection_induced_peak_error_bound(0.1854))
    assert np.isclose(b_exact, 2.0 * 0.1854 / (1.0 - 0.1854**2), rtol=1e-12)

    # 2. Velocity detuning shift df_s
    df_m5 = (
        hlp.analytic_detuned_frequency_shift_hz(
            f_sync_hz=f_s, u0_m_s=u0, delta_u0_m_s=-0.05 * u0, vg_m_s=vg
        )
        * 1e-9
    )
    _assert_detuned_frequency_shift(-3.794, df_m5, rel_tol=0.02)

    orig_det = hlp.analytic_detuned_frequency_shift_hz
    with monkeypatch.context() as mp:
        mp.setattr(hlp, "analytic_detuned_frequency_shift_hz", lambda **kw: -orig_det(**kw))
        df_mut = (
            hlp.analytic_detuned_frequency_shift_hz(
                f_sync_hz=f_s, u0_m_s=u0, delta_u0_m_s=-0.05 * u0, vg_m_s=vg
            )
            * 1e-9
        )
        with pytest.raises(AssertionError):
            _assert_detuned_frequency_shift(-3.794, df_mut, rel_tol=0.02)

    # 3. Standing-wave cavity (R/Q)_circuit,m relation to traveling-wave Re Z_||
    omega_s = 2.0 * np.pi * f_s
    tw_pk = 0.25 * (beta_m1**2) * k_ax * (L**2)
    rq_circ = hlp.analytic_standing_wave_r_over_q_ohm(
        f_mode_hz=f_s,
        beta_n_rad_m=beta_m1,
        k_pierce_ohm=k_ax,
        length_m=L,
        vg_m_s=vg,
        convention="circuit",
    )
    _assert_standing_wave_cavity_identity(rq_circ, omega_s, vg, L, tw_pk, rel_tol=1e-12)

    orig_rq = hlp.analytic_standing_wave_r_over_q_ohm
    with monkeypatch.context() as mp:
        mp.setattr(hlp, "analytic_standing_wave_r_over_q_ohm", lambda **kw: 2.0 * orig_rq(**kw))
        rq_mut = hlp.analytic_standing_wave_r_over_q_ohm(
            f_mode_hz=f_s,
            beta_n_rad_m=beta_m1,
            k_pierce_ohm=k_ax,
            length_m=L,
            vg_m_s=vg,
            convention="circuit",
        )
        with pytest.raises(AssertionError):
            _assert_standing_wave_cavity_identity(rq_mut, omega_s, vg, L, tw_pk)


# ---------------------------------------------------------------------------
# 2. Cached FIT Simulation Results & Referee Audit Checks (Items 1-6, 8)
# ---------------------------------------------------------------------------


def test_cached_fit_results_and_quantitative_summary_table() -> None:
    """Verify all cached FIT simulation outputs (Tasks 1-6), referee fixes, and Quantitative Summary table."""
    assert DRIVER_PATH.is_file(), f"Missing driver script: {DRIVER_PATH}"

    t1 = json.loads((DATA_DIR / "task_01_termination_reflection.json").read_text())
    t2 = json.loads((DATA_DIR / "task_02_2d_length_sweep.json").read_text())
    t3 = json.loads((DATA_DIR / "task_03_fwd_vs_bwd_poynting.json").read_text())
    t4 = json.loads((DATA_DIR / "task_04_mutations_detuning_and_cavity.json").read_text())
    t5 = json.loads((DATA_DIR / "task_05_mesh_convergence.json").read_text())
    t6 = json.loads((DATA_DIR / "task_06_3d_marosi_beam_wake.json").read_text())
    sm = json.loads((DATA_DIR / "summary_metrics.json").read_text())

    # Referee Item 1: Production [0.25, 0.55] taper, unfloored/unclipped PEC normalization
    assert t1["taper_scales"] == [0.25, 0.55]
    assert t1["exit_ramp_down_scales"] == [0.55, 0.25]
    assert "no floor, no clip" in t1["normalization"]
    assert 0.15 < t1["gamma_tapered_at_sync"] < 0.22
    assert t1["reduction_factor_abrupt_over_tapered"] > 2.0

    # Referee Item 2: 2D per 1.0 mm width normalization (width_scale = 1.0)
    assert np.isclose(t2["spec"]["transverse_width_m"], 1.0e-3, rtol=1e-12)
    assert np.isclose(t2["runs"]["20"]["width_scale"], 1.0, rtol=1e-12)

    # Task 2: Length sweep (N = 5, 10, 20, 40)
    lengths = np.array([t2["runs"][str(n)]["active_length_m"] for n in [5, 10, 20, 40]])
    peaks = np.array([t2["runs"][str(n)]["re_z_pk_ohm"] for n in [5, 10, 20, 40]])
    fwhms = np.array([t2["runs"][str(n)]["fwhm_ghz"] * 1e9 for n in [5, 10, 20, 40]])
    l1 = t2["cold_test_L1"]
    _assert_sinc2_scaling_and_width(
        lengths,
        peaks,
        fwhms,
        beta_n_rad_m=l1["beta_m1_rad_m"],
        k_pierce_ohm=l1["k_m1_on_axis_ohm"],
        u0_m_s=t2["runs"]["20"]["u0_m_s"],
        vg_m_s=l1["vg_bwd_m_s"],
        slope_tol=0.02,
        rel_tol=0.07,
    )
    _assert_pierce_k_agreement(
        t2["runs"]["20"]["k_on_axis_beam_ohm"], l1["k_m1_on_axis_ohm"], rel_tol=0.035
    )

    # Referee Item 3: Off-axis sheet-beam y-sweep across {0, 25, 50, 75} um and 7-node mean vs 6.2 L1
    y_sw = t2["sheet_beam_y_sweep"]
    assert y_sw["y_positions_um"] == [0.0, 25.0, 50.0, 75.0]
    assert np.all(np.diff(y_sw["k_m1_at_y_ohm"]) > 0.0), "K_-1(y0) must increase monotonically off axis"
    _assert_pierce_k_agreement(
        y_sw["k_sheet_avg_7node_mean_ohm"], l1["k_m1_beam_avg_ohm"], rel_tol=0.035
    )
    _assert_pierce_k_agreement(
        y_sw["ratio_7node_mean_over_axis"], l1["k_m1_ratio_beam_over_axis"], rel_tol=0.01
    )

    _assert_stopping_power_agreement(
        t2["runs"]["20"]["dw_dz_unattenuated_j_per_m"],
        t2["dw_dz_analytic_unattenuated_j_per_m"],
        rel_tol=0.05,
    )

    # Referee Items 5 & 6: Forward-wave off-axis K_0(y0=50 um) validation, widened band, and smooth-port energy balance
    bwd = t3["backward_wave_glide_m1"]
    fwd = t3["forward_wave_mirror_n0"]
    assert fwd["f_search_band_ghz"] == [95.0, 145.0]
    _assert_pierce_k_agreement(
        fwd["k0_at_sync_y50um_fit_ohm"], fwd["k0_at_sync_y50um_cold_L1_ohm"], rel_tol=0.05
    )
    assert fwd["rel_diff_re_z_pk_vs_nonlinear_cold_L1"] < 0.08

    _assert_poynting_direction_and_balance(
        bwd["integrated_energy_in_j"],
        bwd["integrated_energy_out_j"],
        bwd["w_loss_between_smooth_ports_j"],
        fwd["integrated_energy_in_j"],
        fwd["integrated_energy_out_j"],
        fwd["w_loss_between_smooth_ports_j"],
        balance_rel_tol=0.06,
    )
    with pytest.raises(AssertionError):
        # Swapping upstream/downstream signs or using incomplete tunnel-only energy must fail
        _assert_poynting_direction_and_balance(
            bwd["tunnel_only_groove_plane_energy_in_j"],
            bwd["tunnel_only_groove_plane_energy_out_j"],
            bwd["w_loss_between_smooth_ports_j"],
            fwd["tunnel_only_groove_plane_energy_in_j"],
            fwd["tunnel_only_groove_plane_energy_out_j"],
            fwd["w_loss_between_smooth_ports_j"],
            balance_rel_tol=0.06,
        )

    # Task 4: Detuned velocity & shorted PEC cavity resonances
    d_m5 = t4["detuning_sweep"]["u0_minus_5pct"]
    _assert_detuned_frequency_shift(
        d_m5["delta_f_fit_ghz"], d_m5["delta_f_linear_analytic_ghz"], rel_tol=0.02
    )
    cav = t4["pec_cavity_resonances"]
    assert abs(cav["modes"][1]["f_fit_cavity_ghz"] - cav["modes"][1]["f_cold_test_ghz"]) < 0.25
    assert (
        abs(cav["reconstructed_traveling_wave_z_pk_ohm"] - cav["matched_traveling_wave_z_pk_ohm"])
        / cav["matched_traveling_wave_z_pk_ohm"]
        < 0.10
    )

    # Task 5 & Round-2 Referee Items 1 & 3: Non-monotone sequence, finest-level band, & length vs mesh separation
    assert not t5["is_monotone_k_on_axis"], "n_feat in {4, 6, 8} K_-1(0) sequence is non-monotone"
    assert not t5["richardson_extrapolation_justified"], "No Richardson order justified for non-monotone K_-1(0)"
    assert "k_on_axis_richardson_ohm" not in t5
    assert "conv_k_axis_rich_ohm" not in sm
    assert np.isclose(t5["k_on_axis_finest_band_min_ohm"], min(t5["k_on_axis_ohm_array"][1:]), rtol=1e-12)
    assert np.isclose(t5["k_on_axis_finest_band_max_ohm"], max(t5["k_on_axis_ohm_array"][1:]), rtol=1e-12)
    assert 2.55 < t5["k_on_axis_finest_band_min_ohm"] < t5["k_on_axis_finest_band_max_ohm"] < 2.75
    assert t5["k_on_axis_finest_band_rel_spread"] < 0.03
    assert t5["rel_diff_finest_band_vs_cold_L2_max"] < 0.07
    assert t5["rel_diff_finest_band_vs_cold_ref_max"] < 0.06
    assert (
        t5["runs"]["uniform_staircase_nf4"]["k_on_axis_beam_ohm"]
        > 1.8 * t5["runs"]["uniform_conformal_nf4"]["k_on_axis_beam_ohm"]
    )

    # Task 1 Round-2 Referee Item 4: idx_sync computed from cold-test benchmark data
    f_arr_t1 = np.asarray(t1["f_ghz"], dtype=float)
    expected_idx_sync = int(np.argmin(np.abs(f_arr_t1 - float(t1["f_sync_target_cold_ghz"]))))
    assert np.isclose(t1["f_sync_ghz"], f_arr_t1[expected_idx_sync], rtol=1e-12)

    # Task 6: 3D Marosi glide SWS beam wake vs 3D cold test
    assert t6["rel_diff_re_z_pk_vs_cold_lower_branch"] < 0.05
    assert t6["rel_diff_k_pk_vs_cold_lower_branch"] < 0.10

    # Quantitative Summary & Convergence Separation Tables in script & notebook must match JSON
    expected_md = hlp.build_quantitative_summary_markdown(DATA_DIR)
    expected_conv_md = hlp.build_convergence_separation_markdown(DATA_DIR)
    _assert_summary_table_matches_json(expected_md, sm)

    script_text = SCRIPT_PATH.read_text(encoding="utf-8")
    nb_json = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    nb_text = "\n".join("".join(cell.get("source", [])) for cell in nb_json.get("cells", []))
    _assert_summary_table_matches_json(script_text, sm)
    _assert_summary_table_matches_json(nb_text, sm)
    assert "2.619" not in script_text and "2.619" not in nb_text
    for line in expected_conv_md.splitlines():
        assert line in script_text, f"Convergence separation table line missing from script: {line}"
        assert line in nb_text, f"Convergence separation table line missing from notebook: {line}"

    # Mutating build_quantitative_summary_markdown must cause the table check to fail
    sm_mut = dict(sm)
    sm_mut["n20_re_z_pk_ohm"] = sm["n20_re_z_pk_ohm"] * 1.25
    with pytest.raises(AssertionError):
        _assert_summary_table_matches_json(expected_md, sm_mut)

    # All required setup and result figures (including GIF) must exist and be non-empty
    required_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00_setup_table.md",
        "00a_setup_geometry_marosi.png",
        "00b_setup_mesh_and_excitation_marosi.png",
        "00_setup_table_marosi.md",
        "01_termination_reflection_gamma.png",
        "02_re_z_vs_sinc2_length_sweep.png",
        "03_pierce_k_and_sheet_beam_profile.png",
        "04_stopping_power_and_loss_factor.png",
        "05_forward_vs_backward_poynting_and_width.png",
        "06_sws_bunch_excitation_fwd_vs_bwd.gif",
        "06_sws_bunch_excitation_fwd_vs_bwd.png",
        "07_mutations_detuned_u0_and_cavity_rq.png",
        "08_mesh_convergence.png",
        "09_3d_marosi_glide_beam_wake.png",
    ]
    for fn in required_figs:
        p = FIG_DIR / fn
        assert p.is_file() and p.stat().st_size > 200, f"Missing or empty figure/table: {p}"


# ---------------------------------------------------------------------------
# 3. Live 2D FIT Simulation of L^2 Peak & 1/L FWHM Scaling (N = 4 and N = 8)
# ---------------------------------------------------------------------------


def test_live_fit_length_scaling_and_detuned_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run live 2D FIT beam-wake simulations at N = 4 and N = 8 (quick=True) to verify L^2 and 1/L scaling."""
    spec = hlp.KaretnikovaGratingSpec()
    cold = hlp.load_cold_test_benchmarks()
    l1 = cold["levels_2d"]["L1"]
    u0, _ = hlp.relativistic_beam_velocity(spec.beam_voltage_v)

    res_n4 = hlp.run_2d_karetnikova_beam_wake(
        spec,
        n_periods=4,
        u0_m_s=u0,
        n_feat=4,
        use_graded_mesh=True,
        quick=True,
    )
    res_n8 = hlp.run_2d_karetnikova_beam_wake(
        spec,
        n_periods=8,
        u0_m_s=u0,
        n_feat=4,
        use_graded_mesh=True,
        quick=True,
    )

    lengths = np.array([res_n4["active_length_m"], res_n8["active_length_m"]], dtype=float)
    peaks = np.array([res_n4["re_z_pk_ohm"], res_n8["re_z_pk_ohm"]], dtype=float)
    fwhms = np.array([res_n4["fwhm_ghz"] * 1e9, res_n8["fwhm_ghz"] * 1e9], dtype=float)

    # Verify live FIT L^2 peak scaling, 1/L FWHM scaling, and cold-test Pierce K_-1(0) agreement
    _assert_sinc2_scaling_and_width(
        lengths,
        peaks,
        fwhms,
        beta_n_rad_m=l1["beta_m1_rad_m"],
        k_pierce_ohm=l1["k_m1_on_axis_ohm"],
        u0_m_s=u0,
        vg_m_s=l1["vg_bwd_m_s"],
        slope_tol=0.14,
        rel_tol=0.14,
    )
    _assert_pierce_k_agreement(res_n8["k_on_axis_beam_ohm"], l1["k_m1_on_axis_ohm"], rel_tol=0.08)

    # Code-path mutation 1 on live FIT wake integration: wrong time-centring of the beam current trajectory
    # (shifting ti_s by +0.5 * ti_s in _integrate_synchronous_wake_potential) must fail Pierce K agreement
    orig_integ = hlp._integrate_synchronous_wake_potential
    with monkeypatch.context() as mp:
        mp.setattr(
            hlp,
            "_integrate_synchronous_wake_potential",
            lambda t_arr, z_int, dz_int, ez_int, **kw: orig_integ(
                t_arr, z_int, dz_int, ez_int, **{**kw, "ti_s": 1.5 * float(kw["ti_s"])}
            ),
        )
        res_bad_tcenter = hlp.run_2d_karetnikova_beam_wake(
            spec,
            n_periods=4,
            u0_m_s=u0,
            n_feat=4,
            use_graded_mesh=True,
            quick=True,
        )
        with pytest.raises(AssertionError):
            _assert_pierce_k_agreement(
                res_bad_tcenter["k_on_axis_beam_ohm"], l1["k_m1_on_axis_ohm"], rel_tol=0.08
            )

    # Code-path mutation 2 on live FIT wake integration: flipping the sign of the beam-wave interaction term
    # (ez_int -> -ez_int in _integrate_synchronous_wake_potential) must fail Pierce K agreement
    with monkeypatch.context() as mp:
        mp.setattr(
            hlp,
            "_integrate_synchronous_wake_potential",
            lambda t_arr, z_int, dz_int, ez_int, **kw: orig_integ(
                t_arr, z_int, dz_int, -ez_int, **kw
            ),
        )
        res_flipped_sign = hlp.run_2d_karetnikova_beam_wake(
            spec,
            n_periods=4,
            u0_m_s=u0,
            n_feat=4,
            use_graded_mesh=True,
            quick=True,
        )
        with pytest.raises(AssertionError):
            _assert_pierce_k_agreement(
                res_flipped_sign["k_on_axis_beam_ohm"], l1["k_m1_on_axis_ohm"], rel_tol=0.08
            )

    # Physical mutation check on live FIT run: detuning beam velocity u0 -> 0.95 u0 at N = 8 shifts peak frequency downward
    res_detuned = hlp.run_2d_karetnikova_beam_wake(
        spec,
        n_periods=8,
        u0_m_s=0.95 * u0,
        n_feat=4,
        use_graded_mesh=True,
        quick=True,
    )
    df_shift = res_detuned["f_pk_ghz"] - res_n8["f_pk_ghz"]
    df_expected = (
        hlp.analytic_detuned_frequency_shift_hz(
            f_sync_hz=res_n8["f_pk_ghz"] * 1e9,
            u0_m_s=u0,
            delta_u0_m_s=-0.05 * u0,
            vg_m_s=l1["vg_bwd_m_s"],
        )
        * 1e-9
    )
    _assert_detuned_frequency_shift(df_shift, df_expected, rel_tol=0.12)
