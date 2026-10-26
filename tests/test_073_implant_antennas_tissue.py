# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Tests for Tutorial 4.9: Electric vs magnetic implant antennas in lossy tissue."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = (
    REPO_ROOT
    / "tutorials"
    / "04_radiation_and_antennas"
    / "04-09_implant_antennas_tissue"
)
SCRIPT_PATH = TUTORIAL_DIR / "implant_antennas_tissue.py"
NOTEBOOK_PATH = TUTORIAL_DIR / "implant_antennas_tissue.ipynb"
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as hlp  # noqa: E402


# ---------------------------------------------------------------------------
# Assertion helpers (used in both positive tests and mutation checks)
# ---------------------------------------------------------------------------
def _assert_near_zone_asymptote(
    nz_metrics: dict[str, np.ndarray],
    min_ratio: float = 300.0,
    max_asymptote_rel_diff: float = 0.10,
) -> None:
    """Assert that small-|k|a near-zone loss ratio matches 1/(|k|a)^2 scaling."""
    ratio = float(nz_metrics["excess_loss_ratio"])
    inv_ka2 = float(nz_metrics["asymptotic_inv_ka2"])
    assert ratio >= min_ratio, f"Near-zone excess loss ratio {ratio:.2f} < {min_ratio:.2f}"
    rel_diff = abs(ratio - inv_ka2) / inv_ka2
    assert rel_diff <= max_asymptote_rel_diff, (
        f"Near-zone ratio {ratio:.2f} deviates from 1/(|k|a)^2 = {inv_ka2:.2f} by {rel_diff:.3f}"
    )


def _assert_passive_muscle_fit(
    eps_fit: np.ndarray,
    eps_target: np.ndarray,
    max_rms_err: float = 0.01,
) -> None:
    """Assert that a fitted tissue dielectric spectrum is passive (Im < 0) and accurate."""
    assert np.all(np.imag(eps_fit) < 0.0), "Passivity violated: Im(eps) must be strictly negative in exp(+jwt)"
    rms_err = float(np.sqrt(np.mean(np.abs((eps_fit - eps_target) / eps_target) ** 2)))
    assert rms_err <= max_rms_err, f"Muscle Debye fit RMS error {rms_err:.4f} > {max_rms_err:.4f}"


def _assert_capsule_cavity_boost(
    power_boost_tm: float,
    power_boost_te: float,
) -> None:
    """Assert that a low-permittivity capsule boosts TM exterior power (> 1.8x) while leaving TE ~ 1."""
    assert 1.85 <= power_boost_tm <= 2.35, (
        f"TM capsule cavity power boost {power_boost_tm:.3f} outside [1.85, 2.35]"
    )
    assert abs(power_boost_te - 1.0) <= 0.02, (
        f"TE quasi-static cavity factor {power_boost_te:.4f} deviates from 1.0"
    )


def _assert_poynting_closure(
    p1: float,
    p2: float,
    p_loss_shell: float,
    max_rel_err: float = 0.025,
) -> None:
    """Assert that discrete FIT shell flux difference P1 - P2 equals integrated ohmic loss."""
    dp = p1 - p2
    assert dp > 0.0, "Flux must decrease across lossy tissue shell (P1 > P2)"
    rel_err = abs(dp - p_loss_shell) / abs(dp)
    assert rel_err <= max_rel_err, (
        f"Discrete Poynting closure error {rel_err:.4f} exceeds {max_rel_err:.4f}"
    )


def _assert_balanced_loop_low_loss(
    loss_over_out: float,
    gap_hemi_frac: float,
    peak_sar: float,
    max_loss_over_out: float = 1.0,
    max_hemi_asymmetry: float = 0.01,
    max_peak_sar: float = 50.0,
) -> None:
    """Assert that an ideal balanced magnetic loop has low loss, x-symmetry, and low peak SAR."""
    assert loss_over_out <= max_loss_over_out, (
        f"Loop near-zone loss/out = {loss_over_out:.4f} exceeds balanced-loop bound {max_loss_over_out:.4f}"
    )
    asym = abs(gap_hemi_frac - 0.5)
    assert asym <= max_hemi_asymmetry, (
        f"Hemisphere loss asymmetry |{gap_hemi_frac:.4f} - 0.5| = {asym:.4f} > {max_hemi_asymmetry:.4f}"
    )
    assert peak_sar <= max_peak_sar, (
        f"Peak tissue SAR {peak_sar:.1f} W/kg/mW exceeds balanced-loop bound {max_peak_sar:.1f}"
    )


def _assert_liska_fig2_digitised_bounds(
    r_over_a: np.ndarray,
    eta_403_db: np.ndarray,
    eta_2450_db: np.ndarray,
    tol_db: float = 0.5,
) -> None:
    """Transcription check: assert that Fig. 2 digitised bounds match readings within +-0.5 dB and crossover at r/a = 0.32."""
    r_ref = np.array([0.10, 0.20, 0.32, 0.50, 0.60, 0.75, 1.00], dtype=float)
    b403_ref = np.array([-29.8, -29.8, -29.8, -24.8, -23.4, -22.5, -21.9], dtype=float)
    b2450_ref = np.array([-25.1, -24.9, -24.8, -24.8, -24.8, -24.8, -24.8], dtype=float)

    r_arr = np.asarray(r_over_a, dtype=float)
    e403 = np.asarray(eta_403_db, dtype=float)
    e2450 = np.asarray(eta_2450_db, dtype=float)
    assert r_arr.shape == r_ref.shape and np.allclose(r_arr, r_ref, atol=1e-6)
    assert np.all(np.abs(e403 - b403_ref) <= tol_db), (
        f"403 MHz bound {e403} deviates from digitised Fig. 2 {b403_ref} by > {tol_db} dB"
    )
    assert np.all(np.abs(e2450 - b2450_ref) <= tol_db), (
        f"2.45 GHz bound {e2450} deviates from digitised Fig. 2 {b2450_ref} by > {tol_db} dB"
    )
    assert float(np.ptp(e2450)) <= 0.6, "2.45 GHz bound must be nearly flat across r/a in [0.1, 1.0]"

    # Read TM1m -> TE1m crossover directly from the slope kink in the digitised 403 MHz array
    slopes = np.diff(e403) / np.diff(r_arr)
    assert np.all(np.abs(slopes[:2]) <= 1.0), "403 MHz bound must be flat up to r/a = 0.32"
    kink_indices = np.nonzero(slopes > 5.0)[0]
    assert kink_indices.size > 0, "No TM->TE slope jump found in 403 MHz bound array"
    crossover_read = float(r_arr[int(kink_indices[0])])
    assert np.isclose(crossover_read, 0.32, atol=1e-6), (
        f"Crossover read from digitised array is {crossover_read}, expected 0.32"
    )


def _validate_summary_table_against_json(
    script_text: str,
    j1: dict,
    j2: dict,
    j3: dict,
    j4: dict,
    j5: dict,
) -> None:
    """Verify that every quantitative value in the notebook summary table matches data/*.json."""
    expected_tokens = [
        f"{j1['seawater_1khz_inv_ka2']:.2f}",
        f"{j1['seawater_1khz_ratio_full']:.2f}",
        f"{j1['muscle_debye_rms_fit_error']*100:.2f}%",
        f"{j1['muscle_403mhz_debye_real']:.2f} - {j1['muscle_403mhz_debye_imag']:.2f}j",
        f"{j1['muscle_2450mhz_debye_real']:.2f} - {j1['muscle_2450mhz_debye_imag']:.2f}j",
        f"{j1['nz_excess_ratio_403mhz_ref']:.2f}",
        f"{j1['nz_excess_ratio_2450mhz_ref']:.2f}",
        f"{j2['poynting_closure_rel_err_tm_403mhz']*100:.2f}%",
        f"{j2['cavity_boost_tm_air_2450mhz_fit']:.3f}",
        f"{j2['graded_cavity_boost_tm_air_2450mhz_fit']:.3f}",
        f"{j2['cavity_boost_tm_air_2450mhz_full_theory']:.3f}",
        f"{j2['cavity_boost_tm_air_2450mhz_qs_theory']:.3f}",
        f"{j2['cavity_boost_tm_air_2450mhz_rel_err_vs_full']*100:.2f}%",
        f"{j2['cavity_boost_te_air_403mhz_fit']:.4f}",
        f"{j3['eta_ideal_loop_air_403mhz']*100:.2f}%",
        f"{j3['eta_gapped_loop_air_403mhz']*100:.2f}%",
        f"{j3['gap_contamination_fraction_403mhz']*100:.2f}%",
        f"{j3['eta_gapped_loop_cer_403mhz']*100:.2f}%",
        f"{j4['liska_403mhz_crossover_literature']:.2f}",
        f"{j4['fit_eta_nz_tm_0p5_403mhz']*100:.2f}%",
        f"{j4['fit_p_cu_over_p_out_403mhz_sweep'][0]:.1f}",
        f"{j4['fit_p_cu_over_p_out_403mhz_sweep'][1]:.2f}",
        f"{j4['ferrite_f_fmr_ghz']:.2f}",
        f"{j4['ferrite_lorentz_q']:.0f}",
        f"{j4['ferrite_pout_boost_2450mhz']:.2f}",
        f"{j4['eta_total_gain_ferrite_vs_air_403mhz']:.2f}",
        f"{j4['eta_total_gain_ferrite_vs_air_2450mhz']:.2f}",
        f"{j5['shell_loss_tm_2450mhz_richardson']:.4f}",
        f"{j5['shell_loss_tm_2450mhz_rel_err'][-1]*100:.2f}%",
    ]
    norm_text = script_text.replace("\\", "")
    for tok in expected_tokens:
        assert tok in norm_text, f"Quantitative Summary token {tok!r} missing from {SCRIPT_PATH.name}"


# ---------------------------------------------------------------------------
# Unit & integration tests with mandatory mutation checks
# ---------------------------------------------------------------------------
def test_zhou_pan_2023_closed_form_and_drop_cubic_mutation() -> None:
    """Verify Zhou & Pan (2023) Eqs. (19)-(20), (30) and assert that dropping 1/r^3 fails."""
    f_sea = 1.0e3
    eps_sea = 80.0 - 1j * 4.0 / (2.0 * np.pi * f_sea * hlp.EPS0)
    a_sea = 0.30
    b_sea = 100.0

    tm_a = hlp.zhou_pan_poynting_flux(a_sea, f_sea, eps_sea, mode="TM")
    tm_b = hlp.zhou_pan_poynting_flux(b_sea, f_sea, eps_sea, mode="TM")
    te_a = hlp.zhou_pan_poynting_flux(a_sea, f_sea, eps_sea, mode="TE")
    te_b = hlp.zhou_pan_poynting_flux(b_sea, f_sea, eps_sea, mode="TE")

    assert np.isclose(float(tm_a["kr_abs"]), 0.053315, rtol=1e-4)
    assert np.isclose(float(tm_a["t2a"]), 351.81, rtol=1e-4)
    ratio_full = float((te_b["S_r"] / te_a["S_r"]) / (tm_b["S_r"] / tm_a["S_r"]))
    assert np.isclose(ratio_full, 363.37, rtol=1e-3)

    nz_ok = hlp.near_zone_excess_loss_metrics(a_sea, f_sea, eps_sea, drop_cubic_term=False)
    _assert_near_zone_asymptote(nz_ok)

    # Mutation check: dropping the 1/r^3 quasi-static E term must fail the check
    nz_mut = hlp.near_zone_excess_loss_metrics(a_sea, f_sea, eps_sea, drop_cubic_term=True)
    with pytest.raises(AssertionError):
        _assert_near_zone_asymptote(nz_mut)


def test_gabriel_muscle_debye_pole_fit_and_sign_mutation() -> None:
    """Verify 3-pole Debye fit to Gabriel (1996) muscle and assert that sign/pole mutations fail."""
    f_grid = np.geomspace(150.0e6, 3.5e9, 40)
    eps_cc = hlp.gabriel_1996_muscle_cole_cole(f_grid)
    muscle = hlp.build_dispersive_muscle(n_debye=3)
    eps_deb = muscle.epsilon(f_grid)

    _assert_passive_muscle_fit(eps_deb, eps_cc)
    assert np.isclose( muscle.epsilon(hlp.F_MEDRADIO_HZ), 57.10 - 35.55j, rtol=5e-3 )
    assert np.isclose( muscle.epsilon(hlp.F_ISM_HZ), 52.73 - 12.75j, rtol=5e-3 )

    # Mutation check 1: flipping imaginary sign (non-passive gain medium) must fail
    with pytest.raises(AssertionError):
        _assert_passive_muscle_fit(np.conj(eps_deb), eps_cc)

    # Mutation check 2: scaling permittivity by 1.15 must fail accuracy threshold
    with pytest.raises(AssertionError):
        _assert_passive_muscle_fit(1.15 * eps_deb, eps_cc)


def test_capsule_cavity_factor_and_inversion_mutation() -> None:
    """Verify spherical capsule-interface factor (derived here) and assert inverted ratio fails."""
    eps_m = hlp.gabriel_1996_muscle_cole_cole(hlp.F_ISM_HZ)
    cav_tm = hlp.capsule_cavity_factor(hlp.F_ISM_HZ, 2.25e-3, 1.0, eps_m, mode="TM")
    cav_te = hlp.capsule_cavity_factor(hlp.F_MEDRADIO_HZ, 2.25e-3, 1.0, eps_m, mode="TE")

    _assert_capsule_cavity_boost(
        float(cav_tm["power_boost_qs"]),
        float(cav_te["power_boost_qs"]),
    )

    # Mutation check: inverting the cavity factor (2*eps_m + eps_cap)/(3*eps_m) must fail
    cav_tm_mut = hlp.capsule_cavity_factor(
        hlp.F_ISM_HZ, 2.25e-3, 1.0, eps_m, mode="TM", invert_ratio=True
    )
    with pytest.raises(AssertionError):
        _assert_capsule_cavity_boost(
            float(cav_tm_mut["power_boost_qs"]),
            float(cav_te["power_boost_qs"]),
        )


def test_fit_poynting_power_bookkeeping_and_shift_mutation() -> None:
    """Run a quick 3D FIT capsule simulation and verify discrete Poynting closure + code-path mutations."""
    muscle = hlp.build_dispersive_muscle(n_debye=3)
    dx = 0.80e-3
    r1 = 3.0 * dx
    r2 = 5.0 * dx
    a_cap = 2.0 * dx

    sim = hlp._run_single_shell_sim(
        muscle,
        N=24,
        dx_m=dx,
        a_cap_m=a_cap,
        r1_m=r1,
        r2_m=r2,
        mode="TM",
        eps_cap=1.0,
        record_dft_shell=True,
    )
    _assert_poynting_closure(
        float(sim["p1"][1]),
        float(sim["p2"][1]),
        float(sim["p_loss_box"][1]),
    )

    # Code-path mutation check 1: omitting the r_hat . n_hat = r_box / R obliquity factor in
    # zhou_pan_cube_flux inflates the cube surface flux by 17.1% and must fail a 5% check
    eps_403 = complex(muscle.epsilon(hlp.F_MEDRADIO_HZ))
    s1_tm_ok = hlp.zhou_pan_cube_flux(r1, hlp.F_MEDRADIO_HZ, eps_403, mode="TM", drop_obliquity=False)
    s1_tm_mut = hlp.zhou_pan_cube_flux(r1, hlp.F_MEDRADIO_HZ, eps_403, mode="TM", drop_obliquity=True)
    assert abs(s1_tm_mut - s1_tm_ok) / s1_tm_ok > 0.15

    # Code-path mutation check 2: using the unaligned spherical-inner-boundary integral
    # (use_box_inner=False -> sim["p_loss_sphere"]) instead of the cubic-shell integral
    # (use_box_inner=True -> sim["p_loss_box"]) must fail the [r1, r2] Poynting closure check
    with pytest.raises(AssertionError):
        _assert_poynting_closure(
            float(sim["p1"][1]),
            float(sim["p2"][1]),
            float(sim["p_loss_sphere"][1]),
        )
    with pytest.raises(AssertionError):
        _assert_poynting_closure(
            float(sim["p1"][1]),
            float(-sim["p2"][1]),
            float(sim["p_loss_box"][1]),
        )


def test_real_feeds_unbalanced_gap_mutation_and_bounds() -> None:
    """Verify real-feed gap contamination, Liska (2024) benchmarks, Soares (2023), and ferrite core."""
    j3 = json.loads((DATA_DIR / "task_03_real_feeds_and_sar.json").read_text())
    j4 = json.loads((DATA_DIR / "task_04_bounds_and_ferrite.json").read_text())

    # Positive check: ideal balanced 4-segment loop satisfies low-loss, x-symmetry, and low-SAR bounds
    _assert_balanced_loop_low_loss(
        j3["loss_over_out_ideal_loop_403mhz"],
        j3["gap_hemi_loss_fraction_ideal_loop_403mhz"],
        j3["peak_sar_ideal_loop_403mhz"],
    )

    # Mutation check: unbalancing the loop into a single-gap PEC wire loop on the +x arm creates
    # an E-field hotspot that breaks x-hemisphere symmetry (53.2% vs 50.0%) and spikes peak SAR 8.6x
    with pytest.raises(AssertionError):
        _assert_balanced_loop_low_loss(
            j3["loss_over_out_gapped_loop_403mhz"],
            j3["gap_hemi_loss_fraction_gapped_loop_403mhz"],
            j3["peak_sar_gapped_loop_403mhz"],
        )

    # Transcription check: verify Liska (2024) Fig. 2 digitised bounds (within +-0.5 dB) and crossover read from the digitised array
    liska = hlp.liska_2024_verifiable_benchmarks()
    _assert_liska_fig2_digitised_bounds(
        liska["r_over_a_pts_approx"],
        liska["eta_bound_403mhz_db_approx"],
        liska["eta_bound_2450mhz_db_approx"],
    )
    _assert_liska_fig2_digitised_bounds(
        np.asarray(j4["liska_r_over_a_pts_approx"]),
        np.asarray(j4["liska_403mhz_bound_db_approx"]),
        np.asarray(j4["liska_2450mhz_bound_db_approx"]),
    )
    assert np.isclose(float(liska["sigma_copper_s_m"]), 5.56e7, rtol=1e-4)

    # Mutation check on Liska Fig. 2 transcription check: shifting the crossover kink to r/a = 0.20 or
    # perturbing the 2.45 GHz curve by > 0.5 dB must fail _assert_liska_fig2_digitised_bounds
    e403_shifted_kink = np.array([-29.8, -29.8, -26.5, -24.8, -23.4, -22.5, -21.9])
    with pytest.raises(AssertionError):
        _assert_liska_fig2_digitised_bounds(
            liska["r_over_a_pts_approx"],
            e403_shifted_kink,
            liska["eta_bound_2450mhz_db_approx"],
        )

    assert float(j4["fit_cells_across_smallest_loop"]) >= 4.0 - 1e-6
    assert np.isclose(float(j4["fit_tm_dipole_r_over_a"]), 0.50, atol=1e-6)
    assert float(j4["fit_cpml_clearance_mm"]) > 2.0
    assert float(j4["ferrite_cpml_clearance_mm"]) > 2.0
    assert np.isclose(float(j4["ferrite_f_fmr_ghz"]), 1.15, atol=1e-6)
    assert float(j4["fit_eta_nz_te_ideal_403mhz_sweep"][1]) > 2.0 * float(j4["fit_eta_nz_tm_0p5_403mhz"])

    # Verify r^{-3} copper-to-output power ratio scaling at fixed wire radius
    p_cu_over_out = j4["fit_p_cu_over_p_out_403mhz_sweep"]
    assert np.isclose(p_cu_over_out[0] / p_cu_over_out[1], 8.0, rtol=0.02)
    assert np.isclose(p_cu_over_out[1] / p_cu_over_out[2], 3.375, rtol=0.02)

    # Verify Soares (2023) Table II ordering and Lorentz ferrite core behavior at 403 MHz and 2.45 GHz
    for row in j4["soares_2023_table_ii"]:
        assert row["eta_mag_pct"] > row["eta_elec_pct"]
        assert row["f_opt_mag_mhz"] < row["f_opt_elec_mhz"]
    ferrite = hlp.build_nizn_ferrite_material()
    assert np.real(ferrite.mu(hlp.F_MEDRADIO_HZ)) > 8.0
    assert np.real(ferrite.mu(hlp.F_ISM_HZ)) < 0.0
    assert j4["eta_total_gain_ferrite_vs_air_403mhz"] > 4.0
    assert j4["ferrite_pout_boost_2450mhz"] > 1.8
    assert j4["p_core_2450"] > 5.0 * j4["p_out_ferrite_core_2450mhz_w"]
    assert j4["eta_total_gain_ferrite_vs_air_2450mhz"] < 0.4


def test_quantitative_summary_table_matches_json_and_artifacts() -> None:
    """Verify that all JSON/NPZ/figure artifacts exist and the Quantitative Summary table matches JSON."""
    assert SCRIPT_PATH.exists()
    assert NOTEBOOK_PATH.exists()
    for fig_name in (
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00_setup_table.md",
        "01_muscle_dispersion_and_nz_loss_ratio.png",
        "02_capsule_interface_correction_and_shells.png",
        "03_real_feeds_sar_and_gap_contamination.png",
        "04_liska_bounds_soares_and_ferrite_core.png",
        "05_grid_convergence_and_soares_depth_trend.png",
        "implant_tm_vs_te_wavefronts.gif",
    ):
        p = FIG_DIR / fig_name
        assert p.exists() and p.stat().st_size > 0, f"Missing figure artifact {p}"

    j1 = json.loads((DATA_DIR / "task_01_near_zone_theory_and_muscle_fit.json").read_text())
    j2 = json.loads((DATA_DIR / "task_02_fit_shells_and_capsule_correction.json").read_text())
    j3 = json.loads((DATA_DIR / "task_03_real_feeds_and_sar.json").read_text())
    j4 = json.loads((DATA_DIR / "task_04_bounds_and_ferrite.json").read_text())
    j5 = json.loads((DATA_DIR / "task_05_convergence.json").read_text())

    script_text = SCRIPT_PATH.read_text(encoding="utf-8")
    _validate_summary_table_against_json(script_text, j1, j2, j3, j4, j5)

    # Mutation check: perturbing any JSON value must cause the Quantitative Summary validator to fail
    j2_mut = dict(j2, cavity_boost_tm_air_2450mhz_fit=9.876)
    with pytest.raises(AssertionError):
        _validate_summary_table_against_json(script_text, j1, j2_mut, j3, j4, j5)
