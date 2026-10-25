"""Tests for Tutorial 8.4: Smith-Purcell radiation and the dielectric laser accelerator grating."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = REPO_ROOT / "tutorials" / "08_vacuum_electronics" / "08-04_smith_purcell_dla"
SCRIPT_PATH = TUTORIAL_DIR / "smith_purcell_dla.py"
NOTEBOOK_PATH = TUTORIAL_DIR / "smith_purcell_dla.ipynb"
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as hlp  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers with explicit assertion functions (used for positive & mutation tests)
# ---------------------------------------------------------------------------


def _assert_sp_dispersion_accuracy(
    lam_sim_m: np.ndarray,
    lam_theory_m: np.ndarray,
    max_mean_rel_err: float = 0.035,
) -> None:
    """Assert that simulated Smith-Purcell wavelengths match theory and Doppler monotonicity."""
    rel_err = np.abs(lam_sim_m - lam_theory_m) / lam_theory_m
    mean_err = float(np.mean(rel_err))
    assert mean_err < max_mean_rel_err, (
        f"Mean Smith-Purcell wavelength error {mean_err:.4f} exceeds {max_mean_rel_err:.4f}"
    )
    # Forward emission (smaller theta) must have strictly shorter wavelength than backward emission
    assert float(lam_sim_m[0]) < float(lam_sim_m[-1]), (
        "Forward Smith-Purcell wavelength must be shorter than backward wavelength"
    )


def _assert_sp_flat_surface_null(
    power_flat: float,
    power_grating: float,
    max_ratio: float = 1.0e-3,
) -> None:
    """Assert that a flat surface (h_g = 0) emits negligible radiation compared to a grating."""
    assert power_grating > 0.0, "Grating radiated power must be positive"
    ratio = float(power_flat / power_grating)
    assert ratio < max_ratio, (
        f"Flat-surface radiated power ratio {ratio:.3e} exceeds null threshold {max_ratio:.3e}"
    )


def _assert_evanescent_decay_slope(
    h_values_m: np.ndarray,
    intensities: np.ndarray,
    alpha_theory_1_m: float,
    rel_tol: float = 0.04,
) -> None:
    """Assert that radiated intensity decays exponentially with h at rate alpha = 4*pi/(beta*gamma*lambda)."""
    assert np.all(np.diff(intensities) < 0.0), (
        "Smith-Purcell intensity must decrease monotonically with impact parameter h"
    )
    slope, _ = np.polyfit(h_values_m, np.log(intensities), 1)
    alpha_sim = float(-slope)
    rel_err = abs(alpha_sim - alpha_theory_1_m) / alpha_theory_1_m
    assert rel_err < rel_tol, (
        f"Fitted decay constant {alpha_sim:.3e} m^-1 differs from theory {alpha_theory_1_m:.3e} m^-1 "
        f"by {rel_err:.4f} (> {rel_tol:.4f})"
    )


def _assert_dla_synchronism_and_null(
    g_sync: float,
    g_detuned_low: float,
    g_detuned_high: float,
    g_flat: float,
) -> None:
    """Assert that DLA gradient peaks at synchronous period and vanishes for a flat surface."""
    assert g_sync > 0.25, f"Synchronous DLA factor G/E0 = {g_sync:.4f} is unexpectedly small"
    assert g_sync > 2.5 * g_detuned_low, (
        f"Synchronous G/E0 ({g_sync:.4f}) does not strongly exceed low-period detuned value ({g_detuned_low:.4f})"
    )
    assert g_sync > 2.5 * g_detuned_high, (
        f"Synchronous G/E0 ({g_sync:.4f}) does not strongly exceed high-period detuned value ({g_detuned_high:.4f})"
    )
    assert abs(g_flat) < 1.0e-4, (
        f"Flat-surface DLA factor G/E0 = {g_flat:.3e} violates Lawson-Woodward null (< 1e-4)"
    )


# ---------------------------------------------------------------------------
# 1. Analytical & Literature Benchmark Tests
# ---------------------------------------------------------------------------


def test_smith_purcell_1953_dispersion_and_kinematics() -> None:
    """Verify Smith & Purcell (1953) Eq. (1) lambda = (d/n)(1/beta - cos(theta)) and kinematics."""
    # Smith & Purcell, Phys. Rev. 92, 1069 (1953): d = 1.67 um, E_k = 319-340 keV
    kin_319 = hlp.kinetic_energy_to_beta_gamma(319.0e3)
    kin_340 = hlp.kinetic_energy_to_beta_gamma(340.0e3)
    b_319 = float(kin_319["beta"])
    b_340, g_340 = float(kin_340["beta"]), float(kin_340["gamma"])
    assert 0.785 < b_319 < 0.791
    assert 0.797 < b_340 < 0.802
    assert np.isclose(1.0 / np.sqrt(1.0 - b_340**2), g_340, rtol=1e-12)

    d_m = 1.67e-6
    lam_20 = float(hlp.smith_purcell_wavelength(np.radians(20.0), b_340, period_m=d_m, order=1)["wavelength_m"])
    lam_90 = float(hlp.smith_purcell_wavelength(np.radians(90.0), 0.80, period_m=d_m, order=1)["wavelength_m"])
    lam_90_n2 = float(hlp.smith_purcell_wavelength(np.radians(90.0), 0.80, period_m=d_m, order=2)["wavelength_m"])

    # Forward blue-green observation (~521 nm at theta = 20 deg) and broadside d/beta
    assert 510.0e-9 < lam_20 < 535.0e-9
    assert np.isclose(lam_90, d_m / 0.80, rtol=1e-12)
    assert np.isclose(lam_90_n2, 0.5 * lam_90, rtol=1e-12)

    # Invalid inputs must raise ValueError
    with pytest.raises(ValueError):
        hlp.smith_purcell_wavelength(np.radians(90.0), 1.05, period_m=d_m)
    with pytest.raises(ValueError):
        hlp.smith_purcell_wavelength(np.radians(90.0), 0.80, period_m=d_m, order=0)


def test_evanescent_decay_analytical_scaling() -> None:
    """Verify transverse evanescent attenuation exp(-4*pi*h/(beta*gamma*lambda)) and relativistic dilation."""
    h_arr = np.linspace(0.1e-6, 0.6e-6, 6)
    lam_m = 2.0e-6
    ev_07 = hlp.smith_purcell_evanescent_decay(h_arr, beta=0.70, wavelength_m=lam_m)
    ev_09 = hlp.smith_purcell_evanescent_decay(h_arr, beta=0.90, wavelength_m=lam_m)

    # Power decay constant alpha_h = 2 * kappa_perp = 4*pi / (beta*gamma*lambda)
    assert np.isclose(ev_07["alpha_power_1_m"], 2.0 * ev_07["kappa_field_1_m"], rtol=1e-12)
    # Higher beta*gamma must have smaller attenuation constant (longer evanescent reach)
    assert ev_09["alpha_power_1_m"] < ev_07["alpha_power_1_m"]
    assert ev_09["decay_length_m"] > ev_07["decay_length_m"]
    assert np.allclose(ev_07["power_factor"], ev_07["field_factor"] ** 2, rtol=1e-12)


def test_dla_synchronism_and_literature_benchmarks() -> None:
    """Verify DLA synchronism lambda_p = |m|*beta*lambda_0 and numbers from Peralta (2013) & Sapra (2020)."""
    bench = hlp.literature_benchmarks()

    # Peralta et al., Nature 503, 91 (2013): 60 MeV, lam0 = 800 nm -> lam_p = 800 nm, G = 309.8 MeV/m
    p13 = bench["peralta_2013"]
    assert np.isclose(p13["grating_period_nm"], 800.0, rtol=1e-4)
    assert np.isclose(p13["max_gradient_400nm_mev_m"], 309.8, rtol=1e-4)
    assert np.isclose(p13["max_gradient_800nm_mev_m"], 152.0, rtol=1e-4)
    assert np.isclose(p13["pulse_duration_fwhm_ps"], 1.24, rtol=1e-4)
    assert np.isclose(p13["max_pulse_energy_uj"], 319.0, rtol=1e-4)
    assert np.isclose(p13["damage_fluence_j_cm2"], 0.85, rtol=1e-4)

    # Non-circular Peralta et al. (2013) gap-ratio and forward-predicted gradient check
    comp = hlp.peralta_2013_gradient_comparison(0.3415900475235852, 0.1722993856756825)
    assert np.isclose(comp["gap_ratio_sim_rounded"], 0.342 / 0.172, rtol=1e-6)
    assert abs(comp["gap_ratio_sim"] - comp["gap_ratio_exp"]) <= comp["gap_ratio_exp_err"]
    assert 1.40 < comp["e0_sio2_319uj_gv_m"] < 1.55
    assert 480.0 < comp["g_pred_400_sio2_319uj_mev_m"] < 520.0
    # Mutation check: swapping 400 nm and 800 nm structure factors inverts the gap ratio (< 1)
    comp_mut = hlp.peralta_2013_gradient_comparison(0.1722993856756825, 0.3415900475235852)
    assert abs(comp_mut["gap_ratio_sim"] - comp_mut["gap_ratio_exp"]) > 3.0 * comp_mut["gap_ratio_exp_err"]

    # Sapra et al., Science 367, 79 (2020): 83.4 keV (beta=0.5110), lam0 = 1984 nm -> lam_p = 1014 nm
    s20 = bench["sapra_2020"]
    assert np.isclose(s20["beta"], 0.5110, rtol=1e-3)
    assert np.isclose(s20["synchronous_period_nm"], 1013.8, rtol=1e-3)
    assert np.isclose(s20["max_gradient_mev_m"], 30.5, rtol=1e-4)

    # Yang et al., Nature 613, 42 (2023): SOI photonic crystal flatband ~100x enhancement
    y23 = bench["yang_2023"]
    assert y23["enhancement_factor_measured"] == pytest.approx(100.0)
    assert y23["lattice_period_nm"] == pytest.approx(430.0)

    # Synchronism phase advance per period must be 2*pi*m at lam_p = m*beta*lam0
    sync_info = hlp.dla_synchronous_period(beta=0.75, wavelength_m=800.0e-9, order=1, n_periods=10)
    assert np.isclose(sync_info["period_sync_m"], 600.0e-9, rtol=1e-12)
    assert np.isclose(sync_info["phase_advance_rad"], 2.0 * np.pi, rtol=1e-12)
    assert np.isclose(sync_info["dephasing_factor"], 1.0, rtol=1e-12)

    # Raman-Nath phase-grating model must vanish at h_g = 0, DC = 0, and DC = 1
    assert float(hlp.dla_phase_grating_model(0.0, duty_cycle=0.50)["c1_amplitude"]) == pytest.approx(0.0, abs=1e-15)
    assert float(hlp.dla_phase_grating_model(680e-9, duty_cycle=0.0)["c1_amplitude"]) == pytest.approx(0.0, abs=1e-15)
    assert float(hlp.dla_phase_grating_model(680e-9, duty_cycle=1.0)["c1_amplitude"]) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# 2. Live 3D FIT Simulation Tests & Mutation Checks
# ---------------------------------------------------------------------------


def test_fit3d_smith_purcell_wavelength_vs_angle_and_flat_mutation() -> None:
    """Verify 3D FIT Smith-Purcell wavelength vs angle on PEC & SiO2 gratings and flat-surface null."""
    pec = hlp.run_single_smith_purcell_case(
        material="pec",
        beta=0.80,
        period_m=1.67e-6,
        n_periods=10,
        depth_m=0.40e-6,
        impact_m=0.25e-6,
        dx_m=0.050e-6,
        dz_m=0.050e-6,
        record_frames=0,
    )
    diel = hlp.run_single_smith_purcell_case(
        material=hlp.DEFAULT_EPS_SIO2,
        beta=0.80,
        period_m=1.67e-6,
        n_periods=10,
        depth_m=0.40e-6,
        impact_m=0.25e-6,
        dx_m=0.050e-6,
        dz_m=0.050e-6,
        record_frames=0,
    )
    flat = hlp.run_single_smith_purcell_case(
        material="pec",
        beta=0.80,
        period_m=1.67e-6,
        n_periods=10,
        depth_m=0.0,
        impact_m=0.25e-6,
        dx_m=0.050e-6,
        dz_m=0.050e-6,
        record_frames=0,
    )

    _assert_sp_dispersion_accuracy(pec["lambda_sim_m"], pec["lambda_theory_m"], max_mean_rel_err=0.035)
    _assert_sp_dispersion_accuracy(diel["lambda_sim_m"], diel["lambda_theory_m"], max_mean_rel_err=0.035)
    _assert_sp_flat_surface_null(
        flat["total_radiated_metric"], pec["total_radiated_metric"], max_ratio=1.0e-3
    )

    # Mutation check 1: reversing the angle order (breaking Doppler sign) must fail
    with pytest.raises(AssertionError):
        _assert_sp_dispersion_accuracy(pec["lambda_sim_m"][::-1], pec["lambda_theory_m"], max_mean_rel_err=0.035)

    # Mutation check 2: 15% wavelength bias must fail the tolerance check
    with pytest.raises(AssertionError):
        _assert_sp_dispersion_accuracy(1.15 * pec["lambda_sim_m"], pec["lambda_theory_m"], max_mean_rel_err=0.035)

    # Mutation check 3: passing grating radiated power as the flat surface must fail the null check
    with pytest.raises(AssertionError):
        _assert_sp_flat_surface_null(pec["total_radiated_metric"], pec["total_radiated_metric"], max_ratio=1.0e-3)


def test_fit3d_smith_purcell_impact_parameter_decay() -> None:
    """Verify 3D FIT impact-parameter evanescent decay constant alpha_h = 4*pi/(beta*gamma*lambda)."""
    h_vals = np.array([0.20e-6, 0.35e-6, 0.50e-6])
    beta = 0.80
    lam_90 = 1.67e-6 / beta
    alpha_th = float(hlp.smith_purcell_evanescent_decay(h_vals, beta=beta, wavelength_m=lam_90)["alpha_power_1_m"])

    lams_narrow = np.linspace(1.50e-6, 2.60e-6, 45)
    intens = []
    for hv in h_vals:
        res = hlp.run_single_smith_purcell_case(
            material="pec",
            beta=beta,
            period_m=1.67e-6,
            n_periods=8,
            depth_m=0.40e-6,
            impact_m=float(hv),
            dx_m=0.050e-6,
            dz_m=0.050e-6,
            wavelengths_m=lams_narrow,
            thetas_deg=np.array([90.0]),
            record_frames=0,
        )
        intens.append(float(res["broadside_intensity"]))
    intens_arr = np.asarray(intens)

    _assert_evanescent_decay_slope(h_vals, intens_arr, alpha_th, rel_tol=0.04)

    # Mutation check: non-decaying or inverted intensity profile must fail
    with pytest.raises(AssertionError):
        _assert_evanescent_decay_slope(h_vals, intens_arr[::-1], alpha_th, rel_tol=0.04)
    with pytest.raises(AssertionError):
        _assert_evanescent_decay_slope(h_vals, intens_arr**0.5, alpha_th, rel_tol=0.04)


def test_fit3d_dla_synchronism_peak_and_flat_mutation() -> None:
    """Verify 3D FIT DLA synchronism peak at lambda_p = beta*lambda_0 and flat-surface null (h_g = 0)."""
    sync = hlp.run_single_dla_unit_cell(
        period_m=0.80e-6,
        depth_m=0.68e-6,
        duty_cycle=0.50,
        beta=0.999964,
        n_periods_eval=8,
        dx_m=0.032e-6,
    )
    det_low = hlp.run_single_dla_unit_cell(
        period_m=0.62e-6,
        depth_m=0.68e-6,
        duty_cycle=0.50,
        beta=0.999964,
        n_periods_eval=8,
        dx_m=0.032e-6,
    )
    det_high = hlp.run_single_dla_unit_cell(
        period_m=0.98e-6,
        depth_m=0.68e-6,
        duty_cycle=0.50,
        beta=0.999964,
        n_periods_eval=8,
        dx_m=0.032e-6,
    )
    flat = hlp.run_single_dla_unit_cell(
        period_m=0.80e-6,
        depth_m=0.0,
        duty_cycle=0.50,
        beta=0.999964,
        n_periods_eval=8,
        dx_m=0.032e-6,
    )

    _assert_dla_synchronism_and_null(
        g_sync=sync["g_over_e0_multiperiod"],
        g_detuned_low=det_low["g_over_e0_multiperiod"],
        g_detuned_high=det_high["g_over_e0_multiperiod"],
        g_flat=flat["g_over_e0_cell"],
    )

    # Mutation check 1: swapping synchronous and detuned gradients must fail
    with pytest.raises(AssertionError):
        _assert_dla_synchronism_and_null(
            g_sync=det_low["g_over_e0_multiperiod"],
            g_detuned_low=sync["g_over_e0_multiperiod"],
            g_detuned_high=det_high["g_over_e0_multiperiod"],
            g_flat=flat["g_over_e0_cell"],
        )

    # Mutation check 2: non-zero flat-surface gradient must fail Lawson-Woodward null assertion
    with pytest.raises(AssertionError):
        _assert_dla_synchronism_and_null(
            g_sync=sync["g_over_e0_multiperiod"],
            g_detuned_low=det_low["g_over_e0_multiperiod"],
            g_detuned_high=det_high["g_over_e0_multiperiod"],
            g_flat=0.05,
        )


def test_graded_mesh_vs_uniform_baseline_and_mutation() -> None:
    """Verify locally refined graded mesh interfaces, cell reduction vs uniform Fine baseline, and live DLA agreement."""
    x_sp = hlp.build_sp_graded_x_nodes(depth_m=0.40e-6, impact_m=0.25e-6)
    for iface in (-1.00e-6, -0.40e-6, 0.0, 0.25e-6, 3.40e-6):
        assert np.min(np.abs(x_sp - iface)) < 1.0e-15, f"SP graded axis missed interface x = {iface}"
    dx_sp = np.diff(x_sp)
    assert float(np.min(dx_sp)) < 32.0e-9
    assert float(np.max(dx_sp)) < 70.0e-9
    assert len(dx_sp) < 88  # fewer transverse cells than 50 nm uniform Fine baseline (88 cells)

    x_dla = hlp.build_dla_graded_x_nodes(gap_half_m=0.20e-6, depth_m=0.68e-6)
    for iface in (-2.20e-6, -0.88e-6, -0.20e-6, 0.20e-6, 0.88e-6, 2.20e-6):
        assert np.min(np.abs(x_dla - iface)) < 1.0e-15, f"DLA graded axis missed interface x = {iface}"
    dx_dla = np.diff(x_dla)
    assert float(np.min(dx_dla)) < 19.0e-9
    assert len(dx_dla) < 220  # fewer transverse cells than 20 nm uniform Fine baseline (220 cells)

    # Live 3D FIT comparison on DLA unit cell: Graded vs Uniform Fine baseline
    res_graded = hlp.run_single_dla_unit_cell(
        period_m=0.80e-6,
        depth_m=0.68e-6,
        duty_cycle=0.50,
        beta=0.999964,
        n_periods_eval=8,
        dx_m=0.020e-6,
        graded=True,
    )
    res_unif = hlp.run_single_dla_unit_cell(
        period_m=0.80e-6,
        depth_m=0.68e-6,
        duty_cycle=0.50,
        beta=0.999964,
        n_periods_eval=8,
        dx_m=0.020e-6,
        graded=False,
    )
    res_flat_graded = hlp.run_single_dla_unit_cell(
        period_m=0.80e-6,
        depth_m=0.0,
        duty_cycle=0.50,
        beta=0.999964,
        n_periods_eval=8,
        dx_m=0.020e-6,
        graded=True,
    )
    assert res_graded["nx"] * res_graded["nz"] < res_unif["nx"] * res_unif["nz"]
    rel_diff = abs(res_graded["g_over_e0_cell"] - res_unif["g_over_e0_cell"]) / res_unif["g_over_e0_cell"]
    assert rel_diff < 0.015, f"Graded vs uniform Fine DLA structure factor differed by {rel_diff:.4f}"
    assert res_flat_graded["g_over_e0_cell"] < 1.0e-4

    # Mutation check: asserting that the flat-surface graded result equals the toothed grating result must fail
    with pytest.raises(AssertionError):
        assert abs(res_flat_graded["g_over_e0_cell"] - res_unif["g_over_e0_cell"]) / res_unif["g_over_e0_cell"] < 0.015


# ---------------------------------------------------------------------------
# 3. Precomputed Data, Figures, GIFs (< 5 MB), and CLI Verification
# ---------------------------------------------------------------------------


def test_precomputed_data_and_figures_integrity() -> None:
    """Verify that all tutorial data/*.{npz,json}, figures/*.png, and GIFs (< 5 MB) exist and pass checks."""
    assert SCRIPT_PATH.exists(), f"Missing tutorial script {SCRIPT_PATH}"
    assert NOTEBOOK_PATH.exists(), f"Missing generated notebook {NOTEBOOK_PATH}"

    for task_name in ("smith_purcell", "impact_decay", "dla_grating", "convergence"):
        npz_path = DATA_DIR / f"task_{task_name}.npz"
        json_path = DATA_DIR / f"task_{task_name}.json"
        assert npz_path.exists(), f"Missing {npz_path}"
        assert json_path.exists(), f"Missing {json_path}"

    sp_json = json.loads((DATA_DIR / "task_smith_purcell.json").read_text())
    imp_json = json.loads((DATA_DIR / "task_impact_decay.json").read_text())
    dla_json = json.loads((DATA_DIR / "task_dla_grating.json").read_text())
    conv_json = json.loads((DATA_DIR / "task_convergence.json").read_text())

    assert sp_json["pec_lambda_mean_rel_err"] < 0.03
    assert sp_json["diel_lambda_mean_rel_err"] < 0.03
    assert sp_json["flat_to_pec_power_ratio"] < 1.0e-4
    assert all(err < 0.02 for err in imp_json["alpha_rel_err"])
    assert abs(dla_json["fitted_peak_period_b100_nm"] - 800.0) < 10.0
    assert abs(dla_json["fitted_peak_period_b075_nm"] - 600.0) < 10.0
    assert dla_json["g_over_e0_flat_mutation"] < 1.0e-6
    assert conv_json["dla_ppw_sio2"][-1] >= 20.0
    assert conv_json["sp_graded_cells"] < conv_json["sp_cells"][-1]
    assert conv_json["sp_graded_lambda_mean_rel_err"] <= conv_json["sp_lambda_mean_rel_err"][-1]
    assert conv_json["dla_graded_cells"] < conv_json["dla_cells"][-1]
    assert abs(conv_json["dla_graded_g_over_e0"] - conv_json["dla_g_over_e0"][-1]) / conv_json["dla_g_over_e0"][-1] < 0.01

    setup_table_md = FIG_DIR / "00_setup_table.md"
    assert setup_table_md.exists() and setup_table_md.stat().st_size > 200, f"Missing {setup_table_md}"

    expected_pngs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00c_setup_dla_geometry.png",
        "01_setup_schematic_and_mesh.png",
        "02_sp_field_maps.png",
        "03_sp_dispersion_and_angular_spectrum.png",
        "04_sp_impact_parameter_decay.png",
        "05_dla_synchronism_and_nearfield.png",
        "06_dla_depth_duty_and_peralta2013.png",
        "07_resolution_convergence.png",
    ]
    for fname in expected_pngs:
        fpath = FIG_DIR / fname
        assert fpath.exists() and fpath.stat().st_size > 10_000, f"Missing or empty figure {fpath}"

    for gif_name in ("smith_purcell_pec_vs_dielectric.gif", "dla_synchronous_acceleration.gif"):
        gpath = FIG_DIR / gif_name
        assert gpath.exists(), f"Missing animation {gpath}"
        size_mb = gpath.stat().st_size / (1024.0 * 1024.0)
        assert 0.02 < size_mb < 5.0, f"GIF {gif_name} size {size_mb:.2f} MB violates < 5 MB rule"


def test_cli_list_tasks() -> None:
    """Verify that the tutorial script --list-tasks CLI flag lists all 4 tasks."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--list-tasks"],
        capture_output=True,
        text=True,
        check=True,
    )
    for task_name in ("smith_purcell", "impact_decay", "dla_grating", "convergence"):
        assert task_name in proc.stdout
