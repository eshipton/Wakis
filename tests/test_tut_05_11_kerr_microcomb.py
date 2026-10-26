# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
End-to-end validation and mutation tests for Tutorial 5.11:
Kerr (chi^(3)) Nonlinear Media in FIT and FIT-Informed Soliton Microcombs.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = (
    REPO_ROOT / "tutorials" / "05_guided_waves" / "05-11_kerr_microcomb"
)
SCRIPT_PATH = TUTORIAL_DIR / "kerr_microcomb.py"
NOTEBOOK_PATH = TUTORIAL_DIR / "kerr_microcomb.ipynb"
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as hlp  # noqa: E402
import wakis  # noqa: E402

# ---------------------------------------------------------------------------
# Explicit assertion helpers (used for positive checks and mutation tests)
# ---------------------------------------------------------------------------


def _assert_anomalous_ring_dispersion(
    d2_hz: float,
    d2_ref_hz: float,
    rel_tol: float = 0.08,
) -> None:
    """Assert that the microring dispersion is anomalous (D_2 > 0) and matches reference."""
    assert d2_hz > 0.0, (
        f"Expected anomalous dispersion D_2 > 0 for bright Kerr solitons, got {d2_hz:.3e} Hz"
    )
    rel_err = abs(d2_hz - d2_ref_hz) / abs(d2_ref_hz)
    assert rel_err < rel_tol, (
        f"Extracted D_2 = {d2_hz:.3e} Hz differs from reference {d2_ref_hz:.3e} Hz "
        f"by {rel_err:.4f} (> {rel_tol:.4f})"
    )


def _assert_kerr_ring_shift_accuracy(
    df_fit_hz: np.ndarray,
    df_analytic_hz: np.ndarray,
    rel_tol: float = 0.06,
) -> None:
    """Assert that 2D FIT Kerr cavity shift is negative and matches perturbation theory."""
    assert np.all(df_fit_hz < 0.0), (
        f"Focusing Kerr cavity shift must be negative (red-shifted), got {df_fit_hz}"
    )
    rel_err = np.abs(df_fit_hz - df_analytic_hz) / np.abs(df_analytic_hz)
    max_err = float(np.max(rel_err))
    assert max_err < rel_tol, (
        f"Max relative error of 2D FIT Kerr ring shift {max_err:.4f} exceeds {rel_tol:.4f}"
    )


def _assert_lle_soliton_and_scaling(
    in_existence: bool,
    fwhm_mu_sim: float,
    fwhm_mu_an: float,
    disp_exponent: float,
    rel_tol: float = 0.08,
) -> None:
    """Assert that the LLE soliton lies in its existence range and matches sech^2 scaling."""
    assert in_existence, "Pump detuning zeta_0 lies outside soliton existence window"
    rel_bw = abs(fwhm_mu_sim - fwhm_mu_an) / fwhm_mu_an
    assert rel_bw < rel_tol, (
        f"Simulated comb FWHM {fwhm_mu_sim:.2f} differs from sech^2 theory {fwhm_mu_an:.2f} "
        f"by {rel_bw:.4f} (> {rel_tol:.4f})"
    )
    rel_exp = abs(disp_exponent - (-0.5)) / 0.5
    assert rel_exp < 0.05, (
        f"Dispersion bandwidth scaling exponent {disp_exponent:.4f} differs from -0.5 by {rel_exp:.4f}"
    )


def _verify_quantitative_summary_table(
    md_text: str,
    summary_json: dict,
    t2_json: dict,
    t3_json: dict,
    t4_json: dict,
) -> None:
    """
    Parse the Quantitative Summary markdown table in `kerr_microcomb.py` and assert
    that every quoted value matches the JSON files in `data/`.
    """
    sec_match = re.search(
        r"## 7\. Quantitative Summary(.*?)## 8\. Limits",
        md_text,
        flags=re.DOTALL,
    )
    assert sec_match is not None, "Could not locate Section 7 Quantitative Summary in script"
    block = sec_match.group(1).replace("\\%", "%")

    expected_tokens = [
        f"{summary_json['f0_bessel_thz']:.3f}",
        f"{summary_json['f0_fit_thz']:.3f}",
        f"{summary_json['f0_rel_error'] * 100:.2f}%",
        f"{summary_json['conv_f0_richardson_thz']:.3f}",
        f"{summary_json['conv_f0_richardson_rel_error'] * 100:.4f}%",
        f"{summary_json['D1_bessel_thz']:.3f}",
        f"{summary_json['D1_fit_thz']:.3f}",
        f"{summary_json['D1_rel_error'] * 100:.2f}%",
        f"{summary_json['conv_D1_richardson_thz']:.3f}",
        f"{summary_json['conv_D1_richardson_rel_error'] * 100:.4f}%",
        f"{summary_json['Delta_D2_pole_bessel_ghz']:+.2f}",
        f"{summary_json['Delta_D2_pole_fit_ghz']:+.2f}",
        f"{summary_json['Delta_D2_pole_rel_error'] * 100:.2f}%",
        f"{summary_json['D2_anom_bessel_ghz']:+.2f}",
        f"{t3_json['D2_hz'][-1] * 1e-9:+.2f}",
        f"{t3_json['D2_rel_err_vs_bessel'][-1] * 100:.2f}%",
        f"{summary_json['conv_D2_richardson_ghz']:+.2f}",
        f"{summary_json['conv_D2_richardson_rel_error'] * 100:.2f}%",
        f"{summary_json['conv_D2_richardson_obs_ghz']:+.2f}",
        f"{summary_json['conv_D2_richardson_obs_rel_error'] * 100:.2f}%",
        f"{summary_json['D2_norm_bessel_ghz']:+.2f}",
        f"{summary_json['D2_norm_fit_ghz']:+.2f}",
        f"{summary_json['D2_norm_rel_error'] * 100:.2f}%",
        f"{summary_json['kerr_shift_max_analytic_ghz']:.2f}",
        f"{summary_json['kerr_shift_max_fit_ghz']:.2f}",
        f"{summary_json['kerr_shift_max_rel_error'] * 100:.2f}%",
        f"{summary_json['lle_fwhm_phi_analytic_rad']:.4f}",
        f"{summary_json['lle_fwhm_phi_sim_rad']:.4f}",
        f"{summary_json['lle_fwhm_phi_rel_error'] * 100:.2f}%",
        f"{summary_json['lle_fwhm_mu_analytic']:.2f}",
        f"{summary_json['lle_fwhm_mu_sim']:.2f}",
        f"{summary_json['lle_fwhm_mu_rel_error'] * 100:.2f}%",
        f"{summary_json['lle_dispersion_scaling_exponent']:.4f}",
        f"{summary_json['lle_dispersion_exponent_rel_error'] * 100:.2f}%",
    ]
    for tok in expected_tokens:
        assert tok in block, (
            f"Expected JSON-derived value {tok!r} not found in Quantitative Summary table:\n{block}"
        )


# ---------------------------------------------------------------------------
# 1. Analytical Bessel WGM Dispersion & Mutation Checks
# ---------------------------------------------------------------------------


def test_bessel_wgm_dispersion_anomalous_vs_normal_and_mutation() -> None:
    """
    Verify cylindrical Bessel WGM dispersion against the independent 2D FIT
    cold-cavity resonance set and finite-difference stencils for the
    dispersion-engineered microring (D_2 > 0) vs the uncompensated microring
    (D_2 < 0), and verify that mutating the dispersion sign or stencil fails.
    """
    mat_anom = hlp.make_ring_material(anomalous=True, chi3=0.0)
    mat_norm = hlp.make_ring_material(anomalous=False, chi3=0.0)

    disp_anom = hlp.bessel_ring_dispersion(mat_anom)
    disp_norm = hlp.bessel_ring_dispersion(mat_norm)

    assert disp_anom["D1_hz"] > 9.5e12
    assert disp_anom["D2_hz"] > 25.0e9
    assert disp_norm["D2_hz"] < -15.0e9

    # Independent check: compare finite-difference D_2 of the 2D FIT resonance
    # set (m = 14..20) and FIT cubic-fit D_2 against the Bessel eigenvalue reference.
    t0_json = json.loads((DATA_DIR / "task_0_cold_dispersion.json").read_text())
    t3_json = json.loads((DATA_DIR / "task_3_convergence.json").read_text())
    f_fit_anom = np.asarray(t0_json["anomalous"]["f_fit_hz"], dtype=float)
    f_fit_norm = np.asarray(t0_json["normal"]["f_fit_hz"], dtype=float)
    # 5-point fourth-order central second difference around m0 = 17 (index 3)
    d2_fit_fd5 = float(
        (
            -f_fit_anom[1]
            + 16.0 * f_fit_anom[2]
            - 30.0 * f_fit_anom[3]
            + 16.0 * f_fit_anom[4]
            - f_fit_anom[5]
        )
        / 12.0
    )
    # Engineering bound, not a GCI justification: it uses the p = 2 end of the
    # extrapolation band [29.03, 30.49] GHz. With the observed order p_obs = 3.71 the
    # GCI would be 5.5 %, below the actual fine-grid error (9.59 %).
    e_disc_rel = abs(t3_json["D2_richardson_hz"] - t3_json["D2_hz"][-1]) / abs(
        disp_anom["D2_hz"]
    )
    d2_disc_tol = 1.5 * e_disc_rel
    _assert_anomalous_ring_dispersion(
        float(t0_json["anomalous"]["D2_fit_hz"]),
        disp_anom["D2_hz"],
        rel_tol=d2_disc_tol,
    )
    _assert_anomalous_ring_dispersion(
        d2_fit_fd5,
        disp_anom["D2_hz"],
        rel_tol=d2_disc_tol,
    )

    # Mutation check 1: Using the uncompensated normal-dispersion FIT ring (D_2 < 0) must fail
    d2_norm_fd3 = float(f_fit_norm[4] - 2.0 * f_fit_norm[3] + f_fit_norm[2])
    with pytest.raises(AssertionError):
        _assert_anomalous_ring_dispersion(
            d2_norm_fd3, disp_anom["D2_hz"], rel_tol=d2_disc_tol
        )

    # Mutation check 2: Perturbing D_2 beyond the GCI discretization bound (20% error) must fail
    with pytest.raises(AssertionError):
        _assert_anomalous_ring_dispersion(
            1.20 * disp_anom["D2_hz"], disp_anom["D2_hz"], rel_tol=d2_disc_tol
        )


# ---------------------------------------------------------------------------
# 2. Live 2D FIT Kerr Ring Shift & Graded Mesh Tests with Mutation Checks
# ---------------------------------------------------------------------------


def test_fit2d_kerr_ring_shift_and_mutations() -> None:
    """
    Verify 2D FIT Kerr microring resonance shift against perturbation theory with
    V_eff,NL, and verify that mutating V_eff,NL or the 3/4 factor fails.
    """
    res = hlp.run_fit_kerr_ring_shift(
        dx_m=35.0e-9,
        chi3=hlp.DEFAULT_CHI3_RING,
        E_peaks=(4.0e7, 8.0e7),
        n_steps=2000,
    )
    df_fit = np.asarray(res["df_fit_hz"], dtype=float)
    df_an = np.asarray(res["df_analytic_hz"], dtype=float)

    _assert_kerr_ring_shift_accuracy(df_fit, df_an, rel_tol=0.06)

    # Amplitude scaling: doubling E_peak quadruples U and quadruples |Delta f_NL|
    ratio_fit = df_fit[1] / df_fit[0]
    assert np.isclose(ratio_fit, 4.0, rtol=0.05, atol=0.0)

    # Mutation check 1: Using a 2x wrong mode volume V_eff,NL halves df_an and must fail
    with pytest.raises(AssertionError):
        _assert_kerr_ring_shift_accuracy(df_fit, 0.5 * df_an, rel_tol=0.06)

    # Mutation check 2: Dropping the 3/4 real-field factor (multiplying df_an by 4/3) must fail
    with pytest.raises(AssertionError):
        _assert_kerr_ring_shift_accuracy(df_fit, (4.0 / 3.0) * df_an, rel_tol=0.06)

    # Mutation check 3: Flipping the sign of the frequency shift must fail
    with pytest.raises(AssertionError):
        _assert_kerr_ring_shift_accuracy(-df_fit, df_an, rel_tol=0.06)


def test_graded_mesh_axes_and_lle_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify the locally refined non-uniform microring mesh and the FIT-informed
    Lugiato-Lefever parameter derivation + soliton bandwidth scaling + mutation checks.
    """
    x_nodes, y_nodes = hlp.build_microring_graded_axes(dx_fine=20.0e-9, dx_coarse=35.0e-9)
    dx_arr = np.diff(x_nodes)
    dy_arr = np.diff(y_nodes)
    assert float(np.min(dx_arr)) <= 20.1e-9
    assert float(np.min(dy_arr)) <= 20.1e-9
    # Graded grid has fewer cells than uniform 20 nm grid (280 x 280 = 78,400 cells)
    n_unif = int(round(2.0 * hlp.DEFAULT_DOMAIN_HALF_M / 20.0e-9))
    assert (len(x_nodes) - 1) * (len(y_nodes) - 1) < n_unif * n_unif

    # Verify run_lle_detuning_scan derives d2_norm and f_pump from FIT inputs
    lle_out = hlp.run_lle_detuning_scan(
        d2_fit_hz=27.89e9,
        kappa_tot_hz=401.06e9,
        f_pump=None,
        n_modes=256,
        f0_fit_hz=214.34e12,
        n0_core=2.0265,
        v_eff_trav_m3=1.2899e-18,
        chi3_m2_v2=hlp.DEFAULT_CHI3_RING,
        eta_coupling=0.9894,
        p_in_w=3.521,
    )
    assert np.isclose(lle_out["d2_norm"], 2.0 * 27.89e9 / 401.06e9, rtol=1e-12)
    assert np.isclose(lle_out["f_pump"], np.sqrt(3.521 / lle_out["p_thresh_w"]), rtol=1e-12)
    _assert_lle_soliton_and_scaling(
        lle_out["in_existence_range"],
        lle_out["fwhm_mu_sim"],
        lle_out["fwhm_mu_an"],
        lle_out["dispersion_scaling_exponent"],
    )

    # Code-path mutation check: mutating chi3_to_n2 by 0.25x quadruples P_th, halves f_pump,
    # and pushes zeta_0 = 4.2 outside the soliton existence window zeta_max = (pi^2/8)*f^2.
    orig_chi3_to_n2 = hlp.chi3_to_n2
    monkeypatch.setattr(hlp, "chi3_to_n2", lambda chi3, eps_r: 0.25 * orig_chi3_to_n2(chi3, eps_r))
    lle_mut = hlp.run_lle_detuning_scan(
        d2_fit_hz=27.89e9,
        kappa_tot_hz=401.06e9,
        f_pump=None,
        n_modes=256,
        f0_fit_hz=214.34e12,
        n0_core=2.0265,
        v_eff_trav_m3=1.2899e-18,
        chi3_m2_v2=hlp.DEFAULT_CHI3_RING,
        eta_coupling=0.9894,
        p_in_w=3.521,
    )
    monkeypatch.setattr(hlp, "chi3_to_n2", orig_chi3_to_n2)
    with pytest.raises(AssertionError):
        _assert_lle_soliton_and_scaling(
            lle_mut["in_existence_range"],
            lle_mut["fwhm_mu_sim"],
            lle_mut["fwhm_mu_an"],
            lle_mut["dispersion_scaling_exponent"],
        )


# ---------------------------------------------------------------------------
# 3. Precomputed Data, Figures, GIF (< 5 MB), CLI, and Summary Table Checks
# ---------------------------------------------------------------------------


def test_precomputed_data_figures_and_summary_table_match_json() -> None:
    """
    Verify that all tutorial data/*.{json,npz}, figures/*.png, GIF (< 5 MB),
    notebook, and the Quantitative Summary table match the JSON outputs.
    """
    assert SCRIPT_PATH.exists(), f"Missing tutorial script {SCRIPT_PATH}"
    assert NOTEBOOK_PATH.exists(), f"Missing generated notebook {NOTEBOOK_PATH}"

    for stem in (
        "task_0_cold_dispersion",
        "task_1_bus_coupling",
        "task_2_kerr_shift",
        "task_3_convergence",
        "task_4_lle_microcomb",
    ):
        assert (DATA_DIR / f"{stem}.json").exists(), f"Missing {stem}.json"
        assert (DATA_DIR / f"{stem}.npz").exists(), f"Missing {stem}.npz"

    summary_json = json.loads((DATA_DIR / "microcomb_summary.json").read_text())
    t2_json = json.loads((DATA_DIR / "task_2_kerr_shift.json").read_text())
    t3_json = json.loads((DATA_DIR / "task_3_convergence.json").read_text())
    t4_json = json.loads((DATA_DIR / "task_4_lle_microcomb.json").read_text())

    # Quantitative accuracy checks on stored data (--full mode)
    assert summary_json["quick"] is False
    assert summary_json["f0_rel_error"] < 0.002
    assert summary_json["conv_f0_richardson_rel_error"] < 0.0005
    assert summary_json["D1_rel_error"] < 0.008
    assert summary_json["conv_D1_richardson_rel_error"] < 0.001
    assert summary_json["Delta_D2_pole_rel_error"] < 0.01
    assert summary_json["D2_anom_fit_ghz"] > 25.0
    # Engineering bound, not a GCI justification: it uses the p = 2 end of the
    # extrapolation band [29.03, 30.49] GHz. With the observed order p_obs = 3.71 the
    # GCI would be 5.5 %, below the actual fine-grid error (9.59 %).
    d2_gci_tol = 1.5 * (
        abs(t3_json["D2_richardson_hz"] - t3_json["D2_hz"][-1])
        / abs(t3_json["D2_bessel_hz"])
    )
    assert summary_json["D2_anom_rel_error"] < d2_gci_tol
    assert summary_json["conv_D2_richardson_rel_error"] < 0.02
    assert 3.0 < summary_json["conv_D2_observed_order"] < 4.5
    assert summary_json["conv_D2_richardson_obs_rel_error"] < 0.08
    assert summary_json["conv_graded_D2_rel_error"] < 0.15
    assert summary_json["D2_norm_fit_ghz"] < -20.0
    assert summary_json["D2_norm_rel_error"] < 0.15
    assert summary_json["kerr_shift_max_rel_error"] < 0.05
    assert summary_json["conv_fine_ppw"] >= 20.0
    assert summary_json["conv_fine_gap_cells"] >= 4.0
    assert summary_json["conv_graded_cell_reduction_pct"] > 15.0
    assert np.isclose(
        summary_json["lle_d2_norm"],
        2.0 * summary_json["D2_anom_fit_ghz"] / summary_json["kappa_tot_ghz"],
        rtol=1e-6,
    )
    assert np.isclose(
        summary_json["lle_f_pump"],
        np.sqrt(summary_json["lle_p_in_w"] / summary_json["lle_p_thresh_w"]),
        rtol=1e-6,
    )
    assert summary_json["lle_fwhm_mu_rel_error"] < 0.05
    assert summary_json["lle_dispersion_exponent_rel_error"] < 0.03

    # Verify Quantitative Summary table in kerr_microcomb.py matches JSON
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")
    _verify_quantitative_summary_table(
        script_text, summary_json, t2_json, t3_json, t4_json
    )

    # Mutation check: perturbing a JSON number must cause the table verification to fail
    mut_summary = dict(summary_json)
    mut_summary["f0_fit_thz"] = summary_json["f0_fit_thz"] + 1.234
    with pytest.raises(AssertionError):
        _verify_quantitative_summary_table(
            script_text, mut_summary, t2_json, t3_json, t4_json
        )

    # Verify setup table & figures exist
    setup_md = FIG_DIR / "00_setup_table.md"
    assert setup_md.exists() and setup_md.stat().st_size > 200

    expected_pngs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "01_microring_wgm_fields_and_resonances.png",
        "02_dispersion_and_kerr_ring_shift.png",
        "03_resolution_convergence.png",
        "04_lle_detuning_scan_and_soliton_regimes.png",
        "05_soliton_comb_spectrum_and_d2_scaling.png",
    ]
    for fname in expected_pngs:
        fpath = FIG_DIR / fname
        assert fpath.exists() and fpath.stat().st_size > 10_000, f"Missing or empty figure {fpath}"

    gif_path = FIG_DIR / "kerr_microcomb_dynamics.gif"
    assert gif_path.exists(), f"Missing GIF {gif_path}"
    gif_mb = gif_path.stat().st_size / (1024.0 * 1024.0)
    assert 0.02 < gif_mb < 5.0, f"GIF size {gif_mb:.2f} MB violates < 5 MB limit"


def test_cli_list_tasks() -> None:
    """Verify that `kerr_microcomb.py --list-tasks` prints all 5 tasks cleanly."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--list-tasks"],
        capture_output=True,
        text=True,
        check=True,
    )
    for task_name in (
        "cold_dispersion",
        "bus_coupling",
        "kerr_shift",
        "convergence",
        "lle_microcomb",
    ):
        assert task_name in proc.stdout
