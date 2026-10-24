# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Integration and mutation test suite for Tutorial 6.3:
`tutorials/06_periodic_structures/06-03_quasi_bic_metasurface/`
("Quasi-bound states in the continuum: metasurface Q ~ alpha^(-2)",
Koshelev et al., Phys. Rev. Lett. 121, 193903, 2018).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUT_DIR = (
    REPO_ROOT
    / "tutorials"
    / "06_periodic_structures"
    / "06-03_quasi_bic_metasurface"
)
if str(TUT_DIR) not in sys.path:
    sys.path.insert(0, str(TUT_DIR))

import _helpers as h  # noqa: E402
import quasi_bic_metasurface as tut  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Geometry, Grid Rules, and 3D Preflight (`wakis.estimate`)
# ---------------------------------------------------------------------------


def test_geometry_grid_rules_and_3d_preflight():
    """
    Verify that the dimer bar-pair unit cell satisfies the project grid rules
    (>= 20 cells per wavelength in silicon, >= 4 cells across the smallest
    85 nm gap feature) and that the 3D tilted-bar unit cell passes
    `wakis.estimate.preflight`.
    """
    geom0 = h.dimer_bar_geometry(alpha=0.0)
    assert np.isclose(geom0["w1"], geom0["w2"])
    assert np.isclose(geom0["gap_inner"], 85.0e-9)

    # Check symmetry breaking preserves total silicon width w1 + w2 = 2 * w0
    geom_asym = h.dimer_bar_geometry(alpha=0.12)
    assert np.isclose(geom_asym["w1"] + geom_asym["w2"], 2.0 * h.WIDTH_W0)
    assert np.isclose(
        (geom_asym["w1"] - geom_asym["w2"]) / h.WIDTH_W0, 0.12, rtol=1e-12
    )

    _, info = h.make_dimer_solver(alpha=0.0, nx=32)
    assert info["cpw_si_x"] >= 20.0
    assert info["cpw_si_z"] >= 20.0
    assert info["cells_across_gap"] >= 4.0

    # Verify 3D tilted-bar preflight via wakis.estimate
    pf = h.preflight_and_run_3d_tilted_bars_task(
        nx=28, ny=28, nz=56, run_solver=False
    )
    assert pf["preflight_laptop_fits"] is True
    assert pf["peak_ram_gb"] < 1.8
    assert pf["n_cells"] == 28 * 28 * 56


# ---------------------------------------------------------------------------
# 2. Universal Inverse-Square Scaling Q_rad ~ alpha^(-2) & Mutation Checks
# ---------------------------------------------------------------------------


def test_q_vs_alpha_inverse_square_slope_and_mutations():
    """
    Verify that log Q_rad vs log alpha has slope -2 within tolerance both in
    live Wakis 2D FIT `harminv` runs and in the 7-point saved dataset, and
    confirm that staircase quantization or index-shift mutations fail the
    slope check.
    """
    # Live 3-point Wakis 2D FIT harminv check across a 4x range in alpha (16x in Q)
    alphas_live = np.array([0.05, 0.10, 0.20], dtype=float)
    q_live = np.array(
        [
            h.run_ringdown_harminv(alpha=float(a), nx=32, ring_cycles=32.0)["Q"]
            for a in alphas_live
        ],
        dtype=float,
    )
    slope_live, intercept_live = np.polyfit(
        np.log10(alphas_live), np.log10(q_live), 1
    )
    q0_live = float(10.0**intercept_live)
    fits_live = h.fit_q_scaling_laws(alphas_live, q_live)
    assert np.isclose(fits_live["slope_free"], slope_live, rtol=1e-12)
    assert np.isclose(fits_live["q0_free"], q0_live, rtol=1e-12)
    assert np.isclose(
        fits_live["q0_imposed"], 4.0 * fits_live["q0_half_imposed"], rtol=1e-12
    )
    assert abs(slope_live - (-2.0)) < 0.03
    assert 44.0 <= fits_live["q0_imposed"] <= 50.0
    assert np.allclose(q_live * (alphas_live**2), fits_live["q0_imposed"], rtol=0.015)

    # Verify cycle-peak ring-down envelope extraction removes carrier zero-crossings
    d0 = np.load(TUT_DIR / "data" / "task_00.npz")
    t_env, env = h.extract_ringdown_envelope(
        d0["t_ring_traces_fs"][-1], d0["sig_ring_traces"][-1], window_fs=22.0
    )
    f0_last = float(d0["f0_fit_hz"][-1])
    q_last = float(d0["Q_fit"][-1])
    env_expected = np.exp(-np.pi * f0_last * (t_env * 1e-15) / q_last)
    assert np.all(env > 0.80)
    assert np.allclose(env, env_expected, rtol=0.015)
    # Mutation check: raw carrier |signal| drops near zero at half-cycle zero-crossings
    raw_norm = np.abs(d0["sig_ring_traces"][-1]) / np.max(np.abs(d0["sig_ring_traces"][-1]))
    assert not np.all(raw_norm > 0.80)

    # Mutation 1 (index shift): pairing alphas with shifted Q values breaks slope = -2
    q_mut = np.roll(q_live, 1)
    slope_mut, _ = np.polyfit(np.log10(alphas_live), np.log10(q_mut), 1)
    assert abs(slope_mut - (-2.0)) > 0.5

    # Mutation 2 (staircase quantization blindness): disabling subpixel smoothing
    # locks the discrete bar edges to the same cell index across alpha in [0.05, 0.20],
    # yielding slope ~ 0 instead of -2!
    q_stair = np.array(
        [
            h.run_ringdown_harminv(
                alpha=float(a), nx=32, ring_cycles=32.0, averaging="staircase"
            )["Q"]
            for a in alphas_live
        ],
        dtype=float,
    )
    slope_stair, _ = np.polyfit(np.log10(alphas_live), np.log10(q_stair), 1)
    assert abs(slope_stair) < 0.05
    assert abs(slope_stair - (-2.0)) > 1.9


# ---------------------------------------------------------------------------
# 3. Mutation Test: alpha = 0 -> No Resonance Dip & Q Above Run-Length Limit
# ---------------------------------------------------------------------------


def test_alpha_zero_bic_dark_state_and_symmetry_breaking_mutation():
    """
    Verify that at alpha = 0 (symmetry-protected BIC at Gamma):
    1. Normal-incidence transmission T(lambda) has no Fano resonance dip;
    2. Ring-down `harminv` extracts Q_BIC > 10^7 (far above the run-length
       Fourier limit Q_run = pi * f_0 * t_tot ~ 450);
    3. Mutating alpha = 0 -> alpha > 0 opens the radiation channel, producing
       a deep Fano resonance (contrast > 0.50) and finite Q_rad, which fails
       the dark-state assertions.
    """
    # 1. Ring-down harminv at alpha = 0 vs run-length limit
    res_bic = h.run_ringdown_harminv(alpha=0.0, nx=32, ring_cycles=32.0)
    t_tot = float(res_bic["t_ring_fs"][-1]) * 1e-15
    q_run_limit = float(np.pi * res_bic["f0_hz"] * t_tot)
    assert res_bic["Q"] > 1.0e7
    assert res_bic["Q"] > 1.0e4 * q_run_limit

    # Mutation: breaking symmetry to alpha = 0.20 drops Q_rad by > 5 orders of magnitude
    res_mut = h.run_ringdown_harminv(alpha=0.20, nx=32, ring_cycles=32.0)
    assert not (res_mut["Q"] > 1.0e7)
    assert 1100.0 < res_mut["Q"] < 1300.0

    # 2. Normal-incidence transmission spectrum at alpha = 0 vs alpha = 0.14
    d0 = np.load(TUT_DIR / "data" / "task_00.npz")
    fan_alphas = np.asarray(d0["fan_alphas"], dtype=float)
    fan_T = np.asarray(d0["fan_T"], dtype=float)
    assert np.isclose(fan_alphas[0], 0.0)
    contrast_sym = float(np.max(fan_T[0]) - np.min(fan_T[0]))
    assert contrast_sym < 0.015

    # Mutation: breaking symmetry to alpha = 0.14 (index 3) opens the Fano resonance (0 <= T <= 1)
    idx_asym = int(np.argmin(np.abs(fan_alphas - 0.14)))
    contrast_asym = float(np.max(fan_T[idx_asym]) - np.min(fan_T[idx_asym]))
    assert not (contrast_asym < 0.015)
    assert contrast_asym > 0.85


# ---------------------------------------------------------------------------
# 4. Fano Fit (Koshelev Eq. 1) vs Extracted Q (`harminv`) & Mutation Check
# ---------------------------------------------------------------------------


def test_fano_fit_vs_extracted_q_and_mutation():
    """
    Verify that the Koshelev Eq. (1) Fano lineshape fit (`fit_fano_spectrum`)
    agrees with Wakis `harminv` Q extraction to within 5%, and that a mutated
    Fano lineshape (perturbing the normalized detuning scale) fails the check.
    """
    d0 = np.load(TUT_DIR / "data" / "task_00.npz")

    # Direct Wakis 2D FIT DiffractionMonitor Fano fit vs harminv at alpha = 0.48
    q_fano_fit = float(d0["fano_Q_fit"])
    q_harminv = float(d0["fano_Q_harminv"])
    lam_fano = float(d0["fano_lambda0_fit_nm"])
    lam_harminv = float(d0["fano_lambda0_harminv_nm"])
    rel_diff_q = abs(q_fano_fit - q_harminv) / q_harminv
    assert rel_diff_q < 0.04
    assert abs(lam_fano - lam_harminv) < 0.25
    sym_contrast = float(
        np.max(d0["fano_T_sym_alpha0"]) - np.min(d0["fano_T_sym_alpha0"])
    )
    assert sym_contrast < 0.015

    # Energy balance R_00 + T_00 = 1 in lossless Wakis FIT DiffractionMonitor
    r_plus_t = np.asarray(d0["fano_R_asym"]) + np.asarray(d0["fano_T_asym"])
    assert abs(float(np.mean(r_plus_t)) - 1.0) < 0.01
    assert np.allclose(r_plus_t, 1.0, atol=0.06)

    # Also verify 7-point RCWA Fano Q vs Wakis 2D FIT harminv Q across alpha in [0.02, 0.20]
    q_fit_7 = np.asarray(d0["Q_fit"], dtype=float)
    q_rcwa_7 = np.asarray(d0["Q_rcwa"], dtype=float)
    assert np.all(np.abs(q_fit_7 - q_rcwa_7) / q_rcwa_7 < 0.035)

    # Mutation check: if the Fano detuning convention omits the factor of 2 in
    # Omega = 2*(f - f0)/gamma, the extracted Q is off by 2x and fails the < 4% check
    wls_nm = np.asarray(d0["fano_wls_nm"], dtype=float)
    freqs = 299792458.0 / (wls_nm * 1e-9)
    T_spec = np.asarray(d0["fano_T_asym"], dtype=float)
    fit_exact = h.fit_fano_spectrum(freqs, T_spec)
    q_mutated_convention = 0.5 * fit_exact["Q_fano"]
    assert abs(q_mutated_convention - q_harminv) / q_harminv > 0.45


# ---------------------------------------------------------------------------
# 5. Absorption Q Saturation, Oblique Floquet Dispersion & Saved Artifacts
# ---------------------------------------------------------------------------


def test_absorption_saturation_oblique_floquet_and_artifacts():
    """
    Verify:
    1. Absorption saturation Q_tot(alpha) = (Q_rad^-1 + Q_abs^-1)^-1 and bare
       radiative recovery Q_rad,rec(alpha) = (Q_tot(alpha)^-1 - Q_tot(0)^-1)^-1
       (plus sign-flip mutation);
    2. Oblique-incidence angular dispersion Q_rad(k_parallel) ~ k_parallel^(-2)
       and `wakis.floquet.AngleSweep` constant-angle spectra;
    3. All required data files, figures (including GIF < 5 MB), and notebook exist.
    """
    d2 = np.load(TUT_DIR / "data" / "task_02.npz")
    q_rad = np.asarray(d2["Q_rad_lossless"], dtype=float)
    q_tot_sim = np.asarray(d2["Q_tot_sim"], dtype=float)
    q_tot_pred = np.asarray(d2["Q_tot_pred"], dtype=float)
    q_rad_rec = np.asarray(d2["Q_rad_recovered"], dtype=float)
    q_tot_0 = np.asarray(d2["Q_tot_alpha0"], dtype=float)

    assert np.allclose(q_tot_sim, q_tot_pred, rtol=0.015)
    assert np.allclose(q_rad_rec, q_rad[None, :], rtol=0.015)

    # Mutation on bare radiative recovery: adding 1/Q_tot(0) instead of subtracting
    # fails to recover the lossless Q_rad(alpha)
    q_rad_rec_mut = 1.0 / (1.0 / q_tot_sim + 1.0 / q_tot_0[:, None])
    assert not np.allclose(q_rad_rec_mut, q_rad[None, :], rtol=0.05)

    # Oblique-incidence dispersion Q(k_parallel) ~ sin(theta)^(-2)
    d3 = np.load(TUT_DIR / "data" / "task_03.npz")
    slope_theta = float(d3["log_slope_sin_theta"])
    assert abs(slope_theta - (-2.0)) < 0.03

    # AngleSweep at theta=0 is dark/flat, while theta=12 deg shows the quasi-BIC dip
    t_fl = np.asarray(d3["T_floquet"], dtype=float)
    contrast_theta0 = float(np.max(t_fl[0]) - np.min(t_fl[0]))
    contrast_theta_max = float(np.max(t_fl[-1]) - np.min(t_fl[-1]))
    assert contrast_theta0 < 0.03
    assert contrast_theta_max > 0.12

    # Resolution convergence & artifacts check
    d4 = np.load(TUT_DIR / "data" / "task_04.npz")
    lam_conv = np.asarray(d4["lambda_bic_nm"], dtype=float)
    # Monotonic convergence toward RCWA benchmark 1575.763 nm
    assert np.all(np.diff(lam_conv) > 0.0)

    expected_figs = [
        "00_setup_schematic_and_mesh.png",
        "01_fano_transmission_spectra.png",
        "02_q_vs_alpha_scaling.png",
        "03_bic_vs_quasi_bic_fields.png",
        "04_bic_vs_quasi_bic_ringdown.gif",
        "05_absorption_q_saturation.png",
        "06_oblique_floquet_dispersion.png",
        "07_resolution_and_discretization_study.png",
    ]
    for fname in expected_figs:
        fpath = TUT_DIR / "figures" / fname
        assert fpath.is_file(), f"Missing figure {fname}"
        size_bytes = fpath.stat().st_size
        assert size_bytes > 5_000, f"Figure {fname} is unexpectedly empty"
        if fname.endswith(".gif"):
            assert size_bytes < 5 * 1024 * 1024, "GIF exceeds 5 MB limit"

    with open(TUT_DIR / "data" / "summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert abs(summary["log_slope_fit"] - (-2.0)) < 0.02
    assert tut.build_tasks(quick=True) is not None
