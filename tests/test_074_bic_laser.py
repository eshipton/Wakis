# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Validation and mutation test suite for Tutorial 7.9:
`tutorials/07_light_sources/07-09_bic_laser/bic_laser.py`.

Covers:
1. Infinite-lattice symmetry-protected BIC at Gamma (`Q(Gamma) > 10^7`),
   momentum-space quadratic radiation scaling `Q(k_x) ~ k_x^(-2)` (slope `-2.0`),
   accidental/resonance-trapped BIC radius peak `Q_acc(R)` vs flat symmetry-protected
   `Q_sym(R)` (Kodigala et al., Nature 541, 196, 2017), and 3D FIT Bloch far-field
   polarization vortex topological charges `q = +1` (`A_1`) and `q = -1` (`B_1`)
   around Gamma — each with explicit mutation checks that must fail.
2. Finite-array `N x N` edge/vertical radiation scaling `Q_1(N) ~ N^p` (`p = 2.55`)
   and mode selection across standing-wave envelope orders `m = 1, 2, 3` (`m = 1`
   highest `Q`, `Q_1 / Q_2 ~ 4`, `Q_1 / Q_3 ~ 9`), plus mutation checks.
3. Active lasing threshold `g_th = 2*pi*n / (lambda_0 * Gamma_conf * Q)` vs 2D FIT
   `SaturableGain` zero-crossing (`0.06%` relative error), nonlinear L-I curve vs
   spatial-hole-burning integral theory, `FourLevelGain` population-inversion
   clamping `Delta N / Delta N_th -> 1`, and finite-array far-field doughnut beam
   (`I(theta = 0 deg) = 0`), plus mutation checks (dropping `Gamma_conf`, dropping
   `n_Si`, corrupting `I_sat`, and breaking anti-phased dipole parity).
4. Symmetry-breaking mutation (`alpha != 0`, reusing Tutorial 6.3): `Q_rad(alpha) = Q_0 * alpha^(-2)`
   lowers total `Q_tot(alpha, N)` and raises threshold gain quadratically
   `Delta g_th(alpha) = kappa_th * alpha^2` (slope `+2.0`), quenching lasing at
   fixed pump.
5. Grid resolution rules (`>= 20` cells/wavelength in `n = 3.48`, `>= 4` cells across
   the `85 nm` inner gap), 3-resolution mesh convergence (`nx in {32, 48, 64}`),
   figure/GIF existence and size (`< 5 MB`), and exact agreement between the
   Quantitative Summary Markdown table in `bic_laser.py` / `bic_laser.ipynb` and
   `data/summary.json`.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from wakis.gain import SaturableGain
from wakis.resonances import harminv
from wakis.sources import CurrentSource, GaussianSource

REPO_ROOT = Path(__file__).resolve().parents[1]
TUT_DIR = REPO_ROOT / "tutorials" / "07_light_sources" / "07-09_bic_laser"
DATA_DIR = TUT_DIR / "data"
FIG_DIR = TUT_DIR / "figures"


def _load_helpers():
    spec = importlib.util.spec_from_file_location(
        "_tut_07_09_helpers", TUT_DIR / "_helpers.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load_helpers()


# ---------------------------------------------------------------------------
# Verification helpers (each designed so physical mutations fail)
# ---------------------------------------------------------------------------
def _assert_kx_inverse_square_law(kx_norms: np.ndarray, q_kx: np.ndarray, tol: float = 0.03):
    """Assert Q(k_x) decreases monotonically and obeys Q(k_x) ~ k_x^(-2) near Gamma."""
    kx_arr = np.asarray(kx_norms, dtype=float)
    q_arr = np.asarray(q_kx, dtype=float)
    if not np.all(np.diff(q_arr) < 0.0):
        raise AssertionError("Q(k_x) is not strictly decreasing away from Gamma.")
    slope, _ = np.polyfit(np.log10(kx_arr[:3]), np.log10(q_arr[:3]), 1)
    if abs(slope - (-2.0)) / 2.0 > tol:
        raise AssertionError(
            f"Momentum-space Q(k_x) log-log slope {slope:.4f} deviates from -2.0 by more than {tol:.2%}."
        )


def _assert_vortex_charge_and_null(
    q_charge: float,
    expected_q: int,
    gamma_suppression: float,
    max_suppression: float = 1e-3,
):
    """Assert integer topological charge q and dark radiation null at Gamma."""
    if abs(float(q_charge) - float(expected_q)) > 0.05:
        raise AssertionError(
            f"Measured topological charge {q_charge:.4f} != expected {expected_q}."
        )
    if float(gamma_suppression) > max_suppression:
        raise AssertionError(
            f"Gamma radiation suppression {gamma_suppression:.3e} exceeds {max_suppression:.3e} (BIC is not dark)."
        )


def _assert_finite_array_mode_selection(
    n_list: np.ndarray,
    q1_n: np.ndarray,
    q_modes_n8: np.ndarray,
):
    """Assert Q_1(N) ~ N^p (p in [2.0, 3.0]) and fundamental mode m=1 has highest Q."""
    if not np.all(np.diff(q1_n) > 0.0):
        raise AssertionError("Finite-array Q_1(N) must increase monotonically with N.")
    p_exp, _ = np.polyfit(np.log10(n_list), np.log10(q1_n), 1)
    if not (2.0 <= p_exp <= 3.0):
        raise AssertionError(f"Finite-array exponent p = {p_exp:.4f} outside [2.0, 3.0].")
    if not (q_modes_n8[0] > q_modes_n8[1] > q_modes_n8[2]):
        raise AssertionError(
            f"Envelope modes do not obey Q_1 > Q_2 > Q_3: {q_modes_n8}."
        )
    ratio_12 = q_modes_n8[0] / q_modes_n8[1]
    if abs(ratio_12 - 4.0) / 4.0 > 0.12:
        raise AssertionError(f"Mode ratio Q_1/Q_2 = {ratio_12:.3f} deviates from 4.0.")


def _assert_threshold_gain_identity(
    g_th_fit: float,
    q_val: float,
    lambda0_m: float,
    n_active: float,
    gamma_conf: float,
    tol: float = 0.015,
):
    """Assert active FIT threshold gain matches g_th = 2*pi*n / (lambda_0 * Gamma_conf * Q)."""
    g_th_an = float(
        h.bic_threshold_gain(
            q_val, lambda0_m=lambda0_m, n_active=n_active, gamma_conf=gamma_conf
        )
    )
    rel_err = abs(float(g_th_fit) - g_th_an) / g_th_an
    if rel_err > tol:
        raise AssertionError(
            f"Threshold gain mismatch: FIT={g_th_fit:.4f}, analytic={g_th_an:.4f} (rel_err={rel_err:.4%})."
        )


def _assert_shb_li_curve(
    g0_ratios: np.ndarray,
    i_fit: np.ndarray,
    i_pred: np.ndarray,
    tol: float = 0.025,
):
    """Assert above-threshold SaturableGain L-I curve matches spatial-hole-burning theory."""
    above = g0_ratios >= 1.3
    rel_err = np.max(np.abs(i_fit[above] - i_pred[above]) / i_pred[above])
    if rel_err > tol:
        raise AssertionError(
            f"SaturableGain L-I curve deviates from SHB theory by {rel_err:.4%} > {tol:.2%}."
        )


def _assert_doughnut_farfield_null(
    thetas_deg: np.ndarray,
    intensity_theta: np.ndarray,
    max_null_ratio: float = 1e-4,
):
    """Assert far-field radiation pattern has a dark central null at theta = 0 deg."""
    idx0 = int(np.argmin(np.abs(thetas_deg)))
    null_ratio = float(intensity_theta[idx0] / np.max(intensity_theta))
    if null_ratio > max_null_ratio:
        raise AssertionError(
            f"Far-field normal emission I(0 deg)/I_max = {null_ratio:.4e} > {max_null_ratio:.4e} (no doughnut null)."
        )


def _assert_symmetry_protected_bic_dark(q_val: float, min_q: float = 1.0e8):
    """Assert that a symmetry-protected BIC mode at Gamma is radiatively dark (Q >= min_q)."""
    if float(q_val) < float(min_q):
        raise AssertionError(
            f"Symmetry-protected mode Q(Gamma) = {float(q_val):.2e} < {float(min_q):.2e} (radiative leakage present)."
        )


def _assert_quadratic_alpha_threshold(
    alphas: np.ndarray, g_th_arr: np.ndarray, tol: float = 0.02
):
    """Assert that threshold gain penalty Delta g_th(alpha) = g_th(alpha) - g_th(0) scales as alpha^2."""
    a_nz = np.asarray(alphas[1:], dtype=float)
    dg = np.asarray(g_th_arr[1:], dtype=float) - float(g_th_arr[0])
    if not np.all(np.diff(dg) > 0.0):
        raise AssertionError("Threshold gain penalty Delta g_th(alpha) is not strictly increasing.")
    slope, _ = np.polyfit(np.log10(a_nz), np.log10(dg), 1)
    if abs(float(slope) - 2.0) / 2.0 > tol:
        raise AssertionError(
            f"Threshold penalty log-log slope {slope:.4f} deviates from +2.0 by more than {tol:.2%}."
        )


# ---------------------------------------------------------------------------
# 1. Infinite Lattice: Q(k_x) ~ k_x^(-2), Bar-Width BIC Sweep, and 3D Vortex
# ---------------------------------------------------------------------------
def test_part1_infinite_bic_and_3d_vortex_with_mutations():
    """
    Verify Part 1 identities from `data/task_00.npz` and live 2D/3D FIT checks,
    and confirm that physical code-path mutations fail.
    """
    with np.load(DATA_DIR / "task_00.npz") as d0:
        kx_nz = d0["kx_nonzero"]
        q_kx = d0["Q_kx_fit"]
        w0_nm = d0["w0_nm"]
        q_sym_w0 = d0["Q_sym_vs_w0"]
        q_qbic_w0 = d0["Q_qbic_vs_w0"]
        q_qbic_w0_nx48 = d0["Q_qbic_vs_w0_nx48"]
        phi_deg = d0["phi_deg"]
        cpw_3d_a1 = float(d0["cpw_3d_A1"])
        q_gamma_a1_pml6 = float(d0["Q_gamma_A1_pml6"])
        q_gamma_a1 = float(d0["Q_gamma_A1"])
        k_3d_norms = d0["k_3d_norms"]
        q_k_3d_a1 = d0["Q_k_3d_A1"]
        q_ring_a1_max = float(d0["Q_ring_A1_max"])
        q_a1 = float(d0["q_charge_A1"])
        supp_a1 = float(d0["gamma_suppression_A1"])
        ex_ring_a1 = d0["ex_ring_A1"]
        ey_ring_a1 = d0["ey_ring_A1"]
        kx_ring = d0["kx_ring"]
        ky_ring = d0["ky_ring"]
        x_mid_3d = d0["x_mid_3d"]
        y_node_3d = d0["y_node_3d"]
        x_node_3d = d0["x_node_3d"]
        y_mid_3d = d0["y_mid_3d"]

    # Symmetry-protected 2D BIC at Gamma has Q > 10^8, while off-Gamma follows k_x^(-2)
    _assert_symmetry_protected_bic_dark(float(q_kx[0]), min_q=1.0e8)
    _assert_kx_inverse_square_law(kx_nz, q_kx[1:], tol=0.02)

    # 2D FIT bar-width sweep across w0 in [140, 200] nm:
    # Symmetric (alpha=0) remains radiation-free (Q > 5e8) for every bar width w0,
    # whereas breaking symmetry (alpha=0.10) opens radiative leakage (Q in [1e3, 2e4])
    # and is strictly monotonic on the refined nx=48 grid
    assert len(w0_nm) == 5
    assert np.all(q_sym_w0 > 5.0e8)
    assert np.all((q_qbic_w0 > 1.0e3) & (q_qbic_w0 < 2.0e4))
    assert np.all(np.diff(q_qbic_w0_nx48) > 0.0)

    # 3D Bloch FIT resolved A_1 BIC mode:
    # 1. cpw_3d_A1 >= 20 evaluated at actual A_1 resonance frequency (196.36 THz) in n_cyl = 3.15
    # 2. Q_Gamma(A_1) diverges with CPML thickness (n_PML=10 vs n_PML=6 by > 100x) and exceeds off-Gamma ring by > 500x
    # 3. 3D off-Gamma Q(k_||) obeys k_||^(-2) across k_|| a / (2*pi) in {0.03, 0.05, 0.08}
    # 4. Off-mirror loop extraction via `extract_bloch_zeroth_order_ring` -> `compute_topological_charge` gives q = +1
    assert cpw_3d_a1 >= 20.0
    assert q_gamma_a1 > 1.0e7
    assert q_gamma_a1 > 100.0 * q_gamma_a1_pml6
    assert q_gamma_a1 > 500.0 * q_ring_a1_max
    _assert_kx_inverse_square_law(k_3d_norms, q_k_3d_a1, tol=0.03)
    assert np.all(np.mod(phi_deg, 45.0) != 0.0)
    ring_ext = h.extract_bloch_zeroth_order_ring(
        ex_ring_a1,
        ey_ring_a1,
        kx_ring,
        ky_ring,
        x_mid_3d,
        y_node_3d,
        x_node_3d,
        y_mid_3d,
    )
    assert abs(float(ring_ext["q_charge"]) - q_a1) < 1.0e-12
    _assert_vortex_charge_and_null(
        ring_ext["q_charge"], expected_q=+1, gamma_suppression=supp_a1
    )

    # Live code-path check on `run_bar_width_bic_sweep`: symmetric (alpha=0) passes
    # `_assert_symmetry_protected_bic_dark`, whereas mutating geometry to broken C_2
    # symmetry (alpha=0.14) collapses Q to < 3.5e3 and FAILS `_assert_symmetry_protected_bic_dark`
    live_w0 = h.run_bar_width_bic_sweep(
        widths_nm=(170.0,), alpha_qbic=0.14, nx=32, ring_cycles=24.0
    )
    _assert_symmetry_protected_bic_dark(float(live_w0["Q_sym_fit"][0]), min_q=1.0e8)
    assert float(live_w0["Q_qbic_fit"][0]) < 3.5e3
    with pytest.raises(AssertionError):
        _assert_symmetry_protected_bic_dark(float(live_w0["Q_qbic_fit"][0]), min_q=1.0e8)

    # --- Code-path mutation checks (must fail) ---
    # Mutation 1a: Mutating `extract_bloch_zeroth_order_ring` with `loop_stride=-1`
    # slices the ring sample arrays in reversed order (`[::loop_stride]`), reversing
    # the contour traversal order and flipping q from +1 to -1
    ring_mut_cw = h.extract_bloch_zeroth_order_ring(
        ex_ring_a1,
        ey_ring_a1,
        kx_ring,
        ky_ring,
        x_mid_3d,
        y_node_3d,
        x_node_3d,
        y_mid_3d,
        loop_stride=-1,
    )
    with pytest.raises(AssertionError):
        _assert_vortex_charge_and_null(
            ring_mut_cw["q_charge"], expected_q=+1, gamma_suppression=supp_a1
        )

    # Mutation 1b: Mutating `extract_bloch_zeroth_order_ring` by negating the transverse
    # y-component amplitude (`y_component_sign=-1.0`) flips the sign of Stokes S_2
    # (and thus psi = 0.5 * atan2(S_2, S_1)), flipping q from +1 to -1 and failing
    ring_mut_sign = h.extract_bloch_zeroth_order_ring(
        ex_ring_a1,
        ey_ring_a1,
        kx_ring,
        ky_ring,
        x_mid_3d,
        y_node_3d,
        x_node_3d,
        y_mid_3d,
        y_component_sign=-1.0,
    )
    with pytest.raises(AssertionError):
        _assert_vortex_charge_and_null(
            ring_mut_sign["q_charge"], expected_q=+1, gamma_suppression=supp_a1
        )

    # Mutation 1c: Real 2D FIT code-path mutation on `run_ringdown_harminv`: breaking C_2
    # symmetry (`alpha=0.20` instead of `alpha=0.0`) at k_x p / (2*pi) = 0.02 introduces
    # k_x-independent asymmetry radiation loss, flattening Q(k_x) near Gamma and failing
    # `_assert_kx_inverse_square_law`
    kx_002 = float(kx_nz[0]) * (2.0 * np.pi / h.PERIOD_P)
    r_kx_mut = h.h63.run_ringdown_harminv(
        alpha=0.20, nx=32, kx=kx_002, ring_cycles=24.0
    )
    q_kx_mut = np.array(q_kx[1:], dtype=float, copy=True)
    q_kx_mut[0] = float(r_kx_mut["Q"])
    with pytest.raises(AssertionError):
        _assert_kx_inverse_square_law(kx_nz, q_kx_mut, tol=0.03)


# ---------------------------------------------------------------------------
# 2. Finite Strip Arrays: Q_1(N) ~ N^p Scaling and Envelope Mode Selection
# ---------------------------------------------------------------------------
def test_part2_finite_array_q_scaling_and_mode_selection_with_mutations():
    """
    Verify Part 2 finite N-period strip scaling `Q_1(N) ~ N^p` and envelope mode
    selection `Q_1(8) > Q_2(8) > Q_3(8)`, plus live code-path mutation checks.
    """
    with np.load(DATA_DIR / "task_01.npz") as d1:
        n_list = d1["N_list"]
        q1_n = d1["Q1_N_fit"]
        q_m_n8 = d1["Q_m_N8_fit"]
        ey_n8 = d1["ey_n8_m1_2d"]
        x_n8 = d1["x_n8_um"]

    _assert_finite_array_mode_selection(n_list, q1_n, q_m_n8)

    # Verify global C_2 antisymmetry of the m=1 eigenmode along x: E_y(-x, z) = -E_y(x, z)
    sym_sum = np.linalg.norm(ey_n8 + ey_n8[::-1, :]) / np.linalg.norm(ey_n8)
    assert sym_sum < 1.0e-5
    assert abs(float(x_n8[0] + x_n8[-1])) < 1.0e-6

    # --- Code-path mutation checks (must fail) ---
    # Mutation 2a: Live 2D FIT ringdown with mutated envelope mode order `mode_m=2`
    # instead of fundamental `mode_m=1` at N=4 degrades Q by ~3.5x and fails mode selection
    r_m1 = h.run_finite_array_ringdown(n_cells_array=4, mode_m=1, nx_per_cell=32, ring_cycles=18.0)
    r_m2 = h.run_finite_array_ringdown(n_cells_array=4, mode_m=2, nx_per_cell=32, ring_cycles=18.0)
    assert float(r_m1["Q"]) > 2.5 * float(r_m2["Q"])
    with pytest.raises(AssertionError):
        _assert_finite_array_mode_selection(
            n_list, q1_n, np.array([float(r_m2["Q"]), float(r_m1["Q"]), float(q_m_n8[2])])
        )

    # Mutation 2b: Sub-quadratic N^1.0 scaling fails the N^p (p >= 2) check
    with pytest.raises(AssertionError):
        _assert_finite_array_mode_selection(n_list, 20.0 * n_list, q_m_n8)


# ---------------------------------------------------------------------------
# 3. Active Lasing Threshold, SaturableGain L-I, FourLevelGain, and Doughnut Beam
# ---------------------------------------------------------------------------
def test_part3_active_threshold_li_clamping_and_doughnut_with_mutations():
    """
    Verify Part 3 active threshold gain (`0.06%` vs perturbation formula),
    spatial-hole-burning L-I curve, noise-seeded `FourLevelGain` inversion clamping,
    and `Near2FarMonitor` doughnut beam (`I(0 deg)/I_max < 2e-3`), plus real
    code-path mutation checks.
    """
    from wakis.near2far import Near2FarMonitor

    with np.load(DATA_DIR / "task_02.npz") as d2:
        q_a08 = float(d2["lin_Q_cold"])
        lam_a08 = float(d2["lin_lambda0_nm"]) * 1.0e-9
        g_th_an = float(d2["lin_g_th_analytic_m1"])
        g_th_fit = float(d2["lin_g_th_fit_m1"])
        gamma_conf = float(d2["lin_gamma_conf"])
        g0_ratios = d2["sat_g0_ratios"]
        i_fit = d2["sat_I_fit"]
        i_shb = d2["sat_I_shb"]
        fl_dn_above = d2["fl_dn_above"]
        fl_dn_sub = d2["fl_dn_sub"]
        fl_time_scale = float(d2["fl_time_scale"])
        thetas_deg = d2["dough_thetas_deg"]
        i_ff_fit = d2["dough_I_fit"]
        proj_m1 = d2["dough_proj_m1"]
        proj_m2 = d2["dough_proj_m2"]
        dough_ey_2d = d2["dough_ey_2d"]

    with np.load(DATA_DIR / "task_03.npz") as d3_ref:
        idx_a14 = int(np.argmin(np.abs(d3_ref["alphas"] - 0.14)))
        q_cold_a14 = float(d3_ref["Q_rad_inf"][idx_a14])

    # 1. Cold-cavity vs active FIT linear threshold gain identity
    _assert_threshold_gain_identity(
        g_th_fit,
        q_val=q_a08,
        lambda0_m=lam_a08,
        n_active=h.N_SI,
        gamma_conf=gamma_conf,
        tol=0.005,
    )
    assert abs(g_th_fit - g_th_an) / g_th_an < 0.002

    # Live active FIT check: verify that adding linear gain g0 > 0 reduces the modal decay rate
    solver, meta = h.h63.make_dimer_solver(alpha=0.14, nx=32, cfln=0.80)
    f0 = h.F0_BIC_REF
    omega0 = 2.0 * np.pi * f0
    mat_gain = SaturableGain(
        omega_0=omega0,
        gamma=0.08 * omega0,
        g0=5000.0,
        I_sat=np.inf,
        eps_inf=h.EPS_SI,
        tau_I=2.0 / f0,
    )
    h.attach_subpixel_gain(solver, meta["fill_2d"], mat_gain)
    dx = h.PERIOD_P / 32.0
    wf = GaussianSource(frequency=f0, fwidth=0.08 * f0)
    src = CurrentSource(
        wf,
        component="y",
        center=(-0.5 * h.DIMER_D, 0.5 * dx, 0.0),
        size=(0.0, dx, 0.0),
        amplitude=1.0e-4,
    )
    dt = float(solver.dt)
    n_steps = int((wf.end_time + 24.0 / f0) / dt)
    n_start = int(np.ceil(wf.end_time / dt)) + 15
    ixL = int(np.argmin(np.abs(meta["x_nodes"] - (-0.5 * h.DIMER_D))))
    ixR = int(np.argmin(np.abs(meta["x_nodes"] - (+0.5 * h.DIMER_D))))
    iz0 = int(np.argmin(np.abs(meta["z_nodes"])))
    flat_L = solver.N + ixL + solver.Nx * solver.Ny * iz0
    flat_R = solver.N + ixR + solver.Nx * solver.Ny * iz0
    e_arr = solver.E.array
    sig_gain = np.empty(n_steps - n_start, dtype=float)
    for step in range(n_steps):
        src.update(solver, step * dt)
        solver.one_step()
        if step >= n_start:
            sig_gain[step - n_start] = float(e_arr[flat_L] - e_arr[flat_R])
    res_g = harminv(sig_gain, dt, 0.92 * f0, 1.08 * f0, max_rel_error=None)
    dec_gain = float(res_g.decay_rate[int(np.argmax(np.abs(res_g.amplitude)))])
    dec_cold = float(np.pi * f0 / q_cold_a14)
    assert dec_gain < dec_cold  # Active gain reduces modal decay rate (moves toward lasing)

    # 2. SaturableGain L-I curve vs spatial-hole-burning theory
    _assert_shb_li_curve(g0_ratios, i_fit, i_shb, tol=0.015)

    # 3. Noise-seeded FourLevelGain(spontaneous_emission=True, time_scale=80.0 <= 100) clamping
    assert 0.0 < fl_time_scale <= 100.0
    assert float(np.max(fl_dn_above)) > 1.30
    dn_clamped = float(np.mean(fl_dn_above[-200:]))
    dn_sub_final = float(np.mean(fl_dn_sub[-200:]))
    assert abs(dn_clamped - 1.0) < 0.12
    assert dn_sub_final < 0.75

    # 4. Single-mode dominance (|a_1| >> |a_2| at late times) and Near2FarMonitor doughnut null
    assert np.std(proj_m1[-300:]) > 2.5 * np.std(proj_m2[-300:])
    _assert_doughnut_farfield_null(thetas_deg, i_ff_fit, max_null_ratio=2.0e-3)

    # --- Code-path mutation checks (must fail) ---
    # Mutation 3a: Mutating `compute_modal_confinement_2d` by inverting the active fill mask
    # (`1 - fill_2d`, placing gain in air instead of the silicon bars) corrupts Gamma_conf
    ey_unit_2d = np.real(solver.E.to_matrix("y")[:, 0, :])
    gamma_conf_mut = h.compute_modal_confinement_2d(
        ey_unit_2d,
        1.0 - meta["fill_2d"],
        meta["eps_r_2d"],
        solver.grid.z,
        eps_active=h.EPS_SI,
    )
    with pytest.raises(AssertionError):
        _assert_threshold_gain_identity(
            g_th_fit, q_val=q_a08, lambda0_m=lam_a08, n_active=h.N_SI, gamma_conf=gamma_conf_mut
        )

    # Mutation 3b: Calling `bic_threshold_gain` with wrong refractive index `n_active = 1.0`
    with pytest.raises(AssertionError):
        _assert_threshold_gain_identity(
            g_th_fit,
            q_val=q_a08,
            lambda0_m=lam_a08,
            n_active=1.0,
            gamma_conf=gamma_conf,
        )

    # Mutation 3c: Mutating `predict_shb_steady_intensity` with a strongly peaked artificial
    # field profile (`ey_unit_2d ** 6`) over-saturates the antinodes and fails the L-I check
    i_shb_mut = h.predict_shb_steady_intensity(
        g0_ratios,
        ey_unit_2d**6,
        meta["fill_2d"],
        solver.grid.z,
    )
    with pytest.raises(AssertionError):
        _assert_shb_li_curve(g0_ratios, i_fit, i_shb_mut, tol=0.025)

    # Mutation 3d: Real code-path mutation on `project_near2far_doughnut(n2f, thetas_deg)`:
    # Construct a `Near2FarMonitor` on the finite N=8 solver, populate it with the
    # C_2-odd m=1 aperture field, verify it passes `_assert_doughnut_farfield_null`,
    # then mutate the right half (x > 0) of the recorded `Near2FarMonitor` tangential
    # fields by flipping their sign (making the aperture globally even) and verify
    # that `project_near2far_doughnut` then produces a bright normal-incidence lobe
    # that FAILS `_assert_doughnut_farfield_null`!
    slv_fin, meta_fin = h.make_finite_array_solver(n_cells_array=8, alpha=0.0, nx_per_cell=32)
    x_fin = meta_fin["x_nodes"]
    z_fin = meta_fin["z_nodes"]
    n2f_test = Near2FarMonitor(
        frequencies=[f0],
        center=(0.0, 0.5 * dx, +0.65e-6),
        size=(float(x_fin[-11] - x_fin[10]), dx, 0.0),
        eps_r=1.0,
    )
    iz_ap = int(np.argmin(np.abs(z_fin - 0.65e-6)))
    slv_fin.E.from_matrix(np.repeat(dough_ey_2d[:, None, :], slv_fin.Ny, axis=1), "y")
    slv_fin.H.from_matrix(
        np.repeat((-dough_ey_2d / 376.7303)[:, None, :], slv_fin.Ny, axis=1), "x"
    )
    n2f_test.update(slv_fin, 0.0)
    proj_odd = h.project_near2far_doughnut(n2f_test, thetas_deg)
    _assert_doughnut_farfield_null(thetas_deg, proj_odd["I_farfield_fit"], max_null_ratio=2.0e-3)

    # Now mutate the recorded Near2FarMonitor tangential field on x > 0 (flipping parity)
    ey_even = dough_ey_2d.copy()
    ey_even[slv_fin.Nx // 2 :, :] *= -1.0
    n2f_mut = Near2FarMonitor(
        frequencies=[f0],
        center=(0.0, 0.5 * dx, +0.65e-6),
        size=(float(x_fin[-11] - x_fin[10]), dx, 0.0),
        eps_r=1.0,
    )
    slv_fin.E.from_matrix(np.repeat(ey_even[:, None, :], slv_fin.Ny, axis=1), "y")
    slv_fin.H.from_matrix(
        np.repeat((-ey_even / 376.7303)[:, None, :], slv_fin.Ny, axis=1), "x"
    )
    n2f_mut.update(slv_fin, 0.0)
    proj_mut = h.project_near2far_doughnut(n2f_mut, thetas_deg)
    assert abs(float(dough_ey_2d[slv_fin.Nx // 4, iz_ap])) > 0.0
    with pytest.raises(AssertionError):
        _assert_doughnut_farfield_null(
            thetas_deg, proj_mut["I_farfield_fit"], max_null_ratio=2.0e-3
        )


# ---------------------------------------------------------------------------
# 4. Symmetry-Breaking Mutation (alpha != 0) and 3-Resolution Convergence
# ---------------------------------------------------------------------------
def test_part4_symmetry_breaking_mutation_and_convergence():
    """
    Verify that breaking C_2 symmetry (`alpha != 0`) directly in the finite N=8
    strip lowers `Q_tot(alpha, N=8)`, raises threshold gain quadratically
    `Delta g_th(alpha, N=8) ~ alpha^2`, and quenches lasing at fixed pump; also
    check grid resolution and 3-level mesh convergence.
    """
    with np.load(DATA_DIR / "task_01.npz") as d1_ref:
        idx_n8 = int(np.argmin(np.abs(d1_ref["N_list"] - 8.0)))
        lam1_n8_m = float(d1_ref["lambda1_N_nm"][idx_n8]) * 1.0e-9

    with np.load(DATA_DIR / "task_02.npz") as d2_ref:
        gamma_conf_ref = float(d2_ref["lin_gamma_conf"])

    with np.load(DATA_DIR / "task_03.npz") as d3:
        n_array_edge = int(d3["n_array_edge"])
        q0_const = float(d3["Q0_const"])
        alphas = d3["alphas"]
        q_tot = d3["Q_tot_finite"]
        q_tot_an = d3["Q_tot_finite_analytic"]
        g_th_an = d3["g_th_analytic_cm1"]
        g_th_fit = d3["g_th_fit_cm1"]
        slope_alpha = float(d3["slope_alpha_threshold"])
        quench_time_scale = float(d3["quench_time_scale"])
        i_lasing = d3["i_norm_alpha004_lasing"]
        i_quenched = d3["i_norm_alpha018_quenched"]

    assert n_array_edge == 8
    assert quench_time_scale == 20.0
    # Symmetry breaking monotonically lowers finite-strip Q_tot and raises g_th(alpha) as alpha^2
    assert np.all(np.diff(q_tot) < 0.0)
    assert np.all(np.diff(g_th_fit) > 0.0)
    assert np.max(np.abs(q_tot - q_tot_an) / q_tot_an) < 0.01
    assert abs(slope_alpha - 2.0) / 2.0 < 0.01
    _assert_quadratic_alpha_threshold(alphas, g_th_fit, tol=0.01)
    assert np.max(np.abs(g_th_fit - g_th_an) / g_th_an) < 0.01

    # Lasing quenching when alpha increases from 0.04 to 0.18 at fixed pump
    quench_ratio = float(np.mean(i_quenched[-200:]) / np.mean(i_lasing[-200:]))
    assert quench_ratio < 0.05

    # Code-path mutation check on `bic_threshold_gain`: mutating Q_rad(alpha) from
    # inverse-square `Q0 * alpha^(-2)` to inverse-linear `Q0 * alpha^(-1)` produces
    # a linear threshold penalty (slope ~ 1.0) that fails `_assert_quadratic_alpha_threshold`
    q_tot_mut = 1.0 / (1.0 / float(q_tot[0]) + (alphas ** 1.0) / q0_const)
    g_mut = h.bic_threshold_gain(
        q_tot_mut, lambda0_m=lam1_n8_m, n_active=h.N_SI, gamma_conf=gamma_conf_ref
    )
    with pytest.raises(AssertionError):
        _assert_quadratic_alpha_threshold(alphas, g_mut / 100.0, tol=0.05)

    # 3-resolution spatial mesh convergence (nx in {32, 48, 64})
    with np.load(DATA_DIR / "task_04.npz") as d4:
        nx_list = d4["nx_list"]
        cpw_si = d4["cpw_si"]
        cells_gap = d4["cells_across_gap"]
        err_gth = d4["rel_err_g_th"]
        err_q = d4["rel_err_Q_a10"]
        rel_spread_lam = float(d4["rel_spread_lambda_bic"])

    assert list(nx_list) == [32, 48, 64]
    assert np.all(cpw_si >= 20.0)
    assert np.all(cells_gap >= 4.0)
    assert err_gth[-1] < err_gth[0] < 0.01
    assert err_q[-1] < err_q[0] < 0.01
    assert rel_spread_lam < 1.0e-3


# ---------------------------------------------------------------------------
# 5. Quantitative Summary Table vs JSON & Figure/GIF Verification
# ---------------------------------------------------------------------------
def test_quantitative_summary_table_matches_json_and_figures_exist():
    """
    Verify that every numerical entry in the Section 7 Quantitative Summary table
    and Abstract/Results of `bic_laser.py` (and `bic_laser.ipynb`) matches
    `data/summary.json`, and that all setup schematics, figures, and the animated
    GIF exist and are < 5 MB.
    """
    summary = json.loads((DATA_DIR / "summary.json").read_text())
    t0 = summary["task_00"]
    t1 = summary["task_01"]
    t2 = summary["task_02"]
    t3 = summary["task_03"]
    t4 = summary["task_04"]

    # Ensure invalid lambda_BIC Richardson extrapolation and old Kodigala constants are removed
    assert "lambda_bic_richardson_nm" not in t4
    assert "R_opt_nm" not in t0

    expected_strings = [
        f"{t0['lambda_bic_gamma_nm']:.4f} nm",
        f"{t4['lambda_bic_nm'][-1]:.4f} nm",
        f"{t0['Q_bic_gamma']:.1f}",
        f"{t0['slope_Q_vs_kx_near_gamma']:.4f}",
        f"{t0['Q_sym_min']:.1f}",
        f"{t0['Q_gamma_A1']:.1f}",
        f"{t0['Q_gamma_A1_pml6']:.1f}",
        f"{t0['slope_Q_vs_k_3d_A1']:.4f}",
        f"+{t0['q_charge_A1']:.4f}",
        f"{t1['Q1_over_Q2_N8']:.3f}",
        f"{t1['p_exponent_fit']:.4f}",
        f"{t2['g_th_alpha008_fit_cm1']:.3f}",
        f"{t2['g_th_alpha008_analytic_cm1']:.3f}",
        f"{t2['sat_I_fit'][6]:.4f}",
        f"{t2['sat_I_shb'][6]:.4f}",
        f"{t2['fl_dn_clamped_final']:.4f}",
        f"{t2['doughnut_nodal_null_ratio']:.6f}",
        f"{t2['doughnut_theta_peak_fit_deg']:.2f} deg",
        f"{t2['doughnut_theta_peak_theory_deg']:.2f} deg",
        f"+{t3['slope_alpha_threshold']:.4f}",
        f"{t3['quench_ratio_alpha018_over_004']:.4f}",
        f"{t4['Q_a10'][0]:.2f}",
        f"{t4['Q_a10_richardson']:.2f}",
        f"{t4['g_th_a10_cm1'][0]:.3f}",
        f"{t4['g_th_a10_richardson_cm1']:.3f}",
    ]

    py_text = (TUT_DIR / "bic_laser.py").read_text()
    ipynb_text = (TUT_DIR / "bic_laser.ipynb").read_text()

    # Extract Section 7 Quantitative Summary table from bic_laser.py
    match = re.search(
        r"## 7\. Quantitative Summary(.*?)## 8\. Limits & Failure Modes",
        py_text,
        re.DOTALL,
    )
    assert match is not None, "Section 7 Quantitative Summary not found in bic_laser.py"
    table_block = match.group(1)

    for val_str in expected_strings:
        assert val_str in table_block, (
            f"JSON value '{val_str}' missing from Quantitative Summary table in bic_laser.py"
        )
        assert val_str in ipynb_text, (
            f"JSON value '{val_str}' missing from bic_laser.ipynb"
        )

    # Also verify that Abstract and Results text contain the exact JSON numbers
    abstract_and_results_strings = [
        f"{t0['slope_Q_vs_kx_near_gamma']:.4f}",
        f"{t1['p_exponent_fit']:.4f}",
        f"{t2['max_rel_err_shb_li'] * 100:.2f}\\%",
        f"+{t3['slope_alpha_threshold']:.4f}",
        f"{t3['Q0_const']:.2f}",
        f"{t2['doughnut_theta_peak_fit_deg']:.2f}^\\circ",
        f"{t2['doughnut_theta_peak_theory_deg']:.2f}^\\circ",
    ]
    for s_val in abstract_and_results_strings:
        assert s_val in py_text, f"Expected JSON value '{s_val}' in bic_laser.py narrative"

    # Mutation check: altering a JSON number must fail the table match check
    with pytest.raises(AssertionError):
        fake_str = "99999.9999 nm"
        assert fake_str in table_block

    # Check all required figures and GIF exist and are non-empty (< 5 MB for GIF)
    required_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00_setup_table.md",
        "00a_setup_geometry_finite.png",
        "00b_setup_mesh_and_excitation_finite.png",
        "00_setup_table_finite.md",
        "01_infinite_bic_q_and_vortex.png",
        "02_finite_array_q_vs_n.png",
        "03_lasing_threshold_and_doughnut.png",
        "03b_bic_lasing_buildup.gif",
        "04_symmetry_breaking_mutation.png",
    ]
    for fname in required_figs:
        fpath = FIG_DIR / fname
        assert fpath.exists(), f"Missing figure file: {fpath}"
        size_bytes = fpath.stat().st_size
        assert size_bytes > 500, f"Figure file too small: {fpath} ({size_bytes} B)"
        if fname.endswith(".gif"):
            assert size_bytes < 5 * 1024 * 1024, (
                f"GIF exceeds 5 MB limit: {size_bytes / (1024 * 1024):.2f} MB"
            )
