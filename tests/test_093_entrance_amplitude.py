# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Fast tests for `entrance_edge_transfer` and shape-`p` PEC-wall entrance amplitude bounds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as c_light

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import examples._space_charge_waves_common as scw


def test_bulk_limit_transfer_amplitude():
    """Verify phase-averaged complex `wall=False` bulk transfer against `sinc(be*dz/2)^p * sinc(omega*dt/2)` on 10x474 and 20x948."""
    params = scw.compute_analytic_parameters()
    be = params["beta_e"]
    omega = params["omega"]
    u = params["u_dep"]
    n_phases = 64

    failures = []
    for label, (nx, _, nz) in (
        ("10x474", scw.COARSE_GRID),
        ("20x948", scw.FINE_GRID),
    ):
        dz = scw.TUNNEL_LENGTH_M / nz
        dt, _ = scw._cached_integer_period_dt(nx, nz)
        sigma = float(u * dt / dz)
        zeta0_vals = np.arange(n_phases, dtype=np.float64) * (sigma / n_phases)
        for p in (1, 2, 3):
            h_bulk_single = scw.entrance_edge_transfer(
                p, 0, dz, dt, omega, u, wall=False
            )
            h_complex_phases = [
                scw._entrance_edge_transfer_complex(
                    p,
                    0,
                    dz,
                    dt,
                    omega,
                    u,
                    wall=False,
                    z0_offset_cells=float(zeta0),
                )
                for zeta0 in zeta0_vals
            ]
            h_bulk_avg = float(np.abs(np.mean(h_complex_phases)))
            ref = float(
                (np.sinc(0.5 * be * dz / np.pi) ** p)
                * np.sinc(0.5 * omega * dt / np.pi)
            )
            diff_single = h_bulk_single - ref
            diff_avg = h_bulk_avg - ref
            rel_err_single = abs(diff_single) / ref
            rel_err_avg = abs(diff_avg) / ref
            print(
                f"BULK_LIMIT grid={label} p={p} N={n_phases} "
                f"h_bulk_single={h_bulk_single:.12f} h_bulk_avg={h_bulk_avg:.12f} "
                f"ref={ref:.12f} diff_single={diff_single:+.6e} diff_avg={diff_avg:+.6e} "
                f"rel_err_single={rel_err_single:.6e} rel_err_avg={rel_err_avg:.6e}",
                flush=True,
            )
            if rel_err_single <= rel_err_avg:
                failures.append(
                    f"grid={label} p={p}: unaveraged rel_err_single={rel_err_single:.6e} "
                    f"<= phase-averaged rel_err_avg={rel_err_avg:.6e}"
                )
            if p == 1:
                # For p = 1, the step-integrated B-spline Fourier transform decays only as
                # q_m^{-(p+1)} = q_m^{-2} at the Poisson alias wavenumbers
                # q_m * dz = be * dz + 2 * pi * m / sigma. Averaging over N = 64 start phases
                # cancels all aliases m not divisible by N, leaving m = +-k * N (k >= 1) bounded by
                # sum_{k>=1} 2 * (omega * dt / 2) * (sigma / (pi * k * N))^2 = (be * dz) * sigma^2 / (6 * N^2)
                # (~4.90e-8 on 10x474 and ~2.60e-8 on 20x948; actual rel_err_avg is 2.85e-9 and 9.41e-11).
                p1_alias_bound = float((be * dz) * (sigma**2) / (6.0 * (n_phases**2)))
                if rel_err_avg >= p1_alias_bound:
                    failures.append(
                        f"grid={label} p=1: h_bulk_avg={h_bulk_avg:.12f}, ref={ref:.12f}, "
                        f"rel_err_avg={rel_err_avg:.6e} >= p1_alias_bound={p1_alias_bound:.6e}"
                    )
            elif rel_err_avg >= 1.0e-9:
                failures.append(
                    f"grid={label} p={p}: h_bulk_avg={h_bulk_avg:.12f}, ref={ref:.12f}, "
                    f"diff_avg={diff_avg:+.6e}, rel_err_avg={rel_err_avg:.6e} >= 1e-9"
                )

    assert not failures, "Bulk limit phase-averaged check failed:\n  " + "\n  ".join(
        failures
    )


def test_closed_form_wall_minus_bulk_convergence():
    """Verify `(H_0(wall) - H_0(bulk)) / (be*dz)^2` converges to the cut-kernel moment polynomial in `sigma`.

    Derivation of the small-`(be * dz)` target coefficient `c_p(sigma)`:
    On edge `k = 0`, the PEC-wall cut kernel on `zeta = z / dz >= 0` consists of the
    birth-closure point mass `c_0 = C_0(0)` at `tau_0 = 0` (`c_0 = 1/8` for `p = 2`,
    `c_0 = 1/6` for `p = 3`) plus the truncated smooth spline density
    `f(zeta) = C_0'(zeta) = W_{p-1}(zeta - 1/2)` on `zeta > 0` with `f(0) = 1/2` and
    continuum moments `mu_r = int_0^inf zeta^r f(zeta) d_zeta`:
      - `p = 2`: `c_0 = 1/8`, `mu_1 = 25/48`, `mu_2 = 79/192`
      - `p = 3`: `c_0 = 1/6`, `mu_1 = 13/24`, `mu_2 = 29/60`

    Sampling on discrete steps `tau_j = j * sigma` (`sigma = u * dt / dz`),
    Euler-Maclaurin summation yields the discrete cut-kernel moments:
      - `M_0 = c_0 + (1 - c_0) = 1`
      - `M_1(sigma) = mu_1 - ((1 - c_0) / 2) * sigma + (1 / 24) * sigma^2`
      - `M_2(sigma) = mu_2 - mu_1 * sigma + ((1 - c_0) / 3) * sigma^2 - (1 / 24) * sigma^3 + O(sigma^4)`

    Taking the modulus `H_wall(theta) = |M_0 - i * theta * M_1 - 0.5 * theta^2 * M_2 + O(theta^3)|`
    `= 1 - 0.5 * (M_2 - M_1^2) * theta^2 + O(theta^4)` with `theta = be * dz` (where the
    `+0.5 * M_1^2 * theta^2` term comes from the quadrature phase in `|.|`) and subtracting
    `H_bulk(theta) = sinc(theta / 2)^p * sinc(theta * sigma / 2) = 1 - ((p + sigma^2) / 24) * theta^2 + O(theta^4)`
    gives `c_p(sigma) = (p + sigma^2) / 24 - 0.5 * (M_2(sigma) - M_1(sigma)^2)`:
      - `p = 2`: `61/4608 + (25/768)*sigma + (61/4608)*sigma^2 + (1/384)*sigma^3`
      - `p = 3`: `173/5760 + (13/288)*sigma + (7/576)*sigma^2 + (1/288)*sigma^3`
    """
    params = scw.compute_analytic_parameters()
    be = params["beta_e"]
    omega = params["omega"]
    u = params["u_dep"]

    nx0, _, nz0 = scw.COARSE_GRID
    dz0 = scw.TUNNEL_LENGTH_M / nz0
    dt0, _ = scw._cached_integer_period_dt(nx0, nz0)
    sigma0 = float(u * dt0 / dz0)
    dz_seq = [dz0 / (2**m) for m in range(5)]

    failures = []
    for sig_label, sigma in (
        ("dt_10x474", sigma0),
        ("dt_quarter", 0.25 * sigma0),
    ):
        for dz in dz_seq:
            dt = sigma * dz / u
            hw1 = scw.entrance_edge_transfer(1, 0, dz, dt, omega, u, wall=True)
            hb1 = scw.entrance_edge_transfer(1, 0, dz, dt, omega, u, wall=False)
            print(
                f"CLOSED_FORM sigma_case={sig_label} p=1 dz={dz:.6e} sigma={sigma:.8f} "
                f"h_wall={hw1:.12f} h_bulk={hb1:.12f} diff={hw1 - hb1:+.6e}",
                flush=True,
            )

        c2_target = float(
            61.0 / 4608.0
            + (25.0 / 768.0) * sigma
            + (61.0 / 4608.0) * (sigma**2)
            + (1.0 / 384.0) * (sigma**3)
        )
        c3_target = float(
            173.0 / 5760.0
            + (13.0 / 288.0) * sigma
            + (7.0 / 576.0) * (sigma**2)
            + (1.0 / 288.0) * (sigma**3)
        )
        for p, c_target in ((2, c2_target), (3, c3_target)):
            rel_errs = []
            for dz in dz_seq:
                dt = sigma * dz / u
                hw = scw.entrance_edge_transfer(p, 0, dz, dt, omega, u, wall=True)
                hb = scw.entrance_edge_transfer(p, 0, dz, dt, omega, u, wall=False)
                scaled = (hw - hb) / ((be * dz) ** 2)
                rel_err = abs(scaled - c_target) / c_target
                rel_errs.append(rel_err)
                print(
                    f"CLOSED_FORM sigma_case={sig_label} p={p} dz={dz:.6e} sigma={sigma:.8f} "
                    f"scaled={scaled:.10f} target={c_target:.10f} "
                    f"diff={scaled - c_target:+.6e} rel_err={rel_err:.6e}",
                    flush=True,
                )
            is_decreasing = all(
                rel_errs[i + 1] < rel_errs[i] for i in range(len(rel_errs) - 1)
            )
            if not is_decreasing or rel_errs[-1] >= 1.0e-3:
                failures.append(
                    f"sigma_case={sig_label} p={p}: target={c_target:.10f}, "
                    f"rel_errs={[float(f'{r:.6e}') for r in rel_errs]}, "
                    f"decreasing={is_decreasing}, final_rel_err={rel_errs[-1]:.6e}"
                )

    assert not failures, "Closed-form convergence check failed:\n  " + "\n  ".join(
        failures
    )


def test_p1_wall_agrees_with_bulk():
    """Verify exact rational closure weights `C_0(0)` for `p=1,2,3` and `p=1` wall-minus-bulk within the phase-alias bound."""
    params = scw.compute_analytic_parameters()
    omega = params["omega"]
    u = params["u_dep"]
    n_phases = 64

    failures = []
    for p, expected_c0 in (
        (1, 0.0),
        (2, 1.0 / 8.0),
        (3, 1.0 / 6.0),
    ):
        c0_val = float(scw._charge_tail_fraction(0.0, 0, p))
        c0_diff = abs(c0_val - expected_c0)
        print(
            f"CLOSURE_WEIGHT p={p} c0={c0_val:.18f} expected={expected_c0:.18f} "
            f"abs_diff={c0_diff:.6e}",
            flush=True,
        )
        if c0_diff > 1.0e-15:
            failures.append(
                f"p={p}: C_0(0)={c0_val:.18f}, expected={expected_c0:.18f}, abs_diff={c0_diff:.6e} > 1e-15"
            )

    nx0, _, nz0 = scw.COARSE_GRID
    dz0 = scw.TUNNEL_LENGTH_M / nz0
    dt0, _ = scw._cached_integer_period_dt(nx0, nz0)
    sigma0 = float(u * dt0 / dz0)
    dz_seq = [dz0 / (2**m) for m in range(5)]

    cases = [
        (
            f"grid={label}",
            scw.TUNNEL_LENGTH_M / nz,
            scw._cached_integer_period_dt(nx, nz)[0],
        )
        for label, (nx, _, nz) in (
            ("10x474", scw.COARSE_GRID),
            ("20x948", scw.FINE_GRID),
        )
    ] + [(f"fixed_sigma_m={m}", dz, sigma0 * dz / u) for m, dz in enumerate(dz_seq)]

    for case_label, dz, dt in cases:
        sigma = float(u * dt / dz)
        zeta0_vals = np.arange(n_phases, dtype=np.float64) * (sigma / n_phases)
        hw = scw.entrance_edge_transfer(1, 0, dz, dt, omega, u, wall=True)
        hb = scw.entrance_edge_transfer(1, 0, dz, dt, omega, u, wall=False)
        hb_phases = np.asarray(
            [
                scw.entrance_edge_transfer(
                    1,
                    0,
                    dz,
                    dt,
                    omega,
                    u,
                    wall=False,
                    z0_offset_cells=float(zeta0),
                )
                for zeta0 in zeta0_vals
            ],
            dtype=np.float64,
        )
        hc_phases = [
            scw._entrance_edge_transfer_complex(
                1,
                0,
                dz,
                dt,
                omega,
                u,
                wall=False,
                z0_offset_cells=float(zeta0),
            )
            for zeta0 in zeta0_vals
        ]
        hb_avg = float(np.abs(np.mean(hc_phases)))
        max_alias = float(np.max(np.abs(hb_phases - hb_avg)))
        abs_diff = abs(hw - hb)
        print(
            f"P1_WALL_VS_BULK case={case_label} dz={dz:.6e} sigma={sigma:.8f} "
            f"h_wall={hw:.12f} h_bulk={hb:.12f} h_bulk_avg={hb_avg:.12f} "
            f"diff={hw - hb:+.6e} abs_diff={abs_diff:.6e} max_phase_alias={max_alias:.6e}",
            flush=True,
        )
        if abs_diff > max_alias + 1.0e-15:
            failures.append(
                f"case={case_label}: |h_wall - h_bulk|={abs_diff:.6e} > max_phase_alias={max_alias:.6e} + 1e-15"
            )

    # Mutation check: for p = 2, 3 (where C_0(0) != 0), |H_wall - H_bulk| exceeds the phase-alias amplitude
    zeta0_vals_0 = np.arange(n_phases, dtype=np.float64) * (sigma0 / n_phases)
    for p_mut in (2, 3):
        hw_m = scw.entrance_edge_transfer(p_mut, 0, dz0, dt0, omega, u, wall=True)
        hb_m = scw.entrance_edge_transfer(p_mut, 0, dz0, dt0, omega, u, wall=False)
        hb_phases_m = np.asarray(
            [
                scw.entrance_edge_transfer(
                    p_mut,
                    0,
                    dz0,
                    dt0,
                    omega,
                    u,
                    wall=False,
                    z0_offset_cells=float(zeta0),
                )
                for zeta0 in zeta0_vals_0
            ],
            dtype=np.float64,
        )
        hc_phases_m = [
            scw._entrance_edge_transfer_complex(
                p_mut,
                0,
                dz0,
                dt0,
                omega,
                u,
                wall=False,
                z0_offset_cells=float(zeta0),
            )
            for zeta0 in zeta0_vals_0
        ]
        hb_avg_m = float(np.abs(np.mean(hc_phases_m)))
        max_alias_m = float(np.max(np.abs(hb_phases_m - hb_avg_m)))
        diff_m = abs(hw_m - hb_m)
        print(
            f"P1_MUTATION_CHECK p={p_mut} abs_diff={diff_m:.6e} max_phase_alias={max_alias_m:.6e}",
            flush=True,
        )
        with pytest.raises(AssertionError, match="exceeds"):
            assert diff_m <= max_alias_m + 1.0e-15, (
                f"p={p_mut} wall-minus-bulk {diff_m:.6e} exceeds phase-alias bound {max_alias_m:.6e}"
            )

    assert not failures, "p=1 closure and phase-alias check failed:\n  " + "\n  ".join(
        failures
    )


def test_single_grid_budget_terms_identity():
    """Verify `_single_grid_budget_terms` returns `amp_pred_part` and `tol_amp_part` identical to the helper formulas."""
    params = scw.compute_analytic_parameters()
    be = params["beta_e"]
    bq = params["beta_q"]
    omega = params["omega"]
    u = params["u_dep"]
    v_inj = float(c_light * np.sqrt(1.0 - 1.0 / (params["gamma0"] ** 2)))

    for label, (nx, _, nz) in (
        ("10x474", scw.COARSE_GRID),
        ("20x948", scw.FINE_GRID),
    ):
        dz = scw.TUNNEL_LENGTH_M / nz
        dt, _ = scw._cached_integer_period_dt(nx, nz)
        for p in (1, 2, 3):
            terms = scw._single_grid_budget_terms(nx, nz, params, shape=p)
            h0 = scw.entrance_edge_transfer(p, 0, dz, dt, omega, u, wall=True)
            amp_expected = float(h0 * np.cos(0.5 * bq * dz))
            tol_expected = float(
                2.0 * (v_inj / u - 1.0) * p * ((be * dz) ** 2) / 24.0
                + (1.0 - np.cos(0.5 * bq * dz))
                + 1.0e-12
            )
            print(
                f"BUDGET_ID grid={label} p={p} amp_pred_part={terms['amp_pred_part']:.12f} "
                f"amp_expected={amp_expected:.12f} tol_amp_part={terms['tol_amp_part']:.6e} "
                f"tol_expected={tol_expected:.6e}",
                flush=True,
            )
            np.testing.assert_allclose(
                terms["amp_pred_part"], amp_expected, rtol=0.0, atol=1e-15
            )
            np.testing.assert_allclose(
                terms["tol_amp_part"], tol_expected, rtol=0.0, atol=1e-15
            )


def test_mutation_old_bulk_prediction_vs_new_wall_prediction():
    """Verify old bulk prediction `sinc(be*dz/2)^p * cos(bq*dz/2)` differs from new `amp_pred_part` by `> tol_amp_part` for `p=2,3` and `< tol_amp_part` for `p=1`."""
    params = scw.compute_analytic_parameters()
    be = params["beta_e"]
    bq = params["beta_q"]

    for label, (nx, _, nz) in (
        ("10x474", scw.COARSE_GRID),
        ("20x948", scw.FINE_GRID),
    ):
        dz = scw.TUNNEL_LENGTH_M / nz
        for p in (1, 2, 3):
            terms = scw._single_grid_budget_terms(nx, nz, params, shape=p)
            amp_new = float(terms["amp_pred_part"])
            tol_new = float(terms["tol_amp_part"])
            amp_old = float(
                (np.sinc(0.5 * be * dz / np.pi) ** p) * np.cos(0.5 * bq * dz)
            )
            diff = abs(amp_new - amp_old)
            print(
                f"MUTATION_OLD_BULK grid={label} p={p} amp_new={amp_new:.10f} "
                f"amp_old={amp_old:.10f} diff={diff:.6e} tol_amp_part={tol_new:.6e} "
                f"exceeds_tol={diff > tol_new}",
                flush=True,
            )
            if p == 1:
                assert diff < tol_new, (
                    f"grid={label} p=1: diff={diff:.6e} >= tol_amp_part={tol_new:.6e}"
                )
            else:
                with pytest.raises(AssertionError, match="exceeds"):
                    assert diff <= tol_new, (
                        f"grid={label} p={p}: old bulk vs new wall diff {diff:.6e} "
                        f"exceeds tol_amp_part={tol_new:.6e}"
                    )
