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
    """Verify `wall=False` bulk transfer against `sinc(be*dz/2)^p * sinc(omega*dt/2)` on 10x474 and 20x948."""
    params = scw.compute_analytic_parameters()
    be = params["beta_e"]
    omega = params["omega"]
    u = params["u_dep"]

    failures = []
    for label, (nx, _, nz) in (
        ("10x474", scw.COARSE_GRID),
        ("20x948", scw.FINE_GRID),
    ):
        dz = scw.TUNNEL_LENGTH_M / nz
        dt, _ = scw._cached_integer_period_dt(nx, nz)
        for p in (1, 2, 3):
            h_bulk = scw.entrance_edge_transfer(p, 0, dz, dt, omega, u, wall=False)
            ref = float(
                (np.sinc(0.5 * be * dz / np.pi) ** p)
                * np.sinc(0.5 * omega * dt / np.pi)
            )
            diff = h_bulk - ref
            rel_err = abs(diff) / ref
            print(
                f"BULK_LIMIT grid={label} p={p} h_bulk={h_bulk:.12f} "
                f"ref={ref:.12f} diff={diff:+.6e} rel_err={rel_err:.6e}",
                flush=True,
            )
            if rel_err >= 1.0e-9:
                failures.append(
                    f"grid={label} p={p}: h_bulk={h_bulk:.12f}, ref={ref:.12f}, "
                    f"diff={diff:+.6e}, rel_err={rel_err:.6e} >= 1e-9"
                )

    assert not failures, "Bulk limit rel_err >= 1e-9:\n  " + "\n  ".join(failures)


def test_closed_form_wall_minus_bulk_convergence():
    """Verify `(H_0(wall) - H_0(bulk)) / (be*dz)^2` convergence on decreasing `dz` at fixed `sigma`."""
    params = scw.compute_analytic_parameters()
    be = params["beta_e"]
    omega = params["omega"]
    u = params["u_dep"]

    nx0, _, nz0 = scw.COARSE_GRID
    dz0 = scw.TUNNEL_LENGTH_M / nz0
    dt0, _ = scw._cached_integer_period_dt(nx0, nz0)
    sigma = float(u * dt0 / dz0)
    dz_seq = [dz0 / (2**m) for m in range(5)]

    for dz in dz_seq:
        dt = sigma * dz / u
        hw1 = scw.entrance_edge_transfer(1, 0, dz, dt, omega, u, wall=True)
        hb1 = scw.entrance_edge_transfer(1, 0, dz, dt, omega, u, wall=False)
        print(
            f"CLOSED_FORM p=1 dz={dz:.6e} sigma={sigma:.8f} "
            f"h_wall={hw1:.12f} h_bulk={hb1:.12f} diff={hw1 - hb1:+.6e}",
            flush=True,
        )

    failures = []
    for p, c_target in (
        (2, 5.0 / 384.0 + sigma / 32.0),
        (3, 7.0 / 240.0 + sigma / 24.0),
    ):
        rel_errs = []
        for dz in dz_seq:
            dt = sigma * dz / u
            hw = scw.entrance_edge_transfer(p, 0, dz, dt, omega, u, wall=True)
            hb = scw.entrance_edge_transfer(p, 0, dz, dt, omega, u, wall=False)
            scaled = (hw - hb) / ((be * dz) ** 2)
            rel_err = abs(scaled - c_target) / c_target
            rel_errs.append(rel_err)
            print(
                f"CLOSED_FORM p={p} dz={dz:.6e} sigma={sigma:.8f} "
                f"scaled={scaled:.10f} target={c_target:.10f} "
                f"diff={scaled - c_target:+.6e} rel_err={rel_err:.6e}",
                flush=True,
            )
        is_decreasing = all(
            rel_errs[i + 1] < rel_errs[i] for i in range(len(rel_errs) - 1)
        )
        if not is_decreasing or rel_errs[-1] >= 0.02:
            failures.append(
                f"p={p}: target={c_target:.10f}, rel_errs={[float(f'{r:.6e}') for r in rel_errs]}, "
                f"decreasing={is_decreasing}, final_rel_err={rel_errs[-1]:.6e}"
            )

    assert not failures, "Closed-form convergence check failed:\n  " + "\n  ".join(
        failures
    )


def test_p1_wall_agrees_with_bulk():
    """Verify `p=1` wall and bulk transfer amplitudes agree to `1e-12` on 10x474 and 20x948."""
    params = scw.compute_analytic_parameters()
    omega = params["omega"]
    u = params["u_dep"]

    failures = []
    for label, (nx, _, nz) in (
        ("10x474", scw.COARSE_GRID),
        ("20x948", scw.FINE_GRID),
    ):
        dz = scw.TUNNEL_LENGTH_M / nz
        dt, _ = scw._cached_integer_period_dt(nx, nz)
        hw = scw.entrance_edge_transfer(1, 0, dz, dt, omega, u, wall=True)
        hb = scw.entrance_edge_transfer(1, 0, dz, dt, omega, u, wall=False)
        abs_diff = abs(hw - hb)
        print(
            f"P1_WALL_VS_BULK grid={label} h_wall={hw:.12f} h_bulk={hb:.12f} "
            f"diff={hw - hb:+.6e} abs_diff={abs_diff:.6e}",
            flush=True,
        )
        if abs_diff >= 1.0e-12:
            failures.append(
                f"grid={label}: h_wall={hw:.12f}, h_bulk={hb:.12f}, abs_diff={abs_diff:.6e} >= 1e-12"
            )

    assert not failures, "p=1 wall vs bulk differed by >= 1e-12:\n  " + "\n  ".join(
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
