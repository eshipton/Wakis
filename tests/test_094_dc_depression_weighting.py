# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Fast tests for `compute_discrete_dc_depression(..., weighting=...)` and charge-weighted DC depression."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import examples._space_charge_waves_common as scw


def test_charge_weighting_harmonic_mean_identity():
    """Verify `1 - delta_charge == sum(rho_l) / sum(rho_l / u_l) / u0` to `1e-14` from converged `u_l, rho_l`."""
    params = scw.compute_analytic_parameters()
    u0 = params["u0"]
    b = scw.BEAM_HALF_THICKNESS_M

    rng = np.random.default_rng(12345)
    x_nonunif = b * np.array([-0.75, -0.45, -0.15, 0.05, 0.35, 0.70], dtype=np.float64)
    w_nonunif = rng.uniform(0.5, 2.0, size=len(x_nonunif))

    cases = [
        ("coarse_p2", scw.COARSE_GRID[0], None, None, 2),
        ("fine_p2", scw.FINE_GRID[0], None, None, 2),
        ("coarse_p1", scw.COARSE_GRID[0], None, None, 1),
        ("coarse_p3", scw.COARSE_GRID[0], None, None, 3),
        ("nonunif_p2", scw.COARSE_GRID[0], x_nonunif, w_nonunif, 2),
    ]

    for label, nx, line_x, line_w, p in cases:
        det = scw.compute_discrete_dc_depression(
            nx,
            params,
            line_x=line_x,
            line_w=line_w,
            weighting="charge",
            return_details=True,
            shape=p,
        )
        scalar_chg = scw.compute_discrete_dc_depression(
            nx,
            params,
            line_x=line_x,
            line_w=line_w,
            weighting="charge",
            return_details=False,
            shape=p,
        )
        assert scalar_chg == det["delta_disc"]
        assert det["delta_disc"] == det["delta_charge"]
        assert det["weighting"] == "charge"

        u_l = det["u_l"]
        rho_l = det["rho_l"]
        lhs = 1.0 - float(det["delta_charge"])
        rhs = float(np.sum(rho_l) / np.sum(rho_l / u_l) / u0)
        abs_err = abs(lhs - rhs)
        print(
            f"HARM_IDENTITY case={label} delta_charge={det['delta_charge']:.16e} "
            f"lhs={lhs:.16e} rhs={rhs:.16e} abs_err={abs_err:.6e}",
            flush=True,
        )
        assert abs_err <= 1.0e-14, (
            f"case={label}: |(1 - delta_charge) - sum(rho_l)/sum(rho_l/u_l)/u0| = "
            f"{abs_err:.6e} > 1e-14"
        )

        # Mutation check: flux or density delta does not satisfy the harmonic-mean identity
        err_flux_vs_harm = abs((1.0 - float(det["delta_flux"])) - rhs)
        with pytest.raises(AssertionError, match="exceeds"):
            assert err_flux_vs_harm <= 1.0e-14, (
                f"case={label}: flux vs harmonic identity diff {err_flux_vs_harm:.6e} exceeds 1e-14"
            )


def test_nonuniform_ul_ordering_and_leading_variance():
    """Verify strict ordering `delta_charge > delta_density > delta_flux` and `delta_charge - delta_flux ~ 2*Var(u_l)/u0^2` to 10%."""
    params = scw.compute_analytic_parameters()
    u0 = params["u0"]

    for label, (nx, _, _) in (
        ("coarse_10x474", scw.COARSE_GRID),
        ("fine_20x948", scw.FINE_GRID),
    ):
        det = scw.compute_discrete_dc_depression(nx, params, return_details=True)
        d_chg = float(det["delta_charge"])
        d_den = float(det["delta_density"])
        d_flx = float(det["delta_flux"])

        u_l = det["u_l"]
        rho_l = det["rho_l"]
        u_mean = float(np.sum(rho_l * u_l) / np.sum(rho_l))
        var_u = float(np.sum(rho_l * (u_l - u_mean) ** 2) / np.sum(rho_l))
        lead_2var = float(2.0 * var_u / (u0**2))
        diff_cf = float(d_chg - d_flx)
        rel_err = abs(diff_cf - lead_2var) / lead_2var

        print(
            f"ORDERING grid={label} delta_flux={d_flx:.12e} "
            f"delta_density={d_den:.12e} delta_charge={d_chg:.12e} "
            f"diff_charge_flux={diff_cf:.6e} two_var_over_u0sq={lead_2var:.6e} "
            f"rel_err={rel_err:.6e}",
            flush=True,
        )

        assert d_chg > d_den > d_flx, (
            f"grid={label}: expected strict ordering delta_charge ({d_chg:.12e}) > "
            f"delta_density ({d_den:.12e}) > delta_flux ({d_flx:.12e})"
        )
        assert rel_err <= 0.10, (
            f"grid={label}: |(delta_charge - delta_flux) - 2*Var(u_l)/u0^2| / (2*Var(u_l)/u0^2) "
            f"= {rel_err:.6e} > 0.10"
        )


def test_single_line_and_equal_ul_modes_agree():
    """Verify `delta_charge`, `delta_density`, and `delta_flux` agree to `1e-15` when all `u_l` are equal."""
    params = scw.compute_analytic_parameters()
    nx = scw.COARSE_GRID[0]
    b = scw.BEAM_HALF_THICKNESS_M

    for label, line_x in (
        ("single_line_x0", np.array([0.0], dtype=np.float64)),
        ("symmetric_pair", np.array([-0.5 * b, 0.5 * b], dtype=np.float64)),
    ):
        det = scw.compute_discrete_dc_depression(
            nx, params, line_x=line_x, return_details=True
        )
        u_l = det["u_l"]
        u_spread = float(np.max(u_l) - np.min(u_l))
        d_chg = float(det["delta_charge"])
        d_den = float(det["delta_density"])
        d_flx = float(det["delta_flux"])
        max_diff = max(abs(d_chg - d_den), abs(d_den - d_flx), abs(d_chg - d_flx))

        print(
            f"EQUAL_UL case={label} u_spread={u_spread:.6e} "
            f"delta_flux={d_flx:.16e} delta_density={d_den:.16e} "
            f"delta_charge={d_chg:.16e} max_diff={max_diff:.6e}",
            flush=True,
        )
        assert u_spread <= 1.0e-15 * params["u0"]
        assert max_diff <= 1.0e-15, (
            f"case={label}: max mode difference {max_diff:.6e} > 1e-15"
        )


def test_unknown_weighting_raises_value_error():
    """Verify unknown `weighting` raises `ValueError`."""
    params = scw.compute_analytic_parameters()
    nx = scw.COARSE_GRID[0]

    for bad_weighting in ("unknown", "harmonic", "", "FLUX"):
        with pytest.raises(ValueError, match="weighting") as exc_info:
            scw.compute_discrete_dc_depression(nx, params, weighting=bad_weighting)
        print(
            f"INVALID_WEIGHTING weighting={bad_weighting!r} msg={exc_info.value}",
            flush=True,
        )
        assert "weighting" in str(exc_info.value)


def test_beta_q_and_delta_disc_exact_unchanged_bit_for_bit(monkeypatch):
    """Verify `_single_grid_budget_terms` keeps `delta_disc_exact` and `bq_pred_part` on flux weighting bit-for-bit."""
    params = scw.compute_analytic_parameters()

    for label, (nx, _, nz) in (
        ("coarse_10x474", scw.COARSE_GRID),
        ("fine_20x948", scw.FINE_GRID),
    ):
        for p in (1, 2, 3):
            tols = scw._single_grid_budget_terms(nx, nz, params, shape=p)
            delta_flux = scw.compute_discrete_dc_depression(
                nx, params, weighting="flux", shape=p
            )
            delta_charge = scw.compute_discrete_dc_depression(
                nx, params, weighting="charge", shape=p
            )
            delta_density = scw.compute_discrete_dc_depression(
                nx, params, weighting="density", shape=p
            )

            print(
                f"BUDGET_DELTA grid={label} p={p} "
                f"delta_disc_exact={tols['delta_disc_exact']:.16e} "
                f"delta_flux={delta_flux:.16e} "
                f"delta_disc_charge={tols['delta_disc_charge']:.16e} "
                f"delta_charge={delta_charge:.16e} "
                f"bq_pred_part={tols['bq_pred_part']:.12f}",
                flush=True,
            )

            assert tols["dep_weighting"] == "flux"
            assert tols["delta_disc_exact"] == delta_flux
            assert tols["delta_th_disc"] == delta_flux
            assert tols["delta_disc_flux"] == delta_flux
            assert tols["delta_disc_density"] == delta_density
            assert tols["delta_disc_charge"] == delta_charge
            assert tols["delta_disc_charge"] != tols["delta_disc_exact"]

    # Verify bq_pred_part is bit-for-bit identical to discrete_modal_prediction with explicit flux weighting,
    # and differs if compute_discrete_dc_depression were mutated to return delta_charge by default.
    nx_c, _, nz_c = scw.COARSE_GRID
    tols_c = scw._single_grid_budget_terms(nx_c, nz_c, params)
    orig_fn = scw.compute_discrete_dc_depression

    def _explicit_flux_dep(nx_arg, params_arg, **kwargs):
        kwargs["weighting"] = "flux"
        return orig_fn(nx_arg, params_arg, **kwargs)

    def _mutated_charge_dep(nx_arg, params_arg, **kwargs):
        kwargs["weighting"] = "charge"
        return orig_fn(nx_arg, params_arg, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(scw, "compute_discrete_dc_depression", _explicit_flux_dep)
        bq_flux_explicit = float(
            scw.discrete_modal_prediction("particles", nx_c, nz_c, params)
        )

    with monkeypatch.context() as m:
        m.setattr(scw, "compute_discrete_dc_depression", _mutated_charge_dep)
        bq_if_charge = float(
            scw.discrete_modal_prediction("particles", nx_c, nz_c, params)
        )

    print(
        f"BQ_PRED_CHECK bq_pred_part={tols_c['bq_pred_part']:.16e} "
        f"bq_flux_explicit={bq_flux_explicit:.16e} "
        f"bq_if_charge={bq_if_charge:.16e} "
        f"diff_if_charge={bq_if_charge - tols_c['bq_pred_part']:+.6e}",
        flush=True,
    )
    assert tols_c["bq_pred_part"] == bq_flux_explicit
    assert bq_if_charge != tols_c["bq_pred_part"]

    # Verify measure_empirical_noise populates delta_disc (flux) and delta_disc_charge (charge) bit-for-bit
    n_macro_ens = 256
    b = scw.BEAM_HALF_THICKNESS_M
    k_idx = np.arange(n_macro_ens, dtype=np.float64)
    x_lat_ens = b * (2.0 * (k_idx + 0.5) / float(n_macro_ens) - 1.0)
    expected_ens_flux = scw.compute_discrete_dc_depression(
        nx_c, params, line_x=x_lat_ens, weighting="flux"
    )
    expected_ens_charge = scw.compute_discrete_dc_depression(
        nx_c, params, line_x=x_lat_ens, weighting="charge"
    )
    dz_c = scw.TUNNEL_LENGTH_M / nz_c
    z_c = (np.arange(nz_c - 1, dtype=np.float64) + 0.5) * dz_c
    i1_dummy = (
        params["I1_0"]
        * np.cos(params["beta_q"] * z_c)
        * np.exp(-1j * params["beta_e"] * z_c)
    )
    scw.measure_empirical_noise.cache_clear()
    try:
        with monkeypatch.context() as m:
            m.setattr(
                scw,
                "cached_linearized_beam_drift",
                lambda *a, **kw: {"z_coords": z_c},
            )
            m.setattr(
                scw,
                "cached_particle_drift",
                lambda *a, **kw: {
                    "i1_part": i1_dummy,
                    "i2_part": 1e-5 * i1_dummy,
                    "u_z_mean": params["u0"] * (1.0 - expected_ens_charge),
                },
            )
            m.setattr(
                scw,
                "fit_beta_q_from_envelope",
                lambda *a, **kw: {"beta_q": params["beta_q"], "z0": 0.0},
            )
            noise_info = scw.measure_empirical_noise(scw.COARSE_GRID)
    finally:
        scw.measure_empirical_noise.cache_clear()

    print(
        f"NOISE_DELTA_CHECK delta_disc={noise_info['delta_disc']:.16e} "
        f"expected_flux={expected_ens_flux:.16e} "
        f"delta_disc_charge={noise_info['delta_disc_charge']:.16e} "
        f"expected_charge={expected_ens_charge:.16e}",
        flush=True,
    )
    assert noise_info["delta_disc"] == expected_ens_flux
    assert noise_info["delta_disc_charge"] == expected_ens_charge
    assert noise_info["delta_disc_charge"] > noise_info["delta_disc"]
