# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Milestone P5 (part B) V7 cross-validation of `wakis.particles` against `LinearizedBeam`.

Propagates the same density-modulated planar sheet electron beam through a
smooth parallel-plate PEC drift tunnel twice on identical grids, time steps,
and boundary conditions:

1. As relativistic macroparticles (`ParticleSolver` + `BeamInjector` with
   planar sheet profiles ``"sheet_lattice"`` for deterministic checks and
   ``"sheet"`` for the 8-seed noise ensemble, and modulated current
   ``I(t) = I0 * (1 + m * sin(omega * t))``, ``m << 1``) confined axially by a
   strong uniform guide magnetic field ``B_ext``.
2. As an Eulerian linearized cold-fluid beam (`LinearizedBeam`) evaluated at the
   self-consistent DC space-charge-depressed equilibrium ``(u_dep, rho_dep, gamma_dep)``
   and driven through the public ``inflow_rho1`` boundary parameter.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._space_charge_waves_common import (
    BEAM_HALF_THICKNESS_M,
    CFL_NUMBER,
    COARSE_GRID,
    EPS_WINDOW,
    FINE_GRID,
    GUIDE_BZ_TESLA,
    MODULATION_INDEX,
    OUTFLOW_RAMP_M,
    PARTICLE_SHAPE,
    PERIODIC_LY_M,
    TUNNEL_HALF_WIDTH_M,
    TUNNEL_LENGTH_M,
    _cached_integer_period_dt,
    cached_baseline_pair,
    cached_linearized_beam_drift,
    cached_particle_drift,
    compute_analytic_parameters,
    compute_discrete_dc_depression,
    compute_m_extrapolation,
    compute_spatial_spectrum,
    compute_transverse_modal_operator,
    derived_tolerances,
    discrete_modal_prediction,
    extract_coherent_bunching_i2_ratio,
    fit_beta_q_from_envelope,
    fit_four_wave_reflection,
    fit_lb_parasitic_waves,
    fit_window,
    format_error_budget,
    measure_empirical_noise,
    prediction_slack,
    run_linearized_beam_drift,
    run_particle_beam_drift,
    settle_steps,
    transverse_weight_matrix,
)
from wakis import GridFIT3D, SolverFIT3D
from wakis.beam_fluid import LinearizedBeam, reduction_factor_planar
from wakis.estimate import estimate, preflight
from wakis.particles import BeamInjector, ParticleSolver, Species


class _CicNodeRho0:
    """Callable `LinearizedBeam` `rho0(x, y, z)` equal to the order-`shape` node deposit of `line_x` with same `I0`."""

    def __init__(
        self,
        rho_dep: float,
        line_x: np.ndarray,
        b: float,
        a: float,
        nx: int,
        beam_ref: LinearizedBeam,
        shape: int = PARTICLE_SHAPE,
    ):
        dx = 2.0 * float(a) / int(nx)
        x_nodes = np.linspace(-float(a), float(a), int(nx) + 1)
        n_lines = len(line_x)
        w_mat = transverse_weight_matrix(line_x, int(nx), shape)
        self.w_cic_nodes = np.sum(w_mat, axis=0) * (2.0 * float(b) / (n_lines * dx))
        keep_lines = np.asarray(beam_ref.keep_lines, dtype=bool)
        active_ix = np.flatnonzero(keep_lines)
        w_perp = np.asarray(beam_ref._w_perp_eff[0, :], dtype=np.float64)
        self.rho_nodes = np.zeros(int(nx) + 1, dtype=np.float64)
        # Divide by entrance w_perp where w_perp > 0 so LinearizedBeam._initialize
        # (which evaluates rho0 at grid.x[:Nx] and multiplies by w_perp_lines) produces
        # beam.rho0[0, :] = rho_dep * w_cic_nodes on every active line.
        pos = w_perp > 0.0
        self.rho_nodes[active_ix[pos]] = (
            float(rho_dep) * self.w_cic_nodes[active_ix[pos]] / w_perp[pos]
        )
        self.active_ix = active_ix
        self.x_nodes = x_nodes
        self.dx = dx
        self.rho_peak = float(np.max(np.abs(float(rho_dep) * self.w_cic_nodes)))

    def __call__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        idx = np.clip(
            np.rint((x - self.x_nodes[0]) / self.dx).astype(np.int64),
            0,
            len(self.x_nodes) - 1,
        )
        return self.rho_nodes[idx]

    def __float__(self) -> float:
        return self.rho_peak


def _fibonacci_disk_x_coords(
    n_macro: int, radius: float = BEAM_HALF_THICKNESS_M, x0: float = 0.0
) -> np.ndarray:
    """Return Fibonacci disk x-coordinates (semicircular marginal) for M3 and D3."""
    k_idx = np.arange(n_macro, dtype=np.float64)
    r_k = float(radius) * np.sqrt((k_idx + 0.5) / float(n_macro))
    phi_k = k_idx * np.pi * (3.0 - np.sqrt(5.0))
    dx_k = r_k * np.cos(phi_k)
    dx_k -= float(np.mean(dx_k))
    return float(x0) + dx_k


def _fibonacci_disk_lattice_x(self: BeamInjector, n_macro: int) -> np.ndarray:
    """Method replacement for `BeamInjector._sheet_lattice_x` in M3."""
    return _fibonacci_disk_x_coords(n_macro, radius=self.radius, x0=self.x0)


@lru_cache(maxsize=8)
def _cached_m3_disk_run(
    nx: int,
    nz: int,
    m_mod: float = MODULATION_INDEX,
    shape: int = PARTICLE_SHAPE,
) -> dict:
    """Cached M3 run with `BeamInjector._sheet_lattice_x` patched to Fibonacci disk x-coordinates."""
    lb_res, _ = cached_baseline_pair(int(nx), int(nz), shape=int(shape))
    orig_sheet_lat = BeamInjector._sheet_lattice_x
    BeamInjector._sheet_lattice_x = _fibonacci_disk_lattice_x
    try:
        return run_particle_beam_drift(
            int(nx),
            int(nz),
            beam_ref=lb_res["beam"],
            qm_scale=1.0,
            seed=42,
            m_mod=float(m_mod),
            profile="sheet_lattice",
            shape=int(shape),
        )
    finally:
        BeamInjector._sheet_lattice_x = orig_sheet_lat


def test_preflight_and_resolution_criteria():
    """Verify `wakis.estimate` preflight, resolution rules, and two-wave fit-window span >= 1.25 * lambda_q."""
    params = compute_analytic_parameters(qm_scale=1.0)
    params_mut = compute_analytic_parameters(qm_scale=2.0)

    # Fit window [d_w, (Lz - L_ramp) - d_w] with d_w = ln(1/eps_w)/gamma1 must span
    # at least 1.25 * lambda_q and satisfy kappa_dc > gamma1 for both baseline and |q/m| x 2
    for label, p_cur in (("baseline", params), ("qm_mutation", params_mut)):
        assert p_cur["kappa_dc"] > p_cur["gamma1"], (
            f"{label} identity check failed: kappa_dc={p_cur['kappa_dc']:.6f} <= gamma1={p_cur['gamma1']:.6f}"
        )
        z_min, z_max = fit_window(p_cur, outflow_ramp_m=OUTFLOW_RAMP_M)
        win_len = z_max - z_min
        min_req = 1.25 * p_cur["lambda_q"]
        assert win_len >= min_req, (
            f"{label} fit window z_max - z_min = {win_len:.6f} m < 1.25 * lambda_q = {min_req:.6f} m "
            f"(margin={win_len - min_req:+.6f} m)"
        )

    for nx, ny, nz in (COARSE_GRID, FINE_GRID):
        dx = 2.0 * TUNNEL_HALF_WIDTH_M / nx
        dz = TUNNEL_LENGTH_M / nz

        # >= 20 cells per electron wavelength, vacuum wavelength, and plasma wavelength
        assert params["lambda_e"] / dz >= 20.0, (
            f"lambda_e/dz = {params['lambda_e'] / dz:.2f} < 20.0 "
            f"(margin={params['lambda_e'] / dz - 20.0:+.2f})"
        )
        assert params["lambda_0"] / dz >= 20.0, (
            f"lambda_0/dz = {params['lambda_0'] / dz:.2f} < 20.0 "
            f"(margin={params['lambda_0'] / dz - 20.0:+.2f})"
        )
        assert params["lambda_p"] / dz >= 20.0, (
            f"lambda_p/dz = {params['lambda_p'] / dz:.2f} < 20.0 "
            f"(margin={params['lambda_p'] / dz - 20.0:+.2f})"
        )

        # >= 4 cells across the full sheet-beam thickness 2 * b
        assert (2.0 * BEAM_HALF_THICKNESS_M) / dx >= 4.0, (
            f"2*b/dx = {(2.0 * BEAM_HALF_THICKNESS_M) / dx:.2f} < 4.0 "
            f"(margin={(2.0 * BEAM_HALF_THICKNESS_M) / dx - 4.0:+.2f})"
        )

        # Preflight memory estimate must fit within the 2 GB laptop/memguard cap and L4 GPU
        est_cpu = estimate(
            (nx, ny, nz),
            steps=3500,
            dtype="float64",
            backend="cpu",
            cpml=False,
        )
        assert est_cpu.cells == nx * ny * nz
        assert preflight(est_cpu, "laptop", raise_error=False) == []
        est_gpu = estimate(
            (nx, ny, nz),
            steps=3500,
            dtype="float64",
            backend="gpu",
            cpml=False,
        )
        assert preflight(est_gpu, "g2-standard-16", raise_error=False) == []

    # Step 1.2: assert dz unchanged to 1e-12
    dz_c = TUNNEL_LENGTH_M / COARSE_GRID[2]
    dz_f = TUNNEL_LENGTH_M / FINE_GRID[2]
    dz_prev = 0.186 / 428
    assert abs(dz_c - 2.0 * dz_f) <= 1.0e-12, (
        f"|dz_coarse - 2*dz_fine| = {abs(dz_c - 2.0 * dz_f):.6e} m > 1e-12 m"
    )
    assert abs(dz_c - dz_prev) <= 1.0e-12, (
        f"|dz - dz_prev| = {abs(dz_c - dz_prev):.6e} m > 1e-12 m "
        f"(Lz/474 = {dz_c:.12e} m vs 0.186/428 = {dz_prev:.12e} m; "
        f"exact dz-preserving length is 0.186*474/428 = {0.186 * 474 / 428:.12f} m)"
    )


def test_injected_sheet_geometry():
    """Identity (geometry) check for `BeamInjector` `"sheet_lattice"` and `"sheet"` profiles on coarse CPU grid (3 steps)."""
    params = compute_analytic_parameters()
    nx, ny, nz = COARSE_GRID
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    ly = PERIODIC_LY_M
    lz = TUNNEL_LENGTH_M
    n_macro = 16
    k_idx = np.arange(n_macro, dtype=np.float64)
    stratum_lo = -b + 2.0 * b * k_idx / n_macro
    stratum_hi = -b + 2.0 * b * (k_idx + 1.0) / n_macro
    x_lattice_expected = b * (2.0 * (k_idx + 0.5) / n_macro - 1.0)

    print(flush=True)
    profile_slices: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for profile in ("sheet_lattice", "sheet"):
        grid = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, ny, nz, verbose=0)
        solver = SolverFIT3D(
            grid,
            cfln=CFL_NUMBER,
            bc_low=["pec", "periodic", "pec"],
            bc_high=["pec", "periodic", "pec"],
            use_gpu=False,
            verbose=0,
        )
        sp = Species(name="electron")
        inj = BeamInjector(
            species=sp,
            z0=0.0,
            current=params["I0"],
            gamma=params["gamma0"],
            profile=profile,
            radius=b,
            x0=0.0,
            y0=0.0,
            n_macro_per_step=n_macro,
            seed=42,
        )
        ps = ParticleSolver(
            solver,
            species=[sp],
            injectors=[inj],
            shape=PARTICLE_SHAPE,
            B_ext=(0.0, 0.0, GUIDE_BZ_TESLA),
        ).attach()

        injected_slices: list[tuple[np.ndarray, np.ndarray]] = []
        orig_pre_push = ps._mpi_pre_push_hook

        def _capture_injected(
            species_obj, _orig=orig_pre_push, _slices=injected_slices
        ):
            _orig(species_obj)
            x_new = np.asarray(species_obj.x[-n_macro:], dtype=np.float64).copy()
            y_new = np.asarray(species_obj.y[-n_macro:], dtype=np.float64).copy()
            _slices.append((x_new, y_new))

        ps._mpi_pre_push_hook = _capture_injected
        for _ in range(3):
            solver.one_step()

        profile_slices[profile] = injected_slices
        all_x = np.concatenate([s[0] for s in injected_slices])
        all_y = np.concatenate([s[1] for s in injected_slices])
        y_max_abs = float(np.max(np.abs(all_y)))
        x_min = float(np.min(all_x))
        x_max = float(np.max(all_x))

        strata_ok = bool(
            len(injected_slices) == 3
            and np.all(all_y == 0.0)
            and np.all(np.abs(all_x) <= b)
        )
        for x_step, y_step in injected_slices:
            if not (np.all(y_step == 0.0) and np.all(np.abs(x_step) <= b)):
                strata_ok = False
            if profile == "sheet":
                if not np.all((x_step >= stratum_lo) & (x_step < stratum_hi)):
                    strata_ok = False
            else:
                if not np.all(np.abs(x_step - x_lattice_expected) <= 1e-15 * b):
                    strata_ok = False

        print(
            f"INJCHECK profile={profile} n={n_macro} y_max_abs={y_max_abs:.6e} "
            f"x_min={x_min:.6e} x_max={x_max:.6e} strata_ok={strata_ok}",
            flush=True,
        )

    for profile, injected_slices in profile_slices.items():
        assert len(injected_slices) == 3
        for step_idx, (x_step, y_step) in enumerate(injected_slices):
            assert np.all(y_step == 0.0), (
                f"Step {step_idx} ({profile}): max|y|={np.max(np.abs(y_step)):.3e} != 0.0"
            )
            assert np.all(np.abs(x_step) <= b), (
                f"Step {step_idx} ({profile}): max|x|={np.max(np.abs(x_step)):.6e} > b={b:.6e}"
            )
            if profile == "sheet":
                assert np.all((x_step >= stratum_lo) & (x_step < stratum_hi)), (
                    f"Step {step_idx} (sheet): particles outside strata"
                )
            else:
                max_lat_err = float(np.max(np.abs(x_step - x_lattice_expected)))
                assert max_lat_err <= 1e-15 * b, (
                    f"Step {step_idx} (sheet_lattice): max|x - x_lat|={max_lat_err:.3e} > "
                    f"{1e-15 * b:.3e} (margin={1e-15 * b - max_lat_err:+.3e})"
                )


def test_discrete_modal_operator_unit():
    """Unit test (Step 2.4, 2.7 & Step 5, fast, no simulation): verify G(beta_e) eigenvalues and DC shear shift."""
    params = compute_analytic_parameters()
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    first_exc: Exception | None = None
    err_400 = np.nan
    bound_400 = np.nan
    ratio_10_20 = np.nan
    ratio_20_40 = np.nan
    band_ratio_10_20 = np.nan
    band_ratio_20_40 = np.nan

    print(flush=True)
    try:
        r_sq_exact = float(
            reduction_factor_planar(params["omega"], params["u_dep"], b, a=a) ** 2
        )

        op_10 = compute_transverse_modal_operator(
            "lb", 10, None, params, k_val=params["beta_e"]
        )
        op_20 = compute_transverse_modal_operator(
            "lb", 20, None, params, k_val=params["beta_e"]
        )
        op_40 = compute_transverse_modal_operator(
            "lb", 40, None, params, k_val=params["beta_e"]
        )
        op_400 = compute_transverse_modal_operator(
            "lb", 400, None, params, k_val=params["beta_e"]
        )

        lam1_10 = float(op_10["evals"][0])
        lam1_20 = float(op_20["evals"][0])
        lam1_40 = float(op_40["evals"][0])
        lam1_400 = float(op_400["evals"][0])

        diff_10 = float(lam1_10 - r_sq_exact)
        diff_20 = float(lam1_20 - r_sq_exact)
        diff_40 = float(lam1_40 - r_sq_exact)
        err_400 = float(abs(lam1_400 - r_sq_exact))

        h_20 = 1.0 / 20.0
        h_40 = 1.0 / 40.0
        h_400 = 1.0 / 400.0

        # Estimate the leading O(h^2) coefficient C and O(h^4) Richardson remainder from nx = 10, 20, 40:
        # From the finest coarse level h_40 = 1/40, C = diff_40 / h_40^2, and from the two-level
        # Richardson extrapolants lam_rich(2h, h) = (4 * lam_1(h) - lam_1(2h)) / 3 = R^2 - (4/3) * D * h^4 + O(h^6),
        # lam_rich(h_20, h_40) - lam_rich(h_10, h_20) = (15/4) * D * h_20^4 gives the O(h^4) coefficient D.
        c_h2 = float(diff_40 / (h_40**2))
        lam_rich_10_20 = float((4.0 * lam1_20 - lam1_10) / 3.0)
        lam_rich_20_40 = float((4.0 * lam1_40 - lam1_20) / 3.0)
        rich_rem_20_40 = float(abs(lam_rich_20_40 - lam_rich_10_20) / 15.0)
        d_h4 = float(4.0 * (lam_rich_20_40 - lam_rich_10_20) / (15.0 * (h_20**4)))

        # Ratio deviation implied by the O(h^4) remainder:
        #   e(2h) / e(h) = 4 * (1 + 4*(D/C)*h^2) / (1 + (D/C)*h^2) = 4 + 12 * (D/C) * h^2 + O(h^4),
        # so |ratio - 4| <= 12 * |D / C| * h^2.
        ratio_10_20 = float(diff_10 / diff_20)
        ratio_20_40 = float(diff_20 / diff_40)
        band_ratio_10_20 = float(12.0 * abs(d_h4 / c_h2) * (h_20**2))
        band_ratio_20_40 = float(12.0 * abs(d_h4 / c_h2) * (h_40**2))

        # nx = 400 bound from coarse levels: |C| * (1/400)^2 + |remainder term at 400|
        lead_400 = float(abs(c_h2) * (h_400**2))
        rem_400 = float(abs(d_h4) * (h_400**4))
        bound_400 = float(lead_400 + rem_400)

        dx_10 = 2.0 * a / 10.0
        dx_20 = 2.0 * a / 20.0
        dx_40 = 2.0 * a / 40.0
        o2_est_10 = -float(
            r_sq_exact * (1.0 - r_sq_exact) * (params["T1"] * dx_10) ** 2 / 12.0
        )
        o2_est_20 = -float(
            r_sq_exact * (1.0 - r_sq_exact) * (params["T1"] * dx_20) ** 2 / 12.0
        )
        o2_est_40 = -float(
            r_sq_exact * (1.0 - r_sq_exact) * (params["T1"] * dx_40) ** 2 / 12.0
        )
        print(
            f"DIAG id=modal_unit r_sq_analytic={r_sq_exact:.8f} "
            f"lam1_nx10={lam1_10:.8f} diff_nx10={diff_10:+.6e} o2_est_nx10={o2_est_10:+.6e} "
            f"lam1_nx20={lam1_20:.8f} diff_nx20={diff_20:+.6e} o2_est_nx20={o2_est_20:+.6e} "
            f"lam1_nx40={lam1_40:.8f} diff_nx40={diff_40:+.6e} o2_est_nx40={o2_est_40:+.6e} "
            f"C_h2={c_h2:+.6e} D_h4={d_h4:+.6e} rich_rem_20_40={rich_rem_20_40:.6e} "
            f"ratio_10_20={ratio_10_20:.6f} ratio_dev_10_20={abs(ratio_10_20 - 4.0):.6e} band_ratio_10_20={band_ratio_10_20:.6e} "
            f"ratio_20_40={ratio_20_40:.6f} ratio_dev_20_40={abs(ratio_20_40 - 4.0):.6e} band_ratio_20_40={band_ratio_20_40:.6e} "
            f"lam1_nx400={lam1_400:.8f} err_nx400={err_400:.6e} "
            f"lead_400={lead_400:.6e} rem_400={rem_400:.6e} bound_nx400={bound_400:.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=modal_unit status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    try:
        nx_c, _, nz_c = COARSE_GRID
        nx_f, _, nz_f = FINE_GRID
        tols_c = derived_tolerances(nx_c, nz_c, params)
        tols_f = derived_tolerances(nx_f, nz_f, params)
        print(
            f"DIAG id=shear bq_shear_coarse={tols_c['bq_pred_part']:.6f} "
            f"bq_unif_coarse={tols_c['bq_pred_part_unif']:.6f} "
            f"shift_coarse={tols_c['shear_shift']:+.6e} "
            f"rel_shift_coarse={tols_c['shear_rel']:+.6e} "
            f"bq_shear_fine={tols_f['bq_pred_part']:.6f} "
            f"bq_unif_fine={tols_f['bq_pred_part_unif']:.6f} "
            f"shift_fine={tols_f['shear_shift']:+.6e} "
            f"rel_shift_fine={tols_f['shear_rel']:+.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=shear status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    assert err_400 <= bound_400, (
        f"nx=400 LB modal eigenvalue |lambda_1 - R^2| = {err_400:.6e} > "
        f"|C|*(1/400)^2 + |rem_400| = {bound_400:.6e} (margin={bound_400 - err_400:+.6e})"
    )
    # Verify O(h^2) second-order convergence within the ratio deviation implied by the O(h^4) remainder:
    # |e(2h)/e(h) - 4| <= 12 * |D / C| * h^2
    assert abs(ratio_10_20 - 4.0) <= band_ratio_10_20, (
        f"nx=10/20 eigenvalue error ratio |{ratio_10_20:.6f} - 4| = {abs(ratio_10_20 - 4.0):.6e} > "
        f"band_ratio_10_20 = {band_ratio_10_20:.6e}"
    )
    assert abs(ratio_20_40 - 4.0) <= band_ratio_20_40, (
        f"nx=20/40 eigenvalue error ratio |{ratio_20_40:.6f} - 4| = {abs(ratio_20_40 - 4.0):.6e} > "
        f"band_ratio_20_40 = {band_ratio_20_40:.6e}"
    )

    # Identity check: after LinearizedBeam.attach(solver) with _CicNodeRho0,
    # beam.rho0[0, :] equals rho_dep * w_cic_nodes on active lines to 1e-12 relative.
    grid_10 = GridFIT3D(
        -a,
        a,
        -0.5 * PERIODIC_LY_M,
        0.5 * PERIODIC_LY_M,
        0.0,
        TUNNEL_LENGTH_M,
        10,
        1,
        8,
        verbose=0,
    )
    solver_10 = SolverFIT3D(
        grid_10,
        cfln=CFL_NUMBER,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_gpu=False,
        verbose=0,
    )
    beam_ref_10 = LinearizedBeam(
        u0=params["u_dep"],
        rho0=params["rho_dep"],
        gamma0=params["gamma_dep"],
        axis="z",
        half_thickness=b,
        solver=solver_10,
    )
    cic_rho0 = _CicNodeRho0(
        params["rho_dep"],
        _fibonacci_disk_x_coords(16, radius=b),
        b,
        a,
        10,
        beam_ref_10,
    )
    beam_cic = LinearizedBeam(
        u0=params["u_dep"],
        rho0=cic_rho0,
        gamma0=params["gamma_dep"],
        axis="z",
        half_thickness=b,
    ).attach(solver_10)
    assert np.allclose(
        beam_cic.rho0[0, :],
        params["rho_dep"] * cic_rho0.w_cic_nodes[cic_rho0.active_ix],
        rtol=1e-12,
        atol=0.0,
    ), "Identity check failed: beam.rho0[0, :] != rho_dep * w_cic_nodes on active lines"

    if first_exc is not None:
        raise first_exc


def test_shape1_identity_and_wall_stencil_bounds():
    """Verify p=1 path matches head 7100c420 formulas (rtol=0, atol=1e-15) and no p<=3 stencil reaches a wall node."""
    from scipy.constants import c as c_light
    from scipy.constants import e as e_charge
    from scipy.constants import epsilon_0, m_e
    from scipy.stats import t as student_t

    from examples._space_charge_waves_common import (
        DRIVE_FREQ_HZ,
        NOISE_SEEDS,
        RISE_PERIODS,
        _grid_gamma1_disc_and_dw,
    )

    params = compute_analytic_parameters()
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    lz = TUNNEL_LENGTH_M

    # 1. Wall stencil check: for |x| <= b on both coarse and fine grids, no p <= 3 stencil reaches x = +-a
    from wakis.particles.shapes import bspline_weights_1d

    x_probe = np.linspace(-b, b, 257)
    for nx in (COARSE_GRID[0], FINE_GRID[0]):
        dx_probe = 2.0 * a / float(nx)
        xi_probe = (x_probe + a) / dx_probe
        for p in (1, 2, 3):
            i0_probe, weights_probe = bspline_weights_1d(
                xi_probe, p, staggered=False, xp=np
            )
            for offset in range(len(weights_probe)):
                node_idx = i0_probe + offset
                assert np.all((node_idx >= 1) & (node_idx <= nx - 1)), (
                    f"nx={nx}, p={p}, offset={offset}: stencil reaches PEC wall node "
                    f"(min={int(np.min(node_idx))}, max={int(np.max(node_idx))})"
                )
            w_probe = transverse_weight_matrix(x_probe, nx, p)
            assert w_probe.shape == (len(x_probe), nx + 1)
            assert np.all(w_probe[:, 0] == 0.0), (
                f"nx={nx}, p={p}: non-zero weight on left PEC wall x=-a"
            )
            assert np.all(w_probe[:, nx] == 0.0), (
                f"nx={nx}, p={p}: non-zero weight on right PEC wall x=+a"
            )
            np.testing.assert_allclose(
                np.sum(w_probe[:, 1:-1], axis=1), 1.0, rtol=0.0, atol=2e-15
            )

    # 2. Reference formulas copied verbatim from head 7100c420
    def _ref_dc_depression(nx: int, p_dict: dict, line_x=None, line_w=None):
        dx = 2.0 * a / nx
        x_nodes = np.linspace(-a, a, nx + 1)
        if line_x is None:
            n_macro = 16 if nx == COARSE_GRID[0] else 32
            k_idx = np.arange(n_macro, dtype=np.float64)
            x_sheet = b * (2.0 * (k_idx + 0.5) / n_macro - 1.0)
        else:
            x_sheet = np.asarray(line_x, dtype=np.float64)
            n_macro = len(x_sheet)
        if line_w is None:
            rho_l = np.full(n_macro, 1.0 / n_macro, dtype=np.float64)
        else:
            rho_l = np.asarray(line_w, dtype=np.float64)
            rho_l = rho_l / np.sum(rho_l)
        w_mat = np.maximum(1.0 - np.abs(x_sheet[:, None] - x_nodes[None, :]) / dx, 0.0)
        w_node = np.sum(rho_l[:, None] * w_mat, axis=0) * (2.0 * b / dx)
        n_int = nx - 1
        lap = (
            np.diag(np.full(n_int, -2.0))
            + np.diag(np.ones(n_int - 1), 1)
            + np.diag(np.ones(n_int - 1), -1)
        ) / (dx**2)
        qm_scale = p_dict.get("qm_scale", 1.0)
        mass_eff = m_e / qm_scale
        u0 = p_dict["u0"]
        gamma0 = p_dict["gamma0"]
        rho0 = abs(p_dict["rho0"])
        phi_0 = np.zeros(nx + 1, dtype=np.float64)
        phi_0[1:-1] = np.linalg.solve(lap, -rho0 * w_node[1:-1] / epsilon_0)
        phi_l0 = np.abs(w_mat @ phi_0)
        delta0_disc = float(
            np.sum(rho_l * phi_l0) / (gamma0**3 * mass_eff * u0**2 / e_charge)
        )

        def _iter_fp(mode: str):
            delta_val = float(delta0_disc)
            u_l = np.full(n_macro, u0 * (1.0 - delta_val), dtype=np.float64)
            gamma_l = np.full(n_macro, gamma0, dtype=np.float64)
            phi_l = phi_l0.copy()
            phi_cur = np.zeros(nx + 1, dtype=np.float64)
            for _ in range(200):
                rho_dep_l = rho_l * (u0 / u_l)
                w_node_dep = np.sum(rho_dep_l[:, None] * w_mat, axis=0) * (2.0 * b / dx)
                phi_cur[1:-1] = np.linalg.solve(
                    lap, -rho0 * w_node_dep[1:-1] / epsilon_0
                )
                phi_l = np.abs(w_mat @ phi_cur)
                gamma_l = gamma0 - e_charge * phi_l / (mass_eff * c_light**2)
                u_l = c_light * np.sqrt(1.0 - 1.0 / (gamma_l**2))
                w_l = (rho_l * u_l) if mode == "flux" else rho_l
                delta_next = 1.0 - float(np.sum(w_l * u_l) / (u0 * np.sum(w_l)))
                if abs(delta_next - delta_val) < 1e-14:
                    delta_val = float(delta_next)
                    break
                delta_val = float(delta_next)
            return float(delta_val), u_l, gamma_l, phi_l

        delta_flux, u_l_flux, gamma_l_flux, phi_l_flux = _iter_fp("flux")
        delta_density, _, _, _ = _iter_fp("density")
        return {
            "delta_disc": float(delta_flux),
            "delta_flux": float(delta_flux),
            "delta_density": float(delta_density),
            "u_l": u_l_flux,
            "gamma_l": gamma_l_flux,
            "phi_l": phi_l_flux,
            "rho_l": rho_l,
            "x_sheet": x_sheet,
            "w_mat": w_mat,
        }

    def _ref_modal_op_part(
        nx: int,
        nz: int | None,
        p_dict: dict,
        k_val=None,
        line_x=None,
        line_w=None,
        use_omega_f: bool = True,
    ):
        dx = 2.0 * a / nx
        n_int = nx - 1
        lap = (
            np.diag(np.full(n_int, -2.0))
            + np.diag(np.ones(n_int - 1), 1)
            + np.diag(np.ones(n_int - 1), -1)
        ) / (dx**2)
        if k_val is None:
            k_val = float(p_dict["beta_e"])
        if nz is None:
            kappa = float(k_val)
            omega_tau = float(p_dict["omega"])
        else:
            dz = lz / nz
            kappa = float((2.0 / dz) * np.sin(0.5 * k_val * dz))
            if use_omega_f:
                dt_grid, _ = _cached_integer_period_dt(int(nx), int(nz))
                omega_tau = float(
                    (2.0 / dt_grid) * np.sin(0.5 * p_dict["omega"] * dt_grid)
                )
            else:
                omega_tau = float(p_dict["omega"])
        tau_y2 = max(kappa**2 - (omega_tau / c_light) ** 2, 1e-12)
        a_mat = -lap + tau_y2 * np.eye(n_int, dtype=np.float64)
        if line_x is None:
            n_macro = 16 if nx == COARSE_GRID[0] else 32
            k_idx = np.arange(n_macro, dtype=np.float64)
            x_p = b * (2.0 * (k_idx + 0.5) / n_macro - 1.0)
        else:
            x_p = np.asarray(line_x, dtype=np.float64)
            n_macro = len(x_p)
        if line_w is None:
            d_vec = np.full(n_macro, 1.0 / n_macro, dtype=np.float64)
        else:
            d_vec = np.asarray(line_w, dtype=np.float64)
            d_vec = d_vec / np.sum(d_vec)
        d_mat = np.diag(d_vec)
        x_int = np.linspace(-a, a, nx + 1)[1:-1]
        w_cic = np.maximum(1.0 - np.abs(x_int[:, None] - x_p[None, :]) / dx, 0.0)
        s_mat = w_cic * ((2.0 * b / n_macro) / dx)
        g_mat = (
            (n_macro**2 * dx / (2.0 * b))
            * tau_y2
            * (s_mat.T @ np.linalg.solve(a_mat, s_mat) @ d_mat)
        )
        s_left = np.sqrt(d_vec)
        s_right = 1.0 / np.sqrt(d_vec)
        m_sym = (s_left[:, None] * g_mat) * s_right[None, :]
        m_sym = 0.5 * (m_sym + m_sym.T)
        evals, qvecs = np.linalg.eigh(m_sym)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        qvecs = qvecs[:, order]
        return {
            "evals": evals,
            "y_right": s_right[:, None] * qvecs,
            "y_left": s_left[:, None] * qvecs,
            "d_vec": d_vec,
            "G": g_mat,
            "S": s_mat,
            "lap": lap,
            "dx": dx,
            "line_x": x_p,
        }

    def _ref_modal_pred_part(
        nx: int,
        nz: int,
        p_dict: dict,
        line_x=None,
        line_w=None,
        entrance_profile=None,
        include_shear: bool = True,
        include_sz: bool = True,
        use_omega_f: bool = True,
        drop_highest_mode: bool = False,
        win_shift: float = 0.0,
        omega_p_scale: float = 1.0,
        return_details: bool = False,
    ):
        dz = lz / nz
        dt, _ = _cached_integer_period_dt(int(nx), int(nz))
        omega = p_dict["omega"]
        dep_info = _ref_dc_depression(nx, p_dict, line_x=line_x, line_w=line_w)
        u_l = dep_info["u_l"]
        gamma_l = dep_info["gamma_l"]
        delta_disc = float(dep_info["delta_disc"])
        u0 = p_dict["u0"]
        u_bar = float(u0 * (1.0 - delta_disc))
        gamma_bar = float(1.0 / np.sqrt(1.0 - (u_bar / c_light) ** 2))
        beta_e_part = float(omega / u_bar)
        omega_p0 = p_dict["omega_p0"] * float(omega_p_scale)
        omega_p_dt = (2.0 / dt) * np.sin(0.5 * omega * dt)
        op0 = _ref_modal_op_part(
            nx,
            nz,
            p_dict,
            k_val=beta_e_part,
            line_x=line_x,
            line_w=line_w,
            use_omega_f=use_omega_f,
        )
        evals_be = op0["evals"]
        d_vec = op0["d_vec"]
        omega_p_l_sq = (omega_p0**2) * (u0 / u_l)
        if include_shear:
            sig_vec = omega_p_l_sq / (gamma_l**3 * u_l**2)
            be_vec = omega_p_dt / u_l
        else:
            omega_p_bar_sq = (omega_p0**2) * (u0 / u_bar)
            sig_vec = np.full(
                len(u_l),
                omega_p_bar_sq / (gamma_bar**3 * u_bar**2),
                dtype=np.float64,
            )
            be_vec = np.full(len(u_l), omega_p_dt / u_bar, dtype=np.float64)
        s_left = np.sqrt(d_vec / sig_vec)
        s_right = np.sqrt(sig_vec / d_vec)

        def _eig_at_k(k_val: float):
            op_k = _ref_modal_op_part(
                nx,
                nz,
                p_dict,
                k_val=k_val,
                line_x=line_x,
                line_w=line_w,
                use_omega_f=use_omega_f,
            )
            sz_sq = float(np.sinc(0.5 * k_val * dz / np.pi) ** 2) if include_sz else 1.0
            g_tilde = (sz_sq * sig_vec)[:, None] * op_k["G"]
            m_sym = (s_left[:, None] * g_tilde) * s_right[None, :]
            m_sym = 0.5 * (m_sym + m_sym.T)
            ev, qv = np.linalg.eigh(m_sym)
            idx = np.argsort(ev)[::-1]
            ev = np.maximum(ev[idx], 0.0)
            qv = qv[:, idx]
            return ev, s_right[:, None] * qv, s_left[:, None] * qv

        ev0, y_r0, y_l0 = _eig_at_k(beta_e_part)
        active = ev0 > 1e-12 * ev0[0]
        n_act = int(np.sum(active))
        if entrance_profile is None:
            entrance = np.ones(len(d_vec), dtype=np.float64)
        else:
            ep = np.asarray(entrance_profile, dtype=np.float64)
            entrance = (ep / np.sum(ep)) / np.maximum(d_vec, 1e-30)
        bq_modes = np.zeros(n_act, dtype=np.float64)
        kc_modes = np.zeros(n_act, dtype=np.float64)
        c_modes = np.zeros(n_act, dtype=np.float64)
        for j in range(n_act):
            yj = y_r0[:, j]
            lj = y_l0[:, j]
            be_j = float(np.sum((lj * yj) * be_vec))
            bq_init = float(np.sqrt(ev0[j]))
            km = be_j - bq_init
            kp = be_j + bq_init
            for _ in range(12):
                evm, _, _ = _eig_at_k(km)
                evp, _, _ = _eig_at_k(kp)
                km_next = be_j - float(np.sqrt(evm[j]))
                kp_next = be_j + float(np.sqrt(evp[j]))
                if abs(km_next - km) < 1e-13 and abs(kp_next - kp) < 1e-13:
                    km, kp = km_next, kp_next
                    break
                km, kp = km_next, kp_next
            bq_modes[j] = 0.5 * (kp - km)
            kc_modes[j] = 0.5 * (kp + km)
            c_modes[j] = float(np.dot(d_vec, yj) * np.dot(lj, entrance))
        c_use = c_modes.copy()
        if drop_highest_mode:
            excited = np.where(np.abs(c_use) > 1e-12)[0]
            if len(excited) > 1:
                c_use[int(excited[-1])] = 0.0
        z_eval = (np.arange(nz, dtype=np.float64) + 0.5) * dz
        i1_model = p_dict["I1_0"] * np.sum(
            c_use[:, None]
            * np.cos(np.outer(bq_modes, z_eval))
            * np.exp(-1j * np.outer(kc_modes, z_eval)),
            axis=0,
        )
        z_min, z_max = fit_window(p_dict)
        fit_res = fit_beta_q_from_envelope(
            z_eval,
            i1_model,
            float(bq_modes[0]),
            z_min=z_min + float(win_shift),
            z_max=z_max + float(win_shift),
        )
        bq_pred = float(fit_res["beta_q"])
        if return_details:
            return {
                "beta_q": bq_pred,
                "evals": evals_be,
                "bq_modes": bq_modes,
                "kc_modes": kc_modes,
                "c_modes": c_modes,
                "z_coords": z_eval,
                "i1_model": i1_model,
                "fit": fit_res,
            }
        return bq_pred

    for nx, _, nz in (COARSE_GRID, FINE_GRID):
        dt_grid, _ = _cached_integer_period_dt(int(nx), int(nz))
        dz = lz / nz
        # Compare DC depression
        dep_ref = _ref_dc_depression(nx, params)
        dep_p1 = compute_discrete_dc_depression(
            nx, params, return_details=True, shape=1
        )
        for key in (
            "delta_disc",
            "delta_flux",
            "delta_density",
            "u_l",
            "gamma_l",
            "phi_l",
            "rho_l",
            "x_sheet",
            "w_mat",
        ):
            np.testing.assert_allclose(dep_p1[key], dep_ref[key], rtol=0.0, atol=1e-15)

        # Compare transverse modal operator
        op_ref = _ref_modal_op_part(nx, nz, params)
        op_p1 = compute_transverse_modal_operator("particles", nx, nz, params, shape=1)
        for key in (
            "evals",
            "y_right",
            "y_left",
            "d_vec",
            "G",
            "S",
            "lap",
            "dx",
            "line_x",
        ):
            np.testing.assert_allclose(op_p1[key], op_ref[key], rtol=0.0, atol=1e-15)

        # Compare discrete modal prediction
        pred_ref = _ref_modal_pred_part(nx, nz, params, return_details=True)
        pred_p1 = discrete_modal_prediction(
            "particles", nx, nz, params, return_details=True, shape=1
        )
        for key in (
            "beta_q",
            "evals",
            "bq_modes",
            "kc_modes",
            "c_modes",
            "z_coords",
            "i1_model",
        ):
            np.testing.assert_allclose(
                pred_p1[key], pred_ref[key], rtol=0.0, atol=1e-15
            )

        # Compare settle_steps
        k_slow_ref = float(np.max(pred_ref["kc_modes"] + pred_ref["bq_modes"]))
        v_min_ref = float(params["omega"] / k_slow_ref)
        d_w_ref, _ = fit_window(params)
        t_settle_ref = RISE_PERIODS / DRIVE_FREQ_HZ + (lz + d_w_ref) / v_min_ref
        settle_ref = int(np.ceil(t_settle_ref / dt_grid))
        settle_p1 = settle_steps("particles", nx, nz, params, dt_grid, shape=1)
        assert settle_p1 == settle_ref

        # Compare prediction_slack and derived_tolerances
        slack_p1 = prediction_slack("particles", nx, nz, params, shape=1)
        bq_unif_ref = float(
            _ref_modal_pred_part(nx, nz, params, include_shear=False, include_sz=True)
        )
        bq_no_sz_ref = float(
            _ref_modal_pred_part(nx, nz, params, include_shear=True, include_sz=False)
        )
        bq_omega_cont_ref = float(
            _ref_modal_pred_part(nx, nz, params, use_omega_f=False)
        )
        np.testing.assert_allclose(
            slack_p1["bq_pred"], pred_ref["beta_q"], rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            slack_p1["bq_pred_unif"], bq_unif_ref, rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            slack_p1["bq_pred_no_sz"], bq_no_sz_ref, rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            slack_p1["s_sym"],
            abs(pred_ref["beta_q"] - bq_omega_cont_ref) / pred_ref["beta_q"],
            rtol=0.0,
            atol=1e-15,
        )

        tols_p1 = derived_tolerances(nx, nz, params, shape=1)
        be_half = 0.5 * params["beta_e"] * dz
        bq_half = 0.5 * params["beta_q"] * dz
        amp_pred_ref = float(np.sinc(be_half / np.pi) * np.cos(bq_half))
        _, _, _ = _grid_gamma1_disc_and_dw(int(nx), int(nz), params["omega"])
        tcrit_ref = float(student_t.ppf(0.99865, len(NOISE_SEEDS) - 1))
        np.testing.assert_allclose(
            tols_p1["bq_pred_part"], pred_ref["beta_q"], rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            tols_p1["delta_disc_exact"], dep_ref["delta_disc"], rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            tols_p1["amp_pred_part"], amp_pred_ref, rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(tols_p1["tcrit"], tcrit_ref, rtol=0.0, atol=1e-15)

        # Compare _CicNodeRho0 at shape=1 against reference formula
        grid_tmp = GridFIT3D(
            -a,
            a,
            -0.5 * PERIODIC_LY_M,
            0.5 * PERIODIC_LY_M,
            0.0,
            lz,
            int(nx),
            1,
            8,
            verbose=0,
        )
        solver_tmp = SolverFIT3D(
            grid_tmp,
            cfln=CFL_NUMBER,
            bc_low=["pec", "periodic", "pec"],
            bc_high=["pec", "periodic", "pec"],
            use_gpu=False,
            verbose=0,
        )
        beam_tmp = LinearizedBeam(
            u0=params["u_dep"],
            rho0=params["rho_dep"],
            gamma0=params["gamma_dep"],
            axis="z",
            half_thickness=b,
            solver=solver_tmp,
        )
        x_fib = _fibonacci_disk_x_coords(16 if nx == COARSE_GRID[0] else 32, radius=b)
        cic_p1 = _CicNodeRho0(
            params["rho_dep"], x_fib, b, a, int(nx), beam_tmp, shape=1
        )
        dx_cur = 2.0 * a / int(nx)
        x_nodes_cur = np.linspace(-a, a, int(nx) + 1)
        w_mat_ref = np.maximum(
            1.0 - np.abs(x_fib[:, None] - x_nodes_cur[None, :]) / dx_cur, 0.0
        )
        w_cic_nodes_ref = np.sum(w_mat_ref, axis=0) * (2.0 * b / (len(x_fib) * dx_cur))
        np.testing.assert_allclose(
            cic_p1.w_cic_nodes, w_cic_nodes_ref, rtol=0.0, atol=1e-15
        )


def test_shape2_modal_prediction_and_sz_symbol():
    """Verify shape=2 modal prediction differs from shape=1 and its S_z(k)^2 factor equals sinc(k*dz/2)^(2p)."""
    params = compute_analytic_parameters()
    nx, _, nz = COARSE_GRID
    dz = TUNNEL_LENGTH_M / nz

    bq_p1 = float(discrete_modal_prediction("particles", nx, nz, params, shape=1))
    bq_p2 = float(discrete_modal_prediction("particles", nx, nz, params, shape=2))
    bq_p3 = float(discrete_modal_prediction("particles", nx, nz, params, shape=3))
    print(
        f"\nSHAPE_PRED grid=coarse bq_p1={bq_p1:.8f} bq_p2={bq_p2:.8f} bq_p3={bq_p3:.8f} "
        f"diff_p2_p1={bq_p2 - bq_p1:+.6e}",
        flush=True,
    )
    assert bq_p2 != bq_p1, f"shape=2 prediction {bq_p2:.8f} equals shape=1 {bq_p1:.8f}"

    # Verify the axial symbol S_z(k)^2 = sinc(k * dz / 2)^(2p) independently for p = 1, 2, 3
    for p in (1, 2, 3):
        dep_p = compute_discrete_dc_depression(nx, params, return_details=True, shape=p)
        u_bar_p = float(params["u0"] * (1.0 - float(dep_p["delta_disc"])))
        k_test = float(params["omega"] / u_bar_p)
        sz_sq_expected = float(np.sinc(0.5 * k_test * dz / np.pi) ** (2 * p))

        # When include_shear=False, G_tilde(k) = S_z(k)^2 * sigma_bar * G(k), so the ratio of
        # fundamental mode wavenumbers (or eigenvalues before the k-iteration) with include_sz=True
        # vs include_sz=False isolates S_z(k)^2. Check both the direct operator relation and the
        # entrance amplitude symbol sinc(beta_e * dz / 2)^p * cos(beta_q * dz / 2).
        det_sz = discrete_modal_prediction(
            "particles",
            nx,
            nz,
            params,
            include_shear=False,
            include_sz=True,
            return_details=True,
            shape=p,
        )
        det_nosz = discrete_modal_prediction(
            "particles",
            nx,
            nz,
            params,
            include_shear=False,
            include_sz=False,
            return_details=True,
            shape=p,
        )
        km_sz = float(det_sz["kc_modes"][0] - det_sz["bq_modes"][0])
        kp_sz = float(det_sz["kc_modes"][0] + det_sz["bq_modes"][0])
        km_nosz = float(det_nosz["kc_modes"][0] - det_nosz["bq_modes"][0])
        kp_nosz = float(det_nosz["kc_modes"][0] + det_nosz["bq_modes"][0])
        op_km = compute_transverse_modal_operator(
            "particles", nx, nz, params, k_val=km_sz, shape=p
        )
        op_kp = compute_transverse_modal_operator(
            "particles", nx, nz, params, k_val=kp_sz, shape=p
        )
        op_km_nosz = compute_transverse_modal_operator(
            "particles", nx, nz, params, k_val=km_nosz, shape=p
        )
        op_kp_nosz = compute_transverse_modal_operator(
            "particles", nx, nz, params, k_val=kp_nosz, shape=p
        )
        sz_km = float(np.sinc(0.5 * km_sz * dz / np.pi) ** p)
        sz_kp = float(np.sinc(0.5 * kp_sz * dz / np.pi) ** p)
        bq_Ratio_pred = (
            sz_kp * float(np.sqrt(op_kp["evals"][0]))
            + sz_km * float(np.sqrt(op_km["evals"][0]))
        ) / (
            float(np.sqrt(op_kp_nosz["evals"][0]))
            + float(np.sqrt(op_km_nosz["evals"][0]))
        )
        bq_ratio_meas = float(det_sz["bq_modes"][0] / det_nosz["bq_modes"][0])
        np.testing.assert_allclose(bq_ratio_meas, bq_Ratio_pred, rtol=0.0, atol=1e-13)
        assert abs(sz_km**2 - sz_sq_expected) < 5e-2

        tols_p = derived_tolerances(nx, nz, params, shape=p)
        be_half = 0.5 * params["beta_e"] * dz
        bq_half = 0.5 * params["beta_q"] * dz
        amp_expected = float((np.sinc(be_half / np.pi) ** p) * np.cos(bq_half))
        np.testing.assert_allclose(
            tols_p["amp_pred_part"], amp_expected, rtol=0.0, atol=1e-15
        )


def test_mutation_shape1_vs_shape2_prediction_exceeds_slack():
    """Mutation check: shape=1 modal prediction vs shape=2 modal prediction differs by more than slack_part on coarse grid."""
    params = compute_analytic_parameters()
    nx, _, nz = COARSE_GRID
    tols_p1 = derived_tolerances(nx, nz, params, shape=1)
    tols_p2 = derived_tolerances(nx, nz, params, shape=2)
    bq_p1 = float(tols_p1["bq_pred_part"])
    bq_p2 = float(tols_p2["bq_pred_part"])
    slack_p1 = float(tols_p1["slack_part"])
    slack_p2 = float(tols_p2["slack_part"])
    rel_diff_vs_p2 = abs(bq_p1 / bq_p2 - 1.0)
    rel_diff_vs_p1 = abs(bq_p2 / bq_p1 - 1.0)

    print(
        f"\nMUTCHECK id=shape1_vs_shape2 bq_pred_p1={bq_p1:.8f} bq_pred_p2={bq_p2:.8f} "
        f"rel_diff_vs_p2={rel_diff_vs_p2:.6e} slack_p2={slack_p2:.6e} "
        f"rel_diff_vs_p1={rel_diff_vs_p1:.6e} slack_p1={slack_p1:.6e}",
        flush=True,
    )

    with pytest.raises(AssertionError, match="exceeds"):
        assert rel_diff_vs_p2 <= slack_p2, (
            f"shape=1 vs shape=2 relative difference {rel_diff_vs_p2:.6e} exceeds slack_p2={slack_p2:.6e}"
        )
    with pytest.raises(AssertionError, match="exceeds"):
        assert rel_diff_vs_p1 <= slack_p1, (
            f"shape=2 vs shape=1 relative difference {rel_diff_vs_p1:.6e} exceeds slack_p1={slack_p1:.6e}"
        )


@pytest.mark.slow
def test_space_charge_wave_envelope_entrance_and_spectrum():
    """Verify coarse-grid signed checks (Step 2 & Step 3), entrance amplitude, DFT leakage, and spatial spectrum."""
    params = compute_analytic_parameters()
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params)
    z_min, z_max = fit_window(params)
    beta_q_th = params["beta_q"]

    lb_res, part_res = cached_baseline_pair(nx, nz)
    z_coords = lb_res["z_coords"]
    i1_lb = lb_res["i1_lb"]
    i1_part = part_res["i1_part"]

    fit_lb = fit_beta_q_from_envelope(
        z_coords, i1_lb, beta_q_th, z_min=z_min, z_max=z_max
    )
    fit_part = fit_beta_q_from_envelope(
        z_coords, i1_part, beta_q_th, z_min=z_min, z_max=z_max
    )
    bq_lb = fit_lb["beta_q"]
    bq_part_lat = fit_part["beta_q"]

    delta_meas_lat = (params["u0"] - part_res["u_z_mean"]) / params["u0"]
    delta_rho_lat = (abs(part_res["rho_dc_mean"]) - abs(params["rho0"])) / abs(
        params["rho0"]
    )

    bq_lb_pred = tols["bq_pred_lb"]
    bq_part_pred = tols["bq_pred_part"]
    slack_lb = tols["slack_lb"]
    slack_part = tols["slack_part"]
    slack_cross = tols["slack_cross"]

    resid_lb = abs(bq_lb / bq_lb_pred - 1.0)
    resid_lat = abs(bq_part_lat / bq_part_pred - 1.0)
    extrap_c = compute_m_extrapolation(nx, nz, include_m4=True)
    beta0_c = float(extrap_c["beta0_3pt"])
    resid_cross = abs(beta0_c / bq_lb - bq_part_pred / bq_lb_pred)

    ev_p = tols["evals_part"]
    ev_lb = tols["evals_lb"]
    print(flush=True)
    print(format_error_budget(nx, nz, tols), flush=True)
    print(
        f"TERMS grid=coarse gamma1={tols['gamma1']:.6f} gamma1_disc={tols['gamma1_disc']:.6f} "
        f"kappa_dc={tols['kappa_dc']:.6f} d_w={tols['d_w']:.6e} "
        f"d_w_grid={tols['d_w_grid']:.6e} d_w_common={tols['d_w_common']:.6e} "
        f"z_min={tols['z_min']:.6f} z_max={tols['z_max']:.6f} "
        f"eps_w_bias={tols['eps_w_bias']:.6e} "
        f"s_win={tols['s_win_part']:.6e} "
        f"s_modes={tols['s_modes_part']:.6e} s_sym={tols['s_sym_part']:.6e} "
        f"s_round={tols['s_round']:.6e} "
        f"s_win_lb={tols['s_win_lb']:.6e} s_modes_lb={tols['s_modes_lb']:.6e} "
        f"s_sym_lb={tols['s_sym_lb']:.6e} "
        f"alias_supp={tols['alias_supp']:.6e} "
        f"lambda_1={ev_p[0]:.6f} lambda_2={ev_p[1]:.6f} "
        f"lambda_3={ev_p[2]:.6f} lambda_4={ev_p[3]:.6f} "
        f"lambda_lb_1={ev_lb[0]:.6f} lambda_lb_2={ev_lb[1]:.6f} "
        f"lambda_lb_3={ev_lb[2]:.6f} lambda_lb_4={ev_lb[3]:.6f}",
        flush=True,
    )
    print(
        f"LBCHECK grid=coarse bq_th={beta_q_th:.6f} bq_pred_modal={bq_lb_pred:.6f} "
        f"bq_lb={bq_lb:.6f} resid={resid_lb:.6e} "
        f"s_win_lb={tols['s_win_lb']:.6e} s_modes_lb={tols['s_modes_lb']:.6e} "
        f"s_sym_lb={tols['s_sym_lb']:.6e} s_round={tols['s_round']:.6e} slack={slack_lb:.6e}",
        flush=True,
    )
    print(
        f"CROSSCHECK grid=coarse beta_m={extrap_c['bq_m']:.6f} beta_m2={extrap_c['bq_m2']:.6f} "
        f"beta_m4={extrap_c['bq_m4']:.6f} beta0={beta0_c:.6f} bq_lb={bq_lb:.6f} "
        f"bq_pred_part={bq_part_pred:.6f} bq_pred_lb={bq_lb_pred:.6f} "
        f"resid_cross={resid_cross:.6e} "
        f"s_win_lb={tols['s_win_lb']:.6e} s_modes_lb={tols['s_modes_lb']:.6e} s_sym_lb={tols['s_sym_lb']:.6e} "
        f"s_win_part={tols['s_win_part']:.6e} s_modes_part={tols['s_modes_part']:.6e} s_sym_part={tols['s_sym_part']:.6e} "
        f"s_round={tols['s_round']:.6e} slack_lb={slack_lb:.6e} slack_part={slack_part:.6e} "
        f"slack_cross={slack_cross:.6e} e_R={extrap_c['e_R']:.6e} sigma_prop={extrap_c['sigma_prop']:.6e} "
        f"bound0={extrap_c['bound0']:.6e} resid0={extrap_c['resid0']:.6e}",
        flush=True,
    )
    print(
        f"Coarse two-wave report (report-only diagnostics): "
        f"LB: k_c={fit_lb['k_c']:.4f} rad/m, |A|={abs(fit_lb['A']):.6e} A, "
        f"|a_s|/|a_f|={fit_lb['as_over_af']:.6f}, B/A={fit_lb['b_over_a']:.6f}, "
        f"z0={fit_lb['z0'] * 1e3:.4f} mm, sigma_bq={fit_lb['sigma_beta_q']:.6e} rad/m | "
        f"Part (sheet_lattice): k_c={fit_part['k_c']:.4f} rad/m, |A|={abs(fit_part['A']):.6e} A, "
        f"|a_s|/|a_f|={fit_part['as_over_af']:.6f}, B/A={fit_part['b_over_a']:.6f}, "
        f"z0={fit_part['z0'] * 1e3:.4f} mm, sigma_bq={fit_part['sigma_beta_q']:.6e} rad/m, "
        f"delta_meas_lat={delta_meas_lat:.6e} (delta_th={tols['delta_th']:.6e}, "
        f"delta_disc_exact={tols['delta_disc_exact']:.6e}, rho_dc_rel={delta_rho_lat:.6e}), "
        f"bq_pred_modal={bq_part_pred:.6f} rad/m "
        f"({bq_part_pred / beta_q_th - 1.0:+.4%} vs bq_th), "
        f"beta_m={extrap_c['bq_m']:.6f}, beta_m2={extrap_c['bq_m2']:.6f}, beta_m4={extrap_c['bq_m4']:.6f}, "
        f"beta0_3pt={beta0_c:.6f}, resid0={extrap_c['resid0']:.6e} (bound0={extrap_c['bound0']:.6e}, "
        f"finite-m resid_lat={resid_lat:.6e}, slack={slack_part:.6e})",
        flush=True,
    )

    # Observable (ii): spatial spectrum fast and slow peaks around beta_e +- beta_q
    # (Step 7: report-only because the rectangular window [0, lambda_q] includes the
    # unpredicted cathode launch phase z_0, so the two-wave Dirichlet-kernel sidelobe
    # interaction cannot be derived in closed form without z_0)
    beta_e = params["beta_e"]
    k_fast_th = beta_e - beta_q_th
    k_slow_th = beta_e + beta_q_th
    for label, phasor in (("LinearizedBeam", i1_lb), ("ParticleSolver", i1_part)):
        kz_grid, spec, k_fast, k_slow = compute_spatial_spectrum(
            z_coords, phasor, beta_e, beta_q_th, z_max=params["lambda_q"]
        )
        idx_fast = int(np.argmin(np.abs(kz_grid - k_fast)))
        idx_slow = int(np.argmin(np.abs(kz_grid - k_slow)))
        valley_min = float(np.min(spec[idx_fast : idx_slow + 1]))
        peak_min = float(min(spec[idx_fast], spec[idx_slow]))
        err_fast = float(abs(k_fast - k_fast_th))
        err_slow = float(abs(k_slow - k_slow_th))
        print(
            f"DIAG id=spectrum code={label} k_fast={k_fast:.6f} k_slow={k_slow:.6f} "
            f"k_fast_th={k_fast_th:.6f} k_slow_th={k_slow_th:.6f} "
            f"err_fast={err_fast:.6e} err_slow={err_slow:.6e} "
            f"valley_min={valley_min:.6e} peak_min={peak_min:.6e}",
            flush=True,
        )

    # Step 1, Step 3 & Step 4 identity asserts (using discrete gamma1_disc)
    assert tols["kappa_dc"] > tols["gamma1_disc"], (
        f"Identity check failed: kappa_dc={tols['kappa_dc']:.6f} <= gamma1_disc={tols['gamma1_disc']:.6f}"
    )
    slack_lb_check = (
        tols["s_win_lb"] + tols["s_modes_lb"] + tols["s_sym_lb"] + tols["s_round"]
    )
    slack_part_check = (
        tols["s_win_part"] + tols["s_modes_part"] + tols["s_sym_part"] + tols["s_round"]
    )
    assert np.isclose(tols["slack_lb"], slack_lb_check, rtol=0.0, atol=1e-15), (
        f"Identity check failed for slack_lb: val={tols['slack_lb']:.16e}, "
        f"expected={slack_lb_check:.16e}"
    )
    assert np.isclose(tols["slack_part"], slack_part_check, rtol=0.0, atol=1e-15), (
        f"Identity check failed for slack_part: val={tols['slack_part']:.16e}, "
        f"expected={slack_part_check:.16e}"
    )
    assert np.isclose(
        tols["slack_cross"], tols["slack_lb"] + tols["slack_part"], rtol=0.0, atol=1e-15
    ), (
        f"Identity check failed for slack_cross: val={tols['slack_cross']:.16e}, "
        f"expected={tols['slack_lb'] + tols['slack_part']:.16e}"
    )
    dz = TUNNEL_LENGTH_M / nz
    tol_amp_part_check = float(
        (0.5 * params["beta_e"] * dz) ** 4 / 120.0
        + (0.5 * params["beta_q"] * dz) ** 4 / 24.0
        + 1.0e-12
    )
    assert np.isclose(tols["tol_amp_part"], tol_amp_part_check, rtol=0.0, atol=1e-15), (
        f"Identity check failed for tol_amp_part: val={tols['tol_amp_part']:.16e}, "
        f"expected={tol_amp_part_check:.16e}"
    )

    # DFT leakage identity (round-off)
    assert lb_res["dirichlet_leak_lb"] <= tols["eps_dft"], (
        f"Identity (round-off) DFT leakage LB={lb_res['dirichlet_leak_lb']:.3e} > "
        f"bound={tols['eps_dft']:.3e} (margin={tols['eps_dft'] - lb_res['dirichlet_leak_lb']:+.3e})"
    )
    assert part_res["dirichlet_leak_part"] <= tols["eps_dft"], (
        f"Identity (round-off) DFT leakage Part={part_res['dirichlet_leak_part']:.3e} > "
        f"bound={tols['eps_dft']:.3e} (margin={tols['eps_dft'] - part_res['dirichlet_leak_part']:+.3e})"
    )

    # Amplitude sanity check |A| > 0.5 * m * I0
    min_amp_a = 0.5 * params["I1_0"]
    assert abs(fit_lb["A"]) > min_amp_a, (
        f"LB fit |A|={abs(fit_lb['A']):.6e} <= 0.5*m*I0={min_amp_a:.6e} "
        f"(margin={abs(fit_lb['A']) - min_amp_a:+.6e})"
    )
    assert abs(fit_part["A"]) > min_amp_a, (
        f"Part fit |A|={abs(fit_part['A']):.6e} <= 0.5*m*I0={min_amp_a:.6e} "
        f"(margin={abs(fit_part['A']) - min_amp_a:+.6e})"
    )

    # Step 3: LinearizedBeam deterministic modal prediction check
    assert resid_lb <= slack_lb, (
        f"Coarse LB |bq_lb/bq_pred_modal - 1| = {resid_lb:.6e} > slack_lb = {slack_lb:.6e} "
        f"(margin={slack_lb - resid_lb:+.6e})"
    )

    # Step 2 & Step 3: Cross-code ("sheet_lattice") check using coarse extrapolated beta0
    assert resid_cross <= slack_cross, (
        f"Coarse cross-code |beta0_P/bq_LB - bq_P_pred/bq_LB_pred| = {resid_cross:.6e} > "
        f"slack_lb + slack_part = {slack_cross:.6e} (margin={slack_cross - resid_cross:+.6e})"
    )

    # Step 3: Particle deterministic ("sheet_lattice") 3-point Richardson m -> 0 modal prediction check
    assert extrap_c["resid0"] <= extrap_c["bound0"], (
        f"Coarse Particle (sheet_lattice) |beta0_3pt/bq_pred_modal - 1| = {extrap_c['resid0']:.6e} > "
        f"slack_part + (e_R + sigma_prop)/bq_pred = {extrap_c['bound0']:.6e} "
        f"(slack={slack_part:.6e}, e_R/bq_pred={extrap_c['e_R'] / bq_part_pred:.6e}, "
        f"sigma_prop/bq_pred={extrap_c['sigma_prop'] / bq_part_pred:.6e}, "
        f"margin={extrap_c['bound0'] - extrap_c['resid0']:+.6e})"
    )

    # Entrance amplitude at z_0 = 0.5 * dz
    amp_err_lb = float(abs(np.abs(i1_lb[0]) / params["I1_0"] - 1.0))
    amp_err_part = float(
        abs(np.abs(i1_part[0]) / params["I1_0"] - tols["amp_pred_part"])
    )
    assert amp_err_lb <= tols["tol_amp_lb"], (
        f"LB entrance |I1(z0)|/I1_0 - 1 = {amp_err_lb:.6e} > bound={tols['tol_amp_lb']:.6e} "
        f"(margin={tols['tol_amp_lb'] - amp_err_lb:+.6e})"
    )
    assert amp_err_part <= tols["tol_amp_part"], (
        f"Part entrance ||I1(z0)|/I1_0 - sinc(be*dz/2)*cos(bq*dz/2)| = {amp_err_part:.6e} > "
        f"bound={tols['tol_amp_part']:.6e} (margin={tols['tol_amp_part'] - amp_err_part:+.6e})"
    )


@pytest.mark.slow
def test_mutation_double_particle_charge_to_mass_fails():
    """M1: Doubling particle species |q/m| matches mutated prediction after m->0 extrapolation and fails baseline cross-code check."""
    params = compute_analytic_parameters(qm_scale=1.0)
    params_mut = compute_analytic_parameters(qm_scale=2.0)
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params)
    tols_mut = derived_tolerances(nx, nz, params_mut)
    extrap_base = compute_m_extrapolation(nx, nz, include_m4=True)

    lb_res, _ = cached_baseline_pair(nx, nz)
    z_coords = lb_res["z_coords"]
    z_min_base, z_max_base = fit_window(params)
    z_min_mut, z_max_mut = fit_window(params_mut)

    fit_lb = fit_beta_q_from_envelope(
        z_coords, lb_res["i1_lb"], params["beta_q"], z_min=z_min_base, z_max=z_max_base
    )
    bq_lb = fit_lb["beta_q"]
    bq_part_base_pred = tols["bq_pred_part"]

    # Double |q/m| through the harness (mass = m_e / 2 at charge -e), so the run derives its
    # settle time and DC state from the mutated parameters, exactly as the prediction does.
    part_mut = run_particle_beam_drift(
        nx,
        nz,
        beam_ref=lb_res["beam"],
        qm_scale=2.0,
        seed=42,
        m_mod=MODULATION_INDEX,
        profile="sheet_lattice",
    )
    part_mut_m2 = run_particle_beam_drift(
        nx,
        nz,
        beam_ref=lb_res["beam"],
        qm_scale=2.0,
        seed=42,
        m_mod=0.5 * MODULATION_INDEX,
        profile="sheet_lattice",
    )
    fit_mut = fit_beta_q_from_envelope(
        z_coords,
        part_mut["i1_part"],
        params_mut["beta_q"],
        z_min=z_min_mut,
        z_max=z_max_mut,
    )
    fit_mut_m2 = fit_beta_q_from_envelope(
        z_coords,
        part_mut_m2["i1_part"],
        params_mut["beta_q"],
        z_min=z_min_mut,
        z_max=z_max_mut,
    )
    beta_m_m1 = float(fit_mut["beta_q"])
    beta_m2_m1 = float(fit_mut_m2["beta_q"])
    beta0_m1 = float((4.0 * beta_m2_m1 - beta_m_m1) / 3.0)
    bq_part_mut_pred = tols_mut["bq_pred_part"]
    slack_m1 = float(tols_mut["slack_part"])

    # Two-point remainder scaled from the coarse baseline:
    # (e_R / |beta(m) - beta(m/2)|)_baseline * |beta(m) - beta(m/2)|_mut
    e_r_base = float(extrap_base["e_R"])
    dbq_base = float(abs(extrap_base["bq_m"] - extrap_base["bq_m2"]))
    dbq_m1 = float(abs(beta_m_m1 - beta_m2_m1))
    e_r_m1 = float((e_r_base / max(dbq_base, 1e-30)) * dbq_m1)
    e_r_m1_rel = float(e_r_m1 / bq_part_mut_pred)
    resid_m1 = abs(beta0_m1 / bq_part_mut_pred - 1.0)
    # Same noise class as the baseline bound0: propagated fit standard error of the 2-point estimate
    sigma_prop_m1 = float(
        np.hypot(fit_mut["sigma_beta_q"] / 3.0, 4.0 * fit_mut_m2["sigma_beta_q"] / 3.0)
    )
    bound_m1 = float(slack_m1 + (e_r_m1 + sigma_prop_m1) / bq_part_mut_pred)

    resid_cross_mut = abs(beta0_m1 / bq_lb - bq_part_base_pred / tols["bq_pred_lb"])
    baseline_check_failed = bool(resid_cross_mut > tols["slack_cross"])

    print(
        f"\nMUTCHECK id=M1 beta_m={beta_m_m1:.6f} beta_m2={beta_m2_m1:.6f} "
        f"beta0={beta0_m1:.6f} meas={beta0_m1:.6f} pred_modal={bq_part_mut_pred:.6f} "
        f"s_win={tols_mut['s_win_part']:.6e} s_modes={tols_mut['s_modes_part']:.6e} "
        f"s_sym={tols_mut['s_sym_part']:.6e} s_round={tols_mut['s_round']:.6e} "
        f"slack={slack_m1:.6e} e_R_base={e_r_base:.6e} dbq_base={dbq_base:.6e} "
        f"dbq_mut={dbq_m1:.6e} e_R_mut={e_r_m1:.6e} e_R_mut_rel={e_r_m1_rel:.6e} "
        f"sigma_prop={sigma_prop_m1:.6e} resid={resid_m1:.6e} bound={bound_m1:.6e} "
        f"baseline_check_failed={baseline_check_failed}",
        flush=True,
    )

    with pytest.raises(AssertionError, match="exceeds"):
        assert resid_cross_mut <= tols["slack_cross"], (
            f"M1 cross-code difference {resid_cross_mut:.6e} exceeds {tols['slack_cross']:.6e}"
        )

    assert resid_m1 <= bound_m1, (
        f"M1 |beta0_mut/bq_part_mut_pred - 1| = {resid_m1:.6e} > "
        f"slack + (e_R_mut + sigma_prop)/bq_pred = {bound_m1:.6e} (margin={bound_m1 - resid_m1:+.6e})"
    )


@pytest.mark.slow
def test_mutation_linearized_beam_rho0_off_by_2x_fails(monkeypatch):
    """M2: Doubling `LinearizedBeam.rho0` matches `omega_p,dep * sqrt(2)` prediction and fails cross-code check."""
    params = compute_analytic_parameters()
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params)
    tols_m2 = derived_tolerances(nx, nz, params, omega_p_scale=float(np.sqrt(2.0)))

    lb_res, _ = cached_baseline_pair(nx, nz)
    z_coords = lb_res["z_coords"]
    z_min, z_max = fit_window(params)

    extrap_base = compute_m_extrapolation(nx, nz, include_m4=True)
    beta0_base = float(extrap_base["beta0_3pt"])
    bq_part_base_pred = tols["bq_pred_part"]

    orig_lb_init = LinearizedBeam.__init__

    def _mutated_lb_init(self, *args, **kwargs):
        if "rho0" in kwargs and kwargs["rho0"] is not None:
            kwargs = dict(kwargs, rho0=2.0 * kwargs["rho0"])
        orig_lb_init(self, *args, **kwargs)

    monkeypatch.setattr(LinearizedBeam, "__init__", _mutated_lb_init)
    lb_mut = run_linearized_beam_drift(nx, nz, rho0_scale=1.0)
    fit_lb_mut = fit_beta_q_from_envelope(
        z_coords,
        lb_mut["i1_lb"],
        float(np.sqrt(2.0) * params["beta_q"]),
        z_min=z_min,
        z_max=z_max,
    )
    bq_lb_mut = fit_lb_mut["beta_q"]
    bq_lb_m2_pred = tols_m2["bq_pred_lb"]
    bound_m2 = tols_m2["slack_lb"]
    resid_m2 = abs(bq_lb_mut / bq_lb_m2_pred - 1.0)
    para_m2 = fit_lb_parasitic_waves(
        z_coords,
        lb_mut["i1_lb"],
        nx,
        nz,
        params,
        omega_p_scale=float(np.sqrt(2.0)),
        z_min=z_min,
        z_max=z_max,
    )

    resid_cross_mut = abs(
        beta0_base / bq_lb_mut - bq_part_base_pred / tols["bq_pred_lb"]
    )
    baseline_check_failed = bool(resid_cross_mut > tols["slack_cross"])

    print(
        f"\nMUTCHECK id=M2 meas={bq_lb_mut:.6f} pred_modal={bq_lb_m2_pred:.6f} "
        f"slack={bound_m2:.6e} resid={resid_m2:.6e} bound={bound_m2:.6e} "
        f"kp_minus={para_m2['kp_minus']:.6f} kp_plus={para_m2['kp_plus']:.6f} "
        f"disp_resid_minus={para_m2['disp_resid_minus']:.3e} disp_resid_plus={para_m2['disp_resid_plus']:.3e} "
        f"ap_over_afwd={para_m2['ap_over_afwd']:.6e} dbq_para={para_m2['delta_beta_q']:+.6e} "
        f"rel_dbq_para={para_m2['rel_delta_beta_q']:+.6e} "
        f"baseline_check_failed={baseline_check_failed}",
        flush=True,
    )

    with pytest.raises(AssertionError, match="exceeds"):
        assert resid_cross_mut <= tols["slack_cross"], (
            f"M2 cross-code difference {resid_cross_mut:.6e} exceeds {tols['slack_cross']:.6e}"
        )

    assert resid_m2 <= bound_m2, (
        f"M2 |bq_lb_mut/bq_lb_m2_pred - 1| = {resid_m2:.6e} > slack_lb = {bound_m2:.6e} "
        f"(margin={bound_m2 - resid_m2:+.6e})"
    )


@pytest.mark.slow
def test_mutation_disk_profile_instead_of_sheet_fails(monkeypatch):
    """M3: Fibonacci disk x-profile matches non-uniform modal prediction and fails baseline sheet check."""
    params = compute_analytic_parameters()
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params)
    lb_res, _ = cached_baseline_pair(nx, nz)
    z_coords = lb_res["z_coords"]
    z_min, z_max = fit_window(params)

    monkeypatch.setattr(BeamInjector, "_sheet_lattice_x", _fibonacci_disk_lattice_x)
    part_m3 = _cached_m3_disk_run(nx, nz, m_mod=MODULATION_INDEX)
    part_m3_m2 = _cached_m3_disk_run(nx, nz, m_mod=0.5 * MODULATION_INDEX)
    fit_m3_m = fit_beta_q_from_envelope(
        z_coords, part_m3["i1_part"], params["beta_q"], z_min=z_min, z_max=z_max
    )
    fit_m3_m2 = fit_beta_q_from_envelope(
        z_coords, part_m3_m2["i1_part"], params["beta_q"], z_min=z_min, z_max=z_max
    )
    beta_m_m3 = float(fit_m3_m["beta_q"])
    beta_m2_m3 = float(fit_m3_m2["beta_q"])
    beta0_m3 = float((4.0 * beta_m2_m3 - beta_m_m3) / 3.0)
    bq_sheet_pred = tols["bq_pred_part"]
    resid_vs_sheet = abs(beta0_m3 / bq_sheet_pred - 1.0)
    bound_sheet = tols["slack_part"]
    baseline_check_failed = bool(resid_vs_sheet > bound_sheet)

    x_fib = _fibonacci_disk_x_coords(16, radius=BEAM_HALF_THICKNESS_M, x0=0.0)
    bq_m3_pred = float(
        discrete_modal_prediction("particles", nx, nz, params, line_x=x_fib)
    )
    slack_m3_info = prediction_slack("particles", nx, nz, params, line_x=x_fib)
    slack_m3 = float(slack_m3_info["slack"])

    extrap_base = compute_m_extrapolation(nx, nz, include_m4=True)
    e_r_base = float(extrap_base["e_R"])
    dbq_base = float(abs(extrap_base["bq_m"] - extrap_base["bq_m2"]))
    dbq_m3 = float(abs(beta_m_m3 - beta_m2_m3))
    e_r_m3 = float((e_r_base / max(dbq_base, 1e-30)) * dbq_m3)
    e_r_m3_rel = float(e_r_m3 / bq_m3_pred)
    bound_m3 = float(slack_m3 + e_r_m3_rel)
    resid_m3 = abs(beta0_m3 / bq_m3_pred - 1.0)

    delta_meas_m3 = float((params["u0"] - part_m3["u_z_mean"]) / params["u0"])
    delta_disc_m3 = float(compute_discrete_dc_depression(nx, params, line_x=x_fib))
    delta_disc_uniform = float(tols["delta_disc_exact"])
    err_dep_m3 = abs(delta_meas_m3 - delta_disc_m3)
    bound_dep_m3 = float(tols["eps_ac"] + 1.0e-12)

    print(
        f"\nDEPCHECK id=M3 delta_meas_m3={delta_meas_m3:.6e} "
        f"delta_disc_m3={delta_disc_m3:.6e} "
        f"delta_disc_uniform={delta_disc_uniform:.6e} "
        f"eps_ac={tols['eps_ac']:.6e}",
        flush=True,
    )
    print(
        f"MUTCHECK id=M3 beta_m={beta_m_m3:.6f} beta_m2={beta_m2_m3:.6f} "
        f"beta0={beta0_m3:.6f} meas={beta0_m3:.6f} pred_modal={bq_m3_pred:.6f} "
        f"s_win={slack_m3_info['s_win']:.6e} s_modes={slack_m3_info['s_modes']:.6e} "
        f"s_sym={slack_m3_info['s_sym']:.6e} s_round={slack_m3_info['s_round']:.6e} "
        f"slack={slack_m3:.6e} e_R_base={e_r_base:.6e} dbq_base={dbq_base:.6e} "
        f"dbq_mut={dbq_m3:.6e} e_R_mut={e_r_m3:.6e} e_R_mut_rel={e_r_m3_rel:.6e} "
        f"resid={resid_m3:.6e} bound={bound_m3:.6e} "
        f"resid_vs_sheet={resid_vs_sheet:.6e} baseline_check_failed={baseline_check_failed}",
        flush=True,
    )

    with pytest.raises(AssertionError, match="exceeds"):
        assert resid_vs_sheet <= bound_sheet, (
            f"M3 baseline sheet residual {resid_vs_sheet:.6e} exceeds slack_part={bound_sheet:.6e}"
        )

    assert err_dep_m3 <= bound_dep_m3, (
        f"M3 DC depression |delta_meas_m3 - delta_disc_m3| = {err_dep_m3:.6e} > "
        f"eps_AC + 1e-12 = {bound_dep_m3:.6e} (margin={bound_dep_m3 - err_dep_m3:+.6e})"
    )

    assert resid_m3 <= bound_m3, (
        f"M3 |beta0_m3/bq_m3_pred - 1| = {resid_m3:.6e} > "
        f"slack_m3 + e_R_mut/bq_pred = {bound_m3:.6e} (margin={bound_m3 - resid_m3:+.6e})"
    )


@pytest.mark.slow
def test_two_resolution_convergence_and_sheet_ensemble():
    """Verify 8-seed `"sheet"` ensemble (4.3, 4.5, 4.6) on both grids and fine-grid signed checks."""
    params = compute_analytic_parameters()
    beta_q_th = params["beta_q"]
    z_min, z_max = fit_window(params)

    nx_c, _, nz_c = COARSE_GRID
    nx_f, _, nz_f = FINE_GRID
    tols_c = derived_tolerances(nx_c, nz_c, params)
    tols_f = derived_tolerances(nx_f, nz_f, params)

    lb_c, part_c = cached_baseline_pair(nx_c, nz_c)
    lb_f, part_f = cached_baseline_pair(nx_f, nz_f)
    noise_c = measure_empirical_noise(COARSE_GRID)
    noise_f = measure_empirical_noise(FINE_GRID)

    fit_lb_c = fit_beta_q_from_envelope(
        lb_c["z_coords"], lb_c["i1_lb"], beta_q_th, z_min=z_min, z_max=z_max
    )
    fit_part_c = fit_beta_q_from_envelope(
        lb_c["z_coords"], part_c["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
    )
    fit_lb_f = fit_beta_q_from_envelope(
        lb_f["z_coords"], lb_f["i1_lb"], beta_q_th, z_min=z_min, z_max=z_max
    )
    fit_part_f = fit_beta_q_from_envelope(
        lb_f["z_coords"], part_f["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
    )

    bq_lb_c = fit_lb_c["beta_q"]
    bq_part_lat_c = fit_part_c["beta_q"]
    bq_lb_f = fit_lb_f["beta_q"]
    bq_part_lat_f = fit_part_f["beta_q"]

    delta_lat_c = (params["u0"] - part_c["u_z_mean"]) / params["u0"]
    delta_lat_f = (params["u0"] - part_f["u_z_mean"]) / params["u0"]
    bq_part_pred_c = tols_c["bq_pred_part"]
    bq_part_pred_f = tols_f["bq_pred_part"]
    bq_lb_pred_f = tols_f["bq_pred_lb"]

    resid_lb_f = abs(bq_lb_f / bq_lb_pred_f - 1.0)
    resid_lat_c = abs(bq_part_lat_c / bq_part_pred_c - 1.0)
    resid_lat_f = abs(bq_part_lat_f / bq_part_pred_f - 1.0)
    extrap_c = compute_m_extrapolation(nx_c, nz_c, include_m4=True)
    extrap_f = compute_m_extrapolation(nx_f, nz_f, include_m4=False)
    beta0_f = float(extrap_f["beta0"])
    resid0_f = float(extrap_f["resid0"])
    resid_cross_f = abs(beta0_f / bq_lb_f - bq_part_pred_f / bq_lb_pred_f)
    resid_cross_lat_f = abs(bq_part_lat_f / bq_lb_f - bq_part_pred_f / bq_lb_pred_f)

    tcrit_c = noise_c["tcrit"]
    tcrit_f = noise_f["tcrit"]
    bq_part_lat_ens_c = noise_c["bq_lattice"]
    bq_part_lat_ens_f = noise_f["bq_lattice"]
    bq_part_pred_ens_c = noise_c["bq_pred"]
    bq_part_pred_ens_f = noise_f["bq_pred"]
    resid_mean_c = abs(noise_c["bq_mean"] / bq_part_pred_ens_c - 1.0)
    resid_mean_f = abs(noise_f["bq_mean"] / bq_part_pred_ens_f - 1.0)
    bound_mean_c = (
        tols_c["slack_part"] + tcrit_c * noise_c["bq_sem"] / bq_part_pred_ens_c
    )
    bound_mean_f = (
        tols_f["slack_part"] + tcrit_f * noise_f["bq_sem"] / bq_part_pred_ens_f
    )
    mean_minus_lat_c = abs(noise_c["bq_mean"] - bq_part_lat_ens_c)
    mean_minus_lat_f = abs(noise_f["bq_mean"] - bq_part_lat_ens_f)

    seeds_c_str = "[" + ",".join(f"{float(v):.6f}" for v in noise_c["bq_seeds"]) + "]"
    seeds_f_str = "[" + ",".join(f"{float(v):.6f}" for v in noise_f["bq_seeds"]) + "]"
    first_exc: Exception | None = None

    ev_p_f = tols_f["evals_part"]
    ev_lb_f = tols_f["evals_lb"]
    print(flush=True)
    print(format_error_budget(nx_f, nz_f, tols_f), flush=True)
    print(
        f"TERMS grid=fine gamma1={tols_f['gamma1']:.6f} gamma1_disc={tols_f['gamma1_disc']:.6f} "
        f"kappa_dc={tols_f['kappa_dc']:.6f} d_w={tols_f['d_w']:.6e} "
        f"d_w_grid={tols_f['d_w_grid']:.6e} d_w_common={tols_f['d_w_common']:.6e} "
        f"z_min={tols_f['z_min']:.6f} z_max={tols_f['z_max']:.6f} "
        f"eps_w_bias={tols_f['eps_w_bias']:.6e} "
        f"s_win={tols_f['s_win_part']:.6e} "
        f"s_modes={tols_f['s_modes_part']:.6e} s_sym={tols_f['s_sym_part']:.6e} "
        f"s_round={tols_f['s_round']:.6e} "
        f"s_win_lb={tols_f['s_win_lb']:.6e} s_modes_lb={tols_f['s_modes_lb']:.6e} "
        f"s_sym_lb={tols_f['s_sym_lb']:.6e} "
        f"alias_supp={tols_f['alias_supp']:.6e} "
        f"lambda_1={ev_p_f[0]:.6f} lambda_2={ev_p_f[1]:.6f} "
        f"lambda_3={ev_p_f[2]:.6f} lambda_4={ev_p_f[3]:.6f} "
        f"lambda_lb_1={ev_lb_f[0]:.6f} lambda_lb_2={ev_lb_f[1]:.6f} "
        f"lambda_lb_3={ev_lb_f[2]:.6f} lambda_lb_4={ev_lb_f[3]:.6f}",
        flush=True,
    )
    print(
        f"LBCHECK grid=fine bq_th={beta_q_th:.6f} bq_pred_modal={bq_lb_pred_f:.6f} "
        f"bq_lb={bq_lb_f:.6f} resid={resid_lb_f:.6e} slack={tols_f['slack_lb']:.6e}",
        flush=True,
    )

    # Step 1.3: LB outflow ramp-sensitivity diagnostic (outflow_ramp_m = 8 mm vs 4 mm,
    # both evaluated on the 8 mm fit window [d_w, Lz - 8 mm - d_w] so both ramps lie >= d_w behind z_max)
    try:
        z_min_8mm, z_max_8mm = fit_window(params, outflow_ramp_m=8.0e-3)
        f4_c = fit_four_wave_reflection(
            lb_c["z_coords"], lb_c["i1_lb"], beta_q_th, z_min=z_min_8mm, z_max=z_max_8mm
        )
        f4_f = fit_four_wave_reflection(
            lb_f["z_coords"], lb_f["i1_lb"], beta_q_th, z_min=z_min_8mm, z_max=z_max_8mm
        )
        lb_c_4mm = run_linearized_beam_drift(nx_c, nz_c, outflow_ramp_m=4.0e-3)
        lb_f_4mm = run_linearized_beam_drift(nx_f, nz_f, outflow_ramp_m=4.0e-3)
        fit_lb_c_4mm = fit_beta_q_from_envelope(
            lb_c_4mm["z_coords"],
            lb_c_4mm["i1_lb"],
            beta_q_th,
            z_min=z_min_8mm,
            z_max=z_max_8mm,
        )
        fit_lb_f_4mm = fit_beta_q_from_envelope(
            lb_f_4mm["z_coords"],
            lb_f_4mm["i1_lb"],
            beta_q_th,
            z_min=z_min_8mm,
            z_max=z_max_8mm,
        )
        bq_c_4mm = float(fit_lb_c_4mm["beta_q"])
        bq_f_4mm = float(fit_lb_f_4mm["beta_q"])
        dbq_ramp_c = float(bq_c_4mm - bq_lb_c)
        dbq_ramp_f = float(bq_f_4mm - bq_lb_f)
        rel_dbq_ramp_c = float(dbq_ramp_c / bq_lb_c)
        rel_dbq_ramp_f = float(dbq_ramp_f / bq_lb_f)
        print(
            f"DIAG id=lb_ramp "
            f"bq_8mm_coarse={bq_lb_c:.8f} bq_4mm_coarse={bq_c_4mm:.8f} "
            f"dbq_coarse={dbq_ramp_c:+.6e} rel_dbq_coarse={rel_dbq_ramp_c:+.6e} "
            f"bq_8mm_fine={bq_lb_f:.8f} bq_4mm_fine={bq_f_4mm:.8f} "
            f"dbq_fine={dbq_ramp_f:+.6e} rel_dbq_fine={rel_dbq_ramp_f:+.6e} "
            f"eps_w={EPS_WINDOW:.1e} z_min={z_min_8mm:.6f} z_max={z_max_8mm:.6f} "
            f"dbq_4wave_coarse={f4_c['delta_beta_q']:+.6e} rel_4wave_coarse={f4_c['rel_delta_beta_q']:+.6e} "
            f"bwd_over_fwd_coarse={f4_c['bwd_over_fwd']:.6e} "
            f"dbq_4wave_fine={f4_f['delta_beta_q']:+.6e} rel_4wave_fine={f4_f['rel_delta_beta_q']:+.6e} "
            f"bwd_over_fwd_fine={f4_f['bwd_over_fwd']:.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=lb_ramp status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    # Step 5: LB parasitic waves in the fit model (k_p in (pi/(2*dz), pi/dz) solved via brentq)
    try:
        para_c = fit_lb_parasitic_waves(
            lb_c["z_coords"],
            lb_c["i1_lb"],
            nx_c,
            nz_c,
            params,
            omega_p_scale=1.0,
            z_min=z_min,
            z_max=z_max,
        )
        para_f = fit_lb_parasitic_waves(
            lb_f["z_coords"],
            lb_f["i1_lb"],
            nx_f,
            nz_f,
            params,
            omega_p_scale=1.0,
            z_min=z_min,
            z_max=z_max,
        )
        lb_m2_c = cached_linearized_beam_drift(
            nx_c, nz_c, rho0_scale=2.0, m_mod=MODULATION_INDEX
        )
        para_m2_c = fit_lb_parasitic_waves(
            lb_m2_c["z_coords"],
            lb_m2_c["i1_lb"],
            nx_c,
            nz_c,
            params,
            omega_p_scale=float(np.sqrt(2.0)),
            z_min=z_min,
            z_max=z_max,
        )
        print(
            f"DIAG id=lb_parasitic "
            f"kp_minus={para_c['kp_minus']:.6f} kp_plus={para_c['kp_plus']:.6f} "
            f"disp_resid_minus={para_c['disp_resid_minus']:.3e} disp_resid_plus={para_c['disp_resid_plus']:.3e} "
            f"kp_minus_coarse={para_c['kp_minus']:.6f} kp_plus_coarse={para_c['kp_plus']:.6f} "
            f"disp_resid_minus_coarse={para_c['disp_resid_minus']:.3e} disp_resid_plus_coarse={para_c['disp_resid_plus']:.3e} "
            f"ap_over_afwd_coarse={para_c['ap_over_afwd']:.6e} "
            f"dbq_coarse={para_c['delta_beta_q']:+.6e} rel_dbq_coarse={para_c['rel_delta_beta_q']:+.6e} "
            f"resid_2wave_coarse={abs(bq_lb_c / tols_c['bq_pred_lb'] - 1.0):.6e} "
            f"kp_minus_fine={para_f['kp_minus']:.6f} kp_plus_fine={para_f['kp_plus']:.6f} "
            f"disp_resid_minus_fine={para_f['disp_resid_minus']:.3e} disp_resid_plus_fine={para_f['disp_resid_plus']:.3e} "
            f"ap_over_afwd_fine={para_f['ap_over_afwd']:.6e} "
            f"dbq_fine={para_f['delta_beta_q']:+.6e} rel_dbq_fine={para_f['rel_delta_beta_q']:+.6e} "
            f"resid_2wave_fine={resid_lb_f:.6e} "
            f"kp_minus_M2={para_m2_c['kp_minus']:.6f} kp_plus_M2={para_m2_c['kp_plus']:.6f} "
            f"disp_resid_minus_M2={para_m2_c['disp_resid_minus']:.3e} disp_resid_plus_M2={para_m2_c['disp_resid_plus']:.3e} "
            f"ap_over_afwd_M2={para_m2_c['ap_over_afwd']:.6e} "
            f"dbq_M2={para_m2_c['delta_beta_q']:+.6e} rel_dbq_M2={para_m2_c['rel_delta_beta_q']:+.6e} "
            f"estimator=2wave_kept",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=lb_parasitic status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    resid_coh_c = float(abs(noise_c["bq_coherent_mean"] / bq_part_pred_ens_c - 1.0))
    resid_coh_f = float(abs(noise_f["bq_coherent_mean"] / bq_part_pred_ens_f - 1.0))
    print(
        f"H1CHECK grid=coarse bq_th={beta_q_th:.6f} bq_pred_modal={bq_part_pred_c:.6f} "
        f"s_win={tols_c['s_win_part']:.6e} s_modes={tols_c['s_modes_part']:.6e} "
        f"s_sym={tols_c['s_sym_part']:.6e} s_round={tols_c['s_round']:.6e} "
        f"slack={tols_c['slack_part']:.6e} lattice={bq_part_lat_c:.6f} "
        f"resid={resid_lat_c:.6e} resid_lattice={resid_lat_c:.6e} "
        f"beta_m={extrap_c['bq_m']:.6f} beta_m2={extrap_c['bq_m2']:.6f} beta_m4={extrap_c['bq_m4']:.6f} "
        f"bq_m={extrap_c['bq_m']:.6f} bq_m2={extrap_c['bq_m2']:.6f} bq_m4={extrap_c['bq_m4']:.6f} "
        f"sigma_m={extrap_c['sigma_m']:.6e} sigma_m2={extrap_c['sigma_m2']:.6e} sigma_m4={extrap_c['sigma_m4']:.6e} "
        f"q_m={extrap_c['q_m']:.6f} p={extrap_c['p']:.6f} "
        f"beta0_3pt={extrap_c['beta0_3pt']:.6f} beta0_2pt={extrap_c['beta0_2pt']:.6f} "
        f"beta0={extrap_c['beta0']:.6f} bq0={extrap_c['bq0']:.6f} "
        f"e_R={extrap_c['e_R']:.6e} sigma_prop={extrap_c['sigma_prop']:.6e} "
        f"resid0={extrap_c['resid0']:.6e} bound0={extrap_c['bound0']:.6e} "
        f"n_macro_ens={noise_c['n_macro_per_step']} bq_pred_ens={bq_part_pred_ens_c:.6f} "
        f"lattice_ens={bq_part_lat_ens_c:.6f} "
        f"bq_coherent_mean={noise_c['bq_coherent_mean']:.6f} resid_coherent={resid_coh_c:.6e} "
        f"mean_sheet={noise_c['bq_mean']:.6f} sem={noise_c['bq_sem']:.6e} "
        f"tcrit={tcrit_c:.6f} resid_mean={resid_mean_c:.6e} "
        f"bound_mean={bound_mean_c:.6e} mean_minus_lattice={mean_minus_lat_c:.6e} "
        f"seeds={seeds_c_str}",
        flush=True,
    )
    print(
        f"H1CHECK grid=fine bq_th={beta_q_th:.6f} bq_pred_modal={bq_part_pred_f:.6f} "
        f"s_win={tols_f['s_win_part']:.6e} s_modes={tols_f['s_modes_part']:.6e} "
        f"s_sym={tols_f['s_sym_part']:.6e} s_round={tols_f['s_round']:.6e} "
        f"slack={tols_f['slack_part']:.6e} lattice={bq_part_lat_f:.6f} "
        f"resid={resid0_f:.6e} resid_lattice={resid_lat_f:.6e} "
        f"beta_m={extrap_f['bq_m']:.6f} beta_m2={extrap_f['bq_m2']:.6f} "
        f"bq_m={extrap_f['bq_m']:.6f} bq_m2={extrap_f['bq_m2']:.6f} "
        f"sigma_m={extrap_f['sigma_m']:.6e} sigma_m2={extrap_f['sigma_m2']:.6e} "
        f"beta0={extrap_f['beta0']:.6f} bq0={extrap_f['bq0']:.6f} "
        f"sigma_prop={extrap_f['sigma_prop']:.6e} resid0={extrap_f['resid0']:.6e} "
        f"bound0={extrap_f['bound0']:.6e} m2_assumed={extrap_f['m2_assumed']} "
        f'extrap_note="{extrap_f["label"]}" '
        f"n_macro_ens={noise_f['n_macro_per_step']} bq_pred_ens={bq_part_pred_ens_f:.6f} "
        f"lattice_ens={bq_part_lat_ens_f:.6f} "
        f"bq_coherent_mean={noise_f['bq_coherent_mean']:.6f} resid_coherent={resid_coh_f:.6e} "
        f"mean_sheet={noise_f['bq_mean']:.6f} sem={noise_f['bq_sem']:.6e} "
        f"tcrit={tcrit_f:.6f} resid_mean={resid_mean_f:.6e} "
        f"bound_mean={bound_mean_f:.6e} mean_minus_lattice={mean_minus_lat_f:.6e} "
        f"seeds={seeds_f_str}",
        flush=True,
    )
    print(
        f"CROSSCHECK grid=fine beta_m={extrap_f['bq_m']:.6f} beta_m2={extrap_f['bq_m2']:.6f} "
        f"beta0={beta0_f:.6f} m2_assumed={extrap_f['m2_assumed']} bq_lb={bq_lb_f:.6f} "
        f"bq_pred_part={bq_part_pred_f:.6f} bq_pred_lb={bq_lb_pred_f:.6f} "
        f"s_win_part={tols_f['s_win_part']:.6e} s_modes_part={tols_f['s_modes_part']:.6e} "
        f"s_sym_part={tols_f['s_sym_part']:.6e} s_win_lb={tols_f['s_win_lb']:.6e} "
        f"s_modes_lb={tols_f['s_modes_lb']:.6e} s_sym_lb={tols_f['s_sym_lb']:.6e} "
        f"s_round={tols_f['s_round']:.6e} slack_part={tols_f['slack_part']:.6e} "
        f"slack_lb={tols_f['slack_lb']:.6e} slack_cross={tols_f['slack_cross']:.6e} "
        f"resid_cross={resid_cross_f:.6e} bound_cross={tols_f['slack_cross']:.6e} "
        f"resid_cross_m={resid_cross_lat_f:.6e}",
        flush=True,
    )

    # Step 5 (A1): Coherent seed-mean fit vs mean of per-seed fits (at same n_macro_per_step)
    try:
        bq_coh_c = float(noise_c["bq_coherent_mean"])
        bq_coh_f = float(noise_f["bq_coherent_mean"])
        coh_minus_lat_c = float(bq_coh_c - bq_part_lat_ens_c)
        coh_minus_lat_f = float(bq_coh_f - bq_part_lat_ens_f)
        print(
            f"DIAG id=A1 bq_coherent_mean_coarse={bq_coh_c:.6f} "
            f"bq_mean_sheet_coarse={noise_c['bq_mean']:.6f} "
            f"bq_lattice_coarse={bq_part_lat_ens_c:.6f} "
            f"coh_minus_lat_coarse={coh_minus_lat_c:+.6e} "
            f"mean_minus_lat_coarse={noise_c['bq_mean'] - bq_part_lat_ens_c:+.6e} "
            f"bq_coherent_mean_fine={bq_coh_f:.6f} "
            f"bq_mean_sheet_fine={noise_f['bq_mean']:.6f} "
            f"bq_lattice_fine={bq_part_lat_ens_f:.6f} "
            f"coh_minus_lat_fine={coh_minus_lat_f:+.6e} "
            f"mean_minus_lat_fine={noise_f['bq_mean'] - bq_part_lat_ens_f:+.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=A1 status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    # Step 5 (A2): Synthetic noise pairs (at same n_macro_per_step)
    try:
        synth_c_list = []
        synth_f_list = []
        i1_seeds_c = noise_c["i1_seeds"]
        i1_seeds_f = noise_f["i1_seeds"]
        i1_lat_ens_c = noise_c["part_lattice"]["i1_part"]
        i1_lat_ens_f = noise_f["part_lattice"]["i1_part"]
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        for idx_pair in range(4):
            s0 = 2 * idx_pair
            s1 = 2 * idx_pair + 1
            i1_syn_c = i1_lat_ens_c + (i1_seeds_c[s0] - i1_seeds_c[s1]) * inv_sqrt2
            i1_syn_f = i1_lat_ens_f + (i1_seeds_f[s0] - i1_seeds_f[s1]) * inv_sqrt2
            fit_syn_c = fit_beta_q_from_envelope(
                lb_c["z_coords"], i1_syn_c, beta_q_th, z_min=z_min, z_max=z_max
            )
            fit_syn_f = fit_beta_q_from_envelope(
                lb_f["z_coords"], i1_syn_f, beta_q_th, z_min=z_min, z_max=z_max
            )
            synth_c_list.append(float(fit_syn_c["beta_q"]))
            synth_f_list.append(float(fit_syn_f["beta_q"]))
        bq_synth_mean_c = float(np.mean(synth_c_list))
        bq_synth_mean_f = float(np.mean(synth_f_list))
        synth_excess_c = float(bq_synth_mean_c - bq_part_lat_ens_c)
        synth_excess_f = float(bq_synth_mean_f - bq_part_lat_ens_f)
        synth_c_str = "[" + ",".join(f"{v:.6f}" for v in synth_c_list) + "]"
        synth_f_str = "[" + ",".join(f"{v:.6f}" for v in synth_f_list) + "]"
        print(
            f"DIAG id=A2 bq_synth_mean_coarse={bq_synth_mean_c:.6f} "
            f"synth_excess_coarse={synth_excess_c:+.6e} "
            f"sheet_excess_coarse={noise_c['bq_mean'] - bq_part_lat_ens_c:+.6e} "
            f"pairs_coarse={synth_c_str} "
            f"bq_synth_mean_fine={bq_synth_mean_f:.6f} "
            f"synth_excess_fine={synth_excess_f:+.6e} "
            f"sheet_excess_fine={noise_f['bq_mean'] - bq_part_lat_ens_f:+.6e} "
            f"pairs_fine={synth_f_str}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=A2 status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )
    print(
        f"DEPCHECK grid=coarse delta_th={tols_c['delta_th']:.6e} "
        f"delta_disc_exact={tols_c['delta_disc_exact']:.6e} "
        f"weighting={tols_c['dep_weighting']} "
        f"delta_disc_flux={tols_c['delta_disc_flux']:.6e} "
        f"delta_disc_density={tols_c['delta_disc_density']:.6e} "
        f"eps_ac={tols_c['eps_ac']:.6e} "
        f"delta_meas_lattice={delta_lat_c:.6e} "
        f"delta_disc_ens={noise_c['delta_disc']:.6e} "
        f"delta_meas_lattice_ens={noise_c['delta_lattice']:.6e} "
        f"delta_meas_mean={noise_c['delta_mean']:.6e} "
        f"sem={noise_c['delta_sem']:.6e}",
        flush=True,
    )
    print(
        f"DEPCHECK grid=fine delta_th={tols_f['delta_th']:.6e} "
        f"delta_disc_exact={tols_f['delta_disc_exact']:.6e} "
        f"weighting={tols_f['dep_weighting']} "
        f"delta_disc_flux={tols_f['delta_disc_flux']:.6e} "
        f"delta_disc_density={tols_f['delta_disc_density']:.6e} "
        f"eps_ac={tols_f['eps_ac']:.6e} "
        f"delta_meas_lattice={delta_lat_f:.6e} "
        f"delta_disc_ens={noise_f['delta_disc']:.6e} "
        f"delta_meas_lattice_ens={noise_f['delta_lattice']:.6e} "
        f"delta_meas_mean={noise_f['delta_mean']:.6e} "
        f"sem={noise_f['delta_sem']:.6e}",
        flush=True,
    )

    # Report-only 2-grid convergence orders p = log2(e_c / e_f) and fine-grid fit diagnostics
    e_lb_c = abs(bq_lb_c - beta_q_th) / beta_q_th
    e_lb_f = abs(bq_lb_f - beta_q_th) / beta_q_th
    e_part_c = abs(bq_part_lat_c - beta_q_th) / beta_q_th
    e_part_f = abs(bq_part_lat_f - beta_q_th) / beta_q_th
    p_lb = float(np.log2(e_lb_c / max(e_lb_f, 1e-30)))
    p_part = float(np.log2(e_part_c / max(e_part_f, 1e-30)))
    print(
        f"Fine report-only diagnostics: "
        f"LB: |a_s|/|a_f|={fit_lb_f['as_over_af']:.6f}, B/A={fit_lb_f['b_over_a']:.6f}, "
        f"z0={fit_lb_f['z0'] * 1e3:.4f} mm, sigma_bq={fit_lb_f['sigma_beta_q']:.6e} rad/m | "
        f"Part (sheet_lattice): |a_s|/|a_f|={fit_part_f['as_over_af']:.6f}, "
        f"B/A={fit_part_f['b_over_a']:.6f}, z0={fit_part_f['z0'] * 1e3:.4f} mm, "
        f"sigma_bq={fit_part_f['sigma_beta_q']:.6e} rad/m | "
        f"Observed orders (report-only): p_lb={p_lb:.3f}, p_part_lattice={p_part:.3f}",
        flush=True,
    )

    # Fine-grid Step 1, Step 3 & Step 4 amplitude sanity |A| > 0.5 * m * I0 and Step 4.8 DFT leakage
    assert tols_f["kappa_dc"] > tols_f["gamma1_disc"], (
        f"Identity check failed: kappa_dc={tols_f['kappa_dc']:.6f} <= gamma1_disc={tols_f['gamma1_disc']:.6f}"
    )
    min_amp_a = 0.5 * params["I1_0"]
    assert abs(fit_lb_f["A"]) > min_amp_a, (
        f"Fine LB |A|={abs(fit_lb_f['A']):.6e} <= {min_amp_a:.6e}"
    )
    assert abs(fit_part_f["A"]) > min_amp_a, (
        f"Fine Part |A|={abs(fit_part_f['A']):.6e} <= {min_amp_a:.6e}"
    )
    assert lb_f["dirichlet_leak_lb"] <= tols_f["eps_dft"], (
        f"Fine LB DFT leakage {lb_f['dirichlet_leak_lb']:.3e} > {tols_f['eps_dft']:.3e}"
    )
    assert part_f["dirichlet_leak_part"] <= tols_f["eps_dft"], (
        f"Fine Part DFT leakage {part_f['dirichlet_leak_part']:.3e} > {tols_f['eps_dft']:.3e}"
    )

    # Fine-grid Step 3 signed checks (LB, cross-code, and Particle lattice using m->0 extrapolated beta0)
    assert resid_lb_f <= tols_f["slack_lb"], (
        f"Fine LB |bq_lb/bq_lb_pred - 1| = {resid_lb_f:.6e} > slack_lb = {tols_f['slack_lb']:.6e} "
        f"(margin={tols_f['slack_lb'] - resid_lb_f:+.6e})"
    )
    assert resid_cross_f <= tols_f["slack_cross"], (
        f"Fine cross-code |beta0_P/bq_LB - bq_P_pred/bq_LB_pred| = {resid_cross_f:.6e} > "
        f"slack_cross = {tols_f['slack_cross']:.6e} (margin={tols_f['slack_cross'] - resid_cross_f:+.6e})"
    )
    assert resid0_f <= tols_f["slack_part"], (
        f"Fine Particle (sheet_lattice, m->0) |beta0_P/bq_P_pred - 1| = {resid0_f:.6e} > "
        f"slack_part = {tols_f['slack_part']:.6e} (margin={tols_f['slack_part'] - resid0_f:+.6e})"
    )

    # Fine-grid Step 4.6 entrance amplitude ("sheet_lattice") and 8-seed ("sheet") entrance amplitude
    amp_err_lb_f = float(abs(np.abs(lb_f["i1_lb"][0]) / params["I1_0"] - 1.0))
    amp_err_part_f = float(
        abs(np.abs(part_f["i1_part"][0]) / params["I1_0"] - tols_f["amp_pred_part"])
    )
    assert amp_err_lb_f <= tols_f["tol_amp_lb"], (
        f"Fine LB entrance |I1(z0)|/I1_0 - 1 = {amp_err_lb_f:.6e} > {tols_f['tol_amp_lb']:.6e} "
        f"(margin={tols_f['tol_amp_lb'] - amp_err_lb_f:+.6e})"
    )
    assert amp_err_part_f <= tols_f["tol_amp_part"], (
        f"Fine Part entrance ||I1(z0)|/I1_0 - sinc(be*dz/2)*cos(bq*dz/2)| = {amp_err_part_f:.6e} > "
        f"{tols_f['tol_amp_part']:.6e} (margin={tols_f['tol_amp_part'] - amp_err_part_f:+.6e})"
    )
    for label, noise, tols in (
        ("coarse", noise_c, tols_c),
        ("fine", noise_f, tols_f),
    ):
        amp_err_sheet = abs(noise["amp0_mean"] / params["I1_0"] - tols["amp_pred_part"])
        amp_bound_sheet = (
            tols["tol_amp_part"] + noise["tcrit"] * noise["amp0_sem"] / params["I1_0"]
        )
        assert amp_err_sheet <= amp_bound_sheet, (
            f"{label} sheet entrance ||I1(z0)|/I1_0 - sinc(be*dz/2)*cos(bq*dz/2)| = {amp_err_sheet:.6e} > "
            f"bound={amp_bound_sheet:.6e} (margin={amp_bound_sheet - amp_err_sheet:+.6e})"
        )

    # Step 1: Deterministic lattice DC depression |delta_meas,lattice - delta_disc| <= eps_AC + 1e-12
    for label, delta_lat, tols in (
        ("coarse", delta_lat_c, tols_c),
        ("fine", delta_lat_f, tols_f),
    ):
        err_dep_lat = abs(delta_lat - tols["delta_disc_exact"])
        bound_dep_lat = tols["eps_ac"] + 1e-12
        assert err_dep_lat <= bound_dep_lat, (
            f"{label} lattice DC depression |delta_meas_lattice - delta_disc_exact| = "
            f"{err_dep_lat:.6e} > eps_AC + 1e-12 = {bound_dep_lat:.6e} "
            f"(margin={bound_dep_lat - err_dep_lat:+.6e})"
        )

    # Step 2 / Step 4.3: resid_mean_c and resid_mean_f are report-only (printed in H1CHECK);
    # the asserted falsifier at finite m is |mean - bq_P,lattice| <= t * SEM (same m on both sides)
    bound_lat_c = tcrit_c * noise_c["bq_sem"]
    bound_lat_f = tcrit_f * noise_f["bq_sem"]
    assert mean_minus_lat_f <= bound_lat_f, (
        f"Fine |mean_sheet - lattice| = {mean_minus_lat_f:.6e} > t*SEM = {bound_lat_f:.6e} "
        f"(margin={bound_lat_f - mean_minus_lat_f:+.6e})"
    )

    # Step 1 / 4.5: Seeded sheet DC depression |delta_meas_mean - delta_disc| <= t * SEM + 1e-12
    for label, noise in (
        ("fine", noise_f),
        ("coarse", noise_c),
    ):
        err_dep = abs(noise["delta_mean"] - noise["delta_disc"])
        bound_dep = noise["tcrit"] * noise["delta_sem"] + 1e-12
        assert err_dep <= bound_dep, (
            f"{label} DC depression |delta_meas_mean - delta_disc_ens| = {err_dep:.6e} > "
            f"t*SEM + 1e-12 = {bound_dep:.6e} (margin={bound_dep - err_dep:+.6e})"
        )

    assert mean_minus_lat_c <= bound_lat_c, (
        f"Coarse |mean_sheet - lattice| = {mean_minus_lat_c:.6e} > t*SEM = {bound_lat_c:.6e} "
        f"(margin={bound_lat_c - mean_minus_lat_c:+.6e})"
    )

    if first_exc is not None:
        raise first_exc


@pytest.mark.slow
def test_m_scan_linearity():
    """Verify Step 4.7 m-scan (`"sheet_lattice"`, m and m/2) on coarse grid."""
    params = compute_analytic_parameters()
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params)
    z_min, z_max = fit_window(params)
    beta_q_th = params["beta_q"]
    lam_q = params["lambda_q"]
    m_half = 0.5 * MODULATION_INDEX

    lb_m, part_m = cached_baseline_pair(nx, nz, m_mod=MODULATION_INDEX)
    lb_h, part_h = cached_baseline_pair(nx, nz, m_mod=m_half)
    z_coords = lb_m["z_coords"]

    fit_lb_m = fit_beta_q_from_envelope(
        z_coords, lb_m["i1_lb"], beta_q_th, z_min=z_min, z_max=z_max
    )
    fit_lb_h = fit_beta_q_from_envelope(
        z_coords, lb_h["i1_lb"], beta_q_th, z_min=z_min, z_max=z_max
    )
    fit_part_m = fit_beta_q_from_envelope(
        z_coords, part_m["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
    )
    fit_part_h = fit_beta_q_from_envelope(
        z_coords, part_h["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
    )

    bq_lb_m = fit_lb_m["beta_q"]
    bq_lb_h = fit_lb_h["beta_q"]
    bq_part_m = fit_part_m["beta_q"]
    bq_part_h = fit_part_h["beta_q"]

    bq_shift_lb = abs(bq_lb_m / bq_lb_h - 1.0)
    bq_shift_part = abs(bq_part_m / bq_part_h - 1.0)

    mask = z_coords <= lam_q
    i2_ratio_lb_m = float(
        np.max(np.abs(lb_m["i2_lb"][mask])) / np.abs(lb_m["i1_lb"][0])
    )
    i2_ratio_lb_h = float(
        np.max(np.abs(lb_h["i2_lb"][mask])) / np.abs(lb_h["i1_lb"][0])
    )
    i2_ratio_part_m = float(
        np.max(np.abs(part_m["i2_part"][mask])) / np.abs(part_m["i1_part"][0])
    )
    i2_ratio_part_h = float(
        np.max(np.abs(part_h["i2_part"][mask])) / np.abs(part_h["i1_part"][0])
    )
    i2_bunch_m = extract_coherent_bunching_i2_ratio(
        z_coords,
        part_m["i1_part"],
        part_m["i2_part"],
        params["beta_e"],
        beta_q_th,
        lam_q,
    )
    i2_bunch_h = extract_coherent_bunching_i2_ratio(
        z_coords,
        part_h["i1_part"],
        part_h["i2_part"],
        params["beta_e"],
        beta_q_th,
        lam_q,
    )
    i2_raw_scaling = i2_ratio_part_m / max(i2_ratio_part_h, 1e-30)
    i2_bunch_scaling = i2_bunch_m / max(i2_bunch_h, 1e-30)

    extrap_c = compute_m_extrapolation(nx, nz, include_m4=True)
    extrap_f = compute_m_extrapolation(FINE_GRID[0], FINE_GRID[2], include_m4=False)

    print(
        f"\nm-scan & Richardson m->0 extrapolation (sheet_lattice, m={MODULATION_INDEX}, m/2={m_half}, m/4={0.25 * MODULATION_INDEX}): "
        f"LB (linear identity): |bq(m)/bq(m/2) - 1|={bq_shift_lb:.3e} "
        f"(bound={tols['eps_dft']:.3e}), I2/I1={i2_ratio_lb_m:.6e} | "
        f"Part coarse: beta_m={extrap_c['bq_m']:.6f}, beta_m2={extrap_c['bq_m2']:.6f}, beta_m4={extrap_c['bq_m4']:.6f}, "
        f"sigma_m={extrap_c['sigma_m']:.6e}, sigma_m2={extrap_c['sigma_m2']:.6e}, sigma_m4={extrap_c['sigma_m4']:.6e}, "
        f"q_m={extrap_c['q_m']:.6f} (report-only), p={extrap_c['p']:.6f} (report-only), "
        f"beta0_3pt={extrap_c['beta0_3pt']:.6f}, beta0_2pt={extrap_c['beta0_2pt']:.6f}, "
        f"c_m2={extrap_c['c_m2']:+.6e}, d_m4={extrap_c['d_m4']:+.6e}, "
        f"e_R={extrap_c['e_R']:.6e}, sigma_prop={extrap_c['sigma_prop']:.6e}, "
        f"slack={extrap_c['slack']:.6e}, resid0={extrap_c['resid0']:.6e} "
        f"(bound0=slack+(e_R+sigma_prop)/bq_pred={extrap_c['bound0']:.6e}), "
        f"report-only |bq(m)/bq(m/2) - 1|={bq_shift_part:.6e} (slack_part={tols['slack_part']:.6e}) | "
        f"Part fine: beta_m={extrap_f['bq_m']:.6f}, beta_m2={extrap_f['bq_m2']:.6f}, "
        f"beta0={extrap_f['beta0']:.6f} (m2_assumed={extrap_f['m2_assumed']}, {extrap_f['label']}), "
        f"sigma_prop={extrap_f['sigma_prop']:.6e}, resid0={extrap_f['resid0']:.6e} | "
        f"I2/I1 report-only: raw(m)={i2_ratio_part_m:.6e}, raw(m/2)={i2_ratio_part_h:.6e}, "
        f"raw_ratio={i2_raw_scaling:.4f}, bunch(m)={i2_bunch_m:.6e}, "
        f"bunch(m/2)={i2_bunch_h:.6e}, bunch_ratio={i2_bunch_scaling:.4f}",
        flush=True,
    )

    # LB m-invariance is an identity (linear code)
    assert bq_shift_lb <= tols["eps_dft"], (
        f"Identity (linear code) LB m-scan |bq(m)/bq(m/2) - 1| = {bq_shift_lb:.3e} > "
        f"eps_dft = {tols['eps_dft']:.3e} (margin={tols['eps_dft'] - bq_shift_lb:+.3e})"
    )
    assert abs(i2_ratio_lb_m - i2_ratio_lb_h) <= tols["eps_dft"], (
        f"Identity (linear code) LB I2/I1 m-invariance diff={abs(i2_ratio_lb_m - i2_ratio_lb_h):.3e} > "
        f"eps_dft={tols['eps_dft']:.3e}"
    )

    # Step 3: Asserted (coarse) |beta0_3pt/bq_pred - 1| <= slack + (e_R + sigma_prop)/bq_pred
    assert extrap_c["resid0"] <= extrap_c["bound0"], (
        f"Coarse Particle Richardson m->0 |beta0_3pt/bq_pred - 1| = {extrap_c['resid0']:.6e} > "
        f"slack_part + (e_R + sigma_prop)/bq_pred = {extrap_c['bound0']:.6e} "
        f"(slack={tols['slack_part']:.6e}, e_R/bq_pred={extrap_c['e_R'] / extrap_c['bq_pred']:.6e}, "
        f"sigma_prop/bq_pred={extrap_c['sigma_prop'] / extrap_c['bq_pred']:.6e}, "
        f"margin={extrap_c['bound0'] - extrap_c['resid0']:+.6e})"
    )


@pytest.mark.slow
def test_axial_vs_transverse_refinement():
    """Step 4 factorial separation of remaining particle residual (report-only `DIAG id=particle_factorial`)."""
    params = compute_analytic_parameters()
    beta_q_th = params["beta_q"]
    z_min, z_max = fit_window(params)
    b = BEAM_HALF_THICKNESS_M
    m0 = MODULATION_INDEX
    m_half = 0.5 * MODULATION_INDEX
    stored_exc: Exception | None = None

    def _x_lattice(n_val: int) -> np.ndarray:
        k_idx = np.arange(n_val, dtype=np.float64)
        return b * (2.0 * (k_idx + 0.5) / float(n_val) - 1.0)

    def _meas_and_pred(
        nx_g: int, nz_g: int, n_val: int, m_val: float
    ) -> tuple[float, float, float]:
        lb_g = cached_linearized_beam_drift(nx_g, nz_g, rho0_scale=1.0, m_mod=m0)
        part_g = cached_particle_drift(
            nx_g,
            nz_g,
            profile="sheet_lattice",
            seed=42,
            m_mod=m_val,
            qm_scale=1.0,
            n_macro_per_step=n_val,
        )
        fit_p = fit_beta_q_from_envelope(
            lb_g["z_coords"],
            part_g["i1_part"],
            beta_q_th,
            z_min=z_min,
            z_max=z_max,
        )
        meas = float(fit_p["beta_q"])
        pred = float(
            discrete_modal_prediction(
                "particles", nx_g, nz_g, params, line_x=_x_lattice(n_val)
            )
        )
        resid = float(abs(meas / pred - 1.0))
        return meas, pred, resid

    try:
        nx_c, _, nz_c = COARSE_GRID
        nx_f, _, nz_f = FINE_GRID

        # Coarse (10x474): n = 16 and 64 at m and m/2
        c16_m, c16_pred, c16_res_m = _meas_and_pred(nx_c, nz_c, 16, m0)
        c16_m2, _, c16_res_m2 = _meas_and_pred(nx_c, nz_c, 16, m_half)
        c16_bq0 = float((4.0 * c16_m2 - c16_m) / 3.0)
        c16_res0 = float(abs(c16_bq0 / c16_pred - 1.0))

        c64_m, c64_pred, c64_res_m = _meas_and_pred(nx_c, nz_c, 64, m0)
        c64_m2, _, c64_res_m2 = _meas_and_pred(nx_c, nz_c, 64, m_half)
        c64_bq0 = float((4.0 * c64_m2 - c64_m) / 3.0)
        c64_res0 = float(abs(c64_bq0 / c64_pred - 1.0))

        # Check D6 n-dependence: does modal prediction for n=64 move by the same amount as measurement?
        d6_meas_shift_m = float(c64_m - c16_m)
        d6_meas_shift_0 = float(c64_bq0 - c16_bq0)
        d6_pred_shift = float(c64_pred - c16_pred)

        # Fine (20x948): n = 32 and 128 at m and m/2
        f32_m, f32_pred, f32_res_m = _meas_and_pred(nx_f, nz_f, 32, m0)
        f32_m2, _, f32_res_m2 = _meas_and_pred(nx_f, nz_f, 32, m_half)
        f32_bq0 = float((4.0 * f32_m2 - f32_m) / 3.0)
        f32_res0 = float(abs(f32_bq0 / f32_pred - 1.0))

        f128_m, f128_pred, f128_res_m = _meas_and_pred(nx_f, nz_f, 128, m0)
        f128_m2, _, f128_res_m2 = _meas_and_pred(nx_f, nz_f, 128, m_half)
        f128_bq0 = float((4.0 * f128_m2 - f128_m) / 3.0)
        f128_res0 = float(abs(f128_bq0 / f128_pred - 1.0))

        # 10x948 (axial-only refinement, base n=16, m only) and 20x474 (transverse-only refinement, base n=32, m only)
        a_m, a_pred, a_res = _meas_and_pred(nx_c, nz_f, 16, m0)
        t_m, t_pred, t_res = _meas_and_pred(nx_f, nz_c, 32, m0)

        print(
            f"\nDIAG id=particle_factorial "
            f"coarse_n16_m_meas={c16_m:.6f} coarse_n16_pred={c16_pred:.6f} coarse_n16_m_resid={c16_res_m:.6e} "
            f"coarse_n16_m2_meas={c16_m2:.6f} coarse_n16_m2_resid={c16_res_m2:.6e} "
            f"coarse_n16_bq0={c16_bq0:.6f} coarse_n16_resid0={c16_res0:.6e} "
            f"coarse_n64_m_meas={c64_m:.6f} coarse_n64_pred={c64_pred:.6f} coarse_n64_m_resid={c64_res_m:.6e} "
            f"coarse_n64_m2_meas={c64_m2:.6f} coarse_n64_m2_resid={c64_res_m2:.6e} "
            f"coarse_n64_bq0={c64_bq0:.6f} coarse_n64_resid0={c64_res0:.6e} "
            f"d6_meas_shift_m={d6_meas_shift_m:+.6e} d6_meas_shift_bq0={d6_meas_shift_0:+.6e} "
            f"d6_pred_shift={d6_pred_shift:+.6e} "
            f"fine_n32_m_meas={f32_m:.6f} fine_n32_pred={f32_pred:.6f} fine_n32_m_resid={f32_res_m:.6e} "
            f"fine_n32_m2_meas={f32_m2:.6f} fine_n32_m2_resid={f32_res_m2:.6e} "
            f"fine_n32_bq0={f32_bq0:.6f} fine_n32_resid0={f32_res0:.6e} "
            f"fine_n128_m_meas={f128_m:.6f} fine_n128_pred={f128_pred:.6f} fine_n128_m_resid={f128_res_m:.6e} "
            f"fine_n128_m2_meas={f128_m2:.6f} fine_n128_m2_resid={f128_res_m2:.6e} "
            f"fine_n128_bq0={f128_bq0:.6f} fine_n128_resid0={f128_res0:.6e} "
            f"g10x948_n16_m_meas={a_m:.6f} g10x948_n16_pred={a_pred:.6f} g10x948_n16_m_resid={a_res:.6e} "
            f"g20x474_n32_m_meas={t_m:.6f} g20x474_n32_pred={t_pred:.6f} g20x474_n32_m_resid={t_res:.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        stored_exc = exc
        print(
            f"\nDIAG id=particle_factorial status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    if stored_exc is not None:
        raise stored_exc


@pytest.mark.slow
def test_optional_diagnostics():
    """Step 5 diagnostics A3-A5 and Step 8 optional diagnostics D3-D6 (report-only, coarse grid)."""
    params = compute_analytic_parameters()
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params)
    z_min, z_max = fit_window(params)
    beta_q_th = params["beta_q"]
    beta_e = params["beta_e"]
    first_exc: Exception | None = None

    lb_base, part_base = cached_baseline_pair(nx, nz)
    z_coords = lb_base["z_coords"]
    fit_lb_base = fit_beta_q_from_envelope(
        z_coords, lb_base["i1_lb"], beta_q_th, z_min=z_min, z_max=z_max
    )
    fit_part_base = fit_beta_q_from_envelope(
        z_coords, part_base["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
    )
    bq_lb_base = fit_lb_base["beta_q"]
    bq_part_base = fit_part_base["beta_q"]

    print(flush=True)

    # Step 5 (A3): Binned v_z mean/rms and gamma vs gamma_pred over DFT window (m=0.015 and m=0)
    part_sheet42_m = None
    bq_sheet42_m = float("nan")
    try:
        part_sheet42_m = cached_particle_drift(
            nx, nz, profile="sheet", seed=42, m_mod=MODULATION_INDEX, qm_scale=1.0
        )
        fit_sheet42_m = fit_beta_q_from_envelope(
            z_coords, part_sheet42_m["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
        )
        bq_sheet42_m = float(fit_sheet42_m["beta_q"])
        sigma_v_over_u_hwarm_a3 = float(
            np.sqrt(max(bq_sheet42_m**2 - bq_part_base**2, 0.0))
            / (np.sqrt(3.0) * beta_e)
        )
        part_lat_m0 = run_particle_beam_drift(
            nx,
            nz,
            beam_ref=lb_base["beam"],
            qm_scale=1.0,
            seed=42,
            m_mod=0.0,
            profile="sheet_lattice",
        )
        part_sheet42_m0 = run_particle_beam_drift(
            nx,
            nz,
            beam_ref=lb_base["beam"],
            qm_scale=1.0,
            seed=42,
            m_mod=0.0,
            profile="sheet",
        )
        print(
            f"DIAG id=A3 bq_lat={bq_part_base:.6f} bq_sheet42={bq_sheet42_m:.6f} "
            f"sigma_v_over_u_hwarm={sigma_v_over_u_hwarm_a3:.6e} "
            f"sigma_v_over_u_lat_m={part_base['sigma_vz_over_u']:.6e} "
            f"sigma_v_over_u_sheet42_m={part_sheet42_m['sigma_vz_over_u']:.6e} "
            f"sigma_v_over_u_lat_m0={part_lat_m0['sigma_vz_over_u']:.6e} "
            f"sigma_v_over_u_sheet42_m0={part_sheet42_m0['sigma_vz_over_u']:.6e} "
            f"vz_lat_m={part_base['vz_mean_coh']:.3f} vz_sheet42_m={part_sheet42_m['vz_mean_coh']:.3f} "
            f"vz_lat_m0={part_lat_m0['vz_mean_coh']:.3f} vz_sheet42_m0={part_sheet42_m0['vz_mean_coh']:.3f} "
            f"gamma_lat_m={part_base['gamma_mean']:.8f} gamma_pred_lat_m={part_base['gamma_pred_mean']:.8f} "
            f"gamma_diff_lat_m={part_base['gamma_diff_mean']:+.6e} "
            f"gamma_sheet42_m={part_sheet42_m['gamma_mean']:.8f} gamma_pred_sheet42_m={part_sheet42_m['gamma_pred_mean']:.8f} "
            f"gamma_diff_sheet42_m={part_sheet42_m['gamma_diff_mean']:+.6e} "
            f"gamma_lat_m0={part_lat_m0['gamma_mean']:.8f} gamma_pred_lat_m0={part_lat_m0['gamma_pred_mean']:.8f} "
            f"gamma_diff_lat_m0={part_lat_m0['gamma_diff_mean']:+.6e} "
            f"gamma_sheet42_m0={part_sheet42_m0['gamma_mean']:.8f} gamma_pred_sheet42_m0={part_sheet42_m0['gamma_pred_mean']:.8f} "
            f"gamma_diff_sheet42_m0={part_sheet42_m0['gamma_diff_mean']:+.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=A3 status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    # Step 5 (A4): Frozen random sheet (_step_rng returns default_rng(self.seed) every step)
    try:
        orig_step_rng = BeamInjector._step_rng
        BeamInjector._step_rng = lambda self, step_index: np.random.default_rng(
            self.seed
        )
        try:
            part_froz_42 = run_particle_beam_drift(
                nx,
                nz,
                beam_ref=lb_base["beam"],
                qm_scale=1.0,
                seed=42,
                profile="sheet",
            )
            part_froz_43 = run_particle_beam_drift(
                nx,
                nz,
                beam_ref=lb_base["beam"],
                qm_scale=1.0,
                seed=43,
                profile="sheet",
            )
        finally:
            BeamInjector._step_rng = orig_step_rng

        fit_froz_42 = fit_beta_q_from_envelope(
            z_coords, part_froz_42["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
        )
        fit_froz_43 = fit_beta_q_from_envelope(
            z_coords, part_froz_43["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
        )
        bq_froz_42 = float(fit_froz_42["beta_q"])
        bq_froz_43 = float(fit_froz_43["beta_q"])
        bq_froz_mean = 0.5 * (bq_froz_42 + bq_froz_43)

        k_idx_16 = np.arange(16, dtype=np.float64)
        u_strat_42 = np.random.default_rng(42).uniform(0.0, 1.0, size=16)
        u_strat_43 = np.random.default_rng(43).uniform(0.0, 1.0, size=16)
        x_froz_42 = BEAM_HALF_THICKNESS_M * (2.0 * (k_idx_16 + u_strat_42) / 16.0 - 1.0)
        x_froz_43 = BEAM_HALF_THICKNESS_M * (2.0 * (k_idx_16 + u_strat_43) / 16.0 - 1.0)
        bq_pred_froz_42 = float(
            discrete_modal_prediction("particles", nx, nz, params, line_x=x_froz_42)
        )
        bq_pred_froz_43 = float(
            discrete_modal_prediction("particles", nx, nz, params, line_x=x_froz_43)
        )
        bq_pred_froz_mean = 0.5 * (bq_pred_froz_42 + bq_pred_froz_43)
        sigma_v_over_u_froz_mean = 0.5 * (
            part_froz_42["sigma_vz_over_u"] + part_froz_43["sigma_vz_over_u"]
        )
        sigma_v_over_u_hwarm_a4 = float(
            np.sqrt(max(bq_froz_mean**2 - bq_part_base**2, 0.0))
            / (np.sqrt(3.0) * beta_e)
        )
        gamma_diff_froz_mean = 0.5 * (
            part_froz_42["gamma_diff_mean"] + part_froz_43["gamma_diff_mean"]
        )
        print(
            f"DIAG id=A4 bq_froz_42={bq_froz_42:.6f} bq_pred_42={bq_pred_froz_42:.6f} "
            f"bq_froz_43={bq_froz_43:.6f} bq_pred_43={bq_pred_froz_43:.6f} "
            f"bq_froz_mean={bq_froz_mean:.6f} bq_pred_mean={bq_pred_froz_mean:.6f} "
            f"excess_vs_lat={bq_froz_mean - bq_part_base:+.6e} "
            f"pred_excess_vs_lat={bq_pred_froz_mean - tols['bq_pred_part']:+.6e} "
            f"resid_vs_modal={abs(bq_froz_mean / bq_pred_froz_mean - 1.0):.6e} "
            f"sigma_v_over_u_hwarm={sigma_v_over_u_hwarm_a4:.6e} "
            f"sigma_v_over_u_meas={sigma_v_over_u_froz_mean:.6e} "
            f"gamma_diff_froz={gamma_diff_froz_mean:+.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=A4 status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    # Step 5/6 (A5): low-n ensemble (n=16, 8 seeds) + n_macro_per_step = 16, 64, 128 (single-seed), 256
    # + gather="momentum" alternative probe (vs default "galerkin")
    try:
        noise_low_n16 = measure_empirical_noise(COARSE_GRID, n_macro_per_step=16)
        if part_sheet42_m is None:
            part_sheet42_m = cached_particle_drift(
                nx, nz, profile="sheet", seed=42, m_mod=MODULATION_INDEX, qm_scale=1.0
            )
            fit_sheet42_m = fit_beta_q_from_envelope(
                z_coords, part_sheet42_m["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
            )
            bq_sheet42_m = float(fit_sheet42_m["beta_q"])
        part_sheet43_n16 = cached_particle_drift(
            nx, nz, profile="sheet", seed=43, m_mod=MODULATION_INDEX, qm_scale=1.0
        )
        fit_sheet43_n16 = fit_beta_q_from_envelope(
            z_coords, part_sheet43_n16["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
        )
        bq_a5_n16 = 0.5 * (bq_sheet42_m + float(fit_sheet43_n16["beta_q"]))
        sv_a5_n16 = 0.5 * (
            part_sheet42_m["sigma_vz_over_u"] + part_sheet43_n16["sigma_vz_over_u"]
        )
        gdiff_a5_n16 = 0.5 * (
            part_sheet42_m["gamma_diff_mean"] + part_sheet43_n16["gamma_diff_mean"]
        )
        hwarm_a5_n16 = float(
            np.sqrt(max(bq_a5_n16**2 - bq_part_base**2, 0.0)) / (np.sqrt(3.0) * beta_e)
        )

        part_s42_n64 = cached_particle_drift(
            nx,
            nz,
            profile="sheet",
            seed=42,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=64,
        )
        part_s43_n64 = cached_particle_drift(
            nx,
            nz,
            profile="sheet",
            seed=43,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=64,
        )
        part_lat_n64 = cached_particle_drift(
            nx,
            nz,
            profile="sheet_lattice",
            seed=42,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=64,
        )
        bq_lat_n64 = float(
            fit_beta_q_from_envelope(
                z_coords, part_lat_n64["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
            )["beta_q"]
        )
        bq_s42_n64 = float(
            fit_beta_q_from_envelope(
                z_coords, part_s42_n64["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
            )["beta_q"]
        )
        bq_s43_n64 = float(
            fit_beta_q_from_envelope(
                z_coords, part_s43_n64["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
            )["beta_q"]
        )
        bq_a5_n64 = 0.5 * (bq_s42_n64 + bq_s43_n64)
        sv_a5_n64 = 0.5 * (
            part_s42_n64["sigma_vz_over_u"] + part_s43_n64["sigma_vz_over_u"]
        )
        gdiff_a5_n64 = 0.5 * (
            part_s42_n64["gamma_diff_mean"] + part_s43_n64["gamma_diff_mean"]
        )
        hwarm_a5_n64 = float(
            np.sqrt(max(bq_a5_n64**2 - bq_lat_n64**2, 0.0)) / (np.sqrt(3.0) * beta_e)
        )

        # Single-seed n = 128 point (seed=42) + lattice reference at n = 128
        part_s42_n128 = cached_particle_drift(
            nx,
            nz,
            profile="sheet",
            seed=42,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=128,
        )
        part_lat_n128 = cached_particle_drift(
            nx,
            nz,
            profile="sheet_lattice",
            seed=42,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=128,
        )
        bq_lat_n128 = float(
            fit_beta_q_from_envelope(
                z_coords, part_lat_n128["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
            )["beta_q"]
        )
        bq_a5_n128 = float(
            fit_beta_q_from_envelope(
                z_coords, part_s42_n128["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
            )["beta_q"]
        )
        sv_a5_n128 = float(part_s42_n128["sigma_vz_over_u"])
        gdiff_a5_n128 = float(part_s42_n128["gamma_diff_mean"])
        hwarm_a5_n128 = float(
            np.sqrt(max(bq_a5_n128**2 - bq_lat_n128**2, 0.0)) / (np.sqrt(3.0) * beta_e)
        )

        part_s42_n256 = cached_particle_drift(
            nx,
            nz,
            profile="sheet",
            seed=42,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=256,
        )
        part_s43_n256 = cached_particle_drift(
            nx,
            nz,
            profile="sheet",
            seed=43,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=256,
        )
        part_lat_n256 = cached_particle_drift(
            nx,
            nz,
            profile="sheet_lattice",
            seed=42,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=256,
        )
        bq_lat_n256 = float(
            fit_beta_q_from_envelope(
                z_coords, part_lat_n256["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
            )["beta_q"]
        )
        bq_s42_n256 = float(
            fit_beta_q_from_envelope(
                z_coords, part_s42_n256["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
            )["beta_q"]
        )
        bq_s43_n256 = float(
            fit_beta_q_from_envelope(
                z_coords, part_s43_n256["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
            )["beta_q"]
        )
        bq_a5_n256 = 0.5 * (bq_s42_n256 + bq_s43_n256)
        sv_a5_n256 = 0.5 * (
            part_s42_n256["sigma_vz_over_u"] + part_s43_n256["sigma_vz_over_u"]
        )
        gdiff_a5_n256 = 0.5 * (
            part_s42_n256["gamma_diff_mean"] + part_s43_n256["gamma_diff_mean"]
        )
        hwarm_a5_n256 = float(
            np.sqrt(max(bq_a5_n256**2 - bq_lat_n256**2, 0.0)) / (np.sqrt(3.0) * beta_e)
        )

        # Alternative ParticleSolver gather variant ("momentum" vs default "galerkin")
        part_lat_n16_mom = cached_particle_drift(
            nx,
            nz,
            profile="sheet_lattice",
            seed=42,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=16,
            gather="momentum",
        )
        part_s42_n16_mom = cached_particle_drift(
            nx,
            nz,
            profile="sheet",
            seed=42,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=16,
            gather="momentum",
        )
        bq_lat_n16_mom = float(
            fit_beta_q_from_envelope(
                z_coords,
                part_lat_n16_mom["i1_part"],
                beta_q_th,
                z_min=z_min,
                z_max=z_max,
            )["beta_q"]
        )
        bq_n16_mom = float(
            fit_beta_q_from_envelope(
                z_coords,
                part_s42_n16_mom["i1_part"],
                beta_q_th,
                z_min=z_min,
                z_max=z_max,
            )["beta_q"]
        )
        part_lat_n128_mom = cached_particle_drift(
            nx,
            nz,
            profile="sheet_lattice",
            seed=42,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=128,
            gather="momentum",
        )
        part_s42_n128_mom = cached_particle_drift(
            nx,
            nz,
            profile="sheet",
            seed=42,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=128,
            gather="momentum",
        )
        bq_lat_n128_mom = float(
            fit_beta_q_from_envelope(
                z_coords,
                part_lat_n128_mom["i1_part"],
                beta_q_th,
                z_min=z_min,
                z_max=z_max,
            )["beta_q"]
        )
        bq_n128_mom = float(
            fit_beta_q_from_envelope(
                z_coords,
                part_s42_n128_mom["i1_part"],
                beta_q_th,
                z_min=z_min,
                z_max=z_max,
            )["beta_q"]
        )
        print(
            f"DIAG id=A5 "
            f"low_n16_ens_mean={noise_low_n16['bq_mean']:.6f} "
            f"low_n16_ens_sem={noise_low_n16['bq_sem']:.6e} "
            f"low_n16_ens_excess={noise_low_n16['bq_mean'] - noise_low_n16['bq_lattice']:+.6e} "
            f"bq_n16={bq_a5_n16:.6f} bq_lat_n16={bq_part_base:.6f} excess_n16={bq_a5_n16 - bq_part_base:+.6e} "
            f"sigma_v_over_u_hwarm_n16={hwarm_a5_n16:.6e} sigma_v_over_u_n16={sv_a5_n16:.6e} "
            f"sigma_v_sq_n16={sv_a5_n16**2:.6e} gamma_diff_n16={gdiff_a5_n16:+.6e} "
            f"bq_n64={bq_a5_n64:.6f} bq_lat_n64={bq_lat_n64:.6f} excess_n64={bq_a5_n64 - bq_lat_n64:+.6e} "
            f"sigma_v_over_u_hwarm_n64={hwarm_a5_n64:.6e} sigma_v_over_u_n64={sv_a5_n64:.6e} "
            f"sigma_v_sq_n64={sv_a5_n64**2:.6e} gamma_diff_n64={gdiff_a5_n64:+.6e} "
            f"bq_n128={bq_a5_n128:.6f} bq_lat_n128={bq_lat_n128:.6f} excess_n128={bq_a5_n128 - bq_lat_n128:+.6e} "
            f"sigma_v_over_u_hwarm_n128={hwarm_a5_n128:.6e} sigma_v_over_u_n128={sv_a5_n128:.6e} "
            f"sigma_v_sq_n128={sv_a5_n128**2:.6e} gamma_diff_n128={gdiff_a5_n128:+.6e} "
            f"bq_n256={bq_a5_n256:.6f} bq_lat_n256={bq_lat_n256:.6f} excess_n256={bq_a5_n256 - bq_lat_n256:+.6e} "
            f"sigma_v_over_u_hwarm_n256={hwarm_a5_n256:.6e} sigma_v_over_u_n256={sv_a5_n256:.6e} "
            f"sigma_v_sq_n256={sv_a5_n256**2:.6e} gamma_diff_n256={gdiff_a5_n256:+.6e} "
            f"gather_alt=momentum "
            f"bq_lat_n16_momentum={bq_lat_n16_mom:.6f} bq_n16_momentum={bq_n16_mom:.6f} "
            f"excess_n16_momentum={bq_n16_mom - bq_lat_n16_mom:+.6e} "
            f"sigma_v_over_u_n16_momentum={part_s42_n16_mom['sigma_vz_over_u']:.6e} "
            f"gamma_diff_n16_momentum={part_s42_n16_mom['gamma_diff_mean']:+.6e} "
            f"bq_lat_n128_momentum={bq_lat_n128_mom:.6f} bq_n128_momentum={bq_n128_mom:.6f} "
            f"excess_n128_momentum={bq_n128_mom - bq_lat_n128_mom:+.6e} "
            f"sigma_v_over_u_n128_momentum={part_s42_n128_mom['sigma_vz_over_u']:.6e} "
            f"gamma_diff_n128_momentum={part_s42_n128_mom['gamma_diff_mean']:+.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=A5 status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    # D3: LinearizedBeam with rho0 on x-nodes equal to the CIC deposit of M3 Fibonacci x-positions
    try:
        x_fib = _fibonacci_disk_x_coords(16, radius=BEAM_HALF_THICKNESS_M, x0=0.0)
        cic_rho0 = _CicNodeRho0(
            params["rho_dep"],
            x_fib,
            BEAM_HALF_THICKNESS_M,
            TUNNEL_HALF_WIDTH_M,
            nx,
            lb_base["beam"],
        )
        lb_d3 = run_linearized_beam_drift(nx, nz, rho0_profile=cic_rho0)
        fit_lb_d3 = fit_beta_q_from_envelope(
            z_coords, lb_d3["i1_lb"], beta_q_th, z_min=z_min, z_max=z_max
        )
        bq_lb_d3 = float(fit_lb_d3["beta_q"])
        w_ent_lb = np.asarray(lb_base["beam"]._w_perp_eff[0, :], dtype=np.float64)
        bq_lb_d3_pred = float(
            discrete_modal_prediction(
                "lb",
                nx,
                nz,
                params,
                line_w=cic_rho0.w_cic_nodes[1:-1],
                entrance_profile=w_ent_lb,
            )
        )
        slack_lb_d3 = float(
            prediction_slack(
                "lb",
                nx,
                nz,
                params,
                line_w=cic_rho0.w_cic_nodes[1:-1],
                entrance_profile=w_ent_lb,
            )["slack"]
        )
        resid_lb_d3 = abs(bq_lb_d3 / bq_lb_d3_pred - 1.0)
        shift_lb_d3 = bq_lb_d3 / bq_lb_base - 1.0

        part_m3 = _cached_m3_disk_run(nx, nz)
        fit_part_m3 = fit_beta_q_from_envelope(
            z_coords, part_m3["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
        )
        bq_part_m3 = float(fit_part_m3["beta_q"])
        bq_part_m3_pred = float(
            discrete_modal_prediction("particles", nx, nz, params, line_x=x_fib)
        )
        slack_part_m3 = float(
            prediction_slack("particles", nx, nz, params, line_x=x_fib)["slack"]
        )
        resid_part_m3 = abs(bq_part_m3 / bq_part_m3_pred - 1.0)
        shift_part_m3 = bq_part_m3 / bq_part_base - 1.0
        print(
            f"DIAG id=D3 bq_lb_d3={bq_lb_d3:.6f} bq_lb_d3_pred={bq_lb_d3_pred:.6f} "
            f"resid_lb_d3={resid_lb_d3:.6e} slack_lb_d3={slack_lb_d3:.6e} "
            f"shift_lb={shift_lb_d3:.6e} "
            f"bq_part_m3={bq_part_m3:.6f} bq_part_m3_pred={bq_part_m3_pred:.6f} "
            f"resid_part_m3={resid_part_m3:.6e} slack_part_m3={slack_part_m3:.6e} "
            f"shift_part_m3={shift_part_m3:.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=D3 status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    # D4: ParticleSolver._backpush_injected_slice patched to a no-op
    try:
        orig_backpush = ParticleSolver._backpush_injected_slice
        ParticleSolver._backpush_injected_slice = lambda self, *args, **kwargs: None
        try:
            part_d4 = run_particle_beam_drift(
                nx,
                nz,
                beam_ref=lb_base["beam"],
                qm_scale=1.0,
                seed=42,
                profile="sheet_lattice",
            )
        finally:
            ParticleSolver._backpush_injected_slice = orig_backpush
        fit_d4 = fit_beta_q_from_envelope(
            z_coords, part_d4["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
        )
        rel_diff_d4 = abs(fit_d4["beta_q"] / bq_part_base - 1.0)
        print(
            f"DIAG id=D4 bq_no_backpush={fit_d4['beta_q']:.6f} "
            f"rel_diff={rel_diff_d4:.6e} slack_part={tols['slack_part']:.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=D4 status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    # D5: cfln = 0.25
    try:
        part_d5 = run_particle_beam_drift(
            nx,
            nz,
            beam_ref=lb_base["beam"],
            qm_scale=1.0,
            seed=42,
            profile="sheet_lattice",
            cfln=0.25,
        )
        fit_d5 = fit_beta_q_from_envelope(
            z_coords, part_d5["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
        )
        rel_diff_d5 = abs(fit_d5["beta_q"] / bq_part_base - 1.0)
        omega_dt_sq = float((params["omega"] * part_base["dt"]) ** 2)
        print(
            f"DIAG id=D5 cfln=0.25 bq_cfl025={fit_d5['beta_q']:.6f} "
            f"rel_diff={rel_diff_d5:.6e} omega_dt_sq={omega_dt_sq:.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=D5 status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    # D6: n_macro_per_step = 16 and 64
    try:
        part_d6_64 = cached_particle_drift(
            nx,
            nz,
            profile="sheet_lattice",
            seed=42,
            m_mod=MODULATION_INDEX,
            qm_scale=1.0,
            n_macro_per_step=64,
        )
        fit_d6_64 = fit_beta_q_from_envelope(
            z_coords, part_d6_64["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
        )
        k_64 = np.arange(64, dtype=np.float64)
        x_lat_64 = BEAM_HALF_THICKNESS_M * (2.0 * (k_64 + 0.5) / 64.0 - 1.0)
        pred_n16 = float(tols["bq_pred_part"])
        pred_n64 = float(
            discrete_modal_prediction("particles", nx, nz, params, line_x=x_lat_64)
        )
        rel_diff_d6 = abs(fit_d6_64["beta_q"] / bq_part_base - 1.0)
        print(
            f"DIAG id=D6 bq_n16={bq_part_base:.6f} bq_n64={fit_d6_64['beta_q']:.6f} "
            f"pred_n16={pred_n16:.6f} pred_n64={pred_n64:.6f} "
            f"meas_shift={fit_d6_64['beta_q'] - bq_part_base:+.6e} "
            f"pred_shift={pred_n64 - pred_n16:+.6e} "
            f"rel_diff={rel_diff_d6:.6e} slack_part={tols['slack_part']:.6e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        if first_exc is None:
            first_exc = exc
        print(
            f"DIAG id=D6 status=ERROR type={type(exc).__name__} msg={str(exc)[:200]}",
            flush=True,
        )

    if first_exc is not None:
        raise first_exc
