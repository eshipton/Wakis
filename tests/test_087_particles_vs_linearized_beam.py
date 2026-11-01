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
    FINE_GRID,
    GUIDE_BZ_TESLA,
    MODULATION_INDEX,
    PERIODIC_LY_M,
    TUNNEL_HALF_WIDTH_M,
    TUNNEL_LENGTH_M,
    cached_baseline_pair,
    compute_analytic_parameters,
    compute_spatial_spectrum,
    derived_tolerances,
    extract_coherent_bunching_i2_ratio,
    fit_beta_q_from_envelope,
    fit_window,
    format_error_budget,
    measure_empirical_noise,
    predict_particle_beta_q,
    run_linearized_beam_drift,
    run_particle_beam_drift,
)
from wakis import GridFIT3D, SolverFIT3D
from wakis.beam_fluid import LinearizedBeam
from wakis.estimate import estimate, preflight
from wakis.particles import BeamInjector, ParticleSolver, Species


class _SemicircularRho0:
    """Callable semicircular transverse charge density `rho0(x, y, z) prop sqrt(1 - x^2/b^2)` with same `I0`."""

    def __init__(self, rho_dep: float, b: float, a: float, nx: int):
        self.b = float(b)
        dx = 2.0 * float(a) / int(nx)
        x_nodes = np.linspace(-float(a), float(a), int(nx) + 1)[: int(nx)]
        in_beam = np.abs(x_nodes) < self.b
        profile_nodes = np.zeros_like(x_nodes)
        profile_nodes[in_beam] = np.sqrt(1.0 - (x_nodes[in_beam] / self.b) ** 2)
        discrete_area_weight = float(dx * np.sum(profile_nodes))
        self.rho_peak = float((2.0 * self.b * rho_dep) / discrete_area_weight)

    def __call__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        rel = np.clip(x / self.b, -1.0, 1.0)
        return np.where(
            np.abs(x) < self.b,
            self.rho_peak * np.sqrt(1.0 - rel**2),
            0.0,
        )

    def __float__(self) -> float:
        return self.rho_peak


def _fibonacci_disk_lattice_x(self: BeamInjector, n_macro: int) -> np.ndarray:
    """Return Fibonacci disk x-coordinates (semicircular marginal) for M3."""
    k_idx = np.arange(n_macro, dtype=np.float64)
    r_k = self.radius * np.sqrt((k_idx + 0.5) / float(n_macro))
    phi_k = k_idx * np.pi * (3.0 - np.sqrt(5.0))
    dx_k = r_k * np.cos(phi_k)
    dx_k -= float(np.mean(dx_k))
    return self.x0 + dx_k


@lru_cache(maxsize=4)
def _cached_m3_disk_run(nx: int, nz: int) -> dict:
    """Cached M3 run with `BeamInjector._sheet_lattice_x` patched to Fibonacci disk x-coordinates."""
    lb_res, _ = cached_baseline_pair(int(nx), int(nz))
    orig_sheet_lat = BeamInjector._sheet_lattice_x
    BeamInjector._sheet_lattice_x = _fibonacci_disk_lattice_x
    try:
        return run_particle_beam_drift(
            int(nx),
            int(nz),
            beam_ref=lb_res["beam"],
            qm_scale=1.0,
            seed=42,
            profile="sheet_lattice",
        )
    finally:
        BeamInjector._sheet_lattice_x = orig_sheet_lat


def test_preflight_and_resolution_criteria():
    """Verify `wakis.estimate` preflight, resolution rules, and two-wave fit-window span >= 1.25 * lambda_q."""
    params = compute_analytic_parameters(qm_scale=1.0)
    params_mut = compute_analytic_parameters(qm_scale=2.0)

    # Fit window [3/T1, (Lz - L_ramp) - 3/T1] must span at least 1.25 * lambda_q
    # for both the baseline and the |q/m| x 2 mutation
    for label, p_cur in (("baseline", params), ("qm_mutation", params_mut)):
        z_min, z_max = fit_window(p_cur)
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
            shape=1,
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

        assert len(injected_slices) == 3
        all_x = np.concatenate([s[0] for s in injected_slices])
        all_y = np.concatenate([s[1] for s in injected_slices])
        y_max_abs = float(np.max(np.abs(all_y)))
        x_min = float(np.min(all_x))
        x_max = float(np.max(all_x))

        strata_ok = bool(np.all(all_y == 0.0) and np.all(np.abs(all_x) <= b))
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


def test_space_charge_wave_envelope_entrance_and_spectrum():
    """Verify coarse-grid signed checks 4.1, 4.2, 4.4, 4.6, 4.8, 4.9, and spatial spectrum."""
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
    bq_th_at_delta, bq_part_pred = predict_particle_beta_q(delta_meas_lat, tols)

    bq_lb_pred = tols["bq_lb_pred"]
    s_lb = tols["s_lb"]
    s_part = tols["s_part"]
    s_cross = tols["s_cross"]

    resid_lb = abs(bq_lb / bq_lb_pred - 1.0)
    resid_lat = abs(bq_part_lat / bq_part_pred - 1.0)
    resid_cross = abs(bq_part_lat / bq_lb - bq_part_pred / bq_lb_pred)

    print(flush=True)
    print(format_error_budget(nx, nz, tols), flush=True)
    print(
        f"LBCHECK grid=coarse bq_th={beta_q_th:.6f} bq_lb_pred={bq_lb_pred:.6f} "
        f"bq_lb={bq_lb:.6f} resid={resid_lb:.6e} s_lb={s_lb:.6e} "
        f"eps_ax={tols['eps_ax']:.6e} eps_x_fit={tols['eps_x_fit']:.6e} "
        f"eps_fit={tols['eps_fit']:.6e}",
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
        f"delta_th_disc={tols['delta_th_disc']:.6e}, rho_dc_rel={delta_rho_lat:.6e}), "
        f"bq_th(delta_meas)={bq_th_at_delta:.6f} rad/m, bq_part_pred={bq_part_pred:.6f} rad/m "
        f"({bq_part_pred / beta_q_th - 1.0:+.4%} vs bq_th), resid_lat={resid_lat:.6e} (s_part={s_part:.6e})",
        flush=True,
    )

    # Step 4.9: Budget-sum asserts (identity)
    bq_lb_pred_check = (
        0.5
        * (tols["k_plus"] - tols["k_minus"])
        * (1.0 + tols["eps_x_fit"])
        * (1.0 + tols["eps_fit"])
    )
    s_lb_check = (
        abs(tols["eps_ax"]) + abs(tols["eps_x_fit"]) + abs(tols["eps_fit"])
    ) ** 2 + abs(tols["eps_fit_grid"] - tols["eps_fit_cont"])
    s_part_check = (
        abs(tols["eps_cic"])
        + abs(tols["eps_z_fit"])
        + abs(tols["eps_x_fit"])
        + abs(tols["eps_fit"])
    ) ** 2 + abs(tols["eps_fit_grid"] - tols["eps_fit_cont"])
    assert np.isclose(tols["bq_lb_pred"], bq_lb_pred_check, rtol=0.0, atol=1e-15), (
        f"Identity check failed for bq_lb_pred: val={tols['bq_lb_pred']:.16e}, "
        f"expected={bq_lb_pred_check:.16e}, diff={abs(tols['bq_lb_pred'] - bq_lb_pred_check):.3e}"
    )
    assert np.isclose(tols["s_lb"], s_lb_check, rtol=0.0, atol=1e-15), (
        f"Identity check failed for s_lb: val={tols['s_lb']:.16e}, "
        f"expected={s_lb_check:.16e}, diff={abs(tols['s_lb'] - s_lb_check):.3e}"
    )
    assert np.isclose(tols["s_part"], s_part_check, rtol=0.0, atol=1e-15), (
        f"Identity check failed for s_part: val={tols['s_part']:.16e}, "
        f"expected={s_part_check:.16e}, diff={abs(tols['s_part'] - s_part_check):.3e}"
    )
    assert np.isclose(
        tols["s_cross"], tols["s_lb"] + tols["s_part"], rtol=0.0, atol=1e-15
    ), (
        f"Identity check failed for s_cross: val={tols['s_cross']:.16e}, "
        f"expected={tols['s_lb'] + tols['s_part']:.16e}"
    )
    assert np.isclose(
        tols["tol_amp_part"], tols["tol_amp_lb"] + 1e-12, rtol=0.0, atol=1e-15
    ), (
        f"Identity check failed for tol_amp_part: val={tols['tol_amp_part']:.16e}, "
        f"expected={tols['tol_amp_lb'] + 1e-12:.16e}"
    )

    # Step 4.8: DFT leakage identity (round-off)
    assert lb_res["dirichlet_leak_lb"] <= tols["eps_dft"], (
        f"Identity (round-off) DFT leakage LB={lb_res['dirichlet_leak_lb']:.3e} > "
        f"bound={tols['eps_dft']:.3e} (margin={tols['eps_dft'] - lb_res['dirichlet_leak_lb']:+.3e})"
    )
    assert part_res["dirichlet_leak_part"] <= tols["eps_dft"], (
        f"Identity (round-off) DFT leakage Part={part_res['dirichlet_leak_part']:.3e} > "
        f"bound={tols['eps_dft']:.3e} (margin={tols['eps_dft'] - part_res['dirichlet_leak_part']:+.3e})"
    )

    # Step 3: Amplitude sanity check |A| > 0.5 * m * I0
    min_amp_a = 0.5 * params["I1_0"]
    assert abs(fit_lb["A"]) > min_amp_a, (
        f"LB fit |A|={abs(fit_lb['A']):.6e} <= 0.5*m*I0={min_amp_a:.6e} "
        f"(margin={abs(fit_lb['A']) - min_amp_a:+.6e})"
    )
    assert abs(fit_part["A"]) > min_amp_a, (
        f"Part fit |A|={abs(fit_part['A']):.6e} <= 0.5*m*I0={min_amp_a:.6e} "
        f"(margin={abs(fit_part['A']) - min_amp_a:+.6e})"
    )

    # Observable (ii): spatial spectrum exhibits fast and slow peaks at beta_e +- beta_q
    beta_e = params["beta_e"]
    dz = TUNNEL_LENGTH_M / nz
    k_fast_th = beta_e - beta_q_th
    k_slow_th = beta_e + beta_q_th
    delta_k_disp = float(np.arcsin(k_slow_th * dz) / dz - k_slow_th)
    tol_spec_k = float(
        beta_q_th / (2.0 * np.pi)
        + delta_k_disp
        + abs(tols["eps_ax"]) * beta_q_th
        + s_lb * beta_q_th
    )

    for label, phasor in (("LinearizedBeam", i1_lb), ("ParticleSolver", i1_part)):
        kz_grid, spec, k_fast, k_slow = compute_spatial_spectrum(
            z_coords, phasor, beta_e, beta_q_th, z_max=params["lambda_q"]
        )
        idx_fast = int(np.argmin(np.abs(kz_grid - k_fast)))
        idx_slow = int(np.argmin(np.abs(kz_grid - k_slow)))
        valley_min = float(np.min(spec[idx_fast : idx_slow + 1]))
        valley_bound = 0.25 * float(min(spec[idx_fast], spec[idx_slow]))
        assert valley_min < valley_bound, (
            f"{label} spectrum valley={valley_min:.6e} >= bound={valley_bound:.6e} "
            f"(margin={valley_bound - valley_min:+.6e})"
        )
        err_fast = abs(k_fast - k_fast_th)
        err_slow = abs(k_slow - k_slow_th)
        assert err_fast <= tol_spec_k, (
            f"{label} fast-wave peak err={err_fast:.4f} rad/m > bound={tol_spec_k:.4f} rad/m "
            f"(margin={tol_spec_k - err_fast:+.4f} rad/m)"
        )
        assert err_slow <= tol_spec_k, (
            f"{label} slow-wave peak err={err_slow:.4f} rad/m > bound={tol_spec_k:.4f} rad/m "
            f"(margin={tol_spec_k - err_slow:+.4f} rad/m)"
        )

    # Step 4.1: LinearizedBeam deterministic prediction check
    assert resid_lb <= s_lb, (
        f"Coarse LB |bq_lb/bq_lb_pred - 1| = {resid_lb:.6e} > s_lb = {s_lb:.6e} "
        f"(margin={s_lb - resid_lb:+.6e})"
    )

    # Step 4.4: Cross-code ("sheet_lattice") check
    assert resid_cross <= s_cross, (
        f"Coarse cross-code |bq_P/bq_LB - bq_P_pred/bq_LB_pred| = {resid_cross:.6e} > "
        f"s_lb + s_part = {s_cross:.6e} (margin={s_cross - resid_cross:+.6e})"
    )

    # Step 4.2: Particle deterministic ("sheet_lattice") prediction check
    assert resid_lat <= s_part, (
        f"Coarse Particle (sheet_lattice) |bq_P/bq_P_pred - 1| = {resid_lat:.6e} > "
        f"s_part = {s_part:.6e} (margin={s_part - resid_lat:+.6e})"
    )

    # Step 4.6: Entrance amplitude at z_0 = 0.5 * dz
    amp_err_lb = float(abs(np.abs(i1_lb[0]) / params["I1_0"] - 1.0))
    amp_err_part = float(abs(np.abs(i1_part[0]) / params["I1_0"] - 1.0))
    assert amp_err_lb <= tols["tol_amp_lb"], (
        f"LB entrance |I1(z0)|/I1_0 - 1 = {amp_err_lb:.6e} > bound={tols['tol_amp_lb']:.6e} "
        f"(margin={tols['tol_amp_lb'] - amp_err_lb:+.6e})"
    )
    assert amp_err_part <= tols["tol_amp_part"], (
        f"Part entrance |I1(z0)|/I1_0 - 1 = {amp_err_part:.6e} > bound={tols['tol_amp_part']:.6e} "
        f"(margin={tols['tol_amp_part'] - amp_err_part:+.6e})"
    )


def test_mutation_double_particle_charge_to_mass_fails(monkeypatch):
    """M1: Doubling particle species |q/m| matches mutated prediction and fails baseline cross-code check."""
    params = compute_analytic_parameters(qm_scale=1.0)
    params_mut = compute_analytic_parameters(qm_scale=2.0)
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params)
    tols_mut = derived_tolerances(nx, nz, params_mut)

    lb_res, part_base = cached_baseline_pair(nx, nz)
    z_coords = lb_res["z_coords"]
    z_min_base, z_max_base = fit_window(params)
    z_min_mut, z_max_mut = fit_window(params_mut)

    fit_lb = fit_beta_q_from_envelope(
        z_coords, lb_res["i1_lb"], params["beta_q"], z_min=z_min_base, z_max=z_max_base
    )
    bq_lb = fit_lb["beta_q"]
    delta_meas_base = (params["u0"] - part_base["u_z_mean"]) / params["u0"]
    _, bq_part_base_pred = predict_particle_beta_q(delta_meas_base, tols, qm_scale=1.0)

    orig_init = Species.__init__

    def _mutated_species_init(self, name, charge, mass, *args, **kwargs):
        # Double |q / m| by halving the inertial mass while preserving charge -e
        orig_init(self, name, charge, 0.5 * mass, *args, **kwargs)

    monkeypatch.setattr(Species, "__init__", _mutated_species_init)
    part_mut = run_particle_beam_drift(
        nx,
        nz,
        beam_ref=lb_res["beam"],
        qm_scale=1.0,
        seed=42,
        profile="sheet_lattice",
    )
    fit_mut = fit_beta_q_from_envelope(
        z_coords,
        part_mut["i1_part"],
        params_mut["beta_q"],
        z_min=z_min_mut,
        z_max=z_max_mut,
    )
    bq_part_mut = fit_mut["beta_q"]
    delta_meas_mut = (params_mut["u0"] - part_mut["u_z_mean"]) / params_mut["u0"]
    _, bq_part_mut_pred = predict_particle_beta_q(
        delta_meas_mut, tols_mut, qm_scale=2.0
    )
    resid_m1 = abs(bq_part_mut / bq_part_mut_pred - 1.0)
    bound_m1 = tols_mut["s_part"]

    baseline_check_failed = False
    with pytest.raises(AssertionError, match="exceeds"):
        resid_cross_mut = abs(
            bq_part_mut / bq_lb - bq_part_base_pred / tols["bq_lb_pred"]
        )
        assert resid_cross_mut <= tols["s_cross"], (
            f"M1 cross-code difference {resid_cross_mut:.6e} exceeds {tols['s_cross']:.6e}"
        )
    baseline_check_failed = True

    print(
        f"\nMUTCHECK id=M1 meas={bq_part_mut:.6f} pred={bq_part_mut_pred:.6f} "
        f"resid={resid_m1:.6e} bound={bound_m1:.6e} "
        f"baseline_check_failed={baseline_check_failed}",
        flush=True,
    )

    assert resid_m1 <= bound_m1, (
        f"M1 |bq_part_mut/bq_part_mut_pred - 1| = {resid_m1:.6e} > s_part_mut = {bound_m1:.6e} "
        f"(margin={bound_m1 - resid_m1:+.6e})"
    )


def test_mutation_linearized_beam_rho0_off_by_2x_fails(monkeypatch):
    """M2: Doubling `LinearizedBeam.rho0` matches `omega_p,dep * sqrt(2)` prediction and fails cross-code check."""
    params = compute_analytic_parameters()
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params)
    tols_m2 = derived_tolerances(nx, nz, params, omega_p_scale=float(np.sqrt(2.0)))

    lb_res, part_res = cached_baseline_pair(nx, nz)
    z_coords = lb_res["z_coords"]
    z_min, z_max = fit_window(params)

    fit_part = fit_beta_q_from_envelope(
        z_coords, part_res["i1_part"], params["beta_q"], z_min=z_min, z_max=z_max
    )
    bq_part = fit_part["beta_q"]
    delta_meas_base = (params["u0"] - part_res["u_z_mean"]) / params["u0"]
    _, bq_part_base_pred = predict_particle_beta_q(delta_meas_base, tols)

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
    bq_lb_m2_pred = tols_m2["bq_lb_pred"]
    bound_m2 = tols_m2["s_lb"]
    resid_m2 = abs(bq_lb_mut / bq_lb_m2_pred - 1.0)

    baseline_check_failed = False
    with pytest.raises(AssertionError, match="exceeds"):
        resid_cross_mut = abs(
            bq_part / bq_lb_mut - bq_part_base_pred / tols["bq_lb_pred"]
        )
        assert resid_cross_mut <= tols["s_cross"], (
            f"M2 cross-code difference {resid_cross_mut:.6e} exceeds {tols['s_cross']:.6e}"
        )
    baseline_check_failed = True

    print(
        f"\nMUTCHECK id=M2 meas={bq_lb_mut:.6f} pred={bq_lb_m2_pred:.6f} "
        f"resid={resid_m2:.6e} bound={bound_m2:.6e} "
        f"baseline_check_failed={baseline_check_failed}",
        flush=True,
    )

    assert resid_m2 <= bound_m2, (
        f"M2 |bq_lb_mut/bq_lb_m2_pred - 1| = {resid_m2:.6e} > s_lb = {bound_m2:.6e} "
        f"(margin={bound_m2 - resid_m2:+.6e})"
    )


def test_mutation_disk_profile_instead_of_sheet_fails(monkeypatch):
    """M3: Fibonacci disk x-profile instead of planar sheet increases beta_q and fails Step 4.2."""
    params = compute_analytic_parameters()
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params)
    lb_res, _ = cached_baseline_pair(nx, nz)
    z_coords = lb_res["z_coords"]
    z_min, z_max = fit_window(params)

    monkeypatch.setattr(BeamInjector, "_sheet_lattice_x", _fibonacci_disk_lattice_x)
    part_m3 = _cached_m3_disk_run(nx, nz)
    fit_m3 = fit_beta_q_from_envelope(
        z_coords, part_m3["i1_part"], params["beta_q"], z_min=z_min, z_max=z_max
    )
    bq_part_m3 = fit_m3["beta_q"]
    delta_meas_m3 = (params["u0"] - part_m3["u_z_mean"]) / params["u0"]
    _, bq_part_m3_pred = predict_particle_beta_q(delta_meas_m3, tols)
    resid_m3 = abs(bq_part_m3 / bq_part_m3_pred - 1.0)
    bound_m3 = tols["s_part"]

    baseline_check_failed = False
    with pytest.raises(AssertionError, match="exceeds"):
        assert resid_m3 <= bound_m3, (
            f"M3 Step 4.2 residual {resid_m3:.6e} exceeds s_part={bound_m3:.6e}"
        )
    baseline_check_failed = True

    print(
        f"\nMUTCHECK id=M3 meas={bq_part_m3:.6f} pred={bq_part_m3_pred:.6f} "
        f"resid={resid_m3:.6e} bound={bound_m3:.6e} "
        f"baseline_check_failed={baseline_check_failed}",
        flush=True,
    )

    assert bq_part_m3 > bq_part_m3_pred, (
        f"M3 disk profile beta_q={bq_part_m3:.6f} rad/m not above sheet prediction "
        f"{bq_part_m3_pred:.6f} rad/m (diff={bq_part_m3 - bq_part_m3_pred:+.6f} rad/m)"
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
    _, bq_part_pred_c = predict_particle_beta_q(delta_lat_c, tols_c)
    _, bq_part_pred_f = predict_particle_beta_q(delta_lat_f, tols_f)

    resid_lb_f = abs(bq_lb_f / tols_f["bq_lb_pred"] - 1.0)
    resid_lat_c = abs(bq_part_lat_c / bq_part_pred_c - 1.0)
    resid_lat_f = abs(bq_part_lat_f / bq_part_pred_f - 1.0)
    resid_cross_f = abs(bq_part_lat_f / bq_lb_f - bq_part_pred_f / tols_f["bq_lb_pred"])

    tcrit_c = noise_c["tcrit"]
    tcrit_f = noise_f["tcrit"]
    resid_mean_c = abs(noise_c["bq_mean"] / bq_part_pred_c - 1.0)
    resid_mean_f = abs(noise_f["bq_mean"] / bq_part_pred_f - 1.0)
    bound_mean_c = tols_c["s_part"] + tcrit_c * noise_c["bq_sem"] / bq_part_pred_c
    bound_mean_f = tols_f["s_part"] + tcrit_f * noise_f["bq_sem"] / bq_part_pred_f
    mean_minus_lat_c = abs(noise_c["bq_mean"] - bq_part_lat_c)
    mean_minus_lat_f = abs(noise_f["bq_mean"] - bq_part_lat_f)

    seeds_c_str = "[" + ",".join(f"{float(v):.6f}" for v in noise_c["bq_seeds"]) + "]"
    seeds_f_str = "[" + ",".join(f"{float(v):.6f}" for v in noise_f["bq_seeds"]) + "]"

    print(flush=True)
    print(format_error_budget(nx_f, nz_f, tols_f), flush=True)
    print(
        f"LBCHECK grid=fine bq_th={beta_q_th:.6f} bq_lb_pred={tols_f['bq_lb_pred']:.6f} "
        f"bq_lb={bq_lb_f:.6f} resid={resid_lb_f:.6e} s_lb={tols_f['s_lb']:.6e} "
        f"eps_ax={tols_f['eps_ax']:.6e} eps_x_fit={tols_f['eps_x_fit']:.6e} "
        f"eps_fit={tols_f['eps_fit']:.6e}",
        flush=True,
    )
    print(
        f"H1CHECK grid=coarse bq_th={beta_q_th:.6f} bq_part_pred={bq_part_pred_c:.6f} "
        f"lattice={bq_part_lat_c:.6f} resid_lattice={resid_lat_c:.6e} "
        f"s_part={tols_c['s_part']:.6e} mean_sheet={noise_c['bq_mean']:.6f} "
        f"sem={noise_c['bq_sem']:.6e} tcrit={tcrit_c:.6f} "
        f"resid_mean={resid_mean_c:.6e} bound_mean={bound_mean_c:.6e} "
        f"mean_minus_lattice={mean_minus_lat_c:.6e} seeds={seeds_c_str}",
        flush=True,
    )
    print(
        f"H1CHECK grid=fine bq_th={beta_q_th:.6f} bq_part_pred={bq_part_pred_f:.6f} "
        f"lattice={bq_part_lat_f:.6f} resid_lattice={resid_lat_f:.6e} "
        f"s_part={tols_f['s_part']:.6e} mean_sheet={noise_f['bq_mean']:.6f} "
        f"sem={noise_f['bq_sem']:.6e} tcrit={tcrit_f:.6f} "
        f"resid_mean={resid_mean_f:.6e} bound_mean={bound_mean_f:.6e} "
        f"mean_minus_lattice={mean_minus_lat_f:.6e} seeds={seeds_f_str}",
        flush=True,
    )
    print(
        f"DEPCHECK grid=coarse delta_th={tols_c['delta_th']:.6e} "
        f"delta_th_disc={tols_c['delta_th_disc']:.6e} "
        f"delta_meas_lattice={delta_lat_c:.6e} "
        f"delta_meas_mean={noise_c['delta_mean']:.6e} "
        f"sem={noise_c['delta_sem']:.6e}",
        flush=True,
    )
    print(
        f"DEPCHECK grid=fine delta_th={tols_f['delta_th']:.6e} "
        f"delta_th_disc={tols_f['delta_th_disc']:.6e} "
        f"delta_meas_lattice={delta_lat_f:.6e} "
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

    # Fine-grid Step 3 amplitude sanity |A| > 0.5 * m * I0 and Step 4.8 DFT leakage
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

    # Step 4.3(a): |mean / bq_P,pred - 1| <= s_part + t * SEM / bq_P,pred
    assert resid_mean_c <= bound_mean_c, (
        f"Coarse |mean_sheet/bq_P_pred - 1| = {resid_mean_c:.6e} > bound={bound_mean_c:.6e} "
        f"(margin={bound_mean_c - resid_mean_c:+.6e})"
    )
    assert resid_mean_f <= bound_mean_f, (
        f"Fine |mean_sheet/bq_P_pred - 1| = {resid_mean_f:.6e} > bound={bound_mean_f:.6e} "
        f"(margin={bound_mean_f - resid_mean_f:+.6e})"
    )

    # Step 4.3(b): |mean - bq_P,lattice| <= t * SEM (no hidden stochastic bias)
    bound_lat_c = tcrit_c * noise_c["bq_sem"]
    bound_lat_f = tcrit_f * noise_f["bq_sem"]
    assert mean_minus_lat_f <= bound_lat_f, (
        f"Fine |mean_sheet - lattice| = {mean_minus_lat_f:.6e} > t*SEM = {bound_lat_f:.6e} "
        f"(margin={bound_lat_f - mean_minus_lat_f:+.6e})"
    )

    # Step 4.5: DC depression |delta_meas - delta_th_disc| <= t * SEM(delta_meas) + 1e-12
    for label, noise, tols in (
        ("fine", noise_f, tols_f),
        ("coarse", noise_c, tols_c),
    ):
        err_dep = abs(noise["delta_mean"] - tols["delta_th_disc"])
        bound_dep = noise["tcrit"] * noise["delta_sem"] + 1e-12
        assert err_dep <= bound_dep, (
            f"{label} DC depression |delta_meas_mean - delta_th_disc| = {err_dep:.6e} > "
            f"t*SEM + 1e-12 = {bound_dep:.6e} (margin={bound_dep - err_dep:+.6e})"
        )

    assert mean_minus_lat_c <= bound_lat_c, (
        f"Coarse |mean_sheet - lattice| = {mean_minus_lat_c:.6e} > t*SEM = {bound_lat_c:.6e} "
        f"(margin={bound_lat_c - mean_minus_lat_c:+.6e})"
    )

    # Fine-grid Step 4.1 (LB), Step 4.4 (cross-code), and Step 4.2 (Particle lattice)
    assert resid_lb_f <= tols_f["s_lb"], (
        f"Fine LB |bq_lb/bq_lb_pred - 1| = {resid_lb_f:.6e} > s_lb = {tols_f['s_lb']:.6e} "
        f"(margin={tols_f['s_lb'] - resid_lb_f:+.6e})"
    )
    assert resid_cross_f <= tols_f["s_cross"], (
        f"Fine cross-code |bq_P/bq_LB - bq_P_pred/bq_LB_pred| = {resid_cross_f:.6e} > "
        f"s_cross = {tols_f['s_cross']:.6e} (margin={tols_f['s_cross'] - resid_cross_f:+.6e})"
    )
    assert resid_lat_f <= tols_f["s_part"], (
        f"Fine Particle (sheet_lattice) |bq_P/bq_P_pred - 1| = {resid_lat_f:.6e} > "
        f"s_part = {tols_f['s_part']:.6e} (margin={tols_f['s_part'] - resid_lat_f:+.6e})"
    )

    # Fine-grid Step 4.6 entrance amplitude ("sheet_lattice") and 8-seed ("sheet") entrance amplitude
    amp_err_lb_f = float(abs(np.abs(lb_f["i1_lb"][0]) / params["I1_0"] - 1.0))
    amp_err_part_f = float(abs(np.abs(part_f["i1_part"][0]) / params["I1_0"] - 1.0))
    assert amp_err_lb_f <= tols_f["tol_amp_lb"], (
        f"Fine LB entrance |I1(z0)|/I1_0 - 1 = {amp_err_lb_f:.6e} > {tols_f['tol_amp_lb']:.6e} "
        f"(margin={tols_f['tol_amp_lb'] - amp_err_lb_f:+.6e})"
    )
    assert amp_err_part_f <= tols_f["tol_amp_part"], (
        f"Fine Part entrance |I1(z0)|/I1_0 - 1 = {amp_err_part_f:.6e} > {tols_f['tol_amp_part']:.6e} "
        f"(margin={tols_f['tol_amp_part'] - amp_err_part_f:+.6e})"
    )
    for label, noise, tols in (
        ("coarse", noise_c, tols_c),
        ("fine", noise_f, tols_f),
    ):
        amp_err_sheet = abs(noise["amp0_mean"] / params["I1_0"] - 1.0)
        amp_bound_sheet = (
            tols["tol_amp_part"] + noise["tcrit"] * noise["amp0_sem"] / params["I1_0"]
        )
        assert amp_err_sheet <= amp_bound_sheet, (
            f"{label} sheet entrance |I1(z0)|/I1_0 - 1 = {amp_err_sheet:.6e} > "
            f"bound={amp_bound_sheet:.6e} (margin={amp_bound_sheet - amp_err_sheet:+.6e})"
        )


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

    print(
        f"\nm-scan (sheet_lattice, m={MODULATION_INDEX} vs m/2={m_half}): "
        f"LB (linear identity): |bq(m)/bq(m/2) - 1|={bq_shift_lb:.3e} "
        f"(bound={tols['eps_dft']:.3e}), I2/I1={i2_ratio_lb_m:.6e} | "
        f"Part: bq(m)={bq_part_m:.6f}, bq(m/2)={bq_part_h:.6f}, "
        f"|bq(m)/bq(m/2) - 1|={bq_shift_part:.6e} (s_part={tols['s_part']:.6e}, "
        f"margin={tols['s_part'] - bq_shift_part:+.6e}) | "
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

    # Step 4.7: Particle m-scan nonlinear shift must stay below s_part
    assert bq_shift_part <= tols["s_part"], (
        f"Particle m-scan |bq(m)/bq(m/2) - 1| = {bq_shift_part:.6e} > "
        f"s_part = {tols['s_part']:.6e} (margin={tols['s_part'] - bq_shift_part:+.6e})"
    )


@pytest.mark.slow
def test_optional_diagnostics():
    """Step 8 optional diagnostics D3, D4, D5, D6 (report-only, coarse grid, `"sheet_lattice"`)."""
    params = compute_analytic_parameters()
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params)
    z_min, z_max = fit_window(params)
    beta_q_th = params["beta_q"]

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

    # D3: LinearizedBeam with semicircular rho0(x) prop sqrt(1 - x^2/b^2) for |x| < b
    semi_rho0 = _SemicircularRho0(
        params["rho_dep"], BEAM_HALF_THICKNESS_M, TUNNEL_HALF_WIDTH_M, nx
    )
    lb_d3 = run_linearized_beam_drift(nx, nz, rho0_profile=semi_rho0)
    fit_lb_d3 = fit_beta_q_from_envelope(
        z_coords, lb_d3["i1_lb"], beta_q_th, z_min=z_min, z_max=z_max
    )
    bq_lb_d3 = fit_lb_d3["beta_q"]
    shift_lb_d3 = bq_lb_d3 / bq_lb_base - 1.0

    part_m3 = _cached_m3_disk_run(nx, nz)
    fit_part_m3 = fit_beta_q_from_envelope(
        z_coords, part_m3["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
    )
    shift_part_m3 = fit_part_m3["beta_q"] / bq_part_base - 1.0
    diff_d3 = abs(shift_lb_d3 - shift_part_m3)

    # D4: ParticleSolver._backpush_injected_slice patched to a no-op
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

    # D5: cfln = 0.25
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

    # D6: n_macro_per_step = 16 and 64
    part_d6_64 = run_particle_beam_drift(
        nx,
        nz,
        beam_ref=lb_base["beam"],
        qm_scale=1.0,
        seed=42,
        profile="sheet_lattice",
        n_macro_per_step=64,
    )
    fit_d6_64 = fit_beta_q_from_envelope(
        z_coords, part_d6_64["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
    )
    rel_diff_d6 = abs(fit_d6_64["beta_q"] / bq_part_base - 1.0)

    print(flush=True)
    print(
        f"DIAG id=D3 bq_lb_semi={bq_lb_d3:.6f} shift_lb={shift_lb_d3:.6e} "
        f"shift_part_m3={shift_part_m3:.6e} diff={diff_d3:.6e} s_cross={tols['s_cross']:.6e}",
        flush=True,
    )
    print(
        f"DIAG id=D4 bq_no_backpush={fit_d4['beta_q']:.6f} "
        f"rel_diff={rel_diff_d4:.6e} s_part={tols['s_part']:.6e}",
        flush=True,
    )
    print(
        f"DIAG id=D5 cfln=0.25 bq_cfl025={fit_d5['beta_q']:.6f} "
        f"rel_diff={rel_diff_d5:.6e} omega_dt_sq={omega_dt_sq:.6e}",
        flush=True,
    )
    print(
        f"DIAG id=D6 bq_n16={bq_part_base:.6f} bq_n64={fit_d6_64['beta_q']:.6f} "
        f"rel_diff={rel_diff_d6:.6e} s_part={tols['s_part']:.6e}",
        flush=True,
    )
