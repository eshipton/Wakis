# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Milestone P5 (part 1) V7 cross-validation of `wakis.particles` against `LinearizedBeam`.

Propagates the same density-modulated planar sheet electron beam through a
smooth parallel-plate PEC drift tunnel twice on identical grids, time steps,
and boundary conditions:

1. As relativistic macroparticles (`ParticleSolver` + `BeamInjector` with
   modulated current ``I(t) = I0 * (1 + m * sin(omega * t))``, ``m << 1``)
   confined axially by a strong uniform guide magnetic field ``B_ext``.
2. As an Eulerian linearized cold-fluid beam (`LinearizedBeam`) evaluated at the
   DC space-charge-depressed equilibrium ``(u_dep, rho_dep, gamma_dep)`` (Option
   3(ii)) and driven through the public ``inflow_rho1`` boundary parameter.

Note on Eulerian velocity compensation: unlike the Pierce travelling-wave-tube
helper (which divides ``u0`` by ``sin(beta_e * dz) / (beta_e * dz)`` to maintain
synchronism with an external slow-wave circuit), both solvers here share the
physical equilibrium velocity ``u_dep`` and the Eulerian group-velocity
dispersion ``1 / cos(beta_e * dz) - 1`` and numerical carrier shift
``(d ln R / d ln tau) * gamma_dep**2 * (beta_e * dz)**2 / 6`` are accounted for
explicitly in the predicted `LinearizedBeam` wavenumber and error budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import epsilon_0

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._space_charge_waves_common import (
    BEAM_HALF_THICKNESS_M,
    CFL_NUMBER,
    COARSE_GRID,
    FINE_GRID,
    FIT_WINDOW_LAMBDA_Q,
    GUIDE_BZ_TESLA,
    MODULATION_INDEX,
    OUTFLOW_RAMP_M,
    PERIODIC_LY_M,
    TUNNEL_HALF_WIDTH_M,
    TUNNEL_LENGTH_M,
    cached_baseline_pair,
    compute_analytic_parameters,
    compute_spatial_spectrum,
    derived_tolerances,
    extract_coherent_bunching_i2_ratio,
    fit_beta_q_from_envelope,
    format_error_budget,
    measure_empirical_noise,
    run_linearized_beam_drift,
    run_particle_beam_drift,
)
from wakis import GridFIT3D, SolverFIT3D
from wakis.beam_fluid import LinearizedBeam
from wakis.estimate import estimate, preflight
from wakis.particles import BeamInjector, ParticleSolver, Species


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

    print()
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

        def _capture_injected(species_obj, _orig=orig_pre_push):
            _orig(species_obj)
            x_new = np.asarray(species_obj.x[-n_macro:], dtype=np.float64).copy()
            y_new = np.asarray(species_obj.y[-n_macro:], dtype=np.float64).copy()
            injected_slices.append((x_new, y_new))

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
                    f"Step {step_idx} (sheet_lattice): max|x - x_lat|={max_lat_err:.3e} > {1e-15 * b:.3e}"
                )


def test_preflight_and_resolution_criteria():
    """Verify `wakis.estimate` preflight, resolution rules, and 3-null fit-window margin."""
    params = compute_analytic_parameters()
    z_max_fit = FIT_WINDOW_LAMBDA_Q * params["lambda_q"]

    # Tunnel length must be at least 1.6 * lambda_q so the 3-null fit has redundancy
    assert TUNNEL_LENGTH_M >= 1.6 * params["lambda_q"]

    for nx, ny, nz in (COARSE_GRID, FINE_GRID):
        dx = 2.0 * TUNNEL_HALF_WIDTH_M / nx
        dz = TUNNEL_LENGTH_M / nz

        # >= 20 cells per electron wavelength, vacuum wavelength, and plasma wavelength
        assert params["lambda_e"] / dz >= 20.0
        assert params["lambda_0"] / dz >= 20.0
        assert params["lambda_p"] / dz >= 20.0

        # >= 4 cells across the full sheet-beam thickness 2 * b
        assert (2.0 * BEAM_HALF_THICKNESS_M) / dx >= 4.0

        # 3-null fit window [0, FIT_WINDOW_LAMBDA_Q * lambda_q] clears the outflow ramp
        # with >= 2.5 mm (> 5 * dz_c) margin
        ramp_start_z = TUNNEL_LENGTH_M - OUTFLOW_RAMP_M
        margin_m = ramp_start_z - z_max_fit
        assert margin_m >= 2.5e-3
        assert margin_m >= 5.0 * dz

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


def test_dc_depression_and_dft_leakage_and_linearity():
    """Verify DC depression on coarse and fine grids, DFT leakage, and m-scan 2*omega linearity."""
    params = compute_analytic_parameters()
    noise_c = measure_empirical_noise(COARSE_GRID, m_mod=MODULATION_INDEX)
    noise_f = measure_empirical_noise(FINE_GRID, m_mod=MODULATION_INDEX)
    nx_c, _, nz_c = COARSE_GRID
    nx_f, _, nz_f = FINE_GRID
    tols_c = derived_tolerances(
        nx_c, nz_c, params, noise_c["eps_noise"], noise_f["eps_noise"]
    )
    tols_f = derived_tolerances(
        nx_f, nz_f, params, noise_c["eps_noise"], noise_f["eps_noise"]
    )

    # Item 10: Check DC velocity and charge-density depression on BOTH coarse and fine grids
    delta_u_th = params["delta_u_dc"]
    for label, (nx, nz), tols in (
        ("coarse", (nx_c, nz_c), tols_c),
        ("fine", (nx_f, nz_f), tols_f),
    ):
        _, part_res = cached_baseline_pair(nx, nz)
        delta_u_meas = (params["u0"] - part_res["u_z_mean"]) / params["u0"]
        delta_rho_meas = (abs(part_res["rho_dc_mean"]) - abs(params["rho0"])) / abs(
            params["rho0"]
        )
        dep_tol = tols["dep_tol"]
        err_u_dep = abs(delta_u_meas - delta_u_th)
        err_rho_dep = abs(delta_rho_meas - delta_u_th)
        eps_dc_meas_shift = float(params["dlnbq_ddelta"] * (delta_u_meas - delta_u_th))
        print(
            f"DC depression ({label} {nx}x1x{nz}): th={delta_u_th:.6%}, "
            f"meas_u={delta_u_meas:.6%} (|diff|={err_u_dep:.6%}, "
            f"margin={dep_tol - err_u_dep:+.6%}), "
            f"meas_rho={delta_rho_meas:.6%} (|diff|={err_rho_dep:.6%}, "
            f"margin={dep_tol - err_rho_dep:+.6%}), dep_tol={dep_tol:.6%}, "
            f"induced_bq_shift={eps_dc_meas_shift:+.6%} "
            f"(budget eps_dc_res={tols['eps_dc_res']:.6%})"
        )
        assert err_u_dep <= dep_tol
        assert err_rho_dep <= dep_tol
        assert abs(eps_dc_meas_shift) <= tols["eps_dc_res"]

    lb_res, part_res = cached_baseline_pair(nx_c, nz_c)
    tols = tols_c

    # Residual DC leakage after integer-period dt + time-mean subtraction
    # (dc_test - mean(dc_test) is an algebraic zero identity; dirichlet_leak tests
    # the integer-n_spp orthogonality |sum(ker1)| * (I0 / I1_0))
    print(
        f"DFT leakage: lb={lb_res['dc_leakage_lb']:.3e}, "
        f"part={part_res['dc_leakage_part']:.3e}, bound={tols['eps_dft']:.3e}"
    )
    assert lb_res["dc_leakage_lb"] <= tols["eps_dft"]
    assert part_res["dc_leakage_part"] <= tols["eps_dft"]

    # Item 9: m-scan (m and m/2) for beta_q invariance and 2*omega ballistic bunching
    # with I2 noise measured from the 8-seed ensemble at both m and m/2.
    m_half = 0.5 * MODULATION_INDEX
    noise_h = measure_empirical_noise(COARSE_GRID, m_mod=m_half)
    lb_half, part_half = cached_baseline_pair(nx_c, nz_c, m_mod=m_half)

    z_coords = lb_res["z_coords"]
    beta_q_th = params["beta_q"]
    lam_q = params["lambda_q"]
    z_max_fit = FIT_WINDOW_LAMBDA_Q * lam_q
    bq_lb_m, _ = fit_beta_q_from_envelope(
        z_coords, lb_res["i1_lb"], beta_q_th, z_max=z_max_fit
    )
    bq_lb_h, _ = fit_beta_q_from_envelope(
        z_coords, lb_half["i1_lb"], beta_q_th, z_max=z_max_fit
    )
    bq_part_m, _ = fit_beta_q_from_envelope(
        z_coords, part_res["i1_part"], beta_q_th, z_max=z_max_fit
    )
    bq_part_h, _ = fit_beta_q_from_envelope(
        z_coords, part_half["i1_part"], beta_q_th, z_max=z_max_fit
    )

    # Note: LinearizedBeam m-scan invariance is an algebraic identity for a linear solver
    bq_diff_lb = abs(bq_lb_m - bq_lb_h) / beta_q_th
    bq_diff_part = abs(bq_part_m - bq_part_h) / beta_q_th
    tol_bq_mscan = float(np.hypot(noise_c["eps_noise"], noise_h["eps_noise"]))
    print(
        f"m-scan beta_q invariance: LB (linear identity) |bq(m)-bq(m/2)|/bq_th={bq_diff_lb:.3e} "
        f"(bound {tols['eps_dft']:.3e}) | Part bq(m)={bq_part_m:.4f}, "
        f"bq(m/2)={bq_part_h:.4f}, |diff|/bq_th={bq_diff_part:.6%} "
        f"(bound {tol_bq_mscan:.6%}, margin={tol_bq_mscan - bq_diff_part:+.6%})"
    )
    assert bq_diff_lb <= tols["eps_dft"]
    assert bq_diff_part <= tol_bq_mscan

    mask = z_coords <= lam_q
    i2_ratio_lb_m = float(
        np.max(np.abs(lb_res["i2_lb"][mask])) / np.abs(lb_res["i1_lb"][0])
    )
    i2_ratio_lb_h = float(
        np.max(np.abs(lb_half["i2_lb"][mask])) / np.abs(lb_half["i1_lb"][0])
    )
    i2_ratio_part_m = float(
        np.max(np.abs(part_res["i2_part"][mask])) / np.abs(part_res["i1_part"][0])
    )
    i2_ratio_part_h = float(
        np.max(np.abs(part_half["i2_part"][mask])) / np.abs(part_half["i1_part"][0])
    )

    # LinearizedBeam m-scan invariance is an algebraic identity for a linear solver,
    # and its 2*omega content is bounded by the C^1 cosine turn-on ramp transient tail tol_i2_lb
    assert i2_ratio_lb_m <= tols["tol_i2_lb"]
    assert abs(i2_ratio_lb_m - i2_ratio_lb_h) <= tols["eps_dft"]

    # ParticleSolver max |I2|/|I1(0)| at m and coherent ballistic-bunching amplitude at m and m/2
    # are bounded by the order-of-magnitude ballistic-bunching estimate tol_i2_part = pi * m,
    # and the coherent quadrature ballistic-bunching component scales linearly with m (ratio ~ 2.0)
    # within the 8-seed I2 ensemble noise measured at m and at m/2.
    i2_bunch_m = extract_coherent_bunching_i2_ratio(
        z_coords,
        part_res["i1_part"],
        part_res["i2_part"],
        params["beta_e"],
        beta_q_th,
        lam_q,
    )
    i2_bunch_h = extract_coherent_bunching_i2_ratio(
        z_coords,
        part_half["i1_part"],
        part_half["i2_part"],
        params["beta_e"],
        beta_q_th,
        lam_q,
    )
    bunch_scaling_ratio = i2_bunch_m / i2_bunch_h
    bunch_rel_err = abs(bunch_scaling_ratio - 2.0) / 2.0
    tol_bunch_scale = float(
        np.hypot(
            noise_c["eps_i2_noise"] / noise_c["i2_bunch_mean"],
            noise_h["eps_i2_noise"] / noise_h["i2_bunch_mean"],
        )
    )
    print(
        f"8-seed I2 coherent bunching noise: "
        f"m: mean={noise_c['i2_bunch_mean']:.6%}, 3*sigma={noise_c['eps_i2_noise']:.6%} | "
        f"m/2: mean={noise_h['i2_bunch_mean']:.6%}, 3*sigma={noise_h['eps_i2_noise']:.6%}"
    )
    print(
        f"Second-harmonic |I2|/|I1(0)|: LB={i2_ratio_lb_m:.6%} (m/2 identity: {i2_ratio_lb_h:.6%}, "
        f"transient bound={tols['tol_i2_lb']:.6%}, margin={tols['tol_i2_lb'] - i2_ratio_lb_m:+.6%}) | "
        f"Part raw max={i2_ratio_part_m:.6%} (m/2 raw={i2_ratio_part_h:.6%}, "
        f"order-of-magnitude pi*m estimate={tols['tol_i2_part']:.6%}, "
        f"margin={tols['tol_i2_part'] - i2_ratio_part_m:+.6%}) | "
        f"Part coherent bunching: m={i2_bunch_m:.6%} (pi*m estimate={tols['tol_i2_part']:.6%}, "
        f"margin={tols['tol_i2_part'] - i2_bunch_m:+.6%}), "
        f"m/2={i2_bunch_h:.6%} (0.5*pi*m estimate={0.5 * tols['tol_i2_part']:.6%}, "
        f"margin={0.5 * tols['tol_i2_part'] - i2_bunch_h:+.6%}), "
        f"ratio={bunch_scaling_ratio:.4f} (rel_err={bunch_rel_err:.4%}, "
        f"8-seed I2 noise tol={tol_bunch_scale:.4%}, margin={tol_bunch_scale - bunch_rel_err:+.4%})"
    )
    assert i2_ratio_part_m <= tols["tol_i2_part"]
    assert i2_bunch_m <= tols["tol_i2_part"]
    assert i2_bunch_h <= 0.5 * tols["tol_i2_part"]
    assert bunch_rel_err <= tol_bunch_scale


def test_space_charge_wave_envelope_entrance_and_spectrum():
    """Verify (a) 3-null beta_q & signed 8-seed mean error, (b) entrance |I1(0)| & z0, and (ii) spectrum."""
    params = compute_analytic_parameters()
    noise_c = measure_empirical_noise(COARSE_GRID)
    noise_f = measure_empirical_noise(FINE_GRID)
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(
        nx, nz, params, noise_c["eps_noise"], noise_f["eps_noise"]
    )
    print(format_error_budget(nx, nz, tols))
    print(
        f"Coarse 8-seed noise bq_seeds={np.round(noise_c['bq_seeds'], 4).tolist()}, "
        f"mean={noise_c['bq_mean']:.4f} rad/m, 3*sigma={3.0 * noise_c['bq_std']:.4f} rad/m "
        f"({noise_c['eps_noise']:.6%}), 3*sigma/sqrt(8)={noise_c['eps_noise_mean']:.6%}, "
        f"1*sigma/sqrt(8)={noise_c['sigma_mean']:.6%}"
    )

    # Every signed prediction (including signed eps_fit) and asserted bound equals
    # the exact sum of its constituent terms
    assert np.isclose(
        tols["eps_lead_lb_signed"],
        tols["eps_z_lb"]
        + tols["eps_tau_lb"]
        - tols["eps_z_fit"]
        + tols["eps_x_fit"]
        + tols["eps_fit"],
        rtol=0.0,
        atol=1e-15,
    )
    assert np.isclose(
        tols["eps_lead_part_signed"],
        -tols["eps_cic"] - tols["eps_z_fit"] + tols["eps_x_fit"] + tols["eps_fit"],
        rtol=0.0,
        atol=1e-15,
    )
    assert np.isclose(
        tols["tol_lb"],
        abs(tols["eps_lead_lb_signed"]) + tols["eps_quad_x"] + tols["eps_dft"],
        rtol=0.0,
        atol=1e-15,
    )
    assert np.isclose(
        tols["tol_part"],
        abs(tols["eps_lead_part_signed"])
        + tols["eps_dc_res"]
        + tols["eps_dft"]
        + tols["eps_noise"],
        rtol=0.0,
        atol=1e-15,
    )
    assert np.isclose(
        tols["tol_cross"],
        tols["tol_lb"] + tols["tol_part"],
        rtol=0.0,
        atol=1e-15,
    )

    lb_res, part_res = cached_baseline_pair(nx, nz)
    z_coords = lb_res["z_coords"]
    i1_lb = lb_res["i1_lb"]
    i1_part = part_res["i1_part"]

    beta_q_th = params["beta_q"]
    z_max_fit = FIT_WINDOW_LAMBDA_Q * params["lambda_q"]
    bq_lb, z0_lb, nulls_lb, res_lb = fit_beta_q_from_envelope(
        z_coords, i1_lb, beta_q_th, z_max=z_max_fit, return_residuals=True
    )
    bq_part, z0_part, nulls_part, res_part = fit_beta_q_from_envelope(
        z_coords, i1_part, beta_q_th, z_max=z_max_fit, return_residuals=True
    )

    # Item 8: At least 3 standing-wave nulls detected in [0, FIT_WINDOW_LAMBDA_Q * lambda_q],
    # and null least-squares residuals bounded by derived tol_null_res_lb and tol_null_res_part
    assert len(nulls_lb) >= 3
    assert len(nulls_part) >= 3
    max_null_res_lb = float(np.max(np.abs(res_lb)))
    max_null_res_part = float(np.max(np.abs(res_part)))

    signed_err_lb = (bq_lb - beta_q_th) / beta_q_th
    signed_err_part = (bq_part - beta_q_th) / beta_q_th
    signed_err_part_mean = (noise_c["bq_mean"] - beta_q_th) / beta_q_th
    rel_err_lb = abs(signed_err_lb)
    rel_err_part = abs(signed_err_part)
    rel_diff_cross = abs(bq_part - bq_lb) / bq_lb

    res_signed_lb = abs(signed_err_lb - tols["eps_lead_lb_signed"])
    signed_mean_residual_c = signed_err_part_mean - tols["eps_lead_part_signed"]
    res_signed_part_mean = abs(signed_mean_residual_c)
    z0_diff = abs(z0_part - z0_lb)

    print(
        f"Coarse 3-null fit: bq_th={beta_q_th:.4f} rad/m | "
        f"LB: bq={bq_lb:.4f} rad/m ({signed_err_lb:+.4%}, margin vs tol_lb={tols['tol_lb'] - rel_err_lb:+.4%}, "
        f"signed_res={res_signed_lb:.4%}, slack={tols['slack_lb']:.4%}, "
        f"signed_margin={tols['slack_lb'] - res_signed_lb:+.4%}), "
        f"z0={z0_lb * 1e3:.3f} mm, nulls={np.round(nulls_lb * 1e3, 3).tolist()} mm, "
        f"res={np.round(res_lb * 1e3, 4).tolist()} mm (max={max_null_res_lb * 1e3:.4f} mm, "
        f"bound={tols['tol_null_res_lb'] * 1e3:.4f} mm) | "
        f"Part (seed 42): bq={bq_part:.4f} rad/m ({signed_err_part:+.4%}, "
        f"margin vs tol_part={tols['tol_part'] - rel_err_part:+.4%}), "
        f"z0={z0_part * 1e3:.3f} mm, nulls={np.round(nulls_part * 1e3, 3).tolist()} mm, "
        f"res={np.round(res_part * 1e3, 4).tolist()} mm (max={max_null_res_part * 1e3:.4f} mm, "
        f"bound={tols['tol_null_res_part'] * 1e3:.4f} mm) | "
        f"Part (8-seed mean): bq_mean={noise_c['bq_mean']:.4f} rad/m ({signed_err_part_mean:+.4%}, "
        f"pred={tols['eps_lead_part_signed']:+.4%}, "
        f"signed_mean_residual={signed_mean_residual_c:+.4%}, "
        f"slack_part_mean={tols['slack_part_mean']:.4%}, "
        f"1*sigma/sqrt(8)={tols['sigma_mean']:.4%}) | "
        f"Cross: {rel_diff_cross:.4%} (margin={tols['tol_cross'] - rel_diff_cross:+.4%}) | "
        f"|z0_part - z0_lb|={z0_diff * 1e3:.4f} mm (heuristic bound={tols['tol_z0'] * 1e3:.4f} mm, "
        f"margin={(tols['tol_z0'] - z0_diff) * 1e3:+.4f} mm)"
    )

    assert max_null_res_lb <= tols["tol_null_res_lb"]
    assert max_null_res_part <= tols["tol_null_res_part"]

    # (a) Agreement with depressed analytic beta_q and between solvers
    assert rel_err_lb <= tols["tol_lb"], (
        f"LinearizedBeam beta_q={bq_lb:.4f} rad/m exceeds discretization bound "
        f"{tols['tol_lb']:.4%} vs analytic {beta_q_th:.4f} rad/m"
    )
    assert rel_err_part <= tols["tol_part"], (
        f"ParticleSolver beta_q={bq_part:.4f} rad/m exceeds bound "
        f"{tols['tol_part']:.4%} vs analytic {beta_q_th:.4f} rad/m"
    )
    assert rel_diff_cross <= tols["tol_cross"], (
        f"Cross-solver |beta_q_part - beta_q_lb|/beta_q_lb = {rel_diff_cross:.4%} "
        f"exceeds derived bound {tols['tol_cross']:.4%}"
    )

    # Item 4: Signed check on LinearizedBeam and on ParticleSolver 8-seed MEAN with sigma/sqrt(8)
    assert signed_err_lb > 0.0
    assert res_signed_lb <= tols["slack_lb"]
    assert signed_err_part_mean < 0.0
    assert res_signed_part_mean <= tols["slack_part_mean"]

    # Item 8: Cross-code launch offset z0 agreement within heuristic bound tol_z0
    assert z0_diff <= tols["tol_z0"]

    # (b) Entrance current modulation amplitude |I1(0)| matches m * I0 in both codes
    amp_err_lb = float(abs(np.abs(i1_lb[0]) / params["I1_0"] - 1.0))
    amp_err_part = float(abs(np.abs(i1_part[0]) / params["I1_0"] - 1.0))
    print(
        f"Entrance |I1(0)|/I1_0 - 1: lb={amp_err_lb:.6e} (bound {tols['tol_amp_lb']:.6e}, "
        f"margin={tols['tol_amp_lb'] - amp_err_lb:+.6e}), "
        f"part={amp_err_part:.6e} (bound {tols['tol_amp_part']:.6e}, "
        f"margin={tols['tol_amp_part'] - amp_err_part:+.6e})"
    )
    assert amp_err_lb <= tols["tol_amp_lb"]
    assert amp_err_part <= tols["tol_amp_part"]

    # Observable (ii): spatial spectrum exhibits fast and slow peaks at beta_e +- beta_q
    beta_e = params["beta_e"]
    dz = TUNNEL_LENGTH_M / nz
    k_fast_th = beta_e - beta_q_th
    k_slow_th = beta_e + beta_q_th
    delta_k_disp = float(np.arcsin(k_slow_th * dz) / dz - k_slow_th)
    tol_spec_k = float(
        beta_q_th / (2.0 * np.pi) + delta_k_disp + tols["tol_lb"] * beta_q_th
    )

    for label, phasor in (("LinearizedBeam", i1_lb), ("ParticleSolver", i1_part)):
        kz_grid, spec, k_fast, k_slow = compute_spatial_spectrum(
            z_coords, phasor, beta_e, beta_q_th, z_max=params["lambda_q"]
        )
        idx_fast = int(np.argmin(np.abs(kz_grid - k_fast)))
        idx_slow = int(np.argmin(np.abs(kz_grid - k_slow)))
        valley_min = float(np.min(spec[idx_fast : idx_slow + 1]))
        assert valley_min < 0.25 * min(spec[idx_fast], spec[idx_slow]), (
            f"{label} spectrum lacks a clear null between fast and slow peaks"
        )
        assert abs(k_fast - k_fast_th) <= tol_spec_k, (
            f"{label} fast-wave peak {k_fast:.2f} rad/m far from {k_fast_th:.2f} rad/m"
        )
        assert abs(k_slow - k_slow_th) <= tol_spec_k, (
            f"{label} slow-wave peak {k_slow:.2f} rad/m far from {k_slow_th:.2f} rad/m"
        )


def test_two_resolution_convergence():
    """Verify fine-grid 8-seed noise, 3-null residuals, signed mean residual finding, and order bands."""
    params = compute_analytic_parameters()
    noise_c = measure_empirical_noise(COARSE_GRID)
    noise_f = measure_empirical_noise(FINE_GRID)
    beta_q_th = params["beta_q"]
    z_max_fit = FIT_WINDOW_LAMBDA_Q * params["lambda_q"]

    nx_c, _, nz_c = COARSE_GRID
    nx_f, _, nz_f = FINE_GRID
    tols_c = derived_tolerances(
        nx_c, nz_c, params, noise_c["eps_noise"], noise_f["eps_noise"]
    )
    tols_f = derived_tolerances(
        nx_f, nz_f, params, noise_c["eps_noise"], noise_f["eps_noise"]
    )
    print(format_error_budget(nx_f, nz_f, tols_f))
    print(
        f"Fine 8-seed noise bq_seeds={np.round(noise_f['bq_seeds'], 4).tolist()}, "
        f"mean={noise_f['bq_mean']:.4f} rad/m, 3*sigma={3.0 * noise_f['bq_std']:.4f} rad/m "
        f"({noise_f['eps_noise']:.6%}), 3*sigma/sqrt(8)={noise_f['eps_noise_mean']:.6%}, "
        f"1*sigma/sqrt(8)={noise_f['sigma_mean']:.6%}"
    )

    lb_c, part_c = cached_baseline_pair(nx_c, nz_c)
    lb_f, part_f = cached_baseline_pair(nx_f, nz_f)

    bq_lb_c, _ = fit_beta_q_from_envelope(
        lb_c["z_coords"], lb_c["i1_lb"], beta_q_th, z_max=z_max_fit
    )
    bq_lb_f, z0_lb_f, nulls_lb_f, res_lb_f = fit_beta_q_from_envelope(
        lb_f["z_coords"],
        lb_f["i1_lb"],
        beta_q_th,
        z_max=z_max_fit,
        return_residuals=True,
    )
    bq_part_f, z0_part_f, nulls_part_f, res_part_f = fit_beta_q_from_envelope(
        lb_f["z_coords"],
        part_f["i1_part"],
        beta_q_th,
        z_max=z_max_fit,
        return_residuals=True,
    )

    assert len(nulls_lb_f) >= 3
    assert len(nulls_part_f) >= 3
    max_null_res_lb_f = float(np.max(np.abs(res_lb_f)))
    max_null_res_part_f = float(np.max(np.abs(res_part_f)))
    assert max_null_res_lb_f <= tols_f["tol_null_res_lb"]
    assert max_null_res_part_f <= tols_f["tol_null_res_part"]

    signed_lb_c = (bq_lb_c - beta_q_th) / beta_q_th
    signed_lb_f = (bq_lb_f - beta_q_th) / beta_q_th
    signed_part_mean_c = (noise_c["bq_mean"] - beta_q_th) / beta_q_th
    signed_part_mean_f = (noise_f["bq_mean"] - beta_q_th) / beta_q_th
    signed_part_seed0_f = (bq_part_f - beta_q_th) / beta_q_th

    err_lb_c = abs(signed_lb_c)
    err_lb_f = abs(signed_lb_f)
    err_part_mean_c = abs(signed_part_mean_c)
    err_part_mean_f = abs(signed_part_mean_f)
    err_part_seed0_f = abs(signed_part_seed0_f)
    diff_cross_c = abs(noise_c["bq_mean"] - bq_lb_c) / bq_lb_c
    diff_cross_f = abs(noise_f["bq_mean"] - bq_lb_f) / bq_lb_f

    res_signed_lb_f = abs(signed_lb_f - tols_f["eps_lead_lb_signed"])
    signed_mean_residual_c = signed_part_mean_c - tols_c["eps_lead_part_signed"]
    signed_mean_residual_f = signed_part_mean_f - tols_f["eps_lead_part_signed"]
    res_signed_part_mean_f = abs(signed_mean_residual_f)
    z0_diff_f = abs(z0_part_f - z0_lb_f)

    # Item 4: Report the signed residual of the 8-seed mean against the signed prediction
    # at both grids; if a non-converging positive bias persists on the fine grid, report
    # it explicitly as a FINDING with its size and sign without absorbing it into the budget.
    print(
        f"FINDING (8-seed mean signed residuals vs signed predictions): "
        f"coarse mean={signed_part_mean_c:+.4%} vs pred={tols_c['eps_lead_part_signed']:+.4%} "
        f"-> residual={signed_mean_residual_c:+.4%} "
        f"(slack_part_mean={tols_c['slack_part_mean']:.4%}, 1*sigma/sqrt(8)={tols_c['sigma_mean']:.4%}) | "
        f"fine mean={signed_part_mean_f:+.4%} vs pred={tols_f['eps_lead_part_signed']:+.4%} "
        f"-> residual={signed_mean_residual_f:+.4%} "
        f"(slack_part_mean={tols_f['slack_part_mean']:.4%}, 1*sigma/sqrt(8)={tols_f['sigma_mean']:.4%})"
    )

    assert err_lb_f <= tols_f["tol_lb"]
    assert err_part_seed0_f <= tols_f["tol_part"]
    assert diff_cross_f <= tols_f["tol_cross"]
    assert signed_lb_f > 0.0
    assert res_signed_lb_f <= tols_f["slack_lb"]
    assert z0_diff_f <= tols_f["tol_z0"]

    # Observed 2-grid convergence orders p = log2(err_coarse / err_fine) using 8-seed means
    p_lb = float(np.log2(err_lb_c / err_lb_f))
    p_part_mean = float(np.log2(err_part_mean_c / err_part_mean_f))
    p_cross = float(np.log2(diff_cross_c / diff_cross_f))
    print(
        f"Fine 3-null fit: LB bq={bq_lb_f:.4f} rad/m ({signed_lb_f:+.4%}, "
        f"margin vs tol_lb={tols_f['tol_lb'] - err_lb_f:+.4%}, "
        f"signed_res={res_signed_lb_f:.4%}, slack={tols_f['slack_lb']:.4%}, "
        f"signed_margin={tols_f['slack_lb'] - res_signed_lb_f:+.4%}), "
        f"z0={z0_lb_f * 1e3:.3f} mm, nulls={np.round(nulls_lb_f * 1e3, 3).tolist()} mm, "
        f"res={np.round(res_lb_f * 1e3, 4).tolist()} mm (max={max_null_res_lb_f * 1e3:.4f} mm, "
        f"bound={tols_f['tol_null_res_lb'] * 1e3:.4f} mm) | "
        f"Part (seed 42): bq={bq_part_f:.4f} rad/m ({signed_part_seed0_f:+.4%}, "
        f"margin vs tol_part={tols_f['tol_part'] - err_part_seed0_f:+.4%}), "
        f"z0={z0_part_f * 1e3:.3f} mm, nulls={np.round(nulls_part_f * 1e3, 3).tolist()} mm, "
        f"res={np.round(res_part_f * 1e3, 4).tolist()} mm (max={max_null_res_part_f * 1e3:.4f} mm, "
        f"bound={tols_f['tol_null_res_part'] * 1e3:.4f} mm) | "
        f"Part (8-seed mean): bq_mean={noise_f['bq_mean']:.4f} rad/m ({signed_part_mean_f:+.4%}) | "
        f"Cross={diff_cross_f:.4%} (margin={tols_f['tol_cross'] - diff_cross_f:+.4%}) | "
        f"|z0_part - z0_lb|={z0_diff_f * 1e3:.4f} mm (heuristic bound={tols_f['tol_z0'] * 1e3:.4f} mm, "
        f"margin={(tols_f['tol_z0'] - z0_diff_f) * 1e3:+.4f} mm)"
    )
    print(
        f"Observed orders: "
        f"p_lb={p_lb:.3f} in [{tols_f['order_lb_min']:.3f}, {tols_f['order_lb_max']:.3f}] "
        f"(margins +{p_lb - tols_f['order_lb_min']:.3f}, +{tols_f['order_lb_max'] - p_lb:.3f}) | "
        f"p_part_mean={p_part_mean:.3f}, noise-inclusive band="
        f"[{tols_f['order_part_min']:.3f}, {tols_f['order_part_max']:.3f}] "
        f"(det-only band=[{tols_f['order_part_det_min']:.3f}, {tols_f['order_part_det_max']:.3f}]) | "
        f"p_cross={p_cross:.3f}, uninformative cross band="
        f"[{tols_f['order_cross_min']:.3f}, {tols_f['order_cross_max']:.3f}]"
    )

    assert tols_f["order_lb_min"] <= p_lb <= tols_f["order_lb_max"], (
        f"LinearizedBeam observed order p={p_lb:.3f} outside "
        f"[{tols_f['order_lb_min']:.3f}, {tols_f['order_lb_max']:.3f}]"
    )
    # Item 5: Once 8-seed mean noise is included, the particle order band
    # [order_part_min, order_part_max] contains first order (p = 1.0), so the
    # particle-order assert is dropped as required and the cross band is labeled
    # uninformative.
    if tols_f["order_part_min"] > 1.0:
        assert tols_f["order_part_min"] <= p_part_mean <= tols_f["order_part_max"]


def test_mutation_double_particle_charge_to_mass_fails(monkeypatch):
    """Doubling particle species |q/m| matches mutated prediction, z0 cross-code, and fails (a).

    Also tests the transverse-shear beating hypothesis (Item 7) by running the
    mutated particle case with an analytic external electric field ``E_ext`` that
    cancels the DC transverse self-force ``E_x,DC(x) + u0 * B_y,DC(x)`` at gather,
    and reports the resulting null residuals with and without ``E_ext``.
    """
    params = compute_analytic_parameters(qm_scale=1.0)
    params_mut = compute_analytic_parameters(qm_scale=2.0)
    noise_c = measure_empirical_noise(COARSE_GRID)
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params, noise_c["eps_noise"])
    tols_mut = derived_tolerances(nx, nz, params_mut, noise_c["eps_noise"])

    lb_res, _ = cached_baseline_pair(nx, nz)
    z_coords = lb_res["z_coords"]
    z_max_base = FIT_WINDOW_LAMBDA_Q * params["lambda_q"]
    z_max_mut = FIT_WINDOW_LAMBDA_Q * params_mut["lambda_q"]
    bq_lb, _ = fit_beta_q_from_envelope(
        z_coords, lb_res["i1_lb"], params["beta_q"], z_max=z_max_base
    )

    # Mutated LinearizedBeam reference at 2x rho0 for cross-code z0 comparison
    lb_mut = run_linearized_beam_drift(nx, nz, rho0_scale=2.0)
    _, z0_lb_mut, nulls_lb_mut, res_lb_mut = fit_beta_q_from_envelope(
        z_coords,
        lb_mut["i1_lb"],
        params_mut["beta_q"],
        z_max=z_max_mut,
        return_residuals=True,
    )

    orig_init = Species.__init__

    def _mutated_species_init(self, name, charge, mass, *args, **kwargs):
        # Double |q / m| by halving the inertial mass while preserving charge -e
        orig_init(self, name, charge, 0.5 * mass, *args, **kwargs)

    monkeypatch.setattr(Species, "__init__", _mutated_species_init)
    part_mut = run_particle_beam_drift(
        nx, nz, beam_ref=lb_res["beam"], qm_scale=1.0, seed=42
    )
    bq_part_mut, z0_part_mut, nulls_part_mut, res_part_mut = fit_beta_q_from_envelope(
        z_coords,
        part_mut["i1_part"],
        params_mut["beta_q"],
        z_max=z_max_mut,
        return_residuals=True,
    )

    # Item 7: Test the transverse-shear hypothesis by applying an analytic E_ext(x, y, z, t)
    # that cancels the 1D planar DC transverse self-force inside the sheet beam [-b, b]:
    # E_x,ext(x) = -(rho_dep / (gamma_dep^2 * eps_0)) * clip(x, -b, b)
    b = BEAM_HALF_THICKNESS_M
    e_cancel_slope = -params_mut["rho_dep"] / (params_mut["gamma_dep"] ** 2 * epsilon_0)

    def _cancel_dc_transverse_self_force(x, y, z, t):
        ex = e_cancel_slope * np.clip(x, -b, b)
        zeros = np.zeros_like(ex)
        return ex, zeros, zeros

    part_mut_cancel = run_particle_beam_drift(
        nx,
        nz,
        beam_ref=lb_res["beam"],
        qm_scale=1.0,
        seed=42,
        e_ext=_cancel_dc_transverse_self_force,
    )
    bq_mut_cancel, z0_mut_cancel, nulls_mut_cancel, res_mut_cancel = (
        fit_beta_q_from_envelope(
            z_coords,
            part_mut_cancel["i1_part"],
            params_mut["beta_q"],
            z_max=z_max_mut,
            return_residuals=True,
        )
    )
    # Also evaluate 3-null residuals over [0, lambda_q_base] with and without E_ext
    _, _, _, res_mut_base_win = fit_beta_q_from_envelope(
        z_coords,
        part_mut["i1_part"],
        params_mut["beta_q"],
        z_max=params["lambda_q"],
        return_residuals=True,
    )
    _, _, _, res_cancel_base_win = fit_beta_q_from_envelope(
        z_coords,
        part_mut_cancel["i1_part"],
        params_mut["beta_q"],
        z_max=params["lambda_q"],
        return_residuals=True,
    )

    # Item 7: No eps_ols_mut; assert mutated beta_q directly against tols_mut["tol_part"]
    # and assert mutated z0 cross-code against tols_mut["tol_z0"].
    bq_part_mut_pred = params_mut["beta_q"] * (1.0 + tols_mut["eps_lead_part_signed"])
    mut_pred_err = abs(bq_part_mut - bq_part_mut_pred) / bq_part_mut_pred
    z0_mut_diff = abs(z0_part_mut - z0_lb_mut)
    print(
        f"Part mutation (no eps_ols_mut): meas={bq_part_mut:.4f} rad/m, "
        f"pred={bq_part_mut_pred:.4f} rad/m, err_vs_pred={mut_pred_err:.4%}, "
        f"tol_part_mut={tols_mut['tol_part']:.4%} "
        f"(margin={tols_mut['tol_part'] - mut_pred_err:+.4%}), "
        f"z0_part_mut={z0_part_mut * 1e3:.3f} mm, z0_lb_mut={z0_lb_mut * 1e3:.3f} mm, "
        f"|z0_part_mut - z0_lb_mut|={z0_mut_diff * 1e3:.4f} mm "
        f"(bound={tols_mut['tol_z0'] * 1e3:.4f} mm, "
        f"margin={(tols_mut['tol_z0'] - z0_mut_diff) * 1e3:+.4f} mm), "
        f"nulls={np.round(nulls_part_mut * 1e3, 3).tolist()} mm, "
        f"res={np.round(res_part_mut * 1e3, 4).tolist()} mm"
    )
    print(
        f"Transverse-shear E_ext cancellation test: "
        f"without E_ext: bq={bq_part_mut:.4f} rad/m, z0={z0_part_mut * 1e3:.3f} mm, "
        f"res(1.38*lam_q_mut)={np.round(res_part_mut * 1e3, 4).tolist()} mm, "
        f"res(lam_q_base)={np.round(res_mut_base_win * 1e3, 4).tolist()} mm | "
        f"with E_ext: bq={bq_mut_cancel:.4f} rad/m, z0={z0_mut_cancel * 1e3:.3f} mm, "
        f"nulls={np.round(nulls_mut_cancel * 1e3, 3).tolist()} mm, "
        f"res(1.38*lam_q_mut)={np.round(res_mut_cancel * 1e3, 4).tolist()} mm, "
        f"res(lam_q_base)={np.round(res_cancel_base_win * 1e3, 4).tolist()} mm"
    )
    assert len(nulls_part_mut) >= 3
    assert len(nulls_lb_mut) >= 3
    assert mut_pred_err <= tols_mut["tol_part"]
    assert z0_mut_diff <= tols_mut["tol_z0"]

    with pytest.raises(AssertionError, match="exceeds"):
        rel_diff_cross = abs(bq_part_mut - bq_lb) / bq_lb
        assert rel_diff_cross <= tols["tol_cross"], (
            f"Mutated cross-solver difference {rel_diff_cross:.4%} exceeds {tols['tol_cross']:.4%}"
        )


def test_mutation_linearized_beam_rho0_off_by_2x_fails(monkeypatch):
    """Doubling `LinearizedBeam.rho0` matches dispersed sqrt(2) prediction and fails (a)."""
    params = compute_analytic_parameters()
    params_mut = compute_analytic_parameters(qm_scale=2.0)
    noise_c = measure_empirical_noise(COARSE_GRID)
    nx, _, nz = COARSE_GRID
    tols = derived_tolerances(nx, nz, params, noise_c["eps_noise"])

    lb_res, part_res = cached_baseline_pair(nx, nz)
    z_coords = lb_res["z_coords"]
    z_max_base = FIT_WINDOW_LAMBDA_Q * params["lambda_q"]
    z_max_mut = FIT_WINDOW_LAMBDA_Q * params_mut["lambda_q"]
    bq_part, _ = fit_beta_q_from_envelope(
        z_coords, part_res["i1_part"], params["beta_q"], z_max=z_max_base
    )

    orig_lb_init = LinearizedBeam.__init__

    def _mutated_lb_init(self, *args, **kwargs):
        if "rho0" in kwargs and kwargs["rho0"] is not None:
            kwargs = dict(kwargs, rho0=2.0 * kwargs["rho0"])
        orig_lb_init(self, *args, **kwargs)

    monkeypatch.setattr(LinearizedBeam, "__init__", _mutated_lb_init)
    lb_mut = run_linearized_beam_drift(nx, nz, rho0_scale=1.0)
    bq_lb_mut, z0_lb_mut, nulls_lb_mut, res_lb_mut = fit_beta_q_from_envelope(
        z_coords,
        lb_mut["i1_lb"],
        params_mut["beta_q"],
        z_max=z_max_mut,
        return_residuals=True,
    )

    # Predicted mutated LB beta_q including sqrt(2) density scaling and Eulerian + FIT dispersion
    bq_lb_mut_pred = (
        np.sqrt(2.0) * params["beta_q"] * (1.0 + tols["eps_lead_lb_signed"])
    )
    mut_pred_err = abs(bq_lb_mut - bq_lb_mut_pred) / bq_lb_mut_pred
    print(
        f"LB mutation: meas={bq_lb_mut:.4f} rad/m, pred={bq_lb_mut_pred:.4f} rad/m, "
        f"err_vs_pred={mut_pred_err:.4%}, tol_lb={tols['tol_lb']:.4%} "
        f"(margin={tols['tol_lb'] - mut_pred_err:+.4%}), "
        f"z0={z0_lb_mut * 1e3:.3f} mm, nulls={np.round(nulls_lb_mut * 1e3, 3).tolist()} mm, "
        f"res={np.round(res_lb_mut * 1e3, 4).tolist()} mm"
    )
    assert len(nulls_lb_mut) >= 3
    assert mut_pred_err <= tols["tol_lb"]

    with pytest.raises(AssertionError, match="exceeds"):
        rel_diff_cross = abs(bq_part - bq_lb_mut) / bq_lb_mut
        assert rel_diff_cross <= tols["tol_cross"], (
            f"Mutated cross-solver difference {rel_diff_cross:.4%} exceeds {tols['tol_cross']:.4%}"
        )
