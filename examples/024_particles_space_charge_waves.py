# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Example 024 — Space-Charge Waves: Cross-Validation of Particle-in-Cell Against LinearizedBeam
=============================================================================================

Propagates the same density-modulated planar sheet electron beam through a
smooth parallel-plate PEC drift tunnel twice on identical grids, time steps,
and boundary conditions:

1. **Relativistic macroparticles (`wakis.particles`)**:
   `ParticleSolver` coupled to `SolverFIT3D` with a cathode `BeamInjector` at
   ``z = 0`` emitting a modulated current ``I(t) = I0 * (1 + m * sin(omega * t))``
   (``m << 1``) in a strong axial confining magnetic field ``B_ext``.
2. **Eulerian linearized cold-fluid beam (`wakis.beam_fluid`)**:
   `LinearizedBeam` evaluated at the DC space-charge-depressed equilibrium
   ``(u_dep, rho_dep, gamma_dep)`` (Option 3(ii)) and driven through the public
   ``inflow_rho1`` boundary parameter.

Both solvers evaluate the same observables on the exact same primal ``E_z``
edges (``beam.flat_idx``) and dual cross-sectional areas (``beam.dA_perp``):

- **Observable (i)**: Single-frequency temporal discrete Fourier transform (DFT)
  of the cross-section-integrated axial beam current ``I1(z)`` at the drive
  frequency ``f`` (after integer-period time-step selection and per-edge mean
  subtraction), compared against the analytical standing-wave envelope
  ``|I1(z)| = m * I0 * |cos(beta_q * (z - z0))|`` with reduced plasma wavenumber
  ``beta_q = R * omega_p / (gamma**(3/2) * u)``.
- **Observable (ii)**: Spatial wavenumber spectrum ``|I1(k_z)|``, resolving the
  fast (``beta_e - beta_q``) and slow (``beta_e + beta_q``) space-charge wave
  doublet straddling the electronic wavenumber ``beta_e = omega / u_dep``.

Saves two publication-style figures in ``docs/img/`` (via `wakis.figstyle`) and
writes run-time metrics to ``examples/024_particles_space_charge_waves.json``:
  1. ``docs/img/particles_space_charge_waves_envelope.png``
  2. ``docs/img/particles_space_charge_waves_spectrum.png``

Usage::

    python examples/024_particles_space_charge_waves.py --smoke
    python examples/024_particles_space_charge_waves.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._space_charge_waves_common import (
    BEAM_HALF_THICKNESS_M,
    COARSE_GRID,
    FINE_GRID,
    OUTFLOW_RAMP_M,
    TUNNEL_HALF_WIDTH_M,
    TUNNEL_LENGTH_M,
    USE_GPU,
    compute_analytic_parameters,
    compute_spatial_spectrum,
    derived_tolerances,
    discrete_modal_prediction,
    fit_beta_q_from_envelope,
    fit_window,
    format_error_budget,
    measure_empirical_noise,
    run_linearized_beam_drift,
    run_particle_beam_drift,
)
from wakis import figstyle
from wakis.estimate import estimate, preflight

IMG_DIR = REPO_ROOT / "docs" / "img"
JSON_PATH = Path(__file__).resolve().with_suffix(".json")


def run_resolution_pair(nx: int, nz: int, params: dict[str, float]) -> dict:
    """Run both `LinearizedBeam` and `ParticleSolver` (`profile="sheet_lattice"`) on grid ``(nx, 1, nz)``."""
    beta_e = params["beta_e"]
    beta_q_th = params["beta_q"]
    lam_q = params["lambda_q"]
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    lz = TUNNEL_LENGTH_M

    dz = lz / nz
    dx = 2.0 * a / nx
    z_min, z_max = fit_window(params)
    tols = derived_tolerances(nx, nz, params)

    t0_lb = time.perf_counter()
    lb_res = run_linearized_beam_drift(nx, nz, rho0_scale=1.0)
    wall_lb_s = time.perf_counter() - t0_lb

    dt = lb_res["dt"]
    n_spp = lb_res["n_spp"]
    n_settle = int(np.ceil(1.01 * lz / (params["u_dep"] * dt)))
    n_dft = 6 * n_spp
    n_total = n_settle + n_dft

    est_cpu = estimate((nx, 1, nz), steps=n_total, dtype="float64", backend="cpu")
    preflight(est_cpu, "laptop", raise_error=True)

    t0_part = time.perf_counter()
    part_res = run_particle_beam_drift(
        nx,
        nz,
        beam_ref=lb_res["beam"],
        qm_scale=1.0,
        seed=42,
        profile="sheet_lattice",
    )
    wall_part_s = time.perf_counter() - t0_part

    z_coords = lb_res["z_coords"]
    i1_lb = lb_res["i1_lb"]
    i2_lb = lb_res["i2_lb"]
    i1_part = part_res["i1_part"]
    i2_part = part_res["i2_part"]

    fit_lb = fit_beta_q_from_envelope(
        z_coords, i1_lb, beta_q_th, z_min=z_min, z_max=z_max
    )
    fit_part = fit_beta_q_from_envelope(
        z_coords, i1_part, beta_q_th, z_min=z_min, z_max=z_max
    )
    bq_lb = fit_lb["beta_q"]
    bq_part = fit_part["beta_q"]
    z0_lb = fit_lb["z0"]
    z0_part = fit_part["z0"]

    delta_meas_lat = (params["u0"] - part_res["u_z_mean"]) / params["u0"]
    mod_lb = discrete_modal_prediction("lb", nx, nz, params, return_details=True)
    mod_part = discrete_modal_prediction(
        "particles", nx, nz, params, return_details=True
    )
    bq_lb_pred = tols["bq_pred_lb"]
    bq_part_pred = tols["bq_pred_part"]
    resid_lb = abs(bq_lb / bq_lb_pred - 1.0)
    resid_lat = abs(bq_part / bq_part_pred - 1.0)
    resid_cross = abs(bq_part / bq_lb - bq_part_pred / bq_lb_pred)

    kz_grid, spec_lb, k_fast_lb, k_slow_lb = compute_spatial_spectrum(
        z_coords, i1_lb, beta_e, beta_q_th, z_max=lam_q
    )
    _, spec_part, k_fast_part, k_slow_part = compute_spatial_spectrum(
        z_coords, i1_part, beta_e, beta_q_th, z_max=lam_q
    )

    active_mask = z_coords <= lam_q
    max_harm2_ratio_lb = float(np.max(np.abs(i2_lb[active_mask])) / np.abs(i1_lb[0]))
    max_harm2_ratio_part = float(
        np.max(np.abs(i2_part[active_mask])) / np.abs(i1_part[0])
    )
    rms_harm2_ratio_part = float(
        np.sqrt(np.mean(np.abs(i2_part[active_mask]) ** 2)) / np.abs(i1_part[0])
    )

    i1_fit_lb = fit_lb["a_f"] * np.exp(-1j * fit_lb["k_f"] * z_coords) + fit_lb[
        "a_s"
    ] * np.exp(-1j * fit_lb["k_s"] * z_coords)
    i1_fit_part = fit_part["a_f"] * np.exp(-1j * fit_part["k_f"] * z_coords) + fit_part[
        "a_s"
    ] * np.exp(-1j * fit_part["k_s"] * z_coords)
    i1_model_lb = mod_lb["i1_model"]
    i1_model_part = mod_part["i1_model"]

    n_macro = 16 if nx == COARSE_GRID[0] else 32
    metrics = {
        "grid_shape": [int(nx), 1, int(nz)],
        "dx_m": float(dx),
        "dz_m": float(dz),
        "dt_s": float(dt),
        "n_steps_per_period": int(n_spp),
        "n_steps_total": int(n_total),
        "n_steps_dft": int(n_dft),
        "n_macro_per_step": int(n_macro),
        "fit_window_m": [float(z_min), float(z_max)],
        "cells_per_lambda_e": float(params["lambda_e"] / dz),
        "cells_per_lambda_0": float(params["lambda_0"] / dz),
        "cells_per_lambda_p": float(params["lambda_p"] / dz),
        "cells_across_beam_2b": float((2.0 * b) / dx),
        "estimate_host_bytes": float(est_cpu.host_bytes),
        "wall_time_linearized_beam_s": float(wall_lb_s),
        "wall_time_particles_s": float(wall_part_s),
        "entrance_ratio_linearized_beam": float(np.abs(i1_lb[0]) / params["I1_0"]),
        "entrance_ratio_particles": float(np.abs(i1_part[0]) / params["I1_0"]),
        "bq_th": float(beta_q_th),
        "bq_pred_lb": float(bq_lb_pred),
        "bq_lb": float(bq_lb),
        "resid_lb": float(resid_lb),
        "slack_lb": float(tols["slack_lb"]),
        "s_win_lb": float(tols["s_win_lb"]),
        "s_modes_lb": float(tols["s_modes_lb"]),
        "s_sym_lb": float(tols["s_sym_lb"]),
        "bq_pred_part": float(bq_part_pred),
        "bq_pred_part_no_sz": float(tols["bq_pred_part_no_sz"]),
        "lattice": float(bq_part),
        "resid_lattice": float(resid_lat),
        "slack_part": float(tols["slack_part"]),
        "s_win_part": float(tols["s_win_part"]),
        "s_modes_part": float(tols["s_modes_part"]),
        "s_sym_part": float(tols["s_sym_part"]),
        "alias_supp": float(tols["alias_supp"]),
        "amp_pred_part": float(tols["amp_pred_part"]),
        "resid_cross": float(resid_cross),
        "slack_cross": float(tols["slack_cross"]),
        "delta_th": float(tols["delta_th"]),
        "delta_disc_exact": float(tols["delta_disc_exact"]),
        "delta_disc_flux": float(tols["delta_disc_flux"]),
        "delta_disc_density": float(tols["delta_disc_density"]),
        "delta_meas_lattice": float(delta_meas_lat),
        "beta_q_linearized_beam_rad_m": float(bq_lb),
        "k_c_linearized_beam_rad_m": float(fit_lb["k_c"]),
        "k_f_fit_linearized_beam_rad_m": float(fit_lb["k_f"]),
        "k_s_fit_linearized_beam_rad_m": float(fit_lb["k_s"]),
        "sigma_beta_q_linearized_beam_rad_m": float(fit_lb["sigma_beta_q"]),
        "as_over_af_linearized_beam": float(fit_lb["as_over_af"]),
        "z0_linearized_beam_m": float(z0_lb),
        "beta_q_particles_rad_m": float(bq_part),
        "k_c_particles_rad_m": float(fit_part["k_c"]),
        "k_f_fit_particles_rad_m": float(fit_part["k_f"]),
        "k_s_fit_particles_rad_m": float(fit_part["k_s"]),
        "sigma_beta_q_particles_rad_m": float(fit_part["sigma_beta_q"]),
        "as_over_af_particles": float(fit_part["as_over_af"]),
        "z0_particles_m": float(z0_part),
        "rel_error_linearized_beam_vs_analytic": float((bq_lb - beta_q_th) / beta_q_th),
        "rel_error_particles_vs_analytic": float((bq_part - beta_q_th) / beta_q_th),
        "rel_diff_particles_vs_linearized_beam": float((bq_part - bq_lb) / bq_lb),
        "k_fast_linearized_beam_rad_m": float(k_fast_lb),
        "k_slow_linearized_beam_rad_m": float(k_slow_lb),
        "k_fast_particles_rad_m": float(k_fast_part),
        "k_slow_particles_rad_m": float(k_slow_part),
        "dc_leakage_linearized_beam": float(lb_res["dc_leakage_lb"]),
        "dc_leakage_particles": float(part_res["dc_leakage_part"]),
        "u_z_mean_particles_m_s": float(part_res["u_z_mean"]),
        "rho_dc_mean_particles_c_m3": float(part_res["rho_dc_mean"]),
        "max_second_harmonic_ratio_linearized_beam": float(max_harm2_ratio_lb),
        "max_second_harmonic_ratio_particles": float(max_harm2_ratio_part),
        "rms_second_harmonic_ratio": float(rms_harm2_ratio_part),
    }

    curves = {
        "z_mm": z_coords * 1e3,
        "z_min_mm": float(z_min * 1e3),
        "z_max_mm": float(z_max * 1e3),
        "env_analytic_norm": np.abs(np.cos(beta_q_th * (z_coords - z0_part))),
        "env_linearized_beam_norm": np.abs(i1_lb) / params["I1_0"],
        "env_particles_norm": np.abs(i1_part) / params["I1_0"],
        "env_model_linearized_beam_norm": np.abs(i1_model_lb) / params["I1_0"],
        "env_model_particles_norm": np.abs(i1_model_part) / params["I1_0"],
        "env_fit_linearized_beam_norm": np.abs(i1_fit_lb) / params["I1_0"],
        "env_fit_particles_norm": np.abs(i1_fit_part) / params["I1_0"],
        "carrier_analytic_norm": np.real(
            np.cos(beta_q_th * (z_coords - z0_part)) * np.exp(-1j * beta_e * z_coords)
        ),
        "carrier_linearized_beam_norm": np.imag(i1_lb) / params["I1_0"],
        "carrier_particles_norm": np.imag(i1_part) / params["I1_0"],
        "kz_rad_m": kz_grid,
        "spec_linearized_beam_norm": spec_lb / params["I1_0"],
        "spec_particles_norm": spec_part / params["I1_0"],
    }

    return {"metrics": metrics, "curves": curves}


def make_figure_envelope(
    run_data: dict, params: dict[str, float], out_path: Path
) -> None:
    """Create Figure 1: Axial current envelope ``|I1(z)|``, two-wave fit, and modulated carrier wave."""
    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=3.4,
        nrows=1,
        ncols=2,
        gridspec_kw={"wspace": 0.34},
    )
    ax_env, ax_car = axes[0], axes[1]

    curves = run_data["curves"]
    z_mm = curves["z_mm"]
    z_min_mm = curves["z_min_mm"]
    z_max_mm = curves["z_max_mm"]
    lam_q_mm = params["lambda_q"] * 1e3
    ramp_start_mm = (TUNNEL_LENGTH_M - OUTFLOW_RAMP_M) * 1e3

    ax_env.axvspan(
        z_min_mm,
        z_max_mm,
        color="#e6f2ff",
        alpha=0.55,
        label=r"Two-wave fit window $[3/T_1, L_z - L_{\mathrm{ramp}} - 3/T_1]$",
    )
    ax_env.plot(
        z_mm,
        curves["env_analytic_norm"],
        color=figstyle.INK,
        linestyle="--",
        linewidth=1.5,
        label=r"Analytic $|\cos(\beta_q(z - z_0))|$",
    )
    st_lb = figstyle.series_style(0, with_marker=False, lw=1.9)
    st_part = figstyle.series_style(1, with_marker=True, markevery=12, ms=4.2)
    st_mod_lb = figstyle.series_style(2, with_marker=False, lw=1.4)
    st_mod_lb["linestyle"] = "-."
    st_mod_part = figstyle.series_style(3, with_marker=False, lw=1.4)
    st_mod_part["linestyle"] = ":"
    ax_env.plot(
        z_mm,
        curves["env_linearized_beam_norm"],
        label="LinearizedBeam",
        **st_lb,
    )
    ax_env.plot(
        z_mm,
        curves["env_particles_norm"],
        label="Particles (sheet_lattice)",
        **st_part,
    )
    ax_env.plot(
        z_mm,
        curves["env_model_linearized_beam_norm"],
        label=r"$|I_{1,\mathrm{model}}|$ (LinearizedBeam)",
        **st_mod_lb,
    )
    ax_env.plot(
        z_mm,
        curves["env_model_particles_norm"],
        label=r"$|I_{1,\mathrm{model}}|$ (Particles)",
        **st_mod_part,
    )
    ax_env.axvspan(
        ramp_start_mm,
        TUNNEL_LENGTH_M * 1e3,
        color="#d9d9d9",
        alpha=0.45,
        label="Outflow ramp",
    )
    ax_env.set_xlabel(r"Axial coordinate $z$ [mm]")
    ax_env.set_ylabel(r"Normalized envelope $|I_1(z)| / (m I_0)$")
    ax_env.set_title(r"(a) Space-charge standing-wave envelope")
    ax_env.set_xlim(0.0, TUNNEL_LENGTH_M * 1e3)
    ax_env.set_ylim(-0.04, 1.18)
    figstyle.spectrum_grid(ax_env)
    figstyle.place_legend_below(ax_env, ncol=2, y_offset=-0.24, fontsize=8.0)

    ax_car.plot(
        z_mm,
        curves["carrier_analytic_norm"],
        color=figstyle.INK,
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
        label=r"Analytic $\cos(\beta_q(z - z_0))\cos(\beta_e z)$",
    )
    ax_car.plot(
        z_mm,
        curves["carrier_linearized_beam_norm"],
        label="LinearizedBeam quadrature",
        **st_lb,
    )
    st_part_car = figstyle.series_style(1, with_marker=False, lw=1.4)
    ax_car.plot(
        z_mm,
        curves["carrier_particles_norm"],
        label="Particles quadrature",
        **st_part_car,
    )
    ax_car.set_xlabel(r"Axial coordinate $z$ [mm]")
    ax_car.set_ylabel(r"Quadrature phasor $\mathrm{Im}[I_1(z)] / (m I_0)$")
    ax_car.set_title(r"(b) Carrier wave and phase reversal at nodes")
    ax_car.set_xlim(0.0, lam_q_mm)
    ax_car.set_ylim(-1.20, 1.20)
    figstyle.spectrum_grid(ax_car)
    figstyle.place_legend_below(ax_car, ncol=2, y_offset=-0.24, fontsize=8.0)

    figstyle.save(fig, out_path, auto_legend_below=False)
    matplotlib.pyplot.close(fig)


def make_figure_spectrum(
    run_data: dict, params: dict[str, float], out_path: Path
) -> None:
    """Create Figure 2: Spatial wavenumber spectrum ``|I1(k_z)|`` with ``k_f`` and ``k_s`` marked."""
    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=3.4,
        nrows=1,
        ncols=2,
        gridspec_kw={"wspace": 0.34},
    )
    ax_k, ax_shift = axes[0], axes[1]

    curves = run_data["curves"]
    metrics = run_data["metrics"]
    kz = curves["kz_rad_m"]
    beta_e = params["beta_e"]
    beta_q = params["beta_q"]
    k_fast = beta_e - beta_q
    k_slow = beta_e + beta_q
    kf_fit = metrics["k_f_fit_particles_rad_m"]
    ks_fit = metrics["k_s_fit_particles_rad_m"]

    st_lb = figstyle.series_style(0, with_marker=False, lw=1.9)
    st_part = figstyle.series_style(1, with_marker=True, markevery=35, ms=4.2)

    ax_k.plot(kz, curves["spec_linearized_beam_norm"], label="LinearizedBeam", **st_lb)
    ax_k.plot(kz, curves["spec_particles_norm"], label="Particles", **st_part)
    ax_k.axvline(
        k_fast,
        color=figstyle.INK,
        linestyle="--",
        linewidth=1.2,
        label=r"Analytic $k_f = \beta_e - \beta_q$",
    )
    ax_k.axvline(
        k_slow,
        color=figstyle.INK,
        linestyle="-.",
        linewidth=1.2,
        label=r"Analytic $k_s = \beta_e + \beta_q$",
    )
    ax_k.axvline(
        kf_fit,
        color="#d95f02",
        linestyle=":",
        linewidth=1.1,
        label=r"Fitted $k_f, k_s$ (particles)",
    )
    ax_k.axvline(
        ks_fit,
        color="#d95f02",
        linestyle=":",
        linewidth=1.1,
    )
    ax_k.axvline(
        beta_e,
        color="#666666",
        linestyle=":",
        linewidth=1.0,
        label=r"Carrier $\beta_e = \omega / u_{\mathrm{dep}}$",
    )
    ax_k.set_xlabel(r"Axial wavenumber $k_z$ [rad/m]")
    ax_k.set_ylabel(r"Spatial spectrum $|\hat{I}_1(k_z)| / (m I_0)$")
    ax_k.set_title(r"(a) Fast ($k_f$) and slow ($k_s$) space-charge wave peaks")
    ax_k.set_xlim(kz[0], kz[-1])
    ax_k.set_ylim(-0.02, 0.62)
    figstyle.spectrum_grid(ax_k)
    figstyle.place_legend_below(ax_k, ncol=2, y_offset=-0.24, fontsize=8.0)

    norm_shift = (kz - beta_e) / beta_q
    ax_shift.plot(
        norm_shift,
        curves["spec_linearized_beam_norm"],
        label="LinearizedBeam",
        **st_lb,
    )
    ax_shift.plot(
        norm_shift,
        curves["spec_particles_norm"],
        label="Particles",
        **st_part,
    )
    ax_shift.axvline(
        -1.0,
        color=figstyle.INK,
        linestyle="--",
        linewidth=1.2,
        label=r"$(k_f - \beta_e) / \beta_q = -1$",
    )
    ax_shift.axvline(
        +1.0,
        color=figstyle.INK,
        linestyle="-.",
        linewidth=1.2,
        label=r"$(k_s - \beta_e) / \beta_q = +1$",
    )
    ax_shift.axvline(
        0.0,
        color="#666666",
        linestyle=":",
        linewidth=1.0,
        label=r"Null $\Delta k = 0$",
    )
    ax_shift.set_xlabel(r"Normalized beat shift $(k_z - \beta_e) / \beta_q$")
    ax_shift.set_ylabel(r"Spatial spectrum $|\hat{I}_1(k_z)| / (m I_0)$")
    ax_shift.set_title(r"(b) Doublet splitting in normalized $\beta_q$ units")
    ax_shift.set_xlim(-3.5, 3.5)
    ax_shift.set_ylim(-0.02, 0.62)
    figstyle.spectrum_grid(ax_shift)
    figstyle.place_legend_below(ax_shift, ncol=2, y_offset=-0.24, fontsize=8.0)

    figstyle.save(fig, out_path, auto_legend_below=False)
    matplotlib.pyplot.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validate wakis.particles against LinearizedBeam on space-charge waves."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Run only the coarse resolution {COARSE_GRID} for a fast smoke check.",
    )
    args = parser.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    params = compute_analytic_parameters()

    print(
        f"Depressed analytic parameters: R={params['R']:.6f}, "
        f"beta_e={params['beta_e']:.2f} rad/m, "
        f"beta_q={params['beta_q']:.4f} rad/m (undepressed {params['beta_q_0']:.4f} rad/m), "
        f"delta_u_dc={params['delta_u_dc']:.4%}, "
        f"lambda_q={params['lambda_q'] * 1e3:.2f} mm"
    )

    coarse_out = run_resolution_pair(COARSE_GRID[0], COARSE_GRID[2], params)
    mc = coarse_out["metrics"]
    print(
        f"[Coarse {COARSE_GRID}] "
        f"LB: |I1(0)|/(m*I0)={mc['entrance_ratio_linearized_beam']:.5f}, "
        f"beta_q={mc['beta_q_linearized_beam_rad_m']:.4f} rad/m "
        f"(pred={mc['bq_pred_lb']:.4f} rad/m, resid={mc['resid_lb']:.4%}, slack={mc['slack_lb']:.4%}), "
        f"z0={mc['z0_linearized_beam_m'] * 1e3:.3f} mm; "
        f"Particles (sheet_lattice): |I1(0)|/(m*I0)={mc['entrance_ratio_particles']:.5f}, "
        f"beta_q={mc['beta_q_particles_rad_m']:.4f} rad/m "
        f"(pred={mc['bq_pred_part']:.4f} rad/m, resid={mc['resid_lattice']:.4%}, slack={mc['slack_part']:.4%}), "
        f"z0={mc['z0_particles_m'] * 1e3:.3f} mm"
    )

    resolutions_payload = {"coarse": mc}
    plot_source = coarse_out

    if not args.smoke:
        noise_c = measure_empirical_noise(COARSE_GRID)
        noise_f = measure_empirical_noise(FINE_GRID)
        tols_c = derived_tolerances(COARSE_GRID[0], COARSE_GRID[2], params)
        tols_f = derived_tolerances(FINE_GRID[0], FINE_GRID[2], params)
        print(format_error_budget(COARSE_GRID[0], COARSE_GRID[2], tols_c))
        print(format_error_budget(FINE_GRID[0], FINE_GRID[2], tols_f))

        fine_out = run_resolution_pair(FINE_GRID[0], FINE_GRID[2], params)
        mf = fine_out["metrics"]
        print(
            f"[Fine   {FINE_GRID}] "
            f"LB: |I1(0)|/(m*I0)={mf['entrance_ratio_linearized_beam']:.5f}, "
            f"beta_q={mf['beta_q_linearized_beam_rad_m']:.4f} rad/m "
            f"(pred={mf['bq_pred_lb']:.4f} rad/m, resid={mf['resid_lb']:.4%}, slack={mf['slack_lb']:.4%}), "
            f"z0={mf['z0_linearized_beam_m'] * 1e3:.3f} mm; "
            f"Particles (sheet_lattice): |I1(0)|/(m*I0)={mf['entrance_ratio_particles']:.5f}, "
            f"beta_q={mf['beta_q_particles_rad_m']:.4f} rad/m "
            f"(pred={mf['bq_pred_part']:.4f} rad/m, resid={mf['resid_lattice']:.4%}, slack={mf['slack_part']:.4%}), "
            f"z0={mf['z0_particles_m'] * 1e3:.3f} mm"
        )
        for m_dict, noise in ((mc, noise_c), (mf, noise_f)):
            tcrit = float(noise["tcrit"])
            sem = float(noise["bq_sem"])
            mean_sh = float(noise["bq_mean"])
            pred_p = float(m_dict["bq_pred_part"])
            m_dict["bq_coherent_mean"] = float(noise["bq_coherent_mean"])
            m_dict["mean_sheet"] = mean_sh
            m_dict["sem"] = sem
            m_dict["tcrit"] = tcrit
            m_dict["resid_mean"] = float(abs(mean_sh / pred_p - 1.0))
            m_dict["bound_mean"] = float(m_dict["slack_part"] + tcrit * sem / pred_p)
            m_dict["mean_minus_lattice"] = float(abs(mean_sh - m_dict["lattice"]))
            m_dict["seeds"] = [float(v) for v in noise["bq_seeds"]]
            m_dict["delta_meas_mean"] = float(noise["delta_mean"])
            m_dict["delta_sem"] = float(noise["delta_sem"])

        r_lb = abs(mc["rel_error_linearized_beam_vs_analytic"]) / abs(
            mf["rel_error_linearized_beam_vs_analytic"]
        )
        r_part = abs(mc["rel_error_particles_vs_analytic"]) / abs(
            mf["rel_error_particles_vs_analytic"]
        )
        r_cross = abs(mc["rel_diff_particles_vs_linearized_beam"]) / abs(
            mf["rel_diff_particles_vs_linearized_beam"]
        )
        resolutions_payload["fine"] = mf
        resolutions_payload["convergence"] = {
            "linearized_beam_error_reduction_factor": float(r_lb),
            "linearized_beam_observed_order": float(np.log2(r_lb)),
            "particles_error_reduction_factor": float(r_part),
            "particles_observed_order": float(np.log2(r_part)),
            "cross_code_diff_reduction_factor": float(r_cross),
            "cross_code_observed_order": float(np.log2(r_cross)),
        }
        plot_source = fine_out

    fig1_path = IMG_DIR / "particles_space_charge_waves_envelope.png"
    fig2_path = IMG_DIR / "particles_space_charge_waves_spectrum.png"
    make_figure_envelope(plot_source, params, fig1_path)
    make_figure_spectrum(plot_source, params, fig2_path)

    payload = {
        "mode": "smoke" if args.smoke else "full",
        "backend": "gpu" if USE_GPU else "cpu",
        "analytic": params,
        "resolutions": resolutions_payload,
        "figures": [
            str(fig1_path.relative_to(REPO_ROOT)),
            str(fig2_path.relative_to(REPO_ROOT)),
        ],
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved figures to {fig1_path} and {fig2_path}")
    print(f"Saved metrics to {JSON_PATH}")


if __name__ == "__main__":
    main()
