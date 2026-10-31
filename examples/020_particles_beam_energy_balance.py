# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Example 020 — Relativistic Beam Injection and Discrete FIT Energy Balance
=========================================================================

Demonstrates milestone P4a of ``wakis.particles`` (:class:`BeamInjector` and
:meth:`ParticleSolver.energy_balance` / :meth:`ParticleSolver.field_work`):

1. **Time-domain energy balance of a relativistic electron beam in a PEC cavity**:
   - A relativistic electron beam (``gamma_0 = 2.0``, ``I_0 = 0.25 A``) is
     emitted from a PEC cathode plane ``z_0 = 0`` (``emitter_kind == "pec_wall"``)
     with a smooth half-cosine-squared rise profile (``rise_shape="cos2"``) into
     a closed PEC box containing a downstream :class:`Collector`.
   - Tracks the half-step-staggered discrete FIT field energy ``W_field``,
     synchronised particle kinetic energy ``K_particles``, cumulative injected
     kinetic energy ``K_injected``, cumulative absorbed kinetic energy
     ``K_absorbed``, cumulative discrete field work ``field_work``, the exact
     discrete FIT field identity ``(W_field - W_field(0)) + field_work = 0``,
     and the total closure residual
     ``residual = K_injected - K_absorbed - K_particles - W_field``.
2. **Second-order ``O(dt^2)`` convergence of the total energy closure residual before absorption**:
   - Halves the time step ``dt`` across four values (``0.8, 0.4, 0.2, 0.1`` of
     the 3D FIT CFL limit) at a fixed spatial grid (resolving the beam radius
     with ``>= 4`` cells) and physical integration time ``T`` before collector
     impact using a deterministic ``profile="lattice"`` beam, verifying that
     ``|residual|`` decreases as ``O(dt^2)`` on log-log axes.

Saves two figures (via ``wakis.figstyle``) and writes all numerical results to
JSON. Supports ``--smoke`` for fast (< 60 s) CPU verification and ``--out``,
``--fig-balance``, and ``--fig-convergence`` output path overrides.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.ticker
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import m_e

from wakis import GridFIT3D, SolverFIT3D, figstyle
from wakis.estimate import estimate
from wakis.particles import (
    BeamInjector,
    Collector,
    ParticleSolver,
    Species,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / "docs" / "img"
DEFAULT_JSON_PATH = Path(__file__).resolve().with_suffix(".json")
DEFAULT_FIG_BALANCE = IMG_DIR / "particles_beam_energy_balance_time.png"
DEFAULT_FIG_CONVERGENCE = IMG_DIR / "particles_beam_energy_closure_convergence.png"


def check_resolution(
    dx: float,
    dy: float,
    dz: float,
    radius: float,
    gap_min: float,
    tau_rise: float,
    current: float,
    gamma0: float,
) -> dict[str, float]:
    """Verify grid resolution rules for beam radius, vacuum wavelength, and plasma wavelength."""
    h_max = max(dx, dy, dz)
    beta0 = float(np.sqrt(1.0 - 1.0 / (gamma0 * gamma0)))
    u0 = gamma0 * beta0 * c_light
    f_max = 1.0 / (2.0 * max(tau_rise, 1.0e-12))
    lambda_vac = c_light / f_max
    n_b = abs(current) / (e_charge * np.pi * radius * radius * beta0 * c_light)
    omega_p = float(np.sqrt(n_b * e_charge * e_charge / (eps_0 * m_e)))
    lambda_p = 2.0 * np.pi * u0 / omega_p

    cells_per_radius = radius / max(dx, dy)
    cells_per_gap = gap_min / dz
    cells_per_lambda_vac = lambda_vac / h_max
    cells_per_lambda_p = lambda_p / h_max

    assert cells_per_radius >= 4.0 - 1e-9
    assert cells_per_gap >= 4.0 - 1e-9
    assert cells_per_lambda_vac >= 20.0
    assert cells_per_lambda_p >= 20.0
    return {
        "dx_m": float(dx),
        "dy_m": float(dy),
        "dz_m": float(dz),
        "cells_per_beam_radius": float(cells_per_radius),
        "cells_per_min_gap": float(cells_per_gap),
        "cells_per_vacuum_wavelength": float(cells_per_lambda_vac),
        "cells_per_plasma_wavelength": float(cells_per_lambda_p),
    }


def run_time_history(smoke: bool = False) -> dict:
    """Run beam-in-PEC-box simulation with collector and record energy terms vs time."""
    if smoke:
        nx, ny, nz = 16, 16, 18
        lx, ly, lz = 0.08, 0.08, 0.09
        z_col = 0.060
        n_steps = 48
        n_macro = 8
    else:
        nx, ny, nz = 16, 16, 24
        lx, ly, lz = 0.08, 0.08, 0.12
        z_col = 0.080
        n_steps = 72
        n_macro = 16

    radius = 0.020
    current = 0.25
    gamma0 = 2.0

    grid = GridFIT3D(
        -0.5 * lx,
        0.5 * lx,
        -0.5 * ly,
        0.5 * ly,
        0.0,
        lz,
        nx,
        ny,
        nz,
        verbose=0,
    )
    if nx * ny * nz >= 1_000_000:
        estimate(grid, n_steps=n_steps)

    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        verbose=0,
    )
    dt = float(solver.dt)
    tau_rise = max(12.0 * dt, 0.5e-9)
    res_info = check_resolution(
        dx=lx / nx,
        dy=ly / ny,
        dz=lz / nz,
        radius=radius,
        gap_min=min(z_col, lz - z_col),
        tau_rise=tau_rise,
        current=current,
        gamma0=gamma0,
    )

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    inj = BeamInjector(
        species=sp,
        z0=0.0,
        current=current,
        gamma=gamma0,
        profile="uniform",
        radius=radius,
        n_macro_per_step=n_macro,
        rise_time=tau_rise,
        rise_shape="cos2",
        seed=2026,
    )
    col = Collector(name="collector", plane=("z", z_col, 1))
    ps = ParticleSolver(
        solver,
        species=[sp],
        injectors=[inj],
        collectors=[col],
        pusher="boris",
        shape=2,
        gather="galerkin",
        track_energy=True,
    ).attach()

    t_ns = [0.0]
    w_field = [0.0]
    k_particles = [0.0]
    k_injected = [0.0]
    k_absorbed = [0.0]
    field_work = [0.0]
    residual = [0.0]
    field_identity_err = [0.0]
    gauss_res = [0.0]

    for step in range(1, n_steps + 1):
        solver.one_step()
        eb = ps.energy_balance()
        t_ns.append(float(step * dt * 1.0e9))
        w_field.append(float(eb["W_field"]))
        k_particles.append(float(eb["K_particles"]))
        k_injected.append(float(eb["K_injected"]))
        k_absorbed.append(float(eb["K_absorbed"]))
        field_work.append(float(eb["field_work"]))
        residual.append(float(eb["residual"]))
        field_identity_err.append(float(eb["W_field"] + eb["field_work"]))
        gauss_res.append(float(ps.gauss_residual()))

    return {
        "parameters": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "lx_m": lx,
            "ly_m": ly,
            "lz_m": lz,
            "z_collector_m": z_col,
            "dt_s": dt,
            "n_steps": n_steps,
            "current_A": current,
            "gamma0": gamma0,
            "radius_m": radius,
            "rise_time_s": tau_rise,
            "rise_shape": "cos2",
            "emitter_kind": inj.emitter_kind,
            "n_macro_per_step": n_macro,
        },
        "resolution": res_info,
        "history": {
            "t_ns": t_ns,
            "W_field_J": w_field,
            "K_particles_J": k_particles,
            "K_injected_J": k_injected,
            "K_absorbed_J": k_absorbed,
            "field_work_J": field_work,
            "residual_J": residual,
            "field_identity_error_J": field_identity_err,
            "gauss_residual": gauss_res,
        },
        "final": {
            "W_field_J": w_field[-1],
            "K_particles_J": k_particles[-1],
            "K_injected_J": k_injected[-1],
            "K_absorbed_J": k_absorbed[-1],
            "field_work_J": field_work[-1],
            "residual_J": residual[-1],
            "max_abs_field_identity_error_J": float(np.max(np.abs(field_identity_err))),
            "max_gauss_residual": float(np.max(gauss_res)),
        },
    }


def run_dt_convergence(smoke: bool = False) -> dict:
    """Run CFL = 0.8, 0.4, 0.2, 0.1 dt-halving study before absorption at fixed grid and duration."""
    if smoke:
        nx, ny, nz = 16, 16, 24
        lx, ly, lz = 0.08, 0.08, 0.12
        n_steps_base = 44
        n_macro = 8
    else:
        nx, ny, nz = 16, 16, 24
        lx, ly, lz = 0.08, 0.08, 0.12
        n_steps_base = 44
        n_macro = 16

    radius = 0.020
    z_col = 0.100
    current = 0.50
    gamma0 = 1.8

    grid_ref = GridFIT3D(
        -0.5 * lx,
        0.5 * lx,
        -0.5 * ly,
        0.5 * ly,
        0.0,
        lz,
        nx,
        ny,
        nz,
        verbose=0,
    )
    solver_ref = SolverFIT3D(
        grid_ref,
        cfln=0.8,
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        verbose=0,
    )
    t_total = float(n_steps_base) * float(solver_ref.dt)
    tau_rise = 0.5 * t_total
    check_resolution(
        dx=lx / nx,
        dy=ly / ny,
        dz=lz / nz,
        radius=radius,
        gap_min=min(z_col, lz - z_col),
        tau_rise=tau_rise,
        current=current,
        gamma0=gamma0,
    )

    cfl_factors = [0.8, 0.4, 0.2, 0.1]
    dt_list = []
    abs_residuals = []
    rel_residuals = []
    runs = []

    for cfl_val in cfl_factors:
        grid = GridFIT3D(
            -0.5 * lx,
            0.5 * lx,
            -0.5 * ly,
            0.5 * ly,
            0.0,
            lz,
            nx,
            ny,
            nz,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            cfln=cfl_val,
            bc_low=["pec", "pec", "pec"],
            bc_high=["pec", "pec", "pec"],
            verbose=0,
        )
        n_steps = round(t_total / float(solver.dt))
        solver.dt = t_total / float(n_steps)
        dt = float(solver.dt)

        sp = Species(name="electron", charge=-e_charge, mass=m_e)
        inj = BeamInjector(
            species=sp,
            z0=0.0,
            current=current,
            gamma=gamma0,
            profile="lattice",
            radius=radius,
            n_macro_per_step=n_macro,
            rise_time=0.5 * t_total,
            rise_shape="cos2",
            seed=42,
        )
        col = Collector(name="collector", plane=("z", z_col, 1))
        ps = ParticleSolver(
            solver,
            species=[sp],
            injectors=[inj],
            collectors=[col],
            pusher="boris",
            shape=3,
            gather="galerkin",
            track_energy=True,
        ).attach()

        for _ in range(n_steps):
            solver.one_step()

        eb = ps.energy_balance()
        res_abs = abs(float(eb["residual"]))
        res_rel = res_abs / max(abs(float(eb["W_field"])), 1.0e-30)
        dt_list.append(dt)
        abs_residuals.append(res_abs)
        rel_residuals.append(res_rel)
        runs.append(
            {
                "cfln": float(cfl_val),
                "n_steps": int(n_steps),
                "dt_s": dt,
                "W_field_J": float(eb["W_field"]),
                "K_particles_J": float(eb["K_particles"]),
                "K_injected_J": float(eb["K_injected"]),
                "K_absorbed_J": float(eb["K_absorbed"]),
                "field_work_J": float(eb["field_work"]),
                "residual_J": float(eb["residual"]),
                "abs_residual_J": res_abs,
                "rel_residual_to_W_field": res_rel,
            }
        )

    log_dt = np.log(np.asarray(dt_list, dtype=np.float64))
    log_res = np.log(np.asarray(abs_residuals, dtype=np.float64))
    local_slopes = [float(s) for s in (np.diff(log_res) / np.diff(log_dt))]
    slope_spread = float(np.ptp(local_slopes))
    slope = float(np.polyfit(log_dt, log_res, 1)[0])
    return {
        "cfl_factors": cfl_factors,
        "dt_s": dt_list,
        "abs_residual_J": abs_residuals,
        "rel_residual_to_W_field": rel_residuals,
        "local_slopes": local_slopes,
        "richardson_slope_spread": slope_spread,
        "fitted_order_slope": slope,
        "runs": runs,
    }


def make_figure_balance(time_data: dict, out_path: Path) -> None:
    """Plot energy terms vs time and the closure residual / field identity."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h = time_data["history"]
    t_ns = np.asarray(h["t_ns"], dtype=float)

    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=3.2,
        nrows=1,
        ncols=2,
        gridspec_kw={"wspace": 0.34},
    )
    ax_e, ax_res = axes[0], axes[1]

    scale_uJ = 1.0e6
    s_inj = figstyle.series_style(0, with_marker=False, lw=1.9)
    s_part = figstyle.series_style(1, with_marker=False, lw=1.9)
    s_abs = figstyle.series_style(2, with_marker=False, lw=1.9)
    s_wf = figstyle.series_style(3, with_marker=False, lw=1.8)

    ax_e.plot(
        t_ns,
        np.asarray(h["K_injected_J"]) * scale_uJ,
        label=r"$K_{\mathrm{injected}}$",
        **s_inj,
    )
    ax_e.plot(
        t_ns,
        np.asarray(h["K_particles_J"]) * scale_uJ,
        label=r"$K_{\mathrm{particles}}$ (sync)",
        **s_part,
    )
    ax_e.plot(
        t_ns,
        np.asarray(h["K_absorbed_J"]) * scale_uJ,
        label=r"$K_{\mathrm{absorbed}}$",
        **s_abs,
    )
    ax_e.plot(
        t_ns,
        np.asarray(h["W_field_J"]) * scale_uJ,
        label=r"$W_{\mathrm{field}}$",
        **s_wf,
    )
    ax_e.set_xlabel("Time $t$ [ns]")
    ax_e.set_ylabel(r"Energy [$\mu$J]")
    ax_e.set_title("(a) Energy terms")
    figstyle.spectrum_grid(ax_e)
    figstyle.place_legend_below(ax_e, ncol=2, y_offset=-0.24)

    t_pos = t_ns[1:]
    res_abs = np.maximum(np.abs(np.asarray(h["residual_J"])[1:]), 1.0e-30)
    id_abs = np.maximum(np.abs(np.asarray(h["field_identity_error_J"])[1:]), 1.0e-30)
    wf_abs = np.maximum(np.abs(np.asarray(h["W_field_J"])[1:]), 1.0e-30)

    s_w = figstyle.series_style(0, with_marker=True, markevery=6, ms=4.5)
    s_r = figstyle.series_style(1, with_marker=True, markevery=6, ms=4.5)
    s_id = figstyle.series_style(2, with_marker=True, markevery=6, ms=4.5)

    ax_res.semilogy(t_pos, wf_abs, label=r"$W_{\mathrm{field}}(t)$ [J]", **s_w)
    ax_res.semilogy(
        t_pos,
        res_abs,
        label=r"Closure $|\mathrm{residual}(t)|$ [J]",
        **s_r,
    )
    ax_res.semilogy(
        t_pos,
        id_abs,
        label=r"Field identity $|\Delta W_{\mathrm{field}} + W_{\mathrm{work}}|$ [J]",
        **s_id,
    )
    ax_res.set_xlabel("Time $t$ [ns]")
    ax_res.set_ylabel("Energy magnitude [J]")
    ax_res.set_title("(b) Closure residual")
    figstyle.spectrum_grid(ax_res)
    figstyle.place_legend_below(ax_res, ncol=1, y_offset=-0.24)

    figstyle.save(fig, out_path, auto_legend_below=False)
    matplotlib.pyplot.close(fig)


def make_figure_convergence(conv_data: dict, out_path: Path) -> None:
    """Plot energy closure residual vs dt on log-log axes with an order-2 guide."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dts_ps = np.asarray(conv_data["dt_s"], dtype=float) * 1.0e12
    res_arr = np.asarray(conv_data["abs_residual_J"], dtype=float)
    slope = float(conv_data["fitted_order_slope"])

    fig, ax = figstyle.new_figure(
        width_in=0.75 * figstyle.FULL_WIDTH_IN,
        height_in=4.0,
        nrows=1,
        ncols=1,
    )
    st_num = figstyle.series_style(0, with_marker=True, ms=6.0)
    ax.loglog(
        dts_ps,
        res_arr,
        label=rf"Measured $|\mathrm{{residual}}|$ (fit slope $= {slope:.2f}$)",
        **st_num,
    )
    guide = res_arr[1] * (dts_ps / dts_ps[1]) ** 2
    ax.loglog(
        dts_ps,
        guide,
        color=figstyle.INK,
        linestyle="--",
        linewidth=1.4,
        label=r"$\mathcal{O}(\Delta t^2)$ reference guide",
    )
    ax.set_xlabel(r"Time step $\Delta t$ [ps]")
    ax.set_ylabel(r"Closure residual [J]")
    ax.set_title(r"Energy closure convergence vs $\Delta t$")
    figstyle.spectrum_grid(ax)
    # Label only the measured dt values: log minor tick labels collide on a short decade.
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(dts_ps))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    figstyle.place_legend_below(ax, ncol=1, y_offset=-0.22)

    figstyle.save(fig, out_path, auto_legend_below=False)
    matplotlib.pyplot.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Example 020: Relativistic beam injection and discrete FIT energy balance."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke-test configuration (< 60 s on CPU).",
    )
    parser.add_argument(
        "--out",
        "--data-path",
        dest="out",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--fig-balance",
        "--fig-time",
        dest="fig_balance",
        type=Path,
        default=DEFAULT_FIG_BALANCE,
        help="Output path for Figure 1 (energy balance vs time).",
    )
    parser.add_argument(
        "--fig-convergence",
        "--fig-dt",
        dest="fig_convergence",
        type=Path,
        default=DEFAULT_FIG_CONVERGENCE,
        help="Output path for Figure 2 (residual vs dt log-log convergence).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    """Execute Example 020 and write JSON and figures."""
    args = parse_args(argv)
    t0 = time.perf_counter()
    time_data = run_time_history(smoke=args.smoke)
    conv_data = run_dt_convergence(smoke=args.smoke)
    elapsed_s = float(time.perf_counter() - t0)

    make_figure_balance(time_data, args.fig_balance)
    make_figure_convergence(conv_data, args.fig_convergence)

    payload = {
        "mode": "smoke" if args.smoke else "full",
        "elapsed_s": elapsed_s,
        "time_balance": time_data,
        "dt_convergence": conv_data,
        "figures": {
            "fig_balance": str(args.fig_balance),
            "fig_convergence": str(args.fig_convergence),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        f"[Example 020 ({payload['mode']})] elapsed={elapsed_s:.2f}s | "
        f"max|dW_field + field_work|={time_data['final']['max_abs_field_identity_error_J']:.3e} J | "
        f"final residual={time_data['final']['residual_J']:.3e} J | "
        f"dt-convergence slope={conv_data['fitted_order_slope']:.3f}"
    )
    return payload


if __name__ == "__main__":
    main()
