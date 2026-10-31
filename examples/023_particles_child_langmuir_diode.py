# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Example 023 — Planar Child–Langmuir Diode and Cathode Surface Emission (V6)
===========================================================================

Demonstrates the surface emission models in ``wakis.particles``
(``SpaceChargeLimitedEmitter``, ``ThermionicEmitter``, ``FowlerNordheimEmitter``,
and ``child_langmuir_current_density``) on the V6 1D planar Child–Langmuir
vacuum diode benchmark:

1. **Transient and steady-state anode current density vs time**:
   - Cold electrons are emitted from the grounded PEC cathode at ``z = 0``
     across a planar gap ``d = 5 mm`` biased at ``V = 2.5 kV``
     (weakly relativistic, ``e V / (m_e c^2) << 1``, with Jory–Trivelpiece
     relativistic correction ``J_rel / J_CL = 1 - (3/28) * e V / (m_e c^2)``)
     using the Gauss-law ``SpaceChargeLimitedEmitter``.
   - Compares the time-dependent anode current density ``J_anode(t)`` across
     longitudinal grid resolutions against the analytical Child–Langmuir limit
     ``J_CL = (4 * eps_0 / 9) * sqrt(2 * e / m_e) * V^(3/2) / d^2``, fits the
     physical two-term cathode boundary-layer convergence model
     ``e(dz) = a * (dz / d)**(2/3) + b * (dz / d)`` (effective order
     ``p in [2/3, 1]``), and evaluates the Richardson-extrapolated continuum
     current ``J_inf = J_fine + (J_fine - J_mid) / (2**p - 1)``.
2. **Steady-state electrostatic potential and space-charge density profiles**:
   - Compares the time-averaged potential profile ``phi(z) / V`` and
     space-charge density profile ``|rho(z)| / rho_ref`` against the analytical
     Child–Langmuir power laws ``(z / d)^(4/3)`` and ``(z / d)^(-2/3)``,
     where ``rho_ref = 4 * eps_0 * V / (9 * d^2) = J_CL / v_max``.

Saves two publication-style figures in ``docs/img/`` (via ``wakis.figstyle``)
and writes all run metrics to ``examples/023_particles_child_langmuir_diode.json``:
  1. ``docs/img/particles_child_langmuir_current_vs_time.png``
  2. ``docs/img/particles_child_langmuir_profiles.png``

Usage
-----
Full benchmark run::

    python examples/023_particles_child_langmuir_diode.py

Quick smoke run::

    python examples/023_particles_child_langmuir_diode.py --smoke
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
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0, m_e

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wakis import GridFIT3D, SolverFIT3D, figstyle
from wakis.particles import (
    ParticleSolver,
    SpaceChargeLimitedEmitter,
    Species,
    child_langmuir_current_density,
    forbes_barrier_functions,
    fowler_nordheim_current_density,
    gauss_residual,
    richardson_dushman_current_density,
    schottky_barrier_lowering,
)

IMG_DIR = REPO_ROOT / "docs" / "img"
JSON_PATH = Path(__file__).resolve().with_suffix(".json")


def run_planar_diode_resolution(
    Nz: int,
    *,
    gap_d: float = 0.005,
    voltage: float = 2500.0,
    n_transits: float = 10.0,
) -> dict:
    """
    Simulate the 1D planar Child–Langmuir diode at longitudinal resolution ``Nz``.

    Parameters
    ----------
    Nz : int
        Number of longitudinal FIT cells across ``[0, gap_d]``.
    gap_d : float, optional
        Anode-to-cathode spacing :math:`d` [m]. Default is ``0.005``.
    voltage : float, optional
        Anode bias voltage :math:`V` [V]. Default is ``2500.0``.
    n_transits : float, optional
        Number of space-charge-limited transit times
        :math:`T_{\\text{tr}} = 3 d / v_{\\max}` to integrate. Default is ``10.0``.

    Returns
    -------
    dict
        Time history of the anode current density, time-averaged potential and
        space-charge density profiles, fitted power-law exponents, and discrete
        Gauss's law residual.
    """
    t0 = time.perf_counter()
    Lx = 0.05
    Ly = 0.05
    area = Lx * Ly
    dz = gap_d / float(Nz)

    grid = GridFIT3D(
        -0.5 * Lx, 0.5 * Lx, -0.5 * Ly, 0.5 * Ly, 0.0, gap_d, 1, 1, Nz, verbose=0
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.95,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )
    dt = float(solver.dt)

    electrons = Species(
        name="electron", charge=-q_e, mass=m_e, capacity=max(8 * Nz, 256)
    )
    emitter = SpaceChargeLimitedEmitter(
        solver,
        electrons,
        n_macro_per_cell=1,
        thermal_u=0.0,
        voltage=voltage,
        maintain_voltage=True,
    )
    psolver = ParticleSolver(
        solver,
        species=[electrons],
        injectors=[emitter],
        shape=1,
        gather="momentum",
        sort_every=0,
    ).attach()

    v_max = float(np.sqrt(2.0 * q_e * voltage / m_e))
    t_transit = 3.0 * gap_d / v_max
    t_end = float(n_transits) * t_transit
    n_steps = max(int(t_end / dt), 20)
    n_avg_start = n_steps // 2

    def _field_energy() -> float:
        ez = emitter._ez_3d_view()[0, 0, :Nz]
        return float(0.5 * epsilon_0 * area * dz * np.sum(ez**2))

    def _gap_kinetic_energy() -> float:
        n_p = electrons.n_particles
        if n_p == 0:
            return 0.0
        u2 = electrons.ux[:n_p] ** 2 + electrons.uy[:n_p] ** 2 + electrons.uz[:n_p] ** 2
        gamma = np.sqrt(1.0 + u2 / (c_light**2))
        return float(
            np.sum(
                electrons.w[:n_p]
                * electrons.mass
                * (c_light**2)
                * (u2 / ((gamma + 1.0) * (c_light**2)))
            )
        )

    t_hist = np.empty(n_steps, dtype=np.float64)
    j_anode_hist = np.empty(n_steps, dtype=np.float64)
    ez_sum = np.zeros(Nz, dtype=np.float64)
    n_avg = 0
    q_abs_prev = 0.0
    w_src_start = 0.0
    e_inj_start = 0.0
    k_anode_start = 0.0
    k_cathode_start = 0.0
    k_gap_start = 0.0
    w_field_start = 0.0

    for step in range(n_steps):
        if step == n_avg_start:
            emitter._enforce_gap_voltage()
            w_src_start = float(emitter.voltage_source_work)
            e_inj_start = float(emitter.injected_energy)
            k_anode_start = float(psolver.boundaries.absorbed_energy["z_high"])
            k_cathode_start = float(psolver.boundaries.absorbed_energy["z_low"])
            k_gap_start = _gap_kinetic_energy()
            w_field_start = _field_energy()
        solver.one_step()
        q_abs = float(psolver.boundaries.absorbed_charge["z_high"])
        dq_anode = q_abs - q_abs_prev
        q_abs_prev = q_abs
        t_hist[step] = (step + 1) * dt
        j_anode_hist[step] = abs(dq_anode) / (dt * area)
        if step >= n_avg_start:
            ez_sum += emitter._ez_3d_view()[0, 0, :Nz]
            n_avg += 1

    emitter._enforce_gap_voltage()
    w_in_steady = (float(emitter.voltage_source_work) - w_src_start) + (
        float(emitter.injected_energy) - e_inj_start
    )
    dk_anode_steady = (
        float(psolver.boundaries.absorbed_energy["z_high"]) - k_anode_start
    )
    dk_cathode_steady = (
        float(psolver.boundaries.absorbed_energy["z_low"]) - k_cathode_start
    )
    dk_gap_steady = _gap_kinetic_energy() - k_gap_start
    dw_field_steady = _field_energy() - w_field_start
    w_out_steady = dk_anode_steady + dk_cathode_steady + dk_gap_steady + dw_field_steady
    energy_rel_err = abs(w_in_steady - w_out_steady) / max(abs(w_in_steady), 1e-30)

    J_CL = child_langmuir_current_density(voltage=voltage, gap=gap_d)
    j_win = j_anode_hist[n_avg_start:]
    j_steady = float(np.mean(j_win))
    mid_idx = len(j_win) // 2
    h1 = j_win[:mid_idx]
    h2 = j_win[mid_idx:]
    j_half1 = float(np.mean(h1))
    j_half2 = float(np.mean(h2))
    se_half = float(
        np.sqrt(
            np.var(h1, ddof=1) / max(len(h1), 1) + np.var(h2, ddof=1) / max(len(h2), 1)
        )
    )
    j_stat_err = float(np.hypot(se_half, abs(j_steady) / max(len(j_win), 1)))
    rel_j_error = float(abs(j_steady - J_CL) / J_CL)

    ez_mean = ez_sum / float(max(n_avg, 1))
    z_nodes = np.linspace(0.0, gap_d, Nz + 1)
    phi_nodes = np.zeros(Nz + 1, dtype=np.float64)
    phi_nodes[1:] = -np.cumsum(ez_mean) * dz

    # Interior nodal space-charge density rho(z_k) = eps_0 * (E_z[k] - E_z[k-1]) / dz
    z_interior = z_nodes[1:-1]
    rho_interior = epsilon_0 * np.diff(ez_mean) / dz
    rho_ref = 4.0 * epsilon_0 * voltage / (9.0 * (gap_d**2))

    # Power-law fits over fixed physical range z / d in [2/12, 11/12]
    z_norm_int = z_interior / gap_d
    phi_norm_int = phi_nodes[1:-1] / voltage
    rho_norm_int = np.abs(rho_interior) / rho_ref
    fit_mask = (
        (z_norm_int >= (2.0 / 12.0) - 1e-12)
        & (z_norm_int <= (11.0 / 12.0) + 1e-12)
        & (phi_norm_int > 0.0)
        & (rho_norm_int > 0.0)
    )
    if np.count_nonzero(fit_mask) >= 2:
        phi_exp = float(
            np.polyfit(np.log(z_norm_int[fit_mask]), np.log(phi_norm_int[fit_mask]), 1)[
                0
            ]
        )
        rho_exp = float(
            np.polyfit(np.log(z_norm_int[fit_mask]), np.log(rho_norm_int[fit_mask]), 1)[
                0
            ]
        )
    else:
        phi_exp = float("nan")
        rho_exp = float("nan")

    res_gauss = float(gauss_residual(solver, electrons))
    wall_s = float(time.perf_counter() - t0)

    # Boxcar-smoothed J_anode trace over ~0.08 transit times for clean plotting
    win = max(round(0.08 * t_transit / dt), 1)
    kernel = np.ones(win, dtype=np.float64) / float(win)
    j_smooth = np.convolve(j_anode_hist, kernel, mode="same")
    stride = max(n_steps // 250, 1)

    print(
        f"[V6 Nz={Nz:2d}] J_steady={j_steady:.2f} A/m^2 "
        f"(half1={j_half1:.2f}, half2={j_half2:.2f}, "
        f"|h1-h2|={abs(j_half1 - j_half2):.2f} <= stat_err={j_stat_err:.2f}), "
        f"J_CL={J_CL:.2f} A/m^2, rel_err={rel_j_error:.5f}, "
        f"phi_exp={phi_exp:.5f}, rho_exp={rho_exp:.5f}, "
        f"W_in={w_in_steady:.6e} J, W_out={w_out_steady:.6e} J "
        f"(dK_anode={dk_anode_steady:.4e}, dK_cath={dk_cathode_steady:.2e}, "
        f"dK_gap={dk_gap_steady:.2e}, dW_f={dw_field_steady:.2e}), "
        f"energy_err={energy_rel_err:.5f}, gauss_res={res_gauss:.3e} ({wall_s:.2f}s)"
    )

    return {
        "Nz": int(Nz),
        "dz_m": float(dz),
        "dt_s": float(dt),
        "n_steps": int(n_steps),
        "t_transit_s": float(t_transit),
        "J_CL_A_m2": float(J_CL),
        "J_steady_A_m2": j_steady,
        "J_half1_A_m2": j_half1,
        "J_half2_A_m2": j_half2,
        "J_stat_err_A_m2": j_stat_err,
        "rel_J_error": rel_j_error,
        "phi_exponent": phi_exp,
        "rho_exponent": rho_exp,
        "W_in_steady_J": float(w_in_steady),
        "W_out_steady_J": float(w_out_steady),
        "dK_anode_steady_J": float(dk_anode_steady),
        "dK_cathode_steady_J": float(dk_cathode_steady),
        "dK_gap_steady_J": float(dk_gap_steady),
        "dW_field_steady_J": float(dw_field_steady),
        "energy_rel_err": float(energy_rel_err),
        "gauss_residual": res_gauss,
        "wall_time_s": wall_s,
        "curves": {
            "t_over_t_transit": (t_hist[::stride] / t_transit).tolist(),
            "J_anode_smooth_A_m2": j_smooth[::stride].tolist(),
            "z_over_d_nodes": (z_nodes / gap_d).tolist(),
            "phi_over_V_nodes": (phi_nodes / voltage).tolist(),
            "z_over_d_interior": z_norm_int.tolist(),
            "abs_rho_over_rho_ref": rho_norm_int.tolist(),
        },
    }


def make_figure_current_vs_time(
    runs: list[dict],
    J_CL: float,
    p_fit: float,
    a_fit: float,
    b_fit: float,
    out_path: Path,
) -> None:
    """
    Create Figure 1: (a) Anode current density vs time with ``J_CL`` marked across
    resolutions, and (b) two-term spatial convergence of the steady current.
    """
    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=3.8,
        nrows=1,
        ncols=2,
        gridspec_kw={"wspace": 0.42},
    )
    ax_time, ax_conv = axes[0], axes[1]

    for idx, r in enumerate(runs):
        st = figstyle.series_style(idx, with_marker=False, lw=1.8)
        c_data = r["curves"]
        ax_time.plot(
            c_data["t_over_t_transit"],
            np.asarray(c_data["J_anode_smooth_A_m2"]) * 1e-3,
            label=(
                rf"$N_z={r['Nz']}$ "
                rf"($|J-J_{{\mathrm{{CL}}}}|/J_{{\mathrm{{CL}}}}={r['rel_J_error']:.3f}$)"
            ),
            **st,
        )

    ax_time.axhline(
        J_CL * 1e-3,
        color=figstyle.INK,
        linestyle="--",
        linewidth=1.4,
        label=rf"Analytical $J_{{\mathrm{{CL}}}} = {J_CL * 1e-3:.2f}\ \mathrm{{kA/m^2}}$",
    )
    ax_time.set_xlabel(r"Normalized time $t / T_{\mathrm{tr}}$")
    ax_time.set_ylabel(r"Anode current density $J_{\mathrm{anode}}$ [kA/m$^2$]")
    ax_time.set_title(r"(a) Anode current")
    # The start-up spike (space-charge front reaching the anode) is ~10x J_CL; clip it so
    # the approach to the steady state stays visible.
    ax_time.set_ylim(0.0, 2.5 * J_CL * 1e-3)
    figstyle.spectrum_grid(ax_time)
    figstyle.place_legend_below(ax_time, ncol=1, y_offset=-0.24)

    nz_arr = np.array([r["Nz"] for r in runs], dtype=np.float64)
    dz_over_d = 1.0 / nz_arr
    err_arr = np.array([r["rel_J_error"] for r in runs], dtype=np.float64)

    st_conv = figstyle.series_style(0, with_marker=True, ms=5.5, lw=1.8)
    ax_conv.loglog(
        dz_over_d,
        err_arr,
        label=rf"Measured $|J - J_{{\mathrm{{CL}}}}| / J_{{\mathrm{{CL}}}}$ (slope $p={p_fit:.2f}$)",
        **st_conv,
    )
    h_dense = np.geomspace(dz_over_d[-1], dz_over_d[0], 100)
    two_term_line = a_fit * (h_dense ** (2.0 / 3.0)) + b_fit * h_dense
    ax_conv.loglog(
        h_dense,
        two_term_line,
        color=figstyle.INK,
        linestyle="--",
        linewidth=1.3,
        label=r"Two-term model $a(\Delta z/d)^{2/3} + b(\Delta z/d)$",
    )
    ax_conv.set_xlabel(r"Normalized cell size $\Delta z / d = 1 / N_z$")
    ax_conv.set_ylabel(r"$|J - J_{\mathrm{CL}}| / J_{\mathrm{CL}}$")
    ax_conv.set_title(r"(b) Convergence in $\Delta z$")
    # Label only the measured resolutions: log minor tick labels collide on a short decade.
    ax_conv.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(dz_over_d))
    ax_conv.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _pos: f"1/{round(1.0 / v)}")
    )
    ax_conv.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    figstyle.spectrum_grid(ax_conv)
    figstyle.place_legend_below(ax_conv, ncol=1, y_offset=-0.24)

    figstyle.save(fig, out_path, auto_legend_below=False)
    plt.close(fig)


def make_figure_steady_profiles(fine_run: dict, out_path: Path) -> None:
    """
    Create Figure 2: (a) Steady electrostatic potential profile ``phi(z) / V`` vs
    analytic ``(z / d)^(4/3)``, and (b) steady space-charge density profile
    ``|rho(z)| / rho_ref`` vs analytic ``(z / d)^(-2/3)``.
    """
    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=3.2,
        nrows=1,
        ncols=2,
        gridspec_kw={"wspace": 0.34},
    )
    ax_phi, ax_rho = axes[0], axes[1]

    c_data = fine_run["curves"]
    z_nodes = np.asarray(c_data["z_over_d_nodes"], dtype=np.float64)
    phi_nodes = np.asarray(c_data["phi_over_V_nodes"], dtype=np.float64)
    z_int = np.asarray(c_data["z_over_d_interior"], dtype=np.float64)
    rho_int = np.asarray(c_data["abs_rho_over_rho_ref"], dtype=np.float64)

    z_cont = np.linspace(0.0, 1.0, 300)
    phi_cl = z_cont ** (4.0 / 3.0)
    z_cont_pos = np.linspace(max(z_int[0] * 0.7, 0.01), 1.0, 300)
    rho_cl = z_cont_pos ** (-2.0 / 3.0)

    ax_phi.plot(
        z_cont,
        phi_cl,
        color=figstyle.INK,
        linestyle="--",
        linewidth=1.5,
        label=r"Analytical Child–Langmuir $(z / d)^{4/3}$",
    )
    st_phi = figstyle.series_style(0, with_marker=True, markevery=2, ms=4.5, lw=1.8)
    ax_phi.plot(
        z_nodes,
        phi_nodes,
        label=(
            rf"Wakis $N_z={fine_run['Nz']}$ "
            rf"(fitted exponent ${fine_run['phi_exponent']:.3f}$)"
        ),
        **st_phi,
    )
    ax_phi.set_xlabel(r"Normalized position $z / d$")
    ax_phi.set_ylabel(r"Normalized potential $\phi(z) / V$")
    ax_phi.set_title(r"(a) Potential")
    figstyle.spectrum_grid(ax_phi)
    figstyle.place_legend_below(ax_phi, ncol=1, y_offset=-0.24)

    ax_rho.plot(
        z_cont_pos,
        rho_cl,
        color=figstyle.INK,
        linestyle="--",
        linewidth=1.5,
        label=r"Analytical Child–Langmuir $(z / d)^{-2/3}$",
    )
    st_rho = figstyle.series_style(1, with_marker=True, markevery=2, ms=4.5, lw=1.8)
    ax_rho.plot(
        z_int,
        rho_int,
        label=(
            rf"Wakis $N_z={fine_run['Nz']}$ "
            rf"(fitted exponent ${fine_run['rho_exponent']:.3f}$)"
        ),
        **st_rho,
    )
    ax_rho.set_xlabel(r"Normalized position $z / d$")
    ax_rho.set_ylabel(r"Normalized density $|\rho(z)| / \rho_{\mathrm{CL}}(d)$")
    ax_rho.set_title(r"(b) Space-charge density")
    figstyle.spectrum_grid(ax_rho)
    figstyle.place_legend_below(ax_rho, ncol=1, y_offset=-0.24)

    figstyle.save(fig, out_path, auto_legend_below=False)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V6 planar Child–Langmuir diode benchmark (wakis.particles)."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast smoke check on coarse resolutions (Nz = 8, 16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    gap_d = 0.005  # [m]
    voltage = 2500.0  # [V]
    J_CL = child_langmuir_current_density(voltage=voltage, gap=gap_d)

    if args.smoke:
        resolutions = (12, 24)
        n_transits = 10.0
    else:
        resolutions = (12, 24, 48)
        n_transits = 10.0

    runs = [
        run_planar_diode_resolution(
            nz, gap_d=gap_d, voltage=voltage, n_transits=n_transits
        )
        for nz in resolutions
    ]

    dz_vals = np.array([r["dz_m"] for r in runs], dtype=np.float64)
    h_vals = dz_vals / gap_d
    err_vals = np.array([r["rel_J_error"] for r in runs], dtype=np.float64)
    p_fit = float(np.polyfit(np.log(dz_vals), np.log(err_vals), 1)[0])
    basis_12_24 = np.array(
        [[h_vals[0] ** (2.0 / 3.0), h_vals[0]], [h_vals[1] ** (2.0 / 3.0), h_vals[1]]],
        dtype=np.float64,
    )
    a_fit_arr, b_fit_arr = np.linalg.solve(basis_12_24, err_vals[:2])
    a_fit = float(a_fit_arr)
    b_fit = float(b_fit_arr)
    J_mid = float(runs[-2]["J_steady_A_m2"])
    J_fine = float(runs[-1]["J_steady_A_m2"])
    J_inf = float(J_fine + (J_fine - J_mid) / (2.0**p_fit - 1.0))
    j_inf_rel_err = float((J_inf - J_CL) / J_CL)
    print(
        f"[V6 summary] fitted convergence order p = {p_fit:.4f} (band [2/3, 1]), "
        f"two-term fit a = {a_fit:.6f}, b = {b_fit:.6f}, "
        f"J_inf = {J_inf:.4f} A/m^2 (J_inf/J_CL - 1 = {j_inf_rel_err:+.6e})"
    )

    # Evaluate reference analytical values for the three emission models
    F_ref = 1.0e8  # [V/m]
    dW_eV = float(schottky_barrier_lowering(F_ref) / q_e)
    J_rd = float(
        richardson_dushman_current_density(
            F_ref, temperature=2000.0, work_function_eV=2.5
        )
    )
    f_scaled = 0.5
    v_f, t_f = forbes_barrier_functions(f_scaled)
    J_fn = float(
        fowler_nordheim_current_density(5.0e9, work_function_eV=4.5, beta_fe=1.0)
    )

    fig1_path = IMG_DIR / "particles_child_langmuir_current_vs_time.png"
    fig2_path = IMG_DIR / "particles_child_langmuir_profiles.png"
    make_figure_current_vs_time(runs, J_CL, p_fit, a_fit, b_fit, fig1_path)
    make_figure_steady_profiles(runs[-1], fig2_path)

    payload = {
        "benchmark": "V6_child_langmuir_planar_diode",
        "smoke": bool(args.smoke),
        "parameters": {
            "gap_d_m": gap_d,
            "voltage_V": voltage,
            "eV_over_mc2": float((q_e * voltage) / (m_e * (c_light**2))),
            "rel_jory_trivelpiece_correction": float(
                -(3.0 / 28.0) * (q_e * voltage) / (m_e * (c_light**2))
            ),
            "J_CL_A_m2": float(J_CL),
            "n_transits": n_transits,
            "resolutions_Nz": list(resolutions),
        },
        "convergence": {
            "fitted_order_p": p_fit,
            "derived_order_band": [float(2.0 / 3.0), 1.0],
            "two_term_a": a_fit,
            "two_term_b": b_fit,
            "J_inf_A_m2": J_inf,
            "J_inf_over_J_CL_minus_1": j_inf_rel_err,
            "finest_Nz": int(resolutions[-1]),
            "finest_rel_J_error": float(runs[-1]["rel_J_error"]),
            "finest_phi_exponent": float(runs[-1]["phi_exponent"]),
            "finest_rho_exponent": float(runs[-1]["rho_exponent"]),
            "finest_gauss_residual": float(runs[-1]["gauss_residual"]),
        },
        "formula_spot_checks": {
            "schottky_dW_eV_at_1e8_V_m": dW_eV,
            "richardson_dushman_J_A_m2_T2000K_W2p5eV_F1e8": J_rd,
            "forbes_at_5e9_V_m_W4p5eV": {
                "f": float(f_scaled),
                "v_f": float(v_f),
                "t_f": float(t_f),
                "J_FN_A_m2": J_fn,
            },
        },
        "runs": runs,
        "figures": [
            str(fig1_path.relative_to(REPO_ROOT)),
            str(fig2_path.relative_to(REPO_ROOT)),
        ],
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved figures to {fig1_path} and {fig2_path}")
    print(f"Saved benchmark metrics to {JSON_PATH}")


if __name__ == "__main__":
    main()
