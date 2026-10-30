# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Example 022 — Two-stream instability (with the gamma0^(-3/2) longitudinal-mass correction) growth rate and phase-space vortex (V4).

Demonstrates and validates the self-consistent electromagnetic particle-in-cell coupler
(:class:`~wakis.particles.ParticleSolver`) on the classic counter-streaming electron beam
(two-stream) instability in a quasi-1D periodic domain:

1. **Dispersion relation and discrete FIT / leapfrog correction**:
   Two equal, cold, counter-streaming electron beams at :math:`\\pm v_0`, each with
   plasma frequency :math:`\\omega_b = \\sqrt{n_b e^2 / (\\varepsilon_0 m_e)}` and
   relativistic longitudinal plasma frequency :math:`\\omega_{b,\\mathrm{rel}} = \\gamma_0^{-3/2}\\omega_b`,
   propagate through an immobile neutralising background (`solver._particles_q_bg`).
   The fundamental box mode :math:`k_1 = 2\\pi / L_z` is placed at the fastest-growing
   wavenumber :math:`k_1 v_0 = (\\sqrt{3}/2)\\,\\omega_{b,\\mathrm{rel}}` where the continuum
   growth rate reaches its maximum :math:`\\gamma_{\\max} = \\omega_{b,\\mathrm{rel}} / 2`,
   while modes with :math:`k v_0 \\ge \\sqrt{2}\\,\\omega_{b,\\mathrm{rel}}` are linearly stable.
   The finite-grid and finite-:math:`\\Delta t` growth rate :math:`\\gamma_{\\mathrm{disc}}(k_1)`
   is computed from the discrete Birdsall-Langdon / FIT dispersion relation (which reduces
   identically to :func:`~wakis.particles.cold_plasma_dispersion_discrete` as :math:`v_0 \\to 0`).
   Note that while this example uses :math:`N_z = 36` (`--smoke`)
   and :math:`N_z = 64` (full), resolving the plasma wavelength with
   :math:`N_z \\sqrt{3} / (2\\sqrt{2}) \\ge 20` cells, ``tests/test_085_particles_two_stream.py``
   deliberately includes a coarser :math:`N_z = 16` grid (below the :math:`\\ge 20` rule in
   ``docs/physicsguide_sections/particles_two_stream.md``) to make the discrete grid shift clearly
   visible above the fit tolerance.

2. **Linear growth, wave-particle trapping saturation, and phase-space vortex**:
   Seeding :math:`k_1` with a quiet-start sinusoidal density perturbation produces clean
   exponential growth over the per-case rule-defined linear window
   :math:`[t_{\\mathrm{start}},\\, t_{\\mathrm{end}}(\\alpha_{\\mathrm{sat}})]`
   (with no window selection by trial runs), followed by wave-particle trapping saturation near
   :math:`E_{\\mathrm{trap}} = \\gamma_0^3 m_e k_1 v_0^2 / (2 e)` where the two beams
   roll up into a phase-space vortex in :math:`(z, u_z)` and subsequently phase-mix.

Outputs
-------
- ``docs/img/particles_two_stream_growth.png``
- ``docs/img/particles_two_stream_phase_space.png``
- ``examples/022_particles_two_stream_instability.json``

Usage
-----
.. code-block:: bash

    python examples/022_particles_two_stream_instability.py --smoke
    python examples/022_particles_two_stream_instability.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_DISP_PATH = REPO_ROOT / "tests" / "_two_stream_dispersion.py"
_DISP_SPEC = importlib.util.spec_from_file_location(
    "_two_stream_dispersion", _DISP_PATH
)
assert _DISP_SPEC is not None and _DISP_SPEC.loader is not None
_disp_mod = importlib.util.module_from_spec(_DISP_SPEC)
sys.modules[_DISP_SPEC.name] = _disp_mod
_DISP_SPEC.loader.exec_module(_disp_mod)

ALPHA_SAT = _disp_mod.ALPHA_SAT
TRANSIENT_TOL = _disp_mod.TRANSIENT_TOL
two_stream_continuum_growth_rate = _disp_mod.two_stream_continuum_growth_rate
two_stream_discrete_growth_rate = _disp_mod.two_stream_discrete_growth_rate
two_stream_lattice_alias_bound = _disp_mod.two_stream_lattice_alias_bound
two_stream_nonlinear_bound = _disp_mod.two_stream_nonlinear_bound
two_stream_window_bounds_wb = _disp_mod.two_stream_window_bounds_wb
from wakis import GridFIT3D, SolverFIT3D, figstyle
from wakis.particles import (
    ParticleSolver,
    Species,
    cold_plasma_dispersion_discrete,
    deposit_charge,
)

IMG_DIR = REPO_ROOT / "docs" / "img"
JSON_PATH = Path(__file__).resolve().with_suffix(".json")


def run_two_stream_evolution(smoke: bool = False) -> dict:
    """Run the unstable mode-seeded two-stream evolution through saturation and late time, plus a stable mode check.

    Uses the per-case window bounds from ``two_stream_window_bounds_wb`` shared with
    ``tests/test_085_particles_two_stream.py`` (no window selection by trial runs).

    Parameters
    ----------
    smoke : bool, optional
        If True, use lightweight grid/particle resolution suitable for quick CI/smoke
        execution while still resolving the plasma wavelength with ``>= 20`` cells.

    Returns
    -------
    dict
        Simulation time histories, phase-space snapshots, dispersion curves, and
        validation metrics.
    """
    t0_wall = time.perf_counter()

    # Physical parameters: v0 = 0.06c (<= 0.1c so relativistic shift 1 - gamma0^(-3/2) = O((v0/c)^2))
    v0_over_c = 0.06
    v0 = v0_over_c * c
    gamma0 = 1.0 / float(np.sqrt(1.0 - v0_over_c**2))
    u0 = gamma0 * v0
    omega_b = 2.0 * np.pi * 1.0e9
    wb_rel = omega_b * (gamma0 ** (-1.5))

    # Choose box length Lz so mode k1 = 2*pi/Lz sits at maximum growth k1 * v0 = (sqrt(3)/2) * wb_rel.
    # With Nz = 36 (smoke) or 64 (full), the two-beam plasma wavelength 2*pi*u0 / (sqrt(2)*omega_b)
    # is resolved with Nz * sqrt(3) / (2 * sqrt(2)) >= 20 cells.
    Nz = 36 if smoke else 64
    n_ppc_z = 24 if smoke else 64
    Nx, Ny = (2, 2) if smoke else (4, 4)

    k1 = (np.sqrt(3.0) / 2.0) * wb_rel / v0
    Lz = 2.0 * np.pi / k1
    dz = Lz / float(Nz)
    Lx, Ly = Nx * dz * 6.0, Ny * dz * 6.0

    grid = GridFIT3D(0.0, Lx, 0.0, Ly, 0.0, Lz, Nx, Ny, Nz, verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=0.8,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        verbose=0,
    )
    dt = float(solver.dt)

    nb = epsilon_0 * m_e * (omega_b**2) / (e**2)
    vol = Lx * Ly * Lz
    n_z = int(Nz * n_ppc_z)
    z0 = (np.arange(n_z, dtype=np.float64) + 0.5) * (Lz / n_z)

    # Seed mode k1 with sinusoidal density perturbation eps = 1e-7 so that at t_start_wb
    # the amplitude is well below ALPHA_SAT * E_trap = 1e-2 * E_trap, crosses ALPHA_SAT * E_trap
    # inside the horizon, and subsequently reaches wave-particle trapping saturation and late-time
    # phase mixing.
    eps = 1.0e-7
    z_plus = (z0 - (eps / k1) * np.sin(k1 * z0 + np.pi / 3.0)) % Lz
    z_minus = (z0 - (eps / k1) * np.sin(k1 * z0 - np.pi / 3.0)) % Lz

    dx, dy = Lx / Nx, Ly / Ny
    if smoke:
        tx = np.array([0.5 * dx], dtype=np.float64)
        ty = np.array([0.5 * dy], dtype=np.float64)
    else:
        tx = np.array([0.5 * dx, 2.5 * dx, 0.5 * dx, 2.5 * dx], dtype=np.float64)
        ty = np.array([0.5 * dy, 0.5 * dy, 2.5 * dy, 2.5 * dy], dtype=np.float64)
    n_trans = int(tx.size)
    x_all = np.repeat(tx, n_z)
    y_all = np.repeat(ty, n_z)
    zp_all = np.tile(z_plus, n_trans)
    zm_all = np.tile(z_minus, n_trans)
    w_macro = (nb * vol) / float(zp_all.size)

    sp_plus = Species(name="beam_plus", charge=-e, mass=m_e)
    sp_minus = Species(name="beam_minus", charge=-e, mass=m_e)
    sp_plus.add_particles(
        x=x_all, y=y_all, z=zp_all, vx=0.0, vy=0.0, vz=v0, w=w_macro
    )
    sp_minus.add_particles(
        x=x_all, y=y_all, z=zm_all, vx=0.0, vy=0.0, vz=-v0, w=w_macro
    )

    q_nodes = np.zeros((Nx, Ny, Nz), dtype=np.float64, order="F")
    for sp in (sp_plus, sp_minus):
        deposit_charge(
            sp.x / dx,
            sp.y / dy,
            sp.z / dz,
            sp.charge * sp.w,
            Nx,
            Ny,
            Nz,
            order=1,
            periodic=(True, True, True),
            q_nodes=q_nodes,
        )
    Az = dx * dy
    q_bg_node = 2.0 * e * nb * Az * dz
    q_net = q_nodes + q_bg_node
    Ez_3d = np.cumsum(q_net, axis=2) / (epsilon_0 * Az)
    Ez_3d -= np.mean(Ez_3d, axis=2, keepdims=True)
    solver.E[:, :, :, "z"] = Ez_3d

    ps = ParticleSolver(
        solver,
        species=[sp_plus, sp_minus],
        pusher="boris",
        shape=1,
        gather="galerkin",
        sort_every=0,
    ).attach()
    # gauss_init is zero by construction (an identity from q_bg = div_d0 - q0 in attach());
    # gauss_final after time-stepping is the real conservation check.
    gauss_init = ps.gauss_residual()

    z_edges = (np.arange(Nz, dtype=np.float64) + 0.5) * dz
    exp_k1 = np.exp(-1j * k1 * z_edges)
    Ez_init_1d = np.mean(solver.E[:, :, :, "z"], axis=(0, 1))
    amp_k1_init = float((2.0 / Nz) * np.abs(np.sum(Ez_init_1d * exp_k1)))

    gamma_cont = two_stream_continuum_growth_rate(omega_b, k1, v0, gamma0=gamma0)
    gamma_cont_nr = two_stream_continuum_growth_rate(omega_b, k1, v0, gamma0=1.0)
    gamma_disc = two_stream_discrete_growth_rate(
        omega_b,
        k1,
        v0,
        dt,
        dz,
        order=1,
        gather="galerkin",
        gamma0=gamma0,
    )
    win_bounds = two_stream_window_bounds_wb(
        omega_b,
        gamma_disc,
        gamma0=gamma0,
        eps=eps,
        alpha_sat=ALPHA_SAT,
        transient_tol=TRANSIENT_TOL,
    )
    t_start_wb = float(win_bounds["t_start_wb"])
    t_end_pred_wb = float(win_bounds["t_end_pred_wb"])
    a_grow = float(win_bounds["a_grow"])
    a_osc = float(win_bounds["a_osc"])

    # Analytical trapping time prediction omega_b * t_trap = (omega_b / gamma_disc) * ln(3 / (8 * eps * a_grow))
    # (gamma0^3 cancels in E_0 / E_trap = (8 / 3) * eps since k1 * v0 = (sqrt(3)/2) * wb_rel).
    t_trap_wb_pred = (omega_b / gamma_disc) * float(
        np.log(3.0 / (8.0 * eps * a_grow))
    )
    t_max_wb = 1.5 * t_trap_wb_pred
    assert t_max_wb > t_end_pred_wb
    n_steps = int(t_max_wb / (omega_b * dt))
    t_wb_arr = np.zeros(n_steps, dtype=np.float64)
    amp_k1_arr = np.zeros(n_steps, dtype=np.float64)

    snap_targets_wb = {
        "linear": 0.5 * (t_start_wb + t_end_pred_wb),
        "saturation": t_trap_wb_pred + 2.5,
        "late": t_max_wb,
    }
    snap_steps = {
        name: min(max(int(round(twb / (omega_b * dt))) - 1, 0), n_steps - 1)  # noqa: RUF046 - numpy floats
        for name, twb in snap_targets_wb.items()
    }
    step_to_snap = {step_idx: name for name, step_idx in snap_steps.items()}
    snapshots: dict[str, dict[str, list[float] | float]] = {}

    for s in range(n_steps):
        solver.one_step()
        t_cur = (s + 1) * dt
        t_wb_arr[s] = t_cur * omega_b
        Ez_1d = np.mean(solver.E[:, :, :, "z"], axis=(0, 1))
        amp_k1_arr[s] = (2.0 / Nz) * np.abs(np.sum(Ez_1d * exp_k1))

        if s in step_to_snap:
            s_name = step_to_snap[s]
            # Record the first transverse sub-lattice slice (n_z particles per beam)
            snapshots[s_name] = {
                "t_wb": float(t_wb_arr[s]),
                "z_plus_norm": (sp_plus.z[:n_z] / Lz).tolist(),
                "uz_plus_norm": (sp_plus.uz[:n_z] / u0).tolist(),
                "z_minus_norm": (sp_minus.z[:n_z] / Lz).tolist(),
                "uz_minus_norm": (sp_minus.uz[:n_z] / u0).tolist(),
            }

    gauss_final = ps.gauss_residual()

    # Wave-particle trapping saturation amplitude E_trap = gamma0^3 * m_e * k1 * v0^2 / (2 * e)
    e_trap = 0.5 * (gamma0**3) * m_e * k1 * (v0**2) / e
    after_start = t_wb_arr >= t_start_wb
    exceed_idx = np.flatnonzero(after_start & (amp_k1_arr > ALPHA_SAT * e_trap))
    if exceed_idx.size > 0:
        t_end_idx = int(exceed_idx[0])
        window_end_rule = "alpha_sat"
    else:
        t_end_idx = n_steps
        window_end_rule = "horizon"
    win_mask = after_start & (np.arange(n_steps) < t_end_idx)
    if np.count_nonzero(win_mask) < 5 or not np.all(amp_k1_arr[win_mask] > 0.0):
        raise AssertionError("no linear window")
    t_win_s = t_wb_arr[win_mask] / omega_b
    coeffs, cov = np.polyfit(t_win_s, np.log(amp_k1_arr[win_mask]), 1, cov=True)
    gamma_fit = float(coeffs[0])
    sigma_fit = float(np.sqrt(cov[0, 0]))
    ln_a0_fit = float(coeffs[1])

    # Tolerance components:
    # 1. 3 * sigma_fit: empirical (single-run OLS SE);
    # 2. delta_gamma_nl: order-of-magnitude quadratic (3/16)*|gamma_disc|*(|E_1|/E_trap)^2 bound;
    # 3. delta_gamma_lat: heuristic particle-lattice alias bound.
    amp_end = float(amp_k1_arr[win_mask][-1])
    delta_gamma_nl = two_stream_nonlinear_bound(gamma_disc, amp_end, e_trap)
    delta_gamma_lat = two_stream_lattice_alias_bound(
        gamma_disc, n_particles_beam=n_z, order=1, gather="galerkin"
    )
    tol_gamma = 3.0 * sigma_fit + delta_gamma_nl + delta_gamma_lat

    # Also run a stable-mode companion check at k2 = 2 * k1 (k2 * v0 = sqrt(3) * wb_rel > sqrt(2) * wb_rel)
    # on a periodic box of length Lz_s = 2*pi/k2 = Lz / 2 (Nz_s = Nz // 2) with identical dz and dt.
    Nz_s = Nz // 2
    Lz_s = float(Nz_s) * dz
    n_z_s = int(Nz_s * n_ppc_z)
    z0_s = (np.arange(n_z_s, dtype=np.float64) + 0.5) * (Lz_s / n_z_s)
    grid_s = GridFIT3D(0.0, Lx, 0.0, Ly, 0.0, Lz_s, Nx, Ny, Nz_s, verbose=0)
    solver_s = SolverFIT3D(
        grid_s,
        cfln=0.8,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        verbose=0,
    )
    k2 = 2.0 * k1
    x_all_s = np.repeat(tx, n_z_s)
    y_all_s = np.repeat(ty, n_z_s)
    zp_s = np.tile(
        (z0_s - (eps / k2) * np.sin(k2 * z0_s + np.pi / 3.0)) % Lz_s, n_trans
    )
    zm_s = np.tile(
        (z0_s - (eps / k2) * np.sin(k2 * z0_s - np.pi / 3.0)) % Lz_s, n_trans
    )
    w_macro_s = (nb * (Lx * Ly * Lz_s)) / float(zp_s.size)
    sp_p_s = Species(name="beam_plus_s", charge=-e, mass=m_e)
    sp_m_s = Species(name="beam_minus_s", charge=-e, mass=m_e)
    sp_p_s.add_particles(
        x=x_all_s, y=y_all_s, z=zp_s, vx=0.0, vy=0.0, vz=v0, w=w_macro_s
    )
    sp_m_s.add_particles(
        x=x_all_s, y=y_all_s, z=zm_s, vx=0.0, vy=0.0, vz=-v0, w=w_macro_s
    )

    q_nodes_s = np.zeros((Nx, Ny, Nz_s), dtype=np.float64, order="F")
    for sp in (sp_p_s, sp_m_s):
        deposit_charge(
            sp.x / dx,
            sp.y / dy,
            sp.z / dz,
            sp.charge * sp.w,
            Nx,
            Ny,
            Nz_s,
            order=1,
            periodic=(True, True, True),
            q_nodes=q_nodes_s,
        )
    Ez_s = np.cumsum(q_nodes_s + q_bg_node, axis=2) / (epsilon_0 * Az)
    Ez_s -= np.mean(Ez_s, axis=2, keepdims=True)
    solver_s.E[:, :, :, "z"] = Ez_s
    ParticleSolver(
        solver_s,
        species=[sp_p_s, sp_m_s],
        pusher="boris",
        shape=1,
        gather="galerkin",
        sort_every=0,
    ).attach()

    z_edges_s = (np.arange(Nz_s, dtype=np.float64) + 0.5) * dz
    exp_k2 = np.exp(-1j * k2 * z_edges_s)
    amp_k2_stable = np.zeros(n_steps, dtype=np.float64)
    for s in range(n_steps):
        solver_s.one_step()
        Ez_1d_s = np.mean(solver_s.E[:, :, :, "z"], axis=(0, 1))
        amp_k2_stable[s] = (2.0 / Nz_s) * np.abs(np.sum(Ez_1d_s * exp_k2))

    # Analytical vs discrete dispersion curve over kv0 / wb_rel in [0, 1.6]
    kv_norm_grid = np.linspace(0.0, 1.60, 81)
    gamma_cont_curve = np.zeros_like(kv_norm_grid)
    gamma_disc_curve = np.zeros_like(kv_norm_grid)
    for idx, kv_n in enumerate(kv_norm_grid):
        k_val = float(kv_n) * wb_rel / v0
        gamma_cont_curve[idx] = (
            two_stream_continuum_growth_rate(omega_b, k_val, v0, gamma0=gamma0)
            / wb_rel
        )
        if k_val > 0.0:
            gamma_disc_curve[idx] = (
                two_stream_discrete_growth_rate(
                    omega_b,
                    k_val,
                    v0,
                    dt,
                    dz,
                    order=1,
                    gather="galerkin",
                    gamma0=gamma0,
                )
                / wb_rel
            )

    # Verify stationary limit equivalence with cold_plasma_dispersion_discrete
    wp_disc_v0_zero = cold_plasma_dispersion_discrete(
        np.sqrt(2.0) * omega_b, dt, dz, k1, order=1, gather="galerkin"
    )

    peak_idx = int(np.argmax(amp_k1_arr))
    elapsed_s = time.perf_counter() - t0_wall
    return {
        "config": {
            "smoke": bool(smoke),
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "n_ppc_z": n_ppc_z,
            "v0_over_c": v0_over_c,
            "gamma0": gamma0,
            "relativistic_wb_scaling": float(gamma0 ** (-1.5)),
            "relativistic_neglect_fraction": float(1.0 - gamma0 ** (-1.5)),
            "omega_b_rad_s": omega_b,
            "wb_rel_rad_s": wb_rel,
            "k1_rad_m": k1,
            "Lz_m": Lz,
            "dz_m": dz,
            "dt_s": dt,
            "eps_seed": eps,
            "t_start_wb": t_start_wb,
            "t_end_pred_wb": t_end_pred_wb,
            "t_horizon_wb": t_max_wb,
            "a_grow": a_grow,
            "a_osc": a_osc,
            "alpha_sat": ALPHA_SAT,
            "cells_per_plasma_wavelength": float(
                Nz * np.sqrt(3.0) / (2.0 * np.sqrt(2.0))
            ),
            "stationary_cold_plasma_omega_rad_s": wp_disc_v0_zero,
        },
        "metrics": {
            "gamma_fit_over_wb": gamma_fit / omega_b,
            "gamma_disc_over_wb": gamma_disc / omega_b,
            "gamma_cont_over_wb": gamma_cont / omega_b,
            "gamma_cont_nonrel_over_wb": gamma_cont_nr / omega_b,
            "abs_err_vs_disc_over_wb": abs(gamma_fit - gamma_disc) / omega_b,
            "abs_err_vs_cont_over_wb": abs(gamma_fit - gamma_cont) / omega_b,
            "sigma_fit_over_wb": sigma_fit / omega_b,
            "delta_gamma_nl_over_wb": delta_gamma_nl / omega_b,
            "delta_gamma_lat_over_wb": delta_gamma_lat / omega_b,
            "tol_gamma_over_wb": tol_gamma / omega_b,
            "e_trap_V_m": e_trap,
            "peak_amp_over_e_trap": float(amp_k1_arr[peak_idx] / e_trap),
            "t_peak_wb": float(t_wb_arr[peak_idx]),
            "stable_k2_max_over_init": float(
                np.max(amp_k2_stable) / amp_k2_stable[0]
            ),
            "gauss_residual_init": gauss_init,
            "gauss_residual_final": gauss_final,
            "t_win_start_wb": float(t_wb_arr[win_mask][0]),
            "t_win_end_wb": float(t_wb_arr[win_mask][-1]),
            "t_end_pred_wb": t_end_pred_wb,
            "t_horizon_wb": t_max_wb,
            "window_end_rule": window_end_rule,
            "ln_a0_fit": ln_a0_fit,
            "amp_k1_init_V_m": amp_k1_init,
            "wall_time_s": elapsed_s,
        },
        "curves": {
            "t_wb": t_wb_arr.tolist(),
            "amp_k1_over_etrap": (amp_k1_arr / e_trap).tolist(),
            "amp_k2_stable_over_etrap": (amp_k2_stable / e_trap).tolist(),
            "kv_over_wb": kv_norm_grid.tolist(),
            "gamma_cont_over_wb_curve": gamma_cont_curve.tolist(),
            "gamma_disc_over_wb_curve": gamma_disc_curve.tolist(),
        },
        "snapshots": snapshots,
    }


def make_figure_growth(data: dict, out_path: Path) -> None:
    """Create Figure 1: Mode amplitude growth on log scale with predicted slope and dispersion curve."""
    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=3.3,
        nrows=1,
        ncols=2,
        gridspec_kw={"wspace": 0.34},
    )
    ax_amp, ax_disp = axes[0], axes[1]

    c_dat = data["curves"]
    m_dat = data["metrics"]
    cfg = data["config"]

    t_wb = np.asarray(c_dat["t_wb"], dtype=np.float64)
    amp_k1_norm = np.asarray(c_dat["amp_k1_over_etrap"], dtype=np.float64)
    amp_k2_norm = np.asarray(c_dat["amp_k2_stable_over_etrap"], dtype=np.float64)

    # Panel (a): Exponential mode growth |E_{z,1}(t)| / E_trap vs omega_b * t
    t_win_lo = float(m_dat["t_win_start_wb"])
    t_win_hi = float(m_dat["t_win_end_wb"])
    ax_amp.axvspan(
        t_win_lo,
        t_win_hi,
        color="#e5e7eb",
        alpha=0.65,
        label="Linear fit window",
    )

    st_sim = figstyle.series_style(0, with_marker=False, lw=2.0)
    ax_amp.semilogy(
        t_wb,
        amp_k1_norm,
        label=r"Unstable mode $k_1$ ($k_1 v_0 = \frac{\sqrt{3}}{2}\omega_b$)",
        **st_sim,
    )

    # Predicted discrete and continuum exponential slopes anchored at the window midpoint
    t_line_wb = np.linspace(t_win_lo - 2.0, t_win_hi + 3.5, 60)
    t_line_s = t_line_wb / float(cfg["omega_b_rad_s"])
    t_mid_s = 0.5 * (t_win_lo + t_win_hi) / float(cfg["omega_b_rad_s"])
    amp_mid_norm = np.exp(
        float(m_dat["ln_a0_fit"])
        + float(m_dat["gamma_fit_over_wb"]) * float(cfg["omega_b_rad_s"]) * t_mid_s
    ) / float(m_dat["e_trap_V_m"])
    g_disc_s = float(m_dat["gamma_disc_over_wb"]) * float(cfg["omega_b_rad_s"])
    g_cont_s = float(m_dat["gamma_cont_over_wb"]) * float(cfg["omega_b_rad_s"])

    ax_amp.semilogy(
        t_line_wb,
        amp_mid_norm * np.exp(g_disc_s * (t_line_s - t_mid_s)),
        color=figstyle.INK,
        linestyle="--",
        linewidth=1.6,
        label=rf"Discrete slope $\gamma_{{\mathrm{{disc}}}}/\omega_b = {m_dat['gamma_disc_over_wb']:.4f}$",
    )
    st_cont = figstyle.series_style(2, with_marker=False, lw=1.3)
    st_cont["linestyle"] = ":"
    ax_amp.semilogy(
        t_line_wb,
        amp_mid_norm * np.exp(g_cont_s * (t_line_s - t_mid_s)),
        label=rf"Continuum slope $\gamma_{{\mathrm{{cont}}}}/\omega_b = {m_dat['gamma_cont_over_wb']:.4f}$",
        **st_cont,
    )

    st_stab = figstyle.series_style(1, with_marker=False, lw=1.5)
    ax_amp.semilogy(
        t_wb,
        amp_k2_norm,
        label=r"Stable mode $k_2 = 2 k_1$ ($k_2 v_0 = \sqrt{3}\omega_b$)",
        **st_stab,
    )
    ax_amp.axhline(
        1.0,
        color=figstyle.INK,
        linestyle="-.",
        linewidth=1.0,
        label=r"Trapping level $E_{\mathrm{trap}}$",
    )

    ax_amp.set_xlabel(r"Dimensionless time $\omega_b t$")
    ax_amp.set_ylabel(r"Normalized Fourier mode $|E_{z,m}(t)| / E_{\mathrm{trap}}$")
    ax_amp.set_title("(a) Mode growth & saturation")
    ax_amp.set_xlim(0.0, float(t_wb[-1]))
    ax_amp.set_ylim(1e-8, 5.0)
    figstyle.spectrum_grid(ax_amp)
    figstyle.place_legend_below(ax_amp, ncol=1, y_offset=-0.24)

    # Panel (b): Continuum vs discrete growth rate curve gamma(k) / omega_{b,rel}
    kv_arr = np.asarray(c_dat["kv_over_wb"], dtype=np.float64)
    g_cont_arr = np.asarray(c_dat["gamma_cont_over_wb_curve"], dtype=np.float64)
    g_disc_arr = np.asarray(c_dat["gamma_disc_over_wb_curve"], dtype=np.float64)

    ax_disp.plot(
        kv_arr,
        g_cont_arr,
        color=figstyle.INK,
        linestyle="-",
        linewidth=1.6,
        label=r"Continuum $\gamma(k)/\omega_{b,\mathrm{rel}}$",
    )
    st_dcurve = figstyle.series_style(0, with_marker=False, lw=1.8)
    st_dcurve["linestyle"] = "--"
    ax_disp.plot(
        kv_arr,
        g_disc_arr,
        label=rf"Discrete FIT ($N_z={cfg['Nz']}$, $p=1$)",
        **st_dcurve,
    )

    wb_ratio = float(cfg["omega_b_rad_s"]) / float(cfg["wb_rel_rad_s"])
    st_pt1 = figstyle.series_style(1, with_marker=True, ms=6.5)
    st_pt1["linestyle"] = "None"
    ax_disp.plot(
        [0.5 * np.sqrt(3.0)],
        [float(m_dat["gamma_fit_over_wb"]) * wb_ratio],
        label=r"Fitted unstable mode $k_1$",
        **st_pt1,
    )
    st_pt2 = figstyle.series_style(3, with_marker=True, ms=6.5)
    st_pt2["linestyle"] = "None"
    ax_disp.plot(
        [min(np.sqrt(3.0), 1.55)],
        [0.0],
        label=r"Stable mode ($k v_0 > \sqrt{2}\omega_b$)",
        **st_pt2,
    )
    ax_disp.axvline(np.sqrt(2.0), color=figstyle.INK, linestyle=":", linewidth=1.0)

    ax_disp.set_xlabel(r"Normalized wavenumber $k v_0 / \omega_{b,\mathrm{rel}}$")
    ax_disp.set_ylabel(r"Normalized growth rate $\gamma / \omega_{b,\mathrm{rel}}$")
    ax_disp.set_title("(b) Two-stream dispersion")
    ax_disp.set_xlim(0.0, 1.60)
    ax_disp.set_ylim(-0.03, 0.56)
    figstyle.spectrum_grid(ax_disp)
    figstyle.place_legend_below(ax_disp, ncol=1, y_offset=-0.24)

    figstyle.save(fig, out_path, auto_legend_below=False)
    matplotlib.pyplot.close(fig)


def make_figure_phase_space(data: dict, out_path: Path) -> None:
    """Create Figure 2: Longitudinal phase-space (z, u_z) snapshots at linear, saturation, and late times."""
    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=3.1,
        nrows=1,
        ncols=3,
        gridspec_kw={"wspace": 0.28},
    )
    snaps = data["snapshots"]
    stages = [
        ("linear", "(a) Linear"),
        ("saturation", "(b) Vortex"),
        ("late", "(c) Late"),
    ]

    st_p = figstyle.series_style(0, with_marker=True, ms=1.8)
    st_m = figstyle.series_style(1, with_marker=True, ms=1.8)

    for idx, (key, title_prefix) in enumerate(stages):
        ax = axes[idx]
        s_dat = snaps[key]
        t_wb = float(s_dat["t_wb"])
        zp = np.asarray(s_dat["z_plus_norm"], dtype=np.float64)
        uzp = np.asarray(s_dat["uz_plus_norm"], dtype=np.float64)
        zm = np.asarray(s_dat["z_minus_norm"], dtype=np.float64)
        uzm = np.asarray(s_dat["uz_minus_norm"], dtype=np.float64)

        ax.plot(
            zp,
            uzp,
            linestyle="None",
            marker="o",
            markersize=1.6,
            color=st_p["color"],
            alpha=0.80,
            label=r"Beam $+v_0$",
        )
        ax.plot(
            zm,
            uzm,
            linestyle="None",
            marker="o",
            markersize=1.6,
            color=st_m["color"],
            alpha=0.80,
            label=r"Beam $-v_0$",
        )
        ax.set_xlabel(r"Position $z / L_z$")
        if idx == 0:
            ax.set_ylabel(r"Proper velocity $u_z / u_0$")
        ax.set_title(rf"{title_prefix}" "\n" rf"$\omega_b t={t_wb:.1f}$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-2.4, 2.4)
        figstyle.spectrum_grid(ax)
        if idx == 1:
            figstyle.place_legend_below(ax, ncol=2, y_offset=-0.26)

    figstyle.save(fig, out_path, auto_legend_below=False)
    matplotlib.pyplot.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Wakis V4 two-stream instability growth and phase-space benchmark."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a lightweight smoke configuration for fast verification.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the two-stream benchmark, generate figures, and write JSON metrics."""
    args = parse_args()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    result = run_two_stream_evolution(smoke=args.smoke)
    fig1_path = IMG_DIR / "particles_two_stream_growth.png"
    fig2_path = IMG_DIR / "particles_two_stream_phase_space.png"
    make_figure_growth(result, fig1_path)
    make_figure_phase_space(result, fig2_path)

    payload = {
        "config": result["config"],
        "metrics": result["metrics"],
        "figures": [
            str(fig1_path.relative_to(REPO_ROOT)),
            str(fig2_path.relative_to(REPO_ROOT)),
        ],
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    m = result["metrics"]
    print(
        f"[Example 022 ({'smoke' if args.smoke else 'full'})] "
        f"win=[{m['t_win_start_wb']:.2f}, {m['t_win_end_wb']:.2f}] "
        f"(pred_end={m['t_end_pred_wb']:.2f}, horizon={m['t_horizon_wb']:.2f}, end_rule={m['window_end_rule']}), "
        f"gamma_fit/wb={m['gamma_fit_over_wb']:.6f}, "
        f"gamma_disc/wb={m['gamma_disc_over_wb']:.6f}, "
        f"gamma_cont/wb={m['gamma_cont_over_wb']:.6f}, "
        f"|fit-disc|/wb={m['abs_err_vs_disc_over_wb']:.6f} <= tol/wb={m['tol_gamma_over_wb']:.6f}, "
        f"peak_E1/E_trap={m['peak_amp_over_e_trap']:.3f} at t*wb={m['t_peak_wb']:.2f}, "
        f"gauss={m['gauss_residual_final']:.2e}, "
        f"wall={m['wall_time_s']:.2f}s"
    )
    print(f"Saved figures to {fig1_path} and {fig2_path}")
    print(f"Saved benchmark metrics to {JSON_PATH}")


if __name__ == "__main__":
    main()
