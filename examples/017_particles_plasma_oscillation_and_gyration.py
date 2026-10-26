# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Example 017 — Relativistic Particle Pushers, Exact Charge Conservation, and Cold Plasma Oscillations
====================================================================================================

Demonstrates milestone M1 of `wakis.particles` (`Species`, `LogicalGridMap`,
`push_boris`, `push_vay`, `push_higuera_cary`, `gather_fields`,
`deposit_current_esirkepov`, `ParticleBoundaries`, `Collector`, `BeamInjector`,
`ParticleSolver`, `gauss_residual`, and `cold_plasma_dispersion_discrete`):

1. **V1 — Single-particle relativistic pushers**:
   - Cyclotron gyration at ``gamma = 5.0`` in uniform ``B_z = 0.10 T`` for
     Boris (1970), Vay (2008), and Higuera–Cary (2017) against the analytical
     Larmor radius ``r_L = gamma * m_e * v_perp / (e * B_z)`` and relativistic
     gyrofrequency ``omega_c = e * B_z / (gamma * m_e)``.
   - Relativistic ``E x B`` drift across ``gamma_0 in {1.1, 2, 5, 10, 20}``,
     illustrating the spurious Boris drift scaling as ``gamma^2 (omega_{c0} dt)^2``
     versus the machine-precision cancellation of Vay and Higuera–Cary.
   - Relativistic hyperbolic motion in uniform ``E_z`` and second-order
     ``O(dt^2)`` global trajectory convergence in combined ``(E, B)`` fields.
2. **V2 — Exact discrete Gauss's law conservation on uniform and graded meshes**:
   - Evolves ``1024`` random relativistic electrons crossing cells in all three
     directions for ``1000`` steps on both uniform and smoothly graded 3D
     grids with ``p = 1`` (CIC) and ``p = 2`` (TSC) Esirkepov current
     deposition, maintaining ``gauss_residual <= 1e-13`` relative, compared
     against the ``O(1)`` Gauss violation of a naive ``J = q * v`` deposit.
3. **V3 — Cold plasma oscillation in a 3D periodic box (`32 cells/lambda`)**:
   - Simulates a longitudinal cold electron plasma oscillation
     (``omega_{p0} / 2*pi = 1.2 GHz``) for ``p = 1`` and ``p = 2``, comparing
     the measured frequency against the closed-form Birdsall–Langdon
     finite-grid, finite-``dt`` dispersion relation and verifying reversible
     kinetic ``<->`` electric field energy exchange.

Saves two publication-style figures in ``docs/img/`` (via `wakis.figstyle`) and
writes all quoted benchmark numbers to
``examples/017_particles_plasma_oscillation_and_gyration.json``:
  1. ``docs/img/particles_pushers_and_charge_conservation.png``
  2. ``docs/img/particles_cold_plasma_oscillation.png``
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import m_e
from scipy.optimize import least_squares

import wakis
from wakis import GridFIT3D, SolverFIT3D, figstyle
from wakis.estimate import estimate, preflight
from wakis.particles import (
    BeamInjector,
    Collector,
    ParticleSolver,
    Species,
    cold_plasma_dispersion_discrete,
    integrate_trajectory,
    push_boris,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / "docs" / "img"
JSON_PATH = Path(__file__).resolve().with_suffix(".json")


def run_v1_benchmarks() -> dict:
    """Run V1 single-particle relativistic pusher benchmarks."""
    # 1. Gyration at gamma = 5.0 in B_z = 0.10 T
    B0 = 0.10
    gamma_gyr = 5.0
    beta_gyr = float(np.sqrt(1.0 - 1.0 / (gamma_gyr * gamma_gyr)))
    v_perp = beta_gyr * c_light
    u_perp = gamma_gyr * v_perp
    omega_c = (e_charge * B0) / (gamma_gyr * m_e)
    r_larmor = (gamma_gyr * m_e * v_perp) / (e_charge * B0)
    T_gyr = 2.0 * np.pi / omega_c
    n_steps_per_period = 120
    n_periods = 3
    dt_gyr = T_gyr / n_steps_per_period
    n_tot_gyr = n_steps_per_period * n_periods

    t_gyr = np.arange(n_tot_gyr + 1) * dt_gyr
    x_exact = r_larmor * np.sin(omega_c * t_gyr)
    y_exact = r_larmor * (1.0 - np.cos(omega_c * t_gyr))

    gyr_results = {}
    gyr_trajectories = {"x_exact": x_exact, "y_exact": y_exact, "r_larmor": r_larmor}

    for pusher in ("boris", "vay", "hc"):
        _, x_h, _, _ = integrate_trajectory(
            x0=[0.0, 0.0, 0.0],
            u0=[u_perp, 0.0, 0.0],
            dt=dt_gyr,
            n_steps=n_tot_gyr,
            E_fn=(0.0, 0.0, 0.0),
            B_fn=(0.0, 0.0, B0),
            q=-e_charge,
            m=m_e,
            pusher=pusher,
        )
        xk, yk = x_h[:, 0], x_h[:, 1]
        xc = 0.5 * (np.max(xk) + np.min(xk))
        yc = 0.5 * (np.max(yk) + np.min(yk))
        r_num = float(np.mean(np.hypot(xk - xc, yk - yc)))
        rel_r_err = abs(r_num - r_larmor) / r_larmor

        phi_unwrap = np.unwrap(np.arctan2(xk - xc, yk - yc))
        omega_num = float(abs((phi_unwrap[-1] - phi_unwrap[0]) / (n_tot_gyr * dt_gyr)))
        rel_omega_err = abs(omega_num - omega_c) / omega_c

        gyr_results[pusher] = {
            "r_larmor_m": float(r_larmor),
            "r_num_m": r_num,
            "rel_r_error": float(rel_r_err),
            "omega_c_rad_s": float(omega_c),
            "omega_num_rad_s": omega_num,
            "rel_omega_error": float(rel_omega_err),
        }
        gyr_trajectories[pusher] = (xk, yk)

    # 2. Relativistic E x B drift sweep over gamma_d
    B_exb = 1.0
    omega_c0 = (e_charge * B_exb) / m_e
    theta_step = 0.4
    n_steps_exb = 200

    gammas_exb = [1.1, 2.0, 5.0, 10.0, 20.0]
    exb_errors = {p: [] for p in ("boris", "vay", "hc")}
    for gamma_d in gammas_exb:
        beta_d = float(np.sqrt(1.0 - 1.0 / (gamma_d * gamma_d)))
        v_d = beta_d * c_light
        Ey_exb = v_d * B_exb
        dt_exb = theta_step * gamma_d / omega_c0
        x_drift_exact = v_d * (n_steps_exb * dt_exb)
        u0_exb = [gamma_d * v_d, 0.0, 0.0]
        for pusher in ("boris", "vay", "hc"):
            _, x_h, _, _ = integrate_trajectory(
                x0=[0.0, 0.0, 0.0],
                u0=u0_exb,
                dt=dt_exb,
                n_steps=n_steps_exb,
                E_fn=(0.0, Ey_exb, 0.0),
                B_fn=(0.0, 0.0, B_exb),
                q=-e_charge,
                m=m_e,
                pusher=pusher,
            )
            err_x = abs(float(x_h[-1, 0]) - x_drift_exact) / x_drift_exact
            err_y = float(np.max(np.abs(x_h[:, 1]))) / x_drift_exact
            exb_errors[pusher].append(float(max(np.hypot(err_x, err_y), 1e-16)))

    # 3. Hyperbolic motion in uniform E_z
    E_hyp = 1.5e7  # [V/m]
    t_end_hyp = 2.0e-9
    n_steps_hyp = 400
    dt_hyp = t_end_hyp / n_steps_hyp
    uz_exact_hyp = (e_charge * E_hyp / m_e) * t_end_hyp
    gamma_exact_hyp = float(np.sqrt(1.0 + (uz_exact_hyp / c_light) ** 2))
    z_exact_hyp = float(
        (m_e * c_light * c_light / (e_charge * E_hyp)) * (gamma_exact_hyp - 1.0)
    )
    hyp_results = {}
    for pusher in ("boris", "vay", "hc"):
        _, x_h, _, u_s = integrate_trajectory(
            x0=[0.0, 0.0, 0.0],
            u0=[0.0, 0.0, 0.0],
            dt=dt_hyp,
            n_steps=n_steps_hyp,
            E_fn=(0.0, 0.0, E_hyp),
            B_fn=(0.0, 0.0, 0.0),
            q=e_charge,
            m=m_e,
            pusher=pusher,
        )
        hyp_results[pusher] = {
            "z_exact_m": z_exact_hyp,
            "z_num_m": float(x_h[-1, 2]),
            "rel_z_error": float(abs(x_h[-1, 2] - z_exact_hyp) / z_exact_hyp),
            "rel_uz_error": float(abs(u_s[-1, 2] - uz_exact_hyp) / uz_exact_hyp),
        }

    # 4. Second-order dt convergence in combined (E, B)
    B_conv = 0.08
    E_conv = (2.0e6, -1.5e6, 3.0e6)
    u0_conv = [1.8 * c_light, 0.6 * c_light, 1.2 * c_light]
    T_conv = 1.2e-9
    n_list = [40, 80, 160, 320]
    dt_list = [T_conv / n for n in n_list]

    _, x_ref, _, _ = integrate_trajectory(
        x0=[0.0, 0.0, 0.0],
        u0=u0_conv,
        dt=T_conv / 20480,
        n_steps=20480,
        E_fn=E_conv,
        B_fn=(0.01, -0.02, B_conv),
        q=-e_charge,
        m=m_e,
        pusher="vay",
    )
    r_ref = x_ref[-1]
    r_norm = float(np.linalg.norm(r_ref))

    conv_errors = {p: [] for p in ("boris", "vay", "hc")}
    conv_orders = {}
    for pusher in ("boris", "vay", "hc"):
        for n_s in n_list:
            _, x_n, _, _ = integrate_trajectory(
                x0=[0.0, 0.0, 0.0],
                u0=u0_conv,
                dt=T_conv / n_s,
                n_steps=n_s,
                E_fn=E_conv,
                B_fn=(0.01, -0.02, B_conv),
                q=-e_charge,
                m=m_e,
                pusher=pusher,
            )
            conv_errors[pusher].append(float(np.linalg.norm(x_n[-1] - r_ref) / r_norm))
        slope = float(
            -np.polyfit(
                np.log(np.array(n_list, dtype=float)),
                np.log(np.array(conv_errors[pusher])),
                1,
            )[0]
        )
        conv_orders[pusher] = slope

    return {
        "gyration": gyr_results,
        "gyration_trajectories": gyr_trajectories,
        "exb_gammas": gammas_exb,
        "exb_errors": exb_errors,
        "hyperbolic": hyp_results,
        "convergence_dt": dt_list,
        "convergence_errors": conv_errors,
        "convergence_orders": conv_orders,
    }


def _make_v2_solver(grid_kind: str) -> SolverFIT3D:
    Nx, Ny, Nz = 6, 6, 6
    L = 0.06
    if grid_kind == "uniform":
        grid = GridFIT3D(
            xmin=0.0,
            xmax=L,
            ymin=0.0,
            ymax=L,
            zmin=0.0,
            zmax=L,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            verbose=0,
        )
    else:
        s_x = np.linspace(0.0, 1.0, Nx + 1)
        s_y = np.linspace(0.0, 1.0, Ny + 1)
        s_z = np.linspace(0.0, 1.0, Nz + 1)
        x_nodes = L * (s_x + 0.12 * np.sin(2.0 * np.pi * s_x))
        y_nodes = L * (s_y - 0.10 * np.sin(2.0 * np.pi * s_y))
        z_nodes = L * (s_z + 0.11 * np.sin(2.0 * np.pi * s_z))
        grid = GridFIT3D(x=x_nodes, y=y_nodes, z=z_nodes, verbose=0)
    return SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        verbose=0,
    )


def run_v2_benchmarks() -> dict:
    """Run V2 Gauss residual benchmarks over 1000 steps with 1024 particles."""
    n_p = 1024
    n_steps = 1000
    sample_steps = np.arange(0, n_steps + 1, 50)
    results = {}
    curves = {"steps": sample_steps.tolist()}

    for order in (1, 2):
        for grid_kind in ("uniform", "graded"):
            key = f"p{order}_{grid_kind}"
            est = estimate(
                cells=(6, 6, 6), steps=n_steps, dtype="float64", backend="cpu"
            )
            preflight(est, "laptop", raise_error=True)

            solver = _make_v2_solver(grid_kind)
            rng = np.random.default_rng(2026 + 10 * order + (0 if grid_kind == "uniform" else 1))
            L = 0.06
            sp = Species(name="electron", charge=-e_charge, mass=m_e)
            sp.add_particles(
                x=rng.uniform(0.0, L, size=n_p),
                y=rng.uniform(0.0, L, size=n_p),
                z=rng.uniform(0.0, L, size=n_p),
                ux=rng.uniform(-1.1 * c_light, 1.1 * c_light, size=n_p),
                uy=rng.uniform(-1.1 * c_light, 1.1 * c_light, size=n_p),
                uz=rng.uniform(-1.1 * c_light, 1.1 * c_light, size=n_p),
                w=rng.uniform(0.5, 1.5, size=n_p),
            )
            ps = ParticleSolver(
                solver,
                species=[sp],
                pusher="boris",
                shape=order,
                gather="galerkin",
                sort_every=200,
                B_ext=(0.02, -0.01, 0.04),
            ).attach()

            hist = [max(ps.gauss_residual(), 1e-16)]
            t0 = time.perf_counter()
            for step in range(1, n_steps + 1):
                solver.one_step()
                if step % 50 == 0:
                    hist.append(max(ps.gauss_residual(), 1e-16))
            wall_s = time.perf_counter() - t0
            curves[key] = hist
            results[key] = {
                "max_gauss_residual": float(np.max(hist)),
                "final_gauss_residual": float(hist[-1]),
                "wall_time_s": float(wall_s),
                "particle_steps_per_s": float((n_p * n_steps) / max(wall_s, 1e-9)),
            }

    # Naive deposit mutation curve (over 100 steps on p=1 uniform)
    solver_mut = _make_v2_solver("uniform")
    rng_m = np.random.default_rng(2036)
    sp_mut = Species(name="electron", charge=-e_charge, mass=m_e)
    sp_mut.add_particles(
        x=rng_m.uniform(0.0, 0.06, size=n_p),
        y=rng_m.uniform(0.0, 0.06, size=n_p),
        z=rng_m.uniform(0.0, 0.06, size=n_p),
        ux=rng_m.uniform(-1.1 * c_light, 1.1 * c_light, size=n_p),
        uy=rng_m.uniform(-1.1 * c_light, 1.1 * c_light, size=n_p),
        uz=rng_m.uniform(-1.1 * c_light, 1.1 * c_light, size=n_p),
        w=rng_m.uniform(0.5, 1.5, size=n_p),
    )
    ps_mut = ParticleSolver(
        solver_mut,
        species=[sp_mut],
        pusher="boris",
        shape=1,
        gather="galerkin",
        sort_every=0,
        deposit_scheme="naive",
    ).attach()
    mut_steps = list(range(0, 201, 10))
    mut_hist = [max(ps_mut.gauss_residual(), 1e-16)]
    for step in range(1, 201):
        solver_mut.one_step()
        if step % 10 == 0:
            mut_hist.append(max(ps_mut.gauss_residual(), 1e-16))
    curves["naive_steps"] = mut_steps
    curves["naive_p1_uniform"] = mut_hist
    results["naive_mutation_p1_uniform_max_residual"] = float(np.max(mut_hist))

    return {"metrics": results, "curves": curves}


def run_v3_benchmarks() -> dict:
    """Run V3 cold plasma oscillation benchmark for p = 1 and p = 2 (32 cells/lambda)."""
    Nx, Ny, Nz = 2, 2, 32
    Lx, Ly, Lz = 0.02, 0.02, 0.32  # dz = 0.01 m (32 cells per wavelength)
    dz = Lz / Nz
    k_mode = 2.0 * np.pi / Lz
    omega_p0 = 2.0 * np.pi * 1.2e9
    n0 = eps_0 * m_e * (omega_p0**2) / (e_charge**2)
    vol = Lx * Ly * Lz
    n_steps = 240

    est = estimate(cells=(Nx, Ny, Nz), steps=n_steps, dtype="float64", backend="cpu")
    preflight(est, "laptop", raise_error=True)

    v3_metrics = {}
    v3_curves = {}

    for order in (1, 2):
        grid = GridFIT3D(
            xmin=0.0,
            xmax=Lx,
            ymin=0.0,
            ymax=Ly,
            zmin=0.0,
            zmax=Lz,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            verbose=0,
        )

        n_ppc_z = 8
        n_z_tot = Nz * n_ppc_z
        z_1d = (np.arange(n_z_tot) + 0.5) * (Lz / n_z_tot)
        x_centers = (np.arange(Nx) + 0.5) * (Lx / Nx)
        y_centers = (np.arange(Ny) + 0.5) * (Ly / Ny)
        xx, yy, zz = np.meshgrid(x_centers, y_centers, z_1d, indexing="ij")
        x_p, y_p, z_p = xx.ravel(), yy.ravel(), zz.ravel()
        w_macro = (n0 * vol) / x_p.size

        delta_v = 1.0e-3 * (omega_p0 / k_mode)
        uz_p = delta_v * np.sin(k_mode * z_p)

        sp = Species(name="electron", charge=-e_charge, mass=m_e)
        sp.add_particles(x=x_p, y=y_p, z=z_p, ux=0.0, uy=0.0, uz=uz_p, w=w_macro)

        ps = ParticleSolver(
            solver,
            species=[sp],
            pusher="boris",
            shape=order,
            gather="galerkin",
            sort_every=0,
        ).attach()

        z_edges = (np.arange(Nz) + 0.5) * dz
        sin_basis = np.sin(k_mode * z_edges)

        t_hist = np.zeros(n_steps + 1)
        ez_mode = np.zeros(n_steps + 1)
        ke_hist = np.zeros(n_steps + 1)
        ue_hist = np.zeros(n_steps + 1)

        ke_hist[0] = ps.kinetic_energy(synchronised=True)
        ue_hist[0] = solver.field_energy()

        for step in range(1, n_steps + 1):
            solver.one_step()
            t_hist[step] = step * solver.dt
            Ez_1d = np.mean(solver.E[:, :, :, "z"], axis=(0, 1))
            ez_mode[step] = (2.0 / Nz) * np.sum(Ez_1d * sin_basis)
            ke_hist[step] = ps.kinetic_energy(synchronised=True)
            ue_hist[step] = solver.field_energy()

        omega_pred = cold_plasma_dispersion_discrete(
            omega_p0=omega_p0,
            dt=solver.dt,
            dz=dz,
            k=k_mode,
            order=order,
            gather="galerkin",
        )
        amp0 = float(np.max(np.abs(ez_mode)))
        fit = least_squares(
            lambda p_fit: ez_mode - p_fit[1] * np.sin(p_fit[0] * t_hist + p_fit[2]),
            x0=[omega_pred, amp0, 0.0],
        )
        omega_num = float(abs(fit.x[0]))
        ke0 = float(ke_hist[0])
        e_tot = ke_hist + ue_hist
        rel_energy_drift = float(np.max(np.abs(e_tot - ke0)) / ke0)
        pred_leapfrog_osc = float(0.25 * (omega_pred * solver.dt) ** 2)
        osc_curve = 1.0 + pred_leapfrog_osc * (np.sin(omega_pred * t_hist) ** 2)
        secular_slope = float(np.polyfit(t_hist, (e_tot / ke0) - osc_curve, 1)[0])
        secular_drift = float(abs(secular_slope * t_hist[-1]))

        v3_metrics[f"p{order}"] = {
            "omega_p0_rad_s": float(omega_p0),
            "omega_pred_rad_s": float(omega_pred),
            "omega_num_rad_s": omega_num,
            "rel_omega_error": float(abs(omega_num - omega_pred) / omega_pred),
            "ke0_J": ke0,
            "ue_max_J": float(np.max(ue_hist)),
            "rel_energy_drift": rel_energy_drift,
            "pred_leapfrog_energy_osc": pred_leapfrog_osc,
            "secular_energy_drift": secular_drift,
            "gauss_residual": float(ps.gauss_residual()),
        }
        v3_curves[f"p{order}"] = {
            "t_ns": t_hist * 1.0e9,
            "ez_norm": ez_mode / max(amp0, 1e-30),
            "ez_pred_norm": np.sin(omega_pred * t_hist),
            "ke_norm": ke_hist / ke0,
            "ue_norm": ue_hist / ke0,
            "etot_norm": e_tot / ke0,
        }

    return {"metrics": v3_metrics, "curves": v3_curves}


def measure_throughput() -> dict:
    """Measure vectorized CPU pusher and full particle step throughput on this VM."""
    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f_cpu:
            for line in f_cpu:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpu_model = "unknown"

    n_bench = 50000
    n_iters = 20
    rng = np.random.default_rng(77)
    ux = rng.normal(0.0, 0.5 * c_light, size=n_bench)
    uy = rng.normal(0.0, 0.5 * c_light, size=n_bench)
    uz = rng.normal(1.5 * c_light, 0.2 * c_light, size=n_bench)
    Ex = rng.normal(0.0, 1.0e5, size=n_bench)
    Ey = rng.normal(0.0, 1.0e5, size=n_bench)
    Ez = rng.normal(0.0, 1.0e5, size=n_bench)
    Bx = np.zeros(n_bench)
    By = np.zeros(n_bench)
    Bz = np.full(n_bench, 0.15)
    dt = 1.0e-12

    # Warmup
    push_boris(ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, -e_charge, m_e, dt)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        ux, uy, uz = push_boris(ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, -e_charge, m_e, dt)
    dt_boris = time.perf_counter() - t0
    boris_pushes_per_s = float((n_bench * n_iters) / max(dt_boris, 1e-9))

    # Full particle step throughput (10,000 particles on a 10x10x10 grid for 40 steps)
    step_rates = {}
    for order in (1, 2):
        grid = GridFIT3D(0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 10, 10, 10, verbose=0)
        solver = SolverFIT3D(
            grid,
            cfln=0.45,
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            verbose=0,
        )
        n_part = 10000
        n_steps_bench = 40
        sp = Species(name="electron", charge=-e_charge, mass=m_e)
        sp.add_particles(
            x=rng.uniform(0.0, 0.1, size=n_part),
            y=rng.uniform(0.0, 0.1, size=n_part),
            z=rng.uniform(0.0, 0.1, size=n_part),
            ux=rng.normal(0.0, 0.2 * c_light, size=n_part),
            uy=rng.normal(0.0, 0.2 * c_light, size=n_part),
            uz=rng.normal(0.5 * c_light, 0.1 * c_light, size=n_part),
            w=1.0e5,
        )
        ParticleSolver(
            solver, species=[sp], pusher="boris", shape=order, gather="galerkin", sort_every=20
        ).attach()
        solver.one_step()  # warmup step 0
        t1 = time.perf_counter()
        for _ in range(n_steps_bench):
            solver.one_step()
        dt_step = time.perf_counter() - t1
        step_rates[f"p{order}_full_particle_pushes_per_s"] = float(
            (n_part * n_steps_bench) / max(dt_step, 1e-9)
        )

    return {
        "machine": cpu_model,
        "cpu_count": int(os.cpu_count() or 1),
        "processes": 1,
        "n_particles": 10000,
        "grid_shape": [10, 10, 10],
        "n_steps": 40,
        "standalone_boris_pushes_per_s": boris_pushes_per_s,
        **step_rates,
    }


def make_figure_1(v1: dict, v2: dict, out_path: Path) -> None:
    """Create Figure 1: Relativistic E x B drift & convergence (V1) and Gauss conservation (V2)."""
    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=3.2,
        nrows=1,
        ncols=2,
        gridspec_kw={"wspace": 0.34},
    )
    ax_exb, ax_gauss = axes[0], axes[1]

    labels_p = {"boris": "Boris (1970)", "vay": "Vay (2008)", "hc": "Higuera–Cary (2017)"}
    gammas = np.array(v1["exb_gammas"], dtype=float)
    for idx, p_name in enumerate(("boris", "vay", "hc")):
        st = figstyle.series_style(idx, with_marker=True, ms=5.0)
        ax_exb.semilogy(
            gammas,
            v1["exb_errors"][p_name],
            label=labels_p[p_name],
            **st,
        )
    ax_exb.set_xlabel(r"Drift Lorentz factor $\gamma_d$")
    ax_exb.set_ylabel(r"Relative $\mathbf{E}\times\mathbf{B}$ drift error")
    ax_exb.set_title(r"(a) $\mathbf{E}\times\mathbf{B}$ drift error (V1)")
    ax_exb.set_ylim(1e-17, 1e0)
    figstyle.spectrum_grid(ax_exb)
    figstyle.place_legend_below(ax_exb, ncol=1, y_offset=-0.24)

    # (b) V2 Gauss's law residual over 1000 steps (uniform & graded, p=1 & 2 vs naive)
    c_v2 = v2["curves"]
    steps_arr = np.array(c_v2["steps"], dtype=float)
    v2_series = [
        ("p1_uniform", "Esirkepov $p=1$ (uniform)"),
        ("p1_graded", "Esirkepov $p=1$ (graded)"),
        ("p2_uniform", "Esirkepov $p=2$ (uniform)"),
        ("p2_graded", "Esirkepov $p=2$ (graded)"),
    ]
    for idx, (k_name, lbl) in enumerate(v2_series):
        st = figstyle.series_style(idx, with_marker=True, markevery=5, ms=4.5)
        ax_gauss.semilogy(steps_arr, c_v2[k_name], label=lbl, **st)

    st_naive = figstyle.series_style(4, with_marker=True, markevery=5, ms=4.8)
    ax_gauss.semilogy(
        c_v2["naive_steps"],
        c_v2["naive_p1_uniform"],
        label=r"Naive $\mathbf{J}=q\mathbf{v}$ (mutated)",
        **st_naive,
    )
    ax_gauss.axhline(1.0e-12, color=figstyle.INK, linestyle="--", linewidth=1.0)
    ax_gauss.set_xlabel("Time step $n$")
    ax_gauss.set_ylabel(r"Relative Gauss residual $\|\tilde{\mathbf{S}}\tilde{\mathbf{d}}-q\|_\infty/\|q\|_\infty$")
    ax_gauss.set_title("(b) Gauss charge residual (V2)")
    ax_gauss.set_ylim(1e-16, 1e2)
    figstyle.spectrum_grid(ax_gauss)
    figstyle.place_legend_below(ax_gauss, ncol=1, y_offset=-0.24)

    figstyle.save(fig, out_path, auto_legend_below=False)
    matplotlib.pyplot.close(fig)


def make_figure_2(v3: dict, out_path: Path) -> None:
    """Create Figure 2: Cold plasma oscillation dispersion and energy exchange (V3)."""
    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=3.2,
        nrows=1,
        ncols=2,
        gridspec_kw={"wspace": 0.34},
    )
    ax_osc, ax_eng = axes[0], axes[1]

    c_p1 = v3["curves"]["p1"]
    c_p2 = v3["curves"]["p2"]
    m_p1 = v3["metrics"]["p1"]
    m_p2 = v3["metrics"]["p2"]

    # (a) Electric field mode oscillation vs Birdsall-Langdon prediction
    ax_osc.plot(
        c_p1["t_ns"],
        c_p1["ez_pred_norm"],
        color=figstyle.INK,
        linestyle="-",
        linewidth=1.4,
        label=r"Birdsall–Langdon theory ($p=1$)",
    )
    st1 = figstyle.series_style(0, with_marker=True, markevery=14, ms=4.5)
    st2 = figstyle.series_style(1, with_marker=True, markevery=14, ms=4.5)
    ax_osc.plot(
        c_p1["t_ns"],
        c_p1["ez_norm"],
        label=f"Particles $p=1$ (rel. err {m_p1['rel_omega_error']:.2e})",
        **st1,
    )
    ax_osc.plot(
        c_p2["t_ns"],
        c_p2["ez_norm"],
        label=f"Particles $p=2$ (rel. err {m_p2['rel_omega_error']:.2e})",
        **st2,
    )
    ax_osc.set_xlabel("Time $t$ [ns]")
    ax_osc.set_ylabel(r"Normalized mode $E_{z,1}(t)/E_{\max}$")
    ax_osc.set_title(r"(a) Plasma mode $E_{z,1}(t)$ (V3)")
    ax_osc.set_ylim(-1.20, 1.20)
    figstyle.spectrum_grid(ax_osc)
    figstyle.place_legend_below(ax_osc, ncol=1, y_offset=-0.24)

    # (b) Kinetic <-> Electric field energy exchange and total energy conservation
    st_ke = figstyle.series_style(0, with_marker=False, lw=1.9)
    st_ue = figstyle.series_style(1, with_marker=False, lw=1.9)
    st_tot = figstyle.series_style(2, with_marker=False, lw=2.0)
    ax_eng.plot(c_p1["t_ns"], c_p1["ke_norm"], label=r"Kinetic energy $K(t)/K_0$", **st_ke)
    ax_eng.plot(c_p1["t_ns"], c_p1["ue_norm"], label=r"Electric energy $U_E(t)/K_0$", **st_ue)
    ax_eng.plot(
        c_p1["t_ns"],
        c_p1["etot_norm"],
        label=rf"Total $(K+U_E)/K_0$ (osc. {m_p1['rel_energy_drift']*1e3:.2f}$\times 10^{{-3}}$)",
        **st_tot,
    )
    ax_eng.set_xlabel("Time $t$ [ns]")
    ax_eng.set_ylabel(r"Normalized energy $E(t) / K_0$")
    ax_eng.set_title("(b) Plasma energy exchange (V3)")
    ax_eng.set_ylim(-0.08, 1.18)
    figstyle.spectrum_grid(ax_eng)
    figstyle.place_legend_below(ax_eng, ncol=1, y_offset=-0.24)

    figstyle.save(fig, out_path, auto_legend_below=False)
    matplotlib.pyplot.close(fig)


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    v1_out = run_v1_benchmarks()
    v2_out = run_v2_benchmarks()
    v3_out = run_v3_benchmarks()
    timings = measure_throughput()

    fig1_path = IMG_DIR / "particles_pushers_and_charge_conservation.png"
    fig2_path = IMG_DIR / "particles_cold_plasma_oscillation.png"
    make_figure_1(v1_out, v2_out, fig1_path)
    make_figure_2(v3_out, fig2_path)

    payload = {
        "v1_single_particle": {
            "gyration": v1_out["gyration"],
            "exb_gammas": v1_out["exb_gammas"],
            "exb_errors": v1_out["exb_errors"],
            "hyperbolic": v1_out["hyperbolic"],
            "convergence_dt_s": v1_out["convergence_dt"],
            "convergence_errors": v1_out["convergence_errors"],
            "convergence_orders": v1_out["convergence_orders"],
        },
        "v2_charge_conservation": v2_out["metrics"],
        "v3_cold_plasma_oscillation": v3_out["metrics"],
        "throughput": timings,
        "figures": [str(fig1_path.relative_to(REPO_ROOT)), str(fig2_path.relative_to(REPO_ROOT))],
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved figures to {fig1_path} and {fig2_path}")
    print(f"Saved benchmark metrics to {JSON_PATH}")


if __name__ == "__main__":
    main()
