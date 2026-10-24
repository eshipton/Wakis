# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Analytical dispersion formulas, Wakis FIT simulation runners, and plotting
utilities for Tutorial 1.1 (`01-01_plane_wave`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0
from scipy.interpolate import RegularGridInterpolator
from scipy.special import hankel2

from wakis import GridFIT3D, SolverFIT3D, TFSFSource, figstyle, schematic
from wakis.estimate import estimate, preflight
from wakis.field_monitors import DFTMonitor
from wakis.sources import CurrentSource, GaussianSource

ETA0 = float(np.sqrt(mu0 / eps0))
LAM0 = 1.0e-6  # Reference free-space wavelength [m] (1.0 um)
F0 = c0 / LAM0  # Reference center frequency [Hz] (~299.79 THz)


def discrete_dispersion_1d(
    freqs: np.ndarray | float,
    dz: float,
    dt: float,
    n_bg: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact 1D Yee/FIT discrete dispersion relation along the z-axis.

    Solves
        sin(omega * dt / 2) / (v0 * dt) = sin(k_num * dz / 2) / dz
    for the numerical wavenumber ``k_num`` and normalized phase velocity
    ``vp / v0 = omega / (k_num * v0)``, where ``v0 = c0 / n_bg``.

    Parameters
    ----------
    freqs : float or array_like
        Frequencies [Hz].
    dz : float
        Grid cell spacing along the propagation axis [m].
    dt : float
        Leapfrog time step [s].
    n_bg : float, optional
        Background refractive index [dimensionless]. Default is 1.0.

    Returns
    -------
    k_num : ndarray
        Discrete numerical wavenumber [rad/m].
    vp_over_v0 : ndarray
        Normalized numerical phase velocity ``v_p / v_0`` [dimensionless].
    """
    f_arr = np.atleast_1d(np.asarray(freqs, dtype=float))
    v0 = c0 / float(n_bg)
    omega = 2.0 * np.pi * f_arr
    s_1d = v0 * float(dt) / float(dz)
    arg = np.clip(np.sin(0.5 * omega * float(dt)) / s_1d, -1.0, 1.0)
    k_num = (2.0 / float(dz)) * np.arcsin(arg)
    vp_over_v0 = omega / (k_num * v0)
    return k_num, vp_over_v0


def discrete_dispersion_2d_angle(
    freq: float,
    dx: float,
    dt: float,
    phi_rad: np.ndarray | float,
    n_bg: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact 2D Yee/FIT discrete dispersion relation vs propagation angle ``phi_rad``.

    On a uniform square 2D grid (``dx = dy``), a plane wave propagating at
    angle ``phi`` with wavevector ``k_num * (cos(phi), sin(phi))`` satisfies:

        (sin(omega * dt / 2) / (v0 * dt))^2
            = (sin(0.5 * k_num * cos(phi) * dx) / dx)^2
            + (sin(0.5 * k_num * sin(phi) * dx) / dx)^2.

    Parameters
    ----------
    freq : float
        Wave frequency [Hz].
    dx : float
        Square grid cell size ``dx = dy`` [m].
    dt : float
        Leapfrog time step [s].
    phi_rad : float or array_like
        In-plane propagation angle measured from the x-axis [rad].
    n_bg : float, optional
        Background refractive index [dimensionless]. Default is 1.0.

    Returns
    -------
    k_num : ndarray
        Directional numerical wavenumber ``k_num(phi)`` [rad/m].
    vp_over_v0 : ndarray
        Normalized directional phase velocity ``v_p(phi) / v_0`` [dimensionless].
    """
    phis = np.atleast_1d(np.asarray(phi_rad, dtype=float))
    v0 = c0 / float(n_bg)
    omega = 2.0 * np.pi * float(freq)
    k0 = omega / v0
    rhs_target = (np.sin(0.5 * omega * float(dt)) / (v0 * float(dt))) ** 2

    k_num = np.full_like(phis, k0, dtype=float)
    c_phi = np.cos(phis)
    s_phi = np.sin(phis)
    h = float(dx)

    for _ in range(8):
        kx = k_num * c_phi
        ky = k_num * s_phi
        f_val = (
            (np.sin(0.5 * kx * h) / h) ** 2
            + (np.sin(0.5 * ky * h) / h) ** 2
            - rhs_target
        )
        df_dk = (
            c_phi * np.sin(kx * h) / (2.0 * h)
            + s_phi * np.sin(ky * h) / (2.0 * h)
        )
        k_num = k_num - f_val / np.maximum(df_dk, 1e-30)

    vp_over_v0 = k0 / k_num
    return k_num, vp_over_v0


def courant_amplification_factor(cfl_numbers: np.ndarray | float) -> np.ndarray:
    """Exact von Neumann per-step amplification factor modulus at the Nyquist mode.

    For 3D Courant number ``S = c0 * dt * sqrt(dx^-2 + dy^-2 + dz^-2)``, the
    characteristic polynomial of the leapfrog update at the Nyquist corner
    ``(pi/dx, pi/dy, pi/dz)`` is ``lambda^2 - 2*(1 - 2*S^2)*lambda + 1 = 0``.
    When ``S <= 1``, ``|lambda| = 1``; when ``S > 1``, the unstable root has
    modulus ``|lambda_max(S)| = (S + sqrt(S^2 - 1))^2``.

    Parameters
    ----------
    cfl_numbers : float or array_like
        Courant numbers ``S`` [dimensionless].

    Returns
    -------
    ndarray
        Maximum amplification factor modulus ``|lambda_max(S)|`` per step.
    """
    s_arr = np.atleast_1d(np.asarray(cfl_numbers, dtype=float))
    amp = np.ones_like(s_arr)
    mask = s_arr > 1.0
    s_gt = s_arr[mask]
    amp[mask] = (s_gt + np.sqrt(s_gt**2 - 1.0)) ** 2
    return amp


def sheet_radiated_field_1d(
    freqs: np.ndarray | float,
    dz: float,
    dt: float,
    z_dist: np.ndarray | float = 0.0,
    j0: float = 1.0,
    n_bg: float = 1.0,
) -> np.ndarray:
    """Exact discrete 1D electric-field transfer function ``E_x(z, f) / S(f)``.

    A soft surface-current sheet ``J_x(t) = j0 * s(t) * delta(z - z0)`` on a 1D
    FIT grid radiates symmetric plane waves in ``+-z`` with source-normalized
    complex amplitude:

        E_x(z, f) / S(f) = -(eta_0 / (2 * n_bg))
                           * (cos(omega * dt / 2) / cos(k_num * dz / 2))
                           * j0 * exp(-1j * k_num * |z - z0|).

    Parameters
    ----------
    freqs : float or array_like
        Frequencies [Hz].
    dz : float
        Cell spacing along z [m].
    dt : float
        Leapfrog time step [s].
    z_dist : float or array_like, optional
        Distance ``|z - z0|`` from the current sheet [m]. Default is 0.0.
    j0 : float, optional
        Surface current density amplitude [A/m]. Default is 1.0.
    n_bg : float, optional
        Background refractive index [dimensionless]. Default is 1.0.

    Returns
    -------
    ndarray
        Complex transfer function ``E_x(z, f) / S(f)`` [V / (m s)].
    """
    f_arr = np.atleast_1d(np.asarray(freqs, dtype=float))
    omega = 2.0 * np.pi * f_arr
    k_num, _ = discrete_dispersion_1d(f_arr, dz, dt, n_bg=n_bg)
    eta = ETA0 / float(n_bg)
    imp_discrete = 0.5 * eta * np.cos(0.5 * omega * float(dt)) / np.cos(0.5 * k_num * float(dz))
    z_abs = np.abs(np.asarray(z_dist, dtype=float))
    if z_abs.ndim == 0:
        return -float(j0) * imp_discrete * np.exp(-1j * k_num * float(z_abs))
    return -float(j0) * imp_discrete[:, None] * np.exp(-1j * k_num[:, None] * z_abs[None, :])


def run_soft_source_and_tfsf_task(cfg: dict) -> dict:
    """Run Task 0: Soft CurrentSource sheet, DFT source normalization, and TFSFSource."""
    cpw = int(cfg["cpw"])
    dz = LAM0 / cpw
    n_pml = 10
    nz_phys = int(round(4.0 * cpw))
    nz = nz_phys + 2 * n_pml
    x = np.array([0.0, dz])
    y = np.array([0.0, dz])
    z = np.arange(nz + 1) * dz

    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )

    # Verify discrete vector identity S @ C == 0 (div(curl) == 0) on the FIT grid
    from scipy.sparse import hstack as sp_hstack

    s_div = sp_hstack([solver.Px, solver.Py, solver.Pz])
    sc_prod = s_div @ solver.C
    c_diff = float(np.max(np.abs(sc_prod.data))) if sc_prod.nnz else 0.0

    pulse = GaussianSource(F0, fwidth=0.35 * F0, cutoff=4.5)
    iz_src = n_pml + int(round(1.0 * cpw))
    z_src = float(solver.z[iz_src])
    src = CurrentSource(
        pulse,
        component="x",
        center=(0.5 * dz, 0.5 * dz, z_src),
        size=(dz, dz, 0.0),
        amplitude=1.0,
    )

    freqs = np.linspace(0.75 * F0, 1.25 * F0, 25)
    iz1 = iz_src + int(round(0.6 * cpw))
    iz2 = iz_src + int(round(2.1 * cpw))
    z1, z2 = float(solver.z[iz1]), float(solver.z[iz2])
    mon_line = DFTMonitor(
        freqs,
        ["Ex", "Hy"],
        center=(0.5 * dz, 0.5 * dz, 0.5 * (z1 + z2)),
        size=(0.0, 0.0, z2 - z1),
    )

    nt = int(np.ceil((pulse.end_time + 2.6 * (nz * dz) / c0) / solver.dt))
    est = estimate((grid.Nx, grid.Ny, grid.Nz), steps=nt, cpml=True, monitors=[(len(freqs), iz2 - iz1 + 1, 2)])
    preflight_problems = preflight(est, "laptop", raise_error=False)

    t_arr = np.arange(nt) * solver.dt
    probe_z1 = np.empty(nt, dtype=float)
    probe_z2 = np.empty(nt, dtype=float)
    stride_wf = max(1, nt // 140)
    wf_frames = []
    wf_times = []
    movie_frames_e = []
    movie_frames_h = []
    movie_times = []
    stride_mov = max(1, nt // 36)

    for n in range(nt):
        t = n * solver.dt
        src.update(solver, t)
        solver.one_step()
        mon_line.update(solver, t)
        ex_z = np.asarray(solver.E[0, 0, :-1, "x"], dtype=float)
        hy_z = np.asarray(solver.H[0, 0, :-1, "y"], dtype=float)
        probe_z1[n] = ex_z[iz1]
        probe_z2[n] = ex_z[iz2]
        if n % stride_wf == 0:
            wf_frames.append(ex_z.copy())
            wf_times.append(t)
        if n % stride_mov == 0 and len(movie_frames_e) < 32:
            movie_frames_e.append(ex_z.copy())
            movie_frames_h.append((ETA0 * hy_z).copy())
            movie_times.append(t)

    s_f = pulse.fourier_transform(freqs)
    ex_dft, coords = mon_line.get_field("Ex")
    z_mon = coords[2]
    ex_raw = ex_dft[:, 0, 0, :]
    ex_norm = ex_raw / s_f[:, None]

    # Exact discrete transfer function E_x(z, f) / S(f)
    ex_th = sheet_radiated_field_1d(freqs, dz, solver.dt, z_dist=z_mon - z_src, j0=1.0)
    amp_sim = np.mean(np.abs(ex_norm), axis=-1)
    amp_th = np.mean(np.abs(ex_th), axis=-1)
    norm_rel_err = float(np.max(np.abs(amp_sim - amp_th) / amp_th))

    # Compare analytic fourier_transform(f) vs discrete-time Riemann sum of s(t + dt/2)
    t_dense = np.arange(nt) * solver.dt + 0.5 * solver.dt
    s_t = pulse(t_dense)
    s_dft_num = np.sum(
        s_t[None, :] * np.exp(-2j * np.pi * freqs[:, None] * t_dense[None, :]) * solver.dt,
        axis=-1,
    )
    dc_integral = float(np.abs(np.sum(s_t) * solver.dt) / np.max(np.abs(s_f)))

    # Second mini-run: TFSFSource one-way injection and machine-precision cancellation outside box
    grid_tf = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver_tf = SolverFIT3D(
        grid_tf,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    iz_tf_lo = n_pml + int(round(0.8 * cpw))
    iz_tf_hi = n_pml + int(round(3.2 * cpw))
    z_tf_lo, z_tf_hi = float(z[iz_tf_lo]), float(z[iz_tf_hi])
    tfsf = TFSFSource(
        pulse,
        center=(0.5 * dz, 0.5 * dz, 0.5 * (z_tf_lo + z_tf_hi)),
        size=(None, None, z_tf_hi - z_tf_lo),
        direction="+z",
        polarization="x",
        amplitude=1.0,
    )
    tfsf_profile_max = np.zeros(nz, dtype=float)
    tfsf_snap_e = np.zeros(nz, dtype=float)
    snap_step = int(round((pulse.t0 + 1.0 * LAM0 / c0) / solver_tf.dt))
    for n in range(nt):
        t = n * solver_tf.dt
        tfsf.update(solver_tf, t)
        solver_tf.one_step()
        ex_cur = np.asarray(solver_tf.E[0, 0, :-1, "x"], dtype=float)
        tfsf_profile_max = np.maximum(tfsf_profile_max, np.abs(ex_cur))
        if n == snap_step:
            tfsf_snap_e = ex_cur.copy()

    max_tf_in = float(np.max(tfsf_profile_max[iz_tf_lo + 2 : iz_tf_hi - 2]))
    max_sf_lo = float(np.max(tfsf_profile_max[n_pml + 1 : iz_tf_lo - 1]))
    max_sf_hi = float(np.max(tfsf_profile_max[iz_tf_hi + 1 : nz - n_pml - 1]))
    tfsf_leakage = max(max_sf_lo, max_sf_hi) / max(max_tf_in, 1e-30)

    return {
        "name": cfg["name"],
        "cpw": cpw,
        "dx_background_nm": dz * 1e9,
        "dx_min_nm": dz * 1e9,
        "min_feature_nm": LAM0 * 1e9,
        "cells_across_feature": float(cpw),
        "n_cells": int(grid.Nx * grid.Ny * grid.Nz),
        "n_steps": 2 * nt,
        "dt": float(solver.dt),
        "c_duality_diff": c_diff,
        "dc_integral_rel": dc_integral,
        "est_host_mb": float(est.host_bytes / 1e6),
        "preflight_ok": int(len(preflight_problems) == 0),
        "t_arr": t_arr,
        "s_t": s_t,
        "probe_z1": probe_z1,
        "probe_z2": probe_z2,
        "z_um": solver.z[:-1] * 1e6,
        "z_src_um": z_src * 1e6,
        "z1_um": z1 * 1e6,
        "z2_um": z2 * 1e6,
        "wf_frames": np.asarray(wf_frames, dtype=float),
        "wf_times": np.asarray(wf_times, dtype=float),
        "movie_frames_e": np.asarray(movie_frames_e, dtype=float),
        "movie_frames_h": np.asarray(movie_frames_h, dtype=float),
        "movie_times": np.asarray(movie_times, dtype=float),
        "freqs": freqs,
        "s_f": s_f,
        "s_dft_num": s_dft_num,
        "amp_raw": np.mean(np.abs(ex_raw), axis=-1),
        "amp_sim": amp_sim,
        "amp_th": amp_th,
        "norm_rel_err": norm_rel_err,
        "z_tf_lo_um": z_tf_lo * 1e6,
        "z_tf_hi_um": z_tf_hi * 1e6,
        "tfsf_profile_max": tfsf_profile_max,
        "tfsf_snap_e": tfsf_snap_e,
        "tfsf_leakage": tfsf_leakage,
    }


def run_courant_stability_task(cfg: dict) -> dict:
    """Run Task 1: Courant stability limit and supercritical blow-up growth rate."""
    cpw = int(cfg["cpw"])
    dx = LAM0 / cpw
    nx, ny, nz = 6, 6, 30
    x = np.arange(nx + 1) * dx
    y = np.arange(ny + 1) * dx
    z = np.arange(nz + 1) * dx

    s_list = np.array([0.50, 0.80, 0.95, 0.99, 1.005, 1.01, 1.02, 1.05], dtype=float)
    n_max_steps = 220
    histories = np.full((len(s_list), n_max_steps), np.nan, dtype=float)
    steps_run = np.zeros(len(s_list), dtype=int)
    g_sim_arr = np.ones(len(s_list), dtype=float)
    g_th_arr = courant_amplification_factor(s_list)
    unstable_profile_z = np.zeros(nz, dtype=float)

    pulse = GaussianSource(F0, fwidth=0.40 * F0, cutoff=3.5)

    for idx_s, s_val in enumerate(s_list):
        grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
        solver = SolverFIT3D(
            grid,
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            cfln=float(s_val),
            use_stl=False,
            verbose=0,
        )
        src = CurrentSource(
            pulse,
            component="x",
            center=(3.0 * dx, 3.0 * dx, 15.0 * dx),
            size=(0.0, 0.0, 0.0),
            amplitude=dx**2,
        )
        trace = []
        for n in range(n_max_steps):
            t = n * solver.dt
            src.update(solver, t)
            solver.one_step()
            ex_mat = np.asarray(solver.E.to_matrix("x"), dtype=float)
            val = float(np.max(np.abs(ex_mat)))
            trace.append(val)
            if val > 1.0e16:
                if abs(s_val - 1.02) < 1e-6:
                    prof = ex_mat[3, 3, :]
                    unstable_profile_z = prof / max(float(np.max(np.abs(prof))), 1e-30)
                break
        if abs(s_val - 1.02) < 1e-6 and np.all(unstable_profile_z == 0):
            prof = np.asarray(solver.E.to_matrix("x"), dtype=float)[3, 3, :]
            unstable_profile_z = prof / max(float(np.max(np.abs(prof))), 1e-30)

        trace_arr = np.asarray(trace, dtype=float)
        histories[idx_s, : len(trace_arr)] = trace_arr
        steps_run[idx_s] = len(trace_arr)
        if s_val > 1.0:
            tail = trace_arr[-15:]
            g_sim_arr[idx_s] = float(np.exp(np.polyfit(np.arange(len(tail)), np.log(tail), 1)[0]))
        else:
            # Stable: verify late-time ratio is ~1
            tail = trace_arr[-40:]
            g_sim_arr[idx_s] = float((np.max(tail[-20:]) / np.max(tail[:20])) ** (1.0 / 20.0))

    sup_mask = s_list > 1.0
    max_growth_rel_err = float(
        np.max(np.abs(g_sim_arr[sup_mask] - g_th_arr[sup_mask]) / g_th_arr[sup_mask])
    )

    return {
        "name": cfg["name"],
        "cpw": cpw,
        "dx_background_nm": dx * 1e9,
        "dx_min_nm": dx * 1e9,
        "min_feature_nm": LAM0 * 1e9,
        "cells_across_feature": float(cpw),
        "n_cells": int(nx * ny * nz),
        "n_steps": int(np.sum(steps_run)),
        "s_list": s_list,
        "histories": histories,
        "steps_run": steps_run,
        "g_sim": g_sim_arr,
        "g_th": g_th_arr,
        "max_growth_rel_err": max_growth_rel_err,
        "unstable_profile_z": unstable_profile_z,
    }


def _measure_dispersion_at_cpw(
    cpw: int,
    cfln_1d: float = 0.5,
    cfln_2d: float = 0.5,
    save_field_2d: bool = False,
) -> dict:
    """Measure 1D broadband dispersion and 2D angular dispersion at resolution ``cpw``."""
    dz = LAM0 / cpw
    n_pml = 8
    nz = int(round(3.6 * cpw)) + 2 * n_pml
    x1 = np.array([0.0, dz])
    y1 = np.array([0.0, dz])
    z1_ax = np.arange(nz + 1) * dz

    grid1 = GridFIT3D(x=x1, y=y1, z=z1_ax, verbose=0)
    solver1 = SolverFIT3D(
        grid1,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        cfln=cfln_1d,
        use_stl=False,
        verbose=0,
    )
    pulse = GaussianSource(F0, fwidth=0.35 * F0, cutoff=4.2)
    iz_src = n_pml + int(round(0.6 * cpw))
    src1 = CurrentSource(
        pulse,
        component="x",
        center=(0.5 * dz, 0.5 * dz, float(solver1.z[iz_src])),
        size=(dz, dz, 0.0),
        amplitude=1.0,
    )
    freqs = np.linspace(0.75 * F0, 1.25 * F0, 21)
    za = float(solver1.z[iz_src + int(round(0.6 * cpw))])
    zb = float(solver1.z[iz_src + int(round(2.4 * cpw))])
    mon1 = DFTMonitor(
        freqs,
        ["Ex"],
        center=(0.5 * dz, 0.5 * dz, 0.5 * (za + zb)),
        size=(0.0, 0.0, zb - za),
    )
    nt1 = int(np.ceil((pulse.end_time + 2.5 * (nz * dz) / c0) / solver1.dt))
    solver1.emsolve(nt1, source=[src1], monitors=[mon1])

    ex_f_z, coords1 = mon1.get_field("Ex")
    z_m = coords1[2]
    ex_n = ex_f_z[:, 0, 0, :] / pulse.fourier_transform(freqs)[:, None]
    phi_z = -np.unwrap(np.angle(ex_n), axis=-1)
    k_sim_1d = np.array([np.polyfit(z_m, p_row, 1)[0] for p_row in phi_z], dtype=float)
    vp_sim_1d = (2.0 * np.pi * freqs) / (k_sim_1d * c0)
    _, vp_th_1d = discrete_dispersion_1d(freqs, dz, solver1.dt)

    # 2D (x, y) cylindrical wave dispersion vs propagation angle phi in [0, 90 deg]
    dx = dz
    n_half = int(round(3.2 * cpw)) + n_pml
    x2 = np.arange(-n_half, n_half + 1) * dx
    y2 = np.arange(-n_half, n_half + 1) * dx
    z2 = np.array([0.0, dx])
    grid2 = GridFIT3D(x=x2, y=y2, z=z2, verbose=0)
    solver2 = SolverFIT3D(
        grid2,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=n_pml,
        cfln=cfln_2d,
        use_stl=False,
        verbose=0,
    )
    src2 = CurrentSource(
        pulse,
        component="z",
        center=(0.0, 0.0, 0.5 * dx),
        size=(0.0, 0.0, dx),
        amplitude=1.0,
    )
    l_mon = 2.6 * LAM0
    mon2 = DFTMonitor(
        [F0],
        ["Ez"],
        center=(0.0, 0.0, 0.5 * dx),
        size=(2.0 * l_mon, 2.0 * l_mon, 0.0),
    )
    nt2 = int(np.ceil((pulse.end_time + 5.0 * LAM0 / c0) / solver2.dt))
    solver2.emsolve(nt2, source=[src2], monitors=[mon2])

    ez_xy, coords2 = mon2.get_field("Ez")
    ez_plane = ez_xy[0, :, :, 0] / pulse.fourier_transform(F0)
    interp_re = RegularGridInterpolator((coords2[0], coords2[1]), ez_plane.real, method="cubic")
    interp_im = RegularGridInterpolator((coords2[0], coords2[1]), ez_plane.imag, method="cubic")

    phis_deg = np.linspace(0.0, 90.0, 19)
    phis_rad = np.deg2rad(phis_deg)
    k0 = 2.0 * np.pi / LAM0
    r_ray = np.linspace(0.95 * LAM0, 2.35 * LAM0, 35)
    phi_h0 = -np.unwrap(np.angle(hankel2(0, k0 * r_ray)))
    delta_h0_ray = phi_h0 - k0 * r_ray

    vp_cyl_2d = np.empty(len(phis_rad), dtype=float)
    for idx_p, phi in enumerate(phis_rad):
        pts = np.column_stack([r_ray * np.cos(phi), r_ray * np.sin(phi)])
        e_ray = interp_re(pts) + 1j * interp_im(pts)
        phi_sim_ray = -np.unwrap(np.angle(e_ray)) - delta_h0_ray
        k_num_sim = float(np.polyfit(r_ray, phi_sim_ray, 1)[0])
        vp_cyl_2d[idx_p] = (2.0 * np.pi * F0) / (k_num_sim * c0)

    _, vp_th_2d = discrete_dispersion_2d_angle(F0, dx, solver2.dt, phis_rad)

    # Also verify exact 2D Bloch plane-wave eigenmode at each angle phi
    vp_bloch_2d = np.empty(len(phis_rad), dtype=float)
    xb = np.arange(3) * dx
    yb = np.arange(3) * dx
    zb_ax = np.array([0.0, dx])
    for idx_p, phi in enumerate(phis_rad):
        kx = k0 * np.cos(phi)
        ky = k0 * np.sin(phi)
        gb = GridFIT3D(x=xb, y=yb, z=zb_ax, verbose=0)
        sb = SolverFIT3D(
            gb,
            bc_low=["bloch", "bloch", "periodic"],
            bc_high=["bloch", "bloch", "periodic"],
            k_bloch=(kx, ky, 0.0),
            dtype=np.complex128,
            cfln=cfln_2d,
            use_stl=False,
            verbose=0,
        )
        xx, yy = np.meshgrid(sb.x, sb.y, indexing="ij")
        sb.E[:, :, 0, "z"] = np.exp(+1j * (kx * xx + ky * yy))
        tr = np.empty(40, dtype=float)
        for n in range(40):
            tr[n] = float(sb.E[0, 0, 0, "z"].real)
            sb.one_step()
        m_ok = np.abs(tr[1:-1]) > 0.2
        cos_wdt = float(np.mean(0.5 * (tr[2:][m_ok] + tr[:-2][m_ok]) / tr[1:-1][m_ok]))
        w_sim = float(np.arccos(np.clip(cos_wdt, -1.0, 1.0)) / sb.dt)
        # Convert fixed-k eigenfrequency shift to phase velocity at omega = 2*pi*F0
        vp_bloch_2d[idx_p] = w_sim / (k0 * c0)

    return {
        "dz": dz,
        "dt_1d": float(solver1.dt),
        "dt_2d": float(solver2.dt),
        "n_cells": int(grid2.Nx * grid2.Ny * grid2.Nz),
        "n_steps": int(nt1 + nt2),
        "freqs": freqs,
        "vp_sim_1d": vp_sim_1d,
        "vp_th_1d": vp_th_1d,
        "phis_deg": phis_deg,
        "vp_cyl_2d": vp_cyl_2d,
        "vp_bloch_2d": vp_bloch_2d,
        "vp_th_2d": vp_th_2d,
        "ez_x_um": coords2[0] * 1e6 if save_field_2d else np.zeros(0),
        "ez_y_um": coords2[1] * 1e6 if save_field_2d else np.zeros(0),
        "ez_re": ez_plane.real if save_field_2d else np.zeros((0, 0)),
    }


def run_dispersion_task(cfg: dict) -> dict:
    """Run a 1D & 2D numerical dispersion resolution-study task."""
    cpw = int(cfg["cpw"])
    save_2d = bool(cfg.get("save_field_2d", False))
    res = _measure_dispersion_at_cpw(cpw, save_field_2d=save_2d)

    demo_cpw = 0
    demo_vp_0deg = np.nan
    demo_vp_45deg = np.nan
    demo_th_0deg = np.nan
    demo_th_45deg = np.nan
    if str(cfg.get("level", "")) == "reference":
        # Clearly labelled under-resolved demonstration at cpw = 6 (< 10 cpw rule)
        demo_cpw = 6
        d_res = _measure_dispersion_at_cpw(demo_cpw, save_field_2d=False)
        demo_vp_0deg = float(d_res["vp_cyl_2d"][0])
        demo_vp_45deg = float(d_res["vp_cyl_2d"][9])
        demo_th_0deg = float(d_res["vp_th_2d"][0])
        demo_th_45deg = float(d_res["vp_th_2d"][9])

    max_err_1d = float(np.max(np.abs(res["vp_sim_1d"] - res["vp_th_1d"])))
    max_err_2d = float(np.max(np.abs(res["vp_cyl_2d"] - res["vp_th_2d"])))

    return {
        "name": cfg["name"],
        "level": cfg["level"],
        "cpw": cpw,
        "dx_background_nm": res["dz"] * 1e9,
        "dx_min_nm": res["dz"] * 1e9,
        "min_feature_nm": LAM0 * 1e9,
        "cells_across_feature": float(cpw),
        "n_cells": res["n_cells"],
        "n_steps": res["n_steps"],
        "dt_1d": res["dt_1d"],
        "dt_2d": res["dt_2d"],
        "freqs": res["freqs"],
        "vp_sim_1d": res["vp_sim_1d"],
        "vp_th_1d": res["vp_th_1d"],
        "phis_deg": res["phis_deg"],
        "vp_cyl_2d": res["vp_cyl_2d"],
        "vp_bloch_2d": res["vp_bloch_2d"],
        "vp_th_2d": res["vp_th_2d"],
        "vp_0deg_sim": float(res["vp_cyl_2d"][0]),
        "vp_45deg_sim": float(res["vp_cyl_2d"][9]),
        "vp_0deg_th": float(res["vp_th_2d"][0]),
        "vp_45deg_th": float(res["vp_th_2d"][9]),
        "max_err_1d_vs_th": max_err_1d,
        "max_err_2d_vs_th": max_err_2d,
        "ez_x_um": res["ez_x_um"],
        "ez_y_um": res["ez_y_um"],
        "ez_re": res["ez_re"],
        "demo_cpw": demo_cpw,
        "demo_vp_0deg": demo_vp_0deg,
        "demo_vp_45deg": demo_vp_45deg,
        "demo_th_0deg": demo_th_0deg,
        "demo_th_45deg": demo_th_45deg,
    }


def _build_plane_wave_setup_objects(t_src: dict) -> tuple[SolverFIT3D, CurrentSource, TFSFSource, DFTMonitor, list[dict]]:
    """Construct the actual Wakis solver, source, and monitor objects (without running time steps)."""
    cpw = int(t_src["cpw"])
    dz = LAM0 / cpw
    n_pml = 10
    z_um = np.asarray(t_src["z_um"], dtype=float)
    nz = len(z_um)
    x = np.array([0.0, dz])
    y = np.array([0.0, dz])
    z = np.arange(nz + 1) * dz

    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    pulse = GaussianSource(F0, fwidth=0.35 * F0, cutoff=4.5)
    z_src = float(t_src["z_src_um"]) * 1e-6
    src = CurrentSource(
        pulse,
        component="x",
        center=(0.5 * dz, 0.5 * dz, z_src),
        size=(dz, dz, 0.0),
        amplitude=1.0,
    )
    z_tf_lo = float(t_src["z_tf_lo_um"]) * 1e-6
    z_tf_hi = float(t_src["z_tf_hi_um"]) * 1e-6
    tfsf = TFSFSource(
        pulse,
        center=(0.5 * dz, 0.5 * dz, 0.5 * (z_tf_lo + z_tf_hi)),
        size=(None, None, z_tf_hi - z_tf_lo),
        direction="+z",
        polarization="x",
        amplitude=1.0,
    )
    freqs = np.asarray(t_src["freqs"], dtype=float)
    z1 = float(t_src["z1_um"]) * 1e-6
    z2 = float(t_src["z2_um"]) * 1e-6
    mon_line = DFTMonitor(
        freqs,
        ["Ex", "Hy"],
        center=(0.5 * dz, 0.5 * dz, 0.5 * (z1 + z2)),
        size=(0.0, 0.0, z2 - z1),
    )
    probes = [
        {"pos": (0.5 * dz, 0.5 * dz, z1), "label": f"Probe z₁ ({z1 * 1e6:.2f} µm)"},
        {"pos": (0.5 * dz, 0.5 * dz, z2), "label": f"Probe z₂ ({z2 * 1e6:.2f} µm)"},
    ]
    return solver, src, tfsf, mon_line, probes


def _save_plane_wave_gif(
    solver: SolverFIT3D,
    src: CurrentSource,
    probes: list[dict],
    z_um: np.ndarray,
    z_src_um: float,
    frames_e: np.ndarray,
    frames_h: np.ndarray,
    times: np.ndarray,
    out_path: Path,
    fps: int = 10,
) -> None:
    """Render an animated GIF of a 1D plane wave pulse with figstyle and geometry overlays."""
    if frames_e.size == 0:
        return
    vmax = max(float(np.max(np.abs(frames_e))) * 1.12, 1e-12)
    pil_frames = []
    with figstyle.apply_style():
        for idx in range(len(frames_e)):
            fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=130)
            st0 = figstyle.series_style(0, with_marker=False)
            st1 = figstyle.series_style(1, with_marker=False)
            ax.plot(z_um, frames_e[idx], label=r"$E_x(z, t)$ [V/m]", **st0)
            ax.plot(z_um, frames_h[idx], label=r"$\eta_0 H_y(z, t)$ [V/m]", **st1)
            schematic.overlay_geometry(
                ax,
                solver,
                plane="zx",
                unit="µm",
                orientation="1d",
                sources=[src],
                probes=probes,
                show_boundaries=True,
                show_materials=True,
                show_sources=True,
                show_monitors=True,
            )
            ax.set(
                xlim=(float(z_um[0]), float(z_um[-1])),
                ylim=(-vmax, vmax),
                xlabel="Position z [µm]",
                ylabel="Field amplitude [V/m]",
                title=f"1D plane wave from soft CurrentSource (t = {times[idx] * 1e15:.1f} fs)",
            )
            figstyle.spectrum_grid(ax)
            ax.legend(loc="lower right", frameon=False, fontsize=9.0)
            figstyle.finish(fig, auto_legend_below=False)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            pil_frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE))
            plt.close(fig)

    if pil_frames:
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )


def render_all_figures(data_dir: Path, fig_dir: Path) -> list[Path]:
    """Read ``data_dir/task_*.npz`` and render all tutorial figures into ``fig_dir``."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tasks_data = {}
    for p in sorted(data_dir.glob("task_*.npz")):
        d = dict(np.load(p, allow_pickle=False))
        tasks_data[str(d["name"])] = d

    required = ("soft_source_and_tfsf", "courant_stability")
    for req in required:
        if req not in tasks_data:
            raise FileNotFoundError(
                f"Missing task {req!r} in {data_dir}. Run --quick or --full first."
            )

    t_src = tasks_data["soft_source_and_tfsf"]
    t_cfl = tasks_data["courant_stability"]
    disp_tasks = sorted(
        [d for name, d in tasks_data.items() if name.startswith("dispersion_")],
        key=lambda x: int(x["cpw"]),
    )
    if not disp_tasks:
        raise FileNotFoundError(f"No dispersion_* tasks found in {data_dir}.")

    solver, src, tfsf, mon_line, probes = _build_plane_wave_setup_objects(t_src)
    written: list[Path] = []

    with figstyle.apply_style():
        # ------------------------------------------------------------------
        # Fig 00a & 00b: Setup schematic (Figure A) + Mesh & excitation (Figure B) + table
        # ------------------------------------------------------------------
        old_p0 = fig_dir / "00_setup_schematic.png"
        if old_p0.exists():
            old_p0.unlink()

        setup_res = schematic.save_setup_figures(
            solver,
            fig_dir,
            plane="zx",
            unit="µm",
            sources=[src, tfsf],
            monitors=[mon_line],
            probes=probes,
            wavelength=LAM0,
            title_geometry="Tutorial 1.1 — Setup: geometry, sources, monitors & boundaries",
            title_mesh="Tutorial 1.1 — Discretization and excitation",
            variations=[
                {
                    "name": "Soft-source run (§1.2)",
                    "details": "CurrentSource J_x sheet at z = 1.50 µm; bilateral ±z radiation",
                },
                {
                    "name": "TFSF run (§1.2)",
                    "details": "TFSFSource box z ∈ [1.12 µm, 3.38 µm]; unilateral +z injection",
                },
                {
                    "name": "Dispersion sweep (§1.3)",
                    "details": "cpw ∈ {10, 20, 40, 80} cells/λ₀; periodic z boundary",
                },
                {
                    "name": "Courant sweep (§1.4)",
                    "details": "CFL S ∈ {0.50, 0.99, 1.005, 1.02}",
                },
            ],
        )
        written.append(setup_res["geometry_figure"])
        written.append(setup_res["mesh_figure"])

        # ------------------------------------------------------------------
        # Fig 01: Setup schematic + GaussianSource pulse & Fourier spectrum
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(figstyle.FULL_WIDTH_IN, 6.8), dpi=figstyle.DEFAULT_DPI)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=1.22, wspace=0.34)
        ax_top = fig.add_subplot(gs[0, :])
        ax_b = fig.add_subplot(gs[1, 0])
        ax_c = fig.add_subplot(gs[1, 1])

        schematic.plot_schematic(
            solver,
            plane="zx",
            ax=ax_top,
            unit="µm",
            sources=[src],
            monitors=[mon_line],
            probes=probes,
            wavelength=LAM0,
            title="(a) Simulation box, boundaries, soft CurrentSource sheet & probes",
        )

        # Panel (b): GaussianSource time waveform & running integral (zero DC charge)
        t_fs = t_src["t_arr"] * 1e15
        s_t = t_src["s_t"]
        dt_val = float(t_src["dt"])
        cum_q = np.cumsum(s_t) * dt_val * F0
        m_pulse = t_fs <= 38.0
        ax_b.plot(
            t_fs[m_pulse],
            s_t[m_pulse],
            label=r"Waveform $s(t)$",
            **figstyle.series_style(0, with_marker=False),
        )
        ax_b.plot(
            t_fs[m_pulse],
            cum_q[m_pulse],
            label=r"$f_0 \int_0^t s(\tau)\,\mathrm{d}\tau \to 0$",
            **figstyle.series_style(1, with_marker=False),
        )
        ax_b.axhline(0.0, color="#64748B", ls=":", lw=0.9)
        ax_b.set(
            xlabel="Time t [fs]",
            ylabel="Normalized amplitude",
            title=r"(b) Zero-DC pulse ($\int s(t)\,\mathrm{d}t = 0$)",
        )
        figstyle.spectrum_grid(ax_b)
        ax_b.legend(loc="upper right", frameon=False, fontsize=9.0)

        # Panel (c): Analytic Fourier transform S(f) vs discrete-time DFT
        f_thz = t_src["freqs"] * 1e-12
        s_f_abs = np.abs(t_src["s_f"]) * 1e15
        s_num_abs = np.abs(t_src["s_dft_num"]) * 1e15
        ax_c.plot(
            f_thz,
            s_f_abs,
            label="Analytic S(f)",
            **figstyle.series_style(0, with_marker=False),
        )
        ax_c.plot(
            f_thz[::2],
            s_num_abs[::2],
            linestyle="None",
            marker=figstyle.MARKERS[1],
            color=figstyle.PALETTE[1],
            markersize=5.5,
            markerfacecolor="none",
            markeredgewidth=1.4,
            label="Discrete Riemann DFT",
        )
        ax_c.set(
            xlabel="Frequency f [THz]",
            ylabel="Spectrum |S(f)| [fs]",
            title="(c) Source spectrum S(f)",
        )
        figstyle.spectrum_grid(ax_c)
        ax_c.legend(loc="lower center", frameon=False, fontsize=9.0)

        p1 = fig_dir / "01_fit_grid_and_source_pulse.png"
        figstyle.save(fig, p1, close=True, auto_legend_below=False)
        written.append(p1)

        # ------------------------------------------------------------------
        # Fig 02: Soft CurrentSource radiation, DFT normalization & TFSFSource
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(figstyle.FULL_WIDTH_IN, 5.8), dpi=figstyle.DEFAULT_DPI)
        gs2 = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.52, wspace=0.34)
        ax0 = fig.add_subplot(gs2[0, :])
        ax1 = fig.add_subplot(gs2[1, 0])
        ax2 = fig.add_subplot(gs2[1, 1])

        wf = t_src["wf_frames"]
        wf_t_fs = t_src["wf_times"] * 1e15
        z_um = t_src["z_um"]
        vmin_wf, vmax_wf = figstyle.symmetric_limits(wf, percentile=99.0)
        im = ax0.pcolormesh(
            z_um,
            wf_t_fs,
            wf,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=vmin_wf,
            vmax=vmax_wf,
            shading="auto",
        )
        schematic.overlay_geometry(
            ax0,
            solver,
            plane="zx",
            unit="µm",
            orientation="spacetime",
            sources=[src],
            probes=probes,
            show_boundaries=True,
            show_materials=True,
            show_sources=True,
            show_monitors=True,
        )
        ax0.set(
            xlabel="Position z [µm]",
            ylabel="Time t [fs]",
            title="(a) Space-time wave propagation Ex(z, t) with CPML & probe overlays",
        )
        fig.colorbar(im, ax=ax0, label=r"$E_x(z, t)$ [V/m]")

        # Panel (b): Raw DFT vs Source-normalized |E_x(z, f) / S(f)|
        f_thz = t_src["freqs"] * 1e-12
        raw_scaled = t_src["amp_raw"] / np.max(t_src["amp_raw"]) * (0.5 * ETA0)
        ax1.plot(
            f_thz,
            raw_scaled,
            color="#64748B",
            ls="--",
            lw=1.5,
            label="Unnormalized |Ex(f)|",
        )
        ax1.plot(
            f_thz,
            t_src["amp_th"],
            label="Exact discrete FIT",
            **figstyle.series_style(0, with_marker=False),
        )
        ax1.plot(
            f_thz[::2],
            t_src["amp_sim"][::2],
            linestyle="None",
            marker=figstyle.MARKERS[1],
            color=figstyle.PALETTE[1],
            markersize=5.2,
            label="Normalized |Ex/S(f)|",
        )
        ax1.axhline(0.5 * ETA0, color=figstyle.PALETTE[2], ls=":", lw=1.3, label=r"$\frac{1}{2}\eta_0 = 188.4\;\Omega$")
        ax1.set(
            xlabel="Frequency f [THz]",
            ylabel="|Ex(f) / S(f)| [Ω]",
            ylim=(0.0, 235.0),
            title="(b) Source-normalized impedance",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="lower center", frameon=False, fontsize=8.8)

        # Panel (c): TFSFSource one-way plane-wave box & machine-epsilon cancellation
        prof_max = np.maximum(t_src["tfsf_profile_max"], 1e-17)
        ax2.semilogy(
            z_um,
            prof_max,
            label=r"$\max_t |E_x(z, t)|$",
            **figstyle.series_style(0, with_marker=False),
        )
        ax2.axvline(float(t_src["z_tf_lo_um"]), color=figstyle.PALETTE[1], ls="--", lw=1.4, label="TFSF box faces")
        ax2.axvline(float(t_src["z_tf_hi_um"]), color=figstyle.PALETTE[1], ls="--", lw=1.4)
        ax2.axvspan(
            float(t_src["z_tf_lo_um"]),
            float(t_src["z_tf_hi_um"]),
            color="#DBEAFE",
            alpha=0.5,
            label="Total-field (TF) box",
        )
        schematic.overlay_geometry(
            ax2,
            solver,
            plane="zx",
            unit="µm",
            orientation="1d",
            show_boundaries=True,
            show_materials=False,
        )
        ax2.set(
            xlabel="Position z [µm]",
            ylabel="Peak |Ex(z, t)| [V/m]",
            ylim=(1e-17, 8.0),
            title=f"(c) TFSFSource leakage = {float(t_src['tfsf_leakage']):.1e}",
        )
        figstyle.spectrum_grid(ax2)
        ax2.legend(loc="center right", frameon=False, fontsize=8.8)

        p2 = fig_dir / "02_soft_source_and_normalization.png"
        figstyle.save(fig, p2, close=True)
        written.append(p2)

        # ------------------------------------------------------------------
        # Fig 03: Courant-Friedrichs-Lewy (CFL) stability limit & blow-up
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(figstyle.FULL_WIDTH_IN, 5.6), dpi=figstyle.DEFAULT_DPI)
        gs3 = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.52, wspace=0.34)
        ax0 = fig.add_subplot(gs3[0, :])
        ax1 = fig.add_subplot(gs3[1, 0])
        ax2 = fig.add_subplot(gs3[1, 1])

        s_list = t_cfl["s_list"]
        histories = t_cfl["histories"]
        plot_idx = 0
        for idx_s, s_val in enumerate(s_list):
            if s_val not in (0.50, 0.99, 1.005, 1.02, 1.05):
                continue
            h_row = histories[idx_s]
            valid = ~np.isnan(h_row) & (h_row > 0)
            steps_x = np.arange(len(h_row))[valid]
            status = "stable" if s_val <= 1.0 else "BLOW-UP"
            st = figstyle.series_style(plot_idx, markevery=max(1, len(steps_x) // 8))
            ax0.semilogy(
                steps_x,
                h_row[valid],
                label=f"S = {s_val:.3f} ({status})",
                **st,
            )
            plot_idx += 1
        ax0.set(
            xlabel="Leapfrog time step n",
            ylabel=r"Peak field $\max_{\mathbf{r}} |E_x(\mathbf{r}, t_n)|$ [V/m]",
            ylim=(1e-1, 1e18),
            title="(a) Subcritical stability (S ≤ 1) vs. supercritical blow-up (S > 1)",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="upper left", ncol=3, frameon=False, fontsize=8.8)

        # Panel (b): Per-step growth factor |lambda_sim(S)| vs exact von Neumann root
        s_dense = np.linspace(0.40, 1.06, 300)
        ax1.plot(
            s_dense,
            courant_amplification_factor(s_dense),
            label=r"Exact $|\lambda_{\max}(S)|$",
            **figstyle.series_style(0, with_marker=False),
        )
        ax1.plot(
            s_list,
            t_cfl["g_sim"],
            linestyle="None",
            marker=figstyle.MARKERS[1],
            color=figstyle.PALETTE[1],
            markersize=6.0,
            label=r"3D FIT $|\lambda_{\mathrm{sim}}|$",
        )
        ax1.axvline(1.0, color=figstyle.PALETTE[4], ls=":", lw=1.5, label="Courant limit S = 1")
        ax1.set(
            xlabel="3D Courant number S",
            ylabel=r"Growth factor $|\lambda|$",
            xlim=(0.45, 1.06),
            ylim=(0.95, 2.00),
            title="(b) Von Neumann growth factor",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="upper left", frameon=False, fontsize=8.8)

        # Panel (c): Spatial Nyquist checkerboard mode at S = 1.02
        prof_z = t_cfl["unstable_profile_z"]
        k_idx = np.arange(len(prof_z))
        ax2.plot(
            k_idx,
            prof_z,
            label=r"$E_x(z_k)$ at $S = 1.02$",
            **figstyle.series_style(1, markevery=1),
        )
        ax2.axhline(0.0, color="#64748B", ls=":", lw=1.0)
        ax2.set(
            xlabel="Grid cell index k along z",
            ylabel="Normalized Ex(zk)",
            ylim=(-1.18, 1.18),
            title=r"(c) Nyquist checkerboard ($\lambda_{\mathrm{Nyq}} = 2\Delta z$)",
        )
        figstyle.spectrum_grid(ax2)
        ax2.legend(loc="upper right", frameon=False, fontsize=8.8)

        p3 = fig_dir / "03_courant_stability_limit.png"
        figstyle.save(fig, p3, close=True)
        written.append(p3)

        # ------------------------------------------------------------------
        # Fig 04: Numerical dispersion in 1D (vs frequency) and 2D (vs angle)
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(figstyle.FULL_WIDTH_IN, 5.8), dpi=figstyle.DEFAULT_DPI)
        gs4 = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.52, wspace=0.34)
        ax0 = fig.add_subplot(gs4[0, 0])
        ax1 = fig.add_subplot(gs4[0, 1])
        ax2 = fig.add_subplot(gs4[1, :])

        for idx_d, d_task in enumerate(disp_tasks[:4]):
            cpw_v = int(d_task["cpw"])
            st = figstyle.series_style(idx_d, with_marker=False)
            f_norm = d_task["freqs"] / F0
            ax0.plot(f_norm, d_task["vp_th_1d"], label=f"{cpw_v} cpw", **st)
            ax0.plot(
                f_norm[::3],
                d_task["vp_sim_1d"][::3],
                linestyle="None",
                marker=figstyle.MARKERS[idx_d],
                color=figstyle.PALETTE[idx_d],
                markersize=4.8,
                markerfacecolor="none",
                markeredgewidth=1.3,
            )
        ax0.axhline(1.0, color="#64748B", ls=":", lw=1.0)
        ax0.set(
            xlabel=r"Normalized frequency $f / f_0$",
            ylabel=r"Phase velocity $v_p(f) / c_0$",
            title="(a) 1D dispersion vs. frequency",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="lower left", frameon=False, fontsize=8.8)

        # Panel (b): 2D anisotropic phase velocity v_p(cpw, phi) / c_0 vs angle phi
        for idx_d, d_task in enumerate(disp_tasks[:4]):
            cpw_v = int(d_task["cpw"])
            st = figstyle.series_style(idx_d, with_marker=False)
            phis = d_task["phis_deg"]
            ax1.plot(phis, d_task["vp_th_2d"], label=f"{cpw_v} cpw", **st)
            ax1.plot(
                phis[::3],
                d_task["vp_cyl_2d"][::3],
                linestyle="None",
                marker=figstyle.MARKERS[idx_d],
                color=figstyle.PALETTE[idx_d],
                markersize=4.8,
                markerfacecolor="none",
                markeredgewidth=1.3,
            )
        ax1.axhline(1.0, color="#64748B", ls=":", lw=1.0)
        ax1.set(
            xlabel="Propagation angle φ [deg]",
            ylabel=r"Phase velocity $v_p(\phi) / c_0$",
            xlim=(0.0, 90.0),
            title="(b) 2D grid anisotropy vs. angle φ",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="lower center", ncol=2, frameon=False, fontsize=8.8)

        # Panel (c): 2D cylindrical wavefront snapshot Re[E_z(x, y, f_0)] with geometry overlay
        snap_task = next(
            (d for d in disp_tasks if d["ez_re"].size > 0),
            disp_tasks[0],
        )
        if snap_task["ez_re"].size > 0:
            ex_u = snap_task["ez_x_um"]
            ey_u = snap_task["ez_y_um"]
            ez_r = snap_task["ez_re"]
            ez_scale = max(float(np.percentile(np.abs(ez_r), 98.0)), 1e-18)
            ez_norm = np.clip(ez_r / ez_scale, -1.0, 1.0)
            im2 = ax2.pcolormesh(
                ex_u,
                ey_u,
                ez_norm.T,
                cmap=figstyle.CMAP_DIVERGING,
                vmin=-1.0,
                vmax=1.0,
                shading="auto",
            )
            theta_c = np.linspace(0.0, 2.0 * np.pi, 200)
            for rad_um in (1.0, 2.0):
                ax2.plot(
                    rad_um * np.cos(theta_c),
                    rad_um * np.sin(theta_c),
                    "k--",
                    lw=1.0,
                    marker="none",
                    alpha=0.7,
                )
            ax2.plot(0.0, 0.0, marker="*", linestyle="None", markersize=8.5, color="#D55E00", zorder=7)
            ax2.text(
                0.12,
                0.12,
                "Jz point source",
                fontsize=8.8,
                fontweight="bold",
                color=figstyle.INK,
                zorder=8,
            )
            ax2.text(
                0.02,
                0.94,
                "±x, ±y: CPML | Vacuum (εr = 1)",
                transform=ax2.transAxes,
                fontsize=8.6,
                fontweight="bold",
                color=figstyle.INK,
                va="top",
                ha="left",
                zorder=8,
            )
            fig.colorbar(im2, ax=ax2, label=r"Normalized $\mathrm{Re}[E_z(x, y, f_0)]$")
        ax2.set(
            xlabel="x [µm]",
            ylabel="y [µm]",
            aspect="equal",
            title=f"(c) 2D cylindrical wave ({int(snap_task['cpw'])} cpw) vs. exact isotropic wavefront rings",
        )

        p4 = fig_dir / "04_numerical_dispersion_1d_and_2d.png"
        figstyle.save(fig, p4, close=True)
        written.append(p4)

        # ------------------------------------------------------------------
        # Fig 05: Resolution convergence O(cpw^-2) and the 20 cpw rule
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN, height_in=3.1, ncols=2, nrows=1
        )

        cpw_arr = np.array([int(d["cpw"]) for d in disp_tasks], dtype=float)
        err_0_sim = np.array([abs(1.0 - float(d["vp_0deg_sim"])) for d in disp_tasks], dtype=float)
        err_45_sim = np.array([abs(1.0 - float(d["vp_45deg_sim"])) for d in disp_tasks], dtype=float)
        err_0_th = np.array([abs(1.0 - float(d["vp_0deg_th"])) for d in disp_tasks], dtype=float)
        err_45_th = np.array([abs(1.0 - float(d["vp_45deg_th"])) for d in disp_tasks], dtype=float)

        slope_0 = float(np.polyfit(np.log(cpw_arr), np.log(err_0_sim), 1)[0])
        slope_45 = float(np.polyfit(np.log(cpw_arr), np.log(err_45_sim), 1)[0])

        ax0.loglog(
            cpw_arr,
            err_0_th * 100.0,
            label=rf"$\phi = 0^\circ$ (slope {slope_0:.2f})",
            **figstyle.series_style(0),
        )
        ax0.loglog(
            cpw_arr,
            err_45_th * 100.0,
            label=rf"$\phi = 45^\circ$ (slope {slope_45:.2f})",
            **figstyle.series_style(1),
        )

        ref_t = disp_tasks[-1]
        if int(ref_t.get("demo_cpw", 0)) > 0:
            d_cpw = float(ref_t["demo_cpw"])
            d_e0 = abs(1.0 - float(ref_t["demo_vp_0deg"])) * 100.0
            ax0.loglog(
                [d_cpw],
                [d_e0],
                linestyle="None",
                marker="X",
                color=figstyle.PALETTE[4],
                markersize=7.5,
                label=f"Under-resolved ({int(d_cpw)} cpw)",
            )

        ax0.axvline(20.0, color=figstyle.PALETTE[2], ls=":", lw=1.4, label="20 cpw rule (~0.13%)")
        ax0.set_xticks([6, 10, 14, 20, 28, 40, 60])
        ax0.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax0.get_xaxis().set_minor_formatter(plt.NullFormatter())
        ax0.set(
            xlabel="Resolution [cells / λ₀]",
            ylabel=r"Error $|1 - v_p / c_0|$ [%]",
            title=r"(a) $\mathcal{O}(\mathrm{cpw}^{-2})$ convergence",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="upper right", frameon=False, fontsize=8.8)

        # Panel (b): Cumulative phase error Delta_phi(L) vs propagation distance L / lambda_0
        l_over_lam = np.linspace(0.0, 25.0, 200)
        for idx_d, d_task in enumerate(disp_tasks[:4]):
            cpw_v = int(d_task["cpw"])
            vp_0 = float(d_task["vp_0deg_th"])
            dphi_deg = 360.0 * l_over_lam * abs(1.0 / vp_0 - 1.0)
            ax1.plot(
                l_over_lam,
                dphi_deg,
                label=f"{cpw_v} cpw ({abs(1.0 - vp_0) * 100:.2f}%)",
                **figstyle.series_style(idx_d, markevery=40),
            )
        ax1.axhline(10.0, color=figstyle.PALETTE[4], ls=":", lw=1.3, label="10° phase budget")
        ax1.set(
            xlabel=r"Distance $L / \lambda_0$ [wavelengths]",
            ylabel=r"Cumulative phase lag $\Delta\phi$ [deg]",
            xlim=(0.0, 25.0),
            ylim=(0.0, 65.0),
            title="(b) Phase lag over multi-λ domains",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="upper left", frameon=False, fontsize=8.8)

        p5 = fig_dir / "05_resolution_convergence_20cpw_rule.png"
        figstyle.save(fig, p5, close=True)
        written.append(p5)

        # ------------------------------------------------------------------
        # Fig 06: Animated time-domain GIF
        # ------------------------------------------------------------------
        p6 = fig_dir / "06_plane_wave_propagation.gif"
        _save_plane_wave_gif(
            solver,
            src,
            probes,
            t_src["z_um"],
            float(t_src["z_src_um"]),
            t_src["movie_frames_e"],
            t_src["movie_frames_h"],
            t_src["movie_times"],
            p6,
        )
        written.append(p6)

    return written
