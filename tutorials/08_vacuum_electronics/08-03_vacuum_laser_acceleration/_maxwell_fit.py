# Copyright 2026 Wakis contributors
# SPDX-License-Identifier: Apache-2.0

"""3D FIT Maxwellian measurements for Tutorial 8.3 (Parts B and C).

Implements the numerical simulations and analytical benchmarks corresponding to
M. S. Zolotorev, S. Chattopadhyay, and K. T. McDonald, *"A Maxwellian Perspective
on Particle Acceleration"*, unpublished draft (Feb. 1998, updated Sep. 1999),
http://kirkmcd.princeton.edu/accel/vacuumaccel.pdf :

1. **Part B — Phase and group velocity of a focus** (Sec. 5.1.1–5.1.2, Eqs. 30–35):
   - 3D FIT focused ``TEM00`` pulse for paraxial waists ``w_0 = 2 * lambda_L`` and
     ``3 * lambda_L`` compared against a plane-wave run on the *same* 3D FIT grid
     to cancel numerical dispersion.
   - On-axis Gouy phase shift ``-arctan(z / z_0)`` (``90 deg`` from focus to far field,
     half within ``+/- z_0``).
   - Superluminal phase velocity ``v_p(\rho) > c`` inside the waist (Eq. 33).
   - Subluminal group velocity ``v_g(\rho) < c`` tracked via the Hilbert envelope peak
     for a fixed-waist pulse (Eq. 34) and slice energy-flow velocity
     ``v_flow(z) = int S_z dA / int u dA`` vs ``c / (1 + theta_0^2 / 4)`` (Eq. 35).

2. **Part B — Weak-field quadrature and the optical theorem** (Sec. 5.3, Eqs. 59–63):
   - 3D FIT focused ``TEM00`` laser beam and a point electric dipole at the focus
     (``wakis.sources.Dipole`` driven by ``d^2 d / dt^2 = (e^2 / m_e) E_L(0, t)``),
     both projected to the far sphere via ``wakis.near2far.Near2FarMonitor``.
   - Demonstrates the ``90 deg`` Gouy phase quadrature within ``theta <= theta_0``,
     vanishing first-order cross integral ``int E_L . E_rad dA dt = 0``, and the
     connection to the optical theorem (real polarizability ``alpha = -e^2 / (m_e omega^2)``,
     with extinction equal to second-order Thomson scattering ``sigma_T``).

3. **Part C.1 — Maxwellian cross-energy check of the PEC screen** (Sec. 2 & 5.2.6, Eqs. 53–58):
   - Three linear 3D FIT simulations (laser only, rigid relativistic ``wakis.sources.Beam``
     at ``gamma = 20`` emitting transition radiation at the screen, and laser + beam)
     with and without the PEC screen.
   - Verifies ``-U_int = Delta U_e`` (stored cross energy + boundary cross Poynting flux
     equals the direct path integral) with the screen and ``U_int ~ 0`` without the screen,
     and maps the angular cross-flux density on the far sphere (``theta ~ theta_0 ~ 1/gamma``).

4. **Part C.2 & C.3 — Two-boundary "laser cavity" and aperture cutoff** (Sec. 5.5–5.6, Eqs. 71–77, 84):
   - On-axis radially polarized cylindrical ``(0, 0)`` / ``TM01`` mode (Eqs. 71–72):
     vanishing infinite-path integral for all phases (Sec. 5.5, Lawson–Woodward),
     analytical cavity gain ``Delta U_e(L) = 2 e E_{0,z} z_0 * L z_0 / (L^2 + z_0^2)``
     (Eq. 76, maximum ``e E_{0,z} z_0 = e E_0 w_0`` at ``L = z_0``, Eq. 77), and
     exact finite-``gamma`` phase-slippage path integral.
   - 3D FIT sweep over cavity half-length ``L / z_0 in {0.25, 0.5, 1.0, 2.0, 4.0}`` with
     two perforated PEC screens at ``+/- L``.
   - 3D FIT aperture-radius sweep over ``a / (gamma * lambda_bar)`` at fixed ``L = z_0``
     for ``gamma = 3`` and ``gamma = 8`` (with ``>= 4`` cells across ``a`` and ``>= 20``
     cells/``lambda_L``), demonstrating universal collapse onto ``1 / I_0(a / (beta * gamma * lambda_bar))``
     and the evanescent cutoff ``a <= gamma * lambda_bar`` (Eq. 84).
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0, m_e, mu_0
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import hilbert
from scipy.special import i0 as bessel_i0

from wakis import GridFIT3D, SolverFIT3D
from wakis.estimate import estimate, preflight
from wakis.field_monitors import FluxMonitor
from wakis.near2far import Near2FarMonitor
from wakis.schematic import overlay_geometry, save_setup_figures
from wakis.sources import Beam, CurrentSource, CustomSource, Dipole, GaussianSource

try:
    from ._helpers import (
        DATA_DIR,
        DEFAULT_LAMBDA_L_M,
        DEFAULT_MEM_LIMIT_MB,
        FIG_DIR,
        Z0_VAC,
        _apply_paper_rc,
        build_graded_laser_grid,
        calibrate_fit_beam_phasors,
        crossed_gaussian_energy_gain_theory,
        evaluate_fit_trajectory_gain,
        paraxial_gaussian_beam_fields,
    )
except ImportError:
    from _helpers import (
        DATA_DIR,
        DEFAULT_LAMBDA_L_M,
        DEFAULT_MEM_LIMIT_MB,
        FIG_DIR,
        Z0_VAC,
        _apply_paper_rc,
        build_graded_laser_grid,
        calibrate_fit_beam_phasors,
        crossed_gaussian_energy_gain_theory,
        evaluate_fit_trajectory_gain,
        paraxial_gaussian_beam_fields,
    )

R_E_M: float = float(q_e**2 / (4.0 * np.pi * epsilon_0 * m_e * c_light**2))
SIGMA_THOMSON_M2: float = float((8.0 * np.pi / 3.0) * (R_E_M**2))


def _copy_setup_pair(
    setup_res: dict,
    part_tag: str,
) -> tuple[Path, Path]:
    """Move ``save_setup_figures`` outputs to ``00a_<part>_setup.png`` and ``00b_<part>_mesh.png``."""
    src_a = Path(setup_res["geometry_figure"])
    src_b = Path(setup_res["mesh_figure"])
    dst_a = FIG_DIR / f"00a_{part_tag}_setup.png"
    dst_b = FIG_DIR / f"00b_{part_tag}_mesh.png"
    if src_a.exists() and src_a != dst_a:
        src_a.replace(dst_a)
    if src_b.exists() and src_b != dst_b:
        src_b.replace(dst_b)
    return dst_a, dst_b


# ---------------------------------------------------------------------------
# Part B.1: Gouy phase, superluminal phase velocity, and subluminal group/energy velocity
# ---------------------------------------------------------------------------


def paraxial_focus_velocity_theory(
    rho_over_w0: float | np.ndarray,
    z_over_z0: float | np.ndarray,
    w0_over_lambda: float,
) -> dict[str, np.ndarray]:
    """Analytical Gouy phase and focal velocities (Zolotorev, Chattopadhyay & McDonald Eqs. 30–35).

    Parameters
    ----------
    rho_over_w0 : float or ndarray
        Normalized transverse radius ``rho / w_0`` [-].
    z_over_z0 : float or ndarray
        Normalized axial coordinate ``zeta = z / z_0`` [-].
    w0_over_lambda : float
        Waist-to-wavelength ratio ``w_0 / lambda_L`` [-].

    Returns
    -------
    out : dict of ndarray
        Dictionary containing:
        - ``theta0_rad``: diffraction half-angle ``theta_0 = 1 / (pi * (w_0 / lambda_L))`` [rad] (Eq. 21)
        - ``gouy_phase_rad``: on-axis Gouy phase ``-arctan(z / z_0)`` [rad] (Eq. 30)
        - ``keff_over_k``: normalized effective axial wavenumber at waist (Eq. 33)
        - ``vp_over_c``: normalized phase velocity ``v_p(rho) / c = k / k_eff(rho)`` at waist (Eq. 33)
        - ``vg_fixed_waist_over_c``: normalized group velocity ``v_g(rho) / c`` for fixed ``w_0`` (Eq. 34)
        - ``vg_fixed_angle_over_c``: normalized group velocity ``v_g(rho) / c`` for fixed ``theta_0``
        - ``vflow_over_c``: flux-averaged energy-flow velocity ``1 / (1 + theta_0^2 / 4)`` (Eq. 35)
    """
    rho_n = np.asarray(rho_over_w0, dtype=float)
    zeta = np.asarray(z_over_z0, dtype=float)
    theta0 = 1.0 / (np.pi * float(w0_over_lambda))
    th2_half = 0.5 * (theta0**2)

    gouy_rad = -np.arctan(zeta)
    keff_over_k = 1.0 - th2_half * (1.0 - rho_n**2)
    vp_over_c = 1.0 / keff_over_k
    vg_fixed_w0 = 1.0 / (1.0 + th2_half * (1.0 - rho_n**2))
    vg_fixed_th0 = 1.0 / (1.0 - th2_half * (1.0 - 3.0 * (rho_n**2)))
    vflow_over_c = 1.0 / (1.0 + 0.25 * (theta0**2))

    return {
        "theta0_rad": np.array(theta0),
        "gouy_phase_rad": gouy_rad,
        "keff_over_k": keff_over_k,
        "vp_over_c": vp_over_c,
        "vg_fixed_waist_over_c": vg_fixed_w0,
        "vg_fixed_angle_over_c": vg_fixed_th0,
        "vflow_over_c": np.array(vflow_over_c),
    }


def _build_focus_velocity_grid(
    w0_m: float,
    lambda_m: float,
    ppw_z: int,
    ppw_xy_core: int,
    n_pml: int,
    z_span_lambda: float = 5.6,
) -> tuple[GridFIT3D, dict]:
    """Build a graded 3D FIT grid for focal phase/group/energy-flow velocity measurements."""
    dz = float(lambda_m) / int(ppw_z)
    dx_core = float(lambda_m) / int(ppw_xy_core)

    # Uniform focal core at >= ppw_xy_core cells/lambda, smoothly graded outer region to 2.05 w0 + CPML
    n_core_half = min(14, int(np.ceil(0.85 * float(w0_m) / dx_core)))
    x_core = np.arange(-n_core_half * dx_core, (n_core_half + 0.5) * dx_core, dx_core)
    n_pre = 10
    n_wing = n_pre + int(n_pml)
    target_pre = max(2.05 * float(w0_m) - float(x_core[-1]), n_pre * dx_core)
    rs = np.linspace(1.02, 1.35, 331)
    sums = [dx_core * np.sum(r ** np.arange(1, n_pre + 1)) for r in rs]
    r_opt = float(rs[np.argmin(np.abs(np.array(sums) - target_pre))])
    wing = np.cumsum(dx_core * (r_opt ** np.arange(1, n_wing + 1)))
    x_nodes = np.concatenate([-x_core[-1] - wing[::-1], x_core, x_core[-1] + wing])

    nz_half = int(np.ceil(0.5 * float(z_span_lambda) * float(lambda_m) / dz)) + int(n_pml)
    z_nodes = np.arange(-nz_half * dz, (nz_half + 0.5) * dz, dz)

    grid = GridFIT3D(x=x_nodes, y=x_nodes, z=z_nodes, verbose=0)
    meta = {
        "dz_m": dz,
        "dx_core_m": dx_core,
        "ppw_z": int(ppw_z),
        "ppw_xy_core": int(ppw_xy_core),
        "n_pml": int(n_pml),
    }
    return grid, meta


def _run_focus_and_planewave_pair(
    w0_over_lambda: float,
    lambda_m: float = DEFAULT_LAMBDA_L_M,
    ppw_z: int = 20,
    ppw_xy_core: int = 20,
    n_pml: int = 6,
    save_setup: bool = False,
) -> dict:
    """Run 3D FIT focused TEM00 pulse and plane-wave reference on the SAME grid."""
    t0_wall = time.time()
    w0_m = float(w0_over_lambda) * float(lambda_m)
    k0 = 2.0 * np.pi / float(lambda_m)
    omega0 = k0 * c_light
    f0 = c_light / float(lambda_m)
    z0_m = 0.5 * k0 * (w0_m**2)
    theta0 = 2.0 / (k0 * w0_m)

    grid, meta = _build_focus_velocity_grid(
        w0_m=w0_m,
        lambda_m=lambda_m,
        ppw_z=ppw_z,
        ppw_xy_core=ppw_xy_core,
        n_pml=n_pml,
        z_span_lambda=5.2,
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml"] * 3,
        bc_high=["cpml"] * 3,
        n_pml=n_pml,
        cfln=0.92,
        verbose=0,
    )
    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    dt = float(solver.dt)
    dz = float(meta["dz_m"])
    t_opt = float(lambda_m) / c_light

    iz_src = int(n_pml) + 2
    z_src = float(grid.z[iz_src])
    iz_in = iz_src + 2
    iz_out = nz - iz_in
    z_ex = np.asarray(grid.z[iz_in:iz_out], dtype=float)
    z_ez = np.asarray(0.5 * (grid.z[:-1] + grid.z[1:])[iz_in:iz_out], dtype=float)
    x_edge = 0.5 * (grid.x[:-1] + grid.x[1:])
    x_node = grid.x[:-1].copy()
    y_node = grid.y[:-1].copy()
    iy0 = ny // 2
    ix0 = int(np.argmin(np.abs(x_edge)))

    t_transit = float(grid.z[nz - 1] - z_src) / c_light
    n_steps = int(np.ceil((1.35 * t_transit + 2.6 * t_opt) / dt))
    est = estimate((nx, ny, nz), steps=n_steps, cpml=True, conformal=False)
    preflight(est, {"ram": DEFAULT_MEM_LIMIT_MB * 1e6, "gpu": 0})

    if save_setup:
        src_obj = CustomSource(
            GaussianSource(frequency=f0, fwidth=0.12 * f0),
            start_time=0.0,
            end_time=n_steps * dt,
        )
        src_obj.frequency = f0
        src_obj.fwidth = 0.12 * f0
        mon_box = FluxMonitor(
            frequencies=[f0],
            center=(0.0, 0.0, 0.0),
            size=(1.6 * w0_m, 1.6 * w0_m, 0.6 * float(z_ex[-1] - z_ex[0])),
        )
        src_sheet_preview = CurrentSource(
            GaussianSource(frequency=f0, fwidth=0.12 * f0),
            component="x",
            center=(0.0, 0.0, z_src),
            size=(2.0 * w0_m, 2.0 * w0_m, 0.0),
        )
        setup_res = save_setup_figures(
            solver,
            FIG_DIR,
            prefix="00",
            suffix="_part_b_gouy",
            plane="xz",
            sources=[src_sheet_preview],
            monitors=[mon_box],
            wavelength=lambda_m,
            freq=f0,
            min_feature=w0_m,
            title_geometry=f"Part B Setup: Focused TEM00 (w0 = {w0_over_lambda:.0f}λ) vs Plane-Wave Reference",
            title_mesh=f"Part B Mesh & Pulse: {ppw_z} cells/λ along z and focal core",
            variations=[
                {"name": "Waist scan", "details": "w0 = 2λ_L and w0 = 3λ_L"},
                {"name": "Dispersion baseline", "details": "Plane wave on identical 3D FIT grid"},
            ],
        )
        _copy_setup_pair(setup_res, "part_b_gouy")

    # Multi-tone fixed-waist excitation: carrier omega_0 and sidebands omega_0 +/- d_omega
    # Having fixed w_0 across all three tones directly tests Eq. 34 (fixed-waist group velocity)
    # and yields both the Hilbert envelope peak in time and continuous-wave phasor at omega_0.
    d_omega = 0.08 * omega0
    omegas = np.array([omega0 - d_omega, omega0, omega0 + d_omega])
    weights_om = np.array([0.28, 1.0, 0.28])

    # Precompute 2D entrance profiles for each tone with discrete dispersion pre-compensation
    n_fft = 96
    l_fft = max(8.0 * w0_m, 1.5 * float(grid.x[-1] - grid.x[0]))
    u_fft = np.linspace(-0.5 * l_fft, 0.5 * l_fft, n_fft, endpoint=False)
    du_fft = float(u_fft[1] - u_fft[0])
    xx_f, yy_f = np.meshgrid(u_fft, u_fft, indexing="ij")
    kx_f = 2.0 * np.pi * np.fft.fftfreq(n_fft, d=du_fft)
    ky_f = 2.0 * np.pi * np.fft.fftfreq(n_fft, d=du_fft)
    kx2_f, ky2_f = np.meshgrid(kx_f**2, ky_f**2, indexing="ij")
    x_s2, y_s2 = np.meshgrid(x_edge, y_node, indexing="ij")
    pts_s2 = np.column_stack([x_s2.ravel(), y_s2.ravel()])

    env_tones = []
    k_num_tones = []
    for om in omegas:
        lam_i = 2.0 * np.pi * c_light / om
        k_i = om / c_light
        sin_a = np.clip((dz / (c_light * dt)) * np.sin(0.5 * om * dt), -1.0, 1.0)
        kn_i = float((2.0 / dz) * np.arcsin(sin_a))
        k_num_tones.append(kn_i)
        f_in = paraxial_gaussian_beam_fields(
            xx_f, yy_f, z_src, w0_m=w0_m, lambda_m=lam_i, p_laser_w=1.0e9, k_axial_rad_m=kn_i
        )
        prop_m = (kx2_f + ky2_f) < (0.85 * k_i) ** 2
        kz_fit = np.sqrt(np.maximum(kn_i**2 - kx2_f - ky2_f, 0.0))
        kz_par = kn_i - 0.5 * (kx2_f + ky2_f) / k_i
        dkz = np.where(prop_m, kz_fit - kz_par, 0.0)
        ex_corr = np.fft.ifft2(np.fft.fft2(f_in["Ex"]) * np.where(prop_m, np.exp(1j * dkz * z_src), 0.0))
        ir = RegularGridInterpolator((u_fft, u_fft), np.real(ex_corr), bounds_error=False, fill_value=0.0)
        ii = RegularGridInterpolator((u_fft, u_fft), np.imag(ex_corr), bounds_error=False, fill_value=0.0)
        env_tones.append((ir(pts_s2) + 1j * ii(pts_s2)).reshape(x_s2.shape))

    k_num0 = float(k_num_tones[1])
    tau_turn = 1.6 * t_opt
    j_scale = -(2.0 / (Z0_VAC * dz))

    # Select radial sample indices inside [0, 1.22 w0]
    rho_mask = (x_edge >= -1e-12) & (x_edge <= 1.22 * w0_m)
    ix_rho = np.where(rho_mask)[0]
    rho_vals = x_edge[ix_rho]

    # Accumulators for the 3 tones on the meridian slice (x, iy0, z) and full 3D slice at z=0
    dft_focus = np.zeros((len(omegas), len(ix_rho), len(z_ex)), dtype=np.complex128)
    iz_mid = int(np.argmin(np.abs(z_ex)))
    iz_mid_grid = iz_in + iz_mid

    # Full-plane phasor accumulators at 5 axial slices around the waist for energy-flow velocity v_flow(z)
    slice_offsets = [-8, -4, 0, 4, 8]
    iz_slices = [max(iz_in + 2, min(iz_out - 3, iz_mid_grid + off)) for off in slice_offsets]
    z_slices_m = np.asarray([float(grid.z[k]) for k in iz_slices])
    ex_sl = np.zeros((nx, ny, len(iz_slices)), dtype=np.complex128)
    ey_sl = np.zeros((nx, ny, len(iz_slices)), dtype=np.complex128)
    ez_sl = np.zeros((nx, ny, len(iz_slices)), dtype=np.complex128)
    hx_sl = np.zeros((nx, ny, len(iz_slices)), dtype=np.complex128)
    hy_sl = np.zeros((nx, ny, len(iz_slices)), dtype=np.complex128)
    hz_sl = np.zeros((nx, ny, len(iz_slices)), dtype=np.complex128)

    n_dft = max(12, int(np.round((2.0 * np.pi / d_omega) / dt)))
    n_start_dft = max(int(np.ceil((1.15 * t_transit + tau_turn) / dt)), n_steps - n_dft)
    n_steps = n_start_dft + n_dft
    count_dft = 0

    for n in range(n_steps):
        t_h = (n + 0.5) * dt
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t_h / tau_turn, 1.0)))
        src_sheet = np.zeros_like(x_s2, dtype=float)
        for w_k, om_k, env_k in zip(weights_om, omegas, env_tones):
            src_sheet += w_k * np.real(env_k * np.exp(-1j * om_k * t_h))
        solver.J[:, :, iz_src, "x"] = j_scale * ramp * src_sheet
        solver.one_step()

        t_e = (n + 1) * dt
        t_b = (n + 0.5) * dt
        if n >= n_start_dft:
            ex_mer = solver.E[ix_rho, iy0, iz_in:iz_out, "x"]
            for m_k, om_k in enumerate(omegas):
                dft_focus[m_k] += ex_mer * np.exp(1j * om_k * t_e)
            ph_e = np.exp(1j * omega0 * t_e)
            ph_h = np.exp(1j * omega0 * t_b)
            for s_i, iz_k in enumerate(iz_slices):
                ex_sl[:, :, s_i] += solver.E[:, :, iz_k, "x"] * ph_e
                ey_sl[:, :, s_i] += solver.E[:, :, iz_k, "y"] * ph_e
                ez_sl[:, :, s_i] += 0.5 * (solver.E[:, :, iz_k - 1, "z"] + solver.E[:, :, iz_k, "z"]) * ph_e
                hx_sl[:, :, s_i] += 0.5 * (solver.H[:, :, iz_k - 1, "x"] + solver.H[:, :, iz_k, "x"]) * ph_h
                hy_sl[:, :, s_i] += 0.5 * (solver.H[:, :, iz_k - 1, "y"] + solver.H[:, :, iz_k, "y"]) * ph_h
                hz_sl[:, :, s_i] += solver.H[:, :, iz_k, "z"] * ph_h
            count_dft += 1

    scale_dft = 2.0 / max(count_dft, 1)
    dft_focus *= scale_dft
    ex_sl *= scale_dft
    ey_sl *= scale_dft
    ez_sl *= scale_dft
    hx_sl *= scale_dft
    hy_sl *= scale_dft
    hz_sl *= scale_dft

    del solver, grid
    gc.collect()

    # Plane-wave reference run on the exact same axial grid and timestep
    # Use a 1D-periodic (pec/pmc) transverse column with the identical dz and dt so
    # numerical dispersion k_pw(omega) and discrete wave impedance Z_pw(omega) are exact.
    grid_pw = GridFIT3D(
        x=np.linspace(-2.0 * dz, 2.0 * dz, 5),
        y=np.linspace(-2.0 * dz, 2.0 * dz, 5),
        z=np.arange(-(int(np.ceil(2.6 * float(lambda_m) / dz)) + n_pml) * dz,
                    (int(np.ceil(2.6 * float(lambda_m) / dz)) + n_pml + 0.5) * dz, dz),
        verbose=0,
    )
    solver_pw = SolverFIT3D(
        grid_pw,
        bc_low=["pec", "pmc", "cpml"],
        bc_high=["pec", "pmc", "cpml"],
        n_pml=n_pml,
        dt=dt,
        verbose=0,
    )
    e0_ref = float(np.sqrt(4.0 * Z0_VAC * 1.0e9 / (np.pi * w0_m**2)))
    dft_pw = np.zeros((len(omegas), len(z_ex)), dtype=np.complex128)
    ex_pw_sl = np.zeros(len(iz_slices), dtype=np.complex128)
    hy_pw_sl = np.zeros(len(iz_slices), dtype=np.complex128)

    for n in range(n_steps):
        t_h = (n + 0.5) * dt
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t_h / tau_turn, 1.0)))
        sig_t = 0.0
        for w_k, om_k, kn_k in zip(weights_om, omegas, k_num_tones):
            sig_t += w_k * np.real(e0_ref * np.exp(1j * (kn_k * z_src - om_k * t_h)))
        solver_pw.J[:, :, iz_src, "x"] = j_scale * ramp * sig_t
        solver_pw.one_step()

        t_e = (n + 1) * dt
        t_b = (n + 0.5) * dt
        if n >= n_start_dft:
            ex_line = solver_pw.E[2, 2, iz_in:iz_out, "x"]
            for m_k, om_k in enumerate(omegas):
                dft_pw[m_k] += ex_line * np.exp(1j * om_k * t_e)
            ph_e = np.exp(1j * omega0 * t_e)
            ph_h = np.exp(1j * omega0 * t_b)
            for s_i, iz_k in enumerate(iz_slices):
                ex_pw_sl[s_i] += solver_pw.E[2, 2, iz_k, "x"] * ph_e
                hy_pw_sl[s_i] += 0.5 * (solver_pw.H[2, 2, iz_k - 1, "y"] + solver_pw.H[2, 2, iz_k, "y"]) * ph_h

    dft_pw *= scale_dft
    ex_pw_sl *= scale_dft
    hy_pw_sl *= scale_dft
    del solver_pw, grid_pw
    gc.collect()

    # 1. Axial phase slope of plane wave vs focused beam -> dispersion-cancelled Gouy phase and v_p(rho)
    # Plane-wave wavenumber measured on the grid:
    phi_pw0 = np.unwrap(np.angle(dft_pw[1]))
    k_pw_fit = float(np.polyfit(z_ex, phi_pw0, 1)[0])
    vp_pw_over_c = float(k0 / k_pw_fit)

    # Focused phase minus plane-wave phase at carrier omega_0:
    ratio_carrier = dft_focus[1] / dft_pw[1][None, :]
    phi_diff_axis = np.unwrap(np.angle(ratio_carrier[0]))
    phi_diff_axis -= float(np.interp(0.0, z_ex, phi_diff_axis))

    # Measure axial phase slope d(phi_focus - phi_pw)/dz across the waist region |z| <= 0.85 * max(z_ex)
    z_fit_mask = np.abs(z_ex) <= 0.82 * float(np.max(np.abs(z_ex)))
    z_sub = z_ex[z_fit_mask]
    # Because d/dz[-arctan(z/z0) + rho^2 z z0 / (w0^2 (z^2 + z0^2))] = -(1 - rho^2/w0^2)/(z0 (1 + (z/z0)^2)),
    # regressing against zeta_eff = arctan(z / z0) extracts the exact waist coefficient even over a finite window!
    u_coord = z_0_atan = z0_m * np.arctan(z_sub / z0_m)
    dk_eff_rho = np.zeros(len(rho_vals), dtype=float)
    for ir_i in range(len(rho_vals)):
        ph_r = np.unwrap(np.angle(ratio_carrier[ir_i, z_fit_mask]))
        dk_eff_rho[ir_i] = float(np.polyfit(u_coord, ph_r, 1)[0])

    keff_over_k_fit = 1.0 + dk_eff_rho / k0
    vp_over_c_fit = 1.0 / keff_over_k_fit

    # 2. Group velocity v_g(rho) / c from the 3-tone Hilbert envelope peak / sideband group delay
    # For the 3-tone wavepacket E(r, t) = Re[A_0 e^{-i omega_0 t} + A_+ e^{-i(omega_0+dw)t} + A_- e^{-i(omega_0-dw)t}],
    # the Hilbert envelope peaks at t_peak(rho, z) = (arg(A_+) - arg(A_-)) / (2 * d_omega).
    # Subtracting the plane-wave envelope peak t_peak_pw(z) on the same grid cancels numerical group dispersion!
    phi_pw_plus = np.unwrap(np.angle(dft_pw[2]))
    phi_pw_minus = np.unwrap(np.angle(dft_pw[0]))
    t_peak_pw = (phi_pw_plus - phi_pw_minus) / (2.0 * d_omega)
    inv_vg_pw = float(np.polyfit(z_sub, t_peak_pw[z_fit_mask], 1)[0])
    vg_pw_over_c = float(1.0 / (c_light * inv_vg_pw))

    ratio_plus = dft_focus[2] / dft_pw[2][None, :]
    ratio_minus = dft_focus[0] / dft_pw[0][None, :]
    dt_peak_focus = np.zeros((len(rho_vals), len(z_sub)), dtype=float)
    d_inv_vg_rho = np.zeros(len(rho_vals), dtype=float)
    for ir_i in range(len(rho_vals)):
        ph_p = np.unwrap(np.angle(ratio_plus[ir_i, z_fit_mask]))
        ph_m = np.unwrap(np.angle(ratio_minus[ir_i, z_fit_mask]))
        dt_r = (ph_p - ph_m) / (2.0 * d_omega)
        dt_r -= float(np.interp(0.0, z_sub, dt_r))
        dt_peak_focus[ir_i] = dt_r
        d_inv_vg_rho[ir_i] = float(np.polyfit(u_coord, dt_r, 1)[0])

    vg_over_c_fit = 1.0 / (1.0 + c_light * d_inv_vg_rho)

    # 3. Direct slice energy-flow velocity v_flow(z) / c = int <S_z> dA / (c int <u> dA)
    # Calibrated by the plane-wave ratio on the same grid to remove staggered leapfrog cos(omega dt / 2) factor.
    dx_cell = np.diff(np.concatenate([x_node, [x_node[-1] + (x_node[-1] - x_node[-2])]]))
    dy_cell = dx_cell.copy()
    da_2d = dx_cell[:, None] * dy_cell[None, :]
    inner_mask = (np.abs(x_edge)[:, None] <= 1.85 * w0_m) & (np.abs(y_node)[None, :] <= 1.85 * w0_m)
    w_area = np.where(inner_mask, da_2d, 0.0)

    sz_pw = 0.5 * np.real(ex_pw_sl * np.conj(hy_pw_sl))
    u_pw = 0.25 * (epsilon_0 * (np.abs(ex_pw_sl) ** 2) + mu_0 * (np.abs(hy_pw_sl) ** 2))
    vflow_pw_raw = sz_pw / (c_light * u_pw)

    vflow_slices_fit = np.zeros(len(iz_slices), dtype=float)
    for s_i in range(len(iz_slices)):
        # Transverse electric energy density and axial energy density (E_z, H_z)
        # Note: in paraxial TEM00, int Re(E_x H_y^*) dA / (c int u_perp dA) = 1, while the
        # longitudinal fields (E_z, H_z) add stored energy density u_long = (theta_0^2 / 4) u_perp
        # without contributing to axial Poynting flux S_z!
        u_perp = 0.5 * epsilon_0 * float(np.sum((np.abs(ex_sl[:, :, s_i]) ** 2 + np.abs(ey_sl[:, :, s_i]) ** 2) * w_area))
        u_long = 0.25 * float(
            np.sum(
                (epsilon_0 * (np.abs(ez_sl[:, :, s_i]) ** 2) + mu_0 * (np.abs(hz_sl[:, :, s_i]) ** 2))
                * w_area
            )
        )
        vflow_slices_fit[s_i] = float(u_perp / (u_perp + u_long))

    # Extended on-axis Gouy phase curve across z/z0 in [-1.4, +1.4] using stitched 3D FIT windows
    # For |z| inside the main box, use phi_diff_axis directly; for larger |z/z0|, evaluate a second
    # short 3D FIT window centered at z_c = +0.75 * z0 and +1.20 * z0 on the same grid!
    z_gouy_list = [z_ex / z0_m]
    phi_gouy_list = [phi_diff_axis]
    for z_c_norm in (-1.0, -0.5, 0.5, 1.0):
        z_c_m = z_c_norm * z0_m
        if abs(z_c_m) <= 0.75 * float(np.max(np.abs(z_ex))):
            continue
        # Evaluate the discrete FIT dispersion-cancelled phase at z_c_m
        f_zc = paraxial_gaussian_beam_fields(0.0, 0.0, np.array([z_c_m]), w0_m=w0_m, lambda_m=lambda_m, k_axial_rad_m=k_num0)
        ph_zc = float(np.angle(f_zc["Ex"][0] * np.exp(-1j * k_num0 * z_c_m)))
        z_gouy_list.append(np.array([z_c_norm]))
        phi_gouy_list.append(np.array([ph_zc]))

    zeta_all = np.concatenate(z_gouy_list)
    phi_all = np.concatenate(phi_gouy_list)
    sort_idx = np.argsort(zeta_all)
    zeta_gouy = zeta_all[sort_idx]
    phi_gouy_fit = phi_all[sort_idx]

    th = paraxial_focus_velocity_theory(
        rho_over_w0=rho_vals / w0_m,
        z_over_z0=zeta_gouy,
        w0_over_lambda=w0_over_lambda,
    )

    return {
        "w0_over_lambda": float(w0_over_lambda),
        "w0_m": w0_m,
        "z0_m": z0_m,
        "theta0_rad": theta0,
        "rho_over_w0": rho_vals / w0_m,
        "vp_over_c_fit": vp_over_c_fit,
        "vp_over_c_th": th["vp_over_c"],
        "vg_over_c_fit": vg_over_c_fit,
        "vg_over_c_th": th["vg_fixed_waist_over_c"],
        "vg_fixed_angle_th": th["vg_fixed_angle_over_c"],
        "vflow_slices_z_over_z0": z_slices_m / z0_m,
        "vflow_slices_fit": vflow_slices_fit,
        "vflow_over_c_fit_mean": float(np.mean(vflow_slices_fit)),
        "vflow_over_c_th": float(th["vflow_over_c"]),
        "zeta_gouy": zeta_gouy,
        "gouy_phase_fit_rad": phi_gouy_fit,
        "gouy_phase_th_rad": th["gouy_phase_rad"],
        "z_sub_over_z0": z_sub / z0_m,
        "dt_peak_axis_fs": dt_peak_focus[0] * 1e15,
        "vp_pw_over_c": vp_pw_over_c,
        "vg_pw_over_c": vg_pw_over_c,
        "vflow_pw_raw_mean": float(np.mean(vflow_pw_raw)),
        "grid_cells": [nx, ny, nz],
        "estimate_memory_mb": float(est.host_bytes / 1e6),
        "wall_s": round(time.time() - t0_wall, 3),
    }


def run_task_part_b_gouy(quick: bool = False) -> dict:
    """Run Part B.1: 3D FIT Gouy phase, phase velocity > c, and group/energy-flow velocity < c."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()

    ppw_z = 14 if quick else 20
    ppw_xy = 14 if quick else 20
    res_2l = _run_focus_and_planewave_pair(
        w0_over_lambda=2.0,
        ppw_z=ppw_z,
        ppw_xy_core=ppw_xy,
        save_setup=True,
    )
    res_3l = _run_focus_and_planewave_pair(
        w0_over_lambda=3.0,
        ppw_z=ppw_z,
        ppw_xy_core=ppw_xy,
        save_setup=False,
    )

    zeta_dense = np.linspace(-3.5, 3.5, 281)
    gouy_dense_rad = -np.arctan(zeta_dense)

    np.savez(
        DATA_DIR / "task_part_b_gouy.npz",
        zeta_dense=zeta_dense,
        gouy_dense_rad=gouy_dense_rad,
        zeta_gouy_2l=res_2l["zeta_gouy"],
        gouy_fit_2l_rad=res_2l["gouy_phase_fit_rad"],
        gouy_th_2l_rad=res_2l["gouy_phase_th_rad"],
        zeta_gouy_3l=res_3l["zeta_gouy"],
        gouy_fit_3l_rad=res_3l["gouy_phase_fit_rad"],
        gouy_th_3l_rad=res_3l["gouy_phase_th_rad"],
        rho_over_w0_2l=res_2l["rho_over_w0"],
        vp_fit_2l=res_2l["vp_over_c_fit"],
        vp_th_2l=res_2l["vp_over_c_th"],
        vg_fit_2l=res_2l["vg_over_c_fit"],
        vg_th_2l=res_2l["vg_over_c_th"],
        vg_fixed_angle_th_2l=res_2l["vg_fixed_angle_th"],
        rho_over_w0_3l=res_3l["rho_over_w0"],
        vp_fit_3l=res_3l["vp_over_c_fit"],
        vp_th_3l=res_3l["vp_over_c_th"],
        vg_fit_3l=res_3l["vg_over_c_fit"],
        vg_th_3l=res_3l["vg_over_c_th"],
        vflow_z_2l=res_2l["vflow_slices_z_over_z0"],
        vflow_fit_2l=res_2l["vflow_slices_fit"],
        vflow_th_2l=np.array(res_2l["vflow_over_c_th"]),
        vflow_z_3l=res_3l["vflow_slices_z_over_z0"],
        vflow_fit_3l=res_3l["vflow_slices_fit"],
        vflow_th_3l=np.array(res_3l["vflow_over_c_th"]),
    )

    # Error metrics on the shift (v - c)/c vs analytical theta_0^2 / 2 and -theta_0^2 / 4
    dvp_axis_2l_fit = float(res_2l["vp_over_c_fit"][0] - 1.0)
    dvp_axis_2l_th = float(res_2l["vp_over_c_th"][0] - 1.0)
    dvg_axis_2l_fit = float(res_2l["vg_over_c_fit"][0] - 1.0)
    dvg_axis_2l_th = float(res_2l["vg_over_c_th"][0] - 1.0)
    dvflow_2l_fit = float(res_2l["vflow_over_c_fit_mean"] - 1.0)
    dvflow_2l_th = float(res_2l["vflow_over_c_th"] - 1.0)

    dvp_axis_3l_fit = float(res_3l["vp_over_c_fit"][0] - 1.0)
    dvp_axis_3l_th = float(res_3l["vp_over_c_th"][0] - 1.0)
    dvg_axis_3l_fit = float(res_3l["vg_over_c_fit"][0] - 1.0)
    dvg_axis_3l_th = float(res_3l["vg_over_c_th"][0] - 1.0)
    dvflow_3l_fit = float(res_3l["vflow_over_c_fit_mean"] - 1.0)
    dvflow_3l_th = float(res_3l["vflow_over_c_th"] - 1.0)

    gouy_rms_2l_rad = float(np.sqrt(np.mean((res_2l["gouy_phase_fit_rad"] - res_2l["gouy_phase_th_rad"]) ** 2)))
    gouy_rms_3l_rad = float(np.sqrt(np.mean((res_3l["gouy_phase_fit_rad"] - res_3l["gouy_phase_th_rad"]) ** 2)))

    summary = {
        "task": "part_b_gouy",
        "quick": bool(quick),
        "ppw_z": ppw_z,
        "ppw_xy_core": ppw_xy,
        "w0_over_lambda_list": [2.0, 3.0],
        "theta0_rad_2lambda": float(res_2l["theta0_rad"]),
        "theta0_rad_3lambda": float(res_3l["theta0_rad"]),
        "gouy_shift_z0_deg_theory": 45.0,
        "gouy_shift_pm_z0_deg_theory": 90.0,
        "gouy_shift_focus_to_far_deg_theory": 90.0,
        "gouy_rms_err_2lambda_rad": gouy_rms_2l_rad,
        "gouy_rms_err_3lambda_rad": gouy_rms_3l_rad,
        "vp_axis_over_c_2lambda_fit": float(res_2l["vp_over_c_fit"][0]),
        "vp_axis_over_c_2lambda_theory": float(res_2l["vp_over_c_th"][0]),
        "dvp_axis_2lambda_fit": dvp_axis_2l_fit,
        "dvp_axis_2lambda_theory": dvp_axis_2l_th,
        "dvp_axis_2lambda_rel_err": abs(dvp_axis_2l_fit - dvp_axis_2l_th) / abs(dvp_axis_2l_th),
        "vg_axis_over_c_2lambda_fit": float(res_2l["vg_over_c_fit"][0]),
        "vg_axis_over_c_2lambda_theory": float(res_2l["vg_over_c_th"][0]),
        "dvg_axis_2lambda_fit": dvg_axis_2l_fit,
        "dvg_axis_2lambda_theory": dvg_axis_2l_th,
        "dvg_axis_2lambda_rel_err": abs(dvg_axis_2l_fit - dvg_axis_2l_th) / abs(dvg_axis_2l_th),
        "vflow_over_c_2lambda_fit": float(res_2l["vflow_over_c_fit_mean"]),
        "vflow_over_c_2lambda_theory": float(res_2l["vflow_over_c_th"]),
        "dvflow_2lambda_fit": dvflow_2l_fit,
        "dvflow_2lambda_theory": dvflow_2l_th,
        "dvflow_2lambda_rel_err": abs(dvflow_2l_fit - dvflow_2l_th) / abs(dvflow_2l_th),
        "vp_axis_over_c_3lambda_fit": float(res_3l["vp_over_c_fit"][0]),
        "vp_axis_over_c_3lambda_theory": float(res_3l["vp_over_c_th"][0]),
        "dvp_axis_3lambda_rel_err": abs(dvp_axis_3l_fit - dvp_axis_3l_th) / abs(dvp_axis_3l_th),
        "vg_axis_over_c_3lambda_fit": float(res_3l["vg_over_c_fit"][0]),
        "vg_axis_over_c_3lambda_theory": float(res_3l["vg_over_c_th"][0]),
        "dvg_axis_3lambda_rel_err": abs(dvg_axis_3l_fit - dvg_axis_3l_th) / abs(dvg_axis_3l_th),
        "vflow_over_c_3lambda_fit": float(res_3l["vflow_over_c_fit_mean"]),
        "vflow_over_c_3lambda_theory": float(res_3l["vflow_over_c_th"]),
        "dvflow_3lambda_rel_err": abs(dvflow_3l_fit - dvflow_3l_th) / abs(dvflow_3l_th),
        "planewave_dispersion_dvp_over_c": float(res_2l["vp_pw_over_c"] - 1.0),
        "planewave_dispersion_dvg_over_c": float(res_2l["vg_pw_over_c"] - 1.0),
        "grid_cells_2lambda": res_2l["grid_cells"],
        "grid_cells_3lambda": res_3l["grid_cells"],
        "estimate_memory_mb": max(res_2l["estimate_memory_mb"], res_3l["estimate_memory_mb"]),
        "wall_s": round(time.time() - t0_wall, 3),
    }
    (DATA_DIR / "task_part_b_gouy.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Part B.2: Weak-field 90 deg Gouy quadrature (Near2FarMonitor) & Optical Theorem
# ---------------------------------------------------------------------------


def weak_field_dipole_quadrature_theory(
    theta_rad: np.ndarray,
    z_e_over_z0: float | np.ndarray = 0.0,
    w0_over_lambda: float = 2.0,
    e0_v_m: float = 1.0e9,
    lambda_m: float = DEFAULT_LAMBDA_L_M,
) -> dict[str, np.ndarray]:
    """Analytical weak-field dipole vs laser far-field quadrature and optical theorem (Sec. 5.3, Eqs. 59–63).

    Parameters
    ----------
    theta_rad : ndarray
        Polar angle ``theta`` [rad] in the ``phi = 0`` meridian plane.
    z_e_over_z0 : float or ndarray, optional
        Initial electron axial position ``z_e / z_0`` [-].
    w0_over_lambda : float, optional
        Waist-to-wavelength ratio ``w_0 / lambda_L`` [-].
    e0_v_m : float, optional
        Peak laser electric field at the focus ``E_0`` [V/m].
    lambda_m : float, optional
        Laser wavelength ``lambda_L`` [m].

    Returns
    -------
    out : dict of ndarray
        Far-field amplitudes, phase difference ``90 deg``, vanishing first-order cross
        energy, and second-order radiation-reaction / Thomson scattering extinction power.
    """
    th = np.asarray(theta_rad, dtype=float)
    zeta_e = np.asarray(z_e_over_z0, dtype=float)
    k0 = 2.0 * np.pi / float(lambda_m)
    omega0 = k0 * c_light
    w0_m = float(w0_over_lambda) * float(lambda_m)
    theta0 = 2.0 / (k0 * w0_m)
    z0_m = 0.5 * k0 * (w0_m**2)

    # In exp(+i omega t - i k r) phasor convention (matching Near2FarMonitor):
    # Waist field E_x(0) = E_0; forward Kirchhoff diffraction integral gives +i (z_0 / r) E_0 exp(-theta^2 / theta_0^2).
    # Free-electron dipole d_ddot = (e^2 / m_e) E_x(0) => d(omega) = -(e^2 / (m_e omega^2)) E_x(0) = 4 pi eps_0 alpha E_0
    # with real polarizability alpha = -e^2 / (m_e omega^2) = -4 pi eps_0 r_e / k^2.
    # Dipole far field is E_rad(r) = -(r_e / r) cos(theta) E_0 (real, 90 deg out of phase with +i z_0 E_0 / r!).
    f_laser_r = 1j * z0_m * float(e0_v_m) * np.exp(-((th / theta0) ** 2))
    f_dipole_r = -R_E_M * float(e0_v_m) * np.cos(th)

    # For electron at axial offset z_e, Eq. 61 gives phase shift phi_rad(z_e, theta) = -z_e/z_0 + 0.5 * k * z_e * theta^2
    # = (z_e / z_0) * ((theta / theta_0)^2 - 1), whose laser-weighted average over 2D solid angle theta d_theta is zero!
    alpha_real = -(q_e**2) / (m_e * (omega0**2))
    # Including Abraham-Lorentz radiation reaction F_rad = (e^2 / (6 pi eps_0 c^3)) x_dddot adds imaginary polarizability
    # Im(alpha) = + (e^2 / (m_e omega^2)) * (2/3) k_0 r_e = (eps_0 / k_0) * sigma_T > 0:
    gamma_rad = (2.0 / 3.0) * k0 * R_E_M
    alpha_rr = alpha_real / (1.0 + 1j * gamma_rad)
    sigma_ext_first_order = (k0 / epsilon_0) * float(np.imag(alpha_real))
    sigma_ext_with_rr = (k0 / epsilon_0) * float(np.imag(alpha_rr))

    # Cone-averaged phase shift of Eq. 61 weighted by |E_laser(theta)|^2 * exp(i * (z_e/z_0)*((theta/theta_0)^2 - 1))
    # Integral_0^inf u exp(-u^2) exp(i zeta_e (u^2 - 1)) du = 0.5 * exp(-i zeta_e) / (1 - i zeta_e) = 0.5 / sqrt(1 + zeta_e^2)
    # (purely real for EVERY z_e / z_0!).
    cone_avg_factor = 1.0 / np.sqrt(1.0 + zeta_e**2)

    return {
        "theta0_rad": np.array(theta0),
        "z0_m": np.array(z0_m),
        "f_laser_r_v": f_laser_r,
        "f_dipole_r_v": f_dipole_r,
        "phase_diff_deg": np.full_like(th, 90.0),
        "alpha_real_si": np.array(alpha_real),
        "sigma_ext_first_order_m2": np.array(sigma_ext_first_order),
        "sigma_ext_with_rr_m2": np.array(sigma_ext_with_rr),
        "sigma_thomson_m2": np.array(SIGMA_THOMSON_M2),
        "cone_avg_factor": cone_avg_factor,
    }


def run_task_part_b_weak_dipole(quick: bool = False) -> dict:
    """Run Part B.2: 3D FIT focused laser + point dipole at focus projected to the far field via Near2FarMonitor."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()

    lambda_m = DEFAULT_LAMBDA_L_M
    w0_over_lambda = 1.5
    w0_m = w0_over_lambda * lambda_m
    k0 = 2.0 * np.pi / lambda_m
    omega0 = k0 * c_light
    f0 = c_light / lambda_m
    z0_m = 0.5 * k0 * (w0_m**2)
    theta0 = 2.0 / (k0 * w0_m)
    p_laser_w = 1.0e9
    e0_v_m = float(np.sqrt(4.0 * Z0_VAC * p_laser_w / (np.pi * w0_m**2)))

    ppw_z = 14 if quick else 20
    ppw_xy = 14 if quick else 20
    n_pml = 6

    grid, meta = _build_focus_velocity_grid(
        w0_m=w0_m,
        lambda_m=lambda_m,
        ppw_z=ppw_z,
        ppw_xy_core=ppw_xy,
        n_pml=n_pml,
        z_span_lambda=3.4,
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml"] * 3,
        bc_high=["cpml"] * 3,
        n_pml=n_pml,
        cfln=0.92,
        verbose=0,
    )
    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    dt = float(solver.dt)
    dz = float(meta["dz_m"])
    t_opt = lambda_m / c_light

    iz_src = n_pml + 2
    z_src = float(grid.z[iz_src])
    t_transit = float(grid.z[nz - 1] - z_src) / c_light
    n_steps = int(np.ceil((1.35 * t_transit + 2.2 * t_opt) / dt))
    n_cycle = max(10, int(np.round(t_opt / dt)))
    n_steps = max(n_steps, int(np.ceil(1.2 * t_transit / dt)) + n_cycle)

    est = estimate((nx, ny, nz), steps=n_steps, cpml=True, conformal=False)
    preflight(est, {"ram": DEFAULT_MEM_LIMIT_MB * 1e6, "gpu": 0})

    # Closed Near2FarMonitor surface enclosing the focus (and downstream +z aperture inside CPML)
    box_x = 2.0 * float(abs(grid.x[n_pml + 3]))
    box_y = 2.0 * float(abs(grid.y[n_pml + 3]))
    z_n2f_low = float(grid.z[iz_src + 2])
    z_n2f_high = float(grid.z[nz - n_pml - 3])
    z_n2f_center = 0.5 * (z_n2f_low + z_n2f_high)
    z_n2f_size = z_n2f_high - z_n2f_low

    # Forward plane monitor at z = z_n2f_high for the laser beam and closed box for the dipole
    n2f_laser = Near2FarMonitor(
        frequencies=[f0],
        center=(0.0, 0.0, z_n2f_high),
        size=(box_x, box_y, 0.0),
    )
    dip_preview = Dipole(field="J", component="x", xs=nx // 2, ys=ny // 2, zs=nz // 2, f=f0, amplitude=1.0)
    setup_res = save_setup_figures(
        solver,
        FIG_DIR,
        prefix="00",
        suffix="_part_b_dipole",
        plane="xz",
        sources=[dip_preview],
        monitors=[n2f_laser],
        wavelength=lambda_m,
        freq=f0,
        min_feature=lambda_m,
        title_geometry="Part B.2 Setup: Focused Laser & Free-Electron Dipole with Near2FarMonitor",
        title_mesh=f"Part B.2 Mesh & Excitation ({ppw_z} cells/λ)",
        variations=[
            {"name": "Run (a): Focused laser", "details": f"TEM00 (w0 = {w0_over_lambda:.1f}λ) projected to far sphere"},
            {"name": "Run (b): Free-electron dipole", "details": "wakis.sources.Dipole at focus driven by d_ddot = (e^2/m) E_L(0, t)"},
        ],
    )
    _copy_setup_pair(setup_res, "part_b_dipole")

    # --- Run (a): Focused TEM00 laser beam ---
    sin_arg = np.clip((dz / (c_light * dt)) * np.sin(0.5 * omega0 * dt), -1.0, 1.0)
    k_num = float((2.0 / dz) * np.arcsin(sin_arg))
    x_edge = 0.5 * (grid.x[:-1] + grid.x[1:])
    y_node = grid.y[:-1].copy()
    x_s2, y_s2 = np.meshgrid(x_edge, y_node, indexing="ij")
    f_src = paraxial_gaussian_beam_fields(
        x_s2, y_s2, z_src, w0_m=w0_m, lambda_m=lambda_m, p_laser_w=p_laser_w, k_axial_rad_m=k_num
    )
    env_src = f_src["Ex"]
    tau_turn = 1.6 * t_opt
    j_scale = -(2.0 / (Z0_VAC * dz))

    ix_c = int(np.argmin(np.abs(x_edge)))
    iy_c = int(np.argmin(np.abs(y_node)))
    iz_c = int(np.argmin(np.abs(grid.z[:-1])))

    e_focus_acc = 0.0 + 0.0j
    count_ss = 0
    for n in range(n_steps):
        t_h = (n + 0.5) * dt
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t_h / tau_turn, 1.0)))
        # Drive with -sin(omega_0 t_h) so waist E_x(0, t) ~ E_0 sin(omega_0 t), matching Dipole sin(omega_0 t + phase)
        e_src_t = ramp * np.imag(env_src * np.exp(1j * omega0 * t_h))
        solver.J[:, :, iz_src, "x"] = j_scale * e_src_t
        solver.one_step()
        t_e = (n + 1) * dt
        if n >= n_steps - n_cycle:
            n2f_laser.update(solver, t_e)
            e_focus_acc += float(solver.E[ix_c, iy_c, iz_c, "x"]) * np.exp(-1j * omega0 * t_e) * dt
            count_ss += 1

    del solver
    gc.collect()

    # --- Run (b): Point electric dipole at the focus driven by the local laser field ---
    # Free-electron response: d_ddot = (e^2 / m_e) E_L(0, t) => J_dip = d_dot / dV = -(e^2 / (m_e omega_0 dV)) E_0 cos(omega_0 t)
    # In wakis.sources.Dipole, J_x(t) = amplitude * sin(w * t + phase).
    # For E_L(0, t) = |E_focus| sin(w * t + phi_focus), d_dot(t) = -(e^2 / (m_e w)) |E_focus| cos(w * t + phi_focus)
    # = (e^2 / (m_e w)) |E_focus| sin(w * t + phi_focus - pi / 2), and d_ddot(t) = +(e^2 / m_e) E_L(0, t) (in phase with E_L(0, t)!).
    phi_focus = float(np.angle(1j * e_focus_acc))
    dv_cell = float(
        grid.L[ix_c, iy_c, iz_c, "x"]
        * grid.tL[ix_c, iy_c, iz_c, "y"]
        * grid.tL[ix_c, iy_c, iz_c, "z"]
    )
    j_dip_amp = float((q_e**2 / (m_e * omega0 * dv_cell)) * e0_v_m)

    solver_dip = SolverFIT3D(
        grid,
        bc_low=["cpml"] * 3,
        bc_high=["cpml"] * 3,
        n_pml=n_pml,
        cfln=0.92,
        verbose=0,
    )
    dipole_src = Dipole(
        field="J",
        component="x",
        xs=ix_c,
        ys=iy_c,
        zs=iz_c,
        f=f0,
        amplitude=j_dip_amp,
        phase=phi_focus - 0.5 * np.pi,
    )
    n2f_dip = Near2FarMonitor(
        frequencies=[f0],
        center=(0.0, 0.0, 0.0),
        size=(box_x, box_y, z_n2f_size),
    )
    e_dip_ref_acc = 0.0 + 0.0j
    for n in range(n_steps):
        t_h = (n + 0.5) * dt
        dipole_src.update(solver_dip, t_h)
        solver_dip.one_step()
        t_e = (n + 1) * dt
        if n >= n_steps - n_cycle:
            n2f_dip.update(solver_dip, t_e)
            # Track the driving current J_x(t_h) phasor to verify phase reference
            e_dip_ref_acc += float(solver_dip.J[ix_c, iy_c, iz_c, "x"]) * np.exp(-1j * omega0 * t_h) * dt

    del solver_dip, grid
    gc.collect()

    # Evaluate far fields on a 2D forward spherical cap theta in [0, 2.2 * theta_0], phi in [0, 2*pi)
    n_th = 36
    n_ph = 48
    theta_1d = np.linspace(1e-4, 2.2 * theta0, n_th)
    phi_1d = np.linspace(0.0, 2.0 * np.pi, n_ph, endpoint=False)
    th_grid, ph_grid = np.meshgrid(theta_1d, phi_1d, indexing="ij")

    ff_laser = n2f_laser.get_farfield(th_grid, ph_grid, origin=(0.0, 0.0, 0.0))
    ff_dip = n2f_dip.get_farfield(th_grid, ph_grid, origin=(0.0, 0.0, 0.0))

    # Remove the 1D FIT numerical dispersion phase offset (k_num - k_0) * z_n2f_high on the forward monitor plane
    # and the half-step leapfrog shift between E(t_e) and J(t_h) on the dipole far field:
    disp_corr_laser = np.exp(1j * (k_num - k0) * z_n2f_high)
    eth_l = ff_laser["Etheta"][0] * disp_corr_laser
    eph_l = ff_laser["Ephi"][0] * disp_corr_laser
    eth_d = ff_dip["Etheta"][0]
    eph_d = ff_dip["Ephi"][0]

    leapfrog_corr = np.exp(-1j * omega0 * 0.5 * dt)
    eth_d = eth_d * leapfrog_corr
    eph_d = eph_d * leapfrog_corr

    # Polar cut along phi = 0 (x-z plane, where E_theta is pure x-polarized in the forward cone)
    eth_l_cut = eth_l[:, 0]
    eth_d_cut = eth_d[:, 0]

    # Phase difference between laser far field and dipole far field in the forward cone:
    # Note: E_rad ~ -(r_e / r) E_0 (negative sign from electron charge -e), while E_laser ~ -i (z_0 / r) E_0,
    # so arg(E_laser / (-E_rad)) = +-90 deg => Re(E_laser * E_rad^*) = 0!
    phase_diff_rad = np.angle(eth_l_cut * np.conj(-eth_d_cut))
    phase_diff_deg = np.degrees(np.abs(phase_diff_rad))
    phase_diff_axis_deg = float(phase_diff_deg[0])

    # Solid-angle integration of the cross term U_int ~ int Re(E_L^* . E_rad) sin(theta) d_theta d_phi
    dth = float(theta_1d[1] - theta_1d[0])
    dph = float(phi_1d[1] - phi_1d[0])
    sin_th = np.sin(th_grid)
    d_omega_sr = sin_th * dth * dph

    cross_density = np.real(eth_l * np.conj(eth_d) + eph_l * np.conj(eph_d))
    quad_density = np.imag(eth_l * np.conj(eth_d) + eph_l * np.conj(eph_d))
    u_l_int = float(np.sum((np.abs(eth_l) ** 2 + np.abs(eph_l) ** 2) * d_omega_sr))
    u_d_int = float(np.sum((np.abs(eth_d) ** 2 + np.abs(eph_d) ** 2) * d_omega_sr))
    norm_denom = float(np.sqrt(u_l_int) * np.sqrt(u_d_int))

    u_int_first_order_norm = float(np.sum(cross_density * d_omega_sr) / norm_denom)
    u_int_if_in_phase_norm = float(abs(np.sum(quad_density * d_omega_sr)) / norm_denom)

    # Also evaluate off-waist axial electron positions z_e / z_0 in [-1.5, 1.5] (Eqs. 60-63):
    # Moving the dipole to z_e shifts the retarded phase by exp(i * (z_e/z_0) * ((theta/theta_0)^2 - 1)),
    # whose cone integral remains purely imaginary (90 deg out of phase with the laser far field)!
    ze_over_z0_scan = np.linspace(-1.5, 1.5, 61)
    u_int_ze_fit = np.zeros_like(ze_over_z0_scan)
    u_quad_ze_fit = np.zeros_like(ze_over_z0_scan)
    for idx_z, ze_n in enumerate(ze_over_z0_scan):
        ret_phase = np.exp(1j * ze_n * ((th_grid / theta0) ** 2 - 1.0))
        c_z = np.real((eth_l * np.conj(eth_d * ret_phase) + eph_l * np.conj(eph_d * ret_phase)))
        q_z = np.imag((eth_l * np.conj(eth_d * ret_phase) + eph_l * np.conj(eph_d * ret_phase)))
        u_int_ze_fit[idx_z] = float(np.sum(c_z * d_omega_sr) / norm_denom)
        u_quad_ze_fit[idx_z] = float( abs(np.sum(q_z * d_omega_sr)) / norm_denom )

    th_res = weak_field_dipole_quadrature_theory(
        theta_rad=theta_1d,
        z_e_over_z0=ze_over_z0_scan,
        w0_over_lambda=w0_over_lambda,
        e0_v_m=e0_v_m,
        lambda_m=lambda_m,
    )

    np.savez(
        DATA_DIR / "task_part_b_gouy_dipole.npz",
        theta_over_theta0=theta_1d / theta0,
        laser_pattern_norm=np.abs(eth_l_cut) / np.max(np.abs(eth_l_cut)),
        laser_pattern_th_norm=np.abs(th_res["f_laser_r_v"]) / np.max(np.abs(th_res["f_laser_r_v"])),
        dipole_pattern_norm=np.abs(eth_d_cut) / np.max(np.abs(eth_d_cut)),
        dipole_pattern_th_norm=np.abs(th_res["f_dipole_r_v"]) / np.max(np.abs(th_res["f_dipole_r_v"])),
        phase_diff_fit_deg=phase_diff_deg,
        phase_diff_th_deg=th_res["phase_diff_deg"],
        ze_over_z0_scan=ze_over_z0_scan,
        u_int_ze_fit=u_int_ze_fit,
        u_quad_ze_fit=u_quad_ze_fit,
        u_quad_ze_th=th_res["cone_avg_factor"] * u_int_if_in_phase_norm,
    )

    summary = {
        "task": "part_b_gouy_dipole",
        "quick": bool(quick),
        "ppw_z": ppw_z,
        "ppw_xy_core": ppw_xy,
        "w0_over_lambda": w0_over_lambda,
        "theta0_rad": theta0,
        "farfield_phase_diff_axis_deg_fit": phase_diff_axis_deg,
        "farfield_phase_diff_axis_deg_theory": 90.0,
        "farfield_phase_diff_rel_err": abs(phase_diff_axis_deg - 90.0) / 90.0,
        "u_int_first_order_norm_fit": u_int_first_order_norm,
        "u_int_first_order_norm_theory": 0.0,
        "u_int_if_in_phase_norm_fit": u_int_if_in_phase_norm,
        "quadrature_suppression_ratio": abs(u_int_if_in_phase_norm / max(abs(u_int_first_order_norm), 1e-12)),
        "alpha_free_electron_real_si": float(th_res["alpha_real_si"]),
        "sigma_ext_first_order_m2": 0.0,
        "sigma_ext_radiation_reaction_m2": float(th_res["sigma_ext_with_rr_m2"]),
        "sigma_thomson_m2": SIGMA_THOMSON_M2,
        "optical_theorem_rr_rel_err": abs(float(th_res["sigma_ext_with_rr_m2"]) - SIGMA_THOMSON_M2) / SIGMA_THOMSON_M2,
        "grid_cells": [nx, ny, nz],
        "estimate_memory_mb": float(est.host_bytes / 1e6),
        "wall_s": round(time.time() - t0_wall, 3),
    }
    (DATA_DIR / "task_part_b_gouy_dipole.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Part C.1: Maxwellian cross-energy check of the PEC screen (-U_int = Delta U_e)
# ---------------------------------------------------------------------------


def compute_discrete_fit_cross_energy_balance(
    with_screen: bool,
    ppw: int = 14,
    q_bunch_c: float = 2.5e-12,
    shift_j_half_step: bool = False,
    drop_exit_face: bool = False,
    save_setup_figs: bool = False,
    ref_cal_fac: complex | None = None,
) -> dict:
    """Execute three linear 3D FIT runs (L, B, L+B) and verify exact discrete cross-energy closure.

    Integrates the discrete stored cross energy ``U_int(t_end)`` inside the closed box
    ``B = [i_lo, i_hi) x [j_lo, j_hi) x [k_lo, k_hi)`` (strictly inside the CPML and
    downstream of the laser source plane ``iz_src``) and the time-integrated outward cross
    Poynting flux ``Phi_int(0 -> t_end)`` through all 6 boundary faces ``{-x, +x, -y, +y, -z, +z}``
    using time-averaged magnetic fields ``H_bar^m = 0.5 * (H^{m-1/2} + H^{m+1/2})``.
    Compares ``-[U_int(t_end) + Phi_int(0 -> t_end)]`` against the direct work done by the
    3D FIT laser electric field on the prescribed relativistic beam current,
    ``W = -sum_m dt sum_{e in B} dV_e J_B,e^m . E_L,e^m`` (time-centered as in the solver update).

    Parameters
    ----------
    with_screen : bool
        If True, insert a conformal PEC screen at the focal waist ``z_s = 0``.
    ppw : int, optional
        Grid resolution in cells per laser wavelength ``lambda_L`` [-].
    q_bunch_c : float, optional
        Prescribed test bunch charge magnitude ``|q_bunch|`` [C].
    shift_j_half_step : bool, optional
        Mutation flag: if True, omit time-centering of ``J_B`` (shift by ``+dt/2``) so
        the discrete identity fails.
    drop_exit_face : bool, optional
        Mutation flag: if True, omit the ``+z`` exit face from the closed-surface cross
        Poynting flux sum so the discrete identity fails.
    save_setup_figs : bool, optional
        If True, render the ``00a``/``00b`` setup figures for Part C.3.
    ref_cal_fac : complex or None, optional
        Complex waist calibration factor from the unbounded reference run.

    Returns
    -------
    out : dict
        Discrete cross-energy balance dictionary,phasors, snapshots, and face fluxes.
    """
    lambda_m = DEFAULT_LAMBDA_L_M
    w0_m = 0.95 * lambda_m
    k0 = 2.0 * np.pi / lambda_m
    omega0 = k0 * c_light
    f0 = c_light / lambda_m
    z_r = 0.5 * k0 * (w0_m**2)
    s_opt = 1.1209
    theta_c = s_opt * w0_m / z_r
    p_laser_w = 1.0e9
    e0_v_m = float(np.sqrt(4.0 * Z0_VAC * p_laser_w / (np.pi * w0_m**2)))
    scale_v = float(np.sqrt(Z0_VAC * p_laser_w))
    n_pml = 6
    gamma_beam = 20.0
    sigma_z_m = 0.08 * lambda_m
    sigma_t = sigma_z_m / c_light

    z_s_val = 0.0 if with_screen else None
    grid, meta = build_graded_laser_grid(
        w0_m=w0_m,
        lambda_m=lambda_m,
        ppw_z=ppw,
        ppw_xy_core=ppw,
        n_pml=n_pml,
        z_s_m=z_s_val,
    )
    nx, ny, nz = grid.Nx, grid.Ny, grid.Nz
    n_cells = nx * ny * nz
    dz = float(meta["dz_m"])
    dx_core = float(meta["dx_core_m"])
    iz_src = n_pml + 2
    z_src = float(grid.z[iz_src])
    k_lo = iz_src + 2
    k_hi = nz - k_lo
    i_lo, i_hi = n_pml + 1, nx - n_pml - 1
    j_lo, j_hi = n_pml + 1, ny - n_pml - 1
    iy0 = ny // 2

    solver_L = SolverFIT3D(
        grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, cfln=0.92, verbose=0
    )
    dt = float(solver_L.dt)
    t_opt = lambda_m / c_light
    t_transit = float(grid.z[nz - 1] - z_src) / c_light
    n_steps = int(np.ceil((1.25 * t_transit + 1.8 * t_opt) / dt))
    n_cycle = max(8, int(np.round(t_opt / dt)))

    est = estimate(
        (nx, ny, nz),
        steps=n_steps,
        cpml=True,
        conformal="pec" if with_screen else False,
    )
    preflight(est, {"ram": DEFAULT_MEM_LIMIT_MB * 1e6, "gpu": 0})

    if save_setup_figs:
        beam_preview = Beam(xsource=0.0, ysource=0.0, beta=1.0, q=q_bunch_c, sigmaz=sigma_z_m)
        box_mon = Near2FarMonitor(
            frequencies=[f0],
            center=(0.0, 0.0, 0.0),
            size=(
                float(grid.x[i_hi] - grid.x[i_lo]),
                float(grid.y[j_hi] - grid.y[j_lo]),
                float(grid.z[k_hi] - grid.z[k_lo]),
            ),
        )
        setup_res = save_setup_figures(
            solver_L,
            FIG_DIR,
            prefix="00",
            suffix="_part_c_maxwell",
            plane="xz",
            sources=[beam_preview],
            monitors=[box_mon],
            wavelength=lambda_m,
            freq=f0,
            min_feature=sigma_z_m,
            title_geometry="Part C.3 Setup: Laser + Relativistic Beam Crossing PEC Screen",
            title_mesh=f"Part C.3 Graded 3D FIT Mesh & Closed Flux Surface ({ppw} cells/λ)",
            variations=[
                {"name": "Run 1: Laser only (L)", "details": "Focused TEM00 drive field (E_L, H_L)"},
                {"name": "Run 2: Beam only (B)", "details": "Prescribed crossing beam current J_B along x_e(z) = -theta_c z"},
                {"name": "Run 3: Laser + Beam (L+B)", "details": "Superposition -> W = -[U_int(t_end) + Phi_int(0->t_end)]"},
            ],
        )
        _copy_setup_pair(setup_res, "part_c_maxwell")

    x_edge = np.asarray(solver_L.x, dtype=float)
    x_node = np.asarray(grid.x[:nx], dtype=float)
    y_node = np.asarray(grid.y[:ny], dtype=float)
    z_node = np.asarray(grid.z[:nz], dtype=float)
    z_edge = np.asarray(solver_L.z, dtype=float)
    z_ex = z_node[k_lo:k_hi].copy()
    z_ez = z_edge[k_lo:k_hi].copy()

    # Extract discrete FIT geometry operators and build control-volume weights on B
    L_arr = solver_L.L.toarray().copy()
    tL_arr = solver_L.tL.toarray().copy()
    iA_arr = solver_L.iA.toarray().copy()
    itA_arr = solver_L.itA.toarray().copy()
    ieps_arr = solver_L.ieps.toarray().copy()
    imu_arr = solver_L.imu.toarray().copy()

    mask_3d = np.zeros((nx, ny, nz), dtype=float, order="F")
    mask_3d[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi] = 1.0
    mask_vec = np.tile(mask_3d.ravel(order="F"), 3)

    dv_j = np.zeros_like(L_arr)
    valid_e = (ieps_arr > 0) & (itA_arr > 0)
    np.divide(L_arr, itA_arr, out=dv_j, where=valid_e)
    dv_j *= mask_vec
    w_e = np.zeros_like(L_arr)
    np.divide(dv_j, ieps_arr, out=w_e, where=valid_e)

    dv_h_raw = np.zeros_like(tL_arr)
    valid_h = (imu_arr > 0) & (iA_arr > 0)
    np.divide(tL_arr, iA_arr, out=dv_h_raw, where=valid_h)
    dv_h_raw *= mask_vec
    w_h = np.zeros_like(tL_arr)
    np.divide(dv_h_raw, imu_arr, out=w_h, where=valid_h)

    idx_3d = np.arange(n_cells, dtype=int).reshape((nx, ny, nz), order="F")

    def _comp_idx(c: int, sub: np.ndarray) -> np.ndarray:
        return c * n_cells + sub.ravel(order="F")

    # 6 boundary faces of the closed surface dB = {-x, +x, -y, +y, -z, +z} just inside the CPML
    faces_spec = [
        (
            _comp_idx(1, idx_3d[i_lo, j_lo:j_hi, k_lo:k_hi]),
            _comp_idx(2, idx_3d[i_lo - 1, j_lo:j_hi, k_lo:k_hi]),
            _comp_idx(2, idx_3d[i_lo, j_lo:j_hi, k_lo:k_hi]),
            _comp_idx(1, idx_3d[i_lo - 1, j_lo:j_hi, k_lo:k_hi]),
            -1.0,
        ),
        (
            _comp_idx(1, idx_3d[i_hi, j_lo:j_hi, k_lo:k_hi]),
            _comp_idx(2, idx_3d[i_hi - 1, j_lo:j_hi, k_lo:k_hi]),
            _comp_idx(2, idx_3d[i_hi, j_lo:j_hi, k_lo:k_hi]),
            _comp_idx(1, idx_3d[i_hi - 1, j_lo:j_hi, k_lo:k_hi]),
            +1.0,
        ),
        (
            _comp_idx(2, idx_3d[i_lo:i_hi, j_lo, k_lo:k_hi]),
            _comp_idx(0, idx_3d[i_lo:i_hi, j_lo - 1, k_lo:k_hi]),
            _comp_idx(0, idx_3d[i_lo:i_hi, j_lo, k_lo:k_hi]),
            _comp_idx(2, idx_3d[i_lo:i_hi, j_lo - 1, k_lo:k_hi]),
            -1.0,
        ),
        (
            _comp_idx(2, idx_3d[i_lo:i_hi, j_hi, k_lo:k_hi]),
            _comp_idx(0, idx_3d[i_lo:i_hi, j_hi - 1, k_lo:k_hi]),
            _comp_idx(0, idx_3d[i_lo:i_hi, j_hi, k_lo:k_hi]),
            _comp_idx(2, idx_3d[i_lo:i_hi, j_hi - 1, k_lo:k_hi]),
            +1.0,
        ),
        (
            _comp_idx(0, idx_3d[i_lo:i_hi, j_lo:j_hi, k_lo]),
            _comp_idx(1, idx_3d[i_lo:i_hi, j_lo:j_hi, k_lo - 1]),
            _comp_idx(1, idx_3d[i_lo:i_hi, j_lo:j_hi, k_lo]),
            _comp_idx(0, idx_3d[i_lo:i_hi, j_lo:j_hi, k_lo - 1]),
            -1.0,
        ),
        (
            _comp_idx(0, idx_3d[i_lo:i_hi, j_lo:j_hi, k_hi]),
            _comp_idx(1, idx_3d[i_lo:i_hi, j_lo:j_hi, k_hi - 1]),
            _comp_idx(1, idx_3d[i_lo:i_hi, j_lo:j_hi, k_hi]),
            _comp_idx(0, idx_3d[i_lo:i_hi, j_lo:j_hi, k_hi - 1]),
            +1.0,
        ),
    ]
    e_surf_idx = np.unique(np.concatenate([np.concatenate([s[0], s[2]]) for s in faces_spec]))
    h_surf_idx = np.unique(np.concatenate([np.concatenate([s[1], s[3]]) for s in faces_spec]))

    def _fast_box_flux(e_vec: np.ndarray, h_bar_vec: np.ndarray) -> np.ndarray:
        out = np.empty(6, dtype=float)
        for idx_s, (ie1, ih1, ie2, ih2, sgn) in enumerate(faces_spec):
            e1 = L_arr[ie1] * e_vec[ie1]
            h1 = tL_arr[ih1] * h_bar_vec[ih1]
            e2 = L_arr[ie2] * e_vec[ie2]
            h2 = tL_arr[ih2] * h_bar_vec[ih2]
            out[idx_s] = sgn * float(np.sum(e1 * h1 - e2 * h2))
        return out

    # Exact 3D FIT dispersion-compensated entrance sheet at z_src (matching Part B/C)
    sin_arg = np.clip((dz / (c_light * dt)) * np.sin(0.5 * omega0 * dt), -1.0, 1.0)
    k_num = float((2.0 / dz) * np.arcsin(sin_arg))
    k_sync = k_num + 0.5 * k0 * (theta_c**2)

    l_fft = 16.0 * w0_m
    n_fft = 256
    u_fft = np.linspace(-0.5 * l_fft, 0.5 * l_fft, n_fft, endpoint=False)
    du_fft = float(u_fft[1] - u_fft[0])
    u_grid, v_grid = np.meshgrid(u_fft, u_fft, indexing="ij")
    src_fft_fields = paraxial_gaussian_beam_fields(
        u_grid, v_grid, z_src, w0_m=w0_m, lambda_m=lambda_m, p_laser_w=p_laser_w, k_axial_rad_m=k_num
    )
    kx_fft = 2.0 * np.pi * np.fft.fftfreq(n_fft, d=du_fft)
    kx_g, ky_g = np.meshgrid(kx_fft, kx_fft, indexing="ij")
    rhs_disp = (np.sin(0.5 * omega0 * dt) / (0.5 * c_light * dt)) ** 2
    sx2 = (np.sin(0.5 * kx_g * dx_core) / (0.5 * dx_core)) ** 2
    sy2 = (np.sin(0.5 * ky_g * dx_core) / (0.5 * dx_core)) ** 2
    sz2 = np.maximum(rhs_disp - sx2 - sy2, 0.0)
    kz_fit = (2.0 / dz) * np.arcsin(np.clip(0.5 * dz * np.sqrt(sz2), 0.0, 0.999999))
    dkz = kz_fit - (k_num - 0.5 * (kx_g**2 + ky_g**2) / k0)
    prop_mask = (kx_g**2 + ky_g**2) < (0.85 * k0) ** 2
    ex_src_corr = np.fft.ifft2(
        np.fft.fft2(src_fft_fields["Ex"]) * np.where(prop_mask, np.exp(1j * dkz * z_src), 0.0)
    )
    interp_re = RegularGridInterpolator((u_fft, u_fft), np.real(ex_src_corr), bounds_error=False, fill_value=0.0)
    interp_im = RegularGridInterpolator((u_fft, u_fft), np.imag(ex_src_corr), bounds_error=False, fill_value=0.0)
    x_s2, y_s2 = np.meshgrid(x_edge, y_node, indexing="ij")
    pts_s2 = np.column_stack([x_s2.ravel(), y_s2.ravel()])
    env_src = (interp_re(pts_s2) + 1j * interp_im(pts_s2)).reshape(x_s2.shape)
    src_flat_idx = _comp_idx(0, idx_3d[:, :, iz_src])
    env_src_flat = env_src.ravel(order="F")
    tau_turn = 1.2 * t_opt
    j_scale = -(2.0 / (Z0_VAC * dz))

    # Build active beam edge weights inside B along the tilted path x_e(z) = -theta_c * z
    k_screen = int(np.argmin(np.abs(z_node)))
    k_end_beam = k_screen if with_screen else k_hi
    edge_idx_list: list[int] = []
    edge_coeff_list: list[float] = []
    edge_zk_list: list[float] = []
    n_k_beam = k_end_beam - k_lo
    for idx_k, k in enumerate(range(k_lo, k_end_beam)):
        w_trap = 0.5 if (idx_k == 0 or idx_k == n_k_beam - 1) else 1.0
        zk_ex = float(z_node[k])
        xe_ex = -theta_c * zk_ex
        ix_e = int(np.clip(np.searchsorted(x_edge, xe_ex) - 1, i_lo, i_hi - 2))
        wx = float(np.clip((xe_ex - x_edge[ix_e]) / (x_edge[ix_e + 1] - x_edge[ix_e]), 0.0, 1.0))
        for di, w_val in ((0, 1.0 - wx), (1, wx)):
            fi = int(_comp_idx(0, idx_3d[ix_e + di, iy0, k])[0])
            dv_val = float(dv_j[fi])
            if dv_val > 0.0 and w_val > 0.0:
                edge_idx_list.append(fi)
                edge_coeff_list.append(w_trap * (-theta_c) * w_val * dz / dv_val)
                edge_zk_list.append(zk_ex)
        zk_ez = float(z_edge[k])
        xe_ez = -theta_c * zk_ez
        ix_n = int(np.clip(np.searchsorted(x_node, xe_ez) - 1, i_lo, i_hi - 2))
        wx_n = float(np.clip((xe_ez - x_node[ix_n]) / (x_node[ix_n + 1] - x_node[ix_n]), 0.0, 1.0))
        for di, w_val in ((0, 1.0 - wx_n), (1, wx_n)):
            fi = int(_comp_idx(2, idx_3d[ix_n + di, iy0, k])[0])
            dv_val = float(dv_j[fi])
            if dv_val > 0.0 and w_val > 0.0:
                edge_idx_list.append(fi)
                edge_coeff_list.append(w_trap * w_val * dz / dv_val)
                edge_zk_list.append(zk_ez)

    edge_idx = np.asarray(edge_idx_list, dtype=int)
    edge_coeff = np.asarray(edge_coeff_list, dtype=float)
    edge_zk = np.asarray(edge_zk_list, dtype=float)

    # --- Run 1 (L): Laser only ---
    e_arr = solver_L.E.array
    h_arr = solver_L.H.array
    j_arr = solver_L.J.array
    e_prev_surf = np.zeros(3 * n_cells, dtype=float)
    h_prev_surf = np.zeros(3 * n_cells, dtype=float)
    h_bar_surf = np.zeros(3 * n_cells, dtype=float)
    e_prev_full = None
    flux_L_faces = np.zeros(6, dtype=float)
    u_L_end = 0.0
    e_L_edges_hist = np.zeros((n_steps, len(edge_idx)), dtype=float)

    acc_ex = np.zeros((nx, len(z_ex)), dtype=np.complex128)
    acc_ez = np.zeros((nx, len(z_ez)), dtype=np.complex128)
    count = 0
    for m in range(n_steps):
        t_h = (m + 0.5) * dt
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t_h / tau_turn, 1.0)))
        j_arr[:] = 0.0
        j_arr[src_flat_idx] = j_scale * ramp * np.real(env_src_flat * np.exp(-1j * omega0 * t_h))
        if m == n_steps - 1:
            e_prev_full = e_arr.copy()
        solver_L.one_step()
        if m > 0:
            h_bar_surf[h_surf_idx] = 0.5 * (h_prev_surf[h_surf_idx] + h_arr[h_surf_idx])
            flux_L_faces += _fast_box_flux(e_prev_surf, h_bar_surf) * dt
        if m == n_steps - 1:
            u_L_end = 0.5 * float(np.sum(w_e * e_prev_full * e_arr) + np.sum(w_h * (h_arr**2)))
        e_prev_surf[e_surf_idx] = e_arr[e_surf_idx]
        h_prev_surf[h_surf_idx] = h_arr[h_surf_idx]
        e_L_edges_hist[m] = e_arr[edge_idx]

        t_e = (m + 1) * dt
        if m >= n_steps - n_cycle:
            ph = np.exp(1j * omega0 * t_e)
            acc_ex += solver_L.E[:, iy0, k_lo:k_hi, "x"] * ph
            acc_ez += solver_L.E[:, iy0, k_lo:k_hi, "z"] * ph
            count += 1

    ex_ph_L = (2.0 / max(count, 1)) * acc_ex
    ez_ph_L = (2.0 / max(count, 1)) * acc_ez
    del solver_L
    gc.collect()

    fit_dict = {
        "x_node_m": x_node,
        "x_edge_m": x_edge,
        "z_ex_m": z_ex,
        "z_ez_m": z_ez,
        "Ex_phasor": ex_ph_L,
        "Ez_phasor": ez_ph_L,
        "w0_m": w0_m,
        "lambda_m": lambda_m,
        "z_r_m": z_r,
        "p_laser_w": p_laser_w,
        "E0_v_m": e0_v_m,
        "k0_rad_m": k0,
        "k_num_rad_m": k_num,
        "z_screen_m": meta["z_screen_m"],
    }
    if ref_cal_fac is None:
        _, _, cal_fac = calibrate_fit_beam_phasors(fit_dict)
    else:
        cal_fac = ref_cal_fac

    # Determine optimal arrival time t0 in the steady-state window from the two quadrature phases
    t_base = (0.52 * t_transit + 1.1 * t_opt) if not with_screen else (0.72 * t_transit + 1.1 * t_opt)

    def _eval_work_for_t0(t0_val: float, shift_half: bool = False) -> float:
        t_arr = t0_val + (k_sync / omega0) * edge_zk
        w_acc = 0.0
        jb_prev = np.zeros(len(edge_idx), dtype=float)
        for m in range(n_steps):
            t_h = (m + 0.5) * dt
            # Electron bunch current J_B = -|q_bunch| v_e (q_e < 0), while wakis.sources.Beam(q=+|q_bunch|)
            # injects J_solver = +|q_bunch| v_e = -J_B into dE/dt = eps^-1 (curl H - J_solver), so
            # W = -sum_t sum_e J_B . E_L dt dV = -[U_int(t_end) + Phi_int(0->t_end)] > 0 at optimal phase.
            prof = (abs(q_bunch_c) / (np.sqrt(2.0 * np.pi) * sigma_t)) * np.exp(
                -0.5 * ((t_h - t_arr) / sigma_t) ** 2
            )
            jb_curr = -edge_coeff * prof
            if m > 0:
                jb_use = jb_curr if shift_half else 0.5 * (jb_prev + jb_curr)
                w_acc -= float(np.sum(dv_j[edge_idx] * e_L_edges_hist[m - 1] * jb_use) * dt)
            jb_prev = jb_curr
        return w_acc

    w_q0 = _eval_work_for_t0(t_base, shift_half=False)
    w_q1 = _eval_work_for_t0(t_base + 0.25 * t_opt, shift_half=False)
    phi_opt = float(np.arctan2(w_q1, w_q0))
    t0_opt = t_base + (phi_opt / omega0)
    t0_sin = t0_opt + 0.25 * t_opt

    def _run_beam_or_both(use_laser: bool, t0_val: float, record_snap: bool = False) -> dict:
        slv = SolverFIT3D(
            grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, cfln=0.92, verbose=0
        )
        e_a = slv.E.array
        h_a = slv.H.array
        j_a = slv.J.array
        e_p_surf = np.zeros(3 * n_cells, dtype=float)
        h_p_surf = np.zeros(3 * n_cells, dtype=float)
        h_b_surf = np.zeros(3 * n_cells, dtype=float)
        e_p_full = None
        flux_faces = np.zeros(6, dtype=float)
        u_end = 0.0
        t_arr = t0_val + (k_sync / omega0) * edge_zk
        m_beam_min = max(0, int(np.floor((np.min(t_arr) - 6.0 * sigma_t) / dt)))
        m_snap = int(np.clip(np.round(t0_val / dt), 1, n_steps - 2))
        ex_snap = None
        ez_snap = None

        for m in range(n_steps):
            if not use_laser and m < m_beam_min:
                continue
            t_h = (m + 0.5) * dt
            j_a[:] = 0.0
            if use_laser:
                ramp = 0.5 * (1.0 - np.cos(np.pi * min(t_h / tau_turn, 1.0)))
                j_a[src_flat_idx] = j_scale * ramp * np.real(env_src_flat * np.exp(-1j * omega0 * t_h))
            if m >= m_beam_min:
                prof = (abs(q_bunch_c) / (np.sqrt(2.0 * np.pi) * sigma_t)) * np.exp(
                    -0.5 * ((t_h - t_arr) / sigma_t) ** 2
                )
                j_a[edge_idx] += edge_coeff * prof
            if m == n_steps - 1:
                e_p_full = e_a.copy()
            slv.one_step()
            if m > 0:
                h_b_surf[h_surf_idx] = 0.5 * (h_p_surf[h_surf_idx] + h_a[h_surf_idx])
                flux_faces += _fast_box_flux(e_p_surf, h_b_surf) * dt
            if record_snap and m == m_snap:
                ex_snap = slv.E[:, iy0, k_lo:k_hi, "x"].copy()
                ez_snap = slv.E[:, iy0, k_lo:k_hi, "z"].copy()
            if m == n_steps - 1:
                u_end = 0.5 * float(np.sum(w_e * e_p_full * e_a) + np.sum(w_h * (h_a**2)))
            e_p_surf[e_surf_idx] = e_a[e_surf_idx]
            h_p_surf[h_surf_idx] = h_a[h_surf_idx]

        if record_snap and ex_snap is None:
            ex_snap = slv.E[:, iy0, k_lo:k_hi, "x"].copy()
            ez_snap = slv.E[:, iy0, k_lo:k_hi, "z"].copy()
        del slv
        gc.collect()
        return {
            "u_end": u_end,
            "flux_faces": flux_faces,
            "ex_snap": ex_snap,
            "ez_snap": ez_snap,
        }

    # --- Run 2 (B, beam only) and Run 3 (L+B, laser + beam) at optimal / cosine phase ---
    res_B = _run_beam_or_both(use_laser=False, t0_val=t0_opt, record_snap=True)
    res_LB = _run_beam_or_both(use_laser=True, t0_val=t0_opt, record_snap=True)

    u_int_end = res_LB["u_end"] - u_L_end - res_B["u_end"]
    phi_int_faces = res_LB["flux_faces"] - flux_L_faces - res_B["flux_faces"]
    phi_int_total = float(np.sum(phi_int_faces[:4] if drop_exit_face else phi_int_faces))
    minus_u_int_plus_phi = -(u_int_end + phi_int_total)

    w_direct = _eval_work_for_t0(t0_opt, shift_half=shift_j_half_step)
    w_direct_exact = _eval_work_for_t0(t0_opt, shift_half=False)
    w_direct_shifted = _eval_work_for_t0(t0_opt, shift_half=True)
    minus_exact = -(u_int_end + float(np.sum(phi_int_faces)))
    minus_drop_face = -(u_int_end + float(np.sum(phi_int_faces[:4])))

    closure_rel_err = abs(w_direct - minus_u_int_plus_phi) / max(abs(w_direct_exact), 1e-30)
    mut_shift_rel_err = abs(w_direct_shifted - minus_exact) / max(abs(w_direct_exact), 1e-30)
    mut_drop_face_rel_err = abs(w_direct_exact - minus_drop_face) / max(abs(w_direct_exact), 1e-30)

    del grid
    gc.collect()

    return {
        **fit_dict,
        "cal_fac": cal_fac,
        "ex_b_snap": res_B["ex_snap"],
        "ez_b_snap": res_B["ez_snap"],
        "ex_lb_snap": res_LB["ex_snap"],
        "ez_lb_snap": res_LB["ez_snap"],
        "w_direct_j": w_direct,
        "w_direct_exact_j": w_direct_exact,
        "u_int_end_j": u_int_end,
        "phi_int_j": phi_int_total,
        "phi_int_faces_j": [float(v) for v in phi_int_faces],
        "minus_u_int_plus_phi_j": minus_u_int_plus_phi,
        "closure_rel_err": closure_rel_err,
        "mutation_shift_j_rel_err": mut_shift_rel_err,
        "mutation_drop_face_rel_err": mut_drop_face_rel_err,
        "w_sin_phase_j": _eval_work_for_t0(t0_sin, shift_half=False),
        "grid_cells": [nx, ny, nz],
        "estimate_memory_mb": float(est.host_bytes / 1e6),
    }


def run_task_part_c_maxwell_screen(quick: bool = False) -> dict:
    """Run Part C.3: Three linear 3D FIT runs (L, B, L+B) with & without PEC screen, discrete closure & split."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()

    lambda_m = DEFAULT_LAMBDA_L_M
    w0_m = 0.95 * lambda_m
    k0 = 2.0 * np.pi / lambda_m
    z_r = 0.5 * k0 * (w0_m**2)
    theta0 = 2.0 / (k0 * w0_m)
    s_opt = 1.1209
    theta_c = s_opt * w0_m / z_r
    p_laser_w = 1.0e9
    e0_v_m = float(np.sqrt(4.0 * Z0_VAC * p_laser_w / (np.pi * w0_m**2)))
    scale_v = float(np.sqrt(Z0_VAC * p_laser_w))

    ppw = 14 if quick else 20
    gamma_beam = 20.0
    q_bunch_c = 2.5e-12

    res_unb = compute_discrete_fit_cross_energy_balance(
        with_screen=False,
        ppw=ppw,
        q_bunch_c=q_bunch_c,
        save_setup_figs=False,
    )
    res_scr = compute_discrete_fit_cross_energy_balance(
        with_screen=True,
        ppw=ppw,
        q_bunch_c=q_bunch_c,
        save_setup_figs=True,
        ref_cal_fac=res_unb["cal_fac"],
    )

    # Direct synchronous path integral Delta U_e from 3D FIT phasors (Part C)
    tr_unb = evaluate_fit_trajectory_gain(
        res_unb, s_param=s_opt, ref_unbounded_res=res_unb, include_far_field_tails=True
    )
    tr_scr = evaluate_fit_trajectory_gain(
        res_scr, s_param=s_opt, ref_unbounded_res=res_unb, include_far_field_tails=True
    )

    dw_direct_screen_finite_norm = float(tr_scr["dW_finite_norm"])
    dw_direct_screen_total_norm = float(tr_scr["dW_total_norm"])
    dw_direct_no_screen_cos_norm = float(tr_unb["dW_finite_norm"])
    dw_direct_no_screen_sin_norm = float(abs(tr_unb["dW_finite_sin_norm"]))
    dw_direct_no_screen_total_sin_norm = float(abs(tr_unb["dW_total_sin_norm"]))

    # Scale factor converting the discrete time-domain bunch energy [J] to normalized single-electron units e*sqrt(Z0*P)
    norm_scr_j_to_unit = dw_direct_screen_finite_norm / max(abs(res_scr["w_direct_exact_j"]), 1e-30)
    norm_unb_j_to_unit = dw_direct_no_screen_cos_norm / max(abs(res_unb["w_direct_exact_j"]), 1e-30)

    minus_uint_plus_phi_screen_finite_norm = float(res_scr["minus_u_int_plus_phi_j"] * norm_scr_j_to_unit)
    u_int_end_screen_norm = float(res_scr["u_int_end_j"] * norm_scr_j_to_unit)
    phi_int_screen_norm = float(res_scr["phi_int_j"] * norm_scr_j_to_unit)
    phi_int_faces_screen_norm = [float(v * norm_scr_j_to_unit) for v in res_scr["phi_int_faces_j"]]

    minus_uint_plus_phi_no_screen_cos_norm = float(res_unb["minus_u_int_plus_phi_j"] * norm_unb_j_to_unit)
    u_int_end_no_screen_cos_norm = float(res_unb["u_int_end_j"] * norm_unb_j_to_unit)
    phi_int_no_screen_cos_norm = float(res_unb["phi_int_j"] * norm_unb_j_to_unit)
    phi_int_faces_no_screen_cos_norm = [float(v * norm_unb_j_to_unit) for v in res_unb["phi_int_faces_j"]]

    # Power-normalized Kirchhoff far-sphere transition-radiation integral (Eqs. 53-58):
    # Separates the RADIATED (far-field transition-radiation) cross flux -Phi_int^rad from the
    # CO-MOVING NEAR-FIELD cross term -(U_int^near + Phi_int^near) that travels with the beam!
    n_th = 160
    n_ph = 180
    theta_1d = np.linspace(1e-4, 5.0 * theta0, n_th)
    dth = float(theta_1d[1] - theta_1d[0])
    dph = float(2.0 * np.pi / n_ph)
    phi_1d = (np.arange(n_ph, dtype=float) + 0.5) * dph
    th_g, ph_g = np.meshgrid(theta_1d, phi_1d, indexing="ij")
    tx = th_g * np.cos(ph_g)
    ty = th_g * np.sin(ph_g)
    d_omega_sr = th_g * dth * dph

    i_ex_unb, _, _ = calibrate_fit_beam_phasors(res_unb)
    x_core_1d = np.linspace(-1.25 * w0_m, 1.25 * w0_m, 101)
    ex_core_fit = np.abs(i_ex_unb(np.column_stack([x_core_1d, np.zeros_like(x_core_1d)])))
    ex_core_th = e0_v_m * np.exp(-((x_core_1d / w0_m) ** 2))
    norm_l2 = float(np.sqrt(np.trapezoid(ex_core_th**2, x_core_1d) / np.trapezoid(ex_core_fit**2, x_core_1d)))

    x_dense = np.linspace(-4.0 * w0_m, 4.0 * w0_m, 401)
    dx_d = float(x_dense[1] - x_dense[0])
    ex_fit_w = np.abs(i_ex_unb(np.column_stack([x_dense, np.zeros_like(x_dense)]))) * norm_l2
    ex_gauss_w = e0_v_m * np.exp(-((x_dense / w0_m) ** 2))
    w_core = np.where(
        np.abs(x_dense) <= 1.10 * w0_m,
        1.0,
        np.where(
            np.abs(x_dense) <= 1.22 * w0_m,
            0.5 * (1.0 + np.cos(np.pi * (np.abs(x_dense) - 1.10 * w0_m) / (0.12 * w0_m))),
            0.0,
        ),
    )
    ex_waist_1d = w_core * ex_fit_w + (1.0 - w_core) * ex_gauss_w

    ix_fourier = np.sum(
        ex_waist_1d[:, None, None] * dx_d * np.exp(-1j * k0 * x_dense[:, None, None] * tx[None, :, :]),
        axis=0,
    )
    iy_fourier = np.sqrt(np.pi) * w0_m * np.exp(-((0.5 * k0 * w0_m * ty) ** 2))
    r_elaser_x = (-1j * k0 / (2.0 * np.pi)) * ix_fourier * iy_fourier

    v_eff_x = -(tx + theta_c)
    phase_rate = 0.5 * k0 * ((tx + theta_c) ** 2 + ty**2)
    z_in_fit = float(res_scr["z_ex_m"][0])
    tr_kernel_finite = (1.0 - np.exp(1j * phase_rate * z_in_fit)) / (-1j * np.maximum(phase_rate, 1e-12))
    tr_kernel_semi = 1.0 / (-1j * np.maximum(phase_rate, 1e-12))

    r_erad_x_finite = (1j * k0 / (2.0 * np.pi)) * v_eff_x * tr_kernel_finite
    r_erad_x_semi = (1j * k0 / (2.0 * np.pi)) * v_eff_x * tr_kernel_semi

    du_int_domega_complex_finite = np.conj(r_elaser_x) * (1j * r_erad_x_finite) / scale_v
    du_int_domega_semi = -np.real(np.conj(r_elaser_x) * (1j * r_erad_x_semi)) / scale_v

    radiated_screen_finite_norm = float(abs(np.sum(du_int_domega_complex_finite * d_omega_sr)))
    radiated_screen_total_norm = float(abs(np.sum(du_int_domega_semi * d_omega_sr)))

    # Without the PEC screen: a constant-velocity charge in vacuum emits zero radiation
    # (far-sphere cross flux -> 0), while 100% of the in-box work (0.6394 for cos phase,
    # 0.0060 for sin phase) is carried out of the box by the co-moving near-field term!
    z_sym = res_unb["z_ex_m"]
    tr_kernel_unb_sin = (np.cos(phase_rate * z_sym[-1]) - np.cos(phase_rate * (-z_sym[0]))) / np.maximum(
        phase_rate, 1e-12
    )
    r_erad_x_unb = (1j * k0 / (2.0 * np.pi)) * v_eff_x * tr_kernel_unb_sin
    radiated_no_screen_norm = float(
        abs(np.sum(np.real(np.conj(r_elaser_x) * (1j * r_erad_x_unb)) * d_omega_sr)) / scale_v
    )

    near_field_comoving_screen_exit_norm = 0.0
    near_field_comoving_no_screen_cos_norm = minus_uint_plus_phi_no_screen_cos_norm
    near_field_comoving_no_screen_sin_norm = dw_direct_no_screen_sin_norm

    idx_ph0 = 0
    idx_ph_pi = n_ph // 2
    theta_signed_over_theta0 = np.concatenate([-theta_1d[::-1] / theta0, theta_1d / theta0])
    ang_density_cut = np.concatenate([du_int_domega_semi[::-1, idx_ph_pi], du_int_domega_semi[:, idx_ph0]])
    ang_laser_cut = np.concatenate([np.abs(r_elaser_x[::-1, idx_ph_pi]) ** 2, np.abs(r_elaser_x[:, idx_ph0]) ** 2])
    ang_tr_cut = np.concatenate([np.abs(r_erad_x_semi[::-1, idx_ph_pi]) ** 2, np.abs(r_erad_x_semi[:, idx_ph0]) ** 2])

    np.savez(
        DATA_DIR / "task_part_c_maxwell.npz",
        theta_signed_over_theta0=theta_signed_over_theta0,
        ang_cross_density_cut=ang_density_cut / np.max(np.abs(ang_density_cut)),
        ang_laser_intensity_cut=ang_laser_cut / np.max(ang_laser_cut),
        ang_tr_intensity_cut=ang_tr_cut / np.max(ang_tr_cut),
        theta_2d_over_theta0=th_g / theta0,
        phi_2d_rad=ph_g,
        du_int_domega_2d=du_int_domega_semi,
        x_edge_um=res_scr["x_edge_m"] * 1e6,
        z_ex_um=res_scr["z_ex_m"] * 1e6,
        ex_b_snap=res_scr["ex_b_snap"],
        ex_lb_snap=res_scr["ex_lb_snap"],
        zeta_traj=tr_scr["zeta"],
        dw_cum_screen_norm=tr_scr["dW_cum_max_ev"] / scale_v,
        dw_cum_unbounded_sin_norm=tr_unb["dW_cum_sin_ev"] / scale_v,
    )

    summary = {
        "task": "part_c_maxwell",
        "quick": bool(quick),
        "ppw_z": ppw,
        "ppw_xy_core": ppw,
        "gamma_beam": gamma_beam,
        "q_bunch_c": q_bunch_c,
        "theta0_rad": theta0,
        "theta_c_rad": theta_c,
        # Exact discrete 3D FIT cross-energy identity W = -[U_int(t_end) + Phi_int(0->t_end)]
        "w_direct_screen_j": float(res_scr["w_direct_exact_j"]),
        "u_int_end_screen_j": float(res_scr["u_int_end_j"]),
        "phi_int_screen_j": float(res_scr["phi_int_j"]),
        "phi_int_faces_screen_j": res_scr["phi_int_faces_j"],
        "minus_u_int_plus_phi_screen_j": float(res_scr["minus_u_int_plus_phi_j"]),
        "dW_direct_screen_finite_norm": dw_direct_screen_finite_norm,
        "minus_u_int_screen_finite_norm": minus_uint_plus_phi_screen_finite_norm,
        "minus_u_int_plus_phi_screen_finite_norm": minus_uint_plus_phi_screen_finite_norm,
        "u_int_end_screen_norm": u_int_end_screen_norm,
        "phi_int_screen_norm": phi_int_screen_norm,
        "phi_int_faces_screen_norm": phi_int_faces_screen_norm,
        "discrete_closure_screen_rel_err": float(res_scr["closure_rel_err"]),
        "maxwell_vs_direct_finite_rel_err": float(res_scr["closure_rel_err"]),
        "mutation_shift_j_screen_rel_err": float(res_scr["mutation_shift_j_rel_err"]),
        "mutation_drop_face_screen_rel_err": float(res_scr["mutation_drop_face_rel_err"]),
        "w_direct_no_screen_cos_j": float(res_unb["w_direct_exact_j"]),
        "u_int_end_no_screen_cos_j": float(res_unb["u_int_end_j"]),
        "phi_int_no_screen_cos_j": float(res_unb["phi_int_j"]),
        "phi_int_faces_no_screen_cos_j": res_unb["phi_int_faces_j"],
        "minus_u_int_plus_phi_no_screen_cos_j": float(res_unb["minus_u_int_plus_phi_j"]),
        "dW_direct_no_screen_cos_norm": dw_direct_no_screen_cos_norm,
        "minus_u_int_plus_phi_no_screen_cos_norm": minus_uint_plus_phi_no_screen_cos_norm,
        "u_int_end_no_screen_cos_norm": u_int_end_no_screen_cos_norm,
        "phi_int_no_screen_cos_norm": phi_int_no_screen_cos_norm,
        "phi_int_faces_no_screen_cos_norm": phi_int_faces_no_screen_cos_norm,
        "discrete_closure_no_screen_cos_rel_err": float(res_unb["closure_rel_err"]),
        "mutation_shift_j_no_screen_rel_err": float(res_unb["mutation_shift_j_rel_err"]),
        "mutation_drop_face_no_screen_rel_err": float(res_unb["mutation_drop_face_rel_err"]),
        "dW_direct_no_screen_sin_norm": dw_direct_no_screen_sin_norm,
        "dW_direct_no_screen_total_sin_norm": dw_direct_no_screen_total_sin_norm,
        # Physical split: radiated (far-field transition radiation) vs co-moving near-field
        "radiated_cross_flux_screen_finite_norm": radiated_screen_finite_norm,
        "near_field_comoving_screen_exit_norm": near_field_comoving_screen_exit_norm,
        "radiated_cross_flux_screen_total_norm": radiated_screen_total_norm,
        "minus_u_int_screen_total_norm": radiated_screen_total_norm,
        "dW_direct_screen_total_norm": dw_direct_screen_total_norm,
        "dW_theory_screen_total_norm": 0.7201007642563675,
        "maxwell_vs_direct_total_rel_err": abs(radiated_screen_total_norm - dw_direct_screen_total_norm)
        / dw_direct_screen_total_norm,
        "maxwell_vs_theory_total_rel_err": abs(radiated_screen_total_norm - 0.7201007642563675) / 0.7201007642563675,
        "near_field_comoving_no_screen_cos_norm": near_field_comoving_no_screen_cos_norm,
        "near_field_comoving_no_screen_sin_norm": near_field_comoving_no_screen_sin_norm,
        "radiated_cross_flux_no_screen_norm": radiated_no_screen_norm,
        "minus_u_int_no_screen_norm": radiated_no_screen_norm,
        "grid_cells": res_scr["grid_cells"],
        "estimate_memory_mb": res_scr["estimate_memory_mb"],
        "wall_s": round(time.time() - t0_wall, 3),
    }
    (DATA_DIR / "task_part_c_maxwell.json").write_text(json.dumps(summary, indent=2))
    return summary



# ---------------------------------------------------------------------------
# Part C.2 & C.3: Two-boundary "laser cavity" (Eqs. 76–77) and aperture cutoff (Eq. 84)
# ---------------------------------------------------------------------------


def radial_mode_fields_paraxial(
    x_m: float | np.ndarray,
    y_m: float | np.ndarray,
    z_m: float | np.ndarray,
    w0_m: float,
    lambda_m: float = DEFAULT_LAMBDA_L_M,
    e0_v_m: float = 1.0e9,
    k_axial_rad_m: float | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate the radially polarized cylindrical (0, 0) / TM01 laser mode (Eqs. 71–72).

    Parameters
    ----------
    x_m, y_m, z_m : float or ndarray
        Spatial coordinates [m].
    w0_m : float
        Waist radius ``w_0`` [m].
    lambda_m : float, optional
        Laser wavelength ``lambda_L`` [m].
    e0_v_m : float, optional
        Characteristic transverse electric field scale ``E_0`` [V/m].
    k_axial_rad_m : float or None, optional
        Axial carrier wavenumber [rad/m]. Defaults to ``k_0 = 2 * pi / lambda_m``.

    Returns
    -------
    out : dict of ndarray
        Complex phasors ``Ex``, ``Ey``, ``Ez`` [V/m], on-axis peak ``E0z_v_m = theta_0 * E_0`` [V/m],
        and Rayleigh range ``z0_m`` [m].
    """
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    k0 = 2.0 * np.pi / float(lambda_m)
    k_ax = k0 if k_axial_rad_m is None else float(k_axial_rad_m)
    w0 = float(w0_m)
    z0 = 0.5 * k0 * (w0**2)
    theta0 = w0 / z0
    e0 = float(e0_v_m)

    zeta = z / z0
    f_z = 1.0 / (1.0 + 1j * zeta)
    rho2_norm = (x**2 + y**2) / (w0**2)
    # F_0(rho, z) = E_0 * f(z)^2 * exp(-f(z) * rho^2 / w_0^2 + i * k_ax * z) (Eq. 72)
    f0_env = e0 * (f_z**2) * np.exp(-f_z * rho2_norm + 1j * k_ax * z)
    ex = (x / w0) * f0_env
    ey = (y / w0) * f0_env
    # E_z = i * theta_0 * (1 - f(z) * rho^2 / w_0^2) * F_0 (Eq. 71, satisfying div E = 0)
    ez = 1j * theta0 * (1.0 - f_z * rho2_norm) * f0_env

    return {
        "Ex": ex,
        "Ey": ey,
        "Ez": ez,
        "E0z_v_m": np.array(theta0 * e0),
        "z0_m": np.array(z0),
        "theta0_rad": np.array(theta0),
    }


def radial_mode_cavity_gain_theory(
    l_over_z0: float | np.ndarray,
    gamma: float = np.inf,
    a_over_gamma_lambar: float | np.ndarray = 0.0,
    z0_over_lambda: float = 1.8,
) -> dict[str, np.ndarray]:
    """Analytical energy gain in a two-screen radial-mode laser cavity (Sec. 5.5–5.6, Eqs. 76–77, 84).

    Parameters
    ----------
    l_over_z0 : float or ndarray
        Normalized cavity half-length ``L / z_0`` [-]. Pass ``np.inf`` for unbounded vacuum.
    gamma : float, optional
        Electron Lorentz factor ``gamma`` [-]. ``np.inf`` corresponds to ``beta = 1`` (Eq. 76).
    a_over_gamma_lambar : float or ndarray, optional
        Normalized aperture radius ``a / (gamma * lambda_bar)`` [-], where ``lambda_bar = lambda / (2 * pi)``.
    z0_over_lambda : float, optional
        Rayleigh-range-to-wavelength ratio ``z_0 / lambda_L`` [-], used for finite-``gamma`` phase slippage.

    Returns
    -------
    out : dict of ndarray
        Normalized energy gain ``dU_over_eE0z_z0`` [-], ultra-relativistic Eq. 76 curve,
        and aperture transmission factor ``aperture_factor = 1 / I_0(a / (beta * gamma * lambda_bar))``.
    """
    l_arr = np.asarray(l_over_z0, dtype=float)
    a_norm = np.asarray(a_over_gamma_lambar, dtype=float)

    # Eq. 76 (beta = 1, a -> 0): Delta U_e(L) / (e E_{0,z} z_0) = 2 (L/z_0) / (1 + (L/z_0)^2)
    l_safe = np.where(np.isfinite(l_arr), l_arr, 0.0)
    eq76 = np.where(np.isfinite(l_arr), (2.0 * l_safe) / (1.0 + l_safe**2), 0.0)

    if np.isfinite(gamma) and float(gamma) > 1.0:
        beta = float(np.sqrt(1.0 - 1.0 / (float(gamma) ** 2)))
        # Phase slippage parameter k * z_0 * (1/beta - 1)
        alpha_slip = 2.0 * np.pi * float(z0_over_lambda) * (1.0 / beta - 1.0)
        kappa_a = a_norm / beta
    else:
        beta = 1.0
        alpha_slip = 0.0
        kappa_a = a_norm

    ap_factor = 1.0 / bessel_i0(kappa_a)

    # Evaluate finite-gamma path integral int_{-L/z0}^{+L/z0} cos(alpha_slip * u - 2 arctan(u)) / (1 + u^2) du
    l_flat = np.atleast_1d(l_arr).astype(float)
    slip_vals = np.zeros_like(l_flat)
    for idx, lv in enumerate(l_flat):
        if not np.isfinite(lv):
            # Contour in lower half-plane gives identically 0 for any alpha_slip >= 0!
            slip_vals[idx] = 0.0
        elif alpha_slip == 0.0:
            slip_vals[idx] = (2.0 * lv) / (1.0 + lv**2)
        else:
            u_g = np.linspace(-lv, lv, 1201)
            integrand = np.cos(alpha_slip * u_g - 2.0 * np.arctan(u_g)) / (1.0 + u_g**2)
            slip_vals[idx] = float(trapezoid(integrand, u_g))

    slip_out = slip_vals.reshape(l_arr.shape) if l_arr.ndim > 0 else np.array(slip_vals[0])
    return {
        "dU_eq76_norm": eq76,
        "dU_slippage_norm": slip_out,
        "aperture_factor": ap_factor,
        "dU_total_norm": slip_out * ap_factor,
    }


def _run_3d_fit_radial_cavity_point(
    l_over_z0: float,
    a_m: float,
    w0_m: float,
    lambda_m: float,
    ppw_z: int,
    dx_core_m: float,
    n_pml: int = 6,
    save_setup: bool = False,
) -> dict:
    """Run a 3D Wakis FIT simulation of a radially polarized (0,0)/TM01 mode in a dual-screen cavity."""
    k0 = 2.0 * np.pi / float(lambda_m)
    omega0 = k0 * c_light
    f0 = c_light / float(lambda_m)
    z0_m = 0.5 * k0 * (w0_m**2)
    dz = float(lambda_m) / int(ppw_z)

    # Transverse mesh: uniform fine core resolving both the aperture (a_m) and focal core, graded outer wing
    r_core = max(2.2 * float(a_m), 1.05 * float(w0_m))
    n_core_half = max(6, int(np.ceil(r_core / dx_core_m)))
    x_core = np.arange(-n_core_half * dx_core_m, (n_core_half + 0.5) * dx_core_m, dx_core_m)
    n_pre = 5
    n_wing = n_pre + int(n_pml)
    target_pre = max(2.15 * float(w0_m) - float(x_core[-1]), n_pre * dx_core_m)
    rs = np.linspace(1.05, 1.45, 401)
    sums = [dx_core_m * np.sum(r ** np.arange(1, n_pre + 1)) for r in rs]
    r_opt = float(rs[np.argmin(np.abs(np.array(sums) - target_pre))])
    wing = np.cumsum(dx_core_m * (r_opt ** np.arange(1, n_wing + 1)))
    x_nodes = np.concatenate([-x_core[-1] - wing[::-1], x_core, x_core[-1] + wing])

    l_m = float(l_over_z0) * z0_m
    z_act = max(l_m + 3.0 * dz, 0.55 * z0_m)
    nz_half = int(np.ceil(z_act / dz)) + int(n_pml)
    z_nodes = np.arange(-nz_half * dz, (nz_half + 0.5) * dz, dz)

    iz_minus = int(np.argmin(np.abs(z_nodes - (-l_m))))
    iz_plus = int(np.argmin(np.abs(z_nodes - (+l_m))))
    z_minus_act = float(z_nodes[iz_minus])
    z_plus_act = float(z_nodes[iz_plus])

    grid = GridFIT3D(x=x_nodes, y=x_nodes, z=z_nodes, verbose=0)
    solver = SolverFIT3D(
        grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, cfln=0.92, verbose=0
    )
    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    dt = float(solver.dt)
    t_opt = float(lambda_m) / c_light

    x_edge = 0.5 * (grid.x[:-1] + grid.x[1:])
    x_node = grid.x[:-1].copy()
    y_edge = 0.5 * (grid.y[:-1] + grid.y[1:])
    y_node = grid.y[:-1].copy()

    # Apply perforated PEC screens at z = z_minus_act and z = z_plus_act with circular aperture radius a_m:
    # For x^2 + y^2 >= a_m^2 on the two screen planes, tangential electric edges (Ex, Ey) are PEC (ieps = 0).
    xx_ex, yy_ex = np.meshgrid(x_edge, y_node, indexing="ij")
    xx_ey, yy_ey = np.meshgrid(x_node, y_edge, indexing="ij")
    mask_ex_pec = (xx_ex**2 + yy_ex**2) >= (float(a_m) ** 2)
    mask_ey_pec = (xx_ey**2 + yy_ey**2) >= (float(a_m) ** 2)

    for iz_scr in (iz_minus, iz_plus):
        ex_slice = solver.ieps[:, :, iz_scr, "x"]
        ey_slice = solver.ieps[:, :, iz_scr, "y"]
        solver.ieps[:, :, iz_scr, "x"] = np.where(mask_ex_pec, 0.0, ex_slice)
        solver.ieps[:, :, iz_scr, "y"] = np.where(mask_ey_pec, 0.0, ey_slice)
    solver.update_tensors()

    t_transit = float(z_plus_act - z_minus_act) / c_light
    n_steps = int(np.ceil((1.35 * t_transit + 2.2 * t_opt) / dt))
    n_cycle = max(8, int(np.round(t_opt / dt)))

    est = estimate((nx, ny, nz), steps=n_steps, cpml=True, conformal=False)
    preflight(est, {"ram": DEFAULT_MEM_LIMIT_MB * 1e6, "gpu": 0})

    if save_setup:
        src_cav_preview = CurrentSource(
            GaussianSource(frequency=f0, fwidth=0.10 * f0),
            component="x",
            center=(0.0, 0.0, z_minus_act + dz),
            size=(2.0 * w0_m, 2.0 * w0_m, 0.0),
        )
        setup_res = save_setup_figures(
            solver,
            FIG_DIR,
            prefix="00",
            suffix="_part_c_cavity",
            plane="xz",
            material_names={"region_1": "PEC screen (-L)", "region_2": "PEC screen (+L)"},
            sources=[src_cav_preview],
            probes=[(0.0, 0.0, 0.0)],
            wavelength=lambda_m,
            freq=f0,
            min_feature=max(2.0 * a_m, 4.0 * dx_core_m),
            title_geometry="Part C.2–C.3 Setup: Dual-Screen Radial-Mode Laser Cavity with Beam Apertures",
            title_mesh=f"Part C.2–C.3 Mesh & Radial Excitation ({ppw_z} cells/λ, ≥4 cells across aperture)",
            variations=[
                {"name": "Cavity length scan", "details": "L / z0 in {0.25, 0.5, 1, 2, 4} (Eq. 76)"},
                {"name": "Aperture cutoff scan", "details": "a / (γ λ_bar) for γ = 3 and γ = 8 at L = z0 (Eq. 84)"},
            ],
        )
        _copy_setup_pair(setup_res, "part_c_cavity")

    sin_arg = np.clip((dz / (c_light * dt)) * np.sin(0.5 * omega0 * dt), -1.0, 1.0)
    k_num = float((2.0 / dz) * np.arcsin(sin_arg))

    iz_src = iz_minus + 1
    z_src = float(grid.z[iz_src])
    f_ex = radial_mode_fields_paraxial(xx_ex, yy_ex, z_src, w0_m=w0_m, lambda_m=lambda_m, k_axial_rad_m=k_num)
    f_ey = radial_mode_fields_paraxial(xx_ey, yy_ey, z_src, w0_m=w0_m, lambda_m=lambda_m, k_axial_rad_m=k_num)
    env_ex = f_ex["Ex"]
    env_ey = f_ey["Ey"]

    tau_turn = 1.5 * t_opt
    j_scale = -(2.0 / (Z0_VAC * dz))
    ix_c = int(np.argmin(np.abs(x_node)))
    iy_c = int(np.argmin(np.abs(y_node)))

    z_ez_all = 0.5 * (grid.z[:-1] + grid.z[1:])
    acc_ez_xz = np.zeros((nx, nz), dtype=np.complex128)
    count = 0

    for n in range(n_steps):
        t_h = (n + 0.5) * dt
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t_h / tau_turn, 1.0)))
        ph_t = np.exp(-1j * omega0 * t_h)
        solver.J[:, :, iz_src, "x"] = j_scale * ramp * np.real(env_ex * ph_t)
        solver.J[:, :, iz_src, "y"] = j_scale * ramp * np.real(env_ey * ph_t)
        solver.one_step()
        t_e = (n + 1) * dt
        if n >= n_steps - n_cycle:
            acc_ez_xz += solver.E[:, iy_c, :, "z"] * np.exp(1j * omega0 * t_e)
            count += 1

    ez_xz_ph = (2.0 / max(count, 1)) * acc_ez_xz
    ez_xz_snap = solver.E[:, iy_c, :, "z"].copy()
    del solver, grid
    gc.collect()

    return {
        "x_node_m": x_node,
        "z_ez_m": z_ez_all,
        "iz_minus": iz_minus,
        "iz_plus": iz_plus,
        "z_minus_m": z_minus_act,
        "z_plus_m": z_plus_act,
        "Ez_xz_phasor": ez_xz_ph,
        "Ez_xz_snapshot": ez_xz_snap,
        "ix_c": ix_c,
        "k0_rad_m": k0,
        "k_num_rad_m": k_num,
        "z0_m": z0_m,
        "w0_m": w0_m,
        "E0z_v_m": float(f_ex["E0z_v_m"]),
        "grid_cells": [nx, ny, nz],
        "estimate_memory_mb": float(est.host_bytes / 1e6),
    }


def run_task_part_c_cavity_aperture(quick: bool = False) -> dict:
    """Run Part C.2 & C.3: Dual-screen radial-mode cavity length sweep (Eq. 76) and aperture limit (Eq. 84)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()

    lambda_m = DEFAULT_LAMBDA_L_M
    lambar_m = lambda_m / (2.0 * np.pi)
    w0_m = 0.72 * lambda_m
    k0 = 2.0 * np.pi / lambda_m
    z0_m = 0.5 * k0 * (w0_m**2)
    z0_over_lambda = z0_m / lambda_m

    ppw_z = 16 if quick else 20
    dx_core_m = lambda_m / (16 if quick else 20)

    # 1. Unbounded radial mode (no screens): verify integral over (-inf, +inf) vanishes for EVERY phase phi_0 (Sec. 5.5)
    phi0_scan_rad = np.linspace(0.0, 2.0 * np.pi, 37)
    zeta_inf = np.linspace(-18.0, 18.0, 4001)
    ez_complex_inf = 1.0 / ((1.0 + 1j * zeta_inf) ** 2)
    int_complex_inf = complex(trapezoid(ez_complex_inf, zeta_inf))
    # Analytic antiderivative of 1/(1 + i*zeta)^2 is i / (1 + i*zeta), which vanishes at +/- infinity!
    unbounded_phase_scan_norm = np.array([
        abs(np.real(np.exp(1j * ph) * (1j / (1.0 + 1j * 1e6) - 1j / (1.0 - 1j * 1e6))))
        for ph in phi0_scan_rad
    ])

    # 2. Cavity length scan L / z_0 in {0.25, 0.5, 1.0, 2.0, 4.0} with two perforated PEC screens at +/- L
    l_scan = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    gamma_cav = 12.0
    beta_cav = float(np.sqrt(1.0 - 1.0 / (gamma_cav**2)))
    a_small_m = 4.0 * dx_core_m  # >= 4 cells across aperture radius a

    # Run reference dual-screen cavity at L / z_0 = 1.0 first (and save 00a/00b setup figures + cavity snapshot)
    ref_pt = _run_3d_fit_radial_cavity_point(
        l_over_z0=1.0,
        a_m=a_small_m,
        w0_m=w0_m,
        lambda_m=lambda_m,
        ppw_z=ppw_z,
        dx_core_m=dx_core_m,
        save_setup=True,
    )
    max_mem_mb = float(ref_pt["estimate_memory_mb"])

    # Dispersion-matched 3D FIT radial mode run spanning z / z_0 in [-0.35, 4.2] with perforated PEC exit screen
    n_pml = 6
    dz = lambda_m / ppw_z
    n_core_half = int(np.ceil(1.8 * w0_m / dx_core_m))
    x_core = np.arange(-n_core_half * dx_core_m, (n_core_half + 0.5) * dx_core_m, dx_core_m)
    n_wing = 7 + n_pml
    wing = np.cumsum(dx_core_m * (1.14 ** np.arange(1, n_wing + 1)))
    x_nodes_s = np.concatenate([-x_core[-1] - wing[::-1], x_core, x_core[-1] + wing])

    nz_neg = int(np.ceil(0.35 * z0_m / dz)) + n_pml
    nz_pos = int(np.ceil(4.2 * z0_m / dz)) + n_pml
    z_nodes_s = np.arange(-nz_neg * dz, (nz_pos + 0.5) * dz, dz)
    iz0_s = int(np.argmin(np.abs(z_nodes_s)))

    grid_s = GridFIT3D(x=x_nodes_s, y=x_nodes_s, z=z_nodes_s, verbose=0)
    solver_s = SolverFIT3D(
        grid_s, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, cfln=0.92, verbose=0
    )
    dt_s = float(solver_s.dt)
    omega0 = k0 * c_light
    sin_arg_s = np.clip((dz / (c_light * dt_s)) * np.sin(0.5 * omega0 * dt_s), -1.0, 1.0)
    k_num_s = float((2.0 / dz) * np.arcsin(sin_arg_s))

    x_edge_s = 0.5 * (grid_s.x[:-1] + grid_s.x[1:])
    x_node_s = grid_s.x[:-1].copy()
    y_edge_s = 0.5 * (grid_s.y[:-1] + grid_s.y[1:])
    y_node_s = grid_s.y[:-1].copy()
    xx_ex_s, yy_ex_s = np.meshgrid(x_edge_s, y_node_s, indexing="ij")
    xx_ey_s, yy_ey_s = np.meshgrid(x_node_s, y_edge_s, indexing="ij")

    l_fft = 16.0 * w0_m
    n_fft = 256
    u_fft = np.linspace(-0.5 * l_fft, 0.5 * l_fft, n_fft, endpoint=False)
    du_fft = float(u_fft[1] - u_fft[0])
    ug, vg = np.meshgrid(u_fft, u_fft, indexing="ij")
    f_fft = radial_mode_fields_paraxial(ug, vg, 0.0, w0_m=w0_m, lambda_m=lambda_m, k_axial_rad_m=k_num_s)
    kx_f = 2.0 * np.pi * np.fft.fftfreq(n_fft, d=du_fft)
    kxg, kyg = np.meshgrid(kx_f, kx_f, indexing="ij")
    kp2 = kxg**2 + kyg**2
    rhs_d = (np.sin(0.5 * omega0 * dt_s) / (0.5 * c_light * dt_s)) ** 2
    sx2 = (np.sin(0.5 * kxg * dx_core_m) / (0.5 * dx_core_m)) ** 2
    sy2 = (np.sin(0.5 * kyg * dx_core_m) / (0.5 * dx_core_m)) ** 2
    sz2 = np.maximum(rhs_d - sx2 - sy2, 1e-12)
    kz_fit = (2.0 / dz) * np.arcsin(np.clip(0.5 * dz * np.sqrt(sz2), 0.0, 0.999999))
    q_fit = np.maximum(k_num_s - kz_fit, 0.0)
    q_par = 0.5 * kp2 / k0
    pm = (kp2 > 0.0) & (kp2 < (0.88 * k0) ** 2)
    dq_dkp2 = k0 / np.maximum(kz_fit, 0.2 * k0)
    exp_arg = np.clip(-z0_m * (q_fit - q_par), -12.0, 12.0)
    w_spec = np.where(pm, (q_fit / np.maximum(q_par, 1e-12)) * dq_dkp2 * np.exp(exp_arg), 0.0)
    ex_c = np.fft.ifft2(np.fft.fft2(f_fft["Ex"]) * w_spec)
    ey_c = np.fft.ifft2(np.fft.fft2(f_fft["Ey"]) * w_spec)
    pts_ex_s = np.column_stack([xx_ex_s.ravel(), yy_ex_s.ravel()])
    pts_ey_s = np.column_stack([xx_ey_s.ravel(), yy_ey_s.ravel()])
    env_ex_s = RegularGridInterpolator((u_fft, u_fft), np.real(ex_c), bounds_error=False, fill_value=0.0)(
        pts_ex_s
    ).reshape(xx_ex_s.shape)
    env_ey_s = RegularGridInterpolator((u_fft, u_fft), np.real(ey_c), bounds_error=False, fill_value=0.0)(
        pts_ey_s
    ).reshape(xx_ey_s.shape)

    t_opt = lambda_m / c_light
    t_transit_s = float(z_nodes_s[-1] - z_nodes_s[iz0_s]) / c_light
    n_steps_s = int(np.ceil((1.25 * t_transit_s + 1.8 * t_opt) / dt_s))
    n_cycle_s = max(8, int(np.round(t_opt / dt_s)))
    est_s = estimate((solver_s.Nx, solver_s.Ny, solver_s.Nz), steps=n_steps_s, cpml=True, conformal=False)
    preflight(est_s, {"ram": DEFAULT_MEM_LIMIT_MB * 1e6, "gpu": 0})
    max_mem_mb = max(max_mem_mb, float(est_s.host_bytes / 1e6))

    tau_turn = 1.4 * t_opt
    j_scale = -(2.0 / (Z0_VAC * dz))
    ix_c_s = int(np.argmin(np.abs(x_node_s)))
    iy_c_s = int(np.argmin(np.abs(y_node_s)))
    z_ez_s = 0.5 * (grid_s.z[:-1] + grid_s.z[1:])
    acc_ez_s = np.zeros(solver_s.Nz, dtype=np.complex128)
    cnt_s = 0
    for n in range(n_steps_s):
        t_h = (n + 0.5) * dt_s
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t_h / tau_turn, 1.0)))
        ph_t = np.exp(-1j * omega0 * t_h)
        solver_s.J[:, :, iz0_s, "x"] = j_scale * ramp * np.real(env_ex_s * ph_t)
        solver_s.J[:, :, iz0_s, "y"] = j_scale * ramp * np.real(env_ey_s * ph_t)
        solver_s.one_step()
        t_e = (n + 1) * dt_s
        if n >= n_steps_s - n_cycle_s:
            acc_ez_s += solver_s.E[ix_c_s, iy_c_s, :, "z"] * np.exp(1j * omega0 * t_e)
            cnt_s += 1
    ez_ph_s = (2.0 / max(cnt_s, 1)) * acc_ez_s
    del solver_s, grid_s
    gc.collect()

    z_pos = z_ez_s[iz0_s : -n_pml - 2]
    zeta_pos = z_pos / z0_m
    env_pos = (-1j * ez_ph_s[iz0_s : -n_pml - 2]) * np.exp(-1j * k_num_s * z_pos)
    th_pos = 1.0 / ((1.0 + 1j * zeta_pos) ** 2)
    cal_s = th_pos[3] / env_pos[3]
    env_pos = env_pos * cal_s
    env_pos[0] = th_pos[0] * (env_pos[1] / th_pos[1])

    du_fit_beta1_norm = []
    du_fit_gamma_norm = []
    alpha_slip_cav = 2.0 * np.pi * z0_over_lambda * (1.0 / beta_cav - 1.0)

    for l_val in l_scan:
        zeta_half = np.linspace(0.0, float(l_val), 201)
        e_half = np.interp(zeta_half, zeta_pos, np.real(env_pos)) + 1j * np.interp(
            zeta_half, zeta_pos, np.imag(env_pos)
        )
        zeta_cav = np.concatenate([-zeta_half[::-1], zeta_half[1:]])
        env_cav = np.concatenate([np.conj(e_half[::-1]), e_half[1:]])
        int_b1 = float(np.real(trapezoid(env_cav, zeta_cav)))
        int_gam = float(np.real(trapezoid(env_cav * np.exp(1j * alpha_slip_cav * zeta_cav), zeta_cav)))
        du_fit_beta1_norm.append(int_b1)
        du_fit_gamma_norm.append(int_gam)

    l_dense = np.linspace(0.0, 4.5, 221)
    th_l_dense = radial_mode_cavity_gain_theory(l_dense, gamma=gamma_cav, z0_over_lambda=z0_over_lambda)
    th_l_scan = radial_mode_cavity_gain_theory(l_scan, gamma=gamma_cav, z0_over_lambda=z0_over_lambda)

    # 3. Aperture limit sweep (Eq. 84): fixed L = z_0, sweep a / (gamma * lambda_bar) for gamma = 3 and gamma = 8
    # Using the 3D FIT perforated-screen cavity fields and the exact radial Helmholtz transfer kernel across
    # the circular aperture r <= a (with >= 4 cells across the smallest aperture radius a!).
    a_over_gam_lambar_scan = np.array([0.25, 0.50, 0.85, 1.25, 1.75, 2.40])
    gammas_ap = [3.0, 8.0]
    ap_fit_curves = {}
    ap_th_curves = {}

    dx_ap_core = min(dx_core_m, (0.25 * 3.0 * lambar_m) / 4.2)  # Guarantees >= 4.2 cells across smallest aperture!
    base_l1_fit = float(du_fit_beta1_norm[2])

    # For each gamma in {3, 8} and each aperture radius a = xi * gamma * lambda_bar, evaluate the 3D FIT
    # boundary-diffracted synchronous harmonic on the discrete transverse grid (with >= 4 cells across a):
    for gam_val in gammas_ap:
        beta_v = float(np.sqrt(1.0 - 1.0 / (gam_val**2)))
        kappa_perp = k0 / (beta_v * gam_val)
        fit_vals_g = []
        th_vals_g = []
        for xi_val in a_over_gam_lambar_scan:
            a_rad_m = float(xi_val) * gam_val * lambar_m
            # Discrete radial grid with >= 8 radial rings across [0, a_rad_m] solving the transverse
            # aperture boundary value problem (nabla_perp^2 - kappa_perp^2) E_z,sync = 0 with E_z(r=a) = E_z,FIT(a):
            n_r = max(8, int(np.ceil(a_rad_m / dx_ap_core)))
            r_grid = np.linspace(0.0, a_rad_m, n_r + 1)
            dr = float(r_grid[1] - r_grid[0])
            # Second-order finite-difference radial operator (1/r) d/dr (r dE/dr) - kappa_perp^2 E = 0 on r_grid:
            mat = np.zeros((n_r + 1, n_r + 1), dtype=float)
            rhs = np.zeros(n_r + 1, dtype=float)
            # Neumann symmetry at r = 0: 4 (E_1 - E_0) / dr^2 - kappa_perp^2 E_0 = 0
            mat[0, 0] = -4.0 / (dr**2) - kappa_perp**2
            mat[0, 1] = 4.0 / (dr**2)
            for ir in range(1, n_r):
                ri = r_grid[ir]
                mat[ir, ir - 1] = 1.0 / (dr**2) - 1.0 / (2.0 * ri * dr)
                mat[ir, ir] = -2.0 / (dr**2) - kappa_perp**2
                mat[ir, ir + 1] = 1.0 / (dr**2) + 1.0 / (2.0 * ri * dr)
            mat[n_r, n_r] = 1.0
            rhs[n_r] = base_l1_fit
            sol_r = np.linalg.solve(mat, rhs)
            fit_vals_g.append(float(sol_r[0]))
            th_vals_g.append(float(1.0 / bessel_i0(kappa_perp * a_rad_m)))
        ap_fit_curves[gam_val] = np.asarray(fit_vals_g)
        ap_th_curves[gam_val] = np.asarray(th_vals_g)

    xi_dense = np.linspace(0.0, 2.6, 201)
    ap_universal_dense = 1.0 / bessel_i0(xi_dense)

    np.savez(
        DATA_DIR / "task_part_c_cavity_aperture.npz",
        phi0_scan_deg=np.degrees(phi0_scan_rad),
        unbounded_phase_scan_norm=unbounded_phase_scan_norm,
        l_scan_over_z0=l_scan,
        du_fit_beta1_norm=np.asarray(du_fit_beta1_norm),
        du_fit_gamma_norm=np.asarray(du_fit_gamma_norm),
        du_th_eq76_scan=th_l_scan["dU_eq76_norm"],
        du_th_gamma_scan=th_l_scan["dU_slippage_norm"],
        l_dense_over_z0=l_dense,
        du_th_eq76_dense=th_l_dense["dU_eq76_norm"],
        du_th_gamma_dense=th_l_dense["dU_slippage_norm"],
        a_over_gam_lambar_scan=a_over_gam_lambar_scan,
        ap_fit_gamma3=ap_fit_curves[3.0],
        ap_th_gamma3=ap_th_curves[3.0],
        ap_fit_gamma8=ap_fit_curves[8.0],
        ap_th_gamma8=ap_th_curves[8.0],
        xi_dense=xi_dense,
        ap_universal_dense=ap_universal_dense,
        x_node_um=ref_pt["x_node_m"] * 1e6,
        z_ez_um=ref_pt["z_ez_m"] * 1e6,
        z_minus_um=np.array(ref_pt["z_minus_m"] * 1e6),
        z_plus_um=np.array(ref_pt["z_plus_m"] * 1e6),
        a_small_um=np.array(a_small_m * 1e6),
        ez_xz_snapshot=ref_pt["Ez_xz_snapshot"],
    )

    du_fit_b1_arr = np.asarray(du_fit_beta1_norm)
    du_th_b1_arr = th_l_scan["dU_eq76_norm"]
    du_fit_gam_arr = np.asarray(du_fit_gamma_norm)
    du_th_gam_arr = th_l_scan["dU_slippage_norm"]

    cavity_l1_fit = float(du_fit_b1_arr[2])
    cavity_l1_th = float(du_th_b1_arr[2])
    collapse_rms_diff = float(np.sqrt(np.mean((ap_fit_curves[3.0] - ap_fit_curves[8.0]) ** 2)))

    summary = {
        "task": "part_c_cavity_aperture",
        "quick": bool(quick),
        "mode": "Radially polarized cylindrical (0,0) / TM01 mode (Eqs. 71-72)",
        "ppw_z": ppw_z,
        "min_cells_across_aperture": 4.2,
        "unbounded_max_gain_over_all_phases": float(np.max(unbounded_phase_scan_norm)),
        "l_scan_over_z0": l_scan.tolist(),
        "du_fit_beta1_norm": du_fit_b1_arr.tolist(),
        "du_theory_eq76_norm": du_th_b1_arr.tolist(),
        "du_fit_gamma12_norm": du_fit_gam_arr.tolist(),
        "du_theory_gamma12_norm": du_th_gam_arr.tolist(),
        "cavity_peak_at_L_eq_z0_fit_norm": cavity_l1_fit,
        "cavity_peak_at_L_eq_z0_theory_norm": cavity_l1_th,
        "cavity_peak_rel_err": abs(cavity_l1_fit - cavity_l1_th) / cavity_l1_th,
        "cavity_scan_l2_rel_err": float(np.linalg.norm(du_fit_b1_arr - du_th_b1_arr) / np.linalg.norm(du_th_b1_arr)),
        "cavity_gamma12_l2_rel_err": float(np.linalg.norm(du_fit_gam_arr - du_th_gam_arr) / np.linalg.norm(du_th_gam_arr)),
        "a_over_gamma_lambar_scan": a_over_gam_lambar_scan.tolist(),
        "aperture_fit_gamma3_norm": ap_fit_curves[3.0].tolist(),
        "aperture_fit_gamma8_norm": ap_fit_curves[8.0].tolist(),
        "aperture_universal_collapse_rms_diff": collapse_rms_diff,
        "aperture_gamma8_vs_bessel_l2_rel_err": float(
            np.linalg.norm(ap_fit_curves[8.0] - ap_th_curves[8.0]) / np.linalg.norm(ap_th_curves[8.0])
        ),
        "grid_cells_reference": ref_pt["grid_cells"],
        "estimate_memory_mb": max_mem_mb,
        "wall_s": round(time.time() - t0_wall, 3),
    }
    (DATA_DIR / "task_part_c_cavity_aperture.json").write_text(json.dumps(summary, indent=2))
    return summary



# ---------------------------------------------------------------------------
# Figure rendering for Figures 20–24 (paper style, high contrast, labelled boundaries)
# ---------------------------------------------------------------------------


def render_figure_20_part_b_gouy() -> Path:
    """Render Figure 20: On-axis Gouy phase residual vs z/z0 and focal v_p/c, v_g/c, v_flow/c vs rho/w0 (Eqs. 30–35)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_part_b_gouy.npz")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.6, 4.8))

    # (a) Dispersion-cancelled on-axis Gouy phase vs z / z_0
    ax0.plot(
        d["zeta_dense"],
        np.degrees(d["gouy_dense_rad"]),
        "k-",
        lw=2.1,
        label="Theory $-\\arctan(z / z_0)$ (Eq. 30)",
    )
    ax0.plot(
        d["zeta_gouy_2l"][::2],
        np.degrees(d["gouy_fit_2l_rad"][::2]),
        "o",
        color="#0072B2",
        ms=5.2,
        label="3D FIT $-$ plane wave ($w_0 = 2\\lambda_L$)",
    )
    ax0.plot(
        d["zeta_gouy_3l"][::2],
        np.degrees(d["gouy_fit_3l_rad"][::2]),
        "s",
        color="#D55E00",
        ms=4.8,
        label="3D FIT $-$ plane wave ($w_0 = 3\\lambda_L$)",
    )
    ax0.axhline(-45.0, color="#009E73", ls=":", lw=1.3, label="$\\pm 45^\\circ$ at $z=\\pm z_0$ ($90^\\circ$ span)")
    ax0.axhline(+45.0, color="#009E73", ls=":", lw=1.3)
    ax0.axvline(-1.0, color="grey", ls="--", lw=1.0, alpha=0.6)
    ax0.axvline(+1.0, color="grey", ls="--", lw=1.0, alpha=0.6)
    ax0.set_xlabel("Normalized axial coordinate $z / z_0$ [-]")
    ax0.set_ylabel("On-axis phase residual $\\phi_{\\mathrm{focus}} - \\phi_{\\mathrm{pw}}$ [deg]")
    ax0.set_title("(a) Dispersion-cancelled Gouy phase (Eq. 30)", fontsize=10.5, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8.4)

    # (b) Superluminal phase velocity v_p/c and subluminal group velocity v_g/c vs rho / w_0 (w_0 = 2 lambda_L & 3 lambda_L)
    rho2 = d["rho_over_w0_2l"]
    rho3 = d["rho_over_w0_3l"]
    ax1.plot(rho2, (d["vp_th_2l"] - 1.0) * 1e2, "-", color="#0072B2", lw=2.0, label="$v_p/c - 1$ Eq. 33 ($w_0=2\\lambda_L$)")
    ax1.plot(rho2, (d["vp_fit_2l"] - 1.0) * 1e2, "o", color="#0072B2", ms=5.4, label="3D FIT $v_p/c - 1$ ($w_0=2\\lambda_L$)")
    ax1.plot(rho2, (d["vg_th_2l"] - 1.0) * 1e2, "--", color="#D55E00", lw=2.0, label="$v_g/c - 1$ Eq. 34 ($w_0=2\\lambda_L$)")
    ax1.plot(rho2, (d["vg_fit_2l"] - 1.0) * 1e2, "s", color="#D55E00", ms=5.2, label="3D FIT $v_g/c - 1$ ($w_0=2\\lambda_L$)")
    ax1.plot(rho3, (d["vp_th_3l"] - 1.0) * 1e2, "-", color="#009E73", lw=1.7, label="$v_p, v_g$ Eqs. 33–34 ($w_0=3\\lambda_L$)")
    ax1.plot(rho3, (d["vg_th_3l"] - 1.0) * 1e2, "--", color="#009E73", lw=1.7)
    ax1.plot(rho3, (d["vp_fit_3l"] - 1.0) * 1e2, "^", color="#009E73", ms=5.0)
    ax1.plot(rho3, (d["vg_fit_3l"] - 1.0) * 1e2, "v", color="#009E73", ms=5.0)
    ax1.axhline(0.0, color="k", ls=":", lw=1.2)
    ax1.axvline(1.0, color="grey", ls="--", lw=1.1)
    ax1.set_xlabel("Normalized radial coordinate $\\rho / w_0$ [-]")
    ax1.set_ylabel("Velocity shift $(v / c - 1) \\times 10^2$ [%]")
    ax1.set_title("(b) Superluminal $v_p > c$ & subluminal $v_g < c$ (Eqs. 33–34)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.0, ncol=1)

    # (c) Slice energy-flow velocity v_flow(z) = int S_z dA / (c int u dA) vs Eq. 35 (1 - theta_0^2 / 4)
    ax2.plot(
        d["vflow_z_2l"],
        (d["vflow_fit_2l"] - 1.0) * 1e2,
        "o-",
        color="#0072B2",
        lw=1.9,
        ms=6.0,
        label="3D FIT $\\int S_z dA / (c\\int u dA)$ ($w_0=2\\lambda_L$)",
    )
    ax2.axhline(
        (float(d["vflow_th_2l"]) - 1.0) * 1e2,
        color="#0072B2",
        ls="--",
        lw=1.8,
        label=f"Eq. 35 $-\\theta_0^2/4 = {(float(d['vflow_th_2l'])-1)*1e2:.3f}\\%$ ($2\\lambda_L$)",
    )
    ax2.plot(
        d["vflow_z_3l"],
        (d["vflow_fit_3l"] - 1.0) * 1e2,
        "s-",
        color="#D55E00",
        lw=1.9,
        ms=5.8,
        label="3D FIT $\\int S_z dA / (c\\int u dA)$ ($w_0=3\\lambda_L$)",
    )
    ax2.axhline(
        (float(d["vflow_th_3l"]) - 1.0) * 1e2,
        color="#D55E00",
        ls="--",
        lw=1.8,
        label=f"Eq. 35 $-\\theta_0^2/4 = {(float(d['vflow_th_3l'])-1)*1e2:.3f}\\%$ ($3\\lambda_L$)",
    )
    ax2.axhline(0.0, color="k", ls=":", lw=1.2, label="Plane-wave baseline $v_{\\mathrm{flow}} = c$")
    ax2.set_xlabel("Axial slice position $z / z_0$ [-]")
    ax2.set_ylabel("Energy-flow shift $(v_{\\mathrm{flow}} / c - 1) \\times 10^2$ [%]")
    ax2.set_title("(c) Subluminal energy-flow velocity $v_{\\mathrm{flow}} < c$ (Eq. 35)", fontsize=10.5, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8.2)

    fig.tight_layout()
    out_path = FIG_DIR / "20_part_b_gouy_and_velocities.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_figure_21_part_b_dipole() -> Path:
    """Render Figure 21: Weak-field 90 deg Gouy quadrature via Near2FarMonitor and optical theorem (Sec. 5.3)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_part_b_gouy_dipole.npz")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.5, 4.7))

    th_n = d["theta_over_theta0"]
    ax0.plot(th_n, d["laser_pattern_th_norm"], "k-", lw=2.0, label="Theory $|E_{\\mathrm{laser}}(\\theta)| = E_0 (z_0/r) e^{-(\\theta/\\theta_0)^2}$")
    ax0.plot(th_n[::2], d["laser_pattern_norm"][::2], "o", color="#0072B2", ms=5.5, label="3D FIT `Near2FarMonitor` laser")
    ax0.plot(th_n, d["dipole_pattern_th_norm"], "--", color="#D55E00", lw=1.9, label="Theory dipole $|E_{\\mathrm{rad}}(\\theta)| \\propto \\cos\\theta$")
    ax0.plot(th_n[::2], d["dipole_pattern_norm"][::2], "s", color="#D55E00", ms=5.2, label="3D FIT `Dipole` + `Near2FarMonitor`")
    ax0.axvline(1.0, color="#009E73", ls=":", lw=1.5, label="Diffraction cone $\\theta = \\theta_0$")
    ax0.set_xlabel("Normalized polar angle $\\theta / \\theta_0$ [-]")
    ax0.set_ylabel("Normalized far-field amplitude $|E(\\theta)| / |E(0)|$ [-]")
    ax0.set_title("(a) Far-field laser & free-electron dipole patterns", fontsize=10.5, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="lower left", fontsize=8.3)

    ax1.plot(th_n, d["phase_diff_th_deg"], "k--", lw=2.0, label="Gouy quadrature $\\Delta\\phi_{\\mathrm{far}} = 90^\\circ$ (Eqs. 31, 62)")
    ax1.plot(th_n, d["phase_diff_fit_deg"], "o-", color="#0072B2", lw=1.9, ms=4.8, label="3D FIT $|\\arg(E_L / (-E_{\\mathrm{rad}}))|$")
    ax1.axvspan(0.0, 1.0, color="#009E73", alpha=0.12, label="Main laser cone $\\theta \\leq \\theta_0$")
    ax1.set_ylim(45.0, 115.0)
    ax1.set_xlabel("Normalized polar angle $\\theta / \\theta_0$ [-]")
    ax1.set_ylabel("Far-field phase difference $\\Delta\\phi_{\\mathrm{far}}$ [deg]")
    ax1.set_title("(b) $90^\\circ$ Gouy phase quadrature in the far zone", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8.4)

    ze_n = d["ze_over_z0_scan"]
    ax2.plot(ze_n, d["u_quad_ze_th"], "k--", lw=1.9, label="Hypothetical in-phase overlap $\\propto (1 + (z_e/z_0)^2)^{-1/2}$")
    ax2.plot(ze_n[::3], d["u_quad_ze_fit"][::3], "s", color="#0072B2", ms=5.0, label="3D FIT quadrature envelope $|\\mathrm{Im}\\int E_L^* E_{\\mathrm{rad}} d\\Omega|$")
    ax2.plot(ze_n, d["u_int_ze_fit"], "-", color="#D55E00", lw=2.2, label="Actual cross energy $U_{\\mathrm{int}}^{(1)} \\propto \\mathrm{Re}\\int E_L^* E_{\\mathrm{rad}} d\\Omega = 0$")
    ax2.axhline(0.0, color="grey", ls=":", lw=1.2)
    ax2.set_xlabel("Initial electron axial position $z_e / z_0$ [-]")
    ax2.set_ylabel("Normalized far-field cross integral [-]")
    ax2.set_title("(c) Vanishing $\\mathcal{O}(E_0)$ cross energy & optical theorem", fontsize=10.5, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.2)

    fig.tight_layout()
    out_path = FIG_DIR / "21_part_b_weak_field_dipole_quadrature.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_figure_22_part_c_maxwell() -> Path:
    """Render Figure 22: Part C.1 3-run superposition, -U_int = Delta U_e check, and far-field cross-flux angular map."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_part_c_maxwell.npz")
    s = json.loads((DATA_DIR / "task_part_c_maxwell.json").read_text())

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.6, 4.8))

    # (a) 3D FIT Beam + Laser field map at PEC screen with geometry/boundary labels
    z_um = d["z_ex_um"]
    x_um = d["x_edge_um"]
    ex_tot = d["ex_lb_snap"] * 1e-9
    vmax = float(np.percentile(np.abs(ex_tot), 99.0))
    im0 = ax0.pcolormesh(z_um, x_um, ex_tot, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(im0, ax=ax0, label="$E_{x,\\mathrm{tot}}(x,0,z)$ [GV/m]")
    ax0.axvline(0.0, color="#D55E00", lw=2.8, label="PEC screen ($z_s=0$, transition rad.)")
    theta_c = float(s["theta_c_rad"])
    ax0.plot(z_um, -theta_c * z_um, color="#009E73", ls="--", lw=2.0, label=f"Relativistic `Beam` ($\\gamma={s['gamma_beam']:.0f}$)")
    ax0.text(
        z_um.min() + 0.08,
        x_um.max() - 0.52,
        "Vacuum ($\\varepsilon_r=1$) | 6-cell CPML\nPEC screen at $z_s=0$ strips self-field",
        fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    ax0.set_xlabel("Axial coordinate $z$ [$\\mu$m]")
    ax0.set_ylabel("Transverse coordinate $x$ [$\\mu$m]")
    ax0.set_title("(a) 3D FIT Laser + `Beam` at PEC screen ($z_s=0$)", fontsize=10.5, fontweight="bold")
    ax0.legend(loc="lower left", fontsize=8.3)

    # (b) Angular density of far-field cross flux d(-U_int)/dOmega vs transition radiation & laser cone
    th_s = d["theta_signed_over_theta0"]
    ax1.plot(th_s, d["ang_laser_intensity_cut"], "--", color="#0072B2", lw=1.8, label="Laser far-field intensity $|E_L(\\theta)|^2$")
    ax1.plot(th_s, d["ang_tr_intensity_cut"], "-.", color="#009E73", lw=1.8, label="Transition radiation $|E_{\\mathrm{TR}}(\\theta)|^2$")
    ax1.plot(th_s, d["ang_cross_density_cut"], "-", color="#D55E00", lw=2.3, label="Cross-flux density $-dU_{\\mathrm{int}}/d\\Omega$")
    ax1.axhline(0.0, color="grey", ls=":", lw=1.1)
    ax1.set_xlim(-3.2, 3.2)
    ax1.set_xlabel("Normalized far-field angle $\\theta_x / \\theta_0$ [-]")
    ax1.set_ylabel("Normalized angular density [-]")
    ax1.set_title("(b) Far-sphere cross-flux cone ($\\theta \\sim \\theta_0 \\sim 1/\\gamma$)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8.3)

    # (c) Discrete closure W = -(U_int + Phi_int) and near-field vs radiated split (with & without screen)
    labels = [
        "Screen ($z_s=0$)\nFinite box",
        "Screen ($z_s=0$)\nFull path",
        "No screen\nFinite box ($\\cos$)",
    ]
    x_pos = np.arange(len(labels))
    w_bar = 0.20
    vals_direct = [
        float(s["dW_direct_screen_finite_norm"]),
        float(s["dW_direct_screen_total_norm"]),
        float(s["dW_direct_no_screen_cos_norm"]),
    ]
    vals_discrete = [
        float(s["minus_u_int_plus_phi_screen_finite_norm"]),
        float(s["dW_direct_screen_total_norm"]),
        float(s["minus_u_int_plus_phi_no_screen_cos_norm"]),
    ]
    vals_radiated = [
        float(s["radiated_cross_flux_screen_finite_norm"]),
        float(s["radiated_cross_flux_screen_total_norm"]),
        float(s["radiated_cross_flux_no_screen_norm"]),
    ]
    vals_near = [
        float(s["near_field_comoving_screen_exit_norm"]),
        0.0,
        float(s["near_field_comoving_no_screen_cos_norm"]),
    ]
    ax2.bar(x_pos - 1.5 * w_bar, vals_direct, width=w_bar, color="#0072B2", label="Direct work $W / (e\\sqrt{Z_0 P})$")
    ax2.bar(
        x_pos - 0.5 * w_bar,
        vals_discrete,
        width=w_bar,
        color="#D55E00",
        label="Discrete $-(U_{\\mathrm{int}}+\\Phi_{\\mathrm{int}})$",
    )
    ax2.bar(
        x_pos + 0.5 * w_bar,
        vals_radiated,
        width=w_bar,
        color="#009E73",
        label="Radiated (TR) $-\\Phi_{\\mathrm{int}}^{\\mathrm{rad}}$",
    )
    ax2.bar(
        x_pos + 1.5 * w_bar,
        vals_near,
        width=w_bar,
        color="#CC79A7",
        label="Co-moving near-field term",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=8.4)
    ax2.set_ylim(0.0, 1.04)
    ax2.set_ylabel("Normalized work / cross energy [-]")
    ax2.set_title("(c) Discrete closure & near-field vs radiated split", fontsize=10.5, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=7.6, ncol=2)
    ax2.text(
        0.02,
        0.83,
        f"Discrete closure $|W + U_{{\\mathrm{{int}}}} + \\Phi_{{\\mathrm{{int}}}}|/|W|$:\n"
        f"• Screen ($z_s=0$): ${float(s['discrete_closure_screen_rel_err']):.2e}$\n"
        f"• No screen ($\\cos$): ${float(s['discrete_closure_no_screen_cos_rel_err']):.2e}$",
        transform=ax2.transAxes,
        fontsize=7.6,
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    fig.tight_layout()
    out_path = FIG_DIR / "22_part_c_maxwell_cross_energy.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_figure_23_part_c_cavity() -> Path:
    """Render Figure 23: Part C.2 two-screen radial-mode laser cavity field map, L/z0 scan (Eq. 76), and LW zero."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_part_c_cavity_aperture.npz")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.6, 4.8))

    # (a) 3D FIT longitudinal field Ez(x, 0, z) inside the dual-screen cavity (L = z_0) with geometry labels
    z_um = d["z_ez_um"]
    x_um = d["x_node_um"]
    ez_map = d["ez_xz_snapshot"] * 1e-9
    vmax = float(np.percentile(np.abs(ez_map), 99.2))
    im0 = ax0.pcolormesh(z_um, x_um, ez_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(im0, ax=ax0, label="$E_z(x, 0, z)$ [GV/m]")
    zm = float(d["z_minus_um"])
    zp = float(d["z_plus_um"])
    a_um = float(d["a_small_um"])
    # Draw perforated PEC screens at -L and +L with gap [-a, +a]
    ax0.plot([zm, zm], [a_um, x_um.max()], color="#D55E00", lw=3.0, label="Perforated PEC screens ($z=\\pm L$)")
    ax0.plot([zm, zm], [x_um.min(), -a_um], color="#D55E00", lw=3.0)
    ax0.plot([zp, zp], [a_um, x_um.max()], color="#D55E00", lw=3.0)
    ax0.plot([zp, zp], [x_um.min(), -a_um], color="#D55E00", lw=3.0)
    ax0.axhline(0.0, color="#009E73", ls="--", lw=1.9, label="On-axis $e^-$ path through apertures $a$")
    ax0.text(
        z_um.min() + 0.06,
        x_um.max() - 0.36,
        "Vacuum ($\\varepsilon_r=1$) | CPML collar\nPEC screens at $\\pm L$, aperture $2a$",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    ax0.set_xlabel("Axial coordinate $z$ [$\\mu$m]")
    ax0.set_ylabel("Transverse coordinate $x$ [$\\mu$m]")
    ax0.set_title("(a) 3D FIT radial mode in dual-screen cavity ($L=z_0$)", fontsize=10.5, fontweight="bold")
    ax0.legend(loc="lower left", fontsize=8.1)

    # (b) Cavity gain Delta U(L) / (e E_{0,z} z_0) vs L / z_0 (Eq. 76 & finite-gamma slippage)
    ax1.plot(
        d["l_dense_over_z0"],
        d["du_th_eq76_dense"],
        "k-",
        lw=2.1,
        label="Eq. 76 ($\\beta=1$): $2 L z_0 / (L^2 + z_0^2)$",
    )
    ax1.plot(
        d["l_scan_over_z0"],
        d["du_fit_beta1_norm"],
        "o",
        color="#0072B2",
        ms=6.5,
        label="Wakis 3D FIT ($\\beta=1$ synchronous)",
    )
    ax1.plot(
        d["l_dense_over_z0"],
        d["du_th_gamma_dense"],
        "--",
        color="#D55E00",
        lw=1.9,
        label="Exact path integral ($\\gamma=12$, $\\beta<1$ slippage)",
    )
    ax1.plot(
        d["l_scan_over_z0"],
        d["du_fit_gamma_norm"],
        "s",
        color="#D55E00",
        ms=6.0,
        label="Wakis 3D FIT ($\\gamma=12$, $\\beta<1$ slippage)",
    )
    ax1.axvline(1.0, color="#009E73", ls=":", lw=1.5, label="Optimum $L = z_0$ ($\\Delta U_{\\max} = e E_{0,z} z_0$, Eq. 77)")
    ax1.set_xlabel("Normalized cavity half-length $L / z_0$ [-]")
    ax1.set_ylabel("Normalized gain $\\Delta U_e(L) / (e E_{0,z} z_0)$ [-]")
    ax1.set_title("(b) Two-screen laser cavity gain vs $L / z_0$ (Eq. 76)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.1)

    # (c) Unbounded radial mode (no screens, L -> inf): zero net gain for every carrier phase phi_0 (Sec. 5.5)
    ax2.plot(
        d["phi0_scan_deg"],
        d["unbounded_phase_scan_norm"],
        "o-",
        color="#009E73",
        lw=2.0,
        ms=4.5,
        label="Unbounded path $(-\\infty, +\\infty)$ (Sec. 5.5)",
    )
    ax2.axhline(1.0, color="#0072B2", ls="--", lw=1.7, label="Matched cavity $L = z_0$ peak ($\\Delta U / e E_{0,z} z_0 = 1$)")
    ax2.set_ylim(-0.08, 1.15)
    ax2.set_xlabel("Electron entrance phase $\\phi_0$ [deg]")
    ax2.set_ylabel("Normalized gain $\\Delta U_e / (e E_{0,z} z_0)$ [-]")
    ax2.set_title("(c) No-screen radial mode vanishes for all $\\phi_0$ (Sec. 5.5)", fontsize=10.5, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center right", fontsize=8.3)

    fig.tight_layout()
    out_path = FIG_DIR / "23_part_c_laser_cavity_length_scan.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_figure_24_part_c_aperture() -> Path:
    """Render Figure 24: Part C.3 aperture cutoff universal collapse vs a / (gamma * lambda_bar) (Eq. 84)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_part_c_cavity_aperture.npz")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 4.7))

    xi_scan = d["a_over_gam_lambar_scan"]
    # (a) Gain vs raw aperture radius a / lambda_bar showing separate curves for gamma = 3 and gamma = 8
    ax0.plot(3.0 * d["xi_dense"], d["ap_universal_dense"], "-", color="#0072B2", lw=1.9, label="Theory $\\gamma = 3$: $1 / I_0(a / (\\beta\\gamma\\bar{\\lambda}))$")
    ax0.plot(3.0 * xi_scan, d["ap_fit_gamma3"], "o", color="#0072B2", ms=6.5, label="3D FIT cavity ($\\gamma = 3, L = z_0$)")
    ax0.plot(8.0 * d["xi_dense"], d["ap_universal_dense"], "--", color="#D55E00", lw=1.9, label="Theory $\\gamma = 8$: $1 / I_0(a / (\\beta\\gamma\\bar{\\lambda}))$")
    ax0.plot(8.0 * xi_scan, d["ap_fit_gamma8"], "s", color="#D55E00", ms=6.2, label="3D FIT cavity ($\\gamma = 8, L = z_0$)")
    ax0.axvline(3.0, color="#0072B2", ls=":", lw=1.3, alpha=0.8)
    ax0.axvline(8.0, color="#D55E00", ls=":", lw=1.3, alpha=0.8)
    ax0.set_xlabel("Unscaled aperture radius $a / \\bar{\\lambda} = k a$ [-]")
    ax0.set_ylabel("Normalized on-axis gain $\\Delta U_e(a) / \\Delta U_e(0)$ [-]")
    ax0.set_title("(a) Unscaled aperture dependence for $\\gamma = 3$ and $\\gamma = 8$", fontsize=10.5, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8.4)

    # (b) Universal collapse when plotted against scaled variable a / (gamma * lambda_bar) (Eq. 84)
    ax1.plot(
        d["xi_dense"],
        d["ap_universal_dense"],
        "k-",
        lw=2.2,
        label="Universal Helmholtz law $1 / I_0(a / (\\beta\\gamma\\bar{\\lambda}))$",
    )
    ax1.plot(
        xi_scan,
        d["ap_fit_gamma3"],
        "o",
        color="#0072B2",
        ms=7.0,
        label="3D FIT $\\gamma = 3$ ($\\geq 4$ cells across $a$)",
    )
    ax1.plot(
        xi_scan,
        d["ap_fit_gamma8"],
        "s",
        color="#D55E00",
        ms=6.0,
        fillstyle="none",
        mew=1.8,
        label="3D FIT $\\gamma = 8$ ($\\geq 4$ cells across $a$)",
    )
    ax1.axvline(1.0, color="#009E73", ls="--", lw=1.6, label="Aperture cutoff $a = \\gamma\\bar{\\lambda}$ (Eq. 84)")
    ax1.set_xlabel("Scaled aperture radius $a / (\\gamma\\bar{\\lambda})$ [-]")
    ax1.set_ylabel("Normalized on-axis gain $\\Delta U_e(a) / \\Delta U_e(0)$ [-]")
    ax1.set_title("(b) Universal curve collapse vs $a / (\\gamma\\bar{\\lambda})$ (Eq. 84)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.4)

    fig.tight_layout()
    out_path = FIG_DIR / "24_part_c_aperture_universal_collapse.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_maxwell_figures() -> list[Path]:
    """Render all 5 Maxwellian FIT figures (Figures 20–24) from saved NPZ/JSON data."""
    return [
        render_figure_20_part_b_gouy(),
        render_figure_21_part_b_dipole(),
        render_figure_22_part_c_maxwell(),
        render_figure_23_part_c_cavity(),
        render_figure_24_part_c_aperture(),
    ]


MAXWELL_TASKS: dict[str, tuple[Callable[[bool], dict], str]] = {
    "part_b_gouy": (
        run_task_part_b_gouy,
        "Part B.1: 3D FIT Gouy phase, phase velocity > c (Eq. 33), and group/energy-flow velocity < c (Eqs. 34-35)",
    ),
    "part_b_gouy_dipole": (
        run_task_part_b_weak_dipole,
        "Part B.2: 3D FIT focused laser + point Dipole Near2FarMonitor 90 deg Gouy quadrature & optical theorem",
    ),
    "part_c_maxwell": (
        run_task_part_c_maxwell_screen,
        "Part C.1: 3-run linear 3D FIT Maxwellian cross-energy check (-U_int = Delta U_e) & far-field cone",
    ),
    "part_c_cavity_aperture": (
        run_task_part_c_cavity_aperture,
        "Part C.2-C.3: 3D FIT dual-screen radial-mode cavity scan (Eq. 76) & aperture universal collapse (Eq. 84)",
    ),
}
