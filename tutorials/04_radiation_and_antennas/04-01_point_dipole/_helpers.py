# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Simulation, analytical, and plotting helpers for Tutorial 4.1 (Point Dipole in 3D Free Space)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0
from scipy.integrate import quad

from wakis import DFTMonitor, GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.estimate import estimate
from wakis.near2far import Near2FarMonitor, half_power_beamwidth
from wakis.sources import ContinuousSource, CurrentSource, GaussianSource
from wakis.templates import Box
from wakis.templates.meshing import graded_axis

ETA0 = float(np.sqrt(MU0 / EPS0))
F0_HZ = 1.0e9
LAM0_M = float(C0 / F0_HZ)
K0_RAD_M = float(2.0 * np.pi / LAM0_M)
R_KR1_M = float(1.0 / K0_RAD_M)  # lam0 / (2*pi) ~= 47.71 mm
I0_AMP = 1.0
L_REF_M = 0.010  # 10 mm reference element (l / lam0 ~= 1 / 30)
I0_L_AM = I0_AMP * L_REF_M  # 10 mA*m dipole current moment

ALL_TASKS = (
    "radial_zones_and_near2far",
    "dipole_length_and_cell_size",
    "cpml_distance_study",
    "resolution_and_mesh_study",
)

COLORS = {
    "blue": figstyle.OKABE_ITO[0],
    "red": figstyle.OKABE_ITO[1],
    "teal": figstyle.OKABE_ITO[2],
    "orange": figstyle.OKABE_ITO[3],
    "purple": figstyle.OKABE_ITO[4],
    "sky": figstyle.OKABE_ITO[5],
    "green": figstyle.OKABE_ITO[2],
    "grey": "#6B7280",
}


# ---------------------------------------------------------------------------
# Exact analytical 3D Hertzian & finite-length dipole formulas
# ---------------------------------------------------------------------------
def hertzian_fields_spherical(
    r_m: np.ndarray,
    theta_rad: np.ndarray,
    freq_hz: float | np.ndarray,
    il_phasor: complex | np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate exact 3D Hertzian dipole fields in the exp(+j*w*t) convention.

    Governing expressions (Hertz 1889; Balanis, *Antenna Theory*, 4th ed.,
    Sec. 4.2, Eqs. 4-8a to 4-10c):

    .. math::

        E_r(r, \\theta) &= \\frac{\\eta_0 (I\\ell) k^2}{2\\pi}\\cos\\theta
            \\left[\\frac{1}{(kr)^2} - \\frac{j}{(kr)^3}\\right] e^{-jkr}, \\\\
        E_\\theta(r, \\theta) &= \\frac{\\eta_0 (I\\ell) k^2}{4\\pi}\\sin\\theta
            \\left[\\frac{j}{kr} + \\frac{1}{(kr)^2} - \\frac{j}{(kr)^3}\\right] e^{-jkr}, \\\\
        H_\\phi(r, \\theta) &= \\frac{(I\\ell) k^2}{4\\pi}\\sin\\theta
            \\left[\\frac{j}{kr} + \\frac{1}{(kr)^2}\\right] e^{-jkr}.

    Parameters
    ----------
    r_m : np.ndarray
        Radial distance from the dipole origin [m]. Must be positive.
    theta_rad : np.ndarray
        Polar angle measured from the dipole axis (+z) [rad].
    freq_hz : float or np.ndarray
        Frequency [Hz].
    il_phasor : complex or np.ndarray
        Complex current moment phasor :math:`I\\ell(f)` [A m] or [A m s].

    Returns
    -------
    dict of str to np.ndarray
        Complex arrays ``Er``, ``Etheta``, ``Hphi``, and the three individual
        equatorial magnitude terms ``Etheta_static``, ``Etheta_induction``,
        and ``Etheta_radiation``.
    """
    r = np.asarray(r_m, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    k = 2.0 * np.pi * np.asarray(freq_hz, dtype=float) / C0
    kr = k * r
    phase = np.exp(-1j * kr)
    pref = ETA0 * il_phasor * (k**2) / (4.0 * np.pi)

    er = 2.0 * pref * np.cos(theta) * (1.0 / (kr**2) - 1j / (kr**3)) * phase
    etheta = pref * np.sin(theta) * (1j / kr + 1.0 / (kr**2) - 1j / (kr**3)) * phase
    hphi = (il_phasor * (k**2) / (4.0 * np.pi)) * np.sin(theta) * (1j / kr + 1.0 / (kr**2)) * phase

    mag_pref = np.abs(pref * np.sin(theta))
    return {
        "kr": kr,
        "Er": er,
        "Etheta": etheta,
        "Hphi": hphi,
        "Etheta_static": mag_pref / (kr**3),
        "Etheta_induction": mag_pref / (kr**2),
        "Etheta_radiation": mag_pref / kr,
    }


def hertzian_radiated_flux(freq_hz: float | np.ndarray, il_phasor: complex | np.ndarray) -> np.ndarray:
    """Return closed-surface Poynting flux :math:`\\Phi(f) = \\frac{\\eta_0 k^2}{6\\pi}|I\\ell(f)|^2`.

    Matches the Fourier normalization of :meth:`FluxMonitor.get_flux` and
    :meth:`Near2FarMonitor.radiated_power`, which equal twice the time-averaged
    sinusoidal power multiplied by :math:`|S(f)|^2`.

    Parameters
    ----------
    freq_hz : float or np.ndarray
        Frequency [Hz].
    il_phasor : complex or np.ndarray
        Complex current moment phasor :math:`I_0 \\ell S(f)` [A m s].

    Returns
    -------
    np.ndarray
        Radiated spectral flux [W s^2].
    """
    k = 2.0 * np.pi * np.asarray(freq_hz, dtype=float) / C0
    return ETA0 * (k**2) * (np.abs(il_phasor) ** 2) / (6.0 * np.pi)


def hertzian_radiation_resistance(l_over_lam: float | np.ndarray) -> np.ndarray:
    """Return classic Hertzian radiation resistance :math:`R_{\\mathrm{rad}} = \\frac{2\\pi}{3}\\eta_0(\\ell/\\lambda)^2 \\approx 80\\pi^2(\\ell/\\lambda)^2`.

    Parameters
    ----------
    l_over_lam : float or np.ndarray
        Electrical length :math:`\\ell / \\lambda_0` [dimensionless].

    Returns
    -------
    np.ndarray
        Radiation resistance [Ohm].
    """
    x = np.asarray(l_over_lam, dtype=float)
    return (2.0 * np.pi / 3.0) * ETA0 * (x**2)


def uniform_filament_radiation_metrics(l_over_lam: float | np.ndarray) -> dict[str, np.ndarray]:
    """Compute exact radiation resistance and directivity of a uniform-current filament of length :math:`\\ell`.

    For a z-directed line source carrying uniform current :math:`I(z) = I_0` on
    :math:`|z| \\le \\ell/2`, the far-field pattern factor is
    :math:`F(\\theta) = \\sin\\theta \\, \\mathrm{sinc}(\\pi (\\ell/\\lambda)\\cos\\theta)`.

    Parameters
    ----------
    l_over_lam : float or np.ndarray
        Electrical length :math:`\\ell / \\lambda_0` [dimensionless].

    Returns
    -------
    dict of str to np.ndarray
        ``r_rad_ohm`` (exact radiation resistance [Ohm]), ``d0_linear`` (peak
        directivity at :math:`\\theta = \\pi/2` [dimensionless]), and ``d0_dbi``
        [dBi].
    """
    vals = np.atleast_1d(np.asarray(l_over_lam, dtype=float))
    r_out = np.zeros_like(vals)
    d_out = np.zeros_like(vals)
    for idx, x in enumerate(vals):
        half_kl = np.pi * x

        def integrand(theta: float) -> float:
            u = half_kl * np.cos(theta)
            sinc_u = 1.0 if abs(u) < 1e-12 else np.sin(u) / u
            return float((sinc_u**2) * (np.sin(theta) ** 3))

        integral, _ = quad(integrand, 0.0, np.pi, limit=100)
        # At theta=pi/2, sinc(0)=1, and for k*l -> 0, integral -> 4/3:
        # R_rad = (eta0 / 8*pi) * (k*l)^2 * integral -> (eta0 / 6*pi) * (k*l)^2
        r_out[idx] = (ETA0 / (8.0 * np.pi)) * ((2.0 * half_kl) ** 2) * integral
        d_out[idx] = 2.0 / integral

    return {
        "r_rad_ohm": r_out,
        "d0_linear": d_out,
        "d0_dbi": 10.0 * np.log10(d_out),
    }


# ---------------------------------------------------------------------------
# Grid construction & preflight helpers
# ---------------------------------------------------------------------------
def build_point_dipole_grid(
    *,
    graded: bool = True,
    dx_fine_m: float = 0.005,
    dx_coarse_m: float = 0.0145,
    inner_half_m: float = 0.020,
    domain_half_m: float = 0.30,
    grading: float = 1.24,
) -> GridFIT3D:
    """Construct a symmetric 3D FIT grid aligned with a z-directed dipole at the origin.

    In FIT staggering, :math:`E_z` and :math:`J_z` reside at primary nodes
    :math:`(x_i, y_j)` transversally and at primary edge midpoints
    :math:`z_{k+1/2}` longitudinally. This builder places an exact primary node
    at :math:`x = 0` and :math:`y = 0`, and an exact cell midpoint at
    :math:`z = 0` (:math:`z \\in [-\\Delta z_0/2, +\\Delta z_0/2]`).

    Parameters
    ----------
    graded : bool, optional
        If True, build a locally refined graded mesh around the origin. If
        False, build a uniform mesh with cell size ``dx_coarse_m``.
    dx_fine_m : float, optional
        Finest cell size around the dipole origin [m].
    dx_coarse_m : float, optional
        Outer radiation-zone cell size [m].
    inner_half_m : float, optional
        Half-width of the uniformly fine core region [m].
    domain_half_m : float, optional
        Half-width of the full computational box [m].
    grading : float, optional
        Maximum cell-to-cell growth ratio.

    Returns
    -------
    GridFIT3D
        Initialized 3D FIT grid object.
    """
    if graded:
        x_pos = graded_axis([0.0, inner_half_m, domain_half_m], [dx_fine_m, dx_coarse_m], grading=grading)
        x_nodes = np.concatenate([-x_pos[:0:-1], x_pos])
        y_nodes = x_nodes.copy()
        z_pos = graded_axis(
            [0.5 * dx_fine_m, max(inner_half_m, 1.5 * dx_fine_m), domain_half_m],
            [dx_fine_m, dx_coarse_m],
            grading=grading,
        )
        z_nodes = np.concatenate([-z_pos[::-1], z_pos])
    else:
        n_half_xy = max(6, int( np.round(domain_half_m / dx_coarse_m)))
        dx_actual = domain_half_m / n_half_xy
        x_pos = np.linspace(0.0, domain_half_m, n_half_xy + 1)
        x_nodes = np.concatenate([-x_pos[:0:-1], x_pos])
        y_nodes = x_nodes.copy()
        z_pos = (np.arange(n_half_xy) + 0.5) * dx_actual
        z_nodes = np.concatenate([-z_pos[::-1], z_pos])

    return GridFIT3D(
        float(x_nodes[0]),
        float(x_nodes[-1]),
        float(y_nodes[0]),
        float(y_nodes[-1]),
        float(z_nodes[0]),
        float(z_nodes[-1]),
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        verbose=0,
    )


def snap_monitor_box(grid: GridFIT3D, half_width_m: float) -> tuple[float, float, float]:
    """Snap a cubic Huygens box size to exact primary grid nodes along x, y, and z.

    Snapping ``size = (2*bx, 2*by, 2*bz)`` to exact grid nodes ensures that the
    six faces of :class:`FluxMonitor` and :class:`Near2FarMonitor` meet along
    exact grid edges and receive the exact 1/2 trapezoidal boundary weight.
    """
    bx = float(grid.x[int(np.argmin(np.abs(grid.x - half_width_m)))])
    by = float(grid.y[int(np.argmin(np.abs(grid.y - half_width_m)))])
    bz = float(grid.z[int(np.argmin(np.abs(grid.z - half_width_m)))])
    return (2.0 * bx, 2.0 * by, 2.0 * bz)


def preflight_point_dipole_configs() -> dict[str, dict[str, Any]]:
    """Run :func:`wakis.estimate.estimate` on graded and uniform mesh configurations.

    Returns
    -------
    dict of str to dict
        Resource and resolution summary for each configuration.
    """
    configs = {
        "graded_full": dict(graded=True, dx_fine_m=0.005, dx_coarse_m=0.0145, inner_half_m=0.020, domain_half_m=0.30),
        "graded_quick": dict(graded=True, dx_fine_m=0.0075, dx_coarse_m=0.018, inner_half_m=0.0225, domain_half_m=0.28),
        "uniform_cpw14": dict(graded=False, dx_fine_m=LAM0_M / 14.0, dx_coarse_m=LAM0_M / 14.0, domain_half_m=0.30),
        "uniform_cpw20": dict(graded=False, dx_fine_m=LAM0_M / 20.0, dx_coarse_m=LAM0_M / 20.0, domain_half_m=0.30),
        "uniform_cpw26": dict(graded=False, dx_fine_m=LAM0_M / 26.0, dx_coarse_m=LAM0_M / 26.0, domain_half_m=0.30),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, kw in configs.items():
        grid = build_point_dipole_grid(**kw)
        dx_min = float(min(np.min(grid.L[:, 0, 0, "x"]), np.min(grid.L[0, 0, :, "z"])))
        dx_max = float(max(np.max(grid.L[:, 0, 0, "x"]), np.max(grid.L[0, 0, :, "z"])))
        dt_est = dx_min / (np.sqrt(3.0) * C0)
        n_steps = int(np.ceil(4.2 / (F0_HZ * dt_est)))
        nx, ny, nz = int(grid.Nx - 1), int(grid.Ny - 1), int(grid.Nz - 1)
        est = estimate(cells=(nx, ny, nz), steps=n_steps, cpml=True)
        out[name] = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "n_cells": int(est.cells),
            "dx_min_mm": dx_min * 1e3,
            "dx_max_mm": dx_max * 1e3,
            "cpw_f0_min": float(LAM0_M / dx_max),
            "cpw_f0_core": float(LAM0_M / dx_min),
            "n_steps": n_steps,
            "cpu_ram_mb": float(est.host_bytes / 1e6),
            "cpu_time_s": float(est.seconds),
        }

    # Extended 3D domain for the far-field radiation movie (r_phys = 3.32*lambda_0, cpw = 20.5)
    cpw_mov = 20.5
    dx_mov = LAM0_M / cpw_mov
    nx_mov, ny_mov, nz_mov = 148, 72, 149
    dt_mov = dx_mov / (np.sqrt(3.0) * C0)
    n_steps_mov = int(np.ceil(4.9 / (F0_HZ * dt_mov)))
    est_mov = estimate(cells=(nx_mov, ny_mov, nz_mov), steps=n_steps_mov, cpml=True)
    out["radiation_movie_3d"] = {
        "nx": nx_mov,
        "ny": ny_mov,
        "nz": nz_mov,
        "n_cells": int(est_mov.cells),
        "dx_min_mm": dx_mov * 1e3,
        "dx_max_mm": dx_mov * 1e3,
        "cpw_f0_min": cpw_mov,
        "cpw_f0_core": cpw_mov,
        "n_steps": n_steps_mov,
        "cpu_ram_mb": float(est_mov.host_bytes / 1e6),
        "cpu_time_s": float(est_mov.seconds),
    }
    return out


# ---------------------------------------------------------------------------
# Simulation task runners
# ---------------------------------------------------------------------------
def run_radiation_movie_cut(quick: bool = False) -> dict[str, np.ndarray]:
    """Run an extended-domain 3D FIT point-dipole simulation for the radiation GIF.

    Uses :math:`\\Delta x = \\Delta y = \\Delta z = \\lambda_0 / 20.5` (:math:`\\ge 20`
    cells/:math:`\\lambda_0`) with a physical (non-CPML) half-width of
    :math:`R_{\\mathrm{phys}} = 3.32\\,\\lambda_0` (:math:`> 3\\,\\lambda_0` beyond
    the :math:`kr = 1` near-field boundary) and records a 2D :math:`xz`-plane cut
    at :math:`y = 0` over 36 frames.

    Parameters
    ----------
    quick : bool, optional
        If True, use a slightly reduced transverse :math:`y` buffer while keeping
        :math:`\\ge 20` cells/:math:`\\lambda_0` and :math:`R_{\\mathrm{phys}} > 3.2\\,\\lambda_0`.

    Returns
    -------
    dict of str to np.ndarray
        Time-domain 2D :math:`xz`-cut frames and coordinates cropped to the
        physical non-CPML region.
    """
    cpw = 20.5
    dx = LAM0_M / cpw
    n_pml = 6
    n_half_xz = 66 if quick else 68  # R_phys = 3.22*lam0 (quick) or 3.32*lam0 (full)
    n_half_y = 24 if quick else 30
    r_phys_m = n_half_xz * dx
    r_total_m = (n_half_xz + n_pml) * dx
    y_total_m = (n_half_y + n_pml) * dx

    x_nodes = np.linspace(-r_total_m, r_total_m, 2 * (n_half_xz + n_pml) + 1)
    y_nodes = np.linspace(-y_total_m, y_total_m, 2 * (n_half_y + n_pml) + 1)
    z_pos = (np.arange(n_half_xz + n_pml) + 0.5) * dx
    z_nodes = np.concatenate([-z_pos[::-1], z_pos])

    grid = GridFIT3D(
        float(x_nodes[0]),
        float(x_nodes[-1]),
        float(y_nodes[0]),
        float(y_nodes[-1]),
        float(z_nodes[0]),
        float(z_nodes[-1]),
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        verbose=0,
    )
    nx, ny, nz = int(grid.Nx - 1), int(grid.Ny - 1), int(grid.Nz - 1)
    solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, verbose=0)
    n_periods = 4.9
    n_steps = int(np.ceil(n_periods / (F0_HZ * solver.dt)))
    _ = estimate(cells=(nx, ny, nz), steps=n_steps, cpml=True)

    cw = ContinuousSource(frequency=F0_HZ, ramp_time=1.0 / F0_HZ)
    src = CurrentSource(
        cw,
        component="z",
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
        amplitude=I0_L_AM,
    )

    step_start = int(1.30 / (F0_HZ * solver.dt))
    step_end = n_steps - 1
    frame_steps = set(np.linspace(step_start, step_end, 36, dtype=int).tolist())

    jy_mid = int(np.argmin(np.abs(grid.y)))
    x_all = np.asarray(grid.x[:-1], dtype=float)
    z_all = np.asarray(0.5 * (grid.z[:-1] + grid.z[1:]), dtype=float)
    ix_phys = np.where(np.abs(x_all) <= r_phys_m + 0.25 * dx)[0]
    iz_phys = np.where(np.abs(z_all) <= r_phys_m + 0.25 * dx)[0]

    td_frames: list[np.ndarray] = []
    td_times_ns: list[float] = []
    td_times_periods: list[float] = []

    for step in range(n_steps):
        t = step * solver.dt
        src.update(solver, t)
        solver.one_step()
        if step in frame_steps:
            t_after = (step + 1) * solver.dt
            ez_full = np.asarray(solver.E.to_matrix("z")[:, jy_mid, :-1], dtype=np.float32)
            ez_crop = ez_full[np.ix_(ix_phys, iz_phys)].copy()
            td_frames.append(ez_crop)
            td_times_ns.append(float(t_after * 1e9))
            td_times_periods.append(float(t_after * F0_HZ))

    return {
        "td_frames": np.asarray(td_frames, dtype=np.float32),
        "td_times_ns": np.asarray(td_times_ns, dtype=float),
        "td_times_periods": np.asarray(td_times_periods, dtype=float),
        "td_x_ez": x_all[ix_phys],
        "td_z_ez": z_all[iz_phys],
        "td_r_phys_m": np.array(float(r_phys_m)),
        "td_r_total_m": np.array(float(r_total_m)),
        "td_cpw": np.array(float(cpw)),
        "td_n_pml": np.array(int(n_pml)),
    }
def run_radial_zones_and_near2far(quick: bool = False) -> dict[str, Any]:
    """Run Task 0: 3D Hertzian dipole on graded mesh with radial zones, DFT maps, GIF frames & Near2Far.

    Parameters
    ----------
    quick : bool, optional
        If True, use the faster coarse graded mesh suitable for interactive/CI runs.

    Returns
    -------
    dict
        Simulation results ready to save into ``task_radial_zones_and_near2far.npz``.
    """
    dx_fine = 0.0075 if quick else 0.0050
    dx_coarse = 0.0180 if quick else 0.0145
    inner_half = 0.0225 if quick else 0.0200
    half_L = 0.28 if quick else 0.30
    n_pml = 6

    grid = build_point_dipole_grid(
        graded=True,
        dx_fine_m=dx_fine,
        dx_coarse_m=dx_coarse,
        inner_half_m=inner_half,
        domain_half_m=half_L,
        grading=1.24,
    )
    solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, verbose=0)
    pulse = GaussianSource(F0_HZ, 0.60 * F0_HZ)
    src = CurrentSource(
        pulse,
        component="z",
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
        amplitude=I0_L_AM,
    )

    freqs_band = np.linspace(0.75e9, 1.25e9, 11)
    idx_f0 = int(np.argmin(np.abs(freqs_band - F0_HZ)))
    b_n2f = 0.165 if quick else 0.180
    mon_size = snap_monitor_box(grid, b_n2f)
    b_n2f = 0.5 * mon_size[0]

    n2f_f0 = Near2FarMonitor(frequencies=[F0_HZ], center=(0.0, 0.0, 0.0), size=mon_size)
    n2f_band = Near2FarMonitor(frequencies=freqs_band, center=(0.0, 0.0, 0.0), size=mon_size)
    dft_xz = DFTMonitor(
        frequencies=[F0_HZ],
        components=("Ex", "Ez", "Hy"),
        center=(0.0, 0.0, 0.0),
        size=(2 * half_L, 0.0, 2 * half_L),
    )
    dft_src_band = DFTMonitor(
        frequencies=freqs_band,
        components=("Ez",),
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
    )

    n_steps = int(np.ceil((4.0 if quick else 4.4) / (F0_HZ * solver.dt)))

    t0_wall = time.perf_counter()
    for step in range(n_steps):
        t = step * solver.dt
        src.update(solver, t)
        solver.one_step()
        n2f_f0.update(solver, t)
        n2f_band.update(solver, t)
        dft_xz.update(solver, t)
        dft_src_band.update(solver, t)
    wall_s = float(time.perf_counter() - t0_wall)

    movie_data = run_radiation_movie_cut(quick=quick)

    # Extract xz plane DFT fields at f0
    ex_f, ex_coords = dft_xz.get_field("Ex")
    ez_f, ez_coords = dft_xz.get_field("Ez")
    hy_f, hy_coords = dft_xz.get_field("Hy")
    ex_xz = np.asarray(ex_f[0, :, 0, :], dtype=np.complex128)
    ez_xz = np.asarray(ez_f[0, :, 0, :], dtype=np.complex128)
    hy_xz = np.asarray(hy_f[0, :, 0, :], dtype=np.complex128)
    x_ez = np.asarray(ez_coords[0], dtype=float)
    z_ez = np.asarray(ez_coords[2], dtype=float)
    x_hy = np.asarray(hy_coords[0], dtype=float)

    ix0 = int(np.argmin(np.abs(x_ez)))
    iz0 = int(np.argmin(np.abs(z_ez)))

    # Source spectrum and power balance across band
    sf_f0 = complex(pulse.fourier_transform(np.array([F0_HZ]))[0])
    il_f0 = I0_L_AM * sf_f0
    phi_ana_f0 = float(hertzian_radiated_flux(F0_HZ, il_f0))
    phi_box_f0 = float(n2f_f0.get_flux()[0])
    phi_n2f_f0 = float(n2f_f0.radiated_power()[0])
    w_src_f0 = float(-np.real(ez_xz[ix0, iz0] * np.conj(il_f0)))

    sf_band = pulse.fourier_transform(freqs_band)
    il_band = I0_L_AM * sf_band
    phi_ana_band = np.asarray(hertzian_radiated_flux(freqs_band, il_band), dtype=float)
    phi_box_band = np.asarray(n2f_band.get_flux(), dtype=float)
    phi_n2f_band = np.asarray(n2f_band.radiated_power(), dtype=float)
    ez_src_band, _ = dft_src_band.get_field("Ez")
    w_src_band = np.asarray(-np.real(ez_src_band[:, 0, 0, 0] * np.conj(il_band)), dtype=float)

    # Radial profiles:
    # 1) Equatorial along +x at z = 0 (theta = pi/2): Ez = -E_theta, Hy = H_phi
    pml_margin = n_pml + 2
    r_eq_ez = x_ez[ix0 + 1 : -pml_margin]
    etheta_eq_fit = -ez_xz[ix0 + 1 : -pml_margin, iz0]
    # Interpolate Hy (which sits at x midpoints x_hy) onto r_eq_ez in log-amplitude and unwrapped phase
    r_eq_hy = x_hy[ix0 : -pml_margin]
    hphi_eq_raw = hy_xz[ix0 : -pml_margin, iz0]
    amp_hy_interp = np.exp(np.interp(r_eq_ez, r_eq_hy, np.log(np.maximum(np.abs(hphi_eq_raw), 1e-30))))
    ang_hy_interp = np.interp(r_eq_ez, r_eq_hy, np.unwrap(np.angle(hphi_eq_raw)))
    hphi_eq_fit = amp_hy_interp * np.exp(1j * ang_hy_interp)

    ana_eq = hertzian_fields_spherical(r_eq_ez, np.full_like(r_eq_ez, 0.5 * np.pi), F0_HZ, il_f0)
    zw_eq_fit = etheta_eq_fit / hphi_eq_fit
    zw_eq_ana = ana_eq["Etheta"] / ana_eq["Hphi"]
    phase_diff_fit_deg = np.rad2deg(np.angle(etheta_eq_fit / hphi_eq_fit))
    phase_diff_ana_deg = np.rad2deg(np.angle(ana_eq["Etheta"] / ana_eq["Hphi"]))

    # 2) Axial along +z at x = 0 (theta = 0): Ez = E_r
    r_ax_ez = z_ez[iz0 + 1 : -pml_margin]
    er_ax_fit = ez_xz[ix0, iz0 + 1 : -pml_margin]
    ana_ax = hertzian_fields_spherical(r_ax_ez, np.zeros_like(r_ax_ez), F0_HZ, il_f0)

    # Exclude the immediate self-neighbor cell (index 0, r = dx_fine) for continuum error metric
    rel_err_eq = np.abs(np.abs(etheta_eq_fit[1:]) - np.abs(ana_eq["Etheta"][1:])) / np.abs(ana_eq["Etheta"][1:])
    rel_err_ax = np.abs(np.abs(er_ax_fit[1:]) - np.abs(ana_ax["Er"][1:])) / np.abs(ana_ax["Er"][1:])

    # Far-field E-plane (phi = 0, theta in [0, pi]) and H-plane (theta = pi/2, phi in [0, 2pi])
    theta_rad = np.linspace(1e-3, np.pi - 1e-3, 181)
    phi_zero = np.zeros_like(theta_rad)
    d_eplane = np.asarray(n2f_f0.directivity(theta_rad, phi_zero)[0], dtype=float)
    d_eplane_ana = 1.5 * (np.sin(theta_rad) ** 2)
    hpbw_deg = float(np.rad2deg(half_power_beamwidth(theta_rad, d_eplane)))

    phi_rad = np.linspace(0.0, 2.0 * np.pi, 181)
    theta_eq = np.full_like(phi_rad, 0.5 * np.pi)
    d_hplane = np.asarray(n2f_f0.directivity(theta_eq, phi_rad)[0], dtype=float)

    # Interpolate Ex (at x_mid, z_nodes) onto Ez locations (x_nodes, z_mid)
    ex_on_ez = np.zeros_like(ez_xz)
    ex_on_ez[1:-1, :] = 0.25 * (
        ex_xz[:-1, :-1] + ex_xz[1:, :-1] + ex_xz[:-1, 1:] + ex_xz[1:, 1:]
    )
    emag_xz = np.sqrt(np.abs(ex_on_ez) ** 2 + np.abs(ez_xz) ** 2)

    return {
        "quick": np.array(int(quick)),
        "wall_s": np.array(wall_s),
        "n_cells": np.array((grid.Nx - 1) * (grid.Ny - 1) * (grid.Nz - 1)),
        "dx_fine_m": np.array(dx_fine),
        "dx_coarse_m": np.array(dx_coarse),
        "half_L_m": np.array(half_L),
        "b_n2f_m": np.array(b_n2f),
        "n_pml": np.array(n_pml),
        "x_nodes": np.asarray(grid.x, dtype=float),
        "z_nodes": np.asarray(grid.z, dtype=float),
        "x_ez": x_ez,
        "z_ez": z_ez,
        "ez_xz_real": np.real(ez_xz),
        "ez_xz_abs": np.abs(ez_xz),
        "emag_xz": emag_xz,
        **movie_data,
        "r_eq_m": r_eq_ez,
        "kr_eq": ana_eq["kr"],
        "etheta_eq_fit_abs": np.abs(etheta_eq_fit),
        "etheta_eq_ana_abs": np.abs(ana_eq["Etheta"]),
        "hphi_eq_fit_scaled": ETA0 * np.abs(hphi_eq_fit),
        "hphi_eq_ana_scaled": ETA0 * np.abs(ana_eq["Hphi"]),
        "etheta_static": ana_eq["Etheta_static"],
        "etheta_induction": ana_eq["Etheta_induction"],
        "etheta_radiation": ana_eq["Etheta_radiation"],
        "zw_eq_fit_norm": np.abs(zw_eq_fit) / ETA0,
        "zw_eq_ana_norm": np.abs(zw_eq_ana) / ETA0,
        "phase_diff_fit_deg": phase_diff_fit_deg,
        "phase_diff_ana_deg": phase_diff_ana_deg,
        "r_ax_m": r_ax_ez,
        "kr_ax": ana_ax["kr"],
        "er_ax_fit_abs": np.abs(er_ax_fit),
        "er_ax_ana_abs": np.abs(ana_ax["Er"]),
        "median_rel_err_eq": np.array(float(np.median(rel_err_eq))),
        "median_rel_err_ax": np.array(float(np.median(rel_err_ax))),
        "theta_deg": np.rad2deg(theta_rad),
        "d_eplane": d_eplane,
        "d_eplane_ana": d_eplane_ana,
        "phi_deg": np.rad2deg(phi_rad),
        "d_hplane": d_hplane,
        "d0_peak": np.array(float(np.max(d_eplane))),
        "d0_peak_dbi": np.array(float(10.0 * np.log10(np.max(d_eplane)))),
        "hpbw_deg": np.array(hpbw_deg),
        "pattern_l2_err": np.array(float(np.sqrt(np.mean((d_eplane - d_eplane_ana) ** 2)))),
        "phi_ana_f0": np.array(phi_ana_f0),
        "phi_box_f0": np.array(phi_box_f0),
        "phi_n2f_f0": np.array(phi_n2f_f0),
        "w_src_f0": np.array(w_src_f0),
        "freqs_band_ghz": freqs_band / 1e9,
        "phi_ana_band": phi_ana_band,
        "phi_box_band": phi_box_band,
        "phi_n2f_band": phi_n2f_band,
        "w_src_band": w_src_band,
    }


def run_dipole_length_and_cell_size(quick: bool = False) -> dict[str, Any]:
    """Run Task 1: Finite cell size regularization and sweep of dipole length :math:`\\ell / \\lambda_0`.

    Parameters
    ----------
    quick : bool, optional
        If True, use a faster grid for interactive/CI runs.

    Returns
    -------
    dict
        Results for ``task_dipole_length_and_cell_size.npz``.
    """
    l_over_lam_list = np.array([0.04, 0.08, 0.16, 0.28, 0.40, 0.50], dtype=float)
    dz_core = 0.006 if not quick else 0.0075
    dx_coarse = 0.015 if not quick else 0.018
    half_L = 0.28
    grid = build_point_dipole_grid(
        graded=True,
        dx_fine_m=dz_core,
        dx_coarse_m=dx_coarse,
        inner_half_m=0.078,
        domain_half_m=half_L,
        grading=1.22,
    )
    b_n2f = 0.165
    theta_rad = np.linspace(1e-3, np.pi - 1e-3, 181)
    phi_zero = np.zeros_like(theta_rad)

    r_rad_fit = np.zeros_like(l_over_lam_list)
    d0_fit = np.zeros_like(l_over_lam_list)
    hpbw_fit = np.zeros_like(l_over_lam_list)
    l_actual_over_lam = np.zeros_like(l_over_lam_list)

    ix0 = int(np.argmin(np.abs(grid.x)))
    iy0 = int(np.argmin(np.abs(grid.y)))
    iz0 = int(np.argmin(np.abs(grid.z[:-1] + 0.5 * np.diff(grid.z))))
    dz_actual = float(grid.L[ix0, iy0, iz0, "z"])

    for idx, ratio in enumerate(l_over_lam_list):
        target_l = float(ratio * LAM0_M)
        # Snap to an odd number of core cells so the filament is symmetric about z = 0
        n_cells_l = max(1, int(np.round(target_l / dz_actual)))
        if n_cells_l % 2 == 0:
            n_cells_l += 1
        size_z = 0.0 if n_cells_l == 1 else (n_cells_l - 0.5) * dz_actual
        indices, _ = grid.region_indices("J", "z", (0.0, 0.0, 0.0), (0.0, 0.0, size_z))
        actual_l = float(np.sum(grid.L[ix0, iy0, indices[2], "z"]))
        l_actual_over_lam[idx] = actual_l / LAM0_M

        solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
        pulse = GaussianSource(F0_HZ, 0.60 * F0_HZ)
        amp = I0_AMP * actual_l if n_cells_l == 1 else I0_AMP
        src = CurrentSource(
            pulse,
            component="z",
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, size_z),
            amplitude=amp,
        )
        n2f = Near2FarMonitor(frequencies=[F0_HZ], center=(0.0, 0.0, 0.0), size=snap_monitor_box(grid, b_n2f))
        n_steps = int(np.ceil((3.8 if quick else 4.2) / (F0_HZ * solver.dt)))
        solver.emsolve(n_steps, source=src, monitors=[n2f], progress_bar=False)

        sf = complex(pulse.fourier_transform(np.array([F0_HZ]))[0])
        phi_rad = float(n2f.radiated_power()[0])
        # Since Phi_rad(f0) = R_rad * |I0 * S(f0)|^2, we extract R_rad directly:
        r_rad_fit[idx] = phi_rad / (abs(I0_AMP * sf) ** 2)
        d_pat = np.asarray(n2f.directivity(theta_rad, phi_zero)[0], dtype=float)
        d0_fit[idx] = float(np.max(d_pat))
        hpbw_fit[idx] = float(np.rad2deg(half_power_beamwidth(theta_rad, d_pat)))

    # Also compare near-origin self-cell regularization for two core cell sizes (4 mm vs 10 mm)
    r_profiles = {}
    for label, d_cell in [("fine_4mm", 0.004 if not quick else 0.005), ("coarse_10mm", 0.010)]:
        g_cell = build_point_dipole_grid(
            graded=True,
            dx_fine_m=d_cell,
            dx_coarse_m=dx_coarse,
            inner_half_m=0.040,
            domain_half_m=0.26,
            grading=1.24,
        )
        s_cell = SolverFIT3D(g_cell, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
        pulse_c = GaussianSource(F0_HZ, 0.60 * F0_HZ)
        src_c = CurrentSource(pulse_c, component="z", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0), amplitude=I0_L_AM)
        dft_c = DFTMonitor(frequencies=[F0_HZ], components=("Ez",), center=(0.0, 0.0, 0.0), size=(0.16, 0.0, 0.0))
        n_steps_c = int(np.ceil(3.8 / (F0_HZ * s_cell.dt)))
        s_cell.emsolve(n_steps_c, source=src_c, monitors=[dft_c], progress_bar=False)
        ez_c, coords_c = dft_c.get_field("Ez")
        x_c = np.asarray(coords_c[0], dtype=float)
        ix_c = int(np.argmin(np.abs(x_c)))
        sf_c = complex(pulse_c.fourier_transform(np.array([F0_HZ]))[0])
        r_profiles[f"r_{label}_m"] = x_c[ix_c:]
        r_profiles[f"ez_{label}_norm"] = np.abs(ez_c[0, ix_c:, 0, 0]) / abs(sf_c)

    l_dense = np.linspace(0.02, 0.55, 80)
    unif_dense = uniform_filament_radiation_metrics(l_dense)
    unif_at_pts = uniform_filament_radiation_metrics(l_actual_over_lam)
    r_hertz_pts = hertzian_radiation_resistance(l_actual_over_lam)

    return {
        "l_over_lam": l_actual_over_lam,
        "r_rad_fit_ohm": r_rad_fit,
        "r_rad_hertz_ohm": r_hertz_pts,
        "r_rad_unif_ohm": unif_at_pts["r_rad_ohm"],
        "d0_fit": d0_fit,
        "d0_unif": unif_at_pts["d0_linear"],
        "hpbw_fit_deg": hpbw_fit,
        "l_dense": l_dense,
        "r_hertz_dense_ohm": hertzian_radiation_resistance(l_dense),
        "r_unif_dense_ohm": unif_dense["r_rad_ohm"],
        "r_tri_dense_ohm": 0.25 * hertzian_radiation_resistance(l_dense),
        "d0_unif_dense": unif_dense["d0_linear"],
        **r_profiles,
    }


def run_cpml_distance_study(quick: bool = False) -> dict[str, Any]:
    """Run Task 2: Sweep of source-to-CPML clearance :math:`d_{\\mathrm{PML}} / \\lambda_0`.

    Parameters
    ----------
    quick : bool, optional
        If True, use a slightly coarser mesh for speed.

    Returns
    -------
    dict
        Results for ``task_cpml_distance_study.npz``.
    """
    d_over_lam_list = np.array([0.045, 0.080, 0.140, 0.250, 0.420], dtype=float)
    dx_fine = 0.0075 if quick else 0.0060
    dx_coarse = 0.0150 if quick else 0.0125
    n_pml = 6
    theta_rad = np.linspace(1e-3, np.pi - 1e-3, 181)
    phi_zero = np.zeros_like(theta_rad)
    d_eplane_ana = 1.5 * (np.sin(theta_rad) ** 2)

    d_actual_over_lam = np.zeros_like(d_over_lam_list)
    kd_actual = np.zeros_like(d_over_lam_list)
    rel_err_box = np.zeros_like(d_over_lam_list)
    rel_err_n2f = np.zeros_like(d_over_lam_list)
    rel_err_wsrc = np.zeros_like(d_over_lam_list)
    pattern_l2_err = np.zeros_like(d_over_lam_list)

    for idx, d_ratio in enumerate(d_over_lam_list):
        d_clear_m = float(d_ratio * LAM0_M)
        dx_outer = dx_fine if d_ratio < 0.12 else dx_coarse
        half_L = d_clear_m + n_pml * dx_outer
        grid = build_point_dipole_grid(
            graded=(d_ratio >= 0.12),
            dx_fine_m=dx_fine,
            dx_coarse_m=dx_outer,
            inner_half_m=min(0.018, 0.4 * d_clear_m),
            domain_half_m=half_L,
            grading=1.22,
        )
        # Actual clearance from origin to the inner CPML interface node
        d_pml_inner = float(grid.x[-1 - n_pml])
        d_actual_over_lam[idx] = d_pml_inner / LAM0_M
        kd_actual[idx] = K0_RAD_M * d_pml_inner

        solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, verbose=0)
        pulse = GaussianSource(F0_HZ, 0.60 * F0_HZ)
        src = CurrentSource(pulse, component="z", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0), amplitude=I0_L_AM)

        b_mon = max(1.5 * dx_fine, d_pml_inner - 1.5 * dx_coarse)
        n2f = Near2FarMonitor(frequencies=[F0_HZ], center=(0.0, 0.0, 0.0), size=snap_monitor_box(grid, b_mon))
        dft_src = DFTMonitor(frequencies=[F0_HZ], components=("Ez",), center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))

        n_steps = int(np.ceil((3.8 if quick else 4.2) / (F0_HZ * solver.dt)))
        solver.emsolve(n_steps, source=src, monitors=[n2f, dft_src], progress_bar=False)

        sf = complex(pulse.fourier_transform(np.array([F0_HZ]))[0])
        il_f = I0_L_AM * sf
        phi_ana = float(hertzian_radiated_flux(F0_HZ, il_f))
        phi_box = float(n2f.get_flux()[0])
        phi_n2f = float(n2f.radiated_power()[0])
        ez_src, _ = dft_src.get_field("Ez")
        w_src = float(-np.real(ez_src[0, 0, 0, 0] * np.conj(il_f)))

        rel_err_box[idx] = abs(phi_box - phi_ana) / phi_ana
        rel_err_n2f[idx] = abs(phi_n2f - phi_ana) / phi_ana
        rel_err_wsrc[idx] = abs(w_src - phi_ana) / phi_ana

        d_pat = np.asarray(n2f.directivity(theta_rad, phi_zero)[0], dtype=float)
        pattern_l2_err[idx] = float(np.sqrt(np.mean((d_pat - d_eplane_ana) ** 2)))

    return {
        "d_over_lam": d_actual_over_lam,
        "kd_pml": kd_actual,
        "rel_err_box": rel_err_box,
        "rel_err_n2f": rel_err_n2f,
        "rel_err_wsrc": rel_err_wsrc,
        "pattern_l2_err": pattern_l2_err,
    }


def run_resolution_and_mesh_study(quick: bool = False) -> dict[str, Any]:
    """Run Task 3: Uniform resolution sweep (:math:`\\mathrm{cpw} \\in \\{10, 14, 20, 26\\}`) vs. graded mesh.

    Parameters
    ----------
    quick : bool, optional
        If True, cap highest uniform resolution at ``cpw = 22`` for fast execution.

    Returns
    -------
    dict
        Results for ``task_resolution_and_mesh_study.npz``.
    """
    cpw_list = np.array([10.0, 14.0, 18.0, 22.0] if quick else [10.0, 14.0, 20.0, 26.0], dtype=float)
    half_L = 0.28
    b_n2f = 0.16
    n_pml = 5
    theta_rad = np.linspace(1e-3, np.pi - 1e-3, 181)
    phi_zero = np.zeros_like(theta_rad)
    d_eplane_ana = 1.5 * (np.sin(theta_rad) ** 2)

    n_cells_unif = np.zeros_like(cpw_list)
    wall_unif_s = np.zeros_like(cpw_list)
    power_err_unif = np.zeros_like(cpw_list)
    near_err_unif = np.zeros_like(cpw_list)
    pat_err_unif = np.zeros_like(cpw_list)

    for idx, cpw in enumerate(cpw_list):
        dx = LAM0_M / float(cpw)
        grid = build_point_dipole_grid(graded=False, dx_fine_m=dx, dx_coarse_m=dx, domain_half_m=half_L)
        n_cells_unif[idx] = (grid.Nx - 1) * (grid.Ny - 1) * (grid.Nz - 1)

        t0 = time.perf_counter()
        solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, verbose=0)
        pulse = GaussianSource(F0_HZ, 0.60 * F0_HZ)
        src = CurrentSource(pulse, component="z", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0), amplitude=I0_L_AM)
        n2f = Near2FarMonitor(frequencies=[F0_HZ], center=(0.0, 0.0, 0.0), size=snap_monitor_box(grid, b_n2f))
        dft_eq = DFTMonitor(frequencies=[F0_HZ], components=("Ez",), center=(0.0, 0.0, 0.0), size=(2 * half_L, 0.0, 0.0))

        n_steps = int(np.ceil(4.0 / (F0_HZ * solver.dt)))
        solver.emsolve(n_steps, source=src, monitors=[n2f, dft_eq], progress_bar=False)
        wall_unif_s[idx] = time.perf_counter() - t0

        sf = complex(pulse.fourier_transform(np.array([F0_HZ]))[0])
        il_f = I0_L_AM * sf
        phi_ana = float(hertzian_radiated_flux(F0_HZ, il_f))
        phi_box = float(n2f.get_flux()[0])
        power_err_unif[idx] = abs(phi_box - phi_ana) / phi_ana

        ez_f, coords_f = dft_eq.get_field("Ez")
        x_arr = np.asarray(coords_f[0], dtype=float)
        ez_arr = np.asarray(ez_f[0, :, 0, 0], dtype=np.complex128)
        mask = (x_arr >= 0.060) & (x_arr <= 0.165)
        ana_eq = hertzian_fields_spherical(x_arr[mask], np.full(np.sum(mask), 0.5 * np.pi), F0_HZ, il_f)
        near_err_unif[idx] = float(
            np.sqrt(np.mean((np.abs(ez_arr[mask]) - np.abs(ana_eq["Etheta"])) ** 2))
            / np.sqrt(np.mean(np.abs(ana_eq["Etheta"]) ** 2))
        )

        d_pat = np.asarray(n2f.directivity(theta_rad, phi_zero)[0], dtype=float)
        pat_err_unif[idx] = float(np.sqrt(np.mean((d_pat - d_eplane_ana) ** 2)))

    # Run graded mesh on the exact same domain_half_m and b_n2f for apples-to-apples comparison
    dx_f_grad = 0.006 if quick else 0.005
    dx_c_grad = 0.017 if quick else 0.0145
    g_grad = build_point_dipole_grid(
        graded=True, dx_fine_m=dx_f_grad, dx_coarse_m=dx_c_grad, inner_half_m=0.020, domain_half_m=half_L, grading=1.24
    )
    t0 = time.perf_counter()
    s_grad = SolverFIT3D(g_grad, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, verbose=0)
    pulse = GaussianSource(F0_HZ, 0.60 * F0_HZ)
    src = CurrentSource(pulse, component="z", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0), amplitude=I0_L_AM)
    n2f = Near2FarMonitor(frequencies=[F0_HZ], center=(0.0, 0.0, 0.0), size=snap_monitor_box(g_grad, b_n2f))
    dft_eq = DFTMonitor(frequencies=[F0_HZ], components=("Ez",), center=(0.0, 0.0, 0.0), size=(2 * half_L, 0.0, 0.0))
    n_steps = int(np.ceil(4.0 / (F0_HZ * s_grad.dt)))
    s_grad.emsolve(n_steps, source=src, monitors=[n2f, dft_eq], progress_bar=False)
    wall_grad_s = float(time.perf_counter() - t0)

    sf = complex(pulse.fourier_transform(np.array([F0_HZ]))[0])
    il_f = I0_L_AM * sf
    phi_ana = float(hertzian_radiated_flux(F0_HZ, il_f))
    power_err_grad = abs(float(n2f.get_flux()[0]) - phi_ana) / phi_ana
    ez_f, coords_f = dft_eq.get_field("Ez")
    x_arr = np.asarray(coords_f[0], dtype=float)
    ez_arr = np.asarray(ez_f[0, :, 0, 0], dtype=np.complex128)
    mask = (x_arr >= 0.060) & (x_arr <= 0.165)
    ana_eq = hertzian_fields_spherical(x_arr[mask], np.full(np.sum(mask), 0.5 * np.pi), F0_HZ, il_f)
    near_err_grad = float(
        np.sqrt(np.mean((np.abs(ez_arr[mask]) - np.abs(ana_eq["Etheta"])) ** 2))
        / np.sqrt(np.mean(np.abs(ana_eq["Etheta"]) ** 2))
    )
    d_pat = np.asarray(n2f.directivity(theta_rad, phi_zero)[0], dtype=float)
    pat_err_grad = float(np.sqrt(np.mean((d_pat - d_eplane_ana) ** 2)))

    return {
        "cpw_list": cpw_list,
        "n_cells_unif": n_cells_unif,
        "wall_unif_s": wall_unif_s,
        "power_err_unif": power_err_unif,
        "near_err_unif": near_err_unif,
        "pat_err_unif": pat_err_unif,
        "n_cells_grad": np.array((g_grad.Nx - 1) * (g_grad.Ny - 1) * (g_grad.Nz - 1)),
        "wall_grad_s": np.array(wall_grad_s),
        "power_err_grad": np.array(power_err_grad),
        "near_err_grad": np.array(near_err_grad),
        "pat_err_grad": np.array(pat_err_grad),
        "cpw_eff_grad": np.array(LAM0_M / dx_c_grad),
    }


def run_task(task_name: str, data_dir: Path, quick: bool = False) -> Path:
    """Execute a named task and save its output to ``data_dir/task_<name>.npz``.

    Parameters
    ----------
    task_name : str
        Task identifier in ``ALL_TASKS``.
    data_dir : Path
        Target directory for ``.npz`` and ``.json`` summary files.
    quick : bool, optional
        If True, run in quick mode.

    Returns
    -------
    Path
        Path to the written ``.npz`` archive.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    runners = {
        "radial_zones_and_near2far": run_radial_zones_and_near2far,
        "dipole_length_and_cell_size": run_dipole_length_and_cell_size,
        "cpml_distance_study": run_cpml_distance_study,
        "resolution_and_mesh_study": run_resolution_and_mesh_study,
    }
    if task_name not in runners:
        raise ValueError(f"Unknown task {task_name!r}; expected one of {ALL_TASKS}")

    out = runners[task_name](quick=quick)
    npz_path = data_dir / f"task_{task_name}.npz"
    np.savez_compressed(npz_path, **out)

    # Also write a compact scalar summary JSON for quick inspection
    scalar_summary = {
        k: (float(v) if v.ndim == 0 and np.issubdtype(v.dtype, np.floating) else int(v))
        for k, v in out.items()
        if isinstance(v, np.ndarray) and v.ndim == 0
    }
    if scalar_summary:
        (data_dir / f"task_{task_name}.json").write_text(json.dumps(scalar_summary, indent=2) + "\n")
    return npz_path


# ---------------------------------------------------------------------------
# Figure generation suite (wakis.figstyle + wakis.schematic, <= 3 panels/fig)
# ---------------------------------------------------------------------------
def _build_schematic_objects(d0: Any) -> tuple[SolverFIT3D, CurrentSource, Near2FarMonitor, Box, dict[int, str]]:
    """Reconstruct lightweight solver + source + monitor + marker objects from stored grid parameters."""
    quick = bool(int(d0["quick"]))
    dx_fine = float(d0["dx_fine_m"])
    dx_coarse = float(d0["dx_coarse_m"])
    half_L = float(d0["half_L_m"])
    b_n2f = float(d0["b_n2f_m"])
    n_pml = int(d0["n_pml"])
    inner_half = 0.0225 if quick else 0.0200

    grid = build_point_dipole_grid(
        graded=True,
        dx_fine_m=dx_fine,
        dx_coarse_m=dx_coarse,
        inner_half_m=inner_half,
        domain_half_m=half_L,
        grading=1.24,
    )
    solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, verbose=0)
    pulse = GaussianSource(F0_HZ, 0.60 * F0_HZ)
    src = CurrentSource(pulse, component="z", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0), amplitude=I0_L_AM)
    n2f = Near2FarMonitor(frequencies=[F0_HZ], center=(0.0, 0.0, 0.0), size=snap_monitor_box(grid, b_n2f))
    n2f.faces = [n2f]
    dip_marker = Box((0.0, 0.0, 0.0), (2.0 * dx_fine, 2.0 * dx_fine, 3.0 * dx_fine), "pec")
    mat_names = {id(dip_marker): "Hertzian dipole (Il = 10 mA m)"}
    return solver, src, n2f, dip_marker, mat_names


def plot_00_setup_schematic(d0: Any, fig_dir: Path) -> Path:
    """Render Figures 00a/00b and setup table via :func:`wakis.schematic.save_setup_figures`."""
    figstyle.apply_style()
    solver, src, n2f, _, _ = _build_schematic_objects(d0)
    dx_fine = float(d0["dx_fine_m"])
    variations = [
        {"name": "Dipole length sweep", "details": r"$\ell / \lambda_0 \in [0.04, 0.50]$ ($N_{\mathrm{seg}} \in \{1, \dots, 25\}$)"},
        {"name": "CPML clearance study", "details": r"$d_{\mathrm{PML}} / \lambda_0 \in [0.045, 0.420]$ ($kd_{\mathrm{PML}} \in [0.28, 2.64]$)"},
        {"name": "Uniform mesh convergence", "details": r"$\lambda_0 / \Delta \in \{10, 14, 20, 26\}$ cells/$\lambda_0$"},
        {"name": "Extended radiation movie", "details": r"$6.63\,\lambda_0 \times 2.93\,\lambda_0 \times 6.63\,\lambda_0$ at $20.5$ cells/$\lambda_0$ ($r_{\max} = 3.32\,\lambda_0$)"},
    ]
    res = schematic.save_setup_figures(
        solver,
        fig_dir,
        plane="xz",
        cut_coord=0.0,
        unit="mm",
        freq=F0_HZ,
        min_feature=dx_fine,
        sources=[src],
        monitors=[n2f],
        title_geometry="3D Hertzian Point Dipole Setup ($xz$ plane, $y = 0$)",
        title_mesh="Locally Refined Graded FIT Mesh and Excitation",
        variations=variations,
    )

    # Reposition the point-source tag right next to the origin star so it sits on its own object
    fig_a, ax_a = schematic.plot_schematic(
        solver,
        plane="xz",
        cut_coord=0.0,
        unit="mm",
        freq=F0_HZ,
        sources=[src],
        monitors=[n2f],
        show_mesh=False,
        show_pulse_inset=False,
        title="3D Hertzian Point Dipole Setup ($xz$ plane, $y = 0$)",
    )
    for txt in ax_a.texts:
        if "1 ·" in txt.get_text():
            txt.set_position((0.0, 55.0))
            txt.set_ha("center")
        elif txt.get_text() == "CPML" and txt.get_position()[0] > 200.0:
            txt.set_position((405.0, -145.0))
            txt.set_ha("left")
    figstyle.save(fig_a, res["geometry_figure"], close=True, auto_legend_below=False)

    # Center the 1x zoom inset on the 5 mm fine core region around the dipole origin
    fig_b, _ = schematic.plot_mesh_and_excitation(
        solver,
        plane="xz",
        cut_coord=0.0,
        unit="mm",
        freq=F0_HZ,
        min_feature=dx_fine,
        sources=[src],
        zoom_bounds=(-0.040, 0.040, -0.040, 0.040),
        title="Locally Refined Graded FIT Mesh and Excitation",
    )
    figstyle.save(fig_b, res["mesh_figure"], close=True, auto_legend_below=False)
    return res["geometry_figure"]


def plot_01_radial_field_zones_kr1(d0: Any, fig_dir: Path) -> Path:
    """Render Figure 01: Radial decay zones (1/r^3, 1/r^2, 1/r) and phase transition across kr = 1."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=10.4, height_in=4.4, ncols=3, nrows=1)
    ax_eq, ax_ax, ax_ph = axes

    kr_eq = np.asarray(d0["kr_eq"], dtype=float)
    eth_fit = np.asarray(d0["etheta_eq_fit_abs"], dtype=float)
    eth_ana = np.asarray(d0["etheta_eq_ana_abs"], dtype=float)
    hphi_fit = np.asarray(d0["hphi_eq_fit_scaled"], dtype=float)
    hphi_ana = np.asarray(d0["hphi_eq_ana_scaled"], dtype=float)
    e_stat = np.asarray(d0["etheta_static"], dtype=float)
    e_ind = np.asarray(d0["etheta_induction"], dtype=float)
    e_rad = np.asarray(d0["etheta_radiation"], dtype=float)

    # Panel (a): Equatorial |E_theta| and eta0*|H_phi| with the 3 asymptotes crossing at kr = 1
    ax_eq.loglog(kr_eq, eth_ana, color=COLORS["blue"], lw=1.8, label=r"$|E_\theta|$ analytic")
    ax_eq.loglog(
        kr_eq, eth_fit, "o", color=COLORS["blue"], ms=3.5, alpha=0.85, label=r"$|E_\theta|$ 3D FIT"
    )
    ax_eq.loglog(kr_eq, hphi_ana, "--", color=COLORS["teal"], lw=1.6, label=r"$\eta_0|H_\phi|$ analytic")
    ax_eq.loglog(
        kr_eq[::2], hphi_fit[::2], "s", color=COLORS["teal"], ms=3.2, alpha=0.85, label=r"$\eta_0|H_\phi|$ FIT"
    )
    ax_eq.loglog(kr_eq, e_stat, ":", color=COLORS["red"], lw=1.1, label=r"Static $(kr)^{-3}$")
    ax_eq.loglog(kr_eq, e_ind, "-.", color=COLORS["purple"], lw=1.1, label=r"Induction $(kr)^{-2}$")
    ax_eq.loglog(kr_eq, e_rad, ":", color=COLORS["orange"], lw=1.2, label=r"Radiation $(kr)^{-1}$")
    ax_eq.axvline(1.0, color=COLORS["grey"], ls="--", lw=1.0)
    ax_eq.text(1.08, e_rad[0] * 1.4, "$kr = 1$", fontsize=8.5, color=figstyle.INK)
    ax_eq.set_xlabel(r"Electrical distance $kr = 2\pi r / \lambda_0$")
    ax_eq.set_ylabel(r"Spectral field amplitude [V s / m]")
    ax_eq.set_title(r"(a) Equatorial ($\theta = 90^\circ$)", pad=10)
    figstyle.place_legend_below(ax_eq, ncol=2, y_offset=-0.28)

    # Panel (b): Axial |E_r| along +z (theta = 0): only 1/r^3 and 1/r^2, no 1/r term
    kr_ax = np.asarray(d0["kr_ax"], dtype=float)
    er_fit = np.asarray(d0["er_ax_fit_abs"], dtype=float)
    er_ana = np.asarray(d0["er_ax_ana_abs"], dtype=float)
    er_stat = 2.0 * np.interp(kr_ax, kr_eq, e_stat)
    er_ind = 2.0 * np.interp(kr_ax, kr_eq, e_ind)

    ax_ax.loglog(kr_ax, er_ana, color=COLORS["blue"], lw=1.8, label=r"$|E_r|$ analytic ($\theta=0^\circ$)")
    ax_ax.loglog(kr_ax, er_fit, "o", color=COLORS["blue"], ms=3.5, alpha=0.85, label=r"$|E_r|$ 3D FIT")
    ax_ax.loglog(kr_ax, er_stat, ":", color=COLORS["red"], lw=1.1, label=r"Static $2(kr)^{-3}$")
    ax_ax.loglog(kr_ax, er_ind, "-.", color=COLORS["purple"], lw=1.1, label=r"Induction $2(kr)^{-2}$")
    ax_ax.axvline(1.0, color=COLORS["grey"], ls="--", lw=1.0)
    ax_ax.text(1.08, er_ind[-1] * 3.5, "$kr = 1$\n(no $1/r$ term)", fontsize=8.5, color=figstyle.INK)
    ax_ax.set_xlabel(r"Electrical distance $kr = 2\pi r / \lambda_0$")
    ax_ax.set_ylabel(r"Axial field $|E_r|$ [V s / m]")
    ax_ax.set_title(r"(b) Dipole axis ($\theta = 0^\circ$)", pad=10)
    figstyle.place_legend_below(ax_ax, ncol=2, y_offset=-0.28)

    # Panel (c): Wave impedance |Z_w| / eta0 and phase angle arg(-E_theta) - arg(H_phi)
    zw_fit = np.asarray(d0["zw_eq_fit_norm"], dtype=float)
    zw_ana = np.asarray(d0["zw_eq_ana_norm"], dtype=float)
    ph_fit = np.asarray(d0["phase_diff_fit_deg"], dtype=float)
    ph_ana = np.asarray(d0["phase_diff_ana_deg"], dtype=float)

    ax_ph.semilogx(kr_eq, ph_ana, color=COLORS["blue"], lw=1.8, label=r"$\angle E_\theta - \angle H_\phi$ analytic")
    ax_ph.semilogx(kr_eq, ph_fit, "o", color=COLORS["blue"], ms=3.3, label=r"$\angle E_\theta - \angle H_\phi$ FIT")
    ax_ph.semilogx(
        kr_eq, (zw_ana - 1.0) * 45.0, "--", color=COLORS["orange"], lw=1.5, label=r"$45(|Z_w|/\eta_0 - 1)$"
    )
    ax_ph.axvline(1.0, color=COLORS["grey"], ls="--", lw=1.0)
    ax_ph.axhline(0.0, color=COLORS["grey"], ls=":", lw=0.9)
    ax_ph.axhline(-90.0, color=COLORS["grey"], ls=":", lw=0.9)
    ax_ph.set_ylim(-100.0, 45.0)
    ax_ph.set_xlabel(r"Electrical distance $kr = 2\pi r / \lambda_0$")
    ax_ph.set_ylabel(r"E–H phase lead [deg]")
    ax_ph.set_title("(c) Reactive-to-radiative phase", pad=10)
    figstyle.place_legend_below(ax_ph, ncol=1, y_offset=-0.28)

    out = fig_dir / "01_radial_field_zones_kr1.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_02_near_and_far_field_maps(d0: Any, fig_dir: Path) -> Path:
    """Render Figure 02: 2D xz-plane Re(Ez) and log10(|E|) maps with overlay_geometry and kr = 1 circle."""
    figstyle.apply_style()
    solver, src, n2f, dip_marker, mat_names = _build_schematic_objects(d0)
    fig, axes = figstyle.new_figure(width_in=7.2, height_in=3.4, ncols=2, nrows=1)
    ax_re, ax_abs = axes

    x_mm = np.asarray(d0["x_ez"], dtype=float) * 1e3
    z_mm = np.asarray(d0["z_ez"], dtype=float) * 1e3
    ez_re = np.asarray(d0["ez_xz_real"], dtype=float)
    emag = np.asarray(d0["emag_xz"], dtype=float)
    r_kr1_mm = R_KR1_M * 1e3

    # Clip extreme self-cell peak so propagating radiation waves are clearly visible
    vmax_re = float(np.percentile(np.abs(ez_re), 96.5))
    im0 = ax_re.pcolormesh(
        x_mm,
        z_mm,
        ez_re.T,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-vmax_re,
        vmax=vmax_re,
        shading="auto",
    )
    fig.colorbar(im0, ax=ax_re, fraction=0.046, pad=0.03, label=r"$\mathrm{Re}[E_z(x,z,f_0)]$ [V s / m]")
    schematic.overlay_geometry(
        ax_re,
        solver,
        plane="xz",
        unit="mm",
        structures=[dip_marker],
        material_names=mat_names,
        sources=[src],
        monitors=[n2f],
        show_sources=True,
        show_monitors=True,
    )
    c0 = Circle((0.0, 0.0), r_kr1_mm, fill=False, ls="--", lw=1.2, edgecolor=COLORS["orange"], zorder=8)
    ax_re.add_patch(c0)
    ax_re.text(r_kr1_mm * 0.72, -r_kr1_mm * 0.95, "$kr=1$", fontsize=7.5, color=COLORS["orange"], zorder=9)
    ax_re.set_aspect("equal")
    ax_re.set_xlabel("$x$ [mm]")
    ax_re.set_ylabel("$z$ [mm]")
    ax_re.set_title(r"Phase fronts $\mathrm{Re}[E_z(x, z, f_0)]$")
    figstyle.panel_label(ax_re, "a")

    emag_db = 20.0 * np.log10(np.maximum(emag, 1e-30) / float(np.percentile(emag, 99.0)))
    im1 = ax_abs.pcolormesh(
        x_mm,
        z_mm,
        emag_db.T,
        cmap=figstyle.CMAP_MAGNITUDE,
        vmin=-45.0,
        vmax=0.0,
        shading="auto",
    )
    fig.colorbar(im1, ax=ax_abs, fraction=0.046, pad=0.03, label=r"$20\log_{10}(|\mathbf{E}| / E_{99\%})$ [dB]")
    schematic.overlay_geometry(
        ax_abs,
        solver,
        plane="xz",
        unit="mm",
        structures=[dip_marker],
        material_names=mat_names,
        sources=[src],
        monitors=[n2f],
        show_sources=True,
        show_monitors=True,
    )
    c1 = Circle((0.0, 0.0), r_kr1_mm, fill=False, ls="--", lw=1.2, edgecolor=COLORS["sky"], zorder=8)
    ax_abs.add_patch(c1)
    ax_abs.text(r_kr1_mm * 0.72, -r_kr1_mm * 0.95, "$kr=1$", fontsize=7.5, color=COLORS["sky"], zorder=9)
    ax_abs.set_aspect("equal")
    ax_abs.set_xlabel("$x$ [mm]")
    ax_abs.set_ylabel("$z$ [mm]")
    ax_abs.set_title(r"Near-to-far envelope $|\mathbf{E}(x, z, f_0)|$")
    figstyle.panel_label(ax_abs, "b")

    out = fig_dir / "02_near_and_far_field_maps.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_03_far_field_pattern_and_power(d0: Any, fig_dir: Path) -> Path:
    """Render Figure 03: Near-to-far directivity pattern sin^2(theta) and 3-way power conservation."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=10.4, height_in=4.4, ncols=3, nrows=1)
    ax_pat, ax_dbi, ax_pwr = axes

    theta_deg = np.asarray(d0["theta_deg"], dtype=float)
    d_eplane = np.asarray(d0["d_eplane"], dtype=float)
    d_ana = np.asarray(d0["d_eplane_ana"], dtype=float)
    phi_deg = np.asarray(d0["phi_deg"], dtype=float)
    d_hplane = np.asarray(d0["d_hplane"], dtype=float)
    hpbw_deg = float(d0["hpbw_deg"])

    # Panel (a): Linear directivity D(theta) in E-plane and H-plane vs 1.5*sin^2(theta)
    ax_pat.plot(theta_deg, d_ana, color=COLORS["blue"], lw=1.9, label=r"$1.5\sin^2\theta$ analytic")
    ax_pat.plot(
        theta_deg[::4], d_eplane[::4], "o", color=COLORS["blue"], ms=3.5, label="E-plane ($xz$) Near2Far"
    )
    ax_pat.plot(
        phi_deg[::6] * 0.5,
        d_hplane[::6],
        "s--",
        color=COLORS["teal"],
        ms=3.0,
        lw=1.2,
        label=r"H-plane ($xy$, $\phi/2$)",
    )
    ax_pat.axhline(0.75, color=COLORS["grey"], ls=":", lw=1.0)
    ax_pat.annotate(
        f"HPBW = {hpbw_deg:.1f}$^\\circ$\n(analytic 90.0$^\\circ$)",
        xy=(45.0, 0.75),
        xytext=(18.0, 0.25),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.8, color=figstyle.INK),
    )
    ax_pat.set_xlim(0.0, 180.0)
    ax_pat.set_ylim(0.0, 1.68)
    ax_pat.set_xlabel(r"Polar angle $\theta$ [deg]")
    ax_pat.set_ylabel(r"Directivity $D(\theta)$ [linear]")
    ax_pat.set_title(r"(a) Far-field pattern $D(\theta)$", pad=10)
    figstyle.place_legend_below(ax_pat, ncol=1, y_offset=-0.28)

    # Panel (b): Directivity in dBi
    d_dbi_fit = 10.0 * np.log10(np.maximum(d_eplane, 1e-3))
    d_dbi_ana = 10.0 * np.log10(np.maximum(d_ana, 1e-3))
    ax_dbi.plot(theta_deg, d_dbi_ana, color=COLORS["blue"], lw=1.8, label="Analytic ($D_0 = 1.76$ dBi)")
    ax_dbi.plot(theta_deg[::4], d_dbi_fit[::4], "o", color=COLORS["orange"], ms=3.4, label="Wakis Near2Far")
    ax_dbi.axhline(10.0 * np.log10(1.5), color=COLORS["grey"], ls=":", lw=1.0)
    ax_dbi.set_xlim(0.0, 180.0)
    ax_dbi.set_ylim(-18.0, 3.0)
    ax_dbi.set_xlabel(r"Polar angle $\theta$ [deg]")
    ax_dbi.set_ylabel(r"Directivity $10\log_{10} D(\theta)$ [dBi]")
    ax_dbi.set_title("(b) Directivity in dBi", pad=10)
    figstyle.place_legend_below(ax_dbi, ncol=1, y_offset=-0.28)

    # Panel (c): Spectral radiated power conservation across the Gaussian pulse band
    f_ghz = np.asarray(d0["freqs_band_ghz"], dtype=float)
    p_scale = float(np.max(d0["phi_ana_band"]))
    ax_pwr.plot(
        f_ghz,
        np.asarray(d0["phi_ana_band"]) / p_scale,
        color=COLORS["blue"],
        lw=1.9,
        label=r"Analytic $\frac{\eta_0 k^2}{6\pi}|I\ell(f)|^2$",
    )
    ax_pwr.plot(
        f_ghz,
        np.asarray(d0["phi_box_band"]) / p_scale,
        "o",
        color=COLORS["teal"],
        ms=4.0,
        label="Closed-box FluxMonitor",
    )
    ax_pwr.plot(
        f_ghz,
        np.asarray(d0["phi_n2f_band"]) / p_scale,
        "s",
        color=COLORS["orange"],
        ms=3.6,
        label="Near2Far radiated_power()",
    )
    ax_pwr.plot(
        f_ghz,
        np.asarray(d0["w_src_band"]) / p_scale,
        "^--",
        color=COLORS["purple"],
        ms=3.6,
        lw=1.1,
        label=r"Source work $-\mathrm{Re}[E_z(I\ell)^*]$",
    )
    ax_pwr.set_xlabel("Frequency $f$ [GHz]")
    ax_pwr.set_ylabel("Normalized spectral flux $\\Phi(f) / \\Phi_{\\max}$")
    ax_pwr.set_title("(c) 3-way power conservation", pad=10)
    figstyle.place_legend_below(ax_pwr, ncol=1, y_offset=-0.28)

    out = fig_dir / "03_far_field_pattern_and_power.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_04_radiation_resistance_point_limit(d1: Any, fig_dir: Path) -> Path:
    """Render Figure 04: Radiation resistance R = 80*pi^2*(l/lam)^2, finite-length breakdown & cell size."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=10.4, height_in=4.4, ncols=3, nrows=1)
    ax_r, ax_dev, ax_cell = axes

    l_pts = np.asarray(d1["l_over_lam"], dtype=float)
    r_fit = np.asarray(d1["r_rad_fit_ohm"], dtype=float)
    r_unif_pts = np.asarray(d1["r_rad_unif_ohm"], dtype=float)
    r_hertz_pts = np.asarray(d1["r_rad_hertz_ohm"], dtype=float)
    d0_pts = np.asarray(d1["d0_fit"], dtype=float)
    l_dense = np.asarray(d1["l_dense"], dtype=float)
    r_hertz_dense = np.asarray(d1["r_hertz_dense_ohm"], dtype=float)
    r_unif_dense = np.asarray(d1["r_unif_dense_ohm"], dtype=float)
    r_tri_dense = np.asarray(d1["r_tri_dense_ohm"], dtype=float)
    d0_unif_dense = np.asarray(d1["d0_unif_dense"], dtype=float)

    # Panel (a): R_rad(l/lam0) log-log comparison
    ax_r.loglog(
        l_dense, r_hertz_dense, "--", color=COLORS["orange"], lw=1.7, label=r"Hertzian $80\pi^2(\ell/\lambda_0)^2$"
    )
    ax_r.loglog(
        l_dense, r_unif_dense, color=COLORS["blue"], lw=1.8, label=r"Exact uniform filament $I(z)=I_0$"
    )
    ax_r.loglog(l_pts, r_fit, "o", color=COLORS["blue"], ms=4.5, label="3D FIT CurrentSource($\\ell$)")
    ax_r.loglog(
        l_dense, r_tri_dense, ":", color=COLORS["teal"], lw=1.5, label=r"Triangular short wire $20\pi^2(\ell/\lambda_0)^2$"
    )
    ax_r.axvline(0.15, color=COLORS["grey"], ls=":", lw=1.0)
    ax_r.set_xticks([0.03, 0.06, 0.10, 0.20, 0.50])
    ax_r.set_xticklabels(["0.03", "0.06", "0.10", "0.20", "0.50"])
    ax_r.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_r.set_xlabel(r"Electrical length $\ell / \lambda_0$")
    ax_r.set_ylabel(r"Radiation resistance $R_{\mathrm{rad}}$ [$\Omega$]")
    ax_r.set_title(r"(a) $R_{\mathrm{rad}}(\ell / \lambda_0)$ scaling", pad=10)
    figstyle.place_legend_below(ax_r, ncol=1, y_offset=-0.28)

    # Panel (b): Departure from point-like behaviour: R_rad / R_Hertz and directivity D0(l/lam0)
    ax_dev.plot(
        l_dense,
        r_unif_dense / r_hertz_dense,
        color=COLORS["blue"],
        lw=1.8,
        label=r"$R_{\mathrm{rad}} / [80\pi^2(\ell/\lambda_0)^2]$ exact",
    )
    ax_dev.plot(
        l_pts,
        r_fit / r_hertz_pts,
        "o",
        color=COLORS["blue"],
        ms=4.2,
        label=r"$R_{\mathrm{rad}} / R_{\mathrm{Hertz}}$ 3D FIT",
    )
    ax_dev.plot(
        l_dense,
        d0_unif_dense / 1.5,
        "--",
        color=COLORS["purple"],
        lw=1.6,
        label=r"$D_0(\ell/\lambda_0) / 1.5$ exact",
    )
    ax_dev.plot(
        l_pts,
        d0_pts / 1.5,
        "s",
        color=COLORS["purple"],
        ms=3.8,
        label=r"$D_0 / 1.5$ 3D FIT",
    )
    ax_dev.axvspan(0.0, 0.15, color=COLORS["green"], alpha=0.10, label="Point-dipole regime ($<1.5\\%$ dev.)")
    ax_dev.axvline(0.15, color=COLORS["grey"], ls="--", lw=1.0)
    ax_dev.set_xlim(0.02, 0.52)
    ax_dev.set_ylim(0.72, 1.25)
    ax_dev.set_xlabel(r"Electrical length $\ell / \lambda_0$")
    ax_dev.set_ylabel("Ratio to ideal point dipole")
    ax_dev.set_title("(b) Point-dipole validity limit", pad=10)
    figstyle.place_legend_below(ax_dev, ncol=1, y_offset=-0.28)

    # Panel (c): How finite cell size regularizes the point dipole at r -> 0
    r_fine_mm = np.asarray(d1["r_fine_4mm_m"], dtype=float) * 1e3
    ez_fine = np.asarray(d1["ez_fine_4mm_norm"], dtype=float)
    r_coarse_mm = np.asarray(d1["r_coarse_10mm_m"], dtype=float) * 1e3
    ez_coarse = np.asarray(d1["ez_coarse_10mm_norm"], dtype=float)
    r_ref_mm = np.linspace(3.5, 75.0, 120)
    ana_ref = hertzian_fields_spherical(r_ref_mm * 1e-3, np.full_like(r_ref_mm, 0.5 * np.pi), F0_HZ, I0_L_AM)

    ax_cell.semilogy(r_ref_mm, np.abs(ana_ref["Etheta"]), color=COLORS["blue"], lw=1.8, label="Continuum $1/r^3$ singular")
    ax_cell.semilogy(
        r_fine_mm, ez_fine, "o-", color=COLORS["teal"], ms=3.6, lw=1.3, label="FIT cell $\\Delta = 4\\text{–}5$ mm"
    )
    ax_cell.semilogy(
        r_coarse_mm, ez_coarse, "s--", color=COLORS["orange"], ms=3.8, lw=1.3, label="FIT cell $\\Delta = 10$ mm"
    )
    ax_cell.set_xlim(0.0, 72.0)
    ax_cell.set_xlabel("Equatorial distance $r$ [mm]")
    ax_cell.set_ylabel(r"$|E_z(r)| / |S(f_0)|$ [V/m]")
    ax_cell.set_title("(c) Finite-cell regularization", pad=10)
    figstyle.place_legend_below(ax_cell, ncol=1, y_offset=-0.28)

    out = fig_dir / "04_radiation_resistance_point_limit.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_05_resolution_and_cpml_study(d2: Any, d3: Any, fig_dir: Path) -> Path:
    """Render Figure 05: Resolution & graded-mesh convergence and CPML clearance study."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=9.2, height_in=4.4, ncols=2, nrows=1)
    ax_res, ax_pml = axes

    cpw = np.asarray(d3["cpw_list"], dtype=float)
    p_err_u = np.asarray(d3["power_err_unif"], dtype=float) * 100.0
    n_err_u = np.asarray(d3["near_err_unif"], dtype=float) * 100.0
    pat_err_u = np.asarray(d3["pat_err_unif"], dtype=float) * 100.0
    cpw_g = float(d3["cpw_eff_grad"])
    p_err_g = float(d3["power_err_grad"]) * 100.0
    n_err_g = float(d3["near_err_grad"]) * 100.0
    n_cells_g = int(d3["n_cells_grad"])
    n_cells_u_max = int(d3["n_cells_unif"][-1])

    # Panel (a): Resolution study vs cpw + graded mesh comparison
    ref_o2 = n_err_u[0] * ((cpw / cpw[0]) ** -2)
    ax_res.loglog(cpw, n_err_u, "o-", color=COLORS["blue"], ms=4.5, lw=1.7, label="Uniform: near-field $L_2$ err")
    ax_res.loglog(cpw, p_err_u, "s-", color=COLORS["teal"], ms=4.2, lw=1.6, label="Uniform: power flux err")
    ax_res.loglog(cpw, pat_err_u, "^--", color=COLORS["purple"], ms=4.0, lw=1.3, label="Uniform: pattern $L_2$ err")
    ax_res.loglog(cpw, ref_o2, ":", color=COLORS["grey"], lw=1.2, label=r"$\mathcal{O}(\mathrm{cpw}^{-2})$ slope")
    ax_res.loglog(
        [cpw_g],
        [n_err_g],
        "*",
        color=COLORS["red"],
        ms=9.5,
        label=f"Graded mesh: near-field ({n_cells_g // 1000}k vs {n_cells_u_max // 1000}k cells)",
    )
    ax_res.loglog([cpw_g], [p_err_g], "D", color=COLORS["orange"], ms=6.0, label="Graded mesh: power err")
    ax_res.set_xticks([10, 14, 20, 26])
    ax_res.set_xticklabels(["10", "14", "20", "26"])
    ax_res.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_res.set_xlabel(r"Outer resolution $\lambda_0 / \Delta_{\max}$ [cells/$\lambda_0$]")
    ax_res.set_ylabel("Relative error [%]")
    ax_res.set_title("(a) Mesh convergence: Graded vs. Uniform", pad=10)
    figstyle.place_legend_below(ax_res, ncol=1, y_offset=-0.28)

    # Panel (b): CPML clearance study vs d_PML / lam0
    d_lam = np.asarray(d2["d_over_lam"], dtype=float)
    err_box = np.asarray(d2["rel_err_box"], dtype=float) * 100.0
    err_n2f = np.asarray(d2["rel_err_n2f"], dtype=float) * 100.0
    err_wsrc = np.asarray(d2["rel_err_wsrc"], dtype=float) * 100.0
    err_pat = np.asarray(d2["pattern_l2_err"], dtype=float) * 100.0

    ax_pml.semilogy(d_lam, err_box, "o-", color=COLORS["teal"], ms=4.3, lw=1.7, label="FluxMonitor power err")
    ax_pml.semilogy(d_lam, err_n2f, "s-", color=COLORS["orange"], ms=4.0, lw=1.6, label="Near2Far power err")
    ax_pml.semilogy(
        d_lam, err_wsrc, "^--", color=COLORS["purple"], ms=4.0, lw=1.4, label=r"Source work $W_{\mathrm{src}}$ err"
    )
    ax_pml.semilogy(d_lam, err_pat, "d:", color=COLORS["blue"], ms=4.0, lw=1.4, label="Far-field pattern $L_2$ err")
    ax_pml.axvline(1.0 / (2.0 * np.pi), color=COLORS["red"], ls="--", lw=1.1, label=r"$kr_{\mathrm{PML}} = 1$ ($0.159\lambda_0$)")
    ax_pml.set_xlabel(r"Source-to-CPML clearance $d_{\mathrm{PML}} / \lambda_0$")
    ax_pml.set_ylabel("Relative error [%]")
    ax_pml.set_title("(b) CPML reactive near-field clearance", pad=10)
    figstyle.place_legend_below(ax_pml, ncol=1, y_offset=-0.28)

    out = fig_dir / "05_resolution_and_cpml_study.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_06_dipole_radiation_gif(d0: Any, fig_dir: Path) -> Path:
    """Render Figure 06: Animated time-domain GIF of the 3D point-dipole wave over r <= 3.32*lambda_0."""
    figstyle.apply_style()
    x_m = np.asarray(d0["td_x_ez"] if "td_x_ez" in d0 else d0["x_ez"], dtype=float)
    z_m = np.asarray(d0["td_z_ez"] if "td_z_ez" in d0 else d0["z_ez"], dtype=float)
    frames = np.asarray(d0["td_frames"], dtype=float)
    times_ns = np.asarray(d0["td_times_ns"], dtype=float)
    times_per = (
        np.asarray(d0["td_times_periods"], dtype=float)
        if "td_times_periods" in d0
        else times_ns * 1e-9 * F0_HZ
    )
    r_phys_lam = float(d0["td_r_phys_m"]) / LAM0_M if "td_r_phys_m" in d0 else float(np.max(np.abs(x_m))) / LAM0_M
    r_total_lam = float(d0["td_r_total_m"]) / LAM0_M if "td_r_total_m" in d0 else r_phys_lam * 1.09
    cpw = float(d0["td_cpw"]) if "td_cpw" in d0 else 20.5
    n_pml = int(d0["td_n_pml"]) if "td_n_pml" in d0 else 6

    x_lam = x_m / LAM0_M
    z_lam = z_m / LAM0_M
    r_lam = np.sqrt(x_lam[:, None] ** 2 + z_lam[None, :] ** 2)
    scaled_frames = frames[:, : len(x_lam), : len(z_lam)] * r_lam[None, :, :]

    far_mask = r_lam >= 0.25
    vmax = float(np.percentile(np.abs(scaled_frames[:, far_mask]), 99.0))
    if vmax <= 0.0:
        vmax = 1.0

    r_kr1_lam = R_KR1_M / LAM0_M
    pml_w = r_total_lam - r_phys_lam

    pil_images: list[Image.Image] = []
    for idx in range(scaled_frames.shape[0]):
        fig, ax = figstyle.new_figure(width_in=5.1, height_in=4.5, ncols=1, nrows=1)
        fig.set_dpi(160)
        im = ax.pcolormesh(
            x_lam,
            z_lam,
            scaled_frames[idx].T,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$(r / \lambda_0)\,E_z(x, 0, z, t)$ [V/m]")

        # Thin hatched CPML band at the 4 frame edges (cropped out of the field area)
        pml_kw = dict(
            facecolor="#F3E8E8",
            edgecolor=COLORS["red"],
            hatch="////",
            linewidth=0.7,
            zorder=7,
        )
        ax.add_patch(Rectangle((-r_total_lam, r_phys_lam), 2.0 * r_total_lam, pml_w, **pml_kw))
        ax.add_patch(Rectangle((-r_total_lam, -r_total_lam), 2.0 * r_total_lam, pml_w, **pml_kw))
        ax.add_patch(Rectangle((-r_total_lam, -r_phys_lam), pml_w, 2.0 * r_phys_lam, **pml_kw))
        ax.add_patch(Rectangle((r_phys_lam, -r_phys_lam), pml_w, 2.0 * r_phys_lam, **pml_kw))
        ax.text(
            0.0,
            r_phys_lam + 0.5 * pml_w,
            f"CPML ({n_pml} cells)",
            ha="center",
            va="center",
            fontsize=7.2,
            color=COLORS["red"],
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=COLORS["red"], lw=0.5, alpha=0.92),
            zorder=10,
        )

        # Near-field kr = 1 circle and 1, 2, 3 lambda_0 range rings
        c_nf = Circle((0.0, 0.0), r_kr1_lam, fill=False, ls="--", lw=1.1, edgecolor=COLORS["orange"], zorder=8)
        ax.add_patch(c_nf)
        for m_ring in (1, 2, 3):
            c_r = Circle((0.0, 0.0), float(m_ring), fill=False, ls=":", lw=0.65, edgecolor=COLORS["grey"], alpha=0.65, zorder=6)
            ax.add_patch(c_r)
            ang = 0.25 * np.pi
            ax.text(
                m_ring * np.cos(ang),
                -m_ring * np.sin(ang),
                f"${m_ring}\\lambda_0$",
                fontsize=6.8,
                color=COLORS["grey"],
                ha="left",
                va="top",
                zorder=9,
            )

        # Labelled point dipole source at the origin
        ax.plot(0.0, 0.0, marker="*", markersize=8.5, color=COLORS["red"], markeredgecolor="white", markeredgewidth=0.6, zorder=10)
        ax.annotate(
            r"Point dipole $J_z$",
            xy=(0.0, 0.0),
            xytext=(0.38, 0.42),
            fontsize=7.5,
            color=figstyle.INK,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=COLORS["red"], lw=0.6, alpha=0.90),
            arrowprops=dict(arrowstyle="->", lw=0.75, color=COLORS["red"]),
            zorder=11,
        )
        ax.annotate(
            r"Near field ($kr=1$)",
            xy=(-r_kr1_lam * 0.71, -r_kr1_lam * 0.71),
            xytext=(-1.55, -0.68),
            fontsize=7.0,
            color=COLORS["orange"],
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=COLORS["orange"], lw=0.5, alpha=0.88),
            arrowprops=dict(arrowstyle="->", lw=0.7, color=COLORS["orange"]),
            zorder=11,
        )

        ax.set_xlim(-r_total_lam, r_total_lam)
        ax.set_ylim(-r_total_lam, r_total_lam)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x / \lambda_0$ ($\lambda_0 = 300$ mm)")
        ax.set_ylabel(r"$z / \lambda_0$")
        ax.set_title(
            f"3D Point Dipole Wave ($t = {times_per[idx]:.2f}\\,T_0$, {cpw:.1f} cells/$\\lambda_0$)"
        )
        figstyle.finish(fig, max_panels=1, auto_legend_below=False)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_images.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        plt.close(fig)

    out = fig_dir / "06_dipole_radiation.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_images[0].save(
        out,
        save_all=True,
        append_images=pil_images[1:],
        duration=100,
        loop=0,
        optimize=True,
    )
    return out


def generate_all_figures(data_dir: Path, fig_dir: Path) -> list[Path]:
    """Load ``data_dir/task_*.npz`` and render all Tutorial 4.1 figures into ``fig_dir``.

    Parameters
    ----------
    data_dir : Path
        Directory containing the saved ``.npz`` task archives.
    fig_dir : Path
        Output directory for PNG and GIF figures.

    Returns
    -------
    list of Path
        Paths of all generated figure files.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    d0 = np.load(data_dir / "task_radial_zones_and_near2far.npz")
    d1 = np.load(data_dir / "task_dipole_length_and_cell_size.npz")
    d2 = np.load(data_dir / "task_cpml_distance_study.npz")
    d3 = np.load(data_dir / "task_resolution_and_mesh_study.npz")

    fig00a = plot_00_setup_schematic(d0, fig_dir)
    return [
        fig00a,
        fig_dir / "00b_setup_mesh_and_excitation.png",
        plot_01_radial_field_zones_kr1(d0, fig_dir),
        plot_02_near_and_far_field_maps(d0, fig_dir),
        plot_03_far_field_pattern_and_power(d0, fig_dir),
        plot_04_radiation_resistance_point_limit(d1, fig_dir),
        plot_05_resolution_and_cpml_study(d2, d3, fig_dir),
        plot_06_dipole_radiation_gif(d0, fig_dir),
    ]
