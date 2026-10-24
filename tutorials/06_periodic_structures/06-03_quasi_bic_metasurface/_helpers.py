# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics and numerical helper utilities for Tutorial 6.3:
Quasi-bound states in the continuum: metasurface Q ∝ α⁻² (`06-03_quasi_bic_metasurface`).

Primary reference
-----------------
K. Koshelev, S. Lepeshov, M. Liu, A. Bogdanov, and Y. Kivshar,
"Asymmetric metasurfaces with high-Q resonances governed by bound states in the
continuum", Phys. Rev. Lett. 121(19), 193903 (2018),
DOI: 10.1103/PhysRevLett.121.193903 (arXiv:1809.00330).
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
from scipy.constants import c as c_light, epsilon_0
from scipy.optimize import curve_fit

import wakis
from wakis.diffraction import DiffractionMonitor
from wakis.estimate import estimate, preflight
from wakis.floquet import AngleSweep, FloquetCell, rcwa_1d_grating
from wakis.resonances import ResonanceMonitor
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.meshing import graded_axis

# ---------------------------------------------------------------------------
# Physical & geometric constants for the 2D dimerized silicon bar-pair unit cell
# (Koshelev et al., PRL 121, 193903, 2018, Fig. 1e & Fig. 3b class, scaled so
# lambda_BIC = 1573.9 nm matches Fig. 2c's 1571 nm in the telecom C/L band)
# ---------------------------------------------------------------------------
PERIOD_P = 0.680e-6  # Unit-cell period p = 680 nm (p / lambda_BIC = 0.432 < 0.5) [m]
DIMER_D = 0.375 * PERIOD_P  # Center-to-center bar spacing d = (3/8) p = 255 nm [m]
WIDTH_W0 = 0.250 * PERIOD_P  # Nominal bar width w_0 = (1/4) p = 170 nm [m]
HEIGHT_H = 0.190e-6  # Silicon bar thickness h = 190 nm [m]
N_SI = 3.48  # Silicon refractive index in telecom band (~1.57 um)
EPS_SI = float(N_SI**2)  # Relative permittivity eps_Si = 12.1104
EPS_BG = 1.0  # Free-space background eps_bg = 1.0 (Koshelev et al. Fig. 2)
LAMBDA_BIC_REF = 1.573885e-6  # Reference BIC wavelength at nx = 32 [m]
F0_BIC_REF = float(c_light / LAMBDA_BIC_REF)  # Reference BIC frequency [Hz]

ALPHAS_DEFAULT = (0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20)
THETAS_OBLIQUE_DEG = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)

# ---------------------------------------------------------------------------
# Verified benchmark numbers from Koshelev et al., PRL 121, 193903 (2018)
# ---------------------------------------------------------------------------
KOSHELEV_PAPER_REF: dict[str, Any] = {
    "citation": (
        "K. Koshelev, S. Lepeshov, M. Liu, A. Bogdanov, and Y. Kivshar, "
        "Phys. Rev. Lett. 121, 193903 (2018), DOI: 10.1103/PhysRevLett.121.193903"
    ),
    "fig2_geometry": {
        "period_m": 1.320e-6,
        "semi_axis_a_m": 0.330e-6,
        "semi_axis_b_m": 0.110e-6,
        "height_m": 0.200e-6,
        "inter_bar_d_m": 0.660e-6,
        "eps_bg": 1.0,
        "source": "Koshelev et al. (2018), Fig. 2 caption, p. 2",
    },
    "fig2c_spectra": {
        "lambda_bic_nm": 1571.0,
        "lambda_theta_0p5deg_nm": 1570.7,
        "lambda_theta_3deg_nm": 1569.7,
        "lambda_theta_5deg_nm": 1568.1,
        "lambda_theta_10deg_nm": 1564.0,
        "q_rad_theta_3deg": 5000.0,
        "q_rad_theta_10deg": 450.0,
        "q0_sin2_theta": 13.7,
        "source": "read from Fig. 2(b,c) & Fig. 4 (approx.), pp. 2, 4",
    },
    "eq3_scaling": {
        "exponent": -2.0,
        "formula": "Q_rad(alpha) = Q_0 * alpha**(-2)",
        "source": "Koshelev et al. (2018), Eq. (3) & Fig. 3(a), p. 3",
    },
    "fig4_saturation": {
        "q_abs_si_10e5": 1.0e5,
        "formula": "gamma_rad(alpha) = gamma_tot(alpha) - gamma_tot(0)",
        "source": "Koshelev et al. (2018), Fig. 4 & text on p. 4",
    },
}


def dimer_bar_geometry(
    alpha: float = 0.0,
    p: float = PERIOD_P,
    d: float = DIMER_D,
    w0: float = WIDTH_W0,
    h: float = HEIGHT_H,
) -> dict[str, float]:
    """
    Return bar widths, centers, edge coordinates, and gaps for the dimer unit cell.

    The two dielectric bars are centered at ``x_1 = -d / 2`` and ``x_2 = +d / 2``
    with perturbed widths ``w_1 = w_0 * (1 + alpha / 2)`` and
    ``w_2 = w_0 * (1 - alpha / 2)``, so that the relative width difference is
    ``(w_1 - w_2) / w_0 = alpha`` (and the single-bar relative perturbation is
    ``alpha_half = Delta_w / (2 * w_0) = alpha / 2``).

    Parameters
    ----------
    alpha : float, optional
        Dimensionless width-asymmetry parameter ``(w_1 - w_2) / w_0``. Default 0.0.
    p : float, optional
        Unit-cell period [m]. Default ``PERIOD_P`` (680 nm).
    d : float, optional
        Center-to-center spacing between the two bars [m]. Default ``DIMER_D`` (255 nm).
    w0 : float, optional
        Nominal bar width at ``alpha = 0`` [m]. Default ``WIDTH_W0`` (170 nm).
    h : float, optional
        Bar height along ``z`` [m]. Default ``HEIGHT_H`` (190 nm).

    Returns
    -------
    dict of str to float
        Dictionary with bar widths ``w1``, ``w2``, centers ``x1_c``, ``x2_c``,
        horizontal intervals ``(x1_min, x1_max)``, ``(x2_min, x2_max)``, vertical
        interval ``(z_min, z_max)``, inner gap ``gap_inner``, and outer gap
        ``gap_outer`` [m].
    """
    w1 = float(w0 * (1.0 + 0.5 * alpha))
    w2 = float(w0 * (1.0 - 0.5 * alpha))
    x1_c = -0.5 * d
    x2_c = +0.5 * d
    x1_min, x1_max = x1_c - 0.5 * w1, x1_c + 0.5 * w1
    x2_min, x2_max = x2_c - 0.5 * w2, x2_c + 0.5 * w2
    gap_inner = x2_min - x1_max
    gap_outer = p - (x2_max - x1_min)
    return {
        "alpha": float(alpha),
        "alpha_half": 0.5 * float(alpha),
        "p": float(p),
        "d": float(d),
        "w0": float(w0),
        "h": float(h),
        "w1": w1,
        "w2": w2,
        "x1_c": x1_c,
        "x2_c": x2_c,
        "x1_min": x1_min,
        "x1_max": x1_max,
        "x2_min": x2_min,
        "x2_max": x2_max,
        "z_min": -0.5 * float(h),
        "z_max": +0.5 * float(h),
        "gap_inner": float(gap_inner),
        "gap_outer": float(gap_outer),
    }


def compute_dimer_fill_2d(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    alpha: float = 0.0,
    averaging: str = "subpixel",
    p: float = PERIOD_P,
    d: float = DIMER_D,
    w0: float = WIDTH_W0,
    h: float = HEIGHT_H,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 2D dual-cell silicon volume fraction ``fill[i, k]`` on a Wakis grid.

    For ``Ey`` nodes located at ``(x_nodes[i], z_nodes[k])``, the dual cell along
    the uniform periodic ``x`` axis spans ``[x_nodes[i] - dx/2, x_nodes[i] + dx/2]``,
    and along the (possibly graded) ``z`` axis spans the dual interval bounded by
    adjacent primary cell midpoints.

    Parameters
    ----------
    grid_x : ndarray
        ``solver.grid.x`` array (length ``Nx + 1``).
    grid_z : ndarray
        ``solver.grid.z`` array (length ``Nz + 1``).
    alpha : float, optional
        Dimensionless width-asymmetry parameter ``(w_1 - w_2) / w_0``. Default 0.0.
    averaging : {"subpixel", "staircase"}, optional
        ``"subpixel"`` uses exact analytical cell-interval overlap fractions;
        ``"staircase"`` thresholds the fill fraction at ``0.5`` (midpoint rule).
    p, d, w0, h : float, optional
        Unit-cell period, dimer spacing, nominal bar width, and bar height [m].

    Returns
    -------
    fill : ndarray, shape (Nx, Nz)
        2D volume fraction in ``[0, 1]`` (or ``{0, 1}`` for ``"staircase"``).
    f_x : ndarray, shape (Nx,)
        1D horizontal fill fraction per ``x`` node.
    f_z : ndarray, shape (Nz,)
        1D vertical fill fraction per ``z`` node.
    """
    geom = dimer_bar_geometry(alpha=alpha, p=p, d=d, w0=w0, h=h)
    x_nodes = np.asarray(grid_x[:-1], dtype=float)
    z_nodes = np.asarray(grid_z[:-1], dtype=float)
    dx = float(grid_x[1] - grid_x[0])

    x_lo = x_nodes - 0.5 * dx
    x_hi = x_nodes + 0.5 * dx
    f_x1 = (
        np.clip(
            np.minimum(x_hi, geom["x1_max"]) - np.maximum(x_lo, geom["x1_min"]),
            0.0,
            dx,
        )
        / dx
    )
    f_x2 = (
        np.clip(
            np.minimum(x_hi, geom["x2_max"]) - np.maximum(x_lo, geom["x2_min"]),
            0.0,
            dx,
        )
        / dx
    )
    f_x = f_x1 + f_x2

    z_mids = 0.5 * (grid_z[:-1] + grid_z[1:])
    z_lo = np.empty_like(z_nodes)
    z_lo[0] = float(grid_z[0])
    z_lo[1:] = z_mids[:-1]
    z_hi = z_mids
    dz_dual = z_hi - z_lo
    f_z = (
        np.clip(
            np.minimum(z_hi, geom["z_max"]) - np.maximum(z_lo, geom["z_min"]),
            0.0,
            dz_dual,
        )
        / dz_dual
    )

    avg_mode = averaging.strip().lower()
    if avg_mode == "staircase":
        f_x = (f_x >= 0.5).astype(float)
        f_z = (f_z >= 0.5).astype(float)
    elif avg_mode != "subpixel":
        raise ValueError(f"Unknown averaging={averaging!r}; expected 'subpixel' or 'staircase'.")

    fill = f_x[:, None] * f_z[None, :]
    return fill, f_x, f_z


def make_dimer_solver(
    alpha: float = 0.0,
    nx: int = 32,
    kx: float = 0.0,
    sigma: float = 0.0,
    averaging: str = "subpixel",
    z_max: float = 1.30e-6,
    uniform_z: bool = False,
    dz_uniform: float = 0.019e-6,
    n_pml: int = 10,
    cfln: float = 0.95,
) -> tuple[wakis.SolverFIT3D, dict[str, Any]]:
    """
    Build a 2D TE (`Ey, Hx, Hz`) Wakis FIT unit-cell solver for the silicon bar dimer.

    Parameters
    ----------
    alpha : float, optional
        Asymmetry parameter ``(w_1 - w_2) / w_0``. Default 0.0.
    nx : int, optional
        Number of grid cells across the period ``p = 680 nm``. Default 32
        (giving ``dx = 21.25 nm``, ``cpw_Si = 21.3 >= 20``, and ``4.0`` cells
        across the ``85 nm`` inner gap).
    kx : float, optional
        In-plane Bloch wavenumber ``k_x`` [rad/m]. If non-zero, uses Bloch
        boundary conditions along ``x`` with ``dtype=np.complex128``. Default 0.0.
    sigma : float, optional
        Effective conductivity inside the silicon bars [S/m] used to model
        intrinsic/fabrication absorption loss ``Q_abs``. Default 0.0.
    averaging : {"subpixel", "staircase"}, optional
        Dielectric interface discretization mode. Default ``"subpixel"``.
    z_max : float, optional
        Half-height of the simulation domain along ``z`` [m]. Default ``1.30e-6``.
    uniform_z : bool, optional
        If True, use a uniform ``z`` grid of step ``dz_uniform`` (required when
        attaching ``DiffractionMonitor`` planes). If False (default), use a
        smoothly graded ``z`` mesh via ``wakis.templates.meshing.graded_axis``.
    dz_uniform : float, optional
        Uniform vertical step [m] when ``uniform_z=True``. Default ``19 nm``.
    n_pml : int, optional
        Number of CPML cells on ``z-`` and ``z+``. Default 10.
    cfln : float, optional
        Courant number. Default 0.95.

    Returns
    -------
    solver : SolverFIT3D
        Initialized Wakis solver with dielectric and conductivity tensors populated.
    meta : dict
        Mesh and resolution diagnostics (``dx_nm``, ``dz_min_nm``, ``cpw_si``,
        ``cells_across_gap``, ``d_air_to_pml_nm``, ``n_cells``, ``fill_2d``, etc.).
    """
    dx = PERIOD_P / int(nx)
    scale = 32.0 / float(nx)
    if uniform_z:
        nz = int(round(2.0 * z_max / dz_uniform))
        lz = nz * dz_uniform
        grid = wakis.GridFIT3D(
            -0.5 * PERIOD_P,
            0.5 * PERIOD_P,
            0.0,
            dx,
            -0.5 * lz,
            0.5 * lz,
            int(nx),
            1,
            nz,
            verbose=0,
        )
    else:
        z_grid = graded_axis(
            [-z_max, -0.5 * HEIGHT_H - 0.04e-6, 0.5 * HEIGHT_H + 0.04e-6, z_max],
            [0.045e-6 * scale, 0.015e-6 * scale, 0.045e-6 * scale],
            min_cells=4,
            grading=1.22,
        )
        x_grid = np.linspace(-0.5 * PERIOD_P, 0.5 * PERIOD_P, int(nx) + 1)
        y_grid = np.array([0.0, dx])
        grid = wakis.GridFIT3D(x=x_grid, y=y_grid, z=z_grid, verbose=0)

    use_bloch = abs(kx) > 0.0
    bc_x = "bloch" if use_bloch else "periodic"
    dtype = np.complex128 if use_bloch else float

    solver = wakis.SolverFIT3D(
        grid,
        bc_low=[bc_x, "periodic", "cpml"],
        bc_high=[bc_x, "periodic", "cpml"],
        n_pml=int(n_pml),
        k_bloch=(-float(kx), 0.0, 0.0),
        dtype=dtype,
        cfln=float(cfln),
        verbose=0,
    )

    fill, _, _ = compute_dimer_fill_2d(
        solver.grid.x, solver.grid.z, alpha=alpha, averaging=averaging
    )
    eps_r = EPS_BG + fill * (EPS_SI - EPS_BG)
    solver.ieps[:, 0, :, "y"] = 1.0 / (epsilon_0 * eps_r)

    if sigma > 0.0:
        solver.use_conductivity = True
        solver.sigma[:, 0, :, "y"] = fill * float(sigma)

    solver.update_tensors()
    solver.step_0 = True

    geom = dimer_bar_geometry(alpha=alpha)
    dz_steps = np.diff(solver.grid.z[:-1])
    dz_min = float(np.min(dz_steps))
    lam_si = LAMBDA_BIC_REF / N_SI
    cpw_si_x = lam_si / dx
    cpw_si_z = lam_si / dz_min
    min_feature = min(geom["gap_inner"], geom["w2"], geom["h"])
    cells_across_gap = geom["gap_inner"] / dx
    cells_across_min_feature = min(geom["gap_inner"] / dx, geom["w2"] / dx, geom["h"] / dz_min)
    z_pml_inner = float(solver.grid.z[-(int(n_pml) + 2)] - 0.5 * HEIGHT_H)

    meta = {
        "alpha": float(alpha),
        "alpha_half": 0.5 * float(alpha),
        "nx": int(nx),
        "nz": int(solver.Nz),
        "n_cells": int(solver.Nx * solver.Ny * solver.Nz),
        "dx_nm": float(dx * 1e9),
        "dz_min_nm": float(dz_min * 1e9),
        "dz_max_nm": float(np.max(dz_steps) * 1e9),
        "cpw_si_x": float(cpw_si_x),
        "cpw_si_z": float(cpw_si_z),
        "cpw_si_min": float(min(cpw_si_x, cpw_si_z)),
        "min_feature_nm": float(min_feature * 1e9),
        "cells_across_gap": float(cells_across_gap),
        "cells_across_min_feature": float(cells_across_min_feature),
        "d_air_to_pml_nm": float(z_pml_inner * 1e9),
        "dt_fs": float(solver.dt * 1e15),
        "x_nodes": np.asarray(solver.grid.x[:-1], dtype=float),
        "z_nodes": np.asarray(solver.grid.z[:-1], dtype=float),
        "fill_2d": fill,
        "eps_r_2d": eps_r,
        "geom": geom,
    }
    return solver, meta


class SymmetricDimerMaterialCallback:
    """
    Picklable material callback for :class:`wakis.floquet.FloquetCell` & :class:`AngleSweep`.

    Populates the symmetric (``alpha = 0``) or asymmetric (``alpha != 0``) silicon
    dimer permittivity ``eps_yy(x, z)`` on the uniform grid created by ``FloquetCell``.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        averaging: str = "subpixel",
        p: float = PERIOD_P,
        d: float = DIMER_D,
        w0: float = WIDTH_W0,
        h: float = HEIGHT_H,
    ):
        self.alpha = float(alpha)
        self.averaging = str(averaging)
        self.p = float(p)
        self.d = float(d)
        self.w0 = float(w0)
        self.h = float(h)

    def __call__(self, solver: wakis.SolverFIT3D) -> None:
        dx = float(solver.grid.dx[0])
        dz = float(solver.grid.dz[0])
        x_nodes = np.asarray(solver.grid.x[:-1], dtype=float)
        z_nodes = np.asarray(solver.grid.z[:-1], dtype=float)
        geom = dimer_bar_geometry(
            alpha=self.alpha, p=self.p, d=self.d, w0=self.w0, h=self.h
        )
        x_lo, x_hi = x_nodes - 0.5 * dx, x_nodes + 0.5 * dx
        f_x1 = (
            np.clip(
                np.minimum(x_hi, geom["x1_max"]) - np.maximum(x_lo, geom["x1_min"]),
                0.0,
                dx,
            )
            / dx
        )
        f_x2 = (
            np.clip(
                np.minimum(x_hi, geom["x2_max"]) - np.maximum(x_lo, geom["x2_min"]),
                0.0,
                dx,
            )
            / dx
        )
        f_x = f_x1 + f_x2
        z_lo, z_hi = z_nodes - 0.5 * dz, z_nodes + 0.5 * dz
        f_z = (
            np.clip(
                np.minimum(z_hi, geom["z_max"]) - np.maximum(z_lo, geom["z_min"]),
                0.0,
                dz,
            )
            / dz
        )
        if self.averaging.strip().lower() == "staircase":
            f_x = (f_x >= 0.5).astype(float)
            f_z = (f_z >= 0.5).astype(float)
        fill = f_x[:, None] * f_z[None, :]
        eps_r = EPS_BG + fill * (EPS_SI - EPS_BG)
        solver.ieps[:, :, :, "y"] = 1.0 / (epsilon_0 * eps_r[:, None, :])


def run_ringdown_harminv(
    alpha: float = 0.0,
    nx: int = 32,
    kx: float = 0.0,
    sigma: float = 0.0,
    averaging: str = "subpixel",
    z_max: float = 1.30e-6,
    ring_cycles: float = 38.0,
    f0_drive: float = F0_BIC_REF,
    fwidth_frac: float = 0.08,
    capture_fields: bool = False,
    n_gif_frames: int = 28,
) -> dict[str, Any]:
    """
    Excite the dimer metasurface with a localized ``J_y`` dipole pulse in Bar 1
    and extract the complex resonance frequency and ``Q`` via ``ResonanceMonitor.harminv``.

    Because the dipole is placed off-center at ``x = -d / 2`` inside the left bar,
    it couples strongly to the anti-symmetric (``C_2``-odd) BIC eigenmode even at
    ``alpha = 0`` where the mode is completely decoupled from normal-incidence
    plane waves.

    Parameters
    ----------
    alpha : float, optional
        Asymmetry parameter ``(w_1 - w_2) / w_0``. Default 0.0.
    nx : int, optional
        Number of grid cells per period ``p``. Default 32.
    kx : float, optional
        In-plane Bloch wavenumber ``k_x`` [rad/m]. Default 0.0.
    sigma : float, optional
        Conductivity inside the silicon bars [S/m]. Default 0.0.
    averaging : {"subpixel", "staircase"}, optional
        Dielectric averaging method. Default ``"subpixel"``.
    z_max : float, optional
        Half-height of the simulation domain [m]. Default ``1.30e-6``.
    ring_cycles : float, optional
        Number of optical cycles recorded after ``wf.end_time`` for ``harminv``.
        Default 38.0.
    f0_drive : float, optional
        Center frequency of the Gaussian pulse [Hz]. Default ``F0_BIC_REF``.
    fwidth_frac : float, optional
        Fractional bandwidth ``fwidth / f0_drive``. Default 0.08.
    capture_fields : bool, optional
        If True, record late-time 2D ``(x, z)`` slices of ``Ey``, ``Hx``, ``Hz``
        and a sequence of ``n_gif_frames`` frames during the ring-down.
    n_gif_frames : int, optional
        Number of ring-down frames to record when ``capture_fields=True``. Default 28.

    Returns
    -------
    dict
        Extracted resonance frequency ``f0_hz``, wavelength ``lambda0_nm``,
        quality factor ``Q``, decay rate ``gamma_rad_s``, ring-down trace,
        mesh metadata, and optional 2D field maps.
    """
    t0_wall = time.perf_counter()
    solver, meta = make_dimer_solver(
        alpha=alpha,
        nx=nx,
        kx=kx,
        sigma=sigma,
        averaging=averaging,
        z_max=z_max,
    )
    dx = PERIOD_P / int(nx)
    wf = GaussianSource(frequency=float(f0_drive), fwidth=float(fwidth_frac * f0_drive))
    src = CurrentSource(
        wf,
        component="y",
        center=(-0.5 * DIMER_D, 0.5 * dx, 0.0),
        size=(0.0, dx, 0.0),
    )
    mon_left = ResonanceMonitor(
        "Ey",
        center=(-0.5 * DIMER_D, 0.5 * dx, 0.0),
        t_start=wf.end_time,
    )
    mon_right = ResonanceMonitor(
        "Ey",
        center=(+0.5 * DIMER_D, 0.5 * dx, 0.0),
        t_start=wf.end_time,
    )

    n_steps = int((wf.end_time + float(ring_cycles) / float(f0_drive)) / solver.dt)

    if not capture_fields:
        solver.emsolve(n_steps, source=src, monitors=[mon_left, mon_right])
        frames_ey = None
        frames_t_fs = None
        ey_2d = None
        hx_2d = None
        hz_2d = None
    else:
        n_start_gif = int(round((wf.end_time + 4.0 / f0_drive) / solver.dt))
        n_end_gif = n_steps - 1
        frame_steps = set(
            np.unique(np.linspace(n_start_gif, n_end_gif, int(n_gif_frames), dtype=int))
        )
        frames_list = []
        frames_t_list = []
        best_amp = -1.0
        ey_snap = None
        hx_snap = None
        hz_snap = None
        ix_left = int(np.argmin(np.abs(meta["x_nodes"] - (-0.5 * DIMER_D))))
        iz_mid = int(np.argmin(np.abs(meta["z_nodes"])))
        n_snap_min = int(round((wf.end_time + 8.0 / f0_drive) / solver.dt))
        n_snap_max = int(round((wf.end_time + 12.0 / f0_drive) / solver.dt))

        for step in range(n_steps):
            t = step * float(solver.dt)
            src.update(solver, t)
            solver.one_step()
            mon_left.update(solver, t)
            mon_right.update(solver, t)

            if step in frame_steps:
                ey_mat = np.real(solver.E.to_matrix("y")[:, 0, :]).copy()
                frames_list.append(ey_mat)
                frames_t_list.append((t + float(solver.dt)) * 1e15)

            if n_snap_min <= step <= n_snap_max:
                val_left = float(np.real(solver.E.to_matrix("y")[ix_left, 0, iz_mid]))
                if val_left > best_amp:
                    best_amp = val_left
                    ey_snap = np.real(solver.E.to_matrix("y")[:, 0, :]).copy()
                    hx_snap = np.real(solver.H.to_matrix("x")[:, 0, :]).copy()
                    hz_snap = np.real(solver.H.to_matrix("z")[:, 0, :]).copy()

        frames_ey = np.asarray(frames_list, dtype=np.float32)
        frames_t_fs = np.asarray(frames_t_list, dtype=float)
        ey_2d = ey_snap
        hx_2d = hx_snap
        hz_2d = hz_snap

    res = mon_left.harminv(0.90 * f0_drive, 1.10 * f0_drive)
    if len(res.frequency) == 0:
        res = mon_left.harminv(0.88 * f0_drive, 1.12 * f0_drive, max_rel_error=None)

    idx_mode = int(np.argmax(np.abs(res.amplitude)))
    f0_mode = float(res.frequency[idx_mode])
    q_mode = float(abs(res.Q[idx_mode]))
    gamma_mode = float(abs(res.decay_rate[idx_mode]))
    err_mode = float(res.error[idx_mode])
    lam0_nm = float(c_light / f0_mode * 1e9)

    t_sig, x_left = mon_left.signal()
    _, x_right = mon_right.signal()

    out = {
        "alpha": float(alpha),
        "alpha_half": 0.5 * float(alpha),
        "kx": float(kx),
        "sigma": float(sigma),
        "averaging": str(averaging),
        "f0_hz": f0_mode,
        "lambda0_nm": lam0_nm,
        "Q": q_mode,
        "Q_signed": float(res.Q[idx_mode]),
        "Q_alpha2": float(q_mode * (alpha**2)),
        "Q_alpha_half2": float(q_mode * ((0.5 * alpha) ** 2)),
        "gamma_s_inv": gamma_mode,
        "harminv_error_hz": err_mode,
        "t_ring_fs": (t_sig - t_sig[0]) * 1e15,
        "signal_left": np.real(x_left),
        "signal_right": np.real(x_right),
        "n_steps": int(n_steps),
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
        "meta": meta,
    }
    if capture_fields:
        out["ey_2d"] = ey_2d
        out["hx_2d"] = hx_2d
        out["hz_2d"] = hz_2d
        out["frames_ey"] = frames_ey
        out["frames_t_fs"] = frames_t_fs
    return out


def extract_ringdown_envelope(
    t_fs: np.ndarray,
    signal: np.ndarray,
    window_fs: float = 5.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the normalized cycle-peak envelope of a real ring-down time trace.

    Groups the time trace into non-overlapping optical-cycle windows of width
    ``window_fs`` [fs] and takes the maximum of ``|signal|`` in each window so
    that plotting on a logarithmic axis shows the smooth decay envelope rather
    than carrier zero-crossings.

    Parameters
    ----------
    t_fs : 1D ndarray
        Ring-down time array ``t - t_off`` [fs].
    signal : 1D ndarray
        Real time-domain electric field trace ``E_y(t)`` [V/m].
    window_fs : float, optional
        Cycle-peak window width [fs] (default ``5.5`` fs, roughly one optical
        period at ``lambda_0 ~ 1574 nm``).

    Returns
    -------
    t_env_fs : 1D ndarray
        Window-center times [fs].
    env_norm : 1D ndarray
        Normalized cycle-peak envelope ``|E_y|_peak(t) / |E_y|_peak(0)`` [-].
    """
    t_arr = np.asarray(t_fs, dtype=float)
    s_abs = np.abs(np.asarray(signal, dtype=float))
    if t_arr.size == 0:
        raise ValueError("Empty ring-down signal provided to extract_ringdown_envelope.")
    t_min, t_max = float(t_arr[0]), float(t_arr[-1])
    n_bins = max(4, int(np.floor((t_max - t_min) / max(float(window_fs), 1e-6))))
    edges = np.linspace(t_min, t_max, n_bins + 1)
    t_env, env = [], []
    for i in range(n_bins):
        mask = (t_arr >= edges[i]) & (t_arr <= edges[i + 1])
        if np.any(mask):
            idx_sub = np.flatnonzero(mask)
            idx_pk = int(idx_sub[int(np.argmax(s_abs[idx_sub]))])
            t_env.append(float(t_arr[idx_pk]))
            env.append(float(s_abs[idx_pk]))
    t_env_arr = np.asarray(t_env, dtype=float)
    env_arr = np.asarray(env, dtype=float)
    ref_amp = float(env_arr[0]) if env_arr.size > 0 and env_arr[0] > 0.0 else 1.0
    return t_env_arr, env_arr / ref_amp


def fit_q_scaling_laws(
    alphas: np.ndarray,
    q_vals: np.ndarray,
) -> dict[str, float]:
    """Compute both the slope -2 imposed fit and the free power-law fit for Q(alpha).

    Parameters
    ----------
    alphas : 1D ndarray
        Asymmetry parameter values ``alpha > 0`` [-].
    q_vals : 1D ndarray
        Radiative quality factors ``Q_rad(alpha)`` [-].

    Returns
    -------
    fits : dict[str, float]
        Dictionary with ``q0_imposed`` (from ``Q = Q_0 * alpha**(-2)`` with slope
        ``-2`` imposed), ``q0_half_imposed`` (``0.25 * q0_imposed``), ``slope_free``
        and ``q0_free`` (from an unconstrained log-log linear regression
        ``log10 Q = s * log10 alpha + log10 Q_0,free``).
    """
    a_arr = np.asarray(alphas, dtype=float)
    q_arr = np.asarray(q_vals, dtype=float)
    if np.any(a_arr <= 0.0) or np.any(q_arr <= 0.0):
        raise ValueError("alphas and q_vals must be strictly positive.")
    log_a = np.log10(a_arr)
    log_q = np.log10(q_arr)
    slope_free, intercept_free = np.polyfit(log_a, log_q, 1)
    q0_free = float(10.0**intercept_free)
    # Imposed slope -2 fit in log space: log10(Q0) = mean(log10(Q) + 2*log10(alpha))
    q0_imposed = float(10.0 ** np.mean(log_q + 2.0 * log_a))
    return {
        "q0_imposed": q0_imposed,
        "q0_half_imposed": 0.25 * q0_imposed,
        "slope_free": float(slope_free),
        "q0_free": q0_free,
        "q0_half_free": 0.25 * q0_free,
    }


def fano_lineshape_eq1(
    freq: np.ndarray | float,
    f0: float,
    df_fwhm: float,
    q: float,
    t0: float = 1.0,
    t_bg: float = 0.0,
) -> np.ndarray:
    """
    Evaluate the exact Fano transmission formula (Koshelev et al. Eq. 1, p. 3):

    .. math::

        T(\\omega) = \\frac{T_0}{1 + q^2} \\frac{(q + \\Omega)^2}{1 + \\Omega^2}
                     + T_{\\mathrm{bg}},
        \\qquad \\Omega = \\frac{2(\\omega - \\omega_0)}{\\gamma}
                       = \\frac{2(f - f_0)}{\\Delta f_{\\mathrm{FWHM}}}.

    Parameters
    ----------
    freq : float or array_like
        Frequency [Hz] (or normalized frequency detuning).
    f0 : float
        Resonance center frequency [Hz].
    df_fwhm : float
        Full-width at half-maximum linewidth ``gamma / (2*pi)`` [Hz], so that
        ``Q = f0 / df_fwhm``.
    q : float
        Dimensionless Fano asymmetry parameter (ratio of resonant and direct
        pathway amplitudes).
    t0 : float, optional
        Peak-to-dip Fano modulation depth ``T_max - T_min``. Default 1.0.
    t_bg : float, optional
        Slowly varying background offset ``T_min``. Default 0.0.

    Returns
    -------
    ndarray
        Transmission ``T(freq)`` (dimensionless).
    """
    f_arr = np.asarray(freq, dtype=float)
    omega_dim = 2.0 * (f_arr - float(f0)) / float(df_fwhm)
    return (float(t0) / (1.0 + float(q) ** 2)) * (
        (float(q) + omega_dim) ** 2 / (1.0 + omega_dim**2)
    ) + float(t_bg)


def fit_fano_spectrum(
    freqs: np.ndarray,
    T: np.ndarray,
    single_channel_unitary: bool = False,
) -> dict[str, float]:
    """
    Fit Koshelev et al. Eq. (1) to a transmission spectrum ``T(freqs)``.

    Uses the exact analytical relations between the Fano extrema ``(f_min, T_min)``
    and ``(f_max, T_max)`` to initialize non-linear least squares (`scipy.optimize.curve_fit`)
    in dimensionless coordinates ``u = (f - f_mid) / |f_max - f_min|``:

    * ``Omega_min = -q`` and ``Omega_max = +1 / q``
    * ``f_max - f_min = (Delta_f / 2) * (q + 1 / q)``

    Parameters
    ----------
    freqs : 1D ndarray
        Frequencies [Hz].
    T : 1D ndarray
        Zeroth-order transmission ``T(f)`` in ``[0, 1]``.
    single_channel_unitary : bool, optional
        If True, constrains ``T_0 = 1`` and ``T_bg = 0`` (exact for a lossless,
        single-channel sub-diffractive symmetric slab). Default False.

    Returns
    -------
    dict of str to float
        Fitted parameters ``f0_hz``, ``lambda0_nm``, ``df_fwhm_hz``, ``Q_fano``,
        ``q_fano``, ``t0``, ``t_bg``, ``t_min``, ``t_max``, ``contrast``, and
        ``rms_residual``.
    """
    f_arr = np.asarray(freqs, dtype=float)
    t_arr = np.asarray(T, dtype=float)
    idx_min = int(np.argmin(t_arr))
    idx_max = int(np.argmax(t_arr))
    f_min = float(f_arr[idx_min])
    f_max = float(f_arr[idx_max])
    t_min = float(t_arr[idx_min])
    t_max = float(t_arr[idx_max])
    contrast = t_max - t_min

    if contrast < 0.05:
        f_mid = float(np.mean(f_arr))
        return {
            "f0_hz": f_mid,
            "lambda0_nm": float(c_light / f_mid * 1e9),
            "df_fwhm_hz": 0.0,
            "Q_fano": float("inf"),
            "q_fano": float("nan"),
            "t0": float(contrast),
            "t_bg": float(t_min),
            "t_min": t_min,
            "t_max": t_max,
            "contrast": float(contrast),
            "rms_residual": 0.0,
        }

    f_mid = 0.5 * (f_min + f_max)
    df_scale = max(abs(f_max - f_min), 1e6)
    u = (f_arr - f_mid) / df_scale
    q_sign = 1.0 if f_max > f_min else -1.0

    if single_channel_unitary:

        def _model_u(u_val, u0, g_u, q_val):
            return fano_lineshape_eq1(u_val, u0, g_u, q_val, t0=1.0, t_bg=0.0)

        popt, _ = curve_fit(_model_u, u, t_arr, p0=[0.0, 0.9, q_sign * 1.15], maxfev=10000)
        u0_fit, g_u_fit, q_fit = float(popt[0]), float(abs(popt[1])), float(popt[2])
        t0_fit, t_bg_fit = 1.0, 0.0
    else:

        def _model_u5(u_val, u0, g_u, q_val, t0_val, tbg_val):
            return fano_lineshape_eq1(u_val, u0, g_u, q_val, t0=t0_val, t_bg=tbg_val)

        popt, _ = curve_fit(
            _model_u5,
            u,
            t_arr,
            p0=[0.0, 0.9, q_sign * 1.15, max(contrast, 0.5), max(t_min, 0.0)],
            maxfev=10000,
        )
        u0_fit = float(popt[0])
        g_u_fit = float(abs(popt[1]))
        q_fit = float(popt[2])
        t0_fit = float(popt[3])
        t_bg_fit = float(popt[4])

    f0_fit = f_mid + u0_fit * df_scale
    df_fit = g_u_fit * df_scale
    q_factor = f0_fit / df_fit
    t_pred = fano_lineshape_eq1(f_arr, f0_fit, df_fit, q_fit, t0=t0_fit, t_bg=t_bg_fit)
    rms = float(np.sqrt(np.mean((t_arr - t_pred) ** 2)))

    return {
        "f0_hz": float(f0_fit),
        "lambda0_nm": float(c_light / f0_fit * 1e9),
        "df_fwhm_hz": float(df_fit),
        "Q_fano": float(q_factor),
        "q_fano": float(q_fit),
        "t0": float(t0_fit),
        "t_bg": float(t_bg_fit),
        "t_min": t_min,
        "t_max": t_max,
        "contrast": float(contrast),
        "rms_residual": rms,
    }


def rcwa_fano_sweep(
    alphas: Sequence[float] = ALPHAS_DEFAULT,
    n_harmonics: int = 15,
    nx_profile: int = 4096,
) -> dict[str, Any]:
    """
    Compute exact 1D RCWA (`wakis.floquet.rcwa_1d_grating`) transmission spectra
    and Fano parameter fits across ``alphas`` (including ``alpha = 0``).

    Parameters
    ----------
    alphas : sequence of float, optional
        Asymmetry values ``alpha = (w_1 - w_2) / w_0``. Default ``ALPHAS_DEFAULT``.
    n_harmonics : int, optional
        Fourier truncation order ``M`` (``2*M + 1`` harmonics). Default 15.
    nx_profile : int, optional
        Number of real-space samples across ``p`` for Fourier transform of ``eps(x)``.
        Default 4096.

    Returns
    -------
    dict
        Broadband spectra ``T(lambda; alpha)`` for plotting the Fano fan (analog of
        Koshelev et al. Fig. 2c) and zoomed high-resolution Fano fits for each ``alpha``.
    """
    x = (np.arange(nx_profile) - 0.5 * nx_profile + 0.5) * (PERIOD_P / nx_profile)
    dx = PERIOD_P / nx_profile

    def _eps_profile(a_val: float) -> np.ndarray:
        geom = dimer_bar_geometry(alpha=a_val)
        x_lo, x_hi = x - 0.5 * dx, x + 0.5 * dx
        f1 = (
            np.clip(
                np.minimum(x_hi, geom["x1_max"]) - np.maximum(x_lo, geom["x1_min"]),
                0.0,
                dx,
            )
            / dx
        )
        f2 = (
            np.clip(
                np.minimum(x_hi, geom["x2_max"]) - np.maximum(x_lo, geom["x2_min"]),
                0.0,
                dx,
            )
            / dx
        )
        return EPS_BG + (f1 + f2) * (EPS_SI - EPS_BG)

    wls_cal = np.linspace(1.568e-6, 1.576e-6, 801)
    t0_cal = rcwa_1d_grating(
        c_light / wls_cal,
        0.0,
        PERIOD_P,
        HEIGHT_H,
        _eps_profile(0.0),
        pol="TE",
        n_harmonics=n_harmonics,
        orders=(0,),
    )["T_total"]
    t10_cal = rcwa_1d_grating(
        c_light / wls_cal,
        0.0,
        PERIOD_P,
        HEIGHT_H,
        _eps_profile(0.10),
        pol="TE",
        n_harmonics=n_harmonics,
        orders=(0,),
    )["T_total"]
    t20_cal = rcwa_1d_grating(
        c_light / wls_cal,
        0.0,
        PERIOD_P,
        HEIGHT_H,
        _eps_profile(0.20),
        pol="TE",
        n_harmonics=n_harmonics,
        orders=(0,),
    )["T_total"]
    lam10 = float(wls_cal[int(np.argmax(np.abs(t10_cal - t0_cal)))])
    lam20 = float(wls_cal[int(np.argmax(np.abs(t20_cal - t0_cal)))])
    c2 = (lam20 - lam10) / (0.20**2 - 0.10**2)
    lam_bic_rcwa = lam10 - c2 * (0.10**2)

    alphas_arr = np.asarray(alphas, dtype=float)
    f0_list, lam0_list, q_fano_list, q_param_list, rms_list = [], [], [], [], []
    local_wls_list, local_t_list, local_t_fit_list = [], [], []

    for a_val in alphas_arr:
        lam_est = lam_bic_rcwa + c2 * (a_val**2)
        q_est = 46.5 / (a_val**2)
        dl_fwhm = lam_est / q_est
        wls_loc = np.linspace(lam_est - 14.0 * dl_fwhm, lam_est + 14.0 * dl_fwhm, 241)
        freqs_loc = c_light / wls_loc
        t_loc = rcwa_1d_grating(
            freqs_loc,
            0.0,
            PERIOD_P,
            HEIGHT_H,
            _eps_profile(float(a_val)),
            pol="TE",
            n_harmonics=n_harmonics,
            orders=(0,),
        )["T_total"]
        fit = fit_fano_spectrum(freqs_loc, t_loc, single_channel_unitary=True)
        t_fit = fano_lineshape_eq1(
            freqs_loc, fit["f0_hz"], fit["df_fwhm_hz"], fit["q_fano"], t0=1.0, t_bg=0.0
        )
        f0_list.append(fit["f0_hz"])
        lam0_list.append(fit["lambda0_nm"])
        q_fano_list.append(fit["Q_fano"])
        q_param_list.append(fit["q_fano"])
        rms_list.append(fit["rms_residual"])
        local_wls_list.append(wls_loc * 1e9)
        local_t_list.append(t_loc)
        local_t_fit_list.append(t_fit)

    wls_fan = np.linspace(1.566e-6, 1.578e-6, 481)
    freqs_fan = c_light / wls_fan
    fan_alphas = (0.0, 0.04, 0.08, 0.14, 0.20)
    fan_spectra_t = []
    fan_spectra_r = []
    for a_fan in fan_alphas:
        res_fan = rcwa_1d_grating(
            freqs_fan,
            0.0,
            PERIOD_P,
            HEIGHT_H,
            _eps_profile(a_fan),
            pol="TE",
            n_harmonics=n_harmonics,
            orders=(0,),
        )
        fan_spectra_t.append(res_fan["T_total"])
        fan_spectra_r.append(res_fan["R_total"])

    q_arr = np.asarray(q_fano_list, dtype=float)
    slope_rcwa, intercept_rcwa = np.polyfit(np.log10(alphas_arr), np.log10(q_arr), 1)

    return {
        "lambda_bic_rcwa_nm": float(lam_bic_rcwa * 1e9),
        "alphas": alphas_arr,
        "f0_hz": np.asarray(f0_list, dtype=float),
        "lambda0_nm": np.asarray(lam0_list, dtype=float),
        "Q_fano": q_arr,
        "Q_alpha2": q_arr * (alphas_arr**2),
        "q_fano": np.asarray(q_param_list, dtype=float),
        "rms_residual": np.asarray(rms_list, dtype=float),
        "log_slope": float(slope_rcwa),
        "Q0_fit": float(10.0**intercept_rcwa),
        "local_wls_nm": np.asarray(local_wls_list, dtype=float),
        "local_T": np.asarray(local_t_list, dtype=float),
        "local_T_fit": np.asarray(local_t_fit_list, dtype=float),
        "fan_wls_nm": wls_fan * 1e9,
        "fan_alphas": np.asarray(fan_alphas, dtype=float),
        "fan_T": np.asarray(fan_spectra_t, dtype=float),
        "fan_R": np.asarray(fan_spectra_r, dtype=float),
    }


def run_fano_spectrum_vs_harminv(
    alpha: float = 0.48,
    nx: int = 32,
    n_cycles: float = 260.0,
    n_freqs: int = 251,
) -> dict[str, Any]:
    """
    Run a direct normal-incidence Wakis 2D FIT plane-wave transmission/reflection
    simulation using :class:`DiffractionMonitor` + :class:`ResonanceMonitor` and
    compare the Fano lineshape fit against ``harminv`` on the exact same solver.

    Also runs the symmetry-protected ``alpha = 0`` reference to verify that the
    Fano resonance collapses completely (no dip in ``T(lambda)``).

    Parameters
    ----------
    alpha : float, optional
        Asymmetry parameter ``(w_1 - w_2) / w_0``. Default ``0.48``
        (``alpha_half = 0.24``, giving ``Q ~ 195`` so the full Fano lineshape is
        resolved within ``260`` optical cycles in time-domain DFT).
    nx : int, optional
        Grid cells across ``p = 680 nm``. Default 32.
    n_cycles : float, optional
        Number of optical cycles after pulse end. Default 260.0.
    n_freqs : int, optional
        Number of DFT frequencies across ``[1520, 1570] nm``. Default 251.

    Returns
    -------
    dict
        Simulated ``R(lambda)``, ``T(lambda)``, energy conservation ``|R + T - 1|``,
        Fano Eq. (1) fit parameters (`Q_fano`, `q_fano`, `lambda0_fano_nm`),
        ``harminv`` parameters (`Q_harminv`, `lambda0_harminv_nm`), and the flat
        ``alpha = 0`` dark-mode transmission spectrum ``T_alpha0(lambda)``.
    """
    t0_wall = time.perf_counter()
    dx = PERIOD_P / int(nx)
    dz = 0.019e-6  # 19 nm -> 10 cells across h = 190 nm
    nz = 64
    lz = nz * dz
    wls = np.linspace(1.520e-6, 1.570e-6, int(n_freqs))
    freqs = c_light / wls
    f0 = float(c_light / 1.545e-6)

    def _run_one(a_val: float, cycles: float) -> dict[str, Any]:
        solver, _ = make_dimer_solver(
            alpha=a_val,
            nx=nx,
            z_max=0.5 * lz,
            uniform_z=True,
            dz_uniform=dz,
            n_pml=10,
        )
        wf = GaussianSource(frequency=f0, fwidth=0.08 * f0)
        src = CurrentSource(
            wf,
            component="y",
            center=(0.0, 0.5 * dx, -0.32e-6),
            size=(PERIOD_P, dx, 0.0),
        )
        rmon = ResonanceMonitor(
            "Ey",
            center=(-0.5 * DIMER_D, 0.5 * dx, 0.0),
            t_start=wf.end_time,
        )
        dmon_r = DiffractionMonitor(
            freqs,
            center=(0.0, 0.5 * dx, -0.22e-6),
            size=(PERIOD_P, dx, 0.0),
        )
        dmon_t = DiffractionMonitor(
            freqs,
            center=(0.0, 0.5 * dx, +0.22e-6),
            size=(PERIOD_P, dx, 0.0),
        )
        n_steps = int((wf.end_time + float(cycles) / f0) / solver.dt)
        solver.emsolve(n_steps, source=src, monitors=[rmon, dmon_r, dmon_t])

        im0 = int(np.flatnonzero(dmon_r.orders[0] == 0)[0])
        in0 = int(np.flatnonzero(dmon_r.orders[1] == 0)[0])
        p_inc = dmon_r.get_power()["up"][:, im0, in0]
        r_spec = dmon_r.get_power(incident=p_inc)["down"][:, im0, in0]
        t_spec = dmon_t.get_power(incident=p_inc)["up"][:, im0, in0]

        res_h = rmon.harminv(0.94 * f0, 1.06 * f0, max_rel_error=None)
        if len(res_h.frequency) > 0:
            idx = int(np.argmax(np.abs(res_h.amplitude)))
            f0_h = float(res_h.frequency[idx])
            q_h = float(abs(res_h.Q[idx]))
        else:
            f0_h = float("nan")
            q_h = float("inf")

        return {
            "R": np.asarray(r_spec, dtype=float),
            "T": np.asarray(t_spec, dtype=float),
            "f0_harminv_hz": f0_h,
            "lambda0_harminv_nm": float(c_light / f0_h * 1e9) if np.isfinite(f0_h) else float("nan"),
            "Q_harminv": q_h,
        }

    out_asym = _run_one(float(alpha), float(n_cycles))
    out_sym = _run_one(0.0, 80.0)
    fit_asym = fit_fano_spectrum(freqs, out_asym["T"], single_channel_unitary=False)
    fit_sym = fit_fano_spectrum(freqs, out_sym["T"], single_channel_unitary=False)
    t_fano_curve = fano_lineshape_eq1(
        freqs,
        fit_asym["f0_hz"],
        fit_asym["df_fwhm_hz"],
        fit_asym["q_fano"],
        t0=fit_asym["t0"],
        t_bg=fit_asym["t_bg"],
    )

    rel_diff_q = abs(fit_asym["Q_fano"] - out_asym["Q_harminv"]) / out_asym["Q_harminv"]

    return {
        "alpha": float(alpha),
        "alpha_half": 0.5 * float(alpha),
        "wls_nm": wls * 1e9,
        "freqs_hz": freqs,
        "R_asym": out_asym["R"],
        "T_asym": out_asym["T"],
        "T_fano_fit": t_fano_curve,
        "R_sym_alpha0": out_sym["R"],
        "T_sym_alpha0": out_sym["T"],
        "max_energy_balance_err": float(np.max(np.abs(out_asym["R"] + out_asym["T"] - 1.0))),
        "Q_harminv": out_asym["Q_harminv"],
        "lambda0_harminv_nm": out_asym["lambda0_harminv_nm"],
        "Q_fano": fit_asym["Q_fano"],
        "lambda0_fano_nm": fit_asym["lambda0_nm"],
        "q_fano": fit_asym["q_fano"],
        "t0_fano": fit_asym["t0"],
        "t_bg_fano": fit_asym["t_bg"],
        "fano_rms_residual": fit_asym["rms_residual"],
        "rel_diff_Q_fano_vs_harminv": float(rel_diff_q),
        "contrast_alpha0": fit_sym["contrast"],
        "Q_fano_alpha0": fit_sym["Q_fano"],
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


def run_oblique_floquet_study(
    thetas_deg: Sequence[float] = THETAS_OBLIQUE_DEG,
    nx: int = 32,
    quick: bool = True,
) -> dict[str, Any]:
    """
    Compute the angular BIC dispersion ``Q_rad(k_parallel) ∝ k_parallel^(-2)`` on
    the symmetric (``alpha = 0``) dimer metasurface using both:

    1. Bloch-periodic ``ResonanceMonitor.harminv`` across ``thetas_deg``, and
    2. :class:`wakis.floquet.FloquetCell` + :class:`wakis.floquet.AngleSweep`
       constant-angle resampled transmission spectra ``T(lambda, theta_inc)``,
       cross-checked against 1D RCWA (`wakis.floquet.rcwa_1d_grating`).

    Parameters
    ----------
    thetas_deg : sequence of float, optional
        Oblique incidence angles [deg]. Default ``(1, 2, 3, 5, 7, 10)``.
    nx : int, optional
        Grid cells per period ``p``. Default 32.
    quick : bool, optional
        If True, uses ``n_k = 4`` and ``run_time_cycles = 85`` in ``AngleSweep``;
        if False (`--full`), uses ``n_k = 6`` and ``run_time_cycles = 130``.

    Returns
    -------
    dict
        Angular dispersion ``Q_rad(theta)``, normalized wavenumber ``k_x * p / (2*pi)``,
        log-log slope vs ``sin(theta)``, ``AngleSweep`` resampled spectra, and RCWA
        oblique spectra.
    """
    t0_wall = time.perf_counter()
    thetas_arr = np.asarray(thetas_deg, dtype=float)
    sin_thetas = np.sin(np.deg2rad(thetas_arr))
    k0_ref = 2.0 * np.pi / LAMBDA_BIC_REF

    lam_bloch = []
    f0_bloch = []
    q_bloch = []
    for th in thetas_arr:
        kx = k0_ref * float(np.sin(np.deg2rad(th)))
        r = run_ringdown_harminv(alpha=0.0, nx=nx, kx=kx, ring_cycles=38.0)
        lam_bloch.append(r["lambda0_nm"])
        f0_bloch.append(r["f0_hz"])
        q_bloch.append(r["Q"])

    lam_bloch_arr = np.asarray(lam_bloch, dtype=float)
    f0_bloch_arr = np.asarray(f0_bloch, dtype=float)
    q_bloch_arr = np.asarray(q_bloch, dtype=float)
    kx_norm_arr = (f0_bloch_arr / c_light) * PERIOD_P * sin_thetas
    slope_ang, intercept_ang = np.polyfit(np.log10(sin_thetas), np.log10(q_bloch_arr), 1)

    dx = PERIOD_P / int(nx)
    dz = 0.019e-6
    nz = 64
    lz = nz * dz
    wls_sweep = np.linspace(1.560e-6, 1.620e-6, 121)
    freqs_sweep = c_light / wls_sweep[::-1]
    angles_sweep_deg = np.array([0.0, 6.0, 9.0, 12.0], dtype=float)
    angles_sweep_rad = np.deg2rad(angles_sweep_deg)

    cell = FloquetCell(
        Lx=PERIOD_P,
        Ly=dx,
        zmin=-0.5 * lz,
        zmax=0.5 * lz,
        nx=int(nx),
        ny=1,
        nz=nz,
        z_source=-0.32e-6,
        z_refl=-0.22e-6,
        z_tran=+0.22e-6,
        orders_x=(0,),
        orders_y=(0,),
        n_pml=10,
        material_callback=SymmetricDimerMaterialCallback(alpha=0.0),
        run_time_cycles=85.0 if quick else 130.0,
        cfl=0.95,
    )
    sweep = AngleSweep(
        cell,
        freqs_sweep,
        angles_sweep_rad,
        polarization="TE",
        n_k=4 if quick else 6,
    )
    # Because SymmetricDimerMaterialCallback is a top-level class in _helpers,
    # processes=2 works cleanly under spawn multiprocessing!
    sweep_res = sweep.run(processes=2)
    resampled = sweep_res.resample()
    t_floquet = np.asarray(resampled["T_00"][:, ::-1], dtype=float)
    r_floquet = np.asarray(resampled["R_00"][:, ::-1], dtype=float)

    nx_rcwa = 2048
    x_r = (np.arange(nx_rcwa) - 0.5 * nx_rcwa + 0.5) * (PERIOD_P / nx_rcwa)
    dx_r = PERIOD_P / nx_rcwa
    geom0 = dimer_bar_geometry(alpha=0.0)
    f1 = (
        np.clip(
            np.minimum(x_r + 0.5 * dx_r, geom0["x1_max"])
            - np.maximum(x_r - 0.5 * dx_r, geom0["x1_min"]),
            0.0,
            dx_r,
        )
        / dx_r
    )
    f2 = (
        np.clip(
            np.minimum(x_r + 0.5 * dx_r, geom0["x2_max"])
            - np.maximum(x_r - 0.5 * dx_r, geom0["x2_min"]),
            0.0,
            dx_r,
        )
        / dx_r
    )
    eps0_r = EPS_BG + (f1 + f2) * (EPS_SI - EPS_BG)

    wls_rcwa_obl = np.linspace(1.568e-6, 1.595e-6, 361)
    freqs_rcwa_obl = c_light / wls_rcwa_obl
    rcwa_obl_angles_deg = np.array([0.0, 3.0, 5.0, 7.0, 10.0], dtype=float)
    t_rcwa_obl = []
    for th_deg in rcwa_obl_angles_deg:
        res_o = rcwa_1d_grating(
            freqs_rcwa_obl,
            float(np.deg2rad(th_deg)),
            PERIOD_P,
            HEIGHT_H,
            eps0_r,
            pol="TE",
            n_harmonics=15,
            orders=(0,),
        )
        t_rcwa_obl.append(res_o["T_total"])

    return {
        "thetas_deg": thetas_arr,
        "sin_thetas": sin_thetas,
        "kx_norm": kx_norm_arr,
        "lambda0_bloch_nm": lam_bloch_arr,
        "f0_bloch_hz": f0_bloch_arr,
        "Q_bloch": q_bloch_arr,
        "Q_sin2": q_bloch_arr * (sin_thetas**2),
        "log_slope_sin_theta": float(slope_ang),
        "Q0_sin_theta": float(10.0**intercept_ang),
        "angles_sweep_deg": angles_sweep_deg,
        "wls_sweep_nm": wls_sweep * 1e9,
        "T_floquet": t_floquet,
        "R_floquet": r_floquet,
        "rcwa_obl_angles_deg": rcwa_obl_angles_deg,
        "wls_rcwa_obl_nm": wls_rcwa_obl * 1e9,
        "T_rcwa_obl": np.asarray(t_rcwa_obl, dtype=float),
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


def run_absorption_saturation_study(
    alphas: Sequence[float] = ALPHAS_DEFAULT,
    q_abs_targets: Sequence[float] = (1.0e5, 2.0e4, 5.0e3),
    nx: int = 32,
) -> dict[str, Any]:
    """
    Simulate ``Q`` saturation in the presence of intrinsic/fabrication material loss
    (Koshelev et al. Fig. 4, p. 4) using ``solver.use_conductivity = True``.

    Verifies both:
    1. ``Q_tot(alpha) = (Q_rad(alpha)^(-1) + Q_abs^(-1))^(-1)``, which saturates at
       ``Q_tot(0) = Q_abs`` as ``alpha -> 0``, and
    2. Koshelev et al.'s bare radiative damping recovery formula:
       ``gamma_rad(alpha) = gamma_tot(alpha) - gamma_tot(0)``
       (equivalently ``Q_rad_rec(alpha) = (Q_tot(alpha)^(-1) - Q_tot(0)^(-1))^(-1)``).

    Parameters
    ----------
    alphas : sequence of float, optional
        Non-zero asymmetry values ``alpha``. Default ``ALPHAS_DEFAULT``.
    q_abs_targets : sequence of float, optional
        Target non-radiative absorption quality factors ``Q_abs``.
        Default ``(1e5, 2e4, 5e3)`` (where ``Q_abs = 1e5`` is Koshelev et al. Fig. 4's
        silicon benchmark ``Re(eps)/Im(eps) = 10^5``).
    nx : int, optional
        Grid cells per period ``p``. Default 32.

    Returns
    -------
    dict
        Lossless ``Q_rad(alpha)``, electric confinement factor ``Gamma_conf``,
        simulated ``Q_tot(alpha; Q_abs)``, analytical prediction, and recovered
        bare ``Q_rad_rec(alpha)``.
    """
    t0_wall = time.perf_counter()
    alphas_arr = np.asarray(alphas, dtype=float)
    omega0 = 2.0 * np.pi * F0_BIC_REF

    r_bic_0 = run_ringdown_harminv(alpha=0.0, nx=nx, sigma=0.0)
    q_rad_sim = []
    lam_rad_sim = []
    for a_val in alphas_arr:
        r_a = run_ringdown_harminv(alpha=float(a_val), nx=nx, sigma=0.0)
        q_rad_sim.append(r_a["Q"])
        lam_rad_sim.append(r_a["lambda0_nm"])
    q_rad_arr = np.asarray(q_rad_sim, dtype=float)
    lam_rad_arr = np.asarray(lam_rad_sim, dtype=float)

    sigma_probe = omega0 * epsilon_0 * EPS_SI / 1.0e4
    r_probe = run_ringdown_harminv(alpha=0.0, nx=nx, sigma=sigma_probe)
    gamma_conf = float(1.0e4 / r_probe["Q"])

    q_abs_arr = np.asarray(q_abs_targets, dtype=float)
    q_tot_zero = np.zeros(len(q_abs_arr), dtype=float)
    q_tot_matrix = np.zeros((len(q_abs_arr), len(alphas_arr)), dtype=float)
    q_tot_pred_matrix = np.zeros_like(q_tot_matrix)
    q_rad_rec_matrix = np.zeros_like(q_tot_matrix)

    for iq, q_abs_t in enumerate(q_abs_arr):
        sigma_val = (omega0 * epsilon_0 * EPS_SI / float(q_abs_t)) * gamma_conf
        r0_lossy = run_ringdown_harminv(alpha=0.0, nx=nx, sigma=sigma_val)
        q0_meas = float(r0_lossy["Q"])
        q_tot_zero[iq] = q0_meas
        for ia, a_val in enumerate(alphas_arr):
            ra_lossy = run_ringdown_harminv(alpha=float(a_val), nx=nx, sigma=sigma_val)
            qt = float(ra_lossy["Q"])
            q_tot_matrix[iq, ia] = qt
            q_tot_pred_matrix[iq, ia] = 1.0 / (1.0 / q_rad_arr[ia] + 1.0 / q0_meas)
            q_rad_rec_matrix[iq, ia] = 1.0 / max(1.0 / qt - 1.0 / q0_meas, 1e-12)

    max_rel_err_pred = float(
        np.max(np.abs(q_tot_matrix - q_tot_pred_matrix) / q_tot_pred_matrix)
    )
    mask_rec = q_rad_arr[None, :] <= 5.0 * q_abs_arr[:, None]
    q_rad_bcast = np.broadcast_to(q_rad_arr[None, :], q_tot_matrix.shape)
    max_rel_err_rec = float(
        np.max(np.abs(q_rad_rec_matrix[mask_rec] - q_rad_bcast[mask_rec]) / q_rad_bcast[mask_rec])
    )

    return {
        "alphas": alphas_arr,
        "lambda_bic_nm": r_bic_0["lambda0_nm"],
        "Q_bic_lossless": r_bic_0["Q"],
        "lambda_rad_nm": lam_rad_arr,
        "Q_rad_lossless": q_rad_arr,
        "Q_alpha2_lossless": q_rad_arr * (alphas_arr**2),
        "gamma_confinement": gamma_conf,
        "q_abs_targets": q_abs_arr,
        "Q_tot_alpha0": q_tot_zero,
        "Q_tot_sim": q_tot_matrix,
        "Q_tot_pred": q_tot_pred_matrix,
        "Q_rad_recovered": q_rad_rec_matrix,
        "max_rel_err_Q_tot_pred": max_rel_err_pred,
        "max_rel_err_Q_rad_recovered": max_rel_err_rec,
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


def run_resolution_and_failure_study(quick: bool = True) -> dict[str, Any]:
    """
    Run the mesh convergence ladder, subpixel vs staircase quantization study,
    and CPML air-buffer proximity scan.

    Parameters
    ----------
    quick : bool, optional
        If True (`--quick`), runs ``nx in (32, 48, 64)``; if False (`--full`),
        runs ``nx in (32, 48, 64, 96)``.

    Returns
    -------
    dict
        Convergence arrays vs ``nx``, staircase quantization results across
        ``alphas``, and CPML air-buffer distance scan.
    """
    t0_wall = time.perf_counter()
    nx_list = (32, 48, 64) if quick else (32, 48, 64, 96)
    alpha_ref = 0.10

    dx_nm_list = []
    cpw_si_list = []
    cells_gap_list = []
    n_cells_list = []
    n_steps_list = []
    lam_bic_list = []
    q_bic_list = []
    lam_sub_list = []
    q_sub_list = []
    q_stair_ref_list = []

    for nx_val in nx_list:
        r0 = run_ringdown_harminv(alpha=0.0, nx=nx_val, averaging="subpixel")
        r_sub = run_ringdown_harminv(alpha=alpha_ref, nx=nx_val, averaging="subpixel")
        r_st = run_ringdown_harminv(alpha=alpha_ref, nx=nx_val, averaging="staircase")
        m = r_sub["meta"]
        dx_nm_list.append(m["dx_nm"])
        cpw_si_list.append(m["cpw_si_min"])
        cells_gap_list.append(m["cells_across_gap"])
        n_cells_list.append(m["n_cells"])
        n_steps_list.append(r_sub["n_steps"])
        lam_bic_list.append(r0["lambda0_nm"])
        q_bic_list.append(r0["Q"])
        lam_sub_list.append(r_sub["lambda0_nm"])
        q_sub_list.append(r_sub["Q"])
        q_stair_ref_list.append(r_st["Q"])

    q_stair_sweep = []
    for a_val in ALPHAS_DEFAULT:
        r_s = run_ringdown_harminv(alpha=float(a_val), nx=32, averaging="staircase")
        q_stair_sweep.append(r_s["Q"])
    q_stair_sweep_arr = np.asarray(q_stair_sweep, dtype=float)
    slope_stair, _ = np.polyfit(np.log10(ALPHAS_DEFAULT), np.log10(q_stair_sweep_arr), 1)

    z_max_scan = (0.75e-6, 0.90e-6, 1.05e-6, 1.30e-6)
    d_air_nm_scan = []
    q_pml_bic_scan = []
    q_pml_a02_scan = []
    for zm in z_max_scan:
        r_p0 = run_ringdown_harminv(alpha=0.0, nx=32, z_max=zm)
        r_p2 = run_ringdown_harminv(alpha=0.02, nx=32, z_max=zm)
        d_air_nm_scan.append(r_p0["meta"]["d_air_to_pml_nm"])
        q_pml_bic_scan.append(r_p0["Q"])
        q_pml_a02_scan.append(r_p2["Q"])

    return {
        "nx_list": np.asarray(nx_list, dtype=int),
        "dx_nm": np.asarray(dx_nm_list, dtype=float),
        "cpw_si": np.asarray(cpw_si_list, dtype=float),
        "cells_across_gap": np.asarray(cells_gap_list, dtype=float),
        "n_cells": np.asarray(n_cells_list, dtype=int),
        "n_steps": np.asarray(n_steps_list, dtype=int),
        "lambda_bic_nm": np.asarray(lam_bic_list, dtype=float),
        "Q_bic": np.asarray(q_bic_list, dtype=float),
        "lambda_a10_subpixel_nm": np.asarray(lam_sub_list, dtype=float),
        "Q_a10_subpixel": np.asarray(q_sub_list, dtype=float),
        "Q_alpha2_a10_subpixel": np.asarray(q_sub_list, dtype=float) * (alpha_ref**2),
        "Q_a10_staircase": np.asarray(q_stair_ref_list, dtype=float),
        "alphas_staircase": np.asarray(ALPHAS_DEFAULT, dtype=float),
        "Q_staircase_nx32": q_stair_sweep_arr,
        "slope_staircase_nx32": float(slope_stair),
        "d_air_to_pml_nm": np.asarray(d_air_nm_scan, dtype=float),
        "Q_pml_bic": np.asarray(q_pml_bic_scan, dtype=float),
        "Q_pml_alpha02": np.asarray(q_pml_a02_scan, dtype=float),
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


def preflight_and_run_3d_tilted_bars_task(
    thetas_deg: Sequence[float] = (0.0, 6.0, 9.0, 12.0),
    nx: int = 44,
    ny: int = 44,
    nz: int = 36,
    run_solver: bool = False,
) -> dict[str, Any]:
    """
    Preflight (with :func:`wakis.estimate.estimate` and :func:`wakis.estimate.preflight`)
    and optionally execute the 3D tilted silicon bar-pair unit cell of
    Koshelev et al. Fig. 2(a) (``p = 1320 nm``, ``2a = 660 nm``, ``2b = 220 nm``,
    ``h = 200 nm``, ``d = 660 nm``, ``alpha = sin(theta)``).

    Parameters
    ----------
    thetas_deg : sequence of float, optional
        Bar tilt angles ``theta`` [deg], where ``alpha = sin(theta)``.
        Default ``(0, 6, 9, 12)``.
    nx, ny, nz : int, optional
        3D grid dimensions ``(Nx, Ny, Nz)``. Default ``(44, 44, 36)``.
    run_solver : bool, optional
        If True (invoked via ``--task``), runs the 3D ``SolverFIT3D`` ring-down
        for each tilt angle ``theta`` and extracts ``Q_rad(theta)`` with ``harminv``.
        If False, returns the ``wakis.estimate`` preflight report and analytical/paper
        reference values immediately.

    Returns
    -------
    dict
        Preflight memory/time estimate, target machine compatibility, and
        (when ``run_solver=True``) 3D ``harminv`` resonance wavelengths and ``Q`` factors.
    """
    n_steps = 1600
    est = estimate(
        (int(nx), int(ny), int(nz)),
        steps=n_steps,
        dtype="float64",
        backend="cpu",
        cpml=True,
    )
    pf_laptop = preflight(est, "laptop", raise_error=False)
    pf_workstation = preflight(est, "n2-standard-4", raise_error=False)

    report: dict[str, Any] = {
        "geometry": KOSHELEV_PAPER_REF["fig2_geometry"],
        "grid_shape": [int(nx), int(ny), int(nz)],
        "n_cells": int(est.cells),
        "n_steps": int(n_steps),
        "peak_ram_gb": round(float(est.host_bytes / 1e9), 4),
        "est_wall_seconds_cpu": round(float(est.seconds), 2),
        "preflight_laptop_fits": len(pf_laptop) == 0,
        "preflight_workstation_fits": len(pf_workstation) == 0,
        "thetas_deg": [float(t) for t in thetas_deg],
        "alphas_sin_theta": [float(np.sin(np.deg2rad(t))) for t in thetas_deg],
        "ran_3d_solver": bool(run_solver),
    }

    if not run_solver:
        return report

    t0_wall = time.perf_counter()
    p3 = KOSHELEV_PAPER_REF["fig2_geometry"]["period_m"]
    a_semi = KOSHELEV_PAPER_REF["fig2_geometry"]["semi_axis_a_m"]
    b_semi = KOSHELEV_PAPER_REF["fig2_geometry"]["semi_axis_b_m"]
    h3 = KOSHELEV_PAPER_REF["fig2_geometry"]["height_m"]
    d3 = KOSHELEV_PAPER_REF["fig2_geometry"]["inter_bar_d_m"]
    lz3 = 1.20e-6

    x_g = np.linspace(-0.5 * p3, 0.5 * p3, int(nx) + 1)
    y_g = np.linspace(-0.5 * p3, 0.5 * p3, int(ny) + 1)
    z_g = np.linspace(-0.5 * lz3, 0.5 * lz3, int(nz) + 1)
    dx3 = p3 / int(nx)
    dy3 = p3 / int(ny)
    dz3 = lz3 / int(nz)

    sub_u = (np.arange(4) + 0.5) / 4.0 - 0.5
    su, sv = np.meshgrid(sub_u * dx3, sub_u * dy3, indexing="ij")
    xc = 0.5 * (x_g[:-1] + x_g[1:])
    yc = 0.5 * (y_g[:-1] + y_g[1:])
    zc = 0.5 * (z_g[:-1] + z_g[1:])
    X_sub = xc[:, None, None] + su[None, :, :].reshape(1, 1, 16)
    Y_sub = yc[None, :, None] + sv[None, :, :].reshape(1, 1, 16)

    f0_3d = c_light / 1.571e-6
    lam_3d_list = []
    q_3d_list = []

    for th_deg in thetas_deg:
        th = float(np.deg2rad(th_deg))
        c_t, s_t = float(np.cos(th)), float(np.sin(th))
        dx1 = X_sub + 0.5 * d3
        dy1 = np.broadcast_to(Y_sub, dx1.shape)
        x1_loc = c_t * dx1 + s_t * dy1
        y1_loc = -s_t * dx1 + c_t * dy1
        in_bar1 = (np.abs(x1_loc) <= b_semi) & (np.abs(y1_loc) <= a_semi)

        dx2 = X_sub - 0.5 * d3
        dy2 = np.broadcast_to(Y_sub, dx2.shape)
        x2_loc = c_t * dx2 - s_t * dy2
        y2_loc = s_t * dx2 + c_t * dy2
        in_bar2 = (np.abs(x2_loc) <= b_semi) & (np.abs(y2_loc) <= a_semi)

        fill_xy = np.mean(in_bar1 | in_bar2, axis=-1)

        grid3 = wakis.GridFIT3D(x=x_g, y=y_g, z=z_g, verbose=0)
        slv3 = wakis.SolverFIT3D(
            grid3,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=8,
            cfln=0.95,
            verbose=0,
        )
        z_nodes3 = slv3.grid.z[:-1]
        z_lo, z_hi = z_nodes3 - 0.5 * dz3, z_nodes3 + 0.5 * dz3
        f_z3 = np.clip(np.minimum(z_hi, 0.5 * h3) - np.maximum(z_lo, -0.5 * h3), 0.0, dz3) / dz3
        fill_3d = fill_xy[:, :, None] * f_z3[None, None, :]
        eps_3d = EPS_BG + fill_3d * (EPS_SI - EPS_BG)

        inv_eps = 1.0 / (epsilon_0 * eps_3d)
        slv3.ieps[:, :, :, "x"] = inv_eps
        slv3.ieps[:, :, :, "y"] = inv_eps
        slv3.ieps[:, :, :, "z"] = inv_eps
        slv3.update_tensors()
        slv3.step_0 = True

        wf3 = GaussianSource(frequency=f0_3d, fwidth=0.08 * f0_3d)
        src3 = CurrentSource(
            wf3,
            component="y",
            center=(-0.5 * d3, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
        )
        mon3 = ResonanceMonitor(
            "Ey",
            center=(-0.5 * d3, 0.0, 0.0),
            t_start=wf3.end_time,
        )
        slv3.emsolve(n_steps, source=src3, monitors=[mon3])
        r3 = mon3.harminv(0.88 * f0_3d, 1.12 * f0_3d, max_rel_error=None)
        if len(r3.frequency) > 0:
            idx3 = int(np.argmax(np.abs(r3.amplitude)))
            lam_3d_list.append(float(c_light / r3.frequency[idx3] * 1e9))
            q_3d_list.append(float(abs(r3.Q[idx3])))
        else:
            lam_3d_list.append(float("nan"))
            q_3d_list.append(float("inf"))

    report["lambda0_3d_nm"] = lam_3d_list
    report["Q_3d"] = q_3d_list
    report["elapsed_s"] = round(time.perf_counter() - t0_wall, 2)
    return report
