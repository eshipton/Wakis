# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Physics, k-space geometry, RCWA, and Wakis Floquet simulation helpers for
Tutorial 6.4: AR waveguide combiner — k-space map and grating order efficiencies.

Grounded in verified literature:
1. Z. Tian, X. Zhu, P. A. Surman, Z. Chen, and X. W. Sun, "An achromatic
   metasurface waveguide for augmented reality displays," Light: Science &
   Applications 14, 94 (2025), DOI: 10.1038/s41377-025-01761-w.
2. H. Wang and D. Ma, "Surface relief grating near-eye display waveguide
   design," arXiv:2307.01600 (2023), DOI: 10.48550/arXiv.2307.01600.
3. B. C. Kress and I. Chatterjee, "Waveguide combiners for mixed reality
   headsets: a nanophotonics design perspective," Nanophotonics 10(1), 41-74
   (2021), DOI: 10.1515/nanoph-2020-0410.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.constants import c, epsilon_0

import wakis
from wakis.floquet import (
    AngleSweep,
    FloquetCell,
    FloquetMode,
    FloquetPort,
    _run_single_Solver,
    floquet_s_matrix,
    grating_angles,
    rcwa_1d_grating,
    reciprocal_s_matrix,
    s_parameters_from_ports,
)
from wakis.sources import CurrentSource, GaussianSource

# ---------------------------------------------------------------------------
# Physical constants & benchmark parameters from Tian et al. (2025) and
# Wang & Ma (arXiv:2307.01600, 2023)
# ---------------------------------------------------------------------------
RGB_WAVELENGTHS_M = {
    "B": 442.0e-9,  # Blue [m], m = +6 (IMC) / -6 (OMC) in Tian et al. (2025)
    "G": 530.0e-9,  # Green [m], m = +5 (IMC) / -5 (OMC) in Tian et al. (2025)
    "R": 663.0e-9,  # Red [m], m = +4 (IMC) / -4 (OMC) in Tian et al. (2025)
}
RGB_ACHROMATIC_ORDERS = {
    "B": 6,
    "G": 5,
    "R": 4,
}
PERIOD_ACHROMATIC_M = 1900.0e-9  # Shared period Lambda = 1900 nm (Tian et al. 2025)
PERIOD_GREEN_1ST_M = 380.0e-9  # Conventional m=+1 period Lambda_0 = 1900 / 5 = 380 nm
H_SIN_M = 160.0e-9  # Si3N4 metasurface thickness = 160 nm (Tian et al. 2025)
N_WG_TIAN = 1.90  # High-index glass waveguide n_s = 1.9 (Tian et al. 2025)
N_SIN_TIAN = 2.00  # PECVD silicon nitride refractive index n_SiN = 2.0 (Tian et al. 2025)
N_AIR = 1.00

# Multi-ridge Si3N4 supercell intervals in [0, 1) synthesizing balanced m=(4, 5, 6)
# harmonics across Lambda = 1900 nm (minimum ridge/groove width >= 115 nm)
ACHROMATIC_SUPERCELL_RIDGES = (
    (0.0000, 0.0312),
    (0.1143, 0.1836),
    (0.2441, 0.3477),
    (0.4434, 0.5527),
    (0.6641, 0.8242),
    (0.9238, 1.0000),
)

# Wang & Ma (arXiv:2307.01600, Table 1) parameters:
WL_WANG_M = 525.0e-9  # Design wavelength 525 nm
N_WG_WANG = 1.74  # N-LAF2 waveguide substrate refractive index
WANG_OUTCOUPLER_DEPTHS_NM = np.array([143.0, 158.0, 179.0, 232.0, 304.0, 384.0])
WANG_OUTCOUPLER_COATINGS_NM = np.array([166.0, 166.0, 262.0, 221.0, 144.0, 205.0])
WANG_OUTCOUPLER_FILL = 0.18


def tian_uniformity(efficiencies: np.ndarray) -> float:
    """Compute diffraction efficiency uniformity ``U`` defined in Tian et al. (2025), Eq. (3).

    .. math::

        U = 1 - \\frac{\\eta_{\\max} - \\eta_{\\min}}{\\eta_{\\max} + \\eta_{\\min}}
          = \\frac{2\\,\\eta_{\\min}}{\\eta_{\\max} + \\eta_{\\min}}.

    Parameters
    ----------
    efficiencies : array_like
        Array of positive diffraction efficiencies ``eta`` (dimensionless).

    Returns
    -------
    float
        Uniformity metric ``U`` in ``[0, 1]``.
    """
    arr = np.asarray(efficiencies, dtype=float)
    valid = arr[np.isfinite(arr) & (arr >= 0.0)]
    if valid.size == 0:
        return float("nan")
    eta_min = float(np.min(valid))
    eta_max = float(np.max(valid))
    if eta_max + eta_min <= 1e-30:
        return 0.0
    return float(1.0 - (eta_max - eta_min) / (eta_max + eta_min))


def supercell_uniformity_summary(
    eff_by_channel: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute per-channel and joint-RGB uniformity ``U`` (Tian et al. 2025, Eq. 3).

    Distinguishes single-wavelength angular uniformity ``U_c`` (across FOV angles
    for a single color ``c in {'B', 'G', 'R'}``) from joint RGB+angle uniformity
    ``U_rgb`` (across all wavelengths and angles simultaneously). Note that our
    6-ridge supercell (`ACHROMATIC_SUPERCELL_RIDGES`) is a forward model and not
    a like-for-like reproduction of Tian et al.'s inverse-designed ridge profile
    (whose sub-period coordinates are not tabulated in the paper).

    Parameters
    ----------
    eff_by_channel : dict of str to ndarray
        Mapping from channel key (``'B'``, ``'G'``, ``'R'``) to 1D efficiency array [-].

    Returns
    -------
    out : dict of str to float
        Dictionary with ``U_B``, ``U_G``, ``U_R`` [-], ``U_channel_min``,
        ``U_channel_max``, and ``U_joint_rgb`` [-].
    """
    u_ch = {k: tian_uniformity(v) for k, v in eff_by_channel.items()}
    all_eff = np.concatenate([np.asarray(v, dtype=float).ravel() for v in eff_by_channel.values()])
    u_joint = tian_uniformity(all_eff)
    return {
        **{f"U_{k}": float(val) for k, val in u_ch.items()},
        "U_channel_min": float(min(u_ch.values())),
        "U_channel_max": float(max(u_ch.values())),
        "U_joint_rgb": float(u_joint),
    }


def tir_critical_angle_rad(n_glass: float, n_clad: float = 1.0) -> float:
    """Return the Total Internal Reflection (TIR) critical angle [rad] in the waveguide.

    Parameters
    ----------
    n_glass : float
        Waveguide core refractive index (must be ``>= n_clad``).
    n_clad : float, optional
        Cladding refractive index (default 1.0 for air).

    Returns
    -------
    float
        Critical angle ``theta_c = arcsin(n_clad / n_glass)`` [rad].
    """
    if n_glass < n_clad:
        raise ValueError(f"n_glass ({n_glass}) must be >= n_clad ({n_clad}).")
    return float(np.arcsin(n_clad / n_glass))


def max_symmetric_fov_deg(
    n_glass: float | np.ndarray,
    n_clad: float = 1.0,
    max_wg_angle_deg: float = 90.0,
) -> np.ndarray | float:
    """Maximum symmetric 1D field of view [deg] supported by the TIR annulus ``[n_clad, n_max]``.

    In normalized k-space ``bar_k_x = k_x / k_0``, the guided TIR band spans
    ``[n_clad, n_glass * sin(theta_wg,max)]``. Centering the input FOV cone
    ``sin(theta_in) in [-sin(FOV/2), +sin(FOV/2)]`` at the midpoint
    ``|m| * lambda / Lambda = (n_max + n_clad) / 2`` gives:

    .. math::

        \\sin(\\mathrm{FOV}_{\\max} / 2) = \\min\\!\\left(1,\\; \\frac{n_{\\max} - n_{\\mathrm{clad}}}{2}\\right).

    Parameters
    ----------
    n_glass : float or array_like
        Waveguide refractive index.
    n_clad : float, optional
        Cladding refractive index (default 1.0).
    max_wg_angle_deg : float, optional
        Maximum usable internal propagation angle [deg] inside the waveguide
        (default 90.0 deg theoretical limit; practical systems use ~75-80 deg).

    Returns
    -------
    float or ndarray
        Maximum symmetric full field of view ``FOV_max`` [deg].
    """
    n_g = np.asarray(n_glass, dtype=float)
    n_upper = n_g * np.sin(np.deg2rad(max_wg_angle_deg))
    half_span = np.clip(0.5 * (n_upper - n_clad), 0.0, 1.0)
    fov = 2.0 * np.rad2deg(np.arcsin(half_span))
    return float(fov) if np.ndim(n_glass) == 0 else fov


def kspace_combiner_map(
    fov_x_deg: float = 36.0,
    fov_y_deg: float = 24.0,
    wavelength: float = 530.0e-9,
    period_ic: float = 380.0e-9,
    n_glass: float = N_WG_TIAN,
    n_grid: int = 41,
) -> dict[str, Any]:
    """Compute the 2D normalized k-space trajectories across an IC -> EPE -> OC combiner.

    In normalized direction-cosine coordinates ``(bar_k_x, bar_k_y) = (k_x/k_0, k_y/k_0)``:
    - Input FOV box in air: ``bar_k_x = sin(theta_x) cos(theta_y)``, ``bar_k_y = sin(theta_y)``.
    - In-Coupler (IC) grating vector along ``+x``: ``bar_G_IC = (+lambda / Lambda_IC, 0)``.
    - Eye-Pupil Expander / Fold (EPE) grating vector at ``-45 deg``:
      ``bar_G_EPE = (-lambda / Lambda_IC, +lambda / Lambda_IC)``, which folds the
      guided beam from ``+x`` to ``+y`` while preserving ``|bar_k|``.
    - Out-Coupler (OC) grating vector along ``-y``: ``bar_G_OC = (0, -lambda / Lambda_IC)``.
    - Vector closure: ``bar_G_IC + bar_G_EPE + bar_G_OC = (0, 0)``, guaranteeing
      ``bar_k_out == bar_k_in`` for every ray in the field of view.
    """
    tx = np.deg2rad(0.5 * fov_x_deg)
    ty = np.deg2rad(0.5 * fov_y_deg)
    s = np.linspace(-1.0, 1.0, n_grid)
    th_x = np.concatenate([s * tx, np.full_like(s, tx), -s * tx, np.full_like(s, -tx)])
    th_y = np.concatenate([np.full_like(s, -ty), s * ty, np.full_like(s, ty), -s * ty])

    kx_in = np.sin(th_x) * np.cos(th_y)
    ky_in = np.sin(th_y)

    g_mag = float(wavelength / period_ic)
    g_ic = np.array([g_mag, 0.0])
    g_epe = np.array([-g_mag, +g_mag])
    g_oc = np.array([0.0, -g_mag])

    k_in = np.stack([kx_in, ky_in], axis=-1)
    k_ic = k_in + g_ic
    k_epe = k_ic + g_epe
    k_oc = k_epe + g_oc

    mag_ic = np.linalg.norm(k_ic, axis=-1)
    mag_epe = np.linalg.norm(k_epe, axis=-1)
    guided_mask = (
        (mag_ic >= 1.0)
        & (mag_ic <= n_glass)
        & (mag_epe >= 1.0)
        & (mag_epe <= n_glass)
    )

    return {
        "k_in": k_in,
        "k_ic": k_ic,
        "k_epe": k_epe,
        "k_oc": k_oc,
        "g_ic": g_ic,
        "g_epe": g_epe,
        "g_oc": g_oc,
        "vector_closure_residual": float(np.linalg.norm(g_ic + g_epe + g_oc)),
        "output_input_max_diff": float(np.max(np.abs(k_oc - k_in))),
        "guided_fraction": float(np.mean(guided_mask)),
        "n_glass": float(n_glass),
    }


def rgb_kspace_fov_comparison(
    n_glass: float = N_WG_TIAN,
    period_conv: float = PERIOD_GREEN_1ST_M,
    period_ach: float = PERIOD_ACHROMATIC_M,
    theta_in_deg: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare conventional ``m=+1`` vs Tian et al. (2025) ``(4, 5, 6)`` achromatic RGB k-space."""
    if theta_in_deg is None:
        theta_in_deg = np.linspace(-30.0, 30.0, 241)
    th_rad = np.deg2rad(theta_in_deg)
    sin_in = np.sin(th_rad)

    conv_kx = {}
    ach_kx = {}
    conv_bounds = {}
    ach_bounds = {}

    for col, wl in RGB_WAVELENGTHS_M.items():
        m_ach = RGB_ACHROMATIC_ORDERS[col]
        g_conv = wl / period_conv
        g_ach = (m_ach * wl) / period_ach

        conv_kx[col] = sin_in + g_conv
        ach_kx[col] = sin_in + g_ach

        s_min_c = max(-1.0, 1.0 - g_conv)
        s_max_c = min(+1.0, n_glass - g_conv)
        conv_bounds[col] = (
            float(np.rad2deg(np.arcsin(s_min_c))),
            float(np.rad2deg(np.arcsin(s_max_c))),
        )
        s_min_a = max(-1.0, 1.0 - g_ach)
        s_max_a = min(+1.0, n_glass - g_ach)
        ach_bounds[col] = (
            float(np.rad2deg(np.arcsin(s_min_a))),
            float(np.rad2deg(np.arcsin(s_max_a))),
        )

    conv_min = max(b[0] for b in conv_bounds.values())
    conv_max = min(b[1] for b in conv_bounds.values())
    ach_min = max(b[0] for b in ach_bounds.values())
    ach_max = min(b[1] for b in ach_bounds.values())

    return {
        "theta_in_deg": theta_in_deg,
        "conv_kx": conv_kx,
        "ach_kx": ach_kx,
        "conv_bounds_deg": conv_bounds,
        "ach_bounds_deg": ach_bounds,
        "conv_common_fov_deg": float(max(0.0, conv_max - conv_min)),
        "ach_common_fov_deg": float(max(0.0, ach_max - ach_min)),
        "ach_momentum_shifts": {
            c_key: float(
                RGB_ACHROMATIC_ORDERS[c_key] * RGB_WAVELENGTHS_M[c_key] / period_ach
            )
            for c_key in ("B", "G", "R")
        },
    }


class BinaryGratingCallback:
    """Picklable subpixel-smoothed ``material_callback(solver)`` for 1D binary/multi-ridge gratings."""

    def __init__(
        self,
        period: float,
        height: float,
        fill_factor: float,
        eps_ridge: float,
        eps_groove: float,
        eps_below: float,
        eps_above: float,
        ridges: Sequence[tuple[float, float]] | None = None,
        n_sub: int = 16,
    ):
        self.period = float(period)
        self.height = float(height)
        self.fill_factor = float(fill_factor)
        self.eps_ridge = float(eps_ridge)
        self.eps_groove = float(eps_groove)
        self.eps_below = float(eps_below)
        self.eps_above = float(eps_above)
        self.n_sub = int(n_sub)
        if ridges is None:
            half_f = 0.5 * self.fill_factor
            self.intervals = [(0.0, half_f), (1.0 - half_f, 1.0)]
        else:
            self.intervals = [(float(a), float(b)) for a, b in ridges]

    def _in_ridge(self, u: np.ndarray) -> np.ndarray:
        mask = np.zeros(u.shape, dtype=bool)
        for u0, u1 in self.intervals:
            mask |= (u >= u0) & (u < u1)
        return mask

    def __call__(self, solver) -> None:
        grid = solver.grid
        nodes = (grid.x, grid.y, grid.z)
        sub = (np.arange(self.n_sub) + 0.5) / self.n_sub - 0.5
        ox, oz = np.meshgrid(sub, sub, indexing="ij")
        ox, oz = ox.ravel(), oz.ravel()

        for ci, comp in enumerate("xyz"):
            pos = [
                (n[:-1] + n[1:]) / 2 if d == ci else n[:-1]
                for d, n in enumerate(nodes)
            ]
            dx_c = (
                (grid.x[1:] - grid.x[:-1])
                if ci == 0
                else np.full(grid.Nx, float(grid.dx[0]))
            )
            dz_c = (
                (grid.z[1:] - grid.z[:-1])
                if ci == 2
                else np.full(grid.Nz, float(grid.dz[0]))
            )
            X, _, Z = np.meshgrid(*pos, indexing="ij")
            DX, _, DZ = np.meshgrid(dx_c, [float(grid.dy[0])], dz_c, indexing="ij")

            xs = X[..., None] + DX[..., None] * ox
            zs = Z[..., None] + DZ[..., None] * oz
            u = np.mod(xs / self.period, 1.0)

            in_above = zs >= self.height
            in_below = zs < 0.0
            in_gr = (~in_above) & (~in_below)
            in_r = in_gr & self._in_ridge(u)

            eps_s = np.where(
                in_above,
                self.eps_above,
                np.where(
                    in_below,
                    self.eps_below,
                    np.where(in_r, self.eps_ridge, self.eps_groove),
                ),
            )
            eps_arith = np.mean(eps_s, axis=-1)
            inv_eps_harm = np.mean(1.0 / eps_s, axis=-1)

            is_horiz = (np.abs(Z) < DZ) | (np.abs(Z - self.height) < DZ)
            if ci == 0:
                inv_eps = np.where(is_horiz, 1.0 / eps_arith, inv_eps_harm)
            elif ci == 2:
                inv_eps = np.where(is_horiz, inv_eps_harm, 1.0 / eps_arith)
            else:
                inv_eps = 1.0 / eps_arith

            solver.ieps[:, :, :, comp] = inv_eps / epsilon_0


def make_binary_grating_callback(
    period: float,
    height: float,
    fill_factor: float,
    eps_ridge: float,
    eps_groove: float,
    eps_below: float,
    eps_above: float,
    ridges: Sequence[tuple[float, float]] | None = None,
    n_sub: int = 16,
) -> BinaryGratingCallback:
    """Build a picklable subpixel-smoothed ``material_callback(solver)`` for a 1D grating."""
    return BinaryGratingCallback(
        period=period,
        height=height,
        fill_factor=fill_factor,
        eps_ridge=eps_ridge,
        eps_groove=eps_groove,
        eps_below=eps_below,
        eps_above=eps_above,
        ridges=ridges,
        n_sub=n_sub,
    )


def build_grating_eps_profile(
    n_x: int = 1024,
    fill_factor: float = 0.50,
    eps_ridge: float = N_SIN_TIAN**2,
    eps_groove: float = 1.0,
    ridges: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Construct the 1D permittivity profile ``eps(x)`` across ``[0, period)`` for RCWA."""
    u = (np.arange(n_x) + 0.5) / n_x
    if ridges is None:
        half_f = 0.5 * float(fill_factor)
        intervals = [(0.0, half_f), (1.0 - half_f, 1.0)]
    else:
        intervals = [(float(a), float(b)) for a, b in ridges]
    mask = np.zeros(n_x, dtype=bool)
    for u0, u1 in intervals:
        mask |= (u >= u0) & (u < u1)
    eps_prof = np.where(mask, float(eps_ridge), float(eps_groove))
    return eps_prof[::-1].copy()


def build_coupler_floquet_cell(
    period: float,
    height: float,
    wavelength_min: float,
    fill_factor: float = 0.50,
    eps_ridge: float = N_SIN_TIAN**2,
    eps_groove: float = 1.0,
    eps_below: float = 1.0,
    eps_above: float = N_WG_TIAN**2,
    orders_x: Sequence[int] = (-2, -1, 0, 1, 2),
    cpw: int = 24,
    min_cells_per_feature: int = 4,
    n_pml: int = 14,
    run_time_cycles: float = 28.0,
    ridges: Sequence[tuple[float, float]] | None = None,
) -> tuple[FloquetCell, dict[str, float]]:
    """Build a resolved 2D :class:`~wakis.floquet.FloquetCell` and mesh metrics dictionary."""
    n_max = float(np.sqrt(max(eps_ridge, eps_groove, eps_below, eps_above)))
    lam_min_mat = float(wavelength_min) / n_max
    dx_cpw = lam_min_mat / float(cpw)

    if ridges is None:
        min_w_x = float(period) * min(fill_factor, 1.0 - fill_factor)
    else:
        # Merge wrap-around interval if present at 0 and 1
        r_list = [(float(a), float(b)) for a, b in ridges]
        widths = []
        wrap_w = 0.0
        for a, b in r_list:
            if a <= 1e-6 or b >= 1.0 - 1e-6:
                wrap_w += (b - a) * float(period)
            else:
                widths.append((b - a) * float(period))
        if wrap_w > 0:
            widths.append(wrap_w)
        min_w_x = min(widths) if widths else float(period)
    min_feat = min(min_w_x, float(height))
    dx_feat = min_feat / float(max(min_cells_per_feature, 1))

    dx_target = min(dx_cpw, dx_feat)
    nx = max(16, int(np.ceil(float(period) / dx_target)))
    dx = float(period) / nx

    dz_target = min(dx_cpw, float(height) / float(max(min_cells_per_feature, 4)))
    n_grating_z = max(min_cells_per_feature, int(np.ceil(float(height) / dz_target)))
    dz = float(height) / n_grating_z

    pad_below = max(1.15 * float(wavelength_min), 24 * dz)
    pad_above = max(1.15 * float(wavelength_min), 24 * dz)
    n_below = int(np.ceil(pad_below / dz))
    n_above = int(np.ceil(pad_above / dz))
    zmin = -n_below * dz
    zmax = float(height) + n_above * dz
    nz = n_below + n_grating_z + n_above

    z_source = -0.62 * pad_below
    z_refl = -0.32 * pad_below
    z_tran = float(height) + 0.36 * pad_above

    cb = make_binary_grating_callback(
        period=period,
        height=height,
        fill_factor=fill_factor,
        eps_ridge=eps_ridge,
        eps_groove=eps_groove,
        eps_below=eps_below,
        eps_above=eps_above,
        ridges=ridges,
    )
    cell = FloquetCell(
        Lx=float(period),
        Ly=dz,
        zmin=zmin,
        zmax=zmax,
        nx=nx,
        ny=1,
        nz=nz,
        z_source=z_source,
        z_refl=z_refl,
        z_tran=z_tran,
        z_ref_below=0.0,
        z_ref_above=float(height),
        eps_below=eps_below,
        eps_above=eps_above,
        orders_x=tuple(orders_x),
        orders_y=(0,),
        n_pml=n_pml,
        material_callback=cb,
        run_time_cycles=run_time_cycles,
    )
    mesh_info = {
        "nx": int(nx),
        "nz": int(nz),
        "n_cells_2d": int(nx * nz),
        "dx_nm": float(dx * 1e9),
        "dz_nm": float(dz * 1e9),
        "n_max": n_max,
        "cpw_x": float(lam_min_mat / dx),
        "cpw_z": float(lam_min_mat / dz),
        "min_feature_nm": float(min_feat * 1e9),
        "cells_per_min_feature_x": float(min_w_x / dx),
        "cells_per_grating_height_z": float(height / dz),
    }
    return cell, mesh_info


def simulate_eyebox_pupil_expansion(
    r0_values: np.ndarray,
    t_out_values: np.ndarray,
    incouple_eff: float = 0.78,
) -> dict[str, Any]:
    """Compute extracted pupil intensity across ``N`` TIR bounces in the out-coupler."""
    r0 = np.asarray(r0_values, dtype=float)
    tout = np.asarray(t_out_values, dtype=float)
    if r0.shape != tout.shape:
        raise ValueError("r0_values and t_out_values must have the same shape.")

    n_bounces = r0.size
    extracted = np.zeros(n_bounces, dtype=float)
    guided_rem = np.zeros(n_bounces + 1, dtype=float)
    guided_rem[0] = float(incouple_eff)

    for k in range(n_bounces):
        extracted[k] = guided_rem[k] * tout[k]
        guided_rem[k + 1] = guided_rem[k] * r0[k]

    u_tian = tian_uniformity(extracted)
    mean_ext = float(np.mean(extracted))
    rel_err = (
        float((np.max(extracted) - np.min(extracted)) / (2.0 * mean_ext))
        if mean_ext > 0
        else float("nan")
    )
    return {
        "extracted": extracted,
        "guided_remaining": guided_rem,
        "total_extracted": float(np.sum(extracted)),
        "mean_extracted": mean_ext,
        "uniformity_tian": u_tian,
        "uniformity_error_wang": rel_err,
    }


def capture_coupler_harmonic_field(
    cell: FloquetCell,
    frequency: float,
    theta_inc_rad: float,
    polarization: str = "TE",
) -> dict[str, Any]:
    """Run a 2D Bloch FIT simulation and extract the complex steady-state field phasor."""
    grid = wakis.GridFIT3D(
        -0.5 * cell.Lx,
        0.5 * cell.Lx,
        -0.5 * cell.Ly,
        0.5 * cell.Ly,
        cell.zmin,
        cell.zmax,
        cell.nx,
        cell.ny,
        cell.nz,
        verbose=0,
    )
    k0 = 2.0 * np.pi * float(frequency) / c
    kx = k0 * cell.n_inc * float(np.sin(theta_inc_rad))
    k_bloch = (-kx, 0.0, 0.0)

    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["bloch", "bloch", "cpml"],
        bc_high=["bloch", "bloch", "cpml"],
        n_pml=cell.n_pml,
        k_bloch=k_bloch,
        dtype=np.complex128,
        cfln=cell.cfl,
        verbose=0,
    )
    z_split = 0.5 * (cell.z_refl + cell.z_tran)
    iz_split = int(np.searchsorted(grid.z[:-1], z_split))
    for comp in ("x", "y", "z"):
        solver.ieps[:, :, :iz_split, comp] = 1.0 / (epsilon_0 * cell.eps_below)
        solver.ieps[:, :, iz_split:, comp] = 1.0 / (epsilon_0 * cell.eps_above)
    if cell.material_callback is not None:
        cell.material_callback(solver)

    comp_src = "y" if polarization.upper() in ("TE", "S") else "x"
    src_prof = GaussianSource(frequency=float(frequency), fwidth=0.12 * float(frequency))
    sheet = CurrentSource(
        src_prof,
        component=comp_src,
        center=(0.0, 0.0, float(cell.z_source)),
        size=(float(cell.Lx), float(cell.Ly), 0.0),
        amp_func=lambda x, y, z: np.exp(-1j * kx * x),
    )

    t_total = float(src_prof.end_time) + 14.0 / float(frequency)
    n_steps = int(np.ceil(t_total / solver.dt))
    n_dft_steps = max(40, int(np.round(4.0 / (float(frequency) * solver.dt))))
    start_dft = max(0, n_steps - n_dft_steps)

    phasor = np.zeros((grid.Nx, grid.Nz), dtype=np.complex128)
    omega = 2.0 * np.pi * float(frequency)
    for step in range(n_steps):
        t = step * solver.dt
        sheet.update(solver, t)
        solver.one_step()
        if step >= start_dft:
            field_2d = np.asarray(solver.E[:, 0, :, comp_src], dtype=np.complex128)
            phasor += field_2d * np.exp(-1j * omega * t)
    phasor /= float(max(1, n_steps - start_dft))

    ieps_y = np.asarray(solver.ieps[:, 0, :, "y"], dtype=float)
    eps_map = 1.0 / np.maximum(ieps_y * epsilon_0, 1e-30)

    return {
        "x": np.asarray(grid.x[:-1], dtype=float),
        "z": np.asarray(grid.z[:-1], dtype=float),
        "x_edges": np.asarray(grid.x, dtype=float),
        "z_edges": np.asarray(grid.z, dtype=float),
        "field_phasor": phasor,
        "eps_map": eps_map,
        "kx": float(kx),
        "frequency": float(frequency),
        "theta_inc_rad": float(theta_inc_rad),
    }


def run_task_0_rgb_incoupler(
    quick: bool = True,
    processes: int = 2,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Task 0: k-space map + Wakis vs RCWA RGB in-coupler angular sweeps."""
    t0 = time.perf_counter()
    cpw = 21 if quick else 32
    n_k = 5 if quick else 9
    rtc = 22.0 if quick else 28.0
    angles_pos_deg = (
        np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        if quick
        else np.linspace(0.0, 20.0, 9)
    )
    angles_pos_rad = np.deg2rad(angles_pos_deg)
    # Full symmetric FOV [-20, +20] deg built from m=+1 (theta >= 0) and m=-1 (theta <= 0)
    angles_sym_deg = np.concatenate([-angles_pos_deg[:0:-1], angles_pos_deg])

    kmap = kspace_combiner_map()
    rgb_fov = rgb_kspace_fov_comparison()

    eps_prof = build_grating_eps_profile(
        1024, fill_factor=0.50, eps_ridge=N_SIN_TIAN**2, eps_groove=1.0
    )
    orders_ic = (-2, -1, 0, 1, 2)
    im_m1 = orders_ic.index(-1)
    im_0 = orders_ic.index(0)
    im_p1 = orders_ic.index(+1)

    npz_data: dict[str, np.ndarray] = {
        "angles_pos_deg": angles_pos_deg,
        "angles_sym_deg": angles_sym_deg,
        "k_in": kmap["k_in"],
        "k_ic": kmap["k_ic"],
        "k_epe": kmap["k_epe"],
        "k_oc": kmap["k_oc"],
        "theta_fov_scan_deg": rgb_fov["theta_in_deg"],
    }
    for c_key in ("B", "G", "R"):
        npz_data[f"conv_kx_{c_key}"] = rgb_fov["conv_kx"][c_key]
        npz_data[f"ach_kx_{c_key}"] = rgb_fov["ach_kx"][c_key]

    rgb_metrics: dict[str, Any] = {}
    for col, wl in RGB_WAVELENGTHS_M.items():
        # Channel-matched period Lambda_c = wl / (530 / 380) so lambda_c / Lambda_c = 1.3947
        period_c = wl * (PERIOD_GREEN_1ST_M / RGB_WAVELENGTHS_M["G"])
        f_arr = np.array([c / wl])
        cell, minfo = build_coupler_floquet_cell(
            period=period_c,
            height=H_SIN_M,
            wavelength_min=wl,
            fill_factor=0.50,
            eps_ridge=N_SIN_TIAN**2,
            eps_groove=1.0,
            eps_below=1.0,
            eps_above=N_WG_TIAN**2,
            orders_x=orders_ic,
            cpw=cpw,
            run_time_cycles=rtc,
        )
        sweep = AngleSweep(
            cell, f_arr, angles_pos_rad, polarization="TE", n_k=n_k
        )
        res = sweep.run(processes=processes)
        out = res.resample()

        # RCWA reference at each positive angle
        rcwa_t = np.zeros((angles_pos_deg.size, len(orders_ic)))
        rcwa_r = np.zeros((angles_pos_deg.size, len(orders_ic)))
        for ia, th_r in enumerate(angles_pos_rad):
            rc = rcwa_1d_grating(
                f_arr,
                th_r,
                period_c,
                H_SIN_M,
                eps_prof,
                eps_below=1.0,
                eps_above=N_WG_TIAN**2,
                pol="TE",
                orders=orders_ic,
            )
            rcwa_t[ia] = rc["T_orders"][0]
            rcwa_r[ia] = rc["R_orders"][0]

        wakis_t = out["T_order"][:, 0, :, 0]
        wakis_r = out["R_order"][:, 0, :, 0]
        energy_tot = out["R_total"][:, 0] + out["T_total"][:, 0]

        # By mirror symmetry of the binary tooth: T_{+1}(-theta) = T_{-1}(+theta)
        # Guided order across [-20, +20] deg: for sin(theta) + lambda/Lambda in [1, n_s],
        # at theta >= 0 order m=-1 has |k_x|/k_0 = lambda/Lambda - sin(theta) in [1.05, 1.395]
        # and at theta <= 0 order m=-1 has |k_x|/k_0 = lambda/Lambda + sin(|theta|) in [1.395, 1.74]
        w_sym_guided = np.concatenate([wakis_t[:0:-1, im_p1], wakis_t[:, im_m1]])
        r_sym_guided = np.concatenate([rcwa_t[:0:-1, im_p1], rcwa_t[:, im_m1]])
        w_sym_t0 = np.concatenate([wakis_t[:0:-1, im_0], wakis_t[:, im_0]])
        r_sym_t0 = np.concatenate([rcwa_t[:0:-1, im_0], rcwa_t[:, im_0]])

        max_err = float(np.nanmax(np.abs(w_sym_guided - r_sym_guided)))
        energy_err = float(np.nanmax(np.abs(energy_tot - 1.0)))
        u_val = tian_uniformity(w_sym_guided)

        npz_data[f"wakis_guided_{col}"] = w_sym_guided
        npz_data[f"rcwa_guided_{col}"] = r_sym_guided
        npz_data[f"wakis_t0_{col}"] = w_sym_t0
        npz_data[f"rcwa_t0_{col}"] = r_sym_t0
        npz_data[f"energy_tot_{col}"] = energy_tot

        rgb_metrics[col] = {
            "wavelength_nm": float(wl * 1e9),
            "period_nm": float(period_c * 1e9),
            "mesh": minfo,
            "mean_guided_eff": float(np.mean(w_sym_guided)),
            "uniformity_tian": u_val,
            "max_abs_err_vs_rcwa": max_err,
            "max_energy_residual": energy_err,
        }

    # Dense 2D RCWA map over (theta_in in [-22, 22] deg, lambda in [430, 680] nm) for Figure 3a
    th_map_deg = np.linspace(-22.0, 22.0, 23 if quick else 45)
    wl_map_nm = np.linspace(430.0, 680.0, 21 if quick else 41)
    eff_map_conv = np.zeros((wl_map_nm.size, th_map_deg.size))
    for iw, wl_nm in enumerate(wl_map_nm):
        f_w = np.array([c / (wl_nm * 1e-9)])
        for it, th_d in enumerate(th_map_deg):
            rc_m = rcwa_1d_grating(
                f_w,
                np.deg2rad(th_d),
                PERIOD_GREEN_1ST_M,
                H_SIN_M,
                eps_prof,
                eps_below=1.0,
                eps_above=N_WG_TIAN**2,
                pol="TE",
                orders=(-1,),
            )
            kx_norm = abs(np.sin(np.deg2rad(th_d)) - (wl_nm * 1e-9) / PERIOD_GREEN_1ST_M)
            if 1.0 <= kx_norm <= N_WG_TIAN:
                eff_map_conv[iw, it] = float(rc_m["T_orders"][0, 0])
            else:
                eff_map_conv[iw, it] = np.nan

    npz_data["th_map_deg"] = th_map_deg
    npz_data["wl_map_nm"] = wl_map_nm
    npz_data["eff_map_conv"] = eff_map_conv

    summary = {
        "index": 0,
        "name": "kspace_and_rgb_incoupler_sweep",
        "cpw": cpw,
        "kspace_vector_closure": kmap["vector_closure_residual"],
        "conv_common_fov_deg": rgb_fov["conv_common_fov_deg"],
        "ach_common_fov_deg": rgb_fov["ach_common_fov_deg"],
        "rgb_channels": rgb_metrics,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    return summary, npz_data


def run_task_1_achromatic_supercell(
    quick: bool = True,
    processes: int = 2,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Task 1: Tian et al. (2025) Lambda = 1900 nm achromatic (4, 5, 6) supercell."""
    t0 = time.perf_counter()
    cpw = 20 if quick else 28
    n_k = 3 if quick else 7
    rtc = 20.0 if quick else 28.0
    angles_deg = np.array([0.0, 6.0, 12.0]) if quick else np.linspace(0.0, 16.0, 9)
    angles_rad = np.deg2rad(angles_deg)

    eps_prof = build_grating_eps_profile(
        1024,
        eps_ridge=N_SIN_TIAN**2,
        eps_groove=1.0,
        ridges=ACHROMATIC_SUPERCELL_RIDGES,
    )
    orders_super = (-6, -5, -4, 0, 4, 5, 6)

    npz_data: dict[str, np.ndarray] = {"angles_deg": angles_deg}
    supercell_metrics: dict[str, Any] = {}

    for col in ("B", "G", "R"):
        wl = RGB_WAVELENGTHS_M[col]
        m_target = RGB_ACHROMATIC_ORDERS[col]
        f_arr = np.array([c / wl])

        cell, minfo = build_coupler_floquet_cell(
            period=PERIOD_ACHROMATIC_M,
            height=H_SIN_M,
            wavelength_min=wl,
            eps_ridge=N_SIN_TIAN**2,
            eps_groove=1.0,
            eps_below=1.0,
            eps_above=N_WG_TIAN**2,
            orders_x=orders_super,
            cpw=cpw,
            run_time_cycles=rtc,
            ridges=ACHROMATIC_SUPERCELL_RIDGES,
        )
        sweep = AngleSweep(
            cell, f_arr, angles_rad, polarization="TE", n_k=n_k
        )
        res = sweep.run(processes=processes)
        out = res.resample()

        im_pos = orders_super.index(+m_target)
        im_neg = orders_super.index(-m_target)
        w_pos = out["T_order"][:, 0, im_pos, 0]
        w_neg = out["T_order"][:, 0, im_neg, 0]
        e_tot = out["R_total"][:, 0] + out["T_total"][:, 0]

        r_pos = np.zeros_like(angles_deg)
        r_neg = np.zeros_like(angles_deg)
        for ia, th_r in enumerate(angles_rad):
            rc = rcwa_1d_grating(
                f_arr,
                th_r,
                PERIOD_ACHROMATIC_M,
                H_SIN_M,
                eps_prof,
                eps_below=1.0,
                eps_above=N_WG_TIAN**2,
                pol="TE",
                n_harmonics=20,
                orders=(-m_target, +m_target),
            )
            r_neg[ia] = rc["T_orders"][0, 0]
            r_pos[ia] = rc["T_orders"][0, 1]

        # Guided diffraction angle inside n_s = 1.9 waveguide for order -m_target
        th_wg_deg = np.rad2deg(
            grating_angles(
                f_arr[0],
                angles_rad,
                PERIOD_ACHROMATIC_M,
                [-m_target],
                n_inc=1.0,
                n_out=N_WG_TIAN,
            )[:, 0]
        )

        npz_data[f"wakis_t_m_{col}"] = w_neg
        npz_data[f"rcwa_t_m_{col}"] = r_neg
        npz_data[f"wakis_t_p_{col}"] = w_pos
        npz_data[f"rcwa_t_p_{col}"] = r_pos
        npz_data[f"theta_wg_deg_{col}"] = th_wg_deg
        npz_data[f"energy_tot_{col}"] = e_tot

        supercell_metrics[col] = {
            "order_m": m_target,
            "wavelength_nm": float(wl * 1e9),
            "mesh": minfo,
            "mean_eff_wakis": float(np.mean(w_neg)),
            "mean_eff_rcwa": float(np.mean(r_neg)),
            "uniformity_tian_wakis": tian_uniformity(w_neg),
            "uniformity_tian_rcwa": tian_uniformity(r_neg),
            "max_abs_err_vs_rcwa": float(np.max(np.abs(w_neg - r_neg))),
            "max_energy_residual": float(np.max(np.abs(e_tot - 1.0))),
            "theta_wg_deg_at_0": float(abs(th_wg_deg[0])),
        }

    # Check angular dispersion mismatch across R, G, B in the waveguide
    wg_angles_0 = [supercell_metrics[c_k]["theta_wg_deg_at_0"] for c_k in ("B", "G", "R")]
    angular_spread_deg = float(max(wg_angles_0) - min(wg_angles_0))
    u_wakis_all = supercell_uniformity_summary(
        {c_k: npz_data[f"wakis_t_m_{c_k}"] for c_k in ("B", "G", "R")}
    )
    u_rcwa_all = supercell_uniformity_summary(
        {c_k: npz_data[f"rcwa_t_m_{c_k}"] for c_k in ("B", "G", "R")}
    )

    summary = {
        "index": 1,
        "name": "achromatic_supercell_incoupler",
        "cpw": cpw,
        "period_nm": float(PERIOD_ACHROMATIC_M * 1e9),
        "rgb_angular_spread_deg": angular_spread_deg,
        "joint_rgb_uniformity_wakis": u_wakis_all["U_joint_rgb"],
        "joint_rgb_uniformity_rcwa": u_rcwa_all["U_joint_rgb"],
        "channels": supercell_metrics,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    return summary, npz_data


def run_task_2_outcoupler_and_eyebox(
    quick: bool = True,
    processes: int = 2,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Task 2: Out-coupler TIR sweep, Floquet S-matrix reciprocity, and eye-box uniformity."""
    t0 = time.perf_counter()
    cpw = 22 if quick else 32
    n_k = 6 if quick else 9
    wl_g = RGB_WAVELENGTHS_M["G"]
    f_g = np.array([c / wl_g])

    # Out-coupler parameters: period = 380 nm, height = 140 nm, fill = 0.22
    # Illuminated from inside the glass waveguide (eps_below = n_wg^2, eps_above = 1.0)
    # at guided TIR bounce angles theta_wg in [35, 55] deg (all > theta_c = 31.76 deg)
    period_oc = PERIOD_GREEN_1ST_M
    h_oc = 140.0e-9
    fill_oc = 0.22
    angles_wg_deg = (
        np.array([35.0, 40.0, 45.0, 50.0, 55.0])
        if quick
        else np.linspace(35.0, 55.0, 9)
    )
    angles_wg_rad = np.deg2rad(angles_wg_deg)
    orders_oc = (-2, -1, 0, 1)

    cell_oc, minfo_oc = build_coupler_floquet_cell(
        period=period_oc,
        height=h_oc,
        wavelength_min=wl_g,
        fill_factor=fill_oc,
        eps_ridge=N_SIN_TIAN**2,
        eps_groove=1.0,
        eps_below=N_WG_TIAN**2,
        eps_above=1.0,
        orders_x=orders_oc,
        cpw=cpw,
    )
    sweep_oc = AngleSweep(
        cell_oc, f_g, angles_wg_rad, polarization="TE", n_k=n_k
    )
    res_oc = sweep_oc.run(processes=processes)
    out_oc = res_oc.resample()

    eps_prof_oc = build_grating_eps_profile(
        1024, fill_factor=fill_oc, eps_ridge=N_SIN_TIAN**2, eps_groove=1.0
    )
    im_m1 = orders_oc.index(-1)
    im_0 = orders_oc.index(0)

    rcwa_r0 = np.zeros_like(angles_wg_deg)
    rcwa_rm1 = np.zeros_like(angles_wg_deg)
    rcwa_tm1 = np.zeros_like(angles_wg_deg)
    rcwa_t0 = np.zeros_like(angles_wg_deg)
    for ia, th_r in enumerate(angles_wg_rad):
        rc = rcwa_1d_grating(
            f_g,
            th_r,
            period_oc,
            h_oc,
            eps_prof_oc,
            eps_below=N_WG_TIAN**2,
            eps_above=1.0,
            pol="TE",
            orders=orders_oc,
        )
        rcwa_r0[ia] = rc["R_orders"][0, im_0]
        rcwa_rm1[ia] = rc["R_orders"][0, im_m1]
        rcwa_tm1[ia] = rc["T_orders"][0, im_m1]
        rcwa_t0[ia] = rc["T_orders"][0, im_0]

    wakis_r0 = out_oc["R_00"][:, 0]
    wakis_rm1 = out_oc["R_order"][:, 0, im_m1, 0]
    wakis_tm1 = out_oc["T_order"][:, 0, im_m1, 0]
    wakis_t0 = out_oc["T_00"][:, 0]
    energy_oc = out_oc["R_total"][:, 0] + out_oc["T_total"][:, 0]

    # Verify Lorentz reciprocity between in-coupling (air m=0 -> wg m=-1) and
    # out-coupling (wg m=+1/-1 -> air m=0) at theta_air = 8 deg using FloquetPort
    theta_air_rec = np.deg2rad(8.0)
    kx_air = (2.0 * np.pi / wl_g) * np.sin(theta_air_rec)
    G_oc = 2.0 * np.pi / period_oc
    # In-coupling from air (port 2 above -> port 1 below) vs out-coupling from glass (port 1 below -> port 2 above)
    pb_pos, pa_pos = _run_single_Solver(
        cell_oc, k_par=abs(kx_air - G_oc), phi=np.pi, polarization="TE", frequencies=f_g
    )
    s_col_oc = s_parameters_from_ports(
        [pb_pos, pa_pos], incident_mode=FloquetMode(1, 0, 0, "TE")
    )
    t_out_rec = float(np.abs(s_col_oc[FloquetMode(2, 1, 0, "TE")][0]) ** 2)
    rc_rec_check = rcwa_1d_grating(
        f_g,
        theta_air_rec,
        period_oc,
        h_oc,
        eps_prof_oc,
        eps_below=1.0,
        eps_above=N_WG_TIAN**2,
        pol="TE",
        orders=(-1, 0, 1),
    )
    t_in_rec_rcwa = float(rc_rec_check["T_orders"][0, 0])

    # Eye-box pupil expansion across 6 bounces: Constant depth (179 nm) vs
    # Wang & Ma (arXiv:2307.01600, Table 1) graduated depths (143..384 nm, fill=0.18)
    # etched into the N-LAF2 waveguide surface (n_wg = 1.74)
    th_bounce_rad = np.deg2rad(46.0)
    f_wang = np.array([c / WL_WANG_M])
    eps_prof_wang = build_grating_eps_profile(
        1024,
        fill_factor=0.50,
        eps_ridge=2.10**2,
        eps_groove=1.65**2,
    )
    r0_grad = np.zeros(6)
    tout_grad = np.zeros(6)
    for kb, d_nm in enumerate(WANG_OUTCOUPLER_DEPTHS_NM):
        rc_b = rcwa_1d_grating(
            f_wang,
            th_bounce_rad,
            PERIOD_GREEN_1ST_M,
            d_nm * 1e-9,
            eps_prof_wang,
            eps_below=N_WG_WANG**2,
            eps_above=1.0,
            pol="TE",
            orders=(-1, 0),
        )
        r0_grad[kb] = rc_b["R_orders"][0, 1]
        tout_grad[kb] = rc_b["T_orders"][0, 0] + rc_b["R_orders"][0, 0]

    # Constant-depth baseline (all 6 bounces at 232 nm)
    r0_const = np.full(6, r0_grad[3])
    tout_const = np.full(6, tout_grad[3])
    eb_const = simulate_eyebox_pupil_expansion(r0_const, tout_const, incouple_eff=0.80)
    eb_grad = simulate_eyebox_pupil_expansion(r0_grad, tout_grad, incouple_eff=0.80)

    npz_data = {
        "angles_wg_deg": angles_wg_deg,
        "wakis_r0": wakis_r0,
        "rcwa_r0": rcwa_r0,
        "wakis_rm1": wakis_rm1,
        "rcwa_rm1": rcwa_rm1,
        "wakis_tm1": wakis_tm1,
        "rcwa_tm1": rcwa_tm1,
        "wakis_t0": wakis_t0,
        "rcwa_t0": rcwa_t0,
        "energy_oc": energy_oc,
        "wang_depths_nm": WANG_OUTCOUPLER_DEPTHS_NM,
        "eb_const_extracted": eb_const["extracted"],
        "eb_grad_extracted": eb_grad["extracted"],
        "eb_const_guided": eb_const["guided_remaining"],
        "eb_grad_guided": eb_grad["guided_remaining"],
    }
    summary = {
        "index": 2,
        "name": "outcoupler_tir_and_eyebox_uniformity",
        "cpw": cpw,
        "mesh": minfo_oc,
        "tir_t0_all_nan": bool(np.all(np.isnan(wakis_t0))),
        "mean_r0_wakis": float(np.mean(wakis_r0)),
        "mean_r0_rcwa": float(np.mean(rcwa_r0)),
        "r0_uniformity_tian": tian_uniformity(wakis_r0),
        "tm1_uniformity_tian": tian_uniformity(wakis_tm1),
        "max_abs_err_r0": float(np.max(np.abs(wakis_r0 - rcwa_r0))),
        "max_abs_err_tm1": float(np.max(np.abs(wakis_tm1 - rcwa_tm1))),
        "max_energy_residual": float(np.max(np.abs(energy_oc - 1.0))),
        "reciprocity_t_out": t_out_rec,
        "reciprocity_t_in_rcwa": t_in_rec_rcwa,
        "reciprocity_abs_diff": abs(t_out_rec - t_in_rec_rcwa),
        "eyebox_constant_uniformity_tian": eb_const["uniformity_tian"],
        "eyebox_graduated_uniformity_tian": eb_grad["uniformity_tian"],
        "eyebox_constant_wang_err": eb_const["uniformity_error_wang"],
        "eyebox_graduated_wang_err": eb_grad["uniformity_error_wang"],
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    return summary, npz_data


def run_task_3_nearfield_and_resolution(
    quick: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Task 3: 4-level grid resolution study vs RCWA + near-field phasor maps."""
    t0 = time.perf_counter()
    cpw_levels = [16, 20, 26, 34] if quick else [16, 24, 32, 48]
    wl_g = RGB_WAVELENGTHS_M["G"]
    f_g = np.array([c / wl_g])
    theta_test_deg = 10.0
    theta_test_rad = np.deg2rad(theta_test_deg)
    k_par = float((2.0 * np.pi / wl_g) * np.sin(theta_test_rad))

    eps_prof = build_grating_eps_profile(
        1024, fill_factor=0.50, eps_ridge=N_SIN_TIAN**2, eps_groove=1.0
    )
    orders_conv = (-1, 0, 1)
    rc_exact = rcwa_1d_grating(
        f_g,
        theta_test_rad,
        PERIOD_GREEN_1ST_M,
        H_SIN_M,
        eps_prof,
        eps_below=1.0,
        eps_above=N_WG_TIAN**2,
        pol="TE",
        n_harmonics=25,
        orders=orders_conv,
    )
    t_exact = rc_exact["T_orders"][0]
    r_exact = rc_exact["R_orders"][0]

    err_t_m1 = []
    err_t_0 = []
    err_rms = []
    energy_res = []
    dx_nm_list = []

    for cpw in cpw_levels:
        cell_c, minfo_c = build_coupler_floquet_cell(
            period=PERIOD_GREEN_1ST_M,
            height=H_SIN_M,
            wavelength_min=wl_g,
            fill_factor=0.50,
            eps_ridge=N_SIN_TIAN**2,
            eps_groove=1.0,
            eps_below=1.0,
            eps_above=N_WG_TIAN**2,
            orders_x=orders_conv,
            cpw=cpw,
        )
        pb, pa = _run_single_Solver(
            cell_c, k_par=k_par, phi=0.0, polarization="TE", frequencies=f_g
        )
        a_inc = pb.mode_amplitude((1, 0, 0, "TE"), direction="in")[0]
        t_w = np.array(
            [
                abs(pa.mode_amplitude((2, m, 0, "TE"), direction="out")[0] / a_inc) ** 2
                for m in orders_conv
            ]
        )
        r_w = np.array(
            [
                abs(pb.mode_amplitude((1, m, 0, "TE"), direction="out")[0] / a_inc) ** 2
                for m in orders_conv
            ]
        )
        r_valid = np.where(np.isfinite(r_w), r_w, 0.0)
        t_valid = np.where(np.isfinite(t_w), t_w, 0.0)
        tot = float(np.sum(r_valid) + np.sum(t_valid))

        dx_nm_list.append(minfo_c["dx_nm"])
        err_t_m1.append(float(abs(t_w[0] - t_exact[0])))
        err_t_0.append(float(abs(t_w[1] - t_exact[1])))
        err_rms.append(float(np.sqrt(np.mean((t_w - t_exact) ** 2))))
        energy_res.append(float(abs(tot - 1.0)))

    # Capture steady-state 2D complex field phasor maps for in-coupler and out-coupler
    cpw_field = 24 if quick else 32
    cell_ic_f, _ = build_coupler_floquet_cell(
        period=PERIOD_GREEN_1ST_M,
        height=H_SIN_M,
        wavelength_min=wl_g,
        fill_factor=0.50,
        eps_ridge=N_SIN_TIAN**2,
        eps_groove=1.0,
        eps_below=1.0,
        eps_above=N_WG_TIAN**2,
        orders_x=orders_conv,
        cpw=cpw_field,
    )
    nf_ic = capture_coupler_harmonic_field(
        cell_ic_f, float(f_g[0]), np.deg2rad(10.0), polarization="TE"
    )

    cell_oc_f, _ = build_coupler_floquet_cell(
        period=PERIOD_GREEN_1ST_M,
        height=140.0e-9,
        wavelength_min=wl_g,
        fill_factor=0.22,
        eps_ridge=N_SIN_TIAN**2,
        eps_groove=1.0,
        eps_below=N_WG_TIAN**2,
        eps_above=1.0,
        orders_x=orders_conv,
        cpw=cpw_field,
    )
    nf_oc = capture_coupler_harmonic_field(
        cell_oc_f, float(f_g[0]), np.deg2rad(45.0), polarization="TE"
    )

    npz_data = {
        "cpw_levels": np.asarray(cpw_levels, dtype=int),
        "dx_nm": np.asarray(dx_nm_list, dtype=float),
        "err_t_m1": np.asarray(err_t_m1, dtype=float),
        "err_t_0": np.asarray(err_t_0, dtype=float),
        "err_rms": np.asarray(err_rms, dtype=float),
        "energy_res": np.asarray(energy_res, dtype=float),
        "t_exact": t_exact,
        "r_exact": r_exact,
        "ic_x": nf_ic["x"],
        "ic_z": nf_ic["z"],
        "ic_x_edges": nf_ic["x_edges"],
        "ic_z_edges": nf_ic["z_edges"],
        "ic_phasor": nf_ic["field_phasor"],
        "ic_eps": nf_ic["eps_map"],
        "ic_kx": np.array(nf_ic["kx"]),
        "oc_x": nf_oc["x"],
        "oc_z": nf_oc["z"],
        "oc_phasor": nf_oc["field_phasor"],
        "oc_eps": nf_oc["eps_map"],
        "oc_kx": np.array(nf_oc["kx"]),
    }
    summary = {
        "index": 3,
        "name": "nearfield_and_resolution_study",
        "cpw_levels": cpw_levels,
        "dx_nm": dx_nm_list,
        "err_rms": err_rms,
        "err_t_m1": err_t_m1,
        "energy_res": energy_res,
        "convergence_ratio_first_to_last": float(err_rms[0] / max(err_rms[-1], 1e-15)),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    return summary, npz_data
