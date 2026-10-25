# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Simulation and plotting helpers for Tutorial 4.2: Half-Wave Dipole and Quarter-Wave Monopole.

Covers:
1. Center-fed half-wave dipole in 3D free space with ``wakis.lumped.LumpedPort`` and the
   contour-path / in-cell inductance thin-wire subcell model ``wakis.subcell``
   (``solver.add_thin_wire``): input impedance $Z_{\\text{in}}(f) = R_{\\text{in}} + j X_{\\text{in}}$
   and $|S_{11}(f)|$ vs. Carter's induced-EMF formula ($73.08 + j 42.51\\;\\Omega$ for $L = \\lambda_0/2$,
   $a \\to 0$) and Hallén–King–Middleton moment-method reference (``wakis.analytic.thin_wire``),
   resonant length shortening $L_{\\text{res}} / \\lambda_0 < 0.50$ with wire radius $a$
   (thickness parameter $\\Omega = 2\\ln(L/a)$), wire current distribution $I(z)$ via integral
   Ampère circulation $\\oint \\mathbf{H}\\cdot\\mathrm{d}\\tilde{\\mathbf{s}}$ vs. the sinusoidal
   approximation, and far-field directivity $D_0 \\approx 1.643 = 2.15\\;\\text{dBi}$ ($\\text{HPBW} \\approx 78^\\circ$).
2. Resolved 3D PEC wire ($4\\times 4$ cells across the conductor cross-section on a locally
   refined graded mesh) vs. the subcell thin-wire model on a graded mesh vs. a uniform mesh
   baseline, preflighted with ``wakis.estimate``.
3. Quarter-wave ($\\lambda_0/4$) monopole over an infinite PEC ground plane
   (``bc_low=['cpml','cpml','pec']``, ``solver.add_sheet(plane=('z', 0.0), sigma_s='pec')``,
   ``Near2FarMonitor(..., ground_plane='-z')``): image theory verification
   $Z_{\\text{in,mono}}(f) = \\frac{1}{2} Z_{\\text{in,dip}}(f) \\approx 36.5 + j 21.3\\;\\Omega$,
   upper-hemisphere directivity $D_{0,\\text{mono}} = 2 D_{0,\\text{dip}} \\approx 3.29 = 5.17\\;\\text{dBi}$
   ($+3.01\\;\\text{dB}$), and zero lower-hemisphere radiation.
4. Quarter-wave monopole centered on a finite PEC ground plane of width
   $W_{\\text{gnd}} / \\lambda_0 \\in \\{0.40, 0.80, 1.20\\}$ in 6-face CPML
   (``solver.add_sheet(center=(0,0,0), size=(W_gnd, W_gnd, 0), sigma_s='pec')``): edge
   diffraction tilting the main lobe upward ($\\theta_{\\max} < 90^\\circ$) and spilling
   radiation into the lower shadow hemisphere ($\\theta > 90^\\circ$).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from scipy.constants import c as c0
from scipy.constants import epsilon_0, mu_0
from scipy.optimize import brentq

import wakis
from wakis import figstyle, schematic
from wakis.analytic.thin_wire import (
    dipole_far_field,
    hallen_dipole,
    induced_emf_impedance,
)
from wakis.estimate import estimate
from wakis.templates import Box
from wakis.lumped import LumpedPort
from wakis.near2far import Near2FarMonitor, half_power_beamwidth, plot_pattern_3d
from wakis.sources import GaussianSource

ETA0: float = float(np.sqrt(mu_0 / epsilon_0))
F0_HZ: float = 1.0e9
LAM0_M: float = float(c0 / F0_HZ)
K0_RAD_M: float = float(2.0 * np.pi / LAM0_M)
L_HALF_M: float = 0.50 * LAM0_M
H_QUARTER_M: float = 0.25 * LAM0_M
FWIDTH_HZ: float = 0.50 * F0_HZ

COLORS: dict[str, str] = {
    "blue": figstyle.OKABE_ITO[0],
    "red": figstyle.OKABE_ITO[1],
    "teal": figstyle.OKABE_ITO[2],
    "orange": figstyle.OKABE_ITO[3],
    "purple": figstyle.OKABE_ITO[4],
    "sky": figstyle.OKABE_ITO[5],
    "green": figstyle.OKABE_ITO[2],
    "grey": "#6B7280",
}

TASKS: tuple[str, ...] = (
    "half_wave_thin_wire_radii",
    "resolved_wire_vs_subcell_mesh",
    "quarter_wave_monopole_infinite_gnd",
    "finite_ground_plane_sweep",
)


# ---------------------------------------------------------------------------
# Analytical Thin-Wire, Sinusoidal Pattern, and Resonant Shortening Helpers
# ---------------------------------------------------------------------------


def thickness_parameter_omega(length_m: float, radius_m: float) -> float:
    """Return the Hallén–King wire thickness parameter $\\Omega = 2\\ln(L / a)$.

    Parameters
    ----------
    length_m : float
        Total dipole length $L$ [m].
    radius_m : float
        Wire radius $a$ [m].

    Returns
    -------
    float
        Dimensionless thickness parameter $\\Omega = 2\\ln(L / a)$ [-].
    """
    return float(2.0 * np.log(float(length_m) / float(radius_m)))


def sinusoidal_current_distribution(z_m: np.ndarray, length_m: float, freq_hz: float, i_feed: complex = 1.0 + 0.0j) -> np.ndarray:
    """Return the classical standing-wave sinusoidal current $I(z)$ on a center-fed dipole.

    Parameters
    ----------
    z_m : np.ndarray
        Axial coordinates $z \\in [-L/2, L/2]$ [m].
    length_m : float
        Total dipole length $L$ [m].
    freq_hz : float
        Frequency $f$ [Hz].
    i_feed : complex, optional
        Terminal current $I(0)$ [A]. Default is $1.0\\;\\text{A}$.

    Returns
    -------
    np.ndarray
        Current profile $I(z) = I(0)\\,\\sin[k(L/2 - |z|)] / \\sin(kL/2)$ [A].
    """
    z_arr = np.asarray(z_m, dtype=float)
    k = 2.0 * np.pi * float(freq_hz) / c0
    half_l = 0.5 * float(length_m)
    denom = np.sin(k * half_l)
    if abs(denom) < 1e-12:
        denom = 1.0
    profile = np.maximum(0.0, 1.0 - np.abs(z_arr) / half_l)
    out = complex(i_feed) * np.sin(k * np.maximum(0.0, half_l - np.abs(z_arr))) / denom
    out[np.abs(z_arr) >= half_l] = 0.0
    return out


def half_wave_analytic_directivity(theta_rad: np.ndarray, mono_half_space: bool = False) -> np.ndarray:
    """Return the closed-form directivity $D(\\theta)$ of an ideal $\\lambda/2$ dipole or $\\lambda/4$ monopole.

    For an infinitely thin half-wave dipole in free space:
    $D_{\\text{dip}}(\\theta) = D_{0,\\text{dip}} \\left[\\frac{\\cos((\\pi/2)\\cos\\theta)}{\\sin\\theta}\\right]^2$
    with $D_{0,\\text{dip}} = 4 / \\operatorname{Cin}(2\\pi) \\approx 1.64092$ ($2.15\\;\\text{dBi}$).
    For a quarter-wave monopole over an infinite PEC ground plane ($z \\ge 0$), image theory
    restricts radiation to $0 \\le \\theta \\le \\pi/2$ and doubles the directivity to
    $D_{0,\\text{mono}} = 2 D_{0,\\text{dip}} \\approx 3.28184$ ($5.16\\;\\text{dBi}$).

    Parameters
    ----------
    theta_rad : np.ndarray
        Polar angles $\\theta \\in [0, \\pi]$ [rad].
    mono_half_space : bool, optional
        If ``True``, return the upper-hemisphere monopole directivity ($2 D_{\\text{dip}}$ for
        $\\theta \\le \\pi/2$, zero for $\\theta > \\pi/2$).

    Returns
    -------
    np.ndarray
        Linear directivity $D(\\theta)$ [-].
    """
    th = np.asarray(theta_rad, dtype=float)
    st = np.sin(th)
    ct = np.cos(th)
    pattern = np.zeros_like(th)
    mask = np.abs(st) > 1e-12
    pattern[mask] = (np.cos(0.5 * np.pi * ct[mask]) / st[mask]) ** 2
    # Exact integral int_0^pi [cos((pi/2)cos theta)/sin theta]^2 sin theta dtheta = Cin(2*pi) = 2.4376534
    cin_2pi = 2.437653392906558
    d0_dip = 4.0 / cin_2pi
    if mono_half_space:
        out = 2.0 * d0_dip * pattern
        out[th > 0.5 * np.pi + 1e-9] = 0.0
        return out
    return d0_dip * pattern


def hallen_resonant_length_ratio(radius_over_lam0: float, n_segments: int = 120, gap_over_lam0: float = 1.0 / 38.0) -> tuple[float, float]:
    """Compute the first series-resonant length $L_{\\text{res}} / \\lambda_0$ ($X_{\\text{in}} = 0$) from Hallén MoM.

    Parameters
    ----------
    radius_over_lam0 : float
        Normalized wire radius $a / \\lambda_0$ [-].
    n_segments : int, optional
        Number of Hallén MoM segments. Default is 120.
    gap_over_lam0 : float, optional
        Normalized feed-gap width $g / \\lambda_0$ [-]. Default is 0.0.

    Returns
    -------
    l_res_over_lam0 : float
        Resonant electrical length $L_{\\text{res}} / \\lambda_0$ where $\\operatorname{Im} Z_{\\text{in}} = 0$.
    r_res_ohm : float
        Input resistance $R_{\\text{in}}(L_{\\text{res}})$ at resonance [$\\Omega$].
    """
    a_m = float(radius_over_lam0) * LAM0_M
    g_m = float(gap_over_lam0) * LAM0_M
    l_nom = 0.50 * LAM0_M

    def imag_z(f_ratio: float) -> float:
        z_val, _, _ = hallen_dipole(l_nom, a_m, f_ratio * F0_HZ, n_segments=n_segments, gap=g_m)
        return float(z_val[0].imag)

    f_ratio_res = float(brentq(imag_z, 0.84, 0.998, xtol=1e-4))
    z_res, _, _ = hallen_dipole(l_nom, a_m, f_ratio_res * F0_HZ, n_segments=n_segments, gap=g_m)
    return 0.50 * f_ratio_res, float(z_res[0].real)


def print_reference_table() -> None:
    """Print the analytical thin-wire and monopole benchmarks at $f_0 = 1.0\\;\\text{GHz}$."""
    z_emf = induced_emf_impedance(L_HALF_M, 1e-9, F0_HZ)
    z_mom_thin, _, _ = hallen_dipole(L_HALF_M, 0.2e-3, F0_HZ, n_segments=120, gap=0.0)
    z_mom_thick, _, _ = hallen_dipole(L_HALF_M, 1.5e-3, F0_HZ, n_segments=120, gap=0.0)
    l_res_thin, r_res_thin = hallen_resonant_length_ratio(0.2e-3 / LAM0_M)
    l_res_thick, r_res_thick = hallen_resonant_length_ratio(1.5e-3 / LAM0_M)
    print("=== Tutorial 4.2: Wire Dipoles & Monopoles — Analytical Reference Benchmarks ===")
    print(f"  Frequency f0 = {F0_HZ * 1e-9:.2f} GHz, wavelength lam0 = {LAM0_M * 1e3:.2f} mm")
    print(f"  Half-wave length L = lam0/2 = {L_HALF_M * 1e3:.2f} mm, quarter-wave h = lam0/4 = {H_QUARTER_M * 1e3:.2f} mm")
    print(f"  Induced-EMF limit (L = lam0/2, a -> 0) : Z_in = {z_emf.real:.2f} + j{z_emf.imag:.2f} Ohm, D0 = 1.641 (2.15 dBi)")
    print(
        f"  Hallen MoM a = 0.20 mm (Omega = {thickness_parameter_omega(L_HALF_M, 0.2e-3):.1f}) : "
        f"Z_in(lam0/2) = {z_mom_thin[0].real:.2f} + j{z_mom_thin[0].imag:.2f} Ohm, "
        f"L_res/lam0 = {l_res_thin:.4f} (R_res = {r_res_thin:.1f} Ohm)"
    )
    print(
        f"  Hallen MoM a = 1.50 mm (Omega = {thickness_parameter_omega(L_HALF_M, 1.5e-3):.1f}) : "
        f"Z_in(lam0/2) = {z_mom_thick[0].real:.2f} + j{z_mom_thick[0].imag:.2f} Ohm, "
        f"L_res/lam0 = {l_res_thick:.4f} (R_res = {r_res_thick:.1f} Ohm)"
    )
    print(
        f"  Quarter-wave monopole (image theory)   : Z_mono = 0.5 * Z_dip = {0.5 * z_emf.real:.2f} + j{0.5 * z_emf.imag:.2f} Ohm, "
        f"D0 = 3.282 (5.16 dBi, +3.01 dB)"
    )


# ---------------------------------------------------------------------------
# Graded and Uniform Mesh Builders & Preflight Estimate
# ---------------------------------------------------------------------------


def build_subcell_dipole_grid(
    length_m: float = L_HALF_M,
    n_seg: int = 19,
    quick: bool = False,
) -> tuple[wakis.GridFIT3D, dict[str, Any]]:
    """Build a 3D graded FIT mesh for a center-fed thin-wire dipole along $z$.

    Uses $n_{\\text{seg}}$ (odd) uniform axial cells of height $\\Delta z_w = L / n_{\\text{seg}}$
    across $[-L/2, +L/2]$ so the center segment $k = \\text{iz}_{\\text{feed}}$ sits symmetrically
    across $z = 0$, surrounded by geometrically graded padding cells out to the 6-cell CPML.

    Parameters
    ----------
    length_m : float, optional
        Dipole length $L$ [m]. Default is $\\lambda_0 / 2$.
    n_seg : int, optional
        Number of axial wire segments (forced odd). Default is 19.
    quick : bool, optional
        If ``True``, use a lighter outer padding for rapid smoke runs.

    Returns
    -------
    grid : wakis.GridFIT3D
        Constructed 3D non-uniform grid.
    meta : dict
        Dictionary of wire node indices, gap cell coordinates, and cell sizes.
    """
    n_seg_odd = int(n_seg) if int(n_seg) % 2 == 1 else int(n_seg) + 1
    dz_w = float(length_m) / float(n_seg_odd)
    z_core = np.linspace(-0.5 * length_m, 0.5 * length_m, n_seg_odd + 1)

    n_pad_z = 8 if quick else 10
    z_steps = dz_w * np.geomspace(1.05, 2.0, n_pad_z)
    z_hi = 0.5 * length_m + np.cumsum(z_steps)
    z_nodes = np.concatenate([-z_hi[::-1], z_core, z_hi])

    dx_w = dz_w
    xy_core = np.array([-dx_w, 0.0, dx_w], dtype=float)
    n_pad_xy = 10 if quick else 13
    xy_steps = dx_w * np.geomspace(1.05, 2.1, n_pad_xy)
    xy_hi = dx_w + np.cumsum(xy_steps)
    xy_nodes = np.concatenate([-xy_hi[::-1], xy_core, xy_hi])

    grid = wakis.GridFIT3D(x=xy_nodes, y=xy_nodes, z=z_nodes, use_mpi=False, verbose=0)
    ix0 = int(np.argmin(np.abs(grid.x)))
    iy0 = int(np.argmin(np.abs(grid.y)))
    iz0 = int(np.argmin(np.abs(grid.z - (-0.5 * length_m))))
    iz1 = int(np.argmin(np.abs(grid.z - (0.5 * length_m))))
    iz_feed = (iz0 + iz1) // 2
    z_feed = 0.5 * float(grid.z[iz_feed] + grid.z[iz_feed + 1])

    meta = {
        "ix0": ix0,
        "iy0": iy0,
        "iz0": iz0,
        "iz1": iz1,
        "iz_feed": iz_feed,
        "z_feed": z_feed,
        "dz_w": dz_w,
        "dx_w": dx_w,
        "length_m": float(grid.z[iz1] - grid.z[iz0]),
        "n_seg": iz1 - iz0,
        "dx_min": float(np.min(np.diff(grid.x))),
        "dx_max": float(np.max(np.diff(grid.x))),
        "dz_min": float(np.min(np.diff(grid.z))),
        "dz_max": float(np.max(np.diff(grid.z))),
    }
    return grid, meta


def build_image_theory_pair_grids(
    h_mono_m: float = H_QUARTER_M,
    n_half: int = 10,
    quick: bool = False,
) -> tuple[wakis.GridFIT3D, wakis.GridFIT3D, dict[str, Any]]:
    """Build mirror-symmetric 3D FIT grids for a $\\lambda/4$ monopole ($z \\ge 0$) and $\\lambda/2$ dipole.

    The monopole grid has $z_0 = 0$ on the PEC ground plane with $n_{\\text{half}}$ uniform
    axial cells along $[0, h]$, where the bottom cell $z \\in [0, \\Delta z_w]$ hosts the
    base ``LumpedPort``. The dipole grid is its exact mirror reflection across $z = 0$,
    with the center ``LumpedPort`` spanning $[-\\Delta z_w, +\\Delta z_w]$ so image theory
    $Z_{\\text{in,mono}}(f) = \\frac{1}{2} Z_{\\text{in,dip}}(f)$ holds directly on the discrete grid.
    """
    n_h = max(6, int(n_half))
    dz_w = float(h_mono_m) / float(n_h)
    z_arm = np.linspace(0.0, float(h_mono_m), n_h + 1)
    n_pad_z = 8 if quick else 11
    z_hi = float(h_mono_m) + np.cumsum(dz_w * np.geomspace(1.0, 2.0, n_pad_z))
    z_mono = np.concatenate([z_arm, z_hi])
    z_dip = np.concatenate([-z_mono[:0:-1], z_mono])

    dx_w = dz_w
    xy_core = np.array([-dx_w, 0.0, dx_w], dtype=float)
    n_pad_xy = 10 if quick else 14
    xy_hi = dx_w + np.cumsum(dx_w * np.geomspace(1.0, 2.1, n_pad_xy))
    xy_nodes = np.concatenate([-xy_hi[::-1], xy_core, xy_hi])

    grid_dip = wakis.GridFIT3D(x=xy_nodes, y=xy_nodes, z=z_dip, use_mpi=False, verbose=0)
    grid_mono = wakis.GridFIT3D(x=xy_nodes, y=xy_nodes, z=z_mono, use_mpi=False, verbose=0)
    ix0 = int(np.argmin(np.abs(xy_nodes)))
    iz_top_mono = int(np.argmin(np.abs(z_mono - h_mono_m)))
    iz_top_dip = int(np.argmin(np.abs(z_dip - h_mono_m)))
    bx = float(xy_nodes[ix0 + (5 if quick else 6)])
    bz = float(z_mono[iz_top_mono + 3])

    meta = {
        "dz_w": dz_w,
        "dx_w": dx_w,
        "h_mono_m": float(h_mono_m),
        "l_dip_m": 2.0 * float(h_mono_m),
        "n_half": n_h,
        "ix0": ix0,
        "iz_top_mono": iz_top_mono,
        "iz_top_dip": iz_top_dip,
        "bx": bx,
        "bz": bz,
    }
    return grid_dip, grid_mono, meta


def build_resolved_wire_grid(
    length_m: float,
    radius_m: float,
    n_seg: int = 17,
    quick: bool = False,
) -> tuple[wakis.GridFIT3D, dict[str, Any]]:
    """Build a locally refined 3D FIT mesh resolving a square PEC wire with $4\\times 4$ cells.

    By conformal mapping (Lo 1953; Balanis Table 9.3), a square conductor of full width
    $2w$ has equivalent circular wire radius $a_{\\text{eq}} = 0.59\\,(2w) = 1.18\\,w$.
    We place 4 uniform cells of width $\\Delta_{\\perp,\\text{core}} = w / 2$ across $[-w, +w]$
    along both $x$ and $y$, and grade geometrically outward to $\\Delta_{\\max} \\approx 14\\;\\text{mm}$.
    """
    w_half = float(radius_m) / 1.18
    dx_core = 0.5 * w_half  # 4 cells across [-w_half, +w_half]
    xy_core = np.linspace(-2.0 * dx_core, 2.0 * dx_core, 5)
    n_pad_xy = 10 if quick else 13
    xy_steps = dx_core * np.geomspace(1.4, 18.0, n_pad_xy)
    xy_hi = 2.0 * dx_core + np.cumsum(xy_steps)
    xy_nodes = np.concatenate([-xy_hi[::-1], xy_core, xy_hi])

    n_seg_odd = int(n_seg) if int(n_seg) % 2 == 1 else int(n_seg) + 1
    dz_w = float(length_m) / float(n_seg_odd)
    z_core = np.linspace(-0.5 * length_m, 0.5 * length_m, n_seg_odd + 1)
    n_pad_z = 7 if quick else 9
    z_steps = dz_w * np.geomspace(1.05, 2.1, n_pad_z)
    z_hi = 0.5 * length_m + np.cumsum(z_steps)
    z_nodes = np.concatenate([-z_hi[::-1], z_core, z_hi])

    grid = wakis.GridFIT3D(x=xy_nodes, y=xy_nodes, z=z_nodes, use_mpi=False, verbose=0)
    ix_c = int(np.argmin(np.abs(grid.x)))
    iy_c = int(np.argmin(np.abs(grid.y)))
    iz0 = int(np.argmin(np.abs(grid.z - (-0.5 * length_m))))
    iz1 = int(np.argmin(np.abs(grid.z - (0.5 * length_m))))
    iz_feed = (iz0 + iz1) // 2
    z_feed = 0.5 * float(grid.z[iz_feed] + grid.z[iz_feed + 1])

    meta = {
        "ix_c": ix_c,
        "iy_c": iy_c,
        "iz0": iz0,
        "iz1": iz1,
        "iz_feed": iz_feed,
        "z_feed": z_feed,
        "dx_core": dx_core,
        "w_half": w_half,
        "dz_w": dz_w,
        "dx_max": float(np.max(np.diff(grid.x))),
        "dz_max": float(np.max(np.diff(grid.z))),
    }
    return grid, meta


def preflight_estimates() -> dict[str, dict[str, Any]]:
    """Run ``wakis.estimate`` preflight across subcell graded, resolved graded, and uniform meshes."""
    g_sub, m_sub = build_subcell_dipole_grid(L_HALF_M, n_seg=19, quick=False)
    dt_sub = m_sub["dx_min"] / (np.sqrt(3.0) * c0)
    steps_sub = int(np.ceil((4.5 / F0_HZ) / dt_sub))
    est_sub = estimate(cells=(int(g_sub.Nx), int(g_sub.Ny), int(g_sub.Nz)), steps=steps_sub, cpml=True)

    l_res = 0.48 * LAM0_M
    a_res = 1.8e-3
    g_res, m_res = build_resolved_wire_grid(l_res, a_res, n_seg=17, quick=False)
    dt_res = m_res["dx_core"] / (np.sqrt(3.0) * c0)
    steps_res = int(np.ceil((4.2 / F0_HZ) / dt_res))
    est_res = estimate(cells=(int(g_res.Nx), int(g_res.Ny), int(g_res.Nz)), steps=steps_res, cpml=True)

    # Uniform subcell baseline at cpw = 24 (dx = 12.5 mm)
    dx_u = LAM0_M / 24.0
    nx_u = 48
    dt_u = dx_u / (np.sqrt(3.0) * c0)
    steps_u = int(np.ceil((4.5 / F0_HZ) / dt_u))
    est_unif_sub = estimate(cells=(nx_u, nx_u, nx_u), steps=steps_u, cpml=True)

    # Analytical cost of a uniform mesh that resolves dx_core = 0.763 mm everywhere in (0.60 m)^3
    box_span = float(g_res.xmax - g_res.xmin)
    nx_unif_res = int(np.ceil(box_span / m_res["dx_core"]))
    est_unif_res = estimate(cells=(nx_unif_res, nx_unif_res, nx_unif_res), steps=steps_res, cpml=True)

    out = {
        "graded_subcell": {
            "cells": int(est_sub.cells),
            "nx": int(g_sub.Nx),
            "ny": int(g_sub.Ny),
            "nz": int(g_sub.Nz),
            "dx_min_mm": m_sub["dx_min"] * 1e3,
            "dx_max_mm": m_sub["dx_max"] * 1e3,
            "n_steps": steps_sub,
            "ram_mb": float(est_sub.host_bytes / 1e6),
        },
        "graded_resolved": {
            "cells": int(est_res.cells),
            "nx": int(g_res.Nx),
            "ny": int(g_res.Ny),
            "nz": int(g_res.Nz),
            "dx_min_mm": m_res["dx_core"] * 1e3,
            "dx_max_mm": m_res["dx_max"] * 1e3,
            "n_steps": steps_res,
            "ram_mb": float(est_res.host_bytes / 1e6),
        },
        "uniform_subcell_cpw24": {
            "cells": int(est_unif_sub.cells),
            "nx": nx_u,
            "ny": nx_u,
            "nz": nx_u,
            "dx_min_mm": dx_u * 1e3,
            "dx_max_mm": dx_u * 1e3,
            "n_steps": steps_u,
            "ram_mb": float(est_unif_sub.host_bytes / 1e6),
        },
        "uniform_resolved_4cell": {
            "cells": int(est_unif_res.cells),
            "nx": nx_unif_res,
            "ny": nx_unif_res,
            "nz": nx_unif_res,
            "dx_min_mm": m_res["dx_core"] * 1e3,
            "dx_max_mm": m_res["dx_core"] * 1e3,
            "n_steps": steps_res,
            "ram_mb": float(est_unif_res.host_bytes / 1e6),
        },
    }

    print("\n=== wakis.estimate Preflight: Subcell vs. Resolved Wire & Graded vs. Uniform ===")
    print(f"{'Config':<24} {'Grid (Nx,Ny,Nz)':<18} {'Cells':>12} {'dx_min..max [mm]':>18} {'Steps':>8} {'RAM [MB]':>10}")
    print("-" * 96)
    for name, d in out.items():
        g_str = f"({d['nx']},{d['ny']},{d['nz']})"
        dx_str = f"{d['dx_min_mm']:.2f}..{d['dx_max_mm']:.1f}"
        print(f"{name:<24} {g_str:<18} {d['cells']:>12,d} {dx_str:>18} {d['n_steps']:>8,d} {d['ram_mb']:>10.1f}")
    return out


# ---------------------------------------------------------------------------
# Task 0: Center-Fed Half-Wave Thin-Wire Dipole (Radii, Current, Near/Far Field)
# ---------------------------------------------------------------------------


def run_task_half_wave_thin_wire_radii(quick: bool = False) -> dict[str, Any]:
    """Run Task 0: Center-fed $\\lambda_0/2$ thin-wire dipole across radii + resonance-trimmed dipole.

    Computes:
    - Input impedance $Z_{\\text{in}}(f) = R_{\\text{in}}(f) + j X_{\\text{in}}(f)$ and $|S_{11}(f)|$
      over $0.80\\text{–}1.15\\;\\text{GHz}$ for wire radii $a \\in \\{1.5, 0.5, 0.15\\}\\;\\text{mm}$
      ($\\Omega = 2\\ln(L/a) \\in \\{9.21, 11.41, 13.82\\}$) at $L = 0.50\\lambda_0$, plus a
      resonance-trimmed dipole $L_{\\text{trim}} = 0.470\\lambda_0$ ($a = 0.5\\;\\text{mm}$).
    - Wire current distribution $I(z_k, f_0)$ via FIT Ampère circulation
      $\\oint \\mathbf{H}\\cdot\\mathrm{d}\\tilde{\\mathbf{s}}$ around each wire segment vs.
      $\\sin[k(L/2 - |z|)]$ and Hallén MoM.
    - 2D $xz$-plane near-field $\\operatorname{Re}[E_z(x, 0, z, f_0)]$ and $|\\mathbf{E}(x, 0, z, f_0)|$,
      time-domain GIF frames, and closed-box ``Near2FarMonitor`` directivity $D(\\theta, \\phi)$.
    """
    n_seg = 13 if quick else 19
    freqs = np.linspace(0.80e9, 1.15e9, 71)
    idx_f0 = int(np.argmin(np.abs(freqs - F0_HZ)))
    radii_m = np.array([1.5e-3, 0.5e-3, 0.15e-3], dtype=float)
    omegas = np.array([thickness_parameter_omega(L_HALF_M, r) for r in radii_m], dtype=float)

    grid_base, meta_base = build_subcell_dipole_grid(L_HALF_M, n_seg=n_seg, quick=quick)
    dz_w = meta_base["dz_w"]
    ix0, iy0 = meta_base["ix0"], meta_base["iy0"]
    iz0, iz1, iz_feed = meta_base["iz0"], meta_base["iz1"], meta_base["iz_feed"]
    z_feed = meta_base["z_feed"]

    z_fit_all = np.zeros((len(radii_m), len(freqs)), dtype=complex)
    z_mom_all = np.zeros((len(radii_m), len(freqs)), dtype=complex)
    s11_50_all = np.zeros((len(radii_m), len(freqs)), dtype=complex)
    s11_73_all = np.zeros((len(radii_m), len(freqs)), dtype=complex)
    f_res_fit_ghz = np.zeros(len(radii_m), dtype=float)
    f_res_mom_ghz = np.zeros(len(radii_m), dtype=float)
    r_res_fit_ohm = np.zeros(len(radii_m), dtype=float)
    r_res_mom_ohm = np.zeros(len(radii_m), dtype=float)

    # Variables saved from the first (reference a = 1.5 mm or middle a = 0.5 mm) run
    t0_wall = time.perf_counter()
    for r_idx, a_val in enumerate(radii_m):
        grid, _ = build_subcell_dipole_grid(L_HALF_M, n_seg=n_seg, quick=quick)
        src = GaussianSource(frequency=F0_HZ, fwidth=FWIDTH_HZ)
        port = LumpedPort(
            src_time=src,
            component="z",
            center=(0.0, 0.0, z_feed),
            size=(0.0, 0.0, 0.5 * dz_w),
            impedance=50.0,
        )
        solver = wakis.SolverFIT3D(
            grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
        )
        solver.add_thin_wire(
            [(0.0, 0.0, float(grid.z[iz0])), (0.0, 0.0, float(grid.z[iz1]))],
            radius=float(a_val),
            gap=port,
            correction=True,
        )
        n_steps = int(np.ceil((3.6 if quick else 4.5) / (F0_HZ * solver.dt)))

        if r_idx == 1:
            # Reference run (a = 0.5 mm, Omega = 11.4): record Near2Far, wire I(z), 2D xz DFT, and TD frames
            bx = float(grid.x[ix0 + (5 if quick else 6)])
            bz = float(grid.z[iz1 + 3])
            n2f = Near2FarMonitor(
                frequencies=np.array([F0_HZ]),
                center=(0.0, 0.0, 0.0),
                size=(2.0 * bx, 2.0 * bz if False else 2.0 * bx, 2.0 * bz),
            )
            dt = float(solver.dt)
            w0 = 2.0 * np.pi * F0_HZ
            # Wire segment midpoints and dual-loop lengths around wire
            k_wire = np.arange(iz0, iz1, dtype=int)
            z_seg_mid = 0.5 * (grid.z[k_wire] + grid.z[k_wire + 1])
            tlx = float(grid.tL[ix0, iy0, iz_feed, "x"])
            tly = float(grid.tL[ix0, iy0, iz_feed, "y"])
            i_seg_acc = np.zeros(len(k_wire), dtype=complex)

            ex_xz_acc = np.zeros((grid.Nx, grid.Nz), dtype=complex)
            ez_xz_acc = np.zeros((grid.Nx, grid.Nz), dtype=complex)
            td_frames: list[np.ndarray] = []
            td_times_ns: list[float] = []
            frame_steps = np.linspace(int(0.16 * n_steps), int(0.68 * n_steps), 18, dtype=int)
            frame_set = set(int(s) for s in frame_steps)

            for step in range(n_steps):
                t = step * dt
                port.update(solver, t)
                solver.one_step()
                port.post_step(solver, t)
                n2f.update(solver, t)

                ph_e = dt * np.exp(-1j * w0 * (t + dt))
                ph_h = dt * np.exp(-1j * w0 * (t + 0.5 * dt))
                ex_xz_acc += ph_e * np.asarray(solver.E[:, iy0, :, "x"])
                ez_xz_acc += ph_e * np.asarray(solver.E[:, iy0, :, "z"])

                # Integral Ampere law around each wire segment k in [iz0, iz1)
                hx_lo = np.asarray(solver.H[ix0, iy0 - 1, k_wire, "x"])
                hx_hi = np.asarray(solver.H[ix0, iy0, k_wire, "x"])
                hy_lo = np.asarray(solver.H[ix0 - 1, iy0, k_wire, "y"])
                hy_hi = np.asarray(solver.H[ix0, iy0, k_wire, "y"])
                i_circ = tlx * (hx_lo - hx_hi) + tly * (hy_hi - hy_lo)
                i_seg_acc += ph_h * i_circ

                if step in frame_set:
                    td_frames.append(np.asarray(solver.E[:, iy0, :, "z"], dtype=np.float32).copy())
                    td_times_ns.append((t + dt) * 1e9)

            v_f0 = complex(port.voltage([F0_HZ])[0])
            i_port_f0 = complex(port.port_current([F0_HZ])[0])
            # On the gap segment itself, the conduction current into the port is i_port_f0
            i_seg_per_v = i_seg_acc / v_f0
            i_seg_per_v[iz_feed - iz0] = i_port_f0 / v_f0
        else:
            solver.emsolve(n_steps, source=port)

        z_in = port.impedance_in(freqs)
        z_mom, z_nodes_mom, i_mom_arr = hallen_dipole(
            L_HALF_M, float(a_val), freqs, n_segments=100, gap=dz_w
        )
        z_fit_all[r_idx] = z_in
        z_mom_all[r_idx] = z_mom
        s11_50_all[r_idx] = (z_in - 50.0) / (z_in + 50.0)
        s11_73_all[r_idx] = (z_in - 73.1) / (z_in + 73.1)

        # Interpolate zero-crossing of Im(Z_in) for resonant frequency
        f_res_fit = float(np.interp(0.0, z_in.imag, freqs))
        f_res_mom = float(np.interp(0.0, z_mom.imag, freqs))
        f_res_fit_ghz[r_idx] = f_res_fit * 1e-9
        f_res_mom_ghz[r_idx] = f_res_mom * 1e-9
        r_res_fit_ohm[r_idx] = float(np.interp(f_res_fit, freqs, z_in.real))
        r_res_mom_ohm[r_idx] = float(np.interp(f_res_mom, freqs, z_mom.real))

        if r_idx == 1:
            z_nodes_mom_ref = z_nodes_mom
            i_mom_f0_ref = i_mom_arr[idx_f0]

    # Run resonance-trimmed dipole L_trim = 0.470 * lam0 (a = 0.5 mm) so X_in(f0) ~ 0
    l_trim_m = 0.470 * LAM0_M
    grid_tr, meta_tr = build_subcell_dipole_grid(l_trim_m, n_seg=n_seg, quick=quick)
    src_tr = GaussianSource(frequency=F0_HZ, fwidth=FWIDTH_HZ)
    port_tr = LumpedPort(
        src_time=src_tr,
        component="z",
        center=(0.0, 0.0, meta_tr["z_feed"]),
        size=(0.0, 0.0, 0.5 * meta_tr["dz_w"]),
        impedance=50.0,
    )
    solver_tr = wakis.SolverFIT3D(
        grid_tr, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    solver_tr.add_thin_wire(
        [(0.0, 0.0, float(grid_tr.z[meta_tr["iz0"]])), (0.0, 0.0, float(grid_tr.z[meta_tr["iz1"]]))],
        radius=0.5e-3,
        gap=port_tr,
        correction=True,
    )
    n_steps_tr = int(np.ceil((3.6 if quick else 4.5) / (F0_HZ * solver_tr.dt)))
    solver_tr.emsolve(n_steps_tr, source=port_tr)
    z_trim_fit = port_tr.impedance_in(freqs)
    z_trim_mom, _, _ = hallen_dipole(l_trim_m, 0.5e-3, freqs, n_segments=100, gap=meta_tr["dz_w"])
    s11_trim_50 = (z_trim_fit - 50.0) / (z_trim_fit + 50.0)
    s11_trim_73 = (z_trim_fit - 73.1) / (z_trim_fit + 73.1)

    wall_s = time.perf_counter() - t0_wall

    # Induced-EMF analytical curve across the band (thin-wire limit a = 1e-6 m)
    z_emf_band = induced_emf_impedance(L_HALF_M, 1e-6, freqs)

    # Continuous Hallen resonant-shortening curve L_res / lam0 vs Omega in [8.5, 16.0]
    omega_curve = np.linspace(8.5, 16.0, 16)
    a_over_lam_curve = 0.50 * np.exp(-0.5 * omega_curve)
    l_res_mom_curve = np.zeros_like(omega_curve)
    r_res_mom_curve = np.zeros_like(omega_curve)
    for idx_o, a_lam in enumerate(a_over_lam_curve):
        l_r, r_r = hallen_resonant_length_ratio(float(a_lam), n_segments=100, gap_over_lam0=dz_w / LAM0_M)
        l_res_mom_curve[idx_o] = l_r
        r_res_mom_curve[idx_o] = r_r

    # Far-field cuts and 3D grid from the reference Near2FarMonitor (a = 0.5 mm)
    theta_deg = np.linspace(1.0, 179.0, 179)
    theta_rad = np.deg2rad(theta_deg)
    d_eplane = np.asarray(n2f.directivity(theta_rad, np.zeros_like(theta_rad))[0], dtype=float)
    d_eplane_ana = half_wave_analytic_directivity(theta_rad, mono_half_space=False)
    phi_deg = np.linspace(0.0, 360.0, 181)
    phi_rad = np.deg2rad(phi_deg)
    d_hplane = np.asarray(
        n2f.directivity(np.full_like(phi_rad, 0.5 * np.pi), phi_rad)[0], dtype=float
    )
    d0_max_arr, _ = n2f.max_directivity()
    d0_fit = float(d0_max_arr[0])
    u_eplane = np.asarray(n2f.radiation_intensity(theta_rad, np.zeros_like(theta_rad))[0], dtype=float)
    hpbw_rad = float(half_power_beamwidth(theta_rad, u_eplane))
    hpbw_deg = float(np.rad2deg(hpbw_rad))

    # 2D (theta, phi) grid for 3D radiation pattern rendering
    th_3d = np.linspace(0.0, np.pi, 61)
    ph_3d = np.linspace(0.0, 2.0 * np.pi, 91)
    TH_3D, PH_3D = np.meshgrid(th_3d, ph_3d, indexing="ij")
    d_3d = np.asarray(n2f.directivity(TH_3D, PH_3D)[0], dtype=float)

    # Normalized 2D near-field maps (divided by V(f0))
    ez_xz_norm = ez_xz_acc / v_f0
    ex_xz_norm = ex_xz_acc / v_f0
    emag_xz = np.sqrt(np.abs(ex_xz_norm) ** 2 + np.abs(ez_xz_norm) ** 2)
    x_cell = 0.5 * (grid_base.x[:-1] + grid_base.x[1:])
    z_cell = 0.5 * (grid_base.z[:-1] + grid_base.z[1:])

    return {
        "freqs_ghz": freqs * 1e-9,
        "f0_ghz": F0_HZ * 1e-9,
        "lam0_m": LAM0_M,
        "l_half_m": L_HALF_M,
        "l_trim_m": l_trim_m,
        "radii_mm": radii_m * 1e3,
        "omegas": omegas,
        "z_fit_real": z_fit_all.real,
        "z_fit_imag": z_fit_all.imag,
        "z_mom_real": z_mom_all.real,
        "z_mom_imag": z_mom_all.imag,
        "z_emf_real": z_emf_band.real,
        "z_emf_imag": z_emf_band.imag,
        "s11_50_db": 20.0 * np.log10(np.maximum(np.abs(s11_50_all), 1e-12)),
        "s11_73_db": 20.0 * np.log10(np.maximum(np.abs(s11_73_all), 1e-12)),
        "z_trim_fit_real": z_trim_fit.real,
        "z_trim_fit_imag": z_trim_fit.imag,
        "z_trim_mom_real": z_trim_mom.real,
        "z_trim_mom_imag": z_trim_mom.imag,
        "s11_trim_50_db": 20.0 * np.log10(np.maximum(np.abs(s11_trim_50), 1e-12)),
        "s11_trim_73_db": 20.0 * np.log10(np.maximum(np.abs(s11_trim_73), 1e-12)),
        "f_res_fit_ghz": f_res_fit_ghz,
        "f_res_mom_ghz": f_res_mom_ghz,
        "l_res_fit_over_lam0": 0.50 * (f_res_fit_ghz / (F0_HZ * 1e-9)),
        "l_res_mom_over_lam0": 0.50 * (f_res_mom_ghz / (F0_HZ * 1e-9)),
        "r_res_fit_ohm": r_res_fit_ohm,
        "r_res_mom_ohm": r_res_mom_ohm,
        "omega_curve": omega_curve,
        "l_res_mom_curve": l_res_mom_curve,
        "r_res_mom_curve": r_res_mom_curve,
        "z_seg_mid_m": z_seg_mid,
        "i_seg_per_v_real": i_seg_per_v.real,
        "i_seg_per_v_imag": i_seg_per_v.imag,
        "z_nodes_mom_m": z_nodes_mom_ref,
        "i_mom_per_v_real": i_mom_f0_ref.real,
        "i_mom_per_v_imag": i_mom_f0_ref.imag,
        "theta_deg": theta_deg,
        "d_eplane": d_eplane,
        "d_eplane_ana": d_eplane_ana,
        "phi_deg": phi_deg,
        "d_hplane": d_hplane,
        "d0_fit": d0_fit,
        "d0_ana": 1.64092,
        "hpbw_deg": hpbw_deg,
        "th_3d": th_3d,
        "ph_3d": ph_3d,
        "d_3d": d_3d,
        "x_cell_m": x_cell,
        "z_cell_m": z_cell,
        "ez_xz_real": ez_xz_norm.real,
        "emag_xz": emag_xz,
        "td_frames": np.asarray(td_frames, dtype=np.float32),
        "td_times_ns": np.asarray(td_times_ns, dtype=float),
        "grid_x_m": np.asarray(grid_base.x, dtype=float),
        "grid_y_m": np.asarray(grid_base.y, dtype=float),
        "grid_z_m": np.asarray(grid_base.z, dtype=float),
        "n_cells": int(grid_base.Nx * grid_base.Ny * grid_base.Nz),
        "wall_time_s": float(wall_s),
    }


# ---------------------------------------------------------------------------
# Task 1: Resolved 4x4-Cell PEC Wire vs. Subcell Thin-Wire & Uniform Baseline
# ---------------------------------------------------------------------------


def run_task_resolved_wire_vs_subcell_mesh(quick: bool = False) -> dict[str, Any]:
    """Run Task 1: Compare resolved $4\\times 4$-cell PEC wire vs. subcell thin-wire and uniform baseline.

    For $L = 0.48\\lambda_0 = 143.90\\;\\text{mm}$ and wire radius $a = 1.80\\;\\text{mm}$:
    1. Resolved $4\\times 4$-cell PEC prism of half-width $w = a / 1.18 = 1.525\\;\\text{mm}$
       ($\\Delta_{\\perp,\\text{core}} = 0.763\\;\\text{mm}$, 4 cells across $2w$) on a graded mesh.
    2. Subcell thin-wire model (``solver.add_thin_wire(..., radius=a, correction=True)``) on a
       coarse-core graded mesh ($\\Delta_{\\perp,\\text{core}} = 8.46\\;\\text{mm} > 2a$).
    3. Uncorrected single-edge wire (``correction=False``) on the same graded mesh.
    4. Uniform-mesh subcell baseline ($\\Delta = 12.5\\;\\text{mm}$, $\\text{cpw} = 24$).
    5. Subcell wire segmentation convergence sweep $N_{\\text{seg}} \\in \\{9, 13, 17, 21\\}$.
    """
    l_dip = 0.48 * LAM0_M
    a_wire = 1.8e-3
    freqs = np.linspace(0.80e9, 1.15e9, 71)
    n_seg_res = 13 if quick else 17

    # 1. Resolved 4x4-cell PEC wire on locally graded mesh
    grid_res, meta_res = build_resolved_wire_grid(l_dip, a_wire, n_seg=n_seg_res, quick=quick)
    solver_res = wakis.SolverFIT3D(
        grid_res, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    ix_c, iy_c = meta_res["ix_c"], meta_res["iy_c"]
    iz0, iz1, iz_feed = meta_res["iz0"], meta_res["iz1"], meta_res["iz_feed"]
    dx_core = meta_res["dx_core"]
    dz_w = meta_res["dz_w"]

    for k_start, k_stop in [(iz0, iz_feed), (iz_feed + 1, iz1)]:
        solver_res.ieps[ix_c - 2 : ix_c + 3, iy_c - 2 : iy_c + 3, k_start:k_stop, "z"] = 0.0
        solver_res.ieps[ix_c - 2 : ix_c + 2, iy_c - 2 : iy_c + 3, k_start : k_stop + 1, "x"] = 0.0
        solver_res.ieps[ix_c - 2 : ix_c + 3, iy_c - 2 : iy_c + 2, k_start : k_stop + 1, "y"] = 0.0
    solver_res.update_tensors()
    solver_res.step_0 = True

    src = GaussianSource(frequency=F0_HZ, fwidth=FWIDTH_HZ)
    port_res = LumpedPort(
        src_time=src,
        component="z",
        center=(0.0, 0.0, meta_res["z_feed"]),
        size=(4.1 * dx_core, 4.1 * dx_core, 0.5 * dz_w),
        impedance=50.0,
    )
    n_steps_res = int(np.ceil((3.4 if quick else 4.2) / (F0_HZ * solver_res.dt)))
    t0 = time.perf_counter()
    solver_res.emsolve(n_steps_res, source=port_res)
    wall_res = time.perf_counter() - t0
    z_res = port_res.impedance_in(freqs)

    # 2. Subcell thin-wire model on graded mesh with identical axial segmentation
    grid_sub, meta_sub = build_subcell_dipole_grid(l_dip, n_seg=n_seg_res, quick=quick)
    port_sub = LumpedPort(
        src_time=src,
        component="z",
        center=(0.0, 0.0, meta_sub["z_feed"]),
        size=(0.0, 0.0, 0.5 * meta_sub["dz_w"]),
        impedance=50.0,
    )
    solver_sub = wakis.SolverFIT3D(
        grid_sub, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    solver_sub.add_thin_wire(
        [(0.0, 0.0, float(grid_sub.z[meta_sub["iz0"]])), (0.0, 0.0, float(grid_sub.z[meta_sub["iz1"]]))],
        radius=a_wire,
        gap=port_sub,
        correction=True,
    )
    n_steps_sub = int(np.ceil((3.4 if quick else 4.2) / (F0_HZ * solver_sub.dt)))
    t0 = time.perf_counter()
    solver_sub.emsolve(n_steps_sub, source=port_sub)
    wall_sub = time.perf_counter() - t0
    z_sub = port_sub.impedance_in(freqs)

    # 3. Uncorrected wire (correction=False) on the same graded mesh
    grid_unc, _ = build_subcell_dipole_grid(l_dip, n_seg=n_seg_res, quick=quick)
    port_unc = LumpedPort(
        src_time=src,
        component="z",
        center=(0.0, 0.0, meta_sub["z_feed"]),
        size=(0.0, 0.0, 0.5 * meta_sub["dz_w"]),
        impedance=50.0,
    )
    solver_unc = wakis.SolverFIT3D(
        grid_unc, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    solver_unc.add_thin_wire(
        [(0.0, 0.0, float(grid_unc.z[meta_sub["iz0"]])), (0.0, 0.0, float(grid_unc.z[meta_sub["iz1"]]))],
        radius=a_wire,
        gap=port_unc,
        correction=False,
    )
    solver_unc.emsolve(n_steps_sub, source=port_unc)
    z_unc = port_unc.impedance_in(freqs)

    # 4. Uniform mesh subcell baseline (n_seg_u = 11 in quick, 13 in full, uniform dx = dy = dz)
    n_seg_u = 11 if quick else 13
    dz_u = l_dip / float(n_seg_u)
    nx_u = 28 if quick else 36
    nz_u = n_seg_u + (16 if quick else 22)
    grid_unif = wakis.GridFIT3D(
        -0.5 * nx_u * dz_u,
        0.5 * nx_u * dz_u,
        -0.5 * nx_u * dz_u,
        0.5 * nx_u * dz_u,
        -0.5 * nz_u * dz_u,
        0.5 * nz_u * dz_u,
        nx_u,
        nx_u,
        nz_u,
        use_mpi=False,
        verbose=0,
    )
    iz0_u = (nz_u - n_seg_u) // 2
    iz1_u = iz0_u + n_seg_u
    iz_f_u = (iz0_u + iz1_u) // 2
    z_f_u = 0.5 * float(grid_unif.z[iz_f_u] + grid_unif.z[iz_f_u + 1])
    port_unif = LumpedPort(
        src_time=src,
        component="z",
        center=(0.0, 0.0, z_f_u),
        size=(0.0, 0.0, 0.5 * dz_u),
        impedance=50.0,
    )
    solver_unif = wakis.SolverFIT3D(
        grid_unif, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    solver_unif.add_thin_wire(
        [(0.0, 0.0, float(grid_unif.z[iz0_u])), (0.0, 0.0, float(grid_unif.z[iz1_u]))],
        radius=a_wire,
        gap=port_unif,
        correction=True,
    )
    n_steps_unif = int(np.ceil((3.4 if quick else 4.2) / (F0_HZ * solver_unif.dt)))
    t0 = time.perf_counter()
    solver_unif.emsolve(n_steps_unif, source=port_unif)
    wall_unif = time.perf_counter() - t0
    z_unif = port_unif.impedance_in(freqs)

    # 5. Subcell wire segmentation convergence sweep N_seg in {9, 13, 17, 21} for thin wire a = 0.5 mm
    a_conv = 0.5e-3
    n_seg_list = [9, 13, 17] if quick else [9, 13, 17, 21]
    err_l2_seg: list[float] = []
    cells_seg: list[int] = []
    for ns in n_seg_list:
        g_s, m_s = build_subcell_dipole_grid(l_dip, n_seg=ns, quick=quick)
        p_s = LumpedPort(
            src_time=src,
            component="z",
            center=(0.0, 0.0, m_s["z_feed"]),
            size=(0.0, 0.0, 0.5 * m_s["dz_w"]),
            impedance=50.0,
        )
        s_s = wakis.SolverFIT3D(g_s, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
        s_s.add_thin_wire(
            [(0.0, 0.0, float(g_s.z[m_s["iz0"]])), (0.0, 0.0, float(g_s.z[m_s["iz1"]]))],
            radius=a_conv,
            gap=p_s,
            correction=True,
        )
        s_s.emsolve(int(np.ceil((3.4 if quick else 4.2) / (F0_HZ * s_s.dt))), source=p_s)
        z_s = p_s.impedance_in(freqs)
        z_m_s, _, _ = hallen_dipole(l_dip, a_conv, freqs, n_segments=100, gap=m_s["dz_w"])
        err_l2_seg.append(float(np.linalg.norm(z_s - z_m_s) / np.linalg.norm(z_m_s)))
        cells_seg.append(int(g_s.Nx * g_s.Ny * g_s.Nz))

    z_mom_res, _, _ = hallen_dipole(l_dip, a_wire, freqs, n_segments=100, gap=dz_w)
    rel_diff_res_vs_sub = float(np.linalg.norm(z_res - z_sub) / np.linalg.norm(z_res))

    # Cost of uniform resolved 4-cell mesh from wakis.estimate logic
    box_span = float(grid_res.xmax - grid_res.xmin)
    nx_unif_res = int(np.ceil(box_span / dx_core))
    cells_unif_res = int(nx_unif_res**3)

    return {
        "freqs_ghz": freqs * 1e-9,
        "l_dip_m": l_dip,
        "a_wire_mm": a_wire * 1e3,
        "w_half_mm": meta_res["w_half"] * 1e3,
        "dx_core_res_mm": dx_core * 1e3,
        "dx_core_sub_mm": meta_sub["dx_w"] * 1e3,
        "dx_unif_mm": dz_u * 1e3,
        "z_res_real": z_res.real,
        "z_res_imag": z_res.imag,
        "z_sub_real": z_sub.real,
        "z_sub_imag": z_sub.imag,
        "z_unc_real": z_unc.real,
        "z_unc_imag": z_unc.imag,
        "z_unif_real": z_unif.real,
        "z_unif_imag": z_unif.imag,
        "z_mom_real": z_mom_res.real,
        "z_mom_imag": z_mom_res.imag,
        "rel_diff_res_vs_sub": rel_diff_res_vs_sub,
        "cells_res": int(grid_res.Nx * grid_res.Ny * grid_res.Nz),
        "cells_sub": int(grid_sub.Nx * grid_sub.Ny * grid_sub.Nz),
        "cells_unif": int(grid_unif.Nx * grid_unif.Ny * grid_unif.Nz),
        "cells_unif_res_est": cells_unif_res,
        "steps_res": n_steps_res,
        "steps_sub": n_steps_sub,
        "steps_unif": n_steps_unif,
        "wall_res_s": float(wall_res),
        "wall_sub_s": float(wall_sub),
        "wall_unif_s": float(wall_unif),
        "grid_res_x_m": np.asarray(grid_res.x, dtype=float),
        "grid_res_z_m": np.asarray(grid_res.z, dtype=float),
        "grid_sub_x_m": np.asarray(grid_sub.x, dtype=float),
        "grid_sub_z_m": np.asarray(grid_sub.z, dtype=float),
        "grid_unif_x_m": np.asarray(grid_unif.x, dtype=float),
        "n_seg_list": np.asarray(n_seg_list, dtype=int),
        "err_l2_seg": np.asarray(err_l2_seg, dtype=float),
        "cells_seg": np.asarray(cells_seg, dtype=int),
    }


# ---------------------------------------------------------------------------
# Task 2: Quarter-Wave Monopole Over an Infinite PEC Ground Plane (Image Theory)
# ---------------------------------------------------------------------------


def run_task_quarter_wave_monopole_infinite_gnd(quick: bool = False) -> dict[str, Any]:
    """Run Task 2: Quarter-wave monopole ($h = \\lambda_0/4$) over infinite PEC ground vs. $\\lambda_0/2$ dipole.

    Verifies exact image theory on mirror-symmetric 3D FIT grids:
    - $Z_{\\text{in,mono}}(f) = \\frac{1}{2} Z_{\\text{in,dip}}(f)$ across $0.80\\text{–}1.15\\;\\text{GHz}$,
    - Upper-hemisphere directivity $D_{0,\\text{mono}} = 2 D_{0,\\text{dip}} \\approx 3.29 = 5.17\\;\\text{dBi}$
      ($+3.01\\;\\text{dB}$) via ``Near2FarMonitor(..., ground_plane='-z')``,
    - Current distribution $I_{\\text{mono}}(z)$ on $z \\in [0, h]$ matching the upper arm of $I_{\\text{dip}}(z)$.
    """
    n_half = 8 if quick else 10
    a_ref = 1.0e-3
    a_thin = 0.2e-3
    freqs = np.linspace(0.80e9, 1.15e9, 71)
    idx_f0 = int(np.argmin(np.abs(freqs - F0_HZ)))

    grid_dip, grid_mono, meta = build_image_theory_pair_grids(
        H_QUARTER_M, n_half=n_half, quick=quick
    )
    dz_w = meta["dz_w"]
    ix0 = meta["ix0"]
    bx, bz = meta["bx"], meta["bz"]
    src = GaussianSource(frequency=F0_HZ, fwidth=FWIDTH_HZ)

    # 1. Free-space half-wave dipole (L = 2 * h_mono, gap = [-dz_w, +dz_w])
    port_dip = LumpedPort(
        src_time=src,
        component="z",
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 1.5 * dz_w),
        impedance=50.0,
    )
    n2f_dip = Near2FarMonitor(
        frequencies=np.array([F0_HZ]),
        center=(0.0, 0.0, 0.0),
        size=(2.0 * bx, 2.0 * bx, 2.0 * bz),
    )
    solver_dip = wakis.SolverFIT3D(
        grid_dip, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    solver_dip.add_thin_wire(
        [(0.0, 0.0, -H_QUARTER_M), (0.0, 0.0, H_QUARTER_M)],
        radius=a_ref,
        gap=port_dip,
        correction=True,
    )

    n_steps = int(np.ceil((3.6 if quick else 4.5) / (F0_HZ * solver_dip.dt)))
    dt = float(solver_dip.dt)
    w0 = 2.0 * np.pi * F0_HZ

    iz0_d = int(np.argmin(np.abs(grid_dip.z - (-H_QUARTER_M))))
    iz1_d = int(np.argmin(np.abs(grid_dip.z - H_QUARTER_M)))
    k_dip = np.arange(iz0_d, iz1_d, dtype=int)
    z_dip_mid = 0.5 * (grid_dip.z[k_dip] + grid_dip.z[k_dip + 1])
    tlx = float(grid_dip.tL[ix0, ix0, 0, "x"])
    tly = float(grid_dip.tL[ix0, ix0, 0, "y"])
    i_dip_acc = np.zeros(len(k_dip), dtype=complex)

    for step in range(n_steps):
        t = step * dt
        port_dip.update(solver_dip, t)
        solver_dip.one_step()
        port_dip.post_step(solver_dip, t)
        n2f_dip.update(solver_dip, t)
        ph_h = dt * np.exp(-1j * w0 * (t + 0.5 * dt))
        hx_lo = np.asarray(solver_dip.H[ix0, ix0 - 1, k_dip, "x"])
        hx_hi = np.asarray(solver_dip.H[ix0, ix0, k_dip, "x"])
        hy_lo = np.asarray(solver_dip.H[ix0 - 1, ix0, k_dip, "y"])
        hy_hi = np.asarray(solver_dip.H[ix0, ix0, k_dip, "y"])
        i_dip_acc += ph_h * (tlx * (hx_lo - hx_hi) + tly * (hy_hi - hy_lo))

    z_dip = port_dip.impedance_in(freqs)
    v_dip_f0 = complex(port_dip.voltage([F0_HZ])[0])
    i_dip_port_f0 = complex(port_dip.port_current([F0_HZ])[0])
    i_dip_norm = i_dip_acc / i_dip_port_f0
    mid_d = len(k_dip) // 2
    i_dip_norm[mid_d - 1 : mid_d + 1] = 1.0 + 0.0j

    # 2. Quarter-wave monopole on infinite PEC ground plane at z = 0 (gap = [0, dz_w])
    port_mono = LumpedPort(
        src_time=src,
        component="z",
        center=(0.0, 0.0, 0.5 * dz_w),
        size=(0.0, 0.0, 0.5 * dz_w),
        impedance=50.0,
    )
    n2f_mono = Near2FarMonitor(
        frequencies=np.array([F0_HZ]),
        center=(0.0, 0.0, 0.5 * bz),
        size=(2.0 * bx, 2.0 * bx, bz),
        ground_plane="-z",
    )
    solver_mono = wakis.SolverFIT3D(
        grid_mono,
        bc_low=["cpml", "cpml", "pec"],
        bc_high=["cpml", "cpml", "cpml"],
        n_pml=6,
        verbose=0,
    )
    # Register PEC ground plane tangential edges so add_thin_wire recognizes z = 0 as grounded
    solver_mono.add_sheet(plane=("z", 0.0), sigma_s="pec")
    solver_mono.add_thin_wire(
        [(0.0, 0.0, 0.0), (0.0, 0.0, H_QUARTER_M)],
        radius=a_ref,
        gap=port_mono,
        correction=True,
    )

    k_mono = np.arange(0, meta["iz_top_mono"], dtype=int)
    z_mono_mid = 0.5 * (grid_mono.z[k_mono] + grid_mono.z[k_mono + 1])
    i_mono_acc = np.zeros(len(k_mono), dtype=complex)

    for step in range(n_steps):
        t = step * dt
        port_mono.update(solver_mono, t)
        solver_mono.one_step()
        port_mono.post_step(solver_mono, t)
        n2f_mono.update(solver_mono, t)
        ph_h = dt * np.exp(-1j * w0 * (t + 0.5 * dt))
        hx_lo = np.asarray(solver_mono.H[ix0, ix0 - 1, k_mono, "x"])
        hx_hi = np.asarray(solver_mono.H[ix0, ix0, k_mono, "x"])
        hy_lo = np.asarray(solver_mono.H[ix0 - 1, ix0, k_mono, "y"])
        hy_hi = np.asarray(solver_mono.H[ix0, ix0, k_mono, "y"])
        i_mono_acc += ph_h * (tlx * (hx_lo - hx_hi) + tly * (hy_hi - hy_lo))

    z_mono = port_mono.impedance_in(freqs)
    i_mono_port_f0 = complex(port_mono.port_current([F0_HZ])[0])
    i_mono_norm = i_mono_acc / i_mono_port_f0
    i_mono_norm[0] = 1.0 + 0.0j

    # Also run a thinner monopole (a = 0.2 mm) to show closeness to the 36.5 + j21.25 Ohm EMF benchmark
    _, grid_mono_thin, _ = build_image_theory_pair_grids(H_QUARTER_M, n_half=n_half, quick=quick)
    port_mono_thin = LumpedPort(
        src_time=src,
        component="z",
        center=(0.0, 0.0, 0.5 * dz_w),
        size=(0.0, 0.0, 0.5 * dz_w),
        impedance=50.0,
    )
    solver_mono_thin = wakis.SolverFIT3D(
        grid_mono_thin,
        bc_low=["cpml", "cpml", "pec"],
        bc_high=["cpml", "cpml", "cpml"],
        n_pml=6,
        verbose=0,
    )
    solver_mono_thin.add_sheet(plane=("z", 0.0), sigma_s="pec")
    solver_mono_thin.add_thin_wire(
        [(0.0, 0.0, 0.0), (0.0, 0.0, H_QUARTER_M)],
        radius=a_thin,
        gap=port_mono_thin,
        correction=True,
    )
    solver_mono_thin.emsolve(n_steps, source=port_mono_thin)
    z_mono_thin = port_mono_thin.impedance_in(freqs)

    # Far-field directivity comparison: dipole on [0, 180] deg vs monopole on [0, 90] deg
    theta_deg = np.linspace(1.0, 179.0, 179)
    theta_rad = np.deg2rad(theta_deg)
    d_dip_theta = np.asarray(n2f_dip.directivity(theta_rad, np.zeros_like(theta_rad))[0], dtype=float)
    d_mono_theta = np.asarray(n2f_mono.directivity(theta_rad, np.zeros_like(theta_rad))[0], dtype=float)
    d_mono_theta[theta_deg > 90.0] = 0.0
    d_dip_ana = half_wave_analytic_directivity(theta_rad, mono_half_space=False)
    d_mono_ana = half_wave_analytic_directivity(theta_rad, mono_half_space=True)

    d0_dip = float(n2f_dip.max_directivity()[0][0])
    d0_mono = float(n2f_mono.max_directivity()[0][0])
    image_z_rel_err = float(np.max(np.abs(2.0 * z_mono - z_dip) / np.abs(z_dip)))
    z_emf_mono = 0.5 * induced_emf_impedance(L_HALF_M, 1e-6, freqs)

    return {
        "freqs_ghz": freqs * 1e-9,
        "h_mono_m": H_QUARTER_M,
        "l_dip_m": L_HALF_M,
        "a_ref_mm": a_ref * 1e3,
        "a_thin_mm": a_thin * 1e3,
        "z_dip_real": z_dip.real,
        "z_dip_imag": z_dip.imag,
        "z_mono_real": z_mono.real,
        "z_mono_imag": z_mono.imag,
        "z_mono_thin_real": z_mono_thin.real,
        "z_mono_thin_imag": z_mono_thin.imag,
        "z_emf_mono_real": z_emf_mono.real,
        "z_emf_mono_imag": z_emf_mono.imag,
        "s11_dip_50_db": 20.0 * np.log10(np.maximum(np.abs((z_dip - 50.0) / (z_dip + 50.0)), 1e-12)),
        "s11_mono_50_db": 20.0 * np.log10(np.maximum(np.abs((z_mono - 50.0) / (z_mono + 50.0)), 1e-12)),
        "s11_mono_36_db": 20.0 * np.log10(np.maximum(np.abs((z_mono - 36.54) / (z_mono + 36.54)), 1e-12)),
        "image_z_rel_err": image_z_rel_err,
        "z_dip_f0_ohm": complex(z_dip[idx_f0]),
        "z_mono_f0_ohm": complex(z_mono[idx_f0]),
        "z_dip_mid_m": z_dip_mid,
        "i_dip_norm_abs": np.abs(i_dip_norm),
        "i_dip_norm_deg": np.rad2deg(np.angle(i_dip_norm)),
        "z_mono_mid_m": z_mono_mid,
        "i_mono_norm_abs": np.abs(i_mono_norm),
        "i_mono_norm_deg": np.rad2deg(np.angle(i_mono_norm)),
        "theta_deg": theta_deg,
        "d_dip_theta": d_dip_theta,
        "d_mono_theta": d_mono_theta,
        "d_dip_ana": d_dip_ana,
        "d_mono_ana": d_mono_ana,
        "d0_dip": d0_dip,
        "d0_mono": d0_mono,
        "d0_ratio_db": float(10.0 * np.log10(d0_mono / d0_dip)),
        "grid_mono_x_m": np.asarray(grid_mono.x, dtype=float),
        "grid_mono_y_m": np.asarray(grid_mono.y, dtype=float),
        "grid_mono_z_m": np.asarray(grid_mono.z, dtype=float),
    }


# ---------------------------------------------------------------------------
# Task 3: Quarter-Wave Monopole Over a Finite PEC Ground Plane
# ---------------------------------------------------------------------------


def run_task_finite_ground_plane_sweep(quick: bool = False) -> dict[str, Any]:
    """Run Task 3: Quarter-wave monopole centered on a finite PEC ground plane in 6-face CPML.

    Sweeps square ground-plane width $W_{\\text{gnd}} / \\lambda_0 \\in \\{0.40, 0.80, 1.20\\}$ using
    ``solver.add_sheet(center=(0, 0, 0), size=(W_gnd, W_gnd, 0), sigma_s='pec')`` and measures:
    - Upward main-beam tilt $\\theta_{\\max} < 90^\\circ$ due to edge diffraction (Weiner 1987),
    - Horizon directivity $D(90^\\circ)$ and peak directivity $D_{\\max}$,
    - Lower-hemisphere ($\\theta > 90^\\circ$) shadow power fraction,
    - 2D $xz$-plane near-field slice ($\\operatorname{Re} E_z$ and $|\\mathbf{E}|$) for $W_{\\text{gnd}} = 0.80\\lambda_0$.
    """
    n_half = 8 if quick else 10
    dz_w = H_QUARTER_M / float(n_half)
    z_arm = np.linspace(0.0, H_QUARTER_M, n_half + 1)
    n_pad_z = 8 if quick else 10
    z_hi = H_QUARTER_M + np.cumsum(dz_w * np.geomspace(1.0, 2.0, n_pad_z))
    z_pos = np.concatenate([z_arm, z_hi])
    z_full = np.concatenate([-z_pos[:0:-1], z_pos])

    dx_w = dz_w
    n_core_xy = 9 if quick else 12
    xy_core = np.arange(-n_core_xy, n_core_xy + 1, dtype=float) * dx_w
    n_pad_xy = 7 if quick else 9
    xy_pad = xy_core[-1] + np.cumsum(dx_w * np.geomspace(1.1, 2.0, n_pad_xy))
    xy_nodes = np.concatenate([-xy_pad[::-1], xy_core, xy_pad])

    w_over_lam_list = np.array([0.40, 0.80, 1.20], dtype=float)
    freqs = np.linspace(0.80e9, 1.15e9, 71)
    idx_f0 = int(np.argmin(np.abs(freqs - F0_HZ)))
    theta_deg = np.linspace(1.0, 179.0, 179)
    theta_rad = np.deg2rad(theta_deg)
    src = GaussianSource(frequency=F0_HZ, fwidth=FWIDTH_HZ)

    d_cuts = np.zeros((len(w_over_lam_list), len(theta_deg)), dtype=float)
    z_in_all = np.zeros((len(w_over_lam_list), len(freqs)), dtype=complex)
    theta_max_deg = np.zeros(len(w_over_lam_list), dtype=float)
    d_max_lin = np.zeros(len(w_over_lam_list), dtype=float)
    d_horizon_lin = np.zeros(len(w_over_lam_list), dtype=float)
    back_power_frac = np.zeros(len(w_over_lam_list), dtype=float)

    iy0 = int(np.argmin(np.abs(xy_nodes)))
    bx = float(xy_nodes[-7])
    bz = float(z_full[-7])

    for idx_w, w_lam in enumerate(w_over_lam_list):
        w_m = float(w_lam) * LAM0_M
        grid = wakis.GridFIT3D(x=xy_nodes, y=xy_nodes, z=z_full, use_mpi=False, verbose=0)
        solver = wakis.SolverFIT3D(
            grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
        )
        solver.add_sheet(center=(0.0, 0.0, 0.0), size=(w_m, w_m, 0.0), sigma_s="pec")
        port = LumpedPort(
            src_time=src,
            component="z",
            center=(0.0, 0.0, 0.5 * dz_w),
            size=(0.0, 0.0, 0.5 * dz_w),
            impedance=50.0,
        )
        solver.add_thin_wire(
            [(0.0, 0.0, 0.0), (0.0, 0.0, H_QUARTER_M)],
            radius=1.0e-3,
            gap=port,
            correction=True,
        )
        n2f = Near2FarMonitor(
            frequencies=np.array([F0_HZ]),
            center=(0.0, 0.0, 0.0),
            size=(2.0 * bx, 2.0 * bx, 2.0 * bz),
        )
        n_steps = int(np.ceil((3.6 if quick else 4.5) / (F0_HZ * solver.dt)))

        if idx_w == 1:
            # Record 2D xz-plane DFT slice for W_gnd = 0.80 * lam0
            dt = float(solver.dt)
            w0 = 2.0 * np.pi * F0_HZ
            ex_acc = np.zeros((grid.Nx, grid.Nz), dtype=complex)
            ez_acc = np.zeros((grid.Nx, grid.Nz), dtype=complex)
            for step in range(n_steps):
                t = step * dt
                port.update(solver, t)
                solver.one_step()
                port.post_step(solver, t)
                n2f.update(solver, t)
                ph_e = dt * np.exp(-1j * w0 * (t + dt))
                ex_acc += ph_e * np.asarray(solver.E[:, iy0, :, "x"])
                ez_acc += ph_e * np.asarray(solver.E[:, iy0, :, "z"])
            v_f0 = complex(port.voltage([F0_HZ])[0])
            ez_xz_norm = ez_acc / v_f0
            ex_xz_norm = ex_acc / v_f0
            emag_xz = np.sqrt(np.abs(ex_xz_norm) ** 2 + np.abs(ez_xz_norm) ** 2)
            x_cell = 0.5 * (grid.x[:-1] + grid.x[1:])
            z_cell = 0.5 * (grid.z[:-1] + grid.z[1:])
        else:
            solver.emsolve(n_steps, source=port, monitors=[n2f])

        z_in = port.impedance_in(freqs)
        z_in_all[idx_w] = z_in
        d_cut = np.asarray(n2f.directivity(theta_rad, np.zeros_like(theta_rad))[0], dtype=float)
        u_cut = np.asarray(n2f.radiation_intensity(theta_rad, np.zeros_like(theta_rad))[0], dtype=float)
        d_cuts[idx_w] = d_cut
        idx_pk = int(np.argmax(d_cut))
        theta_max_deg[idx_w] = float(theta_deg[idx_pk])
        d_max_lin[idx_w] = float(d_cut[idx_pk])
        d_horizon_lin[idx_w] = float(d_cut[int(np.argmin(np.abs(theta_deg - 90.0)))])

        mask_low = theta_deg > 90.0
        p_low = float(np.trapezoid(u_cut[mask_low] * np.sin(theta_rad[mask_low]), theta_rad[mask_low]))
        p_tot = float(np.trapezoid(u_cut * np.sin(theta_rad), theta_rad))
        back_power_frac[idx_w] = p_low / max(p_tot, 1e-30)

    return {
        "w_over_lam_list": w_over_lam_list,
        "w_gnd_m_list": w_over_lam_list * LAM0_M,
        "freqs_ghz": freqs * 1e-9,
        "z_in_real": z_in_all.real,
        "z_in_imag": z_in_all.imag,
        "z_f0_real": z_in_all[:, idx_f0].real,
        "z_f0_imag": z_in_all[:, idx_f0].imag,
        "theta_deg": theta_deg,
        "d_cuts": d_cuts,
        "theta_max_deg": theta_max_deg,
        "d_max_lin": d_max_lin,
        "d_max_dbi": 10.0 * np.log10(np.maximum(d_max_lin, 1e-6)),
        "d_horizon_lin": d_horizon_lin,
        "back_power_frac": back_power_frac,
        "x_cell_m": x_cell,
        "z_cell_m": z_cell,
        "ez_xz_real": ez_xz_norm.real,
        "emag_xz": emag_xz,
        "grid_fg_x_m": xy_nodes,
        "grid_fg_y_m": xy_nodes,
        "grid_fg_z_m": z_full,
        "bx_m": bx,
        "bz_m": bz,
    }


def _save_task(data_dir: Path, task_name: str, payload: dict[str, Any]) -> None:
    """Save a task dictionary to ``data_dir/task_<name>.npz`` and a JSON summary."""
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / f"task_{task_name}.npz"
    json_path = data_dir / f"task_{task_name}.json"

    arrays: dict[str, Any] = {}
    summary: dict[str, Any] = {"task": task_name}
    for k, v in payload.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, (float, int, np.floating, np.integer)):
            arrays[k] = np.asarray(v)
            summary[k] = float(v) if isinstance(v, (float, np.floating)) else int(v)
        elif isinstance(v, complex):
            arrays[k + "_real"] = np.asarray(v.real)
            arrays[k + "_imag"] = np.asarray(v.imag)
            summary[k] = [float(v.real), float(v.imag)]
        else:
            arrays[k] = np.asarray(v)

    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(summary, indent=2))


def run_tasks(task_names: list[str], data_dir: Path, quick: bool = False) -> None:
    """Execute the requested Tutorial 4.2 simulation tasks and save ``.npz`` + ``.json`` archives."""
    dispatch = {
        "half_wave_thin_wire_radii": run_task_half_wave_thin_wire_radii,
        "resolved_wire_vs_subcell_mesh": run_task_resolved_wire_vs_subcell_mesh,
        "quarter_wave_monopole_infinite_gnd": run_task_quarter_wave_monopole_infinite_gnd,
        "finite_ground_plane_sweep": run_task_finite_ground_plane_sweep,
    }
    for name in task_names:
        if name not in dispatch:
            raise ValueError(f"Unknown task {name!r}. Valid tasks: {TASKS}")
        t0 = time.perf_counter()
        res = dispatch[name](quick=quick)
        elapsed = time.perf_counter() - t0
        _save_task(data_dir, name, res)
        print(f"  [OK] Task {name} completed in {elapsed:.1f}s -> data/task_{name}.npz")


# ---------------------------------------------------------------------------
# Figure Builders (figstyle + schematic)
# ---------------------------------------------------------------------------


def _build_dipole_schematic_objects(d0: Any) -> tuple[Any, Any, Any, list[Any], dict[int, str]]:
    """Build lightweight solver, port, Near2FarMonitor, and PEC arm markers for dipole schematics."""
    gx = np.asarray(d0["grid_x_m"], dtype=float)
    gy = np.asarray(d0["grid_y_m"], dtype=float)
    gz = np.asarray(d0["grid_z_m"], dtype=float)
    grid = wakis.GridFIT3D(x=gx, y=gy, z=gz, use_mpi=False, verbose=0)
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    dz_w = L_HALF_M / 19.0
    src = GaussianSource(frequency=F0_HZ, fwidth=FWIDTH_HZ)
    port = LumpedPort(
        src_time=src,
        component="z",
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, dz_w),
        impedance=50.0,
    )
    ix0 = int(np.argmin(np.abs(gx)))
    iz1 = int(np.argmin(np.abs(gz - 0.5 * L_HALF_M)))
    bx = float(gx[min(len(gx) - 2, ix0 + 6)])
    bz = float(gz[min(len(gz) - 2, iz1 + 3)])
    n2f = Near2FarMonitor(
        frequencies=np.array([F0_HZ]),
        center=(0.0, 0.0, 0.0),
        size=(2.0 * bx, 2.0 * bx, 2.0 * bz),
    )
    w_vis = 0.010
    arm_len = 0.5 * L_HALF_M - 0.5 * dz_w
    z_arm_c = 0.5 * (0.5 * dz_w + 0.5 * L_HALF_M)
    arm_lo = Box((0.0, 0.0, -z_arm_c), (w_vis, w_vis, arm_len), "pec")
    arm_hi = Box((0.0, 0.0, z_arm_c), (w_vis, w_vis, arm_len), "pec")
    mat_names = {id(arm_lo): "PEC dipole arm", id(arm_hi): "PEC dipole arm"}
    return solver, port, n2f, [arm_lo, arm_hi], mat_names


def _build_monopole_schematic_objects(d2: Any, d3: Any, finite_gnd: bool = False) -> tuple[Any, Any, Any, list[Any], dict[int, str]]:
    """Build lightweight solver, port, Near2FarMonitor, and PEC markers for monopole schematics."""
    if not finite_gnd:
        gx = np.asarray(d2["grid_mono_x_m"], dtype=float)
        gy = np.asarray(d2["grid_mono_y_m"], dtype=float)
        gz = np.asarray(d2["grid_mono_z_m"], dtype=float)
        grid = wakis.GridFIT3D(x=gx, y=gy, z=gz, use_mpi=False, verbose=0)
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["cpml", "cpml", "pec"],
            bc_high=["cpml", "cpml", "cpml"],
            n_pml=6,
            verbose=0,
        )
        dz_w = H_QUARTER_M / 10.0
        src = GaussianSource(frequency=F0_HZ, fwidth=FWIDTH_HZ)
        port = LumpedPort(
            src_time=src,
            component="z",
            center=(0.0, 0.0, 0.5 * dz_w),
            size=(0.0, 0.0, 0.5 * dz_w),
            impedance=50.0,
        )
        ix0 = int(np.argmin(np.abs(gx)))
        iz_top = int(np.argmin(np.abs(gz - H_QUARTER_M)))
        bx = float(gx[min(len(gx) - 2, ix0 + 6)])
        bz = float(gz[min(len(gz) - 2, iz_top + 3)])
        n2f = Near2FarMonitor(
            frequencies=np.array([F0_HZ]),
            center=(0.0, 0.0, 0.5 * bz),
            size=(2.0 * bx, 2.0 * bx, bz),
            ground_plane="-z",
        )
        w_vis = 0.010
        arm_len = H_QUARTER_M - dz_w
        arm = Box((0.0, 0.0, dz_w + 0.5 * arm_len), (w_vis, w_vis, arm_len), "pec")
        return solver, port, n2f, [arm], {id(arm): "PEC monopole (lambda/4)"}

    gx = np.asarray(d3["grid_fg_x_m"], dtype=float)
    gy = np.asarray(d3["grid_fg_y_m"], dtype=float)
    gz = np.asarray(d3["grid_fg_z_m"], dtype=float)
    grid = wakis.GridFIT3D(x=gx, y=gy, z=gz, use_mpi=False, verbose=0)
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    dz_w = H_QUARTER_M / 10.0
    src = GaussianSource(frequency=F0_HZ, fwidth=FWIDTH_HZ)
    port = LumpedPort(
        src_time=src,
        component="z",
        center=(0.0, 0.0, 0.5 * dz_w),
        size=(0.0, 0.0, 0.5 * dz_w),
        impedance=50.0,
    )
    bx = float(d3["bx_m"])
    bz = float(d3["bz_m"])
    n2f = Near2FarMonitor(
        frequencies=np.array([F0_HZ]),
        center=(0.0, 0.0, 0.0),
        size=(2.0 * bx, 2.0 * bx, 2.0 * bz),
    )
    w_gnd = 0.80 * LAM0_M
    w_vis = 0.010
    arm_len = H_QUARTER_M - dz_w
    arm = Box((0.0, 0.0, dz_w + 0.5 * arm_len), (w_vis, w_vis, arm_len), "pec")
    gnd_plate = Box((0.0, 0.0, -0.002), (w_gnd, w_gnd, 0.004), "pec")
    mat_names = {id(arm): "PEC monopole (lambda/4)", id(gnd_plate): "Finite PEC ground (0.8 lambda0)"}
    return solver, port, n2f, [arm, gnd_plate], mat_names


def plot_00_setup_schematic(d0: Any, d2: Any, d3: Any, fig_dir: Path) -> Path:
    """Render Figure 00: 3-panel setup schematic for the three wire-antenna simulation configurations."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=11.0, height_in=4.5, ncols=3, nrows=1)
    ax_dip, ax_mono_inf, ax_mono_fin = axes

    s_dip, p_dip, n2f_dip, st_dip, names_dip = _build_dipole_schematic_objects(d0)
    schematic.plot_schematic(
        s_dip,
        plane="xz",
        ax=ax_dip,
        structures=st_dip,
        material_names=names_dip,
        sources=[p_dip],
        monitors=[n2f_dip],
        show_mesh=False,
        show_pulse_inset=False,
        freq=F0_HZ,
        unit="mm",
        title=r"(a) Free-Space $\lambda_0/2$ Dipole",
    )

    s_mi, p_mi, n2f_mi, st_mi, names_mi = _build_monopole_schematic_objects(d2, d3, finite_gnd=False)
    schematic.plot_schematic(
        s_mi,
        plane="xz",
        ax=ax_mono_inf,
        structures=st_mi,
        material_names=names_mi,
        sources=[p_mi],
        monitors=[n2f_mi],
        show_mesh=False,
        show_pulse_inset=False,
        freq=F0_HZ,
        unit="mm",
        title=r"(b) $\lambda_0/4$ Monopole (Infinite PEC)",
    )

    s_mf, p_mf, n2f_mf, st_mf, names_mf = _build_monopole_schematic_objects(d2, d3, finite_gnd=True)
    schematic.plot_schematic(
        s_mf,
        plane="xz",
        ax=ax_mono_fin,
        structures=st_mf,
        material_names=names_mf,
        sources=[p_mf],
        monitors=[n2f_mf],
        show_mesh=False,
        show_pulse_inset=False,
        freq=F0_HZ,
        unit="mm",
        title=r"(c) $\lambda_0/4$ Monopole (Finite PEC Sheet)",
    )

    out = fig_dir / "00_setup_schematic.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_01_geometry_and_graded_mesh(d0: Any, d1: Any, fig_dir: Path) -> Path:
    """Render Figure 01: Graded mesh in xz plane, zoomed xy resolved-vs-subcell core, and step profiles."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=10.6, height_in=4.4, ncols=3, nrows=1)
    ax_xz, ax_xy, ax_step = axes

    s_dip, p_dip, n2f_dip, st_dip, names_dip = _build_dipole_schematic_objects(d0)
    schematic.draw_mesh(
        ax_xz,
        s_dip,
        plane="xz",
        unit="mm",
        every=1,
        freq=F0_HZ,
        zoom_inset=False,
        show_cpw=True,
    )
    schematic.overlay_geometry(
        ax_xz,
        s_dip,
        plane="xz",
        unit="mm",
        structures=st_dip,
        material_names=names_dip,
        sources=[p_dip],
        monitors=[n2f_dip],
        show_sources=True,
        show_monitors=False,
    )
    ax_xz.set_aspect("equal")
    ax_xz.set_xlabel("$x$ [mm]")
    ax_xz.set_ylabel("$z$ [mm]")
    ax_xz.set_title("(a) Graded $xz$ Mesh Around Wire", pad=10)

    # Panel (b): Zoomed xy cross-section comparing 4x4-cell resolved PEC wire vs subcell thin-wire loop
    gx_res_mm = np.asarray(d1["grid_res_x_m"], dtype=float) * 1e3
    w_half_mm = float(d1["w_half_mm"])
    a_wire_mm = float(d1["a_wire_mm"])
    dx_sub_mm = float(d1["dx_core_sub_mm"])

    for x_v in gx_res_mm:
        if abs(x_v) <= 10.0:
            ax_xy.axvline(x_v, color=COLORS["grey"], lw=0.55, alpha=0.55)
            ax_xy.axhline(x_v, color=COLORS["grey"], lw=0.55, alpha=0.55)

    rect_res = Rectangle(
        (-w_half_mm, -w_half_mm),
        2.0 * w_half_mm,
        2.0 * w_half_mm,
        facecolor=COLORS["blue"],
        edgecolor=figstyle.INK,
        alpha=0.35,
        lw=1.3,
        label=f"Resolved $4\\times 4$ PEC ($2w={2*w_half_mm:.1f}$ mm)",
    )
    ax_xy.add_patch(rect_res)
    circ_wire = Circle(
        (0.0, 0.0),
        a_wire_mm,
        fill=False,
        edgecolor=COLORS["red"],
        ls="-",
        lw=1.8,
        label=f"Circular wire $a={a_wire_mm:.1f}$ mm",
    )
    ax_xy.add_patch(circ_wire)
    rect_sub = Rectangle(
        (-0.5 * dx_sub_mm, -0.5 * dx_sub_mm),
        dx_sub_mm,
        dx_sub_mm,
        fill=False,
        edgecolor=COLORS["teal"],
        ls="--",
        lw=1.6,
        label=f"Subcell $H$-loop ($\\Delta_\\perp={dx_sub_mm:.1f}$ mm)",
    )
    ax_xy.add_patch(rect_sub)
    ax_xy.set_xlim(-6.5, 6.5)
    ax_xy.set_ylim(-6.5, 6.5)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("$x$ [mm]")
    ax_xy.set_ylabel("$y$ [mm]")
    ax_xy.set_title("(b) Transverse $xy$ Wire Core", pad=10)
    figstyle.place_legend_below(ax_xy, ncol=1, y_offset=-0.28)

    # Panel (c): Transverse cell size profile dx(x) for Resolved Graded, Subcell Graded, and Uniform
    xc_res = 0.5 * (gx_res_mm[:-1] + gx_res_mm[1:])
    dxc_res = np.diff(gx_res_mm)
    gx_sub_mm = np.asarray(d1["grid_sub_x_m"], dtype=float) * 1e3
    xc_sub = 0.5 * (gx_sub_mm[:-1] + gx_sub_mm[1:])
    dxc_sub = np.diff(gx_sub_mm)
    dx_unif_mm = float(d1["dx_unif_mm"])

    ax_step.semilogy(xc_res, dxc_res, "o-", color=COLORS["blue"], ms=3.2, lw=1.5, label="Resolved graded $\\Delta x(x)$")
    ax_step.semilogy(xc_sub, dxc_sub, "s-", color=COLORS["teal"], ms=3.4, lw=1.5, label="Subcell graded $\\Delta x(x)$")
    ax_step.axhline(dx_unif_mm, color=COLORS["orange"], ls="--", lw=1.5, label=f"Uniform baseline ($\\Delta={dx_unif_mm:.1f}$ mm)")
    ax_step.set_xlabel("Transverse coordinate $x$ [mm]")
    ax_step.set_ylabel(r"Cell width $\Delta x$ [mm]")
    ax_step.set_title("(c) Non-Uniform Cell Grading", pad=10)
    figstyle.place_legend_below(ax_step, ncol=1, y_offset=-0.28)

    out = fig_dir / "01_geometry_and_graded_mesh.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_02_current_distribution_wire(d0: Any, d2: Any, fig_dir: Path) -> Path:
    """Render Figure 02: Wire current distribution I(z) along dipole and monopole vs. sinusoidal & MoM."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=10.4, height_in=4.4, ncols=3, nrows=1)
    ax_mag, ax_ph, ax_mono = axes

    z_seg_mm = np.asarray(d0["z_seg_mid_m"], dtype=float) * 1e3
    i_seg = np.asarray(d0["i_seg_per_v_real"], dtype=float) + 1j * np.asarray(d0["i_seg_per_v_imag"], dtype=float)
    i_feed = i_seg[len(i_seg) // 2]
    i_seg_norm = i_seg / i_feed

    z_mom_mm = np.asarray(d0["z_nodes_mom_m"], dtype=float) * 1e3
    i_mom = np.asarray(d0["i_mom_per_v_real"], dtype=float) + 1j * np.asarray(d0["i_mom_per_v_imag"], dtype=float)
    i_mom_norm = i_mom / i_mom[len(i_mom) // 2]

    z_sin_m = np.linspace(-0.5 * L_HALF_M, 0.5 * L_HALF_M, 201)
    i_sin_norm = sinusoidal_current_distribution(z_sin_m, L_HALF_M, F0_HZ, i_feed=1.0 + 0.0j)

    # Panel (a): Normalized current magnitude |I(z) / I(0)| along lambda/2 dipole
    ax_mag.plot(z_sin_m * 1e3, np.abs(i_sin_norm), "--", color=COLORS["orange"], lw=1.7, label=r"Sinusoidal $\sin[k(L/2-|z|)]$")
    ax_mag.plot(z_mom_mm, np.abs(i_mom_norm), color=COLORS["blue"], lw=1.8, label="Hallén MoM ($a=0.5$ mm)")
    ax_mag.plot(
        z_seg_mm,
        np.abs(i_seg_norm),
        "o",
        color=COLORS["teal"],
        ms=4.2,
        label=r"3D FIT $\oint \mathbf{H}\cdot\mathrm{d}\tilde{\mathbf{s}} / I(0)$",
    )
    ax_mag.set_xlim(-80.0, 80.0)
    ax_mag.set_ylim(0.0, 1.12)
    ax_mag.set_xlabel("Position along wire $z$ [mm]")
    ax_mag.set_ylabel(r"Normalized current $|I(z) / I(0)|$")
    ax_mag.set_title(r"(a) $\lambda_0/2$ Dipole Current $|I(z)|$", pad=10)
    figstyle.place_legend_below(ax_mag, ncol=1, y_offset=-0.28)

    # Panel (b): Current phase progression angle(I(z)/I(0)) [deg]
    mask_m = np.abs(i_mom_norm) > 0.08
    ph_mom_deg = np.rad2deg(np.angle(i_mom_norm[mask_m]))
    ph_seg_deg = np.rad2deg(np.angle(i_seg_norm))
    ax_ph.axhline(0.0, color=COLORS["orange"], ls="--", lw=1.6, label="Sinusoidal standing wave ($0^\\circ$)")
    ax_ph.plot(z_mom_mm[mask_m], ph_mom_deg, color=COLORS["blue"], lw=1.8, label="Hallén MoM phase")
    ax_ph.plot(z_seg_mm, ph_seg_deg, "o", color=COLORS["teal"], ms=4.2, label="3D FIT circulation phase")
    ax_ph.set_xlim(-76.0, 76.0)
    ax_ph.set_xlabel("Position along wire $z$ [mm]")
    ax_ph.set_ylabel(r"Phase relative to feed $\angle[I(z)/I(0)]$ [deg]")
    ax_ph.set_title("(b) Finite-Radius Phase Shift", pad=10)
    figstyle.place_legend_below(ax_ph, ncol=1, y_offset=-0.28)

    # Panel (c): Quarter-wave monopole current |I_mono(z)| vs upper arm of half-wave dipole
    z_mono_mm = np.asarray(d2["z_mono_mid_m"], dtype=float) * 1e3
    i_mono_abs = np.asarray(d2["i_mono_norm_abs"], dtype=float)
    z_dip2_mm = np.asarray(d2["z_dip_mid_m"], dtype=float) * 1e3
    i_dip2_abs = np.asarray(d2["i_dip_norm_abs"], dtype=float)
    mask_up = z_dip2_mm >= 0.0

    ax_mono.plot(
        z_sin_m[z_sin_m >= 0] * 1e3,
        np.abs(i_sin_norm[z_sin_m >= 0]),
        "--",
        color=COLORS["orange"],
        lw=1.6,
        label=r"Sinusoidal $\cos(kz)$ ($h=\lambda_0/4$)",
    )
    ax_mono.plot(
        z_dip2_mm[mask_up],
        i_dip2_abs[mask_up],
        "s-",
        color=COLORS["blue"],
        ms=4.2,
        lw=1.6,
        label=r"$\lambda_0/2$ dipole upper arm ($z \geq 0$)",
    )
    ax_mono.plot(
        z_mono_mm,
        i_mono_abs,
        "o",
        color=COLORS["red"],
        ms=4.2,
        label=r"$\lambda_0/4$ monopole on PEC ($z \geq 0$)",
    )
    ax_mono.set_xlim(0.0, 78.0)
    ax_mono.set_ylim(0.0, 1.12)
    ax_mono.set_xlabel("Height above ground $z$ [mm]")
    ax_mono.set_ylabel(r"Normalized current $|I(z) / I(0)|$")
    ax_mono.set_title(r"(c) Monopole vs. Dipole Arm $I(z)$", pad=10)
    figstyle.place_legend_below(ax_mono, ncol=1, y_offset=-0.28)

    out = fig_dir / "02_current_distribution_wire.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_03_impedance_s11_and_shortening(d0: Any, fig_dir: Path) -> Path:
    """Render Figure 03: Input impedance Z_in(f), S11(f), and resonant length shortening vs radius."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=10.4, height_in=4.5, ncols=3, nrows=1)
    ax_z, ax_s11, ax_res = axes

    f_ghz = np.asarray(d0["freqs_ghz"], dtype=float)
    r_mm = np.asarray(d0["radii_mm"], dtype=float)
    omegas = np.asarray(d0["omegas"], dtype=float)
    z_fit_r = np.asarray(d0["z_fit_real"], dtype=float)
    z_fit_i = np.asarray(d0["z_fit_imag"], dtype=float)
    z_mom_r = np.asarray(d0["z_mom_real"], dtype=float)
    z_mom_i = np.asarray(d0["z_mom_imag"], dtype=float)
    z_emf_r = np.asarray(d0["z_emf_real"], dtype=float)
    z_emf_i = np.asarray(d0["z_emf_imag"], dtype=float)

    # Panel (a): R_in(f) and X_in(f) for thick (a=1.5 mm) and thin (a=0.15 mm) wires + Induced-EMF 73+j42.5 Ohm
    ax_z.plot(f_ghz, z_mom_r[0], color=COLORS["blue"], lw=1.7, label=f"$R_{{\\mathrm{{in}}}}$ MoM ($a={r_mm[0]:.1f}$ mm)")
    ax_z.plot(f_ghz[::4], z_fit_r[0, ::4], "o", color=COLORS["blue"], ms=3.6, label=f"$R_{{\\mathrm{{in}}}}$ FIT ($a={r_mm[0]:.1f}$ mm)")
    ax_z.plot(f_ghz, z_mom_i[0], "--", color=COLORS["teal"], lw=1.6, label=f"$X_{{\\mathrm{{in}}}}$ MoM ($a={r_mm[0]:.1f}$ mm)")
    ax_z.plot(f_ghz[::4], z_fit_i[0, ::4], "s", color=COLORS["teal"], ms=3.4, label=f"$X_{{\\mathrm{{in}}}}$ FIT ($a={r_mm[0]:.1f}$ mm)")
    ax_z.plot(f_ghz, z_mom_i[2], ":", color=COLORS["purple"], lw=1.6, label=f"$X_{{\\mathrm{{in}}}}$ MoM ($a={r_mm[2]:.2f}$ mm)")
    ax_z.plot(f_ghz[::4], z_fit_i[2, ::4], "^", color=COLORS["purple"], ms=3.4, label=f"$X_{{\\mathrm{{in}}}}$ FIT ($a={r_mm[2]:.2f}$ mm)")
    ax_z.plot([1.0], [73.08], "*", color=COLORS["red"], ms=8.5, label=r"EMF $L=\lambda_0/2$: $73.1\,\Omega$")
    ax_z.plot([1.0], [42.51], "D", color=COLORS["orange"], ms=5.5, label=r"EMF $L=\lambda_0/2$: $+j42.5\,\Omega$")
    ax_z.axhline(0.0, color=COLORS["grey"], ls=":", lw=0.9)
    ax_z.set_xlabel("Frequency $f$ [GHz]")
    ax_z.set_ylabel(r"Input impedance $R_{\mathrm{in}}, X_{\mathrm{in}}$ [$\Omega$]")
    ax_z.set_title(r"(a) Dipole $Z_{\mathrm{in}}(f)$ vs. Radius", pad=10)
    figstyle.place_legend_below(ax_z, ncol=2, y_offset=-0.28)

    # Panel (b): Reflection coefficient |S11(f)| [dB] for L = 0.50*lam0 vs resonance-trimmed L = 0.470*lam0
    s11_50 = np.asarray(d0["s11_50_db"], dtype=float)
    s11_73 = np.asarray(d0["s11_73_db"], dtype=float)
    s11_tr_50 = np.asarray(d0["s11_trim_50_db"], dtype=float)
    s11_tr_73 = np.asarray(d0["s11_trim_73_db"], dtype=float)

    ax_s11.plot(f_ghz, s11_50[1], color=COLORS["blue"], lw=1.7, label=r"$L=0.50\lambda_0$ ($Z_0=50\,\Omega$)")
    ax_s11.plot(f_ghz, s11_73[1], "--", color=COLORS["teal"], lw=1.6, label=r"$L=0.50\lambda_0$ ($Z_0=73\,\Omega$)")
    ax_s11.plot(f_ghz, s11_tr_50, color=COLORS["orange"], lw=1.8, label=r"Trimmed $L=0.47\lambda_0$ ($50\,\Omega$)")
    ax_s11.plot(f_ghz, s11_tr_73, "-.", color=COLORS["red"], lw=1.7, label=r"Trimmed $L=0.47\lambda_0$ ($73\,\Omega$)")
    ax_s11.axhline(-10.0, color=COLORS["grey"], ls=":", lw=1.0, label="$-10$ dB match threshold")
    ax_s11.axvline(1.0, color=COLORS["grey"], ls="--", lw=0.9)
    ax_s11.set_ylim(-30.0, 0.0)
    ax_s11.set_xlabel("Frequency $f$ [GHz]")
    ax_s11.set_ylabel(r"$20\log_{10}|S_{11}(f)|$ [dB]")
    ax_s11.set_title(r"(b) Return Loss & Length Trimming", pad=10)
    figstyle.place_legend_below(ax_s11, ncol=1, y_offset=-0.28)

    # Panel (c): Resonant electrical length L_res / lam0 (X_in = 0) vs thickness parameter Omega = 2 ln(L/a)
    omega_c = np.asarray(d0["omega_curve"], dtype=float)
    l_res_c = np.asarray(d0["l_res_mom_curve"], dtype=float)
    l_res_fit = np.asarray(d0["l_res_fit_over_lam0"], dtype=float)
    l_res_mom = np.asarray(d0["l_res_mom_over_lam0"], dtype=float)

    ax_res.axhline(0.50, color=COLORS["orange"], ls="--", lw=1.5, label=r"Infinitely thin limit $0.50\lambda_0$")
    ax_res.plot(omega_c, l_res_c, color=COLORS["blue"], lw=1.8, label=r"Hallén–King–Middleton $L_{\mathrm{res}}(\Omega)$")
    ax_res.plot(omegas, l_res_mom, "s", color=COLORS["blue"], ms=5.0, label="Hallén MoM at test radii")
    ax_res.plot(omegas, l_res_fit, "o", color=COLORS["red"], ms=5.5, label="3D FIT subcell thin wire")
    ax_res.set_xlim(8.3, 16.2)
    ax_res.set_ylim(0.445, 0.508)
    ax_res.set_xlabel(r"Thickness parameter $\Omega = 2\ln(L/a)$")
    ax_res.set_ylabel(r"Resonant length $L_{\mathrm{res}} / \lambda_0$ ($X_{\mathrm{in}}=0$)")
    ax_res.set_title("(c) Resonant Length Shortening", pad=10)
    figstyle.place_legend_below(ax_res, ncol=1, y_offset=-0.28)

    out = fig_dir / "03_impedance_s11_and_shortening.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_04_subcell_vs_resolved_convergence(d1: Any, fig_dir: Path) -> Path:
    """Render Figure 04: Resolved 4x4-cell PEC wire vs subcell thin-wire, uniform baseline, and N_seg convergence."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=10.4, height_in=4.5, ncols=3, nrows=1)
    ax_imp, ax_conv, ax_cost = axes

    f_ghz = np.asarray(d1["freqs_ghz"], dtype=float)
    zr_r = np.asarray(d1["z_res_real"], dtype=float)
    zr_i = np.asarray(d1["z_res_imag"], dtype=float)
    zs_r = np.asarray(d1["z_sub_real"], dtype=float)
    zs_i = np.asarray(d1["z_sub_imag"], dtype=float)
    zu_r = np.asarray(d1["z_unif_real"], dtype=float)
    zu_i = np.asarray(d1["z_unif_imag"], dtype=float)
    zunc_i = np.asarray(d1["z_unc_imag"], dtype=float)

    # Panel (a): Resolved 4x4 PEC wire vs Subcell Thin-Wire (Graded & Uniform) vs Uncorrected
    ax_imp.plot(f_ghz, zr_r, color=COLORS["blue"], lw=1.8, label=r"$R_{\mathrm{in}}$ Resolved $4\times 4$ PEC")
    ax_imp.plot(f_ghz[::3], zs_r[::3], "o", color=COLORS["teal"], ms=3.8, label=r"$R_{\mathrm{in}}$ Subcell graded")
    ax_imp.plot(f_ghz[::4], zu_r[::4], "s", color=COLORS["orange"], ms=3.5, label=r"$R_{\mathrm{in}}$ Subcell uniform")
    ax_imp.plot(f_ghz, zr_i, "--", color=COLORS["blue"], lw=1.7, label=r"$X_{\mathrm{in}}$ Resolved $4\times 4$ PEC")
    ax_imp.plot(f_ghz[::3], zs_i[::3], "^", color=COLORS["teal"], ms=3.8, label=r"$X_{\mathrm{in}}$ Subcell graded")
    ax_imp.plot(f_ghz, zunc_i, ":", color=COLORS["red"], lw=1.5, label=r"$X_{\mathrm{in}}$ Uncorrected ($\ln(\Delta/a)$ off)")
    ax_imp.axhline(0.0, color=COLORS["grey"], ls=":", lw=0.9)
    ax_imp.set_xlabel("Frequency $f$ [GHz]")
    ax_imp.set_ylabel(r"Input impedance $R_{\mathrm{in}}, X_{\mathrm{in}}$ [$\Omega$]")
    ax_imp.set_title("(a) Subcell vs. Resolved Wire $Z_{\\mathrm{in}}$", pad=10)
    figstyle.place_legend_below(ax_imp, ncol=2, y_offset=-0.28)

    # Panel (b): Grid convergence of subcell Z_in vs wire segmentation N_seg
    n_seg = np.asarray(d1["n_seg_list"], dtype=float)
    err_seg = np.asarray(d1["err_l2_seg"], dtype=float) * 100.0
    ref_o1 = err_seg[0] * (n_seg / n_seg[0]) ** -1.0

    ax_conv.loglog(n_seg, err_seg, "o-", color=COLORS["blue"], ms=4.8, lw=1.8, label=r"$\|Z_{\mathrm{FIT}} - Z_{\mathrm{MoM}}\|_2 / \|Z_{\mathrm{MoM}}\|_2$")
    ax_conv.loglog(n_seg, ref_o1, "--", color=COLORS["grey"], lw=1.3, label=r"$\mathcal{O}(N_{\mathrm{seg}}^{-1})$ reference")
    ax_conv.set_xticks([9, 13, 17, 21])
    ax_conv.set_xticklabels(["9", "13", "17", "21"])
    ax_conv.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_conv.set_xlabel(r"Axial wire segments $N_{\mathrm{seg}}$")
    ax_conv.set_ylabel("Band-integrated impedance error [%]")
    ax_conv.set_title("(b) Axial Segmentation Convergence", pad=10)
    figstyle.place_legend_below(ax_conv, ncol=1, y_offset=-0.28)

    # Panel (c): Cell count and runtime comparison across meshing strategies
    labels = ["Unif.\nResolved\n(est.)", "Unif.\nSubcell", "Graded\nResolved\n(4 cells)", "Graded\nSubcell\n(F3)"]
    cells = np.array(
        [
            float(d1["cells_unif_res_est"]),
            float(d1["cells_unif"]),
            float(d1["cells_res"]),
            float(d1["cells_sub"]),
        ]
    )
    x_pos = np.arange(len(labels))
    bars = ax_cost.bar(
        x_pos,
        cells,
        width=0.56,
        color=[COLORS["red"], COLORS["orange"], COLORS["blue"], COLORS["teal"]],
        alpha=0.85,
    )
    ax_cost.set_yscale("log")
    ax_cost.set_xticks(x_pos)
    ax_cost.set_xticklabels(labels, fontsize=8.0)
    ax_cost.set_ylim(1e4, 2e9)
    ax_cost.set_ylabel("Total 3D grid cells (log scale)")
    ax_cost.set_title(
        f"(c) Mesh Cost ({d1['wall_sub_s']:.1f}s vs. {d1['wall_res_s']:.1f}s)",
        pad=10,
    )
    for b_patch, c_val in zip(bars, cells):
        txt = f"{c_val/1e6:.0f}M" if c_val >= 1e6 else f"{c_val/1e3:.0f}k"
        ax_cost.text(
            b_patch.get_x() + 0.5 * b_patch.get_width(),
            c_val * 1.45,
            txt,
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=figstyle.INK,
        )

    out = fig_dir / "04_subcell_vs_resolved_convergence.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_05_monopole_vs_dipole_image_theory(d2: Any, fig_dir: Path) -> Path:
    """Render Figure 05: Quarter-wave monopole vs half-wave dipole impedance, S11, and +3 dB directivity."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=10.4, height_in=4.5, ncols=3, nrows=1)
    ax_z, ax_s11, ax_pat = axes

    f_ghz = np.asarray(d2["freqs_ghz"], dtype=float)
    zd_r = np.asarray(d2["z_dip_real"], dtype=float)
    zd_i = np.asarray(d2["z_dip_imag"], dtype=float)
    zm_r = np.asarray(d2["z_mono_real"], dtype=float)
    zm_i = np.asarray(d2["z_mono_imag"], dtype=float)
    zmt_r = np.asarray(d2["z_mono_thin_real"], dtype=float)
    zmt_i = np.asarray(d2["z_mono_thin_imag"], dtype=float)

    # Panel (a): Z_mono(f) vs 0.5 * Z_dip(f) and 36.5 + j21.25 Ohm EMF benchmark
    ax_z.plot(f_ghz, 0.5 * zd_r, color=COLORS["blue"], lw=1.8, label=r"$\frac{1}{2} R_{\mathrm{in,dip}}(f)$ ($a=1$ mm)")
    ax_z.plot(f_ghz[::3], zm_r[::3], "o", color=COLORS["red"], ms=3.8, label=r"$R_{\mathrm{in,mono}}(f)$ ($a=1$ mm)")
    ax_z.plot(f_ghz, 0.5 * zd_i, "--", color=COLORS["teal"], lw=1.7, label=r"$\frac{1}{2} X_{\mathrm{in,dip}}(f)$ ($a=1$ mm)")
    ax_z.plot(f_ghz[::3], zm_i[::3], "s", color=COLORS["orange"], ms=3.6, label=r"$X_{\mathrm{in,mono}}(f)$ ($a=1$ mm)")
    ax_z.plot(f_ghz, zmt_r, ":", color=COLORS["purple"], lw=1.5, label=r"$R_{\mathrm{in,mono}}$ ($a=0.2$ mm)")
    ax_z.plot([1.0], [36.54], "*", color=COLORS["red"], ms=8.5, label=r"EMF $h=\lambda_0/4$: $36.5\,\Omega$")
    ax_z.axhline(0.0, color=COLORS["grey"], ls=":", lw=0.9)
    ax_z.set_xlabel("Frequency $f$ [GHz]")
    ax_z.set_ylabel(r"Monopole impedance $R_{\mathrm{in}}, X_{\mathrm{in}}$ [$\Omega$]")
    ax_z.set_title(r"(a) Image Theory $Z_{\mathrm{mono}} = \frac{1}{2}Z_{\mathrm{dip}}$", pad=10)
    figstyle.place_legend_below(ax_z, ncol=2, y_offset=-0.28)

    # Panel (b): S11(f) for monopole (Z0 = 50 Ohm and 36.5 Ohm) vs dipole (Z0 = 50 Ohm)
    s11_d50 = np.asarray(d2["s11_dip_50_db"], dtype=float)
    s11_m50 = np.asarray(d2["s11_mono_50_db"], dtype=float)
    s11_m36 = np.asarray(d2["s11_mono_36_db"], dtype=float)

    ax_s11.plot(f_ghz, s11_d50, "--", color=COLORS["blue"], lw=1.6, label=r"$\lambda_0/2$ dipole ($Z_0=50\,\Omega$)")
    ax_s11.plot(f_ghz, s11_m50, color=COLORS["teal"], lw=1.8, label=r"$\lambda_0/4$ monopole ($Z_0=50\,\Omega$)")
    ax_s11.plot(f_ghz, s11_m36, "-.", color=COLORS["red"], lw=1.7, label=r"$\lambda_0/4$ monopole ($Z_0=36.5\,\Omega$)")
    ax_s11.axhline(-10.0, color=COLORS["grey"], ls=":", lw=1.0, label="$-10$ dB threshold")
    ax_s11.set_ylim(-25.0, 0.0)
    ax_s11.set_xlabel("Frequency $f$ [GHz]")
    ax_s11.set_ylabel(r"$20\log_{10}|S_{11}(f)|$ [dB]")
    ax_s11.set_title(r"(b) Monopole vs. Dipole $|S_{11}|$", pad=10)
    figstyle.place_legend_below(ax_s11, ncol=1, y_offset=-0.28)

    # Panel (c): Directivity in dBi showing +3.01 dB upper-hemisphere doubling (5.17 dBi vs 2.15 dBi)
    th_deg = np.asarray(d2["theta_deg"], dtype=float)
    dd_fit = 10.0 * np.log10(np.maximum(np.asarray(d2["d_dip_theta"], dtype=float), 1e-3))
    dm_fit = 10.0 * np.log10(np.maximum(np.asarray(d2["d_mono_theta"], dtype=float), 1e-3))
    dd_ana = 10.0 * np.log10(np.maximum(np.asarray(d2["d_dip_ana"], dtype=float), 1e-3))
    dm_ana = 10.0 * np.log10(np.maximum(np.asarray(d2["d_mono_ana"], dtype=float), 1e-3))
    mask_up = th_deg <= 90.0

    ax_pat.plot(th_deg, dd_ana, "--", color=COLORS["blue"], lw=1.6, label=r"Dipole analytic ($2.15$ dBi)")
    ax_pat.plot(th_deg[::4], dd_fit[::4], "o", color=COLORS["blue"], ms=3.5, label=r"Dipole Near2Far ($2.17$ dBi)")
    ax_pat.plot(th_deg[mask_up], dm_ana[mask_up], color=COLORS["red"], lw=1.8, label=r"Monopole analytic ($5.16$ dBi)")
    ax_pat.plot(
        th_deg[mask_up][::3],
        dm_fit[mask_up][::3],
        "s",
        color=COLORS["red"],
        ms=3.6,
        label=f"Monopole Near2Far ({10*np.log10(float(d2['d0_mono'])):.2f} dBi)",
    )
    ax_pat.axvspan(90.0, 180.0, color=COLORS["grey"], alpha=0.12, label=r"PEC ground shadow ($\theta > 90^\circ$)")
    ax_pat.set_xlim(0.0, 180.0)
    ax_pat.set_ylim(-15.0, 6.8)
    ax_pat.set_xlabel(r"Polar angle $\theta$ [deg]")
    ax_pat.set_ylabel(r"Directivity $10\log_{10} D(\theta)$ [dBi]")
    ax_pat.set_title(r"(c) Upper-Half-Space $+3$ dB Directivity", pad=10)
    figstyle.place_legend_below(ax_pat, ncol=1, y_offset=-0.28)

    out = fig_dir / "05_monopole_vs_dipole_image_theory.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_06_near_field_maps_dipole_monopole(d0: Any, d2: Any, d3: Any, fig_dir: Path) -> Path:
    """Render Figure 06: 2D xz near-field maps Re(Ez) and |E| with overlay_geometry for dipole & monopole."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=7.6, height_in=3.6, ncols=2, nrows=1)
    ax_dip, ax_mono = axes

    # Panel (a): Center-fed half-wave dipole in free space
    s_dip, p_dip, n2f_dip, st_dip, names_dip = _build_dipole_schematic_objects(d0)
    x_d_mm = np.asarray(d0["x_cell_m"], dtype=float) * 1e3
    z_d_mm = np.asarray(d0["z_cell_m"], dtype=float) * 1e3
    emag_d = np.asarray(d0["emag_xz"], dtype=float)
    emag_d_db = 20.0 * np.log10(np.maximum(emag_d, 1e-30) / float(np.percentile(emag_d, 99.2)))

    im0 = ax_dip.pcolormesh(
        x_d_mm,
        z_d_mm,
        emag_d_db.T,
        cmap=figstyle.CMAP_MAGNITUDE,
        vmin=-40.0,
        vmax=0.0,
        shading="auto",
    )
    fig.colorbar(im0, ax=ax_dip, fraction=0.046, pad=0.03, label=r"$20\log_{10}(|\mathbf{E}| / E_{99\%})$ [dB]")
    schematic.overlay_geometry(
        ax_dip,
        s_dip,
        plane="xz",
        unit="mm",
        structures=st_dip,
        material_names=names_dip,
        sources=[p_dip],
        monitors=[n2f_dip],
        show_sources=True,
        show_monitors=True,
    )
    ax_dip.set_aspect("equal")
    ax_dip.set_xlabel("$x$ [mm]")
    ax_dip.set_ylabel("$z$ [mm]")
    ax_dip.set_title(r"(a) Free-Space $\lambda_0/2$ Dipole $|\mathbf{E}(x,z)|$")

    # Panel (b): Quarter-wave monopole on finite PEC ground plane (W_gnd = 0.80 * lam0)
    s_mf, p_mf, n2f_mf, st_mf, names_mf = _build_monopole_schematic_objects(d2, d3, finite_gnd=True)
    x_m_mm = np.asarray(d3["x_cell_m"], dtype=float) * 1e3
    z_m_mm = np.asarray(d3["z_cell_m"], dtype=float) * 1e3
    emag_m = np.asarray(d3["emag_xz"], dtype=float)
    emag_m_db = 20.0 * np.log10(np.maximum(emag_m, 1e-30) / float(np.percentile(emag_m, 99.2)))

    im1 = ax_mono.pcolormesh(
        x_m_mm,
        z_m_mm,
        emag_m_db.T,
        cmap=figstyle.CMAP_MAGNITUDE,
        vmin=-40.0,
        vmax=0.0,
        shading="auto",
    )
    fig.colorbar(im1, ax=ax_mono, fraction=0.046, pad=0.03, label=r"$20\log_{10}(|\mathbf{E}| / E_{99\%})$ [dB]")
    schematic.overlay_geometry(
        ax_mono,
        s_mf,
        plane="xz",
        unit="mm",
        structures=st_mf,
        material_names=names_mf,
        sources=[p_mf],
        monitors=[n2f_mf],
        show_sources=True,
        show_monitors=True,
    )
    ax_mono.set_aspect("equal")
    ax_mono.set_xlabel("$x$ [mm]")
    ax_mono.set_ylabel("$z$ [mm]")
    ax_mono.set_title(r"(b) $\lambda_0/4$ Monopole on Finite PEC ($0.8\lambda_0$)")

    out = fig_dir / "06_near_field_maps_dipole_monopole.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_07_far_field_3d_and_finite_ground(d0: Any, d2: Any, d3: Any, fig_dir: Path) -> Path:
    """Render Figure 07: 3D far-field directivity torus and finite PEC ground plane edge diffraction."""
    figstyle.apply_style()
    fig = plt.figure(figsize=(10.6, 4.5))
    ax_3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_pat = fig.add_subplot(1, 3, 2)
    ax_tilt = fig.add_subplot(1, 3, 3)

    # Panel (a): 3D far-field directivity pattern D(theta, phi) of the half-wave dipole
    th_3d = np.asarray(d0["th_3d"], dtype=float)
    ph_3d = np.asarray(d0["ph_3d"], dtype=float)
    d_3d = np.asarray(d0["d_3d"], dtype=float)
    TH, PH = np.meshgrid(th_3d, ph_3d, indexing="ij")
    plot_pattern_3d(TH, PH, d_3d, ax=ax_3d, cmap=figstyle.CMAP_MAGNITUDE)
    ax_3d.set_title(r"(a) 3D $\lambda_0/2$ Directivity Torus", pad=8)

    # Panel (b): Elevation directivity D(theta) [linear] for infinite PEC ground vs finite W_gnd / lam0
    th_deg = np.asarray(d3["theta_deg"], dtype=float)
    d_cuts = np.asarray(d3["d_cuts"], dtype=float)
    w_list = np.asarray(d3["w_over_lam_list"], dtype=float)
    d_inf = np.asarray(d2["d_mono_theta"], dtype=float)
    th_max_arr = np.asarray(d3["theta_max_deg"], dtype=float)

    ax_pat.plot(th_deg, d_inf, color=COLORS["blue"], lw=1.9, label=r"Infinite PEC ground ($\theta_{\max}=90^\circ$)")
    palette_fg = [COLORS["orange"], COLORS["teal"], COLORS["purple"]]
    styles_fg = [":", "--", "-."]
    for idx_w, w_lam in enumerate(w_list):
        ax_pat.plot(
            th_deg,
            d_cuts[idx_w],
            styles_fg[idx_w],
            color=palette_fg[idx_w],
            lw=1.7,
            label=f"Finite $W={w_lam:.1f}\\lambda_0$ ($\\theta_{{\\max}}={th_max_arr[idx_w]:.0f}^\\circ$)",
        )
    ax_pat.axvline(90.0, color=COLORS["grey"], ls=":", lw=1.0, label=r"Ground horizon ($\theta=90^\circ$)")
    ax_pat.set_xlim(0.0, 180.0)
    ax_pat.set_ylim(0.0, 3.55)
    ax_pat.set_xlabel(r"Polar angle $\theta$ [deg]")
    ax_pat.set_ylabel(r"Directivity $D(\theta)$ [linear]")
    ax_pat.set_title("(b) Finite Ground Edge Diffraction", pad=10)
    figstyle.place_legend_below(ax_pat, ncol=1, y_offset=-0.28)

    # Panel (c): Main-beam elevation tilt (90 - theta_max) [deg] and lower-hemisphere shadow power [%]
    tilt_deg = 90.0 - th_max_arr
    back_pct = np.asarray(d3["back_power_frac"], dtype=float) * 100.0
    d_max_dbi = np.asarray(d3["d_max_dbi"], dtype=float)

    ax_tilt.plot(w_list, tilt_deg, "o-", color=COLORS["blue"], ms=5.0, lw=1.8, label=r"Beam elevation tilt $90^\circ - \theta_{\max}$ [deg]")
    ax_tilt.plot(w_list, back_pct, "s--", color=COLORS["red"], ms=4.8, lw=1.7, label=r"Lower-hemisphere power ($\theta>90^\circ$) [%]")
    ax_tilt.plot(w_list, d_max_dbi * 10.0, "^:", color=COLORS["teal"], ms=4.8, lw=1.5, label=r"$10 \times D_{\max}$ [dBi $\times 10$]")
    ax_tilt.axhline(0.0, color=COLORS["grey"], ls=":", lw=0.9)
    ax_tilt.set_xlabel(r"Finite ground width $W_{\mathrm{gnd}} / \lambda_0$")
    ax_tilt.set_ylabel("Angle [deg] / Power fraction [%]")
    ax_tilt.set_title("(c) Lobe Tilt & Back-Lobe Leakage", pad=10)
    figstyle.place_legend_below(ax_tilt, ncol=1, y_offset=-0.28)

    out = fig_dir / "07_far_field_3d_and_finite_ground.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_08_wire_antenna_radiation_gif(d0: Any, fig_dir: Path) -> Path:
    """Render Figure 08: Animated time-domain GIF of the center-fed half-wave dipole with overlay_geometry."""
    figstyle.apply_style()
    s_dip, p_dip, n2f_dip, st_dip, names_dip = _build_dipole_schematic_objects(d0)
    x_mm = np.asarray(d0["x_cell_m"], dtype=float) * 1e3
    z_mm = np.asarray(d0["z_cell_m"], dtype=float) * 1e3
    frames = np.asarray(d0["td_frames"], dtype=float)
    times_ns = np.asarray(d0["td_times_ns"], dtype=float)

    vmax = float(np.percentile(np.abs(frames), 98.5))
    if vmax <= 0.0:
        vmax = 1.0

    pil_images: list[Image.Image] = []
    for idx in range(frames.shape[0]):
        fig, ax = figstyle.new_figure(width_in=4.8, height_in=4.3, ncols=1, nrows=1)
        im = ax.pcolormesh(
            x_mm,
            z_mm,
            frames[idx, : len(x_mm), : len(z_mm)].T,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$E_z(x, 0, z, t)$ [V/m]")
        schematic.overlay_geometry(
            ax,
            s_dip,
            plane="xz",
            unit="mm",
            structures=st_dip,
            material_names=names_dip,
            sources=[p_dip],
            monitors=[n2f_dip],
            show_sources=True,
            show_monitors=True,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$z$ [mm]")
        ax.set_title(f"Center-Fed $\\lambda_0/2$ Dipole ($t = {times_ns[idx]:.2f}$ ns)")
        figstyle.finish(fig, max_panels=1, auto_legend_below=False)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_images.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        plt.close(fig)

    out = fig_dir / "08_wire_antenna_radiation.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_images[0].save(
        out,
        save_all=True,
        append_images=pil_images[1:],
        duration=110,
        loop=0,
        optimize=True,
    )
    return out


def generate_all_figures(data_dir: Path, fig_dir: Path) -> list[Path]:
    """Load ``data_dir/task_*.npz`` and render all 9 Tutorial 4.2 figures into ``fig_dir``."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    d0 = np.load(data_dir / "task_half_wave_thin_wire_radii.npz")
    d1 = np.load(data_dir / "task_resolved_wire_vs_subcell_mesh.npz")
    d2 = np.load(data_dir / "task_quarter_wave_monopole_infinite_gnd.npz")
    d3 = np.load(data_dir / "task_finite_ground_plane_sweep.npz")

    return [
        plot_00_setup_schematic(d0, d2, d3, fig_dir),
        plot_01_geometry_and_graded_mesh(d0, d1, fig_dir),
        plot_02_current_distribution_wire(d0, d2, fig_dir),
        plot_03_impedance_s11_and_shortening(d0, fig_dir),
        plot_04_subcell_vs_resolved_convergence(d1, fig_dir),
        plot_05_monopole_vs_dipole_image_theory(d2, fig_dir),
        plot_06_near_field_maps_dipole_monopole(d0, d2, d3, fig_dir),
        plot_07_far_field_3d_and_finite_ground(d0, d2, d3, fig_dir),
        plot_08_wire_antenna_radiation_gif(d0, fig_dir),
    ]
