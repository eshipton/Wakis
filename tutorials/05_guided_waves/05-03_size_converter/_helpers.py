# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Geometry, calibration, and multi-mode port analysis helpers for Tutorial 5.3
(Waveguide Mode Size Converters: Linear/Non-linear Tapers, Graded-Thickness
Luneburg Lens, and 6-Segment Semi-Lens Beam Expander).
"""

from pathlib import Path

from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light, epsilon_0

from wakis import figstyle, schematic
from wakis.field_monitors import DFTMonitor, FluxMonitor
from wakis.gridFIT3D import GridFIT3D
from wakis.mode_solver import ModeMonitor, ModeSource, solve_modes
from wakis.solverFIT3D import SolverFIT3D
from wakis.sources import GaussianSource
from wakis.templates import Box, Simulation, meshing
from wakis.templates.photonics import (
    PICSimulation,
    PolySlab,
    Port,
    _SnapshotRecorder,
    n_to_eps,
    slab_neff,
    straight,
)
from wakis.unstructured import subtri_circles_mesh

# Common optical constants for Tutorial 5.3 (non-dispersive at 1500-1600 nm)
N_SI = 3.48
N_SIO2 = 1.44
N_AIR = 1.0

WL_MIN = 1500e-9
WL_MAX = 1600e-9
WL_REF = 1550e-9
N_FREQS = 51

# Design 1 & 2 common parameters
T_WG_TAPER = 0.11e-6
W_IN_TAPER = 10.0e-6
W_OUT_TAPER = 0.50e-6
R_LENS = 5.50e-6
D_LENS = 11.0e-6

# Design 3 (Abbaslou et al., Opt. Lett. 42, 4383, 2017) 6-segment semi-lens parameters
T_WG_SEMILENS = 0.26e-6
SEMILENS_L_UM = np.array([1.00, 3.61, 0.05, 0.70, 3.11, 1.03], dtype=float)
SEMILENS_W_UM = np.array([0.45, 1.70, 3.30, 9.06, 10.00, 3.19, 9.00], dtype=float)
SEMILENS_M = np.array([3.00, 1.10, 0.01, 3.00, 0.32, 2.55], dtype=float)
L_TOT_SEMILENS = float(np.sum(SEMILENS_L_UM)) * 1e-6  # 9.50 um


class CircularPolySlab(PolySlab):
    """
    Circular disk `PolySlab` with analytical radial point-containment and
    outward normal evaluation for Kottke subpixel smoothing.

    Parameters
    ----------
    center_xy : tuple of float
        Disk center `(cx, cy)` in the x-y plane [m].
    radius : float
        Disk radius `R` [m].
    z_min, z_max : float
        Bottom and top vertical coordinates [m].
    material : float or str
        Relative permittivity `eps_r` or material name.
    n_verts : int, optional
        Number of polygon vertices stored on `.vertices` for plotting and
        3D PyVista extrusion. Default 96.
    """

    def __init__(self, center_xy, radius, z_min, z_max, material, n_verts=96):
        self.cx = float(center_xy[0])
        self.cy = float(center_xy[1])
        self.radius = float(radius)
        theta = np.linspace(0.0, 2.0 * np.pi, int(n_verts), endpoint=False)
        verts = np.column_stack(
            [
                self.cx + self.radius * np.cos(theta),
                self.cy + self.radius * np.sin(theta),
            ]
        )
        super().__init__(verts, z_min=z_min, z_max=z_max, material=material)
        self.n_sub_assign = 6

    def _contains_xy(self, pts, z=None):
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        return (dx * dx + dy * dy) <= self.radius * self.radius

    def _dist_and_normal_xy(self, x_flat, y_flat, chunk_size=8192):
        dx = x_flat - self.cx
        dy = y_flat - self.cy
        r = np.hypot(dx, dy)
        safe_r = np.where(r > 1e-15, r, 1.0)
        nx = np.where(r > 1e-15, dx / safe_r, 1.0)
        ny = np.where(r > 1e-15, dy / safe_r, 0.0)
        return np.abs(r - self.radius), nx, ny


class SymmetricProfileSlab(PolySlab):
    """
    Mirror-symmetric (`y -> -y`) extruded waveguide slab with axial width
    profile `w(x)` along `x_prof` [m].

    Subclasses `PolySlab` so plotting and solver routines work unchanged while
    accelerating 2D subpixel containment via 1D linear interpolation along `x`
    and exploiting mirror symmetry for boundary-normal evaluation.

    Parameters
    ----------
    x_prof : array_like, shape (M,)
        Strictly increasing axial coordinates `x` [m].
    w_prof : array_like, shape (M,)
        Full transverse waveguide width `w(x) > 0` [m] at `x_prof`.
    z_min, z_max : float
        Bottom and top vertical coordinates [m].
    material : float or str
        Relative permittivity `eps_r` or material key.
    y0 : float, optional
        Symmetry axis `y = y0` [m]. Default 0.0.
    """

    def __init__(self, x_prof, w_prof, z_min, z_max, material, y0=0.0):
        self.x_prof = np.asarray(x_prof, dtype=float)
        self.w_prof = np.asarray(w_prof, dtype=float)
        self.half_w = 0.5 * self.w_prof
        self.y0 = float(y0)
        verts = np.vstack(
            [
                np.column_stack([self.x_prof, self.y0 - self.half_w]),
                np.column_stack([self.x_prof[::-1], self.y0 + self.half_w[::-1]]),
            ]
        )
        super().__init__(verts, z_min=z_min, z_max=z_max, material=material)
        self.n_sub_assign = 6

        up_x, up_y = self.x_prof, self.half_w
        p0 = np.vstack(
            [
                [[up_x[0], 0.0]],
                np.column_stack([up_x[:-1], up_y[:-1]]),
                [[up_x[-1], up_y[-1]]],
            ]
        )
        p1 = np.vstack(
            [
                [[up_x[0], up_y[0]]],
                np.column_stack([up_x[1:], up_y[1:]]),
                [[up_x[-1], 0.0]],
            ]
        )
        seg = p1 - p0
        len2 = np.sum(seg * seg, axis=1)
        length = np.sqrt(np.maximum(len2, 1e-30))
        self._up_p0 = p0
        self._up_seg = seg
        self._up_len2 = len2
        self._up_nx = -seg[:, 1] / length
        self._up_ny = seg[:, 0] / length

    def _contains_xy(self, pts, z=None):
        x = pts[:, 0]
        y = np.abs(pts[:, 1] - self.y0)
        in_x = (x >= self.x_prof[0]) & (x <= self.x_prof[-1])
        hw = np.interp(x, self.x_prof, self.half_w, left=0.0, right=0.0)
        return in_x & (y <= hw)

    def _dist_and_normal_xy(self, x_flat, y_flat, chunk_size=16384):
        dy = y_flat - self.y0
        ay = np.abs(dy)
        sgn_y = np.where(dy >= 0.0, 1.0, -1.0)
        m = x_flat.size
        min_d2 = np.empty(m, dtype=float)
        nx_out = np.empty(m, dtype=float)
        ny_out = np.empty(m, dtype=float)
        p0x, p0y = self._up_p0[:, 0][None, :], self._up_p0[:, 1][None, :]
        sx, sy = self._up_seg[:, 0][None, :], self._up_seg[:, 1][None, :]
        l2 = self._up_len2[None, :]
        enx, eny = self._up_nx, self._up_ny
        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            dx = x_flat[start:end, None] - p0x
            ddy = ay[start:end, None] - p0y
            t = np.clip((dx * sx + ddy * sy) / np.maximum(l2, 1e-30), 0.0, 1.0)
            rx, ry = dx - t * sx, ddy - t * sy
            d2 = rx * rx + ry * ry
            d2_min = np.min(d2, axis=1)
            tie = d2 <= d2_min[:, None] * (1.0 + 1e-9) + 1e-28
            nx_sum, ny_sum = tie @ enx, tie @ eny
            norm = np.hypot(nx_sum, ny_sum)
            safe_norm = np.where(norm > 1e-12, norm, 1.0)
            idx = np.argmin(d2, axis=1)
            min_d2[start:end] = d2_min
            nx_out[start:end] = np.where(norm > 1e-12, nx_sum / safe_norm, enx[idx])
            ny_out[start:end] = sgn_y[start:end] * np.where(
                norm > 1e-12, ny_sum / safe_norm, eny[idx]
            )
        return np.sqrt(min_d2), nx_out, ny_out


def calibrate_luneburg_profile(
    n_si=N_SI,
    n_sio2=N_SIO2,
    n_air=N_AIR,
    wavelength=WL_REF,
    t_min_um=0.11,
    t_max_um=0.40,
    n_thick=30,
    n_radii=100,
    r_lens_um=5.5,
    solve_1d_fit=False,
    cpw_1d=50,
):
    """
    Calibrate the graded-thickness silicon Luneburg lens (Badri & Gilarlue 2019).

    Sweeps `n_thick` slab thicknesses in `[t_min_um, t_max_um]` [um], computes
    the fundamental TE0 effective index `n_eff(t)` analytically (and optionally
    via 1D FIT `solve_modes`), fits a cubic polynomial `t_fit(n_eff)` [um], and
    evaluates the radial Luneburg profiles `n_eff(r)` and `t_lens(r)` across
    `r in [0, r_lens_um]` [um].
    """
    t_sweep_um = np.linspace(float(t_min_um), float(t_max_um), int(n_thick))
    neff_ana = np.array(
        [
            slab_neff(n_si, n_air, t * 1e-6, wavelength, n_sub=n_sio2, polarization="TE")
            for t in t_sweep_um
        ],
        dtype=float,
    )

    if solve_1d_fit:
        f0 = c_light / float(wavelength)
        dz = float(wavelength) / (n_si * float(cpw_1d))
        neff_fit = []
        for t in t_sweep_um:
            th = t * 1e-6
            sub = Box(
                (0.0, -0.75e-6, 0.0),
                (10e-6, 1.5e-6, 10e-6),
                material=n_to_eps(n_sio2),
            )
            air = Box(
                (0.0, 0.75e-6, 0.0),
                (10e-6, 1.5e-6, 10e-6),
                material=n_to_eps(n_air),
            )
            core = PolySlab(
                straight((-5e-6, 0.5 * th), (5e-6, 0.5 * th), width=th),
                -5e-6,
                5e-6,
                material=n_to_eps(n_si),
            )
            sim = Simulation(
                f_min=0.95 * f0,
                f_max=f0,
                structures=[sub, air, core],
                boundaries=(
                    ["periodic", "pec", "periodic"],
                    ["periodic", "pec", "periodic"],
                ),
                domain=(-2 * dz, 2 * dz, -1.2e-6, 1.2e-6, -0.5 * dz, 0.5 * dz),
                cells_per_wavelength=cpw_1d,
                averaging="kottke",
            )
            solver = sim.build()
            m = solve_modes(
                solver,
                f0,
                center=(0.0, 0.0, 0.0),
                size=(0.0, np.inf, np.inf),
                n_modes=1,
            )
            neff_fit.append(float(np.real(m[0].neff)))
        neff_fit = np.asarray(neff_fit, dtype=float)
    else:
        neff_fit = neff_ana.copy()

    poly_coeffs = np.polyfit(neff_ana, t_sweep_um, 3)
    n_edge = float(neff_ana[0])
    r_um = np.linspace(0.0, float(r_lens_um), int(n_radii))
    neff_luneburg = n_edge * np.sqrt(np.maximum(2.0 - (r_um / r_lens_um) ** 2, 1.0))
    t_luneburg_um = np.polyval(poly_coeffs, neff_luneburg)
    t_luneburg_um = np.clip(t_luneburg_um, float(t_min_um), float(t_max_um))
    t_luneburg_um[-1] = float(t_min_um)
    neff_realized = np.array(
        [
            slab_neff(n_si, n_air, t * 1e-6, wavelength, n_sub=n_sio2, polarization="TE")
            for t in t_luneburg_um
        ],
        dtype=float,
    )
    return {
        "t_sweep_um": t_sweep_um,
        "neff_sweep_ana": neff_ana,
        "neff_sweep_fit": neff_fit,
        "poly_coeffs": poly_coeffs,
        "n_edge": n_edge,
        "r_um": r_um,
        "neff_luneburg": neff_luneburg,
        "t_luneburg_um": t_luneburg_um,
        "neff_realized": neff_realized,
    }


def taper_width_profile(
    x_taper, length_taper, w_in=W_IN_TAPER, w_out=W_OUT_TAPER, profile="linear"
):
    """
    Evaluate full taper width `w(x)` [m] over `x_taper in [0, length_taper]`.

    Supported `profile` values:
    - `'linear'`: `w_in + (w_out - w_in) * s`
    - `'parabolic'`: `w_out + (w_in - w_out) * (1 - s)**2`
    - `'exponential'`: `w_in * (w_out / w_in)**s`
    - `'cosine'`: `w_in + (w_out - w_in) * 0.5 * (1 - cos(pi * s))`
    """
    s = np.clip(
        np.asarray(x_taper, dtype=float) / max(float(length_taper), 1e-12), 0.0, 1.0
    )
    if profile == "linear":
        return w_in + (w_out - w_in) * s
    if profile == "parabolic":
        return w_out + (w_in - w_out) * ((1.0 - s) ** 2)
    if profile == "exponential":
        return w_in * ((w_out / w_in) ** s)
    if profile == "cosine":
        return w_in + (w_out - w_in) * 0.5 * (1.0 - np.cos(np.pi * s))
    raise ValueError(f"Unknown taper profile {profile!r}.")


# Minimum geometric features [m]
MIN_FEATURE_SEMILENS = 0.05e-6  # Segment 3 length L_3 = 50 nm (x in [4.61, 4.66] um)
W_IN_SEMILENS = 0.45e-6  # Input single-mode strip width w_0 = 450 nm
SEMILENS_SEG3_X_LO = 4.61e-6
SEMILENS_SEG3_X_HI = 4.66e-6
MIN_FEATURE_TAPER = W_OUT_TAPER  # Output single-mode waveguide width w_out = 500 nm
MIN_FEATURE_LUNEBURG = W_OUT_TAPER  # Output single-mode waveguide width w_out = 500 nm


def semilens_mesh_overrides(n_feature):
    """
    Build `mesh_overrides` boxes for Design 3 (6-segment semi-lens expander).

    Refines the 50 nm axial flare of Segment 3 (`x in [4.61, 4.66]` um) with
    `hx <= 50 nm / n_feature` and the 450 nm input waveguide (`y in [-0.225, 0.225]` um)
    with `hy <= 450 nm / n_feature`. When `n_feature <= 1` (under-resolved demo),
    returns an empty list so the mesh stays at the unrefined background cell size.
    """
    if n_feature is None or int(n_feature) <= 1:
        return []
    n_f = float(n_feature)
    hx_cap = MIN_FEATURE_SEMILENS / n_f
    hy_cap = W_IN_SEMILENS / n_f
    return [
        (
            (SEMILENS_SEG3_X_LO, -np.inf, -np.inf),
            (SEMILENS_SEG3_X_HI, np.inf, np.inf),
            (hx_cap, 1.0, 1.0),
        ),
        (
            (-np.inf, -0.5 * W_IN_SEMILENS, -np.inf),
             (np.inf, 0.5 * W_IN_SEMILENS, np.inf),
             (1.0, hy_cap, 1.0),
        ),
    ]


def waveguide_width_mesh_overrides(n_feature, width=W_OUT_TAPER):
    """
    Build a transverse `mesh_overrides` box across `y in [-width/2, +width/2]`
    with `hy <= width / n_feature` [m].
    """
    if n_feature is None or int(n_feature) <= 0:
        return []
    w = float(width)
    hy_cap = w / float(n_feature)
    return [
        (
            (-np.inf, -0.5 * w, -np.inf),
            (np.inf, 0.5 * w, np.inf),
            (1.0, hy_cap, 1.0),
        ),
    ]


def measure_grid_resolution(
    grid,
    min_feature_m,
    feature_bounds,
    feature_axis=0,
    core_x_bounds=None,
):
    """
    Measure resolution metrics directly from an actual `GridFIT3D` mesh.

    Parameters
    ----------
    grid : GridFIT3D
        Constructed 2D/3D tensor-product FIT mesh.
    min_feature_m : float
        Smallest geometric feature size [m].
    feature_bounds : tuple of (float, float)
        Coordinate interval `(lo, hi)` [m] spanned by the smallest feature
        along `feature_axis`.
    feature_axis : {0, 1}, optional
        Axis along which the smallest feature is measured (`0` for x, `1` for y).
    core_x_bounds : tuple of (float, float), optional
        Axial interval `(x_lo, x_hi)` [m] inside the highest-index core region
        used to measure `dx_background_nm`.

    Returns
    -------
    dict
        Dictionary containing `min_feature_nm`, `dx_background_nm`, `dx_min_nm`,
        `cells_across_feature`, `n_cells`, and `grid_shape`.
    """
    dx = np.diff(grid.x)
    dy = np.diff(grid.y)
    xm = 0.5 * (grid.x[:-1] + grid.x[1:])
    if core_x_bounds is not None:
        in_core = (xm >= core_x_bounds[0]) & (xm <= core_x_bounds[1])
        dx_bg_m = float(np.max(dx[in_core])) if np.any(in_core) else float(np.max(dx))
    else:
        dx_bg_m = float(np.max(dx))

    dx_min_m = float(min(np.min(dx), np.min(dy)))

    nodes = grid.x if int(feature_axis) == 0 else grid.y
    d_axis = np.diff(nodes)
    mids = 0.5 * (nodes[:-1] + nodes[1:])
    lo_f, hi_f = float(feature_bounds[0]), float(feature_bounds[1])
    in_feat = (mids > lo_f) & (mids < hi_f)
    if not np.any(in_feat):
        in_feat = (nodes[1:] > lo_f) & (nodes[:-1] < hi_f)
    max_cell_feat_m = float(np.max(d_axis[in_feat]))
    cells_across = float(min_feature_m) / max_cell_feat_m
    grid_shape = [int(grid.Nx), int(grid.Ny), int(grid.Nz)]
    return {
        "min_feature_nm": round(float(min_feature_m) * 1e9, 3),
        "dx_background_nm": round(dx_bg_m * 1e9, 3),
        "dx_min_nm": round(dx_min_m * 1e9, 3),
        "cells_across_feature": round(cells_across, 3),
        "n_cells": int(np.prod(grid_shape)),
        "grid_shape": grid_shape,
    }


def build_taper_converter(
    length_taper=20.0e-6,
    profile="linear",
    w_in=W_IN_TAPER,
    w_out=W_OUT_TAPER,
    t_wg=T_WG_TAPER,
    l_lead=1.20e-6,
    n_pts=80,
):
    """
    Build Design 1 (adiabatic or non-adiabatic taper converter) as a single
    `SymmetricProfileSlab` along with its input/output `Port`s and 2D domain.
    """
    L_t = float(length_taper)
    if L_t <= 1e-9:
        xs = np.array([-l_lead, -1e-9, 1e-9, l_lead], dtype=float)
        ws = np.array([w_in, w_in, w_out, w_out], dtype=float)
    else:
        n_s = 2 if profile == "linear" else max(int(n_pts), 16)
        x_mid = np.linspace(0.0, L_t, n_s)
        w_mid = taper_width_profile(
            x_mid, L_t, w_in=w_in, w_out=w_out, profile=profile
        )
        xs = np.concatenate([[-l_lead], x_mid, [L_t + l_lead]])
        ws = np.concatenate([[w_in], w_mid, [w_out]])

    slab = SymmetricProfileSlab(xs, ws, z_min=0.0, z_max=t_wg, material=n_to_eps(N_SI))
    x_in_port = -0.60e-6
    x_out_port = L_t + 0.60e-6
    y_dom = max(5.65e-6, 0.565 * w_in)
    ports = [
        Port(
            "in",
            center=(x_in_port, 0.0, 0.5 * t_wg),
            size=(0.0, min(10.8e-6, 2.0 * y_dom - 0.4e-6), 6.0 * t_wg),
            direction=+1,
        ),
        Port(
            "out",
            center=(x_out_port, 0.0, 0.5 * t_wg),
            size=(0.0, 1.8e-6, 6.0 * t_wg),
            direction=+1,
        ),
    ]
    domain_2d = (-l_lead, L_t + l_lead, -y_dom, y_dom)
    return [slab], ports, domain_2d


def build_luneburg_converter(
    n_shells=40,
    r_lens=R_LENS,
    w_in=W_IN_TAPER,
    w_out=W_OUT_TAPER,
    t_wg=T_WG_TAPER,
    n_verts=96,
):
    """
    Build Design 2 (11 um graded-thickness Luneburg lens waveguide mode-size
    converter of Badri & Gilarlue 2019) as input/output waveguides plus
    concentric `CircularPolySlab` shells.
    """
    cal = calibrate_luneburg_profile(
        n_si=N_SI,
        n_sio2=N_SIO2,
        n_air=N_AIR,
        wavelength=WL_REF,
        t_min_um=t_wg * 1e6,
        t_max_um=0.40,
        n_thick=30,
        n_radii=max(int(n_shells), 8),
        r_lens_um=r_lens * 1e6,
        solve_1d_fit=False,
    )
    n_edge = cal["n_edge"]
    wg_in = SymmetricProfileSlab(
        [-7.00e-6, 0.0],
        [w_in, w_in],
        z_min=0.0,
        z_max=t_wg,
        material=n_to_eps(n_edge),
    )
    wg_out = SymmetricProfileSlab(
        [0.0, 7.00e-6],
        [w_out, w_out],
        z_min=0.0,
        z_max=t_wg,
        material=n_to_eps(n_edge),
    )
    structures = [wg_in, wg_out]

    r_edges_um = np.linspace(r_lens * 1e6, (r_lens * 1e6) / n_shells, int(n_shells))
    coeffs = cal["poly_coeffs"]
    for i, r_um in enumerate(r_edges_um):
        r_next = r_edges_um[i + 1] if i + 1 < len(r_edges_um) else 0.0
        r_mid = 0.5 * (r_um + r_next)
        n_target = n_edge * np.sqrt(
            np.maximum(2.0 - (r_mid / (r_lens * 1e6)) ** 2, 1.0)
        )
        t_local = float(np.clip(np.polyval(coeffs, n_target), t_wg * 1e6, 0.40)) * 1e-6
        neff_local = slab_neff(
            N_SI, N_AIR, t_local, WL_REF, n_sub=N_SIO2, polarization="TE"
        )
        structures.append(
            CircularPolySlab(
                (0.0, 0.0),
                radius=r_um * 1e-6,
                z_min=0.0,
                z_max=t_local,
                material=n_to_eps(neff_local),
                n_verts=n_verts,
            )
        )

    ports = [
        Port(
            "in",
            center=(-6.05e-6, 0.0, 0.5 * t_wg),
            size=(0.0, 11.4e-6, 6.0 * t_wg),
            direction=+1,
        ),
        Port(
            "out",
            center=(+6.05e-6, 0.0, 0.5 * t_wg),
            size=(0.0, 2.0e-6, 6.0 * t_wg),
            direction=+1,
        ),
    ]
    domain_2d = (-6.65e-6, 6.65e-6, -6.10e-6, 6.10e-6)
    return structures, ports, domain_2d, cal


def semilens_width_profile(n_per_seg=40):
    """
    Evaluate the 6-segment piecewise power-law width profile `w(x)` [m] of
    Abbaslou et al. (Opt. Lett. 42, 4383, 2017) over `x in [0, 9.50]` um.
    """
    xs, ws = [], []
    x_cur = 0.0
    for i, (L_i, m_i) in enumerate(zip(SEMILENS_L_UM, SEMILENS_M)):
        w_prev, w_next = SEMILENS_W_UM[i], SEMILENS_W_UM[i + 1]
        is_last = i == len(SEMILENS_L_UM) - 1
        xi = np.linspace(0.0, L_i, int(n_per_seg), endpoint=is_last)
        wi = (w_prev - w_next) * np.abs((xi - L_i) / L_i) ** m_i + w_next
        xs.extend((x_cur + xi).tolist())
        ws.extend(wi.tolist())
        x_cur += L_i
    return np.asarray(xs, dtype=float) * 1e-6, np.asarray(ws, dtype=float) * 1e-6


def build_semilens_expander(n_per_seg=35, l_in=1.40e-6, l_out=1.70e-6):
    """
    Build Design 3 (9.50 um 6-segment semi-lens beam expander of Abbaslou
    et al. 2017) as a single `SymmetricProfileSlab` along with its ports and
    2D simulation domain.
    """
    x_seg, w_seg = semilens_width_profile(n_per_seg=n_per_seg)
    xs = np.concatenate([[-float(l_in)], x_seg, [L_TOT_SEMILENS + float(l_out)]])
    ws = np.concatenate([[w_seg[0]], w_seg, [w_seg[-1]]])
    slab = SymmetricProfileSlab(
        xs, ws, z_min=0.0, z_max=T_WG_SEMILENS, material=n_to_eps(N_SI)
    )
    ports = [
        Port(
            "in",
            center=(-0.95e-6, 0.0, 0.5 * T_WG_SEMILENS),
            size=(0.0, 1.80e-6, 5.0 * T_WG_SEMILENS),
            direction=+1,
        ),
        Port(
            "out",
            center=(L_TOT_SEMILENS + 1.05e-6, 0.0, 0.5 * T_WG_SEMILENS),
            size=(0.0, 10.2e-6, 5.0 * T_WG_SEMILENS),
            direction=+1,
        ),
    ]
    domain_2d = (-1.25e-6, L_TOT_SEMILENS + 1.45e-6, -5.55e-6, 5.55e-6)
    return [slab], ports, domain_2d


def build_semilens_graded_axes(
    domain,
    n_pml: int = 10,
    cpw: float = 20.0,
    n_feature: int = 8,
    dx_flare: float | None = None,
    dx_lens: float | None = None,
    dx_lead: float | None = None,
    dy_in: float | None = None,
    dy_core: float | None = None,
    dy_clad: float | None = None,
    grade_ratio: float = 1.22,
):
    """
    Construct graded non-uniform Cartesian node arrays `(x, y, z)` for Design 3
    (6-segment semi-lens beam expander):
    - Fine axial spacing `dx_flare` across the 50 nm Segment 3 flare (`x in [4.61, 4.66] um`),
    - Fine isotropic spacing `dx_lens` across the curved refractive lens (`x in [4.66, 8.47] um`),
    - Non-square axial cells `dx_lead > dy_core` along the straight leads and slowly varying
      Segments 1-2 (`x in [xmin, 4.45] um`) and Segment 6 (`x in [8.47, xmax] um`),
    - Fine transverse spacing `dy_in` across the 450 nm single-mode input waveguide (`|y| <= 0.30 um`),
      `dy_core` across the expanded beam (`|y| in [0.30, 4.56] um`), and graded coarse cells
      `dy_clad` in the exterior air cladding (`|y| > 4.56 um`).
    """
    xmin, xmax, ymin, ymax, zmin_2d, zmax_2d = domain
    scale = 20.0 / max(float(cpw), 8.0)
    dx_fl = float(
        MIN_FEATURE_SEMILENS / max(int(n_feature), 1)
        if dx_flare is None
        else dx_flare
    )
    dx_ln = float(25.0e-9 * scale if dx_lens is None else dx_lens)
    dx_ld = float(36.0e-9 * scale if dx_lead is None else dx_lead)
    dy_i = float(22.5e-9 * scale if dy_in is None else dy_in)
    dy_c = float(26.2e-9 * scale if dy_core is None else dy_core)
    dy_cl = float(70.0e-9 * scale if dy_clad is None else dy_clad)

    x_nodes = meshing.graded_axis(
        [xmin, 4.45e-6, SEMILENS_SEG3_X_LO, SEMILENS_SEG3_X_HI, 4.82e-6, 8.47e-6, xmax],
        [dx_ld, dx_ln, dx_fl, dx_ln, dx_ln, dx_ld],
        grading=grade_ratio,
    )
    y_pos = meshing.graded_axis(
        [0.0, 0.30e-6, 4.56e-6, ymax],
        [dy_i, dy_c, dy_cl],
        grading=grade_ratio,
    )
    y_nodes = np.concatenate([-y_pos[:0:-1], y_pos])

    def _pad_pml(nodes: np.ndarray) -> np.ndarray:
        lo_pml = nodes[0] - np.arange(n_pml, 0, -1) * (nodes[1] - nodes[0])
        hi_pml = nodes[-1] + np.arange(1, n_pml + 1) * (nodes[-1] - nodes[-2])
        return np.concatenate([lo_pml, nodes, hi_pml])

    return (
        _pad_pml(x_nodes),
        _pad_pml(y_nodes),
        np.array([zmin_2d, zmax_2d], dtype=float),
    )


class NonUniformSemilensPIC(PICSimulation):
    """
    2D EIM `PICSimulation` with a graded non-uniform Cartesian mesh tailored to
    Design 3 (6-segment ultra-short semi-lens beam expander).
    """

    def __init__(
        self,
        *args,
        n_feature: int = 8,
        dx_flare: float | None = None,
        dx_lens: float | None = None,
        dx_lead: float | None = None,
        dy_in: float | None = None,
        dy_core: float | None = None,
        dy_clad: float | None = None,
        grade_ratio: float = 1.22,
        **kwargs,
    ):
        self.n_feature_target = int(n_feature)
        self.dx_flare = dx_flare
        self.dx_lens = dx_lens
        self.dx_lead = dx_lead
        self.dy_in = dy_in
        self.dy_core = dy_core
        self.dy_clad = dy_clad
        self.grade_ratio = float(grade_ratio)
        super().__init__(*args, **kwargs)

    def build(self):
        structs, domain = self._build_2d_structures_and_domain()
        boundaries = (["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"])
        self.sim = Simulation(
            self.f_min,
            self.f_max,
            structures=structs,
            boundaries=boundaries,
            domain=domain,
            cells_per_wavelength=self.cells_per_wavelength,
            padding=self.padding,
            n_pml=self.n_pml,
            n_freqs=self.n_freqs,
            cfln=self.cfln,
            mesh_overrides=[],
            use_gpu=self.use_gpu,
            averaging=self.averaging,
        )
        axes_xyz = build_semilens_graded_axes(
            domain,
            n_pml=self.sim.n_pml,
            cpw=self.cells_per_wavelength,
            n_feature=self.n_feature_target,
            dx_flare=self.dx_flare,
            dx_lens=self.dx_lens,
            dx_lead=self.dx_lead,
            dy_in=self.dy_in,
            dy_core=self.dy_core,
            dy_clad=self.dy_clad,
            grade_ratio=self.grade_ratio,
        )
        self.sim._axis = lambda axis: axes_xyz[int(axis)]
        self.solver = self.sim.build()
        self.solver.ieps[:, :, :, "z"] = 0.0
        self.grid = self.sim.grid
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        return self.solver


def inspect_task_mesh(task_cfg):
    """
    Build the exact 2D `GridFIT3D` mesh for a study task configuration without
    allocating or running the time-domain solver, and return its measured
    resolution metrics (`min_feature_nm`, `dx_background_nm`, `dx_min_nm`,
    `cells_across_feature`, `n_cells`, `grid_shape`, `dt_s`, `nt_max_est`).
    """
    kind = task_cfg["kind"]
    cpw = float(task_cfg["cpw"])
    n_feat = int(task_cfg.get("n_feature", 4))
    n_pml = int(task_cfg.get("n_pml", 8))

    if kind == "semilens":
        structures, ports, domain_2d = build_semilens_expander()
        n_clad, n_sub = N_AIR, N_SIO2
        overrides = semilens_mesh_overrides(n_feat)
        min_feat_m = MIN_FEATURE_SEMILENS
        feat_bounds = (SEMILENS_SEG3_X_LO, SEMILENS_SEG3_X_HI)
        feat_axis = 0
        core_x = (0.0, 4.0e-6)
    elif kind == "luneburg":
        structures, ports, domain_2d, _ = build_luneburg_converter(
            n_shells=task_cfg.get("n_shells", 40)
        )
        n_clad, n_sub = N_SIO2, N_AIR
        overrides = waveguide_width_mesh_overrides(n_feat, width=W_OUT_TAPER)
        min_feat_m = MIN_FEATURE_LUNEBURG
        feat_bounds = (-0.5 * W_OUT_TAPER, 0.5 * W_OUT_TAPER)
        feat_axis = 1
        core_x = (-3.0e-6, 3.0e-6)
    elif kind in ("taper_single", "taper_sweep"):
        length_um = float(task_cfg.get("length_um", 10.0))
        structures, ports, domain_2d = build_taper_converter(
            length_taper=length_um * 1e-6,
            profile=task_cfg.get("profile", "linear"),
        )
        n_clad, n_sub = N_SIO2, N_AIR
        overrides = waveguide_width_mesh_overrides(n_feat, width=W_OUT_TAPER)
        min_feat_m = MIN_FEATURE_TAPER
        feat_bounds = (-0.5 * W_OUT_TAPER, 0.5 * W_OUT_TAPER)
        feat_axis = 1
        core_x = None
    else:
        raise ValueError(f"Unknown task kind {kind!r}")

    pic_stub = object.__new__(PICSimulation)
    pic_stub.wavelength_min = float(WL_MIN)
    pic_stub.wavelength_max = float(WL_MAX)
    pic_stub.wavelength_ref = float(WL_REF)
    pic_stub.f_min = c_light / pic_stub.wavelength_max
    pic_stub.f_max = c_light / pic_stub.wavelength_min
    pic_stub.f_center = 0.5 * (pic_stub.f_min + pic_stub.f_max)
    pic_stub.structures_3d = list(structures)
    pic_stub.ports = list(ports)
    pic_stub.n_core = float(N_SI)
    pic_stub.n_clad = float(n_clad)
    pic_stub.n_sub = float(n_sub)
    pic_stub.cells_per_wavelength = cpw
    pic_stub.padding = 0.8e-6
    pic_stub._domain_input = domain_2d
    z_mins = [s.bounds[0][2] for s in pic_stub.structures_3d]
    z_maxs = [s.bounds[1][2] for s in pic_stub.structures_3d]
    pic_stub.z_min = min(z_mins)
    pic_stub.z_max = max(z_maxs)
    pic_stub.thickness = pic_stub.z_max - pic_stub.z_min
    pic_stub.n_eff_slab = slab_neff(
        pic_stub.n_core,
        pic_stub.n_clad,
        pic_stub.thickness,
        pic_stub.wavelength_ref,
        n_sub=pic_stub.n_sub,
        polarization="TE",
    )
    structs_2d, domain = pic_stub._build_2d_structures_and_domain()
    if bool(task_cfg.get("nonuniform", False)) and kind == "semilens":
        x, y, z = build_semilens_graded_axes(
            domain,
            n_pml=n_pml,
            cpw=cpw,
            n_feature=n_feat,
            dx_flare=task_cfg.get("dx_flare"),
        )
    else:
        sim = Simulation(
            pic_stub.f_min,
            pic_stub.f_max,
            structures=structs_2d,
            boundaries=(["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"]),
            domain=domain,
            cells_per_wavelength=cpw,
            n_pml=n_pml,
            mesh_overrides=overrides,
        )
        x, y, z = [sim._axis(a) for a in range(3)]
    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    grid.remove_edge_ghost(2)
    metrics = measure_grid_resolution(
        grid,
        min_feature_m=min_feat_m,
        feature_bounds=feat_bounds,
        feature_axis=feat_axis,
        core_x_bounds=core_x,
    )
    if bool(task_cfg.get("nonuniform", False)):
        metrics["dx_background_nm"] = round(
            float(WL_MIN / (pic_stub.n_eff_slab * cpw)) * 1e9, 3
        )
    dt_s = 0.5 / (
        c_light
        * np.sqrt(
            1.0 / np.min(grid.dx) ** 2
            + 1.0 / np.min(grid.dy) ** 2
            + 1.0 / np.min(grid.dz) ** 2
        )
    )
    pulse = GaussianSource(
        frequency=pic_stub.f_center,
        fwidth=1.25 * (pic_stub.f_max - pic_stub.f_min),
        cutoff=4.2,
    )
    lx = grid.xmax - grid.xmin
    max_time = pulse.end_time + 2.0 * lx * 2.8 / c_light
    metrics["dt_s"] = float(dt_s)
    metrics["nt_max_est"] = int(np.ceil(max_time / dt_s))
    metrics["grid"] = grid
    return metrics


def compute_abrupt_step_overlap(
    w_in=W_IN_TAPER,
    w_out=W_OUT_TAPER,
    t_wg=T_WG_TAPER,
    wavelength=WL_REF,
    n_pts=4001,
):
    """
    Analytical 2D-EIM mode-overlap estimate `|<E_in, E_out>|^2` and
    electromagnetic Poynting overlap for an abrupt step between waveguides
    of widths `w_in` and `w_out` [m].
    """
    n_slab = slab_neff(N_SI, N_AIR, t_wg, wavelength, n_sub=N_SIO2, polarization="TE")
    n_clad = N_SIO2
    neff_in = slab_neff(n_slab, n_clad, w_in, wavelength, polarization="TM")
    neff_out = slab_neff(n_slab, n_clad, w_out, wavelength, polarization="TM")
    k0 = 2.0 * np.pi / float(wavelength)

    y_span = 1.6 * max(w_in, w_out)
    y = np.linspace(-0.5 * y_span, 0.5 * y_span, int(n_pts))

    def _quasi_te0_profile(w, neff):
        kappa = k0 * np.sqrt(max(n_slab**2 - neff**2, 0.0))
        gamma = k0 * np.sqrt(max(neff**2 - n_clad**2, 0.0))
        hw = 0.5 * w
        hz = np.where(
            np.abs(y) <= hw,
            np.cos(kappa * y),
            np.cos(kappa * hw) * np.exp(-gamma * (np.abs(y) - hw)),
        )
        eps_y = np.where(np.abs(y) <= hw, n_slab**2, n_clad**2)
        ey = hz / eps_y
        ey_norm = ey / max(np.sqrt(np.trapezoid(ey**2, y)), 1e-30)
        p_norm = max(np.sqrt(np.trapezoid(ey * hz, y)), 1e-30)
        return ey_norm, ey / p_norm, hz / p_norm

    e1, ep1, hp1 = _quasi_te0_profile(w_in, neff_in)
    e2, ep2, hp2 = _quasi_te0_profile(w_out, neff_out)
    overlap_sq = float(np.trapezoid(e1 * e2, y) ** 2)
    em_overlap = float((0.5 * np.trapezoid(ep1 * hp2 + ep2 * hp1, y)) ** 2)
    fresnel = float(4.0 * neff_in * neff_out / (neff_in + neff_out) ** 2)
    return {
        "y": y,
        "e_in": e1,
        "e_out": e2,
        "neff_slab": n_slab,
        "neff_in": neff_in,
        "neff_out": neff_out,
        "overlap_sq": overlap_sq,
        "em_overlap": em_overlap,
        "fresnel_transmission": fresnel * overlap_sq,
    }


def run_converter_simulation(
    structures,
    ports,
    domain_2d,
    n_clad=N_SIO2,
    n_sub=N_AIR,
    averaging="kottke",
    cpw=10,
    n_pml=8,
    n_freqs=N_FREQS,
    energy_decay=1e-4,
    max_time=None,
    mesh_overrides=(),
    wavelength_min=WL_MIN,
    wavelength_max=WL_MAX,
    collect_fields=False,
    record_movie=False,
    movie_frames=32,
    multimode_indices=(),
    use_gpu=False,
    nonuniform=False,
    n_feature=8,
):
    """
    Build and run a 2D EIM converter simulation using `PICSimulation` (or
    `NonUniformSemilensPIC` when `nonuniform=True`), returning the simulation
    instance, S-parameters, output aperture flux transmission, and optional
    multi-mode output amplitudes (`TE0, TE2, TE4, TE6, TE8`).
    """
    sim_cls = NonUniformSemilensPIC if nonuniform else PICSimulation
    extra_kw = {"n_feature": int(n_feature)} if nonuniform else {}
    pic = sim_cls(
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        n_freqs=n_freqs,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=n_clad,
        n_sub=n_sub,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging=averaging,
        cells_per_wavelength=cpw,
        domain=domain_2d,
        n_pml=n_pml,
        mesh_overrides=mesh_overrides,
        field_wavelengths=(WL_REF,),
        record_movie=record_movie,
        movie_frames=movie_frames,
        use_gpu=use_gpu,
        **extra_kw,
    )
    pic.sim.pulse = GaussianSource(
        frequency=pic.f_center,
        fwidth=1.25 * (pic.f_max - pic.f_min),
        cutoff=4.2,
    )

    lx = pic.grid.xmax - pic.grid.xmin
    if max_time is None:
        max_time = pic.sim.pulse.end_time + 2.0 * lx * 2.8 / c_light

    in_port = pic.port_map["in"]
    out_port = pic.port_map["out"]
    in_center, in_size, in_inward = pic._port_plane_args(in_port)
    out_center, out_size, out_inward = pic._port_plane_args(out_port)

    nodes_x = pic.grid.x
    k_mon = int(np.argmin(np.abs(nodes_x[:-1] - in_center[0])))
    k_src = int(np.clip(k_mon - 2 * in_inward, 2, len(nodes_x) - 3))
    src_center = in_center.copy()
    src_center[0] = nodes_x[k_src]

    src = ModeSource(
        pic.sim.pulse,
        center=src_center,
        size=in_size,
        direction="+" if in_inward > 0 else "-",
        mode_index=in_port.mode_index,
        mode_frequency=pic.f_center,
    )

    max_out_mode = max([out_port.mode_index, *multimode_indices])
    mon_in = ModeMonitor(
        pic.frequencies,
        center=in_center,
        size=in_size,
        n_modes=1,
        mode_frequency=pic.f_center,
    )
    mon_out = ModeMonitor(
        pic.frequencies,
        center=out_center,
        size=out_size,
        n_modes=max_out_mode + 1,
        mode_frequency=pic.f_center,
    )
    flux_out_port = FluxMonitor(pic.frequencies, center=out_center, size=out_size)
    monitors_list = [mon_in, mon_out, flux_out_port]

    dft_mon = snap_mon = flux_planes = None
    if collect_fields:
        f_dft = [c_light / w for w in pic.field_wavelengths]
        stride = max(1, int(np.ceil(max(pic.grid.Nx, pic.grid.Ny) / 400)))
        z_mid = 0.5 * (pic.grid.zmin + pic.grid.zmax)
        dft_mon = DFTMonitor(
            f_dft,
            components=("Ex", "Ey"),
            center=(
                0.5 * (pic.grid.xmin + pic.grid.xmax),
                0.5 * (pic.grid.ymin + pic.grid.ymax),
                z_mid,
            ),
            size=(np.inf, np.inf, 0.0),
        )
        monitors_list.append(dft_mon)

        Nt_est = int(np.ceil(max_time / pic.solver.dt))
        snap_mon = _SnapshotRecorder(
            component="y",
            max_frames=pic.movie_frames if pic.record_movie else 0,
            stride_xy=stride,
            n_steps_est=Nt_est,
        )
        ix = int(np.argmin(np.abs(pic.grid.x[:-1] - out_center[0])))
        iy = int(np.argmin(np.abs(pic.grid.y[:-1] - out_center[1])))
        snap_mon.probe_idx = (ix, iy, pic.solver.Nz // 2)
        monitors_list.append(snap_mon)

        n_p = pic.sim.n_pml + 1
        xw = float(pic.grid.x[max(1, k_src - 2)])
        xe = float(pic.grid.x[-1 - n_p])
        ys = float(pic.grid.y[n_p])
        yn = float(pic.grid.y[-1 - n_p])
        xc_b, yc_b = 0.5 * (xw + xe), 0.5 * (ys + yn)
        lx_b, ly_b = xe - xw, yn - ys
        flux_planes = {
            "west": FluxMonitor(
                pic.frequencies, (xw, yc_b, z_mid), (0.0, ly_b, np.inf)
            ),
            "east": FluxMonitor(
                pic.frequencies, (xe, yc_b, z_mid), (0.0, ly_b, np.inf)
            ),
            "south": FluxMonitor(
                pic.frequencies, (xc_b, ys, z_mid), (lx_b, 0.0, np.inf)
            ),
            "north": FluxMonitor(
                pic.frequencies, (xc_b, yn, z_mid), (lx_b, 0.0, np.inf)
            ),
        }
        monitors_list.extend(flux_planes.values())

    Nt = int(np.ceil(max_time / pic.solver.dt))
    pic.solver.emsolve(
        Nt,
        source=[src],
        monitors=monitors_list,
        stop_when_energy_decayed=energy_decay,
    )

    a_plus_in, a_minus_in = mon_in.get_amplitudes(0)
    a_inc = a_plus_in if in_inward > 0 else a_minus_in
    b_refl = a_minus_in if in_inward > 0 else a_plus_in

    a_plus_out, a_minus_out = mon_out.get_amplitudes(out_port.mode_index)
    b_out = a_minus_out if out_inward > 0 else a_plus_out

    s_matrix = {
        ("in", "in"): b_refl / a_inc,
        ("out", "in"): b_out / a_inc,
    }
    p_inc = 2.0 * np.abs(a_inc) ** 2
    t_flux_port = np.abs(flux_out_port.get_flux()) / np.maximum(p_inc, 1e-30)

    modal_s21 = {}
    for m_idx in multimode_indices:
        ap_m, am_m = mon_out.get_amplitudes(int(m_idx))
        bm = am_m if out_inward > 0 else ap_m
        modal_s21[int(m_idx)] = bm / a_inc

    if dft_mon is not None:
        nx, ny = pic.grid.Nx, pic.grid.Ny
        xc = 0.5 * (pic.grid.x[:-1] + pic.grid.x[1:])
        yc = 0.5 * (pic.grid.y[:-1] + pic.grid.y[1:])
        ex_raw, _ = dft_mon.get_field("Ex")
        ey_raw, _ = dft_mon.get_field("Ey")
        ex_c = ex_raw[:, :nx, :ny, 0]
        ey_c = ey_raw[:, :nx, :ny, 0]
        x_dft = xc[::stride].astype(np.float32)
        y_dft = yc[::stride].astype(np.float32)
        for idx_w, w_val in enumerate(pic.field_wavelengths):
            ex = ex_c[idx_w, ::stride, ::stride].astype(np.complex64)
            ey = ey_c[idx_w, ::stride, ::stride].astype(np.complex64)
            pic.dft_fields[w_val] = {
                "x": x_dft,
                "y": y_dft,
                "Ex": ex,
                "Ey": ey,
                "E2": (np.abs(ex) ** 2 + np.abs(ey) ** 2).astype(np.float32),
            }

    if snap_mon is not None:
        pic.port_time_trace = {
            "t": np.asarray(snap_mon.port_trace_t, dtype=np.float32),
            "Ey": np.asarray(snap_mon.port_trace_val, dtype=np.float32),
        }
        if pic.record_movie and snap_mon.frames:
            frames_f32 = np.stack(snap_mon.frames, axis=0)
            peak = max(float(np.max(np.abs(frames_f32))), 1e-30)
            pic.movie_data = {
                "x": snap_mon.x,
                "y": snap_mon.y,
                "times": np.asarray(snap_mon.times, dtype=np.float32),
                "frames": (frames_f32 / peak).astype(np.float16),
                "peak": np.float32(peak),
            }

    if flux_planes is not None:
        f_east = flux_planes["east"].get_flux() / p_inc
        f_west = -flux_planes["west"].get_flux() / p_inc
        f_north = flux_planes["north"].get_flux() / p_inc
        f_south = -flux_planes["south"].get_flux() / p_inc
        pic.flux_balance = {
            "east": f_east,
            "west": f_west,
            "lateral": f_north + f_south,
            "total_box": f_east + f_west + f_north + f_south,
        }

    return {
        "pic": pic,
        "s_matrix": s_matrix,
        "t_flux_port": t_flux_port,
        "modal_s21": modal_s21,
        "n_steps": int(pic.solver.Nt),
    }


def build_schematic_components_converter(
    domain_2d,
    ports,
    n_clad=N_AIR,
    n_pml=10,
    dx_nom=25.3e-9,
    dy_nom=25.3e-9,
    x_grid=None,
    y_grid=None,
    include_flux_box=True,
    out_label="Port 'out' ModeMonitor",
):
    """
    Construct a lightweight 2D `SolverFIT3D`, source dict, monitor dicts, and
    material name map for `wakis.schematic.plot_schematic` and `overlay_geometry`.
    """
    xmin, xmax, ymin, ymax = [float(v) for v in domain_2d[:4]]
    if x_grid is None:
        nx = max(int(round((xmax - xmin) / float(dx_nom))), 48)
        x_in = np.linspace(xmin, xmax, nx + 1)
        dx0 = x_in[1] - x_in[0]
        x_grid = np.concatenate(
            [
                x_in[0] - np.arange(n_pml, 0, -1) * dx0,
                x_in,
                x_in[-1] + np.arange(1, n_pml + 1) * dx0,
            ]
        )
    if y_grid is None:
        ny = max(int(round((ymax - ymin) / float(dy_nom))), 48)
        y_in = np.linspace(ymin, ymax, ny + 1)
        dy0 = y_in[1] - y_in[0]
        y_grid = np.concatenate(
            [
                y_in[0] - np.arange(n_pml, 0, -1) * dy0,
                y_in,
                y_in[-1] + np.arange(1, n_pml + 1) * dy0,
            ]
        )
    z_grid = np.array([0.0, float(dx_nom)], dtype=float)
    grid = GridFIT3D(x=x_grid, y=y_grid, z=z_grid, verbose=0)
    grid.remove_edge_ghost(2)
    solver_s = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=n_pml,
        use_stl=False,
        verbose=0,
    )
    eps_clad = n_to_eps(n_clad)
    solver_s.bg = [eps_clad, 1.0, 0.0]
    for comp in ("x", "y", "z"):
        solver_s.ieps[:, :, :, comp] = 1.0 / (eps_clad * epsilon_0)

    f_min = c_light / float(WL_MAX)
    f_max = c_light / float(WL_MIN)
    f0 = 0.5 * (f_min + f_max)
    fwidth = 1.25 * (f_max - f_min)
    pulse_obj = GaussianSource(frequency=f0, fwidth=fwidth, cutoff=4.2)

    sources_s = []
    monitors_s = []
    for pt in ports:
        cx = float(pt.center[0])
        hy = min(0.5 * float(pt.size[1]), 0.85 * abs(ymax))
        if pt.name == "in":
            x_src = cx - 0.05e-6
            sources_s.append(
                {
                    "kind": "sheet",
                    "type_name": "ModeSource",
                    "label": "Port 'in' ModeSource (TE₀, +x)",
                    "h_bounds": (x_src, x_src),
                    "v_bounds": (-hy, +hy),
                    "e_vec": (0.0, 1.0, 0.0),
                    "k_vec": (1.0, 0.0, 0.0),
                    "h_vec": (0.0, 0.0, 1.0),
                    "pulse": {
                        "f0": f0,
                        "fwidth": fwidth,
                        "t_peak": float(4.2 / (np.pi * fwidth)),
                        "t_end": float(pulse_obj.end_time),
                        "src_time": pulse_obj,
                    },
                }
            )
            monitors_s.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "label": "Port 'in' ModeMonitor (S₁₁)",
                    "h_bounds": (cx, cx),
                    "v_bounds": (-hy, +hy),
                }
            )
        else:
            monitors_s.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "label": out_label,
                    "h_bounds": (cx, cx),
                    "v_bounds": (-hy, +hy),
                }
            )
    if include_flux_box:
        monitors_s.append(
            {
                "kind": "box",
                "type_name": "FluxMonitor",
                "label": "2D Poynting FluxMonitor box",
                "h_bounds": (xmin + 0.22e-6, xmax - 0.15e-6),
                "v_bounds": (ymin + 0.20e-6, ymax - 0.20e-6),
            }
        )
    clad_name = "Air cladding" if abs(n_clad - 1.0) < 0.05 else "SiO₂ cladding"
    material_names = {
        round(float(eps_clad), 2): f"{clad_name} (n = {n_clad:.2f})",
    }
    return solver_s, sources_s, monitors_s, material_names


def draw_polyslab_regions(
    ax,
    structures,
    facecolor: str | None = "#1d4ed8",
    alpha: float = 0.28,
    edgecolor: str = "#0f172a",
    lw: float = 1.35,
    ls: str = "-",
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    label_color: str = "#0f172a",
    zorder: int = 5,
):
    """Draw exact 2D polygon outlines (and optional fills/labels) for `PolySlab` items."""
    n_st = len(structures)
    for idx_s, st in enumerate(structures):
        verts_um = np.asarray(st.vertices, dtype=float) * 1e6
        if n_st > 3 and idx_s >= 2 and facecolor is not None:
            frac = (idx_s - 2) / max(n_st - 3, 1)
            fc_use = plt.cm.Blues(0.32 + 0.52 * frac)
            ec_use = "none" if (idx_s > 2 and idx_s < n_st - 1) else edgecolor
            lw_use = lw if idx_s == 2 else 0.0
        else:
            fc_use = facecolor if facecolor is not None else "none"
            ec_use = edgecolor
            lw_use = lw
        ax.add_patch(
            MplPolygon(
                verts_um,
                closed=True,
                fill=facecolor is not None,
                facecolor=fc_use,
                edgecolor=ec_use,
                lw=lw_use,
                ls=ls,
                alpha=alpha if facecolor is not None else 0.92,
                zorder=zorder,
            )
        )
    if label is not None and label_xy is not None:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            fontsize=7.3,
            fontweight="bold",
            color=label_color,
            ha="center",
            va="center",
            zorder=zorder + 3,
            bbox=dict(
                boxstyle="round,pad=0.20",
                fc="white",
                ec=edgecolor,
                alpha=0.90,
                lw=0.8,
            ),
        )


def draw_launched_mode_on_schematic(
    ax,
    x_port_um: float,
    y_um: np.ndarray,
    ey: np.ndarray,
    neff: float,
    amp_um: float = 0.70,
    label_prefix: str = "Launched TE₀",
    color_fill: str = "#ef4444",
    color_line: str = "#b91c1c",
    text_offset: tuple[float, float] = (0.12, 0.85),
):
    """Overlay a port mode profile $|E_y(y)|$ on a schematic axis (in µm)."""
    y_arr = np.asarray(y_um, dtype=float)
    env = np.abs(np.asarray(ey))
    env = env / max(float(np.max(env)), 1e-30)
    x_curve = float(x_port_um) + float(amp_um) * env
    ax.fill_betweenx(
        y_arr,
        float(x_port_um),
        x_curve,
        color=color_fill,
        alpha=0.25,
        zorder=8,
    )
    ax.plot(
        x_curve,
        y_arr,
        color=color_line,
        lw=1.6,
        marker="none",
        zorder=9,
    )
    ax.text(
        float(x_port_um) + float(text_offset[0]),
        float(text_offset[1]),
        f"{label_prefix} |Ey(y)|\n(n_eff = {neff:.4f})",
        fontsize=6.9,
        fontweight="bold",
        color=color_line,
        ha="left" if amp_um >= 0 else "right",
        va="center",
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.18",
            fc="white",
            ec=color_fill,
            alpha=0.90,
            lw=0.7,
        ),
    )


def plot_setup_schematic_semilens(
    semi_structs,
    semi_ports,
    semi_dom,
    primary_data: dict,
    pic_zoom: PICSimulation,
    out_path: Path,
):
    """
    Render `00_setup_schematic.png` (FIRST figure of Tutorial 5.3, 2 panels):
    - Panel (a): Labelled 2D EIM schematic of Design 3 (6-segment semi-lens beam
      expander, 0.45 -> 9.00 um over L_tot = 9.50 um) with domain extents, 10-cell CPML,
      Si/Air/SiO2 indices, Port `in` ModeSource + launched TE0 mode + pulse inset,
      Port `out` multi-mode ModeMonitor (TE0..TE8) + expanded TE0 mode, and FluxMonitor box.
    - Panel (b): 1x Yee mesh zoom on the 50 nm Segment 3 flare (`x in [4.53, 4.74] um`)
      with cells across feature and cells per wavelength annotated.
    """
    figstyle.apply_style()
    n_eff_slab = float(primary_data.get("n_eff_slab", 2.9545))
    eps_slab = n_to_eps(n_eff_slab)
    solver_s, sources_s, monitors_s, mat_names = build_schematic_components_converter(
        semi_dom,
        semi_ports,
        n_clad=N_AIR,
        n_pml=10,
        include_flux_box=True,
        out_label="Port 'out' ModeMonitor\n(TE₀,₂,₄,₆,₈ + FluxMonitor)",
    )
    fig, (ax_main, ax_mesh) = figstyle.new_figure(
        nrows=1,
        ncols=2,
        width=13.5,
        height=5.5,
        gridspec_kw={"width_ratios": [1.9, 1.05], "wspace": 0.30},
    )
    schematic.plot_schematic(
        solver_s,
        plane="xy",
        ax=ax_main,
        unit="µm",
        structures=(),
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        show_mesh=True,
        mesh_zoom=False,
        show_pulse_inset=True,
        show_wave_packet=False,
        wavelength=WL_REF,
        title="(a) Design 3: 6-segment semi-lens expander setup & monitors",
    )
    draw_polyslab_regions(
        ax_main,
        semi_structs,
        facecolor="#1d4ed8",
        alpha=0.28,
        edgecolor="#0f172a",
        lw=1.4,
        label=(
            f"Si 6-segment semi-lens (t_Si = 260 nm on SiO₂)\n"
            f"n_Si = {N_SI:.2f} → n_eff,slab = {n_eff_slab:.4f} (ε_r = {eps_slab:.2f})\n"
            f"w₀ = 0.45 µm → w₆ = 9.00 µm over L_tot = 9.50 µm"
        ),
        label_xy=(6.65, 0.0),
    )
    ax_main.text(
        1.85,
        -4.20,
        "Lateral cladding: Air (n = 1.00, ε_r = 1.00)\nLower substrate: SiO₂ (n = 1.44)",
        fontsize=7.1,
        color="#334155",
        ha="center",
        va="center",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#94a3b8", alpha=0.90, lw=0.7),
    )
    ax_main.axvline(4.61, color="#e11d48", ls="--", lw=1.0, alpha=0.8)
    ax_main.axvline(4.66, color="#e11d48", ls="--", lw=1.0, alpha=0.8)
    ax_main.annotate(
        "Segment 3 flare\n(L₃ = 50 nm, w: 3.30 → 9.06 µm)",
        xy=(4.635, 3.30),
        xytext=(1.75, 4.15),
        fontsize=7.0,
        fontweight="bold",
        color="#9f1239",
        arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.18", fc="#fff1f2", ec="#e11d48", alpha=0.92, lw=0.7),
        zorder=10,
    )

    if "port_y_in" in primary_data and "port_ey_in" in primary_data:
        draw_launched_mode_on_schematic(
            ax_main,
            x_port_um=-1.00,
            y_um=np.asarray(primary_data["port_y_in"]) * 1e6,
            ey=np.asarray(primary_data["port_ey_in"]),
            neff=float(primary_data.get("port_neff_in", 2.4719)),
            amp_um=0.85,
            label_prefix="Launched TE₀",
            color_fill="#ef4444",
            color_line="#b91c1c",
            text_offset=(0.20, -1.65),
        )
    if "port_y_out" in primary_data and "port_ey_out" in primary_data:
        draw_launched_mode_on_schematic(
            ax_main,
            x_port_um=10.55,
            y_um=np.asarray(primary_data["port_y_out"]) * 1e6,
            ey=np.asarray(primary_data["port_ey_out"]),
            neff=float(primary_data.get("port_neff_out", 2.9647)),
            amp_um=-0.85,
            label_prefix="Expanded TE₀",
            color_fill="#0284c7",
            color_line="#0369a1",
            text_offset=(-0.95, 3.85),
        )

    # Panel (b): 1x Yee mesh zoom on the 50 nm Segment 3 flare
    slv = pic_zoom.solver
    iz = slv.Nz // 2
    ieps_y = slv.ieps.to_matrix("y")[:, :, iz]
    if hasattr(ieps_y, "get"):
        ieps_y = ieps_y.get()
    ieps_arr = np.asarray(ieps_y, dtype=float)
    eps_y = np.where(
        ieps_arr > 0.0,
        1.0 / (np.where(ieps_arr > 0.0, ieps_arr, 1.0) * epsilon_0),
        1.0,
    )
    xe = slv.grid.x * 1e6
    ye = slv.grid.y * 1e6
    im_m = ax_mesh.pcolormesh(
        xe,
        ye,
        eps_y[: slv.Nx, : slv.Ny].T,
        cmap=figstyle.CMAP_EPS,
        vmin=1.0,
        vmax=float(eps_slab) * 1.02,
        edgecolors="white",
        linewidth=0.35,
    )
    draw_polyslab_regions(
        ax_mesh,
        semi_structs,
        facecolor=None,
        edgecolor="#facc15",
        lw=1.5,
        ls="--",
    )
    ax_mesh.axvline(4.61, color="#facc15", ls=":", lw=1.2)
    ax_mesh.axvline(4.66, color="#facc15", ls=":", lw=1.2)
    nf_v = float(primary_data.get("cells_across_feature", 8.0))
    dx_min_v = float(primary_data.get("dx_min_nm", 6.25))
    cpw_v = int(round(float(primary_data.get("cpw", 20))))
    ax_mesh.annotate(
        "",
        xy=(4.61, 4.15),
        xytext=(4.66, 4.15),
        arrowprops=dict(arrowstyle="<->", color="white", lw=1.6),
    )
    ax_mesh.text(
        4.635,
        4.26,
        f"L₃ = 50 nm ({nf_v:.0f} cells, Δx_min = {dx_min_v:.2f} nm)\n{cpw_v} cells/λ in n_eff,slab",
        color="white",
        fontsize=7.1,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.20", fc="#0f172a", ec="#facc15", alpha=0.86, lw=0.8),
    )
    ax_mesh.set(
        xlim=(4.53, 4.74),
        ylim=(1.45, 4.70),
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(b) 1× Yee mesh zoom on 50 nm Seg 3 flare",
    )
    cb = fig.colorbar(im_m, ax=ax_mesh, fraction=0.046, pad=0.03)
    cb.set_label(r"Subpixel $\varepsilon_{yy} / \varepsilon_0$ (Kottke)")
    figstyle.save(fig, out_path)
    plt.close(fig)


def plot_taper_and_luneburg_schematics(out_path: Path):
    """
    Render `00b_taper_and_luneburg_schematics.png` (3 panels <= 3):
    - Panel (a): Labelled setup schematic of Design 1 (Adiabatic taper 10.0 -> 0.50 um, L_t = 20 um).
    - Panel (b): Labelled setup schematic of Design 2 (Graded-index Luneburg lens D = 11.0 um, R = 5.5 um).
    - Panel (c): Body-fitted quadtree + Delaunay hybrid triangular mesh (`wakis.unstructured.subtri_circles_mesh`)
      on the circular R = 5.5 um Luneburg lens boundary.
    """
    figstyle.apply_style()
    fig, (ax0, ax1, ax2) = figstyle.new_figure(
        nrows=1,
        ncols=3,
        width=14.0,
        height=4.5,
        gridspec_kw={"width_ratios": [1.05, 1.05, 0.95], "wspace": 0.30},
    )

    # Panel (a): Design 1 Adiabatic Taper schematic
    tap_s, tap_p, tap_d = build_taper_converter(length_taper=20.0e-6, profile="linear")
    slv_t, src_t, mon_t, mat_t = build_schematic_components_converter(
        tap_d,
        tap_p,
        n_clad=N_SIO2,
        n_pml=10,
        dx_nom=33.8e-9,
        dy_nom=33.8e-9,
        include_flux_box=False,
        out_label="Port 'out'\n(w = 0.50 µm, S₂₁)",
    )
    schematic.plot_schematic(
        slv_t,
        plane="xy",
        ax=ax0,
        unit="µm",
        structures=(),
        material_names=mat_t,
        sources=src_t,
        monitors=mon_t,
        show_mesh=True,
        mesh_zoom=False,
        show_pulse_inset=False,
        show_wave_packet=False,
        wavelength=WL_REF,
        title="(a) Design 1: Linear taper (10 → 0.5 µm)",
    )
    draw_polyslab_regions(
        ax0,
        tap_s,
        facecolor="#2563eb",
        alpha=0.28,
        edgecolor="#0f172a",
        lw=1.35,
        label="Si taper (t = 110 nm)\nn_eff,slab = 2.217",
        label_xy=(8.0, 0.0),
    )
    ov = compute_abrupt_step_overlap()
    m_in = np.abs(ov["y"] * 1e6) <= 5.2
    draw_launched_mode_on_schematic(
        ax0,
        x_port_um=-0.65,
        y_um=ov["y"][m_in] * 1e6,
        ey=ov["e_in"][m_in],
        neff=float(ov["neff_in"]),
        amp_um=2.2,
        label_prefix="Wide TE₀",
        text_offset=(0.4, 3.6),
    )

    # Panel (b): Design 2 Graded Luneburg Lens schematic
    lun_s, lun_p, lun_d, cal = build_luneburg_converter(n_shells=28)
    slv_l, src_l, mon_l, mat_l = build_schematic_components_converter(
        lun_d,
        lun_p,
        n_clad=N_SIO2,
        n_pml=10,
        dx_nom=23.9e-9,
        dy_nom=23.9e-9,
        include_flux_box=False,
        out_label="Port 'out'\n(w = 0.50 µm, S₂₁)",
    )
    schematic.plot_schematic(
        slv_l,
        plane="xy",
        ax=ax1,
        unit="µm",
        structures=(),
        material_names=mat_l,
        sources=src_l,
        monitors=mon_l,
        show_mesh=True,
        mesh_zoom=False,
        show_pulse_inset=False,
        show_wave_packet=False,
        wavelength=WL_REF,
        title="(b) Design 2: Graded Luneburg lens (D = 11 µm)",
    )
    draw_polyslab_regions(
        ax1,
        lun_s,
        facecolor="#1d4ed8",
        alpha=0.80,
        edgecolor="#0f172a",
        lw=1.25,
        label="GRIN Luneburg dome\nt(r) = 110–356 nm\nn_eff(r) = 2.22–3.13",
        label_xy=(0.0, 0.0),
    )
    draw_launched_mode_on_schematic(
        ax1,
        x_port_um=-6.10,
        y_um=ov["y"][m_in] * 1e6,
        ey=ov["e_in"][m_in],
        neff=float(ov["neff_in"]),
        amp_um=1.35,
        label_prefix="Wide TE₀",
        text_offset=(0.25, 3.8),
    )

    # Panel (c): Body-fitted quadtree+Delaunay hybrid mesh (wakis.unstructured.subtri_circles_mesh)
    mesh_u, _, info_u = subtri_circles_mesh(
        circles=[(0.0, 0.0, 5.5)],
        h=0.40,
        levels=1,
        half_width=7.2,
        band=2.0,
        grade=2.0,
        inner_fine=False,
    )
    edge_segs = mesh_u.nodes[mesh_u.edges]
    ax2.add_collection(
        LineCollection(edge_segs, colors="#059669", linewidths=0.45, alpha=0.62, zorder=2)
    )
    theta_c = np.linspace(0.0, 2.0 * np.pi, 240)
    ax2.plot(
        5.5 * np.cos(theta_c),
        5.5 * np.sin(theta_c),
        color="#dc2626",
        lw=1.7,
        marker="none",
        label="Exact circular rim (R = 5.5 µm)",
        zorder=5,
    )
    n_sq = int(info_u["n_squares_total"])
    n_tr = int(info_u["n_triangles"])
    ax2.text(
        0.03,
        0.97,
        (
            f"wakis.unstructured.subtri_circles_mesh\n"
            f"Cells: {n_sq + n_tr:,} ({n_sq} squares + {n_tr} tris)\n"
            f"Body-fitted triangles on R = 5.5 µm rim,\n"
            f"graded quadtree squares in core & cladding"
        ),
        transform=ax2.transAxes,
        fontsize=6.8,
        va="top",
        ha="left",
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#059669", alpha=0.92, lw=0.8),
    )
    ax2.set(
        aspect="equal",
        xlim=(-7.2, 7.2),
        ylim=(-7.2, 7.2),
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(c) Body-fitted hybrid mesh (Luneburg rim)",
    )
    ax2.legend(loc="lower right", fontsize=6.9)
    figstyle.save(fig, out_path)
    plt.close(fig)


def plot_mesh_upgrade_comparison_semilens(
    semi_structs,
    d_uniform: dict,
    d_nonuniform: dict,
    d_ref: dict,
    meta_uniform: dict,
    meta_nonuniform: dict,
    meta_ref: dict,
    grid_uniform: GridFIT3D,
    grid_nonuniform: GridFIT3D,
    out_path: Path,
):
    """
    Render `04b_mesh_upgrade_comparison.png` (3 panels <= 3):
    - Panel (a): Baseline uniform Cartesian mesh with Segment 3 override (`kottke_L3`).
    - Panel (b): Upgraded graded non-uniform Cartesian mesh (`kottke_nonuniform`, `NonUniformSemilensPIC`).
    - Panel (c): Broadband total port flux, even-mode sum, and TE0 transmission vs 40 cpw reference.
    """
    figstyle.apply_style()
    fig, (ax0, ax1, ax2) = figstyle.new_figure(
        nrows=1,
        ncols=3,
        width=13.8,
        height=4.4,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.15], "wspace": 0.30},
    )

    for ax, grid_obj, meta_d, d_npz, ttl, col_grid in (
        (
            ax0,
            grid_uniform,
            meta_uniform,
            d_uniform,
            "(a) Baseline uniform + override (kottke_L3)",
            "#2563eb",
        ),
        (
            ax1,
            grid_nonuniform,
            meta_nonuniform,
            d_nonuniform,
            "(b) Upgraded graded mesh (kottke_nonuniform)",
            "#059669",
        ),
    ):
        xg = np.asarray(grid_obj.x, dtype=float) * 1e6
        yg = np.asarray(grid_obj.y, dtype=float) * 1e6
        xl = (float(xg[0]), float(xg[-1]))
        yl = (float(yg[0]), float(yg[-1]))
        xs = xg[::4]
        ys = yg[::4]
        segs = [[(xv, yl[0]), (xv, yl[1])] for xv in xs] + [
            [(xl[0], yv), (xl[1], yv)] for yv in ys
        ]
        ax.add_collection(
            LineCollection(segs, colors=col_grid, linewidths=0.42, alpha=0.48, zorder=2)
        )
        draw_polyslab_regions(
            ax,
            semi_structs,
            facecolor="#1e40af" if ax is ax0 else "#065f46",
            alpha=0.28,
            edgecolor="#0f172a",
            lw=1.3,
        )
        ax.axvline(4.61, color="#dc2626", ls="--", lw=1.0, alpha=0.8)
        ax.axvline(4.66, color="#dc2626", ls="--", lw=1.0, alpha=0.8)

        i0 = int(np.argmin(np.abs(d_npz["wavelengths"] - WL_REF)))
        t_fl0 = float(d_npz["T_flux_port"][i0])
        t_te0 = float(np.abs(d_npz["S21"][i0]) ** 2)
        t_ev0 = sum(
            float(np.abs(d_npz[f"S21_TE{k}"][i0]) ** 2)
            for k in (0, 2, 4, 6, 8)
            if f"S21_TE{k}" in d_npz
        ) or t_te0
        share0 = (t_te0 / max(t_ev0, 1e-30)) * 100.0
        gs = meta_d["grid_shape"]
        n_c = int(meta_d["n_cells"])
        t_s = float(meta_d["elapsed_s"])
        n_f = float(meta_d["cells_across_feature"])
        dx_min = float(meta_d["dx_min_nm"])

        summary = (
            f"Grid: {gs[0]}×{gs[1]}×1 = {n_c:,} cells\n"
            f"50 nm Seg 3: {n_f:.1f} cells (Δx_min = {dx_min:.2f} nm)\n"
            f"Wall time: {t_s:.1f} s\n"
            f"T_flux(1550) = {t_fl0 * 100:.2f}%, T_even = {t_ev0 * 100:.2f}%\n"
            f"T_TE₀ = {t_te0 * 100:.2f}% (TE₀ share = {share0:.2f}%)"
        )
        if ax is ax1:
            cell_red = (1.0 - n_c / float(meta_uniform["n_cells"])) * 100.0
            time_red = (1.0 - t_s / float(meta_uniform["elapsed_s"])) * 100.0
            summary += f"\nSavings: −{cell_red:.1f}% cells, −{time_red:.1f}% time"

        ax.text(
            0.03,
            0.97,
            summary,
            transform=ax.transAxes,
            fontsize=6.7,
            va="top",
            ha="left",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=col_grid, alpha=0.92, lw=0.8),
        )
        ax.set(
            aspect="equal",
            xlim=xl,
            ylim=yl,
            xlabel="x [µm]",
            ylabel="y [µm]",
            title=ttl,
        )

    # Panel (c): Broadband T_even and T_TE0 comparison vs 40 cpw reference
    for d_k, lbl_k, col_k, ls_ev, ls_te0 in (
        (
            d_uniform,
            f"Uniform L3 ({meta_uniform['n_cells']:,} cells, {meta_uniform['elapsed_s']:.0f} s)",
            "#2563eb",
            "-",
            "--",
        ),
        (
            d_nonuniform,
            f"Graded L3 ({meta_nonuniform['n_cells']:,} cells, {meta_nonuniform['elapsed_s']:.0f} s)",
            "#059669",
            "-.",
            ":",
        ),
        (
            d_ref,
            f"Ref 40 cpw ({meta_ref['n_cells']:,} cells, {meta_ref['elapsed_s']:.0f} s)",
            "#dc2626",
            "-",
            "--",
        ),
    ):
        w = d_k["wavelengths"] * 1e9
        idx_s = np.argsort(w)
        ws = w[idx_s]
        mb = (ws >= 1505.0) & (ws <= 1595.0)
        te0_arr = np.abs(d_k["S21"][idx_s][mb]) ** 2
        ev_arr = sum(
            np.abs(d_k[f"S21_TE{k}"][idx_s][mb]) ** 2
            for k in (0, 2, 4, 6, 8)
            if f"S21_TE{k}" in d_k
        )
        ax2.plot(
            ws[mb],
            ev_arr * 100.0,
            color=col_k,
            ls=ls_ev,
            lw=2.0,
            marker="none",
            label=f"T_even: {lbl_k}",
        )
        ax2.plot(
            ws[mb],
            te0_arr * 100.0,
            color=col_k,
            ls=ls_te0,
            lw=1.7,
            alpha=0.85,
            marker="none",
            label=f"T_TE₀: {lbl_k}",
        )

    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Guided power transmission [%]",
        ylim=(68.0, 100.0),
        title="(c) Equal-accuracy T_even & T_TE₀ vs 40 cpw ref",
    )
    ax2.grid(True, ls=":", alpha=0.45)
    ax2.legend(loc="center right", fontsize=6.4)
    figstyle.save(fig, out_path)
    plt.close(fig)


def render_movie_with_overlay(
    x_m: np.ndarray,
    y_m: np.ndarray,
    frames: np.ndarray,
    times_s: np.ndarray,
    structures,
    domain_2d,
    ports,
    out_path: Path,
    fps: int = 12,
):
    """
    Render `15_pulse_propagation.gif` with `schematic.overlay_geometry` and
    labelled waveguide/cladding regions on every frame (kept < 5 MB).
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    figstyle.apply_style()
    solver_s, _, _, mat_names = build_schematic_components_converter(
        domain_2d,
        ports,
        n_clad=N_AIR,
        n_pml=10,
        include_flux_box=False,
    )
    x_um = np.asarray(x_m, dtype=float) * 1e6
    y_um = np.asarray(y_m, dtype=float) * 1e6
    f_arr = np.asarray(frames, dtype=np.float32)
    t_fs = np.asarray(times_s, dtype=float) * 1e15

    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=82)
    im = ax.pcolormesh(
        x_um,
        y_um,
        f_arr[0].T,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-0.85,
        vmax=0.85,
        shading="auto",
    )
    schematic.overlay_geometry(
        ax,
        solver_s,
        plane="xy",
        unit="µm",
        structures=(),
        material_names=mat_names,
        sources=(),
        monitors=(),
        outline_color="#0f172a",
    )
    draw_polyslab_regions(
        ax,
        structures,
        facecolor=None,
        edgecolor="#0f172a",
        lw=1.25,
        label="Si 6-segment semi-lens\n(n_eff,slab = 2.955)",
        label_xy=(6.65, 0.0),
    )
    ax.text(
        1.80,
        -4.35,
        "Air cladding (n = 1.00)",
        fontsize=7.2,
        color="#0f172a",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#64748b", alpha=0.85, lw=0.7),
    )
    ax.set(
        aspect="equal",
        xlim=(float(x_um.min()), float(x_um.max())),
        ylim=(float(y_um.min()), float(y_um.max())),
        xlabel="x [µm]",
        ylabel="y [µm]",
    )
    ttl = ax.set_title(f"6-segment semi-lens pulse expansion (t = {t_fs[0]:.1f} fs)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"Normalized $E_y(x, y, t)$")
    fig.tight_layout()

    def _update(i):
        im.set_array(f_arr[i].T.ravel())
        ttl.set_text(f"6-segment semi-lens pulse expansion (t = {t_fs[i]:.1f} fs)")
        return (im, ttl)

    anim = FuncAnimation(fig, _update, frames=len(f_arr), interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


