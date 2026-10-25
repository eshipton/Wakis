# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Tutorial 5.4 (Strip-to-Slot Waveguide Mode Converters)
that extend ``wakis.templates.photonics`` with:

1. ``asymmetric_taper`` — planar waveguide rail taper with independent lower
   and upper transverse edge profiles ``y_bot(x)`` and ``y_top(x)`` and
   automatic vertex deduplication at sharp (zero-width) tips.
2. ``slot_straight`` — straight dual-rail slot waveguide polygon pair.
3. ``slot_waveguide_neff`` and ``slot_waveguide_mode`` — exact 1D 5-layer
   symmetric slot waveguide eigenvalue and field-profile solver (Almeida
   et al., Opt. Lett. 29, 1209, 2004) for the in-plane quasi-TE mode
   (normal ``E_y``, tangential ``H_z``).
"""

import numpy as np
from scipy.optimize import brentq

from wakis.templates.photonics import _clean_polygon


def asymmetric_taper(
    x0,
    x1,
    y_bot0,
    y_top0,
    y_bot1,
    y_top1,
    profile="linear",
    n=64,
):
    """
    Counter-clockwise polygon vertices of a planar taper with independent
    lower and upper edges ``y_bot(x)`` and ``y_top(x)``.

    Automatically collapses pinched endpoints (``y_bot == y_top``) into a
    single vertex so that sharp taper tips do not produce degenerate
    zero-length polygon edges.

    Parameters
    ----------
    x0, x1 : float
        Start and end axial coordinates [m] (`x0 != x1`).
    y_bot0, y_top0 : float
        Lower and upper y coordinates at ``x0`` [m] (`y_bot0 <= y_top0`).
    y_bot1, y_top1 : float
        Lower and upper y coordinates at ``x1`` [m] (`y_bot1 <= y_top1`).
    profile : {'linear', 'cosine'} or callable, optional
        Interpolation profile along normalized axial coordinate ``s in [0, 1]``.
    n : int, optional
        Number of axial sample points when ``profile != 'linear'``.

    Returns
    -------
    vertices : ndarray, shape (M, 2)
        Counter-clockwise polygon vertices `(x, y)` [m].
    """
    x0, x1 = float(x0), float(x1)
    y_bot0, y_top0 = float(y_bot0), float(y_top0)
    y_bot1, y_top1 = float(y_bot1), float(y_top1)
    if np.isclose(x0, x1, atol=1e-15):
        raise ValueError("x0 and x1 must be distinct.")
    if y_bot0 > y_top0 + 1e-15 or y_bot1 > y_top1 + 1e-15:
        raise ValueError("Require y_bot <= y_top at both ends of asymmetric_taper.")
    if np.isclose(y_bot0, y_top0, atol=1e-15) and np.isclose(
        y_bot1, y_top1, atol=1e-15
    ):
        raise ValueError("Taper cannot have zero width at both ends.")

    n_pts = 2 if profile == "linear" else max(int(n), 2)
    s = np.linspace(0.0, 1.0, n_pts)
    if profile == "linear":
        u = s
    elif profile == "cosine":
        u = 0.5 * (1.0 - np.cos(np.pi * s))
    elif callable(profile):
        u = np.asarray(profile(s), dtype=float)
    else:
        raise ValueError(f"Unknown taper profile {profile!r}.")

    x = x0 + (x1 - x0) * s
    y_bot = y_bot0 + (y_bot1 - y_bot0) * u
    y_top = y_top0 + (y_top1 - y_top0) * u

    lower = np.column_stack([x, y_bot])
    upper = np.column_stack([x[::-1], y_top[::-1]])

    # Drop duplicate vertex if right end (x1) or left end (x0) is a sharp tip
    if np.allclose(lower[-1], upper[0], atol=1e-14):
        upper = upper[1:]
    if len(upper) > 0 and np.allclose(upper[-1], lower[0], atol=1e-14):
        upper = upper[:-1]

    return _clean_polygon(np.vstack([lower, upper]), ccw=True)


def slot_straight(x0, x1, width_slot, gap_slot, y0=0.0):
    """
    Return counter-clockwise `(verts_bot, verts_top)` polygon vertex arrays [m]
    for a straight symmetric dual-rail slot waveguide along `x`.

    Parameters
    ----------
    x0, x1 : float
        Start and end axial coordinates [m].
    width_slot : float
        Total outer width across both rails and the central slot [m].
    gap_slot : float
        Central low-index slot width [m] (`0 < gap_slot < width_slot`).
    y0 : float, optional
        Transverse centerline coordinate [m].

    Returns
    -------
    verts_bot, verts_top : tuple of ndarray, each shape (4, 2)
        Counter-clockwise vertices [m] of the lower and upper rails.
    """
    w_tot = float(width_slot)
    g = float(gap_slot)
    if not (0.0 < g < w_tot):
        raise ValueError("Require 0 < gap_slot < width_slot.")
    half_tot = 0.5 * w_tot
    half_gap = 0.5 * g
    yc = float(y0)
    verts_bot = asymmetric_taper(
        x0, x1, yc - half_tot, yc - half_gap, yc - half_tot, yc - half_gap
    )
    verts_top = asymmetric_taper(
        x0, x1, yc + half_gap, yc + half_tot, yc + half_gap, yc + half_tot
    )
    return verts_bot, verts_top


def slot_waveguide_neff(n_high, n_low, slot_width, rail_width, wavelength):
    """
    Analytical fundamental effective index `n_eff` of a symmetric 5-layer
    slot waveguide (`n_low / n_high / n_low / n_high / n_low`) for the
    in-plane quasi-TE mode (`E_y` normal to the slot walls, `H_z` tangential).

    Solves the exact 5-layer transcendental dispersion relation of
    Almeida et al., Opt. Lett. 29(11), 1209-1211 (2004).

    Parameters
    ----------
    n_high : float
        Refractive index of the two high-index rails (in 2D EIM, `n_eff,slab`).
    n_low : float
        Refractive index of the central slot and outer cladding (`n_SiO2`).
    slot_width : float
        Central low-index slot width `g` [m].
    rail_width : float
        Single high-index rail width `w_rail = (w_slot - g) / 2` [m].
    wavelength : float
        Vacuum wavelength [m].

    Returns
    -------
    neff : float
        Fundamental symmetric slot-mode effective index.
    """
    n_h = float(n_high)
    n_s = float(n_low)
    if not (n_s < n_h):
        raise ValueError("Require n_low < n_high.")
    a = 0.5 * float(slot_width)
    w_r = float(rail_width)
    if a < 0.0 or w_r <= 0.0:
        raise ValueError("Require slot_width >= 0 and rail_width > 0.")
    k0 = 2.0 * np.pi / float(wavelength)
    r = (n_h / n_s) ** 2

    def dispersion(neff):
        kh = k0 * np.sqrt(max(n_h**2 - neff**2, 0.0))
        gs = k0 * np.sqrt(max(neff**2 - n_s**2, 0.0))
        tanh_gsa = np.tanh(gs * a)
        return np.sin(kh * w_r) * (kh**2 - r**2 * gs**2 * tanh_gsa) - (
            np.cos(kh * w_r) * r * gs * kh * (1.0 + tanh_gsa)
        )

    n_grid = np.linspace(n_s + 1e-7, n_h - 1e-7, 600)
    vals = [dispersion(val) for val in n_grid]
    for i in range(len(n_grid) - 2, -1, -1):
        if vals[i] * vals[i + 1] <= 0.0:
            return float(brentq(dispersion, n_grid[i], n_grid[i + 1], xtol=1e-12))
    raise RuntimeError("No fundamental slot mode root found in (n_low, n_high).")


def slot_waveguide_mode(
    n_high,
    n_low,
    slot_width,
    rail_width,
    wavelength=1550e-9,
    y_span=1.6e-6,
    n_pts=801,
):
    """
    Analytical 1D transverse mode profile `E_y(y)`, `D_y(y)`, and `H_z(y)` of
    a symmetric 5-layer slot waveguide (Almeida et al., 2004).

    Parameters
    ----------
    n_high : float
        High-index rail refractive index (in 2D EIM, `n_eff,slab`).
    n_low : float
        Low-index slot and cladding refractive index (`n_SiO2`).
    slot_width : float
        Central low-index slot width `g` [m].
    rail_width : float
        Width of each high-index rail `w_rail` [m].
    wavelength : float, optional
        Vacuum wavelength [m].
    y_span : float, optional
        Total transverse coordinate span `[-y_span/2, +y_span/2]` [m].
    n_pts : int, optional
        Number of transverse sample points.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - `'y'`: transverse coordinates [m],
        - `'n_profile'`: refractive index profile `n(y)`,
        - `'Hz'`: normalized magnetic field `H_z(y)`,
        - `'Ey'`: normalized normal electric field `E_y(y)` (`max|E_y| = 1`),
        - `'Dy_norm'`: continuous normalized displacement `(n(y)/n_low)^2 E_y(y)`,
        - `'neff'`: fundamental mode effective index,
        - `'discontinuity_ratio'`: `(n_high / n_low)^2` jump at `y = +-slot_width/2`,
        - `'slot_confinement'`: fraction of `|E_y|^2` integral inside `|y| <= slot_width/2`.
    """
    n_h = float(n_high)
    n_s = float(n_low)
    a = 0.5 * float(slot_width)
    w_r = float(rail_width)
    b = a + w_r
    neff = slot_waveguide_neff(n_h, n_s, slot_width, rail_width, wavelength)

    k0 = 2.0 * np.pi / float(wavelength)
    kh = k0 * np.sqrt(max(n_h**2 - neff**2, 0.0))
    gs = k0 * np.sqrt(max(neff**2 - n_s**2, 0.0))
    r = (n_h / n_s) ** 2

    y = np.linspace(-0.5 * float(y_span), 0.5 * float(y_span), int(n_pts))
    ay = np.abs(y)

    in_slot = ay <= a
    in_rail = (ay > a) & (ay <= b)
    in_clad = ay > b

    # Continuity of Hz and (1/n^2) dHz/dy at |y| = a and |y| = b
    B = np.cosh(gs * a)
    C = r * (gs / max(kh, 1e-30)) * np.sinh(gs * a)
    D = B * np.cos(kh * w_r) + C * np.sin(kh * w_r)

    hz = np.empty_like(y)
    hz[in_slot] = np.cosh(gs * ay[in_slot])
    hz[in_rail] = B * np.cos(kh * (ay[in_rail] - a)) + C * np.sin(
        kh * (ay[in_rail] - a)
    )
    hz[in_clad] = D * np.exp(-gs * (ay[in_clad] - b))

    n_prof = np.where(in_rail, n_h, n_s)
    # Ey(y) is proportional to Hz(y) / n^2(y)
    ey_raw = hz / (n_prof**2)
    scale = float(np.max(np.abs(ey_raw))) or 1.0
    ey = ey_raw / scale
    hz_norm = hz / ( float(np.max(np.abs(hz))) or 1.0 )
    dy_norm = ((n_prof / n_s) ** 2) * ey

    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    slot_conf = float(np.sum(ey[in_slot] ** 2) * dy / (np.sum(ey**2) * dy))

    return {
        "y": y,
        "n_profile": n_prof,
        "Hz": hz_norm,
        "Ey": ey,
        "Dy_norm": dy_norm,
        "neff": neff,
        "discontinuity_ratio": r,
        "slot_confinement": slot_conf,
    }


# ---------------------------------------------------------------------------
# Graded non-uniform Cartesian mesh & schematic/overlay helpers (Tutorial 5.4)
# ---------------------------------------------------------------------------

from pathlib import Path
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import matplotlib.pyplot as plt
from scipy.constants import c as c_light, epsilon_0

from wakis import GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.templates import Simulation, meshing
from wakis.templates.photonics import (
    PICSimulation,
    _clean_polygon,
    n_to_eps,
    slab_neff,
)


def build_mmi_graded_axes(
    domain,
    length_conv: float = 6.0e-6,
    length_mmi: float = 1.38e-6,
    width_mmi: float = 1.24e-6,
    width_slot: float = 0.62e-6,
    gap_slot: float = 0.10e-6,
    n_pml: int = 10,
    cpw: float = 20.0,
    n_feature: int = 8,
    dx_junc: float | None = None,
    dx_lead: float | None = None,
    dy_rail: float | None = None,
    dy_mmi: float | None = None,
    dy_clad: float | None = None,
    grade_ratio: float = 1.22,
):
    """
    Construct graded non-uniform Cartesian node arrays ``(x, y, z)`` for Design 1
    (Symmetric MMI Strip-to-Slot Mode Converter):

    - Exact ``n_feature`` cells across the central low-index slot ``|y| <= gap_slot / 2``
      (``dy_slot = gap_slot / n_feature = 12.5 nm`` at ``n_feature = 8``) with grid
      lines aligned with the slot walls ``y = +-gap_slot / 2``,
    - Fine transverse spacing ``dy_rail`` across the silicon strip/slot rails
      ``|y| in [gap_slot / 2, width_slot / 2]`` and ``dy_mmi`` across the MMI body
      ``|y| in [width_slot / 2, width_mmi / 2]``,
    - Smoothly graded coarse cells ``dy_clad`` in the outer ``SiO2`` cladding
      ``|y| > width_mmi / 2``,
    - Fine axial spacing ``dx_junc`` across the abrupt strip-to-MMI entrance
      (``x ~ 0``) and MMI-to-slot opening (``x ~ length_mmi``), paired with
      non-square axial cells ``dx_lead > dy_slot`` along the straight leads and
      adiabatic slot-rail taper.
    """
    if len(domain) == 4:
        xmin, xmax, ymin, ymax = domain
        zmin_2d, zmax_2d = -0.5e-6, 0.5e-6
    else:
        xmin, xmax, ymin, ymax, zmin_2d, zmax_2d = domain

    scale = 20.0 / max(float(cpw), 8.0)
    hg = 0.5 * float(gap_slot)
    hsl = 0.5 * float(width_slot)
    hm = 0.5 * float(width_mmi)
    L_m = float(length_mmi)
    L_c = float(length_conv)

    dy_s = float(gap_slot) / max(int(n_feature), 1)
    dy_r = float(21.67e-9 * scale if dy_rail is None else dy_rail)
    dy_m = float(25.83e-9 * scale if dy_mmi is None else dy_mmi)
    dy_cl = float(65.0e-9 * scale if dy_clad is None else dy_clad)

    dx_j = float(22.0e-9 * scale if dx_junc is None else dx_junc)
    dx_ld = float(34.0e-9 * scale if dx_lead is None else dx_lead)

    x_nodes = meshing.graded_axis(
        [xmin, -0.15e-6, 0.15e-6, L_m - 0.18e-6, L_m + 0.27e-6, L_c, xmax],
        [dx_ld, dx_j, dx_ld, dx_j, dx_ld, dx_ld],
        grading=grade_ratio,
    )
    y_pos = meshing.graded_axis(
        [0.0, hg, hsl, hm, ymax],
        [dy_s, dy_r, dy_m, dy_cl],
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


class NonUniformStripToSlotPIC(PICSimulation):
    """
    2D EIM ``PICSimulation`` subclass that replaces the uniform-background grid
    with ``build_mmi_graded_axes`` (fine ``dy = 12.5 nm`` across the ``100 nm``
    oxide slot, graded coarse ``dy`` in the outer ``SiO2`` cladding, and
    non-square ``dx > dy`` along the leads and adiabatic taper).
    """

    def __init__(
        self,
        *args,
        length_conv: float = 6.0e-6,
        length_mmi: float = 1.38e-6,
        width_mmi: float = 1.24e-6,
        width_slot: float = 0.62e-6,
        gap_slot: float = 0.10e-6,
        n_feature: int = 8,
        grade_ratio: float = 1.22,
        **kwargs,
    ):
        self.length_conv = float(length_conv)
        self.length_mmi = float(length_mmi)
        self.width_mmi = float(width_mmi)
        self.width_slot = float(width_slot)
        self.gap_slot = float(gap_slot)
        self.n_feature = int(n_feature)
        self.grade_ratio = float(grade_ratio)
        kwargs["mesh_overrides"] = ()
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
        axes_xyz = build_mmi_graded_axes(
            domain,
            length_conv=self.length_conv,
            length_mmi=self.length_mmi,
            width_mmi=self.width_mmi,
            width_slot=self.width_slot,
            gap_slot=self.gap_slot,
            n_pml=self.sim.n_pml,
            cpw=self.cells_per_wavelength,
            n_feature=self.n_feature,
            grade_ratio=self.grade_ratio,
        )
        self.sim._axis = lambda axis: axes_xyz[int(axis)]
        self.solver = self.sim.build()
        self.solver.ieps[:, :, :, "z"] = 0.0
        self.grid = self.sim.grid
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        return self.solver


class _SchematicSolverProxy:
    """Lightweight proxy around ``SolverFIT3D`` that hides ``ieps`` from ``wakis.schematic``
    so 2D ``PolySlab`` geometries can be drawn directly without bounding-box fallback rectangles."""

    def __init__(self, solver, eps_clad: float):
        self._solver = solver
        self.grid = solver.grid
        self.Nx = solver.Nx
        self.Ny = solver.Ny
        self.Nz = solver.Nz
        self.dt = solver.dt
        self.n_pml = getattr(solver, "n_pml", 10)
        self.bc_low = getattr(solver, "bc_low", ["pml", "pml", "periodic"])
        self.bc_high = getattr(solver, "bc_high", ["pml", "pml", "periodic"])
        self.bg = [float(eps_clad), 1.0, 0.0]
        self._assigned_structures = ()


def _build_port_sources_and_monitors(pic, include_flux_box: bool = True):
    """Build ``(sources, monitors)`` dicts for ``wakis.schematic`` from ``pic``."""
    xmin, xmax = float(pic.grid.xmin), float(pic.grid.xmax)
    ymin, ymax = float(pic.grid.ymin), float(pic.grid.ymax)
    dx_bg = float(np.median(np.diff(pic.grid.x)))
    dy_bg = float(np.median(np.diff(pic.grid.y)))
    pulse_obj = pic.sim.pulse
    f0 = float(pulse_obj.frequency)
    fwidth = float(pulse_obj.fwidth)

    sources_s = []
    monitors_s = []
    for pt in pic.ports:
        cx = float(pt.center[0])
        hy = min(0.5 * float(pt.size[1]), 0.85 * abs(ymax))
        if pt.name == "in":
            x_src = cx - 0.05e-6
            sources_s.append(
                {
                    "kind": "sheet",
                    "type_name": "ModeSource",
                    "label": "Port 'in' ModeSource (strip TE₀, +x)",
                    "h_bounds": (x_src, x_src),
                    "v_bounds": (-hy, +hy),
                    "e_vec": (0.0, 1.0, 0.0),
                    "k_vec": (1.0, 0.0, 0.0),
                    "h_vec": (0.0, 0.0, 1.0),
                    "pulse": {
                        "f0": f0,
                        "fwidth": fwidth,
                        "t_peak": float(4.0 / (np.pi * fwidth)),
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
                    "label": "Port 'out' ModeMonitor (slot TE₀, S₂₁)",
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
                "h_bounds": (xmin + 2.0 * dx_bg, xmax - 2.0 * dx_bg),
                "v_bounds": (ymin + 2.0 * dy_bg, ymax - 2.0 * dy_bg),
            }
        )
    return sources_s, monitors_s


def _draw_launched_and_slot_modes(
    ax,
    x_in_um: float,
    y_in_um: np.ndarray,
    ey_in: np.ndarray,
    x_out_um: float,
    y_out_um: np.ndarray,
    ey_out: np.ndarray,
    amp_um: float = 0.52,
):
    """Overlay the launched strip TE0 mode and output discontinuous slot TE0 mode."""
    m_in = np.abs(y_in_um) <= 0.70
    yi = y_in_um[m_in]
    ei = np.abs(ey_in[m_in])
    ei = ei / (float(np.max(ei)) or 1.0)
    x_prof_in = x_in_um + amp_um * ei
    ax.fill_betweenx(
        yi, x_in_um, x_prof_in, color="#d62728", alpha=0.22, zorder=8
    )
    ax.plot(
        x_prof_in,
        yi,
        color="#d62728",
        lw=1.7,
        zorder=9,
        label=r"Launched strip $\mathrm{TE}_0$ $|E_y(y)|$",
    )

    m_out = np.abs(y_out_um) <= 0.70
    yo = y_out_um[m_out]
    eo = np.abs(ey_out[m_out])
    eo = eo / (float(np.max(eo)) or 1.0)
    x_prof_out = x_out_um - amp_um * eo
    ax.fill_betweenx(
        yo, x_prof_out, x_out_um, color="#1b9e77", alpha=0.22, zorder=8
    )
    ax.plot(
        x_prof_out,
        yo,
        color="#1b9e77",
        lw=1.7,
        zorder=9,
        label=r"Output slot $\mathrm{TE}_0$ $|E_y(y)|$ ($4.14\times$ jump)",
    )


def _draw_polyslab_polygons(
    ax,
    structures,
    facecolor: str | None = "#2563eb",
    alpha: float = 0.28,
    edgecolor: str = "#0f172a",
    lw: float = 1.3,
    ls: str = "-",
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    zorder: int = 5,
):
    """Draw 2D ``PolySlab`` polygon outlines/fills on ``ax`` (in µm)."""
    for s in structures:
        ax.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                fill=facecolor is not None,
                facecolor=facecolor if facecolor is not None else "none",
                edgecolor=edgecolor,
                lw=lw,
                ls=ls,
                alpha=alpha if facecolor is not None else 0.90,
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
            color="#0f172a",
            ha="center",
            va="center",
            zorder=zorder + 3,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=edgecolor, alpha=0.90, lw=0.8),
        )


def plot_setup_schematic(
    pic,
    primary_data: dict,
    out_path: str | Path,
):
    """
    Render ``00_setup_schematic.png`` (first figure of Tutorial 5.4):
    - Panel (a): Labelled 2D setup schematic of Design 1 (Symmetric MMI
      strip-to-slot mode converter, ``L = 6.0 um``) with domain extents, CPML
      boundaries, Si/SiO2 indices, input strip ``ModeSource`` with launched
      ``TE0`` profile, output slot ``ModeMonitor`` with discontinuous slot
      ``TE0`` profile, and 2D Poynting ``FluxMonitor`` box.
    - Panel (b): ``1x`` mesh view zoomed on the ``g = 100 nm`` oxide slot
      opening (``x = 1.38 um``) showing ``8.0`` cells across the slot
      (``dy_min = 12.5 nm``) and non-square cells along the rails.
    """
    figstyle.apply_style()
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=13.6, height=5.1)

    p_in, p_out = pic.ports[0], pic.ports[1]
    sources_s, monitors_s = _build_port_sources_and_monitors(pic, include_flux_box=True)
    neff_s = float(pic.n_eff_slab)
    eps_s = neff_s**2
    eps_clad = float(n_to_eps(pic.n_clad))
    mat_names = {
        "background": f"SiO₂ slot & clad (n={pic.n_clad:.2f}, ε_r={eps_clad:.2f})",
    }
    schematic.plot_schematic(
        _SchematicSolverProxy(pic.solver, eps_clad),
        plane="xy",
        ax=ax0,
        unit="µm",
        structures=(),
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        show_mesh=True,
        mesh_zoom=False,
        show_pulse_inset=False,
        show_wave_packet=False,
        wavelength=1550e-9,
        title="(a) Design 1: Symmetric MMI strip-to-slot converter setup (L = 6.0 µm)",
    )
    ax0.set_aspect("auto")
    _draw_polyslab_polygons(
        ax0,
        pic.structures_3d,
        facecolor="#2563eb",
        alpha=0.28,
        edgecolor="#0f172a",
        lw=1.35,
        label=f"Si EIM core (h_Si=250 nm, n_Si=3.47)\nn_eff,slab={neff_s:.4f} (ε_r={eps_s:.2f})",
        label_xy=(3.55, 0.78),
    )
    ax0.text(
        3.55,
        -1.08,
        "SiO₂ cladding & 100 nm slot (n = 1.44, ε_r = 2.07)",
        fontsize=7.2,
        color="#334155",
        ha="center",
        va="center",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#94a3b8", alpha=0.90),
    )
    if "port_y_in" in primary_data and "port_y_out" in primary_data:
        _draw_launched_and_slot_modes(
            ax0,
            x_in_um=p_in.center[0] * 1e6,
            y_in_um=primary_data["port_y_in"] * 1e6,
            ey_in=primary_data["port_ey_in"],
            x_out_um=p_out.center[0] * 1e6,
            y_out_um=primary_data["port_y_out"] * 1e6,
            ey_out=primary_data["port_ey_out"],
            amp_um=0.55,
        )
    ax0.annotate(
        "MMI box\n1.38 × 1.24 µm",
        xy=(0.69, 0.62),
        xytext=(0.45, 1.18),
        fontsize=7.8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.7", alpha=0.92),
    )
    ax0.annotate(
        "100 nm SiO₂ slot\nopens at x = 1.38 µm",
        xy=(1.50, 0.0),
        xytext=(2.60, -0.68),
        fontsize=7.6,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#d95f02", alpha=0.92),
    )
    ax0.legend(loc="lower left", fontsize=7.0, framealpha=0.92)

    # Panel (b): 1x mesh zoom on the 100 nm slot opening at x = 1.38 um
    xl_z, yl_z = (1.12, 1.98), (-0.38, 0.38)
    ax0.add_patch(
        Rectangle(
            (xl_z[0], yl_z[0]),
            xl_z[1] - xl_z[0],
            yl_z[1] - yl_z[0],
            fill=False,
            ec="#d95f02",
            lw=1.3,
            ls="--",
            zorder=10,
        )
    )
    slv = pic.solver
    iz = slv.Nz // 2
    ieps_y = np.asarray(slv.ieps[:, :, iz, "y"], dtype=float)
    eps_r = np.where(ieps_y > 0, 1.0 / (np.where(ieps_y > 0, ieps_y, 1.0) * epsilon_0), 1.0)
    xe = slv.grid.x * 1e6
    ye = slv.grid.y * 1e6
    im_m = ax1.pcolormesh(
        xe,
        ye,
        eps_r.T,
        cmap=figstyle.CMAP_EPS,
        shading="flat",
        edgecolors="white",
        linewidth=0.35,
    )
    _draw_polyslab_polygons(
        ax1,
        pic.structures_3d,
        facecolor=None,
        edgecolor="#facc15",
        lw=1.4,
        ls="--",
    )
    ax1.set(
        xlim=xl_z,
        ylim=yl_z,
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(b) 1× mesh zoom on g = 100 nm slot opening (x = 1.38 µm)",
    )
    cb = fig.colorbar(im_m, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label(r"Subpixel $\varepsilon_{yy} / \varepsilon_0$ (Kottke)")
    dy_min_nm = float(np.min(np.diff(pic.grid.y))) * 1e9
    dx_med_nm = float(np.median(np.diff(pic.grid.x))) * 1e9
    cpw_eff = (1500.0 / neff_s) / dx_med_nm
    ax1.annotate(
        "",
        xy=(1.50, -0.05),
        xytext=(1.50, +0.05),
        arrowprops=dict(arrowstyle="<->", color="white", lw=1.5),
    )
    ax1.text(
        1.54,
        0.0,
        f"Slot g = 100 nm\n8.0 cells (Δy_min = {dy_min_nm:.1f} nm)\n"
        f"Δx_lead = {dx_med_nm:.1f} nm ({cpw_eff:.0f} cells/λ)",
        color="white",
        fontsize=7.3,
        fontweight="bold",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="#0f172a", ec="#facc15", alpha=0.88),
    )
    figstyle.save(fig, out_path)
    plt.close(fig)


def plot_taper_converters_schematic(
    pic_sym,
    pic_asym,
    primary_data: dict,
    out_path: str | Path,
):
    """
    Render ``00b_taper_converters_schematic.png`` (3 panels <= 3 per figure):
    - Panel (a): Labelled setup schematic of Design 2 (Symmetric Double-Tip
      Tapered-Finger Coupler, ``L = 8.0 um, d = 60 nm``, Wang et al. 2009).
    - Panel (b): ``1x`` mesh zoom on Design 2's ``d = 60 nm`` finger-to-strip
      gap at ``x = 0`` showing ``8.0`` cells (``dy_min = 7.14 nm``).
    - Panel (c): Labelled setup schematic of Design 3 (Asymmetric Lateral Taper
      Converter, ``L = 20.0 um``, Feng et al. 2007).
    """
    figstyle.apply_style()
    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=15.4, height=4.7)

    neff_s = float(pic_sym.n_eff_slab)
    eps_s = neff_s**2
    eps_clad = float(n_to_eps(pic_sym.n_clad))
    mat_names = {
        "background": f"SiO₂ clad/slot (n={pic_sym.n_clad:.2f}, ε_r={eps_clad:.2f})",
    }

    # Panel (a): Design 2 setup schematic
    p2_in, p2_out = pic_sym.ports[0], pic_sym.ports[1]
    src2, mon2 = _build_port_sources_and_monitors(pic_sym, include_flux_box=False)
    schematic.plot_schematic(
        _SchematicSolverProxy(pic_sym.solver, eps_clad),
        plane="xy",
        ax=ax0,
        unit="µm",
        structures=(),
        material_names=mat_names,
        sources=src2,
        monitors=mon2,
        show_mesh=True,
        mesh_zoom=False,
        show_pulse_inset=False,
        show_wave_packet=False,
        wavelength=1550e-9,
        title="(a) Design 2: Symmetric Tapered-Finger (L = 8 µm)",
    )
    ax0.set_aspect("auto")
    _draw_polyslab_polygons(
        ax0,
        pic_sym.structures_3d,
        facecolor="#2563eb",
        alpha=0.28,
        edgecolor="#0f172a",
        lw=1.25,
        label=f"Si fingers & taper (d=60 nm, g=100 nm)\nn_eff,slab={neff_s:.2f}",
        label_xy=(4.0, 0.78),
    )
    if "port_y_in" in primary_data and "port_y_out" in primary_data:
        _draw_launched_and_slot_modes(
            ax0,
            x_in_um=p2_in.center[0] * 1e6,
            y_in_um=primary_data["port_y_in"] * 1e6,
            ey_in=primary_data["port_ey_in"],
            x_out_um=p2_out.center[0] * 1e6,
            y_out_um=primary_data["port_y_out"] * 1e6,
            ey_out=primary_data["port_ey_out"],
            amp_um=0.65,
        )

    # Panel (b): 1x mesh zoom on Design 2's d = 60 nm finger-to-strip gap
    xl_f, yl_f = (-0.15, 0.90), (0.12, 0.34)
    slv2 = pic_sym.solver
    iz2 = slv2.Nz // 2
    ieps2 = np.asarray(slv2.ieps[:, :, iz2, "y"], dtype=float)
    eps2 = np.where(ieps2 > 0, 1.0 / (np.where(ieps2 > 0, ieps2, 1.0) * epsilon_0), 1.0)
    im2 = ax1.pcolormesh(
        slv2.grid.x * 1e6,
        slv2.grid.y * 1e6,
        eps2.T,
        cmap=figstyle.CMAP_EPS,
        shading="flat",
        edgecolors="white",
        linewidth=0.35,
    )
    _draw_polyslab_polygons(
        ax1,
        pic_sym.structures_3d,
        facecolor=None,
        edgecolor="#facc15",
        lw=1.4,
        ls="--",
    )
    ax1.set(
        xlim=xl_f,
        ylim=yl_f,
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(b) Design 2 1× mesh zoom: d = 60 nm tip gap",
    )
    cb2 = fig.colorbar(im2, ax=ax1, fraction=0.046, pad=0.03)
    cb2.set_label(r"$\varepsilon_{yy} / \varepsilon_0$")
    dy_f_nm = float(np.min(np.diff(pic_sym.grid.y))) * 1e9
    ax1.annotate(
        "",
        xy=(0.04, 0.20),
        xytext=(0.04, 0.26),
        arrowprops=dict(arrowstyle="<->", color="white", lw=1.5),
    )
    ax1.text(
        0.10,
        0.23,
        f"Finger gap d = 60 nm\n8.0 cells (Δy = {dy_f_nm:.2f} nm)\n"
        "Slot g = 100 nm: 14.0 cells",
        color="white",
        fontsize=7.2,
        fontweight="bold",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="#0f172a", ec="#facc15", alpha=0.88),
    )

    # Panel (c): Design 3 setup schematic
    p3_in, p3_out = pic_asym.ports[0], pic_asym.ports[1]
    src3, mon3 = _build_port_sources_and_monitors(pic_asym, include_flux_box=False)
    schematic.plot_schematic(
        _SchematicSolverProxy(pic_asym.solver, eps_clad),
        plane="xy",
        ax=ax2,
        unit="µm",
        structures=(),
        material_names=mat_names,
        sources=src3,
        monitors=mon3,
        show_mesh=True,
        mesh_zoom=False,
        show_pulse_inset=False,
        show_wave_packet=False,
        wavelength=1550e-9,
        title="(c) Design 3: Asymmetric Lateral Taper (L = 20 µm)",
    )
    ax2.set_aspect("auto")
    _draw_polyslab_polygons(
        ax2,
        pic_asym.structures_3d,
        facecolor="#2563eb",
        alpha=0.28,
        edgecolor="#0f172a",
        lw=1.25,
        label=f"Si asymmetric taper (L=20 µm)\nn_eff,slab={neff_s:.2f}",
        label_xy=(10.0, 0.78),
    )
    if "port_y_in" in primary_data and "port_y_out" in primary_data:
        _draw_launched_and_slot_modes(
            ax2,
            x_in_um=p3_in.center[0] * 1e6,
            y_in_um=primary_data["port_y_in"] * 1e6,
            ey_in=primary_data["port_ey_in"],
            x_out_um=p3_out.center[0] * 1e6,
            y_out_um=primary_data["port_y_out"] * 1e6,
            ey_out=primary_data["port_ey_out"],
            amp_um=1.35,
        )

    figstyle.save(fig, out_path)
    plt.close(fig)


def plot_mesh_upgrade_comparison(
    pic_uniform,
    pic_nonuniform,
    td_uniform: dict,
    td_nonuniform: dict,
    td_ref: dict | None,
    meta_uniform: dict | None,
    meta_nonuniform: dict | None,
    wl_ref: float,
    out_path: str | Path,
):
    """
    Render ``04b_mesh_upgrade_comparison.png`` comparing the uniform-background
    ``kottke_L3`` baseline against the graded non-uniform ``kottke_nonuniform``
    mesh on Design 1 (MMI converter) at equal accuracy:
    - Panel (a): Upper half shows ``kottke_L3`` mesh lines; lower half shows
      ``kottke_nonuniform`` graded transverse + non-square axial mesh lines.
    - Panel (b): Transverse cell height ``dy(y)`` [nm] vs ``y`` [um] across the
      slot, rails, MMI box, and outer ``SiO2`` cladding.
    - Panel (c): Broadband C-band conversion efficiency ``T(lambda)`` [%] and
      return loss ``R(lambda)`` comparing ``kottke_L3``, ``kottke_nonuniform``,
      and ``kottke_reference``.
    """
    figstyle.apply_style()
    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=15.4, height=4.6)

    # Panel (a): Split-half mesh view (y >= 0: kottke_L3, y <= 0: kottke_nonuniform)
    iz = pic_nonuniform.solver.Nz // 2
    ieps = np.asarray(pic_nonuniform.solver.ieps[:, :, iz, "y"])
    eps_r = np.where(ieps > 0, 1.0 / (np.where(ieps > 0, ieps, 1.0) * epsilon_0), 1.0)
    xu_nu = pic_nonuniform.grid.x * 1e6
    yu_nu = pic_nonuniform.grid.y * 1e6
    ax0.pcolormesh(xu_nu, yu_nu, eps_r.T, cmap=figstyle.CMAP_EPS, shading="flat", alpha=0.88)

    xl_m, yl_m = (-0.50, 3.20), (-1.45, 1.45)
    xu_u = pic_uniform.grid.x * 1e6
    yu_u = pic_uniform.grid.y * 1e6
    xv_u = xu_u[(xu_u >= xl_m[0]) & (xu_u <= xl_m[1])]
    yv_u = yu_u[(yu_u >= 0.0) & (yu_u <= yl_m[1])]
    segs_u = [[(x, 0.0), (x, yl_m[1])] for x in xv_u] + [
        [(xl_m[0], y), (xl_m[1], y)] for y in yv_u
    ]
    ax0.add_collection(
        LineCollection(segs_u, colors="#08519c", linewidths=0.38, alpha=0.65)
    )

    xv_nu = xu_nu[(xu_nu >= xl_m[0]) & (xu_nu <= xl_m[1])]
    yv_nu = yu_nu[(yu_nu >= yl_m[0]) & (yu_nu <= 0.0)]
    segs_nu = [[(x, yl_m[0]), (x, 0.0)] for x in xv_nu] + [
        [(xl_m[0], y), (xl_m[1], y)] for y in yv_nu
    ]
    ax0.add_collection(
        LineCollection(segs_nu, colors="#d95f02", linewidths=0.42, alpha=0.78)
    )
    ax0.axhline(0.0, color="black", lw=1.1, ls="--")
    for s in pic_nonuniform.structures_3d:
        ax0.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                fill=False,
                edgecolor="black",
                lw=1.1,
            )
        )

    n_u = int(pic_uniform.grid.Nx * pic_uniform.grid.Ny)
    n_nu = int(pic_nonuniform.grid.Nx * pic_nonuniform.grid.Ny)
    t_u_s = float(meta_uniform["elapsed_s"]) if meta_uniform else 146.50
    t_nu_s = float(meta_nonuniform["elapsed_s"]) if meta_nonuniform else 0.0
    red_cells = 100.0 * (1.0 - n_nu / n_u)
    red_time = 100.0 * (1.0 - t_nu_s / t_u_s) if t_u_s > 0 else 0.0

    ax0.text(
        0.05,
        1.15,
        f"Upper (y > 0): kottke_L3 baseline\n"
        f"{pic_uniform.grid.Nx}×{pic_uniform.grid.Ny} = {n_u:,} cells ({t_u_s:.1f} s)",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#08519c", alpha=0.92),
    )
    ax0.text(
        0.05,
        -1.28,
        f"Lower (y < 0): kottke_nonuniform\n"
        f"{pic_nonuniform.grid.Nx}×{pic_nonuniform.grid.Ny} = {n_nu:,} cells "
        f"(-{red_cells:.1f}%, {t_nu_s:.1f} s, -{red_time:.1f}%)",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d95f02", alpha=0.92),
    )
    ax0.set(
        xlim=xl_m,
        ylim=yl_m,
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(a) Baseline (top) vs graded non-uniform mesh (bottom)",
    )

    # Panel (b): Transverse cell size dy(y) [nm] vs y [um]
    yc_u = 0.5 * (yu_u[:-1] + yu_u[1:])
    dy_u = np.diff(yu_u) * 1e3
    yc_nu = 0.5 * (yu_nu[:-1] + yu_nu[1:])
    dy_nu = np.diff(yu_nu) * 1e3
    ax1.axvspan(-0.05, +0.05, color="#fdae61", alpha=0.28, label="100 nm slot (|y| ≤ 50 nm)")
    ax1.axvspan(-0.62, -0.05, color="#2b5c8f", alpha=0.12, label="Rails & MMI (|y| ≤ 620 nm)")
    ax1.axvspan(+0.05, +0.62, color="#2b5c8f", alpha=0.12)
    ax1.step(
        yc_u,
        dy_u,
        where="mid",
        color="#08519c",
        lw=1.8,
        label=f"kottke_L3 ({pic_uniform.grid.Ny} y-cells)",
    )
    ax1.step(
        yc_nu,
        dy_nu,
        where="mid",
        color="#d95f02",
        lw=2.0,
        label=f"kottke_nonuniform ({pic_nonuniform.grid.Ny} y-cells)",
    )
    ax1.set(
        xlim=(-1.65, 1.65),
        xlabel="Transverse coordinate y [µm]",
        ylabel="Cell height Δy(y) [nm]",
        title="(b) Transverse Δy(y) grading (exact 12.5 nm in slot)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper center", fontsize=7.2)

    # Panel (c): Broadband T(lambda) [%] comparison
    def _spec(d):
        w = d["wavelengths"] * 1e9
        o = np.argsort(w)
        ws = w[o]
        m = (ws >= 1505.0) & (ws <= 1595.0)
        t = np.abs(d["S21"][o][m]) ** 2 * 100.0
        i0 = int(np.argmin(np.abs(d["wavelengths"] - wl_ref)))
        t0 = float(np.abs(d["S21"][i0]) ** 2) * 100.0
        return ws[m], t, t0

    w_u, t_u, t0_u = _spec(td_uniform)
    w_nu, t_nu, t0_nu = _spec(td_nonuniform)
    if td_ref is not None:
        w_r, t_r, t0_r = _spec(td_ref)
        ax2.plot(
            w_r,
            t_r,
            color="#111111",
            lw=2.2,
            label=f"kottke_reference (R: T₁₅₅₀={t0_r:.2f}%)",
        )
    ax2.plot(
        w_u,
        t_u,
        "--",
        color="#08519c",
        lw=1.9,
        label=f"kottke_L3 ({n_u:,} cells: T₁₅₅₀={t0_u:.2f}%)",
    )
    ax2.plot(
        w_nu,
        t_nu,
        "-.",
        color="#d95f02",
        lw=2.0,
        label=f"kottke_nonuniform ({n_nu:,} cells: T₁₅₅₀={t0_nu:.2f}%)",
    )
    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Strip-to-slot coupling efficiency $|S_{21}|^2$ [%]",
        ylim=(95.5, 99.5),
        title="(c) Equal-accuracy C-band conversion efficiency",
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=7.2)

    figstyle.save(fig, out_path)
    plt.close(fig)


def overlay_converter_geometry(ax, pic, outline_color: str = "white"):
    """Overlay material outlines/labels, ports, and CPML boundaries on a field map."""
    neff_s = float(pic.n_eff_slab)
    eps_clad = float(n_to_eps(pic.n_clad))
    mat_names = {
        "background": f"SiO₂ (n={pic.n_clad:.2f})",
    }
    sources_s, monitors_s = _build_port_sources_and_monitors(pic, include_flux_box=False)
    schematic.overlay_geometry(
        ax,
        _SchematicSolverProxy(pic.solver, eps_clad),
        plane="xy",
        unit="µm",
        structures=(),
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        show_materials=False,
        label_materials=False,
        outline_color=outline_color,
    )
    _draw_polyslab_polygons(
        ax,
        pic.structures_3d,
        facecolor=None,
        edgecolor=outline_color,
        lw=1.15,
        label=f"Si EIM core (n_eff={neff_s:.2f}) | SiO₂ clad & slot (n={pic.n_clad:.2f})",
        label_xy=(0.5 * (float(pic.grid.xmin) + float(pic.grid.xmax)) * 1e6, 0.85),
    )


def render_field_map_with_overlay(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    pic,
    kind: str,
    title: str,
    out_path: str | Path,
):
    """Render a single-panel 2D field map with ``figstyle`` and ``overlay_geometry``."""
    figstyle.apply_style()
    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=9.4, height=4.2)
    x_um = np.asarray(x) * 1e6
    y_um = np.asarray(y) * 1e6
    if kind == "intensity_log":
        db_floor = -35.0
        f_abs = np.asarray(field, dtype=float)
        data_db = 10.0 * np.log10(
            np.maximum(f_abs / (float(np.max(f_abs)) or 1.0), 10.0 ** (db_floor / 10.0))
        )
        im = ax.pcolormesh(
            x_um, y_um, data_db.T, cmap="inferno", vmin=db_floor, vmax=0.0, shading="auto"
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"$10\log_{10}(|\mathbf{E}|^2 / |\mathbf{E}|_{\max}^2)$ [dB]")
        outline_col = "white"
    else:
        f_re = np.real(field)
        vmax = float(np.percentile(np.abs(f_re), 99.5)) or 1.0
        im = ax.pcolormesh(
            x_um,
            y_um,
            (f_re / vmax).T,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"Normalized $\mathrm{Re}(E_y)$")
        outline_col = "black"

    overlay_converter_geometry(ax, pic, outline_color=outline_col)
    ax.set(
        xlim=(float(x_um.min()), float(x_um.max())),
        ylim=(float(y_um.min()), float(y_um.max())),
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=title,
    )
    figstyle.save(fig, out_path)
    plt.close(fig)


def render_movie_with_overlay(
    x: np.ndarray,
    y: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray,
    pic,
    out_path: str | Path,
    fps: int = 12,
):
    """Render animated GIF ``09_pulse_propagation.gif`` (< 5 MB) with ``overlay_geometry``."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=85)
    x_um = np.asarray(x) * 1e6
    y_um = np.asarray(y) * 1e6
    f_arr = np.asarray(frames, dtype=float)
    vmax = float(np.percentile(np.abs(f_arr), 99.6)) or 1.0

    im = ax.pcolormesh(
        x_um,
        y_um,
        (f_arr[0] / vmax).T,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-1.0,
        vmax=1.0,
        shading="auto",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$E_y(x, y, t) / E_{\max}$")
    overlay_converter_geometry(ax, pic, outline_color="black")
    ax.set(
        xlim=(float(x_um.min()), float(x_um.max())),
        ylim=(float(y_um.min()), float(y_um.max())),
        xlabel="x [µm]",
        ylabel="y [µm]",
    )
    ttl = ax.set_title(
        f"Design 1 MMI strip-to-slot pulse propagation (t = {float(times[0])*1e15:.1f} fs)"
    )
    fig.tight_layout()

    def _update(frame_idx):
        im.set_array((f_arr[frame_idx] / vmax).T.ravel())
        ttl.set_text(
            f"Design 1 MMI strip-to-slot pulse propagation (t = {float(times[frame_idx])*1e15:.1f} fs)"
        )
        return (im, ttl)

    anim = FuncAnimation(fig, _update, frames=len(f_arr), interval=1000 // fps, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)

