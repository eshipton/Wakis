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
    PhotonicCircuit,
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


class NonUniformStripToSlotCircuit(PhotonicCircuit):
    """
    2D EIM ``PhotonicCircuit`` subclass that replaces the uniform-background grid
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


class _PolygonSchematicProxy:
    """Lightweight structure proxy exposing 2D polygon vertices for ``wakis.schematic``."""

    def __init__(self, structure, eps_val: float, name: str):
        self.vertices = np.asarray(structure.vertices, dtype=float)
        self.material = float(eps_val)
        self.name = name
        self.z_min = float(getattr(structure, "z_min", -0.125e-6))
        self.z_max = float(getattr(structure, "z_max", 0.125e-6))


class _SchematicSolverProxy:
    """Lightweight proxy around ``SolverFIT3D`` that hides ``ieps`` from ``wakis.schematic``
    so 2D ``PolySlab`` geometries can be drawn directly without bounding-box fallback rectangles."""

    def __init__(self, solver, eps_clad: float, structures=()):
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
        self._assigned_structures = tuple(structures)


def _build_port_sources_and_monitors(circuit, include_flux_box: bool = True):
    """Build ``(sources, monitors)`` dicts for ``wakis.schematic`` from ``circuit``."""
    xmin, xmax = float(circuit.grid.xmin), float(circuit.grid.xmax)
    ymin, ymax = float(circuit.grid.ymin), float(circuit.grid.ymax)
    dx_bg = float(np.median(np.diff(circuit.grid.x)))
    dy_bg = float(np.median(np.diff(circuit.grid.y)))
    pulse_obj = circuit.sim.pulse
    f0 = float(pulse_obj.frequency)
    fwidth = float(pulse_obj.fwidth)
    freqs = np.asarray(getattr(circuit, "frequencies", [f0]), dtype=float)
    f_min_thz = float(np.min(freqs)) * 1e-12
    f_max_thz = float(np.max(freqs)) * 1e-12
    n_f = int(freqs.size)

    sources_s = []
    monitors_s = []
    for pt in circuit.ports:
        cx = float(pt.center[0])
        hy = min(0.5 * float(pt.size[1]), 0.85 * abs(ymax))
        if pt.name == "in":
            x_src = cx - 0.05e-6
            sources_s.append(
                {
                    "kind": "sheet",
                    "type_name": "ModeSource",
                    "label": "Port 'in' ModeSource (strip TE₀, +x)",
                    "short_word": "ModeSrc",
                    "desc": "Port 'in' ModeSource (strip TE₀, w = 400 nm, +x)",
                    "pos_str": f"x = {x_src * 1e6:.2f} µm, y ∈ [{-hy * 1e6:.2f}, +{hy * 1e6:.2f}] µm",
                    "freq_str": f"f₀ = {f0 * 1e-12:.2f} THz (λ ∈ [1.50, 1.60] µm, N = {n_f})",
                    "polarization": "TE₀ (Ey, Hz, +x)",
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
                    "short_word": "in",
                    "desc": "Port 'in' ModeMonitor (strip TE₀ reflection S₁₁)",
                    "pos_str": f"x = {cx * 1e6:.2f} µm, y ∈ [{-hy * 1e6:.2f}, +{hy * 1e6:.2f}] µm",
                    "freq_str": f"{f_min_thz:.1f}–{f_max_thz:.1f} THz ({n_f} DFT pts)",
                    "frequencies": freqs,
                    "component": "Ey, Hz (mode overlap)",
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
                    "short_word": "out",
                    "desc": "Port 'out' ModeMonitor (slot TE₀ transmission S₂₁, g = 100 nm)",
                    "pos_str": f"x = {cx * 1e6:.2f} µm, y ∈ [{-hy * 1e6:.2f}, +{hy * 1e6:.2f}] µm",
                    "freq_str": f"{f_min_thz:.1f}–{f_max_thz:.1f} THz ({n_f} DFT pts)",
                    "frequencies": freqs,
                    "component": "Ey, Hz (mode overlap)",
                    "h_bounds": (cx, cx),
                    "v_bounds": (-hy, +hy),
                }
            )
    if include_flux_box:
        x_lo = xmin + 2.0 * dx_bg
        x_hi = xmax - 2.0 * dx_bg
        y_lo = ymin + 2.0 * dy_bg
        y_hi = ymax - 2.0 * dy_bg
        monitors_s.append(
            {
                "kind": "box",
                "type_name": "FluxMonitor",
                "label": "2D Poynting FluxMonitor box",
                "short_word": "FluxBox",
                "desc": "2D Poynting FluxMonitor box (4-wall energy budget)",
                "pos_str": (
                    f"x ∈ [{x_lo * 1e6:.2f}, {x_hi * 1e6:.2f}] µm, "
                    f"y ∈ [{y_lo * 1e6:.2f}, {y_hi * 1e6:.2f}] µm"
                ),
                "freq_str": f"{f_min_thz:.1f}–{f_max_thz:.1f} THz ({n_f} DFT pts)",
                "frequencies": freqs,
                "component": "S · n (east + west + north + south)",
                "h_bounds": (x_lo, x_hi),
                "v_bounds": (y_lo, y_hi),
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
        marker="none",
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
        marker="none",
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
    circuit,
    primary_data: dict,
    out_path: str | Path,
):
    """
    Render the two-figure setup layout for Tutorial 5.4 via
    ``wakis.schematic.save_setup_figures``:
    - ``figures/00a_setup_geometry.png``: Labelled 2D setup schematic of Design 1
      (Symmetric MMI strip-to-slot mode converter, ``L = 6.0 um``) with domain
      extents, CPML boundaries, Si/SiO2 indices, input strip ``ModeSource`` with
      launched ``TE0`` profile, output slot ``ModeMonitor`` with discontinuous
      slot ``TE0`` profile, and 2D Poynting ``FluxMonitor`` box.
    - ``figures/00b_setup_mesh_and_excitation.png``: ``1x`` mesh view zoomed on
      the ``g = 100 nm`` oxide slot opening (``x = 1.38 um``) showing ``8.0``
      cells across the slot (``dy_min = 12.5 nm``), wave-packet extent, and
      broadband excitation pulse/spectrum.
    - ``figures/00_setup_table.md``: Markdown setup summary table.
    """
    out_path = Path(out_path)
    fig_dir = out_path.parent if out_path.suffix else out_path
    fig_dir.mkdir(parents=True, exist_ok=True)

    p_in, p_out = circuit.ports[0], circuit.ports[1]
    sources_s, monitors_s = _build_port_sources_and_monitors(circuit, include_flux_box=True)
    neff_s = float(circuit.n_eff_slab)
    eps_s = neff_s**2
    eps_clad = float(n_to_eps(circuit.n_clad))
    core_label = f"Si MMI core (ε_r = {eps_s:.2f})"
    clad_label = f"SiO₂ cladding & 100 nm slot (ε_r = {eps_clad:.2f})"
    mat_names = {
        "background": clad_label,
        eps_s: core_label,
    }
    struct_proxies = [
        _PolygonSchematicProxy(s, eps_s, core_label) for s in circuit.structures_3d
    ]
    solver_proxy = _SchematicSolverProxy(
        circuit.solver, eps_clad, structures=struct_proxies
    )

    variations = [
        {
            "name": "Design 2: Symmetric Tapered-Finger (Wang et al. 2009)",
            "param": "L = 8.0 µm, d_finger = 60 nm, g_slot = 100 nm",
            "note": "Three-PolySlab double-tip finger coupler; T(1550 nm) = 99.80%, N = 8 cells across 60 nm finger gap (Δy_min = 7.14 nm)",
        },
        {
            "name": "Design 3: Asymmetric Lateral Taper (Feng et al. 2007)",
            "param": "L = 20.0 µm, g_slot = 100 nm",
            "note": "Two-PolySlab continuous upper rail + tapered lower rail; T(1550 nm) = 99.66%, N = 8 cells across 100 nm slot",
        },
        {
            "name": "Design 1 MMI length sweep",
            "param": "L ∈ {1.6, 2.2, 3.0, 4.5, 6.0} µm (L_mmi = 1.38 µm)",
            "note": "Coupled-mode adiabatic saturation of slot-rail taper section L_taper = L - L_mmi",
        },
        {
            "name": "Kottke vs staircase convergence ladder",
            "param": "Under-resolved (10, 0), L1=(10, 4), L2=(14, 6), L3=(20, 8), R=(40, 16)",
            "note": "Compares Kottke subpixel smoothing against staircase snapping across 100 nm oxide slot",
        },
        {
            "name": "Graded non-uniform mesh upgrade (kottke_nonuniform)",
            "param": "367 × 111 × 1 = 40,737 cells (-38.1% vs uniform L3)",
            "note": "Δy = 12.5 nm (8 cells across 100 nm slot) with non-square Δx ≤ 34 nm along taper",
        },
    ]

    title_geom = "Tutorial 5.4 — Strip-to-Slot Mode Converter (Design 1: Symmetric MMI, L = 6.0 µm)"
    title_mesh = (
        "Tutorial 5.4 — Strip-to-Slot Mode Converter (Design 1: Symmetric MMI, L = 6.0 µm)"
        " — Mesh & Excitation"
    )
    saved_paths = schematic.save_setup_figures(
        solver_proxy,
        fig_dir,
        plane="xy",
        unit="µm",
        structures=struct_proxies,
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        wavelength=1550e-9,
        min_feature=100e-9,
        variations=variations,
        title_geometry=title_geom,
        title_mesh=title_mesh,
    )

    fig_a, ax0 = schematic.plot_schematic(
        solver_proxy,
        plane="xy",
        unit="µm",
        structures=struct_proxies,
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        wavelength=1550e-9,
        title=title_geom,
        width_in=8.4,
        height_in=5.2,
    )
    ax0.set_aspect("auto")
    if fig_a._suptitle is not None:
        fig_a._suptitle.set_y(1.07)

    # Draw top/bottom CPML bands when using non-uniform y spacing
    yg = np.asarray(circuit.grid.y, dtype=float) * 1e6
    xg = np.asarray(circuit.grid.x, dtype=float) * 1e6
    n_pml = int(getattr(circuit.solver, "n_pml", 10))
    if n_pml > 0 and yg.size > 2 * n_pml:
        x0, x1 = float(xg[0]), float(xg[-1])
        ylo0, ylo1 = float(yg[0]), float(yg[n_pml])
        yhi0, yhi1 = float(yg[-n_pml - 1]), float(yg[-1])
        for y_a, y_b in ((ylo0, ylo1), (yhi0, yhi1)):
            ax0.add_patch(
                Rectangle(
                    (x0, y_a),
                    x1 - x0,
                    y_b - y_a,
                    facecolor="#cbd5e1",
                    edgecolor="#475569",
                    hatch="///",
                    lw=0.8,
                    alpha=0.35,
                    zorder=2,
                )
            )

    for ln in ax0.lines:
        if ln.get_linestyle() in ("--", ":"):
            ln.set_marker("none")

    _draw_polyslab_polygons(
        ax0,
        circuit.structures_3d,
        facecolor="#2563eb",
        alpha=0.28,
        edgecolor="#0f172a",
        lw=1.35,
        label=(
            f"Si EIM core (h_Si=250 nm, n_Si=3.47)\n"
            f"n_eff,slab={neff_s:.4f} (ε_r={eps_s:.2f})"
        ),
        label_xy=(4.05, 0.72),
    )
    ax0.text(
        5.15,
        -1.10,
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
        xytext=(0.69, 1.15),
        fontsize=7.6,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.7", alpha=0.92),
    )
    ax0.annotate(
        "100 nm SiO₂ slot\nopens at x = 1.38 µm",
        xy=(1.50, 0.0),
        xytext=(2.75, -0.78),
        fontsize=7.5,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#d95f02", alpha=0.92),
    )

    tag_positions = {
        "1 · ModeSrc": (-2.45, 0.88, "left"),
        "2 · in": (-1.38, -0.88, "left"),
        "3 · out": (7.55, 0.96, "center"),
        "4 · FluxBox": (4.05, 1.34, "center"),
    }
    for txt in list(ax0.texts):
        s = txt.get_text().strip()
        if s in tag_positions:
            tx, ty, ha_val = tag_positions[s]
            txt.set_position((tx, ty))
            txt.set_ha(ha_val)

    fig_a.savefig(fig_dir / "00a_setup_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig_a)

    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        solver_proxy,
        plane="xy",
        unit="µm",
        structures=struct_proxies,
        material_names=mat_names,
        sources=sources_s,
        wavelength=1550e-9,
        min_feature=100e-9,
        zoom_bounds=(1.12e-6, 1.98e-6, -0.38e-6, 0.38e-6),
        title=title_mesh,
    )
    if len(axes_b) >= 1:
        ax_zoom = axes_b[0]
        dy_min_nm = float(np.min(np.diff(circuit.grid.y))) * 1e9
        dx_med_nm = float(np.median(np.diff(circuit.grid.x))) * 1e9
        ax_zoom.annotate(
            f"Slot g = 100 nm\n8.0 cells (Δy = {dy_min_nm:.1f} nm, Δx_med = {dx_med_nm:.1f} nm)",
            xy=(1.50, 0.0),
            xytext=(1.85, -1.05),
            color="#0f172a",
            fontsize=7.2,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=12,
            arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#d95f02", alpha=0.92),
        )
    fig_b.savefig(
        fig_dir / "00b_setup_mesh_and_excitation.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig_b)
    return saved_paths


def _clean_subpanel_schematic_annotations(ax, x_lo: float, x_hi: float, y_lo: float, y_hi: float):
    """Remove outside-of-axes dimension arrows and reposition CPML labels inside ``ax``."""
    from matplotlib.text import Annotation

    for child in list(ax.get_children()):
        if isinstance(child, Annotation) and child.get_text().strip() == "":
            xy = child.xy
            if xy[0] < x_lo or xy[0] > x_hi or xy[1] < y_lo or xy[1] > y_hi:
                child.remove()
    for txt in list(ax.texts):
        tx, ty = txt.get_position()
        s = txt.get_text().strip()
        if tx < x_lo or tx > x_hi or ty < y_lo or ty > y_hi or s.startswith("Δx ="):
            txt.remove()
    ax.spines["left"].set_position(("outward", 0))
    ax.spines["bottom"].set_position(("outward", 0))
    ax.set_title(ax.get_title(), pad=8, fontsize=9.5, fontweight="bold")
    x_mid = 0.5 * (x_lo + x_hi)
    ax.text(
        x_mid,
        y_hi - 0.14,
        "10-cell CPML boundaries (all 4 in-plane faces)",
        fontsize=6.8,
        fontweight="bold",
        color="#334155",
        ha="center",
        va="center",
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#94a3b8", alpha=0.88),
    )


def plot_taper_converters_schematic(
    circuit_sym,
    circuit_asym,
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
    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(16.4, 4.8), gridspec_kw={"wspace": 0.36}
    )

    neff_s = float(circuit_sym.n_eff_slab)
    eps_clad = float(n_to_eps(circuit_sym.n_clad))
    mat_names = {
        "background": f"SiO₂ clad/slot (n={circuit_sym.n_clad:.2f}, ε_r={eps_clad:.2f})",
    }

    # Panel (a): Design 2 setup schematic
    p2_in, p2_out = circuit_sym.ports[0], circuit_sym.ports[1]
    src2, mon2 = _build_port_sources_and_monitors(circuit_sym, include_flux_box=False)
    schematic.plot_schematic(
        _SchematicSolverProxy(circuit_sym.solver, eps_clad),
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
    if ax0.get_legend() is not None:
        ax0.get_legend().remove()
    for ln in ax0.lines:
        if ln.get_linestyle() in ("--", ":"):
            ln.set_marker("none")
    _clean_subpanel_schematic_annotations(
        ax0,
        float(circuit_sym.grid.xmin) * 1e6,
        float(circuit_sym.grid.xmax) * 1e6,
        float(circuit_sym.grid.ymin) * 1e6,
        float(circuit_sym.grid.ymax) * 1e6,
    )
    _draw_polyslab_polygons(
        ax0,
        circuit_sym.structures_3d,
        facecolor="#2563eb",
        alpha=0.28,
        edgecolor="#0f172a",
        lw=1.25,
        label=f"Si fingers & taper (d=60 nm, g=100 nm)\nn_eff,slab={neff_s:.2f}",
        label_xy=(4.2, 0.78),
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
    for txt in list(ax0.texts):
        s = txt.get_text().strip()
        if s == "1 · ModeSrc":
            txt.set_position((-2.35, 1.12))
            txt.set_ha("left")
        elif s == "2 · in":
            txt.set_position((-1.35, -1.12))
            txt.set_ha("left")
        elif s == "3 · out":
            txt.set_position((9.55, 1.12))
            txt.set_ha("center")

    # Panel (b): 1x mesh zoom on Design 2's d = 60 nm finger-to-strip gap
    xl_f, yl_f = (-0.15, 0.90), (0.12, 0.34)
    slv2 = circuit_sym.solver
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
        circuit_sym.structures_3d,
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
    dy_f_nm = float(np.min(np.diff(circuit_sym.grid.y))) * 1e9
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
    p3_in, p3_out = circuit_asym.ports[0], circuit_asym.ports[1]
    src3, mon3 = _build_port_sources_and_monitors(circuit_asym, include_flux_box=False)
    schematic.plot_schematic(
        _SchematicSolverProxy(circuit_asym.solver, eps_clad),
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
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
    for ln in ax2.lines:
        if ln.get_linestyle() in ("--", ":"):
            ln.set_marker("none")
    _clean_subpanel_schematic_annotations(
        ax2,
        float(circuit_asym.grid.xmin) * 1e6,
        float(circuit_asym.grid.xmax) * 1e6,
        float(circuit_asym.grid.ymin) * 1e6,
        float(circuit_asym.grid.ymax) * 1e6,
    )
    _draw_polyslab_polygons(
        ax2,
        circuit_asym.structures_3d,
        facecolor="#2563eb",
        alpha=0.28,
        edgecolor="#0f172a",
        lw=1.25,
        label=f"Si asymmetric taper (L=20 µm)\nn_eff,slab={neff_s:.2f}",
        label_xy=(10.2, 0.78),
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
    for txt in list(ax2.texts):
        s = txt.get_text().strip()
        if s == "1 · ModeSrc":
            txt.set_position((-2.35, 1.12))
            txt.set_ha("left")
        elif s == "2 · in":
            txt.set_position((-1.35, -1.12))
            txt.set_ha("left")
        elif s == "3 · out":
            txt.set_position((21.2, 1.12))
            txt.set_ha("center")

    figstyle.save(fig, out_path)
    plt.close(fig)


def plot_mesh_upgrade_comparison(
    circuit_uniform,
    circuit_nonuniform,
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
    iz = circuit_nonuniform.solver.Nz // 2
    ieps = np.asarray(circuit_nonuniform.solver.ieps[:, :, iz, "y"])
    eps_r = np.where(ieps > 0, 1.0 / (np.where(ieps > 0, ieps, 1.0) * epsilon_0), 1.0)
    xu_nu = circuit_nonuniform.grid.x * 1e6
    yu_nu = circuit_nonuniform.grid.y * 1e6
    ax0.pcolormesh(xu_nu, yu_nu, eps_r.T, cmap=figstyle.CMAP_EPS, shading="flat", alpha=0.88)

    xl_m, yl_m = (-0.50, 3.20), (-1.45, 1.45)
    xu_u = circuit_uniform.grid.x * 1e6
    yu_u = circuit_uniform.grid.y * 1e6
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
    for s in circuit_nonuniform.structures_3d:
        ax0.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                fill=False,
                edgecolor="black",
                lw=1.1,
            )
        )

    n_u = int(circuit_uniform.grid.Nx * circuit_uniform.grid.Ny)
    n_nu = int(circuit_nonuniform.grid.Nx * circuit_nonuniform.grid.Ny)
    t_u_s = float(meta_uniform["elapsed_s"]) if meta_uniform else 146.50
    t_nu_s = float(meta_nonuniform["elapsed_s"]) if meta_nonuniform else 0.0
    red_cells = 100.0 * (1.0 - n_nu / n_u)
    red_time = 100.0 * (1.0 - t_nu_s / t_u_s) if t_u_s > 0 else 0.0

    ax0.text(
        0.05,
        1.15,
        f"Upper (y > 0): kottke_L3 baseline\n"
        f"{circuit_uniform.grid.Nx}×{circuit_uniform.grid.Ny} = {n_u:,} cells ({t_u_s:.1f} s)",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#08519c", alpha=0.92),
    )
    ax0.text(
        0.05,
        -1.28,
        f"Lower (y < 0): kottke_nonuniform\n"
        f"{circuit_nonuniform.grid.Nx}×{circuit_nonuniform.grid.Ny} = {n_nu:,} cells "
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
        label=f"kottke_L3 ({circuit_uniform.grid.Ny} y-cells)",
    )
    ax1.step(
        yc_nu,
        dy_nu,
        where="mid",
        color="#d95f02",
        lw=2.0,
        label=f"kottke_nonuniform ({circuit_nonuniform.grid.Ny} y-cells)",
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


def overlay_converter_geometry(ax, circuit, outline_color: str = "white"):
    """Overlay material outlines/labels, ports, and CPML boundaries on a field map."""
    neff_s = float(circuit.n_eff_slab)
    eps_clad = float(n_to_eps(circuit.n_clad))
    mat_names = {
        "background": f"SiO₂ (n={circuit.n_clad:.2f})",
    }
    sources_s, monitors_s = _build_port_sources_and_monitors(circuit, include_flux_box=False)
    schematic.overlay_geometry(
        ax,
        _SchematicSolverProxy(circuit.solver, eps_clad),
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
        circuit.structures_3d,
        facecolor=None,
        edgecolor=outline_color,
        lw=1.15,
        label=f"Si EIM core (n_eff={neff_s:.2f}) | SiO₂ clad & slot (n={circuit.n_clad:.2f})",
        label_xy=(0.5 * (float(circuit.grid.xmin) + float(circuit.grid.xmax)) * 1e6, 0.85),
    )


def render_field_map_with_overlay(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    circuit,
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

    overlay_converter_geometry(ax, circuit, outline_color=outline_col)
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
    circuit,
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
    overlay_converter_geometry(ax, circuit, outline_color="black")
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

