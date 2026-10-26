# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Geometry, non-uniform graded meshing, and schematic/plotting helpers for
Tutorial 5.1 (`05-01_y_junction`).
"""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.constants import c as c_light, epsilon_0

from wakis import GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.sources import GaussianSource
from wakis.templates import meshing
from wakis.templates.photonics import (
    PHOTONIC_MATERIALS,
    PhotonicCircuit,
    PolySlab,
    Port,
    n_to_eps,
    sbend,
    slab_neff,
    spline_taper,
    straight,
)
from wakis.templates.simulation import Simulation

# Material refractive indices at 1550 nm (Malitson 1965; Li 1980)
N_SI = PHOTONIC_MATERIALS["Si"]  # 3.476
N_SIO2 = PHOTONIC_MATERIALS["SiO2"]  # 1.444
H_SI = 220e-9  # 220 nm SOI thickness

# Wavelength range: 1500 nm to 1600 nm (centered at 1550 nm)
WL_MIN = 1500e-9
WL_MAX = 1600e-9
WL_REF = 1550e-9
N_FREQS = 51

# Waveguide and Y-junction geometric parameters [m] (Zhang et al. 2013)
W_WG = 0.50e-6
L_IN = 1.40e-6
L_JUNC = 2.00e-6
L_JUNCTION = L_JUNC
L_SBEND = 2.00e-6
L_OUT = 1.20e-6
TIP_GAP = 200e-9  # 200 nm bifurcation tip gap (smallest geometric feature)
MIN_FEATURE = TIP_GAP
Y0_ARM = 0.35e-6  # (1.20 um / 2) - (0.50 um / 2) = 0.35 um -> 200 nm tip gap
DY_ARM = 0.55e-6  # S-bend offset to y_out = +-0.90 um
Y_OUT = Y0_ARM + DY_ARM  # 0.90 um

W_KNOTS_UM = np.array(
    [0.50, 0.50, 0.60, 0.70, 0.90, 1.26, 1.40, 1.40, 1.40, 1.40, 1.31, 1.20, 1.20]
)
W_KNOTS_M = W_KNOTS_UM * 1e-6

DOMAIN_2D = (-1.15e-6, 4.85e-6, -1.85e-6, 1.85e-6)
X_PORT_IN = -0.80e-6
X_PORT_OUT = 4.45e-6


def build_y_junction(n_taper: int = 160, n_sbend: int = 100):
    """Build the Zhang et al. (2013) SOI Y-junction as a single seamless `PolySlab` and 3 `Port`s.

    Parameters
    ----------
    n_taper : int, optional
        Number of samples per edge along the 2.0 um cubic-spline junction.
    n_sbend : int, optional
        Number of samples per edge along each 2.0 um cosine S-bend arm.

    Returns
    -------
    structures : list of PolySlab
        Single merged silicon `PolySlab` centered at `z = 0` with thickness `H_SI`.
    ports : list of Port
        `[Port('in'), Port('out1'), Port('out2')]`.
    """
    x_in = -L_IN
    x_sb_end = L_JUNC + L_SBEND
    x_out = x_sb_end + L_OUT
    hw = 0.5 * W_WG

    taper_v = spline_taper(L_JUNC, W_KNOTS_M, x0=0.0, y0=0.0, n=n_taper)
    taper_bot = taper_v[:n_taper]
    taper_top = taper_v[n_taper:]

    sb_bot_v = sbend(
        (L_JUNC, -Y0_ARM),
        dx=L_SBEND,
        dy=-DY_ARM,
        width=W_WG,
        profile="cosine",
        n=n_sbend,
    )
    sb_bot_outer = sb_bot_v[:n_sbend]
    sb_bot_inner = sb_bot_v[n_sbend:]

    sb_top_v = sbend(
        (L_JUNC, +Y0_ARM),
        dx=L_SBEND,
        dy=+DY_ARM,
        width=W_WG,
        profile="cosine",
        n=n_sbend,
    )
    sb_top_inner = sb_top_v[:n_sbend]
    sb_top_outer = sb_top_v[n_sbend:]

    verts = np.vstack(
        [
            [[x_in, -hw]],
            taper_bot,
            sb_bot_outer[1:],
            [[x_out, -Y_OUT - hw], [x_out, -Y_OUT + hw]],
            sb_bot_inner,
            sb_top_inner,
            [[x_out, +Y_OUT - hw], [x_out, +Y_OUT + hw]],
            sb_top_outer,
            taper_top[1:],
            [[x_in, +hw]],
        ]
    )
    slab = PolySlab(
        verts,
        z_min=-0.5 * H_SI,
        z_max=0.5 * H_SI,
        material=n_to_eps(N_SI),
    )
    ports = [
        Port(
            "in", center=(X_PORT_IN, 0.0, 0.0), size=(0.0, 1.8e-6, 0.8e-6), direction=+1
        ),
        Port(
            "out1",
            center=(X_PORT_OUT, +Y_OUT, 0.0),
            size=(0.0, 1.5e-6, 0.8e-6),
            direction=+1,
        ),
        Port(
            "out2",
            center=(X_PORT_OUT, -Y_OUT, 0.0),
            size=(0.0, 1.5e-6, 0.8e-6),
            direction=+1,
        ),
    ]
    return [slab], ports


class NonUniformYJunctionCircuit(PhotonicCircuit):
    """`PhotonicCircuit` with graded non-uniform, non-square Cartesian meshing for the Y-junction.

    Places fine transverse cells (`dy_tip`) across the 200 nm bifurcation tip gap
    (`y in [-0.20, +0.20] um`), intermediate core cells (`dy_core`) across the
    S-bend arms (`|y| in [0.20, 0.92] um`), and graded coarse cells (`dy_clad`)
    in the outer SiO2 cladding (`|y| > 0.92 um`). Along the propagation axis `x`,
    uses fine cells (`dx_junc`) across the 2.0 um PSO multimode junction and tip
    (`x in [-0.15, 2.40] um`) and non-square cells (`dx_lead > dy_tip`) along the
    smooth input/output leads and S-bends.
    """

    def __init__(
        self,
        *args,
        dy_tip: float | None = None,
        dy_core: float | None = None,
        dy_clad: float | None = None,
        dx_junc: float | None = None,
        dx_lead: float | None = None,
        grade_ratio: float = 1.25,
        **kwargs,
    ):
        cpw = float(kwargs.get("cells_per_wavelength", 28.0))
        scale = 28.0 / max(cpw, 8.0)
        self.dy_tip = float(min(16.67e-9 * scale, 45.0e-9) if dy_tip is None else dy_tip)
        self.dy_core = float(18.82e-9 * scale if dy_core is None else dy_core)
        self.dy_clad = float(52.0e-9 * scale if dy_clad is None else dy_clad)
        self.dx_junc = float(18.82e-9 * scale if dx_junc is None else dx_junc)
        self.dx_lead = float(24.5e-9 * scale if dx_lead is None else dx_lead)
        self.grade_ratio = float(grade_ratio)
        super().__init__(*args, **kwargs)

    def build(self):
        structs, domain = self._build_2d_structures_and_domain()
        xmin, xmax, ymin, ymax, zmin_2d, zmax_2d = domain
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
            mesh_overrides=self.mesh_overrides,
            use_gpu=self.use_gpu,
            averaging=self.averaging,
        )
        n_pml = self.sim.n_pml

        def _custom_axis(axis: int) -> np.ndarray:
            if axis == 2:
                return np.array([zmin_2d, zmax_2d], dtype=float)
            if axis == 0:
                nodes = meshing.graded_axis(
                    [xmin, -0.15e-6, 2.40e-6, xmax],
                    [self.dx_lead, self.dx_junc, self.dx_lead],
                    grading=self.grade_ratio,
                )
            else:
                nodes = meshing.graded_axis(
                    [ymin, -0.92e-6, -0.20e-6, 0.20e-6, 0.92e-6, ymax],
                    [self.dy_clad, self.dy_core, self.dy_tip, self.dy_core, self.dy_clad],
                    grading=self.grade_ratio,
                )
            lo_pml = nodes[0] - np.arange(n_pml, 0, -1) * (nodes[1] - nodes[0])
            hi_pml = nodes[-1] + np.arange(1, n_pml + 1) * (nodes[-1] - nodes[-2])
            return np.concatenate([lo_pml, nodes, hi_pml])

        self.sim._axis = _custom_axis
        self.solver = self.sim.build()
        self.solver.ieps[:, :, :, "z"] = 0.0
        self.grid = self.sim.grid
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        return self.solver


def measure_y_junction_mesh(grid: GridFIT3D, min_feature_m: float = TIP_GAP) -> dict:
    """Measure resolution metrics from a 2D `GridFIT3D` Y-junction mesh.

    Parameters
    ----------
    grid : GridFIT3D
        Constructed 2D FIT grid.
    min_feature_m : float, optional
        Smallest geometric feature size [m] (default 200 nm tip gap).

    Returns
    -------
    dict
        `min_feature_nm`, `dx_background_nm`, `dx_min_nm`, `dy_tip_nm`,
        `dy_clad_max_nm`, `cells_across_feature`, `aspect_ratio_max_core`,
        `grid_shape`, and `n_cells`.
    """
    dx = np.diff(grid.x)
    dy = np.diff(grid.y)
    xm = 0.5 * (grid.x[:-1] + grid.x[1:])
    ym = 0.5 * (grid.y[:-1] + grid.y[1:])

    in_core_x = (xm >= -1.0e-6) & (xm <= 4.6e-6)
    in_tip_y = (ym >= -0.5 * min_feature_m) & (ym <= +0.5 * min_feature_m)
    if not np.any(in_tip_y):
        in_tip_y = np.abs(ym) <= 0.15e-6

    dx_core_max = float(np.max(dx[in_core_x])) if np.any(in_core_x) else float(np.max(dx))
    dy_tip = float(np.max(dy[in_tip_y]))
    dx_min = float(min(np.min(dx), np.min(dy)))
    dy_clad_max = float(np.max(dy))
    cells_across = float(min_feature_m) / dy_tip
    aspect_core = dx_core_max / dy_tip
    shape = [int(grid.Nx), int(grid.Ny), int(grid.Nz)]
    return {
        "min_feature_nm": round(float(min_feature_m) * 1e9, 2),
        "dx_background_nm": round(dx_core_max * 1e9, 2),
        "dx_min_nm": round(dx_min * 1e9, 2),
        "dy_tip_nm": round(dy_tip * 1e9, 2),
        "dy_clad_max_nm": round(dy_clad_max * 1e9, 2),
        "cells_across_feature": round(cells_across, 2),
        "aspect_ratio_max_core": round(aspect_core, 2),
        "grid_shape": shape,
        "n_cells": int(np.prod(shape)),
    }


def draw_polyslab_regions(
    ax,
    structures: list[PolySlab],
    n_core: float = N_SI,
    n_clad: float = N_SIO2,
    h_si: float = H_SI,
    wl_ref: float = WL_REF,
    fill: bool = True,
    facecolor: str = "#93C5FD",
    edgecolor: str = "#1E293B",
    alpha: float = 0.78,
    lw: float = 1.25,
    label_Pos: tuple[float, float] | None = (0.95, 0.0),
) -> None:
    """Draw exact `PolySlab` waveguide polygon(s) and material label on `ax` (in µm)."""
    from matplotlib.patches import Polygon as MplPolygon
    import matplotlib.patheffects as pe

    neff_slab = slab_neff(n_core, n_clad, h_si, wl_ref, polarization="TE")
    for st in structures:
        verts_um = np.asarray(st.vertices, dtype=float) * 1e6
        patch = MplPolygon(
            verts_um,
            closed=True,
            facecolor=facecolor if fill else "none",
            edgecolor=edgecolor,
            lw=lw,
            alpha=alpha if fill else 0.92,
            zorder=5 if fill else 6,
        )
        ax.add_patch(patch)
        for hole in getattr(st, "holes", ()):
            h_um = np.asarray(hole, dtype=float) * 1e6
            ax.add_patch(
                MplPolygon(
                    h_um,
                    closed=True,
                    facecolor="#F8FAFC" if fill else "none",
                    edgecolor=edgecolor,
                    lw=lw,
                    alpha=0.95 if fill else 0.92,
                    zorder=5.5 if fill else 6,
                )
            )
    if label_Pos is not None:
        ax.text(
            label_Pos[0],
            label_Pos[1],
            f"Si core (n = {n_core:.3f}, 2D EIM n_eff = {neff_slab:.3f})",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color=figstyle.INK,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")],
            zorder=8,
        )


class _PolygonSchematicProxy:
    """Lightweight polygon structure proxy exposing `.vertices`, `.material`, and `.name`
    (without `.bounds`) so `wakis.schematic` classifies a 2D `PolySlab` as `kind='polygon'`.
    """

    def __init__(self, slab: PolySlab, material_eps: float, name: str):
        self.vertices = np.asarray(slab.vertices, dtype=float)
        self.material = float(material_eps)
        self.name = str(name)


class _SchematicSolverProxy:
    """Lightweight proxy around `SolverFIT3D` that hides `ieps` from
    `wakis.schematic.extract_setup_info` so explicit polygon `structures` are used.
    """

    def __init__(self, solver: SolverFIT3D, eps_clad: float):
        self._solver = solver
        self.grid = solver.grid
        self.Nx = solver.Nx
        self.Ny = solver.Ny
        self.Nz = solver.Nz
        self.dt = solver.dt
        self.n_pml = getattr(solver, "n_pml", 10)
        self.bc_low = getattr(solver, "bc_low", ["cpml", "cpml", "periodic"])
        self.bc_high = getattr(solver, "bc_high", ["cpml", "cpml", "periodic"])
        self.bg = [float(eps_clad), 1.0, 0.0]
        self._assigned_structures = ()


def build_schematic_components(
    grid: GridFIT3D,
    structures: list[PolySlab],
    ports: list[Port],
    n_core: float = N_SI,
    n_clad: float = N_SIO2,
    h_si: float = H_SI,
    wl_min: float = WL_MIN,
    wl_max: float = WL_MAX,
    wl_ref: float = WL_REF,
    n_freqs: int = N_FREQS,
    n_pml: int = 10,
    drive_port: str = "in",
    include_flux_box: bool = True,
):
    """Build lightweight solver, labeled structures, source dict, and monitor dicts for `wakis.schematic`.

    Returns `(solver_schem, labeled_structs, material_names, sources_list, monitors_list)`.
    """
    solver_raw = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        use_stl=False,
        n_pml=n_pml,
        verbose=0,
    )
    eps_clad = n_to_eps(n_clad)
    neff_slab = slab_neff(n_core, n_clad, h_si, wl_ref, polarization="TE")
    eps_slab = n_to_eps(neff_slab)
    solver_raw.bg = [eps_clad, 1.0, 0.0]
    for comp in ("x", "y", "z"):
        solver_raw.ieps[:, :, :, comp] = 1.0 / (eps_clad * epsilon_0)

    solver_schem = _SchematicSolverProxy(solver_raw, eps_clad=eps_clad)
    labeled_structs = [
        _PolygonSchematicProxy(
            st,
            material_eps=eps_slab,
            name=f"Si core (n = {n_core:.3f}, 2D EIM n_eff = {neff_slab:.3f})",
        )
        for st in structures
    ]
    material_names = {
        "background": f"SiO₂ cladding (n = {n_clad:.3f}, ε_r = {eps_clad:.2f})",
        "region_0": f"Si core (n = {n_core:.3f}, 2D EIM n_eff = {neff_slab:.3f}, ε_eff = {eps_slab:.2f})",
    }

    f_min = c_light / wl_max
    f_max = c_light / wl_min
    f0 = 0.5 * (f_min + f_max)
    fwidth = 1.25 * (f_max - f_min)
    pulse = GaussianSource(frequency=f0, fwidth=fwidth, cutoff=4.2)

    sources_list = []
    monitors_list = []

    for pt in ports:
        cx, cy = float(pt.center[0]), float(pt.center[1])
        sx, sy = float(pt.size[0]), float(pt.size[1])
        axis = pt.axis
        hb = (cx - 0.5 * sx, cx + 0.5 * sx)
        vb = (cy - 0.5 * sy, cy + 0.5 * sy)
        if pt.name == drive_port:
            d_vec = (float(pt.direction), 0.0, 0.0) if axis == 0 else (0.0, float(pt.direction), 0.0)
            x_src = cx - 0.14e-6 if axis == 0 else cx
            hb_src = (x_src, x_src) if axis == 0 else hb
            sources_list.append(
                {
                    "kind": "sheet",
                    "type_name": "ModeSource",
                    "short_word": "ModeSrc",
                    "label": f"Port '{pt.name}' ModeSource (TE₀, +x)",
                    "desc": f"Port '{pt.name}' ModeSource at x = {cx * 1e6:.2f} µm (TE₀, +x)",
                    "h_bounds": hb_src,
                    "v_bounds": vb,
                    "k_vec": d_vec,
                    "e_vec": (0.0, 1.0, 0.0) if axis == 0 else (1.0, 0.0, 0.0),
                    "h_vec": (0.0, 0.0, 1.0),
                    "pulse": {
                        "f0": f0,
                        "fwidth": fwidth,
                        "t_peak": float(4.2 / (np.pi * fwidth)),
                        "t_end": float(pulse.end_time),
                        "src_time": pulse,
                    },
                }
            )
        s_tag = "S₁₁" if pt.name == "in" else ("S₂₁" if pt.name == "out1" else "S₃₁")
        monitors_list.append(
            {
                "kind": "plane",
                "type_name": "ModeMonitor",
                "short_word": f"{pt.name} mon" if pt.name == "in" else pt.name,
                "label": f"Port '{pt.name}' ModeMonitor ({s_tag})",
                "desc": f"Port '{pt.name}' ModeMonitor at x = {cx * 1e6:.2f} µm ({s_tag})",
                "h_bounds": hb,
                "v_bounds": vb,
            }
        )

    if include_flux_box:
        n_p = n_pml + 1
        xw = float(grid.x[n_p + 2])
        xe = float(grid.x[-1 - n_p])
        ys = float(grid.y[n_p])
        yn = float(grid.y[-1 - n_p])
        monitors_list.append(
            {
                "kind": "box",
                "type_name": "FluxMonitor",
                "short_word": "FluxBox",
                "label": "2D Poynting FluxMonitor box",
                "desc": "Closed 2D Poynting FluxMonitor box (4 faces inside CPML)",
                "h_bounds": (xw, xe),
                "v_bounds": (ys, yn),
            }
        )

    return solver_schem, labeled_structs, material_names, sources_list, monitors_list


def draw_launched_mode_on_schematic(
    ax,
    port_x_um: float,
    port_y_um: np.ndarray,
    port_ey: np.ndarray,
    scale_x_um: float = 0.65,
    label: str = "Launched TE0 |Ey(y)|",
) -> None:
    """Overlay the launched transverse TE0 mode profile `|Ey(y)|` on the input port plane of `ax`."""
    env = np.abs(np.asarray(port_ey, dtype=complex))
    peak = max(float(np.max(env)), 1e-30)
    env_norm = env / peak
    y_u = np.asarray(port_y_um, dtype=float)
    x_curve = float(port_x_um) + scale_x_um * env_norm
    ax.fill_betweenx(
        y_u,
        float(port_x_um),
        x_curve,
        color="#D55E00",
        alpha=0.22,
        zorder=7,
    )
    ax.plot(
        x_curve,
        y_u,
        color="#D55E00",
        lw=1.6,
        ls="-",
        marker="none",
        zorder=8,
        label=label,
    )


def plot_setup_schematic_y_junction(
    grid: GridFIT3D,
    structures: list[PolySlab],
    ports: list[Port],
    port_y_in_um: np.ndarray,
    port_ey_in: np.ndarray,
    out_path: Path,
    n_pml: int = 10,
) -> dict:
    """Render `00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`, and
    `00_setup_table.md` for Tutorial 5.1 using `wakis.schematic.save_setup_figures`.
    """
    from matplotlib.patches import Polygon as MplPolygon, Rectangle

    out_path = Path(out_path)
    out_dir = out_path if (out_path.is_dir() or out_path.suffix == "") else out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    solver_s, structs_s, mat_names, srcs, mons = build_schematic_components(
        grid, structures, ports, n_pml=n_pml
    )
    variations = [
        {
            "name": "kottke_coarse..fine (tasks 0–5)",
            "param": "cpw ∈ {14, 20, 28}, averaging ∈ {kottke, staircase}",
            "note": "Uniform square-cell resolution ladder (14, 20, 28 cells/λ)",
        },
        {
            "name": "kottke_reference (task 6)",
            "param": "cpw = 56 (657 × 412 × 1 = 270,684 cells)",
            "note": "Independent 2× fine Kottke reference",
        },
        {
            "name": "kottke_nonuniform (task 7)",
            "param": "Graded non-square Cartesian mesh (298 × 162 × 1 = 48,276 cells)",
            "note": "12 cells across 200 nm tip gap (Δy_tip = 16.7 nm), Δx_lead = 24.5 nm, Δy_clad ≤ 50.4 nm (−34% cells)",
        },
    ]
    title_geom = "Tutorial 5.1 Setup: 2D EIM SOI Y-Junction Power Splitter"
    title_mesh = "Discretization (Graded Non-Square Mesh & 200 nm Tip-Gap Zoom) and Excitation"

    res = schematic.save_setup_figures(
        solver_s,
        out_dir,
        prefix="00",
        plane="xy",
        structures=structs_s,
        material_names=mat_names,
        sources=srcs,
        monitors=mons,
        unit="µm",
        wavelength=WL_REF,
        min_feature=TIP_GAP,
        title_geometry=title_geom,
        title_mesh=title_mesh,
        variations=variations,
    )

    # Enhance 00a_setup_geometry.png with top/bottom CPML shading, launched mode profile, and clean port tags
    fig_a, ax_a = schematic.plot_schematic(
        solver_s,
        plane="xy",
        structures=structs_s,
        material_names=mat_names,
        sources=srcs,
        monitors=mons,
        unit="µm",
        wavelength=WL_REF,
        title=title_geom,
        width_in=7.6,
        height_in=4.6,
    )
    x0_um, x1_um = float(grid.xmin * 1e6), float(grid.xmax * 1e6)
    y0_um, y1_um = float(grid.ymin * 1e6), float(grid.ymax * 1e6)
    th_y_low = float((grid.y[n_pml] - grid.y[0]) * 1e6)
    th_y_high = float((grid.y[-2] - grid.y[-2 - n_pml]) * 1e6)
    for yb_start, th_y in ((y0_um, th_y_low), (y1_um - th_y_high, th_y_high)):
        ax_a.add_patch(
            Rectangle(
                (x0_um, yb_start),
                x1_um - x0_um,
                th_y,
                facecolor="#CBD5E1",
                edgecolor="#64748B",
                hatch="///",
                alpha=0.36,
                lw=0.8,
                zorder=2,
            )
        )
    for ln in ax_a.lines:
        if ln.get_linestyle() in ("--", ":"):
            ln.set_marker("none")

    x_src_um = float(srcs[0]["h_bounds"][0] * 1e6)
    draw_launched_mode_on_schematic(
        ax_a,
        port_x_um=x_src_um,
        port_y_um=port_y_in_um,
        port_ey=port_ey_in,
        scale_x_um=0.42,
    )

    tag_Overrides = {
        "1 ·": (-0.75, 1.15),
        "2 ·": (-0.55, -1.15),
        "3 ·": (3.85, 0.90),
        "4 ·": (3.85, -0.90),
        "5 ·": (1.85, 1.48),
        "Si core": (1.05, 0.0),
        "SiO₂ cladding": (0.75, -1.38),
    }
    for txt in ax_a.texts:
        s = txt.get_text()
        for prefix_key, pos in tag_Overrides.items():
            if s.startswith(prefix_key):
                txt.set_position(pos)
                break

    figstyle.save(fig_a, res["geometry_figure"], close=True, auto_legend_below=False)

    # Enhance 00b_setup_mesh_and_excitation.png with the Y-junction polygon on ax_mesh and 1x tip-gap zoom inset
    zoom_bounds = (1.72e-6, 2.28e-6, -0.34e-6, 0.34e-6)
    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        solver_s,
        plane="xy",
        structures=structs_s,
        material_names=mat_names,
        sources=srcs,
        monitors=mons,
        unit="µm",
        mesh_every=6,
        zoom_bounds=zoom_bounds,
        wavelength=WL_REF,
        min_feature=TIP_GAP,
        title=title_mesh,
        width_in=7.6,
        height_in=5.6,
    )
    ax_mesh = axes_b[0]
    verts_um = np.asarray(structures[0].vertices, dtype=float) * 1e6
    ax_mesh.add_patch(
        MplPolygon(
            verts_um,
            closed=True,
            facecolor="#DCEAF7",
            edgecolor="#1E293B",
            lw=1.1,
            alpha=0.80,
            zorder=2,
        )
    )
    for child_ax in fig_b.axes:
        if getattr(child_ax, "_is_inset", False):
            child_ax.add_patch(
                MplPolygon(
                    verts_um,
                    closed=True,
                    facecolor="#DCEAF7",
                    edgecolor="#1E293B",
                    lw=1.0,
                    alpha=0.80,
                    zorder=1.5,
                )
            )
    figstyle.save(fig_b, res["mesh_figure"], close=True, auto_legend_below=False)
    return res


def render_field_map_with_overlay(
    x_m: np.ndarray,
    y_m: np.ndarray,
    field_2d: np.ndarray,
    grid: GridFIT3D,
    structures: list[PolySlab],
    ports: list[Port],
    kind: str,
    title: str,
    out_path: Path,
    n_pml: int = 10,
    db_floor: float = -35.0,
) -> Path:
    """Render a 2D field map (`intensity_log` or `real`) with `schematic.overlay_geometry` and `figstyle`."""
    solver_s, structs_s, mat_names, srcs, mons = build_schematic_components(
        grid, structures, ports, n_pml=n_pml, include_flux_box=False
    )
    x_um = np.asarray(x_m, dtype=float) * 1e6
    y_um = np.asarray(y_m, dtype=float) * 1e6
    arr = np.asarray(field_2d)

    with figstyle.apply_style():
        fig, ax = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.8,
            ncols=1,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.92, bottom=0.16, top=0.88)
        if kind == "intensity_log":
            mag = np.abs(arr)
            f_max = max(float(np.max(mag)), 1e-30)
            data = 10.0 * np.log10(np.maximum(mag / f_max, 10.0 ** (db_floor / 10.0)))
            im = ax.pcolormesh(
                x_um,
                y_um,
                data.T,
                cmap=figstyle.CMAP_INTENSITY,
                vmin=db_floor,
                vmax=0.0,
                shading="auto",
            )
            cbar_label = r"$10\log_{10}(|\mathbf{E}|^2 / |\mathbf{E}|_{\max}^2)$ [dB]"
        else:
            re_f = np.real(arr)
            vmin, vmax = figstyle.symmetric_limits(re_f, percentile=99.5)
            im = ax.pcolormesh(
                x_um,
                y_um,
                re_f.T / vmax,
                cmap=figstyle.CMAP_DIVERGING,
                vmin=-1.0,
                vmax=1.0,
                shading="auto",
            )
            cbar_label = r"Normalized $\mathrm{Re}(E_y)$"

        schematic.overlay_geometry(
            ax,
            solver_s,
            plane="xy",
            unit="µm",
            structures=[],
            material_names=mat_names,
            sources=srcs,
            monitors=mons,
            show_boundaries=True,
            show_materials=True,
            show_sources=True,
            show_monitors=True,
        )
        draw_polyslab_regions(
            ax,
            structures,
            fill=False,
            edgecolor="white" if kind == "intensity_log" else "#1E293B",
            lw=1.15,
            label_Pos=(0.95, 0.0),
        )
        ax.set(
            aspect="equal",
            xlim=(float(x_um.min()), float(x_um.max())),
            ylim=(float(y_um.min()), float(y_um.max())),
            xlabel="Position x [µm]",
            ylabel="Position y [µm]",
            title=title,
        )
        fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02, label=cbar_label)
        figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return out_path


def render_movie_with_overlay(
    x_m: np.ndarray,
    y_m: np.ndarray,
    frames: np.ndarray,
    times_s: np.ndarray,
    grid: GridFIT3D,
    structures: list[PolySlab],
    ports: list[Port],
    out_path: Path,
    n_pml: int = 10,
    fps: int = 12,
    stride_frames: int = 2,
) -> Path:
    """Render animated pulse-propagation GIF (< 5 MB) with `schematic.overlay_geometry` on every frame."""
    frames_arr = np.asarray(frames, dtype=np.float32)
    if frames_arr.size == 0:
        return out_path
    solver_s, structs_s, mat_names, srcs, mons = build_schematic_components(
        grid, structures, ports, n_pml=n_pml, include_flux_box=False
    )
    x_um = np.asarray(x_m, dtype=float) * 1e6
    y_um = np.asarray(y_m, dtype=float) * 1e6
    times_fs = np.asarray(times_s, dtype=float) * 1e15
    vmax = max(float(np.percentile(np.abs(frames_arr), 99.5)), 0.15)

    idxs = list(range(0, len(frames_arr), max(1, int(stride_frames))))
    pil_frames = []
    with figstyle.apply_style():
        for k in idxs:
            fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=110)
            fig.subplots_adjust(left=0.10, right=0.88, bottom=0.17, top=0.88)
            im = ax.pcolormesh(
                x_um,
                y_um,
                frames_arr[k].T,
                cmap=figstyle.CMAP_DIVERGING,
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
            )
            schematic.overlay_geometry(
                ax,
                solver_s,
                plane="xy",
                unit="µm",
                structures=structs_s,
                material_names=mat_names,
                sources=srcs,
                monitors=mons,
                show_boundaries=True,
                show_materials=True,
                show_sources=True,
                show_monitors=True,
            )
            draw_polyslab_regions(
                ax,
                structures,
                fill=False,
                edgecolor="#1E293B",
                lw=1.05,
                label_Pos=(0.95, 0.0),
            )
            ax.set(
                aspect="equal",
                xlim=(float(x_um.min()), float(x_um.max())),
                ylim=(float(y_um.min()), float(y_um.max())),
                xlabel="Position x [µm]",
                ylabel="Position y [µm]",
                title=f"Y-junction pulse propagation Re(Ey) (t = {times_fs[k]:.1f} fs)",
            )
            fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02, label=r"$E_y / E_{\max}$")
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
            optimize=True,
        )
    return out_path


def plot_mesh_upgrade_comparison(
    grid_uniform: GridFIT3D,
    grid_nonuniform: GridFIT3D,
    structures: list[PolySlab],
    ports: list[Port],
    data_uniform: dict,
    data_nonuniform: dict,
    data_reference: dict | None,
    meta_uniform: dict | None,
    meta_nonuniform: dict | None,
    out_path: Path,
    n_pml: int = 10,
) -> Path:
    """Render `04b_mesh_upgrade_comparison.png` comparing baseline uniform and upgraded non-square graded meshes."""
    solver_u, structs_u, mat_u, _, _ = build_schematic_components(
        grid_uniform, structures, ports, n_pml=n_pml, include_flux_box=False
    )
    solver_nu, structs_nu, mat_nu, _, _ = build_schematic_components(
        grid_nonuniform, structures, ports, n_pml=n_pml, include_flux_box=False
    )
    m_u = measure_y_junction_mesh(grid_uniform, min_feature_m=TIP_GAP)
    m_nu = measure_y_junction_mesh(grid_nonuniform, min_feature_m=TIP_GAP)

    t_u_s = float(meta_uniform.get("elapsed_s", 135.4)) if meta_uniform else 135.4
    t_nu_s = float(meta_nonuniform.get("elapsed_s", 74.7)) if meta_nonuniform else 74.7
    red_pct = 100.0 * (1.0 - m_nu["n_cells"] / max(m_u["n_cells"], 1))

    zoom_box = ((1.35, 3.45), (-1.18, 1.18))

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(
            width_in=11.0,
            height_in=4.2,
            ncols=3,
            nrows=1,
            gridspec_kw={"width_ratios": [1.0, 1.0, 1.18]},
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.29, top=0.83, wspace=0.32)
        ax0, ax1, ax2 = axes

        for ax, slv, sts, mats in (
            (ax0, solver_u, structs_u, mat_u),
            (ax1, solver_nu, structs_nu, mat_nu),
        ):
            schematic.draw_mesh(
                ax,
                slv,
                plane="xy",
                unit="µm",
                every=1,
                color=figstyle.INK,
                alpha=0.45,
                linewidth=0.45,
                show_cpw=False,
            )
            draw_polyslab_regions(
                ax,
                structures,
                fill=True,
                alpha=0.45,
                label_Pos=None,
            )
            ax.set(
                aspect="equal",
                xlim=zoom_box[0],
                ylim=zoom_box[1],
                xlabel="x [µm]",
            )

        ax0.set_ylabel("y [µm]")
        ax0.set_title(
            f"(a) Uniform baseline\n"
            f"{m_u['n_cells']:,} cells ({t_u_s:.0f} s)",
        )
        ax0.text(
            0.03,
            0.96,
            (
                f"Square cells:\n"
                f"Δx = Δy = {m_u['dy_tip_nm']:.1f} nm\n"
                f"Tip: {m_u['cells_across_feature']:.1f} cells"
            ),
            transform=ax0.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#CBD5E1", alpha=0.92),
            zorder=15,
        )

        ax1.set_title(
            f"(b) Graded non-square\n"
            f"{m_nu['n_cells']:,} cells (−{red_pct:.0f}%, {t_nu_s:.0f} s)",
        )
        ax1.text(
            0.03,
            0.96,
            (
                f"Δy_tip = {m_nu['dy_tip_nm']:.1f} nm ({m_nu['cells_across_feature']:.0f} cells)\n"
                f"Δx_lead = {m_nu['dx_background_nm']:.1f} nm\n"
                f"Δy_clad ≤ {m_nu['dy_clad_max_nm']:.0f} nm"
            ),
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#CBD5E1", alpha=0.92),
            zorder=15,
        )

        def _extract_el(d):
            w = d["wavelengths"] * 1e9
            idx = np.argsort(w)
            ws = w[idx]
            m = (ws >= 1505.0) & (ws <= 1595.0)
            t_tot = np.abs(d["S21"][idx][m]) ** 2 + np.abs(d["S31"][idx][m]) ** 2
            el_db = -10.0 * np.log10(np.maximum(t_tot, 1e-12))
            i0 = int(np.argmin(np.abs(d["wavelengths"] - WL_REF)))
            t_1550 = float(np.abs(d["S21"][i0]) ** 2 + np.abs(d["S31"][i0]) ** 2)
            return ws[m], el_db, t_1550

        w_u, el_u, t_u = _extract_el(data_uniform)
        w_nu, el_nu, t_nu = _extract_el(data_nonuniform)

        if data_reference is not None:
            w_ref, el_ref, t_ref = _extract_el(data_reference)
            ax2.plot(
                w_ref,
                el_ref,
                color=figstyle.INK,
                lw=1.8,
                ls=":",
                marker="none",
                label="56 cpw ref",
            )
        else:
            t_ref = t_u

        ax2.plot(
            w_u,
            el_u,
            color="#0072B2",
            lw=1.8,
            ls="--",
            marker="s",
            markevery=8,
            ms=4.5,
            label=f"Uniform (|ΔT|={abs(t_u - t_ref):.1e})",
        )
        ax2.plot(
            w_nu,
            el_nu,
            color="#D55E00",
            lw=1.8,
            ls="-",
            marker="^",
            markevery=8,
            ms=4.8,
            label=f"Graded (|ΔT|={abs(t_nu - t_ref):.1e})",
        )
        ax2.set(
            xlabel="Wavelength λ [nm]",
            ylabel="Excess loss [dB]",
            ylim=(0.30, 0.48),
            title="(c) Excess loss accuracy",
        )
        ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=1,
            frameon=True,
            fontsize=8.2,
        )
        figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return out_path

