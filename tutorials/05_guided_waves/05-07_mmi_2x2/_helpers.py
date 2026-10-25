# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Resolution-ladder, mesh-override, and plotting helpers for Tutorial 5.7
(2x2 MMI power splitter in 2D EIM).
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from scipy.constants import c as c_light, epsilon_0

from wakis import figstyle, schematic
from wakis.gridFIT3D import GridFIT3D
from wakis.solverFIT3D import SolverFIT3D
from wakis.templates import Simulation, meshing
from wakis.templates.photonics import Box, PICSimulation, PolySlab, n_to_eps, slab_neff

# Resolution ladder: level -> (cells_per_wavelength, cells_across_min_feature)
RESOLUTION_LADDER = {
    "L1": (10, 4),
    "L2": (14, 6),
    "L3": (20, 8),
    "R": (40, 16),
}

THROAT_OVERRIDE_X = 0.60e-6


def build_gap_mesh_overrides(
    l_mmi=3.00e-6,
    n_cells_feature=4,
    min_feature=0.20e-6,
    throat_x=THROAT_OVERRIDE_X,
):
    """
    Build two mesh-override boxes over the 0.20 um S-bend throat gaps at the
    left and right MMI faces (``x = +-l_mmi/2``, ``|y| <= min_feature/2``).

    Parameters
    ----------
    l_mmi : float, optional
        Length of the central MMI box [m].
    n_cells_feature : int or float, optional
        Target minimum number of cells ``N`` across ``min_feature``.
    min_feature : float, optional
        Smallest geometric feature size [m] (default: 0.20 um inner throat gap).
    throat_x : float, optional
        Axial extent [m] of each override box into the S-bend throat outside
        each MMI face.

    Returns
    -------
    overrides : list of tuple
        Two ``(lo, hi, (hx, hy, hz))`` mesh-override tuples for ``PICSimulation``.
        ``hz = 1.0`` leaves the single periodic 2D EIM z-cell unrefined.
    """
    if n_cells_feature is None or n_cells_feature <= 0:
        return []
    h_max = float(min_feature) / float(n_cells_feature)
    xh = 0.5 * float(l_mmi)
    hy = 0.5 * float(min_feature)
    tx = float(throat_x)
    return [
        ((-xh - tx, -hy, -np.inf), (-xh, +hy, np.inf), (h_max, h_max, 1.0)),
        ((+xh, -hy, -np.inf), (+xh + tx, +hy, np.inf), (h_max, h_max, 1.0)),
    ]


def measure_grid_resolution(
    grid,
    n_pml=10,
    l_mmi=3.00e-6,
    min_feature=0.20e-6,
    throat_x=THROAT_OVERRIDE_X,
    w_mmi=1.00e-6,
):
    """
    Compute actual resolution metrics from a built ``GridFIT3D`` (or object
    with ``x, y, Nx, Ny, Nz``).

    Parameters
    ----------
    grid : GridFIT3D
        Mesh object exposing node arrays ``x``, ``y`` [m] and cell counts
        ``Nx``, ``Ny``, ``Nz``.
    n_pml : int, optional
        Number of CPML boundary cells per side.
    l_mmi : float, optional
        MMI box length [m].
    min_feature : float, optional
        Smallest geometric feature width [m] (0.20 um throat gap).
    throat_x : float, optional
        Axial throat box width [m].
    w_mmi : float, optional
        MMI box width [m].

    Returns
    -------
    metrics : dict
        Dictionary containing ``min_feature_nm``, ``dx_background_nm``,
        ``dx_min_nm``, ``dx_max_nm``, ``dy_min_nm``, ``dy_max_nm``,
        ``sidewall_err_nm``, ``cells_across_feature``, ``grid_shape``, and
        ``n_cells``.
    """
    x_arr = np.asarray(grid.x, dtype=float)
    y_arr = np.asarray(grid.y, dtype=float)
    # Strip trailing FIT ghost cell if present (len(x) == Nx + 1 and last cell matches interior)
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    n_p = int(n_pml or 0)
    dx_int = dx[n_p : len(dx) - n_p] if len(dx) > 2 * n_p else dx
    dy_int = dy[n_p : len(dy) - n_p] if len(dy) > 2 * n_p else dy

    dx_bg = float(max(np.max(dx_int), np.max(dy_int)))
    dx_min = float(min(np.min(dx_int), np.min(dy_int)))

    hy = 0.5 * float(min_feature)
    y_nodes = y_arr
    in_gap_y = (y_nodes[1:] > -hy + 1e-15) & (y_nodes[:-1] < hy - 1e-15)
    max_dy_gap = float(np.max(dy[in_gap_y]))

    xh = 0.5 * float(l_mmi)
    tx = float(throat_x)
    x_nodes = x_arr
    in_gap_x = (
        (x_nodes[1:] > -xh - tx + 1e-15) & (x_nodes[:-1] < -xh - 1e-15)
    ) | ((x_nodes[1:] > xh + 1e-15) & (x_nodes[:-1] < xh + tx - 1e-15))
    max_dx_gap = float(np.max(dx[in_gap_x])) if np.any(in_gap_x) else max_dy_gap

    max_cell_across_feature = max(max_dy_gap, max_dx_gap)
    cells_across = float(min_feature) / max_cell_across_feature

    y_wall = 0.5 * float(w_mmi)
    sidewall_err = float(np.min(np.abs(y_nodes - y_wall)))

    nx, ny, nz = int(grid.Nx), int(grid.Ny), int(grid.Nz)
    return {
        "min_feature_nm": round(float(min_feature) * 1e9, 3),
        "dx_background_nm": round(dx_bg * 1e9, 3),
        "dx_min_nm": round(dx_min * 1e9, 3),
        "dx_max_nm": round(float(np.max(dx_int)) * 1e9, 3),
        "dy_min_nm": round(float(np.min(dy_int)) * 1e9, 3),
        "dy_max_nm": round(float(np.max(dy_int)) * 1e9, 3),
        "sidewall_err_nm": round(sidewall_err * 1e9, 3),
        "cells_across_feature": round(cells_across, 3),
        "grid_shape": [nx, ny, nz],
        "n_cells": int(nx * ny * nz),
    }


def build_mmi_2x2_graded_axes(
    domain,
    l_mmi: float = 3.00e-6,
    w_mmi: float = 1.00e-6,
    w_wg: float = 0.40e-6,
    min_feature: float = 0.20e-6,
    y_port: float = 1.30e-6,
    throat_x: float = THROAT_OVERRIDE_X,
    n_pml: int = 10,
    cpw: float = 20.0,
    n_cells_feature: int = 8,
    dx_lead: float | None = None,
    dy_sbend: float | None = None,
    dy_clad: float | None = None,
    grade_ratio: float = 1.22,
):
    """
    Construct mirror-symmetric graded non-uniform, non-square Cartesian node
    arrays ``(x, y, z)`` for the 2x2 SOI MMI power splitter:

    - Exact transverse grid nodes at ``y = 0``, ``y = +-0.5 * min_feature``
      (``+-0.10 um`` throat gap edges), ``y = +-0.5 * w_mmi`` (``+-0.50 um``
      MMI sidewalls, eliminating uniform-grid sidewall rounding error), and
      ``y = +-(y_port + 0.5 * w_wg)`` (``+-1.50 um`` outer port waveguide edges),
    - Exact ``n_cells_feature`` cells across the ``200 nm`` throat gap
      (``dy_gap = dx_throat = min_feature / n_cells_feature``),
    - Smoothly graded coarse transverse spacing ``dy_clad`` in the outer
      ``SiO2`` cladding ``|y| > 1.50 um``,
    - Exact axial grid nodes at ``x = +-0.5 * l_mmi`` (MMI faces) and
      ``x = +-(0.5 * l_mmi + throat_x)`` with fine spacing across the MMI cavity
      and throat gap, graded to stretched non-square axial cells
      ``dx_lead > dy_gap`` along the smooth cosine S-bends and straight leads.
    """
    if len(domain) == 4:
        xmin, xmax, ymin, ymax = domain
        zmin_2d, zmax_2d = -0.5e-6, 0.5e-6
    else:
        xmin, xmax, ymin, ymax, zmin_2d, zmax_2d = domain

    scale = 20.0 / max(float(cpw), 5.0)
    h_gap = float(min_feature) / max(float(n_cells_feature), 1.0)
    xh = 0.5 * float(l_mmi)
    x_thr = xh + float(throat_x)
    y_gap = 0.5 * float(min_feature)
    y_wall = 0.5 * float(w_mmi)
    y_sb = float(y_port) + 0.5 * float(w_wg)

    dx_ld = float(38.0e-9 * scale if dx_lead is None else dx_lead)
    dy_sb = float(26.0e-9 * scale if dy_sbend is None else dy_sbend)
    dy_cl = float(70.0e-9 * scale if dy_clad is None else dy_clad)

    # Positive x half-axis: exact uniform steps in [0, xh] and [xh, x_thr], graded in [x_thr, xmax]
    nx_mmi_half = max(1, int(round(xh / h_gap)))
    nx_thr = max(1, int(round(float(throat_x) / h_gap)))
    x_seg0 = np.linspace(0.0, xh, nx_mmi_half + 1)
    x_seg1 = np.linspace(xh, x_thr, nx_thr + 1)[1:]
    x_seg2 = meshing.graded_axis(
        [x_thr, x_thr + 4.0 * h_gap, xmax],
        [h_gap, dx_ld],
        grading=grade_ratio,
    )[1:]
    x_pos = np.concatenate([x_seg0, x_seg1, x_seg2])
    x_int = np.concatenate([-x_pos[:0:-1], x_pos])

    # Positive y half-axis: exact uniform steps in [0, y_gap] and [y_gap, y_wall],
    # graded in [y_wall, y_sb] and [y_sb, ymax]
    ny_gap_half = max(1, int(round(y_gap / h_gap)))
    ny_mmi_side = max(1, int(round((y_wall - y_gap) / h_gap)))
    y_seg0 = np.linspace(0.0, y_gap, ny_gap_half + 1)
    y_seg1 = np.linspace(y_gap, y_wall, ny_mmi_side + 1)[1:]
    y_seg2 = meshing.graded_axis(
        [y_wall, y_sb, ymax],
        [dy_sb, dy_cl],
        grading=grade_ratio,
    )[1:]
    y_pos = np.concatenate([y_seg0, y_seg1, y_seg2])
    y_int = np.concatenate([-y_pos[:0:-1], y_pos])

    n_pml = int(n_pml)

    def _pad_pml(nodes: np.ndarray) -> np.ndarray:
        lo_pml = nodes[0] - np.arange(n_pml, 0, -1) * (nodes[1] - nodes[0])
        hi_pml = nodes[-1] + np.arange(1, n_pml + 1) * (nodes[-1] - nodes[-2])
        return np.concatenate([lo_pml, nodes, hi_pml])

    return (
        _pad_pml(x_int),
        _pad_pml(y_int),
        np.array([zmin_2d, zmax_2d], dtype=float),
    )


class NonUniformMMI2x2PIC(PICSimulation):
    """
    ``PICSimulation`` subclass that builds a mirror-symmetric graded non-uniform,
    non-square Cartesian mesh for the 2x2 MMI coupler while preserving exact
    2D EIM Kottke subpixel smoothing and port eigenmode extraction.
    """

    def __init__(
        self,
        *args,
        l_mmi: float = 3.00e-6,
        w_mmi: float = 1.00e-6,
        w_wg: float = 0.40e-6,
        min_feature: float = 0.20e-6,
        y_port: float = 1.30e-6,
        throat_x: float = THROAT_OVERRIDE_X,
        n_cells_feature: int = 8,
        grade_ratio: float = 1.22,
        **kwargs,
    ):
        self.l_mmi = float(l_mmi)
        self.w_mmi = float(w_mmi)
        self.w_wg = float(w_wg)
        self.min_feature = float(min_feature)
        self.y_port = float(y_port)
        self.throat_x = float(throat_x)
        self.n_cells_feature = int(n_cells_feature)
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
        n_pml = int(self.n_pml or 10)
        self.sim.n_pml = n_pml
        axes_xyz = build_mmi_2x2_graded_axes(
            domain,
            l_mmi=self.l_mmi,
            w_mmi=self.w_mmi,
            w_wg=self.w_wg,
            min_feature=self.min_feature,
            y_port=self.y_port,
            throat_x=self.throat_x,
            n_pml=n_pml,
            cpw=self.cells_per_wavelength,
            n_cells_feature=self.n_cells_feature,
            grade_ratio=self.grade_ratio,
        )
        self.sim._axis = lambda axis: axes_xyz[int(axis)]
        self.solver = self.sim.build()
        self.solver.ieps[:, :, :, "z"] = 0.0
        self.grid = self.sim.grid
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        return self.solver


def build_task_grid_only(
    task_cfg,
    build_device_fn,
    wl_min=1500e-9,
    wl_max=1600e-9,
    wl_ref=1550e-9,
    n_core=3.48,
    n_clad=1.44,
    h_si=220e-9,
):
    """
    Build only the ``GridFIT3D`` mesh for ``task_cfg`` without allocating
    ``SolverFIT3D`` matrices or evaluating subpixel fills on all cells.

    Used by fast unit tests and preflight estimators.
    """
    l_mmi = float(task_cfg.get("l_mmi", 3.00e-6))
    min_feature = float(task_cfg.get("min_feature", 0.20e-6))
    n_cells_feature = task_cfg.get("n_cells_feature", 4)
    cpw = float(task_cfg["cpw"])
    n_pml = int(task_cfg.get("n_pml", 10))

    structures_3d, _, domain_2d = build_device_fn(l_mmi=l_mmi)
    neff_s = slab_neff(n_core, n_clad, h_si, wl_ref, polarization="TE")
    f_min, f_max = c_light / float(wl_max), c_light / float(wl_min)
    h_core = meshing.cell_size(f_max, neff_s**2, cpw)
    xmin, xmax, ymin, ymax = domain_2d[:4]
    domain = (xmin, xmax, ymin, ymax, -0.5 * h_core, +0.5 * h_core)

    if task_cfg.get("nonuniform", False):
        x, y, z = build_mmi_2x2_graded_axes(
            domain,
            l_mmi=l_mmi,
            min_feature=min_feature,
            n_pml=n_pml,
            cpw=cpw,
            n_cells_feature=int(n_cells_feature),
        )
        grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
        grid.remove_edge_ghost(2)
        return grid

    margin = 20.0 * float(wl_max)
    bg_box = Box(
        center=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.0),
        size=(xmax - xmin + margin, ymax - ymin + margin, 4.0 * h_core),
        material=n_to_eps(n_clad),
    )
    lo_s, hi_s = structures_3d[0].bounds
    fg_box = Box(
        center=(0.5 * (lo_s[0] + hi_s[0]), 0.5 * (lo_s[1] + hi_s[1]), 0.0),
        size=(hi_s[0] - lo_s[0], hi_s[1] - lo_s[1], h_core),
        material=n_to_eps(neff_s),
    )
    fg_box.faces = lambda axis: [] if axis in (0, 1) else [-0.5 * h_core, 0.5 * h_core]

    overrides = task_cfg.get("mesh_overrides")
    if overrides is None:
        overrides = build_gap_mesh_overrides(
            l_mmi=l_mmi, n_cells_feature=n_cells_feature, min_feature=min_feature
        )

    sim = Simulation(
        f_min,
        f_max,
        structures=[bg_box, fg_box],
        boundaries=(["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"]),
        domain=domain,
        cells_per_wavelength=cpw,
        n_pml=n_pml,
        mesh_overrides=overrides,
    )
    x, y, z = [sim._axis(a) for a in range(3)]
    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    grid.remove_edge_ghost(2)
    return grid


def extract_zoom_permittivity(
    solver,
    xlim_um=(1.05, 2.25),
    ylim_um=(-0.55, 0.55),
    component="y",
):
    """
    Extract the in-plane relative permittivity ``eps_r`` and node coordinates
    in ``xlim_um x ylim_um`` around the MMI output throat gap.
    """
    g = solver.grid
    iz = solver.Nz // 2
    ieps = solver.ieps[:, :, iz, component]
    if hasattr(ieps, "get"):
        ieps = ieps.get()
    eps_r = np.where(ieps > 0, 1.0 / (np.where(ieps > 0, ieps, 1.0) * epsilon_0), 1.0)

    x_um = g.x * 1e6
    y_um = g.y * 1e6
    ix_lo = max(0, int(np.searchsorted(x_um, xlim_um[0]) - 1))
    ix_hi = min(len(x_um) - 1, int(np.searchsorted(x_um, xlim_um[1]) + 1))
    iy_lo = max(0, int(np.searchsorted(y_um, ylim_um[0]) - 1))
    iy_hi = min(len(y_um) - 1, int(np.searchsorted(y_um, ylim_um[1]) + 1))

    return {
        "zoom_x": g.x[ix_lo : ix_hi + 1].astype(np.float64),
        "zoom_y": g.y[iy_lo : iy_hi + 1].astype(np.float64),
        "zoom_eps_y": eps_r[ix_lo:ix_hi, iy_lo:iy_hi].astype(np.float32),
    }


def plot_throat_gap_mesh_zoom(
    panels,
    structures=(),
    l_mmi=3.00e-6,
    min_feature=0.20e-6,
    throat_x=THROAT_OVERRIDE_X,
    xlim_um=(1.15, 2.15),
    ylim_um=(-0.48, 0.48),
    out_path=None,
):
    """
    Plot zoomed in-plane permittivity maps with Yee cell edges and annotated
    0.20 um throat-gap width across resolution/averaging levels (e.g. L1
    staircase, L1 Kottke, and Reference Kottke).
    """
    n_p = len(panels)
    fig, axes = figstyle.new_figure(
        nrows=1, ncols=n_p, width=4.4 * n_p, height=4.4, sharey=True
    )
    if n_p == 1:
        axes = [axes]

    xh_um = 0.5 * l_mmi * 1e6
    hy_um = 0.5 * min_feature * 1e6
    tx_um = throat_x * 1e6

    for ax, p in zip(axes, panels):
        x_um = np.asarray(p["zoom_x"]) * 1e6
        y_um = np.asarray(p["zoom_y"]) * 1e6
        eps_r = np.asarray(p["zoom_eps_y"])
        pcm = ax.pcolormesh(
            x_um,
            y_um,
            eps_r.T,
            cmap="viridis",
            shading="flat",
            vmin=2.0,
            vmax=8.3,
        )
        ax.set_xlim(*xlim_um)
        ax.set_ylim(*ylim_um)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        xv = x_um[(x_um >= xl[0]) & (x_um <= xl[1])]
        yv = y_um[(y_um >= yl[0]) & (y_um <= yl[1])]
        vlines = [[(x, yl[0]), (x, yl[1])] for x in xv]
        hlines = [[(xl[0], y), (xl[1], y)] for y in yv]
        ax.add_collection(
            LineCollection(
                vlines + hlines, colors="white", linewidths=0.45, alpha=0.65
            )
        )

        for s in structures:
            if isinstance(s, PolySlab):
                ax.add_patch(
                    MplPolygon(
                        s.vertices * 1e6,
                        closed=True,
                        fill=False,
                        edgecolor="#ffdd55",
                        lw=1.2,
                        ls="-",
                        alpha=0.9,
                    )
                )

        # Highlight mesh-override box around the throat gap
        rect = Rectangle(
            (xh_um, -hy_um),
            tx_um,
            2.0 * hy_um,
            fill=False,
            edgecolor="#00e5ff",
            lw=1.5,
            ls="--",
            label="Mesh-override box",
        )
        ax.add_patch(rect)

        # Annotate the 200 nm throat gap width at x = +1.52 um
        x_arr = xh_um + 0.03
        ax.annotate(
            "",
            xy=(x_arr, +hy_um),
            xytext=(x_arr, -hy_um),
            arrowprops=dict(arrowstyle="<->", color="#ff5533", lw=1.9),
        )
        caf = float(p.get("cells_across_feature", 4.0))
        dxmin = float(p.get("dx_min_nm", 50.0))
        dxbg = float(p.get("dx_background_nm", 52.4))
        ax.annotate(
            f"gap = {min_feature * 1e9:.0f} nm\n"
            f"{caf:.1f} cells across\n"
            f"(dx_min = {dxmin:.1f} nm,\n"
            f" dx_bg = {dxbg:.1f} nm)",
            xy=(x_arr, -hy_um),
            xytext=(xh_um - 0.31, -0.34),
            color="white",
            fontsize=8.5,
            fontweight="bold",
            va="center",
            ha="left",
            arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=1.2),
            bbox=dict(
                boxstyle="round,pad=0.22",
                fc="#111822",
                ec="#00e5ff",
                lw=0.9,
                alpha=0.88,
            ),
        )

        ax.set(aspect="equal", xlabel="x [µm]", title=p["title"])
        cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"Relative permittivity $\varepsilon_{r,y}$")

    axes[0].set_ylabel("y [µm]")
    if out_path is not None:
        figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return fig, axes


class _PolygonSchematicProxy:
    """
    Lightweight polygon structure proxy exposing ``.vertices``, ``.material``,
    and ``.name`` (without ``.bounds`` or ``.radius``) so ``wakis.schematic``
    classifies a 2D ``PolySlab`` as ``kind='polygon'`` rather than a bounding-box
    rectangle.
    """

    def __init__(self, slab, material_eps: float, name: str):
        self.vertices = np.asarray(slab.vertices, dtype=float)
        self.material = float(material_eps)
        self.name = str(name)


class _SchematicSolverProxy:
    """
    Lightweight proxy around ``SolverFIT3D`` that hides ``ieps`` from
    ``wakis.schematic.extract_setup_info`` so explicit ``structures`` are used.
    """

    def __init__(self, solver, eps_clad: float):
        self._solver = solver
        self.grid = solver.grid
        self.Nx = solver.Nx
        self.Ny = solver.Ny
        self.Nz = solver.Nz
        self.dt = solver.dt
        self.n_pml = getattr(solver, "n_pml", 8)
        self.bc_low = getattr(solver, "bc_low", ["pml", "pml", "periodic"])
        self.bc_high = getattr(solver, "bc_high", ["pml", "pml", "periodic"])
        self.bg = [float(eps_clad), 1.0, 0.0]
        self._assigned_structures = ()


def _build_mmi_2x2_sources_and_monitors(
    pic,
    *,
    y_port: float = 1.30e-6,
    x_port: float = 6.00e-6,
    include_flux_box: bool = True,
):
    """Build ``(sources, monitors)`` dicts for ``wakis.schematic`` from ``pic``."""
    xmin, xmax = float(pic.grid.xmin), float(pic.grid.xmax)
    ymin, ymax = float(pic.grid.ymin), float(pic.grid.ymax)
    dx_bg = float(np.median(np.diff(pic.grid.x[:-1])))
    dy_bg = float(np.median(np.diff(pic.grid.y[:-1])))
    n_pml = int(getattr(pic.solver, "n_pml", 8))
    pulse_obj = pic.sim.pulse
    f0 = float(pulse_obj.frequency)
    fwidth = float(pulse_obj.fwidth)

    x_src = -float(x_port) - 0.12e-6
    sources_s = [
        {
            "kind": "sheet",
            "type_name": "ModeSource",
            "short_word": "ModeSrc",
            "label": "Port 1 'in1' ModeSource (TE₀, +x)",
            "desc": f"Port 1 'in1' ModeSource at x = {x_src * 1e6:.2f} µm, y = +{y_port * 1e6:.2f} µm (TE₀/EIM, +x)",
            "h_bounds": (x_src, x_src),
            "v_bounds": (float(y_port) - 0.85e-6, float(y_port) + 0.85e-6),
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
    ]
    monitors_s = [
        {
            "kind": "plane",
            "type_name": "ModeMonitor",
            "short_word": "in1/in2",
            "label": "Ports 'in1'/'in2' ModeMonitors (S₁₁, S₂₁)",
            "desc": f"Ports 'in1'/'in2' ModeMonitors at x = -{x_port * 1e6:.2f} µm (y = ±{y_port * 1e6:.2f} µm)",
            "h_bounds": (-float(x_port), -float(x_port)),
            "v_bounds": (-float(y_port) - 0.85e-6, +float(y_port) + 0.85e-6),
        },
        {
            "kind": "plane",
            "type_name": "ModeMonitor",
            "short_word": "out1/out2",
            "label": "Ports 'out1' (bar) / 'out2' (cross) ModeMonitors (S₃₁, S₄₁)",
            "desc": f"Ports 'out1'/'out2' ModeMonitors at x = +{x_port * 1e6:.2f} µm (y = ±{y_port * 1e6:.2f} µm)",
            "h_bounds": (+float(x_port), +float(x_port)),
            "v_bounds": (-float(y_port) - 0.85e-6, +float(y_port) + 0.85e-6),
        },
    ]
    if include_flux_box:
        pad_x = (n_pml + 2) * dx_bg
        pad_y = (n_pml + 2) * dy_bg
        monitors_s.append(
            {
                "kind": "box",
                "type_name": "FluxBox",
                "short_word": "FluxBox",
                "label": "2D Poynting FluxBox",
                "desc": "Closed 2D Poynting FluxBox (4 faces inside CPML)",
                "h_bounds": (xmin + pad_x, xmax - pad_x),
                "v_bounds": (ymin + pad_y, ymax - pad_y),
            }
        )
    return sources_s, monitors_s


def save_mmi_2x2_setup_figures(
    pic,
    structures,
    primary_data: dict,
    out_dir: str | Path,
    *,
    l_mmi: float = 3.00e-6,
    w_mmi: float = 1.00e-6,
    w_wg: float = 0.40e-6,
    min_feature: float = 0.20e-6,
    y_port: float = 1.30e-6,
    x_port: float = 6.00e-6,
) -> dict:
    """
    Generate ``00a_setup_geometry.png``, ``00b_setup_mesh_and_excitation.png``,
    and ``00_setup_table.md`` for Tutorial 5.7 using
    ``wakis.schematic.save_setup_figures``, enhanced with the launched input
    mode profile ``|E_y(y)|`` on Port 1 (`in1`) and the 2x2 MMI polygon on the
    mesh and 1x zoom inset around the 200 nm S-bend throat gap.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    eps_clad = n_to_eps(pic.n_clad)
    eps_slab = n_to_eps(pic.n_eff_slab)
    solver_proxy = _SchematicSolverProxy(pic.solver, eps_clad=eps_clad)
    poly_proxy = _PolygonSchematicProxy(
        structures[0],
        material_eps=eps_slab,
        name=f"SOI 220 nm EIM (n_eff = {pic.n_eff_slab:.2f})",
    )
    mat_names = {
        "background": f"SiO₂ cladding (n = {pic.n_clad:.2f}, ε_r = {eps_clad:.2f})",
        "region_0": f"SOI 220 nm EIM (n_eff = {pic.n_eff_slab:.2f}, ε_eff = {eps_slab:.2f})",
    }
    sources_s, monitors_s = _build_mmi_2x2_sources_and_monitors(
        pic, y_port=y_port, x_port=x_port, include_flux_box=True
    )
    variations = [
        {
            "name": "kottke_L1..L3 / staircase_L1..L3 (tasks 0–5)",
            "param": "L1=(10 cpw, N=4), L2=(14, 6), L3=(20, 8)",
            "note": "Resolution ladder with throat-gap mesh_overrides at L_mmi = 3.00 µm",
        },
        {
            "name": "kottke_reference (task 6)",
            "param": "R=(40 cpw, N=16) (1016 × 403 × 1 = 409,448 cells)",
            "note": "Independent 2× fine Kottke reference",
        },
        {
            "name": "length_*um (tasks 7–10)",
            "param": "L_mmi ∈ {2.50, 3.00, 3.25, 3.50, 3.80} µm at L3=(20, 8)",
            "note": "General-interference sweep (2D EIM 50/50 split at L_mmi = 3.25 µm)",
        },
        {
            "name": "kottke_nonuniform (task 11)",
            "param": "Graded non-square Cartesian mesh (425 × 171 × 1 = 72,675 cells)",
            "note": "Exact nodes at x = ±1.50 µm, y = ±0.10, ±0.50 µm; 8.0 cells across 200 nm gap",
        },
    ]

    res = schematic.save_setup_figures(
        solver_proxy,
        out_path,
        prefix="00",
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=pic.wavelength_ref,
        min_feature=min_feature,
        title_geometry=(
            "2×2 SOI Multimode-Interference Power Splitter (2D EIM Setup Schematic)"
        ),
        title_mesh=(
            "Discretization (Graded Non-Uniform Mesh & 1× Zoom on 200 nm Throat Gap) and Excitation"
        ),
        variations=variations,
    )

    # Enhance 00a_setup_geometry.png with top/bottom CPML shading and launched mode profile |Ey(y)|
    fig_a, ax_a = schematic.plot_schematic(
        solver_proxy,
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=pic.wavelength_ref,
        title="2×2 SOI Multimode-Interference Power Splitter (2D EIM Setup Schematic)",
        width_in=7.6,
        height_in=4.6,
    )
    x0_um = float(pic.grid.xmin * 1e6)
    x1_um = float(pic.grid.xmax * 1e6)
    y0_um = float(pic.grid.ymin * 1e6)
    y1_um = float(pic.grid.ymax * 1e6)
    n_pml = int(getattr(pic.solver, "n_pml", 8))
    th_y_low = float((pic.grid.y[n_pml] - pic.grid.y[0]) * 1e6)
    th_y_high = float((pic.grid.y[-2] - pic.grid.y[-2 - n_pml]) * 1e6)
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
    x_src_um = float(sources_s[0]["h_bounds"][0] * 1e6)
    y_port_um = float(y_port * 1e6)
    for ann in ax_a.texts:
        if ann.get_text() == "" and getattr(ann, "arrow_patch", None) is not None:
            if abs(ann.xy[1]) < 0.35 and abs(ann.xy[0] - x_src_um) < 1.5:
                ann.xy = (x_src_um + 0.82, y_port_um)
                ann.set_position((x_src_um, y_port_um))
    if "port_y_in" in primary_data and "port_ey_in" in primary_data:
        y_m = np.asarray(primary_data["port_y_in"], dtype=float) * 1e6
        ey_m = np.abs(np.asarray(primary_data["port_ey_in"]))
        ey_m = ey_m / max(float(np.max(ey_m)), 1e-30)
        x_prof = x_src_um + 0.90 * ey_m
        ax_a.fill_betweenx(
            y_m, x_src_um, x_prof, color="#D55E00", alpha=0.28, zorder=7.5
        )
        ax_a.plot(
            x_prof,
            y_m,
            color="#D55E00",
            lw=1.8,
            marker="none",
            zorder=8,
            label="Launched mode |Ey(y)|",
        )
    ax_a.text(
        0.0,
        0.0,
        f"MMI: {l_mmi * 1e6:.2f} × {w_mmi * 1e6:.2f} µm",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color=figstyle.INK,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="#FFFFFFE8",
            edgecolor="#B0B8C2",
            lw=0.6,
        ),
        zorder=9,
    )
    for lbl, xp_um, yp_um, ha_val in (
        ("in1 (drive)", -3.35, +1.55, "center"),
        ("in2 (idle)", -3.35, -1.62, "center"),
        ("out1 (bar, S₃₁)", +3.35, +1.55, "center"),
        ("out2 (cross, S₄₁)", +3.35, -1.62, "center"),
    ):
        ax_a.text(
            xp_um,
            yp_um,
            lbl,
            ha=ha_val,
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#006B4E",
            zorder=9,
        )
    tag_positions = {
        "1 · ModeSrc": (-5.90, +2.06, "left"),
        "2 · in1/in2": (-5.80, -0.35, "left"),
        "3 · out1/out2": (+5.80, 0.00, "right"),
        "4 · FluxBox": (0.00, +2.15, "center"),
    }
    for txt in list(ax_a.texts):
        s = txt.get_text().strip()
        if s in tag_positions:
            tx, ty, ha_val = tag_positions[s]
            txt.set_position((tx, ty))
            txt.set_ha(ha_val)
    figstyle.save(fig_a, res["geometry_figure"], close=True, auto_legend_below=False)

    # Enhance 00b_setup_mesh_and_excitation.png with the 2x2 MMI polygon on ax_mesh and zoom inset
    xh = 0.5 * float(l_mmi)
    zoom_bounds = (xh - 0.25e-6, xh + 0.45e-6, -0.35e-6, +0.35e-6)
    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        solver_proxy,
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        mesh_every=6,
        zoom_bounds=zoom_bounds,
        wavelength=pic.wavelength_ref,
        min_feature=min_feature,
        title="Discretization (Graded Non-Uniform Mesh & 1× Zoom on 200 nm Throat Gap) and Excitation",
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


def plot_mesh_upgrade_comparison_mmi_2x2(
    pic_uniform,
    pic_nonuniform,
    structures,
    d_fine: dict,
    d_nonuniform: dict,
    d_ref: dict | None,
    meta_fine: dict | None,
    meta_nonuniform: dict | None,
    out_path: str | Path,
    l_mmi: float = 3.00e-6,
    w_mmi: float = 1.00e-6,
    min_feature: float = 0.20e-6,
):
    """
    Render a 3-panel comparison figure (`04b_mesh_upgrade_comparison.png`):
    (a) Baseline L3 mesh (`kottke_L3`, 20 cpw + throat override) around the MMI box and throat gap,
    (b) Upgraded graded non-uniform, non-square Cartesian mesh (`kottke_nonuniform`),
    (c) Broadband bar/cross/total transmission spectra and cell-count/runtime comparison vs R=(40,16).
    """
    figstyle.apply_style()
    fig, (ax0, ax1, ax2) = figstyle.new_figure(
        nrows=1, ncols=3, width=14.6, height=4.5
    )

    xlim_um = (-0.20, 3.10)
    ylim_um = (-1.65, 1.65)
    verts_um = np.asarray(structures[0].vertices, dtype=float) * 1e6

    for ax, pic_obj, panel_ttl in (
        (
            ax0,
            pic_uniform,
            f"(a) Baseline L3 mesh ({pic_uniform.grid.Nx}×{pic_uniform.grid.Ny})",
        ),
        (
            ax1,
            pic_nonuniform,
            f"(b) Graded non-square mesh ({pic_nonuniform.grid.Nx}×{pic_nonuniform.grid.Ny})",
        ),
    ):
        g = pic_obj.grid
        iz = g.Nz // 2
        ieps_y = pic_obj.solver.ieps[:, :, iz, "y"]
        if hasattr(ieps_y, "get"):
            ieps_y = ieps_y.get()
        ieps_y = np.asarray(ieps_y, dtype=float)
        with np.errstate(divide="ignore"):
            eps_y = np.where(ieps_y > 0.0, 1.0 / ieps_y, 1.0)
        xu = np.asarray(g.x, dtype=float) * 1e6
        yu = np.asarray(g.y, dtype=float) * 1e6
        nx_c = len(xu) - 1
        ny_c = len(yu) - 1
        pcm = ax.pcolormesh(
            xu,
            yu,
            eps_y[:nx_c, :ny_c].T,
            cmap="Blues",
            shading="flat",
            vmin=2.0,
            vmax=float(pic_obj.n_eff_slab**2) * 1.03,
            alpha=0.88,
        )
        ax.set_xlim(*xlim_um)
        ax.set_ylim(*ylim_um)
        xv = xu[(xu >= xlim_um[0]) & (xu <= xlim_um[1])]
        yv = yu[(yu >= ylim_um[0]) & (yu <= ylim_um[1])]
        vlines = [[(x, ylim_um[0]), (x, ylim_um[1])] for x in xv]
        hlines = [[(xlim_um[0], y), (xlim_um[1], y)] for y in yv]
        ax.add_collection(
            LineCollection(
                vlines + hlines, colors="#1A202C", linewidths=0.35, alpha=0.52
            )
        )
        ax.add_patch(
            MplPolygon(
                verts_um,
                closed=True,
                fill=False,
                edgecolor="#D55E00",
                lw=1.35,
            )
        )
        for y_mark in (-0.5 * w_mmi * 1e6, +0.5 * w_mmi * 1e6):
            ax.axhline(y_mark, color="#009E73", ls="--", lw=1.0, alpha=0.85)
        st = measure_grid_resolution(
            g,
            n_pml=pic_obj.n_pml,
            l_mmi=l_mmi,
            min_feature=min_feature,
            w_mmi=w_mmi,
        )
        ax.text(
            0.03,
            0.04,
            f"dx ∈ [{st['dx_min_nm']:.1f}, {st['dx_max_nm']:.1f}] nm\n"
            f"dy ∈ [{st['dy_min_nm']:.1f}, {st['dy_max_nm']:.1f}] nm\n"
            f"gap: {st['cells_across_feature']:.1f} cells, wall err: {st['sidewall_err_nm']:.1f} nm",
            transform=ax.transAxes,
            fontsize=8.5,
            va="bottom",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.24",
                facecolor="#FFFFFFE8",
                edgecolor="#CBD5E0",
                lw=0.7,
            ),
        )
        ax.set(xlabel="x [µm]", title=panel_ttl, aspect="equal")
        cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"$\varepsilon_{r,y}$")
    ax0.set_ylabel("y [µm]")

    def _extract_spectra(d):
        w = np.asarray(d["wavelengths"], dtype=float) * 1e9
        ord_i = np.argsort(w)
        w_s = w[ord_i]
        m = (w_s >= 1505.0) & (w_s <= 1595.0)
        s31 = d["S31"][ord_i][m]
        s41 = d["S41"][ord_i][m]
        tb = np.abs(s31) ** 2
        tc = np.abs(s41) ** 2
        return w_s[m], tb, tc, tb + tc, np.rad2deg(np.angle(s31 * np.conj(s41)))

    wl_f, tb_f, tc_f, tt_f, dp_f = _extract_spectra(d_fine)
    wl_n, tb_n, tc_n, tt_n, dp_n = _extract_spectra(d_nonuniform)
    ax2.plot(
        wl_f,
        tc_f,
        color="#2b5c8f",
        lw=2.0,
        label=r"L3 cross $|S_{41}|^2$ & bar $|S_{31}|^2$",
    )
    ax2.plot(wl_f, tb_f, color="#2b5c8f", lw=1.8, ls="-.")
    ax2.plot(
        wl_n,
        tc_n,
        "--",
        color="#d95f02",
        lw=2.1,
        label=r"Graded non-square $|S_{41}|^2$ & $|S_{31}|^2$",
    )
    ax2.plot(wl_n, tb_n, ":", color="#d95f02", lw=2.0)

    if d_ref is not None:
        wl_r, tb_r, tc_r, tt_r, dp_r = _extract_spectra(d_ref)
        ax2.plot(
            wl_r,
            tc_r,
            ":",
            color="#111111",
            lw=1.9,
            label=r"Reference R (40 cpw, N=16)",
        )
        ax2.plot(wl_r, tb_r, ":", color="#111111", lw=1.7)
        rms_f = float(
            np.sqrt(0.5 * np.mean((tb_f - tb_r) ** 2 + (tc_f - tc_r) ** 2))
        )
        rms_n = float(
            np.sqrt(0.5 * np.mean((tb_n - tb_r) ** 2 + (tc_n - tc_r) ** 2))
        )
    else:
        rms_f = 0.0
        rms_n = float(
            np.sqrt(0.5 * np.mean((tb_n - tb_f) ** 2 + (tc_n - tc_f) ** 2))
        )

    i0_f = int(np.argmin(np.abs(wl_f - 1550.0)))
    i0_n = int(np.argmin(np.abs(wl_n - 1550.0)))
    nc_f = (
        int(meta_fine["n_cells"])
        if meta_fine
        else int(pic_uniform.grid.Nx * pic_uniform.grid.Ny)
    )
    nc_n = (
        int(meta_nonuniform["n_cells"])
        if meta_nonuniform
        else int(pic_nonuniform.grid.Nx * pic_nonuniform.grid.Ny)
    )
    t_f_s = float(meta_fine["elapsed_s"]) if meta_fine else 0.0
    t_n_s = float(meta_nonuniform["elapsed_s"]) if meta_nonuniform else 0.0
    cell_red = 100.0 * (1.0 - nc_n / max(nc_f, 1))

    ax2.text(
        0.03,
        0.04,
        f"Cells: {nc_n:,} vs {nc_f:,} (−{cell_red:.1f}%)\n"
        f"Wall time: {t_n_s:.1f} s vs {t_f_s:.1f} s\n"
        f"Split(1550): {100 * tb_n[i0_n] / tt_n[i0_n]:.2f}% vs {100 * tb_f[i0_f] / tt_f[i0_f]:.2f}% bar\n"
        f"Δφ(1550): {dp_n[i0_n]:.2f}° vs {dp_f[i0_f]:.2f}°\n"
        f"2-port RMS vs ref: {rms_n:.2e} vs {rms_f:.2e}",
        transform=ax2.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#FFFFFFE8",
            edgecolor="#CBD5E0",
            lw=0.7,
        ),
    )
    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Modal power transmission",
        ylim=(0.14, 0.64),
        title="(c) Bar & cross spectra vs reference",
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5)
    figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return fig


def render_mmi_2x2_field_map_with_overlay(
    x: np.ndarray,
    y: np.ndarray,
    field_2d: np.ndarray,
    pic,
    structures,
    *,
    kind: str = "intensity_log",
    title: str = "",
    out_path: str | Path | None = None,
):
    """
    Render a 2D field map (`intensity_log` or `real`) with `wakis.schematic.overlay_geometry`
    (material outlines + labels, port planes, and CPML boundary shading).
    """
    figstyle.apply_style()
    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=10.6, height=4.4)
    x_um = np.asarray(x, dtype=float) * 1e6
    y_um = np.asarray(y, dtype=float) * 1e6

    if kind == "intensity_log":
        f_abs = np.abs(np.asarray(field_2d, dtype=float))
        f_max = max(float(np.max(f_abs)), 1e-30)
        data = 10.0 * np.log10(np.maximum(f_abs / f_max, 1e-4))
        vmin, vmax = -35.0, 0.0
        cmap = "inferno"
        cb_label = r"Normalized intensity $10\log_{10}(|E|^2 / |E|_{\max}^2)$ [dB]"
        outline_col = "white"
    else:
        data = np.real(np.asarray(field_2d, dtype=complex))
        vmax = float(np.percentile(np.abs(data), 99.2)) or 1.0
        vmin = -vmax
        cmap = "RdBu_r"
        cb_label = r"$\mathrm{Re}(E_y)$ [arb. u.]"
        outline_col = "#1E293B"

    im = ax.pcolormesh(
        x_um,
        y_um,
        data.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label(cb_label)

    eps_clad = n_to_eps(pic.n_clad)
    eps_slab = n_to_eps(pic.n_eff_slab)
    solver_proxy = _SchematicSolverProxy(pic.solver, eps_clad=eps_clad)
    poly_proxy = _PolygonSchematicProxy(
        structures[0],
        material_eps=eps_slab,
        name=f"SOI 220 nm EIM (n={pic.n_eff_slab:.2f})",
    )
    sources_s, monitors_s = _build_mmi_2x2_sources_and_monitors(
        pic, include_flux_box=False
    )
    schematic.overlay_geometry(
        ax,
        solver_proxy,
        plane="xy",
        unit="µm",
        structures=[poly_proxy],
        material_names={
            "background": f"SiO₂ (n={pic.n_clad:.2f})",
            "region_0": f"SOI 220 nm (n_eff={pic.n_eff_slab:.2f})",
        },
        sources=sources_s,
        monitors=monitors_s,
        show_boundaries=True,
        show_materials=True,
        show_sources=True,
        show_monitors=True,
        fill_materials=False,
        outline_color=outline_col,
        label_materials=True,
    )
    n_pml = int(getattr(pic.solver, "n_pml", 8))
    y0_um, y1_um = float(pic.grid.ymin * 1e6), float(pic.grid.ymax * 1e6)
    x0_um, x1_um = float(pic.grid.xmin * 1e6), float(pic.grid.xmax * 1e6)
    th_y = float((pic.grid.y[n_pml] - pic.grid.y[0]) * 1e6)
    for yb_s in (y0_um, y1_um - th_y):
        ax.add_patch(
            Rectangle(
                (x0_um, yb_s),
                x1_um - x0_um,
                th_y,
                facecolor="#94A3B8",
                edgecolor="#475569",
                hatch="///",
                alpha=0.18,
                lw=0.6,
                zorder=5,
            )
        )
    ax.set(
        xlim=(float(x_um[0]), float(x_um[-1])),
        ylim=(float(y_um[0]), float(y_um[-1])),
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=title,
    )
    ax.set_aspect("equal", adjustable="box")
    if out_path is not None:
        figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return fig, ax


def render_mmi_2x2_field_comparison_with_overlay(
    x: np.ndarray,
    y: np.ndarray,
    e2_1550: np.ndarray,
    e2_1580: np.ndarray,
    pic,
    structures,
    out_path: str | Path,
):
    """
    Render the side-by-side linear intensity comparison (`09_field_comparison_1550_vs_1580.png`)
    at 1550 nm vs 1580 nm with `wakis.schematic.overlay_geometry` on both panels.
    """
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(
        nrows=1, ncols=2, width=11.6, height=4.2, sharey=True
    )
    x_um = np.asarray(x, dtype=float) * 1e6
    y_um = np.asarray(y, dtype=float) * 1e6
    eps_clad = n_to_eps(pic.n_clad)
    eps_slab = n_to_eps(pic.n_eff_slab)
    solver_proxy = _SchematicSolverProxy(pic.solver, eps_clad=eps_clad)
    poly_proxy = _PolygonSchematicProxy(
        structures[0],
        material_eps=eps_slab,
        name=f"SOI 220 nm EIM (n={pic.n_eff_slab:.2f})",
    )
    sources_s, monitors_s = _build_mmi_2x2_sources_and_monitors(
        pic, include_flux_box=False
    )

    for ax, field_arr, wl_lbl in zip(
        axes,
        (e2_1550, e2_1580),
        (
            r"(a) $\lambda = 1550\;\mathrm{nm}$ (C-band center)",
            r"(b) $\lambda = 1580\;\mathrm{nm}$ (long-band monitor)",
        ),
    ):
        vmax = float(np.percentile(field_arr, 99.5)) or 1.0
        im = ax.pcolormesh(
            x_um,
            y_um,
            np.clip(field_arr.T / vmax, 0.0, 1.0),
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            shading="auto",
        )
        schematic.overlay_geometry(
            ax,
            solver_proxy,
            plane="xy",
            unit="µm",
            structures=[poly_proxy],
            material_names={
                "background": f"SiO₂ (n={pic.n_clad:.2f})",
                "region_0": f"SOI 220 nm (n_eff={pic.n_eff_slab:.2f})",
            },
            sources=sources_s,
            monitors=monitors_s,
            show_boundaries=True,
            show_materials=True,
            show_sources=True,
            show_monitors=True,
            fill_materials=False,
            outline_color="white",
            label_materials=True,
        )
        ax.set(
            aspect="equal",
            xlim=(-6.2, 6.2),
            ylim=(-2.2, 2.2),
            xlabel="x [µm]",
            title=wl_lbl,
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"Normalized $|E|^2 / |E|_{\max}^2$")
    axes[0].set_ylabel("y [µm]")
    figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return fig, axes


def render_mmi_2x2_movie_with_overlay(
    x: np.ndarray,
    y: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray,
    pic,
    structures,
    out_path: str | Path,
    *,
    fps: int = 12,
):
    """
    Render the animated pulse propagation GIF (`10_pulse_propagation.gif`) with
    `wakis.schematic.overlay_geometry` on every frame and file size < 5 MB.
    """
    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=(9.6, 3.9), dpi=110)
    x_um = np.asarray(x, dtype=float) * 1e6
    y_um = np.asarray(y, dtype=float) * 1e6
    frames_arr = np.asarray(frames, dtype=np.float32)
    vmax = float(np.percentile(np.abs(frames_arr), 99.2)) or 1.0

    im = ax.pcolormesh(
        x_um,
        y_um,
        frames_arr[0].T,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label(r"$E_y(x, y, t)$ [arb. u.]")

    eps_clad = n_to_eps(pic.n_clad)
    eps_slab = n_to_eps(pic.n_eff_slab)
    solver_proxy = _SchematicSolverProxy(pic.solver, eps_clad=eps_clad)
    poly_proxy = _PolygonSchematicProxy(
        structures[0],
        material_eps=eps_slab,
        name=f"SOI 220 nm EIM (n={pic.n_eff_slab:.2f})",
    )
    sources_s, monitors_s = _build_mmi_2x2_sources_and_monitors(
        pic, include_flux_box=False
    )
    schematic.overlay_geometry(
        ax,
        solver_proxy,
        plane="xy",
        unit="µm",
        structures=[poly_proxy],
        material_names={
            "background": f"SiO₂ (n={pic.n_clad:.2f})",
            "region_0": f"SOI 220 nm (n_eff={pic.n_eff_slab:.2f})",
        },
        sources=sources_s,
        monitors=monitors_s,
        show_boundaries=True,
        show_materials=True,
        show_sources=True,
        show_monitors=True,
        fill_materials=False,
        outline_color="#1E293B",
        label_materials=True,
    )
    ax.set(
        aspect="equal",
        xlim=(float(x_um[0]), float(x_um[-1])),
        ylim=(float(y_um[0]), float(y_um[-1])),
        xlabel="x [µm]",
        ylabel="y [µm]",
    )
    title_obj = ax.set_title(
        f"2×2 SOI MMI Coupler · Pulse propagation (t = {float(times[0]) * 1e15:.1f} fs)"
    )
    fig.tight_layout()

    def _update(i):
        im.set_array(frames_arr[i].T.ravel())
        title_obj.set_text(
            f"2×2 SOI MMI Coupler · Pulse propagation (t = {float(times[i]) * 1e15:.1f} fs)"
        )
        return (im, title_obj)

    ani = animation.FuncAnimation(
        fig, _update, frames=len(frames_arr), interval=1000 // fps, blit=False
    )
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    return out_path
