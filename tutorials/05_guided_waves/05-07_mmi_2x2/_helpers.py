# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Resolution-ladder, mesh-override, and plotting helpers for Tutorial 5.7
(2x2 MMI power splitter in 2D EIM).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from scipy.constants import c as c_light, epsilon_0

from wakis.gridFIT3D import GridFIT3D
from wakis.templates import Simulation, meshing
from wakis.templates.photonics import Box, PolySlab, n_to_eps, slab_neff

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

    Returns
    -------
    metrics : dict
        Dictionary containing ``min_feature_nm``, ``dx_background_nm``,
        ``dx_min_nm``, ``cells_across_feature``, ``grid_shape``, and
        ``n_cells``.
    """
    dx = np.diff(grid.x)
    dy = np.diff(grid.y)
    n_p = int(n_pml or 0)
    dx_int = dx[n_p : len(dx) - n_p] if len(dx) > 2 * n_p else dx
    dy_int = dy[n_p : len(dy) - n_p] if len(dy) > 2 * n_p else dy

    dx_bg = float(max(np.max(dx_int), np.max(dy_int)))
    dx_min = float(min(np.min(dx_int), np.min(dy_int)))

    hy = 0.5 * float(min_feature)
    y_nodes = grid.y
    in_gap_y = (y_nodes[1:] > -hy + 1e-15) & (y_nodes[:-1] < hy - 1e-15)
    max_dy_gap = float(np.max(dy[in_gap_y]))

    xh = 0.5 * float(l_mmi)
    tx = float(throat_x)
    x_nodes = grid.x
    in_gap_x = (
        (x_nodes[1:] > -xh - tx + 1e-15) & (x_nodes[:-1] < -xh - 1e-15)
    ) | ((x_nodes[1:] > xh + 1e-15) & (x_nodes[:-1] < xh + tx - 1e-15))
    max_dx_gap = float(np.max(dx[in_gap_x])) if np.any(in_gap_x) else max_dy_gap

    max_cell_across_feature = max(max_dy_gap, max_dx_gap)
    cells_across = float(min_feature) / max_cell_across_feature

    nx, ny, nz = int(grid.Nx), int(grid.Ny), int(grid.Nz)
    return {
        "min_feature_nm": round(float(min_feature) * 1e9, 3),
        "dx_background_nm": round(dx_bg * 1e9, 3),
        "dx_min_nm": round(dx_min * 1e9, 3),
        "cells_across_feature": round(cells_across, 3),
        "grid_shape": [nx, ny, nz],
        "n_cells": int(nx * ny * nz),
    }


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
    fig, axes = plt.subplots(
        1, n_p, figsize=(4.3 * n_p, 4.4), dpi=150, sharey=True
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
            fontsize=7.4,
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
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig, axes
