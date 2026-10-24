# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Resolution ladder, mesh-override construction, actual-mesh feature-cell
measurement, and zoomed permittivity/mesh plotting helpers for Tutorial 5.2
(Hybrid Euler 90-degree silicon waveguide bend).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from scipy.constants import c as c_light

from wakis.templates.photonics import PIC_MATERIALS, slab_neff

# Resolution ladder: each level is (cpw, N_min_cells_across_feature)
RESOLUTION_LEVELS = {
    "L1": (10, 4),
    "L2": (14, 6),
    "L3": (20, 8),
    "R": (40, 16),
}


def build_mesh_overrides(
    r_eff=4.0e-6,
    width=0.40e-6,
    cpw=10,
    n_feature=4,
    n_eff_slab=None,
    wl_min=1500e-9,
    wl_ref=1550e-9,
    h_si=210e-9,
    port_offset=1.0e-6,
):
    """
    Build ``mesh_overrides`` boxes ``[(lo, hi, max_cell), ...]`` for a 90-degree
    bend when the target feature cell size ``width / n_feature`` is finer than
    the background core cell size ``wl_min / (n_eff_slab * cpw)``.

    Parameters
    ----------
    r_eff : float
        Effective footprint radius $R_{\\text{eff}}$ [m].
    width : float
        Smallest geometric feature size (waveguide width $w_{\\text{wg}}$) [m].
    cpw : float
        Background cells per wavelength inside the slab effective index.
    n_feature : int
        Minimum target number of cells $N$ across ``width``.
    n_eff_slab : float, optional
        2D EIM vertical slab effective index. Computed if ``None``.
    wl_min, wl_ref, h_si, port_offset : float, optional
        Wavelength bounds, silicon thickness, and port offset [m].

    Returns
    -------
    list of tuple
        List of ``(lo, hi, max_cell)`` override boxes [m], or ``[]`` if the
        background core mesh already resolves ``width`` with $\\ge N$ cells.
    """
    if n_eff_slab is None:
        n_eff_slab = slab_neff(
            PIC_MATERIALS["Si"],
            PIC_MATERIALS["SiO2"],
            h_si,
            wl_ref,
            polarization="TE",
        )
    dx_bg = float(wl_min) / (float(n_eff_slab) * float(cpw))
    h_target = float(width) / float(n_feature)
    if h_target >= dx_bg * (1.0 - 1e-6):
        return []
    hw = 0.5 * float(width)
    r_eff = float(r_eff)
    port_offset = float(port_offset)
    return [
        (
            (-port_offset, -hw, -np.inf),
            (r_eff + hw, r_eff + port_offset, np.inf),
            h_target,
        )
    ]


def measure_mesh_resolution(
    pic,
    r_eff=4.0e-6,
    min_feature=0.40e-6,
    port_offset=1.0e-6,
):
    """
    Measure background cell size, smallest cell size, and actual number of cells
    across the smallest feature directly from ``pic.grid``.

    Parameters
    ----------
    pic : PICSimulation
        Built 2D/3D PIC simulation object.
    r_eff : float
        Effective bend radius $R_{\\text{eff}}$ [m].
    min_feature : float
        Smallest device feature size [m] ($w_{\\text{wg}} = 400\\;\\text{nm}$).
    port_offset : float
        Port offset from the bend entrance/exit [m].

    Returns
    -------
    dict
        Resolution metrics with keys ``min_feature_nm``, ``dx_background_nm``,
        ``dx_min_nm``, ``cells_across_feature``, ``n_cells``, ``n_steps``,
        ``n_steps_max``, and ``grid_shape``.
    """
    x = np.asarray(pic.grid.x, dtype=float)
    y = np.asarray(pic.grid.y, dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)

    min_feature = float(min_feature)
    r_eff = float(r_eff)
    hw = 0.5 * min_feature

    # Cells overlapping the horizontal input waveguide, bend, and vertical output waveguide
    in_feat_x = (x[1:] > -float(port_offset)) & (x[:-1] < r_eff + hw)
    in_feat_y = (y[1:] > -hw) & (y[:-1] < r_eff + float(port_offset))
    dx_feat_max = float(
        max(
            np.max(dx[in_feat_x]) if np.any(in_feat_x) else np.max(dx),
            np.max(dy[in_feat_y]) if np.any(in_feat_y) else np.max(dy),
        )
    )

    dx_background_nm = float(max(np.max(dx), np.max(dy)) * 1e9)
    dx_min_nm = float(min(np.min(dx), np.min(dy)) * 1e9)
    min_feature_nm = float(min_feature * 1e9)
    cells_across_feature = float(min_feature / dx_feat_max)

    nx, ny, nz = int(pic.grid.Nx), int(pic.grid.Ny), int(pic.grid.Nz)
    n_cells = int(nx * ny * nz)

    lx = float(pic.grid.xmax - pic.grid.xmin)
    ly = float(pic.grid.ymax - pic.grid.ymin)
    transit = np.hypot(lx, ly) * float(pic.n_core) / c_light
    max_time = float(pic.sim.pulse.end_time + 15.0 * transit)
    if pic.solver is not None:
        dt = float(pic.solver.dt)
    else:
        dz = np.diff(np.asarray(pic.grid.z, dtype=float))
        inv_h2 = 1.0 / float(np.min(dx)) ** 2 + 1.0 / float(np.min(dy)) ** 2 + 1.0 / float(np.min(dz)) ** 2
        dt = float(pic.cfln) / (c_light * float(np.sqrt(inv_h2)))
    n_steps_max = int(np.ceil(max_time / dt))
    nt_run = getattr(pic.solver, "Nt", None) if pic.solver is not None else None
    n_steps = int(nt_run) if nt_run is not None else n_steps_max

    return {
        "min_feature_nm": round(min_feature_nm, 3),
        "dx_background_nm": round(dx_background_nm, 3),
        "dx_min_nm": round(dx_min_nm, 3),
        "cells_across_feature": round(cells_across_feature, 3),
        "n_cells": n_cells,
        "n_steps": n_steps,
        "n_steps_max": n_steps_max,
        "grid_shape": [nx, ny, nz],
    }


def plot_feature_mesh_zoom(
    pic_l1_stair,
    pic_l1_kottke,
    pic_ref_kottke,
    structures,
    min_feature_m=0.40e-6,
    ref_label="L3 reference (20 cpw)",
    xlim_um=(-0.35, 1.45),
    ylim_um=(-0.45, 0.55),
    out_path=None,
):
    """
    Plot a 3-panel zoomed view of the Yee mesh cell edges and relative
    permittivity $\\varepsilon_{yy}/\\varepsilon_0$ across the $w = 400\\;\\text{nm}$
    waveguide core entering the clothoid transition at L1 (staircase and Kottke)
    and at the reference level, with the feature width explicitly annotated.
    """
    from scipy.constants import epsilon_0

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), dpi=150, sharey=True)
    configs = [
        (axes[0], pic_l1_stair, "(a) L1 Staircase (10 cpw, N=4)"),
        (axes[1], pic_l1_kottke, "(b) L1 Kottke (10 cpw, N=4)"),
        (axes[2], pic_ref_kottke, f"(c) {ref_label}"),
    ]

    hw_um = 0.5 * float(min_feature_m) * 1e6
    im = None
    for ax, pic_obj, title in configs:
        solver = pic_obj.solver
        g = solver.grid
        iz = solver.Nz // 2
        ieps = solver.ieps[:, :, iz, "y"]
        if hasattr(ieps, "get"):
            ieps = ieps.get()
        ieps = np.asarray(ieps, dtype=float)
        with np.errstate(divide="ignore"):
            eps_r = np.where(ieps > 0, 1.0 / (ieps * epsilon_0), 1.0)

        x_um = g.x * 1e6
        y_um = g.y * 1e6
        im = ax.pcolormesh(
            x_um,
            y_um,
            eps_r.T,
            cmap="viridis",
            shading="flat",
        )
        # Draw Yee cell edges inside zoom window
        xm = x_um[(x_um >= xlim_um[0]) & (x_um <= xlim_um[1])]
        ym = y_um[(y_um >= ylim_um[0]) & (y_um <= ylim_um[1])]
        for xv in xm:
            ax.axvline(xv, color="white", lw=0.35, alpha=0.55)
        for yv in ym:
            ax.axhline(yv, color="white", lw=0.35, alpha=0.55)

        for s in structures:
            ax.add_patch(
                MplPolygon(
                    s.vertices * 1e6,
                    closed=True,
                    fill=False,
                    edgecolor="#ffdd55",
                    lw=1.3,
                    ls="--",
                    alpha=0.95,
                )
            )

        res = measure_mesh_resolution(pic_obj, min_feature=min_feature_m)
        n_feat = res["cells_across_feature"]
        dx_nm = res["dx_min_nm"]

        # Annotate the 400 nm waveguide feature width at x = -0.15 um
        x_ann = -0.15
        ax.annotate(
            "",
            xy=(x_ann, hw_um),
            xytext=(x_ann, -hw_um),
            arrowprops=dict(arrowstyle="<->", color="#ffffff", lw=1.6),
        )
        ax.text(
            x_ann + 0.05,
            0.0,
            f"w = {min_feature_m * 1e9:.0f} nm\n({n_feat:.1f} cells,\nΔx = {dx_nm:.1f} nm)",
            color="white",
            fontsize=7.8,
            fontweight="bold",
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.22", fc="black", ec="white", alpha=0.68),
        )
        ax.set(
            aspect="equal",
            xlim=xlim_um,
            ylim=ylim_um,
            xlabel="x [µm]",
        )
        ax.set_title(
            f"{title}\n({n_feat:.1f} cells across w, Δx={dx_nm:.1f} nm)",
            fontsize=9.2,
        )

    axes[0].set_ylabel("y [µm]")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.025)
    cb.set_label(r"Relative permittivity $\varepsilon_{yy} / \varepsilon_0$")
    fig.suptitle(
        "Zoomed mesh cell edges and permittivity across the 400 nm waveguide feature (L1 vs Reference)",
        fontsize=10.5,
        y=0.96,
    )
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
