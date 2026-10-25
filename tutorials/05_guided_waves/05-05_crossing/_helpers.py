# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# SPDX-License-Identifier: Apache-2.0

"""
Helper classes and mesh/plotting utilities for Tutorial 5.5:
Waveguide crossing based on cosine tapers (2D EIM, Chandran et al. 2020).
"""

import copy
from pathlib import Path

from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light, epsilon_0

from wakis import figstyle, schematic
from wakis.templates import meshing
from wakis.templates.photonics import PICSimulation, PolySlab, n_to_eps, slab_neff
from wakis.templates.shapes import Box
from wakis.templates.simulation import Simulation


class FastCrossingPolySlab(PolySlab):
    """
    `PolySlab` subclass for $D_4$-symmetric 4-port waveguide crossings whose
    orthogonal arms follow a single-valued symmetric half-width profile
    ``h(|u|)`` interpolated at ``(u_nodes, h_nodes)``.

    Evaluates `_contains_xy` via vectorized piecewise-linear interpolation on
    ``(|x|, |y|)`` and caches the 2D in-plane volume fill fraction across the
    periodic-$z$ shifts evaluated by `assign_materials`, while culling exterior
    cladding quadrants outside ``max(|x|, |y|) > arm_half_max``. Produces
    bit-identical permittivity tensors (`max |ieps_diff| == 0`) to `PolySlab`
    at ~5x higher speed.

    Parameters
    ----------
    vertices : ndarray, shape (N, 2)
        Counter-clockwise polygon vertices `(x, y)` [m].
    z_min, z_max : float
        Vertical slab bounds [m].
    material : float or str
        Relative permittivity or material key.
    u_nodes : array_like
        Monotonically increasing axial coordinates `u >= 0` [m] from `0` to
        `L_wg_ext`.
    h_nodes : array_like
        Half-width `h(u)` [m] at each `u_nodes` sample.
    """

    def __init__(
        self, vertices, z_min, z_max, material, u_nodes, h_nodes, **kwargs
    ):
        super().__init__(vertices, z_min, z_max, material, **kwargs)
        self._u_nodes = np.asarray(u_nodes, dtype=float)
        self._h_nodes = np.asarray(h_nodes, dtype=float)
        self.arm_half_max = float(np.max(self._h_nodes)) + 0.05e-6
        self._fxy_cache = {}

    def _contains_xy(self, pts, z=None):
        if self._tan_angle != 0.0 or len(self.holes) > 0:
            return super()._contains_xy(pts, z=z)
        ax = np.abs(pts[:, 0])
        ay = np.abs(pts[:, 1])
        umax = self._u_nodes[-1]
        hx = np.interp(
            ax, self._u_nodes, self._h_nodes, left=self._h_nodes[0], right=0.0
        )
        hy = np.interp(
            ay, self._u_nodes, self._h_nodes, left=self._h_nodes[0], right=0.0
        )
        return ((ax < umax) & (ay < hx)) | ((ay < umax) & (ax < hy))

    def fill(self, X, Y, Z, hx, hy, hz, n_sub=6):
        X, Y, Z, hx, hy, hz = np.broadcast_arrays(
            np.asarray(X, dtype=float),
            np.asarray(Y, dtype=float),
            np.asarray(Z, dtype=float),
            np.asarray(hx, dtype=float),
            np.asarray(hy, dtype=float),
            np.asarray(hz, dtype=float),
        )
        if X.ndim != 3 or X.shape[2] != 1:
            return super().fill(X, Y, Z, hx, hy, hz, n_sub=n_sub)

        out_shape = X.shape
        z_lo = np.maximum(Z - 0.5 * hz, self.z_min)
        z_hi = np.minimum(Z + 0.5 * hz, self.z_max)
        fz = np.where(
            hz > 0,
            np.clip((z_hi - z_lo) / np.where(hz > 0, hz, 1.0), 0.0, 1.0),
            np.where((Z >= self.z_min) & (Z <= self.z_max), 1.0, 0.0),
        )
        if not np.any(fz > 0):
            return np.zeros(out_shape, dtype=float)

        x2d, y2d = X[..., 0], Y[..., 0]
        hx2d, hy2d = hx[..., 0], hy[..., 0]
        key = (
            x2d.shape,
            float(x2d[0, 0]),
            float(x2d[-1, -1]),
            float(y2d[0, 0]),
            float(y2d[-1, -1]),
            float(hx2d[0, 0]),
            float(hy2d[0, 0]),
            int(n_sub),
        )
        if key not in self._fxy_cache:
            ah = self.arm_half_max
            in_cross = ((np.abs(x2d) - 0.5 * hx2d) <= ah) | (
                (np.abs(y2d) - 0.5 * hy2d) <= ah
            )
            z_mid = np.full(
                x2d[in_cross].shape, 0.5 * (self.z_min + self.z_max), dtype=float
            )
            hz_mid = np.full_like(z_mid, self.z_max - self.z_min)
            fxy_2d = np.zeros(x2d.shape, dtype=float)
            fxy_2d[in_cross] = super().fill(
                x2d[in_cross],
                y2d[in_cross],
                z_mid,
                hx2d[in_cross],
                hy2d[in_cross],
                hz_mid,
                n_sub=n_sub,
            )
            self._fxy_cache[key] = fxy_2d
        return self._fxy_cache[key][..., None] * fz


def crossing_mesh_overrides(
    cpw,
    n_feature_min,
    min_feature=0.35e-6,
    wavelength_min=1.26e-6,
    neff_slab=None,
    n_si=3.67,
    n_sio2=1.45,
    h_si=0.161e-6,
    wavelength_ref=1.31e-6,
):
    """
    Build `mesh_overrides` boxes across the smallest feature (`min_feature` [m],
    the `0.35 um` access-waveguide width) for resolution level `(cpw, n_feature_min)`.

    Following the PIC resolution rule, an override box with
    `max_cell = min_feature / n_feature_min` is added only when `max_cell` is
    finer than the background 2D EIM slab cell size
    `h_slab = wavelength_min / (neff_slab * cpw)`. Because the 2D grid is a
    tensor product, a box spanning `[-min_feature/2, +min_feature/2]` on both
    `x` and `y` refines all four orthogonal access waveguides and the central
    intersection while grading back out at ratio `<= 1.4`.

    Parameters
    ----------
    cpw : float
        Background cells per wavelength in the slab effective index.
    n_feature_min : int or float
        Minimum target number of cells `N` across `min_feature`.
    min_feature : float, optional
        Smallest geometric feature width [m] (default `0.35e-6`).
    wavelength_min : float, optional
        Shortest wavelength in the simulation band [m] (default `1.26e-6`).
    neff_slab : float, optional
        2D EIM vertical slab effective index. Computed via `slab_neff` if None.

    Returns
    -------
    overrides : list of tuple
        List of `(lo, hi, max_cell)` refinement boxes [m], or `[]` when the
        background slab mesh already satisfies `h_slab <= max_cell`.
    """
    if neff_slab is None:
        neff_slab = slab_neff(
            n_si, n_sio2, h_si, wavelength_ref, polarization="TE"
        )
    f_max = c_light / float(wavelength_min)
    h_slab = meshing.cell_size(f_max, float(neff_slab) ** 2, float(cpw))
    max_cell = float(min_feature) / float(n_feature_min)
    if max_cell < h_slab * (1.0 - 1e-6):
        hw = 0.5 * float(min_feature)
        return [((-hw, -hw, -np.inf), (+hw, +hw, +np.inf), max_cell)]
    return []


def measure_grid_resolution(
    x_nodes,
    y_nodes,
    z_nodes=None,
    min_feature=0.35e-6,
    n_cells_z=1,
    has_ghost=False,
):
    """
    Compute actual mesh resolution metrics across `min_feature` from 1D node arrays.

    Parameters
    ----------
    x_nodes, y_nodes : array_like
        1D mesh node coordinates along `x` and `y` [m].
    z_nodes : array_like, optional
        1D mesh node coordinates along `z` [m].
    min_feature : float, optional
        Smallest geometric feature width [m] (default `0.35e-6`).
    n_cells_z : int, optional
        Number of active cells along `z` (`1` for 2D EIM).
    has_ghost : bool, optional
        True if `x_nodes` and `y_nodes` come from `GridFIT3D` (which appends one
        ghost node beyond `xmax`/`ymax`), False if they come directly from
        `Simulation._axis`.

    Returns
    -------
    metrics : dict
        Dictionary with `min_feature_nm`, `dx_background_nm`, `dx_min_nm`,
        `cells_across_feature` (measured as `min_feature / max_cell_across_feature`),
        `grid_shape`, and `n_cells`.
    """
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)
    if has_ghost:
        x_phys = x[:-1]
        y_phys = y[:-1]
        nx = len(x) - 1
        ny = len(y) - 1
    else:
        x_phys = x
        y_phys = y
        nx = len(x)
        ny = len(y)

    dx = np.diff(x_phys)
    dy = np.diff(y_phys)
    hw = 0.5 * float(min_feature)
    overlap_x = (x_phys[:-1] < hw - 1e-12) & (x_phys[1:] > -hw + 1e-12)
    overlap_y = (y_phys[:-1] < hw - 1e-12) & (y_phys[1:] > -hw + 1e-12)
    dx_feat = dx[overlap_x] if np.any(overlap_x) else dx
    dy_feat = dy[overlap_y] if np.any(overlap_y) else dy
    max_cell_across = max(float(np.max(dx_feat)), float(np.max(dy_feat)))

    dx_bg_nm = max(float(np.max(dx)), float(np.max(dy))) * 1e9
    dx_min_nm = min(float(np.min(dx)), float(np.min(dy))) * 1e9
    cells_across = float(min_feature) / max_cell_across
    nz = int(n_cells_z)
    return {
        "min_feature_nm": float(min_feature) * 1e9,
        "dx_background_nm": dx_bg_nm,
        "dx_min_nm": dx_min_nm,
        "cells_across_feature": cells_across,
        "grid_shape": [int(nx), int(ny), nz],
        "n_cells": int(nx * ny * nz),
    }


def inspect_task_mesh(
    task_cfg,
    build_crossing_fn=None,
    wl_min=1.26e-6,
    wl_max=1.36e-6,
    wl_ref=1.31e-6,
    n_si=3.67,
    n_sio2=1.45,
    h_si=0.161e-6,
    domain_2d=(-6.8e-6, 6.8e-6, -6.8e-6, 6.8e-6),
    min_feature=0.35e-6,
):
    """
    Build the exact 1D mesh axes `(x, y, z)` for `task_cfg` via `Simulation._axis`
    without constructing `SolverFIT3D` or running the solver, and return all
    resolution metrics plus CFL step estimates.
    """
    f_min = c_light / float(wl_max)
    f_max = c_light / float(wl_min)
    cpw = float(task_cfg["cpw"])
    n_feat = int(task_cfg.get("n_feature_min", 4))
    n_pml = int(task_cfg.get("n_pml", 8))

    neff_slab = slab_neff(n_si, n_sio2, h_si, wl_ref, polarization="TE")
    overrides = task_cfg.get("mesh_overrides")
    if overrides is None:
        overrides = crossing_mesh_overrides(
            cpw,
            n_feat,
            min_feature=min_feature,
            wavelength_min=wl_min,
            neff_slab=neff_slab,
        )

    dom2d = task_cfg.get("domain", domain_2d)
    xmin, xmax, ymin, ymax = dom2d[:4]
    h_core = meshing.cell_size(f_max, neff_slab**2, cpw)
    domain = (xmin, xmax, ymin, ymax, -0.5 * h_core, 0.5 * h_core)
    margin = 20.0 * float(wl_max)
    bg_box = Box(
        center=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.0),
        size=(xmax - xmin + margin, ymax - ymin + margin, 4.0 * h_core),
        material=n_to_eps(n_sio2),
    )
    structs_2d = [bg_box]
    if build_crossing_fn is not None:
        structures, _ = build_crossing_fn(kind=task_cfg.get("kind", "cosine"))
        for s in structures:
            sc = copy.deepcopy(s)
            sc.z_min, sc.z_max = -0.5 * h_core, 0.5 * h_core
            sc._lo[2], sc._hi[2] = -0.5 * h_core, 0.5 * h_core
            sc.material = n_to_eps(neff_slab)
            structs_2d.append(sc)
    else:
        structs_2d.append(
            Box(
                center=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.0),
                size=(xmax - xmin, ymax - ymin, h_core),
                material=n_to_eps(neff_slab),
            )
        )

    sim = Simulation(
        f_min,
        f_max,
        structures=structs_2d,
        boundaries=(["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"]),
        domain=domain,
        cells_per_wavelength=cpw,
        padding=0.8e-6,
        n_pml=n_pml,
        mesh_overrides=overrides,
        averaging=task_cfg.get("averaging", "kottke"),
    )
    if task_cfg.get("nonuniform", False):
        x, y, z = build_crossing_graded_axes(
            domain=dom2d[:4],
            w_in=min_feature,
            cpw=cpw,
            n_feature_min=n_feat,
            n_pml=n_pml,
        )
    else:
        x, y, z = [sim._axis(a) for a in range(3)]
    res = measure_grid_resolution(
        x, y, z, min_feature=min_feature, n_cells_z=1, has_ghost=False
    )
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    dt_s = 0.5 / (
        c_light
        * np.sqrt(
            1.0 / float(np.min(dx)) ** 2
            + 1.0 / float(np.min(dy)) ** 2
            + 1.0 / float(np.min(dz)) ** 2
        )
    )
    lx = float(x[-1] - x[0])
    ly = float(y[-1] - y[0])
    transit = np.hypot(lx, ly) * float(n_si) / c_light
    max_time = sim.pulse.end_time + 15.0 * transit
    res["dt_s"] = float(dt_s)
    res["n_steps_max"] = int(np.ceil(max_time / dt_s))
    res["mesh_overrides"] = overrides
    return res


def build_zoomed_2d_solver(
    structures,
    cpw,
    n_feature_min,
    averaging="kottke",
    xlim_um=(-6.45, -5.15),
    ylim_um=(-0.50, 0.50),
    n_pml=10,
    wl_min=1.26e-6,
    wl_max=1.36e-6,
    wl_ref=1.31e-6,
    n_si=3.67,
    n_sio2=1.45,
    h_si=0.161e-6,
    domain_2d=(-6.85e-6, 6.85e-6, -6.85e-6, 6.85e-6),
    min_feature=0.35e-6,
):
    """
    Construct the exact full-domain 1D mesh axes `(x, y, z)` via `Simulation._axis`
    for `(cpw, n_feature_min)`, crop `(x, y)` to the zoomed window `(xlim_um, ylim_um)`
    while preserving every node position and cell size across `min_feature`, and
    evaluate `assign_materials` on that subgrid. Avoids allocating a full-domain
    `1295 x 1295 x 1` `SolverFIT3D` when rendering `04_permittivity_mesh_zoom.png`.
    """
    from wakis.gridFIT3D import GridFIT3D
    from wakis.solverFIT3D import SolverFIT3D
    from wakis.templates.shapes import assign_materials

    f_min = c_light / float(wl_max)
    f_max = c_light / float(wl_min)
    neff_slab = slab_neff(n_si, n_sio2, h_si, wl_ref, polarization="TE")
    overrides = crossing_mesh_overrides(
        cpw,
        n_feature_min,
        min_feature=min_feature,
        wavelength_min=wl_min,
        neff_slab=neff_slab,
    )
    xmin, xmax, ymin, ymax = domain_2d[:4]
    h_core = meshing.cell_size(f_max, neff_slab**2, float(cpw))
    domain = (xmin, xmax, ymin, ymax, -0.5 * h_core, 0.5 * h_core)
    margin = 20.0 * float(wl_max)
    bg_box = Box(
        center=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.0),
        size=(xmax - xmin + margin, ymax - ymin + margin, 4.0 * h_core),
        material=n_to_eps(n_sio2),
    )
    structs_2d = [bg_box]
    for s in structures:
        sc = copy.deepcopy(s)
        sc.z_min, sc.z_max = -0.5 * h_core, 0.5 * h_core
        sc._lo[2], sc._hi[2] = -0.5 * h_core, 0.5 * h_core
        sc.material = n_to_eps(neff_slab)
        structs_2d.append(sc)

    sim = Simulation(
        f_min,
        f_max,
        structures=structs_2d,
        boundaries=(["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"]),
        domain=domain,
        cells_per_wavelength=float(cpw),
        padding=0.8e-6,
        n_pml=int(n_pml),
        mesh_overrides=overrides,
        averaging=averaging,
    )
    x_full, y_full, z_full = [sim._axis(a) for a in range(3)]
    x_sub = x_full[(x_full >= xlim_um[0] * 1e-6) & (x_full <= xlim_um[1] * 1e-6)]
    y_sub = y_full[(y_full >= ylim_um[0] * 1e-6) & (y_full <= ylim_um[1] * 1e-6)]
    grid = GridFIT3D(x=x_sub, y=y_sub, z=z_full, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=sim.bc_low,
        bc_high=sim.bc_high,
        n_pml=1,
        cfln=sim.cfln,
        dtype=sim.dtype,
        verbose=0,
    )
    assign_materials(solver, structs_2d, sim.f_center, averaging=averaging)
    return solver


def _draw_mesh_zoom_panel(
    ax,
    fig,
    solver,
    xlim_um,
    ylim_um,
    title,
    min_feature_um=0.35,
    component="y",
    annotate_feature=False,
):
    g = solver.grid
    iz = solver.Nz // 2
    ieps = solver.ieps[:, :, iz, component]
    if hasattr(ieps, "get"):
        ieps = ieps.get()
    eps_r = np.where(
        ieps > 0, 1.0 / (np.where(ieps > 0, ieps, 1.0) * epsilon_0), 1.0
    )
    x_um, y_um = g.x * 1e6, g.y * 1e6
    pcm = ax.pcolormesh(x_um, y_um, eps_r.T, cmap=figstyle.CMAP_EPS, shading="flat")
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
    if annotate_feature:
        hw = 0.5 * float(min_feature_um)
        ax.axhline(-hw, color="#ffae00", ls="--", lw=1.4)
        ax.axhline(+hw, color="#ffae00", ls="--", lw=1.4)
        x_arr = xl[0] + 0.32 * (xl[1] - xl[0])
        ax.annotate(
            "",
            xy=(x_arr, +hw),
            xytext=(x_arr, -hw),
            arrowprops=dict(arrowstyle="<->", color="#ffae00", lw=1.8),
        )
        m = measure_grid_resolution(
            g.x, g.y, min_feature=min_feature_um * 1e-6, has_ghost=True
        )
        ax.text(
            x_arr + 0.03 * (xl[1] - xl[0]),
            0.0,
            f"$w_{{\\mathrm{{in}}}} = {min_feature_um * 1e3:.0f}$ nm\n"
            f"{m['cells_across_feature']:.1f} cells ($\\Delta y = {m['dx_min_nm']:.1f}$ nm)",
            color="white",
            fontsize=8.0,
            fontweight="bold",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.22",
                fc="#16324f",
                ec="#ffae00",
                alpha=0.88,
            ),
        )
    ax.set(aspect="equal", xlabel="x [µm]", ylabel="y [µm]")
    ax.set_title(title)
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(rf"$\varepsilon_{{r,{component}}}$")


def plot_feature_and_subpixel_mesh(
    solver_l1_kottke,
    solver_ref_kottke,
    solver_l1_stair,
    cpw_l1,
    n_l1,
    cpw_ref,
    n_ref,
    min_feature_um=0.35,
    component="y",
    out_path=None,
    out_path_taper=None,
):
    """
    Render ``04_permittivity_mesh_zoom.png`` (2 panels <= 3: smallest feature
    ``w_in = 350 nm`` at L1 and Reference) and optionally
    ``04c_taper_subpixel_smoothing.png`` (2 panels <= 3: Staircase vs Kottke on
    the curved cosine taper at L1).
    """
    figstyle.apply_style()
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.4)
    feat_xlim = (-6.35, -5.25)
    feat_ylim = (-0.42, 0.42)
    _draw_mesh_zoom_panel(
        ax0,
        fig,
        solver_l1_kottke,
        feat_xlim,
        feat_ylim,
        f"(a) Smallest feature @ L1 (cpw={int(cpw_l1)}, N≥{int(n_l1)})",
        min_feature_um=min_feature_um,
        component=component,
        annotate_feature=True,
    )
    _draw_mesh_zoom_panel(
        ax1,
        fig,
        solver_ref_kottke,
        feat_xlim,
        feat_ylim,
        f"(b) Smallest feature @ Reference (cpw={int(cpw_ref)}, N≥{int(n_ref)})",
        min_feature_um=min_feature_um,
        component=component,
        annotate_feature=True,
    )
    if out_path is not None:
        figstyle.save(fig, out_path)
        plt.close(fig)

    if out_path_taper is None and out_path is not None:
        out_path_taper = Path(out_path).parent / "04c_taper_subpixel_smoothing.png"

    if out_path_taper is not None and solver_l1_stair is not None:
        fig_t, (axt0, axt1) = figstyle.new_figure(
            nrows=1, ncols=2, width=11.6, height=4.4
        )
        taper_xlim = (-3.2, 1.2)
        taper_ylim = (-1.1, 1.1)
        _draw_mesh_zoom_panel(
            axt0,
            fig_t,
            solver_l1_stair,
            taper_xlim,
            taper_ylim,
            f"(a) Staircase on cosine taper (L1, cpw={int(cpw_l1)})",
            min_feature_um=min_feature_um,
            component=component,
            annotate_feature=False,
        )
        _draw_mesh_zoom_panel(
            axt1,
            fig_t,
            solver_l1_kottke,
            taper_xlim,
            taper_ylim,
            f"(b) Kottke subpixel on cosine taper (L1, cpw={int(cpw_l1)})",
            min_feature_um=min_feature_um,
            component=component,
            annotate_feature=False,
        )
        figstyle.save(fig_t, out_path_taper)
        plt.close(fig_t)

    return fig, (ax0, ax1)


def plot_convergence_ladder(
    cpws,
    cells_across,
    err_s,
    err_k,
    rs_vals,
    rk_vals,
    ref_cpw,
    ref_cells_across,
    ref_r_mean,
    level_labels=None,
    nonuniform_point=None,
    out_path=None,
):
    """
    Render ``15_kottke_vs_staircase_convergence.png`` (2 panels <= 3) against
    the independent Kottke reference ``T_ref``, labeling every point with its
    ``cells_across_feature`` and optionally plotting the graded non-uniform
    mesh point ``nonuniform_point``.
    """
    from matplotlib.ticker import NullFormatter

    figstyle.apply_style()
    res = np.asarray(cpws, dtype=float)
    ca = np.asarray(cells_across, dtype=float)
    err_s = np.asarray(err_s, dtype=float)
    err_k = np.asarray(err_k, dtype=float)
    rs_vals = np.asarray(rs_vals, dtype=float)
    rk_vals = np.asarray(rk_vals, dtype=float)
    if level_labels is None:
        level_labels = [f"L{i + 1}" for i in range(len(res))]

    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.6)

    ax0.loglog(
        res,
        err_s,
        "s-",
        color="#d95f02",
        lw=2.0,
        ms=6.5,
        label=r"Staircase RMS $|T - T_{\mathrm{ref}}|$",
    )
    ax0.loglog(
        res,
        err_k,
        "o-",
        color="#1b9e77",
        lw=2.0,
        ms=6.5,
        label=r"Kottke RMS $|T - T_{\mathrm{ref}}|$",
    )
    guide_1 = err_s[0] * (res[0] / res) ** 1
    guide_2 = err_k[0] * (res[0] / res) ** 2
    ax0.loglog(
        res,
        guide_1,
        "--",
        color="#d95f02",
        alpha=0.55,
        lw=1.2,
        label=r"$\mathcal{O}(\Delta x^1)$ guide",
    )
    ax0.loglog(
        res,
        guide_2,
        ":",
        color="#1b9e77",
        alpha=0.65,
        lw=1.4,
        label=r"$\mathcal{O}(\Delta x^2)$ guide",
    )

    for lbl, r_val, c_val, ek, es in zip(level_labels, res, ca, err_k, err_s):
        ax0.annotate(
            f"{lbl}: {c_val:.1f} cells\n({int(r_val)} cpw)",
            xy=(r_val, ek),
            xytext=(0, -24),
            textcoords="offset points",
            ha="center",
            fontsize=7.4,
            color="#1b9e77",
            fontweight="bold",
        )
        ax0.annotate(
            f"{lbl}: {c_val:.1f} cells",
            xy=(r_val, es),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
            color="#d95f02",
        )

    if nonuniform_point is not None:
        ax0.loglog(
            [nonuniform_point["cpw"]],
            [nonuniform_point["err_rms"]],
            "*",
            color="#e7298a",
            ms=11.0,
            zorder=9,
            label=f"Graded non-uniform Kottke ({nonuniform_point['n_cells']:,} cells)",
        )
        ax1.loglog(
            [nonuniform_point["cpw"]],
            [nonuniform_point["r_mean"]],
            "*",
            color="#e7298a",
            ms=11.0,
            zorder=9,
            label=f"Graded non-uniform Kottke ({nonuniform_point['n_cells']:,} cells)",
        )

    ax0.set(
        xlabel="Background resolution [cpw] & cells across w_in = 350 nm",
        ylabel=r"Band RMS error $\|T(\lambda) - T_{\mathrm{ref}}(\lambda)\|_{\mathrm{RMS}}$",
        title=(
            f"(a) Convergence vs Kottke ref "
            f"(cpw={int(ref_cpw)}, {ref_cells_across:.1f} cells)"
        ),
    )
    ax0.set_xlim(res[0] * 0.85, res[-1] * 1.22)
    ax0.set_ylim(
        min(float(np.min(err_k)), float(np.min(guide_2))) * 0.65,
        max(float(np.max(err_s)), float(np.max(err_k))) * 1.38,
    )
    ax0.set_xticks(res)
    ax0.set_xticklabels(
        [f"{int(r)} cpw\n({c:.1f} cells, {lbl})" for r, c, lbl in zip(res, ca, level_labels)]
    )
    ax0.xaxis.set_minor_formatter(NullFormatter())
    ax0.grid(True, which="both", ls=":", alpha=0.4)
    ax0.legend(loc="lower left", fontsize=7.2)

    ax1.loglog(
        res,
        rs_vals,
        "s-",
        color="#d95f02",
        lw=2.0,
        ms=6.5,
        label=r"Staircase mean $|S_{11}|^2$",
    )
    ax1.loglog(
        res,
        rk_vals,
        "o-",
        color="#1b9e77",
        lw=2.0,
        ms=6.5,
        label=r"Kottke mean $|S_{11}|^2$",
    )
    ax1.axhline(
        ref_r_mean,
        color="#2b5c8f",
        ls=":",
        lw=1.5,
        label=f"Ref mean $|S_{{11}}|^2$ (cpw={int(ref_cpw)}, {ref_cells_across:.1f} cells)",
    )
    for lbl, r_val, c_val, rk, rs in zip(level_labels, res, ca, rk_vals, rs_vals):
        ax1.annotate(
            f"{lbl} ({c_val:.1f} cells)",
            xy=(r_val, rk),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=7.3,
            color="#1b9e77",
        )
        ax1.annotate(
            f"{lbl} ({c_val:.1f} cells)",
            xy=(r_val, rs),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
            color="#d95f02",
        )

    ax1.set(
        xlabel="Background resolution [cpw] & cells across w_in = 350 nm",
        ylabel=r"Band-averaged reflection $\langle |S_{11}|^2 \rangle$",
        title="(b) Spurious sidewall reflection vs resolution",
    )
    ax1.set_xlim(res[0] * 0.85, res[-1] * 1.22)
    ax1.set_ylim(
        min(float(np.min(rk_vals)), float(ref_r_mean)) * 0.65,
        max(float(np.max(rs_vals)), float(np.max(rk_vals))) * 1.38,
    )
    ax1.set_xticks(res)
    ax1.set_xticklabels(
        [f"{int(r)} cpw\n({c:.1f} cells, {lbl})" for r, c, lbl in zip(res, ca, level_labels)]
    )
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(loc="center right", fontsize=7.2)

    if out_path is not None:
        figstyle.save(fig, out_path)
        plt.close(fig)
    return fig, (ax0, ax1)


def _grade_positive_span(u_start: float, u_end: float, du_start: float, du_max: float, ratio: float = 1.045):
    """Build monotonically increasing node coordinates on ``[u_start, u_end]`` grading from ``du_start`` to ``du_max``."""
    span = float(u_end - u_start)
    if span <= 1e-15:
        return np.array([u_start], dtype=float)
    steps = []
    cur = float(du_start)
    rem = span
    while rem > 1e-12:
        if rem <= 1.25 * cur:
            steps.append(rem)
            break
        steps.append(cur)
        rem -= cur
        cur = min(cur * ratio, float(du_max))
    arr = np.asarray(steps, dtype=float)
    arr *= span / float(np.sum(arr))
    return float(u_start) + np.concatenate([[0.0], np.cumsum(arr)])


def build_crossing_graded_axes(
    domain=(-6.85e-6, 6.85e-6, -6.85e-6, 6.85e-6),
    w_in: float = 0.35e-6,
    w_cross: float = 1.10e-6,
    l_taper: float = 5.30e-6,
    cpw: float = 20.0,
    n_feature_min: int = 8,
    n_pml: int = 10,
    du_fine: float | None = None,
    du_taper_max: float | None = None,
    du_clad_max: float | None = None,
):
    """
    Construct exact $D_4$-symmetric graded non-uniform Cartesian node arrays
    ``(x, y, z)`` (`x == y`, `x == -x[::-1]`) for the 4-port O-band crossing:

    - Fine uniform spacing ``du_fine = 21.5 nm`` (`16.3` cells across ``w_in = 350 nm``)
      throughout the central crossing square and perpendicular waveguide core/bulge
      ``|u| <= 0.80 um`` (`> w_m = 0.75 um` and `w_out / 2 = 0.55 um`),
    - Smooth geometric grading across the cosine-tapered arms
      ``|u| in [0.80, 5.85] um`` (`du <= du_taper_max = 35.0 nm`) and outer
      straight port leads / CPML ``|u| > 5.85 um`` (`du <= du_clad_max = 42.0 nm`).
      Along the horizontal waveguide arm (`|y| <= 0.75 um`, `|x| > 0.80 um`),
      cells are non-square (`dx > dy`) along the propagation axis, while in the
      4 outer ``SiO2`` cladding quadrants (`|x|, |y| > 0.80 um`, covering 78%
      of the domain area) both ``dx`` and ``dy`` are coarse.
    """
    scale = 20.0 / max(float(cpw), 8.0)
    if du_fine is None:
        du_fine = min(float(w_in) / max(float(n_feature_min), 8.0), 21.5e-9 * scale)
    if du_taper_max is None:
        du_taper_max = 35.0e-9 * scale
    if du_clad_max is None:
        du_clad_max = 42.0e-9 * scale

    u_core_half = max(0.5 * float(w_cross), 0.75e-6) + 0.05e-6  # 0.80 um
    u_taper_end = 0.5 * float(w_cross) + float(l_taper)  # 5.85 um
    u_dom = max(abs(float(domain[0])), abs(float(domain[1])))
    u_outer = u_dom + float(n_pml) * du_clad_max

    n_core_half = max(int(np.ceil(u_core_half / du_fine)), 4)
    u_seg1 = np.linspace(0.0, u_core_half, n_core_half + 1)
    du_actual = float(u_seg1[1] - u_seg1[0])

    u_seg2 = _grade_positive_span(
        u_core_half, u_taper_end, du_actual, du_taper_max, ratio=1.042
    )
    du_end2 = float(u_seg2[-1] - u_seg2[-2])
    u_seg3 = _grade_positive_span(
        u_taper_end, u_outer, du_end2, du_clad_max, ratio=1.045
    )

    u_pos = np.concatenate([u_seg1[:-1], u_seg2[:-1], u_seg3])
    u_full = np.concatenate([-u_pos[:0:-1], u_pos])
    if len(domain) >= 6:
        z_nodes = np.array([float(domain[4]), float(domain[5])], dtype=float)
    else:
        z_nodes = np.array([-0.5 * du_actual, 0.5 * du_actual], dtype=float)
    return u_full, u_full.copy(), z_nodes


class NonUniformCrossingPIC(PICSimulation):
    """
    `PICSimulation` subclass that builds the 2D EIM 4-port waveguide crossing on
    the exact $D_4$-symmetric graded non-uniform Cartesian mesh from
    `build_crossing_graded_axes`.
    """

    def __init__(
        self,
        wavelength_min: float,
        wavelength_max: float,
        w_in: float = 0.35e-6,
        w_cross: float = 1.10e-6,
        l_taper: float = 5.30e-6,
        n_feature_min: int = 8,
        **kwargs,
    ):
        self._nu_w_in = float(w_in)
        self._nu_w_cross = float(w_cross)
        self._nu_l_taper = float(l_taper)
        self._nu_n_feature = int(n_feature_min)
        kwargs["mesh_overrides"] = ()
        super().__init__(wavelength_min, wavelength_max, **kwargs)

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
        axes_xyz = build_crossing_graded_axes(
            domain=domain,
            w_in=self._nu_w_in,
            w_cross=self._nu_w_cross,
            l_taper=self._nu_l_taper,
            cpw=self.cells_per_wavelength,
            n_feature_min=self._nu_n_feature,
            n_pml=self.sim.n_pml,
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


def _build_crossing_port_sources_and_monitors(pic, include_flux_box: bool = True):
    """Build ``(sources, monitors)`` dicts for ``wakis.schematic`` from a 4-port crossing ``pic``."""
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
        cx, cy, _ = (float(v) for v in pt.center)
        sx, sy, _ = (float(v) for v in pt.size)
        if pt.name == "west":
            hy = min(0.5 * sy, 1.4e-6)
            x_src = cx - 0.08e-6
            sources_s.append(
                {
                    "kind": "sheet",
                    "type_name": "ModeSource",
                    "label": "Port 'west' ModeSource\n(TE₀, +x, 1260–1360 nm)",
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
                    "label": "West (S₁₁)",
                    "h_bounds": (cx, cx),
                    "v_bounds": (-hy, +hy),
                }
            )
        elif pt.name == "east":
            hy = min(0.5 * sy, 1.4e-6)
            monitors_s.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "label": "Port 'east' ModeMonitor\n(Thru S₂₁)",
                    "h_bounds": (cx, cx),
                    "v_bounds": (-hy, +hy),
                }
            )
        elif pt.name == "north":
            hx = min(0.5 * sx, 1.4e-6)
            monitors_s.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "label": "Port 'north' (Cross S₃₁)",
                    "h_bounds": (-hx, +hx),
                    "v_bounds": (cy, cy),
                }
            )
        elif pt.name == "south":
            hx = min(0.5 * sx, 1.4e-6)
            monitors_s.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "label": "Port 'south' (Cross S₄₁)",
                    "h_bounds": (-hx, +hx),
                    "v_bounds": (cy, cy),
                }
            )
    if include_flux_box:
        monitors_s.append(
            {
                "kind": "box",
                "type_name": "FluxMonitor",
                "label": "4-wall Poynting FluxMonitor box",
                "h_bounds": (xmin + 2.0 * dx_bg, xmax - 2.0 * dx_bg),
                "v_bounds": (ymin + 2.0 * dy_bg, ymax - 2.0 * dy_bg),
            }
        )
    return sources_s, monitors_s


def _draw_crossing_polyslab(
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
            color="#0f172a",
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


def plot_setup_schematic_crossing(
    pic,
    primary_data: dict,
    out_path: str | Path,
):
    """
    Render ``00_setup_schematic.png`` (first figure of Tutorial 5.5, 2 panels):
    - Panel (a): Labelled 2D EIM setup schematic of the $D_4$-symmetric 4-port
      O-band cosine-tapered waveguide crossing with domain extents, 10-cell CPML,
      periodic $\\pm z$, Si/SiO2 indices, West ``ModeSource`` + launched $\\mathrm{TE}_0$
      mode profile, East/North/South ``ModeMonitor``s, and 4-wall Poynting
      ``FluxMonitor`` box.
    - Panel (b): ``1x`` Yee mesh zoom on the West access waveguide
      (``w_in = 350 nm``) and cosine taper entrance showing ``16.3`` cells
      across the waveguide width (``dy_min = 21.5 nm``).
    """
    figstyle.apply_style()
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=13.6, height=5.8)

    sources_s, monitors_s = _build_crossing_port_sources_and_monitors(
        pic, include_flux_box=True
    )
    neff_s = float(pic.n_eff_slab)
    eps_s = neff_s**2
    eps_clad = float(n_to_eps(pic.n_clad))
    mat_names = {
        "background": f"SiO₂ cladding (n={pic.n_clad:.2f}, ε_r={eps_clad:.2f})",
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
        wavelength=1310e-9,
        title="(a) 4-port O-band cosine-tapered crossing setup (D₄ symmetry)",
    )
    _draw_crossing_polyslab(
        ax0,
        pic.structures_3d,
        facecolor="#2563eb",
        alpha=0.28,
        edgecolor="#0f172a",
        lw=1.35,
        label=(
            f"Si EIM core (h_Si=161 nm, n_Si=3.67)\n"
            f"n_eff,slab={neff_s:.4f} (ε_r={eps_s:.2f})\n"
            "w_in=0.35 µm → 2w_m=1.50 µm → w_out=1.10 µm (l_t=5.30 µm)"
        ),
        label_xy=(3.3, 2.3),
    )
    ax0.text(
        -3.4,
        -3.9,
        "SiO₂ cladding (n = 1.45, ε_r = 2.10)",
        fontsize=7.2,
        color="#334155",
        ha="center",
        va="center",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#94a3b8", alpha=0.90),
    )
    port_y_key = "port_y_in" if "port_y_in" in primary_data else "port_y"
    port_ey_key = "port_ey_in" if "port_ey_in" in primary_data else "port_ey"
    if port_y_key in primary_data and port_ey_key in primary_data:
        yp = np.asarray(primary_data[port_y_key]) * 1e6
        ep = np.abs(np.asarray(primary_data[port_ey_key]))
        ep = ep / (float(np.max(ep)) or 1.0)
        m = np.abs(yp) <= 1.25
        yw = yp[m]
        ew = ep[m]
        xw = float(pic.ports[0].center[0]) * 1e6
        xe = float(pic.ports[1].center[0]) * 1e6
        ax0.fill_betweenx(yw, xw, xw + 0.95 * ew, color="#d62728", alpha=0.24, zorder=8)
        ax0.plot(
            xw + 0.95 * ew,
            yw,
            color="#d62728",
            lw=1.7,
            zorder=9,
            label=r"Launched West $\mathrm{TE}_0$ $|E_y(y)|$",
        )
        ax0.fill_betweenx(yw, xe - 0.95 * ew, xe, color="#1b9e77", alpha=0.24, zorder=8)
        ax0.plot(
            xe - 0.95 * ew,
            yw,
            color="#1b9e77",
            lw=1.7,
            zorder=9,
            label=r"Thru East $\mathrm{TE}_0$ $|E_y(y)|$ ($S_{21}$)",
        )
        ax0.legend(
            loc="lower right",
            bbox_to_anchor=(0.98, 0.14),
            fontsize=7.0,
            framealpha=0.92,
        )

    # Panel (b): 1x mesh zoom on West access waveguide & cosine taper start
    xl_z, yl_z = (-6.00, -4.60), (-0.52, 0.52)
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
    eps_r = np.where(
        ieps_y > 0, 1.0 / (np.where(ieps_y > 0, ieps_y, 1.0) * epsilon_0), 1.0
    )
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
    _draw_crossing_polyslab(
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
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(b) 1× mesh zoom on w_in = 350 nm access & taper start",
    )
    cb = fig.colorbar(im_m, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label(r"Subpixel $\varepsilon_{yy} / \varepsilon_0$ (Kottke)")
    m_res = measure_grid_resolution(
        pic.grid.x, pic.grid.y, min_feature=0.35e-6, has_ghost=True
    )
    ax1.annotate(
        "",
        xy=(-5.75, -0.175),
        xytext=(-5.75, +0.175),
        arrowprops=dict(arrowstyle="<->", color="white", lw=1.5),
    )
    ax1.text(
        -5.65,
        0.0,
        f"w_in = 350 nm\n{m_res['cells_across_feature']:.1f} cells (Δy_min = {m_res['dx_min_nm']:.1f} nm)",
        color="white",
        fontsize=7.3,
        fontweight="bold",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="#0f172a", ec="#facc15", alpha=0.88),
    )
    figstyle.save(fig, out_path)
    plt.close(fig)


def plot_crossing_geometries_schematic(
    pic_linear,
    pic_plain,
    out_path: str | Path,
):
    """
    Render ``00b_crossing_geometries_schematic.png`` (2 panels <= 3) showing the
    labelled setup schematics of the two comparison crossing geometries:
    - Panel (a): Linear-tapered 4-port crossing (``w_in = 0.35 -> w_cross = 1.45 um``,
      ``L_t = 4.80 um``).
    - Panel (b): Plain untapered 4-port waveguide crossing (``w = 0.35 um``).
    """
    figstyle.apply_style()
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=13.2, height=5.5)

    neff_s = float(pic_linear.n_eff_slab)
    eps_clad = float(n_to_eps(pic_linear.n_clad))
    mat_names = {
        "background": f"SiO₂ cladding (n={pic_linear.n_clad:.2f})",
    }
    for ax_i, pic_i, ttl_i, lbl_i in (
        (
            ax0,
            pic_linear,
            "(a) Linear-tapered crossing setup (l_t = 5.30 µm)",
            f"Si linear-tapered crossing\nw: 0.35 → 1.10 µm (n_eff={neff_s:.2f})",
        ),
        (
            ax1,
            pic_plain,
            "(b) Plain untapered crossing setup (w = 0.35 µm)",
            f"Si untapered crossing\nw = 0.35 µm (n_eff={neff_s:.2f})",
        ),
    ):
        src_i, mon_i = _build_crossing_port_sources_and_monitors(
            pic_i, include_flux_box=False
        )
        schematic.plot_schematic(
            _SchematicSolverProxy(pic_i.solver, eps_clad),
            plane="xy",
            ax=ax_i,
            unit="µm",
            structures=(),
            material_names=mat_names,
            sources=src_i,
            monitors=mon_i,
            show_mesh=True,
            mesh_zoom=False,
            show_pulse_inset=False,
            show_wave_packet=False,
            wavelength=1310e-9,
            title=ttl_i,
        )
        _draw_crossing_polyslab(
            ax_i,
            pic_i.structures_3d,
            facecolor="#2563eb",
            alpha=0.28,
            edgecolor="#0f172a",
            lw=1.3,
            label=lbl_i,
            label_xy=(3.3, -3.5),
        )

    figstyle.save(fig, out_path)
    plt.close(fig)


def plot_mesh_upgrade_comparison_crossing(
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
    Render ``04b_mesh_upgrade_comparison.png`` (3 panels <= 3) comparing the
    uniform-background ``kottke_L3`` baseline against the $D_4$-symmetric graded
    non-uniform ``kottke_nonuniform`` mesh on the O-band cosine-tapered crossing:
    - Panel (a): Split-half mesh view (``y >= 0``: ``kottke_L3``, ``y <= 0``:
      ``kottke_nonuniform``).
    - Panel (b): 1D cell spacing ``du(u)`` [nm] vs ``u`` [µm] showing ``21.5 nm``
      across the central intersection/waveguide core and smooth grading to
      ``42 nm`` in the outer cladding quadrants.
    - Panel (c): O-band thru transmission ``T(lambda)`` [%] and insertion loss
      comparing ``kottke_L3``, ``kottke_nonuniform``, and ``kottke_reference``.
    """
    figstyle.apply_style()
    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=15.4, height=4.7)

    # Panel (a): Split-half mesh view (y >= 0: kottke_L3, y <= 0: kottke_nonuniform)
    iz = pic_nonuniform.solver.Nz // 2
    ieps = np.asarray(pic_nonuniform.solver.ieps[:, :, iz, "y"])
    eps_r = np.where(ieps > 0, 1.0 / (np.where(ieps > 0, ieps, 1.0) * epsilon_0), 1.0)
    xu_nu = pic_nonuniform.grid.x * 1e6
    yu_nu = pic_nonuniform.grid.y * 1e6
    ax0.pcolormesh(
        xu_nu, yu_nu, eps_r.T, cmap=figstyle.CMAP_EPS, shading="flat", alpha=0.88
    )

    xl_m, yl_m = (-4.50, 4.50), (-4.50, 4.50)
    xu_u = pic_uniform.grid.x * 1e6
    yu_u = pic_uniform.grid.y * 1e6
    xv_u = xu_u[(xu_u >= xl_m[0]) & (xu_u <= xl_m[1])][::2]
    yv_u = yu_u[(yu_u >= 0.0) & (yu_u <= yl_m[1])][::2]
    segs_u = [[(x, 0.0), (x, yl_m[1])] for x in xv_u] + [
        [(xl_m[0], y), (xl_m[1], y)] for y in yv_u
    ]
    ax0.add_collection(
        LineCollection(segs_u, colors="#08519c", linewidths=0.38, alpha=0.65)
    )

    xv_nu = xu_nu[(xu_nu >= xl_m[0]) & (xu_nu <= xl_m[1])][::2]
    yv_nu = yu_nu[(yu_nu >= yl_m[0]) & (yu_nu <= 0.0)][::2]
    segs_nu = [[(x, yl_m[0]), (x, 0.0)] for x in xv_nu] + [
        [(xl_m[0], y), (xl_m[1], y)] for y in yv_nu
    ]
    ax0.add_collection(
        LineCollection(segs_nu, colors="#d95f02", linewidths=0.42, alpha=0.78)
    )
    ax0.axhline(0.0, color="black", lw=1.1, ls="--")
    _draw_crossing_polyslab(
        ax0,
        pic_nonuniform.structures_3d,
        facecolor=None,
        edgecolor="black",
        lw=1.1,
    )

    n_u = int(pic_uniform.grid.Nx * pic_uniform.grid.Ny)
    n_nu = int(pic_nonuniform.grid.Nx * pic_nonuniform.grid.Ny)
    t_u_s = float(meta_uniform["elapsed_s"]) if meta_uniform else 1061.10
    t_nu_s = float(meta_nonuniform["elapsed_s"]) if meta_nonuniform else 0.0
    red_cells = 100.0 * (1.0 - n_nu / n_u)
    red_time = 100.0 * (1.0 - t_nu_s / t_u_s) if t_u_s > 0 else 0.0

    ax0.text(
        -4.30,
        3.55,
        f"Upper (y > 0): kottke_L3 baseline\n"
        f"{pic_uniform.grid.Nx}×{pic_uniform.grid.Ny} = {n_u:,} cells ({t_u_s:.1f} s)",
        fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#08519c", alpha=0.92),
    )
    ax0.text(
        -4.30,
        -4.20,
        f"Lower (y < 0): kottke_nonuniform\n"
        f"{pic_nonuniform.grid.Nx}×{pic_nonuniform.grid.Ny} = {n_nu:,} cells "
        f"(-{red_cells:.1f}%, {t_nu_s:.1f} s, -{red_time:.1f}%)",
        fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d95f02", alpha=0.92),
    )
    ax0.set(
        xlim=xl_m,
        ylim=yl_m,
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(a) Baseline (top) vs D₄ graded mesh (bottom, 1/2 lines)",
    )

    # Panel (b): 1D cell spacing du(u) [nm] vs u [um]
    uc_u = 0.5 * (xu_u[:-1] + xu_u[1:])
    du_u = np.diff(xu_u) * 1e3
    uc_nu = 0.5 * (xu_nu[:-1] + xu_nu[1:])
    du_nu = np.diff(xu_nu) * 1e3
    ax1.axvspan(
        -0.80,
        +0.80,
        color="#fdae61",
        alpha=0.26,
        label="Core & bulge (|u| ≤ 0.80 µm)",
    )
    ax1.axvspan(
        -5.85,
        -0.80,
        color="#2b5c8f",
        alpha=0.12,
        label="Cosine taper (|u| ∈ [0.80, 5.85] µm)",
    )
    ax1.axvspan(+0.80, +5.85, color="#2b5c8f", alpha=0.12)
    ax1.step(
        uc_u,
        du_u,
        where="mid",
        color="#08519c",
        lw=1.8,
        label=f"kottke_L3 ({pic_uniform.grid.Nx} cells/axis)",
    )
    ax1.step(
        uc_nu,
        du_nu,
        where="mid",
        color="#d95f02",
        lw=2.0,
        label=f"kottke_nonuniform ({pic_nonuniform.grid.Nx} cells/axis)",
    )
    ax1.set(
        xlim=(-7.2, 7.2),
        xlabel="Coordinate u ∈ {x, y} [µm]",
        ylabel="Cell spacing Δu(u) [nm]",
        title="(b) D₄-symmetric Δu(u) grading (21.5 nm in core)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper center", fontsize=7.0)

    # Panel (c): O-band thru transmission T(lambda) [%] comparison
    def _spec(d):
        w = d["wavelengths"] * 1e9
        o = np.argsort(w)
        ws = w[o]
        m = (ws >= 1265.0) & (ws <= 1355.0)
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
            label=f"kottke_reference (R: T₁₃₁₀={t0_r:.2f}%)",
        )
    ax2.plot(
        w_u,
        t_u,
        "--",
        color="#08519c",
        lw=1.9,
        label=f"kottke_L3 ({n_u:,} cells: T₁₃₁₀={t0_u:.2f}%)",
    )
    ax2.plot(
        w_nu,
        t_nu,
        "-.",
        color="#d95f02",
        lw=2.0,
        label=f"kottke_nonuniform ({n_nu:,} cells: T₁₃₁₀={t0_nu:.2f}%)",
    )
    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Thru transmission $|S_{21}|^2$ [%]",
        ylim=(93.0, 99.5),
        title="(c) Equal-accuracy O-band thru transmission",
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=7.1)

    figstyle.save(fig, out_path)
    plt.close(fig)


def overlay_crossing_geometry(ax, pic, outline_color: str = "white"):
    """Overlay material outlines/labels, ports, and CPML boundaries on a 2D crossing field map."""
    neff_s = float(pic.n_eff_slab)
    eps_clad = float(n_to_eps(pic.n_clad))
    mat_names = {
        "background": f"SiO₂ (n={pic.n_clad:.2f})",
    }
    sources_s, monitors_s = _build_crossing_port_sources_and_monitors(
        pic, include_flux_box=False
    )
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
    _draw_crossing_polyslab(
        ax,
        pic.structures_3d,
        facecolor=None,
        edgecolor=outline_color,
        lw=1.15,
        label=f"Si EIM core (n_eff={neff_s:.2f}) | SiO₂ clad (n={pic.n_clad:.2f})",
        label_xy=(-3.2, -3.5),
    )


def render_crossing_field_map_with_overlay(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    pic,
    kind: str,
    title: str,
    out_path: str | Path,
):
    """Render a single-panel 2D crossing field map with ``figstyle`` and ``overlay_geometry``."""
    figstyle.apply_style()
    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=7.2, height=6.4)
    x_um = np.asarray(x) * 1e6
    y_um = np.asarray(y) * 1e6
    if kind == "intensity_log":
        db_floor = -40.0
        f_abs = np.asarray(field, dtype=float)
        data_db = 10.0 * np.log10(
            np.maximum(
                f_abs / (float(np.max(f_abs)) or 1.0), 10.0 ** (db_floor / 10.0)
            )
        )
        im = ax.pcolormesh(
            x_um,
            y_um,
            data_db.T,
            cmap="inferno",
            vmin=db_floor,
            vmax=0.0,
            shading="auto",
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

    overlay_crossing_geometry(ax, pic, outline_color=outline_col)
    ax.set(
        xlim=(float(x_um.min()), float(x_um.max())),
        ylim=(float(y_um.min()), float(y_um.max())),
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=title,
    )
    figstyle.save(fig, out_path)
    plt.close(fig)


def render_crossing_movie_with_overlay(
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
    fig, ax = plt.subplots(figsize=(5.4, 4.9), dpi=82)
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
    overlay_crossing_geometry(ax, pic, outline_color="black")
    ax.set(
        xlim=(float(x_um.min()), float(x_um.max())),
        ylim=(float(y_um.min()), float(y_um.max())),
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
    )
    ttl = ax.set_title(
        f"Cosine crossing pulse propagation (t = {float(times[0])*1e15:.1f} fs)"
    )
    fig.tight_layout()

    def _update(frame_idx):
        im.set_array((f_arr[frame_idx] / vmax).T.ravel())
        ttl.set_text(
            f"Cosine crossing pulse propagation (t = {float(times[frame_idx])*1e15:.1f} fs)"
        )
        return (im, ttl)

    anim = FuncAnimation(
        fig, _update, frames=len(f_arr), interval=1000 // fps, blit=False
    )
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)

