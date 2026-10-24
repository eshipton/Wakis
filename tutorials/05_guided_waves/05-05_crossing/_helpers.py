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

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.constants import c as c_light, epsilon_0

from wakis.templates import meshing
from wakis.templates.photonics import PolySlab, n_to_eps, slab_neff
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
):
    """
    Render `04_permittivity_mesh_zoom.png` as a 2x2 figure:
    - Top row (a, b): Zoomed on the smallest feature (`w_in = 0.35 um` access
      waveguide at `x in [-6.35, -5.25] um, y in [-0.45, +0.45] um`) showing
      the Yee cell edges and `eps_{r,y}` at L1 and at the reference level, with
      the `w_in = 0.35 um (350 nm)` feature width and cell count annotated.
    - Bottom row (c, d): Zoomed on the curved cosine-taper & crossing corner
      (`x in [-3.2, 1.2] um, y in [-1.1, 1.1] um`) comparing Staircase vs Kottke
      subpixel smoothing at L1.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), dpi=150)

    def _draw_panel(ax, solver, xlim_um, ylim_um, title, annotate_feature=False):
        g = solver.grid
        iz = solver.Nz // 2
        ieps = solver.ieps[:, :, iz, component]
        if hasattr(ieps, "get"):
            ieps = ieps.get()
        eps_r = np.where(
            ieps > 0, 1.0 / (np.where(ieps > 0, ieps, 1.0) * epsilon_0), 1.0
        )
        x_um, y_um = g.x * 1e6, g.y * 1e6
        pcm = ax.pcolormesh(x_um, y_um, eps_r.T, cmap="viridis", shading="flat")
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
                fontsize=8.2,
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
        ax.set_title(title, fontsize=9.8)
        cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(rf"$\varepsilon_{{r,{component}}}$")

    feat_xlim = (-6.35, -5.25)
    feat_ylim = (-0.42, 0.42)
    _draw_panel(
        axes[0, 0],
        solver_l1_kottke,
        feat_xlim,
        feat_ylim,
        f"(a) Smallest feature @ L1 (cpw={int(cpw_l1)}, N≥{int(n_l1)})",
        annotate_feature=True,
    )
    _draw_panel(
        axes[0, 1],
        solver_ref_kottke,
        feat_xlim,
        feat_ylim,
        f"(b) Smallest feature @ Reference (cpw={int(cpw_ref)}, N≥{int(n_ref)})",
        annotate_feature=True,
    )

    taper_xlim = (-3.2, 1.2)
    taper_ylim = (-1.1, 1.1)
    _draw_panel(
        axes[1, 0],
        solver_l1_stair,
        taper_xlim,
        taper_ylim,
        f"(c) Staircase on cosine taper (L1, cpw={int(cpw_l1)})",
        annotate_feature=False,
    )
    _draw_panel(
        axes[1, 1],
        solver_l1_kottke,
        taper_xlim,
        taper_ylim,
        f"(d) Kottke subpixel on cosine taper (L1, cpw={int(cpw_l1)})",
        annotate_feature=False,
    )

    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig, axes


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
    out_path=None,
):
    """
    Render `15_kottke_vs_staircase_convergence.png` against the independent
    Kottke reference `T_ref`, labeling every point with its `cells_across_feature`
    and adding a secondary top x-axis next to `cpw`.
    """
    from matplotlib.ticker import NullFormatter

    res = np.asarray(cpws, dtype=float)
    ca = np.asarray(cells_across, dtype=float)
    err_s = np.asarray(err_s, dtype=float)
    err_k = np.asarray(err_k, dtype=float)
    rs_vals = np.asarray(rs_vals, dtype=float)
    rk_vals = np.asarray(rk_vals, dtype=float)
    if level_labels is None:
        level_labels = [f"L{i + 1}" for i in range(len(res))]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=150)

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
            fontsize=7.6,
            color="#1b9e77",
            fontweight="bold",
        )
        ax0.annotate(
            f"{lbl}: {c_val:.1f} cells",
            xy=(r_val, es),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.4,
            color="#d95f02",
        )

    ax0.set(
        xlabel="Background resolution [cells per wavelength (cpw)]",
        ylabel=r"Band RMS error $\|T(\lambda) - T_{\mathrm{ref}}(\lambda)\|_{\mathrm{RMS}}$",
        title=(
            f"(a) Convergence vs Kottke ref "
            f"(cpw={int(ref_cpw)}, {ref_cells_across:.1f} cells)"
        ),
    )
    ax0.set_xlim(res[0] * 0.85, res[-1] * 1.20)
    ax0.set_ylim(
        min(float(np.min(err_k)), float(np.min(guide_2))) * 0.72,
        max(float(np.max(err_s)), float(np.max(err_k))) * 1.32,
    )
    ax0.set_xticks(res)
    ax0.set_xticklabels([f"{int(r)}" for r in res])
    ax0.xaxis.set_minor_formatter(NullFormatter())
    ax0.grid(True, which="both", ls=":", alpha=0.4)
    ax0.legend(loc="lower left", fontsize=8.0)

    ax0_top = ax0.twiny()
    ax0_top.set_xscale("log")
    ax0_top.set_xlim(ax0.get_xlim())
    ax0_top.set_xticks(res)
    ax0_top.set_xticklabels([f"{c:.1f} cells" for c in ca])
    ax0_top.xaxis.set_minor_formatter(NullFormatter())
    ax0_top.set_xlabel(r"Cells across smallest feature ($w_{\mathrm{in}} = 350\;\mathrm{nm}$)")

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
            fontsize=7.5,
            color="#1b9e77",
        )
        ax1.annotate(
            f"{lbl} ({c_val:.1f} cells)",
            xy=(r_val, rs),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.4,
            color="#d95f02",
        )

    ax1.set(
        xlabel="Background resolution [cells per wavelength (cpw)]",
        ylabel=r"Band-averaged reflection $\langle |S_{11}|^2 \rangle$",
        title="(b) Spurious sidewall reflection vs resolution",
    )
    ax1.set_xlim(res[0] * 0.85, res[-1] * 1.20)
    ax1.set_ylim(
        min(float(np.min(rk_vals)), float(ref_r_mean)) * 0.68,
        max(float(np.max(rs_vals)), float(np.max(rk_vals))) * 1.32,
    )
    ax1.set_xticks(res)
    ax1.set_xticklabels([f"{int(r)}" for r in res])
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(loc="center right", fontsize=8.0)

    ax1_top = ax1.twiny()
    ax1_top.set_xscale("log")
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(res)
    ax1_top.set_xticklabels([f"{c:.1f} cells" for c in ca])
    ax1_top.xaxis.set_minor_formatter(NullFormatter())
    ax1_top.set_xlabel(r"Cells across smallest feature ($w_{\mathrm{in}} = 350\;\mathrm{nm}$)")

    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig, (ax0, ax1)
