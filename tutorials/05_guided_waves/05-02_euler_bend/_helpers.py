# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Resolution ladder, mesh-override construction, mirror-symmetric graded
Cartesian mesh (`NonUniformEulerBendPIC`), setup schematic, mesh upgrade
comparison, and geometry-overlay helpers for Tutorial 5.2
(Hybrid Euler 90-degree silicon waveguide bend).
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light, epsilon_0

from wakis import figstyle, schematic
from wakis.gridFIT3D import GridFIT3D
from wakis.solverFIT3D import SolverFIT3D
from wakis.sources import GaussianSource
from wakis.templates import meshing
from wakis.templates.photonics import PIC_MATERIALS, PICSimulation, n_to_eps, slab_neff
from wakis.templates.simulation import Simulation

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


class NonUniformEulerBendPIC(PICSimulation):
    """
    2D EIM ``PICSimulation`` with a mirror-symmetric non-square/graded Cartesian
    mesh tailored to a $90^\\circ$ waveguide bend.

    Exploits the exact folding symmetry $(x, y) \\leftrightarrow (R_{\\text{eff}} - y, R_{\\text{eff}} - x)$
    of the $90^\\circ$ bend so the orthogonal input ($+x$) and output ($+y$) ports
    share identical transverse and normal cell spacings:
    - Across the curved bend and waveguide cores ($x \\in [-0.22, R_{\\text{eff}} + 0.22]\\;\\mu\\text{m}$,
      $y \\in [-0.22, R_{\\text{eff}} + 0.22]\\;\\mu\\text{m}$), uses fine spacing ``du_core``
      ($26.6\\;\\text{nm}$ at ``L3 = 20 cpw``, $15.05$ cells across $w = 400\\;\\text{nm}$).
    - Along the straight input/output leads ($x \\in [x_{\\min}, -0.22]\\;\\mu\\text{m}$ and
      $y \\in [R_{\\text{eff}} + 0.22, y_{\\max}]\\;\\mu\\text{m}$), uses non-square axial cells
      ``du_lead > du_core`` that continue smoothly into the port CPMLs.
    - In the exterior $\\text{SiO}_2$ cladding buffers ($x > R_{\\text{eff}} + 0.22\\;\\mu\\text{m}$
      and $y < -0.22\\;\\mu\\text{m}$), grades smoothly (adjacent-cell ratio $\\le$ ``grade_ratio``)
      to ``du_clad``.
    """

    def __init__(
        self,
        *args,
        du_core: float | None = None,
        du_lead: float | None = None,
        du_clad: float | None = None,
        grade_ratio: float = 1.22,
        mesh_only: bool = False,
        **kwargs,
    ):
        cpw = float(kwargs.get("cells_per_wavelength", 20.0))
        scale = 20.0 / max(cpw, 8.0)
        self.du_core = float(26.64e-9 * scale if du_core is None else du_core)
        self.du_lead = float(32.0e-9 * scale if du_lead is None else du_lead)
        self.du_clad = float(68.0e-9 * scale if du_clad is None else du_clad)
        self.grade_ratio = float(grade_ratio)
        self._mesh_only = bool(mesh_only)
        super().__init__(*args, **kwargs)

    def build(self):
        structs, domain = self._build_2d_structures_and_domain()
        xmin, xmax, ymin, ymax, zmin_2d, zmax_2d = domain
        r_eff = xmax - abs(xmin)
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
        x_nodes = meshing.graded_axis(
            [xmin, -0.22e-6, r_eff + 0.22e-6, xmax],
            [self.du_lead, self.du_core, self.du_clad],
            grading=self.grade_ratio,
        )
        y_nodes = r_eff - x_nodes[::-1]

        def _custom_axis(axis: int) -> np.ndarray:
            if axis == 2:
                return np.array([zmin_2d, zmax_2d], dtype=float)
            nodes = x_nodes if axis == 0 else y_nodes
            lo_pml = nodes[0] - np.arange(n_pml, 0, -1) * (nodes[1] - nodes[0])
            hi_pml = nodes[-1] + np.arange(1, n_pml + 1) * (nodes[-1] - nodes[-2])
            return np.concatenate([lo_pml, nodes, hi_pml])

        self.sim._axis = _custom_axis
        if self._mesh_only:
            x, y, z = self.sim._axes = [self.sim._axis(a) for a in range(3)]
            self.grid = self.sim.grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
            self.frequencies = self.sim.frequencies
            self.wavelengths = c_light / self.frequencies
            return None

        self.solver = self.sim.build()
        self.solver.ieps[:, :, :, "z"] = 0.0
        self.grid = self.sim.grid
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        return self.solver


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
        ``dx_min_nm``, ``du_lead_nm``, ``du_clad_max_nm``, ``aspect_ratio_lead``,
        ``cells_across_feature``, ``n_cells``, ``n_steps``, ``n_steps_max``,
        and ``grid_shape``.
    """
    x = np.asarray(pic.grid.x, dtype=float)
    y = np.asarray(pic.grid.y, dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)

    min_feature = float(min_feature)
    r_eff = float(r_eff)
    hw = 0.5 * min_feature

    # Cells overlapping the curved bend and transverse waveguide cores
    in_feat_x = (x[1:] > -hw) & (x[:-1] < r_eff + hw)
    in_feat_y = (y[1:] > -hw) & (y[:-1] < r_eff + hw)
    dx_feat_max = float(
        max(
            np.max(dx[in_feat_x]) if np.any(in_feat_x) else np.max(dx),
            np.max(dy[in_feat_y]) if np.any(in_feat_y) else np.max(dy),
        )
    )
    in_lead_x = x[1:] < -0.25e-6
    du_lead_m = float(np.median(dx[in_lead_x])) if np.any(in_lead_x) else dx_feat_max

    if isinstance(pic, NonUniformEulerBendPIC):
        dx_background_nm = float(dx_feat_max * 1e9)
    else:
        dx_background_nm = float(max(np.max(dx), np.max(dy)) * 1e9)
    dx_min_nm = float(min(np.min(dx), np.min(dy)) * 1e9)
    du_clad_max_nm = float(max(np.max(dx), np.max(dy)) * 1e9)
    min_feature_nm = float(min_feature * 1e9)
    cells_across_feature = float(min_feature / dx_feat_max)
    aspect_ratio_lead = float(du_lead_m / max(dx_feat_max, 1e-30))

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
        inv_h2 = (
            1.0 / float(np.min(dx)) ** 2
            + 1.0 / float(np.min(dy)) ** 2
            + 1.0 / float(np.min(dz)) ** 2
        )
        dt = float(pic.cfln) / (c_light * float(np.sqrt(inv_h2)))
    n_steps_max = int(np.ceil(max_time / dt))
    nt_run = getattr(pic.solver, "Nt", None) if pic.solver is not None else None
    n_steps = int(nt_run) if nt_run is not None else n_steps_max

    return {
        "min_feature_nm": round(min_feature_nm, 3),
        "dx_background_nm": round(dx_background_nm, 3),
        "dx_min_nm": round(dx_min_nm, 3),
        "du_lead_nm": round(float(du_lead_m * 1e9), 3),
        "du_clad_max_nm": round(du_clad_max_nm, 3),
        "aspect_ratio_lead": round(aspect_ratio_lead, 3),
        "cells_across_feature": round(cells_across_feature, 3),
        "n_cells": n_cells,
        "n_steps": n_steps,
        "n_steps_max": n_steps_max,
        "grid_shape": [nx, ny, nz],
    }


class _PolygonSchematicProxy:
    """Lightweight polygon structure proxy exposing `.vertices`, `.material`, and `.name`
    (without `.bounds`) so `wakis.schematic` classifies a 2D `PolySlab` as `kind='polygon'`.
    """

    def __init__(self, slab, material_eps: float, name: str):
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
    structures,
    domain,
    r_eff: float = 4.0e-6,
    port_offset: float = 1.0e-6,
    n_eff_slab: float = 2.8134,
    n_clad: float = 1.444,
    wl_min: float = 1500e-9,
    wl_max: float = 1600e-9,
    wl_ref: float = 1550e-9,
    n_pml: int = 10,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    include_flux_box: bool = True,
):
    """
    Build a lightweight 2D ``SolverFIT3D`` plus normalized source and monitor
    dictionaries compatible with ``wakis.schematic.plot_schematic`` and
    ``wakis.schematic.overlay_geometry``.
    """
    xmin, xmax, ymin, ymax = [float(v) for v in domain[:4]]
    dz = 50.0e-9
    if x_grid is None or y_grid is None:
        du = float(wl_min) / (float(n_eff_slab) * 20.0)
        x_core = np.linspace(xmin, xmax, int(round((xmax - xmin) / du)) + 1)
        y_core = np.linspace(ymin, ymax, int(round((ymax - ymin) / du)) + 1)
        pml_lo = np.arange(n_pml, 0, -1, dtype=float) * du
        pml_hi = np.arange(1, n_pml + 1, dtype=float) * du
        x_grid = np.concatenate([x_core[0] - pml_lo, x_core, x_core[-1] + pml_hi])
        y_grid = np.concatenate([y_core[0] - pml_lo, y_core, y_core[-1] + pml_hi])
    z_grid = np.array([-0.5 * dz, 0.5 * dz], dtype=float)

    grid = GridFIT3D(x=x_grid, y=y_grid, z=z_grid, verbose=0)
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

    f_min = c_light / float(wl_max)
    f_max = c_light / float(wl_min)
    f0 = 0.5 * (f_min + f_max)
    fwidth = 1.25 * (f_max - f_min)
    pulse_obj = GaussianSource(frequency=f0, fwidth=fwidth, cutoff=4.2)

    x_in_mon = -0.75 * float(port_offset)
    x_in_src = x_in_mon - 0.05e-6
    y_out_mon = float(r_eff) + 0.75 * float(port_offset)

    sources_s = [
        {
            "kind": "sheet",
            "type_name": "ModeSource",
            "short_word": "ModeSrc",
            "label": "Port 1 ModeSource (TE₀, +x)",
            "desc": f"Port 1 ModeSource at x = {x_in_src * 1e6:.2f} µm (TE₀, +x)",
            "h_bounds": (x_in_src, x_in_src),
            "v_bounds": (-0.55e-6, +0.55e-6),
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
    ]
    monitors_s = [
        {
            "kind": "plane",
            "type_name": "ModeMonitor",
            "short_word": "in mon",
            "label": "Port 1 ModeMonitor (S₁₁)",
            "desc": f"Port 1 ModeMonitor at x = {x_in_mon * 1e6:.2f} µm (S₁₁)",
            "h_bounds": (x_in_mon, x_in_mon),
            "v_bounds": (-0.55e-6, +0.55e-6),
        },
        {
            "kind": "plane",
            "type_name": "ModeMonitor",
            "short_word": "out mon",
            "label": "Port 2 ModeMonitor (S₂₁, +y)",
            "desc": f"Port 2 ModeMonitor at y = {y_out_mon * 1e6:.2f} µm (S₂₁, +y)",
            "h_bounds": (float(r_eff) - 0.55e-6, float(r_eff) + 0.55e-6),
            "v_bounds": (y_out_mon, y_out_mon),
        },
    ]
    if include_flux_box:
        monitors_s.append(
            {
                "kind": "box",
                "type_name": "FluxMonitor",
                "short_word": "FluxBox",
                "label": "2D Poynting FluxMonitor box",
                "desc": "Closed 2D Poynting FluxMonitor box (4 faces inside CPML)",
                "h_bounds": (x_in_src - 0.08e-6, float(xmax) - 0.06e-6),
                "v_bounds": (float(ymin) + 0.06e-6, float(ymax) - 0.06e-6),
            }
        )
    material_names = {
        round(float(eps_clad), 2): f"SiO₂ cladding\n(n = {n_clad:.3f}, ε_r = {eps_clad:.2f})",
    }
    return solver_s, sources_s, monitors_s, material_names


def draw_polyslab_regions(
    ax,
    structures,
    facecolor: str | None = "#1d4ed8",
    alpha: float = 0.28,
    edgecolor: str = "#1e293b",
    lw: float = 1.35,
    ls: str = "-",
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    label_color: str = "#0f172a",
    zorder: int = 5,
):
    """Draw exact 2D polygon outlines (and optional fills/labels) for ``PolySlab`` items."""
    for st in structures:
        verts_um = np.asarray(st.vertices, dtype=float) * 1e6
        ax.add_patch(
            MplPolygon(
                verts_um,
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
            fontsize=7.6,
            fontweight="bold",
            color=label_color,
            ha="center",
            va="center",
            zorder=zorder + 3,
            bbox=dict(
                boxstyle="round,pad=0.22",
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
    amp_um: float = 0.42,
    y_out_port_um: float | None = None,
    x_out_um: np.ndarray | None = None,
    ex_out: np.ndarray | None = None,
    neff_out: float | None = None,
):
    """
    Overlay the launched $\\text{TE}_0$ mode profile $|E_y(y)|$ on Port 1 (`+x` launch)
    and optionally the rotated output $\\text{TE}_0$ profile $|E_x(x)|$ on Port 2 (`+y` exit).
    """
    y_arr = np.asarray(y_um, dtype=float)
    env = np.abs(np.asarray(ey))
    env = env / max(float(np.max(env)), 1e-30)
    x_curve = float(x_port_um) + float(amp_um) * env
    ax.fill_betweenx(
        y_arr,
        float(x_port_um),
        x_curve,
        color="#ef4444",
        alpha=0.26,
        zorder=8,
    )
    ax.plot(
        x_curve,
        y_arr,
        color="#b91c1c",
        lw=1.7,
        marker="none",
        zorder=9,
    )
    ax.text(
        float(x_port_um) + float(amp_um) + 0.06,
        0.58,
        f"Launched TE₀ |Ey(y)|\n(n_eff = {neff:.4f})",
        fontsize=7.1,
        fontweight="bold",
        color="#991b1b",
        ha="left",
        va="bottom",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.18", fc="#fef2f2", ec="#ef4444", alpha=0.92, lw=0.7),
    )

    if y_out_port_um is not None and x_out_um is not None and ex_out is not None:
        x_o = np.asarray(x_out_um, dtype=float)
        env_o = np.abs(np.asarray(ex_out))
        env_o = env_o / max(float(np.max(env_o)), 1e-30)
        y_curve = float(y_out_port_um) - 0.36 * env_o
        ax.fill_between(
            x_o,
            y_curve,
            float(y_out_port_um),
            color="#0284c7",
            alpha=0.24,
            zorder=8,
        )
        ax.plot(
            x_o,
            y_curve,
            color="#0369a1",
            lw=1.6,
            marker="none",
            zorder=9,
        )
        neff_o_val = float(neff if neff_out is None else neff_out)
        ax.text(
            float(np.mean(x_o)) - 0.55,
            float(y_out_port_um) - 0.55,
            f"Rotated TE₀ |Ex(x)|\n(n_eff = {neff_o_val:.4f})",
            fontsize=7.0,
            fontweight="bold",
            color="#075985",
            ha="right",
            va="top",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.18", fc="#f0f9ff", ec="#0284c7", alpha=0.92, lw=0.7),
        )


def plot_setup_schematic_euler_bend(
    structures,
    domain,
    euler_info: dict,
    primary_data: dict,
    out_path: Path,
    pic_mesh: PICSimulation | None = None,
):
    """Render `00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`, and
    `00_setup_table.md` for Tutorial 5.2 using `wakis.schematic.save_setup_figures`.
    """
    from matplotlib.patches import Rectangle

    figstyle.apply_style()
    out_path = Path(out_path)
    out_dir = out_path if (out_path.is_dir() or out_path.suffix == "") else out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n_eff_slab = float(primary_data.get("n_eff_slab", 2.8134))
    eps_slab = n_to_eps(n_eff_slab)
    n_clad = float(PIC_MATERIALS["SiO2"])
    eps_clad = n_to_eps(n_clad)
    r_eff = float(primary_data.get("r_eff", 4.0e-6))

    x_g = pic_mesh.grid.x if pic_mesh is not None else None
    y_g = pic_mesh.grid.y if pic_mesh is not None else None
    solver_raw, sources_s, monitors_s, _ = build_schematic_components(
        structures,
        domain,
        r_eff=r_eff,
        n_eff_slab=n_eff_slab,
        n_clad=n_clad,
        x_grid=x_g,
        y_grid=y_g,
        include_flux_box=True,
    )
    solver_s = _SchematicSolverProxy(solver_raw, eps_clad=eps_clad)
    poly_proxy = _PolygonSchematicProxy(
        structures[0],
        material_eps=eps_slab,
        name=f"Si core (w = 400 nm, 2D EIM n_eff = {n_eff_slab:.3f})",
    )
    mat_names = {
        "background": f"SiO₂ cladding (n = {n_clad:.3f}, ε_r = {eps_clad:.2f})",
        "region_0": f"Si core (w = 400 nm, 2D EIM n_eff = {n_eff_slab:.3f}, ε_eff = {eps_slab:.2f})",
    }
    variations = [
        {
            "name": "kottke_L1..L3, staircase_L1..L3 (tasks 0–5)",
            "param": "cpw ∈ {10, 14, 20}, N_feat ∈ {4, 6, 8}, averaging ∈ {kottke, staircase}",
            "note": "Uniform square-cell resolution ladder at R_eff = 4.0 µm",
        },
        {
            "name": "kottke_reference (task 6)",
            "param": "cpw = 40, N_feat = 16 (509 × 509 × 1 = 259,081 cells)",
            "note": "Independent 2× fine Kottke reference",
        },
        {
            "name": "sweep_circular_R*, sweep_euler_R* (tasks 7–14)",
            "param": "R_eff ∈ {2.5, 3.0, 3.5, 4.0} µm × {circular, hybrid Euler}",
            "note": "8-run radius sweep comparing circular vs hybrid Euler 90° bends",
        },
        {
            "name": "kottke_nonuniform (task 15)",
            "param": "Mirror-symmetric graded Cartesian mesh (238 × 238 × 1 = 56,644 cells)",
            "note": "15.0 cells across w = 400 nm (Δu_core = 26.6 nm), Δu_lead = 32.0 nm, Δu_clad ≤ 68.0 nm (−19.3% cells, −43.5% time)",
        },
    ]
    title_geom = "Tutorial 5.2 Setup: 2D EIM Hybrid Euler 90° Waveguide Bend"
    title_mesh = "Discretization (Yee Mesh & w = 400 nm Clothoid Zoom) and Excitation"

    res = schematic.save_setup_figures(
        solver_s,
        out_dir,
        prefix="00",
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=1550e-9,
        min_feature=0.40e-6,
        title_geometry=title_geom,
        title_mesh=title_mesh,
        variations=variations,
    )

    # Enhance 00a_setup_geometry.png with top/bottom CPML shading, mode profiles, C^2 junctions, and non-overlapping port tags
    fig_a, ax_a = schematic.plot_schematic(
        solver_s,
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=1550e-9,
        title=title_geom,
        width_in=7.6,
        height_in=5.4,
    )
    grid_a = solver_raw.grid
    n_pml = int(getattr(solver_raw, "n_pml", 10))
    x0_um, x1_um = float(grid_a.xmin * 1e6), float(grid_a.xmax * 1e6)
    y0_um, y1_um = float(grid_a.ymin * 1e6), float(grid_a.ymax * 1e6)
    th_y_low = float((grid_a.y[n_pml] - grid_a.y[0]) * 1e6)
    th_y_high = float((grid_a.y[-2] - grid_a.y[-2 - n_pml]) * 1e6)
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

    if "port_y_in" in primary_data and "port_ey_in" in primary_data:
        y_arr = np.asarray(primary_data["port_y_in"], dtype=float) * 1e6
        env = np.abs(np.asarray(primary_data["port_ey_in"]))
        env = env / max(float(np.max(env)), 1e-30)
        x_port_um = -0.80
        x_curve = x_port_um + 0.45 * env
        ax_a.fill_betweenx(
            y_arr, x_port_um, x_curve, color="#D55E00", alpha=0.25, zorder=7.5
        )
        ax_a.plot(
            x_curve, y_arr, color="#D55E00", lw=1.7, marker="none", zorder=8
        )
    if "port_x_out" in primary_data and "port_ex_out" in primary_data:
        x_o = np.asarray(primary_data["port_x_out"], dtype=float) * 1e6
        env_o = np.abs(np.asarray(primary_data["port_ex_out"]))
        env_o = env_o / max(float(np.max(env_o)), 1e-30)
        y_port_um = 4.75
        y_curve = y_port_um - 0.36 * env_o
        ax_a.fill_between(
            x_o, y_curve, y_port_um, color="#006B4E", alpha=0.22, zorder=7.5
        )
        ax_a.plot(
            x_o, y_curve, color="#006B4E", lw=1.6, marker="none", zorder=8
        )

    jx = [euler_info["xp"] * 1e6, (r_eff - euler_info["yp"]) * 1e6]
    jy = [euler_info["yp"] * 1e6, (r_eff - euler_info["xp"]) * 1e6]
    ax_a.plot(jx, jy, "o", color="#e11d48", ms=5.0, zorder=9)

    tag_Overrides = {
        "1 ·": (-0.42, 0.88),
        "2 ·": (-0.38, -0.78),
        "3 ·": (2.95, 4.18),
        "4 ·": (1.10, 4.75),
        "Si core": (1.50, 2.35),
        "SiO₂ cladding": (2.15, -0.75),
    }
    for txt in ax_a.texts:
        s = txt.get_text()
        if s == "" and hasattr(txt, "xyann"):
            if abs(txt.xy[1] - 2.0) < 0.3 and txt.xy[0] < 0.0:
                txt.xy = (-0.45, 0.0)
                txt.set_position((-0.80, 0.0))
            continue
        if "µm" in s and txt.get_position()[1] > 6.0:
            px, py = txt.get_position()
            txt.set_position((px, 6.62))
            continue
        for prefix_key, pos in tag_Overrides.items():
            if s.startswith(prefix_key):
                txt.set_position(pos)
                break
    if getattr(fig_a, "_suptitle", None) is not None:
        fig_a._suptitle.set_y(1.01)

    figstyle.save(fig_a, res["geometry_figure"], close=True, auto_legend_below=False)

    # Enhance 00b_setup_mesh_and_excitation.png with the Euler bend polygon on ax_mesh and 1x clothoid zoom inset
    zoom_bounds = (-0.30e-6, 1.25e-6, -0.42e-6, 0.58e-6)
    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        solver_s,
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        mesh_every=6,
        zoom_bounds=zoom_bounds,
        wavelength=1550e-9,
        min_feature=0.40e-6,
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


def plot_mesh_upgrade_comparison_euler(
    pic_uniform: PICSimulation,
    pic_nonuniform: PICSimulation,
    structures,
    d_uniform: dict,
    d_nonuniform: dict,
    d_ref: dict,
    meta_uniform: dict,
    meta_nonuniform: dict,
    meta_ref: dict,
    out_path: Path,
):
    """
    Render ``04b_mesh_upgrade_comparison.png`` (3 panels):
    - Panel (a): Uniform Cartesian baseline (`kottke_L3`, 265×265 = 70,225 cells, 105.12 s).
    - Panel (b): Upgraded mirror-symmetric non-square/graded Cartesian mesh
      (`kottke_nonuniform`, 238×238 = 56,644 cells, -19.3% cells, 59.42 s, -43.5% time).
    - Panel (c): Broadband insertion loss across 1505–1595 nm vs
      independent `kottke_reference` (40 cpw, 259,081 cells, 700.74 s).
    """
    figstyle.apply_style()
    fig, (ax0, ax1, ax2) = figstyle.new_figure(
        nrows=1,
        ncols=3,
        width=13.6,
        height=4.4,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.15], "wspace": 0.32},
    )

    for ax, pic_obj, meta_d, d_npz, ttl, col_grid in (
        (
            ax0,
            pic_uniform,
            meta_uniform,
            d_uniform,
            "(a) Uniform baseline (kottke_L3)",
            "#2563eb",
        ),
        (
            ax1,
            pic_nonuniform,
            meta_nonuniform,
            d_nonuniform,
            "(b) Graded mesh (kottke_nonuniform)",
            "#059669",
        ),
    ):
        xg = np.asarray(pic_obj.grid.x, dtype=float) * 1e6
        yg = np.asarray(pic_obj.grid.y, dtype=float) * 1e6
        xl = (float(xg[0]), float(xg[-1]))
        yl = (float(yg[0]), float(yg[-1]))
        # Draw every 3rd grid line so grading and non-square lead/cladding cells are clearly visible
        xs = xg[::3]
        ys = yg[::3]
        segs = [[(xv, yl[0]), (xv, yl[1])] for xv in xs] + [
            [(xl[0], yv), (xl[1], yv)] for yv in ys
        ]
        ax.add_collection(
            LineCollection(segs, colors=col_grid, linewidths=0.45, alpha=0.48, zorder=2)
        )
        draw_polyslab_regions(
            ax,
            structures,
            facecolor="#1e40af" if ax is ax0 else "#065f46",
            alpha=0.30,
            edgecolor="#0f172a",
            lw=1.35,
        )
        # Highlight fine core box [-0.22, 4.22] um on the graded mesh
        if ax is ax1:
            ax.axvline(-0.22, color="#d97706", ls="--", lw=1.1, alpha=0.85)
            ax.axvline(4.22, color="#d97706", ls="--", lw=1.1, alpha=0.85)
            ax.axhline(-0.22, color="#d97706", ls="--", lw=1.1, alpha=0.85)
            ax.axhline(4.22, color="#d97706", ls="--", lw=1.1, alpha=0.85)

        w = d_npz["wavelengths"] * 1e9
        m = (w >= 1515.0) & (w <= 1585.0)
        t_m = float(np.mean(np.abs(d_npz["S21"][m]) ** 2))
        il_m = -10.0 * np.log10(max(t_m, 1e-12))
        r_m_db = 10.0 * np.log10(max(float(np.mean(np.abs(d_npz["S11"][m]) ** 2)), 1e-12))
        gs = meta_d["grid_shape"]
        n_c = int(meta_d["n_cells"])
        t_s = float(meta_d["elapsed_s"])
        n_f = float(meta_d["cells_across_feature"])
        dx_min = float(meta_d["dx_min_nm"])
        du_max = float(meta_d.get("du_clad_max_nm", meta_d["dx_background_nm"]))

        summary = (
            f"Grid: {gs[0]}×{gs[1]}×1 = {n_c:,} cells\n"
            f"w = 400 nm: {n_f:.1f} cells (Δ_min = {dx_min:.1f} nm)\n"
            f"Cladding max Δ = {du_max:.1f} nm\n"
            f"Wall time: {t_s:.1f} s\n"
            f"⟨IL⟩ = {il_m:.4f} dB, ⟨|S₁₁|²⟩ = {r_m_db:.1f} dB"
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
            fontsize=6.9,
            va="top",
            ha="left",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=col_grid, alpha=0.92, lw=0.9),
        )
        ax.set(
            aspect="equal",
            xlim=xl,
            ylim=yl,
            xlabel="x [µm]",
            ylabel="y [µm]",
            title=ttl,
        )

    # Panel (c): Broadband insertion loss comparison
    series_list = [
        (
            d_uniform,
            f"Uniform L3 ({meta_uniform['n_cells']:,} cells, {meta_uniform['elapsed_s']:.1f} s)",
            "#2563eb",
            "-",
        ),
        (
            d_nonuniform,
            f"Graded L3 ({meta_nonuniform['n_cells']:,} cells, {meta_nonuniform['elapsed_s']:.1f} s)",
            "#059669",
            "--",
        ),
        (
            d_ref,
            f"Ref 40 cpw ({meta_ref['n_cells']:,} cells, {meta_ref['elapsed_s']:.1f} s)",
            "#dc2626",
            ":",
        ),
    ]
    for d_k, lbl_k, col_k, ls_k in series_list:
        w = d_k["wavelengths"] * 1e9
        idx_s = np.argsort(w)
        ws = w[idx_s]
        mb = (ws >= 1505.0) & (ws <= 1595.0)
        il_db = -10.0 * np.log10(np.maximum(np.abs(d_k["S21"][idx_s][mb]) ** 2, 1e-12))
        ax2.plot(
            ws[mb],
            il_db,
            color=col_k,
            ls=ls_k,
            lw=2.1,
            marker="none",
            label=lbl_k,
        )

    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Insertion loss $-10\log_{10}|S_{21}|^2$ [dB]",
        ylim=(-0.003, 0.042),
        title="(c) Equal-accuracy insertion loss vs 40 cpw ref",
    )
    ax2.grid(True, ls=":", alpha=0.45)
    ax2.legend(loc="upper left", fontsize=7.0)
    figstyle.save(fig, out_path)
    plt.close(fig)


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
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(
        nrows=1,
        ncols=3,
        width=figstyle.FULL_WIDTH_IN * 1.15,
        height=4.2,
        sharey=True,
    )
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
            cmap=figstyle.CMAP_EPS,
            shading="flat",
        )
        xm = x_um[(x_um >= xlim_um[0]) & (x_um <= xlim_um[1])]
        ym = y_um[(y_um >= ylim_um[0]) & (y_um <= ylim_um[1])]
        for xv in xm:
            ax.axvline(xv, color="white", lw=0.35, alpha=0.55)
        for yv in ym:
            ax.axhline(yv, color="white", lw=0.35, alpha=0.55)

        draw_polyslab_regions(
            ax,
            structures,
            facecolor=None,
            edgecolor="#facc15",
            lw=1.35,
            ls="--",
        )

        res = measure_mesh_resolution(pic_obj, min_feature=min_feature_m)
        n_feat = res["cells_across_feature"]
        dx_nm = res["dx_min_nm"]

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
            fontsize=7.6,
            fontweight="bold",
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.22", fc="black", ec="white", alpha=0.72),
        )
        ax.set(
            aspect="equal",
            xlim=xlim_um,
            ylim=ylim_um,
            xlabel="x [µm]",
        )
        ax.set_title(
            f"{title}\n({n_feat:.1f} cells across w, Δx={dx_nm:.1f} nm)",
            fontsize=9.0,
        )

    axes[0].set_ylabel("y [µm]")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.025)
    cb.set_label(r"Relative permittivity $\varepsilon_{yy} / \varepsilon_0$")
    if out_path is not None:
        figstyle.save(fig, out_path)
    plt.close(fig)
    return fig


def render_field_map_with_overlay(
    x_m: np.ndarray,
    y_m: np.ndarray,
    field_2d: np.ndarray,
    structures,
    domain,
    kind: str,
    title: str,
    out_path: Path,
    db_floor: float = -35.0,
):
    """
    Render a 2D steady-state field map (`intensity_log` or `real`) with
    ``wakis.schematic.overlay_geometry`` and exact ``PolySlab`` outlines/labels.
    """
    figstyle.apply_style()
    solver_s, sources_s, monitors_s, mat_names = build_schematic_components(
        structures,
        domain,
        include_flux_box=False,
    )
    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=7.2, height=6.0)
    x_um = np.asarray(x_m, dtype=float) * 1e6
    y_um = np.asarray(y_m, dtype=float) * 1e6

    if kind == "intensity_log":
        e2 = np.asarray(field_2d, dtype=float)
        fmax = max(float(np.max(e2)), 1e-30)
        data = 10.0 * np.log10(np.maximum(e2 / fmax, 10.0 ** (db_floor / 10.0)))
        im = ax.pcolormesh(
            x_um,
            y_um,
            data.T,
            cmap=figstyle.CMAP_MAGNITUDE,
            vmin=db_floor,
            vmax=0.0,
            shading="auto",
        )
        cb_lbl = r"$10\log_{10}(|\mathbf{E}|^2 / |\mathbf{E}|_{\max}^2)$ [dB]"
        out_col = "#ffffff"
    else:
        re_f = np.real(np.asarray(field_2d))
        vmax = max(float(np.percentile(np.abs(re_f), 99.5)), 1e-30)
        im = ax.pcolormesh(
            x_um,
            y_um,
            re_f.T / vmax,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        cb_lbl = r"Normalized $\mathrm{Re}(E)$"
        out_col = "#0f172a"

    schematic.overlay_geometry(
        ax,
        solver_s,
        plane="xy",
        unit="µm",
        structures=(),
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        outline_color=out_col,
    )
    draw_polyslab_regions(
        ax,
        structures,
        facecolor=None,
        edgecolor=out_col,
        lw=1.25,
        label="Si core (n_eff,slab = 2.813)",
        label_xy=(1.55, 2.15),
    )
    ax.text(
        2.10,
        -0.85,
        "SiO₂ cladding (n = 1.444)",
        fontsize=7.3,
        color=out_col,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.18",
            fc="#0f172a" if kind == "intensity_log" else "white",
            ec=out_col,
            alpha=0.80,
            lw=0.7,
        ),
    )
    ax.set(
        aspect="equal",
        xlim=(float(x_um.min()), float(x_um.max())),
        ylim=(float(y_um.min()), float(y_um.max())),
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=title,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cb_lbl)
    figstyle.save(fig, out_path)
    plt.close(fig)


def render_movie_with_overlay(
    x_m: np.ndarray,
    y_m: np.ndarray,
    frames: np.ndarray,
    times_s: np.ndarray,
    structures,
    domain,
    out_path: Path,
    fps: int = 12,
):
    """
    Render ``10_pulse_propagation.gif`` with ``schematic.overlay_geometry`` and
    labelled waveguide/cladding regions on every frame (kept $< 5\\;\\text{MB}$).
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    figstyle.apply_style()
    solver_s, sources_s, monitors_s, mat_names = build_schematic_components(
        structures,
        domain,
        include_flux_box=False,
    )
    x_um = np.asarray(x_m, dtype=float) * 1e6
    y_um = np.asarray(y_m, dtype=float) * 1e6
    f_arr = np.asarray(frames, dtype=np.float32)
    t_fs = np.asarray(times_s, dtype=float) * 1e15

    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=105)
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
        sources=sources_s,
        monitors=monitors_s,
        outline_color="#0f172a",
    )
    draw_polyslab_regions(
        ax,
        structures,
        facecolor=None,
        edgecolor="#0f172a",
        lw=1.2,
        label="Si hybrid Euler bend\n(n_eff,slab = 2.813)",
        label_xy=(1.55, 2.25),
    )
    ax.text(
        2.10,
        -0.88,
        "SiO₂ cladding (n = 1.444)",
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
    ttl = ax.set_title(f"90° hybrid Euler bend pulse propagation (t = {t_fs[0]:.1f} fs)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"Normalized $E_y(x, y, t)$")
    fig.tight_layout()

    def _update(i):
        im.set_array(f_arr[i].T.ravel())
        ttl.set_text(f"90° hybrid Euler bend pulse propagation (t = {t_fs[i]:.1f} fs)")
        return (im, ttl)

    anim = FuncAnimation(fig, _update, frames=len(f_arr), interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
