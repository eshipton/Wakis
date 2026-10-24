# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Photonic-integrated-circuit (PIC) simulation wrapper on `QuadMesh` and
`SubgridFIT2D`.

`SubgridPICSimulation` mirrors `wakis.templates.photonics.PICSimulation`
(`two_d=True`, 2D effective-index method with `polarization="TE"` in
`SubgridFIT2D`, i.e. in-plane `(E_x, E_y)` and out-of-plane `H_z`), adding
local quadtree refinement (`refine`, `ratio`, `max_level`, or automatic
conversion of `mesh_overrides`), optional local time stepping (`lts=p`), and
run-cost diagnostics (`timing` / `summary()`).
"""

import resource
import time
from types import SimpleNamespace

import numpy as np
from scipy.constants import c as c_light

from ..sources import GaussianSource
from ..templates import meshing
from ..templates.photonics import PIC_MATERIALS, n_to_eps, slab_neff
from .geometry import PolySlabGeometry, choose_refinement, refine_boxes
from .mesh import QuadMesh
from .monitors import SubgridDFTMonitor, SubgridFluxMonitor, SubgridModeMonitor
from .ports import PIC_POLARIZATION, SubgridModeSource, solve_port_modes_2d
from .solver import SubgridFIT2D

__all__ = ["SubgridPICSimulation"]


class SubgridPICSimulation:
    """
    2D effective-index PIC simulation on a locally refined `QuadMesh` with
    `SubgridFIT2D(polarization="TE")`.

    Accepts the same geometry (`PolySlab`s), `Port`s, wavelength band,
    `cells_per_wavelength`, `n_pml`, and `mesh_overrides` as
    `wakis.templates.photonics.PICSimulation`, plus quadtree refinement
    controls (`refine`, `ratio`, `max_level`) and local time stepping (`lts`).

    Parameters
    ----------
    wavelength_min, wavelength_max : float, optional
        Wavelength band [m]. Default `1500e-9`, `1600e-9`.
    structures : sequence of PolySlab, optional
        3D/2D waveguide structures.
    ports : sequence of Port, optional
        Named waveguide ports.
    n_core, n_clad : float, optional
        Core and cladding refractive indices. Default Si (`3.476`), SiO2 (`1.444`).
    n_sub : float, optional
        Substrate refractive index (defaults to `n_clad`).
    wavelength_ref : float, optional
        Reference wavelength [m] for the vertical slab effective index.
        Default `1550e-9`.
    two_d : bool, optional
        Must be True (`SubgridFIT2D` is a 2D solver). Default True.
    averaging : str, optional
        Accepted for compatibility with `PICSimulation`; `SubgridFIT2D` uses
        its `n_sub x n_sub` sub-pixel series/parallel strip integration.
    cells_per_wavelength : float, optional
        Coarse cells per wavelength at `f_max` in the effective slab medium.
        Default 20.
    n_freqs : int, optional
        Number of frequency points across `[f_min, f_max]`. Default 51.
    domain : tuple of 4 (or 6) float, optional
        Interior `(xmin, xmax, ymin, ymax)` [m] inside the PML.
    padding : float, optional
        Transverse padding [m] added to structure bounds when `domain` is None.
        Default `0.8e-6`.
    n_pml : int, optional
        PML thickness in coarse cells. Default 10.
    cfln : float, optional
        Accepted for compatibility with `PICSimulation`.
    cfl : float, optional
        Fraction of the exact Lanczos stability limit `dt_max` of
        `SubgridFIT2D`. Default 0.9.
    mesh_overrides : sequence of tuple, optional
        Refinement boxes `((x0, y0, z0), (x1, y1, z1), max_cell)` or
        `(x0, x1, y0, y1, max_cell)` [m]. Converted into local quadtree
        patches when `refine` is not given.
    refine : callable or sequence of tuple, optional
        Either a `QuadMesh` predicate `refine(level, x0, x1, y0, y1) -> bool`
        or a list of `(x0, x1, y0, y1[, level])` boxes passed to `refine_boxes`.
    ratio : int, optional
        Refinement ratio per level (2 or 3). Chosen automatically when
        `mesh_overrides` is used and `ratio` is None; otherwise defaults to 2.
    max_level : int, optional
        Maximum refinement level (`0` = uniform coarse grid). Chosen
        automatically from `refine` or `mesh_overrides` when None.
    n_sub : int, optional
        Sub-pixel samples per direction in cut cells. Default 8.
    lts : int, optional
        Local time stepping sub-step ratio `p` passed to `SubgridFIT2D`.
    lts_kw : dict, optional
        Extra keyword arguments for `wakis.unstructured.lts.build_lts`.
    field_wavelengths : sequence of float, optional
        Wavelengths [m] at which 2D `(Ex, Ey)` DFT maps are recorded when
        `collect_fields=True`. Default `(1550e-9,)`.
    record_movie : bool, optional
        Record downsampled `Ey` time-domain frames. Default False.
    movie_frames : int, optional
        Maximum number of movie frames. Default 50.
    use_gpu : bool, optional
        Accepted for API compatibility (CPU only). Default False.
    """

    def __init__(
        self,
        wavelength_min=1500e-9,
        wavelength_max=1600e-9,
        structures=(),
        ports=(),
        n_core=PIC_MATERIALS["Si"],
        n_clad=PIC_MATERIALS["SiO2"],
        n_sub=None,
        wavelength_ref=1550e-9,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=20,
        n_freqs=51,
        domain=None,
        padding=None,
        n_pml=10,
        cfln=0.5,
        cfl=None,
        mesh_overrides=(),
        refine=None,
        ratio=None,
        max_level=None,
        n_sub_samples=8,
        lts=None,
        lts_kw=None,
        pml_R0=1e-8,
        pml_order=3,
        field_wavelengths=(1550e-9,),
        record_movie=False,
        movie_frames=50,
        use_gpu=False,
    ):
        if not 0 < wavelength_min < wavelength_max:
            raise ValueError("Require 0 < wavelength_min < wavelength_max.")
        if not two_d:
            raise ValueError("SubgridPICSimulation only supports 2D EIM (`two_d=True`).")

        self.wavelength_min = float(wavelength_min)
        self.wavelength_max = float(wavelength_max)
        self.wavelength_ref = float(wavelength_ref)
        self.f_min = c_light / self.wavelength_max
        self.f_max = c_light / self.wavelength_min
        self.f_center = 0.5 * (self.f_min + self.f_max)
        self.frequencies = np.linspace(self.f_min, self.f_max, int(n_freqs))
        self.wavelengths = c_light / self.frequencies
        self.pulse = GaussianSource(
            frequency=self.f_center, fwidth=0.5 * (self.f_max - self.f_min)
        )

        self.structures_3d = list(structures)
        self.ports = list(ports)
        self.port_map = {p.name: p for p in self.ports}
        self.n_core = float(n_core)
        self.n_clad = float(n_clad)
        self.n_sub = float(n_clad if n_sub is None else n_sub)
        self.two_d = True
        self.averaging = averaging
        self.cells_per_wavelength = float(cells_per_wavelength)
        self.n_freqs = int(n_freqs)
        self.padding = 0.8e-6 if padding is None else float(padding)
        self.n_pml = int(n_pml) if n_pml is not None else None
        self.cfln = float(cfln)
        self.cfl = (0.8 if lts else 0.9) if cfl is None else float(cfl)
        self.mesh_overrides = list(mesh_overrides)
        self.refine_input = refine
        self.ratio_input = ratio
        self.max_level_input = max_level
        self.n_sub_samples = int(n_sub_samples)
        self.lts = lts
        self.lts_kw = lts_kw
        self.pml_R0 = float(pml_R0)
        self.pml_order = int(pml_order)
        self.field_wavelengths = tuple(float(w) for w in field_wavelengths)
        self.record_movie = bool(record_movie)
        self.movie_frames = min(int(movie_frames), 60)
        self.use_gpu = bool(use_gpu)

        z_mins = [s.bounds[0][2] for s in self.structures_3d if hasattr(s, "bounds")]
        z_maxs = [s.bounds[1][2] for s in self.structures_3d if hasattr(s, "bounds")]
        self.z_min = min(z_mins) if z_mins else -110e-9
        self.z_max = max(z_maxs) if z_maxs else 110e-9
        self.thickness = self.z_max - self.z_min
        self.n_eff_slab = slab_neff(
            self.n_core,
            self.n_clad,
            self.thickness,
            self.wavelength_ref,
            n_sub=self.n_sub,
            polarization="TE",
        )
        self._domain_input = domain
        self.mesh = self.geometry = self.solver = self.grid = None
        self.dft_fields = {}
        self.movie_data = None
        self.port_time_trace = None
        self.flux_balance = None
        self.timing = {}
        self.build()

    def _interior_domain(self):
        if self._domain_input is not None:
            xmin, xmax, ymin, ymax = [float(v) for v in self._domain_input[:4]]
        else:
            all_lo = np.min([s.bounds[0] for s in self.structures_3d], axis=0)
            all_hi = np.max([s.bounds[1] for s in self.structures_3d], axis=0)
            xmin, xmax = float(all_lo[0]), float(all_hi[0])
            ymin, ymax = (
                float(all_lo[1]) - self.padding,
                float(all_hi[1]) + self.padding,
            )
        return xmin, xmax, ymin, ymax

    def _resolve_refinement(self, dx, dy, interior_box):
        xmin, xmax, ymin, ymax = interior_box
        # Keep refined patches inside the non-PML interior (with a 2-cell
        # buffer when LTS is active so LTS fine/buffer nodes never enter PML)
        pad_x = 2.0 * dx if self.lts else 1e-6 * dx
        pad_y = 2.0 * dy if self.lts else 1e-6 * dy
        clip_box = (xmin + pad_x, xmax - pad_x, ymin + pad_y, ymax - pad_y)

        if self.max_level_input is not None and int(self.max_level_input) == 0:
            r = 2 if self.ratio_input is None else int(self.ratio_input)
            return r, 0, None

        if self.refine_input is not None:
            if callable(self.refine_input):
                raw_fn = self.refine_input
                r = 2 if self.ratio_input is None else int(self.ratio_input)
                ml = 1 if self.max_level_input is None else int(self.max_level_input)
            else:
                boxes = list(self.refine_input)
                raw_fn = refine_boxes(boxes)
                r = 2 if self.ratio_input is None else int(self.ratio_input)
                if self.max_level_input is not None:
                    ml = int(self.max_level_input)
                else:
                    levs = [b[4] for b in boxes if len(b) >= 5 and b[4] is not None]
                    ml = max(levs) if levs else 1
        elif self.mesh_overrides:
            allowed = (
                (int(self.ratio_input),)
                if self.ratio_input is not None
                else (2, 3)
            )
            plan = choose_refinement(
                max(dx, dy),
                self.mesh_overrides,
                allowed_ratios=allowed,
                clip_box=clip_box,
            )
            r = plan.ratio
            ml = (
                plan.max_level
                if self.max_level_input is None
                else int(self.max_level_input)
            )
            raw_fn = plan.refine
        else:
            r = 2 if self.ratio_input is None else int(self.ratio_input)
            return r, 0, None

        if raw_fn is None or ml <= 0:
            return r, 0, None

        cx0, cx1, cy0, cy1 = clip_box

        def clipped_refine(level, x0, x1, y0, y1):
            in_interior = (
                (x0 >= cx0 - 1e-12)
                & (x1 <= cx1 + 1e-12)
                & (y0 >= cy0 - 1e-12)
                & (y1 <= cy1 + 1e-12)
            )
            return in_interior & np.asarray(
                raw_fn(level, x0, x1, y0, y1), dtype=bool
            )

        return r, ml, clipped_refine

    def build(self):
        """Build the `QuadMesh`, `PolySlabGeometry`, and `SubgridFIT2D` solver."""
        t0 = time.perf_counter()
        xmin, xmax, ymin, ymax = self._interior_domain()
        self.interior_box = (xmin, xmax, ymin, ymax)

        h_core = meshing.cell_size(
            self.f_max, self.n_eff_slab**2, self.cells_per_wavelength
        )
        h_bg = meshing.cell_size(self.f_max, 1.0, self.cells_per_wavelength)
        n_pml = self.n_pml or meshing.pml_cells(self.f_min, h_bg)
        self.n_pml = int(n_pml)

        nx_int = max(1, int(np.ceil((xmax - xmin) / h_core - 1e-6)))
        ny_int = max(1, int(np.ceil((ymax - ymin) / h_core - 1e-6)))
        x_int = np.linspace(xmin, xmax, nx_int + 1)
        y_int = np.linspace(ymin, ymax, ny_int + 1)
        dx = float(x_int[1] - x_int[0])
        dy = float(y_int[1] - y_int[0])

        xc = np.concatenate(
            [
                x_int[0] - dx * np.arange(n_pml, 0, -1),
                x_int,
                x_int[-1] + dx * np.arange(1, n_pml + 1),
            ]
        )
        yc = np.concatenate(
            [
                y_int[0] - dy * np.arange(n_pml, 0, -1),
                y_int,
                y_int[-1] + dy * np.arange(1, n_pml + 1),
            ]
        )
        self._x_int, self._y_int = x_int, y_int

        ratio, max_level, refine_fn = self._resolve_refinement(
            dx, dy, self.interior_box
        )
        self.ratio = ratio
        self.max_level = max_level

        self.mesh = QuadMesh(
            xc, yc, ratio=ratio, max_level=max_level, refine=refine_fn, balance=True
        )
        z0 = 0.5 * (self.z_min + self.z_max)
        self.geometry = PolySlabGeometry(
            self.structures_3d,
            background_eps=n_to_eps(self.n_clad),
            z=z0,
            eim_map=(n_to_eps(self.n_core), n_to_eps(self.n_eff_slab)),
            frequency=self.f_center,
        )
        pml_cfg = dict(
            box=self.interior_box, R0=self.pml_R0, order=self.pml_order
        )
        self.solver = SubgridFIT2D(
            self.mesh,
            geometry=self.geometry,
            polarization=PIC_POLARIZATION,
            pml=pml_cfg,
            cfl=self.cfl,
            n_sub=self.n_sub_samples,
            lts=self.lts,
            lts_kw=self.lts_kw,
        )

        # Lightweight grid descriptor matching the attributes accessed on
        # `PICSimulation.grid` by tutorials and plotting helpers
        self.grid = SimpleNamespace(
            x=xc,
            y=yc,
            z=np.array([self.z_min, self.z_max]),
            dx=np.diff(xc),
            dy=np.diff(yc),
            dz=np.array([self.thickness]),
            Nx=len(xc) - 1,
            Ny=len(yc) - 1,
            Nz=1,
            xmin=float(xc[0]),
            xmax=float(xc[-1]),
            ymin=float(yc[0]),
            ymax=float(yc[-1]),
            zmin=float(self.z_min),
            zmax=float(self.z_max),
        )
        setup_time = time.perf_counter() - t0
        self.timing = {
            "n_cells": int(self.mesh.n_cells),
            "n_nodes": int(self.mesh.n_nodes),
            "n_dof": int(self.solver.n_dof),
            "n_edges": int(self.solver.n_edges),
            "n_unknowns": int(self.solver.n_unknowns),
            "level_counts": self.mesh.level_counts().tolist(),
            "ratio": int(self.mesh.ratio),
            "max_level": int(self.mesh.max_level),
            "dt": float(self.solver.dt),
            "n_steps": 0,
            "setup_time": float(setup_time),
            "wall_time": 0.0,
            "peak_rss_mb": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
            "lts": self.lts,
        }
        return self.solver

    def _port_plane_args(self, port):
        center = np.asarray(port.center, dtype=float).copy()
        size = np.asarray(port.size, dtype=float).copy()
        a = port.axis
        mid_a = 0.5 * (
            (self.grid.xmin + self.grid.xmax)
            if a == 0
            else (self.grid.ymin + self.grid.ymax)
        )
        inward = 1 if center[a] <= mid_a else -1
        return center, size, inward

    def solve_port_modes(self, port_name, frequencies=None, n_modes=1):
        """Solve the 1D eigenmodes on `port_name` at `frequencies` [Hz]."""
        port = self.port_map[port_name]
        center, size, _ = self._port_plane_args(port)
        nm = max(int(n_modes), port.mode_index + 1)
        if frequencies is None:
            return solve_port_modes_2d(
                self.solver, self.f_center, center, size, n_modes=nm
            )
        if np.ndim(frequencies) == 0:
            return solve_port_modes_2d(
                self.solver, float(frequencies), center, size, n_modes=nm
            )
        return [
            solve_port_modes_2d(self.solver, float(f), center, size, n_modes=nm)
            for f in frequencies
        ]

    def run_s_parameters(
        self, drive=None, energy_decay=1e-4, max_time=None, collect_fields=True
    ):
        """
        Run one time-domain simulation per driven port and return `S[(out, in)](f)`.

        Normalized by the forward modal amplitude measured at the driven port's
        own `SubgridModeMonitor`, matching `PICSimulation.run_s_parameters`.
        Also records run statistics in `self.timing` (`summary()`).
        """
        if not self.ports:
            raise ValueError("No ports defined on SubgridPICSimulation.")
        if drive is None:
            drive = [self.ports[0].name]
        elif isinstance(drive, str):
            drive = [p.name for p in self.ports] if drive == "all" else [drive]

        if max_time is None:
            lx = self.grid.xmax - self.grid.xmin
            ly = self.grid.ymax - self.grid.ymin
            transit = np.hypot(lx, ly) * self.n_core / c_light
            max_time = self.pulse.end_time + 15.0 * transit

        s_matrix = {}
        t_wall_start = time.perf_counter()
        total_steps = 0

        for run_idx, in_name in enumerate(drive):
            in_port = self.port_map[in_name]
            in_center, in_size, in_inward = self._port_plane_args(in_port)
            a_in = in_port.axis
            nodes_a = self.grid.x if a_in == 0 else self.grid.y
            k_mon = int(np.argmin(np.abs(nodes_a[:-1] - in_center[a_in])))
            k_src = int(np.clip(k_mon - 2 * in_inward, 2, len(nodes_a) - 3))
            src_center = in_center.copy()
            src_center[a_in] = nodes_a[k_src]

            self.solver.clear_currents()
            self.solver.reset()

            src = SubgridModeSource(
                self.pulse,
                center=src_center,
                size=in_size,
                direction="+" if in_inward > 0 else "-",
                mode_index=in_port.mode_index,
                mode_frequency=self.f_center,
            ).register(self.solver)

            port_monitors, monitors_list = {}, []
            for p in self.ports:
                p_center, p_size, _ = self._port_plane_args(p)
                mon = SubgridModeMonitor(
                    self.frequencies,
                    center=p_center,
                    size=p_size,
                    n_modes=p.mode_index + 1,
                    mode_frequency=self.f_center,
                )
                port_monitors[p.name] = mon
                monitors_list.append(mon)

            dft_mon = flux_planes = None
            probe_My = None
            movie_My = None
            x_dft = y_dft = None
            trace_t, trace_val = [], []
            movie_frames_list, movie_times = [], []

            if collect_fields and run_idx == 0:
                f_dft = [c_light / w for w in self.field_wavelengths]
                stride = max(
                    1, int(np.ceil(max(self.grid.Nx, self.grid.Ny) / 400))
                )
                xc_mid = 0.5 * (self.grid.x[:-1] + self.grid.x[1:])
                yc_mid = 0.5 * (self.grid.y[:-1] + self.grid.y[1:])
                x_dft = xc_mid[::stride].astype(np.float32)
                y_dft = yc_mid[::stride].astype(np.float32)
                dft_mon = SubgridDFTMonitor(
                    f_dft,
                    points=(x_dft.astype(float), y_dft.astype(float)),
                    components=("Ex", "Ey"),
                )
                monitors_list.append(dft_mon)

                trace_port = self.ports[1] if len(self.ports) > 1 else in_port
                tp_c, _, _ = self._port_plane_args(trace_port)
                _, probe_My = self.solver.edge_field_matrices(
                    np.array([[tp_c[0], tp_c[1]]])
                )
                if self.record_movie and self.movie_frames > 0:
                    Xg, Yg = np.meshgrid(
                        x_dft.astype(float), y_dft.astype(float), indexing="ij"
                    )
                    _, movie_My = self.solver.edge_field_matrices(
                        np.column_stack([Xg.ravel(), Yg.ravel()])
                    )

                n_p = self.n_pml + 1
                xw = float(self.grid.x[max(1, k_src - 2)])
                xe = float(self.grid.x[-1 - n_p])
                ys = float(self.grid.y[n_p])
                yn = float(self.grid.y[-1 - n_p])
                xc_b, yc_b = 0.5 * (xw + xe), 0.5 * (ys + yn)
                lx_b, ly_b = xe - xw, yn - ys
                flux_planes = {
                    "west": SubgridFluxMonitor(
                        self.frequencies, (xw, yc_b), (0.0, ly_b)
                    ),
                    "east": SubgridFluxMonitor(
                        self.frequencies, (xe, yc_b), (0.0, ly_b)
                    ),
                    "south": SubgridFluxMonitor(
                        self.frequencies, (xc_b, ys), (lx_b, 0.0)
                    ),
                    "north": SubgridFluxMonitor(
                        self.frequencies, (xc_b, yn), (lx_b, 0.0)
                    ),
                }
                monitors_list.extend(flux_planes.values())

            dt = self.solver.dt
            Nt = int(np.ceil(max_time / dt))
            t_off = float(src.end_time)
            energy_max = 0.0
            check_every = 50
            frame_interval = (
                max(1, Nt // max(1, self.movie_frames))
                if (self.record_movie and self.movie_frames > 0)
                else Nt + 1
            )

            steps_run = Nt
            for n in range(Nt):
                t = n * dt
                self.solver.one_step()
                for mon in monitors_list:
                    mon.update(self.solver, t)
                if probe_My is not None:
                    trace_t.append(float(t + 0.5 * dt))
                    trace_val.append(float((probe_My @ self.solver.g)[0]))
                if (
                    movie_My is not None
                    and n % frame_interval == 0
                    and len(movie_frames_list) < self.movie_frames
                ):
                    frame = (movie_My @ self.solver.g).reshape(
                        x_dft.size, y_dft.size
                    )
                    movie_frames_list.append(frame.astype(np.float32))
                    movie_times.append(float(t + 0.5 * dt))

                if energy_decay is not None and (n + 1) % check_every == 0:
                    e_cur = float(np.abs(self.solver.energy()))
                    energy_max = max(energy_max, e_cur)
                    if (
                        (n + 1) * dt > t_off
                        and energy_max > 0.0
                        and e_cur < energy_decay * energy_max
                    ):
                        steps_run = n + 1
                        break

            total_steps += steps_run

            a_plus_in, a_minus_in = port_monitors[in_name].get_amplitudes(
                in_port.mode_index
            )
            a_inc = a_plus_in if in_inward > 0 else a_minus_in

            for p in self.ports:
                _, _, p_inward = self._port_plane_args(p)
                a_p, a_m = port_monitors[p.name].get_amplitudes(p.mode_index)
                b_out = a_m if p_inward > 0 else a_p
                s_matrix[(p.name, in_name)] = b_out / a_inc

            if dft_mon is not None:
                ex_raw, _ = dft_mon.get_field("Ex")
                ey_raw, _ = dft_mon.get_field("Ey")
                for idx_w, w_val in enumerate(self.field_wavelengths):
                    ex = ex_raw[idx_w].astype(np.complex64)
                    ey = ey_raw[idx_w].astype(np.complex64)
                    self.dft_fields[w_val] = {
                        "x": x_dft,
                        "y": y_dft,
                        "Ex": ex,
                        "Ey": ey,
                        "E2": (np.abs(ex) ** 2 + np.abs(ey) ** 2).astype(
                            np.float32
                        ),
                    }

            if probe_My is not None:
                self.port_time_trace = {
                    "t": np.asarray(trace_t, dtype=np.float32),
                    "Ey": np.asarray(trace_val, dtype=np.float32),
                }
                if self.record_movie and movie_frames_list:
                    frames_f32 = np.stack(movie_frames_list, axis=0)
                    peak = max(float(np.max(np.abs(frames_f32))), 1e-30)
                    self.movie_data = {
                        "x": x_dft,
                        "y": y_dft,
                        "times": np.asarray(movie_times, dtype=np.float32),
                        "frames": (frames_f32 / peak).astype(np.float16),
                        "peak": np.float32(peak),
                    }

            if flux_planes is not None:
                p_inc = 2.0 * np.abs(a_inc) ** 2
                f_east = flux_planes["east"].get_flux() / p_inc
                f_west = -flux_planes["west"].get_flux() / p_inc
                f_north = flux_planes["north"].get_flux() / p_inc
                f_south = -flux_planes["south"].get_flux() / p_inc
                self.flux_balance = {
                    "east": f_east,
                    "west": f_west,
                    "lateral": f_north + f_south,
                    "total_box": f_east + f_west + f_north + f_south,
                }

        wall_time = time.perf_counter() - t_wall_start
        self.timing.update(
            n_steps=int(total_steps),
            wall_time=float(wall_time),
            peak_rss_mb=float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
        )
        return s_matrix

    def summary(self):
        """Return mesh, timestep, and run-cost statistics as a dict."""
        return dict(self.timing)
