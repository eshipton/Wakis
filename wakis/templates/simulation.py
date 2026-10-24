# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
`Simulation`: a declarative wrapper that builds GridFIT3D + SolverFIT3D from
solids, sources and monitors, chooses the mesh, padding, CPML thickness and run
length automatically, runs, and gives access to the results.
"""

import copy

import numpy as np
import pyvista as pv
from scipy.constants import c as c_light

from ..gridFIT3D import GridFIT3D
from ..lumped import LumpedElement, LumpedPort
from ..solverFIT3D import SolverFIT3D
from ..sources import GaussianSource
from . import meshing
from .shapes import Sheet, assign_materials, material_eps


class Simulation:
    def __init__(
        self,
        f_min,
        f_max,
        structures=(),
        boundaries="cpml",
        domain=None,
        cells_per_wavelength=20,
        min_cells_per_feature=2,
        max_cell=None,
        mesh_overrides=(),
        padding=None,
        n_pml=None,
        grading=1.4,
        n_freqs=101,
        cfln=0.5,
        dtype=np.float64,
        k_bloch=None,
        use_gpu=False,
        averaging="kottke",
    ):
        """
        Time-domain simulation over the band [f_min, f_max] with automatic setup.

        Add excitations with `add_source` (ports, current or mode sources,
        normally driven by `self.pulse`) and named monitors with `add_monitor`,
        then call `run()`.

        Parameters
        ----------
        f_min, f_max : float
            Band of interest [Hz]. The excitation `pulse` covers it and the
            spectra are evaluated on `frequencies`.
        structures : list of Box or Cylinder, optional
            Solids, in order of increasing priority (see `shapes`).
        boundaries : str or (list, list), optional
            One boundary for all faces ('cpml', 'pec', 'pmc', 'periodic') or
            the (bc_low, bc_high) lists of SolverFIT3D. Default 'cpml'.
        domain : tuple of 6 float, optional
            (xmin, xmax, ymin, ymax, zmin, zmax) [m] of the region inside the
            CPML; None entries (or None) are chosen automatically: the
            bounding box of the structures and ports plus `padding` towards
            CPML faces. Required along periodic axes (the period) and for PEC
            walls without structures.
        cells_per_wavelength : float, optional
            Cells per wavelength at f_max in each material. Default 20.
        min_cells_per_feature : int, optional
            Minimum number of cells between two consecutive structure faces
            (e.g. across a substrate). Default 2.
        max_cell : float, optional
            Upper bound on the cell size [m].
        mesh_overrides : list of (lo, hi, max_cell), optional
            Refinement boxes: `lo`, `hi` are (x, y, z) corners [m] (use
            +-inf to span an axis) and `max_cell` [m] (a float, or one value
            per axis) caps the cell size between them on each axis. The mesh is a tensor product, so a box
            refines the full strip through the domain along each axis; the
            cell sizes grade back out at `grading`. Use them to put several
            cells across features narrower than lambda / cells_per_wavelength
            (coupler gaps, slots, grating teeth).
        padding : float, optional
            Distance from the structures to the CPML [m]. Default lambda/4 at
            f_min.
        n_pml : int, optional
            CPML thickness in cells. Default `meshing.pml_cells`.
        grading : float, optional
            Maximum ratio of adjacent cell sizes. Default 1.4.
        n_freqs : int, optional
            Number of frequencies in `frequencies`. Default 101.
        cfln : float, optional
            CFL number of the time step. Default 0.5.
        dtype : numpy dtype, optional
            Field dtype; complex for Bloch boundaries with k_bloch != 0.
        k_bloch : tuple of float, optional
            Bloch wavevector [rad/m] (with 'bloch' boundaries).
        use_gpu : bool, optional
            Run on GPU (untested for templates).
        averaging : {'kottke', 'arithmetic', 'staircase'}, optional
            Subpixel permittivity averaging rule for shapes with `fill`.
            Default 'kottke'.
        """
        if not 0 < f_min < f_max:
            raise ValueError("Require 0 < f_min < f_max.")
        self.f_min, self.f_max = float(f_min), float(f_max)
        self.f_center = 0.5 * (f_min + f_max)
        self.frequencies = np.linspace(f_min, f_max, n_freqs)
        # Gaussian spectrum at 0.6 of its peak at the band edges
        self.pulse = GaussianSource(frequency=self.f_center, fwidth=(f_max - f_min) / 2)
        self.structures = list(structures)
        if isinstance(boundaries, str):
            boundaries = ([boundaries] * 3, [boundaries] * 3)
        self.bc_low = [b.lower() for b in boundaries[0]]
        self.bc_high = [b.lower() for b in boundaries[1]]
        self.domain = domain
        self.cells_per_wavelength = cells_per_wavelength
        self.min_cells_per_feature = min_cells_per_feature
        self.max_cell = max_cell
        self.mesh_overrides = [
            (
                np.asarray(lo, float),
                np.asarray(hi, float),
                np.broadcast_to(np.asarray(h, float), 3),
            )
            for lo, hi, h in mesh_overrides
        ]
        self.padding = meshing.padding(f_min) if padding is None else padding
        self.n_pml = n_pml
        self.grading = grading
        self.cfln = cfln
        self.dtype = dtype
        self.k_bloch = k_bloch
        self.use_gpu = use_gpu
        self.averaging = averaging
        self.sources = []
        self.monitors = {}
        self.grid = None
        self.solver = None
        self.design = {}  # filled by templates with design-formula values

    # ------------------------------------------------------------------ setup
    def add_source(self, source):
        """Add a source, port or lumped element (anything `emsolve` accepts)."""
        self.sources.append(source)
        return source

    def add_monitor(self, name, monitor):
        """Add a monitor under `name` (DFT, flux, near-to-far, mode monitors)."""
        self.monitors[name] = monitor
        return monitor

    @property
    def background_cell(self):
        """Cell size in vacuum [m]: lambda(f_max) / cells_per_wavelength."""
        h = meshing.cell_size(self.f_max, 1.0, self.cells_per_wavelength)
        return h if self.max_cell is None else min(h, self.max_cell)

    def _snap_points(self, axis):
        """Coordinates that must be mesh nodes: faces and lumped gaps."""
        points = []
        for shape in self.structures:
            points += list(shape.faces(axis))
        for src in self.sources:
            if isinstance(src, LumpedElement):
                c, s = src.center[axis], src.size[axis]
                points += [c - s / 2, c + s / 2] if s > 0 else [c]
        return points

    def _bounding_box(self, axis):
        points = self._snap_points(axis)
        # Shapes with non-snapping edges (Sheet, PolySlab) still define the extent
        points += [
            b[axis] for s in self.structures if not s.faces(axis) for b in s.bounds
        ]
        if not points:
            raise ValueError(
                f"No structure or port along {'xyz'[axis]}: give `domain` explicitly."
            )
        return min(points), max(points)

    def _axis(self, axis):
        lo_bc, hi_bc = self.bc_low[axis], self.bc_high[axis]
        periodic = lo_bc in ("periodic", "bloch")
        h_bg = self.background_cell
        if self.domain is not None and self.domain[2 * axis] is not None:
            lo, hi = self.domain[2 * axis], self.domain[2 * axis + 1]
        else:
            if periodic:
                raise ValueError("Periodic axes need `domain` (the period).")
            lo, hi = self._bounding_box(axis)
            lo -= self.padding if lo_bc == "cpml" else 0.0
            hi += self.padding if hi_bc == "cpml" else 0.0
        points = [lo, hi] + [p for p in self._snap_points(axis) if lo <= p <= hi]
        overrides = [
            (max(o_lo[axis], lo), min(o_hi[axis], hi), h[axis])
            for o_lo, o_hi, h in self.mesh_overrides
            if o_lo[axis] < hi and o_hi[axis] > lo
        ]
        points = meshing.merge_points(points, 1e-6 * h_bg)

        def material_size(a, b):
            mid = 0.5 * (a + b)
            eps = [1.0]
            for shape in self.structures:
                s_lo, s_hi = shape.bounds
                eps_s = material_eps(shape.material, self.f_max)
                overlaps = (
                    s_lo[axis] <= mid <= s_hi[axis]
                    if shape.faces(axis)
                    else (s_lo[axis] < b and s_hi[axis] > a)
                )
                if overlaps and np.isfinite(eps_s):
                    eps.append(eps_s)
            return min(h_bg, h_bg / np.sqrt(max(eps)))

        # Material cell sizes come from the intervals between structure faces;
        # override boxes only split those intervals and cap the size inside
        # them, so they never coarsen the mesh elsewhere.
        base_points = points
        base_sizes = [material_size(a, b) for a, b in zip(points[:-1], points[1:])]
        points = meshing.merge_points(
            list(points) + [p for a, b, _ in overrides for p in (a, b)], 1e-6 * h_bg
        )

        sizes, min_cells = [], []
        struct_lo, struct_hi = self._bounding_box(axis) if self.structures else (lo, hi)
        for a, b in zip(points[:-1], points[1:]):
            mid = 0.5 * (a + b)
            parent = min(
                int(np.searchsorted(base_points, mid)) - 1, len(base_sizes) - 1
            )
            size = base_sizes[max(parent, 0)]
            for o_a, o_b, h in overrides:
                if o_a <= mid <= o_b:
                    size = min(size, h)
            sizes.append(size)
            inside = struct_lo <= mid <= struct_hi
            min_cells.append(self.min_cells_per_feature if inside else 1)
        n_pml = self.n_pml or meshing.pml_cells(self.f_min, h_bg)
        self.n_pml = n_pml
        if periodic:
            # Uniform across the periodic seam; faces are staircased to the mesh
            n = int(np.ceil((hi - lo) / min(sizes) - 1e-6))
            return np.linspace(lo, hi, n + 1)
        nodes = meshing.graded_axis(points, sizes, min_cells, self.grading)
        if lo_bc == "cpml":
            d = nodes[1] - nodes[0]
            nodes = np.concatenate([nodes[0] - d * np.arange(n_pml, 0, -1), nodes])
        # High side: CPML cells (GridFIT3D adds the ghost layer itself, so the
        # wall of a non-periodic axis sits on the last node)
        if hi_bc != "cpml":
            return nodes
        d = nodes[-1] - nodes[-2]
        return np.concatenate([nodes, nodes[-1] + d * np.arange(1, n_pml + 1)])

    def build(self):
        """
        Mesh the domain, create the solver and assign the materials.

        Returns
        -------
        solver : SolverFIT3D
        """
        x, y, z = self._axes = [self._axis(a) for a in range(3)]
        sheets = {
            f"sheet{i}": shape.to_pyvista()
            for i, shape in enumerate(self.structures)
            if isinstance(shape, Sheet)
        }
        self.grid = GridFIT3D(x=x, y=y, z=z, sheets=sheets, verbose=0)
        self.solver = self._new_solver(with_structures=True)
        return self.solver

    def _new_solver(self, with_structures):
        grid = self.grid
        if not with_structures and self.grid.sheets:
            # The empty reference domain without the sheets of the grid
            x, y, z = self._axes
            grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
        solver = SolverFIT3D(
            grid,
            bc_low=self.bc_low,
            bc_high=self.bc_high,
            n_pml=self.n_pml,
            cfln=self.cfln,
            dtype=self.dtype,
            k_bloch=self.k_bloch,
            use_gpu=self.use_gpu,
            verbose=0,
        )
        if with_structures:
            assign_materials(
                solver, self.structures, self.f_center, averaging=self.averaging
            )
        return solver

    # -------------------------------------------------------------------- run
    def run(self, energy_decay=1e-4, max_time=None, reference=False):
        """
        Run until the field energy has decayed (steady-state criterion).

        Parameters
        ----------
        energy_decay : float, optional
            Stop once all sources have ended and the field energy is below
            this fraction of its maximum (1e-4 = -40 dB). None runs the full
            `max_time`. Resonant structures (high Q) need a low value: the
            spectra are converged to about sqrt(energy_decay).
        max_time : float, optional
            Upper limit of the simulated time [s]. Default: pulse duration plus
            300 light-crossing times of the domain.
        reference : bool, optional
            Also run the empty domain (same mesh, sources and monitors, no
            structures), e.g. to normalize transmission spectra. Its monitors
            are returned by `reference_monitor`.

        Returns
        -------
        Simulation
            self, for chaining.
        """
        if self.solver is None or not self.solver.step_0:
            self.build()
        if max_time is None:
            diagonal = np.linalg.norm(
                [self.grid.xmax - self.grid.xmin, self.grid.ymax - self.grid.ymin,
                 self.grid.zmax - self.grid.zmin]
            )  # fmt: skip
            max_time = self.pulse.end_time + 300 * diagonal / c_light
        self.Nt_max = int(np.ceil(max_time / self.solver.dt))
        self.run_sources, self.run_monitors = self._run(
            self.solver, energy_decay, self.Nt_max
        )
        if reference:
            ref_solver = self._new_solver(with_structures=False)
            _, self.reference_monitors = self._run(
                ref_solver, energy_decay, self.Nt_max
            )
        return self

    def _run(self, solver, energy_decay, Nt):
        # Fresh copies, so that a Simulation can be run again (sweeps)
        sources = copy.deepcopy(self.sources)
        monitors = copy.deepcopy(self.monitors)
        solver.emsolve(
            Nt,
            source=sources,
            monitors=list(monitors.values()),
            stop_when_energy_decayed=energy_decay,
        )
        return sources, monitors

    # ---------------------------------------------------------------- results
    @property
    def n_steps(self):
        """Number of time steps of the last run."""
        return self.solver.Nt

    def monitor(self, name):
        """Monitor `name` of the last run."""
        return self.run_monitors[name]

    def reference_monitor(self, name):
        """Monitor `name` of the empty-domain reference run."""
        return self.reference_monitors[name]

    def port(self, index=0):
        """The `index`-th LumpedPort of the last run."""
        ports = [s for s in self.run_sources if isinstance(s, LumpedPort)]
        return ports[index]

    def s11(self, freqs=None, port=0):
        """Reflection coefficient of a lumped port, see `LumpedPort.s11`."""
        freqs = self.frequencies if freqs is None else freqs
        return self.port(port).s11(freqs)

    def input_impedance(self, freqs=None, port=0):
        """Input impedance [Ohm] of a lumped port, see `LumpedPort.impedance_in`."""
        freqs = self.frequencies if freqs is None else freqs
        return self.port(port).impedance_in(freqs)

    def geometry(self):
        """Structures as a `pyvista.MultiBlock` for plotting or STL export."""
        return pv.MultiBlock([shape.to_pyvista() for shape in self.structures])

    def summary(self):
        """Mesh and run parameters as a dict (cells, cell sizes, dt, steps)."""
        grid = self.grid
        info = {
            "cells": (grid.Nx, grid.Ny, grid.Nz),
            "min_cell": (grid.dx.min(), grid.dy.min(), grid.dz.min()),
            "max_cell": (grid.dx.max(), grid.dy.max(), grid.dz.max()),
            "n_pml": self.n_pml,
            "dt": self.solver.dt,
        }
        if hasattr(self.solver, "Nt"):
            info["steps"] = self.solver.Nt
        return info
