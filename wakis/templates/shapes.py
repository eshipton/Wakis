# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Simple solids (boxes, cylinders) with a material, assigned to the E edges of
the FIT grid.

Materials are

* ``'pec'``: perfect electric conductor. All E edges inside the closed solid
  (faces included) are removed, so a box of zero thickness is a PEC sheet and
  a cylinder of zero radius a PEC wire along one line of edges;
* a number: relative permittivity of a lossless dielectric;
* a key of `wakis.materials.dispersive_lib` (e.g. ``'fr4'``, ``'silicon'``) or
  a `DispersiveMaterial`: dispersive dielectric (boxes only). The interior
  edges use the ADE model of `SolverFIT3D.add_dispersive_material`; the edges
  on the faces use the real permittivity at the band centre.

Dielectric edges inside get the material; edges lying on a face get the
arithmetic mean with what was there before (weight 1/2 per face), the correct
average for E tangential to an interface. Later solids override earlier ones.

The assignment works on the edges directly instead of importing STL files
because the STL cell masks of `GridFIT3D` attach a whole cell to the three
edges leaving its lower corner, which cannot represent zero-thickness metal
(patches, ground planes, wires) and shifts faces by up to one cell.
`to_pyvista()` gives the surface for plotting or STL export.
"""

import numpy as np
import pyvista as pv
from scipy.constants import epsilon_0

from ..dispersive import DispersiveMaterial, dispersive_lib


class Box:
    def __init__(self, center, size, material):
        """
        Axis-aligned box.

        Parameters
        ----------
        center, size : tuple of float
            Centre and extent (x, y, z) [m]. A zero extent gives a sheet
            (useful for PEC).
        material : str, float or DispersiveMaterial
            See the module docstring.
        """
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        if np.any(self.size < 0):
            raise ValueError("size must be non-negative.")
        self.material = resolve_material(material)

    @property
    def bounds(self):
        """(lo, hi) corners [m], each an array of 3."""
        return self.center - self.size / 2, self.center + self.size / 2

    def faces(self, axis):
        """Face coordinates along `axis` (0, 1, 2) [m], used to snap the mesh."""
        lo, hi = self.bounds
        return [lo[axis], hi[axis]]

    def weight(self, x, y, z, tol):
        """
        Fill weight of the samples at the broadcast coordinates (x, y, z):
        1 inside, 1/2 per face the sample lies on, 0 outside.
        """
        w = 1.0
        for pos, c, s in zip((x, y, z), self.center, self.size):
            dist = s / 2 - np.abs(pos - c)
            w = w * np.where(dist > tol, 1.0, np.where(dist > -tol, 0.5, 0.0))
        return w

    def to_pyvista(self):
        """Surface as `pyvista.PolyData` [m]."""
        lo, hi = self.bounds
        return pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))


class Cylinder:
    def __init__(self, center, radius, length, material, axis="z"):
        """
        Circular cylinder along a coordinate axis.

        Parameters
        ----------
        center : tuple of float
            Centre (x, y, z) [m].
        radius : float
            Radius [m]. Zero gives a line of edges (thin PEC wire).
        length : float
            Length along `axis` [m].
        material : str, float or DispersiveMaterial
            See the module docstring. Dispersive cylinders are not supported.
        axis : {'x', 'y', 'z'}, optional
            Cylinder axis. Default 'z'.
        """
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.length = float(length)
        self.axis = "xyz".index(axis)
        self.material = resolve_material(material)
        if isinstance(self.material, DispersiveMaterial):
            raise ValueError("Dispersive cylinders are not supported; use a Box.")

    @property
    def bounds(self):
        half = np.full(3, self.radius)
        half[self.axis] = self.length / 2
        return self.center - half, self.center + half

    def faces(self, axis):
        lo, hi = self.bounds
        if axis == self.axis or self.radius == 0:
            return [lo[axis], hi[axis]] if axis == self.axis else [self.center[axis]]
        return [lo[axis], self.center[axis], hi[axis]]

    def weight(self, x, y, z, tol):
        pos = (x, y, z)
        a, b, c = self.axis, (self.axis + 1) % 3, (self.axis + 2) % 3
        rho = np.hypot(pos[b] - self.center[b], pos[c] - self.center[c])
        radial = np.where(
            rho < self.radius - tol,
            1.0,
            np.where(rho < self.radius + tol, 0.5, 0.0),
        )
        dist = self.length / 2 - np.abs(pos[a] - self.center[a])
        axial = np.where(dist > tol, 1.0, np.where(dist > -tol, 0.5, 0.0))
        return radial * axial

    def to_pyvista(self):
        direction = np.zeros(3)
        direction[self.axis] = 1.0
        if self.radius == 0:
            return pv.Line(
                self.center - direction * self.length / 2,
                self.center + direction * self.length / 2,
            )
        return pv.Cylinder(
            center=self.center,
            direction=direction,
            radius=self.radius,
            height=self.length,
        )


class Sheet:
    def __init__(self, center, size):
        """
        Zero-thickness rectangular PEC sheet with conformal edges.

        Unlike a PEC `Box` of zero thickness, the sheet edges need not lie on
        mesh nodes: the grid only gets a node plane at the sheet, and
        `GridFIT3D(sheets=...)` covers the cut edges and faces partially and
        corrects the knife-edge field (see "Thin PEC sheets" in the physics
        guide), so the sheet acts at its physical size.

        Parameters
        ----------
        center, size : tuple of float
            Centre and extent (x, y, z) [m]; exactly one extent is zero (the
            sheet normal).
        """
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        if np.count_nonzero(self.size == 0) != 1 or np.any(self.size < 0):
            raise ValueError("A Sheet needs exactly one zero extent (its normal).")
        self.normal = int(np.flatnonzero(self.size == 0)[0])
        self.material = "pec"

    @property
    def bounds(self):
        """(lo, hi) corners [m], each an array of 3."""
        return self.center - self.size / 2, self.center + self.size / 2

    def faces(self, axis):
        """Only the sheet plane is snapped to the mesh, not its edges."""
        return [self.center[axis]] if axis == self.normal else []

    def to_pyvista(self):
        """Surface as `pyvista.PolyData` [m]."""
        from ..conformal import rectangle_sheet

        return rectangle_sheet(self.center, self.size)


def resolve_material(material):
    """
    Normalize a material: 'pec', a float eps_r or a DispersiveMaterial
    (library keys of `dispersive_lib` are looked up).
    """
    if isinstance(material, DispersiveMaterial):
        return material
    if isinstance(material, str):
        key = material.lower()
        if key == "pec":
            return "pec"
        if key in dispersive_lib:
            return dispersive_lib[key]
        raise ValueError(
            f"Unknown material {material!r}: use 'pec', a relative permittivity "
            f"or one of {sorted(dispersive_lib)}."
        )
    eps_r = float(material)
    if eps_r < 1:
        raise ValueError("Relative permittivity must be >= 1.")
    return eps_r


def material_eps(material, frequency):
    """Real relative permittivity of a material at `frequency` [Hz] (inf: PEC)."""
    if isinstance(material, str):
        return np.inf
    if isinstance(material, DispersiveMaterial):
        return float(np.real(material.epsilon(frequency)))
    return material


def assign_materials(solver, structures, frequency):
    """
    Write the structures into `solver.ieps` and register dispersive boxes.

    Parameters
    ----------
    solver : SolverFIT3D
        Freshly built solver (before the first time step).
    structures : list of Box, Cylinder or Sheet
        Solids in order of increasing priority. Sheets are skipped: they are
        part of the grid (`GridFIT3D(sheets=...)`).
    frequency : float
        Frequency [Hz] at which the real permittivity of dispersive materials
        is written into `ieps`. The interior edges of dispersive boxes are
        then stepped with the ADE model (the `ieps` value only feeds the mode
        solver); their face edges stay non-dispersive at this frequency.
    """
    grid = solver.grid
    nodes = (grid.x, grid.y, grid.z)
    tol = 1e-6 * min(np.min(grid.dx), np.min(grid.dy), np.min(grid.dz))
    ieps = np.asarray(_host(solver.ieps.toarray()), dtype=float).copy()
    # Conformal sheets (grid.sheets) scale 1/eps on their edges: average the
    # materials without that factor and apply it to the result
    factor = getattr(solver, "conformal_ieps_factor", None)
    if factor is not None:
        ieps = np.where(factor > 0, ieps / np.where(factor > 0, factor, 1.0), 0.0)
    structures = [s for s in structures if not isinstance(s, Sheet)]
    eps = np.where(ieps == 0, np.inf, 1 / np.where(ieps == 0, 1, ieps) / epsilon_0)
    N = solver.N
    # Periodic images: a face on one cell boundary is also on the opposite one
    shifts = [
        (0.0, -(n[-1] - n[0]), n[-1] - n[0])
        if str(solver.bc_low[d]).lower() in ("periodic", "bloch")
        else (0.0,)
        for d, n in enumerate(nodes)
    ]
    for comp in range(3):
        pos = [
            (n[:-1] + n[1:]) / 2 if d == comp else n[:-1] for d, n in enumerate(nodes)
        ]
        X, Y, Z = np.meshgrid(*pos, indexing="ij")
        block = eps[comp * N : (comp + 1) * N].reshape(X.shape, order="F")
        for shape in structures:
            w = np.broadcast_to(
                _periodic_weight(shape, (X, Y, Z), shifts, tol), X.shape
            )
            if shape.material == "pec":
                block[w > 0] = np.inf
                continue
            eps_shape = material_eps(shape.material, frequency)
            full, part = w == 1, (w > 0) & (w < 1) & np.isfinite(block)
            block[full] = eps_shape
            block[part] = (1 - w[part]) * block[part] + w[part] * eps_shape
        eps[comp * N : (comp + 1) * N] = block.ravel(order="F")

    ieps = np.where(np.isinf(eps), 0.0, 1 / (epsilon_0 * eps))
    if factor is not None:
        ieps = ieps * factor
    solver.ieps.fromarray(solver.ieps.xp.asarray(ieps) if solver.use_gpu else ieps)
    if not solver.activate_cpml:
        # Without CPML the step uses operators precomputed from ieps
        if solver.use_gpu or solver.imported_mkl:
            raise NotImplementedError(
                "Structures without CPML faces are only supported on the plain "
                "NumPy/SciPy backend."
            )
        solver.update_tensors("ieps")
        solver.step_0 = True  # update_tensors skips the first-step setup
    for shape in structures:
        if isinstance(shape.material, DispersiveMaterial):
            _add_dispersive_interior(solver, shape, shifts)


def _add_dispersive_interior(solver, box, shifts):
    """
    ADE update on the interior edges of a dispersive box. The edges on its
    faces keep the averaged permittivity at the reference frequency written
    in `ieps` (non-dispersive), so that `ieps` describes the structure
    consistently for the mode solver; the ADE update needs the value in
    `ieps` on its edges to be the background it mixes with, which is only
    true in the interior.
    """
    cells = (solver.grid.dx, solver.grid.dy, solver.grid.dz)
    size = box.size.copy()
    for d, shift in enumerate(shifts):
        if max(shift) > 0 and size[d] >= max(shift) * (1 - 1e-9):
            size[d] = 3 * max(shift)  # whole period: no faces
        elif size[d] > 0:
            # Move the faces off the face nodes, a small fraction of a cell
            size[d] -= 2e-3 * np.min(cells[d])
    solver.add_dispersive_material(box.material, box.center, size)


def _periodic_weight(shape, pos, shifts, tol):
    """
    Fill weight with periodic images: a solid spanning a whole period fills
    that axis; a smaller one is summed over its images, so that a face on one
    cell boundary is also seen on the opposite one.
    """
    pos, shifts = list(pos), list(shifts)
    lo, hi = shape.bounds
    for d, shift in enumerate(shifts):
        period = max(shift)
        if period > 0 and hi[d] - lo[d] >= period - tol:
            pos[d] = shape.center[d]
            shifts[d] = (0.0,)
    w = 0.0
    for sx in shifts[0]:
        for sy in shifts[1]:
            for sz in shifts[2]:
                w = w + shape.weight(pos[0] + sx, pos[1] + sy, pos[2] + sz, tol)
    return np.minimum(w, 1.0)


def _host(array):
    return array.get() if hasattr(array, "get") else array
