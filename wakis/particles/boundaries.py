# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Particle boundary conditions, collectors, PEC/CPML absorption, and beam injection.

Handles:
- Domain boundary wrapping (periodic) or absorption (PEC, Dirichlet, PMC, ABC).
- Absorption at the CPML inner interface (particles never enter the PML region).
- Absorption at solid/PEC cells from the grid mask.
- User-defined :class:`Collector` regions with partial-path Esirkepov
  truncation and recorded absorbed charge, kinetic energy, and current history.
- :class:`BeamInjector` for cathode/emitter-plane particle injection with a
  fractional charge accumulator.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge


def _parse_bc_face(bc_val) -> str:
    """Normalise a Wakis boundary condition string."""
    if bc_val is None:
        return "periodic"
    s = str(bc_val).strip().lower()
    if s in ("periodic", "bloch"):
        return "periodic"
    if s in ("pml", "cpml"):
        return "cpml"
    return s


class Collector:
    """User-defined absorbing collector region or plane.

    Records absorbed charge [C], absorbed kinetic energy [J], macroparticle
    count, and per-step collector current history. When a particle is absorbed
    at the collector entry point ``xi_stop``, Esirkepov current deposition stops
    at ``xi_stop`` and the absorbed charge ``+q`` is added to
    ``solver._particles_q_bg`` at ``xi_stop``, so a collector placed inside the
    vacuum domain retains its collected charge as fixed immobile space charge.

    Parameters
    ----------
    name : str
        Unique identifier for this collector.
    bounds : tuple of 6 float, optional
        Axis-aligned bounding box ``(xmin, xmax, ymin, ymax, zmin, zmax)`` [m].
    plane : tuple of (str or int, float, int), optional
        Absorbing plane ``(axis, position, direction)``, where ``axis`` is
        ``'x'``, ``'y'``, ``'z'`` (or 0, 1, 2), ``position`` is the plane
        coordinate [m], and ``direction`` is ``+1`` (absorbs ``x^{n+1} >= pos``)
        or ``-1`` (absorbs ``x^{n+1} <= pos``).
    mask : callable, optional
        Custom predicate ``mask(x, y, z) -> bool ndarray`` returning True for
        positions inside the collector.
    """

    def __init__(
        self,
        name: str,
        bounds: tuple[float, float, float, float, float, float] | None = None,
        plane: tuple[str | int, float, int] | None = None,
        mask=None,
    ):
        self.name = str(name)
        self.bounds = tuple(float(v) for v in bounds) if bounds is not None else None
        if plane is not None:
            ax_raw, pos, direction = plane
            if isinstance(ax_raw, str):
                ax_map = {"x": 0, "y": 1, "z": 2}
                axis = ax_map[ax_raw.strip().lower()]
            else:
                axis = int(ax_raw)
            self.plane = (axis, float(pos), 1 if int(direction) >= 0 else -1)
        else:
            self.plane = None
        self.mask = mask

        if self.bounds is None and self.plane is None and self.mask is None:
            raise ValueError(
                "Collector requires at least one of bounds, plane, or mask."
            )

        self._absorbed_charge = 0.0
        self._absorbed_energy = 0.0
        self._absorbed_count = 0
        self._parent_boundaries = None
        self.current_history: list[tuple[float, float]] = []

    def _sync_if_needed(self) -> None:
        if self._parent_boundaries is not None and getattr(
            self._parent_boundaries, "_gpu_stats_dirty", False
        ):
            self._parent_boundaries._sync_gpu_stats()

    @property
    def absorbed_charge(self) -> float:
        self._sync_if_needed()
        return self._absorbed_charge

    @absorbed_charge.setter
    def absorbed_charge(self, val: float) -> None:
        self._absorbed_charge = float(val)

    @property
    def absorbed_energy(self) -> float:
        self._sync_if_needed()
        return self._absorbed_energy

    @absorbed_energy.setter
    def absorbed_energy(self, val: float) -> None:
        self._absorbed_energy = float(val)

    @property
    def absorbed_count(self) -> int:
        self._sync_if_needed()
        return self._absorbed_count

    @absorbed_count.setter
    def absorbed_count(self, val: int) -> None:
        self._absorbed_count = int(val)

    def reset(self) -> None:
        """Reset recorded charge, energy, count, and history."""
        self._absorbed_charge = 0.0
        self._absorbed_energy = 0.0
        self._absorbed_count = 0
        self.current_history.clear()

    def check_crossing(self, x0, y0, z0, x1, y1, z1, xp=np):
        """Return ``(hit_mask, alpha)`` for straight segment ``(x0,y0,z0) -> (x1,y1,z1)``.

        ``alpha`` in ``[0, 1]`` is the fractional path parameter up to the
        collector entry point (``1.0`` for particles not hitting the collector).
        """
        n_p = x1.size
        alpha = xp.ones(n_p, dtype=np.float64)
        hit = xp.zeros(n_p, dtype=bool)
        if n_p == 0:
            return hit, alpha

        if self.plane is not None:
            axis, pos, direction = self.plane
            c0 = (x0, y0, z0)[axis]
            c1 = (x1, y1, z1)[axis]
            if direction > 0:
                hit = c1 >= pos
                dc = xp.maximum(c1 - c0, 1e-30)
                a_cross = xp.clip((pos - c0) / dc, 0.0, 1.0)
                alpha = xp.where(hit & (c0 < pos), a_cross, xp.where(hit, 0.0, 1.0))
            else:
                hit = c1 <= pos
                dc = xp.minimum(c1 - c0, -1e-30)
                a_cross = xp.clip((pos - c0) / dc, 0.0, 1.0)
                alpha = xp.where(hit & (c0 > pos), a_cross, xp.where(hit, 0.0, 1.0))
            return hit, alpha

        if self.bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            hit = (
                (x1 >= xmin)
                & (x1 <= xmax)
                & (y1 >= ymin)
                & (y1 <= ymax)
                & (z1 >= zmin)
                & (z1 <= zmax)
            )
            if bool(xp.any(hit)):
                # Slab intersection for particles starting outside the box
                t_enter = xp.zeros(n_p, dtype=np.float64)
                for c0, c1, bmin, bmax in (
                    (x0, x1, xmin, xmax),
                    (y0, y1, ymin, ymax),
                    (z0, z1, zmin, zmax),
                ):
                    dc = c1 - c0
                    enter_low = (c0 < bmin) & (c1 >= bmin)
                    enter_high = (c0 > bmax) & (c1 <= bmax)
                    t_low = xp.where(
                        enter_low, (bmin - c0) / xp.where(enter_low, dc, 1.0), 0.0
                    )
                    t_high = xp.where(
                        enter_high, (bmax - c0) / xp.where(enter_high, dc, -1.0), 0.0
                    )
                    t_enter = xp.maximum(t_enter, xp.maximum(t_low, t_high))
                alpha = xp.where(hit, xp.clip(t_enter, 0.0, 1.0), 1.0)
            return hit, alpha

        hit = xp.asarray(self.mask(x1, y1, z1), dtype=bool)
        alpha = xp.where(hit, 0.5, 1.0)
        return hit, alpha


class ParticleBoundaries:
    """Boundary handler for domain walls, CPML inner faces, PEC solids, and collectors.

    Parameters
    ----------
    grid_map : LogicalGridMap
        Logical-to-physical coordinate mapper.
    bc_low, bc_high : sequence of 3 str, optional
        Boundary conditions on low/high ``(x, y, z)`` faces. Default is all
        ``"periodic"``.
    n_pml : int, optional
        Number of CPML cells on faces with ``"pml"`` or ``"cpml"`` BCs.
        Particles are absorbed at the CPML inner interface so they never
        enter the PML region.
    solid_mask : ndarray of bool, shape (Nx, Ny, Nz), optional
        Boolean mask of solid/PEC cells. Particles entering a solid cell are
        absorbed at the cell interface.
    collectors : sequence of Collector, optional
        Optional user-defined collector regions.
    """

    FACE_NAMES = ("x_low", "x_high", "y_low", "y_high", "z_low", "z_high", "pec")

    def __init__(
        self,
        grid_map,
        bc_low=("periodic", "periodic", "periodic"),
        bc_high=("periodic", "periodic", "periodic"),
        n_pml: int = 10,
        solid_mask=None,
        collectors=None,
    ):
        self.grid_map = grid_map
        self.xp = grid_map.xp
        self.bc_low = tuple(_parse_bc_face(b) for b in bc_low)
        self.bc_high = tuple(_parse_bc_face(b) for b in bc_high)
        self.periodic = tuple(
            self.bc_low[a] == "periodic" and self.bc_high[a] == "periodic"
            for a in range(3)
        )
        self.n_pml = int(n_pml)

        # Effective active logical bounds [xi_min, xi_max] per axis.
        # On CPML faces, particles are absorbed at the CPML inner face.
        # When GridFIT3D(boundary_at_domain_edge=True) appends an inert ghost
        # cell beyond max on non-periodic axes, edge_ghosts[a] == 1 and the
        # physical upper boundary lies at node index counts[a] - 1.
        counts = (grid_map.Nx, grid_map.Ny, grid_map.Nz)
        edge_ghosts = getattr(grid_map, "edge_ghosts", (0, 0, 0))
        xi_min = []
        xi_max = []
        for a in range(3):
            n_phys = counts[a] - (0 if self.periodic[a] else int(edge_ghosts[a]))
            if self.bc_low[a] == "mpi":
                low = -1.0e30
            elif self.bc_low[a] == "cpml":
                low = float(self.n_pml)
            else:
                low = 0.0

            if self.bc_high[a] == "mpi":
                high = 1.0e30
            elif self.bc_high[a] == "cpml":
                high = float(n_phys - self.n_pml)
            else:
                high = float(n_phys)

            if high <= low:
                raise ValueError(
                    f"Active particle region along axis {a} is empty "
                    f"(low={low}, high={high})."
                )
            xi_min.append(low)
            xi_max.append(high)
        self.xi_min = tuple(xi_min)
        self.xi_max = tuple(xi_max)

        if solid_mask is not None:
            sm = self.xp.asarray(solid_mask, dtype=bool)
            if sm.shape != counts:
                raise ValueError(
                    f"solid_mask shape {sm.shape} does not match grid {counts}."
                )
            self.solid_mask = sm if bool(self.xp.any(sm)) else None
        else:
            self.solid_mask = None

        self.collectors: list[Collector] = (
            list(collectors) if collectors is not None else []
        )
        for col in self.collectors:
            col._parent_boundaries = self

        self._absorbed_charge: dict[str, float] = {k: 0.0 for k in self.FACE_NAMES}
        self._absorbed_energy: dict[str, float] = {k: 0.0 for k in self.FACE_NAMES}
        self._absorbed_count: dict[str, int] = {k: 0 for k in self.FACE_NAMES}
        for col in self.collectors:
            self._absorbed_charge[col.name] = 0.0
            self._absorbed_energy[col.name] = 0.0
            self._absorbed_count[col.name] = 0

        self._gpu_stats_dirty = False
        self._last_hit_code = None
        self._d_absorbed_charge = None
        self._d_absorbed_energy = None
        self._d_absorbed_count = None
        self._solid_mask_src = None
        self._solid_mask_f_u8 = None
        self._col_types_gpu = None
        self._col_params_gpu = None
        if self.xp is not np:
            self._ensure_gpu_buffers()

    def _ensure_gpu_buffers(self) -> None:
        """Allocate or refresh device-side boundary metadata and accumulators."""
        if self.xp is np:
            return
        n_slots = len(self.FACE_NAMES) + len(self.collectors)
        if self._d_absorbed_charge is None or self._d_absorbed_charge.size != n_slots:
            self._d_absorbed_charge = self.xp.zeros(n_slots, dtype=np.float64)
            self._d_absorbed_energy = self.xp.zeros(n_slots, dtype=np.float64)
            self._d_absorbed_count = self.xp.zeros(n_slots, dtype=np.uint64)

        if self.solid_mask is None:
            self._solid_mask_src = None
            self._solid_mask_f_u8 = None
        elif self._solid_mask_src is not self.solid_mask:
            self._solid_mask_src = self.solid_mask
            self._solid_mask_f_u8 = self.xp.asarray(
                self.solid_mask, dtype=np.uint8
            ).reshape(-1, order="F")

        if self.collectors and self._col_types_gpu is None:
            types_h = np.zeros(len(self.collectors), dtype=np.int32)
            params_h = np.zeros(6 * len(self.collectors), dtype=np.float64)
            for idx, col in enumerate(self.collectors):
                if col.plane is not None:
                    types_h[idx] = 0
                    params_h[6 * idx + 0] = float(col.plane[0])
                    params_h[6 * idx + 1] = float(col.plane[1])
                    params_h[6 * idx + 2] = float(col.plane[2])
                elif col.bounds is not None:
                    types_h[idx] = 1
                    for j in range(6):
                        params_h[6 * idx + j] = float(col.bounds[j])
                else:
                    types_h[idx] = -1
            self._col_types_gpu = self.xp.asarray(types_h, dtype=np.int32)
            self._col_params_gpu = self.xp.asarray(params_h, dtype=np.float64)

    def _sync_gpu_stats(self) -> None:
        """Flush device-side absorbed charge/energy/count accumulators to host dicts."""
        if not self._gpu_stats_dirty or self._d_absorbed_charge is None:
            return
        q_h = self._d_absorbed_charge.get()
        e_h = self._d_absorbed_energy.get()
        c_h = self._d_absorbed_count.get()
        self._d_absorbed_charge.fill(0.0)
        self._d_absorbed_energy.fill(0.0)
        self._d_absorbed_count.fill(0)
        self._gpu_stats_dirty = False

        for code_idx, name in enumerate(self.FACE_NAMES):
            self._absorbed_charge[name] += float(q_h[code_idx])
            self._absorbed_energy[name] += float(e_h[code_idx])
            self._absorbed_count[name] += int(c_h[code_idx])

        for c_idx, col in enumerate(self.collectors):
            slot = len(self.FACE_NAMES) + c_idx
            dq = float(q_h[slot])
            de = float(e_h[slot])
            cnt = int(c_h[slot])
            col._absorbed_charge += dq
            col._absorbed_energy += de
            col._absorbed_count += cnt
            self._absorbed_charge[col.name] += dq
            self._absorbed_energy[col.name] += de
            self._absorbed_count[col.name] += cnt

    @property
    def absorbed_charge(self) -> dict[str, float]:
        if self._gpu_stats_dirty:
            self._sync_gpu_stats()
        return self._absorbed_charge

    @property
    def absorbed_energy(self) -> dict[str, float]:
        if self._gpu_stats_dirty:
            self._sync_gpu_stats()
        return self._absorbed_energy

    @property
    def absorbed_count(self) -> dict[str, int]:
        if self._gpu_stats_dirty:
            self._sync_gpu_stats()
        return self._absorbed_count

    @classmethod
    def from_solver(cls, solver, collectors=None, solid_mask=None):
        """Construct :class:`ParticleBoundaries` directly from a ``SolverFIT3D``."""
        xp = getattr(solver.E, "xp", np)
        gmap = solver.grid if hasattr(solver, "to_logical") else None
        if gmap is None:
            from .mapping import LogicalGridMap

            gmap = LogicalGridMap.from_grid(solver, xp=xp)

        bc_low = getattr(solver.grid, "bc_low", ("periodic", "periodic", "periodic"))
        bc_high = getattr(solver.grid, "bc_high", ("periodic", "periodic", "periodic"))
        if hasattr(solver, "bc_low") and solver.bc_low is not None:
            bc_low = solver.bc_low
        if hasattr(solver, "bc_high") and solver.bc_high is not None:
            bc_high = solver.bc_high

        if solid_mask is None and (
            getattr(solver, "use_stl", False)
            or getattr(solver, "use_conductors", False)
        ):
            # A cell is solid PEC if all 3 of its primal edges have ieps == 0
            ieps_x = solver.ieps.to_matrix("x")
            ieps_y = solver.ieps.to_matrix("y")
            ieps_z = solver.ieps.to_matrix("z")
            solid_mask = (ieps_x == 0.0) & (ieps_y == 0.0) & (ieps_z == 0.0)

        return cls(
            grid_map=gmap,
            bc_low=bc_low,
            bc_high=bc_high,
            n_pml=getattr(solver, "n_pml", 10),
            solid_mask=solid_mask,
            collectors=collectors,
        )

    def prepare_step_paths(
        self,
        species,
        x0,
        y0,
        z0,
        xi0,
        eta0,
        zeta0,
        xi1,
        eta1,
        zeta1,
        t: float = 0.0,
    ):
        """Truncate logical trajectories at absorbing boundaries and return removal mask.

        For particles crossing a domain wall, CPML inner face, solid PEC cell,
        or user :class:`Collector`, computes the first intersection parameter
        ``alpha in [0, 1]`` along the straight segment ``xi0 -> xi1`` so that
        Esirkepov current deposition deposits the exact partial path up to the
        boundary crossing point. Records absorbed charge and kinetic energy
        per boundary/collector.

        Returns
        -------
        xi1_dep, eta1_dep, zeta1_dep : tuple of 3 ndarray
            Logical endpoints for current deposition (truncated at boundaries
            for absorbed particles).
        keep_mask : ndarray of bool
            Boolean mask of particles remaining in the active simulation domain.
        """
        xp = self.xp
        n_p = species.n_particles
        if n_p == 0:
            self._last_hit_code = xp.zeros(0, dtype=np.int64)
            empty_b = xp.zeros(0, dtype=bool)
            return xi1, eta1, zeta1, empty_b

        alpha_min = xp.ones(n_p, dtype=np.float64)
        hit_code = xp.full(n_p, -1, dtype=np.int64)
        # Codes: 0..5 for x_low, x_high, y_low, y_high, z_low, z_high;
        # 6 for pec; 7 + c_idx for collector c_idx.

        coords0 = (xi0, eta0, zeta0)
        coords1 = (xi1, eta1, zeta1)

        # 1. Domain / CPML inner-face crossings on non-periodic axes
        for a in range(3):
            if self.periodic[a]:
                continue
            c0 = coords0[a]
            c1 = coords1[a]
            c_min = self.xi_min[a]
            c_max = self.xi_max[a]

            cross_low = c1 < c_min
            if bool(xp.any(cross_low)):
                dc = xp.minimum(c1 - c0, -1e-30)
                a_low = xp.where(
                    c0 > c_min, xp.clip((c_min - c0) / dc, 0.0, 1.0), 0.0
                )
                better = cross_low & (a_low <= alpha_min)
                alpha_min = xp.where(better, a_low, alpha_min)
                hit_code = xp.where(better, 2 * a, hit_code)

            cross_high = c1 > c_max
            if bool(xp.any(cross_high)):
                dc = xp.maximum(c1 - c0, 1e-30)
                a_high = xp.where(
                    c0 < c_max, xp.clip((c_max - c0) / dc, 0.0, 1.0), 0.0
                )
                better = cross_high & (a_high <= alpha_min)
                alpha_min = xp.where(better, a_high, alpha_min)
                hit_code = xp.where(better, 2 * a + 1, hit_code)

        # 2. Solid / PEC cell mask crossings
        if self.solid_mask is not None:
            Nx, Ny, Nz = self.grid_map.Nx, self.grid_map.Ny, self.grid_map.Nz
            ix1 = xp.clip(xp.floor(xi1).astype(np.int64), 0, Nx - 1)
            iy1 = xp.clip(xp.floor(eta1).astype(np.int64), 0, Ny - 1)
            iz1 = xp.clip(xp.floor(zeta1).astype(np.int64), 0, Nz - 1)
            in_solid = self.solid_mask[ix1, iy1, iz1] & (hit_code < 0)
            if bool(xp.any(in_solid)):
                ix0 = xp.clip(xp.floor(xi0).astype(np.int64), 0, Nx - 1)
                iy0 = xp.clip(xp.floor(eta0).astype(np.int64), 0, Ny - 1)
                iz0 = xp.clip(xp.floor(zeta0).astype(np.int64), 0, Nz - 1)

                a_pec = xp.ones(n_p, dtype=np.float64)
                for c0, c1, i0, i1 in (
                    (xi0, xi1, ix0, ix1),
                    (eta0, eta1, iy0, iy1),
                    (zeta0, zeta1, iz0, iz1),
                ):
                    di = i1 - i0
                    face = xp.where(di > 0, i1.astype(np.float64), (i0).astype(np.float64))
                    dc = c1 - c0
                    crossed_axis = in_solid & (di != 0) & (xp.abs(dc) > 1e-30)
                    a_ax = xp.where(
                        crossed_axis,
                        xp.clip((face - c0) / xp.where(crossed_axis, dc, 1.0), 0.0, 1.0),
                        1.0,
                    )
                    a_pec = xp.minimum(a_pec, a_ax)
                a_pec = xp.where(
                    (ix0 == ix1) & (iy0 == iy1) & (iz0 == iz1), 0.0, a_pec
                )
                better = in_solid & (a_pec <= alpha_min)
                alpha_min = xp.where(better, a_pec, alpha_min)
                hit_code = xp.where(better, 6, hit_code)

        # 3. User-defined collectors
        x1, y1, z1 = species.x, species.y, species.z
        for c_idx, col in enumerate(self.collectors):
            c_hit, c_alpha = col.check_crossing(x0, y0, z0, x1, y1, z1, xp=xp)
            better = c_hit & (c_alpha <= alpha_min)
            alpha_min = xp.where(better, c_alpha, alpha_min)
            hit_code = xp.where(better, 7 + c_idx, hit_code)

        self._last_hit_code = hit_code
        absorbed = hit_code >= 0
        if not bool(xp.any(absorbed)):
            return xi1, eta1, zeta1, ~absorbed

        # Truncate logical endpoints at the boundary intersection
        xi1_dep = xp.where(absorbed, xi0 + alpha_min * (xi1 - xi0), xi1)
        eta1_dep = xp.where(absorbed, eta0 + alpha_min * (eta1 - eta0), eta1)
        zeta1_dep = xp.where(absorbed, zeta0 + alpha_min * (zeta1 - zeta0), zeta1)

        for a, (c_dep, c_min, c_max) in enumerate(
            (
                (xi1_dep, self.xi_min[0], self.xi_max[0]),
                (eta1_dep, self.xi_min[1], self.xi_max[1]),
                (zeta1_dep, self.xi_min[2], self.xi_max[2]),
            )
        ):
            if not self.periodic[a]:
                c_clipped = xp.clip(c_dep, c_min, c_max)
                if a == 0:
                    xi1_dep = xp.where(absorbed, c_clipped, xi1_dep)
                elif a == 1:
                    eta1_dep = xp.where(absorbed, c_clipped, eta1_dep)
                else:
                    zeta1_dep = xp.where(absorbed, c_clipped, zeta1_dep)

        # Bookkeep absorbed charge and kinetic energy per boundary/collector
        qw_part = species.charge * species.w
        ke_part = species.kinetic_energy_per_particle()

        for code_idx, name in enumerate(self.FACE_NAMES):
            m_face = hit_code == code_idx
            if bool(xp.any(m_face)):
                self.absorbed_charge[name] += float(xp.sum(qw_part[m_face]))
                self.absorbed_energy[name] += float(xp.sum(ke_part[m_face]))
                self.absorbed_count[name] += int(xp.sum(m_face))

        for c_idx, col in enumerate(self.collectors):
            m_col = hit_code == (7 + c_idx)
            dq = float(xp.sum(qw_part[m_col])) if bool(xp.any(m_col)) else 0.0
            de = float(xp.sum(ke_part[m_col])) if bool(xp.any(m_col)) else 0.0
            cnt = int(xp.sum(m_col)) if bool(xp.any(m_col)) else 0
            col.absorbed_charge += dq
            col.absorbed_energy += de
            col.absorbed_count += cnt
            col.current_history.append((float(t), dq))
            self.absorbed_charge[col.name] += dq
            self.absorbed_energy[col.name] += de
            self.absorbed_count[col.name] += cnt

        return xi1_dep, eta1_dep, zeta1_dep, ~absorbed

    def apply_periodic_wrap(self, species) -> None:
        """Wrap physical particle coordinates into ``[min, max)`` on periodic axes."""
        if getattr(species, "on_gpu", False):
            from .gpu_kernels import launch_periodic_wrap_gpu

            launch_periodic_wrap_gpu(self, species)
            return
        if species.n_particles == 0:
            return
        gmap = self.grid_map
        if self.xp is np:
            n = species.n_particles
            if self.periodic[0]:
                Lx = gmap.xmax - gmap.xmin
                if gmap.xmin != 0.0:
                    species._x[:n] -= gmap.xmin
                species._x[:n] %= Lx
                if gmap.xmin != 0.0:
                    species._x[:n] += gmap.xmin
            if self.periodic[1]:
                Ly = gmap.ymax - gmap.ymin
                if gmap.ymin != 0.0:
                    species._y[:n] -= gmap.ymin
                species._y[:n] %= Ly
                if gmap.ymin != 0.0:
                    species._y[:n] += gmap.ymin
            if self.periodic[2]:
                Lz = gmap.zmax - gmap.zmin
                if gmap.zmin != 0.0:
                    species._z[:n] -= gmap.zmin
                species._z[:n] %= Lz
                if gmap.zmin != 0.0:
                    species._z[:n] += gmap.zmin
            return
        if self.periodic[0]:
            Lx = gmap.xmax - gmap.xmin
            species.x = gmap.xmin + (species.x - gmap.xmin) % Lx
        if self.periodic[1]:
            Ly = gmap.ymax - gmap.ymin
            species.y = gmap.ymin + (species.y - gmap.ymin) % Ly
        if self.periodic[2]:
            Lz = gmap.zmax - gmap.zmin
            species.z = gmap.zmin + (species.z - gmap.zmin) % Lz

    def total_absorbed_charge(self) -> float:
        """Sum of absorbed charge [C] across all boundaries and collectors."""
        return float(sum(self.absorbed_charge.values()))

    def total_absorbed_energy(self) -> float:
        """Sum of absorbed kinetic energy [J] across all boundaries and collectors."""
        return float(sum(self.absorbed_energy.values()))


@runtime_checkable
class InjectorProtocol(Protocol):
    """Protocol for particle injectors and surface emitters attached to :class:`~wakis.particles.ParticleSolver`.

    Every object passed in ``ParticleSolver(..., injectors=[...])`` must provide
    this interface so the time-stepping hooks and checkpoint subsystem can bind,
    advance, back-push by :math:`-\\Delta t / 2`, and serialise all injectors
    uniformly without per-class branching.

    Attributes
    ----------
    species : Species
        Target macroparticle species receiving emitted particles.
    z0 : float
        Physical axial coordinate ``z`` [m] of the emitter plane.
    emitter_kind : {"pec_wall", "interior_sheet"} or None
        Boundary classification of the emitter plane (``None`` before
        :meth:`_bind` has been called).
    injected_charge : float
        Cumulative physical charge [C] injected on the owning rank.
    injected_energy : float
        Cumulative relativistic kinetic energy [J] injected on the owning rank.
    """

    species: Any
    z0: float
    emitter_kind: str | None
    injected_charge: float
    injected_energy: float

    def _bind(self, solver) -> Any:
        """Validate and bind the injector to ``solver`` and record ``emitter_kind``."""
        ...

    def inject(
        self,
        t: float,
        dt: float,
        step_index: int | None = None,
        *,
        owning: bool = True,
    ) -> int:
        """Advance step ``[t, t + dt]`` and return the number of appended macroparticles."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable dictionary of cumulative counters and internal state."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> Any:
        """Restore cumulative counters and internal state from ``state`` in-place."""
        ...


class BeamInjector:
    """Macroparticle emitter on a transverse plane ``z = z0``.

    Injects particles from an emitter plane ``z = z0`` and subtracts the
    deposited charge ``q_inj`` from ``solver._particles_q_bg`` at ``z0``. When
    ``z0`` lies on a planar domain PEC boundary face (``z_min`` or ``z_max``
    where tangential ``E = 0``), ``emitter_kind == "pec_wall"`` and this
    represents the wall image charge; if ``z0`` is placed inside the vacuum
    interior (or on a non-PEC boundary face), ``emitter_kind == "interior_sheet"``
    and ``-q_inj`` remains at ``z = z0`` as a fixed immobile opposite
    space-charge sheet balancing the emitted beam charge so Gauss's law holds.
    Embedded PEC solid cathodes (internal STL or conductor staircase solids at
    the emitter position) are not supported and raise :class:`ValueError` in
    :meth:`_bind`. Uses a fractional charge accumulator (``_charge_accumulator``)
    so the time-averaged injected current matches ``I(t)`` exactly.

    Parameters
    ----------
    species : Species
        Target particle species.
    z0 : float
        Axial coordinate of the emitter plane [m].
    current : float or callable
        Total beam current magnitude ``I`` [A] or callable ``current(t) -> float``
        [A] (total current in amperes [A], not linear current density in [A/m]).
        For the planar strip profiles ``"sheet"`` and ``"sheet_lattice"``, ``I``
        is the total current carried by the strip of full width ``2 * radius``,
        so the linear (surface) current density is ``K = I / (2 * radius)``
        [A/m], and each macroparticle has weight ``w = n_real / n``
        (``n = n_macro_per_step``, ``n_real = |I| * dt / |q| + accumulator``)
        just as for the other profiles. In a 2D slab setup (``Ny = 1`` with
        periodic ``y`` of period ``L_y``), the ``|I| * dt / |q|`` real
        particles injected over step ``dt`` occupy the slab volume
        ``2 * radius * L_y * v_z0 * dt``, where ``v_z0 = beta_0 * c`` is the
        axial drift *speed* (not the momentum per mass ``uz0 = gamma_0 *
        v_z0``), so the number density is ``n = |I| / (|q| * 2 * radius * L_y
        * v_z0)`` [m^-3] and the charge density is ``rho = I / (2 * radius *
        L_y * v_z0)`` [C/m^3].
    gamma : float, optional
        Initial Lorentz factor ``gamma_0 > 1``. Mutually exclusive with
        ``energy_eV``.
    energy_eV : float, optional
        Initial kinetic energy [eV].
    profile : {"uniform", "gaussian", "lattice", "sheet", "sheet_lattice"}, optional
        Transverse profile. ``"uniform"`` samples a uniform disk of radius
        ``radius``; ``"gaussian"`` samples a 2D Gaussian with RMS radius
        ``sigma_r = radius`` (unbounded support; Gaussian tails can leave the
        transverse domain); ``"lattice"`` places ``n_macro_per_step``
        macroparticles deterministically on a uniform-area Fibonacci disk
        lattice of radius ``radius`` (independent of ``dt`` and step index);
        ``"sheet"`` samples a planar thin strip at ``y = y0`` exactly with
        ``x`` uniform on ``[x0 - radius, x0 + radius]`` (where ``radius`` is
        the sheet half-width) using stratified sampling of :meth:`_step_rng`
        with one uniform draw per equal sub-interval of width
        ``2 * radius / n`` (``n = n_macro_per_step``), keeping the empirical
        transverse charge uniform to ``O(1 / n)`` per step and reducing the
        variance of the per-step sample mean ``x_bar = (1 / n) * sum_k x_k``
        from ``Var_iid(x_bar) = radius**2 / (3 * n)`` (unstratified i.i.d.
        draws, each of variance ``(2 * radius)**2 / 12 = radius**2 / 3``) to
        ``Var_strat(x_bar) = (1 / n**2) * sum_k (2 * radius / n)**2 / 12 = radius**2 / (3 * n**3)``,
        a variance reduction by a factor of ``n**2``; ``"sheet_lattice"``
        places ``n_macro_per_step`` macroparticles deterministically at the
        stratum midpoints ``x_k = x0 + radius * (2 * (k + 0.5) / n - 1)``,
        ``y_k = y0`` (zero transverse shot noise). On periodic ``x`` or ``y``
        axes, coordinates outside ``[min, max)`` wrap periodically across cell
        indices during field gather and charge/current deposition and are
        wrapped into ``[min, max)`` at the end of each coupled step by
        :meth:`ParticleBoundaries.apply_periodic_wrap`. Default is ``"uniform"``.
    radius : float, optional
        Disk radius (for ``"uniform"`` and ``"lattice"``), RMS radius
        ``sigma_r`` (for ``"gaussian"``), or planar sheet half-width along ``x``
        (for ``"sheet"`` and ``"sheet_lattice"``) [m]. Must satisfy
        ``radius >= 0`` (raises :class:`ValueError` in ``__init__`` if negative).
        Default is ``1e-3``.
    x0, y0 : float, optional
        Transverse beam centre [m]. Default is ``0.0``.
    n_macro_per_step : int, optional
        Number of macroparticles injected per step. Default is 16.
    rise_time : float, optional
        Current ramp duration ``tau_r`` [s] applied when ``current``
        is a constant float. Default is ``0.0``.
    direction : {+1, -1}, optional
        Propagation direction along ``z``. Default is ``+1``.
    thermal_u : float, optional
        RMS transverse proper velocity ``sigma_{u_x, u_y}`` [m/s]. Default ``0.0``.
    seed : int, optional
        Deterministic RNG seed combined with the solver step index
        ``n_particle_steps`` via :class:`numpy.random.SeedSequence`.
        Default is 42.
    rise_shape : {"linear", "cos2"}, optional
        Current ramp profile for ``0 <= t < rise_time`` when ``current`` is a
        constant float: ``"linear"`` gives ``I * (t / tau_r)`` and ``"cos2"``
        gives ``I * sin^2(pi * t / (2 * tau_r))``. Default is ``"linear"``.
    velocity_modulation : callable, optional
        Optional callable ``velocity_modulation(t) -> float`` returning the
        dimensionless relative axial proper-velocity modulation
        ``delta = Delta u_z / u_z0`` applied to particles injected at midpoint
        time ``t + dt / 2`` as ``u_z = u_z0 * (1 + delta)``. Default is ``None``.
    """

    _VALID_Z_BCS = ("pec", "electric", "pmc", "magnetic", "periodic", "cpml")

    def __init__(
        self,
        species,
        z0: float,
        current,
        gamma: float | None = None,
        energy_eV: float | None = None,
        profile: str = "uniform",
        radius: float = 1.0e-3,
        x0: float = 0.0,
        y0: float = 0.0,
        n_macro_per_step: int = 16,
        rise_time: float = 0.0,
        direction: int = 1,
        thermal_u: float = 0.0,
        seed: int = 42,
        rise_shape: str = "linear",
        velocity_modulation=None,
    ):
        self.species = species
        self.z0 = float(z0)
        self.current = current
        self.profile = str(profile).strip().lower()
        if self.profile not in (
            "uniform",
            "gaussian",
            "lattice",
            "sheet",
            "sheet_lattice",
        ):
            raise ValueError(
                f"Unknown profile={profile!r}; expected 'uniform', 'gaussian', "
                "'lattice', 'sheet', or 'sheet_lattice'."
            )
        self.radius = float(radius)
        if self.radius < 0.0:
            raise ValueError(f"radius must be >= 0, got {radius}.")
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.n_macro_per_step = int(n_macro_per_step)
        self.rise_time = float(rise_time)
        if self.rise_time < 0.0:
            raise ValueError(f"rise_time must be >= 0, got {rise_time}.")
        if callable(self.current) and self.rise_time > 0.0:
            raise ValueError(
                "Cannot specify rise_time > 0 when current is callable; "
                "include the rise profile in the callable instead."
            )
        self.rise_shape = str(rise_shape).strip().lower()
        if self.rise_shape not in ("linear", "cos2"):
            raise ValueError(
                f"Unknown rise_shape={rise_shape!r}; expected 'linear' or 'cos2'."
            )
        if velocity_modulation is not None and not callable(velocity_modulation):
            raise ValueError(
                "velocity_modulation must be a callable f(t) -> float or None."
            )
        self.velocity_modulation = velocity_modulation
        self.direction = 1 if int(direction) >= 0 else -1
        self.thermal_u = float(thermal_u)
        self.seed = int(seed)
        self.step_index = 0
        self.emitter_kind: str | None = None

        if gamma is None and energy_eV is None:
            raise ValueError("Provide either gamma or energy_eV for BeamInjector.")
        if gamma is not None:
            self.gamma0 = float(gamma)
        else:
            rest_J = species.mass * c_light * c_light
            self.gamma0 = 1.0 + (float(energy_eV) * e_charge) / rest_J
        if self.gamma0 <= 1.0:
            raise ValueError(f"Injector gamma must be > 1, got {self.gamma0}.")

        self.beta0 = float(np.sqrt(1.0 - 1.0 / (self.gamma0 * self.gamma0)))
        self.uz0 = self.direction * self.gamma0 * self.beta0 * c_light

        self._charge_accumulator = 0.0  # [real particles]
        self.injected_charge = 0.0  # [C]
        self.injected_energy = 0.0  # [J]
        self.injected_count = 0

    def _bind(self, solver) -> BeamInjector:
        """Validate ``z0`` against ``solver``'s grid and BCs and record ``emitter_kind``.

        Notes
        -----
        Under MPI (``solver.use_mpi=True`` and ``solver.size > 1``), :meth:`_bind`
        is a collective operation across ``solver.comm`` (broadcasting the
        global low-``z`` and high-``z`` boundary conditions from ranks ``0`` and
        ``size - 1`` when not yet cached on ``solver``) and must be called by
        all ranks simultaneously.

        Parameters
        ----------
        solver : SolverFIT3D
            Electromagnetic FIT solver instance.

        Returns
        -------
        BeamInjector
            ``self`` after recording ``emitter_kind`` as ``"pec_wall"`` or
            ``"interior_sheet"``.

        Raises
        ------
        ValueError
            If ``z0`` lies outside the physical z-domain, inside a PML/CPML
            absorbing layer, on a boundary face with outward ``direction``,
            if the transverse emitter footprint leaves a non-periodic physical
            transverse domain ``[xmin, xmax]`` / ``[ymin, ymax]`` or reaches
            into a transverse ``"cpml"`` absorbing layer (``x0 +- radius`` and
            ``y0 +- radius`` for ``"uniform"`` and ``"lattice"``; ``x0 +- radius``
            and strictly interior ``ymin < y0 < ymax`` for ``"sheet"`` and
            ``"sheet_lattice"``; on periodic ``x`` or ``y`` axes, ``2 * radius <= L_axis``
            for disk/lattice and ``2 * radius <= L_x`` with unconstrained ``y0``
            for sheet profiles; no bound for ``"gaussian"`` because its support
            is unbounded), if any cell under the emitter footprint (disk for
            ``"uniform"`` and ``"lattice"``, :math:`3\\sigma` disk for
            ``"gaussian"``, or strip ``[x0 - radius, x0 + radius] x {y0}`` for
            ``"sheet"`` and ``"sheet_lattice"``) overlaps an embedded PEC solid
            cell, or if the solver uses ``"abc"`` or an unknown/unresolved
            global ``z`` boundary condition.
        """
        global_bc_z = getattr(solver, "_global_bc_z", None)
        if (
            global_bc_z is None
            and bool(getattr(solver, "use_mpi", False))
            and int(getattr(solver, "size", 1)) > 1
            and getattr(solver, "comm", None) is not None
        ):
            comm = solver.comm
            size = int(solver.size)
            raw_z_low = comm.bcast(solver.bc_low[2], root=0)
            raw_z_high = comm.bcast(solver.bc_high[2], root=size - 1)
            bc_z_low = _parse_bc_face(raw_z_low)
            bc_z_high = _parse_bc_face(raw_z_high)
            solver._global_bc_z = (bc_z_low, bc_z_high)
            global_bc_z = solver._global_bc_z

        for bc_face in (
            *getattr(solver, "bc_low", ()),
            *getattr(solver, "bc_high", ()),
        ):
            if _parse_bc_face(bc_face) == "abc":
                raise ValueError(
                    "BeamInjector does not support 'abc' boundary conditions; "
                    "use 'pec', 'pmc', 'periodic', or 'cpml' (with z0 outside the PML)."
                )
        if bool(getattr(solver, "activate_abc", False)):
            raise ValueError(
                "BeamInjector does not support 'abc' boundary conditions; "
                "use 'pec', 'pmc', 'periodic', or 'cpml' (with z0 outside the PML)."
            )

        bc_z_low = _parse_bc_face(solver.bc_low[2])
        bc_z_high = _parse_bc_face(solver.bc_high[2])
        if (bc_z_low == "mpi" or bc_z_high == "mpi") and global_bc_z is not None:
            if bc_z_low == "mpi":
                bc_z_low = _parse_bc_face(global_bc_z[0])
            if bc_z_high == "mpi":
                bc_z_high = _parse_bc_face(global_bc_z[1])

        if bc_z_low == "abc" or bc_z_high == "abc":
            raise ValueError(
                "BeamInjector does not support 'abc' boundary conditions; "
                "use 'pec', 'pmc', 'periodic', or 'cpml' (with z0 outside the PML)."
            )
        if bc_z_low not in self._VALID_Z_BCS or bc_z_high not in self._VALID_Z_BCS:
            raise ValueError(
                "Unknown or unresolved global z boundary condition in BeamInjector._bind: "
                f"bc_low[2]={bc_z_low!r}, bc_high[2]={bc_z_high!r}."
            )

        if (
            bool(getattr(solver, "use_mpi", False))
            and getattr(solver.grid, "Z_nodes", None) is not None
        ):
            z_nodes = np.asarray(solver.grid.Z_nodes, dtype=np.float64)
            n_phys = int(z_nodes.size - 1)
        else:
            periodic_z = bc_z_low == "periodic" and bc_z_high == "periodic"
            edge_ghost_z = (
                0
                if periodic_z
                else int(getattr(solver.grid, "edge_ghosts", (0, 0, 0))[2])
            )
            n_phys = int(solver.Nz) - edge_ghost_z
            z_nodes = np.asarray(solver.grid.z[: n_phys + 1], dtype=np.float64)

        z_min = float(z_nodes[0])
        z_max = float(z_nodes[-1])
        if self.z0 < z_min or self.z0 > z_max:
            raise ValueError(
                f"BeamInjector z0={self.z0} is outside the physical z-domain "
                f"[{z_min}, {z_max}]."
            )

        if self.z0 == z_min:
            zeta0 = 0.0
        elif self.z0 == z_max:
            zeta0 = float(n_phys)
        else:
            k = int(
                np.clip(
                    np.searchsorted(z_nodes, self.z0, side="right") - 1,
                    0,
                    n_phys - 1,
                )
            )
            zeta0 = float(k) + (self.z0 - float(z_nodes[k])) / float(
                z_nodes[k + 1] - z_nodes[k]
            )

        n_pml = int(getattr(solver, "n_pml", 10))
        if bc_z_low == "cpml" and zeta0 <= float(n_pml):
            raise ValueError(
                f"BeamInjector z0={self.z0} (zeta={zeta0:.3f}) lies inside the "
                f"low-z PML/CPML layer [0, {n_pml}]."
            )
        if bc_z_high == "cpml" and zeta0 >= float(n_phys - n_pml):
            raise ValueError(
                f"BeamInjector z0={self.z0} (zeta={zeta0:.3f}) lies inside the "
                f"high-z PML/CPML layer [{n_phys - n_pml}, {n_phys}]."
            )

        on_low_face = abs(zeta0) <= 1e-12
        on_high_face = abs(zeta0 - float(n_phys)) <= 1e-12
        if on_low_face and bc_z_low != "periodic" and self.direction != 1:
            raise ValueError(
                f"BeamInjector on low-z boundary z0={self.z0} must emit into the "
                f"domain with direction=+1, got direction={self.direction}."
            )
        if on_high_face and bc_z_high != "periodic" and self.direction != -1:
            raise ValueError(
                f"BeamInjector on high-z boundary z0={self.z0} must emit into the "
                f"domain with direction=-1, got direction={self.direction}."
            )

        edge_ghosts = getattr(solver.grid, "edge_ghosts", (0, 0, 0))
        bc_x_low = _parse_bc_face(solver.bc_low[0])
        bc_x_high = _parse_bc_face(solver.bc_high[0])
        periodic_x = bc_x_low == "periodic" and bc_x_high == "periodic"
        nx_phys = int(solver.Nx) - (0 if periodic_x else int(edge_ghosts[0]))
        x_phys_nodes = np.asarray(solver.grid.x[: nx_phys + 1], dtype=np.float64)
        x_min = float(x_phys_nodes[0])
        x_max = float(x_phys_nodes[-1])

        bc_y_low = _parse_bc_face(solver.bc_low[1])
        bc_y_high = _parse_bc_face(solver.bc_high[1])
        periodic_y = bc_y_low == "periodic" and bc_y_high == "periodic"
        ny_phys = int(solver.Ny) - (0 if periodic_y else int(edge_ghosts[1]))
        y_phys_nodes = np.asarray(solver.grid.y[: ny_phys + 1], dtype=np.float64)
        y_min = float(y_phys_nodes[0])
        y_max = float(y_phys_nodes[-1])

        if self.profile in ("uniform", "lattice", "sheet", "sheet_lattice"):
            # On non-periodic, non-CPML faces, disk/lattice and sheet-x endpoints
            # may touch the domain wall (measure-zero boundary set under continuous
            # uniform/stratified sampling, and strict interior midpoint nodes for
            # 'lattice' and 'sheet_lattice'). On a non-periodic y axis, 'sheet' and
            # 'sheet_lattice' place every particle at y = y0 (measure 1), and
            # ParticleBoundaries.prepare_step_paths uses strict wall crossing
            # (c1 < c_min / c1 > c_max), so y0 must lie strictly inside (y_min, y_max).
            x_lo = self.x0 - self.radius
            x_hi = self.x0 + self.radius
            if periodic_x:
                L_x = x_max - x_min
                if 2.0 * self.radius > L_x:
                    raise ValueError(
                        f"BeamInjector (profile={self.profile!r}) x-extent width "
                        f"2*radius={2.0 * self.radius} (radius={self.radius}) "
                        f"exceeds the periodic x-domain period L_x={L_x} "
                        f"([{x_min}, {x_max}])."
                    )
            else:
                if x_lo < x_min or x_hi > x_max:
                    raise ValueError(
                        f"BeamInjector (profile={self.profile!r}) x-extent "
                        f"[{x_lo}, {x_hi}] (x0={self.x0}, radius={self.radius}) "
                        f"is outside the physical x-domain [{x_min}, {x_max}]."
                    )
                if bc_x_low == "cpml":
                    x_pml_low = float(x_phys_nodes[min(n_pml, nx_phys)])
                    if x_lo <= x_pml_low:
                        raise ValueError(
                            f"BeamInjector (profile={self.profile!r}) x-extent "
                            f"[{x_lo}, {x_hi}] (x0={self.x0}, radius={self.radius}) "
                            f"reaches into the low-x PML/CPML layer "
                            f"[{x_min}, {x_pml_low}] (n_pml={n_pml})."
                        )
                if bc_x_high == "cpml":
                    x_pml_high = float(x_phys_nodes[max(nx_phys - n_pml, 0)])
                    if x_hi >= x_pml_high:
                        raise ValueError(
                            f"BeamInjector (profile={self.profile!r}) x-extent "
                            f"[{x_lo}, {x_hi}] (x0={self.x0}, radius={self.radius}) "
                            f"reaches into the high-x PML/CPML layer "
                            f"[{x_pml_high}, {x_max}] (n_pml={n_pml})."
                        )

            if self.profile in ("uniform", "lattice"):
                y_lo = self.y0 - self.radius
                y_hi = self.y0 + self.radius
                if periodic_y:
                    L_y = y_max - y_min
                    if 2.0 * self.radius > L_y:
                        raise ValueError(
                            f"BeamInjector (profile={self.profile!r}) y-extent width "
                            f"2*radius={2.0 * self.radius} (radius={self.radius}) "
                            f"exceeds the periodic y-domain period L_y={L_y} "
                            f"([{y_min}, {y_max}])."
                        )
                else:
                    if y_lo < y_min or y_hi > y_max:
                        raise ValueError(
                            f"BeamInjector (profile={self.profile!r}) y-extent "
                            f"[{y_lo}, {y_hi}] (y0={self.y0}, radius={self.radius}) "
                            f"is outside the physical y-domain [{y_min}, {y_max}]."
                        )
                    if bc_y_low == "cpml":
                        y_pml_low = float(y_phys_nodes[min(n_pml, ny_phys)])
                        if y_lo <= y_pml_low:
                            raise ValueError(
                                f"BeamInjector (profile={self.profile!r}) y-extent "
                                f"[{y_lo}, {y_hi}] (y0={self.y0}, radius={self.radius}) "
                                f"reaches into the low-y PML/CPML layer "
                                f"[{y_min}, {y_pml_low}] (n_pml={n_pml})."
                            )
                    if bc_y_high == "cpml":
                        y_pml_high = float(y_phys_nodes[max(ny_phys - n_pml, 0)])
                        if y_hi >= y_pml_high:
                            raise ValueError(
                                f"BeamInjector (profile={self.profile!r}) y-extent "
                                f"[{y_lo}, {y_hi}] (y0={self.y0}, radius={self.radius}) "
                                f"reaches into the high-y PML/CPML layer "
                                f"[{y_pml_high}, {y_max}] (n_pml={n_pml})."
                            )
            elif not periodic_y:
                if self.y0 <= y_min or self.y0 >= y_max:
                    raise ValueError(
                        f"BeamInjector (profile={self.profile!r}) y-extent "
                        f"y0={self.y0} must lie strictly inside the non-periodic "
                        f"physical y-domain ({y_min}, {y_max})."
                    )
                if bc_y_low == "cpml":
                    y_pml_low = float(y_phys_nodes[min(n_pml, ny_phys)])
                    if self.y0 <= y_pml_low:
                        raise ValueError(
                            f"BeamInjector (profile={self.profile!r}) y-extent "
                            f"y0={self.y0} lies inside the low-y PML/CPML layer "
                            f"[{y_min}, {y_pml_low}] (n_pml={n_pml})."
                        )
                if bc_y_high == "cpml":
                    y_pml_high = float(y_phys_nodes[max(ny_phys - n_pml, 0)])
                    if self.y0 >= y_pml_high:
                        raise ValueError(
                            f"BeamInjector (profile={self.profile!r}) y-extent "
                            f"y0={self.y0} lies inside the high-y PML/CPML layer "
                            f"[{y_pml_high}, {y_max}] (n_pml={n_pml})."
                        )

        # Reject embedded PEC solid cathodes (internal STL or conductor solids at emitter)
        ps_att = getattr(solver, "_particle_solver", None)
        solid_mask = getattr(getattr(ps_att, "boundaries", None), "solid_mask", None)
        if solid_mask is None and (
            bool(getattr(solver, "use_stl", False))
            or bool(getattr(solver, "use_conductors", False))
        ):
            ieps_x = solver.ieps.to_matrix("x")
            ieps_y = solver.ieps.to_matrix("y")
            ieps_z = solver.ieps.to_matrix("z")
            solid_mask = (ieps_x == 0.0) & (ieps_y == 0.0) & (ieps_z == 0.0)
        if solid_mask is not None:
            sm_h = (
                solid_mask.get()
                if hasattr(solid_mask, "get")
                else np.asarray(solid_mask, dtype=bool)
            )
            x_nodes = np.asarray(solver.grid.x, dtype=np.float64)
            y_nodes = np.asarray(solver.grid.y, dtype=np.float64)
            z_loc_nodes = np.asarray(solver.grid.z, dtype=np.float64)
            if z_loc_nodes[0] <= self.z0 <= z_loc_nodes[-1]:
                nx_c = int(solver.Nx)
                ny_c = int(solver.Ny)
                ix0 = int(
                    np.clip(
                        np.searchsorted(x_nodes, self.x0, side="right") - 1,
                        0,
                        nx_c - 1,
                    )
                )
                iy0 = int(
                    np.clip(
                        np.searchsorted(y_nodes, self.y0, side="right") - 1,
                        0,
                        ny_c - 1,
                    )
                )
                iz0 = int(
                    np.clip(
                        np.searchsorted(z_loc_nodes, self.z0, side="right") - 1,
                        0,
                        int(solver.Nz) - 1,
                    )
                )
                dx_close = (
                    np.clip(self.x0, x_nodes[:nx_c], x_nodes[1 : nx_c + 1]) - self.x0
                )
                dy_close = (
                    np.clip(self.y0, y_nodes[:ny_c], y_nodes[1 : ny_c + 1]) - self.y0
                )
                if self.profile in ("sheet", "sheet_lattice"):
                    foot_xy_mask = (np.abs(dx_close)[:, None] <= self.radius) & (
                        np.abs(dy_close)[None, :] <= 0.0
                    )
                else:
                    r_foot = (
                        3.0 * self.radius if self.profile == "gaussian" else self.radius
                    )
                    foot_xy_mask = (
                        dx_close[:, None] * dx_close[:, None]
                        + dy_close[None, :] * dy_close[None, :]
                    ) <= r_foot * r_foot
                foot_xy_mask[ix0, iy0] = True
                if bool(np.any(sm_h[:nx_c, :ny_c, iz0][foot_xy_mask])):
                    raise ValueError(
                        f"BeamInjector emitter at (x0={self.x0}, y0={self.y0}, z0={self.z0}, "
                        f"radius={self.radius}) overlaps an embedded PEC solid cell; "
                        "embedded PEC solid cathodes are not supported (place z0 on a "
                        "planar domain PEC boundary or in vacuum)."
                    )

        on_low_pec = on_low_face and bc_z_low in ("pec", "electric")
        on_high_pec = on_high_face and bc_z_high in ("pec", "electric")
        self.emitter_kind = (
            "pec_wall" if (on_low_pec or on_high_pec) else "interior_sheet"
        )
        return self

    def _sheet_lattice_x(self, n_macro: int) -> np.ndarray:
        """Return deterministic midpoint x-coordinates [m] for ``profile="sheet_lattice"``."""
        k_idx = np.arange(n_macro, dtype=np.float64)
        return self.x0 + self.radius * (2.0 * (k_idx + 0.5) / float(n_macro) - 1.0)

    def state_dict(self) -> dict:
        """Return a serialisable dictionary of mutable injector restart state.

        Returns
        -------
        dict
            Dictionary containing ``injected_charge`` [C], ``injected_energy``
            [J], ``injected_count``, ``_charge_accumulator`` [real particles],
            ``step_index``, ``_next_id`` (from ``self.species._next_id``),
            ``emitter_kind``, and ``seed``.
        """
        return {
            "injected_charge": float(self.injected_charge),
            "injected_energy": float(self.injected_energy),
            "injected_count": int(self.injected_count),
            "_charge_accumulator": float(self._charge_accumulator),
            "step_index": int(self.step_index),
            "_next_id": int(getattr(self.species, "_next_id", 0)),
            "emitter_kind": self.emitter_kind,
            "seed": int(self.seed),
        }

    def load_state_dict(self, state: dict) -> BeamInjector:
        """Restore mutable injector restart state from ``state``.

        Parameters
        ----------
        state : dict
            State dictionary produced by :meth:`state_dict`.

        Returns
        -------
        BeamInjector
            ``self`` after restoring all cumulative counters, accumulator,
            step index, and species ``_next_id``.
        """
        self.injected_charge = float(state["injected_charge"])
        self.injected_energy = float(state["injected_energy"])
        self.injected_count = int(state["injected_count"])
        self._charge_accumulator = float(state["_charge_accumulator"])
        self.step_index = int(state["step_index"])
        if "_next_id" in state and hasattr(self.species, "_next_id"):
            self.species._next_id = int(state["_next_id"])
        if "emitter_kind" in state:
            self.emitter_kind = state["emitter_kind"]
        if "seed" in state:
            self.seed = int(state["seed"])
        return self

    def _step_rng(self, step_index: int) -> np.random.Generator:
        """Return the deterministic per-step RNG for ``(self.seed, step_index)``."""
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, int(step_index)])
        )

    def _plan_step(
        self,
        t: float,
        dt: float,
        step_index: int | None = None,
        *,
        owning: bool = True,
    ) -> tuple[int, float, float, int]:
        """Advance step counter and charge accumulator for time step ``[t, t + dt]``.

        Parameters
        ----------
        t : float
            Step start time [s].
        dt : float
            Time step size [s].
        step_index : int or None, optional
            Solver particle step index ``n_particle_steps`` used to seed the
            per-step RNG so restarts and injectors rebuilt at step ``k``
            produce identical particles. Defaults to ``self.step_index``.
        owning : bool, optional
            If False (non-owning MPI rank), also advances
            ``self.species._next_id`` by ``n_macro`` so global particle IDs
            stay synchronised across ranks.

        Returns
        -------
        n_macro : int
            Number of macroparticles to emit this step (``0`` if no emission).
        w_macro : float
            Macro-weight [real particles / macroparticle] for this step.
        t_mid : float
            Step midpoint time ``t + 0.5 * dt`` [s].
        step_idx : int
            Resolved integer step index used to seed :meth:`_step_rng`.
        """
        step_idx = int(self.step_index if step_index is None else step_index)
        self.step_index = step_idx + 1
        t_mid = t + 0.5 * dt
        I_val = self.current_at(t_mid)
        q_abs = abs(self.species.charge)
        n_real_step = (I_val * dt) / q_abs + self._charge_accumulator

        n_macro = self.n_macro_per_step
        if n_macro <= 0 or n_real_step <= 0.0:
            self._charge_accumulator = n_real_step
            return 0, 0.0, t_mid, step_idx

        w_macro = n_real_step / n_macro
        self._charge_accumulator = 0.0
        if not owning and self.species.track_ids:
            self.species._next_id += n_macro
        return n_macro, w_macro, t_mid, step_idx

    def current_at(self, t: float) -> float:
        """Evaluate beam current magnitude ``|I(t)|`` [A] at time ``t`` [s]."""
        if callable(self.current):
            val = abs(float(self.current(t)))
        else:
            val = abs(float(self.current))
            if self.rise_time > 0.0 and t < self.rise_time:
                if t <= 0.0:
                    return 0.0
                if self.rise_shape == "linear":
                    val *= t / self.rise_time
                else:
                    s = float(np.sin(0.5 * np.pi * t / self.rise_time))
                    val *= s * s
        return val

    def _batch_kinetic_energy(self, ux, uy, uz, w) -> float:
        """Return total relativistic kinetic energy [J] of an injected batch."""
        u2 = ux * ux + uy * uy + uz * uz
        g = np.sqrt(1.0 + u2 / (c_light * c_light))
        return float(np.sum(w * self.species.mass * (u2 / (g + 1.0))))

    def inject(
        self,
        t: float,
        dt: float,
        step_index: int | None = None,
        *,
        owning: bool = True,
    ) -> int:
        """Inject macroparticles at ``z = z0`` for time step ``[t, t + dt]``."""
        n_macro, w_macro, t_mid, step_idx = self._plan_step(
            t, dt, step_index=step_index, owning=owning
        )
        if not owning or n_macro == 0:
            return 0

        rng = self._step_rng(step_idx)

        if self.profile == "uniform":
            r = self.radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_macro))
            phi = rng.uniform(0.0, 2.0 * np.pi, size=n_macro)
            x = self.x0 + r * np.cos(phi)
            y = self.y0 + r * np.sin(phi)
        elif self.profile == "lattice":
            if n_macro == 1:
                x = np.full(1, self.x0, dtype=np.float64)
                y = np.full(1, self.y0, dtype=np.float64)
            else:
                k_idx = np.arange(n_macro, dtype=np.float64)
                r = self.radius * np.sqrt((k_idx + 0.5) / float(n_macro))
                golden_angle = np.pi * (3.0 - np.sqrt(5.0))
                phi = golden_angle * k_idx
                dx = r * np.cos(phi)
                dy = r * np.sin(phi)
                dx -= float(np.mean(dx))
                dy -= float(np.mean(dy))
                x = self.x0 + dx
                y = self.y0 + dy
        elif self.profile == "sheet":
            k_idx = np.arange(n_macro, dtype=np.float64)
            u_strat = rng.uniform(0.0, 1.0, size=n_macro)
            x = self.x0 + self.radius * (2.0 * (k_idx + u_strat) / float(n_macro) - 1.0)
            y = np.full(n_macro, self.y0, dtype=np.float64)
        elif self.profile == "sheet_lattice":
            x = self._sheet_lattice_x(n_macro)
            y = np.full(n_macro, self.y0, dtype=np.float64)
        else:
            x = self.x0 + rng.normal(0.0, self.radius, size=n_macro)
            y = self.y0 + rng.normal(0.0, self.radius, size=n_macro)

        z = np.full(n_macro, self.z0, dtype=np.float64)

        if self.thermal_u > 0.0:
            ux = rng.normal(0.0, self.thermal_u, size=n_macro)
            uy = rng.normal(0.0, self.thermal_u, size=n_macro)
        else:
            ux = np.zeros(n_macro, dtype=np.float64)
            uy = np.zeros(n_macro, dtype=np.float64)

        if self.velocity_modulation is not None:
            delta_uz = float(self.velocity_modulation(t_mid))
            uz = np.full(n_macro, self.uz0 * (1.0 + delta_uz), dtype=np.float64)
        else:
            uz = np.full(n_macro, self.uz0, dtype=np.float64)
        w = np.full(n_macro, w_macro, dtype=np.float64)

        ke_added = self._batch_kinetic_energy(ux, uy, uz, w)
        q_added = float(self.species.charge * np.sum(w))

        self.species.add_particles(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w)
        self.injected_charge += q_added
        self.injected_energy += ke_added
        self.injected_count += n_macro
        return n_macro
