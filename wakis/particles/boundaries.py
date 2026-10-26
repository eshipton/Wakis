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
            low = float(self.n_pml) if self.bc_low[a] == "cpml" else 0.0
            high = (
                float(n_phys - self.n_pml)
                if self.bc_high[a] == "cpml"
                else float(n_phys)
            )
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


class BeamInjector:
    """Macroparticle emitter on a transverse plane ``z = z0``.

    Injects particles from an emitter plane ``z = z0`` and subtracts the
    deposited charge ``q_inj`` from ``solver._particles_q_bg`` at ``z0``. When
    ``z0`` lies on a PEC boundary face (cathode wall where tangential ``E = 0``),
    this represents the wall image charge; if ``z0`` is placed inside the vacuum
    interior, ``-q_inj`` remains at ``z = z0`` as a fixed immobile opposite
    space-charge sheet balancing the emitted beam charge so Gauss's law holds.
    Uses a fractional charge accumulator (``_charge_accumulator``) so the
    time-averaged injected current matches ``I(t)`` exactly.

    Parameters
    ----------
    species : Species
        Target particle species.
    z0 : float
        Axial coordinate of the emitter plane [m].
    current : float or callable
        Beam current magnitude [A] or callable ``current(t) -> float`` [A].
    gamma : float, optional
        Initial Lorentz factor ``gamma_0 > 1``. Mutually exclusive with
        ``energy_eV``.
    energy_eV : float, optional
        Initial kinetic energy [eV].
    profile : {"uniform", "gaussian"}, optional
        Transverse radial profile. Default is ``"uniform"``.
    radius : float, optional
        Disk radius (for ``"uniform"``) or RMS radius ``sigma_r`` (for
        ``"gaussian"``) [m]. Default is ``1e-3``.
    x0, y0 : float, optional
        Transverse beam centre [m]. Default is ``0.0``.
    n_macro_per_step : int, optional
        Number of macroparticles injected per step. Default is 16.
    rise_time : float, optional
        Linear current ramp duration ``tau_r`` [s] applied when ``current``
        is a constant float. Default is ``0.0``.
    direction : {+1, -1}, optional
        Propagation direction along ``z``. Default is ``+1``.
    thermal_u : float, optional
        RMS transverse proper velocity ``sigma_{u_x, u_y}`` [m/s]. Default ``0.0``.
    seed : int, optional
        Deterministic RNG seed. Default is 42.
    """

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
    ):
        self.species = species
        self.z0 = float(z0)
        self.current = current
        self.profile = str(profile).strip().lower()
        if self.profile not in ("uniform", "gaussian"):
            raise ValueError(
                f"Unknown profile={profile!r}; expected 'uniform' or 'gaussian'."
            )
        self.radius = float(radius)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.n_macro_per_step = int(n_macro_per_step)
        self.rise_time = float(rise_time)
        self.direction = 1 if int(direction) >= 0 else -1
        self.thermal_u = float(thermal_u)
        self.rng = np.random.default_rng(seed)

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

    def current_at(self, t: float) -> float:
        """Evaluate beam current magnitude ``|I(t)|`` [A] at time ``t`` [s]."""
        if callable(self.current):
            val = abs(float(self.current(t)))
        else:
            val = abs(float(self.current))
            if self.rise_time > 0.0 and t < self.rise_time:
                val *= max(0.0, t / self.rise_time)
        return val

    def inject(self, t: float, dt: float) -> int:
        """Inject macroparticles at ``z = z0`` for time step ``[t, t + dt]``."""
        I_val = self.current_at(t + 0.5 * dt)
        q_abs = abs(self.species.charge)
        n_real_step = (I_val * dt) / q_abs + self._charge_accumulator

        n_macro = self.n_macro_per_step
        if n_macro <= 0 or n_real_step <= 0.0:
            self._charge_accumulator = n_real_step
            return 0

        w_macro = n_real_step / n_macro
        self._charge_accumulator = 0.0

        if self.profile == "uniform":
            r = self.radius * np.sqrt(self.rng.uniform(0.0, 1.0, size=n_macro))
            phi = self.rng.uniform(0.0, 2.0 * np.pi, size=n_macro)
            x = self.x0 + r * np.cos(phi)
            y = self.y0 + r * np.sin(phi)
        else:
            x = self.x0 + self.rng.normal(0.0, self.radius, size=n_macro)
            y = self.y0 + self.rng.normal(0.0, self.radius, size=n_macro)

        z = np.full(n_macro, self.z0, dtype=np.float64)

        if self.thermal_u > 0.0:
            ux = self.rng.normal(0.0, self.thermal_u, size=n_macro)
            uy = self.rng.normal(0.0, self.thermal_u, size=n_macro)
        else:
            ux = np.zeros(n_macro, dtype=np.float64)
            uy = np.zeros(n_macro, dtype=np.float64)
        uz = np.full(n_macro, self.uz0, dtype=np.float64)
        w = np.full(n_macro, w_macro, dtype=np.float64)

        u2 = ux * ux + uy * uy + uz * uz
        g = np.sqrt(1.0 + u2 / (c_light * c_light))
        ke_added = float(np.sum(w * self.species.mass * (u2 / (g + 1.0))))
        q_added = float(self.species.charge * np.sum(w))

        self.species.add_particles(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w)
        self.injected_charge += q_added
        self.injected_energy += ke_added
        self.injected_count += n_macro
        return n_macro
