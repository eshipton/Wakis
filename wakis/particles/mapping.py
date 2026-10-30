# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Piecewise-linear mapping between physical and logical grid coordinates.

On tensor-product grids (uniform or graded), topological charge conservation
of the Esirkepov deposition holds to machine precision when particle shape
functions and trajectories are evaluated in logical (index) coordinates
``xi = i + (x - x_i) / dx_i``, where ``x_i`` are the primal grid nodes and
``dx_i = x_{i+1} - x_i`` are the cell edge lengths.
"""

from __future__ import annotations

import numpy as np


class LogicalGridMap:
    """Piecewise-linear coordinate map between physical ``(x, y, z)`` and
    logical cell coordinates ``(xi, eta, zeta)``.

    Parameters
    ----------
    x_nodes, y_nodes, z_nodes : array-like of float
        1D monotonically increasing arrays of primal node coordinates [m]
        with lengths ``(Nx + 1, Ny + 1, Nz + 1)``.
    xp : module, optional
        Array module (``numpy`` or ``cupy``). Default is ``numpy``.
    """

    def __init__(
        self,
        x_nodes,
        y_nodes,
        z_nodes,
        xp=np,
        edge_ghosts=(0, 0, 0),
        use_mpi: bool = False,
    ):
        self.xp = xp
        self.edge_ghosts = tuple(int(g) for g in edge_ghosts)
        self.use_mpi = bool(use_mpi)
        self.x_nodes = xp.asarray(x_nodes, dtype=np.float64)
        self.y_nodes = xp.asarray(y_nodes, dtype=np.float64)
        self.z_nodes = xp.asarray(z_nodes, dtype=np.float64)

        if (
            self.x_nodes.ndim != 1
            or self.y_nodes.ndim != 1
            or self.z_nodes.ndim != 1
            or self.x_nodes.size < 2
            or self.y_nodes.size < 2
            or self.z_nodes.size < 2
        ):
            raise ValueError("Node arrays must be 1D with at least 2 nodes per axis.")

        self.dx = self.x_nodes[1:] - self.x_nodes[:-1]
        self.dy = self.y_nodes[1:] - self.y_nodes[:-1]
        self.dz = self.z_nodes[1:] - self.z_nodes[:-1]

        if (
            bool(xp.any(self.dx <= 0.0))
            or bool(xp.any(self.dy <= 0.0))
            or bool(xp.any(self.dz <= 0.0))
        ):
            raise ValueError("Node coordinates must be strictly increasing.")

        self.Nx = int(self.dx.size)
        self.Ny = int(self.dy.size)
        self.Nz = int(self.dz.size)

        self.xmin = float(self.x_nodes[0])
        self.xmax = float(self.x_nodes[-1])
        self.ymin = float(self.y_nodes[0])
        self.ymax = float(self.y_nodes[-1])
        self.zmin = float(self.z_nodes[0])
        self.zmax = float(self.z_nodes[-1])

        self._nodes = (self.x_nodes, self.y_nodes, self.z_nodes)
        self._deltas = (self.dx, self.dy, self.dz)
        self._counts = (self.Nx, self.Ny, self.Nz)
        self._mins = (self.xmin, self.ymin, self.zmin)
        self._maxs = (self.xmax, self.ymax, self.zmax)
        non_periodic = (
            bool(self.edge_ghosts[0]),
            bool(self.edge_ghosts[1]),
            bool(self.edge_ghosts[2]) or self.use_mpi,
        )
        self._spans = tuple(
            -1.0 if np_ax else float(mx - mn)
            for np_ax, mn, mx in zip(non_periodic, self._mins, self._maxs)
        )
        self._dx0 = (float(self.dx[0]), float(self.dy[0]), float(self.dz[0]))

        # Detect uniform spacing per axis for fast O(1) affine mapping while
        # preserving exact piecewise-linear searchsorted mapping on graded axes.
        self._uniform = tuple(
            bool(xp.max(xp.abs(d - d0)) <= 1e-14 * d0)
            for d, d0 in zip(self._deltas, self._dx0)
        )
        self.is_uniform = all(self._uniform)

    @classmethod
    def from_grid(cls, grid_or_solver, xp=None) -> "LogicalGridMap":
        """Construct a :class:`LogicalGridMap` from a ``GridFIT3D`` or ``SolverFIT3D``.

        Parameters
        ----------
        grid_or_solver : GridFIT3D or SolverFIT3D
            Wakis grid or solver instance.
        xp : module, optional
            Array module override. Defaults to ``solver.E.xp`` if available,
            else ``numpy``.

        Returns
        -------
        LogicalGridMap
        """
        if hasattr(grid_or_solver, "E") and hasattr(grid_or_solver, "grid"):
            grid = grid_or_solver.grid
            use_mpi = bool(
                getattr(grid_or_solver, "use_mpi", False)
                or getattr(grid, "use_mpi", False)
            )
        else:
            grid = grid_or_solver
            use_mpi = bool(getattr(grid, "use_mpi", False))
        if xp is None:
            field_E = getattr(grid_or_solver, "E", None)
            xp = getattr(field_E, "xp", np)
        edge_ghosts = getattr(grid, "edge_ghosts", (0, 0, 0))
        return cls(
            grid.x,
            grid.y,
            grid.z,
            xp=xp,
            edge_ghosts=edge_ghosts,
            use_mpi=use_mpi,
        )

    def to_logical_1d(
        self, pos, axis: int, use_searchsorted: bool = False, out=None
    ):
        """Map 1D physical coordinates [m] to logical coordinates along ``axis``.

        For ``pos`` inside cell ``i`` (``nodes[i] <= pos < nodes[i+1]``),
        returns ``xi = i + (pos - nodes[i]) / d[i]``. Positions outside the
        domain are linearly extrapolated from the nearest boundary cell so
        that straight-line trajectories crossing a boundary remain continuous.

        Parameters
        ----------
        pos : ndarray of float
            Physical coordinates [m].
        axis : {0, 1, 2}
            Axis index (0 for x, 1 for y, 2 for z).
        use_searchsorted : bool, optional
            If True, force ``searchsorted`` even on uniform axes.
        out : ndarray of float, optional
            Optional pre-allocated output array of matching shape and
            ``float64`` dtype to avoid temporary buffer allocations.

        Returns
        -------
        xi : ndarray of float
            Logical coordinates in ``(-inf, +inf)``, with ``[0, N_axis]``
            spanning the grid domain.
        """
        xp = self.xp
        pos = xp.asarray(pos, dtype=np.float64)
        nodes = self._nodes[axis]
        deltas = self._deltas[axis]
        n_cells = self._counts[axis]

        if self._uniform[axis] and not use_searchsorted:
            if out is not None and out.shape == pos.shape and out.dtype == np.float64:
                xp.subtract(pos, self._mins[axis], out=out)
                out /= self._dx0[axis]
                return out
            return (pos - self._mins[axis]) / self._dx0[axis]

        p_min = self._mins[axis]
        L_ax = self._spans[axis]
        if L_ax > 0.0:
            n_wrap = xp.floor((pos - p_min) / L_ax)
            pos_mod = p_min + (pos - p_min) - n_wrap * L_ax
            idx = xp.searchsorted(nodes, pos_mod, side="right") - 1
            idx = xp.clip(idx, 0, n_cells - 1)
            xi_mod = idx.astype(np.float64) + (pos_mod - nodes[idx]) / deltas[idx]
            res = n_wrap * float(n_cells) + xi_mod
            if out is not None and out.shape == pos.shape:
                out[...] = res
                return out
            return res

        idx = xp.searchsorted(nodes, pos, side="right") - 1
        idx = xp.clip(idx, 0, n_cells - 1)
        res = idx.astype(np.float64) + (pos - nodes[idx]) / deltas[idx]
        if out is not None and out.shape == pos.shape:
            out[...] = res
            return out
        return res

    def to_logical(
        self, x, y, z, use_searchsorted: bool = False, out=None
    ):
        """Map 3D physical coordinates ``(x, y, z)`` [m] to ``(xi, eta, zeta)``.

        Parameters
        ----------
        x, y, z : ndarray of float
            Physical particle coordinates [m].
        use_searchsorted : bool, optional
            If True, force ``searchsorted`` even on uniform axes.
        out : tuple of 3 ndarray of float, optional
            Optional pre-allocated output arrays ``(xi_out, eta_out, zeta_out)``.

        Returns
        -------
        xi, eta, zeta : tuple of ndarray of float
            Logical coordinates along x, y, and z.
        """
        out_x, out_y, out_z = (None, None, None) if out is None else out
        return (
            self.to_logical_1d(x, 0, use_searchsorted=use_searchsorted, out=out_x),
            self.to_logical_1d(y, 1, use_searchsorted=use_searchsorted, out=out_y),
            self.to_logical_1d(z, 2, use_searchsorted=use_searchsorted, out=out_z),
        )

    def to_physical_1d(self, xi, axis: int):
        """Map 1D logical coordinates ``xi`` back to physical coordinates [m].

        Parameters
        ----------
        xi : ndarray of float
            Logical coordinates.
        axis : {0, 1, 2}
            Axis index (0 for x, 1 for y, 2 for z).

        Returns
        -------
        pos : ndarray of float
            Physical coordinates [m].
        """
        xp = self.xp
        xi = xp.asarray(xi, dtype=np.float64)
        nodes = self._nodes[axis]
        deltas = self._deltas[axis]
        n_cells = self._counts[axis]

        if self._uniform[axis]:
            return self._mins[axis] + xi * self._dx0[axis]

        idx = xp.floor(xi).astype(np.int64)
        idx = xp.clip(idx, 0, n_cells - 1)
        frac = xi - idx.astype(np.float64)
        return nodes[idx] + frac * deltas[idx]

    def to_physical(self, xi, eta, zeta):
        """Map 3D logical coordinates ``(xi, eta, zeta)`` to physical ``(x, y, z)`` [m]."""
        return (
            self.to_physical_1d(xi, 0),
            self.to_physical_1d(eta, 1),
            self.to_physical_1d(zeta, 2),
        )

    def linear_cell_index(self, x, y, z):
        """Compute Fortran-ordered linear 3D cell index ``i + Nx * j + Nx * Ny * k``.

        Parameters
        ----------
        x, y, z : ndarray of float
            Physical coordinates [m].

        Returns
        -------
        cell_idx : ndarray of int64
            Linear cell index in ``[0, Nx * Ny * Nz - 1]``.
        """
        xp = self.xp
        if xp is np and np.size(x) > 65536:
            n = int(np.size(x))
            idx_dt = (
                np.int32 if self.Nx * self.Ny * self.Nz < 2**31 else np.int64
            )
            out = np.empty(n, dtype=idx_dt)
            chunk = 65536
            for s0 in range(0, n, chunk):
                s1 = min(s0 + chunk, n)
                xi = self.to_logical_1d(x[s0:s1], 0)
                eta = self.to_logical_1d(y[s0:s1], 1)
                zeta = self.to_logical_1d(z[s0:s1], 2)
                ix = np.clip(np.floor(xi).astype(idx_dt), 0, self.Nx - 1)
                iy = np.clip(np.floor(eta).astype(idx_dt), 0, self.Ny - 1)
                iz = np.clip(np.floor(zeta).astype(idx_dt), 0, self.Nz - 1)
                out[s0:s1] = ix + idx_dt(self.Nx) * (iy + idx_dt(self.Ny) * iz)
            return out
        xi, eta, zeta = self.to_logical(x, y, z)
        ix = xp.clip(xp.floor(xi).astype(np.int64), 0, self.Nx - 1)
        iy = xp.clip(xp.floor(eta).astype(np.int64), 0, self.Ny - 1)
        iz = xp.clip(xp.floor(zeta).astype(np.int64), 0, self.Nz - 1)
        return ix + self.Nx * (iy + self.Ny * iz)

