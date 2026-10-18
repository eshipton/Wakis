# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np
from scipy.constants import epsilon_0 as eps_0
from scipy.sparse import coo_matrix, diags

from ..dispersive import _ade_coefficients
from ..field_monitors import _flat_index


def _host(array):
    return np.asarray(array.get() if hasattr(array, "get") else array)


def _linear_weights(x, lo, hi, n):
    """
    Linear interpolation from n nodes spanning [lo, hi] to the points x.

    Returns the left node index and the weight of the right node; points
    outside [lo, hi] take the value of the nearest end node.
    """
    if n == 1:
        return np.zeros(x.size, dtype=int), np.zeros(x.size)
    u = np.clip((x - lo) / (hi - lo) * (n - 1), 0.0, n - 1)
    i0 = np.minimum(np.floor(u).astype(int), n - 2)
    return i0, u - i0


class MaterialGrid:
    def __init__(self, center, size, shape, eps_min=1.0, eps_max=4.0, material=None):
        """
        Design region whose relative permittivity is set by a density rho.

        The density is defined on a regular design grid of `shape` nodes
        spanning the box [center - size/2, center + size/2] (a single node
        sits at the centre along axes with one node). Each E edge inside the
        box receives eps_r = eps_min + (eps_max - eps_min) * (P rho), where P
        is the sparse (tri)linear interpolation matrix from the design nodes
        to the edge midpoints. The map is affine in rho, so its transpose
        P^T, used by the adjoint, is exact.

        Parameters
        ----------
        center, size : tuple of float
            Centre and extent of the design box (x, y, z) [m]. A zero size
            selects the single nearest layer of edges along that axis, as in
            `GridFIT3D.region_indices`.
        shape : tuple of int
            Number of design nodes (nx, ny, nz); use 1 along zero-size axes.
        eps_min, eps_max : float
            Relative permittivity at rho = 0 and rho = 1.
        material : DispersiveMaterial, optional
            If given, rho interpolates between the background eps_min
            (rho = 0) and this dispersive material (rho = 1): each design edge
            gets the material with fill fraction w = P rho, as a partially
            filled edge of `SolverFIT3D.add_dispersive_material`
            (eps_inf, pole and conductivity contributions scaled by w).
            `eps_max` is then ignored.

        Notes
        -----
        Only isotropic dielectrics are designed; permeability and
        conductivity in the region are left unchanged. With `material`, the
        stability of the scheme is checked for the material itself (w = 1);
        intermediate densities mix two stable media.
        """
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("shape must contain three positive integers.")
        if eps_min <= 0 or eps_max <= 0:
            raise ValueError("eps_min and eps_max must be positive.")
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.material = material
        self.P = None

    def _build(self, solver):
        """Select the design edges and assemble the interpolation matrix P."""
        lo = self.center - self.size / 2
        hi = self.center + self.size / 2
        flat, rows, cols, vals = [], [], [], []
        n_rows = 0
        for comp in "xyz":
            indices, coords = solver.grid.region_indices(
                "E", comp, self.center, self.size
            )
            I, J, K = np.meshgrid(*indices, indexing="ij")
            X = np.meshgrid(*coords, indexing="ij")
            flat.append(_flat_index(solver, comp, I, J, K).ravel())
            # Trilinear weights: product of the 1D linear weights along x, y, z
            per_axis = [
                _linear_weights(X[d].ravel(), lo[d], hi[d], self.shape[d])
                for d in range(3)
            ]
            m = I.size
            for corner in np.ndindex(2, 2, 2):
                idx = np.zeros(m, dtype=int)
                w = np.ones(m)
                for d in range(3):
                    i0, t = per_axis[d]
                    if self.shape[d] == 1:
                        w = w * (1.0 - corner[d])
                        continue
                    idx = idx * self.shape[d] + i0 + corner[d]
                    w = w * (t if corner[d] else 1.0 - t)
                keep = w != 0
                rows.append(n_rows + np.flatnonzero(keep))
                cols.append(idx[keep])
                vals.append(w[keep])
            n_rows += m
        flat_idx = np.concatenate(flat)
        P = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_rows, int(np.prod(self.shape))),
        ).tocsr()
        # E is held at zero on PEC edges (ieps = 0, e.g. walls at the domain
        # bounds), so their permittivity has no effect: leave them out
        keep = _host(solver.ieps.array)[flat_idx] != 0
        self.flat_idx = flat_idx[keep]
        self.P = P[keep]

    def _check(self, rho):
        rho = np.asarray(rho, dtype=float)
        if rho.shape != self.shape:
            raise ValueError(f"rho has shape {rho.shape}, expected {self.shape}.")
        return rho

    def eps_r(self, rho):
        """
        Relative permittivity on the design edges for density rho.

        Parameters
        ----------
        rho : ndarray
            Density of shape `shape`, nominally in [0, 1].

        Returns
        -------
        ndarray
            eps_r on the design edges, ordered as `flat_idx`.
        """
        rho = self._check(rho)
        return self.eps_min + (self.eps_max - self.eps_min) * (self.P @ rho.ravel())

    def apply(self, solver, rho):
        """
        Write the permittivity of density rho into the solver's design edges.

        Sets ``solver.ieps`` = 1 / (eps_0 eps_r) [m/F] on the design edges and,
        for the standard (non-CPML) time step, rescales the matching rows of
        the precomputed operator ``itDaiDepsDstC`` = diag(ieps) * K, which
        would otherwise keep the old permittivity. ``update_tensors`` is not
        used because it needs operators deleted after the first step and
        disables the first-step initialization. GPU solvers are updated on
        the device; with MPI, each rank sets the design edges it owns.

        Parameters
        ----------
        solver : SolverFIT3D
            Solver that has not started time-stepping yet.
        rho : ndarray
            Density of shape `shape`.

        Raises
        ------
        RuntimeError
            If the solver has already taken a time step.
        """
        if not solver.step_0:
            raise RuntimeError("Apply the design before the first time step.")
        if self.P is None:
            # Built once: all solvers of one design problem share the mesh
            self._build(solver)

        xp = solver.E.xp
        idx = xp.asarray(self.flat_idx)
        ieps_old = solver.ieps.array[idx]
        eps_r = self.eps_r(rho) if self.material is None else self.eps_min
        ieps_new = xp.asarray(1.0 / (eps_0 * eps_r) + np.zeros(self.flat_idx.size))
        solver.ieps.array[idx] = ieps_new

        if not solver.activate_cpml:
            # B = diag(ieps) K: only the rows of the design edges change.
            # Scaling the stored entries keeps the operator's format and
            # backend (SciPy, MKL or CuPy; CSR or CSC)
            scale = xp.ones(3 * solver.N)
            scale[idx] = ieps_new / ieps_old
            B = solver.itDaiDepsDstC
            if getattr(B, "format", "csr") == "csc":
                B.data *= scale[B.indices]
            elif xp is np:  # CSR on CPU
                B.data *= np.repeat(scale, np.diff(B.indptr))
            else:  # CSR on GPU (cupy.repeat takes no array of counts)
                rows = xp.searchsorted(B.indptr, xp.arange(B.nnz), side="right") - 1
                B.data *= scale[rows]
            if hasattr(solver, "iDeps"):
                solver.iDeps = diags(_host(solver.ieps.toarray()), dtype=solver.dtype)
        if self.material is not None:
            self._apply_dispersive(solver, self.P @ rho.ravel())

    def _apply_dispersive(self, solver, w):
        """
        Register the design edges as a dispersive group with fill fraction w.

        `_add_dispersive_edges` drops edges with zero fill and checks the
        stability of the material, so the group is created with w = 1 and the
        fill-dependent coefficients are then set as it would compute them:
        eps_e = (1 - w) eps_min + w eps_inf, a1 = eps_e - w (s + b),
        den = eps_e + w (s + b), with s = sigma dt / (2 eps_0) and b the sum
        of the Debye beta coefficients.
        """
        solver._add_dispersive_edges(self.material, self.flat_idx, np.ones(w.size))
        group = solver._dispersive_groups[-1]
        if not np.array_equal(group["idx_host"], self.flat_idx):
            raise ValueError("The design region overlaps a PEC region.")
        eps_e = (1 - w) * self.eps_min + w * self.material.eps_inf
        loss = self._ade_loss(solver.dt)
        xp = solver.E.xp
        group["weight"] = xp.asarray(w, dtype=solver.dtype)
        group["a1"] = xp.asarray(eps_e - w * loss, dtype=solver.dtype)
        group["den"] = xp.asarray(eps_e + w * loss, dtype=solver.dtype)
        solver._material_grid_group = group

    def _ade_loss(self, dt):
        """sigma dt / (2 eps_0) + sum of the Debye beta coefficients."""
        _, debye = _ade_coefficients(self.material, dt)
        return self.material.sigma * dt / eps_0 / 2 + sum(b for _, b in debye)

    def dispersive_group(self, solver):
        """The dispersive group of this design in `solver`, or None."""
        if self.material is None:
            return None
        return getattr(solver, "_material_grid_group", None)

    def ade_state(self, group, solver):
        """Debye term sum_D (kappa - 1) p_D on the design edges before a step."""
        terms = solver.E.xp.zeros(self.flat_idx.size, dtype=solver.E.array.dtype)
        for kappa, _, p in group["debye"]:
            terms = terms + (kappa - 1) * p
        return terms

    def ade_sensitivity(self, group, solver, e_old, debye_old):
        """
        Derivative of the ADE accumulator with respect to the fill fraction,
        d(acc)/dw - E^{n+1} d(den)/dw, after a step (see the physics guide).
        """
        d_eps = self.material.eps_inf - self.eps_min
        loss = self._ade_loss(solver.dt)
        e_new = solver.E.array[group["idx"]]
        lorentz = sum((p - p_old) for _, _, _, p, p_old in group["lorentz"])
        return (d_eps - loss) * e_old - lorentz - debye_old - (d_eps + loss) * e_new

    def vjp(self, rho, grad_ieps):
        """
        Pull the gradient with respect to ieps on the design edges back to rho.

        dF/drho = (eps_max - eps_min) P^T [dF/dieps * dieps/deps_r], with
        ieps = 1/(eps_0 eps_r), so dieps/deps_r = -1/(eps_0 eps_r^2).

        Parameters
        ----------
        rho : ndarray
            Density at which the gradient was computed.
        grad_ieps : ndarray
            dF/dieps on the design edges, ordered as `flat_idx` [F/m * units
            of F].

        Returns
        -------
        ndarray
            dF/drho with the shape of rho.
        """
        grad_ieps = _host(grad_ieps)
        if self.material is not None:
            # grad_ieps holds dF/dw for the fill fraction w = P rho
            return (self.P.T @ grad_ieps).reshape(self.shape)
        eps_r = self.eps_r(rho)
        grad_eps = -grad_ieps / (eps_0 * eps_r**2)
        grad = (self.eps_max - self.eps_min) * (self.P.T @ grad_eps)
        return grad.reshape(self.shape)
