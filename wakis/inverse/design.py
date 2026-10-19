# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np
from scipy.constants import epsilon_0 as eps_0
from scipy.sparse import coo_matrix, diags, identity, kron

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


def _difference_matrix(n, spacing):
    """Central differences (one-sided at the ends) on n nodes [1/m]."""
    if n == 1:
        return coo_matrix((1, 1))
    rows = np.r_[0, 0, np.arange(1, n - 1).repeat(2), n - 1, n - 1]
    cols = np.r_[0, 1, np.c_[np.arange(n - 2), np.arange(2, n)].ravel(), n - 2, n - 1]
    vals = np.r_[-1, 1, np.tile([-0.5, 0.5], n - 2), -1, 1] / spacing
    return coo_matrix((vals, (rows, cols)), shape=(n, n))


# Regularization of the normal m = g / sqrt(|g|^2 + delta^2) of a density
# gradient g: delta = _NORMAL_REG / (design node spacing). At an interface
# |g| ~ 1 / spacing, so m is the unit normal to within 1e-4; in uniform
# regions m -> 0 and the tensor becomes isotropic (and stays differentiable).
_NORMAL_REG = 1e-2


class MaterialGrid:
    def __init__(
        self,
        center,
        size,
        shape,
        eps_min=1.0,
        eps_max=4.0,
        material=None,
        smoothing=None,
    ):
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
        smoothing : {None, 'tensor'}, optional
            None (default): the isotropic interpolation above. 'tensor':
            subpixel smoothing with the full averaged tensor. The
            interpolated density f = P rho, clipped to [0, 1], is the fill
            fraction of the eps_max material in the averaging pixel of each
            edge and the gradient g of the interpolated nodal central
            differences of rho gives the interface normal
            m = g / sqrt(|g|^2 + delta^2). Each edge then gets the
            inverse-permittivity tensor
            T = I / <eps> + (<1/eps> - 1/<eps>) m m^T, with
            <eps> = eps_min + (eps_max - eps_min) f and
            <1/eps> = (1 - f) / eps_min + f / eps_max: its diagonal entry
            on the edge and its off-diagonal entries through the corner
            couplings of `wakis.tensor_averaging`. rho should be the
            filtered (and projected) density; smooth level-set designs then
            get boundaries that move continuously within the cells. The map
            rho -> T is differentiable and `vjp` is exact (see "Tensor
            subpixel averaging" in the physics guide). Not combined with
            `material`; real fields only.

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
        if smoothing not in (None, "tensor"):
            raise ValueError("smoothing must be None or 'tensor'.")
        if smoothing is not None and material is not None:
            # Dispersive subpixel averaging: the static tensor uses eps_inf
            self.eps_max = float(material.eps_inf)
        self.smoothing = smoothing
        self.P = None
        self.network = None

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
        self.N = solver.N
        if self.smoothing is None:
            return
        # Gradient of the interpolated nodal central differences of rho
        spacing = np.where(
            np.array(self.shape) > 1,
            self.size / np.maximum(np.array(self.shape) - 1, 1),
            np.inf,
        )
        self.G = []
        for d in range(3):
            factors = [identity(n) for n in self.shape]
            factors[d] = _difference_matrix(self.shape[d], spacing[d])
            D = kron(kron(factors[0], factors[1]), factors[2]).tocsr()
            self.G.append((self.P @ D).tocsr())
        # (no gradient at all when every axis has a single node)
        self.delta = max(_NORMAL_REG / np.min(spacing), 1e-30)

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
        if self.smoothing == "tensor":
            terms = self._tensor_terms(self._check(rho))
            self._terms = terms
            ieps_new = xp.asarray(terms["ieps"])
        else:
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
        op = getattr(solver, "tensor_operator", None)
        if self.material is not None and self.smoothing == "tensor":
            self._apply_network(solver)
            if op is not None:  # couplings to the design edges are dropped
                op.install(solver)
            return
        if self.material is not None:
            self._apply_dispersive(solver, self.P @ rho.ravel())
        if self.smoothing == "tensor":
            self._apply_tensor(solver, op)
        elif op is not None:
            # K' = K diag(1/ieps) depends on the new diagonal
            op.install(solver)

    def _tensor_terms(self, rho):
        """
        Fill fraction, normal and tensor entries on the design edges.

        Returns a dict with f (clipped P rho), inside (0 < P rho < 1), g
        (density gradient [1/m]), q = |g|^2 + delta^2, m = g / sqrt(q), the
        averages A = 1/<eps_r>, B = <1/eps_r>, aniso = B - A, comp (edge
        component) and ieps = (A + aniso m_comp^2) / eps_0 [m/F].
        """
        r = rho.ravel()
        raw = self.P @ r
        f = np.clip(raw, 0.0, 1.0)
        g = np.stack([G @ r for G in self.G], axis=1)
        q = np.sum(g**2, axis=1) + self.delta**2
        m = g / np.sqrt(q)[:, None]
        A = 1.0 / (self.eps_min + (self.eps_max - self.eps_min) * f)
        B = (1.0 - f) / self.eps_min + f / self.eps_max
        aniso = np.clip(B - A, 0.0, None)
        comp = self.flat_idx // self.N
        n = np.arange(f.size)
        ieps = (A + aniso * m[n, comp] ** 2) / eps_0
        inside = (raw > 0) & (raw < 1)
        return dict(f=f, inside=inside, g=g, q=q, m=m, A=A, B=B, aniso=aniso,
                    comp=comp, ieps=ieps)  # fmt: skip

    def _apply_tensor(self, solver, op):
        """Write the off-diagonal tensor of the design edges and check dt."""
        from .. import tensor_averaging as ta

        if np.iscomplexobj(solver.E.array):
            raise NotImplementedError("Tensor smoothing needs real fields.")
        if op is None:
            op = ta.TensorOperator(solver)
            solver.tensor_operator = op
        t = self._terms
        op.set_edges(self.flat_idx, t["aniso"] / eps_0, t["m"])
        op.install(solver)
        dt_base = getattr(solver, "conformal_dt_max", None) or ta.yee_dt(solver)
        dt_max = ta.bound_stable_dt(solver, op, dt_base)
        if solver.dt > dt_max:
            raise ValueError(
                f"dt={solver.dt:.3e} s exceeds the stability bound {dt_max:.3e} s "
                "of the tensor-smoothed design; use cfln <= 0.5."
            )

    def _network_params(self, t):
        """
        Sub-capacitor parameters of the dispersive diagonal average of the
        design edges (tangential capacitors first, then the series ones of
        the material), see `wakis.tensor_dispersive`.

        Returns c (weights), eps_inf and mu (pole multipliers) of the 2 n_e
        capacitors and p = m_c^2.
        """
        n = np.arange(t["f"].size)
        p = t["m"][n, t["comp"]] ** 2
        f = t["f"]
        eps_t = (1 - f) * self.eps_min + f * self.eps_max
        c = np.concatenate([1 - p, p * f])
        eps_inf = np.concatenate([eps_t, np.full(f.size, self.eps_max)])
        mu = np.concatenate([f, np.ones(f.size)])
        return c, eps_inf, mu, p

    def _apply_network(self, solver):
        """
        Dispersive subpixel averaging of the design edges: the diagonal of
        the averaged tensor at every frequency, realized by a tangential
        (parallel) and a series sub-capacitor per edge with the normal m of
        the density gradient (no off-diagonal couplings).
        """
        from .. import tensor_averaging as ta
        from .. import tensor_dispersive as td

        if getattr(solver, "dispersive_network", None) is not None:
            raise NotImplementedError(
                "A dispersive tensor design cannot be combined with conformal "
                "dispersive solids."
            )
        if np.iscomplexobj(solver.E.array):
            raise NotImplementedError("Tensor smoothing needs real fields.")
        c, eps_inf, mu, _ = self._network_params(self._terms)
        n_e = self.flat_idx.size
        edge = np.tile(np.arange(n_e), 2)
        members = [(np.arange(2 * n_e), mu)]
        net = td.network_from_edges(
            solver, self.flat_idx, edge, c, eps_inf, [self.material], members
        )
        dt_base = getattr(solver, "conformal_dt_max", None) or ta.yee_dt(solver)
        if net.stability_bound(dt_base) > 1:
            raise ValueError(
                f"dt={solver.dt:.3e} s violates the stability bound of the "
                "dispersive tensor design; use a smaller cfln."
            )
        td.install(solver, net)
        self.network = net

    def network_of(self, solver):
        """The dispersive sub-capacitor network of this design, or None."""
        net = getattr(solver, "dispersive_network", None)
        return net if net is not None and net is self.network else None

    def _network_vjp(self, rho, g_ieps, g_c, g_eps, g_mu):
        """
        dF/drho for a dispersive tensor design from the gradients with
        respect to ieps (per edge) and the capacitor parameters c, eps_inf
        and mu (2 n_e each), through f = clip(P rho) and p = m_c^2.
        """
        t = self._tensor_terms(self._check(rho))
        f, n_e = t["f"], t["f"].size
        n = np.arange(n_e)
        p = t["m"][n, t["comp"]] ** 2
        e0, e1 = self.eps_min, self.eps_max
        eps_t = (1 - f) * e0 + f * e1
        gT, gS = (slice(0, n_e), slice(n_e, 2 * n_e))
        # ieps = (p ((1 - f)/e0 + f/e1) + (1 - p)/eps_t) / eps_0
        d_f = g_ieps * (p * (1 / e1 - 1 / e0) - (1 - p) * (e1 - e0) / eps_t**2) / eps_0
        d_p = g_ieps * (((1 - f) / e0 + f / e1) - 1 / eps_t) / eps_0
        # c_T = 1 - p, c_S = p f, eps_inf_T = eps_t, mu_T = f
        d_p += -g_c[gT] + f * g_c[gS]
        d_f += p * g_c[gS] + (e1 - e0) * g_eps[gT] + g_mu[gT]
        d_f = np.where(t["inside"], d_f, 0.0)
        mc = t["m"][n, t["comp"]]
        d_m = np.zeros((n_e, 3))
        d_m[n, t["comp"]] = 2 * mc * d_p
        sq = np.sqrt(t["q"])
        d_g = (
            d_m / sq[:, None]
            - (np.einsum("ij,ij->i", d_m, t["g"]) / sq**3)[:, None] * t["g"]
        )
        grad = self.P.T @ d_f
        for k in range(3):
            grad = grad + self.G[k].T @ d_g[:, k]
        return np.asarray(grad).reshape(self.shape)

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

    def vjp(self, rho, grad_ieps, pair_grad=None):
        """
        Pull the gradient with respect to ieps on the design edges back to rho.

        dF/drho = (eps_max - eps_min) P^T [dF/dieps * dieps/deps_r], with
        ieps = 1/(eps_0 eps_r), so dieps/deps_r = -1/(eps_0 eps_r^2). With
        ``smoothing='tensor'`` the chain rule runs through the fill fraction
        and the normal of every edge (see `_tensor_vjp`).

        Parameters
        ----------
        rho : ndarray
            Density at which the gradient was computed.
        grad_ieps : ndarray
            dF/dieps on the design edges, ordered as `flat_idx` [F/m * units
            of F].

        pair_grad : tuple (ea, eb, w), optional
            Tensor smoothing: coupled edge pairs (flat indices) and
            dF/dT_ij of each pair, where T_ij(a) and T_ij(b) enter the
            coupling S_ab = c_ab (T_ij(a) + T_ij(b)).

        Returns
        -------
        ndarray
            dF/drho with the shape of rho.
        """
        grad_ieps = _host(grad_ieps)
        if self.smoothing == "tensor":
            return self._tensor_vjp(rho, grad_ieps, pair_grad)
        if self.material is not None:
            # grad_ieps holds dF/dw for the fill fraction w = P rho
            return (self.P.T @ grad_ieps).reshape(self.shape)
        eps_r = self.eps_r(rho)
        grad_eps = -grad_ieps / (eps_0 * eps_r**2)
        grad = (self.eps_max - self.eps_min) * (self.P.T @ grad_eps)
        return grad.reshape(self.shape)

    def _tensor_vjp(self, rho, grad_ieps, pair_grad):
        """
        dF/drho through ieps_a = (A + a m_c^2) / eps_0 and the off-diagonal
        entries T_cj(a) = a m_c m_j / eps_0 of every design edge (component
        c), with A = 1/<eps>, a = <1/eps> - 1/<eps>, f = P rho and
        m = g / sqrt(|g|^2 + delta^2), g = G rho.
        """
        t = self._tensor_terms(self._check(rho))
        n_e = self.flat_idx.size
        n = np.arange(n_e)
        comp, m, aniso = t["comp"], t["m"], t["aniso"]
        # dF/dT_cj(a) for the design edges (column j)
        sT = np.zeros((n_e, 3))
        if pair_grad is not None:
            ea, eb, w = (np.asarray(x) for x in pair_grad)
            order = np.argsort(self.flat_idx)
            sorted_idx = self.flat_idx[order]
            for e_own, e_other in ((ea, eb), (eb, ea)):
                pos = np.minimum(np.searchsorted(sorted_idx, e_own), n_e - 1)
                hit = sorted_idx[pos] == e_own
                np.add.at(sT, (order[pos[hit]], e_other[hit] // self.N), w[hit])
        sT[n, comp] = 0.0
        s_d = grad_ieps / eps_0
        sT = sT / eps_0
        mc = m[n, comp]
        off = np.einsum("ij,ij->i", sT, m)  # sum_j sT_j m_j
        # aniso enters ieps (times m_c^2) and T_cj (times m_c m_j)
        d_aniso = s_d * mc**2 + mc * off
        d_A = s_d - d_aniso
        d_B = d_aniso
        de = self.eps_max - self.eps_min
        d_f = d_A * (-de * t["A"] ** 2) + d_B * (1 / self.eps_max - 1 / self.eps_min)
        d_f = np.where(t["inside"], d_f, 0.0)
        # dF/dm_k: from ieps (2 a s_d m_c at k = c) and T_cj (a sT_j m_c, k = j;
        # a sum_j sT_j m_j at k = c)
        d_m = aniso[:, None] * sT * mc[:, None]
        d_m[n, comp] += aniso * (2 * s_d * mc + off)
        sq = np.sqrt(t["q"])
        d_g = (
            d_m / sq[:, None]
            - (np.einsum("ij,ij->i", d_m, t["g"]) / sq**3)[:, None] * t["g"]
        )
        grad = self.P.T @ d_f
        for k in range(3):
            grad = grad + self.G[k].T @ d_g[:, k]
        return np.asarray(grad).reshape(self.shape)
