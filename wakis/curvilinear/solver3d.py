# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
3D nonorthogonal FIT on a structured curvilinear hexahedral block
(Schuhmann & Weiland 2000).

State variables (source-free, PEC outer boundary):

* ``b``: magnetic fluxes on all primal faces at half-integer times ``n - 1/2``
  [Wb];
* ``d``: electric fluxes on dual faces of unknown (non-PEC) primal edges at
  integer times ``n`` [C].

Tangential boundary edges lie on the PEC walls where ``e = 0`` and ``d = 0``
are fixed; boundary-normal edges and all interior edges are unknowns. With
``C_u = C[:, unknown_edges]`` and ``W_eps`` restricted to the unknown edges:

.. math::

    e^n &= W_\\varepsilon d^n, \\\\
    b^{n+1/2} &= b^{n-1/2} - \\Delta t \\, C_u e^n, \\\\
    h^{n+1/2} &= W_\\nu b^{n+1/2}, \\\\
    d^{n+1} &= d^n + \\Delta t \\, C_u^T h^{n+1/2}.

Both Hodges are assembled corner by corner (eight corners per hexahedron) in
the factored symmetric positive semi-definite form
``W = sum_{h, c} R_c^T (V_c / p_h) R_c``, where ``V_c = V_hex / 8`` and
``R_c`` (or ``Q_c``) reconstructs the local Cartesian vector ``B_c`` or
``D_c`` from the three primal-face fluxes or three dual-face fluxes meeting at
corner ``c`` (exact for uniform fields). At PEC boundary nodes, each electric
octant reconstructs ``D_c`` in the feasible subspace ``{D : D x n_k = 0}`` of
incident area-weighted wall normals ``n_k``: along ``span(n)`` by least-squares
fit over the non-wall (free) edges when all incident wall normals are parallel,
and ``D_c = 0`` at sharp PEC block edges and corners where incident wall
normals are not all parallel.

Discrete energy conservation
----------------------------
Define the discrete energy at step ``n``:

.. math::

    E^n = \\frac{1}{2} (d^n)^T W_\\varepsilon d^n
        + \\frac{1}{2} (b^{n+1/2})^T W_\\nu b^{n-1/2}.

Taking the difference ``E^{n+1} - E^n`` over one step and using the symmetry
``W_\\varepsilon^T = W_\\varepsilon`` and ``W_\\nu^T = W_\\nu``:

* the electric difference gives

  .. math::

      \\frac{1}{2} (d^{n+1} + d^n)^T W_\\varepsilon (d^{n+1} - d^n)
      = \\frac{\\Delta t}{2} (d^{n+1} + d^n)^T W_\\varepsilon C_u^T W_\\nu b^{n+1/2};

* the magnetic difference gives

  .. math::

      \\frac{1}{2} (b^{n+3/2} - b^{n-1/2})^T W_\\nu b^{n+1/2}
      = -\\frac{\\Delta t}{2} \\left(C_u W_\\varepsilon (d^{n+1} + d^n)\\right)^T W_\\nu b^{n+1/2}
      = -\\frac{\\Delta t}{2} (d^{n+1} + d^n)^T W_\\varepsilon C_u^T W_\\nu b^{n+1/2}.

The two contributions cancel identically, so ``E^{n+1} = E^n`` to floating-point
round-off.

Stability
---------
The quadratic form ``E^n`` is positive definite and the leapfrog update is
stable if and only if

.. math::

    \\Delta t < \\frac{2}{\\sqrt{\\lambda_{\\max}(W_\\varepsilon^{1/2} C_u^T W_\\nu C_u W_\\varepsilon^{1/2})}}
              = \\frac{2}{\\sqrt{\\lambda_{\\max}(C_u^T W_\\nu C_u W_\\varepsilon)}}.

Because ``W_\\nu = \\tilde{R}_\\nu^T \\tilde{R}_\\nu`` in factored corner form,
the nonzero spectrum of ``C_u^T W_\\nu C_u W_\\varepsilon`` coincides with that
of the symmetric positive semi-definite operator
``(\\tilde{R}_\\nu C_u) W_\\varepsilon (\\tilde{R}_\\nu C_u)^T``, whose largest
eigenvalue is computed with Lanczos (`scipy.sparse.linalg.eigsh`).
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0
from scipy.sparse.linalg import LinearOperator, eigs, eigsh

from .hexmesh import _CORNER_VTK, HexMesh
from .mesh3d import HexBlock3D

__all__ = ["CurvilinearFIT3D"]


def _cell_property(val, shape, n_cells, name):
    """Broadcast a scalar or array material property to ``shape`` and validate."""
    if val is None:
        return np.ones(shape, dtype=float)
    arr = np.asarray(val, dtype=float)
    if arr.ndim == 0:
        arr = np.full(shape, float(arr), dtype=float)
    elif arr.shape == (n_cells,) and shape != (n_cells,):
        arr = arr.reshape(shape)
    elif arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return arr


class CurvilinearFIT3D:
    """
    Single-block 3D curvilinear hexahedral FIT solver with explicit leapfrog
    time stepping and PEC outer boundaries.

    Parameters
    ----------
    block : HexBlock3D
        Structured curvilinear hexahedral mesh block.
    eps_r : array_like of shape (Nx, Ny, Nz) or float, optional
        Relative permittivity per hexahedron (default 1.0).
    mu_r : array_like of shape (Nx, Ny, Nz) or float, optional
        Relative permeability per hexahedron (default 1.0).
    dt : float, optional
        Time step [s]. Defaults to ``cfl * max_stable_dt()`` on first use.
    cfl : float, optional
        Default Courant fraction used when ``dt`` is not supplied.
    W_nu_override, W_eps_override : scipy.sparse.spmatrix, optional
        Override assembled Hodges (for mutation testing).
    _pec_reconstruction : {'constrained', 'drop_only'}, optional
        Electric-side PEC wall octant reconstruction mode (default
        ``'constrained'``; ``'drop_only'`` reproduces the unconstrained corner
        inverse for baseline/mutation testing).
    """

    def __init__(
        self,
        block,
        eps_r=None,
        mu_r=None,
        dt=None,
        cfl=0.5,
        W_nu_override=None,
        W_eps_override=None,
        _pec_reconstruction="constrained",
    ):
        if isinstance(block, HexBlock3D):
            self._is_block = True
            self.block = block
            self.mesh = HexMesh.from_block(block)
            self.mesh.node_normals = block.node_normals.reshape(-1, 3)
        elif isinstance(block, HexMesh):
            self._is_block = False
            self.mesh = block
            self.block = block
        else:
            raise TypeError("block must be a HexBlock3D or HexMesh instance")

        if _pec_reconstruction not in ("constrained", "drop_only"):
            raise ValueError(
                "_pec_reconstruction must be 'constrained' or 'drop_only', "
                f"got {_pec_reconstruction!r}"
            )
        self._pec_reconstruction = _pec_reconstruction
        self.eps_r = _cell_property(eps_r, block.nc, self.mesh.n_cells, "eps_r")
        self.mu_r = _cell_property(mu_r, block.nc, self.mesh.n_cells, "mu_r")
        self.eps = eps_0 * self.eps_r
        self.mu = mu_0 * self.mu_r

        self.n_faces = self.mesh.n_faces
        self.n_edges = self.mesh.n_edges
        self.unknown_edges = block.unknown_edges
        self.n_unknown_edges = block.n_unknown_edges

        C_msh = self.mesh.curl()
        W_nu_msh, R_nu_sqrt_msh = self._assemble_hodge_magnetic()
        W_eps_full_msh, Q_eps_sqrt_full_msh = self._assemble_hodge_electric()

        if self._is_block:
            pf = self.mesh.block_face_perm
            pe = self.mesh.block_edge_perm
            self.C = C_msh[pf, :][:, pe].tocsr()
            self.W_nu = W_nu_msh[pf, :][:, pf].tocsr()
            self._R_nu_sqrt = R_nu_sqrt_msh[:, pf].tocsr()
            self.W_eps_full = W_eps_full_msh[pe, :][:, pe].tocsr()
            self._Q_eps_sqrt_full = Q_eps_sqrt_full_msh[:, pe].tocsr()
        else:
            self.C = C_msh
            self.W_nu = W_nu_msh
            self._R_nu_sqrt = R_nu_sqrt_msh
            self.W_eps_full = W_eps_full_msh
            self._Q_eps_sqrt_full = Q_eps_sqrt_full_msh

        self.C_u = self.C[:, self.unknown_edges].tocsr()
        self.C_u_T = self.C_u.T.tocsr()

        self.W_eps_u = self.W_eps_full[self.unknown_edges, :][
            :, self.unknown_edges
        ].tocsr()
        self.W_eps = self.W_eps_u
        self._Q_eps_sqrt = self._Q_eps_sqrt_full[:, self.unknown_edges].tocsr()
        self.Q_eps_sqrt = self._Q_eps_sqrt

        self._has_override = W_nu_override is not None or W_eps_override is not None
        if W_nu_override is not None:
            self.W_nu = sp.csr_matrix(W_nu_override)
        if W_eps_override is not None:
            W_override = sp.csr_matrix(W_eps_override)
            if W_override.shape == (self.n_unknown_edges, self.n_unknown_edges):
                self.W_eps_u = W_override
                self.W_eps = W_override
            elif W_override.shape == (self.n_edges, self.n_edges):
                self.W_eps_full = W_override
                self.W_eps_u = W_override[self.unknown_edges, :][
                    :, self.unknown_edges
                ].tocsr()
                self.W_eps = self.W_eps_u
            else:
                raise ValueError("invalid W_eps_override shape")

        self.cfl = float(cfl)
        self._dt = None if dt is None else float(dt)
        self._dt_max = None

    @property
    def dt(self):
        """Time step [s]."""
        if self._dt is None:
            self._dt = self.cfl * self.max_stable_dt()
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = None if value is None else float(value)

    # --------------------------------------------------------------- assembly
    def _assemble_hodge_magnetic(self):
        """Assemble the symmetric PSD magnetic Hodge ``W_nu`` (``n_faces x n_faces``)
        and its factored square-root ``R_nu_sqrt``."""
        msh = self.mesh
        nc = msh.n_cells
        weight = (msh.corner_volumes / self.mu.ravel()).ravel()
        sqrt_w = np.sqrt(weight)

        rows, cols, vals = [], [], []
        r_rows, r_cols, r_vals = [], [], []
        A_f = msh.face_areas
        R_c_all = np.empty((2, 2, 2, nc, 3, 3), dtype=float)

        corner_id = 0
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    f0 = msh.hex_faces[:, 0, a]
                    f1 = msh.hex_faces[:, 1, b]
                    f2 = msh.hex_faces[:, 2, c]
                    f_loc = np.stack([f0, f1, f2], axis=1)  # (nc, 3)

                    s0 = msh.hex_face_signs[:, 0, a]
                    s1 = msh.hex_face_signs[:, 1, b]
                    s2 = msh.hex_face_signs[:, 2, c]
                    s_loc = np.stack([s0, s1, s2], axis=1)  # (nc, 3)

                    M_c = s_loc[:, :, None] * A_f[f_loc]  # (nc, 3, 3)
                    det = np.linalg.det(M_c)
                    scale = np.prod(np.linalg.norm(M_c, axis=2), axis=1)
                    if (
                        not np.all(np.isfinite(det))
                        or np.any(scale <= 0.0)
                        or np.any(np.abs(det) <= 1e-12 * scale)
                    ):
                        raise ValueError(
                            "singular corner face matrix M_c in magnetic Hodge"
                        )

                    R_c = np.linalg.inv(M_c) * s_loc[:, None, :]  # (nc, 3, 3)
                    R_c_all[a, b, c] = R_c
                    W_loc = weight[:, None, None] * np.einsum("nji,njk->nik", R_c, R_c)
                    for p in range(3):
                        for q in range(3):
                            rows.append(f_loc[:, p])
                            cols.append(f_loc[:, q])
                            vals.append(W_loc[:, p, q])

                    base = (corner_id * 3) * nc
                    for comp in range(3):
                        r_idx = base + comp * nc + np.arange(nc)
                        for q in range(3):
                            r_rows.append(r_idx)
                            r_cols.append(f_loc[:, q])
                            r_vals.append(sqrt_w * R_c[:, comp, q])
                    corner_id += 1

        self._R_c = R_c_all
        W_nu = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_faces, self.n_faces),
        )
        W_nu = (0.5 * (W_nu + W_nu.T)).tocsr()
        R_nu_sqrt = sp.csr_matrix(
            (
                np.concatenate(r_vals),
                (np.concatenate(r_rows), np.concatenate(r_cols)),
            ),
            shape=(24 * nc, self.n_faces),
        )
        return W_nu, R_nu_sqrt

    def _assemble_hodge_electric(self):
        """Assemble the symmetric PSD electric Hodge ``W_eps`` (``n_edges x n_edges``)
        and its factored square-root ``Q_eps_sqrt``."""
        msh = self.mesh
        nc = msh.n_cells
        weight = (msh.corner_volumes / self.eps.ravel()).ravel()
        sqrt_w = np.sqrt(weight)

        rows, cols, vals = [], [], []
        q_rows, q_cols, q_vals = [], [], []
        S_e = msh.dual_face_areas
        bnd_e = msh.boundary_edge_mask
        Q_c_all = np.empty((2, 2, 2, nc, 3, 3), dtype=float)
        n_wall_octants = 0
        n_sharp_octants = 0

        corner_id = 0
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    c_nodes = msh.hexes[:, _CORNER_VTK[a, b, c]]

                    e0 = msh.hex_edges[:, 0, b, c]
                    e1 = msh.hex_edges[:, 1, c, a]
                    e2 = msh.hex_edges[:, 2, a, b]
                    e_loc = np.stack([e0, e1, e2], axis=1)  # (nc, 3)

                    s0 = msh.hex_edge_signs[:, 0, b, c]
                    s1 = msh.hex_edge_signs[:, 1, c, a]
                    s2 = msh.hex_edge_signs[:, 2, a, b]
                    s_loc = np.stack([s0, s1, s2], axis=1)  # (nc, 3)

                    N_c = s_loc[:, :, None] * S_e[e_loc]  # (nc, 3, 3)
                    det = np.linalg.det(N_c)
                    scale = np.prod(np.linalg.norm(N_c, axis=2), axis=1)
                    if (
                        not np.all(np.isfinite(det))
                        or np.any(scale <= 0.0)
                        or np.any(np.abs(det) <= 1e-12 * scale)
                    ):
                        raise ValueError(
                            "singular corner dual-face matrix N_c in electric Hodge"
                        )

                    Q_c = np.linalg.inv(N_c) * s_loc[:, None, :]  # (nc, 3, 3)

                    on_bnd = msh.boundary_node_mask[c_nodes]
                    if np.any(on_bnd):
                        bnd_idx = np.flatnonzero(on_bnd)
                        is_sharp = msh.effective_sharp_nodes[c_nodes[bnd_idx]]
                        wall_oct = bnd_idx[~is_sharp]
                        sharp_oct = bnd_idx[is_sharp]
                        n_wall_octants += int(wall_oct.size)
                        n_sharp_octants += int(sharp_oct.size)

                        if self._pec_reconstruction == "constrained":
                            if sharp_oct.size > 0:
                                Q_c[sharp_oct] = 0.0
                            if wall_oct.size > 0:
                                n_vec = msh.node_normals[c_nodes[wall_oct]]
                                is_free = ~bnd_e[e_loc[wall_oct]]  # (n_wall, 3)
                                n_free = np.sum(is_free, axis=1)
                                S_sub = S_e[e_loc[wall_oct]]  # (n_wall, 3, 3)
                                sn = np.einsum("mqj,mj->mq", S_sub, n_vec) * is_free
                                denom = np.sum(sn * sn, axis=1)
                                s_scale = np.sum(
                                    np.sum(S_sub * S_sub, axis=2) * is_free,
                                    axis=1,
                                )
                                bad = (
                                    (n_free == 0)
                                    | (~np.isfinite(denom))
                                    | (denom <= 1e-24 * s_scale)
                                    | (s_scale <= 0.0)
                                )
                                if np.any(bad):
                                    b_i = wall_oct[int(np.flatnonzero(bad)[0])]
                                    raise ValueError(
                                        "degenerate wall octant in electric Hodge at node "
                                        f"{int(c_nodes[b_i])}"
                                    )
                                Q_c[wall_oct] = (
                                    n_vec[:, :, None] * sn[:, None, :]
                                ) / denom[:, None, None]

                    Q_c_all[a, b, c] = Q_c
                    W_loc = weight[:, None, None] * np.einsum("nji,njk->nik", Q_c, Q_c)
                    for p in range(3):
                        for q in range(3):
                            rows.append(e_loc[:, p])
                            cols.append(e_loc[:, q])
                            vals.append(W_loc[:, p, q])

                    base = (corner_id * 3) * nc
                    for comp in range(3):
                        r_idx = base + comp * nc + np.arange(nc)
                        for q in range(3):
                            q_rows.append(r_idx)
                            q_cols.append(e_loc[:, q])
                            q_vals.append(sqrt_w * Q_c[:, comp, q])
                    corner_id += 1

        self._Q_c = Q_c_all
        self.n_wall_octants = n_wall_octants
        self.n_sharp_octants = n_sharp_octants
        W_eps = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_edges, self.n_edges),
        )
        W_eps = (0.5 * (W_eps + W_eps.T)).tocsr()
        Q_eps_sqrt = sp.csr_matrix(
            (
                np.concatenate(q_vals),
                (np.concatenate(q_rows), np.concatenate(q_cols)),
            ),
            shape=(24 * nc, self.n_edges),
        )
        return W_eps, Q_eps_sqrt

    # -------------------------------------------------------------- operators
    def curl(self, restricted=False):
        """
        Return the topological curl matrix ``C``.

        Parameters
        ----------
        restricted : bool, optional
            If True, return ``C_u = C[:, unknown_edges]`` of shape
            ``(n_faces, n_unknown_edges)``. Default is False (full
            ``n_faces x n_edges`` matrix).

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        return self.C_u if restricted else self.C

    def hodge_magnetic(self):
        """
        Return the magnetic Hodge matrix ``W_nu`` (``n_faces x n_faces``) [1/H].

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        return self.W_nu

    def hodge_electric(self, restricted=True):
        """
        Return the electric Hodge matrix ``W_eps`` [1/F].

        Parameters
        ----------
        restricted : bool, optional
            If True (default), return the restricted matrix
            ``W_eps[unknown_edges, unknown_edges]`` of shape
            ``(n_unknown_edges, n_unknown_edges)``. If False, return the full
            ``(n_edges, n_edges)`` matrix including PEC boundary edges.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        return self.W_eps_u if restricted else self.W_eps_full

    # ---------------------------------------------------------- time stepping
    def step(self, b_half_prev, d, dt=None):
        """
        Advance one explicit leapfrog step ``(b^{n-1/2}, d^n) -> (b^{n+1/2}, d^{n+1})``.

        Parameters
        ----------
        b_half_prev : ndarray of shape (n_faces,)
            Primal face magnetic fluxes ``b^{n-1/2}`` [Wb].
        d : ndarray of shape (n_unknown_edges,)
            Unknown dual-face electric fluxes ``d^n`` [C].
        dt : float, optional
            Time step [s] (defaults to ``self.dt``).

        Returns
        -------
        b_half_next : ndarray of shape (n_faces,)
            Updated magnetic fluxes ``b^{n+1/2}`` [Wb].
        d_next : ndarray of shape (n_unknown_edges,)
            Updated electric fluxes ``d^{n+1}`` [C].
        """
        dt_val = self.dt if dt is None else float(dt)
        e = self.W_eps_u @ d
        b_half_next = b_half_prev - dt_val * (self.C_u @ e)
        h_half_next = self.W_nu @ b_half_next
        d_next = d + dt_val * (self.C_u_T @ h_half_next)
        return b_half_next, d_next

    def transpose_step(self, b_adj, d_adj, dt=None):
        """
        Exact transpose ``S^T`` of the linear one-step map ``S(b, d)``.

        Applies the transposed sub-steps in reverse order:

        .. math::

            \\hat{b} &= b_{\\text{adj}} + \\Delta t \\, W_\\nu^T C_u d_{\\text{adj}}, \\\\
            \\hat{d} &= d_{\\text{adj}} - \\Delta t \\, W_\\varepsilon^T C_u^T \\hat{b}.

        Parameters
        ----------
        b_adj : ndarray of shape (n_faces,)
            Co-state vector for ``b^{n+1/2}``.
        d_adj : ndarray of shape (n_unknown_edges,)
            Co-state vector for ``d^{n+1}``.
        dt : float, optional
            Time step [s] (defaults to ``self.dt``).

        Returns
        -------
        b_adj_prev : ndarray of shape (n_faces,)
            Transposed co-state for ``b^{n-1/2}``.
        d_adj_prev : ndarray of shape (n_unknown_edges,)
            Transposed co-state for ``d^n``.
        """
        dt_val = self.dt if dt is None else float(dt)
        b_hat = b_adj + dt_val * (self.W_nu.T @ (self.C_u @ d_adj))
        d_hat = d_adj - dt_val * (self.W_eps_u.T @ (self.C_u_T @ b_hat))
        return b_hat, d_hat

    def energy(self, b_half_prev, d, dt=None):
        """
        Conserved discrete energy ``E^n`` [J]:

        .. math::

            E^n = \\frac{1}{2} (d^n)^T W_\\varepsilon d^n
                + \\frac{1}{2} (b^{n+1/2})^T W_\\nu b^{n-1/2},

        where ``b^{n+1/2} = b^{n-1/2} - dt C_u W_eps d^n``.

        Parameters
        ----------
        b_half_prev : ndarray of shape (n_faces,)
            Magnetic flux ``b^{n-1/2}`` [Wb].
        d : ndarray of shape (n_unknown_edges,)
            Electric flux ``d^n`` [C].
        dt : float, optional
            Time step [s] (defaults to ``self.dt``).

        Returns
        -------
        float
            Discrete electromagnetic energy [J].
        """
        dt_val = self.dt if dt is None else float(dt)
        e = self.W_eps_u @ d
        b_half_next = b_half_prev - dt_val * (self.C_u @ e)
        return float(0.5 * (d @ e) + 0.5 * (b_half_next @ (self.W_nu @ b_half_prev)))

    # ------------------------------------------------- stability & resonances
    def max_stable_dt(self):
        """
        Leapfrog stability limit ``2 / sqrt(lambda_max(C_u^T W_nu C_u W_eps_u))`` [s].

        Computed via Lanczos (`scipy.sparse.linalg.eigsh`) on the isospectral
        symmetric positive semi-definite operator
        ``(R_nu_sqrt C_u) W_eps_u (R_nu_sqrt C_u)^T``.

        Returns
        -------
        float
            Maximum stable time step [s].
        """
        if self._dt_max is not None:
            return self._dt_max

        rng = np.random.default_rng(0)
        if not self._has_override:
            B = (self._R_nu_sqrt @ self.C_u).tocsr()
            BT = B.T.tocsr()
            W_e = self.W_eps_u
            n = B.shape[0]
            op = LinearOperator(
                shape=(n, n),
                matvec=lambda v: B @ (W_e @ (BT @ v)),
                dtype=float,
            )
            v0 = rng.standard_normal(n)
            lam = float(
                eigsh(
                    op,
                    k=1,
                    which="LM",
                    v0=v0,
                    tol=1e-10,
                    return_eigenvectors=False,
                )[0]
            )
        else:
            K = (self.C_u_T @ self.W_nu @ self.C_u @ self.W_eps_u).tocsr()
            v0 = rng.standard_normal(K.shape[0])
            lam = float(
                np.abs(
                    eigs(
                        K,
                        k=1,
                        which="LM",
                        v0=v0,
                        tol=1e-10,
                        return_eigenvectors=False,
                    )[0]
                )
            )

        self._dt_max = 2.0 / np.sqrt(lam)
        return self._dt_max

    def cavity_frequencies(self, k=6, return_vectors=False):
        """
        Lowest ``k`` positive discrete cavity resonant angular frequencies
        ``omega_h`` [rad/s], sorted in ascending order.

        Solves the semi-discrete eigenvalue problem
        ``C_u^T W_nu C_u W_eps_u d = omega_h^2 d`` via shift-invert Lanczos
        (`scipy.sparse.linalg.eigsh` with ``sigma=0``) on the equivalent
        symmetric positive-definite generalized system

        .. math::

            (W_{\\varepsilon, u} C_u^T W_\\nu C_u W_{\\varepsilon, u}
             + \\alpha G_u G_u^T) d = \\omega_h^2 W_{\\varepsilon, u} d,

        where ``G_u`` is the interior-node to unknown-edge topological gradient.
        Because ``C_u G_u = 0``, every physical cavity mode satisfies
        ``G_u^T d = 0`` and is completely unaffected by the penalty ``alpha``,
        while the static irrotational nullspace ``ker(C_u W_eps_u)`` is shifted
        above the physical modes of interest.

        Parameters
        ----------
        k : int, optional
            Number of lowest resonant angular frequencies to return (default 6).
        return_vectors : bool, optional
            If True, also return the corresponding unknown-edge electric flux
            eigenvectors ``d`` of shape ``(n_unknown_edges, k)`` [C] sorted in
            the same order as ``omega_h``. Default is False.

        Returns
        -------
        omega_h : ndarray of shape (k,)
            Sorted angular frequencies ``omega_h`` [rad/s].
        vecs : ndarray of shape (n_unknown_edges, k), optional
            Unknown-edge electric flux eigenvectors ``d`` [C], returned only
            when ``return_vectors=True``.
        """
        k = int(k)
        if k < 1 or k >= self.n_unknown_edges:
            raise ValueError(f"k must be in [1, {self.n_unknown_edges - 1}], got {k}")

        CW = (self.C_u @ self.W_eps_u).tocsr()
        A = (CW.T @ self.W_nu @ CW).tocsr()
        A = (0.5 * (A + A.T)).tocsr()

        G_u = self.block.gradient_unknown()
        if G_u.shape[1] > 0:
            alpha = float(np.max(A.diagonal()))
            H = (A + alpha * (G_u @ G_u.T)).tocsr()
        else:
            H = A

        rng = np.random.default_rng(0)
        v0 = rng.standard_normal(self.n_unknown_edges)
        if not return_vectors:
            w2 = eigsh(
                H,
                k=k,
                M=self.W_eps_u,
                sigma=0.0,
                which="LM",
                v0=v0,
                return_eigenvectors=False,
            )
            w2 = np.sort(np.maximum(w2.real, 0.0))
            return np.sqrt(w2[:k])

        w2, vecs = eigsh(
            H,
            k=k,
            M=self.W_eps_u,
            sigma=0.0,
            which="LM",
            v0=v0,
            return_eigenvectors=True,
        )
        order = np.argsort(w2.real)[:k]
        omegas = np.sqrt(np.maximum(w2.real[order], 0.0))
        return omegas, vecs[:, order].real
