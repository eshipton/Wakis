# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Single-block structured curvilinear hexahedral mesh for 3D nonorthogonal FIT.

A `HexBlock3D` is defined by a node coordinate array ``X`` of shape
``(Nx + 1, Ny + 1, Nz + 1, 3)`` [m]. Hexahedral cell ``(i, j, k)`` has the
eight corners ``X[i + a, j + b, k + c]`` with ``a, b, c in {0, 1}``.

Index conventions (C-order within each family, families concatenated in
``(x, y, z)`` order, matching the compact Cartesian FIT block layout without
ghost cells):

* Primal edges in three families ``(ex, ey, ez)``:
  - ``ex`` (axis 0): shape ``(Nx, Ny + 1, Nz + 1)``, connecting
    ``(i, j, k) -> (i + 1, j, k)``;
  - ``ey`` (axis 1): shape ``(Nx + 1, Ny, Nz + 1)``, connecting
    ``(i, j, k) -> (i, j + 1, k)``;
  - ``ez`` (axis 2): shape ``(Nx + 1, Ny + 1, Nz)``, connecting
    ``(i, j, k) -> (i, j, k + 1)``.
* Primal faces in three families ``(fx, fy, fz)``:
  - ``fx`` (axis 0): shape ``(Nx + 1, Ny, Nz)``;
  - ``fy`` (axis 1): shape ``(Nx, Ny + 1, Nz)``;
  - ``fz`` (axis 2): shape ``(Nx, Ny, Nz + 1)``.

Geometric quantities:

* Primal face vector area ``A_f = 1/2 (p2 - p0) x (p3 - p1)`` (shoelace /
  bilinear surface area of the four face corners ordered by the right-hand
  rule around the positive logical axis, exact for a uniform field).
* Cell volume ``V_hex`` integrated from the trilinear map Jacobian over the
  reference cube ``[0, 1]^3`` by ``2 x 2 x 2`` Gauss-Legendre quadrature
  (exact for the trilinear map), and corner volume ``V_c = V_hex / 8``.
* Dual face vector area ``S_e`` of each primal edge ``e``, summed over all
  hexahedra sharing ``e`` from the quadrilateral pieces
  ``(m_e, g_a, x_hex, g_b)`` evaluated with the shoelace formula.
"""

import numpy as np
import scipy.sparse as sp

__all__ = ["HexBlock3D"]


class HexBlock3D:
    """
    Single-block structured curvilinear hexahedral mesh.

    Parameters
    ----------
    X : array_like of shape (Nx + 1, Ny + 1, Nz + 1, 3)
        Node coordinates [m].

    Raises
    ------
    ValueError
        If ``X`` has invalid shape, non-finite coordinates, or any inverted
        or degenerate hexahedron (non-positive corner Jacobian or cell volume).
    """

    def __init__(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 4 or X.shape[3] != 3 or min(X.shape[:3]) < 2:
            raise ValueError(
                "X must have shape (Nx + 1, Ny + 1, Nz + 1, 3) with Nx, Ny, Nz >= 1"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("node coordinates X must be finite")

        self.X = X
        self.Nx, self.Ny, self.Nz = (
            X.shape[0] - 1,
            X.shape[1] - 1,
            X.shape[2] - 1,
        )
        self.nc = (self.Nx, self.Ny, self.Nz)
        self.nn = (self.Nx + 1, self.Ny + 1, self.Nz + 1)
        self.n_nodes = int(np.prod(self.nn))
        self.n_cells = int(np.prod(self.nc))

        self.edge_shape = []
        self.face_shape = []
        for a in range(3):
            es = list(self.nn)
            es[a] = self.nc[a]
            fs = list(self.nc)
            fs[a] = self.nn[a]
            self.edge_shape.append(tuple(es))
            self.face_shape.append(tuple(fs))

        ne = [int(np.prod(s)) for s in self.edge_shape]
        nf = [int(np.prod(s)) for s in self.face_shape]
        self.edge_off = np.concatenate([[0], np.cumsum(ne)])
        self.face_off = np.concatenate([[0], np.cumsum(nf)])
        self.n_edges = int(self.edge_off[-1])
        self.n_faces = int(self.face_off[-1])

        self.corner_jacobians = self._compute_corner_jacobians()
        if np.any(self.corner_jacobians <= 0.0):
            n_bad = int(np.sum(self.corner_jacobians <= 0.0))
            raise ValueError(
                f"inverted or degenerate hexahedron: {n_bad} non-positive corner Jacobians"
            )

        self.cell_volumes = self._compute_cell_volumes()
        if np.any(self.cell_volumes <= 0.0):
            raise ValueError("non-positive hexahedral cell volume")
        self.corner_volumes = self.cell_volumes / 8.0
        self.cell_centroids = self._compute_cell_centroids()

        (
            self.edge_vectors,
            self.edge_midpoints,
            self.edge_lengths,
            self.boundary_edge_mask,
        ) = self._compute_edge_geometry()
        self.unknown_edges = np.flatnonzero(~self.boundary_edge_mask)
        self.n_unknown_edges = int(self.unknown_edges.size)

        self.face_areas, self.face_centroids = self._compute_face_geometry()
        self.A_f = self.face_areas

        self.dual_face_pieces, self.dual_face_areas = self._compute_dual_faces()
        self.S_e = self.dual_face_areas

    @classmethod
    def cartesian(cls, x_nodes, y_nodes, z_nodes):
        """
        Construct a `HexBlock3D` from 1D Cartesian node arrays.

        Parameters
        ----------
        x_nodes, y_nodes, z_nodes : array_like
            1D node coordinates [m] along x, y, and z (nonuniform spacing
            allowed).

        Returns
        -------
        HexBlock3D
        """
        axes = [np.asarray(v, dtype=float) for v in (x_nodes, y_nodes, z_nodes)]
        for v in axes:
            if v.ndim != 1 or v.size < 2:
                raise ValueError(
                    "Cartesian node arrays must be 1D with at least 2 nodes"
                )
            if not np.all(np.isfinite(v)):
                raise ValueError("Cartesian node arrays must be finite")
        X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        return cls(X)

    # --------------------------------------------------------------- indexing
    def edge_index(self, a, i, j, k):
        """Flat global index of primal edge in family ``a`` at ``(i, j, k)``."""
        return self.edge_off[a] + np.ravel_multi_index((i, j, k), self.edge_shape[a])

    def face_index(self, a, i, j, k):
        """Flat global index of primal face in family ``a`` at ``(i, j, k)``."""
        return self.face_off[a] + np.ravel_multi_index((i, j, k), self.face_shape[a])

    def cell_index(self, i, j, k):
        """Flat global index of hexahedral cell ``(i, j, k)``."""
        return np.ravel_multi_index((i, j, k), self.nc)

    def edge_grid(self, a):
        """Index arrays ``(i, j, k)`` for all edges of family ``a``."""
        return np.meshgrid(*[np.arange(s) for s in self.edge_shape[a]], indexing="ij")

    def face_grid(self, a):
        """Index arrays ``(i, j, k)`` for all faces of family ``a``."""
        return np.meshgrid(*[np.arange(s) for s in self.face_shape[a]], indexing="ij")

    def cell_grid(self):
        """Index arrays ``(i, j, k)`` for all cells."""
        return np.meshgrid(*[np.arange(s) for s in self.nc], indexing="ij")

    # --------------------------------------------------------------- geometry
    def _compute_corner_jacobians(self):
        """Signed Jacobian determinant of the trilinear map at the 8 corners
        of every cell, shape ``(Nx, Ny, Nz, 2, 2, 2)`` [m^3]."""
        X = self.X
        Nx, Ny, Nz = self.nc
        J = np.empty((Nx, Ny, Nz, 2, 2, 2), dtype=float)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    dxi = X[1:, b : Ny + b, c : Nz + c] - X[:-1, b : Ny + b, c : Nz + c]
                    deta = (
                        X[a : Nx + a, 1:, c : Nz + c] - X[a : Nx + a, :-1, c : Nz + c]
                    )
                    dzeta = (
                        X[a : Nx + a, b : Ny + b, 1:] - X[a : Nx + a, b : Ny + b, :-1]
                    )
                    J[:, :, :, a, b, c] = np.einsum(
                        "...i,...i->...", np.cross(dxi, deta), dzeta
                    )
        return J

    def _compute_cell_volumes(self):
        """Exact trilinear hexahedron volume [m^3] via 2x2x2 Gauss-Legendre
        quadrature on ``[0, 1]^3``, shape ``(Nx, Ny, Nz)``."""
        X = self.X
        Nx, Ny, Nz = self.nc
        corners = np.empty((2, 2, 2, Nx, Ny, Nz, 3), dtype=float)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    corners[a, b, c] = X[a : Nx + a, b : Ny + b, c : Nz + c]

        dx = corners[1] - corners[0]  # (2, 2, Nx, Ny, Nz, 3) in (b, c)
        dy = corners[:, 1] - corners[:, 0]  # (2, 2, Nx, Ny, Nz, 3) in (a, c)
        dz = corners[:, :, 1] - corners[:, :, 0]  # (2, 2, Nx, Ny, Nz, 3) in (a, b)

        s = 0.5 / np.sqrt(3.0)
        g_pts = (0.5 - s, 0.5 + s)
        vol = np.zeros((Nx, Ny, Nz), dtype=float)
        for xi in g_pts:
            wx = (1.0 - xi, xi)
            for eta in g_pts:
                wy = (1.0 - eta, eta)
                for zeta in g_pts:
                    wz = (1.0 - zeta, zeta)
                    dxi = (
                        wy[0] * wz[0] * dx[0, 0]
                        + wy[0] * wz[1] * dx[0, 1]
                        + wy[1] * wz[0] * dx[1, 0]
                        + wy[1] * wz[1] * dx[1, 1]
                    )
                    deta = (
                        wx[0] * wz[0] * dy[0, 0]
                        + wx[0] * wz[1] * dy[0, 1]
                        + wx[1] * wz[0] * dy[1, 0]
                        + wx[1] * wz[1] * dy[1, 1]
                    )
                    dzeta = (
                        wx[0] * wy[0] * dz[0, 0]
                        + wx[0] * wy[1] * dz[0, 1]
                        + wx[1] * wy[0] * dz[1, 0]
                        + wx[1] * wy[1] * dz[1, 1]
                    )
                    vol += 0.125 * np.einsum(
                        "...i,...i->...", np.cross(dxi, deta), dzeta
                    )
        return vol

    def _compute_cell_centroids(self):
        """Hex centroids (average of the 8 corners) [m], shape ``(Nx, Ny, Nz, 3)``."""
        X = self.X
        c0 = 0.5 * (X[:-1] + X[1:])
        c01 = 0.5 * (c0[:, :-1] + c0[:, 1:])
        return 0.5 * (c01[:, :, :-1] + c01[:, :, 1:])

    def _compute_edge_geometry(self):
        """Primal edge vectors [m], midpoints [m], lengths [m], and tangential
        boundary mask."""
        X = self.X
        vecs, mids, bmask = [], [], []
        for a in range(3):
            s0 = [slice(None), slice(None), slice(None)]
            s1 = [slice(None), slice(None), slice(None)]
            s0[a] = slice(0, self.nc[a])
            s1[a] = slice(1, self.nc[a] + 1)
            p0 = X[tuple(s0)]
            p1 = X[tuple(s1)]
            vecs.append((p1 - p0).reshape(-1, 3))
            mids.append((0.5 * (p0 + p1)).reshape(-1, 3))

            ijk = self.edge_grid(a)
            on_bnd = np.zeros(self.edge_shape[a], dtype=bool)
            for ax in range(3):
                if ax != a:
                    on_bnd |= (ijk[ax] == 0) | (ijk[ax] == self.nc[ax])
            bmask.append(on_bnd.ravel())

        edge_vectors = np.concatenate(vecs, axis=0)
        edge_midpoints = np.concatenate(mids, axis=0)
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        boundary_edge_mask = np.concatenate(bmask, axis=0)
        return edge_vectors, edge_midpoints, edge_lengths, boundary_edge_mask

    def _compute_face_geometry(self):
        """Primal face shoelace vector areas ``A_f`` [m^2] and centroids [m]."""
        X = self.X
        areas, cens = [], []
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3

            def _corner(db, dc, b=b, c=c):
                sl = [slice(None), slice(None), slice(None)]
                sl[b] = slice(db, self.nc[b] + db)
                sl[c] = slice(dc, self.nc[c] + dc)
                return X[tuple(sl)]

            p0 = _corner(0, 0)
            p1 = _corner(1, 0)
            p2 = _corner(1, 1)
            p3 = _corner(0, 1)
            af = 0.5 * np.cross(p2 - p0, p3 - p1)
            gf = 0.5 * (0.5 * (p0 + p1) + 0.5 * (p3 + p2))
            areas.append(af.reshape(-1, 3))
            cens.append(gf.reshape(-1, 3))
        return np.concatenate(areas, axis=0), np.concatenate(cens, axis=0)

    def _compute_dual_faces(self):
        """Per-hex dual face quadrilateral pieces [m^2] and assembled whole dual
        face vector areas ``S_e`` [m^2] for each primal edge."""
        Nx, Ny, Nz = self.nc
        pieces = np.empty((Nx, Ny, Nz, 3, 2, 2, 3), dtype=float)
        S_e = np.zeros((self.n_edges, 3), dtype=float)

        ijk = self.cell_grid()
        x_hex = self.cell_centroids.reshape(-1, 3)

        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            for db in (0, 1):
                for dc in (0, 1):
                    q_e = [x.copy() for x in ijk]
                    q_e[b] = q_e[b] + db
                    q_e[c] = q_e[c] + dc
                    e_idx = self.edge_index(a, *q_e).ravel()

                    q_fb = [x.copy() for x in ijk]
                    q_fb[b] = q_fb[b] + db
                    fb_idx = self.face_index(b, *q_fb).ravel()

                    q_fc = [x.copy() for x in ijk]
                    q_fc[c] = q_fc[c] + dc
                    fc_idx = self.face_index(c, *q_fc).ravel()

                    m_e = self.edge_midpoints[e_idx]
                    g_fb = self.face_centroids[fb_idx]
                    g_fc = self.face_centroids[fc_idx]

                    # Right-hand rule around +u_a: for (1 - 2 db)(1 - 2 dc) > 0,
                    # the loop (m_e -> g_fc -> x_hex -> g_fb) is counter-clockwise.
                    sgn = (1 - 2 * db) * (1 - 2 * dc)
                    piece = (0.5 * sgn) * np.cross(x_hex - m_e, g_fb - g_fc)
                    pieces[:, :, :, a, db, dc, :] = piece.reshape(Nx, Ny, Nz, 3)
                    np.add.at(S_e, e_idx, piece)

        return pieces, S_e

    # -------------------------------------------------------------- operators
    def curl(self):
        """
        Topological curl matrix ``C`` (``n_faces x n_edges``) mapping primal
        edge voltages to primal face circulations, with entries in
        ``{-1, 0, +1}``.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        rows, cols, vals = [], [], []
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            ijk = self.face_grid(a)
            f = self.face_index(a, *ijk).ravel()
            for comp, shift, sign in (
                (b, None, 1.0),
                (c, b, 1.0),
                (b, c, -1.0),
                (c, None, -1.0),
            ):
                q = [x.copy() for x in ijk]
                if shift is not None:
                    q[shift] = q[shift] + 1
                e = self.edge_index(comp, *q).ravel()
                rows.append(f)
                cols.append(e)
                vals.append(np.full(f.size, sign, dtype=float))
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_faces, self.n_edges),
        )

    def divergence(self):
        """
        Topological cell-face divergence matrix ``D`` (``n_cells x n_faces``)
        with entries in ``{-1, 0, +1}``.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        rows, cols, vals = [], [], []
        ijk = self.cell_grid()
        cell = self.cell_index(*ijk).ravel()
        for a in range(3):
            for d, sign in ((0, -1.0), (1, 1.0)):
                q = [x.copy() for x in ijk]
                q[a] = q[a] + d
                f = self.face_index(a, *q).ravel()
                rows.append(cell)
                cols.append(f)
                vals.append(np.full(cell.size, sign, dtype=float))
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_cells, self.n_faces),
        )

    def gradient_unknown(self):
        """
        Topological node-to-edge gradient matrix ``G_u`` restricted to unknown
        edges and interior nodes, satisfying ``C_u @ G_u == 0``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Matrix of shape ``(n_unknown_edges, n_interior_nodes)``.
        """
        n_int_shape = (self.Nx - 1, self.Ny - 1, self.Nz - 1)
        if min(n_int_shape) <= 0:
            return sp.csr_matrix((self.n_unknown_edges, 0), dtype=float)
        n_int = int(np.prod(n_int_shape))
        unk_map = -np.ones(self.n_edges, dtype=np.int64)
        unk_map[self.unknown_edges] = np.arange(self.n_unknown_edges)

        rows, cols, vals = [], [], []
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            # Unknown edges of direction a have 1 <= ijk[b] <= nc[b]-1 and
            # 1 <= ijk[c] <= nc[c]-1.
            ranges = [None, None, None]
            ranges[a] = np.arange(self.nc[a])
            ranges[b] = np.arange(1, self.nc[b])
            ranges[c] = np.arange(1, self.nc[c])
            ijk = np.meshgrid(*ranges, indexing="ij")
            e_row = unk_map[self.edge_index(a, *ijk).ravel()]

            for shift, sign in ((0, -1.0), (1, 1.0)):
                na = ijk[a] + shift
                valid = (na >= 1) & (na <= self.nc[a] - 1)
                n_ijk = [x[valid] - 1 for x in ijk]
                n_ijk[a] = na[valid] - 1
                col = np.ravel_multi_index(tuple(n_ijk), n_int_shape)
                rows.append(e_row[valid.ravel()])
                cols.append(col.ravel())
                vals.append(np.full(col.size, sign, dtype=float))

        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_unknown_edges, n_int),
        )
