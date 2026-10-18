# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Simplicial (triangle / tetrahedron) cell complexes for unstructured FIT.

`SimplexMesh` holds the primal grid G (nodes, edges, faces, cells), its
oriented incidence matrices (the topological gradient, curl and divergence
of the Finite Integration Technique) and the metric of the circumcentric
(Voronoi) dual grid. The dual quantities are stored per (primal object, cell)
pair so that materials that are constant per cell can be assembled exactly
into the material (Hodge) matrices, see the physics guide section
"Unstructured FIT".

Conventions
-----------
* Edges are oriented from the lower to the higher node index; ``G`` is the
  (n_edges x n_nodes) edge-node incidence (-1 start, +1 end).
* 3D faces are oriented by their sorted node triple (i < j < k) with normal
  (x_j - x_i) x (x_k - x_i); ``C`` (n_faces x n_edges) is the face-edge
  incidence (discrete curl) and ``D`` (n_cells x n_faces) the cell-face
  incidence with outward normals (discrete divergence). ``D @ C = 0`` and
  ``C @ G = 0`` hold exactly (integers).
* 2D cells are re-oriented counter-clockwise; ``C`` (n_cells x n_edges) is
  the triangle-edge incidence and ``C @ G = 0``.
* Dual measures are *signed*: a circumcentre lying outside its simplex gives
  a negative partial measure. For a Delaunay mesh the summed measures of all
  interior primal objects are non-negative; `diagnostics` reports
  violations.
"""

import numpy as np
import scipy.sparse as sp


def _incidence(rows, cols, vals, shape):
    return sp.csr_matrix(
        (np.asarray(vals, dtype=float), (np.ravel(rows), np.ravel(cols))),
        shape=shape,
    )


def _circumcenter_tri(P):
    """Circumcentres of triangles P (n, 3, d), d = 2 or 3."""
    a = P[:, 1] - P[:, 0]
    b = P[:, 2] - P[:, 0]
    aa = np.einsum("ij,ij->i", a, a)
    bb = np.einsum("ij,ij->i", b, b)
    ab = np.einsum("ij,ij->i", a, b)
    den = 2 * (aa * bb - ab**2)
    alpha = bb * (aa - ab) / den
    beta = aa * (bb - ab) / den
    return P[:, 0] + alpha[:, None] * a + beta[:, None] * b


def _circumcenter_tet(P):
    """Circumcentres of tetrahedra P (n, 4, 3)."""
    A = P[:, 1:] - P[:, :1]  # (n, 3, 3)
    rhs = 0.5 * np.einsum("nij,nij->ni", A, A)
    return P[:, 0] + np.linalg.solve(A, rhs[..., None])[..., 0]


class SimplexMesh:
    """
    Triangular (2D) or tetrahedral (3D) mesh with primal/dual FIT geometry.

    Parameters
    ----------
    nodes : array_like, shape (n_nodes, d)
        Node coordinates [m], d = 2 or 3.
    cells : array_like of int, shape (n_cells, d + 1)
        Node indices of each triangle / tetrahedron. Orientation is fixed
        internally (cells are re-ordered to positive volume).
    cell_tags : array_like of int, optional
        Material tag of each cell (default 0). Materials are attached to tags
        by the solvers.

    Attributes
    ----------
    dim : int
        Spatial dimension.
    edges : ndarray (n_edges, 2)
        Sorted node pairs.
    faces : ndarray (n_faces, 3)
        Sorted node triples (3D only; in 2D the faces are the cells).
    G, C, D : scipy.sparse.csr_matrix
        Gradient (edge-node), curl (face-edge) and divergence (cell-face, 3D)
        incidence matrices.
    edge_length, edge_tangent, edge_midpoint : ndarray
        Primal edge length L_e [m], unit tangent and midpoint.
    face_area : ndarray
        Primal face area A_f [m^2] (3D).
    cell_volume : ndarray
        Triangle area [m^2] or tetrahedron volume [m^3].
    cell_circumcenter : ndarray (n_cells, d)
        Circumcentres (the dual-grid nodes).
    """

    def __init__(self, nodes, cells, cell_tags=None):
        nodes = np.asarray(nodes, dtype=float)
        cells = np.array(cells, dtype=np.int64)
        self.dim = nodes.shape[1]
        if self.dim not in (2, 3) or cells.shape[1] != self.dim + 1:
            raise ValueError(
                "nodes must be (n, 2) with triangles (m, 3) or (n, 3) with "
                f"tetrahedra (m, 4); got {nodes.shape} and {cells.shape}"
            )
        used = np.zeros(len(nodes), bool)
        used[cells.ravel()] = True
        if not used.all():
            # Drop unreferenced nodes so that every node carries a dof
            remap = np.cumsum(used) - 1
            nodes, cells = nodes[used], remap[cells]
        self.nodes = nodes
        self.cell_tags = (
            np.zeros(len(cells), dtype=np.int64)
            if cell_tags is None
            else np.asarray(cell_tags, dtype=np.int64).copy()
        )
        if self.cell_tags.shape != (len(cells),):
            raise ValueError("cell_tags must have one entry per cell")
        self.cells = self._orient(cells)
        self._build_topology()
        self._build_geometry()

    # ------------------------------------------------------------ basics
    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def n_cells(self):
        return len(self.cells)

    @property
    def n_faces(self):
        return len(self.faces)

    def _signed_volume(self, cells):
        P = self.nodes[cells]
        A = P[:, 1:] - P[:, :1]
        return np.linalg.det(A) / (2 if self.dim == 2 else 6)

    def _orient(self, cells):
        vol = self._signed_volume(cells)
        if np.any(np.abs(vol) <= 1e-14 * np.max(np.abs(vol))):
            raise ValueError("mesh contains degenerate (zero-volume) cells")
        cells = cells.copy()
        neg = vol < 0
        cells[neg, :2] = cells[neg, 1::-1]
        return cells

    # ------------------------------------------------------------ topology
    def _build_topology(self):
        cells, d = self.cells, self.dim
        nc = len(cells)
        # Local edges of a simplex: all vertex pairs (a < b in local order)
        loc_e = np.array([(a, b) for a in range(d + 1) for b in range(a + 1, d + 1)])
        pairs = cells[:, loc_e]  # (nc, ne_loc, 2)
        key = np.sort(pairs.reshape(-1, 2), axis=1)
        self.edges, inv = np.unique(key, axis=0, return_inverse=True)
        self.cell_edges = inv.reshape(nc, len(loc_e))
        self._local_edges = loc_e
        ne = len(self.edges)
        self.G = _incidence(
            np.repeat(np.arange(ne), 2),
            self.edges.ravel(),
            np.tile([-1.0, 1.0], ne),
            (ne, self.n_nodes),
        )
        if d == 2:
            # Counter-clockwise boundary (v0 v1, v1 v2, v2 v0) of each triangle
            rows, cols, vals = [], [], []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                k = np.flatnonzero((loc_e == sorted((a, b))).all(1))[0]
                s = np.where(cells[:, a] < cells[:, b], 1.0, -1.0)
                rows.append(np.arange(nc))
                cols.append(self.cell_edges[:, k])
                vals.append(s)
            self.faces = cells
            self.cell_faces = np.arange(nc)[:, None]
            self.C = _incidence(
                np.concatenate(rows),
                np.concatenate(cols),
                np.concatenate(vals),
                (nc, ne),
            )
            self.D = None
            counts = np.bincount(self.cell_edges.ravel(), minlength=ne)
            self.boundary_edges = np.flatnonzero(counts == 1)
            bn = np.zeros(self.n_nodes, bool)
            bn[self.edges[self.boundary_edges].ravel()] = True
            self.boundary_nodes = np.flatnonzero(bn)
            return

        # 3D: local faces opposite each vertex
        loc_f = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
        tri = cells[:, loc_f]  # (nc, 4, 3)
        fkey = np.sort(tri.reshape(-1, 3), axis=1)
        self.faces, finv = np.unique(fkey, axis=0, return_inverse=True)
        self.cell_faces = finv.reshape(nc, 4)
        self._local_faces = loc_f
        nf = len(self.faces)
        # Face boundary i->j, j->k, k->i for the sorted triple (i, j, k)
        f = self.faces
        e_lookup = {tuple(e): n for n, e in enumerate(map(tuple, self.edges))}
        eij = np.array([e_lookup[(a, b)] for a, b in zip(f[:, 0], f[:, 1])])
        ejk = np.array([e_lookup[(a, b)] for a, b in zip(f[:, 1], f[:, 2])])
        eik = np.array([e_lookup[(a, b)] for a, b in zip(f[:, 0], f[:, 2])])
        self.face_edges = np.stack([eij, ejk, eik], 1)
        self.C = _incidence(
            np.repeat(np.arange(nf), 3),
            self.face_edges.ravel(),
            np.tile([1.0, 1.0, -1.0], nf),
            (nf, ne),
        )
        # Outward orientation of each local face relative to its sorted normal
        X = self.nodes
        n_sorted = np.cross(X[f[:, 1]] - X[f[:, 0]], X[f[:, 2]] - X[f[:, 0]])
        fc = X[f].mean(1)
        cc = X[cells].mean(1)
        out = np.einsum(
            "nkj,nkj->nk", n_sorted[self.cell_faces], fc[self.cell_faces] - cc[:, None]
        )
        self.D = _incidence(
            np.repeat(np.arange(nc), 4),
            self.cell_faces.ravel(),
            np.sign(out).ravel(),
            (nc, nf),
        )
        counts = np.bincount(self.cell_faces.ravel(), minlength=nf)
        self.boundary_faces = np.flatnonzero(counts == 1)
        be = np.zeros(ne, bool)
        be[self.face_edges[self.boundary_faces].ravel()] = True
        self.boundary_edges = np.flatnonzero(be)
        bn = np.zeros(self.n_nodes, bool)
        bn[self.faces[self.boundary_faces].ravel()] = True
        self.boundary_nodes = np.flatnonzero(bn)

    # ------------------------------------------------------------ geometry
    def _build_geometry(self):
        X, cells, d = self.nodes, self.cells, self.dim
        t = X[self.edges[:, 1]] - X[self.edges[:, 0]]
        self.edge_length = np.linalg.norm(t, axis=1)
        self.edge_tangent = t / self.edge_length[:, None]
        self.edge_midpoint = 0.5 * (X[self.edges[:, 1]] + X[self.edges[:, 0]])
        self.cell_volume = self._signed_volume(cells)
        P = X[cells]
        if d == 2:
            self.cell_circumcenter = _circumcenter_tri(P)
            self._dual_2d()
        else:
            self.cell_circumcenter = _circumcenter_tet(P)
            self._dual_3d()

    def _dual_2d(self):
        """Signed distances edge midpoint -> triangle circumcentre."""
        X, cells = self.nodes, self.cells
        cc = self.cell_circumcenter
        nc = self.n_cells
        h = np.zeros((nc, 3))  # per local edge (loc_e order)
        for k, (a, b) in enumerate(self._local_edges):
            opp = 3 - a - b
            m = 0.5 * (X[cells[:, a]] + X[cells[:, b]])
            tv = X[cells[:, b]] - X[cells[:, a]]
            nrm = np.stack([-tv[:, 1], tv[:, 0]], 1)
            nrm /= np.linalg.norm(nrm, axis=1)[:, None]
            # Orient the normal toward the opposite vertex
            s = np.sign(np.einsum("ij,ij->i", nrm, X[cells[:, opp]] - m))
            h[:, k] = s * np.einsum("ij,ij->i", nrm, cc - m)
        self.cell_dual_edge = h  # h_{e,T}: part of the dual edge of e in T
        ce = self.cell_edges.ravel()
        cols = np.repeat(np.arange(nc), 3)
        self.dual_edge_partial = sp.csr_matrix(
            (h.ravel(), (ce, cols)), shape=(self.n_edges, nc)
        )
        # Dual area of node i in T: two right triangles per incident edge
        L = self.edge_length[self.cell_edges]  # (nc, 3)
        part = 0.25 * L * h  # area between m_e, c_T and each edge end
        rows, vals = [], []
        for k, (a, b) in enumerate(self._local_edges):
            rows += [cells[:, a], cells[:, b]]
            vals += [part[:, k], part[:, k]]
        self.dual_area_partial = sp.csr_matrix(
            (
                np.concatenate(vals),
                (np.concatenate(rows), np.tile(np.arange(nc), 6)),
            ),
            shape=(self.n_nodes, nc),
        )
        # Barycentric dual (fallback): A_T / 3 per vertex, always positive
        self.dual_area_barycentric = sp.csr_matrix(
            (
                np.repeat(self.cell_volume / 3, 3),
                (cells.ravel(), np.repeat(np.arange(nc), 3)),
            ),
            shape=(self.n_nodes, nc),
        )

    def _dual_3d(self):
        """Signed partial dual areas (edge, tet) and dual lengths (face, tet)."""
        X, cells = self.nodes, self.cells
        nc = self.n_cells
        cT = self.cell_circumcenter
        # Face circumcentres and areas
        F = X[self.faces]
        self.face_circumcenter = _circumcenter_tri(F)
        nf_vec = np.cross(F[:, 1] - F[:, 0], F[:, 2] - F[:, 0])
        self.face_area = 0.5 * np.linalg.norm(nf_vec, axis=1)
        self.face_normal = nf_vec / (2 * self.face_area[:, None])
        cf = self.face_circumcenter
        # s_{f,T}: signed distance c_f -> c_T, positive toward the opposite vertex
        s_fT = np.zeros((nc, 4))
        for k in range(4):
            fid = self.cell_faces[:, k]
            nrm = self.face_normal[fid]
            sgn = np.sign(np.einsum("ij,ij->i", nrm, X[cells[:, k]] - cf[fid]))
            s_fT[:, k] = sgn * np.einsum("ij,ij->i", nrm, cT - cf[fid])
        self.cell_dual_edge = s_fT
        self.dual_edge_partial = sp.csr_matrix(
            (s_fT.ravel(), (self.cell_faces.ravel(), np.repeat(np.arange(nc), 4))),
            shape=(self.n_faces, nc),
        )
        # A_{e,T} = sum over the 2 faces f of T containing e of
        #           1/2 s_{e,f} s_{f,T}  (right triangle m_e, c_f, c_T)
        A_eT = np.zeros((nc, 6))
        for k, (a, b) in enumerate(self._local_edges):
            m = 0.5 * (X[cells[:, a]] + X[cells[:, b]])
            tv = X[cells[:, b]] - X[cells[:, a]]
            tv /= np.linalg.norm(tv, axis=1)[:, None]
            for opp_face_vertex in set(range(4)) - {a, b}:
                # local face not containing `opp_face_vertex` contains a, b
                third = ({0, 1, 2, 3} - {a, b, opp_face_vertex}).pop()
                fid = self.cell_faces[:, opp_face_vertex]
                w = X[cells[:, third]] - m
                w -= np.einsum("ij,ij->i", w, tv)[:, None] * tv
                w /= np.linalg.norm(w, axis=1)[:, None]
                s_ef = np.einsum("ij,ij->i", w, cf[fid] - m)
                A_eT[:, k] += 0.5 * s_ef * s_fT[:, opp_face_vertex]
        self.cell_dual_area = A_eT
        self.dual_area_partial = sp.csr_matrix(
            (A_eT.ravel(), (self.cell_edges.ravel(), np.repeat(np.arange(nc), 6))),
            shape=(self.n_edges, nc),
        )
        self._clip_interface_dual_areas()
        # Boundary dual length of boundary edges: sum over boundary faces of
        # the in-face distance m_e -> c_f (used by absorbing boundaries)
        bl = np.zeros(self.n_edges)
        bf = self.boundary_faces
        # face_edges = (ij, jk, ik): the third vertices are k, i, j
        for le, third_col in ((0, 2), (1, 0), (2, 1)):
            e = self.face_edges[bf, le]
            m = self.edge_midpoint[e]
            tv = self.edge_tangent[e]
            w = X[self.faces[bf, third_col]] - m
            w -= np.einsum("ij,ij->i", w, tv)[:, None] * tv
            w /= np.linalg.norm(w, axis=1)[:, None]
            np.add.at(bl, e, np.einsum("ij,ij->i", w, cf[bf] - m))
        self.boundary_dual_length = bl

    def _clip_interface_dual_areas(self):
        """
        Exact material split of the dual faces of material-interface edges.

        The dual (Voronoi) face of an edge lies in the bisector plane of the
        edge, and so do the tetrahedra around it as angular sectors bounded
        by the rays of their two faces through the edge. Attributing each
        signed piece (m_e, c_f, c_T) to its own tetrahedron T is exact only
        when c_T lies inside T. On edges where materials meet, every piece
        is instead clipped against the angular sectors of all tetrahedra
        around the edge, so that the part of the dual face inside each
        material is exact even for circumcentres outside their tetrahedron.
        Interior edges only (sectors cover the full angle).
        """
        X, cells, tags = self.nodes, self.cells, self.cell_tags
        ne = self.n_edges
        rows = self.cell_edges.ravel()
        cols = np.repeat(np.arange(self.n_cells), 6)
        locs = np.tile(np.arange(6), self.n_cells)
        order = np.argsort(rows, kind="stable")
        rows, cols, locs = rows[order], cols[order], locs[order]
        start = np.searchsorted(rows, np.arange(ne + 1))
        boundary = np.zeros(ne, bool)
        boundary[self.boundary_edges] = True
        P = self.dual_area_partial.tolil()
        cf, cT = self.face_circumcenter, self.cell_circumcenter
        changed = False
        for e in range(ne):
            ts, ks = cols[start[e] : start[e + 1]], locs[start[e] : start[e + 1]]
            if boundary[e] or np.unique(tags[ts]).size < 2:
                continue
            m = self.edge_midpoint[e]
            t = self.edge_tangent[e]
            u1 = np.cross(t, [1.0, 0, 0] if abs(t[0]) < 0.9 else [0, 1.0, 0])
            u1 /= np.linalg.norm(u1)
            u2 = np.cross(t, u1)

            def plane(x):
                d = np.asarray(x) - m
                return np.array([d @ u1, d @ u2])

            sectors, pieces = [], []
            for T, k in zip(ts, ks):
                la, lb = self._local_edges[k]
                o = [v for v in range(4) if v not in (la, lb)]
                # The face opposite o[1] contains o[0]; it meets the bisector
                # plane on the ray from m_e toward the projection of o[0]
                rays = [plane(X[cells[T, o[0]]]), plane(X[cells[T, o[1]]])]
                faces = [self.cell_faces[T, o[1]], self.cell_faces[T, o[0]]]
                ang = [np.arctan2(r[1], r[0]) for r in rays]
                mid = plane(X[cells[T, o]].mean(0))
                w = (ang[1] - ang[0]) % (2 * np.pi)
                if (np.arctan2(mid[1], mid[0]) - ang[0]) % (2 * np.pi) > w:
                    ang, faces = ang[::-1], faces[::-1]
                    w = 2 * np.pi - w
                sectors.append((ang[0], w, tags[T]))
                p0, p1, pc = plane(cf[faces[0]]), plane(cf[faces[1]]), plane(cT[T])
                pieces += [(p0, pc), (pc, p1)]
            area = {}
            for p, q in pieces:
                for tag, val in _clip_fan(p, q, sectors).items():
                    area[tag] = area.get(tag, 0.0) + val
            for T in ts:
                P[e, T] = 0.0
            for tag, val in area.items():
                P[e, ts[tags[ts] == tag][0]] = val
            changed = True
        if changed:
            self.dual_area_partial = P.tocsr()

    # ------------------------------------------------------------ dual sums
    def dual_measure(self, cell_weight=None):
        """
        Material-weighted dual measures of the electric primal objects.

        3D: sum_T w_T A_{e,T} (dual face area of each edge) [m^2];
        2D: sum_T w_T A_{n,T} (dual area of each node) [m^2].

        Parameters
        ----------
        cell_weight : array_like, optional
            Per-cell factor (e.g. relative permittivity); 1 if None.
        """
        w = np.ones(self.n_cells) if cell_weight is None else cell_weight
        P = self.dual_area_partial
        return np.asarray(P @ np.asarray(w, dtype=float)).ravel()

    def dual_length(self, cell_weight=None):
        """
        Material-weighted dual edge lengths sum_T w_T h_{x,T} [m].

        3D: one per face (dual edge through face f); 2D: one per edge.
        """
        w = np.ones(self.n_cells) if cell_weight is None else cell_weight
        P = self.dual_edge_partial
        return np.asarray(P @ np.asarray(w, dtype=float)).ravel()

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self, rtol=1e-10):
        """
        Delaunay / well-centredness report of the circumcentric dual.

        Returns
        -------
        report : dict
            ``n_cells``, ``h_mean`` (mean edge length [m]), counts and
            fractions of negative cell-wise dual partials
            (circumcentre outside its simplex: ``not_well_centered``),
            negative summed dual measures on interior primal objects
            (violations of the Delaunay property: ``negative_dual_*``),
            minimum normalized dual measures and the minimum cell quality
            (radius ratio, 1 for regular simplices).
        """
        d = self.dim
        h = self.edge_length.mean()
        rep = {"dim": d, "n_cells": self.n_cells, "h_mean": h}
        rep["not_well_centered"] = int(np.sum(np.any(self.cell_dual_edge < 0, axis=1)))
        if d == 2:
            interior_e = np.setdiff1d(np.arange(self.n_edges), self.boundary_edges)
            lens = self.dual_length()
            rep["negative_dual_length"] = int(np.sum(lens[interior_e] < -rtol * h))
            rep["min_dual_length_rel"] = float(lens[interior_e].min() / h)
            area = self.dual_measure()
            rep["negative_dual_area"] = int(np.sum(area < -rtol * h**2))
            rep["min_dual_area_rel"] = float(area.min() / h**2)
        else:
            interior_e = np.setdiff1d(np.arange(self.n_edges), self.boundary_edges)
            interior_f = np.setdiff1d(np.arange(self.n_faces), self.boundary_faces)
            area = self.dual_measure()
            lens = self.dual_length()
            rep["negative_dual_area"] = int(np.sum(area[interior_e] < -rtol * h**2))
            rep["negative_dual_length"] = int(np.sum(lens[interior_f] < -rtol * h))
            rep["min_dual_area_rel"] = float(area[interior_e].min() / h**2)
            rep["min_dual_length_rel"] = float(lens[interior_f].min() / h)
            rep["negative_boundary_dual_area"] = int(
                np.sum(area[self.boundary_edges] < -rtol * h**2)
            )
        rep["min_quality"] = float(self.quality().min())
        return rep

    def quality(self):
        """Radius ratio d * r_in / R_circ of each cell (1 = regular simplex)."""
        X, cells, d = self.nodes, self.cells, self.dim
        R = np.linalg.norm(self.cell_circumcenter - X[cells[:, 0]], axis=1)
        if d == 2:
            L = self.edge_length[self.cell_edges]
            r_in = 2 * self.cell_volume / L.sum(1)
        else:
            surf = (
                self.face_area[self.cell_faces].sum(1)
                if hasattr(self, "face_area")
                else None
            )
            r_in = 3 * self.cell_volume / surf
        return d * r_in / R

    # ------------------------------------------------------------ queries
    def locate(self, points, tol=1e-12):
        """
        Containing cell and barycentric coordinates of points.

        Parameters
        ----------
        points : array_like (n, d)
            Query points [m].

        Returns
        -------
        cell : ndarray of int
            Cell index of each point (-1 if outside the mesh).
        bary : ndarray (n, d + 1)
            Barycentric coordinates in that cell.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        X0 = self.nodes[self.cells[:, 0]]
        A = self.nodes[self.cells[:, 1:]] - X0[:, None]  # (nc, d, d) rows = edges
        Ainv = np.linalg.inv(np.transpose(A, (0, 2, 1)))
        cell = np.full(len(pts), -1)
        bary = np.zeros((len(pts), self.dim + 1))
        for n, p in enumerate(pts):
            lam = np.einsum("nij,nj->ni", Ainv, p - X0)
            lam = np.concatenate([1 - lam.sum(1, keepdims=True), lam], 1)
            ok = np.flatnonzero(lam.min(1) >= -tol)
            if ok.size:
                c = ok[np.argmax(lam[ok].min(1))]
                cell[n], bary[n] = c, lam[c]
        return cell, bary

    def barycentric_gradients(self, cells=None):
        """Gradients of the barycentric coordinates, shape (n, d + 1, d) [1/m]."""
        cells = np.arange(self.n_cells) if cells is None else np.asarray(cells)
        X0 = self.nodes[self.cells[cells, 0]]
        A = self.nodes[self.cells[cells, 1:]] - X0[:, None]
        g = np.transpose(np.linalg.inv(A), (0, 2, 1))  # grads of lambda_1..d
        g0 = -g.sum(1, keepdims=True)
        return np.concatenate([g0, g], 1)

    def whitney_edge(self, points):
        """
        Whitney edge-function values at points.

        Returns
        -------
        cell : ndarray of int
            Containing cell of each point.
        W : ndarray (n, n_local_edges, d)
            W_e = lambda_a grad lambda_b - lambda_b grad lambda_a for the local
            edges (a, b) of the cell, oriented along the global edge [1/m].
        """
        cell, lam = self.locate(points)
        if np.any(cell < 0):
            raise ValueError("point outside the mesh")
        grad = self.barycentric_gradients(cell)
        W = []
        for a, b in self._local_edges:
            w = lam[:, a, None] * grad[:, b] - lam[:, b, None] * grad[:, a]
            sgn = np.where(self.cells[cell, a] < self.cells[cell, b], 1.0, -1.0)
            W.append(sgn[:, None] * w)
        return cell, np.stack(W, 1)

    def edges_to_cell_vectors(self, e):
        """
        Constant-per-cell vector field from edge voltages (Whitney, centroid).

        Parameters
        ----------
        e : array_like (n_edges,)
            Line integrals along the oriented edges [V for E].

        Returns
        -------
        ndarray (n_cells, d)
            Field at the cell centroids [V/m].
        """
        grad = self.barycentric_gradients()
        lam = 1.0 / (self.dim + 1)
        out = np.zeros((self.n_cells, self.dim), dtype=np.result_type(e, float))
        for k, (a, b) in enumerate(self._local_edges):
            sgn = np.where(self.cells[:, a] < self.cells[:, b], 1.0, -1.0)
            out += (
                (sgn * e[self.cell_edges[:, k]])[:, None]
                * lam
                * (grad[:, b] - grad[:, a])
            )
        return out

    # ------------------------------------------------------------ I/O
    def to_meshio(self):
        """`meshio.Mesh` with the cells and their ``tag`` cell data."""
        import meshio

        kind = "triangle" if self.dim == 2 else "tetra"
        return meshio.Mesh(
            self.nodes, [(kind, self.cells)], cell_data={"tag": [self.cell_tags]}
        )

    @classmethod
    def from_meshio(cls, mesh, tag_key=None):
        """
        Build from a `meshio.Mesh` (e.g. ``meshio.read("part.msh")``).

        The simplices of highest dimension are used (tetrahedra if present,
        else triangles). Cell tags are read from ``cell_data[tag_key]``; by
        default the first of ``gmsh:physical``, ``tag``, ``gmsh:geometrical``
        that exists is used.

        Parameters
        ----------
        mesh : meshio.Mesh
        tag_key : str, optional
            Name of the integer cell data holding material tags.
        """
        kind = "tetra" if "tetra" in mesh.cells_dict else "triangle"
        dim = 3 if kind == "tetra" else 2
        blocks = [i for i, c in enumerate(mesh.cells) if c.type == kind]
        cells = np.concatenate([mesh.cells[i].data for i in blocks])
        keys = [tag_key] if tag_key else ["gmsh:physical", "tag", "gmsh:geometrical"]
        tags = None
        for k in keys:
            if k in mesh.cell_data:
                tags = np.concatenate([mesh.cell_data[k][i] for i in blocks])
                break
        nodes = mesh.points[:, :dim]
        return cls(nodes, cells, tags)

    @classmethod
    def read(cls, filename, tag_key=None):
        """Read any mesh file supported by meshio (``.msh``, ``.vtu``, ...)."""
        import meshio

        return cls.from_meshio(meshio.read(filename), tag_key=tag_key)

    def write(self, filename):
        """Write the mesh and cell tags with meshio (format from the suffix)."""
        self.to_meshio().write(filename)


def _clip_fan(p, q, sectors):
    """
    Signed area of the triangle (0, p, q) inside each angular sector.

    Parameters
    ----------
    p, q : (2,) float
        Triangle vertices in a plane whose origin is the apex.
    sectors : list of (start angle, width, tag)
        Counter-clockwise sectors.

    Returns
    -------
    dict
        tag -> signed area (positive for counter-clockwise (0, p, q)).
    """
    cross = p[0] * q[1] - p[1] * q[0]
    out = {}
    if abs(cross) < 1e-300:
        return out
    th_p = np.arctan2(p[1], p[0])
    th_q = np.arctan2(q[1], q[0])
    sign = 1.0 if cross > 0 else -1.0
    start, p0, q0 = (th_p, p, q) if sign > 0 else (th_q, q, p)
    sweep = (np.arctan2(q0[1], q0[0]) - start) % (2 * np.pi)
    d = q0 - p0

    def point(phi):
        r = np.array([np.cos(phi), np.sin(phi)])
        t = -(r[0] * p0[1] - r[1] * p0[0]) / (r[0] * d[1] - r[1] * d[0])
        return p0 + t * d

    for a0, w, tag in sectors:
        lo = (a0 - start) % (2 * np.pi)
        # overlap of [0, sweep] with [lo, lo + w] modulo 2 pi
        for shift in (lo - 2 * np.pi, lo):
            x0, x1 = max(0.0, shift), min(sweep, shift + w)
            if x1 > x0:
                pa, pb = point(start + x0), point(start + x1)
                val = 0.5 * (pa[0] * pb[1] - pa[1] * pb[0])
                out[tag] = out.get(tag, 0.0) + sign * val
    return out
