# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
General polygonal 2D mesh (cells, edges, nodes) with FIT dual geometry.

Every cell is a simple polygon (triangle, rectangle, Voronoi polygon, ...)
with a *dual point* c_C. The dual grid joins the dual points of neighbouring
cells through the shared primal edge: within cell C the dual edge of primal
edge e is the segment from c_C to its foot point x_{e,C} on the line of e.

Two geometric quantities per (edge, cell) pair define the FIT Hodge:

* h_{e,C}: signed distance from c_C to the line of e (positive when c_C is on
  the cell side), the part of the dual edge of e inside C;
* s_{e,C}: position of the foot point along e, measured from the first edge
  node (edges[e, 0]).

The dual grid is **orthogonal** at an interior edge when both cells have the
same foot point (s_{e,C1} = s_{e,C2}): the dual edge c_C1 -> c_C2 is then a
straight segment perpendicular to e, and a diagonal Hodge is consistent
(exact for uniform fields on the "diamond" of the edge). This holds for

* Delaunay triangles with circumcentres (the foot is the edge midpoint),
* rectangles with their centres (midpoint),
* hybrid meshes of the two (shared edges have midpoint feet on both sides),
* Voronoi polygons with their generators as dual points (the generators of
  two neighbouring cells are mirror images across the shared edge; the foot
  is the midpoint of the generator pair, not of the Voronoi edge).

Otherwise a non-diagonal, symmetric positive definite *mimetic* inner product
per cell is available (`mfd_local_matrices`), see the physics guide.
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

__all__ = ["PolyMesh"]


def _shoelace(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


class PolyMesh:
    """
    2D mesh of simple polygons with FIT primal/dual geometry.

    Parameters
    ----------
    nodes : array_like (n_nodes, 2)
        Node coordinates [m].
    cells : sequence of sequences of int
        Node indices of each polygon (any orientation; re-ordered
        counter-clockwise). Consecutive nodes are joined by edges.
    cell_tags : array_like of int, optional
        Material tag per cell (default 0).
    dual_points : array_like (n_cells, 2), optional
        Dual point of each cell (e.g. the Voronoi generators). Default: the
        circumcentre of cyclic polygons (triangles, rectangles, regular
        polygons), else the centroid.
    cyclic_rtol : float
        Relative spread of the vertex distances below which a polygon is
        treated as cyclic for the default dual point.

    Attributes
    ----------
    edges : ndarray (n_edges, 2)
        Sorted node pairs; the tangent t_e points from edges[e, 0] to
        edges[e, 1] and the flux normal is n_e = t_e x z.
    G : csr_matrix (n_edges, n_nodes)
        Edge-node incidence (discrete gradient).
    corner_cell, corner_edge, corner_sign : ndarray (n_corners,)
        One entry per (cell, edge) pair; sign +1 when n_e is the outward
        normal of the cell.
    corner_h, corner_s : ndarray (n_corners,)
        h_{e,C} [m] and s_{e,C} [m] (see the module docstring).
    cell_area, cell_centroid, dual_point : ndarray
    orthogonality_defect : ndarray (n_edges,)
        |s_{e,C1} - s_{e,C2}| / L_e at interior edges (0 at boundary edges).
    """

    def __init__(self, nodes, cells, cell_tags=None, dual_points=None, cyclic_rtol=1e-8):
        nodes = np.asarray(nodes, float)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes must be (n, 2)")
        cells = [np.asarray(c, np.int64) for c in cells]
        used = np.zeros(len(nodes), bool)
        for c in cells:
            if len(c) < 3:
                raise ValueError("cells need at least 3 nodes")
            used[c] = True
        if not used.all():
            remap = np.cumsum(used) - 1
            nodes = nodes[used]
            cells = [remap[c] for c in cells]
        oriented = []
        for c in cells:
            oriented.append(c if _shoelace(nodes[c]) > 0 else c[::-1])
        self.nodes = nodes
        self.cell_ptr = np.concatenate([[0], np.cumsum([len(c) for c in oriented])])
        self.cell_nodes = np.concatenate(oriented)
        self.cell_size = np.diff(self.cell_ptr)
        nc = len(oriented)
        self.cell_tags = (
            np.zeros(nc, np.int64)
            if cell_tags is None
            else np.asarray(cell_tags, np.int64).copy()
        )
        if self.cell_tags.shape != (nc,):
            raise ValueError("cell_tags must have one entry per cell")
        self._build_topology()
        self._build_geometry(dual_points, cyclic_rtol)

    # ------------------------------------------------------------ basics
    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def n_cells(self):
        return len(self.cell_ptr) - 1

    def cell(self, i):
        """Node indices of cell `i` (counter-clockwise)."""
        return self.cell_nodes[self.cell_ptr[i] : self.cell_ptr[i + 1]]

    def _build_topology(self):
        ptr, cn = self.cell_ptr, self.cell_nodes
        nc = self.n_cells
        self.corner_cell = np.repeat(np.arange(nc), self.cell_size)
        # next node within each cell
        nxt = np.arange(len(cn)) + 1
        nxt[ptr[1:] - 1] = ptr[:-1]
        a, b = cn, cn[nxt]
        self.corner_node = a
        self.corner_next = b
        pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], 1)
        self.edges, inv = np.unique(pairs, axis=0, return_inverse=True)
        inv = inv.ravel()
        self.corner_edge = inv
        self.corner_sign = np.where(a == self.edges[inv, 0], 1, -1)
        cnt = np.bincount(inv, minlength=len(self.edges))
        if np.any(cnt > 2):
            raise ValueError("an edge is shared by more than two cells")
        if np.any(np.bincount(inv * 2 + (self.corner_sign > 0), minlength=2 * len(self.edges)) > 1):
            raise ValueError("inconsistent cell orientation (overlapping cells)")
        self.boundary_edges = np.flatnonzero(cnt == 1)
        bn = np.zeros(self.n_nodes, bool)
        bn[self.edges[self.boundary_edges].ravel()] = True
        self.boundary_nodes = np.flatnonzero(bn)
        ne = len(self.edges)
        self.G = sp.csr_matrix(
            (
                np.r_[-np.ones(ne), np.ones(ne)],
                (np.r_[np.arange(ne), np.arange(ne)], np.r_[self.edges[:, 0], self.edges[:, 1]]),
            ),
            shape=(ne, self.n_nodes),
        )
        # cells of each edge (second = -1 on the boundary)
        ec = np.full((ne, 2), -1)
        order = np.argsort(inv, kind="stable")
        first = np.r_[True, inv[order][1:] != inv[order][:-1]]
        ec[inv[order][first], 0] = self.corner_cell[order][first]
        ec[inv[order][~first], 1] = self.corner_cell[order][~first]
        self.edge_cells = ec

    def _build_geometry(self, dual_points, rtol):
        X = self.nodes
        t = X[self.edges[:, 1]] - X[self.edges[:, 0]]
        self.edge_length = np.linalg.norm(t, axis=1)
        if np.any(self.edge_length <= 0):
            raise ValueError("zero-length edge")
        self.edge_tangent = t / self.edge_length[:, None]
        self.edge_normal = np.stack([self.edge_tangent[:, 1], -self.edge_tangent[:, 0]], 1)
        self.edge_midpoint = 0.5 * (X[self.edges[:, 0]] + X[self.edges[:, 1]])
        nc = self.n_cells
        C = self.corner_cell
        P, Q = X[self.corner_node], X[self.corner_next]
        cr = P[:, 0] * Q[:, 1] - Q[:, 0] * P[:, 1]
        area = 0.5 * np.bincount(C, cr, minlength=nc)
        cen = np.stack(
            [
                np.bincount(C, (P[:, 0] + Q[:, 0]) * cr, minlength=nc),
                np.bincount(C, (P[:, 1] + Q[:, 1]) * cr, minlength=nc),
            ],
            1,
        ) / (6 * area[:, None])
        # least-squares circumcentre per cell, batched by polygon size
        dual = cen.copy()
        self.cell_cyclic = np.zeros(nc, bool)
        if dual_points is None:
            for m in np.unique(self.cell_size):
                sel = np.flatnonzero(self.cell_size == m)
                idx = self.cell_ptr[sel][:, None] + np.arange(m)
                V = X[self.cell_nodes[idx]]  # (k, m, 2)
                A = 2 * (V[:, 1:] - V[:, :1])
                b = np.sum(V[:, 1:] ** 2 - V[:, :1] ** 2, axis=2)
                AtA = np.einsum("kij,kil->kjl", A, A)
                Atb = np.einsum("kij,ki->kj", A, b)
                c = np.linalg.solve(AtA, Atb[..., None])[..., 0]
                r = np.linalg.norm(V - c[:, None], axis=2)
                res = (r.max(1) - r.min(1)) / r.mean(1)
                cyc = (res < rtol) | (m == 3)
                self.cell_cyclic[sel] = cyc
                dual[sel[cyc]] = c[cyc]
        if dual_points is not None:
            dual = np.asarray(dual_points, float).copy()
            if dual.shape != (nc, 2):
                raise ValueError("dual_points must be (n_cells, 2)")
        self.cell_area = area
        self.cell_centroid = cen
        self.dual_point = dual
        self._corner_geometry()

    def _corner_geometry(self):
        e, C, sg = self.corner_edge, self.corner_cell, self.corner_sign
        X0 = self.nodes[self.edges[e, 0]]
        t, n = self.edge_tangent[e], self.edge_normal[e]
        d = self.dual_point[C] - X0
        self.corner_s = np.einsum("ij,ij->i", d, t)
        self.corner_h = -sg * np.einsum("ij,ij->i", d, n)
        # same with the centroid and the edge midpoint (median dual)
        dm = self.cell_centroid[C] - X0
        self.corner_h_centroid = -sg * np.einsum("ij,ij->i", dm, n)
        ne = self.n_edges
        defect = np.zeros(ne)
        two = self.edge_cells[:, 1] >= 0
        s_sum = np.bincount(e, self.corner_s, minlength=ne)
        s_sq = np.bincount(e, self.corner_s**2, minlength=ne)
        # for two cells: |s1 - s2| = sqrt(2 (s1^2 + s2^2) - (s1 + s2)^2)
        defect[two] = np.sqrt(np.clip(2 * s_sq[two] - s_sum[two] ** 2, 0, None))
        self.orthogonality_defect = defect / self.edge_length

    # ------------------------------------------------------------ dual measures
    def dual_edge_partial(self):
        """(n_edges, n_cells) csr of h_{e,C} [m]."""
        return sp.csr_matrix(
            (self.corner_h, (self.corner_edge, self.corner_cell)),
            shape=(self.n_edges, self.n_cells),
        )

    def dual_area_partial(self, kind="orthogonal"):
        """
        (n_nodes, n_cells) csr of the part A_{n,C} [m^2] of the dual cell of
        node n inside cell C.

        kind='orthogonal': quadrilaterals (n, x_{e1,C}, c_C, x_{e2,C}) built
        on the dual point and its foot points (the Voronoi/circumcentric dual;
        sums to the cell area); kind='median': the same with the centroid and
        the edge midpoints (positive on convex cells).
        """
        e, C = self.corner_edge, self.corner_cell
        L = self.edge_length[e]
        if kind == "orthogonal":
            s, h = self.corner_s, self.corner_h
            a0, a1 = 0.5 * s * h, 0.5 * (L - s) * h
        elif kind == "median":
            h = self.corner_h_centroid
            a0 = a1 = 0.25 * L * h
        else:
            raise ValueError("kind must be 'orthogonal' or 'median'")
        rows = np.r_[self.edges[e, 0], self.edges[e, 1]]
        return sp.csr_matrix(
            (np.r_[a0, a1], (rows, np.r_[C, C])), shape=(self.n_nodes, self.n_cells)
        )

    def boundary_node_length(self, kind="orthogonal"):
        """Length of the outer boundary in the dual cell of each node [m]."""
        out = np.zeros(self.n_nodes)
        be = self.boundary_edges
        L = self.edge_length[be]
        if kind == "orthogonal":
            pos = np.zeros(self.n_edges)
            pos[self.corner_edge] = self.corner_s  # one corner per boundary edge
            s = np.clip(pos[be], 0, L)
        else:
            s = 0.5 * L
        np.add.at(out, self.edges[be, 0], s)
        np.add.at(out, self.edges[be, 1], L - s)
        return out

    def mfd_local_matrices(self, gamma_scale=1.0):
        """
        Mimetic (consistency + stability) flux inner products per cell.

        For cell C with edges e (outward normals n_e, lengths L_e, midpoints
        m_e) and centroid x_C, the fluxes of a uniform field u are F = N u with
        N_e = L_e n_e^T, and R_e = (m_e - x_C)^T satisfies R^T N = |C| I
        (divergence theorem). The local matrix

            M_C = R R^T / |C| + gamma (I - N (N^T N)^-1 N^T),
            gamma = gamma_scale * trace(R R^T) / (|C| n_C)

        is symmetric positive definite and exact for uniform fields:
        F^T M_C F = |C| |u|^2 and M_C N = R (Brezzi, Lipnikov & Shashkov 2005).
        Entries are in the outward-flux orientation of the cell; multiply by
        corner_sign on both sides for the global edge orientation.

        Returns
        -------
        list of ndarray (n_C, n_C), ordered like the cell's corners.
        """
        out = []
        for i in range(self.n_cells):
            k = slice(self.cell_ptr[i], self.cell_ptr[i + 1])
            e = self.corner_edge[k]
            sg = self.corner_sign[k]
            n = self.edge_normal[e] * sg[:, None]
            N = self.edge_length[e][:, None] * n
            R = self.edge_midpoint[e] - self.cell_centroid[i]
            A = self.cell_area[i]
            M = R @ R.T / A
            P = np.eye(len(e)) - N @ np.linalg.solve(N.T @ N, N.T)
            gam = gamma_scale * np.trace(M) / len(e)
            out.append(M + gam * P)
        return out

    # ------------------------------------------------------------ point queries
    def locate(self, points):
        """Cell index containing each point (-1 if outside)."""
        points = np.atleast_2d(points)
        tree = cKDTree(self.cell_centroid)
        k = min(12, self.n_cells)
        _, cand = tree.query(points, k=k)
        cand = np.atleast_2d(cand)
        out = np.full(len(points), -1)
        for i, p in enumerate(points):
            for c in cand[i]:
                if self._inside(c, p):
                    out[i] = c
                    break
            else:
                d = np.linalg.norm(self.cell_centroid - p, axis=1)
                for c in np.argsort(d)[:200]:
                    if self._inside(c, p):
                        out[i] = c
                        break
        return out

    def _inside(self, c, p, tol=1e-12):
        P = self.nodes[self.cell(c)]
        Q = np.roll(P, -1, 0)
        cr = (Q[:, 0] - P[:, 0]) * (p[1] - P[:, 1]) - (Q[:, 1] - P[:, 1]) * (p[0] - P[:, 0])
        scale = np.max(np.abs(Q - P)) ** 2
        return np.all(cr >= -tol * scale)

    def interpolation_matrix(self, points):
        """
        Sparse (n_points, n_nodes) mean-value interpolation of node values.

        Mean-value coordinates (Floater 2003) reproduce linear functions on
        any polygon, so the interpolation is second-order accurate.
        """
        points = np.atleast_2d(points)
        cells = self.locate(points)
        if np.any(cells < 0):
            raise ValueError("point outside the mesh")
        rows, cols, vals = [], [], []
        for i, (p, c) in enumerate(zip(points, cells)):
            idx = self.cell(c)
            w = _mean_value_weights(self.nodes[idx], p)
            rows += [i] * len(idx)
            cols += list(idx)
            vals += list(w)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(points), self.n_nodes))

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self):
        """Dictionary of mesh statistics (counts, edge lengths, orthogonality)."""
        H = self.dual_edge_partial()
        h = np.asarray(H.sum(1)).ravel()
        return dict(
            n_nodes=self.n_nodes,
            n_edges=self.n_edges,
            n_cells=self.n_cells,
            cells_by_size=np.bincount(self.cell_size).tolist(),
            min_edge=float(self.edge_length.min()),
            mean_edge=float(self.edge_length.mean()),
            min_dual_edge=float(h.min()),
            max_orthogonality_defect=float(self.orthogonality_defect.max()),
            min_node_area=float(np.asarray(self.dual_area_partial().sum(1)).min()),
            area=float(self.cell_area.sum()),
        )

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_simplex(cls, mesh):
        """PolyMesh of a 2D `SimplexMesh` (circumcentric dual)."""
        return cls(mesh.nodes, list(mesh.cells), mesh.cell_tags)

    @classmethod
    def rectilinear(cls, x, y, cell_tags=None):
        """Tensor-product grid of rectangles on node coordinates x, y [m]."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        nx, ny = len(x), len(y)
        X, Y = np.meshgrid(x, y, indexing="xy")
        nodes = np.stack([X.ravel(), Y.ravel()], 1)
        i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="xy")
        i, j = i.ravel(), j.ravel()
        n0 = i + nx * j
        cells = np.stack([n0, n0 + 1, n0 + 1 + nx, n0 + nx], 1)
        return cls(nodes, list(cells), cell_tags)


def _mean_value_weights(P, x):
    """Mean-value coordinates of point x in polygon P (counter-clockwise)."""
    d = P - x
    r = np.linalg.norm(d, axis=1)
    n = len(P)
    hit = np.flatnonzero(r < 1e-14 * max(r.max(), 1e-300))
    w = np.zeros(n)
    if hit.size:
        w[hit[0]] = 1.0
        return w
    dn = np.roll(d, -1, 0)
    rn = np.roll(r, -1)
    cross = d[:, 0] * dn[:, 1] - d[:, 1] * dn[:, 0]
    dot = np.einsum("ij,ij->i", d, dn)
    # on (or within round-off of) an edge: the angle is ~pi and the
    # half-angle tangent below would overflow
    on_edge = np.flatnonzero(r * rn + dot <= 1e-9 * r * rn)
    if on_edge.size:
        k = on_edge[0]
        k1 = (k + 1) % n
        w[k] = rn[k] / (r[k] + rn[k])
        w[k1] = r[k] / (r[k] + rn[k])
        return w
    # tan(alpha_k / 2) for the angle between d_k and d_{k+1}
    tan_half = cross / (r * rn + dot)
    w = (np.roll(tan_half, 1) + tan_half) / r
    return w / w.sum()
