# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
One dense Cartesian FIT block: topology and per-cell metric contributions.

A block is a tensor grid with node lines ``xn, yn, zn`` (non-uniform allowed).
Along a periodic axis the node ``n`` is identified with node ``0`` times a
Bloch phase ``phase[a]`` (field at ``x + L_a`` = ``phase[a]`` x field at
``x``), so that axis has ``n`` distinct node lines instead of ``n + 1``.

Indexing (C order inside each component, components concatenated):

* edges of direction ``a``: cells along ``a``, node lines along the other two;
* faces of normal ``a``: node lines along ``a``, cells along the other two;
* cells: ``(nx, ny, nz)``.

The face circulation of normal ``a`` with ``(b, c)`` the cyclic successors is
``e_b(p) + e_c(p + u_b) - e_b(p + u_c) - e_c(p)``: the topological curl of FIT
(the same incidence as `wakis.boundaries._assemble_curl`, compact layout
without the ghost samples of `SolverFIT3D`).
"""

import numpy as np
import scipy.sparse as sp

__all__ = ["CartesianBlock"]


class CartesianBlock:
    """
    Dense Cartesian FIT block.

    Parameters
    ----------
    xn, yn, zn : array_like
        Node coordinates [m], strictly increasing.
    periodic : tuple of bool
        Periodic (Bloch) axes.
    phase : tuple of complex
        Bloch phase per axis (ignored on non-periodic axes).
    """

    def __init__(self, xn, yn, zn, periodic=(False, False, False), phase=(1.0, 1.0, 1.0)):
        self.nodes = [np.asarray(v, float) for v in (xn, yn, zn)]
        for v in self.nodes:
            if v.ndim != 1 or len(v) < 2 or np.any(np.diff(v) <= 0):
                raise ValueError("node arrays must be 1D and strictly increasing")
        self.periodic = tuple(bool(p) for p in periodic)
        self.phase = tuple(complex(p) if per else 1.0 for p, per in zip(phase, self.periodic))
        self.h = [np.diff(v) for v in self.nodes]
        self.nc = tuple(len(h) for h in self.h)
        self.nn = tuple(n if per else n + 1 for n, per in zip(self.nc, self.periodic))
        self.complex = any(p != 1.0 for p in self.phase)

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
        self.n_cells = int(np.prod(self.nc))

    # ------------------------------------------------------------------ index
    def _wrap(self, axis, idx):
        """Wrap a node index along `axis`; return (index, phase factor)."""
        idx = np.asarray(idx)
        if not self.periodic[axis]:
            return idx, np.ones(idx.shape, complex if self.complex else float)
        n = self.nn[axis]
        q = np.floor_divide(idx, n)
        ph = np.power(self.phase[axis], q.astype(float)) if self.complex else np.ones(idx.shape)
        return idx - q * n, ph

    def edge_index(self, a, i, j, k):
        """Flat edge index of direction `a` (node indices wrapped, with phase)."""
        ijk = [np.asarray(i), np.asarray(j), np.asarray(k)]
        ph = 1.0
        for ax in range(3):
            if ax != a:
                ijk[ax], p = self._wrap(ax, ijk[ax])
                ph = ph * p
        idx = self.edge_off[a] + np.ravel_multi_index(tuple(ijk), self.edge_shape[a])
        return idx, ph

    def face_index(self, a, i, j, k):
        """Flat face index of normal `a` (node index along `a` wrapped)."""
        ijk = [np.asarray(i), np.asarray(j), np.asarray(k)]
        ijk[a], ph = self._wrap(a, ijk[a])
        idx = self.face_off[a] + np.ravel_multi_index(tuple(ijk), self.face_shape[a])
        return idx, ph

    def edge_grid(self, a):
        """Integer index arrays (i, j, k) of every edge of direction `a`."""
        return np.meshgrid(*[np.arange(s) for s in self.edge_shape[a]], indexing="ij")

    def face_grid(self, a):
        return np.meshgrid(*[np.arange(s) for s in self.face_shape[a]], indexing="ij")

    def cell_grid(self):
        return np.meshgrid(*[np.arange(s) for s in self.nc], indexing="ij")

    # --------------------------------------------------------------- geometry
    def edge_geometry(self):
        """Edge midpoints (n_edges, 3) [m], lengths [m] and directions."""
        mids, lens, dirs = [], [], []
        for a in range(3):
            ijk = self.edge_grid(a)
            p = []
            for ax in range(3):
                v = self.nodes[ax]
                if ax == a:
                    p.append(0.5 * (v[ijk[ax]] + v[ijk[ax] + 1]))
                else:
                    p.append(v[ijk[ax]])
            mids.append(np.stack([q.ravel() for q in p], 1))
            lens.append(self.h[a][ijk[a]].ravel())
            dirs.append(np.full(ijk[0].size, a))
        return np.concatenate(mids), np.concatenate(lens), np.concatenate(dirs)

    def face_geometry(self):
        """Face centres (n_faces, 3) [m], areas [m^2] and normals."""
        cens, areas, norms = [], [], []
        for a in range(3):
            ijk = self.face_grid(a)
            p, area = [], 1.0
            for ax in range(3):
                v = self.nodes[ax]
                if ax == a:
                    p.append(v[ijk[ax]])
                else:
                    p.append(0.5 * (v[ijk[ax]] + v[ijk[ax] + 1]))
                    area = area * self.h[ax][ijk[ax]]
            cens.append(np.stack([q.ravel() for q in p], 1))
            areas.append(np.asarray(area).ravel())
            norms.append(np.full(ijk[0].size, a))
        return np.concatenate(cens), np.concatenate(areas), np.concatenate(norms)

    def cell_centres(self):
        ijk = self.cell_grid()
        return [0.5 * (self.nodes[a][ijk[a]] + self.nodes[a][ijk[a] + 1]) for a in range(3)]

    # -------------------------------------------------------------- operators
    def curl(self):
        """Face-edge incidence C (n_faces x n_edges), FIT orientation."""
        rows, cols, vals = [], [], []
        dt = complex if self.complex else float
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            ijk = self.face_grid(a)
            f, _ = self.face_index(a, *ijk)
            for comp, shift, sign in ((b, None, 1.0), (c, b, 1.0), (b, c, -1.0), (c, None, -1.0)):
                q = [x.copy() for x in ijk]
                if shift is not None:
                    q[shift] = q[shift] + 1
                e, ph = self.edge_index(comp, *q)
                rows.append(f.ravel())
                cols.append(e.ravel())
                vals.append((sign * np.asarray(ph, dt)).ravel())
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_faces, self.n_edges),
        )

    def divergence(self):
        """Cell-face incidence (n_cells x n_faces): outward flux sum."""
        rows, cols, vals = [], [], []
        dt = complex if self.complex else float
        ijk = self.cell_grid()
        cell = np.ravel_multi_index(tuple(ijk), self.nc).ravel()
        for a in range(3):
            for d, sign in ((0, -1.0), (1, 1.0)):
                q = [x.copy() for x in ijk]
                q[a] = q[a] + d
                f, ph = self.face_index(a, *q)
                rows.append(cell)
                cols.append(f.ravel())
                vals.append((sign * np.asarray(ph, dt)).ravel())
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_cells, self.n_faces),
        )

    def cell_edges(self):
        """
        Edges of every cell with their lumped-mass geometric weight.

        Returns
        -------
        cell : (12 n_cells,) int
        edge : (12 n_cells,) int
        w : (12 n_cells,) float
            Quarter of the cell cross-section normal to the edge divided by
            the edge length [m]: the cell's share of ``A_dual / L``.
        """
        ijk = self.cell_grid()
        cell = np.ravel_multi_index(tuple(ijk), self.nc).ravel()
        hc = [self.h[a][ijk[a]].ravel() for a in range(3)]
        cs, es, ws = [], [], []
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            w = hc[b] * hc[c] / 4.0 / hc[a]
            for db in (0, 1):
                for dc in (0, 1):
                    q = [x.copy() for x in ijk]
                    q[b] = q[b] + db
                    q[c] = q[c] + dc
                    e, _ = self.edge_index(a, *q)
                    cs.append(cell)
                    es.append(e.ravel())
                    ws.append(w)
        return np.concatenate(cs), np.concatenate(es), np.concatenate(ws)

    def cell_faces(self):
        """
        Faces of every cell with the cell's share of ``L_dual / A`` [1/m].
        """
        ijk = self.cell_grid()
        cell = np.ravel_multi_index(tuple(ijk), self.nc).ravel()
        hc = [self.h[a][ijk[a]].ravel() for a in range(3)]
        cs, fs, ws = [], [], []
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            w = (hc[a] / 2.0) / (hc[b] * hc[c])
            for d in (0, 1):
                q = [x.copy() for x in ijk]
                q[a] = q[a] + d
                f, _ = self.face_index(a, *q)
                cs.append(cell)
                fs.append(f.ravel())
                ws.append(w)
        return np.concatenate(cs), np.concatenate(fs), np.concatenate(ws)

    def contains(self, p):
        return all(self.nodes[a][0] - 1e-15 <= p[a] <= self.nodes[a][-1] + 1e-15 for a in range(3))
