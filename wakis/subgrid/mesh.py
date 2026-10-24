# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Nested Cartesian (quadtree) meshes with hanging nodes for 2D subgridding.

A coarse rectilinear grid (node coordinates ``x``, ``y``, possibly graded) is
refined cell by cell with an integer ratio ``r`` (2 or 3, ...), recursively
up to ``max_level`` levels. Every leaf cell is an axis-aligned rectangle; all
cell corners live on the *lattice* obtained by splitting each coarse interval
into ``R = r**max_level`` equal parts, so the topology is exact integer
arithmetic.

Where a coarse leaf borders finer leaves, the corners of the fine leaves that
lie strictly inside a side of the coarse leaf are *hanging nodes*. They are not
unknowns: their value is the linear interpolant of the two end nodes of that
side (the trace of the bilinear Q1 function of the coarse cell), which makes
the scalar field (Ez in TM, Hz in TE) continuous. This is collected in the
prolongation matrix ``P`` (all nodes x free nodes, rows sum to 1), so that
``u_all = P u_free``.

Edges are the sides of the leaf cells, merged when two leaves share the same
side. Along a coarse/fine interface the coarse side and the ``r`` collinear
fine sides are *different* edges (each belongs to its own cell), exactly as
in a conforming finite-element assembly.
"""

import numpy as np
import scipy.sparse as sp

__all__ = ["QuadMesh", "refine_near"]


class QuadMesh:
    """
    Composite Cartesian mesh: coarse grid plus nested refined patches.

    Parameters
    ----------
    x, y : array_like
        Coarse node coordinates [m], strictly increasing (may be graded).
    ratio : int
        Refinement ratio per level (2 or 3 recommended).
    max_level : int
        Number of refinement levels (0: plain coarse grid).
    refine : callable, optional
        ``refine(level, x0, x1, y0, y1) -> bool array``: which cells of
        `level` (arrays of cell bounds [m]) are split into ``ratio**2``
        children. Ignored when ``max_level == 0``.
    balance : bool
        Enforce the 2:1-type balance (a leaf side never touches leaves more
        than one level finer) by refining offending cells.
    periodic_y : bool
        Identify y = y[0] with y = y[-1] (Bloch phase via `bloch_phase` of
        the solver / `gradient`).

    Attributes
    ----------
    cells : dict of ndarray
        ``i, j`` lattice corner, ``n`` lattice size, ``level``; ``x0, x1, y0,
        y1`` bounds [m].
    nodes : ndarray (n_nodes, 2)
        Node coordinates [m].
    hanging : ndarray of bool (n_nodes,)
    P : csr_matrix (n_nodes, n_free)
        Prolongation from free nodes to all nodes.
    free : ndarray of int
        Indices of the free (non-hanging) nodes.
    edges : ndarray (n_edges, 2) of int
        Node indices (a, b), a at the lower coordinate.
    edge_dir : ndarray (n_edges,) of int
        0: x-directed, 1: y-directed.
    edge_wrap : ndarray (n_edges,) of bool
        Node b reached through the periodic y boundary.
    cell_edges : ndarray (n_cells, 4) of int
        Edges of each cell: bottom, top, left, right.
    cell_nodes : ndarray (n_cells, 4) of int
        Corner nodes: (x0,y0), (x1,y0), (x0,y1), (x1,y1).
    """

    def __init__(self, x, y, ratio=2, max_level=1, refine=None, balance=True,
                 periodic_y=False):  # fmt: skip
        self.xc = np.asarray(x, float)
        self.yc = np.asarray(y, float)
        self.ratio = int(ratio)
        self.max_level = int(max_level) if refine is not None else 0
        self.periodic_y = bool(periodic_y)
        R = self.ratio**self.max_level
        self.R = R
        self.xl = _lattice(self.xc, R)
        self.yl = _lattice(self.yc, R)
        self.NI = len(self.xl) - 1
        self.NJ = len(self.yl) - 1
        self._build_cells(refine, balance)
        self._build_nodes()
        self._build_edges()

    # ------------------------------------------------------------------ cells
    def _bounds(self, i, j, n):
        return self.xl[i], self.xl[i + n], self.yl[j], self.yl[j + n]

    def _build_cells(self, refine, balance):
        R, r = self.R, self.ratio
        nx, ny = len(self.xc) - 1, len(self.yc) - 1
        ii, jj = np.meshgrid(np.arange(nx) * R, np.arange(ny) * R, indexing="ij")
        act_i, act_j = ii.ravel(), jj.ravel()
        leaves = []
        n = R
        for lev in range(self.max_level):
            flag = np.asarray(refine(lev, *self._bounds(act_i, act_j, n)), bool)
            flag = np.broadcast_to(flag, act_i.shape)
            leaves.append((act_i[~flag], act_j[~flag], np.full((~flag).sum(), n),
                           np.full((~flag).sum(), lev)))  # fmt: skip
            ci, cj = act_i[flag], act_j[flag]
            n //= r
            di, dj = np.meshgrid(np.arange(r) * n, np.arange(r) * n, indexing="ij")
            act_i = (ci[:, None] + di.ravel()[None]).ravel()
            act_j = (cj[:, None] + dj.ravel()[None]).ravel()
        leaves.append((act_i, act_j, np.full(act_i.size, n),
                       np.full(act_i.size, self.max_level)))  # fmt: skip
        ci, cj, cn, cl = (np.concatenate(a) for a in zip(*leaves))
        if balance and self.max_level > 1:
            ci, cj, cn, cl = self._balance(ci, cj, cn, cl)
        order = np.lexsort((ci, cj))
        ci, cj, cn, cl = ci[order], cj[order], cn[order], cl[order]
        x0, x1, y0, y1 = self._bounds(ci, cj, cn)
        self.cells = dict(i=ci, j=cj, n=cn, level=cl, x0=x0, x1=x1, y0=y0, y1=y1)
        self.n_cells = ci.size

    def _corner_keys(self, ci, cj, cn):
        NJ = self.NJ + 1
        return np.stack([ci * NJ + cj, (ci + cn) * NJ + cj, ci * NJ + cj + cn,
                         (ci + cn) * NJ + cj + cn], 1)  # fmt: skip

    def _balance(self, ci, cj, cn, cl):
        """Split leaves whose sides contain corners two or more levels finer."""
        r = self.ratio
        for _ in range(4 * self.max_level + 4):
            keys = np.unique(self._corner_keys(ci, cj, cn))
            bad = np.zeros(ci.size, bool)
            big = np.flatnonzero(cn >= r * r)
            for side in range(4):
                for k in range(1, r * r):
                    if k % r == 0:
                        continue
                    off = cn[big] * k // (r * r)
                    if side == 0:
                        pi, pj = ci[big] + off, cj[big]
                    elif side == 1:
                        pi, pj = ci[big] + off, cj[big] + cn[big]
                    elif side == 2:
                        pi, pj = ci[big], cj[big] + off
                    else:
                        pi, pj = ci[big] + cn[big], cj[big] + off
                    bad[big[np.isin(pi * (self.NJ + 1) + pj, keys)]] = True
            if not bad.any():
                return ci, cj, cn, cl
            n_ch = cn[bad] // r
            di, dj = np.meshgrid(np.arange(r), np.arange(r), indexing="ij")
            new_i = (ci[bad][:, None] + n_ch[:, None] * di.ravel()).ravel()
            new_j = (cj[bad][:, None] + n_ch[:, None] * dj.ravel()).ravel()
            new_n = np.repeat(n_ch, r * r)
            new_l = np.repeat(cl[bad] + 1, r * r)
            ci = np.concatenate([ci[~bad], new_i])
            cj = np.concatenate([cj[~bad], new_j])
            cn = np.concatenate([cn[~bad], new_n])
            cl = np.concatenate([cl[~bad], new_l])
        raise RuntimeError("balancing did not converge")

    # ------------------------------------------------------------------ nodes
    def _wrap_j(self, j):
        return j % self.NJ if self.periodic_y else j

    def _build_nodes(self):
        c = self.cells
        NJ1 = self.NJ + 1
        ci, cj, cn = c["i"], c["j"], c["n"]
        corners_i = np.stack([ci, ci + cn, ci, ci + cn], 1)
        corners_j = self._wrap_j(np.stack([cj, cj, cj + cn, cj + cn], 1))
        keys = corners_i * NJ1 + corners_j
        ukeys, inv = np.unique(keys.ravel(), return_inverse=True)
        self.node_keys = ukeys
        self.cell_nodes = inv.reshape(-1, 4)
        self.node_i, self.node_j = ukeys // NJ1, ukeys % NJ1
        self.nodes = np.stack([self.xl[self.node_i], self.yl[self.node_j]], 1)
        n_nodes = ukeys.size
        self.n_nodes = n_nodes

        # hanging nodes: lattice points strictly inside a leaf side that are
        # corners of other leaves; constraint = linear interpolation
        rows, cols, vals, wraps = [], [], [], []
        hang = np.zeros(n_nodes, bool)
        big = np.flatnonzero(cn > 1)
        for side in range(4 if big.size else 0):
            # side endpoints (lattice) and direction
            if side in (0, 1):  # bottom/top, x-directed
                jj = cj[big] + (cn[big] if side == 1 else 0)
                ia, ja = ci[big], jj
                ib, jb = ci[big] + cn[big], jj
                di, dj = 1, 0
            else:  # left/right, y-directed
                ii = ci[big] + (cn[big] if side == 3 else 0)
                ia, ja = ii, cj[big]
                ib, jb = ii, cj[big] + cn[big]
                di, dj = 0, 1
            ka = self._key(ia, ja)
            kb = self._key(ib, jb)
            nmax = cn[big].max()
            for k in range(1, nmax):
                ok = k < cn[big]
                if not ok.any():
                    break
                pk = self._key(ia[ok] + di * k, ja[ok] + dj * k)
                pos = np.searchsorted(ukeys, pk)
                pos = np.minimum(pos, n_nodes - 1)
                hit = ukeys[pos] == pk
                if not hit.any():
                    continue
                h = pos[hit]
                t = k / cn[big][ok][hit]
                a = np.searchsorted(ukeys, ka[ok][hit])
                b = np.searchsorted(ukeys, kb[ok][hit])
                wb = np.broadcast_to(
                    self.periodic_y & (jb[ok][hit] == self.NJ), h.shape
                )
                rows.append(h)
                cols.append(np.stack([a, b], 1))
                vals.append(t)
                wraps.append(wb)
                hang[h] = True
        self.hanging = hang
        free = np.flatnonzero(~hang)
        self.free = free
        if rows:
            h = np.concatenate(rows)
            ab = np.concatenate(cols)
            t = np.concatenate(vals)
            wb = np.concatenate(wraps)
            # a hanging node lies inside exactly one leaf side: keep one copy
            _, first = np.unique(h, return_index=True)
            self._constraints = (h[first], ab[first], t[first], wb[first])
        else:
            e = np.array([], int)
            self._constraints = (e, np.zeros((0, 2), int), np.array([]), e.astype(bool))
        self.P = self.prolongation()

    def prolongation(self, bloch_phase=1.0):
        """
        P (n_nodes x n_free): hanging nodes interpolate their side's ends.

        With `periodic_y`, a master reached through the periodic boundary
        carries `bloch_phase`. Chains (masters that hang themselves, in
        unbalanced meshes) are resolved recursively.
        """
        h, ab, t, wb = self._constraints
        n = self.n_nodes
        hang = self.hanging
        vb = t * np.where(wb, bloch_phase, 1.0)
        rows = np.concatenate([h, h])
        cols = np.concatenate([ab[:, 0], ab[:, 1]])
        vals = np.concatenate([(1 - t).astype(vb.dtype), vb])
        C = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        C = (C + sp.diags((~hang).astype(float))).tocsr()
        for _ in range(4 * self.max_level + 2):
            used = np.unique(C.indices[np.abs(C.data) > 0])
            if not hang[used].any():
                break
            C = (C @ C).tocsr()
        C.eliminate_zeros()
        return C[:, self.free].tocsr()

    def _key(self, i, j):
        return i * (self.NJ + 1) + self._wrap_j(j)

    # ------------------------------------------------------------------ edges
    def _build_edges(self):
        cn = self.cell_nodes
        c = self.cells
        # sides: bottom (0,1), top (2,3), left (0,2), right (1,3)
        a = np.stack([cn[:, 0], cn[:, 2], cn[:, 0], cn[:, 1]], 1)
        b = np.stack([cn[:, 1], cn[:, 3], cn[:, 2], cn[:, 3]], 1)
        d = np.broadcast_to(np.array([0, 0, 1, 1]), a.shape)
        top_j = np.stack([c["j"], c["j"] + c["n"], c["j"] + c["n"], c["j"] + c["n"]], 1)
        wrap = (d == 1) & (top_j == self.NJ) & self.periodic_y
        keys = a.astype(np.int64) * self.n_nodes + b
        ukeys, first, inv = np.unique(keys.ravel(), return_index=True,
                                      return_inverse=True)  # fmt: skip
        self.edges = np.stack([ukeys // self.n_nodes, ukeys % self.n_nodes], 1)
        self.edge_dir = d.ravel()[first]
        self.edge_wrap = wrap.ravel()[first]
        self.cell_edges = inv.reshape(-1, 4)
        self.n_edges = ukeys.size
        # edge length from any cell side owning it
        side_len = np.stack([c["x1"] - c["x0"]] * 2 + [c["y1"] - c["y0"]] * 2, 1)
        self.edge_length = side_len.ravel()[first]
        pa = self.nodes[self.edges[:, 0]]
        self.edge_mid = pa + 0.5 * self.edge_length[:, None] * np.stack(
            [self.edge_dir == 0, self.edge_dir == 1], 1
        )

    # --------------------------------------------------------------- operators
    def gradient(self, bloch_phase=1.0):
        """
        Signed incidence G (n_edges x n_nodes): (G u)_e = u_b - u_a.

        With `periodic_y`, node b of wrapping edges carries `bloch_phase`
        (u(y + L_y) = phase u(y)).
        """
        ne = self.n_edges
        rows = np.repeat(np.arange(ne), 2)
        cols = self.edges.ravel()
        vb = np.where(self.edge_wrap, bloch_phase, 1.0)
        vals = np.stack([-np.ones(ne), vb], 1).ravel()
        dtype = complex if np.iscomplexobj(vals) else float
        return sp.csr_matrix((vals.astype(dtype), (rows, cols)),
                             shape=(ne, self.n_nodes))  # fmt: skip

    def locate(self, points):
        """Index of the leaf cell containing each point (last one on ties)."""
        points = np.atleast_2d(points)
        c = self.cells
        Ii = np.clip(np.searchsorted(self.xl, points[:, 0], side="right") - 1, 0,
                    self.NI - 1)  # fmt: skip
        J = np.clip(np.searchsorted(self.yl, points[:, 1], side="right") - 1, 0,
                    self.NJ - 1)  # fmt: skip
        key_cell = c["i"].astype(np.int64) * (self.NJ + 1) + c["j"]
        order = np.argsort(key_cell)
        ks = key_cell[order]
        out = np.full(len(points), -1)
        for n in np.unique(c["n"]):
            i0 = (Ii // n) * n
            j0 = (J // n) * n
            kk = i0.astype(np.int64) * (self.NJ + 1) + j0
            pos = np.minimum(np.searchsorted(ks, kk), ks.size - 1)
            hit = (ks[pos] == kk) & (c["n"][order[pos]] == n)
            out[hit] = order[pos[hit]]
        if (out < 0).any():
            raise ValueError("point outside the mesh")
        return out

    def interpolation_matrix(self, points, bloch_phase=1.0):
        """
        Sparse bilinear interpolation (n_points x n_free) of the nodal field.

        Inside each leaf cell the field is the Q1 interpolant of its four
        corners; hanging corners are expanded with `P`.
        """
        points = np.atleast_2d(points)
        k = self.locate(points)
        c = self.cells
        tx = (points[:, 0] - c["x0"][k]) / (c["x1"][k] - c["x0"][k])
        ty = (points[:, 1] - c["y0"][k]) / (c["y1"][k] - c["y0"][k])
        top_wrap = self.periodic_y & ((c["j"][k] + c["n"][k]) == self.NJ)
        bp = np.where(top_wrap, bloch_phase, 1.0)
        w = np.stack(
            [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty * bp, tx * ty * bp], 1
        )
        rows = np.repeat(np.arange(len(points)), 4)
        dtype = complex if np.iscomplexobj(w) or np.iscomplexobj(bloch_phase) else float
        S = sp.csr_matrix(
            (w.ravel().astype(dtype), (rows, self.cell_nodes[k].ravel())),
            shape=(len(points), self.n_nodes),
        )
        P = self.P if bloch_phase == 1.0 else self.prolongation(bloch_phase)
        return (S @ P).tocsr()

    # -------------------------------------------------------------- summaries
    def level_counts(self):
        """Number of leaf cells per level."""
        return np.bincount(self.cells["level"], minlength=self.max_level + 1)

    def is_balanced(self):
        """True if every hanging-node weight is a multiple of 1/ratio."""
        P = self.P.tocoo()
        frac = P.data[(P.data > 1e-12) & (P.data < 1 - 1e-12)]
        r = self.ratio
        return bool(np.allclose(frac * r, np.round(frac * r), atol=1e-9))


def _lattice(xc, R):
    """Split every coarse interval into R equal parts."""
    t = np.arange(R) / R
    xs = (xc[:-1, None] + np.diff(xc)[:, None] * t[None]).ravel()
    return np.concatenate([xs, xc[-1:]])


def refine_near(levels):
    """
    Refinement predicate from distance functions.

    Parameters
    ----------
    levels : list of callable
        ``levels[l](xm, ym, half_diag) -> bool``: refine a cell of level l
        (centre xm, ym; half diagonal [m]) if True.
    """

    def refine(level, x0, x1, y0, y1):
        if level >= len(levels):
            return np.zeros(np.shape(x0), bool)
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        hd = 0.5 * np.hypot(x1 - x0, y1 - y0)
        return levels[level](xm, ym, hd)

    return refine
