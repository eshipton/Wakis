# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Octree of refined Cartesian boxes inside a (non-uniform) tensor grid.

The root block is the coarse `GridFIT3D` tensor grid. Each patch is a box of
whole cells of its parent block, every parent cell split into ``r`` equal
parts per axis (``r`` = 2 or 3). Parent cells inside a child box are
inactive. The edges of a child block that lie on an *interface* side of its
box are hanging: their voltages are the Nedelec (Whitney) trace of the
coarse face they lie in,

* collinear with a coarse edge:            ``e_f = e_c / r``;
* inside a coarse face, at fraction ``t`` between the two parallel coarse
  edges ``e_0`` and ``e_1`` of that face:   ``e_f = ((1 - t) e_0 + t e_1) / r``.

These rows form the prolongation ``P_e`` (all edges x free edges). Balance:
an interface side of a child is at least one parent cell away from every
interface side of the parent, so hanging edges only reference free edges.
"""

import numpy as np

from .block import CartesianBlock

__all__ = ["SubgridMesh3D", "Patch"]

SIDE_TYPES = ("pec", "pmc", "periodic", "interface")


class Patch:
    """Refined box: parent block index, parent-cell range [lo, hi), ratio."""

    def __init__(self, parent, lo, hi, ratio):
        self.parent = int(parent)
        self.lo = tuple(int(v) for v in lo)
        self.hi = tuple(int(v) for v in hi)
        self.ratio = int(ratio)
        if self.ratio not in (2, 3):
            raise ValueError("ratio must be 2 or 3")
        if any(h <= l for l, h in zip(self.lo, self.hi)):
            raise ValueError("empty patch")


def _axes_from(x, y, z):
    """Accept node arrays or a `GridFIT3D` (its ghost cells are dropped)."""
    if y is None and z is None and hasattr(x, "x"):
        g = x
        out = []
        for a, name in enumerate("xyz"):
            v = np.asarray(getattr(g, name), float)
            ghosts = getattr(g, "edge_ghosts", [0, 0, 0])
            if ghosts[a]:
                v = v[:-1]
            out.append(v)
        return out
    return [np.asarray(v, float) for v in (x, y, z)]


class SubgridMesh3D:
    """
    Coarse tensor grid plus nested refined boxes.

    Parameters
    ----------
    x, y, z : array_like or GridFIT3D
        Coarse node coordinates [m], or a `GridFIT3D` as ``x`` alone.
    bc : sequence of 3 entries
        Per axis ``(low, high)`` with ``'pec'`` or ``'pmc'`` each, or the
        string ``'periodic'``. Default: PEC everywhere.
    phase : sequence of 3 complex
        Bloch phase per periodic axis, ``e(x + L) = phase e(x)``.
    """

    def __init__(self, x, y=None, z=None, bc=None, phase=(1.0, 1.0, 1.0)):
        axes = _axes_from(x, y, z)
        if bc is None:
            bc = [("pec", "pec")] * 3
        sides, periodic = [], []
        for a in range(3):
            v = bc[a]
            if isinstance(v, str):
                if v != "periodic":
                    v = (v, v)
                else:
                    v = ("periodic", "periodic")
            v = tuple(s.lower() for s in v)
            for s in v:
                if s not in ("pec", "pmc", "periodic"):
                    raise ValueError(f"unknown boundary {s}")
            if ("periodic" in v) and v != ("periodic", "periodic"):
                raise ValueError("periodic must be set on both sides")
            sides.append(v)
            periodic.append(v[0] == "periodic")
        self.root = CartesianBlock(*axes, periodic=periodic, phase=phase)
        self.root_sides = sides
        self.patches = []
        self._built = False

    # ---------------------------------------------------------------- patches
    def add_patch(self, lo, hi, ratio=2, parent=0):
        """Refine parent cells [lo, hi) (parent-cell indices). Returns block id."""
        if self._built:
            raise RuntimeError("mesh already built")
        self.patches.append(Patch(parent, lo, hi, ratio))
        return len(self.patches)

    def add_box(self, bounds, ratio=2, parent=0):
        """
        Refine the parent cells inside ``bounds = ((x0, x1), (y0, y1), (z0, z1))``.

        The bounds must lie on node lines of the parent block (to 1e-9 of a cell).
        """
        par = self._block_axes(parent)
        lo, hi = [], []
        for a in range(3):
            v = par[a]
            tol = 1e-9 * np.min(np.diff(v))
            i0 = np.flatnonzero(np.abs(v - bounds[a][0]) < tol)
            i1 = np.flatnonzero(np.abs(v - bounds[a][1]) < tol)
            if len(i0) != 1 or len(i1) != 1:
                raise ValueError(f"box bound on axis {a} is not on a parent node line")
            lo.append(int(i0[0]))
            hi.append(int(i1[0]))
        return self.add_patch(lo, hi, ratio, parent)

    def _block_axes(self, b):
        if b == 0:
            return self.root.nodes
        p = self.patches[b - 1]
        par = self._block_axes(p.parent)
        return [_subdivide(par[a][p.lo[a] : p.hi[a] + 1], p.ratio) for a in range(3)]

    # ------------------------------------------------------------------ build
    def build(self):
        if self._built:
            return self
        self.blocks = [self.root]
        self.parent = [-1]
        self.level = [0]
        self.sides = [self.root_sides]
        for n, p in enumerate(self.patches):
            if not 0 <= p.parent <= n:
                raise ValueError("patch parent must be an earlier block")
            pb = self.blocks[p.parent]
            psides = self.sides[p.parent]
            per, ph, sd = [], [], []
            for a in range(3):
                if p.lo[a] < 0 or p.hi[a] > pb.nc[a]:
                    raise ValueError("patch outside its parent")
                span = p.lo[a] == 0 and p.hi[a] == pb.nc[a]
                if pb.periodic[a] and span:
                    per.append(True)
                    ph.append(pb.phase[a])
                    sd.append(("periodic", "periodic"))
                    continue
                per.append(False)
                ph.append(1.0)
                s = []
                for side, at_bound in ((0, p.lo[a] == 0), (1, p.hi[a] == pb.nc[a])):
                    if pb.periodic[a] and at_bound:
                        raise ValueError("a patch touching a periodic seam must span the axis")
                    if at_bound:
                        if psides[a][side] == "interface":
                            raise ValueError(
                                "unbalanced patch: an interface side of a child must be "
                                "at least one parent cell from the parent's interface"
                            )
                        s.append(psides[a][side])
                    else:
                        s.append("interface")
                sd.append(tuple(s))
            nodes = [_subdivide(pb.nodes[a][p.lo[a] : p.hi[a] + 1], p.ratio) for a in range(3)]
            self.blocks.append(CartesianBlock(*nodes, periodic=per, phase=ph))
            self.parent.append(p.parent)
            self.level.append(self.level[p.parent] + 1)
            self.sides.append(sd)
        self._check_siblings()
        self.n_blocks = len(self.blocks)
        self.edge_off = np.concatenate([[0], np.cumsum([b.n_edges for b in self.blocks])])
        self.face_off = np.concatenate([[0], np.cumsum([b.n_faces for b in self.blocks])])
        self.cell_off = np.concatenate([[0], np.cumsum([b.n_cells for b in self.blocks])])
        self.n_edges = int(self.edge_off[-1])
        self.n_faces = int(self.face_off[-1])
        self.n_cells = int(self.cell_off[-1])
        self._active_cells()
        self._built = True
        return self

    def _check_siblings(self):
        for i, p in enumerate(self.patches):
            for q in self.patches[i + 1 :]:
                if p.parent != q.parent:
                    continue
                gap = [max(q.lo[a] - p.hi[a], p.lo[a] - q.hi[a]) for a in range(3)]
                if max(gap) < 1:
                    raise ValueError("sibling patches must be at least one parent cell apart")

    def _active_cells(self):
        self.active = []
        for b in self.blocks:
            self.active.append(np.ones(b.nc, bool))
        for n, p in enumerate(self.patches):
            sl = tuple(slice(p.lo[a], p.hi[a]) for a in range(3))
            self.active[p.parent][sl] = False

    # ----------------------------------------------------- edge classification
    def edge_sides(self, bi):
        """
        Per-edge flags of block `bi`: (on_pec_side, on_pmc_side, on_interface_side).
        """
        blk = self.blocks[bi]
        sd = self.sides[bi]
        flags = {t: [] for t in ("pec", "pmc", "interface")}
        for a in range(3):
            ijk = blk.edge_grid(a)
            f = {t: np.zeros(blk.edge_shape[a], bool) for t in flags}
            for ax in range(3):
                if ax == a or blk.periodic[ax]:
                    continue
                for side, at in ((0, ijk[ax] == 0), (1, ijk[ax] == blk.nc[ax])):
                    t = sd[ax][side]
                    f[t] |= at
            for t in flags:
                flags[t].append(f[t].ravel())
        return tuple(np.concatenate(flags[t]) for t in ("pec", "pmc", "interface"))

    def hanging_rows(self, mutate=None):
        """
        Trace constraints of all hanging edges.

        Returns
        -------
        rows, cols : int arrays (global all-edge indices: hanging, parent)
        vals : weights (complex across Bloch seams)
        hanging : bool mask over all edges
        """
        self.build()
        rows, cols, vals = [], [], []
        hanging = np.zeros(self.n_edges, bool)
        for bi in range(1, self.n_blocks):
            blk = self.blocks[bi]
            p = self.patches[bi - 1]
            pb = self.blocks[p.parent]
            r = p.ratio
            pec, _, itf = self.edge_sides(bi)
            hang = itf & ~pec
            hanging[self.edge_off[bi] : self.edge_off[bi + 1]] = hang
            for a in range(3):
                e0, e1 = blk.edge_off[a], blk.edge_off[a + 1]
                loc = np.flatnonzero(hang[e0:e1])
                if loc.size == 0:
                    continue
                ijk = np.unravel_index(loc, blk.edge_shape[a])
                q = [None] * 3
                rem = [None] * 3
                for ax in range(3):
                    if ax == a:
                        q[ax] = p.lo[ax] + ijk[ax] // r
                    else:
                        q[ax] = p.lo[ax] + ijk[ax] // r
                        rem[ax] = ijk[ax] % r
                b, c = [ax for ax in range(3) if ax != a]
                both = (rem[b] != 0) & (rem[c] != 0)
                if np.any(both):
                    raise RuntimeError("hanging edge not on a coarse face (internal error)")
                g = self.edge_off[bi] + e0 + loc
                # term on the lower parallel coarse edge
                t = np.where(rem[b] != 0, rem[b] / r, np.where(rem[c] != 0, rem[c] / r, 0.0))
                w0 = (1.0 - t) / r
                w1 = t / r
                if mutate == "fraction":
                    w0, w1 = 1.25 * w0, 1.25 * w1
                if mutate == "missing_transverse":
                    w1 = 0.0 * w1
                pidx, ph = pb.edge_index(a, *q)
                rows.append(g)
                cols.append(self.edge_off[p.parent] + pidx)
                vals.append(w0 * ph)
                inner = t > 0
                if np.any(inner):
                    q1 = [v.copy() for v in q]
                    sb = rem[b] != 0
                    q1[b] = np.where(sb, q1[b] + 1, q1[b])
                    q1[c] = np.where(~sb, q1[c] + 1, q1[c])
                    pidx1, ph1 = pb.edge_index(a, *[v[inner] for v in q1])
                    rows.append(g[inner])
                    cols.append(self.edge_off[p.parent] + pidx1)
                    vals.append((w1 * np.ones_like(t))[inner] * ph1)
        if rows:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            vals = np.concatenate([np.asarray(v, complex) for v in vals])
        else:
            rows = cols = np.zeros(0, int)
            vals = np.zeros(0, complex)
        return rows, cols, vals, hanging

    # ---------------------------------------------------------------- lookups
    def finest_block(self, point):
        """Index of the deepest block whose active cells contain `point`."""
        self.build()
        best = 0
        for bi in range(self.n_blocks):
            if self.blocks[bi].contains(point) and self.level[bi] >= self.level[best]:
                best = bi
        return best


def _subdivide(v, r):
    v = np.asarray(v, float)
    out = [v[0]]
    for a, b in zip(v[:-1], v[1:]):
        out.extend(a + (b - a) * np.arange(1, r) / r)
        out.append(b)
    return np.array(out)
