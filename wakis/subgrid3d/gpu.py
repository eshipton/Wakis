# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Block kernels for `SubgridFIT3D`: dense per-block curls + sparse trace.

``K e = C P_e e`` is evaluated as

1. trace: ``e_all[free] = e``, ``e_all[hanging] = P_h e`` (sparse rows of the
   hanging edges only, <= 2 nonzeros each);
2. per block, the FIT curl on the dense edge arrays by array slicing
   (four shifted views per face component), on all faces of the block;

and ``K^H y = P_e^H C^T y`` by the transposed slicing per block plus
``P_h^T`` scatter of the hanging part. The state then keeps ``b`` on *all*
faces with the face Hodge ``W_all`` (zero on faces of inactive cells and
PMC-only faces), so no gather is needed.

The kernels are array-module generic: ``xp = numpy`` (CPU reference) or
``cupy`` (GPU). Same arithmetic as the CSR operators up to the order of the
additions: per row at most 4 curl terms and 2 trace terms (see the physics
guide, "GPU block kernels", for the round-off bound).

Not supported: Bloch-periodic blocks and complex fields (use the CSR path).
"""

import numpy as np
import scipy.sparse as sp

__all__ = ["BlockOps", "get_xp"]


def get_xp(backend):
    if backend in ("numpy", "cpu", None):
        return np, sp
    if backend in ("cupy", "gpu"):
        import cupy as cp
        import cupyx.scipy.sparse as csp

        return cp, csp
    raise ValueError(f"unknown backend {backend}")


def _sl(arr, axis, start, stop):
    idx = [slice(None)] * 3
    idx[axis] = slice(start, stop)
    return arr[tuple(idx)]


class BlockOps:
    """
    ``K`` and ``K^H`` of a `SubgridFIT3D` as block kernels.

    Parameters
    ----------
    solver : SubgridFIT3D
    backend : {'numpy', 'cupy'}
    """

    def __init__(self, solver, backend="numpy"):
        s = solver
        M = s.mesh
        if any(any(b.periodic) for b in M.blocks):
            raise NotImplementedError("block kernels: Bloch-periodic blocks use the CSR path")
        if s.dtype is not float and np.dtype(s.dtype).kind == "c":
            raise NotImplementedError("block kernels: real fields only")
        xp, xsp = get_xp(backend)
        self.xp, self.xsp = xp, xsp
        self.backend = backend
        self.nE, self.nF = M.n_edges, M.n_faces
        self.free = xp.asarray(s.free_edges)
        hang = np.flatnonzero(s.hanging)
        Ph = s.P.tocsr()[hang].real.tocsr()
        self.hang = xp.asarray(hang)
        self.Ph = xsp.csr_matrix(Ph)
        self.PhT = xsp.csr_matrix(Ph.T.tocsr())
        self.n_free = s.n_free
        self.faces = xp.asarray(s.faces)
        self.W_all = xp.asarray(np.real(s.W_all).astype(float))
        self.blocks = []
        for bi, blk in enumerate(M.blocks):
            e0, f0 = int(M.edge_off[bi]), int(M.face_off[bi])
            eo = [e0 + int(blk.edge_off[a]) for a in range(4)]
            fo = [f0 + int(blk.face_off[a]) for a in range(4)]
            self.blocks.append((eo, fo, blk.edge_shape, blk.face_shape))
        self._eall = xp.zeros(self.nE)
        self._gall = xp.zeros(self.nE)
        self.Dinv = xp.asarray(s.Dinv.real.astype(float))

    # ---------------------------------------------------------------- trace
    def prolong(self, e, out=None):
        """``P_e e`` on all edges."""
        out = self._eall if out is None else out
        out[...] = 0.0
        out[self.free] = e
        if self.hang.size:
            out[self.hang] = self.Ph @ e
        return out

    def restrict(self, g):
        """``P_e^T g`` (all edges -> free edges)."""
        r = g[self.free].copy()
        if self.hang.size:
            r += self.PhT @ g[self.hang]
        return r

    # ----------------------------------------------------------------- curl
    def curl_all(self, e_all, out):
        """``C e_all`` on all faces (block slicing)."""
        for eo, fo, es, fs in self.blocks:
            E = [e_all[eo[a] : eo[a + 1]].reshape(es[a]) for a in range(3)]
            for a in range(3):
                b, c = (a + 1) % 3, (a + 2) % 3
                F = out[fo[a] : fo[a + 1]].reshape(fs[a])
                F[...] = _sl(E[b], c, 0, -1) + _sl(E[c], b, 1, None)
                F -= _sl(E[b], c, 1, None) + _sl(E[c], b, 0, -1)
        return out

    def curl_T_all(self, y_all, out):
        """``C^T y_all`` on all edges (transposed slicing)."""
        out[...] = 0.0
        for eo, fo, es, fs in self.blocks:
            G = [out[eo[a] : eo[a + 1]].reshape(es[a]) for a in range(3)]
            for a in range(3):
                b, c = (a + 1) % 3, (a + 2) % 3
                Y = y_all[fo[a] : fo[a + 1]].reshape(fs[a])
                _sl(G[b], c, 0, -1)[...] += Y
                _sl(G[b], c, 1, None)[...] -= Y
                _sl(G[c], b, 1, None)[...] += Y
                _sl(G[c], b, 0, -1)[...] -= Y
        return out

    # -------------------------------------------------------- K and K^H
    def K(self, e, out):
        """``b_all <- C P_e e`` (all faces)."""
        return self.curl_all(self.prolong(e), out)

    def KH(self, y_all):
        """``P_e^T C^T y_all`` (free edges)."""
        return self.restrict(self.curl_T_all(y_all, self._gall))

    def to_faces(self, b_all):
        return b_all[self.faces]

    def from_faces(self, b):
        out = self.xp.zeros(self.nF)
        out[self.faces] = self.xp.asarray(b)
        return out
