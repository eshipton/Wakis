# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Leapfrog FIT on a 3D Cartesian octree with hanging edges (`SubgridFIT3D`).

Unknowns: free edge voltages ``e`` [V] and face fluxes ``b`` [Wb] of every
leaf cell face. With the prolongation ``P_e`` (all edges x free edges), the
block-diagonal face-edge incidence ``C`` of the blocks and ``K = C P_e``:

    b^{n+1/2} = b^{n-1/2} - dt K e^n
    D (e^{n+1} - e^n) = dt (K^H W b^{n+1/2} - P_e^H i^{n+1/2})

``W`` is the diagonal face Hodge (series sum of the half cells that own the
face, ``(h/2) / (mu A)``), ``m`` the diagonal edge masses of all edges (sum
of ``eps A_quarter / L`` over the active cells that own the edge) and
``D = |P_e|^T (m o |P_e| 1)`` its row-sum lumping onto the free edges.
``i`` [A] is an impressed current through the dual face of each edge.

Conserved (lossless, source-free):
``E = 1/2 b^{n+1/2} . W b^{n+1/2} + 1/2 e^n . D e^{n+1}``,
positive iff ``dt < 2 / sqrt(lambda_max(D^-1 K^H W K))``.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0

from .mesh import SubgridMesh3D

__all__ = ["SubgridFIT3D", "Box"]


class Box:
    """Axis-aligned box ``((x0, x1), (y0, y1), (z0, z1))`` [m] as a cell test."""

    def __init__(self, bounds, value=True):
        self.b = np.asarray(bounds, float)
        self.value = value

    def __call__(self, X, Y, Z):
        b = self.b
        return (
            (X > b[0, 0])
            & (X < b[0, 1])
            & (Y > b[1, 0])
            & (Y < b[1, 1])
            & (Z > b[2, 0])
            & (Z < b[2, 1])
        )


def _material(f, X, Y, Z, default):
    if f is None:
        return np.full(X.shape, default, float)
    if np.isscalar(f):
        return np.full(X.shape, float(f))
    return np.broadcast_to(np.asarray(f(X, Y, Z), float), X.shape)


def _mask(f, X, Y, Z):
    if f is None:
        return np.zeros(X.shape, bool)
    return np.broadcast_to(np.asarray(f(X, Y, Z), bool), X.shape)


class SubgridFIT3D:
    """
    3D FIT with nested Cartesian refinement patches (hanging edges).

    Parameters
    ----------
    mesh : SubgridMesh3D
    eps_r, mu_r : callable(X, Y, Z) or float, optional
        Relative permittivity / permeability sampled at cell centres
        (isotropic, diagonal Hodge; exact for grid-aligned interfaces).
    pec, pmc : callable(X, Y, Z) -> bool, optional
        Cells that are perfect electric conductors (eps -> inf: all 12 edges
        zero) or perfect magnetic conductors (mu -> inf: the cell adds
        nothing to the face Hodge, so H = 0 inside and the faces on its
        surface see the half cell outside only, the natural boundary).
    dt : float, optional
        Time step [s]; default ``cfl * dt_max`` with ``dt_max`` from Lanczos.
    cfl : float
    mutate : {None, 'fraction', 'missing_transverse', 'inject'}
        Deliberately wrong interface operators (verification only).
    edge_scale, face_scale : callable, optional
        ``f(mid (n, 3), direction (n,)) -> complex factors`` applied to the
        edge masses / face Hodge before lumping (complex stretching).
    """

    def __init__(self, mesh, eps_r=None, mu_r=None, pec=None, pmc=None, dt=None, cfl=0.95,
                 mutate=None, edge_scale=None, face_scale=None):  # fmt: skip
        if not isinstance(mesh, SubgridMesh3D):
            raise TypeError("mesh must be a SubgridMesh3D")
        self.mesh = mesh.build()
        self.mutate = mutate
        M = self.mesh
        nE, nF = M.n_edges, M.n_faces
        m_all = np.zeros(nE)
        W_all = np.zeros(nF)
        pec_edge = np.zeros(nE, bool)
        pec_side = np.zeros(nE, bool)
        self.eps_cells, self.mu_cells, self.pec_cells = [], [], []
        mids, lens, dirs, eblk = [], [], [], []
        cens, areas, norms = [], [], []
        curls, divs = [], []
        for bi, blk in enumerate(M.blocks):
            X, Y, Z = blk.cell_centres()
            eps = _material(eps_r, X, Y, Z, 1.0).ravel()
            mu = _material(mu_r, X, Y, Z, 1.0).ravel()
            pc = _mask(pec, X, Y, Z).ravel()
            pm = _mask(pmc, X, Y, Z).ravel()
            act = M.active[bi].ravel()
            self.eps_cells.append(eps)
            self.mu_cells.append(mu)
            self.pec_cells.append(pc & act)
            e0, f0 = M.edge_off[bi], M.face_off[bi]
            c, e, w = blk.cell_edges()
            ok = act[c]
            np.add.at(m_all, e0 + e[ok], EPS0 * eps[c[ok]] * w[ok])
            bad = ok & pc[c]
            pec_edge[e0 + e[bad]] = True
            # PMC cells: mu -> infinity, so they add nothing to the face Hodge
            # (the dual of PEC freezing the edges); faces with W = 0 are dropped
            c, f, w = blk.cell_faces()
            ok = act[c] & ~pm[c]
            np.add.at(W_all, f0 + f[ok], w[ok] / (MU0 * mu[c[ok]]))
            ps, _, _ = M.edge_sides(bi)
            pec_side[e0 : e0 + blk.n_edges] = ps
            mi, le, di = blk.edge_geometry()
            mids.append(mi)
            lens.append(le)
            dirs.append(di)
            eblk.append(np.full(len(le), bi))
            ce, ar, no = blk.face_geometry()
            cens.append(ce)
            areas.append(ar)
            norms.append(no)
            curls.append(blk.curl())
            divs.append(blk.divergence())
        self.edge_mid = np.concatenate(mids)
        self.edge_len = np.concatenate(lens)
        self.edge_dir = np.concatenate(dirs)
        self.edge_block = np.concatenate(eblk)
        self.face_cen = np.concatenate(cens)
        self.face_area = np.concatenate(areas)
        self.face_normal = np.concatenate(norms)
        self.C_all = sp.block_diag(curls, format="csr")
        self.div_all = sp.block_diag(divs, format="csr")
        self.active_cells = np.concatenate([a.ravel() for a in M.active])
        self.pec_cells_all = np.concatenate(self.pec_cells)

        m_all = m_all.astype(complex) if edge_scale is not None else m_all
        W_all = W_all.astype(complex) if face_scale is not None else W_all
        if edge_scale is not None:
            m_all = m_all * edge_scale(self.edge_mid, self.edge_dir)
        if face_scale is not None:
            W_all = W_all * face_scale(self.face_cen, self.face_normal)
        self.m_all = m_all
        self.W_all = W_all

        rows, cols, vals, hanging = M.hanging_rows(
            mutate=mutate if mutate != "inject" else None
        )
        dead = pec_side | pec_edge | ((np.abs(m_all) == 0) & ~hanging)
        hang_pec = hanging & pec_edge
        if np.any(hang_pec):
            # a PEC fine cell on an interface needs PEC coarse parents
            sel = np.isin(rows, np.flatnonzero(hang_pec))
            if np.any(~dead[cols[sel]]):
                raise ValueError(
                    "a PEC cell touches a patch interface whose coarse edges are not PEC; "
                    "align PEC objects with coarse cells across interfaces"
                )
        hanging = hanging & ~dead
        self._pec_dead = pec_side | pec_edge
        free = ~dead & ~hanging
        self.free_edges = np.flatnonzero(free)
        self.hanging = hanging
        self.dead = dead
        self.n_free = len(self.free_edges)
        col_of = -np.ones(nE, int)
        col_of[self.free_edges] = np.arange(self.n_free)
        keep = hanging[rows] & free[cols]
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        cplx = (
            any(blk.complex for blk in M.blocks)
            or np.iscomplexobj(m_all)
            or np.iscomplexobj(W_all)
        )
        self.dtype = complex if cplx else float
        if not any(blk.complex for blk in M.blocks):
            vals = vals.real
        P = sp.csr_matrix(
            (
                np.concatenate([np.ones(self.n_free), vals]),
                (
                    np.concatenate([self.free_edges, rows]),
                    np.concatenate([np.arange(self.n_free), col_of[cols]]),
                ),
            ),
            shape=(nE, self.n_free),
        )
        self.P = P
        self.faces = np.flatnonzero(np.abs(W_all) > 0)
        self.n_faces = len(self.faces)
        Cf = self.C_all[self.faces]
        self.K = (Cf @ P).tocsr()
        self.W = W_all[self.faces]
        absP = abs(P)
        self.D = absP.T @ (m_all * np.asarray(absP.sum(axis=1)).ravel())
        if np.any(np.abs(self.D) == 0):
            raise RuntimeError("free edge with zero mass (internal error)")
        self.Dinv = 1.0 / self.D
        if mutate == "inject":
            Q = _injection(P, self.hanging, self.free_edges)
            self.KH = (Cf @ Q).conj().T.tocsr()
        else:
            self.KH = self.K.conj().T.tocsr()
        self._lam_max = None
        if dt is None:
            dt = cfl * self.dt_max
        self.dt = float(dt)
        self.sources = []
        self.probes = []
        self.reset()

    # ------------------------------------------------------------- stability
    @property
    def lambda_max(self):
        """Largest eigenvalue of D^-1 K^H W K [1/s^2] (Lanczos)."""
        if self._lam_max is None:
            self._lam_max = _lanczos_max(self.K, self.W, self.D)
        return self._lam_max

    @property
    def dt_max(self):
        """Stability limit 2 / sqrt(lambda_max) [s]."""
        return 2.0 / np.sqrt(self.lambda_max)

    def block_lambda_bounds(self):
        """
        lambda_max of every block alone (its own edges, natural boundaries,
        no constraints). Their maximum bounds the composite lambda_max.
        """
        M = self.mesh
        out = []
        for bi in range(M.n_blocks):
            e0, e1 = M.edge_off[bi], M.edge_off[bi + 1]
            f0, f1 = M.face_off[bi], M.face_off[bi + 1]
            m = self.m_all[e0:e1]
            keep_e = (np.abs(m) > 0) & ~self._pec_dead[e0:e1]
            fsel = self.faces[(self.faces >= f0) & (self.faces < f1)]
            Cb = self.C_all[fsel][:, e0 + np.flatnonzero(keep_e)]
            out.append(_lanczos_max(Cb.tocsr(), self.W_all[fsel], m[keep_e]))
        return np.array(out)

    # --------------------------------------------------------------- fields
    def reset(self):
        self.e = np.zeros(self.n_free, self.dtype)
        self.b = np.zeros(self.n_faces, self.dtype)
        self.t = 0.0
        self.n = 0
        for p in self.probes:
            p["values"] = []
            if p["dft"] is not None:
                p["dft"][:] = 0

    def e_all(self, e=None):
        """Voltages of all edges (hanging ones from the trace) [V]."""
        return self.P @ (self.e if e is None else e)

    def b_all(self, b=None):
        out = np.zeros(self.mesh.n_faces, self.dtype)
        out[self.faces] = self.b if b is None else b
        return out

    def energy(self, e_n, e_np1, b_half):
        """Discrete energy 1/2 b.W b + 1/2 e^n.D e^{n+1} [J]."""
        return float(
            0.5 * np.real(np.vdot(b_half, self.W * b_half))
            + 0.5 * np.real(np.vdot(e_n, self.D * e_np1))
        )

    def divergence(self, b=None):
        """Flux out of every active leaf cell [Wb]."""
        d = self.div_all @ self.b_all(b)
        return d[self.active_cells]

    def interface_flux_mismatch(self, b=None):
        """
        Coarse face flux minus the sum of its r^2 fine sub-face fluxes, for
        every coarse face on an interface side [Wb].
        """
        S = self._interface_aggregation()
        ball = self.b_all(b)
        return S @ ball

    def _interface_aggregation(self):
        if getattr(self, "_S", None) is not None:
            return self._S
        M = self.mesh
        rows, cols, vals = [], [], []
        nrow = 0
        for bi in range(1, M.n_blocks):
            blk = M.blocks[bi]
            p = M.patches[bi - 1]
            pb = M.blocks[p.parent]
            r = p.ratio
            for a in range(3):
                if blk.periodic[a]:
                    continue
                b, c = (a + 1) % 3, (a + 2) % 3
                for side in (0, 1):
                    if M.sides[bi][a][side] != "interface":
                        continue
                    nf = blk.nn[a] - 1 if side else 0
                    npar = p.hi[a] if side else p.lo[a]
                    ijk = blk.face_grid(a)
                    sel = ijk[a] == nf
                    q = [v[sel] for v in ijk]
                    fid, _ = blk.face_index(a, *q)
                    pq = [None] * 3
                    pq[a] = np.full(q[a].shape, npar)
                    pq[b] = p.lo[b] + q[b] // r
                    pq[c] = p.lo[c] + q[c] // r
                    pid, _ = pb.face_index(a, *pq)
                    upid, inv = np.unique(pid, return_inverse=True)
                    rows.append(nrow + np.arange(len(upid)))
                    cols.append(M.face_off[p.parent] + upid)
                    vals.append(np.ones(len(upid)))
                    rows.append(nrow + inv)
                    cols.append(M.face_off[bi] + fid)
                    vals.append(-np.ones(len(fid)))
                    nrow += len(upid)
        self._S = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(nrow, M.n_faces),
        )
        return self._S

    # -------------------------------------------------------------- lookups
    def locate_edge(self, point, component):
        """All-edge index of the edge of direction `component` nearest `point`
        in the finest block containing it."""
        bi = self.mesh.finest_block(point)
        sel = np.flatnonzero(
            (self.edge_block == bi) & (self.edge_dir == component) & ~self.dead
        )
        d = np.linalg.norm(self.edge_mid[sel] - np.asarray(point, float), axis=1)
        return int(sel[np.argmin(d)])

    # ------------------------------------------------------ sources/probes
    def add_current(self, point, component, waveform, amplitude=1.0):
        """
        Soft current source ``i(t) = amplitude * waveform(t)`` [A] through the
        dual face of the edge nearest `point` (direction 0/1/2 = x/y/z).
        """
        k = self.locate_edge(point, component)
        row = self.P[k].conj()
        self.sources.append({"edge": k, "row": row.indices.copy(), "w": row.data.copy(),
                             "f": waveform, "amp": float(amplitude)})  # fmt: skip
        return k

    def add_probe(self, point, component, freqs=None):
        """Point probe of E_component [V/m] at the nearest edge; optional DFT."""
        k = self.locate_edge(point, component)
        row = self.P[k]
        freqs = None if freqs is None else np.atleast_1d(np.asarray(freqs, float))
        self.probes.append({"edge": k, "row": row, "L": self.edge_len[k], "values": [],
                            "freqs": freqs,
                            "dft": None if freqs is None else np.zeros(len(freqs), complex)})  # fmt: skip
        return len(self.probes) - 1

    def _source_vector(self, t):
        if not self.sources:
            return None
        s = np.zeros(self.n_free, self.dtype)
        for src in self.sources:
            s[src["row"]] += src["amp"] * src["f"](t) * src["w"]
        return s

    def _record(self):
        for p in self.probes:
            v = (p["row"] @ self.e).item() / p["L"]
            p["values"].append(v)
            if p["dft"] is not None:
                p["dft"] += v * np.exp(-1j * 2 * np.pi * p["freqs"] * self.t) * self.dt

    # ------------------------------------------------------------ stepping
    def one_step(self):
        dt = self.dt
        self.b -= dt * (self.K @ self.e)
        rhs = self.KH @ (self.W * self.b)
        s = self._source_vector(self.t + 0.5 * dt)
        if s is not None:
            rhs -= s
        self.e += dt * self.Dinv * rhs
        self.n += 1
        self.t = self.n * dt

    def run(self, n_steps, energy=False):
        """Advance `n_steps`; record probes at every integer time level.
        Returns the energy history (length n_steps) if `energy`."""
        hist = []
        if self.n == 0:
            self._record()
        for _ in range(int(n_steps)):
            e_old = self.e.copy() if energy else None
            self.one_step()
            self._record()
            if energy:
                hist.append(self.energy(e_old, self.e, self.b))
        return np.array(hist) if energy else None

    def linear_step(self, b, e, i_all=None):
        """
        One step as a linear map of (b^{n-1/2}, e^n, i^{n+1/2} on all edges).
        Returns (b^{n+1/2}, e^{n+1}).
        """
        dt = self.dt
        b1 = b - dt * (self.K @ e)
        rhs = self.KH @ (self.W * b1)
        if i_all is not None:
            rhs = rhs - self.P.conj().T @ i_all
        return b1, e + dt * self.Dinv * rhs

    def linear_step_T(self, bb1, eb1):
        """
        Exact transpose of `linear_step`: returns (bb, eb, ib_all) with
        <step(x), y> = <x, step_T(y)> (bilinear, no conjugation).
        """
        dt = self.dt
        g = self.Dinv * eb1
        bt = bb1 + dt * self.W * (self.KH.T @ g)
        eb = eb1 - dt * (self.K.T @ bt)
        ib = -dt * (self.P.conj() @ g)
        return bt, eb, ib

    # -------------------------------------------------------------- modes
    def stiffness(self):
        """(K^H W K, D) of the semi-discrete eigenproblem K^H W K x = w^2 D x."""
        return (self.KH @ sp.diags(self.W) @ self.K).tocsc(), self.D

    def eigenfrequencies(self, k, f_guess):
        """
        The `k` lowest resonance frequencies above ``0.5 f_guess`` [Hz] of
        the semi-discrete operator (shift-invert Lanczos).
        """
        A, D = self.stiffness()
        sigma = (2 * np.pi * 0.5 * f_guess) ** 2
        # extra vectors: ARPACK can miss copies of (near-)degenerate modes
        kk = min(A.shape[0] - 2, k + 8)
        lam = sla.eigsh(A, k=kk, M=sp.diags(D).tocsc(), sigma=sigma, which="LA",
                        ncv=min(A.shape[0] - 1, max(4 * kk, 40)), tol=1e-12,
                        return_eigenvectors=False)  # fmt: skip
        return (np.sort(np.sqrt(np.abs(lam))) / (2 * np.pi))[:k]


def _injection(P, hanging, free_edges):
    """Mutation: each hanging row sends its whole weight to its largest parent."""
    P = P.tocsr()
    rows, cols, vals = [], [], []
    for r in range(P.shape[0]):
        s, t = P.indptr[r], P.indptr[r + 1]
        if t == s:
            continue
        d = P.data[s:t]
        j = int(np.argmax(np.abs(d)))
        rows.append(r)
        cols.append(P.indices[s + j])
        vals.append(d.sum())
    return sp.csr_matrix((vals, (rows, cols)), shape=P.shape)


def _lanczos_max(K, W, D):
    n = K.shape[1]
    s = 1.0 / np.sqrt(D)
    KH = K.conj().T.tocsr()

    def mv(x):
        return s * (KH @ (W * (K @ (s * x))))

    dtype = np.result_type(K.dtype, np.asarray(W).dtype, np.asarray(D).dtype)
    op = sla.LinearOperator((n, n), matvec=mv, dtype=dtype)
    v0 = np.random.default_rng(1).standard_normal(n).astype(dtype)
    lam = sla.eigsh(
        op, k=1, which="LA", tol=1e-12, v0=v0, return_eigenvectors=False, ncv=min(n, 40)
    )
    return float(np.real(lam[0]))
