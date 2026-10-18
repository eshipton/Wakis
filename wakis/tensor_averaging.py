# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Full-tensor subpixel averaging of the permittivity on the Cartesian FIT grid.

At a dielectric interface with unit normal n crossing the dual cell of an E
edge with fill fraction f, the effective inverse permittivity is the tensor
[Kottke, Farjadpour & Johnson 2008]

    T = <1/eps> n n^T + (I - n n^T) / <eps> = I / <eps> + a n n^T,
    a = <1/eps> - 1/<eps> >= 0.

Conformal mode 'all' keeps only the diagonal T_cc of every edge. This module
adds the off-diagonal couplings T_ij, i != j, between the x, y and z edges
meeting at the corners of the primary cells, through a symmetric corner
(octant) Hodge operator, so that the material operator becomes

    E = N D,   N = diag(ieps) + K,   W N symmetric positive definite,

with W = diag(L tA) the dual volumes of the E edges (the energy weights of
the FIT update). Each octant o of a primary cell (volume V/8) contains one
x, one y and one z edge (the three edges leaving the cell corner into the
cell) and adds w_o t_ab to the symmetric matrix S = W K for every pair (a, b)
of them, with the coupling of the edges a (component i) and b (component j)

    t_ab = (T_ij(a) + T_ij(b)) / 2,   T_ij(e) = a_e m_i(e) m_j(e),

the average of the off-diagonal tensors of the two edges' own averaging
pixels (the symmetric part of the Meep-type 4-point interpolation of the
other D components). Per octant the off-diagonals are scaled down where
needed so that diag(d) + offdiag(t) is positive definite (d the per-octant
share of the diagonal), which makes W N positive definite and the leapfrog
update stable with a conserved discrete energy.

The time step applies N as the solver's diagonal step followed by a local
correction on the coupled edges R:

    E' = E* + K' (E* - E),   K' = K diag(1/ieps) on R,

where E* = E + dt ieps (curl H - J) is the diagonal step. This keeps every
backend (NumPy/MKL, CuPy, CPML, MPI) unchanged and the step linear in the
fields; its transpose is E* -> (I + K'^H), E -> -K'^H on R. See "Tensor
subpixel averaging" in the physics guide.
"""

import types

import numpy as np
from scipy.constants import c as c_light
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, eigs

# Per-octant positive-definiteness margin: lambda_min(I + O) >= 1 - _PD_LIMIT
_PD_LIMIT = 0.99

# Component pairs (i, j) of the off-diagonal entries T_ij of an octant
_PAIRS = ((0, 1), (0, 2), (1, 2))


def _host(array):
    return np.asarray(array.get() if hasattr(array, "get") else array)


class TensorOperator:
    def __init__(self, solver):
        """
        Off-diagonal (tensor) part of the inverse permittivity of a solver.

        Holds, per E edge with an anisotropic averaging pixel, the
        coefficient a = <1/eps> - 1/<eps> [m/F] and the vector m (|m| <= 1,
        the unit interface normal, or a regularized density gradient for a
        `MaterialGrid`), and assembles the correction K' on the coupled edges.

        Parameters
        ----------
        solver : SolverFIT3D
            Solver before its first time step.
        """
        self.shape = (solver.Nx, solver.Ny, solver.Nz)
        self.N = solver.N
        self.periodic = [
            not solver.grid.edge_ghosts[a]
            and str(solver.bc_low[a]).lower() in ("periodic", "bloch")
            for a in range(3)
        ]
        self.phase = [complex(p) for p in getattr(solver, "bloch_phase", [1, 1, 1])]
        self.edges = np.zeros(0, dtype=np.int64)
        self.aniso = np.zeros(0)
        self.vec = np.zeros((0, 3))
        self.rows = None
        self.Kp = None
        self.last_increment = None
        self.n_clamped = 0
        self.mu_max = 0.0
        self.installed = False

    # ------------------------------------------------------------- edge data
    def set_edges(self, edges, aniso, vec):
        """
        Set (or replace) the anisotropic data of the E edges `edges`.

        Parameters
        ----------
        edges : ndarray of int
            Flat E indices.
        aniso : ndarray
            a = <1/eps> - 1/<eps> per edge [m/F].
        vec : ndarray, shape (len(edges), 3)
            m per edge; T_ij = a m_i m_j.
        """
        edges = np.asarray(edges, dtype=np.int64)
        keep = ~np.isin(self.edges, edges)
        new = np.asarray(aniso) > 0
        all_edges = np.concatenate([self.edges[keep], edges[new]])
        order = np.argsort(all_edges)
        self.edges = all_edges[order]
        self.aniso = np.concatenate([self.aniso[keep], np.asarray(aniso)[new]])[order]
        self.vec = np.concatenate([self.vec[keep], np.asarray(vec)[new]])[order]

    def _lookup(self, flat):
        """Anisotropy and vector of the edges `flat` (zero if not stored)."""
        flat = np.asarray(flat, dtype=np.int64)
        if self.edges.size == 0:
            return np.zeros(flat.shape), np.zeros((*flat.shape, 3))
        pos = np.minimum(np.searchsorted(self.edges, flat), self.edges.size - 1)
        hit = self.edges[pos] == flat
        aniso = np.where(hit, self.aniso[pos], 0.0)
        vec = np.where(hit[:, None], self.vec[pos], 0.0)
        return aniso, vec

    # -------------------------------------------------------------- geometry
    def _flat(self, comp, i, j, k):
        nx, ny, _ = self.shape
        return comp * self.N + i + nx * j + nx * ny * k

    def _cells(self):
        """Primary cells (i, j, k) touching an anisotropic edge."""
        nx, ny, nz = self.shape
        e = self.edges
        comp, rest = np.divmod(e, self.N)
        k, rest = np.divmod(rest, nx * ny)
        j, i = np.divmod(rest, nx)
        cells = []
        for c in range(3):
            sel = comp == c
            ii, jj, kk = i[sel], j[sel], k[sel]
            # The four cells sharing an edge along c: offsets 0/-1 transversally
            u, v = [a for a in range(3) if a != c]
            for du in (0, -1):
                for dv in (0, -1):
                    idx = [ii.copy(), jj.copy(), kk.copy()]
                    idx[u] += du
                    idx[v] += dv
                    cells.append(np.stack(idx, axis=1))
        cells = np.unique(np.concatenate(cells or [np.zeros((0, 3), int)]), axis=0)
        ok = np.ones(len(cells), dtype=bool)
        for a in range(3):
            n = self.shape[a]
            if self.periodic[a]:
                cells[:, a] %= n
            else:
                # Cell i spans nodes i and i+1; the last index is a ghost layer
                ok &= (cells[:, a] >= 0) & (cells[:, a] <= n - 2)
        return np.unique(cells[ok], axis=0)

    def _wrap(self, idx, axis):
        """Index along `axis` with periodic wrap; returns (index, wrapped)."""
        n = self.shape[axis]
        if self.periodic[axis]:
            return idx % n, idx >= n
        return idx, np.zeros(idx.shape, dtype=bool)

    def _dual_volume(self, comp, i, j, k, d):
        """Sum of V/8 over the in-domain octants that contain the edges."""
        idx = [i, j, k]
        u, v = [a for a in range(3) if a != comp]
        total = np.zeros(i.shape)
        for du in (0, -1):
            for dv in (0, -1):
                cell = [x.copy() for x in idx]
                cell[u] = cell[u] + du
                cell[v] = cell[v] + dv
                ok = np.ones(i.shape, dtype=bool)
                vol = np.ones(i.shape)
                for a in range(3):
                    n = self.shape[a]
                    if self.periodic[a]:
                        cell[a] = cell[a] % n
                    else:
                        ok &= (cell[a] >= 0) & (cell[a] <= n - 2)
                        cell[a] = np.clip(cell[a], 0, n - 2)
                    vol = vol * d[a][cell[a]]
                # Two octants (both ends of the edge) of each adjacent cell
                total += np.where(ok, vol / 4, 0.0)
        return total

    # -------------------------------------------------------------- assembly
    def assemble(self, solver, exclude=None):
        """
        Assemble K' on the coupled edges from the stored edge data.

        Parameters
        ----------
        solver : SolverFIT3D
            Solver with final ieps, edge lengths and dual areas.
        exclude : ndarray of int, optional
            Flat E indices that must not be coupled (e.g. dispersive edges,
            corrected by their own ADE update).
        """
        ieps = _host(solver.ieps.array).astype(float)
        if exclude is not None and len(exclude):
            ieps = ieps.copy()
            ieps[np.asarray(exclude, dtype=np.int64)] = 0.0
        L = (
            _host(solver.L.toarray())
            if hasattr(solver, "L")
            else _host(solver.grid.L.toarray())
        )
        itA = _host(solver.grid.itA.toarray())
        W = np.where(itA != 0, L / np.where(itA != 0, itA, 1.0), 0.0)
        d = [
            np.asarray(solver.grid.dx),
            np.asarray(solver.grid.dy),
            np.asarray(solver.grid.dz),
        ]
        # Periodic axes have one cell per sample; others one cell less
        d = [np.resize(d[a], self.shape[a]) for a in range(3)]

        cells = self._cells()
        if len(cells) == 0:
            self._finish(
                solver, np.zeros(0, int), np.zeros(0, int), np.zeros(0), W, ieps
            )
            return
        ci, cj, ck = cells.T
        vol8 = d[0][ci] * d[1][cj] * d[2][ck] / 8
        recs_a, recs_b, recs_w = [], [], []
        self.n_clamped = 0
        self.mu_max = 0.0
        for a, b, c in np.ndindex(2, 2, 2):
            ix, wx = self._wrap(ci + a, 0)
            jy, wy = self._wrap(cj + b, 1)
            kz, wz = self._wrap(ck + c, 2)
            flats = [
                self._flat(0, ci, jy, kz),
                self._flat(1, ix, cj, kz),
                self._flat(2, ix, jy, ck),
            ]
            wraps = [
                np.stack([np.zeros_like(wx), wy, wz], 1),
                np.stack([wx, np.zeros_like(wy), wz], 1),
                np.stack([wx, wy, np.zeros_like(wz)], 1),
            ]
            data = [self._lookup(f) for f in flats]
            diag = []
            for comp in range(3):
                f = flats[comp]
                vgeo = self._dual_volume(comp, *self._unflat(f), d)
                diag.append(
                    np.where(
                        vgeo > 0, ieps[f] * W[f] / np.where(vgeo > 0, vgeo, 1.0), 0.0
                    )
                )
            diag = np.stack(diag, 1)  # (n, 3) per-octant diagonal shares
            t = np.zeros((len(cells), 3, 3))
            for p, q in _PAIRS:
                ap, mp = data[p]
                aq, mq = data[q]
                val = 0.5 * (ap * mp[:, p] * mp[:, q] + aq * mq[:, p] * mq[:, q])
                val = np.where((diag[:, p] > 0) & (diag[:, q] > 0), val, 0.0)
                t[:, p, q] = t[:, q, p] = val
            active = np.any(t != 0, axis=(1, 2))
            if not active.any():
                continue
            # Normalized off-diagonal block O = D^-1/2 t D^-1/2
            s = np.ones(len(cells))
            dsq = np.sqrt(np.where(diag > 0, diag, 1.0))
            O = t[active] / (dsq[active][:, :, None] * dsq[active][:, None, :])
            lam = np.linalg.eigvalsh(O)
            lo = lam[:, 0]
            scale = np.where(lo < -_PD_LIMIT, _PD_LIMIT / np.maximum(-lo, 1e-300), 1.0)
            s[active] = scale
            self.n_clamped += int(np.sum(scale < 1))
            self.mu_max = max(self.mu_max, float(np.max(lam[:, -1] * scale)))
            for p, q in _PAIRS:
                sel = active & (t[:, p, q] != 0)
                rel = wraps[q][sel].astype(int) - wraps[p][sel].astype(int)
                ph = np.ones(sel.sum(), dtype=complex)
                for ax in range(3):
                    ph *= self.phase[ax] ** rel[:, ax]
                recs_a.append(flats[p][sel])
                recs_b.append(flats[q][sel])
                recs_w.append(vol8[sel] * s[sel] * ph / 2)
        if not recs_a:
            self._finish(
                solver, np.zeros(0, int), np.zeros(0, int), np.zeros(0), W, ieps
            )
            return
        ea = np.concatenate(recs_a)
        eb = np.concatenate(recs_b)
        cw = np.concatenate(recs_w)
        # Aggregate octants sharing a pair: S_ab = c_ab (T(a) + T(b))
        key = np.stack([ea, eb], 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        coef = np.zeros(len(uniq), dtype=complex)
        np.add.at(coef, inv.ravel(), cw)
        if np.all(coef.imag == 0):
            coef = coef.real
        self._finish(solver, uniq[:, 0], uniq[:, 1], coef, W, ieps)

    def _unflat(self, flat):
        nx, ny, _ = self.shape
        rest = flat % self.N
        k, rest = np.divmod(rest, nx * ny)
        j, i = np.divmod(rest, nx)
        return i, j, k

    def pair_values(self, ea, eb):
        """T(a) + T(b) of the pairs (a, b), components from their axes."""
        ca, cb = ea // self.N, eb // self.N
        aa, ma = self._lookup(ea)
        ab, mb = self._lookup(eb)
        n = np.arange(len(ea))
        return aa * ma[n, ca] * ma[n, cb] + ab * mb[n, ca] * mb[n, cb]

    def _finish(self, solver, ea, eb, coef, W, ieps):
        """Build K'_ab = S_ab / (W_a ieps_b) on the coupled edges."""
        self.pair_a, self.pair_b, self.pair_coef = ea, eb, coef
        self.W = W
        S = coef * self.pair_values(ea, eb)
        rows = np.unique(np.concatenate([ea, eb]))
        self.rows = rows
        ra = np.searchsorted(rows, ea)
        rb = np.searchsorted(rows, eb)
        vals_ab = S / (W[ea] * ieps[eb]) if len(ea) else S
        vals_ba = np.conj(S) / (W[eb] * ieps[ea]) if len(ea) else S
        dtype = np.result_type(np.asarray(S).dtype, solver.E.array.dtype)
        m = rows.size
        self.Kp_host = coo_matrix(
            (np.concatenate([vals_ab, vals_ba]).astype(dtype),
             (np.concatenate([ra, rb]), np.concatenate([rb, ra]))),
            shape=(m, m),
        ).tocsr()  # fmt: skip
        self._to_backend(solver)

    def _to_backend(self, solver):
        if solver.use_gpu:
            import cupy
            from cupyx.scipy.sparse import csr_matrix as gpu_csr

            self.rows_dev = cupy.asarray(self.rows)
            self.Kp = gpu_csr(self.Kp_host)
        else:
            self.rows_dev = self.rows
            self.Kp = self.Kp_host

    # ------------------------------------------------------------- operators
    def apply_N(self, x, ieps):
        """N x = ieps x + K x (host arrays), for the stability estimate."""
        y = ieps * x
        if self.rows.size:
            r = self.rows
            y = y.astype(np.result_type(y, self.Kp_host.dtype))
            y[r] += self.Kp_host @ (ieps[r] * x[r])
        return y

    def install(self, solver):
        """
        Insert the tensor correction after the solver's current `one_step`.

        Couplings to edges of dispersive groups (corrected by their own ADE
        update) are dropped first, so that the corrections act on disjoint
        edges and commute.
        """
        groups = getattr(solver, "_dispersive_groups", [])
        exclude = [g["idx_host"] for g in groups]
        net = getattr(solver, "dispersive_network", None)
        if net is not None:  # dispersive subpixel network (diagonal only)
            exclude.append(net.rows)
        exclude = np.concatenate(exclude) if exclude else None
        self.assemble(solver, exclude=exclude)
        if self.installed:
            return
        solver._one_step_untensored = solver.one_step
        solver.one_step = types.MethodType(one_step_tensor, solver)
        self.installed = True

    def correct(self, solver, e_old):
        """E <- E* + K'(E* - E) on the coupled edges, after the base step."""
        E = solver.E.array
        r = self.rows_dev
        delta = E[r] - e_old
        self.last_increment = delta
        E[r] += self.Kp @ delta


def one_step_tensor(solver):
    """Diagonal solver step followed by the tensor correction (bound method)."""
    op = solver.tensor_operator
    e_old = solver.E.array[op.rows_dev]
    solver._one_step_untensored()
    if op.rows.size:
        op.correct(solver, e_old)
    if solver.use_mpi and solver.grid.use_mpi:
        # Ghost layers must hold the corrected neighbour values (on every
        # rank, also those without coupled edges: the exchange is collective)
        solver._mpi_communicate(solver.E)


def max_stable_dt(solver, op, tol=1e-3):
    """
    Largest stable leapfrog time step with the tensor material operator.

    The source-free step is stable for dt < 2 / sqrt(lambda_max(B A)),
    B A = N itDa C^H tDs G C Ds. B A is similar to a Hermitian positive
    semi-definite matrix (W N is symmetric positive definite), so its
    eigenvalues are real; the largest is computed with Arnoldi iterations.

    Returns
    -------
    float
        2 / sqrt(lambda_max) [s].
    """
    ieps = _host(solver.ieps.toarray()).astype(float)
    itda = solver.itDa.diagonal()
    tds = solver.tDs.diagonal()
    ds = solver.Ds.diagonal()
    imu = _host(solver.imu.toarray())
    iDa = solver.iDa.tocsr()
    C = solver.C.tocsr()
    CH = C.conj().T.tocsr()

    def matvec(x):
        y = itda * (CH @ (tds * (iDa @ (imu * (C @ (ds * x))))))
        return op.apply_N(y, ieps)

    n = C.shape[1]
    dtype = np.result_type(solver.dtype, op.Kp_host.dtype)
    lin = LinearOperator((n, n), matvec=matvec, dtype=dtype)
    lam = eigs(lin, k=1, which="LR", tol=tol, return_eigenvectors=False)[0]
    return 2.0 / np.sqrt(abs(lam.real) * (1.0 + 10 * tol))


def bound_stable_dt(solver, op, dt_base):
    """
    Guaranteed stable time step from the per-octant bound.

    Every octant block satisfies D^-1/2 (D + t) D^-1/2 <= (1 + mu_max) I,
    so W N <= (1 + mu_max) diag(W ieps) and the eigenvalues of the tensor
    update are at most (1 + mu_max) times those of the diagonal update,
    which is stable up to `dt_base`.
    """
    return dt_base / np.sqrt(1.0 + max(op.mu_max, 0.0))


def yee_dt(solver):
    """Vacuum Courant limit 1 / (c sqrt(sum 1/d_min^2)) of the grid [s]."""
    return 1.0 / (
        c_light
        * np.sqrt(
            1 / np.min(solver.grid.dx) ** 2
            + 1 / np.min(solver.grid.dy) ** 2
            + 1 / np.min(solver.grid.dz) ** 2
        )
    )
