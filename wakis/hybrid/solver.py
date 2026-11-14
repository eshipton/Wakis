# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Hybrid Cartesian / tetrahedral FIT in 3D (`HybridFIT3D`).

A tensor-product Cartesian grid fills the domain except an axis-aligned box of
whole cells, which is filled by a tetrahedral mesh whose boundary triangles
refine the Cartesian squares of the box surface. Unknowns:

* Cartesian edge voltages and face fluxes of the active (outside) cells, with
  the diagonal FIT masses (row sums of quarter cells, trapezoidal lumping of
  the hexahedral Nedelec element);
* tetrahedral edge voltages and face fluxes, with the consistent Whitney
  1-form and 2-form mass matrices (SPD on any mesh);
* tetrahedral edges on the box surface are *hanging*: their voltage is the
  line integral of the tangential trace of the lowest-order hexahedral Nedelec
  field of the Cartesian square that contains them (`box_trace`).

With the prolongation ``P`` (all edges x free edges), the block-diagonal
face-edge incidence ``C`` and ``K = C P``::

    b^{n+1/2} = b^{n-1/2} - dt K e^n
    M (e^{n+1} - e^n) = dt (K^T W b^{n+1/2} - P^T i^{n+1/2})

with ``W = diag(W_cart) (+) M2_tet`` and ``M = P^T (diag(m_cart) (+) M1_tet) P``
(Galerkin restriction of the composite edge mass). ``W`` and ``M`` are
symmetric positive definite, so the leapfrog conserves

    E = 1/2 b^{n+1/2} . W b^{n+1/2} + 1/2 e^n . M e^{n+1}

and is stable iff ``dt < 2 / sqrt(lambda_max(M^-1 K^T W K))``; that
``lambda_max`` is at most the larger of the two blocks' own maxima (mediant
inequality), so ``dt`` is limited by the finer of the two regions only.
See ``docs/physicsguide_sections/hybrid_cart_tet.md``.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0

from ..subgrid3d.block import CartesianBlock
from ..unstructured._core import cell_materials
from ..unstructured.solver3d import whitney_mass_matrices
from .mesh import box_surface_check

__all__ = ["HybridFIT3D", "box_trace", "MassSolver"]


def _material(f, X, Y, Z, default):
    if f is None:
        return np.full(X.shape, default, float)
    if np.isscalar(f):
        return np.full(X.shape, float(f))
    return np.broadcast_to(np.asarray(f(X, Y, Z), float), X.shape)


def box_trace(block, lo, hi, p, q, mutate=None, tol=1e-9):
    """
    Line integrals of the hexahedral Nedelec trace along segments on a box.

    On the Cartesian square ``[u0, u1] x [v0, v1]`` (tangential axes b, c of a
    box face of normal a) the trace built from the four edge voltages is
    ``E_b = [e_b0 (v1 - v) + e_b1 (v - v0)] / (h_u h_v)`` and
    ``E_c = [e_c0 (u1 - u) + e_c1 (u - u0)] / (h_u h_v)``. It is linear along
    any segment, so the line integral from p to q is exact with the midpoint
    ``(u_m, v_m)``::

        w(e_b0) = du/hu (v1 - v_m)/hv,  w(e_b1) = du/hu (v_m - v0)/hv,
        w(e_c0) = dv/hv (u1 - u_m)/hu,  w(e_c1) = dv/hv (u_m - u0)/hu.

    Parameters
    ----------
    block : CartesianBlock
    lo, hi : (3,) int
        Node-index bounds of the box.
    p, q : (n, 3) float
        Segment end points [m], each segment inside one square of the box
        surface (closed; checked).
    mutate : {None, 'nearest', 'inject', 'scale'}
        Deliberately wrong trace (verification): 'nearest' rounds the
        transverse interpolation to the nearer parallel edge; 'inject' sends
        each row's total weight to its largest parent; 'scale' multiplies
        every weight by 1.01.

    Returns
    -------
    rows, cols, vals : ndarray
        Segment index, Cartesian edge index and weight.
    """
    p, q = np.atleast_2d(p), np.atleast_2d(q)
    nodes = block.nodes
    scale = max(nodes[a][hi[a]] - nodes[a][lo[a]] for a in range(3))
    tol = tol * scale
    n = len(p)
    done = np.zeros(n, bool)
    rows, cols, vals = [], [], []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for na in (lo[a], hi[a]):
            pl = nodes[a][na]
            sel = (
                ~done
                & (np.abs(p[:, a] - pl) < tol)
                & (np.abs(q[:, a] - pl) < tol)
            )
            if not sel.any():
                continue
            done |= sel
            idx = np.flatnonzero(sel)
            m = 0.5 * (p[idx] + q[idx])
            ib = np.clip(np.searchsorted(nodes[b], m[:, b], "right") - 1, lo[b], hi[b] - 1)
            ic = np.clip(np.searchsorted(nodes[c], m[:, c], "right") - 1, lo[c], hi[c] - 1)
            u0, u1 = nodes[b][ib], nodes[b][ib + 1]
            v0, v1 = nodes[c][ic], nodes[c][ic + 1]
            for X in (p[idx], q[idx]):
                if (
                    np.any(X[:, b] < u0 - tol)
                    or np.any(X[:, b] > u1 + tol)
                    or np.any(X[:, c] < v0 - tol)
                    or np.any(X[:, c] > v1 + tol)
                ):
                    raise ValueError("a box-surface edge straddles two Cartesian squares")
            hu, hv = u1 - u0, v1 - v0
            du = q[idx, b] - p[idx, b]
            dv = q[idx, c] - p[idx, c]
            tv = (m[:, c] - v0) / hv
            tu = (m[:, b] - u0) / hu
            if mutate == "nearest":
                tv = np.round(tv)
                tu = np.round(tu)
            wb0, wb1 = du / hu * (1 - tv), du / hu * tv
            wc0, wc1 = dv / hv * (1 - tu), dv / hv * tu
            NA = np.full(len(idx), na)
            ijk = [None] * 3
            ijk[a], ijk[b], ijk[c] = NA, ib, ic
            eb0, _ = block.edge_index(b, *ijk)
            ijk[c] = ic + 1
            eb1, _ = block.edge_index(b, *ijk)
            ijk[b], ijk[c] = ib, ic
            ec0, _ = block.edge_index(c, *ijk)
            ijk[b] = ib + 1
            ec1, _ = block.edge_index(c, *ijk)
            W = np.stack([wb0, wb1, wc0, wc1], 1)
            E = np.stack([eb0, eb1, ec0, ec1], 1)
            if mutate == "scale":
                W = W * 1.01
            if mutate == "inject":
                j = np.argmax(np.abs(W), axis=1)
                tot = W.sum(1)
                W = np.zeros_like(W)
                W[np.arange(len(idx)), j] = tot
            rows.append(np.repeat(idx, 4))
            cols.append(E.ravel())
            vals.append(W.ravel())
    if not done.all():
        raise ValueError("segment not on the box surface")
    rows, cols, vals = (np.concatenate(v) for v in (rows, cols, vals))
    nz = vals != 0
    return rows[nz], cols[nz], vals[nz]


class MassSolver:
    """
    Linear solves with a sparse SPD mass matrix that is diagonal outside a
    coupled block (the rows with off-diagonal entries).

    The coupled block ``B`` is either factorized (SuperLU, ``method='lu'``)
    or inverted by a fixed-degree Chebyshev iteration on the Jacobi-scaled
    matrix ``S = D^-1/2 B D^-1/2`` (``method='chebyshev'``)::

        x = D^-1/2 p_k(S) D^-1/2 y,   p_k(s) ~ 1/s on [lam_min, lam_max].

    The Chebyshev operator is a fixed polynomial of a symmetric matrix: it is
    exactly linear and symmetric (its transpose is itself), so the leapfrog
    step built on it is exactly transposable and conserves the energy of the
    SPD mass ``p_k(S)^-1`` (which differs from ``B`` by the relative
    polynomial error, at most `tol`). Mass matrices are spectrally equivalent
    to their diagonal (Wathen 1987), so ``k`` does not grow with refinement.
    The degree is ``k = ceil(log(2 / tol) / log(1 / q))`` with
    ``q = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)`` from Lanczos estimates of the
    spectrum with safety factors.
    """

    def __init__(self, M, method="auto", tol=1e-16, lu_max=4000):
        M = sp.csr_matrix(M)
        self.M = M
        d = M.diagonal()
        off = M - sp.diags(d)
        off.eliminate_zeros()
        coupled = np.diff(off.tocsr().indptr) > 0
        self.S = np.flatnonzero(coupled)
        self.dinv = np.where(coupled, 0.0, 1.0 / np.where(d == 0, 1.0, d))
        if np.any((d == 0) & ~coupled):
            raise RuntimeError("zero mass on an uncoupled edge")
        if method == "auto":
            method = "lu" if self.S.size <= lu_max else "chebyshev"
        self.method = method
        self.lu = None
        self.degree = 0
        if not self.S.size:
            return
        B = M[self.S][:, self.S]
        if method == "lu":
            self.lu = sla.splu(B.tocsc(), permc_spec="MMD_AT_PLUS_A")
            return
        if method != "chebyshev":
            raise ValueError("method must be 'auto', 'lu' or 'chebyshev'")
        if np.any(B.diagonal() <= 0):
            raise RuntimeError("non-positive diagonal in the coupled mass block")
        self.sc = 1.0 / np.sqrt(B.diagonal())
        Sm = (sp.diags(self.sc) @ B @ sp.diags(self.sc)).tocsr()
        self.Sm = Sm
        rng = np.random.default_rng(3)
        v0 = rng.standard_normal(Sm.shape[0])
        hi = sla.eigsh(Sm, k=1, which="LA", tol=1e-6, v0=v0, return_eigenvectors=False)[0]
        lo = sla.eigsh(Sm, k=1, which="SA", tol=1e-6, v0=v0, maxiter=50000,
                       return_eigenvectors=False)[0]  # fmt: skip
        if lo <= 0:
            raise RuntimeError("coupled mass block is not positive definite")
        self.lam_lo, self.lam_hi = 0.9 * lo, 1.02 * hi
        kap = self.lam_hi / self.lam_lo
        q = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
        self.degree = int(np.ceil(np.log(2 / tol) / np.log(1 / q)))

    def _cheb(self, y):
        """p_k(Sm) y (Chebyshev semi-iteration for Sm x = y from x = 0)."""
        a, b = self.lam_lo, self.lam_hi
        theta, delta = 0.5 * (b + a), 0.5 * (b - a)
        sigma = theta / delta
        rho = 1.0 / sigma
        x = np.zeros_like(y)
        r = y.copy()
        d = r / theta
        for _ in range(self.degree):
            x += d
            r -= self.Sm @ d
            rho_new = 1.0 / (2 * sigma - rho)
            d = rho_new * rho * d + (2 * rho_new / delta) * r
            rho = rho_new
        return x

    def solve(self, x, trans="N"):
        y = self.dinv * x
        if not self.S.size:
            return y
        xs = np.ascontiguousarray(x[self.S])
        if self.lu is not None:
            y[self.S] = self.lu.solve(xs, trans=trans)
        else:  # symmetric operator: trans is itself
            y[self.S] = self.sc * self._cheb(self.sc * xs)
        return y


class HybridFIT3D:
    """
    Cartesian FIT with a tetrahedral FIT box patch, energy-conserving leapfrog.

    Parameters
    ----------
    xn, yn, zn : array_like
        Cartesian node lines [m] (non-uniform allowed).
    box : ((i0, i1), (j0, j1), (k0, k1)) or None
        Node-index bounds of the tetrahedral box (at least one cell from the
        outer walls). None: pure Cartesian (the baseline).
    tet_mesh : SimplexMesh
        Tetrahedral mesh filling the box; boundary triangles on the box
        surface each inside one Cartesian square. Other boundary faces
        (holes) are PEC.
    materials : dict, optional
        ``{tag: [eps_r, mu_r]}`` or ``"pec"`` for the tetrahedra.
    eps_r, mu_r : float or callable(X, Y, Z), optional
        Cartesian cell materials (sampled at cell centres).
    pec : callable(X, Y, Z) -> bool, optional
        Cartesian PEC cells (staircase).
    outer_bc : {'pec', 'pmc'}
        Outer walls of the Cartesian grid.
    edge_scale, face_scale : callable, optional
        ``f(mid (n, 3), direction (n,)) -> factors`` multiplying the Cartesian
        edge masses / face Hodge entries (complex coordinate stretching for
        frequency-domain absorbers; time stepping needs real values).
    dt, cfl : float
        Time step [s] or fraction of the computed stability limit.
    mass_solver : {'auto', 'lu', 'chebyshev'}
        Solver of the coupled (tetrahedral + interface) mass block, see
        `MassSolver`.
    mass_tol : float
        Chebyshev tolerance. Any value < 1 keeps the scheme exactly linear,
        transposable and energy conserving, for the SPD mass
        ``M~ = (Chebyshev operator)^-1`` with ``|M~ - M| <= mass_tol M``
        (spectral sense); `energy` evaluates with ``M``, so it then
        oscillates within ``~mass_tol`` without drift. A looser tolerance
        cuts the polynomial degree (cost per step) proportionally to
        ``log(2 / mass_tol)``.
    mutate : {None, 'nearest', 'inject', 'scale', 'asym'}
        Deliberately wrong couplings for the verification tests: 'nearest',
        'inject' and 'scale' alter the trace (`box_trace`; still energy
        conserving, inconsistent);
        'asym' uses the 'inject' trace only in the e-update (K^T replaced by a
        different matrix: not energy conserving, not transposable).
    """

    def __init__(self, xn, yn, zn, box=None, tet_mesh=None, materials=None, eps_r=None,
                 mu_r=None, pec=None, outer_bc="pec", edge_scale=None, face_scale=None,
                 dt=None, cfl=0.95, mutate=None, mass_solver="auto", mass_tol=1e-16):  # fmt: skip
        if outer_bc not in ("pec", "pmc"):
            raise ValueError("outer_bc must be 'pec' or 'pmc'")
        if (box is None) != (tet_mesh is None):
            raise ValueError("give both box and tet_mesh, or neither")
        self.block = blk = CartesianBlock(xn, yn, zn)
        self.mutate = mutate
        self.mass_solver = mass_solver
        self.mass_tol = mass_tol
        nc = blk.nc
        # ---------------------------------------------------- Cartesian part
        active = np.ones(nc, bool)
        if box is not None:
            lo = np.array([box[a][0] for a in range(3)])
            hi = np.array([box[a][1] for a in range(3)])
            if np.any(lo < 1) or np.any(hi > np.array(nc) - 1) or np.any(hi <= lo):
                raise ValueError("box must lie at least one cell inside the walls")
            active[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = False
            self.box_lo, self.box_hi = lo, hi
        X, Y, Z = blk.cell_centres()
        eps = _material(eps_r, X, Y, Z, 1.0).ravel()
        mu = _material(mu_r, X, Y, Z, 1.0).ravel()
        pc = (
            np.zeros(X.size, bool)
            if pec is None
            else np.broadcast_to(np.asarray(pec(X, Y, Z), bool), X.shape).ravel()
        )
        act = active.ravel()
        self.active_cells = act
        nEc, nFc = blk.n_edges, blk.n_faces
        m_c = np.zeros(nEc)
        dead_c = np.zeros(nEc, bool)
        c, e, w = blk.cell_edges()
        ok = act[c]
        np.add.at(m_c, e[ok], EPS0 * eps[c[ok]] * w[ok])
        dead_c[e[ok & pc[c]]] = True
        self.pec_edges_c = np.zeros(nEc, bool)
        self.pec_edges_c[e[ok & pc[c]]] = True
        W_c = np.zeros(nFc)
        c, f, w = blk.cell_faces()
        ok = act[c]
        np.add.at(W_c, f[ok], w[ok] / (MU0 * mu[c[ok]]))
        mid, L, d = blk.edge_geometry()
        cen, area, nrm = blk.face_geometry()
        self.edge_mid, self.edge_len, self.edge_dir = mid, L, d
        self.face_cen, self.face_area_c, self.face_normal = cen, area, nrm
        if outer_bc == "pec":
            for a in range(3):
                ijk = blk.edge_grid(a)
                wall = np.zeros(ijk[0].shape, bool)
                for ax in range(3):
                    if ax != a:
                        wall |= (ijk[ax] == 0) | (ijk[ax] == blk.nn[ax] - 1)
                dead_c[blk.edge_off[a] + np.flatnonzero(wall.ravel())] = True
        if edge_scale is not None:
            m_c = m_c * edge_scale(mid, d)
        if face_scale is not None:
            W_c = W_c * face_scale(cen, nrm)
        self.m_c, self.W_c = m_c, W_c
        free_c = (np.abs(m_c) > 0) & ~dead_c
        self.dead_c = ~free_c
        # ---------------------------------------------- tetrahedral part
        self.tet = tet_mesh
        if tet_mesh is not None:
            tm = tet_mesh
            box_surface_check(
                tm,
                blk.nodes[0][lo[0] : hi[0] + 1],
                blk.nodes[1][lo[1] : hi[1] + 1],
                blk.nodes[2][lo[2] : hi[2] + 1],
                allow_other=True,
            )
            eps_t, mu_t, sig_t, pec_t, disp = cell_materials(tm, materials)
            if np.any(sig_t != 0) or disp:
                raise NotImplementedError("lossy/dispersive tetrahedra: not in milestone 1")
            live = (~pec_t).astype(float)
            M1, M2 = whitney_mass_matrices(tm, eps_t, mu_t, live)
            self.M1, self.M2 = M1.tocsr(), M2.tocsr()
            nEt = tm.n_edges
            Xn = tm.nodes
            p, q = Xn[tm.edges[:, 0]], Xn[tm.edges[:, 1]]
            # edges on the box surface: edges of box-surface boundary faces
            lines = [blk.nodes[a][lo[a] : hi[a] + 1] for a in range(3)]
            bf_box = box_surface_check(tm, *lines, allow_other=True)
            on_box = np.zeros(nEt, bool)
            on_box[tm.face_edges[bf_box].ravel()] = True
            fixed = np.zeros(nEt, bool)
            fixed[tm.cell_edges[pec_t].ravel()] = True
            other_bf = np.setdiff1d(tm.boundary_faces, bf_box)
            fixed[tm.face_edges[other_bf].ravel()] = True  # holes are PEC
            touched = np.zeros(nEt, bool)
            touched[tm.cell_edges[~pec_t].ravel()] = True
            fixed |= ~touched
            if np.any(fixed & on_box):
                raise ValueError("PEC tetrahedra touch the box surface")
            self.tet_hanging = np.flatnonzero(on_box)
            self.tet_fixed = np.flatnonzero(fixed)
            free_t = ~on_box & ~fixed
            hr, hc, hv = box_trace(
                blk, lo, hi, p[on_box], q[on_box],
                mutate=mutate if mutate in ("nearest", "inject", "scale") else None,
            )  # fmt: skip
            hr = self.tet_hanging[hr]
            if mutate == "asym":
                ar, ac, av = box_trace(blk, lo, hi, p[on_box], q[on_box], mutate="inject")
                ar = self.tet_hanging[ar]
        else:
            nEt = 0
            free_t = np.zeros(0, bool)
            self.tet_hanging = self.tet_fixed = np.zeros(0, int)
            hr = hc = np.zeros(0, int)
            hv = np.zeros(0)
        self.nEc, self.nEt, self.nFc = nEc, nEt, nFc
        self.nFt = 0 if tet_mesh is None else tet_mesh.n_faces
        # -------------------------------------------------- prolongation
        free = np.r_[free_c, free_t]
        self.free_edges = np.flatnonzero(free)
        self.n_free = self.free_edges.size
        col_of = -np.ones(nEc + nEt, int)
        col_of[self.free_edges] = np.arange(self.n_free)

        def build_P(hr, hc, hv):
            keep = free_c[hc]  # parents on PEC edges contribute zero
            r_, c_, v_ = nEc + hr[keep], col_of[hc[keep]], hv[keep]
            return sp.csr_matrix(
                (
                    np.r_[np.ones(self.n_free), v_],
                    (np.r_[self.free_edges, r_], np.r_[np.arange(self.n_free), c_]),
                ),
                shape=(nEc + nEt, self.n_free),
            )

        self.P = build_P(hr, hc, hv)
        # -------------------------------------------------- operators
        Cc = blk.curl()
        cplx = np.iscomplexobj(m_c) or np.iscomplexobj(W_c)
        self.dtype = complex if cplx else float
        fc = np.flatnonzero(np.abs(W_c) > 0)
        if tet_mesh is not None:
            Ct = tet_mesh.C
            ft = np.flatnonzero(self.M2.diagonal() > 0)
            self.C_all = sp.block_diag([Cc, Ct], format="csr")
            Wt = self.M2[ft][:, ft]
            self.W = sp.block_diag([sp.diags(W_c[fc]), Wt], format="csr")
            self.M_all = sp.block_diag([sp.diags(m_c), self.M1], format="csr")
            self.faces = np.r_[fc, nFc + ft]
        else:
            self.C_all = Cc.tocsr()
            self.W = sp.diags(W_c[fc]).tocsr()
            self.M_all = sp.diags(m_c).tocsr()
            self.faces = fc
        self.n_faces = self.faces.size
        Cf = self.C_all[self.faces]
        self.Cf = Cf
        self.K = (Cf @ self.P).tocsr()
        if mutate == "asym":
            Pa = build_P(ar, ac, av)
            self.KT = (Cf @ Pa).T.tocsr()
        else:
            self.KT = self.K.T.tocsr()
        self.M = (self.P.T @ self.M_all @ self.P).tocsr()
        self._msolver = None
        self._lam = None
        if not cplx:
            if dt is None:
                dt = cfl * self.dt_max
            self.dt = float(dt)
        else:
            self.dt = None
        self.sources = []
        self.probes = []
        self.reset()

    # ------------------------------------------------------------ helpers
    @property
    def msolver(self):
        if self._msolver is None:
            self._msolver = MassSolver(self.M, method=self.mass_solver, tol=self.mass_tol)
        return self._msolver

    @property
    def lambda_max(self):
        """Largest eigenvalue of M^-1 K^T W K [1/s^2] (Lanczos)."""
        if self._lam is None:
            self._lam = _gen_lanczos_max(self.K, self.W, self.M, self.msolver)
        return self._lam

    @property
    def dt_max(self):
        """Leapfrog stability limit 2 / sqrt(lambda_max) [s]."""
        return 2.0 / np.sqrt(self.lambda_max)

    def block_lambda_bounds(self):
        """
        (lambda_cart, lambda_tet): lambda_max of each region alone, with all
        its interface edges free (natural boundary on the interface). The
        composite lambda_max is at most their maximum.
        """
        nEc = self.nEc
        live_c = np.flatnonzero(~self.dead_c)
        fc = self.faces[self.faces < self.nFc]
        Kc = self.C_all[fc][:, live_c].tocsr()
        Wc = sp.diags(self.W_c[fc])
        Mc = sp.diags(self.m_c[live_c])
        lam_c = _gen_lanczos_max(Kc, Wc, Mc, MassSolver(Mc))
        if self.tet is None:
            return lam_c, None
        live_t = np.setdiff1d(np.arange(self.nEt), self.tet_fixed)
        ft = self.faces[self.faces >= self.nFc]
        Kt = self.C_all[ft][:, nEc + live_t].tocsr()
        Wt = self.W[self.faces >= self.nFc][:, self.faces >= self.nFc]
        Mt = self.M1[live_t][:, live_t]
        lam_t = _gen_lanczos_max(Kt, Wt, Mt, MassSolver(Mt))
        return lam_c, lam_t

    def reset(self):
        self.e = np.zeros(self.n_free, self.dtype)
        self.b = np.zeros(self.n_faces, self.dtype)
        self.n = 0
        for p in self.probes:
            p["values"] = []

    def e_all(self, e=None):
        """Voltages on all edges, Cartesian then tetrahedral [V]."""
        return self.P @ (self.e if e is None else e)

    def energy(self, e_n, e_np1, b_half):
        """Discrete energy 1/2 b.W b + 1/2 e^n.M e^{n+1} [J]."""
        return float(0.5 * b_half @ (self.W @ b_half) + 0.5 * e_n @ (self.M @ e_np1))

    def divergence(self, b=None):
        """Net flux out of every active Cartesian cell and every tetrahedron [Wb]."""
        ball = np.zeros(self.C_all.shape[0], self.dtype)
        ball[self.faces] = self.b if b is None else b
        out = [self.block.divergence() @ ball[: self.nFc]]
        out[0] = out[0][self.active_cells]
        if self.tet is not None:
            out.append(self.tet.D @ ball[self.nFc :])
        return np.concatenate(out)

    def _interface_aggregation(self):
        """Rows: box-surface Cartesian squares; +1 on the square flux, -s on
        each tetrahedral triangle inside it (s = orientation vs. the square)."""
        if getattr(self, "_S", None) is not None:
            return self._S
        blk, tm = self.block, self.tet
        lo, hi = self.box_lo, self.box_hi
        X = tm.nodes
        bf = box_surface_check(
            tm, *[blk.nodes[a][lo[a] : hi[a] + 1] for a in range(3)], allow_other=True
        )
        cen = X[tm.faces[bf]].mean(1)
        pos = {f: i for i, f in enumerate(self.faces)}
        rows, cols, vals = [], [], []
        nrow = 0
        tol = 1e-9 * max(blk.nodes[a][hi[a]] - blk.nodes[a][lo[a]] for a in range(3))
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            for na in (lo[a], hi[a]):
                sel = np.flatnonzero(np.abs(cen[:, a] - blk.nodes[a][na]) < tol)
                ib = np.searchsorted(blk.nodes[b], cen[sel, b]) - 1
                ic = np.searchsorted(blk.nodes[c], cen[sel, c]) - 1
                ijk = [None] * 3
                ijk[a], ijk[b], ijk[c] = np.full(sel.size, na), ib, ic
                fq, _ = blk.face_index(a, *ijk)
                uq, inv = np.unique(fq, return_inverse=True)
                rows.append(nrow + np.arange(uq.size))
                cols.append(np.array([pos[f] for f in uq]))
                vals.append(np.ones(uq.size))
                ft = bf[sel]
                sgn = np.sign(tm.face_normal[ft, a])
                rows.append(nrow + inv)
                cols.append(np.array([pos[self.nFc + f] for f in ft]))
                vals.append(-sgn)
                nrow += uq.size
        self._S = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(nrow, self.n_faces),
        )
        return self._S

    def interface_flux_mismatch(self, b=None):
        """Flux of every box-surface Cartesian square minus the sum of the
        fluxes of its tetrahedral triangles [Wb]; constant in time."""
        return self._interface_aggregation() @ (self.b if b is None else b)

    # ------------------------------------------------------------ sources
    def cartesian_edge(self, point, component):
        """Index (into all edges) of the free Cartesian edge nearest `point`."""
        sel = np.flatnonzero((self.edge_dir == component) & ~self.dead_c)
        d = np.linalg.norm(self.edge_mid[sel] - np.asarray(point, float), axis=1)
        return int(sel[np.argmin(d)])

    def add_current(self, point, component, waveform, amplitude=1.0):
        """Soft current i(t) = amplitude waveform(t) [A] on the nearest
        Cartesian edge of direction `component` (0/1/2 = x/y/z)."""
        k = self.cartesian_edge(point, component)
        row = self.P[k]
        self.sources.append((row.indices.copy(), row.data.copy(), waveform, float(amplitude)))
        return k

    def add_probe(self, point, component):
        """Record E_component [V/m] on the nearest Cartesian edge every step."""
        k = self.cartesian_edge(point, component)
        self.probes.append({"row": self.P[k], "L": self.edge_len[k], "values": []})
        return len(self.probes) - 1

    def _source(self, t):
        s = np.zeros(self.n_free, self.dtype)
        for idx, w, f, amp in self.sources:
            s[idx] += amp * f(t) * w
        return s

    # ------------------------------------------------------------ stepping
    def one_step(self):
        dt = self.dt
        self.b -= dt * (self.K @ self.e)
        r = self.KT @ (self.W @ self.b)
        if self.sources:
            r -= self._source((self.n + 0.5) * dt)
        self.e += dt * self.msolver.solve(r)
        self.n += 1
        for p in self.probes:
            p["values"].append((p["row"] @ self.e).item() / p["L"])

    def run(self, n_steps, energy=False):
        """Advance `n_steps`; return the energy history if `energy`."""
        hist = []
        for _ in range(int(n_steps)):
            e0 = self.e.copy() if energy else None
            self.one_step()
            if energy:
                hist.append(self.energy(e0, self.e, self.b))
        return np.array(hist) if energy else None

    def linear_step(self, b, e, i_all=None):
        """One step as a linear map of (b^{n-1/2}, e^n, i^{n+1/2} on all edges)."""
        dt = self.dt
        b1 = b - dt * (self.K @ e)
        r = self.KT @ (self.W @ b1)
        if i_all is not None:
            r = r - self.P.T @ i_all
        return b1, e + dt * self.msolver.solve(r)

    def linear_step_T(self, bb1, eb1):
        """Exact transpose of `linear_step`: returns (bb, eb, ib_all)."""
        dt = self.dt
        g = self.msolver.solve(eb1, trans="T")
        bt = bb1 + dt * (self.W.T @ (self.KT.T @ g))
        eb = eb1 - dt * (self.K.T @ bt)
        ib = -dt * (self.P @ g)
        return bt, eb, ib

    # --------------------------------------------------- frequency domain
    def stiffness(self):
        """(K^T W K, M) of the semi-discrete problem K^T W K e = w^2 M e."""
        return (self.KT @ self.W @ self.K).tocsc(), self.M.tocsc()

    def solve_frequency(self, omega, rhs=None, fixed=None):
        """
        Solve ``(K^T W K - omega^2 M) e = rhs`` on the free edges.

        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        rhs : (n_free,) array, optional
            Right-hand side (for a current ``i`` with the exp(-i w t)
            convention: ``1j * omega * P.T @ i_all``).
        fixed : (idx, values), optional
            Prescribed voltages on non-free edges (indices into all edges,
            Cartesian then tetrahedral), e.g. minus the incident voltages on
            PEC edges in the scattered-field formulation.

        Returns
        -------
        e : (n_free,) complex
        """
        A, M = self.stiffness()
        A = (A - omega**2 * M).tocsc()
        b = np.zeros(self.n_free, complex) if rhs is None else np.asarray(rhs, complex)
        if fixed is not None:
            idx, val = np.asarray(fixed[0], int), np.asarray(fixed[1], complex)
            nall = self.nEc + self.nEt
            Q = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))),
                              shape=(nall, idx.size))  # fmt: skip
            Kd = self.Cf @ Q
            Afd = self.KT @ (self.W @ Kd) - omega**2 * (self.P.T @ (self.M_all @ Q))
            b = b - Afd @ val
        perm = _nested_dissection(A, self.free_points())
        lu = sla.splu(A[perm][:, perm].tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0,
                      options=dict(SymmetricMode=True))  # fmt: skip
        x = np.empty(self.n_free, complex)
        x[perm] = lu.solve(b[perm])
        return x

    def free_points(self):
        """Midpoints of the free edges [m] (Cartesian, then tetrahedral)."""
        mids = [self.edge_mid] + ([self.tet.edge_midpoint] if self.tet is not None else [])
        return np.vstack(mids)[self.free_edges]

    def eigenfrequencies(self, k, f_lo, f_hi):
        """
        The `k` lowest resonance frequencies [Hz] in ``[f_lo, f_hi]``.

        Shift-invert Lanczos at ``sigma = w_lo^2 / 2 + w_hi^2 / 2``: every
        eigenvalue in the band is closer to the shift than the static
        (gradient) null space at 0, so the null space is never returned.
        """
        A, M = self.stiffness()
        wl, wh = 2 * np.pi * f_lo, 2 * np.pi * f_hi
        sigma = 0.5 * (wl**2 + wh**2)
        kk = min(A.shape[0] - 2, k + 10)
        lam = sla.eigsh(A, k=kk, M=M, sigma=sigma, which="LM", tol=1e-11,
                        ncv=min(A.shape[0] - 1, max(4 * kk, 60)),
                        return_eigenvectors=False)  # fmt: skip
        lam = np.sort(lam[(lam > wl**2) & (lam < wh**2)])
        return (np.sqrt(lam) / (2 * np.pi))[:k]


def _nested_dissection(A, X, leaf=128):
    """
    Geometric nested-dissection ordering of the unknowns at points X.

    Recursive median bisection along the longest extent; the separator is
    the part of the left half coupled to the right half (graph of A + A^T)
    and is ordered last. For 3D meshes this cuts the LU fill several times
    below COLAMD (measured 10.6M vs 27.7M entries on a 21k-unknown
    staircase sphere; COLAMD exceeded 1.8 GB on a 33k hybrid case where
    this ordering needs 25.7M entries).
    """
    n = A.shape[0]
    G = (abs(A) + abs(A.T)).tocsr()
    G.data[:] = 1.0
    out = []
    stack = [np.arange(n)]
    # iterative post-order: children first, then separator
    todo = []
    while stack:
        idx = stack.pop()
        if len(idx) <= leaf:
            todo.append(("leaf", idx))
            continue
        P = X[idx]
        ax = int(np.argmax(P.max(0) - P.min(0)))
        med = np.median(P[:, ax])
        left, right = idx[P[:, ax] <= med], idx[P[:, ax] > med]
        if not len(left) or not len(right):
            todo.append(("leaf", idx))
            continue
        inR = np.zeros(n)
        inR[right] = 1.0
        touch = (G[left] @ inR) > 0
        todo.append(("sep", left[touch]))
        stack += [right, left[~touch]]
    for kind, idx in reversed(todo):
        out.append(idx)
    perm = np.concatenate(out)
    assert perm.size == n
    return perm


def _gen_lanczos_max(K, W, M, msolver):
    n = K.shape[1]
    KT = K.T.tocsr()
    A = sla.LinearOperator((n, n), matvec=lambda x: KT @ (W @ (K @ x)), dtype=float)
    Minv = sla.LinearOperator((n, n), matvec=msolver.solve, dtype=float)
    v0 = np.random.default_rng(1).standard_normal(n)
    lam = sla.eigsh(A, k=1, M=M, Minv=Minv, which="LA", tol=1e-10, v0=v0,
                    ncv=min(n, 40), return_eigenvectors=False)  # fmt: skip
    return float(lam[0])
