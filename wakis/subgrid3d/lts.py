# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Multi-level local time stepping (LTS) for `SubgridFIT3D`.

Second-order form of the subgrid leapfrog on the free edges:
``e^{n+1} - 2 e^n + e^{n-1} = -dt^2 A e^n`` with ``A = D^-1 S``,
``S = K^H W K``. Level ``l`` (0 = coarse root) holds the free edges of the
blocks of depth ``>= l`` (plus, optionally, the coarse parents of their
hanging edges), ``F_L c ... c F_1 c F_0 = all``. Level ``l`` takes ``p_l``
substeps of its parent level, ``tau_l = tau_{l-1} / p_l``, ``tau_0 = dt``.

Recursive leapfrog-Chebyshev LTS (Diaz & Grote 2009, 2015; damped
polynomial of Carle, Hochbruck & Sturm 2020, as `wakis.unstructured.lts`):

    B_L = A,   B_{l-1} = B_l r_l(X_l),   X_l = tau_{l-1}^2 P_l B_l,

with ``r_l`` the damped Chebyshev polynomial of degree ``p_l - 1``
(``r(0) = 1``) and ``P_l`` the 0/1 projector on ``F_l``. Then
``B_0 = A R_0`` with ``R_0 = r_L(X_L) ... r_1(X_1)`` (applied right to
left: ``r_1`` first) and the scheme is

    b^{n+1/2} = b^{n-1/2} - dt K R_0 e^n,
    e^{n+1}   = e^n + dt D^-1 (K^H W b^{n+1/2} - P_e^H i^{n+1/2}).

Properties (derived in the physics guide, "Local time stepping"):

* ``S_l := D B_l`` is Hermitian for every ``l`` (induction: ``S_{l-1} =
  sum_j c_j S_l (P_l D^-1 S_l)^j``), so the scheme is an ordinary leapfrog
  with the Hermitian stiffness ``S_0 = S R_0`` and conserves
  ``E = 1/2 [w^H (D - dt^2/4 S_0) w + ybar^H S_0 ybar]``
  (``w = (e^{n+1} - e^n)/dt``, ``ybar = (e^{n+1} + e^n)/2``); it is stable
  iff ``S_0 >= 0`` and ``rho = dt/2 sqrt(lambda_max(D^-1 S_0)) < 1``,
  measured by Lanczos (`stability_ratio`).
* Off ``F_1`` and its buffer ``R_0 e = e``; ``r(0) = 1`` makes the scheme
  consistent and second order.
* Linear, with the exact (bilinear) transpose
  ``R_0^T = r_1(X_1^T) ... r_L(X_L^T)``, ``X_l^T = tau^2 R_l^T A^T P_l``.

Cost per coarse step: the full leapfrog once, plus for level ``l`` the rows
of ``A`` on ``F_l`` (columns ``F_l`` and its one-face buffer ``C_l``)
``prod_{m<=l} p_m - prod_{m<l} p_m`` times: level ``l`` edges are updated
``p_1 ... p_l`` times in total, exactly the multi-level count.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from ..unstructured.lts import chebyshev_coefficients, stability_polynomial

__all__ = ["SubgridLTS", "stability_ratio", "lts_dt_max", "level_margins", "is_stable",
           "positivity_limit", "stability_polynomial"]


class _Level:
    pass


class SubgridLTS:
    """
    Recursive LTS operator ``e -> R_0 e`` of a `SubgridFIT3D`.

    Parameters
    ----------
    solver : SubgridFIT3D
    dt : float
        Coarse (level-0) time step [s].
    nu : float or sequence of float
        Chebyshev damping (0: Diaz-Grote polynomial, marginal at mid
        spectrum; > 0: modes inside ``F_l`` keep a margin). A scalar gives
        ``nu_l = l * nu``: nested levels need strictly increasing damping,
        since the top of level ``l+1``'s spectrum, seen at the step of level
        ``l``, lands exactly on the positivity limit of ``r_l`` when
        ``nu_{l+1} = nu_l`` (physics guide, "Nested levels").
    ratios : sequence of int, optional
        Substeps ``p_l`` per level ``l = 1..L``; default: the (largest)
        refinement ratio of the patches at that level.
    interface_parents : bool
        Put the coarse free edges that carry hanging edges of a level-``l``
        block into ``F_l`` (their effective stiffness sees the fine faces).
    grow : int
        Extra layers of edges (face neighbours) added to every ``F_l``.
    edge_level : ndarray of int, optional
        Explicit level per free edge (overrides the automatic partition).
    mutate : {None, 'consistency'}
        Verification only: ``'consistency'`` scales every ``r_l`` by 1.05
        (``r(0) != 1``: symmetric and stable, but inconsistent).
    """

    def __init__(self, solver, dt, nu=0.05, ratios=None, interface_parents=True,
                 grow=0, edge_level=None, mutate=None):  # fmt: skip
        s = solver
        M = s.mesh
        self.s = s
        self.dt = float(dt)
        self.nu = nu
        n = s.n_free
        self.n = n
        blk_level = np.asarray(M.level)
        L = int(blk_level.max())
        if ratios is None:
            ratios = []
            for lev in range(1, L + 1):
                rr = [M.patches[b - 1].ratio for b in range(1, M.n_blocks) if blk_level[b] == lev]
                ratios.append(max(rr))
        ratios = [int(p) for p in ratios]
        if len(ratios) != L:
            raise ValueError(f"need {L} ratios, got {len(ratios)}")
        self.ratios = ratios
        self.L = L
        if np.ndim(nu) == 0:
            nus = [float(nu) * lv for lv in range(1, L + 1)]
        else:
            nus = [float(v) for v in nu]
            if len(nus) != L:
                raise ValueError(f"need {L} damping values, got {len(nus)}")
        self.nus = nus
        K = s.K.tocsr()
        Kc = K.tocsc()
        # face-edge adjacency on free edges (for buffers and growth)
        G = (abs(K) > 0).astype(np.float64)
        GtG = (G.T @ G).tocsr()  # edges sharing a face
        if edge_level is None:
            lev = blk_level[s.edge_block[s.free_edges]].astype(int)
            if interface_parents and L > 0:
                Pc = s.P.tocsc()
                hang_rows = np.flatnonzero(s.hanging)
                Ph = s.P.tocsr()[hang_rows]
                child_lev = blk_level[s.edge_block[hang_rows]]
                Ph = Ph.tocoo()
                np.maximum.at(lev, Ph.col, child_lev[Ph.row])
                del Pc
            for _ in range(int(grow)):
                for lv in range(L, 0, -1):
                    m = (lev >= lv).astype(float)
                    nb = (GtG @ m) > 0
                    lev = np.where(nb & (lev < lv), lv, lev)
        else:
            lev = np.asarray(edge_level, int)
        # nesting: the buffer of F_l must lie in F_{l-1}
        for lv in range(L, 1, -1):
            for _ in range(8):
                m = (lev >= lv).astype(float)
                buf = ((GtG @ m) > 0) & (lev < lv - 1)
                if not buf.any():
                    break
                lev = np.where(buf, lv - 1, lev)
        self.edge_level = lev
        tau = [self.dt]
        for p in ratios:
            tau.append(tau[-1] / p)
        self.tau = tau
        self.levels = [None]
        Fprev = np.arange(n)
        W = s.W
        Dinv = s.Dinv
        for lv in range(1, L + 1):
            d = _Level()
            d.p = ratios[lv - 1]
            d.tau2 = tau[lv - 1] ** 2  # X_l = tau_{l-1}^2 P_l B_l
            F = np.flatnonzero(lev >= lv)
            d.F = F
            if F.size == 0:
                d.Phi = np.zeros(0, int)
                d.C = np.zeros(0, int)
            else:
                KF = Kc[:, F]
                Phi = np.unique(KF.tocoo().row)
                Kphi = K[Phi]
                cols = np.unique(Kphi.indices)
                isF = np.zeros(n, bool)
                isF[F] = True
                C = cols[~isF[cols]]
                d.Phi, d.C = Phi, C
                Kphi = Kphi.tocsc()
                d.K_F = Kphi[:, F].tocsr()
                d.K_C = Kphi[:, C].tocsr()
                d.KH_F = d.K_F.conj().T.tocsr()
                d.W = W[Phi]
                d.Dinv = Dinv[F]
            pos = -np.ones(n, int)
            pos[Fprev] = np.arange(Fprev.size)
            d.iF = pos[d.F]
            d.iC = pos[d.C]
            if np.any(d.iF < 0) or np.any(d.iC < 0):
                raise RuntimeError("LTS levels not nested (internal error)")
            d.nu = nus[lv - 1]
            (d.delta, d.beta, d.T, d.kappa, d.out) = chebyshev_coefficients(d.p, d.nu)
            if mutate == "consistency":
                d.out = 1.05 * d.out
            self.levels.append(d)
            Fprev = F
        self.dtype = s.dtype
        self._xp = np

    # ------------------------------------------------------------ fast form
    def apply(self, e):
        """``R_0 e`` (works on F_l and its buffer only)."""
        if self.L == 0:
            return e
        return self._R(0, e)

    def _R(self, lv, v):
        """``R_lv v`` restricted to F_lv (lv = 0: all free edges)."""
        if lv >= self.L:
            return v
        d = self.levels[lv + 1]
        if d.F.size == 0:
            return v
        u = v.copy()
        uF = v[d.iF] + self._corr(lv + 1, v[d.iF], v[d.iC])
        u[d.iF] = self._R(lv + 1, uF)
        return u

    def _corr(self, m, yF, yC):
        """``(r_m(X_m) y - y)`` on F_m, from y on F_m and its buffer C_m."""
        d = self.levels[m]
        xp = self._xp
        if d.p == 1:
            return xp.zeros_like(yF)
        g = d.K_C @ yC
        d_old, dk = xp.zeros_like(yF), yF
        for k in range(1, d.p):
            z = self._R(m, dk)
            Xd = d.tau2 * d.Dinv * (d.KH_F @ (d.W * (d.K_F @ z + d.kappa[k] * g)))
            d_new = 2 * d.delta * dk - 2 * d.beta * Xd + 2 * d.T[k] * yF - d_old
            d_old, dk = dk, d_new
        return d.out * dk - yF

    def to_backend(self, backend):
        """Move the level operators to ``'cupy'`` (GPU) or back to ``'numpy'``."""
        from .gpu import get_xp

        xp, xsp = get_xp(backend)
        self._xp = xp
        for d in self.levels[1:]:
            if d.F.size == 0:
                continue
            for name in ("K_F", "K_C", "KH_F"):
                m = getattr(d, name)
                if hasattr(m, "get"):
                    m = m.get()
                setattr(d, name, xsp.csr_matrix(m) if backend in ("cupy", "gpu") else m)
            for name in ("W", "Dinv"):
                v = getattr(d, name)
                v = v.get() if hasattr(v, "get") else v
                setattr(d, name, xp.asarray(v))
            for name in ("iF", "iC"):
                v = getattr(d, name)
                v = v.get() if hasattr(v, "get") else v
                setattr(d, name, xp.asarray(v))
        return self

    # ------------------------------------------- full-length reference form
    def _poly(self, y, X, d):
        d_old, dk = np.zeros_like(y), y.copy()
        for k in range(1, d.p):
            d_new = 2 * d.delta * dk - 2 * d.beta * X(dk) + 2 * d.T[k] * y - d_old
            d_old, dk = dk, d_new
        return d.out * dk

    def _A(self, v):
        s = self.s
        return s.Dinv * (s.KH @ (s.W * (s.K @ v)))

    def _AT(self, v):
        s = self.s
        return s.K.T @ (s.W * (s.KH.T @ (s.Dinv * v)))

    def _mask(self, m, v):
        out = np.zeros_like(v)
        F = self.levels[m].F
        out[F] = v[F]
        return out

    def R_full(self, lv, v):
        """``R_lv v`` with full-length vectors (reference for `apply`)."""
        if lv >= self.L:
            return v
        m = lv + 1
        d = self.levels[m]
        u = self._poly(v, lambda w: d.tau2 * self._mask(m, self._A(self.R_full(m, w))), d)
        return self.R_full(m, u)

    def apply_full(self, e):
        return self.R_full(0, np.asarray(e, dtype=np.result_type(e, self.dtype)))

    def RT_full(self, lv, v):
        """``R_lv^T v`` (bilinear transpose)."""
        if lv >= self.L:
            return v
        m = lv + 1
        d = self.levels[m]
        u = self.RT_full(m, v)
        return self._poly(u, lambda w: d.tau2 * self.RT_full(m, self._AT(self._mask(m, w))), d)

    def apply_T(self, z):
        """``R_0^T z`` (exact transpose of `apply`, for adjoints)."""
        return self.RT_full(0, np.asarray(z, dtype=np.result_type(z, self.dtype)))

    # ------------------------------------------------------------- matrices
    def matrix(self):
        """Sparse ``R_0`` (small meshes: eigen-analysis and tests)."""
        s = self.s
        n = self.n
        Dinv = sp.diags(s.Dinv)
        S = (s.KH @ sp.diags(s.W) @ s.K).tocsr()
        eye = sp.identity(n, format="csr", dtype=self.dtype)

        def Rmat(lv):
            if lv >= self.L:
                return eye
            m = lv + 1
            d = self.levels[m]
            Rm = Rmat(m)
            Pm = np.zeros(n)
            Pm[d.F] = 1.0
            X = (d.tau2 * sp.diags(Pm) @ Dinv @ S @ Rm).tocsr()
            d_old, dk = sp.csr_matrix((n, n), dtype=self.dtype), eye
            for k in range(1, d.p):
                d_new = 2 * d.delta * dk - 2 * d.beta * (X @ dk) + 2 * d.T[k] * eye - d_old
                d_old, dk = dk, d_new
            return (Rm @ (d.out * dk)).tocsr()

        return Rmat(0)

    def stiffness(self):
        """``(S_0, D)``: Hermitian LTS stiffness ``K^H W K R_0`` and the mass."""
        s = self.s
        S0 = (s.KH @ sp.diags(s.W) @ s.K @ self.matrix()).tocsr()
        return S0, s.D

    # ---------------------------------------------------------------- info
    def work_per_step(self):
        """
        Edge and face updates per coarse step: the leapfrog once plus every
        product with the level rows (faces Phi_l, edges F_l).
        """
        s = self.s
        full = s.n_free + s.n_faces
        extra = 0
        mult = 1  # number of r_l evaluations per coarse step
        for lv in range(1, self.L + 1):
            d = self.levels[lv]
            # r_l is evaluated once per evaluation of R_{l-1} ... counted
            # recursively: calls(l) = calls(l-1) * p_{l-1}, calls(1) = 1
            extra += mult * (d.p - 1) * (d.F.size + d.Phi.size)
            mult *= d.p
        return full + extra

    def summary(self):
        out = {"L": self.L, "ratios": self.ratios, "nu": self.nus, "dt": self.dt,
               "n_free": int(self.n), "work_per_step": int(self.work_per_step())}  # fmt: skip
        for lv in range(1, self.L + 1):
            d = self.levels[lv]
            out[f"F{lv}"] = int(d.F.size)
            out[f"C{lv}"] = int(d.C.size)
            out[f"Phi{lv}"] = int(d.Phi.size)
        return out


def stability_ratio(lts, tol=1e-10, which="LA", return_mode=False):
    """
    ``rho = dt/2 sqrt(lambda_max(D^-1 S_0))`` by Lanczos on the Hermitian
    ``D^-1/2 S_0 D^-1/2`` (``which='SA'``: the smallest eigenvalue of that
    operator instead, in 1/s^2, to check ``S_0 >= 0``).
    """
    s = lts.s
    sc = 1.0 / np.sqrt(s.D)
    n = s.n_free

    def mv(x):
        x = np.asarray(x).ravel()
        return sc * (s.KH @ (s.W * (s.K @ lts.apply(sc * x))))

    op = sla.LinearOperator((n, n), matvec=mv, dtype=s.dtype)
    v0 = np.random.default_rng(3).standard_normal(n).astype(s.dtype)
    lam, vec = sla.eigsh(op, k=1, which=which, tol=tol, v0=v0, ncv=min(n, 40))
    lam = float(np.real(lam[0]))
    if which != "LA":
        return lam
    rho = 0.5 * lts.dt * np.sqrt(max(lam, 0.0))
    return (rho, np.abs(vec[:, 0]) ** 2) if return_mode else rho


def positivity_limit(p, nu):
    """
    Largest ``x`` with ``r(x') >= 0`` on ``[0, x]``: ``2 delta / beta`` for
    even ``p`` (where ``T_p(delta - beta x) = T_p(-delta) = T_p(delta)``),
    infinite for odd ``p`` (``T_p(s) < T_p(delta)`` for every ``s < delta``).
    """
    if p == 1:
        return np.inf
    delta, beta, _, _, _ = chebyshev_coefficients(p, nu)
    return 2 * delta / beta if p % 2 == 0 else np.inf


def level_margins(lts, tol=1e-8):
    """
    Positivity margins ``tau_{l-1}^2 mu_l / x_+(p_l, nu)`` per level ``l``.

    ``S_{l-1} = S_l^{1/2} r_l(Y_l) S_l^{1/2}`` (``D``-scaled) with ``Y_l``
    having the nonzero spectrum of ``tau_{l-1}^2 P_l D^-1 S_l P_l``, whose
    largest eigenvalue is ``tau_{l-1}^2 mu_l`` (``mu_l`` by Lanczos on the
    rows/columns ``F_l`` of ``D^-1/2 S_l D^-1/2``). So ``S_0 >= 0`` if every
    margin is ``<= 1`` (``S_L = S >= 0``, induction).
    """
    out = []
    for lv in range(1, lts.L + 1):
        d = lts.levels[lv]
        if d.F.size == 0:
            out.append(0.0)
            continue
        xp_ = positivity_limit(d.p, d.nu)
        if not np.isfinite(xp_):
            out.append(0.0)
            continue
        sc = 1.0 / np.sqrt(1.0 / d.Dinv)
        nF = d.F.size

        def mv(x, d=d, sc=sc, lv=lv):
            x = np.asarray(x).ravel()
            z = lts._R(lv, sc * x)
            return sc * (d.KH_F @ (d.W * (d.K_F @ z)))

        op = sla.LinearOperator((nF, nF), matvec=mv, dtype=lts.dtype)
        v0 = np.random.default_rng(5).standard_normal(nF).astype(lts.dtype)
        mu = sla.eigsh(op, k=1, which="LA", tol=tol, v0=v0, ncv=min(nF, 40),
                       return_eigenvectors=False)  # fmt: skip
        out.append(float(d.tau2 * np.real(mu[0]) / xp_))
    return out


def is_stable(lts):
    """``(stable, rho, margins)``: ``rho < 1`` and every margin ``<= 1``."""
    marg = level_margins(lts)
    rho = stability_ratio(lts)
    return (rho < 1.0 and max(marg, default=0.0) <= 1.0), rho, marg


def lts_dt_max(solver, dt_hi, dt_lo=None, rtol=2e-3, **kw):
    """
    Largest coarse step with ``rho < 1`` and ``S_0 >= 0`` (bisection on the
    Lanczos ratio and the level positivity margins, `is_stable`).

    Parameters
    ----------
    solver : SubgridFIT3D
    dt_hi : float
        Upper bracket [s] (must be unstable or is returned).
    dt_lo : float, optional
        Lower bracket [s] (default: the global ``solver.dt_max``).
    rtol : float
        Relative bracket width at exit.
    kw : passed to `SubgridLTS`.

    Returns
    -------
    dt_max : float (the stable end of the final bracket)
    hist : list of (dt, rho, max margin)
    """
    lo = solver.dt_max if dt_lo is None else float(dt_lo)
    hi = float(dt_hi)
    hist = []
    ok, r, m = is_stable(SubgridLTS(solver, hi, **kw))
    hist.append((hi, r, max(m, default=0.0)))
    if ok:
        return hi, hist
    while hi / lo - 1 > rtol:
        mid = np.sqrt(lo * hi)
        ok, r, m = is_stable(SubgridLTS(solver, mid, **kw))
        hist.append((mid, r, max(m, default=0.0)))
        if ok:
            lo = mid
        else:
            hi = mid
    return lo, hist
