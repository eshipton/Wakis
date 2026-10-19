# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Local time stepping (LTS) for the leapfrog of the unstructured FIT solvers.

Both 2D polarizations reduce, after eliminating the edge fluxes, to the
second-order system ``M y'' = -K y`` (``y`` = Ez or Hz on the free nodes,
``M`` the diagonal node mass, ``K = G^T diag(stiff) G``). The global leapfrog
is ``y^{n+1} - 2 y^n + y^{n-1} = -dt^2 A y^n`` with ``A = M^-1 K`` and is
stable for ``dt^2 lambda_max(A) < 4``: the smallest element sets ``dt``
everywhere.

Two-level LTS (Diaz & Grote 2009; stabilized as proposed by Grote, Michel &
Sauter 2021 and Carle, Hochbruck & Sturm 2020). Let ``P`` be the diagonal 0/1
projector on the *fine* nodes F (those whose local CFL limit is below the
coarse step ``dt``) and ``X = dt^2 P A``. The scheme is

    y^{n+1} - 2 y^n + y^{n-1} = -dt^2 A r(X) y^n,                       (1)

    r(x) = 2 (1 - T_p(delta - beta x) / T_p(delta)) / x,
    delta = 1 + nu / p^2,  beta = T_p(delta) / (2 T_p'(delta)),

a polynomial of degree p - 1 with r(0) = 1. ``r(X) y`` equals ``y`` off F and
costs p - 1 products with the rows of A on F (the fine nodes and their
first neighbours: the "buffer" is implicit). For nu = 0, (1) is exactly the
Diaz-Grote LTS (p leapfrog substeps of dt/p on F with the coarse force frozen
at t^n); nu > 0 damps the Chebyshev polynomial so that ``x r(x)`` stays
below ``2 + 2 / T_p(delta) < 4`` instead of touching 4 at interior points.

Properties (proved in the physics guide, "Local time stepping"):

* ``M A r(X) = K r(X)`` is symmetric, so (1) conserves the discrete energy
  ``E = 1/2 [w^T (M - dt^2/4 K_p) w + ybar^T K_p ybar]`` with
  ``w = (y^{n+1} - y^n)/dt``, ``ybar = (y^{n+1} + y^n)/2``, ``K_p = K r(X)``;
  it is stable iff ``rho = dt/2 sqrt(lambda_max(M^-1 K_p)) < 1``, which
  `stability_ratio` evaluates (Lanczos) at setup.
* Consistency: r(0) = 1 and r is smooth, so (1) is second order.
* Linear, with the exact transpose ``r(X)^T = r(X^T)`` (`apply_T`).

In the first-order (node field, edge flux) form used by the solvers, (1) is
the ordinary leapfrog with the flux update fed by ``r(X) y`` instead of
``y``: ``v -= dt G r(X) y`` (TM), ``v += dt G r(X) y`` (TE).
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh


def chebyshev_coefficients(p, nu):
    """
    Scalars of the stabilized Chebyshev recurrence.

    Returns
    -------
    delta, beta : float
        Shift and scale of the argument ``delta - beta x``.
    T : ndarray (p + 1,)
        ``T_k(delta)``.
    kappa : ndarray (p + 1,)
        ``D_k(delta) = T_k'(delta)``, the value of the divided-difference
        recurrence on nodes outside F.
    out : float
        ``2 beta / T_p(delta)``, so that ``r(X) y = out * d_p``.
    """
    delta = 1.0 + nu / p**2
    T = np.zeros(p + 1)
    T[0] = 1.0
    if p >= 1:
        T[1] = delta
    for k in range(1, p):
        T[k + 1] = 2 * delta * T[k] - T[k - 1]
    kappa = np.zeros(p + 1)
    if p >= 1:
        kappa[1] = 1.0
    for k in range(1, p):
        kappa[k + 1] = 2 * delta * kappa[k] + 2 * T[k] - kappa[k - 1]
    beta = T[p] / (2 * kappa[p])  # kappa_p = T_p'(delta)
    return delta, beta, T, kappa, 2 * beta / T[p]


def stability_polynomial(x, p, nu):
    """``x r(x)``: dt^2 times the LTS eigenvalue of a mode inside F (scalar)."""
    delta, beta, _, _, _ = chebyshev_coefficients(p, nu)
    Tp = np.polynomial.chebyshev.chebval(delta - beta * np.asarray(x), [0] * p + [1])
    return 2 * (1 - Tp / np.polynomial.chebyshev.chebval(delta, [0] * p + [1]))


def gershgorin_rate(K, mass):
    """
    Node-wise upper bound of the local leapfrog rate, ``sum_m |K_nm| / M_n``.

    ``max_n`` of it bounds ``lambda_max(M^-1 K)`` (row-sum norm); node n can
    safely take a step ``dt`` if ``dt^2 * rate_n <= 4`` in isolation.
    """
    return np.asarray(abs(K).sum(axis=1)).ravel() / mass


class LocalTimeStepping:
    """
    Stabilized two-level leapfrog LTS operator ``y -> r(X) y``, X = dt^2 P M^-1 K.

    Parameters
    ----------
    K : sparse (n, n)
        Symmetric stiffness on the free nodes.
    mass : ndarray (n,)
        Positive diagonal mass.
    fine : ndarray of bool (n,)
        Fine nodes F (take p substeps).
    p : int
        Step ratio (coarse dt / fine substep).
    nu : float
        Chebyshev damping (0: Diaz-Grote LTS).
    dt : float
        Coarse timestep [s].
    """

    def __init__(self, K, mass, fine, p, nu, dt):
        self.p, self.nu, self.dt = int(p), float(nu), float(dt)
        self.n = K.shape[0]
        self.F = np.flatnonzero(fine)
        K = sp.csr_matrix(K)
        A_F = (sp.diags(1 / mass[self.F]) @ K[self.F]).tocsr()
        cols = np.unique(A_F.indices)  # column indices (CSR)
        A_F = A_F.tocsc()
        isF = np.zeros(self.n, bool)
        isF[self.F] = True
        self.C = cols[~isF[cols]]  # coarse nodes coupled to F (buffer)
        self.A_F = A_F.tocsr()  # rows F, all columns
        self.A_FF = A_F[:, self.F].tocsr()
        self.A_FC = A_F[:, self.C].tocsr()
        self.A_F_T = self.A_F.T.tocsr()
        (self.delta, self.beta, self.T, self.kappa, self.out) = chebyshev_coefficients(
            self.p, self.nu
        )

    # ---------------------------------------------------------------- forward
    def correction(self, y):
        """``(r(X) y - y)[F]``: the only entries where r(X) y differs from y."""
        p, dt2 = self.p, self.dt**2
        yF = y[self.F]
        if p == 1 or self.F.size == 0:
            return np.zeros(self.F.size)
        g = self.A_FC @ y[self.C]
        d_old, d = np.zeros_like(yF), yF
        for k in range(1, p):
            Xd = dt2 * (self.A_FF @ d + self.kappa[k] * g)
            d_new = 2 * self.delta * d - 2 * self.beta * Xd + 2 * self.T[k] * yF - d_old
            d_old, d = d, d_new
        return self.out * d - yF

    def apply(self, y):
        """``r(X) y`` (efficient: work on F and its buffer only)."""
        out = np.array(y, dtype=float, copy=True)
        out[self.F] += self.correction(y)
        return out

    # ------------------------------------------------- generic (verification)
    def _poly(self, y, X):
        d_old, d = np.zeros_like(y), np.array(y, dtype=float)
        for k in range(1, self.p):
            d_new = (
                2 * self.delta * d - 2 * self.beta * X(d) + 2 * self.T[k] * y - d_old
            )
            d_old, d = d, d_new
        return self.out * d

    def X(self, z):
        """``X z = dt^2 P M^-1 K z`` (full vectors)."""
        out = np.zeros(self.n)
        out[self.F] = self.dt**2 * (self.A_F @ z)
        return out

    def X_T(self, z):
        """``X^T z = dt^2 K M^-1 P z``."""
        return self.dt**2 * (self.A_F_T @ z[self.F])

    def apply_full(self, y):
        """``r(X) y`` with full-length vectors (reference for `apply`)."""
        return self._poly(np.asarray(y, float), self.X)

    def apply_T(self, z):
        """``r(X)^T z = r(X^T) z`` (exact transpose, for adjoints)."""
        return self._poly(np.asarray(z, float), self.X_T)

    @property
    def fine_fraction(self):
        return self.F.size / self.n

    @property
    def buffer_size(self):
        return self.C.size


def stability_ratio(K, mass, lts, tol=1e-8):
    """
    ``rho = dt/2 sqrt(lambda_max(M^-1 K r(X)))``; the LTS is stable iff rho < 1.

    Lanczos on the symmetric ``M^-1/2 K r(X) M^-1/2`` (``K r(X)`` is symmetric).
    """
    s = 1 / np.sqrt(mass)
    n = K.shape[0]

    def mv(z):
        z = np.asarray(z).ravel()
        return s * (K @ lts.apply(s * z))

    op = LinearOperator((n, n), matvec=mv, dtype=float)
    lam = eigsh(op, k=1, which="LA", return_eigenvectors=False, tol=tol)[0]
    return 0.5 * lts.dt * np.sqrt(max(lam, 0.0))


def build_lts(K, mass, dt_coarse, p=2, nu=0.15, theta=1.0, rho_max=0.98, lam=None,
              max_iter=12, fine=None):  # fmt: skip
    """
    Partition the nodes and build a stable two-level LTS.

    Node n is fine if its normalized Gershgorin rate exceeds ``theta (2 /
    dt_coarse)^2`` (the rates are scaled so that their maximum equals
    ``lam = lambda_max(M^-1 K)``). The stability ratio of the resulting
    operator is then computed by Lanczos; while it exceeds `rho_max`, theta
    is lowered by 15 % (F grows).

    Parameters
    ----------
    K, mass : stiffness and node mass on the free nodes.
    dt_coarse : float
        Coarse timestep [s].
    p : int
        Steps of F per coarse step.
    nu : float
        Chebyshev damping (see module docstring).
    theta : float
        Initial partition threshold (1: node-local CFL limit).
    rho_max : float
        Required stability ratio.
    lam : float, optional
        lambda_max(M^-1 K), for the normalization (Lanczos if None).
    fine : ndarray of bool, optional
        Use this partition instead (no search); rho is still computed.

    Returns
    -------
    lts : LocalTimeStepping
    info : dict
        ``rho``, ``theta``, ``fine_fraction``, ``buffer``, ``history``.
    """
    rate = gershgorin_rate(K, mass)
    if lam is None:
        from ._core import max_eigenvalue

        lam = max_eigenvalue(K, M_inv_sqrt=1 / np.sqrt(mass))
    rate = rate * lam / rate.max()
    history = []
    for _ in range(max_iter):
        mask = fine if fine is not None else rate * dt_coarse**2 > 4 * theta
        lts = LocalTimeStepping(K, mass, mask, p, nu, dt_coarse)
        rho = stability_ratio(K, mass, lts)
        history.append((theta, lts.fine_fraction, rho))
        if rho <= rho_max or fine is not None:
            break
        theta *= 0.85
    info = dict(rho=rho, theta=theta, fine_fraction=lts.fine_fraction,
                buffer=lts.buffer_size, history=history)  # fmt: skip
    if rho > rho_max and fine is None:
        raise ValueError(
            f"LTS not stable: rho = {rho:.3f} > {rho_max} after {max_iter} "
            "partitions; lower dt or raise p"
        )
    return lts, info


def lts_matrix(lts):
    """Sparse ``r(X)`` (for eigen-analysis; the solvers use `apply`)."""
    n = lts.n
    X = sp.csr_matrix((lts.dt**2 * lts.A_F.data, lts.A_F.indices, lts.A_F.indptr),
                      shape=(lts.F.size, n))  # fmt: skip
    X = sp.csr_matrix(sp.eye(n, format="csr")[:, lts.F] @ X)  # rows embedded
    eye = sp.eye(n, format="csr")
    d_old, d = sp.csr_matrix((n, n)), eye
    for k in range(1, lts.p):
        d_new = 2 * lts.delta * d - 2 * lts.beta * (X @ d) + 2 * lts.T[k] * eye - d_old
        d_old, d = d, d_new
    return (lts.out * d).tocsr()


# -------------------------------------------------------------------- analysis
def _r(solver, y):
    return y if solver.lts is None else solver.lts.apply(y)


def _r_T(solver, y):
    return y if solver.lts is None else solver.lts.apply_T(y)


def modified_energy(solver, y0, y1):
    """
    Exactly conserved quadratic form of the (LTS) leapfrog, source-free.

    ``E = 1/2 [w^T M w - dt^2/4 w^T K_p w + ybar^T K_p ybar]`` with
    ``w = (y1 - y0)/dt``, ``ybar = (y1 + y0)/2``, ``K_p = K r(X)``; y0, y1 are
    consecutive values of the node unknown (Ez^n, Ez^{n+1} in TM; Hz^{n-1/2},
    Hz^{n+1/2} in TE). It is the energy of the time-differentiated field
    [J/m/s^2]; positive definite iff rho < 1.
    """
    dt = solver.dt
    w, yb = (y1 - y0) / dt, 0.5 * (y1 + y0)

    def Kp(z):
        return solver.K @ _r(solver, z)

    return 0.5 * (w @ (solver.mass * w) - dt**2 / 4 * w @ Kp(w) + yb @ Kp(yb))


def _coeffs(solver):
    dt = solver.dt
    a = solver.mass - solver.loss * dt / 2
    b = solver.mass + solver.loss * dt / 2
    ce = 0.5 * dt * solver.edge_rate
    return dt, a, b, (1 - ce) / (1 + ce), dt / (1 + ce)


def linear_step(solver, u, v):
    """
    Source-free, non-dispersive step ``(u, v) -> (u', v')`` of
    `UnstructuredFIT2D` as a pure function (same arithmetic as `one_step`).
    """
    dt, a, b, al, ga = _coeffs(solver)
    G, S = solver.G, solver.stiff
    if solver.pol == "TM":
        v1 = al * v - ga * (G @ _r(solver, u))
        u1 = (a * u + dt * (G.T @ (S * v1))) / b
    else:
        u1 = (a * u - dt * (G.T @ (S * v))) / b
        v1 = al * v + ga * (G @ _r(solver, u1))
    return u1, v1


def linear_step_T(solver, ub1, vb1):
    """Exact transpose of `linear_step` (reverse-mode adjoint of one step)."""
    dt, a, b, al, ga = _coeffs(solver)
    G, S = solver.G, solver.stiff
    if solver.pol == "TM":
        vbt = vb1 + dt * S * (G @ (ub1 / b))
        ub = a / b * ub1 - _r_T(solver, G.T @ (ga * vbt))
        vb = al * vbt
    else:
        ubt = ub1 + _r_T(solver, G.T @ (ga * vb1))
        ub = a / b * ubt
        vb = al * vb1 - dt * S * (G @ (ubt / b))
    return ub, vb
