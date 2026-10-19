# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Reference-triangle machinery for nodal discontinuous Galerkin methods.

Written from the published mathematics (orthonormal Dubiner/Koornwinder
basis, Warburton's warp-and-blend interpolation nodes, collapsed-coordinate
Gauss cubature); see Hesthaven & Warburton, *Nodal Discontinuous Galerkin
Methods* (Springer 2008, DOI 10.1007/978-0-387-72067-8) and Warburton,
J. Eng. Math. 56, 247-262 (2006).

Reference triangle: vertices v1 = (-1, -1), v2 = (1, -1), v3 = (-1, 1),
counter-clockwise. Faces: f1 = v1-v2 (s = -1), f2 = v2-v3 (r + s = 0),
f3 = v3-v1 (r = -1); each face is parametrized by t in [-1, 1] along the
counter-clockwise traversal.
"""

from functools import lru_cache

import numpy as np
from scipy.special import eval_jacobi, gammaln, roots_jacobi

__all__ = ["ReferenceTriangle", "jacobi_normalized", "gll_nodes"]


def jacobi_normalized(x, n, alpha, beta):
    """Jacobi polynomial P_n^(alpha, beta)(x), orthonormal on [-1, 1]
    with weight (1 - x)^alpha (1 + x)^beta."""
    x = np.asarray(x, float)
    if n < 0:
        return np.zeros_like(x)
    lg = (
        (alpha + beta + 1) * np.log(2)
        - np.log(2 * n + alpha + beta + 1)
        + gammaln(n + alpha + 1)
        + gammaln(n + beta + 1)
        - gammaln(n + alpha + beta + 1)
        - gammaln(n + 1)
    )
    return eval_jacobi(n, alpha, beta, x) / np.exp(0.5 * lg)


def _djacobi_normalized(x, n, alpha, beta):
    """Derivative of `jacobi_normalized`."""
    if n == 0:
        return np.zeros_like(np.asarray(x, float))
    # d/dx P_n^(a,b) = (n + a + b + 1) / 2 P_{n-1}^(a+1,b+1) (unnormalized);
    # with orthonormal polynomials the factor is sqrt(n (n + a + b + 1)).
    return np.sqrt(n * (n + alpha + beta + 1)) * jacobi_normalized(
        x, n - 1, alpha + 1, beta + 1
    )


def gll_nodes(n):
    """Gauss-Lobatto-Legendre nodes (n + 1 points) on [-1, 1]."""
    if n == 1:
        return np.array([-1.0, 1.0])
    inner = roots_jacobi(n - 1, 1, 1)[0]
    return np.concatenate([[-1.0], inner, [1.0]])


_ALPHA_OPT = [0.0, 0.0, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 1.2832,
              1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]  # fmt: skip


def _warp_factor(p, rout):
    """1D warp (GLL minus equispaced, interpolated) divided by 1 - r^2."""
    req = np.linspace(-1, 1, p + 1)
    rgll = gll_nodes(p)
    V1 = np.stack([jacobi_normalized(req, i, 0, 0) for i in range(p + 1)], 1)
    Pr = np.stack([jacobi_normalized(rout, i, 0, 0) for i in range(p + 1)], 1)
    Lmat = np.linalg.solve(V1.T, Pr.T)  # Lagrange basis on req at rout
    warp = Lmat.T @ (rgll - req)
    zerof = np.abs(rout) < 1.0 - 1e-10
    sf = 1.0 - (zerof * rout) ** 2
    return warp / sf + warp * (zerof - 1)


def warp_blend_nodes(p):
    """Warburton's warp-and-blend nodes on the reference triangle (r, s)."""
    if p == 0:
        return np.array([-1 / 3]), np.array([-1 / 3])
    alpha = _ALPHA_OPT[p - 1] if p < 16 else 5 / 3
    L1, L3 = [], []
    for n in range(p + 1):
        for m in range(p + 1 - n):
            L1.append(n / p)
            L3.append(m / p)
    L1, L3 = np.array(L1), np.array(L3)
    L2 = 1.0 - L1 - L3
    X = -L2 + L3
    Y = (-L2 - L3 + 2 * L1) / np.sqrt(3)
    blend1, blend2, blend3 = 4 * L2 * L3, 4 * L1 * L3, 4 * L1 * L2
    w1 = _warp_factor(p, L3 - L2)
    w2 = _warp_factor(p, L1 - L3)
    w3 = _warp_factor(p, L2 - L1)
    w1 = blend1 * w1 * (1 + (alpha * L1) ** 2)
    w2 = blend2 * w2 * (1 + (alpha * L2) ** 2)
    w3 = blend3 * w3 * (1 + (alpha * L3) ** 2)
    X = X + w1 + np.cos(2 * np.pi / 3) * w2 + np.cos(4 * np.pi / 3) * w3
    Y = Y + np.sin(2 * np.pi / 3) * w2 + np.sin(4 * np.pi / 3) * w3
    # equilateral (X, Y) -> reference (r, s)
    L1 = (np.sqrt(3) * Y + 1) / 3
    L2 = (-3 * X - np.sqrt(3) * Y + 2) / 6
    L3 = (3 * X - np.sqrt(3) * Y + 2) / 6
    return -L2 + L3 - L1, -L2 - L3 + L1


def _rs_to_ab(r, s):
    a = np.where(np.abs(s - 1) > 1e-12, 2 * (1 + r) / (1 - s + 1e-300) - 1, -1.0)
    return a, s


def basis(p, r, s):
    """Orthonormal basis psi_m(r, s), m = 0..Np-1; returns (len(r), Np)."""
    a, b = _rs_to_ab(np.asarray(r, float), np.asarray(s, float))
    cols = []
    for i in range(p + 1):
        for j in range(p + 1 - i):
            cols.append(
                np.sqrt(2)
                * jacobi_normalized(a, i, 0, 0)
                * jacobi_normalized(b, j, 2 * i + 1, 0)
                * (1 - b) ** i
            )
    return np.stack(cols, 1)


def grad_basis(p, r, s):
    """(d psi/dr, d psi/ds), each (len(r), Np)."""
    a, b = _rs_to_ab(np.asarray(r, float), np.asarray(s, float))
    dr, ds = [], []
    for i in range(p + 1):
        for j in range(p + 1 - i):
            fa = jacobi_normalized(a, i, 0, 0)
            dfa = _djacobi_normalized(a, i, 0, 0)
            gb = jacobi_normalized(b, j, 2 * i + 1, 0)
            dgb = _djacobi_normalized(b, j, 2 * i + 1, 0)
            pim1 = (1 - b) ** (i - 1) if i >= 1 else np.zeros_like(b)
            # d/dr = 2/(1-b) d/da
            dpr = 2 * dfa * gb * pim1
            # d/ds = (1+a)/(1-b) d/da + d/db
            dps = dfa * gb * (1 + a) * pim1
            dps = dps + fa * dgb * (1 - b) ** i
            if i >= 1:
                dps = dps - i * fa * gb * pim1
            dr.append(np.sqrt(2) * dpr)
            ds.append(np.sqrt(2) * dps)
    return np.stack(dr, 1), np.stack(ds, 1)


def collapsed_cubature(q):
    """Gauss cubature on the reference triangle, exact for degree 2 q - 1
    (q x q collapsed Gauss-Legendre / Gauss-Jacobi(1, 0))."""
    xa, wa = np.polynomial.legendre.leggauss(q)
    xb, wb = roots_jacobi(q, 1, 0)
    A, B = np.meshgrid(xa, xb, indexing="ij")
    W = np.outer(wa, wb) * 0.5
    r = 0.5 * (1 + A) * (1 - B) - 1
    return r.ravel(), B.ravel(), W.ravel()


class ReferenceTriangle:
    """
    Nodal operators of degree p on the reference triangle.

    Attributes
    ----------
    r, s : (Np,) node coordinates; V : Vandermonde; Dr, Ds : differentiation
    matrices; M : mass matrix; Fmask : (3, Nfp) node indices on each face,
    ordered along the counter-clockwise traversal; tf : (Nfp,) face-node
    parameters t in [-1, 1]; M1D : 1D face mass matrix; LIFT : (Np, 3 Nfp)
    reference lift M^{-1} E M1D (multiply by sJ / J per face node).
    """

    def __init__(self, p):
        if p < 1:
            raise ValueError("p >= 1")
        self.p = p
        self.Np = (p + 1) * (p + 2) // 2
        self.Nfp = p + 1
        self.r, self.s = warp_blend_nodes(p)
        self.V = basis(p, self.r, self.s)
        Vr, Vs = grad_basis(p, self.r, self.s)
        Vinv = np.linalg.inv(self.V)
        self.Vinv = Vinv
        self.Dr, self.Ds = Vr @ Vinv, Vs @ Vinv
        self.M = Vinv.T @ Vinv
        tol = 1e-10
        f1 = np.flatnonzero(np.abs(self.s + 1) < tol)
        f2 = np.flatnonzero(np.abs(self.r + self.s) < tol)
        f3 = np.flatnonzero(np.abs(self.r + 1) < tol)
        # order along ccw traversal; t = parameter in [-1, 1]
        f1 = f1[np.argsort(self.r[f1])]
        f2 = f2[np.argsort(self.s[f2])]
        f3 = f3[np.argsort(-self.s[f3])]
        self.Fmask = np.stack([f1, f2, f3])
        self.tf = self.r[f1].copy()  # GLL-like, identical on every face
        V1 = np.stack([jacobi_normalized(self.tf, i, 0, 0) for i in range(p + 1)], 1)
        self.V1D = V1
        self.M1D = np.linalg.inv(V1 @ V1.T)
        Emat = np.zeros((self.Np, 3 * self.Nfp))
        for f in range(3):
            Emat[np.ix_(self.Fmask[f], f * self.Nfp + np.arange(self.Nfp))] = self.M1D
        self.Emat = Emat
        self.LIFT = np.linalg.solve(self.M, Emat)

    # -------------------------------------------------------------- helpers
    def interp_matrix(self, r, s):
        """Values of the nodal Lagrange basis at points (r, s): (n, Np)."""
        return basis(self.p, r, s) @ self.Vinv

    def grad_interp_matrices(self, r, s):
        Vr, Vs = grad_basis(self.p, r, s)
        return Vr @ self.Vinv, Vs @ self.Vinv

    def face_interp_matrix(self, t):
        """1D Lagrange basis on the face nodes evaluated at t: (n, Nfp)."""
        P = np.stack([jacobi_normalized(t, i, 0, 0) for i in range(self.p + 1)], 1)
        return P @ np.linalg.inv(self.V1D)

    def face_points(self, f, t):
        """Reference (r, s) of the face-f point with parameter t."""
        t = np.asarray(t, float)
        if f == 0:
            return t, -np.ones_like(t)
        if f == 1:
            return -t, t
        return -np.ones_like(t), -t


@lru_cache(maxsize=16)
def reference_triangle(p):
    return ReferenceTriangle(p)
