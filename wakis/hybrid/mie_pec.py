# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Near field of a plane wave scattered by a perfectly conducting sphere.

Bohren & Huffman (1983), Ch. 4, exp(-i w t) convention: incident
``E_i = x_hat exp(i k z)`` (unit amplitude), sphere of radius ``a`` at the
origin. With the vector spherical harmonics of the generating function
``cos(phi) P_n^1(cos theta) z_n(k r)`` (B&H 4.50),

    E_i = sum_n E_n (M_o1n^(1) - i N_e1n^(1)),
    E_s = sum_n E_n (i a_n N_e1n^(3) - b_n M_o1n^(3)),
    E_n = i^n (2n + 1) / (n (n + 1)),

and for a perfect conductor (m -> infinity in B&H 4.53)
``a_n = psi_n'(x) / xi_n'(x)``, ``b_n = psi_n(x) / xi_n(x)`` with
``psi_n = x j_n(x)``, ``xi_n = x h_n^(1)(x)``, ``x = k a``.

Self-checks (`tests/test_105_hybrid_cart_tet.py`): the incident series equals
``exp(i k z) x_hat``, and the total tangential field vanishes on ``r = a``.
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn

__all__ = ["pec_sphere_field", "segment_line_integrals", "plane_wave_line_integrals"]


def _pi_tau(mu, n_max):
    pi = np.zeros((n_max + 1,) + mu.shape)
    tau = np.zeros_like(pi)
    pi[1] = 1.0
    for n in range(2, n_max + 1):
        pi[n] = (2 * n - 1) / (n - 1) * mu * pi[n - 1] - n / (n - 1) * pi[n - 2]
    for n in range(1, n_max + 1):
        tau[n] = n * mu * pi[n] - (n + 1) * pi[n - 1]
    return pi, tau


def _zn(n, rho, kind):
    j = spherical_jn(n, rho)
    jp = spherical_jn(n, rho, derivative=True)
    if kind == 1:
        return j, j + rho * jp  # z_n, (rho z_n)'
    y = spherical_yn(n, rho)
    yp = spherical_yn(n, rho, derivative=True)
    h, hp = j + 1j * y, jp + 1j * yp
    return h, h + rho * hp


def pec_sphere_field(points, k, a, which="scattered", n_max=None):
    """
    Cartesian E field [V/m] (unit incident amplitude) at `points` (n, 3) [m].

    Parameters
    ----------
    k : float
        Wavenumber [1/m].
    a : float
        Sphere radius [m].
    which : {'scattered', 'incident', 'total'}
    """
    P = np.atleast_2d(np.asarray(points, float))
    x = k * a
    r = np.linalg.norm(P, axis=1)
    if n_max is None:
        xm = x if which == "scattered" else max(x, k * r.max())
        n_max = int(np.ceil(xm + 4 * xm ** (1 / 3) + 2)) + 6
    rho = k * r
    th = np.arccos(np.clip(P[:, 2] / np.maximum(r, 1e-300), -1, 1))
    ph = np.arctan2(P[:, 1], P[:, 0])
    mu, st = np.cos(th), np.sin(th)
    cp, sp_ = np.cos(ph), np.sin(ph)
    pi, tau = _pi_tau(mu, n_max)
    Er = np.zeros(len(P), complex)
    Et = np.zeros(len(P), complex)
    Ep = np.zeros(len(P), complex)
    for n in range(1, n_max + 1):
        En = 1j**n * (2 * n + 1) / (n * (n + 1))
        terms = []
        if which in ("incident", "total"):
            terms.append((1, 1.0, -1j))  # (kind, coef of M_o1n, coef of N_e1n)
        if which in ("scattered", "total"):
            psi = x * spherical_jn(n, x)
            dpsi = spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True)
            hn = spherical_jn(n, x) + 1j * spherical_yn(n, x)
            dhn = (spherical_jn(n, x, derivative=True)
                   + 1j * spherical_yn(n, x, derivative=True))  # fmt: skip
            xi, dxi = x * hn, hn + x * dhn
            an, bn = dpsi / dxi, psi / xi
            terms.append((3, -bn, 1j * an))
        for kind, cM, cN in terms:
            z, dz = _zn(n, rho, kind)
            # M_o1n = cos(phi) pi z e_th - sin(phi) tau z e_ph
            # N_e1n = cos(phi) n(n+1) sin(th) pi z/rho e_r
            #         + cos(phi) tau (rho z)'/rho e_th - sin(phi) pi (rho z)'/rho e_ph
            Er += En * cN * cp * n * (n + 1) * st * pi[n] * z / rho
            Et += En * (cM * cp * pi[n] * z + cN * cp * tau[n] * dz / rho)
            Ep += En * (-cM * sp_ * tau[n] * z - cN * sp_ * pi[n] * dz / rho)
    er = np.stack([st * cp, st * sp_, mu], 1)
    et = np.stack([mu * cp, mu * sp_, -st], 1)
    ep = np.stack([-sp_, cp, np.zeros_like(cp)], 1)
    return Er[:, None] * er + Et[:, None] * et + Ep[:, None] * ep


def segment_line_integrals(field, p, q, n_gauss=4):
    """Gauss-Legendre line integrals of ``field(points) -> (n, 3)`` from p to q."""
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    t = 0.5 * (xg + 1)
    d = q - p
    out = 0
    for tt, ww in zip(t, wg):
        out = out + 0.5 * ww * np.einsum("ij,ij->i", field(p + tt * d), d)
    return out


def plane_wave_line_integrals(k, p, q):
    """Exact line integrals of ``x_hat exp(i k z)`` along segments p -> q."""
    d = q - p
    zm = 0.5 * (p[:, 2] + q[:, 2])
    arg = 0.5 * k * d[:, 2]
    return d[:, 0] * np.exp(1j * k * zm) * np.sinc(arg / np.pi)
