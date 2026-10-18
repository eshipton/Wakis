"""
Exact references for Drude spheres: the TM_n modes of a Drude sphere in a
spherical PEC cavity (Riccati-Bessel matching, lossless) and the complex
dipole (a_1) resonance of an isolated sphere (Mie), plus the quasi-static
limit.

Riccati-Bessel functions psi_n(z) = z j_n(z), xi_n(z) = z h_n^(2)(z) for the
outgoing wave in the exp(+i w t) convention of Wakis.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn


def _psi(n, z, deriv=False):
    j = spherical_jn(n, z)
    if not deriv:
        return z * j
    return j + z * spherical_jn(n, z, derivative=True)


def _chi(n, z, deriv=False):
    y = spherical_yn(n, z)
    if not deriv:
        return z * y
    return y + z * spherical_yn(n, z, derivative=True)


def drude_eps(k, k_p, gamma_k=0.0, eps_inf=1.0):
    """eps(k) = eps_inf - k_p^2 / (k^2 - i gamma k) (exp(+i w t), k = w / c)."""
    return eps_inf - k_p**2 / (k**2 - 1j * gamma_k * k)


def cavity_tm(a, b, k_p, n=1, kmin=None, kmax=None, n_scan=20000):
    """
    TM_n resonances (E crossing the sphere surface) of a lossless Drude
    sphere (radius b, eps = 1 - (k_p/k)^2) concentric in a PEC sphere
    (radius a): u = r f with u'(a) = 0, u and u'/eps continuous at b.

    Returns
    -------
    ndarray
        Vacuum wavenumbers k [1/m].
    """
    kmin = kmin or 0.02 * k_p
    kmax = kmax or 1.5 * k_p

    def det(k):
        e = 1 - (k_p / k) ** 2
        A, B = _chi(n, k * a, True), -_psi(n, k * a, True)
        u_out = A * _psi(n, k * b) + B * _chi(n, k * b)
        du_out = k * (A * _psi(n, k * b, True) + B * _chi(n, k * b, True))
        m = np.sqrt(complex(e))
        z = m * k * b
        # logarithmic derivative of psi_n (real for real or imaginary z)
        ld = (m * k * _psi(n, z, True) / _psi(n, z)).real
        return ld / e * u_out - du_out

    ks = np.linspace(kmin, kmax, n_scan)
    vals = np.array([det(k) for k in ks])
    out = []
    for i in np.flatnonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0):
        r = brentq(det, ks[i], ks[i + 1], xtol=1e-15 * kmax)
        if abs(det(r)) < 1e-6 * (1 + np.max(np.abs(vals))):
            out.append(r)
    return np.array(out)


def mie_a1_pole(b, k_p, gamma_k=0.0, eps_inf=1.0, k_guess=None):
    """
    Complex dipole resonance k = k' + i k'' of an isolated Drude sphere: the
    zero of the denominator of the Mie coefficient a_1,
    m psi_1(m x) xi_1'(x) - xi_1(x) psi_1'(m x) (x = k b, m = sqrt(eps)),
    Bohren & Huffman (1983), ch. 4. With exp(+i w t) a decaying mode
    has Im k > 0 (w = c k, field ~ exp(i w t) = exp(i c k' t - c k'' t)).
    """
    from scipy.optimize import fsolve

    def xi(z, d=False):
        return _psi(1, z, d) - 1j * _chi(1, z, d)  # z h^(2)

    def f(v):
        k = v[0] + 1j * v[1]
        e = drude_eps(k, k_p, gamma_k, eps_inf)
        m = np.sqrt(e)
        x = k * b
        # divided by psi_1(m x): removes the trivial root m = 0 (eps = 0)
        val = m * xi(x, True) - xi(x) * _psi(1, m * x, True) / _psi(1, m * x)
        return [val.real, val.imag]

    k0 = k_guess if k_guess is not None else k_p / np.sqrt(eps_inf + 2)
    sol = fsolve(f, [np.real(k0), 0.05 * np.real(k0)], xtol=1e-14)
    return sol[0] + 1j * sol[1]


def quasistatic_k(k_p, gamma_k=0.0, eps_inf=1.0, eps_bg=1.0):
    """Froehlich condition Re eps = -2 eps_bg: k = k_p / sqrt(eps_inf + 2 eps_bg)."""
    return k_p / np.sqrt(eps_inf + 2 * eps_bg)
