# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Mie theory: scattering of a plane wave by a homogeneous sphere.

An independent implementation of the Lorenz-Mie series following Bohren &
Huffman (1983), chapter 4, used as the analytic reference for the scattering
tutorials and tests.

Conventions
-----------
* Size parameter ``x = 2 pi n_medium r / lambda0 = k r`` with ``k`` the
  wavenumber in the surrounding (lossless) medium; relative refractive index
  ``m = n_sphere / n_medium``.
* Bohren & Huffman use the time dependence ``exp(-i w t)``, in which an
  absorbing sphere has ``Im(m) > 0``. Wakis (and most engineering texts) use
  ``exp(+i w t)``, in which the same sphere has ``Im(m) < 0``. Both inputs are
  accepted: a negative imaginary part is read as the ``exp(+i w t)``
  convention. Efficiencies and cross sections do not depend on the
  convention; the complex amplitudes ``a_n, b_n, S1, S2`` are returned in the
  convention of the input (complex conjugates of the Bohren & Huffman ones for
  ``Im(m) < 0``). Gain media are not supported.
* The series is truncated after ``n_stop = x + 4 x^(1/3) + 2`` terms
  (Wiscombe 1980, as in Bohren & Huffman's BHMIE). The logarithmic derivative
  ``D_n(m x) = psi_n'(m x) / psi_n(m x)`` is computed by downward recurrence,
  which is stable for complex and large ``|m x|``; the Riccati-Bessel
  functions of the real argument ``x`` by upward recurrence.

References
----------
C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small
Particles (Wiley, 1983), ch. 4 and appendix A.
W. J. Wiscombe, "Improved Mie scattering algorithms," Appl. Opt. 19, 1505
(1980).
"""

import numpy as np

__all__ = [
    "mie_coefficients",
    "mie_efficiencies",
    "mie_cross_sections",
    "mie_s1_s2",
    "mie_differential_cross_section",
    "rayleigh_efficiencies",
]


def _n_stop(x):
    return int(np.floor(x + 4.0 * x ** (1.0 / 3.0) + 2.0))


def _split_convention(m):
    """Return (m in the exp(-iwt) convention, whether to conjugate outputs)."""
    m = complex(m)
    if m.imag < 0:
        return m.conjugate(), True
    return m, False


def _coefficients_bh(m, x, n_terms=None):
    """a_n, b_n (n = 1..N) in the Bohren & Huffman exp(-iwt) convention."""
    if x <= 0:
        raise ValueError("size parameter x must be > 0.")
    if m == 0:
        raise ValueError("relative refractive index m must be nonzero.")
    n_stop = _n_stop(x) if n_terms is None else int(n_terms)
    mx = m * x
    n_mx = int(max(n_stop, abs(mx))) + 16
    # Logarithmic derivative D_n(mx) by downward recurrence from D = 0
    D = np.zeros(n_mx + 1, dtype=complex)
    for n in range(n_mx, 0, -1):
        D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)
    n = np.arange(1, n_stop + 1)
    # Riccati-Bessel psi_n(x) = x j_n(x), chi_n(x) = -x y_n(x), upward
    psi = np.empty(n_stop + 1)
    chi = np.empty(n_stop + 1)
    psi[0], chi[0] = np.sin(x), np.cos(x)
    psi_m1, chi_m1 = np.cos(x), -np.sin(x)  # n = -1
    prev_psi, prev_chi = psi_m1, chi_m1
    for k in range(1, n_stop + 1):
        psi[k] = (2 * k - 1) / x * psi[k - 1] - prev_psi
        chi[k] = (2 * k - 1) / x * chi[k - 1] - prev_chi
        prev_psi, prev_chi = psi[k - 1], chi[k - 1]
    xi = psi - 1j * chi
    Dn = D[1 : n_stop + 1]
    ta = Dn / m + n / x
    tb = m * Dn + n / x
    a = (ta * psi[1:] - psi[:-1]) / (ta * xi[1:] - xi[:-1])
    b = (tb * psi[1:] - psi[:-1]) / (tb * xi[1:] - xi[:-1])
    return a, b


def mie_coefficients(m, x, n_terms=None):
    """
    Mie scattering coefficients a_n and b_n of a homogeneous sphere.

    Parameters
    ----------
    m : complex
        Relative refractive index n_sphere / n_medium (see the module notes
        on the sign of the imaginary part).
    x : float
        Size parameter k r (k in the medium).
    n_terms : int, optional
        Number of terms; default x + 4 x^(1/3) + 2.

    Returns
    -------
    a, b : ndarray of complex
        Coefficients for n = 1 ... n_terms (dimensionless).
    """
    m_bh, conj = _split_convention(m)
    a, b = _coefficients_bh(m_bh, float(x), n_terms)
    return (a.conj(), b.conj()) if conj else (a, b)


def _efficiencies_scalar(m, x):
    a, b = _coefficients_bh(m, x)
    n = np.arange(1, a.size + 1)
    w = 2 * n + 1
    q_sca = 2.0 / x**2 * np.sum(w * (np.abs(a) ** 2 + np.abs(b) ** 2))
    q_ext = 2.0 / x**2 * np.sum(w * (a.real + b.real))
    q_back = np.abs(np.sum(w * (-1.0) ** n * (a - b))) ** 2 / x**2
    return q_ext, q_sca, q_ext - q_sca, q_back


def mie_efficiencies(m, x):
    """
    Extinction, scattering, absorption and backscattering efficiencies.

    Q = sigma / (pi r^2), with sigma the cross section. Q_back is defined as
    in Bohren & Huffman (ch. 4): 4 pi times the differential scattering
    cross section in the backward direction divided by pi r^2 (the radar
    cross section over pi r^2).

    Parameters
    ----------
    m : complex
        Relative refractive index.
    x : float or array_like
        Size parameter(s) k r.

    Returns
    -------
    q_ext, q_sca, q_abs, q_back : float or ndarray
        Dimensionless efficiencies, with the shape of x.
    """
    m_bh, _ = _split_convention(m)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.array([_efficiencies_scalar(m_bh, xi) for xi in x_arr.ravel()])
    out = out.T.reshape(4, *x_arr.shape)
    if np.ndim(x) == 0:
        return tuple(float(v[0]) for v in out)
    return tuple(out)


def mie_cross_sections(m_sphere, radius, wavelength, n_medium=1.0):
    """
    Extinction, scattering, absorption and backscattering cross sections.

    Parameters
    ----------
    m_sphere : complex or array_like
        Refractive index of the sphere (may depend on wavelength: one value
        per wavelength).
    radius : float
        Sphere radius [m].
    wavelength : float or array_like
        Vacuum wavelength(s) [m].
    n_medium : float, optional
        Refractive index of the lossless surrounding medium.

    Returns
    -------
    sigma_ext, sigma_sca, sigma_abs, sigma_back : ndarray
        Cross sections [m^2].
    """
    wavelength = np.atleast_1d(np.asarray(wavelength, dtype=float))
    m = np.broadcast_to(np.asarray(m_sphere, dtype=complex), wavelength.shape)
    x = 2 * np.pi * n_medium * radius / wavelength
    q = np.array(
        [mie_efficiencies(mi / n_medium, xi) for mi, xi in zip(m.ravel(), x.ravel())]
    ).T
    return tuple(qi.reshape(wavelength.shape) * np.pi * radius**2 for qi in q)


def mie_s1_s2(m, x, theta):
    """
    Scattering amplitudes S1(theta) and S2(theta) (Bohren & Huffman ch. 4).

    For an incident wave polarized along x and travelling along +z, the far
    field at polar angle theta and azimuth phi is
    E_theta = E0 exp(ikr)/(-ikr) cos(phi) S2,
    E_phi = -E0 exp(ikr)/(-ikr) sin(phi) S1 (exp(-iwt) convention).

    Parameters
    ----------
    m : complex
        Relative refractive index.
    x : float
        Size parameter k r.
    theta : float or array_like
        Scattering angle(s) from the forward direction [rad].

    Returns
    -------
    S1, S2 : ndarray of complex
        Dimensionless amplitudes with the shape of theta.
    """
    m_bh, conj = _split_convention(m)
    a, b = _coefficients_bh(m_bh, float(x))
    mu = np.cos(np.atleast_1d(np.asarray(theta, dtype=float)))
    n_stop = a.size
    S1 = np.zeros(mu.shape, dtype=complex)
    S2 = np.zeros(mu.shape, dtype=complex)
    pi_prev = np.zeros_like(mu)  # pi_0
    pi_n = np.ones_like(mu)  # pi_1
    for n in range(1, n_stop + 1):
        tau_n = n * mu * pi_n - (n + 1) * pi_prev
        f = (2 * n + 1) / (n * (n + 1))
        S1 += f * (a[n - 1] * pi_n + b[n - 1] * tau_n)
        S2 += f * (a[n - 1] * tau_n + b[n - 1] * pi_n)
        pi_next = ((2 * n + 1) * mu * pi_n - (n + 1) * pi_prev) / n
        pi_prev, pi_n = pi_n, pi_next
    if conj:
        S1, S2 = S1.conj(), S2.conj()
    shape = np.shape(theta)
    return S1.reshape(shape), S2.reshape(shape)


def mie_differential_cross_section(m, radius, wavelength, theta, phi, n_medium=1.0):
    """
    Differential scattering cross section dsigma/dOmega for x-polarized
    incidence along +z.

    dsigma/dOmega = (|S2|^2 cos^2(phi) + |S1|^2 sin^2(phi)) / k^2.

    Parameters
    ----------
    m : complex
        Refractive index of the sphere.
    radius : float
        Sphere radius [m].
    wavelength : float
        Vacuum wavelength [m].
    theta, phi : float or array_like
        Scattering angles [rad] (broadcast together).
    n_medium : float, optional
        Refractive index of the surrounding medium.

    Returns
    -------
    ndarray
        dsigma/dOmega [m^2 / sr].
    """
    k = 2 * np.pi * n_medium / wavelength
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    S1, S2 = mie_s1_s2(complex(m) / n_medium, k * radius, theta.ravel())
    out = (
        np.abs(S2) ** 2 * np.cos(phi.ravel()) ** 2
        + np.abs(S1) ** 2 * np.sin(phi.ravel()) ** 2
    ) / k**2
    return out.reshape(theta.shape)


def rayleigh_efficiencies(m, x):
    """
    Small-sphere (Rayleigh, x << 1, |m| x << 1) limit of the efficiencies.

    Q_sca = 8/3 x^4 |L|^2 and Q_abs = 4 x Im(L) with
    L = (m^2 - 1)/(m^2 + 2) (Bohren & Huffman ch. 5, in the exp(-iwt)
    convention; the sign of Im(m) is normalized as in the module).

    Parameters
    ----------
    m : complex
        Relative refractive index.
    x : float or array_like
        Size parameter.

    Returns
    -------
    q_sca, q_abs : float or ndarray
    """
    m_bh, _ = _split_convention(m)
    x = np.asarray(x, dtype=float)
    L = (m_bh**2 - 1) / (m_bh**2 + 2)
    return 8.0 / 3.0 * x**4 * np.abs(L) ** 2, 4.0 * x * L.imag
