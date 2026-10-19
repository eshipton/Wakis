# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Scattering of a plane wave by an infinite homogeneous circular cylinder at
normal incidence (the 2D Mie problem), after Bohren & Huffman (1983), ch. 8.

Polarizations (normal incidence, cylinder axis along z):

* ``'TM'`` -- E parallel to the axis (E_z, in-plane H). Bohren & Huffman's
  case I; only the coefficients b_n are nonzero.
* ``'TE'`` -- E perpendicular to the axis (in-plane E, H_z). Bohren &
  Huffman's case II; only the coefficients a_n are nonzero.

(Some texts swap the TE/TM labels; the field components above are the
definition used here.)

With size parameter ``x = k a`` and relative index ``m``, the efficiencies
per unit length, normalized by the projected width 2a, are

    Q_sca = (2/x) [ |c_0|^2 + 2 sum_{n>=1} |c_n|^2 ],
    Q_ext = (2/x) Re[ c_0 + 2 sum_{n>=1} c_n ],

with c_n = b_n (TM) or a_n (TE), i.e. C_sca = Q_sca 2a is the scattering
width [m] (scattered power per unit length divided by the incident
intensity). The amplitude T(theta) = c_0 + 2 sum c_n cos(n theta) gives the
differential scattering width dC/dtheta = 2 |T|^2 / (pi k).

The ratio J_n'(m x)/J_n(m x) is computed by downward recurrence (stable for
complex and large |m x|); J_n, Y_n of the real argument x by SciPy. Complex
m follows the convention notes of `wakis.analytic.mie` (Im m < 0 is read as
the exp(+i w t) convention).

References
----------
C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small
Particles (Wiley, 1983), ch. 8.
"""

import numpy as np
from scipy.special import jv, jvp, yv, yvp

from .mie import _split_convention

__all__ = [
    "cylinder_coefficients",
    "cylinder_efficiencies",
    "cylinder_amplitude",
    "cylinder_scattering_width",
    "cylinder_rayleigh_efficiencies",
]


def _n_stop(x):
    return int(np.floor(x + 4.0 * x ** (1.0 / 3.0) + 2.0))


def _log_derivative(z, n_max):
    """D_n(z) = J_n'(z)/J_n(z), n = 0..n_max, by downward recurrence."""
    n_start = int(max(n_max, abs(z))) + 32
    # r_n = J_{n-1}(z) / J_n(z) = 2n/z - 1/r_{n+1}, started with J_{N+1}/J_N ~ 0
    inv_r = 0.0  # 1 / r_{N+1} = J_{N+1}/J_N
    r = np.zeros(n_start + 1, dtype=complex)
    for n in range(n_start, -1, -1):
        r[n] = 2 * n / z - inv_r
        inv_r = 1.0 / r[n]
    n = np.arange(n_max + 1)
    return r[: n_max + 1] - n / z  # J_n' = J_{n-1} - (n/z) J_n


def _coefficients_bh(m, x, polarization, n_terms=None):
    if x <= 0:
        raise ValueError("size parameter x must be > 0.")
    pol = polarization.upper()
    if pol not in ("TM", "TE"):
        raise ValueError("polarization must be 'TM' (E along the axis) or 'TE'.")
    n_max = _n_stop(x) if n_terms is None else int(n_terms)
    n = np.arange(n_max + 1)
    D = _log_derivative(m * x, n_max)
    J, dJ = jv(n, x), jvp(n, x)
    H, dH = J + 1j * yv(n, x), dJ + 1j * yvp(n, x)
    if pol == "TM":  # b_n, case I
        return (dJ - m * D * J) / (dH - m * D * H)
    return (m * dJ - D * J) / (m * dH - D * H)  # a_n, case II


def cylinder_coefficients(m, x, polarization="TM", n_terms=None):
    """
    Scattering coefficients of an infinite cylinder at normal incidence.

    Parameters
    ----------
    m : complex
        Relative refractive index n_cylinder / n_medium.
    x : float
        Size parameter k a (k in the medium, a the radius).
    polarization : {'TM', 'TE'}
        'TM': E along the axis (b_n); 'TE': E perpendicular to it (a_n).
    n_terms : int, optional
        Highest order n (default x + 4 x^(1/3) + 2).

    Returns
    -------
    ndarray of complex
        c_n for n = 0 ... n_terms.
    """
    m_bh, conj = _split_convention(m)
    c = _coefficients_bh(m_bh, float(x), polarization, n_terms)
    return c.conj() if conj else c


def cylinder_efficiencies(m, x, polarization="TM"):
    """
    Extinction, scattering and absorption efficiencies per unit length.

    Q = C / (2a), with C the cross section per unit length (scattering
    width) [m].

    Parameters
    ----------
    m : complex
        Relative refractive index.
    x : float or array_like
        Size parameter(s) k a.
    polarization : {'TM', 'TE'}

    Returns
    -------
    q_ext, q_sca, q_abs : float or ndarray
    """
    m_bh, _ = _split_convention(m)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty((3, xs.size))
    for i, xi in enumerate(xs.ravel()):
        c = _coefficients_bh(m_bh, xi, polarization)
        w = np.full(c.size, 2.0)
        w[0] = 1.0
        q_sca = 2.0 / xi * np.sum(w * np.abs(c) ** 2)
        q_ext = 2.0 / xi * np.sum(w * c.real)
        out[:, i] = q_ext, q_sca, q_ext - q_sca
    out = out.reshape(3, *xs.shape)
    if np.ndim(x) == 0:
        return tuple(float(v[0]) for v in out)
    return tuple(out)


def cylinder_amplitude(m, x, theta, polarization="TM"):
    """
    Scattering amplitude T(theta) = c_0 + 2 sum c_n cos(n theta).

    Parameters
    ----------
    theta : float or array_like
        Scattering angle from the forward direction, in the plane normal to
        the axis [rad].

    Returns
    -------
    ndarray of complex
    """
    c = cylinder_coefficients(m, x, polarization)
    theta = np.asarray(theta, dtype=float)
    n = np.arange(c.size)
    w = np.full(c.size, 2.0)
    w[0] = 1.0
    return np.tensordot(np.cos(np.multiply.outer(theta, n)), w * c, axes=([-1], [0]))


def cylinder_scattering_width(m_cyl, radius, wavelength, theta=None,
                              polarization="TM", n_medium=1.0):  # fmt: skip
    """
    Scattering width C_sca [m] (or dC/dtheta [m/rad] if theta is given).

    Parameters
    ----------
    m_cyl : complex
        Refractive index of the cylinder.
    radius : float
        Radius [m].
    wavelength : float or array_like
        Vacuum wavelength [m] (scalar when theta is given).
    theta : array_like, optional
        Scattering angles [rad]; returns the differential width
        2 |T|^2 / (pi k).
    polarization : {'TM', 'TE'}
    n_medium : float
        Index of the lossless surrounding medium.
    """
    k = 2 * np.pi * n_medium / np.asarray(wavelength, dtype=float)
    m = complex(m_cyl) / n_medium
    if theta is not None:
        T = cylinder_amplitude(m, float(k * radius), theta, polarization)
        return 2 * np.abs(T) ** 2 / (np.pi * k)
    q_sca = cylinder_efficiencies(m, k * radius, polarization)[1]
    return q_sca * 2 * radius


def cylinder_rayleigh_efficiencies(m, x, polarization="TM"):
    """
    Thin-cylinder limit (x << 1, |m| x << 1) of Q_sca.

    TM (E along the axis, no depolarization): Q_sca = pi^2 x^3 |m^2 - 1|^2 / 8.
    TE (E across the axis, 2D depolarization):
    Q_sca = pi^2 x^3 |(m^2 - 1)/(m^2 + 1)|^2 / 4.
    Both follow from the radiation of the induced line dipole (see the
    tutorial for the derivation).
    """
    m_bh, _ = _split_convention(m)
    x = np.asarray(x, dtype=float)
    if polarization.upper() == "TM":
        return np.pi**2 * x**3 * np.abs(m_bh**2 - 1) ** 2 / 8
    return np.pi**2 * x**3 * np.abs((m_bh**2 - 1) / (m_bh**2 + 1)) ** 2 / 4
