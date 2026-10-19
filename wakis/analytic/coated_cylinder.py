# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plane-wave scattering by a coated (two-layer) infinite circular cylinder at
normal incidence: core radius a1 (index m1), shell out to a2 (index m2), in a
medium of index 1. Same conventions as `wakis.analytic.mie_cylinder`
(Bohren & Huffman, exp(-i w t); TM = E along the axis, coefficients b_n;
TE = H along the axis, coefficients a_n).

Per order n the field u (E_z for TM, H_z for TE) is

    core : A J_n(m1 k r)
    shell: B J_n(m2 k r) + C Y_n(m2 k r)
    out  : J_n(k r) - c_n H_n^(1)(k r)      (times i^n U0 e^{i n theta})

with u and (1/p) du/dr continuous at r = a1 and a2, p = 1 (TM, non-magnetic)
or p = m^2 (TE: (1/eps) dH_z/dr is the tangential E). The 4 x 4 system is
solved directly. For m1 = m2 it reduces to `cylinder_coefficients`; the
efficiencies follow as Q_sca = (2/x) sum_{n=-N..N} |c_n|^2 with x = k a2.
"""

import numpy as np
from scipy.special import hankel1, h1vp, jv, jvp, yv, yvp

from .mie import _split_convention

__all__ = ["coated_cylinder_coefficients", "coated_cylinder_efficiencies"]


def coated_cylinder_coefficients(m1, m2, x1, x2, polarization="TM", n_terms=None):
    """
    Scattering coefficients c_n (n = 0..N) of a coated cylinder.

    Parameters
    ----------
    m1, m2 : complex
        Relative indices of the core and the shell (Im m < 0 is read as the
        exp(+i w t) convention, as in `wakis.analytic.mie`).
    x1, x2 : float
        Size parameters k a1 and k a2 (x1 < x2).
    polarization : {'TM', 'TE'}
    n_terms : int, optional
        Highest order (default x2 + 4 x2^(1/3) + 2).

    Returns
    -------
    ndarray of complex
    """
    m1b, conj1 = _split_convention(m1)
    m2b, conj2 = _split_convention(m2)
    if conj1 != conj2 and np.imag(m1) != 0 and np.imag(m2) != 0:
        raise ValueError("m1 and m2 must use the same sign convention")
    conj = conj1 or conj2
    pol = polarization.upper()
    n_max = int(np.floor(x2 + 4 * x2 ** (1 / 3) + 2)) if n_terms is None else int(n_terms)
    p1, p2 = (1.0, 1.0) if pol == "TM" else (m1b**2, m2b**2)
    out = np.zeros(n_max + 1, complex)
    for n in range(n_max + 1):
        z11, z21, z22 = m1b * x1, m2b * x1, m2b * x2
        M = np.zeros((4, 4), complex)
        rhs = np.zeros(4, complex)
        # unknowns A, B, C, c ; derivatives d/d(k r)
        M[0] = [jv(n, z11), -jv(n, z21), -yv(n, z21), 0]
        M[1] = [m1b / p1 * jvp(n, z11), -m2b / p2 * jvp(n, z21), -m2b / p2 * yvp(n, z21), 0]
        M[2] = [0, jv(n, z22), yv(n, z22), hankel1(n, x2)]
        M[3] = [0, m2b / p2 * jvp(n, z22), m2b / p2 * yvp(n, z22), h1vp(n, x2)]
        rhs[2] = jv(n, x2)
        rhs[3] = jvp(n, x2)
        out[n] = np.linalg.solve(M, rhs)[3]
    return out.conj() if conj else out


def coated_cylinder_efficiencies(m1, m2, x1, x2, polarization="TM"):
    """(Q_ext, Q_sca) per unit length, normalized by the outer diameter 2 a2."""
    c = coated_cylinder_coefficients(m1, m2, x1, x2, polarization)
    q_sca = 2 / x2 * (abs(c[0]) ** 2 + 2 * np.sum(np.abs(c[1:]) ** 2))
    q_ext = 2 / x2 * np.real(c[0] + 2 * np.sum(c[1:]))
    return q_ext, q_sca
