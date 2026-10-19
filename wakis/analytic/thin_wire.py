# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Thin-wire straight dipole: Hallen moment method and induced-EMF formulas.

Convention exp(+j w t) (engineering; Z = R + j X with X > 0 inductive), as
Wakis lumped ports. A straight PEC wire of length L and radius a along z,
centre-fed by a delta-gap generator of voltage V at z = 0.

Hallen's equation (reduced kernel, current on the axis, fields matched on
the surface; e.g. Balanis, Antenna Theory, sec. 8.3). With A_z the vector
potential on the wire, E_z = (d^2/dz^2 + k^2) A_z / (j w mu eps) and the
total tangential E vanishing on the wire except across the gap,
(d^2/dz^2 + k^2) A_z = -j w mu eps V delta(z), whose solution is
A_z = C cos k z - j sqrt(mu eps) (V / 2) sin k|z|. Hence

    int_{-L/2}^{L/2} I(z') K(z - z') dz' = C' cos k z - j (2 pi / eta) V sin k|z|,

(for a gap of finite width g the sin k|z| / 2 becomes `gap_excitation`),
    K(u) = exp(-j k R) / R,  R = sqrt(u^2 + a^2).

Discretization: N (even) segments, piecewise-linear (triangle) basis on the
N - 1 interior nodes (I = 0 at the ends), unknown constant C'; matching at
the N - 1 interior nodes and at the end node z = L/2 (the other end follows
by symmetry). The quasi-singular 1/R part of the kernel is integrated
analytically against the linear basis; the smooth remainder
(exp(-j k R) - 1)/R by Gauss-Legendre quadrature. Z_in = V / I(0).

The radiated power is computed independently from the far field of the
current, P_rad = (eta k^2 / (32 pi^2)) int |sin(theta) int I e^{j k z cos theta}
dz|^2 dOmega, and must equal Re(V conj(I(0))) / 2 (energy balance).

References
----------
C. A. Balanis, Antenna Theory: Analysis and Design, 3rd ed. (Wiley, 2005):
sec. 8.3 (Hallen's equation), sec. 8.5 (induced-EMF self impedance,
eqs. 8-60a,b), sec. 4.6 (half-wave dipole: 73 + j 42.5 Ohm).
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0, mu_0
from scipy.special import sici

ETA0 = np.sqrt(mu_0 / epsilon_0)
EULER = 0.5772156649015329

__all__ = [
    "hallen_dipole",
    "gap_excitation",
    "induced_emf_impedance",
    "dipole_far_field",
]


def _gauss(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1), 0.5 * w  # on [0, 1]


def gap_excitation(z, k, gap):
    """
    Right-hand side shape of Hallen's equation for a feed gap of width g.

    The generator field E = -V / g on |z| < g / 2 gives the particular
    solution (per volt, times -j 4 pi / eta)
    sin(k|z|) sin(k g / 2) / (k g) for |z| >= g / 2 and
    (1 - cos(k g / 2) cos(k z)) / (k g) inside; g -> 0 recovers the delta
    gap sin(k|z|) / 2.
    """
    z = np.abs(np.asarray(z, float))
    if gap == 0:
        return 0.5 * np.sin(k * z)
    out = np.sin(k * z) * np.sin(k * gap / 2) / (k * gap)
    inside = z < gap / 2
    out[inside] = (1 - np.cos(k * gap / 2) * np.cos(k * z[inside])) / (k * gap)
    return out


def hallen_dipole(length, radius, freq, n_segments=100, n_gauss=8, gap=0.0):
    """
    Input impedance and current of a centre-fed thin-wire dipole (Hallen MoM).

    Parameters
    ----------
    length, radius : float
        Total length L and wire radius a [m].
    freq : float or array_like
        Frequency [Hz].
    n_segments : int
        Number of segments N (made even).
    n_gauss : int
        Gauss points per half-segment for the smooth kernel part.
    gap : float
        Feed-gap width g [m] (uniform generator field over |z| < g / 2);
        0 is the delta gap, whose input reactance does not converge with N
        (gap capacitance).

    Returns
    -------
    z_in : complex ndarray
        Input impedance [Ohm] (exp(+j w t)).
    z_nodes : ndarray (N + 1,)
        Node positions [m].
    currents : complex ndarray (n_freq, N + 1)
        Node currents [A] for V = 1 V.
    """
    N = int(n_segments) + int(n_segments) % 2
    L, a = float(length), float(radius)
    d = L / N
    z = -L / 2 + d * np.arange(N + 1)
    match = np.r_[z[1:N], z[N]]  # interior nodes + upper end
    xi, wi = _gauss(n_gauss)
    freqs = np.atleast_1d(np.asarray(freq, float))
    out_z, out_I = [], []
    # analytic part, independent of k: int T_i(z') / R dz'
    S = np.zeros((N, N - 1))
    for i in range(1, N):
        for side, (z0, z1) in enumerate(((z[i - 1], z[i]), (z[i], z[i + 1]))):
            u0, u1 = z0 - match, z1 - match  # u = z' - z_m

            def F(u, zl):
                return np.sqrt(u**2 + a**2) + (match - zl) * np.arcsinh(u / a)

            if side == 0:  # rising: T = (z' - z_{i-1}) / d
                S[:, i - 1] += (F(u1, z0) - F(u0, z0)) / d
            else:  # falling: T = (z_{i+1} - z') / d = 1 - (z' - z_i) / d
                S[:, i - 1] += np.arcsinh(u1 / a) - np.arcsinh(u0 / a)
                S[:, i - 1] -= (F(u1, z0) - F(u0, z0)) / d
    # quadrature nodes of the smooth part
    zq, Tq, iq = [], [], []
    for i in range(1, N):
        zr = z[i - 1] + d * xi
        zf = z[i] + d * xi
        zq += [zr, zf]
        Tq += [xi * wi * d, (1 - xi) * wi * d]
        iq += [np.full(xi.size, i - 1)] * 2
    zq, Tq, iq = np.concatenate(zq), np.concatenate(Tq), np.concatenate(iq)
    R = np.sqrt((match[:, None] - zq[None, :]) ** 2 + a**2)
    for f in freqs:
        k = 2 * np.pi * f / c_light
        G = (np.exp(-1j * k * R) - 1) / R * Tq[None, :]
        A = S.astype(complex)
        for col in range(N - 1):
            A[:, col] += G[:, iq == col].sum(1)
        M = np.zeros((N, N), complex)
        M[:, : N - 1] = A
        M[:, N - 1] = -np.cos(k * match)
        rhs = -1j * 4 * np.pi / ETA0 * gap_excitation(match, k, gap)
        sol = np.linalg.solve(M, rhs)
        I = np.zeros(N + 1, complex)
        I[1:N] = sol[: N - 1]
        out_I.append(I)
        out_z.append(1.0 / I[N // 2])
    return np.array(out_z), z, np.array(out_I)


def dipole_far_field(z_nodes, current, freq, theta):
    """
    Far-field pattern of a z-directed line current (piecewise linear).

    Returns the radiation intensity U(theta) [W/sr] (exp(+j w t), peak
    phasors) and the radiated power P_rad [W].
    """
    k = 2 * np.pi * freq / c_light
    theta = np.asarray(theta, float)
    # piecewise-linear current, integrate with fine trapezoid
    zf = np.linspace(z_nodes[0], z_nodes[-1], 20 * (len(z_nodes) - 1) + 1)
    If = np.interp(zf, z_nodes, current.real) + 1j * np.interp(
        zf, z_nodes, current.imag
    )
    ph = np.exp(1j * k * np.outer(np.cos(theta), zf))
    Ft = np.trapezoid(ph * If[None, :], zf, axis=1)
    U = ETA0 * k**2 / (32 * np.pi**2) * np.abs(np.sin(theta) * Ft) ** 2
    tq = np.linspace(0, np.pi, 721)
    Fq = np.trapezoid(
        np.exp(1j * k * np.outer(np.cos(tq), zf)) * If[None, :], zf, axis=1
    )
    Uq = ETA0 * k**2 / (32 * np.pi**2) * np.abs(np.sin(tq) * Fq) ** 2
    P = 2 * np.pi * np.trapezoid(Uq * np.sin(tq), tq)
    return U, P


def induced_emf_impedance(length, radius, freq):
    """
    Induced-EMF self impedance of a dipole with a sinusoidal current,
    referred to the current maximum (Balanis eqs. 8-60a,b), and transferred
    to the input terminals by dividing by sin^2(k L / 2).

    For L = lambda / 2 and a -> 0: 73.1 + j 42.5 Ohm.
    """
    k = 2 * np.pi * np.asarray(freq, float) / c_light
    kl = k * length
    si1, ci1 = sici(kl)
    si2, ci2 = sici(2 * kl)
    _, ci3 = sici(2 * k * radius**2 / length)
    R = ETA0 / (2 * np.pi) * (
        EULER + np.log(kl) - ci1 + 0.5 * np.sin(kl) * (si2 - 2 * si1)
        + 0.5 * np.cos(kl) * (EULER + np.log(kl / 2) + ci2 - 2 * ci1)
    )  # fmt: skip
    X = ETA0 / (4 * np.pi) * (
        2 * si1 + np.cos(kl) * (2 * si1 - si2)
        - np.sin(kl) * (2 * ci1 - ci2 - ci3)
    )  # fmt: skip
    return (R + 1j * X) / np.sin(kl / 2) ** 2
