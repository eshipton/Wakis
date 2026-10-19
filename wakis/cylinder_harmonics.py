# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Cylindrical-harmonic analysis of 2D scattered fields sampled on circles.

A scalar 2D field u (E_z for TM, H_z for TE) that satisfies the Helmholtz
equation in a source-free annulus a < r < R is, at one frequency,

    u(r, theta) = sum_n [alpha_n H_n^(1)(k r) + gamma_n H_n^(2)(k r)] e^{i n theta}

in the exp(-i omega t) convention of Bohren & Huffman: alpha_n are the
outgoing (scattered) waves and gamma_n incoming waves, e.g. reflections from
an imperfect outer boundary. Sampling u on two or more concentric circles,
projecting on e^{i n theta} (FFT) and fitting the radial dependence in least
squares separates the two.

For a plane wave u_inc = U0 e^{i k x} = U0 sum_n i^n J_n(k r) e^{i n theta}
scattered by a circular cylinder centred on the origin, the exact scattered
field is u_s = -U0 sum_n i^n c_n H_n^(1)(k r) e^{i n theta} with c_n = b_n
(TM) or a_n (TE) of `wakis.analytic.mie_cylinder`. Writing the total field
outside the cylinder as incoming plus outgoing waves,

    I_n = i^n U0 / 2 + gamma_n,     O_n = i^n U0 / 2 + alpha_n,

the cylinder's scattering matrix S_n = O_n / I_n = 1 - 2 c_n holds whatever
the incoming waves are, so

    c_n = (1 - O_n / I_n) / 2

de-embeds the reflections of the outer boundary (which re-illuminate the
cylinder), while the naive estimate c_n = -alpha_n / (i^n U0) ignores them.
Q_sca = (2 / x) sum_{n=-N..N} |c_n|^2 with x = k a.

The DFTs of Wakis (kernel exp(-2 pi i f t)) are phasors of the exp(+i omega t)
convention; pass ``convention='wakis'`` (default) to conjugate them.
"""

import numpy as np
from scipy.special import hankel1, jv

__all__ = [
    "circle_points",
    "project_harmonics",
    "fit_harmonics",
    "mie_coefficients_from_fit",
    "q_sca_from_coefficients",
]


def circle_points(radius, n_points, center=(0.0, 0.0)):
    """
    Uniformly spaced points on a circle.

    Parameters
    ----------
    radius : float
        Radius [m].
    n_points : int
        Number of points (use > 2 N_max + 1 for harmonics up to N_max).
    center : (2,) float
        Centre [m].

    Returns
    -------
    points : ndarray (n_points, 2)
    theta : ndarray (n_points,)
        Polar angles [rad], theta_m = 2 pi m / n_points.
    """
    th = 2 * np.pi * np.arange(n_points) / n_points
    pts = np.stack([np.cos(th), np.sin(th)], 1) * radius + np.asarray(center)
    return pts, th


def project_harmonics(samples, n_max, convention="wakis"):
    """
    Angular Fourier coefficients F_n = (1/M) sum_m u(theta_m) e^{-i n theta_m}.

    Parameters
    ----------
    samples : ndarray (..., M)
        Complex field on M equispaced angles (see `circle_points`).
    n_max : int
        Highest |n| returned (must be < M / 2).
    convention : {'wakis', 'bh'}
        'wakis': phasors of exp(+i omega t) (Wakis DFTs), conjugated here;
        'bh': already exp(-i omega t).

    Returns
    -------
    ndarray (..., 2 n_max + 1)
        F_n for n = -n_max ... n_max.
    """
    samples = np.asarray(samples)
    M = samples.shape[-1]
    if 2 * n_max + 1 > M:
        raise ValueError("n_max too large for the number of samples")
    if convention == "wakis":
        samples = samples.conj()
    elif convention != "bh":
        raise ValueError("convention must be 'wakis' or 'bh'")
    F = np.fft.fft(samples, axis=-1) / M
    n = np.arange(-n_max, n_max + 1)
    return F[..., n % M]


def fit_harmonics(F, k, radii, n_max, j_floor=1e-3):
    """
    Outgoing/incoming amplitudes from harmonics on several radii.

    Fits F_n(r_j) = A_n H_n^(1)(k r_j) + B_n J_n(k r_j) in least squares
    (columns scaled to unit norm), then alpha_n = A_n + B_n / 2 and
    gamma_n = B_n / 2 (J = (H^(1) + H^(2)) / 2). Orders whose regular part
    is negligible on the circles (max_j |J_n(k r_j)| < `j_floor`
    max_j |H_n^(1)(k r_j)|) cannot be separated: gamma_n = 0 there.

    Parameters
    ----------
    F : ndarray (n_freq, n_radii, 2 n_max + 1)
        Harmonics from `project_harmonics` (exp(-i omega t)).
    k : ndarray (n_freq,)
        Wavenumber in the background [1/m].
    radii : sequence of float
        Circle radii [m] (at least 2).
    n_max : int

    Returns
    -------
    alpha, gamma : ndarray (n_freq, 2 n_max + 1)
    resid : ndarray (n_freq,)
        Relative least-squares residual (consistency of the model).
    """
    F = np.asarray(F)
    radii = np.asarray(radii, float)
    k = np.atleast_1d(k)
    n = np.arange(-n_max, n_max + 1)
    alpha = np.zeros(F.shape[::2], complex)
    gamma = np.zeros_like(alpha)
    resid = np.zeros(F.shape[0])
    for i, kk in enumerate(k):
        kr = kk * radii
        num = den = 0.0
        for jn, nn in enumerate(n):
            H = hankel1(nn, kr)
            J = jv(nn, kr)
            y = F[i, :, jn]
            if np.abs(J).max() < j_floor * np.abs(H).max() or len(radii) < 2:
                A = np.vdot(H, y) / np.vdot(H, H)
                B = 0.0
                r = y - A * H
            else:
                Mx = np.stack([H / np.linalg.norm(H), J / np.linalg.norm(J)], 1)
                sol, *_ = np.linalg.lstsq(Mx, y, rcond=None)
                A = sol[0] / np.linalg.norm(H)
                B = sol[1] / np.linalg.norm(J)
                r = y - Mx @ sol
            alpha[i, jn] = A + B / 2
            gamma[i, jn] = B / 2
            num += np.sum(np.abs(r) ** 2)
            den += np.sum(np.abs(y) ** 2)
        resid[i] = np.sqrt(num / max(den, 1e-300))
    return alpha, gamma, resid


def mie_coefficients_from_fit(alpha, gamma, u0, n_max, deembed=True):
    """
    Scattering coefficients c_n (b_n for TM, a_n for TE) of a centred cylinder.

    Parameters
    ----------
    alpha, gamma : ndarray (n_freq, 2 n_max + 1)
        From `fit_harmonics`.
    u0 : ndarray (n_freq,)
        Incident-field phasor at the origin (exp(-i omega t), i.e. the
        conjugate of a Wakis DFT).
    deembed : bool
        Use the scattering-matrix relation c_n = (1 - O_n / I_n) / 2 (removes
        the re-illumination by boundary reflections); False: -alpha_n /
        (i^n U0).

    Returns
    -------
    ndarray (n_freq, 2 n_max + 1)
        c_n for n = -n_max ... n_max.
    """
    n = np.arange(-n_max, n_max + 1)
    inc = (1j**n)[None, :] * np.asarray(u0)[:, None] / 2
    if not deembed:
        return -alpha / (2 * inc)
    return 0.5 * (1 - (inc + alpha) / (inc + gamma))


def q_sca_from_coefficients(c, x):
    """Q_sca = (2 / x) sum_{n=-N..N} |c_n|^2 for each frequency."""
    return 2 / np.asarray(x) * np.sum(np.abs(c) ** 2, axis=-1)
