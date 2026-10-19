# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Cross sections of 2D scatterers from scattered fields sampled on circles.

An arm delivers the scattered field u_s (E_z for TM, H_z for TE) as Wakis
DFT phasors (exp(+i w t)) on concentric circles centred on the origin
(`wakis.cylinder_harmonics.circle_points`, M points each) and the incident
phasor U0 at the origin. With the outgoing/incoming harmonic fit of
`wakis.cylinder_harmonics.fit_harmonics` (amplitudes alpha_n, gamma_n in
exp(-i w t)), the generalised scattering coefficients for a plane wave
travelling at angle phi are

    c_n = -alpha_n / I_n,   I_n = U0 i^n e^{-i n phi},

so that, for any scatterer (Bohren & Huffman ch. 8 for one cylinder; the
optical theorem for the general case, see `wakis.analytic.multi_cylinder`),

    C_ext = (4 / k) Re sum_n c_n,   C_sca = (4 / k) sum_n |c_n|^2.

For a rotationally symmetric scatterer centred on the origin the
scattering matrix is diagonal, and the reflections of the outer boundary
(incoming waves gamma_n) are removed exactly by c_n = (1 - O_n / I'_n) / 2
with I'_n = I_n / 2 + gamma_n, O_n = I_n / 2 + alpha_n (``deembed=True``).
Other scatterers need an absorber good enough that the naive estimate is
accurate; `boundary_reflection` reports |gamma| / |alpha| as a diagnostic.
"""

import numpy as np
from scipy.constants import c as c_light

from ..cylinder_harmonics import fit_harmonics, project_harmonics

__all__ = ["cross_sections_from_circles", "harmonic_order"]


def harmonic_order(k_max, r_max):
    """Harmonics needed on circles up to r_max: k r + 4 (k r)^(1/3) + 6."""
    x = k_max * r_max
    return int(np.ceil(x + 4 * x ** (1 / 3) + 6))


def cross_sections_from_circles(
    samples, u0, freqs, radii, n_background=1.0, direction=0.0, n_max=None,
    deembed=False,
):  # fmt: skip
    """
    Extinction, scattering and absorption widths from circle samples.

    Parameters
    ----------
    samples : ndarray (n_freq, n_radii, M)
        Wakis-convention DFT phasors of the scattered u on the circles.
    u0 : ndarray (n_freq,)
        Wakis-convention incident phasor at the origin.
    freqs : ndarray (n_freq,)
        Frequencies [Hz].
    radii : sequence of float
        Circle radii [m].
    n_background : float
        Refractive index of the lossless background.
    direction : float
        Propagation angle of the plane wave [rad].
    n_max : int, optional
        Highest harmonic fitted (default from `harmonic_order`).
    deembed : bool
        Use the diagonal S-matrix de-embedding (only valid for a rotationally
        symmetric scatterer centred on the origin).

    Returns
    -------
    dict
        ``c_ext``, ``c_sca``, ``c_abs`` widths [m] (n_freq,), ``refl``
        (|gamma| / |alpha|, boundary reflection), ``resid`` (fit residual),
        ``c`` (n_freq, 2 n_max + 1) coefficients.
    """
    samples = np.asarray(samples)
    freqs = np.asarray(freqs, float)
    radii = np.asarray(radii, float)
    k = 2 * np.pi * freqs * n_background / c_light
    if n_max is None:
        n_max = min(harmonic_order(k.max(), radii.max()), samples.shape[-1] // 2 - 1)
    F = project_harmonics(samples, n_max, convention="wakis")
    alpha, gamma, resid = fit_harmonics(F, k, radii, n_max)
    n = np.arange(-n_max, n_max + 1)
    U0 = np.conj(np.asarray(u0))[:, None]
    I = U0 * (1j**n * np.exp(-1j * n * direction))[None, :]
    if deembed:
        c = 0.5 * (1 - (I / 2 + alpha) / (I / 2 + gamma))
    else:
        c = -alpha / I
    c_ext = 4 / k * np.real(np.sum(c, 1))
    c_sca = 4 / k * np.sum(np.abs(c) ** 2, 1)
    refl = np.sqrt(np.sum(np.abs(gamma) ** 2, 1) / np.sum(np.abs(alpha) ** 2, 1))
    return dict(
        c_ext=c_ext, c_sca=c_sca, c_abs=c_ext - c_sca, refl=refl, resid=resid, c=c
    )
