# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Closed-form design formulas used by the templates and their validation.

These are textbook engineering approximations, not exact solutions: the
transmission-line model of the patch antenna is typically within a few
percent of measured resonances, the induced-EMF impedance of a dipole assumes
a sinusoidal current on an infinitely thin wire, etc. See the physics guide
(section "Design templates") for the references and validity ranges.
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0, mu_0
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import j0, sici

ETA0 = np.sqrt(mu_0 / epsilon_0)
EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------- patch (TLM)
def patch_effective_permittivity(width, height, eps_r):
    """
    Effective permittivity of a microstrip of width W on a substrate of
    height h (W/h > 1): (eps_r+1)/2 + (eps_r-1)/2 (1 + 12 h/W)^(-1/2).
    """
    return (eps_r + 1) / 2 + (eps_r - 1) / 2 / np.sqrt(1 + 12 * height / width)


def patch_length_extension(width, height, eps_r):
    """
    Fringing-field length extension Delta L [m] of each radiating edge
    (Hammerstad): 0.412 h (eps_eff+0.3)(W/h+0.264) / ((eps_eff-0.258)(W/h+0.8)).
    """
    eps_eff = patch_effective_permittivity(width, height, eps_r)
    u = width / height
    return (
        0.412 * height * (eps_eff + 0.3) * (u + 0.264) / ((eps_eff - 0.258) * (u + 0.8))
    )


def patch_design(frequency, eps_r, height):
    """
    Rectangular patch dimensions for resonance at `frequency` (TM010 mode),
    transmission-line model.

    W = c / (2 f) sqrt(2 / (eps_r + 1)),  L = c / (2 f sqrt(eps_eff)) - 2 Delta L.

    Parameters
    ----------
    frequency : float
        Design frequency [Hz].
    eps_r : float
        Substrate relative permittivity.
    height : float
        Substrate thickness [m].

    Returns
    -------
    dict
        'width', 'length', 'eps_eff', 'delta_l' [m or -].
    """
    width = c_light / (2 * frequency) * np.sqrt(2 / (eps_r + 1))
    eps_eff = patch_effective_permittivity(width, height, eps_r)
    delta_l = patch_length_extension(width, height, eps_r)
    length = c_light / (2 * frequency * np.sqrt(eps_eff)) - 2 * delta_l
    return dict(width=width, length=length, eps_eff=eps_eff, delta_l=delta_l)


def patch_resonance(length, width, eps_r, height):
    """
    TM010 resonant frequency [Hz] of a patch of given length (along the
    current) and width, f = c / (2 (L + 2 Delta L) sqrt(eps_eff)).
    """
    eps_eff = patch_effective_permittivity(width, height, eps_r)
    delta_l = patch_length_extension(width, height, eps_r)
    return c_light / (2 * (length + 2 * delta_l) * np.sqrt(eps_eff))


def _slot_integrand(theta, k0, width, k0_spacing=None):
    x = k0 * width / 2 * np.cos(theta)
    # sin(x)/cos(theta) without the removable singularity at theta = pi/2
    factor = (k0 * width / 2 * np.sinc(x / np.pi)) ** 2 * np.sin(theta) ** 3
    if k0_spacing is not None:
        factor *= j0(k0_spacing * np.sin(theta))
    return factor


def patch_edge_resistance(length, width, frequency):
    """
    Input resistance [Ohm] at the radiating edge, R(0) = 1 / (2 (G1 + G12)),
    with the slot self-conductance G1 and mutual conductance G12 of the
    transmission-line model:

    G1 = I1 / (120 pi^2),  I1 = int_0^pi [sin(k0 W cos t / 2) / cos t]^2 sin^3 t dt
    G12 = 1 / (120 pi^2) int_0^pi [...]^2 J0(k0 L sin t) sin^3 t dt.
    """
    k0 = 2 * np.pi * frequency / c_light
    I1 = quad(_slot_integrand, 0, np.pi, args=(k0, width))[0]
    I12 = quad(_slot_integrand, 0, np.pi, args=(k0, width, k0 * length))[0]
    G1, G12 = I1 / (120 * np.pi**2), I12 / (120 * np.pi**2)
    return 1 / (2 * (G1 + G12))


def patch_feed_offset(length, edge_resistance, z0=50.0):
    """
    Distance [m] of the feed from the radiating edge for an input resistance
    z0, from R(y0) = R(0) cos^2(pi y0 / L) (cavity / TLM model).
    """
    if z0 >= edge_resistance:
        return 0.0
    return length / np.pi * np.arccos(np.sqrt(z0 / edge_resistance))


def patch_directivity(length, width, frequency):
    """
    Directivity (linear) of the two-slot transmission-line model over an
    infinite ground plane: D = D0 * 2 / (1 + g12), with the single-slot
    directivity D0 = (2 pi W / lambda0)^2 / I1 and g12 = G12 / G1 (integrals
    of `patch_edge_resistance`). Typical values are 5-7 dBi, increasing with
    W / lambda0 (i.e. decreasing with eps_r).
    """
    k0 = 2 * np.pi * frequency / c_light
    I1 = quad(_slot_integrand, 0, np.pi, args=(k0, width))[0]
    I12 = quad(_slot_integrand, 0, np.pi, args=(k0, width, k0 * length))[0]
    D0 = (k0 * width) ** 2 / I1
    return D0 * 2 / (1 + I12 / I1)


# --------------------------------------------------------------------- dipole
def dipole_impedance(length, radius, frequency):
    """
    Input impedance [Ohm] of a centre-fed thin dipole by the induced-EMF
    method (sinusoidal current, valid for radius << length, lambda).

    R_in = R_r / sin^2(kl/2) and X_in = X_m / sin^2(kl/2) with
    R_r = eta/(2 pi) {C + ln(kl) - Ci(kl) + sin(kl)/2 [Si(2kl) - 2 Si(kl)]
          + cos(kl)/2 [C + ln(kl/2) + Ci(2kl) - 2 Ci(kl)]},
    X_m = eta/(4 pi) {2 Si(kl) + cos(kl) [2 Si(kl) - Si(2kl)]
          - sin(kl) [2 Ci(kl) - Ci(2kl) - Ci(2 k a^2 / l)]}.
    For l = lambda/2 this gives 73.1 + 42.5j Ohm.
    """
    kl = 2 * np.pi * np.asarray(frequency, dtype=float) / c_light * length
    si1, ci1 = sici(kl)
    si2, ci2 = sici(2 * kl)
    _, ci_a = sici(2 * kl * radius**2 / length**2)
    C = EULER_GAMMA
    R_r = (
        ETA0
        / (2 * np.pi)
        * (
            C
            + np.log(kl)
            - ci1
            + 0.5 * np.sin(kl) * (si2 - 2 * si1)
            + 0.5 * np.cos(kl) * (C + np.log(kl / 2) + ci2 - 2 * ci1)
        )
    )
    X_m = (
        ETA0
        / (4 * np.pi)
        * (2 * si1 + np.cos(kl) * (2 * si1 - si2) - np.sin(kl) * (2 * ci1 - ci2 - ci_a))
    )
    return (R_r + 1j * X_m) / np.sin(kl / 2) ** 2


# ------------------------------------------------------------------ waveguide
def te10_cutoff(a):
    """Cutoff frequency [Hz] of the TE10 mode of a guide of broad wall a [m]."""
    return c_light / (2 * a)


def te10_beta(a, frequency):
    """Propagation constant [1/m] of TE10, sqrt(k^2 - (pi/a)^2)."""
    k = 2 * np.pi * np.asarray(frequency) / c_light
    return np.sqrt(k**2 - (np.pi / a) ** 2 + 0j)


def inductive_iris_s11(a, opening, frequency):
    """
    Approximate reflection coefficient of a thin symmetric inductive
    diaphragm (opening d centred in the broad wall a) in TE10, from its
    normalized shunt susceptance B/Y0 = -(lambda_g / a) cot^2(pi d / (2 a)):
    S11 = -j b / (2 + j b). Small-aperture approximation (a few percent to
    ~10 % in |S11|).
    """
    beta = te10_beta(a, frequency).real
    b = -(2 * np.pi / beta) / a / np.tan(np.pi * opening / (2 * a)) ** 2
    return -1j * b / (2 + 1j * b)


# ---------------------------------------------------------------- thin films
def slab_reflection_transmission(eps_r, thickness, frequency, n_out=1.0):
    """
    Normal-incidence power reflectance and transmittance of a homogeneous
    slab (Airy formulas, characteristic-matrix form).

    Parameters
    ----------
    eps_r : complex or array_like
        Slab relative permittivity (exp(+j w t) convention: Im <= 0 is lossy).
    thickness : float
        Slab thickness [m].
    frequency : array_like
        Frequencies [Hz].
    n_out : float, optional
        Refractive index of the medium on both sides. Default 1.

    Returns
    -------
    R, T : ndarray
        Power reflectance and transmittance. For a lossless slab in vacuum
        T = 1 / (1 + (n - 1/n)^2 sin^2(k0 n d) / 4) and R = 1 - T.
    """
    n = np.sqrt(np.asarray(eps_r, dtype=complex))
    n = np.where(n.imag > 0, -n, n)  # passive branch for exp(+jwt)
    delta = 2 * np.pi * np.asarray(frequency) / c_light * n * thickness
    m11, m22 = np.cos(delta), np.cos(delta)
    m12, m21 = 1j * np.sin(delta) / n, 1j * n * np.sin(delta)
    den = n_out * m11 + n_out**2 * m12 + m21 + n_out * m22
    r = (n_out * m11 + n_out**2 * m12 - m21 - n_out * m22) / den
    t = 2 * n_out / den
    return np.abs(r) ** 2, np.abs(t) ** 2


# --------------------------------------------------------------- slab guides
def slab_waveguide_neff(n_core, n_clad, thickness, wavelength, polarization="TE"):
    """
    Effective index of the fundamental mode of a symmetric slab waveguide,
    root of tan(kappa d / 2) = rho gamma / kappa with kappa = k0 sqrt(n1^2 -
    neff^2), gamma = k0 sqrt(neff^2 - n2^2), rho = 1 (TE) or n1^2 / n2^2 (TM).
    """
    k0 = 2 * np.pi / wavelength
    rho = 1.0 if polarization.upper() == "TE" else (n_core / n_clad) ** 2

    def mismatch(neff):
        kappa = k0 * np.sqrt(n_core**2 - neff**2)
        gamma = k0 * np.sqrt(neff**2 - n_clad**2)
        return np.tan(kappa * thickness / 2) - rho * gamma / kappa

    # First branch: kappa d / 2 < pi / 2
    neff_lo = np.sqrt(max(n_core**2 - (np.pi / (k0 * thickness)) ** 2, n_clad**2))
    return brentq(mismatch, neff_lo + 1e-12, n_core - 1e-12)
