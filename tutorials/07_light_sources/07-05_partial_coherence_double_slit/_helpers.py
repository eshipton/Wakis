# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Helper functions for Tutorial 7.5: Young's double slit with partially coherent
light (van Cittert-Zernike extended incoherent sources, Gaussian Schell-model
Mercer coherent-mode decomposition, and broadband temporal coherence in 2D FIT).
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.special import eval_hermite, factorial, hankel2

from wakis import GridFIT3D, SolverFIT3D
from wakis.dispersive import DispersiveMaterial, Drude, dispersive_lib
from wakis.templates.meshing import graded_axis
from wakis.templates.shapes import Box, assign_materials


# ---------------------------------------------------------------------------
# 1. Analytical Coherence & 2D Diffraction Benchmarks
# ---------------------------------------------------------------------------


def van_cittert_zernike_sinc(D, d, wavelength=0.5e-6, L1=8.0e-6):
    """
    Paraxial van Cittert-Zernike complex degree of spatial coherence between
    two slits separated by ``d`` illuminated by a uniform incoherent strip of
    width ``D`` at upstream distance ``L1``.

    Parameters
    ----------
    D : float or array_like
        Incoherent primary source width [m].
    d : float
        Center-to-center slit separation [m].
    wavelength : float, optional
        Center vacuum wavelength [m]. Default 0.5e-6.
    L1 : float, optional
        Propagation distance from primary source plane to slit plane [m].
        Default 8.0e-6.

    Returns
    -------
    mu_12 : float or ndarray
        Signed real degree of spatial coherence ``sinc(pi * D * d / (wavelength * L1))``
        (dimensionless, in ``[-0.2172, 1.0]``). Note ``np.sinc(u) = sin(pi*u)/(pi*u)``.
    """
    u = np.asarray(D, dtype=float) * float(d) / (float(wavelength) * float(L1))
    res = np.sinc(u)
    return float(res) if np.ndim(D) == 0 else res


def van_cittert_zernike_2d_hankel(D, d, wavelength=0.5e-6, L1=8.0e-6, n_quad=201):
    """
    Exact 2D cylindrical-wave (Hankel H_0^(2)) complex degree of spatial
    coherence between two slits at ``(+-d/2, 0)`` illuminated by a continuous
    uniform incoherent strip of width ``D`` at ``y = -L1``.

    Parameters
    ----------
    D : float or array_like
        Incoherent primary source width [m].
    d : float
        Center-to-center slit separation [m].
    wavelength : float, optional
        Vacuum wavelength [m]. Default 0.5e-6.
    L1 : float, optional
        Upstream distance [m]. Default 8.0e-6.
    n_quad : int, optional
        Number of midpoint quadrature nodes across ``[-D/2, +D/2]``. Default 201.

    Returns
    -------
    mu_12 : float or ndarray
        Real part of the normalized mutual intensity ``W(-d/2, +d/2) / I(-d/2)``
        (dimensionless).
    """
    k0 = 2.0 * np.pi / float(wavelength)
    d_arr = np.atleast_1d(np.asarray(D, dtype=float))
    u = (np.arange(n_quad, dtype=float) + 0.5) / n_quad - 0.5
    out = np.empty_like(d_arr, dtype=float)
    for idx, width in enumerate(d_arr):
        if width <= 1e-15:
            out[idx] = 1.0
            continue
        xs = u * width
        r1 = np.hypot(xs + 0.5 * d, L1)
        r2 = np.hypot(xs - 0.5 * d, L1)
        e1 = hankel2(0, k0 * r1)
        e2 = hankel2(0, k0 * r2)
        denom = np.sqrt(np.mean(np.abs(e1) ** 2) * np.mean(np.abs(e2) ** 2))
        out[idx] = float(np.real(np.mean(np.conj(e1) * e2)) / denom)
    return float(out[0]) if np.ndim(D) == 0 else out


def gsm_visibility_analytic(d, sigma_mu):
    """
    Analytical fringe visibility ``V = |mu_12(d)| = exp(-d^2 / (2 * sigma_mu^2))``
    for a Gaussian Schell-Model (GSM) beam illuminating two symmetric slits.

    Parameters
    ----------
    d : float
        Center-to-center slit separation [m].
    sigma_mu : float or array_like
        RMS transverse spatial coherence width [m] at the slit plane.

    Returns
    -------
    visibility : float or ndarray
        Michelson fringe visibility on the optical axis (dimensionless, in ``[0, 1]``).
    """
    s = np.asarray(sigma_mu, dtype=float)
    with np.errstate(divide="ignore"):
        v = np.where(np.isinf(s), 1.0, np.exp(-0.5 * (float(d) / s) ** 2))
    return float(v) if np.ndim(sigma_mu) == 0 else v


def temporal_coherence_envelope(delta_L, wavelength_0=0.5e-6, rel_bw_sigma=0.08):
    """
    Wiener-Khinchin temporal coherence degree ``|gamma_11(tau)| = exp(-delta_L^2 / (2 * L_c^2))``
    for a Gaussian power spectrum with RMS fractional bandwidth ``sigma_f / f_0``.

    Parameters
    ----------
    delta_L : float or array_like
        Optical path difference ``r_1(x) - r_2(x)`` [m].
    wavelength_0 : float, optional
        Center wavelength [m]. Default 0.5e-6.
    rel_bw_sigma : float, optional
        Dimensionless RMS fractional bandwidth ``sigma_f / f_0``. Default 0.08.

    Returns
    -------
    gamma_abs : float or ndarray
        Magnitude of the complex degree of temporal coherence (dimensionless).
    L_c : float
        Longitudinal coherence length ``L_c = wavelength_0 / (2 * pi * rel_bw_sigma)`` [m].
    """
    if rel_bw_sigma <= 0.0:
        dl = np.asarray(delta_L, dtype=float)
        ones = np.ones_like(dl)
        return (1.0 if np.ndim(delta_L) == 0 else ones), np.inf
    L_c = float(wavelength_0) / (2.0 * np.pi * float(rel_bw_sigma))
    dl = np.asarray(delta_L, dtype=float)
    env = np.exp(-0.5 * (dl / L_c) ** 2)
    return (float(env) if np.ndim(delta_L) == 0 else env), L_c


def double_slit_2d_hankel_intensity(
    x_det,
    mu_12,
    d=2.0e-6,
    a=0.4e-6,
    L2=6.0e-6,
    wavelength=0.5e-6,
    n_slit=21,
):
    """
    2D cylindrical-wave (Hankel H_0^(2)) diffraction pattern on an observation
    screen at ``y = L2`` produced by two slits of width ``a`` at ``x = +-d/2``
    with mutual coherence ``mu_12``.

    Parameters
    ----------
    x_det : array_like
        Transverse detector coordinates [m].
    mu_12 : float or complex
        Complex degree of spatial coherence between slit 1 and slit 2.
    d : float, optional
        Slit separation [m]. Default 2.0e-6.
    a : float, optional
        Individual slit width [m]. Default 0.4e-6.
    L2 : float, optional
        Slit-to-screen distance [m]. Default 6.0e-6.
    wavelength : float, optional
        Vacuum wavelength [m]. Default 0.5e-6.
    n_slit : int, optional
        Quadrature points across each slit opening. Default 21.

    Returns
    -------
    intensity : ndarray
        Normalized screen intensity profile ``I(x_det) / max(I_coherent)`` (dimensionless).
    envelope : ndarray
        Normalized incoherent single-slit sum ``(I_1(x) + I_2(x)) / max(I_coherent)``.
    """
    x_det = np.asarray(x_det, dtype=float)
    k0 = 2.0 * np.pi / float(wavelength)
    u = np.linspace(-0.5 * a, 0.5 * a, n_slit)
    w_u = np.cos(np.pi * u / a)
    e1 = np.sum(
        w_u[:, None] * hankel2(0, k0 * np.hypot(x_det[None, :] - (-0.5 * d + u[:, None]), L2)),
        axis=0,
    )
    e2 = np.sum(
        w_u[:, None] * hankel2(0, k0 * np.hypot(x_det[None, :] - (+0.5 * d + u[:, None]), L2)),
        axis=0,
    )
    i_coh_0 = float(np.abs(e1[len(x_det) // 2] + e2[len(x_det) // 2]) ** 2) or 1.0
    i_inc = (np.abs(e1) ** 2 + np.abs(e2) ** 2) / i_coh_0
    cross = 2.0 * np.real(mu_12 * np.conj(e1) * e2) / i_coh_0
    return i_inc + cross, i_inc


# ---------------------------------------------------------------------------
# 2. Gaussian Schell-Model (GSM) Coherent-Mode (Mercer) Decomposition
# ---------------------------------------------------------------------------


def gsm_hermite_mode(m, x, w_m=1.4e-6):
    """
    Orthonormal 1D Hermite-Gaussian Mercer eigenmode ``Phi_m(x)`` of a Gaussian
    Schell-Model source (Starikov & Wolf, J. Opt. Soc. Am. 72, 923-928, 1982).

    Satisfies the L2 orthonormality relation ``int_{-inf}^{inf} Phi_m(x) Phi_n(x) dx = delta_{mn}``.

    Parameters
    ----------
    m : int
        Mode order ``m >= 0``.
    x : array_like
        Transverse coordinate [m].
    w_m : float, optional
        Modal scale parameter ``w_m = 1 / sqrt(2*c)`` [m]. Default 1.4e-6.

    Returns
    -------
    phi_m : ndarray
        Real mode profile ``Phi_m(x)`` [m^(-1/2)].
    """
    m = int(m)
    if m < 0:
        raise ValueError("Hermite-Gaussian mode order m must be >= 0.")
    x_arr = np.asarray(x, dtype=float)
    xi = x_arr / float(w_m)
    norm = (np.pi * float(w_m) ** 2) ** (-0.25) / np.sqrt((2.0**m) * float(factorial(m)))
    return norm * eval_hermite(m, xi) * np.exp(-0.5 * xi**2)


def gsm_mercer_params(sigma_s, sigma_mu, n_modes=10, i0=1.0):
    """
    Compute the exact Starikov & Wolf (1982) Mercer coherent-mode parameters
    for a 1D Gaussian Schell-Model source with RMS intensity width ``sigma_s``
    and RMS coherence width ``sigma_mu``.

    Parameters
    ----------
    sigma_s : float
        RMS beam intensity width [m] (where ``I(x) = i0 * exp(-x^2 / (2*sigma_s^2))``).
    sigma_mu : float
        RMS spatial coherence width [m] (where ``mu(dx) = exp(-dx^2 / (2*sigma_mu^2))``).
    n_modes : int, optional
        Number of Mercer eigenvalues to return. Default 10.
    i0 : float, optional
        Peak on-axis intensity. Default 1.0.

    Returns
    -------
    params : dict
        Dictionary with keys ``a``, ``b``, ``c`` [m^-2], ``w_m`` [m], geometric
        ratio ``q``, eigenvalues ``lambdas``, normalized weights ``weights``,
        global coherence parameter ``beta_c = sigma_mu / sigma_s``, and purity
        ``purity = (1 - q) / (1 + q)``.
    """
    if sigma_s <= 0 or sigma_mu <= 0:
        raise ValueError("sigma_s and sigma_mu must be > 0.")
    a = 1.0 / (4.0 * float(sigma_s) ** 2)
    b = 0.0 if np.isinf(sigma_mu) else 1.0 / (2.0 * float(sigma_mu) ** 2)
    c = float(np.sqrt(a**2 + 2.0 * a * b))
    w_m = 1.0 / float(np.sqrt(2.0 * c))
    q = float(b / (a + b + c)) if b > 0 else 0.0
    m_idx = np.arange(int(n_modes), dtype=float)
    lam0 = float(i0) * float(np.sqrt(np.pi / (a + b + c)))
    lambdas = lam0 * (q**m_idx)
    weights = (1.0 - q) * (q**m_idx)
    return {
        "a": a,
        "b": b,
        "c": c,
        "w_m": w_m,
        "q": q,
        "lambdas": lambdas,
        "weights": weights,
        "beta_c": float(sigma_mu) / float(sigma_s),
        "purity": (1.0 - q) / (1.0 + q),
        "truncation_error": float(q ** int(n_modes)),
    }


def gsm_parameters_from_modal_scale(sigma_mu, w_m=1.4e-6, n_modes=10):
    """
    Mehler-kernel parameterization of a Gaussian Schell-Model beam sharing a
    fixed Hermite-Gaussian modal scale ``w_m = 1 / sqrt(2*c)`` across arbitrary
    spatial coherence widths ``sigma_mu``.

    By Mehler's formula, fixing ``w_m`` allows a single bank of simulated
    Hermite-Gaussian modes ``Phi_m(x)`` to synthesize exact GSM beams for any
    ``sigma_mu > 0`` simply by adjusting the geometric eigenvalue ratio ``q``.

    Parameters
    ----------
    sigma_mu : float
        Target RMS spatial coherence width [m] (may be ``np.inf``).
    w_m : float, optional
        Fixed Hermite-Gaussian modal scale [m]. Default 1.4e-6.
    n_modes : int, optional
        Number of modes ``M``. Default 10.

    Returns
    -------
    info : dict
        Keys ``q``, ``sigma_s`` [m], ``sigma_mu`` [m], ``w_m`` [m],
        ``weights`` (array of length ``n_modes``), and ``truncation_error`` (``q**n_modes``).
    """
    if np.isinf(sigma_mu):
        q = 0.0
        sigma_s = float(w_m) / np.sqrt(2.0)
    else:
        if sigma_mu <= 0:
            raise ValueError("sigma_mu must be > 0.")
        gamma = (float(w_m) / float(sigma_mu)) ** 2
        q = float((np.sqrt(1.0 + gamma**2) - 1.0) / gamma)
        sigma_s = (float(w_m) / np.sqrt(2.0)) * float(np.sqrt((1.0 + q) / (1.0 - q)))
    m_idx = np.arange(int(n_modes), dtype=float)
    weights = (1.0 - q) * (q**m_idx)
    return {
        "q": q,
        "sigma_s": sigma_s,
        "sigma_mu": float(sigma_mu),
        "w_m": float(w_m),
        "weights": weights,
        "truncation_error": float(q ** int(n_modes)),
    }


def gsm_modal_slit_coherence(sigma_mu, d=2.0e-6, w_m=1.4e-6, n_modes=10):
    """
    Exact ``M``-mode truncated degree of spatial coherence ``mu_12^(M)`` between
    ``x_1 = -d/2`` and ``x_2 = +d/2`` for the Hermite-Gaussian Mercer expansion.

    Parameters
    ----------
    sigma_mu : float
        RMS spatial coherence width [m].
    d : float, optional
        Slit separation [m]. Default 2.0e-6.
    w_m : float, optional
        Modal scale [m]. Default 1.4e-6.
    n_modes : int, optional
        Truncation order ``M >= 1``. Default 10.

    Returns
    -------
    mu_12_M : float
        Truncated degree of coherence ``W_M(-d/2, +d/2) / W_M(+d/2, +d/2)``.
    """
    info = gsm_parameters_from_modal_scale(sigma_mu, w_m=w_m, n_modes=n_modes)
    w = info["weights"]
    phi_vals = np.array([gsm_hermite_mode(m, 0.5 * d, w_m=w_m) for m in range(int(n_modes))])
    parity = (-1.0) ** np.arange(int(n_modes))
    num = float(np.sum(w * parity * (phi_vals**2)))
    den = float(np.sum(w * (phi_vals**2)))
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# 3. Incoherent Ensemble Superposition & Fringe Visibility Extraction
# ---------------------------------------------------------------------------


def incoherent_power_sum(fields, weights=None):
    """
    Compute the phase-incoherent power (intensity) superposition
    ``I(r) = sum_m w_m |E^(m)(r)|^2`` across mutually uncorrelated realizations
    or orthogonal Mercer modes.

    Parameters
    ----------
    fields : array_like
        Complex field array of shape ``(n_realizations, ...)``.
    weights : array_like, optional
        Non-negative quadrature or modal weights of shape ``(n_realizations,)``.
        Default is uniform weights ``1 / n_realizations``.

    Returns
    -------
    intensity : ndarray
        Ensemble-averaged intensity ``<|E(r)|^2>`` of shape ``fields.shape[1:]``.
    """
    f_arr = np.asarray(fields)
    n_r = f_arr.shape[0]
    if weights is None:
        w = np.full(n_r, 1.0 / max(n_r, 1), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("Incoherent power weights must be non-negative.")
    w_b = w.reshape((n_r,) + (1,) * (f_arr.ndim - 1))
    return np.sum(w_b * (np.abs(f_arr) ** 2), axis=0)


def coherent_field_sum_intensity(fields, weights=None):
    """
    Compute the coherent amplitude-summed intensity ``|sum_m sqrt(w_m) E^(m)(r)|^2``
    (demonstrating the unphysical cross-term interference artifact when mutually
    incoherent emitters are driven simultaneously in one deterministic run).

    Parameters
    ----------
    fields : array_like
        Complex field array of shape ``(n_realizations, ...)``.
    weights : array_like, optional
        Non-negative weights of shape ``(n_realizations,)``.

    Returns
    -------
    intensity_coherent : ndarray
        Coherently summed intensity of shape ``fields.shape[1:]``.
    """
    f_arr = np.asarray(fields)
    n_r = f_arr.shape[0]
    if weights is None:
        w = np.full(n_r, 1.0 / max(n_r, 1), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    amp_w = np.sqrt(np.maximum(w, 0.0)).reshape((n_r,) + (1,) * (f_arr.ndim - 1))
    return np.abs(np.sum(amp_w * f_arr, axis=0)) ** 2


def complex_degree_of_coherence(e1_samples, e2_samples, weights=None):
    """
    Compute the complex degree of coherence ``mu_12`` between two points from an
    ensemble of complex field samples ``(E_1^(m), E_2^(m))``.

    Parameters
    ----------
    e1_samples, e2_samples : array_like
        1D arrays of complex field values at slit 1 and slit 2 across ``M`` runs.
    weights : array_like, optional
        Non-negative ensemble weights of length ``M``.

    Returns
    -------
    mu_12 : complex
        Normalized cross-spectral density ``<E_1^* E_2> / sqrt(<|E_1|^2> <|E_2|^2>)``.
    """
    e1 = np.asarray(e1_samples, dtype=np.complex128)
    e2 = np.asarray(e2_samples, dtype=np.complex128)
    if weights is None:
        w = np.ones(len(e1), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    i1 = float(np.sum(w * (np.abs(e1) ** 2)))
    i2 = float(np.sum(w * (np.abs(e2) ** 2)))
    denom = np.sqrt(i1 * i2)
    if denom <= 1e-30:
        return 0.0 + 0.0j
    w12 = complex(np.sum(w * np.conj(e1) * e2))
    return w12 / denom


def extract_fringe_visibility(
    x_det, intensity, d=2.0e-6, L2=6.0e-6, wavelength=0.5e-6
):
    """
    Extract the on-axis Michelson fringe visibility ``V = (I_max - I_min)/(I_max + I_min)``
    and signed coherence ``mu_signed`` from a 1D detector screen intensity profile
    ``I(x_det)``, accounting for the slowly varying single-slit diffraction envelope.

    Fits ``I(x) = I_env(x) * [1 + mu_c cos(Delta_phi(x)) + mu_s sin(Delta_phi(x))]``
    with quadratic envelope ``I_env(x) = a_0 + a_2 x^2`` over the central 1.5
    fringe periods ``|Delta_phi(x)| <= 3*pi``.

    Parameters
    ----------
    x_det : array_like
        Transverse detector coordinates [m].
    intensity : array_like
        Real intensity profile ``I(x_det)`` on the detector screen.
    d : float, optional
        Slit separation [m]. Default 2.0e-6.
    L2 : float, optional
        Slit-to-screen distance [m]. Default 6.0e-6.
    wavelength : float, optional
        Center wavelength [m]. Default 0.5e-6.

    Returns
    -------
    visibility : float
        Unsigned fringe visibility ``V in [0, 1]`` (dimensionless).
    mu_signed : float
        Signed fringe contrast ``mu_c in [-1, 1]`` (negative when a dark fringe
        sits on the optical axis ``x = 0`` due to coherence phase reversal).
    """
    x_arr = np.asarray(x_det, dtype=float)
    i_arr = np.asarray(intensity, dtype=float)
    k0 = 2.0 * np.pi / float(wavelength)
    dphi = k0 * (np.hypot(x_arr + 0.5 * d, L2) - np.hypot(x_arr - 0.5 * d, L2))
    mask = np.abs(dphi) <= 3.0 * np.pi
    xw = x_arr[mask] * 1e6
    ph = dphi[mask]
    iw = i_arr[mask]
    basis = np.column_stack(
        [
            np.ones_like(xw),
            xw**2,
            np.cos(ph),
            (xw**2) * np.cos(ph),
            np.sin(ph),
        ]
    )
    coef, *_ = np.linalg.lstsq(basis, iw, rcond=None)
    a0 = max(float(coef[0]), 1e-30)
    mu_signed = float(np.clip(coef[2] / a0, -1.0, 1.0))
    vis = float(np.clip(np.hypot(coef[2], coef[4]) / a0, 0.0, 1.0))
    return vis, mu_signed


# ---------------------------------------------------------------------------
# 4. Polychromatic Temporal Coherence Quadrature
# ---------------------------------------------------------------------------


def polychromatic_screen_intensity(
    freqs, dft_ez_even, dft_ez_odd, pulse, f0, rel_bw_sigma, x_det=None
):
    """
    Integrate pulse-normalized broadband ``DFTMonitor`` spectra over a Gaussian
    power spectral density ``S(f; sigma_f)`` to obtain the polychromatic screen
    intensity ``I_poly(x)``, incoherent single-slit envelope ``I_inc(x)``, and
    local fringe contrast ``C(x) = (I_poly(x) - I_inc(x)) / I_inc(x)``.

    Parameters
    ----------
    freqs : array_like
        Monitored frequencies [Hz], shape ``(n_freqs,)``.
    dft_ez_even : array_like
        Complex DFT field ``E_z(f, x)`` for symmetric (in-phase) two-slit illumination,
        shape ``(n_freqs, nx)``.
    dft_ez_odd : array_like
        Complex DFT field ``E_z(f, x)`` for antisymmetric (out-of-phase) two-slit
        illumination (scaled to unit slit amplitude), shape ``(n_freqs, nx)``.
    pulse : GaussianSource
        Time-domain pulse used in the simulation (for ``pulse.fourier_transform(freqs)``).
    f0 : float
        Center frequency [Hz].
    rel_bw_sigma : float
        RMS fractional frequency bandwidth ``sigma_f / f0`` (0 for monochromatic).
    x_det : array_like, optional
        Detector x-coordinates [m] (used to normalize each frequency slice at ``x = 0``).

    Returns
    -------
    i_poly : ndarray
        Polychromatic interference intensity profile ``I(x; sigma_f)``.
    i_inc : ndarray
        Polychromatic incoherent two-slit envelope ``I_1(x) + I_2(x)``.
    contrast : ndarray
        Envelope-normalized fringe signal ``(I_poly - I_inc) / I_inc``.
    """
    freqs = np.asarray(freqs, dtype=float)
    s_pulse = pulse.fourier_transform(freqs)[:, None]
    h_even = np.abs(np.asarray(dft_ez_even) / s_pulse) ** 2
    h_odd = np.abs(np.asarray(dft_ez_odd) / s_pulse) ** 2
    h_inc = 0.5 * (h_even + h_odd)

    if x_det is not None:
        i0_idx = int(np.argmin(np.abs(np.asarray(x_det) - 0.0)))
        norm_f = np.maximum(h_inc[:, i0_idx : i0_idx + 1], 1e-30)
    else:
        norm_f = np.maximum(np.mean(h_inc, axis=1, keepdims=True), 1e-30)

    i_even_f = h_even / norm_f
    i_inc_f = h_inc / norm_f

    if rel_bw_sigma <= 1e-9:
        ic = int(np.argmin(np.abs(freqs - f0)))
        i_poly = i_even_f[ic]
        i_inc = i_inc_f[ic]
    else:
        sigma_f = float(rel_bw_sigma) * float(f0)
        df = np.gradient(freqs)
        w_f = df * np.exp(-0.5 * ((freqs - f0) / sigma_f) ** 2)
        w_f /= np.sum(w_f)
        i_poly = np.sum(w_f[:, None] * i_even_f, axis=0)
        i_inc = np.sum(w_f[:, None] * i_inc_f, axis=0)

    contrast = (i_poly - i_inc) / np.maximum(i_inc, 1e-30)
    return i_poly, i_inc, contrast


# ---------------------------------------------------------------------------
# 5. Dispersive Silver Screen & 2D FIT Double-Slit Solver Builder
# ---------------------------------------------------------------------------


def silver_drude_material(wavelength_ref=0.5e-6):
    """
    Construct a 1-pole Drude ``DispersiveMaterial`` matching the Rakic et al.
    (1998) complex permittivity of silver at ``wavelength_ref = 500 nm``
    (``eps = -7.632 - 0.731j``, skin depth ``delta_skin = 28.77 nm``), while
    remaining explicitly stable at coarse 10-cpw FDTD/FIT timesteps.

    Parameters
    ----------
    wavelength_ref : float, optional
        Reference wavelength [m]. Default 0.5e-6.

    Returns
    -------
    mat : DispersiveMaterial
        Single-pole Drude silver material.
    skin_depth_m : float
        Electric-field skin depth ``lambda_0 / (2 * pi * Im(sqrt(conj(eps))))`` [m].
    """
    f0 = c_light / float(wavelength_ref)
    w0 = 2.0 * np.pi * f0
    eps_rakic = complex(dispersive_lib["silver"].epsilon(f0))
    eps_inf = 3.7
    chi_target = eps_rakic - eps_inf
    inv_neg_chi = -1.0 / chi_target
    wp = float(np.sqrt((w0**2) / np.real(inv_neg_chi)))
    gamma = float(-np.imag(inv_neg_chi) * (wp**2) / w0)
    mat = DispersiveMaterial(eps_inf, [Drude(wp, gamma)], name="silver_drude_500nm")
    n_complex = np.sqrt(np.conj(eps_rakic))
    skin_depth_m = float(wavelength_ref) / (2.0 * np.pi * float(np.imag(n_complex)))
    return mat, skin_depth_m


def build_double_slit_solver(
    x_bounds=(-5.0e-6, 5.0e-6),
    y_bounds=(-2.0e-6, 6.5e-6),
    wavelength=0.5e-6,
    cpw=10,
    n_pml=8,
    slit_sep=2.0e-6,
    slit_width=0.4e-6,
    screen_thickness=0.2e-6,
    screen_material="pec",
    cfln=None,
    cells_per_skin_depth=None,
    grid_only=False,
):
    """
    Build a 2D TM (``Ez, Hx, Hy``, ``Nz = 1`` with periodic z-boundaries)
    Wakis ``GridFIT3D`` and ``SolverFIT3D`` containing a double-slit screen at ``y = 0``.

    Grid axes are constructed with ``wakis.templates.meshing.graded_axis`` so
    that mesh nodes snap exactly to the slit aperture walls ``x = +-d/2 +- a/2``
    and screen faces ``y = +-t_screen/2``, enforcing ``>= 4`` cells across the
    screen thickness and ``>= 8`` cells across each slit opening. For dispersive
    silver (or when ``cells_per_skin_depth`` is given as a positive float), a
    symmetric skin-depth collar ``+-2.5 * delta_E`` around each slit wall and
    screen face is refined to ``dx_min = delta_E / cells_per_skin_depth`` and
    graded smoothly into the background at ratio ``<= 1.4``.

    Parameters
    ----------
    x_bounds, y_bounds : tuple of 2 float
        Physical interior domain ``(lo, hi)`` [m] before adding ``n_pml`` cells.
    wavelength : float, optional
        Reference vacuum wavelength [m]. Default 0.5e-6.
    cpw : float, optional
        Background cells per wavelength ``dx_bg = wavelength / cpw``. Default 10.
    n_pml : int, optional
        CPML absorber thickness in cells. Default 8.
    slit_sep : float, optional
        Center-to-center slit separation ``d`` [m]. Default 2.0e-6.
    slit_width : float, optional
        Individual slit opening width ``a`` [m]. Default 0.4e-6.
    screen_thickness : float, optional
        Screen thickness ``t_screen`` along ``y`` [m]. Default 0.2e-6.
    screen_material : {'pec', 'silver'}, optional
        Screen material: ``'pec'`` or ``'silver'`` (Drude dispersive metal).
    cfln : float, optional
        Courant number (default 0.65 for both PEC and dispersive silver).
    cells_per_skin_depth : float, 'uniform', or None, optional
        Target cells per electric-field skin depth ``delta_E = lambda_0 / (2*pi*kappa)``
        (28.77 nm at 500 nm; note ``delta_I = lambda_0 / (4*pi*kappa) = delta_E / 2``
        is 14.39 nm). Defaults to ``8.0`` when ``screen_material == 'silver'``
        (giving ``8.0`` cells per ``delta_E`` and ``4.0`` cells per ``delta_I``),
        and ``None`` (snapped background mesh) when ``screen_material == 'pec'``.
        Pass ``'uniform'`` to force an unrefined uniform background mesh for
        under-resolved demonstration runs.
    grid_only : bool, optional
        If True, return ``(None, grid, screen_boxes)`` without instantiating
        ``SolverFIT3D``. Default False.

    Returns
    -------
    solver : SolverFIT3D or None
        Configured 2D TM FIT solver (or None when ``grid_only=True``).
    grid : GridFIT3D
        Underlying 2D FIT mesh.
    screen_boxes : list of Box
        The three screen segments (left jaw, central barrier, right jaw).
    """
    f0 = c_light / float(wavelength)
    dx_bg = float(wavelength) / float(cpw)
    n_pml = int(n_pml)

    x_outer_edge = 0.5 * float(slit_sep) + 0.5 * float(slit_width)
    x_inner_edge = 0.5 * float(slit_sep) - 0.5 * float(slit_width)
    y_half_screen = 0.5 * float(screen_thickness)

    if cells_per_skin_depth is None and screen_material == "silver":
        cells_per_skin_depth = 8.0

    use_skin_grading = (
        cells_per_skin_depth is not None
        and cells_per_skin_depth != "uniform"
        and float(cells_per_skin_depth) > 0.0
    )

    if use_skin_grading:
        _, skin_e = silver_drude_material(wavelength)
        dx_fine = min(dx_bg, float(skin_e) / float(cells_per_skin_depth))
        dx_slit = min(dx_bg, float(slit_width) / 16.0)
        collar = 2.5 * float(skin_e)
        x_pts = [
            float(x_bounds[0]),
            -x_outer_edge - collar,
            -x_outer_edge,
            -x_outer_edge + collar,
            -x_inner_edge - collar,
            -x_inner_edge,
            -x_inner_edge + collar,
            +x_inner_edge - collar,
            +x_inner_edge,
            +x_inner_edge + collar,
            +x_outer_edge - collar,
            +x_outer_edge,
            +x_outer_edge + collar,
            float(x_bounds[1]),
        ]
        x_sizes = [
            dx_bg,
            dx_fine,
            dx_fine,
            dx_slit,
            dx_fine,
            dx_fine,
            dx_bg,
            dx_fine,
            dx_fine,
            dx_slit,
            dx_fine,
            dx_fine,
            dx_bg,
        ]
        x_int = graded_axis(x_pts, x_sizes, grading=1.4)
        y_pts = [
            float(y_bounds[0]),
            -y_half_screen - collar,
            -y_half_screen,
            +y_half_screen,
            +y_half_screen + collar,
            float(y_bounds[1]),
        ]
        y_sizes = [dx_bg, dx_fine, dx_fine, dx_fine, dx_bg]
        y_int = graded_axis(y_pts, y_sizes, grading=1.4)
    else:
        min_c_x = (
            [1, 1, 1, 1, 1]
            if cells_per_skin_depth == "uniform"
            else [1, 8, 4, 8, 1]
        )
        min_c_y = (
            [1, 1, 1] if cells_per_skin_depth == "uniform" else [1, 4, 1]
        )
        x_pts = [
            float(x_bounds[0]),
            -x_outer_edge,
            -x_inner_edge,
            +x_inner_edge,
            +x_outer_edge,
            float(x_bounds[1]),
        ]
        x_sizes = [dx_bg] * 5
        x_int = graded_axis(x_pts, x_sizes, min_cells=min_c_x, grading=1.4)
        y_pts = [
            float(y_bounds[0]),
            -y_half_screen,
            +y_half_screen,
            float(y_bounds[1]),
        ]
        y_sizes = [dx_bg] * 3
        y_int = graded_axis(y_pts, y_sizes, min_cells=min_c_y, grading=1.4)

    if abs(float(x_bounds[0]) + float(x_bounds[1])) < 1e-12:
        x_int = 0.5 * (x_int - x_int[::-1])

    dx_lo, dx_hi = float(x_int[1] - x_int[0]), float(x_int[-1] - x_int[-2])
    x = np.concatenate(
        [
            x_int[0] - dx_lo * np.arange(n_pml, 0, -1),
            x_int,
            x_int[-1] + dx_hi * np.arange(1, n_pml + 1),
        ]
    )
    dy_lo, dy_hi = float(y_int[1] - y_int[0]), float(y_int[-1] - y_int[-2])
    y = np.concatenate(
        [
            y_int[0] - dy_lo * np.arange(n_pml, 0, -1),
            y_int,
            y_int[-1] + dy_hi * np.arange(1, n_pml + 1),
        ]
    )
    z = np.array([-5.0 * dx_bg, 5.0 * dx_bg])

    if cfln is None:
        cfln = 0.65

    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)

    x_max_grid = float(x[-1]) + 2.0 * dx_bg
    w_outer = x_max_grid - x_outer_edge
    xc_outer = 0.5 * (x_max_grid + x_outer_edge)
    w_center = 2.0 * x_inner_edge

    screen_boxes = [
        Box((-xc_outer, 0.0, 0.0), (w_outer, screen_thickness, 20.0 * dx_bg), "pec"),
        Box((0.0, 0.0, 0.0), (w_center, screen_thickness, 20.0 * dx_bg), "pec"),
        Box((+xc_outer, 0.0, 0.0), (w_outer, screen_thickness, 20.0 * dx_bg), "pec"),
    ]

    if grid_only:
        return None, grid, screen_boxes

    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=n_pml,
        cfln=float(cfln),
        verbose=0,
    )

    if screen_material == "pec":
        assign_materials(solver, screen_boxes, f0)
    elif screen_material == "silver":
        ag_mat, _ = silver_drude_material(wavelength)
        for box in screen_boxes:
            solver.add_dispersive_material(ag_mat, center=box.center, size=box.size)
    else:
        raise ValueError(f"Unknown screen_material {screen_material!r}.")

    return solver, grid, screen_boxes


def measure_mesh_resolution(
    grid,
    wavelength=0.5e-6,
    slit_sep=2.0e-6,
    slit_width=0.4e-6,
    screen_thickness=0.2e-6,
    screen_material="pec",
    n_steps=0,
):
    """
    Measure background cell size, minimum cell size, cells across the smallest
    geometric feature (screen thickness and slit width), and skin-depth
    resolution directly from a constructed ``GridFIT3D`` mesh.

    Parameters
    ----------
    grid : GridFIT3D
        Constructed 2D FIT mesh.
    wavelength : float, optional
        Reference vacuum wavelength [m]. Default 0.5e-6.
    slit_sep : float, optional
        Center-to-center slit separation [m]. Default 2.0e-6.
    slit_width : float, optional
        Individual slit opening width [m]. Default 0.4e-6.
    screen_thickness : float, optional
        Screen thickness along ``y`` [m]. Default 0.2e-6.
    screen_material : {'pec', 'silver'}, optional
        Screen material. Default ``'pec'``.
    n_steps : int, optional
        Number of time steps executed per solve. Default 0.

    Returns
    -------
    metrics : dict
        Resolution dictionary containing ``dx_background_nm``, ``dx_min_nm``,
        ``min_feature_nm``, ``cells_across_feature``, ``cells_across_slit``,
        ``n_cells``, ``n_steps``, and (for ``screen_material == 'silver'``)
        ``skin_depth_nm``, ``cells_per_skin_depth``, ``skin_depth_power_nm``,
        and ``cells_per_skin_depth_power``.
    """
    dx_bg_m = float(max(np.max(grid.dx), np.max(grid.dy)))
    dx_min_m = float(min(np.min(grid.dx), np.min(np.diff(grid.y))))

    yc = 0.5 * (grid.y[:-1] + grid.y[1:])
    tol_y = 1e-12
    cells_screen = int(
        np.sum((yc >= -0.5 * screen_thickness - tol_y) & (yc <= 0.5 * screen_thickness + tol_y))
    )

    xc = 0.5 * (grid.x[:-1] + grid.x[1:])
    x_lo = 0.5 * slit_sep - 0.5 * slit_width
    x_hi = 0.5 * slit_sep + 0.5 * slit_width
    tol_x = 1e-12
    cells_slit = int(np.sum((xc >= x_lo - tol_x) & (xc <= x_hi + tol_x)))

    min_feature_nm = round(float(min(screen_thickness, slit_width)) * 1e9, 2)
    cells_across_feature = int(min(cells_screen, cells_slit))

    info = {
        "dx_background_nm": round(dx_bg_m * 1e9, 3),
        "dx_min_nm": round(dx_min_m * 1e9, 3),
        "min_feature_nm": min_feature_nm,
        "cells_across_feature": cells_across_feature,
        "cells_across_slit": cells_slit,
        "n_cells": int(grid.Nx * grid.Ny),
        "n_steps": int(n_steps),
    }
    if screen_material == "silver":
        _, skin_e_m = silver_drude_material(wavelength)
        skin_e_nm = float(skin_e_m) * 1e9
        skin_i_nm = 0.5 * skin_e_nm
        info["skin_depth_nm"] = round(skin_e_nm, 2)
        info["cells_per_skin_depth"] = round(skin_e_nm / (dx_min_m * 1e9), 2)
        info["skin_depth_power_nm"] = round(skin_i_nm, 2)
        info["cells_per_skin_depth_power"] = round(skin_i_nm / (dx_min_m * 1e9), 2)
    return info


class CustomLineSource:
    """
    Transverse line source injecting ``J_z(x, y_src, t) = A(x) * pulse(t)``
    across a 1D array of grid cells in a 2D TM (``Ez, Hx, Hy``) FIT simulation.

    Used to excite individual Hermite-Gaussian Mercer modes ``Phi_m(x)``,
    upstream point-dipole line sources ``delta(x - x_s)``, or symmetric/antisymmetric
    two-slit aperture modes.

    Parameters
    ----------
    ix : int or array_like of int
        Grid x-indices of the source cells.
    iy : int or array_like of int
        Grid y-indices of the source cells.
    weights : float or array_like of float
        Spatial amplitude weights ``A(x)`` at each cell.
    pulse : callable or object with ``eval(t)``
        Time-domain waveform (e.g. ``wakis.sources.GaussianSource``).
    """

    def __init__(self, ix, iy, weights, pulse):
        self.ix = np.atleast_1d(np.asarray(ix, dtype=int))
        self.iy = np.atleast_1d(np.asarray(iy, dtype=int))
        self.weights = np.atleast_1d(np.asarray(weights, dtype=float))
        self.pulse = pulse

    def update(self, solver, t):
        val = self.pulse.eval(t) if hasattr(self.pulse, "eval") else self.pulse(t)
        if val != 0.0:
            solver.J[self.ix, self.iy, 0, "z"] = self.weights * val
        else:
            solver.J[self.ix, self.iy, 0, "z"] = 0.0


# ---------------------------------------------------------------------------
# 6. Schematic & Geometry Overlay Helpers (`wakis.schematic`)
# ---------------------------------------------------------------------------

_DOUBLE_SLIT_OVERLAY_CACHE: dict[str, dict] = {}


def build_double_slit_schematic_objects(
    mode: str = "vcz",
    cpw: int = 20,
    screen_material: str = "pec",
    cells_per_skin_depth: float | None = None,
) -> dict:
    """Build solver, grid, structures, sources, and monitors for `wakis.schematic`."""
    from wakis.field_monitors import DFTMonitor
    from wakis.sources import CurrentSource, GaussianSource

    wavelength = 0.50e-6
    f0 = c_light / wavelength
    slit_sep = 2.00e-6
    slit_width = 0.40e-6
    screen_thickness = 0.20e-6
    y_bounds = (-8.5e-6, 6.4e-6) if mode == "vcz" else (-1.6e-6, 6.4e-6)

    solver, grid, screen_boxes = build_double_slit_solver(
        x_bounds=(-5.0e-6, 5.0e-6),
        y_bounds=y_bounds,
        wavelength=wavelength,
        cpw=cpw,
        n_pml=8,
        slit_sep=slit_sep,
        slit_width=slit_width,
        screen_thickness=screen_thickness,
        screen_material=screen_material,
        cells_per_skin_depth=cells_per_skin_depth,
    )
    dz = float(grid.z[-1] - grid.z[0])
    pulse = GaussianSource(f0, 0.20 * f0)

    if mode == "vcz":
        xs_dip = np.linspace(-1.6e-6, 1.6e-6, 9)
        sources = [
            CurrentSource(
                pulse,
                component="z",
                center=(float(xs), -8.0e-6, 0.0),
                size=(0.0, 0.0, dz),
            )
            for xs in xs_dip
        ]
    else:
        src = CurrentSource(
            pulse,
            component="z",
            center=(0.0, -1.0e-6, 0.0),
            size=(8.0e-6, 0.0, dz),
        )
        src.normal = 1
        sources = [src]

    monitors = [
        DFTMonitor(
            [f0],
            components=("Ez",),
            center=(0.0, 0.0, 0.0),
            size=(9.2e-6, 0.0, 0.0),
        ),
        DFTMonitor(
            [f0],
            components=("Ez",),
            center=(0.0, 6.0e-6, 0.0),
            size=(9.2e-6, 0.0, 0.0),
        ),
    ]
    return {
        "solver": solver,
        "grid": grid,
        "structures": screen_boxes,
        "sources": sources,
        "monitors": monitors,
    }


def tidy_double_slit_schematic_axes(
    ax,
    mode: str = "vcz",
    xlim: tuple[float, float] = (-5.2, 5.2),
    ylim: tuple[float, float] | None = None,
    fontsize: float = 6.5,
) -> None:
    """Declutter `plot_schematic` auto-labels and draw crisp double-slit setup callouts."""
    import matplotlib.patches as mpatches

    if ylim is None:
        ylim = (-8.8, 6.8) if mode == "vcz" else (-1.8, 6.8)
    ax.set_aspect("auto")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(ax.get_title(), pad=26)

    drop_keys = (
        "boundary:",
        "Out-of-plane",
        "PEC",
        "silver",
        "ε =",
        "Pulse:",
        "CurrentSource",
        "DFTMonitor",
        "E (E_z)",
        "$E_z$",
        "\\mathbf{E}",
        "\\mathbf{k}",
    )
    for t_obj in list(ax.texts):
        s = t_obj.get_text()
        if any(k in s for k in drop_keys):
            t_obj.remove()
        elif "Domain" in s:
            t_obj.set_text(r"Domain $L_x = 10.8\;\mu\mathrm{m}$ ($\lambda_0 = 600\;\mathrm{nm}$, $\geq 20\;\mathrm{cells}/\lambda_0$)")
            t_obj.set_position((0.0, ylim[1] + 0.03 * (ylim[1] - ylim[0])))
        elif s == "" and getattr(t_obj, "xy", None) is not None and t_obj.xy[1] > ylim[1]:
            t_obj.remove()

    # Draw the 3 double-slit screen segments at y in [-0.10, +0.10] um
    for x_lo, x_hi in [(-5.2, -1.2), (-0.8, 0.8), (1.2, 5.2)]:
        ax.add_patch(
            mpatches.Rectangle(
                (x_lo, -0.10),
                x_hi - x_lo,
                0.20,
                facecolor="#334155",
                edgecolor="#0f172a",
                lw=1.0,
                zorder=5,
            )
        )

    bbox_kw = {
        "boxstyle": "round,pad=0.13",
        "facecolor": "white",
        "edgecolor": "#475569",
        "linewidth": 0.5,
        "alpha": 0.90,
    }

    # Monitor lines at y = 0 (slits) and y = +6.0 um (detector screen)
    ax.plot([-4.6, 4.6], [6.0, 6.0], color="#059669", ls="--", lw=1.4, zorder=6)
    ax.text(
        0.0,
        5.55 if mode == "vcz" else 5.65,
        r"Detector DFTMonitor ($y = +L_2 = +6.0\;\mu\mathrm{m}$, $\Lambda_f = 1.50\;\mu\mathrm{m}$)",
        ha="center",
        va="top",
        fontsize=fontsize - 0.2,
        color="#065f46",
        bbox=bbox_kw,
        zorder=9,
    )
    ax.annotate(
        "PEC / Drude Ag Screen ($y = 0$, $t = 200\\;\\mathrm{nm}$)\n2 slits: $a = 0.40\\;\\mu\\mathrm{m}$ at $x = \\pm 1.00\\;\\mu\\mathrm{m}$ ($d = 2.00\\;\\mu\\mathrm{m}$)",
        xy=(1.0, 0.10),
        xytext=(0.0, 1.55),
        ha="center",
        va="center",
        fontsize=fontsize - 0.3,
        color="#0f172a",
        bbox=bbox_kw,
        arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 0.9},
        zorder=9,
    )
    ax.text(
        -4.85,
        ylim[1] - 0.04 * (ylim[1] - ylim[0]),
        r"$\pm x, \pm y$: CPML (8 cells) | $\pm z$: Periodic ($E_z$ TM)",
        ha="left",
        va="top",
        fontsize=fontsize - 0.4,
        color="#1e293b",
        bbox=bbox_kw,
        zorder=9,
    )

    if mode == "vcz":
        xs_dip = np.linspace(-1.6, 1.6, 9)
        ax.plot([-1.8, 1.8], [-8.0, -8.0], color="#d97706", lw=2.2, zorder=6)
        ax.plot(xs_dip, np.full_like(xs_dip, -8.0), "*", color="#f59e0b", ms=6.2, mec="black", mew=0.4, zorder=7)
        ax.annotate(
            "Method A: Extended Incoherent Strip $D \\in [0, 3.6\\;\\mu\\mathrm{m}]$\n37 uncorrelated $J_z$ line dipoles at $y = -L_1 = -8.0\\;\\mu\\mathrm{m}$\n(First VCZ null $D_0 = \\lambda_0 L_1 / d = 2.00\\;\\mu\\mathrm{m}$)",
            xy=(0.0, -8.0),
            xytext=(0.0, -5.2),
            ha="center",
            va="center",
            fontsize=fontsize - 0.3,
            color="#78350f",
            bbox=bbox_kw,
            arrowprops={"arrowstyle": "->", "color": "#d97706", "lw": 0.9},
            zorder=9,
        )
    else:
        x_prof = np.linspace(-3.8, 3.8, 200)
        phi0 = 0.48 * np.exp(-0.5 * (x_prof / 1.4) ** 2)
        ax.plot([-4.2, 4.2], [-1.0, -1.0], color="#2563eb", lw=1.6, zorder=6)
        ax.fill_between(x_prof, -1.0, -1.0 + phi0, color="#60a5fa", alpha=0.35, zorder=5)
        ax.annotate(
            "Methods B & C: GSM Mercer Eigenmodes $\\Phi_m(x)$\n& Broadband Pulse Line Source ($y = -1.0\\;\\mu\\mathrm{m}$, $w_m = 1.40\\;\\mu\\mathrm{m}$)",
            xy=(0.0, -1.0),
            xytext=(0.0, 3.65),
            ha="center",
            va="center",
            fontsize=fontsize - 0.3,
            color="#1e3a8a",
            bbox=bbox_kw,
            arrowprops={"arrowstyle": "->", "color": "#2563eb", "lw": 0.9},
            zorder=9,
        )


def overlay_double_slit_geometry(
    ax,
    line_color: str = "white",
    show_source: bool = True,
    show_detector: bool = True,
    show_labels: bool = True,
    compact: bool = False,
    fontsize: float = 6.0,
) -> None:
    """Overlay double-slit screen, slit openings, CPML boundaries, source & detector via `wakis.schematic`."""
    import matplotlib.patches as mpatches
    from wakis import schematic

    if "gsm" not in _DOUBLE_SLIT_OVERLAY_CACHE:
        _DOUBLE_SLIT_OVERLAY_CACHE["gsm"] = build_double_slit_schematic_objects(
            mode="gsm", cpw=10, screen_material="pec"
        )
    obj = _DOUBLE_SLIT_OVERLAY_CACHE["gsm"]
    schematic.overlay_geometry(
        ax,
        obj["solver"],
        plane="xy",
        unit="um",
        structures=obj["structures"],
        show_boundaries=True,
        show_materials=False,
        outline_color=line_color,
        label_materials=False,
    )

    # Crisp double-slit screen jaws at y in [-0.10, +0.10] um
    for x_lo, x_hi in [(-4.9, -1.2), (-0.8, 0.8), (1.2, 4.9)]:
        ax.add_patch(
            mpatches.Rectangle(
                (x_lo, -0.10),
                x_hi - x_lo,
                0.20,
                facecolor="#334155" if line_color == "black" else "none",
                edgecolor=line_color,
                lw=1.1,
                alpha=0.85,
                zorder=7,
            )
        )

    if show_source:
        ax.axhline(-1.0, color="#38bdf8", ls=":", lw=1.0, alpha=0.85, zorder=7)
    if show_detector:
        ax.axhline(6.0, color="#34d399", ls="--", lw=1.1, alpha=0.90, zorder=7)

    if not show_labels:
        return

    bbox_kw = {
        "boxstyle": "round,pad=0.10",
        "facecolor": "white",
        "edgecolor": "#475569",
        "linewidth": 0.4,
        "alpha": 0.82,
    }
    if not compact:
        ax.text(
            0.0,
            0.22,
            "PEC/Ag screen (a=0.4 µm, d=2.0 µm)",
            ha="center",
            va="bottom",
            fontsize=fontsize - 0.2,
            color="#0f172a",
            bbox=bbox_kw,
            zorder=9,
        )
        if show_detector:
            ax.text(
                4.55,
                5.85,
                "Detector (y=6 µm) | CPML",
                ha="right",
                va="top",
                fontsize=fontsize - 0.3,
                color="#065f46",
                bbox=bbox_kw,
                zorder=9,
            )
        if show_source:
            ax.text(
                -4.55,
                -0.88,
                "GSM source (y=−1 µm)",
                ha="left",
                va="bottom",
                fontsize=fontsize - 0.4,
                color="#1e3a8a",
                bbox=bbox_kw,
                zorder=9,
            )
    else:
        ax.text(
            0.0,
            0.22,
            "Slits (±1 µm, a=0.4 µm)",
            ha="center",
            va="bottom",
            fontsize=fontsize - 0.5,
            color="#0f172a",
            bbox=bbox_kw,
            zorder=9,
        )
        if show_detector:
            ax.text(
                4.55,
                5.82,
                "Detector | CPML",
                ha="right",
                va="top",
                fontsize=fontsize - 0.5,
                color="#065f46",
                bbox=bbox_kw,
                zorder=9,
            )



