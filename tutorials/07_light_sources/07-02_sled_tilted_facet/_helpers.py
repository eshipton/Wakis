"""Helper functions for Tutorial 7.2: SLED tilted waveguide facet tutorial.

Provides exact analytical models (Marcuse 1989 Gaussian and step-index slab
overlap integrals, transfer-matrix thin-film AR reflectance, Snell far-field
refraction, Hakki-Paoli Fabry-Perot ASE spectral ripple) and Wakis 2D FIT
simulation wrappers for tilted-facet reflectivity, far-field projection, and
incoherent spontaneous-emission dipole ensembles.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.constants import c as C_LIGHT, epsilon_0 as EPS_0
from scipy.optimize import brentq

from wakis.field_monitors import DFTMonitor
from wakis.mode_solver import ModeMonitor, ModeSource
from wakis.near2far import Near2FarMonitor
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates import Box, Simulation
from wakis.templates.photonics import PolySlab, n_to_eps, straight

# Default physical parameters inspired by Zah et al., Electron. Lett. 23, 990 (1987)
# and Marcuse, J. Lightwave Technol. 7, 336 (1989).
WAVELENGTH_0 = 1.49e-6  # [m] center free-space wavelength
FREQ_0 = C_LIGHT / WAVELENGTH_0  # [Hz] center frequency (~201.203 THz)
N_CORE = 3.337  # [-] effective ridge core refractive index (Snell 7 deg -> 24 deg)
N_CLAD = 3.311  # [-] lateral cladding effective index (Delta_n = 0.026, NA = 0.416)
WIDTH_SM = 1.75e-6  # [m] single-mode core width (V_half = 1.534 < pi/2)
WIDTH_MM = 2.80e-6  # [m] two-mode core width (V_half = 2.455 in (pi/2, pi))
N_AR_COAT = 1.72  # [-] single-layer Al2O3/SiOxNy quarter-wave coating index
T_AR_COAT = WAVELENGTH_0 / (4.0 * N_AR_COAT)  # [m] quarter-wave thickness (~216.6 nm)


def transfer_matrix_ar_reflectance(
    theta_deg: float | np.ndarray,
    n_inc: float,
    wavelength: float | np.ndarray = WAVELENGTH_0,
    n_ar: float | None = None,
    t_ar: float = 0.0,
    n_out: float = 1.0,
    polarization: str = "scalar",
) -> np.ndarray:
    """Compute facet power reflectance with an optional single-layer AR coating.

    Parameters
    ----------
    theta_deg : float or np.ndarray
        Facet tilt angle relative to the waveguide cross-section [deg].
    n_inc : float
        Effective refractive index of the incident guided mode [-].
    wavelength : float or np.ndarray, optional
        Free-space wavelength [m]. Default is 1.49e-6 m.
    n_ar : float or None, optional
        Refractive index of the single-layer AR coating [-]. If None or
        ``t_ar <= 0``, computes the bare-facet Fresnel reflectance.
    t_ar : float, optional
        Physical thickness of the AR coating normal to the facet [m].
    n_out : float, optional
        Exterior medium refractive index [-]. Default is 1.0 (air).
    polarization : {"scalar", "te", "tm"}, optional
        Polarization model for oblique Fresnel coefficients. Default is
        ``"scalar"`` (normal-incidence impedance ratio used in Marcuse 1989).

    Returns
    -------
    R_f : np.ndarray
        Power reflectance at the facet [-].
    """
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    wl = np.asarray(wavelength, dtype=float)
    theta_b, wl_b = np.broadcast_arrays(theta, wl)

    sin_t = np.sin(theta_b)
    cos_1 = np.cos(theta_b)

    def _interface_r(n1: float, cos1: np.ndarray, n2: float) -> np.ndarray:
        sin2_sq = (n1 / n2) ** 2 * (1.0 - cos1**2)
        cos2 = np.sqrt(np.maximum(1.0 - sin2_sq, 0.0) + 0.0j)
        if polarization == "scalar":
            return (n1 - n2) / (n1 + n2) + 0.0j * cos1
        if polarization == "te":
            return (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
        if polarization == "tm":
            return (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
        raise ValueError(f"Unknown polarization: {polarization}")

    if n_ar is None or t_ar <= 0.0:
        r12 = _interface_r(n_inc, cos_1, n_out)
        return np.abs(r12) ** 2

    sin_ar_sq = (n_inc / n_ar) ** 2 * sin_t**2
    cos_ar = np.sqrt(np.maximum(1.0 - sin_ar_sq, 0.0) + 0.0j)
    r12 = _interface_r(n_inc, cos_1, n_ar)
    r23 = _interface_r(n_ar, np.real(cos_ar), n_out)
    delta = (2.0 * np.pi / wl_b) * n_ar * t_ar * cos_ar
    r_tot = (r12 + r23 * np.exp(2j * delta)) / (1.0 + r12 * r23 * np.exp(2j * delta))
    return np.abs(r_tot) ** 2


def marcuse_gaussian_r00(
    theta_deg: float | np.ndarray,
    neff: float,
    w0: float,
    wavelength: float | np.ndarray = WAVELENGTH_0,
    n_ar: float | None = None,
    t_ar: float = 0.0,
    use_tan: bool = True,
) -> np.ndarray:
    """Compute Marcuse (1989) Gaussian-mode fundamental power reflectivity R_00.

    Implements Eq. (12)-(13) of D. Marcuse, J. Lightwave Technol. 7, 336 (1989),
    for a fundamental mode approximated by a Gaussian field profile
    ``psi_0(y) = (2 / (pi * w0^2))^(1/4) * exp(-(y / w0)^2)``:

        R_00(theta_t) = R_f(theta_t) * exp(-(2 * pi * neff * w0 * tan(theta_t) / lambda)^2)

    Parameters
    ----------
    theta_deg : float or np.ndarray
        Facet tilt angle [deg].
    neff : float
        Fundamental mode effective refractive index [-].
    w0 : float
        Gaussian 1/e field half-width (waist radius) [m].
    wavelength : float or np.ndarray, optional
        Free-space wavelength [m]. Default is 1.49e-6 m.
    n_ar : float or None, optional
        AR coating refractive index [-]. Default is None (bare facet).
    t_ar : float, optional
        AR coating thickness [m]. Default is 0.0.
    use_tan : bool, optional
        If True (default), uses ``tan(theta_t)`` from the exact tilted boundary
        ``x = y * tan(theta_t)``. If False, uses the small-angle ``theta_t``.

    Returns
    -------
    R_00 : np.ndarray
        Fundamental-to-fundamental modal power reflectivity [-].
    """
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    wl = np.asarray(wavelength, dtype=float)
    slope = np.tan(theta) if use_tan else theta
    u = 2.0 * np.pi * neff * w0 * slope / wl
    r_f = transfer_matrix_ar_reflectance(
        theta_deg, n_inc=neff, wavelength=wl, n_ar=n_ar, t_ar=t_ar
    )
    return r_f * np.exp(-(u**2))


def marcuse_gaussian_r01(
    theta_deg: float | np.ndarray,
    neff: float,
    w0: float,
    wavelength: float | np.ndarray = WAVELENGTH_0,
    n_ar: float | None = None,
    t_ar: float = 0.0,
    use_tan: bool = True,
) -> np.ndarray:
    """Compute Hermite-Gaussian TE0 -> TE1 mode-conversion reflectivity R_01.

    For the first odd Hermite-Gaussian mode ``psi_1(y) = (2 * y / w0) * psi_0(y)``,
    the tilted-facet phase mismatch produces the analytical overlap power:

        R_01(theta_t) = R_f(theta_t) * u^2 * exp(-u^2)

    where ``u = 2 * pi * neff * w0 * tan(theta_t) / lambda``. It vanishes at
    ``theta_t = 0`` by symmetry and peaks at ``u = 1`` with ``R_01,max = R_f / e``.

    Parameters
    ----------
    theta_deg : float or np.ndarray
        Facet tilt angle [deg].
    neff : float
        Average effective refractive index of the coupled modes [-].
    w0 : float
        Gaussian 1/e field half-width [m].
    wavelength : float or np.ndarray, optional
        Free-space wavelength [m]. Default is 1.49e-6 m.
    n_ar : float or None, optional
        AR coating refractive index [-]. Default is None.
    t_ar : float, optional
        AR coating thickness [m]. Default is 0.0.
    use_tan : bool, optional
        Whether to use ``tan(theta_t)`` (True) or ``theta_t`` (False).

    Returns
    -------
    R_01 : np.ndarray
        Mode-conversion power reflectivity from TE0 into backward TE1 [-].
    """
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    wl = np.asarray(wavelength, dtype=float)
    slope = np.tan(theta) if use_tan else theta
    u = 2.0 * np.pi * neff * w0 * slope / wl
    r_f = transfer_matrix_ar_reflectance(
        theta_deg, n_inc=neff, wavelength=wl, n_ar=n_ar, t_ar=t_ar
    )
    return r_f * (u**2) * np.exp(-(u**2))


def solve_1d_slab_modes(
    n_core: float = N_CORE,
    n_clad: float = N_CLAD,
    width: float = WIDTH_SM,
    wavelength: float = WAVELENGTH_0,
    y_max: float = 18.0e-6,
    n_pts: int = 7201,
) -> dict:
    """Solve the exact 1D symmetric step-index slab waveguide dispersion relation.

    Finds the guided TE/TM slab eigenmodes on ``y in [-y_max, y_max]``, normalizes
    each mode such that ``integral |psi_m(y)|^2 dy = 1``, and fits the optimal
    Gaussian waist ``w0`` to the fundamental mode ``psi_0(y)``.

    Parameters
    ----------
    n_core : float, optional
        Core refractive index [-].
    n_clad : float, optional
        Cladding refractive index [-].
    width : float, optional
        Full core width ``w`` [m].
    wavelength : float, optional
        Free-space wavelength [m].
    y_max : float, optional
        Half-width of the transverse coordinate grid [m].
    n_pts : int, optional
        Number of grid points across ``[-y_max, y_max]``.

    Returns
    -------
    info : dict
        Dictionary containing ``y`` [m], ``v_number`` [-], ``neff_0`` [-],
        ``psi_0`` [m^-1/2], ``w0_overlap`` [m], ``w0_moment`` [m],
        ``confinement_0`` [-], and (if supported) ``neff_1`` [-] and
        ``psi_1`` [m^-1/2].
    """
    k0 = 2.0 * np.pi / wavelength
    a = 0.5 * width
    v_num = k0 * a * np.sqrt(n_core**2 - n_clad**2)
    y = np.linspace(-y_max, y_max, n_pts)

    # Fundamental even mode: u * tan(u) = w_decay, with u in (0, min(v_num, pi/2))
    u_max_0 = min(v_num * (1.0 - 1e-12), 0.5 * np.pi * (1.0 - 1e-12))
    u0 = brentq(
        lambda u: u * np.tan(u) - np.sqrt(max(v_num**2 - u**2, 0.0)),
        1e-9,
        u_max_0,
    )
    w0_dec = np.sqrt(max(v_num**2 - u0**2, 0.0))
    ky0 = u0 / a
    gamma0 = w0_dec / a
    beta0 = np.sqrt((n_core * k0) ** 2 - ky0**2)
    neff0 = float(beta0 / k0)

    psi0 = np.where(
        np.abs(y) <= a,
        np.cos(ky0 * y),
        np.cos(u0) * np.exp(-gamma0 * (np.abs(y) - a)),
    )
    psi0 = psi0 / np.sqrt(np.trapezoid(psi0**2, y))

    # Confinement factor in [-a, a]
    mask_core = np.abs(y) <= a
    gamma_conf0 = float(np.trapezoid(psi0[mask_core] ** 2, y[mask_core]))

    # Second-moment Gaussian waist: w0^2 = 4 * int y^2 |psi0|^2 dy
    w0_moment = float(np.sqrt(4.0 * np.trapezoid((y**2) * (psi0**2), y)))

    # Maximum power-overlap Gaussian waist
    w_grid = np.linspace(0.6 * w0_moment, 1.4 * w0_moment, 400)
    overlaps = []
    for w_cand in w_grid:
        g_cand = (2.0 / (np.pi * w_cand**2)) ** 0.25 * np.exp(-((y / w_cand) ** 2))
        overlaps.append(float(np.trapezoid(psi0 * g_cand, y) ** 2))
    idx_best = int(np.argmax(overlaps))
    w0_overlap = float(w_grid[idx_best])

    out = {
        "y": y,
        "width": width,
        "wavelength": wavelength,
        "v_number": float(2.0 * v_num),  # full-width V = k0 * w * NA
        "v_half": float(v_num),  # half-width V_half = k0 * (w/2) * NA
        "neff_0": neff0,
        "psi_0": psi0,
        "w0_overlap": w0_overlap,
        "w0_moment": w0_moment,
        "max_gauss_overlap": float(overlaps[idx_best]),
        "confinement_0": gamma_conf0,
        "neff_1": None,
        "psi_1": None,
    }

    # First odd mode exists when half-width V > pi/2 (full-width V > pi)
    if v_num > 0.5 * np.pi + 1e-6:
        u_min_1 = 0.5 * np.pi * (1.0 + 1e-12)
        u_max_1 = min(v_num * (1.0 - 1e-12), np.pi * (1.0 - 1e-12))
        u1 = brentq(
            lambda u: -u / np.tan(u) - np.sqrt(max(v_num**2 - u**2, 0.0)),
            u_min_1,
            u_max_1,
        )
        w1_dec = np.sqrt(max(v_num**2 - u1**2, 0.0))
        ky1 = u1 / a
        gamma1 = w1_dec / a
        beta1 = np.sqrt((n_core * k0) ** 2 - ky1**2)
        neff1 = float(beta1 / k0)

        psi1 = np.where(
            np.abs(y) <= a,
            np.sin(ky1 * y),
            np.sign(y) * np.sin(u1) * np.exp(-gamma1 * (np.abs(y) - a)),
        )
        psi1 = psi1 / np.sqrt(np.trapezoid(psi1**2, y))
        out["neff_1"] = neff1
        out["psi_1"] = psi1

    return out


def slab_overlap_reflectivity(
    theta_deg: float | np.ndarray,
    slab_info: dict,
    mode_out: int = 0,
    n_ar: float | None = None,
    t_ar: float = 0.0,
) -> np.ndarray:
    """Compute exact step-index slab overlap reflectivity R_{0,mode_out}(theta_t).

    Evaluates the Marcuse (1989) tilted-facet mode-overlap integral using the
    exact step-index slab eigenfunctions ``psi_0(y)`` and ``psi_m(y)``:

        R_{0m}(theta_t) = R_f(theta_t) * |int psi_0(y) psi_m(y) exp(i * (beta_0 + beta_m) * y * tan(theta_t)) dy|^2

    Unlike the Gaussian approximation, the finite core boundary at ``y = +-w/2``
    produces a sidewall diffraction interference minimum near ``theta_t ~ 10 deg``
    followed by higher-angle sidelobes.

    Parameters
    ----------
    theta_deg : float or np.ndarray
        Facet tilt angle [deg].
    slab_info : dict
        Output dictionary from :func:`solve_1d_slab_modes`.
    mode_out : {0, 1}, optional
        Reflected guided mode index (0 for fundamental TE0, 1 for first odd TE1).
    n_ar : float or None, optional
        AR coating index [-]. Default is None.
    t_ar : float, optional
        AR coating thickness [m]. Default is 0.0.

    Returns
    -------
    R_0m : np.ndarray
        Modal power reflectivity into ``mode_out`` [-].
    """
    y = slab_info["y"]
    psi0 = slab_info["psi_0"]
    neff0 = slab_info["neff_0"]
    wl = slab_info["wavelength"]
    k0 = 2.0 * np.pi / wl

    if mode_out == 0:
        psim = psi0
        neffm = neff0
    elif mode_out == 1:
        if slab_info["psi_1"] is None:
            raise ValueError("Slab waveguide does not support the TE1 mode.")
        psim = slab_info["psi_1"]
        neffm = slab_info["neff_1"]
    else:
        raise ValueError(f"Unsupported mode_out={mode_out}")

    thetas = np.atleast_1d(np.asarray(theta_deg, dtype=float))
    r_f = transfer_matrix_ar_reflectance(
        thetas, n_inc=0.5 * (neff0 + neffm), wavelength=wl, n_ar=n_ar, t_ar=t_ar
    )
    prod = psi0 * psim
    q_vals = (neff0 + neffm) * k0 * np.tan(np.deg2rad(thetas))
    phase = np.exp(1j * np.outer(q_vals, y))
    ov = np.trapezoid(phase * prod[None, :], y, axis=1)
    r_modal = r_f * (np.abs(ov) ** 2)
    return r_modal[0] if np.ndim(theta_deg) == 0 else r_modal


def snell_farfield_angle_deg(
    theta_deg: float | np.ndarray,
    neff: float = N_CORE,
    n_out: float = 1.0,
) -> np.ndarray:
    """Compute Snell-refracted output beam angle relative to the facet normal [deg].

    In Zah et al. (1987), a waveguide tilted by ``theta_t = 7 deg`` from the
    cleaved facet normal refracts into air at ``arcsin(neff * sin(theta_t)) ~ 24 deg``
    relative to the facet normal.

    Parameters
    ----------
    theta_deg : float or np.ndarray
        Facet tilt angle [deg].
    neff : float, optional
        Effective refractive index of the guided mode [-].
    n_out : float, optional
        Exterior refractive index [-]. Default is 1.0.

    Returns
    -------
    theta_normal_deg : np.ndarray
        Refracted far-field peak angle relative to the facet normal [deg].
    """
    sin_out = (neff / n_out) * np.sin(np.deg2rad(theta_deg))
    return np.rad2deg(np.arcsin(np.clip(sin_out, -1.0, 1.0)))


def hakki_paoli_ripple_db(g_r_product: float | np.ndarray) -> np.ndarray:
    """Compute Fabry-Perot spectral ripple contrast [dB] from round-trip feedback.

    Uses the Hakki-Paoli peak-to-valley power ratio of a single-transverse-mode
    active resonator:

        Delta_G_dB = 20 * log10((1 + G_s * sqrt(R1 * R2)) / (1 - G_s * sqrt(R1 * R2)))

    Parameters
    ----------
    g_r_product : float or np.ndarray
        Dimensionless round-trip amplitude feedback ``gamma_rt = G_s * sqrt(R1 * R2)``
        in ``[0, 1)`` [-].

    Returns
    -------
    ripple_db : np.ndarray
        Peak-to-valley spectral modulation depth [dB].
    """
    gamma = np.clip(np.asarray(g_r_product, dtype=float), 0.0, 0.999999)
    return 20.0 * np.log10((1.0 + gamma) / (1.0 - gamma))


def sled_ase_spectrum(
    wavelengths: np.ndarray,
    g_peak_linear: float,
    r1: float,
    r2: float,
    neff: float = N_CORE,
    ng: float = 3.65,
    cavity_length: float = 380.0e-6,
    wl_center: float = 1.482e-6,
    fwhm_nm: float = 55.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the Fabry-Perot modulated ASE output spectrum of a SLED cavity.

    Models a single-pass power gain profile ``G_s(lambda)`` with Gaussian
    gain roll-off around ``wl_center`` and evaluates the round-trip resonator
    transfer function:

        S_ASE(lambda) = S_0(lambda) * (G_s - 1) * (1 + R1 * G_s) * (1 - R2)
                        / |1 - sqrt(R1 * R2) * G_s * exp(i * 4 * pi * ng * L / lambda)|^2

    Parameters
    ----------
    wavelengths : np.ndarray
        Free-space wavelength array [m].
    g_peak_linear : float
        Peak single-pass power gain ``G_s(wl_center)`` [-].
    r1 : float
        Rear-facet modal power reflectivity ``R_1`` [-].
    r2 : float
        Output-facet modal power reflectivity ``R_2`` [-].
    neff : float, optional
        Effective phase index [-].
    ng : float, optional
        Effective group index [-]. Default is 3.65 (InGaAsP at 1.49 um).
    cavity_length : float, optional
        Active waveguide length ``L`` [m]. Default is 380 um.
    wl_center : float, optional
        Peak gain wavelength [m]. Default is 1.482e-6 m (Zah et al. Fig. 4).
    fwhm_nm : float, optional
        Gain-assisted envelope FWHM [nm]. Default is 55.0 nm.

    Returns
    -------
    spectrum_norm : np.ndarray
        ASE power spectrum normalized to its envelope maximum [-].
    envelope_norm : np.ndarray
        Unmodulated travelling-wave ASE envelope (``R1 = R2 = 0``) [-].
    ripple_peak_db : float
        Peak-to-valley Fabry-Perot ripple at ``wl_center`` [dB].
    """
    sigma_wl = (fwhm_nm * 1e-9) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gain_shape = np.exp(-0.5 * ((wavelengths - wl_center) / sigma_wl) ** 2)
    g_s = np.power(g_peak_linear, gain_shape)

    phi_rt = 4.0 * np.pi * ng * cavity_length / wavelengths
    r_geom = np.sqrt(r1 * r2)
    denom = np.abs(1.0 - r_geom * g_s * np.exp(1j * phi_rt)) ** 2

    num = np.maximum(g_s - 1.0, 0.0) * (1.0 + r1 * g_s) * (1.0 - r2)
    spec = num / np.maximum(denom, 1e-18)
    env = np.maximum(g_s - 1.0, 0.0) * (1.0 + r1 * g_s) * (1.0 - r2)

    norm = float(np.max(env)) if np.max(env) > 0 else 1.0
    ripple_db = float(hakki_paoli_ripple_db(g_peak_linear * r_geom))
    return spec / norm, env / norm, ripple_db


def _build_tilted_facet_structures(
    theta_deg: float,
    width: float,
    h_cell: float,
    n_core: float = N_CORE,
    n_clad: float = N_CLAD,
    ar_coated: bool = False,
    n_ar: float = N_AR_COAT,
    t_ar: float = T_AR_COAT,
    include_facet: bool = True,
) -> list:
    """Build painter's-order CSG structures for the 2D tilted-facet waveguide."""
    zmin, zmax = -0.5 * h_cell, 0.5 * h_cell
    x_left, x_right = -10.0e-6, 10.0e-6
    y_bot, y_top = -10.0e-6, 10.0e-6

    bg_clad = Box(
        (0.0, 0.0, 0.0),
        (30.0e-6, 30.0e-6, 4.0 * h_cell),
        material=n_to_eps(n_clad),
    )
    core_slab = PolySlab(
        straight((x_left, 0.0), (x_right, 0.0), width=width),
        zmin,
        zmax,
        material=n_to_eps(n_core),
    )
    structures = [bg_clad, core_slab]
    if not include_facet:
        return structures

    th = np.deg2rad(theta_deg)
    tan_th = np.tan(th)
    cos_th = np.cos(th)

    if ar_coated and t_ar > 0.0:
        dx_ar = t_ar / cos_th
        ar_verts = np.array(
            [
                [y_bot * tan_th, y_bot],
                [x_right, y_bot],
                [x_right, y_top],
                [y_top * tan_th, y_top],
            ]
        )
        air_verts = np.array(
            [
                [y_bot * tan_th + dx_ar, y_bot],
                [x_right, y_bot],
                [x_right, y_top],
                [y_top * tan_th + dx_ar, y_top],
            ]
        )
        structures.append(PolySlab(ar_verts, zmin, zmax, material=n_to_eps(n_ar)))
        structures.append(PolySlab(air_verts, zmin, zmax, material=1.0))
    else:
        air_verts = np.array(
            [
                [y_bot * tan_th, y_bot],
                [x_right, y_bot],
                [x_right, y_top],
                [y_top * tan_th, y_top],
            ]
        )
        structures.append(PolySlab(air_verts, zmin, zmax, material=1.0))
    return structures


def create_tilted_facet_simulation(
    theta_deg: float = 0.0,
    width: float = WIDTH_SM,
    cpw: int = 14,
    wavelength: float = WAVELENGTH_0,
    n_core: float = N_CORE,
    n_clad: float = N_CLAD,
    ar_coated: bool = False,
    n_ar: float = N_AR_COAT,
    t_ar: float = T_AR_COAT,
    averaging: str = "kottke",
    n_pml: int = 10,
    include_facet: bool = True,
    n_freqs: int = 1,
    xmin_nom: float = -4.2e-6,
    xmax_nom: float = 1.8e-6,
    ymin: float = -6.0e-6,
    ymax: float = 6.0e-6,
) -> tuple[Simulation, dict]:
    """Create a locally refined 2D Wakis Simulation for the tilted-facet waveguide.

    Uses :class:`wakis.templates.Simulation` with ``mesh_overrides`` so that:

    1. The background grid satisfies ``>= 10`` cells per wavelength at the
       shortest broadband wavelength ``lambda_min = 1.43 um`` inside ``n_core``
       (``dx_background_nm <= 41.1 nm < 42.85 nm``).
    2. The quarter-wave AR coating interval ``x in [0, t_ar]`` has exact mesh
       nodes at ``x = 0.0`` and ``x = t_ar`` and at least ``>= 6`` cells across
       ``t_ar = 216.57 nm`` (``>= 4`` cells required by Rule 2).
    3. The longitudinal spacing ``dx = hx = t_ar / n_ar_cells`` is kept strictly
       uniform from ``xmin`` through the mode source, mode monitor, facet, and
       AR coating to ``xmax_ref``, eliminating numerical grid-impedance
       reflections upstream of the facet so the ``~10^-7`` Marcuse null at
       ``theta_t = 10 deg`` is cleanly resolved.
    4. The core waveguide walls ``y = +-width / 2`` and symmetry axis ``y = 0``
       lie on exact ``y`` mesh nodes with uniform spacing ``dy = hy`` across the
       guided-mode waist ``[-y_waist, +y_waist]``, while the single periodic
       ``z`` cell retains the background step ``hz`` so the 3D CFL time step is
       not penalized by ``dz``.

    Parameters
    ----------
    theta_deg : float, optional
        Facet tilt angle [deg]. Default is 0.0.
    width : float, optional
        Waveguide core width [m]. Default is 1.75e-6 m.
    cpw : int, optional
        Target cells per minimum wavelength ``lambda_0 / n_core`` in the core
        and facet refinement zone [-]. Default is 14.
    wavelength : float, optional
        Center free-space wavelength [m]. Default is 1.49e-6 m.
    n_core : float, optional
        Core effective refractive index [-].
    n_clad : float, optional
        Cladding effective refractive index [-].
    ar_coated : bool, optional
        If True, inserts a single-layer AR coating on the facet.
    n_ar : float, optional
        AR coating refractive index [-].
    t_ar : float, optional
        AR coating normal thickness [m].
    averaging : {"kottke", "staircase"}, optional
        Subpixel dielectric smoothing scheme. Default is ``"kottke"``.
    n_pml : int, optional
        Number of CPML cells on the x and y boundaries [-]. Default is 10.
    include_facet : bool, optional
        If True, includes the tilted facet at ``x = y * tan(theta_t)``.
    n_freqs : int, optional
        Number of DFT monitor frequencies [-]. Default is 1.
    xmin_nom : float, optional
        Nominal left domain boundary [m]. Default is -4.2e-6 m.
    xmax_nom : float, optional
        Nominal right domain boundary [m]. Default is 1.8e-6 m.
    ymin : float, optional
        Bottom domain boundary [m]. Default is -6.0e-6 m.
    ymax : float, optional
        Top domain boundary [m]. Default is 6.0e-6 m.

    Returns
    -------
    sim : Simulation
        Configured Wakis :class:`wakis.templates.Simulation` instance before
        building the heavy 3D solver fields.
    geom_info : dict
        Geometry and target cell-size parameters (``xmin``, ``xmax``, ``ymin``,
        ``ymax``, ``zmin``, ``zmax``, ``hx``, ``hy``, ``hz``, ``t_ar_eff``).
    """
    f_max_grid = max(C_LIGHT / 1.43e-6, C_LIGHT / wavelength)
    h_target = wavelength / (n_core * float(cpw))
    t_ar_eff = float(t_ar) if t_ar > 0.0 else T_AR_COAT

    n_ar_cells = max(6, int(np.ceil(t_ar_eff / h_target)))
    hx = t_ar_eff / n_ar_cells

    n_core_y = max(10, 2 * int(np.ceil(0.5 * width / h_target)))
    hy = width / n_core_y

    cpw_bg = max(10.5, 0.75 * float(cpw))
    hz = (C_LIGHT / f_max_grid) / (n_core * cpw_bg)
    zmin, zmax = -0.5 * hz, 0.5 * hz

    xmin = -hx * int(np.ceil(abs(xmin_nom) / hx))
    xmax_ref_target = 0.95e-6 if include_facet else xmax_nom
    n_right = max(1, int(np.ceil(max(xmax_ref_target - t_ar_eff, hx) / hx)))
    xmax_ref = t_ar_eff + n_right * hx
    xmax = max(xmax_nom, xmax_ref) if include_facet else xmax_ref

    y_waist_target = min(0.5 * width + 0.85e-6, abs(ymin))
    n_clad_y = max(2, int(np.ceil(max(y_waist_target - 0.5 * width, hy) / hy)))
    y_waist = min(0.5 * width + n_clad_y * hy, abs(ymin))

    structures = _build_tilted_facet_structures(
        theta_deg=theta_deg,
        width=width,
        h_cell=hz,
        n_core=n_core,
        n_clad=n_clad,
        ar_coated=ar_coated,
        n_ar=n_ar,
        t_ar=t_ar_eff,
        include_facet=include_facet,
    )

    mesh_overrides = [
        ((xmin, -y_waist, zmin), (0.0, y_waist, zmax), (hx, hy, np.inf)),
        ((0.0, -y_waist, zmin), (t_ar_eff, y_waist, zmax), (hx, hy, np.inf)),
        ((t_ar_eff, -y_waist, zmin), (xmax_ref, y_waist, zmax), (hx, hy, np.inf)),
    ]

    sim = Simulation(
        f_min=C_LIGHT / 1.55e-6,
        f_max=f_max_grid,
        structures=structures,
        boundaries=(["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"]),
        domain=(xmin, xmax, ymin, ymax, zmin, zmax),
        cells_per_wavelength=cpw_bg,
        n_pml=int(n_pml),
        n_freqs=int(n_freqs),
        cfln=0.75,
        averaging=averaging,
        mesh_overrides=mesh_overrides,
    )
    geom_info = {
        "xmin": float(xmin),
        "xmax": float(xmax),
        "ymin": float(ymin),
        "ymax": float(ymax),
        "zmin": float(zmin),
        "zmax": float(zmax),
        "hx": float(hx),
        "hy": float(hy),
        "hz": float(hz),
        "t_ar_eff": float(t_ar_eff),
    }
    return sim, geom_info


def inspect_simulation_mesh(
    sim: Simulation,
    min_feature: float = T_AR_COAT,
    wavelength: float = WAVELENGTH_0,
    run_time_extra_s: float = 160.0e-15,
    fwidth_rel: float = 0.22,
) -> dict:
    """Measure mesh resolution metrics directly from the 1D Simulation axes.

    Builds only the 1D coordinate arrays ``(x, y, z)`` if ``sim.grid`` has not
    yet been constructed, allowing fast mesh-only verification without
    allocating 3D FIT tensors or running the time-domain solver.

    Parameters
    ----------
    sim : Simulation
        Configured Wakis :class:`wakis.templates.Simulation` instance.
    min_feature : float, optional
        Smallest geometric feature thickness [m]. Default is ``T_AR_COAT``
        (216.57 nm quarter-wave AR layer).
    wavelength : float, optional
        Center free-space wavelength [m] used to estimate the pulse duration
        when ``n_steps`` is requested before running the solver.
    run_time_extra_s : float, optional
        Post-pulse ring-down time [s]. Default is 160.0e-15 s.
    fwidth_rel : float, optional
        Relative Gaussian source bandwidth ``fwidth / f0`` [-]. Default is 0.22.

    Returns
    -------
    metrics : dict
        Mesh resolution metrics including ``dx_background_nm``, ``dx_min_nm``,
        ``min_feature_nm``, ``cells_across_feature``, ``n_cells``, ``n_steps``,
        ``nx``, ``ny``, ``dt_s``, and ``x_node_err_nm``.
    """
    if sim.grid is not None:
        x = np.asarray(sim.grid.x, dtype=float)
        y = np.asarray(sim.grid.y, dtype=float)
        z = np.asarray(sim.grid.z, dtype=float)
    else:
        x = np.asarray(sim._axis(0), dtype=float)
        y = np.asarray(sim._axis(1), dtype=float)
        z = np.asarray(sim._axis(2), dtype=float)

    dx_arr = np.diff(x)
    dy_arr = np.diff(y)
    dz_arr = np.diff(z)

    dx_background_nm = float(
        max(np.max(dx_arr), np.max(dy_arr), np.max(dz_arr)) * 1e9
    )
    dx_min_nm = float(min(np.min(dx_arr), np.min(dy_arr)) * 1e9)
    min_feature_nm = float(min_feature * 1e9)

    x_centers = 0.5 * (x[:-1] + x[1:])
    cells_across_feature = int(
        np.sum((x_centers > 0.0) & (x_centers < float(min_feature)))
    )
    x_node_err_nm = float(
        max(np.min(np.abs(x - 0.0)), np.min(np.abs(x - float(min_feature)))) * 1e9
    )

    nx = int(len(dx_arr))
    ny = int(len(dy_arr))
    nz = int(len(dz_arr))
    n_cells = int(nx * ny * nz)

    inv_d2 = (
        1.0 / float(np.min(dx_arr)) ** 2
        + 1.0 / float(np.min(dy_arr)) ** 2
        + 1.0 / float(np.min(dz_arr)) ** 2
    )
    dt_s = float(sim.cfln / (C_LIGHT * np.sqrt(inv_d2)))
    f0 = C_LIGHT / wavelength
    pulse = GaussianSource(frequency=f0, fwidth=fwidth_rel * f0)
    t_max = pulse.end_time + run_time_extra_s
    n_steps = int(np.ceil(t_max / dt_s))

    return {
        "dx_background_nm": dx_background_nm,
        "dx_min_nm": dx_min_nm,
        "min_feature_nm": min_feature_nm,
        "cells_across_feature": cells_across_feature,
        "n_cells": n_cells,
        "n_steps": n_steps,
        "nx": nx,
        "ny": ny,
        "dt_s": dt_s,
        "x_node_err_nm": x_node_err_nm,
    }


def run_tilted_facet_fit(
    theta_deg: float,
    width: float = WIDTH_SM,
    cpw: int = 14,
    wavelength: float = WAVELENGTH_0,
    n_core: float = N_CORE,
    n_clad: float = N_CLAD,
    ar_coated: bool = False,
    n_ar: float = N_AR_COAT,
    t_ar: float = T_AR_COAT,
    averaging: str = "kottke",
    n_pml: int = 10,
    launch_mode_index: int = 0,
    compute_bimodal: bool = False,
    record_fields: bool = False,
    record_farfield: bool = False,
    record_movie: bool = False,
    freqs: np.ndarray | None = None,
) -> dict:
    """Run a 2D Wakis FIT simulation of a tilted waveguide facet.

    Launches a guided mode with :class:`wakis.mode_solver.ModeSource` and measures
    forward/backward modal amplitudes on a :class:`wakis.mode_solver.ModeMonitor`
    upstream of the tilted facet ``x = y * tan(theta_t)``.

    Parameters
    ----------
    theta_deg : float
        Facet tilt angle [deg].
    width : float, optional
        Waveguide core width [m]. Default is 1.75e-6 m.
    cpw : int, optional
        Cells per minimum wavelength ``lambda_0 / n_core`` in the core/facet
        refinement region [-]. Default is 14.
    wavelength : float, optional
        Center free-space wavelength [m]. Default is 1.49e-6 m.
    n_core : float, optional
        Core effective refractive index [-].
    n_clad : float, optional
        Cladding effective refractive index [-].
    ar_coated : bool, optional
        If True, inserts a single-layer AR coating of index ``n_ar`` and
        thickness ``t_ar`` on the tilted facet.
    n_ar : float, optional
        AR coating refractive index [-].
    t_ar : float, optional
        AR coating normal thickness [m].
    averaging : {"kottke", "staircase"}, optional
        Subpixel dielectric smoothing scheme. Default is ``"kottke"``.
    n_pml : int, optional
        Number of CPML cells on the x and y boundaries [-]. Default is 10.
    launch_mode_index : int, optional
        Guided mode index to launch (0 for TE0, 1 for TE1). Default is 0.
    compute_bimodal : bool, optional
        If True, solves 2 modes on the monitor to measure both ``R_00`` and
        ``R_01``.
    record_fields : bool, optional
        If True, records a 2D steady-state DFT field map over the x-y plane.
    record_farfield : bool, optional
        If True, places a :class:`wakis.near2far.Near2FarMonitor` in the air
        region and returns the 2D far-field radiation pattern ``U(theta)``.
    record_movie : bool, optional
        If True, saves time-domain ``Ey(x, y, t)`` snapshots for animation.
    freqs : np.ndarray or None, optional
        Frequency array [Hz] for DFT monitors. Defaults to ``[C_LIGHT / wavelength]``.

    Returns
    -------
    result : dict
        Simulation results including ``r00``, ``r01``, ``r00_spectrum``,
        ``neff_0``, ``neff_1``, ``runtime_s``, mesh resolution metrics, and
        optional field/far-field arrays.
    """
    t0 = time.perf_counter()
    f0 = C_LIGHT / wavelength
    if freqs is None:
        freqs = np.array([f0], dtype=float)
    else:
        freqs = np.asarray(freqs, dtype=float)
    idx_f0 = int(np.argmin(np.abs(freqs - f0)))

    sim, geom_info = create_tilted_facet_simulation(
        theta_deg=theta_deg,
        width=width,
        cpw=cpw,
        wavelength=wavelength,
        n_core=n_core,
        n_clad=n_clad,
        ar_coated=ar_coated,
        n_ar=n_ar,
        t_ar=t_ar,
        averaging=averaging,
        n_pml=n_pml,
        include_facet=True,
        n_freqs=len(freqs),
    )
    xmin = geom_info["xmin"]
    xmax = geom_info["xmax"]
    ymin = geom_info["ymin"]
    ymax = geom_info["ymax"]
    h_cell = geom_info["hx"]
    hz = geom_info["hz"]

    solver = sim.build()
    solver.ieps[:, :, :, "z"] = 0.0
    mesh_metrics = inspect_simulation_mesh(
        sim,
        min_feature=geom_info["t_ar_eff"],
        wavelength=wavelength,
    )

    pulse = GaussianSource(frequency=f0, fwidth=0.22 * f0)
    port_size = (0.0, 10.0e-6, np.inf)
    src = ModeSource(
        pulse,
        center=(-3.3e-6, 0.0, 0.0),
        size=port_size,
        direction="+",
        mode_index=launch_mode_index,
        mode_frequency=f0,
    )
    n_mon_modes = 2 if (compute_bimodal or launch_mode_index == 1) else 1
    mon = ModeMonitor(
        freqs,
        center=(-2.6e-6, 0.0, 0.0),
        size=port_size,
        n_modes=n_mon_modes,
        mode_frequency=f0,
    )
    monitors = [mon]

    dft_2d = None
    if record_fields:
        dft_2d = DFTMonitor(
            [f0],
            components=("Ex", "Ey", "Hz"),
            center=(0.5 * (xmin + xmax), 0.0, 0.0),
            size=(xmax - xmin, ymax - ymin, 0.0),
        )
        monitors.append(dft_2d)

    n2f = None
    if record_farfield:
        x_ff = 0.85e-6
        n2f = Near2FarMonitor(
            [f0],
            center=(x_ff, 0.0, 0.0),
            size=(0.0, 9.6e-6, hz),
            eps_r=1.0,
        )
        monitors.append(n2f)

    t_max = pulse.end_time + 160.0e-15
    nt = int(np.ceil(t_max / solver.dt))

    movie_frames = []
    movie_times_fs = []
    if record_movie:
        n_start = int(0.08 * nt)
        n_end = int(0.68 * nt)
        frame_steps = set(np.linspace(n_start, n_end, 18, dtype=int).tolist())
        dt = float(solver.dt)
        for step in range(nt):
            t_cur = step * dt
            src.update(solver, t_cur)
            solver.one_step()
            for m in monitors:
                m.update(solver, t_cur)
            if step in frame_steps:
                ey_snap = np.array(solver.E[:, :, 0, "y"], dtype=float).copy()
                movie_frames.append(ey_snap)
                movie_times_fs.append(float(t_cur * 1e15))
    else:
        solver.emsolve(
            nt,
            source=[src],
            monitors=monitors,
            stop_when_energy_decayed=1e-7,
        )

    ap0, am0 = mon.get_amplitudes(0)
    inc_amp = ap0
    if launch_mode_index == 1:
        ap1, _ = mon.get_amplitudes(1)
        inc_amp = ap1

    r00_spec = np.abs(am0 / inc_amp) ** 2
    r00_val = float(r00_spec[idx_f0])

    r01_spec = np.zeros_like(r00_spec)
    r01_val = 0.0
    neff1_val = np.nan
    ey1_profile = None
    if n_mon_modes >= 2:
        _, am1 = mon.get_amplitudes(1)
        r01_spec = np.abs(am1 / inc_amp) ** 2
        r01_val = float(r01_spec[idx_f0])
        neff1_val = float(mon.modes[0][1].neff)
        ey1_profile = np.real(mon.modes[0][1].e["b"][:, 0]).copy()

    eps_x = 1.0 / np.maximum(
        EPS_0 * np.array(solver.ieps[:, :, 0, "x"], dtype=float), 1e-30
    )
    eps_y = 1.0 / np.maximum(
        EPS_0 * np.array(solver.ieps[:, :, 0, "y"], dtype=float), 1e-30
    )

    res = {
        "theta_deg": float(theta_deg),
        "width": float(width),
        "cpw": int(cpw),
        "h_cell": float(h_cell),
        "dx_background_nm": mesh_metrics["dx_background_nm"],
        "dx_min_nm": mesh_metrics["dx_min_nm"],
        "min_feature_nm": mesh_metrics["min_feature_nm"],
        "cells_across_feature": mesh_metrics["cells_across_feature"],
        "n_cells": mesh_metrics["n_cells"],
        "n_steps": int(nt),
        "ar_coated": bool(ar_coated),
        "averaging": str(averaging),
        "n_pml": int(n_pml),
        "r00": r00_val,
        "r01": r01_val,
        "freqs": freqs,
        "r00_spectrum": r00_spec,
        "r01_spectrum": r01_spec,
        "neff_0": float(mon.modes[0][0].neff),
        "neff_1": float(neff1_val),
        "x_coords": np.asarray(solver.grid.x[:-1], dtype=float),
        "y_coords": np.asarray(solver.grid.y[:-1], dtype=float),
        "ey0_mode": np.real(mon.modes[0][0].e["b"][:, 0]).copy(),
        "ey1_mode": ey1_profile,
        "eps_x": eps_x,
        "eps_y": eps_y,
        "runtime_s": float(time.perf_counter() - t0),
    }

    if dft_2d is not None:
        ex_arr, _ = dft_2d.get_field("Ex")
        ey_arr, coords = dft_2d.get_field("Ey")
        hz_arr, _ = dft_2d.get_field("Hz")
        res["dft_x"] = np.asarray(coords[0], dtype=float)
        res["dft_y"] = np.asarray(coords[1], dtype=float)
        res["ex_2d"] = ex_arr[0, :, :, 0]
        res["ey_2d"] = ey_arr[0, :, :, 0]
        res["hz_2d"] = hz_arr[0, :, :, 0]

    if n2f is not None:
        phi_x_deg = np.linspace(-65.0, 65.0, 521)
        phi_x_rad = np.deg2rad(phi_x_deg)
        u_ff = n2f.radiation_intensity(theta=0.5 * np.pi, phi=phi_x_rad)[0]
        res["phi_x_deg"] = phi_x_deg
        res["theta_normal_deg"] = phi_x_deg + float(theta_deg)
        res["ff_intensity"] = np.asarray(u_ff, dtype=float)

    if record_movie:
        res["movie_frames"] = np.asarray(movie_frames)
        res["movie_times_fs"] = np.asarray(movie_times_fs)

    return res


def run_pml_floor_study(
    n_pml_list: list[int] | tuple[int, ...] = (6, 8, 10, 12, 14),
    cpw: int = 14,
    width: float = WIDTH_SM,
    wavelength: float = WAVELENGTH_0,
) -> dict:
    """Measure the straight-waveguide CPML parasitic reflectivity floor vs n_pml.

    Launches the fundamental mode into a straight waveguide entering the +x CPML
    absorber (no facet) and measures ``|a_0^-(f0) / a_0^+(f0)|^2`` directly on
    the :class:`wakis.mode_solver.ModeMonitor`.

    Parameters
    ----------
    n_pml_list : sequence of int, optional
        CPML thickness values in cells [-].
    cpw : int, optional
        Grid resolution in cells per wavelength [-]. Default is 14.
    width : float, optional
        Waveguide core width [m].
    wavelength : float, optional
        Free-space wavelength [m].

    Returns
    -------
    out : dict
        Dictionary with ``n_pml``, ``r_floor`` [-], and mesh resolution metrics.
    """
    f0 = C_LIGHT / wavelength
    r_floors = []
    mesh_metrics = None

    for n_pml in n_pml_list:
        sim, geom_info = create_tilted_facet_simulation(
            theta_deg=0.0,
            width=width,
            cpw=cpw,
            wavelength=wavelength,
            include_facet=False,
            n_pml=int(n_pml),
            n_freqs=3,
        )
        solver = sim.build()
        solver.ieps[:, :, :, "z"] = 0.0
        pulse = GaussianSource(frequency=f0, fwidth=0.22 * f0)
        port_size = (0.0, 10.0e-6, np.inf)
        src = ModeSource(
            pulse,
            center=(-3.3e-6, 0.0, 0.0),
            size=port_size,
            direction="+",
            mode_index=0,
            mode_frequency=f0,
        )
        mon = ModeMonitor(
            [f0],
            center=(-2.6e-6, 0.0, 0.0),
            size=port_size,
            n_modes=1,
            mode_frequency=f0,
        )
        nt = int(np.ceil((pulse.end_time + 160.0e-15) / solver.dt))
        if mesh_metrics is None:
            mesh_metrics = inspect_simulation_mesh(
                sim,
                min_feature=geom_info["t_ar_eff"],
                wavelength=wavelength,
            )
            mesh_metrics["n_steps"] = int(nt)
        solver.emsolve(
            nt,
            source=[src],
            monitors=[mon],
            stop_when_energy_decayed=1e-8,
        )
        ap, am = mon.get_amplitudes(0)
        r_floors.append(float(np.abs(am[0] / ap[0]) ** 2))

    return {
        "n_pml": np.asarray(n_pml_list, dtype=int),
        "r_floor": np.asarray(r_floors, dtype=float),
        "dx_background_nm": mesh_metrics["dx_background_nm"],
        "dx_min_nm": mesh_metrics["dx_min_nm"],
        "min_feature_nm": mesh_metrics["min_feature_nm"],
        "cells_across_feature": mesh_metrics["cells_across_feature"],
        "n_cells": mesh_metrics["n_cells"],
        "n_steps": mesh_metrics["n_steps"],
    }

def run_incoherent_dipole_ensemble_fit(
    cpw: int = 14,
    width: float = WIDTH_MM,
    wavelength: float = WAVELENGTH_0,
    n_unique_half: int = 5,
    seed: int = 42,
) -> dict:
    """Simulate an incoherent spontaneous-emission dipole ensemble in Wakis 2D FIT.

    Places ``N_dip = 2 * n_unique_half - 1`` point dipoles (:class:`CurrentSource`
    along ``y``) across the active core ``y in [-w/2, w/2]`` of the two-mode
    waveguide. Exploits mirror symmetry ``y_j <-> -y_j`` so only ``n_unique_half``
    FDTD/FIT runs are needed to obtain the exact complex fields ``Ey_j(x, y)``
    and guided mode amplitudes ``(a_{0,j}^+, a_{1,j}^+)`` for all ``N_dip`` dipoles.

    Compares:
    1. Incoherent power sum ``(1/N_dip) sum_j |a_{m,j}^+|^2`` (spontaneous emission).
    2. Coherent field sum ``|(1/N_dip) sum_j a_{m,j}^+|^2`` (phased array artifact).
    3. Random-phase Monte Carlo ensemble average over ``N_real`` realizations,
       showing the ``O(N_real^{-1/2})`` convergence to the incoherent power sum.

    Parameters
    ----------
    cpw : int, optional
        Grid resolution in cells per wavelength [-]. Default is 14.
    width : float, optional
        Waveguide core width [m]. Default is 2.80e-6 m (two-mode guide).
    wavelength : float, optional
        Free-space wavelength [m]. Default is 1.49e-6 m.
    n_unique_half : int, optional
        Number of unique non-negative dipole positions ``y_j >= 0`` [-].
        Default is 5, giving ``N_dip = 9`` dipoles across the active core.
    seed : int, optional
        Random number generator seed for stochastic phase realizations.

    Returns
    -------
    out : dict
        Dictionary with incoherent vs coherent 2D intensity maps, modal power
        fractions ``eta_te0`` and ``eta_te1``, quadrature convergence vs
        ``N_dip``, Monte Carlo convergence vs ``N_real``, and mesh resolution
        metrics.
    """
    f0 = C_LIGHT / wavelength
    xmin_nom, xmax_nom = -2.4e-6, 2.4e-6
    ymin, ymax = -5.0e-6, 5.0e-6

    y_pos_half = np.linspace(0.0, 0.42 * width, n_unique_half)
    ey_2d_half = []
    a0_half = []
    a1_half = []
    dft_x, dft_y = None, None
    mesh_metrics = None

    for y_d in y_pos_half:
        sim, geom_info = create_tilted_facet_simulation(
            theta_deg=0.0,
            width=width,
            cpw=cpw,
            wavelength=wavelength,
            include_facet=False,
            n_pml=8,
            n_freqs=3,
            xmin_nom=xmin_nom,
            xmax_nom=xmax_nom,
            ymin=ymin,
            ymax=ymax,
        )
        xmin = geom_info["xmin"]
        xmax = geom_info["xmax"]
        solver = sim.build()
        solver.ieps[:, :, :, "z"] = 0.0

        pulse = GaussianSource(frequency=f0, fwidth=0.25 * f0)
        src = CurrentSource(
            pulse,
            component="y",
            center=(-1.0e-6, float(y_d), 0.0),
            size=(0.0, 0.0, np.inf),
        )
        mon = ModeMonitor(
            [f0],
            center=(1.0e-6, 0.0, 0.0),
            size=(0.0, 8.6e-6, np.inf),
            n_modes=2,
            mode_frequency=f0,
        )
        dft_2d = DFTMonitor(
            [f0],
            components=("Ey",),
            center=(0.5 * (xmin + xmax), 0.0, 0.0),
            size=(xmax - xmin, ymax - ymin, 0.0),
        )
        nt = int(np.ceil((pulse.end_time + 110.0e-15) / solver.dt))
        if mesh_metrics is None:
            mesh_metrics = inspect_simulation_mesh(
                sim,
                min_feature=geom_info["t_ar_eff"],
                wavelength=wavelength,
                run_time_extra_s=110.0e-15,
                fwidth_rel=0.25,
            )
            mesh_metrics["n_steps"] = int(nt)
        solver.emsolve(
            nt,
            source=[src],
            monitors=[mon, dft_2d],
            stop_when_energy_decayed=1e-6,
        )
        ap0, _ = mon.get_amplitudes(0)
        ap1, _ = mon.get_amplitudes(1)
        a0_half.append(complex(ap0[0]))
        a1_half.append(complex(ap1[0]))
        ey_arr, coords = dft_2d.get_field("Ey")
        ey_2d_half.append(ey_arr[0, :, :, 0].copy())
        if dft_x is None:
            dft_x = np.asarray(coords[0], dtype=float)
            dft_y = np.asarray(coords[1], dtype=float)

    # Reconstruct full symmetric dipole ensemble: y_full has 2*n_unique_half - 1 dipoles
    y_dip_all = []
    ey_2d_all = []
    a0_all = []
    a1_all = []

    for idx in range(n_unique_half - 1, 0, -1):
        y_dip_all.append(-float(y_pos_half[idx]))
        ey_2d_all.append(ey_2d_half[idx][:, ::-1])
        # Even mode amplitude is unchanged under y -> -y; odd mode flips sign!
        a0_all.append(a0_half[idx])
        a1_all.append(-a1_half[idx])

    for idx in range(n_unique_half):
        y_dip_all.append(float(y_pos_half[idx]))
        ey_2d_all.append(ey_2d_half[idx])
        a0_all.append(a0_half[idx])
        a1_all.append(a1_half[idx])

    y_dip_all = np.asarray(y_dip_all)
    ey_2d_all = np.asarray(ey_2d_all)
    a0_arr = np.asarray(a0_all)
    a1_arr = np.asarray(a1_all)
    n_dip_total = len(y_dip_all)

    p0_incoh = float(np.mean(np.abs(a0_arr) ** 2))
    p1_incoh = float(np.mean(np.abs(a1_arr) ** 2))
    eta_te1_incoh = p1_incoh / max(p0_incoh + p1_incoh, 1e-30)

    p0_coh = float(np.abs(np.mean(a0_arr)) ** 2)
    p1_coh = float(np.abs(np.mean(a1_arr)) ** 2)
    eta_te1_coh = p1_coh / max(p0_coh + p1_coh, 1e-30)

    intensity_incoh = np.mean(np.abs(ey_2d_all) ** 2, axis=0)
    intensity_coh = np.abs(np.mean(ey_2d_all, axis=0)) ** 2

    rng = np.random.default_rng(seed)
    n_real_grid = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=int)
    n_trials = 60
    rel_err_mc_te0 = []
    rel_err_mc_te1 = []
    for n_r in n_real_grid:
        phases = np.exp(
            1j * rng.uniform(0.0, 2.0 * np.pi, size=(n_trials, n_r, n_dip_total))
        )
        a0_real = np.sum(phases * a0_arr[None, None, :], axis=2) / np.sqrt(n_dip_total)
        a1_real = np.sum(phases * a1_arr[None, None, :], axis=2) / np.sqrt(n_dip_total)
        p0_est = np.mean(np.abs(a0_real) ** 2, axis=1)
        p1_est = np.mean(np.abs(a1_real) ** 2, axis=1)
        rel_err_mc_te0.append(
            float(np.sqrt(np.mean(((p0_est - p0_incoh) / p0_incoh) ** 2)))
        )
        rel_err_mc_te1.append(
            float(np.sqrt(np.mean(((p1_est - p1_incoh) / max(p1_incoh, 1e-30)) ** 2)))
        )

    slab_mm = solve_1d_slab_modes(width=width, wavelength=wavelength)
    y_slab = slab_mm["y"]
    psi0_s = slab_mm["psi_0"]
    psi1_s = slab_mm["psi_1"]
    mask_c = np.abs(y_slab) <= 0.42 * width
    p0_exact_quad = float(
        np.trapezoid(psi0_s[mask_c] ** 2, y_slab[mask_c]) / (0.84 * width)
    )
    p1_exact_quad = float(
        np.trapezoid(psi1_s[mask_c] ** 2, y_slab[mask_c]) / (0.84 * width)
    )

    n_dip_grid = np.array([1, 3, 5, 7, 9, 13, 17, 25], dtype=int)
    quad_err_te0 = []
    quad_err_te1 = []
    for nd in n_dip_grid:
        pts = np.linspace(-0.42 * width, 0.42 * width, nd)
        v0 = np.interp(pts, y_slab, psi0_s**2)
        v1 = np.interp(pts, y_slab, psi1_s**2)
        quad_err_te0.append(float(abs(np.mean(v0) - p0_exact_quad) / p0_exact_quad))
        quad_err_te1.append(float(abs(np.mean(v1) - p1_exact_quad) / p1_exact_quad))

    return {
        "x_coords": dft_x,
        "y_coords": dft_y,
        "y_dip_all": y_dip_all,
        "a0_dipoles": np.abs(a0_arr) ** 2,
        "a1_dipoles": np.abs(a1_arr) ** 2,
        "p0_incoh": p0_incoh,
        "p1_incoh": p1_incoh,
        "eta_te1_incoh": float(eta_te1_incoh),
        "p0_coh": p0_coh,
        "p1_coh": p1_coh,
        "eta_te1_coh": float(eta_te1_coh),
        "intensity_incoh": intensity_incoh,
        "intensity_coh": intensity_coh,
        "n_real_grid": n_real_grid,
        "rel_err_mc_te0": np.asarray(rel_err_mc_te0),
        "rel_err_mc_te1": np.asarray(rel_err_mc_te1),
        "n_dip_grid": n_dip_grid,
        "quad_err_te0": np.asarray(quad_err_te0),
        "quad_err_te1": np.asarray(quad_err_te1),
        "dx_background_nm": mesh_metrics["dx_background_nm"],
        "dx_min_nm": mesh_metrics["dx_min_nm"],
        "min_feature_nm": mesh_metrics["min_feature_nm"],
        "cells_across_feature": mesh_metrics["cells_across_feature"],
        "n_cells": mesh_metrics["n_cells"],
        "n_steps": mesh_metrics["n_steps"],
    }


def build_sled_schematic_objects(
    theta_deg: float = 7.0,
    width: float = WIDTH_SM,
    cpw: int = 14,
    ar_coated: bool = True,
    n_pml: int = 10,
) -> dict:
    """Build lightweight solver/grid/source/monitor objects for tilted-facet schematic."""
    from types import SimpleNamespace
    from wakis import GridFIT3D

    sim, geom_info = create_tilted_facet_simulation(
        theta_deg=theta_deg,
        width=width,
        cpw=cpw,
        ar_coated=ar_coated,
        n_pml=n_pml,
        include_facet=True,
    )
    x = np.asarray(sim._axis(0), dtype=float)
    y = np.asarray(sim._axis(1), dtype=float)
    z = np.asarray(sim._axis(2), dtype=float)
    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver_stub = SimpleNamespace(
        grid=grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=n_pml,
    )
    structures = sim.structures
    names = ["Cladding (n = 3.311)", f"Ridge core (w = {width*1e6:.2f} µm, n = 3.337)"]
    if ar_coated:
        names.extend(["λ/4 AR film (216.6 nm, n = 1.72)", "Air (n = 1.00)"])
    else:
        names.append("Air (n = 1.00)")
    for st, nm in zip(structures, names):
        st.name = nm

    pulse = GaussianSource(frequency=FREQ_0, fwidth=0.22 * FREQ_0)
    port_size = (0.0, 9.6e-6, np.inf)
    src = ModeSource(
        pulse,
        center=(-3.3e-6, 0.0, 0.0),
        size=port_size,
        direction="+",
        mode_index=0,
        mode_frequency=FREQ_0,
    )
    src.normal = 0
    src.index = int(np.argmin(np.abs(x - (-3.3e-6))))
    mon = ModeMonitor(
        [FREQ_0],
        center=(-2.6e-6, 0.0, 0.0),
        size=port_size,
        n_modes=1,
        mode_frequency=FREQ_0,
    )
    n2f = Near2FarMonitor(
        [FREQ_0],
        center=(0.85e-6, 0.0, 0.0),
        size=(0.0, 9.6e-6, geom_info["hz"]),
        eps_r=1.0,
    )
    n2f.label = "Near2Far plane (x = 0.85 µm)"
    mesh_info = inspect_simulation_mesh(sim, min_feature=geom_info["t_ar_eff"])
    return {
        "solver": solver_stub,
        "grid": grid,
        "structures": structures,
        "sources": [src],
        "monitors": [mon, n2f],
        "geom_info": geom_info,
        "mesh_info": mesh_info,
    }


def build_incoherent_dipole_schematic_objects(
    width: float = WIDTH_MM,
    cpw: int = 14,
    n_unique_half: int = 5,
    n_pml: int = 8,
) -> dict:
    """Build lightweight solver/grid/source/monitor objects for the 9-dipole ensemble."""
    from types import SimpleNamespace
    from wakis import GridFIT3D

    sim, geom_info = create_tilted_facet_simulation(
        theta_deg=0.0,
        width=width,
        cpw=cpw,
        ar_coated=False,
        include_facet=False,
        n_pml=n_pml,
        xmin_nom=-2.4e-6,
        xmax_nom=2.4e-6,
        ymin=-5.0e-6,
        ymax=5.0e-6,
    )
    x = np.asarray(sim._axis(0), dtype=float)
    y = np.asarray(sim._axis(1), dtype=float)
    z = np.asarray(sim._axis(2), dtype=float)
    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver_stub = SimpleNamespace(
        grid=grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=n_pml,
    )
    structures = sim.structures
    structures[0].name = "Cladding (n = 3.311)"
    structures[1].name = f"Bimodal core (w₂ = {width*1e6:.2f} µm, n = 3.337)"

    pulse = GaussianSource(frequency=FREQ_0, fwidth=0.25 * FREQ_0)
    y_dips = np.linspace(-0.42 * width, 0.42 * width, 2 * n_unique_half - 1)
    sources = [
        CurrentSource(
            pulse,
            component="y",
            center=(-1.0e-6, float(yd), 0.0),
            size=(0.0, 0.0, np.inf),
        )
        for yd in y_dips
    ]
    mon = ModeMonitor(
        [FREQ_0],
        center=(1.0e-6, 0.0, 0.0),
        size=(0.0, 8.6e-6, np.inf),
        n_modes=2,
        mode_frequency=FREQ_0,
    )
    return {
        "solver": solver_stub,
        "grid": grid,
        "structures": structures,
        "sources": sources,
        "monitors": [mon],
        "y_dips_um": y_dips * 1e6,
        "geom_info": geom_info,
    }


def tidy_sled_schematic_axes(
    ax,
    xlim: tuple[float, float] = (-4.2, 1.8),
    ylim: tuple[float, float] = (-6.0, 7.2),
    is_dipole: bool = False,
    n_pml: int = 10,
    fontsize: float = 7.8,
) -> None:
    """Declutter internal schematic text and place clear region/boundary callouts."""
    ax.set_aspect("auto")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(ax.get_title(), pad=22)
    drop_keys = ("boundary:", "Out-of-plane", "Cladding", "Ridge core", "Bimodal core", "AR film", "Air (")
    src_texts = []
    for t_obj in list(ax.texts):
        s = t_obj.get_text()
        if any(k in s for k in drop_keys):
            t_obj.remove()
        elif "CurrentSource" in s:
            src_texts.append(t_obj)
    if len(src_texts) > 1:
        for t_obj in src_texts[1:]:
            t_obj.remove()
        src_texts[0].set_text("9 incoherent dipoles ($J_y$, random $\\phi_j$)")
        src_texts[0].set_position((-0.85, 1.55))

    bbox_kw = {
        "boxstyle": "round,pad=0.14",
        "facecolor": "white",
        "edgecolor": "#555555",
        "linewidth": 0.5,
        "alpha": 0.88,
    }
    ax.text(
        xlim[0] + 0.15,
        ylim[1] - 1.65,
        f"±x, ±y: CPML ({n_pml} cells) | ±z: Periodic",
        ha="left",
        va="center",
        fontsize=fontsize - 0.4,
        color="#222222",
        bbox=bbox_kw,
        zorder=9,
    )
    ax.text(
        xlim[0] + 0.18,
        -3.5 if not is_dipole else -3.3,
        "Cladding (n = 3.311)",
        ha="left",
        va="center",
        fontsize=fontsize,
        color="#111111",
        bbox=bbox_kw,
        zorder=9,
    )
    if is_dipole:
        ax.text(
            0.0,
            0.0,
            "Active core (w₂ = 2.80 µm, n = 3.337)",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#111111",
            bbox=bbox_kw,
            zorder=9,
        )
    else:
        ax.text(
            -1.45,
            0.0,
            "Ridge core (w₁ = 1.75 µm, n = 3.337)",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#111111",
            bbox=bbox_kw,
            zorder=9,
        )
        ax.text(
            1.15,
            -3.5,
            "Air (n = 1.00)",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#111111",
            bbox=bbox_kw,
            zorder=9,
        )
        ax.annotate(
            "λ/4 AR film (216.6 nm, n = 1.72)\nTilted facet θt = 7°",
            xy=(0.11, 2.2),
            xytext=(-1.65, 3.65),
            ha="center",
            va="center",
            fontsize=fontsize - 0.2,
            color="#111111",
            bbox=bbox_kw,
            arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 0.8},
            zorder=9,
        )


def overlay_sled_geometry(
    ax,
    theta_deg: float = 7.0,
    width: float = WIDTH_SM,
    ar_coated: bool = False,
    is_dipole_ensemble: bool = False,
    dipole_y_um: np.ndarray | None = None,
    line_color: str = "white",
    show_monitors: bool = True,
    show_snell_beam: bool = False,
    fontsize: float = 6.5,
) -> None:
    """Overlay SLED waveguide core, tilted facet, AR layer, CPML, and monitors/dipoles."""
    from wakis import schematic

    if is_dipole_ensemble:
        obj = build_incoherent_dipole_schematic_objects(width=width, cpw=10)
    else:
        obj = build_sled_schematic_objects(
            theta_deg=theta_deg, width=width, cpw=10, ar_coated=ar_coated
        )

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

    bbox_kw = {
        "boxstyle": "round,pad=0.14",
        "facecolor": "white",
        "edgecolor": "#555555",
        "linewidth": 0.5,
        "alpha": 0.82,
    }
    w_half_um = 0.5 * width * 1e6

    if is_dipole_ensemble:
        x_lo, x_hi = -2.4, 2.4
        ax.plot([x_lo, x_hi], [w_half_um, w_half_um], ls="--", lw=1.1, color=line_color, zorder=7)
        ax.plot([x_lo, x_hi], [-w_half_um, -w_half_um], ls="--", lw=1.1, color=line_color, zorder=7)
        if dipole_y_um is not None:
            ax.plot(
                np.full_like(dipole_y_um, -1.0),
                dipole_y_um,
                "*",
                color="#00e5ff" if line_color == "white" else "#d95f02",
                ms=6.5,
                mec="black",
                mew=0.4,
                zorder=9,
            )
            ax.text(
                -1.0,
                float(np.max(dipole_y_um)) + 0.55,
                "9 dipoles $J_y$",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color="#111111",
                bbox=bbox_kw,
                zorder=9,
            )
        if show_monitors:
            ax.plot([1.0, 1.0], [-4.3, 4.3], ls="-.", lw=1.1, color="#2ca25f", zorder=8)
            ax.text(
                1.05,
                -4.1,
                "ModeMonitor\n(TE0, TE1)",
                ha="left",
                va="bottom",
                fontsize=fontsize - 0.3,
                color="#1b7837",
                bbox=bbox_kw,
                zorder=9,
            )
        ax.text(
            -2.25,
            -4.2,
            f"Core w₂ = {width*1e6:.2f} µm (n = 3.337)\nCladding n = 3.311 | CPML",
            ha="left",
            va="bottom",
            fontsize=fontsize - 0.3,
            color="#111111",
            bbox=bbox_kw,
            zorder=9,
        )
        return

    # Tilted-facet waveguide outlines
    th_rad = np.deg2rad(theta_deg)
    tan_th = np.tan(th_rad)
    y_span = np.linspace(-5.2, 5.2, 200)
    x_fac = y_span * tan_th
    x_top_end = w_half_um * tan_th
    x_bot_end = -w_half_um * tan_th
    ax.plot([-3.8, x_top_end], [w_half_um, w_half_um], ls=":", lw=1.1, color=line_color, zorder=7)
    ax.plot([-3.8, x_bot_end], [-w_half_um, -w_half_um], ls=":", lw=1.1, color=line_color, zorder=7)
    ax.plot(x_fac, y_span, ls="--", lw=1.35, color=line_color, zorder=7)
    if ar_coated:
        dx_ar_um = (T_AR_COAT / np.cos(th_rad)) * 1e6
        ax.plot(x_fac + dx_ar_um, y_span, ls="-", lw=1.0, color="#ffd92f", zorder=7)

    if show_monitors:
        ax.plot([-3.3, -3.3], [-4.8, 4.8], ls="--", lw=1.0, color="#e6550d", alpha=0.88, zorder=7)
        ax.plot([-2.6, -2.6], [-4.8, 4.8], ls="-.", lw=1.0, color="#2ca25f", alpha=0.88, zorder=7)
        ax.plot([0.85, 0.85], [-4.8, 4.8], ls=":", lw=1.1, color="#3182bd", alpha=0.88, zorder=7)
        ax.text(-3.28, 4.45, "ModeSource", fontsize=fontsize - 0.4, color="#a63603", bbox=bbox_kw, zorder=9)
        ax.text(-2.55, -4.65, "ModeMonitor", fontsize=fontsize - 0.4, color="#006d2c", bbox=bbox_kw, zorder=9)
        ax.text(0.82, 4.45, "Near2Far", ha="right", fontsize=fontsize - 0.4, color="#08519c", bbox=bbox_kw, zorder=9)

    ax.text(
        -3.65,
        0.0,
        f"Core (n = 3.337)\nw₁ = {width*1e6:.2f} µm",
        ha="left",
        va="center",
        fontsize=fontsize - 0.2,
        color="#111111",
        bbox=bbox_kw,
        zorder=9,
    )
    ax.text(
        -3.65,
        -3.4,
        "Cladding (n = 3.311)",
        ha="left",
        va="center",
        fontsize=fontsize - 0.2,
        color="#111111",
        bbox=bbox_kw,
        zorder=9,
    )
    ax.text(
        1.48,
        -3.4,
        f"Air (n = 1.00)\nFacet θt = {theta_deg:.0f}°",
        ha="right",
        va="center",
        fontsize=fontsize - 0.2,
        color="#111111",
        bbox=bbox_kw,
        zorder=9,
    )
    if show_snell_beam and abs(theta_deg - 7.0) < 1e-3:
        # Refracted beam at +24 deg from facet normal (+17 deg from +x axis)
        ang_x = np.deg2rad(24.0 - 7.0)
        ax.annotate(
            "Snell 24°",
            xy=(1.25 * np.cos(ang_x), 1.25 * np.sin(ang_x)),
            xytext=(0.15, 0.05),
            fontsize=fontsize - 0.3,
            color="#d95f02",
            bbox=bbox_kw,
            arrowprops={"arrowstyle": "->", "color": "#d95f02", "lw": 1.1},
            zorder=9,
        )

