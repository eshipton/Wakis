"""Physics, 3D FDTD/FIT runners, and plotting helpers for Tutorial 8.2 (Cherenkov radiation)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.constants import c as c_light
from scipy.constants import epsilon_0, mu_0
from scipy.fft import dstn, idstn, irfft, rfft, rfftfreq
from scipy.special import k0, k1

import wakis
from wakis import FluxMonitor, GridFIT3D, SolverFIT3D
from wakis.dispersive import DispersiveMaterial, Lorentz
from wakis.estimate import estimate, preflight
from wakis.sources import Beam
from wakis.visualize import (
    box_bounds,
    cpml_inner_bounds,
    plot_scene,
    save_plotter,
    setup_rendering,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

Z0_VAC = float(np.sqrt(mu_0 / epsilon_0))
DEFAULT_MEM_LIMIT_MB = 1800.0
DEFAULT_Q_C = 1.0e-9
DEFAULT_SIGMAZ_M = 1.5e-3
DEFAULT_N_DIEL = 1.5
ANALYSED_FMIN_HZ = 12.0e9
ANALYSED_FMAX_HZ = 30.0e9


# ---------------------------------------------------------------------------
# Analytical theory helpers
# ---------------------------------------------------------------------------


def cherenkov_threshold_beta(n_refr: float | np.ndarray) -> float | np.ndarray:
    """Return the Cherenkov threshold velocity beta_th = 1 / n.

    Parameters
    ----------
    n_refr : float or ndarray
        Real refractive index of the medium [-].

    Returns
    -------
    beta_th : float or ndarray
        Threshold normalized speed v_th / c = 1 / n [-].
    """
    return 1.0 / np.asarray(n_refr, dtype=float)


def cherenkov_cone_angle(beta: float | np.ndarray, n_refr: float | np.ndarray) -> dict[str, np.ndarray]:
    """Compute the analytical Cherenkov wavevector angle theta_c and Mach half-angle alpha_c.

    Parameters
    ----------
    beta : float or ndarray
        Normalized particle speed v / c [-].
    n_refr : float or ndarray
        Refractive index n of the medium [-].

    Returns
    -------
    out : dict of ndarray
        Dictionary with keys:
        - ``above_threshold`` : bool array where ``beta * n > 1``
        - ``cos_theta_c`` : ``1 / (beta * n)`` (clipped to ``<= 1`` below threshold)
        - ``theta_c_rad`` : wavevector angle ``arccos(1 / (beta * n))`` [rad] (NaN below threshold)
        - ``theta_c_deg`` : wavevector angle [deg] (NaN below threshold)
        - ``shock_half_angle_deg`` : Mach cone half-angle ``90 - theta_c_deg`` [deg]
    """
    b = np.asarray(beta, dtype=float)
    n = np.asarray(n_refr, dtype=float)
    bn = b * n
    above = bn > 1.0
    cos_tc = np.where(above, 1.0 / np.maximum(bn, 1.0), np.nan)
    theta_rad = np.arccos(cos_tc)
    theta_deg = np.degrees(theta_rad)
    alpha_deg = 90.0 - theta_deg
    return {
        "above_threshold": above,
        "cos_theta_c": cos_tc,
        "theta_c_rad": theta_rad,
        "theta_c_deg": theta_deg,
        "shock_half_angle_deg": alpha_deg,
    }


def bunch_form_factor(
    freqs_hz: np.ndarray,
    beta: float,
    sigmaz_m: float,
) -> np.ndarray:
    """Power form factor F(omega) = exp(-(omega * sigma_z / v)^2) of a rigid Gaussian bunch.

    A rigid Gaussian line bunch with longitudinal charge density
    lambda(zeta) = (q / (sqrt(2*pi)*sigma_z)) * exp(-zeta^2 / (2*sigma_z^2))
    has current Fourier transform proportional to exp(-(k_z * sigma_z)^2 / 2)
    with k_z = omega / v, so the radiated spectral power scales with the
    coherent bunch form factor |lambda_hat(omega/v)|^2 = exp(-(omega*sigma_z/v)^2).

    Parameters
    ----------
    freqs_hz : ndarray
        Frequencies [Hz].
    beta : float
        Normalized bunch velocity v / c [-].
    sigmaz_m : float
        RMS bunch length sigma_z [m].

    Returns
    -------
    form_factor : ndarray
        Dimensionless power form factor F(omega) in [0, 1] [-].
    """
    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    v = float(beta) * c_light
    return np.exp(-((omega * float(sigmaz_m) / v) ** 2))


def frank_tamm_spectrum(
    freqs_hz: np.ndarray,
    beta: float,
    n_complex: float | complex | np.ndarray,
    q_c: float = DEFAULT_Q_C,
    sigmaz_m: float | None = DEFAULT_SIGMAZ_M,
    mu_r: float = 1.0,
    r_obs_m: float | None = None,
) -> dict[str, np.ndarray]:
    """Analytical Frank-Tamm spectral energy density per unit angular frequency and length.

    The Frank-Tamm formula in SI units is:
        d^2 W / (d omega dz) = (q^2 * mu * omega / (4 * pi)) * (1 - 1 / (beta^2 * n(omega)^2))
    for frequencies where beta * Re(n(omega)) > 1, and zero below threshold.
    For a finite Gaussian bunch of RMS length sigma_z, coherent radiation is
    multiplied by the bunch form factor F(omega) = exp(-(omega * sigma_z / v)^2).

    Parameters
    ----------
    freqs_hz : ndarray
        Frequencies [Hz].
    beta : float
        Normalized velocity v / c [-].
    n_complex : float, complex, or ndarray
        Complex refractive index n(omega) = sqrt(eps_r(omega) * mu_r) [-].
    q_c : float, optional
        Total bunch charge [C].
    sigmaz_m : float or None, optional
        RMS bunch length [m]. If None or 0, returns the point-charge spectrum.
    mu_r : float, optional
        Relative permeability [-].
    r_obs_m : float or None, optional
        Radial distance of the cylindrical/box flux surface [m]; when provided
        for a lossy dispersive medium, includes radial attenuation
        ``exp(-2 * |Im(k_perp)| * r_obs)``.

    Returns
    -------
    out : dict of ndarray
        Dictionary with keys:
        - ``omega`` : angular frequency [rad/s]
        - ``n_real`` : real part of refractive index [-]
        - ``above_threshold`` : bool mask ``beta * n_real > 1``
        - ``form_factor`` : Gaussian bunch form factor [-]
        - ``d2W_domega_dz_point`` : point-charge Frank-Tamm spectrum [J s / m]
        - ``d2W_domega_dz_bunch`` : bunch-suppressed Frank-Tamm spectrum [J s / m]
        - ``d2W_domega_dz_attenuated`` : spectrum including radial damping at ``r_obs_m`` [J s / m]
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    omega = 2.0 * np.pi * freqs
    n_arr = np.broadcast_to(np.asarray(n_complex, dtype= np.complex128), freqs.shape)
    n_real = np.real(n_arr)
    above = (float(beta) * n_real) > 1.0
    factor = np.where(above, 1.0 - 1.0 / np.maximum((float(beta) * n_real) ** 2, 1.0), 0.0)
    pref = (float(q_c) ** 2) * (float(mu_r) * mu_0) * omega / (4.0 * np.pi)
    d2w_point = pref * factor
    if sigmaz_m is None or sigmaz_m <= 0.0:
        ff = np.ones_like(freqs)
    else:
        ff = bunch_form_factor(freqs, beta, sigmaz_m)
    d2w_bunch = d2w_point * ff
    if r_obs_m is not None and np.any(np.abs(np.imag(n_arr)) > 0):
        eps_mu = n_arr**2
        k_perp = (omega / c_light) * np.sqrt(eps_mu - 1.0 / (float(beta) ** 2) + 0j)
        atten = np.exp(-2.0 * np.abs(np.imag(k_perp)) * float(r_obs_m))
        d2w_atten = d2w_bunch * atten
    else:
        d2w_atten = d2w_bunch.copy()
    return {
        "omega": omega,
        "n_real": n_real,
        "above_threshold": above,
        "form_factor": ff,
        "d2W_domega_dz_point": d2w_point,
        "d2W_domega_dz_bunch": d2w_bunch,
        "d2W_domega_dz_attenuated": d2w_atten,
    }


def subthreshold_coulomb_field(
    r_m: float,
    zeta_m: np.ndarray,
    beta: float,
    n_refr: float,
    q_c: float = DEFAULT_Q_C,
    sigmaz_m: float = DEFAULT_SIGMAZ_M,
) -> dict[str, np.ndarray]:
    """Analytical Lorentz-contracted Coulomb field of a Gaussian bunch below Cherenkov threshold.

    When beta * n < 1, define the effective contraction factor
    gamma_n = 1 / sqrt(1 - beta^2 * n^2). For each longitudinal wavenumber k_z,
    a point charge has radial and longitudinal Fourier components involving
    modified Bessel functions K_1(k_z r / gamma_n) and K_0(k_z r / gamma_n),
    which decay exponentially at large r (evanescent, non-radiating).

    Parameters
    ----------
    r_m : float
        Transverse radial distance r = sqrt(x^2 + y^2) [m].
    zeta_m : ndarray
        Comoving coordinate zeta = z - v*t [m].
    beta : float
        Normalized speed v / c [-] with ``beta * n_refr < 1``.
    n_refr : float
        Refractive index n [-].
    q_c : float, optional
        Bunch charge [C].
    sigmaz_m : float, optional
        RMS bunch length [m].

    Returns
    -------
    out : dict of ndarray
        ``Ez`` [V/m] and ``Er`` [V/m] on ``zeta_m``.
    """
    bn = float(beta) * float(n_refr)
    if bn >= 1.0:
        raise ValueError(f"subthreshold_coulomb_field requires beta*n < 1, got {bn:.4f}")
    gamma_n = 1.0 / np.sqrt(1.0 - bn**2)
    eps = (float(n_refr) ** 2) * epsilon_0
    zeta = np.asarray(zeta_m, dtype=float)
    # Quadrature over k_z in [0, 8 / sigmaz]
    kz_max = 8.0 / float(sigmaz_m)
    kz = np.linspace(1e-4 / float(sigmaz_m), kz_max, 400)
    dkz = kz[1] - kz[0]
    arg = kz * float(r_m) / gamma_n
    gauss_k = np.exp(-0.5 * (kz * float(sigmaz_m)) ** 2)
    # In the Lorenz gauge, the comoving scalar potential Phi(r, zeta) > 0 is even in zeta = z - v*t:
    #   Phi(r, zeta) = (q / (2*pi^2 * eps)) * int_0^inf K0(kz*r/gamma_n) * exp(-kz^2*sigmaz^2/2) * cos(kz*zeta) dkz
    # Since A_z = (beta^2 * n^2 / v) * Phi and d/dt = -v * d/dzeta, the longitudinal field is
    #   Ez = -dPhi/dz - dA_z/dt = -(1 - beta^2 * n^2) * dPhi/dzeta = -(1 / gamma_n^2) * dPhi/dzeta.
    # Differentiating cos(kz * zeta) with respect to zeta introduces a second minus sign (-kz * sin(kz * zeta)),
    # so the two minus signs cancel: for q > 0, Ez(r, zeta > 0) > 0 ahead of the bunch and Ez(r, zeta < 0) < 0 behind.
    #   Ez(r, zeta) = + (q / (2*pi^2 * eps * gamma_n^2)) * int_0^inf kz * K0(kz*r/gamma_n) * exp(-kz^2*sigmaz^2/2) * sin(kz*zeta) dkz
    #   Er(r, zeta) = + (q / (2*pi^2 * eps * gamma_n))   * int_0^inf kz * K1(kz*r/gamma_n) * exp(-kz^2*sigmaz^2/2) * cos(kz*zeta) dkz
    phase = np.outer(zeta, kz)
    pref_z = float(q_c) / (2.0 * (np.pi**2) * eps * (gamma_n**2))
    pref_r = float(q_c) / (2.0 * (np.pi**2) * eps * gamma_n)
    ez = pref_z * (np.sin(phase) @ (kz * k0(arg) * gauss_k)) * dkz
    er = pref_r * (np.cos(phase) @ (kz * k1(arg) * gauss_k)) * dkz
    return {"Ez": ez, "Er": er, "gamma_n": np.array(gamma_n)}


def yee_vs_tete_phase_velocity_z(
    kz_rad_m: np.ndarray,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    cfl: float = 1.0,
) -> dict[str, np.ndarray]:
    """Return the numerical phase velocity v_pz(k_z)/c along z for Yee and TETE in vacuum.

    Parameters
    ----------
    kz_rad_m : ndarray
        Longitudinal wavenumber k_z [rad/m].
    dx_m, dy_m, dz_m : float
        Cell sizes [m].
    cfl : float, optional
        Courant factor for the Yee scheme [-].

    Returns
    -------
    out : dict of ndarray
        ``vp_yee_over_c``, ``vp_tete_over_c``, and ``ppw`` (cells per wavelength).
    """
    kz = np.asarray(kz_rad_m, dtype=float)
    dt_yee = float(cfl) / (
        c_light * np.sqrt(1.0 / dx_m**2 + 1.0 / dy_m**2 + 1.0 / dz_m**2)
    )
    s_z = c_light * dt_yee / dz_m
    arg = np.clip(s_z * np.sin(0.5 * kz * dz_m), -1.0, 1.0)
    omega_yee = (2.0 / dt_yee) * np.arcsin(arg)
    vp_yee = np.where(kz > 0, omega_yee / (c_light * kz), 1.0)
    vp_tete = np.ones_like(vp_yee)
    ppw = np.where(kz > 0, 2.0 * np.pi / (kz * dz_m), np.inf)
    return {
        "vp_yee_over_c": vp_yee,
        "vp_tete_over_c": vp_tete,
        "ppw": ppw,
        "dt_yee": np.array(dt_yee),
        "dt_tete": np.array(dz_m / c_light),
    }


# ---------------------------------------------------------------------------
# Preflight and extraction helpers
# ---------------------------------------------------------------------------


def preflight_solver(
    solver: SolverFIT3D,
    n_steps: int,
    mem_limit_mb: float = DEFAULT_MEM_LIMIT_MB,
) -> dict[str, float]:
    """Run ``wakis.estimate`` and ``preflight`` on a 3D solver before time-stepping.

    Parameters
    ----------
    solver : SolverFIT3D
        Initialized 3D solver.
    n_steps : int
        Number of timesteps to execute.
    mem_limit_mb : float, optional
        Memory ceiling [MB].

    Returns
    -------
    est : dict
        Resource estimate dictionary with ``memory_mb`` and ``seconds``.
    """
    est = estimate(
        (int(solver.Nx), int(solver.Ny), int(solver.Nz)),
        steps=int(n_steps),
        cpml=bool(getattr(solver, "activate_cpml", False)),
    )
    preflight(est, {"ram": float(mem_limit_mb) * 1e6 / 0.85, "gpu": 0}, raise_error=True)
    return {
        "memory_mb": float(est.host_bytes / 1e6),
        "seconds": float(est.seconds),
        "cells": int(est.cells),
    }


def lateral_box_flux(box_monitor: FluxMonitor) -> np.ndarray:
    """Return the outward spectral flux [J/Hz^2] across the 4 lateral faces of a box FluxMonitor.

    A 3D ``FluxMonitor`` with non-zero ``size`` builds 6 faces in axis order
    ``(-x, +x, -y, +y, -z, +z)`` with 2 polarization terms per face. The first
    8 terms correspond to the 4 lateral faces ``x = +/- r_box`` and ``y = +/- r_box``,
    with trapezoidal 0.5 edge weights already applied on shared box edges.
    """
    flux = np.zeros(box_monitor.frequencies.size, dtype=float)
    for term in box_monitor._terms[:8]:
        density = box_monitor.xp.real(term["E_acc"] * box_monitor.xp.conj(term["H_acc"]))
        density = density.get() if hasattr(density, "get") else density
        flux += density @ term["weight"]
    return flux


def measure_cone_from_ez_slice(
    x_arr: np.ndarray,
    z_arr: np.ndarray,
    ez_xz: np.ndarray,
    z_bunch: float,
    sigmaz_m: float = DEFAULT_SIGMAZ_M,
    r_min_m: float = 2.8e-3,
    r_max_m: float = 7.0e-3,
    beta: float = 0.90,
    f_max_hz: float | None = None,
) -> dict[str, float | bool | np.ndarray]:
    """Detect a Cherenkov shock cone and fit the wavevector angle theta_c from an Ez(x, z) slice.

    The Cherenkov shock surface trails behind the bunch at
    ``z_bunch - z_shock(r) = r * tan(theta_c)``, where ``theta_c = arccos(1/(beta*n))``
    is the wavevector angle and ``90 deg - theta_c`` is the Mach cone half-angle.
    Below threshold (``beta * n < 1``), the bound Coulomb field is antisymmetric
    in ``z - z_bunch`` (equal amplitude ahead of and behind the bunch), so the
    backward-to-forward off-axis peak ratio is ~ 1. Above threshold, the causal
    Cherenkov wake lies strictly behind the bunch, yielding a large backward-to-forward
    ratio and a linear shock ridge ``z_bunch - z_ridge(r) = m * r + c_0`` with
    ``m = tan(theta_c)``.

    Parameters
    ----------
    x_arr : ndarray
        Transverse coordinates [m], shape ``(Nx,)``.
    z_arr : ndarray
        Longitudinal coordinates [m], shape ``(Nz,)``.
    ez_xz : ndarray
        Longitudinal electric field slice ``E_z(x, y=0, z)`` [V/m], shape ``(Nx, Nz)``.
    z_bunch : float
        Instantaneous axial position of the bunch centroid [m].
    sigmaz_m : float, optional
        RMS bunch length [m].
    r_min_m, r_max_m : float, optional
        Transverse radial fitting window [m].
    beta : float, optional
        Normalized bunch velocity v / c [-] used to map ``f_max_hz`` to longitudinal
        wavenumber ``k_z = 2*pi*f / (beta*c)``.
    f_max_hz : float or None, optional
        Upper bound of the analysed frequency band [Hz] (default: 30 GHz, where the
        finest grid achieves >= 20 cells/wavelength in n = 1.5). When provided, the
        ridge extraction is band-limited along z to ``f <= f_max_hz``.

    Returns
    -------
    out : dict
        Dictionary with ``cone_detected`` (bool), ``theta_c_deg``, ``cos_theta_c``,
        ``shock_half_angle_deg``, ``slope``, ``r2``, ``back_to_fwd_ratio``,
        ``r_fit``, and ``z_ridge``.
    """
    r_abs = np.abs(x_arr)
    r_mask = (r_abs >= float(r_min_m)) & (r_abs <= float(r_max_m))
    back_z_mask = (z_arr <= float(z_bunch) - 1.2 * float(sigmaz_m)) & (
        z_arr >= z_arr[0] + 3.0e-3
    )
    fwd_z_mask = (z_arr >= float(z_bunch) + 1.5 * float(sigmaz_m)) & (
        z_arr <= z_arr[-1] - 3.0e-3
    )

    peak_back = float(np.max(np.abs(ez_xz[np.ix_(r_mask, back_z_mask)]))) + 1e-30
    peak_fwd = float(np.max(np.abs(ez_xz[np.ix_(r_mask, fwd_z_mask)]))) + 1e-30
    back_to_fwd_ratio = peak_back / peak_fwd

    # Band-limit ridge extraction along z to the analysed band f <= f_max_hz where >= 20 cells/lambda holds
    if f_max_hz is not None and f_max_hz > 0.0 and z_arr.size > 4:
        dz_grid = float(z_arr[1] - z_arr[0])
        kz_fft = 2.0 * np.pi * rfftfreq(z_arr.size, d=dz_grid)
        kz_cut = 2.0 * np.pi * float(f_max_hz) / (max(float(beta), 0.1) * c_light)
        filt_k = np.exp(-0.5 * (kz_fft / kz_cut) ** 6)
        ez_ridge_src = irfft(rfft(ez_xz, axis=1) * filt_k[None, :], n=z_arr.size, axis=1)
    else:
        ez_ridge_src = ez_xz

    # Extract the leading negative Cherenkov shock ridge (for q > 0 moving in +z, Ez < 0 on the leading Mach front)
    ix0 = int(np.argmin(np.abs(x_arr)))
    pos_x_idx = np.where((x_arr >= float(r_min_m)) & (x_arr <= float(r_max_m)))[0]
    r_pts = x_arr[pos_x_idx]
    ridge_z_mask = (z_arr <= float(z_bunch) - 1.0 * float(sigmaz_m)) & (
        z_arr >= z_arr[0] + 3.0e-3
    )
    z_sub = z_arr[ridge_z_mask]

    z_ridge = np.zeros_like(r_pts)
    for idx_r, ix in enumerate(pos_x_idx):
        di = int(ix - ix0)
        ix_sym = max(0, min(ez_ridge_src.shape[0] - 1, ix0 - di))
        prof = 0.5 * (ez_ridge_src[ix, ridge_z_mask] + ez_ridge_src[ix_sym, ridge_z_mask])
        k_min = int(np.argmin(prof))
        if 0 < k_min < prof.size - 1:
            y_m1, y_0, y_p1 = prof[k_min - 1], prof[k_min], prof[k_min + 1]
            denom = y_m1 - 2.0 * y_0 + y_p1
            shift = 0.5 * (y_m1 - y_p1) / denom if abs(denom) > 1e-30 else 0.0
            shift = float(np.clip(shift, -0.5, 0.5))
            dz_loc = z_sub[1] - z_sub[0]
            z_ridge[idx_r] = z_sub[k_min] + shift * dz_loc
        else:
            z_ridge[idx_r] = z_sub[k_min]

    lag = float(z_bunch) - z_ridge
    slope, intercept = np.polyfit(r_pts, lag, 1)
    lag_fit = slope * r_pts + intercept
    ss_res = float(np.sum((lag - lag_fit) ** 2))
    ss_tot = float(np.sum((lag - np.mean(lag)) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot

    cone_detected = bool((back_to_fwd_ratio > 5.0) and (r2 > 0.96) and (slope > 0.15))
    if cone_detected:
        theta_c_rad = float(np.arctan(slope))
        theta_c_deg = float(np.degrees(theta_c_rad))
        cos_theta_c = float(1.0 / np.sqrt(1.0 + slope**2))
        shock_half_deg = 90.0 - theta_c_deg
    else:
        theta_c_deg = float("nan")
        cos_theta_c = float("nan")
        shock_half_deg = float("nan")

    return {
        "cone_detected": cone_detected,
        "theta_c_deg": theta_c_deg,
        "cos_theta_c": cos_theta_c,
        "shock_half_angle_deg": shock_half_deg,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "back_to_fwd_ratio": float(back_to_fwd_ratio),
        "r_fit": r_pts,
        "z_ridge": z_ridge,
    }


# ---------------------------------------------------------------------------
# Single-run 3D dielectric Cherenkov helper (used by tasks and tests)
# ---------------------------------------------------------------------------


def run_single_dielectric_case(
    beta: float = 0.90,
    n_refr: float = DEFAULT_N_DIEL,
    dx_m: float = 0.50e-3,
    dz_m: float = 0.45e-3,
    l_xy_m: float = 11.0e-3,
    zmin_m: float = 0.0,
    zmax_m: float = 42.0e-3,
    sigmaz_m: float = DEFAULT_SIGMAZ_M,
    q_c: float = DEFAULT_Q_C,
    z_snap_m: float = 32.0e-3,
    record_spectrum: bool = True,
    n_frames: int = 0,
    mem_limit_mb: float = DEFAULT_MEM_LIMIT_MB,
) -> dict:
    """Run a 3D FDTD/FIT simulation of a moving Gaussian charge in a dielectric with CPML.

    Parameters
    ----------
    beta : float, optional
        Normalized bunch velocity v / c [-].
    n_refr : float, optional
        Refractive index n = sqrt(eps_r) of the background medium [-].
    dx_m, dz_m : float, optional
        Transverse and longitudinal cell sizes [m].
    l_xy_m : float, optional
        Half-width of the square transverse domain [-l_xy_m, +l_xy_m] [m].
    zmin_m, zmax_m : float, optional
        Longitudinal domain bounds [m].
    sigmaz_m : float, optional
        RMS bunch length [m].
    q_c : float, optional
        Bunch charge [C].
    z_snap_m : float, optional
        Bunch centroid position z_b at which the 2D field snapshot is recorded [m].
    record_spectrum : bool, optional
        If True, continue running until the Cherenkov wave clears the coaxial
        ``FluxMonitor`` box and compute ``d^2 W / (d omega dz)``.
    n_frames : int, optional
        Number of meridian animation frames to record (0 disables frame recording).
    mem_limit_mb : float, optional
        Memory ceiling for ``wakis.estimate.preflight`` [MB].

    Returns
    -------
    res : dict
        Simulation results including slices, cone measurement, and spectrum.
    """
    nx = int(round(2.0 * l_xy_m / dx_m))
    if nx % 2 == 1:
        nx += 1
    ny = nx
    l_xy_actual = 0.5 * nx * float(dx_m)
    nz = int(round((zmax_m - zmin_m) / dz_m))
    zmax_actual = float(zmin_m) + nz * float(dz_m)
    eps_r = float(n_refr) ** 2

    # Below threshold (beta * n < 1), the wave speed c/n exceeds the bunch speed v = beta*c by
    # only (1/(beta*n) - 1)*v. Pad the simulation z-domain upstream and downstream so the causal
    # entrance turn-on wavefront completely outruns the forward (zeta > 0) Coulomb lobe before
    # z_snap_m and the downstream CPML does not overlap the forward tail, then slice back to [zmin_m, zmax_actual].
    if float(beta) * float(n_refr) < 1.0:
        n_pad_low = int(round(22.0e-3 / dz_m))
        n_pad_high = int(round(4.0e-3 / dz_m))
    else:
        n_pad_low = 0
        n_pad_high = 0
    zmin_sim = float(zmin_m) - n_pad_low * float(dz_m)
    zmax_sim = zmax_actual + n_pad_high * float(dz_m)
    nz_sim = nz + n_pad_low + n_pad_high
    z_slice = slice(n_pad_low, n_pad_low + nz)

    grid = GridFIT3D(
        -l_xy_actual,
        l_xy_actual,
        -l_xy_actual,
        l_xy_actual,
        zmin_sim,
        zmax_sim,
        nx,
        ny,
        nz_sim,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "cpml"],
        bc_high=["cpml", "cpml", "cpml"],
        bg=[eps_r, 1.0],
        verbose=0,
    )
    # Direct current injection allows arbitrary sub-luminal and relativistic beta in (0, 1]
    solver.source_type = "direct"

    v = float(beta) * c_light
    ti = 4.5 * float(sigmaz_m) / v
    beam = Beam(q=float(q_c), sigmaz=float(sigmaz_m), beta=float(beta), xsource=0.0, ysource=0.0, ti=ti)
    z0_min = float(solver.z.min())
    t_snap = beam.ti + (float(z_snap_m) - z0_min) / v

    freqs = np.linspace(12.0e9, 80.0e9, 69)
    r_box = round(3.2e-3 / float(dx_m)) * float(dx_m)
    l_box = 2.0 * round(6.0e-3 / float(dz_m)) * float(dz_m)
    z_mid = float(zmin_m) + (nz // 2) * float(dz_m)
    if record_spectrum:
        box_mon = FluxMonitor(freqs, center=(0.0, 0.0, z_mid), size=(2.0 * r_box, 2.0 * r_box, l_box))
        t_end = beam.ti + (zmax_sim - z0_min + 4.5 * float(sigmaz_m) + 2.0 * r_box * float(n_refr)) / v
    else:
        box_mon = None
        t_end = t_snap + solver.dt

    nt = int(np.ceil(t_end / solver.dt))
    est = preflight_solver(solver, n_steps=nt, mem_limit_mb=mem_limit_mb)

    n_snap_step = int(round(t_snap / solver.dt))
    x_arr = solver.grid.x[: solver.Nx].copy()
    y_arr = solver.grid.y[: solver.Ny].copy()
    iy0 = int(np.argmin(np.abs(y_arr)))
    ix0 = int(np.argmin(np.abs(x_arr)))
    z_arr = solver.z[z_slice].copy()

    # Optional animation frame steps while the bunch crosses [zmin_m + 3 mm, zmax_m - 3 mm]
    if n_frames > 0:
        t_anim_start = beam.ti + (zmin_m + 3.0e-3 - z0_min) / v
        t_anim_stop = beam.ti + (zmax_m - 3.5e-3 - z0_min) / v
        frame_steps = set(
            np.unique(
                np.clip(
                    np.round(np.linspace(t_anim_start, t_anim_stop, n_frames) / solver.dt).astype(int),
                    0,
                    nt - 1,
                )
            ).tolist()
        )
    else:
        frame_steps = set()

    frames_ez = []
    frames_zb = []
    ez_snap = None
    emag_snap = None
    zb_snap_actual = float(z_snap_m)

    t0_wall = time.perf_counter()
    for step in range(nt):
        t = step * solver.dt
        beam.update(solver, t)
        solver.one_step()
        if box_mon is not None:
            box_mon.update(solver, t)

        zb_now = z0_min + v * (t + 0.5 * solver.dt - beam.ti)
        if step == n_snap_step or (ez_snap is None and step == nt - 1):
            ez_snap = solver.E[:, iy0, z_slice, "z"].copy()
            ex_snap = solver.E[:, iy0, z_slice, "x"].copy()
            ey_snap = solver.E[:, iy0, z_slice, "y"].copy()
            emag_snap = np.sqrt(ex_snap**2 + ey_snap**2 + ez_snap**2)
            zb_snap_actual = float(zb_now)

        if step in frame_steps:
            frames_ez.append(solver.E[:, iy0, z_slice, "z"].copy())
            frames_zb.append(float(zb_now))

    wall_s = time.perf_counter() - t0_wall
    cone_info = measure_cone_from_ez_slice(
        x_arr=x_arr,
        z_arr=z_arr,
        ez_xz=ez_snap,
        z_bunch=zb_snap_actual,
        sigmaz_m=sigmaz_m,
        beta=beta,
    )
    th_cone = cherenkov_cone_angle(beta, n_refr)

    out = {
        "beta": float(beta),
        "n_refr": float(n_refr),
        "sigmaz_m": float(sigmaz_m),
        "q_c": float(q_c),
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "dx_m": float(solver.grid.dx[0]),
        "dz_m": float(solver.grid.dz[0]),
        "dt_s": float(solver.dt),
        "nt": int(nt),
        "wall_s": float(wall_s),
        "memory_mb": float(est["memory_mb"]),
        "x_m": x_arr,
        "z_m": z_arr,
        "z_bunch_snap_m": zb_snap_actual,
        "Ez_xz": ez_snap,
        "Emag_xz": emag_snap,
        "Ez_axis": ez_snap[ix0, :].copy(),
        "cone": cone_info,
        "theory_cone": th_cone,
    }

    if n_frames > 0:
        out["frames_Ez"] = np.asarray(frames_ez)
        out["frames_zb_m"] = np.asarray(frames_zb)

    if box_mon is not None:
        lat_flux = lateral_box_flux(box_mon)
        d2w_sim = np.maximum(lat_flux, 0.0) / (np.pi * l_box)
        ft = frank_tamm_spectrum(
            freqs_hz=freqs,
            beta=beta,
            n_complex=n_refr,
            q_c=q_c,
            sigmaz_m=sigmaz_m,
        )
        # Integrated radiated energy per unit length over the analysed band [12 GHz, 30 GHz]
        # where the finest 3D resolution meets >= 20 cells per wavelength in n = 1.5
        band_mask = (freqs >= ANALYSED_FMIN_HZ) & (freqs <= ANALYSED_FMAX_HZ)
        dw_dz_sim = float(np.trapezoid(d2w_sim[band_mask], ft["omega"][band_mask]))
        dw_dz_th = float(np.trapezoid(ft["d2W_domega_dz_bunch"][band_mask], ft["omega"][band_mask]))
        out.update(
            {
                "freqs_hz": freqs,
                "omega_rad_s": ft["omega"],
                "d2W_domega_dz_sim": d2w_sim,
                "d2W_domega_dz_theory": ft["d2W_domega_dz_bunch"],
                "d2W_domega_dz_point": ft["d2W_domega_dz_point"],
                "form_factor": ft["form_factor"],
                "dW_dz_sim_J_m": dw_dz_sim,
                "dW_dz_theory_J_m": dw_dz_th,
                "r_box_m": float(r_box),
                "l_box_m": float(l_box),
                "z_mid_box_m": float(z_mid),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Task 1: Dielectric (n = 1.5) above & below threshold + beta sweep
# ---------------------------------------------------------------------------


def run_task_dielectric(quick: bool = False) -> dict:
    """Run Task 1: Dielectric (n = 1.5) at beta = 0.90 and 0.60 plus beta sweep."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dx = 0.55e-3 if quick else 0.45e-3
    dz = dx
    n_frames = 18 if quick else 26

    sup = run_single_dielectric_case(
        beta=0.90,
        n_refr=DEFAULT_N_DIEL,
        dx_m=dx,
        dz_m=dz,
        record_spectrum=True,
        n_frames=n_frames,
    )
    sub = run_single_dielectric_case(
        beta=0.60,
        n_refr=DEFAULT_N_DIEL,
        dx_m=dx,
        dz_m=dz,
        record_spectrum=True,
        n_frames=n_frames,
    )

    # Beta sweep for cone angle vs 1/(beta*n)
    beta_sweep = np.array([0.60, 0.72, 0.80, 0.90, 0.96]) if quick else np.array([0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.96])
    dx_sw = 0.55e-3 if quick else 0.45e-3
    dz_sw = dx_sw

    theta_sim_deg = []
    cos_sim = []
    theta_th_deg = []
    cos_th = []
    detected_flags = []
    back_fwd_ratios = []

    for b_val in beta_sweep:
        if abs(b_val - 0.90) < 1e-9 and abs(dx_sw - dx) < 1e-9:
            r_b = sup
        elif abs(b_val - 0.60) < 1e-9 and abs(dx_sw - dx) < 1e-9:
            r_b = sub
        else:
            r_b = run_single_dielectric_case(
                beta=float(b_val),
                n_refr=DEFAULT_N_DIEL,
                dx_m=dx_sw,
                dz_m=dz_sw,
                record_spectrum=False,
                n_frames=0,
            )
        c_info = r_b["cone"]
        t_info = r_b["theory_cone"]
        detected_flags.append(bool(c_info["cone_detected"]))
        theta_sim_deg.append(float(c_info["theta_c_deg"]))
        cos_sim.append(float(c_info["cos_theta_c"]))
        theta_th_deg.append(float(t_info["theta_c_deg"]))
        cos_th.append(float(t_info["cos_theta_c"]))
        back_fwd_ratios.append(float(c_info["back_to_fwd_ratio"]))

    # Off-axis radial Coulomb slice at beta = 0.60 (r = 3.5 mm) vs analytical sub-threshold formula
    r_probe_target = 3.5e-3
    ix_probe = int(np.argmin(np.abs(sub["x_m"] - r_probe_target)))
    r_probe_actual = float(abs(sub["x_m"][ix_probe]))
    zeta_sub = sub["z_m"] - sub["z_bunch_snap_m"]
    coul_th = subthreshold_coulomb_field(
        r_m=r_probe_actual,
        zeta_m=zeta_sub,
        beta=0.60,
        n_refr=DEFAULT_N_DIEL,
        q_c=DEFAULT_Q_C,
        sigmaz_m=DEFAULT_SIGMAZ_M,
    )

    npz_dict = {
        "x_m": sup["x_m"],
        "z_m": sup["z_m"],
        "z_bunch_sup_m": np.array(sup["z_bunch_snap_m"]),
        "z_bunch_sub_m": np.array(sub["z_bunch_snap_m"]),
        "Ez_sup_xz": sup["Ez_xz"],
        "Emag_sup_xz": sup["Emag_xz"],
        "Ez_sub_xz": sub["Ez_xz"],
        "Emag_sub_xz": sub["Emag_xz"],
        "r_fit_sup_m": sup["cone"]["r_fit"],
        "z_ridge_sup_m": sup["cone"]["z_ridge"],
        "theta_c_sup_deg": np.array(sup["cone"]["theta_c_deg"]),
        "theta_c_sup_theory_deg": np.array(sup["theory_cone"]["theta_c_deg"]),
        "cos_theta_c_sup": np.array(sup["cone"]["cos_theta_c"]),
        "cos_theta_c_sup_theory": np.array(sup["theory_cone"]["cos_theta_c"]),
        "freqs_hz": sup["freqs_hz"],
        "omega_rad_s": sup["omega_rad_s"],
        "d2W_sup_sim": sup["d2W_domega_dz_sim"],
        "d2W_sup_theory": sup["d2W_domega_dz_theory"],
        "d2W_sup_point": sup["d2W_domega_dz_point"],
        "form_factor_sup": sup["form_factor"],
        "d2W_sub_sim": sub["d2W_domega_dz_sim"],
        "d2W_sub_theory": sub["d2W_domega_dz_theory"],
        "beta_sweep": beta_sweep,
        "theta_sweep_sim_deg": np.asarray(theta_sim_deg),
        "theta_sweep_theory_deg": np.asarray(theta_th_deg),
        "cos_sweep_sim": np.asarray(cos_sim),
        "cos_sweep_theory": np.asarray(cos_th),
        "cone_detected_sweep": np.asarray(detected_flags),
        "back_to_fwd_sweep": np.asarray(back_fwd_ratios),
        "r_probe_sub_m": np.array(r_probe_actual),
        "zeta_sub_m": zeta_sub,
        "Ez_sub_probe_sim": sub["Ez_xz"][ix_probe, :],
        "Ez_sub_probe_theory": coul_th["Ez"],
        "frames_Ez_sup": sup["frames_Ez"],
        "frames_zb_sup_m": sup["frames_zb_m"],
        "frames_Ez_sub": sub["frames_Ez"],
        "frames_zb_sub_m": sub["frames_zb_m"],
    }
    np.savez_compressed(DATA_DIR / "task_dielectric.npz", **npz_dict)

    peak_idx = int(np.argmax(sup["d2W_domega_dz_theory"]))
    spec_rel_err_peak = float(
        abs(sup["d2W_domega_dz_sim"][peak_idx] - sup["d2W_domega_dz_theory"][peak_idx])
        / sup["d2W_domega_dz_theory"][peak_idx]
    )
    ez_sub_sim = sub["Ez_xz"][ix_probe, :]
    ez_sub_th = coul_th["Ez"]
    mask_probe = np.abs(zeta_sub) <= 10.0e-3
    coul_corr = float(
        np.dot(ez_sub_sim[mask_probe], ez_sub_th[mask_probe])
        / (np.linalg.norm(ez_sub_sim[mask_probe]) * np.linalg.norm(ez_sub_th[mask_probe]))
    )
    coul_l2_rel_err = float(
        np.linalg.norm(ez_sub_sim[mask_probe] - ez_sub_th[mask_probe])
        / np.linalg.norm(ez_sub_th[mask_probe])
    )
    coul_peak_rel_err = float(
        abs(np.max(np.abs(ez_sub_sim[mask_probe])) - np.max(np.abs(ez_sub_th[mask_probe])))
        / np.max(np.abs(ez_sub_th[mask_probe]))
    )
    idx_sim_min = int(np.argmin(ez_sub_sim[mask_probe]))
    idx_sim_max = int(np.argmax(ez_sub_sim[mask_probe]))
    idx_th_min = int(np.argmin(ez_sub_th[mask_probe]))
    idx_th_max = int(np.argmax(ez_sub_th[mask_probe]))
    zeta_probe_win = zeta_sub[mask_probe]

    summary = {
        "task": "dielectric",
        "quick": bool(quick),
        "n_refr": float(DEFAULT_N_DIEL),
        "analysed_band_ghz": [
            round(float(ANALYSED_FMIN_HZ * 1e-9), 6),
            round(float(ANALYSED_FMAX_HZ * 1e-9), 6),
        ],
        "beta_threshold": float(1.0 / DEFAULT_N_DIEL),
        "beta_super": 0.90,
        "beta_sub": 0.60,
        "theta_c_sim_deg": float(sup["cone"]["theta_c_deg"]),
        "theta_c_theory_deg": float(sup["theory_cone"]["theta_c_deg"]),
        "shock_half_angle_sim_deg": float(sup["cone"]["shock_half_angle_deg"]),
        "shock_half_angle_theory_deg": float(sup["theory_cone"]["shock_half_angle_deg"]),
        "cos_theta_c_sim": float(sup["cone"]["cos_theta_c"]),
        "cos_theta_c_theory": float(sup["theory_cone"]["cos_theta_c"]),
        "cos_theta_c_rel_err": float(
            abs(sup["cone"]["cos_theta_c"] - sup["theory_cone"]["cos_theta_c"])
            / sup["theory_cone"]["cos_theta_c"]
        ),
        "dW_dz_super_sim_J_m": float(sup["dW_dz_sim_J_m"]),
        "dW_dz_super_theory_J_m": float(sup["dW_dz_theory_J_m"]),
        "dW_dz_sub_sim_J_m": float(sub["dW_dz_sim_J_m"]),
        "sub_to_super_power_ratio": float(sub["dW_dz_sim_J_m"] / max(sup["dW_dz_sim_J_m"], 1e-30)),
        "spectrum_peak_rel_err": spec_rel_err_peak,
        "cone_detected_super": bool(sup["cone"]["cone_detected"]),
        "cone_detected_sub": bool(sub["cone"]["cone_detected"]),
        "r_probe_sub_mm": float(r_probe_actual * 1e3),
        "coulomb_probe_corr": coul_corr,
        "coulomb_probe_l2_rel_err": coul_l2_rel_err,
        "coulomb_probe_peak_rel_err": coul_peak_rel_err,
        "ez_sub_probe_min_kv_m": float(ez_sub_sim[mask_probe][idx_sim_min] * 1e-3),
        "ez_sub_probe_max_kv_m": float(ez_sub_sim[mask_probe][idx_sim_max] * 1e-3),
        "zeta_sub_sim_min_mm": float(zeta_probe_win[idx_sim_min] * 1e3),
        "zeta_sub_sim_max_mm": float(zeta_probe_win[idx_sim_max] * 1e3),
        "ez_sub_theory_min_kv_m": float(ez_sub_th[mask_probe][idx_th_min] * 1e-3),
        "ez_sub_theory_max_kv_m": float(ez_sub_th[mask_probe][idx_th_max] * 1e-3),
        "zeta_sub_theory_min_mm": float(zeta_probe_win[idx_th_min] * 1e3),
        "zeta_sub_theory_max_mm": float(zeta_probe_win[idx_th_max] * 1e3),
    }
    (DATA_DIR / "task_dielectric.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Task 2: Dispersive Lorentz medium (Cherenkov passband vs stopband edge)
# ---------------------------------------------------------------------------


def run_task_dispersive(quick: bool = False) -> dict:
    """Run Task 2: Cherenkov radiation in a Lorentz dispersive medium with a band edge."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    f0_hz = 38.0e9
    w0 = 2.0 * np.pi * f0_hz
    gamma = 2.0 * np.pi * 1.5e9
    eps_inf = 1.0
    delta_eps = 1.25  # static eps_s = 2.25 -> n(0) = 1.5
    beta = 0.85       # threshold 1/beta = 1.1765
    sigmaz_m = DEFAULT_SIGMAZ_M
    q_c = DEFAULT_Q_C

    l_xy_m = 11.0e-3
    zmin_m, zmax_m = -8.0e-3, 18.0e-3
    dx_m = 0.55e-3 if quick else 0.44e-3
    dz_m = 0.44e-3 if quick else 0.35e-3

    nx = int(round(2.0 * l_xy_m / dx_m))
    if nx % 2 == 1:
        nx += 1
    ny = nx
    nz = int(round((zmax_m - zmin_m) / dz_m))

    grid = GridFIT3D(-l_xy_m, l_xy_m, -l_xy_m, l_xy_m, zmin_m, zmax_m, nx, ny, nz, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "cpml"],
        bc_high=["cpml", "cpml", "cpml"],
        bg=[1.0, 1.0],
        verbose=0,
    )
    solver.source_type = "direct"

    mat = DispersiveMaterial(
        eps_inf=eps_inf,
        poles=[Lorentz(delta_eps=delta_eps, omega_0=w0, gamma=gamma)],
    )
    solver.add_dispersive_material(
        mat,
        center=(0.0, 0.0, 0.5 * (zmin_m + zmax_m)),
        size=(2.0 * l_xy_m, 2.0 * l_xy_m, zmax_m - zmin_m),
    )

    freqs = np.linspace(5.0e9, 70.0e9, 66)
    r_box = 4.5e-3
    l_box = 8.0e-3
    z_mid = 5.0e-3
    box_mon = FluxMonitor(freqs, center=(0.0, 0.0, z_mid), size=(2.0 * r_box, 2.0 * r_box, l_box))

    v = beta * c_light
    beam = Beam(q=q_c, sigmaz=sigmaz_m, beta=beta, xsource=0.0, ysource=0.0)
    z0_min = float(solver.z.min())
    z_snap_m = 9.0e-3
    t_snap = beam.ti + (z_snap_m - z0_min) / v
    t_end = beam.ti + (zmax_m - z0_min + 6.5 * sigmaz_m) / v
    nt = int(np.ceil(t_end / solver.dt))
    n_snap = int(round(t_snap / solver.dt))

    preflight_solver(solver, n_steps=nt)

    iy0 = ny // 2
    ez_snap = None
    zb_snap = z_snap_m
    for step in range(nt):
        t = step * solver.dt
        beam.update(solver, t)
        solver.one_step()
        box_mon.update(solver, t)
        if step == n_snap:
            ez_snap = solver.E[:, iy0, :, "z"].copy()
            zb_snap = float(z0_min + v * (t + 0.5 * solver.dt - beam.ti))

    lat_flux = lateral_box_flux(box_mon)
    d2w_sim = np.maximum(lat_flux, 0.0) / (np.pi * l_box)

    eps_w = mat.epsilon(f=freqs)
    n_complex = np.sqrt(eps_w)
    ft = frank_tamm_spectrum(
        freqs_hz=freqs,
        beta=beta,
        n_complex=n_complex,
        q_c=q_c,
        sigmaz_m=sigmaz_m,
        r_obs_m=r_box,
    )

    pass_mask = freqs <= 32.0e9
    stop_mask = freqs >= 44.0e9
    pass_energy = float(np.trapezoid(d2w_sim[pass_mask], ft["omega"][pass_mask]))
    stop_energy = float(np.trapezoid(d2w_sim[stop_mask], ft["omega"][stop_mask]))
    stop_to_pass_ratio = stop_energy / max(pass_energy, 1e-30)

    np.savez_compressed(
        DATA_DIR / "task_dispersive.npz",
        freqs_hz=freqs,
        omega_rad_s=ft["omega"],
        eps_real=np.real(eps_w),
        eps_imag=np.imag(eps_w),
        n_real=np.real(n_complex),
        n_imag=np.abs(np.imag(n_complex)),
        above_threshold=ft["above_threshold"],
        d2W_sim=d2w_sim,
        d2W_theory_unattenuated=ft["d2W_domega_dz_bunch"],
        d2W_theory_attenuated=ft["d2W_domega_dz_attenuated"],
        x_m=solver.grid.x[: solver.Nx].copy(),
        z_m=solver.z.copy(),
        z_bunch_snap_m=np.array(zb_snap),
        Ez_xz=ez_snap,
        beta=np.array(beta),
        f0_hz=np.array(f0_hz),
    )

    summary = {
        "task": "dispersive",
        "quick": bool(quick),
        "beta": float(beta),
        "n_threshold": float(1.0 / beta),
        "f0_ghz": float(f0_hz * 1e-9),
        "gamma_ghz": float(gamma / (2.0 * np.pi) * 1e-9),
        "eps_inf": float(eps_inf),
        "delta_eps": float(delta_eps),
        "passband_energy_J_m": pass_energy,
        "stopband_energy_J_m": stop_energy,
        "stop_to_pass_ratio": float(stop_to_pass_ratio),
    }
    (DATA_DIR / "task_dispersive.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Task 3: Vacuum (beta = 0.999) Yee vs TETE numerical Cherenkov & resolution
# ---------------------------------------------------------------------------


def run_single_vacuum_scheme(
    scheme: str = "yee",
    beta: float = 0.999,
    dz_m: float = 0.80e-3,
    a_pipe_m: float = 16.0e-3,
    zmin_m: float = -24.0e-3,
    zmax_m: float = 36.0e-3,
    sigmaz_m: float = 1.8e-3,
    q_c: float = DEFAULT_Q_C,
    z0_m: float = -8.0e-3,
    z_snap_m: float = 20.0e-3,
    n_frames: int = 0,
    mem_limit_mb: float = DEFAULT_MEM_LIMIT_MB,
) -> dict:
    """Run a vacuum drift-pipe simulation at beta = 0.999 with Yee or TETE.

    The bunch is initialized at ``z0_m`` with its exact discrete Lorentz-contracted
    comoving self-field inside the smooth PEC pipe (``dx = dy = 2 * dz`` so the
    magic step ``c * dt = dz`` of ``scheme='tete'`` is stable), with CPML on
    both z ends.
    """
    dx_target = 2.0 * float(dz_m)
    ncells_x = int(round(float(a_pipe_m) / dx_target))
    if ncells_x % 2 == 1:
        ncells_x += 1
    dx_actual = float(a_pipe_m) / ncells_x
    dz_actual = 0.5 * dx_actual
    ncells_z = int(round((float(zmax_m) - float(zmin_m)) / dz_actual))

    grid = GridFIT3D(
        -0.5 * a_pipe_m,
        0.5 * a_pipe_m,
        -0.5 * a_pipe_m,
        0.5 * a_pipe_m,
        zmin_m,
        zmin_m + ncells_z * dz_actual,
        ncells_x,
        ncells_x,
        ncells_z,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["pec", "pec", "cpml"],
        bc_high=["pec", "pec", "cpml"],
        n_pml=8,
        bg=[1.0, 1.0],
        scheme=scheme,
        verbose=0,
    )
    solver.source_type = "direct"

    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    v = float(beta) * c_light
    gamma_rel = 1.0 / np.sqrt(1.0 - float(beta) ** 2)
    z0_min = float(solver.z.min())
    beam = Beam(
        q=float(q_c),
        sigmaz=float(sigmaz_m),
        beta=float(beta),
        xsource=0.0,
        ysource=0.0,
        ti=(z0_min - float(z0_m)) / v,
    )
    beam.update(solver, 0.0)
    ixs, iys = beam.ixs, beam.iys

    # Discrete 2D DST-I + 1D FFT solver for the initial comoving Lorentz-contracted self-field
    n_int = nx - 2
    m_idx = np.arange(1, n_int + 1)
    kx_d = (2.0 / dx_actual) * np.sin(0.5 * np.pi * m_idx / (nx - 1))
    kx2, ky2 = np.meshgrid(kx_d**2, kx_d**2, indexing="ij")
    kperp2 = kx2 + ky2

    delta_xy = np.zeros((n_int, n_int), dtype=float)
    delta_xy[ixs - 1, iys - 1] = 1.0 / (solver.tdx[ixs] * solver.tdy[iys])
    delta_hat = dstn(delta_xy, type=1)

    kz = 2.0 * np.pi * rfftfreq(nz, d=dz_actual)
    kz_d = (2.0 / dz_actual) * np.sin(0.5 * kz * dz_actual)

    t_eperp = -0.5 * solver.dt if scheme == "tete" else 0.0
    t_ez = 0.0
    t_hperp = 0.0 if scheme == "tete" else -0.5 * solver.dt

    def _phi_3d(z_nodes: np.ndarray, t_val: float) -> np.ndarray:
        prof = np.exp(-((z_nodes - (z0_m + v * t_val)) ** 2) / (2.0 * sigmaz_m**2)) / np.sqrt(
            2.0 * np.pi * sigmaz_m**2
        )
        prof_k = rfft(prof)
        phi_hat = (
            (q_c / epsilon_0)
            * delta_hat[:, :, None]
            * prof_k[None, None, :]
            / (kperp2[:, :, None] + (kz_d[None, None, :] / gamma_rel) ** 2)
        )
        phi_int = irfft(idstn(phi_hat, type=1, axes=(0, 1)), n=nz, axis=2)
        phi_full = np.zeros((nx, ny, nz), dtype=float)
        phi_full[1:-1, 1:-1, :] = phi_int
        return phi_full

    phi_ep = _phi_3d(solver.grid.z[:nz], t_eperp)
    phi_hp = _phi_3d(solver.z, t_hperp)
    phi_ez_lo = _phi_3d(solver.grid.z[:nz], t_ez)
    phi_ez_hi = _phi_3d(solver.grid.z[1 : nz + 1], t_ez)

    ex0 = np.zeros((nx, ny, nz), dtype=float)
    ey0 = np.zeros((nx, ny, nz), dtype=float)
    ex0[:-1, :, :] = -(phi_ep[1:, :, :] - phi_ep[:-1, :, :]) / dx_actual
    ey0[:, :-1, :] = -(phi_ep[:, 1:, :] - phi_ep[:, :-1, :]) / dx_actual
    ez0 = -(phi_ez_hi - phi_ez_lo) / (dz_actual * gamma_rel**2)

    hx0 = np.zeros((nx, ny, nz), dtype=float)
    hy0 = np.zeros((nx, ny, nz), dtype=float)
    ex_h = -(phi_hp[1:, :, :] - phi_hp[:-1, :, :]) / dx_actual
    ey_h = -(phi_hp[:, 1:, :] - phi_hp[:, :-1, :]) / dx_actual
    hy0[:-1, :, :] = (beta / Z0_VAC) * ex_h
    hx0[:, :-1, :] = -(beta / Z0_VAC) * ey_h

    t_snap = (float(z_snap_m) - float(z0_m)) / v
    nt = int(round(t_snap / solver.dt))
    preflight_solver(solver, n_steps=nt, mem_limit_mb=mem_limit_mb)

    solver.one_step()
    solver.E[:, :, :, "x"] = ex0
    solver.E[:, :, :, "y"] = ey0
    solver.E[:, :, :, "z"] = ez0
    solver.H[:, :, :, "x"] = hx0
    solver.H[:, :, :, "y"] = hy0
    solver.H[:, :, :, "z"] = 0.0
    solver.J.fromarray(np.zeros_like(solver.J.array))
    beam.Jold = np.zeros_like(beam.Jold)

    if n_frames > 0:
        frame_steps = set(
            np.unique(np.round(np.linspace(0, nt - 1, n_frames)).astype(int)).tolist()
        )
    else:
        frame_steps = set()

    iy0 = ny // 2
    ix0 = nx // 2
    frames_ez = []
    frames_zb = []
    for step in range(nt):
        t = step * solver.dt
        beam.update(solver, t)
        solver.one_step()
        if step in frame_steps:
            frames_ez.append(solver.E[:, iy0, :, "z"].copy())
            frames_zb.append(float(z0_m + v * (t + solver.dt)))

    zb_final = float(z0_m + v * nt * solver.dt)
    z_arr = solver.z.copy()
    x_arr = solver.grid.x[:nx].copy()
    ez_slice = solver.E[:, iy0, :, "z"].copy()

    # Trailing wake region behind the bunch: [-2 mm, zb_final - 4.5*sigmaz]
    wake_mask = (z_arr >= -2.0e-3) & (z_arr <= zb_final - 4.5 * float(sigmaz_m))
    ez_wake_3d = solver.E[1:-1, 1:-1, wake_mask, "z"]
    cell_vol = dx_actual * dx_actual * dz_actual
    w_wake_j = float(0.5 * epsilon_0 * np.sum(ez_wake_3d**2) * cell_vol)
    peak_ez_wake = float(np.max(np.abs(ez_wake_3d)))
    rms_ez_wake = float(np.sqrt(np.mean(ez_wake_3d**2)))

    out = {
        "scheme": scheme,
        "beta": float(beta),
        "dx_m": float(dx_actual),
        "dz_m": float(dz_actual),
        "dt_s": float(solver.dt),
        "sigmaz_over_dz": float(sigmaz_m / dz_actual),
        "ppw_char": float(2.0 * np.pi * sigmaz_m / dz_actual),
        "x_m": x_arr,
        "z_m": z_arr,
        "z_bunch_m": zb_final,
        "Ez_xz": ez_slice,
        "Ez_axis": ez_slice[ix0, :].copy(),
        "W_wake_J": w_wake_j,
        "peak_Ez_wake_V_m": peak_ez_wake,
        "rms_Ez_wake_V_m": rms_ez_wake,
    }
    if n_frames > 0:
        out["frames_Ez"] = np.asarray(frames_ez)
        out["frames_zb_m"] = np.asarray(frames_zb)
    return out


def run_task_numerical_cherenkov(quick: bool = False) -> dict:
    """Run Task 3: Vacuum (beta = 0.999) Yee vs TETE comparison and resolution scaling."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    beta = 0.999
    sigmaz_m = 1.8e-3
    dz_list = [1.0e-3, 0.8e-3, 0.571e-3] if quick else [1.0e-3, 0.8e-3, 0.571e-3, 0.4e-3]
    n_frames = 18 if quick else 26

    w_wake_yee = []
    w_wake_tete = []
    peak_yee = []
    peak_tete = []
    dz_actual_list = []
    sigma_over_dz_list = []
    ppw_list = []

    snap_yee = None
    snap_tete = None

    for idx, dz_val in enumerate(dz_list):
        rec_frames = n_frames if idx == 1 else 0
        r_yee = run_single_vacuum_scheme(
            scheme="yee",
            beta=beta,
            dz_m=dz_val,
            sigmaz_m=sigmaz_m,
            n_frames=rec_frames,
        )
        r_tete = run_single_vacuum_scheme(
            scheme="tete",
            beta=beta,
            dz_m=dz_val,
            sigmaz_m=sigmaz_m,
            n_frames=rec_frames,
        )
        dz_actual_list.append(r_yee["dz_m"])
        sigma_over_dz_list.append(r_yee["sigmaz_over_dz"])
        ppw_list.append(r_yee["ppw_char"])
        w_wake_yee.append(r_yee["W_wake_J"])
        w_wake_tete.append(r_tete["W_wake_J"])
        peak_yee.append(r_yee["peak_Ez_wake_V_m"])
        peak_tete.append(r_tete["peak_Ez_wake_V_m"])
        if idx == 1:
            snap_yee = r_yee
            snap_tete = r_tete

    # Analytical 1D longitudinal dispersion curves at the snapshot grid spacing
    dz_ref = snap_yee["dz_m"]
    dx_ref = snap_yee["dx_m"]
    kz_grid = np.linspace(1e-3 / dz_ref, np.pi / dz_ref, 300)
    disp = yee_vs_tete_phase_velocity_z(kz_grid, dx_ref, dx_ref, dz_ref)

    np.savez_compressed(
        DATA_DIR / "task_numerical_cherenkov.npz",
        beta=np.array(beta),
        sigmaz_m=np.array(sigmaz_m),
        dz_m=np.asarray(dz_actual_list),
        sigmaz_over_dz=np.asarray(sigma_over_dz_list),
        ppw_char=np.asarray(ppw_list),
        W_wake_yee_J=np.asarray(w_wake_yee),
        W_wake_tete_J=np.asarray(w_wake_tete),
        peak_Ez_yee_V_m=np.asarray(peak_yee),
        peak_Ez_tete_V_m=np.asarray(peak_tete),
        x_m=snap_yee["x_m"],
        z_m=snap_yee["z_m"],
        z_bunch_m=np.array(snap_yee["z_bunch_m"]),
        Ez_yee_xz=snap_yee["Ez_xz"],
        Ez_tete_xz=snap_tete["Ez_xz"],
        Ez_yee_axis=snap_yee["Ez_axis"],
        Ez_tete_axis=snap_tete["Ez_axis"],
        frames_Ez_yee=snap_yee["frames_Ez"],
        frames_Ez_tete=snap_tete["frames_Ez"],
        frames_zb_m=snap_yee["frames_zb_m"],
        kz_dz=kz_grid * dz_ref,
        vp_yee_over_c=disp["vp_yee_over_c"],
        vp_tete_over_c=disp["vp_tete_over_c"],
    )

    ratios = np.asarray(w_wake_yee) / np.maximum(np.asarray(w_wake_tete), 1e-30)
    summary = {
        "task": "numerical_cherenkov",
        "quick": bool(quick),
        "beta": float(beta),
        "dz_mm": [float(d * 1e3) for d in dz_actual_list],
        "sigmaz_over_dz": [float(s) for s in sigma_over_dz_list],
        "W_wake_yee_J": [float(w) for w in w_wake_yee],
        "W_wake_tete_J": [float(w) for w in w_wake_tete],
        "yee_to_tete_energy_ratio": [float(r) for r in ratios],
        "peak_Ez_yee_V_m": [float(p) for p in peak_yee],
        "peak_Ez_tete_V_m": [float(p) for p in peak_tete],
    }
    (DATA_DIR / "task_numerical_cherenkov.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Task 4: Resolution convergence study (cone angle and Frank-Tamm spectrum)
# ---------------------------------------------------------------------------


def run_task_convergence(quick: bool = False) -> dict:
    """Run Task 4: 3-level grid resolution study for Cherenkov cone angle and spectrum."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if quick:
        levels = [
            ("coarse", 0.60e-3, 0.60e-3),
            ("medium", 0.45e-3, 0.45e-3),
            ("fine", 0.33e-3, 0.33e-3),
        ]
    else:
        levels = [
            ("coarse", 0.56e-3, 0.56e-3),
            ("medium", 0.42e-3, 0.42e-3),
            ("fine", 0.32e-3, 0.32e-3),
        ]

    beta = 0.90
    n_refr = DEFAULT_N_DIEL
    # Shortest wavelength in the analysed band [12 GHz, 30 GHz] in n = 1.5 dielectric:
    # lambda_30GHz = c / (n * f_max) = 6.662 mm -> fine grid (0.32 mm) achieves 20.8 cells/lambda >= 20
    lambda_30ghz_m = c_light / (n_refr * ANALYSED_FMAX_HZ)
    lambda_50ghz_m = c_light / (n_refr * 50.0e9)

    names = []
    dx_list = []
    dz_list = []
    ppw_30ghz_list = []
    ppw_50ghz_list = []
    theta_deg_list = []
    cos_theta_list = []
    cos_rel_err_list = []
    spec_l2_rel_err_list = []
    dw_dz_rel_err_list = []
    wall_s_list = []
    mem_mb_list = []
    spectra_sim = []
    freqs_ref = None
    spec_theory = None

    for label, dx_val, dz_val in levels:
        res = run_single_dielectric_case(
            beta=beta,
            n_refr=n_refr,
            dx_m=dx_val,
            dz_m=dz_val,
            sigmaz_m=1.2e-3,
            record_spectrum=True,
            n_frames=0,
        )
        names.append(label)
        dx_list.append(res["dx_m"])
        dz_list.append(res["dz_m"])
        h_max = max(res["dx_m"], res["dz_m"])
        ppw_30ghz_list.append(lambda_30ghz_m / h_max)
        ppw_50ghz_list.append(lambda_50ghz_m / h_max)
        theta_deg_list.append(res["cone"]["theta_c_deg"])
        cos_theta_list.append(res["cone"]["cos_theta_c"])
        cos_th = res["theory_cone"]["cos_theta_c"]
        cos_rel_err_list.append(abs(res["cone"]["cos_theta_c"] - cos_th) / cos_th)

        # Spectral L2 relative error across the analysed band [12 GHz, 30 GHz]
        # where the finest grid meets >= 20 cells per wavelength in n = 1.5
        freqs_ref = res["freqs_hz"]
        spec_theory = res["d2W_domega_dz_theory"]
        band = (freqs_ref >= ANALYSED_FMIN_HZ) & (freqs_ref <= ANALYSED_FMAX_HZ)
        l2_err = float(
            np.linalg.norm(res["d2W_domega_dz_sim"][band] - spec_theory[band])
            / np.linalg.norm(spec_theory[band])
        )
        spec_l2_rel_err_list.append(l2_err)
        dw_err = abs(res["dW_dz_sim_J_m"] - res["dW_dz_theory_J_m"]) / res["dW_dz_theory_J_m"]
        dw_dz_rel_err_list.append(float(dw_err))
        wall_s_list.append(res["wall_s"])
        mem_mb_list.append(res["memory_mb"])
        spectra_sim.append(res["d2W_domega_dz_sim"])

    np.savez_compressed(
        DATA_DIR / "task_convergence.npz",
        level_names=np.asarray(names),
        dx_m=np.asarray(dx_list),
        dz_m=np.asarray(dz_list),
        ppw_30ghz=np.asarray(ppw_30ghz_list),
        ppw_50ghz=np.asarray(ppw_50ghz_list),
        analysed_band_ghz=np.array(
            [round(float(ANALYSED_FMIN_HZ * 1e-9), 6), round(float(ANALYSED_FMAX_HZ * 1e-9), 6)]
        ),
        theta_c_deg=np.asarray(theta_deg_list),
        theta_c_theory_deg=np.array(
            float(cherenkov_cone_angle(beta, n_refr)["theta_c_deg"])
        ),
        cos_theta_c=np.asarray(cos_theta_list),
        cos_theta_c_theory=np.array(
            float(cherenkov_cone_angle(beta, n_refr)["cos_theta_c"])
        ),
        cos_rel_err=np.asarray(cos_rel_err_list),
        spec_l2_rel_err=np.asarray(spec_l2_rel_err_list),
        dw_dz_rel_err=np.asarray(dw_dz_rel_err_list),
        wall_s=np.asarray(wall_s_list),
        memory_mb=np.asarray(mem_mb_list),
        freqs_hz=freqs_ref,
        spectra_sim=np.asarray(spectra_sim),
        spectrum_theory=spec_theory,
    )

    summary = {
        "task": "convergence",
        "quick": bool(quick),
        "analysed_band_ghz": [
            round(float(ANALYSED_FMIN_HZ * 1e-9), 6),
            round(float(ANALYSED_FMAX_HZ * 1e-9), 6),
        ],
        "lambda_30ghz_mm": float(lambda_30ghz_m * 1e3),
        "levels": names,
        "dx_mm": [float(x * 1e3) for x in dx_list],
        "dz_mm": [float(z * 1e3) for z in dz_list],
        "ppw_30ghz": [float(p) for p in ppw_30ghz_list],
        "ppw_50ghz": [float(p) for p in ppw_50ghz_list],
        "theta_c_deg": [float(t) for t in theta_deg_list],
        "theta_c_theory_deg": float(cherenkov_cone_angle(beta, n_refr)["theta_c_deg"]),
        "cos_rel_err": [float(e) for e in cos_rel_err_list],
        "spec_l2_rel_err": [float(e) for e in spec_l2_rel_err_list],
        "dw_dz_rel_err": [float(e) for e in dw_dz_rel_err_list],
        "wall_s": [float(w) for w in wall_s_list],
        "memory_mb": [float(m) for m in mem_mb_list],
    }
    (DATA_DIR / "task_convergence.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Figure and GIF rendering helpers
# ---------------------------------------------------------------------------


def render_geometry_and_mesh_figure() -> Path:
    """Render Figure 1: 3D setup/mesh view and Cherenkov cone geometry diagram."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_rendering(force_off_screen=True)

    l_xy_m = 11.0e-3
    zmin_m, zmax_m = -8.0e-3, 18.0e-3
    grid = GridFIT3D(-l_xy_m, l_xy_m, -l_xy_m, l_xy_m, zmin_m, zmax_m, 44, 44, 65, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml"] * 3,
        bc_high=["cpml"] * 3,
        bg=[DEFAULT_N_DIEL**2, 1.0],
        verbose=0,
    )
    r_box = 4.5e-3
    l_box = 8.0e-3
    z_mid = 5.0e-3
    flux_bounds = box_bounds((0.0, 0.0, z_mid), (2.0 * r_box, 2.0 * r_box, l_box))
    beam_bounds = box_bounds((0.0, 0.0, 0.5 * (zmin_m + zmax_m)), (0.8e-3, 0.8e-3, zmax_m - zmin_m))
    pl = plot_scene(
        grid,
        boxes={
            "Coaxial FluxMonitor": (flux_bounds, dict(color="#0072B2", label="FluxMonitor box")),
            "Beam axis (z)": (beam_bounds, dict(color="#D55E00", label="Beam trajectory (z)")),
        },
        cpml=(grid.grid.bounds, cpml_inner_bounds(solver)),
        solids=False,
        show_grid=True,
        length_unit="mm",
        window_size=(900, 680),
        off_screen=True,
    )
    scene_img = pl.screenshot(return_img=True)
    pl.close()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.08, 1.0]})
    ax0.imshow(scene_img)
    ax0.set_axis_off()
    ax0.set_title("(a) 3D FIT grid, CPML shell, and coaxial FluxMonitor", fontsize=11, fontweight="bold")

    # (b) Meridian schematic of Huygens wavelets, cone angle theta_c, and shock half-angle 90 - theta_c
    beta = 0.90
    n_refr = DEFAULT_N_DIEL
    cone = cherenkov_cone_angle(beta, n_refr)
    theta_c_rad = float(cone["theta_c_rad"])
    theta_c_deg = float(cone["theta_c_deg"])
    alpha_deg = float(cone["shock_half_angle_deg"])

    ax1.set_xlim(-6.5, 13.5)
    ax1.set_ylim(-9.5, 9.5)
    ax1.set_aspect("equal")
    ax1.axhspan(-9.5, 9.5, color="#eef5fb", alpha=0.55, label=f"Dielectric (n = {n_refr:.2f})")

    # CPML collar shading
    pml_w = 2.2
    ax1.axvspan(-6.5, -6.5 + pml_w, color="#999999", alpha=0.18, label="CPML absorbing layer")
    ax1.axvspan(13.5 - pml_w, 13.5, color="#999999", alpha=0.18)
    ax1.axhspan(9.5 - pml_w, 9.5, color="#999999", alpha=0.18)
    ax1.axhspan(-9.5, -9.5 + pml_w, color="#999999", alpha=0.18)

    # Coaxial FluxMonitor outline
    rect = Rectangle(
        (z_mid * 1e3 - 0.5 * l_box * 1e3, -r_box * 1e3),
        l_box * 1e3,
        2.0 * r_box * 1e3,
        fill=False,
        edgecolor="#0072B2",
        linestyle="--",
        linewidth=1.8,
        label="Coaxial FluxMonitor",
    )
    ax1.add_patch(rect)

    # Huygens spherical wavelets and Mach cone
    zb = 9.5
    for dt_z in [3.0, 6.0, 9.0, 12.0]:
        z_emit = zb - dt_z
        r_wave = dt_z / (beta * n_refr)
        circle = plt.Circle(
            (z_emit, 0.0),
            r_wave,
            fill=False,
            color="#56B4E9",
            linestyle=":",
            linewidth=1.1,
        )
        ax1.add_patch(circle)
        ax1.plot(z_emit, 0.0, ".", color="#56B4E9", ms=4)

    r_line = np.linspace(0.0, 7.6, 100)
    z_shock = zb - r_line * np.tan(theta_c_rad)
    ax1.plot(z_shock, +r_line, color="#D55E00", lw=2.3, label=f"Cherenkov shock front ($90^\\circ-\\theta_c={alpha_deg:.1f}^\\circ$)")
    ax1.plot(z_shock, -r_line, color="#D55E00", lw=2.3)

    # Wavevector k arrow normal to upper shock front
    z_k0, x_k0 = zb - 3.6 * np.tan(theta_c_rad), 3.6
    len_k = 3.2
    arr_k = FancyArrowPatch(
        (z_k0, x_k0),
        (z_k0 + len_k * np.cos(theta_c_rad), x_k0 + len_k * np.sin(theta_c_rad)),
        arrowstyle="->",
        mutation_scale=13,
        color="#009E73",
        lw=2.0,
    )
    ax1.add_patch(arr_k)
    ax1.text(
        z_k0 + 0.55,
        x_k0 + 2.35,
        f"$\\mathbf{{k}}$ ($\\theta_c={theta_c_deg:.1f}^\\circ$)\n$\\cos\\theta_c = 1/(\\beta n)$",
        color="#009E73",
        fontsize=9.5,
        fontweight="bold",
    )

    # Axis and moving bunch
    ax1.axhline(0.0, color="k", lw=0.9, alpha=0.5)
    ax1.plot(zb, 0.0, "o", color="#D55E00", ms=9, zorder=6, label="Gaussian bunch $q(z - \\beta ct)$")
    ax1.set_xlabel("Axial coordinate $z$ [mm]")
    ax1.set_ylabel("Transverse coordinate $x$ [mm]")
    ax1.set_title(f"(b) Huygens construction ($\\beta={beta:.2f},\\;n={n_refr:.2f},\\;\\beta n={beta*n_refr:.2f}>1$)", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=8.2, framealpha=0.92)

    fig.tight_layout()
    out_path = FIG_DIR / "01_geometry_and_mesh.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_field_maps_figure() -> Path:
    """Render Figure 2: Super-threshold Cherenkov cone (beta=0.9) vs sub-threshold Coulomb field (beta=0.6)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "task_dielectric.npz")
    x_mm = d["x_m"] * 1e3
    z_mm = d["z_m"] * 1e3
    zb_sup_mm = float(d["z_bunch_sup_m"]) * 1e3
    zb_sub_mm = float(d["z_bunch_sub_m"]) * 1e3
    ez_sup = d["Ez_sup_xz"]
    ez_sub = d["Ez_sub_xz"]
    theta_sim = float(d["theta_c_sup_deg"])
    theta_th = float(d["theta_c_sup_theory_deg"])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), gridspec_kw={"height_ratios": [1.15, 0.9]})
    ax_sup, ax_sub = axes[0]
    ax_ridge, ax_coul = axes[1]

    vmax_sup = float(np.percentile(np.abs(ez_sup), 98.5))
    im0 = ax_sup.pcolormesh(z_mm, x_mm, ez_sup * 1e-3, cmap="RdBu_r", vmin=-vmax_sup * 1e-3, vmax=vmax_sup * 1e-3, shading="auto")
    fig.colorbar(im0, ax=ax_sup, label="$E_z(x, 0, z)$ [kV/m]")
    r_line_mm = np.linspace(0.0, 9.0, 100)
    z_cone_th = zb_sup_mm - r_line_mm * np.tan(np.radians(theta_th))
    ax_sup.plot(z_cone_th, +r_line_mm, "k--", lw=1.8, label=f"Analytic cone $\\theta_c={theta_th:.1f}^\\circ$")
    ax_sup.plot(z_cone_th, -r_line_mm, "k--", lw=1.8)
    ax_sup.plot(d["z_ridge_sup_m"] * 1e3, d["r_fit_sup_m"] * 1e3, "o", color="#009E73", ms=4.5, label=f"Fitted ridge ($\\theta_c={theta_sim:.1f}^\\circ$)")
    ax_sup.plot(zb_sup_mm, 0.0, "*", color="#F0E442", markeredgecolor="k", ms=12, label="Bunch centroid")
    ax_sup.set_title("(a) Super-threshold ($\\beta = 0.90,\\;\\beta n = 1.35 > 1$): Cherenkov cone", fontsize=10.5, fontweight="bold")
    ax_sup.set_xlabel("$z$ [mm]")
    ax_sup.set_ylabel("$x$ [mm]")
    ax_sup.legend(loc="upper left", fontsize=8.2)

    vmax_sub = float(np.percentile(np.abs(ez_sub), 98.5))
    im1 = ax_sub.pcolormesh(z_mm, x_mm, ez_sub * 1e-3, cmap="RdBu_r", vmin=-vmax_sub * 1e-3, vmax=vmax_sub * 1e-3, shading="auto")
    fig.colorbar(im1, ax=ax_sub, label="$E_z(x, 0, z)$ [kV/m]")
    ax_sub.plot(zb_sub_mm, 0.0, "*", color="#F0E442", markeredgecolor="k", ms=12, label="Bunch centroid")
    ax_sub.axvline(zb_sub_mm, color="k", ls=":", lw=1.1, label="Symmetry plane $z = z_b$")
    ax_sub.set_title("(b) Sub-threshold ($\\beta = 0.60,\\;\\beta n = 0.90 < 1$): Contracted Coulomb field", fontsize=10.5, fontweight="bold")
    ax_sub.set_xlabel("$z$ [mm]")
    ax_sub.set_ylabel("$x$ [mm]")
    ax_sub.legend(loc="upper left", fontsize=8.2)

    # (c) Ridge linear fit z_b - z_ridge(r) vs r * tan(theta_c)
    r_fit_mm = d["r_fit_sup_m"] * 1e3
    lag_sim_mm = zb_sup_mm - d["z_ridge_sup_m"] * 1e3
    slope_fit, intercept_fit = np.polyfit(r_fit_mm, lag_sim_mm, 1)
    r_grid = np.linspace(r_fit_mm.min() - 0.4, r_fit_mm.max() + 0.4, 100)
    ax_ridge.plot(
        r_grid,
        np.tan(np.radians(theta_th)) * r_grid + intercept_fit,
        "k-",
        lw=1.8,
        label=f"Theory slope $\\tan\\theta_c = {np.tan(np.radians(theta_th)):.3f}$ ($\\theta_c={theta_th:.2f}^\\circ$)",
    )
    ax_ridge.plot(
        r_fit_mm,
        lag_sim_mm,
        "o",
        color="#D55E00",
        ms=5.5,
        label=f"Wakis (FIT) ridge ($\\theta_c={theta_sim:.2f}^\\circ,\\;\\cos\\theta_c={float(d['cos_theta_c_sup']):.4f}$)",
    )
    ax_ridge.set_xlabel("Transverse distance $r = |x|$ [mm]")
    ax_ridge.set_ylabel("Axial shock lag $z_b - z_{\\mathrm{ridge}}(r)$ [mm]")
    ax_ridge.set_title("(c) Wavefront cone-angle extraction at $\\beta = 0.90$", fontsize=10.5, fontweight="bold")
    ax_ridge.grid(True, alpha=0.3)
    ax_ridge.legend(loc="upper left", fontsize=8.5)

    # (d) Sub-threshold off-axis probe at beta = 0.60 vs exact contracted Coulomb integral
    zeta_mm = d["zeta_sub_m"] * 1e3
    r_p_mm = float(d["r_probe_sub_m"]) * 1e3
    mask_z = (zeta_mm >= -10.0) & (zeta_mm <= 10.0)
    zeta_dense_mm = np.linspace(-10.0, 10.0, 300)
    coul_dense = subthreshold_coulomb_field(
        r_m=float(d["r_probe_sub_m"]),
        zeta_m=zeta_dense_mm * 1e-3,
        beta=0.60,
        n_refr=DEFAULT_N_DIEL,
        q_c=DEFAULT_Q_C,
        sigmaz_m=DEFAULT_SIGMAZ_M,
    )
    ax_coul.plot(
        zeta_dense_mm,
        coul_dense["Ez"] * 1e-3,
        "k-",
        lw=1.8,
        label=f"Contracted Coulomb theory ($r={r_p_mm:.1f}$ mm)",
    )
    ax_coul.plot(
        zeta_mm[mask_z],
        d["Ez_sub_probe_sim"][mask_z] * 1e-3,
        "o",
        color="#0072B2",
        ms=4.2,
        label="Wakis (FIT) off-axis probe",
    )
    ax_coul.axhline(0.0, color="grey", lw=0.8)
    ax_coul.axvline(0.0, color="grey", ls=":", lw=0.9)
    ax_coul.set_xlim(-10.2, 10.2)
    ax_coul.set_xlabel("Comoving coordinate $\\zeta = z - vt$ [mm]")
    ax_coul.set_ylabel("$E_z(r, \\zeta)$ [kV/m]")
    ax_coul.set_title("(d) Evanescent antisymmetric Coulomb profile ($\\beta = 0.60$)", fontsize=10.5, fontweight="bold")
    ax_coul.grid(True, alpha=0.3)
    ax_coul.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    out_path = FIG_DIR / "02_field_maps_cone_vs_coulomb.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_cone_angle_vs_beta_figure() -> Path:
    """Render Figure 3: Cherenkov cone angle theta_c and cos(theta_c) vs beta with threshold."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "task_dielectric.npz")
    beta_sw = d["beta_sweep"]
    theta_sim = d["theta_sweep_sim_deg"]
    cos_sim = d["cos_sweep_sim"]
    det = d["cone_detected_sweep"].astype(bool)

    n_refr = DEFAULT_N_DIEL
    beta_th = 1.0 / n_refr
    beta_dense = np.linspace(beta_th + 1e-4, 0.999, 300)
    th_dense = cherenkov_cone_angle(beta_dense, n_refr)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.2, 4.8))

    ax0.axvspan(0.52, beta_th, color="#999999", alpha=0.18, label=f"Below threshold ($\\beta < 1/n = {beta_th:.3f}$)")
    ax0.axvline(beta_th, color="#D55E00", ls="--", lw=1.6)
    ax0.plot(beta_dense, th_dense["theta_c_deg"], "k-", lw=2.0, label="Theory $\\theta_c = \\arccos(1/(\\beta n))$")
    ax0.plot(beta_dense, th_dense["shock_half_angle_deg"], color="#0072B2", ls="-.", lw=1.7, label="Shock half-angle $90^\\circ - \\theta_c$")
    ax0.plot(beta_sw[det], theta_sim[det], "o", color="#D55E00", ms=7, label="Wakis (FIT) $\\theta_c$")
    ax0.plot(beta_sw[det], 90.0 - theta_sim[det], "s", color="#0072B2", ms=6.5, label="Wakis (FIT) $90^\\circ - \\theta_c$")
    for b_sub in beta_sw[~det]:
        ax0.plot(b_sub, 0.0, "x", color="#333333", ms=8, mew=2, label="No cone detected ($\\beta n < 1$)")
    ax0.set_xlim(0.55, 1.0)
    ax0.set_ylim(-3.0, 60.0)
    ax0.set_xlabel("Normalized particle velocity $\\beta = v/c$ [-]")
    ax0.set_ylabel("Angle [deg]")
    ax0.set_title("(a) Wavevector angle $\\theta_c$ and Mach half-angle $90^\\circ - \\theta_c$", fontsize=10.5, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=8.3)

    ax1.axvspan(0.52, beta_th, color="#999999", alpha=0.18, label="Sub-threshold (evanescent Coulomb)")
    ax1.axvline(beta_th, color="#D55E00", ls="--", lw=1.6, label=f"Threshold $\\beta_{{\\mathrm{{th}}}} = 1/n = {beta_th:.3f}$")
    ax1.plot(beta_dense, th_dense["cos_theta_c"], "k-", lw=2.0, label="Theory $\\cos\\theta_c = 1/(\\beta n)$")
    ax1.plot(beta_sw[det], cos_sim[det], "o", color="#D55E00", ms=7, label="Wakis (FIT) $\\cos\\theta_c$")
    ax1.set_xlim(0.55, 1.0)
    ax1.set_ylim(0.60, 1.05)
    ax1.set_xlabel("Normalized particle velocity $\\beta = v/c$ [-]")
    ax1.set_ylabel("$\\cos\\theta_c$ [-]")
    ax1.set_title("(b) Cherenkov coherence condition $\\cos\\theta_c = 1/(\\beta n)$", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    out_path = FIG_DIR / "03_cone_angle_vs_beta.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_frank_tamm_spectrum_figure() -> Path:
    """Render Figure 4: Radiated spectrum per unit length vs Frank-Tamm and Gaussian bunch form factor."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "task_dielectric.npz")
    f_ghz = d["freqs_hz"] * 1e-9
    d2w_sup_sim = d["d2W_sup_sim"]
    d2w_sup_th = d["d2W_sup_theory"]
    d2w_sup_pt = d["d2W_sup_point"]
    d2w_sub_sim = d["d2W_sub_sim"]
    ff = d["form_factor_sup"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 4.9))

    # (a) Log-scale spectral energy density per unit length
    f_min_an = ANALYSED_FMIN_HZ * 1e-9
    f_max_an = ANALYSED_FMAX_HZ * 1e-9
    ax0.axvspan(f_min_an, f_max_an, color="#009E73", alpha=0.10, label=f"Analysed band {f_min_an:.0f}–{f_max_an:.0f} GHz ($\\geq 20\\;\\mathrm{{cells}}/\\lambda$)")
    ax0.semilogy(f_ghz, d2w_sup_pt, "k--", lw=1.5, label="Point-charge Frank–Tamm $\\propto \\omega$")
    ax0.semilogy(
        f_ghz,
        d2w_sup_th,
        "k-",
        lw=2.0,
        label="Frank–Tamm $\\times\\;F(\\omega) = e^{-(\\omega\\sigma_z/v)^2}$ ($\\beta=0.90$)",
    )
    ax0.semilogy(
        f_ghz[::2],
        np.maximum(d2w_sup_sim[::2], 1e-21),
        "o",
        color="#D55E00",
        ms=5.2,
        label="Wakis (FIT) $\\beta = 0.90$ ($\\beta n = 1.35 > 1$)",
    )
    ax0.semilogy(
        f_ghz[::2],
        np.maximum(d2w_sub_sim[::2], 1e-21),
        "s",
        color="#0072B2",
        ms=4.8,
        label="Wakis (FIT) $\\beta = 0.60$ ($\\beta n = 0.90 < 1$, floor)",
    )
    f_cutoff_ghz = (0.90 * c_light / (2.0 * np.pi * DEFAULT_SIGMAZ_M)) * 1e-9
    ax0.axvline(f_cutoff_ghz, color="#009E73", ls=":", lw=1.5, label=f"Bunch roll-off $v/(2\\pi\\sigma_z)={f_cutoff_ghz:.1f}$ GHz")
    ax0.set_ylim(1e-21, 2e-14)
    ax0.set_xlabel("Frequency $f = \\omega/(2\\pi)$ [GHz]")
    ax0.set_ylabel("$d^2 W / (d\\omega\\,dz)$ [J$\\cdot$s/m]")
    ax0.set_title("(a) Radiated spectrum vs Frank–Tamm (log scale)", fontsize=10.5, fontweight="bold")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(loc="lower left", fontsize=7.8)

    # (b) Linear scale + Gaussian bunch form factor on twin axis
    ax1.axvspan(f_min_an, f_max_an, color="#009E73", alpha=0.10, label=f"Analysed band ({f_min_an:.0f}–{f_max_an:.0f} GHz)")
    ax1.plot(f_ghz, d2w_sup_th * 1e15, "k-", lw=2.0, label="Frank–Tamm $\\times\\;F(\\omega)$ theory")
    ax1.plot(f_ghz[::2], d2w_sup_sim[::2] * 1e15, "o", color="#D55E00", ms=5.2, label="Wakis (FIT) $\\beta = 0.90$")
    ax1.plot(f_ghz[::2], d2w_sub_sim[::2] * 1e15, "s", color="#0072B2", ms=4.5, label="Wakis (FIT) $\\beta = 0.60$ (below threshold)")
    ax1.set_xlabel("Frequency $f$ [GHz]")
    ax1.set_ylabel("$d^2 W / (d\\omega\\,dz)$ [$10^{-15}$ J$\\cdot$s/m]")
    ax1.set_title("(b) Linear spectrum and Gaussian bunch form factor $F(\\omega)$", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax1_r = ax1.twinx()
    ax1_r.plot(f_ghz, ff, color="#009E73", ls="--", lw=1.6, label="Form factor $F(\\omega) = e^{-(\\omega\\sigma_z/v)^2}$")
    ax1_r.set_ylabel("Bunch form factor $F(\\omega)$ [-]", color="#009E73")
    ax1_r.tick_params(axis="y", labelcolor="#009E73")
    ax1_r.set_ylim(-0.02, 1.08)

    lines_l, labels_l = ax1.get_legend_handles_labels()
    lines_r, labels_r = ax1_r.get_legend_handles_labels()
    ax1.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right", fontsize=8.2)

    fig.tight_layout()
    out_path = FIG_DIR / "04_frank_tamm_spectrum.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_dispersive_band_figure() -> Path:
    """Render Figure 5: Dispersive Lorentz medium refractive index and Cherenkov band edge."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "task_dispersive.npz")
    f_ghz = d["freqs_hz"] * 1e-9
    n_real = d["n_real"]
    n_imag = d["n_imag"]
    beta = float(d["beta"])
    f0_ghz = float(d["f0_hz"]) * 1e-9
    n_th = 1.0 / beta
    d2w_sim = d["d2W_sim"]
    d2w_th_unatt = d["d2W_theory_unattenuated"]
    d2w_th_att = d["d2W_theory_attenuated"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 4.9))

    # (a) Refractive index n(omega) vs threshold 1/beta
    ax0.axvspan(f_ghz.min(), f0_ghz, color="#009E73", alpha=0.12, label="Cherenkov passband ($\\beta\\,\\mathrm{Re}\\,n(\\omega) > 1$)")
    ax0.axvspan(f0_ghz, f_ghz.max(), color="#999999", alpha=0.18, label="Reststrahlen stopband ($\\beta\\,\\mathrm{Re}\\,n(\\omega) < 1$)")
    ax0.plot(f_ghz, n_real, "k-", lw=2.0, label="$\\mathrm{Re}\\,n(\\omega)$")
    ax0.plot(f_ghz, n_imag, color="#0072B2", ls="--", lw=1.7, label="$\\mathrm{Im}\\,n(\\omega)$ (extinction)")
    ax0.axhline(n_th, color="#D55E00", ls="-.", lw=1.8, label=f"Threshold $1/\\beta = {n_th:.3f}$ ($\\beta={beta:.2f}$)")
    ax0.axvline(f0_ghz, color="#CC79A7", ls=":", lw=1.6, label=f"Lorentz resonance $f_0 = {f0_ghz:.0f}$ GHz")
    ax0.set_xlim(f_ghz.min(), f_ghz.max())
    ax0.set_ylim(0.0, 5.2)
    ax0.set_xlabel("Frequency $f$ [GHz]")
    ax0.set_ylabel("Refractive index $n(\\omega)$ [-]")
    ax0.set_title("(a) Lorentz medium dispersion and Cherenkov threshold", fontsize=10.5, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=8.0)

    # (b) Radiated spectrum showing band edge at f0
    ax1.axvspan(f_ghz.min(), f0_ghz, color="#009E73", alpha=0.12, label="Passband ($\\beta\\,\\mathrm{Re}\\,n > 1$)")
    ax1.axvspan(f0_ghz, f_ghz.max(), color="#999999", alpha=0.18, label="Stopband ($\\beta\\,\\mathrm{Re}\\,n < 1$)")
    ax1.semilogy(f_ghz, np.maximum(d2w_th_unatt, 1e-20), "k--", lw=1.5, label="Lossless Frank–Tamm $\\times\\;F(\\omega)$")
    ax1.semilogy(f_ghz, np.maximum(d2w_th_att, 1e-20), "k-", lw=2.0, label="Attenuated Frank–Tamm at $r_{\\mathrm{box}}$")
    ax1.semilogy(f_ghz, np.maximum(d2w_sim, 1e-20), "o", color="#D55E00", ms=5.0, label="Wakis (FIT + ADE Lorentz)")
    ax1.axvline(f0_ghz, color="#CC79A7", ls=":", lw=1.8, label=f"Band edge $f_0 = {f0_ghz:.0f}$ GHz")
    ax1.set_xlim(f_ghz.min(), f_ghz.max())
    ax1.set_ylim(1e-19, 1e-14)
    ax1.set_xlabel("Frequency $f$ [GHz]")
    ax1.set_ylabel("$d^2 W / (d\\omega\\,dz)$ [J$\\cdot$s/m]")
    ax1.set_title("(b) Cherenkov spectrum cut off at the dispersive band edge", fontsize=10.5, fontweight="bold")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.0)

    fig.tight_layout()
    out_path = FIG_DIR / "05_dispersive_cherenkov_band.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_numerical_cherenkov_figure() -> Path:
    """Render Figure 6: Vacuum beta=0.999 Yee vs TETE numerical Cherenkov comparison and scaling."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "task_numerical_cherenkov.npz")
    x_mm = d["x_m"] * 1e3
    z_mm = d["z_m"] * 1e3
    zb_mm = float(d["z_bunch_m"]) * 1e3
    ez_yee = d["Ez_yee_xz"]
    ez_tete = d["Ez_tete_xz"]
    kz_dz = d["kz_dz"]
    vp_yee = d["vp_yee_over_c"]
    vp_tete = d["vp_tete_over_c"]
    beta = float(d["beta"])
    ppw = d["ppw_char"]
    w_yee = d["W_wake_yee_J"]
    w_tete = d["W_wake_tete_J"]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    ax_yee, ax_tete = axes[0]
    ax_disp, ax_scale = axes[1]

    vmax = float(np.percentile(np.abs(ez_yee), 99.0))
    im_y = ax_yee.pcolormesh(z_mm, x_mm, ez_yee * 1e-3, cmap="RdBu_r", vmin=-vmax * 1e-3, vmax=vmax * 1e-3, shading="auto")
    fig.colorbar(im_y, ax=ax_yee, label="$E_z(x, 0, z)$ [kV/m]")
    ax_yee.plot(zb_mm, 0.0, "*", color="#F0E442", markeredgecolor="k", ms=12, label="Bunch ($\\beta=0.999$)")
    ax_yee.set_title("(a) Standard Yee scheme in vacuum: spurious numerical Cherenkov wake", fontsize=10.2, fontweight="bold")
    ax_yee.set_xlabel("$z$ [mm]")
    ax_yee.set_ylabel("$x$ [mm]")
    ax_yee.legend(loc="upper left", fontsize=8.5)

    im_t = ax_tete.pcolormesh(z_mm, x_mm, ez_tete * 1e-3, cmap="RdBu_r", vmin=-vmax * 1e-3, vmax=vmax * 1e-3, shading="auto")
    fig.colorbar(im_t, ax=ax_tete, label="$E_z(x, 0, z)$ [kV/m]")
    ax_tete.plot(zb_mm, 0.0, "*", color="#F0E442", markeredgecolor="k", ms=12, label="Bunch ($\\beta=0.999$)")
    ax_tete.set_title("(b) Explicit TE/TM (`scheme='tete'`) in vacuum: no spurious cone", fontsize=10.2, fontweight="bold")
    ax_tete.set_xlabel("$z$ [mm]")
    ax_tete.set_ylabel("$x$ [mm]")
    ax_tete.legend(loc="upper left", fontsize=8.5)

    # (c) Numerical phase velocity along z vs k_z * dz
    ax_disp.plot(kz_dz, vp_tete, color="#0072B2", lw=2.2, label="TE/TM (`scheme='tete'`, $c\\Delta t = \\Delta z$): $v_{p,z}/c \\equiv 1$")
    ax_disp.plot(kz_dz, vp_yee, color="#D55E00", lw=2.2, label="Standard Yee (`scheme='yee'`): $v_{p,z}(k_z)/c < 1$")
    ax_disp.axhline(beta, color="k", ls="--", lw=1.5, label=f"Beam velocity $\\beta = {beta:.3f}$")
    ncr_mask = vp_yee <= beta
    if np.any(ncr_mask):
        kz_crit = float(kz_dz[np.argmax(ncr_mask)])
        ax_disp.axvspan(kz_crit, kz_dz.max(), color="#D55E00", alpha=0.14, label="Spurious Cherenkov band ($v_{p,z}^{\\mathrm{Yee}} < \\beta c$)")
    ax_disp.set_xlim(0.0, np.pi)
    ax_disp.set_ylim(0.88, 1.015)
    ax_disp.set_xlabel("Normalized longitudinal wavenumber $k_z \\Delta z$ [rad]")
    ax_disp.set_ylabel("Numerical phase velocity $v_{p,z} / c$ [-]")
    ax_disp.set_title("(c) Longitudinal numerical dispersion: Yee vs TE/TM", fontsize=10.2, fontweight="bold")
    ax_disp.grid(True, alpha=0.3)
    ax_disp.legend(loc="lower left", fontsize=8.0)

    # (d) Spurious trailing wake energy vs cells per characteristic wavelength 2*pi*sigma_z / dz
    ax_scale.loglog(ppw, w_yee, "o-", color="#D55E00", lw=1.8, ms=6.5, label="Yee (`scheme='yee'`) trailing wake energy")
    ax_scale.loglog(ppw, w_tete, "s-", color="#0072B2", lw=1.8, ms=6.5, label="TE/TM (`scheme='tete'`) residual floor")
    ax_scale.set_xlim(10.8, 30.5)
    ax_scale.set_xticks([12, 15, 20, 25, 30])
    ax_scale.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_scale.get_xaxis().set_minor_formatter(NullFormatter())
    ax_scale.set_xlabel("Cells per characteristic bunch wavelength $2\\pi\\sigma_z / \\Delta z$ [-]")
    ax_scale.set_ylabel("Trailing wake electric energy $W_{\\mathrm{wake}}$ [J]")
    ax_scale.set_title("(d) Spurious vacuum wake energy vs longitudinal resolution", fontsize=10.2, fontweight="bold")
    ax_scale.grid(True, which="both", alpha=0.3)
    ax_scale.legend(loc="upper right", fontsize=8.3)

    fig.tight_layout()
    out_path = FIG_DIR / "06_numerical_cherenkov_yee_vs_tete.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_resolution_convergence_figure() -> Path:
    """Render Figure 7: Grid convergence of Cherenkov cone angle and Frank-Tamm spectrum."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "task_convergence.npz")
    ppw = d["ppw_30ghz"] if "ppw_30ghz" in d else d["ppw_50ghz"] * (50.0 / 30.0)
    cos_err = d["cos_rel_err"]
    spec_err = d["spec_l2_rel_err"]
    dw_err = d["dw_dz_rel_err"]
    f_ghz = d["freqs_hz"] * 1e-9
    spectra_sim = d["spectra_sim"]
    spec_th = d["spectrum_theory"]
    names = [str(s) for s in d["level_names"]]
    f_min_an = ANALYSED_FMIN_HZ * 1e-9
    f_max_an = ANALYSED_FMAX_HZ * 1e-9

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    ax0.loglog(ppw, cos_err * 100.0, "o-", color="#D55E00", lw=1.9, ms=7, label="Cone-angle error $|\\cos\\theta_c - 1/(\\beta n)| / (1/(\\beta n))$")
    ax0.loglog(ppw, spec_err * 100.0, "s-", color="#0072B2", lw=1.9, ms=6.5, label=f"Spectrum $L_2$ error ({f_min_an:.0f}–{f_max_an:.0f} GHz)")
    ax0.loglog(ppw, dw_err * 100.0, "^-", color="#009E73", lw=1.9, ms=6.5, label=f"Integrated $dW/dz$ error ({f_min_an:.0f}–{f_max_an:.0f} GHz)")
    # O(Delta x^2) reference slope
    ref_line = cos_err[0] * 100.0 * (ppw[0] / ppw) ** 2
    ax0.loglog(ppw, ref_line, "k--", lw=1.3, label="$\\mathcal{O}(\\Delta x^2)$ second-order reference")
    ax0.axvline(20.0, color="#333333", ls=":", lw=1.3, label="20 cells/$\\lambda$ rule at 30 GHz")
    ax0.set_xlim(11.0, 22.5)
    ax0.set_xticks([12, 15, 20])
    ax0.get_xaxis().set_major_formatter(ScalarFormatter())
    ax0.get_xaxis().set_minor_formatter(NullFormatter())
    ax0.set_yticks([0.6, 1, 2, 3, 5, 7])
    ax0.get_yaxis().set_major_formatter(ScalarFormatter())
    ax0.get_yaxis().set_minor_formatter(NullFormatter())
    ax0.set_xlabel("Cells per wavelength at $30$ GHz in $n=1.5$ ($\\lambda_{30\\,\\mathrm{GHz}} / \\Delta x$) [-]")
    ax0.set_ylabel("Relative error [%]")
    ax0.set_title("(a) Convergence of cone angle and Frank–Tamm spectrum", fontsize=10.5, fontweight="bold")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(loc="lower left", fontsize=7.8)

    colors = ["#E69F00", "#0072B2", "#D55E00"]
    markers = ["^", "s", "o"]
    ax1.axvspan(f_min_an, f_max_an, color="#009E73", alpha=0.10, label=f"Analysed band ({f_min_an:.0f}–{f_max_an:.0f} GHz)")
    ax1.axvline(f_max_an, color="#009E73", ls=":", lw=1.4)
    ax1.plot(f_ghz, spec_th * 1e15, "k-", lw=2.0, label="Analytical Frank–Tamm $\\times\\;F(\\omega)$")
    for idx, label in enumerate(names):
        ax1.plot(
            f_ghz[::2],
            spectra_sim[idx][::2] * 1e15,
            marker=markers[idx % len(markers)],
            linestyle="none",
            color=colors[idx % len(colors)],
            ms=5.0,
            label=f"Wakis ({label}, $\\lambda_{{30\\,\\mathrm{{GHz}}}}/\\Delta x={ppw[idx]:.1f}$)",
        )
    ax1.set_xlabel("Frequency $f$ [GHz]")
    ax1.set_ylabel("$d^2 W / (d\\omega\\,dz)$ [$10^{-15}$ J$\\cdot$s/m]")
    ax1.set_title("(b) Spectral convergence across 3D grid resolutions", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.0)

    fig.tight_layout()
    out_path = FIG_DIR / "07_resolution_convergence.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_animations() -> list[Path]:
    """Render the 3 required animated GIFs (< 5 MB each)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d_diel = np.load(DATA_DIR / "task_dielectric.npz")
    d_ncr = np.load(DATA_DIR / "task_numerical_cherenkov.npz")

    x_mm = d_diel["x_m"] * 1e3
    z_mm = d_diel["z_m"] * 1e3
    frames_sup = d_diel["frames_Ez_sup"] * 1e-3
    zb_sup_mm = d_diel["frames_zb_sup_m"] * 1e3
    frames_sub = d_diel["frames_Ez_sub"] * 1e-3
    zb_sub_mm = d_diel["frames_zb_sub_m"] * 1e3
    theta_th_rad = np.radians(float(d_diel["theta_c_sup_theory_deg"]))

    # --- GIF 1: Single-panel dielectric Cherenkov cone at beta = 0.90 ---
    gif1_path = FIG_DIR / "cherenkov_cone_beta09.gif"
    vmax1 = float(np.percentile(np.abs(frames_sup), 99.0))
    fig1, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=100)
    im1 = ax1.pcolormesh(
        z_mm,
        x_mm,
        frames_sup[0],
        cmap="RdBu_r",
        vmin=-vmax1,
        vmax=vmax1,
        shading="auto",
    )
    fig1.colorbar(im1, ax=ax1, label="$E_z(x, 0, z)$ [kV/m]")
    r_line_mm = np.linspace(0.0, 9.2, 80)
    (cone_up,) = ax1.plot([], [], "k--", lw=1.7, label="Analytic cone $\\cos\\theta_c = 1/(\\beta n)$")
    (cone_dn,) = ax1.plot([], [], "k--", lw=1.7)
    (bunch_pt,) = ax1.plot([], [], "*", color="#F0E442", markeredgecolor="k", ms=11, label="Bunch centroid")
    ax1.set_xlim(z_mm.min(), z_mm.max())
    ax1.set_ylim(x_mm.min(), x_mm.max())
    ax1.set_xlabel("$z$ [mm]")
    ax1.set_ylabel("$x$ [mm]")
    ax1.legend(loc="upper left", fontsize=8.2)

    def _update1(frame_idx: int):
        im1.set_array(frames_sup[frame_idx].ravel())
        zb = zb_sup_mm[frame_idx]
        z_c = zb - r_line_mm * np.tan(theta_th_rad)
        cone_up.set_data(z_c, +r_line_mm)
        cone_dn.set_data(z_c, -r_line_mm)
        bunch_pt.set_data([zb], [0.0])
        ax1.set_title(
            f"Cherenkov cone in dielectric ($n=1.5,\\;\\beta=0.90,\\;z_b={zb:5.1f}$ mm)",
            fontsize=10,
            fontweight="bold",
        )
        return im1, cone_up, cone_dn, bunch_pt

    fig1.tight_layout()
    ani1 = animation.FuncAnimation(fig1, _update1, frames=len(frames_sup), interval=90, blit=False)
    ani1.save(gif1_path, writer=animation.PillowWriter(fps=11))
    plt.close(fig1)

    # --- GIF 2: Side-by-side beta = 0.60 (no cone) vs beta = 0.90 (Cherenkov cone) ---
    gif2_path = FIG_DIR / "cherenkov_sub_vs_super_threshold.gif"
    n_f2 = min(len(frames_sub), len(frames_sup))
    vmax_sub = float(np.percentile(np.abs(frames_sub), 99.0))
    fig2, (ax2_l, ax2_r) = plt.subplots(1, 2, figsize=(10.5, 4.1), dpi=95)
    im2_l = ax2_l.pcolormesh(z_mm, x_mm, frames_sub[0], cmap="RdBu_r", vmin=-vmax_sub, vmax=vmax_sub, shading="auto")
    fig2.colorbar(im2_l, ax=ax2_l, label="$E_z$ [kV/m]")
    (pt2_l,) = ax2_l.plot([], [], "*", color="#F0E442", markeredgecolor="k", ms=10)
    ax2_l.set_xlabel("$z$ [mm]")
    ax2_l.set_ylabel("$x$ [mm]")

    im2_r = ax2_r.pcolormesh(z_mm, x_mm, frames_sup[0], cmap="RdBu_r", vmin=-vmax1, vmax=vmax1, shading="auto")
    fig2.colorbar(im2_r, ax=ax2_r, label="$E_z$ [kV/m]")
    (pt2_r,) = ax2_r.plot([], [], "*", color="#F0E442", markeredgecolor="k", ms=10)
    (c2_up,) = ax2_r.plot([], [], "k--", lw=1.5)
    (c2_dn,) = ax2_r.plot([], [], "k--", lw=1.5)
    ax2_r.set_xlim(z_mm.min(), z_mm.max())
    ax2_r.set_ylim(x_mm.min(), x_mm.max())
    ax2_r.set_xlabel("$z$ [mm]")
    ax2_r.set_ylabel("$x$ [mm]")

    def _update2(frame_idx: int):
        im2_l.set_array(frames_sub[frame_idx].ravel())
        im2_r.set_array(frames_sup[frame_idx].ravel())
        zb_l = zb_sub_mm[frame_idx]
        zb_r = zb_sup_mm[frame_idx]
        pt2_l.set_data([zb_l], [0.0])
        pt2_r.set_data([zb_r], [0.0])
        z_c = zb_r - r_line_mm * np.tan(theta_th_rad)
        c2_up.set_data(z_c, +r_line_mm)
        c2_dn.set_data(z_c, -r_line_mm)
        ax2_l.set_title(f"$\\beta=0.60$ ($\\beta n=0.90<1$): Coulomb field", fontsize=9.5, fontweight="bold")
        ax2_r.set_title(f"$\\beta=0.90$ ($\\beta n=1.35>1$): Cherenkov cone", fontsize=9.5, fontweight="bold")
        return im2_l, im2_r, pt2_l, pt2_r, c2_up, c2_dn

    fig2.tight_layout()
    ani2 = animation.FuncAnimation(fig2, _update2, frames=n_f2, interval=95, blit=False)
    ani2.save(gif2_path, writer=animation.PillowWriter(fps=10))
    plt.close(fig2)

    # --- GIF 3: Vacuum beta = 0.999 Yee vs TETE side-by-side ---
    gif3_path = FIG_DIR / "numerical_cherenkov_yee_vs_tete.gif"
    x_v_mm = d_ncr["x_m"] * 1e3
    z_v_mm = d_ncr["z_m"] * 1e3
    f_yee = d_ncr["frames_Ez_yee"] * 1e-3
    f_tete = d_ncr["frames_Ez_tete"] * 1e-3
    zb_v_mm = d_ncr["frames_zb_m"] * 1e3
    vmax3 = float(np.percentile(np.abs(f_yee), 99.0))

    fig3, (ax3_l, ax3_r) = plt.subplots(1, 2, figsize=(10.5, 4.1), dpi=95)
    im3_l = ax3_l.pcolormesh(z_v_mm, x_v_mm, f_yee[0], cmap="RdBu_r", vmin=-vmax3, vmax=vmax3, shading="auto")
    fig3.colorbar(im3_l, ax=ax3_l, label="$E_z$ [kV/m]")
    (pt3_l,) = ax3_l.plot([], [], "*", color="#F0E442", markeredgecolor="k", ms=10)
    ax3_l.set_xlabel("$z$ [mm]")
    ax3_l.set_ylabel("$x$ [mm]")

    im3_r = ax3_r.pcolormesh(z_v_mm, x_v_mm, f_tete[0], cmap="RdBu_r", vmin=-vmax3, vmax=vmax3, shading="auto")
    fig3.colorbar(im3_r, ax=ax3_r, label="$E_z$ [kV/m]")
    (pt3_r,) = ax3_r.plot([], [], "*", color="#F0E442", markeredgecolor="k", ms=10)
    ax3_r.set_xlabel("$z$ [mm]")
    ax3_r.set_ylabel("$x$ [mm]")

    def _update3(frame_idx: int):
        im3_l.set_array(f_yee[frame_idx].ravel())
        im3_r.set_array(f_tete[frame_idx].ravel())
        zb = zb_v_mm[frame_idx]
        pt3_l.set_data([zb], [0.0])
        pt3_r.set_data([zb], [0.0])
        ax3_l.set_title(f"Vacuum $\\beta=0.999$ Yee: numerical Cherenkov", fontsize=9.5, fontweight="bold")
        ax3_r.set_title(f"Vacuum $\\beta=0.999$ TE/TM (`tete`): dispersion-free", fontsize=9.5, fontweight="bold")
        return im3_l, im3_r, pt3_l, pt3_r

    fig3.tight_layout()
    ani3 = animation.FuncAnimation(fig3, _update3, frames=len(f_yee), interval=95, blit=False)
    ani3.save(gif3_path, writer=animation.PillowWriter(fps=10))
    plt.close(fig3)

    return [gif1_path, gif2_path, gif3_path]


def render_all_figures() -> list[Path]:
    """Render all 7 static PNG figures and 3 animated GIFs from saved data/."""
    paths = [
        render_geometry_and_mesh_figure(),
        render_field_maps_figure(),
        render_cone_angle_vs_beta_figure(),
        render_frank_tamm_spectrum_figure(),
        render_dispersive_band_figure(),
        render_numerical_cherenkov_figure(),
        render_resolution_convergence_figure(),
    ]
    paths.extend(render_animations())
    return paths
