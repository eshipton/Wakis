"""Physics, 3D FIT runners, and plotting helpers for Tutorial 8.4 (Smith-Purcell & DLA)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.constants import c as c_light
from scipy.constants import e as q_elem
from scipy.constants import epsilon_0, m_e, mu_0

from wakis import GridFIT3D, SolverFIT3D
from wakis.estimate import estimate, preflight
from wakis.field_monitors import DFTMonitor
from wakis.sources import Beam, ContinuousSource, CurrentSource, GaussianSource
from wakis.templates.meshing import graded_axis
from wakis.templates.shapes import Box, assign_materials

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

Z0_VAC = float(np.sqrt(mu_0 / epsilon_0))
ME_EV = float(m_e * c_light**2 / q_elem)  # Electron rest energy ~ 510998.95 eV
DEFAULT_MEM_LIMIT_MB = 1800.0

# Default Smith-Purcell parameters (Smith & Purcell, Phys. Rev. 92, 1069, 1953)
DEFAULT_SP_PERIOD_M = 1.67e-6  # 15,000 lines/inch ruled grating pitch (1.67 um)
DEFAULT_SP_DEPTH_M = 0.40e-6   # Groove depth [m]
DEFAULT_SP_DUTY = 0.50         # Tooth duty cycle w_tooth / d [-]
DEFAULT_SP_BETA = 0.80         # Relativistic velocity v/c (~340 keV, Smith & Purcell 1953)
DEFAULT_SP_IMPACT_M = 0.25e-6  # Beam-to-grating clearance h [m]
DEFAULT_SP_SIGMAZ_M = 0.20e-6  # RMS Gaussian bunch length [m]
DEFAULT_SP_Q_C = 1.0e-12       # Bunch charge [C]

# Default Dielectric Laser Accelerator parameters (Peralta et al., Nature 503, 91, 2013)
DEFAULT_DLA_LAM0_M = 0.80e-6       # Ti:sapphire drive laser wavelength (800 nm)
DEFAULT_DLA_PERIOD_M = 0.80e-6     # Synchronous grating period for 60 MeV electrons (800 nm)
DEFAULT_DLA_DEPTH_M = 0.68e-6      # Fused-silica tooth height [m] (~650-680 nm optimal)
DEFAULT_DLA_DUTY = 0.50            # Tooth duty cycle [-]
DEFAULT_DLA_GAP_HALF_M = 0.20e-6   # Half-aperture a for the 2a = 400 nm channel [m]
DEFAULT_N_SIO2 = 1.453             # Fused silica refractive index at 800 nm [-]
DEFAULT_EPS_SIO2 = DEFAULT_N_SIO2**2


# ---------------------------------------------------------------------------
# Analytical theory helpers: Smith-Purcell & Dielectric Laser Acceleration
# ---------------------------------------------------------------------------


def kinetic_energy_to_beta_gamma(energy_ev: float | np.ndarray) -> dict[str, np.ndarray]:
    """Convert electron kinetic energy [eV] to Lorentz factor gamma and normalized speed beta.

    Parameters
    ----------
    energy_ev : float or ndarray
        Electron kinetic energy E_k [eV].

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``gamma`` [-], ``beta`` [-], and ``beta_gamma`` [-].
    """
    ek = np.asarray(energy_ev, dtype=float)
    gamma = 1.0 + ek / ME_EV
    beta = np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (gamma**2)))
    return {
        "gamma": gamma,
        "beta": beta,
        "beta_gamma": beta * gamma,
    }


def smith_purcell_wavelength(
    theta_rad: float | np.ndarray,
    beta: float,
    period_m: float = DEFAULT_SP_PERIOD_M,
    order: int = 1,
) -> dict[str, np.ndarray]:
    """Compute the analytical Smith-Purcell dispersion relation and phase advance.

    A charge moving at velocity v = beta * c along +z above a grating of period d
    excites Floquet-Bloch space harmonics with axial wavenumbers
        k_{z,n}(omega) = omega / (beta * c) - 2 * pi * n / d.
    Matching k_{z,n} = (omega / c) * cos(theta) for radiation into vacuum at polar
    observation angle theta in (0, pi) measured from +z yields (Smith & Purcell,
    Phys. Rev. 92, 1069, 1953):
        lambda_n(theta) = (d / n) * (1 / beta - cos(theta)).

    Parameters
    ----------
    theta_rad : float or ndarray
        Polar observation angle(s) measured from the +z electron velocity axis [rad].
    beta : float
        Normalized electron velocity v / c in (0, 1) [-].
    period_m : float, optional
        Grating period d along z [m].
    order : int, optional
        Positive diffraction order n >= 1 [-].

    Returns
    -------
    out : dict of ndarray
        Dictionary containing:
        - ``theta_rad`` and ``theta_deg`` : observation angle [rad, deg]
        - ``wavelength_m`` : emitted vacuum wavelength lambda_n(theta) [m]
        - ``frequency_hz`` : emitted frequency f_n(theta) = c / lambda_n(theta) [Hz]
        - ``kz_rad_m`` : radiating harmonic axial wavenumber k_0 * cos(theta) [rad/m]
        - ``kx_rad_m`` : radiating harmonic transverse wavenumber k_0 * sin(theta) [rad/m]
        - ``kz0_rad_m`` : evanescent source wavenumber omega / (beta * c) [rad/m]
        - ``unit_cell_phase_advance_rad`` : 2 * pi * n [rad]
    """
    if not (0.0 < float(beta) <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")
    if int(order) < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    th = np.asarray(theta_rad, dtype=float)
    d = float(period_m)
    n = int(order)
    b = float(beta)
    lam = (d / n) * (1.0 / b - np.cos(th))
    freq = c_light / lam
    k0 = 2.0 * np.pi / lam
    kz = k0 * np.cos(th)
    kx = k0 * np.sin(th)
    kz0 = k0 / b
    return {
        "theta_rad": th,
        "theta_deg": np.degrees(th),
        "wavelength_m": lam,
        "frequency_hz": freq,
        "kz_rad_m": kz,
        "kx_rad_m": kx,
        "kz0_rad_m": kz0,
        "unit_cell_phase_advance_rad": np.full_like(th, 2.0 * np.pi * n),
    }


def smith_purcell_evanescent_decay(
    h_m: float | np.ndarray,
    beta: float,
    wavelength_m: float,
) -> dict[str, np.ndarray]:
    """Compute the evanescent impact-parameter coupling and decay constants.

    In vacuum above the grating, the comoving harmonic at frequency omega = 2*pi*c/lambda
    has axial wavenumber k_{z,0} = omega / (beta * c) and transverse decay constant
        kappa_0 = sqrt(k_{z,0}^2 - k_0^2) = 2 * pi / (beta * gamma * lambda).
    The field amplitude at the grating surface x = 0 scales as exp(-kappa_0 * h), so
    the radiated Smith-Purcell power scales as
        P_SP(h) / P_SP(0) = exp(-2 * kappa_0 * h) = exp(-4 * pi * h / (beta * gamma * lambda)).

    Parameters
    ----------
    h_m : float or ndarray
        Impact parameter (beam-to-grating surface clearance) h [m].
    beta : float
        Normalized velocity v / c in (0, 1) [-].
    wavelength_m : float
        Emitted vacuum wavelength lambda [m].

    Returns
    -------
    out : dict of ndarray
        Dictionary with:
        - ``gamma`` : Lorentz factor (1 - beta^2)^(-1/2) [-]
        - ``kappa_field_1_m`` : field amplitude decay constant 2*pi/(beta*gamma*lambda) [1/m]
        - ``alpha_power_1_m`` : power decay constant 4*pi/(beta*gamma*lambda) [1/m]
        - ``decay_length_m`` : 1/e power decay length beta*gamma*lambda/(4*pi) [m]
        - ``field_factor`` : exp(-kappa_field * h) [-]
        - ``power_factor`` : exp(-alpha_power * h) [-]
    """
    b = float(beta)
    if not (0.0 < b < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    gamma = 1.0 / np.sqrt(1.0 - b**2)
    lam = float(wavelength_m)
    h = np.asarray(h_m, dtype=float)
    kappa = 2.0 * np.pi / (b * gamma * lam)
    alpha = 2.0 * kappa
    return {
        "gamma": np.array(gamma),
        "kappa_field_1_m": np.array(kappa),
        "alpha_power_1_m": np.array(alpha),
        "decay_length_m": np.array(1.0 / alpha),
        "field_factor": np.exp(-kappa * h),
        "power_factor": np.exp(-alpha * h),
    }


def dla_synchronous_period(
    beta: float,
    wavelength_m: float = DEFAULT_DLA_LAM0_M,
    order: int = 1,
    period_m: float | np.ndarray | None = None,
    n_periods: int = 8,
) -> dict[str, np.ndarray]:
    """Compute the DLA synchronous grating period and multi-period dephasing envelope.

    Under normal-incidence laser illumination of vacuum wavelength lambda_0, the m-th
    diffracted space harmonic has axial wavenumber k_{z,m} = 2*pi*m / lambda_p and phase
    velocity v_{p,m} = (lambda_p / (m * lambda_0)) * c. Synchronism with an electron of
    velocity v = beta * c requires
        lambda_{p,sync} = m * beta * lambda_0.
    For a detuned period lambda_p, the phase slip per period is
        Delta_psi = 2 * pi * m * (lambda_p / lambda_{p,sync} - 1),
    yielding the N_p-period dephasing factor |sin(N_p * Delta_psi / 2) / (N_p * sin(Delta_psi / 2))|.

    Parameters
    ----------
    beta : float
        Normalized electron speed v / c in (0, 1] [-].
    wavelength_m : float, optional
        Drive laser vacuum wavelength lambda_0 [m].
    order : int, optional
        Diffraction space-harmonic order m >= 1 [-].
    period_m : float, ndarray, or None, optional
        Grating period(s) lambda_p [m] at which to evaluate the phase slip and
        N_p-period dephasing factor. Defaults to ``lambda_{p,sync}``.
    n_periods : int, optional
        Number of grating periods N_p in the interaction region [-].

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``period_sync_m``, ``phase_advance_rad``, ``phase_slip_rad``,
        and ``dephasing_factor``.
    """
    b = float(beta)
    lam0 = float(wavelength_m)
    m = int(order)
    lam_sync = m * b * lam0
    p_arr = np.asarray(lam_sync if period_m is None else period_m, dtype=float)
    phase_adv = 2.0 * np.pi * m * (p_arr / (b * lam0))
    slip = phase_adv - 2.0 * np.pi * m
    half_s = 0.5 * slip
    denom = float(n_periods) * np.sin(half_s)
    deph = np.where(
        np.abs(np.sin(half_s)) < 1e-12,
        1.0,
        np.abs(np.sin(float(n_periods) * half_s) / np.where(np.abs(denom) < 1e-30, 1.0, denom)),
    )
    return {
        "period_sync_m": np.array(lam_sync),
        "phase_advance_rad": phase_adv,
        "phase_slip_rad": slip,
        "dephasing_factor": deph,
    }


def dla_phase_grating_model(
    h_g_m: float | np.ndarray,
    duty_cycle: float | np.ndarray = DEFAULT_DLA_DUTY,
    wavelength_m: float = DEFAULT_DLA_LAM0_M,
    n_refr: float = DEFAULT_N_SIO2,
    gap_half_m: float = DEFAULT_DLA_GAP_HALF_M,
    beta: float = 0.999964,
) -> dict[str, np.ndarray]:
    """Analytical scalar phase-grating (Raman-Nath) model for a Dielectric Laser Accelerator.

    A plane wave of wavelength lambda_0 traversing rectangular dielectric teeth of refractive
    index n, height h_g, and duty cycle DC = w_tooth / lambda_p accumulates a differential
    phase shift Delta_phi = 2 * pi * (n - 1) * h_g / lambda_0 relative to the vacuum groove.
    The first Fourier harmonic amplitude is proportional to
        |c_1(h_g, DC)| = (2 / pi) * sin(pi * DC) * |sin(pi * (n - 1) * h_g / lambda_0)|,
    which reaches its first maximum at the pi-phase-shift depth
        h_{g,pi} = lambda_0 / (2 * (n - 1))
    and at 50% duty cycle (DC = 0.50), while vanishing for a flat surface (h_g = 0, DC = 0,
    or DC = 1).

    Parameters
    ----------
    h_g_m : float or ndarray
        Grating tooth depth h_g [m].
    duty_cycle : float or ndarray, optional
        Tooth duty cycle w_tooth / lambda_p in [0, 1] [-].
    wavelength_m : float, optional
        Laser vacuum wavelength lambda_0 [m].
    n_refr : float, optional
        Dielectric refractive index n [-].
    gap_half_m : float, optional
        Half-gap a of the vacuum channel [m].
    beta : float, optional
        Normalized electron velocity v / c [-].

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``h_g_pi_m``, ``delta_phi_rad``, ``c1_amplitude``, and
        ``channel_cosh_factor``.
    """
    hg = np.asarray(h_g_m, dtype=float)
    dc = np.asarray(duty_cycle, dtype=float)
    lam0 = float(wavelength_m)
    n = float(n_refr)
    h_pi = lam0 / (2.0 * (n - 1.0))
    dphi = 2.0 * np.pi * (n - 1.0) * hg / lam0
    c1 = (2.0 / np.pi) * np.sin(np.pi * np.clip(dc, 0.0, 1.0)) * np.abs(np.sin(0.5 * dphi))
    b = min(float(beta), 0.999999)
    gamma = 1.0 / np.sqrt(1.0 - b**2)
    kappa = 2.0 * np.pi / (b * gamma * lam0)
    chan_factor = 1.0 / np.cosh(kappa * float(gap_half_m))
    return {
        "h_g_pi_m": np.array(h_pi),
        "delta_phi_rad": dphi,
        "c1_amplitude": c1,
        "channel_cosh_factor": np.array(chan_factor),
    }


def literature_benchmarks() -> dict[str, dict]:
    """Return verified parameters and quantitative benchmarks from the four reference papers.

    References
    ----------
    1. S. J. Smith and E. M. Purcell, Phys. Rev. 92, 1069 (1953), doi:10.1103/PhysRev.92.1069.
    2. Y. Yang et al., Nature 613, 42-47 (2023), doi:10.1038/s41586-022-05495-2 (arXiv:2110.03550).
    3. E. A. Peralta et al., Nature 503, 91-94 (2013), doi:10.1038/nature12664.
    4. N. V. Sapra et al., Science 367, 79-83 (2020), doi:10.1126/science.aay5734 (arXiv:1905.12822).
    """
    sp_340 = kinetic_energy_to_beta_gamma(340.0e3)
    sp_319 = kinetic_energy_to_beta_gamma(319.0e3)
    peralta_60m = kinetic_energy_to_beta_gamma(60.0e6)
    sapra_83k = kinetic_energy_to_beta_gamma(83.4e3)
    return {
        "smith_purcell_1953": {
            "doi": "10.1103/PhysRev.92.1069",
            "grating_period_um": 1.67,
            "beam_voltage_kv": [319.0, 340.0],
            "beta_range": [float(sp_319["beta"]), float(sp_340["beta"])],
            "theta_obs_deg": 20.0,
            "lambda_20deg_nm": [
                float(1670.0 * (1.0 / sp_319["beta"] - np.cos(np.radians(20.0)))),
                float(1670.0 * (1.0 / sp_340["beta"] - np.cos(np.radians(20.0)))),
            ],
        },
        "yang_2023": {
            "doi": "10.1038/s41586-022-05495-2",
            "arxiv": "2110.03550",
            "lattice_period_nm": 430.0,
            "si_thickness_nm": 512.0,
            "sio2_box_nm": 1023.0,
            "beam_voltage_kv_range": [20.0, 40.0],
            "enhancement_factor_measured": 100.0,
        },
        "peralta_2013": {
            "doi": "10.1038/nature12664",
            "beam_energy_mev": 60.0,
            "beta": float(peralta_60m["beta"]),
            "gamma": float(peralta_60m["gamma"]),
            "laser_wavelength_nm": 800.0,
            "grating_period_nm": 800.0,
            "channel_gaps_nm": [400.0, 800.0],
            "interaction_length_um": 550.0,
            "pulse_duration_fwhm_ps": 1.24,
            "pulse_duration_err_ps": 0.12,
            "spot_rms_um": [30.0, 300.0],
            "max_pulse_energy_uj": 319.0,
            "damage_fluence_j_cm2": 0.85,
            "damage_fluence_err_j_cm2": 0.14,
            "max_gradient_400nm_mev_m": 309.8,
            "max_gradient_400nm_err_mev_m": 20.7,
            "max_gradient_800nm_mev_m": 152.0,
            "max_gradient_800nm_err_mev_m": 15.0,
            "gap_ratio_exp": 309.8 / 152.0,
            "gap_ratio_exp_err": float(
                (309.8 / 152.0) * np.hypot(20.7 / 309.8, 15.0 / 152.0)
            ),
            "headline_gradient_mev_m": 250.0,
        },
        "sapra_2020": {
            "doi": "10.1126/science.aay5734",
            "arxiv": "1905.12822",
            "beam_energy_kev": 83.4,
            "beta": float(sapra_83k["beta"]),
            "laser_wavelength_nm": 1984.0,
            "synchronous_period_nm": float(sapra_83k["beta"] * 1984.0),
            "interaction_length_um": 30.0,
            "max_energy_gain_kev": 0.915,
            "max_gradient_mev_m": 30.5,
            "max_gradient_err_mev_m": 0.7,
        },
    }


def peralta_2013_gradient_comparison(
    g_over_e0_400: float = 0.3415900475235852,
    g_over_e0_800: float = 0.1722993856756825,
    n_sio2: float = DEFAULT_N_SIO2,
) -> dict[str, float]:
    """Compute non-circular Peralta et al. (2013) gap-ratio and forward-predicted gradient benchmarks.

    Uses the laser parameters reported in Peralta et al., Nature 503, 91-94 (2013),
    pp. 92-94: Ti:sapphire drive laser at lambda_0 = 800 nm, FWHM pulse duration
    tau = 1.24 +/- 0.12 ps, rms spot size sigma_x * sigma_z = 30 um * 300 um,
    pulse energy up to U_max = 319 uJ (p. 93, col. 1), and measured grating
    damage-threshold fluence F_th = 0.85 +/- 0.14 J/cm^2 (pp. 92-94).

    Parameters
    ----------
    g_over_e0_400 : float, optional
        Simulated synchronous structure factor G / E_0 for the 400 nm channel gap [-].
    g_over_e0_800 : float, optional
        Simulated synchronous structure factor G / E_0 for the 800 nm channel gap [-].
    n_sio2 : float, optional
        Refractive index of fused silica at 800 nm [-].

    Returns
    -------
    comp : dict[str, float]
        Dictionary containing the simulated and experimental gap ratios
        G(400 nm) / G(800 nm) with plot-reading uncertainty, peak laser fields
        in vacuum and inside SiO2 [GV/m], and forward-predicted gradients
        G_pred = (G / E_0) * E_0 [MeV/m] vs measured 309.8 +/- 20.7 MeV/m.
    """
    tau_s = 1.24e-12
    tau_err_s = 0.12e-12
    sigma_x_m = 30.0e-6
    sigma_z_m = 300.0e-6
    u_max_j = 319.0e-6
    f_th_j_m2 = 0.85e4
    f_th_err_j_m2 = 0.14e4

    # Peak on-axis fluence F = U / (2 * pi * sigma_x * sigma_z) at U_max = 319 uJ (p. 93)
    f_319_j_m2 = u_max_j / (2.0 * np.pi * sigma_x_m * sigma_z_m)
    gauss_t_factor = float(np.sqrt(4.0 * np.log(2.0) / np.pi))

    # Peak intensities I_0 = sqrt(4 ln 2 / pi) * F / tau [W/m^2]
    i0_319_w_m2 = gauss_t_factor * f_319_j_m2 / tau_s
    i0_319_rel_err = tau_err_s / tau_s
    i0_th_w_m2 = gauss_t_factor * f_th_j_m2 / tau_s
    i0_th_rel_err = float(np.hypot(f_th_err_j_m2 / f_th_j_m2, tau_err_s / tau_s))

    # Peak vacuum electric field E_0,vac = sqrt(2 * Z_0 * I_0) [GV/m]
    e0_vac_319_gv_m = float(np.sqrt(2.0 * Z0_VAC * i0_319_w_m2) * 1e-9)
    e0_vac_319_err_gv_m = 0.5 * i0_319_rel_err * e0_vac_319_gv_m
    e0_vac_th_gv_m = float(np.sqrt(2.0 * Z0_VAC * i0_th_w_m2) * 1e-9)
    e0_vac_th_err_gv_m = 0.5 * i0_th_rel_err * e0_vac_th_gv_m

    # Normal-incidence Fresnel transmission into the fused-silica substrate t = 2 / (1 + n)
    t_fresnel = 2.0 / (1.0 + float(n_sio2))
    e0_sio2_319_gv_m = t_fresnel * e0_vac_319_gv_m
    e0_sio2_319_err_gv_m = t_fresnel * e0_vac_319_err_gv_m
    e0_sio2_th_gv_m = t_fresnel * e0_vac_th_gv_m
    e0_sio2_th_err_gv_m = t_fresnel * e0_vac_th_err_gv_m

    # Forward predictions G_pred = (G / E_0) * E_0 [MeV/m]
    g400 = float(g_over_e0_400)
    g800 = float(g_over_e0_800)
    g_pred_400_sio2_319 = g400 * e0_sio2_319_gv_m * 1.0e3
    g_pred_400_sio2_319_err = g400 * e0_sio2_319_err_gv_m * 1.0e3
    g_pred_400_vac_319 = g400 * e0_vac_319_gv_m * 1.0e3
    g_pred_400_vac_319_err = g400 * e0_vac_319_err_gv_m * 1.0e3
    g_pred_400_sio2_th = g400 * e0_sio2_th_gv_m * 1.0e3
    g_pred_400_sio2_th_err = g400 * e0_sio2_th_err_gv_m * 1.0e3
    g_pred_400_vac_th = g400 * e0_vac_th_gv_m * 1.0e3
    g_pred_400_vac_th_err = g400 * e0_vac_th_err_gv_m * 1.0e3

    g_pred_800_sio2_319 = g800 * e0_sio2_319_gv_m * 1.0e3
    g_pred_800_sio2_319_err = g800 * e0_sio2_319_err_gv_m * 1.0e3

    # Headline non-circular check: channel-gap ratio G(400 nm) / G(800 nm)
    gap_ratio_sim = g400 / g800
    gap_ratio_sim_rounded = round(g400, 3) / round(g800, 3)
    g_exp_400 = 309.8
    g_exp_400_err = 20.7
    g_exp_800 = 152.0
    g_exp_800_err = 15.0
    gap_ratio_exp = g_exp_400 / g_exp_800
    gap_ratio_exp_err = float(
        gap_ratio_exp * np.hypot(g_exp_400_err / g_exp_400, g_exp_800_err / g_exp_800)
    )
    gap_ratio_rel_diff = abs(gap_ratio_sim - gap_ratio_exp) / gap_ratio_exp

    return {
        "g_over_e0_400": g400,
        "g_over_e0_800": g800,
        "gap_ratio_sim": float(gap_ratio_sim),
        "gap_ratio_sim_rounded": float(gap_ratio_sim_rounded),
        "gap_ratio_exp": float(gap_ratio_exp),
        "gap_ratio_exp_err": float(gap_ratio_exp_err),
        "gap_ratio_rel_diff": float(gap_ratio_rel_diff),
        "fluence_319uj_j_cm2": float(f_319_j_m2 * 1e-4),
        "fluence_damage_j_cm2": 0.85,
        "fluence_damage_err_j_cm2": 0.14,
        "fresnel_t_sio2": float(t_fresnel),
        "e0_vac_319uj_gv_m": float(e0_vac_319_gv_m),
        "e0_vac_319uj_err_gv_m": float(e0_vac_319_err_gv_m),
        "e0_sio2_319uj_gv_m": float(e0_sio2_319_gv_m),
        "e0_sio2_319uj_err_gv_m": float(e0_sio2_319_err_gv_m),
        "e0_vac_damage_gv_m": float(e0_vac_th_gv_m),
        "e0_vac_damage_err_gv_m": float(e0_vac_th_err_gv_m),
        "e0_sio2_damage_gv_m": float(e0_sio2_th_gv_m),
        "e0_sio2_damage_err_gv_m": float(e0_sio2_th_err_gv_m),
        "g_pred_400_sio2_319uj_mev_m": float(g_pred_400_sio2_319),
        "g_pred_400_sio2_319uj_err_mev_m": float(g_pred_400_sio2_319_err),
        "g_pred_400_vac_319uj_mev_m": float(g_pred_400_vac_319),
        "g_pred_400_vac_319uj_err_mev_m": float(g_pred_400_vac_319_err),
        "g_pred_400_sio2_damage_mev_m": float(g_pred_400_sio2_th),
        "g_pred_400_sio2_damage_err_mev_m": float(g_pred_400_sio2_th_err),
        "g_pred_400_vac_damage_mev_m": float(g_pred_400_vac_th),
        "g_pred_400_vac_damage_err_mev_m": float(g_pred_400_vac_th_err),
        "g_pred_800_sio2_319uj_mev_m": float(g_pred_800_sio2_319),
        "g_pred_800_sio2_319uj_err_mev_m": float(g_pred_800_sio2_319_err),
        "g_exp_400_mev_m": g_exp_400,
        "g_exp_400_err_mev_m": g_exp_400_err,
        "g_exp_800_mev_m": g_exp_800,
        "g_exp_800_err_mev_m": g_exp_800_err,
    }


# ---------------------------------------------------------------------------
# Preflight and sub-bin peak extraction helpers
# ---------------------------------------------------------------------------


def preflight_solver(
    solver: SolverFIT3D,
    n_steps: int,
    mem_limit_mb: float = DEFAULT_MEM_LIMIT_MB,
) -> dict[str, float]:
    """Run ``wakis.estimate`` and ``preflight`` on a 3D solver before time-stepping."""
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


def _parabolic_peak(x_arr: np.ndarray, y_arr: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """Return quadratic sub-bin peak location and peak value of ``y_arr(x_arr)``."""
    if mask is not None and np.any(mask):
        idx_sub = np.flatnonzero(mask)
        k_local = int(np.argmax(y_arr[idx_sub]))
        k = int(idx_sub[k_local])
    else:
        k = int(np.argmax(y_arr))
    if 0 < k < y_arr.size - 1:
        y_m1, y_0, y_p1 = float(y_arr[k - 1]), float(y_arr[k]), float(y_arr[k + 1])
        denom = y_m1 - 2.0 * y_0 + y_p1
        shift = 0.5 * (y_m1 - y_p1) / denom if abs(denom) > 1e-30 else 0.0
        shift = float(np.clip(shift, -0.5, 0.5))
        dx_loc = float(x_arr[k + 1] - x_arr[k]) if shift >= 0 else float(x_arr[k] - x_arr[k - 1])
        x_peak = float(x_arr[k]) + shift * dx_loc
    else:
        x_peak = float(x_arr[k])
    return x_peak, float(y_arr[k])


# ---------------------------------------------------------------------------
# Part 1: Smith-Purcell FIT simulation helpers
# ---------------------------------------------------------------------------


def build_smith_purcell_shapes(
    material: str | float,
    period_m: float,
    n_periods: int,
    depth_m: float,
    duty_cycle: float,
    xmin_m: float,
    zmin_m: float,
    zmax_m: float,
    dy_m: float,
) -> list[Box]:
    """Build flush-grooved grating shapes in ``[-depth_m, 0]`` on a substrate ``[xmin_m, -depth_m]``.

    Placing flat entry/exit shoulders flush at ``x = 0`` outside the grooved grating
    section ``[-0.5*n_periods*period_m, +0.5*n_periods*period_m]`` ensures that
    setting ``depth_m = 0`` (or ``duty_cycle = 1``) produces a smooth flat plane at
    ``x = 0`` with identical entrance and exit boundaries.
    """
    shapes: list[Box] = []
    if material == "vacuum":
        return shapes
    depth = max(0.0, float(depth_m))
    duty = float(np.clip(duty_cycle, 0.0, 1.0))
    x_base_top = -depth if (depth > 0.0 and duty < 1.0) else 0.0
    if x_base_top > xmin_m:
        shapes.append(
            Box(
                center=(0.5 * (xmin_m + x_base_top), 0.0, 0.5 * (zmin_m + zmax_m)),
                size=(x_base_top - xmin_m, dy_m, zmax_m - zmin_m),
                material=material,
            )
        )
    if depth > 0.0 and 0.0 < duty < 1.0:
        z_g_min = -0.5 * float(n_periods) * float(period_m)
        z_g_max = +0.5 * float(n_periods) * float(period_m)
        if z_g_min > zmin_m:
            shapes.append(
                Box(
                    center=(-0.5 * depth, 0.0, 0.5 * (zmin_m + z_g_min)),
                    size=(depth, dy_m, z_g_min - zmin_m),
                    material=material,
                )
            )
        if zmax_m > z_g_max:
            shapes.append(
                Box(
                    center=(-0.5 * depth, 0.0, 0.5 * (z_g_max + zmax_m)),
                    size=(depth, dy_m, zmax_m - z_g_max),
                    material=material,
                )
            )
        for m in range(int(n_periods)):
            zc = z_g_min + (m + 0.5) * float(period_m)
            shapes.append(
                Box(
                    center=(-0.5 * depth, 0.0, zc),
                    size=(depth, dy_m, duty * float(period_m)),
                    material=material,
                )
            )
    return shapes


def run_single_smith_purcell_case(
    material: str | float = "pec",
    beta: float = DEFAULT_SP_BETA,
    period_m: float = DEFAULT_SP_PERIOD_M,
    n_periods: int = 10,
    depth_m: float = DEFAULT_SP_DEPTH_M,
    duty_cycle: float = DEFAULT_SP_DUTY,
    impact_m: float = DEFAULT_SP_IMPACT_M,
    sigmaz_m: float = DEFAULT_SP_SIGMAZ_M,
    q_c: float = DEFAULT_SP_Q_C,
    dx_m: float = 0.05e-6,
    dz_m: float = 0.05e-6,
    wavelengths_m: np.ndarray | None = None,
    thetas_deg: np.ndarray | None = None,
    subtract_flat: bool = True,
    record_frames: int = 0,
    mem_limit_mb: float = DEFAULT_MEM_LIMIT_MB,
) -> dict:
    """Run a 3D FIT simulation (extruded 2D periodic slab Ny=1) of Smith-Purcell radiation.

    Parameters
    ----------
    material : {'pec', 'vacuum'} or float, optional
        Grating material: ``'pec'`` for a metallic grating, relative permittivity
        ``eps_r`` (e.g. ``2.111`` for fused silica) for a dielectric grating, or
        ``'vacuum'`` for free space.
    beta : float, optional
        Normalized electron velocity v / c [-].
    period_m : float, optional
        Grating period d [m].
    n_periods : int, optional
        Number of grating periods N_g [-].
    depth_m : float, optional
        Grating groove depth h_g [m]. Set ``depth_m = 0.0`` for the flat-surface mutation.
    duty_cycle : float, optional
        Tooth duty cycle w_tooth / d [-].
    impact_m : float, optional
        Electron impact parameter (clearance above x = 0) h [m].
    sigmaz_m : float, optional
        RMS Gaussian bunch length [m].
    q_c : float, optional
        Bunch charge [C].
    dx_m, dz_m : float, optional
        Transverse and longitudinal cell sizes [m].
    wavelengths_m : ndarray or None, optional
        Monitored vacuum wavelengths [m].
    thetas_deg : ndarray or None, optional
        Observation angles [deg] from +z at which to extract peak wavelengths.
    subtract_flat : bool, optional
        If True and ``depth_m > 0``, subtract the reference field of the corresponding
        flat surface (``depth_m = 0``) to isolate the scattered grating space harmonics.
        If ``depth_m == 0``, subtracts the homogeneous vacuum bunch self-field so the
        flat-surface radiation test is never a tautological self-subtraction.
    record_frames : int, optional
        Number of time-domain meridian ``Ez(x, z)`` frames to record (0 disables).
    mem_limit_mb : float, optional
        Memory ceiling for preflight [MB].

    Returns
    -------
    res : dict
        Simulation dictionary with angular-spectral map, extracted wavelengths vs angle,
        radiated energy metric, snapshot slices, and optional animation frames.
    """
    xmin_m, xmax_m = -1.0e-6, 3.4e-6
    m_half = max(4, int(round(0.5 * float(period_m) / float(dz_m))))
    dz_actual = float(period_m) / (2.0 * m_half)
    n_pad = int(round(3.0e-6 / dz_actual))
    nz = int(n_periods) * (2 * m_half) + 2 * n_pad
    half_lz = 0.5 * nz * dz_actual
    zmin_m, zmax_m = -half_lz, +half_lz
    nx = int(round((xmax_m - xmin_m) / float(dx_m)))
    dy_m = float(dx_m)

    if wavelengths_m is None:
        wavelengths_m = np.linspace(0.65e-6, 4.10e-6, 116)
    else:
        wavelengths_m = np.asarray(wavelengths_m, dtype=float)
    freqs_hz = c_light / wavelengths_m

    if thetas_deg is None:
        thetas_deg = np.array([40.0, 55.0, 70.0, 85.0, 90.0, 105.0, 120.0, 135.0, 150.0])
    else:
        thetas_deg = np.asarray(thetas_deg, dtype=float)

    x_mon_m = 2.25e-6
    z_mon_span = float(n_periods) * float(period_m) + 2.0e-6

    def _execute_single(mat_val: str | float, dep_val: float, n_fr: int) -> dict:
        grid = GridFIT3D(
            xmin_m,
            xmax_m,
            -0.5 * dy_m,
            0.5 * dy_m,
            zmin_m,
            zmax_m,
            nx,
            1,
            nz,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            bc_low=["cpml", "periodic", "cpml"],
            bc_high=["cpml", "periodic", "cpml"],
            verbose=0,
        )
        solver.source_type = "direct"
        shapes = build_smith_purcell_shapes(
            material=mat_val,
            period_m=period_m,
            n_periods=n_periods,
            depth_m=dep_val,
            duty_cycle=duty_cycle,
            xmin_m=xmin_m,
            zmin_m=zmin_m,
            zmax_m=zmax_m,
            dy_m=dy_m,
        )
        if shapes:
            assign_materials(solver, shapes, frequency=float(freqs_hz[len(freqs_hz) // 2]))

        v = float(beta) * c_light
        ti = 4.0 * float(sigmaz_m) / v
        beam = Beam(
            q=float(q_c),
            sigmaz=float(sigmaz_m),
            beta=float(beta),
            xsource=float(impact_m),
            ysource=0.0,
            ti=ti,
        )
        mon = DFTMonitor(
            freqs_hz,
            components=("Ez", "Hy"),
            center=(x_mon_m, 0.0, 0.0),
            size=(0.0, 0.0, z_mon_span),
        )
        z0_min = float(solver.z.min())
        t_snap = ti + (0.18 * float(n_periods) * float(period_m) - z0_min) / v
        t_end = ti + (zmax_m - zmin_m + 5.0 * float(sigmaz_m) + 1.2 * (xmax_m - xmin_m)) / v
        nt = int(np.ceil(t_end / solver.dt))
        est = preflight_solver(solver, n_steps=nt, mem_limit_mb=mem_limit_mb)

        step_snap = int(round(t_snap / solver.dt))
        if n_fr > 0:
            t_a0 = ti + (-0.42 * float(n_periods) * float(period_m) - z0_min) / v
            t_a1 = ti + (+0.45 * float(n_periods) * float(period_m) - z0_min) / v
            frame_steps = set(
                np.unique(
                    np.clip(np.round(np.linspace(t_a0, t_a1, n_fr) / solver.dt).astype(int), 0, nt - 1)
                ).tolist()
            )
        else:
            frame_steps = set()

        ix_beam = int(np.argmin(np.abs(solver.grid.x[: solver.Nx] - float(impact_m))))
        z_arr = solver.z.copy()
        dz_grid = float(z_arr[1] - z_arr[0])
        z_int_mask = np.abs(z_arr) <= 0.5 * float(n_periods) * float(period_m)

        ez_snap = None
        hy_snap = None
        zb_snap = 0.0
        frames_ez = []
        frames_zb = []
        wake_work_j = 0.0

        t0_w = time.perf_counter()
        for step in range(nt):
            t = step * solver.dt
            beam.update(solver, t)
            solver.one_step()
            mon.update(solver, t)

            # Accumulate instantaneous work done by Ez on the bunch current inside grating window
            ez_line = solver.E[ix_beam, 0, z_int_mask, "z"]
            jz_line = solver.J[ix_beam, 0, z_int_mask, "z"]
            wake_work_j += float(
                np.sum(ez_line * jz_line)
                * solver.tdx[ix_beam]
                * solver.tdy[0]
                * dz_grid
                * solver.dt
            )

            zb_now = z0_min + v * (t + 0.5 * solver.dt - ti)
            if step == step_snap or (ez_snap is None and step == nt - 1):
                ez_snap = solver.E[:, 0, :, "z"].copy()
                hy_snap = solver.H[:, 0, :, "y"].copy()
                zb_snap = float(zb_now)
            if step in frame_steps:
                frames_ez.append(solver.E[:, 0, :, "z"].copy())
                frames_zb.append(float(zb_now))

        wall_s = time.perf_counter() - t0_w
        ez_f, coords = mon.get_field("Ez")
        hy_f, _ = mon.get_field("Hy")
        return {
            "Ez_mon": ez_f[:, 0, 0, :],
            "Hy_mon": hy_f[:, 0, 0, :],
            "z_mon_m": coords[2],
            "x_m": solver.grid.x[: solver.Nx].copy(),
            "z_m": z_arr,
            "Ez_xz": ez_snap,
            "Hy_xz": hy_snap,
            "z_bunch_snap_m": zb_snap,
            "frames_Ez": np.asarray(frames_ez) if n_fr > 0 else None,
            "frames_zb_m": np.asarray(frames_zb) if n_fr > 0 else None,
            "wake_loss_J": -wake_work_j,
            "nt": nt,
            "dt_s": float(solver.dt),
            "wall_s": wall_s,
            "memory_mb": float(est["memory_mb"]),
        }

    main_run = _execute_single(material, depth_m, record_frames)
    if subtract_flat:
        if float(depth_m) > 0.0 and 0.0 < float(duty_cycle) < 1.0 and material != "vacuum":
            ref_run = _execute_single(material, 0.0, record_frames)
        else:
            # For a flat surface (depth_m == 0), subtract only the homogeneous vacuum self-field
            ref_run = _execute_single("vacuum", 0.0, record_frames)
        d_hy = main_run["Hy_mon"] - ref_run["Hy_mon"]
        d_ez = main_run["Ez_mon"] - ref_run["Ez_mon"]
        ez_scat_xz = main_run["Ez_xz"] - ref_run["Ez_xz"]
        frames_scat = (
            main_run["frames_Ez"] - ref_run["frames_Ez"]
            if record_frames > 0 and main_run["frames_Ez"] is not None
            else None
        )
        wake_rad_loss_j = main_run["wake_loss_J"] - ref_run["wake_loss_J"]
    else:
        d_hy = main_run["Hy_mon"]
        d_ez = main_run["Ez_mon"]
        ez_scat_xz = main_run["Ez_xz"]
        frames_scat = main_run["frames_Ez"]
        wake_rad_loss_j = main_run["wake_loss_J"]

    # Compute angular-spectral intensity map U(theta, lambda) via cylindrical Stratton-Chu / Fourier projection
    z_mon = main_run["z_mon_m"]
    dz_mon = float(z_mon[1] - z_mon[0])
    win = np.hanning(z_mon.size)[None, :]
    k0 = 2.0 * np.pi * freqs_hz / c_light
    gamma = 1.0 / np.sqrt(1.0 - float(beta) ** 2)
    # Mild surface-coupling weight exp(-2*pi*h/(beta*gamma*lambda)) for visual map balance
    norm_weight = np.maximum(
        np.exp(-2.0 * np.pi * float(impact_m) / (float(beta) * gamma * wavelengths_m)),
        1e-3,
    )

    theta_grid_deg = np.linspace(30.0, 155.0, 101)
    theta_grid_rad = np.radians(theta_grid_deg)
    angular_map_raw = np.zeros((theta_grid_deg.size, wavelengths_m.size), dtype=float)
    angular_map_deweighted = np.zeros_like(angular_map_raw)

    for ith, th_r in enumerate(theta_grid_rad):
        kz_t = k0 * np.cos(th_r)
        proj = np.sum(d_hy * win * np.exp(+1j * kz_t[:, None] * z_mon[None, :]), axis=1) * dz_mon
        raw_p = np.abs(proj) ** 2
        angular_map_raw[ith, :] = raw_p
        angular_map_deweighted[ith, :] = raw_p / norm_weight

    # Extract peak wavelength lambda_sim(theta) at requested angles
    lam_sim_m = np.zeros(thetas_deg.size, dtype=float)
    lam_theory_m = np.zeros(thetas_deg.size, dtype=float)
    peak_power_theta = np.zeros(thetas_deg.size, dtype=float)

    for i, th_d in enumerate(thetas_deg):
        th_r = float(np.radians(th_d))
        lam_th = float(period_m) * (1.0 / float(beta) - np.cos(th_r))
        lam_theory_m[i] = lam_th
        kz_t = k0 * np.cos(th_r)
        proj = np.sum(d_hy * win * np.exp(+1j * kz_t[:, None] * z_mon[None, :]), axis=1) * dz_mon
        raw_p = np.abs(proj) ** 2
        spec_de = raw_p / norm_weight
        # Search within first-order (n=1) window to avoid n=2 harmonic branch at large angles
        n1_mask = (wavelengths_m >= 0.72 * lam_th) & (wavelengths_m <= 1.28 * lam_th)
        lam_pk, _ = _parabolic_peak(wavelengths_m, spec_de, mask=n1_mask)
        lam_sim_m[i] = lam_pk
        _, pk_raw = _parabolic_peak(wavelengths_m, raw_p, mask=n1_mask)
        peak_power_theta[i] = pk_raw

    rel_errs = np.abs(lam_sim_m - lam_theory_m) / lam_theory_m
    # Broadside (theta = 90 deg) intensity at exact wavelength lambda_1(90 deg) = d / beta for impact-parameter decay
    lam90 = float(period_m) / float(beta)
    proj90 = np.sum(d_hy * win, axis=1) * dz_mon
    raw90 = np.abs(proj90) ** 2
    intensity_90 = float(np.interp(lam90, wavelengths_m, raw90))

    return {
        "material": str(material),
        "beta": float(beta),
        "gamma": float(gamma),
        "period_m": float(period_m),
        "n_periods": int(n_periods),
        "depth_m": float(depth_m),
        "duty_cycle": float(duty_cycle),
        "impact_m": float(impact_m),
        "dx_m": float(dx_m),
        "dz_m": float(dz_m),
        "nx": int(nx),
        "nz": int(nz),
        "nt": int(main_run["nt"]),
        "wall_s": float(main_run["wall_s"]),
        "memory_mb": float(main_run["memory_mb"]),
        "wavelengths_m": wavelengths_m,
        "freqs_hz": freqs_hz,
        "theta_grid_deg": theta_grid_deg,
        "angular_map_raw": angular_map_raw,
        "angular_map_deweighted": angular_map_deweighted,
        "thetas_deg": thetas_deg,
        "lambda_sim_m": lam_sim_m,
        "lambda_theory_m": lam_theory_m,
        "lambda_rel_errs": rel_errs,
        "lambda_mean_rel_err": float(np.mean(rel_errs)),
        "lambda_max_rel_err": float(np.max(rel_errs)),
        "peak_power_theta": peak_power_theta,
        "broadside_intensity": float(intensity_90),
        "total_radiated_metric": float(np.mean(peak_power_theta[:6])),
        "wake_rad_loss_J": float(wake_rad_loss_j),
        "x_m": main_run["x_m"],
        "z_m": main_run["z_m"],
        "Ez_tot_xz": main_run["Ez_xz"],
        "Ez_scat_xz": ez_scat_xz,
        "z_bunch_snap_m": float(main_run["z_bunch_snap_m"]),
        "frames_Ez_tot": main_run["frames_Ez"],
        "frames_Ez_scat": frames_scat,
        "frames_zb_m": main_run["frames_zb_m"],
    }


# ---------------------------------------------------------------------------
# Part 2: Dielectric Laser Accelerator (DLA) FIT simulation helpers
# ---------------------------------------------------------------------------


def run_single_dla_unit_cell(
    period_m: float = DEFAULT_DLA_PERIOD_M,
    depth_m: float = DEFAULT_DLA_DEPTH_M,
    duty_cycle: float = DEFAULT_DLA_DUTY,
    wavelength_m: float = DEFAULT_DLA_LAM0_M,
    n_refr: float = DEFAULT_N_SIO2,
    gap_half_m: float = DEFAULT_DLA_GAP_HALF_M,
    beta: float = 0.999964,
    n_periods_eval: int = 8,
    dual_grating: bool = True,
    dx_m: float = 0.025e-6,
    record_2d_slice: bool = False,
    mem_limit_mb: float = DEFAULT_MEM_LIMIT_MB,
) -> dict:
    """Run a 3D FIT unit-cell simulation of a fused-silica DLA grating under normal laser drive.

    Evaluates both the single-cell space-harmonic decomposition |E_{z,m}| / E_0 and
    the N_p-period wake-formalism trajectory line integral along z(t) = beta * c * t:
        G / E_0 = (1 / (N_p * lambda_p * E_0)) * max_{phi_0} |int_0^{N_p * lambda_p} E_z(x_b, z, t(z)) dz|.

    Parameters
    ----------
    period_m : float, optional
        Grating period lambda_p [m].
    depth_m : float, optional
        Grating tooth height h_g [m]. Set ``depth_m = 0.0`` for the flat-surface mutation.
    duty_cycle : float, optional
        Grating tooth duty cycle w_tooth / lambda_p in [0, 1] [-].
    wavelength_m : float, optional
        Drive laser vacuum wavelength lambda_0 [m].
    n_refr : float, optional
        Dielectric refractive index n [-].
    gap_half_m : float, optional
        Vacuum channel half-gap a [m] (channel width 2a = 400 nm for a = 200 nm).
    beta : float, optional
        Normalized electron velocity v / c in (0, 1] [-].
    n_periods_eval : int, optional
        Number of periods N_p used for the multi-period trajectory line integral [-].
    dual_grating : bool, optional
        If True, models the dual-grating vacuum channel of Peralta et al. (2013);
        if False, models a single-sided grating with beam at ``x = +gap_half_m``.
    dx_m : float, optional
        Target cell size [m].
    record_2d_slice : bool, optional
        If True, also records the complex 2D phasor slice ``Ez(x, z; f_0) / E_0``.
    mem_limit_mb : float, optional
        Memory ceiling for preflight [MB].

    Returns
    -------
    out : dict
        Dictionary with ``g_over_e0_cell`` (single-period factor), ``g_over_e0_multiperiod``
        (N_p-period wake integral), space harmonics ``harmonics_m`` and ``harmonics_amp``,
        and optional 2D phasor slice.
    """
    lam0 = float(wavelength_m)
    f0 = c_light / lam0
    omega0 = 2.0 * np.pi * f0
    eps_diel = float(n_refr) ** 2
    lam_p = float(period_m)

    nz = max(16, int(round(lam_p / float(dx_m))))
    dz_actual = lam_p / nz
    dy_m = float(dx_m)
    xmin_m, xmax_m = -2.2e-6, 2.2e-6
    nx = int(round((xmax_m - xmin_m) / float(dx_m)))

    grid = GridFIT3D(
        xmin_m,
        xmax_m,
        -0.5 * dy_m,
        0.5 * dy_m,
        -0.5 * lam_p,
        +0.5 * lam_p,
        nx,
        1,
        nz,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "periodic", "periodic"],
        bc_high=["cpml", "periodic", "periodic"],
        verbose=0,
    )

    depth = max(0.0, float(depth_m))
    duty = float(np.clip(duty_cycle, 0.0, 1.0))
    a_gap = float(gap_half_m)

    shapes: list[Box] = []
    # Lower silica wafer and grating teeth
    x_lo_bot = -(a_gap + depth)
    shapes.append(
        Box(
            center=(0.5 * (xmin_m + x_lo_bot), 0.0, 0.0),
            size=(abs(x_lo_bot - xmin_m), dy_m, lam_p),
            material=eps_diel,
        )
    )
    if depth > 0.0 and duty > 0.0:
        shapes.append(
            Box(
                center=(-a_gap - 0.5 * depth, 0.0, 0.0),
                size=(depth, dy_m, duty * lam_p),
                material=eps_diel,
            )
        )
    if dual_grating:
        x_hi_top = +(a_gap + depth)
        shapes.append(
            Box(
                center=(0.5 * (xmax_m + x_hi_top), 0.0, 0.0),
                size=(abs(xmax_m - x_hi_top), dy_m, lam_p),
                material=eps_diel,
            )
        )
        if depth > 0.0 and duty > 0.0:
            shapes.append(
                Box(
                    center=(+a_gap + 0.5 * depth, 0.0, 0.0),
                    size=(depth, dy_m, duty * lam_p),
                    material=eps_diel,
                )
            )
    assign_materials(solver, shapes, frequency=f0)

    gw = GaussianSource(frequency=f0, fwidth=0.18 * f0, cutoff=4.5)
    x_src = -1.45e-6
    src = CurrentSource(
        gw,
        component="z",
        center=(x_src, 0.0, 0.0),
        size=(0.0, dy_m, lam_p),
        amplitude=1.0,
    )

    x_beam = 0.0 if dual_grating else (+a_gap - float(gap_half_m) + 0.20e-6)
    mon_line = DFTMonitor(
        [f0],
        components=("Ez",),
        center=(x_beam, 0.0, 0.0),
        size=(0.0, 0.0, lam_p),
    )
    mon_2d = (
        DFTMonitor(
            [f0],
            components=("Ez",),
            center=(0.0, 0.0, 0.0),
            size=(2.4e-6, 0.0, lam_p),
        )
        if record_2d_slice
        else None
    )

    t_end = gw.end_time + 14.0 / f0
    nt = int(np.ceil(t_end / solver.dt))
    est = preflight_solver(solver, n_steps=nt, mem_limit_mb=mem_limit_mb)

    t0_w = time.perf_counter()
    for step in range(nt):
        t = step * solver.dt
        src.update(solver, t)
        solver.one_step()
        mon_line.update(solver, t)
        if mon_2d is not None:
            mon_2d.update(solver, t)
    wall_s = time.perf_counter() - t0_w

    S0 = gw.fourier_transform(f0)
    e0_ref = 0.5 * (Z0_VAC / float(n_refr)) * float(np.abs(S0))
    ez_f, coords_line = mon_line.get_field("Ez")
    ez_norm = ez_f[0, 0, 0, :nz] / e0_ref
    z_cell = coords_line[2][:nz]

    # Space-harmonic Fourier series coefficients E_{z,m} / E_0 for m in [-4..+4]
    m_orders = np.arange(-4, 5, dtype=int)
    harm_amps = np.zeros(m_orders.size, dtype=float)
    for idx_m, m_val in enumerate(m_orders):
        km = 2.0 * np.pi * m_val / lam_p
        cm = np.sum(ez_norm * np.exp(-1j * km * z_cell)) / z_cell.size
        harm_amps[idx_m] = float(np.abs(cm))

    # Wake-formalism slow-wave line integral along electron path z(t) = beta * c * t
    # Subtract the m=0 fast-wave mean (k_{z,0}=0, Lawson-Woodward zero net gain) to avoid boxcar edge leakage
    ez_slow = ez_norm - np.mean(ez_norm)
    kz_beam = omega0 / (float(beta) * c_light)
    c_plus = np.sum(ez_slow * np.exp(+1j * kz_beam * z_cell)) / z_cell.size
    c_minus = np.sum(ez_slow * np.exp(-1j * kz_beam * z_cell)) / z_cell.size
    g_cell = float(max(abs(c_plus), abs(c_minus)))

    # Multi-period integral over z in [0, N_p * lam_p) with phase advance per period
    dpsi_cell = kz_beam * lam_p
    m_seq = np.arange(int(n_periods_eval), dtype=float)
    arr_plus = np.sum(np.exp(+1j * m_seq * dpsi_cell)) / float(n_periods_eval)
    arr_minus = np.sum(np.exp(-1j * m_seq * dpsi_cell)) / float(n_periods_eval)
    g_multiperiod = float(max(abs(c_plus * arr_plus), abs(c_minus * arr_minus)))

    out = {
        "period_m": lam_p,
        "depth_m": depth,
        "duty_cycle": duty,
        "wavelength_m": lam0,
        "n_refr": float(n_refr),
        "gap_half_m": a_gap,
        "beta": float(beta),
        "n_periods_eval": int(n_periods_eval),
        "dx_m": float(dx_m),
        "dz_m": float(dz_actual),
        "nx": int(nx),
        "nz": int(nz),
        "nt": int(nt),
        "wall_s": float(wall_s),
        "memory_mb": float(est["memory_mb"]),
        "z_cell_m": z_cell,
        "Ez_cell_norm": ez_norm,
        "harmonics_m": m_orders,
        "harmonics_amp": harm_amps,
        "g_over_e0_cell": g_cell,
        "g_over_e0_multiperiod": g_multiperiod,
    }
    if mon_2d is not None:
        ez_2d, coords_2d = mon_2d.get_field("Ez")
        # Rotate global phase so synchronous crest at (x=0, z=0) is real and positive
        idx_cx = int(np.argmin(np.abs(coords_2d[0])))
        idx_cz = int(np.argmin(np.abs(coords_2d[2])))
        ref_ph = np.angle(ez_2d[0, idx_cx, 0, idx_cz])
        out["x_2d_m"] = coords_2d[0]
        out["z_2d_m"] = coords_2d[2]
        out["Ez_2d_norm"] = (ez_2d[0, :, 0, :] / e0_ref) * np.exp(-1j * ref_ph)
    return out


def run_dla_multiperiod_tracking(
    n_periods: int = 6,
    period_m: float = DEFAULT_DLA_PERIOD_M,
    depth_m: float = DEFAULT_DLA_DEPTH_M,
    duty_cycle: float = DEFAULT_DLA_DUTY,
    wavelength_m: float = DEFAULT_DLA_LAM0_M,
    n_refr: float = DEFAULT_N_SIO2,
    gap_half_m: float = DEFAULT_DLA_GAP_HALF_M,
    beta_sync: float = 1.0,
    beta_detuned: float = 0.80,
    dx_m: float = 0.025e-6,
    record_frames: int = 20,
    mem_limit_mb: float = DEFAULT_MEM_LIMIT_MB,
) -> dict:
    """Track test electrons step-by-step in time domain across an N_p-period DLA supercell.

    Parameters
    ----------
    n_periods : int, optional
        Number of grating periods N_p in the time-domain supercell [-].
    period_m : float, optional
        Grating period lambda_p [m].
    depth_m : float, optional
        Tooth depth h_g [m].
    duty_cycle : float, optional
        Tooth duty cycle [-].
    wavelength_m : float, optional
        Drive laser wavelength lambda_0 [m].
    n_refr : float, optional
        Refractive index n [-].
    gap_half_m : float, optional
        Vacuum channel half-gap a [m].
    beta_sync : float, optional
        Synchronized electron velocity v / c = lambda_p / lambda_0 [-].
    beta_detuned : float, optional
        Asynchronous electron velocity v / c != lambda_p / lambda_0 [-].
    dx_m : float, optional
        Cell size [m].
    record_frames : int, optional
        Number of 2D field frames to record during the electron transit [-].
    mem_limit_mb : float, optional
        Memory ceiling [MB].

    Returns
    -------
    out : dict
        Time-domain trajectories, cumulative energy gain profiles for synchronized,
        detuned, and flat-surface cases, and animation frames.
    """
    lam0 = float(wavelength_m)
    f0 = c_light / lam0
    omega0 = 2.0 * np.pi * f0
    eps_diel = float(n_refr) ** 2
    lam_p = float(period_m)
    lz = float(n_periods) * lam_p

    nz = int(round(lz / float(dx_m)))
    dz_actual = lz / nz
    dy_m = float(dx_m)
    xmin_m, xmax_m = -2.0e-6, 2.0e-6
    nx = int(round((xmax_m - xmin_m) / float(dx_m)))
    a_gap = float(gap_half_m)

    def _run_supercell(dep: float, n_fr: int) -> dict:
        grid = GridFIT3D(
            xmin_m,
            xmax_m,
            -0.5 * dy_m,
            0.5 * dy_m,
            0.0,
            lz,
            nx,
            1,
            nz,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            bc_low=["cpml", "periodic", "periodic"],
            bc_high=["cpml", "periodic", "periodic"],
            verbose=0,
        )
        shapes: list[Box] = []
        x_lo_bot = -(a_gap + dep)
        shapes.append(
            Box(
                center=(0.5 * (xmin_m + x_lo_bot), 0.0, 0.5 * lz),
                size=(abs(x_lo_bot - xmin_m), dy_m, lz),
                material=eps_diel,
            )
        )
        x_hi_top = +(a_gap + dep)
        shapes.append(
            Box(
                center=(0.5 * (xmax_m + x_hi_top), 0.0, 0.5 * lz),
                size=(abs(xmax_m - x_hi_top), dy_m, lz),
                material=eps_diel,
            )
        )
        if dep > 0.0:
            for m in range(int(n_periods)):
                zc = (m + 0.5) * lam_p
                shapes.append(
                    Box(
                        center=(-a_gap - 0.5 * dep, 0.0, zc),
                        size=(dep, dy_m, float(duty_cycle) * lam_p),
                        material=eps_diel,
                    )
                )
                shapes.append(
                    Box(
                        center=(+a_gap + 0.5 * dep, 0.0, zc),
                        size=(dep, dy_m, float(duty_cycle) * lam_p),
                        material=eps_diel,
                    )
                )
        assign_materials(solver, shapes, frequency=f0)

        cw = ContinuousSource(frequency=f0, ramp_time=3.0 / f0)
        src = CurrentSource(
            cw,
            component="z",
            center=(-1.35e-6, 0.0, 0.5 * lz),
            size=(0.0, dy_m, lz),
            amplitude=1.0,
        )
        e0_ref = 0.5 * (Z0_VAC / float(n_refr))

        t_inject = 7.5 / f0
        v_min = min(float(beta_sync), float(beta_detuned)) * c_light
        t_transit_max = lz / v_min
        nt = int(np.ceil((t_inject + t_transit_max + 1.2 / f0) / solver.dt))
        preflight_solver(solver, n_steps=nt, mem_limit_mb=mem_limit_mb)

        ix0 = int(np.argmin(np.abs(solver.grid.x[: solver.Nx])))
        step_start = int(round(t_inject / solver.dt))

        ez_xt = []
        t_arr = []
        frames_2d = []
        frames_t = []

        if n_fr > 0:
            t_sync_end = t_inject + lz / (float(beta_sync) * c_light)
            fr_steps = set(
                np.unique(
                    np.clip(
                        np.round(np.linspace(t_inject, t_sync_end, n_fr) / solver.dt).astype(int),
                        step_start,
                        nt - 1,
                    )
                ).tolist()
            )
        else:
            fr_steps = set()

        for step in range(nt):
            t = step * solver.dt
            src.update(solver, t)
            solver.one_step()
            if step >= step_start:
                ez_xt.append(solver.E[ix0, 0, :, "z"].copy() / e0_ref)
                t_arr.append(t + solver.dt)
            if step in fr_steps:
                frames_2d.append(solver.E[:, 0, :, "z"].copy() / e0_ref)
                frames_t.append(t + solver.dt)

        return {
            "ez_xt": np.asarray(ez_xt),
            "t_arr": np.asarray(t_arr),
            "t_inject": t_inject,
            "x_m": solver.grid.x[: solver.Nx].copy(),
            "z_m": solver.z.copy(),
            "frames_Ez": np.asarray(frames_2d) if n_fr > 0 else None,
            "frames_t": np.asarray(frames_t) if n_fr > 0 else None,
        }

    grat_sim = _run_supercell(float(depth_m), record_frames)
    flat_sim = _run_supercell(0.0, 0)

    z_pts = np.linspace(0.0, lz, 240, endpoint=False)
    dz_pt = float(z_pts[1] - z_pts[0])

    def _integrate_path(sim_d: dict, beta_val: float) -> tuple[float, float, np.ndarray, np.ndarray]:
        v_e = float(beta_val) * c_light
        ez_xt = sim_d["ez_xt"]
        t_arr = sim_d["t_arr"]
        t_inj = sim_d["t_inject"]
        best_g = -1.0
        best_phi = 0.0
        best_cum = np.zeros_like(z_pts)
        best_inst = np.zeros_like(z_pts)
        for phi0 in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
            t_pts = t_inj + phi0 / omega0 + z_pts / v_e
            it = np.clip(np.searchsorted(t_arr, t_pts) - 1, 0, len(t_arr) - 2)
            wt = (t_pts - t_arr[it]) / (t_arr[it + 1] - t_arr[it])
            iz = np.clip(np.floor(z_pts / dz_actual).astype(int), 0, nz - 1)
            ez_along = (1.0 - wt) * ez_xt[it, iz] + wt * ez_xt[it + 1, iz]
            cum_v = np.cumsum(ez_along) * dz_pt
            # Ensure accelerating sign (net positive energy gain)
            if cum_v[-1] > best_g * lz:
                best_g = float(cum_v[-1] / lz)
                best_phi = float(phi0)
                best_cum = cum_v / lz
                best_inst = ez_along
        return best_g, best_phi, best_cum, best_inst

    g_sync, phi_sync, cum_sync, inst_sync = _integrate_path(grat_sim, beta_sync)
    g_det, _, cum_det, inst_det = _integrate_path(grat_sim, beta_detuned)
    g_flat, _, cum_flat, inst_flat = _integrate_path(flat_sim, beta_sync)

    # Compute electron z position for each recorded animation frame at optimal phase phi_sync
    if record_frames > 0 and grat_sim["frames_t"] is not None:
        v_s = float(beta_sync) * c_light
        z_e_frames = np.clip(
            (grat_sim["frames_t"] - grat_sim["t_inject"] - phi_sync / omega0) * v_s,
            0.0,
            lz,
        )
    else:
        z_e_frames = None

    return {
        "n_periods": int(n_periods),
        "period_m": lam_p,
        "lz_m": lz,
        "z_path_m": z_pts,
        "g_sync_over_e0": float(g_sync),
        "g_detuned_over_e0": float(g_det),
        "g_flat_over_e0": float(abs(g_flat)),
        "cum_gain_sync": cum_sync,
        "cum_gain_detuned": cum_det,
        "cum_gain_flat": cum_flat,
        "inst_ez_sync": inst_sync,
        "inst_ez_detuned": inst_det,
        "inst_ez_flat": inst_flat,
        "x_m": grat_sim["x_m"],
        "z_m": grat_sim["z_m"],
        "frames_Ez": grat_sim["frames_Ez"],
        "frames_ze_m": z_e_frames,
    }


# ---------------------------------------------------------------------------
# Task runners (Tasks 1-4)
# ---------------------------------------------------------------------------


def run_task_smith_purcell(quick: bool = False) -> dict:
    """Run Task 1: Smith-Purcell radiation over PEC and dielectric gratings + beta sweep."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dx = 0.05e-6 if quick else 0.04e-6
    n_periods = 10 if quick else 12
    n_frames = 16 if quick else 22

    pec = run_single_smith_purcell_case(
        material="pec",
        beta=DEFAULT_SP_BETA,
        period_m=DEFAULT_SP_PERIOD_M,
        n_periods=n_periods,
        depth_m=DEFAULT_SP_DEPTH_M,
        impact_m=DEFAULT_SP_IMPACT_M,
        dx_m=dx,
        dz_m=dx,
        record_frames=n_frames,
    )
    diel = run_single_smith_purcell_case(
        material=DEFAULT_EPS_SIO2,
        beta=DEFAULT_SP_BETA,
        period_m=DEFAULT_SP_PERIOD_M,
        n_periods=n_periods,
        depth_m=DEFAULT_SP_DEPTH_M,
        impact_m=DEFAULT_SP_IMPACT_M,
        dx_m=dx,
        dz_m=dx,
        record_frames=n_frames,
    )
    flat = run_single_smith_purcell_case(
        material="pec",
        beta=DEFAULT_SP_BETA,
        period_m=DEFAULT_SP_PERIOD_M,
        n_periods=n_periods,
        depth_m=0.0,
        impact_m=DEFAULT_SP_IMPACT_M,
        dx_m=dx,
        dz_m=dx,
        record_frames=0,
    )

    # Beta sweep for wavelength vs angle across relativistic velocities
    beta_sweep = np.array([0.68, 0.80, 0.92])
    lam_sim_beta = []
    lam_th_beta = []
    rel_err_beta = []
    for b_val in beta_sweep:
        if abs(b_val - DEFAULT_SP_BETA) < 1e-9:
            r_b = pec
        else:
            r_b = run_single_smith_purcell_case(
                material="pec",
                beta=float(b_val),
                period_m=DEFAULT_SP_PERIOD_M,
                n_periods=n_periods,
                depth_m=DEFAULT_SP_DEPTH_M,
                impact_m=DEFAULT_SP_IMPACT_M,
                dx_m=dx,
                dz_m=dx,
                record_frames=0,
            )
        lam_sim_beta.append(r_b["lambda_sim_m"])
        lam_th_beta.append(r_b["lambda_theory_m"])
        rel_err_beta.append(float(r_b["lambda_mean_rel_err"]))

    flat_to_pec_power_ratio = float(
        flat["total_radiated_metric"] / max(pec["total_radiated_metric"], 1e-30)
    )

    np.savez_compressed(
        DATA_DIR / "task_smith_purcell.npz",
        x_m=pec["x_m"],
        z_m=pec["z_m"],
        z_bunch_snap_m=np.array(pec["z_bunch_snap_m"]),
        Ez_pec_scat_xz=pec["Ez_scat_xz"],
        Ez_diel_scat_xz=diel["Ez_scat_xz"],
        Ez_flat_scat_xz=flat["Ez_scat_xz"],
        wavelengths_m=pec["wavelengths_m"],
        theta_grid_deg=pec["theta_grid_deg"],
        angular_map_pec=pec["angular_map_deweighted"],
        angular_map_diel=diel["angular_map_deweighted"],
        thetas_deg=pec["thetas_deg"],
        lambda_sim_pec_m=pec["lambda_sim_m"],
        lambda_sim_diel_m=diel["lambda_sim_m"],
        lambda_theory_m=pec["lambda_theory_m"],
        peak_power_pec=pec["peak_power_theta"],
        peak_power_diel=diel["peak_power_theta"],
        peak_power_flat=flat["peak_power_theta"],
        beta_sweep=beta_sweep,
        lambda_sim_beta_m=np.asarray(lam_sim_beta),
        lambda_theory_beta_m=np.asarray(lam_th_beta),
        frames_Ez_pec=pec["frames_Ez_scat"],
        frames_Ez_diel=diel["frames_Ez_scat"],
        frames_zb_m=pec["frames_zb_m"],
    )

    summary = {
        "task": "smith_purcell",
        "quick": bool(quick),
        "period_um": float(DEFAULT_SP_PERIOD_M * 1e6),
        "depth_um": float(DEFAULT_SP_DEPTH_M * 1e6),
        "beta": float(DEFAULT_SP_BETA),
        "impact_um": float(DEFAULT_SP_IMPACT_M * 1e6),
        "pec_lambda_mean_rel_err": float(pec["lambda_mean_rel_err"]),
        "pec_lambda_max_rel_err": float(pec["lambda_max_rel_err"]),
        "diel_lambda_mean_rel_err": float(diel["lambda_mean_rel_err"]),
        "diel_lambda_max_rel_err": float(diel["lambda_max_rel_err"]),
        "beta_sweep": beta_sweep.tolist(),
        "beta_sweep_mean_rel_err": rel_err_beta,
        "flat_to_pec_power_ratio": flat_to_pec_power_ratio,
        "pec_wake_rad_loss_J": float(pec["wake_rad_loss_J"]),
        "diel_wake_rad_loss_J": float(diel["wake_rad_loss_J"]),
    }
    (DATA_DIR / "task_smith_purcell.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_task_impact_decay(quick: bool = False) -> dict:
    """Run Task 2: Smith-Purcell evanescent decay vs impact parameter h and beta."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dx = 0.05e-6 if quick else 0.04e-6
    n_periods = 8 if quick else 10
    h_values_m = np.array([0.20e-6, 0.30e-6, 0.40e-6, 0.50e-6, 0.60e-6])
    betas = np.array([0.80, 0.90])

    intensities_all = []
    spectra_90_b0 = []
    alpha_sim_list = []
    alpha_th_list = []
    decay_len_sim_list = []
    decay_len_th_list = []
    rel_err_list = []

    lams_narrow = np.linspace(1.30e-6, 2.80e-6, 61)
    freqs_narrow = c_light / lams_narrow

    for ib, b_val in enumerate(betas):
        lam_90 = DEFAULT_SP_PERIOD_M / float(b_val)
        th_info = smith_purcell_evanescent_decay(h_values_m, beta=float(b_val), wavelength_m=lam_90)
        alpha_th = float(th_info["alpha_power_1_m"])
        intens_b = []
        for hv in h_values_m:
            r_h = run_single_smith_purcell_case(
                material="pec",
                beta=float(b_val),
                period_m=DEFAULT_SP_PERIOD_M,
                n_periods=n_periods,
                depth_m=DEFAULT_SP_DEPTH_M,
                impact_m=float(hv),
                dx_m=dx,
                dz_m=dx,
                wavelengths_m=lams_narrow,
                thetas_deg=np.array([90.0]),
                record_frames=0,
            )
            intens_b.append(float(r_h["broadside_intensity"]))
            if ib == 0:
                idx90 = int(np.argmin(np.abs(r_h["theta_grid_deg"] - 90.0)))
                spectra_90_b0.append(r_h["angular_map_raw"][idx90, :])

        intens_arr = np.asarray(intens_b, dtype=float)
        intensities_all.append(intens_arr)
        slope, _ = np.polyfit(h_values_m, np.log(intens_arr), 1)
        alpha_sim = float(-slope)
        rel_err = abs(alpha_sim - alpha_th) / alpha_th
        alpha_sim_list.append(alpha_sim)
        alpha_th_list.append(alpha_th)
        decay_len_sim_list.append(float(1.0 / alpha_sim))
        decay_len_th_list.append(float(1.0 / alpha_th))
        rel_err_list.append(float(rel_err))

    np.savez_compressed(
        DATA_DIR / "task_impact_decay.npz",
        h_values_m=h_values_m,
        betas=betas,
        wavelengths_m=lams_narrow,
        freqs_hz=freqs_narrow,
        spectra_90_beta08=np.asarray(spectra_90_b0),
        intensities_vs_h=np.asarray(intensities_all),
        alpha_sim_1_m=np.asarray(alpha_sim_list),
        alpha_theory_1_m=np.asarray(alpha_th_list),
        decay_length_sim_m=np.asarray(decay_len_sim_list),
        decay_length_theory_m=np.asarray(decay_len_th_list),
    )

    summary = {
        "task": "impact_decay",
        "quick": bool(quick),
        "h_values_um": (h_values_m * 1e6).tolist(),
        "betas": betas.tolist(),
        "alpha_sim_1_m": alpha_sim_list,
        "alpha_theory_1_m": alpha_th_list,
        "decay_length_sim_um": [x * 1e6 for x in decay_len_sim_list],
        "decay_length_theory_um": [x * 1e6 for x in decay_len_th_list],
        "alpha_rel_err": rel_err_list,
    }
    (DATA_DIR / "task_impact_decay.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_task_dla_grating(quick: bool = False) -> dict:
    """Run Task 3: Inverse process (DLA) synchronism, depth/duty sweeps, and Peralta 2013 comparison."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dx = 0.025e-6 if quick else 0.020e-6
    n_periods_eval = 20

    # 1. Synchronous reference cell at Peralta et al. (2013) parameters (lam_0 = lam_p = 800 nm, 2a = 400 nm)
    sync_400 = run_single_dla_unit_cell(
        period_m=DEFAULT_DLA_PERIOD_M,
        depth_m=DEFAULT_DLA_DEPTH_M,
        duty_cycle=DEFAULT_DLA_DUTY,
        gap_half_m=0.20e-6,
        beta=0.999964,
        n_periods_eval=n_periods_eval,
        dx_m=dx,
        record_2d_slice=True,
    )
    sync_800 = run_single_dla_unit_cell(
        period_m=DEFAULT_DLA_PERIOD_M,
        depth_m=DEFAULT_DLA_DEPTH_M,
        duty_cycle=DEFAULT_DLA_DUTY,
        gap_half_m=0.40e-6,
        beta=0.999964,
        n_periods_eval=n_periods_eval,
        dx_m=dx,
        record_2d_slice=False,
    )

    # 2. Grating period scan around synchronism for beta = 1.0 (lam_sync = 800 nm) and beta = 0.75 (lam_sync = 600 nm)
    periods_scan_m = np.linspace(0.50e-6, 1.00e-6, 11 if quick else 15)
    g_vs_period_b100 = []
    g_vs_period_b075 = []
    for lp in periods_scan_m:
        r_100 = run_single_dla_unit_cell(
            period_m=float(lp),
            depth_m=DEFAULT_DLA_DEPTH_M,
            duty_cycle=DEFAULT_DLA_DUTY,
            beta=0.999964,
            n_periods_eval=n_periods_eval,
            dx_m=dx,
        )
        r_075 = run_single_dla_unit_cell(
            period_m=float(lp),
            depth_m=0.50e-6,
            duty_cycle=DEFAULT_DLA_DUTY,
            gap_half_m=0.10e-6,
            beta=0.75,
            n_periods_eval=n_periods_eval,
            dx_m=dx,
        )
        g_vs_period_b100.append(float(r_100["g_over_e0_multiperiod"]))
        g_vs_period_b075.append(float(r_075["g_over_e0_multiperiod"]))

    g_vs_period_b100 = np.asarray(g_vs_period_b100)
    g_vs_period_b075 = np.asarray(g_vs_period_b075)
    lam_peak_b100, _ = _parabolic_peak(periods_scan_m, g_vs_period_b100)
    lam_peak_b075, _ = _parabolic_peak(periods_scan_m, g_vs_period_b075)

    # 3. Tooth depth scan h_g in [0, 950] nm (including h_g = 0 flat mutation)
    depths_scan_m = np.array([0.0, 0.15e-6, 0.30e-6, 0.45e-6, 0.60e-6, 0.68e-6, 0.80e-6, 0.95e-6])
    g_vs_depth = []
    for hg in depths_scan_m:
        r_hg = run_single_dla_unit_cell(
            period_m=DEFAULT_DLA_PERIOD_M,
            depth_m=float(hg),
            duty_cycle=DEFAULT_DLA_DUTY,
            beta=0.999964,
            n_periods_eval=n_periods_eval,
            dx_m=dx,
        )
        g_vs_depth.append(float(r_hg["g_over_e0_cell"]))
    g_vs_depth = np.asarray(g_vs_depth)

    # 4. Duty-cycle scan DC in [0.0, 1.0]
    duty_scan = np.array([0.0, 0.15, 0.30, 0.45, 0.50, 0.60, 0.75, 0.85, 1.0])
    g_vs_duty = []
    for dc in duty_scan:
        r_dc = run_single_dla_unit_cell(
            period_m=DEFAULT_DLA_PERIOD_M,
            depth_m=DEFAULT_DLA_DEPTH_M,
            duty_cycle=float(dc),
            beta=0.999964,
            n_periods_eval=n_periods_eval,
            dx_m=dx,
        )
        g_vs_duty.append(float(r_dc["g_over_e0_cell"]))
    g_vs_duty = np.asarray(g_vs_duty)

    # 5. Multi-period step-by-step time-domain electron tracking + GIF frames
    track = run_dla_multiperiod_tracking(
        n_periods=6,
        period_m=DEFAULT_DLA_PERIOD_M,
        depth_m=DEFAULT_DLA_DEPTH_M,
        duty_cycle=DEFAULT_DLA_DUTY,
        dx_m=dx,
        record_frames=16 if quick else 22,
    )

    # Peralta et al. (2013) gradient [MeV/m] vs incident E_0 [GV/m]
    e0_gv_m = np.linspace(0.0, 2.00, 41)
    grad_400_mev_m = float(sync_400["g_over_e0_cell"]) * e0_gv_m * 1.0e3
    grad_800_mev_m = float(sync_800["g_over_e0_cell"]) * e0_gv_m * 1.0e3
    peralta_comp = peralta_2013_gradient_comparison(
        g_over_e0_400=float(sync_400["g_over_e0_cell"]),
        g_over_e0_800=float(sync_800["g_over_e0_cell"]),
    )

    np.savez_compressed(
        DATA_DIR / "task_dla_grating.npz",
        x_2d_m=sync_400["x_2d_m"],
        z_2d_m=sync_400["z_2d_m"],
        Ez_2d_norm_real=np.real(sync_400["Ez_2d_norm"]),
        z_cell_m=sync_400["z_cell_m"],
        Ez_cell_norm_real=np.real(sync_400["Ez_cell_norm"]),
        harmonics_m=sync_400["harmonics_m"],
        harmonics_amp_400=sync_400["harmonics_amp"],
        harmonics_amp_800=sync_800["harmonics_amp"],
        periods_scan_m=periods_scan_m,
        g_vs_period_b100=g_vs_period_b100,
        g_vs_period_b075=g_vs_period_b075,
        depths_scan_m=depths_scan_m,
        g_vs_depth=g_vs_depth,
        duty_scan=duty_scan,
        g_vs_duty=g_vs_duty,
        e0_gv_m=e0_gv_m,
        grad_400_mev_m=grad_400_mev_m,
        grad_800_mev_m=grad_800_mev_m,
        z_path_m=track["z_path_m"],
        cum_gain_sync=track["cum_gain_sync"],
        cum_gain_detuned=track["cum_gain_detuned"],
        cum_gain_flat=track["cum_gain_flat"],
        inst_ez_sync=track["inst_ez_sync"],
        inst_ez_flat=track["inst_ez_flat"],
        track_x_m=track["x_m"],
        track_z_m=track["z_m"],
        frames_dla_Ez=track["frames_Ez"],
        frames_dla_ze_m=track["frames_ze_m"],
    )

    summary = {
        "task": "dla_grating",
        "quick": bool(quick),
        "laser_wavelength_nm": float(DEFAULT_DLA_LAM0_M * 1e9),
        "synchronous_period_b100_nm": 800.0,
        "fitted_peak_period_b100_nm": float(lam_peak_b100 * 1e9),
        "synchronous_period_b075_nm": 600.0,
        "fitted_peak_period_b075_nm": float(lam_peak_b075 * 1e9),
        "g_over_e0_400nm_gap": float(sync_400["g_over_e0_cell"]),
        "g_over_e0_800nm_gap": float(sync_800["g_over_e0_cell"]),
        "gap_ratio_400_to_800_sim": float(peralta_comp["gap_ratio_sim"]),
        "gap_ratio_400_to_800_sim_rounded": float(peralta_comp["gap_ratio_sim_rounded"]),
        "gap_ratio_400_to_800_exp": float(peralta_comp["gap_ratio_exp"]),
        "gap_ratio_400_to_800_exp_err": float(peralta_comp["gap_ratio_exp_err"]),
        "g_over_e0_flat_mutation": float(g_vs_depth[0]),
        "g_over_e0_duty0_mutation": float(g_vs_duty[0]),
        "g_over_e0_duty1_mutation": float(g_vs_duty[-1]),
        "optimal_depth_nm": float(depths_scan_m[int(np.argmax(g_vs_depth))] * 1e9),
        "optimal_duty_cycle": float(duty_scan[int(np.argmax(g_vs_duty))]),
        "time_domain_tracked_g_sync": float(track["g_sync_over_e0"]),
        "time_domain_tracked_g_detuned": float(track["g_detuned_over_e0"]),
        "time_domain_tracked_g_flat": float(track["g_flat_over_e0"]),
        "peralta_2013_max_gradient_400nm_mev_m": 309.8,
        "peralta_2013_max_gradient_400nm_err_mev_m": 20.7,
        "peralta_2013_max_gradient_800nm_mev_m": 152.0,
        "peralta_2013_max_gradient_800nm_err_mev_m": 15.0,
        "peralta_2013_e0_sio2_319uj_gv_m": float(peralta_comp["e0_sio2_319uj_gv_m"]),
        "peralta_2013_e0_vac_319uj_gv_m": float(peralta_comp["e0_vac_319uj_gv_m"]),
        "peralta_2013_e0_sio2_damage_gv_m": float(peralta_comp["e0_sio2_damage_gv_m"]),
        "peralta_2013_e0_vac_damage_gv_m": float(peralta_comp["e0_vac_damage_gv_m"]),
        "peralta_2013_pred_gradient_400nm_sio2_319uj_mev_m": float(
            peralta_comp["g_pred_400_sio2_319uj_mev_m"]
        ),
        "peralta_2013_pred_gradient_400nm_sio2_damage_mev_m": float(
            peralta_comp["g_pred_400_sio2_damage_mev_m"]
        ),
    }
    (DATA_DIR / "task_dla_grating.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_task_convergence(quick: bool = False) -> dict:
    """Run Task 4: 3-level resolution convergence study for Smith-Purcell & DLA."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    levels = ["Coarse", "Medium", "Fine"]
    # Part 1 SP grid spacings [m] commensurable with h = 250 nm & h_g = 400 nm (ppw in SiO2 = 17.2, 23.0, 28.7 >= 20)
    dx_sp_list = [0.0835e-6, 0.0625e-6, 0.050e-6]
    # Part 2 DLA grid spacings [m] commensurable with 200 nm half-gap & 400 nm tooth (Fine dx=20 nm gives 27.5 cells/lambda >= 20)
    dx_dla_list = [0.050e-6, 0.025e-6, 0.020e-6]

    sp_lam_errs = []
    sp_ppw_sio2 = []
    dla_g_vals = []
    dla_ppw_sio2 = []

    lam_broadside_sio2 = (DEFAULT_SP_PERIOD_M / DEFAULT_SP_BETA) / DEFAULT_N_SIO2
    lam_dla_sio2 = DEFAULT_DLA_LAM0_M / DEFAULT_N_SIO2

    for idx in range(3):
        dx_sp = dx_sp_list[idx]
        r_sp = run_single_smith_purcell_case(
            material="pec",
            beta=DEFAULT_SP_BETA,
            period_m=DEFAULT_SP_PERIOD_M,
            n_periods=10,
            depth_m=DEFAULT_SP_DEPTH_M,
            impact_m=DEFAULT_SP_IMPACT_M,
            dx_m=dx_sp,
            dz_m=dx_sp,
            record_frames=0,
        )
        sp_lam_errs.append(float(r_sp["lambda_mean_rel_err"]))
        sp_ppw_sio2.append(float(lam_broadside_sio2 / dx_sp))

        dx_dla = dx_dla_list[idx]
        r_dla = run_single_dla_unit_cell(
            period_m=DEFAULT_DLA_PERIOD_M,
            depth_m=DEFAULT_DLA_DEPTH_M,
            duty_cycle=DEFAULT_DLA_DUTY,
            dx_m=dx_dla,
        )
        dla_g_vals.append(float(r_dla["g_over_e0_cell"]))
        dla_ppw_sio2.append(float(lam_dla_sio2 / dx_dla))

    # Richardson extrapolation for DLA G/E0 from the two finest levels
    r_ratio = dx_dla_list[1] / dx_dla_list[2]
    g_rich = dla_g_vals[2] + (dla_g_vals[2] - dla_g_vals[1]) / (r_ratio**2 - 1.0)
    dla_rel_errs = [float(abs(g - g_rich) / abs(g_rich)) for g in dla_g_vals]

    np.savez_compressed(
        DATA_DIR / "task_convergence.npz",
        levels=np.asarray(levels),
        dx_sp_nm=np.asarray(dx_sp_list) * 1e9,
        sp_ppw_sio2=np.asarray(sp_ppw_sio2),
        sp_lambda_mean_rel_err=np.asarray(sp_lam_errs),
        dx_dla_nm=np.asarray(dx_dla_list) * 1e9,
        dla_ppw_sio2=np.asarray(dla_ppw_sio2),
        dla_g_over_e0=np.asarray(dla_g_vals),
        dla_g_richardson=np.array(g_rich),
        dla_rel_err=np.asarray(dla_rel_errs),
    )

    summary = {
        "task": "convergence",
        "quick": bool(quick),
        "levels": levels,
        "dx_sp_nm": [round(x * 1e9, 2) for x in dx_sp_list],
        "sp_ppw_sio2": [round(x, 2) for x in sp_ppw_sio2],
        "sp_lambda_mean_rel_err": sp_lam_errs,
        "dx_dla_nm": [round(x * 1e9, 2) for x in dx_dla_list],
        "dla_ppw_sio2": [round(x, 2) for x in dla_ppw_sio2],
        "dla_g_over_e0": dla_g_vals,
        "dla_g_richardson": float(g_rich),
        "dla_rel_err": dla_rel_errs,
    }
    (DATA_DIR / "task_convergence.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Publication-style figure & animation rendering (7 PNGs + 2 GIFs)
# ---------------------------------------------------------------------------

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#E69F00",
    "sky": "#56B4E9",
    "black": "#111111",
    "grey": "#777777",
}


def _apply_paper_style() -> None:
    """Configure high-contrast paper matplotlib style (white bg, sans-serif, >= 8.5 pt)."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.9,
            "grid.color": "#DDDDDD",
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
        }
    )


def _draw_sp_grating_overlay(
    ax: plt.Axes,
    z_min_um: float,
    z_max_um: float,
    x_min_um: float,
    x_max_um: float,
    period_um: float = 1.67,
    n_periods: int = 10,
    depth_um: float = 0.40,
    duty: float = 0.50,
    mat_label: str = "PEC grating",
    face_color: str = "#444444",
    cpml_um: float = 0.50,
) -> None:
    """Overlay grating teeth, substrate, and CPML frame on an (x, z) field map."""
    # CPML frame dashed boundary
    ax.add_patch(
        Rectangle(
            (z_min_um + cpml_um, x_min_um + cpml_um),
            (z_max_um - z_min_um) - 2.0 * cpml_um,
            (x_max_um - x_min_um) - 2.0 * cpml_um,
            fill=False,
            edgecolor="#555555",
            linestyle="--",
            linewidth=0.9,
            zorder=5,
        )
    )
    ax.text(
        z_max_um - cpml_um - 0.3,
        x_max_um - cpml_um - 0.28,
        "CPML",
        fontsize=8.5,
        color="#333333",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.82, edgecolor="none"),
        zorder=6,
    )
    x_base_top = -depth_um if depth_um > 0.0 else 0.0
    ax.add_patch(
        Rectangle(
            (z_min_um, x_min_um),
            z_max_um - z_min_um,
            x_base_top - x_min_um,
            facecolor=face_color,
            edgecolor="#111111",
            alpha=0.35,
            linewidth=0.8,
            zorder=4,
        )
    )
    if depth_um > 0.0:
        zg_min = -0.5 * n_periods * period_um
        zg_max = +0.5 * n_periods * period_um
        ax.add_patch(
            Rectangle(
                (z_min_um, -depth_um),
                zg_min - z_min_um,
                depth_um,
                facecolor=face_color,
                edgecolor="#111111",
                alpha=0.35,
                linewidth=0.8,
                zorder=4,
            )
        )
        ax.add_patch(
            Rectangle(
                (zg_max, -depth_um),
                z_max_um - zg_max,
                depth_um,
                facecolor=face_color,
                edgecolor="#111111",
                alpha=0.35,
                linewidth=0.8,
                zorder=4,
            )
        )
        for m in range(n_periods):
            zc = zg_min + (m + 0.5) * period_um
            w = duty * period_um
            ax.add_patch(
                Rectangle(
                    (zc - 0.5 * w, -depth_um),
                    w,
                    depth_um,
                    facecolor=face_color,
                    edgecolor="#111111",
                    alpha=0.42,
                    linewidth=0.8,
                    zorder=4,
                )
            )
    ax.axhline(0.0, color="#222222", linestyle=":", linewidth=0.7, zorder=4)
    ax.text(
        z_min_um + cpml_um + 0.3,
        x_min_um + 0.16,
        mat_label,
        fontsize=8.5,
        fontweight="bold",
        color="#111111",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.88, edgecolor="#888888", lw=0.6),
        zorder=6,
    )


def plot_01_setup_schematic_and_mesh() -> Path:
    """Render Figure 1: Labelled setup schematics (Part 1 SP & Part 2 DLA) + graded FIT mesh view."""
    _apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)

    # Panel (a): Part 1 Smith-Purcell schematic
    ax = axes[0]
    ax.set_xlim(-11.0, 11.0)
    ax.set_ylim(-1.1, 3.4)
    _draw_sp_grating_overlay(
        ax,
        z_min_um=-11.0,
        z_max_um=11.0,
        x_min_um=-1.0,
        x_max_um=3.3,
        period_um=1.67,
        n_periods=8,
        depth_um=0.40,
        duty=0.50,
        mat_label=r"PEC ($\sigma\to\infty$) or $\mathrm{SiO}_2$ ($\varepsilon_r=2.11, \mu_r=1$)",
        face_color="#56B4E9",
        cpml_um=0.6,
    )
    # Beam line and Gaussian bunch
    h_um = 0.35
    ax.axhline(h_um, color=OKABE_ITO["orange"], linestyle="-.", linewidth=1.2, zorder=5)
    ax.plot([-2.2], [h_um], marker="o", ms=7, color=OKABE_ITO["orange"], zorder=7)
    ax.annotate(
        r"Rigid bunch $\mathbf{v}=\beta c\,\hat{\mathbf{z}}$" + "\n" + r"($q, \sigma_z=0.20\;\mu\mathrm{m}$, TM: $E_x, E_z, H_y$)",
        xy=(-2.2, h_um),
        xytext=(-9.8, 1.05),
        fontsize=8.5,
        color=OKABE_ITO["orange"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=OKABE_ITO["orange"], lw=1.1),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor=OKABE_ITO["orange"], lw=0.7),
        zorder=8,
    )
    # Impact parameter h annotation
    ax.annotate(
        "",
        xy=(4.8, h_um),
        xytext=(4.8, 0.0),
        arrowprops=dict(arrowstyle="<->", color="#111111", lw=1.0),
    )
    ax.text(5.1, 0.5 * h_um, r"$h$", fontsize=9.0, fontweight="bold", va="center")
    # Monitor plane
    x_mon_um = 2.25
    ax.plot([-8.5, 8.5], [x_mon_um, x_mon_um], color=OKABE_ITO["green"], lw=2.0, ls="-", zorder=6)
    ax.text(
        -8.3,
        x_mon_um + 0.14,
        r"Near2Far / DFTMonitor ($x_{\mathrm{mon}}=2.25\;\mu\mathrm{m}$)",
        fontsize=8.5,
        color=OKABE_ITO["green"],
        fontweight="bold",
    )
    # Emitted ray at angle theta
    th_r = np.radians(65.0)
    ax.add_patch(
        FancyArrowPatch(
            (0.0, 0.0),
            (2.2 * np.cos(th_r), 2.2 * np.sin(th_r)),
            arrowstyle="->",
            mutation_scale=12,
            color=OKABE_ITO["blue"],
            lw=1.6,
            zorder=7,
        )
    )
    ax.text(
        1.15,
        1.35,
        r"$\theta$: $\lambda = \frac{d}{n}(\frac{1}{\beta}-\cos\theta)$",
        fontsize=8.5,
        color=OKABE_ITO["blue"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.88, edgecolor="none"),
    )
    ax.set_title("(a) Part 1 · Smith–Purcell Setup (Periodic along y)")
    ax.set_xlabel(r"Longitudinal axis $z$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"Surface normal $x$ [$\mu\mathrm{m}$]")

    # Panel (b): Part 2 DLA schematic (Peralta et al. 2013)
    ax = axes[1]
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-2.1, 2.1)
    # Periodic z boundaries, CPML x boundaries
    ax.axhline(1.75, color="#555555", ls="--", lw=0.9)
    ax.axhline(-1.75, color="#555555", ls="--", lw=0.9)
    ax.axvline(-0.80, color=OKABE_ITO["purple"], ls="--", lw=1.0)
    ax.axvline(+0.80, color=OKABE_ITO["purple"], ls="--", lw=1.0)
    ax.text(0.0, 1.84, r"$+x$ CPML ($10\;\Delta x$)", ha="center", va="bottom", fontsize=8.5, color="#444444")
    ax.text(0.0, -1.84, r"$-x$ CPML ($10\;\Delta x$)", ha="center", va="top", fontsize=8.5, color="#444444")
    ax.text(-0.84, 0.0, "Periodic $z$", rotation=90, ha="right", va="center", fontsize=8.5, color=OKABE_ITO["purple"])

    # Dual fused silica gratings (2a = 400 nm, h_g = 680 nm, lam_p = 800 nm)
    a_um = 0.20
    hg_um = 0.68
    lp_um = 0.80
    # Lower wafer & teeth
    ax.add_patch(Rectangle((-0.80, -1.75), 1.60, 1.75 - (a_um + hg_um), facecolor="#56B4E9", alpha=0.35, edgecolor="#111111"))
    ax.add_patch(Rectangle((-0.80, a_um + hg_um), 1.60, 1.75 - (a_um + hg_um), facecolor="#56B4E9", alpha=0.35, edgecolor="#111111"))
    for zc in [-0.40, 0.40]:
        ax.add_patch(Rectangle((zc - 0.20, -(a_um + hg_um)), 0.40, hg_um, facecolor="#56B4E9", alpha=0.5, edgecolor="#111111"))
        ax.add_patch(Rectangle((zc - 0.20, a_um), 0.40, hg_um, facecolor="#56B4E9", alpha=0.5, edgecolor="#111111"))

    ax.text(0.0, -1.25, r"Fused silica $\mathrm{SiO}_2$ ($n=1.453, \varepsilon_r=2.11$)", ha="center", fontsize=8.5, fontweight="bold")
    # Laser source plane
    ax.plot([-0.78, 0.78], [-1.45, -1.45], color=OKABE_ITO["blue"], lw=2.0)
    ax.add_patch(FancyArrowPatch((-0.55, -1.45), (-0.55, -0.95), arrowstyle="->", mutation_scale=11, color=OKABE_ITO["blue"], lw=1.4))
    ax.text(0.05, -1.58, r"Laser $\lambda_0=800\;\mathrm{nm}$ ($\mathbf{k}_{\mathrm{inc}}=+k_0\hat{\mathbf{x}}, \mathbf{E}_0\parallel\hat{\mathbf{z}}$)", ha="center", fontsize=8.5, color=OKABE_ITO["blue"], fontweight="bold")
    # Electron channel at x = 0
    ax.plot([-0.78, 0.78], [0.0, 0.0], color=OKABE_ITO["orange"], ls="-.", lw=1.4)
    ax.add_patch(FancyArrowPatch((-0.65, 0.0), (0.65, 0.0), arrowstyle="->", mutation_scale=12, color=OKABE_ITO["orange"], lw=1.6))
    ax.text(0.0, 0.07, r"$e^-$ beam ($60\;\mathrm{MeV}, \beta\approx 1, 2a=400\;\mathrm{nm}$)", ha="center", va="bottom", fontsize=8.5, color=OKABE_ITO["orange"], fontweight="bold")
    ax.annotate("", xy=(-0.40, a_um + hg_um + 0.12), xytext=(0.40, a_um + hg_um + 0.12), arrowprops=dict(arrowstyle="<->", lw=1.0))
    ax.text(0.0, a_um + hg_um + 0.16, r"$\lambda_p = \beta\lambda_0 = 800\;\mathrm{nm}$", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_title("(b) Part 2 · Inverse Process: DLA (Peralta et al. 2013)")
    ax.set_xlabel(r"Longitudinal coordinate $z$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"Transverse coordinate $x$ [$\mu\mathrm{m}$]")

    # Panel (c): Graded FIT mesh view snapped to tooth faces
    ax = axes[2]
    x_faces = np.array([-0.88e-6, -0.20e-6, 0.0, 0.20e-6, 0.88e-6])
    x_sizes = np.array([0.025e-6, 0.020e-6, 0.020e-6, 0.025e-6])
    x_nodes = graded_axis(x_faces, x_sizes, min_cells=8, grading=1.25) * 1e6
    z_faces = np.array([-0.40e-6, -0.20e-6, 0.20e-6, 0.40e-6])
    z_sizes = np.array([0.025e-6, 0.020e-6, 0.025e-6])
    z_nodes = graded_axis(z_faces, z_sizes, min_cells=8, grading=1.25) * 1e6

    for zn in z_nodes:
        ax.axvline(zn, color="#CCCCCC", lw=0.55, zorder=1)
    for xn in x_nodes:
        ax.axhline(xn, color="#CCCCCC", lw=0.55, zorder=1)
    ax.add_patch(Rectangle((-0.20, -0.88), 0.40, 0.68, facecolor="#56B4E9", alpha=0.45, edgecolor=OKABE_ITO["blue"], lw=1.4, zorder=3))
    ax.add_patch(Rectangle((-0.20, +0.20), 0.40, 0.68, facecolor="#56B4E9", alpha=0.45, edgecolor=OKABE_ITO["blue"], lw=1.4, zorder=3))
    ax.axhline(0.0, color=OKABE_ITO["orange"], ls="-.", lw=1.3, zorder=4)
    ax.text(
        0.0,
        0.0,
        r"$\Delta x_{\min}=20\;\mathrm{nm}$ ($\lambda_{\mathrm{SiO}_2}/\Delta x = 27.5 \geq 20$)"
        + "\n"
        + r"$20\;\mathrm{cells}$ across tooth, $20\;\mathrm{cells}$ across $2a$",
        ha="center",
        va="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.92, edgecolor="#333333", lw=0.7),
        zorder=5,
    )
    ax.set_xlim(-0.40, 0.40)
    ax.set_ylim(-0.88, 0.88)
    ax.set_title("(c) Snapped FIT Mesh View (Unit Cell)")
    ax.set_xlabel(r"$z$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"$x$ [$\mu\mathrm{m}$]")

    out = FIG_DIR / "01_setup_schematic_and_mesh.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_02_sp_field_maps(sp_npz: np.lib.npyio.NpzFile) -> Path:
    """Render Figure 2: Smith-Purcell Ez(x, z) field maps (PEC grating, SiO2 grating, flat mutation)."""
    _apply_paper_style()
    x_um = sp_npz["x_m"] * 1e6
    z_um = sp_npz["z_m"] * 1e6
    zb_um = float(sp_npz["z_bunch_snap_m"]) * 1e6
    ez_pec = sp_npz["Ez_pec_scat_xz"]
    ez_diel = sp_npz["Ez_diel_scat_xz"]
    ez_flat = sp_npz["Ez_flat_scat_xz"]

    # Mask above the grating top x > 0.35 um to compute clean radiation color scale
    rad_mask = x_um >= 0.45
    vmax = float(np.percentile(np.abs(ez_pec[rad_mask, :]), 99.2))
    vmax = max(vmax, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), constrained_layout=True, sharey=True)
    panels = [
        (r"(a) Metallic (PEC) Grating ($h_g=0.40\;\mu\mathrm{m}$)", ez_pec, r"PEC grating ($d=1.67\;\mu\mathrm{m}$)", "#555555"),
        (r"(b) Fused-Silica Grating ($\varepsilon_r=2.11, n=1.453$)", ez_diel, r"$\mathrm{SiO}_2$ grating ($\varepsilon_r=2.11$)", "#56B4E9"),
        (r"(c) Flat-Surface Mutation ($h_g=0$, No Radiation)", ez_flat, r"Flat PEC surface ($h_g=0$)", "#777777"),
    ]
    im = None
    for idx, (title, field_xz, mlabel, fcol) in enumerate(panels):
        ax = axes[idx]
        im = ax.pcolormesh(
            z_um,
            x_um,
            field_xz,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=+vmax,
            shading="auto",
        )
        dep_um = 0.0 if idx == 2 else 0.40
        _draw_sp_grating_overlay(
            ax,
            z_min_um=float(z_um[0]),
            z_max_um=float(z_um[-1]),
            x_min_um=float(x_um[0]),
            x_max_um=float(x_um[-1]),
            period_um=1.67,
            n_periods=int(round((z_um[-1] - z_um[0] - 6.0) / 1.67)),
            depth_um=dep_um,
            duty=0.50,
            mat_label=mlabel,
            face_color=fcol,
            cpml_um=0.50,
        )
        ax.plot([zb_um], [0.25], marker="o", ms=6.5, color=OKABE_ITO["orange"], mec="white", mew=0.8, zorder=8)
        ax.text(
            zb_um - 0.5,
            0.42,
            r"$q$ ($\beta=0.80$)",
            fontsize=8.5,
            color=OKABE_ITO["orange"],
            fontweight="bold",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85, edgecolor="none"),
            zorder=8,
        )
        ax.set_title(title)
        ax.set_xlabel(r"$z$ [$\mu\mathrm{m}$]")
        if idx == 0:
            ax.set_ylabel(r"$x$ [$\mu\mathrm{m}$]")

    cbar = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.015)
    cbar.set_label(r"Scattered longitudinal field $E_z^{\mathrm{SP}}(x, z)$ [V/m]")
    out = FIG_DIR / "02_sp_field_maps.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_03_sp_dispersion_and_angular_spectrum(sp_npz: np.lib.npyio.NpzFile) -> Path:
    """Render Figure 3: Smith-Purcell angular spectrum map, dispersion vs beta, and PEC/SiO2/flat radiation."""
    _apply_paper_style()
    lams_um = sp_npz["wavelengths_m"] * 1e6
    theta_grid = sp_npz["theta_grid_deg"]
    ang_map = sp_npz["angular_map_pec"]
    thetas_deg = sp_npz["thetas_deg"]
    lam_pec_um = sp_npz["lambda_sim_pec_m"] * 1e6
    lam_diel_um = sp_npz["lambda_sim_diel_m"] * 1e6
    beta_sweep = sp_npz["beta_sweep"]
    lam_sim_beta_um = sp_npz["lambda_sim_beta_m"] * 1e6
    pw_pec = sp_npz["peak_power_pec"]
    pw_diel = sp_npz["peak_power_diel"]
    pw_flat = sp_npz["peak_power_flat"]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), constrained_layout=True)

    # Panel (a): 2D angular-spectral map U(theta, lambda) + n=1,2 theory curves
    ax = axes[0]
    norm_map = ang_map / max(float(np.max(ang_map)), 1e-30)
    pcm = ax.pcolormesh(
        theta_grid,
        lams_um,
        norm_map.T,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )
    th_dense = np.linspace(30.0, 155.0, 200)
    lam_n1 = 1.67 * (1.0 / 0.80 - np.cos(np.radians(th_dense)))
    lam_n2 = (1.67 / 2.0) * (1.0 / 0.80 - np.cos(np.radians(th_dense)))
    ax.plot(th_dense, lam_n1, "-", color=OKABE_ITO["orange"], lw=1.8, label=r"Theory $n=1$: $d(1/\beta-\cos\theta)$")
    ax.plot(th_dense, lam_n2, "--", color=OKABE_ITO["purple"], lw=1.5, label=r"Theory $n=2$: $(d/2)(1/\beta-\cos\theta)$")
    ax.plot(thetas_deg, lam_pec_um, "o", color="white", mec=OKABE_ITO["black"], ms=5.5, mew=1.1, label="Wakis FIT ($n=1$ peaks)")
    ax.set_xlim(30.0, 155.0)
    ax.set_ylim(float(lams_um[0]), float(lams_um[-1]))
    ax.set_title(r"(a) Angular Spectrum $U(\theta, \lambda)$ ($\beta=0.80$)")
    ax.set_xlabel(r"Observation angle $\theta$ from $+\hat{\mathbf{z}}$ [deg]")
    ax.set_ylabel(r"Emitted wavelength $\lambda$ [$\mu\mathrm{m}$]")
    ax.legend(loc="upper left", framealpha=0.92)
    fig.colorbar(pcm, ax=ax, shrink=0.86, label="Normalized spectral intensity [-]")

    # Panel (b): Wavelength vs angle for PEC & SiO2 and across beta = 0.68, 0.80, 0.92
    ax = axes[1]
    colors_b = [OKABE_ITO["purple"], OKABE_ITO["blue"], OKABE_ITO["green"]]
    markers_b = ["^", "o", "s"]
    for ib, b_val in enumerate(beta_sweep):
        lam_th_curve = 1.67 * (1.0 / float(b_val) - np.cos(np.radians(th_dense)))
        ax.plot(th_dense, lam_th_curve, "-", color=colors_b[ib], lw=1.4, label=rf"Theory $\beta={b_val:.2f}$")
        ax.plot(
            thetas_deg,
            lam_sim_beta_um[ib],
            linestyle="none",
            marker=markers_b[ib],
            ms=5.5,
            color=colors_b[ib],
            label=rf"Wakis PEC ($\beta={b_val:.2f}$)",
        )
    ax.plot(
        thetas_deg,
        lam_diel_um,
        linestyle="none",
        marker="D",
        ms=5.0,
        mfc="none",
        mec=OKABE_ITO["orange"],
        mew=1.4,
        label=r"Wakis $\mathrm{SiO}_2$ ($\beta=0.80$)",
    )
    ax.set_title(r"(b) Dispersion $\lambda = \frac{d}{n}(\frac{1}{\beta} - \cos\theta)$ vs $\beta$")
    ax.set_xlabel(r"Observation angle $\theta$ [deg]")
    ax.set_ylabel(r"Peak wavelength $\lambda_1(\theta)$ [$\mu\mathrm{m}$]")
    ax.grid(True)
    ax.legend(loc="upper left", ncol=1, fontsize=8.0, framealpha=0.92)

    # Panel (c): Peak spectral radiation vs angle for PEC grating, SiO2 grating, and flat mutation
    ax = axes[2]
    scale_p = max(float(np.max(pw_pec)), 1e-30)
    ax.plot(thetas_deg, pw_pec / scale_p, "-o", color=OKABE_ITO["blue"], lw=1.5, ms=5.0, label=r"PEC grating ($h_g=0.40\;\mu\mathrm{m}$)")
    ax.plot(thetas_deg, pw_diel / scale_p, "--s", color=OKABE_ITO["orange"], lw=1.5, ms=5.0, label=r"$\mathrm{SiO}_2$ grating ($\varepsilon_r=2.11$)")
    ax.plot(thetas_deg, pw_flat / scale_p, ":^", color=OKABE_ITO["grey"], lw=1.5, ms=5.0, label=r"Flat surface ($h_g=0$ mutation)")
    ax.set_title("(c) Angular Radiation Intensity & Flat Null")
    ax.set_xlabel(r"Observation angle $\theta$ [deg]")
    ax.set_ylabel("Normalized $n=1$ peak intensity [-]")
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.92)

    out = FIG_DIR / "03_sp_dispersion_and_angular_spectrum.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_04_sp_impact_parameter_decay(imp_npz: np.lib.npyio.NpzFile) -> Path:
    """Render Figure 4: Smith-Purcell evanescent decay exp(-4*pi*h/(beta*gamma*lambda)) vs impact parameter h."""
    _apply_paper_style()
    h_um = imp_npz["h_values_m"] * 1e6
    betas = imp_npz["betas"]
    lams_um = imp_npz["wavelengths_m"] * 1e6
    spec90 = imp_npz["spectra_90_beta08"]
    intens = imp_npz["intensities_vs_h"]
    alpha_sim = imp_npz["alpha_sim_1_m"] * 1e-6  # [1/um]
    alpha_th = imp_npz["alpha_theory_1_m"] * 1e-6  # [1/um]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), constrained_layout=True)

    # Panel (a): Broadside (theta = 90 deg) spectrum for each impact parameter h at beta = 0.80
    ax = axes[0]
    norm_s = max(float(np.max(spec90)), 1e-30)
    palette = [OKABE_ITO["blue"], OKABE_ITO["sky"], OKABE_ITO["green"], OKABE_ITO["orange"], OKABE_ITO["purple"]]
    ls_list = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for ih, hv in enumerate(h_um):
        ax.plot(
            lams_um,
            spec90[ih] / norm_s,
            linestyle=ls_list[ih % len(ls_list)],
            color=palette[ih % len(palette)],
            lw=1.5,
            label=rf"$h = {hv:.2f}\;\mu\mathrm{{m}}$",
        )
    ax.axvline(1.67 / 0.80, color="#222222", ls=":", lw=1.0, label=r"$\lambda_1(90^\circ)=d/\beta$")
    ax.set_title(r"(a) Broadside Spectrum ($\theta=90^\circ, \beta=0.80$)")
    ax.set_xlabel(r"Wavelength $\lambda$ [$\mu\mathrm{m}$]")
    ax.set_ylabel("Normalized intensity [-]")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    # Panel (b): Semi-log intensity vs h compared against exp(-4*pi*h/(beta*gamma*lambda))
    ax = axes[1]
    h_dense = np.linspace(float(h_um[0]), float(h_um[-1]), 100)
    cols = [OKABE_ITO["blue"], OKABE_ITO["orange"]]
    mks = ["o", "s"]
    for ib, b_val in enumerate(betas):
        y_norm = intens[ib] / intens[ib, 0]
        y_th = np.exp(-alpha_th[ib] * (h_dense - float(h_um[0])))
        ax.semilogy(
            h_dense,
            y_th,
            "-",
            color=cols[ib],
            lw=1.5,
            label=rf"Theory $\exp(-4\pi h/(\beta\gamma\lambda))$ ($\beta={b_val:.2f}$)",
        )
        ax.semilogy(
            h_um,
            y_norm,
            linestyle="none",
            marker=mks[ib],
            ms=6.0,
            color=cols[ib],
            label=rf"Wakis FIT ($\beta={b_val:.2f}, \alpha={alpha_sim[ib]:.2f}\;\mu\mathrm{{m}}^{{-1}}$)",
        )
    ax.set_title(r"(b) Evanescent Coupling Decay vs $h$")
    ax.set_xlabel(r"Impact parameter $h$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"Normalized power $I_{\mathrm{SP}}(h) / I_{\mathrm{SP}}(h_0)$ [-]")
    ax.grid(True, which="both")
    ax.legend(loc="lower left", fontsize=8.0, framealpha=0.92)

    # Panel (c): Relativistic dilation of decay constant alpha_h(beta) = 4*pi / (beta*gamma*lambda_90)
    ax = axes[2]
    b_curve = np.linspace(0.55, 0.96, 120)
    gamma_curve = 1.0 / np.sqrt(1.0 - b_curve**2)
    lam90_curve_um = 1.67 / b_curve
    alpha_curve_um = 4.0 * np.pi / (b_curve * gamma_curve * lam90_curve_um)
    ax.plot(b_curve, alpha_curve_um, "-", color=OKABE_ITO["black"], lw=1.6, label=r"Exact $\alpha_h = \frac{4\pi}{\beta\gamma\lambda_1(90^\circ)}$")
    ax.plot(betas, alpha_sim, "o", color=OKABE_ITO["orange"], ms=7.0, label="Wakis FIT fitted slope")
    for ib, b_val in enumerate(betas):
        err_pct = 100.0 * abs(alpha_sim[ib] - alpha_th[ib]) / alpha_th[ib]
        ax.annotate(
            rf"err = {err_pct:.2f}%",
            xy=(float(b_val), float(alpha_sim[ib])),
            xytext=(float(b_val) - 0.12, float(alpha_sim[ib]) + 0.55),
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.9),
        )
    ax.set_title(r"(c) Decay Constant $\alpha_h(\beta)$ & $\beta\gamma$ Dilation")
    ax.set_xlabel(r"Normalized electron velocity $\beta = v/c$ [-]")
    ax.set_ylabel(r"Power decay constant $\alpha_h$ [$\mu\mathrm{m}^{-1}$]")
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.92)

    out = FIG_DIR / "04_sp_impact_parameter_decay.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_05_dla_synchronism_and_nearfield(dla_npz: np.lib.npyio.NpzFile) -> Path:
    """Render Figure 5: DLA near-field map, cumulative energy gain, and synchronism period scan."""
    _apply_paper_style()
    x_2d_nm = dla_npz["x_2d_m"] * 1e9
    z_2d_nm = dla_npz["z_2d_m"] * 1e9
    ez_2d = dla_npz["Ez_2d_norm_real"]
    z_path_um = dla_npz["z_path_m"] * 1e6
    cum_sync = dla_npz["cum_gain_sync"]
    cum_det = dla_npz["cum_gain_detuned"]
    cum_flat = dla_npz["cum_gain_flat"]
    periods_nm = dla_npz["periods_scan_m"] * 1e9
    g_b100 = dla_npz["g_vs_period_b100"]
    g_b075 = dla_npz["g_vs_period_b075"]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), constrained_layout=True)

    # Panel (a): 2D near-field Re(Ez)/E0 in the Peralta et al. (2013) unit cell
    ax = axes[0]
    vmax = float(np.max(np.abs(ez_2d)))
    pcm = ax.pcolormesh(z_2d_nm, x_2d_nm, ez_2d, cmap="RdBu_r", vmin=-vmax, vmax=+vmax, shading="auto")
    # Overlay fused silica teeth ([-200, +200] nm in z, [200, 880] nm and [-880, -200] nm in x)
    ax.add_patch(Rectangle((-200.0, -880.0), 400.0, 680.0, fill=False, edgecolor="#111111", lw=1.4))
    ax.add_patch(Rectangle((-200.0, +200.0), 400.0, 680.0, fill=False, edgecolor="#111111", lw=1.4))
    ax.axhline(-880.0, color="#111111", lw=1.1)
    ax.axhline(+880.0, color="#111111", lw=1.1)
    ax.axhline(0.0, color=OKABE_ITO["orange"], ls="-.", lw=1.3)
    ax.text(0.0, -540.0, r"$\mathrm{SiO}_2$ tooth", ha="center", va="center", fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="none"))
    ax.text(0.0, +540.0, r"$\mathrm{SiO}_2$ tooth", ha="center", va="center", fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="none"))
    ax.text(0.0, 55.0, r"$e^-$ axis ($2a=400\;\mathrm{nm}$)", ha="center", va="bottom", fontsize=8.5, color=OKABE_ITO["orange"], fontweight="bold")
    ax.set_ylim(-1150.0, 1150.0)
    ax.set_title(r"(a) DLA Unit-Cell Field $\mathrm{Re}\{\tilde{E}_z\}/E_0$")
    ax.set_xlabel(r"$z$ [nm] ($\lambda_p = 800\;\mathrm{nm}$)")
    ax.set_ylabel(r"$x$ [nm]")
    fig.colorbar(pcm, ax=ax, shrink=0.86, label=r"$\mathrm{Re}\{\tilde{E}_z(x,z)\} / E_0$ [-]")

    # Panel (b): Cumulative wake-formalism energy gain across 6 periods
    ax = axes[1]
    ax.plot(z_path_um, cum_sync, "-", color=OKABE_ITO["blue"], lw=1.8, label=r"Synchronized ($\beta=1.0, \lambda_p=\beta\lambda_0$)")
    ax.plot(z_path_um, cum_det, "--", color=OKABE_ITO["orange"], lw=1.5, label=r"Detuned ($\beta=0.80$, phase slip)")
    ax.plot(z_path_um, cum_flat, ":", color=OKABE_ITO["grey"], lw=1.6, label=r"Flat surface ($h_g=0$, Lawson–Woodward)")
    ax.axhline(0.0, color="#444444", lw=0.7)
    ax.set_title(r"(b) Cumulative Energy Gain Along $e^-$ Path")
    ax.set_xlabel(r"Interaction distance $z$ [$\mu\mathrm{m}$] ($6\,\lambda_p$)")
    ax.set_ylabel(r"Normalized voltage gain $\Delta V(z) / (E_0 L_{\mathrm{int}})$ [-]")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

    # Panel (c): Accelerating gradient G/E0 vs grating period lam_p (synchronism curve)
    ax = axes[2]
    p_dense_nm = np.linspace(float(periods_nm[0]), float(periods_nm[-1]), 300)
    deph_100 = dla_synchronous_period(beta=1.0, wavelength_m=800e-9, period_m=p_dense_nm * 1e-9, n_periods=20)["dephasing_factor"]
    deph_075 = dla_synchronous_period(beta=0.75, wavelength_m=800e-9, period_m=p_dense_nm * 1e-9, n_periods=20)["dephasing_factor"]
    ax.plot(p_dense_nm, float(np.max(g_b100)) * deph_100, "-", color=OKABE_ITO["blue"], lw=1.3, label=r"Sinc envelope ($\beta=1.00, \lambda_{p,\mathrm{sync}}=800\;\mathrm{nm}$)")
    ax.plot(periods_nm, g_b100, "o", color=OKABE_ITO["blue"], ms=5.5, label=r"Wakis FIT ($\beta=1.00$, Peralta 2013)")
    ax.plot(p_dense_nm, float(np.max(g_b075)) * deph_075, "--", color=OKABE_ITO["orange"], lw=1.3, label=r"Sinc envelope ($\beta=0.75, \lambda_{p,\mathrm{sync}}=600\;\mathrm{nm}$)")
    ax.plot(periods_nm, g_b075, "s", color=OKABE_ITO["orange"], ms=5.5, label=r"Wakis FIT ($\beta=0.75$, sub-relativistic)")
    ax.axvline(800.0, color=OKABE_ITO["blue"], ls=":", lw=1.0)
    ax.axvline(600.0, color=OKABE_ITO["orange"], ls=":", lw=1.0)
    ax.set_title(r"(c) Synchronism Peak: $G(\lambda_p)/E_0$ vs Period")
    ax.set_xlabel(r"Grating period $\lambda_p$ [nm]")
    ax.set_ylabel(r"Multi-period accelerating factor $G / E_0$ [-]")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.8, framealpha=0.92)

    out = FIG_DIR / "05_dla_synchronism_and_nearfield.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_06_dla_depth_duty_and_peralta2013(dla_npz: np.lib.npyio.NpzFile) -> Path:
    """Render Figure 6: DLA accelerating factor vs tooth depth, duty cycle, and Peralta et al. (2013) Fig. 3b."""
    _apply_paper_style()
    depths_nm = dla_npz["depths_scan_m"] * 1e9
    g_depth = dla_npz["g_vs_depth"]
    duty_scan = dla_npz["duty_scan"]
    g_duty = dla_npz["g_vs_duty"]
    e0_gv_m = dla_npz["e0_gv_m"]
    grad_400 = dla_npz["grad_400_mev_m"]
    grad_800 = dla_npz["grad_800_mev_m"]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), constrained_layout=True)

    # Panel (a): G/E0 vs tooth depth h_g
    ax = axes[0]
    hg_dense_m = np.linspace(0.0, 1.0e-6, 200)
    rn_depth = dla_phase_grating_model(hg_dense_m, duty_cycle=0.50)["c1_amplitude"]
    rn_depth_scaled = rn_depth * (float(np.max(g_depth)) / max(float(np.max(rn_depth)), 1e-30))
    ax.plot(hg_dense_m * 1e9, rn_depth_scaled, "--", color=OKABE_ITO["grey"], lw=1.4, label=r"Scalar phase-grating $\propto |\sin(\pi(n-1)h_g/\lambda_0)|$")
    ax.plot(depths_nm, g_depth, "-o", color=OKABE_ITO["blue"], lw=1.6, ms=6.0, label=r"Wakis FIT ($\lambda_p=800\;\mathrm{nm}, \mathrm{DC}=0.50$)")
    ax.axvline(680.0, color=OKABE_ITO["orange"], ls=":", lw=1.1, label=r"Optimal depth $h_g\approx 680\;\mathrm{nm}$")
    ax.set_title(r"(a) Accelerating Factor vs Tooth Depth $h_g$")
    ax.set_xlabel(r"Fused-silica tooth depth $h_g$ [nm]")
    ax.set_ylabel(r"Accelerating structure factor $G / E_0$ [-]")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92)

    # Panel (b): G/E0 vs duty cycle DC
    ax = axes[1]
    dc_dense = np.linspace(0.0, 1.0, 200)
    rn_dc = np.sin(np.pi * dc_dense) * float(np.max(g_duty))
    ax.plot(dc_dense, rn_dc, "--", color=OKABE_ITO["grey"], lw=1.4, label=r"Fourier model $\propto \sin(\pi\,\mathrm{DC})$")
    ax.plot(duty_scan, g_duty, "-s", color=OKABE_ITO["green"], lw=1.6, ms=6.0, label=r"Wakis FIT ($h_g=680\;\mathrm{nm}$)")
    ax.set_title(r"(b) Accelerating Factor vs Duty Cycle")
    ax.set_xlabel(r"Tooth duty cycle $\mathrm{DC} = w_{\mathrm{tooth}} / \lambda_p$ [-]")
    ax.set_ylabel(r"Accelerating structure factor $G / E_0$ [-]")
    ax.grid(True)
    ax.legend(loc="lower center", fontsize=8.0, framealpha=0.92)

    # Panel (c): Comparison with Peralta et al., Nature 503, 91 (2013), Fig. 3b & Sapra et al., Science 367, 79 (2020)
    ax = axes[2]
    g400_fac = float(grad_400[-1]) / (float(e0_gv_m[-1]) * 1.0e3)
    g800_fac = float(grad_800[-1]) / (float(e0_gv_m[-1]) * 1.0e3)
    comp = peralta_2013_gradient_comparison(g400_fac, g800_fac)
    e0_curve = np.linspace(0.0, 2.0, 41)
    ax.plot(
        e0_curve,
        g400_fac * e0_curve * 1.0e3,
        "-",
        color=OKABE_ITO["blue"],
        lw=1.8,
        label=rf"Wakis FIT $2a=400\;\mathrm{{nm}}$ ($G/E_0={g400_fac:.3f}$)",
    )
    ax.plot(
        e0_curve,
        g800_fac * e0_curve * 1.0e3,
        "--",
        color=OKABE_ITO["sky"],
        lw=1.8,
        label=rf"Wakis FIT $2a=800\;\mathrm{{nm}}$ ($G/E_0={g800_fac:.3f}$)",
    )
    e0_319 = comp["e0_sio2_319uj_gv_m"]
    e0_319_err = comp["e0_sio2_319uj_err_gv_m"]
    g_pred_319 = comp["g_pred_400_sio2_319uj_mev_m"]
    g_pred_319_err = comp["g_pred_400_sio2_319uj_err_mev_m"]
    ax.errorbar(
        [e0_319],
        [g_pred_319],
        xerr=[e0_319_err],
        yerr=[g_pred_319_err],
        fmt="s",
        color=OKABE_ITO["blue"],
        mfc="white",
        mew=1.6,
        ms=6.5,
        capsize=3.5,
        label=rf"FIT pred. at $319\;\mu\mathrm{{J}}$: ${g_pred_319:.0f}\pm {g_pred_319_err:.0f}\;\mathrm{{MeV/m}}$",
    )
    ax.errorbar(
        [e0_319],
        [comp["g_exp_400_mev_m"]],
        xerr=[e0_319_err],
        yerr=[comp["g_exp_400_err_mev_m"]],
        fmt="o",
        color=OKABE_ITO["orange"],
        ms=6.5,
        capsize=3.5,
        label=r"Peralta (2013) $400\;\mathrm{nm}$: $309.8\pm 20.7\;\mathrm{MeV/m}$",
    )
    ax.errorbar(
        [e0_319],
        [comp["g_exp_800_mev_m"]],
        xerr=[e0_319_err],
        yerr=[comp["g_exp_800_err_mev_m"]],
        fmt="^",
        color=OKABE_ITO["purple"],
        ms=6.5,
        capsize=3.5,
        label=r"Peralta (2013) $800\;\mathrm{nm}$: ${\sim}152\pm 15\;\mathrm{MeV/m}$ (Fig. 3b)",
    )
    ax.axhline(
        30.5,
        color=OKABE_ITO["green"],
        ls=":",
        lw=1.3,
        label=r"Sapra et al. (2020) on-chip: $30.5\;\mathrm{MeV/m}$",
    )
    ax.text(
        0.97,
        0.06,
        (
            "Headline gap ratio $G(400)/G(800)$:\n"
            f"FIT: {g400_fac:.3f}/{g800_fac:.3f} = {comp['gap_ratio_sim_rounded']:.2f}\n"
            f"Exp.: 309.8/~152 = {comp['gap_ratio_exp']:.2f} ± {comp['gap_ratio_exp_err']:.2f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.94),
    )
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 720.0)
    ax.set_title("(c) Gap Ratio & Forward Prediction vs Peralta (2013)")
    ax.set_xlabel(r"Incident substrate field $E_0$ in $\mathrm{SiO}_2$ [GV/m]")
    ax.set_ylabel(r"Accelerating gradient $G$ [MeV/m]")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.0, framealpha=0.92)

    out = FIG_DIR / "06_dla_depth_duty_and_peralta2013.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_07_resolution_convergence(conv_npz: np.lib.npyio.NpzFile) -> Path:
    """Render Figure 7: 3-level grid convergence (Part 1 SP & Part 2 DLA) + space-harmonic diagram."""
    _apply_paper_style()
    dx_sp_nm = conv_npz["dx_sp_nm"]
    sp_ppw = conv_npz["sp_ppw_sio2"]
    sp_err = conv_npz["sp_lambda_mean_rel_err"] * 100.0
    dx_dla_nm = conv_npz["dx_dla_nm"]
    dla_ppw = conv_npz["dla_ppw_sio2"]
    dla_g = conv_npz["dla_g_over_e0"]
    dla_g_rich = float(conv_npz["dla_g_richardson"])

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), constrained_layout=True)

    # Panel (a): Smith-Purcell dispersion error vs resolution
    ax = axes[0]
    ax.plot(sp_ppw, sp_err, "-o", color=OKABE_ITO["blue"], lw=1.7, ms=6.5, label=r"Wakis FIT mean $|\Delta\lambda|/\lambda$")
    ax.axvline(20.0, color=OKABE_ITO["orange"], ls="--", lw=1.2, label=r"$\geq 20\;\mathrm{cells}/\lambda$ threshold")
    y_span = max(float(np.max(sp_err) - np.min(sp_err)), 0.8)
    ax.set_ylim(max(0.0, float(np.min(sp_err)) - 0.35 * y_span), float(np.max(sp_err)) + 0.55 * y_span)
    ax.set_xlim(float(np.min(sp_ppw)) - 2.0, float(np.max(sp_ppw)) + 2.5)
    for i, nm in enumerate(dx_sp_nm):
        ax.annotate(
            f"{nm:.1f} nm ({sp_err[i]:.2f}%)",
            xy=(float(sp_ppw[i]), float(sp_err[i])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )
    ax.set_title("(a) Part 1 · Smith–Purcell Grid Convergence")
    ax.set_xlabel(r"Resolution in $\mathrm{SiO}_2$ [$\mathrm{cells}/\lambda$]")
    ax.set_ylabel(r"Mean wavelength error $|\lambda_{\mathrm{sim}}-\lambda_{\mathrm{th}}|/\lambda_{\mathrm{th}}$ [%]")
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.92)

    # Panel (b): DLA synchronous accelerating factor convergence
    ax = axes[1]
    ax.plot(dla_ppw, dla_g, "-s", color=OKABE_ITO["green"], lw=1.7, ms=6.5, label=r"Wakis FIT $G / E_0$")
    ax.axhline(dla_g_rich, color=OKABE_ITO["black"], ls=":", lw=1.4, label=f"Richardson extrap. ({dla_g_rich:.4f})")
    ax.axvline(20.0, color=OKABE_ITO["orange"], ls="--", lw=1.2, label=r"$\geq 20\;\mathrm{cells}/\lambda_{\mathrm{SiO}_2}$")
    ax.set_title(r"(b) Part 2 · DLA Gradient Factor Convergence")
    ax.set_xlabel(r"Resolution in $\mathrm{SiO}_2$ at $\lambda_0=800\;\mathrm{nm}$ [$\mathrm{cells}/\lambda$]")
    ax.set_ylabel(r"Synchronous structure factor $G / E_0$ [-]")
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.92)

    # Panel (c): Space-harmonic (omega, k_z) diagram unifying SWS (6.2), Cherenkov (8.2), VLA (8.3), and SP/DLA (8.4)
    ax = axes[2]
    kz_norm = np.linspace(-1.2, 2.2, 300)
    # Light cone |kz| <= omega/c
    ax.fill_between(kz_norm, np.abs(kz_norm), 2.5, color="#E8F4FA", alpha=0.8, label=r"Vacuum light cone ($|k_z| \leq \omega/c$: SP radiation)")
    ax.plot(kz_norm, np.abs(kz_norm), "-", color=OKABE_ITO["black"], lw=1.5, label=r"Vacuum light line $\omega = c|k_z|$")
    beta_ex = 0.78
    ax.plot(kz_norm, beta_ex * np.maximum(kz_norm, 0.0), "--", color=OKABE_ITO["orange"], lw=1.6, label=rf"Bare $e^-$ line $\omega = \beta c k_z$ ($n=0$, evanescent)")
    ax.plot(kz_norm, beta_ex * (kz_norm + 1.0), "-", color=OKABE_ITO["blue"], lw=1.8, label=r"Space harmonic $n=+1$: $\omega = \beta c(k_z + 2\pi/d)$")
    ax.plot([0.0], [beta_ex], "o", color=OKABE_ITO["green"], ms=7.5, zorder=6, label=r"Normal-incidence DLA / $\theta=90^\circ$ SP ($k_z=0$)")
    ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(0.0, 1.8)
    ax.set_title(r"(c) Space-Harmonic Diagram ($\omega d / 2\pi c$ vs $k_z d / 2\pi$)")
    ax.set_xlabel(r"Normalized axial wavenumber $k_z d / (2\pi)$ [-]")
    ax.set_ylabel(r"Normalized frequency $\omega d / (2\pi c) = d / \lambda$ [-]")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

    out = FIG_DIR / "07_resolution_convergence.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_sp_animation_gif(sp_npz: np.lib.npyio.NpzFile) -> Path:
    """Render animated GIF (< 5 MB) comparing Smith-Purcell radiation on PEC vs SiO2 gratings."""
    _apply_paper_style()
    x_um = sp_npz["x_m"] * 1e6
    z_um = sp_npz["z_m"] * 1e6
    fr_pec = sp_npz["frames_Ez_pec"]
    fr_diel = sp_npz["frames_Ez_diel"]
    fr_zb = sp_npz["frames_zb_m"] * 1e6

    rad_mask = x_um >= 0.45
    vmax = float(np.percentile(np.abs(fr_pec[:, rad_mask, :]), 99.0))
    vmax = max(vmax, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True, sharey=True)
    n_per = int(round((z_um[-1] - z_um[0] - 6.0) / 1.67))

    ims = []
    dots = []
    titles = [
        r"(a) Metallic (PEC) Grating ($d=1.67\;\mu\mathrm{m}, \beta=0.80$)",
        r"(b) Fused-Silica Grating ($\varepsilon_r=2.11, n=1.453$)",
    ]
    mlabels = ["PEC grating", r"$\mathrm{SiO}_2$ grating ($\varepsilon_r=2.11$)"]
    fcols = ["#555555", "#56B4E9"]

    for idx, ax in enumerate(axes):
        field0 = fr_pec[0] if idx == 0 else fr_diel[0]
        im = ax.pcolormesh(z_um, x_um, field0, cmap="RdBu_r", vmin=-vmax, vmax=+vmax, shading="auto")
        _draw_sp_grating_overlay(
            ax,
            z_min_um=float(z_um[0]),
            z_max_um=float(z_um[-1]),
            x_min_um=float(x_um[0]),
            x_max_um=float(x_um[-1]),
            period_um=1.67,
            n_periods=n_per,
            depth_um=0.40,
            duty=0.50,
            mat_label=mlabels[idx],
            face_color=fcols[idx],
            cpml_um=0.50,
        )
        (dot,) = ax.plot([fr_zb[0]], [0.25], "o", color=OKABE_ITO["orange"], ms=6.5, mec="white", mew=0.8, zorder=8)
        ax.set_title(titles[idx])
        ax.set_xlabel(r"$z$ [$\mu\mathrm{m}$]")
        if idx == 0:
            ax.set_ylabel(r"$x$ [$\mu\mathrm{m}$]")
        ims.append(im)
        dots.append(dot)

    fig.colorbar(ims[0], ax=axes, shrink=0.86, label=r"Scattered $E_z^{\mathrm{SP}}(x, z, t)$ [V/m]")

    def _update(frame_idx: int):
        ims[0].set_array(fr_pec[frame_idx].ravel())
        ims[1].set_array(fr_diel[frame_idx].ravel())
        dots[0].set_data([fr_zb[frame_idx]], [0.25])
        dots[1].set_data([fr_zb[frame_idx]], [0.25])
        return [*ims, *dots]

    ani = animation.FuncAnimation(fig, _update, frames=len(fr_zb), interval=110, blit=False)
    out = FIG_DIR / "smith_purcell_pec_vs_dielectric.gif"
    ani.save(out, writer=animation.PillowWriter(fps=9), dpi=115)
    plt.close(fig)
    return out


def render_dla_animation_gif(dla_npz: np.lib.npyio.NpzFile) -> Path:
    """Render animated GIF (< 5 MB) of synchronous electron acceleration in the DLA grating."""
    _apply_paper_style()
    x_um = dla_npz["track_x_m"] * 1e6
    z_um = dla_npz["track_z_m"] * 1e6
    fr_ez = dla_npz["frames_dla_Ez"]
    fr_ze_um = dla_npz["frames_dla_ze_m"] * 1e6
    z_path_um = dla_npz["z_path_m"] * 1e6
    cum_sync = dla_npz["cum_gain_sync"]
    cum_flat = dla_npz["cum_gain_flat"]

    x_mask = np.abs(x_um) <= 1.25
    x_sub = x_um[x_mask]
    vmax = float(np.max(np.abs(fr_ez[:, x_mask, :])))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)

    ax0 = axes[0]
    im = ax0.pcolormesh(z_um, x_sub, fr_ez[0, x_mask, :], cmap="RdBu_r", vmin=-vmax, vmax=+vmax, shading="auto")
    for m in range(6):
        zc = (m + 0.5) * 0.80
        ax0.add_patch(Rectangle((zc - 0.20, -0.88), 0.40, 0.68, fill=False, edgecolor="#111111", lw=1.2))
        ax0.add_patch(Rectangle((zc - 0.20, +0.20), 0.40, 0.68, fill=False, edgecolor="#111111", lw=1.2))
    ax0.axhline(-0.88, color="#111111", lw=1.0)
    ax0.axhline(+0.88, color="#111111", lw=1.0)
    ax0.axhline(0.0, color=OKABE_ITO["orange"], ls="-.", lw=1.2)
    ax0.text(0.25, -1.08, r"$\mathrm{SiO}_2$ ($\varepsilon_r=2.11$), CPML $\pm x$", fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.88, edgecolor="none"))
    (e_dot,) = ax0.plot([fr_ze_um[0]], [0.0], "o", color=OKABE_ITO["orange"], ms=7.5, mec="white", mew=1.0, zorder=8)
    ax0.set_title(r"(a) DLA Field $E_z(x, z, t)/E_0$ & Surfing $e^-$")
    ax0.set_xlabel(r"$z$ [$\mu\mathrm{m}$] ($\lambda_p = 0.80\;\mu\mathrm{m}$)")
    ax0.set_ylabel(r"$x$ [$\mu\mathrm{m}$]")
    fig.colorbar(im, ax=ax0, shrink=0.86, label=r"$E_z(x, z, t) / E_0$ [-]")

    ax1 = axes[1]
    ax1.plot(z_path_um, cum_sync, "--", color="#BBBBBB", lw=1.1, label="Full synchronized trajectory")
    ax1.plot(z_path_um, cum_flat, ":", color=OKABE_ITO["grey"], lw=1.4, label=r"Flat surface ($h_g=0$, Lawson–Woodward)")
    (line_sync,) = ax1.plot([z_path_um[0]], [cum_sync[0]], "-", color=OKABE_ITO["blue"], lw=2.2, label=r"Cumulative gain $\Delta V(z) / (E_0 L)$")
    (gain_dot,) = ax1.plot([z_path_um[0]], [cum_sync[0]], "o", color=OKABE_ITO["orange"], ms=7.0)
    ax1.set_xlim(0.0, float(z_path_um[-1]))
    ax1.set_ylim(-0.04, float(np.max(cum_sync)) * 1.15)
    ax1.set_title(r"(b) Synchronous Energy Gain vs Flat Null")
    ax1.set_xlabel(r"Electron position $z_e(t)$ [$\mu\mathrm{m}$]")
    ax1.set_ylabel(r"Normalized energy gain $\Delta V(z) / (E_0 L_{\mathrm{int}})$ [-]")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

    def _update(idx: int):
        im.set_array(fr_ez[idx, x_mask, :].ravel())
        ze = float(fr_ze_um[idx])
        e_dot.set_data([ze], [0.0])
        k_end = max(1, int(np.searchsorted(z_path_um, ze)))
        line_sync.set_data(z_path_um[:k_end], cum_sync[:k_end])
        gain_dot.set_data([z_path_um[k_end - 1]], [cum_sync[k_end - 1]])
        return [im, e_dot, line_sync, gain_dot]

    ani = animation.FuncAnimation(fig, _update, frames=len(fr_ze_um), interval=120, blit=False)
    out = FIG_DIR / "dla_synchronous_acceleration.gif"
    ani.save(out, writer=animation.PillowWriter(fps=8), dpi=115)
    plt.close(fig)
    return out


def render_all_figures() -> list[Path]:
    """Render all 7 diagnostic PNG figures and 2 animated GIFs from precomputed data/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sp_npz = np.load(DATA_DIR / "task_smith_purcell.npz")
    imp_npz = np.load(DATA_DIR / "task_impact_decay.npz")
    dla_npz = np.load(DATA_DIR / "task_dla_grating.npz")
    conv_npz = np.load(DATA_DIR / "task_convergence.npz")

    paths = [
        plot_01_setup_schematic_and_mesh(),
        plot_02_sp_field_maps(sp_npz),
        plot_03_sp_dispersion_and_angular_spectrum(sp_npz),
        plot_04_sp_impact_parameter_decay(imp_npz),
        plot_05_dla_synchronism_and_nearfield(dla_npz),
        plot_06_dla_depth_duty_and_peralta2013(dla_npz),
        plot_07_resolution_convergence(conv_npz),
        render_sp_animation_gif(sp_npz),
        render_dla_animation_gif(dla_npz),
    ]
    return paths
