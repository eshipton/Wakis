"""Physics, 2D FIT runners, and publication plotting helpers for Tutorial 7.8 (Microdisk WGM Laser)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from scipy.constants import c as c_light
from scipy.constants import epsilon_0, hbar, mu_0
from scipy.optimize import brentq, root
from scipy.special import hankel1, jv, jvp

from wakis import GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.estimate import estimate, preflight
from wakis.field_monitors import DFTMonitor
from wakis.gain import FourLevelGain
from wakis.resonances import ResonanceMonitor, harminv
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.shapes import Box, Cylinder

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

Z0_VAC = float(np.sqrt(mu_0 / epsilon_0))
DEFAULT_MEM_LIMIT_MB = 1800.0

# Default microdisk geometry (McCall et al., Appl. Phys. Lett. 60, 289, 1992)
DEFAULT_R_DISK_M = 1.0e-6       # Microdisk radius R = 1.0 um (diameter D = 2.0 um)
DEFAULT_N_DISK = 2.0            # Effective refractive index n_disk = 2.0 (eps_r = 4.0)
DEFAULT_N_OUT = 1.0             # Cladding refractive index (air, n_out = 1.0)
DEFAULT_M_TARGET = 6            # Target azimuthal WGM order m = 6
DEFAULT_Q_TARGET = 1            # Target radial WGM order q = 1
DEFAULT_R_IN_FRAC = 0.55        # Inner radius fraction R_in / R for annular ring pump [-]
DEFAULT_BUS_WIDTH_M = 0.22e-6   # Evanescently coupled bus waveguide width w_bus [m]
DEFAULT_BUS_GAP_M = 0.30e-6     # Edge-to-edge coupling gap g_bus between disk rim and bus [m]

# Default 4-level atomic gain parameters (with F11b time_scale = 100 <= 100)
DEFAULT_N_TOTAL = 5.0e23        # Total active density N_total [1/m^3] (5e17 cm^-3)
DEFAULT_GAMMA_A_HZ = 10.0e12    # Homogeneous linewidth gamma_a / (2*pi) [Hz] (<< 28.5 THz WGM FSR)
DEFAULT_GAMMA_A = 2.0 * np.pi * DEFAULT_GAMMA_A_HZ
DEFAULT_TIME_SCALE = 100.0      # Rate-equation lifetime acceleration factor s <= 100 [-]
DEFAULT_TAU_21 = 6.0e-12        # Unscaled upper-level lifetime tau_21 [s] (effective tau_21 / s = 60 fs)
DEFAULT_TAU_10 = 200.0e-15      # Unscaled lower-level lifetime tau_10 [s] (effective tau_10 / s = 2 fs, tau_21 / tau_10 = 30)
DEFAULT_TAU_32 = 200.0e-15      # Unscaled pump-level relaxation tau_32 [s] (effective tau_32 / s = 2 fs)
DEFAULT_TAU_21_PHYS = 6.0e-12   # Unscaled physical upper-state lifetime [s]
DEFAULT_TAU_10_PHYS = 200.0e-15 # Unscaled physical lower-state lifetime [s]
DEFAULT_TAU_32_PHYS = 200.0e-15 # Unscaled physical pump-relaxation lifetime [s]
DEFAULT_NOISE_AMP = 0.10        # FDT spontaneous-emission noise multiplier for 2D cell volume [-]
DEFAULT_NOISE_SEED = 42         # Reproducible RNG seed for Langevin polarization noise [-]


# ---------------------------------------------------------------------------
# Analytical theory helpers: 2D Bessel/Hankel WGM complex roots & confinement
# ---------------------------------------------------------------------------


def _h1p(m: int, z: complex | np.ndarray) -> complex | np.ndarray:
    """Derivative dH_m^(1)(z)/dz of the Hankel function of the first kind."""
    return 0.5 * (hankel1(m - 1, z) - hankel1(m + 1, z))


def wgm_characteristic_residual(
    x: complex,
    m: int,
    n_disk: float = DEFAULT_N_DISK,
    n_out: float = DEFAULT_N_OUT,
    polarization: str = "TM",
) -> complex:
    """Evaluate the exact 2D dielectric-cylinder WGM characteristic determinant.

    For a circular dielectric cylinder of radius ``R`` and refractive index
    ``n_disk`` in a background medium ``n_out``, continuity of tangential E and H
    at ``r = R`` for azimuthal order ``m`` (fields proportional to ``exp(i m phi - i omega t)``)
    yields the transcendental characteristic equation in ``x = k_0 R = omega R / c``:

    * **TM** (``E_z`` out of plane, ``H_r, H_phi`` in plane):
      ``n_disk * J_m'(n_disk * x) * H_m^(1)(n_out * x) - n_out * J_m(n_disk * x) * H_m^(1)'(n_out * x) = 0``
    * **TE** (``H_z`` out of plane, ``E_r, E_phi`` in plane):
      ``n_out * J_m'(n_disk * x) * H_m^(1)(n_out * x) - n_disk * J_m(n_disk * x) * H_m^(1)'(n_out * x) = 0``

    Parameters
    ----------
    x : complex
        Dimensionless complex size parameter ``x = k_0 * R = (omega * R / c)`` [-],
        with ``Im(x) < 0`` for outgoing leaky modes in the ``exp(-i omega t)`` convention.
    m : int
        Azimuthal mode order ``m >= 0`` [-].
    n_disk : float, optional
        Disk refractive index [-].
    n_out : float, optional
        Exterior cladding refractive index [-].
    polarization : {'TM', 'TE'}, optional
        Polarization state (``'TM'`` has ``E_z`` along the cylinder axis;
        ``'TE'`` has ``H_z`` along the cylinder axis).

    Returns
    -------
    res : complex
        Complex residual of the boundary-matching determinant [-].
    """
    m_int = int(m)
    if m_int < 0:
        raise ValueError(f"Azimuthal order m must be >= 0, got {m}")
    nd = float(n_disk)
    no = float(n_out)
    pol = polarization.strip().upper()
    if pol not in ("TM", "TE"):
        raise ValueError(f"polarization must be 'TM' or 'TE', got {polarization!r}")

    z_in = nd * complex(x)
    z_out = no * complex(x)
    jm = jv(m_int, z_in)
    jmp = jvp(m_int, z_in, 1)
    hm = hankel1(m_int, z_out)
    hmp = _h1p(m_int, z_out)

    if pol == "TM":
        return complex(nd * jmp * hm - no * jm * hmp)
    return complex(no * jmp * hm - nd * jm * hmp)


def _initial_wgm_guesses(
    m: int,
    n_disk: float,
    n_out: float,
    polarization: str,
    n_roots: int = 3,
) -> list[complex]:
    """Bracket real size parameter intervals and polish complex roots for radial orders q = 1..n_roots."""
    x_grid = np.linspace(max(0.4, 0.35 * m), (m + 4.5 * n_roots + 4.0) / n_disk, 4000)
    # Use the real part of the characteristic equation (with Y_m contribution suppressed near real axis)
    # by scanning local minima of |wgm_characteristic_residual(x - 0.01j)|
    vals = np.array(
        [abs(wgm_characteristic_residual(xv - 0.015j, m, n_disk, n_out, polarization)) for xv in x_grid]
    )
    roots_found: list[complex] = []
    for idx in range(1, len(x_grid) - 1):
        if vals[idx] <= vals[idx - 1] and vals[idx] <= vals[idx + 1]:
            x0 = float(x_grid[idx])
            sol = root(
                lambda v: [
                    wgm_characteristic_residual(v[0] + 1j * v[1], m, n_disk, n_out, polarization).real,
                    wgm_characteristic_residual(v[0] + 1j * v[1], m, n_disk, n_out, polarization).imag,
                ],
                [x0, -0.02],
                tol=1e-13,
            )
            if sol.success and sol.x[0] > 0.2 and sol.x[1] < -1e-9:
                zc = complex(sol.x[0], sol.x[1])
                if abs(wgm_characteristic_residual(zc, m, n_disk, n_out, polarization)) < 1e-9:
                    if not any(abs(zc.real - r.real) < 0.15 for r in roots_found):
                        roots_found.append(zc)
                        if len(roots_found) >= n_roots:
                            break
    roots_found.sort(key=lambda z: z.real)
    return roots_found


def wgm_radial_profile_and_confinement(
    m: int,
    x_root: complex,
    r_disk_m: float = DEFAULT_R_DISK_M,
    n_disk: float = DEFAULT_N_DISK,
    n_out: float = DEFAULT_N_OUT,
    polarization: str = "TM",
    r_in_frac: float = DEFAULT_R_IN_FRAC,
    r_max_frac: float = 1.45,
    n_pts: int = 600,
) -> dict[str, float | np.ndarray]:
    """Compute the radial WGM field profile and electric-energy confinement factors.

    For a leaky WGM resonance with complex size parameter ``x = k_0 R``, the
    radial dependence is ``J_m(n_disk k_0 r)`` for ``r <= R`` and
    ``[J_m(n_disk x) / H_m^(1)(n_out x)] H_m^(1)(n_out k_0 r)`` for ``r > R``.
    The modal electric-energy confinement factor in a radial annulus ``[R_1, R_2]`` is
    (derived here from complex Poynting / perturbation theory):

        ``Gamma(R_1, R_2) = int_{R_1}^{R_2} eps_r(r) |E(r)|^2 r dr / int_0^{R_rad} eps_r(r) |E(r)|^2 r dr``

    where ``R_rad = r_max_frac * R`` lies inside the radiation caustic ``m / k_0``.

    Parameters
    ----------
    m : int
        Azimuthal mode order ``m >= 0`` [-].
    x_root : complex
        Complex dimensionless root ``x = k_0 R`` [-].
    r_disk_m : float, optional
        Disk radius ``R`` [m].
    n_disk : float, optional
        Disk refractive index [-].
    n_out : float, optional
        Exterior refractive index [-].
    polarization : {'TM', 'TE'}, optional
        Polarization state.
    r_in_frac : float, optional
        Inner radius fraction ``R_in / R`` of the annular pump ring [-].
    r_max_frac : float, optional
        Outer truncation radius ``R_max / R`` for normalization of the quasi-normal mode [-].
    n_pts : int, optional
        Number of radial quadrature samples [-].

    Returns
    -------
    out : dict
        Dictionary with ``r_m`` [m], ``u_norm`` [-], ``eps_r`` [-], ``confinement_disk`` [-],
        ``confinement_ring`` [-], and ``confinement_core`` [-].
    """
    R = float(r_disk_m)
    nd = float(n_disk)
    no = float(n_out)
    pol = polarization.strip().upper()
    r_arr = np.linspace(1e-9 * R, float(r_max_frac) * R, int(n_pts))
    u_r = r_arr / R
    inside = u_r <= 1.0

    xr = float(np.real(x_root))
    if pol == "TM":
        e_in = jv(m, nd * xr * u_r[inside])
        scale_out = jv(m, nd * xr) / hankel1(m, no * x_root)
        e_out = scale_out * hankel1(m, no * x_root * u_r[~inside])
        e_sq = np.empty_like(r_arr)
        e_sq[inside] = np.abs(e_in) ** 2
        e_sq[~inside] = np.abs(e_out) ** 2
    else:
        # For TE (H_z out of plane), |E_r|^2 + |E_phi|^2 is proportional to
        # (1 / eps_r^2) * (|m H_z / (k_0 r)|^2 + |dH_z / (k_0 dr)|^2)
        hz_in = jv(m, nd * xr * u_r[inside])
        dhz_in = nd * jvp(m, nd * xr * u_r[inside], 1)
        scale_out = jv(m, nd * xr) / hankel1(m, no * x_root)
        hz_out = scale_out * hankel1(m, no * x_root * u_r[~inside])
        dhz_out = scale_out * no * _h1p(m, no * x_root * u_r[~inside])
        e_sq = np.empty_like(r_arr)
        e_sq[inside] = (np.abs(m * hz_in / (xr * u_r[inside])) ** 2 + np.abs(dhz_in) ** 2) / (nd**4)
        e_sq[~inside] = (np.abs(m * hz_out / (xr * u_r[~inside])) ** 2 + np.abs(dhz_out) ** 2) / (no**4)

    eps_prof = np.where(inside, nd**2, no**2)
    integrand = eps_prof * e_sq * r_arr
    total_energy = float(np.trapezoid(integrand, r_arr))
    disk_energy = float(np.trapezoid(integrand[inside], r_arr[inside]))
    gamma_disk_trunc = disk_energy / max(total_energy, 1e-30)

    ring_mask = inside & (u_r >= float(r_in_frac))
    core_mask = inside & (u_r < float(r_in_frac))
    ring_energy = float(np.trapezoid(integrand[ring_mask], r_arr[ring_mask]))
    core_energy = float(np.trapezoid(integrand[core_mask], r_arr[core_mask]))

    # Exact QNM disk confinement factor from the implicit derivative dx / dn_disk of F_m(x, n_disk) = 0
    eps_step = 1e-7
    df_dx = (
        wgm_characteristic_residual(x_root + eps_step, m, nd, no, pol)
        - wgm_characteristic_residual(x_root - eps_step, m, nd, no, pol)
    ) / (2.0 * eps_step)
    df_dn = (
        wgm_characteristic_residual(x_root, m, nd + eps_step, no, pol)
        - wgm_characteristic_residual(x_root, m, nd - eps_step, no, pol)
    ) / (2.0 * eps_step)
    dx_dn = -df_dn / df_dx
    gamma_disk = float(-(nd / xr) * np.real(dx_dn))

    ring_frac = ring_energy / max(disk_energy, 1e-30)
    core_frac = core_energy / max(disk_energy, 1e-30)

    u_norm = e_sq / max(float(np.max(e_sq[inside])), 1e-30)
    return {
        "r_m": r_arr,
        "u_norm": u_norm,
        "eps_r": eps_prof,
        "confinement_disk": gamma_disk,
        "confinement_disk_trunc": float(gamma_disk_trunc),
        "confinement_ring": gamma_disk * ring_frac,
        "confinement_core": gamma_disk * core_frac,
    }


def solve_wgm_complex_root(
    m: int,
    q: int = 1,
    r_disk_m: float = DEFAULT_R_DISK_M,
    n_disk: float = DEFAULT_N_DISK,
    n_out: float = DEFAULT_N_OUT,
    polarization: str = "TM",
    r_in_frac: float = DEFAULT_R_IN_FRAC,
) -> dict[str, float | int | str]:
    """Solve for the exact complex Bessel/Hankel root and threshold gain of WGM ``(m, q)``.

    Derives the resonant frequency ``f_{m,q} = c * Re(x_{m,q}) / (2 pi R)``,
    radiative quality factor ``Q_{m,q} = -Re(x_{m,q}) / (2 Im(x_{m,q}))``,
    cold-cavity amplitude decay rate ``gamma_{rad} = -c * Im(x_{m,q}) / R``,
    modal confinement factor ``Gamma_{m,q}``, and threshold power gain coefficient
    (derived here):

        ``g_th = (n_disk * omega_{m,q}) / (c * Gamma_{m,q} * Q_{m,q})``

    Parameters
    ----------
    m : int
        Azimuthal mode order ``m >= 1`` [-].
    q : int, optional
        Radial mode order ``q >= 1`` (``q = 1`` is the fundamental rim-hugging mode) [-].
    r_disk_m : float, optional
        Disk radius ``R`` [m].
    n_disk : float, optional
        Disk refractive index [-].
    n_out : float, optional
        Cladding refractive index [-].
    polarization : {'TM', 'TE'}, optional
        Polarization state.
    r_in_frac : float, optional
        Inner radius fraction ``R_in / R`` of the annular ring pump [-].

    Returns
    -------
    out : dict
        Dictionary with exact complex root, frequency [Hz, THz], wavelength [m],
        radiative ``Q``, decay rate [1/s], confinement factors [-], and threshold
        gain ``g_th`` [1/m, 1/cm].
    """
    if int(q) < 1:
        raise ValueError(f"Radial order q must be >= 1, got {q}")
    roots = _initial_wgm_guesses(int(m), float(n_disk), float(n_out), polarization, n_roots=int(q) + 1)
    if len(roots) < int(q):
        raise RuntimeError(f"Could not locate radial order q={q} for m={m}, pol={polarization}")
    zc = roots[int(q) - 1]
    xr = float(zc.real)
    xi = float(-zc.imag)  # positive magnitude of -Im(x)
    R = float(r_disk_m)
    k0_r = xr / R
    omega = c_light * k0_r
    freq_hz = omega / (2.0 * np.pi)
    lam_m = c_light / freq_hz
    q_rad = xr / (2.0 * xi)
    gamma_rad = c_light * xi / R  # amplitude decay rate -Im(omega) [1/s]

    prof = wgm_radial_profile_and_confinement(
        m=int(m),
        x_root=zc,
        r_disk_m=R,
        n_disk=float(n_disk),
        n_out=float(n_out),
        polarization=polarization,
        r_in_frac=r_in_frac,
    )
    gamma_disk = float(prof["confinement_disk"])
    gamma_disk_trunc = float(prof["confinement_disk_trunc"])
    gamma_ring = float(prof["confinement_ring"])
    gamma_core = float(prof["confinement_core"])

    g_th_unconf_1_m = float(n_disk) * omega / (c_light * q_rad)
    g_th_disk_1_m = g_th_unconf_1_m / gamma_disk
    g_th_ring_1_m = g_th_unconf_1_m / gamma_ring

    return {
        "m": int(m),
        "q": int(q),
        "polarization": polarization.strip().upper(),
        "x_real": xr,
        "x_imag": xi,
        "frequency_hz": float(freq_hz),
        "frequency_thz": float(freq_hz * 1e-12),
        "wavelength_m": float(lam_m),
        "q_rad": float(q_rad),
        "decay_rate_1_s": float(gamma_rad),
        "confinement_disk": gamma_disk,
        "confinement_disk_trunc": gamma_disk_trunc,
        "confinement_ring": gamma_ring,
        "confinement_core": gamma_core,
        "g_th_1_m": float(g_th_disk_1_m),
        "g_th_cm1": float(g_th_disk_1_m * 1e-2),
        "g_th_ring_1_m": float(g_th_ring_1_m),
        "g_th_ring_cm1": float(g_th_ring_1_m * 1e-2),
        "g_th_unconfined_1_m": float(g_th_unconf_1_m),
        "g_th_unconfined_cm1": float(g_th_unconf_1_m * 1e-2),
        "residual_abs": float(abs(wgm_characteristic_residual(zc, m, n_disk, n_out, polarization))),
    }


def wgm_four_level_li_curve(
    pump_ratios: float | np.ndarray,
    g_th_1_m: float,
    omega_a: float,
    gamma_a: float = DEFAULT_GAMMA_A,
    tau_21: float = DEFAULT_TAU_21,
    tau_10: float = DEFAULT_TAU_10,
    tau_32: float = DEFAULT_TAU_32,
    N_total: float = DEFAULT_N_TOTAL,
    n_disk: float = DEFAULT_N_DISK,
    kappa_a: float | None = None,
    m_order: int = DEFAULT_M_TARGET,
    x_root_real: float = 4.213883,
    r_in_frac: float = DEFAULT_R_IN_FRAC,
) -> dict[str, float | np.ndarray]:
    """Compute the analytical 4-level WGM laser L-pump curve with and without spatial hole burning.

    In a microdisk of refractive index ``n_disk``, a standing-wave whispering-gallery
    mode of azimuthal order ``m`` has dimensionless intensity profile
    ``u(r, phi) = [J_m(n_disk x_{m,q} r / R)^2 / J_{max}^2] * cos^2(m phi)`` inside the
    pumped annulus ``r in [R_in, R]``. Under steady-state oscillation at
    normalized peak intensity ``s = I_peak / I_sat``, the saturated inversion is
    ``Delta N(r, phi) = Delta N_0(W_p) / (1 + s * u(r, phi))``. Gain clamping requires
    the modal overlap integral of ``Delta N(r, phi)`` with ``u(r, phi)`` to equal
    ``Delta N_th = g_th / sigma_21`` (derived here):

    * **Spatially homogenized gain** (``spatial_averaging=True``, fast carrier diffusion):
      ``s_hom(r_p) = max(0, Delta N_0(W_p) / Delta N_th - 1)``
    * **2D WGM spatial hole burning** (``spatial_averaging=False``, localized carriers):
      Solved for both a single azimuthal standing wave ``u(r) cos^2(m phi)`` (``s_shb``)
      and the radial / dual-partner WGM profile ``u(r)`` (``s_rad_ring``) over ``r in [R_in, R]``.
    """
    fg_ref = FourLevelGain(
        omega_a=float(omega_a),
        gamma_a=float(gamma_a),
        tau_21=float(tau_21),
        tau_10=float(tau_10),
        tau_32=float(tau_32),
        N_total=float(N_total),
        pump_rate=0.0,
        eps_inf=float(n_disk) ** 2,
        kappa_a=kappa_a,
    )
    wp_th = float(fg_ref.threshold_pump_rate(float(g_th_1_m)))
    sigma_21 = float(fg_ref.emission_cross_section)
    dn_th = float(g_th_1_m) / sigma_21
    i_sat = float(fg_ref.saturation_intensity(W_p=wp_th))

    r_arr = np.atleast_1d(np.asarray(pump_ratios, dtype=float))
    u_r_grid = np.linspace(float(r_in_frac), 1.0, 240)
    jm_sq = np.abs(jv(int(m_order), float(n_disk) * float(x_root_real) * u_r_grid)) ** 2
    u_rad = jm_sq / max(float(np.max(jm_sq)), 1e-30)
    norm_rad = float(np.trapezoid(0.5 * u_rad * u_r_grid, u_r_grid))
    ring_area = float(np.trapezoid(u_r_grid, u_r_grid))
    u_rad_mean = float(np.trapezoid(u_rad * u_r_grid, u_r_grid)) / max(ring_area, 1e-30)

    s_hom = np.zeros_like(r_arr)
    s_shb = np.zeros_like(r_arr)
    s_rad_ring = np.zeros_like(r_arr)
    s_shb_ring = np.zeros_like(r_arr)
    dn_clamped = np.zeros_like(r_arr)

    for idx, rp in enumerate(r_arr):
        wp = max(0.0, float(rp) * wp_th)
        dn0 = float(fg_ref.steady_state_inversion(W_p=wp, intensity=0.0))
        inv_ratio = dn0 / dn_th
        if inv_ratio <= 1.0:
            s_hom[idx] = 0.0
            s_shb[idx] = 0.0
            s_rad_ring[idx] = 0.0
            s_shb_ring[idx] = 0.0
            dn_clamped[idx] = dn0
        else:
            s_hom[idx] = inv_ratio - 1.0
            dn_clamped[idx] = dn_th

            def _shb_res(s_val: float) -> float:
                if s_val <= 1e-12:
                    return inv_ratio - 1.0
                su = np.maximum(s_val * u_rad, 1e-15)
                azi_avg = (1.0 - 1.0 / np.sqrt(1.0 + su)) / su
                modal_gain_ratio = inv_ratio * float(np.trapezoid(azi_avg * u_rad * u_r_grid, u_r_grid)) / norm_rad
                return modal_gain_ratio - 1.0

            def _rad_res(s_val: float) -> float:
                if s_val <= 1e-12:
                    return inv_ratio - 1.0
                su = s_val * u_rad
                modal_gain_ratio = inv_ratio * float(np.trapezoid((u_rad / (1.0 + su)) * u_r_grid, u_r_grid)) / (2.0 * norm_rad)
                return modal_gain_ratio - 1.0

            s_shb[idx] = float(brentq(_shb_res, 0.0, max(20.0 * (inv_ratio - 1.0), 10.0), rtol=1e-10))
            s_rad_pk = float(brentq(_rad_res, 0.0, max(20.0 * (inv_ratio - 1.0), 10.0), rtol=1e-10))
            s_rad_ring[idx] = s_rad_pk * u_rad_mean
            s_shb_ring[idx] = 0.5 * s_shb[idx] * u_rad_mean

    return {
        "pump_ratios": r_arr,
        "W_p_th_1_s": wp_th,
        "sigma_21_m2": sigma_21,
        "dN_th_m3": dn_th,
        "I_sat_w_m2": i_sat,
        "s_hom": s_hom,
        "s_shb": s_shb,
        "s_rad_ring": s_rad_ring,
        "s_shb_ring": s_shb_ring,
        "I_hom_w_m2": s_hom * i_sat,
        "I_shb_w_m2": s_shb * i_sat,
        "I_rad_ring_w_m2": s_rad_ring * i_sat,
        "dN_clamped_m3": dn_clamped,
    }


def lifetime_scaling_laws(
    scale_factor: float = DEFAULT_TIME_SCALE,
    tau_21_phys: float = DEFAULT_TAU_21_PHYS,
    tau_10_phys: float = DEFAULT_TAU_10_PHYS,
    tau_32_phys: float = DEFAULT_TAU_32_PHYS,
    omega_a: float = 2.0 * np.pi * 201.059e12,
    gamma_a: float = DEFAULT_GAMMA_A,
    n_disk: float = DEFAULT_N_DISK,
    N_total: float = DEFAULT_N_TOTAL,
    g_th_1_m: float = 1.4477e5,
    q_cold: float = 64.04,
) -> dict[str, float]:
    """Verify the exact mathematical scaling laws for ``FourLevelGain(..., time_scale=S)`` (S <= 100).

    Compares both:
    1. Built-in F11b rate-equation time scaling ``FourLevelGain(..., time_scale=S)`` with
       ``S <= 100``, where ``tau_ij -> tau_ij / S``, ``W_p -> S * W_p``, and
       ``W_stim -> S * W_stim`` while holding ``kappa_a`` fixed at ``tau_21_phys``:
       - ``sigma_21``, ``Delta N_th``, ``W_{p,th}``, ``I_sat``, and steady-state ``I_out``
         are all **strictly invariant** (``S^0``).
       - Turn-on delay ``t_th(S) = t_th(1) / S`` and effective saturation recovery time
         ``tau_sat,eff(S) = tau_sat(1) / S`` scale as ``S^(-1)``.
       - Undamped relaxation-oscillation frequency ``omega_RO,0(S)`` scales as ``S^(1/2)``.
    2. Direct lifetime replacement ``(tau_21, tau_10, tau_32) -> (tau_21, tau_10, tau_32) / S``
       at fixed ``kappa_a`` without scaling ``W_stim``:
       - ``sigma_21`` and ``Delta N_th`` are invariant (``S^0``), while ``W_{p,th}`` and
         ``I_sat`` scale linearly with ``S`` (``S^1``).
    """
    S = float(scale_factor)
    fg_phys = FourLevelGain(
        omega_a=omega_a,
        gamma_a=gamma_a,
        tau_21=tau_21_phys,
        tau_10=tau_10_phys,
        tau_32=tau_32_phys,
        N_total=N_total,
        eps_inf=n_disk**2,
        time_scale=1.0,
    )
    fg_time_scaled = FourLevelGain(
        omega_a=omega_a,
        gamma_a=gamma_a,
        tau_21=tau_21_phys,
        tau_10=tau_10_phys,
        tau_32=tau_32_phys,
        N_total=N_total,
        eps_inf=n_disk**2,
        time_scale=S,
    )
    fg_scaled_fixed_kappa = FourLevelGain(
        omega_a=omega_a,
        gamma_a=gamma_a,
        tau_21=tau_21_phys / S,
        tau_10=tau_10_phys / S,
        tau_32=tau_32_phys / S,
        N_total=N_total,
        eps_inf=n_disk**2,
        kappa_a=fg_phys.kappa_a,
    )
    sigma_ratio = fg_scaled_fixed_kappa.emission_cross_section / fg_phys.emission_cross_section
    dn_th_phys = g_th_1_m / fg_phys.emission_cross_section
    dn_th_scaled = g_th_1_m / fg_scaled_fixed_kappa.emission_cross_section
    wp_th_phys = fg_phys.threshold_pump_rate(g_th_1_m)
    wp_th_scaled = fg_scaled_fixed_kappa.threshold_pump_rate(g_th_1_m)
    isat_phys = fg_phys.saturation_intensity(W_p=wp_th_phys)
    isat_scaled = fg_scaled_fixed_kappa.saturation_intensity(W_p=wp_th_scaled)

    tau_p = q_cold / omega_a
    ro_1 = fg_phys.relaxation_oscillation_parameters(pump_ratio=2.4, tau_p=tau_p, W_p=2.4 * wp_th_phys, time_scale=1.0)
    ro_s = fg_time_scaled.relaxation_oscillation_parameters(pump_ratio=2.4, tau_p=tau_p, W_p=2.4 * wp_th_phys, time_scale=S)
    t_th_1 = fg_phys.turn_on_delay(g_th_1_m, W_p=2.4 * wp_th_phys, time_scale=1.0)
    t_th_s = fg_time_scaled.turn_on_delay(g_th_1_m, W_p=2.4 * wp_th_phys, time_scale=S)

    dt_ref = 0.99 * 0.032e-6 / (c_light * np.sqrt(2.0))
    ts_lim = fg_time_scaled.time_scaling_limits(
        tau_p=tau_p, dt=dt_ref, W_p=2.4 * wp_th_phys, time_scale=S
    )

    gamma_cav = omega_a / q_cold
    f_ro_phys_hz = np.sqrt(gamma_cav / tau_21_phys) / (2.0 * np.pi)
    f_ro_scaled_hz = np.sqrt(gamma_cav / (tau_21_phys / S)) / (2.0 * np.pi)
    gamma_ro_bare_scaled = 0.5 * 2.4 * S / tau_21_phys
    omega_ro_bare_scaled = 2.0 * np.pi * f_ro_scaled_hz

    return {
        "scale_factor": S,
        "sigma_ratio_fixed_kappa": float(sigma_ratio),
        "dN_th_ratio_fixed_kappa": float(dn_th_scaled / dn_th_phys),
        "W_p_th_ratio_fixed_kappa": float(wp_th_scaled / wp_th_phys),
        "I_sat_ratio_fixed_kappa": float(isat_scaled / isat_phys),
        "W_p_th_ratio_time_scale": float(fg_time_scaled.threshold_pump_rate(g_th_1_m) / wp_th_phys),
        "I_sat_ratio_time_scale": float(fg_time_scaled.saturation_intensity(W_p=wp_th_phys) / isat_phys),
        "turn_on_delay_ratio": float(t_th_1 / t_th_s),
        "t_th_phys_ps": float(t_th_1 * 1e12),
        "t_th_scaled_fs": float(t_th_s * 1e15),
        "tau_p_fs": float(tau_p * 1e15),
        "tau_21_eff_fs": float((tau_21_phys / S) * 1e15),
        "tau_sat_eff_fs": float(ro_s["tau_sat_eff"] * 1e15),
        "tau_sat_eff_ratio": float(ro_1["tau_sat_eff"] / ro_s["tau_sat_eff"]),
        "f_ro_ratio": float(f_ro_scaled_hz / f_ro_phys_hz),
        "f_ro_phys_ghz": float(f_ro_phys_hz * 1e-9),
        "f_ro_scaled_ghz": float(f_ro_scaled_hz * 1e-9),
        "f_ro_damped_phys_ghz": float(ro_1["f_ro"] * 1e-9),
        "f_ro_undamped_phys_ghz": float(ro_1["f_ro_undamped"] * 1e-9),
        "gamma_ro_phys_1_s": float(ro_1["gamma_ro"]),
        "underdamped_phys": bool(ro_1["underdamped"]),
        "f_ro_damped_scaled_ghz": float(ro_s["f_ro"] * 1e-9),
        "f_ro_undamped_scaled_ghz": float(ro_s["f_ro_undamped"] * 1e-9),
        "omega_ro_undamped_scaled_1_s": float(ro_s["omega_ro_undamped"]),
        "gamma_ro_scaled_1_s": float(ro_s["gamma_ro"]),
        "damping_ratio_scaled": float(ro_s["gamma_ro"] / max(ro_s["omega_ro_undamped"], 1e-30)),
        "omega_ro_bare_scaled_1_s": float(omega_ro_bare_scaled),
        "gamma_ro_bare_scaled_1_s": float(gamma_ro_bare_scaled),
        "overdamped_bare_scaled": bool(gamma_ro_bare_scaled > omega_ro_bare_scaled),
        "time_scaling_limits": {
            "time_scale": round(float(ts_lim["time_scale"]), 2),
            "sat_rate_over_kappa_cav": round(float(ts_lim["sat_rate_over_kappa_cav"]), 4),
            "rate_21_over_kappa_cav": round(float(ts_lim["rate_21_over_kappa_cav"]), 4),
            "rate_21_over_gamma_a": round(float(ts_lim["rate_21_over_gamma_a"]), 4),
            "sat_rate_over_gamma_a": round(float(ts_lim["sat_rate_over_gamma_a"]), 4),
            "max_rate_over_gamma_a": round(float(ts_lim["max_rate_over_gamma_a"]), 4),
            "s_max_cavity": round(float(ts_lim["s_max_cavity"]), 2),
            "s_max_dephasing": round(float(ts_lim["s_max_dephasing"]), 2),
            "s_max_fast_dephasing": round(float(ts_lim["s_max_fast_dephasing"]), 2),
            "max_rate_dt": round(float(ts_lim["max_rate_dt"]), 4),
            "s_max_cn_positivity": round(float(ts_lim["s_max_cn_positivity"]), 2),
        },
    }


# ---------------------------------------------------------------------------
# Resource preflight & 2D FIT microdisk geometry builders
# ---------------------------------------------------------------------------


def preflight_solver(
    solver: SolverFIT3D,
    n_steps: int,
    n_monitors: list[tuple[int, int, int]] | None = None,
    dispersive_edges: int = 0,
    n_poles: int = 0,
    mem_limit_mb: float = DEFAULT_MEM_LIMIT_MB,
) -> dict[str, float]:
    """Run ``wakis.estimate.estimate`` and ``preflight`` before executing a 2D FIT run.

    Parameters
    ----------
    solver : SolverFIT3D
        Configured Wakis FIT solver instance.
    n_steps : int
        Number of time steps to execute [-].
    n_monitors : list of tuple, optional
        Monitor dimensions ``[(n_freqs, n_samples, n_comps), ...]``.
    dispersive_edges : int, optional
        Number of active/dispersive E-edges [-].
    n_poles : int, optional
        Number of Lorentz/gain poles per dispersive edge [-].
    mem_limit_mb : float, optional
        Maximum allowed host memory [MB].

    Returns
    -------
    info : dict
        Estimated host memory [MB] and wall time [s].
    """
    est = estimate(
        cells=(int(solver.Nx), int(solver.Ny), int(solver.Nz)),
        steps=int(n_steps),
        backend="cpu",
        dtype="float64",
        cpml=any(str(b).lower() in ("cpml", "pml") for b in list(solver.bc_low) + list(solver.bc_high)),
        monitors=tuple(n_monitors or ()),
        dispersive_edges=int(dispersive_edges),
        n_poles=int(n_poles),
    )
    preflight(est, machine="laptop")
    mem_mb = float(est.host_bytes / 1e6)
    if mem_mb > float(mem_limit_mb):
        raise MemoryError(f"Estimated host memory {mem_mb:.1f} MB exceeds limit {mem_limit_mb:.1f} MB")
    return {
        "memory_mb": mem_mb,
        "estimated_seconds": float(est.seconds),
        "cells": int(est.cells),
    }


def compute_subpixel_fill_fractions(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    dx: float,
    r_disk_m: float = DEFAULT_R_DISK_M,
    r_in_frac: float = DEFAULT_R_IN_FRAC,
    bus_width_m: float | None = None,
    bus_gap_m: float = DEFAULT_BUS_GAP_M,
    subpixel: bool = True,
    n_sub: int = 8,
) -> dict[str, np.ndarray]:
    """Compute 2D subpixel (or staircase) fill fractions on primary/dual FIT grid locations.

    For curved dielectric interfaces in 2D FIT, subpixel area-fraction averaging of
    the out-of-plane permittivity ``eps_zz`` at ``(x_i, y_j)`` and harmonic/arithmetic
    averaging of the in-plane permittivities ``eps_xx`` at ``(x_i + dx/2, y_j)`` and
    ``eps_yy`` at ``(x_i, y_j + dy/2)`` suppress first-order ``O(dx)`` staircase
    sidewall-roughness scattering and restore second-order ``O(dx^2)`` convergence.

    Parameters
    ----------
    x_arr, y_arr : ndarray
        1D grid coordinates [m].
    dx : float
        Uniform cell size [m].
    r_disk_m : float, optional
        Microdisk radius ``R`` [m].
    r_in_frac : float, optional
        Inner radius fraction ``R_in / R`` for the annular pump ring [-].
    bus_width_m : float or None, optional
        If provided, include a straight bus waveguide of width ``bus_width_m`` [m]
        below the disk separated by edge-to-edge gap ``bus_gap_m`` [m].
    bus_gap_m : float, optional
        Edge-to-edge coupling gap between ``y = -R`` and the top of the bus waveguide [m].
    subpixel : bool, optional
        If ``True`` (default), use ``n_sub x n_sub`` quadrature per cell; if ``False``,
        use cell-center staircase sampling.
    n_sub : int, optional
        Subpixel quadrature subdivisions per axis [-].

    Returns
    -------
    fills : dict of ndarray
        Dictionary with ``fill_ez_disk``, ``fill_ex_disk``, ``fill_ey_disk``,
        ``fill_ez_ring``, ``fill_ez_core``, ``fill_ez_bus``, and ``fill_ez_total``.
    """
    R = float(r_disk_m)
    r_in = float(r_in_frac) * R
    X, Y = np.meshgrid(x_arr, y_arr, indexing="ij")

    if subpixel and int(n_sub) > 1:
        offs = (np.arange(int(n_sub)) + 0.5) / float(n_sub) - 0.5
    else:
        offs = np.array([0.0])
    OX, OY = np.meshgrid(offs * float(dx), offs * float(dx), indexing="ij")

    # E_z lives at (x_i, y_j)
    Xs_ez = X[:, :, None, None] + OX[None, None, :, :]
    Ys_ez = Y[:, :, None, None] + OY[None, None, :, :]
    Rs_ez = np.hypot(Xs_ez, Ys_ez)
    fill_ez_disk = np.mean(Rs_ez <= R, axis=(2, 3))
    fill_ez_ring = np.mean((Rs_ez <= R) & (Rs_ez >= r_in), axis=(2, 3))
    fill_ez_core = np.mean(Rs_ez < r_in, axis=(2, 3))

    # E_x lives at (x_i + dx/2, y_j); E_y lives at (x_i, y_j + dy/2)
    Rs_ex = np.hypot(Xs_ez + 0.5 * float(dx), Ys_ez)
    Rs_ey = np.hypot(Xs_ez, Ys_ez + 0.5 * float(dx))
    fill_ex_disk = np.mean(Rs_ex <= R, axis=(2, 3))
    fill_ey_disk = np.mean(Rs_ey <= R, axis=(2, 3))

    if bus_width_m is not None and float(bus_width_m) > 0.0:
        w_b = float(bus_width_m)
        g_b = float(bus_gap_m)
        y_top = -(R + g_b)
        y_bot = y_top - w_b
        fill_ez_bus = np.mean((Ys_ez <= y_top) & (Ys_ez >= y_bot), axis=(2, 3))
    else:
        fill_ez_bus = np.zeros_like(fill_ez_disk)

    fill_ez_total = np.clip(fill_ez_disk + fill_ez_bus, 0.0, 1.0)
    return {
        "fill_ez_disk": fill_ez_disk,
        "fill_ex_disk": fill_ex_disk,
        "fill_ey_disk": fill_ey_disk,
        "fill_ez_ring": fill_ez_ring,
        "fill_ez_core": fill_ez_core,
        "fill_ez_bus": fill_ez_bus,
        "fill_ez_total": fill_ez_total,
    }


def build_microdisk_solver(
    dx_m: float = 0.028e-6,
    domain_half_m: float = 1.80e-6,
    r_disk_m: float = DEFAULT_R_DISK_M,
    n_disk: float = DEFAULT_N_DISK,
    n_out: float = DEFAULT_N_OUT,
    bus_width_m: float | None = None,
    bus_gap_m: float = DEFAULT_BUS_GAP_M,
    subpixel: bool = True,
    n_pml: int = 10,
) -> tuple[SolverFIT3D, dict[str, np.ndarray], list]:
    """Construct a 2D FIT solver (``Nz=1``, periodic along ``z``, CPML on ``+-x, +-y``) for a microdisk.

    Parameters
    ----------
    dx_m : float, optional
        Transverse grid spacing ``dx = dy`` [m].
    domain_half_m : float, optional
        Half-width of the square computational domain ``[-L/2, +L/2]^2`` [m].
    r_disk_m : float, optional
        Microdisk radius ``R`` [m].
    n_disk, n_out : float, optional
        Disk and exterior refractive indices [-].
    bus_width_m : float or None, optional
        Width of the coupled bus waveguide [m], or ``None`` for an isolated disk.
    bus_gap_m : float, optional
        Edge-to-edge coupling gap between disk rim and bus waveguide [m].
    subpixel : bool, optional
        Whether to apply subpixel permittivity smoothing (``True``) or staircase (``False``).
    n_pml : int, optional
        Number of CPML absorbing cells on ``+-x, +-y`` [-].

    Returns
    -------
    solver : SolverFIT3D
        Configured Wakis solver.
    fills : dict of ndarray
        Subpixel or staircase fill-fraction maps.
    structures : list
        List of ``Cylinder`` and ``Box`` shape descriptors for ``wakis.schematic``.
    """
    dx = float(dx_m)
    nx = int(2 * round(float(domain_half_m) / dx))
    half_l = 0.5 * nx * dx
    lz = 10.0 * dx

    grid = GridFIT3D(
        -half_l,
        +half_l,
        -half_l,
        +half_l,
        -0.5 * lz,
        +0.5 * lz,
        nx,
        nx,
        1,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=int(n_pml),
        verbose=0,
    )

    fills = compute_subpixel_fill_fractions(
        solver.x,
        solver.y,
        dx=dx,
        r_disk_m=r_disk_m,
        bus_width_m=bus_width_m,
        bus_gap_m=bus_gap_m,
        subpixel=subpixel,
    )

    eps_disk = float(n_disk) ** 2
    eps_bg = float(n_out) ** 2
    eps_zz = eps_bg + (eps_disk - eps_bg) * fills["fill_ez_total"]

    # For TE in-plane components (E_x, E_y), use subpixel-weighted permittivity
    X, Y = np.meshgrid(solver.x, solver.y, indexing="ij")
    theta_ex = np.arctan2(Y, X + 0.5 * dx)
    nx_ex2 = np.cos(theta_ex) ** 2
    ny_ex2 = np.sin(theta_ex) ** 2
    fx = fills["fill_ex_disk"]
    eps_par_x = eps_bg + (eps_disk - eps_bg) * fx
    eps_perp_x = 1.0 / (fx / eps_disk + (1.0 - fx) / eps_bg)
    eps_xx = nx_ex2 * eps_perp_x + ny_ex2 * eps_par_x if subpixel else eps_par_x

    theta_ey = np.arctan2(Y + 0.5 * dx, X)
    nx_ey2 = np.cos(theta_ey) ** 2
    ny_ey2 = np.sin(theta_ey) ** 2
    fy = fills["fill_ey_disk"]
    eps_par_y = eps_bg + (eps_disk - eps_bg) * fy
    eps_perp_y = 1.0 / (fy / eps_disk + (1.0 - fy) / eps_bg)
    eps_yy = ny_ey2 * eps_perp_y + nx_ey2 * eps_par_y if subpixel else eps_par_y

    n3 = solver.Nx * solver.Ny * solver.Nz
    solver.ieps.array[0:n3] = (1.0 / (eps_xx * epsilon_0)).ravel(order="F")
    solver.ieps.array[n3 : 2 * n3] = (1.0 / (eps_yy * epsilon_0)).ravel(order="F")
    solver.ieps.array[2 * n3 : 3 * n3] = (1.0 / (eps_zz * epsilon_0)).ravel(order="F")
    solver.update_tensors("ieps")

    disk_cyl = Cylinder(
        center=(0.0, 0.0, 0.0),
        radius=float(r_disk_m),
        length=lz,
        material=eps_disk,
        axis="z",
    )
    disk_cyl.name = f"Microdisk (n = {n_disk:.1f}, R = {r_disk_m * 1e6:.1f} µm)"
    structures = [disk_cyl]

    if bus_width_m is not None and float(bus_width_m) > 0.0:
        w_b = float(bus_width_m)
        g_b = float(bus_gap_m)
        y_center_bus = -(float(r_disk_m) + g_b + 0.5 * w_b)
        bus_box = Box(
            center=(0.0, y_center_bus, 0.0),
            size=(2.0 * half_l, w_b, lz),
            material=eps_disk,
        )
        bus_box.thickness = w_b
        bus_box.name = f"Bus waveguide (w = {w_b * 1e9:.0f} nm, g = {g_b * 1e9:.0f} nm)"
        structures.append(bus_box)

    return solver, fills, structures


def run_cold_cavity_harminv(
    dx_m: float = 0.028e-6,
    polarization: str = "TM",
    subpixel: bool = True,
    m_orders: tuple[int, ...] = (4, 5, 6, 7, 8, 9),
    q_orders: tuple[int, ...] = (1, 2),
    t_after_s: float = 320.0e-15,
    record_dft_m6: bool = False,
    mem_limit_mb: float = DEFAULT_MEM_LIMIT_MB,
) -> dict:
    """Run a 2D FIT cold-cavity broadband ringdown and extract WGM frequencies and Q via Harminv.

    Parameters
    ----------
    dx_m : float, optional
        Grid cell size ``dx = dy`` [m].
    polarization : {'TM', 'TE'}, optional
        ``'TM'`` excites out-of-plane ``E_z``; ``'TE'`` excites in-plane ``E_y`` / out-of-plane ``H_z``.
    subpixel : bool, optional
        Subpixel permittivity smoothing (``True``) vs staircase (``False``).
    m_orders : tuple of int, optional
        Azimuthal mode orders ``m`` to match against exact Bessel/Hankel roots [-].
    q_orders : tuple of int, optional
        Radial mode orders ``q`` to match [-].
    t_after_s : float, optional
        Ringdown duration after source turn-off [s].
    record_dft_m6 : bool, optional
        If ``True``, record 2D DFT field slices at ``(6, 1)`` and ``(6, 2)``.
    mem_limit_mb : float, optional
        Host memory limit [MB].

    Returns
    -------
    out : dict
        Matched modes, spectrum, ringdown trace, and optional 2D DFT mode profiles.
    """
    pol = polarization.strip().upper()
    solver, fills, structures = build_microdisk_solver(
        dx_m=dx_m,
        domain_half_m=1.80e-6,
        r_disk_m=DEFAULT_R_DISK_M,
        n_disk=DEFAULT_N_DISK,
        subpixel=subpixel,
        n_pml=10,
    )
    lz = float(solver.grid.zmax - solver.grid.zmin)
    only_q2 = tuple(q_orders) == (2,)
    if only_q2:
        f0 = 260.0e12
        df = 100.0e12
        gw = GaussianSource(frequency=f0, fwidth=df, cutoff=3.6)
    else:
        f0 = 215.0e12
        df = 80.0e12
        gw = GaussianSource(frequency=f0, fwidth=df, cutoff=4.0)

    comp_src = "z" if pol == "TM" else "y"
    comp_prb = "Ez" if pol == "TM" else "Hz"

    # Place rim source at r = 0.849 R (node of q = 2) for pure q = 1 excitation,
    # or inner-lobe source at r = 0.43 R with diametrically opposite parity probes
    # at (+r_s, 0) and (-r_s, 0) when extracting low-Q radial order q = 2 so that
    # even-m and odd-m modes (e.g. TM_{6,2} at 287.2 THz vs high-Q TM_{9,1} at 284.9 THz)
    # are separated by exact pi-rotation symmetry.
    if only_q2:
        r_s = 0.43 * DEFAULT_R_DISK_M
        src_pos = (r_s, 0.0, 0.0)
        prb_pos = (r_s, 0.0, 0.0)
        prb_opp_pos = (-r_s, 0.0, 0.0)
        after_t = gw.end_time + 8.0e-15
        t_ring = min(float(t_after_s), 130.0e-15)
    else:
        src_pos = (0.72 * DEFAULT_R_DISK_M, 0.45 * DEFAULT_R_DISK_M, 0.0)
        prb_pos = (-0.72 * DEFAULT_R_DISK_M, 0.45 * DEFAULT_R_DISK_M, 0.0)
        prb_opp_pos = None
        after_t = gw.end_time + 25.0e-15
        t_ring = float(t_after_s)

    s1 = CurrentSource(
        gw,
        component=comp_src,
        center=src_pos,
        size=(0.0, 0.0, lz),
        amplitude=1.0,
    )

    prb_q1 = ResonanceMonitor(
        component=comp_prb,
        center=prb_pos,
        t_start=after_t,
    )
    prb_opp = (
        ResonanceMonitor(
            component=comp_prb,
            center=prb_opp_pos,
            t_start=after_t,
        )
        if prb_opp_pos is not None
        else None
    )

    t_end = gw.end_time + t_ring
    nt = int(np.ceil(t_end / solver.dt))

    mon_dft = None
    mon_specs = []
    if record_dft_m6 and pol == "TM":
        r61 = solve_wgm_complex_root(6, 1, polarization="TM")
        r62 = solve_wgm_complex_root(6, 2, polarization="TM")
        mon_dft = DFTMonitor(
            [r61["frequency_hz"], r62["frequency_hz"]],
            components=("Ez",),
            center=(0.0, 0.0, 0.0),
            size=(2.8e-6, 2.8e-6, 0.0),
        )
        mon_specs.append((2, solver.Nx * solver.Ny, 1))

    est = preflight_solver(solver, n_steps=nt, n_monitors=mon_specs, mem_limit_mb=mem_limit_mb)

    n_dft_start = int(np.ceil(after_t / solver.dt))
    t0_wall = time.perf_counter()
    for step in range(nt):
        t = step * solver.dt
        s1.update(solver, t)
        solver.one_step()
        prb_q1.update(solver, t)
        if prb_opp is not None:
            prb_opp.update(solver, t)
        if mon_dft is not None and step >= n_dft_start:
            mon_dft.update(solver, t)
    wall_s = time.perf_counter() - t0_wall

    matched: dict[tuple[int, int], dict[str, float]] = {}
    for q_val in q_orders:
        for m_val in m_orders:
            if q_val == 2 and m_val > 7:
                continue
            ref = solve_wgm_complex_root(m_val, q_val, polarization=pol)
            f_ref = ref["frequency_hz"]
            if prb_opp is not None:
                _, sig_0 = prb_q1.signal()
                _, sig_pi = prb_opp.signal()
                sig_par = 0.5 * (sig_0 + sig_pi) if (m_val % 2 == 0) else 0.5 * (sig_0 - sig_pi)
                res_win = harminv(
                    sig_par,
                    solver.dt,
                    fmin=f_ref * 0.94,
                    fmax=f_ref * 1.06,
                    min_rel_amplitude=1e-4,
                )
            else:
                res_win = prb_q1.harminv(fmin=f_ref * 0.955, fmax=f_ref * 1.045, min_rel_amplitude=1e-3)
                if res_win.frequency.size == 0:
                    res_win = prb_q1.harminv(fmin=f_ref * 0.93, fmax=f_ref * 1.07, min_rel_amplitude=1e-4)
            idx_best = int(np.argmin(np.abs(res_win.frequency - f_ref)))
            f_sim = float(res_win.frequency[idx_best])
            q_sim = float(res_win.Q[idx_best])
            dec_sim = float(res_win.decay_rate[idx_best])
            matched[(m_val, q_val)] = {
                "f_sim_thz": f_sim * 1e-12,
                "f_ana_thz": ref["frequency_thz"],
                "f_rel_err": (f_sim - f_ref) / f_ref,
                "q_sim": q_sim,
                "q_ana": ref["q_rad"],
                "q_rel_err": (q_sim - ref["q_rad"]) / ref["q_rad"],
                "decay_sim_1_s": dec_sim,
                "decay_ana_1_s": ref["decay_rate_1_s"],
            }

    # Compute ringdown FFT spectrum for plotting
    t_arr, sig_ring = prb_q1.signal()
    win = np.hanning(sig_ring.size)
    n_fft = 16384
    freqs_fft = np.fft.rfftfreq(n_fft, d=solver.dt)
    spec_fft = np.abs(np.fft.rfft(sig_ring * win, n=n_fft))
    spec_fft /= max(float(np.max(spec_fft)), 1e-30)

    out = {
        "dx_m": float(dx_m),
        "nx": int(solver.Nx),
        "cells": int(est["cells"]),
        "nt": int(nt),
        "wall_s": float(wall_s),
        "memory_mb": float(est["memory_mb"]),
        "polarization": pol,
        "subpixel": bool(subpixel),
        "matched": matched,
        "freqs_thz": freqs_fft * 1e-12,
        "spec_norm": spec_fft,
        "t_fs": t_arr * 1e15,
        "signal_q1": sig_ring,
    }
    if mon_dft is not None:
        ez_dft, coords_dft = mon_dft.get_field("Ez")
        out["x_dft_m"] = coords_dft[0]
        out["y_dft_m"] = coords_dft[1]
        out["ez_m6_q1"] = np.real(ez_dft[0, :, :, 0])
        out["ez_m6_q2"] = np.real(ez_dft[1, :, :, 0])
    return out


def run_task_cold_cavity(quick: bool = False) -> dict:
    """Run Task 1: Cold-cavity WGM resonances, radiative Q(m), 3-level convergence, and subpixel vs staircase."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dx_levels = (0.044e-6, 0.032e-6, 0.024e-6) if quick else (0.040e-6, 0.028e-6, 0.020e-6)
    t_after = 220.0e-15 if quick else 320.0e-15
    m_orders = (4, 5, 6, 7, 8, 9)

    conv_sub = []
    conv_stair = []
    for idx_res, dx_val in enumerate(dx_levels):
        is_mid = idx_res == 1
        r_sub = run_cold_cavity_harminv(
            dx_m=dx_val,
            polarization="TM",
            subpixel=True,
            m_orders=m_orders,
            q_orders=(1,),
            t_after_s=t_after,
            record_dft_m6=is_mid,
        )
        r_st = run_cold_cavity_harminv(
            dx_m=dx_val,
            polarization="TM",
            subpixel=False,
            m_orders=m_orders,
            q_orders=(1,),
            t_after_s=t_after,
            record_dft_m6=False,
        )
        conv_sub.append(r_sub)
        conv_stair.append(r_st)

    # Extract TM radial order q=2 and TE polarization on the finest grid
    r_tm_q2 = run_cold_cavity_harminv(
        dx_m=dx_levels[-1],
        polarization="TM",
        subpixel=True,
        m_orders=(4, 5, 6, 7),
        q_orders=(2,),
        t_after_s=t_after,
        record_dft_m6=False,
    )
    conv_sub[-1]["matched"].update(r_tm_q2["matched"])

    r_te_fine = run_cold_cavity_harminv(
        dx_m=dx_levels[-1],
        polarization="TE",
        subpixel=True,
        m_orders=(4, 5, 6, 7, 8),
        q_orders=(1,),
        t_after_s=t_after,
        record_dft_m6=False,
    )

    fine_sub = conv_sub[-1]
    fine_stair = conv_stair[-1]
    mid_sub = conv_sub[1]

    # Richardson extrapolation on (6, 1) frequency and Q using medium (dx_1) and fine (dx_2) grids
    dx1, dx2 = float(dx_levels[1]), float(dx_levels[2])
    r2_ratio = (dx1 / dx2) ** 2
    f6_m = conv_sub[1]["matched"][(6, 1)]["f_sim_thz"]
    f6_f = conv_sub[2]["matched"][(6, 1)]["f_sim_thz"]
    f6_ana = conv_sub[2]["matched"][(6, 1)]["f_ana_thz"]
    f6_rich = (r2_ratio * f6_f - f6_m) / (r2_ratio - 1.0)
    f6_rich_rel_err = (f6_rich - f6_ana) / f6_ana
    f6_rich_uncert_rel = abs(f6_f - f6_m) / ((r2_ratio - 1.0) * f6_ana)

    q8_m = conv_sub[1]["matched"][(8, 1)]["q_sim"]
    q8_f = conv_sub[2]["matched"][(8, 1)]["q_sim"]
    q8_ana = conv_sub[2]["matched"][(8, 1)]["q_ana"]
    q8_rich = (r2_ratio * q8_f - q8_m) / (r2_ratio - 1.0)
    q8_rich_rel_err = (q8_rich - q8_ana) / q8_ana
    # Discretization uncertainty estimate |Q_rich - Q_fine| / Q_exact
    q8_rich_uncert_rel = abs(q8_f - q8_m) / ((r2_ratio - 1.0) * q8_ana)

    # Observed convergence order p from the 3 levels on f_{6,1}
    err_f6_sub = [abs(c["matched"][(6, 1)]["f_rel_err"]) for c in conv_sub]
    err_f6_stair = [abs(c["matched"][(6, 1)]["f_rel_err"]) for c in conv_stair]
    err_q8_sub = [abs(c["matched"][(8, 1)]["q_rel_err"]) for c in conv_sub]
    err_q8_stair = [abs(c["matched"][(8, 1)]["q_rel_err"]) for c in conv_stair]
    p_obs_f6 = float(np.log(err_f6_sub[0] / err_f6_sub[2]) / np.log(dx_levels[0] / dx_levels[2]))

    # Assemble mode tables for TM q=1, TM q=2, and TE q=1 on the fine grid
    tm_q1_table = []
    for m_val in m_orders:
        m_sub = fine_sub["matched"][(m_val, 1)]
        m_st = fine_stair["matched"][(m_val, 1)]
        ref = solve_wgm_complex_root(m_val, 1, polarization="TM")
        tm_q1_table.append(
            {
                "m": int(m_val),
                "f_ana_thz": float(ref["frequency_thz"]),
                "f_sub_thz": float(m_sub["f_sim_thz"]),
                "f_sub_rel_err": float(m_sub["f_rel_err"]),
                "f_stair_thz": float(m_st["f_sim_thz"]),
                "f_stair_rel_err": float(m_st["f_rel_err"]),
                "q_ana": float(ref["q_rad"]),
                "q_sub": float(m_sub["q_sim"]),
                "q_sub_rel_err": float(m_sub["q_rel_err"]),
                "q_stair": float(m_st["q_sim"]),
                "q_stair_rel_err": float(m_st["q_rel_err"]),
                "confinement_disk": float(ref["confinement_disk"]),
                "confinement_disk_trunc": float(ref["confinement_disk_trunc"]),
                "confinement_ring": float(ref["confinement_ring"]),
            }
        )

    tm_q2_table = []
    for m_val in (4, 5, 6, 7, 8):
        if (m_val, 2) in fine_sub["matched"]:
            m_sub = fine_sub["matched"][(m_val, 2)]
            ref = solve_wgm_complex_root(m_val, 2, polarization="TM")
            tm_q2_table.append(
                {
                    "m": int(m_val),
                    "f_ana_thz": float(ref["frequency_thz"]),
                    "f_sub_thz": float(m_sub["f_sim_thz"]),
                    "f_sub_rel_err": float(m_sub["f_rel_err"]),
                    "q_ana": float(ref["q_rad"]),
                    "q_sub": float(m_sub["q_sim"]),
                    "q_sub_rel_err": float(m_sub["q_rel_err"]),
                    "confinement_disk": float(ref["confinement_disk"]),
                    "confinement_disk_trunc": float(ref["confinement_disk_trunc"]),
                    "confinement_ring": float(ref["confinement_ring"]),
                }
            )

    te_q1_table = []
    for m_val in (4, 5, 6, 7, 8):
        if (m_val, 1) in r_te_fine["matched"]:
            m_sub = r_te_fine["matched"][(m_val, 1)]
            ref = solve_wgm_complex_root(m_val, 1, polarization="TE")
            te_q1_table.append(
                {
                    "m": int(m_val),
                    "f_ana_thz": float(ref["frequency_thz"]),
                    "f_sub_thz": float(m_sub["f_sim_thz"]),
                    "f_sub_rel_err": float(m_sub["f_rel_err"]),
                    "q_ana": float(ref["q_rad"]),
                    "q_sub": float(m_sub["q_sim"]),
                    "q_sub_rel_err": float(m_sub["q_rel_err"]),
                }
            )

    lam_m6_disk = (c_light / (f6_ana * 1e12)) / DEFAULT_N_DISK
    cpw_levels = [float(lam_m6_disk / d) for d in dx_levels]

    summary = {
        "task": "cold_cavity",
        "quick": bool(quick),
        "r_disk_um": DEFAULT_R_DISK_M * 1e6,
        "n_disk": DEFAULT_N_DISK,
        "dx_levels_nm": [float(d * 1e9) for d in dx_levels],
        "cpw_m6_levels": cpw_levels,
        "nx_levels": [int(c["nx"]) for c in conv_sub],
        "cells_levels": [int(c["cells"]) for c in conv_sub],
        "wall_sub_s": [float(c["wall_s"]) for c in conv_sub],
        "wall_stair_s": [float(c["wall_s"]) for c in conv_stair],
        "wall_te_s": float(r_te_fine["wall_s"]),
        "memory_mb_levels": [float(c["memory_mb"]) for c in conv_sub],
        "err_f6_sub": err_f6_sub,
        "err_f6_stair": err_f6_stair,
        "err_q8_sub": err_q8_sub,
        "err_q8_stair": err_q8_stair,
        "p_obs_f6_sub": p_obs_f6,
        "f6_ana_thz": float(f6_ana),
        "f6_fine_sub_thz": float(f6_f),
        "f6_fine_sub_rel_err": float(conv_sub[2]["matched"][(6, 1)]["f_rel_err"]),
        "f6_rich_thz": float(f6_rich),
        "f6_rich_rel_err": float(f6_rich_rel_err),
        "f6_rich_uncert_rel": float(f6_rich_uncert_rel),
        "q6_ana": float(conv_sub[2]["matched"][(6, 1)]["q_ana"]),
        "q6_fine_sub": float(conv_sub[2]["matched"][(6, 1)]["q_sim"]),
        "q6_fine_sub_rel_err": float(conv_sub[2]["matched"][(6, 1)]["q_rel_err"]),
        "q6_fine_stair": float(conv_stair[2]["matched"][(6, 1)]["q_sim"]),
        "q6_fine_stair_rel_err": float(conv_stair[2]["matched"][(6, 1)]["q_rel_err"]),
        "q8_ana": float(q8_ana),
        "q8_fine_sub": float(q8_f),
        "q8_fine_sub_rel_err": float(conv_sub[2]["matched"][(8, 1)]["q_rel_err"]),
        "q8_rich": float(q8_rich),
        "q8_rich_rel_err": float(q8_rich_rel_err),
        "q8_rich_uncert_rel": float(q8_rich_uncert_rel),
        "q8_fine_stair": float(conv_stair[2]["matched"][(8, 1)]["q_sim"]),
        "q8_fine_stair_rel_err": float(conv_stair[2]["matched"][(8, 1)]["q_rel_err"]),
        "tm_q1_modes": tm_q1_table,
        "tm_q2_modes": tm_q2_table,
        "te_q1_modes": te_q1_table,
    }

    with open(DATA_DIR / "task_01_cold_cavity.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    np.savez_compressed(
        DATA_DIR / "task_01_cold_cavity.npz",
        freqs_thz=fine_sub["freqs_thz"],
        spec_tm_sub=fine_sub["spec_norm"],
        spec_tm_stair=fine_stair["spec_norm"],
        freqs_te_thz=r_te_fine["freqs_thz"],
        spec_te_sub=r_te_fine["spec_norm"],
        x_dft_m=mid_sub["x_dft_m"],
        y_dft_m=mid_sub["y_dft_m"],
        ez_m6_q1=mid_sub["ez_m6_q1"],
        ez_m6_q2=mid_sub["ez_m6_q2"],
        dx_levels_nm=np.array(summary["dx_levels_nm"]),
        err_f6_sub=np.array(err_f6_sub),
        err_f6_stair=np.array(err_f6_stair),
        err_q8_sub=np.array(err_q8_sub),
        err_q8_stair=np.array(err_q8_stair),
    )
    return summary


def run_task_threshold(quick: bool = False) -> dict:
    """Run Task 2: Linear Lorentz gain pole threshold scan across m = 4..8 vs derived formula."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dx_m = 0.036e-6 if quick else 0.032e-6
    t_after_s = 190.0e-15 if quick else 240.0e-15
    m_orders = (4, 5, 6, 7, 8)

    gamma_a = 2.0 * np.pi * 50.0e12  # Wide homogeneous gain pole for clean linear-gain extraction
    n_tot_lin = 1.0e24
    g_ratios = np.array([0.0, 0.55, 0.90, 1.25])

    mode_results = []
    all_decay_curves = {}
    total_wall_s = 0.0
    max_mem_mb = 0.0
    nx_grid = 0
    cells_grid = 0

    for m_val in m_orders:
        ref = solve_wgm_complex_root(m_val, 1, polarization="TM")
        f_mode = float(ref["frequency_hz"])
        omega_m = 2.0 * np.pi * f_mode
        gamma_conf = float(ref["confinement_disk"])
        gamma_conf_trunc = float(ref["confinement_disk_trunc"])
        g_th_bessel_1_m = float(ref["g_th_1_m"])

        decays_m = []
        g_vals_1_m = []
        q_cold_sim = float(ref["q_rad"])
        g_th_unconf_1_m = DEFAULT_N_DISK * omega_m / (c_light * q_cold_sim)
        g_th_pred_1_m = g_th_unconf_1_m / gamma_conf

        for gr in g_ratios:
            g_target = 0.0 if gr == 0.0 else float(gr * g_th_pred_1_m)
            g_vals_1_m.append(g_target)

            solver, fills, _ = build_microdisk_solver(
                dx_m=dx_m,
                domain_half_m=1.80e-6,
                r_disk_m=DEFAULT_R_DISK_M,
                n_disk=DEFAULT_N_DISK,
                subpixel=True,
                n_pml=10,
            )
            nx_grid = int(solver.Nx)
            lz = float(solver.grid.zmax - solver.grid.zmin)
            n3 = solver.Nx * solver.Ny * solver.Nz
            fill_ez = fills["fill_ez_disk"].ravel(order="F")
            sel = np.flatnonzero(fill_ez > 0.0)

            if g_target > 0.0:
                kappa_lin = g_target * epsilon_0 * c_light * DEFAULT_N_DISK * gamma_a / n_tot_lin
                fg_lin = FourLevelGain(
                    omega_a=omega_m,
                    gamma_a=gamma_a,
                    kappa_a=kappa_lin,
                    tau_21=np.inf,
                    tau_10=np.inf,
                    tau_32=np.inf,
                    N_total=n_tot_lin,
                    pump_rate=0.0,
                    eps_inf=DEFAULT_N_DISK**2,
                    initial_populations=(0.0, 0.0, n_tot_lin, 0.0),
                    enable_stimulated=False,
                )
                solver._add_gain_edges(fg_lin, sel + 2 * n3, fill_ez[sel], eps_bg=1.0)

            gw = GaussianSource(frequency=f_mode, fwidth=0.06 * f_mode, cutoff=3.8)
            src = CurrentSource(
                gw,
                component="z",
                center=(0.72 * DEFAULT_R_DISK_M, 0.45 * DEFAULT_R_DISK_M, 0.0),
                size=(0.0, 0.0, lz),
                amplitude=1.0e-4,
            )
            # Place probe at the diametrically opposite point (-x_0, -y_0) where Delta_phi = pi
            # so |cos(m * pi)| = 1 is an antinode for every integer azimuthal order m
            prb = ResonanceMonitor(
                component="Ez",
                center=(-0.72 * DEFAULT_R_DISK_M, -0.45 * DEFAULT_R_DISK_M, 0.0),
                t_start=gw.end_time + 25.0e-15,
            )
            nt = int(np.ceil((gw.end_time + t_after_s) / solver.dt))
            n_disp = int(sel.size) if g_target > 0.0 else 0
            est = preflight_solver(
                solver,
                n_steps=nt,
                dispersive_edges=n_disp,
                n_poles=1 if g_target > 0.0 else 0,
            )
            cells_grid = int(est["cells"])
            max_mem_mb = max(max_mem_mb, float(est["memory_mb"]))

            t0_w = time.perf_counter()
            for step in range(nt):
                t = step * solver.dt
                src.update(solver, t)
                solver.one_step()
                prb.update(solver, t)
            total_wall_s += time.perf_counter() - t0_w

            res = prb.harminv(
                fmin=f_mode * 0.95,
                fmax=f_mode * 1.05,
                min_rel_amplitude=1e-3,
            )
            cand = np.flatnonzero(np.abs(res.frequency - f_mode) / f_mode < 0.015)
            if cand.size > 0:
                idx_k = int(cand[np.argmax(np.abs(res.amplitude[cand]))])
            else:
                idx_k = int(np.argmin(np.abs(res.frequency - f_mode)))
            decays_m.append(float(res.decay_rate[idx_k]))

            if gr == 0.0:
                q_cold_sim = float(res.Q[idx_k])
                f_mode = float(res.frequency[idx_k])
                omega_m = 2.0 * np.pi * f_mode
                # Theoretical threshold gain including confinement factor Gamma_{m,1}:
                # g_th = n_disk * omega_m / (c * Gamma_{m,1} * Q_cold)
                g_th_unconf_1_m = DEFAULT_N_DISK * omega_m / (c_light * q_cold_sim)
                g_th_pred_1_m = g_th_unconf_1_m / gamma_conf

        g_arr_cm1 = np.array(g_vals_1_m) * 1e-2
        dec_arr = np.array(decays_m)
        # Linear fit of decay_rate(g) = a * g + b -> zero crossing g_th_fit = -b / a
        slope, intercept = np.polyfit(g_arr_cm1, dec_arr, 1)
        g_th_fit_cm1 = float(-intercept / slope)
        g_th_pred_cm1 = float(g_th_pred_1_m * 1e-2)
        g_th_unconf_cm1 = float(g_th_unconf_1_m * 1e-2)
        g_th_bessel_cm1 = float(g_th_bessel_1_m * 1e-2)

        rel_err_vs_pred = (g_th_fit_cm1 - g_th_pred_cm1) / g_th_pred_cm1
        rel_err_vs_bessel = (g_th_fit_cm1 - g_th_bessel_cm1) / g_th_bessel_cm1
        rel_err_unconf = (g_th_unconf_cm1 - g_th_fit_cm1) / g_th_fit_cm1

        all_decay_curves[f"m{m_val}_g_cm1"] = g_arr_cm1
        all_decay_curves[f"m{m_val}_decay_1_s"] = dec_arr

        mode_results.append(
            {
                "m": int(m_val),
                "f_sim_thz": float(f_mode * 1e-12),
                "q_cold_sim": float(q_cold_sim),
                "q_cold_ana": float(ref["q_rad"]),
                "confinement_disk": gamma_conf,
                "confinement_disk_trunc": gamma_conf_trunc,
                "g_th_fit_cm1": g_th_fit_cm1,
                "g_th_pred_cm1": g_th_pred_cm1,
                "g_th_bessel_cm1": g_th_bessel_cm1,
                "g_th_unconfined_cm1": g_th_unconf_cm1,
                "rel_err_vs_pred": float(rel_err_vs_pred),
                "rel_err_vs_bessel": float(rel_err_vs_bessel),
                "rel_err_unconfined_vs_fit": float(rel_err_unconf),
            }
        )

    mean_abs_rel_err = float(np.mean([abs(r["rel_err_vs_pred"]) for r in mode_results]))
    mean_abs_rel_err_bessel = float(np.mean([abs(r["rel_err_vs_bessel"]) for r in mode_results]))
    mean_unconf_err = float(np.mean([abs(r["rel_err_unconfined_vs_fit"]) for r in mode_results]))

    summary = {
        "task": "threshold",
        "quick": bool(quick),
        "dx_nm": float(dx_m * 1e9),
        "nx": nx_grid,
        "cells": cells_grid,
        "total_wall_s": total_wall_s,
        "memory_mb": max_mem_mb,
        "mean_abs_rel_err_confined": mean_abs_rel_err,
        "mean_abs_rel_err_bessel": mean_abs_rel_err_bessel,
        "mean_abs_rel_err_unconfined": mean_unconf_err,
        "modes": mode_results,
    }

    with open(DATA_DIR / "task_02_threshold.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    np.savez_compressed(DATA_DIR / "task_02_threshold.npz", **all_decay_curves)
    return summary


def run_single_four_level_microdisk(
    pump_ratio: float = 2.4,
    pump_profile: str = "ring",
    kappa_scale: float = 1.0,
    spatial_averaging: bool = False,
    with_bus: bool = True,
    dx_m: float = 0.032e-6,
    n_steps: int = 24000,
    seed_amp: float = 0.0,
    spontaneous_emission: bool = True,
    noise_amp: float = DEFAULT_NOISE_AMP,
    seed: int = DEFAULT_NOISE_SEED,
    time_scale: float = DEFAULT_TIME_SCALE,
    record_2d_frames: int = 0,
    record_2d_inversion: bool = False,
    mem_limit_mb: float = DEFAULT_MEM_LIMIT_MB,
) -> dict:
    """Run one 2D FIT active microdisk simulation with ``FourLevelGain`` seeded by spontaneous emission.

    Parameters
    ----------
    pump_ratio : float, optional
        Normalized pump rate ``r_p = W_p / W_{p,th}`` [-].
    pump_profile : {'ring', 'full', 'core'}, optional
        Spatial pump distribution inside the disk:
        - ``'ring'``: annular pump ``r in [0.55 R, R]`` (favours ``q = 1`` WGM).
        - ``'full'``: uniform full-disk pump ``r in [0, R]``.
        - ``'core'``: central core pump ``r in [0, 0.55 R)`` (minimal overlap with ``q = 1``).
    kappa_scale : float, optional
        Multiplier on the nominal oscillator coupling ``kappa_a`` [-] (used for mutation checks).
    spatial_averaging : bool, optional
        Whether to spatially homogenize stimulated transitions across the gain region.
    with_bus : bool, optional
        Whether to include the evanescently coupled bus waveguide output port.
    dx_m : float, optional
        Grid cell size ``dx = dy`` [m].
    n_steps : int, optional
        Number of leapfrog steps to execute [-].
    seed_amp : float, optional
        Optional external Gaussian current seed amplitude [A] (defaults to ``0.0`` for pure
        spontaneous-emission noise startup).
    spontaneous_emission : bool, optional
        Enable ``FourLevelGain`` Langevin spontaneous-emission dipole noise (defaults to ``True``).
    noise_amp : float, optional
        Dimensionless spontaneous-emission fluctuation scale ``noise_amplitude`` [-].
    seed : int, optional
        Deterministic PRNG seed for the spontaneous-emission noise generator [-].
    time_scale : float, optional
        Dimensionless rate-equation compression factor ``s <= 100`` [-].
    record_2d_frames : int, optional
        Number of 2D ``E_z(x, y)`` frames to capture during build-up for GIF animation [-].
    record_2d_inversion : bool, optional
        If ``True``, return the 2D steady-state population inversion map ``Delta N(x, y)``.
    mem_limit_mb : float, optional
        Host memory limit [MB].

    Returns
    -------
    out : dict
        Time traces, spectra, steady-state intensities, clamped inversion, and optional 2D maps.
    """
    ref_m6 = solve_wgm_complex_root(DEFAULT_M_TARGET, DEFAULT_Q_TARGET, polarization="TM")
    f_m6 = float(ref_m6["frequency_hz"])
    omega_a = 2.0 * np.pi * f_m6
    g_th_ring = float(ref_m6["g_th_ring_1_m"])

    fg_ref = FourLevelGain(
        omega_a=omega_a,
        gamma_a=DEFAULT_GAMMA_A,
        tau_21=DEFAULT_TAU_21,
        tau_10=DEFAULT_TAU_10,
        tau_32=DEFAULT_TAU_32,
        N_total=DEFAULT_N_TOTAL,
        pump_rate=0.0,
        eps_inf=DEFAULT_N_DISK**2,
        time_scale=float(time_scale),
    )
    wp_th = float(fg_ref.threshold_pump_rate(g_th_ring))
    sigma_21_nom = float(fg_ref.emission_cross_section)
    dn_th_nom = g_th_ring / sigma_21_nom
    i_sat_nom = float(fg_ref.saturation_intensity(W_p=wp_th))

    wp = max(0.0, float(pump_ratio) * wp_th)
    bus_w = DEFAULT_BUS_WIDTH_M if with_bus else None

    solver, fills, structures = build_microdisk_solver(
        dx_m=dx_m,
        domain_half_m=1.792e-6,
        r_disk_m=DEFAULT_R_DISK_M,
        n_disk=DEFAULT_N_DISK,
        bus_width_m=bus_w,
        bus_gap_m=DEFAULT_BUS_GAP_M,
        subpixel=True,
        n_pml=10,
    )
    lz = float(solver.grid.zmax - solver.grid.zmin)

    prof_key = pump_profile.strip().lower()
    if prof_key == "ring":
        fill_gain = fills["fill_ez_ring"]
    elif prof_key == "full":
        fill_gain = fills["fill_ez_disk"]
    elif prof_key == "core":
        fill_gain = fills["fill_ez_core"]
    else:
        raise ValueError(f"Unknown pump_profile {pump_profile!r}; expected 'ring', 'full', or 'core'.")

    fg = FourLevelGain(
        omega_a=omega_a,
        gamma_a=DEFAULT_GAMMA_A,
        kappa_a=fg_ref.kappa_a * float(kappa_scale),
        tau_21=DEFAULT_TAU_21,
        tau_10=DEFAULT_TAU_10,
        tau_32=DEFAULT_TAU_32,
        N_total=DEFAULT_N_TOTAL,
        pump_rate=wp,
        eps_inf=DEFAULT_N_DISK**2,
        initial_populations="steady_state" if wp > 0.0 else "ground",
        enable_stimulated=True,
        spatial_averaging=bool(spatial_averaging),
        spontaneous_emission=bool(spontaneous_emission),
        noise_components="z",
        noise_amplitude=float(noise_amp),
        seed=int(seed),
        time_scale=float(time_scale),
    )

    n3 = solver.Nx * solver.Ny * solver.Nz
    weights_all = fill_gain.ravel(order="F")
    sel = np.flatnonzero(weights_all > 0.0)
    ez_idx = sel + 2 * n3
    weights = weights_all[sel]
    eps_zz_all = (1.0 + (DEFAULT_N_DISK**2 - 1.0) * fills["fill_ez_total"]).ravel(order="F")
    eps_r_sel = eps_zz_all[sel]
    eps_bg = np.where(
        weights < 1.0,
        (eps_r_sel - weights * (DEFAULT_N_DISK**2)) / np.maximum(1.0 - weights, 1e-12),
        1.0,
    )
    solver._add_gain_edges(fg, ez_idx, weights, eps_bg=eps_bg)

    s_rim = None
    if float(seed_amp) > 0.0:
        gw = GaussianSource(frequency=f_m6, fwidth=0.08 * f_m6, cutoff=3.5)
        s_rim = CurrentSource(
            gw,
            component="z",
            center=(0.82 * DEFAULT_R_DISK_M, 0.0, 0.0),
            size=(0.0, 0.0, lz),
            amplitude=float(seed_amp),
        )

    nt = int(n_steps)
    est = preflight_solver(
        solver,
        n_steps=nt,
        dispersive_edges=int(sel.size),
        n_poles=1,
        mem_limit_mb=mem_limit_mb,
    )

    x_arr = solver.x
    y_arr = solver.y
    ix_rim = int(np.argmin(np.abs(x_arr - 0.84 * DEFAULT_R_DISK_M)))
    iy_rim = int(np.argmin(np.abs(y_arr)))

    y_bus_center = -(DEFAULT_R_DISK_M + DEFAULT_BUS_GAP_M + 0.5 * DEFAULT_BUS_WIDTH_M)
    ix_bus = int(np.argmin(np.abs(x_arr - 1.15e-6)))
    iy_bus = int(np.argmin(np.abs(y_arr - y_bus_center)))

    ez_rim = np.empty(nt, dtype=float)
    ez_bus = np.empty(nt, dtype=float)
    i_cav_tr = np.empty(nt, dtype=float)
    dn_mean = np.empty(nt, dtype=float)
    dn_modal = np.empty(nt, dtype=float)

    g_grp = solver._gain_groups[0]
    cw = g_grp["cell_weight"] / g_grp["eff_n_cells"]

    n_tail = min(6000, max(500, nt // 4))
    step_tail_start = nt - n_tail
    e2_tail_acc = np.zeros(sel.size, dtype=float) if record_2d_inversion else None

    frames_2d = []
    frames_t_fs = []
    frames_dn_norm = []
    if int(record_2d_frames) > 0:
        fr_steps = set(
            np.unique(
                np.linspace(int(0.05 * nt), nt - 1, int(record_2d_frames)).round().astype(int)
            ).tolist()
        )
    else:
        fr_steps = set()

    t0_w = time.perf_counter()
    for step in range(nt):
        t = step * solver.dt
        if s_rim is not None:
            s_rim.update(solver, t)
        solver.one_step()
        ez_rim[step] = float(solver.E[ix_rim, iy_rim, 0, "z"])
        ez_bus[step] = float(solver.E[ix_bus, iy_bus, 0, "z"])
        e2_c = solver.E.array[ez_idx] ** 2
        i_cav_tr[step] = c_light * epsilon_0 * DEFAULT_N_DISK * float(e2_c @ cw)
        if e2_tail_acc is not None and step >= step_tail_start:
            e2_tail_acc += e2_c
        n_pop = g_grp["N"]
        dn_c = n_pop[2] - n_pop[1]
        dn_mean[step] = float(dn_c @ cw)
        w_mod = e2_c * cw
        dn_modal[step] = float((dn_c @ w_mod) / max(float(np.sum(w_mod)), 1e-30))
        if step in fr_steps:
            frames_2d.append(solver.E[:, :, 0, "z"].copy())
            frames_t_fs.append((t + solver.dt) * 1e15)
            frames_dn_norm.append(float(dn_modal[step] / dn_th_nom))
    wall_s = time.perf_counter() - t0_w

    t_arr = np.arange(nt) * solver.dt
    i_cav_late = float(np.mean(i_cav_tr[-n_tail:]))
    i_bus_late = c_light * epsilon_0 * DEFAULT_N_DISK * float(np.mean(ez_bus[-n_tail:] ** 2))

    # Build-up ratio (late steady state vs early spontaneous-emission noise) and clamping ratio
    i_early_0 = min(200, max(20, nt // 25))
    i_early_1 = min(1000, max(i_early_0 + 100, nt // 6))
    n_clamp_win = min(3000, max(500, n_tail // 2))
    i_prev_0 = max(i_early_1, nt - 2 * n_clamp_win)
    i_prev_1 = nt - n_clamp_win

    rms_early = float(np.sqrt(np.mean(i_cav_tr[i_early_0:i_early_1]) / (c_light * epsilon_0 * DEFAULT_N_DISK)))
    rms_prev = float(np.sqrt(np.mean(i_cav_tr[i_prev_0:i_prev_1]) / (c_light * epsilon_0 * DEFAULT_N_DISK)))
    rms_late = float(np.sqrt(np.mean(i_cav_tr[-n_tail:]) / (c_light * epsilon_0 * DEFAULT_N_DISK)))
    rms_clamp_late = float(np.sqrt(np.mean(i_cav_tr[-n_clamp_win:]) / (c_light * epsilon_0 * DEFAULT_N_DISK)))

    buildup_gain_ratio = rms_late / max(rms_early, 1e-30)
    late_clamping_ratio = rms_clamp_late / max(rms_prev, 1e-30)

    # Threshold-crossing time (first step where intracavity intensity reaches 0.5 * I_sat)
    s_tr = i_cav_tr / i_sat_nom
    idx_cross = np.flatnonzero(s_tr >= 0.5)
    t_end_fs = float(nt * solver.dt * 1e15)
    t_cross_fs = float(t_arr[int(idx_cross[0])] * 1e15) if idx_cross.size > 0 else t_end_fs
    post_th_duration_fs = float(max(0.0, t_end_fs - t_cross_fs))
    tau21_eff_fs = float((DEFAULT_TAU_21 / float(time_scale)) * 1e15)
    post_th_tau21_eff = float(post_th_duration_fs / max(tau21_eff_fs, 1e-30))

    # Smooth dn_modal over 1 optical cycle (~120 steps) for plotting without 2*omega_0 carrier ripple
    n_cyc = min(120, max(1, nt // 20))
    pad_l = n_cyc // 2
    pad_r = n_cyc - 1 - pad_l
    dn_modal_padded = np.pad(dn_modal / dn_th_nom, (pad_l, pad_r), mode="edge")
    dn_modal_smooth = np.convolve(dn_modal_padded, np.ones(n_cyc, dtype=float) / float(n_cyc), mode="valid")

    # Late-time spectrum over the saturated half of the trajectory
    sig_late = ez_rim[nt // 2 :]
    win = np.hanning(sig_late.size)
    n_fft = 16384
    freqs_fft = np.fft.rfftfreq(n_fft, d=solver.dt)
    spec_fft = np.abs(np.fft.rfft(sig_late * win, n=n_fft))
    spec_pwr = spec_fft**2
    freqs_thz = freqs_fft * 1e-12
    mask_band = (freqs_thz >= 150.0) & (freqs_thz <= 260.0)
    idx_band = np.flatnonzero(mask_band)
    idx_pk = int(idx_band[np.argmax(spec_pwr[idx_band])])
    f_lasing_thz = float(freqs_thz[idx_pk])

    # Side-mode suppression ratio between dominant TM_{6,1} (~202 THz) and adjacent TM_{7,1} (~230 THz)
    mask_m6 = (freqs_thz >= 195.0) & (freqs_thz <= 208.0)
    mask_m7 = (freqs_thz >= 224.0) & (freqs_thz <= 236.0)
    p_m6 = float(np.max(spec_pwr[mask_m6])) if np.any(mask_m6) else 1e-30
    p_m7 = float(np.max(spec_pwr[mask_m7])) if np.any(mask_m7) else 1e-30
    smsr_db = float(10.0 * np.log10(max(p_m6, 1e-30) / max(p_m7, 1e-30)))

    out = {
        "pump_ratio": float(pump_ratio),
        "pump_profile": prof_key,
        "kappa_scale": float(kappa_scale),
        "spatial_averaging": bool(spatial_averaging),
        "spontaneous_emission": bool(spontaneous_emission),
        "time_scale": float(time_scale),
        "W_p_1_s": float(wp),
        "W_p_th_1_s": float(wp_th),
        "dN_th_m3": float(dn_th_nom),
        "I_sat_w_m2": float(i_sat_nom),
        "I_cav_late_w_m2": float(i_cav_late),
        "I_bus_late_w_m2": float(i_bus_late),
        "s_cav_late": float(i_cav_late / i_sat_nom),
        "rms_mid_v_m": rms_early,
        "rms_prev_v_m": rms_prev,
        "rms_late_v_m": rms_late,
        "gain_ratio": float(buildup_gain_ratio),
        "buildup_gain_ratio": float(buildup_gain_ratio),
        "late_clamping_ratio": float(late_clamping_ratio),
        "t_end_fs": t_end_fs,
        "t_cross_fs": t_cross_fs,
        "post_th_duration_fs": post_th_duration_fs,
        "post_th_tau21_eff": post_th_tau21_eff,
        "dN_init_over_th": float(dn_mean[0] / dn_th_nom),
        "dN_prev_over_th": float(np.mean(dn_mean[i_prev_0:i_prev_1]) / dn_th_nom),
        "dN_final_over_th": float(np.mean(dn_mean[-n_clamp_win:]) / dn_th_nom),
        "dN_modal_prev_over_th": float(np.mean(dn_modal[i_prev_0:i_prev_1]) / dn_th_nom),
        "dN_modal_final_over_th": float(np.mean(dn_modal[-n_clamp_win:]) / dn_th_nom),
        "f_lasing_thz": f_lasing_thz,
        "smsr_tm6_over_tm7_db": smsr_db,
        "wall_s": float(wall_s),
        "memory_mb": float(est["memory_mb"]),
        "t_fs": t_arr * 1e15,
        "ez_rim": ez_rim,
        "ez_bus": ez_bus,
        "i_cav_over_sat": i_cav_tr / i_sat_nom,
        "dn_over_th": dn_mean / dn_th_nom,
        "dn_modal_over_th": dn_modal_smooth,
        "freqs_thz": freqs_thz,
        "spec_power": spec_pwr,
    }

    if record_2d_inversion and e2_tail_acc is not None:
        dn_2d = np.zeros(solver.Nx * solver.Ny, dtype=float)
        # Steady-state standing-wave saturation map locked to the final standing-wave phase
        s_cells_tail = (c_light * epsilon_0 * DEFAULT_N_DISK * (e2_tail_acc / float(n_tail))) / i_sat_nom
        e2_final_c = solver.E.array[ez_idx] ** 2
        s_sw_cells = e2_final_c * (float(np.mean(s_cells_tail)) / max(float(np.mean(e2_final_c)), 1e-30))
        dn_init_norm = float(dn_mean[0] / dn_th_nom)
        dn_cells_steady = dn_init_norm / (1.0 + s_sw_cells)
        dn_2d[g_grp["cells"]] = dn_cells_steady
        out["x_m"] = x_arr.copy()
        out["y_m"] = y_arr.copy()
        out["dn_2d_over_th"] = dn_2d.reshape((solver.Nx, solver.Ny), order="F")
        out["ez_2d_final"] = solver.E[:, :, 0, "z"].copy()

    if int(record_2d_frames) > 0:
        out["frames_ez"] = np.asarray(frames_2d)
        out["frames_t_fs"] = np.asarray(frames_t_fs)
        out["frames_dn_norm"] = np.asarray(frames_dn_norm)
    return out


def run_task_lasing(quick: bool = False) -> dict:
    """Run Task 3: Active FourLevelGain microdisk noise startup, clamping, L-pump sweep, and mutations."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dx_m = 0.032e-6
    n_steps = 8500 if quick else 24000
    n_frames = 16 if quick else 24

    # 1. Main above-threshold lasing build-up trace (r_p = 2.4, ring pump, spatial hole burning enabled)
    above_shb = run_single_four_level_microdisk(
        pump_ratio=2.40,
        pump_profile="ring",
        spatial_averaging=False,
        with_bus=True,
        dx_m=dx_m,
        n_steps=n_steps,
        record_2d_frames=n_frames,
        record_2d_inversion=True,
    )

    # 2. Below-threshold mutation check (r_p = 0.50, ring pump -> remains weak ASE noise)
    below_mut = run_single_four_level_microdisk(
        pump_ratio=0.50,
        pump_profile="ring",
        spatial_averaging=False,
        with_bus=True,
        dx_m=dx_m,
        n_steps=n_steps,
        record_2d_frames=0,
        record_2d_inversion=False,
    )

    # 3. L-pump sweep across r_p in [0.0, 0.50, 0.85, 1.45, 2.05, 2.40]
    pump_ratios = [0.0, 0.50, 0.85, 1.45, 2.05, 2.40]
    li_shb_cav = []
    li_shb_bus = []
    li_hom_cav = []
    li_gain_ratios = []
    li_clamping_ratios = []
    li_dn_final = []
    li_dn_shb_modal = []
    li_dn_shb_ring = []
    total_wall_s = float(above_shb["wall_s"] + below_mut["wall_s"])
    max_mem_mb = max(float(above_shb["memory_mb"]), float(below_mut["memory_mb"]))

    prof_ring = None
    mut_kappa_nom = None
    above_hom = None
    for rp in pump_ratios:
        if abs(rp - 2.40) < 1e-9:
            r_s = above_shb
        elif abs(rp - 0.50) < 1e-9:
            r_s = below_mut
        else:
            r_s = run_single_four_level_microdisk(
                pump_ratio=rp,
                pump_profile="ring",
                spatial_averaging=False,
                with_bus=True,
                dx_m=dx_m,
                n_steps=n_steps,
            )
            total_wall_s += float(r_s["wall_s"])
        if abs(rp - 2.05) < 1e-9:
            prof_ring = r_s
        if abs(rp - 1.45) < 1e-9:
            mut_kappa_nom = r_s
        if abs(rp - 0.0) < 1e-9:
            r_h = r_s
        else:
            r_h = run_single_four_level_microdisk(
                pump_ratio=rp,
                pump_profile="ring",
                spatial_averaging=True,
                with_bus=True,
                dx_m=dx_m,
                n_steps=n_steps,
            )
            total_wall_s += float(r_h["wall_s"])
        if abs(rp - 2.40) < 1e-9:
            above_hom = r_h
        li_shb_cav.append(float(r_s["I_cav_late_w_m2"]))
        li_shb_bus.append(float(r_s["I_bus_late_w_m2"]))
        li_hom_cav.append(float(r_h["I_cav_late_w_m2"]))
        li_gain_ratios.append(float(r_s["gain_ratio"]))
        li_clamping_ratios.append(float(r_s["late_clamping_ratio"]))
        li_dn_final.append(float(r_h["dN_final_over_th"]))
        li_dn_shb_modal.append(float(r_s["dN_modal_final_over_th"]))
        li_dn_shb_ring.append(float(r_s["dN_final_over_th"]))

    # 4. Pump-profile mode selection at r_p = 2.05: ring vs full disk vs core
    assert prof_ring is not None
    assert mut_kappa_nom is not None
    assert above_hom is not None
    prof_full = run_single_four_level_microdisk(
        pump_ratio=2.05,
        pump_profile="full",
        spatial_averaging=False,
        with_bus=True,
        dx_m=dx_m,
        n_steps=n_steps,
    )
    prof_core = run_single_four_level_microdisk(
        pump_ratio=2.05,
        pump_profile="core",
        spatial_averaging=False,
        with_bus=True,
        dx_m=dx_m,
        n_steps=n_steps,
    )
    total_wall_s += float(prof_full["wall_s"] + prof_core["wall_s"])

    # 5. Coupling mutation check: kappa_a -> 0.5 * kappa_a, 1.0 * kappa_a, 2.0 * kappa_a at r_p = 1.45
    mut_kappa_half = run_single_four_level_microdisk(
        pump_ratio=1.45,
        pump_profile="ring",
        kappa_scale=0.5,
        spatial_averaging=False,
        with_bus=True,
        dx_m=dx_m,
        n_steps=n_steps,
    )
    mut_kappa_double = run_single_four_level_microdisk(
        pump_ratio=1.45,
        pump_profile="ring",
        kappa_scale=2.0,
        spatial_averaging=False,
        with_bus=True,
        dx_m=dx_m,
        n_steps=n_steps,
    )
    total_wall_s += float(mut_kappa_half["wall_s"] + mut_kappa_double["wall_s"])

    ref_m6 = solve_wgm_complex_root(DEFAULT_M_TARGET, DEFAULT_Q_TARGET, polarization="TM")
    ref_m7 = solve_wgm_complex_root(7, 1, polarization="TM")
    cpw_m6 = float((c_light / (DEFAULT_N_DISK * float(ref_m6["frequency_hz"]))) / dx_m)
    cpw_m7 = float((c_light / (DEFAULT_N_DISK * float(ref_m7["frequency_hz"]))) / dx_m)

    li_theory = wgm_four_level_li_curve(
        pump_ratios=np.linspace(0.0, 2.5, 101),
        g_th_1_m=float(ref_m6["g_th_ring_1_m"]),
        omega_a=2.0 * np.pi * float(ref_m6["frequency_hz"]),
    )
    li_theory_pts = wgm_four_level_li_curve(
        pump_ratios=np.array(pump_ratios, dtype=float),
        g_th_1_m=float(ref_m6["g_th_ring_1_m"]),
        omega_a=2.0 * np.pi * float(ref_m6["frequency_hz"]),
    )
    scaling_info = lifetime_scaling_laws(
        scale_factor=DEFAULT_TIME_SCALE,
        g_th_1_m=float(ref_m6["g_th_ring_1_m"]),
        q_cold=float(ref_m6["q_rad"]),
    )

    f_ana_m6 = float(ref_m6["frequency_thz"])
    f_las_sim = float(above_shb["f_lasing_thz"])
    f_las_rel_err = (f_las_sim - f_ana_m6) / f_ana_m6

    i_sat_val = float(above_shb["I_sat_w_m2"])
    s_hom_sim = [float(v / i_sat_val) for v in li_hom_cav]
    s_shb_sim = [float(v / i_sat_val) for v in li_shb_cav]
    s_hom_th_pts = [float(v) for v in li_theory_pts["s_hom"]]
    s_shb_th_pts = [float(v) for v in li_theory_pts["s_rad_ring"]]
    s_shb_sw_th_pts = [float(v) for v in li_theory_pts["s_shb_ring"]]

    shb_red_sim_pct = float(100.0 * (1.0 - s_shb_sim[-1] / max(s_hom_sim[-1], 1e-30)))
    shb_red_th_pct = float(100.0 * (1.0 - s_shb_th_pts[-1] / max(s_hom_th_pts[-1], 1e-30)))
    shb_sw_red_th_pct = float(100.0 * (1.0 - s_shb_sw_th_pts[-1] / max(s_hom_th_pts[-1], 1e-30)))
    li_hom_rel_err_r240 = float((s_hom_sim[-1] - s_hom_th_pts[-1]) / s_hom_th_pts[-1])
    li_shb_rel_err_r240 = float((s_shb_sim[-1] - s_shb_th_pts[-1]) / s_shb_th_pts[-1])
    li_hom_rel_err_above = [
        float((s_hom_sim[i] - s_hom_th_pts[i]) / s_hom_th_pts[i]) for i in (3, 4, 5)
    ]
    li_shb_rel_err_above = [
        float((s_shb_sim[i] - s_shb_th_pts[i]) / s_shb_th_pts[i]) for i in (3, 4, 5)
    ]

    summary = {
        "task": "lasing",
        "quick": bool(quick),
        "dx_nm": float(dx_m * 1e9),
        "cpw_m6": cpw_m6,
        "cpw_m7": cpw_m7,
        "n_steps": int(n_steps),
        "t_end_fs": float(above_shb["t_end_fs"]),
        "above_shb_t_cross_fs": float(above_shb["t_cross_fs"]),
        "above_shb_post_th_fs": float(above_shb["post_th_duration_fs"]),
        "above_shb_post_th_tau21_eff": float(above_shb["post_th_tau21_eff"]),
        "above_hom_t_cross_fs": float(above_hom["t_cross_fs"]),
        "above_hom_post_th_fs": float(above_hom["post_th_duration_fs"]),
        "above_hom_post_th_tau21_eff": float(above_hom["post_th_tau21_eff"]),
        "above_shb_dN_modal_prev_over_th": float(above_shb["dN_modal_prev_over_th"]),
        "above_shb_dN_modal_final_over_th": float(above_shb["dN_modal_final_over_th"]),
        "above_hom_dN_prev_over_th": float(above_hom["dN_prev_over_th"]),
        "above_hom_dN_final_over_th": float(above_hom["dN_final_over_th"]),
        "spontaneous_emission": True,
        "noise_amplitude": float(DEFAULT_NOISE_AMP),
        "noise_seed": int(DEFAULT_NOISE_SEED),
        "time_scale": float(DEFAULT_TIME_SCALE),
        "total_wall_s": total_wall_s,
        "memory_mb": max_mem_mb,
        "W_p_th_1_s": float(above_shb["W_p_th_1_s"]),
        "dN_th_m3": float(above_shb["dN_th_m3"]),
        "I_sat_w_m2": i_sat_val,
        "f_m6_ana_thz": f_ana_m6,
        "f_lasing_sim_thz": f_las_sim,
        "f_lasing_rel_err": float(f_las_rel_err),
        "smsr_tm6_over_tm7_db": float(above_shb["smsr_tm6_over_tm7_db"]),
        "above_th_gain_ratio": float(above_shb["gain_ratio"]),
        "above_th_late_clamping_ratio": float(above_shb["late_clamping_ratio"]),
        "below_th_gain_ratio": float(below_mut["gain_ratio"]),
        "below_th_late_clamping_ratio": float(below_mut["late_clamping_ratio"]),
        "above_over_below_intensity_ratio": float(
            above_shb["I_cav_late_w_m2"] / max(below_mut["I_cav_late_w_m2"], 1e-30)
        ),
        "shb_reduction_pct_r240": shb_red_sim_pct,
        "shb_theory_reduction_pct_r240": shb_red_th_pct,
        "shb_sw_theory_reduction_pct_r240": shb_sw_red_th_pct,
        "li_hom_vs_theory_rel_err_r240": li_hom_rel_err_r240,
        "li_shb_vs_theory_rel_err_r240": li_shb_rel_err_r240,
        "li_hom_vs_theory_rel_err_above": li_hom_rel_err_above,
        "li_shb_vs_theory_rel_err_above": li_shb_rel_err_above,
        "pump_ratios": pump_ratios,
        "li_shb_cav_w_m2": li_shb_cav,
        "li_shb_bus_w_m2": li_shb_bus,
        "li_hom_cav_w_m2": li_hom_cav,
        "li_hom_s_sim": s_hom_sim,
        "li_shb_s_sim": s_shb_sim,
        "li_hom_s_theory": s_hom_th_pts,
        "li_shb_s_theory": s_shb_th_pts,
        "li_shb_sw_s_theory": s_shb_sw_th_pts,
        "li_gain_ratios": li_gain_ratios,
        "li_clamping_ratios": li_clamping_ratios,
        "li_dn_final_over_th": li_dn_final,
        "li_dn_shb_modal_final_over_th": li_dn_shb_modal,
        "li_dn_shb_ring_final_over_th": li_dn_shb_ring,
        "profile_selection": {
            "ring_I_cav_w_m2": float(prof_ring["I_cav_late_w_m2"]),
            "full_I_cav_w_m2": float(prof_full["I_cav_late_w_m2"]),
            "core_I_cav_w_m2": float(prof_core["I_cav_late_w_m2"]),
            "ring_gain_ratio": float(prof_ring["gain_ratio"]),
            "full_gain_ratio": float(prof_full["gain_ratio"]),
            "core_gain_ratio": float(prof_core["gain_ratio"]),
            "ring_to_core_intensity_ratio": float(
                prof_ring["I_cav_late_w_m2"] / max(prof_core["I_cav_late_w_m2"], 1e-30)
            ),
            "ring_area_saving_pct": float(100.0 * DEFAULT_R_IN_FRAC**2),
            "ring_confinement_retention_pct": float(
                100.0 * ref_m6["confinement_ring"] / ref_m6["confinement_disk"]
            ),
        },
        "mutation_kappa_a": {
            "pump_ratio_test": 1.45,
            "kappa_half_gain_ratio": float(mut_kappa_half["gain_ratio"]),
            "kappa_nom_gain_ratio": float(mut_kappa_nom["gain_ratio"]),
            "kappa_double_gain_ratio": float(mut_kappa_double["gain_ratio"]),
            "kappa_half_I_cav_w_m2": float(mut_kappa_half["I_cav_late_w_m2"]),
            "kappa_nom_I_cav_w_m2": float(mut_kappa_nom["I_cav_late_w_m2"]),
            "kappa_double_I_cav_w_m2": float(mut_kappa_double["I_cav_late_w_m2"]),
        },
        "lifetime_scaling": scaling_info,
    }

    with open(DATA_DIR / "task_03_lasing.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    np.savez_compressed(
        DATA_DIR / "task_03_lasing.npz",
        t_fs=above_shb["t_fs"],
        ez_rim_above=above_shb["ez_rim"],
        ez_bus_above=above_shb["ez_bus"],
        i_cav_above=above_shb["i_cav_over_sat"],
        dn_above=above_shb["dn_modal_over_th"],
        dn_ring_above=above_shb["dn_over_th"],
        dn_hom_above=above_hom["dn_over_th"],
        ez_rim_below=below_mut["ez_rim"],
        dn_below=below_mut["dn_over_th"],
        freqs_thz=above_shb["freqs_thz"],
        spec_above=above_shb["spec_power"],
        spec_below=below_mut["spec_power"],
        spec_ring=prof_ring["spec_power"],
        spec_full=prof_full["spec_power"],
        spec_core=prof_core["spec_power"],
        pump_ratios=np.array(pump_ratios),
        li_shb_cav_w_m2=np.array(li_shb_cav),
        li_shb_bus_w_m2=np.array(li_shb_bus),
        li_hom_cav_w_m2=np.array(li_hom_cav),
        li_dn_final_over_th=np.array(li_dn_final),
        li_dn_shb_modal_over_th=np.array(li_dn_shb_modal),
        theory_pump_ratios=li_theory["pump_ratios"],
        theory_s_hom=li_theory["s_hom"],
        theory_s_shb=li_theory["s_rad_ring"],
        theory_s_shb_sw=li_theory["s_shb_ring"],
        theory_dN_clamped=li_theory["dN_clamped_m3"] / li_theory["dN_th_m3"],
        x_m=above_shb["x_m"],
        y_m=above_shb["y_m"],
        dn_2d_over_th=above_shb["dn_2d_over_th"],
        ez_2d_final=above_shb["ez_2d_final"],
        frames_ez=above_shb["frames_ez"],
        frames_t_fs=above_shb["frames_t_fs"],
        frames_dn_norm=above_shb["frames_dn_norm"],
    )
    return summary


# ---------------------------------------------------------------------------
# Publication plotting helpers (wakis.figstyle & wakis.schematic)
# ---------------------------------------------------------------------------


import logging

_OKABE_MAP: dict[str, str] = {
    "blue": figstyle.OKABE_ITO[0],
    "red": figstyle.OKABE_ITO[1],
    "vermilion": figstyle.OKABE_ITO[1],
    "green": figstyle.OKABE_ITO[2],
    "orange": figstyle.OKABE_ITO[3],
    "purple": figstyle.OKABE_ITO[4],
    "sky": figstyle.OKABE_ITO[5],
    "ochre": figstyle.OKABE_ITO[6],
    "black": figstyle.OKABE_ITO[7],
    "grey": "#64748b",
}


def _apply_paper_style() -> None:
    """Apply Wakis paper figure style."""
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    figstyle.apply_style()


def plot_00_setup_schematics() -> list[Path]:
    """Generate 00a/00b setup figures and 00_setup_table.md via wakis.schematic.save_setup_figures."""
    _apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    solver, _, structures = build_microdisk_solver(
        dx_m=0.028e-6,
        domain_half_m=1.792e-6,
        r_disk_m=DEFAULT_R_DISK_M,
        n_disk=DEFAULT_N_DISK,
        bus_width_m=DEFAULT_BUS_WIDTH_M,
        bus_gap_m=DEFAULT_BUS_GAP_M,
        subpixel=True,
        n_pml=10,
    )
    lz = float(solver.grid.zmax - solver.grid.zmin)
    ref_m6 = solve_wgm_complex_root(DEFAULT_M_TARGET, DEFAULT_Q_TARGET, polarization="TM")
    f0 = float(ref_m6["frequency_hz"])
    lam0 = c_light / f0

    gw = GaussianSource(frequency=f0, fwidth=0.08 * f0, cutoff=3.5)
    src = CurrentSource(
        gw,
        component="z",
        center=(0.72 * DEFAULT_R_DISK_M, 0.45 * DEFAULT_R_DISK_M, 0.0),
        size=(0.0, 0.0, lz),
        amplitude=1.0e-4,
    )
    mon_dft = DFTMonitor(
        [f0],
        components=("Ez",),
        center=(0.0, 0.0, 0.0),
        size=(2.6e-6, 2.6e-6, 0.0),
    )
    y_bus_c = -(DEFAULT_R_DISK_M + DEFAULT_BUS_GAP_M + 0.5 * DEFAULT_BUS_WIDTH_M)
    probes = [
        {"pos": (-0.72 * DEFAULT_R_DISK_M, 0.45 * DEFAULT_R_DISK_M, 0.0), "label": "WGM rim probe"},
        {"pos": (0.85e-6, y_bus_c, 0.0), "label": "Bus outport probe"},
    ]

    res = schematic.save_setup_figures(
        solver,
        FIG_DIR,
        plane="xy",
        structures=structures,
        sources=[src],
        monitors=[mon_dft],
        probes=probes,
        unit="µm",
        wavelength=lam0,
        freq=f0,
        min_feature=DEFAULT_BUS_WIDTH_M,
        title_geometry="Microdisk WGM Laser Setup: Gain Disk, Coupled Bus Waveguide, Sources & CPML (XY plane)",
        title_mesh="2D FIT Subpixel Mesh (1× Rim/Bus Coupling Gap Zoom) & Excitation Spectrum",
        variations=[
            {
                "name": "Part 1 · Cold cavity",
                "details": "Passive subpixel vs staircase microdisk (R = 1.0 µm, n = 2.0), TM & TE WGMs (m = 4..9, q = 1, 2)",
            },
            {
                "name": "Part 2 · Linear gain threshold",
                "details": "Unsaturated Lorentz gain pole scanning g across g_th = n ω / (c Γ Q) for m = 4..8",
            },
            {
                "name": "Part 3 · FourLevelGain lasing from noise",
                "details": (
                    "Spontaneous-emission noise startup (spontaneous_emission=True, s = 100) in annular ring "
                    "(r ∈ [0.55 R, R]) vs full-disk & core pump, coupled to 220 nm bus across 300 nm gap"
                ),
            },
        ],
    )
    table_path = FIG_DIR / "00_setup_table.md"
    table_path.write_text(res["table"] + "\n", encoding="utf-8")
    return [Path(res["geometry_figure"]), Path(res["mesh_figure"]), table_path]


def plot_01_cold_cavity_wgm(cold_json: dict, cold_npz: np.lib.npyio.NpzFile) -> Path:
    """Render Figure 1: Cold-cavity WGM spectra, analytical Bessel/Hankel benchmark, and 2D mode profiles."""
    _apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    okabe = _OKABE_MAP

    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN * 1.95,
        height_in=4.2,
        ncols=3,
        nrows=1,
        layout="constrained",
    )

    # Panel (a): Broadband TM & TE ringdown spectra + exact Bessel/Hankel vertical markers
    ax = axes[0]
    freqs_tm = cold_npz["freqs_thz"]
    spec_tm = cold_npz["spec_tm_sub"]
    freqs_te = cold_npz["freqs_te_thz"]
    spec_te = cold_npz["spec_te_sub"]

    mask_f = (freqs_tm >= 130.0) & (freqs_tm <= 330.0)
    ax.semilogy(freqs_tm[mask_f], np.maximum(spec_tm[mask_f], 1e-4), color=okabe["blue"], lw=1.6, label="TM ($E_z$) FIT spectrum")
    mask_te = (freqs_te >= 130.0) & (freqs_te <= 330.0)
    ax.semilogy(freqs_te[mask_te], np.maximum(spec_te[mask_te] * 0.4, 1e-4), color=okabe["orange"], lw=1.3, alpha=0.85, label="TE ($H_z$) FIT spectrum (×0.4)")

    for row in cold_json["tm_q1_modes"]:
        f_a = row["f_ana_thz"]
        m_v = row["m"]
        ax.axvline(f_a, color=okabe["blue"], ls="--", lw=0.9, alpha=0.65)
        ax.text(f_a, 1.18, f"TM$_{{{m_v},1}}$", rotation=90, ha="center", va="bottom", fontsize=7.5, color=okabe["blue"])

    ax.set_xlim(132.0, 310.0)
    ax.set_ylim(2e-4, 4.5)
    ax.set_title("(a) Cold-Cavity Ringdown Spectra")
    ax.set_xlabel("Frequency $f$ [THz]")
    ax.set_ylabel("Normalized spectral amplitude [-]")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="lower right", fontsize=8.0)

    # Panel (b): Radiative Q_rad(m) exponential growth with azimuthal order m (TM q=1,2 and TE q=1)
    ax = axes[1]
    m_tm1 = [r["m"] for r in cold_json["tm_q1_modes"]]
    q_tm1_ana = [r["q_ana"] for r in cold_json["tm_q1_modes"]]
    q_tm1_sub = [r["q_sub"] for r in cold_json["tm_q1_modes"]]
    q_tm1_st = [r["q_stair"] for r in cold_json["tm_q1_modes"]]

    m_tm2 = [r["m"] for r in cold_json["tm_q2_modes"]]
    q_tm2_ana = [r["q_ana"] for r in cold_json["tm_q2_modes"]]
    q_tm2_sub = [r["q_sub"] for r in cold_json["tm_q2_modes"]]

    m_te1 = [r["m"] for r in cold_json["te_q1_modes"]]
    q_te1_ana = [r["q_ana"] for r in cold_json["te_q1_modes"]]
    q_te1_sub = [r["q_sub"] for r in cold_json["te_q1_modes"]]

    ax.semilogy(m_tm1, q_tm1_ana, "-", color=okabe["blue"], lw=1.8, label="TM $q=1$ Bessel/Hankel exact")
    ax.semilogy(m_tm1, q_tm1_sub, "o", color=okabe["blue"], ms=6.0, mfc="white", mew=1.5, label="TM $q=1$ FIT (subpixel)")
    ax.semilogy(m_tm1, q_tm1_st, "x", color=okabe["red"], ms=6.0, mew=1.5, label="TM $q=1$ FIT (staircase)")

    ax.semilogy(m_te1, q_te1_ana, "--", color=okabe["orange"], lw=1.6, label="TE $q=1$ Bessel/Hankel exact")
    ax.semilogy(m_te1, q_te1_sub, "s", color=okabe["orange"], ms=5.5, mfc="white", mew=1.4, label="TE $q=1$ FIT (subpixel)")

    ax.semilogy(m_tm2, q_tm2_ana, ":", color=okabe["green"], lw=1.6, label="TM $q=2$ Bessel/Hankel exact")
    ax.semilogy(m_tm2, q_tm2_sub, "^", color=okabe["green"], ms=5.5, mfc="white", mew=1.4, label="TM $q=2$ FIT (subpixel)")

    ax.set_title("(b) Exponential Growth of $Q_{\\mathrm{rad}}(m)$")
    ax.set_xlabel("Azimuthal mode order $m$ [-]")
    ax.set_ylabel("Radiative quality factor $Q_{\\mathrm{rad}}$ [-]")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper left", fontsize=7.6, ncol=1)

    # Panel (c): 2D WGM mode profile Re(E_z) for TM_{6,1} with schematic.overlay_geometry
    ax = axes[2]
    x_um = cold_npz["x_dft_m"] * 1e6
    y_um = cold_npz["y_dft_m"] * 1e6
    ez_61 = cold_npz["ez_m6_q1"]
    vmax = max(float(np.percentile(np.abs(ez_61), 99.5)), 1e-12)
    im = ax.pcolormesh(
        x_um,
        y_um,
        ez_61.T / vmax,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-1.0,
        vmax=+1.0,
        shading="auto",
    )
    solver_ov, _, structs_ov = build_microdisk_solver(dx_m=0.028e-6, domain_half_m=1.4e-6, bus_width_m=None)
    schematic.overlay_geometry(
        ax,
        solver_ov,
        plane="xy",
        unit="µm",
        structures=structs_ov,
        show_sources=False,
        show_monitors=False,
        show_boundaries=False,
        fill_materials=False,
    )
    # Mark inner boundary of annular pump r = 0.55 R
    ax.add_patch(
        Circle((0.0, 0.0), DEFAULT_R_IN_FRAC * DEFAULT_R_DISK_M * 1e6, fill=False, ls="--", lw=1.1, edgecolor=okabe["orange"])
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title(r"(c) $\mathrm{TM}_{6,1}$ Mode $\mathrm{Re}[E_z(x,y)]$ ($2m=12$ lobes)")
    ax.set_xlabel("$x$ [µm]")
    ax.set_ylabel("$y$ [µm]")
    fig.colorbar(im, ax=ax, shrink=0.84, label=r"Normalized $\mathrm{Re}(E_z)$ [-]")

    figstyle.finish(fig, auto_legend_below=False)
    out = figstyle.save(fig, FIG_DIR / "01_cold_cavity_wgm.png", auto_legend_below=False)
    plt.close(fig)
    return Path(out)


def plot_02_convergence_and_staircase(cold_json: dict, cold_npz: np.lib.npyio.NpzFile) -> Path:
    """Render Figure 2: Log-log grid convergence with order guides and subpixel vs staircase Q error."""
    _apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    okabe = _OKABE_MAP

    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN * 1.75,
        height_in=4.1,
        ncols=3,
        nrows=1,
        layout="constrained",
    )

    dx_nm = cold_npz["dx_levels_nm"]
    err_f6_sub = np.abs(cold_npz["err_f6_sub"]) * 100.0
    err_f6_st = np.abs(cold_npz["err_f6_stair"]) * 100.0
    err_q8_sub = np.abs(cold_npz["err_q8_sub"]) * 100.0
    err_q8_st = np.abs(cold_npz["err_q8_stair"]) * 100.0

    # Panel (a): Log-log frequency convergence |df_{6,1}|/f_{6,1} vs dx with O(dx^2) and O(dx) guides
    ax = axes[0]
    ax.loglog(dx_nm, err_f6_sub, "o-", color=okabe["blue"], lw=1.9, ms=6.5, label=f"Subpixel FIT ($p={cold_json['p_obs_f6_sub']:.2f}$)")
    ax.loglog(dx_nm, err_f6_st, "s--", color=okabe["red"], lw=1.7, ms=6.0, label="Staircase FIT")
    dx_guide = np.linspace(float(dx_nm[-1]) * 0.9, float(dx_nm[0]) * 1.05, 50)
    guide_2nd = err_f6_sub[-1] * (dx_guide / dx_nm[-1]) ** 2
    guide_1st = err_f6_st[-1] * (dx_guide / dx_nm[-1]) ** 1
    ax.loglog(dx_guide, guide_2nd, ":", color=okabe["black"], lw=1.3, label=r"$\mathcal{O}(\Delta x^2)$ reference")
    ax.loglog(dx_guide, guide_1st, "-.", color=okabe["grey"], lw=1.2, label=r"$\mathcal{O}(\Delta x)$ reference")
    ax.axhline(abs(cold_json["f6_rich_rel_err"]) * 100.0, color=okabe["green"], ls="--", lw=1.2, label=f"Richardson ({abs(cold_json['f6_rich_rel_err'])*100:.3f}%)")
    ax.set_title(r"(a) $\mathrm{TM}_{6,1}$ Frequency Convergence vs $\Delta x$")
    ax.set_xlabel(r"Grid cell size $\Delta x$ [nm]")
    ax.set_ylabel(r"Relative error $|f_{\mathrm{FIT}} - f_{\mathrm{exact}}| / f_{\mathrm{exact}}$ [%]")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="lower right", fontsize=7.8)

    # Panel (b): Log-log radiative Q convergence |dQ_{8,1}|/Q_{8,1} vs dx
    ax = axes[1]
    ax.loglog(dx_nm, err_q8_sub, "o-", color=okabe["blue"], lw=1.9, ms=6.5, label="Subpixel FIT $Q_{8,1}$")
    ax.loglog(dx_nm, err_q8_st, "s--", color=okabe["red"], lw=1.7, ms=6.0, label="Staircase FIT $Q_{8,1}$ (roughness loss)")
    q_guide2 = err_q8_sub[-1] * (dx_guide / dx_nm[-1]) ** 2
    ax.loglog(dx_guide, q_guide2, ":", color=okabe["black"], lw=1.3, label=r"$\mathcal{O}(\Delta x^2)$ reference")
    q8_unc = cold_json.get("q8_rich_uncert_rel", 0.014) * 100.0
    ax.axhline(
        abs(cold_json["q8_rich_rel_err"]) * 100.0,
        color=okabe["green"],
        ls="--",
        lw=1.2,
        label=f"Richardson ({abs(cold_json['q8_rich_rel_err'])*100:.2f}% ± {q8_unc:.2f}%)",
    )
    ax.set_title(r"(b) High-$Q$ ($\mathrm{TM}_{8,1}$) Radiative $Q$ Convergence")
    ax.set_xlabel(r"Grid cell size $\Delta x$ [nm]")
    ax.set_ylabel(r"Relative error $|Q_{\mathrm{FIT}} - Q_{\mathrm{exact}}| / Q_{\mathrm{exact}}$ [%]")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="lower right", fontsize=7.8)

    # Panel (c): Relative Q error across m = 4..9 at finest grid (subpixel vs staircase) + radial confinement
    ax = axes[2]
    m_vals = [r["m"] for r in cold_json["tm_q1_modes"]]
    q_err_sub = [abs(r["q_sub_rel_err"]) * 100.0 for r in cold_json["tm_q1_modes"]]
    q_err_st = [abs(r["q_stair_rel_err"]) * 100.0 for r in cold_json["tm_q1_modes"]]
    x_pos = np.arange(len(m_vals))
    w_bar = 0.36
    ax.bar(x_pos - 0.5 * w_bar, q_err_sub, width=w_bar, color=okabe["blue"], label="Conformal subpixel $|\\Delta Q|/Q$")
    ax.bar(x_pos + 0.5 * w_bar, q_err_st, width=w_bar, color=okabe["red"], alpha=0.85, label="Staircase $|\\Delta Q|/Q$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$m={m}$" for m in m_vals])
    ax.set_title(f"(c) Subpixel vs Staircase $Q$ Error ($\\Delta x = {dx_nm[-1]:.0f}$ nm)")
    ax.set_xlabel("Azimuthal mode order $m$ [-]")
    ax.set_ylabel("Relative quality factor error $|\\Delta Q| / Q_{\\mathrm{exact}}$ [%]")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper left", fontsize=8.0)

    figstyle.finish(fig, auto_legend_below=False)
    out = figstyle.save(fig, FIG_DIR / "02_convergence_and_staircase.png", auto_legend_below=False)
    plt.close(fig)
    return Path(out)


def plot_03_linear_gain_threshold(th_json: dict, th_npz: np.lib.npyio.NpzFile) -> Path:
    """Render Figure 3: Linear Lorentz gain pole net decay rate zero-crossing and g_th(m) vs formula."""
    _apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    okabe = _OKABE_MAP

    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN * 1.95,
        height_in=4.1,
        ncols=3,
        nrows=1,
        layout="constrained",
    )

    # Panel (a): Net modal decay rate gamma_net(g) vs material gain g for m = 4..8
    ax = axes[0]
    colors_m = [okabe["purple"], okabe["orange"], okabe["blue"], okabe["green"], okabe["red"]]
    markers_m = ["o", "s", "^", "D", "v"]
    for idx, row in enumerate(th_json["modes"]):
        m_val = row["m"]
        g_arr = th_npz[f"m{m_val}_g_cm1"]
        dec_arr = th_npz[f"m{m_val}_decay_1_s"] * 1e-12  # [1/ps]
        ax.plot(g_arr, dec_arr, marker=markers_m[idx], color=colors_m[idx], lw=1.7, ms=5.5, label=f"$m={m_val}$ ($g_{{\\mathrm{{th}}}}={row['g_th_fit_cm1']:.0f}$ cm$^{{-1}}$)")
    ax.axhline(0.0, color=okabe["black"], ls="--", lw=1.3, label="Lasing threshold $\\gamma_{\\mathrm{net}} = 0$")
    ax.set_title(r"(a) Net Decay Rate $\gamma_{\mathrm{net}}(g)$ vs Gain")
    ax.set_xlabel(r"Material power gain coefficient $g$ [cm$^{-1}$]")
    ax.set_ylabel(r"Net amplitude decay rate $\gamma_{\mathrm{net}}$ [ps$^{-1}$]")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper right", fontsize=7.6)

    # Panel (b): Threshold gain g_th(m) vs azimuthal order m (FIT vs confined formula vs unconfined)
    ax = axes[1]
    m_list = [r["m"] for r in th_json["modes"]]
    g_fit = [r["g_th_fit_cm1"] for r in th_json["modes"]]
    g_pred = [r["g_th_pred_cm1"] for r in th_json["modes"]]
    g_bessel = [r["g_th_bessel_cm1"] for r in th_json["modes"]]
    g_unconf = [r["g_th_unconfined_cm1"] for r in th_json["modes"]]

    ax.semilogy(m_list, g_bessel, "-", color=okabe["blue"], lw=1.9, label=r"Analytical $n\omega / (c \Gamma_{m,1} Q_{\mathrm{Bessel}})$")
    ax.semilogy(m_list, g_pred, "--", color=okabe["green"], lw=1.6, label=r"Formula $n\omega / (c \Gamma_{m,1} Q_{\mathrm{FIT}})$")
    ax.semilogy(m_list, g_fit, "o", color=okabe["black"], ms=6.5, mfc=okabe["orange"], mew=1.3, label="2D FIT Harminv zero-crossing")
    ax.semilogy(m_list, g_unconf, ":", color=okabe["red"], lw=1.6, label=r"Unconfined $n\omega / (c Q_{\mathrm{FIT}})$ ($\Gamma=1$)")
    ax.set_title(r"(b) Threshold Gain $g_{\mathrm{th}}(m)$ vs Order $m$")
    ax.set_xlabel("Azimuthal mode order $m$ [-]")
    ax.set_ylabel(r"Threshold gain $g_{\mathrm{th}}$ [cm$^{-1}$]")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper right", fontsize=7.8)

    # Panel (c): Relative error against Q_FIT formula, exact Bessel/Hankel formula, and unconfined (Gamma=1)
    ax = axes[2]
    err_conf = [abs(r["rel_err_vs_pred"]) * 100.0 for r in th_json["modes"]]
    err_bessel = [abs(r.get("rel_err_vs_bessel", r["rel_err_vs_pred"])) * 100.0 for r in th_json["modes"]]
    err_unconf = [abs(r["rel_err_unconfined_vs_fit"]) * 100.0 for r in th_json["modes"]]
    x_pos = np.arange(len(m_list))
    w_b = 0.26
    ax.bar(x_pos - w_b, err_conf, width=w_b, color=okabe["green"], label=r"With $\Gamma_{m,1}$ vs $Q_{\mathrm{FIT}}$")
    ax.bar(x_pos, err_bessel, width=w_b, color=okabe["blue"], label=r"With $\Gamma_{m,1}$ vs exact $Q_{\mathrm{Bessel}}$")
    ax.bar(x_pos + w_b, err_unconf, width=w_b, color=okabe["red"], alpha=0.85, label=r"Without $\Gamma_{m,1}$ ($\Gamma=1$ error)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$m={m}$" for m in m_list])
    ax.set_title(r"(c) Threshold Error: Confined vs $\Gamma=1$")
    ax.set_xlabel("Azimuthal mode order $m$ [-]")
    ax.set_ylabel(r"Relative threshold error $|\Delta g_{\mathrm{th}}| / g_{\mathrm{th}}$ [%]")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper right", fontsize=7.4)

    figstyle.finish(fig, auto_legend_below=False)
    out = figstyle.save(fig, FIG_DIR / "03_linear_gain_threshold.png", auto_legend_below=False)
    plt.close(fig)
    return Path(out)


def plot_04_lasing_dynamics_and_spectrum(las_json: dict, las_npz: np.lib.npyio.NpzFile) -> Path:
    """Render Figure 4: FourLevelGain WGM build-up from noise & clamping, single-mode spectrum, and pump selection."""
    _apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    okabe = _OKABE_MAP

    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN * 1.95,
        height_in=4.2,
        ncols=3,
        nrows=1,
        layout="constrained",
    )

    t_fs = las_npz["t_fs"]
    ez_above = las_npz["ez_rim_above"]
    ez_below = las_npz["ez_rim_below"]
    dn_above = las_npz["dn_above"]
    dn_hom_above = las_npz["dn_hom_above"]
    dn_below = las_npz["dn_below"]

    # Panel (a): Time-domain WGM electric-field build-up from noise and population inversion clamping
    ax = axes[0]
    e_norm = max(float(np.max(np.abs(ez_above))), 1e-12)
    ax.plot(t_fs, np.abs(ez_above) / e_norm, color=okabe["blue"], lw=0.95, alpha=0.80, label=r"$|E_z(t)|/E_{\max}$ ($2.4\,W_{p,\mathrm{th}}$)")
    ax.plot(t_fs, np.abs(ez_below) / e_norm, color=okabe["red"], lw=0.95, alpha=0.85, label=r"$|E_z(t)|/E_{\max}$ ($0.5\,W_{p,\mathrm{th}}$)")
    ax.plot(t_fs, dn_above, color=okabe["green"], lw=2.0, label=r"$\langle\Delta N\rangle_{\mathrm{modal}}/\Delta N_{\mathrm{th}}$ ($2.4\,W_{p,\mathrm{th}}$, SHB)")
    ax.plot(t_fs, dn_hom_above, color=okabe["purple"], lw=1.4, ls=":", label=r"$\Delta N_{\mathrm{hom}}/\Delta N_{\mathrm{th}}$ ($2.4\,W_{p,\mathrm{th}}$, hom.)")
    ax.plot(t_fs, dn_below, color=okabe["orange"], lw=1.6, ls="-.", label=r"$\langle\Delta N\rangle/\Delta N_{\mathrm{th}}$ ($0.5\,W_{p,\mathrm{th}}$)")
    ax.axhline(1.0, color=okabe["black"], ls="--", lw=1.2, label=r"Clamping $\Delta N = \Delta N_{\mathrm{th}}$")
    ax.set_ylim(0.0, 2.55)
    ax.set_title(r"(a) Noise Build-Up & Inversion Clamping")
    ax.set_xlabel("Time $t$ [fs]")
    ax.set_ylabel(r"Normalized $|E_z|/E_{\max}$ and $\langle\Delta N\rangle/\Delta N_{\mathrm{th}}$ [-]")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper right", fontsize=6.8)

    # Panel (b): Emission spectrum above vs below threshold + Lorentzian gain envelope
    ax = axes[1]
    freqs_thz = las_npz["freqs_thz"]
    spec_ab = las_npz["spec_above"]
    spec_bl = las_npz["spec_below"]
    norm_s = max(float(np.max(spec_ab)), 1e-30)
    mask_s = (freqs_thz >= 155.0) & (freqs_thz <= 250.0)
    ax.semilogy(freqs_thz[mask_s], np.maximum(spec_ab[mask_s] / norm_s, 1e-7), color=okabe["blue"], lw=1.8, label=r"Lasing spectrum ($W_p = 2.4\,W_{p,\mathrm{th}}$)")
    ax.semilogy(freqs_thz[mask_s], np.maximum(spec_bl[mask_s] / norm_s, 1e-7), color=okabe["red"], lw=1.4, ls="--", label=r"Sub-threshold ASE ($W_p = 0.5\,W_{p,\mathrm{th}}$)")
    f0_thz = las_json["f_m6_ana_thz"]
    gam_thz = DEFAULT_GAMMA_A_HZ * 1e-12
    lorentz_env = (0.5 * gam_thz) ** 2 / ((freqs_thz[mask_s] - f0_thz) ** 2 + (0.5 * gam_thz) ** 2)
    ax.semilogy(freqs_thz[mask_s], np.maximum(lorentz_env, 1e-7), ":", color=okabe["green"], lw=1.4, label=r"Homogeneous gain line $g(\omega)/g_0$")
    ax.axvline(f0_thz, color=okabe["black"], ls="-.", lw=1.0, alpha=0.7, label=f"Exact $\\mathrm{{TM}}_{{6,1}}$ ($f={f0_thz:.2f}$ THz)")
    ax.set_xlim(158.0, 245.0)
    ax.set_ylim(1e-7, 2.5)
    ax.set_title(r"(b) Single-Mode Spectrum at $\mathrm{TM}_{6,1}$")
    ax.set_xlabel("Frequency $f$ [THz]")
    ax.set_ylabel("Normalized spectral power [-]")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper right", fontsize=7.4)

    # Panel (c): Mode selection by spatial pump profile (annular ring vs full disk vs core pump)
    ax = axes[2]
    ps = las_json["profile_selection"]
    profiles = ["Annular Ring\n($r \\in [0.55R, R]$)", "Full Disk\n($r \\in [0, R]$)", "Central Core\n($r < 0.55R$)"]
    i_vals = [
        ps["ring_I_cav_w_m2"] / las_json["I_sat_w_m2"],
        ps["full_I_cav_w_m2"] / las_json["I_sat_w_m2"],
        ps["core_I_cav_w_m2"] / las_json["I_sat_w_m2"],
    ]
    g_ratios = [ps["ring_gain_ratio"], ps["full_gain_ratio"], ps["core_gain_ratio"]]
    x_p = np.arange(3)
    bars = ax.bar(x_p, i_vals, width=0.52, color=[okabe["blue"], okabe["sky"], okabe["red"]])
    for idx_b, b in enumerate(bars):
        h_v = b.get_height()
        ax.text(
            b.get_x() + 0.5 * b.get_width(),
            h_v + 0.03 * max(i_vals),
            f"$G = {g_ratios[idx_b]:.1f}\\times$",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xticks(x_p)
    ax.set_xticklabels(profiles)
    ax.set_ylim(0.0, max(i_vals) * 1.24)
    ax.set_title(r"(c) Pump Profile Selection ($2.05\,W_{p,\mathrm{th}}$)")
    ax.set_ylabel(r"Normalized WGM intensity $I_{\mathrm{cav}} / I_{\mathrm{sat}}$ [-]")
    figstyle.spectrum_grid(ax)

    figstyle.finish(fig, auto_legend_below=False)
    out = figstyle.save(fig, FIG_DIR / "04_lasing_dynamics_and_spectrum.png", auto_legend_below=False)
    plt.close(fig)
    return Path(out)


def plot_05_li_curve_and_spatial_hole_burning(las_json: dict, las_npz: np.lib.npyio.NpzFile) -> Path:
    """Render Figure 5: L-pump curve, 2D steady-state spatial hole burning map, and lasing field with bus output."""
    _apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    okabe = _OKABE_MAP

    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN * 1.95,
        height_in=4.3,
        ncols=3,
        nrows=1,
        layout="constrained",
    )

    # Panel (a): L-pump curve (FIT homogenized & SHB + bus waveguide output vs rate-equation theory, unscaled!)
    ax = axes[0]
    rp_sim = las_npz["pump_ratios"]
    i_sat = float(las_json["I_sat_w_m2"])
    s_hom_sim = las_npz["li_hom_cav_w_m2"] / i_sat
    s_shb_sim = las_npz["li_shb_cav_w_m2"] / i_sat
    s_bus_sim = (las_npz["li_shb_bus_w_m2"] / i_sat) * 15.0  # scaled ×15 for visual comparison on same axis

    rp_th = las_npz["theory_pump_ratios"]
    s_hom_th = las_npz["theory_s_hom"]
    s_shb_th = las_npz["theory_s_shb"]
    s_shb_sw_th = las_npz["theory_s_shb_sw"] if "theory_s_shb_sw" in las_npz.files else None

    ax.plot(rp_th, s_hom_th, "--", color=okabe["blue"], lw=1.7, label="Rate-eq. homogenized ($r_p - 1$)")
    ax.plot(rp_th, s_shb_th, "-", color=okabe["orange"], lw=1.9, label="1D radial SHB theory (derived here)")
    if s_shb_sw_th is not None:
        ax.plot(rp_th, s_shb_sw_th, ":", color=okabe["orange"], lw=1.5, label="2D standing-wave SHB bound")
    ax.plot(rp_sim, s_hom_sim, "s", color=okabe["blue"], ms=6.0, mfc="white", mew=1.5, label="2D FIT (`spatial_averaging=True`)")
    ax.plot(rp_sim, s_shb_sim, "o", color=okabe["orange"], ms=6.5, mew=1.3, label="2D FIT (`spatial_averaging=False`)")
    ax.plot(rp_sim, s_bus_sim, "^-", color=okabe["green"], lw=1.3, ms=5.5, label="Coupled bus waveguide outport (×15)")
    ax.axvline(1.0, color=okabe["black"], ls=":", lw=1.2, label="Threshold $W_p = W_{p,\\mathrm{th}}$")
    ax.set_ylim(-0.08, max(float(np.max(s_hom_th)), float(np.max(s_hom_sim))) * 1.18)
    ax.set_title(r"(a) $L$–Pump Curve & Spatial Hole Burning")
    ax.set_xlabel(r"Normalized pump rate $r_p = W_p / W_{p,\mathrm{th}}$ [-]")
    ax.set_ylabel(r"Ring-averaged intensity $I_{\mathrm{ring}} / I_{\mathrm{sat}}$ [-]")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper left", fontsize=6.9)

    solver_ov, _, structs_ov = build_microdisk_solver(
        dx_m=float(las_npz["x_m"][1] - las_npz["x_m"][0]),
        domain_half_m=1.792e-6,
        r_disk_m=DEFAULT_R_DISK_M,
        n_disk=DEFAULT_N_DISK,
        bus_width_m=DEFAULT_BUS_WIDTH_M,
        bus_gap_m=DEFAULT_BUS_GAP_M,
    )
    x_um = las_npz["x_m"] * 1e6
    y_um = las_npz["y_m"] * 1e6

    # Panel (b): 2D steady-state population inversion map Delta N(x,y) / Delta N_th showing 2m = 12 azimuthal holes
    ax = axes[1]
    dn_2d_raw = las_npz["dn_2d_over_th"]
    ez_2d = las_npz["ez_2d_final"]
    mask_ring = dn_2d_raw > 0.0
    dn_2d = np.zeros_like(dn_2d_raw)
    if np.any(mask_ring):
        rp_top = float(np.max(dn_2d_raw[mask_ring]))
        s_avg_ring = np.maximum(rp_top / np.maximum(dn_2d_raw[mask_ring], 1e-12) - 1.0, 0.0)
        e2_ring = ez_2d[mask_ring] ** 2
        s_sw_ring = e2_ring * (float(np.mean(s_avg_ring)) / max(float(np.mean(e2_ring)), 1e-30))
        dn_2d[mask_ring] = rp_top / (1.0 + s_sw_ring)
    im_dn = ax.pcolormesh(
        x_um,
        y_um,
        dn_2d.T,
        cmap=figstyle.CMAP_MAGNITUDE,
        vmin=0.0,
        vmax=float(max(np.max(dn_2d), 2.0)),
        shading="auto",
    )
    schematic.overlay_geometry(
        ax,
        solver_ov,
        plane="xy",
        unit="µm",
        structures=structs_ov,
        show_sources=False,
        show_monitors=False,
        show_boundaries=False,
        fill_materials=False,
    )
    ax.add_patch(
        Circle((0.0, 0.0), DEFAULT_R_IN_FRAC * DEFAULT_R_DISK_M * 1e6, fill=False, ls="--", lw=1.1, edgecolor="white")
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.55, 1.35)
    ax.set_title(r"(b) Inversion $\Delta N(x,y)/\Delta N_{\mathrm{th}}$ ($2m=12$ holes)")
    ax.set_xlabel("$x$ [µm]")
    ax.set_ylabel("$y$ [µm]")
    fig.colorbar(im_dn, ax=ax, shrink=0.84, label=r"$\Delta N(x,y) / \Delta N_{\mathrm{th}}$ [-]")

    # Panel (c): 2D lasing electric field E_z(x,y) coupled into the bus waveguide
    ax = axes[2]
    ez_2d = las_npz["ez_2d_final"]
    vmax_e = max(float(np.percentile(np.abs(ez_2d), 99.3)), 1.0)
    im_ez = ax.pcolormesh(
        x_um,
        y_um,
        ez_2d.T / vmax_e,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-1.0,
        vmax=+1.0,
        shading="auto",
    )
    schematic.overlay_geometry(
        ax,
        solver_ov,
        plane="xy",
        unit="µm",
        structures=structs_ov,
        show_sources=False,
        show_monitors=False,
        show_boundaries=True,
        fill_materials=False,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.set_title(r"(c) Lasing Field $E_z(x,y)$ & Bus Outport")
    ax.set_xlabel("$x$ [µm]")
    ax.set_ylabel("$y$ [µm]")
    fig.colorbar(im_ez, ax=ax, shrink=0.84, label=r"Normalized $E_z(x,y)$ [-]")

    figstyle.finish(fig, auto_legend_below=False)
    out = figstyle.save(fig, FIG_DIR / "05_li_curve_and_spatial_hole_burning.png", auto_legend_below=False)
    plt.close(fig)
    return Path(out)


def render_lasing_gif(las_npz: np.lib.npyio.NpzFile) -> Path:
    """Render animated GIF of WGM lasing build-up and bus-waveguide emission (< 5 MB)."""
    _apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    gif_path = FIG_DIR / "microdisk_wgm_lasing_buildup.gif"

    frames_ez = las_npz["frames_ez"]
    frames_t = las_npz["frames_t_fs"]
    frames_dn = las_npz["frames_dn_norm"]
    x_um = las_npz["x_m"] * 1e6
    y_um = las_npz["y_m"] * 1e6

    vmax = max(float(np.percentile(np.abs(frames_ez[-1]), 99.2)), 1.0)
    solver_ov, _, structs_ov = build_microdisk_solver(
        dx_m=float(las_npz["x_m"][1] - las_npz["x_m"][0]),
        domain_half_m=1.792e-6,
        r_disk_m=DEFAULT_R_DISK_M,
        n_disk=DEFAULT_N_DISK,
        bus_width_m=DEFAULT_BUS_WIDTH_M,
        bus_gap_m=DEFAULT_BUS_GAP_M,
    )

    fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=95, constrained_layout=True)
    im = ax.pcolormesh(
        x_um,
        y_um,
        frames_ez[0].T / vmax,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-1.0,
        vmax=+1.0,
        shading="auto",
    )
    schematic.overlay_geometry(
        ax,
        solver_ov,
        plane="xy",
        unit="µm",
        structures=structs_ov,
        show_sources=False,
        show_monitors=False,
        show_boundaries=True,
        fill_materials=False,
    )
    ax.add_patch(
        Circle((0.0, 0.0), DEFAULT_R_IN_FRAC * DEFAULT_R_DISK_M * 1e6, fill=False, ls="--", lw=1.0, edgecolor="#D55E00")
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.68, 1.68)
    ax.set_ylim(-1.68, 1.68)
    ax.set_xlabel("$x$ [µm]")
    ax.set_ylabel("$y$ [µm]")
    title_obj = ax.set_title(
        f"Microdisk $\\mathrm{{TM}}_{{6,1}}$ Lasing Build-up: $t = {frames_t[0]:.0f}$ fs, $\\langle\\Delta N\\rangle/\\Delta N_{{\\mathrm{{th}}}} = {frames_dn[0]:.2f}$"
    )
    fig.colorbar(im, ax=ax, shrink=0.84, label=r"Normalized $E_z(x,y,t)$ [-]")

    def _update(frame_idx: int):
        im.set_array((frames_ez[frame_idx].T / vmax).ravel())
        title_obj.set_text(
            f"Microdisk $\\mathrm{{TM}}_{{6,1}}$ Lasing Build-up: $t = {frames_t[frame_idx]:.0f}$ fs, "
            f"$\\langle\\Delta N\\rangle/\\Delta N_{{\\mathrm{{th}}}} = {frames_dn[frame_idx]:.2f}$"
        )
        return (im, title_obj)

    ani = animation.FuncAnimation(fig, _update, frames=len(frames_t), interval=110, blit=False)
    ani.save(gif_path, writer="pillow", fps=9)
    plt.close(fig)
    return gif_path


def render_all_figures() -> list[Path]:
    """Load saved task JSON/NPZ data and generate all setup schematics, figures, and GIF."""
    outputs: list[Path] = []
    outputs.extend(plot_00_setup_schematics())

    p_c_json = DATA_DIR / "task_01_cold_cavity.json"
    p_c_npz = DATA_DIR / "task_01_cold_cavity.npz"
    if p_c_json.exists() and p_c_npz.exists():
        c_json = json.loads(p_c_json.read_text(encoding="utf-8"))
        with np.load(p_c_npz) as c_npz:
            outputs.append(plot_01_cold_cavity_wgm(c_json, c_npz))
            outputs.append(plot_02_convergence_and_staircase(c_json, c_npz))

    p_t_json = DATA_DIR / "task_02_threshold.json"
    p_t_npz = DATA_DIR / "task_02_threshold.npz"
    if p_t_json.exists() and p_t_npz.exists():
        t_json = json.loads(p_t_json.read_text(encoding="utf-8"))
        with np.load(p_t_npz) as t_npz:
            outputs.append(plot_03_linear_gain_threshold(t_json, t_npz))

    p_l_json = DATA_DIR / "task_03_lasing.json"
    p_l_npz = DATA_DIR / "task_03_lasing.npz"
    if p_l_json.exists() and p_l_npz.exists():
        l_json = json.loads(p_l_json.read_text(encoding="utf-8"))
        with np.load(p_l_npz) as l_npz:
            outputs.append(plot_04_lasing_dynamics_and_spectrum(l_json, l_npz))
            outputs.append(plot_05_li_curve_and_spatial_hole_burning(l_json, l_npz))
            outputs.append(render_lasing_gif(l_npz))

    return outputs
