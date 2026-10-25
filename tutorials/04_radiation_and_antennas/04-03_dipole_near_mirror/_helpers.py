"""Helper functions and analytical baselines for Tutorial 4.3: Dipole near a mirror and planar microcavity.

Provides:
- Exact 2D and 3D PEC mirror Drexhage Purcell factors and angular radiation patterns.
- Exact 2D Chance-Prock-Silbey (CPS) Sommerfeld integrals for a line dipole above a lossy
  dispersive (Drude silver) mirror, separating radiative and non-radiative (SPP + Ohmic) channels.
- Exact 2D and 3D planar microcavity Purcell factors F_P(d) (comparing 2D 1/sqrt Van Hove
  singularities against 3D step DOS of Abram et al., IEEE JQE 34, 71, 1998).
- 1D Transfer-Matrix Method (TMM) and Fabry-Perot / Benisty (IEEE JQE 34, 1612, 1998) angular
  extraction models for DBR and metal-DBR resonant-cavity LEDs (RCLEDs).
- 2D cylindrical Near2Far conversion from Wakis Near2FarMonitor and SlabXZ subpixel shape primitive
  for (x, z) cross-section simulations.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c as c0
from scipy.constants import e as qe
from scipy.constants import epsilon_0 as eps0
from scipy.constants import hbar
from scipy.constants import mu_0 as mu0
from scipy.integrate import quad
from scipy.special import j0, j1, jn

ETA0 = float(np.sqrt(mu0 / eps0))
EV_TO_RAD_S = float(qe / hbar)


# ---------------------------------------------------------------------------
# 1. Exact 2D and 3D PEC Mirror Drexhage Baselines
# ---------------------------------------------------------------------------
def pec_mirror_purcell_2d(kh: np.ndarray | float, pol: str = "y") -> np.ndarray:
    """Exact 2D Purcell factor F_P(kh) = P'(h) / P'_0 for a line dipole above a PEC mirror.

    Parameters
    ----------
    kh : array_like or float
        Normalized dipole-to-mirror distance k * h = (2 * pi * n / lambda_0) * h
        [dimensionless].
    pol : {"y", "x", "z", "te_par", "tm_par", "tm_perp", "iso"}
        Dipole orientation relative to the mirror plane z = 0:
        - "y" or "te_par": Out-of-plane parallel line dipole J_y (TE / s-polarized).
        - "x" or "tm_par": In-plane parallel line dipole J_x (TM / p-polarized).
        - "z" or "tm_perp": Vertical perpendicular line dipole J_z (TM / p-polarized).
        - "iso": Unpolarized 3-component average (F_x + F_y + F_z) / 3.

    Returns
    -------
    F_P : np.ndarray
        Normalized radiated power per unit length P'(h) / P'_0 [dimensionless].
    """
    u = 2.0 * np.asarray(kh, dtype=float)
    pol_key = pol.lower()

    if pol_key in ("y", "te_par", "te"):
        return 1.0 - j0(u)

    if pol_key in ("x", "tm_par"):
        return 1.0 - j0(u) + jn(2, u)

    if pol_key in ("z", "tm_perp", "perp"):
        return 1.0 + j0(u) + jn(2, u)

    if pol_key == "iso":
        fy = 1.0 - j0(u)
        fx = 1.0 - j0(u) + jn(2, u)
        fz = 1.0 + j0(u) + jn(2, u)
        return (fx + fy + fz) / 3.0

    raise ValueError(f"Unknown polarization {pol!r}")


def pec_mirror_purcell_3d(kh: np.ndarray | float, pol: str = "par") -> np.ndarray:
    """Exact 3D Purcell factor F_P(kh) for a point dipole above a PEC mirror.

    Parameters
    ----------
    kh : array_like or float
        Normalized dipole-to-mirror distance k * h [dimensionless].
    pol : {"par", "perp", "iso"}
        Dipole orientation ("par" parallel to mirror, "perp" perpendicular, "iso" 3D average).

    Returns
    -------
    F_P : np.ndarray
        Normalized 3D radiated power P(h) / P_0 [dimensionless].
    """
    u = 2.0 * np.asarray(kh, dtype=float)
    u_safe = np.where(np.abs(u) < 1e-8, 1e-8, u)
    pol_key = pol.lower()

    f_par = np.where(
        np.abs(u) < 1e-8,
        0.0,
        1.0
        - 1.5 * (np.sin(u_safe) / u_safe + np.cos(u_safe) / u_safe**2 - np.sin(u_safe) / u_safe**3),
    )
    f_perp = np.where(
        np.abs(u) < 1e-8,
        2.0,
        1.0 - 3.0 * (np.cos(u_safe) / u_safe**2 - np.sin(u_safe) / u_safe**3),
    )

    if pol_key in ("par", "x", "y"):
        return f_par
    if pol_key in ("perp", "z"):
        return f_perp
    if pol_key == "iso":
        return (2.0 * f_par + f_perp) / 3.0
    raise ValueError(f"Unknown 3D polarization {pol!r}")


def pec_mirror_pattern_2d(
    theta: np.ndarray | float, kh: float, pol: str = "y"
) -> np.ndarray:
    """Exact 2D angular power density dP'(theta)/dtheta / P'_0 above a PEC mirror.

    The polar angle ``theta`` in [-pi/2, pi/2] is measured from the +z surface normal
    (theta = 0 is normal emission along +z; theta = +/-pi/2 is grazing along +/-x).
    Integrating ``pec_mirror_pattern_2d(theta, kh, pol)`` over [-pi/2, pi/2] yields
    ``pec_mirror_purcell_2d(kh, pol)``.

    Parameters
    ----------
    theta : array_like
        Polar angle from +z normal [rad].
    kh : float
        Normalized dipole height k * h [dimensionless].
    pol : {"y", "x", "z"}
        Dipole orientation.

    Returns
    -------
    pattern : np.ndarray
        Normalized angular power density (1 / P'_0) dP'/dtheta [1/rad].
    """
    th = np.asarray(theta, dtype=float)
    mask = np.abs(th) <= (0.5 * np.pi + 1e-12)
    ct = np.cos(th)
    st = np.sin(th)
    pol_key = pol.lower()

    if pol_key in ("y", "te_par", "te"):
        val = (2.0 / np.pi) * np.sin(kh * ct) ** 2
    elif pol_key in ("x", "tm_par"):
        val = (4.0 / np.pi) * (ct**2) * (np.sin(kh * ct) ** 2)
    elif pol_key in ("z", "tm_perp", "perp"):
        val = (4.0 / np.pi) * (st**2) * (np.cos(kh * ct) ** 2)
    else:
        raise ValueError(f"Unknown polarization {pol!r}")

    return np.where(mask, val, 0.0)


# ---------------------------------------------------------------------------
# 2. Lossy Dispersive Metal (Drude Ag) & 2D CPS Sommerfeld Integrals
# ---------------------------------------------------------------------------
def drude_silver_permittivity(
    wavelength_m: np.ndarray | float,
    eps_inf: float = 3.7,
    wp_ev: float = 9.01,
    gamma_ev: float = 0.048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drude dielectric function, complex index, and E-field skin depth of silver.

    Uses the physics time-harmonic convention exp(-i * omega * t) where Im(eps) > 0:
        eps(omega) = eps_inf - wp^2 / (omega * (omega + i * gamma)).

    Parameters
    ----------
    wavelength_m : array_like or float
        Free-space wavelength lambda_0 [m].
    eps_inf : float
        High-frequency background permittivity [dimensionless].
    wp_ev : float
        Plasma energy hbar * omega_p [eV].
    gamma_ev : float
        Collision damping energy hbar * gamma [eV].

    Returns
    -------
    eps_m : np.ndarray
        Complex relative permittivity [dimensionless].
    n_complex : np.ndarray
        Complex refractive index n + i * kappa [dimensionless].
    skin_depth_e : np.ndarray
        Electric-field 1/e penetration depth delta_E = c / (omega * kappa) [m].
    """
    lam = np.asarray(wavelength_m, dtype=float)
    omega = 2.0 * np.pi * c0 / lam
    wp = wp_ev * EV_TO_RAD_S
    gamma = gamma_ev * EV_TO_RAD_S
    eps_m = eps_inf - (wp**2) / (omega * (omega + 1j * gamma))
    n_complex = np.sqrt(eps_m)
    k0 = omega / c0
    skin_depth_e = 1.0 / (k0 * np.imag(n_complex))
    return eps_m, n_complex, skin_depth_e


def _fresnel_2d(
    u: np.ndarray | float, eps_m: complex, eps_1: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fresnel reflection coefficients r_s, r_p and normalized vertical wavenumber kz1."""
    u_arr = np.asarray(u, dtype=float)
    kz1 = np.sqrt(1.0 - u_arr**2 + 0j)
    kz2 = np.sqrt(eps_m / eps_1 - u_arr**2 + 0j)
    kz1 = np.where(np.imag(kz1) < 0, -kz1, kz1)
    kz2 = np.where(np.imag(kz2) < 0, -kz2, kz2)
    r_s = (kz1 - kz2) / (kz1 + kz2)
    r_p = (eps_m * kz1 - eps_1 * kz2) / (eps_m * kz1 + eps_1 * kz2)
    return r_s, r_p, kz1


def cps_sommerfeld_integrand_2d(
    u: np.ndarray | float,
    kh: float,
    eps_m: complex,
    eps_1: float = 1.0,
    pol: str = "y",
) -> np.ndarray:
    """Wavenumber-resolved power dissipation density dF_tot/du for a 2D line dipole.

    For 0 <= u < 1 (radiative cone, u = k_parallel / k_1), the integrand represents the
    plane-wave power spectrum. For u > 1 (evanescent wavevectors outside the light cone),
    the integrand is non-zero only when Im(eps_m) > 0, exhibiting a sharp surface-plasmon-
    polariton (SPP) pole at u_SPP = Re(sqrt(eps_m * eps_1 / (eps_m + eps_1))) for TM
    orientations ("x", "z") and a short-range Ohmic lossy-wave tail.

    Parameters
    ----------
    u : array_like
        Normalized in-plane wavevector u = k_x / k_1 [dimensionless].
    kh : float
        Normalized height k_1 * h [dimensionless].
    eps_m : complex
        Metal relative permittivity (Im(eps_m) > 0).
    eps_1 : float
        Upper half-space dielectric permittivity [dimensionless].
    pol : {"y", "x", "z"}
        Dipole orientation.

    Returns
    -------
    dF_du : np.ndarray
        Spectral power dissipation density [dimensionless].
    """
    u_arr = np.asarray(u, dtype=float)
    r_s, r_p, kz1 = _fresnel_2d(u_arr, eps_m, eps_1)
    phase = np.exp(2j * kz1 * kh)
    pol_key = pol.lower()

    if pol_key in ("y", "te_par", "te"):
        return (2.0 / np.pi) * np.real((1.0 + r_s * phase) / kz1)
    if pol_key in ("x", "tm_par"):
        return (4.0 / np.pi) * np.real(kz1 * (1.0 - r_p * phase))
    if pol_key in ("z", "tm_perp", "perp"):
        return (4.0 / np.pi) * np.real((u_arr**2 / kz1) * (1.0 + r_p * phase))
    raise ValueError(f"Unknown polarization {pol!r}")


def cps_sommerfeld_2d(
    kh_array: np.ndarray | float,
    eps_m: complex,
    eps_1: float = 1.0,
    pol: str = "y",
    u_max: float = 40.0,
) -> dict[str, np.ndarray]:
    """2D Chance-Prock-Silbey (CPS) Sommerfeld calculation of F_tot, F_rad, and F_nonrad.

    Parameters
    ----------
    kh_array : array_like or float
        Normalized dipole heights k_1 * h [dimensionless].
    eps_m : complex
        Complex permittivity of the mirror (Im(eps_m) > 0).
    eps_1 : float
        Permittivity of the upper medium [dimensionless].
    pol : {"y", "x", "z"}
        Dipole orientation.
    u_max : float
        Upper integration cutoff in normalized wavenumber u = k_x / k_1.

    Returns
    -------
    result : dict of np.ndarray
        Dictionary with keys ``"F_tot"``, ``"F_rad"``, ``"F_nonrad"``, and ``"eta_rad"``
        (radiative quantum efficiency F_rad / F_tot).
    """
    kh_vals = np.atleast_1d(np.asarray(kh_array, dtype=float))
    f_tot = np.zeros_like(kh_vals)
    f_rad = np.zeros_like(kh_vals)
    pol_key = pol.lower()

    # Locate the SPP pole for adaptive quadrature splitting in TM polarizations
    u_spp = float(np.real(np.sqrt(eps_m * eps_1 / (eps_m + eps_1))))
    has_spp = (u_spp > 1.0001) and (pol_key not in ("y", "te_par", "te"))

    for idx, kh in enumerate(kh_vals):
        # Change of variables u = sin(theta) on [0, 1] removes 1/sqrt(1 - u^2) singularity
        def _prop_tot(theta: float) -> float:
            u = np.sin(theta)
            ct = np.cos(theta)
            r_s, r_p, kz1 = _fresnel_2d(u, eps_m, eps_1)
            phase = np.exp(2j * kz1 * kh)
            if pol_key in ("y", "te_par", "te"):
                return float((2.0 / np.pi) * np.real(1.0 + r_s * phase))
            if pol_key in ("x", "tm_par"):
                return float((4.0 / np.pi) * (ct**2) * np.real(1.0 - r_p * phase))
            return float((4.0 / np.pi) * (u**2) * np.real(1.0 + r_p * phase))

        def _prop_rad(theta: float) -> float:
            u = np.sin(theta)
            ct = np.cos(theta)
            r_s, r_p, kz1 = _fresnel_2d(u, eps_m, eps_1)
            phase = np.exp(2j * kz1 * kh)
            if pol_key in ("y", "te_par", "te"):
                val = 0.5 * (1.0 + np.abs(r_s) ** 2 + 2.0 * np.real(r_s * phase))
                return float((2.0 / np.pi) * val)
            if pol_key in ("x", "tm_par"):
                val = 0.5 * (1.0 + np.abs(r_p) ** 2 - 2.0 * np.real(r_p * phase))
                return float((4.0 / np.pi) * (ct**2) * val)
            val = 0.5 * (1.0 + np.abs(r_p) ** 2 + 2.0 * np.real(r_p * phase))
            return float((4.0 / np.pi) * (u**2) * val)

        rad_part_tot, _ = quad(_prop_tot, 0.0, 0.5 * np.pi, limit=200)
        rad_part_up, _ = quad(_prop_rad, 0.0, 0.5 * np.pi, limit=200)

        # Evanescent integral u in (1, u_max): u = cosh(t), kz1 = i * sinh(t), du / |kz1| = dt
        def _evan_t(t: float) -> float:
            u = np.cosh(t)
            kappa = np.sinh(t)
            r_s, r_p, _ = _fresnel_2d(u, eps_m, eps_1)
            decay = np.exp(-2.0 * kappa * kh)
            if pol_key in ("y", "te_par", "te"):
                return float((2.0 / np.pi) * np.imag(r_s) * decay)
            if pol_key in ("x", "tm_par"):
                return float((4.0 / np.pi) * (kappa**2) * np.imag(r_p) * decay)
            return float((4.0 / np.pi) * (u**2) * np.imag(r_p) * decay)

        t_max = float(np.arccosh(max(u_max, 2.0)))
        if has_spp and u_spp < u_max:
            t_spp = float(np.arccosh(u_spp))
            pts = [max(0.0, t_spp - 0.08), t_spp, min(t_max, t_spp + 0.08)]
            evan_part, _ = quad(_evan_t, 0.0, t_max, points=pts, limit=300)
        else:
            evan_part, _ = quad(_evan_t, 0.0, t_max, limit=250)

        f_tot[idx] = rad_part_tot + evan_part
        f_rad[idx] = rad_part_up

    f_nonrad = np.maximum(0.0, f_tot - f_rad)
    eta_rad = np.where(f_tot > 0, f_rad / f_tot, 0.0)
    return {
        "F_tot": f_tot,
        "F_rad": f_rad,
        "F_nonrad": f_nonrad,
        "eta_rad": eta_rad,
    }


# ---------------------------------------------------------------------------
# 3. Planar Microcavity Purcell Factor F_P(d) in 2D vs. 3D
# ---------------------------------------------------------------------------
def planar_cavity_purcell_2d(
    d: np.ndarray | float, pol: str = "y", z_rel: float = 0.5
) -> np.ndarray:
    """Exact 2D Purcell factor F_P(d) for a line dipole inside a planar PEC microcavity.

    Derived via Poisson summation of the 2D image-dipole series (equivalently by summing
    the 1D k_x waveguide normal modes of a parallel-plate resonator of thickness L):

    - At each guided mode cutoff ``d_m = m / 2`` (or ``m - 1/2`` for symmetric modes at
      the cavity midplane ``z_rel = 0.5``), the 1D density of states exhibits an inverse-
      square-root Van Hove singularity ``1 / sqrt(d^2 - d_m^2)`` for TE_par ("y") and
      TM_par ("x"), whereas TM_perp ("z") turns on continuously as ``sqrt(1 - (m/d)^2)``
      on top of the Cut-off-free TEM_0 background ``2 / (pi * d)``.

    Parameters
    ----------
    d : array_like or float
        Normalized optical cavity thickness d = n * L / lambda_0 [dimensionless].
    pol : {"y", "x", "z", "te_par", "tm_par", "tm_perp"}
        Dipole orientation.
    z_rel : float
        Relative dipole position z_0 / L in (0, 1). Default 0.5 (cavity midplane).

    Returns
    -------
    F_P : np.ndarray
        2D cavity Purcell factor F_P(d) [dimensionless].
    """
    d_arr = np.atleast_1d(np.asarray(d, dtype=float))
    out = np.zeros_like(d_arr)
    pol_key = pol.lower()

    for idx, dv in enumerate(d_arr):
        if dv <= 0:
            out[idx] = 0.0
            continue
        m_max = int(np.floor(2.0 * dv - 1e-12))
        if pol_key in ("y", "te_par", "te"):
            s = 0.0
            for m in range(1, m_max + 1):
                w = np.sin(m * np.pi * z_rel) ** 2
                if w > 1e-15:
                    s += w / np.sqrt(1.0 - (m / (2.0 * dv)) ** 2)
            out[idx] = (2.0 / (np.pi * dv)) * s
        elif pol_key in ("x", "tm_par"):
            s = 0.0
            for m in range(1, m_max + 1):
                w = np.sin(m * np.pi * z_rel) ** 2
                if w > 1e-15:
                    ratio = m / (2.0 * dv)
                    s += w * (ratio**2) / np.sqrt(1.0 - ratio**2)
            out[idx] = (4.0 / (np.pi * dv)) * s
        elif pol_key in ("z", "tm_perp", "perp"):
            s = 0.5  # m = 0 TEM mode weight
            for m in range(1, m_max + 1):
                w = np.cos(m * np.pi * z_rel) ** 2
                if w > 1e-15:
                    ratio = m / (2.0 * dv)
                    s += w * np.sqrt(1.0 - ratio**2)
            out[idx] = (4.0 / (np.pi * dv)) * s
        else:
            raise ValueError(f"Unknown polarization {pol!r}")

    return out[0] if np.ndim(d) == 0 else out


def planar_cavity_purcell_3d(d: np.ndarray | float, pol: str = "par") -> np.ndarray:
    """Exact 3D Purcell factor F_P(d) at the midplane of a planar PEC microcavity.

    Implements Eq. (7) of Abram, Robert & Kuszelewicz, IEEE J. Quantum Electron.
    34(1), 71 (1998), doi:10.1109/3.655009. Unlike the 2D 1/sqrt Van Hove divergence,
    the 2D in-plane k_parallel integration in 3D yields finite step discontinuities at
    half-integer thicknesses d = m - 1/2.

    Parameters
    ----------
    d : array_like or float
        Normalized optical cavity thickness d = n * L / lambda_0 [dimensionless].
    pol : {"par", "perp", "iso"}
        3D dipole orientation relative to the cavity mirrors.

    Returns
    -------
    F_P : np.ndarray
        3D cavity Purcell factor F_P(d) [dimensionless].
    """
    d_arr = np.atleast_1d(np.asarray(d, dtype=float))
    f_par = np.zeros_like(d_arr)
    f_perp = np.zeros_like(d_arr)

    for idx, dv in enumerate(d_arr):
        if dv <= 0:
            continue
        m_par = int(np.floor(dv + 0.5 - 1e-12))
        s_par = 0.0
        for m in range(1, m_par + 1):
            s_par += 1.0 + ((m - 0.5) / dv) ** 2
        f_par[idx] = (0.75 / dv) * s_par

        m_perp = int(np.floor(dv - 1e-12))
        s_perp = 0.0
        for m in range(1, m_perp + 1):
            s_perp += 1.0 - (m / dv) ** 2
        f_perp[idx] = (0.5 / dv) + (1.5 / dv) * s_perp

    pol_key = pol.lower()
    if pol_key in ("par", "x", "y"):
        res = f_par
    elif pol_key in ("perp", "z"):
        res = f_perp
    elif pol_key == "iso":
        res = (2.0 * f_par + f_perp) / 3.0
    else:
        raise ValueError(f"Unknown 3D polarization {pol!r}")

    return res[0] if np.ndim(d) == 0 else res


# ---------------------------------------------------------------------------
# 4. 1D Transfer Matrix Method (TMM) & Fabry-Perot RCLED Model
# ---------------------------------------------------------------------------
def tmm_stack_reflectance(
    wavelengths: np.ndarray | float,
    layers: list[tuple[float, complex | float]],
    n_inc: float = 2.39,
    n_sub: complex | float = 1.0,
    theta_inc: float = 0.0,
    pol: str = "te",
) -> dict[str, np.ndarray]:
    """1D Transfer-Matrix Method (TMM) reflection and transmission of a multilayer stack.

    Parameters
    ----------
    wavelengths : array_like or float
        Free-space wavelengths [m].
    layers : list of (thickness_m, refractive_index)
        Ordered list of planar layers from the incident side to the substrate side.
    n_inc : float
        Refractive index of the semi-infinite incident medium [dimensionless].
    n_sub : complex or float
        Refractive index of the semi-infinite exit/substrate medium [dimensionless].
    theta_inc : float
        Incident angle in medium ``n_inc`` [rad].
    pol : {"te", "tm"}
        Polarization ("te" / s-polarized or "tm" / p-polarized).

    Returns
    -------
    res : dict of np.ndarray
        Dictionary with keys ``"r"``, ``"t"``, ``"R"``, ``"T"``.
    """
    lams = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    k0 = 2.0 * np.pi / lams
    sin_inc = np.sin(theta_inc)
    pol_key = pol.lower()

    n_list = [complex(n_inc)] + [complex(n) for _, n in layers] + [complex(n_sub)]
    d_list = [0.0] + [float(d) for d, _ in layers] + [0.0]

    cos_list = []
    q_list = []
    for n_val in n_list:
        ct = np.sqrt(1.0 - (n_inc * sin_inc / n_val) ** 2 + 0j)
        if np.imag(n_val * ct) < 0:
            ct = -ct
        cos_list.append(ct)
        if pol_key == "te":
            q_list.append(n_val * ct)
        elif pol_key == "tm":
            q_list.append(ct / n_val)
        else:
            raise ValueError(f"Unknown polarization {pol!r}")

    m00 = np.ones_like(lams, dtype=complex)
    m01 = np.zeros_like(lams, dtype=complex)
    m10 = np.zeros_like(lams, dtype=complex)
    m11 = np.ones_like(lams, dtype=complex)

    for idx in range(len(n_list) - 1):
        q_a = q_list[idx]
        q_b = q_list[idx + 1]
        r_ab = (q_a - q_b) / (q_a + q_b)
        t_ab = 2.0 * q_a / (q_a + q_b)

        # Interface matrix I_ab = (1 / t_ab) * [[1, r_ab], [r_ab, 1]]
        n00 = (m00 + m01 * r_ab) / t_ab
        n01 = (m00 * r_ab + m01) / t_ab
        n10 = (m10 + m11 * r_ab) / t_ab
        n11 = (m10 * r_ab + m11) / t_ab
        m00, m01, m10, m11 = n00, n01, n10, n11

        # Propagation matrix in layer idx + 1 (if not the substrate)
        if idx + 1 < len(n_list) - 1:
            phi = k0 * n_list[idx + 1] * cos_list[idx + 1] * d_list[idx + 1]
            ep = np.exp(-1j * phi)
            em = np.exp(1j * phi)
            m00 = m00 * ep
            m01 = m01 * em
            m10 = m10 * ep
            m11 = m11 * em

    r = m10 / m00
    t = 1.0 / m00
    R = np.abs(r) ** 2
    T = (np.real(q_list[-1]) / np.real(q_list[0])) * (np.abs(t) ** 2)
    return {"r": r, "t": t, "R": R, "T": T}


def rcled_fabry_perot_pattern_2d(
    theta_ext: np.ndarray,
    wavelength: float,
    cavity_length: float,
    z_dipole: float,
    n_cav: float,
    top_layers: list[tuple[float, complex | float]],
    bot_layers: list[tuple[float, complex | float]],
    n_ext: float = 1.0,
    n_sub: float = 2.39,
    pol: str = "te",
) -> np.ndarray:
    """Exact 2D planar microcavity angular emission pattern I_ext(theta_ext) via TMM.

    Implements the planar microcavity extraction formula of Benisty, De Neve & Weisbuch,
    IEEE J. Quantum Electron. 34(9), 1612 (1998), using exact angle-dependent TMM mirror
    amplitudes ``r_top(theta)``, ``t_top(theta)``, and ``r_bot(theta)`` from the top and
    bottom DBR stacks.

    Parameters
    ----------
    theta_ext : np.ndarray
        External polar angle in medium ``n_ext`` (air) [rad], in (-pi/2, pi/2).
    wavelength : float
        Free-space wavelength lambda_0 [m].
    cavity_length : float
        Physical cavity thickness L_cav [m].
    z_dipole : float
        Dipole height above the bottom mirror interface in (0, L_cav) [m].
    n_cav : float
        Refractive index of the cavity spacer [dimensionless].
    top_layers : list of (thickness_m, n)
        Top mirror layers ordered from cavity toward air.
    bot_layers : list of (thickness_m, n)
        Bottom mirror layers ordered from cavity toward substrate.
    n_ext : float
        External medium refractive index (default 1.0 for air).
    n_sub : float
        Bottom substrate refractive index (default 2.39 for GaN).
    pol : {"te", "tm"}
        Polarization ("te" for J_y).

    Returns
    -------
    pattern : np.ndarray
        Angular power density into the top medium (air) [arbitrary units].
    """
    th_ext = np.asarray(theta_ext, dtype=float)
    out = np.zeros_like(th_ext)
    k0 = 2.0 * np.pi / wavelength

    for i, th in enumerate(th_ext):
        if np.abs(th) >= 0.5 * np.pi - 1e-6:
            continue
        sin_cav = (n_ext / n_cav) * np.sin(th)
        if np.abs(sin_cav) >= 1.0:
            continue
        th_cav = float(np.arcsin(sin_cav))
        kz_cav = k0 * n_cav * np.cos(th_cav)

        top = tmm_stack_reflectance(
            wavelength, top_layers, n_inc=n_cav, n_sub=n_ext, theta_inc=th_cav, pol=pol
        )
        bot = tmm_stack_reflectance(
            wavelength, bot_layers, n_inc=n_cav, n_sub=n_sub, theta_inc=th_cav, pol=pol
        )
        r_top = top["r"][0]
        T_top = top["T"][0]
        r_bot = bot["r"][0]

        phi_bot = 2.0 * kz_cav * z_dipole
        phi_rt = 2.0 * kz_cav * cavity_length

        source_interf = np.abs(1.0 + r_bot * np.exp(1j * phi_bot)) ** 2
        airy_denom = np.abs(1.0 - r_top * r_bot * np.exp(1j * phi_rt)) ** 2
        # In 2D, dk_x = k0 * n_ext * cos(th_ext) dth_ext = k0 * n_cav * cos(th_cav) dth_cav
        jacobian = (n_ext * np.cos(th)) / (n_cav * np.cos(th_cav))
        out[i] = T_top * (source_interf / airy_denom) * jacobian

    return out


# ---------------------------------------------------------------------------
# 5. Wakis 2D Far-Field and Subpixel Geometry Helpers
# ---------------------------------------------------------------------------
def farfield_2d_from_monitor(
    n2f_monitor,
    theta: np.ndarray,
    ly: float,
    freq_idx: int = 0,
) -> np.ndarray:
    """Compute exact 2D angular radiation intensity dP'(theta)/dtheta from a Near2FarMonitor.

    In a 2D periodic Wakis simulation (period ``ly`` along y, cross-section in the x-z plane),
    ``Near2FarMonitor`` evaluates the 3D Stratton-Chu surface integral over the monitor faces
    (including PEC image currents when ``ground_plane='-z'`` is specified). In the x-z plane
    (phi = 0 for x >= 0, phi = pi for x < 0), the y-integrals scale linearly with ``ly``, so
    the 3D radiation intensity ``U = Near2FarMonitor.radiation_intensity(theta, phi)`` converts
    exactly to the 2D cylindrical power per unit length per radian via:
        dP'(theta)/dtheta = (2 * pi / (k * ly^2)) * U(theta, phi).

    Parameters
    ----------
    n2f_monitor : wakis.near2far.Near2FarMonitor
        Populated ``Near2FarMonitor`` after a Wakis simulation run.
    theta : np.ndarray
        Polar angles in [-pi/2, pi/2] measured from +z toward +x [rad].
    ly : float
        Span of the ``Near2FarMonitor`` along the periodic y axis [m].
    freq_idx : int
        Frequency index in ``n2f_monitor.frequencies``.

    Returns
    -------
    dP_dtheta : np.ndarray
        2D angular power density dP'(theta)/dtheta [W s^2 / m / rad], on the same normalization
        scale as ``n2f_monitor.get_flux() / ly``.
    """
    th = np.asarray(theta, dtype=float)
    abs_th = np.abs(th)
    phi = np.where(th >= 0.0, 0.0, np.pi)
    k = float(np.atleast_1d(n2f_monitor.wavenumber)[freq_idx])
    scale_2d = 2.0 * np.pi / (k * (ly**2))
    U = np.asarray(n2f_monitor.radiation_intensity(abs_th, phi)[freq_idx], dtype=float)
    return scale_2d * U


class SlabXZ:
    """Axis-aligned 2D slab in the (x, z) plane with optional grid-unsnapped interfaces.

    ``wakis.templates.photonics.PolySlab`` is hardcoded to polygons in the (x, y) plane
    extruded along z, while ``wakis.templates.shapes.Box`` always returns its boundary
    planes from ``faces(axis)``, causing ``Simulation`` auto-meshing to snap grid lines to
    the box faces. ``SlabXZ`` provides a clean 2D (x, z) slab primitive with exact
    subpixel volume-fraction ``fill()`` and outward ``normal()`` methods so that
    ``assign_materials(..., averaging="kottke" | "staircase")`` can be benchmarked on
    off-grid planar interfaces.

    Parameters
    ----------
    xmin, xmax : float
        Horizontal extent along x [m].
    zmin, zmax : float
        Vertical extent along z [m].
    ymin, ymax : float
        Extent along the periodic y axis [m].
    snap : bool
        If False (default), ``faces(axis)`` returns ``[]`` so the grid does not snap to
        the slab interfaces, exercising subpixel smoothing.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        zmin: float,
        zmax: float,
        material: float | str = 1.0,
        ymin: float = -1.0,
        ymax: float = 1.0,
        snap: bool = False,
        name: str | None = None,
    ):
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.ymin, self.ymax = float(ymin), float(ymax)
        self.zmin, self.zmax = float(zmin), float(zmax)
        self.material = (
            float(material)
            if isinstance(material, (int, float, np.number))
            else material
        )
        self.snap = bool(snap)
        self.name = name

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) 3D corner arrays [m]."""
        return (
            np.array([self.xmin, self.ymin, self.zmin], dtype=float),
            np.array([self.xmax, self.ymax, self.zmax], dtype=float),
        )

    def faces(self, axis: str) -> list[float]:
        """Return grid-snapping planes along ``axis`` (empty when ``snap=False``)."""
        if not self.snap:
            return []
        return {
            "x": [self.xmin, self.xmax],
            "y": [self.ymin, self.ymax],
            "z": [self.zmin, self.zmax],
        }[axis]

    def contains(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Boolean indicator for points inside the slab."""
        return (
            (x >= self.xmin)
            & (x <= self.xmax)
            & (y >= self.ymin)
            & (y <= self.ymax)
            & (z >= self.zmin)
            & (z <= self.zmax)
        )

    def fill(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        hx: np.ndarray,
        hy: np.ndarray,
        hz: np.ndarray,
        n_sub: int = 16,
    ) -> np.ndarray:
        """Exact analytical volume-overlap fraction of each dual cell with the slab.

        Handles both uniform and non-uniform (``graded_axis``) grids: when samples lie on
        primary nodes ``nodes[:-1]``, the FIT dual cell around node ``k`` is the asymmetric
        interval ``[nodes[k] - 0.5 * L[k-1], nodes[k] + 0.5 * L[k]]`` of total width
        ``tL[k] = 0.5 * (L[k-1] + L[k])``.
        """
        del n_sub

        def _exact_bounds_1d(c_1d: np.ndarray, h_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if c_1d.size <= 2 or np.all(h_1d <= 0):
                return c_1d - 0.5 * h_1d, c_1d + 0.5 * h_1d
            dc = np.diff(c_1d)
            if np.allclose(h_1d[1:-1], 0.5 * (dc[:-1] + dc[1:]), rtol=1e-6, atol=1e-15):
                last_l = max(0.0, 2.0 * h_1d[-1] - dc[-1])
                l_this = np.append(dc, last_l)
                l_prev = np.append(max(0.0, 2.0 * h_1d[0] - dc[0]), dc)
                return c_1d - 0.5 * l_prev, c_1d + 0.5 * l_this
            return c_1d - 0.5 * h_1d, c_1d + 0.5 * h_1d

        def _overlap(
            c: np.ndarray, h: np.ndarray, lo: float, hi: float, axis: int
        ) -> np.ndarray:
            if c.ndim == 3:
                c_1d = np.moveaxis(c, axis, -1)[0, 0, :]
                h_1d = np.moveaxis(h, axis, -1)[0, 0, :]
                cell_lo, cell_hi = _exact_bounds_1d(c_1d, h_1d)
                w_cell = cell_hi - cell_lo
                w_safe = np.where(w_cell > 0, w_cell, 1.0)
                a = np.maximum(cell_lo, lo)
                b = np.minimum(cell_hi, hi)
                ov_1d = np.where(
                    w_cell > 0,
                    np.clip((b - a) / w_safe, 0.0, 1.0),
                    np.where((c_1d >= lo) & (c_1d <= hi), 1.0, 0.0),
                )
                shape = [1, 1, 1]
                shape[axis] = -1
                return np.broadcast_to(ov_1d.reshape(shape), c.shape)
            h_safe = np.where(h > 0, h, 1.0)
            a = np.maximum(c - 0.5 * h_safe, lo)
            b = np.minimum(c + 0.5 * h_safe, hi)
            return np.where(
                h > 0,
                np.clip((b - a) / h_safe, 0.0, 1.0),
                np.where((c >= lo) & (c <= hi), 1.0, 0.0),
            )

        return (
            _overlap(X, hx, self.xmin, self.xmax, 0)
            * _overlap(Y, hy, self.ymin, self.ymax, 1)
            * _overlap(Z, hz, self.zmin, self.zmax, 2)
        )

    def normal(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Unit normal vector array of shape (..., 3) at the closest slab face."""
        dx = np.minimum(np.abs(x - self.xmin), np.abs(x - self.xmax))
        dz = np.minimum(np.abs(z - self.zmin), np.abs(z - self.zmax))
        nx = np.where(
            dx < dz,
            np.where(np.abs(x - self.xmin) < np.abs(x - self.xmax), -1.0, 1.0),
            0.0,
        )
        ny = np.zeros_like(x, dtype=float)
        nz = np.where(
            dz <= dx,
            np.where(np.abs(z - self.zmin) < np.abs(z - self.zmax), -1.0, 1.0),
            0.0,
        )
        return np.stack([nx, ny, nz], axis=-1)


def count_cells_in_interval(nodes: np.ndarray, a: float, b: float) -> float:
    """Measure the number of 1D mesh cells in ``nodes`` spanning ``[a, b]``.

    Parameters
    ----------
    nodes : np.ndarray
        Strictly increasing 1D mesh node coordinates [m].
    a, b : float
        Interval endpoints [m].

    Returns
    -------
    n_cells : float
        Number of cells spanning ``[min(a, b), max(a, b)]``, counting fractional cells
        proportionally to their overlap with the interval.
    """
    pts = np.asarray(nodes, dtype=float)
    lo, hi = min(float(a), float(b)), max(float(a), float(b))
    c_lo = pts[:-1]
    c_hi = pts[1:]
    dz = c_hi - c_lo
    overlap = np.clip(np.minimum(c_hi, hi) - np.maximum(c_lo, lo), 0.0, dz)
    return float(np.sum(overlap / np.maximum(dz, 1e-30)))


def set_monitor_bounds(mon, label: str | None = None):
    """Populate ``mon.bounds`` from ``mon.center`` and ``mon.size`` prior to ``emsolve``.

    ``wakis.schematic.extract_setup_info`` uses ``mon.bounds`` when present, falling
    back to the full domain extent for uninitialized box monitors. Setting ``bounds``
    explicitly ensures that ``plot_schematic`` and ``overlay_geometry`` draw the exact
    monitor aperture or Huygens box without running a time-stepping loop.

    Parameters
    ----------
    mon : Near2FarMonitor or FluxMonitor
        Monitor object with ``center`` and ``size`` attributes [m].
    label : str, optional
        Human-readable label stored in ``mon.name``.

    Returns
    -------
    mon : Near2FarMonitor or FluxMonitor
        The same monitor with ``bounds`` (and optional ``name``) populated.
    """
    cx, cy, cz = (float(v) for v in mon.center)
    sx, sy, sz = (float(v) for v in mon.size)
    mon.bounds = (
        cx - 0.5 * sx,
        cx + 0.5 * sx,
        cy - 0.5 * sy,
        cy + 0.5 * sy,
        cz - 0.5 * sz,
        cz + 0.5 * sz,
    )
    if label is not None:
        mon.name = label
    return mon


def overlay_xz_boundaries_and_boxes(
    ax,
    solver,
    unit: str = "um",
    box_monitors: list | None = None,
    pec_z_label: str | None = "PEC mirror (z = 0)",
    show_z_pml: bool = True,
) -> None:
    """Annotate vertical (-z / +z) boundaries and 2D FluxMonitor boxes on an xz plot.

    Workaround for two ``wakis.schematic`` limitations on ``plane="xz"`` cuts:
    (1) ``_draw_schematic_panel`` and ``overlay_geometry`` draw shaded PML strips only
        along the horizontal axis (-x / +x) and summarize -z / +z in the subtitle;
    (2) ``extract_setup_info`` classifies 2D ``FluxMonitor`` boxes (``size[0] > 0`` and
        ``size[2] > 0``) as ``kind="plane"``, which would render as a diagonal segment
        rather than a closed 2D rectangle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes in ``unit`` coordinates.
    solver : SolverFIT3D
        Configured Wakis solver instance.
    unit : {"m", "mm", "um", "nm"}
        Coordinate unit used on ``ax``.
    box_monitors : list of tuple(FluxMonitor, str), optional
        Closed 2D ``FluxMonitor`` boxes and their text labels to draw as rectangles.
    pec_z_label : str or None
        Label for the ``-z`` (or ``z = 0``) PEC mirror plane if present.
    show_z_pml : bool
        If True, shade and label ``-z`` / ``+z`` CPML absorbing layers inside ``ax``.
    """
    from matplotlib.patches import Rectangle

    scale = {"m": 1.0, "mm": 1e3, "um": 1e6, "nm": 1e9}[unit]
    x_nodes = np.asarray(solver.grid.x, dtype=float) * scale
    z_nodes = np.asarray(solver.grid.z, dtype=float) * scale
    x_grid_lo, x_grid_hi = float(x_nodes[0]), float(x_nodes[-1])
    z_lo, z_hi = float(z_nodes[0]), float(z_nodes[-1])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_vis_lo, x_vis_hi = min(xlim), max(xlim)
    z_vis_lo, z_vis_hi = min(ylim), max(ylim)
    x_lo = max(x_grid_lo, x_vis_lo)
    x_hi = min(x_grid_hi, x_vis_hi)
    x_span = max(x_hi - x_lo, 1e-12)
    z_vis_span = max(z_vis_hi - z_vis_lo, 1e-12)

    bc_low_z = str(solver.bc_low[2]).lower()
    bc_high_z = str(solver.bc_high[2]).lower()
    n_pml = int(getattr(solver, "n_pml", 8))

    if show_z_pml and bc_high_z in ("pml", "cpml") and len(z_nodes) > n_pml:
        z_pml_top = float(z_nodes[-1 - n_pml])
        top_hi = min(z_hi, z_vis_hi)
        if z_pml_top < top_hi - 0.02 * z_vis_span:
            rect_top = Rectangle(
                (x_lo, z_pml_top),
                x_span,
                top_hi - z_pml_top,
                facecolor="#64748b",
                edgecolor="#475569",
                alpha=0.14,
                hatch="//",
                lw=0.8,
                zorder=3,
            )
            ax.add_patch(rect_top)
            ax.axhline(z_pml_top, color="#475569", ls=":", lw=0.9, zorder=4)
            ax.text(
                x_lo + 0.03 * x_span,
                0.5 * (z_pml_top + top_hi),
                f"+z: CPML ({n_pml} cells)",
                fontsize=7.0,
                color="#1e293b",
                va="center",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.82, edgecolor="none"),
                zorder=7,
            )

    if show_z_pml and bc_low_z in ("pml", "cpml") and len(z_nodes) > n_pml:
        z_pml_bot = float(z_nodes[n_pml])
        bot_lo = max(z_lo, z_vis_lo)
        if z_pml_bot > bot_lo + 0.02 * z_vis_span:
            rect_bot = Rectangle(
                (x_lo, bot_lo),
                x_span,
                z_pml_bot - bot_lo,
                facecolor="#64748b",
                edgecolor="#475569",
                alpha=0.14,
                hatch="//",
                lw=0.8,
                zorder=3,
            )
            ax.add_patch(rect_bot)
            ax.axhline(z_pml_bot, color="#475569", ls=":", lw=0.9, zorder=4)
            ax.text(
                x_lo + 0.03 * x_span,
                0.5 * (bot_lo + z_pml_bot),
                f"-z: CPML ({n_pml} cells)",
                fontsize=7.0,
                color="#1e293b",
                va="center",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.82, edgecolor="none"),
                zorder=7,
            )

    if bc_low_z == "pec" and pec_z_label is not None and z_lo >= z_vis_lo - 0.02 * z_vis_span:
        ax.axhline(z_lo, color="#0f172a", lw=2.2, zorder=6)
        ax.text(
            x_hi - 0.03 * x_span,
            max(z_lo, z_vis_lo) + 0.035 * z_vis_span,
            pec_z_label,
            fontsize=7.2,
            fontweight="bold",
            color="#0f172a",
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.88, edgecolor="#0f172a", lw=0.7),
            zorder=7,
        )

    if box_monitors:
        for mon, lbl in box_monitors:
            cx, _, cz = (float(v) * scale for v in mon.center)
            sx, _, sz = (float(v) * scale for v in mon.size)
            rect = Rectangle(
                (cx - 0.5 * sx, cz - 0.5 * sz),
                sx,
                sz,
                facecolor="none",
                edgecolor="#0f766e",
                lw=1.3,
                ls="--",
                zorder=6,
            )
            ax.add_patch(rect)
            if lbl:
                ax.text(
                    cx + 0.5 * sx + 0.015 * x_span,
                    cz,
                    lbl,
                    fontsize=7.0,
                    color="#0f766e",
                    va="center",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.85, edgecolor="none"),
                    zorder=7,
                )


