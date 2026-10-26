"""Physics, 3D FIT runners, spherical multipole decomposition, and plotting helpers for Tutorial 4.7 (Magnetic Toroid)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0
from scipy.optimize import brentq
from scipy.special import jn_zeros, spherical_jn, sph_harm_y

from wakis import GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.dispersive import DispersiveMaterial, Lorentz
from wakis.estimate import estimate, preflight
from wakis.field_monitors import DFTMonitor
from wakis.lumped import LumpedPort
from wakis.near2far import Near2FarMonitor
from wakis.sources import ContinuousSource, CurrentSource, GaussianSource, MagneticCurrentSource
from wakis.templates.meshing import graded_axis
from wakis.templates.shapes import Box

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

ETA0 = float(np.sqrt(MU0 / EPS0))
DEFAULT_MEM_LIMIT_MB = 1800.0

# Canonical parameters for Tutorial 4.7
F0_HZ = 0.75e9  # 750 MHz carrier -> lambda_0 = 399.72 mm, k_0 = 15.719 rad/m
LAM0_M = float(C0 / F0_HZ)
K0_RAD_M = float(2.0 * np.pi / LAM0_M)
OMEGA0_RAD_S = float(2.0 * np.pi * F0_HZ)

R_MAJOR_M = 0.015  # Torus major radius R = 15.0 mm (k_0 R = 0.2358)
R_MINOR_M = 0.005  # Torus minor radius r = 5.0 mm (minor thickness 2r = 10.0 mm)
MU_DC_DEFAULT = 20.0  # Lorentz core DC relative permeability (in 20-100 range)
F_RES_CORE_HZ = 6.0e9  # Lorentz magnetic resonance frequency well above operating band
GAMMA_CORE_HZ = 0.04e9  # Low-loss Lorentz damping rate (tan_delta_m ~ 8e-4 at 0.75 GHz)


def _json_clean(obj: Any) -> Any:
    """Recursively convert NumPy scalars and arrays into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}
    return obj


# ---------------------------------------------------------------------------
# 1. Material & Analytical Physics Helpers
# ---------------------------------------------------------------------------


def make_core_material(
    mu_dc: float = MU_DC_DEFAULT,
    f_res_hz: float = F_RES_CORE_HZ,
    gamma_hz: float = GAMMA_CORE_HZ,
    eps_inf: float = 1.0,
    sigma_m: float = 0.0,
    name: str = "Lorentz permeable core",
) -> DispersiveMaterial:
    """Create an isotropic dispersive magnetic core with a single Lorentz mu pole.

    In the Wakis ``exp(+i*omega*t)`` convention, the complex relative permeability is
        mu_r(omega) = mu_inf + delta_mu * omega_0^2 / (omega_0^2 - omega^2 + i * gamma * omega)
                      + sigma_m / (i * omega * mu_0),
    with ``Im(mu_r) <= 0`` (passive medium, ``mu_r = mu' - i * mu''`` with ``mu'' >= 0``).

    Parameters
    ----------
    mu_dc : float, optional
        Static (DC) relative permeability ``mu_r(0) = 1 + delta_mu`` [-]. Default is 20.0.
    f_res_hz : float, optional
        Magnetic Lorentz resonance frequency [Hz]. Default is 6.0e9.
    gamma_hz : float, optional
        Magnetic Lorentz damping frequency [Hz]. Default is 0.04e9.
    eps_inf : float, optional
        Relative permittivity of the core [-]. Default is 1.0.
    sigma_m : float, optional
        Optional magnetic conductivity [Ohm/m]. Default is 0.0.
    name : str, optional
        Display label for the material.

    Returns
    -------
    DispersiveMaterial
        Configured magnetic dispersive material.
    """
    if mu_dc < 1.0:
        raise ValueError("mu_dc must be >= 1.0 for a passive Lorentz magnetic pole.")
    delta_mu = float(mu_dc - 1.0)
    w_res = 2.0 * np.pi * float(f_res_hz)
    gamma = 2.0 * np.pi * float(gamma_hz)
    poles = [Lorentz(delta_mu, w_res, gamma)] if delta_mu > 0.0 else []
    return DispersiveMaterial(
        eps_inf=float(eps_inf),
        poles=[],
        sigma=0.0,
        mu_inf=1.0,
        mu_poles=poles,
        sigma_m=float(sigma_m),
        name=name,
    )


def small_loop_radiation_resistance_theory(
    ka: float | np.ndarray,
    mu_eff: float | np.ndarray = 1.0,
    n_turns: int = 1,
) -> dict[str, np.ndarray]:
    """Compute the analytical radiation resistance and magnetic dipole moment of a small loop.

    For an electrically small circular loop of radius ``a`` (``ka = k_0 * a << 1``)
    with ``N`` turns on a core of effective relative permeability ``mu_eff``, the
    axial magnetic dipole moment for peak current ``I_0`` is
        m_z = mu_eff * N * I_0 * (pi * a^2)   [A m^2],
    radiating time-averaged power
        P_rad = (eta_0 * k_0^4 / (12 * pi)) * |m_z|^2,
    which yields the classic ``(ka)^4`` radiation resistance
        R_rad = 2 * P_rad / |I_0|^2 = (pi / 6) * eta_0 * N^2 * mu_eff^2 * (ka)^4
              ~= 20 * pi^2 * N^2 * mu_eff^2 * (ka)^4   [Ohm].

    Parameters
    ----------
    ka : float or ndarray
        Electrical loop radius ``k_0 * a`` [-].
    mu_eff : float or ndarray, optional
        Effective relative permeability of the loop core [-]. Default is 1.0.
    n_turns : int, optional
        Number of loop turns [-]. Default is 1.

    Returns
    -------
    dict of ndarray
        Dictionary with ``ka``, ``r_rad_exact_ohm``, and ``r_rad_20pi2_ohm``.
    """
    ka_arr = np.asarray(ka, dtype=float)
    if np.any(ka_arr < 0.0):
        raise ValueError("ka must be non-negative.")
    mu_arr = np.asarray(mu_eff, dtype=float)
    n2 = float(n_turns**2)
    r_exact = (np.pi / 6.0) * ETA0 * n2 * (mu_arr**2) * (ka_arr**4)
    r_20pi2 = 20.0 * (np.pi**2) * n2 * (mu_arr**2) * (ka_arr**4)
    return {
        "ka": ka_arr,
        "r_rad_exact_ohm": r_exact,
        "r_rad_20pi2_ohm": r_20pi2,
    }


def toroidal_dipole_radiation_theory(
    freq_hz: float | np.ndarray,
    t_moment_am3: complex | np.ndarray,
    r_major_m: float = 0.0,
    theta_rad: float | np.ndarray | None = None,
    r_ff_m: float = 10.0,
    r_minor_m: float = 0.0,
) -> dict[str, Any]:
    """Compute the analytical equivalent electric dipole p_eff, far field, and radiated power of a toroidal dipole T.

    In the Wakis ``exp(+i * omega * t)`` harmonic convention (where a point electric
    dipole ``p`` has current moment ``I * l = +i * omega * p``; ``wakis/near2far.py:56``),
    an axial toroidal dipole ``T = T_z * z_hat`` defined by
        T = (1 / 10) * int [(r . J) r - 2 r^2 J] dV = (1 / 2) * int (r x M) dV   [A m^3]
    has second-order Taylor radiation vector ``N_perp = -(k_0^2 / 2) int (r_hat . r')^2 J_perp dV = +k_0^2 * T_perp``
    (since integrating ``int x'^2 (curl M)_z dV`` by parts gives ``-2 int x' M_y dV = -2 T_z``).
    Matching ``+i * omega * p_eff = +k_0^2 * T`` yields the exact effective axial electric
    dipole moment (derived here for Wakis's ``exp(+i * omega * t)`` convention):
        p_eff = -(i * k_0 / c_0) * T   (or +(i * k_0 / c_0) * T in the physics exp(-i * omega * t) convention).

    Projecting ``r * exp(+i * k_0 * r) * E_theta^ff = +i * (eta_0 * k_0^3 / (4 * pi)) * T_z * F_1(k_0 R) * sin(theta)``
    onto the normalized TM_10 vector spherical harmonic ``Phi_10,theta = -sqrt(3 / (8 * pi)) * sin(theta)``
    gives the exact complex electric dipole multipole coefficient:
        a_10 = -i * sqrt(eta_0 / (12 * pi)) * k_0^3 * T_z * F_1(k_0 R),   arg(a_10 / T_z) = -pi / 2.

    For a finite ring of major radius ``R = r_major_m``, the exact spherical Bessel
    form factor for the ``l = 1`` TM mode multiplies ``T_z`` by
        F_1(k_0 R) = 3 * j_1(k_0 R) / (k_0 R) = 1 - (k_0 R)^2 / 10 + O((k_0 R)^4),
    which has its first radiation null at the spherical Bessel zero ``k_0 R = 4.49340946``
    (``tan(x) = x``), and the filamentary-ring electric octupole-to-dipole amplitude ratio is
        |a_30| / |a_10|_ring = sqrt(7 / 8) * j_3(k_0 R) / j_1(k_0 R) = (sqrt(14) / 140) * (k_0 R)^2 + O((k_0 R)^4).
    For a finite circular cross-section of minor radius ``r = r_minor_m``, averaging the
    ``l = 3`` spherical-harmonic spatial polynomial ``rho * (rho^2 - 4 * z^2)`` (with ``H_phi ~ 1 / rho``
    cancelling the volume-element factor ``rho``) over ``(rho - R)^2 + z^2 <= r^2`` gives the
    leading finite-cross-section correction factor (derived here):
        (1 / (pi * r^2 * R^2)) * iint (rho^2 - 4 * z^2) dA = 1 - (3 / 4) * (r / R)^2,
        |a_30| / |a_10|_finite_r ≈ sqrt(7 / 8) * (j_3(k_0 R) / j_1(k_0 R)) * (1 - 0.75 * (r / R)^2).

    Parameters
    ----------
    freq_hz : float or ndarray
        Frequency [Hz].
    t_moment_am3 : complex or ndarray
        Axial toroidal dipole moment ``T_z`` [A m^3].
    r_major_m : float, optional
        Torus major radius ``R`` [m] for the finite-size ``3*j_1(kR)/(kR)`` factor. Default is 0.0.
    theta_rad : float or ndarray, optional
        Polar angles [rad] at which to evaluate the far field.
    r_ff_m : float, optional
        Far-field radius ``r`` [m]. Default is 10.0.
    r_minor_m : float, optional
        Torus minor radius ``r`` [m] for the ``1 - 0.75 * (r / R)^2`` octupole correction. Default is 0.0.

    Returns
    -------
    dict
        Analytical ``p_eff_cm`` [C m], ``il_eff_am`` [A m], ``a10_point`` [W^{1/2}],
        ``a10_j1`` [W^{1/2}], ``p_rad_point_w`` [W], ``p_rad_j1_w`` [W],
        ``form_factor_j1`` [-], and optional ``Etheta`` [V/m].
    """
    f = np.asarray(freq_hz, dtype=float)
    if np.any(f <= 0.0):
        raise ValueError("freq_hz must be positive.")
    omega = 2.0 * np.pi * f
    k0 = omega / C0
    tz = np.asarray(t_moment_am3, dtype=np.complex128)

    if r_major_m > 0.0:
        kr = np.maximum(k0 * float(r_major_m), 1e-12)
        f_j1 = 3.0 * spherical_jn(1, kr) / kr
        a30_over_a10_ring = np.sqrt(7.0 / 8.0) * (
            spherical_jn(3, kr) / np.maximum(np.abs(spherical_jn(1, kr)), 1e-30)
        )
        finite_r_factor = (
            max(1.0 - 0.75 * ((float(r_minor_m) / float(r_major_m)) ** 2), 0.0)
            if r_minor_m > 0.0
            else 1.0
        )
        a30_over_a10_finite_r = a30_over_a10_ring * finite_r_factor
    else:
        f_j1 = np.ones_like(k0)
        a30_over_a10_ring = np.zeros_like(k0)
        finite_r_factor = 1.0
        a30_over_a10_finite_r = np.zeros_like(k0)

    # In Wakis exp(+i*omega*t): I*l_eff = +i * omega * p_eff = +k0^2 * T_z -> p_eff = -(i * k0 / c0) * T_z
    p_eff_point = -(1j * k0 / C0) * tz
    p_eff_j1 = p_eff_point * f_j1
    il_eff_point = +(k0**2) * tz
    il_eff_j1 = il_eff_point * f_j1

    # Complex VSH electric-dipole coefficient a_10 (phase -pi/2 relative to T_z in exp(+i*omega*t))
    a10_point = -1j * np.sqrt(ETA0 / (12.0 * np.pi)) * (k0**3) * tz
    a10_j1 = a10_point * f_j1

    # Sinusoidal time-averaged radiated power:
    # P_rad = (eta_0 * k0^4 * c0^2 / (12 * pi)) * |p_eff|^2 = (eta_0 * k0^6 / (12 * pi)) * |T_z|^2
    p_rad_point = (ETA0 * (k0**6) / (12.0 * np.pi)) * (np.abs(tz) ** 2)
    p_rad_j1 = p_rad_point * (f_j1**2)

    out: dict[str, Any] = {
        "freq_hz": f,
        "k0_rad_m": k0,
        "p_eff_point_cm": p_eff_point,
        "p_eff_j1_cm": p_eff_j1,
        "il_eff_point_am": il_eff_point,
        "il_eff_j1_am": il_eff_j1,
        "a10_point": a10_point,
        "a10_j1": a10_j1,
        "a10_over_tz_phase_rad": float(-0.5 * np.pi),
        "form_factor_j1": f_j1,
        "a30_over_a10_ring": a30_over_a10_ring,
        "finite_r_octupole_factor": float(finite_r_factor),
        "a30_over_a10_finite_r": a30_over_a10_finite_r,
        "p_rad_point_w": p_rad_point,
        "p_rad_j1_w": p_rad_j1,
    }
    if theta_rad is not None:
        th = np.asarray(theta_rad, dtype=float)
        # E_theta = -(i * k0 * eta_0 * (I*l_eff) * (-sin(theta)) / (4 * pi * r)) * exp(-i * k0 * r)
        #         = (+i * k0^3 * eta_0 * T_z * F_1 / (4 * pi * r)) * sin(theta) * exp(-i * k0 * r)
        pref = (1j * k0 * ETA0 * il_eff_j1 / (4.0 * np.pi * float(r_ff_m))) * np.exp(-1j * k0 * float(r_ff_m))
        out["Etheta"] = pref[..., None] * np.sin(th) if np.ndim(pref) > 0 else pref * np.sin(th)
        out["Ephi"] = np.zeros_like(out["Etheta"])
    return out


def marinov_toroidal_formulas(
    n_turns: int,
    current_a: float,
    r_major_m: float = R_MAJOR_M,
    r_minor_m: float = R_MINOR_M,
    mu_eff: float | complex = 1.0,
) -> dict[str, float | complex]:
    """Evaluate Marinov et al. (New J. Phys. 9, 324, 2007) Eqs. (6), (9), and Fig. 1 caption.

    For a circular torus of major radius ``d = r_major_m`` and minor radius
    ``R = r_minor_m`` wound with ``N`` poloidal turns carrying current ``I``:
      * Air-core toroidal dipole moment (Marinov Eq. 6: ``T = pi * N * I * d * R^2 / 2``,
        extended here by core effective permeability ``mu_eff``):
            T_z = (1 / 2) * pi * N * I * r_major * r_minor^2 * mu_eff   [A m^3]
      * Parasitic axial magnetic dipole of a single-layer helical winding advancing
        by one net azimuthal turn around the torus (Marinov Eq. 9; McDonald 2008):
            m_z,parasitic = pi * r_major^2 * I * (1 + 0.5 * (r_minor / r_major)^2)   [A m^2]
      * Radius of the reverse-current equatorial compensating loop ``I_comp = -I``
        that cancels ``m_z,parasitic`` (Marinov Eq. 9 and Fig. 1 caption):
            R_comp = r_major * sqrt(1 + 0.5 * (r_minor / r_major)^2)   [m]
    """
    if n_turns <= 0 or r_major_m <= 0.0 or r_minor_m <= 0.0:
        raise ValueError("n_turns, r_major_m, and r_minor_m must be positive.")
    tz = 0.5 * np.pi * float(n_turns) * float(current_a) * float(r_major_m) * (float(r_minor_m) ** 2) * mu_eff
    corr = 1.0 + 0.5 * ((float(r_minor_m) / float(r_major_m)) ** 2)
    mz_par = np.pi * (float(r_major_m) ** 2) * float(current_a) * corr
    r_comp = float(r_major_m) * float(np.sqrt(corr))
    return {
        "t_z_am3": tz,
        "m_z_parasitic_am2": float(mz_par),
        "r_comp_m": r_comp,
        "minor_to_major_corr": float(corr),
    }


def hesitivity_efficiency_and_chu_q_theory(
    freq_hz: float | np.ndarray,
    mu_r: complex | np.ndarray,
    r_major_m: float = R_MAJOR_M,
    r_minor_m: float = R_MINOR_M,
    eps_r: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute toroidal radiation efficiency, magnetic hesitivity h_m, TE01 cutoff, and Chu Q bound.

    For a torus of major radius ``R`` and minor radius ``r`` (volume
    ``V_core = 2 * pi^2 * R * r^2``, enclosing sphere radius ``a = R + r``) with
    uniform azimuthal field ``H_phi`` inside a core of complex permeability
    ``mu_r = mu' - i * mu''``:
      * Toroidal dipole (core + winding): ``T_z = (1 / 2) * R * mu_r * H_phi * V_core``.
      * Radiated power: ``P_rad = (eta_0 * k_0^6 / (12 * pi)) * |T_z|^2 * F_1(k_0 R)^2``,
        where ``F_1(k_0 R) = 3 * j_1(k_0 R) / (k_0 R)``.
      * Core magnetic loss power: ``P_loss = (1 / 2) * omega * mu_0 * mu'' * |H_phi|^2 * V_core``.
      * Loss figure of merit: ``f(mu) = mu'' / |mu_r - 1|^2`` and magnetic hesitivity
        ``h_m = omega * mu_0 * |mu_r - 1|^2 / (2 * mu'') = omega * mu_0 / (2 * f(mu))`` [Ohm/m^3].
      * Radiation efficiency (derived here):
            eta_rad = P_rad / (P_rad + P_loss)
                    = 1 / (1 + (12 / (pi * (k_0 r)^2 * (k_0 R)^3 * F_1(k_0 R)^2)) * (mu'' / |mu_r|^2))
      * Radiation quality factor:
            Q_rad = 2 * omega * W_m / P_rad
                  = (12 * mu') / (pi * |mu_r|^2 * (k_0 r)^2 * (k_0 R)^3 * F_1(k_0 R)^2)
                  ≈ 12 / (pi * mu' * (k_0 r)^2 * (k_0 R)^3 * F_1(k_0 R)^2).
      * Core TE01 slab/cylinder trapped-mode onset frequency:
            f_TE01 = c_0 / (4 * r_minor * sqrt(mu' * eps_r)).
      * Chu lower bound for a single TM10 mode in enclosing sphere ``a = R + r``:
            Q_Chu = 1 / (k_0 a)^3 + 1 / (k_0 a).
    """
    f = np.asarray(freq_hz, dtype=float)
    omega = 2.0 * np.pi * f
    k0 = omega / C0
    mu = np.asarray(mu_r, dtype=np.complex128)
    mu_p = np.real(mu)
    mu_pp = np.maximum(-np.imag(mu), 0.0)
    tan_delta_m = mu_pp / np.maximum(mu_p, 1e-30)

    v_core = 2.0 * (np.pi**2) * float(r_major_m) * (float(r_minor_m) ** 2)
    a_sphere = float(r_major_m + r_minor_m)
    ka = k0 * a_sphere
    kr_maj = np.maximum(k0 * float(r_major_m), 1e-12)
    f_j1 = 3.0 * spherical_jn(1, kr_maj) / kr_maj

    # Loss function f(mu) = mu'' / |mu_r - 1|^2 and hesitivity h_m [Ohm/m^3]
    chi_sq = np.maximum(np.abs(mu - 1.0) ** 2, 1e-30)
    mu_sq = np.maximum(np.abs(mu) ** 2, 1e-30)
    f_mu = mu_pp / chi_sq
    h_m = omega * MU0 / np.maximum(2.0 * f_mu, 1e-30)

    # Ratio P_loss / P_rad = [24*pi / (k0^5 * R^2 * V_core * F_1^2)] * (mu'' / |mu_r|^2)
    #                      = [12 / (pi * (k0*r)^2 * (k0*R)^3 * F_1^2)] * (mu'' / |mu_r|^2)
    geom_factor = 24.0 * np.pi / np.maximum((k0**5) * (float(r_major_m) ** 2) * v_core * (f_j1**2), 1e-30)
    loss_to_rad_ratio = geom_factor * (mu_pp / mu_sq)
    eta_rad = 1.0 / (1.0 + loss_to_rad_ratio)

    # Radiation Q = 2 * omega * W_mag / P_rad (with W_mag = 0.25 * mu_0 * mu' * |H|^2 * V_core)
    q_rad_toroid = geom_factor * (mu_p / mu_sq)
    q_chu = 1.0 / (ka**3) + 1.0 / ka

    f_te01 = C0 / (4.0 * float(r_minor_m) * np.sqrt(np.maximum(mu_p * float(eps_r), 1.0)))

    return {
        "freq_hz": f,
        "ka": ka,
        "form_factor_j1": f_j1,
        "v_core_m3": np.asarray(v_core),
        "tan_delta_m": tan_delta_m,
        "f_mu": f_mu,
        "hesitivity_ohm_m3": h_m,
        "loss_to_rad_ratio": loss_to_rad_ratio,
        "eta_rad_theory": eta_rad,
        "eta_rad": eta_rad,
        "q_rad_toroid": q_rad_toroid,
        "q_chu": q_chu,
        "f_te01_hz": f_te01,
    }


def small_loop_radiation_resistance(freq_hz: float, a_m: float, mu_eff: float = 1.0, n_turns: int = 1) -> float:
    """Convenience wrapper returning exact small-loop radiation resistance R_rad [Ohm] at freq_hz and radius a_m."""
    ka = (2.0 * np.pi * float(freq_hz) / C0) * float(a_m)
    return float(small_loop_radiation_resistance_theory(ka, mu_eff=mu_eff, n_turns=n_turns)["r_rad_exact_ohm"])


def marinov_toroidal_moment_quasistatic(
    n_turns: int,
    current_a: float,
    r_major_m: float = R_MAJOR_M,
    r_minor_m: float = R_MINOR_M,
    mu_r_eff: float = 1.0,
) -> dict[str, float]:
    """Return Marinov et al. (2007) quasi-static and exact Pappus toroidal dipole moments [A m^3]."""
    res = marinov_toroidal_formulas(
        n_turns=n_turns,
        current_a=current_a,
        r_major_m=r_major_m,
        r_minor_m=r_minor_m,
        mu_eff=float(mu_r_eff),
    )
    tz_val = float(np.real(res["t_z_am3"]))
    return {
        "t_quasistatic_am3": tz_val,
        "t_exact_am3": tz_val,
        "m_z_parasitic_am2": float(res["m_z_parasitic_am2"]),
        "r_comp_m": float(res["r_comp_m"]),
    }


def toroidal_efficiency_and_q_theory(
    freq_hz: float,
    r_major_m: float = R_MAJOR_M,
    r_minor_m: float = R_MINOR_M,
    mu_real: float = MU_DC_DEFAULT,
    tan_delta_m: float = 1e-3,
    eps_r: float = 1.0,
) -> dict[str, float]:
    """Return scalar dictionary of analytical toroidal radiation efficiency, Q_rad, and Chu Q."""
    mu_c = float(mu_real) * (1.0 - 1j * float(tan_delta_m))
    res = hesitivity_efficiency_and_chu_q_theory(
        freq_hz=freq_hz,
        mu_r=mu_c,
        r_major_m=r_major_m,
        r_minor_m=r_minor_m,
        eps_r=eps_r,
    )
    return {k: float(np.asarray(v)) for k, v in res.items()}


def hertzian_dipole_fields(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    freq_hz: float,
    p_z: complex = 1.0 + 0.0j,
    r_min_clip: float = 0.005,
) -> dict[str, np.ndarray]:
    """Exact full-space (near + intermediate + far) E and H fields of a z-directed Hertzian dipole p_z.

    Evaluates Jackson Eq. (9.18) in the Wakis ``exp(+i * omega * t)`` convention
    (propagating phase ``exp(-i * k_0 * r)``):
        H_phi = +(i * omega * p_z / (4 * pi)) * sin(theta) * [i * k_0 / r + 1 / r^2] * exp(-i * k_0 * r)
        E_r   = (p_z / (2 * pi * eps_0)) * cos(theta) * [i * k_0 / r^2 + 1 / r^3] * exp(-i * k_0 * r)
        E_th  = (p_z / (4 * pi * eps_0)) * sin(theta) * [-k_0^2 / r + i * k_0 / r^2 + 1 / r^3] * exp(-i * k_0 * r)
    """
    omega = 2.0 * np.pi * float(freq_hz)
    k0 = omega / C0
    r = np.maximum(np.sqrt(x**2 + y**2 + z**2), float(r_min_clip))
    rho = np.sqrt(x**2 + y**2)
    sin_th = rho / r
    cos_th = z / r
    cos_ph = np.where(rho > 1e-12, x / rho, 1.0)
    sin_ph = np.where(rho > 1e-12, y / rho, 0.0)

    exp_kr = np.exp(-1j * k0 * r)
    er = (p_z / (2.0 * np.pi * EPS0)) * cos_th * (1j * k0 / (r**2) + 1.0 / (r**3)) * exp_kr
    eth = (p_z / (4.0 * np.pi * EPS0)) * sin_th * (-(k0**2) / r + 1j * k0 / (r**2) + 1.0 / (r**3)) * exp_kr
    hph = (1j * omega * p_z / (4.0 * np.pi)) * sin_th * (1j * k0 / r + 1.0 / (r**2)) * exp_kr

    ex = (er * sin_th + eth * cos_th) * cos_ph
    ey = (er * sin_th + eth * cos_th) * sin_ph
    ez = er * cos_th - eth * sin_th
    hx = -hph * sin_ph
    hy = hph * cos_ph
    hz = np.zeros_like(hx)
    return {"Ex": ex, "Ey": ey, "Ez": ez, "Hx": hx, "Hy": hy, "Hz": hz}


# ---------------------------------------------------------------------------
# 2. Exact Spherical Multipole Decomposition (Far-Field VSH & Volume Currents)
# ---------------------------------------------------------------------------


def make_quadrature_sphere(
    n_theta: int = 24,
    n_phi: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build Gauss-Legendre polar and uniform azimuthal quadrature nodes on the unit sphere."""
    mu, w_mu = np.polynomial.legendre.leggauss(n_theta)
    order = np.argsort(-mu)
    mu = mu[order]
    w_mu = w_mu[order]
    theta_1d = np.arccos(mu)
    phi_1d = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    w_phi = np.full(n_phi, 2.0 * np.pi / n_phi)
    tt, pp = np.meshgrid(theta_1d, phi_1d, indexing="ij")
    return theta_1d, phi_1d, w_mu, w_phi, tt.ravel(), pp.ravel()


def spherical_multipole_from_farfield(
    theta_1d: np.ndarray,
    phi_1d: np.ndarray,
    w_theta: np.ndarray,
    w_phi: np.ndarray,
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    l_max: int = 5,
    r_ff: float = 1.0,
    k0: float = 1.0,
    s_norm: complex = 1.0 + 0.0j,
) -> dict[str, Any]:
    """Project simulated far-field (E_theta, E_phi) onto orthonormal electric (a_lm) and magnetic (b_lm) VSHs.

    On a far-field sphere of radius ``r_ff``, the transverse electric field is
    expanded into orthonormal Vector Spherical Harmonics (Jackson Ch. 9):
        Phi_{lm}(theta, phi) = [l(l+1)]^{-1/2} * r * grad_perp Y_{lm}(theta, phi)   (TM / electric)
        Psi_{lm}(theta, phi) = r_hat x Phi_{lm}(theta, phi)                         (TE / magnetic)
    normalized such that
        P_rad = (1 / (2 * eta_0)) * oint |E_FF / S(f)|^2 r^2 dOmega
              = sum_{l=1}^{l_max} sum_{m=-l}^{l} (|a_{lm}|^2 + |b_{lm}|^2).

    Parameters
    ----------
    theta_1d, phi_1d : ndarray
        1D polar and azimuthal quadrature angles [rad].
    w_theta, w_phi : ndarray
        1D quadrature weights for ``d(cos theta)`` and ``dphi``.
    e_theta, e_phi : ndarray
        Complex far-field components on the ``(n_theta * n_phi,)`` or ``(n_theta, n_phi)`` grid.
    l_max : int, optional
        Maximum multipole degree ``l`` (default 5).
    r_ff : float, optional
        Far-field evaluation radius [m]. Default is 1.0.
    k0 : float, optional
        Free-space wavenumber [rad/m]. Default is 1.0.
    s_norm : complex, optional
        Source Fourier spectrum ``S(f_0)`` for normalizing DFT fields. Default is 1.0.

    Returns
    -------
    dict
        Dictionary containing ``a_lm``, ``b_lm``, ``power_a_l``, ``power_b_l``,
        ``p_rad_quad``, ``p_rad_multipole_sum``, ``b1_over_a1``, ``b_tot_over_a_tot``,
        ``a30_over_a10``, ``c4_b_over_a10``, and ``cross_pol_power_ratio``.
    """
    n_t = len(theta_1d)
    n_p = len(phi_1d)
    tt, pp = np.meshgrid(theta_1d, phi_1d, indexing="ij")
    weights = w_theta[:, None] * w_phi[None, :]

    phase_corr = (float(r_ff) * np.exp(1j * float(k0) * float(r_ff))) / complex(s_norm)
    et = np.asarray(e_theta, dtype=np.complex128).reshape(n_t, n_p) * phase_corr
    ep = np.asarray(e_phi, dtype=np.complex128).reshape(n_t, n_p) * phase_corr

    a_lm: dict[tuple[int, int], complex] = {}
    b_lm: dict[tuple[int, int], complex] = {}
    power_a_l = np.zeros(l_max + 1, dtype=float)
    power_b_l = np.zeros(l_max + 1, dtype=float)

    dtheta_fd = 1e-6
    scale_pwr = np.sqrt(1.0 / (2.0 * ETA0))
    for l in range(1, l_max + 1):
        norm_l = 1.0 / np.sqrt(l * (l + 1.0))
        for m in range(-l, l + 1):
            y_lm = sph_harm_y(l, m, tt, pp)
            y_plus = sph_harm_y(l, m, np.clip(tt + dtheta_fd, 0.0, np.pi), pp)
            y_minus = sph_harm_y(l, m, np.clip(tt - dtheta_fd, 0.0, np.pi), pp)
            dy_dtheta = (y_plus - y_minus) / (2.0 * dtheta_fd)
            im_y_over_sin = (1j * m * y_lm) / np.maximum(np.sin(tt), 1e-12)

            phi_t = norm_l * dy_dtheta
            phi_p = norm_l * im_y_over_sin
            psi_t = -phi_p
            psi_p = phi_t

            c_a = complex(np.sum(weights * (np.conj(phi_t) * et + np.conj(phi_p) * ep)) * scale_pwr)
            c_b = complex(np.sum(weights * (np.conj(psi_t) * et + np.conj(psi_p) * ep)) * scale_pwr)
            a_lm[(l, m)] = c_a
            b_lm[(l, m)] = c_b
            power_a_l[l] += abs(c_a) ** 2
            power_b_l[l] += abs(c_b) ** 2

    p_theta = float(np.sum(weights * (np.abs(et) ** 2)) / (2.0 * ETA0))
    p_phi = float(np.sum(weights * (np.abs(ep) ** 2)) / (2.0 * ETA0))
    p_rad_quad = p_theta + p_phi

    a1_amp = float(np.sqrt(max(power_a_l[1], 1e-60)))
    b1_amp = float(np.sqrt(max(power_b_l[1], 0.0)))
    a_tot_amp = float(np.sqrt(max(np.sum(power_a_l[1:]), 1e-60)))
    b_tot_amp = float(np.sqrt(max(np.sum(power_b_l[1:]), 0.0)))
    a10_amp = max(abs(a_lm[(1, 0)]), 1e-30)

    # C4 azimuthal electric and magnetic residuals (|m| == 4 for l = 4, ..., l_max)
    c4_a_pwr = 0.0
    c4_b_pwr = 0.0
    for l_c4 in range(4, l_max + 1):
        c4_a_pwr += abs(a_lm[(l_c4, 4)]) ** 2 + abs(a_lm[(l_c4, -4)]) ** 2
        c4_b_pwr += abs(b_lm[(l_c4, 4)]) ** 2 + abs(b_lm[(l_c4, -4)]) ** 2
    c4_tot_pwr = c4_a_pwr + c4_b_pwr

    p_vsh_tot = float(np.sum(power_a_l) + np.sum(power_b_l))
    p_a10_frac = float((abs(a_lm[(1, 0)]) ** 2) / max(p_vsh_tot, 1e-30))

    return {
        "a_lm": a_lm,
        "b_lm": b_lm,
        "power_a_l": power_a_l,
        "power_b_l": power_b_l,
        "p_a_l": power_a_l[1:],
        "p_b_l": power_b_l[1:],
        "p_vsh_tot": p_vsh_tot,
        "p_a10_frac": p_a10_frac,
        "p_rad_quad": p_rad_quad,
        "p_rad_multipole_sum": p_vsh_tot,
        "p_theta": p_theta,
        "p_phi": p_phi,
        "cross_pol_power_ratio": float(p_phi / max(p_theta, 1e-60)),
        "a10": a_lm[(1, 0)],
        "b10": b_lm[(1, 0)],
        "a1_amp": a1_amp,
        "b1_amp": b1_amp,
        "b1_over_a1": float(b1_amp / a1_amp),
        "b1_over_a10": float(b1_amp / a10_amp),
        "b10_over_a10": float(abs(b_lm[(1, 0)]) / a10_amp),
        "b_tot_over_a_tot": float(b_tot_amp / a_tot_amp),
        "a20_over_a10": float(abs(a_lm[(2, 0)]) / a10_amp) if l_max >= 2 else 0.0,
        "a30_over_a10": float(abs(a_lm[(3, 0)]) / a10_amp) if l_max >= 3 else 0.0,
        "a40_over_a10": float(abs(a_lm[(4, 0)]) / a10_amp) if l_max >= 4 else 0.0,
        "c4_a_over_a10": float(np.sqrt(c4_a_pwr) / a10_amp) if l_max >= 4 else 0.0,
        "c4_b_over_a10": float(np.sqrt(c4_b_pwr) / a10_amp) if l_max >= 4 else 0.0,
        "c4_m4_over_a10": float(np.sqrt(c4_tot_pwr) / a10_amp) if l_max >= 4 else 0.0,
    }


def compute_dubovik_tugushev_and_magnetization_moments(
    grid: GridFIT3D,
    solver: SolverFIT3D,
    hx_f: np.ndarray,
    hy_f: np.ndarray,
    hz_f: np.ndarray,
    mx_mask: np.ndarray,
    my_mask: np.ndarray,
    mz_mask: np.ndarray,
    chi_m: complex,
    j_free_flat: np.ndarray,
    freq_hz: float = F0_HZ,
) -> dict[str, Any]:
    """Compute volume-integrated primitive moments (p, m, T_M, T_DT) on the 3D FIT grid.

    Verifies two key first-principles identities on the discrete staggered grid:
      1. Closed magnetization loop: ``m = int M dV = 0`` for a symmetric torus or C4 rod square.
      2. Dubovik-Tugushev identity: for ``J_b = curl_dual(M)``, the toroidal moment
            T_DT = (1 / 10) * int [(r . J_b) r - 2 r^2 J_b] dV
         equals the magnetization loop integral
            T_M  = (1 / 2) * int (r x M) dV
         identically via integration by parts.
    """
    nx, ny, nz = grid.Nx, grid.Ny, grid.Nz
    n_cells = nx * ny * nz
    omega = 2.0 * np.pi * float(freq_hz)

    tl_arr = grid.tL.toarray()
    ia_arr = grid.iA.toarray()
    l_arr = grid.L.toarray()
    ita_arr = grid.itA.toarray()

    dv_hx = (tl_arr[:n_cells] / ia_arr[:n_cells]).reshape(nx, ny, nz, order="F")
    dv_hy = (tl_arr[n_cells : 2 * n_cells] / ia_arr[n_cells : 2 * n_cells]).reshape(nx, ny, nz, order="F")
    dv_hz = (tl_arr[2 * n_cells :] / ia_arr[2 * n_cells :]).reshape(nx, ny, nz, order="F")

    dv_jx = (l_arr[:n_cells] / ita_arr[:n_cells]).reshape(nx, ny, nz, order="F")
    dv_jy = (l_arr[n_cells : 2 * n_cells] / ita_arr[n_cells : 2 * n_cells]).reshape(nx, ny, nz, order="F")
    dv_jz = (l_arr[2 * n_cells :] / ita_arr[2 * n_cells :]).reshape(nx, ny, nz, order="F")

    xc = 0.5 * (grid.x[:-1] + grid.x[1:])
    yc = 0.5 * (grid.y[:-1] + grid.y[1:])
    zc = 0.5 * (grid.z[:-1] + grid.z[1:])
    xn, yn, zn = grid.x[:-1], grid.y[:-1], grid.z[:-1]

    xx_hx, yy_hx, zz_hx = np.meshgrid(xn, yc, zc, indexing="ij")
    xx_hy, yy_hy, zz_hy = np.meshgrid(xc, yn, zc, indexing="ij")
    xx_hz, yy_hz, zz_hz = np.meshgrid(xc, yc, zn, indexing="ij")

    mx = chi_m * hx_f * mx_mask
    my = chi_m * hy_f * my_mask
    mz = chi_m * hz_f * mz_mask

    # Net magnetic dipole m = int M dV
    m_vec = np.array([
        complex(np.sum(mx * dv_hx)),
        complex(np.sum(my * dv_hy)),
        complex(np.sum(mz * dv_hz)),
    ])
    m_abs_integral = float(
        np.sum(np.abs(mx) * dv_hx) + np.sum(np.abs(my) * dv_hy) + np.sum(np.abs(mz) * dv_hz)
    )
    m_norm_residual = float(np.linalg.norm(m_vec) / max(m_abs_integral, 1e-30))

    # Toroidal dipole from M: T_M = 0.5 * int (r x M) dV
    tx_core = 0.5 * complex(np.sum(yy_hz * mz * dv_hz) - np.sum(zz_hy * my * dv_hy))
    ty_core = 0.5 * complex(np.sum(zz_hx * mx * dv_hx) - np.sum(xx_hz * mz * dv_hz))
    tz_core = 0.5 * complex(np.sum(xx_hy * my * dv_hy) - np.sum(yy_hx * mx * dv_hx))

    # Bound current J_b = itDa * C^T * tDs * M on primary E-edges
    m_flat = np.concatenate([mx.ravel(order="F"), my.ravel(order="F"), mz.ravel(order="F")])
    if hasattr(solver, "_curl_dual_op"):
        jb_flat = np.asarray(solver._curl_dual_op @ m_flat, dtype=np.complex128)
    else:
        jb_flat = np.asarray(solver.itDa @ (solver.C.T @ (solver.tDs @ m_flat)), dtype=np.complex128)

    xx_jx, yy_jx, zz_jx = np.meshgrid(xc, yn, zn, indexing="ij")
    xx_jy, yy_jy, zz_jy = np.meshgrid(xn, yc, zn, indexing="ij")
    xx_jz, yy_jz, zz_jz = np.meshgrid(xn, yn, zc, indexing="ij")

    def dubovik_tugushev_from_j(j_vec_flat: np.ndarray, factor_r2: float = 2.0) -> tuple[complex, np.ndarray]:
        jx = j_vec_flat[:n_cells].reshape(nx, ny, nz, order="F")
        jy = j_vec_flat[n_cells : 2 * n_cells].reshape(nx, ny, nz, order="F")
        jz = j_vec_flat[2 * n_cells :].reshape(nx, ny, nz, order="F")
        r2_z = xx_jz**2 + yy_jz**2 + zz_jz**2
        tz_dt = 0.1 * complex(
            np.sum(xx_jx * zz_jx * jx * dv_jx)
            + np.sum(yy_jy * zz_jy * jy * dv_jy)
            + np.sum((zz_jz**2 - factor_r2 * r2_z) * jz * dv_jz)
        )
        # Primitive electric dipole p = (1 / (i * omega)) * int J dV
        p_vec = (1.0 / (1j * omega)) * np.array([
            complex(np.sum(jx * dv_jx)),
            complex(np.sum(jy * dv_jy)),
            complex(np.sum(jz * dv_jz)),
        ])
        return tz_dt, p_vec

    tz_core_dt, p_bound = dubovik_tugushev_from_j(jb_flat, factor_r2=2.0)
    tz_core_dt_mut, _ = dubovik_tugushev_from_j(jb_flat, factor_r2=1.0)
    tz_free_dt, p_free = dubovik_tugushev_from_j(j_free_flat, factor_r2=2.0)
    tz_tot = tz_core + tz_free_dt
    p_tot = p_bound + p_free

    dt_vs_m_rel_err = float(abs(tz_core_dt - tz_core) / max(abs(tz_core), 1e-30))
    dt_mutated_rel_err = float(abs(tz_core_dt_mut - tz_core) / max(abs(tz_core), 1e-30))

    return {
        "m_vec_am2": m_vec,
        "m_abs_integral_am2": m_abs_integral,
        "m_norm_residual": m_norm_residual,
        "m_cancel_ratio": m_norm_residual,
        "t_core_vec_am3": np.array([tx_core, ty_core, tz_core]),
        "tz_core_m_am3": tz_core,
        "tz_mag_core_am3": tz_core,
        "tz_core_dt_am3": tz_core_dt,
        "tz_core_dt_mutated_am3": tz_core_dt_mut,
        "tz_free_dt_am3": tz_free_dt,
        "tz_tot_am3": tz_tot,
        "dt_vs_m_rel_err": dt_vs_m_rel_err,
        "dt_vs_mag_rel_err": dt_vs_m_rel_err,
        "dt_mutated_rel_err": dt_mutated_rel_err,
        "p_tot_vec_cm": p_tot,
    }


# ---------------------------------------------------------------------------
# 3. 3D FIT Grid, Core & Source Builders
# ---------------------------------------------------------------------------


def build_toroid_grid(
    *,
    dx_core_m: float = 0.0025,
    dx_vac_m: float = 0.015,
    core_xy_half_m: float = 0.0225,
    core_z_half_m: float = 0.0075,
    n2f_xy_half_m: float = 0.055,
    n2f_z_half_m: float = 0.045,
    domain_xy_half_m: float = 0.150,
    domain_z_half_m: float = 0.125,
    grading: float = 1.32,
    graded: bool = True,
    grid_asymmetry_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> GridFIT3D:
    """Construct a 3D FIT mesh for the magnetic toroid and 4-rod square studies.

    Satisfies all project grid rules inside the dispersive permeable core
    (``n_core = sqrt(20.3) = 4.51``, ``lambda_core = 88.7 mm`` at 750 MHz):
      * ``dx_core_m = 2.5 mm`` -> ``35.5 cells/lambda_core >= 20`` and ``4.0 cells``
        across the ``2r = 10 mm`` core thickness;
      * ``dx_core_m = 2.0 mm`` -> ``44.4 cells/lambda_core >= 20`` and ``5.0 cells``
        across the ``2r = 10 mm`` core thickness.
    """
    if graded:
        dx_mid = min(0.007, 2.6 * dx_core_m)
        x = graded_axis(
            [-domain_xy_half_m, -n2f_xy_half_m, -core_xy_half_m, core_xy_half_m, n2f_xy_half_m, domain_xy_half_m],
            [dx_vac_m, dx_mid, dx_core_m, dx_mid, dx_vac_m],
            grading=grading,
        )
        z = graded_axis(
            [-domain_z_half_m, -n2f_z_half_m, -core_z_half_m, core_z_half_m, n2f_z_half_m, domain_z_half_m],
            [dx_vac_m, dx_mid, dx_core_m, dx_mid, dx_vac_m],
            grading=grading,
        )
    else:
        nx_half = max(8, int(np.round(domain_xy_half_m / dx_core_m)))
        nz_half = max(6, int(np.round(domain_z_half_m / dx_core_m)))
        x = np.linspace(-domain_xy_half_m, domain_xy_half_m, 2 * nx_half + 1)
        z = np.linspace(-domain_z_half_m, domain_z_half_m, 2 * nz_half + 1)

    y = x.copy()
    if any(abs(v) > 0.0 for v in grid_asymmetry_m):
        x = x + float(grid_asymmetry_m[0])
        y = y + float(grid_asymmetry_m[1])
        z = z + float(grid_asymmetry_m[2])

    return GridFIT3D(
        float(x[0]),
        float(x[-1]),
        float(y[0]),
        float(y[-1]),
        float(z[0]),
        float(z[-1]),
        x=x,
        y=y,
        z=z,
        verbose=0,
    )


class DivergenceFreeCurrentSource:
    """Impressed divergence-free electric current J = itDa * C^T * tDs * H_drive on the FIT grid."""

    def __init__(self, src_time: Any, j_flat: np.ndarray, center: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.src_time = src_time
        self.j_flat = np.asarray(j_flat, dtype=float)
        self.center = np.asarray(center, dtype=float)
        self.size = np.array([0.030, 0.030, 0.010], dtype=float)
        self.component = "z"
        self.is_first_update = True

    def _setup(self, solver: SolverFIT3D) -> None:
        nz_idx = np.flatnonzero(np.abs(self.j_flat) > 0.0)
        self.flat_idx = solver.J.xp.asarray(nz_idx)
        self.weights = solver.J.xp.asarray(self.j_flat[nz_idx], dtype=solver.J.dtype)
        self.J_old = solver.J.xp.zeros_like(self.weights)

    def update(self, solver: SolverFIT3D, t: float) -> None:
        if self.is_first_update:
            self._setup(solver)
            self.is_first_update = False
        a = float(self.src_time(t + 0.5 * solver.dt))
        j_new = self.weights * a
        solver.J.array[self.flat_idx] += j_new - self.J_old
        self.J_old = j_new


class AzimuthalMagneticRingSource:
    """Impressed azimuthal magnetic current ring M_phi on the dual H-edges of a torus."""

    def __init__(self, src_time: Any, m_flat: np.ndarray):
        self.src_time = src_time
        self.m_flat = np.asarray(m_flat, dtype=float)
        self.center = np.array([0.0, 0.0, 0.0])
        self.size = np.array([0.030, 0.030, 0.010])
        self.component = "x"
        self.is_first_update = True

    def _setup(self, solver: SolverFIT3D) -> None:
        from wakis.field import Field

        if solver.M is None:
            solver.M = Field(solver.Nx, solver.Ny, solver.Nz, use_gpu=solver.use_gpu, dtype=solver.dtype)
        nz_idx = np.flatnonzero(np.abs(self.m_flat) > 0.0)
        self.flat_idx = solver.H.xp.asarray(nz_idx)
        self.weights = solver.H.xp.asarray(self.m_flat[nz_idx], dtype=solver.H.dtype)
        self.M_old = solver.H.xp.zeros_like(self.weights)

    def update(self, solver: SolverFIT3D, t: float) -> None:
        if self.is_first_update:
            self._setup(solver)
            self.is_first_update = False
        a = float(self.src_time(t))
        m_new = self.weights * a
        solver.M.array[self.flat_idx] += m_new - self.M_old
        self.M_old = m_new


def _split_polyline_at_grid_planes(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    z_pts: np.ndarray,
    xn: np.ndarray,
    yn: np.ndarray,
    zn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split each segment of a closed 3D polyline at every Cartesian grid plane crossing.

    Every resulting sub-segment lies strictly within a single grid cell ``(i, j, k)``, enabling
    exact charge-conserving (Villasenor-Buneman) edge current deposition on ``GridFIT3D``.
    """
    if (
        abs(float(x_pts[0] - x_pts[-1])) > 1e-15
        or abs(float(y_pts[0] - y_pts[-1])) > 1e-15
        or abs(float(z_pts[0] - z_pts[-1])) > 1e-15
    ):
        x_pts = np.append(x_pts, x_pts[0])
        y_pts = np.append(y_pts, y_pts[0])
        z_pts = np.append(z_pts, z_pts[0])

    x0_list: list[float] = []
    y0_list: list[float] = []
    z0_list: list[float] = []
    x1_list: list[float] = []
    y1_list: list[float] = []
    z1_list: list[float] = []
    for i in range(len(x_pts) - 1):
        xa, ya, za = float(x_pts[i]), float(y_pts[i]), float(z_pts[i])
        xb, yb, zb = float(x_pts[i + 1]), float(y_pts[i + 1]), float(z_pts[i + 1])
        ts = [0.0, 1.0]
        if xb != xa:
            lo, hi = min(xa, xb), max(xa, xb)
            for p in xn[(xn > lo + 1e-14) & (xn < hi - 1e-14)]:
                ts.append((float(p) - xa) / (xb - xa))
        if yb != ya:
            lo, hi = min(ya, yb), max(ya, yb)
            for p in yn[(yn > lo + 1e-14) & (yn < hi - 1e-14)]:
                ts.append((float(p) - ya) / (yb - ya))
        if zb != za:
            lo, hi = min(za, zb), max(za, zb)
            for p in zn[(zn > lo + 1e-14) & (zn < hi - 1e-14)]:
                ts.append((float(p) - za) / (zb - za))
        ts_sorted = sorted(set(ts))
        for k in range(len(ts_sorted) - 1):
            t0, t1 = ts_sorted[k], ts_sorted[k + 1]
            if t1 - t0 < 1e-14:
                continue
            x0_list.append(xa + t0 * (xb - xa))
            y0_list.append(ya + t0 * (yb - ya))
            z0_list.append(za + t0 * (zb - za))
            x1_list.append(xa + t1 * (xb - xa))
            y1_list.append(ya + t1 * (yb - ya))
            z1_list.append(za + t1 * (zb - za))
    return (
        np.asarray(x0_list, dtype=float),
        np.asarray(y0_list, dtype=float),
        np.asarray(z0_list, dtype=float),
        np.asarray(x1_list, dtype=float),
        np.asarray(y1_list, dtype=float),
        np.asarray(z1_list, dtype=float),
    )


def deposit_closed_wire_on_edges(
    grid: GridFIT3D,
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    z_pts: np.ndarray,
    current_amp: float,
) -> np.ndarray:
    """Deposit a closed 3D wire current path carrying ``current_amp`` [A] onto primary electric edges.

    Uses cell-split Villasenor-Buneman edge line-integral weights so that:
    1. Discrete nodal charge conservation ``div_FIT J == 0`` holds to machine precision (~1e-14)
       for any closed wire path without ad-hoc projection.
    2. Linear dipole moments ``0.5 * int (r x J) dV = 0.5 * I * oint (r x dl)`` are preserved
       continuously across sub-cell wire positions with zero rescaling of ``current_amp``.
    """
    xn, yn, zn = grid.x, grid.y, grid.z
    nx, ny, nz = grid.Nx, grid.Ny, grid.Nz
    ix_cur = np.zeros((nx, ny, nz), dtype=float)
    iy_cur = np.zeros((nx, ny, nz), dtype=float)
    iz_cur = np.zeros((nx, ny, nz), dtype=float)

    x0, y0, z0, x1, y1, z1 = _split_polyline_at_grid_planes(x_pts, y_pts, z_pts, xn, yn, zn)
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    zm = 0.5 * (z0 + z1)
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    ix = np.clip(np.searchsorted(xn, xm) - 1, 0, nx - 2)
    iy = np.clip(np.searchsorted(yn, ym) - 1, 0, ny - 2)
    iz = np.clip(np.searchsorted(zn, zm) - 1, 0, nz - 2)

    hx_c = xn[ix + 1] - xn[ix]
    hy_c = yn[iy + 1] - yn[iy]
    hz_c = zn[iz + 1] - zn[iz]

    u = (xm - xn[ix]) / hx_c
    v = (ym - yn[iy]) / hy_c
    w = (zm - zn[iz]) / hz_c
    du = dx / hx_c
    dv = dy / hy_c
    dw = dz / hz_c

    for bj in (0, 1):
        wy = v if bj == 1 else (1.0 - v)
        s_y = 1.0 if bj == 1 else -1.0
        for bk in (0, 1):
            wz = w if bk == 1 else (1.0 - w)
            s_z = 1.0 if bk == 1 else -1.0
            weight = wy * wz + (s_y * s_z / 12.0) * dv * dw
            np.add.at(ix_cur, (ix, iy + bj, iz + bk), float(current_amp) * du * weight)

    for bi in (0, 1):
        wx = u if bi == 1 else (1.0 - u)
        s_x = 1.0 if bi == 1 else -1.0
        for bk in (0, 1):
            wz = w if bk == 1 else (1.0 - w)
            s_z = 1.0 if bk == 1 else -1.0
            weight = wx * wz + (s_x * s_z / 12.0) * du * dw
            np.add.at(iy_cur, (ix + bi, iy, iz + bk), float(current_amp) * dv * weight)

    for bi in (0, 1):
        wx = u if bi == 1 else (1.0 - u)
        s_x = 1.0 if bi == 1 else -1.0
        for bj in (0, 1):
            wy = v if bj == 1 else (1.0 - v)
            s_y = 1.0 if bj == 1 else -1.0
            weight = wx * wy + (s_x * s_y / 12.0) * du * dv
            np.add.at(iz_cur, (ix + bi, iy + bj, iz), float(current_amp) * dw * weight)

    tda = 1.0 / np.maximum(grid.itA.toarray(), 1e-30)
    j_flat = np.concatenate([
        ix_cur.ravel(order="F"),
        iy_cur.ravel(order="F"),
        iz_cur.ravel(order="F"),
    ]) / tda
    return j_flat


def compute_fit_divergence_of_j(solver: SolverFIT3D, j_flat: np.ndarray) -> dict[str, Any]:
    """Compute the discrete FIT dual divergence ``div_FIT J`` of a primary edge current ``j_flat``.

    In the Finite Integration Technique, primary electric currents ``J`` [A/m^2] are weighted
    by the dual face areas ``tilde_D_A = (itDa)^{-1}`` [m^2] to form dual face fluxes
    ``j_hat = tilde_D_A * J`` [A], and the discrete dual divergence operator onto dual cell
    volumes around primary nodes is ``tilde_S = -G^T = -[Px^T, Py^T, Pz^T]``.
    Dividing ``tilde_S j_hat`` by the local dual area scale gives ``div_FIT J`` in the same
    units as ``J`` so ``||div_FIT J||_2 / ||J||_2 == ||tilde_S j_hat||_2 / ||j_hat||_2``.
    """
    n_cells = solver.N
    j_arr = np.asarray(j_flat, dtype=float)
    tda = 1.0 / np.maximum(solver.grid.itA.toarray(), 1e-30)
    j_flux = tda * j_arr
    div_flux = -(
        solver.Px.T @ j_flux[:n_cells]
        + solver.Py.T @ j_flux[n_cells : 2 * n_cells]
        + solver.Pz.T @ j_flux[2 * n_cells :]
    )
    j_flux_norm = float(np.linalg.norm(j_flux))
    div_flux_norm = float(np.linalg.norm(div_flux))
    rel_div_norm = float(div_flux_norm / max(j_flux_norm, 1e-30))
    return {
        "div_flux_a": div_flux,
        "div_flux_norm_a": div_flux_norm,
        "j_flux_norm_a": j_flux_norm,
        "rel_div_norm": rel_div_norm,
    }


def compute_free_current_magnetic_dipole(grid: GridFIT3D, j_flat: np.ndarray) -> np.ndarray:
    """Compute the magnetic dipole moment ``m_free = 0.5 * int (r x J) dV`` [A m^2] of a primary edge current J."""
    nx, ny, nz = grid.Nx, grid.Ny, grid.Nz
    n_cells = nx * ny * nz
    l_arr = grid.L.toarray()
    ita_arr = grid.itA.toarray()
    dv_jx = (l_arr[:n_cells] / ita_arr[:n_cells]).reshape(nx, ny, nz, order="F")
    dv_jy = (l_arr[n_cells : 2 * n_cells] / ita_arr[n_cells : 2 * n_cells]).reshape(nx, ny, nz, order="F")
    dv_jz = (l_arr[2 * n_cells :] / ita_arr[2 * n_cells :]).reshape(nx, ny, nz, order="F")

    xc = 0.5 * (grid.x[:-1] + grid.x[1:])
    yc = 0.5 * (grid.y[:-1] + grid.y[1:])
    zc = 0.5 * (grid.z[:-1] + grid.z[1:])
    xn, yn, zn = grid.x[:-1], grid.y[:-1], grid.z[:-1]

    xx_jx, yy_jx, zz_jx = np.meshgrid(xc, yn, zn, indexing="ij")
    xx_jy, yy_jy, zz_jy = np.meshgrid(xn, yc, zn, indexing="ij")
    xx_jz, yy_jz, zz_jz = np.meshgrid(xn, yn, zc, indexing="ij")

    jx = j_flat[:n_cells].reshape(nx, ny, nz, order="F")
    jy = j_flat[n_cells : 2 * n_cells].reshape(nx, ny, nz, order="F")
    jz = j_flat[2 * n_cells :].reshape(nx, ny, nz, order="F")

    mx = 0.5 * float(np.sum(yy_jz * jz * dv_jz) - np.sum(zz_jy * jy * dv_jy))
    my = 0.5 * float(np.sum(zz_jx * jx * dv_jx) - np.sum(xx_jz * jz * dv_jz))
    mz = 0.5 * float(np.sum(xx_jy * jy * dv_jy) - np.sum(yy_jx * jx * dv_jx))
    return np.array([mx, my, mz], dtype=float)


def setup_toroid_core_and_source(
    solver: SolverFIT3D,
    core_mat: DispersiveMaterial | None,
    pulse: Any,
    *,
    r_major_m: float = R_MAJOR_M,
    r_minor_m: float = R_MINOR_M,
    n_windings: int = 8,
    active_windings: list[int] | None = None,
    winding_phases: list[float] | None = None,
    gap_angle_rad: float = 0.0,
    amp_h_drive: float = 1.0,
    continuous_sheet: bool = False,
    helical_parasitic_turn: bool = False,
    marinov_compensating_loop: bool = False,
    marinov_loop_radius_m: float | None = None,
    center_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[DivergenceFreeCurrentSource, dict[str, Any]]:
    """Assign a dispersive toroidal core on dual H-edges and build the poloidal winding current J."""
    g = solver.grid
    nx, ny, nz = g.Nx, g.Ny, g.Nz
    n_cells = nx * ny * nz

    x0_off, y0_off, z0_off = (float(center_offset_m[0]), float(center_offset_m[1]), float(center_offset_m[2]))
    xc = 0.5 * (g.x[:-1] + g.x[1:]) - x0_off
    yc = 0.5 * (g.y[:-1] + g.y[1:]) - y0_off
    zc = 0.5 * (g.z[:-1] + g.z[1:]) - z0_off
    xn, yn, zn = g.x[:-1] - x0_off, g.y[:-1] - y0_off, g.z[:-1] - z0_off

    def in_torus(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, r_min_local: float, gap_rad: float = 0.0) -> np.ndarray:
        rho = np.sqrt(xx**2 + yy**2)
        mask = (rho - r_major_m) ** 2 + zz**2 <= (r_min_local + 1e-12) ** 2
        if gap_rad > 0.0:
            phi = np.arctan2(yy, xx)
            mask = mask & (np.abs(phi) > 0.5 * gap_rad)
        return mask

    xx_hx, yy_hx, zz_hx = np.meshgrid(xn, yc, zc, indexing="ij")
    xx_hy, yy_hy, zz_hy = np.meshgrid(xc, yn, zc, indexing="ij")
    xx_hz, yy_hz, zz_hz = np.meshgrid(xc, yc, zn, indexing="ij")

    mx_core = in_torus(xx_hx, yy_hx, zz_hx, r_minor_m, gap_angle_rad)
    my_core = in_torus(xx_hy, yy_hy, zz_hy, r_minor_m, gap_angle_rad)
    mz_core = np.zeros_like(xx_hz, dtype=bool)

    if core_mat is not None:
        idx_all = np.concatenate([
            np.flatnonzero(mx_core.ravel(order="F")),
            n_cells + np.flatnonzero(my_core.ravel(order="F")),
        ])
        if len(idx_all) > 0:
            solver._add_dispersive_h_edges(core_mat, idx_all, np.ones(len(idx_all), dtype=float), mu_bg=1.0)

    if active_windings is None:
        active_windings = list(range(n_windings))
    if winding_phases is None:
        winding_phases = [1.0] * n_windings

    def sector_weight(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        if continuous_sheet:
            return np.ones_like(xx)
        phi = np.mod(np.arctan2(yy, xx), 2.0 * np.pi)
        w = np.zeros_like(xx)
        dphi_sector = 2.0 * np.pi / n_windings
        half_width = 0.35 * dphi_sector
        # Normalize by filling fraction so total Ampere-turns are preserved vs continuous sheet
        norm_fill = (0.5 * dphi_sector) / half_width
        for k in range(n_windings):
            if k not in active_windings:
                continue
            phi_k = (k + 0.5) * dphi_sector
            diff = np.abs(np.angle(np.exp(1j * (phi - phi_k))))
            w = np.where(diff <= half_width, float(winding_phases[k]) * norm_fill, w)
        return w

    rho_hx = np.maximum(np.sqrt(xx_hx**2 + yy_hx**2), 1e-12)
    rho_hy = np.maximum(np.sqrt(xx_hy**2 + yy_hy**2), 1e-12)

    m_wind_x = in_torus(xx_hx, yy_hx, zz_hx, r_minor_m, 0.0)
    m_wind_y = in_torus(xx_hy, yy_hy, zz_hy, r_minor_m, 0.0)

    hx_drive = np.where(
        m_wind_x,
        -amp_h_drive * (r_major_m / rho_hx) * (yy_hx / rho_hx) * sector_weight(xx_hx, yy_hx),
        0.0,
    )
    hy_drive = np.where(
        m_wind_y,
        amp_h_drive * (r_major_m / rho_hy) * (xx_hy / rho_hy) * sector_weight(xx_hy, yy_hy),
        0.0,
    )
    hz_drive = np.zeros_like(xx_hz)

    if helical_parasitic_turn:
        # Model a real single-layer 3D helical wire current path wound directly around the torus surface:
        #   rho(t) = R + r * cos(2*pi*N*t),  phi(t) = 2*pi*t,  z(t) = -r * sin(2*pi*N*t),  t in [0, 1],
        # discretised onto primary electric edges via cell-split Villasenor-Buneman weights (`deposit_closed_wire_on_edges`),
        # carrying turn current I_turn = 2*pi*R*amp_h_drive / N (so total poloidal Ampere-turns is N * I_turn = 2*pi*R*amp_h_drive)
        # with ZERO rescaling and no artificial H_z volume taper.
        # By Marinov et al. (2007) Eq. (9) and McDonald (2008), this 3D helical wire path has exact axial magnetic dipole moment
        #   m_{z,hel} = (1/2) * I_turn * int_0^{2*pi} rho(phi)^2 dphi = pi * I_turn * (R^2 + 0.5 * r^2) = pi * I_turn * R_1^2,
        # where R_1 = R * sqrt(1 + 0.5 * (r / R)^2).
        hx_drive = np.zeros_like(xx_hx)
        hy_drive = np.zeros_like(xx_hy)
        hz_drive = np.zeros_like(xx_hz)
        i_turn = (2.0 * np.pi * r_major_m * amp_h_drive) / float(max(n_windings, 1))
        t_hel = np.linspace(0.0, 1.0, 721)
        phi_hel = 2.0 * np.pi * t_hel
        rho_hel = r_major_m + r_minor_m * np.cos(2.0 * np.pi * float(n_windings) * t_hel)
        x_hel = x0_off + rho_hel * np.cos(phi_hel)
        y_hel = y0_off + rho_hel * np.sin(phi_hel)
        z_hel = z0_off - r_minor_m * np.sin(2.0 * np.pi * float(n_windings) * t_hel)
        j_flat = deposit_closed_wire_on_edges(g, x_hel, y_hel, z_hel, i_turn)

        if marinov_compensating_loop or (marinov_loop_radius_m is not None):
            # Separate planar circular return wire loop in the equatorial plane z = 0 of radius R_loop
            # (defaulting to Marinov Eq. 9: R_1 = R * sqrt(1 + 0.5 * (r / R)^2)) driven by the EXACT SAME
            # wire current -I_turn on primary electric edges with zero rescaling.
            r1_marinov = float(r_major_m * np.sqrt(1.0 + 0.5 * ((r_minor_m / r_major_m) ** 2)))
            r_loop = float(marinov_loop_radius_m) if marinov_loop_radius_m is not None else r1_marinov
            if r_loop > 0.0:
                x_loop = x0_off + r_loop * np.cos(phi_hel)
                y_loop = y0_off + r_loop * np.sin(phi_hel)
                z_loop = np.full_like(phi_hel, z0_off)
                j_flat = j_flat + deposit_closed_wire_on_edges(g, x_loop, y_loop, z_loop, -i_turn)
    else:
        h_drive_flat = np.concatenate([
            hx_drive.ravel(order="F"),
            hy_drive.ravel(order="F"),
            hz_drive.ravel(order="F"),
        ])
        j_flat = np.asarray(solver.itDa @ (solver.C.T @ (solver.tDs @ h_drive_flat)), dtype=float)
    src = DivergenceFreeCurrentSource(pulse, j_flat)
    return src, {
        "mx_core": mx_core,
        "my_core": my_core,
        "mz_core": mz_core,
        "hx_drive": hx_drive,
        "hy_drive": hy_drive,
        "hz_drive": hz_drive,
        "j_flat": j_flat,
    }


def setup_four_rods_square_and_source(
    solver: SolverFIT3D,
    core_mat: DispersiveMaterial | None,
    pulse: Any,
    *,
    r_sq_m: float = R_MAJOR_M,
    rod_half_width_m: float = R_MINOR_M,
    active_rods: list[int] | None = None,
    rod_phases: list[float] | None = None,
    amp_h_drive: float = 1.0,
) -> tuple[DivergenceFreeCurrentSource, dict[str, Any]]:
    """Assign 4 permeable rods arranged in a C4 square and build poloidal feed coils around each rod.

    The 4 rods form a closed square magnetization path in the xy plane:
      * Rod 0 (at ``x = +r_sq``): ``y in [-r_sq, +r_sq]``, magnetization ``+y``
      * Rod 1 (at ``y = +r_sq``): ``x in [-r_sq, +r_sq]``, magnetization ``-x``
      * Rod 2 (at ``x = -r_sq``): ``y in [-r_sq, +r_sq]``, magnetization ``-y``
      * Rod 3 (at ``y = -r_sq``): ``x in [-r_sq, +r_sq]``, magnetization ``+x``
    so ``r x M = +r_sq * M * z_hat`` on all 4 rods (counter-clockwise circulation).
    """
    g = solver.grid
    nx, ny, nz = g.Nx, g.Ny, g.Nz
    n_cells = nx * ny * nz

    xc = 0.5 * (g.x[:-1] + g.x[1:])
    yc = 0.5 * (g.y[:-1] + g.y[1:])
    zc = 0.5 * (g.z[:-1] + g.z[1:])
    xn, yn, zn = g.x[:-1], g.y[:-1], g.z[:-1]

    if active_rods is None:
        active_rods = [0, 1, 2, 3]
    if rod_phases is None:
        rod_phases = [1.0, 1.0, 1.0, 1.0]

    w = float(rod_half_width_m)
    L_half = float(r_sq_m + w)
    coil_half = float(max(0.45 * r_sq_m, 1.5 * w))

    def rod_masks(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, half_len: float) -> list[np.ndarray]:
        w_tol = w + 1e-12
        hl_tol = half_len + 1e-12
        z_ok = np.abs(zz) <= w_tol
        r0 = (np.abs(xx - r_sq_m) <= w_tol) & (np.abs(yy) <= hl_tol) & z_ok
        r1 = (np.abs(yy - r_sq_m) <= w_tol) & (np.abs(xx) <= hl_tol) & z_ok
        r2 = (np.abs(xx + r_sq_m) <= w_tol) & (np.abs(yy) <= hl_tol) & z_ok
        r3 = (np.abs(yy + r_sq_m) <= w_tol) & (np.abs(xx) <= hl_tol) & z_ok
        return [r0, r1, r2, r3]

    xx_hx, yy_hx, zz_hx = np.meshgrid(xn, yc, zc, indexing="ij")
    xx_hy, yy_hy, zz_hy = np.meshgrid(xc, yn, zc, indexing="ij")
    xx_hz, yy_hz, zz_hz = np.meshgrid(xc, yc, zn, indexing="ij")

    rods_hx = rod_masks(xx_hx, yy_hx, zz_hx, L_half)
    rods_hy = rod_masks(xx_hy, yy_hy, zz_hy, L_half)
    rods_hz = rod_masks(xx_hz, yy_hz, zz_hz, L_half)

    mx_core = np.zeros_like(xx_hx, dtype=bool)
    my_core = np.zeros_like(xx_hy, dtype=bool)
    mz_core = np.zeros_like(xx_hz, dtype=bool)
    for k in active_rods:
        mx_core |= rods_hx[k]
        my_core |= rods_hy[k]
        mz_core |= rods_hz[k]

    if core_mat is not None:
        idx_all = np.concatenate([
            np.flatnonzero(mx_core.ravel(order="F")),
            n_cells + np.flatnonzero(my_core.ravel(order="F")),
            2 * n_cells + np.flatnonzero(mz_core.ravel(order="F")),
        ])
        if len(idx_all) > 0:
            solver._add_dispersive_h_edges(core_mat, idx_all, np.ones(len(idx_all), dtype=float), mu_bg=1.0)

    coils_hx = rod_masks(xx_hx, yy_hx, zz_hx, coil_half)
    coils_hy = rod_masks(xx_hy, yy_hy, zz_hy, coil_half)

    hx_drive = np.zeros_like(xx_hx)
    hy_drive = np.zeros_like(xx_hy)
    hz_drive = np.zeros_like(xx_hz)

    if 0 in active_rods:
        hy_drive = np.where(coils_hy[0], +float(rod_phases[0]) * amp_h_drive, hy_drive)
    if 1 in active_rods:
        hx_drive = np.where(coils_hx[1], -float(rod_phases[1]) * amp_h_drive, hx_drive)
    if 2 in active_rods:
        hy_drive = np.where(coils_hy[2], -float(rod_phases[2]) * amp_h_drive, hy_drive)
    if 3 in active_rods:
        hx_drive = np.where(coils_hx[3], +float(rod_phases[3]) * amp_h_drive, hx_drive)

    h_drive_flat = np.concatenate([
        hx_drive.ravel(order="F"),
        hy_drive.ravel(order="F"),
        hz_drive.ravel(order="F"),
    ])
    j_flat = np.asarray(solver.itDa @ (solver.C.T @ (solver.tDs @ h_drive_flat)), dtype=float)
    src = DivergenceFreeCurrentSource(pulse, j_flat)
    return src, {
        "mx_core": mx_core,
        "my_core": my_core,
        "mz_core": mz_core,
        "hx_drive": hx_drive,
        "hy_drive": hy_drive,
        "hz_drive": hz_drive,
        "j_flat": j_flat,
    }


def run_single_fit_case(
    *,
    geometry: str = "toroid",
    mu_dc: float = MU_DC_DEFAULT,
    f_res_hz: float = F_RES_CORE_HZ,
    gamma_hz: float = GAMMA_CORE_HZ,
    eps_core: float = 1.0,
    freq_hz: float = F0_HZ,
    fwidth_hz: float = 0.22e9,
    r_major_m: float = R_MAJOR_M,
    r_minor_m: float = R_MINOR_M,
    n_windings: int = 8,
    active_windings: list[int] | None = None,
    winding_phases: list[float] | None = None,
    gap_angle_rad: float = 0.0,
    continuous_sheet: bool = False,
    helical_parasitic_turn: bool = False,
    marinov_compensating_loop: bool = False,
    marinov_loop_radius_m: float | None = None,
    dx_core_m: float = 0.0025,
    graded: bool = True,
    n_cycles: float = 4.5,
    record_2d_cuts: bool = False,
    freqs_extra: np.ndarray | None = None,
    center_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    grid_asymmetry_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Run a single 3D FIT simulation (preflighted with wakis.estimate) and extract multipoles & energy balance."""
    core_xy_half = max(0.0225, r_major_m + r_minor_m + 2.0 * dx_core_m)
    core_z_half = max(0.0075, r_minor_m + dx_core_m)
    n2f_xy_half = max(0.055, core_xy_half + 0.025)
    n2f_z_half = max(0.045, core_z_half + 0.030)
    dom_xy_half = max(0.150, n2f_xy_half + 0.095)
    dom_z_half = max(0.125, n2f_z_half + 0.080)

    grid = build_toroid_grid(
        dx_core_m=dx_core_m,
        dx_vac_m=0.015,
        core_xy_half_m=core_xy_half,
        core_z_half_m=core_z_half,
        n2f_xy_half_m=n2f_xy_half,
        n2f_z_half_m=n2f_z_half,
        domain_xy_half_m=dom_xy_half,
        domain_z_half_m=dom_z_half,
        graded=graded,
        grid_asymmetry_m=grid_asymmetry_m,
    )
    n_pml = 6
    solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, verbose=0)
    solver._curl_dual_op = solver.itDa @ (solver.C.T @ solver.tDs)

    core_mat = (
        make_core_material(mu_dc=mu_dc, f_res_hz=f_res_hz, gamma_hz=gamma_hz, eps_inf=eps_core)
        if mu_dc > 1.0
        else None
    )
    if eps_core > 1.0:
        # Set dielectric permittivity inside the torus core on primary E-edges
        g_e = solver.grid
        xc_e = 0.5 * (g_e.x[:-1] + g_e.x[1:])
        yc_e = 0.5 * (g_e.y[:-1] + g_e.y[1:])
        zc_e = 0.5 * (g_e.z[:-1] + g_e.z[1:])
        xn_e, yn_e, zn_e = g_e.x[:-1], g_e.y[:-1], g_e.z[:-1]
        for comp_e, (ax1, ax2, ax3) in [
            ("x", (xc_e, yn_e, zn_e)),
            ("y", (xn_e, yc_e, zn_e)),
            ("z", (xn_e, yn_e, zc_e)),
        ]:
            xx_e, yy_e, zz_e = np.meshgrid(ax1, ax2, ax3, indexing="ij")
            rho_e = np.sqrt(xx_e**2 + yy_e**2)
            mask_e = (rho_e - r_major_m) ** 2 + zz_e**2 <= r_minor_m**2
            arr_ieps = np.asarray(solver.ieps[:, :, :, comp_e]).reshape((g_e.Nx, g_e.Ny, g_e.Nz), order="F")
            arr_ieps[mask_e] = 1.0 / (EPS0 * float(eps_core))
            solver.ieps[:, :, :, comp_e] = arr_ieps

    pulse = GaussianSource(frequency=freq_hz, fwidth=fwidth_hz)

    if geometry == "toroid":
        src, info = setup_toroid_core_and_source(
            solver,
            core_mat,
            pulse,
            r_major_m=r_major_m,
            r_minor_m=r_minor_m,
            n_windings=n_windings,
            active_windings=active_windings,
            winding_phases=winding_phases,
            gap_angle_rad=gap_angle_rad,
            continuous_sheet=continuous_sheet,
            helical_parasitic_turn=helical_parasitic_turn,
            marinov_compensating_loop=marinov_compensating_loop,
            marinov_loop_radius_m=marinov_loop_radius_m,
            center_offset_m=center_offset_m,
        )
    elif geometry == "four_rods":
        src, info = setup_four_rods_square_and_source(
            solver,
            core_mat,
            pulse,
            r_sq_m=r_major_m,
            rod_half_width_m=r_minor_m,
            active_rods=active_windings,
            rod_phases=winding_phases,
        )
    elif geometry == "magnetic_ring":
        # Ideal equivalence-principle azimuthal MagneticCurrentSource ring of radius r_major_m
        g = solver.grid
        xc = 0.5 * (g.x[:-1] + g.x[1:])
        yc = 0.5 * (g.y[:-1] + g.y[1:])
        zc = 0.5 * (g.z[:-1] + g.z[1:])
        xn, yn, zn = g.x[:-1], g.y[:-1], g.z[:-1]
        xx_hx, yy_hx, zz_hx = np.meshgrid(xn, yc, zc, indexing="ij")
        xx_hy, yy_hy, zz_hy = np.meshgrid(xc, yn, zc, indexing="ij")
        xx_hz, yy_hz, zz_hz = np.meshgrid(xc, yc, zn, indexing="ij")
        rho_hx = np.maximum(np.sqrt(xx_hx**2 + yy_hx**2), 1e-12)
        rho_hy = np.maximum(np.sqrt(xx_hy**2 + yy_hy**2), 1e-12)
        m_mask_x = (rho_hx - r_major_m) ** 2 + zz_hx**2 <= r_minor_m**2
        m_mask_y = (rho_hy - r_major_m) ** 2 + zz_hy**2 <= r_minor_m**2
        # Magnetic current density J_m = omega_0 * mu_0 * M_phi [V/m^2]
        jm_amp = OMEGA0_RAD_S * MU0 * max(mu_dc - 1.0, 1.0)
        mx_drive = np.where(m_mask_x, -jm_amp * (r_major_m / rho_hx) * (yy_hx / rho_hx), 0.0)
        my_drive = np.where(m_mask_y, +jm_amp * (r_major_m / rho_hy) * (xx_hy / rho_hy), 0.0)
        mz_drive = np.zeros_like(xx_hz)
        m_flat = np.concatenate([mx_drive.ravel(order="F"), my_drive.ravel(order="F"), mz_drive.ravel(order="F")])
        src = AzimuthalMagneticRingSource(pulse, m_flat)
        info = {
            "mx_core": m_mask_x,
            "my_core": m_mask_y,
            "mz_core": np.zeros_like(xx_hz, dtype=bool),
            "hx_drive": mx_drive / (OMEGA0_RAD_S * MU0),
            "hy_drive": my_drive / (OMEGA0_RAD_S * MU0),
            "hz_drive": mz_drive,
            "j_flat": np.zeros(3 * grid.Nx * grid.Ny * grid.Nz, dtype=float),
        }
    elif geometry == "electric_dipole":
        src = CurrentSource(pulse, component="z", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 2.0 * r_minor_m), amplitude=1.0)
        info = {
            "mx_core": np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool),
            "my_core": np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool),
            "mz_core": np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool),
            "j_flat": np.zeros(3 * grid.Nx * grid.Ny * grid.Nz, dtype=float),
        }
    elif geometry == "magnetic_loop":
        # Single circular electric wire loop of radius r_major_m in the xy plane (magnetic dipole m_z).
        # Use 5x5 sub-cell quadrature over the disk rho <= r_major_m to eliminate O(Delta/a) Gauss-circle
        # staircase area jumps on coarse grids while preserving exact discrete curl(H_drive) divergence-freedom.
        g = solver.grid
        xc = 0.5 * (g.x[:-1] + g.x[1:])
        yc = 0.5 * (g.y[:-1] + g.y[1:])
        zc = 0.5 * (g.z[:-1] + g.z[1:])
        xn, yn, zn = g.x[:-1], g.y[:-1], g.z[:-1]
        dx_c = g.x[1:] - g.x[:-1]
        dy_c = g.y[1:] - g.y[:-1]
        xx_hx, yy_hx, zz_hx = np.meshgrid(xn, yc, zc, indexing="ij")
        xx_hy, yy_hy, zz_hy = np.meshgrid(xc, yn, zc, indexing="ij")
        xx_hz, yy_hz, zz_hz = np.meshgrid(xc, yc, zn, indexing="ij")
        rho_hz = np.sqrt(xx_hz**2 + yy_hz**2)
        mx_core = (np.sqrt(xx_hx**2 + yy_hx**2) <= r_major_m) & (np.abs(zz_hx) <= r_minor_m)
        my_core = (np.sqrt(xx_hy**2 + yy_hy**2) <= r_major_m) & (np.abs(zz_hy) <= r_minor_m)
        mz_core = (rho_hz <= r_major_m) & (np.abs(zz_hz) <= r_minor_m)
        if core_mat is not None:
            n_cells = g.Nx * g.Ny * g.Nz
            idx_all = np.concatenate([
                np.flatnonzero(mx_core.ravel(order="F")),
                n_cells + np.flatnonzero(my_core.ravel(order="F")),
                2 * n_cells + np.flatnonzero(mz_core.ravel(order="F")),
            ])
            solver._add_dispersive_h_edges(core_mat, idx_all, np.ones(len(idx_all), dtype=float), mu_bg=1.0)
        w_disk_2d = np.zeros((g.Nx, g.Ny), dtype=float)
        sub_pts = np.linspace(-0.4, 0.4, 5)
        for ox in sub_pts:
            for oy in sub_pts:
                xx_sub, yy_sub = np.meshgrid(xc + ox * dx_c, yc + oy * dy_c, indexing="ij")
                w_disk_2d += (xx_sub**2 + yy_sub**2 <= r_major_m**2).astype(float)
        w_disk_2d /= float(len(sub_pts) ** 2)
        hz_drive = np.where(np.abs(zz_hz) <= dx_core_m, w_disk_2d[:, :, None] / (2.0 * dx_core_m), 0.0)
        h_drive_flat = np.concatenate([
            np.zeros(2 * g.Nx * g.Ny * g.Nz, dtype=float),
            hz_drive.ravel(order="F"),
        ])
        j_flat = np.asarray(solver.itDa @ (solver.C.T @ (solver.tDs @ h_drive_flat)), dtype=float)
        src = DivergenceFreeCurrentSource(pulse, j_flat)
        info = {
            "mx_core": mx_core,
            "my_core": my_core,
            "mz_core": mz_core,
            "j_flat": j_flat,
        }
    elif geometry == "lumped_wire_loop":
        # Thin PEC wire square loop in the z=0 plane driven by a 50-Ohm LumpedPort (wakis.lumped)
        g = solver.grid
        xn, yn, zn = g.x[:-1], g.y[:-1], g.z[:-1]
        iz0 = int(np.argmin(np.abs(zn)))
        ix_lo = int(np.argmin(np.abs(xn + r_major_m)))
        ix_hi = int(np.argmin(np.abs(xn - r_major_m)))
        iy_lo = int(np.argmin(np.abs(yn + r_major_m)))
        iy_hi = int(np.argmin(np.abs(yn - r_major_m)))
        ix_mid = int(np.argmin(np.abs(xn)))
        # Set PEC (ieps = 0) along the 4 sides of the square wire loop except 2 gap edges at (ix_mid-1, ix_mid) on y_lo
        ieps_x = np.asarray(solver.ieps[:, :, :, "x"]).reshape((g.Nx, g.Ny, g.Nz), order="F")
        ieps_y = np.asarray(solver.ieps[:, :, :, "y"]).reshape((g.Nx, g.Ny, g.Nz), order="F")
        ieps_x[ix_lo:ix_hi, iy_hi, iz0] = 0.0
        for ix_w in range(ix_lo, ix_hi):
            if ix_w not in (ix_mid - 1, ix_mid):
                ieps_x[ix_w, iy_lo, iz0] = 0.0
        ieps_y[ix_lo, iy_lo:iy_hi, iz0] = 0.0
        ieps_y[ix_hi, iy_lo:iy_hi, iz0] = 0.0
        solver.ieps[:, :, :, "x"] = ieps_x
        solver.ieps[:, :, :, "y"] = ieps_y
        gap_w = float(xn[ix_mid + 1] - xn[ix_mid - 1])
        src = LumpedPort(
            pulse,
            component="x",
            center=(0.0, float(yn[iy_lo]), float(zn[iz0])),
            size=(0.8 * gap_w, 0.0, 0.0),
            impedance=50.0,
            amplitude=1.0,
        )
        info = {
            "mx_core": np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool),
            "my_core": np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool),
            "mz_core": np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool),
            "j_flat": np.zeros(3 * grid.Nx * grid.Ny * grid.Nz, dtype=float),
        }
    else:
        raise ValueError(f"Unknown geometry={geometry!r}")

    t_end = pulse.t0 + float(n_cycles) / (2.0 * np.pi * fwidth_hz) + 0.16 / C0
    n_steps = int(np.ceil(t_end / solver.dt))

    # Preflight every FIT run as required
    est = estimate(
        cells=(grid.Nx, grid.Ny, grid.Nz),
        steps=n_steps,
        backend="cpu",
        cpml=True,
        n_poles=1 if core_mat is not None else 0,
    )
    preflight(est, {"ram": DEFAULT_MEM_LIMIT_MB * 1e6, "gpu": 0}, raise_error=True)

    mon_freqs = [freq_hz] if freqs_extra is None else list(np.atleast_1d(freqs_extra))
    n2f = Near2FarMonitor(
        frequencies=mon_freqs,
        center=(0.0, 0.0, 0.0),
        size=(2.0 * n2f_xy_half, 2.0 * n2f_xy_half, 2.0 * n2f_z_half),
    )
    dft_core = DFTMonitor(
        frequencies=mon_freqs,
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        center=(0.0, 0.0, 0.0),
        size=(2.0 * core_xy_half, 2.0 * core_xy_half, 2.0 * core_z_half),
    )
    monitors_list = [n2f, dft_core]

    if record_2d_cuts:
        dft_xz = DFTMonitor(
            frequencies=[freq_hz],
            components=("Ex", "Ez", "Hy"),
            center=(0.0, 0.0, 0.0),
            size=(2.0 * n2f_xy_half, 0.0, 2.0 * n2f_z_half),
        )
        dft_xy = DFTMonitor(
            frequencies=[freq_hz],
            components=("Ex", "Ey", "Hx", "Hy", "Ez"),
            center=(0.0, 0.0, 0.0),
            size=(2.0 * n2f_xy_half, 2.0 * n2f_xy_half, 0.0),
        )
        monitors_list.extend([dft_xz, dft_xy])

    t0_wall = time.perf_counter()
    for step in range(n_steps):
        t = step * solver.dt
        src.update(solver, t)
        solver.one_step()
        if hasattr(src, "post_step"):
            src.post_step(solver, t)
        for m in monitors_list:
            m.update(solver, t)
    wall_s = float(time.perf_counter() - t0_wall)

    s_f0 = complex(pulse.fourier_transform(np.array([freq_hz]))[0])
    w0 = 2.0 * np.pi * freq_hz
    k0 = w0 / C0

    # 1. Far-field spherical multipole decomposition (l <= 5)
    theta_1d, phi_1d, w_t, w_p, tt_flat, pp_flat = make_quadrature_sphere(24, 48)
    r_ff = 10.0 * (C0 / freq_hz)
    ff = n2f.get_farfield(tt_flat, pp_flat, r=r_ff)
    idx_f0 = int(np.argmin(np.abs(np.asarray(mon_freqs) - freq_hz)))
    vsh = spherical_multipole_from_farfield(
        theta_1d,
        phi_1d,
        w_t,
        w_p,
        ff["Etheta"][idx_f0],
        ff["Ephi"][idx_f0],
        l_max=5,
        r_ff=r_ff,
        k0=k0,
        s_norm=s_f0,
    )

    p_rad_n2f = 0.5 * float(n2f.radiated_power()[idx_f0]) / (abs(s_f0) ** 2)
    p_rad_flux = 0.5 * float(n2f.get_flux()[idx_f0]) / (abs(s_f0) ** 2)

    # Polar cut at phi = 0 for radiation pattern plots
    theta_plot = np.linspace(0.0, np.pi, 91)
    ff_cut = n2f.get_farfield(theta_plot, np.zeros_like(theta_plot), r=r_ff)
    et_cut = ff_cut["Etheta"][idx_f0] / s_f0
    ep_cut = ff_cut["Ephi"][idx_f0] / s_f0

    # 2. Extract full-grid H and E on core region for moments & energy balance
    nx, ny, nz = grid.Nx, grid.Ny, grid.Nz
    n_cells = nx * ny * nz
    hx_full = np.zeros((nx, ny, nz), dtype=np.complex128)
    hy_full = np.zeros((nx, ny, nz), dtype=np.complex128)
    hz_full = np.zeros((nx, ny, nz), dtype=np.complex128)
    ex_full = np.zeros((nx, ny, nz), dtype=np.complex128)
    ey_full = np.zeros((nx, ny, nz), dtype=np.complex128)
    ez_full = np.zeros((nx, ny, nz), dtype=np.complex128)

    for comp, arr_full in [
        ("Hx", hx_full),
        ("Hy", hy_full),
        ("Hz", hz_full),
        ("Ex", ex_full),
        ("Ey", ey_full),
        ("Ez", ez_full),
    ]:
        sub, _ = dft_core.get_field(comp)
        ind, _ = grid.region_indices(comp[0], comp[1].lower(), dft_core.center, dft_core.size)
        arr_full[np.ix_(*ind)] = sub[idx_f0] / s_f0

    mu_disc = complex(core_mat.mu_discrete(freq_hz, solver.dt)) if core_mat is not None else (1.0 + 0.0j)
    chi_m = mu_disc - 1.0
    moments = compute_dubovik_tugushev_and_magnetization_moments(
        grid,
        solver,
        hx_full,
        hy_full,
        hz_full,
        info["mx_core"],
        info["my_core"],
        info["mz_core"],
        chi_m,
        info["j_flat"],
        freq_hz=freq_hz,
    )

    # Energy balance: P_in vs P_rad + P_loss_core
    tl_arr, ia_arr = grid.tL.toarray(), grid.iA.toarray()
    l_arr, ita_arr = grid.L.toarray(), grid.itA.toarray()
    dv_h = tl_arr / ia_arr
    dv_e = l_arr / ita_arr
    dv_hx = dv_h[:n_cells].reshape(nx, ny, nz, order="F")
    dv_hy = dv_h[n_cells : 2 * n_cells].reshape(nx, ny, nz, order="F")
    dv_hz = dv_h[2 * n_cells :].reshape(nx, ny, nz, order="F")
    dv_ex = dv_e[:n_cells].reshape(nx, ny, nz, order="F")
    dv_ey = dv_e[n_cells : 2 * n_cells].reshape(nx, ny, nz, order="F")
    dv_ez = dv_e[2 * n_cells :].reshape(nx, ny, nz, order="F")

    # Discrete static integral T_z^{disc,drive} = 0.5 * Re(mu_disc) * int (r x H_drive)_z dV on the same staircase
    if "hx_drive" in info and "hy_drive" in info:
        xc_g = 0.5 * (grid.x[:-1] + grid.x[1:])
        yc_g = 0.5 * (grid.y[:-1] + grid.y[1:])
        zc_g = 0.5 * (grid.z[:-1] + grid.z[1:])
        xn_g, yn_g = grid.x[:-1], grid.y[:-1]
        xx_hx_g, yy_hx_g, _ = np.meshgrid(xn_g, yc_g, zc_g, indexing="ij")
        xx_hy_g, yy_hy_g, _ = np.meshgrid(xc_g, yn_g, zc_g, indexing="ij")
        tz_disc_drive = 0.5 * float(np.real(mu_disc)) * float(
            np.sum(xx_hy_g * info["hy_drive"] * dv_hy) - np.sum(yy_hx_g * info["hx_drive"] * dv_hx)
        )
        moments["tz_disc_static_drive_am3"] = tz_disc_drive
        moments["tz_sim_over_disc_residual_ratio"] = float(abs(moments["tz_tot_am3"]) / max(abs(tz_disc_drive), 1e-30))
        kc_r_local = float(k0 * np.sqrt(max(eps_core * float(np.real(mu_disc)), 1.0)) * r_minor_m)
        moments["analytic_interior_factor_f0"] = float(1.0 + 0.125 * (kc_r_local**2))
        xx_ez_g, yy_ez_g, zz_ez_g = np.meshgrid(xn_g, yn_g, zc_g, indexing="ij")
        ez_core_mask = (np.hypot(xx_ez_g, yy_ez_g) - r_major_m) ** 2 + zz_ez_g**2 <= r_minor_m**2
        moments["p_pol_z_cm"] = complex(EPS0 * (eps_core - 1.0) * np.sum(ez_full * ez_core_mask * dv_ez))

    w_tilde = 2.0 / solver.dt * np.sin(0.5 * w0 * solver.dt)
    mu_pp = max(-float(np.imag(mu_disc)), 0.0)
    mu_p = float(np.real(mu_disc))
    h2_core_int = float(
        np.sum((np.abs(hx_full) ** 2) * info["mx_core"] * dv_hx)
        + np.sum((np.abs(hy_full) ** 2) * info["my_core"] * dv_hy)
        + np.sum((np.abs(hz_full) ** 2) * info["mz_core"] * dv_hz)
    )
    p_loss_core = 0.5 * w_tilde * MU0 * mu_pp * h2_core_int
    w_stored_mag_core = 0.25 * MU0 * mu_p * h2_core_int

    jx_f = info["j_flat"][:n_cells].reshape(nx, ny, nz, order="F")
    jy_f = info["j_flat"][n_cells : 2 * n_cells].reshape(nx, ny, nz, order="F")
    jz_f = info["j_flat"][2 * n_cells :].reshape(nx, ny, nz, order="F")
    cos_half = float(np.cos(0.5 * w0 * solver.dt))
    p_in = -0.5 * cos_half * float(
        np.sum(np.real(ex_full) * jx_f * dv_ex)
        + np.sum(np.real(ey_full) * jy_f * dv_ey)
        + np.sum(np.real(ez_full) * jz_f * dv_ez)
    )
    # Also compute mutated P_in with a half-step phase error exp(+i*w0*dt/2) for mutation tests
    exp_mut = np.exp(0.5j * w0 * solver.dt) * cos_half
    p_in_mutated_shift = -0.5 * float(
        np.sum(np.real(ex_full * exp_mut) * jx_f * dv_ex)
        + np.sum(np.real(ey_full * exp_mut) * jy_f * dv_ey)
        + np.sum(np.real(ez_full * exp_mut) * jz_f * dv_ez)
    )

    th_rad = toroidal_dipole_radiation_theory(freq_hz, moments["tz_tot_am3"], r_major_m=r_major_m)

    out: dict[str, Any] = {
        "geometry": geometry,
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "n_cells": int(n_cells),
        "n_steps": int(n_steps),
        "dt_s": float(solver.dt),
        "wall_s": wall_s,
        "est_ram_mb": float(est.host_bytes / 1e6),
        "est_time_s": float(est.seconds),
        "dx_min_mm": float(np.min(grid.dx) * 1e3),
        "dx_max_mm": float(np.max(grid.dx) * 1e3),
        "mu_disc": mu_disc,
        "vsh": vsh,
        "moments": moments,
        "p_rad_n2f_w": p_rad_n2f,
        "p_rad_flux_w": p_rad_flux,
        "p_rad_theory_point_w": float(th_rad["p_rad_point_w"]),
        "p_rad_theory_j1_w": float(th_rad["p_rad_j1_w"]),
        "p_eff_j1_cm": complex(th_rad["p_eff_j1_cm"]),
        "p_loss_core_w": p_loss_core,
        "p_in_w": p_in,
        "p_in_mutated_shift_w": p_in_mutated_shift,
        "energy_closure_rel_err": float(abs(p_in - (p_rad_flux + p_loss_core)) / max(abs(p_in), 1e-30)),
        "energy_closure_mutated_rel_err": float(
            abs(p_in_mutated_shift - (p_rad_flux + p_loss_core)) / max(abs(p_in_mutated_shift), 1e-30)
        ),
        "eta_rad_sim": float(p_rad_flux / max(p_rad_flux + p_loss_core, 1e-30)),
        "w_stored_mag_core_j": w_stored_mag_core,
        "theta_plot_rad": theta_plot,
        "etheta_cut": et_cut,
        "ephi_cut": ep_cut,
    }

    if freqs_extra is not None:
        s_band = pulse.fourier_transform(np.asarray(mon_freqs))
        p_rad_band = 0.5 * np.asarray(n2f.radiated_power(), dtype=float) / (np.abs(s_band) ** 2)
        sub_hx, _ = dft_core.get_field("Hx")
        sub_hy, _ = dft_core.get_field("Hy")
        sub_hz, _ = dft_core.get_field("Hz")
        ind_hx, _ = grid.region_indices("H", "x", dft_core.center, dft_core.size)
        ind_hy, _ = grid.region_indices("H", "y", dft_core.center, dft_core.size)
        ind_hz, _ = grid.region_indices("H", "z", dft_core.center, dft_core.size)
        tz_tot_band = []
        p_rad_j1_band = []
        for k_f, f_val in enumerate(mon_freqs):
            hx_k = np.zeros((nx, ny, nz), dtype=np.complex128)
            hy_k = np.zeros((nx, ny, nz), dtype=np.complex128)
            hz_k = np.zeros((nx, ny, nz), dtype=np.complex128)
            hx_k[np.ix_(*ind_hx)] = sub_hx[k_f] / s_band[k_f]
            hy_k[np.ix_(*ind_hy)] = sub_hy[k_f] / s_band[k_f]
            hz_k[np.ix_(*ind_hz)] = sub_hz[k_f] / s_band[k_f]
            mu_k = complex(core_mat.mu_discrete(float(f_val), solver.dt)) if core_mat is not None else (1.0 + 0.0j)
            mom_k = compute_dubovik_tugushev_and_magnetization_moments(
                grid,
                solver,
                hx_k,
                hy_k,
                hz_k,
                info["mx_core"],
                info["my_core"],
                info["mz_core"],
                mu_k - 1.0,
                info["j_flat"],
                freq_hz=float(f_val),
            )
            tz_tot_band.append(complex(mom_k["tz_tot_am3"]))
            th_k = toroidal_dipole_radiation_theory(float(f_val), mom_k["tz_tot_am3"], r_major_m=r_major_m)
            p_rad_j1_band.append(float(th_k["p_rad_j1_w"]))
        out["freqs_band_hz"] = np.asarray(mon_freqs, dtype=float)
        out["p_rad_band_w"] = p_rad_band
        out["tz_tot_band_am3"] = np.asarray(tz_tot_band, dtype=np.complex128)
        out["p_rad_theory_j1_band_w"] = np.asarray(p_rad_j1_band, dtype=float)

    if record_2d_cuts:
        ex_xz, c_ex_xz = dft_xz.get_field("Ex")
        ez_xz, c_ez_xz = dft_xz.get_field("Ez")
        hy_xz, c_hy_xz = dft_xz.get_field("Hy")
        ex_xy, c_ex_xy = dft_xy.get_field("Ex")
        ey_xy, c_ey_xy = dft_xy.get_field("Ey")
        ez_xy, c_ez_xy = dft_xy.get_field("Ez")
        hx_xy, c_hx_xy = dft_xy.get_field("Hx")
        hy_xy, c_hy_xy = dft_xy.get_field("Hy")
        ind_hx_xy, _ = grid.region_indices("H", "x", dft_xy.center, dft_xy.size)
        ind_hy_xy, _ = grid.region_indices("H", "y", dft_xy.center, dft_xy.size)
        nx_xz = min(ex_xz.shape[1], ez_xz.shape[1], hy_xz.shape[1], len(c_ez_xz[0]), len(c_hy_xz[0]))
        nz_xz = min(ex_xz.shape[3], ez_xz.shape[3], hy_xz.shape[3], len(c_ez_xz[2]), len(c_hy_xz[2]))
        out["cuts"] = {
            "x_xz_m": np.asarray(c_hy_xz[0][:nx_xz], dtype=float),
            "z_xz_m": np.asarray(c_hy_xz[2][:nz_xz], dtype=float),
            "ex_xz": np.asarray(ex_xz[0, :nx_xz, 0, :nz_xz] / s_f0, dtype=np.complex128),
            "ez_xz": np.asarray(ez_xz[0, :nx_xz, 0, :nz_xz] / s_f0, dtype=np.complex128),
            "hy_xz": np.asarray(hy_xz[0, :nx_xz, 0, :nz_xz] / s_f0, dtype=np.complex128),
            "x_xy_m": np.asarray(c_hx_xy[0], dtype=float),
            "y_xy_m": np.asarray(c_hx_xy[1], dtype=float),
            "hx_xy": np.asarray(hx_xy[0, :, :, 0] / s_f0, dtype=np.complex128),
            "hy_xy": np.asarray(hy_xy[0, :, :, 0] / s_f0, dtype=np.complex128),
            "ez_xy": np.asarray(ez_xy[0, :, :, 0] / s_f0, dtype=np.complex128),
            "mx_mask_xy": np.asarray(info["mx_core"][np.ix_(*ind_hx_xy)][:, :, 0], dtype=bool),
            "my_mask_xy": np.asarray(info["my_core"][np.ix_(*ind_hy_xy)][:, :, 0], dtype=bool),
        }

    if isinstance(src, LumpedPort):
        z_in = complex(src.impedance_in([freq_hz])[0])
        i_port = complex(src.port_current([freq_hz])[0] / s_f0)
        out["z_in_ohm"] = z_in
        out["i_port_a"] = i_port

    return out


# ==============================================================================
# 4. Study task runners (FIT runs executed strictly one at a time)
# ==============================================================================


def list_study_tasks() -> list[dict[str, Any]]:
    """Return metadata for the 6 sequential 3D FIT study tasks."""
    return [
        {
            "id": 1,
            "name": "01_baseline_dipoles",
            "description": "Single magnetic loop (m_z), lumped-port PEC wire loop, and short electric dipole (p_z); (ka)^4 scaling",
        },
        {
            "id": 2,
            "name": "02_closed_toroid_multipoles",
            "description": "Closed dispersive magnetic toroid: Dubovik-Tugushev identity, l<=5 multipoles, P_rad vs p_eff, omega^6 and R/lambda sweeps",
        },
        {
            "id": 3,
            "name": "03_square_four_rods",
            "description": "C4 square of four permeable rods: m=0 guarantee and C4 residual scaling vs electrical size",
        },
        {
            "id": 4,
            "name": "04_break_the_loop",
            "description": "Symmetry/loop-breaking tests: core air gap, 7/8 feeds, Mode 1 antiphase, 3/4 rods, helical turn + Marinov compensating loop",
        },
        {
            "id": 5,
            "name": "05_equivalence_limits",
            "description": "Where equivalence ends: electrical size, core TE01-like trapped mode, finite N windings, and near-field maps vs Hertzian dipole",
        },
        {
            "id": 6,
            "name": "06_efficiency_q_and_convergence",
            "description": "Core loss tangent sweep (energy bookkeeping P_in = P_rad + P_loss), Q vs Chu bound, and 3-level mesh convergence + Richardson",
        },
    ]


def run_task_01_baseline_dipoles(quick: bool = True, out_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Run Task 1: Baseline electric dipole (p_z), magnetic loop (m_z), LumpedPort wire loop, and (ka)^4 scaling."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dx_core = 0.0025 if quick else 0.0020

    # 1. Short axial electric dipole (CurrentSource along z)
    res_edip = run_single_fit_case(geometry="electric_dipole", mu_dc=1.0, r_minor_m=0.005, dx_core_m=dx_core)

    # 2. Small magnetic loop in xy plane (magnetic dipole m_z) at 3 radii a in {10.0, 12.5, 15.0} mm for (ka)^4 scaling
    radii_m = [0.010, 0.0125, 0.015]
    loop_runs = []
    for a_m in radii_m:
        r_loop = run_single_fit_case(
            geometry="magnetic_loop",
            mu_dc=1.0,
            r_major_m=a_m,
            r_minor_m=0.005,
            dx_core_m=dx_core,
        )
        loop_runs.append(r_loop)

    # 3. Small magnetic loop on a permeable core (mu_DC = 20, a = 12.5 mm)
    res_loop_core = run_single_fit_case(
        geometry="magnetic_loop",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=0.0125,
        r_minor_m=0.005,
        dx_core_m=dx_core,
    )

    # 4. Thin PEC wire loop driven by a 50-Ohm LumpedPort (wakis.lumped)
    res_lumped_loop = run_single_fit_case(
        geometry="lumped_wire_loop",
        mu_dc=1.0,
        r_major_m=0.0125,
        r_minor_m=0.005,
        dx_core_m=dx_core,
    )

    p_rad_loops = [float(r["p_rad_n2f_w"]) for r in loop_runs]
    ka_vals = [float(K0_RAD_M * a_m) for a_m in radii_m]
    r_rad_exact = [float(small_loop_radiation_resistance(F0_HZ, a)) for a in radii_m]
    # Fit log-log slope of P_rad vs (ka) for small loop (expected slope = 4 because m = I * pi * a^2 -> P_rad ~ m^2 ~ a^4)
    slope_ka4 = float(np.polyfit(np.log(ka_vals), np.log(p_rad_loops), 1)[0])
    ka4_ratio_sim = float(p_rad_loops[-1] / p_rad_loops[0])
    ka4_ratio_th = float((radii_m[-1] / radii_m[0]) ** 4)
    f1_10 = float(3.0 * spherical_jn(1, ka_vals[0]) / ka_vals[0])
    f1_15 = float(3.0 * spherical_jn(1, ka_vals[-1]) / ka_vals[-1])
    ka4_ratio_exact_j1 = float(ka4_ratio_th * ((f1_15 / f1_10) ** 2))
    ka4_rel_err = float(abs(ka4_ratio_sim - ka4_ratio_th) / ka4_ratio_th)
    ka4_rel_err_vs_exact_j1 = float(abs(ka4_ratio_sim - ka4_ratio_exact_j1) / ka4_ratio_exact_j1)

    # Analytic sin^2(theta) pattern comparison
    theta = res_edip["theta_plot_rad"]
    sin_th = np.sin(theta)
    et_edip_norm = np.abs(res_edip["etheta_cut"]) / max(float(np.max(np.abs(res_edip["etheta_cut"]))), 1e-30)
    ep_loop_norm = np.abs(loop_runs[1]["ephi_cut"]) / max(float(np.max(np.abs(loop_runs[1]["ephi_cut"]))), 1e-30)
    edip_pat_l2_err = float(np.sqrt(np.mean((et_edip_norm - sin_th) ** 2)))
    loop_pat_l2_err = float(np.sqrt(np.mean((ep_loop_norm - sin_th) ** 2)))

    i_port_abs = float(abs(res_lumped_loop["i_port_a"]))
    p_rad_lumped = float(res_lumped_loop["p_rad_n2f_w"])
    r_rad_from_prad = float(2.0 * p_rad_lumped / max(i_port_abs**2, 1e-30))
    r_rad_circ_12p5 = float(small_loop_radiation_resistance(F0_HZ, 0.0125))
    r_rad_sq_12p5 = float(r_rad_circ_12p5 * ((4.0 / np.pi) ** 2))
    a1_over_b1_lumped = float(1.0 / max(res_lumped_loop["vsh"]["b1_over_a1"], 1e-30))
    edip_boost_lumped = float(1.0 + a1_over_b1_lumped**2)
    unexplained_res_lumped = float(r_rad_from_prad / max(r_rad_sq_12p5 * edip_boost_lumped, 1e-30))
    x_cos = 0.25 * np.pi
    analytic_cos_factor = float(((np.sin(x_cos) / x_cos) / np.cos(x_cos)) ** 2)

    data = {
        "dx_core_mm": float(dx_core * 1e3),
        "electric_dipole": {
            "p_a10_frac": float(res_edip["vsh"]["p_a10_frac"]),
            "b1_over_a1": float(res_edip["vsh"]["b1_over_a1"]),
            "cross_pol_ratio": float(
                np.max(np.abs(res_edip["ephi_cut"])) / max(float(np.max(np.abs(res_edip["etheta_cut"]))), 1e-30)
            ),
            "pattern_sin_theta_l2_err": edip_pat_l2_err,
            "wall_s": float(res_edip["wall_s"]),
            "n_cells": int(res_edip["n_cells"]),
        },
        "magnetic_loop_air": {
            "a_mm": 12.5,
            "p_b1_frac": float(loop_runs[1]["vsh"]["p_b_l"][0] / max(loop_runs[1]["vsh"]["p_vsh_tot"], 1e-30)),
            "a1_over_b1": float(1.0 / max(loop_runs[1]["vsh"]["b1_over_a1"], 1e-30)),
            "cross_pol_ratio": float(
                np.max(np.abs(loop_runs[1]["etheta_cut"])) / max(float(np.max(np.abs(loop_runs[1]["ephi_cut"]))), 1e-30)
            ),
            "pattern_sin_theta_l2_err": loop_pat_l2_err,
            "p_rad_w": float(loop_runs[1]["p_rad_n2f_w"]),
            "wall_s": float(loop_runs[1]["wall_s"]),
        },
        "magnetic_loop_core": {
            "mu_dc": float(MU_DC_DEFAULT),
            "p_rad_w": float(res_loop_core["p_rad_n2f_w"]),
            "power_gain_over_air": float(res_loop_core["p_rad_n2f_w"] / max(loop_runs[1]["p_rad_n2f_w"], 1e-30)),
            "b1_over_a1": float(res_loop_core["vsh"]["b1_over_a1"]),
            "wall_s": float(res_loop_core["wall_s"]),
        },
        "lumped_wire_loop": {
            "a_mm": 12.5,
            "z_in_real_ohm": float(np.real(res_lumped_loop["z_in_ohm"])),
            "z_in_imag_ohm": float(np.imag(res_lumped_loop["z_in_ohm"])),
            "l_loop_nh": float(np.imag(res_lumped_loop["z_in_ohm"]) / OMEGA0_RAD_S * 1e9),
            "i_port_abs_a": i_port_abs,
            "p_rad_n2f_w": p_rad_lumped,
            "r_rad_from_prad_ohm": r_rad_from_prad,
            "r_rad_circular_small_loop_ohm": r_rad_circ_12p5,
            "r_rad_square_small_loop_ohm": r_rad_sq_12p5,
            "electric_dipole_power_boost_factor": edip_boost_lumped,
            "unexplained_residual_factor": unexplained_res_lumped,
            "analytic_cos_current_factor_pi_over_4": analytic_cos_factor,
            "b1_over_a1": float(res_lumped_loop["vsh"]["b1_over_a1"]),
            "a1_over_b1_feed_asymmetry": a1_over_b1_lumped,
            "wall_s": float(res_lumped_loop["wall_s"]),
        },
        "ka4_scaling": {
            "radii_mm": [float(a * 1e3) for a in radii_m],
            "ka": ka_vals,
            "p_rad_sim_w": p_rad_loops,
            "r_rad_analytic_ohm": r_rad_exact,
            "fitted_slope": slope_ka4,
            "ratio_15_to_10_sim": ka4_ratio_sim,
            "ratio_15_to_10_theory": ka4_ratio_th,
            "ratio_15_to_10_exact_j1_theory": ka4_ratio_exact_j1,
            "ratio_rel_err": ka4_rel_err,
            "ratio_rel_err_vs_exact_j1": ka4_rel_err_vs_exact_j1,
        },
        "theta_rad": [float(x) for x in theta],
        "etheta_edip_norm": [float(x) for x in et_edip_norm],
        "ephi_edip_norm": [
            float(x)
            for x in (np.abs(res_edip["ephi_cut"]) / max(float(np.max(np.abs(res_edip["etheta_cut"]))), 1e-30))
        ],
        "ephi_loop_norm": [float(x) for x in ep_loop_norm],
        "etheta_loop_norm": [
            float(x)
            for x in (np.abs(loop_runs[1]["etheta_cut"]) / max(float(np.max(np.abs(loop_runs[1]["ephi_cut"]))), 1e-30))
        ],
    }
    (out_dir / "task_01_baseline_dipoles.json").write_text(json.dumps(_json_clean(data), indent=2) + "\n")
    return data


def run_task_02_closed_toroid_multipoles(quick: bool = True, out_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Run Task 2: Closed dispersive magnetic toroid, Dubovik-Tugushev identity, l<=5 multipoles, omega^6 & R/lambda sweeps."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dx_core = 0.0025 if quick else 0.0020

    # Broadband frequency vector for omega^6 scaling
    freqs_band = np.linspace(0.55e9, 0.95e9, 9)

    # 1. Nominal closed 8-winding dispersive toroid (with 2D cuts and frequency band)
    res_tor8 = run_single_fit_case(
        geometry="toroid",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        n_windings=8,
        continuous_sheet=False,
        dx_core_m=dx_core,
        record_2d_cuts=True,
        freqs_extra=freqs_band,
    )

    # 2. Continuous poloidal winding sheet on the same dispersive core
    res_sheet = run_single_fit_case(
        geometry="toroid",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        continuous_sheet=True,
        dx_core_m=dx_core,
    )

    # 3. Ideal azimuthal MagneticCurrentSource ring (equivalence-principle source)
    res_mring = run_single_fit_case(
        geometry="magnetic_ring",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        dx_core_m=dx_core,
    )

    # 4. Sweep R / lambda_0 to verify electric octupole |a_30|/|a_10| ~ (k_0 R)^2
    r_sweep_m = [0.010, 0.015, 0.020, 0.025]
    a3_over_a1_sim = []
    a3_over_a1_th = []
    a3_over_a1_finite_r_th = []
    b1_over_a1_r = []
    kr_vals = []
    wall_r_sweep = []
    for r_maj in r_sweep_m:
        r_run = (
            res_tor8
            if abs(r_maj - R_MAJOR_M) < 1e-9
            else run_single_fit_case(
                geometry="toroid",
                mu_dc=MU_DC_DEFAULT,
                r_major_m=r_maj,
                r_minor_m=R_MINOR_M,
                n_windings=8,
                dx_core_m=dx_core,
            )
        )
        kr = float(K0_RAD_M * r_maj)
        kr_vals.append(kr)
        a3_over_a1_sim.append(float(r_run["vsh"]["a30_over_a10"]))
        b1_over_a1_r.append(float(r_run["vsh"]["b1_over_a1"]))
        wall_r_sweep.append(float(r_run["wall_s"]))
        th_r = toroidal_dipole_radiation_theory(F0_HZ, 1.0, r_major_m=r_maj, r_minor_m=R_MINOR_M)
        a3_over_a1_th.append(float(th_r["a30_over_a10_ring"]))
        a3_over_a1_finite_r_th.append(float(th_r["a30_over_a10_finite_r"]))

    slope_a3_kr2 = float(np.polyfit(np.log(kr_vals), np.log(a3_over_a1_sim), 1)[0])

    # Marinov quasi-static toroidal moment formula check
    # In our setup, H_phi(R) = amp_h_drive = 1.0 A/m -> N * I = 2 * pi * R * H_phi(R)
    ni_amp_turns = 2.0 * np.pi * R_MAJOR_M * 1.0
    mu_eff_real = float(np.real(res_tor8["mu_disc"]))
    marinov_core = marinov_toroidal_moment_quasistatic(
        n_turns=8,
        current_a=ni_amp_turns / 8.0,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        mu_r_eff=mu_eff_real - 1.0,
    )
    marinov_tot = marinov_toroidal_moment_quasistatic(
        n_turns=8,
        current_a=ni_amp_turns / 8.0,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        mu_r_eff=mu_eff_real,
    )

    # Frequency scaling analysis: fixed T vs fixed drive current
    # Divide out the actual simulated |T_tot(f) / T_tot(f_0)|^2 (which includes both Lorentz mu(f)
    # dispersion and the finite-frequency internal Helmholtz wave enhancement inside the permeable core)
    # to isolate the exact fixed-T omega^6 scaling and verify broadband self-consistency.
    idx_f0_band = int(np.argmin(np.abs(freqs_band - F0_HZ)))
    tz_band_abs = np.abs(np.asarray(res_tor8["tz_tot_band_am3"], dtype=np.complex128))
    p_rad_j1_band = np.asarray(res_tor8["p_rad_theory_j1_band_w"], dtype=float)
    t_scale_band = (tz_band_abs / tz_band_abs[idx_f0_band]) ** 2
    p_rad_band_fixed_i = np.asarray(res_tor8["p_rad_band_w"], dtype=float)
    p_rad_band_fixed_t = p_rad_band_fixed_i / t_scale_band
    k_band = 2.0 * np.pi * freqs_band / C0
    form_factor_band = (3.0 * spherical_jn(1, k_band * R_MAJOR_M) / (k_band * R_MAJOR_M)) ** 2
    p_rad_band_fixed_t_point = p_rad_band_fixed_t / form_factor_band
    p_rad_over_j1_band = p_rad_band_fixed_i / np.maximum(p_rad_j1_band, 1e-30)

    slope_omega_fixed_t_raw = float(np.polyfit(np.log(freqs_band / F0_HZ), np.log(p_rad_band_fixed_t), 1)[0])
    slope_omega_fixed_t_point = float(np.polyfit(np.log(freqs_band / F0_HZ), np.log(p_rad_band_fixed_t_point), 1)[0])
    slope_omega_fixed_i = float(np.polyfit(np.log(freqs_band / F0_HZ), np.log(p_rad_band_fixed_i), 1)[0])

    core_mat_ref = make_core_material(mu_dc=MU_DC_DEFAULT)
    mu_band_c = np.array([complex(core_mat_ref.mu(float(fv))) for fv in freqs_band])
    mu_band_abs = np.abs(mu_band_c)
    mu_band_real = np.real(mu_band_c)
    kc_r_band = k_band * np.sqrt(np.maximum(mu_band_real, 1.0)) * R_MINOR_M
    analytic_interior_band = 1.0 + 0.125 * (kc_r_band**2)
    mu_band_rise_frac = float((mu_band_abs[-1] / mu_band_abs[0]) - 1.0)
    analytic_interior_band_rise_frac = float((analytic_interior_band[-1] / analytic_interior_band[0]) - 1.0)
    tz_sim_band_rise_frac = float((tz_band_abs[-1] / tz_band_abs[0]) - 1.0)
    th_f0_tor8 = toroidal_dipole_radiation_theory(F0_HZ, res_tor8["moments"]["tz_tot_am3"], r_major_m=R_MAJOR_M)

    theta = res_tor8["theta_plot_rad"]
    sin_th = np.sin(theta)
    et_tor_norm = np.abs(res_tor8["etheta_cut"]) / max(float(np.max(np.abs(res_tor8["etheta_cut"]))), 1e-30)
    tor_pat_l2_err = float(np.sqrt(np.mean((et_tor_norm - sin_th) ** 2)))

    # Save 2D cuts to compressed npz for fast plotting
    cuts = res_tor8["cuts"]
    np.savez_compressed(
        out_dir / "task_02_toroid_cuts.npz",
        x_xz_m=cuts["x_xz_m"],
        z_xz_m=cuts["z_xz_m"],
        ex_xz=cuts["ex_xz"],
        ez_xz=cuts["ez_xz"],
        hy_xz=cuts["hy_xz"],
        x_xy_m=cuts["x_xy_m"],
        y_xy_m=cuts["y_xy_m"],
        hx_xy=cuts["hx_xy"],
        hy_xy=cuts["hy_xy"],
        ez_xy=cuts["ez_xy"],
        mx_mask_xy=cuts["mx_mask_xy"],
        my_mask_xy=cuts["my_mask_xy"],
    )

    p_rad_sim = float(res_tor8["p_rad_n2f_w"])
    p_rad_j1 = float(res_tor8["p_rad_theory_j1_w"])
    p_rad_pt = float(res_tor8["p_rad_theory_point_w"])

    data = {
        "dx_core_mm": float(dx_core * 1e3),
        "n_cells": int(res_tor8["n_cells"]),
        "n_steps": int(res_tor8["n_steps"]),
        "wall_s": float(res_tor8["wall_s"]),
        "est_ram_mb": float(res_tor8["est_ram_mb"]),
        "moments": res_tor8["moments"],
        "marinov_quasistatic": {
            "tz_core_qs_am3": float(marinov_core["t_quasistatic_am3"]),
            "tz_core_exact_am3": float(marinov_core["t_exact_am3"]),
            "tz_tot_exact_am3": float(marinov_tot["t_exact_am3"]),
            "tz_core_sim_abs_am3": float(abs(res_tor8["moments"]["tz_mag_core_am3"])),
            "tz_tot_sim_abs_am3": float(abs(res_tor8["moments"]["tz_tot_am3"])),
            "rel_err_tot_sim_vs_exact": float(
                abs(abs(res_tor8["moments"]["tz_tot_am3"]) - marinov_tot["t_exact_am3"]) / marinov_tot["t_exact_am3"]
            ),
        },
        "vsh_toroid_8feed": {
            "p_a_l_frac": [
                float(x / max(res_tor8["vsh"]["p_vsh_tot"], 1e-30)) for x in res_tor8["vsh"]["p_a_l"]
            ],
            "p_b_l_frac": [
                float(x / max(res_tor8["vsh"]["p_vsh_tot"], 1e-30)) for x in res_tor8["vsh"]["p_b_l"]
            ],
            "p_a10_frac": float(res_tor8["vsh"]["p_a10_frac"]),
            "a10_complex": complex(res_tor8["vsh"]["a10"]),
            "a10_theory_complex": complex(th_f0_tor8["a10_j1"]),
            "a10_over_tz_phase_rad": float(np.angle(res_tor8["vsh"]["a10"] / res_tor8["moments"]["tz_tot_am3"])),
            "a10_over_tz_phase_deg": float(np.degrees(np.angle(res_tor8["vsh"]["a10"] / res_tor8["moments"]["tz_tot_am3"]))),
            "a10_over_tz_theory_phase_deg": -90.0,
            "b1_over_a1": float(res_tor8["vsh"]["b1_over_a1"]),
            "b_tot_over_a_tot": float(res_tor8["vsh"]["b_tot_over_a_tot"]),
            "a30_over_a10": float(res_tor8["vsh"]["a30_over_a10"]),
            "even_l_power_frac": float(
                (
                    res_tor8["vsh"]["p_a_l"][1]
                    + res_tor8["vsh"]["p_a_l"][3]
                    + res_tor8["vsh"]["p_b_l"][1]
                    + res_tor8["vsh"]["p_b_l"][3]
                )
                / max(res_tor8["vsh"]["p_vsh_tot"], 1e-30)
            ),
            "pattern_sin_theta_l2_err": tor_pat_l2_err,
            "cross_pol_ratio": float(
                np.max(np.abs(res_tor8["ephi_cut"])) / max(float(np.max(np.abs(res_tor8["etheta_cut"]))), 1e-30)
            ),
        },
        "feed_comparison": {
            "toroid_8_windings": {
                "tz_tot_abs_am3": float(abs(res_tor8["moments"]["tz_tot_am3"])),
                "b1_over_a1": float(res_tor8["vsh"]["b1_over_a1"]),
                "p_a10_frac": float(res_tor8["vsh"]["p_a10_frac"]),
                "p_rad_w": p_rad_sim,
                "wall_s": float(res_tor8["wall_s"]),
            },
            "continuous_poloidal_sheet": {
                "tz_tot_abs_am3": float(abs(res_sheet["moments"]["tz_tot_am3"])),
                "b1_over_a1": float(res_sheet["vsh"]["b1_over_a1"]),
                "p_a10_frac": float(res_sheet["vsh"]["p_a10_frac"]),
                "p_rad_w": float(res_sheet["p_rad_n2f_w"]),
                "wall_s": float(res_sheet["wall_s"]),
            },
            "ideal_magnetic_current_ring": {
                "b1_over_a1": float(res_mring["vsh"]["b1_over_a1"]),
                "p_a10_frac": float(res_mring["vsh"]["p_a10_frac"]),
                "p_rad_w": float(res_mring["p_rad_n2f_w"]),
                "wall_s": float(res_mring["wall_s"]),
            },
        },
        "power_equivalence": {
            "p_rad_n2f_w": p_rad_sim,
            "p_rad_flux_w": float(res_tor8["p_rad_flux_w"]),
            "p_rad_theory_j1_w": p_rad_j1,
            "p_rad_theory_point_w": p_rad_pt,
            "p_eff_abs_cm": float(abs(res_tor8["p_eff_j1_cm"])),
            "rel_err_n2f_vs_j1": float(abs(p_rad_sim - p_rad_j1) / p_rad_j1),
            "rel_err_flux_vs_j1": float(abs(res_tor8["p_rad_flux_w"] - p_rad_j1) / p_rad_j1),
            "rel_err_n2f_vs_point": float(abs(p_rad_sim - p_rad_pt) / p_rad_pt),
        },
        "frequency_scaling": {
            "freqs_ghz": [float(f / 1e9) for f in freqs_band],
            "tz_tot_band_abs_am3": [float(x) for x in tz_band_abs],
            "p_rad_theory_j1_band_w": [float(x) for x in p_rad_j1_band],
            "p_rad_over_theory_j1_band": [float(x) for x in p_rad_over_j1_band],
            "p_rad_fixed_i_w": [float(x) for x in p_rad_band_fixed_i],
            "p_rad_fixed_t_w": [float(x) for x in p_rad_band_fixed_t],
            "p_rad_fixed_t_point_corrected_w": [float(x) for x in p_rad_band_fixed_t_point],
            "slope_omega_fixed_t_raw": slope_omega_fixed_t_raw,
            "slope_omega_fixed_t_point_corrected": slope_omega_fixed_t_point,
            "slope_omega_fixed_i": slope_omega_fixed_i,
            "mu_band_rise_frac": mu_band_rise_frac,
            "analytic_interior_band_rise_frac": analytic_interior_band_rise_frac,
            "tz_sim_band_rise_frac": tz_sim_band_rise_frac,
        },
        "octupole_scaling": {
            "r_major_mm": [float(r * 1e3) for r in r_sweep_m],
            "r_over_lambda0": [float(r / LAM0_M) for r in r_sweep_m],
            "k0_r": kr_vals,
            "a30_over_a10_sim": a3_over_a1_sim,
            "a30_over_a10_theory": a3_over_a1_th,
            "a30_over_a10_finite_r_theory": a3_over_a1_finite_r_th,
            "b1_over_a1_sim": b1_over_a1_r,
            "fitted_slope_vs_kr": slope_a3_kr2,
            "wall_s": wall_r_sweep,
            "sim_over_theory_ratio": [float(s / t) for s, t in zip(a3_over_a1_sim, a3_over_a1_th)],
            "sim_over_finite_r_theory_ratio": [
                float(s / t) for s, t in zip(a3_over_a1_sim, a3_over_a1_finite_r_th)
            ],
        },
        "theta_rad": [float(x) for x in theta],
        "etheta_toroid_norm": [float(x) for x in et_tor_norm],
        "ephi_toroid_norm": [
            float(x)
            for x in (np.abs(res_tor8["ephi_cut"]) / max(float(np.max(np.abs(res_tor8["etheta_cut"]))), 1e-30))
        ],
    }
    (out_dir / "task_02_closed_toroid.json").write_text(json.dumps(_json_clean(data), indent=2) + "\n")
    return data


def run_task_03_square_four_rods(quick: bool = True, out_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Run Task 3: Square of 4 permeable rods (C4 symmetry guarantees m=0, C4 residuals grow with k R_sq)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dx_core = 0.0025 if quick else 0.0020

    r_sq_list_m = [0.0125, 0.0175, 0.0225]
    runs_sq = []
    for r_sq in r_sq_list_m:
        r_run = run_single_fit_case(
            geometry="four_rods",
            mu_dc=MU_DC_DEFAULT,
            r_major_m=r_sq,
            r_minor_m=0.005,
            dx_core_m=dx_core,
            record_2d_cuts=(abs(r_sq - 0.015) < 0.003),
        )
        runs_sq.append(r_run)

    # Save 2D XY cut of the 4-rod square for plotting
    cuts_sq = runs_sq[0]["cuts"]
    np.savez_compressed(
        out_dir / "task_03_four_rods_cuts.npz",
        x_xy_m=cuts_sq["x_xy_m"],
        y_xy_m=cuts_sq["y_xy_m"],
        hx_xy=cuts_sq["hx_xy"],
        hy_xy=cuts_sq["hy_xy"],
        ez_xy=cuts_sq["ez_xy"],
        mx_mask_xy=cuts_sq["mx_mask_xy"],
        my_mask_xy=cuts_sq["my_mask_xy"],
    )

    records = []
    for r_sq, r_run in zip(r_sq_list_m, runs_sq):
        vsh = r_run["vsh"]
        mom = r_run["moments"]
        # Extract C4 azimuth harmonic m = +/-4 amplitude at l = 4, 5
        a_m4_pow = float(abs(vsh["a_lm"][(4, 4)]) ** 2 + abs(vsh["a_lm"][(4, -4)]) ** 2 + abs(vsh["a_lm"][(5, 4)]) ** 2 + abs(vsh["a_lm"][(5, -4)]) ** 2)
        b_m4_pow = float(abs(vsh["b_lm"][(4, 4)]) ** 2 + abs(vsh["b_lm"][(4, -4)]) ** 2 + abs(vsh["b_lm"][(5, 4)]) ** 2 + abs(vsh["b_lm"][(5, -4)]) ** 2)
        p_a10 = float(abs(vsh["a_lm"][(1, 0)]) ** 2)
        c4_a_over_a10 = float(np.sqrt(a_m4_pow / max(p_a10, 1e-30)))
        c4_b_over_a10 = float(np.sqrt(b_m4_pow / max(p_a10, 1e-30)))
        c4_over_a10 = float(np.sqrt((a_m4_pow + b_m4_pow) / max(p_a10, 1e-30)))
        non_a10_frac = float(1.0 - vsh["p_a10_frac"])
        records.append({
            "r_sq_mm": float(r_sq * 1e3),
            "k0_r_sq": float(K0_RAD_M * r_sq),
            "m_cancel_ratio": float(mom["m_cancel_ratio"]),
            "tz_tot_abs_am3": float(abs(mom["tz_tot_am3"])),
            "p_a10_frac": float(vsh["p_a10_frac"]),
            "non_a10_power_frac": non_a10_frac,
            "b1_over_a1": float(vsh["b1_over_a1"]),
            "b_tot_over_a_tot": float(vsh["b_tot_over_a_tot"]),
            "c4_a_over_a10": c4_a_over_a10,
            "c4_b_over_a10": c4_b_over_a10,
            "c4_m4_over_a10": c4_over_a10,
            "p_rad_n2f_w": float(r_run["p_rad_n2f_w"]),
            "p_rad_theory_j1_w": float(r_run["p_rad_theory_j1_w"]),
            "wall_s": float(r_run["wall_s"]),
        })

    data = {
        "dx_core_mm": float(dx_core * 1e3),
        "records": records,
    }
    (out_dir / "task_03_square_four_rods.json").write_text(json.dumps(_json_clean(data), indent=2) + "\n")
    return data


def run_task_04_break_the_loop(quick: bool = True, out_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Run Task 4: Symmetry and loop-breaking failure tests (gap, 7/8 feeds, Mode 1, 3/4 rods, helical turn + Marinov loop)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dx_core = 0.0025 if quick else 0.0020

    # 1. Baseline closed 8-feed torus
    r_closed = run_single_fit_case(
        geometry="toroid",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        n_windings=8,
        dx_core_m=dx_core,
    )

    # 2. Air gap in the toroidal core (gap_angle = 60 deg = pi/3 rad)
    r_gap = run_single_fit_case(
        geometry="toroid",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        n_windings=8,
        gap_angle_rad=float(np.pi / 3.0),
        dx_core_m=dx_core,
        record_2d_cuts=True,
    )

    # 3. 7 of 8 feeds active (Canos Valero et al. 2024 broken-symmetry case)
    r_7of8 = run_single_fit_case(
        geometry="toroid",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        n_windings=8,
        active_windings=[0, 1, 2, 3, 4, 5, 6],
        dx_core_m=dx_core,
        record_2d_cuts=True,
    )

    # 4. Unbalanced feed phases: Mode 1 antiphase "loop mode" ([+1,+1,+1,+1,-1,-1,-1,-1])
    r_mode1 = run_single_fit_case(
        geometry="toroid",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        n_windings=8,
        winding_phases=[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        dx_core_m=dx_core,
    )

    # 5. 3 of 4 rods in the square (one rod removed)
    r_4rods = run_single_fit_case(
        geometry="four_rods",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=0.0125,
        r_minor_m=0.005,
        active_windings=[0, 1, 2, 3],
        dx_core_m=dx_core,
    )
    r_3rods = run_single_fit_case(
        geometry="four_rods",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=0.0125,
        r_minor_m=0.005,
        active_windings=[0, 1, 2],
        dx_core_m=dx_core,
    )

    # 6. Helical winding with 1 net azimuthal turn vs Marinov's compensating reverse loop
    r1_marinov_m = float(R_MAJOR_M * np.sqrt(1.0 + 0.5 * ((R_MINOR_M / R_MAJOR_M) ** 2)))
    r_helical = run_single_fit_case(
        geometry="toroid",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        n_windings=8,
        helical_parasitic_turn=True,
        marinov_compensating_loop=False,
        dx_core_m=dx_core,
    )
    r_marinov = run_single_fit_case(
        geometry="toroid",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        n_windings=8,
        helical_parasitic_turn=True,
        marinov_compensating_loop=True,
        marinov_loop_radius_m=r1_marinov_m,
        dx_core_m=dx_core,
    )

    # 7. Sweep compensating loop radius R_loop across R_1 = R * sqrt(1 + 0.5 * (r/R)^2) (Marinov Eq. 9)
    # driven by the EXACT SAME current -I_turn (zero rescaling)
    r_loop_sweep_m = [0.0130, 0.0145, 0.0150, r1_marinov_m, 0.0158, 0.0170]
    sweep_b1_over_a1: list[float] = []
    sweep_a1_over_a1_closed: list[float] = []
    a1_closed_ref = float(r_closed["vsh"]["a1_amp"])
    for rl_m in r_loop_sweep_m:
        if abs(rl_m - r1_marinov_m) < 1e-12:
            r_sw = r_marinov
        else:
            r_sw = run_single_fit_case(
                geometry="toroid",
                mu_dc=MU_DC_DEFAULT,
                r_major_m=R_MAJOR_M,
                r_minor_m=R_MINOR_M,
                n_windings=8,
                helical_parasitic_turn=True,
                marinov_compensating_loop=True,
                marinov_loop_radius_m=rl_m,
                dx_core_m=dx_core,
            )
        sweep_b1_over_a1.append(float(r_sw["vsh"]["b1_over_a1"]))
        sweep_a1_over_a1_closed.append(float(r_sw["vsh"]["a1_amp"] / max(a1_closed_ref, 1e-30)))

    mu_c_abs = abs(complex(make_core_material(mu_dc=MU_DC_DEFAULT).mu(F0_HZ)))
    f1_tor = float(toroidal_dipole_radiation_theory(F0_HZ, 1.0, r_major_m=R_MAJOR_M)["form_factor_j1"])
    b1_a1_hel_theory = float(
        (2.0 * (r1_marinov_m**2)) / (8.0 * mu_c_abs * (K0_RAD_M * R_MAJOR_M) * (R_MINOR_M**2) * f1_tor)
    )
    idx_r_major = int(np.argmin(np.abs(np.asarray(r_loop_sweep_m) - R_MAJOR_M)))
    idx_r1 = int(np.argmin(np.abs(np.asarray(r_loop_sweep_m) - r1_marinov_m)))
    marinov_sweep = {
        "r_major_mm": float(R_MAJOR_M * 1e3),
        "r_minor_mm": float(R_MINOR_M * 1e3),
        "r1_marinov_mm": float(r1_marinov_m * 1e3),
        "i_turn_a": float((2.0 * np.pi * R_MAJOR_M * 1.0) / 8.0),
        "b1_over_a1_uncompensated": float(r_helical["vsh"]["b1_over_a1"]),
        "b1_over_a1_theory_uncompensated": b1_a1_hel_theory,
        "r_loop_mm": [float(x * 1e3) for x in r_loop_sweep_m],
        "b1_over_a1": sweep_b1_over_a1,
        "a1_over_a1_closed": sweep_a1_over_a1_closed,
        "b1_over_a1_at_r_major": float(sweep_b1_over_a1[idx_r_major]),
        "b1_over_a1_at_r1": float(sweep_b1_over_a1[idx_r1]),
        "suppression_factor_at_r1": float(
            r_helical["vsh"]["b1_over_a1"] / max(sweep_b1_over_a1[idx_r1], 1e-30)
        ),
        "improvement_over_naive_r_major": float(
            sweep_b1_over_a1[idx_r_major] / max(sweep_b1_over_a1[idx_r1], 1e-30)
        ),
    }

    def _summarize_case(label: str, run: dict[str, Any]) -> dict[str, Any]:
        vsh = run["vsh"]
        mom = run["moments"]
        p_b_frac = float(np.sum(vsh["p_b_l"]) / max(vsh["p_vsh_tot"], 1e-30))
        p_a_frac = float(np.sum(vsh["p_a_l"]) / max(vsh["p_vsh_tot"], 1e-30))
        return {
            "label": label,
            "b1_over_a1": float(vsh["b1_over_a1"]),
            "a1_over_a1_closed": float(vsh["a1_amp"] / max(a1_closed_ref, 1e-30)),
            "b1_over_a1_closed": float(vsh["b1_amp"] / max(a1_closed_ref, 1e-30)),
            "b_tot_over_a_tot": float(vsh["b_tot_over_a_tot"]),
            "p_a10_frac": float(vsh["p_a10_frac"]),
            "p_electric_frac": p_a_frac,
            "p_magnetic_frac": p_b_frac,
            "m_cancel_ratio": float(mom["m_cancel_ratio"]),
            "tz_tot_abs_am3": float(abs(mom["tz_tot_am3"])),
            "p_rad_n2f_w": float(run["p_rad_n2f_w"]),
            "wall_s": float(run["wall_s"]),
            "etheta_abs": [float(abs(x)) for x in run["etheta_cut"]],
            "ephi_abs": [float(abs(x)) for x in run["ephi_cut"]],
        }

    cases = {
        "closed_8_feeds": _summarize_case("Closed torus (8 in-phase feeds)", r_closed),
        "core_air_gap_60deg": _summarize_case("Core air gap (60° sector removed)", r_gap),
        "seven_of_eight_feeds": _summarize_case("7 of 8 feeds active (1 feed open)", r_7of8),
        "mode1_antiphase": _summarize_case("Mode 1 antiphase (++ ++ -- --)", r_mode1),
        "four_rods_closed": _summarize_case("Square of 4 rods (C4 closed)", r_4rods),
        "three_of_four_rods": _summarize_case("3 of 4 rods (1 rod removed)", r_3rods),
        "helical_uncompensated": _summarize_case("Helical winding (1 net azimuthal turn)", r_helical),
        "helical_marinov_compensated": _summarize_case("Helical + Marinov compensating loop", r_marinov),
    }

    np.savez_compressed(
        out_dir / "task_04_broken_cuts.npz",
        x_xy_m=r_gap["cuts"]["x_xy_m"],
        y_xy_m=r_gap["cuts"]["y_xy_m"],
        hx_gap_xy=r_gap["cuts"]["hx_xy"],
        hy_gap_xy=r_gap["cuts"]["hy_xy"],
        mx_gap_xy=r_gap["cuts"]["mx_mask_xy"],
        my_gap_xy=r_gap["cuts"]["my_mask_xy"],
        hx_7of8_xy=r_7of8["cuts"]["hx_xy"],
        hy_7of8_xy=r_7of8["cuts"]["hy_xy"],
        mx_7of8_xy=r_7of8["cuts"]["mx_mask_xy"],
        my_7of8_xy=r_7of8["cuts"]["my_mask_xy"],
    )

    data = {
        "dx_core_mm": float(dx_core * 1e3),
        "theta_rad": [float(x) for x in r_closed["theta_plot_rad"]],
        "cases": cases,
        "marinov_r_loop_sweep": marinov_sweep,
    }
    (out_dir / "task_04_break_the_loop.json").write_text(json.dumps(_json_clean(data), indent=2) + "\n")
    return data


def run_task_05_equivalence_limits(quick: bool = True, out_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Run Task 5: Equivalence limits (electrical size, core TE01-like trapped mode, finite N windings, near-field vs dipole)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dx_core = 0.0025 if quick else 0.0020

    # 1. Finite N windings sweep: N in {2, 4, 8} + continuous sheet (N -> inf)
    n_wind_list = [2, 4, 8]
    winding_records = []
    for nw in n_wind_list:
        r_nw = run_single_fit_case(
            geometry="toroid",
            mu_dc=MU_DC_DEFAULT,
            r_major_m=R_MAJOR_M,
            r_minor_m=R_MINOR_M,
            n_windings=nw,
            continuous_sheet=False,
            dx_core_m=dx_core,
        )
        winding_records.append({
            "n_windings": str(nw),
            "tz_tot_abs_am3": float(abs(r_nw["moments"]["tz_tot_am3"])),
            "m_cancel_ratio": float(r_nw["moments"]["m_cancel_ratio"]),
            "p_a10_frac": float(r_nw["vsh"]["p_a10_frac"]),
            "b1_over_a1": float(r_nw["vsh"]["b1_over_a1"]),
            "p_rad_w": float(r_nw["p_rad_n2f_w"]),
            "wall_s": float(r_nw["wall_s"]),
        })
    r_inf = run_single_fit_case(
        geometry="toroid",
        mu_dc=MU_DC_DEFAULT,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        continuous_sheet=True,
        dx_core_m=dx_core,
        record_2d_cuts=True,
    )
    winding_records.append({
        "n_windings": "inf",
        "tz_tot_abs_am3": float(abs(r_inf["moments"]["tz_tot_am3"])),
        "m_cancel_ratio": float(r_inf["moments"]["m_cancel_ratio"]),
        "p_a10_frac": float(r_inf["vsh"]["p_a10_frac"]),
        "b1_over_a1": float(r_inf["vsh"]["b1_over_a1"]),
        "p_rad_w": float(r_inf["p_rad_n2f_w"]),
        "wall_s": float(r_inf["wall_s"]),
    })

    # 2. Core internal dielectric-magnetic TE01-like mode sweep: vary eps_core in {1.0, 10.0, 22.0}
    # Resolved at >= 20 cells/lambda_core in the core (dx_core = 2.50, 1.38, 0.93 mm -> 35.5, 20.3, 20.3 cells/lambda_core).
    # For 2r = 10 mm and mu_r = 20.30, f_TE01 = c0 / (2 * sqrt(eps_core * mu_r) * 2r):
    #   eps_core = 1.0  -> f_TE01 = 3.33 GHz (2r / lambda_core = 0.113, quasi-static uniform M_phi)
    #   eps_core = 10.0 -> f_TE01 = 1.05 GHz (2r / lambda_core = 0.356, approaching first internal resonance)
    #   eps_core = 22.0 -> f_TE01 = 0.71 GHz (2r / lambda_core = 0.529 > 0.5, above f_TE01 threshold!)
    ni_amp_turns = 2.0 * np.pi * R_MAJOR_M * 1.0
    tz_qs_exact = float(
        marinov_toroidal_moment_quasistatic(
            n_turns=8,
            current_a=ni_amp_turns / 8.0,
            r_major_m=R_MAJOR_M,
            r_minor_m=R_MINOR_M,
            mu_r_eff=float(np.real(make_core_material(mu_dc=MU_DC_DEFAULT).mu(F0_HZ))),
        )["t_exact_am3"]
    )
    eps_dx_list = [(1.0, 0.0025), (10.0, 0.00138), (22.0, 0.00093)]
    te01_records = []
    tz_eps1_ref = float(abs(r_inf["moments"]["tz_tot_am3"]))
    pr_eps1_ref = float(r_inf["p_rad_n2f_w"])
    for eps_c, dx_te in eps_dx_list:
        r_te = (
            r_inf
            if abs(eps_c - 1.0) < 1e-9
            else run_single_fit_case(
                geometry="toroid",
                mu_dc=MU_DC_DEFAULT,
                eps_core=eps_c,
                r_major_m=R_MAJOR_M,
                r_minor_m=R_MINOR_M,
                continuous_sheet=True,
                dx_core_m=dx_te,
            )
        )
        n_core_val = float(np.sqrt(eps_c * float(np.real(r_te["mu_disc"]))))
        lam_core_m = float(LAM0_M / n_core_val)
        f_te01_ghz = float(C0 / (2.0 * n_core_val * (2.0 * R_MINOR_M)) / 1e9)
        tz_abs = float(abs(r_te["moments"]["tz_tot_am3"]))
        p_tor_eff = complex(r_te["p_eff_j1_cm"])
        p_pol_z = complex(r_te["moments"]["p_pol_z_cm"])
        p_hyb_tot = p_tor_eff + p_pol_z
        p_hyb_pred_w = float((ETA0 * (OMEGA0_RAD_S**4) / (12.0 * np.pi * (C0**2))) * (abs(p_hyb_tot) ** 2))
        te01_records.append({
            "eps_core": float(eps_c),
            "dx_core_mm": float(dx_te * 1e3),
            "cells_per_lambda_core": float(lam_core_m / dx_te),
            "n_core": n_core_val,
            "two_r_over_lambda_core": float((2.0 * R_MINOR_M) / lam_core_m),
            "f_te01_ghz": f_te01_ghz,
            "tz_tot_abs_am3": tz_abs,
            "p_tor_eff_abs_cm": float(abs(p_tor_eff)),
            "p_pol_z_abs_cm": float(abs(p_pol_z)),
            "p_hybrid_tot_abs_cm": float(abs(p_hyb_tot)),
            "p_rad_sim_w": float(r_te["p_rad_n2f_w"]),
            "p_rad_hybrid_pred_w": p_hyb_pred_w,
            "p_rad_quasistatic_pred_w": pr_eps1_ref,
            "ratio_to_eps1": float(tz_abs / max(tz_eps1_ref, 1e-30)),
            "ratio_to_quasistatic_formula": float(tz_abs / max(tz_qs_exact, 1e-30)),
            "ratio_to_quasistatic": float(tz_abs / max(tz_eps1_ref, 1e-30)),
            "wall_s": float(r_te["wall_s"]),
        })

    # 3. Near-field comparison: Toroid vs Hertzian dipole with the exact same effective dipole moment p_eff
    cuts_tor = r_inf["cuts"]
    x_xz = cuts_tor["x_xz_m"]
    z_xz = cuts_tor["z_xz_m"]
    xx_xz, zz_xz = np.meshgrid(x_xz, z_xz, indexing="ij")
    yy_xz = np.zeros_like(xx_xz)
    hd_fields = hertzian_dipole_fields(
        xx_xz,
        yy_xz,
        zz_xz,
        F0_HZ,
        p_z=complex(r_inf["p_eff_j1_cm"]),
        r_min_clip=0.006,
    )
    np.savez_compressed(
        out_dir / "task_05_nearfield_vs_dipole.npz",
        x_xz_m=x_xz,
        z_xz_m=z_xz,
        ex_tor_xz=cuts_tor["ex_xz"],
        ez_tor_xz=cuts_tor["ez_xz"],
        hy_tor_xz=cuts_tor["hy_xz"],
        ex_dip_xz=hd_fields["Ex"],
        ez_dip_xz=hd_fields["Ez"],
        hy_dip_xz=hd_fields["Hy"],
    )

    # Quantify near-field vs outer-zone difference using the SAME metric (|E| and |Hy|) in both zones:
    #   Inner reactive zone: r in [20, 30] mm (k_0 r in [0.31, 0.47], just outside bounding sphere a_sph = 20 mm)
    #   Outer reactive/intermediate zone: r in [45, 55] mm (k_0 r in [0.71, 0.86])
    r_grid = np.sqrt(xx_xz**2 + zz_xz**2)
    mask_near = (r_grid >= 0.020) & (r_grid <= 0.030)
    mask_outer = (r_grid >= 0.045) & (r_grid <= 0.055)
    e_tor_mag = np.sqrt(np.abs(cuts_tor["ex_xz"]) ** 2 + np.abs(cuts_tor["ez_xz"]) ** 2)
    e_dip_mag = np.sqrt(np.abs(hd_fields["Ex"]) ** 2 + np.abs(hd_fields["Ez"]) ** 2)
    hy_tor_mag = np.abs(cuts_tor["hy_xz"])
    hy_dip_mag = np.abs(hd_fields["Hy"])

    near_e_rel_diff = float(
        np.linalg.norm(e_tor_mag[mask_near] - e_dip_mag[mask_near]) / max(np.linalg.norm(e_dip_mag[mask_near]), 1e-30)
    )
    outer_e_rel_diff = float(
        np.linalg.norm(e_tor_mag[mask_outer] - e_dip_mag[mask_outer]) / max(np.linalg.norm(e_dip_mag[mask_outer]), 1e-30)
    )
    near_hy_rel_diff = float(
        np.linalg.norm(hy_tor_mag[mask_near] - hy_dip_mag[mask_near]) / max(np.linalg.norm(hy_dip_mag[mask_near]), 1e-30)
    )
    outer_hy_rel_diff = float(
        np.linalg.norm(hy_tor_mag[mask_outer] - hy_dip_mag[mask_outer]) / max(np.linalg.norm(hy_dip_mag[mask_outer]), 1e-30)
    )

    # Analytical spherical Hankel octupole near-field enhancement |h_3^(2)(k_0 r) / h_1^(2)(k_0 r)| at r = 25 mm vs 50 mm
    from scipy.special import spherical_yn

    def _h_ratio_3_to_1(kr_val: float) -> float:
        h1 = complex(spherical_jn(1, kr_val) - 1j * spherical_yn(1, kr_val))
        h3 = complex(spherical_jn(3, kr_val) - 1j * spherical_yn(3, kr_val))
        return float(abs(h3 / h1))

    h3_over_h1_25mm = _h_ratio_3_to_1(float(K0_RAD_M * 0.025))
    h3_over_h1_50mm = _h_ratio_3_to_1(float(K0_RAD_M * 0.050))

    # Analytical electrical-size curve: ring form factor [3 j_1(k R) / (k R)]^2 showing the first zero at k R = 4.49340946
    kr_curve = np.linspace(0.01, 5.2, 200)
    form_factor_sq = (3.0 * spherical_jn(1, kr_curve) / kr_curve) ** 2
    a3_over_a1_curve = np.sqrt(7.0 / 8.0) * np.abs(
        spherical_jn(3, kr_curve) / np.maximum(np.abs(spherical_jn(1, kr_curve)), 1e-12)
    )

    data = {
        "dx_core_mm": float(dx_core * 1e3),
        "winding_convergence": winding_records,
        "core_te01_mode_sweep": te01_records,
        "nearfield_vs_dipole": {
            "p_eff_abs_cm": float(abs(r_inf["p_eff_j1_cm"])),
            "near_zone_20_to_30mm_e_rel_diff": near_e_rel_diff,
            "outer_zone_45_to_55mm_e_rel_diff": outer_e_rel_diff,
            "e_metric_reduction_factor": float(near_e_rel_diff / max(outer_e_rel_diff, 1e-30)),
            "near_zone_20_to_30mm_hy_rel_diff": near_hy_rel_diff,
            "outer_zone_45_to_55mm_hy_rel_diff": outer_hy_rel_diff,
            "hy_metric_reduction_factor": float(near_hy_rel_diff / max(outer_hy_rel_diff, 1e-30)),
            "hankel_h3_over_h1_at_25mm": h3_over_h1_25mm,
            "hankel_h3_over_h1_at_50mm": h3_over_h1_50mm,
            "hankel_octupole_decay_25_to_50mm": float(h3_over_h1_25mm / h3_over_h1_50mm),
        },
        "electrical_size_theory": {
            "kr_first_bessel_zero": float(brentq(lambda x: float(spherical_jn(1, x)), 4.0, 5.0)),
            "circumference_over_lambda_at_zero": float(brentq(lambda x: float(spherical_jn(1, x)), 4.0, 5.0)),
            "kr_equatorial_j1_zero": float(jn_zeros(1, 1)[0]),
            "kr_curve": [float(x) for x in kr_curve],
            "form_factor_sq": [float(x) for x in form_factor_sq],
            "a3_over_a1_curve": [float(x) for x in a3_over_a1_curve],
        },
    }
    (out_dir / "task_05_equivalence_limits.json").write_text(json.dumps(_json_clean(data), indent=2) + "\n")
    return data


def run_task_06_efficiency_q_and_convergence(quick: bool = True, out_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Run Task 6: Core loss sweep (energy bookkeeping P_in = P_rad + P_loss), Q vs Chu bound, and 3-level mesh convergence."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dx_core = 0.0025 if quick else 0.0020

    # 1. Core loss tangent sweep: vary Lorentz damping gamma_hz
    gamma_list_hz = [0.02e9, 0.10e9, 0.30e9, 1.00e9]
    loss_records = []
    for gam in gamma_list_hz:
        r_loss = run_single_fit_case(
            geometry="toroid",
            mu_dc=MU_DC_DEFAULT,
            gamma_hz=gam,
            r_major_m=R_MAJOR_M,
            r_minor_m=R_MINOR_M,
            n_windings=8,
            dx_core_m=dx_core,
        )
        mu_d = complex(r_loss["mu_disc"])
        mu_p = float(np.real(mu_d))
        mu_pp = max(-float(np.imag(mu_d)), 0.0)
        tan_d = float(mu_pp / max(mu_p, 1e-30))
        th_eff = toroidal_efficiency_and_q_theory(
            F0_HZ,
            R_MAJOR_M,
            R_MINOR_M,
            mu_real=mu_p,
            tan_delta_m=tan_d,
        )
        # Simulated radiation Q from stored magnetic energy inside the core: Q_rad = 2 * omega_0 * W_m / P_rad
        q_rad_sim = float(2.0 * OMEGA0_RAD_S * r_loss["w_stored_mag_core_j"] / max(r_loss["p_rad_flux_w"], 1e-30))
        loss_records.append({
            "gamma_ghz": float(gam / 1e9),
            "mu_real": mu_p,
            "mu_imag": mu_pp,
            "tan_delta_m": tan_d,
            "p_in_w": float(r_loss["p_in_w"]),
            "p_rad_flux_w": float(r_loss["p_rad_flux_w"]),
            "p_rad_n2f_w": float(r_loss["p_rad_n2f_w"]),
            "p_loss_core_w": float(r_loss["p_loss_core_w"]),
            "p_out_tot_w": float(r_loss["p_rad_flux_w"] + r_loss["p_loss_core_w"]),
            "energy_closure_rel_err": float(r_loss["energy_closure_rel_err"]),
            "energy_closure_mutated_rel_err": float(r_loss["energy_closure_mutated_rel_err"]),
            "eta_rad_sim": float(r_loss["eta_rad_sim"]),
            "eta_rad_theory": float(th_eff["eta_rad"]),
            "q_rad_sim": q_rad_sim,
            "q_rad_theory": float(th_eff["q_rad_toroid"]),
            "q_chu_enclosing_sphere": float(th_eff["q_chu"]),
            "q_ratio_over_chu_sim": float(q_rad_sim / th_eff["q_chu"]),
            "wall_s": float(r_loss["wall_s"]),
        })

    # 2. 3-level mesh resolution convergence study on the closed 8-feed toroid (all levels >= 4 cells across 2r):
    #    Level 1: dx_core = 2.500 mm (35.5 cells/lambda_core, 4.0 cells across 2r = 10 mm)
    #    Level 2: dx_core = 2.000 mm (44.4 cells/lambda_core, 5.0 cells across 2r = 10 mm)
    #    Level 3: dx_core = 1.667 mm (53.2 cells/lambda_core, 6.0 cells across 2r = 10 mm)
    ni_amp_turns = 2.0 * np.pi * R_MAJOR_M * 1.0
    marinov_tot = marinov_toroidal_moment_quasistatic(
        n_turns=8,
        current_a=ni_amp_turns / 8.0,
        r_major_m=R_MAJOR_M,
        r_minor_m=R_MINOR_M,
        mu_r_eff=float(np.real(make_core_material(mu_dc=MU_DC_DEFAULT).mu(F0_HZ))),
    )
    tz_exact = float(marinov_tot["t_exact_am3"])

    dx_conv_list = [0.0025, 0.0020, 0.010 / 6.0]
    conv_records = []
    for dx_c in dx_conv_list:
        r_c = run_single_fit_case(
            geometry="toroid",
            mu_dc=MU_DC_DEFAULT,
            r_major_m=R_MAJOR_M,
            r_minor_m=R_MINOR_M,
            n_windings=8,
            dx_core_m=dx_c,
        )
        n_core = float(np.sqrt(float(np.real(r_c["mu_disc"]))))
        lam_core_m = float(LAM0_M / n_core)
        tz_sim_abs = float(abs(r_c["moments"]["tz_tot_am3"]))
        tz_disc_static = float(r_c["moments"]["tz_disc_static_drive_am3"])
        res_ratio = float(r_c["moments"]["tz_sim_over_disc_residual_ratio"])
        pr_n2f = float(r_c["p_rad_n2f_w"])
        pr_j1 = float(r_c["p_rad_theory_j1_w"])
        conv_records.append({
            "dx_core_mm": float(dx_c * 1e3),
            "cells_per_lambda_core": float(lam_core_m / dx_c),
            "cells_across_2r": float((2.0 * R_MINOR_M) / dx_c),
            "nx": int(r_c["nx"]),
            "ny": int(r_c["ny"]),
            "nz": int(r_c["nz"]),
            "n_cells": int(r_c["n_cells"]),
            "n_steps": int(r_c["n_steps"]),
            "wall_s": float(r_c["wall_s"]),
            "est_ram_mb": float(r_c["est_ram_mb"]),
            "tz_tot_abs_am3": tz_sim_abs,
            "tz_disc_static_drive_am3": tz_disc_static,
            "tz_disc_over_cont_static": float(tz_disc_static / tz_exact),
            "tz_sim_over_disc_residual_ratio": res_ratio,
            "p_rad_n2f_w": pr_n2f,
            "p_rad_flux_w": float(r_c["p_rad_flux_w"]),
            "p_rad_theory_j1_w": pr_j1,
            "p_rad_self_consistency_rel_diff": float(abs(pr_n2f - pr_j1) / pr_j1),
            "b1_over_a1": float(r_c["vsh"]["b1_over_a1"]),
            "p_a10_frac": float(r_c["vsh"]["p_a10_frac"]),
            "energy_closure_rel_err": float(r_c["energy_closure_rel_err"]),
        })

    # Grid convergence criterion: |Delta P_rad| / P_rad between the two finest levels < 2.0%.
    # Because the unsmoothed binary staircase of the circular cross-section and 8 angular sectors
    # oscillates by +/-9% in discrete drive volume between node-aligned and cell-centered grids,
    # |T_z^sim| (1.268 -> 1.279 -> 1.096e-6 A m^3) and P_rad (2.391 -> 2.435 -> 1.788e-4 W, -26.6% on
    # the finest grid vs medium, exceeding the 2.0% threshold) are NOT grid-converged on these 4-6 cells/2r
    # grids, and Richardson extrapolation is not applicable.
    pr_nom = float(conv_records[0]["p_rad_n2f_w"])
    pr_med = float(conv_records[1]["p_rad_n2f_w"])
    pr_fine = float(conv_records[-1]["p_rad_n2f_w"])
    p_rad_rel_change_fine_vs_med = float((pr_fine - pr_med) / max(pr_med, 1e-30))
    p_rad_rel_change_fine_vs_nom = float((pr_fine - pr_nom) / max(pr_nom, 1e-30))
    grid_conv_criterion_rel_dp_max = 0.02
    is_grid_converged = bool(abs(p_rad_rel_change_fine_vs_med) < grid_conv_criterion_rel_dp_max)
    mu_f0_real = float(np.real(make_core_material(mu_dc=MU_DC_DEFAULT).mu(F0_HZ)))
    kc_r_f0 = float(K0_RAD_M * np.sqrt(max(mu_f0_real, 1.0)) * R_MINOR_M)
    analytic_interior_f0 = float(1.0 + 0.125 * (kc_r_f0**2))

    # Also estimate an ultra-fine uniform production run (dx = 1.25 mm everywhere) for the report
    est_prod = estimate(cells=(240, 240, 200), steps=1600, backend="cpu", cpml=True, n_poles=1)

    data = {
        "loss_sweep": loss_records,
        "convergence": {
            "levels": conv_records,
            "tz_exact_continuum_am3": tz_exact,
            "analytic_interior_factor_f0": analytic_interior_f0,
            "tz_rel_err_levels": [float(abs(r["tz_tot_abs_am3"] - tz_exact) / tz_exact) for r in conv_records],
            "tz_disc_static_rel_err_levels": [
                float(abs(r["tz_disc_static_drive_am3"] - tz_exact) / tz_exact) for r in conv_records
            ],
            "p_rad_rel_change_fine_vs_med": p_rad_rel_change_fine_vs_med,
            "p_rad_rel_change_fine_vs_nom": p_rad_rel_change_fine_vs_nom,
            "grid_convergence_criterion_rel_dp_max": grid_conv_criterion_rel_dp_max,
            "is_grid_converged": is_grid_converged,
        },
        "production_batch_estimate": {
            "description": "Un-graded uniform 1.25 mm 3D FIT grid (240x240x200 = 11.52M cells, 1600 steps)",
            "cells": [240, 240, 200],
            "steps": 1600,
            "host_ram_gb": float(est_prod.host_bytes / 1e9),
            "est_cpu_seconds": float(est_prod.seconds),
            "command": (
                "/root/computational-electromagnetics/bin/memguard -m 4000 -- "
                "/root/computational-electromagnetics/wakis/.venv/bin/python "
                "tutorials/04_radiation_and_antennas/04-07_magnetic_toroid/magnetic_toroid.py --full"
            ),
        },
    }
    (out_dir / "task_06_efficiency_q_and_convergence.json").write_text(json.dumps(_json_clean(data), indent=2) + "\n")
    return data


def compute_all_study_data(
    quick: bool = True,
    out_dir: Path = DATA_DIR,
    selected_task: int | str | None = None,
) -> dict[str, Any]:
    """Run all 6 study tasks (or a selected task) sequentially one FIT simulation at a time and write summary JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_map = {
        1: ("01_baseline_dipoles", run_task_01_baseline_dipoles, "task_01_baseline_dipoles.json"),
        2: ("02_closed_toroid_multipoles", run_task_02_closed_toroid_multipoles, "task_02_closed_toroid.json"),
        3: ("03_square_four_rods", run_task_03_square_four_rods, "task_03_square_four_rods.json"),
        4: ("04_break_the_loop", run_task_04_break_the_loop, "task_04_break_the_loop.json"),
        5: ("05_equivalence_limits", run_task_05_equivalence_limits, "task_05_equivalence_limits.json"),
        6: (
            "06_efficiency_q_and_convergence",
            run_task_06_efficiency_q_and_convergence,
            "task_06_efficiency_q_and_convergence.json",
        ),
    }

    target_ids: list[int] = []
    if selected_task is None:
        target_ids = [1, 2, 3, 4, 5, 6]
    else:
        s_str = str(selected_task).strip().lower()
        for tid, (tname, _, _) in task_map.items():
            if s_str in (str(tid), tname.lower(), f"task_{tid:02d}"):
                target_ids.append(tid)
        if not target_ids:
            raise ValueError(f"Unknown task selector {selected_task!r}")

    results: dict[str, Any] = {}
    for tid, (tname, fn, fname) in task_map.items():
        fpath = out_dir / fname
        if tid in target_ids:
            results[tname] = fn(quick=quick, out_dir=out_dir)
        elif fpath.exists():
            results[tname] = json.loads(fpath.read_text())

    if len(results) == 6:
        results["summary"] = build_summary_json(data_dir=out_dir)

    return results


# ==============================================================================
# 5. Setup schematics, diagnostic figures, field maps & GIF generator
# ==============================================================================


def _mag_box(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    mu_r: float,
    eps_r: float = 1.0,
) -> Box:
    """Create a Box carrying ``[eps_r, mu_r]`` metadata for `wakis.schematic`."""
    b = Box(center, size, eps_r)
    b.material = [float(eps_r), float(mu_r)]
    return b


def plot_00_setup_figures(fig_dir: Path = FIG_DIR) -> dict[str, Any]:
    """Render 00a/00b setup schematics and Markdown setup tables for all 3 FIT geometries."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dx_core = 0.0025
    core_xy_half = max(0.0225, R_MAJOR_M + R_MINOR_M + 2.0 * dx_core)
    core_z_half = max(0.0075, R_MINOR_M + dx_core)
    n2f_xy_half = max(0.055, core_xy_half + 0.025)
    n2f_z_half = max(0.045, core_z_half + 0.030)
    dom_xy_half = max(0.150, n2f_xy_half + 0.095)
    dom_z_half = max(0.125, n2f_z_half + 0.080)
    grid = build_toroid_grid(
        dx_core_m=dx_core,
        dx_vac_m=0.015,
        core_xy_half_m=core_xy_half,
        core_z_half_m=core_z_half,
        n2f_xy_half_m=n2f_xy_half,
        n2f_z_half_m=n2f_z_half,
        domain_xy_half_m=dom_xy_half,
        domain_z_half_m=dom_z_half,
        graded=True,
    )
    solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
    core_mat = make_core_material(mu_dc=MU_DC_DEFAULT)
    mu_p = float(np.real(core_mat.mu(F0_HZ)))
    pulse = GaussianSource(frequency=F0_HZ, fwidth=0.22e9)

    n2f_mon = Near2FarMonitor(frequencies=[F0_HZ], center=(0.0, 0.0, 0.0), size=(0.110, 0.110, 0.090))
    dft_mon = DFTMonitor(
        frequencies=[F0_HZ],
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        center=(0.0, 0.0, 0.0),
        size=(0.045, 0.045, 0.015),
    )
    src_pol = CurrentSource(pulse, component="z", center=(0.0, 0.0, 0.0), size=(0.030, 0.030, 0.010))

    # 1. Toroid XZ cross-section (shows the two minor-radius circles/boxes at x = -R and x = +R)
    structs_xz = [
        _mag_box((-R_MAJOR_M, 0.0, 0.0), (2.0 * R_MINOR_M, 2.0 * R_MINOR_M, 2.0 * R_MINOR_M), mu_p),
        _mag_box((+R_MAJOR_M, 0.0, 0.0), (2.0 * R_MINOR_M, 2.0 * R_MINOR_M, 2.0 * R_MINOR_M), mu_p),
    ]
    mat_names_xz = {
        "region_0": "Lorentz magnetic torus (-x cross-section, μ_DC = 20)",
        "region_1": "Lorentz magnetic torus (+x cross-section, μ_DC = 20)",
    }
    solver._assigned_structures = structs_xz
    res_xz = schematic.save_setup_figures(
        solver,
        fig_dir,
        plane="xz",
        cut_coord=0.0,
        structures=structs_xz,
        material_names=mat_names_xz,
        sources=[src_pol],
        monitors=[n2f_mon, dft_mon],
        unit="mm",
        wavelength=LAM0_M,
        freq=F0_HZ,
        min_feature=2.0 * R_MINOR_M,
        title_geometry="Dispersive magnetic toroid (R = 15 mm, r = 5 mm, N = 8 windings, XZ cut)",
        variations=[
            {"name": "Nominal closed torus", "details": "N = 8 in-phase poloidal windings, μ_DC = 20, Δ_core = 2.5 mm"},
            {"name": "Ideal magnetic current ring", "details": "Azimuthal MagneticCurrentSource ring at ρ = R"},
            {"name": "Broken-symmetry counter-cases", "details": "60° core gap, 7/8 feeds, Mode 1 antiphase, helical turn"},
        ],
    )

    # 2. Toroid XY equatorial view
    structs_xy = [
        _mag_box((0.0, 0.0, 0.0), (2.0 * (R_MAJOR_M + R_MINOR_M), 2.0 * (R_MAJOR_M + R_MINOR_M), 2.0 * R_MINOR_M), mu_p),
    ]
    mat_names_xy = {
        "region_0": "Lorentz magnetic torus ring (R = 15 mm, r = 5 mm, μ_DC = 20)",
    }
    solver._assigned_structures = structs_xy
    res_xy = schematic.save_setup_figures(
        solver,
        fig_dir,
        suffix="_top",
        plane="xy",
        cut_coord=0.0,
        structures=structs_xy,
        material_names=mat_names_xy,
        sources=[src_pol],
        monitors=[n2f_mon, dft_mon],
        unit="mm",
        wavelength=LAM0_M,
        freq=F0_HZ,
        min_feature=2.0 * R_MINOR_M,
        title_geometry="Dispersive magnetic toroid with 8 poloidal feeds (XY equatorial plane)",
    )

    # 3. Four-rod C4 square XY view
    r_sq = 0.0125
    w_rod = 0.005
    l_rod = 2.0 * (r_sq + w_rod)
    structs_rods = [
        _mag_box((+r_sq, 0.0, 0.0), (2.0 * w_rod, l_rod, 2.0 * w_rod), mu_p),
        _mag_box((0.0, +r_sq, 0.0), (l_rod, 2.0 * w_rod, 2.0 * w_rod), mu_p),
        _mag_box((-r_sq, 0.0, 0.0), (2.0 * w_rod, l_rod, 2.0 * w_rod), mu_p),
        _mag_box((0.0, -r_sq, 0.0), (l_rod, 2.0 * w_rod, 2.0 * w_rod), mu_p),
    ]
    mat_names_rods = {
        "region_0": "Permeable rod 1 (+x, +y magnetization)",
        "region_1": "Permeable rod 2 (+y, -x magnetization)",
        "region_2": "Permeable rod 3 (-x, -y magnetization)",
        "region_3": "Permeable rod 4 (-y, +x magnetization)",
    }
    solver._assigned_structures = structs_rods
    res_rods = schematic.save_setup_figures(
        solver,
        fig_dir,
        suffix="_rods",
        plane="xy",
        cut_coord=0.0,
        structures=structs_rods,
        material_names=mat_names_rods,
        sources=[src_pol],
        monitors=[n2f_mon, dft_mon],
        unit="mm",
        wavelength=LAM0_M,
        freq=F0_HZ,
        min_feature=2.0 * w_rod,
        title_geometry="C4 square of four permeable rods (R_sq = 12.5 mm, w = 10 mm, XY plane)",
    )

    # 4. Baseline PEC wire loop with LumpedPort XY view
    a_loop = 0.0125
    structs_loop = [
        Box((0.0, +a_loop, 0.0), (2.0 * a_loop, 0.0025, 0.0025), "pec"),
        Box((-a_loop, 0.0, 0.0), (0.0025, 2.0 * a_loop, 0.0025), "pec"),
        Box((+a_loop, 0.0, 0.0), (0.0025, 2.0 * a_loop, 0.0025), "pec"),
        Box((0.0, -a_loop, 0.0), (2.0 * a_loop, 0.0025, 0.0025), "pec"),
    ]
    mat_names_loop = {
        "region_0": "PEC wire loop top segment",
        "region_1": "PEC wire loop left segment",
        "region_2": "PEC wire loop right segment",
        "region_3": "PEC wire loop bottom feed segment",
    }
    port_loop = LumpedPort(
        pulse,
        component="x",
        center=(0.0, -a_loop, 0.0),
        size=(0.005, 0.0, 0.0),
        impedance=50.0,
    )
    solver._assigned_structures = structs_loop
    res_loop = schematic.save_setup_figures(
        solver,
        fig_dir,
        suffix="_loop",
        plane="xy",
        cut_coord=0.0,
        structures=structs_loop,
        material_names=mat_names_loop,
        sources=[port_loop],
        monitors=[n2f_mon],
        unit="mm",
        wavelength=LAM0_M,
        freq=F0_HZ,
        min_feature=0.010,
        title_geometry="Baseline single PEC wire loop driven by 50-Ω LumpedPort (XY plane)",
    )
    plt.close("all")
    return {"xz": res_xz, "xy": res_xy, "rods": res_rods, "loop": res_loop}


def plot_01_baseline_dipoles(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> Path:
    """Plot Figure 1: Baseline electric dipole (p_z) vs magnetic loop (m_z) polarization and (ka)^4 scaling."""
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    t1 = json.loads((Path(data_dir) / "task_01_baseline_dipoles.json").read_text())
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=11.2, height_in=3.4, constrained_layout=True)
    ax0, ax1, ax2 = axes

    theta_deg = np.degrees(np.asarray(t1["theta_rad"], dtype=float))
    sin_th = np.sin(np.asarray(t1["theta_rad"], dtype=float))

    # Panel (a): Short electric dipole p_z -> E_theta ~ sin(theta), E_phi = 0
    ax0.plot(theta_deg, sin_th, "k--", lw=1.5, label=r"Analytic $\sin\theta$")
    ax0.plot(theta_deg, t1["etheta_edip_norm"], **figstyle.series_style(0, markevery=12), label=r"FIT $|E_\theta|$ (TM $a_{10}$)")
    ax0.plot(theta_deg, t1["ephi_edip_norm"], **figstyle.series_style(1, markevery=12), label=r"FIT $|E_\phi|$ (cross-pol)")
    ax0.set_xlabel(r"Polar angle $\theta$ [deg]")
    ax0.set_ylabel("Normalized far field")
    ax0.set_title(r"(a) Electric dipole $\mathbf{p}_z$", fontsize=10.5)
    ax0.set_xlim(0.0, 180.0)
    ax0.set_ylim(-0.05, 1.15)
    figstyle.spectrum_grid(ax0)
    ax0.legend(loc="lower center", fontsize=8.6)

    # Panel (b): Small magnetic loop m_z -> E_phi ~ sin(theta), E_theta = 0
    ax1.plot(theta_deg, sin_th, "k--", lw=1.5, label=r"Analytic $\sin\theta$")
    ax1.plot(theta_deg, t1["ephi_loop_norm"], **figstyle.series_style(2, markevery=12), label=r"FIT $|E_\phi|$ (TE $b_{10}$)")
    ax1.plot(theta_deg, t1["etheta_loop_norm"], **figstyle.series_style(3, markevery=12), label=r"FIT $|E_\theta|$ (cross-pol)")
    ax1.set_xlabel(r"Polar angle $\theta$ [deg]")
    ax1.set_ylabel("Normalized far field")
    ax1.set_title(r"(b) Single loop $\mathbf{m}_z$ ($E_\phi$ pol)", fontsize=10.5)
    ax1.set_xlim(0.0, 180.0)
    ax1.set_ylim(-0.05, 1.15)
    figstyle.spectrum_grid(ax1)
    ax1.legend(loc="lower center", fontsize=8.6)

    # Panel (c): Log-log (ka)^4 radiation power / resistance scaling
    ka_arr = np.asarray(t1["ka4_scaling"]["ka"], dtype=float)
    p_sim = np.asarray(t1["ka4_scaling"]["p_rad_sim_w"], dtype=float)
    p_norm_sim = p_sim / p_sim[0]
    ka_dense = np.geomspace(ka_arr[0] * 0.92, ka_arr[-1] * 1.08, 50)
    p_norm_th = (ka_dense / ka_arr[0]) ** 4

    ax2.loglog(ka_dense, p_norm_th, "k--", lw=1.6, label=r"Analytic $(k_0 a)^4$ guide")
    ax2.loglog(
        ka_arr,
        p_norm_sim,
        **figstyle.series_style(0, with_line=False, ms=6.5),
        label=f"3D FIT (slope = {t1['ka4_scaling']['fitted_slope']:.2f})",
    )
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xticks(ka_arr)
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax2.set_xlabel(r"Electrical loop radius $k_0 a$")
    ax2.set_ylabel(r"Normalized $P_{\mathrm{rad}}(a) / P_{\mathrm{rad}}(a_0)$")
    ax2.set_title(r"(c) Loop $(k_0 a)^4$ radiation scaling", fontsize=10.5)
    figstyle.spectrum_grid(ax2)
    ax2.legend(loc="upper left", fontsize=8.6)

    return figstyle.save(fig, Path(fig_dir) / "01_baseline_dipoles_and_ka4.png", close=True, auto_legend_below=False)


def plot_02_closed_toroid_fields(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> Path:
    """Plot Figure 2: Closed azimuthal magnetization loop M_phi(x,y) inside the torus and poloidal E(x,z) outside."""
    cuts = np.load(Path(data_dir) / "task_02_toroid_cuts.npz")
    t2 = json.loads((Path(data_dir) / "task_02_closed_toroid.json").read_text())

    grid = build_toroid_grid(dx_core_m=0.0025, graded=True)
    solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
    mu_p = float(np.real(make_core_material(mu_dc=MU_DC_DEFAULT).mu(F0_HZ)))

    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width_in=8.8, height_in=3.5, constrained_layout=True)
    ax0, ax1 = axes

    # Panel (a): Equatorial XY plane: |H_xy| and closed azimuthal magnetization vectors M_phi
    x_xy_mm = cuts["x_xy_m"] * 1e3
    y_xy_mm = cuts["y_xy_m"] * 1e3
    hx_xy = cuts["hx_xy"]
    hy_xy = cuts["hy_xy"]
    mx_mask = cuts["mx_mask_xy"]
    my_mask = cuts["my_mask_xy"]
    nx_c = min(hx_xy.shape[0], hy_xy.shape[0], len(x_xy_mm), len(y_xy_mm))
    hx_c = np.real(hx_xy[:nx_c, :nx_c])
    hy_c = np.real(hy_xy[:nx_c, :nx_c])
    m_mask_c = mx_mask[:nx_c, :nx_c] | my_mask[:nx_c, :nx_c]
    # Bound magnetization M = (mu_r - 1) H inside the torus core
    mx_bound = np.where(m_mask_c, (mu_p - 1.0) * hx_c, 0.0)
    my_bound = np.where(m_mask_c, (mu_p - 1.0) * hy_c, 0.0)
    m_mag = np.sqrt(mx_bound**2 + my_bound**2)

    im0 = ax0.pcolormesh(
        x_xy_mm[:nx_c],
        y_xy_mm[:nx_c],
        m_mag.T,
        cmap=figstyle.CMAP_MAGNITUDE,
        shading="auto",
    )
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)
    cb0.set_label(r"$|\mathbf{M}(x, y, 0)|$ [A/m]", fontsize=9.0)

    # Quiver of closed azimuthal M vectors inside the torus ring
    stride = 2
    xx_q, yy_q = np.meshgrid(x_xy_mm[:nx_c:stride], y_xy_mm[:nx_c:stride], indexing="ij")
    mx_q = mx_bound[::stride, ::stride]
    my_q = my_bound[::stride, ::stride]
    mask_q = m_mag[::stride, ::stride] > 0.15 * float(np.max(m_mag))
    ax0.quiver(
        xx_q[mask_q],
        yy_q[mask_q],
        mx_q[mask_q],
        my_q[mask_q],
        color="#ffffff",
        scale=float(np.max(m_mag)) * 9.5,
        width=0.0055,
        zorder=8,
    )
    # Draw exact inner and outer torus radii R - r = 10 mm and R + r = 20 mm
    for rad_mm, lbl in [(10.0, None), (20.0, "Torus core")]:
        circ = Circle((0.0, 0.0), rad_mm, fill=False, edgecolor="#38bdf8", ls="--", lw=1.2, zorder=9, label=lbl)
        ax0.add_patch(circ)
    ax0.set_xlim(-32.0, 32.0)
    ax0.set_ylim(-32.0, 32.0)
    ax0.set_aspect("equal")
    ax0.set_xlabel(r"$x$ [mm]")
    ax0.set_ylabel(r"$y$ [mm]")
    ax0.set_title(
        f"(a) Closed loop $\\mathbf{{M}}_\\phi$ ($|\\int\\mathbf{{M}}dV|/\\int|\\mathbf{{M}}|dV = {t2['moments']['m_cancel_ratio']:.1e}$)",
        fontsize=10.2,
    )

    # Panel (b): Meridional XZ plane: Re(E_z) showing axial electric-dipole-like radiation + overlay_geometry
    x_xz_mm = cuts["x_xz_m"] * 1e3
    z_xz_mm = cuts["z_xz_m"] * 1e3
    ez_real = np.real(cuts["ez_xz"])
    vmin, vmax = figstyle.symmetric_limits(ez_real, percentile=97.0)
    im1 = ax1.pcolormesh(
        x_xz_mm,
        z_xz_mm,
        ez_real.T,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cb1.set_label(r"$\mathrm{Re}\{E_z(x, 0, z)\}$ [V/m]", fontsize=9.0)

    structs_xz = [
        _mag_box((-R_MAJOR_M, 0.0, 0.0), (2.0 * R_MINOR_M, 2.0 * R_MINOR_M, 2.0 * R_MINOR_M), mu_p),
        _mag_box((+R_MAJOR_M, 0.0, 0.0), (2.0 * R_MINOR_M, 2.0 * R_MINOR_M, 2.0 * R_MINOR_M), mu_p),
    ]
    schematic.overlay_geometry(
        ax1,
        solver,
        plane="xz",
        structures=structs_xz,
        material_names={"region_0": "Core (-x)", "region_1": "Core (+x)"},
        unit="mm",
        show_boundaries=False,
    )
    ax1.set_xlim(float(x_xz_mm[0]), float(x_xz_mm[-1]))
    ax1.set_ylim(float(z_xz_mm[0]), float(z_xz_mm[-1]))
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x$ [mm]")
    ax1.set_ylabel(r"$z$ [mm]")
    ax1.set_title(r"(b) Meridional $\mathrm{Re}\{E_z(x,0,z)\}$ & core cross-section", fontsize=10.2)

    return figstyle.save(fig, Path(fig_dir) / "02_closed_toroid_fields.png", close=True, auto_legend_below=False)


def plot_03_toroid_multipoles_and_scaling(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> Path:
    """Plot Figure 3: Spherical multipole spectrum (l <= 5), far-field pattern, and omega^6 & octupole (kR)^2 scaling."""
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    t2 = json.loads((Path(data_dir) / "task_02_closed_toroid.json").read_text())
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=11.2, height_in=3.4, constrained_layout=True)
    ax0, ax1, ax2 = axes

    # Panel (a): Spherical multipole spectrum l = 1..5 (electric a_l vs magnetic b_l)
    l_vals = np.arange(1, 6)
    p_a = np.maximum(np.asarray(t2["vsh_toroid_8feed"]["p_a_l_frac"], dtype=float), 1e-12)
    p_b = np.maximum(np.asarray(t2["vsh_toroid_8feed"]["p_b_l_frac"], dtype=float), 1e-12)
    w_bar = 0.34
    ax0.bar(l_vals - 0.5 * w_bar, p_a, width=w_bar, color=figstyle.PALETTE[0], label=r"Electric TM ($a_{lm}$)")
    ax0.bar(l_vals + 0.5 * w_bar, p_b, width=w_bar, color=figstyle.PALETTE[1], label=r"Magnetic TE ($b_{lm}$)")
    ax0.set_yscale("log")
    ax0.set_ylim(1e-12, 3.0)
    ax0.set_xticks(l_vals)
    ax0.set_xlabel(r"Multipole degree $l$")
    ax0.set_ylabel(r"Power fraction $P_l / P_{\mathrm{tot}}$")
    ax0.set_title(f"(a) Multipoles ($|b_1|/|a_1| = {t2['vsh_toroid_8feed']['b1_over_a1']:.2e}$)", fontsize=10.3)
    figstyle.spectrum_grid(ax0)
    ax0.legend(loc="upper right", fontsize=8.5)

    # Panel (b): Toroid far-field pattern vs analytic axial electric dipole sin(theta)
    theta_deg = np.degrees(np.asarray(t2["theta_rad"], dtype=float))
    sin_th = np.sin(np.asarray(t2["theta_rad"], dtype=float))
    ax1.plot(theta_deg, sin_th, "k--", lw=1.6, label=r"Analytic $\sin\theta$ ($p_{\mathrm{eff}}$)")
    ax1.plot(
        theta_deg,
        t2["etheta_toroid_norm"],
        **figstyle.series_style(0, markevery=12),
        label=r"Toroid $|E_\theta|$ (co-pol)",
    )
    ax1.plot(
        theta_deg,
        t2["ephi_toroid_norm"],
        **figstyle.series_style(1, markevery=12),
        label=r"Toroid $|E_\phi|$ (cross-pol)",
    )
    ax1.set_xlabel(r"Polar angle $\theta$ [deg]")
    ax1.set_ylabel("Normalized far field")
    ax1.set_title(r"(b) Pattern $E_\theta \propto \sin\theta$ ($E_\phi \approx 0$)", fontsize=10.3)
    ax1.set_xlim(0.0, 180.0)
    ax1.set_ylim(-0.05, 1.15)
    figstyle.spectrum_grid(ax1)
    ax1.legend(loc="lower center", fontsize=8.5)

    # Panel (c): Frequency scaling P_rad ~ omega^6 (fixed T) vs dispersive fixed-I and octupole |a_30|/|a_10| ~ (k_0 R)^2
    f_ghz = np.asarray(t2["frequency_scaling"]["freqs_ghz"], dtype=float)
    f_norm = f_ghz / float(F0_HZ / 1e9)
    p_fixed_t = np.asarray(t2["frequency_scaling"]["p_rad_fixed_t_point_corrected_w"], dtype=float)
    idx0 = int(np.argmin(np.abs(f_norm - 1.0)))
    p_norm_t = p_fixed_t / p_fixed_t[idx0]
    f_dense = np.geomspace(f_norm[0] * 0.96, f_norm[-1] * 1.04, 60)

    kr_arr = np.asarray(t2["octupole_scaling"]["k0_r"], dtype=float)
    kr_norm = kr_arr / kr_arr[1]
    a3_sim = np.asarray(t2["octupole_scaling"]["a30_over_a10_sim"], dtype=float)
    a3_norm = a3_sim / a3_sim[1]

    ax2.loglog(f_dense, f_dense**6, "k-", lw=1.5, label=r"$\propto (\omega/\omega_0)^6$ (fixed $T$)")
    ax2.loglog(f_dense, f_dense**4, color="#64748b", ls="--", lw=1.4, label=r"$\propto (\omega/\omega_0)^4$ (fixed $p$)")
    ax2.loglog(
        f_norm,
        p_norm_t,
        **figstyle.series_style(0, with_line=False, ms=5.8),
        label=f"FIT fixed-$T$ (slope {t2['frequency_scaling']['slope_omega_fixed_t_point_corrected']:.2f})",
    )
    p_fixed_i = np.asarray(t2["frequency_scaling"]["p_rad_fixed_i_w"], dtype=float)
    ax2.loglog(
        f_norm,
        p_fixed_i / p_fixed_i[idx0],
        **figstyle.series_style(3, ms=5.0),
        label=f"FIT fixed-$I$ (slope {t2['frequency_scaling']['slope_omega_fixed_i']:.2f})",
    )
    ax2.loglog(
        kr_norm,
        a3_norm,
        **figstyle.series_style(2, ms=5.8),
        label=f"FIT $|a_{{30}}/a_{{10}}|$ vs $R/R_0$ (slope {t2['octupole_scaling']['fitted_slope_vs_kr']:.2f})",
    )
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xticks([0.7, 1.0, 1.3, 1.6])
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax2.set_xlabel(r"Normalized $\omega/\omega_0$ or $R/R_0$")
    ax2.set_ylabel("Normalized power / octupole")
    ax2.set_title(r"(c) $\omega^6$ power & $(k_0 R)^2$ octupole", fontsize=10.3)
    figstyle.spectrum_grid(ax2)
    ax2.legend(loc="upper left", fontsize=7.8)

    return figstyle.save(fig, Path(fig_dir) / "03_toroid_multipoles_and_scaling.png", close=True, auto_legend_below=False)


def plot_04_square_four_rods(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> Path:
    """Plot Figure 4: C4 square of four permeable rods — magnetization loop and C4 residual scaling vs k_0 R_sq."""
    t3 = json.loads((Path(data_dir) / "task_03_square_four_rods.json").read_text())
    cuts_sq = np.load(Path(data_dir) / "task_03_four_rods_cuts.npz")

    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width_in=8.8, height_in=3.4, constrained_layout=True)
    ax0, ax1 = axes

    x_xy_mm = cuts_sq["x_xy_m"] * 1e3
    y_xy_mm = cuts_sq["y_xy_m"] * 1e3
    hx_xy = cuts_sq["hx_xy"]
    hy_xy = cuts_sq["hy_xy"]
    mx_mask = cuts_sq["mx_mask_xy"]
    my_mask = cuts_sq["my_mask_xy"]
    nx_c = min(hx_xy.shape[0], hy_xy.shape[0], len(x_xy_mm), len(y_xy_mm))
    mu_p = float(np.real(make_core_material(mu_dc=MU_DC_DEFAULT).mu(F0_HZ)))
    m_mask = mx_mask[:nx_c, :nx_c] | my_mask[:nx_c, :nx_c]
    mx_b = np.where(m_mask, (mu_p - 1.0) * np.real(hx_xy[:nx_c, :nx_c]), 0.0)
    my_b = np.where(m_mask, (mu_p - 1.0) * np.real(hy_xy[:nx_c, :nx_c]), 0.0)
    m_mag = np.sqrt(mx_b**2 + my_b**2)

    im0 = ax0.pcolormesh(x_xy_mm[:nx_c], y_xy_mm[:nx_c], m_mag.T, cmap=figstyle.CMAP_MAGNITUDE, shading="auto")
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)
    cb0.set_label(r"$|\mathbf{M}(x, y, 0)|$ [A/m]", fontsize=9.0)

    stride = 2
    xx_q, yy_q = np.meshgrid(x_xy_mm[:nx_c:stride], y_xy_mm[:nx_c:stride], indexing="ij")
    mx_q = mx_b[::stride, ::stride]
    my_q = my_b[::stride, ::stride]
    mask_q = m_mag[::stride, ::stride] > 0.15 * float(np.max(m_mag))
    ax0.quiver(
        xx_q[mask_q],
        yy_q[mask_q],
        mx_q[mask_q],
        my_q[mask_q],
        color="#ffffff",
        scale=float(np.max(m_mag)) * 9.5,
        width=0.0055,
        zorder=8,
    )
    ax0.set_xlim(-30.0, 30.0)
    ax0.set_ylim(-30.0, 30.0)
    ax0.set_aspect("equal")
    ax0.set_xlabel(r"$x$ [mm]")
    ax0.set_ylabel(r"$y$ [mm]")
    ax0.set_title(
        f"(a) $C_{{4v}}$ four-rod square ($|\\int\\mathbf{{M}}dV|/\\int|\\mathbf{{M}}|dV = {t3['records'][0]['m_cancel_ratio']:.1e}$)",
        fontsize=10.2,
    )

    # Panel (b): Growth of non-a10 multipole power and magnetic/C4 residuals vs electrical size k_0 R_sq
    recs = t3["records"]
    kr_sq = np.asarray([r["k0_r_sq"] for r in recs], dtype=float)
    non_a10 = np.asarray([r["non_a10_power_frac"] for r in recs], dtype=float)
    b_tot = np.asarray([r["b_tot_over_a_tot"] for r in recs], dtype=float)
    c4_m4 = np.asarray([r["c4_m4_over_a10"] for r in recs], dtype=float)

    ax1.semilogy(kr_sq, non_a10, **figstyle.series_style(0), label=r"Non-$a_{10}$ power $1 - P_{a_{10}}/P_{\mathrm{tot}}$")
    ax1.semilogy(kr_sq, b_tot, **figstyle.series_style(1), label=r"Magnetic-to-electric $\|b_{lm}\|/\|a_{lm}\|$")
    ax1.semilogy(kr_sq, c4_m4, **figstyle.series_style(2), label=r"$C_{4v}$ harmonic $|m=\pm 4| / |a_{10}|$")
    ax1.set_xlabel(r"Square electrical half-side $k_0 R_{\mathrm{sq}}$")
    ax1.set_ylabel("Multipole residual ratio")
    ax1.set_title(r"(b) $C_{4v}$ square residuals vs $k_0 R_{\mathrm{sq}}$", fontsize=10.2)
    figstyle.spectrum_grid(ax1)
    ax1.legend(loc="lower right", fontsize=8.4)

    return figstyle.save(fig, Path(fig_dir) / "04_square_four_rods.png", close=True, auto_legend_below=False)


def plot_05_break_the_loop(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> Path:
    """Plot Figure 5: Breaking the closed magnetization loop — |b_1|/|a_1| bar chart, TE E_phi pattern, and Marinov R_1 sweep."""
    t4 = json.loads((Path(data_dir) / "task_04_break_the_loop.json").read_text())

    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=11.4, height_in=3.5, constrained_layout=True)
    ax0, ax1, ax2 = axes

    # Panel (a): Horizontal log bar chart of |b_1|/|a_1| across all 8 configurations
    order_keys = [
        ("closed_8_feeds", "Closed 8-feed torus"),
        ("helical_marinov_compensated", "Helical + Marinov loop"),
        ("four_rods_closed", "Closed 4-rod square"),
        ("core_air_gap_60deg", "60° core air gap"),
        ("three_of_four_rods", "3 of 4 rods (1 removed)"),
        ("seven_of_eight_feeds", "7 of 8 feeds active"),
        ("helical_uncompensated", "Helical (1 net φ turn)"),
        ("mode1_antiphase", "Mode 1 (++ ++ -- --)"),
    ]
    y_pos = np.arange(len(order_keys))
    b1_ratios = [float(t4["cases"][k]["b1_over_a1"]) for k, _ in order_keys]
    colors = [
        figstyle.PALETTE[2] if r < 0.01 else figstyle.PALETTE[1]
        for r in b1_ratios
    ]
    ax0.barh(y_pos, b1_ratios, color=colors, height=0.62)
    ax0.axvline(1.0, color="k", ls="--", lw=1.1, label=r"$|b_1| = |a_1|$")
    ax0.set_xscale("log")
    ax0.set_xlim(1e-6, 1e7)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels([lbl for _, lbl in order_keys], fontsize=8.6)
    ax0.set_xlabel(r"Magnetic-to-electric dipole ratio $|b_1| / |a_1|$")
    ax0.set_title(r"(a) Symmetry breaking turns on $b_1$", fontsize=10.3)
    figstyle.spectrum_grid(ax0)

    # Panel (b): Far-field polarization ratio |E_phi| (TE magnetic) vs |E_theta| (TM electric) for closed vs broken
    theta_deg = np.degrees(np.asarray(t4["theta_rad"], dtype=float))
    c_closed = t4["cases"]["closed_8_feeds"]
    c_7of8 = t4["cases"]["seven_of_eight_feeds"]
    c_hel = t4["cases"]["helical_uncompensated"]
    c_mar = t4["cases"]["helical_marinov_compensated"]

    et_closed = np.asarray(c_closed["etheta_abs"]) / max(np.max(c_closed["etheta_abs"]), 1e-30)
    ep_7of8 = np.asarray(c_7of8["ephi_abs"]) / max(np.max(c_7of8["etheta_abs"]), 1e-30)
    ep_hel = np.asarray(c_hel["ephi_abs"]) / max(np.max(c_hel["etheta_abs"]), 1e-30)
    ep_mar = np.asarray(c_mar["ephi_abs"]) / max(np.max(c_mar["etheta_abs"]), 1e-30)

    ax1.plot(theta_deg, et_closed, "k--", lw=1.5, label=r"Closed $|E_\theta|$ ($\sin\theta$)")
    ax1.plot(theta_deg, ep_7of8, **figstyle.series_style(1, markevery=12), label=r"7/8 feeds $|E_\phi|/|E_\theta|_{\max}$")
    ax1.plot(theta_deg, ep_hel, **figstyle.series_style(3, markevery=12), label=r"Helical $|E_\phi|/|E_\theta|_{\max}$")
    ax1.plot(theta_deg, ep_mar, **figstyle.series_style(2, markevery=12), label=r"+ Marinov loop $|E_\phi|/|E_\theta|_{\max}$")
    ax1.set_xlabel(r"Polar angle $\theta$ [deg]")
    ax1.set_ylabel(r"Normalized field amplitude")
    ax1.set_title(r"(b) Appearance of TE $E_\phi$ radiation", fontsize=10.3)
    ax1.set_xlim(0.0, 180.0)
    ax1.set_ylim(-0.05, 1.58)
    figstyle.spectrum_grid(ax1)
    ax1.legend(loc="upper right", fontsize=7.8, framealpha=0.92)

    # Panel (c): Marinov Eq. 9 compensating loop radius sweep |b_1|/|a_1| vs R_loop (same current -I_turn, no rescaling)
    msw = t4["marinov_r_loop_sweep"]
    r_loop_mm = np.asarray(msw["r_loop_mm"], dtype=float)
    b1_sw = np.asarray(msw["b1_over_a1"], dtype=float)
    r1_mm = float(msw["r1_marinov_mm"])
    r_maj_mm = float(msw["r_major_mm"])
    b1_uncomp = float(msw["b1_over_a1_uncompensated"])
    r_dense_mm = np.linspace(r_loop_mm[0] * 0.98, r_loop_mm[-1] * 1.02, 200)
    b1_analytic_curve = np.maximum(b1_uncomp * np.abs(1.0 - (r_dense_mm / r1_mm) ** 2), float(msw["b1_over_a1_at_r1"]))

    ax2.semilogy(
        r_dense_mm,
        b1_analytic_curve,
        "k--",
        lw=1.3,
        label=r"Analytic $|1 - (R_{\mathrm{loop}}/R_1)^2|$",
    )
    ax2.semilogy(
        r_loop_mm,
        b1_sw,
        **figstyle.series_style(0, ms=6.2),
        label=r"FIT wire ($-I_{\mathrm{turn}}$, no rescaling)",
    )
    ax2.axvline(r_maj_mm, color="#64748b", ls=":", lw=1.2, label=rf"Major $R = {r_maj_mm:.1f}\;\mathrm{{mm}}$")
    ax2.axvline(
        r1_mm,
        color=figstyle.PALETTE[2],
        ls="-.",
        lw=1.4,
        label=rf"Marinov $R_1 = {r1_mm:.2f}\;\mathrm{{mm}}$",
    )
    ax2.set_ylim(1.5e-5, 1.2)
    ax2.set_xlabel(r"Compensating loop radius $R_{\mathrm{loop}}$ [mm]")
    ax2.set_ylabel(r"Parasitic ratio $|b_1| / |a_1|$")
    ax2.set_title(r"(c) Marinov loop minimum at $R_1 = R\sqrt{1+\frac{1}{2}(r/R)^2}$", fontsize=10.1)
    figstyle.spectrum_grid(ax2)
    ax2.legend(loc="lower left", fontsize=7.2, framealpha=0.92)

    return figstyle.save(fig, Path(fig_dir) / "05_break_the_loop.png", close=True, auto_legend_below=False)


def plot_06_equivalence_limits(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> Path:
    """Plot Figure 6: Where equivalence ends — near field vs Hertzian dipole, electrical size zero, and core TE01 mode."""
    t5 = json.loads((Path(data_dir) / "task_05_equivalence_limits.json").read_text())
    nf = np.load(Path(data_dir) / "task_05_nearfield_vs_dipole.npz")

    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=11.4, height_in=3.4, constrained_layout=True)
    ax0, ax1, ax2 = axes

    x_mm = nf["x_xz_m"] * 1e3
    z_mm = nf["z_xz_m"] * 1e3
    e_tor = np.log10(np.maximum(np.sqrt(np.abs(nf["ex_tor_xz"]) ** 2 + np.abs(nf["ez_tor_xz"]) ** 2), 1e-6))
    e_dip = np.log10(np.maximum(np.sqrt(np.abs(nf["ex_dip_xz"]) ** 2 + np.abs(nf["ez_dip_xz"]) ** 2), 1e-6))
    vmin = float(np.percentile(e_tor, 10))
    vmax = float(np.percentile(e_tor, 99))

    # Panel (a): Magnetic toroid near-field log10(|E|) in XZ plane
    im0 = ax0.pcolormesh(x_mm, z_mm, e_tor.T, cmap=figstyle.CMAP_MAGNITUDE, vmin=vmin, vmax=vmax, shading="auto")
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)
    cb0.set_label(r"$\log_{10}|\mathbf{E}_{\mathrm{toroid}}|$ [V/m]", fontsize=8.8)
    for sign in (-1.0, 1.0):
        circ = Circle((sign * 15.0, 0.0), 5.0, fill=False, edgecolor="#38bdf8", ls="--", lw=1.3)
        ax0.add_patch(circ)
    ax0.set_aspect("equal")
    ax0.set_xlabel(r"$x$ [mm]")
    ax0.set_ylabel(r"$z$ [mm]")
    ax0.set_title(r"(a) Magnetic toroid near field $|\mathbf{E}|$", fontsize=10.3)

    # Panel (b): Equivalent Hertzian dipole near-field log10(|E|) with identical far-field p_eff
    im1 = ax1.pcolormesh(x_mm, z_mm, e_dip.T, cmap=figstyle.CMAP_MAGNITUDE, vmin=vmin, vmax=vmax, shading="auto")
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cb1.set_label(r"$\log_{10}|\mathbf{E}_{\mathrm{dipole}}|$ [V/m]", fontsize=8.8)
    ax1.plot([0.0], [0.0], marker="o", color="#38bdf8", ms=6.0, label=r"$p_{\mathrm{eff}}$ origin")
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x$ [mm]")
    ax1.set_ylabel(r"$z$ [mm]")
    ax1.set_title(r"(b) Hertzian dipole $|\mathbf{E}|$ (same $p_{\mathrm{eff}}$)", fontsize=10.3)

    # Panel (c): Electrical size breakdown [3 j_1(kR)/(kR)]^2 zero (bottom axis: k_0 R)
    # & TE01 core resonance enhancement (twin top axis: 2r / lambda_core)
    kr_c = np.asarray(t5["electrical_size_theory"]["kr_curve"], dtype=float)
    ff_sq = np.maximum(np.asarray(t5["electrical_size_theory"]["form_factor_sq"], dtype=float), 1e-6)
    te_recs = t5["core_te01_mode_sweep"]
    two_r_lam = np.asarray([r["two_r_over_lambda_core"] for r in te_recs], dtype=float)
    ratio_qs = np.asarray([r["ratio_to_eps1"] for r in te_recs], dtype=float)

    (ln_ff,) = ax2.semilogy(
        kr_c, ff_sq, **figstyle.series_style(0, with_marker=False), label=r"Analytic $[3 j_1(k_0 R)/(k_0 R)]^2$"
    )
    ln_null = ax2.axvline(
        float(t5["electrical_size_theory"]["kr_first_bessel_zero"]),
        color=figstyle.PALETTE[0],
        ls="--",
        lw=1.2,
        label=r"Null $k_0 R = 4.493$",
    )
    ax2.set_xlim(0.0, 5.2)
    ax2.set_ylim(2e-7, 6.0)
    ax2.set_xlabel(r"Ring electrical size $k_0 R$ (bottom curve)")
    ax2.set_ylabel("Normalized radiation / moment factor")
    figstyle.spectrum_grid(ax2)

    ax2_top = ax2.twiny()
    ax2_top._is_inset = True
    ax2_top.spines["right"].set_visible(False)
    ax2_top.spines["left"].set_visible(False)
    ax2_top.spines["bottom"].set_visible(False)
    ax2_top.spines["top"].set_visible(True)
    ax2_top.spines["top"].set_edgecolor(figstyle.INK)
    (ln_te,) = ax2_top.semilogy(
        two_r_lam,
        ratio_qs,
        **figstyle.series_style(1, ms=6.0),
        label=r"Core $|T_z(\varepsilon)/T_z(1)|$ (top axis)",
    )
    ln_cut = ax2_top.axvline(
        0.5,
        color=figstyle.PALETTE[1],
        ls=":",
        lw=1.2,
        label=r"$\mathrm{TE}_{01\delta}$ cutoff $2r/\lambda_{\mathrm{core}} = 0.5$",
    )
    ax2_top.set_xlim(0.0, 0.62)
    ax2_top.set_xlabel(r"Core thickness $2r / \lambda_{\mathrm{core}}$ (top markers)", fontsize=8.8)
    ax2_top.tick_params(axis="x", labelsize=8.0)

    lines_c = [ln_ff, ln_null, ln_te, ln_cut]
    labels_c = [ln.get_label() for ln in lines_c]
    ax2.legend(lines_c, labels_c, loc="lower left", fontsize=7.0, framealpha=0.92)

    return figstyle.save(fig, Path(fig_dir) / "06_equivalence_limits.png", close=True, auto_legend_below=False)


def plot_07_efficiency_q_and_convergence(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> Path:
    """Plot Figure 7: Core loss tangent sweep (energy bookkeeping, radiation efficiency, Q vs Chu) and mesh convergence."""
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    t6 = json.loads((Path(data_dir) / "task_06_efficiency_q_and_convergence.json").read_text())
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=11.4, height_in=3.4, constrained_layout=True)
    ax0, ax1, ax2 = axes

    loss_recs = t6["loss_sweep"]
    tan_d = np.asarray([r["tan_delta_m"] for r in loss_recs], dtype=float)
    eta_sim = np.asarray([r["eta_rad_sim"] for r in loss_recs], dtype=float)
    err_exact = np.asarray([r["energy_closure_rel_err"] * 100.0 for r in loss_recs], dtype=float)
    err_mut = np.asarray([r["energy_closure_mutated_rel_err"] * 100.0 for r in loss_recs], dtype=float)

    # Panel (a): Radiation efficiency eta_rad vs core magnetic loss tangent tan(delta_m)
    tan_dense = np.geomspace(tan_d[0] * 0.8, tan_d[-1] * 1.2, 60)
    mu_p_ref = float(loss_recs[0]["mu_real"])
    eta_dense = [
        toroidal_efficiency_and_q_theory(F0_HZ, R_MAJOR_M, R_MINOR_M, mu_real=mu_p_ref, tan_delta_m=float(td))["eta_rad"]
        for td in tan_dense
    ]
    ax0.loglog(tan_dense, eta_dense, "k--", lw=1.6, label=r"Analytic $\eta_{\mathrm{rad}}(\tan\delta_m)$")
    ax0.loglog(tan_d, eta_sim, **figstyle.series_style(0, with_line=False, ms=6.5), label=r"3D FIT $P_{\mathrm{rad}}/(P_{\mathrm{rad}}+P_{\mathrm{loss}})$")
    ax0.set_xlabel(r"Core magnetic loss tangent $\tan\delta_m = \mu''/\mu'$")
    ax0.set_ylabel(r"Radiation efficiency $\eta_{\mathrm{rad}}$")
    ax0.set_title(r"(a) Radiation efficiency vs $\tan\delta_m$", fontsize=10.3)
    figstyle.spectrum_grid(ax0)
    ax0.legend(loc="lower left", fontsize=8.3)

    # Panel (b): 3D FIT Energy bookkeeping closure error vs half-step phase mutation
    ax1.loglog(
        tan_d,
        err_exact,
        **figstyle.series_style(2, ms=6.0),
        label=r"Time-centered $\cos(\omega_0\Delta t/2)$ ($<0.08\%$)",
    )
    ax1.loglog(
        tan_d,
        err_mut,
        **figstyle.series_style(1, ms=6.0),
        label=r"Mutated $e^{+i\omega_0\Delta t/2}$ shift ($>34\%$)",
    )
    ax1.set_xlabel(r"Core loss tangent $\tan\delta_m$")
    ax1.set_ylabel(r"Energy balance error $|P_{\mathrm{in}} - P_{\mathrm{out}}|/P_{\mathrm{in}}$ [%]")
    ax1.set_title(r"(b) FIT energy bookkeeping closure", fontsize=10.3)
    figstyle.spectrum_grid(ax1)
    ax1.legend(loc="center left", fontsize=8.3)

    # Panel (c): 3-level mesh resolution study across Delta_core in {2.50, 2.00, 1.67} mm (>= 4 cells across 2r)
    conv = t6["convergence"]
    dx_mm = np.asarray([r["dx_core_mm"] for r in conv["levels"]], dtype=float)
    tz_disc_err = np.asarray(conv["tz_disc_static_rel_err_levels"], dtype=float)
    tz_sim_err = np.asarray(conv["tz_rel_err_levels"], dtype=float)
    pr_vs_peff = np.asarray(
        [r["p_rad_self_consistency_rel_diff"] for r in conv["levels"]],
        dtype=float,
    )
    dx_guide = np.geomspace(dx_mm[-1] * 0.92, dx_mm[0] * 1.08, 40)
    guide_2nd = pr_vs_peff[-1] * ((dx_guide / dx_mm[-1]) ** 2)

    ax2.loglog(dx_guide, guide_2nd, "k--", lw=1.5, label=r"$\mathcal{O}(\Delta_{\mathrm{core}}^2)$ (reference slope only)")
    ax2.loglog(
        dx_mm,
        tz_disc_err,
        **figstyle.series_style(0, ms=6.2),
        label=r"Drive staircase $|T_z^{\mathrm{disc}} - T_z^{\mathrm{qs}}| / T_z^{\mathrm{qs}}$",
    )
    ax2.loglog(
        dx_mm,
        tz_sim_err,
        **figstyle.series_style(2, ms=6.2),
        label=r"Dynamic $|T_z^{\mathrm{sim}} - T_z^{\mathrm{qs}}| / T_z^{\mathrm{qs}}$",
    )
    ax2.loglog(
        dx_mm,
        pr_vs_peff,
        **figstyle.series_style(1, ms=5.8),
        label=r"Self-consistency $|P_{\mathrm{rad}} - P_{\mathrm{dip}}(T_{\mathrm{sim}})| / P_{\mathrm{dip}}$",
    )
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xticks(dx_mm)
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.set_ylim(1.5e-5, 0.45)
    ax2.set_xlabel(r"Core cell size $\Delta_{\mathrm{core}}$ [mm]")
    ax2.set_ylabel("Relative offset / self-consistency")
    ax2.set_title(r"(c) 3D FIT grid study (4–6 cells/$2r$)", fontsize=10.3)
    figstyle.spectrum_grid(ax2)
    ax2.legend(loc="lower left", fontsize=7.2, framealpha=0.92)

    return figstyle.save(fig, Path(fig_dir) / "07_efficiency_q_and_convergence.png", close=True, auto_legend_below=False)


def plot_08_toroid_vs_dipole_wave_gif(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> Path:
    """Render an animated GIF of time-harmonic H_y(x, 0, z, t) comparing the magnetic toroid and equivalent electric dipole."""
    figstyle.apply_style()
    nf = np.load(Path(data_dir) / "task_05_nearfield_vs_dipole.npz")
    x_mm = nf["x_xz_m"] * 1e3
    z_mm = nf["z_xz_m"] * 1e3
    hy_tor = nf["hy_tor_xz"]
    hy_dip = nf["hy_dip_xz"]

    # Scale outside the torus ring so radiated wavefronts in the outer region are clearly visible
    xx, zz = np.meshgrid(x_mm, z_mm, indexing="ij")
    r_mm = np.sqrt(xx**2 + zz**2)
    outer_mask = r_mm >= 24.0
    vmax = float(np.percentile(np.abs(hy_tor[outer_mask]), 96))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 3.1), dpi=130)
    fig.patch.set_facecolor(figstyle.PAPER)

    n_frames = 20
    phases = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)

    im0 = ax0.pcolormesh(
        x_mm,
        z_mm,
        np.real(hy_tor).T,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    im1 = ax1.pcolormesh(
        x_mm,
        z_mm,
        np.real(hy_dip).T,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    for sign in (-1.0, 1.0):
        ax0.add_patch(Circle((sign * 15.0, 0.0), 5.0, fill=False, edgecolor="#1a1a1a", lw=1.4))
    ax1.plot([0.0], [0.0], marker="o", color="#1a1a1a", ms=5.5)

    for ax, title_str in [
        (ax0, r"(a) Magnetic toroid $H_y(x,0,z,t)$"),
        (ax1, r"(b) Equivalent electric dipole $p_{\mathrm{eff}}$"),
    ]:
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [mm]")
        ax.set_ylabel(r"$z$ [mm]")
        ax.set_title(title_str, fontsize=10.0, fontweight="bold")

    fig.tight_layout()

    def _update(frame_idx: int):
        ph = np.exp(-1j * phases[frame_idx])
        im0.set_array(np.real(hy_tor * ph).T.ravel())
        im1.set_array(np.real(hy_dip * ph).T.ravel())
        return [im0, im1]

    ani = animation.FuncAnimation(fig, _update, frames=n_frames, interval=80, blit=True)
    gif_path = Path(fig_dir) / "magnetic_toroid_vs_dipole_wave.gif"
    ani.save(gif_path, writer="pillow", fps=12)
    plt.close(fig)
    return gif_path


def build_quantitative_summary_rows(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Build structured rows for the Quantitative Summary table directly from `data/*.json`."""
    d1 = json.loads((Path(data_dir) / "task_01_baseline_dipoles.json").read_text())
    d2 = json.loads((Path(data_dir) / "task_02_closed_toroid.json").read_text())
    d3 = json.loads((Path(data_dir) / "task_03_square_four_rods.json").read_text())
    d4 = json.loads((Path(data_dir) / "task_04_break_the_loop.json").read_text())
    d5 = json.loads((Path(data_dir) / "task_05_equivalence_limits.json").read_text())
    d6 = json.loads((Path(data_dir) / "task_06_efficiency_q_and_convergence.json").read_text())

    e1 = d1["electric_dipole"]
    m1 = d1["magnetic_loop_air"]
    ka4 = d1["ka4_scaling"]
    mom2 = d2["moments"]
    vsh2 = d2["vsh_toroid_8feed"]
    peq2 = d2["power_equivalence"]
    f2 = d2["frequency_scaling"]
    r2 = d2["octupole_scaling"]
    s3 = d3["records"][0]
    b4 = d4["cases"]
    nf5 = d5["nearfield_vs_dipole"]
    l6 = d6["loss_sweep"][0]
    c6 = d6["convergence"]
    c6_fine = c6["levels"][2]

    def _cabs(v: Any) -> float:
        if isinstance(v, dict):
            return float(np.hypot(v["real"], v["imag"]))
        return float(abs(v))

    tz_mag_abs = _cabs(mom2["tz_mag_core_am3"])
    tz_dt_abs = _cabs(mom2["tz_core_dt_am3"])
    dt_rel_err = float(mom2["dt_vs_mag_rel_err"])

    return [
        {
            "metric": "Baseline short electric dipole TM fraction ($|a_{10}|^2 / P_{\\mathrm{VSH}}$)",
            "theory": "100.00% ($|b_1|/|a_1| = 0$)",
            "simulation": f"{e1['p_a10_frac'] * 100:.4f}% ($|b_1|/|a_1| = {e1['b1_over_a1']:.2e}$)",
            "rel_err": f"{e1['pattern_sin_theta_l2_err'] * 100:.3f}% pattern $L_2$",
            "num_sim": float(e1["p_a10_frac"]),
            "num_err": float(e1["pattern_sin_theta_l2_err"]),
        },
        {
            "metric": "Baseline small magnetic loop TE fraction ($|b_1|^2 / P_{\\mathrm{VSH}}$)",
            "theory": "100.00% ($|a_1|/|b_1| = 0$)",
            "simulation": f"{m1['p_b1_frac'] * 100:.4f}% ($|a_1|/|b_1| = {m1['a1_over_b1']:.2e}$)",
            "rel_err": f"{m1['pattern_sin_theta_l2_err'] * 100:.3f}% pattern $L_2$",
            "num_sim": float(m1["p_b1_frac"]),
            "num_err": float(m1["pattern_sin_theta_l2_err"]),
        },
        {
            "metric": "Loop radiation scaling $P_{\\mathrm{rad}}(15\\;\\mathrm{mm}) / P_{\\mathrm{rad}}(10\\;\\mathrm{mm})$",
            "theory": f"{ka4['ratio_15_to_10_theory']:.4f} ($(a_2/a_1)^4$), {ka4['ratio_15_to_10_exact_j1_theory']:.4f} ($J_1^2$)",
            "simulation": f"{ka4['ratio_15_to_10_sim']:.4f} (slope {ka4['fitted_slope']:.2f})",
            "rel_err": f"{ka4['ratio_rel_err'] * 100:.2f}% vs $(a_2/a_1)^4$ ({ka4['ratio_rel_err_vs_exact_j1'] * 100:.2f}% vs $J_1^2$)",
            "num_sim": float(ka4["ratio_15_to_10_sim"]),
            "num_err": float(ka4["ratio_rel_err"]),
        },
        {
            "metric": "Closed toroid net magnetic dipole $|\\int \\mathbf{M}\\,\\mathrm{d}V| / \\int |\\mathbf{M}|\\,\\mathrm{d}V$",
            "theory": "0.00",
            "simulation": f"{mom2['m_cancel_ratio']:.2e}",
            "rel_err": f"{mom2['m_cancel_ratio']:.2e} residual",
            "num_sim": float(mom2["m_cancel_ratio"]),
            "num_err": float(mom2["m_cancel_ratio"]),
        },
        {
            "metric": "Dubovik–Tugushev identity $|T_{z,\\mathrm{DT}}(\\mathbf{J}_b)|$ vs $|T_{z,\\mathrm{mag}}(\\mathbf{M})|$",
            "theory": f"{tz_mag_abs:.6e} $\\mathrm{{A\\,m^3}}$",
            "simulation": f"{tz_dt_abs:.6e} $\\mathrm{{A\\,m^3}}$",
            "rel_err": f"{dt_rel_err:.2e}",
            "num_sim": float(tz_dt_abs),
            "num_err": float(dt_rel_err),
        },
        {
            "metric": "Closed toroid TM electric dipole fraction ($|a_{10}|^2 / P_{\\mathrm{VSH}}$)",
            "theory": "100.00% ($|b_1|/|a_1| = 0$)",
            "simulation": f"{vsh2['p_a10_frac'] * 100:.4f}% ($|b_1|/|a_1| = {vsh2['b1_over_a1']:.2e}$)",
            "rel_err": f"{vsh2['pattern_sin_theta_l2_err'] * 100:.3f}% pattern $L_2$",
            "num_sim": float(vsh2["p_a10_frac"]),
            "num_err": float(vsh2["pattern_sin_theta_l2_err"]),
        },
        {
            "metric": "Multipole-to-far-field self-consistency $P_{\\mathrm{rad}}^{\\mathrm{N2F}}$ vs $P_{\\mathrm{dip}}(p_{\\mathrm{eff}} = -\\mathrm{i}k_0 T_{\\mathrm{tot}}^{\\mathrm{sim}} F_1/c_0)$",
            "theory": f"{peq2['p_rad_theory_j1_w']:.4e} W (from simulated $T_{{\\mathrm{{tot}}}}$)",
            "simulation": f"{peq2['p_rad_n2f_w']:.4e} W (N2F), {peq2['p_rad_flux_w']:.4e} W (Flux)",
            "rel_err": f"{peq2['rel_err_n2f_vs_j1'] * 100:.2f}% self-consistency",
            "num_sim": float(peq2["p_rad_n2f_w"]),
            "num_err": float(peq2["rel_err_n2f_vs_j1"]),
        },
        {
            "metric": "Fixed-$T$ power scaling exponent $\\mathrm{d}\\ln(P_{\\mathrm{rad}}/|T_{\\mathrm{tot}}|^2)/\\mathrm{d}\\ln\\omega$",
            "theory": "6.00 ($\\omega^6$)",
            "simulation": f"{f2['slope_omega_fixed_t_point_corrected']:.2f}",
            "rel_err": f"{abs(f2['slope_omega_fixed_t_point_corrected'] - 6.0) / 6.0 * 100:.2f}%",
            "num_sim": float(f2["slope_omega_fixed_t_point_corrected"]),
            "num_err": float(abs(f2["slope_omega_fixed_t_point_corrected"] - 6.0) / 6.0),
        },
        {
            "metric": "Electric octupole scaling exponent $\\mathrm{d}\\ln(|a_{30}|/|a_{10}|)/\\mathrm{d}\\ln(k_0 R)$",
            "theory": "2.00 ($(k_0 R)^2$)",
            "simulation": f"{r2['fitted_slope_vs_kr']:.2f}",
            "rel_err": f"{abs(r2['fitted_slope_vs_kr'] - 2.0) / 2.0 * 100:.2f}%",
            "num_sim": float(r2["fitted_slope_vs_kr"]),
            "num_err": float(abs(r2["fitted_slope_vs_kr"] - 2.0) / 2.0),
        },
        {
            "metric": "Square of 4 rods ($C_4$) $|b_1|/|a_1|$ and $|m=\\pm 4|/|a_{10}|$ ($R_{\\mathrm{sq}}=12.5\\;\\mathrm{mm}$)",
            "theory": "$|b_1|/|a_1| = 0$, $C_4$ residual $> 0$",
            "simulation": f"$|b_1|/|a_1| = {s3['b1_over_a1']:.2e}$, $|m=\\pm 4|/|a_{{10}}| = {s3['c4_m4_over_a10']:.2e}$",
            "rel_err": f"{s3['p_a10_frac'] * 100:.4f}% in $a_{{10}}$",
            "num_sim": float(s3["b1_over_a1"]),
            "num_err": float(s3["c4_m4_over_a10"]),
        },
        {
            "metric": "Broken loop: 7 of 8 feeds $|b_1|/|a_1|$ vs closed 8/8 feeds",
            "theory": "$|b_1|/|a_1| \\sim \\mathcal{O}(1)$ (Valero 2024)",
            "simulation": f"{b4['seven_of_eight_feeds']['b1_over_a1']:.4f} (vs {b4['closed_8_feeds']['b1_over_a1']:.2e} closed)",
            "rel_err": f"{b4['seven_of_eight_feeds']['b1_over_a1'] / b4['closed_8_feeds']['b1_over_a1']:.2e}$\\times$ jump",
            "num_sim": float(b4["seven_of_eight_feeds"]["b1_over_a1"]),
            "num_err": float(b4["closed_8_feeds"]["b1_over_a1"]),
        },
        {
            "metric": "Helical 1-turn axial $m_z$ vs Marinov compensating loop $|b_1|/|a_1|$",
            "theory": "Uncompensated $> 0.1$, compensated $\\to 0$",
            "simulation": f"{b4['helical_uncompensated']['b1_over_a1']:.4f} $\\to$ {b4['helical_marinov_compensated']['b1_over_a1']:.2e}",
            "rel_err": f"{b4['helical_uncompensated']['b1_over_a1'] / b4['helical_marinov_compensated']['b1_over_a1']:.2e}$\\times$ suppression",
            "num_sim": float(b4["helical_uncompensated"]["b1_over_a1"]),
            "num_err": float(b4["helical_marinov_compensated"]["b1_over_a1"]),
        },
        {
            "metric": "Consistent near-field vs outer-zone error $|\\Delta\\mathbf{E}|/|\\mathbf{E}|$ ($20\\text{–}30\\;\\mathrm{mm}$ vs $45\\text{–}55\\;\\mathrm{mm}$)",
            "theory": (
                f"Outer $h_3^{{(2)}}/h_1^{{(2)}}$ octupole decays "
                f"{nf5['hankel_octupole_decay_25_to_50mm']:.2f}$\\times$ "
                f"({nf5['hankel_h3_over_h1_at_25mm']:.2f} $\\to$ {nf5['hankel_h3_over_h1_at_50mm']:.2f})"
            ),
            "simulation": f"{nf5['near_zone_20_to_30mm_e_rel_diff'] * 100:.2f}% $\\to$ {nf5['outer_zone_45_to_55mm_e_rel_diff'] * 100:.2f}% ($|\\Delta H_y|/|H_y|$: {nf5['near_zone_20_to_30mm_hy_rel_diff'] * 100:.2f}% $\\to$ {nf5['outer_zone_45_to_55mm_hy_rel_diff'] * 100:.2f}%)",
            "rel_err": f"{nf5['e_metric_reduction_factor']:.2f}$\\times$ ($|\\mathbf{{E}}|$), {nf5['hy_metric_reduction_factor']:.2f}$\\times$ ($|H_y|$) reduction",
            "num_sim": float(nf5["near_zone_20_to_30mm_e_rel_diff"]),
            "num_err": float(nf5["outer_zone_45_to_55mm_e_rel_diff"]),
        },
        {
            "metric": "Energy bookkeeping $|P_{\\mathrm{in}} - (P_{\\mathrm{rad}} + P_{\\mathrm{loss,core}})| / P_{\\mathrm{in}}$ ($\\gamma = 0.02\\;\\mathrm{GHz}$)",
            "theory": "0.00% (Poynting theorem)",
            "simulation": f"{l6['energy_closure_rel_err'] * 100:.3f}% (mutated: {l6['energy_closure_mutated_rel_err'] * 100:.1f}%)",
            "rel_err": f"{l6['energy_closure_rel_err'] * 100:.3f}%",
            "num_sim": float(l6["energy_closure_rel_err"]),
            "num_err": float(l6["energy_closure_mutated_rel_err"]),
        },
        {
            "metric": "Toroid radiation quality factor $Q_{\\mathrm{rad}}$ vs Chu bound $Q_{\\mathrm{Chu}}(k_0 a_{\\mathrm{sph}})$",
            "theory": f"$Q_{{\\mathrm{{rad,th}}}} = {l6['q_rad_theory']:.1f}$, $Q_{{\\mathrm{{Chu}}}} = {l6['q_chu_enclosing_sphere']:.2f}$",
            "simulation": f"$Q_{{\\mathrm{{rad,sim}}}} = {l6['q_rad_sim']:.1f}$ ({l6['q_ratio_over_chu_sim']:.1f}$\\times Q_{{\\mathrm{{Chu}}}}$)",
            "rel_err": f"{abs(l6['q_rad_sim'] - l6['q_rad_theory']) / l6['q_rad_theory'] * 100:.2f}%",
            "num_sim": float(l6["q_rad_sim"]),
            "num_err": float(abs(l6["q_rad_sim"] - l6["q_rad_theory"]) / l6["q_rad_theory"]),
        },
        {
            "metric": (
                f"3-grid resolution study ($\\Delta_{{\\mathrm{{core}}}} = {c6['levels'][0]['dx_core_mm']:.2f}, "
                f"{c6['levels'][1]['dx_core_mm']:.2f}, {c6_fine['dx_core_mm']:.2f}\\;\\mathrm{{mm}}$; 4–6 cells/$2r$)"
            ),
            "theory": f"$T_z^{{\\mathrm{{qs}}}} = {c6['tz_exact_continuum_am3']:.3e}\\;\\mathrm{{A\\,m^3}}$, $P_{{\\mathrm{{rad}}}} = P_{{\\mathrm{{dip}}}}(T_{{\\mathrm{{sim}}}})$",
            "simulation": (
                f"$|T_z^{{\\mathrm{{sim}}}}| = {c6['levels'][0]['tz_tot_abs_am3']:.3e} \\to {c6['levels'][1]['tz_tot_abs_am3']:.3e} \\to {c6_fine['tz_tot_abs_am3']:.3e}$; "
                f"$P_{{\\mathrm{{rad}}}} = {c6['levels'][0]['p_rad_n2f_w']:.3e} \\to {c6['levels'][1]['p_rad_n2f_w']:.3e} \\to {c6_fine['p_rad_n2f_w']:.3e}$ W"
            ),
            "rel_err": (
                f"Not grid-converged ($\\pm 9\\%$ drive staircase, {c6['p_rad_rel_change_fine_vs_med'] * 100:.0f}% $P_{{\\mathrm{{rad}}}}$ on fine); "
                f"{c6_fine['p_rad_self_consistency_rel_diff'] * 100:.2f}–{c6['levels'][0]['p_rad_self_consistency_rel_diff'] * 100:.2f}% N2F self-consistency"
            ),
            "num_sim": float(c6_fine["p_rad_n2f_w"]),
            "num_err": float(c6_fine["p_rad_self_consistency_rel_diff"]),
        },
    ]


def build_quantitative_summary_markdown(data_dir: Path = DATA_DIR) -> str:
    """Render the Markdown Quantitative Summary table directly from `data/*.json`."""
    rows = build_quantitative_summary_rows(data_dir=data_dir)
    lines = [
        "| Physical Quantity / Validation Target | Analytical / Literature Target | 3D FIT Result (`wakis`) | Relative Error / Status |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for r in rows:
        lines.append(f"| {r['metric']} | {r['theory']} | {r['simulation']} | {r['rel_err']} |")
    return "\n".join(lines)


def build_summary_json(data_dir: Path = DATA_DIR, write: bool = True) -> dict[str, Any]:
    """Build (and optionally write) `magnetic_toroid_summary.json` deterministically from `task_01..06.json`."""
    out_dir = Path(data_dir)
    t1 = json.loads((out_dir / "task_01_baseline_dipoles.json").read_text())
    t2 = json.loads((out_dir / "task_02_closed_toroid.json").read_text())
    t3 = json.loads((out_dir / "task_03_square_four_rods.json").read_text())
    t4 = json.loads((out_dir / "task_04_break_the_loop.json").read_text())
    t5 = json.loads((out_dir / "task_05_equivalence_limits.json").read_text())
    t6 = json.loads((out_dir / "task_06_efficiency_q_and_convergence.json").read_text())
    c6 = t6["convergence"]
    crit_rel_dp = float(c6.get("grid_convergence_criterion_rel_dp_max", 0.02))
    is_conv = bool(abs(float(c6["p_rad_rel_change_fine_vs_med"])) < crit_rel_dp)

    summary = {
        "f0_ghz": float(F0_HZ / 1e9),
        "lambda0_mm": float(LAM0_M * 1e3),
        "r_major_mm": float(R_MAJOR_M * 1e3),
        "r_minor_mm": float(R_MINOR_M * 1e3),
        "mu_dc": float(MU_DC_DEFAULT),
        "headline_metrics": {
            "loop_ka4_ratio_sim": float(t1["ka4_scaling"]["ratio_15_to_10_sim"]),
            "loop_ka4_ratio_theory": float(t1["ka4_scaling"]["ratio_15_to_10_theory"]),
            "loop_ka4_rel_err": float(t1["ka4_scaling"]["ratio_rel_err"]),
            "dubovik_tugushev_rel_err": float(t2["moments"]["dt_vs_mag_rel_err"]),
            "dubovik_tugushev_mutated_rel_err": float(t2["moments"]["dt_mutated_rel_err"]),
            "toroid_m_cancel_ratio": float(t2["moments"]["m_cancel_ratio"]),
            "toroid_p_a10_frac": float(t2["vsh_toroid_8feed"]["p_a10_frac"]),
            "toroid_b1_over_a1": float(t2["vsh_toroid_8feed"]["b1_over_a1"]),
            "toroid_pattern_l2_err": float(t2["vsh_toroid_8feed"]["pattern_sin_theta_l2_err"]),
            "toroid_prad_sim_w": float(t2["power_equivalence"]["p_rad_n2f_w"]),
            "toroid_prad_theory_j1_w": float(t2["power_equivalence"]["p_rad_theory_j1_w"]),
            "toroid_prad_rel_err": float(t2["power_equivalence"]["rel_err_n2f_vs_j1"]),
            "omega6_slope_fixed_t": float(t2["frequency_scaling"]["slope_omega_fixed_t_point_corrected"]),
            "omega_slope_fixed_i": float(t2["frequency_scaling"]["slope_omega_fixed_i"]),
            "octupole_slope_kr2": float(t2["octupole_scaling"]["fitted_slope_vs_kr"]),
            "four_rods_m_cancel_ratio": float(t3["records"][0]["m_cancel_ratio"]),
            "four_rods_b1_over_a1": float(t3["records"][0]["b1_over_a1"]),
            "broken_gap_b1_over_a1": float(t4["cases"]["core_air_gap_60deg"]["b1_over_a1"]),
            "broken_7of8_b1_over_a1": float(t4["cases"]["seven_of_eight_feeds"]["b1_over_a1"]),
            "broken_mode1_b1_over_a1": float(t4["cases"]["mode1_antiphase"]["b1_over_a1"]),
            "broken_3of4_rods_b1_over_a1": float(t4["cases"]["three_of_four_rods"]["b1_over_a1"]),
            "helical_uncomp_b1_over_a1": float(t4["cases"]["helical_uncompensated"]["b1_over_a1"]),
            "helical_marinov_b1_over_a1": float(t4["cases"]["helical_marinov_compensated"]["b1_over_a1"]),
            "nearfield_rel_diff_vs_dipole": float(t5["nearfield_vs_dipole"]["near_zone_20_to_30mm_e_rel_diff"]),
            "energy_closure_rel_err": float(t6["loss_sweep"][0]["energy_closure_rel_err"]),
            "energy_closure_mutated_rel_err": float(t6["loss_sweep"][0]["energy_closure_mutated_rel_err"]),
            "tz_rel_err_levels": [float(x) for x in c6["tz_rel_err_levels"]],
            "tz_disc_static_rel_err_levels": [float(x) for x in c6["tz_disc_static_rel_err_levels"]],
            "grid_convergence_criterion_rel_dp_max": crit_rel_dp,
            "is_grid_converged": is_conv,
        },
        "quantitative_summary_rows": build_quantitative_summary_rows(data_dir=out_dir),
        "quantitative_summary_markdown": build_quantitative_summary_markdown(data_dir=out_dir),
    }
    if write:
        (out_dir / "magnetic_toroid_summary.json").write_text(json.dumps(_json_clean(summary), indent=2) + "\n")
    return summary


def build_narrative_markdown_from_json(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Generate all numerical narrative Markdown sections and tables directly from `data/*.json` via f-strings."""
    out_dir = Path(data_dir)
    t1 = json.loads((out_dir / "task_01_baseline_dipoles.json").read_text())
    t2 = json.loads((out_dir / "task_02_closed_toroid.json").read_text())
    t3 = json.loads((out_dir / "task_03_square_four_rods.json").read_text())
    t4 = json.loads((out_dir / "task_04_break_the_loop.json").read_text())
    t5 = json.loads((out_dir / "task_05_equivalence_limits.json").read_text())
    t6 = json.loads((out_dir / "task_06_efficiency_q_and_convergence.json").read_text())

    e1 = t1["electric_dipole"]
    m1 = t1["magnetic_loop_air"]
    mc1 = t1["magnetic_loop_core"]
    lw1 = t1["lumped_wire_loop"]
    ka4 = t1["ka4_scaling"]

    mom2 = t2["moments"]
    vsh2 = t2["vsh_toroid_8feed"]
    peq2 = t2["power_equivalence"]
    f2 = t2["frequency_scaling"]
    r2 = t2["octupole_scaling"]

    b4 = t4["cases"]
    msw4 = t4["marinov_r_loop_sweep"]
    nf5 = t5["nearfield_vs_dipole"]
    el5 = t5["electrical_size_theory"]
    te5 = t5["core_te01_mode_sweep"]
    loss6 = t6["loss_sweep"]
    conv6 = t6["convergence"]
    lvls6 = conv6["levels"]
    l0 = lvls6[0]
    l1 = lvls6[1]
    l2 = lvls6[2]

    band_rel_diffs = [abs(1.0 - float(x)) * 100.0 for x in f2["p_rad_over_theory_j1_band"]]
    band_err_min = min(band_rel_diffs)
    band_err_max = max(band_rel_diffs)
    oct_errs_pct = [abs(float(r) - 1.0) * 100.0 for r in r2["sim_over_theory_ratio"]]
    oct_fin_ratios = r2.get(
        "sim_over_finite_r_theory_ratio",
        [
            float(s) / (float(th) * (1.0 - 0.75 * (R_MINOR_M / (float(rm) * 1e-3)) ** 2))
            for s, th, rm in zip(r2["a30_over_a10_sim"], r2["a30_over_a10_theory"], r2["r_major_mm"])
        ],
    )
    oct_fin_errs_pct = [abs(float(r) - 1.0) * 100.0 for r in oct_fin_ratios]
    jump_7of8 = float(b4["seven_of_eight_feeds"]["b1_over_a1"]) / max(float(b4["closed_8_feeds"]["b1_over_a1"]), 1e-30)
    jump_7of8_log10 = float(np.log10(jump_7of8))
    crit_rel_dp_pct = float(conv6.get("grid_convergence_criterion_rel_dp_max", 0.02)) * 100.0

    # Table for Task 4 (Break the Loop)
    b4_rows = [
        "| Configuration | Symmetry / Topology | $|b_1|/|a_1|$ | $|a_1|/|a_{1,\\mathrm{closed}}|$ | $|b_1|/|a_{1,\\mathrm{closed}}|$ | Electric Power Fraction | Magnetic Power Fraction | $|\\int \\mathbf{M}\\,\\mathrm{d}V|/\\int|\\mathbf{M}|\\,\\mathrm{d}V$ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for key, c in b4.items():
        b4_rows.append(
            f"| **{c['label']}** | `{key}` | `{c['b1_over_a1']:.3e}` | `{c['a1_over_a1_closed']:.3e}` | `{c['b1_over_a1_closed']:.3e}` | "
            f"{c['p_electric_frac'] * 100:.2f}% | {c['p_magnetic_frac'] * 100:.2f}% | `{c['m_cancel_ratio']:.2e}` |"
        )
    task_04_table_md = "\n".join(b4_rows)

    # Table for Task 6 (Convergence)
    conv_md = [
        "| Grid Level | $\\Delta_{\\mathrm{core}}$ [mm] | $\\mathrm{cells}/\\lambda_{\\mathrm{core}}$ | Cells Across $2r$ | Grid $N_x \\times N_y \\times N_z$ (Total) | Wall Time [s] | $T_z^{\\mathrm{disc}}$ [$\\mathrm{A\\,m^3}$] ($\\Delta T_z^{\\mathrm{disc}}/T_z^{\\mathrm{qs}}$) | $|T_z^{\\mathrm{sim}}|$ [$\\mathrm{A\\,m^3}$] ($|T_z^{\\mathrm{sim}}|/T_z^{\\mathrm{disc}}$) | $P_{\\mathrm{rad}}^{\\mathrm{N2F}}$ [W] | Self-Consistency $|P_{\\mathrm{rad}} - P_{\\mathrm{dip}}(T_{\\mathrm{sim}})|/P_{\\mathrm{dip}}$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    labels_lvl = ["Level 1 (Nominal)", "Level 2 (Medium)", "Level 3 (Fine)"]
    for i, r in enumerate(lvls6):
        p_err_th = r["p_rad_self_consistency_rel_diff"] * 100.0
        disc_shift_pct = (r["tz_disc_over_cont_static"] - 1.0) * 100.0
        conv_md.append(
            f"| **{labels_lvl[i]}** | {r['dx_core_mm']:.2f} | {r['cells_per_lambda_core']:.1f} | {r['cells_across_2r']:.1f} | "
            f"{r['nx']}×{r['ny']}×{r['nz']} ({r['n_cells']:,}) | {r['wall_s']:.1f} | "
            f"`{r['tz_disc_static_drive_am3']:.4e}` ({disc_shift_pct:+.2f}%) | "
            f"`{r['tz_tot_abs_am3']:.4e}` ({r['tz_sim_over_disc_residual_ratio']:.4f}$\\times$) | "
            f"`{r['p_rad_n2f_w']:.4e}` | **{p_err_th:.2f}%** |"
        )
    convergence_table_md = "\n".join(conv_md)

    sq_area_factor = (4.0 / np.pi) ** 2
    r_after_edip_ohm = lw1["r_rad_square_small_loop_ohm"] * lw1["electric_dipole_power_boost_factor"]
    r_rel_diff_port_vs_n2f = (
        abs(lw1["z_in_real_ohm"] - lw1["r_rad_from_prad_ohm"]) / max(lw1["z_in_real_ohm"], 1e-30) * 100.0
    )

    te1 = te5[1]
    te2 = te5[2]
    pr_ratio_te1 = te1["p_rad_sim_w"] / max(te5[0]["p_rad_sim_w"], 1e-30)
    pr_ratio_te2 = te2["p_rad_sim_w"] / max(te5[0]["p_rad_sim_w"], 1e-30)
    hyb_rel_err_te1 = abs(te1["p_rad_sim_w"] - te1["p_rad_hybrid_pred_w"]) / max(te1["p_rad_sim_w"], 1e-30) * 100.0

    closure_min_pct = min(r["energy_closure_rel_err"] for r in loss6) * 100.0
    closure_max_pct = max(r["energy_closure_rel_err"] for r in loss6) * 100.0
    mut_min_pct = min(r["energy_closure_mutated_rel_err"] for r in loss6) * 100.0
    mut_max_pct = max(r["energy_closure_mutated_rel_err"] for r in loss6) * 100.0
    eta_rel_errs = [abs(r["eta_rad_sim"] - r["eta_rad_theory"]) / max(r["eta_rad_theory"], 1e-30) * 100.0 for r in loss6]
    mu_c_f0 = complex(make_core_material(mu_dc=MU_DC_DEFAULT).mu(F0_HZ))
    mu_p_f0 = float(np.real(mu_c_f0))
    mu_pp_f0 = abs(float(np.imag(mu_c_f0)))

    return {
        "theory_4p2_countercases": (
            f"1. **Air gap in the toroidal core ($60^\\circ$ sector removed):** North and south magnetic poles form on the two cut faces of the high-$\\mu$ core, creating a large transverse magnetic dipole $\\mathbf{{m}}_\\perp \\ne \\mathbf{{0}}$ and making $|b_1|/|a_1| = {b4['core_air_gap_60deg']['b1_over_a1']:.3f} > 1$.\n"
            f"2. **7 of 8 feeds active (1 feed open):** Inspired by Canós Valero et al. (2024) [3] (who model an 8-element metallic toroidal ring of individual loop diameter $5.5\\;\\mathrm{{mm}}$ evaluated at $1000\\;\\mathrm{{MHz}}$ and use a horn rated $\\ge 0.7\\;\\mathrm{{GHz}}$; we adapt their 7-of-8 open-feed symmetry-breaking test to our permeable torus with $R = {R_MAJOR_M * 1e3:.0f}\\;\\mathrm{{mm}}$, $r = {R_MINOR_M * 1e3:.0f}\\;\\mathrm{{mm}}$ at $f_0 = {F0_HZ / 1e9:.2f}\\;\\mathrm{{GHz}}$), turning off one of the 8 loops breaks $C_{{8v}}$ symmetry and leaves an uncancelled magnetic dipole $m_\\phi$ comparable in radiated power to the toroidal dipole $T_z$ ($|b_1|/|a_1| = {b4['seven_of_eight_feeds']['b1_over_a1']:.4f}$).\n"
            f"3. **Unbalanced feed phases — Mode 1 antiphase split-winding \"loop mode\" (`[++ ++ -- --]`):** Whereas Mode 1 of Yousefi & Diaz (2017) [10] is a circularly polarized quadrature-phased excitation ($\\mathrm{{e}}^{{\\pm\\mathrm{{i}}\\phi_n}}$), driving the $+y$ half of the torus with $+I_0$ and the $-y$ half with $-I_0$ (`[++ ++ -- --]`) is its linearly polarized antiphase counterpart: $\\mathbf{{M}}$ reverses on one half so both halves point in the same transverse direction. By antisymmetry, the axial toroidal moment cancels ($T_z = 0 \\implies |a_1|_{{\\text{{Mode 1}}}}/|a_1|_{{\\text{{closed}}}} = {b4['mode1_antiphase']['a1_over_a1_closed']:.3e}$, at the numerical noise floor) while the two halves add constructively as a transverse magnetic dipole ($|b_1|_{{\\text{{Mode 1}}}}/|a_1|_{{\\text{{closed}}}} = {b4['mode1_antiphase']['b1_over_a1_closed']:.3f}$, so $|b_1|/|a_1| = {b4['mode1_antiphase']['b1_over_a1']:.3e}$ reflects the noise-floor denominator $a_1 \\approx 0$).\n"
            f"4. **3 of 4 rods in the square (1 rod removed):** Breaking $C_{{4v}}$ down to 3 rods leaves an uncancelled rod magnetization $\\mathbf{{m}} \\ne \\mathbf{{0}}$, causing $|b_1|/|a_1|$ to jump from ${b4['four_rods_closed']['b1_over_a1']:.2e}$ to ${b4['three_of_four_rods']['b1_over_a1']:.3f}$.\n"
            f"5. **3D helical winding pitch vs Marinov's planar compensating loop [1, 11]:** A real poloidal coil wound continuously as a single-layer 3D helical wire path $\\rho(t) = R + r\\cos(2\\pi N t), \\phi(t) = 2\\pi t, z(t) = -r\\sin(2\\pi N t)$ discretised onto primary electric edges (`deposit_closed_wire_on_edges`) advances by $2\\pi$ around $\\hat{{\\boldsymbol{{\\phi}}}}$ after $N$ poloidal turns, producing a parasitic axial magnetic dipole moment $m_z^{{\\mathrm{{par}}}} = \\pi I_{{\\mathrm{{turn}}}} R^2 [1 + \\frac{{1}}{{2}}(r/R)^2]$ (Marinov et al. 2007 Eq. 9 [1]; McDonald 2008 [11]). Routing a separate planar circular return wire loop in the equatorial plane $z = 0$ of radius $R_{{\\mathrm{{loop}}}}$ driven by the **exact same current** $-I_{{\\mathrm{{turn}}}}$ (zero rescaling) makes $|b_1|/|a_1|$ pass through a sharp minimum at Marinov's Eq. 9 radius $R_1 = R\\sqrt{{1 + \\frac{{1}}{{2}}(r/R)^2}} = {msw4['r1_marinov_mm']:.2f}\\;\\mathrm{{mm}}$, reducing $|b_1|/|a_1|$ from ${b4['helical_uncompensated']['b1_over_a1']:.4f}$ (analytical ${msw4['b1_over_a1_theory_uncompensated']:.4f}$) down to ${b4['helical_marinov_compensated']['b1_over_a1']:.2e}$ (${msw4['suppression_factor_at_r1']:.0f}\\times$ suppression, and ${msw4['improvement_over_naive_r_major']:.1f}\\times$ lower than a naive loop at $R_{{\\mathrm{{loop}}}} = R = {msw4['r_major_mm']:.1f}\\;\\mathrm{{mm}}$ where $|b_1|/|a_1| = {msw4['b1_over_a1_at_r_major']:.4f}$)."
        ),
        "setup_mesh_bullets": (
            f"We model the magnetic toroid and its comparison geometries in `wakis.SolverFIT3D` at nominal frequency $f_0 = {F0_HZ / 1e9:.2f}\\;\\mathrm{{GHz}}$ ($\\lambda_0 = {LAM0_M * 1e3:.2f}\\;\\mathrm{{mm}}, k_0 = {K0_RAD_M:.3f}\\;\\mathrm{{rad/m}}$) in an open 6-face `CPML` domain:\n"
            f"- **Dispersive magnetic core:** `wakis.dispersive.DispersiveMaterial` with a single Lorentz magnetic pole (`mu_poles`), $\\mu_{{\\mathrm{{DC}}}} = {MU_DC_DEFAULT:.1f}$, magnetic resonance $f_{{\\mathrm{{res}}}} = 6.0\\;\\mathrm{{GHz}}$ ($8\\times$ above $f_0$), and low damping $\\gamma / 2\\pi = 0.04\\;\\mathrm{{GHz}}$ (giving $\\mu_r(f_0) = {mu_p_f0:.2f} - {mu_pp_f0:.3f}\\mathrm{{i}}$, refractive index $n_{{\\mathrm{{core}}}} = {np.sqrt(mu_p_f0):.2f}$, and internal wavelength $\\lambda_{{\\mathrm{{core}}}} = \\lambda_0 / n_{{\\mathrm{{core}}}} = {LAM0_M * 1e3 / np.sqrt(mu_p_f0):.2f}\\;\\mathrm{{mm}}$).\n"
            f"- **Locally refined graded mesh:** Built with `wakis.templates.meshing.graded_axis`, placing uniform $\\Delta_{{\\mathrm{{core}}}} = {l0['dx_core_mm']:.2f}\\;\\mathrm{{mm}}$ cells across the core region (${l0['cells_per_lambda_core']:.1f}\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{core}}}} \\ge 20$ and ${l0['cells_across_2r']:.1f}\\;\\mathrm{{cells}}$ across the $2r = {2.0 * R_MINOR_M * 1e3:.1f}\\;\\mathrm{{mm}}$ core minor diameter on the ${l0['nx']}\\times {l0['ny']}\\times {l0['nz']} = {l0['n_cells']:,}$-cell nominal grid; refined to $\\Delta_{{\\mathrm{{core}}}} = {l1['dx_core_mm']:.2f}\\;\\mathrm{{mm}}$, ${l1['cells_per_lambda_core']:.1f}\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{core}}}}$, and ${l1['cells_across_2r']:.1f}\\;\\mathrm{{cells}}$ across $2r$ on the ${l1['nx']}\\times {l1['ny']}\\times {l1['nz']} = {l1['n_cells']:,}$-cell medium grid, and to $\\Delta_{{\\mathrm{{core}}}} = {l2['dx_core_mm']:.2f}\\;\\mathrm{{mm}}$, ${l2['cells_per_lambda_core']:.1f}\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{core}}}}$, and ${l2['cells_across_2r']:.1f}\\;\\mathrm{{cells}}$ across $2r$ on the ${l2['nx']}\\times {l2['ny']}\\times {l2['nz']} = {l2['n_cells']:,}$-cell fine convergence grid).\n"
            f"- **Full-domain verification (Rule 1):** All simulations—especially the broken-loop counter-cases in Task 4—are run on the full 3D domain without symmetry planes that could mask asymmetric multipoles."
        ),
        "results_task_01": (
            f"Figure 1 establishes the dual baseline before closing the magnetic loop:\n"
            f"- **Panel (a) — Short axial electric dipole $\\mathbf{{p}} = p_z\\hat{{\\mathbf{{z}}}}$:** Driven by an axial `CurrentSource`, the radiated far field is purely polar-polarized ($E_\\theta \\propto \\sin\\theta$, cross-polarization $E_\\phi \\approx 0$), with ${e1['p_a10_frac'] * 100:.4f}\\%$ of the vector spherical harmonic ($\\mathrm{{VSH}}$) power in the electric dipole coefficient $a_{{10}}$ ($|b_1|/|a_1| = {e1['b1_over_a1']:.2e}$).\n"
            f"- **Panel (b) — Small magnetic loop $\\mathbf{{m}} = m_z\\hat{{\\mathbf{{z}}}}$:** Driven by a single loop in the $xy$-plane, the radiated far field is the exact dual—purely azimuthally polarized ($E_\\phi \\propto \\sin\\theta$, $E_\\theta \\approx 0$), with ${m1['p_b1_frac'] * 100:.4f}\\%$ of the radiated power in the magnetic dipole coefficient $b_{{10}}$ ($|a_1|/|b_1| = {m1['a1_over_b1']:.2e}$).\n"
            f"- **Panel (c) — $(k_0 a)^4$ radiation scaling, permeable core boost, and real `LumpedPort` wire loop:** Sweeping the loop radius across $a \\in \\{{{ka4['radii_mm'][0]:.1f}, {ka4['radii_mm'][1]:.1f}, {ka4['radii_mm'][2]:.1f}\\}}\\;\\mathrm{{mm}}$ confirms the rapid $(k_0 a)^4$ power scaling: with $5\\times 5$ sub-cell disk area weights on the $\\Delta_{{\\mathrm{{core}}}} = 2.50\\;\\mathrm{{mm}}$ grid, $P_{{\\mathrm{{rad}}}}(15\\;\\mathrm{{mm}})/P_{{\\mathrm{{rad}}}}(10\\;\\mathrm{{mm}}) = {ka4['ratio_15_to_10_sim']:.4f}$ (log-log slope ${ka4['fitted_slope']:.2f}$), within ${ka4['ratio_rel_err'] * 100:.2f}\\%$ of $(1.5)^4 = {ka4['ratio_15_to_10_theory']:.4f}$ and within ${ka4['ratio_rel_err_vs_exact_j1'] * 100:.2f}\\%$ of the finite-$k_0 a$ Bessel-corrected ratio ${ka4['ratio_15_to_10_exact_j1_theory']:.4f}$. Inserting the $\\mu_{{\\mathrm{{DC}}}} = {mc1['mu_dc']:.0f}$ permeable core inside the loop increases the radiated power by a factor of ${mc1['power_gain_over_air']:.2f}\\times$ while preserving $|b_1|/|a_1| = {mc1['b1_over_a1']:.2e} \\gg 10^4$. Feeding a thin PEC square wire loop of half-side $a = {lw1['a_mm']:.1f}\\;\\mathrm{{mm}}$ (side $2a = {2.0 * lw1['a_mm']:.0f}\\;\\mathrm{{mm}}$, perimeter $8a = {8.0 * lw1['a_mm']:.0f}\\;\\mathrm{{mm}} = {8.0 * lw1['a_mm'] / (LAM0_M * 1e3):.3f}\\,\\lambda_0$) at a single gap on the $-y$ arm with a $50\\;\\Omega$ `LumpedPort` (`Setup 2`) yields $\\operatorname{{Re}} Z_{{\\mathrm{{in}}}} = {lw1['z_in_real_ohm']:.3f}\\;\\Omega$ vs the small circular loop formula $R_{{\\mathrm{{rad}}}}^{{\\mathrm{{circ}}}}(k_0 a = {K0_RAD_M * lw1['a_mm'] * 1e-3:.4f}) = {lw1['r_rad_circular_small_loop_ohm']:.3f}\\;\\Omega$: first, $2 P_{{\\mathrm{{rad}}}}^{{\\mathrm{{N2F}}}} / |I_{{\\mathrm{{port}}}}|^2 = {lw1['r_rad_from_prad_ohm']:.3f}\\;\\Omega$ matches $\\operatorname{{Re}} Z_{{\\mathrm{{in}}}} = {lw1['z_in_real_ohm']:.3f}\\;\\Omega$ within ${r_rel_diff_port_vs_n2f:.1f}\\%$ (confirming zero artificial wire or port loss); second, the square loop area $4a^2 = (4/\\pi)\\pi a^2$ raises the magnetic-dipole radiation resistance by $(4/\\pi)^2 = {sq_area_factor:.3f}\\times$ to ${lw1['r_rad_square_small_loop_ohm']:.3f}\\;\\Omega$; third, single-port feed asymmetry excites an electric dipole mode with $|a_1|/|b_1| = {lw1['a1_over_b1_feed_asymmetry']:.3f}$, adding $1 + |a_1/b_1|^2 = {lw1['electric_dipole_power_boost_factor']:.3f}\\times$ power (${r_after_edip_ohm:.3f}\\;\\Omega$); and fourth, the remaining factor $R_{{\\mathrm{{meas}}}} / (R_{{\\mathrm{{sq}}}} \\cdot [1 + |a_1/b_1|^2]) = {lw1['unexplained_residual_factor']:.3f}\\times$ (${r_after_edip_ohm:.3f} \\times {lw1['unexplained_residual_factor']:.3f} = {lw1['r_rad_from_prad_ohm']:.3f}\\;\\Omega$) is an **unexplained residual** on the finite-thickness wire/port grid (for reference, the **analytic estimate** for a standing-wave current $I(s) \\propto \\cos(k_0(L/2 - s))$ on a $\\lambda_0/4$ loop gives a loop-to-port current-squared ratio $[(\\sin x / x)/\\cos x]^2 = {lw1['analytic_cos_current_factor_pi_over_4']:.3f}$ at $x = \\pi/4$)."
        ),
        "results_task_02": (
            f"Next we wind $N = 8$ in-phase poloidal loops around the closed Lorentz dispersive torus ($R = {R_MAJOR_M * 1e3:.1f}\\;\\mathrm{{mm}}, r = {R_MINOR_M * 1e3:.1f}\\;\\mathrm{{mm}}, \\mu_{{\\mathrm{{DC}}}} = {MU_DC_DEFAULT:.1f}$).\n\n"
            f"Figure 2 visualizes the simulated 3D FIT fields (`DFTMonitor` cuts overlaid with `wakis.schematic.overlay_geometry`):\n"
            f"- **Panel (a) — Equatorial cut ($z = 0$):** Inside the permeable ring ($10\\;\\mathrm{{mm}} \\le \\rho \\le 20\\;\\mathrm{{mm}}$), the magnetic flux density $|\\mathbf{{B}}|$ is strongly concentrated and the magnetization vectors $\\operatorname{{Re}}\\{{\\mathbf{{M}}\\}} $ form a closed counterclockwise azimuthal vortex. Integrating $\\mathbf{{M}}$ over the 3D core confirms that the net magnetic dipole moment cancels to $|\\int \\mathbf{{M}}\\,\\mathrm{{d}}V| / \\int |\\mathbf{{M}}|\\,\\mathrm{{d}}V = {mom2['m_cancel_ratio']:.2e} < 10^{{-6}}$.\n"
            f"- **Panel (b) — Meridional cut ($y = 0$):** The external magnetic field $H_y(x, 0, z)$ emerging from the torus has even symmetry in $z$ and opposite sign on $+x$ and $-x$ ($H_\\phi \\propto \\sin\\theta$), matching a vertically oriented $\\mathrm{{TM}}_{{10}}$ electric dipole.\n"
            f"- **Panel (c) — Axial electric field $E_z(x, y, 0)$:** Induced by Faraday's law $\\oint \\mathbf{{E}}\\cdot\\mathrm{{d}}\\mathbf{{l}} = \\mathrm{{i}}\\omega\\Phi_B$ threading the azimuthal magnetic flux tube, $E_z$ peaks along the central $z$-axis like a virtual displacement-current rod along $\\hat{{\\mathbf{{z}}}}$.\n\n"
            f"Figure 3 quantifies the spherical multipole spectrum ($l \\le 5$), pattern, and scaling laws:\n"
            f"- **Dubovik–Tugushev identity on the 3D FIT grid:** Evaluating $\\mathbf{{J}}_b = \\nabla_{{\\mathrm{{FIT}}}}\\times\\mathbf{{M}}$ via the discrete FIT dual-curl operator `solver.itDa @ (solver.C.T @ solver.tDs)` and integrating $\\mathbf{{T}}_{{\\mathrm{{DT}}}}(\\mathbf{{J}}_b) = \\frac{{1}}{{10}}\\int [\\mathbf{{r}}(\\mathbf{{r}}\\cdot\\mathbf{{J}}_b) - 2r^2\\mathbf{{J}}_b]\\,\\mathrm{{d}}^3r$ matches $\\mathbf{{T}}_{{\\mathrm{{mag}}}}(\\mathbf{{M}}) = \\frac{{1}}{{2}}\\int (\\mathbf{{r}}\\times\\mathbf{{M}})\\,\\mathrm{{d}}^3r$ to machine precision (`rel_err = {mom2['dt_vs_mag_rel_err']:.2e}`), whereas mutating the coefficient $-2r^2\\mathbf{{J}}_b \\to -1r^2\\mathbf{{J}}_b$ produces the exact ${mom2['dt_mutated_rel_err'] * 100:.1f}\\%$ operator difference.\n"
            f"- **Broadband multipole-to-far-field self-consistency:** Computing $\\mathbf{{p}}_{{\\mathrm{{eff}}}}(f) = -(\\mathrm{{i}} k_0 / c_0)\\mathbf{{T}}_{{\\mathrm{{tot}}}}^{{\\mathrm{{sim}}}}(f) F_1(k_0 R)$ directly from the simulated 3D volume integral $\\mathbf{{T}}_{{\\mathrm{{tot}}}}^{{\\mathrm{{sim}}}}(f)$ at each of the {len(f2['freqs_ghz'])} frequencies across ${f2['freqs_ghz'][0]:.2f}\\text{{–}}{f2['freqs_ghz'][-1]:.2f}\\;\\mathrm{{GHz}}$ matches the simulated `Near2FarMonitor` radiated power $P_{{\\mathrm{{rad}}}}^{{\\mathrm{{N2F}}}}(f)$ within **${band_err_min:.2f}\\%\\text{{–}}{band_err_max:.2f}\\%$** across the entire band ($P_{{\\mathrm{{rad}}}}^{{\\mathrm{{N2F}}}}(f) / P_{{\\mathrm{{dip}}}}(\\mathbf{{p}}_{{\\mathrm{{eff}}}}(f)) \\in [{min(f2['p_rad_over_theory_j1_band']):.4f}, {max(f2['p_rad_over_theory_j1_band']):.4f}]$, and **${peq2['rel_err_flux_vs_j1'] * 100:.2f}\\%$** vs the closed-box `FluxMonitor` integral at $f_0$). Because both $\\mathbf{{T}}_{{\\mathrm{{tot}}}}^{{\\mathrm{{sim}}}}$ and $P_{{\\mathrm{{rad}}}}^{{\\mathrm{{N2F}}}}$ are extracted from the same simulated fields, we label this ${peq2['rel_err_n2f_vs_j1'] * 100:.2f}\\%$ agreement at $f_0$ as **internal multipole-to-far-field self-consistency**.\n"
            f"- **$\\omega^6$ fixed-$T$ vs fixed-$I$ scaling and $(k_0 R)^2$ octupole scaling:** Across ${f2['freqs_ghz'][0]:.2f}\\text{{–}}{f2['freqs_ghz'][-1]:.2f}\\;\\mathrm{{GHz}}$, the Lorentz permeability $|\\mu(f)|$ rises by **${f2['mu_band_rise_frac'] * 100:.1f}\\%$** and the **analytic estimate** for the interior Bessel factor $1 + (k_c r)^2/8$ rises by **${f2['analytic_interior_band_rise_frac'] * 100:.1f}\\%$** (at $f_0$, $1 + (k_c r)^2/8 = {mom2['analytic_interior_factor_f0']:.3f}$, which does not explain the full dynamic residual $|T_z^{{\\mathrm{{sim}}}}|/T_z^{{\\mathrm{{disc}}}} = {mom2['tz_sim_over_disc_residual_ratio']:.3f}$), while the measured simulated toroidal moment $|T_{{\\mathrm{{tot}}}}^{{\\mathrm{{sim}}}}(f)|$ rises by **${f2['tz_sim_band_rise_frac'] * 100:.1f}\\%$**, giving a fixed-current log-log slope of **${f2['slope_omega_fixed_i']:.4f}$** (${f2['slope_omega_fixed_i']:.2f}$). Dividing $P_{{\\mathrm{{rad}}}}(f)$ by the actual simulated $|T_{{\\mathrm{{tot}}}}^{{\\mathrm{{sim}}}}(f)/T_{{\\mathrm{{tot}}}}^{{\\mathrm{{sim}}}}(f_0)|^2 F_1(k_0 R)^2$ at each frequency isolates the exact point-dipole fixed-$T$ scaling with a measured slope of **${f2['slope_omega_fixed_t_point_corrected']:.4f}$** vs theoretical **$6.0000$** (${abs(f2['slope_omega_fixed_t_point_corrected'] - 6.0) / 6.0 * 100:.2f}\\%$ error!). Sweeping $R \\in \\{{{int(r2['r_major_mm'][0])}, {int(r2['r_major_mm'][1])}, {int(r2['r_major_mm'][2])}, {int(r2['r_major_mm'][3])}\\}}\\;\\mathrm{{mm}}$ confirms that the electric octupole ratio $|a_{{30}}|/|a_{{10}}|$ follows the thin-ring formula $\\sqrt{{7/8}}\\,j_3(k_0 R)/j_1(k_0 R)$ within ${min(oct_errs_pct):.1f}\\%\\text{{–}}{max(oct_errs_pct):.1f}\\%$ and matches the derived finite cross-section octupole formula $\\sqrt{{7/8}}\\,[1 - \\frac{{3}}{{4}}(r/R)^2]\\,j_3(k_0 R)/j_1(k_0 R)$ within **${min(oct_fin_errs_pct):.2f}\\%\\text{{–}}{max(oct_fin_errs_pct):.2f}\\%$**, with a fitted $(k_0 R)^2$ log-log slope of **${r2['fitted_slope_vs_kr']:.2f}$**."
        ),
        "results_task_03": (
            f"Figure 4 tests the discrete $C_4$ geometry—four separate permeable rods ($\\mu_{{\\mathrm{{DC}}}} = {MU_DC_DEFAULT:.0f}$) arranged in a square of radius $R_{{\\mathrm{{sq}}}} \\in \\{{{t3['records'][0]['r_sq_mm']:.1f}, {t3['records'][1]['r_sq_mm']:.1f}, {t3['records'][2]['r_sq_mm']:.1f}\\}}\\;\\mathrm{{mm}}$ with head-to-tail magnetization $+M\\hat{{\\mathbf{{y}}}}, -M\\hat{{\\mathbf{{x}}}}, -M\\hat{{\\mathbf{{y}}}}, +M\\hat{{\\mathbf{{x}}}}$:\n"
            f"- **Panel (a):** Even though the four rods are physically separated at the corners, the four magnetization vectors sum to zero ($|\\int \\mathbf{{M}}\\,\\mathrm{{d}}V|/\\int |\\mathbf{{M}}|\\,\\mathrm{{d}}V = {t3['records'][0]['m_cancel_ratio']:.2e}\\text{{–}}{t3['records'][-1]['m_cancel_ratio']:.2e}$) and drive an axial electric field $E_z$ through the center of the square just like the continuous torus.\n"
            f"- **Panel (b):** Across all three square sizes, $|b_1|/|a_1| = {t3['records'][0]['b1_over_a1']:.2e}\\text{{–}}{t3['records'][-1]['b1_over_a1']:.2e} < 2\\times 10^{{-5}}$ vanishes to grid precision and ${t3['records'][-1]['p_a10_frac'] * 100:.4f}\\%\\text{{–}}{t3['records'][0]['p_a10_frac'] * 100:.4f}\\%$ of the radiated power remains in $a_{{10}}$.\n"
            f"- **Panel (c):** The leading non-axisymmetric $C_4$ multipole residual $|m = \\pm 4| / |a_{{10}}|$ (at $l = 4, 5$) is small (${t3['records'][0]['c4_m4_over_a10']:.2e} \\to {t3['records'][1]['c4_m4_over_a10']:.2e} \\to {t3['records'][2]['c4_m4_over_a10']:.2e}$) and grows monotonically with electrical size $k_0 R_{{\\mathrm{{sq}}}}$. "[:-1]
        ),
        "results_task_04": (
            f"Figure 5 demonstrates what happens when the closed magnetization loop or winding symmetry is broken—each case serves as a physical falsification test where magnetic multipoles $b_{{1m}}$ immediately appear and alter the far-field polarization and pattern:\n"
            f"- **$60^\\circ$ air gap in the toroidal core:** Magnetic poles at the cut faces produce $|b_1|/|a_1| = {b4['core_air_gap_60deg']['b1_over_a1']:.3f}$ (${b4['core_air_gap_60deg']['p_magnetic_frac'] * 100:.1f}\\%$ of the radiated power shifts into magnetic $\\mathrm{{TE}}$ modes!).\n"
            f"- **7 of 8 feeds active (1 feed open):** Reproducing the broken-feed scenario of Canós Valero et al. (2024) [3], turning off a single feed out of 8 causes $|b_1|/|a_1|$ to jump by **${jump_7of8:.2e}\\times$ (${jump_7of8_log10:.1f}$ orders of magnitude)** from ${b4['closed_8_feeds']['b1_over_a1']:.2e}$ to ${b4['seven_of_eight_feeds']['b1_over_a1']:.4f}$, making the parasitic magnetic dipole $m$ comparable to the toroidal dipole $T$.\n"
            f"- **Mode 1 antiphase feeds (`[++ ++ -- --]`):** Reversing the phase on half the torus cancels $T_z$ by antisymmetry ($|a_1|_{{\\text{{Mode 1}}}}/|a_1|_{{\\text{{closed}}}} = {b4['mode1_antiphase']['a1_over_a1_closed']:.3e}$ at the numerical noise floor) and turns the structure into a pure transverse magnetic loop antenna ($|b_1|_{{\\text{{Mode 1}}}}/|a_1|_{{\\text{{closed}}}} = {b4['mode1_antiphase']['b1_over_a1_closed']:.3f}$; the formal ratio $|b_1|/|a_1| = {b4['mode1_antiphase']['b1_over_a1']:.3e}$ reflects division by the noise-floor $a_1$).\n"
            f"- **3 of 4 rods in the square (1 rod removed):** Removing one rod breaks $C_{{4v}}$ symmetry and increases $|b_1|/|a_1|$ from ${b4['four_rods_closed']['b1_over_a1']:.2e}$ to ${b4['three_of_four_rods']['b1_over_a1']:.3f}$.\n"
            f"- **3D helical winding (1 net azimuthal turn) vs Marinov's planar compensating loop [1, 11]:** Discretising the real 3D helical wire path $\\rho(t) = R + r\\cos(2\\pi N t), \\phi(t) = 2\\pi t, z(t) = -r\\sin(2\\pi N t)$ onto primary electric edges (`deposit_closed_wire_on_edges`) with turn current $I_{{\\mathrm{{turn}}}} = {msw4['i_turn_a'] * 1e3:.2f}\\;\\mathrm{{mA}}$ (zero rescaling) magnetizes the permeable core ($|a_1|/|a_{{1,\\mathrm{{closed}}}}| = {b4['helical_uncompensated']['a1_over_a1_closed']:.4f}$) while exciting a parasitic axial magnetic dipole $m_z^{{\\mathrm{{par}}}}$ (Marinov Eq. 9 [1]) with $|b_1|/|a_1| = {b4['helical_uncompensated']['b1_over_a1']:.4f}$ (matching the analytical ratio ${msw4['b1_over_a1_theory_uncompensated']:.4f}$ within ${abs(b4['helical_uncompensated']['b1_over_a1'] - msw4['b1_over_a1_theory_uncompensated']) / msw4['b1_over_a1_theory_uncompensated'] * 100:.1f}\\%$). Sweeping the radius $R_{{\\mathrm{{loop}}}} \\in \\{{{msw4['r_loop_mm'][0]:.1f}, {msw4['r_loop_mm'][1]:.1f}, {msw4['r_loop_mm'][2]:.1f}, {msw4['r_loop_mm'][3]:.2f}, {msw4['r_loop_mm'][4]:.1f}, {msw4['r_loop_mm'][5]:.1f}\\}}\\;\\mathrm{{mm}}$ of a separate planar return wire loop in $z = 0$ driven by the **exact same current** $-I_{{\\mathrm{{turn}}}}$ (Figure 5c, zero rescaling) shows $|b_1|/|a_1|$ passing through a sharp V-shaped minimum at Marinov's Eq. 9 radius $R_1 = R\\sqrt{{1 + \\frac{{1}}{{2}}(r/R)^2}} = {msw4['r1_marinov_mm']:.2f}\\;\\mathrm{{mm}}$: $|b_1|/|a_1|$ drops from ${b4['helical_uncompensated']['b1_over_a1']:.4f}$ to ${msw4['b1_over_a1_at_r_major']:.4f}$ at the naive major radius $R_{{\\mathrm{{loop}}}} = R = {msw4['r_major_mm']:.1f}\\;\\mathrm{{mm}}$, reaches its minimum **${b4['helical_marinov_compensated']['b1_over_a1']:.2e}$** at $R_{{\\mathrm{{loop}}}} = R_1 = {msw4['r1_marinov_mm']:.2f}\\;\\mathrm{{mm}}$ (**${msw4['suppression_factor_at_r1']:.0f}\\times$ amplitude suppression** vs uncompensated and **${msw4['improvement_over_naive_r_major']:.1f}\\times$ lower** than at $R_{{\\mathrm{{loop}}}} = R$), and rises back to ${msw4['b1_over_a1'][-1]:.4f}$ at $R_{{\\mathrm{{loop}}}} = {msw4['r_loop_mm'][-1]:.1f}\\;\\mathrm{{mm}}$."
        ),
        "task_04_table": task_04_table_md,
        "results_task_05": (
            f"Why is a magnetic toroid **not** simply an ordinary electric dipole everywhere in space? Figure 6 and the animated wave GIF (`figures/magnetic_toroid_vs_dipole_wave.gif`) expose three fundamental regimes:\n"
            f"1. **Reactive near field $\\ne$ Hertzian dipole near field (Figure 6a–b):** Outside the bounding sphere $r > a_{{\\mathrm{{sph}}}} = R + r = 20\\;\\mathrm{{mm}}$, both the magnetic toroid and a point electric dipole satisfy the source-free Maxwell equations ($\\nabla\\cdot\\mathbf{{E}} = 0$), and the toroid's exterior field equals the full dipole field (including its $1/r^3$ and $1/r^2$ reactive near-zone terms) plus higher odd-$l$ electric multipoles ($l = 3, 5, \\dots$). Inside and just outside the bounding sphere ($r \\sim a_{{\\mathrm{{sph}}}}$), the point dipole's $1/r^3$ field diverges toward $r \\to 0$ whereas the finite-radius torus ring remains nonsingular at the origin and carries an electric octupole $a_{{30}}$ whose spherical Hankel near field $|h_3^{{(2)}}(k_0 r) / h_1^{{(2)}}(k_0 r)| \\propto (k_0 r)^{{-2}}$ equals **${nf5['hankel_h3_over_h1_at_25mm']:.2f}$** at $r = 25\\;\\mathrm{{mm}}$ ($k_0 r = 0.393$) and decays by **${nf5['hankel_octupole_decay_25_to_50mm']:.2f}\\times$** to **${nf5['hankel_h3_over_h1_at_50mm']:.2f}$** at $r = 50\\;\\mathrm{{mm}}$ ($k_0 r = 0.786$, still in the reactive/intermediate zone). Evaluated with the **exact same metric** across both zones, the relative electric-field difference $\\|\\,|\\mathbf{{E}}_{{\\mathrm{{toroid}}}}| - |\\mathbf{{E}}_{{\\mathrm{{dipole}}}}|\\,\\|_2 / \\|\\,|\\mathbf{{E}}_{{\\mathrm{{dipole}}}}|\\,\\|_2$ drops from **${nf5['near_zone_20_to_30mm_e_rel_diff'] * 100:.2f}\\%$** in the inner reactive zone $r \\in [20, 30]\\;\\mathrm{{mm}}$ ($k_0 r \\in [0.31, 0.47]$) to **${nf5['outer_zone_45_to_55mm_e_rel_diff'] * 100:.2f}\\%$** in the outer reactive/intermediate zone $r \\in [45, 55]\\;\\mathrm{{mm}}$ ($k_0 r \\in [0.71, 0.86]$) — a **${nf5['e_metric_reduction_factor']:.2f}\\times$ reduction** — while the magnetic-field metric $\\|\\,|H_{{y,\\mathrm{{toroid}}}}| - |H_{{y,\\mathrm{{dipole}}}}|\\,\\|_2 / \\|\\,|H_{{y,\\mathrm{{dipole}}}}|\\,\\|_2$ drops from **${nf5['near_zone_20_to_30mm_hy_rel_diff'] * 100:.2f}\\%$** to **${nf5['outer_zone_45_to_55mm_hy_rel_diff'] * 100:.2f}\\%$** (**${nf5['hy_metric_reduction_factor']:.2f}\\times$ reduction**).\n"
            f"2. **Electrical size ($k_0 R$ roll-off and the $j_1(k_0 R)$ null, Figure 6c):** As an **analytic estimate** from the ring multipole expansion, the dipole form factor $F_1(k_0 R)^2 = [3 j_1(k_0 R)/(k_0 R)]^2$ rolls off with electrical size $k_0 R = 2\\pi R/\\lambda_0$ while the octupole-to-dipole ratio grows as $|a_{{30}}|/|a_{{10}}| = \\sqrt{{7/8}}\\,|j_3(k_0 R)/j_1(k_0 R)|$ (verified by our 3D FIT sweep across $R \\in \\{{{int(r2['r_major_mm'][0])}, {int(r2['r_major_mm'][1])}, {int(r2['r_major_mm'][2])}, {int(r2['r_major_mm'][3])}\\}}\\;\\mathrm{{mm}}$, $k_0 R \\in [{r2['k0_r'][0]:.3f}, {r2['k0_r'][-1]:.3f}]$, where $|a_{{30}}/a_{{10}}|_{{\\mathrm{{sim}}}} = {r2['a30_over_a10_sim'][0]:.5f} \\to {r2['a30_over_a10_sim'][-1]:.5f}$ vs analytical ${r2['a30_over_a10_theory'][0]:.5f} \\to {r2['a30_over_a10_theory'][-1]:.5f}$). At the first zero of the spherical Bessel function $j_1(k_0 R)$ (**$k_0 R = {el5['kr_first_bessel_zero']:.4f}$**, i.e. circumference $2\\pi R = {el5['circumference_over_lambda_at_zero']:.4f}\\,\\lambda_0$, where $\\tan(k_0 R) = k_0 R$), the axial electric dipole radiation $a_{{10}}$ vanishes completely while higher odd electric multipoles ($a_{{30}}, a_{{50}}$) dominate. (By contrast, at the first zero of the cylindrical Bessel function $J_1(k_0 R)$, $k_0 R = {el5['kr_equatorial_j1_zero']:.4f}$, only the equatorial radiation at $\\theta = \\pi/2$ vanishes.)\n"
            f"3. **Resolved internal core $\\mathrm{{TE}}_{{01\\delta}}$ dielectric-magnetic mode sweep ($\\ge 20\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{core}}}}$, Figure 6c) and breakdown of point-dipole equivalence:** Resolving the core at $\\Delta_{{\\mathrm{{core}}}} \\in \\{{{te5[0]['dx_core_mm']:.2f}, {te1['dx_core_mm']:.2f}, {te2['dx_core_mm']:.2f}\\}}\\;\\mathrm{{mm}}$ (${te5[0]['cells_per_lambda_core']:.1f}, {te1['cells_per_lambda_core']:.1f}, {te2['cells_per_lambda_core']:.1f}\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{core}}}}$) across $\\varepsilon_{{\\mathrm{{core}}}} \\in \\{{{te5[0]['eps_core']:.0f}, {te1['eps_core']:.0f}, {te2['eps_core']:.0f}\\}}$ ($n_{{\\mathrm{{core}}}} \\in \\{{{te5[0]['n_core']:.2f}, {te1['n_core']:.2f}, {te2['n_core']:.2f}\\}}$) shows why the simple point-toroid power law $P_{{\\mathrm{{rad}}}} \\propto |T_z|^2$ breaks down when $\\varepsilon_{{\\mathrm{{core}}}} > 1$:\n"
            f"   - Below the internal $\\mathrm{{TE}}_{{01\\delta}}$ cutoff at $\\varepsilon_{{\\mathrm{{core}}}} = {te1['eps_core']:.0f}$ ($2r/\\lambda_{{\\mathrm{{core}}}} = {te1['two_r_over_lambda_core']:.3f} < 0.5$, $f_{{\\mathrm{{TE01}}}} = {te1['f_te01_ghz']:.2f}\\;\\mathrm{{GHz}} > f_0$), Faraday induction drives an axial electric field $E_z$ inside the dielectric core whose polarization current $\\mathbf{{J}}_{{\\mathrm{{pol}}}} = +\\mathrm{{i}}\\omega\\varepsilon_0(\\varepsilon_{{\\mathrm{{core}}}}-1)\\mathbf{{E}}$ creates a true primitive electric dipole $\\mathbf{{p}}_{{\\mathrm{{pol}}}} = \\varepsilon_0(\\varepsilon_{{\\mathrm{{core}}}}-1)\\int_{{\\mathrm{{core}}}}\\mathbf{{E}}\\,\\mathrm{{d}}V$ (${te1['p_pol_z_abs_cm']:.2e}\\;\\mathrm{{C\\,m}}$) in phase with the toroidal dipole $\\mathbf{{p}}_{{\\mathrm{{eff}}}}(\\mathbf{{T}}_{{\\mathrm{{tot}}}})$ (${te1['p_tor_eff_abs_cm']:.2e}\\;\\mathrm{{C\\,m}}$). Their coherent sum $|\\mathbf{{p}}_{{\\mathrm{{pol}}}} + \\mathbf{{p}}_{{\\mathrm{{eff}}}}|^2$ raises $P_{{\\mathrm{{rad}}}}$ by **${pr_ratio_te1:.2f}\\times$** (${te1['p_rad_sim_w']:.3e}\\;\\mathrm{{W}}$ vs hybrid dipole prediction ${te1['p_rad_hybrid_pred_w']:.3e}\\;\\mathrm{{W}}$, ${hyb_rel_err_te1:.1f}\\%$ agreement) while $|T_z(\\varepsilon_{{\\mathrm{{core}}}}=10)| / |T_z(\\varepsilon_{{\\mathrm{{core}}}}=1)|$ rises by **${te1['ratio_to_eps1']:.3f}\\times$** (${te1['ratio_to_quasistatic_formula']:.3f}\\times$ relative to the continuum quasi-static formula $T_z^{{\\mathrm{{qs}}}}$).\n"
            f"   - Above cutoff at $\\varepsilon_{{\\mathrm{{core}}}} = {te2['eps_core']:.0f}$ ($2r/\\lambda_{{\\mathrm{{core}}}} = {te2['two_r_over_lambda_core']:.3f} > 0.5$, $f_{{\\mathrm{{TE01}}}} = {te2['f_te01_ghz']:.3f}\\;\\mathrm{{GHz}} < f_0$), the internal $\\mathrm{{TE}}_{{01\\delta}}$ standing wave reverses sign across the minor cross-section and traps wave energy inside the high-index ($n_{{\\mathrm{{core}}}} = {te2['n_core']:.1f}$) core, causing $P_{{\\mathrm{{rad}}}}$ to drop to ${te2['p_rad_sim_w']:.3e}\\;\\mathrm{{W}}$ (${pr_ratio_te2:.3f}\\times$ of $\\varepsilon_{{\\mathrm{{core}}}}=1$) while $|T_z(\\varepsilon_{{\\mathrm{{core}}}}=22)| / |T_z(\\varepsilon_{{\\mathrm{{core}}}}=1)| = {te2['ratio_to_eps1']:.3f}$ (${te2['ratio_to_quasistatic_formula']:.3f}\\times$ vs $T_z^{{\\mathrm{{qs}}}}$)."
        ),
        "results_task_06": (
            f"Figure 7a–b benchmarks energy conservation, radiation efficiency $\\eta_{{\\mathrm{{rad}}}}$, and radiation quality factor $Q_{{\\mathrm{{rad}}}}$ across four Lorentz damping rates $\\gamma / 2\\pi \\in \\{{{loss6[0]['gamma_ghz']:.2f}, {loss6[1]['gamma_ghz']:.2f}, {loss6[2]['gamma_ghz']:.2f}, {loss6[3]['gamma_ghz']:.2f}\\}}\\;\\mathrm{{GHz}}$ ($\\tan\\delta_m \\in [{loss6[0]['tan_delta_m']:.2e}, {loss6[-1]['tan_delta_m']:.2e}]$):\n"
            f"- **Exact Poynting energy bookkeeping:** On the staggered FIT grid, the source input power $P_{{\\mathrm{{in}}}}$ (including the exact leapfrog half-step time-averaging factor $\\cos(\\omega_0\\Delta t/2)$ between electric fields at $t^n$ and currents at $t^{{n+1/2}}$) matches the sum of radiated power $P_{{\\mathrm{{rad}}}}$ and core magnetic dissipation $P_{{\\mathrm{{loss,core}}}} = \\frac{{1}}{{2}}\\tilde{{\\omega}}\\mu_0\\int \\mu''|\\mathbf{{H}}|^2\\,\\mathrm{{d}}V$ within **${closure_min_pct:.3f}\\%\\text{{–}}{closure_max_pct:.3f}\\%$** across all loss tangents! Shifting the source current phase by half a timestep in our mutation check causes the energy balance to fail with **${mut_min_pct:.0f}\\%\\text{{–}}{mut_max_pct:.0f}\\%$** error.\n"
            f"- **Efficiency & Chu bound:** Because $Q_{{\\mathrm{{rad}}}} \\approx {loss6[0]['q_rad_sim']:.0f}$ (analytical $Q_{{\\mathrm{{rad,th}}}} = {loss6[0]['q_rad_theory']:.1f}$ including $1/\\pi$ and $F_1(k_0 R)^2$) exceeds the enclosing-sphere Chu bound $Q_{{\\mathrm{{Chu}}}}(k_0 a_{{\\mathrm{{sph}}}}) = {loss6[0]['q_chu_enclosing_sphere']:.2f}$ by ${loss6[0]['q_ratio_over_chu_sim']:.1f}\\times$, even a modest magnetic loss tangent $\\tan\\delta_m \\sim 10^{{-3}}$ reduces $\\eta_{{\\mathrm{{rad}}}} = 1/(1 + Q_{{\\mathrm{{rad}}}}\\tan\\delta_m)$ substantially, matching our closed-form analytical curve within ${min(eta_rel_errs):.1f}\\%\\text{{–}}{max(eta_rel_errs):.1f}\\%$."
        ),
        "convergence_section": (
            f"Figure 7c and the table below report our 3-level spatial resolution study on the closed 8-winding dispersive magnetic toroid across three core resolutions that **all satisfy $\\ge 4$ cells across $2r = 10\\;\\mathrm{{mm}}$ and $\\ge 20\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{core}}}}$**: `Level 1 (Nominal)` $\\Delta_{{\\mathrm{{core}}}} = {l0['dx_core_mm']:.2f}\\;\\mathrm{{mm}}$ (${l0['cells_across_2r']:.1f}$ cells across $2r$, ${l0['cells_per_lambda_core']:.1f}\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{core}}}}$, ${l0['nx']}\\times {l0['ny']}\\times {l0['nz']} = {l0['n_cells']:,}$ cells), `Level 2 (Medium)` $\\Delta_{{\\mathrm{{core}}}} = {l1['dx_core_mm']:.2f}\\;\\mathrm{{mm}}$ (${l1['cells_across_2r']:.1f}$ cells across $2r$, ${l1['cells_per_lambda_core']:.1f}\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{core}}}}$, ${l1['nx']}\\times {l1['ny']}\\times {l1['nz']} = {l1['n_cells']:,}$ cells), and `Level 3 (Fine)` $\\Delta_{{\\mathrm{{core}}}} = {l2['dx_core_mm']:.2f}\\;\\mathrm{{mm}}$ (${l2['cells_across_2r']:.1f}$ cells across $2r$, ${l2['cells_per_lambda_core']:.1f}\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{core}}}}$, ${l2['nx']}\\times {l2['ny']}\\times {l2['nz']} = {l2['n_cells']:,}$ cells).\n\n"
            f"**Behavior across the 3 grids vs the continuum static formula $T_z^{{\\mathrm{{qs}}}} = {conv6['tz_exact_continuum_am3']:.4e}\\;\\mathrm{{A\\,m^3}}$:**\n"
            f"1. **Non-monotonic binary drive staircase ($\\pm 9\\%$) and lack of grid convergence:** On an unsmoothed binary staircase of the circular cross-section $(\\rho - R)^2 + z^2 \\le r^2$ and 8 angular sectors, the discrete static drive integral $T_z^{{\\mathrm{{disc}}}}(\\Delta) = \\frac{{1}}{{2}}\\operatorname{{Re}}(\\mu_{{\\mathrm{{disc}}}})\\int (\\mathbf{{r}}\\times\\mathbf{{H}}_{{\\mathrm{{drive}}}})_z\\,\\mathrm{{d}}V$ shifts by ${(l0['tz_disc_over_cont_static'] - 1.0) * 100:+.2f}\\%$ at $\\Delta = {l0['dx_core_mm']:.2f}\\;\\mathrm{{mm}}$ (${l0['tz_disc_static_drive_am3']:.4e}\\;\\mathrm{{A\\,m^3}}$), ${(l1['tz_disc_over_cont_static'] - 1.0) * 100:+.2f}\\%$ at $\\Delta = {l1['dx_core_mm']:.2f}\\;\\mathrm{{mm}}$ (${l1['tz_disc_static_drive_am3']:.4e}\\;\\mathrm{{A\\,m^3}}$), and ${(l2['tz_disc_over_cont_static'] - 1.0) * 100:+.2f}\\%$ at $\\Delta = {l2['dx_core_mm']:.2f}\\;\\mathrm{{mm}}$ (${l2['tz_disc_static_drive_am3']:.4e}\\;\\mathrm{{A\\,m^3}}$). Consequently, the simulated dynamic moment $|T_z^{{\\mathrm{{sim}}}}|$ varies non-monotonically as **${l0['tz_tot_abs_am3']:.3e} \\to {l1['tz_tot_abs_am3']:.3e} \\to {l2['tz_tot_abs_am3']:.3e}\\;\\mathrm{{A\\,m^3}}$** (${(l0['tz_tot_abs_am3'] / conv6['tz_exact_continuum_am3'] - 1.0) * 100:+.2f}\\%, {(l1['tz_tot_abs_am3'] / conv6['tz_exact_continuum_am3'] - 1.0) * 100:+.2f}\\%, {(l2['tz_tot_abs_am3'] / conv6['tz_exact_continuum_am3'] - 1.0) * 100:+.2f}\\%$ vs $T_z^{{\\mathrm{{qs}}}}$) and the radiated power varies as **$P_{{\\mathrm{{rad}}}} = {l0['p_rad_n2f_w']:.3e} \\to {l1['p_rad_n2f_w']:.3e} \\to {l2['p_rad_n2f_w']:.3e}\\;\\mathrm{{W}}$** (**${conv6['p_rad_rel_change_fine_vs_med'] * 100:.0f}\\%$** on the finest grid vs Level 2, or ${conv6['p_rad_rel_change_fine_vs_nom'] * 100:.1f}\\%$ vs Level 1). Evaluating the stated grid-convergence criterion $|\\Delta P_{{\\mathrm{{rad}}}}|/P_{{\\mathrm{{rad}}}} < {crit_rel_dp_pct:.1f}\\%$ between the last two levels gives $|{conv6['p_rad_rel_change_fine_vs_med'] * 100:.1f}\\%| \\ge {crit_rel_dp_pct:.1f}\\%$, so `is_grid_converged = {bool(abs(conv6['p_rad_rel_change_fine_vs_med']) < conv6.get('grid_convergence_criterion_rel_dp_max', 0.02))}` is computed directly from the data, we do not apply Richardson extrapolation, and the $\\mathcal{{O}}(\\Delta_{{\\mathrm{{core}}}}^2)$ guide in Figure 7c is shown as a reference slope only.\n"
            f"2. **Dynamic residual $|T_z^{{\\mathrm{{sim}}}}|/T_z^{{\\mathrm{{disc}}}}$ vs analytic interior factor:** Relative to the discrete static drive integral $T_z^{{\\mathrm{{disc}}}}(\\Delta)$ on each grid, the simulated dynamic moment has residual ratio $|T_z^{{\\mathrm{{sim}}}}(\\Delta)| / T_z^{{\\mathrm{{disc}}}}(\\Delta) = {l0['tz_sim_over_disc_residual_ratio']:.4f}, {l1['tz_sim_over_disc_residual_ratio']:.4f}, {l2['tz_sim_over_disc_residual_ratio']:.4f}$ ($+{(l0['tz_sim_over_disc_residual_ratio'] - 1.0) * 100:.1f}\\%, +{(l1['tz_sim_over_disc_residual_ratio'] - 1.0) * 100:.1f}\\%, +{(l2['tz_sim_over_disc_residual_ratio'] - 1.0) * 100:.1f}\\%$). The **analytic estimate** for the fundamental interior Bessel enhancement inside the circular cross-section at $f_0 = {F0_HZ / 1e9:.2f}\\;\\mathrm{{GHz}}$ is $1 + (k_c r)^2/8 = {conv6['analytic_interior_factor_f0']:.3f}$ ($+{(conv6['analytic_interior_factor_f0'] - 1.0) * 100:.1f}\\%$), which **does not explain** the full $+4.9\\%\\text{{–}}8.3\\%$ dynamic residual on these coarse cross-section grids. Meanwhile, on every grid level the internal multipole-to-far-field self-consistency $|P_{{\\mathrm{{rad}}}}^{{\\mathrm{{N2F}}}} - P_{{\\mathrm{{dip}}}}(T_{{\\mathrm{{sim}}}})| / P_{{\\mathrm{{dip}}}}(T_{{\\mathrm{{sim}}}})$ holds within **${l0['p_rad_self_consistency_rel_diff'] * 100:.2f}\\%, {l1['p_rad_self_consistency_rel_diff'] * 100:.2f}\\%, {l2['p_rad_self_consistency_rel_diff'] * 100:.2f}\\%$**."
        ),
        "convergence_table": convergence_table_md,
        "quantitative_summary_table": build_quantitative_summary_markdown(data_dir=out_dir),
        "limits_and_failure_modes": (
            f"Every physical equivalence and quasi-static approximation relied upon in this tutorial has strict validity conditions that we tested with explicit counter-cases:\n\n"
            f"1. **Assumption 1 — Closed, balanced azimuthal magnetization loop ($\\nabla\\cdot\\mathbf{{M}} = 0$, $C_{{Nv}}$ feed symmetry with $N \\ge 4$):**\n"
            f"   - *Condition:* Vanishing of the magnetic dipole moment ($\\mathbf{{m}} = \\mathbf{{0}}$) and of all $\\mathrm{{TE}}$ multipoles ($b_{{1m}} = 0$) requires both an unbroken magnetic flux path and balanced in-phase excitation around the perimeter.\n"
            f"   - *Counter-cases tested (Task 4, Figure 5):* Cutting a $60^\\circ$ air gap in the permeable core creates magnetic poles on the cut faces and drives $|b_1|/|a_1| = {b4['core_air_gap_60deg']['b1_over_a1']:.3f}$; disconnecting just 1 of 8 feeds [3] raises $|b_1|/|a_1|$ from ${b4['closed_8_feeds']['b1_over_a1']:.2e}$ to ${b4['seven_of_eight_feeds']['b1_over_a1']:.4f}$; reversing the phase on half the torus (`Mode 1`) cancels $a_1$ to ${b4['mode1_antiphase']['a1_over_a1_closed']:.3e}\\times |a_1|_{{\\mathrm{{closed}}}}$ while driving a transverse magnetic dipole $|b_1| = {b4['mode1_antiphase']['b1_over_a1_closed']:.3f}\\times |a_1|_{{\\mathrm{{closed}}}}$; removing 1 of 4 rods in the square raises $|b_1|/|a_1|$ from ${b4['four_rods_closed']['b1_over_a1']:.2e}$ to ${b4['three_of_four_rods']['b1_over_a1']:.3f}$; and winding the poloidal coil as a 3D helical wire on edges without Marinov's reverse equatorial loop [1, 11] leaves a 1-turn azimuthal current with $|b_1|/|a_1| = {b4['helical_uncompensated']['b1_over_a1']:.4f}$, which passes through a sharp minimum of ${b4['helical_marinov_compensated']['b1_over_a1']:.2e}$ (${msw4['suppression_factor_at_r1']:.0f}\\times$ suppression, and ${msw4['improvement_over_naive_r_major']:.1f}\\times$ lower than at $R = {msw4['r_major_mm']:.1f}\\;\\mathrm{{mm}}$) when compensated by a planar $z = 0$ return wire loop of radius $R_1 = R\\sqrt{{1 + \\frac{{1}}{{2}}(r/R)^2}} = {msw4['r1_marinov_mm']:.2f}\\;\\mathrm{{mm}}$ driven by the same current $-I_{{\\mathrm{{turn}}}}$." + "\n"
            f"2. **Assumption 2 — Far-field observation ($r \\gg R$) vs reactive near-field topology ($r \\sim R$):**\n"
            f"   - *Condition:* Outside the bounding sphere $r > a_{{\\mathrm{{sph}}}} = R + r$, the toroid field equals the full dipole field (including its $1/r^3$ and $1/r^2$ terms) plus odd-$l \\ge 3$ electric multipoles ($a_{{30}}, a_{{50}}, \\dots$).\n"
            f"   - *Counter-case tested (Task 5, Figure 6a–b):* Inside and near the bounding sphere ($r \\in [20, 30]\\;\\mathrm{{mm}}$), the point electric dipole diverges as $1/r^3$ toward the origin while the finite torus ring stays bounded at $r = 0$ and carries strong octupole near fields $\\propto h_3^{{(2)}}(k_0 r) / h_1^{{(2)}}(k_0 r)$ (**${nf5['hankel_h3_over_h1_at_25mm']:.2f}$** at $25\\;\\mathrm{{mm}}$ vs **${nf5['hankel_h3_over_h1_at_50mm']:.2f}$** at $50\\;\\mathrm{{mm}}$, a **${nf5['hankel_octupole_decay_25_to_50mm']:.2f}\\times$** decay), producing a **${nf5['near_zone_20_to_30mm_e_rel_diff'] * 100:.2f}\\%$** near-zone $|\\mathbf{{E}}|$ difference that decays by **${nf5['e_metric_reduction_factor']:.2f}\\times$** to **${nf5['outer_zone_45_to_55mm_e_rel_diff'] * 100:.2f}\\%$** in the outer reactive/intermediate zone $r \\in [45, 55]\\;\\mathrm{{mm}}$ (**${nf5['near_zone_20_to_30mm_hy_rel_diff'] * 100:.2f}\\% \\to {nf5['outer_zone_45_to_55mm_hy_rel_diff'] * 100:.2f}\\%$** in $|H_y|$, an **${nf5['hy_metric_reduction_factor']:.2f}\\times$** reduction).\n"
            f"3. **Assumption 3 — Electrically thin core cross-section ($2r \\ll \\lambda_{{\\mathrm{{core}}}} = \\lambda_0 / \\sqrt{{\\varepsilon_{{\\mathrm{{core}}}}\\mu_r}}$) and electrically small ring ($k_0 R \\ll {el5['kr_first_bessel_zero']:.4f}$):**\n"
            f"   - *Condition:* Marinov's quasi-static toroidal moment formula $T_z \\approx \\frac{{1}}{{2}}\\pi N I \\mu_{{\\mathrm{{eff}}}} r^2 R$ assumes uniform in-phase magnetization across the core cross-section, and the dipole radiation formula $P_{{\\mathrm{{rad}}}} \\propto |T_z|^2$ assumes $\\varepsilon_{{\\mathrm{{core}}}} = 1$ and $k_0 R \\ll 1$.\n"
            f"   - *Counter-cases tested (Task 5, Figure 6c):* When $\\varepsilon_{{\\mathrm{{core}}}} = {te1['eps_core']:.0f}$ ($2r/\\lambda_{{\\mathrm{{core}}}} = {te1['two_r_over_lambda_core']:.3f} < 0.5$, resolved at ${te1['cells_per_lambda_core']:.1f}\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{core}}}}$), induced dielectric polarization currents create a primitive electric dipole $\\mathbf{{p}}_{{\\mathrm{{pol}}}}$ in phase with $\\mathbf{{p}}_{{\\mathrm{{eff}}}}(\\mathbf{{T}}_{{\\mathrm{{tot}}}})$, raising $P_{{\\mathrm{{rad}}}}$ by ${pr_ratio_te1:.2f}\\times$ while $|T_z(\\varepsilon_{{\\mathrm{{core}}}})|/|T_z(\\varepsilon_{{\\mathrm{{core}}}}=1)|$ rises by ${te1['ratio_to_eps1']:.3f}\\times$; when $\\varepsilon_{{\\mathrm{{core}}}} = {te2['eps_core']:.0f}$ ($2r/\\lambda_{{\\mathrm{{core}}}} = {te2['two_r_over_lambda_core']:.3f} > 0.5$, resolved at ${te2['cells_per_lambda_core']:.1f}\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{core}}}}$), crossing the internal $\\mathrm{{TE}}_{{01\\delta}}$ cutoff traps wave energy inside the high-index core so $P_{{\\mathrm{{rad}}}}$ drops to ${pr_ratio_te2:.3f}\\times$ of $\\varepsilon_{{\\mathrm{{core}}}}=1$ while $|T_z(\\varepsilon_{{\\mathrm{{core}}}}=22)|/|T_z(\\varepsilon_{{\\mathrm{{core}}}}=1)| = {te2['ratio_to_eps1']:.3f}$. Likewise, when the ring electrical radius reaches the first zero of $j_1(k_0 R)$ at **$k_0 R = {el5['kr_first_bessel_zero']:.4f}$** ($3 j_1(k_0 R)/(k_0 R) = 0$), the $\\mathrm{{TM}}_{{10}}$ dipole radiation vanishes completely and higher odd electric multipoles ($a_{{30}}, a_{{50}}$) take over.\n"
            f"4. **Assumption 4 — High-$\\mu$, low-loss magnetic core well below ferromagnetic resonance ($f_0 \\ll f_{{\\mathrm{{res}}}}$, Snoek's limit) and sub-cell geometry representation:**\n"
            f"   - *Condition:* Because $Q_{{\\mathrm{{rad}}}}^{{(\\mathrm{{toroid}})}} / Q_{{\\mathrm{{Chu}}}} \\propto [\\mu'(k_0 R)^2]^{{-1}} \\gg 1$ ($Q_{{\\mathrm{{rad}}}} = {loss6[0]['q_rad_sim']:.1f}$ vs $Q_{{\\mathrm{{Chu}}}} = {loss6[0]['q_chu_enclosing_sphere']:.2f}$ at $f_0 = {F0_HZ / 1e9:.2f}\\;\\mathrm{{GHz}}$), achieving high radiation efficiency $\\eta_{{\\mathrm{{rad}}}} = (1 + Q_{{\\mathrm{{rad}}}}\\tan\\delta_m)^{{-1}}$ requires $\\tan\\delta_m \\ll Q_{{\\mathrm{{rad}}}}^{{-1}} \\approx 4\\times 10^{{-4}}$.\n"
            f"   - *Counter-cases tested (Task 6, Figure 7a,c):* In a commercially available NiZn ferrite or ferromagnetic alloy laminate operating near GHz frequencies, Snoek's law [8] links higher static permeability $\\mu_{{\\mathrm{{DC}}}}$ to a lower ferromagnetic resonance frequency $f_{{\\mathrm{{res}}}}$ and rising magnetic loss tangent $\\tan\\delta_m = \\mu''/\\mu' \\gtrsim 10^{{-2}}$, which reduces $\\eta_{{\\mathrm{{rad}}}}$ to ${loss6[-1]['eta_rad_sim'] * 100:.2f}\\%$ at $\\gamma / 2\\pi = {loss6[-1]['gamma_ghz']:.2f}\\;\\mathrm{{GHz}}$ ($\\tan\\delta_m = {loss6[-1]['tan_delta_m']:.2e}$). Furthermore, on Cartesian grids with 4–6 cells across $2r$, the unsmoothed binary staircase of the circular core cross-section oscillates by $\\pm 9\\%$ in volume, causing $|T_z^{{\\mathrm{{sim}}}}|$ and $P_{{\\mathrm{{rad}}}}$ (${conv6['p_rad_rel_change_fine_vs_med'] * 100:.0f}\\%$ on the finest grid vs Level 2) not to converge monotonically without sub-cell material/source averaging. Likewise, in Task 4 the 3D helical coil and Marinov return loop are modelled as infinitely thin current filaments deposited on primary grid edges (`deposit_closed_wire_on_edges`, no physical wire diameter is modelled, so the $\\ge 4$-cells-per-wire rule does not apply; the compensated residual floor $|b_1|/|a_1| = {b4['helical_marinov_compensated']['b1_over_a1']:.2e}$ is set by grid-edge deposition and staircase discretization error)."
        ),
    }


def generate_all_figures(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> dict[str, Any]:
    """Generate all setup schematics, static diagnostic figures, and the animated wave GIF."""
    setup_out = plot_00_setup_figures(fig_dir=fig_dir)
    f1 = plot_01_baseline_dipoles(data_dir=data_dir, fig_dir=fig_dir)
    f2 = plot_02_closed_toroid_fields(data_dir=data_dir, fig_dir=fig_dir)
    f3 = plot_03_toroid_multipoles_and_scaling(data_dir=data_dir, fig_dir=fig_dir)
    f4 = plot_04_square_four_rods(data_dir=data_dir, fig_dir=fig_dir)
    f5 = plot_05_break_the_loop(data_dir=data_dir, fig_dir=fig_dir)
    f6 = plot_06_equivalence_limits(data_dir=data_dir, fig_dir=fig_dir)
    f7 = plot_07_efficiency_q_and_convergence(data_dir=data_dir, fig_dir=fig_dir)
    gif = plot_08_toroid_vs_dipole_wave_gif(data_dir=data_dir, fig_dir=fig_dir)
    return {
        "setup": setup_out,
        "figures": [str(f1), str(f2), str(f3), str(f4), str(f5), str(f6), str(f7)],
        "gif": str(gif),
    }


