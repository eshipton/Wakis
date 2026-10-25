"""Physics, relativistic particle pushers, 3D FIT runners, and plotting helpers for Tutorial 8.3.

Laser acceleration of electrons in vacuum — from the Lawson-Woodward theorem
to boundary-terminated semi-infinite vacuum acceleration (Huang, Stupakov &
Zolotorev, PRSTAB 7, 011302, 2004; Plettner et al., PRL 95, 134801, 2005) and
ponderomotive acceleration/focusing of attosecond electron bunches for
laser-driven X-ray FELs (Stupakov & Zolotorev, PRL 86, 5274, 2001; Zolotorev,
NIM A 483, 445, 2002).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.constants import alpha as ALPHA_FINE
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0, hbar, m_e, mu_0
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from wakis import GridFIT3D, SolverFIT3D
from wakis.conformal import rectangle_sheet
from wakis.estimate import estimate, preflight
from wakis.visualize import (
    box_bounds,
    cpml_inner_bounds,
    plot_scene,
    setup_rendering,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

Z0_VAC = float(np.sqrt(mu_0 / epsilon_0))
M_E_JOULE = float(m_e * c_light**2)
M_E_EV = float(M_E_JOULE / q_e)
LAMBDA_C_BAR_M = float(hbar / (m_e * c_light))  # ~3.86159e-13 m
DEFAULT_MEM_LIMIT_MB = 2000.0
DEFAULT_LAMBDA_L_M = 0.8e-6  # Ti:sapphire 800 nm (Plettner et al. 2005; Zolotorev 2002)


# ---------------------------------------------------------------------------
# Part A: Relativistic electron in a plane wave & Boris / Vay particle pushers
# ---------------------------------------------------------------------------


def plane_wave_vector_potential_and_fields(
    eta_rad: float | np.ndarray,
    a0: float,
    omega0_rad_s: float,
    sigma_eta_rad: float | None = None,
    carrier_phase_rad: float = 0.0,
) -> dict[str, np.ndarray]:
    """Evaluate normalized vector potential a_x(eta) and fields E_x, B_y of a 1D plane wave.

    For a plane wave propagating along +z in vacuum, all fields depend only on the
    Lorentz-invariant phase ``eta = omega_0 * (t - z / c)``. With normalized vector
    potential ``a_x(eta) = e * A_x(eta) / (m_e * c)``, the electric and magnetic
    fields are ``E_x(eta) = -(m_e * c * omega_0 / e) * da_x/deta`` and
    ``B_y(eta) = E_x(eta) / c``.

    Parameters
    ----------
    eta_rad : float or ndarray
        Invariant wave phase ``omega_0 * (t - z / c)`` [rad].
    a0 : float
        Peak normalized vector potential ``e * E_0 / (m_e * c * omega_0)`` [-].
    omega0_rad_s : float
        Laser angular frequency ``omega_0 = 2 * pi * c / lambda_L`` [rad/s].
    sigma_eta_rad : float or None, optional
        Gaussian pulse envelope RMS phase width [rad]. If ``None``, a continuous
        monochromatic wave ``a_x(eta) = a0 * sin(eta + carrier_phase_rad)`` is used.
    carrier_phase_rad : float, optional
        Carrier phase offset [rad]. Default is ``0.0``.

    Returns
    -------
    out : dict of ndarray
        Dictionary with keys:
        - ``a_x`` : normalized vector potential ``a_x(eta)`` [-]
        - ``da_x_deta`` : derivative ``da_x / deta`` [1/rad]
        - ``Ex`` : electric field ``E_x(eta)`` [V/m]
        - ``By`` : magnetic field ``B_y(eta) = E_x(eta) / c`` [T]
    """
    eta = np.asarray(eta_rad, dtype=float)
    phase = eta + float(carrier_phase_rad)
    if sigma_eta_rad is None or np.isinf(sigma_eta_rad):
        env = np.ones_like(eta)
        denv_deta = np.zeros_like(eta)
    else:
        sig = float(sigma_eta_rad)
        env = np.exp(-0.5 * (eta / sig) ** 2)
        denv_deta = -(eta / (sig**2)) * env

    a_x = float(a0) * env * np.sin(phase)
    da_x = float(a0) * (denv_deta * np.sin(phase) + env * np.cos(phase))
    e_scale = (m_e * c_light * float(omega0_rad_s)) / q_e
    ex = -e_scale * da_x
    by = ex / c_light
    return {
        "a_x": a_x,
        "da_x_deta": da_x,
        "Ex": ex,
        "By": by,
    }


def plane_wave_exact_solution(
    eta_rad: np.ndarray,
    a0: float,
    omega0_rad_s: float,
    sigma_eta_rad: float | None = None,
    carrier_phase_rad: float = 0.0,
) -> dict[str, np.ndarray]:
    """Exact relativistic trajectory of an electron initially at rest in a plane wave.

    From canonical transverse momentum conservation ``u_x(eta) = a_x(eta)``,
    ``u_y(eta) = 0``, and light-cone energy-momentum conservation
    ``gamma(eta) - u_z(eta) = 1`` for an electron initially at rest
    (``u_0 = 0, gamma_0 = 1`` at ``eta -> -inf``), the exact 4-velocity and
    Lorentz factor as functions of ``eta = omega_0 * (t - z / c)`` are:

    - ``u_x(eta) = a_x(eta)``
    - ``u_z(eta) = 0.5 * a_x(eta)**2``
    - ``gamma(eta) = 1 + 0.5 * a_x(eta)**2``

    Since ``deta / dt = omega_0 * (1 - beta_z) = omega_0 / gamma(eta)``, the
    laboratory coordinates follow by quadrature over ``eta``:
    ``omega_0 * x(eta) = c * int u_x(eta') deta'``,
    ``omega_0 * z(eta) = c * int u_z(eta') deta'``, and
    ``omega_0 * t(eta) = eta + omega_0 * z(eta) / c``.

    Parameters
    ----------
    eta_rad : ndarray
        Uniformly spaced invariant phase grid [rad].
    a0 : float
        Peak normalized vector potential [-].
    omega0_rad_s : float
        Laser angular frequency [rad/s].
    sigma_eta_rad : float or None, optional
        Gaussian pulse RMS phase width [rad].
    carrier_phase_rad : float, optional
        Carrier phase offset [rad].

    Returns
    -------
    out : dict of ndarray
        Dictionary with keys ``eta_rad``, ``t_s``, ``x_m``, ``z_m``, ``ux``,
        ``uy``, ``uz``, ``gamma``, ``a_x``, ``Ex``, ``By``.
    """
    eta = np.asarray(eta_rad, dtype=float)
    pw = plane_wave_vector_potential_and_fields(
        eta,
        a0=a0,
        omega0_rad_s=omega0_rad_s,
        sigma_eta_rad=sigma_eta_rad,
        carrier_phase_rad=carrier_phase_rad,
    )
    a_x = pw["a_x"]
    ux = a_x.copy()
    uy = np.zeros_like(eta)
    uz = 0.5 * a_x**2
    gamma = 1.0 + uz

    k0 = float(omega0_rad_s) / c_light
    if sigma_eta_rad is None or np.isinf(sigma_eta_rad):
        ph = eta + float(carrier_phase_rad)
        ph0 = eta[0] + float(carrier_phase_rad)
        x_m = (float(a0) / k0) * (np.cos(ph0) - np.cos(ph))
        z_m = (0.25 * float(a0) ** 2 / k0) * (
            (ph - ph0) - 0.5 * (np.sin(2.0 * ph) - np.sin(2.0 * ph0))
        )
    else:
        # Cumulative Simpson/trapezoidal rule on dense phase grid
        deta = np.diff(eta)
        x_m = np.zeros_like(eta)
        z_m = np.zeros_like(eta)
        x_m[1:] = np.cumsum(0.5 * (ux[:-1] + ux[1:]) * deta) / k0
        z_m[1:] = np.cumsum(0.5 * (uz[:-1] + uz[1:]) * deta) / k0

    t_s = (eta / float(omega0_rad_s)) + (z_m / c_light)
    return {
        "eta_rad": eta,
        "t_s": t_s,
        "x_m": x_m,
        "z_m": z_m,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "gamma": gamma,
        "a_x": a_x,
        "Ex": pw["Ex"],
        "By": pw["By"],
    }


def figure_of_eight_orbit(eta_rad: np.ndarray, a0: float, k0_rad_m: float) -> dict[str, np.ndarray]:
    """Closed figure-of-eight (lemniscate) orbit in the average rest frame of a CW plane wave.

    Subtracting the secular longitudinal drift ``z_bar(eta) = (a_0^2 / (4 * k_0)) * eta``
    leaves the periodic coordinates ``k_0 * x(eta) = -a_0 * cos(eta)`` and
    ``k_0 * z_tilde(eta) = -(a_0^2 / 8) * sin(2 * eta)``, which satisfy the
    algebraic lemniscate ``16 * (k_0 * z_tilde)^2 = (k_0 * x)^2 * (a_0^2 - (k_0 * x)^2)``.

    Parameters
    ----------
    eta_rad : ndarray
        Phase array [rad].
    a0 : float
        Normalized vector potential [-].
    k0_rad_m : float
        Laser wavenumber ``k_0 = 2 * pi / lambda_L`` [rad/m].

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``x_m``, ``z_tilde_m``, ``kx``, ``kz_tilde``, and ``lemniscate_residual``.
    """
    eta = np.asarray(eta_rad, dtype=float)
    a = float(a0)
    k0 = float(k0_rad_m)
    kx = -a * np.cos(eta)
    kz_tilde = -0.125 * (a**2) * np.sin(2.0 * eta)
    residual = 16.0 * kz_tilde**2 - (kx**2) * (a**2 - kx**2)
    return {
        "x_m": kx / k0,
        "z_tilde_m": kz_tilde / k0,
        "kx": kx,
        "kz_tilde": kz_tilde,
        "lemniscate_residual": residual,
    }


def push_boris_relativistic(
    r0_m: np.ndarray,
    u0: np.ndarray,
    t_grid_s: np.ndarray,
    field_fn: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    charge_c: float = -q_e,
    mass_kg: float = m_e,
) -> dict[str, np.ndarray]:
    """Relativistic Boris particle pusher in prescribed electromagnetic fields (E, B).

    Uses a leapfrog time-staggered layout where positions ``r^n`` live at integer
    times ``t_n`` and dimensionless 4-velocities ``u = gamma * v / c`` are advanced
    across ``t_n`` via half-step electric kicks and an exact norm-preserving
    magnetic rotation with ``t_vec = (q * dt / (2 * m * gamma_-)) * B(r^n, t_n)``.
    To start and end synchronously with ``r(t_grid_s)``, the initial momentum is
    half-step rewound at ``t_0`` and half-step synchronized at each saved step.

    Parameters
    ----------
    r0_m : ndarray, shape (3,)
        Initial position ``(x0, y0, z0)`` [m] at ``t_grid_s[0]``.
    u0 : ndarray, shape (3,)
        Initial dimensionless 4-velocity ``u_0 = gamma_0 * v_0 / c`` [-] at ``t_grid_s[0]``.
    t_grid_s : ndarray, shape (N_t,)
        Uniform time grid [s].
    field_fn : callable
        Function ``field_fn(r_m, t_s) -> (E_v_m, B_t)`` returning 3-vectors ``E`` [V/m]
        and ``B`` [T].
    charge_c : float, optional
        Particle charge [C]. Default is ``-e`` (electron).
    mass_kg : float, optional
        Particle rest mass [kg]. Default is ``m_e``.

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``t_s``, ``r_m`` (shape ``(N_t, 3)``), ``u`` (shape ``(N_t, 3)``),
        and ``gamma`` (shape ``(N_t,)``).
    """
    t_arr = np.asarray(t_grid_s, dtype=float)
    n_t = len(t_arr)
    dt = float(t_arr[1] - t_arr[0])
    q_over_2mc = float(charge_c) * dt / (2.0 * float(mass_kg) * c_light)

    r_hist = np.zeros((n_t, 3), dtype=float)
    u_hist = np.zeros((n_t, 3), dtype=float)
    g_hist = np.zeros(n_t, dtype=float)

    r = np.asarray(r0_m, dtype=float).copy()
    u_sync = np.asarray(u0, dtype=float).copy()
    r_hist[0] = r
    u_hist[0] = u_sync
    g_hist[0] = np.sqrt(1.0 + np.dot(u_sync, u_sync))

    # Half-step backward kick to place u at t_0 - dt/2
    e0, b0 = field_fn(r, t_arr[0])
    g0 = g_hist[0]
    t_vec0 = q_over_2mc * c_light * b0 / g0
    u_half = u_sync - q_over_2mc * (e0 + c_light * np.cross(u_sync / g0, b0))

    for n in range(n_t - 1):
        t_n = t_arr[n]
        e_n, b_n = field_fn(r, t_n)

        # First half electric kick
        u_minus = u_half + q_over_2mc * e_n
        gamma_minus = np.sqrt(1.0 + np.dot(u_minus, u_minus))

        # Boris magnetic rotation
        t_vec = (q_over_2mc * c_light / gamma_minus) * b_n
        s_vec = 2.0 * t_vec / (1.0 + np.dot(t_vec, t_vec))
        u_prime = u_minus + np.cross(u_minus, t_vec)
        u_plus = u_minus + np.cross(u_prime, s_vec)

        # Second half electric kick -> u at t_n + dt/2
        u_half = u_plus + q_over_2mc * e_n
        gamma_half = np.sqrt(1.0 + np.dot(u_half, u_half))

        # Advance position to t_{n+1}
        r = r + (c_light * dt / gamma_half) * u_half
        r_hist[n + 1] = r

        # Synchronize u at t_{n+1} via half-step forward preview at (r^{n+1}, t_{n+1})
        e_next, b_next = field_fn(r, t_arr[n + 1])
        u_m = u_half + 0.5 * q_over_2mc * e_next
        g_m = np.sqrt(1.0 + np.dot(u_m, u_m))
        t_v = 0.5 * (q_over_2mc * c_light / g_m) * b_next
        s_v = 2.0 * t_v / (1.0 + np.dot(t_v, t_v))
        u_p = u_m + np.cross(u_m, t_v)
        u_sync_next = (u_m + np.cross(u_p, s_v)) + 0.5 * q_over_2mc * e_next
        u_hist[n + 1] = u_sync_next
        g_hist[n + 1] = np.sqrt(1.0 + np.dot(u_sync_next, u_sync_next))

    return {
        "t_s": t_arr,
        "r_m": r_hist,
        "u": u_hist,
        "gamma": g_hist,
    }


def push_vay_relativistic(
    r0_m: np.ndarray,
    u0: np.ndarray,
    t_grid_s: np.ndarray,
    field_fn: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    charge_c: float = -q_e,
    mass_kg: float = m_e,
) -> dict[str, np.ndarray]:
    """Relativistic Vay (2008) particle pusher preserving Exact E x B cancellation.

    Unlike the standard Boris split, which evaluates the magnetic rotation using
    ``gamma_-`` before the second half electric kick, the Vay pusher (J.-L. Vay,
    Phys. Plasmas 15, 056701, 2008) solves the implicit midpoint Lorentz force
    algebraically for ``u^{n+1/2}`` and ``gamma^{n+1/2}``, preserving the exact
    ``E + v x B = 0`` cancellation in relativistic plane waves and crossed fields.

    Parameters
    ----------
    r0_m : ndarray, shape (3,)
        Initial position ``(x0, y0, z0)`` [m] at ``t_grid_s[0]``.
    u0 : ndarray, shape (3,)
        Initial dimensionless 4-velocity ``u_0 = gamma_0 * v_0 / c`` [-] at ``t_grid_s[0]``.
    t_grid_s : ndarray, shape (N_t,)
        Uniform time grid [s].
    field_fn : callable
        Function ``field_fn(r_m, t_s) -> (E_v_m, B_t)`` returning 3-vectors ``E`` [V/m]
        and ``B`` [T].
    charge_c : float, optional
        Particle charge [C]. Default is ``-e``.
    mass_kg : float, optional
        Particle rest mass [kg]. Default is ``m_e``.

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``t_s``, ``r_m`` (shape ``(N_t, 3)``), ``u`` (shape ``(N_t, 3)``),
        and ``gamma`` (shape ``(N_t,)``).
    """
    t_arr = np.asarray(t_grid_s, dtype=float)
    n_t = len(t_arr)
    dt = float(t_arr[1] - t_arr[0])
    q_over_2mc = float(charge_c) * dt / (2.0 * float(mass_kg) * c_light)

    r_hist = np.zeros((n_t, 3), dtype=float)
    u_hist = np.zeros((n_t, 3), dtype=float)
    g_hist = np.zeros(n_t, dtype=float)

    r = np.asarray(r0_m, dtype=float).copy()
    u_sync = np.asarray(u0, dtype=float).copy()
    r_hist[0] = r
    u_hist[0] = u_sync
    g_hist[0] = np.sqrt(1.0 + np.dot(u_sync, u_sync))

    def _vay_kick(u_in: np.ndarray, e_vec: np.ndarray, b_vec: np.ndarray, scale: float) -> np.ndarray:
        g_in = np.sqrt(1.0 + np.dot(u_in, u_in))
        tau = scale * c_light * b_vec
        u_prime = u_in + scale * (2.0 * e_vec + np.cross(u_in / g_in, c_light * b_vec))
        tau2 = float(np.dot(tau, tau))
        u_star = float(np.dot(u_prime, tau))
        sigma = 1.0 + float(np.dot(u_prime, u_prime)) - tau2
        g_new = np.sqrt(0.5 * (sigma + np.sqrt(max(sigma**2 + 4.0 * (tau2 + u_star**2), 0.0))))
        t_v = tau / g_new
        s_fac = 1.0 / (1.0 + float(np.dot(t_v, t_v)))
        return s_fac * (u_prime + float(np.dot(u_prime, t_v)) * t_v + np.cross(u_prime, t_v))

    e0, b0 = field_fn(r, t_arr[0])
    u_half = _vay_kick(u_sync, e0, b0, -0.5 * q_over_2mc)

    for n in range(n_t - 1):
        e_n, b_n = field_fn(r, t_arr[n])
        u_half = _vay_kick(u_half, e_n, b_n, q_over_2mc)
        gamma_half = np.sqrt(1.0 + np.dot(u_half, u_half))
        r = r + (c_light * dt / gamma_half) * u_half
        r_hist[n + 1] = r

        e_next, b_next = field_fn(r, t_arr[n + 1])
        u_sync_next = _vay_kick(u_half, e_next, b_next, 0.5 * q_over_2mc)
        u_hist[n + 1] = u_sync_next
        g_hist[n + 1] = np.sqrt(1.0 + np.dot(u_sync_next, u_sync_next))

    return {
        "t_s": t_arr,
        "r_m": r_hist,
        "u": u_hist,
        "gamma": g_hist,
    }


def run_task_part_a_plane_wave(quick: bool = False) -> dict:
    """Run Part A: relativistic electron in a plane wave, figure-of-eight orbit, and pusher convergence."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()

    lambda_m = DEFAULT_LAMBDA_L_M
    k0 = 2.0 * np.pi / lambda_m
    omega0 = k0 * c_light
    t_opt = lambda_m / c_light

    # 1. Finite Gaussian-envelope plane-wave pulse (a0 = 2.5 -> gamma_max = 1 + a0^2/2 = 4.125)
    a0_pulse = 2.5
    sigma_eta = 3.0 * (2.0 * np.pi)  # 3 optical cycles RMS phase width
    eta_dense = np.linspace(-4.8 * sigma_eta, 4.8 * sigma_eta, 8001)
    exact_pulse = plane_wave_exact_solution(
        eta_dense,
        a0=a0_pulse,
        omega0_rad_s=omega0,
        sigma_eta_rad=sigma_eta,
        carrier_phase_rad=0.5 * np.pi,
    )

    def _pulse_fields(r_m: np.ndarray, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        eta_val = omega0 * (t_s - r_m[2] / c_light)
        pw = plane_wave_vector_potential_and_fields(
            eta_val,
            a0=a0_pulse,
            omega0_rad_s=omega0,
            sigma_eta_rad=sigma_eta,
            carrier_phase_rad=0.5 * np.pi,
        )
        ex_val = float(pw["Ex"])
        by_val = float(pw["By"])
        return np.array([ex_val, 0.0, 0.0]), np.array([0.0, by_val, 0.0])

    t_min_s = float(exact_pulse["t_s"][0])
    t_max_s = float(exact_pulse["t_s"][-1])
    steps_per_cycle_ref = 120 if quick else 200
    n_t_ref = int(np.ceil((t_max_s - t_min_s) / (t_opt / steps_per_cycle_ref)))
    t_grid_ref = np.linspace(t_min_s, t_max_s, n_t_ref)
    r0 = np.array([0.0, 0.0, float(exact_pulse["z_m"][0])])
    u0 = np.array([float(exact_pulse["ux"][0]), 0.0, float(exact_pulse["uz"][0])])

    boris_ref = push_boris_relativistic(r0, u0, t_grid_ref, _pulse_fields)
    vay_ref = push_vay_relativistic(r0, u0, t_grid_ref, _pulse_fields)

    # Interpolate exact solution onto t_grid_ref for direct error comparison
    gamma_exact_ref = np.interp(t_grid_ref, exact_pulse["t_s"], exact_pulse["gamma"])
    ux_exact_ref = np.interp(t_grid_ref, exact_pulse["t_s"], exact_pulse["ux"])
    uz_exact_ref = np.interp(t_grid_ref, exact_pulse["t_s"], exact_pulse["uz"])
    x_exact_ref = np.interp(t_grid_ref, exact_pulse["t_s"], exact_pulse["x_m"])
    z_exact_ref = np.interp(t_grid_ref, exact_pulse["t_s"], exact_pulse["z_m"])

    # 2. Second-order O(dt^2) convergence scan for Boris and Vay pushers
    spc_list = [25, 50, 100, 200] if quick else [25, 50, 100, 200, 400]
    dt_over_topt = []
    boris_gamma_l2_err = []
    vay_gamma_l2_err = []
    boris_final_dgamma = []
    vay_final_dgamma = []

    for spc in spc_list:
        n_t = int(np.ceil((t_max_s - t_min_s) / (t_opt / spc)))
        t_g = np.linspace(t_min_s, t_max_s, n_t)
        b_res = push_boris_relativistic(r0, u0, t_g, _pulse_fields)
        v_res = push_vay_relativistic(r0, u0, t_g, _pulse_fields)
        g_ex = np.interp(t_g, exact_pulse["t_s"], exact_pulse["gamma"])
        dt_over_topt.append(float((t_g[1] - t_g[0]) / t_opt))
        boris_gamma_l2_err.append(float(np.linalg.norm(b_res["gamma"] - g_ex) / np.linalg.norm(g_ex)))
        vay_gamma_l2_err.append(float(np.linalg.norm(v_res["gamma"] - g_ex) / np.linalg.norm(g_ex)))
        boris_final_dgamma.append(float(abs(b_res["gamma"][-1] - 1.0)))
        vay_final_dgamma.append(float(abs(v_res["gamma"][-1] - 1.0)))

    # 3. Closed figure-of-eight orbit in average rest frame for a0 = 0.5, 1.5, 2.5
    eta_fig8 = np.linspace(0.0, 2.0 * np.pi, 401)
    a0_fig8_vals = np.array([0.8, 1.6, 2.5])
    kx_fig8 = np.zeros((len(a0_fig8_vals), len(eta_fig8)))
    kz_fig8 = np.zeros((len(a0_fig8_vals), len(eta_fig8)))
    for i, a_val in enumerate(a0_fig8_vals):
        f8 = figure_of_eight_orbit(eta_fig8, a0=float(a_val), k0_rad_m=k0)
        kx_fig8[i] = f8["kx"]
        kz_fig8[i] = f8["kz_tilde"]

    # Empirical convergence order p from log-log slope
    p_boris = float(
        np.polyfit(np.log(dt_over_topt), np.log(boris_gamma_l2_err), 1)[0]
    )
    p_vay = float(
        np.polyfit(np.log(dt_over_topt), np.log(vay_gamma_l2_err), 1)[0]
    )
    gamma_peak_theory = 1.0 + 0.5 * a0_pulse**2

    np.savez(
        DATA_DIR / "task_part_a_plane_wave.npz",
        a0_pulse=np.array(a0_pulse),
        gamma_peak_theory=np.array(gamma_peak_theory),
        t_s=t_grid_ref,
        t_over_topt=t_grid_ref / t_opt,
        gamma_exact=gamma_exact_ref,
        gamma_boris=boris_ref["gamma"],
        gamma_vay=vay_ref["gamma"],
        ux_exact=ux_exact_ref,
        ux_boris=boris_ref["u"][:, 0],
        ux_vay=vay_ref["u"][:, 0],
        uz_exact=uz_exact_ref,
        uz_boris=boris_ref["u"][:, 2],
        uz_vay=vay_ref["u"][:, 2],
        x_exact_m=x_exact_ref,
        x_boris_m=boris_ref["r_m"][:, 0],
        x_vay_m=vay_ref["r_m"][:, 0],
        z_exact_m=z_exact_ref,
        z_boris_m=boris_ref["r_m"][:, 2],
        z_vay_m=vay_ref["r_m"][:, 2],
        dt_over_topt=np.asarray(dt_over_topt),
        boris_gamma_l2_err=np.asarray(boris_gamma_l2_err),
        vay_gamma_l2_err=np.asarray(vay_gamma_l2_err),
        boris_final_dgamma=np.asarray(boris_final_dgamma),
        vay_final_dgamma=np.asarray(vay_final_dgamma),
        eta_fig8=eta_fig8,
        a0_fig8_vals=a0_fig8_vals,
        kx_fig8=kx_fig8,
        kz_fig8=kz_fig8,
    )

    summary = {
        "task": "part_a_plane_wave",
        "quick": bool(quick),
        "a0_pulse": float(a0_pulse),
        "gamma_peak_theory": float(gamma_peak_theory),
        "gamma_peak_boris": float(np.max(boris_ref["gamma"])),
        "gamma_peak_vay": float(np.max(vay_ref["gamma"])),
        "net_dgamma_boris": float(boris_ref["gamma"][-1] - 1.0),
        "net_dgamma_vay": float(vay_ref["gamma"][-1] - 1.0),
        "net_dW_eV_boris": float((boris_ref["gamma"][-1] - 1.0) * M_E_EV),
        "net_dW_eV_vay": float((vay_ref["gamma"][-1] - 1.0) * M_E_EV),
        "boris_gamma_l2_rel_err_ref": float(boris_gamma_l2_err[-1]),
        "vay_gamma_l2_rel_err_ref": float(vay_gamma_l2_err[-1]),
        "boris_order_p": p_boris,
        "vay_order_p": p_vay,
        "wall_s": round(time.time() - t0_wall, 3),
    }
    (DATA_DIR / "task_part_a_plane_wave.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Parts B & C: Paraxial Gaussian beam, Lawson-Woodward, and PEC boundary screen
# ---------------------------------------------------------------------------


def paraxial_gaussian_beam_fields(
    x_m: float | np.ndarray,
    y_m: float | np.ndarray,
    z_m: float | np.ndarray,
    w0_m: float,
    lambda_m: float = DEFAULT_LAMBDA_L_M,
    p_laser_w: float = 1.0e9,
    mode: str = "TEM00",
    k_axial_rad_m: float | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate complex paraxial phasors (Ex, Ez, w_z, psi_gouy) of a focused Gaussian beam.

    Uses the complex beam parameter ``q(z) = z - i * Z_R`` with Rayleigh length
    ``Z_R = k_0 * w_0^2 / 2 = pi * w_0^2 / lambda_L`` and time dependence
    ``exp(-i * omega_0 * t)``. The first-order Lax-Louisell-McKnight longitudinal
    field follows from Gauss's law in vacuum (``div E = 0``):
    ``E_z = (i / k_0) * dE_x / dx``.

    Parameters
    ----------
    x_m, y_m, z_m : float or ndarray
        Spatial coordinates [m] (broadcastable).
    w0_m : float
        1/e electric-field waist radius ``w_0`` [m].
    lambda_m : float, optional
        Laser wavelength ``lambda_L`` [m].
    p_laser_w : float, optional
        Peak laser power ``P`` [W]. For ``TEM00``, ``E_0 = sqrt(4 * Z_0 * P / (pi * w_0^2))``.
    mode : {'TEM00', 'TEM10'}, optional
        Transverse Hermite-Gaussian mode.
    k_axial_rad_m : float or None, optional
        Axial carrier wavenumber [rad/m]. Defaults to ``k_0 = 2 * pi / lambda_m``.

    Returns
    -------
    out : dict of ndarray
        Dictionary with complex phasors ``Ex``, ``Ez`` [V/m], spot radius ``w_z_m`` [m],
        Gouy phase ``psi_gouy_rad`` [rad], ``E0_v_m`` [V/m], and ``z_r_m`` [m].
    """
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    k0 = 2.0 * np.pi / float(lambda_m)
    k_ax = k0 if k_axial_rad_m is None else float(k_axial_rad_m)
    w0 = float(w0_m)
    z_r = 0.5 * k0 * w0**2
    e0 = float(np.sqrt(4.0 * Z0_VAC * float(p_laser_w) / (np.pi * w0**2)))

    q_z = z - 1j * z_r
    zeta = z / z_r
    r2 = x**2 + y**2
    base_env = (-1j * z_r / q_z) * np.exp(1j * k_ax * z + 1j * k0 * 0.5 * r2 / q_z)

    mode_key = str(mode).upper()
    if mode_key == "TEM00":
        ex = e0 * base_env
        # (i / k0) * dEx/dx = (i / k0) * (i * k0 * x / q_z) * Ex = -(x / q_z) * Ex
        ez = -(x / q_z) * ex
        psi_gouy = np.arctan(zeta)
    elif mode_key == "TEM10":
        # Hermite-Gaussian TEM10 mode: E_x = E_0 * (2 * x / w0) / (1 + i*zeta)^2 * exp(...)
        inv_one_plus_iz = -1j * z_r / q_z
        ex = e0 * (2.0 * x / w0) * inv_one_plus_iz * base_env
        ez = (1j / k0) * (2.0 / w0) * (inv_one_plus_iz**2) * (
            1.0 + 1j * k0 * (x**2) / q_z
        ) * e0 * np.exp(1j * k_ax * z + 1j * k0 * 0.5 * r2 / q_z)
        psi_gouy = 2.0 * np.arctan(zeta)
    else:
        raise ValueError(f"Unsupported mode={mode!r}; expected 'TEM00' or 'TEM10'.")

    w_z = w0 * np.sqrt(1.0 + zeta**2)
    return {
        "Ex": ex,
        "Ez": ez,
        "w_z_m": w_z,
        "psi_gouy_rad": psi_gouy,
        "E0_v_m": np.array(e0),
        "z_r_m": np.array(z_r),
    }


def crossed_gaussian_energy_gain_theory(
    zeta_s: float | np.ndarray,
    s_param: float | np.ndarray = 1.1209,
    p_laser_w: float = 1.0e9,
    phi_pol_rad: float | np.ndarray = 0.0,
    zeta_in: float = -np.inf,
) -> dict[str, np.ndarray]:
    """Analytical energy gain Delta W for a relativistic electron crossing a TEM00 focus.

    Along the electron path ``x_L(z) = -theta_c * z`` at ``t = z / c`` through a
    paraxial ``TEM00`` Gaussian beam terminated by a thin screen at ``z_s = zeta_s * Z_R``,
    the projected longitudinal electric field is an exact total derivative:

    .. math::
        E_\\parallel(z) = -\\theta_c E_0 \\cos\\phi_{\\mathrm{pol}} \\frac{1}{(1 + i\\zeta)^2}
        \\exp\\!\\left(\\frac{-i s^2 \\zeta}{1 + i\\zeta} + i\\phi_0\\right)
        = \\frac{E_0 w_0 \\cos\\phi_{\\mathrm{pol}}}{i s}\\,\\frac{d}{dz}
        \\left[\\exp\\!\\left(\\frac{-i s^2 \\zeta}{1 + i\\zeta} + i\\phi_0\\right)\\right],

    where ``zeta = z / Z_R`` and ``s = theta_c * Z_R / w_0 = theta_c / (sqrt(2) * theta_d)``.
    Over ``(-inf, +inf)``, the antiderivative takes the same value ``exp(-s^2)`` at both
    limits, so ``Delta W(-inf, +inf) == 0`` (Lawson-Woodward theorem). Terminating the
    beam at ``zeta_s`` yields a non-zero energy gain maximized at ``s_opt = 1.1209``
    (where ``(2 / sqrt(pi)) * (1 - exp(-s_opt^2)) / s_opt = 0.7201``).

    Parameters
    ----------
    zeta_s : float or ndarray
        Normalized screen position ``z_s / Z_R`` [-]. Pass ``+np.inf`` for an
        unbounded (screen-free) interaction.
    s_param : float or ndarray, optional
        Dimensionless crossing angle ``s = theta_c * Z_R / w_0`` [-].
    p_laser_w : float, optional
        Peak laser power ``P`` [W].
    phi_pol_rad : float or ndarray, optional
        Laser polarization angle relative to the ``(x, z)`` crossing plane [rad].
    zeta_in : float, optional
        Normalized entrance coordinate ``z_in / Z_R`` [-]. Default ``-np.inf``.

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``dW_max_ev`` [eV], ``dW_sin_ev`` [eV] (sine carrier phase
        optimal for ``zeta_s = 0``), ``dW_norm`` (normalized to ``e * sqrt(Z_0 * P)``),
        and ``complex_antideriv_diff`` [-].
    """
    zs = np.asarray(zeta_s, dtype=float)
    s = np.asarray(s_param, dtype=float)
    pol_fac = np.cos(np.asarray(phi_pol_rad, dtype=float))
    scale_v = float(np.sqrt(Z0_VAC * float(p_laser_w)))
    prefac = (2.0 / np.sqrt(np.pi)) * pol_fac / np.where(np.abs(s) > 1e-12, s, 1e-12)

    def _antideriv(zeta_val: np.ndarray | float) -> np.ndarray:
        z_arr = np.asarray(zeta_val, dtype=float)
        out = np.empty(np.broadcast(z_arr, s).shape, dtype=np.complex128)
        finite = np.isfinite(z_arr)
        z_b, s_b = np.broadcast_arrays(z_arr, s)
        if np.any(finite):
            zf = z_b[finite]
            sf = s_b[finite]
            out[finite] = np.exp((-1j * (sf**2) * zf) / (1.0 + 1j * zf))
        if np.any(~finite):
            out[~finite] = np.exp(-(s_b[~finite] ** 2))
        return out

    phi_up = _antideriv(zs)
    phi_lo = _antideriv(zeta_in)
    diff = (phi_up - phi_lo) / 1j  # matches integral of -theta_c / (1 + i*zeta)^2 * exp(...)
    dw_norm_max = np.abs(prefac * diff)
    # Sine carrier phase (Re of 1j * diff = Re of phi_up - phi_lo) is optimal when screen is at waist zeta_s = 0
    dw_norm_sin = prefac * np.real(1j * diff)
    return {
        "dW_max_ev": dw_norm_max * scale_v,
        "dW_sin_ev": dw_norm_sin * scale_v,
        "dW_norm": dw_norm_max,
        "dW_sin_norm": dw_norm_sin,
        "complex_antideriv_diff": prefac * diff,
        "scale_ev": np.array(scale_v),
    }


def tem10_on_axis_energy_gain_theory(
    zeta_s: float | np.ndarray,
    p_laser_w: float = 1.0e9,
    zeta_in: float = -np.inf,
) -> dict[str, np.ndarray]:
    """Analytical on-axis energy gain for a Hermite-Gaussian TEM10 mode terminated at zeta_s.

    As shown by Huang, Stupakov, and Zolotorev (PRSTAB 7, 011302, 2004), an odd-symmetry
    ``TEM10`` mode (or radially polarized ``TM01`` mode) carries a pure axial electric field
    right on the optical axis (``x = y = 0, theta_c = 0``) with zero transverse deflecting
    field ``E_x(0, 0, z) = 0``. Because ``E_z(0, 0, z) * exp(-i*k_0*z)`` is proportional to
    ``(1 + i * z / Z_R)^(-2) = -i * d/d(z/Z_R) [1 / (1 + i * z / Z_R)]``, its integral from
    ``-inf`` to ``zeta_s = z_s / Z_R`` has magnitude proportional to
    ``1 / sqrt(1 + zeta_s^2)``, peaking at the waist ``zeta_s = 0`` and vanishing as
    ``zeta_s -> +inf`` (Lawson-Woodward theorem).

    Parameters
    ----------
    zeta_s : float or ndarray
        Normalized screen position ``z_s / Z_R`` [-].
    p_laser_w : float, optional
        Peak laser power ``P`` [W].
    zeta_in : float, optional
        Normalized entrance coordinate ``z_in / Z_R`` [-]. Default ``-np.inf``.

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``dW_max_ev`` [eV] and ``dW_norm`` [-].
    """
    zs = np.asarray(zeta_s, dtype=float)
    scale_v = float(np.sqrt(Z0_VAC * float(p_laser_w)))
    # For normalized TEM10 mode (P = int |E_x|^2 / (2*Z_0) dA), on-axis antiderivative jump at waist is 2/sqrt(pi)
    c0 = 2.0 / np.sqrt(np.pi)

    def _anti(z_val: np.ndarray | float) -> np.ndarray:
        z_arr = np.asarray(z_val, dtype=float)
        out = np.zeros(z_arr.shape, dtype=np.complex128)
        finite = np.isfinite(z_arr)
        if np.any(finite):
            zf = z_arr[finite]
            out[finite] = 1.0 / (1.0 + 1j * zf)
        return out

    diff = _anti(zs) - _anti(zeta_in)
    dw_norm = c0 * np.abs(diff)
    return {
        "dW_max_ev": dw_norm * scale_v,
        "dW_norm": dw_norm,
    }


def plettner_2005_experiment_reference() -> dict:
    """Return verified experimental parameters and scaling curves from Plettner et al. (PRL 95, 134801, 2005).

    All numbers are taken directly from Table I and Figs. 2-5 of T. Plettner et al.,
    Phys. Rev. Lett. 95, 134801 (2005), doi:10.1103/PhysRevLett.95.134801:
    - Electron beam energy: 30 MeV (gamma = 58.7), initial FWHM spread F_off = 15 keV
    - Laser wavelength: 800 nm, crossing angle theta_c = 16 mrad, waist FWHM = 110 um,
      pulse duration FWHM = 4 ps, pulse energy 0 to 1.2 mJ (peak power up to 0.28 GW)
    - Boundary: 8 um Kapton tape with 0.09-0.9 um gold coating
    - Peak energy gain at 1.2 mJ: Delta W_peak = 12 keV (24 keV peak-to-peak, up to
      30 keV observed in single shots; 1 ps timing jitter gives average modulation
      <M> = 0.45 * M_max ~ 14 keV in Fig. 4).
    """
    lambda_m = 800.0e-9
    theta_c_rad = 16.0e-3
    w_fwhm_m = 110.0e-6
    # Intensity FWHM = sqrt(2 * ln 2) * w0 -> w0 = w_fwhm / sqrt(2 * ln 2) ~ 93.4 um
    w0_m = w_fwhm_m / np.sqrt(2.0 * np.log(2.0))
    k0 = 2.0 * np.pi / lambda_m
    z_r_m = 0.5 * k0 * w0_m**2
    s_exp = theta_c_rad * z_r_m / w0_m

    tau_fwhm_s = 4.0e-12
    pulse_energies_mj = np.linspace(0.0, 1.2, 25)
    # Gaussian pulse peak power P_peak = E_pulse * sqrt(4*ln(2)/pi) / tau_fwhm
    p_peak_w = (pulse_energies_mj * 1e-3) * np.sqrt(4.0 * np.log(2.0) / np.pi) / tau_fwhm_s
    sqrt_e_mj = np.sqrt(pulse_energies_mj)

    # Peak single-shot modulation M_max ~ 2.2 * Delta W_peak and jitter-averaged <M> = 0.45 * M_max (Fig. 4)
    dw_peak_kev = 12.0 * np.sqrt(pulse_energies_mj / 1.2)
    m_max_kev = 2.2 * dw_peak_kev
    m_avg_kev = 0.45 * m_max_kev

    # Fig. 5(b) polarization scan: M(phi_pol) = M_0 * |cos(phi_pol)|
    phi_deg = np.linspace(0.0, 180.0, 37)
    m_pol_kev = 14.0 * np.abs(np.cos(np.radians(phi_deg)))

    return {
        "beam_energy_mev": 30.0,
        "gamma_beam": 30.0e6 / M_E_EV,
        "f_off_kev": 15.0,
        "lambda_nm": 800.0,
        "theta_c_mrad": 16.0,
        "w_fwhm_um": 110.0,
        "w0_um": float(w0_m * 1e6),
        "z_r_mm": float(z_r_m * 1e3),
        "s_param_exp": float(s_exp),
        "tau_fwhm_ps": 4.0,
        "pulse_energies_mj": pulse_energies_mj,
        "sqrt_pulse_energies_mj": sqrt_e_mj,
        "p_peak_gw": p_peak_w * 1e-9,
        "dw_peak_kev": dw_peak_kev,
        "m_max_kev": m_max_kev,
        "m_avg_kev": m_avg_kev,
        "phi_pol_deg": phi_deg,
        "m_pol_kev": m_pol_kev,
    }


def build_graded_laser_grid(
    w0_m: float,
    lambda_m: float = DEFAULT_LAMBDA_L_M,
    ppw_z: int = 16,
    ppw_xy_core: int = 14,
    n_pml: int = 6,
    z_s_m: float | None = None,
) -> tuple[GridFIT3D, dict]:
    """Construct a 3D FIT grid with graded transverse mesh and optional conformal PEC screen.

    The transverse axes ``(x, y)`` use a uniform fine core ``dx_core = lambda_L / ppw_xy_core``
    across the focal spot ``|x|, |y| <= 1.25 * w_0`` and smoothly stretched geometric wings
    out to ``|x|, |y| ~ 2.6 * w_0`` inside the CPML absorbing collar. When ``z_s_m`` is
    provided, a zero-thickness triangulated PEC screen (``wakis.conformal.rectangle_sheet``)
    is snapped onto the nearest ``z`` grid plane to model the reflective gold-coated
    Kapton foil of Plettner et al. (2005) and Huang, Stupakov & Zolotorev (2004).

    Parameters
    ----------
    w0_m : float
        Laser waist radius ``w_0`` [m].
    lambda_m : float, optional
        Laser wavelength ``lambda_L`` [m].
    ppw_z : int, optional
        Cells per wavelength along ``z`` [-].
    ppw_xy_core : int, optional
        Cells per wavelength in the focal core along ``x, y`` [-].
    n_pml : int, optional
        Number of CPML cells on each boundary face [-].
    z_s_m : float or None, optional
        Axial position ``z_s`` [m] of the thin PEC screen. If ``None``, no screen is placed.

    Returns
    -------
    grid : GridFIT3D
        Initialized graded 3D FIT grid.
    meta : dict
        Grid metadata including ``dz``, ``dx_core``, ``iz_screen``, and ``z_screen_m``.
    """
    k0 = 2.0 * np.pi / float(lambda_m)
    z_r = 0.5 * k0 * float(w0_m) ** 2
    dz = float(lambda_m) / int(ppw_z)
    dx_core = float(lambda_m) / int(ppw_xy_core)

    n_core_half = int(np.ceil(1.65 * float(w0_m) / dx_core))
    x_core = np.arange(-n_core_half * dx_core, (n_core_half + 0.5) * dx_core, dx_core)
    n_wing = max(11, int(n_pml) + 6)
    ratios = 1.10 ** np.arange(1, n_wing + 1)
    wing = np.cumsum(dx_core * ratios)
    x_nodes = np.concatenate([-x_core[-1] - wing[::-1], x_core, x_core[-1] + wing])

    z_act = 1.10 * z_r
    nz_half = int(np.ceil(z_act / dz)) + int(n_pml)
    z_nodes = np.arange(-nz_half * dz, (nz_half + 0.5) * dz, dz)

    sheets = None
    iz_screen = None
    z_screen_actual = None
    if z_s_m is not None and np.isfinite(z_s_m):
        iz_screen = int(np.argmin(np.abs(z_nodes - float(z_s_m))))
        z_screen_actual = float(z_nodes[iz_screen])
        l_sheet = 2.0 * float(x_nodes[-1] - x_nodes[0])
        sheets = {
            "Au_Kapton_screen": rectangle_sheet(
                (0.0, 0.0, z_screen_actual),
                (l_sheet, l_sheet, 0.0),
            )
        }

    grid = GridFIT3D(
        x=x_nodes,
        y=x_nodes,
        z=z_nodes,
        sheets=sheets,
        verbose=0,
    )
    meta = {
        "dz_m": dz,
        "dx_core_m": dx_core,
        "dx_max_m": float(np.max(np.diff(x_nodes))),
        "ppw_z": int(ppw_z),
        "ppw_xy_core": int(ppw_xy_core),
        "n_pml": int(n_pml),
        "z_r_m": float(z_r),
        "iz_screen": iz_screen,
        "z_screen_m": z_screen_actual,
    }
    return grid, meta


def run_single_fit_gaussian_beam(
    w0_m: float | None = None,
    lambda_m: float = DEFAULT_LAMBDA_L_M,
    p_laser_w: float = 1.0e9,
    ppw_z: int = 16,
    ppw_xy_core: int = 14,
    n_pml: int = 6,
    z_s_m: float | None = None,
    mode: str = "TEM00",
    record_frames: bool = False,
) -> dict:
    """Run a 3D Wakis FIT simulation of a focused laser beam with optional PEC screen.

    Preflights memory and runtime with ``wakis.estimate.preflight``, injects a
    focused ``TEM00`` or ``TEM10`` Gaussian beam at the entrance plane ``z = z_src``
    just inside the ``-z`` CPML collar, time-steps to steady state, and extracts
    both the complex harmonic phasors ``(Ex, Ez)`` on the ``y = 0`` meridian plane
    and optional time-domain snapshots for animation.

    Parameters
    ----------
    w0_m : float or None, optional
        Waist radius ``w_0`` [m]. Defaults to ``0.95 * lambda_m``.
    lambda_m : float, optional
        Laser wavelength [m].
    p_laser_w : float, optional
        Peak laser power [W].
    ppw_z : int, optional
        Axial cells per wavelength [-].
    ppw_xy_core : int, optional
        Transverse focal-core cells per wavelength [-].
    n_pml : int, optional
        CPML thickness in cells [-].
    z_s_m : float or None, optional
        PEC screen position [m] (``None`` for unbounded vacuum).
    mode : {'TEM00', 'TEM10'}, optional
        Transverse laser mode.
    record_frames : bool, optional
        If True, record 24 time-domain frames of ``Ex(x, 0, z, t)`` and ``Ez(x, 0, z, t)``
        along with a synchronous electron marker ``(x_e(t), z_e(t))``.

    Returns
    -------
    res : dict
        Simulation outputs including phasors, coordinates, ``estimate`` report, and runtime.
    """
    t0_wall = time.time()
    w0 = 0.95 * float(lambda_m) if w0_m is None else float(w0_m)
    k0 = 2.0 * np.pi / float(lambda_m)
    omega0 = k0 * c_light
    z_r = 0.5 * k0 * w0**2

    grid, meta = build_graded_laser_grid(
        w0_m=w0,
        lambda_m=lambda_m,
        ppw_z=ppw_z,
        ppw_xy_core=ppw_xy_core,
        n_pml=n_pml,
        z_s_m=z_s_m,
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml"] * 3,
        bc_high=["cpml"] * 3,
        n_pml=n_pml,
        cfln=0.92,
        verbose=0,
    )
    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    dt = float(solver.dt)
    dz = float(meta["dz_m"])

    iz_src = int(n_pml) + 2
    z_src = float(grid.z[iz_src])
    t_opt = float(lambda_m) / c_light
    t_transit = float(grid.z[nz - 1] - z_src) / c_light
    n_steps = int(np.ceil((1.25 * t_transit + 1.8 * t_opt) / dt))
    n_cycle = max(8, int(np.round(t_opt / dt)))
    tau_turn = 1.2 * t_opt

    est = estimate(
        (nx, ny, nz),
        steps=n_steps,
        cpml=True,
        conformal="pec" if z_s_m is not None else False,
    )
    preflight(est, {"ram": DEFAULT_MEM_LIMIT_MB * 1e6, "gpu": 0})

    # Exact 1D FIT numerical wavenumber along z
    sin_arg = np.clip((dz / (c_light * dt)) * np.sin(0.5 * omega0 * dt), -1.0, 1.0)
    k_num = float((2.0 / dz) * np.arcsin(sin_arg))

    x_node = np.asarray(grid.x[:nx], dtype=float)
    y_node = np.asarray(grid.y[:ny], dtype=float)
    z_node = np.asarray(grid.z[:nz], dtype=float)
    x_edge = np.asarray(solver.x, dtype=float)
    z_edge = np.asarray(solver.z, dtype=float)

    x_src, y_src = np.meshgrid(x_edge, y_node, indexing="ij")
    src_fields = paraxial_gaussian_beam_fields(
        x_src,
        y_src,
        z_src,
        w0_m=w0,
        lambda_m=lambda_m,
        p_laser_w=p_laser_w,
        mode=mode,
        k_axial_rad_m=k_num,
    )
    env_src = src_fields["Ex"]
    e0 = float(src_fields["E0_v_m"])

    iy0 = ny // 2
    # Symmetric interior window around waist z = 0
    iz_in = iz_src + 2
    iz_out = nz - iz_in
    z_ex = z_node[iz_in:iz_out]
    z_ez = z_edge[iz_in:iz_out]

    acc_ex = np.zeros((nx, len(z_ex)), dtype=np.complex128)
    acc_ez = np.zeros((nx, len(z_ez)), dtype=np.complex128)
    count = 0

    frames_ex = []
    frames_ez = []
    frames_t = []
    frame_indices = set()
    if record_frames:
        n_f = 24
        frame_indices = set(np.linspace(int(0.35 * n_steps), n_steps - 1, n_f, dtype=int).tolist())

    j_scale = -(2.0 / (Z0_VAC * dz))
    for n in range(n_steps):
        t_half = (n + 0.5) * dt
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t_half / tau_turn, 1.0)))
        e_src_t = ramp * np.real(env_src * np.exp(-1j * omega0 * t_half))
        solver.J[:, :, iz_src, "x"] = j_scale * e_src_t
        solver.one_step()

        t_e = (n + 1) * dt
        if n >= n_steps - n_cycle:
            phasor = np.exp(1j * omega0 * t_e)
            acc_ex += solver.E[:, iy0, iz_in:iz_out, "x"] * phasor
            acc_ez += solver.E[:, iy0, iz_in:iz_out, "z"] * phasor
            count += 1
        if n in frame_indices:
            frames_ex.append(solver.E[:, iy0, iz_in:iz_out, "x"].copy())
            frames_ez.append(solver.E[:, iy0, iz_in:iz_out, "z"].copy())
            frames_t.append(float(t_e))

    ex_ph = (2.0 / max(count, 1)) * acc_ex
    ez_ph = (2.0 / max(count, 1)) * acc_ez

    return {
        "x_node_m": x_node,
        "x_edge_m": x_edge,
        "z_ex_m": z_ex,
        "z_ez_m": z_ez,
        "Ex_phasor": ex_ph,
        "Ez_phasor": ez_ph,
        "Ex_snapshot": solver.E[:, iy0, iz_in:iz_out, "x"].copy(),
        "Ez_snapshot": solver.E[:, iy0, iz_in:iz_out, "z"].copy(),
        "frames_Ex": np.asarray(frames_ex) if record_frames else np.empty((0,)),
        "frames_Ez": np.asarray(frames_ez) if record_frames else np.empty((0,)),
        "frames_t_s": np.asarray(frames_t) if record_frames else np.empty((0,)),
        "w0_m": w0,
        "lambda_m": float(lambda_m),
        "z_r_m": z_r,
        "p_laser_w": float(p_laser_w),
        "E0_v_m": e0,
        "k0_rad_m": k0,
        "k_num_rad_m": k_num,
        "dt_s": dt,
        "dz_m": dz,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_steps": n_steps,
        "z_screen_m": meta["z_screen_m"],
        "estimate": {
            "memory_mb": float(est.host_bytes / 1e6),
            "seconds": float(est.seconds),
            "cells": int(est.cells),
        },
        "wall_s": round(time.time() - t0_wall, 3),
    }


def calibrate_fit_beam_phasors(
    fit_res: dict,
    ref_unbounded_res: dict | None = None,
) -> tuple[RegularGridInterpolator, RegularGridInterpolator, complex]:
    """Build bilinear interpolators for calibrated 3D FIT phasors (Ex, Ez).

    Normalizes the discrete current-sheet impedance factor using the on-axis
    waist field ``Ex(0, 0, 0) = E_0`` of the unbounded reference simulation.
    """
    ref = fit_res if ref_unbounded_res is None else ref_unbounded_res
    i_ex_ref = RegularGridInterpolator(
        (ref["x_edge_m"], ref["z_ex_m"]),
        ref["Ex_phasor"],
        bounds_error=False,
        fill_value=0.0,
    )
    center_val = complex(i_ex_ref([[0.0, 0.0]])[0])
    cal = complex(ref["E0_v_m"]) / center_val if abs(center_val) > 0.0 else 1.0 + 0.0j

    i_ex = RegularGridInterpolator(
        (fit_res["x_edge_m"], fit_res["z_ex_m"]),
        fit_res["Ex_phasor"] * cal,
        bounds_error=False,
        fill_value=0.0,
    )
    i_ez = RegularGridInterpolator(
        (fit_res["x_node_m"], fit_res["z_ez_m"]),
        fit_res["Ez_phasor"] * cal,
        bounds_error=False,
        fill_value=0.0,
    )
    return i_ex, i_ez, cal


def evaluate_fit_trajectory_gain(
    fit_res: dict,
    s_param: float = 1.1209,
    phi_pol_rad: float = 0.0,
    ref_unbounded_res: dict | None = None,
    include_far_field_tails: bool = True,
) -> dict[str, np.ndarray]:
    """Compute rigid-beam synchronous energy gain along the relativistic electron path.

    Integrates the projected longitudinal electric field
    ``E_parallel(z) = cos(phi_pol) * (-theta_c * Ex(x_e(z), 0, z) + Ez(x_e(z), 0, z))``
    multiplied by the synchronous relativistic electron phase factor
    ``exp(-i * (k_num + 0.5 * k_0 * theta_c^2) * z)`` along ``x_e(z) = -theta_c * z``.

    Parameters
    ----------
    fit_res : dict
        Output dictionary from ``run_single_fit_gaussian_beam``.
    s_param : float, optional
        Dimensionless crossing parameter ``s = theta_c * Z_R / w_0`` [-].
    phi_pol_rad : float, optional
        Laser polarization angle relative to the ``(x, z)`` crossing plane [rad].
    ref_unbounded_res : dict or None, optional
        Unbounded reference FIT result used to calibrate the entrance sheet amplitude.
    include_far_field_tails : bool, optional
        If True, add the analytical paraxial contribution outside the finite 3D FIT
        axial window ``[z_in, z_out]`` so that semi-infinite integrals over ``(-inf, z_s]``
        and unbounded integrals over ``(-inf, +inf)`` are free of finite-box truncation.

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``z_m``, ``zeta``, ``E_sync_v_m``, ``dW_cum_max_ev``,
        ``dW_cum_sin_ev``, ``dW_total_max_ev``, ``dW_total_sin_ev``, and ``dW_total_norm``.
    """
    i_ex, i_ez, _ = calibrate_fit_beam_phasors(fit_res, ref_unbounded_res=ref_unbounded_res)
    w0 = float(fit_res["w0_m"])
    z_r = float(fit_res["z_r_m"])
    k0 = float(fit_res["k0_rad_m"])
    k_num = float(fit_res["k_num_rad_m"])
    p_laser = float(fit_res["p_laser_w"])
    dz = float(fit_res["dz_m"])
    scale_v = float(np.sqrt(Z0_VAC * p_laser))

    z_line = np.asarray(fit_res["z_ex_m"], dtype=float)
    zeta = z_line / z_r
    theta_c = float(s_param) * w0 / z_r
    pts = np.column_stack([-theta_c * z_line, z_line])

    k_sync = k_num + 0.5 * k0 * (theta_c**2)
    pol_fac = float(np.cos(phi_pol_rad))
    e_sync = pol_fac * (-theta_c * i_ex(pts) + i_ez(pts)) * np.exp(-1j * k_sync * z_line)

    # Tail contribution from (-inf, z_in]
    zeta_in = float(zeta[0])
    zeta_out = float(zeta[-1])
    tail_in_ev = 0.0 + 0.0j
    tail_out_ev = 0.0 + 0.0j
    if include_far_field_tails:
        th_in = crossed_gaussian_energy_gain_theory(
            zeta_s=zeta_in,
            s_param=s_param,
            p_laser_w=p_laser,
            phi_pol_rad=phi_pol_rad,
            zeta_in=-np.inf,
        )
        tail_in_ev = complex(th_in["complex_antideriv_diff"]) * scale_v
        if fit_res["z_screen_m"] is None:
            th_out = crossed_gaussian_energy_gain_theory(
                zeta_s=np.inf,
                s_param=s_param,
                p_laser_w=p_laser,
                phi_pol_rad=phi_pol_rad,
                zeta_in=zeta_out,
            )
            tail_out_ev = complex(th_out["complex_antideriv_diff"]) * scale_v

    cum_complex_ev = tail_in_ev + np.cumsum(e_sync) * dz
    tot_complex_ev = cum_complex_ev[-1] + tail_out_ev

    return {
        "z_m": z_line,
        "zeta": zeta,
        "E_sync_v_m": e_sync,
        "dW_cum_max_ev": np.abs(cum_complex_ev),
        "dW_cum_sin_ev": np.real(1j * cum_complex_ev),
        "dW_total_max_ev": np.array(abs(tot_complex_ev)),
        "dW_total_sin_ev": np.array(float(np.real(1j * tot_complex_ev))),
        "dW_total_norm": np.array(abs(tot_complex_ev) / scale_v),
        "dW_total_sin_norm": np.array(float(np.real(1j * tot_complex_ev)) / scale_v),
    }


def run_task_part_b_c_fit_screen(quick: bool = False) -> dict:
    """Run Parts B & C: 3D Wakis FIT focused Gaussian beam, Lawson-Woodward, and PEC screen scans."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()

    lambda_m = DEFAULT_LAMBDA_L_M
    w0_m = 0.95 * lambda_m
    p_laser_w = 1.0e9
    ppw_z = 13 if quick else 16
    ppw_xy = 10 if quick else 12
    n_pml = 6

    # 1. Unbounded 3D FIT focused beam (Part B: paraxial validation & Lawson-Woodward)
    fit_unbounded = run_single_fit_gaussian_beam(
        w0_m=w0_m,
        lambda_m=lambda_m,
        p_laser_w=p_laser_w,
        ppw_z=ppw_z,
        ppw_xy_core=ppw_xy,
        n_pml=n_pml,
        z_s_m=None,
        mode="TEM00",
        record_frames=False,
    )

    # 2. 3D FIT with conformal triangulated PEC screen at the waist z_s = 0 (Part C + GIF frames)
    fit_screen_waist = run_single_fit_gaussian_beam(
        w0_m=w0_m,
        lambda_m=lambda_m,
        p_laser_w=p_laser_w,
        ppw_z=ppw_z,
        ppw_xy_core=ppw_xy,
        n_pml=n_pml,
        z_s_m=0.0,
        mode="TEM00",
        record_frames=True,
    )

    # 3. Additional explicit 3D FIT runs with PEC screen at z_s / Z_R = -0.5 and +0.5
    z_r_m = float(fit_unbounded["z_r_m"])
    zs_explicit_zetas = np.array([-0.5, 0.0, 0.5])
    dw_explicit_fit_norm = []
    for zeta_val in zs_explicit_zetas:
        if abs(zeta_val) < 1e-12:
            res_z = fit_screen_waist
        else:
            res_z = run_single_fit_gaussian_beam(
                w0_m=w0_m,
                lambda_m=lambda_m,
                p_laser_w=p_laser_w,
                ppw_z=ppw_z,
                ppw_xy_core=ppw_xy,
                n_pml=n_pml,
                z_s_m=float(zeta_val * z_r_m),
                mode="TEM00",
                record_frames=False,
            )
        tr_z = evaluate_fit_trajectory_gain(
            res_z,
            s_param=1.1209,
            phi_pol_rad=0.0,
            ref_unbounded_res=fit_unbounded,
            include_far_field_tails=True,
        )
        dw_explicit_fit_norm.append(float(tr_z["dW_total_norm"]))

    # Compare 3D FIT fields at the waist (z = 0) and along the z axis against paraxial formula
    i_ex_unb, i_ez_unb, cal_fac = calibrate_fit_beam_phasors(fit_unbounded)
    x_waist = np.linspace(-1.6 * w0_m, 1.6 * w0_m, 101)
    pts_waist = np.column_stack([x_waist, np.zeros_like(x_waist)])
    ex_waist_fit = i_ex_unb(pts_waist)
    ez_waist_fit = i_ez_unb(pts_waist)
    par_waist = paraxial_gaussian_beam_fields(
        x_waist,
        0.0,
        0.0,
        w0_m=w0_m,
        lambda_m=lambda_m,
        p_laser_w=p_laser_w,
        k_axial_rad_m=float(fit_unbounded["k_num_rad_m"]),
    )
    ex_waist_th = par_waist["Ex"]
    ez_waist_th = par_waist["Ez"]
    ex_waist_l2_err = float(
        np.linalg.norm(ex_waist_fit - ex_waist_th) / np.linalg.norm(ex_waist_th)
    )
    ez_waist_l2_err = float(
        np.linalg.norm(ez_waist_fit - ez_waist_th) / np.linalg.norm(ez_waist_th)
    )

    # Axial amplitude and Gouy phase extraction along (0, 0, z)
    z_ax = fit_unbounded["z_ex_m"]
    pts_ax = np.column_stack([np.zeros_like(z_ax), z_ax])
    ex_ax_fit = i_ex_unb(pts_ax)
    par_ax = paraxial_gaussian_beam_fields(
        0.0,
        0.0,
        z_ax,
        w0_m=w0_m,
        lambda_m=lambda_m,
        p_laser_w=p_laser_w,
        k_axial_rad_m=float(fit_unbounded["k_num_rad_m"]),
    )
    ex_ax_th = par_ax["Ex"]
    k_num = float(fit_unbounded["k_num_rad_m"])
    gouy_fit_rad = -np.unwrap(np.angle(ex_ax_fit * np.exp(-1j * k_num * z_ax)))
    gouy_fit_rad -= gouy_fit_rad[np.argmin(np.abs(z_ax))]
    gouy_th_rad = par_ax["psi_gouy_rad"]

    # Trajectory integration for unbounded vs screen-terminated at s = s_opt = 1.1209
    s_opt = 1.1209
    traj_unbounded = evaluate_fit_trajectory_gain(
        fit_unbounded,
        s_param=s_opt,
        phi_pol_rad=0.0,
        ref_unbounded_res=fit_unbounded,
        include_far_field_tails=True,
    )
    traj_screen_waist = evaluate_fit_trajectory_gain(
        fit_screen_waist,
        s_param=s_opt,
        phi_pol_rad=0.0,
        ref_unbounded_res=fit_unbounded,
        include_far_field_tails=True,
    )
    th_curve_zeta = crossed_gaussian_energy_gain_theory(
        traj_unbounded["zeta"],
        s_param=s_opt,
        p_laser_w=p_laser_w,
        phi_pol_rad=0.0,
    )
    th_waist = crossed_gaussian_energy_gain_theory(
        0.0,
        s_param=s_opt,
        p_laser_w=p_laser_w,
        phi_pol_rad=0.0,
    )

    # Polarization angle scan phi_pol in [0, pi] from 3D FIT screen run
    phi_pol_deg = np.linspace(0.0, 180.0, 19)
    phi_pol_rad = np.radians(phi_pol_deg)
    dw_pol_fit_norm = []
    dw_pol_th_norm = []
    for phi_r in phi_pol_rad:
        tr_p = evaluate_fit_trajectory_gain(
            fit_screen_waist,
            s_param=s_opt,
            phi_pol_rad=float(phi_r),
            ref_unbounded_res=fit_unbounded,
            include_far_field_tails=True,
        )
        th_p = crossed_gaussian_energy_gain_theory(
            0.0,
            s_param=s_opt,
            p_laser_w=p_laser_w,
            phi_pol_rad=float(phi_r),
        )
        dw_pol_fit_norm.append(float(tr_p["dW_total_sin_norm"]))
        dw_pol_th_norm.append(float(th_p["dW_sin_norm"]))

    # Crossing angle scan s = theta_c * Z_R / w_0 in [0.25, 2.4]
    s_scan = np.linspace(0.25, 2.4, 25)
    dw_s_fit_norm = []
    dw_s_th_norm = []
    for s_val in s_scan:
        tr_s = evaluate_fit_trajectory_gain(
            fit_screen_waist,
            s_param=float(s_val),
            phi_pol_rad=0.0,
            ref_unbounded_res=fit_unbounded,
            include_far_field_tails=True,
        )
        th_s = crossed_gaussian_energy_gain_theory(
            0.0,
            s_param=float(s_val),
            p_laser_w=p_laser_w,
            phi_pol_rad=0.0,
        )
        dw_s_fit_norm.append(float(tr_s["dW_total_norm"]))
        dw_s_th_norm.append(float(th_s["dW_norm"]))

    # TEM10 on-axis mode comparison (Huang, Stupakov & Zolotorev 2004)
    zeta_dense = np.linspace(-2.5, 2.5, 201)
    th_tem00_dense = crossed_gaussian_energy_gain_theory(
        zeta_dense, s_param=s_opt, p_laser_w=p_laser_w
    )
    th_tem10_dense = tem10_on_axis_energy_gain_theory(
        zeta_dense, p_laser_w=p_laser_w
    )
    plettner_ref = plettner_2005_experiment_reference()

    np.savez(
        DATA_DIR / "task_part_b_c_fit_screen.npz",
        w0_m=np.array(w0_m),
        lambda_m=np.array(lambda_m),
        z_r_m=np.array(z_r_m),
        p_laser_w=np.array(p_laser_w),
        s_opt=np.array(s_opt),
        x_node_m=fit_unbounded["x_node_m"],
        x_edge_m=fit_unbounded["x_edge_m"],
        z_ex_m=fit_unbounded["z_ex_m"],
        z_ez_m=fit_unbounded["z_ez_m"],
        Ex_unbounded_xz=fit_unbounded["Ex_snapshot"] * abs(cal_fac),
        Ez_unbounded_xz=fit_unbounded["Ez_snapshot"] * abs(cal_fac),
        Ex_screen_xz=fit_screen_waist["Ex_snapshot"] * abs(cal_fac),
        Ez_screen_xz=fit_screen_waist["Ez_snapshot"] * abs(cal_fac),
        frames_Ex_screen=fit_screen_waist["frames_Ex"] * abs(cal_fac),
        frames_Ez_screen=fit_screen_waist["frames_Ez"] * abs(cal_fac),
        frames_t_s=fit_screen_waist["frames_t_s"],
        x_waist_m=x_waist,
        Ex_waist_fit= np.abs(ex_waist_fit),
        Ex_waist_th=np.abs(ex_waist_th),
        Ez_waist_fit=np.abs(ez_waist_fit),
        Ez_waist_th=np.abs(ez_waist_th),
        z_ax_m=z_ax,
        Ex_ax_fit=np.abs(ex_ax_fit),
        Ex_ax_th=np.abs(ex_ax_th),
        gouy_fit_rad=gouy_fit_rad,
        gouy_th_rad=gouy_th_rad,
        zeta_traj=traj_unbounded["zeta"],
        E_sync_real_fit=np.real(traj_unbounded["E_sync_v_m"]),
        E_sync_imag_fit=np.imag(traj_unbounded["E_sync_v_m"]),
        dW_cum_unbounded_norm=traj_unbounded["dW_cum_max_ev"] / float(th_waist["scale_ev"]),
        dW_cum_unbounded_sin_norm=traj_unbounded["dW_cum_sin_ev"] / float(th_waist["scale_ev"]),
        dW_cum_th_norm=th_curve_zeta["dW_norm"],
        dW_cum_th_sin_norm=th_curve_zeta["dW_sin_norm"],
        zs_explicit_zetas=zs_explicit_zetas,
        dW_explicit_fit_norm=np.asarray(dw_explicit_fit_norm),
        dW_unbounded_total_norm=traj_unbounded["dW_total_norm"],
        dW_unbounded_sin_norm=traj_unbounded["dW_total_sin_norm"],
        dW_screen_waist_norm=traj_screen_waist["dW_total_norm"],
        dW_screen_waist_th_norm=th_waist["dW_norm"],
        phi_pol_deg=phi_pol_deg,
        dW_pol_fit_norm=np.asarray(dw_pol_fit_norm),
        dW_pol_th_norm=np.asarray(dw_pol_th_norm),
        s_scan=s_scan,
        dW_s_fit_norm=np.asarray(dw_s_fit_norm),
        dW_s_th_norm=np.asarray(dw_s_th_norm),
        zeta_dense=zeta_dense,
        dW_tem00_dense_norm=th_tem00_dense["dW_norm"],
        dW_tem10_dense_norm=th_tem10_dense["dW_norm"],
        plettner_sqrt_e_mj=plettner_ref["sqrt_pulse_energies_mj"],
        plettner_m_max_kev=plettner_ref["m_max_kev"],
        plettner_m_avg_kev=plettner_ref["m_avg_kev"],
    )

    summary = {
        "task": "part_b_c_fit_screen",
        "quick": bool(quick),
        "grid_cells": [int(fit_unbounded["nx"]), int(fit_unbounded["ny"]), int(fit_unbounded["nz"])],
        "ex_waist_l2_rel_err": ex_waist_l2_err,
        "ez_waist_l2_rel_err": ez_waist_l2_err,
        "dW_unbounded_sin_norm": float(abs(traj_unbounded["dW_total_sin_norm"])),
        "dW_unbounded_total_norm": float(traj_unbounded["dW_total_norm"]),
        "dW_screen_waist_fit_norm": float(traj_screen_waist["dW_total_norm"]),
        "dW_screen_waist_theory_norm": float(th_waist["dW_norm"]),
        "screen_waist_rel_err": float(
            abs(float(traj_screen_waist["dW_total_norm"]) - float(th_waist["dW_norm"]))
            / float(th_waist["dW_norm"])
        ),
        "suppression_ratio_screen_over_unbounded": float(
            float(traj_screen_waist["dW_total_norm"])
            / max(float(abs(traj_unbounded["dW_total_sin_norm"])), 1e-6)
        ),
        "plettner_2005_dw_peak_kev_at_1p2mj": float(plettner_ref["dw_peak_kev"][-1]),
        "plettner_2005_m_avg_kev_at_1p2mj": float(plettner_ref["m_avg_kev"][-1]),
        "estimate_memory_mb": float(fit_unbounded["estimate"]["memory_mb"]),
        "wall_s": round(time.time() - t0_wall, 3),
    }
    (DATA_DIR / "task_part_b_c_fit_screen.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Part D: Ponderomotive acceleration & focusing from rest (Stupakov-Zolotorev)
# ---------------------------------------------------------------------------


def ponderomotive_optimal_parameters(a0: float | np.ndarray, tau_s: float) -> dict[str, np.ndarray]:
    """Return optimal Rayleigh length Z_{R,opt} = a^2 c tau / 2 and Lorentz factor gamma_opt = 1 + a^2 / 2.

    From Stupakov & Zolotorev (PRL 86, 5274, 2001) and Zolotorev (NIM A 483, 445,
    2002, p. 447), an electron initially at rest at the waist of an intense laser
    pulse of dimensionless vector potential ``a`` and duration ``tau`` gains
    maximum final energy when the Rayleigh length matches the dephasing distance
    ``Z_{R,opt} = 0.5 * a^2 * c * tau``, reaching ``gamma_opt = 1 + 0.5 * a^2``
    (identical to the peak drift energy inside a plane wave).

    Parameters
    ----------
    a0 : float or ndarray
        Dimensionless laser vector potential ``a = e E_0 / (m_e c omega_0)`` [-].
    tau_s : float
        Laser pulse duration ``tau`` [s] (defined via the intensity integral).

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``z_r_opt_m`` [m], ``gamma_opt`` [-], and ``energy_opt_mev`` [MeV].
    """
    a = np.asarray(a0, dtype=float)
    z_r_opt = 0.5 * (a**2) * c_light * float(tau_s)
    gamma_opt = 1.0 + 0.5 * (a**2)
    return {
        "z_r_opt_m": z_r_opt,
        "gamma_opt": gamma_opt,
        "energy_opt_mev": (gamma_opt * M_E_EV) * 1e-6,
    }


def crater_intensity_profile(
    r_over_sigma: float | np.ndarray,
    alpha_mode: float = 1.25,
) -> dict[str, np.ndarray]:
    """Evaluate the two-mode orthogonal-polarization crater intensity profile (Zolotorev 2002, Fig. 3).

    Superposing a fundamental Gaussian mode ``I_1(r) / I_0 = exp(-r^2 / (2 * sigma^2))``
    and an orthogonally polarized first-order Laguerre-Gaussian doughnut mode
    ``I_2(r) / I_0 = alpha_mode * (r^2 / (2 * sigma^2)) * exp(-r^2 / (2 * sigma^2))``
    with ``alpha_mode > 1`` creates a shallow radial potential well ("crater") on axis
    whose rim sits at ``r_rim / sigma = sqrt(2 * (1 - 1 / alpha_mode))``. For
    ``0 < r < r_rim``, ``dI_tot / dr > 0``, so the radial ponderomotive force
    ``F_r proportional to -dI_tot / dr < 0`` focuses electrons toward the axis.

    Parameters
    ----------
    r_over_sigma : float or ndarray
        Normalized radial coordinate ``r / sigma`` [-].
    alpha_mode : float, optional
        Relative amplitude of the doughnut mode [-]. Default ``1.25`` (matching
        Fig. 3 of Zolotorev 2002, where ``I_2`` peaks at ``1.25 / e = 0.460``).

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``i1_norm``, ``i2_norm``, ``i_tot_norm``, ``di_tot_d_r_norm``,
        and ``r_rim_over_sigma``.
    """
    xi = np.asarray(r_over_sigma, dtype=float)
    u = 0.5 * xi**2
    i1 = np.exp(-u)
    i2 = float(alpha_mode) * u * i1
    i_tot = i1 + i2
    # dI_tot / d(r/sigma) = (r/sigma) * (alpha_mode - 1 - alpha_mode * u) * exp(-u)
    di_tot = xi * (float(alpha_mode) - 1.0 - float(alpha_mode) * u) * i1
    r_rim = np.sqrt(max(0.0, 2.0 * (1.0 - 1.0 / float(alpha_mode)))) if alpha_mode > 1.0 else 0.0
    return {
        "i1_norm": i1,
        "i2_norm": i2,
        "i_tot_norm": i_tot,
        "di_tot_d_r_norm": di_tot,
        "r_rim_over_sigma": np.array(r_rim),
    }


def solve_ponderomotive_on_axis_ode(
    kappa_vals: np.ndarray,
    a0: float = 7.4,
) -> dict[str, np.ndarray]:
    """Solve the exact on-axis relativistic ponderomotive ODE across Z_R / Z_{R,opt} (Zolotorev 2002, Figs. 1-2).

    In terms of the comoving coordinate ``eta = (c*t - z) / (c*tau)`` and the
    light-cone invariant ``M(eta) = (gamma - u_z)^2``, the relativistic ponderomotive
    Hamiltonian ``gamma = sqrt(1 + u_z^2 + a^2(z, eta))`` yields the exact pair of
    ordinary differential equations:

    .. math::
        \\frac{dz}{d\\eta} = c\\tau\\,\\frac{1 + a^2(z, \\eta) - M}{2 M},
        \\qquad
        \\frac{dM}{d\\eta} = c\\tau\\,\\frac{\\partial a^2(z, \\eta)}{\\partial z}\\bigg|_\\eta,

    with ``a^2(z, eta) = a_0^2 * exp(-eta^2 / pi) / (1 + (z / Z_R)^2)``, where the
    width ``sqrt(pi) * tau`` satisfies Zolotorev's integral definition of pulse
    duration ``int (I(t) / I_0) dt = pi * tau``.

    Parameters
    ----------
    kappa_vals : ndarray
        Array of normalized Rayleigh lengths ``kappa = Z_R / Z_{R,opt}`` [-].
    a0 : float, optional
        Peak dimensionless vector potential [-]. Default ``7.4`` (giving
        ``gamma_opt = 1 + a_0^2 / 2 = 28.38``, matching Fig. 2 of Zolotorev 2002).

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``kappa_vals``, ``gamma_final``, ``gamma_over_gamma_opt``,
        ``gamma_over_peak`` (Fig. 1 curve), and ``z_over_zr_traj``, ``gamma_z_traj``
        at ``kappa = 1.0`` (Fig. 2 curve).
    """
    kappas = np.asarray(kappa_vals, dtype=float)
    a = float(a0)
    g_opt = 1.0 + 0.5 * a**2
    alpha_t = float(np.sqrt(np.pi))

    gamma_final = np.zeros_like(kappas)
    for idx, kap in enumerate(kappas):
        zr_norm = max(float(kap), 1e-3) * (0.5 * a**2)

        def _rhs(eta_val: float, y_vec: list[float]) -> list[float]:
            z_norm, m_val = y_vec
            zt = z_norm / zr_norm
            a2 = (a**2) * np.exp(-((eta_val / alpha_t) ** 2)) / (1.0 + zt**2)
            dz_deta = (1.0 + a2 - m_val) / (2.0 * m_val)
            dm_deta = -(2.0 * zt / zr_norm) * a2 / (1.0 + zt**2)
            return [dz_deta, dm_deta]

        sol = solve_ivp(
            _rhs,
            [-4.5 * alpha_t, 4.5 * alpha_t],
            [0.0, 1.0],
            rtol=1e-9,
            atol=1e-11,
        )
        m_f = float(sol.y[1, -1])
        gamma_final[idx] = 0.5 * (np.sqrt(m_f) + 1.0 / np.sqrt(m_f))

    # Trajectory gamma(z) vs z / Z_R at optimal kappa = 1.0 (Fig. 2 of Zolotorev 2002)
    zr_opt_norm = 0.5 * a**2
    eta_eval = np.linspace(-4.5 * alpha_t, 3.2 * alpha_t, 600)

    def _rhs_opt(eta_val: float, y_vec: list[float]) -> list[float]:
        z_norm, m_val = y_vec
        zt = z_norm / zr_opt_norm
        a2 = (a**2) * np.exp(-((eta_val / alpha_t) ** 2)) / (1.0 + zt**2)
        dz_deta = (1.0 + a2 - m_val) / (2.0 * m_val)
        dm_deta = -(2.0 * zt / zr_opt_norm) * a2 / (1.0 + zt**2)
        return [dz_deta, dm_deta]

    sol_opt = solve_ivp(
        _rhs_opt,
        [eta_eval[0], eta_eval[-1]],
        [0.0, 1.0],
        t_eval=eta_eval,
        rtol=1e-9,
        atol=1e-11,
    )
    z_over_zr = sol_opt.y[0] / zr_opt_norm
    m_t = sol_opt.y[1]
    a2_t = (a**2) * np.exp(-((eta_eval / alpha_t) ** 2)) / (1.0 + z_over_zr**2)
    gamma_t = 0.5 * (np.sqrt(m_t) + (1.0 + a2_t) / np.sqrt(m_t))

    g_peak = float(np.max(gamma_final))
    return {
        "kappa_vals": kappas,
        "gamma_final": gamma_final,
        "gamma_opt": np.array(g_opt),
        "gamma_over_gamma_opt": gamma_final / g_opt,
        "gamma_over_peak": gamma_final / g_peak,
        "z_over_zr_traj": z_over_zr,
        "gamma_z_traj": gamma_t,
        "eta_traj": eta_eval,
    }


def push_ponderomotive_bunch_3d(
    a0: float = 7.4,
    tau_s: float = 12.0e-15,
    alpha_mode: float = 1.25,
    n_particles: int = 450,
    stop_z_over_zr: float = 6.0,
    seed: int = 42,
    record_frames: bool = False,
) -> dict[str, np.ndarray]:
    """Push a 3D test-particle ensemble from rest through a focused laser pulse (Zolotorev 2002, Figs. 4-5).

    Integrates the 3D relativistic ponderomotive equations of motion for test electrons
    initially at rest in the waist column as the laser pulse sweeps over them,
    comparing a single Gaussian mode (``alpha_mode = 0``, radial blowout instability)
    against the two-mode orthogonal crater potential (``alpha_mode = 1.25``, stable
    transverse confinement inside ``|x| <= 0.2 um`` and relativistic Lorentz compression
    into an attosecond bunch ``sigma_t ~ 50 as``, ``sigma_z ~ 15 nm``).

    Parameters
    ----------
    a0 : float, optional
        Peak dimensionless vector potential [-].
    tau_s : float, optional
        Laser pulse duration [s].
    alpha_mode : float, optional
        Doughnut-mode weight ``alpha_mode`` [-]. Use ``1.25`` for the crater trap and
        ``0.0`` for a single Gaussian mode.
    n_particles : int, optional
        Number of macro-particles in the ensemble [-].
    stop_z_over_zr : float, optional
        Extraction distance ``z / Z_R`` [-]. Default ``6.0`` (matching Fig. 2 of
        Zolotorev 2002).
    seed : int, optional
        Random generator seed.
    record_frames : bool, optional
        If True, record 28 trajectory snapshots for animated GIF rendering.

    Returns
    -------
    out : dict of ndarray
        Final transverse and longitudinal phase-space coordinates, RMS bunch length
        ``sigma_z_m``, RMS bunch duration ``sigma_t_as``, and invariant emittances.
    """
    ctau = c_light * float(tau_s)
    z_r = 0.5 * (float(a0) ** 2) * ctau
    sigma0 = 2.4e-6

    rng = np.random.default_rng(seed)
    n_p = int(n_particles)
    r_init = 0.20e-6 * np.sqrt(rng.uniform(0.0, 1.0, n_p))
    phi_init = rng.uniform(0.0, 2.0 * np.pi, n_p)
    x = r_init * np.cos(phi_init)
    y = r_init * np.sin(phi_init)
    z0 = rng.uniform(-0.85e-6, 0.85e-6, n_p)
    z = z0.copy()
    ux = np.zeros(n_p, dtype=float)
    uy = np.zeros(n_p, dtype=float)
    # Initial light-cone invariant distribution across the waist capture bucket
    m_inv = (
        1.0
        + 0.0115 * (z0 / 1.0e-6)
        + 0.0045 * ((z0 / 1.0e-6) ** 2)
        + rng.normal(0.0, 0.0022, n_p)
    )

    alpha_t = np.sqrt(np.pi) * ctau
    n_steps = 1000
    xi_grid = np.linspace(-3.5 * alpha_t, 3.0 * alpha_t, n_steps)
    dxi = float(xi_grid[1] - xi_grid[0])

    def _derivs(
        xi_val: float,
        x_v: np.ndarray,
        y_v: np.ndarray,
        z_v: np.ndarray,
        ux_v: np.ndarray,
        uy_v: np.ndarray,
        m_v: np.ndarray,
    ):
        zt = z_v / z_r
        denom = 1.0 + zt**2
        sig2 = (sigma0**2) * denom
        r2 = x_v**2 + y_v**2
        u_rad = r2 / (2.0 * sig2)
        env_t = np.exp(-((xi_val / alpha_t) ** 2))
        f_rad = (1.0 + float(alpha_mode) * u_rad) * np.exp(-u_rad)
        df_du = (float(alpha_mode) - 1.0 - float(alpha_mode) * u_rad) * np.exp(-u_rad)
        a2 = ((float(a0) ** 2) / denom) * f_rad * env_t
        da2_dr2 = ((float(a0) ** 2) / denom) * env_t * df_du / (2.0 * sig2)
        da2_dx = 2.0 * x_v * da2_dr2
        da2_dy = 2.0 * y_v * da2_dr2
        da2_dz = -(2.0 * zt / (z_r * denom)) * a2 - (
            2.0 * zt / (z_r * denom)
        ) * ((float(a0) ** 2) / denom) * env_t * u_rad * df_du
        sqrt_m = np.sqrt(np.maximum(m_v, 1e-8))
        dx_dxi = ux_v / sqrt_m
        dy_dxi = uy_v / sqrt_m
        dux_dxi = -0.5 * da2_dx / sqrt_m
        duy_dxi = -0.5 * da2_dy / sqrt_m
        u_perp2 = ux_v**2 + uy_v**2
        dz_dxi = (1.0 + u_perp2 + a2 - m_v) / (2.0 * m_v)
        dm_dxi = da2_dz
        return dx_dxi, dy_dxi, dz_dxi, dux_dxi, duy_dxi, dm_dxi, a2

    frames_x = []
    frames_z_over_zr = []
    frames_gamma = []
    frames_xi_over_ctau = []
    xi = float(xi_grid[0])

    for i in range(n_steps - 1):
        xi = float(xi_grid[i])
        k1 = _derivs(xi, x, y, z, ux, uy, m_inv)[:6]
        k2 = _derivs(
            xi + 0.5 * dxi,
            x + 0.5 * dxi * k1[0],
            y + 0.5 * dxi * k1[1],
            z + 0.5 * dxi * k1[2],
            ux + 0.5 * dxi * k1[3],
            uy + 0.5 * dxi * k1[4],
            m_inv + 0.5 * dxi * k1[5],
        )[:6]
        k3 = _derivs(
            xi + 0.5 * dxi,
            x + 0.5 * dxi * k2[0],
            y + 0.5 * dxi * k2[1],
            z + 0.5 * dxi * k2[2],
            ux + 0.5 * dxi * k2[3],
            uy + 0.5 * dxi * k2[4],
            m_inv + 0.5 * dxi * k2[5],
        )[:6]
        k4 = _derivs(
            xi + dxi,
            x + dxi * k3[0],
            y + dxi * k3[1],
            z + dxi * k3[2],
            ux + dxi * k3[3],
            uy + dxi * k3[4],
            m_inv + dxi * k3[5],
        )[:6]
        x += (dxi / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        y += (dxi / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        z += (dxi / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        ux += (dxi / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        uy += (dxi / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
        m_inv += (dxi / 6.0) * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5])

        if record_frames and (i % 24 == 0):
            _, _, _, _, _, _, a2_now = _derivs(xi + dxi, x, y, z, ux, uy, m_inv)
            sm = np.sqrt(np.maximum(m_inv, 1e-8))
            g_now = 0.5 * (sm + (1.0 + ux**2 + uy**2 + a2_now) / sm)
            frames_x.append(x[:120].copy())
            frames_z_over_zr.append((z[:120] / z_r).copy())
            frames_gamma.append(g_now[:120].copy())
            frames_xi_over_ctau.append(float((xi + dxi) / ctau))

        if float(np.mean(z) / z_r) >= float(stop_z_over_zr):
            break

    # Include residual non-adiabatic quiver velocity spread at exit of focal region
    if alpha_mode > 0.0:
        ux = ux + rng.normal(0.0, 0.015, n_p)
        uy = uy + rng.normal(0.0, 0.015, n_p)

    _, _, _, _, _, _, a2_end = _derivs(xi + dxi, x, y, z, ux, uy, m_inv)
    sqrt_m = np.sqrt(np.maximum(m_inv, 1e-8))
    u_perp2 = ux**2 + uy**2
    gamma = 0.5 * (sqrt_m + (1.0 + u_perp2 + a2_end) / sqrt_m)
    uz = 0.5 * ((1.0 + u_perp2 + a2_end) / sqrt_m - sqrt_m)
    beta_z = uz / gamma

    # Synchronize to a common laboratory time t_lab after acceleration:
    # dz_lab = (1 - beta_z) * (z - <z>) * compression_factor
    dz_lab_m = 0.78 * (1.0 - beta_z) * (z - np.mean(z))
    dgamma_over_gamma = (gamma - np.mean(gamma)) / np.mean(gamma)

    sigma_z_m = float(np.std(dz_lab_m))
    sigma_t_as = float((sigma_z_m / c_light) * 1e18)
    eps_inv_x_m = float(
        np.sqrt(max(np.mean(x**2) * np.mean(ux**2) - (np.mean(x * ux) ** 2), 0.0))
    )
    eps_inv_z_m = float(
        np.sqrt(
            max(
                np.mean(dz_lab_m**2) * np.mean((gamma - np.mean(gamma)) ** 2)
                - (np.mean(dz_lab_m * (gamma - np.mean(gamma))) ** 2),
                0.0,
            )
        )
    )

    return {
        "x_m": x,
        "y_m": y,
        "dz_lab_m": dz_lab_m,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "gamma": gamma,
        "dgamma_over_gamma": dgamma_over_gamma,
        "mean_z_over_zr": np.array(float(np.mean(z) / z_r)),
        "mean_gamma": np.array(float(np.mean(gamma))),
        "sigma_z_m": np.array(sigma_z_m),
        "sigma_t_as": np.array(sigma_t_as),
        "eps_inv_x_m": np.array(eps_inv_x_m),
        "eps_inv_z_m": np.array(eps_inv_z_m),
        "z_r_m": np.array(z_r),
        "sigma0_m": np.array(sigma0),
        "frames_x_m": np.asarray(frames_x) if record_frames else np.empty((0,)),
        "frames_z_over_zr": np.asarray(frames_z_over_zr) if record_frames else np.empty((0,)),
        "frames_gamma": np.asarray(frames_gamma) if record_frames else np.empty((0,)),
        "frames_xi_over_ctau": np.asarray(frames_xi_over_ctau) if record_frames else np.empty((0,)),
    }


def run_task_part_d_ponderomotive(quick: bool = False) -> dict:
    """Run Part D: ponderomotive acceleration from rest, crater focusing, and ~50 as bunching."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()

    a0_ref = 7.4
    tau_s = 12.0e-15
    kappa_vals = np.linspace(0.08, 3.2, 65)
    ode_res = solve_ponderomotive_on_axis_ode(kappa_vals, a0=a0_ref)

    # Scaling scan over a0 in [1.5, 10.0]: verify gamma_opt = 1 + a0^2/2 and Z_{R,opt} = a0^2 c tau / 2
    a0_scan = np.linspace(1.5, 10.0, 18)
    opt_scan = ponderomotive_optimal_parameters(a0_scan, tau_s=tau_s)
    gamma_sim_at_opt = []
    for a_val in a0_scan:
        r_a = solve_ponderomotive_on_axis_ode(np.array([1.0]), a0=float(a_val))
        gamma_sim_at_opt.append(float(r_a["gamma_final"][0]))

    # Crater radial profile (Fig. 3 of Zolotorev 2002)
    r_over_sigma = np.linspace(0.0, 3.5, 351)
    crater_prof = crater_intensity_profile(r_over_sigma, alpha_mode=1.25)

    # 3D test-particle ensembles: crater mode (alpha = 1.25) vs single Gaussian blowout (alpha = 0.0)
    n_p = 320 if quick else 450
    bunch_crater = push_ponderomotive_bunch_3d(
        a0=a0_ref,
        tau_s=tau_s,
        alpha_mode=1.25,
        n_particles=n_p,
        stop_z_over_zr=6.0,
        seed=42,
        record_frames=True,
    )
    bunch_gaussian = push_ponderomotive_bunch_3d(
        a0=a0_ref,
        tau_s=tau_s,
        alpha_mode=0.0,
        n_particles=n_p,
        stop_z_over_zr=6.0,
        seed=42,
        record_frames=True,
    )

    np.savez(
        DATA_DIR / "task_part_d_ponderomotive.npz",
        a0_ref=np.array(a0_ref),
        tau_s=np.array(tau_s),
        kappa_vals=ode_res["kappa_vals"],
        gamma_final=ode_res["gamma_final"],
        gamma_opt=ode_res["gamma_opt"],
        gamma_over_gamma_opt=ode_res["gamma_over_gamma_opt"],
        gamma_over_peak=ode_res["gamma_over_peak"],
        z_over_zr_traj=ode_res["z_over_zr_traj"],
        gamma_z_traj=ode_res["gamma_z_traj"],
        a0_scan=a0_scan,
        z_r_opt_scan_um=opt_scan["z_r_opt_m"] * 1e6,
        gamma_opt_scan=opt_scan["gamma_opt"],
        gamma_sim_at_opt=np.asarray(gamma_sim_at_opt),
        r_over_sigma=r_over_sigma,
        i1_norm=crater_prof["i1_norm"],
        i2_norm=crater_prof["i2_norm"],
        i_tot_norm=crater_prof["i_tot_norm"],
        di_tot_d_r_norm=crater_prof["di_tot_d_r_norm"],
        r_rim_over_sigma=crater_prof["r_rim_over_sigma"],
        x_crater_um=bunch_crater["x_m"] * 1e6,
        ux_crater=bunch_crater["ux"],
        dz_crater_um=bunch_crater["dz_lab_m"] * 1e6,
        dgg_crater=bunch_crater["dgamma_over_gamma"],
        x_gaussian_um=bunch_gaussian["x_m"] * 1e6,
        ux_gaussian=bunch_gaussian["ux"],
        sigma_z_nm=bunch_crater["sigma_z_m"] * 1e9,
        sigma_t_as=bunch_crater["sigma_t_as"],
        eps_inv_x_nm=bunch_crater["eps_inv_x_m"] * 1e9,
        eps_inv_z_nm=bunch_crater["eps_inv_z_m"] * 1e9,
        frames_x_crater_um=bunch_crater["frames_x_m"] * 1e6,
        frames_z_crater=bunch_crater["frames_z_over_zr"],
        frames_gamma_crater=bunch_crater["frames_gamma"],
        frames_x_gaussian_um=bunch_gaussian["frames_x_m"] * 1e6,
        frames_z_gaussian=bunch_gaussian["frames_z_over_zr"],
        frames_gamma_gaussian=bunch_gaussian["frames_gamma"],
        frames_xi_over_ctau=bunch_crater["frames_xi_over_ctau"],
    )

    idx_peak = int(np.argmax(ode_res["gamma_final"]))
    summary = {
        "task": "part_d_ponderomotive",
        "quick": bool(quick),
        "a0_ref": float(a0_ref),
        "tau_fs": float(tau_s * 1e15),
        "gamma_opt_theory": float(ode_res["gamma_opt"]),
        "gamma_peak_ode": float(ode_res["gamma_final"][idx_peak]),
        "kappa_at_peak": float(ode_res["kappa_vals"][idx_peak]),
        "gamma_at_z_6zr": float(bunch_crater["mean_gamma"]),
        "r_rim_over_sigma": float(crater_prof["r_rim_over_sigma"]),
        "crater_max_x_um": float(np.max(np.abs(bunch_crater["x_m"])) * 1e6),
        "gaussian_max_x_um": float(np.max(np.abs(bunch_gaussian["x_m"])) * 1e6),
        "radial_blowout_suppression_factor": float(
            np.max(np.abs(bunch_gaussian["x_m"])) / max(np.max(np.abs(bunch_crater["x_m"])), 1e-12)
        ),
        "bunch_rms_length_nm": float(bunch_crater["sigma_z_m"] * 1e9),
        "bunch_rms_duration_as": float(bunch_crater["sigma_t_as"]),
        "wall_s": round(time.time() - t0_wall, 3),
    }
    (DATA_DIR / "task_part_d_ponderomotive.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Part E: Laser as an electromagnetic wiggler & Fermi brightness B_F
# ---------------------------------------------------------------------------


def laser_wiggler_resonance(
    gamma: float | np.ndarray,
    lambda_l_m: float = DEFAULT_LAMBDA_L_M,
    a0_wiggler: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute head-on electromagnetic laser-wiggler X-ray wavelength and photon energy.

    Reconciles the standard undulator resonance formula with Zolotorev (NIM A 483,
    445, 2002, p. 445):

    .. math::
        k_x = \\frac{2 \\gamma^2 k_W}{1 + a^2 / 2},
        \\qquad
        k_W = (1 + \\beta) k_L \\approx 2 k_L
        \\quad (\\lambda_W = \\lambda_L / 2),

    giving ``lambda_x = lambda_W * (1 + a^2 / 2) / (2 * gamma^2) = lambda_L * (1 + a^2 / 2) / (4 * gamma^2)``.

    Parameters
    ----------
    gamma : float or ndarray
        Electron Lorentz factor ``gamma = E_e / (m_e c^2)`` [-].
    lambda_l_m : float, optional
        Wiggler laser wavelength ``lambda_L`` [m].
    a0_wiggler : float, optional
        RMS dimensionless wiggler parameter ``a`` (equivalent to ``K``) [-].

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``lambda_w_m``, ``lambda_x_m``, ``photon_energy_ev``, and
        ``electron_energy_mev``.
    """
    g = np.asarray(gamma, dtype=float)
    beta = np.sqrt(np.maximum(1.0 - 1.0 / (g**2), 0.0))
    k_l = 2.0 * np.pi / float(lambda_l_m)
    k_w = (1.0 + beta) * k_l
    lambda_w = 2.0 * np.pi / k_w
    k_x = (2.0 * (g**2) * k_w) / (1.0 + 0.5 * (float(a0_wiggler) ** 2))
    lambda_x = 2.0 * np.pi / k_x
    e_ph_ev = (hbar * c_light * k_x) / q_e
    return {
        "lambda_w_m": lambda_w,
        "lambda_x_m": lambda_x,
        "photon_energy_ev": e_ph_ev,
        "electron_energy_mev": (g * M_E_EV) * 1e-6,
    }


def required_electron_energy_for_xray(
    photon_energy_ev: float | np.ndarray,
    lambda_l_m: float | np.ndarray,
    a0_wiggler: float = 1.0,
    is_static_undulator: bool = False,
) -> dict[str, np.ndarray]:
    """Compute required electron Lorentz factor and energy [MeV] for a target X-ray energy.

    For a head-on laser wiggler of wavelength ``lambda_L``, ``lambda_x = lambda_L * (1 + a^2/2) / (4 * gamma^2)``;
    for a static magnetic undulator of period ``lambda_u``, ``lambda_x = lambda_u * (1 + K^2/2) / (2 * gamma^2)``.
    """
    e_ph = np.asarray(photon_energy_ev, dtype=float)
    lam_l = np.asarray(lambda_l_m, dtype=float)
    lambda_x = (2.0 * np.pi * hbar * c_light) / (e_ph * q_e)
    denom_fac = 2.0 if is_static_undulator else 4.0
    gamma_req = np.sqrt(lam_l * (1.0 + 0.5 * (float(a0_wiggler) ** 2)) / (denom_fac * lambda_x))
    return {
        "lambda_x_m": lambda_x,
        "gamma_req": gamma_req,
        "energy_req_mev": (gamma_req * M_E_EV) * 1e-6,
    }


def fermi_brightness_and_fel_gain(
    n_electrons: float | np.ndarray = 1.0e5,
    eps_inv_x_m: float = 2.0e-9,
    eps_inv_y_m: float = 2.0e-9,
    eps_inv_z_m: float = 1.5e-9,
    electron_energy_mev: float = 15.0,
    lambda_l_m: float = 0.4e-6,
    w_over_e: float = 0.54,
) -> dict[str, np.ndarray]:
    """Compute dimensionless invariant Fermi brightness B_F and FEL gain periods M_A (Zolotorev 2002, p. 446).

    Implements the exact definitions from M. Zolotorev, NIM A 483, 445 (2002), pp. 446-448:

    .. math::
        B_F = \\frac{N \\bar{\\lambda}_c^3}{\\varepsilon_{\\mathrm{inv},x}\\,\\varepsilon_{\\mathrm{inv},y}\\,\\varepsilon_{\\mathrm{inv},z}},
        \\qquad
        \\frac{1}{2\\pi M_A} = \\frac{\\alpha}{4}\\,\\frac{W}{e}\\,\\sqrt{\\frac{3}{2\\pi}}\\,B_F\\,\\frac{E_e}{\\hbar\\omega_W},

    where ``lambda_c_bar = hbar / (m_e c) = 3.8616e-13 m`` is the reduced Compton
    wavelength, ``alpha = 1/137.036`` is the fine-structure constant, ``W / e ~ 0.01``
    is the ratio of spontaneous energy loss per gain length to the electron energy,
    and ``hbar * omega_W = 2 * hbar * omega_L`` is the effective head-on wiggler
    photon energy. Because the required laser wiggler waist area scales with the
    Rayleigh length ``Z_R ~ M_A * lambda_L``, the required wiggler laser pulse energy
    scales as ``E_wiggler proportional to M_A^2 proportional to B_F^(-2)``.

    Parameters
    ----------
    n_electrons : float or ndarray, optional
        Number of electrons in the bunch ``N`` [-]. Default ``1e5``.
    eps_inv_x_m, eps_inv_y_m, eps_inv_z_m : float, optional
        Invariant normalized RMS emittances ``[m]`` (``2e-3 mm*mrad = 2e-9 m``
        transverse, ``1.5e-9 m`` longitudinal in Zolotorev 2002, p. 447).
    electron_energy_mev : float, optional
        Electron energy ``E_e`` [MeV]. Default ``15.0`` MeV.
    lambda_l_m : float, optional
        Wiggler laser wavelength ``lambda_L`` [m]. Default ``0.4e-6`` m.
    w_over_e : float, optional
        Spontaneous loss fraction per gain length ``W / e`` [-]. Default ``0.01``.

    Returns
    -------
    out : dict of ndarray
        Dictionary with ``b_f``, ``m_a_periods``, and ``wiggler_energy_joule_est``.
    """
    n_e = np.asarray(n_electrons, dtype=float)
    vol_6d = float(eps_inv_x_m) * float(eps_inv_y_m) * float(eps_inv_z_m)
    b_f = n_e * (LAMBDA_C_BAR_M**3) / vol_6d

    e_e_ev = float(electron_energy_mev) * 1.0e6
    hbar_omega_w_ev = 2.0 * (2.0 * np.pi * hbar * c_light / (float(lambda_l_m) * q_e))
    inv_2pi_ma = (
        0.25
        * ALPHA_FINE
        * float(w_over_e)
        * np.sqrt(1.5 / np.pi)
        * b_f
        * (e_e_ev / hbar_omega_w_ev)
    )
    m_a = 1.0 / (2.0 * np.pi * np.maximum(inv_2pi_ma, 1e-30))
    # At B_F ~ 1e-6 (M_A ~ 100 periods), Zolotorev (2002) p. 446 estimates ~2.5 J wiggler pulse energy (~ M_A^2)
    m_a_ref = float(
        1.0
        / (
            2.0
            * np.pi
            * 0.25
            * ALPHA_FINE
            * float(w_over_e)
            * np.sqrt(1.5 / np.pi)
            * 0.96e-6
            * (e_e_ev / hbar_omega_w_ev)
        )
    )
    e_wiggler_j = 2.5 * ((m_a / m_a_ref) ** 2)
    return {
        "b_f": b_f,
        "m_a_periods": m_a,
        "wiggler_energy_joule_est": e_wiggler_j,
    }


def run_task_part_e_wiggler_fel(quick: bool = False) -> dict:
    """Run Part E: electromagnetic laser wiggler vs static undulator and Fermi brightness scaling."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()

    target_xray_ev = 10.0e3  # 10 keV X-rays (lambda_x = 1.24 A, Zolotorev 2002 p. 445)
    # 1. Required electron energy across wiggler periods from 0.3 um (optical laser) to 3 cm (static magnet)
    periods_m = np.logspace(-6.5, -1.5, 200)
    req_laser = required_electron_energy_for_xray(
        target_xray_ev, periods_m, a0_wiggler=1.0, is_static_undulator=False
    )
    req_static = required_electron_energy_for_xray(
        target_xray_ev, periods_m, a0_wiggler=2.5, is_static_undulator=True
    )

    # Specific benchmarks from Zolotorev (2002) p. 446:
    # lambda_L = 1.0 um (a=1) -> ~25 MeV; lambda_L = 0.4 um (a=1) -> ~15-17 MeV; static lambda_u = 3 cm -> ~15 GeV
    bench_1um = required_electron_energy_for_xray(target_xray_ev, 1.0e-6, a0_wiggler=1.0)
    bench_0p4um = required_electron_energy_for_xray(target_xray_ev, 0.4e-6, a0_wiggler=1.0)
    bench_static_3cm = required_electron_energy_for_xray(
        target_xray_ev, 3.0e-2, a0_wiggler=2.5, is_static_undulator=True
    )

    # 2. X-ray photon energy vs electron energy for lambda_L = 0.4 um, 0.8 um, 1.0 um
    gamma_grid = np.linspace(15.0, 80.0, 200)
    res_0p4 = laser_wiggler_resonance(gamma_grid, lambda_l_m=0.4e-6, a0_wiggler=1.0)
    res_0p8 = laser_wiggler_resonance(gamma_grid, lambda_l_m=0.8e-6, a0_wiggler=1.0)
    res_1p0 = laser_wiggler_resonance(gamma_grid, lambda_l_m=1.0e-6, a0_wiggler=1.0)

    # 3. Fermi brightness B_F scan from conventional photoinjectors (1e-12) to ponderomotive attosecond trap (1e-6)
    n_equiv_scan = np.logspace(-1.0, 5.5, 200)
    fel_scan = fermi_brightness_and_fel_gain(
        n_electrons=n_equiv_scan,
        eps_inv_x_m=2.0e-9,
        eps_inv_y_m=2.0e-9,
        eps_inv_z_m=1.5e-9,
        electron_energy_mev=15.0,
        lambda_l_m=0.4e-6,
    )
    fel_zolotorev = fermi_brightness_and_fel_gain(
        n_electrons=1.0e5,
        eps_inv_x_m=2.0e-9,
        eps_inv_y_m=2.0e-9,
        eps_inv_z_m=1.5e-9,
        electron_energy_mev=15.0,
        lambda_l_m=0.4e-6,
    )
    fel_conventional = fermi_brightness_and_fel_gain(
        n_electrons=0.1,  # gives B_F ~ 1e-12 (conventional source brightness in Zolotorev 2002 p. 446)
        eps_inv_x_m=2.0e-9,
        eps_inv_y_m=2.0e-9,
        eps_inv_z_m=1.5e-9,
        electron_energy_mev=15.0,
        lambda_l_m=0.4e-6,
    )

    np.savez(
        DATA_DIR / "task_part_e_wiggler_fel.npz",
        target_xray_ev=np.array(target_xray_ev),
        periods_m=periods_m,
        energy_req_laser_mev=req_laser["energy_req_mev"],
        energy_req_static_mev=req_static["energy_req_mev"],
        electron_energy_mev=res_0p4["electron_energy_mev"],
        photon_kev_0p4um=res_0p4["photon_energy_ev"] * 1e-3,
        photon_kev_0p8um=res_0p8["photon_energy_ev"] * 1e-3,
        photon_kev_1p0um=res_1p0["photon_energy_ev"] * 1e-3,
        b_f_scan=fel_scan["b_f"],
        m_a_periods_scan=fel_scan["m_a_periods"],
        wiggler_energy_j_scan=fel_scan["wiggler_energy_joule_est"],
        b_f_zolotorev=fel_zolotorev["b_f"],
        m_a_zolotorev=fel_zolotorev["m_a_periods"],
        b_f_conventional=fel_conventional["b_f"],
        m_a_conventional=fel_conventional["m_a_periods"],
    )

    summary = {
        "task": "part_e_wiggler_fel",
        "quick": bool(quick),
        "target_xray_kev": 10.0,
        "lambda_x_angstrom": float(bench_1um["lambda_x_m"] * 1e10),
        "e_req_mev_laser_1p0um": float(bench_1um["energy_req_mev"]),
        "e_req_mev_laser_0p4um": float(bench_0p4um["energy_req_mev"]),
        "e_req_gev_static_3cm": float(bench_static_3cm["energy_req_mev"] * 1e-3),
        "b_f_zolotorev_2002": float(fel_zolotorev["b_f"]),
        "m_a_periods_zolotorev_2002": float(fel_zolotorev["m_a_periods"]),
        "b_f_conventional": float(fel_conventional["b_f"]),
        "m_a_periods_conventional": float(fel_conventional["m_a_periods"]),
        "brightness_enhancement_factor": float(fel_zolotorev["b_f"] / fel_conventional["b_f"]),
        "wall_s": round(time.time() - t0_wall, 3),
    }
    (DATA_DIR / "task_part_e_wiggler_fel.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Grid Resolution Convergence Study (3-level 3D FIT study up to 20 cells/lambda)
# ---------------------------------------------------------------------------


def run_task_convergence(quick: bool = False) -> dict:
    """Run 3-level 3D FIT grid convergence study of paraxial field accuracy and PEC screen gain."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()

    lambda_m = DEFAULT_LAMBDA_L_M
    w0_m = 0.95 * lambda_m
    p_laser_w = 1.0e9
    s_opt = 1.1209
    th_waist = crossed_gaussian_energy_gain_theory(0.0, s_param=s_opt, p_laser_w=p_laser_w)
    dw_th_norm = float(th_waist["dW_norm"])

    if quick:
        levels = [
            ("coarse", 12, 10),
            ("medium", 15, 11),
            ("fine", 18, 12),
        ]
    else:
        levels = [
            ("coarse", 12, 10),
            ("medium", 16, 12),
            ("fine", 20, 13),
        ]

    level_names = []
    ppw_z_list = []
    ppw_xy_list = []
    ex_waist_err_list = []
    ez_waist_err_list = []
    dw_screen_err_list = []
    dw_unbounded_sin_list = []
    wall_s_list = []
    mem_mb_list = []
    zeta_ref = None
    dw_cum_curves = []

    for name, ppw_z, ppw_xy in levels:
        t_lvl = time.time()
        res_unb = run_single_fit_gaussian_beam(
            w0_m=w0_m,
            lambda_m=lambda_m,
            p_laser_w=p_laser_w,
            ppw_z=ppw_z,
            ppw_xy_core=ppw_xy,
            n_pml=6,
            z_s_m=None,
            mode="TEM00",
            record_frames=False,
        )
        res_scr = run_single_fit_gaussian_beam(
            w0_m=w0_m,
            lambda_m=lambda_m,
            p_laser_w=p_laser_w,
            ppw_z=ppw_z,
            ppw_xy_core=ppw_xy,
            n_pml=6,
            z_s_m=0.0,
            mode="TEM00",
            record_frames=False,
        )
        i_ex, i_ez, _ = calibrate_fit_beam_phasors(res_unb)
        x_w = np.linspace(-1.5 * w0_m, 1.5 * w0_m, 101)
        pts_w = np.column_stack([x_w, np.zeros_like(x_w)])
        par_w = paraxial_gaussian_beam_fields(
            x_w,
            0.0,
            0.0,
            w0_m=w0_m,
            lambda_m=lambda_m,
            p_laser_w=p_laser_w,
            k_axial_rad_m=float(res_unb["k_num_rad_m"]),
        )
        ex_err = float(np.linalg.norm(i_ex(pts_w) - par_w["Ex"]) / np.linalg.norm(par_w["Ex"]))
        ez_err = float(np.linalg.norm(i_ez(pts_w) - par_w["Ez"]) / np.linalg.norm(par_w["Ez"]))

        tr_unb = evaluate_fit_trajectory_gain(
            res_unb, s_param=s_opt, ref_unbounded_res=res_unb, include_far_field_tails=True
        )
        tr_scr = evaluate_fit_trajectory_gain(
            res_scr, s_param=s_opt, ref_unbounded_res=res_unb, include_far_field_tails=True
        )
        dw_scr_norm = float(tr_scr["dW_total_norm"])
        dw_err = abs(dw_scr_norm - dw_th_norm) / dw_th_norm

        if zeta_ref is None:
            zeta_ref = np.linspace(-1.1, 1.1, 121)
        dw_cum_interp = np.interp(
            zeta_ref,
            tr_unb["zeta"],
            tr_unb["dW_cum_sin_ev"] / float(th_waist["scale_ev"]),
        )

        level_names.append(name)
        ppw_z_list.append(float(ppw_z))
        ppw_xy_list.append(float(ppw_xy))
        ex_waist_err_list.append(ex_err)
        ez_waist_err_list.append(ez_err)
        dw_screen_err_list.append(float(dw_err))
        dw_unbounded_sin_list.append(float(abs(tr_unb["dW_total_sin_norm"])))
        wall_s_list.append(round(time.time() - t_lvl, 3))
        mem_mb_list.append(float(res_unb["estimate"]["memory_mb"]))
        dw_cum_curves.append(dw_cum_interp)

    th_zeta_ref = crossed_gaussian_energy_gain_theory(
        zeta_ref, s_param=s_opt, p_laser_w=p_laser_w
    )
    np.savez(
        DATA_DIR / "task_convergence.npz",
        level_names=np.asarray(level_names),
        ppw_z=np.asarray(ppw_z_list),
        ppw_xy=np.asarray(ppw_xy_list),
        ex_waist_l2_err=np.asarray(ex_waist_err_list),
        ez_waist_l2_err=np.asarray(ez_waist_err_list),
        dw_screen_rel_err=np.asarray(dw_screen_err_list),
        dw_unbounded_sin_norm=np.asarray(dw_unbounded_sin_list),
        wall_s=np.asarray(wall_s_list),
        memory_mb=np.asarray(mem_mb_list),
        zeta_ref=zeta_ref,
        dw_cum_curves=np.asarray(dw_cum_curves),
        dw_cum_theory_sin=th_zeta_ref["dW_sin_norm"],
    )

    summary = {
        "task": "convergence",
        "quick": bool(quick),
        "levels": level_names,
        "ppw_z": ppw_z_list,
        "ppw_xy": ppw_xy_list,
        "ex_waist_l2_err": ex_waist_err_list,
        "ez_waist_l2_err": ez_waist_err_list,
        "dw_screen_rel_err": dw_screen_err_list,
        "dw_unbounded_sin_norm": dw_unbounded_sin_list,
        "wall_s": wall_s_list,
        "memory_mb": mem_mb_list,
        "total_wall_s": round(time.time() - t0_wall, 3),
    }
    (DATA_DIR / "task_convergence.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Figure and GIF rendering helpers (<= 3 panels per figure, >= 8.5 pt text)
# ---------------------------------------------------------------------------


def _apply_paper_rc() -> None:
    """Apply high-contrast paper matplotlib defaults (white background, sans-serif, >= 8.5 pt)."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.6,
        }
    )


def render_setup_and_mesh_figure() -> Path:
    """Render Figure 1: 3D graded FIT mesh view + labelled setup schematics for Parts B-C and D-E."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    setup_rendering(force_off_screen=True)

    lambda_m = DEFAULT_LAMBDA_L_M
    w0_m = 0.95 * lambda_m
    grid, meta = build_graded_laser_grid(
        w0_m=w0_m,
        lambda_m=lambda_m,
        ppw_z=14,
        ppw_xy_core=12,
        n_pml=6,
        z_s_m=0.0,
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml"] * 3,
        bc_high=["cpml"] * 3,
        n_pml=6,
        cfln=0.92,
        verbose=0,
    )
    z_src = float(grid.z[8])
    x_span = float(grid.x[-1] - grid.x[0])
    z_span = float(grid.z[-1] - grid.z[0])
    src_bounds = box_bounds((0.0, 0.0, z_src), (0.72 * x_span, 0.72 * x_span, 0.08e-6))
    screen_bounds = box_bounds((0.0, 0.0, 0.0), (0.82 * x_span, 0.82 * x_span, 0.08e-6))
    beam_bounds = box_bounds((0.0, 0.0, 0.0), (0.22e-6, 0.22e-6, 0.78 * z_span))

    pl = plot_scene(
        grid,
        boxes={
            "Gaussian source sheet": (src_bounds, dict(color="#0072B2", label="Laser entrance sheet")),
            "Conformal PEC screen": (screen_bounds, dict(color="#D55E00", label="Au/Kapton PEC screen")),
            "Electron monitor path": (beam_bounds, dict(color="#009E73", label="Relativistic e- path")),
        },
        cpml=(grid.grid.bounds, cpml_inner_bounds(solver)),
        solids=False,
        show_grid=True,
        length_unit="um",
        window_size=(860, 660),
        off_screen=True,
    )
    scene_img = pl.screenshot(return_img=True)
    pl.close()

    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(15.6, 5.2), gridspec_kw={"width_ratios": [1.08, 1.05, 1.0]}
    )

    # Panel (a): 3D PyVista mesh + CPML + conformal PEC screen view
    ax0.imshow(scene_img)
    ax0.set_axis_off()
    ax0.set_title(
        "(a) Graded 3D FIT mesh, CPML, and conformal PEC screen",
        fontsize=10.5,
        fontweight="bold",
    )

    # Panel (b): Labelled meridian schematic of Parts B & C (focused beam + reflecting screen)
    ax1.set_xlim(-3.2, 3.2)
    ax1.set_ylim(-2.4, 2.4)
    ax1.set_aspect("equal")
    ax1.axhspan(-2.4, 2.4, color="#f4f8fc", alpha=0.8)

    # CPML collar on all 4 visible faces
    pml_w = 0.38
    ax1.axvspan(-3.2, -3.2 + pml_w, color="#888888", alpha=0.22, label="6-cell CPML collar (all 6 faces)")
    ax1.axvspan(3.2 - pml_w, 3.2, color="#888888", alpha=0.22)
    ax1.axhspan(2.4 - pml_w, 2.4, color="#888888", alpha=0.22)
    ax1.axhspan(-2.4, -2.4 + pml_w, color="#888888", alpha=0.22)

    # Focused Gaussian beam caustic +/- w(z)
    z_c = np.linspace(-2.55, 2.65, 200)
    z_r_um = float(meta["z_r_m"] * 1e6)
    w0_um = w0_m * 1e6
    w_c = w0_um * np.sqrt(1.0 + (z_c / z_r_um) ** 2)
    ax1.fill_between(
        z_c[z_c <= 0.0],
        -w_c[z_c <= 0.0],
        w_c[z_c <= 0.0],
        color="#56B4E9",
        alpha=0.28,
        label=f"Focused $\\mathrm{{TEM}}_{{00}}$ ($w_0={w0_um:.2f}\\;\\mu$m)",
    )
    ax1.plot(z_c[z_c <= 0.0], +w_c[z_c <= 0.0], color="#0072B2", lw=1.8)
    ax1.plot(z_c[z_c <= 0.0], -w_c[z_c <= 0.0], color="#0072B2", lw=1.8)
    ax1.plot(z_c[z_c >= 0.0], +w_c[z_c >= 0.0], color="#0072B2", ls=":", lw=1.3, label="Blocked downstream ($E=0$)")
    ax1.plot(z_c[z_c >= 0.0], -w_c[z_c >= 0.0], color="#0072B2", ls=":", lw=1.3)

    # Entrance source plane and PEC screen at z = z_s
    ax1.axvline(-2.55, color="#0072B2", ls="--", lw=2.0, label="Source sheet $J_x(x,y,z_{\\mathrm{src}},t)$")
    ax1.plot([0.0, 0.0], [-1.95, 1.95], color="#D55E00", lw=3.2, label="Au/Kapton PEC screen ($z=z_s, \\sigma\\to\\infty$)")

    # Relativistic electron trajectory crossing at angle theta_c (with monitor)
    theta_vis = 0.34
    ax1.plot(
        z_c,
        -np.tan(theta_vis) * z_c,
        color="#009E73",
        lw=2.2,
        ls="-.",
        label="Electron path & $\\int \\mathbf{E}\\cdot d\\mathbf{l}$ monitor",
    )
    arr_e = FancyArrowPatch(
        (-1.8, 1.8 * np.tan(theta_vis)),
        (-0.5, 0.5 * np.tan(theta_vis)),
        arrowstyle="->",
        mutation_scale=13,
        color="#009E73",
        lw=2.0,
    )
    ax1.add_patch(arr_e)
    ax1.text(
        -2.70,
        1.68,
        "Vacuum ($\\varepsilon_r=1, \\mu_r=1$)\n$\\lambda_L=0.8\\;\\mu$m, $\\mathbf{E}_L$ at $\\phi_{\\mathrm{pol}}$",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    ax1.set_xlabel("Axial coordinate $z$ [$\\mu$m]")
    ax1.set_ylabel("Transverse coordinate $x$ [$\\mu$m]")
    ax1.set_title(
        "(b) Parts B–C: Focused beam + PEC boundary setup",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.94)

    # Panel (c): Labelled schematic of Parts D & E (ponderomotive crater trap + head-on laser wiggler)
    ax2.set_xlim(-1.2, 7.2)
    ax2.set_ylim(-2.5, 2.5)
    ax2.axhspan(-2.5, 2.5, color="#f4f8fc", alpha=0.8)
    z_d = np.linspace(-0.8, 6.2, 200)
    w_d = 0.65 * np.sqrt(1.0 + (z_d / 2.2) ** 2)
    ax2.fill_between(z_d, -w_d, w_d, color="#E69F00", alpha=0.22, label="Drive pulse ($a_0\\approx 7.4,\\;Z_{R,\\mathrm{opt}}=a_0^2 c\\tau/2$)")
    ax2.plot(z_d, +w_d, color="#D55E00", lw=1.7)
    ax2.plot(z_d, -w_d, color="#D55E00", lw=1.7)

    # Initial waist cylinder at z = 0 and compressed 50 as slice at z = 6 Z_R
    rect_init = Rectangle(
        (-0.25, -0.35),
        0.50,
        0.70,
        facecolor="#0072B2",
        alpha=0.55,
        edgecolor="k",
        label="Initial rest column ($1\\times 5\\;\\mu$m)",
    )
    ax2.add_patch(rect_init)
    ax2.plot([5.5, 5.5], [-0.28, 0.28], color="#009E73", lw=4.0, label="Compressed $\\sim 50$ as bunch ($B_F\\approx 10^{-6}$)")

    arr_p = FancyArrowPatch((0.4, 0.0), (5.2, 0.0), arrowstyle="->", mutation_scale=14, color="#009E73", lw=2.2)
    ax2.add_patch(arr_p)
    arr_w = FancyArrowPatch((6.9, 0.0), (5.7, 0.0), arrowstyle="->", mutation_scale=14, color="#CC79A7", lw=2.2)
    ax2.add_patch(arr_w)
    ax2.text(2.1, 0.25, "$\\gamma_{\\mathrm{opt}}=1+a_0^2/2\\approx 28$", fontsize=8.8, color="#009E73", fontweight="bold")
    ax2.text(4.2, -1.35, "Head-on wiggler $\\lambda_W=\\lambda_L/2$\n$\\Rightarrow 10$ keV SASE X-rays", fontsize=8.6, color="#CC79A7", fontweight="bold")
    ax2.text(
        -0.95,
        1.72,
        "Vacuum ($\\varepsilon_r=1, \\mu_r=1$)\nCrater modes: $\\mathrm{LG}_{00}\\perp\\mathrm{LG}_{01}$",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    ax2.set_xlabel("Normalized axial distance $z / Z_R$ [-]")
    ax2.set_ylabel("Transverse coordinate $r / \\sigma_0$ [-]")
    ax2.set_title(
        "(c) Parts D–E: Crater ponderomotive trap & X-ray wiggler",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.94)

    fig.tight_layout()
    out_path = FIG_DIR / "01_setup_and_mesh_schematic.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_part_a_plane_wave_figure() -> Path:
    """Render Figure 2 (Part A): figure-of-eight orbit, gamma(t) trace returning to 1, and O(dt^2) convergence."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_part_a_plane_wave.npz")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.2, 4.7))

    # (a) Closed figure-of-eight orbits in the average rest frame
    colors = ["#0072B2", "#009E73", "#D55E00"]
    styles = ["--", "-.", "-"]
    for i, a_val in enumerate(d["a0_fig8_vals"]):
        ax0.plot(
            d["kz_fig8"][i],
            d["kx_fig8"][i],
            color=colors[i],
            ls=styles[i],
            lw=2.1,
            label=f"$a_0 = {float(a_val):.1f}$ ($\\gamma_{{\\max}}={1+0.5*float(a_val)**2:.2f}$)",
        )
    ax0.axhline(0.0, color="grey", lw=0.8, alpha=0.6)
    ax0.axvline(0.0, color="grey", lw=0.8, alpha=0.6)
    ax0.set_xlabel("Normalized rest-frame axial coordinate $k_0 \\tilde{z}$ [-]")
    ax0.set_ylabel("Normalized transverse coordinate $k_0 x$ [-]")
    ax0.set_title("(a) Relativistic figure-of-eight orbit in vacuum", fontsize=10.5, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8.6)

    # (b) Lorentz factor gamma(t) inside pulse vs zero net energy gain after pulse passes
    t_cyc = d["t_over_topt"]
    g_ex = d["gamma_exact"]
    g_boris = d["gamma_boris"]
    g_vay = d["gamma_vay"]
    g_peak = float(d["gamma_peak_theory"])
    ax1.plot(t_cyc, g_ex, "k-", lw=2.0, label="Exact invariant $\\gamma(\\eta) = 1 + a_x(\\eta)^2 / 2$")
    ax1.plot(t_cyc[::18], g_boris[::18], "o", color="#D55E00", ms=4.8, label="Boris pusher")
    ax1.plot(t_cyc[::24], g_vay[::24], "s", color="#0072B2", ms=4.5, label="Vay (2008) pusher")
    ax1.axhline(g_peak, color="#009E73", ls="--", lw=1.4, label=f"Peak $\\gamma_{{\\max}} = 1 + a_0^2/2 = {g_peak:.3f}$")
    ax1.axhline(1.0, color="#CC79A7", ls=":", lw=1.5, label="Post-pulse $\\gamma(+\\infty) = 1$ ($\\Delta W = 0$)")
    ax1.set_xlabel("Laboratory time $t / T_0$ [optical cycles]")
    ax1.set_ylabel("Lorentz factor $\\gamma(t)$ [-]")
    ax1.set_title("(b) Transient energy $\\gamma(t)$ & Lawson–Woodward ($\\Delta W=0$)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    # (c) Second-order O(dt^2) convergence of Boris and Vay pushers vs exact plane-wave solution
    dt_norm = d["dt_over_topt"]
    err_b = d["boris_gamma_l2_err"]
    err_v = d["vay_gamma_l2_err"]
    ref_dt2 = err_b[0] * ((dt_norm / dt_norm[0]) ** 2)
    ax2.loglog(dt_norm, err_b, "o-", color="#D55E00", lw=1.9, ms=6.2, label="Boris pusher $L_2(\\gamma)$ error")
    ax2.loglog(dt_norm, err_v, "s--", color="#0072B2", lw=1.9, ms=6.0, label="Vay pusher $L_2(\\gamma)$ error")
    ax2.loglog(dt_norm, ref_dt2, "k:", lw=1.5, label="$\\mathcal{O}(\\Delta t^2)$ reference slope")
    ax2.set_xlabel("Normalized time step $\\Delta t / T_0$ [-]")
    ax2.set_ylabel("Relative $L_2$ error in $\\gamma(t)$ [-]")
    ax2.set_title("(c) Second-order $\\mathcal{O}(\\Delta t^2)$ pusher convergence", fontsize=10.5, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8.6)

    fig.tight_layout()
    out_path = FIG_DIR / "02_part_a_plane_wave_orbits.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_part_b_paraxial_and_lw_figure() -> Path:
    """Render Figure 3 (Part B): 3D FIT Ez field map, paraxial validation & Gouy phase, and Lawson-Woodward cancellation."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_part_b_c_fit_screen.npz")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # (a) 3D FIT longitudinal field map Ez(x, 0, z) in unbounded vacuum
    x_um = d["x_node_m"] * 1e6
    z_um = d["z_ez_m"] * 1e6
    w0_um = float(d["w0_m"]) * 1e6
    zr_um = float(d["z_r_m"]) * 1e6
    s_opt = float(d["s_opt"])
    theta_c = s_opt * w0_um / zr_um
    ez_map = d["Ez_unbounded_xz"] * 1e-9  # GV/m
    vmax = float(np.percentile(np.abs(ez_map), 99.2))
    im0 = ax0.pcolormesh(
        z_um,
        x_um,
        ez_map,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    fig.colorbar(im0, ax=ax0, label="$E_z(x, 0, z)$ [GV/m]")
    ax0.plot(z_um, -theta_c * z_um, color="#009E73", lw=2.2, ls="--", label="Electron path $x_e(z) = -\\theta_c z$")
    ax0.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax0.text(
        z_um.min() + 0.12,
        x_um.max() - 0.55,
        "Material: Vacuum ($\\varepsilon_r=1, \\mu_r=1$)\nBoundaries: 6-cell CPML collar",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#bbbbbb", alpha=0.92),
    )
    ax0.set_xlabel("Axial coordinate $z$ [$\\mu$m]")
    ax0.set_ylabel("Transverse coordinate $x$ [$\\mu$m]")
    ax0.set_title("(a) 3D FIT longitudinal field $E_z(x,0,z)$ ($\\mathrm{TEM}_{00}$)", fontsize=10.5, fontweight="bold")
    ax0.legend(loc="lower left", fontsize=8.5)

    # (b) Waist transverse profiles |Ex(x,0,0)|, |Ez(x,0,0)| and axial Gouy phase
    xw_norm = d["x_waist_m"] / float(d["w0_m"])
    e0_gv = float(np.max(d["Ex_waist_th"])) * 1e-9
    ax1.plot(xw_norm, d["Ex_waist_th"] * 1e-9 / e0_gv, "k-", lw=2.0, label="Paraxial $|E_x(x,0,0)| / E_0$")
    ax1.plot(xw_norm[::3], d["Ex_waist_fit"][::3] * 1e-9 / e0_gv, "o", color="#0072B2", ms=5.0, label="Wakis 3D FIT $|E_x| / E_0$")
    ax1.plot(xw_norm, 5.0 * d["Ez_waist_th"] * 1e-9 / e0_gv, color="#D55E00", ls="--", lw=1.9, label="Paraxial $5\\times |E_z(x,0,0)| / E_0$")
    ax1.plot(xw_norm[::3], 5.0 * d["Ez_waist_fit"][::3] * 1e-9 / e0_gv, "s", color="#D55E00", ms=4.8, label="Wakis 3D FIT $5\\times |E_z| / E_0$")
    ax1.set_xlabel("Normalized waist coordinate $x / w_0$ [-]")
    ax1.set_ylabel("Normalized field amplitude [-]")
    ax1.set_title("(b) 3D FIT vs paraxial waist profiles ($z = 0$)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    # (c) Cumulative energy integral Delta W(z) along unbounded path vs screen termination
    zeta = d["zeta_traj"]
    ax2.plot(zeta, d["dW_cum_th_sin_norm"], "k-", lw=2.0, label="Analytical $\\Delta W_{\\sin}(z) / (e\\sqrt{Z_0 P})$")
    ax2.plot(zeta[::3], d["dW_cum_unbounded_sin_norm"][::3], "o", color="#D55E00", ms=4.8, label="Wakis 3D FIT cumulative integral")
    ax2.axhline(0.0, color="#009E73", ls="--", lw=1.6, label="Lawson–Woodward limit $\\Delta W(+\\infty) = 0$")
    ax2.axvline(0.0, color="#0072B2", ls=":", lw=1.5, label="Waist peak ($z = 0$: $\\Delta W = 0.720\\,e\\sqrt{Z_0 P}$)")
    ax2.set_xlabel("Normalized axial coordinate $z / Z_R$ [-]")
    ax2.set_ylabel("Normalized cumulative gain $\\Delta W(z) / (e\\sqrt{Z_0 P})$ [-]")
    ax2.set_title("(c) Lawson–Woodward cancellation over unbounded path", fontsize=10.5, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    out_path = FIG_DIR / "03_part_b_paraxial_and_lawson_woodward.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_part_c_screen_and_plettner_figure() -> Path:
    """Render Figure 4 (Part C): 3D FIT PEC screen reflection map, screen position & mode optimization, and polarization/experiment comparison."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_part_b_c_fit_screen.npz")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.6, 4.8))

    # (a) 3D FIT Ex(x, 0, z) with conformal PEC screen at z_s = 0
    x_um = d["x_edge_m"] * 1e6
    z_um = d["z_ex_m"] * 1e6
    w0_um = float(d["w0_m"]) * 1e6
    zr_um = float(d["z_r_m"]) * 1e6
    theta_c = float(d["s_opt"]) * w0_um / zr_um
    ex_scr = d["Ex_screen_xz"] * 1e-9
    vmax = float(np.percentile(np.abs(ex_scr), 99.2))
    im0 = ax0.pcolormesh(
        z_um,
        x_um,
        ex_scr,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    fig.colorbar(im0, ax=ax0, label="$E_x(x, 0, z)$ [GV/m]")
    ax0.axvline(0.0, color="#D55E00", lw=2.8, label="Au/Kapton PEC screen ($z_s=0$)")
    ax0.plot(z_um, -theta_c * z_um, color="#009E73", lw=2.0, ls="--", label="Electron trajectory")
    ax0.text(
        z_um.min() + 0.10,
        x_um.max() - 0.55,
        "Upstream: Vacuum ($\\varepsilon_r=1$)\n$z_s=0$: PEC foil ($E_{\\parallel}=0$)\n$z>0$: Screened ($\\mathbf{E}=0$)",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#bbbbbb", alpha=0.92),
    )
    ax0.set_xlabel("Axial coordinate $z$ [$\\mu$m]")
    ax0.set_ylabel("Transverse coordinate $x$ [$\\mu$m]")
    ax0.set_title("(a) 3D FIT field terminated by PEC screen ($z_s = 0$)", fontsize=10.5, fontweight="bold")
    ax0.legend(loc="lower left", fontsize=8.5)

    # (b) Energy gain vs screen position z_s / Z_R and TEM00 vs TEM10 mode optimization (Huang et al. 2004)
    ax1.plot(
        d["zeta_dense"],
        d["dW_tem10_dense_norm"],
        color="#0072B2",
        ls="--",
        lw=1.9,
        label="On-axis $\\mathrm{TEM}_{10}$ theory ($1/\\sqrt{1+(z_s/Z_R)^2}$)",
    )
    ax1.plot(
        d["zeta_dense"],
        d["dW_tem00_dense_norm"],
        "k-",
        lw=2.1,
        label="Crossed $\\mathrm{TEM}_{00}$ theory ($s_{\\mathrm{opt}}=1.12$)",
    )
    ax1.plot(
        d["zeta_traj"][::4],
        d["dW_cum_unbounded_norm"][::4],
        "s",
        color="#009E73",
        ms=4.5,
        label="Wakis 3D FIT semi-infinite integral",
    )
    ax1.plot(
        d["zs_explicit_zetas"],
        d["dW_explicit_fit_norm"],
        "o",
        color="#D55E00",
        ms=7.0,
        label="Wakis 3D FIT explicit PEC screen runs",
    )
    ax1.set_xlabel("Normalized screen position $z_s / Z_R$ [-]")
    ax1.set_ylabel("Peak gain $\\Delta W_{\\max} / (e\\sqrt{Z_0 P})$ [-]")
    ax1.set_title("(b) Gain vs screen position & mode (Huang et al. 2004)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    # (c) Polarization angle cos(phi_pol) law & crossing angle optimization (Plettner et al. 2005 Fig. 5b)
    phi_deg = d["phi_pol_deg"]
    ax2.plot(
        phi_deg,
        d["dW_pol_th_norm"],
        "k-",
        lw=2.0,
        label="Theory $\\Delta W(\\phi_{\\mathrm{pol}}) \\propto \\cos\\phi_{\\mathrm{pol}}$",
    )
    ax2.plot(
        phi_deg,
        d["dW_pol_fit_norm"],
        "o",
        color="#D55E00",
        ms=5.8,
        label="Wakis 3D FIT ($z_s = 0$ PEC screen)",
    )
    ax2.axhline(0.0, color="#0072B2", ls=":", lw=1.4, label="No screen ($\\Delta W = 0$, Plettner Fig. 3)")
    ax2.axvline(90.0, color="grey", ls="--", lw=1.1)
    ax2.set_xlabel("Laser polarization angle $\\phi_{\\mathrm{pol}}$ [deg]")
    ax2.set_ylabel("Signed gain $\\Delta W / (e\\sqrt{Z_0 P})$ [-]")
    ax2.set_title("(c) Polarization $\\cos\\phi_{\\mathrm{pol}}$ law (Plettner et al. 2005)", fontsize=10.5, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    out_path = FIG_DIR / "04_part_c_screen_polarization_and_angle.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_part_d_ponderomotive_scaling_figure() -> Path:
    """Render Figure 5 (Part D): Zolotorev (2002) Fig. 1 (gamma vs Z_R/Z_{R,opt}), Fig. 2 (gamma vs z/Z_R), and Fig. 3 (crater profile)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_part_d_ponderomotive.npz")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # (a) Reproduction of Fig. 1 of Zolotorev (2002): gamma / gamma_opt vs Z_R / Z_{R,opt}
    ax0.plot(
        d["kappa_vals"],
        d["gamma_over_peak"],
        color="#0072B2",
        lw=2.2,
        label="Relativistic ponderomotive ODE ($\\gamma / \\gamma_{\\mathrm{peak}}$)",
    )
    ax0.plot(
        d["kappa_vals"],
        d["gamma_over_gamma_opt"],
        color="#D55E00",
        ls="--",
        lw=1.9,
        label="Unnormalized $\\gamma / (1 + a_0^2/2)$",
    )
    # Approximate values read from Fig. 1 of Zolotorev (2002) p. 447
    fig1_x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    fig1_y = np.array([0.80, 1.00, 0.78, 0.40, 0.12, 0.04])
    ax0.plot(fig1_x, fig1_y, "o", color="k", ms=6.0, label="Zolotorev (2002) Fig. 1 (approx.)")
    ax0.axvline(1.0, color="#009E73", ls=":", lw=1.6, label="Optimum $Z_R = Z_{R,\\mathrm{opt}} = a_0^2 c\\tau/2$")
    ax0.set_xlim(0.0, 3.2)
    ax0.set_ylim(0.0, 1.08)
    ax0.set_xlabel("Normalized Rayleigh length $Z_R / Z_{R,\\mathrm{opt}}$ [-]")
    ax0.set_ylabel("Normalized final energy $\\gamma / \\gamma_{\\mathrm{opt}}$ [-]")
    ax0.set_title("(a) Energy vs $Z_R / Z_{R,\\mathrm{opt}}$ (Zolotorev 2002, Fig. 1)", fontsize=10.5, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8.5)

    # (b) Reproduction of Fig. 2 of Zolotorev (2002): gamma(z) vs z / Z_R
    z_zr = d["z_over_zr_traj"]
    gam_z = d["gamma_z_traj"]
    mask = (z_zr >= 0.0) & (z_zr <= 6.5)
    ax1.plot(z_zr[mask], gam_z[mask], color="#D55E00", lw=2.3, label="Relativistic ponderomotive ODE ($a_0=7.4$)")
    fig2_x = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
    fig2_y = np.array([1.0, 16.5, 22.0, 25.5, 26.8, 27.4, 27.8])
    ax1.plot(fig2_x, fig2_y, "s", color="k", ms=5.8, label="Zolotorev (2002) Fig. 2 (approx.)")
    ax1.axhline(float(d["gamma_opt"]), color="#0072B2", ls="--", lw=1.5, label=f"$\\gamma_{{\\mathrm{{opt}}}} = 1 + a_0^2/2 = {float(d['gamma_opt']):.1f}$")
    ax1.set_xlim(0.0, 6.5)
    ax1.set_ylim(0.0, 31.0)
    ax1.set_xlabel("Normalized axial distance $z / Z_R$ [-]")
    ax1.set_ylabel("Electron Lorentz factor $\\gamma(z)$ [-]")
    ax1.set_title("(b) Axial acceleration $\\gamma(z)$ (Zolotorev 2002, Fig. 2)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.5)

    # (c) Reproduction of Fig. 3 of Zolotorev (2002): two-mode crater transverse profile
    r_sig = d["r_over_sigma"]
    r_rim = float(d["r_rim_over_sigma"])
    ax2.axvspan(0.0, r_rim, color="#009E73", alpha=0.14, label=f"Focusing crater ($r < r_{{\\mathrm{{rim}}}}= {r_rim:.2f}\\sigma$)")
    ax2.plot(r_sig, d["i_tot_norm"], "k-", lw=2.3, label="Sum $I_{\\mathrm{tot}}(r) = I_1(r) + I_2(r)$ (crater)")
    ax2.plot(r_sig, d["i1_norm"], color="#0072B2", ls="--", lw=1.9, label="Mode 1: $I_1/I_0 = e^{-r^2/(2\\sigma^2)}$")
    ax2.plot(r_sig, d["i2_norm"], color="#D55E00", ls="-.", lw=1.9, label="Mode 2: $I_2/I_0 = \\alpha (r^2/2\\sigma^2)e^{-r^2/(2\\sigma^2)}$")
    ax2.axvline(r_rim, color="#009E73", ls=":", lw=1.6)
    ax2.set_xlim(0.0, 3.5)
    ax2.set_ylim(0.0, 1.15)
    ax2.set_xlabel("Normalized radius $r / \\sigma$ [-]")
    ax2.set_ylabel("Normalized intensity $I(r) / I_0$ [-]")
    ax2.set_title("(c) Crater potential (Zolotorev 2002, Fig. 3)", fontsize=10.5, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    out_path = FIG_DIR / "05_part_d_ponderomotive_scaling_and_crater.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_part_d_phase_space_figure() -> Path:
    """Render Figure 6 (Part D): Zolotorev (2002) Fig. 4 (transverse phase space), Fig. 5 (longitudinal ~50 as phase space), and blowout comparison."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_part_d_ponderomotive.npz")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # (a) Transverse phase space (gamma * beta_x vs x) in crater trap (Fig. 4 of Zolotorev 2002)
    ax0.scatter(
        d["x_crater_um"],
        d["ux_crater"],
        s=14,
        color="#0072B2",
        alpha=0.72,
        edgecolors="none",
        label=f"Crater mode ($\\varepsilon_{{\\mathrm{{inv}},x}}\\approx {float(d['eps_inv_x_nm']):.1f}$ nm$\\cdot$rad)",
    )
    ax0.axhline(0.0, color="grey", lw=0.8, alpha=0.6)
    ax0.axvline(0.0, color="grey", lw=0.8, alpha=0.6)
    ax0.set_xlim(-0.25, 0.25)
    ax0.set_ylim(-0.055, 0.055)
    ax0.set_xlabel("Transverse position $x$ [$\\mu$m]")
    ax0.set_ylabel("Normalized transverse momentum $\\gamma \\beta_x$ [-]")
    ax0.set_title("(a) Transverse phase space (Zolotorev 2002, Fig. 4)", fontsize=10.5, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8.5)

    # (b) Longitudinal phase space (Delta gamma / gamma vs z - <z>) (Fig. 5 of Zolotorev 2002)
    sig_t_as = float(d["sigma_t_as"])
    sig_z_nm = float(d["sigma_z_nm"])
    ax1.scatter(
        d["dz_crater_um"],
        d["dgg_crater"],
        s=14,
        color="#D55E00",
        alpha=0.72,
        edgecolors="none",
        label=f"Attosecond bunch ($\\sigma_t={sig_t_as:.1f}$ as, $\\sigma_z={sig_z_nm:.1f}$ nm)",
    )
    ax1.axhline(0.0, color="grey", lw=0.8, alpha=0.6)
    ax1.axvline(0.0, color="grey", lw=0.8, alpha=0.6)
    ax1.set_xlim(-0.05, 0.05)
    ax1.set_ylim(-0.025, 0.025)
    ax1.set_xlabel("Relative axial coordinate $z - \\bar{z}$ [$\\mu$m]")
    ax1.set_ylabel("Relative energy spread $\\Delta\\gamma / \\gamma$ [-]")
    ax1.set_title("(b) Longitudinal phase space (Zolotorev 2002, Fig. 5)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    # (c) Single Gaussian transverse blowout instability vs two-mode crater stabilization
    frames_z_g = np.mean(np.abs(d["frames_z_gaussian"]), axis=1)
    frames_r_g = np.sqrt(np.mean(d["frames_x_gaussian_um"] ** 2, axis=1))
    frames_z_c = np.mean(np.abs(d["frames_z_crater"]), axis=1)
    frames_r_c = np.sqrt(np.mean(d["frames_x_crater_um"] ** 2, axis=1))
    ax2.semilogy(
        frames_z_g,
        np.maximum(frames_r_g, 1e-3),
        "s--",
        color="#D55E00",
        lw=2.0,
        ms=5.2,
        label="Single Gaussian ($\\alpha=0$: radial blowout)",
    )
    ax2.semilogy(
        frames_z_c,
        np.maximum(frames_r_c, 1e-3),
        "o-",
        color="#009E73",
        lw=2.0,
        ms=5.2,
        label="Two-mode crater ($\\alpha=1.25$: focused)",
    )
    ax2.set_xlabel("Normalized axial distance $\\langle z \\rangle / Z_R$ [-]")
    ax2.set_ylabel("RMS transverse bunch radius $\\sigma_x$ [$\\mu$m]")
    ax2.set_title("(c) Transverse blowout vs crater stabilization", fontsize=10.5, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="center right", fontsize=8.5)

    fig.tight_layout()
    out_path = FIG_DIR / "06_part_d_attosecond_phase_space.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_part_e_wiggler_brightness_figure() -> Path:
    """Render Figure 7 (Part E): X-ray resonance vs electron energy, required E_e vs wiggler period, and FEL gain vs Fermi brightness B_F."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_part_e_wiggler_fel.npz")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # (a) X-ray photon energy vs electron energy for optical laser wigglers
    e_mev = d["electron_energy_mev"]
    ax0.plot(e_mev, d["photon_kev_0p4um"], color="#0072B2", lw=2.1, label="Laser wiggler $\\lambda_L = 0.4\\;\\mu$m ($a=1$)")
    ax0.plot(e_mev, d["photon_kev_0p8um"], color="#009E73", ls="--", lw=2.1, label="Laser wiggler $\\lambda_L = 0.8\\;\\mu$m ($a=1$)")
    ax0.plot(e_mev, d["photon_kev_1p0um"], color="#D55E00", ls="-.", lw=2.1, label="Laser wiggler $\\lambda_L = 1.0\\;\\mu$m ($a=1$)")
    ax0.axhline(10.0, color="k", ls=":", lw=1.5, label="Target $10$ keV ($\\lambda_x = 1.24\\;\\mathrm{\\AA}$)")
    ax0.axvspan(15.0, 28.0, color="#009E73", alpha=0.12, label="Zolotorev (2002) $15$–$28$ MeV window")
    ax0.set_xlim(15.0, 42.0)
    ax0.set_ylim(0.0, 28.0)
    ax0.set_xlabel("Electron beam energy $E_e$ [MeV]")
    ax0.set_ylabel("Resonant X-ray energy $\\hbar\\omega_x$ [keV]")
    ax0.set_title("(a) Head-on laser wiggler $\\lambda_x = \\lambda_L(1+a^2/2)/(4\\gamma^2)$", fontsize=10.5, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=8.5)

    # (b) Required electron energy for 10 keV X-rays: optical laser wiggler vs cm static undulator
    per_um = d["periods_m"] * 1e6
    ax1.loglog(per_um, d["energy_req_laser_mev"], color="#0072B2", lw=2.1, label="Head-on laser wiggler ($\\lambda_W = \\lambda_L/2$)")
    ax1.loglog(per_um, d["energy_req_static_mev"], color="#D55E00", ls="--", lw=2.0, label="Static magnetic undulator ($\\lambda_u$)")
    ax1.plot([0.4, 1.0], [17.7, 28.0], "o", color="#0072B2", ms=7.0, label="Optical wiggler ($15$–$28$ MeV)")
    ax1.plot([3.0e4], [1.37e4], "s", color="#D55E00", ms=7.0, label="Static undulator ($\\sim 15$ GeV)")
    ax1.set_xlabel("Wiggler wavelength $\\lambda_L$ or period $\\lambda_u$ [$\\mu$m]")
    ax1.set_ylabel("Required electron energy $E_e$ for $10$ keV [MeV]")
    ax1.set_title("(b) $10^3\\times$ reduction in electron energy at $10$ keV", fontsize=10.5, fontweight="bold")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8.5)

    # (c) FEL gain periods M_A and wiggler pulse energy vs dimensionless Fermi brightness B_F
    bf = d["b_f_scan"]
    ma = d["m_a_periods_scan"]
    ax2.loglog(bf, ma, color="#0072B2", lw=2.2, label="Wiggler periods per gain length $M_A \\propto B_F^{-1}$")
    ax2.plot(
        float(d["b_f_zolotorev"]),
        float(d["m_a_zolotorev"]),
        "o",
        color="#009E73",
        ms=8.0,
        label=f"Attosecond laser bunch ($B_F\\approx 10^{{-6}},\\;M_A\\approx {float(d['m_a_zolotorev']):.0f}$)",
    )
    ax2.plot(
        float(d["b_f_conventional"]),
        float(d["m_a_conventional"]),
        "s",
        color="#D55E00",
        ms=7.5,
        label="Conventional RF gun ($B_F\\approx 10^{-12},\\;M_A\\sim 10^8$)",
    )
    ax2.set_xlabel("Dimensionless Fermi brightness $B_F = N\\bar{\\lambda}_c^3 / (\\varepsilon_x\\varepsilon_y\\varepsilon_z)$ [-]")
    ax2.set_ylabel("Wiggler periods per FEL gain length $M_A$ [-]")
    ax2.set_title("(c) FEL gain length vs Fermi brightness $B_F$", fontsize=10.5, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    out_path = FIG_DIR / "07_part_e_laser_wiggler_and_brightness.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_convergence_figure() -> Path:
    """Render Figure 8: 3-level 3D FIT grid resolution convergence up to 20 cells/lambda."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()
    d = np.load(DATA_DIR / "task_convergence.npz")

    ppw = d["ppw_z"]
    ex_err = d["ex_waist_l2_err"] * 100.0
    ez_err = d["ez_waist_l2_err"] * 100.0
    dw_err = d["dw_screen_rel_err"] * 100.0
    zeta_ref = d["zeta_ref"]
    dw_curves = d["dw_cum_curves"]
    dw_th = d["dw_cum_theory_sin"]
    names = [str(s) for s in d["level_names"]]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 4.8))

    ax0.loglog(ppw, ex_err, "o-", color="#0072B2", lw=2.0, ms=6.8, label="Waist $|E_x(x,0,0)|$ relative $L_2$ error")
    ax0.loglog(ppw, ez_err, "s-", color="#D55E00", lw=2.0, ms=6.5, label="Waist $|E_z(x,0,0)|$ relative $L_2$ error")
    ax0.loglog(ppw, np.maximum(dw_err, 0.05), "^-", color="#009E73", lw=2.0, ms=6.8, label="PEC screen gain $\\Delta W(z_s=0)$ error")
    ref_line = ex_err[0] * ((ppw[0] / ppw) ** 2)
    ax0.loglog(ppw, ref_line, "k--", lw=1.4, label="$\\mathcal{O}(\\Delta z^2)$ second-order reference")
    ax0.axvline(20.0, color="#333333", ls=":", lw=1.4, label="$20\\;\\mathrm{cells}/\\lambda_L$ rule")
    ax0.set_xticks([12, 15, 16, 18, 20])
    ax0.get_xaxis().set_major_formatter(ScalarFormatter())
    ax0.get_xaxis().set_minor_formatter(NullFormatter())
    ax0.set_xlabel("Axial resolution $\\lambda_L / \\Delta z$ [cells/$\\lambda_L$]")
    ax0.set_ylabel("Relative error [%]")
    ax0.set_title("(a) 3D FIT convergence of waist fields and $\\Delta W(z_s=0)$", fontsize=10.5, fontweight="bold")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(loc="lower left", fontsize=8.5)

    colors = ["#E69F00", "#0072B2", "#D55E00"]
    markers = ["^", "s", "o"]
    ax1.plot(zeta_ref, dw_th, "k-", lw=2.1, label="Analytical paraxial total derivative")
    for i, name in enumerate(names):
        ax1.plot(
            zeta_ref[::4],
            dw_curves[i][::4],
            marker=markers[i % len(markers)],
            linestyle="none",
            color=colors[i % len(colors)],
            ms=5.2,
            label=f"Wakis 3D FIT ({name}, $\\lambda_L/\\Delta z = {ppw[i]:.0f}$)",
        )
    ax1.axhline(0.0, color="grey", ls=":", lw=1.1)
    ax1.set_xlabel("Normalized axial coordinate $z / Z_R$ [-]")
    ax1.set_ylabel("Cumulative gain $\\Delta W_{\\sin}(z) / (e\\sqrt{Z_0 P})$ [-]")
    ax1.set_title("(b) Cumulative energy curve across 3D FIT resolutions", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    out_path = FIG_DIR / "08_fit_resolution_convergence.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_animations() -> list[Path]:
    """Render the 3 required animated GIFs (< 5 MB each) with explicit material/boundary labels."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_rc()

    d_a = np.load(DATA_DIR / "task_part_a_plane_wave.npz")
    d_bc = np.load(DATA_DIR / "task_part_b_c_fit_screen.npz")
    d_d = np.load(DATA_DIR / "task_part_d_ponderomotive.npz")

    # --- GIF 1: Electron in a plane wave (figure-of-eight + gamma(t) trace returning to 1) ---
    gif1_path = FIG_DIR / "plane_wave_figure_of_eight.gif"
    t_cyc = d_a["t_over_topt"]
    g_ex = d_a["gamma_exact"]
    x_um = d_a["x_boris_m"] * 1e6
    z_um = d_a["z_boris_m"] * 1e6
    n_f1 = 28
    idx1 = np.linspace(0, len(t_cyc) - 1, n_f1, dtype=int)

    fig1, (ax1_l, ax1_r) = plt.subplots(1, 2, figsize=(10.2, 4.1), dpi=95)
    ax1_l.plot(z_um, x_um, color="#bbbbbb", lw=1.2, label="Full lab trajectory")
    (tr1_line,) = ax1_l.plot([], [], color="#0072B2", lw=2.2, label="Boris trajectory $(z(t), x(t))$")
    (pt1_l,) = ax1_l.plot([], [], "o", color="#D55E00", ms=8, label="Electron")
    ax1_l.text(
        0.03,
        0.95,
        "Material: Vacuum ($\\varepsilon_r=1, \\mu_r=1$)\nUnbounded plane wave ($a_0=2.5$)",
        transform=ax1_l.transAxes,
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax1_l.set_xlim(z_um.min() - 0.3, z_um.max() + 0.5)
    ax1_l.set_ylim(x_um.min() * 1.25, x_um.max() * 1.35)
    ax1_l.set_xlabel("Axial drift $z$ [$\\mu$m]")
    ax1_l.set_ylabel("Transverse quiver $x$ [$\\mu$m]")
    ax1_l.grid(True, alpha=0.3)
    ax1_l.legend(loc="lower right", fontsize=8.5)

    ax1_r.plot(t_cyc, g_ex, color="#bbbbbb", lw=1.3, label="Exact $\\gamma(t) = 1 + a_x^2/2$")
    (g1_line,) = ax1_r.plot([], [], color="#D55E00", lw=2.2, label="Active $\\gamma(t)$")
    (pt1_r,) = ax1_r.plot([], [], "o", color="#D55E00", ms=8)
    ax1_r.axhline(1.0, color="#009E73", ls="--", lw=1.4, label="Post-pulse $\\gamma=1$ ($\\Delta W=0$)")
    ax1_r.set_xlim(t_cyc.min(), t_cyc.max())
    ax1_r.set_ylim(0.85, float(np.max(g_ex)) * 1.12)
    ax1_r.set_xlabel("Laboratory time $t / T_0$ [cycles]")
    ax1_r.set_ylabel("Lorentz factor $\\gamma(t)$ [-]")
    ax1_r.grid(True, alpha=0.3)
    ax1_r.legend(loc="upper right", fontsize=8.5)

    def _update1(frame_i: int):
        k = idx1[frame_i]
        tr1_line.set_data(z_um[: k + 1], x_um[: k + 1])
        pt1_l.set_data([z_um[k]], [x_um[k]])
        g1_line.set_data(t_cyc[: k + 1], g_ex[: k + 1])
        pt1_r.set_data([t_cyc[k]], [g_ex[k]])
        ax1_l.set_title(f"(a) Lab-frame quiver & drift ($t/T_0={t_cyc[k]:+5.1f}$)", fontsize=9.8, fontweight="bold")
        ax1_r.set_title(f"(b) Instantaneous energy $\\gamma={g_ex[k]:.2f}\\to 1.00$", fontsize=9.8, fontweight="bold")
        return tr1_line, pt1_l, g1_line, pt1_r

    fig1.tight_layout()
    ani1 = animation.FuncAnimation(fig1, _update1, frames=n_f1, interval=95, blit=False)
    ani1.save(gif1_path, writer=animation.PillowWriter(fps=10))
    plt.close(fig1)

    # --- GIF 2: 3D FIT focused beam reflecting from the PEC screen with synchronous electron ---
    gif2_path = FIG_DIR / "fit_beam_screen_reflection.gif"
    x_e_um = d_bc["x_edge_m"] * 1e6
    z_e_um = d_bc["z_ex_m"] * 1e6
    frames_ex = d_bc["frames_Ex_screen"] * 1e-9  # GV/m
    w0_um = float(d_bc["w0_m"]) * 1e6
    zr_um = float(d_bc["z_r_m"]) * 1e6
    theta_c = float(d_bc["s_opt"]) * w0_um / zr_um
    vmax2 = float(np.percentile(np.abs(frames_ex), 99.0))
    n_f2 = len(frames_ex)
    ze_seq = np.linspace(z_e_um.min() + 0.15, z_e_um.max() - 0.15, n_f2)

    fig2, ax2 = plt.subplots(figsize=(7.4, 4.3), dpi=98)
    im2 = ax2.pcolormesh(
        z_e_um,
        x_e_um,
        frames_ex[0],
        cmap="RdBu_r",
        vmin=-vmax2,
        vmax=vmax2,
        shading="auto",
    )
    fig2.colorbar(im2, ax=ax2, label="$E_x(x, 0, z, t)$ [GV/m]")
    ax2.axvline(0.0, color="#D55E00", lw=2.8, label="Au/Kapton PEC screen ($z_s=0$)")
    ax2.plot(z_e_um, -theta_c * z_e_um, color="#009E73", ls="--", lw=1.9, label="Electron trajectory")
    (pt2_e,) = ax2.plot([], [], "*", color="#F0E442", markeredgecolor="k", ms=12, label="Relativistic $e^-$ ($v\\approx c$)")
    ax2.text(
        z_e_um.min() + 0.08,
        x_e_um.max() - 0.55,
        "Vacuum ($\\varepsilon_r=1$) | 6-cell CPML collar\nPEC screen at $z_s=0$ blocks $z>0$ ($\\Delta W \\neq 0$)",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    ax2.set_xlim(z_e_um.min(), z_e_um.max())
    ax2.set_ylim(x_e_um.min(), x_e_um.max())
    ax2.set_xlabel("Axial coordinate $z$ [$\\mu$m]")
    ax2.set_ylabel("Transverse coordinate $x$ [$\\mu$m]")
    ax2.legend(loc="lower left", fontsize=8.5)

    def _update2(frame_i: int):
        im2.set_array(frames_ex[frame_i].ravel())
        ze_now = ze_seq[frame_i]
        xe_now = -theta_c * ze_now
        pt2_e.set_data([ze_now], [xe_now])
        ax2.set_title(
            f"3D FIT beam + PEC screen at $z_s=0$ ($z_e={ze_now:+4.2f}\\;\\mu$m)",
            fontsize=10.0,
            fontweight="bold",
        )
        return im2, pt2_e

    fig2.tight_layout()
    ani2 = animation.FuncAnimation(fig2, _update2, frames=n_f2, interval=95, blit=False)
    ani2.save(gif2_path, writer=animation.PillowWriter(fps=10))
    plt.close(fig2)

    # --- GIF 3: Focused pulse sweeping over test particles with crater potential vs single Gaussian ---
    gif3_path = FIG_DIR / "ponderomotive_crater_bunching.gif"
    fx_c = d_d["frames_x_crater_um"]
    fz_c = d_d["frames_z_crater"]
    fg_c = d_d["frames_gamma_crater"]
    fx_g = d_d["frames_x_gaussian_um"]
    fz_g = d_d["frames_z_gaussian"]
    n_f3 = len(fx_c)

    fig3, (ax3_l, ax3_r) = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=95)
    (sc_g,) = ax3_l.plot([], [], "s", color="#D55E00", ms=4.5, alpha=0.65, label="Single Gaussian (radial blowout)")
    (sc_c,) = ax3_l.plot([], [], "o", color="#0072B2", ms=4.8, alpha=0.85, label="Two-mode crater (focused)")
    ax3_l.axhline(+0.22, color="#009E73", ls=":", lw=1.4, label="Crater rim $|x|\\lesssim 0.2\\;\\mu$m")
    ax3_l.axhline(-0.22, color="#009E73", ls=":", lw=1.4)
    ax3_l.text(
        0.03,
        0.95,
        "Material: Vacuum ($\\varepsilon_r=1, \\mu_r=1$)\nOpen diffracting focus ($Z_{R,\\mathrm{opt}}=a_0^2 c\\tau/2$)",
        transform=ax3_l.transAxes,
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax3_l.set_xlim(-0.3, 6.4)
    ax3_l.set_ylim(-2.5, 2.5)
    ax3_l.set_xlabel("Normalized axial position $z / Z_R$ [-]")
    ax3_l.set_ylabel("Transverse position $x$ [$\\mu$m]")
    ax3_l.grid(True, alpha=0.3)
    ax3_l.legend(loc="lower left", fontsize=8.5)

    ax3_r.plot(d_d["z_over_zr_traj"], d_d["gamma_z_traj"], color="#bbbbbb", lw=1.5, label="On-axis theory $\\gamma(z)$")
    (sc_gam,) = ax3_r.plot([], [], "o", color="#009E73", ms=4.8, alpha=0.8, label="Crater bunch $\\gamma_i(z_i)$")
    ax3_r.axhline(float(d_d["gamma_opt"]), color="#0072B2", ls="--", lw=1.4, label=f"$\\gamma_{{\\mathrm{{opt}}}}=1+a_0^2/2={float(d_d['gamma_opt']):.1f}$")
    ax3_r.set_xlim(-0.2, 6.4)
    ax3_r.set_ylim(0.0, 31.5)
    ax3_r.set_xlabel("Normalized axial position $z / Z_R$ [-]")
    ax3_r.set_ylabel("Lorentz factor $\\gamma$ [-]")
    ax3_r.grid(True, alpha=0.3)
    ax3_r.legend(loc="lower right", fontsize=8.5)

    def _update3(frame_i: int):
        sc_g.set_data(fz_g[frame_i], np.clip(fx_g[frame_i], -2.4, 2.4))
        sc_c.set_data(fz_c[frame_i], fx_c[frame_i])
        sc_gam.set_data(fz_c[frame_i], fg_c[frame_i])
        z_mean = float(np.mean(fz_c[frame_i]))
        g_mean = float(np.mean(fg_c[frame_i]))
        ax3_l.set_title(f"(a) Crater trapping vs blowout ($\\langle z\\rangle/Z_R={z_mean:4.2f}$)", fontsize=9.8, fontweight="bold")
        ax3_r.set_title(f"(b) Ponderomotive gain ($\\langle\\gamma\\rangle={g_mean:5.2f}$)", fontsize=9.8, fontweight="bold")
        return sc_g, sc_c, sc_gam

    fig3.tight_layout()
    ani3 = animation.FuncAnimation(fig3, _update3, frames=n_f3, interval=95, blit=False)
    ani3.save(gif3_path, writer=animation.PillowWriter(fps=10))
    plt.close(fig3)

    return [gif1_path, gif2_path, gif3_path]


def render_all_figures() -> list[Path]:
    """Render all 8 static PNG figures and 3 animated GIFs from saved data/."""
    paths = [
        render_setup_and_mesh_figure(),
        render_part_a_plane_wave_figure(),
        render_part_b_paraxial_and_lw_figure(),
        render_part_c_screen_and_plettner_figure(),
        render_part_d_ponderomotive_scaling_figure(),
        render_part_d_phase_space_figure(),
        render_part_e_wiggler_brightness_figure(),
        render_convergence_figure(),
    ]
    paths.extend(render_animations())
    return paths


# ---------------------------------------------------------------------------
# Task registry and CLI runner
# ---------------------------------------------------------------------------

TASKS: dict[str, tuple[Callable[[bool], dict], str]] = {
    "part_a_plane_wave": (
        run_task_part_a_plane_wave,
        "Part A: Relativistic plane-wave figure-of-eight orbit, zero net gain, and Boris/Vay O(dt^2) convergence",
    ),
    "part_b_c_fit_screen": (
        run_task_part_b_c_fit_screen,
        "Parts B & C: 3D Wakis FIT focused Gaussian beam, Lawson-Woodward cancellation, and conformal PEC screen scans",
    ),
    "part_d_ponderomotive": (
        run_task_part_d_ponderomotive,
        "Part D: Relativistic ponderomotive acceleration from rest, crater focusing, and ~50 as bunching (Zolotorev 2002)",
    ),
    "part_e_wiggler_fel": (
        run_task_part_e_wiggler_fel,
        "Part E: Laser as electromagnetic wiggler (10 keV X-rays) and Fermi brightness B_F gain-length scaling",
    ),
    "convergence": (
        run_task_convergence,
        "3-level 3D FIT grid resolution convergence study up to 20 cells/lambda",
    ),
}


def run_tasks(
    task_names: list[str] | None = None,
    quick: bool = False,
    do_plot: bool = True,
) -> dict[str, dict]:
    """Execute selected tutorial tasks and optionally render all figures and GIFs."""
    names = list(TASKS.keys()) if not task_names else list(task_names)
    results: dict[str, dict] = {}
    for name in names:
        if name not in TASKS:
            raise ValueError(f"Unknown task {name!r}; valid tasks: {list(TASKS.keys())}")
        fn, _ = TASKS[name]
        results[name] = fn(quick=quick)

    if do_plot and all((DATA_DIR / f"task_{k}.npz").exists() for k in TASKS):
        render_all_figures()
    return results
