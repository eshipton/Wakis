# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Shared setup, analytical dispersion, error budget, and runners for V7 space-charge waves.

Used by both ``tests/test_087_particles_vs_linearized_beam.py`` and
``examples/024_particles_space_charge_waves.py`` to avoid duplicating the planar
sheet sampler, depressed-equilibrium dispersion parameters, term-by-term error
budget, standing-wave null/envelope fitter, spatial wavenumber spectrum, and
time-domain drift runners.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import epsilon_0, m_e
from scipy.optimize import brentq, minimize
from scipy.stats import t as student_t

from wakis import GridFIT3D, SolverFIT3D
from wakis.beam_fluid import (
    LinearizedBeam,
    reduction_factor_planar,
    relativistic_gamma,
)
from wakis.particles import BeamInjector, ParticleSolver, Species

try:
    import cupy as cp

    USE_GPU = bool(cp.cuda.is_available())
except ImportError:
    USE_GPU = False


# Physical and geometric parameters for the planar sheet-beam drift tunnel
V0_VOLTS = 10.0e3  # DC beam acceleration voltage [V]
TUNNEL_HALF_WIDTH_M = 2.0e-3  # Parallel-plate tunnel half-width a [m]
BEAM_HALF_THICKNESS_M = 0.8e-3  # Sheet-beam half-thickness b [m]
PERIODIC_LY_M = 1.0e-3  # Periodic transverse depth Ly [m]
TUNNEL_LENGTH_M = 0.186  # Drift-tunnel axial length Lz >= 1.6 * lambda_q [m]
OUTFLOW_RAMP_M = 8.0e-3  # Outflow axial taper length [m]
RISE_PERIODS = 2.0  # C^1 raised-cosine entrance turn-on ramp duration in RF periods [-]
DRIVE_FREQ_HZ = 6.5e9  # AC density-modulation frequency f [Hz]
PLASMA_FREQ_HZ = 0.90e9  # Unreduced injection plasma frequency omega_p / (2 * pi) [Hz]
MODULATION_INDEX = 0.015  # Fractional AC density modulation depth m [-]
GUIDE_BZ_TESLA = 50.0  # Strong axial confining magnetic field B_z [T]
CFL_NUMBER = 0.5  # Courant number cfln [-]

COARSE_GRID = (10, 1, 428)  # (Nx, Ny, Nz) coarse resolution
FINE_GRID = (20, 1, 856)  # (Nx, Ny, Nz) 2x refined resolution
NOISE_SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)  # 8 seeds for empirical noise


def fit_window(params: dict[str, float]) -> tuple[float, float]:
    """Return the shared axial fit window ``(z_min, z_max)`` [m].

    Excludes ``3 / T1`` at the entrance (evanescent higher-order cathode zone,
    where ``1 / T1`` is the transverse decay length) and ``3 / T1`` upstream of
    the outflow taper start ``TUNNEL_LENGTH_M - OUTFLOW_RAMP_M``.
    """
    margin = 3.0 / params["T1"]
    return float(margin), float((TUNNEL_LENGTH_M - OUTFLOW_RAMP_M) - margin)


def extract_coherent_bunching_i2_ratio(
    z_coords: np.ndarray,
    i1_phasor: np.ndarray,
    i2_phasor: np.ndarray,
    beta_e: float,
    beta_q: float,
    z_max: float | None = None,
) -> float:
    """Project `I2(z)` onto the coherent ballistic-bunching mode `z * sin(beta_q * z) * exp(-2i * beta_e * z)`.

    For an entrance drive ``I(0, t) = I0 * (1 + m * sin(omega * t))`` with entrance
    phasor phase ``phi1 = arg(I1(0))``, quadratic convective bunching generates a
    second-harmonic current phasor in quadrature with ``exp(2i * phi1)`` with axial
    profile ``b(z) = z * sin(beta_q * z) * exp(-2i * beta_e * z)``, whereas the
    linear cross-term between step-to-step transverse RNG jitter ``eta(t)`` and the
    ``m * sin(omega * t)`` weight modulation is in-phase with ``exp(i * phi1)``.
    Returns the maximum coherent ballistic-bunching ratio ``|I2_bunch(z)| / |I1(0)|``
    on ``z in [0, z_max]``.
    """
    if z_max is None:
        z_max = 2.0 * np.pi / beta_q
    mask = z_coords <= z_max
    zm = z_coords[mask]
    basis = zm * np.sin(beta_q * zm)
    phase_sq = (i1_phasor[0] / np.abs(i1_phasor[0])) ** 2
    demod = i2_phasor[mask] * np.exp(2j * beta_e * zm) / phase_sq
    c_quad = float(np.dot(basis, np.imag(demod)) / np.dot(basis, basis))
    return float(abs(c_quad) * np.max(np.abs(basis)) / np.abs(i1_phasor[0]))


def _compute_multimode_fit_shift(
    tau: float,
    b: float,
    a: float,
    omega_p_dep: float,
    gamma_dep: float,
    u_dep: float,
    beta_e: float,
    nx: int | None = None,
    nz: int | None = None,
) -> tuple[float, float, float, float]:
    """Compute fundamental `T1`, second-mode `T2`, and signed multi-mode shifts `eps_fit_cont`, `eps_fit_grid`.

    A uniform top-hat entrance modulation across ``x in [-b, b]`` excites the
    orthogonal Sturm-Liouville slab eigenmodes ``cos(T_n * x)`` (``n = 1, 2, ...``).
    The weak higher-order modes (``n >= 2``, total power weight ~1%) shift the
    fitted two-wave ``beta_q`` upward relative to the pure ``n = 1`` mode.
    Runs :func:`fit_beta_q_from_envelope` over the window ``[3 / T1, (Lz - L_ramp) - 3 / T1]``
    on the continuum multi-mode model ``sum_n w_n cos(beta_qn * z) * exp(-i * beta_e * z)``
    both with continuum weights ``w_n`` (``eps_fit_cont``) and with discrete
    transverse CIC mode weights on the ``nx``-cell grid (``eps_fit_grid``).
    """
    rhs = tau * b / np.tanh(tau * (a - b))
    t_modes = []
    r_modes = []
    w_cont_list = []
    w_grid_list = []
    dx = (2.0 * a / nx) if nx is not None else 0.0
    if nx is not None:
        x_nodes = np.linspace(-a, a, nx + 1)
        n_macro = 16 if nx == COARSE_GRID[0] else 32
        k_idx = np.arange(n_macro, dtype=np.float64)
        x_lat = b * (2.0 * (k_idx + 0.5) / n_macro - 1.0)
        w_cic = np.zeros(nx + 1, dtype=np.float64)
        for xp in x_lat:
            rel = np.abs(x_nodes - xp) / dx
            mask = rel < 1.0
            w_cic[mask] += (1.0 - rel[mask]) * (2.0 * b / n_macro)
    else:
        x_nodes = np.empty(0)
        w_cic = np.empty(0)

    for n in range(1, 5):
        lo = (n - 1) * np.pi + 1e-12
        hi = (n - 0.5) * np.pi - 1e-12
        tb_n = float(brentq(lambda x: x * np.tan(x) - rhs, lo, hi))
        t_n = tb_n / b
        r_n = 1.0 / np.sqrt(1.0 + (t_n / tau) ** 2)
        # Sturm-Liouville norm of phi_n(x) = cos(T_n * x) on the beam support [0, b]
        norm_in = 0.5 * b * (1.0 + np.sin(2.0 * tb_n) / (2.0 * tb_n))
        overlap_cont = np.sin(tb_n) / t_n
        w_n_cont = float((overlap_cont**2) / (b * norm_in))
        if nx is not None and (t_n * dx) < np.pi:
            overlap_grid = float(0.5 * np.sum(w_cic * np.cos(t_n * x_nodes)))
            w_n_grid = float((overlap_grid**2) / (b * norm_in))
        elif nx is None:
            w_n_grid = w_n_cont
        else:
            w_n_grid = 0.0
        t_modes.append(t_n)
        r_modes.append(r_n)
        w_cont_list.append(w_n_cont)
        w_grid_list.append(w_n_grid)

    w_cont = np.asarray(w_cont_list, dtype=np.float64)
    w_cont /= np.sum(w_cont)
    w_grid = np.asarray(w_grid_list, dtype=np.float64)
    w_grid /= np.sum(w_grid)
    bq_modes = (
        np.asarray(r_modes, dtype=np.float64) * omega_p_dep / (gamma_dep**1.5 * u_dep)
    )
    bq_1 = float(bq_modes[0])
    t1_val = float(t_modes[0])
    t2_val = float(t_modes[1]) if len(t_modes) > 1 else t1_val
    margin = 3.0 / t1_val
    z_min_fit = float(margin)
    z_max_fit = float((TUNNEL_LENGTH_M - OUTFLOW_RAMP_M) - margin)

    if nz is not None:
        dz = TUNNEL_LENGTH_M / nz
        z_eval = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    else:
        dz = TUNNEL_LENGTH_M / FINE_GRID[2]
        z_eval = (np.arange(FINE_GRID[2], dtype=np.float64) + 0.5) * dz

    carrier = np.exp(-1j * beta_e * z_eval)
    cos_modes = np.cos(np.outer(bq_modes, z_eval))
    i1_mm_cont = np.sum(w_cont[:, None] * cos_modes, axis=0) * carrier
    fit_cont = fit_beta_q_from_envelope(
        z_eval, i1_mm_cont, bq_1, z_min=z_min_fit, z_max=z_max_fit
    )
    eps_fit_cont = float((fit_cont["beta_q"] - bq_1) / bq_1)

    if nx is None:
        eps_fit_grid = eps_fit_cont
    else:
        i1_mm_grid = np.sum(w_grid[:, None] * cos_modes, axis=0) * carrier
        fit_grid = fit_beta_q_from_envelope(
            z_eval, i1_mm_grid, bq_1, z_min=z_min_fit, z_max=z_max_fit
        )
        eps_fit_grid = float((fit_grid["beta_q"] - bq_1) / bq_1)
    return t1_val, t2_val, eps_fit_cont, eps_fit_grid


def r_of_tau(
    tau_val: float, u_dep: float, gamma_dep: float, b: float, a: float
) -> float:
    """Evaluate `reduction_factor_planar` at transverse decay wavenumber ``tau_val`` [rad/m]."""
    omega_eff = float(tau_val * gamma_dep * u_dep)
    return float(reduction_factor_planar(omega_eff, u_dep, b, a=a))


def compute_analytic_parameters(qm_scale: float = 1.0) -> dict[str, float]:
    """Compute continuum space-charge-wave parameters at the self-consistent DC-depressed equilibrium.

    In an unneutralized planar electron sheet ``x in [-b, b]`` between grounded
    parallel plates at ``x = +-a``, the steady DC space-charge electric field
    ``E_x,DC(x)`` creates a potential trough ``phi_DC(x) < 0`` that slows the
    injected beam from ``u0`` to the cross-section-averaged equilibrium velocity
    ``u_dep = u0 * (1 - delta_u_dc)`` and increases the steady charge density by
    continuity to ``rho_dep = rho0 * (u0 / u_dep)``. Starting from::

        delta0 = (omega_p0**2 * b * (a - (2 / 3) * b)) / (gamma0**3 * u0**2)

    the self-consistent fractional velocity depression is iterated to convergence::

        delta <- delta0 * (1 - delta)**(-3) * (gamma0 / gamma(u0 * (1 - delta)))**3

    Parameters
    ----------
    qm_scale : float, optional
        Multiplier on the species charge-to-mass ratio ``|q / m|`` (``1.0`` for
        physical electrons, ``2.0`` for the particle species mass mutation).

    Returns
    -------
    dict of str to float
        Dictionary of nominal and DC-depressed analytical wavenumbers,
        wavelengths, currents, transverse eigenvalues, and multi-mode fit shift
        in SI units.
    """
    u0, gamma0 = relativistic_gamma(voltage=V0_VOLTS)
    omega = 2.0 * np.pi * DRIVE_FREQ_HZ
    omega_p0 = 2.0 * np.pi * PLASMA_FREQ_HZ * np.sqrt(qm_scale)
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    ly = PERIODIC_LY_M

    rho0 = -epsilon_0 * m_e * (2.0 * np.pi * PLASMA_FREQ_HZ) ** 2 / e_charge
    area_beam = 2.0 * b * ly
    i0_dc = abs(rho0) * area_beam * u0
    i1_entrance = MODULATION_INDEX * i0_dc

    # Undepressed injection parameters
    beta_e_0 = omega / u0
    r_0 = float(reduction_factor_planar(omega, u0, b, a=a))
    beta_q_0 = r_0 * omega_p0 / (gamma0**1.5 * u0)

    # Self-consistent 1D planar DC space-charge velocity depression across [-b, b]
    delta0 = (omega_p0**2 * b * (a - (2.0 / 3.0) * b)) / (gamma0**3 * u0**2)
    delta_u_dc = float(delta0)
    for _ in range(50):
        u_try = u0 * (1.0 - delta_u_dc)
        gamma_try = 1.0 / np.sqrt(1.0 - (u_try / c_light) ** 2)
        delta_u_dc = float(
            delta0 * ((1.0 - delta_u_dc) ** -3) * ((gamma0 / gamma_try) ** 3)
        )

    u_dep = u0 * (1.0 - delta_u_dc)
    gamma_dep = float(1.0 / np.sqrt(1.0 - (u_dep / c_light) ** 2))
    rho_dep = rho0 * (u0 / u_dep)
    omega_p_dep = omega_p0 * np.sqrt(u0 / u_dep)

    beta_e = omega / u_dep
    k0 = omega / c_light
    tau = float(np.sqrt(max(beta_e**2 - k0**2, 0.0)))
    r_factor = float(reduction_factor_planar(omega, u_dep, b, a=a))
    omega_q = r_factor * omega_p_dep / (gamma_dep**1.5)
    beta_q = omega_q / u_dep

    t1_transverse, t2_transverse, eps_fit_cont, eps_fit_grid = (
        _compute_multimode_fit_shift(tau, b, a, omega_p_dep, gamma_dep, u_dep, beta_e)
    )

    # Logarithmic sensitivity d(ln R)/d(ln tau) from (T1*b)*tan(T1*b) = (tau*b)*coth(tau*(a-b))
    tb1 = t1_transverse * b
    tau_g = tau * (a - b)
    dln_t1_dln_tau = (1.0 - 2.0 * tau_g / np.sinh(2.0 * tau_g)) / (
        1.0 + 2.0 * tb1 / np.sin(2.0 * tb1)
    )
    dlnr_dlntau = float((1.0 - r_factor**2) * (1.0 - dln_t1_dln_tau))

    # Sensitivity d(ln beta_q)/d(delta_u_dc) of depressed beta_q to DC velocity depression
    beta_rel_dep = u_dep / c_light
    dlnbq_ddelta = float((1.5 + dlnr_dlntau) * (1.0 + (beta_rel_dep * gamma_dep) ** 2))

    return {
        "u0": float(u0),
        "gamma0": float(gamma0),
        "rho0": float(rho0),
        "omega_p0": float(omega_p0),
        "beta_e_0": float(beta_e_0),
        "R_0": float(r_0),
        "beta_q_0": float(beta_q_0),
        "delta0_u_dc": float(delta0),
        "delta_u_dc": float(delta_u_dc),
        "u_dep": float(u_dep),
        "gamma_dep": float(gamma_dep),
        "rho_dep": float(rho_dep),
        "omega": float(omega),
        "omega_p": float(omega_p_dep),
        "beta_e": float(beta_e),
        "k0": float(k0),
        "tau": float(tau),
        "R": float(r_factor),
        "dlnr_dlntau": float(dlnr_dlntau),
        "dlnbq_ddelta": float(dlnbq_ddelta),
        "omega_q": float(omega_q),
        "beta_q": float(beta_q),
        "T1": float(t1_transverse),
        "T2": float(t2_transverse),
        "eps_fit": float(eps_fit_cont),
        "eps_fit_cont": float(eps_fit_cont),
        "eps_fit_grid": float(eps_fit_grid),
        "lambda_e": float(2.0 * np.pi / beta_e),
        "lambda_0": float(2.0 * np.pi / k0),
        "lambda_p": float(2.0 * np.pi * u_dep / omega_p_dep),
        "lambda_q": float(2.0 * np.pi / beta_q),
        "I0": float(i0_dc),
        "I1_0": float(i1_entrance),
        "qm_scale": float(qm_scale),
    }


def compute_beta_q_at_depression(delta_meas: float, qm_scale: float = 1.0) -> float:
    """Evaluate analytical ``beta_q,th(delta_meas)`` [rad/m] at measured DC depression ``delta_meas``."""
    u0, _ = relativistic_gamma(voltage=V0_VOLTS)
    omega = 2.0 * np.pi * DRIVE_FREQ_HZ
    omega_p0 = 2.0 * np.pi * PLASMA_FREQ_HZ * np.sqrt(qm_scale)
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M

    u_dep_m = u0 * (1.0 - float(delta_meas))
    gamma_dep_m = float(1.0 / np.sqrt(1.0 - (u_dep_m / c_light) ** 2))
    omega_p_dep_m = omega_p0 * np.sqrt(u0 / u_dep_m)
    r_m = float(reduction_factor_planar(omega, u_dep_m, b, a=a))
    return float(r_m * omega_p_dep_m / (gamma_dep_m**1.5 * u_dep_m))


def predict_particle_beta_q(
    delta_meas: float, tols: dict[str, float], qm_scale: float = 1.0
) -> tuple[float, float]:
    """Return ``(beta_q_th_at_delta_meas, beta_q_part_pred)`` [rad/m] for Step 4.2."""
    bq_th_m = compute_beta_q_at_depression(delta_meas, qm_scale=qm_scale)
    bq_pred = float(
        bq_th_m
        * (1.0 - tols["eps_cic"])
        * (1.0 - tols["eps_z_fit"])
        * (1.0 + tols["eps_x_fit"])
        * (1.0 + tols["eps_fit"])
    )
    return bq_th_m, bq_pred


def compute_discrete_dc_depression(nx: int, params: dict[str, float]) -> float:
    """Solve the 1D discrete Poisson problem across ``x`` on ``nx`` cells for ``delta_th,disc``.

    Deposits the top-hat sheet ``[-b, b]`` onto the ``nx + 1`` transverse nodes
    using CIC weights, solves the 3-point discrete Poisson equation with PEC
    boundary conditions ``phi(+-a) = 0``, takes the CIC particle-weighted mean
    potential across the sheet, and divides by ``(gamma**3 * m * u**2 / e)``
    with self-consistent iteration ``delta <- delta0_disc * (1 - delta)**(-3) * (gamma0 / gamma(u))**3``.
    """
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    dx = 2.0 * a / nx
    x_nodes = np.linspace(-a, a, nx + 1)

    n_macro = 16 if nx == COARSE_GRID[0] else 32
    k_idx = np.arange(n_macro, dtype=np.float64)
    x_sheet = b * (2.0 * (k_idx + 0.5) / n_macro - 1.0)
    w_mat = np.maximum(1.0 - np.abs(x_sheet[:, None] - x_nodes[None, :]) / dx, 0.0)
    w_node = np.sum(w_mat, axis=0) * (2.0 * b / (n_macro * dx))

    n_int = nx - 1
    lap = (
        np.diag(np.full(n_int, -2.0))
        + np.diag(np.ones(n_int - 1), 1)
        + np.diag(np.ones(n_int - 1), -1)
    ) / (dx**2)

    qm_scale = params.get("qm_scale", 1.0)
    mass_eff = m_e / qm_scale
    u0 = params["u0"]
    gamma0 = params["gamma0"]
    rho0 = abs(params["rho0"])

    phi_0 = np.zeros(nx + 1, dtype=np.float64)
    phi_0[1:-1] = np.linalg.solve(lap, -rho0 * w_node[1:-1] / epsilon_0)
    phi_mean_0 = float(np.mean(w_mat @ phi_0))
    delta0_disc = float(abs(phi_mean_0) / (gamma0**3 * mass_eff * u0**2 / e_charge))

    delta_disc = delta0_disc
    for _ in range(50):
        u_try = u0 * (1.0 - delta_disc)
        gamma_try = 1.0 / np.sqrt(1.0 - (u_try / c_light) ** 2)
        delta_disc = float(
            delta0_disc * ((1.0 - delta_disc) ** -3) * ((gamma0 / gamma_try) ** 3)
        )
    return float(delta_disc)


def _single_grid_budget_terms(
    nx: int,
    nz: int,
    params: dict[str, float],
    omega_p_scale: float = 1.0,
) -> dict[str, float]:
    """Compute derived discretization terms and slacks for grid ``(nx, 1, nz)``."""
    dx = 2.0 * TUNNEL_HALF_WIDTH_M / nx
    dz = TUNNEL_LENGTH_M / nz
    b = BEAM_HALF_THICKNESS_M
    a = TUNNEL_HALF_WIDTH_M
    ly = PERIODIC_LY_M
    lz = TUNNEL_LENGTH_M

    omega = params["omega"]
    u_dep = params["u_dep"]
    gamma_dep = params["gamma_dep"]
    omega_p_dep = params["omega_p"] * float(omega_p_scale)
    beta_e = params["beta_e"]
    beta_q = params["beta_q"] * float(omega_p_scale)
    k0 = params["k0"]
    t1 = params["T1"]
    r_fac = params["R"]
    dlnr_dlntau = params["dlnr_dlntau"]

    # Integer-period time step dt on (nx, 1, nz)
    grid = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, 1, nz, verbose=0)
    dt, _ = _integer_period_dt(grid)

    # 0. Multi-mode fit shifts eps_fit,cont and eps_fit,grid on (nx, 1, nz)
    _, t2, eps_fit_cont, eps_fit_grid = _compute_multimode_fit_shift(
        params["tau"],
        b,
        a,
        omega_p_dep,
        gamma_dep,
        u_dep,
        beta_e,
        nx=nx,
        nz=nz,
    )
    eps_fit = float(eps_fit_cont)
    eps_fit_diff = float(abs(eps_fit_grid - eps_fit_cont))

    # Shared SolverFIT3D field-discretization terms:
    # (a) 3-point transverse FIT Laplacian under-estimates T1^2 by -(T1*dx)^2/12
    eps_x_fit = float(0.5 * (1.0 - r_fac**2) * (t1 * dx) ** 2 / 12.0)
    # (b) Staggered Yee axial curl-curl symbol acting on the physical carrier beta_e
    eps_z_fit = float(dlnr_dlntau * (gamma_dep**2) * (beta_e * dz) ** 2 / 24.0)

    # 1. LinearizedBeam exact discrete axial dispersion:
    #    (2/dt)*tan(omega*dt/2) -+ omega_q(k_+-) = u_dep * sin(k_+- * dz) / dz
    omega_dt_lhs = (2.0 / dt) * np.tan(0.5 * omega * dt)

    def _omega_q_yee(k_val: float) -> float:
        kappa = (2.0 / dz) * np.sin(0.5 * k_val * dz)
        tau_y = float(np.sqrt(max(kappa**2 - k0**2, 1e-12)))
        r_y = r_of_tau(tau_y, u_dep, gamma_dep, b, a)
        return float(r_y * omega_p_dep / (gamma_dep**1.5))

    k_lo = 0.5 * beta_e
    k_hi = 0.99 * np.pi / (2.0 * dz)
    k_minus = float(
        brentq(
            lambda k: omega_dt_lhs - _omega_q_yee(k) - u_dep * np.sin(k * dz) / dz,
            k_lo,
            k_hi,
        )
    )
    k_plus = float(
        brentq(
            lambda k: omega_dt_lhs + _omega_q_yee(k) - u_dep * np.sin(k * dz) / dz,
            k_lo,
            k_hi,
        )
    )
    bq_ax_disc = 0.5 * (k_plus - k_minus)
    eps_ax = float(bq_ax_disc / beta_q - 1.0)
    bq_lb_pred = float(bq_ax_disc * (1.0 + eps_x_fit) * (1.0 + eps_fit))
    s_lb = float((abs(eps_ax) + abs(eps_x_fit) + abs(eps_fit)) ** 2 + eps_fit_diff)

    # 2. ParticleSolver deterministic ("sheet_lattice") discretization factors & slack:
    eps_cic = float(1.0 - np.sqrt(1.0 - (2.0 / 3.0) * np.sin(0.5 * t1 * dx) ** 2))
    part_disc_factor = float(
        (1.0 - eps_cic) * (1.0 - eps_z_fit) * (1.0 + eps_x_fit) * (1.0 + eps_fit)
    )
    s_part = float(
        (abs(eps_cic) + abs(eps_z_fit) + abs(eps_x_fit) + abs(eps_fit)) ** 2
        + eps_fit_diff
    )
    s_cross = float(s_lb + s_part)

    # 3. Discrete 1D Poisson DC depression prediction on this grid:
    delta_th_disc = compute_discrete_dc_depression(nx, params)

    # 4. Entrance amplitude bounds at first primal edge midpoint z_0 = 0.5 * dz:
    tol_amp_lb = float(0.5 * (beta_q * dz) ** 2)
    tol_amp_part = float(tol_amp_lb + 1e-12)

    # 5. Residual DFT leakage bound (identity / round-off):
    eps_dft = float(100.0 * np.finfo(np.float64).eps / MODULATION_INDEX)

    # 6. LinearizedBeam 2*omega C^1 turn-on ramp transient tail:
    n_rise = float(RISE_PERIODS)
    n_per = 6.0
    tol_i2_lb = float(1.0 / ((2.0 * n_rise) ** 2 * np.pi * n_per))

    # 7. Two-sided 99.73% Student-t critical value for 8 seeds (dof = 7):
    tcrit = float(student_t.ppf(0.99865, len(NOISE_SEEDS) - 1))

    return {
        "t2_dx": float(t2 * dx),
        "k_minus": k_minus,
        "k_plus": k_plus,
        "bq_ax_disc": bq_ax_disc,
        "eps_ax": eps_ax,
        "eps_x_fit": eps_x_fit,
        "eps_z_fit": eps_z_fit,
        "eps_cic": eps_cic,
        "eps_fit": eps_fit,
        "eps_fit_cont": eps_fit_cont,
        "eps_fit_grid": eps_fit_grid,
        "eps_fit_diff": eps_fit_diff,
        "bq_lb_pred": bq_lb_pred,
        "part_disc_factor": part_disc_factor,
        "s_lb": s_lb,
        "s_part": s_part,
        "s_cross": s_cross,
        "delta_th": float(params["delta_u_dc"]),
        "delta_th_disc": delta_th_disc,
        "tol_amp_lb": tol_amp_lb,
        "tol_amp_part": tol_amp_part,
        "eps_dft": eps_dft,
        "tol_i2_lb": tol_i2_lb,
        "tcrit": tcrit,
    }


def derived_tolerances(
    nx: int,
    nz: int,
    params: dict[str, float],
    omega_p_scale: float = 1.0,
) -> dict[str, float]:
    """Build the Step 4 predictions and discretization slacks for grid ``(nx, 1, nz)``.

    Parameters
    ----------
    nx, nz : int
        Transverse and axial grid cell counts.
    params : dict of str to float
        Output of :func:`compute_analytic_parameters`.
    omega_p_scale : float, optional
        Multiplier on ``omega_p,dep`` for `LinearizedBeam` (``1.0`` for baseline,
        ``sqrt(2)`` for the M2 ``rho0 * 2`` mutation). Default is ``1.0``.

    Returns
    -------
    dict of str to float
        Dictionary of discrete dispersion predictions, individual epsilon terms,
        slacks ``s_lb``, ``s_part``, ``s_cross``, discrete DC depression
        ``delta_th_disc``, entrance amplitude bounds, and DFT round-off bound.
    """
    return _single_grid_budget_terms(nx, nz, params, omega_p_scale=omega_p_scale)


def format_error_budget(nx: int, nz: int, tols: dict[str, float]) -> str:
    """Format every Step 4 budget term and slack for grid ``(nx, 1, nz)``."""
    lines = [
        f"Error budget for grid ({nx}, 1, {nz}):",
        f"  k_-, k_+ (exact discrete LB branches)             : ({tols['k_minus']:.4f}, {tols['k_plus']:.4f}) rad/m",
        f"  eps_ax ((k_+ - k_-)/(2*bq_th) - 1)                : {tols['eps_ax']:+.6%}",
        f"  eps_x_fit (0.5*(1-R^2)*(T1*dx)^2/12)              : {tols['eps_x_fit']:+.6%}",
        f"  eps_z_fit' (dlnR/dlntau*gamma^2*(beta_e*dz)^2/24) : {tols['eps_z_fit']:+.6%}",
        f"  eps_cic (transverse CIC)                          : {tols['eps_cic']:+.6%}",
        f"  eps_fit,cont (continuum multi-mode shift)         : {tols['eps_fit_cont']:+.6%} (T2*dx={tols['t2_dx']:.3f} rad)",
        f"  eps_fit,grid (discrete CIC multi-mode shift)      : {tols['eps_fit_grid']:+.6%}",
        f"  |eps_fit,grid - eps_fit,cont|                     : {tols['eps_fit_diff']:.6%}",
        f"  bq_lb_pred                                        : {tols['bq_lb_pred']:.6f} rad/m",
        f"  s_lb (LB slack)                                   : {tols['s_lb']:.6%}",
        f"  s_part (Particle slack)                           : {tols['s_part']:.6%}",
        f"  s_cross (s_lb + s_part)                           : {tols['s_cross']:.6%}",
        f"  delta_th (self-consistent continuum DC dep)       : {tols['delta_th']:.6%}",
        f"  delta_th_disc (1D discrete Poisson CIC DC dep)    : {tols['delta_th_disc']:.6%}",
        f"  tol_amp_lb (0.5*(bq*dz)^2)                        : {tols['tol_amp_lb']:.6e}",
        f"  tol_amp_part (tol_amp_lb + 1e-12)                 : {tols['tol_amp_part']:.6e}",
        f"  eps_dft (100*eps_mach/m)                          : {tols['eps_dft']:.6e}",
    ]
    return "\n".join(lines)


def fit_beta_q_from_envelope(
    z_coords: np.ndarray,
    i1_phasor: np.ndarray,
    beta_q_guess: float,
    z_min: float | None = None,
    z_max: float | None = None,
) -> dict[str, Any]:
    """Fit the complex two-wave model to ``I1(z)`` by variable-projection least squares.

    On axial edge midpoints ``z_k`` inside ``[z_min, z_max]``, fits the complex
    fast/slow space-charge-wave model::

        I1(z) = a_f * exp(-i * k_f * z) + a_s * exp(-i * k_s * z)

    with complex amplitudes ``a_f, a_s`` and real wavenumbers ``k_f < k_s``.
    This is identical to::

        I1(z) = exp(-i * k_c * z) * [A * cos(beta_q * z) + B * sin(beta_q * z)]

    with ``k_c = (k_f + k_s) / 2``, ``beta_q = (k_s - k_f) / 2``,
    ``A = a_f + a_s``, and ``B = i * (a_s - a_f)``. The carrier wavenumber
    ``k_c`` is fitted rather than fixed to ``beta_e`` because `LinearizedBeam`'s
    numerical carrier differs from ``beta_e`` by ``(beta_e * dz)**2 / 6``, and
    fixing it would bias ``beta_q``.

    For a given ``(k_c, beta_q)``, ``(a_f, a_s)`` follow from a 2-column complex
    linear least-squares solve (`numpy.linalg.lstsq` on the complex design matrix
    ``[exp(-i * k_f * z), exp(-i * k_s * z)]``). The residual sum of squares
    ``RSS(k_c, beta_q)`` is minimised over a ``201 x 401`` coarse grid with
    ``k_c`` in ``k_c0 +- 3%`` (``k_c0`` from the unwrapped carrier phase slope
    via `numpy.polyfit`) and ``beta_q`` in ``[0.5, 1.8] * beta_q_guess``, followed
    by `scipy.optimize.minimize` (``method="Nelder-Mead"``,
    ``xatol=1e-10 * beta_q_guess``) starting from the best grid point.

    Parameters
    ----------
    z_coords : np.ndarray
        Axial edge midpoint coordinates [m].
    i1_phasor : np.ndarray
        Complex fundamental axial current phasor ``I1(z)`` [A].
    beta_q_guess : float
        Reference reduced plasma wavenumber [rad/m] centering the initial scan.
    z_min, z_max : float, optional
        Lower and upper axial bounds [m] of the fit window. If omitted,
        computed via :func:`fit_window`.

    Returns
    -------
    dict of str to Any
        Dictionary with keys ``beta_q``, ``k_c``, ``A``, ``B``, ``a_f``, ``a_s``,
        ``rss``, ``sigma_beta_q`` (from the Gauss-Newton covariance
        ``s**2 * (J^H J)^(-1)`` with ``s**2 = RSS / (2 * N - 6)``), plus
        ``k_f``, ``k_s``, ``z0``, ``as_over_af``, and ``b_over_a``.
    """
    if z_min is None or z_max is None:
        z_min_def, z_max_def = fit_window(compute_analytic_parameters())
        if z_min is None:
            z_min = z_min_def
        if z_max is None:
            z_max = z_max_def

    mask = (z_coords >= z_min) & (z_coords <= z_max)
    zm = np.asarray(z_coords[mask], dtype=np.float64)
    ym = np.asarray(i1_phasor[mask], dtype=np.complex128)
    n_pts = len(zm)

    # Carrier slope k_c0 from unwrapped phase (removing integer-pi standing-wave nodal jumps)
    phi_u = np.unwrap(np.angle(ym))
    amp = np.abs(ym)
    lobe = (amp[:-1] > 0.5 * np.median(amp)) & (amp[1:] > 0.5 * np.median(amp))
    if np.any(lobe):
        kc_lobe = -float(
            np.median(np.angle(ym[1:] * np.conj(ym[:-1]))[lobe] / np.diff(zm)[lobe])
        )
        demod = phi_u + kc_lobe * zm
        demod = demod - np.round((demod - demod[0]) / np.pi) * np.pi
        phi_carrier = demod - kc_lobe * zm
    else:
        phi_carrier = phi_u
    k_c0 = -float(np.polyfit(zm, phi_carrier, 1)[0])

    kc_half = max(0.03 * abs(k_c0), 1.0)
    kc_grid = np.linspace(k_c0 - kc_half, k_c0 + kc_half, 201)
    bq_grid = np.linspace(0.50 * beta_q_guess, 1.80 * beta_q_guess, 401)

    # Vectorized 201 x 401 variable-projection coarse grid evaluation
    # Design columns u_f = exp(-i*(k_c - bq)*z), u_s = exp(-i*(k_c + bq)*z)
    # have Gram entries <u_f, u_f> = <u_s, u_s> = N and <u_f, u_s> = sum_k exp(-2i*bq*z_k).
    ym_energy = float(np.real(np.vdot(ym, ym)))
    exp_kc_pos = np.exp(1j * np.outer(kc_grid, zm))
    exp_bq_neg = np.exp(-1j * np.outer(bq_grid, zm))
    exp_bq_pos = np.conj(exp_bq_neg)
    p_f = (exp_kc_pos * ym[None, :]) @ exp_bq_neg.T
    p_s = (exp_kc_pos * ym[None, :]) @ exp_bq_pos.T
    g_12 = np.sum(exp_bq_neg**2, axis=1)
    denom = np.maximum(float(n_pts**2) - np.abs(g_12) ** 2, 1e-30)[None, :]
    num_proj = float(n_pts) * (np.abs(p_f) ** 2 + np.abs(p_s) ** 2) - 2.0 * np.real(
        np.conj(p_f) * g_12[None, :] * p_s
    )
    rss_grid = ym_energy - num_proj / denom
    idx_kc, idx_bq = np.unravel_index(int(np.argmin(rss_grid)), rss_grid.shape)
    kc_init = float(kc_grid[idx_kc])
    bq_init = float(bq_grid[idx_bq])

    def _varpro_lstsq(kc_val: float, bq_val: float) -> tuple[complex, complex, float]:
        bq_pos = abs(bq_val)
        kf_val = kc_val - bq_pos
        ks_val = kc_val + bq_pos
        design = np.column_stack([np.exp(-1j * kf_val * zm), np.exp(-1j * ks_val * zm)])
        coef, _, _, _ = np.linalg.lstsq(design, ym, rcond=None)
        res = ym - design @ coef
        rss_val = float(np.real(np.vdot(res, res)))
        return complex(coef[0]), complex(coef[1]), rss_val

    opt = minimize(
        lambda x: _varpro_lstsq(float(x[0]), float(x[1]))[2],
        x0=np.array([kc_init, bq_init], dtype=np.float64),
        method="Nelder-Mead",
        options={
            "xatol": 1e-10 * beta_q_guess,
            "fatol": 1e-16 * max(ym_energy, 1e-30),
            "maxiter": 2000,
        },
    )
    k_c = float(opt.x[0])
    beta_q = float(abs(opt.x[1]))
    a_f, a_s, rss = _varpro_lstsq(k_c, beta_q)
    a_cap = complex(a_f + a_s)
    b_cap = complex(1j * (a_s - a_f))

    # Gauss-Newton covariance s^2 * (J^H J)^(-1) with s^2 = RSS / (2*N - 6)
    # for the 6 real parameters [Re(a_f), Im(a_f), Re(a_s), Im(a_s), k_c, beta_q]
    u_f = np.exp(-1j * (k_c - beta_q) * zm)
    u_s = np.exp(-1j * (k_c + beta_q) * zm)
    f_model = a_f * u_f + a_s * u_s
    df_dkc = -1j * zm * f_model
    df_dbq = 1j * zm * (a_f * u_f - a_s * u_s)
    j_complex = np.column_stack([u_f, 1j * u_f, u_s, 1j * u_s, df_dkc, df_dbq])
    jtj = np.real(j_complex.conj().T @ j_complex)
    s2 = rss / max(2 * n_pts - 6, 1)
    cov = s2 * np.linalg.pinv(jtj)
    sigma_beta_q = float(np.sqrt(max(cov[5, 5], 0.0)))

    b_over_a = complex(b_cap / a_cap) if abs(a_cap) > 0.0 else 0.0j
    z0_eff = float(np.arctan(np.real(b_over_a)) / beta_q) if beta_q > 0.0 else 0.0
    as_over_af = float(abs(a_s) / max(abs(a_f), 1e-30))

    return {
        "beta_q": beta_q,
        "k_c": k_c,
        "A": a_cap,
        "B": b_cap,
        "a_f": a_f,
        "a_s": a_s,
        "rss": rss,
        "sigma_beta_q": sigma_beta_q,
        "k_f": float(k_c - beta_q),
        "k_s": float(k_c + beta_q),
        "z0": z0_eff,
        "as_over_af": as_over_af,
        "b_over_a": b_over_a,
    }


def compute_spatial_spectrum(
    z_coords: np.ndarray,
    i1_phasor: np.ndarray,
    beta_e: float,
    beta_q: float,
    z_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute the spatial wavenumber spectrum ``|I1(k_z)|`` around ``beta_e``.

    Over one full plasma beat wavelength ``z_max = 2 * pi / beta_q``, the fast
    (``beta_e - beta_q``) and slow (``beta_e + beta_q``) space-charge waves are
    orthogonal to ``exp(+i * beta_e * z)``, producing two symmetric peaks
    straddling a deep null at ``k_z = beta_e``.

    Parameters
    ----------
    z_coords : np.ndarray
        Axial edge midpoint coordinates [m].
    i1_phasor : np.ndarray
        Complex axial current phasor ``I1(z)`` [A].
    beta_e : float
        Electronic wavenumber ``omega / u_dep`` [rad/m].
    beta_q : float
        Reduced plasma wavenumber [rad/m].
    z_max : float, optional
        Spatial window length [m]. Defaults to ``2 * pi / beta_q``.

    Returns
    -------
    kz_grid : np.ndarray
        Wavenumber grid [rad/m].
    spectrum : np.ndarray
        Normalized spatial spectrum amplitude ``|I1(k_z)|`` [A].
    k_fast_peak : float
        Measured fast-wave peak wavenumber [rad/m] in ``k_z < beta_e``.
    k_slow_peak : float
        Measured slow-wave peak wavenumber [rad/m] in ``k_z > beta_e``.
    """
    if z_max is None:
        z_max = 2.0 * np.pi / beta_q
    mask = z_coords <= z_max
    z_m = z_coords[mask]
    i1_m = i1_phasor[mask]
    dz = float(z_m[1] - z_m[0])
    length = float(len(z_m) * dz)

    kz_grid = np.linspace(beta_e - 3.5 * beta_q, beta_e + 3.5 * beta_q, 2001)
    kernel = np.exp(1j * np.outer(z_m, kz_grid))
    spectrum = np.abs(np.dot(i1_m, kernel)) * (dz / length)

    mid = len(kz_grid) // 2
    k_fast_peak = float(kz_grid[int(np.argmax(spectrum[:mid]))])
    k_slow_peak = float(kz_grid[mid + int(np.argmax(spectrum[mid:]))])
    return kz_grid, spectrum, k_fast_peak, k_slow_peak


def _integer_period_dt(grid: GridFIT3D, cfln: float = CFL_NUMBER) -> tuple[float, int]:
    """Choose ``dt <= dt_cfl`` so ``1 / (DRIVE_FREQ_HZ * dt)`` is an exact integer `n_spp`."""
    s_tmp = SolverFIT3D(
        grid,
        cfln=cfln,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_gpu=False,
        verbose=0,
    )
    dt_cfl = float(s_tmp.dt)
    n_spp = int(np.ceil(1.0 / (DRIVE_FREQ_HZ * dt_cfl)))
    dt = 1.0 / (DRIVE_FREQ_HZ * n_spp)
    return float(dt), int(n_spp)


def run_linearized_beam_drift(
    nx: int,
    nz: int,
    rho0_scale: float = 1.0,
    m_mod: float = MODULATION_INDEX,
    rho0_profile: Any = None,
) -> dict[str, Any]:
    """Run `LinearizedBeam` in the planar drift tunnel using the public `inflow_rho1` API.

    Parameters
    ----------
    nx, nz : int
        Transverse and axial cell counts.
    rho0_scale : float, optional
        Multiplier applied to the unperturbed charge density ``rho0`` (``1.0``
        for baseline, ``2.0`` for the mutation test). When `LinearizedBeam.__init__`
        is monkeypatched in the mutation test, ``kwargs['rho0']`` inside
        `LinearizedBeam` doubles while `inflow_rho1` remains anchored at the
        baseline entrance density modulation ``m * rho_dep * sin(omega * t)``.
    m_mod : float, optional
        Fractional AC density modulation depth ``m``. Default is ``MODULATION_INDEX``.
    rho0_profile : callable or array_like, optional
        Optional transverse/3D unperturbed charge density ``rho0`` passed to
        `LinearizedBeam` (used in D3 for the semicircular profile). Default is ``None``.

    Returns
    -------
    dict of str to Any
        Dictionary with keys ``beam``, ``z_coords``, ``i1_lb``, ``i2_lb``,
        ``dc_leakage_lb``, ``dirichlet_leak_lb``, ``dt``, and ``n_spp``.
    """
    params = compute_analytic_parameters()
    u_dep = params["u_dep"]
    gamma_dep = params["gamma_dep"]
    omega = params["omega"]
    rho_dep = params["rho_dep"]
    rho0_val = rho0_profile if rho0_profile is not None else (rho0_scale * rho_dep)
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    ly = PERIODIC_LY_M
    lz = TUNNEL_LENGTH_M
    t_rise = RISE_PERIODS / DRIVE_FREQ_HZ

    dz = lz / nz
    n_ramp = round(OUTFLOW_RAMP_M / dz)
    grid = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, 1, nz, verbose=0)
    dt, n_spp = _integer_period_dt(grid)
    solver = SolverFIT3D(
        grid,
        cfln=CFL_NUMBER,
        dt=dt,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_gpu=USE_GPU,
        verbose=0,
    )

    def inflow_rho1(t: float) -> float:
        ramp_t = 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))
        return float(m_mod * ramp_t * rho_dep * np.sin(omega * t))

    beam = LinearizedBeam(
        u0=u_dep,
        rho0=rho0_val,
        gamma0=gamma_dep,
        axis="z",
        half_thickness=b,
        ramp_cells=n_ramp,
        inflow_rho1=inflow_rho1,
        solver=solver,
    )

    n_settle = int(np.ceil(1.01 * lz / (u_dep * dt)))
    n_dft = 6 * n_spp
    n_total = n_settle + n_dft

    iz_hist = np.zeros((n_dft, beam.M), dtype=np.float64)
    t_mids = np.zeros(n_dft, dtype=np.float64)
    for step in range(n_total):
        solver.one_step()
        if step >= n_settle:
            k = step - n_settle
            t_mids[k] = (step + 0.5) * dt
            iz_hist[k, :] = beam.axial_current()

    iz_dc = np.mean(iz_hist, axis=0)
    iz_ac = iz_hist - iz_dc[None, :]
    ker1 = np.exp(-1j * omega * t_mids) * (2.0 / n_dft)
    ker2 = np.exp(-2j * omega * t_mids) * (2.0 / n_dft)
    i1_lb = np.dot(ker1, iz_ac)
    i2_lb = np.dot(ker2, iz_ac)

    # Residual DC leakage:
    # (1) dc_leak removes the mean of a time-constant array (an algebraic zero identity);
    # (2) dirichlet_leak = |sum(ker1)| * (I0 / I1_0) tests the actual physical fix:
    #     choosing dt so N_spp = 1 / (f * dt) is an exact integer makes sum(ker1) = 0
    #     to floating-point round-off.
    dc_test = np.broadcast_to(iz_dc[None, :], iz_hist.shape)
    i1_scale = m_mod * params["I0"]
    dc_leak = (
        np.max(np.abs(np.dot(ker1, dc_test - np.mean(dc_test, axis=0)))) / i1_scale
    )
    dirichlet_leak = float(np.abs(np.sum(ker1)) * (params["I0"] / i1_scale))

    return {
        "beam": beam,
        "z_coords": beam.s_coords.copy(),
        "i1_lb": i1_lb,
        "i2_lb": i2_lb,
        "dc_leakage_lb": float(max(dc_leak, dirichlet_leak)),
        "dirichlet_leak_lb": dirichlet_leak,
        "dt": dt,
        "n_spp": n_spp,
    }


def run_particle_beam_drift(
    nx: int,
    nz: int,
    beam_ref: LinearizedBeam,
    qm_scale: float = 1.0,
    seed: int = 42,
    m_mod: float = MODULATION_INDEX,
    profile: str = "sheet_lattice",
    cfln: float = CFL_NUMBER,
    n_macro_per_step: int | None = None,
) -> dict[str, Any]:
    """Run `ParticleSolver` + `BeamInjector` on the identical drift-tunnel grid.

    Integrates the Esirkepov-deposited particle current density ``ps.J_particles``
    over the exact same primal ``E_z`` edges (``beam_ref.flat_idx``) and dual
    areas (``beam_ref.dA_perp``) used by `LinearizedBeam.axial_current`, with
    identical integer-period ``dt`` and time-mean subtraction before the DFT.
    Also measures the interior DC velocity ``<u_z>(z)`` and deposited particle
    charge density ``rho_DC(z)`` time-averaged over one RF period across the
    settled interior ``z in [0.2 * Lz, 0.8 * Lz]``.

    Profile choice:
    - Deterministic reference run = ``"sheet_lattice"`` (stratum midpoints, zero
      transverse shot noise, identical for all seeds). Used for every signed
      physics check, both mutations of the particle code, the 2-resolution check,
      and the m-scan.
    - 8-seed noise set = ``"sheet"`` (one seeded uniform draw per stratum per step
      from ``_step_rng``).

    Parameters
    ----------
    nx, nz : int
        Transverse and axial cell counts.
    beam_ref : LinearizedBeam
        Reference `LinearizedBeam` on the same grid providing ``flat_idx`` and ``dA_perp``.
    qm_scale : float, optional
        Multiplier on the species charge-to-mass ratio ``|q / m|`` implemented
        by scaling ``mass = m_e / qm_scale`` at fixed ``charge = -e`` and ``gamma = gamma0``.
    seed : int, optional
        The `BeamInjector` seed (only affects ``profile='sheet'``). Default is ``42``.
    m_mod : float, optional
        Fractional AC density modulation depth ``m``. Default is ``MODULATION_INDEX``.
    profile : str, optional
        Transverse `BeamInjector` sampling profile (``"sheet_lattice"`` for
        deterministic stratum midpoints or ``"sheet"`` for seeded stratified
        draws). Default is ``"sheet_lattice"``.
    cfln : float, optional
        Courant number passed to `SolverFIT3D`. Default is ``CFL_NUMBER``.
    n_macro_per_step : int, optional
        Number of macroparticles injected per step. If ``None``, defaults to
        ``16`` on ``COARSE_GRID`` and ``32`` on ``FINE_GRID``.

    Returns
    -------
    dict of str to Any
        Dictionary with keys ``i1_part``, ``i2_part``, ``dc_leakage_part``,
        ``dirichlet_leak_part``, ``u_z_mean``, ``rho_dc_mean``, ``u_z_profile``,
        ``rho_dc_profile``, ``dt``, and ``n_spp``.
    """
    params = compute_analytic_parameters(qm_scale=qm_scale)
    u_dep = params["u_dep"]
    gamma0 = params["gamma0"]
    omega = params["omega"]
    i0_dc = params["I0"]
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    ly = PERIODIC_LY_M
    lz = TUNNEL_LENGTH_M
    t_rise = RISE_PERIODS / DRIVE_FREQ_HZ

    grid = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, 1, nz, verbose=0)
    dt, n_spp = _integer_period_dt(grid, cfln=cfln)
    solver = SolverFIT3D(
        grid,
        cfln=cfln,
        dt=dt,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_gpu=USE_GPU,
        verbose=0,
    )

    sp = Species(name="electron", charge=-e_charge, mass=m_e / qm_scale)

    def current_modulation(t: float) -> float:
        ramp_t = 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))
        return i0_dc * (1.0 + m_mod * ramp_t * np.sin(omega * t))

    n_macro = (
        int(n_macro_per_step)
        if n_macro_per_step is not None
        else (16 if nx == COARSE_GRID[0] else 32)
    )
    inj = BeamInjector(
        species=sp,
        z0=0.0,
        current=current_modulation,
        gamma=gamma0,
        profile=profile,
        radius=b,
        x0=0.0,
        y0=0.0,
        n_macro_per_step=n_macro,
        seed=seed,
    )
    ps = ParticleSolver(
        solver,
        species=[sp],
        injectors=[inj],
        shape=1,
        B_ext=(0.0, 0.0, GUIDE_BZ_TESLA),
    ).attach()

    n_settle = int(np.ceil(1.01 * lz / (u_dep * dt)))
    n_dft = 6 * n_spp
    n_total = n_settle + n_dft

    iz_hist = np.zeros((n_dft, beam_ref.M), dtype=np.float64)
    t_mids = np.zeros(n_dft, dtype=np.float64)

    dz = lz / nz
    z_edges = np.linspace(0.0, lz, nz + 1)
    w_sum_dc = np.zeros(nz, dtype=np.float64)
    wv_sum_dc = np.zeros(nz, dtype=np.float64)
    n_dc_samples = 0

    for step in range(n_total):
        solver.one_step()
        if step >= n_settle:
            k = step - n_settle
            t_mids[k] = (step + 0.5) * dt
            j_raw = ps.J_particles[beam_ref.flat_idx]
            j_host = np.asarray(j_raw.get() if hasattr(j_raw, "get") else j_raw)
            j_edges = j_host.reshape(beam_ref.M, beam_ref.N_lines)
            iz_hist[k, :] = np.sum(j_edges * beam_ref.dA_perp, axis=1)

            # Accumulate DC velocity and density profiles over 1 RF period (n_spp steps)
            # at the start of the steady-state window to cancel AC modulation
            if k < n_spp:
                zp = np.asarray(sp.z.get() if hasattr(sp.z, "get") else sp.z)
                uz_p = np.asarray(sp.uz.get() if hasattr(sp.uz, "get") else sp.uz)
                wp = np.asarray(sp.w.get() if hasattr(sp.w, "get") else sp.w)
                vz_p = uz_p / np.sqrt(1.0 + (uz_p / c_light) ** 2)
                bin_idx = np.clip(np.digitize(zp, z_edges) - 1, 0, nz - 1)
                w_sum_dc += np.bincount(bin_idx, weights=wp, minlength=nz)
                wv_sum_dc += np.bincount(bin_idx, weights=wp * vz_p, minlength=nz)
                n_dc_samples += 1

    iz_dc = np.mean(iz_hist, axis=0)
    iz_ac = iz_hist - iz_dc[None, :]
    ker1 = np.exp(-1j * omega * t_mids) * (2.0 / n_dft)
    ker2 = np.exp(-2j * omega * t_mids) * (2.0 / n_dft)
    i1_part = np.dot(ker1, iz_ac)
    i2_part = np.dot(ker2, iz_ac)

    # Residual DC leakage: dc_leak is a zero-mean algebraic identity; dirichlet_leak
    # tests the integer-n_spp orthogonality |sum(ker1)| * (I0 / I1_0).
    dc_test = np.broadcast_to(iz_dc[None, :], iz_hist.shape)
    i1_scale = m_mod * i0_dc
    dc_leak = (
        np.max(np.abs(np.dot(ker1, dc_test - np.mean(dc_test, axis=0)))) / i1_scale
    )
    dirichlet_leak = float(np.abs(np.sum(ker1)) * (i0_dc / i1_scale))

    w_mean_cell = w_sum_dc / max(n_dc_samples, 1)
    wv_mean_cell = wv_sum_dc / max(n_dc_samples, 1)
    u_z_profile = np.where(
        w_mean_cell > 0.0, wv_mean_cell / np.maximum(w_mean_cell, 1e-30), u_dep
    )

    area_beam = 2.0 * b * ly
    rho_dc_profile = (sp.charge * w_mean_cell) / (area_beam * dz)

    z_coords = beam_ref.s_coords
    interior = (z_coords >= 0.2 * lz) & (z_coords <= 0.8 * lz)
    u_z_mean = float(np.sum(wv_mean_cell[interior]) / np.sum(w_mean_cell[interior]))
    rho_dc_mean = float(np.mean(rho_dc_profile[interior]))

    return {
        "i1_part": i1_part,
        "i2_part": i2_part,
        "dc_leakage_part": float(max(dc_leak, dirichlet_leak)),
        "dirichlet_leak_part": dirichlet_leak,
        "u_z_mean": u_z_mean,
        "rho_dc_mean": rho_dc_mean,
        "u_z_profile": u_z_profile,
        "rho_dc_profile": rho_dc_profile,
        "dt": dt,
        "n_spp": n_spp,
    }


@lru_cache(maxsize=16)
def cached_linearized_beam_drift(
    nx: int,
    nz: int,
    rho0_scale: float = 1.0,
    m_mod: float = MODULATION_INDEX,
) -> dict[str, Any]:
    """Cache `LinearizedBeam` runs by ``(nx, nz, rho0_scale, m_mod)``."""
    return run_linearized_beam_drift(nx, nz, rho0_scale=rho0_scale, m_mod=m_mod)


@lru_cache(maxsize=64)
def cached_particle_drift(
    nx: int,
    nz: int,
    profile: str = "sheet_lattice",
    seed: int = 42,
    m_mod: float = MODULATION_INDEX,
    qm_scale: float = 1.0,
) -> dict[str, Any]:
    """Cache `ParticleSolver` runs by ``(nx, nz, profile, seed, m_mod, qm_scale)``."""
    lb_ref = cached_linearized_beam_drift(
        nx, nz, rho0_scale=1.0, m_mod=MODULATION_INDEX
    )
    return run_particle_beam_drift(
        nx,
        nz,
        beam_ref=lb_ref["beam"],
        qm_scale=qm_scale,
        seed=seed,
        m_mod=m_mod,
        profile=profile,
    )


@lru_cache(maxsize=8)
def cached_baseline_pair(
    nx: int, nz: int, m_mod: float = MODULATION_INDEX
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cache deterministic ``("sheet_lattice")`` baseline `(LinearizedBeam, ParticleSolver)` runs."""
    lb_res = cached_linearized_beam_drift(nx, nz, rho0_scale=1.0, m_mod=m_mod)
    part_res = cached_particle_drift(
        nx,
        nz,
        profile="sheet_lattice",
        seed=NOISE_SEEDS[0],
        m_mod=m_mod,
        qm_scale=1.0,
    )
    return lb_res, part_res


@lru_cache(maxsize=8)
def measure_empirical_noise(
    grid: tuple[int, int, int] = COARSE_GRID,
    m_mod: float = MODULATION_INDEX,
) -> dict[str, Any]:
    """Measure empirical 8-seed ``profile='sheet'`` statistics on ``grid``.

    Runs seeds ``42..49`` on ``grid`` at modulation depth ``m_mod`` using
    ``profile='sheet'`` (one seeded uniform draw per stratum per step from
    ``BeamInjector._step_rng``), fits the two-wave model over ``fit_window(params)``
    for each seed, and computes the 8-seed mean, standard error of the mean
    ``SEM = std(ddof=1) / sqrt(8)``, and two-sided 99.73% Student-t critical
    value ``t = scipy.stats.t.ppf(0.99865, 7)`` for ``beta_q``, DC depression
    ``delta_meas``, and entrance amplitude ``|I1(z_0)|``.
    """
    params = compute_analytic_parameters()
    nx, _, nz = grid
    lb_res = cached_linearized_beam_drift(nx, nz, rho0_scale=1.0, m_mod=m_mod)
    z_coords = lb_res["z_coords"]
    beta_e = params["beta_e"]
    beta_q_th = params["beta_q"]
    lam_q = params["lambda_q"]
    z_min, z_max = fit_window(params)
    u0 = params["u0"]

    bq_list: list[float] = []
    z0_list: list[float] = []
    delta_list: list[float] = []
    amp0_list: list[float] = []
    i2_bunch_list: list[float] = []
    i1_list: list[np.ndarray] = []

    for seed in NOISE_SEEDS:
        res = cached_particle_drift(
            nx,
            nz,
            profile="sheet",
            seed=seed,
            m_mod=m_mod,
            qm_scale=1.0,
        )
        i1 = res["i1_part"]
        i2 = res["i2_part"]
        fit_s = fit_beta_q_from_envelope(
            z_coords, i1, beta_q_th, z_min=z_min, z_max=z_max
        )
        i2_bunch_s = extract_coherent_bunching_i2_ratio(
            z_coords, i1, i2, beta_e, beta_q_th, z_max=lam_q
        )
        bq_list.append(float(fit_s["beta_q"]))
        z0_list.append(float(fit_s["z0"]))
        delta_list.append(float((u0 - res["u_z_mean"]) / u0))
        amp0_list.append(float(np.abs(i1[0])))
        i2_bunch_list.append(float(i2_bunch_s))
        i1_list.append(i1)

    n_seeds = len(NOISE_SEEDS)
    tcrit = float(student_t.ppf(0.99865, n_seeds - 1))

    bq_seeds = np.asarray(bq_list, dtype=np.float64)
    z0_seeds = np.asarray(z0_list, dtype=np.float64)
    delta_seeds = np.asarray(delta_list, dtype=np.float64)
    amp0_seeds = np.asarray(amp0_list, dtype=np.float64)
    i2_bunch_seeds = np.asarray(i2_bunch_list, dtype=np.float64)
    i1_mean = np.mean(np.stack(i1_list, axis=0), axis=0)

    bq_mean = float(np.mean(bq_seeds))
    bq_std = float(np.std(bq_seeds, ddof=1))
    bq_sem = float(bq_std / np.sqrt(n_seeds))

    z0_mean = float(np.mean(z0_seeds))
    z0_std = float(np.std(z0_seeds, ddof=1))

    delta_mean = float(np.mean(delta_seeds))
    delta_std = float(np.std(delta_seeds, ddof=1))
    delta_sem = float(delta_std / np.sqrt(n_seeds))

    amp0_mean = float(np.mean(amp0_seeds))
    amp0_std = float(np.std(amp0_seeds, ddof=1))
    amp0_sem = float(amp0_std / np.sqrt(n_seeds))

    i2_bunch_mean = float(np.mean(i2_bunch_seeds))
    i2_bunch_std = float(np.std(i2_bunch_seeds, ddof=1))

    return {
        "seeds": NOISE_SEEDS,
        "tcrit": tcrit,
        "bq_seeds": bq_seeds,
        "bq_mean": bq_mean,
        "bq_std": bq_std,
        "bq_sem": bq_sem,
        "z0_seeds": z0_seeds,
        "z0_mean": z0_mean,
        "z0_std": z0_std,
        "delta_seeds": delta_seeds,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_sem": delta_sem,
        "amp0_seeds": amp0_seeds,
        "amp0_mean": amp0_mean,
        "amp0_std": amp0_std,
        "amp0_sem": amp0_sem,
        "i2_bunch_seeds": i2_bunch_seeds,
        "i2_bunch_mean": i2_bunch_mean,
        "i2_bunch_std": i2_bunch_std,
        "i1_mean": i1_mean,
    }
