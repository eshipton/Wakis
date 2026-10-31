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

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import epsilon_0, m_e
from scipy.optimize import brentq

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
FIT_WINDOW_LAMBDA_Q = 1.38  # Envelope fit window in units of lambda_q (3 interior nulls) [-]
RISE_PERIODS = 2.0  # C^1 raised-cosine entrance turn-on ramp duration in RF periods [-]
DRIVE_FREQ_HZ = 6.5e9  # AC density-modulation frequency f [Hz]
PLASMA_FREQ_HZ = 0.90e9  # Unreduced injection plasma frequency omega_p / (2 * pi) [Hz]
MODULATION_INDEX = 0.015  # Fractional AC density modulation depth m [-]
GUIDE_BZ_TESLA = 50.0  # Strong axial confining magnetic field B_z [T]
CFL_NUMBER = 0.5  # Courant number cfln [-]

COARSE_GRID = (10, 1, 428)  # (Nx, Ny, Nz) coarse resolution
FINE_GRID = (20, 1, 856)  # (Nx, Ny, Nz) 2x refined resolution
NOISE_SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)  # 8 seeds for empirical 3-sigma noise


class PlanarSheetRng:
    """Seeded antithetic-stratified RNG mapping `BeamInjector` disk draws to a planar sheet.

    `BeamInjector` with ``profile='uniform'`` samples radial and azimuthal arrays via::

        r = radius * sqrt(rng.uniform(0.0, 1.0, size=n_macro))
        phi = rng.uniform(0.0, 2.0 * pi, size=n_macro)
        x = x0 + r * cos(phi)
        y = y0 + r * sin(phi)

    By returning squared antithetic stratified coordinates ``s**2`` on ``[0, 1]``
    on odd calls and ``phi in {0, pi}`` on even calls, ``x`` is distributed
    uniformly across ``[-radius, +radius]`` with one jittered macroparticle per
    stratum and ``y = y0`` identically, without modifying `BeamInjector`.

    Parameters
    ----------
    seed : int, optional
        Deterministic seed for ``numpy.random.default_rng``. Default is ``42``.
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._call_count = 0

    def uniform(
        self, low: float = 0.0, high: float = 1.0, size: int | None = None
    ) -> np.ndarray:
        """Return squared stratified radial draws on odd calls and ``{0, pi}`` on even calls."""
        self._call_count += 1
        n = int(size)
        half = n // 2
        if self._call_count % 2 == 1:
            u = self._rng.uniform(0.0, 1.0, size=half)
            idx = np.arange(half, dtype=np.float64)
            s_pos = (idx + u) / half
            s_neg = (idx + (1.0 - u)) / half
            s = np.concatenate([s_pos, s_neg])
            if n > 2 * half:
                s = np.concatenate([s, self._rng.uniform(0.0, 1.0, size=n - 2 * half)])
            return s * s
        return np.concatenate(
            [
                np.zeros(half, dtype=np.float64),
                np.full(n - half, np.pi, dtype=np.float64),
            ]
        )


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
    nx: int | None = None,
    nz: int | None = None,
) -> tuple[float, float, float]:
    """Compute fundamental `T1`, second-mode `T2`, and signed multi-mode null shift `eps_fit`.

    A uniform top-hat entrance modulation across ``x in [-b, b]`` excites the
    orthogonal Sturm-Liouville slab eigenmodes ``cos(T_n * x)`` (``n = 1, 2, ...``).
    The weak higher-order modes (``n >= 2``, total power weight ~1%) shift the
    standing-wave nulls of ``|I1(z)|**2`` upward in ``beta_q`` relative to the
    pure ``n = 1`` mode. When ``nx`` and ``nz`` are provided, the multi-mode
    standing-wave envelope is evaluated on the discrete axial grid with
    transverse modes restricted to those resolved below the Nyquist limit
    ``T_n * dx < pi`` (on the coarse grid ``T_2 * dx ~ 1.7 rad``, so ``n = 1, 2``
    are resolved; on the fine grid ``n = 1..4`` are resolved), and fitted over
    ``[0, FIT_WINDOW_LAMBDA_Q * lambda_q]`` using :func:`fit_beta_q_from_envelope`
    to obtain the signed correction ``eps_fit = (beta_q_mm - beta_q_1) / beta_q_1``.
    """
    rhs = tau * b / np.tanh(tau * (a - b))
    t_modes = []
    r_modes = []
    w_modes = []
    dx = (2.0 * a / nx) if nx is not None else 0.0
    for n in range(1, 5):
        lo = (n - 1) * np.pi + 1e-12
        hi = (n - 0.5) * np.pi - 1e-12
        tb_n = float(brentq(lambda x: x * np.tan(x) - rhs, lo, hi))
        t_n = tb_n / b
        if nx is not None and n >= 2 and (t_n * dx) >= np.pi:
            continue
        r_n = 1.0 / np.sqrt(1.0 + (t_n / tau) ** 2)
        # Sturm-Liouville norm of phi_n(x) = cos(T_n * x) on the beam support [0, b]
        norm_in = 0.5 * b * (1.0 + np.sin(2.0 * tb_n) / (2.0 * tb_n))
        overlap = np.sin(tb_n) / t_n
        w_n = float((overlap**2) / (b * norm_in))
        t_modes.append(t_n)
        r_modes.append(r_n)
        w_modes.append(w_n)

    w_arr = np.asarray(w_modes, dtype=np.float64)
    w_arr /= np.sum(w_arr)
    bq_modes = (
        np.asarray(r_modes, dtype=np.float64) * omega_p_dep / (gamma_dep**1.5 * u_dep)
    )
    bq_1 = float(bq_modes[0])
    lam_q = 2.0 * np.pi / bq_1
    z_max_fit = FIT_WINDOW_LAMBDA_Q * lam_q

    if nz is not None:
        dz = TUNNEL_LENGTH_M / nz
        z_eval = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    else:
        z_eval = np.linspace(0.0, TUNNEL_LENGTH_M, 4001)
    i1_mm = np.sum(w_arr[:, None] * np.cos(np.outer(bq_modes, z_eval)), axis=0).astype(
        np.complex128
    )
    bq_mm, _ = fit_beta_q_from_envelope(z_eval, i1_mm, bq_1, z_max=z_max_fit)
    eps_fit = float((bq_mm - bq_1) / bq_1)
    t2_val = float(t_modes[1]) if len(t_modes) > 1 else float(t_modes[0])
    return float(t_modes[0]), t2_val, eps_fit


def compute_analytic_parameters(qm_scale: float = 1.0) -> dict[str, float]:
    """Compute continuum space-charge-wave parameters at the DC-depressed equilibrium.

    In an unneutralized planar electron sheet ``x in [-b, b]`` between grounded
    parallel plates at ``x = +-a``, the steady DC space-charge electric field
    ``E_x,DC(x)`` creates a potential trough ``phi_DC(x) < 0`` that slows the
    injected beam from ``u0`` to the cross-section-averaged equilibrium velocity
    ``u_dep = u0 * (1 - delta_u_dc)`` and increases the steady charge density by
    continuity to ``rho_dep = rho0 * (u0 / u_dep)``, where::

        delta_u_dc = (omega_p**2 * b * (a - (2 / 3) * b)) / (gamma0**3 * u0**2)

    Evaluating both `LinearizedBeam` and the analytical planar dispersion
    relation at the depressed equilibrium ``(u_dep, rho_dep, gamma_dep)`` removes
    the leading ``O(delta_u_dc)`` DC velocity-depression bias (Option 3(ii)).

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

    # Continuum 1D planar DC space-charge velocity depression across [-b, b]
    delta_u_dc = (omega_p0**2 * b * (a - (2.0 / 3.0) * b)) / (gamma0**3 * u0**2)
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

    t1_transverse, t2_transverse, eps_fit = _compute_multimode_fit_shift(
        tau, b, a, omega_p_dep, gamma_dep, u_dep
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
        "eps_fit": float(eps_fit),
        "lambda_e": float(2.0 * np.pi / beta_e),
        "lambda_0": float(2.0 * np.pi / k0),
        "lambda_p": float(2.0 * np.pi * u_dep / omega_p_dep),
        "lambda_q": float(2.0 * np.pi / beta_q),
        "I0": float(i0_dc),
        "I1_0": float(i1_entrance),
    }


def _single_grid_budget_terms(
    nx: int, nz: int, params: dict[str, float], eps_noise: float
) -> dict[str, float]:
    """Compute derived discretization and noise terms for a single grid ``(nx, 1, nz)``."""
    dx = 2.0 * TUNNEL_HALF_WIDTH_M / nx
    dz = TUNNEL_LENGTH_M / nz
    b = BEAM_HALF_THICKNESS_M
    a = TUNNEL_HALF_WIDTH_M
    beta_e = params["beta_e"]
    beta_q = params["beta_q"]
    t1 = params["T1"]
    r_fac = params["R"]
    dlnr_dlntau = params["dlnr_dlntau"]
    dlnbq_ddelta = params["dlnbq_ddelta"]
    delta_u_dc = params["delta_u_dc"]
    gamma_dep = params["gamma_dep"]

    # Relativistic factor d(ln tau)/d(ln k) = k^2 / tau^2 = gamma_dep^2 for tau^2 = k^2 - k0^2
    dln_tau_dln_k = float(gamma_dep**2)

    # 0. Discrete signed multi-mode Sturm-Liouville envelope-fit shift on (nx, 1, nz):
    #    Evaluated using modes resolved below transverse Nyquist (T_n * dx < pi;
    #    on coarse grid T_2 * dx ~ 1.7 rad).
    _, t2, eps_fit = _compute_multimode_fit_shift(
        params["tau"],
        b,
        a,
        params["omega_p"],
        gamma_dep,
        params["u_dep"],
        nx=nx,
        nz=nz,
    )

    # Shared SolverFIT3D field-discretization terms (apply to BOTH solvers):
    # (a) 3-point transverse FIT Laplacian under-estimates T1^2 by -(T1*dx)^2/12;
    #     from R^2 = tau^2/(tau^2 + T1^2), d(ln R) = -0.5*(1 - R^2)*d(ln T1^2):
    eps_x_fit = float(0.5 * (1.0 - r_fac**2) * (t1 * dx) ** 2 / 12.0)
    # (b) Staggered Yee axial curl-curl operator has symbol (2/dz)*sin(k*dz/2)
    #     = k*(1 - (k*dz)^2/24), shifting tau and R(tau) downward in both codes
    #     with d(ln tau)/d(ln k) = gamma_dep^2:
    eps_z_fit = float(dlnr_dlntau * dln_tau_dln_k * (beta_e * dz) ** 2 / 24.0)

    # 1. LinearizedBeam Eulerian carrier dispersion + numerical carrier k_car in R(tau)
    #    + signed multi-mode null shift eps_fit:
    eps_z_lb = float(1.0 / np.cos(beta_e * dz) - 1.0)
    eps_tau_lb = float(dlnr_dlntau * dln_tau_dln_k * (beta_e * dz) ** 2 / 6.0)
    eps_lead_lb_signed = float(eps_z_lb + eps_tau_lb - eps_z_fit + eps_x_fit + eps_fit)

    # 2. ParticleSolver transverse CIC deposit/gather smoothing + shared FIT terms
    #    + signed multi-mode null shift eps_fit:
    eps_cic = float(1.0 - np.sqrt(1.0 - (2.0 / 3.0) * np.sin(0.5 * t1 * dx) ** 2))
    eps_lead_part_signed = float(-eps_cic - eps_z_fit + eps_x_fit + eps_fit)

    # 3. DC space-charge depression tolerance and induced beta_q shift budget:
    dep_tol = float(delta_u_dc**2 + (dx / (2.0 * b)) ** 2 * delta_u_dc)
    eps_dc_res = float(dlnbq_ddelta * dep_tol)

    # 4. Step-edge transverse interface quadrature bound at x = +-b in LinearizedBeam:
    #    The jump in phi_1''(x) across the top-hat beam edge x = +-b introduces an
    #    O((dx / (2*b))^2) boundary-cell quadrature error in R^2 = tau^2 / (tau^2 + T1^2):
    eps_quad_x = float(0.5 * (1.0 - r_fac**2) * (dx / (2.0 * b)) ** 2)

    # 5. Residual DFT leakage after integer-period dt + time-mean subtraction
    eps_dft = float(100.0 * np.finfo(np.float64).eps / MODULATION_INDEX)

    # 6. Empirical 3-sigma macroparticle noise measured across 8 seeds on this grid,
    #    and the 8-seed ensemble mean noise (3*sigma/sqrt(8) and 1*sigma/sqrt(8)):
    n_seeds = float(len(NOISE_SEEDS))
    eps_noise_val = float(eps_noise)
    eps_noise_mean = float(eps_noise_val / np.sqrt(n_seeds))
    sigma_mean = float(eps_noise_val / (3.0 * np.sqrt(n_seeds)))

    # Slack terms and total bounds: exact sum of constituent magnitudes
    slack_lb = float(eps_quad_x + eps_dft)
    slack_part = float(eps_dc_res + eps_dft + eps_noise_val)
    slack_part_mean = float(eps_dc_res + eps_dft + eps_noise_mean)
    slack_part_det = float(eps_dc_res + eps_dft)
    tol_lb = float(abs(eps_lead_lb_signed) + slack_lb)
    tol_part = float(abs(eps_lead_part_signed) + slack_part)
    tol_cross = float(tol_lb + tol_part)

    # Entrance amplitude bounds at first primal edge midpoint z_0 = 0.5 * dz:
    tol_amp_lb = float(0.5 * (beta_q * dz) ** 2)
    tol_amp_part = float(0.5 * (beta_q * dz) ** 2 + 0.5 * (dx / (2.0 * b)) ** 2)

    # Heuristic cross-code launch-offset bound |z0_part - z0_lb|:
    # half-cell entrance stagger 0.5 * dz plus cathode-sheath depression scale delta_u_dc / beta_q
    tol_z0 = float(0.5 * dz + delta_u_dc / beta_q)

    # Derived 3-null least-squares residual bounds |z_m - (z0 + (m + 0.5)*pi/beta_q)|:
    # - LinearizedBeam: half-cell axial sampling 0.5*dz plus multi-mode null shift eps_fit*lambda_q
    # - ParticleSolver: adds half-beat noise & DC residual displacement 0.5*(eps_noise + eps_dc_res)*lambda_q
    lam_q = params["lambda_q"]
    tol_null_res_lb = float(0.5 * dz + abs(eps_fit) * lam_q)
    tol_null_res_part = float(
        tol_null_res_lb + 0.5 * (eps_noise_val + eps_dc_res) * lam_q
    )

    # Second-harmonic bounds over z in [0, lambda_q]:
    # - LinearizedBeam: C^1 raised-cosine turn-on ramp (N_rise = RISE_PERIODS) spectral tail
    #   leaking across the N_per = 6-period rectangular DFT window: 1 / ((2*N_rise)^2 * pi * N_per)
    # - ParticleSolver: order-of-magnitude ballistic bunching estimate X(lambda_q) = pi * m
    n_rise = float(RISE_PERIODS)
    n_per = 6.0
    tol_i2_lb = float(1.0 / ((2.0 * n_rise) ** 2 * np.pi * n_per))
    tol_i2_part = float(0.5 * MODULATION_INDEX * beta_q * lam_q)

    return {
        "t2_dx": float(t2 * dx),
        "eps_z_lb": eps_z_lb,
        "eps_tau_lb": eps_tau_lb,
        "eps_z_fit": eps_z_fit,
        "eps_x_fit": eps_x_fit,
        "eps_quad_x": eps_quad_x,
        "eps_fit": eps_fit,
        "eps_lead_lb_signed": eps_lead_lb_signed,
        "eps_cic": eps_cic,
        "eps_lead_part_signed": eps_lead_part_signed,
        "dep_tol": dep_tol,
        "eps_dc_res": eps_dc_res,
        "eps_dft": eps_dft,
        "eps_noise": eps_noise_val,
        "eps_noise_mean": eps_noise_mean,
        "sigma_mean": sigma_mean,
        "slack_lb": slack_lb,
        "slack_part": slack_part,
        "slack_part_mean": slack_part_mean,
        "slack_part_det": slack_part_det,
        "tol_lb": tol_lb,
        "tol_part": tol_part,
        "tol_cross": tol_cross,
        "tol_amp_lb": tol_amp_lb,
        "tol_amp_part": tol_amp_part,
        "tol_z0": tol_z0,
        "tol_null_res_lb": tol_null_res_lb,
        "tol_null_res_part": tol_null_res_part,
        "tol_i2_lb": tol_i2_lb,
        "tol_i2_part": tol_i2_part,
        "tol_i2_ratio": tol_i2_part,
    }


def derived_tolerances(
    nx: int,
    nz: int,
    params: dict[str, float],
    eps_noise_coarse: float,
    eps_noise_fine: float | None = None,
) -> dict[str, float]:
    """Build the error budget and convergence-order bands term by term from analytical derivations.

    Every budget term is computed without ad-hoc multipliers:

    1. **Shared `SolverFIT3D` field operators (`eps_x_fit`, `eps_z_fit`)**:
       Both solvers couple to the same `SolverFIT3D` grid. From
       ``R**2 = tau**2 / (tau**2 + T1**2)``, a relative error ``-(T1 * dx)**2 / 12``
       in the 3-point transverse FIT Laplacian ``T1**2`` shifts ``beta_q`` upward
       in both codes by ``eps_x_fit = +0.5 * (1 - R**2) * (T1 * dx)**2 / 12``.
       Simultaneously, the staggered Yee axial curl-curl stencil has symbol
       ``(2 / dz) * sin(k * dz / 2) = k * (1 - (k * dz)**2 / 24)``, shifting
       ``R(tau)`` downward in both codes by
       ``eps_z_fit = (d ln R / d ln tau) * gamma_dep**2 * (beta_e * dz)**2 / 24``.
    2. **Discrete signed multi-mode envelope-fit shift (`eps_fit`)**:
       Higher-order Sturm-Liouville slab modes ``n >= 2`` resolved on the grid
       (``T_n * dx < pi``; on the coarse grid ``T_2 * dx ~ 1.7 rad``) shift the
       standing-wave nulls upward in ``beta_q`` by ``eps_fit``, which is included
       directly in ``eps_lead_lb_signed`` and ``eps_lead_part_signed``.
    3. **Eulerian dispersion of `LinearizedBeam` (`eps_z_lb`, `eps_tau_lb`)**:
       Centered axial differences ``(f_{k+1} - f_{k-1}) / (2 * dz)`` give a
       carrier group velocity ``u_g = u_dep * cos(beta_e * dz)``, shifting
       ``beta_q = omega_q / u_g`` upward by ``eps_z_lb = 1 / cos(beta_e * dz) - 1``.
       In addition, `SolverFIT3D` sees `LinearizedBeam`'s numerical carrier
       wavenumber ``k_car = arcsin(beta_e * dz) / dz = beta_e * (1 + (beta_e * dz)**2 / 6)``,
       which increases ``tau`` and shifts ``R(tau)`` upward by
       ``eps_tau_lb = (d ln R / d ln tau) * gamma_dep**2 * (beta_e * dz)**2 / 6``.
       Thus ``eps_lead_lb_signed = +eps_z_lb + eps_tau_lb - eps_z_fit + eps_x_fit + eps_fit``.
    4. **CIC smoothing in `ParticleSolver` (`eps_cic`)**:
       Staggered axial CIC gather/Esirkepov deposit has ``kappa_z**2 == 1``
       (`wakis/particles/solver.py:420`), while nodal transverse CIC
       deposit/gather (``p = p_g = 1``) attenuates ``T1`` by
       ``kappa_x**2 = 1 - (2 / 3) * sin(T1 * dx / 2)**2``, shifting ``beta_q``
       downward by ``eps_cic = 1 - sqrt(1 - (2 / 3) * sin(T1 * dx / 2)**2)``.
       Combined with the shared field-solver and multi-mode terms,
       ``eps_lead_part_signed = -eps_cic - eps_z_fit + eps_x_fit + eps_fit``.
    5. **DC space-charge depression tolerance (`dep_tol`, `eps_dc_res`)**:
       Higher-order self-consistent density enhancement ``delta_u_dc**2`` plus
       transverse grid/CIC quadrature ``(dx / (2 * b))**2 * delta_u_dc`` bounds
       the DC depression residual by ``dep_tol``, which shifts ``beta_q`` by at
       most ``eps_dc_res = (d ln beta_q / d delta_u) * dep_tol``.
    6. **Residual DFT leakage (`eps_dft`)**:
       With ``dt = 1 / (f * N_spp)`` chosen so the 6-period DFT window contains
       an exact integer number of steps ``6 * N_spp`` and per-edge mean
       subtraction applied before the DFT, residual DC leakage is bounded by
       ``eps_dft = 100 * eps_mach / m``.
    7. **Empirical macroparticle noise (`eps_noise`, `eps_noise_mean`)**:
       Measured ``3 * sigma / beta_q`` across 8 independent transverse RNG seeds
       on each grid (`eps_noise_coarse` on `COARSE_GRID`, `eps_noise_fine` on
       `FINE_GRID`), with 8-seed mean uncertainty ``eps_noise / sqrt(8)``.
    8. **Derived convergence-order bands**:
       Computed from the signed predictions ``e_c, e_f`` and slacks ``s_c, s_f``
       (including 8-seed mean noise ``slack_part_mean`` for `ParticleSolver`).

    Parameters
    ----------
    nx, nz : int
        Transverse and axial grid cell counts.
    params : dict of str to float
        Output of :func:`compute_analytic_parameters`.
    eps_noise_coarse : float
        Empirical ``3 * sigma / beta_q`` measured across 8 seeds on `COARSE_GRID`.
    eps_noise_fine : float, optional
        Empirical ``3 * sigma / beta_q`` measured across 8 seeds on `FINE_GRID`.
        If omitted, defaults to ``eps_noise_coarse`` when ``(nx, nz) == COARSE_GRID[::2]``
        or the cell-count scaling estimate otherwise.

    Returns
    -------
    dict of str to float
        Dictionary containing every individual budget term, total bounds, and
        derived convergence-order bands.
    """
    if eps_noise_fine is None:
        eps_noise_fine = float(
            eps_noise_coarse
            * np.sqrt(
                (COARSE_GRID[0] / FINE_GRID[0]) * (COARSE_GRID[2] / FINE_GRID[2])
            )
        )
    eps_noise_this = eps_noise_coarse if nx == COARSE_GRID[0] else eps_noise_fine

    terms = _single_grid_budget_terms(nx, nz, params, eps_noise_this)
    tc = _single_grid_budget_terms(
        COARSE_GRID[0], COARSE_GRID[2], params, eps_noise_coarse
    )
    tf = _single_grid_budget_terms(FINE_GRID[0], FINE_GRID[2], params, eps_noise_fine)

    # Derived 2-grid convergence-order bands p in log2((|e_c| -+ s_c) / (|e_f| +- s_f))
    e_lb_c, s_lb_c = abs(tc["eps_lead_lb_signed"]), tc["slack_lb"]
    e_lb_f, s_lb_f = abs(tf["eps_lead_lb_signed"]), tf["slack_lb"]
    order_lb_min = float(np.log2((e_lb_c - s_lb_c) / (e_lb_f + s_lb_f)))
    order_lb_max = float(np.log2((e_lb_c + s_lb_c) / (e_lb_f - s_lb_f)))

    # ParticleSolver 8-seed mean convergence band including 8-seed mean noise slack_part_mean
    e_part_c, s_part_c = abs(tc["eps_lead_part_signed"]), tc["slack_part_mean"]
    e_part_f, s_part_f = abs(tf["eps_lead_part_signed"]), tf["slack_part_mean"]
    num_min = max(e_part_c - s_part_c, 1e-12)
    den_min = max(e_part_f - s_part_f, 1e-12)
    order_part_min = float(np.log2(num_min / (e_part_f + s_part_f)))
    order_part_max = float(np.log2((e_part_c + s_part_c) / den_min))

    # Deterministic-only particle band (without seed noise) for reference
    order_part_det_min = float(
        np.log2(
            (e_part_c - tc["slack_part_det"]) / (e_part_f + tf["slack_part_det"])
        )
    )
    order_part_det_max = float(
        np.log2(
            (e_part_c + tc["slack_part_det"])
            / max(e_part_f - tf["slack_part_det"], 1e-12)
        )
    )

    # Cross-code band (uninformative because noise + slacks span nearly three octaves)
    e_cross_c = abs(tc["eps_lead_part_signed"] - tc["eps_lead_lb_signed"]) / (
        1.0 + tc["eps_lead_lb_signed"]
    )
    e_cross_f = abs(tf["eps_lead_part_signed"] - tf["eps_lead_lb_signed"]) / (
        1.0 + tf["eps_lead_lb_signed"]
    )
    s_cross_c = tc["slack_lb"] + tc["slack_part_mean"]
    s_cross_f = tf["slack_lb"] + tf["slack_part_mean"]
    order_cross_min = float(
        np.log2(max(e_cross_c - s_cross_c, 1e-12) / (e_cross_f + s_cross_f))
    )
    order_cross_max = float(
        np.log2((e_cross_c + s_cross_c) / max(e_cross_f - s_cross_f, 1e-12))
    )

    terms.update(
        {
            "order_lb_min": order_lb_min,
            "order_lb_max": order_lb_max,
            "order_part_min": order_part_min,
            "order_part_max": order_part_max,
            "order_part_det_min": order_part_det_min,
            "order_part_det_max": order_part_det_max,
            "order_cross_min": order_cross_min,
            "order_cross_max": order_cross_max,
        }
    )
    return terms


def format_error_budget(nx: int, nz: int, tols: dict[str, float]) -> str:
    """Format every budget term and asserted bound for grid ``(nx, 1, nz)``."""
    lines = [
        f"Error budget for grid ({nx}, 1, {nz}):",
        f"  eps_z_lb (1/cos(beta_e*dz) - 1)                   : {tols['eps_z_lb']:.6%}",
        f"  eps_tau_lb (dlnR/dlntau*gamma^2*(beta_e*dz)^2/6)  : {tols['eps_tau_lb']:.6%}",
        f"  eps_z_fit (dlnR/dlntau*gamma^2*(beta_e*dz)^2/24)  : {tols['eps_z_fit']:.6%}",
        f"  eps_x_fit (0.5*(1-R^2)*(T1*dx)^2/12)              : {tols['eps_x_fit']:.6%}",
        f"  eps_fit (discrete signed multi-mode null shift)   : {tols['eps_fit']:+.6%} (T2*dx={tols['t2_dx']:.3f} rad)",
        f"  eps_lead_lb_signed                                : {tols['eps_lead_lb_signed']:+.6%}",
        f"  eps_cic (transverse CIC)                          : {tols['eps_cic']:.6%}",
        f"  eps_lead_part_signed                              : {tols['eps_lead_part_signed']:+.6%}",
        f"  dep_tol (DC depression bound)                     : {tols['dep_tol']:.6%}",
        f"  eps_dc_res (dlnbq/ddelta * dep_tol)               : {tols['eps_dc_res']:.6%}",
        f"  eps_quad_x (0.5*(1-R^2)*(dx/(2b))^2)              : {tols['eps_quad_x']:.6%}",
        f"  eps_dft (residual DC leakage)                     : {tols['eps_dft']:.6e}",
        f"  eps_noise (measured 8-seed 3*sigma)               : {tols['eps_noise']:.6%}",
        f"  eps_noise_mean (measured 3*sigma/sqrt(8))         : {tols['eps_noise_mean']:.6%} (1*sigma/sqrt(8)={tols['sigma_mean']:.6%})",
        f"  slack_lb (eps_quad_x + eps_dft)                   : {tols['slack_lb']:.6%}",
        f"  slack_part_mean (eps_dc_res + eps_dft + 3s/sqrt8) : {tols['slack_part_mean']:.6%}",
        f"  tol_lb (asserted LB bound)                        : {tols['tol_lb']:.6%}",
        f"  tol_part (asserted single-seed Part bound)        : {tols['tol_part']:.6%}",
        f"  tol_cross (asserted cross bound)                  : {tols['tol_cross']:.6%}",
        f"  tol_amp_lb (entrance |I1(0)|)                     : {tols['tol_amp_lb']:.6e}",
        f"  tol_amp_part (entrance |I1(0)|)                   : {tols['tol_amp_part']:.6e}",
        f"  tol_z0 (heuristic |z0_part - z0_lb|)              : {tols['tol_z0'] * 1e3:.4f} mm",
        f"  tol_null_res_lb (derived 3-null residual bound)   : {tols['tol_null_res_lb'] * 1e3:.4f} mm",
        f"  tol_null_res_part (derived 3-null residual bound) : {tols['tol_null_res_part'] * 1e3:.4f} mm",
        f"  tol_i2_lb (2*omega ramp transient)                : {tols['tol_i2_lb']:.6%}",
        f"  tol_i2_part (order-of-magnitude pi*m estimate)    : {tols['tol_i2_part']:.6%}",
        f"  order_lb band                                     : [{tols['order_lb_min']:.3f}, {tols['order_lb_max']:.3f}]",
        f"  order_part band (with 8-seed mean noise)          : [{tols['order_part_min']:.3f}, {tols['order_part_max']:.3f}]",
        f"  order_cross band (uninformative)                  : [{tols['order_cross_min']:.3f}, {tols['order_cross_max']:.3f}]",
    ]
    return "\n".join(lines)


def fit_beta_q_from_envelope(
    z_coords: np.ndarray,
    i1_phasor: np.ndarray,
    beta_q_guess: float,
    z_max: float | None = None,
    return_residuals: bool = False,
) -> tuple[float, float] | tuple[float, float, np.ndarray, np.ndarray]:
    """Fit ``(beta_q, z0)`` by least squares over all standing-wave nulls of ``|I1(z)|**2``.

    For an entrance density modulation with effective launch offset ``z0``,
    ``|I1(z)|**2 = A * cos(2 * beta_q * (z - z0)) + c`` has standing-wave nulls
    at ``z_m = z0 + (m + 1/2) * pi / beta_q`` (``m = 0, 1, 2, ...``). First obtains
    a rough period ``pi / beta_q_rough`` and phase ``z0_rough`` via 3-parameter
    harmonic least squares, then locates each interior null in ``[0, z_max]`` using
    a local quadratic least-squares fit ``c_2 * dz**2 + c_1 * dz + c_0`` of
    ``|I1(z)|**2`` around the minimum (with convexity ``c_2 > 0``, vertex-offset
    outlier check, and linear-lattice prediction for ``m >= 2`` so noisy cells
    cannot jump the vertex by 2 cells), and fits the linear model
    ``z_m = z0 + (m + 0.5) * L_half`` (``L_half = pi / beta_q``) by ordinary
    least squares over all detected nulls in the window.

    Parameters
    ----------
    z_coords : np.ndarray
        Axial edge midpoint coordinates [m].
    i1_phasor : np.ndarray
        Complex fundamental axial current phasor ``I1(z)`` [A].
    beta_q_guess : float
        Reference reduced plasma wavenumber [rad/m] centering the initial scan.
    z_max : float, optional
        Upper axial limit [m] of the fit window. Defaults to
        ``FIT_WINDOW_LAMBDA_Q * 2 * pi / beta_q_guess`` (covering 3 interior nulls).
    return_residuals : bool, optional
        If ``True``, also return the array of detected null positions ``z_nulls`` [m]
        and least-squares residuals ``z_nulls - z_fit`` [m]. Default is ``False``.

    Returns
    -------
    beta_q_fit : float
        Fitted reduced plasma wavenumber ``beta_q`` [rad/m].
    z0_fit : float
        Fitted effective envelope launch offset ``z0`` [m].
    z_nulls : np.ndarray, optional
        Detected standing-wave null coordinates [m] (returned when ``return_residuals=True``).
    residuals : np.ndarray, optional
        Least-squares null residuals ``z_m - (z0 + (m + 0.5) * pi / beta_q)`` [m]
        (returned when ``return_residuals=True``).
    """
    if z_max is None:
        z_max = FIT_WINDOW_LAMBDA_Q * (2.0 * np.pi / beta_q_guess)
    mask = z_coords <= z_max
    zm = z_coords[mask]
    y = np.abs(i1_phasor[mask]) ** 2
    dz = float(zm[1] - zm[0])

    bq_grid = np.linspace(0.50 * beta_q_guess, 1.80 * beta_q_guess, 2601)
    best_rss = np.inf
    bq_rough = float(beta_q_guess)
    z0_rough = 0.0
    ones = np.ones_like(zm)
    for bq in bq_grid:
        x_mat = np.column_stack([np.cos(2.0 * bq * zm), np.sin(2.0 * bq * zm), ones])
        coef, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
        res = y - x_mat @ coef
        rss = float(np.dot(res, res))
        if rss < best_rss:
            best_rss = rss
            bq_rough = float(bq)
            z0_rough = float(0.5 * np.arctan2(coef[1], coef[0]) / bq)

    l_half = np.pi / bq_rough
    nulls: list[float] = []
    m_indices: list[int] = []
    for m_idx in range(6):
        if len(nulls) >= 2:
            # Linear extrapolation from the already-located nulls prevents drift at m >= 2
            m_prev = np.asarray(m_indices, dtype=np.float64) + 0.5
            z_prev = np.asarray(nulls, dtype=np.float64)
            p_prev, _, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones_like(m_prev), m_prev]), z_prev, rcond=None
            )
            z_center = float(p_prev[0] + (m_idx + 0.5) * p_prev[1])
            l_half_cur = float(p_prev[1])
            search_half = 0.14 * l_half_cur
        else:
            z_center = z0_rough + (m_idx + 0.5) * l_half
            l_half_cur = l_half
            search_half = 0.30 * l_half_cur

        if 0.15 * l_half <= z_center <= z_max - 0.06 * l_half:
            w_mask = np.abs(zm - z_center) <= search_half
            idxs = np.where(w_mask)[0]
            if len(idxs) >= 5:
                # Smooth 3-point moving average to suppress single-cell noise spikes
                # before centering the local quadratic least-squares fit
                k_min = int(idxs[np.argmin(y[idxs])])
                fit_mask = np.abs(zm - zm[k_min]) <= max(0.10 * l_half_cur, 3.5 * dz)
                fit_idxs = np.where(fit_mask)[0]
                if len(fit_idxs) >= 5:
                    z_loc = zm[fit_idxs] - zm[k_min]
                    y_loc = y[fit_idxs]
                    q_mat = np.column_stack([z_loc**2, z_loc, np.ones_like(z_loc)])
                    q_coef, _, _, _ = np.linalg.lstsq(q_mat, y_loc, rcond=None)
                    c2, c1 = float(q_coef[0]), float(q_coef[1])
                    if c2 > 0.0:
                        delta_z = -0.5 * c1 / c2
                        if abs(delta_z) <= max(0.08 * l_half_cur, 2.5 * dz):
                            z_v = float(zm[k_min] + delta_z)
                        else:
                            z_v = float(zm[k_min])
                    else:
                        z_v = float(zm[k_min])
                else:
                    z_v = float(zm[k_min])
                # Outlier check against jumps > 0.08 * L_half from linear lattice for m >= 2
                if len(nulls) >= 2 and abs(z_v - z_center) > 0.08 * l_half_cur:
                    w_tight = np.abs(zm - z_center) <= 0.08 * l_half_cur
                    t_idxs = np.where(w_tight)[0]
                    if len(t_idxs) >= 3:
                        z_loc = zm[t_idxs] - z_center
                        y_loc = y[t_idxs]
                        q_mat = np.column_stack([z_loc**2, z_loc, np.ones_like(z_loc)])
                        q_coef, _, _, _ = np.linalg.lstsq(q_mat, y_loc, rcond=None)
                        c2, c1 = float(q_coef[0]), float(q_coef[1])
                        if c2 > 0.0 and abs(-0.5 * c1 / c2) <= 0.08 * l_half_cur:
                            z_v = float(z_center - 0.5 * c1 / c2)
                nulls.append(z_v)
                m_indices.append(m_idx)

    z_arr = np.asarray(nulls, dtype=np.float64)
    if len(z_arr) >= 2:
        m_arr = np.asarray(m_indices, dtype=np.float64) + 0.5
        design = np.column_stack([np.ones_like(m_arr), m_arr])
        params_ls, _, _, _ = np.linalg.lstsq(design, z_arr, rcond=None)
        z0_fit = float(params_ls[0])
        l_half_fit = float(params_ls[1])
        bq_fit = float(np.pi / l_half_fit)
        residuals = z_arr - (z0_fit + m_arr * l_half_fit)
        if return_residuals:
            return bq_fit, z0_fit, z_arr, residuals
        return bq_fit, z0_fit
    if return_residuals:
        return bq_rough, z0_rough, z_arr, np.zeros_like(z_arr)
    return bq_rough, z0_rough


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


def _integer_period_dt(grid: GridFIT3D) -> tuple[float, int]:
    """Choose ``dt <= dt_cfl`` so ``1 / (DRIVE_FREQ_HZ * dt)`` is an exact integer `n_spp`."""
    s_tmp = SolverFIT3D(
        grid,
        cfln=CFL_NUMBER,
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

    Returns
    -------
    dict of str to Any
        Dictionary with keys ``beam``, ``z_coords``, ``i1_lb``, ``i2_lb``,
        ``dc_leakage_lb``, ``dt``, and ``n_spp``.
    """
    params = compute_analytic_parameters()
    u_dep = params["u_dep"]
    gamma_dep = params["gamma_dep"]
    omega = params["omega"]
    rho_dep = params["rho_dep"]
    rho0_val = rho0_scale * rho_dep
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
    e_ext: Any = None,
) -> dict[str, Any]:
    """Run `ParticleSolver` + `BeamInjector` on the identical drift-tunnel grid.

    Integrates the Esirkepov-deposited particle current density ``ps.J_particles``
    over the exact same primal ``E_z`` edges (``beam_ref.flat_idx``) and dual
    areas (``beam_ref.dA_perp``) used by `LinearizedBeam.axial_current`, with
    identical integer-period ``dt`` and time-mean subtraction before the DFT.
    Also measures the interior DC velocity ``<u_z>(z)`` and deposited particle
    charge density ``rho_DC(z)`` time-averaged over one RF period across the
    settled interior ``z in [0.2 * Lz, 0.8 * Lz]``.

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
        Seed for :class:`PlanarSheetRng`. Default is ``42``.
    m_mod : float, optional
        Fractional AC density modulation depth ``m``. Default is ``MODULATION_INDEX``.
    e_ext : tuple or callable, optional
        Optional external electric field ``E_ext`` passed to `ParticleSolver`
        (used in the mutation test to cancel the DC transverse self-potential
        shear across the sheet beam). Default is ``None``.

    Returns
    -------
    dict of str to Any
        Dictionary with keys ``i1_part``, ``i2_part``, ``dc_leakage_part``,
        ``u_z_mean``, ``rho_dc_mean``, ``u_z_profile``, ``rho_dc_profile``,
        ``dt``, and ``n_spp``.
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

    sp = Species(name="electron", charge=-e_charge, mass=m_e / qm_scale)

    def current_modulation(t: float) -> float:
        ramp_t = 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))
        return i0_dc * (1.0 + m_mod * ramp_t * np.sin(omega * t))

    n_macro = 16 if nx == COARSE_GRID[0] else 32
    inj = BeamInjector(
        species=sp,
        z0=0.0,
        current=current_modulation,
        gamma=gamma0,
        profile="uniform",
        radius=b,
        n_macro_per_step=n_macro,
        seed=seed,
    )
    inj.rng = PlanarSheetRng(seed=seed)
    ps = ParticleSolver(
        solver,
        species=[sp],
        injectors=[inj],
        shape=1,
        E_ext=e_ext,
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
        "u_z_mean": u_z_mean,
        "rho_dc_mean": rho_dc_mean,
        "u_z_profile": u_z_profile,
        "rho_dc_profile": rho_dc_profile,
        "dt": dt,
        "n_spp": n_spp,
    }


@lru_cache(maxsize=4)
def cached_baseline_pair(
    nx: int, nz: int, m_mod: float = MODULATION_INDEX
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cache baseline `(LinearizedBeam, ParticleSolver)` runs for `(nx, nz, m_mod)`."""
    lb_res = run_linearized_beam_drift(nx, nz, rho0_scale=1.0, m_mod=m_mod)
    part_res = run_particle_beam_drift(
        nx, nz, beam_ref=lb_res["beam"], qm_scale=1.0, seed=NOISE_SEEDS[0], m_mod=m_mod
    )
    return lb_res, part_res


@lru_cache(maxsize=4)
def measure_empirical_noise(
    grid: tuple[int, int, int] = COARSE_GRID,
    m_mod: float = MODULATION_INDEX,
) -> dict[str, Any]:
    """Measure empirical 8-seed `beta_q` and coherent-bunching `I2` statistics.

    Runs seeds ``42..49`` on ``grid`` at modulation depth ``m_mod`` (reusing seed
    ``42`` from :func:`cached_baseline_pair` and executing the remaining 7 seeds
    via `ThreadPoolExecutor`), fits ``(beta_q, z0)`` over all 3 standing-wave nulls
    in ``[0, FIT_WINDOW_LAMBDA_Q * lambda_q]`` for each seed, and extracts the
    coherent ballistic-bunching second-harmonic ratio ``|I2_bunch| / |I1(0)|``.
    """
    params = compute_analytic_parameters()
    nx, _, nz = grid
    lb_res, part_seed0 = cached_baseline_pair(nx, nz, m_mod=m_mod)
    beam_ref = lb_res["beam"]
    z_coords = lb_res["z_coords"]
    beta_e = params["beta_e"]
    beta_q_th = params["beta_q"]
    lam_q = params["lambda_q"]
    z_max_fit = FIT_WINDOW_LAMBDA_Q * lam_q

    def _run_seed(
        seed: int,
    ) -> tuple[float, float, np.ndarray, np.ndarray, float, np.ndarray]:
        if seed == NOISE_SEEDS[0]:
            res = part_seed0
        else:
            res = run_particle_beam_drift(
                nx, nz, beam_ref=beam_ref, qm_scale=1.0, seed=seed, m_mod=m_mod
            )
        i1 = res["i1_part"]
        i2 = res["i2_part"]
        bq_s, z0_s, nulls_s, res_s = fit_beta_q_from_envelope(
            z_coords, i1, beta_q_th, z_max=z_max_fit, return_residuals=True
        )
        i2_bunch_s = extract_coherent_bunching_i2_ratio(
            z_coords, i1, i2, beta_e, beta_q_th, z_max=lam_q
        )
        return bq_s, z0_s, nulls_s, res_s, i2_bunch_s, i1

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_run_seed, NOISE_SEEDS))

    bq_seeds = np.array([r[0] for r in results], dtype=np.float64)
    z0_seeds = np.array([r[1] for r in results], dtype=np.float64)
    nulls_seeds = [r[2] for r in results]
    res_seeds = [r[3] for r in results]
    i2_bunch_seeds = np.array([r[4] for r in results], dtype=np.float64)
    i1_mean = np.mean(np.stack([r[5] for r in results], axis=0), axis=0)

    bq_mean = float(np.mean(bq_seeds))
    bq_std = float(np.std(bq_seeds, ddof=1))
    z0_mean = float(np.mean(z0_seeds))
    z0_std = float(np.std(z0_seeds, ddof=1))
    eps_noise = float(3.0 * bq_std / beta_q_th)
    eps_noise_mean = float(eps_noise / np.sqrt(len(NOISE_SEEDS)))
    sigma_mean = float(bq_std / (beta_q_th * np.sqrt(len(NOISE_SEEDS))))

    i2_bunch_mean = float(np.mean(i2_bunch_seeds))
    i2_bunch_std = float(np.std(i2_bunch_seeds, ddof=1))
    eps_i2_noise = float(3.0 * i2_bunch_std)

    return {
        "seeds": NOISE_SEEDS,
        "bq_seeds": bq_seeds,
        "z0_seeds": z0_seeds,
        "nulls_seeds": nulls_seeds,
        "res_seeds": res_seeds,
        "bq_mean": bq_mean,
        "bq_std": bq_std,
        "z0_mean": z0_mean,
        "z0_std": z0_std,
        "eps_noise": eps_noise,
        "eps_noise_mean": eps_noise_mean,
        "sigma_mean": sigma_mean,
        "i2_bunch_seeds": i2_bunch_seeds,
        "i2_bunch_mean": i2_bunch_mean,
        "i2_bunch_std": i2_bunch_std,
        "eps_i2_noise": eps_i2_noise,
        "i1_mean": i1_mean,
    }
