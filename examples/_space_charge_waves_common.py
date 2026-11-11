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

import gc
import hashlib
import os
import pickle
import tempfile
from functools import lru_cache
from pathlib import Path
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
from wakis.particles.shapes import bspline_weights_1d

try:
    import cupy as cp

    USE_GPU = bool(cp.cuda.is_available())
    _GPU_BIN_KERNEL = (
        cp.RawKernel(
            r"""
extern "C" __global__ void bin_beam_moments(
    const int n_p,
    const int nx,
    const int nz,
    const double dz,
    const double c2,
    const double gamma0,
    const double q_over_mc2,
    const double* __restrict__ w_mat,
    const double* __restrict__ zp,
    const double* __restrict__ uxp,
    const double* __restrict__ uyp,
    const double* __restrict__ uzp,
    const double* __restrict__ wp,
    const double* __restrict__ phi_nodes,
    double* __restrict__ out_bins
) {
    extern __shared__ double smem[];
    double* s_w   = smem;
    double* s_wv  = smem + nz;
    double* s_wv2 = smem + 2 * nz;
    double* s_wg  = smem + 3 * nz;
    double* s_wgp = smem + 4 * nz;

    for (int j = threadIdx.x; j < 5 * nz; j += blockDim.x) {
        smem[j] = 0.0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n_p; i += stride) {
        double z = zp[i];
        int b = (int)floor(z / dz);
        if (b < 0) b = 0;
        else if (b >= nz) b = nz - 1;
        if (b < nz - 1 && z >= (double)(b + 1) * dz) b += 1;

        double ux = uxp[i], uy = uyp[i], uz = uzp[i];
        double w = wp[i];
        double gamma = sqrt(1.0 + (ux*ux + uy*uy + uz*uz) / c2);
        double vz = uz / gamma;

        const double* w_row = w_mat + i * (nx + 1);
        double phi = 0.0;
        for (int ix = 1; ix < nx; ++ix) {
            phi += w_row[ix] * phi_nodes[ix * nz + b];
        }
        double gamma_pred = gamma0 + q_over_mc2 * phi;

        atomicAdd(&s_w[b], w);
        atomicAdd(&s_wv[b], w * vz);
        atomicAdd(&s_wv2[b], w * vz * vz);
        atomicAdd(&s_wg[b], w * gamma);
        atomicAdd(&s_wgp[b], w * gamma_pred);
    }
    __syncthreads();

    for (int j = threadIdx.x; j < 5 * nz; j += blockDim.x) {
        atomicAdd(&out_bins[j], smem[j]);
    }
}
""",
            "bin_beam_moments",
        )
        if USE_GPU
        else None
    )
except ImportError:
    cp = None
    USE_GPU = False
    _GPU_BIN_KERNEL = None


# Physical and geometric parameters for the planar sheet-beam drift tunnel
V0_VOLTS = 10.0e3  # DC beam acceleration voltage [V]
TUNNEL_HALF_WIDTH_M = 2.0e-3  # Parallel-plate tunnel half-width a [m]
BEAM_HALF_THICKNESS_M = 0.8e-3  # Sheet-beam half-thickness b [m]
PERIODIC_LY_M = 1.0e-3  # Periodic transverse depth Ly [m]
TUNNEL_LENGTH_M = (
    0.186 * 474 / 428
)  # Lz ~ 0.206 m >= 1.6 * lambda_q, keeps dz = 0.186/428 exactly [m]
OUTFLOW_RAMP_M = 8.0e-3  # Outflow axial taper length [m]
EPS_WINDOW = (
    1.0e-5  # Relative evanescent near-field amplitude allowed at fit-window edge [-]
)
RISE_PERIODS = 2.0  # C^1 raised-cosine entrance turn-on ramp duration in RF periods [-]
DRIVE_FREQ_HZ = 6.5e9  # AC density-modulation frequency f [Hz]
PLASMA_FREQ_HZ = 0.90e9  # Unreduced injection plasma frequency omega_p / (2 * pi) [Hz]
MODULATION_INDEX = 0.015  # Fractional AC density modulation depth m [-]
GUIDE_BZ_TESLA = 50.0  # Strong axial confining magnetic field B_z [T]
CFL_NUMBER = 0.5  # Courant number cfln [-]
PARTICLE_SHAPE = 2  # B-spline shape order p of the particle runs [-]
PARTICLE_FILTER = None  # current filter of the particle runs (None or "binomial")
SETTLE_FACTOR = None  # override: settle = SETTLE_FACTOR * Lz / (u_dep * dt); None = derived settle_steps() [-]

COARSE_GRID = (10, 1, 474)  # (Nx, Ny, Nz) coarse resolution
FINE_GRID = (20, 1, 948)  # (Nx, Ny, Nz) 2x refined resolution
NOISE_SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)  # 8 seeds for empirical noise


def transverse_weight_matrix(
    x_lines: np.ndarray, nx: int, p: int = PARTICLE_SHAPE
) -> np.ndarray:
    """Evaluate order-``p`` primal B-spline transverse weights on ``[-a, a]`` with PEC walls.

    Both the Esirkepov ``J_z`` (and ``rho``) deposit and the Galerkin ``E_z`` gather
    on ``z``-edges at transverse primal ``x``-nodes use the order-``p`` nodal
    B-spline ``bspline_weights_1d(xi, p, staggered=False)`` with
    ``xi = (x + a) / dx``. Weights landing on PEC wall nodes ``x = +-a``
    (indices ``0`` and ``nx``) or outside ``(-a, a)`` are dropped because
    ``E_z(+-a) = 0``, matching ``_wrap_or_mask_1d`` for non-periodic axes.

    Parameters
    ----------
    x_lines : np.ndarray
        Transverse line or particle ``x`` positions [m] of shape ``(n_lines,)``.
    nx : int
        Transverse cell count across ``[-a, a]``.
    p : int, optional
        B-spline shape order ``p`` in ``{1, 2, 3}``. Default is ``PARTICLE_SHAPE``.

    Returns
    -------
    np.ndarray
        Weight matrix of shape ``(len(x_lines), nx + 1)`` with zeros in columns
        ``0`` and ``nx``.
    """
    xp_mod = cp if (cp is not None and isinstance(x_lines, cp.ndarray)) else np
    a = TUNNEL_HALF_WIDTH_M
    dx = 2.0 * a / float(nx)
    x_arr = xp_mod.asarray(x_lines, dtype=xp_mod.float64)
    xi = (x_arr + a) / dx
    i0, weights = bspline_weights_1d(xi, int(p), staggered=False, xp=xp_mod)
    n_lines = len(x_arr)
    w_mat = xp_mod.zeros((n_lines, int(nx) + 1), dtype=xp_mod.float64)
    if n_lines == 0:
        return w_mat
    x_nodes = xp_mod.linspace(-a, a, int(nx) + 1) if int(p) == 1 else None
    rows = xp_mod.arange(n_lines, dtype=xp_mod.int64)
    for offset, w_m in enumerate(weights):
        node_idx = i0 + offset
        valid = (node_idx >= 1) & (node_idx < int(nx))
        if int(p) == 1:
            # Evaluate W_1(|x - x_node| / dx) via bspline_weights_1d from the node-relative
            # distance so p=1 avoids integer-offset (xi - i0) cancellation and matches the
            # node-coordinate formula bit-for-bit.
            dist_norm = xp_mod.minimum(
                xp_mod.abs(x_arr[valid] - x_nodes[node_idx[valid]]) / dx, 1.0
            )
            _, w_node_1d = bspline_weights_1d(dist_norm, 1, staggered=False, xp=xp_mod)
            w_val = xp_mod.where(dist_norm < 1.0, w_node_1d[0], 0.0)
        else:
            w_val = w_m[valid]
        w_mat[rows[valid], node_idx[valid]] += w_val
    return w_mat


def _grid_gamma1_disc_and_dw(
    nx: int, nz: int, omega: float
) -> tuple[float, float, float]:
    """Compute discrete evanescent decay rate ``gamma1_disc`` [1/m] and window margin ``d_w_grid`` [m].

    On grid ``(nx, 1, nz)`` with ``dx = 2 * a / nx``, ``dz = Lz / nz``, and integer-period
    time step ``dt``, evaluates::

        lambda_x = (2 / dx)**2 * sin(pi * dx / (4 * a))**2
        Omega_f = (2 / dt) * sin(omega * dt / 2)
        gamma1_disc = (2 / dz) * arcsinh((dz / 2) * sqrt(lambda_x - (Omega_f / c)**2))
        d_w_grid = ln(1 / EPS_WINDOW) / gamma1_disc
    """
    a = TUNNEL_HALF_WIDTH_M
    dx = 2.0 * a / float(nx)
    dz = TUNNEL_LENGTH_M / float(nz)
    dt_grid, _ = _cached_integer_period_dt(int(nx), int(nz))
    omega_f_grid = float((2.0 / dt_grid) * np.sin(0.5 * omega * dt_grid))
    lambda_x = float(((2.0 / dx) * np.sin(np.pi * dx / (4.0 * a))) ** 2)
    rad = float(np.sqrt(max(lambda_x - (omega_f_grid / c_light) ** 2, 0.0)))
    gamma1_disc = float((2.0 / dz) * np.arcsinh(0.5 * dz * rad))
    d_w_grid = float(np.log(1.0 / EPS_WINDOW) / gamma1_disc)
    return gamma1_disc, d_w_grid, omega_f_grid


def settle_steps(
    code: str,
    nx: int,
    nz: int,
    params: dict[str, float],
    dt: float,
    omega_p_scale: float = 1.0,
    shape: int = PARTICLE_SHAPE,
) -> int:
    """Number of steps before the DFT window so that no turn-on transient remains in the tunnel.

    The C^1 entrance ramp (duration ``t_rise``) launches transient space-charge waves. For the
    slab dispersion ``omega = k u -/+ omega_q R(k)`` with ``R(k) = k / sqrt(k**2 + kappa**2)``
    (``k R'(k) <= R``), the group velocity obeys ``v_g = u -/+ omega_q R'(k) >= omega / k_s``,
    the slow-wave phase velocity. The last transient therefore leaves ``z = Lz + d_w`` (one
    evanescent window ``d_w`` past the exit) no later than
    ``t_settle = t_rise + (Lz + d_w) / v_min`` with ``v_min = omega / max_j(k_c,j + beta_q,j)``
    taken over the excited modes of :func:`discrete_modal_prediction`.

    Returns
    -------
    int
        ``ceil(t_settle / dt)``, or ``ceil(SETTLE_FACTOR * Lz / (u_dep * dt))`` when the
        module-level ``SETTLE_FACTOR`` override is set.
    """
    if SETTLE_FACTOR is not None:
        return int(np.ceil(SETTLE_FACTOR * TUNNEL_LENGTH_M / (params["u_dep"] * dt)))
    det = discrete_modal_prediction(
        code,
        nx,
        nz,
        params,
        omega_p_scale=omega_p_scale,
        return_details=True,
        shape=shape,
    )
    k_slow = float(np.max(det["kc_modes"] + det["bq_modes"]))
    v_min = float(params["omega"] / k_slow)
    d_w, _ = fit_window(params)
    t_settle = RISE_PERIODS / DRIVE_FREQ_HZ + (TUNNEL_LENGTH_M + d_w) / v_min
    return int(np.ceil(t_settle / dt))


def fit_window(
    params: dict[str, float], outflow_ramp_m: float = OUTFLOW_RAMP_M
) -> tuple[float, float]:
    """Return the shared axial fit window ``(z_min, z_max)`` [m].

    Excludes ``d_w_common = max(d_w_coarse, d_w_fine)`` at the entrance (where
    ``d_w_grid = ln(1 / eps_w) / gamma1_disc`` is evaluated with the discrete
    parallel-plate decay constant ``gamma1_disc`` on each grid at relative
    amplitude threshold ``eps_w = 1e-5``) and ``d_w_common`` upstream of the
    outflow taper start ``z_ramp = TUNNEL_LENGTH_M - outflow_ramp_m``.
    """
    d_w = float(params.get("d_w_common", np.log(1.0 / EPS_WINDOW) / params["gamma1"]))
    z_ramp = float(TUNNEL_LENGTH_M - outflow_ramp_m)
    return float(d_w), float(z_ramp - d_w)


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

    the per-line relativistic energy-conserving fixed point::

        phi_l(delta) = phi_l,0 / (1 - delta_bar)
        gamma_l = gamma0 - e * |phi_l| / (m_eff * c**2)
        u_l = c * sqrt(1 - 1 / gamma_l**2)
        delta_bar = 1 - sum_l(w_l * u_l) / (u0 * sum_l(w_l))

    is iterated to ``1e-14``.

    Parameters
    ----------
    qm_scale : float, optional
        Multiplier on the species charge-to-mass ratio ``|q / m|`` (``1.0`` for
        physical electrons, ``2.0`` for the particle species mass mutation).

    Returns
    -------
    dict of str to float
        Dictionary of nominal and DC-depressed analytical wavenumbers,
        wavelengths, currents, and transverse eigenvalues in SI units.
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
    # via exact per-line relativistic energy conservation:
    #   phi_l(delta) = phi_l,0 / (1 - delta_bar)
    #   gamma_l = gamma0 - e * |phi_l| / (m_eff * c^2)
    #   u_l = c * sqrt(1 - 1 / gamma_l^2)
    #   delta_bar = 1 - sum_l(w_l * u_l) / (u0 * sum_l(w_l))
    delta0 = (omega_p0**2 * b * (a - (2.0 / 3.0) * b)) / (gamma0**3 * u0**2)
    mass_eff = m_e / qm_scale
    n_quad = 1024
    x_quad = b * (np.arange(n_quad, dtype=np.float64) + 0.5) / n_quad
    phi_0_quad = (abs(rho0) / (2.0 * epsilon_0)) * (2.0 * a * b - b**2 - x_quad**2)
    rho_quad = np.ones(n_quad, dtype=np.float64)

    def _solve_continuum_depression(weighting: str) -> float:
        delta_val = float(delta0)
        for _ in range(200):
            phi_l = phi_0_quad / (1.0 - delta_val)
            gamma_l = gamma0 - e_charge * phi_l / (mass_eff * c_light**2)
            u_l = c_light * np.sqrt(1.0 - 1.0 / (gamma_l**2))
            w_l = (rho_quad * u_l) if weighting == "flux" else rho_quad
            delta_next = 1.0 - float(np.sum(w_l * u_l) / (u0 * np.sum(w_l)))
            if abs(delta_next - delta_val) < 1e-14:
                return float(delta_next)
            delta_val = float(delta_next)
        return float(delta_val)

    delta_u_dc_flux = _solve_continuum_depression("flux")
    delta_u_dc_density = _solve_continuum_depression("density")
    delta_u_dc = float(delta_u_dc_flux)

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

    rhs_trans = tau * b / np.tanh(tau * (a - b))
    tb1 = float(brentq(lambda x: x * np.tan(x) - rhs_trans, 1e-12, 0.5 * np.pi - 1e-12))
    tb2 = float(
        brentq(lambda x: x * np.tan(x) - rhs_trans, np.pi + 1e-12, 1.5 * np.pi - 1e-12)
    )
    t1_transverse = float(tb1 / b)
    t2_transverse = float(tb2 / b)

    # Logarithmic sensitivity d(ln R)/d(ln tau) from (T1*b)*tan(T1*b) = (tau*b)*coth(tau*(a-b))
    tau_g = tau * (a - b)
    dln_t1_dln_tau = (1.0 - 2.0 * tau_g / np.sinh(2.0 * tau_g)) / (
        1.0 + 2.0 * tb1 / np.sin(2.0 * tb1)
    )
    dlnr_dlntau = float((1.0 - r_factor**2) * (1.0 - dln_t1_dln_tau))

    # Sensitivity d(ln beta_q)/d(delta_u_dc) of depressed beta_q to DC velocity depression
    beta_rel_dep = u_dep / c_light
    dlnbq_ddelta = float((1.5 + dlnr_dlntau) * (1.0 + (beta_rel_dep * gamma_dep) ** 2))

    # Discrete evanescent near-field decay constant per grid:
    #   lambda_x = (2 / dx)^2 * sin^2(pi * dx / (4 * a))
    #   gamma1_disc = (2 / dz) * arcsinh((dz / 2) * sqrt(lambda_x - (Omega_f / c)^2))
    # Use the largest d_w (smallest gamma1_disc over coarse and fine grids) as the one common window:
    gamma1_c, d_w_c, omega_f = _grid_gamma1_disc_and_dw(
        COARSE_GRID[0], COARSE_GRID[2], omega
    )
    gamma1_f, d_w_f, _ = _grid_gamma1_disc_and_dw(FINE_GRID[0], FINE_GRID[2], omega)
    gamma1 = float(min(gamma1_c, gamma1_f))
    d_w_common = float(max(d_w_c, d_w_f))
    d_w = d_w_common
    kappa_dc = float(
        t1_transverse * np.sqrt(1.0 - (omega_p_dep / (u_dep * t1_transverse)) ** 2)
    )

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
        "delta_u_dc_flux": float(delta_u_dc_flux),
        "delta_u_dc_density": float(delta_u_dc_density),
        "u_dep": float(u_dep),
        "gamma_dep": float(gamma_dep),
        "rho_dep": float(rho_dep),
        "omega": float(omega),
        "omega_f": float(omega_f),
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
        "gamma1": float(gamma1),
        "gamma1_disc_coarse": float(gamma1_c),
        "gamma1_disc_fine": float(gamma1_f),
        "kappa_dc": float(kappa_dc),
        "eps_w": float(EPS_WINDOW),
        "d_w": float(d_w),
        "d_w_common": float(d_w_common),
        "d_w_coarse": float(d_w_c),
        "d_w_fine": float(d_w_f),
        "lambda_e": float(2.0 * np.pi / beta_e),
        "lambda_0": float(2.0 * np.pi / k0),
        "lambda_p": float(2.0 * np.pi * u_dep / omega_p_dep),
        "lambda_q": float(2.0 * np.pi / beta_q),
        "I0": float(i0_dc),
        "I1_0": float(i1_entrance),
        "qm_scale": float(qm_scale),
    }


def compute_discrete_dc_depression(
    nx: int,
    params: dict[str, float],
    line_x: np.ndarray | None = None,
    line_w: np.ndarray | None = None,
    weighting: str = "flux",
    return_details: bool = False,
    shape: int = PARTICLE_SHAPE,
) -> float | dict[str, Any]:
    """Solve the 1D discrete Poisson problem across ``x`` on ``nx`` cells for ``delta_disc``.

    Deposits the sheet lines across ``[-b, b]`` onto the ``nx + 1`` transverse
    nodes using order-``shape`` B-spline weights, solves the 3-point discrete
    Poisson equation with PEC boundary conditions ``phi(+-a) = 0``, and iterates
    the exact per-line relativistic energy-conserving fixed point::

        phi_l(delta) = phi_l,0 / (1 - delta_bar)
        gamma_l = gamma0 - e * |phi_l| / (m_eff * c**2)
        u_l = c * sqrt(1 - 1 / gamma_l**2)
        delta_bar = 1 - sum_l(w_l * u_l) / (u0 * sum_l(w_l))

    to ``1e-14`` for current/flux weighting (``w_l = rho_l * u_l``), injected-fraction
    density weighting (``w_l = rho_l``), and steady-state line-charge weighting
    (``w_l = rho_l / u_l``, giving the harmonic mean
    ``1 - delta_charge = sum_l(rho_l) / (u0 * sum_l(rho_l / u_l))``).

    Parameters
    ----------
    nx : int
        Transverse cell count across ``[-a, a]`` [-].
    params : dict of str to float
        Output of :func:`compute_analytic_parameters`.
    line_x : np.ndarray, optional
        Injected particle line ``x`` coordinates [m].
    line_w : np.ndarray, optional
        Injected current weights ``rho_l`` across lines [-].
    weighting : {"flux", "density", "charge"}, optional
        Line weighting mode for ``delta_disc``: ``"flux"`` (``w_l = rho_l * u_l``,
        default), ``"density"`` (``w_l = rho_l``), or ``"charge"``
        (``w_l = rho_l / u_l``).
    return_details : bool, optional
        If ``True``, return a dictionary with ``delta_disc``, ``delta_flux``,
        ``delta_density``, ``delta_charge``, and per-line profiles. Default is ``False``.
    shape : int, optional
        B-spline particle shape order ``p`` in ``{1, 2, 3}`` [-]. Default is
        ``PARTICLE_SHAPE``.

    Returns
    -------
    float or dict of str to Any
        Discrete DC relative velocity depression ``delta_disc`` [-] (or detail
        dictionary when ``return_details=True``).

    Raises
    ------
    ValueError
        If ``weighting`` is not one of ``{"flux", "density", "charge"}``.
    """
    if weighting not in ("flux", "density", "charge"):
        raise ValueError(
            f"Unknown weighting={weighting!r}; expected 'flux', 'density', or 'charge'."
        )

    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    dx = 2.0 * a / nx

    if line_x is None:
        n_macro = 16 if nx == COARSE_GRID[0] else 32
        k_idx = np.arange(n_macro, dtype=np.float64)
        x_sheet = b * (2.0 * (k_idx + 0.5) / n_macro - 1.0)
    else:
        x_sheet = np.asarray(line_x, dtype=np.float64)
        n_macro = len(x_sheet)

    if line_w is None:
        rho_l = np.full(n_macro, 1.0 / n_macro, dtype=np.float64)
    else:
        rho_l = np.asarray(line_w, dtype=np.float64)
        rho_l = rho_l / np.sum(rho_l)

    w_mat = transverse_weight_matrix(x_sheet, nx, shape)
    w_node = np.sum(rho_l[:, None] * w_mat, axis=0) * (2.0 * b / dx)

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
    phi_l0 = np.abs(w_mat @ phi_0)
    delta0_disc = float(
        np.sum(rho_l * phi_l0) / (gamma0**3 * mass_eff * u0**2 / e_charge)
    )

    def _iterate_fixed_point(
        mode: str,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        delta_val = float(delta0_disc)
        u_l = np.full(n_macro, u0 * (1.0 - delta_val), dtype=np.float64)
        gamma_l = np.full(n_macro, gamma0, dtype=np.float64)
        phi_l = phi_l0.copy()
        phi_cur = np.zeros(nx + 1, dtype=np.float64)
        for _ in range(200):
            # Solve 1D discrete Poisson with actual per-line depressed charge density prop rho_l * (u0 / u_l)
            rho_dep_l = rho_l * (u0 / u_l)
            w_node_dep = np.sum(rho_dep_l[:, None] * w_mat, axis=0) * (2.0 * b / dx)
            phi_cur[1:-1] = np.linalg.solve(lap, -rho0 * w_node_dep[1:-1] / epsilon_0)
            phi_l = np.abs(w_mat @ phi_cur)
            gamma_l = gamma0 - e_charge * phi_l / (mass_eff * c_light**2)
            u_l = c_light * np.sqrt(1.0 - 1.0 / (gamma_l**2))
            if mode == "flux":
                w_l = rho_l * u_l
            elif mode == "density":
                w_l = rho_l
            else:
                w_l = rho_l / u_l
            delta_next = 1.0 - float(np.sum(w_l * u_l) / (u0 * np.sum(w_l)))
            if abs(delta_next - delta_val) < 1e-14:
                delta_val = float(delta_next)
                break
            delta_val = float(delta_next)
        return float(delta_val), u_l, gamma_l, phi_l

    delta_flux, u_l_flux, gamma_l_flux, phi_l_flux = _iterate_fixed_point("flux")
    delta_density, u_l_dens, gamma_l_dens, phi_l_dens = _iterate_fixed_point("density")
    delta_charge, u_l_chg, gamma_l_chg, phi_l_chg = _iterate_fixed_point("charge")

    if weighting == "flux":
        delta_disc, u_l, gamma_l, phi_l = delta_flux, u_l_flux, gamma_l_flux, phi_l_flux
    elif weighting == "density":
        delta_disc, u_l, gamma_l, phi_l = (
            delta_density,
            u_l_dens,
            gamma_l_dens,
            phi_l_dens,
        )
    else:
        delta_disc, u_l, gamma_l, phi_l = (
            delta_charge,
            u_l_chg,
            gamma_l_chg,
            phi_l_chg,
        )

    if return_details:
        return {
            "delta_disc": float(delta_disc),
            "delta_flux": float(delta_flux),
            "delta_density": float(delta_density),
            "delta_charge": float(delta_charge),
            "weighting": weighting,
            "u_l": u_l,
            "gamma_l": gamma_l,
            "phi_l": phi_l,
            "rho_l": rho_l,
            "x_sheet": x_sheet,
            "w_mat": w_mat,
        }
    return float(delta_disc)


@lru_cache(maxsize=16)
def _lb_top_hat_weights(nx: int) -> tuple[np.ndarray, np.ndarray]:
    """Read `_w_perp_eff` and interior `keep_lines` from a constructed `LinearizedBeam` on `nx` cells."""
    params = compute_analytic_parameters()
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    ly = PERIODIC_LY_M
    lz = TUNNEL_LENGTH_M
    grid = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, int(nx), 1, 8, verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=CFL_NUMBER,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_gpu=False,
        verbose=0,
    )
    beam = LinearizedBeam(
        u0=params["u_dep"],
        rho0=params["rho_dep"],
        gamma0=params["gamma_dep"],
        axis="z",
        half_thickness=b,
        solver=solver,
    )
    w_eff = np.asarray(beam._w_perp_eff[0], dtype=np.float64).copy()
    keep_int = np.asarray(beam.keep_lines[1:-1], dtype=bool).copy()
    return w_eff, keep_int


def compute_transverse_modal_operator(
    code: str,
    nx: int,
    nz: int | None,
    params: dict[str, float],
    k_val: float | None = None,
    line_x: np.ndarray | None = None,
    line_w: np.ndarray | None = None,
    use_omega_f: bool = True,
    shape: int = PARTICLE_SHAPE,
) -> dict[str, Any]:
    """Build the discrete transverse modal operator ``G(k)`` and its eigensystem at wavenumber ``k_val``.

    Parameters
    ----------
    code : {"lb", "particles"}
        Model selector (`LinearizedBeam` node lines or `ParticleSolver` order-``shape`` lines).
    nx : int
        Transverse cell count across ``[-a, a]``.
    nz : int or None
        Axial cell count across ``[0, Lz]``. If ``None``, evaluates the continuum
        axial symbol ``kappa(k) = k``; otherwise ``kappa(k) = (2 / dz) * sin(k * dz / 2)``.
    params : dict of str to float
        Output of :func:`compute_analytic_parameters`.
    k_val : float, optional
        Axial wavenumber [rad/m]. Defaults to ``params["beta_e"]``.
    line_x : np.ndarray, optional
        Injected particle line ``x`` coordinates [m] for ``code == "particles"``.
    line_w : np.ndarray, optional
        Unnormalized line charge weights.
    use_omega_f : bool, optional
        If ``True`` (default) and ``nz`` is not ``None``, uses the leapfrog field
        time symbol ``Omega_f = (2 / dt) * sin(omega * dt / 2)`` in
        ``tau_Y(k)**2 = kappa(k)**2 - (Omega_f / c)**2``; otherwise uses ``omega``.
    shape : int, optional
        B-spline particle shape order ``p`` in ``{1, 2, 3}`` for ``code == "particles"``.
        Default is ``PARTICLE_SHAPE``.

    Returns
    -------
    dict of str to Any
        Dictionary with ``evals``, ``y_right``, ``y_left``, ``d_vec``, ``G``,
        ``lap``, ``dx``, ``w_perp_eff``, and ``keep_int``.
    """
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    dx = 2.0 * a / nx
    n_int = nx - 1
    lap = (
        np.diag(np.full(n_int, -2.0))
        + np.diag(np.ones(n_int - 1), 1)
        + np.diag(np.ones(n_int - 1), -1)
    ) / (dx**2)

    if k_val is None:
        k_val = float(params["beta_e"])
    if nz is None:
        kappa = float(k_val)
        omega_tau = float(params["omega"])
    else:
        dz = TUNNEL_LENGTH_M / nz
        kappa = float((2.0 / dz) * np.sin(0.5 * k_val * dz))
        if use_omega_f:
            dt_grid, _ = _cached_integer_period_dt(int(nx), int(nz))
            omega_tau = float((2.0 / dt_grid) * np.sin(0.5 * params["omega"] * dt_grid))
        else:
            omega_tau = float(params["omega"])
    tau_y2 = max(kappa**2 - (omega_tau / c_light) ** 2, 1e-12)
    a_mat = -lap + tau_y2 * np.eye(n_int, dtype=np.float64)

    if code == "lb":
        w_eff, keep_int = _lb_top_hat_weights(nx)
        if line_w is None:
            w_line = w_eff.copy()
        else:
            lw = np.asarray(line_w, dtype=np.float64)
            if len(lw) == n_int:
                w_line = lw[keep_int]
            elif len(lw) == nx + 1:
                w_line = lw[1:-1][keep_int]
            else:
                w_line = lw.copy()
        d_vec = w_line * dx / (2.0 * b)
        d_vec = d_vec / np.sum(d_vec)
        d_mat = np.diag(d_vec)

        # In LinearizedBeam, E_z gather on active lines is I and J_z deposit is
        # (2*b/dx)*D = diag(w_line); in the w_perp_eff-scaled basis where
        # S = diag(w_perp_eff) and D_s = (2*b/dx)*diag(d_vec / w_perp_eff^2),
        # G(k) = tau_Y(k)^2 * S^T A(k)^{-1} S D_s is similarity-equivalent to
        # G_lb(k) = (2*b/dx) * tau_Y(k)^2 * A_sub(k)^{-1} D.
        a_inv_sub = np.linalg.inv(a_mat)[np.ix_(keep_int, keep_int)]
        g_mat = (2.0 * b / dx) * tau_y2 * (a_inv_sub @ d_mat)
        s_left = np.sqrt(d_vec)
        s_right = 1.0 / np.sqrt(d_vec)
        m_sym = (s_left[:, None] * g_mat) * s_right[None, :]
        m_sym = 0.5 * (m_sym + m_sym.T)
        evals, qvecs = np.linalg.eigh(m_sym)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        qvecs = qvecs[:, order]
        y_right = s_right[:, None] * qvecs
        y_left = s_left[:, None] * qvecs
        return {
            "evals": evals,
            "y_right": y_right,
            "y_left": y_left,
            "d_vec": d_vec,
            "G": g_mat,
            "lap": lap,
            "dx": dx,
            "w_perp_eff": w_eff,
            "w_line": w_line,
            "keep_int": keep_int,
        }

    if code == "particles":
        if line_x is None:
            n_macro = 16 if nx == COARSE_GRID[0] else 32
            k_idx = np.arange(n_macro, dtype=np.float64)
            x_p = b * (2.0 * (k_idx + 0.5) / n_macro - 1.0)
        else:
            x_p = np.asarray(line_x, dtype=np.float64)
            n_macro = len(x_p)

        if line_w is None:
            d_vec = np.full(n_macro, 1.0 / n_macro, dtype=np.float64)
        else:
            d_vec = np.asarray(line_w, dtype=np.float64)
            d_vec = d_vec / np.sum(d_vec)
        d_mat = np.diag(d_vec)

        # contiguous copy: a transposed view changes the BLAS summation order (ulp-level G change
        # that rotates the coarse grid's null-space eigenvectors and broke p = 1 bit-identity)
        w_p = np.ascontiguousarray(transverse_weight_matrix(x_p, nx, shape)[:, 1:-1].T)
        s_mat = w_p * ((2.0 * b / n_macro) / dx)
        g_mat = (
            (n_macro**2 * dx / (2.0 * b))
            * tau_y2
            * (s_mat.T @ np.linalg.solve(a_mat, s_mat) @ d_mat)
        )
        s_left = np.sqrt(d_vec)
        s_right = 1.0 / np.sqrt(d_vec)
        m_sym = (s_left[:, None] * g_mat) * s_right[None, :]
        m_sym = 0.5 * (m_sym + m_sym.T)
        evals, qvecs = np.linalg.eigh(m_sym)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        qvecs = qvecs[:, order]
        y_right = s_right[:, None] * qvecs
        y_left = s_left[:, None] * qvecs
        return {
            "evals": evals,
            "y_right": y_right,
            "y_left": y_left,
            "d_vec": d_vec,
            "G": g_mat,
            "S": s_mat,
            "lap": lap,
            "dx": dx,
            "line_x": x_p,
        }

    raise ValueError(f"Unknown code={code!r}; expected 'lb' or 'particles'.")


def _even_subspace_basis(n_lines: int) -> np.ndarray:
    """Build the orthonormal mirror-even line basis ``P`` of shape ``(N, N // 2)`` [-]."""
    h = int(n_lines) // 2
    p_mat = np.zeros((int(n_lines), h), dtype=np.float64)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for j in range(h):
        p_mat[j, j] = inv_sqrt2
        p_mat[int(n_lines) - 1 - j, j] = inv_sqrt2
    return p_mat


def _multistream_dc_entrance_phase(
    nx: int,
    dz: float,
    dt: float,
    params: dict[str, float],
    dep_info: dict[str, Any],
    discrete: bool = True,
) -> np.ndarray:
    """Compute the per-line ballistic DC entrance phase ``theta_l`` [rad].

    Near the ``z = 0`` PEC entrance wall (Dirichlet ``phi(x, 0) = 0``), each transverse
    eigenmode ``(lambda_m, e_m)`` of ``-Delta_x`` relaxes toward its bulk depressed
    potential ``phi_{m,inf}`` as ``phi_m(z) = phi_{m,inf} * (1 - exp(-q_m * z))``, where
    ``q_m = -ln(r_m) / dz`` with ``r_m + 1 / r_m = 2 + lambda_m * dz**2`` (or
    ``q_m = sqrt(lambda_m)`` when ``discrete=False``). Energy conservation
    ``gamma_l(z) = gamma0 - e * phi_l(z) / (m * c**2)`` determines ``u_l(z)`` and the
    accumulated ballistic transit phase relative to the bulk stream velocity ``u_l``::

        theta_l = int_0^infty (Omega / u_l(z) - Omega / u_l) dz,
        Omega = (2 / dt) * sin(omega * dt / 2).

    Parameters
    ----------
    nx : int
        Transverse cell count across ``[-a, a]`` [-].
    dz : float
        Axial cell size [m].
    dt : float
        Time step [s].
    params : dict of str to float
        Output of :func:`compute_analytic_parameters`.
    dep_info : dict of str to Any
        Output of :func:`compute_discrete_dc_depression` with ``return_details=True``.
    discrete : bool, optional
        If ``True`` (default), use the discrete axial decay rate ``q_m = -ln(r_m) / dz``;
        if ``False``, use continuum ``sqrt(lambda_m)``.

    Returns
    -------
    np.ndarray
        Per-line ballistic entrance phase ``theta_l`` [rad] of shape ``(N,)``.
    """
    a_half = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    dx = 2.0 * a_half / nx
    n_int = nx - 1
    lap = (
        np.diag(np.full(n_int, -2.0))
        + np.diag(np.ones(n_int - 1), 1)
        + np.diag(np.ones(n_int - 1), -1)
    ) / dx**2
    rho_l = dep_info["rho_l"]
    w_mat = dep_info["w_mat"]
    u_l = dep_info["u_l"]
    u0 = params["u0"]
    w_node = np.sum((rho_l * u0 / u_l)[:, None] * w_mat, axis=0) * (2.0 * b / dx)
    phi_int = np.linalg.solve(lap, -abs(params["rho0"]) * w_node[1:-1] / epsilon_0)
    lam, e_vecs = np.linalg.eigh(-lap)
    coef = e_vecs.T @ phi_int
    if discrete:
        rr = 1.0 + 0.5 * lam * dz**2
        q_modes = -np.log(rr - np.sqrt(rr**2 - 1.0)) / dz
    else:
        q_modes = np.sqrt(lam)
    zz = np.linspace(0.0, 30.0 / float(np.min(q_modes)), 20001)
    phi_z = (e_vecs * coef[None, :]) @ (1.0 - np.exp(-np.outer(q_modes, zz)))
    phi_lz = np.abs(w_mat[:, 1:-1] @ phi_z)
    mass = m_e / params.get("qm_scale", 1.0)
    gamma_lz = params["gamma0"] - e_charge * phi_lz / (mass * c_light**2)
    u_lz = c_light * np.sqrt(1.0 - 1.0 / gamma_lz**2)
    om_dt = (2.0 / dt) * np.sin(0.5 * params["omega"] * dt)
    integrand = om_dt / u_lz - (om_dt / u_lz[:, -1])[:, None]
    return np.trapezoid(integrand, zz, axis=1)


def _multistream_wall_correction(
    op0: dict[str, Any],
    dz: float,
    dt: float,
    omega: float,
    sig_vec: np.ndarray,
    d_vec: np.ndarray,
    k_waves: np.ndarray,
    x_waves: np.ndarray,
    a_waves: np.ndarray,
    p_basis: np.ndarray,
    shape: int = PARTICLE_SHAPE,
    use_omega_f: bool = True,
    include_sz: bool = True,
    discrete: bool = True,
) -> np.ndarray:
    """Compute the first-order (Born) PEC-wall near-field wave amplitude shift ``wall_da`` [-].

    At ``z = 0``, the PEC wall enforces ``E_x(x, 0) = 0``, corresponding to a Neumann
    (even) image continuation of ``J_z`` across ``z = 0`` rather than the unbounded
    spatial harmonic continuation assumed by the bulk symbol ``G(k)``. The resulting
    evanescent discrepancy field ``Delta F(z) = sum_m f_m * exp(-q_m * z)`` drives the
    beam lines and produces an asymptotic wave-amplitude shift::

        Delta a_n = sum_m (R (-sigma * f_m))_n / (q_m - i * k_n),
        R = V_e^{-1} [0; I] P^T,   V_e = [P^T X; P^T X diag(-i * k_n)].
    """
    s_mat = op0["S"]
    d_mat = np.diag(d_vec)
    n_lines = len(d_vec)
    b = BEAM_HALF_THICKNESS_M
    dx = op0["dx"]
    c_scale = n_lines**2 * dx / (2.0 * b)
    lam, e_vecs = np.linalg.eigh(op0["lap"] * -1.0)
    om_tau = (2.0 / dt) * np.sin(0.5 * omega * dt) if use_omega_f else float(omega)
    k0sq = (om_tau / c_light) ** 2
    s2 = lam - k0sq
    p_left = s_mat.T @ e_vecs
    p_right = e_vecs.T @ s_mat @ d_mat
    pr = p_basis.T
    px = pr @ x_waves
    v_mat = np.vstack([px, px * (-1j * k_waves)[None, :]])
    h_sub = pr.shape[0]
    r_mat = (
        np.linalg.solve(v_mat, np.vstack([np.zeros((h_sub, h_sub)), np.eye(h_sub)]))
        @ pr
    )
    da = np.zeros(len(k_waves), dtype=np.complex128)
    p_int = int(shape)
    for m in range(len(lam)):
        sm = np.sqrt(s2[m])
        if discrete:
            rr = 1.0 + 0.5 * s2[m] * dz**2
            rho = rr - np.sqrt(rr**2 - 1.0)
            qm = -np.log(rho) / dz
            h_m = lam[m] * rho * dz**2 / (1.0 - rho**2)
            amp = (
                h_m
                * np.sqrt(rho)
                * (
                    np.exp(-0.5j * k_waves * dz)
                    / (1.0 - rho * np.exp(-1j * k_waves * dz))
                    - np.exp(0.5j * k_waves * dz)
                    / (1.0 - rho * np.exp(1j * k_waves * dz))
                )
            )
        else:
            qm = sm
            amp = lam[m] / (2.0 * sm) * (-2j * k_waves) / (sm**2 + k_waves**2)
        if include_sz:
            amp = amp * (np.sinc(0.5 * k_waves * dz / np.pi) ** p_int)
            gq = (np.sinh(0.5 * qm * dz) / (0.5 * qm * dz)) ** p_int
        else:
            gq = 1.0
        fvec = (
            -c_scale
            * gq
            * p_left[:, m]
            * np.sum(a_waves * (p_right[m] @ x_waves) * amp)
        )
        drive = -sig_vec * fvec
        da += (r_mat @ drive) / (qm - 1j * k_waves)
    return da


def _multistream_leapfrog_omega(
    omega: float, dt: float, use_omega_f: bool = True
) -> float:
    """Return the leapfrog angular frequency symbol ``Omega_f = (2 / dt) * sin(omega * dt / 2)`` [rad/s]."""
    if not use_omega_f:
        return float(omega)
    return float((2.0 / dt) * np.sin(0.5 * omega * dt))


def _multistream_entrance_dk(k_waves: np.ndarray, be_vec: np.ndarray) -> np.ndarray:
    """Return the per-line, per-wave Doppler shift ``k_n - beta_l`` [rad/m] for ``v_1(0) = 0`` matching."""
    return k_waves[None, :] - be_vec[:, None]


def _multistream_observable_weights(
    a_waves: np.ndarray,
    d_vec: np.ndarray,
    x_waves: np.ndarray,
    k_waves: np.ndarray,
    dz: float,
    shape: int,
    use_obs_sinc: bool,
) -> np.ndarray:
    """Compute the per-wave deposited-current observable weights ``o_n`` [-]."""
    o_waves = a_waves * (d_vec @ x_waves)
    if use_obs_sinc:
        o_waves = o_waves * (np.sinc(0.5 * k_waves * dz / np.pi) ** int(shape))
    return o_waves


def multistream_waves(
    nx: int,
    nz: int,
    params: dict[str, float],
    *,
    line_x: np.ndarray | None = None,
    line_w: np.ndarray | None = None,
    shape: int = PARTICLE_SHAPE,
    use_omega_f: bool = True,
    include_sz: bool = True,
    include_shear: bool = True,
    omega_p_scale: float = 1.0,
    wall_image: str = "neumann",
    dc_entrance: bool = True,
    entrance_profile: np.ndarray | None = None,
    freeze_g: bool = False,
    subspace: str | None = None,
    observable_sinc: bool | None = None,
) -> dict[str, Any]:
    """Solve the exact per-line cold multistream boundary-value problem for ``I_1(z)``.

    Governing Equations
    -------------------
    Each macro-particle stream ``l = 0, ..., N - 1`` at transverse position ``x_l`` with
    DC velocity ``u_l``, Lorentz factor ``gamma_l``, and normalized charge weight ``d_l``
    satisfies the linearized cold-stream equation of motion and continuity equation at
    discrete leapfrog frequency ``Omega = (2 / dt) * sin(omega * dt / 2)``::

        (d/dz + i * beta_l) v_{1,l}(z) = (q / (m * gamma_l**3 * u_l)) * E_{z,l}(z),
        (d/dz + i * beta_l) i_{1,l}(z) = -i * (Omega / u_l**2) * I_{0,l} * v_{1,l}(z),

    where ``beta_l = Omega / u_l`` [rad/m]. Eliminating ``v_{1,l}`` for spatial harmonic
    ``i_1(z) = X * exp(-i * k * z)`` yields the ``N``-line nonlinear eigenvalue problem::

        T(k) X = [diag((k - beta_l)**2) - M(k)] X = 0,
        M(k) = S_z(k)**2 * diag(sigma_l) * G(k),
        sigma_l = (omega_p_scale * omega_p0)**2 * (u0 / u_l) / (gamma_l**3 * u_l**2),
        S_z(k)**2 = sinc(k * dz / 2)**(2 * p).

    In the active subspace of dimension ``h_sub``, each root ``k_n`` is required to
    satisfy the quadratic-pencil backward-error bound derived from IEEE-754 ``float64``
    machine precision ``eps_mach = np.finfo(np.float64).eps`` and the relative condition
    ratio ``kappa_T(k_n)`` of ``T(k)``::

        root_resid[n] = sigma_min(T(k_n)) / sigma_max(T(k_n)) <= root_tol[n],
        root_tol[n] = h_sub * eps_mach * kappa_T(k_n),
        kappa_T(k_n) = (|k_n| + max_l(beta_l))**2 / sigma_max(T(k_n)),

    with pairwise root separation ``|k_n - k_m| > root_k_tol = 1e-13 * max_n(|k_n|)``.

    At the ``z = 0`` entrance plane, pure density modulation with ``v_{1,l}(0) = 0`` and
    DC entrance transit phase ``theta_l`` imposes the ``2 * N`` matching conditions across
    the ``2 * N`` fast/slow multistream waves ``(k_n, X_n)``::

        sum_n a_n * X_{l,n} = e_l * exp(-i * theta_l),
        sum_n a_n * (k_n - beta_l) * X_{l,n} = 0,

    solved directly via :func:`numpy.linalg.solve` (with relative residual
    ``match_resid <= 1e-12``), followed by the first-order (Born) PEC-wall near-field
    correction ``a <- a + wall_da`` from the ``z = 0`` image continuation of ``J_z`` and
    the deposited-current observable symbol
    ``o_n = a_n * sinc(k_n * dz / 2)**p * (d^T X_n)`` [-].

    Limits & Failure Modes
    ----------------------
    - **Linear in modulation index ``m``**: solves the ``m -> 0`` linearized multistream
      equations; quadratic harmonic generation and ``O(m**2)`` ballistic bunching shifts at
      finite ``m`` are not included.
    - **Discrete-particle alias/Doppler terms not modelled**: retains only the principal
      ``m = 0`` axial spatial harmonic ``S_z(k)**2``; higher Brillouin zone aliases
      ``k_m = k + 2 * pi * m / dz`` are Doppler-detuned and omitted.
    - **Complex ballistic band evaluated at ``Re(k)``**: for weakly unstable complex
      ballistic roots near ``beta_bar`` (``|Im(k)| <= 0.184`` rad/m for ``q/m = 1`` and
      ``<= 0.426`` rad/m for ``q/m = 2``), ``M(k)`` and ``root_resid`` are evaluated at
      ``Re(k)``; dropping the complex ballistic band changes fitted ``beta_q`` by
      ``<= 4.8e-6`` relative.
    - **PEC-wall near-field term is first order**: the ``z = 0`` wall correction uses the
      first-order Born approximation in the evanescent image field and models the first
      cell through Fourier/modal symbols rather than the cut near-wall Yee stencil.
    - **Thin-region DC entrance phase**: ``theta_l`` integrates the 1D modal potential rise
      along unperturbed trajectories and ignores near-wall density depletion.

    Parameters
    ----------
    nx, nz : int
        Transverse and axial cell counts [-].
    params : dict of str to float
        Output of :func:`compute_analytic_parameters`.
    line_x : np.ndarray, optional
        Injected particle line ``x`` positions [m] of shape ``(N,)``.
    line_w : np.ndarray, optional
        Line charge weights [-] of shape ``(N,)``.
    shape : int, optional
        B-spline particle shape order ``p`` in ``{1, 2, 3}`` [-]. Default is ``PARTICLE_SHAPE``.
    use_omega_f : bool, optional
        If ``True`` (default), use ``Omega_f = (2 / dt) * sin(omega * dt / 2)`` in the
        transverse Green's operator and wall kernel; if ``False``, use ``omega``.
    include_sz : bool, optional
        If ``True`` (default), include the axial B-spline shape factor ``S_z(k)**2``.
    include_shear : bool, optional
        If ``True`` (default), use per-line depressed ``(u_l, gamma_l)``; if ``False``,
        use uniform ``(u_bar, gamma_bar)``.
    omega_p_scale : float, optional
        Multiplier on ``omega_p0`` [-]. Default is ``1.0``.
    wall_image : {"neumann", "none", "dirichlet"}, optional
        PEC entrance wall near-field treatment: ``"neumann"`` (default physical even ``J_z``
        image), ``"none"`` (no wall correction), or ``"dirichlet"`` (flipped-sign control).
    dc_entrance : bool, optional
        If ``True`` (default), include the per-line DC entrance transit phase ``theta_l``.
    entrance_profile : np.ndarray, optional
        Entrance modulation profile across lines [-].
    freeze_g : bool, optional
        If ``True``, freeze ``G(k)`` and ``S_z(k)**2`` at ``k_ref = omega / u_bar``.
        Default is ``False``.
    subspace : {"even", "full"} or None, optional
        Force ``"even"`` mirror subspace or ``"full"`` line space. Default ``None``
        selects ``"even"`` when lines, weights, ``beta_l``, and entrance profile are mirror
        symmetric, and ``"full"`` otherwise.
    observable_sinc : bool or None, optional
        Override whether the observable symbol ``sinc(k_n * dz / 2)**p`` is applied in
        ``o_waves``. When ``None`` (default), enabled iff ``include_sz and not freeze_g``.

    Returns
    -------
    dict of str to Any
        Dictionary with keys ``k_waves`` (``(n,)`` [rad/m]), ``x_waves`` (``(N, n)`` [-]),
        ``a_waves`` (``(n,)`` [-]), ``o_waves`` (``(n,)`` [-], ``I_1 / I_{1,0}`` per wave),
        ``root_resid`` (``(n,)`` [-]), ``root_tol`` (``(n,)`` [-]),
        ``root_k_tol`` (float [rad/m]), ``match_resid`` (float [-]),
        ``wall_da`` (``(n,)`` [-]), ``theta_l`` (``(N,)`` [rad]), and
        ``subspace`` (``"even"`` or ``"full"``).

    Raises
    ------
    RuntimeError
        If ``tau_y2 <= 1e-12`` m^-2, any root residual exceeds ``root_tol``, two roots
        coincide within ``root_k_tol``, the entrance matching system is singular, or
        ``match_resid > 1e-12``.
    """
    if wall_image not in ("neumann", "none", "dirichlet"):
        raise ValueError(
            f"Unknown wall_image={wall_image!r}; expected 'neumann', 'none', or 'dirichlet'."
        )
    if subspace not in (None, "even", "full"):
        raise ValueError(
            f"Unknown subspace={subspace!r}; expected None, 'even', or 'full'."
        )

    dz = TUNNEL_LENGTH_M / nz
    dt, _ = _cached_integer_period_dt(int(nx), int(nz))
    omega = params["omega"]

    dep_info = compute_discrete_dc_depression(
        nx,
        params,
        line_x=line_x,
        line_w=line_w,
        return_details=True,
        shape=shape,
    )
    u_l = dep_info["u_l"]
    gamma_l = dep_info["gamma_l"]
    delta_disc = float(dep_info["delta_disc"])
    u0 = params["u0"]
    u_bar = float(u0 * (1.0 - delta_disc))
    gamma_bar = float(1.0 / np.sqrt(1.0 - (u_bar / c_light) ** 2))
    k_ref = float(omega / u_bar)
    omega_p0 = params["omega_p0"] * float(omega_p_scale)
    omega_p_dt = _multistream_leapfrog_omega(omega, dt, use_omega_f=True)

    op0 = compute_transverse_modal_operator(
        "particles",
        nx,
        nz,
        params,
        k_val=k_ref,
        line_x=line_x,
        line_w=line_w,
        use_omega_f=use_omega_f,
        shape=shape,
    )
    d_vec = op0["d_vec"]
    lx = op0["line_x"]
    n_lines = len(d_vec)

    omega_p_l_sq = (omega_p0**2) * (u0 / u_l)
    if include_shear:
        sig_vec = omega_p_l_sq / (gamma_l**3 * u_l**2)
        be_vec = omega_p_dt / u_l
    else:
        omega_p_bar_sq = (omega_p0**2) * (u0 / u_bar)
        sig_vec = np.full(
            n_lines,
            omega_p_bar_sq / (gamma_bar**3 * u_bar**2),
            dtype=np.float64,
        )
        be_vec = np.full(n_lines, omega_p_dt / u_bar, dtype=np.float64)

    if entrance_profile is None:
        e_base = np.ones(n_lines, dtype=np.float64)
    else:
        ep = np.asarray(entrance_profile, dtype=np.float64)
        ep_norm = ep / np.sum(ep)
        e_base = ep_norm / np.maximum(d_vec, 1.0e-30)

    is_symmetric = (
        (n_lines % 2 == 0)
        and np.allclose(lx, -lx[::-1], atol=1.0e-12 * BEAM_HALF_THICKNESS_M)
        and np.allclose(
            be_vec, be_vec[::-1], rtol=0.0, atol=1.0e-9 * float(np.max(be_vec))
        )
        and np.allclose(d_vec, d_vec[::-1], rtol=1.0e-12)
        and np.allclose(e_base, e_base[::-1], rtol=1.0e-12, atol=1.0e-12)
    )
    if subspace is None:
        subspace_use = "even" if is_symmetric else "full"
    else:
        subspace_use = subspace

    p_basis = (
        _even_subspace_basis(n_lines)
        if subspace_use == "even"
        else np.eye(n_lines, dtype=np.float64)
    )
    be_sub = p_basis.T @ np.diag(be_vec) @ p_basis
    be_max = float(np.max(be_vec))
    h_sub = p_basis.shape[1]
    eps_mach = float(np.finfo(np.float64).eps)
    b = BEAM_HALF_THICKNESS_M
    dx = op0["dx"]
    lap = op0["lap"]
    s_mat = op0["S"]
    d_mat = np.diag(d_vec)
    eye_int = np.eye(nx - 1, dtype=np.float64)
    g_scale = (n_lines**2 * dx) / (2.0 * b)
    omega_tau = _multistream_leapfrog_omega(omega, dt, use_omega_f=use_omega_f)

    def _m_mat(k_val: float) -> np.ndarray:
        kk = k_ref if freeze_g else float(np.real(k_val))
        kappa = float((2.0 / dz) * np.sin(0.5 * kk * dz))
        tau_y2 = kappa**2 - (omega_tau / c_light) ** 2
        if tau_y2 <= 1.0e-12:
            raise RuntimeError(
                f"multistream_waves: transverse decay parameter tau_y2={tau_y2:.6e} m^-2 "
                f"<= 1.0e-12 m^-2 at k={kk:.6e} rad/m "
                f"(kappa^2={kappa**2:.6e}, (omega_tau/c)^2={(omega_tau / c_light) ** 2:.6e})."
            )
        a_mat = -lap + tau_y2 * eye_int
        g_mat = g_scale * tau_y2 * (s_mat.T @ np.linalg.solve(a_mat, s_mat) @ d_mat)
        sz_sq = (
            float(np.sinc(0.5 * kk * dz / np.pi) ** (2 * int(shape)))
            if include_sz
            else 1.0
        )
        return (sz_sq * sig_vec)[:, None] * g_mat

    def _t_sub(k_val: complex) -> np.ndarray:
        return (
            p_basis.T
            @ (np.diag((k_val - be_vec) ** 2) - _m_mat(float(np.real(k_val))))
            @ p_basis
        )

    def _root_tol_at(k_val: complex, sv_max: float) -> float:
        kappa_t = ((abs(k_val) + be_max) ** 2) / sv_max
        return float(h_sub * eps_mach * kappa_t)

    fd_h = 1.0e-3

    def _dt_sub(k_val: complex) -> np.ndarray:
        kr = float(np.real(k_val))
        dm = (_m_mat(kr + fd_h) - _m_mat(kr - fd_h)) / (2.0 * fd_h)
        return p_basis.T @ (2.0 * np.diag(k_val - be_vec) - dm) @ p_basis

    def _qep_sub(m_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m_sub = p_basis.T @ m_full @ p_basis
        a0_mat = be_sub @ be_sub - m_sub
        a1_mat = -2.0 * be_sub
        comp = np.block([[np.zeros((h_sub, h_sub)), np.eye(h_sub)], [-a0_mat, -a1_mat]])
        kr_vals, v_vecs = np.linalg.eig(comp)
        return kr_vals, v_vecs[:h_sub, :]

    m_ref_sub = p_basis.T @ _m_mat(k_ref) @ p_basis
    a0_init = be_sub @ be_sub - m_ref_sub
    a1_init = -2.0 * be_sub
    comp_init = np.block(
        [[np.zeros((h_sub, h_sub)), np.eye(h_sub)], [-a0_init, -a1_init]]
    )
    k0_seeds = np.linalg.eigvals(comp_init)

    ks_list: list[complex] = []
    w_list: list[np.ndarray] = []
    res_list: list[float] = []
    tol_list: list[float] = []
    if not freeze_g and include_shear:
        for k_seed in k0_seeds:
            k_cur = complex(k_seed)
            for _ in range(30):
                t_cur = _t_sub(k_cur)
                u_svd, _, vh_svd = np.linalg.svd(t_cur)
                x_vec, y_vec = vh_svd[-1].conj(), u_svd[:, -1]
                denom = y_vec.conj() @ _dt_sub(k_cur) @ x_vec
                if abs(denom) == 0.0:
                    break
                dk = -(y_vec.conj() @ t_cur @ x_vec) / denom
                k_cur = k_cur + dk
                if abs(dk) < 1.0e-13 * abs(k_cur):
                    break
            _, sv_end, vh_end = np.linalg.svd(_t_sub(k_cur))
            ks_list.append(k_cur)
            w_list.append(vh_end[-1].conj())
            res_list.append(float(sv_end[-1] / sv_end[0]))
            tol_list.append(_root_tol_at(k_cur, float(sv_end[0])))

        k_waves = np.asarray(ks_list, dtype=np.complex128)
        w_sub = np.asarray(w_list, dtype=np.complex128).T
        root_resid = np.asarray(res_list, dtype=np.float64)
        root_tol = np.asarray(tol_list, dtype=np.float64)
        dk_mat = np.abs(k_waves[:, None] - k_waves[None, :]) + np.eye(len(k_waves))
        min_dk = float(np.min(dk_mat))
    else:
        k_waves = np.empty(0, dtype=np.complex128)
        w_sub = np.empty((h_sub, 0), dtype=np.complex128)
        root_resid = np.array([1.0], dtype=np.float64)
        root_tol = np.array([0.0], dtype=np.float64)
        dk_mat = np.zeros((1, 1), dtype=np.float64)
        min_dk = 0.0

    # Fallback to successive substitution when Newton has degenerate roots or fails residual check
    if (
        bool(np.any(root_resid > root_tol))
        or min_dk <= 1.0e-8
        or freeze_g
        or not include_shear
    ):
        k0_s, w0_s = _qep_sub(_m_mat(k_ref))
        has_newton = len(k_waves) == len(k0_s)
        ks_fb: list[complex] = []
        w_fb: list[np.ndarray] = []
        res_fb: list[float] = []
        tol_fb: list[float] = []
        for idx, kn in enumerate(k0_s):
            if (
                has_newton
                and root_resid[idx] <= root_tol[idx]
                and float(np.min(dk_mat[idx])) > 1.0e-8
            ):
                ks_fb.append(complex(k_waves[idx]))
                w_fb.append(w_sub[:, idx])
                res_fb.append(float(root_resid[idx]))
                tol_fb.append(float(root_tol[idx]))
                continue
            k_cur = complex(kn)
            w_cur = w0_s[:, idx] / np.linalg.norm(w0_s[:, idx])
            t_init = _t_sub(k_cur)
            _, sv_init, _ = np.linalg.svd(t_init)
            if not freeze_g and not (
                len(sv_init) > 1 and sv_init[-2] < 1.0e-10 * sv_init[0]
            ):
                for _ in range(40):
                    kr_all, w_all = _qep_sub(_m_mat(float(np.real(k_cur))))
                    w_all_norm = w_all / np.linalg.norm(w_all, axis=0)[None, :]
                    ov = np.abs(w_all_norm.conj().T @ w_cur)
                    valid = np.where(ov > 0.5)[0]
                    if len(valid) == 0:
                        valid = np.arange(len(kr_all))
                    i_best = int(valid[np.argmin(np.abs(kr_all[valid] - k_cur))])
                    dk_abs = abs(kr_all[i_best] - k_cur)
                    k_cur = complex(kr_all[i_best])
                    w_cur = w_all_norm[:, i_best]
                    if dk_abs < 1.0e-13 * abs(k_cur):
                        break
            t_fin = _t_sub(k_cur)
            _, sv_fin, _ = np.linalg.svd(t_fin)
            vec_res = float(np.linalg.norm(t_fin @ w_cur) / sv_fin[0])
            ks_fb.append(k_cur)
            w_fb.append(w_cur)
            res_fb.append(max(float(sv_fin[-1] / sv_fin[0]), vec_res))
            tol_fb.append(_root_tol_at(k_cur, float(sv_fin[0])))
        k_waves = np.asarray(ks_fb, dtype=np.complex128)
        w_sub = np.asarray(w_fb, dtype=np.complex128).T
        root_resid = np.asarray(res_fb, dtype=np.float64)
        root_tol = np.asarray(tol_fb, dtype=np.float64)

    bad_idx = int(np.argmax(root_resid / root_tol))
    if root_resid[bad_idx] > root_tol[bad_idx]:
        raise RuntimeError(
            f"multistream_waves: root {bad_idx} at k={k_waves[bad_idx]:.6e} rad/m "
            f"has relative residual {root_resid[bad_idx]:.3e} > threshold "
            f"{root_tol[bad_idx]:.3e} (h_sub={h_sub}, eps_mach={eps_mach:.3e}, "
            f"kappa_T={root_tol[bad_idx] / (h_sub * eps_mach):.3e})."
        )

    root_k_tol = float(1.0e-13 * float(np.max(np.abs(k_waves))))
    dk_final = np.abs(k_waves[:, None] - k_waves[None, :])
    np.fill_diagonal(dk_final, np.inf)
    min_dk_final = float(np.min(dk_final))
    if min_dk_final <= root_k_tol:
        i_coinc, j_coinc = np.unravel_index(int(np.argmin(dk_final)), dk_final.shape)
        raise RuntimeError(
            f"multistream_waves: roots {i_coinc} and {j_coinc} coincide with "
            f"|dk|={min_dk_final:.3e} rad/m <= root tolerance {root_k_tol:.3e} rad/m "
            f"(k[{i_coinc}]={k_waves[i_coinc]:.6e}, k[{j_coinc}]={k_waves[j_coinc]:.6e})."
        )

    x_waves = p_basis @ w_sub

    if dc_entrance:
        theta_l = _multistream_dc_entrance_phase(
            nx=nx, dz=dz, dt=dt, params=params, dep_info=dep_info, discrete=True
        )
        e_vec = e_base.astype(np.complex128) * np.exp(-1j * theta_l)
    else:
        theta_l = np.zeros(n_lines, dtype=np.float64)
        e_vec = e_base.astype(np.complex128)

    dk_grid = _multistream_entrance_dk(k_waves, be_vec)
    pr = p_basis.T
    a_match = np.vstack([pr @ x_waves, pr @ (x_waves * (-1j * dk_grid))])
    rhs_match = np.concatenate([pr @ e_vec, np.zeros(pr.shape[0], dtype=np.complex128)])
    try:
        a0 = np.linalg.solve(a_match, rhs_match)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"multistream_waves: singular entrance matching system of shape "
            f"{a_match.shape} (min_dk={min_dk_final:.3e} rad/m)."
        ) from exc

    r_i1 = float(np.linalg.norm(x_waves @ a0 - e_vec) / np.linalg.norm(e_vec))
    r_v1 = float(
        np.linalg.norm((x_waves * dk_grid) @ a0)
        / (np.linalg.norm(be_vec) * np.linalg.norm(e_vec))
    )
    match_resid = float(max(r_i1, r_v1))
    match_tol = 1.0e-12
    if match_resid > match_tol:
        raise RuntimeError(
            f"multistream_waves: entrance matching residual match_resid={match_resid:.3e} "
            f"(r_i1={r_i1:.3e}, r_v1={r_v1:.3e}) exceeds threshold {match_tol:.3e}."
        )

    if wall_image == "none":
        wall_da = np.zeros_like(a0)
    else:
        da_neumann = _multistream_wall_correction(
            op0=op0,
            dz=dz,
            dt=dt,
            omega=omega,
            sig_vec=sig_vec,
            d_vec=d_vec,
            k_waves=k_waves,
            x_waves=x_waves,
            a_waves=a0,
            p_basis=p_basis,
            shape=shape,
            use_omega_f=use_omega_f,
            include_sz=include_sz,
            discrete=True,
        )
        wall_da = da_neumann if wall_image == "neumann" else -da_neumann

    a_waves = a0 + wall_da
    use_obs_sinc = (
        bool(include_sz and not freeze_g)
        if observable_sinc is None
        else bool(observable_sinc)
    )
    o_waves = _multistream_observable_weights(
        a_waves=a_waves,
        d_vec=d_vec,
        x_waves=x_waves,
        k_waves=k_waves,
        dz=dz,
        shape=int(shape),
        use_obs_sinc=use_obs_sinc,
    )

    return {
        "k_waves": k_waves,
        "x_waves": x_waves,
        "a_waves": a_waves,
        "o_waves": o_waves,
        "root_resid": root_resid,
        "root_tol": root_tol,
        "root_k_tol": root_k_tol,
        "match_resid": match_resid,
        "wall_da": wall_da,
        "theta_l": theta_l,
        "subspace": subspace_use,
    }


def discrete_modal_prediction(
    code: str,
    nx: int,
    nz: int,
    params: dict[str, float],
    line_x: np.ndarray | None = None,
    line_w: np.ndarray | None = None,
    entrance_profile: np.ndarray | None = None,
    include_shear: bool = True,
    include_sz: bool = True,
    use_omega_f: bool = True,
    drop_highest_mode: bool = False,
    win_shift: float = 0.0,
    omega_p_scale: float = 1.0,
    return_details: bool = False,
    shape: int = PARTICLE_SHAPE,
    method: str = "multistream",
    wall_image: str = "neumann",
    dc_entrance: bool = True,
    freeze_g: bool = False,
) -> float | dict[str, Any]:
    """Compute the Step 2 discrete space-charge wave prediction ``beta_q,pred`` [rad/m].

    For ``code == "particles"``, ``method="multistream"`` (default, Decision D20) solves
    the exact ``N``-line coupled multistream boundary-value problem via
    :func:`multistream_waves`:

    1. Roots ``(k_n, X_n)`` of ``[diag((k - beta_l)**2) - S_z(k)**2 diag(sigma_l) G(k)] X = 0``;
    2. Entrance matching ``sum_n a_n X_n = e * exp(-i * theta_l)`` and
       ``sum_n a_n (k_n - beta_l) X_{l,n} = 0``;
    3. First-order (Born) PEC-wall near-field correction ``a <- a + wall_da`` and
       deposited-current observable ``I_1(z) = I_{1,0} sum_n o_n exp(-i * k_n * z)`` with
       ``o_n = a_n * sinc(k_n * dz / 2)**p * (d^T X_n)``.

    Passing ``method="modal"`` evaluates the legacy uncoupled cos-form modal synthesis
    ``I_1(z) = I_{1,0} sum_j c_j cos(beta_{q,j} * z) exp(-i * k_{c,j} * z)``. For
    ``code == "lb"``, the `LinearizedBeam` modal dispersion is always used.

    Limits & Failure Modes
    ----------------------
    - **Linear in modulation index ``m``**: predicts the ``m -> 0`` linear wave number
      ``beta_0``; ``O(m**2)`` nonlinear bunching shifts are bounded separately by ``eps_ac``.
    - **Discrete-particle alias/Doppler terms not modelled**: retains only the principal
      ``m = 0`` axial spatial harmonic ``S_z(k)**2``.
    - **PEC-wall near-field term is first order**: the ``z = 0`` wall correction in
      ``method="multistream"`` is first-order (Born) in the evanescent image field.

    Parameters
    ----------
    code : {"lb", "particles"}
        Code selector (`"lb"` for `LinearizedBeam`, `"particles"` for `ParticleSolver`).
    nx, nz : int
        Transverse and axial cell counts [-].
    params : dict of str to float
        Output of :func:`compute_analytic_parameters`.
    line_x : np.ndarray, optional
        Injected particle line ``x`` positions [m] (for ``code == "particles"``).
    line_w : np.ndarray, optional
        Line charge weights [-] (for non-uniform profiles such as D3).
    entrance_profile : np.ndarray, optional
        Entrance modulation profile across lines [-] (defaults to ``_w_perp_eff`` for
        `LinearizedBeam` and uniform charge per line for `ParticleSolver`).
    include_shear : bool, optional
        For ``code == "particles"``, include the per-line DC velocity shear
        ``(u_l, gamma_l)`` from Step 1. Default is ``True``.
    include_sz : bool, optional
        For ``code == "particles"``, include the drifting-beam principal-mode axial
        shape factor ``S_z(k)**2 = sinc(k * dz / 2)**(2 * p)`` (Step 3). Default is ``True``.
    use_omega_f : bool, optional
        If ``True`` (default), use the leapfrog field time symbol ``Omega_f`` in
        ``tau_Y(k)**2 = kappa(k)**2 - (Omega_f / c)**2``; if ``False``, use
        continuum ``omega`` (for ``s_sym``).
    drop_highest_mode : bool, optional
        If ``True``, omit the highest-order excited transverse mode (used for ``s_modes``).
    win_shift : float, optional
        Shift [m] applied to both bounds of ``fit_window(params)`` (used for ``s_win``).
    omega_p_scale : float, optional
        Multiplier on ``omega_p`` (``sqrt(2)`` for M2) [-]. Default is ``1.0``.
    return_details : bool, optional
        If ``True``, return a dictionary with modal eigenvalues, wavenumbers,
        weights, multistream waves, and synthesized ``I1_model(z)``. Default is ``False``.
    shape : int, optional
        B-spline particle shape order ``p`` in ``{1, 2, 3}`` for ``code == "particles"``.
        Default is ``PARTICLE_SHAPE``.
    method : {"multistream", "modal"}, optional
        Prediction method for ``code == "particles"``. Default is ``"multistream"``.
    wall_image : {"neumann", "none", "dirichlet"}, optional
        PEC wall near-field treatment passed to :func:`multistream_waves` when
        ``method == "multistream"``. Default is ``"neumann"``.
    dc_entrance : bool, optional
        Whether to include the DC entrance transit phase ``theta_l`` in
        :func:`multistream_waves` when ``method == "multistream"``. Default is ``True``.
    freeze_g : bool, optional
        If ``True``, freeze ``G(k)`` at ``k_ref = omega / u_bar`` for ``code == "particles"``.
        Default is ``False``.

    Returns
    -------
    float or dict of str to Any
        Predicted two-wave fitted ``beta_q`` [rad/m] (or detail dictionary when
        ``return_details=True``).
    """
    if method not in ("multistream", "modal"):
        raise ValueError(
            f"Unknown method={method!r}; expected 'multistream' or 'modal'."
        )
    b = BEAM_HALF_THICKNESS_M
    lz = TUNNEL_LENGTH_M
    dz = lz / nz

    dt, _ = _cached_integer_period_dt(int(nx), int(nz))
    omega = params["omega"]
    ms_info: dict[str, Any] | None = None
    mode_assign: np.ndarray | None = None

    if code == "lb":
        # LB has no DC space-charge physics; keep uniform u_dep and beta_e (including D3)
        beta_e = params["beta_e"]
        op0 = compute_transverse_modal_operator(
            "lb",
            nx,
            nz,
            params,
            k_val=beta_e,
            line_x=line_x,
            line_w=line_w,
            use_omega_f=use_omega_f,
        )
        evals_be = op0["evals"]
        d_vec = op0["d_vec"]
        w_eff = op0["w_perp_eff"]
        keep_int = op0["keep_int"]
        if entrance_profile is None:
            ent_raw = w_eff.copy()
        else:
            ep = np.asarray(entrance_profile, dtype=np.float64)
            if ep.ndim == 2:
                ep = ep[0, :]
            if len(ep) == nx - 1:
                ent_raw = ep[keep_int]
            elif len(ep) == nx + 1:
                ent_raw = ep[1:-1][keep_int]
            else:
                ent_raw = ep.copy()
        ent_norm = ent_raw * (2.0 * b / (dx_lb := op0["dx"])) / np.sum(ent_raw)
        entrance = ent_norm / np.maximum(d_vec * (2.0 * b / dx_lb), 1e-30)

        u_dep = params["u_dep"]
        gamma_dep = params["gamma_dep"]
        omega_p_dep = params["omega_p"] * float(omega_p_scale)
        wp_fac = omega_p_dep / (gamma_dep**1.5)
        omega_lhs = (2.0 / dt) * np.tan(0.5 * omega * dt)
        k_lo = 0.5 * beta_e
        k_hi = 0.99 * np.pi / (2.0 * dz)

        active = evals_be > 1e-12 * evals_be[0]
        n_act = int(np.sum(active))
        y_r0 = op0["y_right"][:, :n_act]
        y_l0 = op0["y_left"][:, :n_act]
        bq_modes = np.zeros(n_act, dtype=np.float64)
        kc_modes = np.zeros(n_act, dtype=np.float64)
        c_modes = np.zeros(n_act, dtype=np.float64)

        def _lam_j_lb(k_val: float, j_idx: int) -> float:
            op_k = compute_transverse_modal_operator(
                "lb",
                nx,
                nz,
                params,
                k_val=k_val,
                line_w=line_w,
                use_omega_f=use_omega_f,
            )
            return float(op_k["evals"][j_idx])

        for j in range(n_act):
            km = float(
                brentq(
                    lambda k, jj=j: (
                        omega_lhs
                        - wp_fac * np.sqrt(_lam_j_lb(k, jj))
                        - u_dep * np.sin(k * dz) / dz
                    ),
                    k_lo,
                    k_hi,
                )
            )
            kp = float(
                brentq(
                    lambda k, jj=j: (
                        omega_lhs
                        + wp_fac * np.sqrt(_lam_j_lb(k, jj))
                        - u_dep * np.sin(k * dz) / dz
                    ),
                    k_lo,
                    k_hi,
                )
            )
            bq_modes[j] = 0.5 * (kp - km)
            kc_modes[j] = 0.5 * (kp + km)
            c_modes[j] = float(np.dot(d_vec, y_r0[:, j]) * np.dot(y_l0[:, j], entrance))

    else:
        # Step 2: compute discrete DC potential, per-line (u_l, gamma_l) and mean delta
        # from the actual (line_x, line_w) passed to the prediction; never reuse uniform params["u_dep"].
        dep_info = compute_discrete_dc_depression(
            nx,
            params,
            line_x=line_x,
            line_w=line_w,
            return_details=True,
            shape=shape,
        )
        u_l = dep_info["u_l"]
        gamma_l = dep_info["gamma_l"]
        delta_disc = float(dep_info["delta_disc"])
        u0 = params["u0"]
        u_bar = float(u0 * (1.0 - delta_disc))
        gamma_bar = float(1.0 / np.sqrt(1.0 - (u_bar / c_light) ** 2))
        beta_e_part = float(omega / u_bar)
        omega_p0 = params["omega_p0"] * float(omega_p_scale)
        omega_p_dt = (2.0 / dt) * np.sin(0.5 * omega * dt)

        op0 = compute_transverse_modal_operator(
            "particles",
            nx,
            nz,
            params,
            k_val=beta_e_part,
            line_x=line_x,
            line_w=line_w,
            use_omega_f=use_omega_f,
            shape=shape,
        )
        evals_be = op0["evals"]
        d_vec = op0["d_vec"]

        # Per-line plasma frequency squared omega_{p,l}^2 = omega_p0^2 * (u0 / u_l) (density prop 1 / u_l)
        omega_p_l_sq = (omega_p0**2) * (u0 / u_l)
        if include_shear:
            sig_vec = omega_p_l_sq / (gamma_l**3 * u_l**2)
            be_vec = omega_p_dt / u_l
        else:
            omega_p_bar_sq = (omega_p0**2) * (u0 / u_bar)
            sig_vec = np.full(
                len(u_l),
                omega_p_bar_sq / (gamma_bar**3 * u_bar**2),
                dtype=np.float64,
            )
            be_vec = np.full(len(u_l), omega_p_dt / u_bar, dtype=np.float64)

        s_left = np.sqrt(d_vec / sig_vec)
        s_right = np.sqrt(sig_vec / d_vec)

        def _eig_part_at_k(
            k_val: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            op_k = compute_transverse_modal_operator(
                "particles",
                nx,
                nz,
                params,
                k_val=k_val,
                line_x=line_x,
                line_w=line_w,
                use_omega_f=use_omega_f,
                shape=shape,
            )
            # Drifting-beam axial shape factor S_z(k)^2 = sinc^(2p)(k*dz/2) from
            # order-p Esirkepov J_z deposit symbol sinc^p(k*dz/2) and order-(p-1)
            # Galerkin E_z gather symbol sinc^p(k*dz/2) (principal m=0 mode; aliases
            # k_m = k + 2*pi*m/dz are Doppler-detuned by 2*pi*|m|*u_dep/dz >> omega_q).
            sz_sq = (
                float(np.sinc(0.5 * k_val * dz / np.pi) ** (2 * int(shape)))
                if include_sz
                else 1.0
            )
            g_tilde = (sz_sq * sig_vec)[:, None] * op_k["G"]
            m_sym = (s_left[:, None] * g_tilde) * s_right[None, :]
            m_sym = 0.5 * (m_sym + m_sym.T)
            ev, qv = np.linalg.eigh(m_sym)
            idx = np.argsort(ev)[::-1]
            ev = np.maximum(ev[idx], 0.0)
            qv = qv[:, idx]
            return ev, s_right[:, None] * qv, s_left[:, None] * qv

        ev0, y_r0, y_l0 = _eig_part_at_k(beta_e_part)
        active = ev0 > 1e-12 * ev0[0]
        n_act = int(np.sum(active))
        if entrance_profile is None:
            entrance = np.ones(len(d_vec), dtype=np.float64)
        else:
            ep = np.asarray(entrance_profile, dtype=np.float64)
            ep_norm = ep / np.sum(ep)
            entrance = ep_norm / np.maximum(d_vec, 1e-30)

        bq_modes = np.zeros(n_act, dtype=np.float64)
        kc_modes = np.zeros(n_act, dtype=np.float64)
        c_modes = np.zeros(n_act, dtype=np.float64)

        for j in range(n_act):
            yj = y_r0[:, j]
            lj = y_l0[:, j]
            be_j = float(np.sum((lj * yj) * be_vec))
            bq_init = float(np.sqrt(ev0[j]))
            km = be_j - bq_init
            kp = be_j + bq_init
            if not freeze_g:
                for _ in range(12):
                    evm, _, _ = _eig_part_at_k(km)
                    evp, _, _ = _eig_part_at_k(kp)
                    km_next = be_j - float(np.sqrt(evm[j]))
                    kp_next = be_j + float(np.sqrt(evp[j]))
                    if abs(km_next - km) < 1e-13 and abs(kp_next - kp) < 1e-13:
                        km, kp = km_next, kp_next
                        break
                    km, kp = km_next, kp_next
            bq_modes[j] = 0.5 * (kp - km)
            kc_modes[j] = 0.5 * (kp + km)
            c_modes[j] = float(np.dot(d_vec, yj) * np.dot(lj, entrance))

        if method == "multistream":
            ms_info = multistream_waves(
                nx,
                nz,
                params,
                line_x=line_x,
                line_w=line_w,
                shape=shape,
                use_omega_f=use_omega_f,
                include_sz=include_sz,
                include_shear=include_shear,
                omega_p_scale=omega_p_scale,
                wall_image=wall_image,
                dc_entrance=dc_entrance,
                entrance_profile=entrance_profile,
                freeze_g=freeze_g,
            )
            # Assign each wave to the transverse mode j at k_ref with maximal |y_left,j^T X_n|
            overlaps = np.abs(y_l0.T @ ms_info["x_waves"])
            mode_assign = np.argmax(overlaps, axis=0)

    z_eval = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    if code == "particles" and method == "multistream" and ms_info is not None:
        o_use = ms_info["o_waves"].copy()
        if drop_highest_mode and mode_assign is not None:
            excited = np.where(np.abs(c_modes) > 1e-12)[0]
            if len(excited) > 1:
                o_use[mode_assign == int(excited[-1])] = 0.0
        i1_model = (
            params["I1_0"] * np.exp(-1j * np.outer(z_eval, ms_info["k_waves"])) @ o_use
        )
    else:
        c_use = c_modes.copy()
        if drop_highest_mode:
            excited = np.where(np.abs(c_use) > 1e-12)[0]
            if len(excited) > 1:
                c_use[int(excited[-1])] = 0.0
        i1_model = params["I1_0"] * np.sum(
            c_use[:, None]
            * np.cos(np.outer(bq_modes, z_eval))
            * np.exp(-1j * np.outer(kc_modes, z_eval)),
            axis=0,
        )

    z_min, z_max = fit_window(params)
    fit_res = fit_beta_q_from_envelope(
        z_eval,
        i1_model,
        float(bq_modes[0]),
        z_min=z_min + float(win_shift),
        z_max=z_max + float(win_shift),
    )
    bq_pred = float(fit_res["beta_q"])

    if return_details:
        det: dict[str, Any] = {
            "beta_q": bq_pred,
            "evals": evals_be,
            "bq_modes": bq_modes,
            "kc_modes": kc_modes,
            "c_modes": c_modes,
            "z_coords": z_eval,
            "i1_model": i1_model,
            "fit": fit_res,
        }
        if ms_info is not None:
            det.update(
                k_waves=ms_info["k_waves"],
                x_waves=ms_info["x_waves"],
                a_waves=ms_info["a_waves"],
                o_waves=ms_info["o_waves"],
                root_resid=ms_info["root_resid"],
                root_tol=ms_info["root_tol"],
                root_k_tol=ms_info["root_k_tol"],
                match_resid=ms_info["match_resid"],
                wall_da=ms_info["wall_da"],
                theta_l=ms_info["theta_l"],
                subspace=ms_info["subspace"],
                mode_assign=mode_assign,
            )
        return det
    return bq_pred


def prediction_slack(
    code: str,
    nx: int,
    nz: int,
    params: dict[str, float],
    line_x: np.ndarray | None = None,
    line_w: np.ndarray | None = None,
    entrance_profile: np.ndarray | None = None,
    omega_p_scale: float = 1.0,
    shape: int = PARTICLE_SHAPE,
    method: str = "multistream",
) -> dict[str, Any]:
    """Compute the Step 3 parameter-free prediction slack ``s_win + s_modes + s_sym + s_round``."""
    base = discrete_modal_prediction(
        code,
        nx,
        nz,
        params,
        line_x=line_x,
        line_w=line_w,
        entrance_profile=entrance_profile,
        include_shear=True,
        use_omega_f=True,
        drop_highest_mode=False,
        win_shift=0.0,
        omega_p_scale=omega_p_scale,
        return_details=True,
        shape=shape,
        method=method,
    )
    bq_pred = float(base["beta_q"])
    z_eval = base["z_coords"]
    i1_model = base["i1_model"]
    bq_modes = base["bq_modes"]
    kc_modes = base["kc_modes"]
    c_modes = base["c_modes"]
    z_min, z_max = fit_window(params)
    shift_m = 1.0 / params["T1"]

    # s_win: maximum sensitivity to shifting the fit window by +-1/T1
    bq_win_plus = float(
        fit_beta_q_from_envelope(
            z_eval,
            i1_model,
            float(bq_modes[0]),
            z_min=z_min + shift_m,
            z_max=z_max + shift_m,
        )["beta_q"]
    )
    bq_win_minus = float(
        fit_beta_q_from_envelope(
            z_eval,
            i1_model,
            float(bq_modes[0]),
            z_min=z_min - shift_m,
            z_max=z_max - shift_m,
        )["beta_q"]
    )
    s_win = float(
        max(abs(bq_win_plus - bq_pred), abs(bq_win_minus - bq_pred)) / bq_pred
    )

    # s_modes: change when dropping the highest excited transverse mode
    c_trunc = c_modes.copy()
    excited = np.where(np.abs(c_trunc) > 1e-12)[0]
    if len(excited) > 1:
        if code == "particles" and method == "multistream":
            o_trunc = base["o_waves"].copy()
            o_trunc[base["mode_assign"] == int(excited[-1])] = 0.0
            i1_trunc = (
                params["I1_0"]
                * np.exp(-1j * np.outer(z_eval, base["k_waves"]))
                @ o_trunc
            )
        else:
            c_trunc[int(excited[-1])] = 0.0
            i1_trunc = params["I1_0"] * np.sum(
                c_trunc[:, None]
                * np.cos(np.outer(bq_modes, z_eval))
                * np.exp(-1j * np.outer(kc_modes, z_eval)),
                axis=0,
            )
        bq_no_high = float(
            fit_beta_q_from_envelope(
                z_eval, i1_trunc, float(bq_modes[0]), z_min=z_min, z_max=z_max
            )["beta_q"]
        )
        s_modes = float(abs(bq_pred - bq_no_high) / bq_pred)
    else:
        s_modes = 0.0

    # s_sym: time-symbol ambiguity |beta_pred(Omega_f in tau_Y^2) - beta_pred(omega in tau_Y^2)| / beta_pred
    bq_omega_cont = float(
        discrete_modal_prediction(
            code,
            nx,
            nz,
            params,
            line_x=line_x,
            line_w=line_w,
            entrance_profile=entrance_profile,
            include_shear=True,
            use_omega_f=False,
            drop_highest_mode=False,
            win_shift=0.0,
            omega_p_scale=omega_p_scale,
            return_details=False,
            shape=shape,
            method=method,
        )
    )
    s_sym = float(abs(bq_pred - bq_omega_cont) / bq_pred)
    s_round = 1.0e-12
    slack = float(s_win + s_modes + s_sym + s_round)

    res: dict[str, Any] = {
        "bq_pred": bq_pred,
        "s_win": s_win,
        "s_modes": s_modes,
        "s_sym": s_sym,
        "s_round": s_round,
        "slack": slack,
        "evals": base["evals"],
        "bq_modes": bq_modes,
        "kc_modes": kc_modes,
        "c_modes": c_modes,
        "z_coords": z_eval,
        "i1_model": i1_model,
    }
    if code == "particles":
        if "root_resid" in base:
            res["root_resid"] = base["root_resid"]
            res["root_tol"] = base["root_tol"]
            res["root_k_tol"] = base["root_k_tol"]
            res["match_resid"] = float(base["match_resid"])
        bq_unif = float(
            discrete_modal_prediction(
                "particles",
                nx,
                nz,
                params,
                line_x=line_x,
                line_w=line_w,
                entrance_profile=entrance_profile,
                include_shear=False,
                include_sz=True,
                use_omega_f=True,
                drop_highest_mode=False,
                win_shift=0.0,
                omega_p_scale=omega_p_scale,
                return_details=False,
                shape=shape,
                method=method,
            )
        )
        bq_no_sz = float(
            discrete_modal_prediction(
                "particles",
                nx,
                nz,
                params,
                line_x=line_x,
                line_w=line_w,
                entrance_profile=entrance_profile,
                include_shear=True,
                include_sz=False,
                use_omega_f=True,
                drop_highest_mode=False,
                win_shift=0.0,
                omega_p_scale=omega_p_scale,
                return_details=False,
                shape=shape,
                method=method,
            )
        )
        res["bq_pred_unif"] = bq_unif
        res["bq_pred_no_sz"] = bq_no_sz
        res["shear_shift"] = float(bq_pred - bq_unif)
        res["shear_rel"] = float(bq_pred / bq_unif - 1.0)
    return res


def _cardinal_bspline(d: np.ndarray | float, p: int) -> np.ndarray:
    """Evaluate the centred cardinal B-spline ``W_p(d)`` of degree ``p`` at cell offset ``d`` [-].

    Parameters
    ----------
    d : np.ndarray or float
        Cell-normalised coordinate offset ``d = (z - z_m) / dz`` [-].
    p : int
        B-spline degree ``p`` in ``{1, 2, 3}`` [-].

    Returns
    -------
    np.ndarray
        Centred cardinal B-spline value ``W_p(d)`` [-].
    """
    p_int = int(p)
    abs_d = np.abs(np.asarray(d, dtype=np.float64))
    if p_int == 1:
        return np.where(abs_d < 1.0, 1.0 - abs_d, 0.0)
    if p_int == 2:
        return np.where(
            abs_d <= 0.5,
            0.75 - abs_d * abs_d,
            np.where(abs_d < 1.5, 0.5 * (1.5 - abs_d) ** 2, 0.0),
        )
    if p_int == 3:
        return np.where(
            abs_d <= 1.0,
            (4.0 - 6.0 * abs_d * abs_d + 3.0 * (abs_d**3)) / 6.0,
            np.where(abs_d < 2.0, ((2.0 - abs_d) ** 3) / 6.0, 0.0),
        )
    raise ValueError(f"Unsupported B-spline degree p={p!r}; expected 1, 2, or 3.")


def _charge_tail_fraction(zeta: np.ndarray | float, k: int, p: int) -> np.ndarray:
    """Evaluate ``C_k(zeta) = sum_{m >= k + 1} W_p(zeta - m)`` at cell coordinate ``zeta = z / dz`` [-].

    Parameters
    ----------
    zeta : np.ndarray or float
        Particle axial position in cell units ``z / dz`` [-].
    k : int
        Axial ``E_z`` edge index ``k`` (spanning between node ``k`` and ``k + 1``) [-].
    p : int
        B-spline degree ``p`` in ``{1, 2, 3}`` [-].

    Returns
    -------
    np.ndarray
        Charge fraction ``C_k(zeta)`` deposited on primal nodes ``m >= k + 1`` [-].
    """
    zeta_arr = np.asarray(zeta, dtype=np.float64)
    m_min = int(k) + 1
    m_max = int(np.floor(float(np.max(zeta_arr)) + 0.5 * (int(p) + 1))) + 1
    if m_max < m_min:
        return np.zeros_like(zeta_arr, dtype=np.float64)
    m_nodes = np.arange(m_min, m_max + 1, dtype=np.float64)
    return np.sum(_cardinal_bspline(zeta_arr[..., None] - m_nodes, int(p)), axis=-1)


def _entrance_edge_transfer_complex(
    p: int,
    k: int,
    dz: float,
    dt: float,
    omega: float,
    u: float,
    wall: bool = True,
    z0_offset_cells: float = 0.0,
) -> complex:
    """Compute the complex discrete-time order-``p`` Esirkepov current transfer sum on edge ``k`` [-]."""
    p_int = int(p)
    k_int = int(k)
    dz_f = float(dz)
    dt_f = float(dt)
    omega_f = float(omega)
    u_f = float(u)
    offset_f = float(z0_offset_cells)
    if p_int not in (1, 2, 3):
        raise ValueError(f"Unsupported B-spline degree p={p!r}; expected 1, 2, or 3.")

    z0_base = 0.0 if wall else -float(k_int + p_int + 3) * dz_f
    z0 = z0_base if offset_f == 0.0 else (z0_base + offset_f * dz_f)
    z_end = float(k_int + p_int + 2) * dz_f
    step_z = u_f * dt_f
    n_steps = int(np.ceil((z_end - z0) / step_z))
    while (z0 + step_z * n_steps) / dz_f < float(k_int + p_int + 2):
        n_steps += 1

    j_nodes = np.arange(n_steps + 1, dtype=np.float64)
    zeta_j = (z0 + step_z * j_nodes) / dz_f
    ck_j = _charge_tail_fraction(zeta_j, k_int, p_int)
    dc_j = ck_j[1:] - ck_j[:-1]
    if wall:
        dc_j[0] += float(_charge_tail_fraction(0.0, k_int, p_int))
    j_idx = np.arange(n_steps, dtype=np.float64)
    h_sum = np.sum(dc_j * np.exp(-1j * omega_f * dt_f * j_idx))
    if offset_f != 0.0:
        h_sum = h_sum * np.exp(-1j * (omega_f * dz_f / u_f) * offset_f)
    return complex(h_sum)


def entrance_edge_transfer(
    p: int,
    k: int,
    dz: float,
    dt: float,
    omega: float,
    u: float,
    wall: bool = True,
    z0_offset_cells: float = 0.0,
) -> float:
    """Compute the discrete-time order-``p`` Esirkepov current transfer amplitude ``H_k`` on edge ``k`` [-].

    Evaluates the exact discrete-time unperturbed (no-plasma) transfer modulus::

        C_k(zeta) = sum_{m >= k + 1} W_p(zeta - m)
        dC_j = C_k(z_{j + 1} / dz) - C_k(z_j / dz),   j = 0 .. n - 1
        dC_0 += C_k(0)                                (when wall=True)
        H_k = | sum_{j = 0}^{n - 1} dC_j * exp(-i * omega * dt * j) |

    for a particle moving at uniform velocity ``u`` along ``z_j = z_0 + u * dt * j``
    up to ``z_n / dz >= k + p + 2``, starting at ``z_0 = 0`` with PEC birth closure
    when ``wall=True`` or far upstream at ``z_0 = -(k + p + 3) * dz`` without
    closure when ``wall=False``.

    Parameters
    ----------
    p : int
        B-spline shape order ``p`` in ``{1, 2, 3}`` [-].
    k : int
        Axial ``E_z`` edge index ``k`` [-].
    dz : float
        Axial cell size [m].
    dt : float
        Time step [s].
    omega : float
        Angular modulation frequency [rad/s].
    u : float
        Uniform axial drift velocity [m/s].
    wall : bool, optional
        If ``True`` (default), start at ``z_0 = 0`` and add the PEC wall birth
        closure ``C_k(0)`` to ``dC_0``. If ``False``, start far upstream at
        ``z_0 = -(k + p + 3) * dz`` without closure (bulk kernel).
    z0_offset_cells : float, optional
        Sub-step start-phase offset ``zeta_0`` in cell units [cells] added to
        ``z_0 / dz`` for bulk phase-alias sampling. Default is ``0.0``.

    Returns
    -------
    float
        Transfer amplitude ``H_k`` [-].
    """
    return float(
        np.abs(
            _entrance_edge_transfer_complex(
                p=p,
                k=k,
                dz=dz,
                dt=dt,
                omega=omega,
                u=u,
                wall=wall,
                z0_offset_cells=z0_offset_cells,
            )
        )
    )


def _single_grid_budget_terms(
    nx: int,
    nz: int,
    params: dict[str, float],
    omega_p_scale: float = 1.0,
    shape: int = PARTICLE_SHAPE,
    cfln: float = CFL_NUMBER,
) -> dict[str, Any]:
    """Compute Step 2 modal predictions and Step 3 slacks for grid ``(nx, 1, nz)``."""
    dz = TUNNEL_LENGTH_M / nz
    beta_e = params["beta_e"]
    beta_q = params["beta_q"] * float(omega_p_scale)

    gamma1_disc, d_w_grid, _ = _grid_gamma1_disc_and_dw(
        int(nx), int(nz), params["omega"]
    )
    gamma1 = gamma1_disc
    kappa_dc = float(params["kappa_dc"])
    d_w_common = float(params["d_w_common"])
    d_w = d_w_common
    z_min, z_max = fit_window(params)
    eps_w_bias = float(EPS_WINDOW * (1.0 / gamma1_disc) / (z_max - z_min))

    lb_modal = prediction_slack("lb", nx, nz, params, omega_p_scale=omega_p_scale)
    part_modal = prediction_slack(
        "particles", nx, nz, params, omega_p_scale=omega_p_scale, shape=shape
    )

    # Brillouin alias suppression factor (omega_q * dz / (2 * pi * u_dep))^2 for m != 0:
    omega_q_eff = params["omega_q"] * float(omega_p_scale)
    alias_supp = float((omega_q_eff * dz / (2.0 * np.pi * params["u_dep"])) ** 2)

    # Discrete 1D Poisson DC depression prediction on this grid (Step 1):
    dep_info = compute_discrete_dc_depression(
        nx, params, return_details=True, shape=shape
    )
    delta_th_disc = float(dep_info["delta_disc"])
    eps_ac = float(0.5 * (MODULATION_INDEX * beta_q / beta_e) ** 2)

    # Entrance amplitude prediction and bound at first primal edge k = 0 (midpoint z_0 = 0.5 * dz):
    # At the PEC entrance wall z = 0, the order-p edge-0 kernel is truncated at z < 0 and
    # closed by the birth closure C_0(0), giving H_0 * cos(beta_q * dz / 2) with analytic bound
    # 2 * (v_inj / u_dep - 1) * p * (beta_e * dz)^2 / 24 + (1 - cos(beta_q * dz / 2)) + 1e-12.
    tol_amp_lb = float(0.5 * (beta_q * dz) ** 2)
    bq_half = 0.5 * beta_q * dz
    dt_grid, _ = _cached_integer_period_dt(int(nx), int(nz), cfln=float(cfln))
    h0_part = entrance_edge_transfer(
        int(shape),
        0,
        dz,
        dt_grid,
        params["omega"],
        params["u_dep"],
        wall=True,
    )
    v_inj = float(c_light * np.sqrt(1.0 - 1.0 / (params["gamma0"] ** 2)))
    amp_pred_part = float(h0_part * np.cos(bq_half))
    tol_amp_part = float(
        2.0 * (v_inj / params["u_dep"] - 1.0) * int(shape) * ((beta_e * dz) ** 2) / 24.0
        + (1.0 - np.cos(bq_half))
        + 1.0e-12
    )

    # Residual DFT leakage bound (identity / round-off):
    eps_dft = float(100.0 * np.finfo(np.float64).eps / MODULATION_INDEX)

    # LinearizedBeam 2*omega C^1 turn-on ramp transient tail:
    n_rise = float(RISE_PERIODS)
    n_per = 6.0
    tol_i2_lb = float(1.0 / ((2.0 * n_rise) ** 2 * np.pi * n_per))

    # Two-sided 99.73% Student-t critical value for 8 seeds (dof = 7):
    tcrit = float(student_t.ppf(0.99865, len(NOISE_SEEDS) - 1))

    return {
        "gamma1": gamma1,
        "gamma1_disc": gamma1_disc,
        "kappa_dc": kappa_dc,
        "eps_w": float(EPS_WINDOW),
        "eps_w_bias": eps_w_bias,
        "d_w": d_w,
        "d_w_grid": d_w_grid,
        "d_w_common": d_w_common,
        "z_min": float(z_min),
        "z_max": float(z_max),
        "bq_pred_lb": float(lb_modal["bq_pred"]),
        "s_win_lb": float(lb_modal["s_win"]),
        "s_modes_lb": float(lb_modal["s_modes"]),
        "s_sym_lb": float(lb_modal["s_sym"]),
        "s_round": float(lb_modal["s_round"]),
        "slack_lb": float(lb_modal["slack"]),
        "evals_lb": lb_modal["evals"],
        "i1_model_lb": lb_modal["i1_model"],
        "bq_pred_part": float(part_modal["bq_pred"]),
        "bq_pred_part_unif": float(part_modal["bq_pred_unif"]),
        "bq_pred_part_no_sz": float(part_modal["bq_pred_no_sz"]),
        "shear_shift": float(part_modal["shear_shift"]),
        "shear_rel": float(part_modal["shear_rel"]),
        "s_win_part": float(part_modal["s_win"]),
        "s_modes_part": float(part_modal["s_modes"]),
        "s_sym_part": float(part_modal["s_sym"]),
        "slack_part": float(part_modal["slack"]),
        "evals_part": part_modal["evals"],
        "i1_model_part": part_modal["i1_model"],
        "root_resid_part": part_modal.get("root_resid"),
        "root_tol_part": part_modal.get("root_tol"),
        "match_resid_part": part_modal.get("match_resid"),
        "slack_cross": float(lb_modal["slack"] + part_modal["slack"]),
        "alias_supp": alias_supp,
        "delta_th": float(params["delta_u_dc"]),
        "delta_th_flux": float(params["delta_u_dc_flux"]),
        "delta_th_density": float(params["delta_u_dc_density"]),
        "delta_th_disc": delta_th_disc,
        "delta_disc_exact": delta_th_disc,
        "delta_disc_flux": float(dep_info["delta_flux"]),
        "delta_disc_density": float(dep_info["delta_density"]),
        "delta_disc_charge": float(dep_info["delta_charge"]),
        "dep_weighting": str(dep_info["weighting"]),
        "eps_ac": eps_ac,
        "amp_pred_part": amp_pred_part,
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
    shape: int = PARTICLE_SHAPE,
) -> dict[str, Any]:
    """Build the Step 2 modal predictions and Step 3 slacks for grid ``(nx, 1, nz)``."""
    return _single_grid_budget_terms(
        nx, nz, params, omega_p_scale=omega_p_scale, shape=shape
    )


def format_error_budget(nx: int, nz: int, tols: dict[str, Any]) -> str:
    """Format every Step 2 modal prediction and Step 3 slack term for grid ``(nx, 1, nz)``."""
    ev_lb = tols["evals_lb"]
    ev_p = tols["evals_part"]
    lines = [
        f"Modal prediction and slack budget for grid ({nx}, 1, {nz}):",
        f"  gamma1_disc / kappa_dc / d_w_grid / d_w_common    : gamma1_disc={tols['gamma1_disc']:.4f} 1/m, kappa_dc={tols['kappa_dc']:.4f} 1/m, d_w_grid={tols['d_w_grid'] * 1e3:.4f} mm, d_w_common={tols['d_w_common'] * 1e3:.4f} mm, [{tols['z_min'] * 1e3:.4f}, {tols['z_max'] * 1e3:.4f}] mm, eps_w_bias={tols['eps_w_bias']:.6e}",
        f"  bq_pred_lb (Step 2 LB modal prediction)           : {tols['bq_pred_lb']:.6f} rad/m",
        f"  slack_lb = s_win + s_modes + s_sym + s_round      : {tols['slack_lb']:.6%}",
        f"    (s_win={tols['s_win_lb']:.3e}, s_modes={tols['s_modes_lb']:.3e}, s_sym={tols['s_sym_lb']:.3e}, s_round={tols['s_round']:.1e})",
        f"  LB eigenvalues lambda_1..4(beta_e)                : ({ev_lb[0]:.6f}, {ev_lb[1]:.6f}, {ev_lb[2]:.6f}, {ev_lb[3]:.6f})",
        f"  bq_pred_part (D20 Particle multistream prediction): {tols['bq_pred_part']:.6f} rad/m (no_Sz={tols['bq_pred_part_no_sz']:.6f}, unif={tols['bq_pred_part_unif']:.6f}, shear={tols['shear_shift']:+.6e})",
        f"  slack_part = s_win + s_modes + s_sym + s_round    : {tols['slack_part']:.6%}",
        f"    (s_win={tols['s_win_part']:.3e}, s_modes={tols['s_modes_part']:.3e}, s_sym={tols['s_sym_part']:.3e}, s_round={tols['s_round']:.1e})",
        f"  Particle eigenvalues lambda_1..4(beta_e)          : ({ev_p[0]:.6f}, {ev_p[1]:.6f}, {ev_p[2]:.6f}, {ev_p[3]:.6f})",
        f"  alias_supp ((omega_q*dz/(2*pi*u_dep))^2)          : {tols['alias_supp']:.6e}",
        f"  slack_cross (slack_lb + slack_part)               : {tols['slack_cross']:.6%}",
        f"  delta_th (continuum DC dep, flux/density)         : {tols['delta_th_flux']:.6%} / {tols['delta_th_density']:.6%}",
        f"  delta_disc_exact (1D Poisson DC dep, {tols['dep_weighting']})        : {tols['delta_disc_exact']:.6%} (flux={tols['delta_disc_flux']:.6%}, density={tols['delta_disc_density']:.6%})",
        f"  eps_ac (0.5*(m*bq/be)^2)                          : {tols['eps_ac']:.6e}",
        f"  amp_pred_part (H_0(wall)*cos(bq*dz/2))            : {tols['amp_pred_part']:.8f}",
        f"  tol_amp_lb (0.5*(bq*dz)^2)                        : {tols['tol_amp_lb']:.6e}",
        f"  tol_amp_part (2*(v_inj/u_dep-1)*p*(be*dz)^2/24+(1-cos(bq*dz/2))+1e-12): {tols['tol_amp_part']:.6e}",
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


def fit_four_wave_reflection(
    z_coords: np.ndarray,
    i1_phasor: np.ndarray,
    beta_q_guess: float,
    z_min: float | None = None,
    z_max: float | None = None,
) -> dict[str, Any]:
    """Refit complex phasor ``i1_phasor(z)`` with a four-wave forward+backward variable-projection model.

    Adds the reflected/backward ``exp(+i * k_f * z)`` and ``exp(+i * k_s * z)``
    partner waves to the two-wave forward design matrix on ``[z_min, z_max]``
    and solves for ``(k_c, beta_q)`` and the four complex wave amplitudes
    ``(a_f_fwd, a_s_fwd, a_f_bwd, a_s_bwd)``.
    """
    fit2 = fit_beta_q_from_envelope(
        z_coords, i1_phasor, beta_q_guess, z_min=z_min, z_max=z_max
    )
    if z_min is None:
        z_min = float(z_coords[0])
    if z_max is None:
        z_max = float(z_coords[-1])
    mask = (z_coords >= z_min) & (z_coords <= z_max)
    zm = np.asarray(z_coords[mask], dtype=np.float64)
    ym = np.asarray(i1_phasor[mask], dtype=np.complex128)
    ym_energy = float(np.real(np.vdot(ym, ym)))

    def _varpro4(kc_val: float, bq_val: float) -> tuple[np.ndarray, float]:
        bq_pos = abs(bq_val)
        kf_val = kc_val - bq_pos
        ks_val = kc_val + bq_pos
        design = np.column_stack(
            [
                np.exp(-1j * kf_val * zm),
                np.exp(-1j * ks_val * zm),
                np.exp(1j * kf_val * zm),
                np.exp(1j * ks_val * zm),
            ]
        )
        coef, _, _, _ = np.linalg.lstsq(design, ym, rcond=None)
        res = ym - design @ coef
        rss_val = float(np.real(np.vdot(res, res)))
        return coef, rss_val

    opt = minimize(
        lambda x: _varpro4(float(x[0]), float(x[1]))[1],
        x0=np.array([fit2["k_c"], fit2["beta_q"]], dtype=np.float64),
        method="Nelder-Mead",
        options={
            "xatol": 1e-10 * beta_q_guess,
            "fatol": 1e-16 * max(ym_energy, 1e-30),
            "maxiter": 2000,
        },
    )
    kc4 = float(opt.x[0])
    bq4 = float(abs(opt.x[1]))
    coef4, rss4 = _varpro4(kc4, bq4)
    af_fwd, as_fwd, af_bwd, as_bwd = (
        complex(coef4[0]),
        complex(coef4[1]),
        complex(coef4[2]),
        complex(coef4[3]),
    )
    fwd_norm = float(np.hypot(abs(af_fwd), abs(as_fwd)))
    bwd_norm = float(np.hypot(abs(af_bwd), abs(as_bwd)))

    return {
        "beta_q_2": float(fit2["beta_q"]),
        "beta_q_4": bq4,
        "k_c_4": kc4,
        "delta_beta_q": float(bq4 - fit2["beta_q"]),
        "rel_delta_beta_q": float((bq4 - fit2["beta_q"]) / fit2["beta_q"]),
        "bwd_over_fwd": float(bwd_norm / max(fwd_norm, 1e-30)),
        "bwd_f_over_fwd_f": float(abs(af_bwd) / max(abs(af_fwd), 1e-30)),
        "bwd_s_over_fwd_s": float(abs(as_bwd) / max(abs(as_fwd), 1e-30)),
        "rss_4": rss4,
    }


def fit_lb_parasitic_waves(
    z_coords: np.ndarray,
    i1_phasor: np.ndarray,
    nx: int,
    nz: int,
    params: dict[str, float],
    omega_p_scale: float = 1.0,
    z_min: float | None = None,
    z_max: float | None = None,
) -> dict[str, Any]:
    """Fit ``I1(z)`` with the two physical LB modes plus two fixed parasitic odd-even modes.

    The centered-difference `LinearizedBeam` advection operator ``sin(k * dz) / dz``
    supports parasitic branches ``k_p in (pi / (2 * dz), pi / dz)`` at the same ``omega``
    with negative group velocity that solve the discrete `LinearizedBeam` dispersion
    relation::

        Omega_LB -+ omega_{q,1}(k_p) = u_dep * sin(k_p * dz) / dz

    with the fundamental transverse eigenvalue ``lambda_1(k_p)`` (and hence the
    discrete reduction factor) evaluated at ``k_p`` via `scipy.optimize.brentq`.
    Fixing ``(kp_minus, kp_plus)`` from these discrete roots adds two linear
    complex amplitudes ``(a_{p,-}, a_{p,+})`` and no new wavenumber parameters to
    the variable-projection solve for ``(k_c, beta_q)``.
    """
    beta_q_guess = float(params["beta_q"] * omega_p_scale)
    if z_min is None or z_max is None:
        z_min_def, z_max_def = fit_window(params)
        if z_min is None:
            z_min = z_min_def
        if z_max is None:
            z_max = z_max_def

    fit2 = fit_beta_q_from_envelope(
        z_coords, i1_phasor, beta_q_guess, z_min=z_min, z_max=z_max
    )
    dz = TUNNEL_LENGTH_M / nz
    dt, _ = _cached_integer_period_dt(int(nx), int(nz))
    omega = params["omega"]
    u_dep = params["u_dep"]
    gamma_dep = params["gamma_dep"]
    omega_p_dep = params["omega_p"] * float(omega_p_scale)
    wp_fac = omega_p_dep / (gamma_dep**1.5)
    omega_lhs = float((2.0 / dt) * np.tan(0.5 * omega * dt))

    def _omega_q1_lb(k_val: float) -> float:
        op_k = compute_transverse_modal_operator(
            "lb",
            nx,
            nz,
            params,
            k_val=k_val,
            use_omega_f=True,
        )
        return float(wp_fac * np.sqrt(op_k["evals"][0]))

    def _disp_func(k_val: float, branch_sign: float) -> float:
        return float(
            omega_lhs
            + branch_sign * _omega_q1_lb(k_val)
            - u_dep * np.sin(k_val * dz) / dz
        )

    def _solve_parasitic_root(branch_sign: float, label: str) -> tuple[float, float]:
        eps_k = 1.0e-10 / dz
        k_a = float(0.5 * np.pi / dz + eps_k)
        k_b = float(np.pi / dz - eps_k)
        fa = _disp_func(k_a, branch_sign)
        fb = _disp_func(k_b, branch_sign)
        if fa * fb > 0.0:
            k_scan = np.linspace(k_a, k_b, 65)
            f_scan = np.array([_disp_func(float(kv), branch_sign) for kv in k_scan])
            sign_changes = np.where(f_scan[:-1] * f_scan[1:] <= 0.0)[0]
            if len(sign_changes) == 0:
                raise ValueError(
                    f"No sign-change bracket for LB parasitic {label} root in open interval "
                    f"(pi/(2*dz), pi/dz) = ({0.5 * np.pi / dz:.6f}, {np.pi / dz:.6f}) rad/m: "
                    f"f(k_a)={fa:+.6e}, f(k_b)={fb:+.6e}."
                )
            idx0 = int(sign_changes[0])
            k_a = float(k_scan[idx0])
            k_b = float(k_scan[idx0 + 1])
        root_k = float(
            brentq(
                lambda kv: _disp_func(float(kv), branch_sign),
                k_a,
                k_b,
                xtol=1e-14,
                rtol=1e-14,
            )
        )
        rel_resid = float(abs(_disp_func(root_k, branch_sign)) / abs(omega_lhs))
        return root_k, rel_resid

    kp_minus, disp_resid_minus = _solve_parasitic_root(-1.0, "minus")
    kp_plus, disp_resid_plus = _solve_parasitic_root(+1.0, "plus")

    mask = (z_coords >= z_min) & (z_coords <= z_max)
    zm = np.asarray(z_coords[mask], dtype=np.float64)
    ym = np.asarray(i1_phasor[mask], dtype=np.complex128)
    ym_energy = float(np.real(np.vdot(ym, ym)))
    u_pm = np.exp(-1j * kp_minus * zm)
    u_pp = np.exp(-1j * kp_plus * zm)

    def _varpro_para(kc_val: float, bq_val: float) -> tuple[np.ndarray, float]:
        bq_pos = abs(bq_val)
        kf_val = kc_val - bq_pos
        ks_val = kc_val + bq_pos
        design = np.column_stack(
            [np.exp(-1j * kf_val * zm), np.exp(-1j * ks_val * zm), u_pm, u_pp]
        )
        coef, _, _, _ = np.linalg.lstsq(design, ym, rcond=None)
        res = ym - design @ coef
        rss_val = float(np.real(np.vdot(res, res)))
        return coef, rss_val

    opt = minimize(
        lambda x: _varpro_para(float(x[0]), float(x[1]))[1],
        x0=np.array([fit2["k_c"], fit2["beta_q"]], dtype=np.float64),
        method="Nelder-Mead",
        options={
            "xatol": 1e-10 * beta_q_guess,
            "fatol": 1e-16 * max(ym_energy, 1e-30),
            "maxiter": 2000,
        },
    )
    kcp = float(opt.x[0])
    bqp = float(abs(opt.x[1]))
    coef_p, rssp = _varpro_para(kcp, bqp)
    af_fwd, as_fwd, ap_m, ap_p = (
        complex(coef_p[0]),
        complex(coef_p[1]),
        complex(coef_p[2]),
        complex(coef_p[3]),
    )
    fwd_norm = float(np.hypot(abs(af_fwd), abs(as_fwd)))
    para_norm = float(np.hypot(abs(ap_m), abs(ap_p)))
    dbq = float(bqp - fit2["beta_q"])

    return {
        "beta_q_2": float(fit2["beta_q"]),
        "beta_q_para": bqp,
        "k_c_para": kcp,
        "kp_minus": kp_minus,
        "kp_plus": kp_plus,
        "disp_resid_minus": disp_resid_minus,
        "disp_resid_plus": disp_resid_plus,
        "delta_beta_q": dbq,
        "rel_delta_beta_q": float(dbq / fit2["beta_q"]),
        "ap_over_afwd": float(para_norm / max(fwd_norm, 1e-30)),
        "rss_para": rssp,
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


@lru_cache(maxsize=32)
def _cached_integer_period_dt(
    nx: int, nz: int, cfln: float = CFL_NUMBER
) -> tuple[float, int]:
    """Cached integer-period ``(dt, n_spp)`` for grid shape ``(nx, 1, nz)``."""
    a = TUNNEL_HALF_WIDTH_M
    ly = PERIODIC_LY_M
    lz = TUNNEL_LENGTH_M
    grid = GridFIT3D(
        -a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, int(nx), 1, int(nz), verbose=0
    )
    return _integer_period_dt(grid, cfln=cfln)


def run_linearized_beam_drift(
    nx: int,
    nz: int,
    rho0_scale: float = 1.0,
    m_mod: float = MODULATION_INDEX,
    rho0_profile: Any = None,
    outflow_ramp_m: float = OUTFLOW_RAMP_M,
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
    outflow_ramp_m : float, optional
        Axial length [m] of the outflow sine-squared taper. Default is ``OUTFLOW_RAMP_M``.

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
    n_ramp = round(outflow_ramp_m / dz)
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

    # M2-type runs scale rho0, hence omega_p by sqrt(rho0_scale); the settle bound must use it
    n_settle = settle_steps(
        "lb",
        nx,
        nz,
        params,
        dt,
        omega_p_scale=float(np.sqrt(rho0_scale)) if rho0_profile is None else 1.0,
    )
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


def _particle_current_lines(
    beam_ref: LinearizedBeam,
    solver: SolverFIT3D,
    mode: str = "all",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(flat_idx_2d, dA_perp)`` of shapes ``(M, L)`` for particle current integration.

    Uses the same axial slice ``beam_ref.k_indices``, flat-index formula
    ``flat_3d``, and dual-area extraction ``tA_all`` as
    ``LinearizedBeam._initialize``.

    Parameters
    ----------
    beam_ref : LinearizedBeam
        Reference `LinearizedBeam` on the same grid providing ``axis``,
        ``k_indices``, ``flat_idx_2d``, and ``dA_perp``.
    solver : SolverFIT3D
        Electromagnetic FIT solver providing ``grid``, ``ieps``, and grid dimensions.
    mode : {"all", "lb_keep"}, optional
        Transverse line selection mode:
        - ``"lb_keep"``: return ``(beam_ref.flat_idx_2d, beam_ref.dA_perp)``
          (restricted to `LinearizedBeam` top-hat ``keep_lines``).
        - ``"all"``: include every transverse ``E_s`` line that has at least one
          edge with ``ieps != 0`` across ``beam_ref.k_indices`` (without the
          top-hat ``w_perp > 0`` filter). Default is ``"all"``.

    Returns
    -------
    flat_idx_2d : np.ndarray
        2D integer array of flat ``E_s``/``J_s`` indices of shape ``(M, L)``.
    dA_perp : np.ndarray
        2D float array of dual facet areas [m^2] of shape ``(M, L)``.
    """
    if mode == "lb_keep":
        return beam_ref.flat_idx_2d, beam_ref.dA_perp
    if mode != "all":
        raise ValueError(
            f"Unknown meas_lines mode={mode!r}; expected 'all' or 'lb_keep'."
        )

    grid = solver.grid
    c_ax = "xyz".index(beam_ref.axis)
    dims = (solver.Nx, solver.Ny, solver.Nz)
    k_indices = np.asarray(beam_ref.k_indices, dtype=int)
    M = k_indices.size
    sl: list[Any] = [slice(None), slice(None), slice(None)]
    sl[c_ax] = k_indices

    tA_all = np.where(grid.itA.toarray() != 0, 1.0 / grid.itA.toarray(), 0.0)
    ieps_host = np.asarray(
        solver.ieps.array.get() if solver.use_gpu else solver.ieps.array
    )

    I, J, K = np.meshgrid(
        np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing="ij"
    )
    N = solver.N
    flat_3d = c_ax * N + I + dims[0] * J + dims[0] * dims[1] * K
    flat_sub = np.moveaxis(flat_3d[tuple(sl)], c_ax, 0).reshape(M, -1)
    ieps_sub = ieps_host[flat_sub]

    keep_lines = np.any(ieps_sub != 0, axis=0)
    if not np.any(keep_lines):
        raise ValueError("Beam cross-section contains no non-PEC E edges.")

    flat_idx_2d = flat_sub[:, keep_lines]
    dA_perp = np.moveaxis(
        tA_all[c_ax * N : (c_ax + 1) * N].reshape(dims, order="F")[tuple(sl)],
        c_ax,
        0,
    ).reshape(M, -1)[:, keep_lines]
    return flat_idx_2d, dA_perp


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
    gather: str = "galerkin",
    shape: int = PARTICLE_SHAPE,
    meas_lines: str = "all",
    wall_stencil: str = "masked",
) -> dict[str, Any]:
    """Run `ParticleSolver` + `BeamInjector` on the identical drift-tunnel grid.

    By default (``meas_lines="all"``), integrates the Esirkepov-deposited
    particle current density ``ps.J_particles`` over all non-PEC transverse
    primal ``E_z`` lines across ``beam_ref.k_indices`` using the same flat-index
    and dual-area (``dA_perp``) conventions as `LinearizedBeam.axial_current`,
    with identical integer-period ``dt`` and time-mean subtraction before the
    DFT. Summing over all non-PEC transverse lines measures the total beam
    current ``I(z)`` because higher-order B-spline transverse deposits
    (``shape >= 2``) place part of the particle current on lines outside the
    `LinearizedBeam` top-hat ``keep_lines`` mask that
    :func:`discrete_modal_prediction` (``"particles"`` branch) includes.
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
        Reference `LinearizedBeam` on the same grid providing ``axis``,
        ``k_indices``, ``M``, ``s_coords``, and (for ``meas_lines="lb_keep"``)
        ``flat_idx_2d`` and ``dA_perp``.
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
    gather : str, optional
        Field gather mode passed to `ParticleSolver`. Default is ``"galerkin"``.
    shape : int, optional
        B-spline particle shape order ``p`` in ``{1, 2, 3}`` passed to
        `ParticleSolver`. Default is ``PARTICLE_SHAPE``.
    meas_lines : {"all", "lb_keep"}, optional
        Transverse line set passed to :func:`_particle_current_lines` for the
        ``iz_hist`` cross-section current sum (``"all"`` for all non-PEC
        transverse lines or ``"lb_keep"`` for ``beam_ref.keep_lines`` only).
        Default is ``"all"``.
    wall_stencil : {"masked", "image"}, optional
        ``ParticleSolver`` treatment of the shape stencil at the PEC walls (default
        ``"masked"``, the released behaviour; ``"image"`` folds it back with the PEC image
        symmetry, see :mod:`wakis.particles.wall_images`).

    Returns
    -------
    dict of str to Any
        Dictionary with keys ``i1_part``, ``i2_part``, ``dc_leakage_part``,
        ``dirichlet_leak_part``, ``u_z_mean``, ``rho_dc_mean``, ``u_z_profile``,
        ``rho_dc_profile``, ``dt``, and ``n_spp``.
    """
    if meas_lines not in ("all", "lb_keep"):
        raise ValueError(
            f"Unknown meas_lines={meas_lines!r}; expected 'all' or 'lb_keep'."
        )
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

    n_macro = (
        int(n_macro_per_step)
        if n_macro_per_step is not None
        else (16 if nx == COARSE_GRID[0] else 32)
    )
    n_settle = settle_steps("particles", nx, nz, params, dt, shape=shape)
    n_dft = 6 * n_spp
    n_total = n_settle + n_dft

    est_cap = max(int((n_settle + 16) * n_macro), 1024)
    sp = Species(
        name="electron",
        charge=-e_charge,
        mass=m_e / qm_scale,
        capacity=est_cap,
    )

    def current_modulation(t: float) -> float:
        ramp_t = 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))
        return i0_dc * (1.0 + m_mod * ramp_t * np.sin(omega * t))

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
        shape=int(shape),
        filter=PARTICLE_FILTER,
        gather=gather,
        B_ext=(0.0, 0.0, GUIDE_BZ_TESLA),
        wall_stencil=wall_stencil,
    ).attach()

    meas_flat_idx_2d, meas_dA_perp = _particle_current_lines(
        beam_ref, solver, meas_lines
    )
    meas_flat_idx = solver.E.xp.asarray(meas_flat_idx_2d.ravel())
    iz_hist = np.zeros((n_dft, beam_ref.M), dtype=np.float64)
    t_mids = np.zeros(n_dft, dtype=np.float64)

    dx = 2.0 * a / nx
    dz = lz / nz
    z_edges = np.linspace(0.0, lz, nz + 1)
    x_node_frac = np.arange(nx + 1, dtype=np.float64)[:, None] / float(nx)
    w_sum_dc = np.zeros(nz, dtype=np.float64)
    wv_sum_dc = np.zeros(nz, dtype=np.float64)
    n_dc_samples = 0

    w_win = np.zeros((n_dft, nz), dtype=np.float64)
    wv_win = np.zeros((n_dft, nz), dtype=np.float64)
    wv2_win = np.zeros((n_dft, nz), dtype=np.float64)
    wg_win = np.zeros((n_dft, nz), dtype=np.float64)
    wgp_win = np.zeros((n_dft, nz), dtype=np.float64)

    use_gpu_bin = bool(
        USE_GPU and sp.on_gpu and _GPU_BIN_KERNEL is not None and 5 * nz * 8 <= 49152
    )
    if use_gpu_bin:
        out_bins_d = cp.zeros((5, nz), dtype=np.float64)
        phi_nodes_d = cp.zeros((nx + 1, nz), dtype=np.float64)
        c2_val = float(c_light**2)
        q_over_mc2_val = float(e_charge / (sp.mass * c_light**2))
        smem_bytes = int(5 * nz * 8)

    for step in range(n_total):
        solver.one_step()
        if step >= n_settle:
            k = step - n_settle
            t_mids[k] = (step + 0.5) * dt
            j_raw = ps.J_particles[meas_flat_idx]
            j_host = np.asarray(j_raw.get() if hasattr(j_raw, "get") else j_raw)
            j_edges = j_host.reshape(meas_flat_idx_2d.shape)
            iz_hist[k, :] = np.sum(j_edges * meas_dA_perp, axis=1)

            # Deposited-field transverse potential phi(x, z) with PEC phi(+-a) = 0
            ex_raw = solver.E[:, :, :, "x"]
            ex_host = np.asarray(ex_raw.get() if hasattr(ex_raw, "get") else ex_raw)
            ex_cell = 0.5 * (ex_host[:nx, 0, :nz] + ex_host[:nx, 0, 1 : nz + 1])
            phi_nodes = np.zeros((nx + 1, nz), dtype=np.float64)
            phi_nodes[1:, :] = -dx * np.cumsum(ex_cell, axis=0)
            phi_nodes -= x_node_frac * phi_nodes[-1:, :]

            if use_gpu_bin:
                n_p_cur = int(sp.n_particles)
                out_bins_d.fill(0.0)
                if n_p_cur > 0:
                    phi_nodes_d.set(phi_nodes)
                    w_mat_d = transverse_weight_matrix(sp._x[:n_p_cur], nx, shape)
                    n_blocks = min(max((n_p_cur + 255) // 256, 1), 256)
                    _GPU_BIN_KERNEL(
                        (n_blocks,),
                        (256,),
                        (
                            n_p_cur,
                            int(nx),
                            int(nz),
                            float(dz),
                            c2_val,
                            float(gamma0),
                            q_over_mc2_val,
                            w_mat_d,
                            sp._z,
                            sp._ux,
                            sp._uy,
                            sp._uz,
                            sp._w,
                            phi_nodes_d,
                            out_bins_d,
                        ),
                        shared_mem=smem_bytes,
                    )
                bins_host = out_bins_d.get()
                w_k = bins_host[0]
                wv_k = bins_host[1]
                w_win[k, :] = w_k
                wv_win[k, :] = wv_k
                wv2_win[k, :] = bins_host[2]
                wg_win[k, :] = bins_host[3]
                wgp_win[k, :] = bins_host[4]
            else:
                xp = np.asarray(sp.x.get() if hasattr(sp.x, "get") else sp.x)
                zp = np.asarray(sp.z.get() if hasattr(sp.z, "get") else sp.z)
                ux_p = np.asarray(sp.ux.get() if hasattr(sp.ux, "get") else sp.ux)
                uy_p = np.asarray(sp.uy.get() if hasattr(sp.uy, "get") else sp.uy)
                uz_p = np.asarray(sp.uz.get() if hasattr(sp.uz, "get") else sp.uz)
                wp = np.asarray(sp.w.get() if hasattr(sp.w, "get") else sp.w)

                gamma_p = np.sqrt(1.0 + (ux_p**2 + uy_p**2 + uz_p**2) / (c_light**2))
                vz_p = uz_p / gamma_p
                bin_idx = np.clip(np.digitize(zp, z_edges) - 1, 0, nz - 1)

                w_mat_p = transverse_weight_matrix(xp, nx, shape)
                phi_p = np.sum(w_mat_p * phi_nodes[:, bin_idx].T, axis=1)
                gamma_pred_p = gamma0 + (e_charge / (sp.mass * c_light**2)) * phi_p

                w_k = np.bincount(bin_idx, weights=wp, minlength=nz)
                wv_k = np.bincount(bin_idx, weights=wp * vz_p, minlength=nz)
                w_win[k, :] = w_k
                wv_win[k, :] = wv_k
                wv2_win[k, :] = np.bincount(
                    bin_idx, weights=wp * (vz_p**2), minlength=nz
                )
                wg_win[k, :] = np.bincount(bin_idx, weights=wp * gamma_p, minlength=nz)
                wgp_win[k, :] = np.bincount(
                    bin_idx, weights=wp * gamma_pred_p, minlength=nz
                )

            # Accumulate DC velocity and density profiles over 1 RF period (n_spp steps)
            # at the start of the steady-state window to cancel AC modulation
            if k < n_spp:
                w_sum_dc += w_k
                wv_sum_dc += wv_k
                n_dc_samples += 1

    sp_charge = float(sp.charge)
    del ps, inj, sp, solver, grid
    if USE_GPU and cp is not None:
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    iz_dc = np.mean(iz_hist, axis=0)
    iz_ac = iz_hist - iz_dc[None, :]
    ker1 = np.exp(-1j * omega * t_mids) * (2.0 / n_dft)
    ker2 = np.exp(-2j * omega * t_mids) * (2.0 / n_dft)
    i1_part = np.dot(ker1, iz_ac)
    i2_part = np.dot(ker2, iz_ac)

    # Residual DC leakage: dc_leak is a zero-mean algebraic identity; dirichlet_leak
    # tests the integer-n_spp orthogonality |sum(ker1)| * (I0 / I1_0).
    dc_test = np.broadcast_to(iz_dc[None, :], iz_hist.shape)
    i1_scale = max(abs(m_mod), MODULATION_INDEX) * i0_dc
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
    rho_dc_profile = (sp_charge * w_mean_cell) / (area_beam * dz)

    z_coords = beam_ref.s_coords
    interior = (z_coords >= 0.2 * lz) & (z_coords <= 0.8 * lz)
    u_z_mean = float(np.sum(wv_mean_cell[interior]) / np.sum(w_mean_cell[interior]))
    rho_dc_mean = float(np.mean(rho_dc_profile[interior]))

    # Step 5 (A3): Per-z-cell weighted mean and rms of v_z, and mean gamma, over the
    # DFT window after removing the coherent omega harmonic (a + b*cos(omega*t) + c*sin(omega*t))
    basis_t = np.column_stack(
        [np.ones(n_dft), np.cos(omega * t_mids), np.sin(omega * t_mids)]
    )
    w_tot_win = np.sum(w_win, axis=0)
    vz_coh_profile = np.full(nz, u_dep, dtype=np.float64)
    sigma_vz_profile = np.zeros(nz, dtype=np.float64)
    gamma_profile = np.full(nz, gamma0, dtype=np.float64)
    gamma_pred_profile = np.full(nz, gamma0, dtype=np.float64)

    for j in range(nz):
        wj = w_win[:, j]
        w_sum_j = float(w_tot_win[j])
        if w_sum_j <= 0.0:
            continue
        wvj = wv_win[:, j]
        wv2j = wv2_win[:, j]
        wgj = wg_win[:, j]
        wgpj = wgp_win[:, j]
        w_safe = np.maximum(wj, 1e-30)
        v_bar_j = wvj / w_safe

        xtw = basis_t.T * wj[None, :]
        gram_j = xtw @ basis_t
        theta_v = np.linalg.lstsq(gram_j, basis_t.T @ wvj, rcond=None)[0]
        theta_g = np.linalg.lstsq(gram_j, basis_t.T @ wgj, rcond=None)[0]
        v_fit_j = basis_t @ theta_v
        g_fit_j = basis_t @ theta_g

        within_ss = np.sum(np.maximum(wv2j - (wvj**2) / w_safe, 0.0))
        temp_ss = np.sum(wj * ((v_bar_j - v_fit_j) ** 2))
        vz_coh_profile[j] = float(np.sum(wj * v_fit_j) / w_sum_j)
        sigma_vz_profile[j] = float(np.sqrt(max((within_ss + temp_ss) / w_sum_j, 0.0)))
        gamma_profile[j] = float(np.sum(wj * g_fit_j) / w_sum_j)
        gamma_pred_profile[j] = float(np.sum(wgpj) / w_sum_j)

    w_int = w_tot_win[interior]
    w_int_sum = float(np.sum(w_int))
    vz_mean_coh = float(np.sum(w_int * vz_coh_profile[interior]) / w_int_sum)
    sigma_vz_mean = float(
        np.sqrt(np.sum(w_int * (sigma_vz_profile[interior] ** 2)) / w_int_sum)
    )
    sigma_vz_over_u = float(sigma_vz_mean / vz_mean_coh)
    gamma_mean = float(np.sum(w_int * gamma_profile[interior]) / w_int_sum)
    gamma_pred_mean = float(np.sum(w_int * gamma_pred_profile[interior]) / w_int_sum)

    return {
        "i1_part": i1_part,
        "i2_part": i2_part,
        "dc_leakage_part": float(max(dc_leak, dirichlet_leak)),
        "dirichlet_leak_part": dirichlet_leak,
        "u_z_mean": u_z_mean,
        "rho_dc_mean": rho_dc_mean,
        "u_z_profile": u_z_profile,
        "rho_dc_profile": rho_dc_profile,
        "vz_mean_coh": vz_mean_coh,
        "sigma_vz_mean": sigma_vz_mean,
        "sigma_vz_over_u": sigma_vz_over_u,
        "gamma_mean": gamma_mean,
        "gamma_pred_mean": gamma_pred_mean,
        "gamma_diff_mean": float(gamma_mean - gamma_pred_mean),
        "vz_coh_profile": vz_coh_profile,
        "sigma_vz_profile": sigma_vz_profile,
        "gamma_profile": gamma_profile,
        "gamma_pred_profile": gamma_pred_profile,
        "dt": dt,
        "n_spp": n_spp,
        "n_settle": n_settle,
        "iz_hist_dec": iz_hist[
            :, ::8
        ].copy(),  # I_z(t, z) on every 8th axial edge, DFT window [A]
    }


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCW_FINGERPRINT_FILES: list[Path] = sorted(
    [
        *_REPO_ROOT.glob("wakis/particles/*.py"),
        *(
            p
            for p in (
                _REPO_ROOT / "wakis" / "solverFIT3D.py",
                _REPO_ROOT / "wakis" / "gridFIT3D.py",
                Path(__file__).resolve(),
            )
            if p.is_file()
        ),
    ],
    key=lambda p: str(p),
)
_BEAM_REBUILD_TAG = "__scw_linearized_beam_v1__"
_BEAM_STATE_ATTRS = (
    "M",
    "N_lines",
    "k_indices",
    "s_coords",
    "ds_line",
    "keep_lines",
    "flat_idx_2d",
    "flat_idx_host",
    "dA_perp",
    "dV",
    "n_ramp",
    "w_axial",
    "area",
    "area_eff",
    "rho0_peak",
    "_w_perp_eff",
    "rho0",
    "n0",
    "v1",
    "rho1",
    "v1_prev",
    "rho1_prev",
    "J1",
    "_step_count",
)


def scw_source_fingerprint() -> str:
    """Return SHA-256 hex digest over the bytes of all fingerprinted SCW source files."""
    h = hashlib.sha256()
    for path in sorted(_SCW_FINGERPRINT_FILES, key=lambda p: str(p)):
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def _normalise_scw_func_and_args(
    func_name: str, args: tuple[Any, ...] | list[Any] | dict[str, Any]
) -> tuple[str, tuple[int | float | str, ...]]:
    """Normalise function name and argument sequence/dict to canonical `(func, args_tuple)`."""
    if func_name in ("cached_linearized_beam_drift", "lb"):
        canon_func = "cached_linearized_beam_drift"
        if isinstance(args, dict):
            nx = int(args["nx"])
            nz = int(args["nz"])
            rho0_scale = float(args.get("rho0_scale", 1.0))
            m_mod = float(args.get("m_mod", MODULATION_INDEX))
        else:
            seq = list(args)
            nx = int(seq[0])
            nz = int(seq[1])
            rho0_scale = float(seq[2]) if len(seq) > 2 else 1.0
            m_mod = float(seq[3]) if len(seq) > 3 else float(MODULATION_INDEX)
        return canon_func, (nx, nz, rho0_scale, m_mod)

    if func_name in (
        "_cached_particle_drift_impl",
        "cached_particle_drift",
        "particle",
    ):
        canon_func = "_cached_particle_drift_impl"
        if isinstance(args, dict):
            nx = int(args["nx"])
            nz = int(args["nz"])
            profile = str(args.get("profile", "sheet_lattice"))
            raw_seed = int(args.get("seed", NOISE_SEEDS[0]))
            eff_seed = int(NOISE_SEEDS[0]) if profile == "sheet_lattice" else raw_seed
            m_mod = float(args.get("m_mod", MODULATION_INDEX))
            qm_scale = float(args.get("qm_scale", 1.0))
            n_macro_raw = args.get("n_macro", args.get("n_macro_per_step"))
            n_macro = (
                int(n_macro_raw)
                if n_macro_raw is not None
                else (16 if nx == COARSE_GRID[0] else 32)
            )
            gather = str(args.get("gather", "galerkin"))
            shape = int(args.get("shape", PARTICLE_SHAPE))
            wall_stencil = str(args.get("wall_stencil", "masked"))
        else:
            seq = list(args)
            nx = int(seq[0])
            nz = int(seq[1])
            profile = str(seq[2]) if len(seq) > 2 else "sheet_lattice"
            raw_seed = int(seq[3]) if len(seq) > 3 else int(NOISE_SEEDS[0])
            eff_seed = int(NOISE_SEEDS[0]) if profile == "sheet_lattice" else raw_seed
            m_mod = float(seq[4]) if len(seq) > 4 else float(MODULATION_INDEX)
            qm_scale = float(seq[5]) if len(seq) > 5 else 1.0
            n_macro_raw = seq[6] if len(seq) > 6 else None
            n_macro = (
                int(n_macro_raw)
                if n_macro_raw is not None
                else (16 if nx == COARSE_GRID[0] else 32)
            )
            gather = str(seq[7]) if len(seq) > 7 else "galerkin"
            shape = int(seq[8]) if len(seq) > 8 else int(PARTICLE_SHAPE)
            wall_stencil = str(seq[9]) if len(seq) > 9 else "masked"
        key_tuple = (
            nx,
            nz,
            profile,
            eff_seed,
            m_mod,
            qm_scale,
            n_macro,
            gather,
            shape,
        )
        # The default wall stencil keeps the historical 9-tuple key.
        if wall_stencil != "masked":
            key_tuple = (*key_tuple, wall_stencil)
        return canon_func, key_tuple

    canon_func = str(func_name)
    raw_items = list(args.values()) if isinstance(args, dict) else list(args)
    norm_items: list[int | float | str] = []
    for item in raw_items:
        if isinstance(item, (bool, int, np.integer)):
            norm_items.append(int(item))
        elif isinstance(item, (float, np.floating)):
            norm_items.append(float(item))
        else:
            norm_items.append(str(item))
    return canon_func, tuple(norm_items)


def scw_cache_key(
    func_name: str, args: tuple[Any, ...] | list[Any] | dict[str, Any]
) -> str:
    """Compute SHA-256 cache key from function name, normalised argument tuple, and source fingerprint."""
    canon_func, norm_args = _normalise_scw_func_and_args(func_name, args)
    arg_repr = "(" + ", ".join(repr(v) for v in norm_args) + ")"
    fp = scw_source_fingerprint()
    h = hashlib.sha256()
    h.update(f"{canon_func}:{arg_repr}:{fp}".encode())
    return h.hexdigest()


def _build_reference_linearized_beam(
    nx: int,
    nz: int,
    rho0_scale: float = 1.0,
    m_mod: float = MODULATION_INDEX,
    outflow_ramp_m: float = OUTFLOW_RAMP_M,
) -> LinearizedBeam:
    """Rebuild a deterministic `LinearizedBeam` on grid ``(nx, 1, nz)``."""
    params = compute_analytic_parameters()
    u_dep = params["u_dep"]
    gamma_dep = params["gamma_dep"]
    omega = params["omega"]
    rho_dep = params["rho_dep"]
    rho0_val = float(rho0_scale) * rho_dep
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    ly = PERIODIC_LY_M
    lz = TUNNEL_LENGTH_M
    t_rise = RISE_PERIODS / DRIVE_FREQ_HZ

    dz = lz / int(nz)
    n_ramp = round(float(outflow_ramp_m) / dz)
    grid = GridFIT3D(
        -a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, int(nx), 1, int(nz), verbose=0
    )
    dt, _ = _integer_period_dt(grid)
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

    return LinearizedBeam(
        u0=u_dep,
        rho0=rho0_val,
        gamma0=gamma_dep,
        axis="z",
        half_thickness=b,
        ramp_cells=n_ramp,
        inflow_rho1=inflow_rho1,
        solver=solver,
    )


def _to_host_copy(val: Any) -> Any:
    if hasattr(val, "get"):
        return np.asarray(val.get()).copy()
    if isinstance(val, np.ndarray):
        return val.copy()
    return val


def _is_picklable(val: Any) -> bool:
    try:
        pickle.dumps(val, protocol=5)
        return True
    except Exception:  # noqa: BLE001
        return False


def _serialize_scw_result(
    res: dict[str, Any], norm_args: tuple[Any, ...]
) -> dict[str, Any]:
    """Prepare simulation result dictionary for protocol-5 pickling."""
    payload: dict[str, Any] = {}
    for k, v in res.items():
        if k == "beam":
            if isinstance(v, LinearizedBeam):
                solver_ref = getattr(v, "_solver", None)
                grid_nx = (
                    int(solver_ref.Nx) if solver_ref is not None else int(norm_args[0])
                )
                grid_nz = (
                    int(solver_ref.Nz) if solver_ref is not None else int(norm_args[1])
                )
                rho0_scale = float(norm_args[2]) if len(norm_args) > 2 else 1.0
                m_mod = float(norm_args[3]) if len(norm_args) > 3 else MODULATION_INDEX
                state = {
                    attr: _to_host_copy(getattr(v, attr))
                    for attr in _BEAM_STATE_ATTRS
                    if hasattr(v, attr)
                }
                payload[k] = {
                    "__tag__": _BEAM_REBUILD_TAG,
                    "grid_nx": grid_nx,
                    "grid_nz": grid_nz,
                    "rho0_scale": rho0_scale,
                    "m_mod": m_mod,
                    "state": state,
                }
            elif _is_picklable(v):
                payload[k] = v
            else:
                state = {
                    attr: _to_host_copy(attr_val)
                    for attr, attr_val in getattr(v, "__dict__", {}).items()
                    if _is_picklable(_to_host_copy(attr_val))
                }
                payload[k] = {
                    "__tag__": _BEAM_REBUILD_TAG,
                    "grid_nx": int(norm_args[0]),
                    "grid_nz": int(norm_args[1]),
                    "rho0_scale": float(norm_args[2]) if len(norm_args) > 2 else 1.0,
                    "m_mod": (
                        float(norm_args[3]) if len(norm_args) > 3 else MODULATION_INDEX
                    ),
                    "state": state,
                }
        else:
            payload[k] = _to_host_copy(v)
    return payload


def _deserialize_scw_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore simulation result dictionary from pickled payload, rebuilding `beam` if present."""
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if (
            k == "beam"
            and isinstance(v, dict)
            and v.get("__tag__") == _BEAM_REBUILD_TAG
        ):
            beam = _build_reference_linearized_beam(
                int(v["grid_nx"]),
                int(v["grid_nz"]),
                rho0_scale=float(v["rho0_scale"]),
                m_mod=float(v["m_mod"]),
            )
            for attr, attr_val in v["state"].items():
                setattr(beam, attr, _to_host_copy(attr_val))
            if hasattr(beam, "flat_idx_host"):
                beam.flat_idx = beam.xp.asarray(beam.flat_idx_host)
                beam.J_old = beam.xp.zeros(beam.flat_idx_host.size, dtype=beam.dtype)
            out[k] = beam
        else:
            out[k] = _to_host_copy(v)
    return out


def _scw_disk_cached_call(
    canon_func: str,
    norm_args: tuple[int | float | str, ...],
    compute_fn: Any,
) -> dict[str, Any]:
    """Execute `compute_fn()` with optional disk cache (`CEM_SCW_CACHE_DIR`) and strict mode (`CEM_SCW_CACHE_STRICT=1`)."""
    cache_dir = os.environ.get("CEM_SCW_CACHE_DIR", "")
    strict = os.environ.get("CEM_SCW_CACHE_STRICT", "") == "1"
    if not cache_dir and not strict:
        return compute_fn()

    key = scw_cache_key(canon_func, norm_args)
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        target_path = cache_dir_path / f"{canon_func}-{key}.pkl"
        if target_path.is_file():
            with open(target_path, "rb") as f:
                payload = pickle.load(f)
            return _deserialize_scw_result(payload)

    if strict:
        raise RuntimeError(
            f"SCW disk cache miss for {canon_func}{norm_args!r} (key={key})"
        )

    res = compute_fn()
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        target_path = cache_dir_path / f"{canon_func}-{key}.pkl"
        payload = _serialize_scw_result(res, norm_args)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{canon_func}-{key}.",
            suffix=".tmp",
            dir=str(cache_dir_path),
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(payload, f, protocol=5)
            os.replace(tmp_path, target_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    return res


@lru_cache(maxsize=16)
def cached_linearized_beam_drift(
    nx: int,
    nz: int,
    rho0_scale: float = 1.0,
    m_mod: float = MODULATION_INDEX,
) -> dict[str, Any]:
    """Cache `LinearizedBeam` runs by ``(nx, nz, rho0_scale, m_mod)``."""
    norm_args = (int(nx), int(nz), float(rho0_scale), float(m_mod))
    return _scw_disk_cached_call(
        "cached_linearized_beam_drift",
        norm_args,
        lambda: run_linearized_beam_drift(nx, nz, rho0_scale=rho0_scale, m_mod=m_mod),
    )


@lru_cache(maxsize=64)
def _cached_particle_drift_impl(
    nx: int,
    nz: int,
    profile: str,
    seed: int,
    m_mod: float,
    qm_scale: float,
    n_macro: int,
    gather: str,
    shape: int,
    wall_stencil: str = "masked",
) -> dict[str, Any]:
    norm_args = (
        int(nx),
        int(nz),
        str(profile),
        int(seed),
        float(m_mod),
        float(qm_scale),
        int(n_macro),
        str(gather),
        int(shape),
    )
    if str(wall_stencil) != "masked":
        norm_args = (*norm_args, str(wall_stencil))

    def _run() -> dict[str, Any]:
        lb_ref = cached_linearized_beam_drift(
            nx, nz, rho0_scale=1.0, m_mod=MODULATION_INDEX
        )
        extra = {} if str(wall_stencil) == "masked" else {"wall_stencil": str(wall_stencil)}
        return run_particle_beam_drift(
            nx,
            nz,
            beam_ref=lb_ref["beam"],
            qm_scale=qm_scale,
            seed=seed,
            m_mod=m_mod,
            profile=profile,
            n_macro_per_step=n_macro,
            gather=gather,
            shape=shape,
            **extra,
        )

    return _scw_disk_cached_call("_cached_particle_drift_impl", norm_args, _run)


def cached_particle_drift(
    nx: int,
    nz: int,
    profile: str = "sheet_lattice",
    seed: int = 42,
    m_mod: float = MODULATION_INDEX,
    qm_scale: float = 1.0,
    n_macro_per_step: int | None = None,
    gather: str = "galerkin",
    shape: int = PARTICLE_SHAPE,
    wall_stencil: str = "masked",
) -> dict[str, Any]:
    """Cache `ParticleSolver` runs by ``(nx, nz, profile, seed, m_mod, qm_scale, n_macro_per_step, gather, shape)``.

    ``wall_stencil="image"`` appends the stencil to the key; the default ``"masked"`` keeps the
    historical 9-element key.
    """
    n_macro = (
        int(n_macro_per_step)
        if n_macro_per_step is not None
        else (16 if int(nx) == COARSE_GRID[0] else 32)
    )
    eff_seed = int(NOISE_SEEDS[0]) if profile == "sheet_lattice" else int(seed)
    return _cached_particle_drift_impl(
        int(nx),
        int(nz),
        str(profile),
        eff_seed,
        float(m_mod),
        float(qm_scale),
        n_macro,
        str(gather),
        int(shape),
        str(wall_stencil),
    )


@lru_cache(maxsize=8)
def cached_baseline_pair(
    nx: int,
    nz: int,
    m_mod: float = MODULATION_INDEX,
    shape: int = PARTICLE_SHAPE,
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
        shape=int(shape),
    )
    return lb_res, part_res


@lru_cache(maxsize=8)
def compute_m_extrapolation(
    nx: int,
    nz: int,
    include_m4: bool = True,
    shape: int = PARTICLE_SHAPE,
) -> dict[str, Any]:
    """Compute Step 3 Richardson extrapolation of deterministic lattice ``beta_q(m)`` as ``m -> 0``.

    On the coarse grid (``include_m4=True``), fits ``beta(m) = beta0 + c * m**2 + d * m**4``
    exactly through the three points ``(m, m/2, m/4)`` with linear Lagrange/Richardson
    weights ``(1/45, -20/45, 64/45)`` to obtain ``beta0_3pt``, two-point remainder
    ``e_R = |beta0_3pt - beta0_2pt(m/2, m/4)|`` where
    ``beta0_2pt(m/2, m/4) = (4 * beta(m/4) - beta(m/2)) / 3``, and propagated
    Gauss-Newton standard error ``sigma_prop`` across the three single-amplitude fits.
    """
    params = compute_analytic_parameters()
    tols = derived_tolerances(nx, nz, params, shape=shape)
    bq_pred = float(tols["bq_pred_part"])
    slack_part = float(tols["slack_part"])
    z_min, z_max = fit_window(params)
    beta_q_th = params["beta_q"]

    lb_m, part_m = cached_baseline_pair(
        nx, nz, m_mod=MODULATION_INDEX, shape=int(shape)
    )
    z_coords = lb_m["z_coords"]
    part_m2 = cached_particle_drift(
        nx,
        nz,
        profile="sheet_lattice",
        seed=NOISE_SEEDS[0],
        m_mod=0.5 * MODULATION_INDEX,
        qm_scale=1.0,
        shape=int(shape),
    )
    fit_m = fit_beta_q_from_envelope(
        z_coords, part_m["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
    )
    fit_m2 = fit_beta_q_from_envelope(
        z_coords, part_m2["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
    )
    bq_m = float(fit_m["beta_q"])
    bq_m2 = float(fit_m2["beta_q"])
    sigma_m = float(fit_m["sigma_beta_q"])
    sigma_m2 = float(fit_m2["sigma_beta_q"])
    beta0_2pt_12 = float((4.0 * bq_m2 - bq_m) / 3.0)
    sigma_prop_2pt = float(np.hypot(sigma_m / 3.0, 4.0 * sigma_m2 / 3.0))

    if include_m4:
        part_m4 = cached_particle_drift(
            nx,
            nz,
            profile="sheet_lattice",
            seed=NOISE_SEEDS[0],
            m_mod=0.25 * MODULATION_INDEX,
            qm_scale=1.0,
            shape=int(shape),
        )
        fit_m4 = fit_beta_q_from_envelope(
            z_coords, part_m4["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
        )
        bq_m4 = float(fit_m4["beta_q"])
        sigma_m4 = float(fit_m4["sigma_beta_q"])
        denom = bq_m2 - bq_m4
        q_m = float((bq_m - bq_m2) / denom) if abs(denom) > 1e-30 else float("nan")
        p_m = float(np.log2(q_m)) if q_m > 0.0 else float("nan")

        # Exact 3-point polynomial fit beta(m) = beta0 + c * m^2 + d * m^4 through (m, m/2, m/4)
        m1 = float(MODULATION_INDEX)
        m2 = 0.5 * m1
        m4 = 0.25 * m1
        vander = np.array(
            [
                [1.0, m1**2, m1**4],
                [1.0, m2**2, m2**4],
                [1.0, m4**2, m4**4],
            ],
            dtype=np.float64,
        )
        poly_coef = np.linalg.solve(
            vander, np.array([bq_m, bq_m2, bq_m4], dtype=np.float64)
        )
        beta0_3pt = float((64.0 * bq_m4 - 20.0 * bq_m2 + bq_m) / 45.0)
        beta0_2pt = float((4.0 * bq_m4 - bq_m2) / 3.0)
        e_r = float(abs(beta0_3pt - beta0_2pt))
        sigma_prop = float(
            np.sqrt(
                (sigma_m / 45.0) ** 2
                + (20.0 * sigma_m2 / 45.0) ** 2
                + (64.0 * sigma_m4 / 45.0) ** 2
            )
        )
        resid0 = float(abs(beta0_3pt / bq_pred - 1.0))
        bound0 = float(slack_part + (e_r + sigma_prop) / bq_pred)
        return {
            "bq_m": bq_m,
            "bq_m2": bq_m2,
            "bq_m4": bq_m4,
            "beta_m": bq_m,
            "beta_m2": bq_m2,
            "beta_m4": bq_m4,
            "sigma_m": sigma_m,
            "sigma_m2": sigma_m2,
            "sigma_m4": sigma_m4,
            "q_m": q_m,
            "p": p_m,
            "beta0_3pt": beta0_3pt,
            "beta0_2pt": beta0_2pt,
            "beta0_2pt_m_m2": beta0_2pt_12,
            "beta0": beta0_3pt,
            "bq0": beta0_3pt,
            "bq0_24": beta0_2pt,
            "c_m2": float(poly_coef[1]),
            "d_m4": float(poly_coef[2]),
            "e_R": e_r,
            "sigma_prop": sigma_prop,
            "resid0": resid0,
            "bound0": bound0,
            "bq_pred": bq_pred,
            "slack": slack_part,
        }

    resid0_2pt = float(abs(beta0_2pt_12 / bq_pred - 1.0))
    return {
        "bq_m": bq_m,
        "bq_m2": bq_m2,
        "bq_m4": float("nan"),
        "beta_m": bq_m,
        "beta_m2": bq_m2,
        "beta_m4": float("nan"),
        "sigma_m": sigma_m,
        "sigma_m2": sigma_m2,
        "sigma_m4": float("nan"),
        "q_m": float("nan"),
        "p": float("nan"),
        "beta0_3pt": float("nan"),
        "beta0_2pt": beta0_2pt_12,
        "beta0_2pt_m_m2": beta0_2pt_12,
        "beta0": beta0_2pt_12,
        "bq0": beta0_2pt_12,
        "bq0_24": float("nan"),
        "c_m2": float("nan"),
        "d_m4": float("nan"),
        "e_R": float("nan"),
        "sigma_prop": sigma_prop_2pt,
        "resid0": resid0_2pt,
        "bound0": slack_part,
        "bq_pred": bq_pred,
        "slack": slack_part,
        "m2_assumed": 1,
        "label": "m^2 assumed, untested on fine",
    }


@lru_cache(maxsize=8)
def measure_empirical_noise(
    grid: tuple[int, int, int] = COARSE_GRID,
    m_mod: float = MODULATION_INDEX,
    n_macro_per_step: int | None = None,
    shape: int = PARTICLE_SHAPE,
) -> dict[str, Any]:
    """Measure empirical 8-seed ``profile='sheet'`` statistics on ``grid``.

    Runs seeds ``42..49`` on ``grid`` at modulation depth ``m_mod`` using
    ``profile='sheet'`` (one seeded uniform draw per stratum per step from
    ``BeamInjector._step_rng``) at ``n_macro_per_step = 256`` (coarse) or ``512``
    (fine) alongside the ``profile='sheet_lattice'`` reference at the same
    ``n_macro_per_step``, fits the two-wave model over ``fit_window(params)``
    for each seed, and computes the 8-seed mean, standard error of the mean
    ``SEM = std(ddof=1) / sqrt(8)``, and two-sided 99.73% Student-t critical
    value ``t = scipy.stats.t.ppf(0.99865, 7)`` for ``beta_q``, DC depression
    ``delta_meas``, and entrance amplitude ``|I1(z_0)|``.
    """
    params = compute_analytic_parameters()
    nx, _, nz = grid
    n_macro = (
        int(n_macro_per_step)
        if n_macro_per_step is not None
        else (256 if nx == COARSE_GRID[0] else 512)
    )
    lb_res = cached_linearized_beam_drift(nx, nz, rho0_scale=1.0, m_mod=m_mod)
    z_coords = lb_res["z_coords"]
    beta_e = params["beta_e"]
    beta_q_th = params["beta_q"]
    lam_q = params["lambda_q"]
    z_min, z_max = fit_window(params)
    u0 = params["u0"]
    b = BEAM_HALF_THICKNESS_M

    k_idx = np.arange(n_macro, dtype=np.float64)
    x_lat_n = b * (2.0 * (k_idx + 0.5) / float(n_macro) - 1.0)
    bq_pred_n = float(
        discrete_modal_prediction(
            "particles", nx, nz, params, line_x=x_lat_n, shape=int(shape)
        )
    )
    delta_disc_n = float(
        compute_discrete_dc_depression(nx, params, line_x=x_lat_n, shape=int(shape))
    )
    delta_disc_charge_n = float(
        compute_discrete_dc_depression(
            nx, params, line_x=x_lat_n, weighting="charge", shape=int(shape)
        )
    )
    part_lat_n = cached_particle_drift(
        nx,
        nz,
        profile="sheet_lattice",
        seed=NOISE_SEEDS[0],
        m_mod=m_mod,
        qm_scale=1.0,
        n_macro_per_step=n_macro,
        shape=int(shape),
    )
    fit_lat_n = fit_beta_q_from_envelope(
        z_coords, part_lat_n["i1_part"], beta_q_th, z_min=z_min, z_max=z_max
    )
    bq_lat_n = float(fit_lat_n["beta_q"])
    delta_lat_n = float((u0 - part_lat_n["u_z_mean"]) / u0)

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
            n_macro_per_step=n_macro,
            shape=int(shape),
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
    i1_seeds = np.stack(i1_list, axis=0)
    i1_mean = np.mean(i1_seeds, axis=0)
    fit_coherent = fit_beta_q_from_envelope(
        z_coords, i1_mean, beta_q_th, z_min=z_min, z_max=z_max
    )
    bq_coherent_mean = float(fit_coherent["beta_q"])

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
        "n_macro_per_step": n_macro,
        "tcrit": tcrit,
        "bq_seeds": bq_seeds,
        "bq_mean": bq_mean,
        "bq_std": bq_std,
        "bq_sem": bq_sem,
        "bq_coherent_mean": bq_coherent_mean,
        "bq_lattice": bq_lat_n,
        "delta_lattice": delta_lat_n,
        "bq_pred": bq_pred_n,
        "delta_disc": delta_disc_n,
        "delta_disc_charge": delta_disc_charge_n,
        "part_lattice": part_lat_n,
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
        "i1_seeds": i1_seeds,
        "i1_mean": i1_mean,
    }
