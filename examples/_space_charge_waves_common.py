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
    _GPU_BIN_KERNEL = (
        cp.RawKernel(
            r"""
extern "C" __global__ void bin_beam_moments(
    const int n_p,
    const int nx,
    const int nz,
    const double a,
    const double dx,
    const double dz,
    const double c2,
    const double gamma0,
    const double q_over_mc2,
    const double* __restrict__ xp,
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

        double xi = (xp[i] + a) / dx;
        int ix0 = (int)floor(xi);
        if (ix0 < 0) ix0 = 0;
        else if (ix0 > nx - 1) ix0 = nx - 1;
        double wx1 = xi - (double)ix0;
        if (wx1 < 0.0) wx1 = 0.0;
        else if (wx1 > 1.0) wx1 = 1.0;
        double wx0 = 1.0 - wx1;
        double phi = wx0 * phi_nodes[ix0 * nz + b] + wx1 * phi_nodes[(ix0 + 1) * nz + b];
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
TUNNEL_LENGTH_M = 0.206  # Drift-tunnel axial length Lz >= 1.6 * lambda_q [m]
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

COARSE_GRID = (10, 1, 474)  # (Nx, Ny, Nz) coarse resolution
FINE_GRID = (20, 1, 948)  # (Nx, Ny, Nz) 2x refined resolution
NOISE_SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)  # 8 seeds for empirical noise


def fit_window(
    params: dict[str, float], outflow_ramp_m: float = OUTFLOW_RAMP_M
) -> tuple[float, float]:
    """Return the shared axial fit window ``(z_min, z_max)`` [m].

    Excludes ``d_w = ln(1 / eps_w) / gamma1`` at the entrance (where ``gamma1 =
    sqrt((pi / (2 * a))**2 - (Omega_f / c)**2)`` is the decay constant of the
    slowest even TM evanescent mode of the parallel-plate guide at relative
    amplitude threshold ``eps_w = 1e-5``) and ``d_w`` upstream of the outflow
    taper start ``z_ramp = TUNNEL_LENGTH_M - outflow_ramp_m``.
    """
    d_w = float(np.log(1.0 / EPS_WINDOW) / params["gamma1"])
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

    # Evanescent near-field decay constant of the slowest even TM parallel-plate mode
    # gamma1 = sqrt((pi / (2*a))^2 - (Omega_f / c)^2) and DC beam transition
    # kappa_dc = T1 * sqrt(1 - (omega_p / (u * T1))^2) > gamma1:
    dt_c, _ = _cached_integer_period_dt(COARSE_GRID[0], COARSE_GRID[2])
    omega_f = float((2.0 / dt_c) * np.sin(0.5 * omega * dt_c))
    gamma1 = float(np.sqrt((np.pi / (2.0 * a)) ** 2 - (omega_f / c_light) ** 2))
    kappa_dc = float(
        t1_transverse * np.sqrt(1.0 - (omega_p_dep / (u_dep * t1_transverse)) ** 2)
    )
    d_w = float(np.log(1.0 / EPS_WINDOW) / gamma1)

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
        "kappa_dc": float(kappa_dc),
        "eps_w": float(EPS_WINDOW),
        "d_w": float(d_w),
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
) -> float | dict[str, Any]:
    """Solve the 1D discrete Poisson problem across ``x`` on ``nx`` cells for ``delta_disc``.

    Deposits the sheet lines across ``[-b, b]`` onto the ``nx + 1`` transverse
    nodes using CIC weights, solves the 3-point discrete Poisson equation with
    PEC boundary conditions ``phi(+-a) = 0``, and iterates the exact per-line
    relativistic energy-conserving fixed point::

        phi_l(delta) = phi_l,0 / (1 - delta_bar)
        gamma_l = gamma0 - e * |phi_l| / (m_eff * c**2)
        u_l = c * sqrt(1 - 1 / gamma_l**2)
        delta_bar = 1 - sum_l(w_l * u_l) / (u0 * sum_l(w_l))

    to ``1e-14`` for both current/flux weighting (``w_l prop rho_l * u_l``) and
    density weighting (``w_l prop rho_l``).
    """
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    dx = 2.0 * a / nx
    x_nodes = np.linspace(-a, a, nx + 1)

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

    w_mat = np.maximum(1.0 - np.abs(x_sheet[:, None] - x_nodes[None, :]) / dx, 0.0)
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
            w_l = (rho_l * u_l) if mode == "flux" else rho_l
            delta_next = 1.0 - float(np.sum(w_l * u_l) / (u0 * np.sum(w_l)))
            if abs(delta_next - delta_val) < 1e-14:
                delta_val = float(delta_next)
                break
            delta_val = float(delta_next)
        return float(delta_val), u_l, gamma_l, phi_l

    delta_flux, u_l_flux, gamma_l_flux, phi_l_flux = _iterate_fixed_point("flux")
    delta_density, u_l_dens, gamma_l_dens, phi_l_dens = _iterate_fixed_point("density")

    if weighting == "flux":
        delta_disc, u_l, gamma_l, phi_l = delta_flux, u_l_flux, gamma_l_flux, phi_l_flux
    else:
        delta_disc, u_l, gamma_l, phi_l = (
            delta_density,
            u_l_dens,
            gamma_l_dens,
            phi_l_dens,
        )

    if return_details:
        return {
            "delta_disc": float(delta_disc),
            "delta_flux": float(delta_flux),
            "delta_density": float(delta_density),
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
) -> dict[str, Any]:
    """Build the discrete transverse modal operator ``G(k)`` and its eigensystem at wavenumber ``k_val``.

    Parameters
    ----------
    code : {"lb", "particles"}
        Model selector (`LinearizedBeam` node lines or `ParticleSolver` CIC lines).
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

        x_int = np.linspace(-a, a, nx + 1)[1:-1]
        w_cic = np.maximum(1.0 - np.abs(x_int[:, None] - x_p[None, :]) / dx, 0.0)
        s_mat = w_cic * ((2.0 * b / n_macro) / dx)
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
) -> float | dict[str, Any]:
    """Compute the Step 2 exact discrete transverse modal prediction ``beta_q,pred`` [rad/m].

    Parameters
    ----------
    code : {"lb", "particles"}
        Code selector (`"lb"` for `LinearizedBeam`, `"particles"` for `ParticleSolver`).
    nx, nz : int
        Transverse and axial cell counts.
    params : dict of str to float
        Output of :func:`compute_analytic_parameters`.
    line_x : np.ndarray, optional
        Injected particle line ``x`` positions [m] (for ``code == "particles"``).
    line_w : np.ndarray, optional
        Line charge weights (for non-uniform profiles such as D3).
    entrance_profile : np.ndarray, optional
        Entrance modulation profile across lines (defaults to ``_w_perp_eff`` for
        `LinearizedBeam` and uniform charge per line for `ParticleSolver`).
    include_shear : bool, optional
        For ``code == "particles"``, include the per-line DC velocity shear
        ``(u_l, gamma_l)`` from Step 1. Default is ``True``.
    include_sz : bool, optional
        For ``code == "particles"``, include the drifting-beam principal-mode axial
        shape factor ``S_z(k)**2 = sinc(k * dz / 2)**2`` (Step 3). Default is ``True``.
    use_omega_f : bool, optional
        If ``True`` (default), use the leapfrog field time symbol ``Omega_f`` in
        ``tau_Y(k)**2 = kappa(k)**2 - (Omega_f / c)**2``; if ``False``, use
        continuum ``omega`` (for ``s_sym``).
    drop_highest_mode : bool, optional
        If ``True``, omit the highest-order excited transverse mode (used for ``s_modes``).
    win_shift : float, optional
        Shift [m] applied to both bounds of ``fit_window(params)`` (used for ``s_win``).
    omega_p_scale : float, optional
        Multiplier on ``omega_p`` (``sqrt(2)`` for M2). Default is ``1.0``.
    return_details : bool, optional
        If ``True``, return a dictionary with modal eigenvalues, wavenumbers,
        weights, and synthesized ``I1_model(z)``. Default is ``False``.

    Returns
    -------
    float or dict of str to Any
        Predicted two-wave fitted ``beta_q`` [rad/m] (or detail dictionary when
        ``return_details=True``).
    """
    b = BEAM_HALF_THICKNESS_M
    lz = TUNNEL_LENGTH_M
    dz = lz / nz

    dt, _ = _cached_integer_period_dt(int(nx), int(nz))
    omega = params["omega"]

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
            nx, params, line_x=line_x, line_w=line_w, return_details=True
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
            )
            # Drifting-beam axial shape factor S_z(k)^2 = sinc^2(k*dz/2) from
            # p=1 Esirkepov J_z deposit symbol sinc(k*dz/2) and p_g=0 Galerkin
            # E_z gather symbol sinc(k*dz/2) (principal m=0 mode; aliases
            # k_m = k + 2*pi*m/dz are Doppler-detuned by 2*pi*|m|*u_dep/dz >> omega_q).
            sz_sq = float(np.sinc(0.5 * k_val * dz / np.pi) ** 2) if include_sz else 1.0
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

    c_use = c_modes.copy()
    if drop_highest_mode:
        excited = np.where(np.abs(c_use) > 1e-12)[0]
        if len(excited) > 1:
            c_use[int(excited[-1])] = 0.0

    z_eval = (np.arange(nz, dtype=np.float64) + 0.5) * dz
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
        return {
            "beta_q": bq_pred,
            "evals": evals_be,
            "bq_modes": bq_modes,
            "kc_modes": kc_modes,
            "c_modes": c_modes,
            "z_coords": z_eval,
            "i1_model": i1_model,
            "fit": fit_res,
        }
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
            )
        )
        res["bq_pred_unif"] = bq_unif
        res["bq_pred_no_sz"] = bq_no_sz
        res["shear_shift"] = float(bq_pred - bq_unif)
        res["shear_rel"] = float(bq_pred / bq_unif - 1.0)
    return res


def _single_grid_budget_terms(
    nx: int,
    nz: int,
    params: dict[str, float],
    omega_p_scale: float = 1.0,
) -> dict[str, Any]:
    """Compute Step 2 modal predictions and Step 3 slacks for grid ``(nx, 1, nz)``."""
    a = TUNNEL_HALF_WIDTH_M
    dz = TUNNEL_LENGTH_M / nz
    beta_e = params["beta_e"]
    beta_q = params["beta_q"] * float(omega_p_scale)

    dt_grid, _ = _cached_integer_period_dt(int(nx), int(nz))
    omega_f_grid = float((2.0 / dt_grid) * np.sin(0.5 * params["omega"] * dt_grid))
    gamma1 = float(np.sqrt((np.pi / (2.0 * a)) ** 2 - (omega_f_grid / c_light) ** 2))
    kappa_dc = float(params["kappa_dc"])
    d_w = float(np.log(1.0 / EPS_WINDOW) / gamma1)
    z_min, z_max = fit_window(params)

    lb_modal = prediction_slack("lb", nx, nz, params, omega_p_scale=omega_p_scale)
    part_modal = prediction_slack(
        "particles", nx, nz, params, omega_p_scale=omega_p_scale
    )

    # Brillouin alias suppression factor (omega_q * dz / (2 * pi * u_dep))^2 for m != 0:
    omega_q_eff = params["omega_q"] * float(omega_p_scale)
    alias_supp = float((omega_q_eff * dz / (2.0 * np.pi * params["u_dep"])) ** 2)

    # Discrete 1D Poisson DC depression prediction on this grid (Step 1):
    dep_info = compute_discrete_dc_depression(nx, params, return_details=True)
    delta_th_disc = float(dep_info["delta_disc"])
    eps_ac = float(0.5 * (MODULATION_INDEX * beta_q / beta_e) ** 2)

    # Entrance amplitude bounds at first primal edge midpoint z_0 = 0.5 * dz:
    # For particles, edge-0 current is the flux averaged over a one-cell transit,
    # giving predicted ratio sinc(beta_e * dz / 2) * cos(beta_q * dz / 2) and bound
    # from the first neglected Taylor terms (beta_e*dz/2)^4/120 + (beta_q*dz/2)^4/24 + 1e-12.
    tol_amp_lb = float(0.5 * (beta_q * dz) ** 2)
    be_half = 0.5 * beta_e * dz
    bq_half = 0.5 * beta_q * dz
    amp_pred_part = float(np.sinc(be_half / np.pi) * np.cos(bq_half))
    tol_amp_part = float((be_half**4) / 120.0 + (bq_half**4) / 24.0 + 1.0e-12)

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
        "kappa_dc": kappa_dc,
        "eps_w": float(EPS_WINDOW),
        "d_w": d_w,
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
        "slack_cross": float(lb_modal["slack"] + part_modal["slack"]),
        "alias_supp": alias_supp,
        "delta_th": float(params["delta_u_dc"]),
        "delta_th_flux": float(params["delta_u_dc_flux"]),
        "delta_th_density": float(params["delta_u_dc_density"]),
        "delta_th_disc": delta_th_disc,
        "delta_disc_exact": delta_th_disc,
        "delta_disc_flux": float(dep_info["delta_flux"]),
        "delta_disc_density": float(dep_info["delta_density"]),
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
) -> dict[str, Any]:
    """Build the Step 2 modal predictions and Step 3 slacks for grid ``(nx, 1, nz)``."""
    return _single_grid_budget_terms(nx, nz, params, omega_p_scale=omega_p_scale)


def format_error_budget(nx: int, nz: int, tols: dict[str, Any]) -> str:
    """Format every Step 2 modal prediction and Step 3 slack term for grid ``(nx, 1, nz)``."""
    ev_lb = tols["evals_lb"]
    ev_p = tols["evals_part"]
    lines = [
        f"Modal prediction and slack budget for grid ({nx}, 1, {nz}):",
        f"  gamma1 / kappa_dc / d_w / window                  : gamma1={tols['gamma1']:.4f} 1/m, kappa_dc={tols['kappa_dc']:.4f} 1/m, d_w={tols['d_w'] * 1e3:.4f} mm, [{tols['z_min'] * 1e3:.4f}, {tols['z_max'] * 1e3:.4f}] mm",
        f"  bq_pred_lb (Step 2 LB modal prediction)           : {tols['bq_pred_lb']:.6f} rad/m",
        f"  slack_lb = s_win + s_modes + s_sym + s_round      : {tols['slack_lb']:.6%}",
        f"    (s_win={tols['s_win_lb']:.3e}, s_modes={tols['s_modes_lb']:.3e}, s_sym={tols['s_sym_lb']:.3e}, s_round={tols['s_round']:.1e})",
        f"  LB eigenvalues lambda_1..4(beta_e)                : ({ev_lb[0]:.6f}, {ev_lb[1]:.6f}, {ev_lb[2]:.6f}, {ev_lb[3]:.6f})",
        f"  bq_pred_part (Step 2+3 Particle modal prediction) : {tols['bq_pred_part']:.6f} rad/m (no_Sz={tols['bq_pred_part_no_sz']:.6f}, unif={tols['bq_pred_part_unif']:.6f}, shear={tols['shear_shift']:+.6e})",
        f"  slack_part = s_win + s_modes + s_sym + s_round    : {tols['slack_part']:.6%}",
        f"    (s_win={tols['s_win_part']:.3e}, s_modes={tols['s_modes_part']:.3e}, s_sym={tols['s_sym_part']:.3e}, s_round={tols['s_round']:.1e})",
        f"  Particle eigenvalues lambda_1..4(beta_e)          : ({ev_p[0]:.6f}, {ev_p[1]:.6f}, {ev_p[2]:.6f}, {ev_p[3]:.6f})",
        f"  alias_supp ((omega_q*dz/(2*pi*u_dep))^2)          : {tols['alias_supp']:.6e}",
        f"  slack_cross (slack_lb + slack_part)               : {tols['slack_cross']:.6%}",
        f"  delta_th (continuum DC dep, flux/density)         : {tols['delta_th_flux']:.6%} / {tols['delta_th_density']:.6%}",
        f"  delta_disc_exact (1D Poisson DC dep, {tols['dep_weighting']})        : {tols['delta_disc_exact']:.6%} (flux={tols['delta_disc_flux']:.6%}, density={tols['delta_disc_density']:.6%})",
        f"  eps_ac (0.5*(m*bq/be)^2)                          : {tols['eps_ac']:.6e}",
        f"  amp_pred_part (sinc(be*dz/2)*cos(bq*dz/2))        : {tols['amp_pred_part']:.8f}",
        f"  tol_amp_lb (0.5*(bq*dz)^2)                        : {tols['tol_amp_lb']:.6e}",
        f"  tol_amp_part ((be*dz/2)^4/120+(bq*dz/2)^4/24+1e-12): {tols['tol_amp_part']:.6e}",
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
    gather: str = "galerkin",
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

    n_macro = (
        int(n_macro_per_step)
        if n_macro_per_step is not None
        else (16 if nx == COARSE_GRID[0] else 32)
    )
    n_settle = int(np.ceil(1.01 * lz / (u_dep * dt)))
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
        shape=1,
        gather=gather,
        B_ext=(0.0, 0.0, GUIDE_BZ_TESLA),
    ).attach()

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
            j_raw = ps.J_particles[beam_ref.flat_idx]
            j_host = np.asarray(j_raw.get() if hasattr(j_raw, "get") else j_raw)
            j_edges = j_host.reshape(beam_ref.M, beam_ref.N_lines)
            iz_hist[k, :] = np.sum(j_edges * beam_ref.dA_perp, axis=1)

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
                    n_blocks = min(max((n_p_cur + 255) // 256, 1), 256)
                    _GPU_BIN_KERNEL(
                        (n_blocks,),
                        (256,),
                        (
                            n_p_cur,
                            int(nx),
                            int(nz),
                            float(a),
                            float(dx),
                            float(dz),
                            c2_val,
                            float(gamma0),
                            q_over_mc2_val,
                            sp._x,
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

                xi_p = (xp + a) / dx
                ix0 = np.clip(np.floor(xi_p).astype(np.int64), 0, nx - 1)
                wx1 = np.clip(xi_p - ix0, 0.0, 1.0)
                wx0 = 1.0 - wx1
                phi_p = (
                    wx0 * phi_nodes[ix0, bin_idx] + wx1 * phi_nodes[ix0 + 1, bin_idx]
                )
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
def _cached_particle_drift_impl(
    nx: int,
    nz: int,
    profile: str,
    seed: int,
    m_mod: float,
    qm_scale: float,
    n_macro: int,
    gather: str,
) -> dict[str, Any]:
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
        n_macro_per_step=n_macro,
        gather=gather,
    )


def cached_particle_drift(
    nx: int,
    nz: int,
    profile: str = "sheet_lattice",
    seed: int = 42,
    m_mod: float = MODULATION_INDEX,
    qm_scale: float = 1.0,
    n_macro_per_step: int | None = None,
    gather: str = "galerkin",
) -> dict[str, Any]:
    """Cache `ParticleSolver` runs by ``(nx, nz, profile, seed, m_mod, qm_scale, n_macro_per_step, gather)``."""
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
def compute_m_extrapolation(
    nx: int, nz: int, include_m4: bool = True
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
    tols = derived_tolerances(nx, nz, params)
    bq_pred = float(tols["bq_pred_part"])
    slack_part = float(tols["slack_part"])
    z_min, z_max = fit_window(params)
    beta_q_th = params["beta_q"]

    lb_m, part_m = cached_baseline_pair(nx, nz, m_mod=MODULATION_INDEX)
    z_coords = lb_m["z_coords"]
    part_m2 = cached_particle_drift(
        nx,
        nz,
        profile="sheet_lattice",
        seed=NOISE_SEEDS[0],
        m_mod=0.5 * MODULATION_INDEX,
        qm_scale=1.0,
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
        discrete_modal_prediction("particles", nx, nz, params, line_x=x_lat_n)
    )
    delta_disc_n = float(compute_discrete_dc_depression(nx, params, line_x=x_lat_n))
    part_lat_n = cached_particle_drift(
        nx,
        nz,
        profile="sheet_lattice",
        seed=NOISE_SEEDS[0],
        m_mod=m_mod,
        qm_scale=1.0,
        n_macro_per_step=n_macro,
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
