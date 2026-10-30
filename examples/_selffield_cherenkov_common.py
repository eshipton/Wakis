# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Shared dispersion, resonance, and simulation helpers for the numerical Cherenkov benchmark.

Imported by both ``tests/test_084_particles_selffield_cherenkov.py`` and
``examples/021_particles_numerical_cherenkov.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c, m_e
from scipy.constants import e as e_charge
from scipy.optimize import brentq

from wakis import GridFIT3D, SolverFIT3D
from wakis.estimate import estimate
from wakis.particles import ParticleSolver, Species


def binomial_transfer_1d(u: float | np.ndarray, compensated: bool = True):
    """Evaluate the 1D 1-2-1 binomial filter transfer function at ``u = k * Delta``.

    Parameters
    ----------
    u : float or ndarray
        Dimensionless wavenumber ``k * Delta`` [rad].
    compensated : bool, optional
        If True (default), include the second sharpening pass ``alpha_2 = -1/6``
        with symbol ``1 - 4 * alpha_2 * sin^2(u/2) = 1 + (2/3) * sin^2(u/2)``.
        If False, return the single-pass uncompensated symbol ``cos^2(u/2)``.

    Returns
    -------
    float or ndarray
        Filter amplitude transfer factor ``T(u)`` [-].
    """
    s2 = np.sin(0.5 * np.asarray(u)) ** 2
    t_uncomp = 1.0 - s2
    if not compensated:
        return float(t_uncomp) if np.ndim(u) == 0 else t_uncomp
    t_comp = t_uncomp * (1.0 + (2.0 / 3.0) * s2)
    return float(t_comp) if np.ndim(u) == 0 else t_comp


def compute_dispersion_and_resonance(
    beta_fast: float = 0.86,
    beta_ctrl: float = 0.50,
    cfln: float = 0.9,
    dz: float = 0.002,
    n_xy: int = 16,
    dt: float | None = None,
) -> dict:
    """Compute discrete FIT/Yee dispersion curves, v_min, m=0 and m=+-1 resonances, and filter transfers.

    Parameters
    ----------
    beta_fast : float, optional
        Super-``v_min`` beam normalized velocity ``v_0 / c`` [-]. Default is ``0.86``.
    beta_ctrl : float, optional
        Sub-``v_min`` control beam normalized velocity ``v_0 / c`` [-]. Default is ``0.50``.
    cfln : float, optional
        3D Courant factor ``c * dt * sqrt(3) / dz`` [-]. Default is ``0.9``.
    dz : float, optional
        Uniform cubic cell spacing ``Delta`` [m]. Default is ``0.002``.
    n_xy : int, optional
        Transverse cell count ``N_x = N_y = N_perp`` of the square PEC pipe [-].
        Default is ``16``.
    dt : float or None, optional
        Explicit time step [s]. If ``None``, computed as ``cfln * dz / (sqrt(3) * c)``.

    Returns
    -------
    dict
        Dispersion parameters, predicted ``m = 0`` and ``m = -1`` resonant
        wavenumbers, binomial filter transfers, and gridded curves for plotting.
    """
    if dt is None:
        dt = float(cfln * dz / (np.sqrt(3.0) * c))
    sigma = float(c * dt / dz)
    v_min = float((2.0 / (c * dt)) * np.arcsin(sigma) * (dz / np.pi) * c)

    f_1d = lambda u: np.sin(0.5 * beta_fast * sigma * u) - sigma * np.sin(0.5 * u)
    u_res_1d = float(brentq(f_1d, 1.0e-3, np.pi))
    k_res_1d = float(u_res_1d / dz)
    omega_res_1d = float(k_res_1d * beta_fast * c)

    s_perp2 = float(2.0 * np.sin(np.pi / (2.0 * n_xy)) ** 2)
    f_3d = lambda u: (
        np.sin(0.5 * beta_fast * sigma * u)
        - sigma * np.sqrt(s_perp2 + np.sin(0.5 * u) ** 2)
    )
    u_res_3d = float(brentq(f_3d, 1.0e-3, np.pi))
    k_res_3d = float(u_res_3d / dz)
    omega_res_3d = float(k_res_3d * beta_fast * c)

    # Aliased m = -1 branch |omega_{-1}(u)| = (2*pi - u)*beta*c/dz:
    # For beta_fast = 0.86 > v_min/c, |omega_{-1}(u)| > omega_d(k_{perp,11}, u/dz)
    # on [0, pi] (no root on the (1, 1) waveguide branch, though high-k_perp oblique
    # branches can resonate since 2*arcsin(sqrt(3)*sigma) ~ 2.24 > pi*beta_fast*sigma ~ 1.40).
    # For beta_ctrl = 0.50 < v_min/c, |omega_{-1}(u)| intersects omega_d in (0, pi).
    f_ctrl_m1_1d = lambda u: (
        np.sin(0.5 * beta_ctrl * sigma * (2.0 * np.pi - u)) - sigma * np.sin(0.5 * u)
    )
    u_res_ctrl_m1_1d = float(brentq(f_ctrl_m1_1d, 1.0e-3, np.pi))
    f_ctrl_m1_3d = lambda u: (
        np.sin(0.5 * beta_ctrl * sigma * (2.0 * np.pi - u))
        - sigma * np.sqrt(s_perp2 + np.sin(0.5 * u) ** 2)
    )
    u_res_ctrl_m1_3d = float(brentq(f_ctrl_m1_3d, 1.0e-3, np.pi))
    omega_res_ctrl_m1_3d = float((2.0 * np.pi - u_res_ctrl_m1_3d) * beta_ctrl * c / dz)
    half_arg_m1 = 0.5 * (u_res_ctrl_m1_3d - 2.0 * np.pi)
    cic_alias_weight_ctrl_m1 = float((np.sin(half_arg_m1) / half_arg_m1) ** 4)

    t_uncomp_1d = float(binomial_transfer_1d(u_res_1d, compensated=False))
    t_comp_1d = float(binomial_transfer_1d(u_res_1d, compensated=True))
    t_uncomp_3d = float(binomial_transfer_1d(u_res_3d, compensated=False))
    t_comp_3d = float(binomial_transfer_1d(u_res_3d, compensated=True))
    t_perp_1d = float(binomial_transfer_1d(np.pi / n_xy, compensated=True))

    u_grid = np.linspace(0.0, np.pi, 400)
    kz_grid = u_grid / dz
    omega_vac = c * kz_grid
    omega_d_1d = (2.0 / dt) * np.arcsin(np.clip(sigma * np.sin(0.5 * u_grid), 0.0, 1.0))
    omega_d_wg = (2.0 / dt) * np.arcsin(
        np.clip(sigma * np.sqrt(s_perp2 + np.sin(0.5 * u_grid) ** 2), 0.0, 1.0)
    )
    omega_beam_fast = kz_grid * (beta_fast * c)
    omega_beam_ctrl = kz_grid * (beta_ctrl * c)
    omega_beam_fast_m_minus1 = (2.0 * np.pi / dz - kz_grid) * (beta_fast * c)
    omega_beam_fast_m_plus1 = (2.0 * np.pi / dz + kz_grid) * (beta_fast * c)
    omega_beam_ctrl_m_minus1 = (2.0 * np.pi / dz - kz_grid) * (beta_ctrl * c)
    omega_beam_ctrl_m_plus1 = (2.0 * np.pi / dz + kz_grid) * (beta_ctrl * c)

    t_uncomp_grid = binomial_transfer_1d(u_grid, compensated=False)
    t_comp_grid = binomial_transfer_1d(u_grid, compensated=True)

    return {
        "dt": float(dt),
        "dz": float(dz),
        "sigma": sigma,
        "v_min": v_min,
        "v_min_m_s": v_min,
        "v_min_over_c": float(v_min / c),
        "beta_fast": float(beta_fast),
        "beta_ctrl": float(beta_ctrl),
        "s_perp2": s_perp2,
        "u_res": u_res_1d,
        "k_res": k_res_1d,
        "u_res_1d_rad": u_res_1d,
        "k_res_1d_m_inv": k_res_1d,
        "omega_res_1d_rad_s": omega_res_1d,
        "u_res_3d": u_res_3d,
        "k_res_3d": k_res_3d,
        "u_res_3d_rad": u_res_3d,
        "k_res_3d_m_inv": k_res_3d,
        "omega_res_3d_rad_s": omega_res_3d,
        "u_res_ctrl_m1_1d_rad": u_res_ctrl_m1_1d,
        "u_res_ctrl_m1_3d_rad": u_res_ctrl_m1_3d,
        "omega_res_ctrl_m1_3d_rad_s": omega_res_ctrl_m1_3d,
        "cic_alias_weight_ctrl_m1": cic_alias_weight_ctrl_m1,
        "t_uncomp": t_uncomp_1d,
        "t_comp": t_comp_1d,
        "t_uncomp_at_res": t_uncomp_3d,
        "t_uncomp_power_at_res": float(t_uncomp_3d**2),
        "t_comp_at_res": t_comp_3d,
        "t_comp_power_at_res": float((t_comp_3d**2) * (t_perp_1d**4)),
        "t_perp_1d": t_perp_1d,
        "u_grid": u_grid,
        "kz_grid": kz_grid,
        "omega_vac": omega_vac,
        "omega_d_1d": omega_d_1d,
        "omega_d_wg": omega_d_wg,
        "omega_beam_fast": omega_beam_fast,
        "omega_beam_ctrl": omega_beam_ctrl,
        "omega_beam_fast_m_minus1": omega_beam_fast_m_minus1,
        "omega_beam_fast_m_plus1": omega_beam_fast_m_plus1,
        "omega_beam_ctrl_m_minus1": omega_beam_ctrl_m_minus1,
        "omega_beam_ctrl_m_plus1": omega_beam_ctrl_m_plus1,
        "t_uncomp_grid": t_uncomp_grid,
        "t_comp_grid": t_comp_grid,
    }


def compute_ctrl_no_resonance_threshold(
    beta_ctrl: float, dt: float, dz: float, n_xy: int, u_res: float
) -> float:
    """Compute a heuristic off-resonance control bound on ``|Delta E_tilde(k_res)|^2 / median_{k_z>0} |Delta E_tilde(k_z)|^2``.

    In the co-moving frame of the beam, a driven waveguide mode with detuning
    ``Delta(k_z) = omega_d(k_{perp,11}, k_z) - k_z * v_0`` obeys
    ``da/dt = i * Delta * a + s``, with solution
    ``a(t) = s * (exp(i * Delta * t) - 1) / (i * Delta)``.
    The difference of the two half-window means over ``[0, T/2]`` and ``[T/2, T]`` is
    ``Delta a = s * (exp(i * Delta * T / 2) - 1)^2 / ((i * Delta)^2 * (T / 2))``,
    which equals ``s * T / 2`` on resonance (``Delta -> 0``) and is bounded off
    resonance by ``|Delta a| <= 8 * |s| / (Delta^2 * T)`` (so the per-mode power
    envelope scales as ``|s|^2 / Delta(k_z)^4``).
    For the sub-``v_min`` control (``beta_ctrl < v_min / c``), ``Delta(u) > 0``
    throughout ``(0, pi]`` (note that ``Delta(u)`` is not monotone across the
    upper Brillouin zone: at ``beta_ctrl = 0.50``, ``Delta * dt`` is ``~0.360``
    at ``u_med = pi / 2``, reaches ``~0.404`` near ``u_res``, and decreases to
    ``~0.277`` at ``u = pi`` as ``omega_d`` flattens at the zone boundary).
    Combining the envelope ratio ``(Delta(u_med) / Delta(u_res))^4`` with the
    ``alpha = 1e-3`` exponential shot-noise quantile ``-ln(1e-3) / ln(2)`` gives a
    heuristic no-resonance threshold on
    ``|Delta E_tilde(k_res)|^2 / median(|Delta E_tilde|^2)``.
    """
    sigma = c * dt / dz
    s_perp2 = 2.0 * np.sin(np.pi / (2.0 * n_xy)) ** 2
    u_med = 0.5 * np.pi
    omega_med = (2.0 / dt) * np.arcsin(
        np.clip(sigma * np.sqrt(s_perp2 + np.sin(0.5 * u_med) ** 2), 0.0, 1.0)
    )
    omega_res = (2.0 / dt) * np.arcsin(
        np.clip(sigma * np.sqrt(s_perp2 + np.sin(0.5 * u_res) ** 2), 0.0, 1.0)
    )
    delta_med = omega_med - (u_med / dz) * (beta_ctrl * c)
    delta_res = omega_res - (u_res / dz) * (beta_ctrl * c)
    return float(((delta_med / delta_res) ** 4) * (-np.log(1.0e-3) / np.log(2.0)))


def run_cherenkov_case(
    beta0: float,
    filter_mode: str | None = None,
    n_xy: int = 16,
    n_z: int = 64,
    n_steps: int = 480,
    n_slices: int = 64,
    seed: int = 42,
) -> dict:
    """Run a periodic-z PEC pipe beam case and return its m=0 and m=-1 demodulated k_z spectra.

    Separating radiated energy from the co-moving self-field
    --------------------------------------------------------
    Because ``E(0) = 0`` at ``attach()``, ``ParticleSolver`` records a stationary
    neutralising background charge ``q_bg = -q_nodes(0)``. A rigid beam moving at
    constant velocity ``v0 = beta0 * c`` carries a steady co-moving self-field
    ``E_x^{self}(x, y, z - v0 * t)`` whose longitudinal Fourier transform evolves
    by phase translation ``exp(-i * k_z * v0 * t)``, while the stationary
    background field is time-independent in the lab frame and has phase
    ``exp(+i * k_z * v0 * t)`` in the co-moving frame.
    Transforming to the ``m = 0`` beam co-moving frame via
    ``E_tilde(k_z, t) = exp(+i * k_z * v0 * t) * E_hat_x(k_z, t)`` and taking the
    difference between the second half ``[T/2, T]`` and first half ``[0, T/2]``
    of the run cancels the time-independent co-moving self-field identically for
    any ``T``, while suppressing the oscillatory ``exp(+i * k_z * v0 * t)``
    contribution of the frozen background charge ``q_bg`` by
    ``O(1 / (k_z * v0 * T / 2))`` (``<= 4 / (k_z * v0 * (T / 2))``) and isolating
    the secularly growing Cherenkov radiation mode at ``omega_d(k_z) = k_z * v0``.
    """
    dx = 0.002
    l_xy = n_xy * dx
    l_z = n_z * dx
    n_p = ((n_xy - 1) ** 2) * n_slices

    # Preflight memory footprint before allocating grid and particle arrays
    est = estimate(cells=(n_xy, n_xy, n_z), steps=n_steps, backend="cpu")
    assert est.host_bytes < 1.5e9, (
        f"Preflight host memory {est.host_bytes} exceeds 1.5 GB"
    )

    grid = GridFIT3D(
        xmin=-0.5 * l_xy,
        xmax=0.5 * l_xy,
        ymin=-0.5 * l_xy,
        ymax=0.5 * l_xy,
        zmin=0.0,
        zmax=l_z,
        Nx=n_xy,
        Ny=n_xy,
        Nz=n_z,
        use_mpi=False,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.9,
        bc_low=["pec", "pec", "periodic"],
        bc_high=["pec", "pec", "periodic"],
        use_stl=False,
        verbose=0,
    )

    rng = np.random.default_rng(seed)
    xs_1d = (np.arange(1, n_xy) - 0.5 * n_xy) * dx
    xx, yy = np.meshgrid(xs_1d, xs_1d, indexing="ij")
    w_trans = (np.cos(np.pi * xx / l_xy) * np.cos(np.pi * yy / l_xy)).ravel()
    x_trans = xx.ravel()
    y_trans = yy.ravel()

    z_slices = l_z * rng.random(n_slices)
    x_p = np.tile(x_trans, n_slices)
    y_p = np.tile(y_trans, n_slices)
    z_p = np.repeat(z_slices, len(x_trans))
    w_rel = np.tile(w_trans, n_slices)

    gamma0 = 1.0 / np.sqrt(1.0 - beta0 * beta0)
    v0 = beta0 * c
    uz0 = beta0 * gamma0 * c
    i_beam_a = 1.0e-6
    q_total = i_beam_a * l_z / v0
    w_p = w_rel * (q_total / (e_charge * np.sum(w_rel)))

    sp = Species(name="beam", charge=e_charge, mass=m_e, capacity=n_p + 16)
    sp.add_particles(
        x=x_p,
        y=y_p,
        z=z_p,
        ux=np.zeros(n_p, dtype=np.float64),
        uy=np.zeros(n_p, dtype=np.float64),
        uz=np.full(n_p, uz0, dtype=np.float64),
        w=w_p,
    )
    ParticleSolver(
        solver,
        species=[sp],
        pusher="boris",
        shape=1,
        filter=filter_mode,
        sort_every=0,
    ).attach()

    kz_arr = 2.0 * np.pi * np.fft.rfftfreq(n_z, d=dx)
    omega_m1_arr = (kz_arr - 2.0 * np.pi / dx) * v0
    half = n_steps // 2
    e_beam_w1 = np.zeros(len(kz_arr), dtype=np.complex128)
    e_beam_w2 = np.zeros(len(kz_arr), dtype=np.complex128)
    e_m1_w1 = np.zeros(len(kz_arr), dtype=np.complex128)
    e_m1_w2 = np.zeros(len(kz_arr), dtype=np.complex128)
    i_obs = n_xy // 2 + 4
    j_obs = n_xy // 2
    dt = float(solver.dt)

    for step in range(n_steps):
        solver.one_step()
        t_now = (step + 1) * dt
        ex_line = solver.E[i_obs, j_obs, :, "x"]
        ex_fft = np.fft.rfft(ex_line)
        ex_hat = ex_fft * np.exp(1j * kz_arr * v0 * t_now)
        ex_hat_m1 = ex_fft * np.exp(1j * omega_m1_arr * t_now)
        if step < half:
            e_beam_w1 += ex_hat
            e_m1_w1 += ex_hat_m1
        else:
            e_beam_w2 += ex_hat
            e_m1_w2 += ex_hat_m1

    e_beam_w1 /= half
    e_beam_w2 /= n_steps - half
    delta_e_beam = e_beam_w2 - e_beam_w1
    spec_rad = np.abs(delta_e_beam) ** 2

    e_m1_w1 /= half
    e_m1_w2 /= n_steps - half
    spec_rad_m_minus_1 = np.abs(e_m1_w2 - e_m1_w1) ** 2

    return {
        "kz_arr": kz_arr,
        "u_arr": kz_arr * dx,
        "spec_rad": spec_rad,
        "spec_rad_m_minus_1": spec_rad_m_minus_1,
        "dt": dt,
        "dx": dx,
        "n_xy": n_xy,
        "n_z": n_z,
        "n_steps": n_steps,
        "estimate": est,
    }
