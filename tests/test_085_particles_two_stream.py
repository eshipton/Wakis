# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""V4 verification suite for the two-stream instability (with the gamma0^(-3/2) longitudinal-mass correction) in ``wakis.particles``.

Covers:
1. Analytical and discrete Birdsall-Langdon / FIT dispersion relation for two cold
   counter-streaming electron beams at ``+/- v0`` over an immobile neutralising
   background, reusing ``cold_plasma_dispersion_discrete`` in the ``v0 -> 0`` limit.
2. Linear growth rate check (a): exponential fit of the ``k1`` Fourier mode of ``E_z``
   over the per-case rule-defined linear window ``[t_start(case), t_end(ALPHA_SAT)]``
   (where ``t_start(case)`` is set from the analytic ``omega_+`` share of the position-only
   seed via ``(a_osc / a_grow) * exp(-gamma_disc(case) * t_start) <= 1e-3``, the
   predicted window end is ``t_end = ln(3 * ALPHA_SAT / (8 * eps * a_grow)) / gamma_disc(case)``
   with ``E_0 / E_trap = 8 * eps / 3``, and each unstable case runs to its per-case
   horizon ``1.5 * t_end_pred(case)``, with no window selection by trial runs and no
   ``gamma_fit = 0.0`` fallback) against the discrete prediction within the tolerance
   ``3 * sigma_fit + delta_gamma_nl + delta_gamma_lat`` (where ``3 * sigma_fit`` is labelled
   empirical (single-run OLS SE), ``delta_gamma_nl = (3/16) * |gamma_disc| * (|E_1| / E_trap)^2``
   is an order-of-magnitude quadratic nonlinear detuning bound, and ``delta_gamma_lat`` is a
   heuristic particle-lattice alias bound), exact discrete Gauss conservation at machine
   round-off (with ``gauss_init`` labelled as zero by construction and ``gauss_final`` as
   the real conservation check), absence of growth for a stable wavenumber
   ``k_2 * v0 = sqrt(3) * omega_{b,rel} > sqrt(2) * omega_{b,rel}`` (verified by an
   argument-principle winding-number root count confirming 0 upper-half-plane roots at
   ``k_2`` and 1 at ``k_1``, a dominant-alias ``|m| <= 4`` complex-root search, a dedicated
   short stable horizon ``T_stable = 3 * pi / omega_-(k_2)`` with ``max |E_{k1}| <= 1e-3 * |E_{k2}(0)|``
   guarding against second-harmonic contamination, and a last-third vs first-third beat
   envelope comparison within ``tol_env = (A_- / A_max) * (1 - cos(pi * omega_- / omega_+))``
   with stated minimum detectable growth rate), and verification of an asymmetric-density
   case ``n_+ : n_- = 1 : 0.5`` with a complex growing root.
3. Convergence check (b): 2nd-order spatial grid convergence (``Nz in {16, 32}``) of
   ``|gamma_sim - gamma_continuum|`` (where ``Nz = 16`` resolves the plasma wavelength
   with ``Nz * sqrt(3) / (2 * sqrt(2))`` cells as a deliberate deviation below the ``>= 20`` rule in
   ``docs/physicsguide_sections/particles_two_stream.md`` so the discrete grid shift
   is clearly separated from ``tol_gamma``) and ``1 / sqrt(N_ppc)`` macroparticle
   noise convergence (``N_ppc in {8, 128}`` across 16 seeds in parallel with the
   ``chi^2``-derived standard-deviation-ratio order band, noting its low statistical power
   that cannot distinguish order ``0.5`` from ``0.25`` or ``0.75``) using the same per-case
   ``t_start``, ``ALPHA_SAT``, and per-case horizon ``1.5 * t_end_pred(case)``.
4. Code-path dispersion mutations and periodic neutrality input check:
   - Monkeypatching the momentum push so one species sees the opposite force sign
     (``-q E``) while its charge deposit keeps the true charge ``q`` changes the
     dispersion relation to ``1 = omega_b^2 / (omega - k v0)^2 - omega_b^2 / (omega + k v0)^2``;
     the measured growth rate matches that mutated discrete root within its tolerance
     and lies outside the unmutated tolerance.
   - Monkeypatching ``gather_fields`` to halve ``E_z`` halves ``omega_b^2`` to give
     continuum growth rate ``(1 / sqrt(2)) * sqrt(sqrt(7) - 2.5) * omega_{b,rel}``
     at ``k1 * v0 = (sqrt(3) / 2) * omega_{b,rel}``; the measured growth rate matches
     the corresponding discrete root within its tolerance and lies outside
     the unmutated tolerance.
   - Input check (non-neutral periodic load): on a periodic domain ``sum(S_tilde d_tilde) = 0``
     identically, so a non-neutral load raises ``ValueError`` in ``run_two_stream_simulation``
     and produces the documented compensating point-charge jump artefact at node ``0``
     when inferred via ``q_bg = div_d0 - q0`` in ``ParticleSolver.attach()``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Set OMP_NUM_THREADS=1 before numpy/scipy are imported so worker processes and
# OpenMP-linked BLAS/LAPACK routines do not oversubscribe CPU cores.
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest
from scipy.constants import c, e, epsilon_0, m_e

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_DISP_PATH = Path(__file__).resolve().parent / "_two_stream_dispersion.py"
_DISP_SPEC = importlib.util.spec_from_file_location(
    "_two_stream_dispersion", _DISP_PATH
)
assert _DISP_SPEC is not None and _DISP_SPEC.loader is not None
_disp_mod = importlib.util.module_from_spec(_DISP_SPEC)
sys.modules[_DISP_SPEC.name] = _disp_mod
_DISP_SPEC.loader.exec_module(_disp_mod)

ALPHA_SAT = _disp_mod.ALPHA_SAT
HORIZON_SAFETY_FACTOR = _disp_mod.HORIZON_SAFETY_FACTOR
T_HORIZON_WB = _disp_mod.T_HORIZON_WB
TRANSIENT_TOL = _disp_mod.TRANSIENT_TOL
two_stream_cfl_dt = _disp_mod.two_stream_cfl_dt
two_stream_continuum_growth_rate = _disp_mod.two_stream_continuum_growth_rate
two_stream_continuum_root = _disp_mod.two_stream_continuum_root
two_stream_discrete_complex_roots = _disp_mod.two_stream_discrete_complex_roots
two_stream_discrete_growth_rate = _disp_mod.two_stream_discrete_growth_rate
two_stream_discrete_root = _disp_mod.two_stream_discrete_root
two_stream_discrete_uhp_root_count = _disp_mod.two_stream_discrete_uhp_root_count
two_stream_lattice_alias_bound = _disp_mod.two_stream_lattice_alias_bound
two_stream_nonlinear_bound = _disp_mod.two_stream_nonlinear_bound
two_stream_seed_mode_shares = _disp_mod.two_stream_seed_mode_shares
two_stream_stable_beat_envelope_bound = _disp_mod.two_stream_stable_beat_envelope_bound
two_stream_window_bounds_wb = _disp_mod.two_stream_window_bounds_wb
from wakis import GridFIT3D, SolverFIT3D
from wakis.particles import (
    ParticleSolver,
    Species,
    cold_plasma_dispersion_discrete,
    deposit_charge,
)


def run_two_stream_simulation(
    Nz: int = 16,
    n_ppc_z: int = 32,
    v0_over_c: float = 0.06,
    eps: float = 1e-8,
    t_start_wb: float | None = None,
    t_end_wb: float | None = None,
    alpha_sat: float = ALPHA_SAT,
    bg_scale: float = 1.0,
    s_plus: float = 1.0,
    s_minus: float = 1.0,
    density_ratio_minus: float = 1.0,
    sigma_v_frac: float = 0.0,
    rng_seed: int = 0,
    seed_harmonic: int = 1,
) -> dict[str, float | str | np.ndarray]:
    """Run a quasi-1D periodic two-stream simulation to its per-case derived horizon and fit the linear growth rate.

    The linear fit window ``[t_start, t_end]`` is determined strictly per case from the
    analytic ``omega_+`` share of the position-only seed via
    ``(a_osc / a_grow) * exp(-gamma_disc(case) * t_start) <= TRANSIENT_TOL`` and the
    trapping-fraction end rule ``|E_{z,m}(t)| <= ALPHA_SAT * E_trap`` (with analytical
    prediction ``t_end_pred = ln(3 * ALPHA_SAT / (8 * eps * a_grow)) / gamma_disc(case)``
    and per-case horizon ``1.5 * t_end_pred(case)`` for unstable modes, or
    ``T_stable = 3 * pi / omega_-(k_2)`` for the stable ``k_2 = 2 * k_1`` mode),
    with no window selection by trial runs and no ``gamma_fit = 0.0`` fallback (raising
    ``AssertionError("no linear window")`` if fewer than 5 points lie in the window).

    Parameters
    ----------
    Nz : int, optional
        Number of longitudinal cells along ``z``. Default is ``16``.
    n_ppc_z : int, optional
        Macroparticles per ``z``-cell per beam. Default is ``32``.
    v0_over_c : float, optional
        Normalized beam speed ``v0 / c`` (kept ``<= 0.1``). Default is ``0.06``.
    eps : float, optional
        Relative initial sinusoidal density perturbation amplitude ``delta_n / n_b`` [-].
        Default is ``1e-8``.
    t_start_wb : float or None, optional
        Dimensionless start of the linear fit window ``t_start * omega_b`` [-].
        If ``None`` (default), computed per case from ``two_stream_window_bounds_wb``
        using ``gamma_disc(case)``.
    t_end_wb : float or None, optional
        Dimensionless integration horizon ``t_horizon * omega_b`` [-].
        If ``None`` (default), set per case to ``1.5 * omega_b * t_end_pred(case)`` for
        unstable modes and ``3 * pi * omega_b / omega_-(k_seed)`` for stable modes.
    alpha_sat : float, optional
        Trapping-field fraction ``ALPHA_SAT`` defining the upper end of the linear
        fit window. Default is ``ALPHA_SAT`` (``1e-2``).
    bg_scale : float, optional
        Multiplier on the immobile neutralising background charge; must equal ``1.0``
        on a periodic domain (non-neutral loads raise ``ValueError``). Default is ``1.0``.
    s_plus : float, optional
        Dimensionless susceptibility weight of the ``+v0`` beam in the dispersion
        prediction [-]. Default is ``1.0``.
    s_minus : float, optional
        Dimensionless susceptibility weight of the ``-v0`` beam in the dispersion
        prediction [-]. Default is ``1.0``.
    density_ratio_minus : float, optional
        Unperturbed number-density ratio ``n_- / n_+`` of the ``-v0`` beam relative
        to the ``+v0`` beam [-]. Default is ``1.0``.
    sigma_v_frac : float, optional
        Optional statistical thermal/sampling velocity jitter ``sigma_v / v0`` per
        macroparticle for ``1 / sqrt(N_ppc)`` noise convergence tests. Default is ``0.0``.
    rng_seed : int, optional
        Deterministic random seed used when ``sigma_v_frac > 0``. Default is ``0``.
    seed_harmonic : int, optional
        Harmonic ``m`` of ``k1`` to seed (``m = 1`` for the fastest-growing unstable mode
        ``k1 * v0 = (sqrt(3)/2) * omega_{b,rel}``; ``m = 2`` for the stable mode
        ``k2 * v0 = sqrt(3) * omega_{b,rel} > sqrt(2) * omega_{b,rel}``). Default is ``1``.

    Returns
    -------
    dict
        Dictionary containing fitted growth rate, theoretical predictions, derived
        tolerance components, Gauss residual, window bounds, window end rule, and mode
        amplitude histories.
    """
    if abs(float(bg_scale) - 1.0) > 1e-12:
        raise ValueError(
            f"Non-neutral periodic load (bg_scale={bg_scale}): on a periodic domain "
            "sum(S_tilde d_tilde) == 0 identically, so net charge must vanish."
        )

    v0 = float(v0_over_c) * c
    gamma0 = 1.0 / float(np.sqrt(1.0 - (v0 / c) ** 2))
    omega_b = 2.0 * np.pi * 1.0e9
    wb_rel = omega_b * (gamma0 ** (-1.5))

    # Fundamental wavenumber k1 sits at k1*v0 = (sqrt(3)/2)*wb_rel on the periodic box
    # Lz = 2*pi/k1 with Nz cells (cell size dz = Lz / Nz). Keeping the full box Lz = 2*pi/k1
    # when seed_harmonic == 2 retains the unseeded unstable fundamental k1 in the domain so
    # its amplitude |E_{k1}(t)| can be monitored directly against |E_{k2}(0)|.
    k1 = (np.sqrt(3.0) / 2.0) * wb_rel / v0
    Lz = 2.0 * np.pi / k1
    dz = Lz / float(Nz)

    Nx, Ny = 2, 2
    Lx, Ly = Nx * dz * 8.0, Ny * dz * 8.0
    grid = GridFIT3D(0.0, Lx, 0.0, Ly, 0.0, Lz, Nx, Ny, int(Nz), verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=0.8,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        verbose=0,
    )
    dt = float(solver.dt)

    nb = epsilon_0 * m_e * (omega_b**2) / (e**2)
    vol = Lx * Ly * Lz
    n_z = int(Nz * n_ppc_z)
    z0 = (np.arange(n_z, dtype=np.float64) + 0.5) * (Lz / n_z)

    k_seed = float(seed_harmonic) * k1
    if eps > 0.0:
        z_plus = z0 - (eps / k_seed) * np.sin(k_seed * z0 + np.pi / 3.0)
        z_minus = z0 - (eps / k_seed) * np.sin(k_seed * z0 - np.pi / 3.0)
    else:
        z_plus = z0.copy()
        z_minus = z0.copy()
    z_plus %= Lz
    z_minus %= Lz

    if sigma_v_frac > 0.0:
        rng = np.random.default_rng(rng_seed)
        vz_plus = v0 + (sigma_v_frac * v0) * rng.standard_normal(n_z)
        vz_minus = -v0 + (sigma_v_frac * v0) * rng.standard_normal(n_z)
    else:
        vz_plus = v0
        vz_minus = -v0

    # Placing particles at (0.5 * dx, 0.5 * dy) on the 2x2 periodic transverse grid
    # gives identical 0.25 CIC weights on all 4 transverse nodes.
    x0 = np.full(n_z, 0.5 * (Lx / Nx), dtype=np.float64)
    y0 = np.full(n_z, 0.5 * (Ly / Ny), dtype=np.float64)
    w_macro_plus = (nb * vol) / float(n_z)
    w_macro_minus = float(density_ratio_minus) * w_macro_plus

    sp_plus = Species(name="beam_plus", charge=-e, mass=m_e)
    sp_minus = Species(name="beam_minus", charge=-e, mass=m_e)
    sp_plus.add_particles(
        x=x0, y=y0, z=z_plus, vx=0.0, vy=0.0, vz=vz_plus, w=w_macro_plus
    )
    sp_minus.add_particles(
        x=x0, y=y0, z=z_minus, vx=0.0, vy=0.0, vz=vz_minus, w=w_macro_minus
    )

    # Deposit initial beam node charge and solve 1D discrete Gauss's law for E_z(0)
    q_nodes = np.zeros((Nx, Ny, int(Nz)), dtype=np.float64, order="F")
    for sp in (sp_plus, sp_minus):
        deposit_charge(
            sp.x / (Lx / Nx),
            sp.y / (Ly / Ny),
            sp.z / dz,
            sp.charge * sp.w,
            Nx,
            Ny,
            int(Nz),
            order=1,
            periodic=(True, True, True),
            q_nodes=q_nodes,
        )
    Az = (Lx / Nx) * (Ly / Ny)
    q_bg_neutral_node = (1.0 + float(density_ratio_minus)) * e * nb * Az * dz
    q_net = q_nodes + q_bg_neutral_node
    Ez_3d = np.cumsum(q_net, axis=2) / (epsilon_0 * Az)
    Ez_3d -= np.mean(Ez_3d, axis=2, keepdims=True)
    solver.E[:, :, :, "z"] = Ez_3d

    ps = ParticleSolver(
        solver,
        species=[sp_plus, sp_minus],
        pusher="boris",
        shape=1,
        gather="galerkin",
        sort_every=0,
    ).attach()

    gauss_init = ps.gauss_residual()
    q_bg_interior_ratio = float(
        np.mean(solver._particles_q_bg[:, :, 1:]) / q_bg_neutral_node
    )

    gamma_cont = two_stream_continuum_growth_rate(
        omega_b, k_seed, v0, gamma0=gamma0, s_plus=s_plus, s_minus=s_minus
    )
    gamma_cont_nr = two_stream_continuum_growth_rate(
        omega_b, k_seed, v0, gamma0=1.0, s_plus=s_plus, s_minus=s_minus
    )
    gamma_disc = two_stream_discrete_growth_rate(
        omega_b,
        k_seed,
        v0,
        dt,
        dz,
        order=1,
        gather="galerkin",
        gamma0=gamma0,
        s_plus=s_plus,
        s_minus=s_minus,
    )

    # Per-case window start t_start_wb, analytical saturation end t_end_pred_wb,
    # and per-case horizon t_horizon_wb = 1.5 * t_end_pred_wb (for unstable cases)
    # or t_stable_wb = 3 * pi * omega_b / omega_-(k_seed) (for stable cases).
    if gamma_disc > 0.0:
        win_bounds = two_stream_window_bounds_wb(
            omega_b,
            gamma_disc,
            gamma0=gamma0,
            eps=max(float(eps), 1e-30),
            alpha_sat=alpha_sat,
            transient_tol=TRANSIENT_TOL,
            s_plus=s_plus,
            s_minus=s_minus,
            density_ratio_minus=density_ratio_minus,
            kv_over_wb_rel=float(k_seed * v0 / wb_rel),
        )
        t_start_case_wb = float(win_bounds["t_start_wb"])
        t_end_pred_wb = float(win_bounds["t_end_pred_wb"])
        t_horizon_wb = float(
            win_bounds["t_horizon_wb"] if t_end_wb is None else t_end_wb
        )
        a_grow = float(win_bounds["a_grow"])
        a_osc = float(win_bounds["a_osc"])
        a_decay = float(win_bounds["a_decay"])
        n_steps = int(t_horizon_wb / (omega_b * dt))
    else:
        beat_bounds = two_stream_stable_beat_envelope_bound(
            omega_b,
            k_seed,
            v0,
            dt,
            dz,
            order=1,
            gather="galerkin",
            gamma0=gamma0,
        )
        t_start_case_wb = 0.0
        t_end_pred_wb = float(beat_bounds["t_stable_wb"])
        t_horizon_wb = float(
            beat_bounds["t_stable_wb"] if t_end_wb is None else t_end_wb
        )
        a_grow = 0.0
        a_osc = float(beat_bounds["a_max_over_e0"])
        a_decay = 0.0
        n_steps = 3 * int(np.ceil(t_horizon_wb / (3.0 * omega_b * dt)))

    t_start_eff_wb = float(t_start_case_wb if t_start_wb is None else t_start_wb)

    z_edges = (np.arange(int(Nz), dtype=np.float64) + 0.5) * dz
    exp_k1 = np.exp(-1j * k1 * z_edges)
    exp_k2 = np.exp(-1j * (2.0 * k1) * z_edges)
    exp_k3 = np.exp(-1j * (3.0 * k1) * z_edges)

    t_arr = np.zeros(n_steps, dtype=np.float64)
    amp_k1 = np.zeros(n_steps, dtype=np.float64)
    amp_k2 = np.zeros(n_steps, dtype=np.float64)
    amp_k3 = np.zeros(n_steps, dtype=np.float64)

    Ez_init_1d = np.mean(solver.E[:, :, :, "z"], axis=(0, 1))
    amp_k1_init = float((2.0 / float(Nz)) * np.abs(np.sum(Ez_init_1d * exp_k1)))

    for s in range(n_steps):
        solver.one_step()
        t_arr[s] = (s + 1) * dt
        Ez_1d = np.mean(solver.E[:, :, :, "z"], axis=(0, 1))
        amp_k1[s] = (2.0 / float(Nz)) * np.abs(np.sum(Ez_1d * exp_k1))
        amp_k2[s] = (2.0 / float(Nz)) * np.abs(np.sum(Ez_1d * exp_k2))
        amp_k3[s] = (2.0 / float(Nz)) * np.abs(np.sum(Ez_1d * exp_k3))

    gauss_final = ps.gauss_residual()

    # Wave-particle trapping saturation amplitude E_trap = gamma0^3 * m_e * k1 * v0^2 / (2 * e)
    # Related to reference initial amplitude via E_0 / E_trap = (8 / 3) * eps
    # (gamma0^3 cancels since k1 * v0 = (sqrt(3) / 2) * wb_rel).
    e_trap = 0.5 * (gamma0**3) * m_e * k1 * (v0**2) / e
    amp_sat_limit = float(alpha_sat) * e_trap

    amp_seeded = amp_k1 if int(seed_harmonic) == 1 else amp_k2
    after_start = t_arr * omega_b >= t_start_eff_wb
    exceed_idx = np.flatnonzero(after_start & (amp_seeded > amp_sat_limit))
    if exceed_idx.size > 0:
        t_end_idx = int(exceed_idx[0])
        window_end_rule = "alpha_sat"
    else:
        t_end_idx = n_steps
        window_end_rule = "horizon"
    win_mask = after_start & (np.arange(n_steps) < t_end_idx)
    if np.count_nonzero(win_mask) < 5 or not np.all(amp_seeded[win_mask] > 0.0):
        raise AssertionError("no linear window")

    coeffs, cov = np.polyfit(
        t_arr[win_mask], np.log(amp_seeded[win_mask]), 1, cov=True
    )
    gamma_fit = float(coeffs[0])
    sigma_fit = float(np.sqrt(cov[0, 0]))
    dt_win = float(t_arr[win_mask][-1] - t_arr[win_mask][0])
    amp_end = float(amp_seeded[win_mask][-1])
    t_win_start_wb = float(t_arr[win_mask][0] * omega_b)
    t_win_end_wb = float(t_arr[win_mask][-1] * omega_b)

    # Also compute empirical half-alpha_sat shift when at least 5 points exist below 0.5 * alpha_sat
    exceed_half = np.flatnonzero(after_start & (amp_seeded > 0.5 * amp_sat_limit))
    t_half_idx = int(exceed_half[0]) if exceed_half.size > 0 else n_steps
    win_half = after_start & (np.arange(n_steps) < t_half_idx)
    if np.count_nonzero(win_half) >= 5 and np.all(amp_seeded[win_half] > 0.0):
        gamma_fit_half = float(
            np.polyfit(t_arr[win_half], np.log(amp_seeded[win_half]), 1)[0]
        )
        delta_gamma_nl_emp = abs(gamma_fit - gamma_fit_half)
    else:
        delta_gamma_nl_emp = 0.0

    # Tolerance components:
    # 1. 3 * sigma_fit: empirical (single-run OLS SE) from one run with autocorrelated
    #    residual omega_+ beat modulation;
    # 2. delta_gamma_nl: order-of-magnitude quadratic (3/16)*|gamma_disc|*(|E_1|/E_trap)^2
    #    weakly nonlinear amplitude detuning bound;
    # 3. delta_gamma_lat: heuristic particle-lattice alias bound.
    delta_gamma_nl = two_stream_nonlinear_bound(gamma_disc, amp_end, e_trap)
    delta_gamma_lat = two_stream_lattice_alias_bound(
        gamma_disc, n_particles_beam=int(Nz * n_ppc_z), order=1, gather="galerkin"
    )
    tol_gamma = 3.0 * sigma_fit + delta_gamma_nl + delta_gamma_lat

    return {
        "gamma_fit": gamma_fit,
        "sigma_fit": sigma_fit,
        "gamma_disc": gamma_disc,
        "gamma_cont": gamma_cont,
        "gamma_cont_nr": gamma_cont_nr,
        "omega_b": omega_b,
        "wb_rel": wb_rel,
        "delta_gamma_nl": delta_gamma_nl,
        "delta_gamma_nl_emp": delta_gamma_nl_emp,
        "delta_gamma_lat": delta_gamma_lat,
        "tol_gamma": tol_gamma,
        "gauss_init": gauss_init,
        "gauss_final": gauss_final,
        "q_bg_interior_ratio": q_bg_interior_ratio,
        "e_trap": e_trap,
        "amp_k1_init": amp_k1_init,
        "amp_k1": amp_k1,
        "amp_k2": amp_k2,
        "amp_k3": amp_k3,
        "t_arr": t_arr,
        "t_start_case_wb": t_start_case_wb,
        "t_end_pred_wb": t_end_pred_wb,
        "t_horizon_wb": t_horizon_wb,
        "window_end_rule": window_end_rule,
        "a_grow": a_grow,
        "a_osc": a_osc,
        "a_decay": a_decay,
        "t_win_start_wb": t_win_start_wb,
        "t_win_end_wb": t_win_end_wb,
        "dt_win": dt_win,
        "k1": k1,
        "v0": v0,
        "gamma0": gamma0,
        "dt": dt,
        "dz": dz,
    }


def test_dispersion_identities_and_stationary_limit():
    """Verify analytical and discrete dispersion relations and exact reduction to ``cold_plasma_dispersion_discrete``."""
    omega_b = 2.0 * np.pi * 1.0e9
    v0 = 0.06 * c
    gamma0 = 1.0 / np.sqrt(1.0 - (v0 / c) ** 2)

    # 1. Maximum growth rate gamma_max = omega_b / 2 at k * v0 = (sqrt(3) / 2) * omega_b
    k_opt = (np.sqrt(3.0) / 2.0) * omega_b / v0
    g_max = two_stream_continuum_growth_rate(omega_b, k_opt, v0, gamma0=1.0)
    assert abs(g_max - 0.5 * omega_b) / omega_b <= 1e-15

    # Derivative d(gamma)/dk vanishes at k_opt: nearby wavenumbers have strictly smaller growth rate
    for frac in (0.95, 1.05):
        assert (
            two_stream_continuum_growth_rate(omega_b, frac * k_opt, v0, gamma0=1.0)
            < g_max
        )

    # 2. Marginal stability cutoff at k * v0 = sqrt(2) * omega_b
    k_crit = np.sqrt(2.0) * omega_b / v0
    assert (
        two_stream_continuum_growth_rate(omega_b, 0.999 * k_crit, v0, gamma0=1.0) > 0.0
    )
    assert two_stream_continuum_growth_rate(omega_b, k_crit, v0, gamma0=1.0) == 0.0
    assert (
        two_stream_continuum_growth_rate(omega_b, 1.2 * k_crit, v0, gamma0=1.0) == 0.0
    )

    # 3. Relativistic scaling omega_{b,rel} = gamma0^(-3/2) * omega_b; for v0 = 0.06c,
    #    1 - gamma0^(-3/2) = (3/4)*(v0/c)^2 + O((v0/c)^4)
    g_max_rel = two_stream_continuum_growth_rate(
        omega_b, k_opt * (gamma0 ** (-1.5)), v0, gamma0=gamma0
    )
    rel_neglect = 1.0 - (g_max_rel / g_max)
    expected_neglect = 1.0 - (gamma0 ** (-1.5))
    assert abs(rel_neglect - expected_neglect) <= 1e-15
    assert rel_neglect < 3.0e-3

    # 4. In the stationary limit v0 -> 0, two cold electron beams of plasma frequency omega_b
    #    form a single cold plasma of frequency omega_p0 = sqrt(2) * omega_b, and the discrete
    #    two-stream dispersion reduces identically to cold_plasma_dispersion_discrete.
    dz = (2.0 * np.pi / k_opt) / 16.0
    dt = 0.5 * dz / c
    omega_p0 = np.sqrt(2.0) * omega_b
    for order in (1, 2):
        for gather in ("galerkin", "momentum"):
            wp_disc = cold_plasma_dispersion_discrete(
                omega_p0, dt, dz, k_opt, order=order, gather=gather
            )
            # Evaluate the discrete two-stream susceptibility sum at v0 = 0 and omega = wp_disc
            is_gal = gather == "galerkin"
            p_g = (order - 1) if is_gal else order
            m_arr = np.arange(-80, 81, dtype=np.float64)
            km = k_opt + 2.0 * np.pi * m_arr / dz
            arg = 0.5 * km * dz
            sinc_w = np.where(np.abs(arg) < 1e-14, 1.0, np.sin(arg) / arg) ** (
                order + p_g + 1
            )
            omega_t = (2.0 / dt) * np.sin(0.5 * wp_disc * dt)
            # Two beams of omega_b at v0 = 0: sum_beams (omega_b^2 / omega_t^2) * sum_m sinc_w.
            # For p=1 Galerkin (exponent 2), truncating the infinite alias series at |m| <= M=80
            # leaves an analytical Euler-Maclaurin tail 2 * sin^2(k*dz/2) / (pi^2 * (M + 0.5)).
            exp_tot = order + p_g + 1
            tail = (
                2.0 * (np.sin(0.5 * k_opt * dz) ** 2) / ((np.pi**2) * 80.5)
                if exp_tot == 2
                else 0.0
            )
            lhs = (
                2.0 * (omega_b**2) * (float(np.sum(sinc_w)) + tail) / (omega_t**2)
            )
            assert abs(lhs - 1.0) <= 1e-7

    # 5. Analytic normal-mode shares of the position-only seed at k_opt * v0 = (sqrt(3)/2) * omega_b:
    #    a_grow = 1/2, a_decay = 1/4, and non-decaying omega_+ envelope share a_osc = 3 / sqrt(10).
    shares_sym = two_stream_seed_mode_shares(s_plus=1.0, s_minus=1.0, density_ratio_minus=1.0)
    assert abs(shares_sym["a_grow"] - 0.5) <= 1e-14
    assert abs(shares_sym["a_decay"] - 0.25) <= 1e-14
    assert abs(shares_sym["a_osc"] - 3.0 / np.sqrt(10.0)) <= 1e-14

    # Verify E_0 / E_trap == (8 / 3) * eps exactly (gamma0^3 cancels since k1_rel is chosen from omega_{b,rel})
    k1_rel = k_opt * (gamma0 ** (-1.5))
    nb_rel = epsilon_0 * m_e * (omega_b**2) / (e**2)
    eps_test = 1.0e-8
    e0_rel = e * nb_rel * eps_test / (epsilon_0 * k1_rel)
    e_trap_rel = 0.5 * (gamma0**3) * m_e * k1_rel * (v0**2) / e
    assert abs((e0_rel / e_trap_rel) - (8.0 / 3.0) * eps_test) / eps_test <= 1e-15

    # 6. Upper-half-plane root count by the argument principle (winding number of the
    #    pole-removed discrete dispersion function around the Nyquist-bounded UHP contour)
    #    plus dominant-alias (|m| <= 4; not exhaustive) root search at k1 (unstable) vs k2 = 2 * k1 (stable):
    dz_rel = (2.0 * np.pi / k1_rel) / 16.0
    dt_rel = two_stream_cfl_dt(dz_rel)
    uhp_count_k1 = two_stream_discrete_uhp_root_count(
        omega_b, k1_rel, v0, dt_rel, dz_rel, order=1, gather="galerkin", gamma0=gamma0
    )
    uhp_count_k2 = two_stream_discrete_uhp_root_count(
        omega_b,
        2.0 * k1_rel,
        v0,
        dt_rel,
        dz_rel,
        order=1,
        gather="galerkin",
        gamma0=gamma0,
    )
    assert uhp_count_k1 == 1
    assert uhp_count_k2 == 0

    roots_k1 = two_stream_discrete_complex_roots(
        omega_b, k1_rel, v0, dt_rel, dz_rel, order=1, gather="galerkin", gamma0=gamma0
    )
    roots_k2 = two_stream_discrete_complex_roots(
        omega_b,
        2.0 * k1_rel,
        v0,
        dt_rel,
        dz_rel,
        order=1,
        gather="galerkin",
        gamma0=gamma0,
    )
    assert max(float(np.imag(r)) for r in roots_k1) > 0.4 * omega_b
    assert max(float(np.imag(r)) for r in roots_k2) == 0.0


def _init_worker_omp() -> None:
    """Set OMP_NUM_THREADS=1 in worker processes."""
    os.environ["OMP_NUM_THREADS"] = "1"


def _run_ppc_seed_pair(args: tuple[int, int, int]) -> tuple[float, float, float, float]:
    """Worker for parallel 1/sqrt(N_ppc) ensemble runs (one process per case, 1 OMP thread)."""
    seed, n_ppc_low, n_ppc_high = args
    rl = run_two_stream_simulation(
        Nz=16,
        n_ppc_z=n_ppc_low,
        sigma_v_frac=1.5e-8,
        rng_seed=seed,
    )
    rh = run_two_stream_simulation(
        Nz=16,
        n_ppc_z=n_ppc_high,
        sigma_v_frac=1.5e-8,
        rng_seed=seed,
    )
    assert rl["window_end_rule"] == "alpha_sat"
    assert rh["window_end_rule"] == "alpha_sat"
    assert abs(float(rl["t_horizon_wb"]) - 1.5 * float(rl["t_end_pred_wb"])) <= 1e-12
    assert abs(float(rh["t_horizon_wb"]) - 1.5 * float(rh["t_end_pred_wb"])) <= 1e-12
    assert float(rl["t_horizon_wb"]) > float(rl["t_win_end_wb"])
    assert float(rh["t_horizon_wb"]) > float(rh["t_win_end_wb"])
    err_l = (float(rl["gamma_fit"]) - float(rl["gamma_disc"])) / float(rl["omega_b"])
    err_h = (float(rh["gamma_fit"]) - float(rh["gamma_disc"])) / float(rh["omega_b"])
    mask_l = np.asarray(rl["t_arr"]) * float(rl["omega_b"]) <= float(rl["t_win_end_wb"])
    mask_h = np.asarray(rh["t_arr"]) * float(rh["omega_b"]) <= float(rh["t_win_end_wb"])
    noise_l = float(np.sqrt(np.mean(np.asarray(rl["amp_k3"])[mask_l] ** 2)))
    noise_h = float(np.sqrt(np.mean(np.asarray(rh["amp_k3"])[mask_h] ** 2)))
    return err_l, err_h, noise_l, noise_h


def test_two_stream_linear_growth_and_stable_mode():
    """Check (a): linear growth rate matches discrete prediction within derived tolerance, stable mode does not grow, and asymmetric density root holds."""
    # Note: Nz = 16 resolves the two-beam plasma wavelength
    # lambda_p = 2*pi*u0 / (sqrt(2)*omega_b) with Nz * sqrt(3) / (2 * sqrt(2)) cells,
    # deliberately below the >= 20 cells/lambda_p production rule in docs/physicsguide_sections/particles_two_stream.md
    # so that the discrete grid correction |gamma_cont - gamma_disc| is large enough to be
    # cleanly separated from tol_gamma.
    n_ppc_z = 32
    res = run_two_stream_simulation(Nz=16, n_ppc_z=n_ppc_z)

    omega_b = float(res["omega_b"])
    gamma_fit = float(res["gamma_fit"])
    gamma_disc = float(res["gamma_disc"])
    gamma_cont = float(res["gamma_cont"])
    tol_gamma = float(res["tol_gamma"])

    err_disc = abs(gamma_fit - gamma_disc)
    err_cont_vs_disc = abs(gamma_cont - gamma_disc)

    print(
        f"\n[V4a] win=[{res['t_win_start_wb']:.2f}, {res['t_win_end_wb']:.2f}] "
        f"(pred_end={res['t_end_pred_wb']:.2f}, horizon={res['t_horizon_wb']:.2f}, end_rule={res['window_end_rule']}), "
        f"a_grow={res['a_grow']:.4f}, a_osc={res['a_osc']:.4f}, "
        f"gamma_fit/wb={gamma_fit / omega_b:.6f}, "
        f"gamma_disc/wb={gamma_disc / omega_b:.6f}, "
        f"gamma_cont/wb={gamma_cont / omega_b:.6f}, "
        f"|fit-disc|/wb={err_disc / omega_b:.6f}, "
        f"tol/wb={tol_gamma / omega_b:.6f} "
        f"(3sig_emp(single-run OLS SE)={3.0 * float(res['sigma_fit']) / omega_b:.6e}, "
        f"nl_quad_oom={float(res['delta_gamma_nl']) / omega_b:.6e}, "
        f"nl_emp={float(res['delta_gamma_nl_emp']) / omega_b:.6e}, "
        f"lat_heur={float(res['delta_gamma_lat']) / omega_b:.6e}), "
        f"|disc-cont|/wb={err_cont_vs_disc / omega_b:.6f}, "
        f"gauss={res['gauss_final']:.2e}"
    )

    # Verify that the CFL dt formula in two_stream_suite_horizon_wb equals solver.dt
    dt_cfl = two_stream_cfl_dt(float(res["dz"]), cfln=0.8, transverse_aspect=8.0)
    assert abs(dt_cfl - float(res["dt"])) <= 1e-15 * float(res["dt"])

    # Per-case horizon is 1.5 * t_end_pred(case), exceeds actual t_end, and window ends by alpha_sat
    assert abs(float(res["t_horizon_wb"]) - 1.5 * float(res["t_end_pred_wb"])) <= 1e-12
    assert float(res["t_horizon_wb"]) > float(res["t_end_pred_wb"])
    assert float(res["t_horizon_wb"]) > float(res["t_win_end_wb"])
    assert res["window_end_rule"] == "alpha_sat"

    # 1. Discrete Gauss's law: gauss_init is zero by construction
    #    (an identity from q_bg = div_d0 - q0 in ParticleSolver.attach());
    #    gauss_final is the real conservation check after time-stepping.
    assert float(res["gauss_init"]) <= 1e-12
    assert float(res["gauss_final"]) <= 1e-12
    assert abs(float(res["q_bg_interior_ratio"]) - 1.0) <= 1e-12

    # 2. Tolerance separates discrete from continuum dispersion
    assert tol_gamma < err_cont_vs_disc
    assert err_disc <= tol_gamma

    # 3. Dedicated stable-wavenumber check at k2 = 2 * k1 (k2 * v0 = sqrt(3)*wb_rel > sqrt(2)*wb_rel):
    #    (i) Argument-principle winding number of the pole-removed discrete dispersion function
    #        around the Nyquist-bounded upper-half-plane contour confirms 0 roots with Im(omega) > 0
    #        at k2 and 1 root at k1 (supplemented by the dominant-alias |m| <= 4 search, not exhaustive).
    #    (ii) The stable run uses its own derived horizon T_stable = 3 * pi / omega_-(k2), the
    #         shortest horizon covering one slow half-period pi / omega_-(k2) in each of the three
    #         equal thirds of the run (and (3/2) * (omega_+(k2)/omega_-(k2) - 1) beat periods
    #         2 * pi / (omega_+(k2) - omega_-(k2))). Over [0, T_stable], the unseeded unstable
    #         fundamental mode k1 stays below 1e-3 * |E_{k2}(0)| so its second harmonic cannot
    #         contaminate the k2 envelope test.
    #    (iii) Compare the last-third vs first-third beat envelope maxima M_3 and M_1 within
    #          tol_env = (A_- / A_max) * (1 - cos(Delta_theta)) with Delta_theta = pi * omega_- / omega_+
    #          (since the fast extremum of the required opposite sign recurs every 2*pi / omega_+),
    #          and report the minimum detectable growth rate gamma_min = ln(1 + tol_env) / ((2/3) * T_stable).
    res_stable = run_two_stream_simulation(
        Nz=16,
        n_ppc_z=16,
        seed_harmonic=2,
    )
    k1_s = float(res_stable["k1"])
    k2 = 2.0 * k1_s
    uhp_count_k1 = two_stream_discrete_uhp_root_count(
        omega_b,
        k1_s,
        float(res_stable["v0"]),
        float(res_stable["dt"]),
        float(res_stable["dz"]),
        order=1,
        gather="galerkin",
        gamma0=float(res_stable["gamma0"]),
    )
    uhp_count_k2 = two_stream_discrete_uhp_root_count(
        omega_b,
        k2,
        float(res_stable["v0"]),
        float(res_stable["dt"]),
        float(res_stable["dz"]),
        order=1,
        gather="galerkin",
        gamma0=float(res_stable["gamma0"]),
    )
    assert uhp_count_k1 == 1
    assert uhp_count_k2 == 0

    roots_k2 = two_stream_discrete_complex_roots(
        omega_b,
        k2,
        float(res_stable["v0"]),
        float(res_stable["dt"]),
        float(res_stable["dz"]),
        order=1,
        gather="galerkin",
        gamma0=float(res_stable["gamma0"]),
        n_alias_roots=4,
    )
    max_imag_k2 = max(float(np.imag(r)) for r in roots_k2)
    assert max_imag_k2 == 0.0

    beat_info = two_stream_stable_beat_envelope_bound(
        omega_b,
        k2,
        float(res_stable["v0"]),
        float(res_stable["dt"]),
        float(res_stable["dz"]),
        order=1,
        gather="galerkin",
        gamma0=float(res_stable["gamma0"]),
    )
    amp_k1_s = np.asarray(res_stable["amp_k1"])
    amp_k2_s = np.asarray(res_stable["amp_k2"])
    t_arr_s = np.asarray(res_stable["t_arr"])
    assert float(res_stable["gauss_final"]) <= 1e-12
    assert res_stable["window_end_rule"] == "horizon"
    assert (
        abs(float(res_stable["t_horizon_wb"]) - float(beat_info["t_stable_wb"]))
        <= 1e-12
    )

    n_s = int(amp_k2_s.size)
    n_third = n_s // 3
    om_m_k2 = float(beat_info["omega_minus_disc"])
    om_p_k2 = float(beat_info["omega_plus_disc"])
    assert n_third * float(res_stable["dt"]) >= np.pi / om_m_k2
    n_beat_periods = float(t_arr_s[-1]) * (om_p_k2 - om_m_k2) / (2.0 * np.pi)
    assert n_beat_periods > 4.0

    # Guard against second-harmonic contamination from unseeded unstable mode k1
    max_k1_over_k2_init = float(np.max(amp_k1_s) / amp_k2_s[0])
    assert max_k1_over_k2_init <= 1.0e-3

    env_max_first_third = float(np.max(amp_k2_s[:n_third]))
    env_max_last_third = float(np.max(amp_k2_s[n_s - n_third :]))
    env_rel_diff = abs(env_max_last_third - env_max_first_third) / max(
        env_max_first_third, env_max_last_third
    )
    delta_theta = float(beat_info["delta_theta"])
    tol_env_rel = float(beat_info["tol_env_rel"])
    dt_thirds_s = (2.0 / 3.0) * float(t_arr_s[-1])
    gamma_min_detectable = float(np.log(1.0 + tol_env_rel) / dt_thirds_s)
    beat_supremum_bound = float(beat_info["a_max_over_e0"]) * float(amp_k2_s[0])

    print(
        f"[V4a-stable] horizon={res_stable['t_horizon_wb']:.2f} (end_rule={res_stable['window_end_rule']}, "
        f"n_beat_periods={n_beat_periods:.2f}), "
        f"uhp_roots(k1={uhp_count_k1}, k2={uhp_count_k2}), "
        f"dominant_alias_|m|<=4_roots_k2={len(roots_k2)} (not exhaustive, max_Im/wb={max_imag_k2 / omega_b:.3e}), "
        f"max_Ek1/Ek2(0)={max_k1_over_k2_init:.3e} <= 1.0e-3, "
        f"omega_-(k2)/wb={om_m_k2 / omega_b:.6f}, "
        f"omega_+(k2)/wb={om_p_k2 / omega_b:.6f}, "
        f"delta_theta={delta_theta:.4f}, "
        f"M_first_third={env_max_first_third:.4e}, M_last_third={env_max_last_third:.4e}, "
        f"|M3-M1|/max={env_rel_diff:.6e} <= tol_env={tol_env_rel:.6e}, "
        f"min_detectable_gamma/wb={gamma_min_detectable / omega_b:.6e}"
    )
    assert float(np.max(amp_k2_s)) <= beat_supremum_bound
    assert env_rel_diff <= tol_env_rel

    # 4. Asymmetric-density case n_+ : n_- = 1 : 0.5 so a sign or species bug
    #    cannot hide behind beam symmetry. Its unstable root is complex (omega = omega_r + i*gamma
    #    with omega_r != 0); we fit the growth rate gamma from |E_{k1}| over its per-case window.
    res_asym = run_two_stream_simulation(
        Nz=16,
        n_ppc_z=n_ppc_z,
        s_plus=1.0,
        s_minus=0.5,
        density_ratio_minus=0.5,
    )
    w_cont_asym = two_stream_continuum_root(
        omega_b,
        float(res_asym["k1"]),
        float(res_asym["v0"]),
        gamma0=float(res_asym["gamma0"]),
        s_plus=1.0,
        s_minus=0.5,
    )
    w_disc_asym = two_stream_discrete_root(
        omega_b,
        float(res_asym["k1"]),
        float(res_asym["v0"]),
        float(res_asym["dt"]),
        float(res_asym["dz"]),
        order=1,
        gather="galerkin",
        gamma0=float(res_asym["gamma0"]),
        s_plus=1.0,
        s_minus=0.5,
    )
    assert abs(np.real(w_cont_asym)) > 0.1 * omega_b
    assert abs(np.real(w_disc_asym)) > 0.1 * omega_b
    assert (
        abs(float(res_asym["t_horizon_wb"]) - 1.5 * float(res_asym["t_end_pred_wb"]))
        <= 1e-12
    )
    assert float(res_asym["t_horizon_wb"]) > float(res_asym["t_end_pred_wb"])
    assert float(res_asym["t_horizon_wb"]) > float(res_asym["t_win_end_wb"])
    assert res_asym["window_end_rule"] == "alpha_sat"
    g_fit_asym = float(res_asym["gamma_fit"])
    g_disc_asym = float(res_asym["gamma_disc"])
    g_cont_asym = float(res_asym["gamma_cont"])
    tol_asym = float(res_asym["tol_gamma"])
    err_asym = abs(g_fit_asym - g_disc_asym)
    print(
        f"[V4a-asym 1:0.5] win=[{res_asym['t_win_start_wb']:.2f}, {res_asym['t_win_end_wb']:.2f}] "
        f"(pred_end={res_asym['t_end_pred_wb']:.2f}, horizon={res_asym['t_horizon_wb']:.2f}, end_rule={res_asym['window_end_rule']}), "
        f"a_grow={res_asym['a_grow']:.4f}, a_osc={res_asym['a_osc']:.4f}, "
        f"omega_disc/wb={np.real(w_disc_asym) / omega_b:.6f}+{np.imag(w_disc_asym) / omega_b:.6f}j, "
        f"omega_cont/wb={np.real(w_cont_asym) / omega_b:.6f}+{g_cont_asym / omega_b:.6f}j, "
        f"gamma_fit/wb={g_fit_asym / omega_b:.6f}, "
        f"|fit-disc_asym|/wb={err_asym / omega_b:.6f} <= tol_asym/wb={tol_asym / omega_b:.6f}"
    )
    assert float(res_asym["gauss_final"]) <= 1e-12
    assert err_asym <= tol_asym
    assert abs(g_fit_asym - gamma_disc) > tol_gamma


def test_two_stream_grid_and_particle_convergence():
    """Check (b): 2nd-order spatial grid convergence and 1/sqrt(N_ppc) macroparticle noise convergence."""
    # 1. Spatial grid convergence at Nz = 16 vs Nz = 32 against the continuum prediction.
    #    In SolverFIT3D with dx = dy = 8 * dz and cfln = 0.8, dt = (0.8 / sqrt(33/32)) * dz / c
    #    scales linearly with dz, and the CFL-derived dimensionless parameters
    #    cells_per_lambda_p_16 = 16 * sqrt(3) / (2 * sqrt(2)), (wb * dt16)^2, (wb * dt32)^2,
    #    (k1 * dz16)^2 = (pi / 8)^2, and their ratio (wb * dt16 / (k1 * dz16))^2 are computed
    #    directly in code below.
    r16 = run_two_stream_simulation(Nz=16, n_ppc_z=32)
    r32 = run_two_stream_simulation(Nz=32, n_ppc_z=32)
    wb = float(r16["omega_b"])

    assert abs(float(r16["t_horizon_wb"]) - 1.5 * float(r16["t_end_pred_wb"])) <= 1e-12
    assert abs(float(r32["t_horizon_wb"]) - 1.5 * float(r32["t_end_pred_wb"])) <= 1e-12
    assert float(r16["t_horizon_wb"]) > float(r16["t_end_pred_wb"])
    assert float(r32["t_horizon_wb"]) > float(r32["t_end_pred_wb"])
    assert float(r16["t_horizon_wb"]) > float(r16["t_win_end_wb"])
    assert float(r32["t_horizon_wb"]) > float(r32["t_win_end_wb"])
    assert r16["window_end_rule"] == "alpha_sat"
    assert r32["window_end_rule"] == "alpha_sat"

    dt16, dt32 = float(r16["dt"]), float(r32["dt"])
    dz16, dz32 = float(r16["dz"]), float(r32["dz"])
    assert abs((dt16 / dz16) - (dt32 / dz32)) / (dt16 / dz16) <= 1e-14

    cells_per_lambda_p_16 = 16.0 * float(np.sqrt(3.0) / (2.0 * np.sqrt(2.0)))
    wb_dt16_sq = (wb * dt16) ** 2
    wb_dt32_sq = (wb * dt32) ** 2
    k1_dz16_sq = (float(r16["k1"]) * dz16) ** 2
    temp_to_spatial_ratio = wb_dt16_sq / k1_dz16_sq
    assert wb_dt16_sq < 5.0e-4
    assert abs(wb_dt16_sq / wb_dt32_sq - 4.0) <= 1e-12

    tol16 = float(r16["tol_gamma"])
    tol32 = float(r32["tol_gamma"])
    assert abs(float(r16["gamma_fit"]) - float(r16["gamma_disc"])) <= tol16
    assert abs(float(r32["gamma_fit"]) - float(r32["gamma_disc"])) <= tol32

    err_grid_16 = abs(float(r16["gamma_fit"]) - float(r16["gamma_cont"])) / wb
    err_grid_32 = abs(float(r32["gamma_fit"]) - float(r32["gamma_cont"])) / wb
    grid_ratio = err_grid_16 / err_grid_32
    grid_order = float(np.log2(grid_ratio))

    # Derive the spatial convergence order band directly from the discrete shifts
    # delta_disc(Nz) = |gamma_disc(Nz) - gamma_cont| and fit tolerances tol(Nz):
    shift16 = abs(float(r16["gamma_disc"]) - float(r16["gamma_cont"]))
    shift32 = abs(float(r32["gamma_disc"]) - float(r32["gamma_cont"]))
    grid_order_lo = float(np.log2((shift16 - tol16) / (shift32 + tol32)))
    grid_order_hi = float(np.log2((shift16 + tol16) / (shift32 - tol32)))

    print(
        f"\n[V4b-grid] win16=[{r16['t_win_start_wb']:.2f}, {r16['t_win_end_wb']:.2f}] (end_rule={r16['window_end_rule']}), "
        f"win32=[{r32['t_win_start_wb']:.2f}, {r32['t_win_end_wb']:.2f}] (end_rule={r32['window_end_rule']}), "
        f"cells/lambda_p(Nz=16)={cells_per_lambda_p_16:.2f}, "
        f"(wb*dt16)^2={wb_dt16_sq:.3e}, (wb*dt32)^2={wb_dt32_sq:.3e}, "
        f"(k1*dz16)^2={k1_dz16_sq:.4f}, (wb*dt/(k1*dz))^2={temp_to_spatial_ratio:.3e}, "
        f"err(Nz=16)={err_grid_16:.6f}, err(Nz=32)={err_grid_32:.6f}, ratio={grid_ratio:.3f}, "
        f"order={grid_order:.3f} in derived band [{grid_order_lo:.3f}, {grid_order_hi:.3f}]"
    )
    assert grid_order_lo <= grid_order <= grid_order_hi

    # 2. Particle-count convergence at N_ppc = 8 vs N_ppc = 128 across N_s = 16 seeds
    #    run in parallel using max_workers=os.cpu_count() and _init_worker_omp.
    #    Note: the chi^2 delta-method band [0.5 - 3*sigma_p, 0.5 + 3*sigma_p] with
    #    sigma_p = 1 / (sqrt(N_s - 1) * ln(N_ppc_high / N_ppc_low)) applies to the
    #    standard-deviation ratio std_err_order only (not to the ratio of mean RMS mode
    #    amplitudes noise_order), and at N_s = 16 it has low statistical power: the
    #    3*sigma_p band cannot distinguish order 0.5 from 0.25 or 0.75.
    n_ppc_low, n_ppc_high = 8, 128
    seeds = tuple(range(10, 170, 10))
    tasks = [(s, n_ppc_low, n_ppc_high) for s in seeds]
    with ProcessPoolExecutor(
        max_workers=os.cpu_count(), initializer=_init_worker_omp
    ) as pool:
        results = list(pool.map(_run_ppc_seed_pair, tasks))

    errs_low = [r[0] for r in results]
    errs_high = [r[1] for r in results]
    noise_low = [r[2] for r in results]
    noise_high = [r[3] for r in results]

    noise_ratio = float(np.mean(noise_low) / np.mean(noise_high))
    noise_order = float(np.log(noise_ratio) / np.log(n_ppc_high / n_ppc_low))
    std_err_ratio = float(np.std(errs_low, ddof=1) / np.std(errs_high, ddof=1))
    std_err_order = float(np.log(std_err_ratio) / np.log(n_ppc_high / n_ppc_low))

    sigma_p_chi2 = 1.0 / (
        float(np.sqrt(len(seeds) - 1)) * float(np.log(n_ppc_high / n_ppc_low))
    )
    ppc_order_lo = 0.5 - 3.0 * sigma_p_chi2
    ppc_order_hi = 0.5 + 3.0 * sigma_p_chi2

    print(
        f"\n[V4b-ppc (N_seeds={len(seeds)}, end_rule=alpha_sat)] "
        f"gamma_std_ratio={std_err_ratio:.3f} (std_order={std_err_order:.3f}), "
        f"chi2 std-ratio 3-sigma band=[{ppc_order_lo:.3f}, {ppc_order_hi:.3f}] "
        f"(low power at N_seeds={len(seeds)}: cannot tell 0.5 from 0.25 or 0.75), "
        f"diagnostic_k3_rms_ratio={noise_ratio:.3f} (order={noise_order:.3f})"
    )
    assert ppc_order_lo <= std_err_order <= ppc_order_hi
    assert float(np.sqrt(np.mean(np.square(errs_high)))) < float(
        np.sqrt(np.mean(np.square(errs_low)))
    )


def test_two_stream_mutations_fail_growth_check(monkeypatch):
    """Verify code-path mutations against their derived mutated roots and test the non-neutral input check."""
    import wakis.particles.solver as solver_mod
    from wakis.particles.pushers import push_momentum as orig_push_momentum

    # 0. Verify that an empty linear window raises AssertionError("no linear window") rather than returning 0.0
    with pytest.raises(AssertionError, match="no linear window"):
        run_two_stream_simulation(
            Nz=16, n_ppc_z=16, t_start_wb=15.0, t_end_wb=15.05, alpha_sat=1e-12
        )

    # Baseline unmutated reference run at its per-case derived horizon 1.5 * t_end_pred(ref)
    ref = run_two_stream_simulation(Nz=16, n_ppc_z=16)
    wb = float(ref["omega_b"])
    g_disc_ref = float(ref["gamma_disc"])
    tol_ref = float(ref["tol_gamma"])
    assert abs(float(ref["t_horizon_wb"]) - 1.5 * float(ref["t_end_pred_wb"])) <= 1e-12
    assert float(ref["t_horizon_wb"]) > float(ref["t_end_pred_wb"])
    assert float(ref["t_horizon_wb"]) > float(ref["t_win_end_wb"])
    assert ref["window_end_rule"] == "alpha_sat"
    assert abs(float(ref["gamma_fit"]) - g_disc_ref) <= tol_ref

    # Mutation 1 (MUST-FIX 1b): Monkeypatch push_momentum so ONE species (beam_minus) sees
    # the opposite force sign (-q E), while its deposit keeps the true charge (-e).
    # The cold-beam relation becomes 1 = omega_b^2 / (omega - k v0)^2 - omega_b^2 / (omega + k v0)^2
    # (s_plus = +1, s_minus = -1).
    def mutated_push_opposite_force(
        species, Ex, Ey, Ez, Bx, By, Bz, dt, pusher="boris"
    ):
        if species.name == "beam_minus":
            return orig_push_momentum(
                species, -Ex, -Ey, -Ez, -Bx, -By, -Bz, dt=dt, pusher=pusher
            )
        return orig_push_momentum(
            species, Ex, Ey, Ez, Bx, By, Bz, dt=dt, pusher=pusher
        )

    with monkeypatch.context() as mp:
        mp.setattr(solver_mod, "push_momentum", mutated_push_opposite_force)
        mut_opp = run_two_stream_simulation(
            Nz=16, n_ppc_z=16, s_plus=1.0, s_minus=-1.0
        )

    assert (
        abs(float(mut_opp["t_horizon_wb"]) - 1.5 * float(mut_opp["t_end_pred_wb"]))
        <= 1e-12
    )
    assert float(mut_opp["t_horizon_wb"]) > float(mut_opp["t_end_pred_wb"])
    assert float(mut_opp["t_horizon_wb"]) > float(mut_opp["t_win_end_wb"])
    assert mut_opp["window_end_rule"] == "alpha_sat"
    g_fit_opp = float(mut_opp["gamma_fit"])
    g_disc_opp = float(mut_opp["gamma_disc"])
    g_cont_opp = float(mut_opp["gamma_cont"])
    tol_opp = float(mut_opp["tol_gamma"])
    err_opp_vs_mut_root = abs(g_fit_opp - g_disc_opp)
    err_opp_vs_unmut_root = abs(g_fit_opp - g_disc_ref)
    print(
        f"\n[Mutation 1: opposite-force push on one species] "
        f"win=[{mut_opp['t_win_start_wb']:.2f}, {mut_opp['t_win_end_wb']:.2f}] "
        f"(pred_end={mut_opp['t_end_pred_wb']:.2f}, horizon={mut_opp['t_horizon_wb']:.2f}, end_rule={mut_opp['window_end_rule']}), "
        f"a_grow={mut_opp['a_grow']:.4f}, a_osc={mut_opp['a_osc']:.4f}, "
        f"gamma_fit/wb={g_fit_opp / wb:.6f}, "
        f"gamma_disc_mut/wb={g_disc_opp / wb:.6f}, "
        f"gamma_cont_mut/wb={g_cont_opp / wb:.6f}, "
        f"|fit-disc_mut|/wb={err_opp_vs_mut_root / wb:.6f} <= tol_mut/wb={tol_opp / wb:.6f}, "
        f"|fit-disc_ref|/wb={err_opp_vs_unmut_root / wb:.6f} > tol_ref/wb={tol_ref / wb:.6f}"
    )
    assert err_opp_vs_mut_root <= tol_opp
    assert err_opp_vs_unmut_root > tol_ref

    # Input check (MUST-FIX 1c, not a dispersion mutation):
    # On a periodic domain sum(S_tilde d_tilde) == 0 identically, so a non-neutral load
    # raises ValueError in run_two_stream_simulation, and if loaded into SolverFIT3D + attach()
    # produces the documented compensating point-charge jump artefact at node 0.
    with pytest.raises(ValueError, match="Non-neutral periodic load"):
        run_two_stream_simulation(Nz=16, n_ppc_z=16, bg_scale=0.5)

    dz_in = float(ref["dz"])
    Lz_in = 16.0 * dz_in
    Lx_in = 2.0 * dz_in * 8.0
    grid_in = GridFIT3D(0.0, Lx_in, 0.0, Lx_in, 0.0, Lz_in, 2, 2, 16, verbose=0)
    solver_in = SolverFIT3D(
        grid_in,
        cfln=0.8,
        bc_low=["periodic"] * 3,
        bc_high=["periodic"] * 3,
        verbose=0,
    )
    nb_in = epsilon_0 * m_e * (wb**2) / (e**2)
    Az_in = (Lx_in / 2.0) ** 2
    q_bg_neutral = 2.0 * e * nb_in * Az_in * dz_in
    sp_in = Species(name="beam_in", charge=-e, mass=m_e)
    z_in = (np.arange(16, dtype=np.float64) + 0.5) * dz_in
    sp_in.add_particles(
        x=np.full(16, 0.25 * Lx_in),
        y=np.full(16, 0.25 * Lx_in),
        z=z_in,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        w=(2.0 * nb_in * Lx_in * Lx_in * Lz_in) / 16.0,
    )
    q_nodes_in = np.zeros((2, 2, 16), dtype=np.float64, order="F")
    deposit_charge(
        sp_in.x / (0.5 * Lx_in),
        sp_in.y / (0.5 * Lx_in),
        sp_in.z / dz_in,
        sp_in.charge * sp_in.w,
        2,
        2,
        16,
        order=1,
        periodic=(True, True, True),
        q_nodes=q_nodes_in,
    )
    q_net_non_neutral = q_nodes_in + 0.5 * q_bg_neutral
    Ez_non_neutral = np.cumsum(q_net_non_neutral, axis=2) / (epsilon_0 * Az_in)
    Ez_non_neutral -= np.mean(Ez_non_neutral, axis=2, keepdims=True)
    solver_in.E[:, :, :, "z"] = Ez_non_neutral
    ParticleSolver(solver_in, species=[sp_in], shape=1).attach()
    q_bg_node0_ratio = float(
        np.mean(solver_in._particles_q_bg[:, :, 0]) / q_bg_neutral
    )
    q_bg_interior_ratio = float(
        np.mean(solver_in._particles_q_bg[:, :, 1:]) / q_bg_neutral
    )
    print(
        f"\n[Input check: non-neutral periodic load artefact] "
        f"q_bg_interior/q_neutral={q_bg_interior_ratio:.4f}, "
        f"q_bg_node0/q_neutral={q_bg_node0_ratio:.4f} (expected 0.5 + 16*0.5 = 8.5)"
    )
    assert abs(q_bg_interior_ratio - 0.5) <= 1e-12
    assert abs(q_bg_node0_ratio - (0.5 + 16.0 * 0.5)) <= 1e-12

    # Mutation 2 (MUST-FIX 1d): Code-path monkeypatch on wakis.particles.solver.gather_fields
    # Halving the gathered longitudinal electric field scales omega_b^2 by 0.5 (s_plus = s_minus = 0.5),
    # giving continuum prediction (1 / sqrt(2)) * sqrt(sqrt(7) - 2.5) * omega_{b,rel} at k1 * v0 = (sqrt(3)/2) * omega_{b,rel}.
    orig_gather = solver_mod.gather_fields

    def mutated_gather(*args, **kwargs):
        ex, ey, ez, bx, by, bz = orig_gather(*args, **kwargs)
        return ex, ey, 0.5 * ez, bx, by, bz

    with monkeypatch.context() as mp:
        mp.setattr(solver_mod, "gather_fields", mutated_gather)
        mut_gather = run_two_stream_simulation(
            Nz=16, n_ppc_z=32, s_plus=0.5, s_minus=0.5
        )

    assert (
        abs(
            float(mut_gather["t_horizon_wb"])
            - 1.5 * float(mut_gather["t_end_pred_wb"])
        )
        <= 1e-12
    )
    assert abs(float(mut_gather["t_horizon_wb"]) - T_HORIZON_WB) <= 1e-12
    assert float(mut_gather["t_horizon_wb"]) > float(mut_gather["t_end_pred_wb"])
    assert float(mut_gather["t_horizon_wb"]) > float(mut_gather["t_win_end_wb"])
    assert mut_gather["window_end_rule"] == "alpha_sat"
    g_fit_gat = float(mut_gather["gamma_fit"])
    g_disc_gat = float(mut_gather["gamma_disc"])
    g_cont_gat = float(mut_gather["gamma_cont"])
    wb_rel = float(mut_gather["wb_rel"])
    g_cont_gat_analytical = (
        (1.0 / np.sqrt(2.0)) * np.sqrt(np.sqrt(7.0) - 2.5) * wb_rel
    )
    assert abs(g_cont_gat - g_cont_gat_analytical) / wb <= 1e-15

    tol_gat = float(mut_gather["tol_gamma"])
    err_gat_vs_mut_root = abs(g_fit_gat - g_disc_gat)
    err_gat_vs_unmut_root = abs(g_fit_gat - g_disc_ref)
    print(
        f"\n[Mutation 2: gather monkeypatch 0.5*Ez] "
        f"win=[{mut_gather['t_win_start_wb']:.2f}, {mut_gather['t_win_end_wb']:.2f}] "
        f"(pred_end={mut_gather['t_end_pred_wb']:.2f}, horizon={mut_gather['t_horizon_wb']:.2f}, end_rule={mut_gather['window_end_rule']}), "
        f"a_grow={mut_gather['a_grow']:.4f}, a_osc={mut_gather['a_osc']:.4f}, "
        f"gamma_fit/wb={g_fit_gat / wb:.6f}, "
        f"gamma_disc_mut/wb={g_disc_gat / wb:.6f}, "
        f"gamma_cont_mut/wb={g_cont_gat / wb:.6f}, "
        f"|fit-disc_mut|/wb={err_gat_vs_mut_root / wb:.6f} <= tol_mut/wb={tol_gat / wb:.6f}, "
        f"|fit-disc_ref|/wb={err_gat_vs_unmut_root / wb:.6f} > tol_ref/wb={tol_ref / wb:.6f}"
    )
    assert err_gat_vs_mut_root <= tol_gat
    assert err_gat_vs_unmut_root > tol_ref
