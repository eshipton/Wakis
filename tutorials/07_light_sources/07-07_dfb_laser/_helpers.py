"""Helper functions for Tutorial 7.7: DFB laser with a quarter-wave phase shift."""

from __future__ import annotations

import json
import logging
import resource
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as eps_0
from scipy.ndimage import uniform_filter1d
from scipy.optimize import brentq, root

from wakis import GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.estimate import estimate, preflight
from wakis.gain import FourLevelGain, SaturableGain
from wakis.resonances import ResonanceMonitor, harminv
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.shapes import Box

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

TUTORIAL_DIR = Path(__file__).resolve().parent
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

# Baseline physical parameters (telecom 1.50 um index-coupled semiconductor DFB)
DEFAULT_LAM0 = 1.50e-6  # Bragg vacuum wavelength [m]
DEFAULT_N0 = 3.0  # Background effective refractive index [-]
DEFAULT_LAMBDA = DEFAULT_LAM0 / (2.0 * DEFAULT_N0)  # First-order grating pitch 250 nm [m]
DEFAULT_F_B = c0 / DEFAULT_LAM0  # Bragg frequency ~199.86 THz [Hz]
DEFAULT_OMEGA_B = 2.0 * np.pi * DEFAULT_F_B  # Bragg angular frequency [rad/s]
DEFAULT_N_PERIODS = 40  # Number of grating periods (L = 10 um)
DEFAULT_CPP = 24  # Cells per period (48 cells/wavelength in n0, 12 cells per tooth)
DEFAULT_I_SAT = 1.0e7  # Homogeneous saturation intensity [W/m^2]


def _rss_mb() -> float:
    """Return current peak resident set size in MiB."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def preflight_dfb_solver(
    nx: int,
    ny: int,
    nz: int,
    n_steps: int,
    n_poles: int = 1,
    has_cpml: bool = False,
    label: str = "DFB FIT",
) -> dict[str, Any]:
    """Run ``wakis.estimate.estimate`` and ``preflight`` prior to a FIT simulation.

    Parameters
    ----------
    nx, ny, nz : int
        Grid cell counts along x, y, and z.
    n_steps : int
        Total number of time steps.
    n_poles : int, optional
        Number of active auxiliary Lorentz/gain poles. Default is 1.
    has_cpml : bool, optional
        Whether CPML boundaries are active. Default is False.
    label : str, optional
        Human-readable run label.

    Returns
    -------
    est_dict : dict
        Resource estimate dictionary derived from ``wakis.estimate.estimate``.
    """
    est = estimate(
        (int(nx), int(ny), int(nz)),
        steps=int(n_steps),
        cpml=bool(has_cpml),
        dispersive_edges=int(3 * nx * ny * nz) if n_poles > 0 else 0,
        n_poles=max(1, int(n_poles)),
        backend="cpu",
    )
    preflight(est, "laptop")
    return {
        "label": str(label),
        "cells": int(est.cells),
        "ram_bytes": float(est.host_bytes),
        "seconds": float(est.seconds),
    }


# ==============================================================================
# 1. Coupled-Wave Theory (Kogelnik & Shank 1972; Haus & Shank 1976)
# ==============================================================================


def cwt_transfer_matrix(
    length: float,
    kappa: float,
    alpha: float,
    delta: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Compute the 2x2 coupled-wave transfer matrix for a uniform grating segment.

    Integrates the Kogelnik-Shank coupled-wave equations (with ``exp(+i*omega*t)``
    phasor convention) over a segment of length ``length`` [m] with coupling
    coefficient ``kappa`` [1/m], field amplitude gain ``alpha`` [1/m], detuning
    ``delta = (omega - omega_B) * n0 / c0`` [1/m], and grating phase ``phase`` [rad]:

    .. math::

        \\frac{\\mathrm{d}R}{\\mathrm{d}z} - (\\alpha - \\mathrm{i}\\delta) R
            = -\\mathrm{i}\\kappa e^{-\\mathrm{i}\\phi} S, \\qquad
        -\\frac{\\mathrm{d}S}{\\mathrm{d}z} - (\\alpha - \\mathrm{i}\\delta) S
            = -\\mathrm{i}\\kappa e^{+\\mathrm{i}\\phi} R.

    Parameters
    ----------
    length : float
        Segment length [m].
    kappa : float
        Index-coupling coefficient [1/m].
    alpha : float
        Field amplitude gain coefficient [1/m] (power gain is ``g = 2 * alpha``).
    delta : float
        Bragg wavenumber detuning ``beta - beta_0`` [1/m].
    phase : float, optional
        Grating phase ``phi`` [rad]. Default is ``0.0``.

    Returns
    -------
    M : ndarray of shape (2, 2), complex
        Transfer matrix mapping ``[R(z1), S(z1)]^T`` to ``[R(z2), S(z2)]^T``.
    """
    q = complex(alpha) - 1j * complex(delta)
    gamma_c = np.sqrt(complex(kappa) ** 2 + q**2)
    u = gamma_c * float(length)
    ch = np.cosh(u)
    sh_over_gamma = float(length) if abs(u) < 1e-12 else np.sinh(u) / gamma_c
    ep = np.exp(-1j * float(phase))
    em = np.exp(1j * float(phase))
    m11 = ch + q * sh_over_gamma
    m12 = -1j * complex(kappa) * ep * sh_over_gamma
    m21 = 1j * complex(kappa) * em * sh_over_gamma
    m22 = ch - q * sh_over_gamma
    return np.array([[m11, m12], [m21, m22]], dtype=complex)


def cwt_characteristic_residual(
    alpha_L: float,
    delta_L: float,
    kappa_L: float,
    phase_shift: float = 0.0,
) -> complex:
    """Return the non-trivial CWT characteristic residual for a phase-shifted DFB cavity.

    For a grating of total length ``L`` split at ``z = 0`` into two symmetric
    halves of length ``L/2`` with grating phases ``-phase_shift`` and ``+phase_shift``
    (corresponding to a one-way optical phase shift ``Phi = phase_shift`` and
    round-trip phase jump ``2*Phi``), non-reflecting facets ``R(-L/2) = 0`` and
    ``S(+L/2) = 0`` require ``M_total[1, 1] = 0``.

    Parameters
    ----------
    alpha_L : float
        Normalized amplitude threshold gain ``alpha * L`` [-].
    delta_L : float
        Normalized detuning ``delta * L`` [rad].
    kappa_L : float
        Normalized coupling strength ``kappa * L`` [-].
    phase_shift : float, optional
        One-way defect phase shift ``Phi`` [rad] (``0`` for a uniform grating,
        ``pi/2`` for a ``lambda/4``-shifted grating). Default is ``0.0``.

    Returns
    -------
    res : complex
        Complex secular determinant ``M_total[1, 1]`` divided by the trivial root
        factor ``sinh(gamma_c * L / 2) / (gamma_c * L / 2)``.
    """
    m_left = cwt_transfer_matrix(
        0.5, float(kappa_L), float(alpha_L), float(delta_L), phase=-float(phase_shift)
    )
    m_right = cwt_transfer_matrix(
        0.5, float(kappa_L), float(alpha_L), float(delta_L), phase=float(phase_shift)
    )
    m_tot = m_right @ m_left
    return complex(m_tot[1, 1])


def solve_cwt_uniform(kappa_L: float, m_order: int = 1) -> dict[str, float]:
    """Solve the Kogelnik-Shank threshold equation for a uniform index-coupled DFB laser.

    Solves the transcendental dispersion relation

    .. math::

        \\gamma_c L \\cosh(\\gamma_c L) - (\\alpha L - \\mathrm{i}\\delta L)\\sinh(\\gamma_c L) = 0,
        \\qquad (\\gamma_c L)^2 = (\\kappa L)^2 + (\\alpha L - \\mathrm{i}\\delta L)^2,

    eliminating the trivial ``gamma_c = 0`` root by factoring ``sinh(gamma_c L) / (gamma_c L)``.

    Parameters
    ----------
    kappa_L : float
        Normalized coupling strength ``kappa * L`` [-], ``> 0``.
    m_order : int, optional
        Longitudinal mode order ``m >= 1`` (``m = 1`` gives the lowest-threshold
        stopband-edge doublet at ``±|delta_1 L|``). Default is 1.

    Returns
    -------
    sol : dict
        Dictionary with keys ``'kappa_L'``, ``'alpha_th_L'``, ``'delta_th_L'``,
        ``'g_th_L'`` (power threshold gain ``2 * alpha_th_L``), and high-``kappa_L``
        asymptotic estimates ``'alpha_asymp_L'`` and ``'delta_asymp_L'``.
    """
    kL = float(kappa_L)
    if kL <= 0.0:
        raise ValueError(f"kappa_L must be > 0, got {kappa_L}.")
    m = max(1, int(m_order))

    def _residual(x: np.ndarray) -> list[float]:
        gL = complex(float(x[0]), float(x[1]))
        sh_over_g = 1.0 + (gL**2) / 6.0 if abs(gL) < 1e-8 else np.sinh(gL) / gL
        val = sh_over_g - ((-1) ** (m - 1)) * (-1j / kL)
        return [float(val.real), float(val.imag)]

    u_guess = float(np.arcsinh(m * np.pi / kL))
    v_guess = (m - 0.15) * np.pi
    res = root(_residual, [u_guess, v_guess], tol=1e-14)
    gL = complex(abs(float(res.x[0])), abs(float(res.x[1])))
    qL = gL / np.tanh(gL)
    a_th, d_th = float(qL.real), abs(float(qL.imag))

    alpha_asymp = (m * np.pi / kL) ** 2
    delta_asymp = float(np.sqrt(kL**2 + (m * np.pi) ** 2))
    return {
        "kappa_L": kL,
        "alpha_th_L": a_th,
        "delta_th_L": d_th,
        "g_th_L": 2.0 * a_th,
        "alpha_asymp_L": float(alpha_asymp),
        "delta_asymp_L": delta_asymp,
    }


def solve_cwt_uniform_dispersive(
    kappa_L: float,
    d_gamma_L: float,
    m_order: int = 1,
) -> dict[str, float]:
    """Solve the gain-dispersive Kogelnik-Shank threshold equation for a uniform DFB laser.

    Includes a complex Lorentzian gain susceptibility centered at the Bragg
    frequency ``omega_B`` with homogeneous half-linewidth ``delta_gamma * L =
    gamma_a * n_0 * L / (2 * c_0)`` in the coupled-wave propagation constant:

    .. math::

        q_{\\mathrm{disp}}(\\delta L; \\alpha_0 L) L
            = \\frac{\\alpha_0 L}{1 + \\mathrm{i}(\\delta L / \\delta_\\gamma L)} - \\mathrm{i}\\delta L.

    Non-reflecting facets ``M_{22}(q_{\\mathrm{disp}} L, \\kappa L) = 0`` require
    ``q_{\\mathrm{disp}} L = \\alpha_{\\mathrm{th},m}^{\\mathrm{CWT}} L -
    \\mathrm{i}\\delta_{\\mathrm{th},m}^{\\mathrm{CWT}} L``, yielding the exact
    closed-form Kramers-Kronig mode-pulled detuning and line-center threshold gain:

    .. math::

        \\delta_{\\mathrm{th},m}^{\\mathrm{disp}} L
            = \\frac{\\delta_{\\mathrm{th},m}^{\\mathrm{CWT}} L}{1 + \\alpha_{\\mathrm{th},m}^{\\mathrm{CWT}} L / (\\delta_\\gamma L)},
        \\qquad
        \\alpha_{0,\\mathrm{th},m}^{\\mathrm{disp}} L
            = \\alpha_{\\mathrm{th},m}^{\\mathrm{CWT}} L
              \\left[1 + \\left(\\frac{\\delta_{\\mathrm{th},m}^{\\mathrm{disp}} L}{\\delta_\\gamma L}\\right)^2\\right].

    Parameters
    ----------
    kappa_L : float
        Normalized coupling strength ``kappa * L`` [-], ``> 0``.
    d_gamma_L : float
        Normalized Lorentzian half-linewidth ``gamma_a * n_0 * L / (2 * c_0)`` [rad], ``> 0``.
    m_order : int, optional
        Longitudinal mode order ``m >= 1``. Default is 1.

    Returns
    -------
    sol : dict
        Dictionary with dispersionless CWT reference values (``'alpha_cwt_L'``,
        ``'delta_cwt_L'``), gain-dispersive threshold values (``'alpha0_th_disp_L'``,
        ``'delta_th_disp_L'``, ``'g0_th_disp_L'``, ``'r_p_th_disp'``), mode-pulling
        factor ``'pulling_factor'``, Lorentzian line factors at the dispersionless
        and gain-pulled stopband edges, and transfer-matrix residual ``'m22_residual'``.
    """
    dgL = float(d_gamma_L)
    if dgL <= 0.0:
        raise ValueError(f"d_gamma_L must be > 0, got {d_gamma_L}.")
    cwt_undisp = solve_cwt_uniform(kappa_L, m_order=m_order)
    a_cwt = float(cwt_undisp["alpha_th_L"])
    d_cwt = float(cwt_undisp["delta_th_L"])

    pulling_factor = 1.0 / (1.0 + a_cwt / dgL)
    d_disp = d_cwt * pulling_factor
    r_th_disp = 1.0 + (d_disp / dgL) ** 2
    a0_disp = a_cwt * r_th_disp

    q_disp_L = a0_disp / (1.0 + 1j * (d_disp / dgL)) - 1j * d_disp
    m_disp = cwt_transfer_matrix(
        1.0, float(kappa_L), float(q_disp_L.real), float(-q_disp_L.imag), phase=0.0
    )
    return {
        "kappa_L": float(kappa_L),
        "d_gamma_L": dgL,
        "alpha_cwt_L": a_cwt,
        "delta_cwt_L": d_cwt,
        "alpha0_th_disp_L": float(a0_disp),
        "delta_th_disp_L": float(d_disp),
        "g0_th_disp_L": float(2.0 * a0_disp),
        "r_p_th_disp": float(r_th_disp),
        "pulling_factor": float(pulling_factor),
        "lorentz_gain_factor_at_cwt_edge": float(1.0 / (1.0 + (d_cwt / dgL) ** 2)),
        "lorentz_gain_factor_at_disp_edge": float(1.0 / r_th_disp),
        "m22_residual": float(abs(m_disp[1, 1])),
    }


def solve_cwt_qws(kappa_L: float, m_order: int = 0) -> dict[str, float]:
    """Solve the Haus-Shank threshold equation for a quarter-wave-shifted DFB laser.

    For ``m_order == 0`` (the fundamental Bragg defect mode at ``delta_0 L = 0``),
    the round-trip phase shift ``2*Phi = pi`` across ``z = 0`` reduces the
    eigenvalue condition to the real scalar equation

    .. math::

        \\gamma_0 L \\cosh(\\tfrac{1}{2}\\gamma_0 L)
        - (\\alpha_0 L + \\kappa L)\\sinh(\\tfrac{1}{2}\\gamma_0 L) = 0,
        \\qquad \\gamma_0 = \\sqrt{\\kappa^2 + \\alpha_0^2}.

    For ``m_order >= 1``, solves for the nearest higher-order side modes at
    ``±|delta_1 L|`` outside the Bragg stopband.

    Parameters
    ----------
    kappa_L : float
        Normalized coupling strength ``kappa * L`` [-], ``> 0``.
    m_order : int, optional
        Mode index (``0`` for the Bragg defect mode at ``delta = 0``, ``1`` for
        the first side-mode doublet). Default is 0.

    Returns
    -------
    sol : dict
        Dictionary with keys ``'kappa_L'``, ``'alpha_th_L'``, ``'delta_th_L'``,
        ``'g_th_L'``, and ``'alpha_asymp_L'``.
    """
    kL = float(kappa_L)
    if kL <= 0.0:
        raise ValueError(f"kappa_L must be > 0, got {kappa_L}.")

    if int(m_order) == 0:

        def _f0(aL: float) -> float:
            gL = np.sqrt(kL**2 + aL**2)
            return float(gL * np.cosh(0.5 * gL) - (aL + kL) * np.sinh(0.5 * gL))

        a_th = float(brentq(_f0, 1e-9, max(10.0, 4.0 / max(kL, 0.1))))
        d_th = 0.0
        alpha_asymp = float(2.0 * kL * np.exp(-kL))
    else:
        m = int(m_order)
        d0 = float(np.sqrt(kL**2 + ((m + 0.5) * np.pi) ** 2))
        a0 = max(0.8, 2.0 / max(kL, 0.3))

        def _res_side(x: np.ndarray) -> list[float]:
            val = cwt_characteristic_residual(
                float(x[0]), float(x[1]), kL, phase_shift=0.5 * np.pi
            )
            return [float(val.real), float(val.imag)]

        sol = root(_res_side, [a0, d0], tol=1e-14)
        a_th, d_th = float(sol.x[0]), abs(float(sol.x[1]))
        res_check = abs(
            cwt_characteristic_residual(a_th, d_th, kL, phase_shift=0.5 * np.pi)
        )
        if res_check > 1e-10 or a_th <= 0.0:
            raise RuntimeError(
                f"solve_cwt_qws failed for kappa_L={kL}, m_order={m}: |M22|={res_check:.3e}"
            )
        alpha_asymp = float(((m + 0.5) * np.pi / kL) ** 2)

    return {
        "kappa_L": kL,
        "alpha_th_L": a_th,
        "delta_th_L": d_th,
        "g_th_L": 2.0 * a_th,
        "alpha_asymp_L": alpha_asymp,
    }


def solve_cwt_phase_shifted(kappa_L: float, phase_shift: float) -> dict[str, float]:
    """Solve for the lowest-threshold CWT mode at an arbitrary central phase shift ``Phi``.

    Parameters
    ----------
    kappa_L : float
        Normalized coupling strength ``kappa * L`` [-].
    phase_shift : float
        One-way phase shift ``Phi`` [rad] at ``z = 0`` (round-trip phase jump ``2*Phi``).

    Returns
    -------
    sol : dict
        Dictionary with ``'phase_shift_rad'``, ``'phase_shift_deg'``, ``'alpha_th_L'``,
        ``'delta_th_L'``, and ``'g_th_L'``.
    """
    kL = float(kappa_L)
    phi = float(phase_shift)
    best: tuple[float, float] | None = None
    for sgn in (1.0, -1.0):
        target = sgn * np.exp(-1j * phi)
        for d_guess in (-3.6, -2.0, -0.5, 0.0, 0.5, 2.0, 3.6):

            def _res(x: np.ndarray, _t: complex = target) -> list[float]:
                aL, dL = float(x[0]), float(x[1])
                qL = aL - 1j * dL
                gL = np.sqrt(kL**2 + qL**2)
                sh_over_x = 1.0 if abs(gL) < 1e-12 else np.sinh(0.5 * gL) / (0.5 * gL)
                ch = np.cosh(0.5 * gL)
                val = -1j * kL * 0.5 * sh_over_x - _t * (ch - 0.5 * qL * sh_over_x)
                return [float(val.real), float(val.imag)]

            sol = root(_res, [0.75, d_guess], tol=1e-12)
            if (
                sol.success
                and sol.x[0] > 0.02
                and abs(sol.fun[0]) + abs(sol.fun[1]) < 1e-8
            ):
                cand = (float(sol.x[0]), float(sol.x[1]))
                if best is None or cand[0] < best[0] - 1e-7:
                    best = cand

    if best is None:
        raise RuntimeError(f"Failed to converge CWT root for kappa_L={kL}, Phi={phi}.")
    return {
        "phase_shift_rad": phi,
        "phase_shift_deg": float(np.rad2deg(phi)),
        "alpha_th_L": best[0],
        "delta_th_L": best[1],
        "g_th_L": 2.0 * best[0],
    }


def cwt_mode_profile(
    z_over_L: np.ndarray,
    kappa_L: float,
    alpha_L: float | None = None,
    delta_L: float | None = None,
    phase_shift: float = 0.5 * np.pi,
) -> dict[str, Any]:
    """Compute the longitudinal CWT mode envelopes ``R(z/L)``, ``S(z/L)``, and intensity.

    Normalized so that the outgoing wave amplitude at the right facet ``z/L = +1/2``
    is ``|R(+1/2)|^2 = 1`` (with ``S(+1/2) = 0`` and ``R(-1/2) = 0``).

    Parameters
    ----------
    z_over_L : ndarray
        Normalized longitudinal coordinates ``z / L`` in ``[-0.5, +0.5]``.
    kappa_L : float
        Normalized coupling strength ``kappa * L`` [-].
    alpha_L, delta_L : float or None, optional
        Normalized threshold gain and detuning. If None, solved automatically via
        ``solve_cwt_phase_shifted``.
    phase_shift : float, optional
        One-way phase shift ``Phi`` [rad] at ``z = 0``. Default is ``pi/2`` (``lambda/4``).

    Returns
    -------
    prof : dict
        Dictionary with arrays ``'z_over_L'``, ``'R'``, ``'S'``, ``'I_norm'``
        (``|R|^2 + |S|^2`` normalized to unit facet output), ``'peak_to_edge'``,
        and ``'energy_balance_check'`` (``alpha_th * L * int_{-1/2}^{1/2} I_norm du == 1``).
    """
    u_arr = np.asarray(z_over_L, dtype=float)
    if alpha_L is None or delta_L is None:
        eig = solve_cwt_phase_shifted(kappa_L, phase_shift)
        alpha_L = eig["alpha_th_L"]
        delta_L = eig["delta_th_L"]

    kL = float(kappa_L)
    aL = float(alpha_L)
    dL = float(delta_L)
    phi = float(phase_shift)

    R = np.zeros_like(u_arr, dtype=complex)
    S = np.zeros_like(u_arr, dtype=complex)

    # Integrate backwards from u = +0.5 where R(+0.5) = 1, S(+0.5) = 0
    for i, u in enumerate(u_arr):
        if u >= 0.0:
            m_inv = cwt_transfer_matrix(u - 0.5, kL, aL, dL, phase=phi)
            vec = m_inv @ np.array([1.0 + 0j, 0.0 + 0j])
        else:
            m_right_inv = cwt_transfer_matrix(-0.5, kL, aL, dL, phase=phi)
            m_left_inv = cwt_transfer_matrix(u, kL, aL, dL, phase=-phi)
            vec = m_left_inv @ (m_right_inv @ np.array([1.0 + 0j, 0.0 + 0j]))
        R[i] = vec[0]
        S[i] = vec[1]

    I_norm = np.abs(R) ** 2 + np.abs(S) ** 2
    energy_balance = float(aL * np.trapezoid(I_norm, u_arr))
    i_mid = int(np.argmin(np.abs(u_arr)))
    return {
        "z_over_L": u_arr,
        "R": R,
        "S": S,
        "I_norm": I_norm,
        "peak_to_edge": float(I_norm[i_mid] / max(I_norm[-1], 1e-30)),
        "energy_balance_check": energy_balance,
        "alpha_th_L": aL,
        "delta_th_L": dL,
    }


def cwt_rigrod_output(
    pump_ratios: np.ndarray | list[float],
    kappa_L: float = 2.0,
    phase_shift: float = 0.5 * np.pi,
) -> dict[str, Any]:
    """Compute rate-equation and coupled-wave Rigrod L-I curves vs pump ``g_0 / g_th``.

    Derives the steady-state single-facet output intensity ``I_out / I_sat`` from
    Poynting energy conservation across ``z in [-L/2, +L/2]``:

    1. **Linearized SHB slope efficiency** near threshold:
       ``I_out / I_sat = eta_SHB * max(0, g_0 / g_th - 1)`` with
       ``eta_SHB = (int I_norm du) / (int I_norm^2 du)``.
    2. **Coupled-wave Rigrod integral** including both longitudinal envelope
       spatial hole burning ``I_norm(u)`` and Bragg standing-wave grating saturation
       ``<I(u, theta) / (1 + I(u, theta) / I_sat)>``.

    Parameters
    ----------
    pump_ratios : array_like
        Normalized small-signal gain values ``r = g_0 / g_th``.
    kappa_L : float, optional
        Normalized coupling strength ``kappa * L`` [-]. Default is ``2.0``.
    phase_shift : float, optional
        Central phase shift ``Phi`` [rad]. Default is ``pi/2``.

    Returns
    -------
    out : dict
        Dictionary with ``'pump_ratios'``, ``'I_out_linear_shb'``,
        ``'I_out_rigrod_env'``, ``'I_out_rigrod_sw'``, and ``'eta_shb'``.
    """
    r_arr = np.asarray(pump_ratios, dtype=float)
    u = np.linspace(0.0, 0.5, 1001)
    prof = cwt_mode_profile(
        np.concatenate([-u[::-1], u[1:]]), kappa_L=kappa_L, phase_shift=phase_shift
    )
    psi = prof["I_norm"]
    R_arr = prof["R"]
    S_arr = prof["S"]
    u_full = prof["z_over_L"]

    int1 = float(np.trapezoid(psi, u_full))
    int2 = float(np.trapezoid(psi**2, u_full))
    eta_shb = int1 / int2

    I_lin = np.where(r_arr > 1.0, eta_shb * (r_arr - 1.0), 0.0)
    I_env = np.zeros_like(r_arr)
    I_sw = np.zeros_like(r_arr)
    prod = 2.0 * np.abs(R_arr) * np.abs(S_arr)

    for i, r in enumerate(r_arr):
        if r <= 1.0:
            continue
        I_env[i] = float(
            brentq(
                lambda x: np.trapezoid(r * psi / (1.0 + x * psi), u_full) - int1,
                0.0,
                20.0 * (r - 1.0) + 1.0,
            )
        )

        def _sw_res(x: float, _r: float = float(r)) -> float:
            if x <= 1e-12:
                return (_r - 1.0) * int1
            sat_avg = (1.0 / x) * (
                1.0 - 1.0 / np.sqrt((1.0 + x * psi) ** 2 - (x * prod) ** 2)
            )
            return float(np.trapezoid(_r * sat_avg, u_full) - int1)

        I_sw[i] = float(brentq(_sw_res, 1e-9, 20.0 * (r - 1.0) + 1.0))

    return {
        "pump_ratios": r_arr,
        "I_out_linear_shb": I_lin,
        "I_out_rigrod_env": I_env,
        "I_out_rigrod_sw": I_sw,
        "eta_shb": float(eta_shb),
    }


# ==============================================================================
# 2. 1D and 2D Slab-Waveguide DFB FIT Solver Builders
# ==============================================================================


def _grating_modulation(
    z_coords: np.ndarray,
    Lambda: float,
    phase_shift: float,
    kappa_sign: float = 1.0,
) -> np.ndarray:
    """Evaluate the unit-amplitude longitudinal Bragg grating modulation profile ``mod(z)``.

    Parameters
    ----------
    z_coords : ndarray
        Longitudinal coordinates ``z`` [m].
    Lambda : float
        First-order Bragg grating period [m].
    phase_shift : float
        One-way central phase shift ``Phi`` [rad] (``0`` for uniform, ``pi/2`` for QWS).
    kappa_sign : float, optional
        Sign of the coupling modulation (``+1`` or ``-1``). Default is ``1.0``.

    Returns
    -------
    mod : ndarray
        Spatial modulation profile in ``[-1, +1]`` with the same shape as ``z_coords``.
    """
    phi = float(phase_shift)
    sgn = float(np.sign(kappa_sign))
    if abs(phi) < 1e-12:
        # Uniform grating: sin(2*beta_0*z) vanishes at z = +/- L/2 (continuous facets)
        return sgn * np.sin(2.0 * np.pi * z_coords / float(Lambda))
    if abs(phi - 0.5 * np.pi) < 1e-12:
        # Quarter-wave-shifted grating: -sin(2*beta_0*|z|) has a pi phase jump at z=0
        # and vanishes at z = +/- L/2
        return -sgn * np.sin(2.0 * np.pi * np.abs(z_coords) / float(Lambda))
    theta = np.where(z_coords < 0.0, -phi, phi)
    return sgn * np.cos(2.0 * np.pi * z_coords / float(Lambda) + theta)


def _slab_core_index_modulation(
    kappa: float,
    lam0: float,
    n_eff: float,
    n_core: float,
    Gamma_core: float,
) -> float:
    """Compute the 2D slab core index perturbation ``n1_core`` for modal coupling ``kappa``.

    By waveguide coupled-mode theory (Yariv 1973), a core index perturbation
    ``n1_core`` produces effective modal coupling
    ``kappa = (pi / lam0) * (n_core / n_eff) * Gamma_core * n1_core``.
    """
    return (
        float(kappa)
        * (float(lam0) / np.pi)
        * (float(n_eff) / float(n_core))
        / float(Gamma_core)
    )


def build_1d_dfb_solver(
    kappa_L: float = 2.0,
    phase_shift: float = 0.5 * np.pi,
    kappa_sign: float = 1.0,
    n_periods: int = DEFAULT_N_PERIODS,
    cells_per_period: int = DEFAULT_CPP,
    pad_cells: int = 16,
    lam0: float = DEFAULT_LAM0,
    n0: float = DEFAULT_N0,
    cfln_med: float = 0.90,
    gain_g0: float = 0.0,
    gain_I_sat: float = np.inf,
    gain_gamma_factor: float = 1.5,
    gain_tau_cycles: float = 6.0,
) -> tuple[SolverFIT3D, dict[str, Any]]:
    """Construct a 1D index-coupled DFB grating solver in ``SolverFIT3D``.

    Uses periodic transverse boundaries along ``x`` and ``y`` (exact 1D TEM plane
    wave) and index-matched Mur absorbing boundary conditions (``bg=[n0**2, 1, 0]``)
    at ``z_min`` and ``z_max`` to realize the non-reflecting facets ``R(-L/2) = 0``
    and ``S(+L/2) = 0`` of Kogelnik & Shank (1972).

    Parameters
    ----------
    kappa_L : float, optional
        Normalized coupling coefficient ``kappa * L`` [-]. Default is ``2.0``.
    phase_shift : float, optional
        Central one-way phase shift ``Phi`` [rad] (``0`` for a uniform grating,
        ``pi/2`` for a ``lambda/4``-shifted grating). Default is ``pi/2``.
    kappa_sign : float, optional
        Sign of the index modulation (``+1.0`` or ``-1.0``). Default is ``+1.0``.
    n_periods : int, optional
        Number of grating periods ``N_Lambda``. Default is ``40``.
    cells_per_period : int, optional
        Grid cells per Bragg period ``Lambda = lam0 / (2 * n0)``. Default is ``24``
        (corresponding to ``48`` cells per wavelength in ``n0`` and ``12`` cells
        across each ``Lambda/2`` half-period tooth).
    pad_cells : int, optional
        Index-matched cladding buffer cells outside each grating facet. Default is ``16``.
    lam0 : float, optional
        Bragg vacuum wavelength [m]. Default is ``1.50e-6``.
    n0 : float, optional
        Background refractive index [-]. Default is ``3.0``.
    cfln_med : float, optional
        1D Courant number in the dielectric medium. Default is ``0.90``.
    gain_g0 : float, optional
        Small-signal power gain coefficient ``g_0`` [1/m] inside ``z in [-L/2, +L/2]``.
        Default is ``0.0`` (passive).
    gain_I_sat : float, optional
        Saturation intensity ``I_sat`` [W/m^2]. Use ``np.inf`` for linear gain.
    gain_gamma_factor : float, optional
        Homogeneous linewidth ``gamma / omega_B``. Default is ``1.5``.
    gain_tau_cycles : float, optional
        Intensity averaging time constant ``tau_I / T_0`` in optical cycles.
        Default is ``6.0``.

    Returns
    -------
    solver : SolverFIT3D
        Configured FIT solver instance.
    meta : dict
        Geometry, mesh resolution, and index-modulation metadata.
    """
    f_B = c0 / float(lam0)
    omega_B = 2.0 * np.pi * f_B
    T0 = 1.0 / f_B
    Lambda = float(lam0) / (2.0 * float(n0))
    L = int(n_periods) * Lambda
    cpp = int(cells_per_period)
    dz = Lambda / cpp
    nz_grat = int(n_periods) * cpp
    nz_total = nz_grat + 2 * int(pad_cells)
    z_min = -0.5 * L - int(pad_cells) * dz
    z_max = 0.5 * L + int(pad_cells) * dz
    dx = 50.0 * dz

    kappa = float(kappa_L) / L
    n1 = kappa * float(lam0) / np.pi
    dt_opt = float(cfln_med) * (float(n0) - abs(n1)) * dz / c0

    grid = GridFIT3D(
        -0.5 * dx,
        0.5 * dx,
        -0.5 * dx,
        0.5 * dx,
        z_min,
        z_max,
        1,
        1,
        nz_total,
        use_mpi=False,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        dt=dt_opt,
        bc_low=["periodic", "periodic", "abc"],
        bc_high=["periodic", "periodic", "abc"],
        bg=[float(n0) ** 2, 1.0, 0.0],
        verbose=0,
    )

    z_nodes = grid.z[: solver.Nz]
    in_grat = (z_nodes >= -0.5 * L) & (z_nodes <= 0.5 * L)
    # The grating modulates permittivity as eps(z) = eps_0_grat + 2*n0*n1*mod(z), omitting
    # the second-harmonic n1^2*mod(z)^2 term of n(z)^2 = (n0 + n1*mod(z))^2. Because
    # SolverFIT3D stores cell inverse permittivity 1/eps(z), the harmonic-mean effective
    # permittivity of eps_0_grat + 2*n0*n1*cos(2*beta_0*z) is sqrt(eps_0_grat^2 - 4*n0^2*n1^2)
    # ~= eps_0_grat - 2*n1^2, while second-order Bloch-Floquet self-coupling and finite-grid
    # Yee dispersion shift the discrete stopband center by ~1.5*(n1/n0)^2. Setting
    # eps_0_grat = n0^2 * (1 + 1.5*(n1/n0)^2) is a second-order Bragg-center calibration
    # that keeps the discrete m=0 defect resonance centered on f_B across kappa_L.
    eps_0_grat = (float(n0) ** 2) * (1.0 + 1.5 * (n1 / float(n0)) ** 2)
    mod = _grating_modulation(
        z_nodes, Lambda=Lambda, phase_shift=phase_shift, kappa_sign=kappa_sign
    )

    eps_z = np.where(in_grat, eps_0_grat + 2.0 * float(n0) * n1 * mod, float(n0) ** 2)
    for d in ("x", "y", "z"):
        solver.ieps[:, :, :, d] = (1.0 / (eps_z * eps_0))[None, None, :]
    solver._isolate_edge_ghosts()
    solver.update_tensors()

    if abs(float(gain_g0)) > 0.0:
        sg = SaturableGain(
            omega_0=omega_B,
            gamma=float(gain_gamma_factor) * omega_B,
            g0=float(gain_g0),
            I_sat=float(gain_I_sat),
            tau_I=float(gain_tau_cycles) * T0,
            eps_inf=float(n0) ** 2,
        )
        solver.add_gain_material(sg, center=(0.0, 0.0, 0.0), size=(2.0 * dx, 2.0 * dx, L))
        # Preserve the spatially modulated static grating permittivity eps_s(z) in the gain update
        grp = solver._gain_groups[-1]
        grp["a1"] = grp["eps_s"].copy()
        grp["den"] = grp["eps_s"].copy()

    n_max = float(n0) + abs(n1)
    cpw_min = (float(lam0) / n_max) / dz
    cells_per_tooth = 0.5 * cpp
    meta = {
        "lam0": float(lam0),
        "n0": float(n0),
        "n1": float(n1),
        "n_max": n_max,
        "f_B": f_B,
        "omega_B": omega_B,
        "T0": T0,
        "Lambda": Lambda,
        "L": L,
        "n_periods": int(n_periods),
        "cells_per_period": cpp,
        "cells_per_wavelength_min": float(cpw_min),
        "cells_per_tooth": float(cells_per_tooth),
        "dz": float(dz),
        "dx": float(dx),
        "dt": float(solver.dt),
        "pad_cells": int(pad_cells),
        "nz_grat": nz_grat,
        "nz_total": nz_total,
        "iz_center": int(pad_cells) + nz_grat // 2,
        "iz_third": int(pad_cells) + nz_grat // 3,
        "iz_out_right": nz_total - max(4, int(pad_cells) // 2),
        "z_nodes": z_nodes,
        "eps_z": eps_z,
        "kappa_L": float(kappa_L),
        "phase_shift": float(phase_shift),
        "kappa_sign": float(kappa_sign),
    }
    return solver, meta


def solve_discrete_slab_te0(
    k0: float,
    x_nodes: np.ndarray,
    d_core: float,
    n_core: float,
    n_clad: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve the discrete 1D transverse FIT eigenproblem for the fundamental TE0 slab mode.

    Parameters
    ----------
    k0 : float
        Vacuum wavenumber ``2 * pi / lam0`` [1/m].
    x_nodes : ndarray
        Transverse node coordinates [m] from ``-x_max`` to ``+x_max``.
    d_core : float
        Slab core thickness [m].
    n_core, n_clad : float
        Core and cladding refractive indices [-].

    Returns
    -------
    n_eff : float
        Discrete TE0 effective index ``beta / k0`` [-].
    psi : ndarray
        Normalized transverse mode profile ``Ey(x)`` on ``x_nodes``.
    in_core : ndarray of bool
        Boolean mask of core nodes ``|x| < d_core / 2``.
    """
    dx = float(x_nodes[1] - x_nodes[0])
    nx = len(x_nodes) - 1
    in_core = np.abs(x_nodes) < 0.5 * float(d_core)
    eps_1d = np.where(in_core, float(n_core) ** 2, float(n_clad) ** 2)
    n_int = nx - 1
    diag = -2.0 / (dx**2) + (float(k0) ** 2) * eps_1d[1:-1]
    off = np.ones(n_int - 1, dtype=float) / (dx**2)
    evals, evecs = np.linalg.eigh(np.diag(diag) + np.diag(off, 1) + np.diag(off, -1))
    beta2 = float(evals[-1])
    psi = np.zeros(nx + 1, dtype=float)
    psi[1:-1] = evecs[:, -1]
    if psi[nx // 2] < 0.0:
        psi = -psi
    psi /= np.linalg.norm(psi)
    return float(np.sqrt(beta2) / float(k0)), psi, in_core


def build_2d_slab_dfb_solver(
    kappa_L: float = 2.0,
    phase_shift: float = 0.5 * np.pi,
    n_periods: int = 30,
    cells_per_period: int = 20,
    dx: float = 20.0e-9,
    nx: int = 40,
    d_core: float = 0.32e-6,
    n_core: float = 3.0,
    n_clad: float = 2.4,
    lam0: float = DEFAULT_LAM0,
    gain_g0_core: float = 0.0,
) -> tuple[SolverFIT3D, dict[str, Any]]:
    """Construct a 2D TE slab-waveguide DFB grating solver in ``SolverFIT3D``.

    Resolves the transverse waveguide core ``|x| < d_core / 2`` with ``16`` cells
    across ``d_core = 320 nm`` (``dx = 20 nm``, ``25`` cells/wavelength in the core)
    and ``20`` cells per Bragg period ``Lambda = lam0 / (2 * n_eff)`` along ``z``.

    Parameters
    ----------
    kappa_L : float, optional
        Effective modal coupling strength ``kappa * L`` [-]. Default is ``2.0``.
    phase_shift : float, optional
        Central one-way phase shift ``Phi`` [rad]. Default is ``pi/2``.
    n_periods : int, optional
        Number of grating periods along ``z``. Default is ``30``.
    cells_per_period : int, optional
        Longitudinal cells per period ``Lambda``. Default is ``20``.
    dx : float, optional
        Transverse cell size [m]. Default is ``20e-9`` (20 nm).
    nx : int, optional
        Number of transverse cells across ``[-nx*dx/2, +nx*dx/2]``. Default is ``40``.
    d_core : float, optional
        Core thickness [m]. Default is ``0.32e-6`` (320 nm, 16 cells).
    n_core, n_clad : float, optional
        Core and cladding refractive indices. Defaults are ``3.0`` and ``2.4``.
    lam0 : float, optional
        Bragg vacuum wavelength [m]. Default is ``1.50e-6``.
    gain_g0_core : float, optional
        Linear power gain coefficient [1/m] inside the active core. Default is ``0.0``.

    Returns
    -------
    solver : SolverFIT3D
        Configured 2D slab-waveguide FIT solver.
    meta : dict
        Waveguide modal parameters (``n_eff``, ``n_g``, ``Gamma_core``, ``mode_x``)
        and mesh metadata.
    """
    f_B = c0 / float(lam0)
    omega_B = 2.0 * np.pi * f_B
    T0 = 1.0 / f_B
    k0 = 2.0 * np.pi / float(lam0)
    x_nodes = np.linspace(-0.5 * int(nx) * float(dx), 0.5 * int(nx) * float(dx), int(nx) + 1)

    n_eff, mode_x, in_core_1d = solve_discrete_slab_te0(k0, x_nodes, d_core, n_core, n_clad)
    dk = 1e-4 * k0
    n_eff_p, _, _ = solve_discrete_slab_te0(k0 + dk, x_nodes, d_core, n_core, n_clad)
    n_eff_m, _, _ = solve_discrete_slab_te0(k0 - dk, x_nodes, d_core, n_core, n_clad)
    n_g = float(n_eff + k0 * (n_eff_p - n_eff_m) / (2.0 * dk))
    Gamma_core = float(np.sum(mode_x[in_core_1d] ** 2) / np.sum(mode_x**2))

    Lambda = float(lam0) / (2.0 * n_eff)
    L = int(n_periods) * Lambda
    cpp = int(cells_per_period)
    dz = Lambda / cpp
    pad_z = 12
    nz_grat = int(n_periods) * cpp
    nz_total = nz_grat + 2 * pad_z
    z_min = -0.5 * L - pad_z * dz
    z_max = 0.5 * L + pad_z * dz
    dy = 50.0 * dz

    kappa = float(kappa_L) / L
    # By waveguide coupled-mode theory (Yariv 1973), core index perturbation n1_core
    # produces modal coupling kappa = (pi / lam0) * (n_core / n_eff) * Gamma_core * n1_core
    n1_core = _slab_core_index_modulation(
        kappa=kappa,
        lam0=float(lam0),
        n_eff=n_eff,
        n_core=float(n_core),
        Gamma_core=Gamma_core,
    )
    dt_opt = 0.90 * float(n_clad) / (c0 * np.sqrt(1.0 / float(dx) ** 2 + 1.0 / dz**2))

    grid = GridFIT3D(
        -0.5 * int(nx) * float(dx),
        0.5 * int(nx) * float(dx),
        -0.5 * dy,
        0.5 * dy,
        z_min,
        z_max,
        int(nx),
        1,
        nz_total,
        use_mpi=False,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        dt=dt_opt,
        bc_low=["pec", "periodic", "abc"],
        bc_high=["pec", "periodic", "abc"],
        bg=[n_eff**2, 1.0, 0.0],
        verbose=0,
    )

    z_nodes = grid.z[: solver.Nz]
    X2, Z2 = np.meshgrid(x_nodes, z_nodes, indexing="ij")
    in_core = np.abs(X2) < 0.5 * float(d_core)
    in_grat = (Z2 >= -0.5 * L) & (Z2 <= 0.5 * L)
    mod = _grating_modulation(
        Z2, Lambda=Lambda, phase_shift=phase_shift, kappa_sign=1.0
    )

    eps_0_core = (float(n_core) ** 2) * (1.0 + 1.5 * (n1_core / float(n_core)) ** 2)
    eps_2d = np.where(in_core, float(n_core) ** 2, float(n_clad) ** 2)
    eps_2d = np.where(
        in_core & in_grat, eps_0_core + 2.0 * float(n_core) * n1_core * mod, eps_2d
    )
    for d in ("x", "y", "z"):
        solver.ieps[:, :, :, d] = 1.0 / (eps_2d[:, None, :] * eps_0)
    solver._isolate_edge_ghosts()
    solver.update_tensors()

    if abs(float(gain_g0_core)) > 0.0:
        sg = SaturableGain(
            omega_0=omega_B,
            gamma=1.5 * omega_B,
            g0=float(gain_g0_core),
            I_sat=np.inf,
            eps_inf=float(n_core) ** 2,
        )
        solver.add_gain_material(
            sg, center=(0.0, 0.0, 0.0), size=(float(d_core) - float(dx), 2.0 * dy, L)
        )
        grp = solver._gain_groups[-1]
        grp["a1"] = grp["eps_s"].copy()
        grp["den"] = grp["eps_s"].copy()

    n_max = float(n_core) + abs(n1_core)
    cpw_x = (float(lam0) / n_max) / float(dx)
    cpw_z = (float(lam0) / n_max) / dz
    meta = {
        "lam0": float(lam0),
        "f_B": f_B,
        "omega_B": omega_B,
        "T0": T0,
        "n_core": float(n_core),
        "n_clad": float(n_clad),
        "d_core": float(d_core),
        "n_eff": float(n_eff),
        "n_g": float(n_g),
        "Gamma_core": float(Gamma_core),
        "n1_core": float(n1_core),
        "Lambda": float(Lambda),
        "L": float(L),
        "n_periods": int(n_periods),
        "cells_per_period": cpp,
        "dx": float(dx),
        "dz": float(dz),
        "dy": float(dy),
        "dt": float(solver.dt),
        "nx": int(nx),
        "nz_total": nz_total,
        "pad_z": pad_z,
        "iz_center": pad_z + nz_grat // 2,
        "iz_third": pad_z + nz_grat // 3,
        "x_nodes": x_nodes,
        "z_nodes": z_nodes,
        "mode_x": mode_x,
        "eps_2d": eps_2d,
        "cells_across_core": float(d_core / dx),
        "cells_per_tooth": float(0.5 * cpp),
        "cells_per_wavelength_min": float(min(cpw_x, cpw_z)),
    }
    return solver, meta


# ==============================================================================
# 3. Simulation Study Tasks
# ==============================================================================


def run_1d_harminv_resonance(
    kappa_L: float = 2.0,
    phase_shift: float = 0.5 * np.pi,
    kappa_sign: float = 1.0,
    n_periods: int = DEFAULT_N_PERIODS,
    cells_per_period: int = DEFAULT_CPP,
    pad_cells: int = 16,
    cfln_med: float = 0.90,
    gain_g0: float = 0.0,
    n_cycles: float = 175.0,
    excite_mode: str = "auto",
) -> dict[str, Any]:
    """Run a 1D DFB FIT ring-down simulation and extract complex resonances with ``harminv``.

    Parameters
    ----------
    kappa_L : float, optional
        Normalized coupling strength ``kappa * L`` [-]. Default is ``2.0``.
    phase_shift : float, optional
        Central phase shift ``Phi`` [rad]. Default is ``pi/2``.
    kappa_sign : float, optional
        Sign of ``kappa`` (``+1`` or ``-1``). Default is ``+1``.
    n_periods : int, optional
        Number of grating periods. Default is ``40``.
    cells_per_period : int, optional
        Cells per period ``Lambda``. Default is ``24``.
    pad_cells : int, optional
        Index-matched cladding buffer cells outside each grating facet. Default is ``16``.
    cfln_med : float, optional
        Courant number in medium. Default is ``0.90``.
    gain_g0 : float, optional
        Linear power gain ``g_0`` [1/m]. Default is ``0.0``.
    n_cycles : float, optional
        Total simulated optical cycles. Default is ``175.0``.
    excite_mode : {'auto', 'center', 'asymmetric'}, optional
        Source placement strategy. Default is ``'auto'``.

    Returns
    -------
    res_dict : dict
        Extracted complex resonance frequency, decay rate, normalized ``alpha_L``
        and ``delta_L``, field snapshot, wall time, and memory usage.
    """
    t_wall0 = time.perf_counter()
    solver, meta = build_1d_dfb_solver(
        kappa_L=kappa_L,
        phase_shift=phase_shift,
        kappa_sign=kappa_sign,
        n_periods=n_periods,
        cells_per_period=cells_per_period,
        pad_cells=pad_cells,
        cfln_med=cfln_med,
        gain_g0=gain_g0,
        gain_I_sat=np.inf,
    )
    f_B = meta["f_B"]
    T0 = meta["T0"]
    L = meta["L"]
    n0 = meta["n0"]
    dt = meta["dt"]
    n_steps = int(float(n_cycles) * T0 / dt)
    est = preflight_dfb_solver(
        1, 1, meta["nz_total"], n_steps, n_poles=1 if gain_g0 != 0 else 0, label="1D DFB harminv"
    )

    iz_c = meta["iz_center"]
    iz_3 = meta["iz_third"]
    t0 = 15.0 * T0
    tau = 4.0 * T0
    trace = np.zeros(n_steps, dtype=float)
    snap_step = int(95.0 * T0 / dt)
    ex_profile = np.zeros(solver.Nz, dtype=float)

    is_qws = abs(float(phase_shift) - 0.5 * np.pi) < 1e-9
    for n in range(n_steps):
        t = n * dt
        if t < 3.5 * t0:
            env = np.exp(-0.5 * ((t - t0) / tau) ** 2) * np.sin(
                2.0 * np.pi * f_B * (t - t0)
            )
            if excite_mode == "center" or (excite_mode == "auto" and is_qws):
                solver.E[:, :, iz_c, "x"] += env
            elif excite_mode == "asymmetric" or (excite_mode == "auto" and abs(phase_shift) < 1e-9):
                solver.E[:, :, iz_3, "x"] += env
            else:
                solver.E[:, :, iz_c, "x"] += env
                solver.E[:, :, iz_3, "x"] += 0.5 * env
        solver.one_step()
        trace[n] = float(solver.E[0, 0, iz_3, "x"])
        if n == snap_step:
            ex_profile = np.asarray(solver.E[0, 0, :, "x"], dtype=float).copy()

    i_start = int(4.0 * t0 / dt)
    res = harminv(
        trace[i_start:],
        dt,
        fmin=0.91 * f_B,
        fmax=1.09 * f_B,
        max_rel_error=1.5e-2,
        min_rel_amplitude=1e-3,
    )
    if len(res) == 0:
        raise RuntimeError("harminv returned no modes in the DFB stopband window.")

    order = np.argsort(res.decay_rate)
    i0 = int(order[0])
    alpha_L = float(res.decay_rate[i0] * (n0 / c0) * L)
    delta_L = float(2.0 * np.pi * n0 * (res.frequency[i0] - f_B) / c0 * L)

    second_mode = None
    if len(order) > 1:
        i1 = int(order[1])
        second_mode = {
            "f_hz": float(res.frequency[i1]),
            "decay_rate": float(res.decay_rate[i1]),
            "alpha_L": float(res.decay_rate[i1] * (n0 / c0) * L),
            "delta_L": float(2.0 * np.pi * n0 * (res.frequency[i1] - f_B) / c0 * L),
        }

    wall_s = time.perf_counter() - t_wall0
    return {
        "f_hz": float(res.frequency[i0]),
        "decay_rate": float(res.decay_rate[i0]),
        "Q": float(res.Q[i0]),
        "error_hz": float(res.error[i0]),
        "alpha_L": alpha_L,
        "delta_L": delta_L,
        "second_mode": second_mode,
        "ex_profile": ex_profile,
        "z_nodes": meta["z_nodes"],
        "iz_center": iz_c,
        "L": L,
        "wall_time_s": float(wall_s),
        "rss_mb": _rss_mb(),
        "estimate_ram_mb": float(est["ram_bytes"]) / (1024.0**2),
        "n_steps": n_steps,
    }


def run_task_00_cwt_and_linear_gain(
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run Study 1 & 2: CWT dispersion curves + 1D and 2D FIT linear-gain threshold extraction.

    Parameters
    ----------
    quick : bool, optional
        If True, uses a reduced resolution/sweep for fast pytest execution.
    force : bool, optional
        If False and cached JSON/NPZ exist, loads and returns cached results.

    Returns
    -------
    summary : dict
        JSON-serializable summary of CWT and linear-gain FIT results.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "cwt_and_linear_gain.json"
    npz_path = DATA_DIR / "cwt_and_linear_gain.npz"
    if not force and not quick and json_path.exists() and npz_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))

    t_start = time.perf_counter()
    # 1. Dense CWT curves for plotting
    kL_dense = np.linspace(0.4, 4.2, 77)
    cwt_uni_a = np.zeros_like(kL_dense)
    cwt_uni_d = np.zeros_like(kL_dense)
    cwt_uni_a_asymp = np.zeros_like(kL_dense)
    cwt_uni_d_asymp = np.zeros_like(kL_dense)
    cwt_qws_a = np.zeros_like(kL_dense)
    cwt_qws_a_asymp = np.zeros_like(kL_dense)
    cwt_qws_side_a = np.zeros_like(kL_dense)
    cwt_qws_side_d = np.zeros_like(kL_dense)

    for i, kL in enumerate(kL_dense):
        u1 = solve_cwt_uniform(float(kL), m_order=1)
        q0 = solve_cwt_qws(float(kL), m_order=0)
        q1 = solve_cwt_qws(float(kL), m_order=1)
        cwt_uni_a[i] = u1["alpha_th_L"]
        cwt_uni_d[i] = u1["delta_th_L"]
        cwt_uni_a_asymp[i] = u1["alpha_asymp_L"]
        cwt_uni_d_asymp[i] = u1["delta_asymp_L"]
        cwt_qws_a[i] = q0["alpha_th_L"]
        cwt_qws_a_asymp[i] = q0["alpha_asymp_L"]
        cwt_qws_side_a[i] = q1["alpha_th_L"]
        cwt_qws_side_d[i] = q1["delta_th_L"]

    # 2. 1D FIT linear-gain zero-crossing sweep at kL = 2.0
    cpp = 20 if quick else DEFAULT_CPP
    n_per = 30 if quick else DEFAULT_N_PERIODS
    n_cyc = 135.0 if quick else 175.0
    kL_ref = 2.0
    L_ref = n_per * DEFAULT_LAMBDA
    cwt_qws_2 = solve_cwt_qws(kL_ref, m_order=0)
    cwt_qws_side_2 = solve_cwt_qws(kL_ref, m_order=1)
    cwt_uni_2 = solve_cwt_uniform(kL_ref, m_order=1)
    g_th_qws_cwt = 2.0 * cwt_qws_2["alpha_th_L"] / L_ref
    g_th_uni_cwt = 2.0 * cwt_uni_2["alpha_th_L"] / L_ref

    gain_ratios = np.array([0.0, 0.4, 0.8, 0.95, 1.05], dtype=float)
    qws_decay_vs_g = np.zeros_like(gain_ratios)
    qws_delta_vs_g = np.zeros_like(gain_ratios)
    uni_decay_vs_g = np.zeros_like(gain_ratios)
    uni_delta_vs_g = np.zeros_like(gain_ratios)
    fit_runs_log: list[dict[str, Any]] = []

    for i, r in enumerate(gain_ratios):
        rq = run_1d_harminv_resonance(
            kappa_L=kL_ref,
            phase_shift=0.5 * np.pi,
            n_periods=n_per,
            cells_per_period=cpp,
            gain_g0=float(r * g_th_qws_cwt),
            n_cycles=n_cyc,
        )
        qws_decay_vs_g[i] = rq["decay_rate"]
        qws_delta_vs_g[i] = rq["delta_L"]
        fit_runs_log.append({"run": f"1d_qws_gain_r{r:.2f}", "wall_s": rq["wall_time_s"], "rss_mb": rq["rss_mb"]})

        ru = run_1d_harminv_resonance(
            kappa_L=kL_ref,
            phase_shift=0.0,
            n_periods=n_per,
            cells_per_period=cpp,
            gain_g0=float(r * g_th_uni_cwt),
            n_cycles=n_cyc,
        )
        uni_decay_vs_g[i] = ru["decay_rate"]
        uni_delta_vs_g[i] = abs(ru["delta_L"])
        fit_runs_log.append({"run": f"1d_uni_gain_r{r:.2f}", "wall_s": ru["wall_time_s"], "rss_mb": ru["rss_mb"]})

    p_qws = np.polyfit(gain_ratios * g_th_qws_cwt, qws_decay_vs_g, 1)
    g_th_qws_zc = float(-p_qws[1] / p_qws[0])
    alpha_th_qws_zc_L = 0.5 * g_th_qws_zc * L_ref
    delta_th_qws_zc_L = float(np.interp(g_th_qws_zc, gain_ratios * g_th_qws_cwt, qws_delta_vs_g))

    p_uni = np.polyfit(gain_ratios * g_th_uni_cwt, uni_decay_vs_g, 1)
    g_th_uni_zc = float(-p_uni[1] / p_uni[0])
    alpha_th_uni_zc_L = 0.5 * g_th_uni_zc * L_ref
    delta_th_uni_zc_L = float(np.interp(g_th_uni_zc, gain_ratios * g_th_uni_cwt, uni_delta_vs_g))

    # 3. 1D FIT sweep across kappa_L in [0.5 .. 4.0] for both gratings
    kL_points = np.array([0.5, 1.0, 2.0, 3.0] if quick else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], dtype=float)
    fit_qws_a = np.zeros_like(kL_points)
    fit_qws_d = np.zeros_like(kL_points)
    cwt_qws_pts_a = np.zeros_like(kL_points)
    cwt_qws_pts_d = np.zeros_like(kL_points)
    fit_uni_a = np.zeros_like(kL_points)
    fit_uni_d = np.zeros_like(kL_points)
    cwt_uni_pts_a = np.zeros_like(kL_points)
    cwt_uni_pts_d = np.zeros_like(kL_points)

    for i, kL in enumerate(kL_points):
        cq = solve_cwt_qws(float(kL), m_order=0)
        cu = solve_cwt_uniform(float(kL), m_order=1)
        cwt_qws_pts_a[i] = cq["alpha_th_L"]
        cwt_qws_pts_d[i] = cq["delta_th_L"]
        cwt_uni_pts_a[i] = cu["alpha_th_L"]
        cwt_uni_pts_d[i] = cu["delta_th_L"]

        rq = run_1d_harminv_resonance(
            kappa_L=float(kL),
            phase_shift=0.5 * np.pi,
            n_periods=n_per,
            cells_per_period=cpp,
            n_cycles=n_cyc,
        )
        fit_qws_a[i] = rq["alpha_L"]
        fit_qws_d[i] = rq["delta_L"]
        fit_runs_log.append({"run": f"1d_qws_kL{kL:.1f}", "wall_s": rq["wall_time_s"], "rss_mb": rq["rss_mb"]})

        ru = run_1d_harminv_resonance(
            kappa_L=float(kL),
            phase_shift=0.0,
            n_periods=n_per,
            cells_per_period=cpp,
            n_cycles=n_cyc,
        )
        if ru["second_mode"] is not None:
            fit_uni_a[i] = 0.5 * (ru["alpha_L"] + ru["second_mode"]["alpha_L"])
            fit_uni_d[i] = 0.5 * abs(ru["delta_L"] - ru["second_mode"]["delta_L"])
        else:
            fit_uni_a[i] = ru["alpha_L"]
            fit_uni_d[i] = abs(ru["delta_L"])
        fit_runs_log.append({"run": f"1d_uni_kL{kL:.1f}", "wall_s": ru["wall_time_s"], "rss_mb": ru["rss_mb"]})

    # 4. 2D TE slab-waveguide DFB check at kL = 2.0 across linear gain in core
    slab_ratios = np.array([0.0, 0.5, 1.0] if quick else [0.0, 0.5, 0.9, 1.05], dtype=float)
    slab_decays = np.zeros_like(slab_ratios)
    slab_deltas = np.zeros_like(slab_ratios)
    slab_freqs = np.zeros_like(slab_ratios)
    slab_ey_map = None
    slab_eps_map = None
    slab_x_um = None
    slab_z_um = None
    slab_meta_ref: dict[str, Any] = {}

    for i, r_s in enumerate(slab_ratios):
        t_w2 = time.perf_counter()
        # Build once to get Gamma_core, n_eff, L
        _, m_tmp = build_2d_slab_dfb_solver(kappa_L=kL_ref, n_periods=24 if quick else 30)
        g_th_core_cwt = (
            (2.0 * cwt_qws_2["alpha_th_L"] / m_tmp["L"])
            * (m_tmp["n_eff"] / m_tmp["n_core"])
            / m_tmp["Gamma_core"]
        )
        slv2d, m2d = build_2d_slab_dfb_solver(
            kappa_L=kL_ref,
            n_periods=24 if quick else 30,
            gain_g0_core=float(r_s * g_th_core_cwt),
        )
        slab_meta_ref = m2d
        f_B2 = m2d["f_B"]
        T0_2 = m2d["T0"]
        dt2 = m2d["dt"]
        n_steps_2d = int((125.0 if quick else 150.0) * T0_2 / dt2)
        preflight_dfb_solver(
            m2d["nx"], 1, m2d["nz_total"], n_steps_2d, n_poles=1 if r_s > 0 else 0, label="2D Slab DFB"
        )
        iz_src2 = m2d["iz_center"]
        iz_prb2 = m2d["iz_third"]
        mode_x = m2d["mode_x"]
        t0_2 = 15.0 * T0_2
        tau_2 = 4.5 * T0_2
        trace2 = np.zeros(n_steps_2d, dtype=float)
        snap_step2 = int(95.0 * T0_2 / dt2)
        for n in range(n_steps_2d):
            t = n * dt2
            if t < 3.5 * t0_2:
                env = np.exp(-0.5 * ((t - t0_2) / tau_2) ** 2) * np.sin(
                    2.0 * np.pi * f_B2 * (t - t0_2)
                )
                slv2d.E[:, :, iz_src2, "y"] += (env * mode_x)[:, None]
            slv2d.one_step()
            trace2[n] = float(np.dot(slv2d.E[:, 0, iz_prb2, "y"], mode_x))
            if i == 0 and n == snap_step2:
                slab_ey_map = np.asarray(slv2d.E[:, 0, :, "y"], dtype=float).copy()
                slab_eps_map = m2d["eps_2d"].copy()
                slab_x_um = m2d["x_nodes"] * 1e6
                slab_z_um = m2d["z_nodes"] * 1e6

        i_start2 = int(4.0 * t0_2 / dt2)
        res2 = harminv(
            trace2[i_start2:],
            dt2,
            fmin=0.96 * f_B2,
            fmax=1.04 * f_B2,
            max_rel_error=2e-2,
            min_rel_amplitude=1e-3,
        )
        idx_2 = int(np.argmin(np.abs(res2.frequency - f_B2)))
        slab_decays[i] = float(res2.decay_rate[idx_2])
        slab_freqs[i] = float(res2.frequency[idx_2])
        slab_deltas[i] = float(
            2.0 * np.pi * m2d["n_g"] * (res2.frequency[idx_2] - f_B2) / c0 * m2d["L"]
        )
        fit_runs_log.append(
            {"run": f"2d_slab_r{r_s:.2f}", "wall_s": time.perf_counter() - t_w2, "rss_mb": _rss_mb()}
        )

    p_slab = np.polyfit(slab_ratios, slab_decays, 1)
    r_zc_slab = float(-p_slab[1] / p_slab[0])
    alpha_th_2d_L = float(r_zc_slab * cwt_qws_2["alpha_th_L"])
    delta_th_2d_L = float(np.interp(r_zc_slab, slab_ratios, slab_deltas))

    idx_k2 = int(np.argmin(np.abs(kL_points - 2.0)))
    summary = {
        "kappa_L_ref": kL_ref,
        "cwt_qws_kL2_alpha_L": round(cwt_qws_2["alpha_th_L"], 5),
        "cwt_qws_kL2_delta_L": round(cwt_qws_2["delta_th_L"], 5),
        "cwt_qws_kL2_side_alpha_L": round(cwt_qws_side_2["alpha_th_L"], 5),
        "cwt_qws_kL2_side_delta_L": round(cwt_qws_side_2["delta_th_L"], 5),
        "cwt_qws_kL2_side_gain_margin": round(
            cwt_qws_side_2["alpha_th_L"] - cwt_qws_2["alpha_th_L"], 5
        ),
        "cwt_uni_kL2_alpha_L": round(cwt_uni_2["alpha_th_L"], 5),
        "cwt_uni_kL2_delta_L": round(cwt_uni_2["delta_th_L"], 5),
        "fit_qws_kL2_alpha_L": round(float(fit_qws_a[idx_k2]), 5),
        "fit_qws_kL2_delta_L": round(float(fit_qws_d[idx_k2]), 5),
        "fit_qws_zc_alpha_L": round(alpha_th_qws_zc_L, 5),
        "fit_qws_zc_delta_L": round(delta_th_qws_zc_L, 5),
        "fit_uni_kL2_alpha_L": round(float(fit_uni_a[idx_k2]), 5),
        "fit_uni_kL2_delta_L": round(float(fit_uni_d[idx_k2]), 5),
        "fit_uni_zc_alpha_L": round(alpha_th_uni_zc_L, 5),
        "fit_uni_zc_delta_L": round(delta_th_uni_zc_L, 5),
        "rel_err_qws_kL2_alpha_pct": round(
            abs(float(fit_qws_a[idx_k2]) - cwt_qws_2["alpha_th_L"])
            / cwt_qws_2["alpha_th_L"]
            * 100.0,
            3,
        ),
        "rel_err_uni_kL2_alpha_pct": round(
            abs(float(fit_uni_a[idx_k2]) - cwt_uni_2["alpha_th_L"])
            / cwt_uni_2["alpha_th_L"]
            * 100.0,
            3,
        ),
        "rel_err_uni_kL2_delta_pct": round(
            abs(float(fit_uni_d[idx_k2]) - cwt_uni_2["delta_th_L"])
            / cwt_uni_2["delta_th_L"]
            * 100.0,
            3,
        ),
        "max_rel_err_qws_sweep_pct": round(
            float(np.max(np.abs(fit_qws_a - cwt_qws_pts_a) / cwt_qws_pts_a) * 100.0), 3
        ),
        "max_rel_err_uni_sweep_pct": round(
            float(np.max(np.abs(fit_uni_a - cwt_uni_pts_a) / cwt_uni_pts_a) * 100.0), 3
        ),
        "slab_2d_n_eff": round(slab_meta_ref["n_eff"], 5),
        "slab_2d_n_g": round(slab_meta_ref["n_g"], 5),
        "slab_2d_Gamma_core": round(slab_meta_ref["Gamma_core"], 5),
        "slab_2d_alpha_th_L": round(alpha_th_2d_L, 5),
        "slab_2d_delta_th_L": round(delta_th_2d_L, 5),
        "slab_2d_rel_err_alpha_pct": round(
            abs(alpha_th_2d_L - cwt_qws_2["alpha_th_L"]) / cwt_qws_2["alpha_th_L"] * 100.0,
            3,
        ),
        "kL_points": [float(x) for x in kL_points],
        "fit_qws_alpha_L": [round(float(x), 5) for x in fit_qws_a],
        "cwt_qws_alpha_L": [round(float(x), 5) for x in cwt_qws_pts_a],
        "fit_uni_alpha_L": [round(float(x), 5) for x in fit_uni_a],
        "cwt_uni_alpha_L": [round(float(x), 5) for x in cwt_uni_pts_a],
        "fit_uni_delta_L": [round(float(x), 5) for x in fit_uni_d],
        "cwt_uni_delta_L": [round(float(x), 5) for x in cwt_uni_pts_d],
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "fit_runs_log": fit_runs_log,
    }

    if not quick:
        np.savez_compressed(
            npz_path,
            kL_dense=kL_dense,
            cwt_uni_a=cwt_uni_a,
            cwt_uni_d=cwt_uni_d,
            cwt_uni_a_asymp=cwt_uni_a_asymp,
            cwt_uni_d_asymp=cwt_uni_d_asymp,
            cwt_qws_a=cwt_qws_a,
            cwt_qws_a_asymp=cwt_qws_a_asymp,
            cwt_qws_side_a=cwt_qws_side_a,
            cwt_qws_side_d=cwt_qws_side_d,
            gain_ratios=gain_ratios,
            qws_decay_vs_g=qws_decay_vs_g,
            qws_delta_vs_g=qws_delta_vs_g,
            uni_decay_vs_g=uni_decay_vs_g,
            uni_delta_vs_g=uni_delta_vs_g,
            kL_points=kL_points,
            fit_qws_a=fit_qws_a,
            fit_qws_d=fit_qws_d,
            cwt_qws_pts_a=cwt_qws_pts_a,
            cwt_qws_pts_d=cwt_qws_pts_d,
            fit_uni_a=fit_uni_a,
            fit_uni_d=fit_uni_d,
            cwt_uni_pts_a=cwt_uni_pts_a,
            cwt_uni_pts_d=cwt_uni_pts_d,
            slab_ratios=slab_ratios,
            slab_decays=slab_decays,
            slab_deltas=slab_deltas,
            slab_ey_map=slab_ey_map,
            slab_eps_map=slab_eps_map,
            slab_x_um=slab_x_um,
            slab_z_um=slab_z_um,
            slab_mode_x=slab_meta_ref["mode_x"],
        )
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_task_01_convergence(
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run Study 2b: Grid convergence in cells per period ``N_Lambda`` + Richardson extrapolation.

    Parameters
    ----------
    quick : bool, optional
        If True, runs 3 resolution levels ``(12, 16, 24)`` on a shorter grating.
    force : bool, optional
        If False and cached JSON/NPZ exist, loads and returns cached results.

    Returns
    -------
    summary : dict
        Convergence and Richardson-extrapolation summary dictionary.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "convergence.json"
    npz_path = DATA_DIR / "convergence.npz"
    if not force and not quick and json_path.exists() and npz_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))

    t_start = time.perf_counter()
    cpp_list = [12, 16, 24] if quick else [12, 16, 24, 32, 48]
    n_per = 30 if quick else DEFAULT_N_PERIODS
    n_cyc = 140.0 if quick else 180.0
    kL = 2.0

    cwt_qws = solve_cwt_qws(kL, m_order=0)
    cwt_uni = solve_cwt_uniform(kL, m_order=1)

    qws_a = np.zeros(len(cpp_list), dtype=float)
    qws_d = np.zeros(len(cpp_list), dtype=float)
    uni_a = np.zeros(len(cpp_list), dtype=float)
    uni_d = np.zeros(len(cpp_list), dtype=float)
    runs_log: list[dict[str, Any]] = []

    for i, cpp in enumerate(cpp_list):
        rq = run_1d_harminv_resonance(
            kappa_L=kL,
            phase_shift=0.5 * np.pi,
            n_periods=n_per,
            cells_per_period=int(cpp),
            pad_cells=int(cpp),
            cfln_med=0.50,
            n_cycles=n_cyc,
        )
        qws_a[i] = rq["alpha_L"]
        qws_d[i] = rq["delta_L"]
        runs_log.append({"run": f"conv_qws_cpp{cpp}", "wall_s": rq["wall_time_s"], "rss_mb": rq["rss_mb"]})

        ru = run_1d_harminv_resonance(
            kappa_L=kL,
            phase_shift=0.0,
            n_periods=n_per,
            cells_per_period=int(cpp),
            pad_cells=int(cpp),
            cfln_med=0.50,
            n_cycles=n_cyc,
        )
        if ru["second_mode"] is not None:
            uni_a[i] = 0.5 * (ru["alpha_L"] + ru["second_mode"]["alpha_L"])
            uni_d[i] = 0.5 * abs(ru["delta_L"] - ru["second_mode"]["delta_L"])
        else:
            uni_a[i] = ru["alpha_L"]
            uni_d[i] = abs(ru["delta_L"])
        runs_log.append({"run": f"conv_uni_cpp{cpp}", "wall_s": ru["wall_time_s"], "rss_mb": ru["rss_mb"]})

    # Richardson extrapolation (p = 2) using finest pair with ratio r
    r_pair = float(cpp_list[-1]) / float(cpp_list[-2])
    qws_a_rich = float((r_pair**2 * qws_a[-1] - qws_a[-2]) / (r_pair**2 - 1.0))
    qws_d_rich = float((r_pair**2 * qws_d[-1] - qws_d[-2]) / (r_pair**2 - 1.0))
    uni_a_rich = float((r_pair**2 * uni_a[-1] - uni_a[-2]) / (r_pair**2 - 1.0))
    uni_d_rich = float((r_pair**2 * uni_d[-1] - uni_d[-2]) / (r_pair**2 - 1.0))

    err_qws_a_cwt = np.abs(qws_a - cwt_qws["alpha_th_L"])
    err_qws_d_cwt = np.abs(qws_d - cwt_qws["delta_th_L"])
    err_uni_a_cwt = np.abs(uni_a - cwt_uni["alpha_th_L"])
    err_uni_d_cwt = np.abs(uni_d - cwt_uni["delta_th_L"])

    err_qws_a_rich = np.abs(qws_a - qws_a_rich)
    err_uni_a_rich = np.abs(uni_a - uni_a_rich)

    # Observed log-log slope across resolutions
    p_obs_qws_a = float(
        -np.polyfit(np.log(np.array(cpp_list[:-1], dtype=float)), np.log(err_qws_a_rich[:-1]), 1)[0]
    )
    p_obs_qws_d = float(
        -np.polyfit(np.log(np.array(cpp_list, dtype=float)), np.log(err_qws_d_cwt), 1)[0]
    )
    p_obs_uni_a = float(
        -np.polyfit(np.log(np.array(cpp_list[:-1], dtype=float)), np.log(err_uni_a_rich[:-1]), 1)[0]
    )

    summary = {
        "cpp_list": [int(x) for x in cpp_list],
        "cpw_list": [int(2 * x) for x in cpp_list],
        "qws_alpha_L": [round(float(x), 6) for x in qws_a],
        "qws_delta_L": [round(float(x), 6) for x in qws_d],
        "uni_alpha_L": [round(float(x), 6) for x in uni_a],
        "uni_delta_L": [round(float(x), 6) for x in uni_d],
        "qws_alpha_L_richardson": round(qws_a_rich, 6),
        "qws_delta_L_richardson": round(qws_d_rich, 6),
        "uni_alpha_L_richardson": round(uni_a_rich, 6),
        "uni_delta_L_richardson": round(uni_d_rich, 6),
        "cwt_qws_alpha_L": round(cwt_qws["alpha_th_L"], 6),
        "cwt_uni_alpha_L": round(cwt_uni["alpha_th_L"], 6),
        "cwt_uni_delta_L": round(cwt_uni["delta_th_L"], 6),
        "finest_qws_alpha_rel_err_pct": round(
            float(err_qws_a_cwt[-1] / cwt_qws["alpha_th_L"] * 100.0), 4
        ),
        "finest_uni_alpha_rel_err_pct": round(
            float(err_uni_a_cwt[-1] / cwt_uni["alpha_th_L"] * 100.0), 4
        ),
        "finest_uni_delta_rel_err_pct": round(
            float(err_uni_d_cwt[-1] / cwt_uni["delta_th_L"] * 100.0), 4
        ),
        "p_obs_qws_alpha": round(p_obs_qws_a, 2),
        "p_obs_qws_delta": round(p_obs_qws_d, 2),
        "p_obs_uni_alpha": round(p_obs_uni_a, 2),
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "fit_runs_log": runs_log,
    }

    if not quick:
        np.savez_compressed(
            npz_path,
            cpp_list=np.array(cpp_list, dtype=int),
            qws_a=qws_a,
            qws_d=qws_d,
            uni_a=uni_a,
            uni_d=uni_d,
            err_qws_a_cwt=err_qws_a_cwt,
            err_qws_d_cwt=err_qws_d_cwt,
            err_uni_a_cwt=err_uni_a_cwt,
            err_uni_d_cwt=err_uni_d_cwt,
            err_qws_a_rich=err_qws_a_rich,
            err_uni_a_rich=err_uni_a_rich,
        )
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_task_02_lasing_dynamics(
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run Study 3: Above-threshold saturable-gain FIT lasing, SHB profile, and L-I curve.

    Parameters
    ----------
    quick : bool, optional
        If True, uses fewer optical cycles and pump points for fast pytest execution.
    force : bool, optional
        If False and cached JSON/NPZ exist, loads and returns cached results.

    Returns
    -------
    summary : dict
        Lasing spectrum, spatial hole burning profile, and L-I curve summary dictionary.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "lasing_dynamics.json"
    npz_path = DATA_DIR / "lasing_dynamics.npz"
    if not force and not quick and json_path.exists() and npz_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))

    t_start = time.perf_counter()
    n_per = 30
    cpp = 20
    kL = 2.0
    r_main = 1.60
    n_cyc_main = 420.0 if quick else 600.0
    n_tail_cyc = 90.0 if quick else 130.0
    L = n_per * DEFAULT_LAMBDA

    cwt_qws = solve_cwt_qws(kL, m_order=0)
    cwt_uni = solve_cwt_uniform(kL, m_order=1)
    g_th_qws = 2.0 * cwt_qws["alpha_th_L"] / L
    g_th_uni = 2.0 * cwt_uni["alpha_th_L"] / L

    runs_log: list[dict[str, Any]] = []

    def _run_lasing_case(
        phase_shift: float,
        g_th: float,
        r_pump: float,
        n_cycles: float,
        record_frames: bool = False,
        label: str = "lasing",
    ) -> dict[str, Any]:
        tw0 = time.perf_counter()
        solver, meta = build_1d_dfb_solver(
            kappa_L=kL,
            phase_shift=phase_shift,
            n_periods=n_per,
            cells_per_period=cpp,
            gain_g0=float(r_pump * g_th),
            gain_I_sat=DEFAULT_I_SAT,
            gain_gamma_factor=1.5,
            gain_tau_cycles=6.0,
        )
        dt = meta["dt"]
        T0 = meta["T0"]
        f_B = meta["f_B"]
        n0 = meta["n0"]
        n_steps = int(float(n_cycles) * T0 / dt)
        n_tail = int(float(n_tail_cyc) * T0 / dt)
        preflight_dfb_solver(1, 1, meta["nz_total"], n_steps, n_poles=1, label=label)

        iz_src = meta["iz_center"] if abs(phase_shift - 0.5 * np.pi) < 1e-9 else meta["iz_third"]
        iz_out = meta["iz_out_right"]
        t0 = 12.0 * T0
        tau = 3.5 * T0
        trace_out = np.zeros(n_steps, dtype=float)
        I_z_accum = np.zeros(solver.Nz, dtype=float)
        count_tail = 0

        frame_steps = (
            np.linspace(int(20.0 * T0 / dt), n_steps - 1, 32, dtype=int)
            if record_frames
            else np.array([], dtype=int)
        )
        frame_set = set(int(s) for s in frame_steps)
        frames_I_env: list[np.ndarray] = []
        frames_t_ps: list[float] = []
        window_I = np.zeros(solver.Nz, dtype=float)
        n_win = max(1, int(2.0 * T0 / dt))

        for n in range(n_steps):
            t = n * dt
            if t < 3.5 * t0:
                # Weak seed pulse (peak intensity ~ 5e-3 * I_sat << I_sat)
                env = 200.0 * np.exp(-0.5 * ((t - t0) / tau) ** 2) * np.sin(
                    2.0 * np.pi * f_B * (t - t0)
                )
                solver.E[:, :, iz_src, "x"] += env
            solver.one_step()
            ex_1d = np.asarray(solver.E[0, 0, :, "x"], dtype=float)
            trace_out[n] = float(ex_1d[iz_out])
            I_inst = c0 * eps_0 * n0 * (ex_1d**2)
            window_I = (1.0 - 1.0 / n_win) * window_I + (1.0 / n_win) * I_inst
            if n in frame_set:
                frames_I_env.append(uniform_filter1d(window_I, size=cpp).copy())
                frames_t_ps.append(float(t * 1e12))
            if n >= n_steps - n_tail:
                I_z_accum += I_inst
                count_tail += 1

        I_z = I_z_accum / max(1, count_tail)
        I_env = uniform_filter1d(I_z, size=cpp)
        I_out = float(c0 * eps_0 * n0 * np.mean(trace_out[-n_tail:] ** 2))

        sig = trace_out[-n_tail:] * np.hanning(n_tail)
        n_fft = n_tail * 8
        freqs = np.fft.rfftfreq(n_fft, d=dt)
        spec = np.abs(np.fft.rfft(sig, n=n_fft)) ** 2
        band = (freqs >= 0.90 * f_B) & (freqs <= 1.10 * f_B)
        f_band = freqs[band]
        s_band = spec[band] / max(float(np.max(spec[band])), 1e-30)
        delta_L_axis = 2.0 * np.pi * n0 * (f_band - f_B) / c0 * L

        runs_log.append(
            {"run": label, "wall_s": time.perf_counter() - tw0, "rss_mb": _rss_mb()}
        )
        return {
            "trace_out": trace_out,
            "dt": dt,
            "T0": T0,
            "z_nodes": meta["z_nodes"],
            "L": L,
            "pad_cells": meta["pad_cells"],
            "nz_grat": meta["nz_grat"],
            "iz_center": meta["iz_center"],
            "I_z": I_z,
            "I_env": I_env,
            "I_out": I_out,
            "delta_L_axis": delta_L_axis,
            "spec_norm": s_band,
            "frames_I_env": np.array(frames_I_env, dtype=float) if record_frames else None,
            "frames_t_ps": np.array(frames_t_ps, dtype=float) if record_frames else None,
        }

    # 1. Single-mode QWS vs two-mode uniform DFB at r = 1.60
    qws_main = _run_lasing_case(
        0.5 * np.pi, g_th_qws, r_main, n_cyc_main, record_frames=not quick, label="lasing_qws_r1.6"
    )
    uni_main = _run_lasing_case(
        0.0, g_th_uni, r_main, n_cyc_main, record_frames=not quick, label="lasing_uni_r1.6"
    )

    # CWT longitudinal profile comparison on [-0.5, +0.5]
    z_grat_mask = (qws_main["z_nodes"] >= -0.5 * L) & (qws_main["z_nodes"] <= 0.5 * L)
    u_grat = qws_main["z_nodes"][z_grat_mask] / L
    cwt_prof_qws = cwt_mode_profile(u_grat, kappa_L=kL, phase_shift=0.5 * np.pi)
    cwt_prof_uni = cwt_mode_profile(u_grat, kappa_L=kL, phase_shift=0.0)

    fit_I_norm_qws = qws_main["I_env"][z_grat_mask] / max(qws_main["I_out"], 1e-30)
    fit_I_raw_qws = qws_main["I_z"][z_grat_mask] / max(qws_main["I_out"], 1e-30)
    fit_I_norm_uni = uni_main["I_env"][z_grat_mask] / max(uni_main["I_out"], 1e-30)
    g_sat_ratio_qws = 1.0 / (1.0 + qws_main["I_env"][z_grat_mask] / DEFAULT_I_SAT)

    i_mid_g = len(u_grat) // 2
    peak_to_edge_fit_qws = float(fit_I_norm_qws[i_mid_g] / max(fit_I_norm_qws[-1], 1e-30))
    peak_to_edge_cwt_qws = float(cwt_prof_qws["peak_to_edge"])

    # Spectral peaks
    d_ax = qws_main["delta_L_axis"]
    qws_peak_delta_L = float(d_ax[int(np.argmax(qws_main["spec_norm"]))])
    # Side-mode suppression ratio in QWS spectrum (ratio of main peak at |dL|<1 to max outside |dL|>2.0)
    side_mask = np.abs(d_ax) > 2.0
    qws_smsr_db = float(
        10.0 * np.log10(1.0 / max(float(np.max(qws_main["spec_norm"][side_mask])), 1e-12))
    )

    pos_m = d_ax > 0.5
    neg_m = d_ax < -0.5
    uni_peak_pos_delta_L = float(d_ax[pos_m][int(np.argmax(uni_main["spec_norm"][pos_m]))])
    uni_peak_neg_delta_L = float(d_ax[neg_m][int(np.argmax(uni_main["spec_norm"][neg_m]))])
    uni_secondary_ratio = float(
        min(
            np.max(uni_main["spec_norm"][pos_m]),
            np.max(uni_main["spec_norm"][neg_m]),
        )
        / max(
            np.max(uni_main["spec_norm"][pos_m]),
            np.max(uni_main["spec_norm"][neg_m]),
        )
    )

    # 2. Pump sweep (L-I curve) for the QWS DFB laser
    pump_pts = (
        np.array([0.8, 1.5, 2.0, 2.5], dtype=float)
        if quick
        else np.array([0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5], dtype=float)
    )
    I_out_over_Isat_fit = np.zeros_like(pump_pts)
    for i, r_p in enumerate(pump_pts):
        if abs(r_p - r_main) < 1e-9:
            I_out_over_Isat_fit[i] = qws_main["I_out"] / DEFAULT_I_SAT
            continue
        n_c_p = (460.0 if r_p < 1.35 else 380.0) if quick else (680.0 if r_p < 1.35 else 540.0)
        res_p = _run_lasing_case(
            0.5 * np.pi, g_th_qws, float(r_p), n_c_p, record_frames=False, label=f"li_qws_r{r_p:.2f}"
        )
        I_out_over_Isat_fit[i] = res_p["I_out"] / DEFAULT_I_SAT

    rigrod_dense = cwt_rigrod_output(np.linspace(0.4, 2.6, 111), kappa_L=kL, phase_shift=0.5 * np.pi)
    rigrod_pts = cwt_rigrod_output(pump_pts, kappa_L=kL, phase_shift=0.5 * np.pi)

    # Slope efficiency above threshold (fit on r >= 1.5)
    above = pump_pts >= 1.49
    slope_fit = float(np.polyfit(pump_pts[above] - 1.0, I_out_over_Isat_fit[above], 1)[0])
    slope_rigrod_sw = float(
        np.polyfit(pump_pts[above] - 1.0, rigrod_pts["I_out_rigrod_sw"][above], 1)[0]
    )

    idx_r2 = int(np.argmin(np.abs(pump_pts - 2.0)))
    summary = {
        "r_main": r_main,
        "qws_peak_delta_L": round(qws_peak_delta_L, 4),
        "qws_smsr_db": round(qws_smsr_db, 2),
        "uni_peak_pos_delta_L": round(uni_peak_pos_delta_L, 4),
        "uni_peak_neg_delta_L": round(uni_peak_neg_delta_L, 4),
        "uni_half_splitting_delta_L": round(0.5 * (uni_peak_pos_delta_L - uni_peak_neg_delta_L), 4),
        "cwt_uni_delta_L": round(cwt_uni["delta_th_L"], 4),
        "uni_secondary_mode_ratio": round(uni_secondary_ratio, 3),
        "qws_peak_to_edge_fit": round(peak_to_edge_fit_qws, 4),
        "qws_peak_to_edge_cwt": round(peak_to_edge_cwt_qws, 4),
        "qws_peak_to_edge_rel_diff_pct": round(
            abs(peak_to_edge_fit_qws - peak_to_edge_cwt_qws) / peak_to_edge_cwt_qws * 100.0, 2
        ),
        "qws_central_gain_compression": round(float(g_sat_ratio_qws[i_mid_g]), 4),
        "cwt_energy_balance_check": round(cwt_prof_qws["energy_balance_check"], 6),
        "pump_points": [float(x) for x in pump_pts],
        "I_out_over_Isat_fit": [round(float(x), 5) for x in I_out_over_Isat_fit],
        "I_out_over_Isat_rigrod_sw": [round(float(x), 5) for x in rigrod_pts["I_out_rigrod_sw"]],
        "I_out_over_Isat_linear_shb": [round(float(x), 5) for x in rigrod_pts["I_out_linear_shb"]],
        "I_out_r2_fit": round(float(I_out_over_Isat_fit[idx_r2]), 4),
        "I_out_r2_rigrod_sw": round(float(rigrod_pts["I_out_rigrod_sw"][idx_r2]), 4),
        "slope_fit": round(slope_fit, 4),
        "slope_rigrod_sw": round(slope_rigrod_sw, 4),
        "slope_rel_diff_pct": round(
            abs(slope_fit - slope_rigrod_sw) / slope_rigrod_sw * 100.0, 2
        ),
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "fit_runs_log": runs_log,
    }

    if not quick:
        stride_t = max(1, len(qws_main["trace_out"]) // 2400)
        t_ps_sub = np.arange(0, len(qws_main["trace_out"]), stride_t) * qws_main["dt"] * 1e12
        # Rolling RMS envelope of output trace
        win_t = max(1, int(3.0 * qws_main["T0"] / qws_main["dt"]))
        env_qws_t = np.sqrt(
            2.0 * uniform_filter1d(qws_main["trace_out"] ** 2, size=win_t)
        )[::stride_t]
        env_uni_t = np.sqrt(
            2.0 * uniform_filter1d(uni_main["trace_out"] ** 2, size=win_t)
        )[::stride_t]

        np.savez_compressed(
            npz_path,
            t_ps_sub=t_ps_sub,
            env_qws_t=env_qws_t,
            env_uni_t=env_uni_t,
            delta_L_axis=d_ax,
            spec_qws=qws_main["spec_norm"],
            spec_uni=uni_main["spec_norm"],
            u_grat=u_grat,
            fit_I_norm_qws=fit_I_norm_qws,
            fit_I_raw_qws=fit_I_raw_qws,
            fit_I_norm_uni=fit_I_norm_uni,
            cwt_I_norm_qws=cwt_prof_qws["I_norm"],
            cwt_I_norm_uni=cwt_prof_uni["I_norm"],
            g_sat_ratio_qws=g_sat_ratio_qws,
            pump_pts=pump_pts,
            I_out_over_Isat_fit=I_out_over_Isat_fit,
            pump_dense=rigrod_dense["pump_ratios"],
            I_out_linear_dense=rigrod_dense["I_out_linear_shb"],
            I_out_rigrod_env_dense=rigrod_dense["I_out_rigrod_env"],
            I_out_rigrod_sw_dense=rigrod_dense["I_out_rigrod_sw"],
            frames_t_ps=qws_main["frames_t_ps"],
            frames_qws_I_env=qws_main["frames_I_env"][:, z_grat_mask] / DEFAULT_I_SAT,
            frames_uni_I_env=uni_main["frames_I_env"][:, z_grat_mask] / DEFAULT_I_SAT,
        )
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_task_03_mutations(
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run Study 4: Mutation checks (phase shift != pi/2, kappa sign flip, gain off).

    Parameters
    ----------
    quick : bool, optional
        If True, uses shorter ring-down runs for fast pytest execution.
    force : bool, optional
        If False and cached JSON/NPZ exist, loads and returns cached results.

    Returns
    -------
    summary : dict
        Mutation-check results dictionary.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "mutations.json"
    npz_path = DATA_DIR / "mutations.npz"
    if not force and not quick and json_path.exists() and npz_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))

    t_start = time.perf_counter()
    kL = 2.0
    n_per = 30 if quick else DEFAULT_N_PERIODS
    cpp = 20 if quick else DEFAULT_CPP
    n_cyc = 140.0 if quick else 180.0
    runs_log: list[dict[str, Any]] = []

    # 1. Phase-shift sweep Phi in [0, pi] (CWT curve + FIT points at 0, 45, 90, 135 deg)
    phi_dense_deg = np.linspace(0.0, 180.0, 73)
    cwt_phi_a = np.zeros_like(phi_dense_deg)
    cwt_phi_d = np.zeros_like(phi_dense_deg)
    for i, deg in enumerate(phi_dense_deg):
        s = solve_cwt_phase_shifted(kL, np.deg2rad(float(deg)))
        cwt_phi_a[i] = s["alpha_th_L"]
        cwt_phi_d[i] = s["delta_th_L"]

    phi_pts_deg = np.array([0.0, 45.0, 90.0, 135.0], dtype=float)
    fit_phi_a = np.zeros_like(phi_pts_deg)
    fit_phi_d = np.zeros_like(phi_pts_deg)
    cwt_pts_a = np.zeros_like(phi_pts_deg)
    cwt_pts_d = np.zeros_like(phi_pts_deg)

    for i, deg in enumerate(phi_pts_deg):
        phi_rad = float(np.deg2rad(deg))
        sc = solve_cwt_phase_shifted(kL, phi_rad)
        cwt_pts_a[i] = sc["alpha_th_L"]
        cwt_pts_d[i] = sc["delta_th_L"]
        rf = run_1d_harminv_resonance(
            kappa_L=kL,
            phase_shift=phi_rad,
            n_periods=n_per,
            cells_per_period=cpp,
            n_cycles=n_cyc,
            excite_mode="both",
        )
        fit_phi_a[i] = rf["alpha_L"]
        fit_phi_d[i] = rf["delta_L"]
        runs_log.append(
            {"run": f"mut_phi_{deg:.0f}deg", "wall_s": rf["wall_time_s"], "rss_mb": rf["rss_mb"]}
        )

    # 2. Coupling-sign mutation (kappa > 0 vs kappa < 0 at Phi = pi/2, central excitation)
    r_pos_k = run_1d_harminv_resonance(
        kappa_L=kL,
        phase_shift=0.5 * np.pi,
        kappa_sign=+1.0,
        n_periods=n_per,
        cells_per_period=cpp,
        n_cycles=n_cyc,
        excite_mode="center",
    )
    runs_log.append(
        {"run": "mut_kappa_pos", "wall_s": r_pos_k["wall_time_s"], "rss_mb": r_pos_k["rss_mb"]}
    )
    r_neg_k_center = run_1d_harminv_resonance(
        kappa_L=kL,
        phase_shift=0.5 * np.pi,
        kappa_sign=-1.0,
        n_periods=n_per,
        cells_per_period=cpp,
        n_cycles=n_cyc,
        excite_mode="center",
    )
    runs_log.append(
        {"run": "mut_kappa_neg_center", "wall_s": r_neg_k_center["wall_time_s"], "rss_mb": r_neg_k_center["rss_mb"]}
    )
    # Also excite kappa < 0 asymmetrically to extract the odd-parity m=0 mode profile (node at z=0)
    r_neg_k_asym = run_1d_harminv_resonance(
        kappa_L=kL,
        phase_shift=0.5 * np.pi,
        kappa_sign=-1.0,
        n_periods=n_per,
        cells_per_period=cpp,
        n_cycles=n_cyc,
        excite_mode="asymmetric",
    )
    runs_log.append(
        {"run": "mut_kappa_neg_asym", "wall_s": r_neg_k_asym["wall_time_s"], "rss_mb": r_neg_k_asym["rss_mb"]}
    )

    iz_c = r_pos_k["iz_center"]
    half_win = cpp
    prof_pos = r_pos_k["ex_profile"][iz_c - 2 * half_win : iz_c + 2 * half_win + 1]
    prof_pos /= max(float(np.max(np.abs(prof_pos))), 1e-30)
    prof_neg = r_neg_k_asym["ex_profile"][iz_c - 2 * half_win : iz_c + 2 * half_win + 1]
    prof_neg /= max(float(np.max(np.abs(prof_neg))), 1e-30)
    z_zoom_over_Lambda = (
        r_pos_k["z_nodes"][iz_c - 2 * half_win : iz_c + 2 * half_win + 1] / DEFAULT_LAMBDA
    )
    center_norm_pos_k = float(abs(prof_pos[2 * half_win]))
    center_norm_neg_k = float(abs(prof_neg[2 * half_win]))

    # 3. Gain-off mutation (g0 = 0 vs g0 = 1.6 * g_th)
    tw_off = time.perf_counter()
    solver_off, meta_off = build_1d_dfb_solver(
        kappa_L=kL,
        phase_shift=0.5 * np.pi,
        n_periods=30,
        cells_per_period=20,
        gain_g0=0.0,
        gain_I_sat=DEFAULT_I_SAT,
    )
    dt_off = meta_off["dt"]
    T0_off = meta_off["T0"]
    f_B_off = meta_off["f_B"]
    n_steps_off = int((300.0 if quick else 450.0) * T0_off / dt_off)
    n_tail_off = int(60.0 * T0_off / dt_off)
    preflight_dfb_solver(1, 1, meta_off["nz_total"], n_steps_off, n_poles=0, label="mut_gain_off")
    trace_off = np.zeros(n_steps_off, dtype=float)
    t0_off = 12.0 * T0_off
    tau_off = 3.5 * T0_off
    for n in range(n_steps_off):
        t = n * dt_off
        if t < 3.5 * t0_off:
            env = 200.0 * np.exp(-0.5 * ((t - t0_off) / tau_off) ** 2) * np.sin(
                2.0 * np.pi * f_B_off * (t - t0_off)
            )
            solver_off.E[:, :, meta_off["iz_center"], "x"] += env
        solver_off.one_step()
        trace_off[n] = float(solver_off.E[0, 0, meta_off["iz_out_right"], "x"])

    I_out_gain_off = float(
        c0 * eps_0 * meta_off["n0"] * np.mean(trace_off[-n_tail_off:] ** 2)
    )
    runs_log.append(
        {"run": "mut_gain_off", "wall_s": time.perf_counter() - tw_off, "rss_mb": _rss_mb()}
    )

    cwt_side_1 = solve_cwt_qws(kL, m_order=1)
    summary = {
        "phi_pts_deg": [float(x) for x in phi_pts_deg],
        "fit_phi_alpha_L": [round(float(x), 4) for x in fit_phi_a],
        "cwt_phi_alpha_L": [round(float(x), 4) for x in cwt_pts_a],
        "fit_phi_delta_L": [round(float(x), 4) for x in fit_phi_d],
        "cwt_phi_delta_L": [round(float(x), 4) for x in cwt_pts_d],
        "phi_45_splitting_fit_delta_L": round(float(fit_phi_d[3] - fit_phi_d[1]), 4),
        "phi_45_splitting_cwt_delta_L": round(float(cwt_pts_d[3] - cwt_pts_d[1]), 4),
        "kappa_pos_alpha_L": round(r_pos_k["alpha_L"], 4),
        "kappa_neg_asym_alpha_L": round(r_neg_k_asym["alpha_L"], 4),
        "kappa_neg_asym_delta_L": round(r_neg_k_asym["delta_L"], 4),
        "kappa_neg_center_excited_alpha_L": round(r_neg_k_center["alpha_L"], 4),
        "kappa_neg_center_excited_delta_L": round(abs(r_neg_k_center["delta_L"]), 4),
        "cwt_side_m1_alpha_L": round(cwt_side_1["alpha_th_L"], 4),
        "cwt_side_m1_delta_L": round(cwt_side_1["delta_th_L"], 4),
        "kappa_pos_center_field_norm": round(center_norm_pos_k, 4),
        "kappa_neg_center_field_norm": round(center_norm_neg_k, 4),
        "gain_off_I_out_over_Isat": float(I_out_gain_off / DEFAULT_I_SAT),
        "gain_off_decay_rate": round(r_pos_k["decay_rate"], 3),
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "fit_runs_log": runs_log,
    }

    if not quick:
        stride_o = max(1, len(trace_off) // 1500)
        t_off_ps = np.arange(0, len(trace_off), stride_o) * dt_off * 1e12
        win_o = max(1, int(3.0 * T0_off / dt_off))
        env_off_t = np.sqrt(2.0 * uniform_filter1d(trace_off**2, size=win_o))[::stride_o]
        np.savez_compressed(
            npz_path,
            phi_dense_deg=phi_dense_deg,
            cwt_phi_a=cwt_phi_a,
            cwt_phi_d=cwt_phi_d,
            phi_pts_deg=phi_pts_deg,
            fit_phi_a=fit_phi_a,
            fit_phi_d=fit_phi_d,
            cwt_pts_a=cwt_pts_a,
            cwt_pts_d=cwt_pts_d,
            z_zoom_over_Lambda=z_zoom_over_Lambda,
            prof_pos=prof_pos,
            prof_neg=prof_neg,
            t_off_ps=t_off_ps,
            env_off_t=env_off_t,
        )
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_task_04_lasing_from_noise(
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run Study 5 (Part D): Self-started lasing from spontaneous-emission noise with ``FourLevelGain``.

    Uses physical semiconductor/solid-state gain parameters (Lorentzian linewidth
    ``gamma_a / (2*pi) = 12 THz``, upper-level lifetime ``tau_21 = 15 ps``, lower-level
    lifetime ``tau_10 = 20 fs``, pump relaxation ``tau_32 = 10 fs``) and uniform
    lifetime scaling ``s <= 100`` (``time_scale = 100.0``) with zero external
    deterministic sources (``E = H = 0`` at ``t = 0``).

    Parameters
    ----------
    quick : bool, optional
        If True, runs a reduced step count and fewer ``time_scale`` points for fast pytest.
    force : bool, optional
        If False and cached JSON/NPZ exist, loads and returns cached results.

    Returns
    -------
    summary : dict
        Lasing-from-noise clamping, spectral purity, SHB, and time-scale invariance metrics.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "lasing_from_noise.json"
    npz_path = DATA_DIR / "lasing_from_noise.npz"
    if not force and not quick and json_path.exists() and npz_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))

    t_start = time.perf_counter()
    n_per = 30
    cpp = 20
    kL = 2.0
    L = n_per * DEFAULT_LAMBDA
    r_pump = 2.0
    gamma_fwhm_hz = 12.0e12  # Physical 12 THz homogeneous linewidth (~0.06 * f_B)
    gamma_a = 2.0 * np.pi * gamma_fwhm_hz
    tau_21 = 15.0e-12  # 15 ps upper-state lifetime (~3000 optical cycles unscaled)
    tau_10 = 20.0e-15  # 20 fs fast lower-state depletion
    tau_32 = 10.0e-15  # 10 fs fast pump relaxation
    sigma_a = 1.0e-4  # Stimulated-emission coupling [C^2 / (kg m^3)]
    N_total = 5.0e24  # Active carrier/ion density [m^-3]
    time_scale_main = 100.0  # Uniform lifetime scaling factor s <= 100
    noise_amp = 0.20

    cwt_qws = solve_cwt_qws(kL, m_order=0)
    cwt_uni = solve_cwt_uniform(kL, m_order=1)
    g_th_qws = 2.0 * cwt_qws["alpha_th_L"] / L
    g_th_uni = 2.0 * cwt_uni["alpha_th_L"] / L
    runs_log: list[dict[str, Any]] = []

    def _run_four_level_noise_case(
        phase_shift: float,
        g_th: float,
        r_p: float,
        time_scale: float,
        spatial_averaging: bool,
        n_cycles: float,
        seed: int = 42,
        label: str = "noise_dfb",
    ) -> dict[str, Any]:
        tw0 = time.perf_counter()
        solver, meta = build_1d_dfb_solver(
            kappa_L=kL,
            phase_shift=phase_shift,
            n_periods=n_per,
            cells_per_period=cpp,
            gain_g0=0.0,
        )
        dt = meta["dt"]
        T0 = meta["T0"]
        f_B = meta["f_B"]
        omega_B = meta["omega_B"]
        n0 = meta["n0"]
        n_steps = int(float(n_cycles) * T0 / dt)
        n_tail = int((180.0 if not quick else 120.0) * T0 / dt)
        preflight_dfb_solver(1, 1, meta["nz_total"], n_steps, n_poles=1, label=label)

        # Threshold inversion dN_th from peak gain g_th = sigma_a * dN_th / (c0 * eps_0 * n0 * gamma_a)
        dN_th = g_th * c0 * eps_0 * n0 * gamma_a / sigma_a
        dN_0 = float(r_p * dN_th)
        tau_eff = tau_21 - tau_10
        wp = dN_0 / (N_total * tau_eff - dN_0 * (tau_32 + tau_21 + tau_10))

        gain = FourLevelGain(
            omega_a=omega_B,
            gamma_a=gamma_a,
            kappa_a=sigma_a,
            N_total=N_total,
            tau_32=tau_32,
            tau_21=tau_21,
            tau_10=tau_10,
            pump_rate=wp,
            eps_inf=n0**2,
            spontaneous_emission=True,
            noise_amplitude=noise_amp,
            noise_components="x",
            seed=int(seed),
            time_scale=float(time_scale),
            spatial_averaging=bool(spatial_averaging),
        )
        in_grat = (meta["z_nodes"] >= -0.5 * L) & (meta["z_nodes"] <= 0.5 * L)
        idx_x = np.flatnonzero(in_grat)
        solver._add_gain_edges(gain, idx_x, np.ones(idx_x.size, dtype=float))
        grp = solver._gain_groups[-1]
        grp["a1"] = grp["eps_s"].copy()
        grp["den"] = grp["eps_s"].copy()
        for ch in grp["noise_channels"]:
            ch["den"] = grp["den"][ch["pos"]]

        iz_out = meta["iz_out_right"]
        trace_out = np.zeros(n_steps, dtype=float)
        dn_mean_t = np.zeros(n_steps, dtype=float)
        I_z_accum = np.zeros(solver.Nz, dtype=float)
        dn_z_accum = np.zeros(idx_x.size, dtype=float)
        count_tail = 0

        for n in range(n_steps):
            solver.one_step()
            ex_1d = np.asarray(solver.E[0, 0, :, "x"], dtype=float)
            trace_out[n] = float(ex_1d[iz_out])
            dn_inst = grp["N"][2] - grp["N"][1]
            dn_mean_t[n] = float(np.mean(dn_inst))
            if n >= n_steps - n_tail:
                I_z_accum += c0 * eps_0 * n0 * (ex_1d**2)
                dn_z_accum += dn_inst
                count_tail += 1

        I_z = I_z_accum / max(1, count_tail)
        I_env = uniform_filter1d(I_z, size=cpp)
        I_out = float(c0 * eps_0 * n0 * np.mean(trace_out[-n_tail:] ** 2))
        I_sat_eff = float(gain.saturation_intensity())
        dn_profile_over_th = (dn_z_accum / max(1, count_tail)) / dN_th

        # Early vs late RMS envelope amplification and late-window clamping ratio
        n_early = max(10, int(25.0 * T0 / dt))
        rms_early = float(np.sqrt(np.mean(trace_out[:n_early] ** 2)))
        rms_late1 = float(np.sqrt(np.mean(trace_out[-n_tail : -n_tail // 2] ** 2)))
        rms_late2 = float(np.sqrt(np.mean(trace_out[-n_tail // 2 :] ** 2)))

        sig = trace_out[-n_tail:] * np.hanning(n_tail)
        n_fft = n_tail * 8
        freqs = np.fft.rfftfreq(n_fft, d=dt)
        spec = np.abs(np.fft.rfft(sig, n=n_fft)) ** 2
        band = (freqs >= 0.90 * f_B) & (freqs <= 1.10 * f_B)
        f_band = freqs[band]
        s_band = spec[band] / max(float(np.max(spec[band])), 1e-30)
        delta_L_axis = 2.0 * np.pi * n0 * (f_band - f_B) / c0 * L

        tau_p_mode = float(n0 / (c0 * g_th))
        ts_limits_raw = gain.time_scaling_limits(tau_p=tau_p_mode, dt=dt)

        runs_log.append(
            {"run": label, "wall_s": time.perf_counter() - tw0, "rss_mb": _rss_mb()}
        )
        return {
            "trace_out": trace_out,
            "dn_mean_over_th": dn_mean_t / dN_th,
            "dn_final_over_th": float(np.mean(dn_mean_t[-n_tail:]) / dN_th),
            "dn_profile_over_th": dn_profile_over_th,
            "dt": dt,
            "T0": T0,
            "z_nodes": meta["z_nodes"],
            "in_grat": in_grat,
            "I_env": I_env,
            "I_out_over_Isat": float(I_out / I_sat_eff),
            "I_sat_eff": float(I_sat_eff),
            "dN_th": float(dN_th),
            "delta_L_axis": delta_L_axis,
            "spec_norm": s_band,
            "buildup_gain_ratio": float(rms_late2 / max(rms_early, 1e-30)),
            "late_clamping_ratio": float(rms_late2 / max(rms_late1, 1e-30)),
            "time_scaling_limits": ts_limits_raw,
        }

    n_cyc_main = 850.0 if quick else 1350.0
    # 1. QWS DFB from noise with spatial_averaging=True (exact modal inversion clamping <dN>/dN_th -> 1)
    qws_savg = _run_four_level_noise_case(
        0.5 * np.pi,
        g_th_qws,
        r_pump,
        time_scale=time_scale_main,
        spatial_averaging=True,
        n_cycles=n_cyc_main,
        seed=42,
        label="noise_qws_savg_s100",
    )
    # 2. QWS DFB from noise with spatial_averaging=False (full local SHB carrier hole burning)
    qws_shb = _run_four_level_noise_case(
        0.5 * np.pi,
        g_th_qws,
        r_pump,
        time_scale=time_scale_main,
        spatial_averaging=False,
        n_cycles=n_cyc_main,
        seed=42,
        label="noise_qws_shb_s100",
    )
    # 3. Uniform DFB from noise with spatial_averaging=False (spontaneous two-mode stopband lasing)
    uni_shb = _run_four_level_noise_case(
        0.0,
        g_th_uni,
        r_pump,
        time_scale=time_scale_main,
        spatial_averaging=False,
        n_cycles=n_cyc_main,
        seed=42,
        label="noise_uni_shb_s100",
    )

    # 4. Time-scale invariance sweep s in {25, 50, 100} (all s <= 100)
    s_list = [50.0, 100.0] if quick else [25.0, 50.0, 100.0]
    s_dn_clamped: list[float] = []
    s_I_out: list[float] = []
    for s_val in s_list:
        if abs(s_val - time_scale_main) < 1e-9:
            s_dn_clamped.append(round(qws_savg["dn_final_over_th"], 4))
            s_I_out.append(round(qws_savg["I_out_over_Isat"], 4))
            continue
        # Scale n_cycles mildly when s is smaller so relaxation oscillations damp out
        n_cyc_s = n_cyc_main * min(2.0, (time_scale_main / s_val) ** 0.45)
        res_s = _run_four_level_noise_case(
            0.5 * np.pi,
            g_th_qws,
            r_pump,
            time_scale=float(s_val),
            spatial_averaging=True,
            n_cycles=n_cyc_s,
            seed=42,
            label=f"noise_qws_savg_s{int(s_val)}",
        )
        s_dn_clamped.append(round(res_s["dn_final_over_th"], 4))
        s_I_out.append(round(res_s["I_out_over_Isat"], 4))

    d_ax = qws_shb["delta_L_axis"]
    qws_peak_delta_L = float(d_ax[int(np.argmax(qws_shb["spec_norm"]))])
    side_mask = np.abs(d_ax) > 2.0
    qws_smsr_db = float(
        10.0 * np.log10(1.0 / max(float(np.max(qws_shb["spec_norm"][side_mask])), 1e-12))
    )
    pos_m = d_ax > 0.5
    neg_m = d_ax < -0.5
    uni_peak_pos_delta_L = float(d_ax[pos_m][int(np.argmax(uni_shb["spec_norm"][pos_m]))])
    uni_peak_neg_delta_L = float(d_ax[neg_m][int(np.argmax(uni_shb["spec_norm"][neg_m]))])

    rigrod_r2 = cwt_rigrod_output(np.array([r_pump]), kappa_L=kL, phase_shift=0.5 * np.pi)
    I_out_rigrod_r2 = float(rigrod_r2["I_out_rigrod_sw"][0])
    I_out_linear_r2 = float(rigrod_r2["I_out_linear_shb"][0])
    # Rate-equation steady state without spatial hole burning (spatial_averaging=True):
    # Poynting energy balance gives I_out / <I>_L = alpha_th,0 * L, while homogeneous
    # modal clamping gives <I>_L / I_sat = r_pump - 1 => I_out^savg / I_sat = alpha_th,0 * L * (r_pump - 1).
    I_out_savg_theory_r2 = float(cwt_qws["alpha_th_L"] * (r_pump - 1.0))

    # Gain-dispersive CWT threshold and Kramers-Kronig mode pulling for uniform DFB
    d_gamma_L = float(0.5 * gamma_a * DEFAULT_N0 * L / c0)
    cwt_uni_disp = solve_cwt_uniform_dispersive(kL, d_gamma_L, m_order=1)
    uni_half_splitting = 0.5 * (uni_peak_pos_delta_L - uni_peak_neg_delta_L)

    s_dn_spread_pct = float(
        (max(s_dn_clamped) - min(s_dn_clamped)) / np.mean(s_dn_clamped) * 100.0
    )
    s_I_spread_pct = float(
        (max(s_I_out) - min(s_I_out)) / np.mean(s_I_out) * 100.0
    )

    ts_raw = qws_savg["time_scaling_limits"]
    ts_summary = {
        "time_scale": round(float(ts_raw["time_scale"]), 2),
        "sat_rate_over_kappa_cav": round(float(ts_raw["sat_rate_over_kappa_cav"]), 4),
        "rate_21_over_kappa_cav": round(float(ts_raw["rate_21_over_kappa_cav"]), 4),
        "rate_21_over_gamma_a": round(float(ts_raw["rate_21_over_gamma_a"]), 4),
        "gamma_a_over_rate_21": round(float(1.0 / ts_raw["rate_21_over_gamma_a"]), 2),
        "max_rate_over_gamma_a": round(float(ts_raw["max_rate_over_gamma_a"]), 2),
        "s_max_cavity": round(float(ts_raw["s_max_cavity"]), 2),
        "s_max_dephasing": round(float(ts_raw["s_max_dephasing"]), 2),
        "s_max_fast_dephasing": round(float(ts_raw["s_max_fast_dephasing"]), 4),
        "max_rate_dt": round(float(ts_raw["max_rate_dt"]), 4),
        "s_max_cn_positivity": round(float(ts_raw["s_max_cn_positivity"]), 2),
    }

    dn_prof_shb = uniform_filter1d(qws_shb["dn_profile_over_th"], size=cpp)
    summary = {
        "r_pump": r_pump,
        "gamma_fwhm_thz": round(gamma_fwhm_hz * 1e-12, 2),
        "d_gamma_L": round(d_gamma_L, 4),
        "tau_21_ps": round(tau_21 * 1e12, 2),
        "tau_10_fs": round(tau_10 * 1e15, 2),
        "tau_32_fs": round(tau_32 * 1e15, 2),
        "time_scale_main": time_scale_main,
        "time_scaling_limits": ts_summary,
        "qws_savg_dn_clamped_over_th": round(qws_savg["dn_final_over_th"], 4),
        "qws_savg_dn_clamping_err_pct": round(
            abs(qws_savg["dn_final_over_th"] - 1.0) * 100.0, 2
        ),
        "qws_shb_dn_mean_over_th": round(qws_shb["dn_final_over_th"], 4),
        "qws_shb_dn_center_over_th": round(
            float(dn_prof_shb[len(dn_prof_shb) // 2]), 4
        ),
        "qws_shb_I_out_over_Isat": round(qws_shb["I_out_over_Isat"], 4),
        "qws_rigrod_sw_I_out_over_Isat": round(I_out_rigrod_r2, 4),
        "qws_shb_vs_rigrod_rel_err_pct": round(
            abs(qws_shb["I_out_over_Isat"] - I_out_rigrod_r2) / I_out_rigrod_r2 * 100.0, 2
        ),
        "qws_savg_I_out_over_Isat": round(qws_savg["I_out_over_Isat"], 4),
        "qws_savg_theory_I_out_over_Isat": round(I_out_savg_theory_r2, 4),
        "qws_savg_vs_theory_rel_err_pct": round(
            abs(qws_savg["I_out_over_Isat"] - I_out_savg_theory_r2)
            / I_out_savg_theory_r2
            * 100.0,
            2,
        ),
        "qws_linear_I_out_over_Isat": round(I_out_linear_r2, 4),
        "qws_peak_delta_L": round(qws_peak_delta_L, 4),
        "qws_smsr_db": round(qws_smsr_db, 2),
        "uni_peak_pos_delta_L": round(uni_peak_pos_delta_L, 4),
        "uni_peak_neg_delta_L": round(uni_peak_neg_delta_L, 4),
        "uni_half_splitting_delta_L": round(uni_half_splitting, 4),
        "cwt_uni_delta_L": round(cwt_uni["delta_th_L"], 4),
        "cwt_uni_disp_alpha0_L": round(cwt_uni_disp["alpha0_th_disp_L"], 4),
        "cwt_uni_disp_delta_L": round(cwt_uni_disp["delta_th_disp_L"], 4),
        "cwt_uni_disp_r_th": round(cwt_uni_disp["r_p_th_disp"], 4),
        "cwt_uni_lorentz_factor_cwt_edge": round(
            cwt_uni_disp["lorentz_gain_factor_at_cwt_edge"], 4
        ),
        "cwt_uni_lorentz_factor_disp_edge": round(
            cwt_uni_disp["lorentz_gain_factor_at_disp_edge"], 4
        ),
        "uni_disp_delta_rel_err_pct": round(
            abs(uni_half_splitting - cwt_uni_disp["delta_th_disp_L"])
            / cwt_uni_disp["delta_th_disp_L"]
            * 100.0,
            2,
        ),
        "uni_cwt_undisp_delta_rel_diff_pct": round(
            (uni_half_splitting - cwt_uni["delta_th_L"]) / cwt_uni["delta_th_L"] * 100.0,
            2,
        ),
        "buildup_gain_ratio": round(qws_shb["buildup_gain_ratio"], 2),
        "late_clamping_ratio": round(qws_shb["late_clamping_ratio"], 4),
        "time_scale_list": [float(s) for s in s_list],
        "time_scale_dn_clamped": s_dn_clamped,
        "time_scale_I_out_over_Isat": s_I_out,
        "time_scale_dn_spread_pct": round(s_dn_spread_pct, 2),
        "time_scale_I_out_spread_pct": round(s_I_spread_pct, 2),
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "fit_runs_log": runs_log,
    }

    if not quick:
        stride = max(1, len(qws_shb["trace_out"]) // 2400)
        t_ps = np.arange(0, len(qws_shb["trace_out"]), stride) * qws_shb["dt"] * 1e12
        win_t = max(1, int(3.0 * qws_shb["T0"] / qws_shb["dt"]))
        env_qws_shb = np.sqrt(
            2.0 * uniform_filter1d(qws_shb["trace_out"] ** 2, size=win_t)
        )[::stride]
        env_qws_savg = np.sqrt(
            2.0 * uniform_filter1d(qws_savg["trace_out"] ** 2, size=win_t)
        )[::stride]
        u_grat = qws_shb["z_nodes"][qws_shb["in_grat"]] / L
        dn_prof_shb = uniform_filter1d(qws_shb["dn_profile_over_th"], size=cpp)
        np.savez_compressed(
            npz_path,
            t_ps=t_ps,
            env_qws_shb=env_qws_shb,
            env_qws_savg=env_qws_savg,
            dn_qws_savg=qws_savg["dn_mean_over_th"][::stride],
            dn_qws_shb=qws_shb["dn_mean_over_th"][::stride],
            delta_L_axis=d_ax,
            spec_qws_shb=qws_shb["spec_norm"],
            spec_uni_shb=uni_shb["spec_norm"],
            u_grat=u_grat,
            dn_prof_shb=dn_prof_shb,
            dn_prof_raw=qws_shb["dn_profile_over_th"],
        )
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ==============================================================================
# 4. Setup Figures, Result Figures, and Animated GIF
# ==============================================================================


def generate_setup_figures(fig_dir: str | Path = FIG_DIR) -> dict[str, dict[str, Any]]:
    """Generate 1D and 2D setup figures (``00a``, ``00b``) and Markdown tables via ``wakis.schematic``.

    Parameters
    ----------
    fig_dir : str or Path, optional
        Output directory for figures and Markdown setup tables.

    Returns
    -------
    res : dict
        Mapping ``{'1d': out_1d, '2d': out_2d}`` from ``schematic.save_setup_figures``.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. 1D DFB Cavity Setup (ZX cut plane)
    solver_1d, meta_1d = build_1d_dfb_solver(
        kappa_L=2.0,
        phase_shift=0.5 * np.pi,
        n_periods=DEFAULT_N_PERIODS,
        cells_per_period=DEFAULT_CPP,
        gain_g0=2.0 * 0.697069 / (DEFAULT_N_PERIODS * DEFAULT_LAMBDA),
        gain_I_sat=DEFAULT_I_SAT,
    )
    L1 = meta_1d["L"]
    dx1 = meta_1d["dx"]
    f_B = meta_1d["f_B"]
    lam0 = meta_1d["lam0"]

    grat_left = Box(
        center=(0.0, 0.0, -0.25 * L1),
        size=(dx1, dx1, 0.5 * L1),
        material=meta_1d["n0"] ** 2,
    )
    grat_left.name = "Left DFB grating (z ∈ [-L/2, 0], φ = -π/2, SaturableGain)"
    grat_right = Box(
        center=(0.0, 0.0, 0.25 * L1),
        size=(dx1, dx1, 0.5 * L1),
        material=meta_1d["n0"] ** 2,
    )
    grat_right.name = "Right DFB grating (z ∈ [0, +L/2], φ = +π/2, λ/4 shift at z = 0)"
    wf_1d = GaussianSource(frequency=f_B, fwidth=0.08 * f_B)
    src_1d = CurrentSource(
        wf_1d,
        component="x",
        center=(0.0, 0.0, 0.0),
        size=(dx1, dx1, 0.0),
    )
    mon_harminv = ResonanceMonitor(
        component="Ex",
        center=(0.0, 0.0, -L1 / 6.0),
        t_start=wf_1d.end_time,
    )
    probes_1d = [
        {"pos": (0.0, 0.0, 0.5 * L1 + 0.5 * meta_1d["pad_cells"] * meta_1d["dz"]), "label": "Facet output probe z_out"},
    ]

    variations_1d = [
        {
            "Case": "λ/4-shifted DFB (Φ = π/2)",
            "Grating phase": "φ(z) = sign(z) π/2",
            "Gain model": "Linear (I_sat = ∞) & Saturable (I_sat = 10 MW/m²)",
            "Mode behavior": "Single Bragg mode at δL = 0",
        },
        {
            "Case": "Uniform DFB (Φ = 0)",
            "Grating phase": "φ(z) = 0",
            "Gain model": "Linear (I_sat = ∞) & Saturable (I_sat = 10 MW/m²)",
            "Mode behavior": "Degenerate stopband doublet at ±|δ₁L|",
        },
    ]

    solver_1d.bg = [meta_1d["n0"] ** 2, 1.0, 0.0]
    out_1d = schematic.save_setup_figures(
        solver_1d,
        out_dir=fig_dir,
        prefix="00",
        suffix="",
        plane="zx",
        structures=[grat_left, grat_right],
        material_names={
            "background": "Index-matched waveguide cladding (n₀ = 3.00, ε_r = 9.00)"
        },
        sources=[src_1d],
        monitors=[mon_harminv],
        probes=probes_1d,
        unit="um",
        wavelength=lam0,
        freq=f_B,
        min_feature=0.5 * meta_1d["Lambda"],
        title_geometry="1D Index-Coupled DFB Laser Cavity (λ/4 Phase Shift at z = 0)",
        title_mesh="1D DFB Mesh Discretization (24 cells/Λ, 48 cells/λ) & Seed Pulse",
        variations=variations_1d,
    )

    # 2. 2D TE Slab-Waveguide DFB Setup (XZ cut plane)
    solver_2d, meta_2d = build_2d_slab_dfb_solver(kappa_L=2.0, n_periods=30)
    solver_2d.bg = [meta_2d["n_clad"] ** 2, 1.0, 0.0]
    L2 = meta_2d["L"]
    d_core = meta_2d["d_core"]
    dy2 = meta_2d["dy"]
    core_left = Box(
        center=(0.0, 0.0, -0.25 * L2),
        size=(d_core, dy2, 0.5 * L2),
        material=meta_2d["n_core"] ** 2,
    )
    core_left.name = "Corrugated active slab core (n_core = 3.0, d = 320 nm, φ = -π/2)"
    core_right = Box(
        center=(0.0, 0.0, 0.25 * L2),
        size=(d_core, dy2, 0.5 * L2),
        material=meta_2d["n_core"] ** 2,
    )
    core_right.name = "Corrugated active slab core (n_core = 3.0, φ = +π/2, λ/4 shift at z = 0)"
    wf_2d = GaussianSource(frequency=f_B, fwidth=0.08 * f_B)
    src_2d = CurrentSource(
        wf_2d,
        component="y",
        center=(0.0, 0.0, 0.0),
        size=(d_core, dy2, 0.0),
    )
    mon_2d = ResonanceMonitor(
        component="Ey",
        center=(0.0, 0.0, -L2 / 6.0),
        t_start=wf_2d.end_time,
    )

    out_2d = schematic.save_setup_figures(
        solver_2d,
        out_dir=fig_dir,
        prefix="00",
        suffix="_2d",
        plane="zx",
        structures=[core_left, core_right],
        material_names={
            "background": "Dielectric waveguide cladding (n_clad = 2.40, ε_r = 5.76)"
        },
        sources=[src_2d],
        monitors=[mon_2d],
        unit="um",
        wavelength=lam0,
        freq=f_B,
        min_feature=0.5 * meta_2d["Lambda"],
        title_geometry="2D TE Slab-Waveguide DFB Check (Core n = 3.0, Cladding n = 2.4)",
        title_mesh="2D Slab DFB Mesh (Δx = 20 nm, 16 cells across core, 20 cells/Λ)",
    )
    return {"1d": out_1d, "2d": out_2d}


def plot_01_cwt_and_linear_gain(fig_dir: str | Path = FIG_DIR) -> Path:
    """Plot Figure 1: Linear-gain ``harminv`` zero crossing and ``alpha_th L``, ``delta_th L`` vs ``kappa L``."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "cwt_and_linear_gain.npz")
    meta = json.loads((DATA_DIR / "cwt_and_linear_gain.json").read_text(encoding="utf-8"))

    figstyle.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.FULL_WIDTH_IN, 4.1))

    # (a) Decay rate vs normalized linear gain g0 / g_th,CWT at kL = 2.0
    ax = axes[0]
    ax.axhline(0.0, color=figstyle.INK, lw=1.0, ls=":", alpha=0.8)
    ax.axvline(1.0, color="#64748B", lw=1.0, ls="--", label="CWT $g_0 = g_{\\mathrm{th}}$")
    ax.plot(
        d["gain_ratios"],
        d["qws_decay_vs_g"] * 1e-12,
        "o-",
        color="#0072B2",
        lw=1.8,
        ms=5.5,
        label="1D $\\lambda/4$ ($\\kappa L = 2$)",
    )
    ax.plot(
        d["gain_ratios"],
        d["uni_decay_vs_g"] * 1e-12,
        "s--",
        color="#D55E00",
        lw=1.6,
        ms=5.0,
        label="1D uniform ($\\kappa L = 2$)",
    )
    ax.plot(
        d["slab_ratios"],
        d["slab_decays"] * 1e-12,
        "^:",
        color="#009E73",
        lw=1.5,
        ms=5.5,
        label="2D TE$_0$ slab ($\\kappa L = 2$)",
    )
    ax.set_xlabel("Linear gain $g_0 / g_{\\mathrm{th,CWT}}$ [-]")
    ax.set_ylabel("Decay rate $\\gamma_m$ [$\\mathrm{ps}^{-1}$]")
    ax.set_title("(a) Zero crossing")
    figstyle.place_legend_below(ax, ncol=1, y_offset=-0.24, fontsize=8.5)

    # (b) Threshold gain alpha_th * L vs kappa * L
    ax = axes[1]
    kL_d = d["kL_dense"]
    ax.plot(kL_d, d["cwt_uni_a"], "-", color="#D55E00", lw=2.0, label="Uniform CWT ($m=\\pm 1$)")
    ax.plot(
        kL_d,
        d["cwt_uni_a_asymp"],
        ":",
        color="#D55E00",
        lw=1.2,
        alpha=0.85,
        label="Asymp. $(\\pi/\\kappa L)^2$",
    )
    ax.plot(kL_d, d["cwt_qws_a"], "-", color="#0072B2", lw=2.0, label="$\\lambda/4$ CWT ($m=0$)")
    ax.plot(
        kL_d,
        d["cwt_qws_a_asymp"],
        ":",
        color="#0072B2",
        lw=1.2,
        alpha=0.85,
        label="Asymp. $2\\kappa L e^{-\\kappa L}$",
    )
    ax.plot(
        kL_d,
        d["cwt_qws_side_a"],
        "-.",
        color="#64748B",
        lw=1.3,
        label="$\\lambda/4$ side ($m=\\pm 1$)",
    )
    ax.plot(
        d["kL_points"],
        d["fit_uni_a"],
        "s",
        color="#D55E00",
        ms=6.0,
        mec="black",
        mew=0.5,
        label="1D FIT uniform",
    )
    ax.plot(
        d["kL_points"],
        d["fit_qws_a"],
        "o",
        color="#0072B2",
        ms=6.0,
        mec="black",
        mew=0.5,
        label="1D FIT $\\lambda/4$",
    )
    ax.plot(
        [2.0],
        [meta["slab_2d_alpha_th_L"]],
        "^",
        color="#009E73",
        ms=7.5,
        mec="black",
        mew=0.6,
        label="2D slab FIT",
    )
    ax.set_yscale("log")
    ax.set_ylim(0.10, 4.5)
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.1, 0.2, 0.5, 1.0, 2.0, 4.0]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Coupling $\\kappa L$ [-]")
    ax.set_ylabel("Threshold gain $\\alpha_{\\mathrm{th}} L$ [-]")
    ax.set_title("(b) Threshold gain")
    figstyle.place_legend_below(ax, ncol=1, y_offset=-0.24, fontsize=8.5)

    # (c) Threshold detuning delta_th * L vs kappa * L
    ax = axes[2]
    ax.fill_between(
        kL_d,
        -kL_d,
        kL_d,
        color="#E2E8F0",
        alpha=0.75,
        label="Stopband $|\\delta L| \\leq \\kappa L$",
    )
    ax.plot(kL_d, d["cwt_uni_d"], "-", color="#D55E00", lw=2.0, label="Uniform CWT ($\\pm\\delta_1 L$)")
    ax.plot(kL_d, -d["cwt_uni_d"], "-", color="#D55E00", lw=2.0)
    ax.plot(
        kL_d,
        np.zeros_like(kL_d),
        "-",
        color="#0072B2",
        lw=2.0,
        label="$\\lambda/4$ CWT ($\\delta_0 L = 0$)",
    )
    ax.plot(
        d["kL_points"],
        d["fit_uni_d"],
        "s",
        color="#D55E00",
        ms=5.8,
        mec="black",
        mew=0.5,
        label="1D FIT uniform",
    )
    ax.plot(
        d["kL_points"],
        -d["fit_uni_d"],
        "s",
        color="#D55E00",
        ms=5.8,
        mec="black",
        mew=0.5,
    )
    ax.plot(
        d["kL_points"],
        d["fit_qws_d"],
        "o",
        color="#0072B2",
        ms=5.8,
        mec="black",
        mew=0.5,
        label="1D FIT $\\lambda/4$",
    )
    ax.set_xlabel("Coupling $\\kappa L$ [-]")
    ax.set_ylabel("Bragg detuning $\\delta_{\\mathrm{th}} L$ [rad]")
    ax.set_title("(c) Mode detuning")
    figstyle.place_legend_below(ax, ncol=1, y_offset=-0.24, fontsize=8.5)

    fig.subplots_adjust(wspace=0.42, bottom=0.35)
    out_path = fig_dir / "01_cwt_and_linear_gain.png"
    figstyle.save(fig, out_path, dpi=220, close=True, auto_legend_below=False)
    return out_path


def plot_02_2d_slab_mode(fig_dir: str | Path = FIG_DIR) -> Path:
    """Plot Figure 2: 2D TE slab-waveguide DFB defect mode field map with ``overlay_geometry``."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "cwt_and_linear_gain.npz")
    meta = json.loads((DATA_DIR / "cwt_and_linear_gain.json").read_text(encoding="utf-8"))

    solver_2d, m2d = build_2d_slab_dfb_solver(kappa_L=2.0, n_periods=30)
    L2 = m2d["L"]
    d_core = m2d["d_core"]
    dy2 = m2d["dy"]
    core_box = Box(
        center=(0.0, 0.0, 0.0),
        size=(d_core, dy2, L2),
        material=m2d["n_core"] ** 2,
    )
    core_box.name = "Active slab core (n = 3.0)"

    figstyle.apply_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(figstyle.FULL_WIDTH_IN, 3.2), gridspec_kw={"width_ratios": [1.75, 1.0]}
    )

    # (a) 2D Ey(z, x) standing-wave field map with schematic.overlay_geometry
    ax = axes[0]
    ey_map = d["slab_ey_map"]
    ey_norm = ey_map / max(float(np.max(np.abs(ey_map))), 1e-30)
    z_um = d["slab_z_um"]
    x_um = d["slab_x_um"]

    im = ax.pcolormesh(
        z_um,
        x_um,
        ey_norm,
        shading="auto",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Normalized $E_y(x, z)$ [-]", fontsize=8.5)

    # Call wakis.schematic.overlay_geometry on the 2D slab solver and overlay core boundaries
    schematic.overlay_geometry(
        ax,
        solver_2d,
        plane="zx",
        unit="um",
        structures=[core_box],
        show_boundaries=True,
        show_materials=True,
        outline_color="#0F172A",
        label_materials=True,
    )
    ax.axvline(0.0, color="#F59E0B", lw=1.4, ls="--", zorder=8)
    ax.text(
        -3.95,
        -0.34,
        "$\\lambda/4$ shift ($z=0$) · Core $n=3.0$, Clad $n=2.4$",
        fontsize=8.5,
        color="#0F172A",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.90, "edgecolor": "#475569"},
        zorder=9,
    )
    ax.set_xlim(float(z_um[0]), float(z_um[-1]))
    ax.set_ylim(float(x_um[0]), float(x_um[-1]))
    ax.set_xlabel("Longitudinal coordinate $z$ [$\\mu\\mathrm{m}$]")
    ax.set_ylabel("Transverse coordinate $x$ [$\\mu\\mathrm{m}$]")
    ax.set_title("(a) 2D TE$_0$ slab DFB mode $E_y(x, z)$")

    # (b) Transverse TE0 mode profile vs core boundaries
    ax2 = axes[1]
    mode_x = d["slab_mode_x"]
    mode_norm = (mode_x / np.max(np.abs(mode_x))) ** 2
    ax2.axvspan(
        -0.5 * d_core * 1e6,
        0.5 * d_core * 1e6,
        color="#DBEAFE",
        alpha=0.75,
        label=f"Core ($\\Gamma = {meta['slab_2d_Gamma_core']:.3f}$)",
    )
    ax2.plot(
        x_um,
        mode_norm,
        "-",
        color="#0072B2",
        lw=2.0,
        label=f"TE$_0$ $|E_y|^2$ ($n_{{\\mathrm{{eff}}}}={meta['slab_2d_n_eff']:.3f}$)",
    )
    iz_mid = len(z_um) // 2
    slice_fit = (ey_norm[:, iz_mid] / max(float(np.max(np.abs(ey_norm[:, iz_mid]))), 1e-30)) ** 2
    ax2.plot(
        x_um,
        slice_fit,
        "o",
        color="#D55E00",
        ms=4.2,
        alpha=0.85,
        label="2D FIT slice ($z = 0$)",
    )
    ax2.set_xlabel("Transverse coordinate $x$ [$\\mu\\mathrm{m}$]")
    ax2.set_ylabel("Normalized $|E_y(x)|^2$ [-]")
    ax2.set_title("(b) Transverse TE$_0$ profile")
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.subplots_adjust(wspace=0.35)
    out_path = fig_dir / "02_2d_slab_mode.png"
    figstyle.save(fig, out_path, dpi=220, close=True, auto_legend_below=True)
    return out_path


def plot_03_convergence(fig_dir: str | Path = FIG_DIR) -> Path:
    """Plot Figure 3: Log-log grid convergence in cells per period ``N_Lambda`` with order guides."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "convergence.npz")
    meta = json.loads((DATA_DIR / "convergence.json").read_text(encoding="utf-8"))

    cpp = d["cpp_list"].astype(float)
    figstyle.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.FULL_WIDTH_IN, 3.2))

    # (a) Convergence of threshold gain alpha_th * L
    ax = axes[0]
    ax.loglog(
        cpp[:-1],
        d["err_qws_a_rich"][:-1],
        "o-",
        color="#0072B2",
        lw=1.9,
        ms=6.0,
        label=f"$\\lambda/4$ vs Rich. ($p={meta['p_obs_qws_alpha']:.2f}$)",
    )
    ax.loglog(
        cpp[:-1],
        d["err_uni_a_rich"][:-1],
        "s--",
        color="#D55E00",
        lw=1.8,
        ms=5.8,
        label=f"Uniform vs Rich. ($p={meta['p_obs_uni_alpha']:.2f}$)",
    )
    ax.loglog(
        cpp,
        d["err_qws_a_cwt"],
        "^:",
        color="#009E73",
        lw=1.5,
        ms=5.5,
        label="$\\lambda/4$ vs CWT",
    )
    guide_a = d["err_qws_a_rich"][0] * (cpp / cpp[0]) ** (-2.0)
    ax.loglog(cpp, guide_a, "k--", lw=1.1, alpha=0.75, label="$O(N_\\Lambda^{-2})$ guide")
    ax.set_xlabel("Cells per Bragg period $N_\\Lambda = \\Lambda / \\Delta z$ [-]")
    ax.set_ylabel("Gain error $|\\Delta(\\alpha_{\\mathrm{th}} L)|$ [-]")
    ax.set_title("(a) Threshold gain convergence")
    ax.legend(loc="lower left", fontsize=8.5)

    # (b) Convergence of Bragg detuning delta_th * L
    ax2 = axes[1]
    ax2.loglog(
        cpp,
        d["err_qws_d_cwt"],
        "o-",
        color="#0072B2",
        lw=1.9,
        ms=6.0,
        label=f"$\\lambda/4$ $|\\delta_0 L|$ ($p={meta['p_obs_qws_delta']:.2f}$)",
    )
    err_uni_d_rich = np.abs(d["uni_d"] - meta["uni_delta_L_richardson"])
    ax2.loglog(
        cpp[:-1],
        err_uni_d_rich[:-1],
        "s--",
        color="#D55E00",
        lw=1.8,
        ms=5.8,
        label="Uniform $|\\delta_1 L - \\delta_{1,\\mathrm{Rich}} L|$",
    )
    guide_d = d["err_qws_d_cwt"][0] * (cpp / cpp[0]) ** (-2.0)
    ax2.loglog(cpp, guide_d, "k--", lw=1.1, alpha=0.75, label="$O(N_\\Lambda^{-2})$ guide")
    ax2.set_xlabel("Cells per Bragg period $N_\\Lambda = \\Lambda / \\Delta z$ [-]")
    ax2.set_ylabel("Detuning error $|\\Delta(\\delta_{\\mathrm{th}} L)|$ [rad]")
    ax2.set_title("(b) Bragg detuning convergence")
    ax2.legend(loc="lower left", fontsize=8.5)

    fig.subplots_adjust(wspace=0.35)
    out_path = fig_dir / "03_convergence.png"
    figstyle.save(fig, out_path, dpi=220, close=True, auto_legend_below=True)
    return out_path


def plot_04_lasing_spectra(fig_dir: str | Path = FIG_DIR) -> Path:
    """Plot Figure 4: Single-mode QWS vs two-mode uniform DFB output build-up and spectra."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "lasing_dynamics.npz")
    meta = json.loads((DATA_DIR / "lasing_dynamics.json").read_text(encoding="utf-8"))

    figstyle.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.FULL_WIDTH_IN, 3.2))

    # (a) Output envelope build-up & clamping vs two-mode beating
    ax = axes[0]
    ax.plot(
        d["t_ps_sub"],
        d["env_qws_t"] * 1e-4,
        "-",
        color="#0072B2",
        lw=1.8,
        label="$\\lambda/4$ DFB (single-mode clamping)",
    )
    ax.plot(
        d["t_ps_sub"],
        d["env_uni_t"] * 1e-4,
        "-",
        color="#D55E00",
        lw=1.2,
        alpha=0.85,
        label="Uniform DFB (two-mode beating)",
    )
    ax.set_xlabel("Time $t$ [ps]")
    ax.set_ylabel("Output $|E_x(z_{\\mathrm{out}}, t)|$ [$10^4\\;\\mathrm{V/m}$]")
    ax.set_title(f"(a) Lasing build-up ($g_0 / g_{{\\mathrm{{th}}}} = {meta['r_main']:.1f}$)")
    ax.legend(loc="lower right", fontsize=8.5)

    # (b) Normalized output power spectra vs detuning delta * L
    ax2 = axes[1]
    ax2.axvspan(-2.0, 2.0, color="#E2E8F0", alpha=0.75, label="Stopband $|\\delta L| \\leq 2$")
    ax2.axvline(0.0, color="#0072B2", ls=":", lw=1.0)
    ax2.axvline(meta["cwt_uni_delta_L"], color="#D55E00", ls=":", lw=1.0)
    ax2.axvline(-meta["cwt_uni_delta_L"], color="#D55E00", ls=":", lw=1.0)
    ax2.semilogy(
        d["delta_L_axis"],
        np.maximum(d["spec_qws"], 1e-6),
        "-",
        color="#0072B2",
        lw=1.9,
        label=f"$\\lambda/4$ ($\\delta_0 L = {meta['qws_peak_delta_L']:+.2f}$, SMSR $= {meta['qws_smsr_db']:.1f}$ dB)",
    )
    ax2.semilogy(
        d["delta_L_axis"],
        np.maximum(d["spec_uni"], 1e-6),
        "--",
        color="#D55E00",
        lw=1.6,
        label=f"Uniform ($\\delta L = {meta['uni_peak_neg_delta_L']:+.2f}, {meta['uni_peak_pos_delta_L']:+.2f}$)",
    )
    ax2.set_xlim(-7.5, 7.5)
    ax2.set_ylim(1e-5, 2.0)
    ax2.set_xlabel("Normalized detuning $\\delta L = (\\omega - \\omega_B) n_0 L / c$ [rad]")
    ax2.set_ylabel("Normalized PSD $|E_x(\\delta L)|^2$ [-]")
    ax2.set_title("(b) Lasing output spectra")
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.subplots_adjust(wspace=0.35)
    out_path = fig_dir / "04_lasing_spectra.png"
    figstyle.save(fig, out_path, dpi=220, close=True, auto_legend_below=True)
    return out_path


def plot_05_shb_profile_and_li_curve(fig_dir: str | Path = FIG_DIR) -> Path:
    """Plot Figure 5: Longitudinal intensity / spatial hole burning profile and L-I curve."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "lasing_dynamics.npz")
    meta = json.loads((DATA_DIR / "lasing_dynamics.json").read_text(encoding="utf-8"))

    figstyle.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.FULL_WIDTH_IN, 3.25))

    # (a) Longitudinal intensity profile & spatial hole burning at z = 0
    ax = axes[0]
    u = d["u_grat"]
    ax.plot(
        u,
        d["fit_I_raw_qws"],
        "-",
        color="#93C5FD",
        lw=0.6,
        alpha=0.75,
        label="FIT standing-wave $I(z)/I_{\\mathrm{out}}$",
    )
    ax.plot(
        u,
        d["fit_I_norm_qws"],
        "-",
        color="#0072B2",
        lw=2.1,
        label=f"FIT $\\lambda/4$ env. (peak/edge $= {meta['qws_peak_to_edge_fit']:.2f}$)",
    )
    ax.plot(
        u,
        d["cwt_I_norm_qws"],
        "--",
        color="#0F172A",
        lw=1.6,
        label=f"CWT $|R|^2+|S|^2$ (peak/edge $= {meta['qws_peak_to_edge_cwt']:.2f}$)",
    )
    ax.plot(
        u,
        d["fit_I_norm_uni"],
        "-.",
        color="#D55E00",
        lw=1.5,
        label="FIT uniform envelope",
    )
    ax.plot(
        u,
        d["g_sat_ratio_qws"],
        ":",
        color="#009E73",
        lw=1.9,
        label=f"Saturated gain $g(z)/g_0$ (min $= {meta['qws_central_gain_compression']:.2f}$)",
    )

    ax.axvline(0.0, color="#64748B", ls=":", lw=1.0)
    ax.set_xlabel("Normalized cavity position $z / L$ [-]")
    ax.set_ylabel("Intensity $I(z)/I_{\\mathrm{out}}$ or Gain $g(z)/g_0$ [-]")
    ax.set_title("(a) Spatial hole burning profile")
    ax.legend(loc="upper right", fontsize=8.5)

    # (b) Output intensity I_out / I_sat vs pump ratio g0 / g_th
    ax2 = axes[1]
    ax2.axvline(1.0, color="#64748B", ls=":", lw=1.0, label="Threshold $g_0 / g_{\\mathrm{th}} = 1$")
    ax2.plot(
        d["pump_dense"],
        d["I_out_linear_dense"],
        ":",
        color="#64748B",
        lw=1.4,
        label="Rate-eq. linear $\\eta_{\\mathrm{SHB}}(g_0/g_{\\mathrm{th}}-1)$",
    )
    ax2.plot(
        d["pump_dense"],
        d["I_out_rigrod_env_dense"],
        "--",
        color="#009E73",
        lw=1.5,
        label="CWT Rigrod (envelope)",
    )
    ax2.plot(
        d["pump_dense"],
        d["I_out_rigrod_sw_dense"],
        "-",
        color="#0F172A",
        lw=1.8,
        label=f"CWT Rigrod + SHB (slope $= {meta['slope_rigrod_sw']:.3f}$)",
    )
    ax2.plot(
        d["pump_pts"],
        d["I_out_over_Isat_fit"],
        "o",
        color="#0072B2",
        ms=6.5,
        mec="black",
        mew=0.6,
        label=f"1D FIT $\\lambda/4$ DFB (slope $= {meta['slope_fit']:.3f}$)",
    )
    ax2.set_xlabel("Normalized pump $g_0 / g_{\\mathrm{th}}$ [-]")
    ax2.set_ylabel("Single-facet output $I_{\\mathrm{out}} / I_{\\mathrm{sat}}$ [-]")
    ax2.set_title("(b) L–I curve vs Rigrod theory")
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.subplots_adjust(wspace=0.35)
    out_path = fig_dir / "05_shb_profile_and_li_curve.png"
    figstyle.save(fig, out_path, dpi=220, close=True, auto_legend_below=True)
    return out_path


def plot_06_mutations(fig_dir: str | Path = FIG_DIR) -> Path:
    """Plot Figure 6: Mutation checks (phase shift != pi/2, kappa sign flip, gain off)."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "mutations.npz")
    d_las = np.load(DATA_DIR / "lasing_dynamics.npz")
    meta = json.loads((DATA_DIR / "mutations.json").read_text(encoding="utf-8"))

    figstyle.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.FULL_WIDTH_IN, 3.25))

    # (a) Phase shift Phi != pi/2 mode detuning & threshold gain
    ax = axes[0]
    ax.axhline(0.0, color="#64748B", ls=":", lw=0.9)
    ax.axvline(90.0, color="#009E73", ls="--", lw=1.0, label="$\\Phi = \\pi/2$ ($\\lambda/4$)")
    ax.plot(
        d["phi_dense_deg"],
        d["cwt_phi_d"],
        "-",
        color="#0072B2",
        lw=1.9,
        label="CWT $\\delta_{\\mathrm{th}} L(\\Phi)$",
    )
    ax.plot(
        d["phi_pts_deg"],
        d["fit_phi_d"],
        "o",
        color="#0072B2",
        ms=6.0,
        mec="black",
        mew=0.5,
        label="FIT $\\delta_{\\mathrm{th}} L$",
    )
    ax.plot(
        d["phi_dense_deg"],
        d["cwt_phi_a"],
        "--",
        color="#D55E00",
        lw=1.6,
        label="CWT $\\alpha_{\\mathrm{th}} L(\\Phi)$",
    )
    ax.plot(
        d["phi_pts_deg"],
        d["fit_phi_a"],
        "s",
        color="#D55E00",
        ms=5.5,
        mec="black",
        mew=0.5,
        label="FIT $\\alpha_{\\mathrm{th}} L$",
    )
    ax.set_xlabel("Phase shift $\\Phi$ [deg]")
    ax.set_ylabel("Detuning $\\delta_{\\mathrm{th}} L$ / Gain $\\alpha_{\\mathrm{th}} L$")
    ax.set_title("(a) $\\Phi \\neq \\pi/2$ splitting")
    ax.legend(loc="upper left", fontsize=8.5)

    # (b) Coupling sign flip kappa -> -kappa standing-wave parity at z = 0
    ax2 = axes[1]
    ax2.axvline(0.0, color="#64748B", ls="--", lw=1.1, label="Defect $z = 0$")
    ax2.plot(
        d["z_zoom_over_Lambda"],
        d["prof_pos"],
        "-",
        color="#0072B2",
        lw=1.9,
        label=f"$\\kappa > 0$ ($|E_x(0)|={meta['kappa_pos_center_field_norm']:.2f}$)",
    )
    ax2.plot(
        d["z_zoom_over_Lambda"],
        d["prof_neg"],
        "--",
        color="#D55E00",
        lw=1.8,
        label=f"$\\kappa < 0$ ($|E_x(0)|={meta['kappa_neg_center_field_norm']:.2f}$)",
    )
    ax2.set_xlabel("Position $z / \\Lambda$ [-]")
    ax2.set_ylabel("Normalized $E_x(z)$ [-]")
    ax2.set_title("(b) $\\kappa \\to -\\kappa$ parity")
    ax2.legend(loc="lower right", fontsize=8.5)

    # (c) Gain off (g0 = 0) vs gain on (g0 = 1.6 g_th)
    ax3 = axes[2]
    ax3.semilogy(
        d_las["t_ps_sub"],
        np.maximum(d_las["env_qws_t"], 1e-6),
        "-",
        color="#0072B2",
        lw=1.9,
        label="Gain on ($g_0 = 1.6\\,g_{\\mathrm{th}}$)",
    )
    ax3.semilogy(
        d["t_off_ps"],
        np.maximum(d["env_off_t"], 1e-6),
        "--",
        color="#D55E00",
        lw=1.8,
        label="Gain off ($g_0 = 0$)",
    )
    ax3.set_ylim(1e-5, 1e5)
    ax3.set_xlabel("Time $t$ [ps]")
    ax3.set_ylabel("Envelope $|E_x(z_{\\mathrm{out}}, t)|$ [V/m]")
    ax3.set_title("(c) Gain off $\\Rightarrow$ decay")
    ax3.legend(loc="center right", fontsize=8.5)

    fig.subplots_adjust(wspace=0.40)
    out_path = fig_dir / "06_mutations.png"
    figstyle.save(fig, out_path, dpi=220, close=True, auto_legend_below=True)
    return out_path


def generate_07_buildup_gif(fig_dir: str | Path = FIG_DIR) -> Path:
    """Generate animated GIF ``07_dfb_lasing_buildup.gif`` of the lasing mode build-up."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "lasing_dynamics.npz")

    u = d["u_grat"]
    frames_t = d["frames_t_ps"]
    frames_qws = d["frames_qws_I_env"]
    frames_uni = d["frames_uni_I_env"]
    y_max = float(max(np.max(frames_qws), np.max(frames_uni)) * 1.12)

    figstyle.apply_style()
    pil_frames: list[Image.Image] = []
    for k in range(len(frames_t)):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), dpi=110)
        ax1, ax2 = axes
        ax1.axvline(0.0, color="#64748B", ls="--", lw=1.0, label="$\\lambda/4$ phase shift ($z=0$)")
        ax1.plot(
            u,
            frames_qws[k],
            "-",
            color="#0072B2",
            lw=2.0,
            label="$\\lambda/4$-shifted $I_{\\mathrm{env}}(z)/I_{\\mathrm{sat}}$",
        )
        ax1.plot(
            u,
            frames_uni[k],
            "--",
            color="#D55E00",
            lw=1.7,
            label="Uniform DFB $I_{\\mathrm{env}}(z)/I_{\\mathrm{sat}}$",
        )
        ax1.set_xlim(-0.5, 0.5)
        ax1.set_ylim(0.0, y_max)
        ax1.set_xlabel("Cavity position $z / L$ [-]")
        ax1.set_ylabel("Intracavity $I_{\\mathrm{env}}(z) / I_{\\mathrm{sat}}$ [-]")
        ax1.set_title(f"Intracavity intensity build-up ($t = {frames_t[k]:.2f}$ ps)")
        ax1.legend(loc="upper right", fontsize=6.8, frameon=True)

        g_qws = 1.0 / (1.0 + frames_qws[k])
        g_uni = 1.0 / (1.0 + frames_uni[k])
        ax2.axhline(1.0 / 1.6, color="#64748B", ls=":", lw=1.0, label="Threshold $g_{\\mathrm{th}}/g_0 = 0.625$")
        ax2.plot(u, g_qws, "-", color="#009E73", lw=2.0, label="$\\lambda/4$-shifted $g(z)/g_0$ (SHB hole)")
        ax2.plot(u, g_uni, "--", color="#D55E00", lw=1.6, label="Uniform DFB $g(z)/g_0$")
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_ylim(0.40, 1.05)
        ax2.set_xlabel("Cavity position $z / L$ [-]")
        ax2.set_ylabel("Normalized gain $g(z, t) / g_0$ [-]")
        ax2.set_title("Spatial hole burning saturation")
        ax2.legend(loc="lower right", fontsize=6.8, frameon=True)

        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_frames.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)

    out_gif = fig_dir / "07_dfb_lasing_buildup.gif"
    pil_frames[0].save(
        out_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    return out_gif


def plot_08_lasing_from_noise(fig_dir: str | Path = FIG_DIR) -> Path:
    """Plot Figure 8: Self-started lasing from spontaneous-emission noise (``FourLevelGain``)."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "lasing_from_noise.npz")
    meta = json.loads((DATA_DIR / "lasing_from_noise.json").read_text(encoding="utf-8"))

    figstyle.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.FULL_WIDTH_IN, 3.8))

    # (a) Population inversion clamping <dN>/dN_th and output envelope from noise
    ax = axes[0]
    ax.axhline(1.0, color="#64748B", ls=":", lw=1.1, label="Threshold $\\Delta N_{\\mathrm{th}}$")
    ax.plot(
        d["t_ps"],
        d["dn_qws_savg"],
        "-",
        color="#0072B2",
        lw=1.9,
        label=f"Modal $\\langle\\Delta N\\rangle / \\Delta N_{{\\mathrm{{th}}}}$ ($\\to {meta['qws_savg_dn_clamped_over_th']:.2f}$)",
    )
    ax.plot(
        d["t_ps"],
        d["dn_qws_shb"],
        "--",
        color="#009E73",
        lw=1.7,
        label=f"SHB $\\langle\\Delta N\\rangle / \\Delta N_{{\\mathrm{{th}}}}$ ($\\to {meta['qws_shb_dn_mean_over_th']:.2f}$)",
    )
    env_norm = d["env_qws_shb"] / max(float(np.max(d["env_qws_shb"])), 1e-30) * 1.5
    ax.plot(
        d["t_ps"],
        env_norm,
        "-.",
        color="#D55E00",
        lw=1.4,
        alpha=0.85,
        label="Normalized $|E_x(z_{\\mathrm{out}}, t)|$",
    )
    ax.set_ylim(0.0, 2.15)
    ax.set_xlabel("Time $t$ [ps]")
    ax.set_ylabel("Inversion $\\langle\\Delta N\\rangle / \\Delta N_{\\mathrm{th}}$ [-]")
    ax.set_title("(a) Noise clamping")
    figstyle.place_legend_below(ax, ncol=1, y_offset=-0.24, fontsize=8.5)

    # (b) Self-started output spectra (QWS single mode vs uniform gain-pulled doublet)
    ax2 = axes[1]
    ax2.axvspan(-2.0, 2.0, color="#E2E8F0", alpha=0.75, label="Stopband $|\\delta L| \\leq 2$")
    cwt_d_undisp = float(meta["cwt_uni_delta_L"])
    cwt_d_disp = float(meta["cwt_uni_disp_delta_L"])
    for sgn_i, sgn in enumerate((-1.0, 1.0)):
        ax2.axvline(
            sgn * cwt_d_undisp,
            color="#64748B",
            ls=":",
            lw=1.0,
            label=f"Dispersionless CWT ($\\pm {cwt_d_undisp:.2f}$ rad)" if sgn_i == 0 else None,
        )
        ax2.axvline(
            sgn * cwt_d_disp,
            color="#009E73",
            ls="-.",
            lw=1.2,
            label=f"Gain-dispersive CWT ($\\pm {cwt_d_disp:.2f}$ rad)" if sgn_i == 0 else None,
        )
    ax2.semilogy(
        d["delta_L_axis"],
        np.maximum(d["spec_qws_shb"], 1e-6),
        "-",
        color="#0072B2",
        lw=1.9,
        label=f"$\\lambda/4$ noise ($\\delta_0 L = {meta['qws_peak_delta_L']:+.2f}$, {meta['qws_smsr_db']:.1f} dB)",
    )
    ax2.semilogy(
        d["delta_L_axis"],
        np.maximum(d["spec_uni_shb"], 1e-6),
        "--",
        color="#D55E00",
        lw=1.6,
        label=f"Uniform noise ($\\pm {meta['uni_half_splitting_delta_L']:.2f}$ rad)",
    )
    ax2.set_xlim(-7.5, 7.5)
    ax2.set_ylim(1e-5, 2.0)
    ax2.set_xlabel("Normalized detuning $\\delta L$ [rad]")
    ax2.set_ylabel("Normalized PSD $|E_x(\\delta L)|^2$ [-]")
    ax2.set_title("(b) Output spectra")
    figstyle.place_legend_below(ax2, ncol=1, y_offset=-0.24, fontsize=8.0)

    # (c) Clamped inversion spatial hole burning profile across the grating
    ax3 = axes[2]
    ax3.axhline(1.0, color="#64748B", ls=":", lw=1.0, label="$\\Delta N / \\Delta N_{\\mathrm{th}} = 1$")
    ax3.plot(
        d["u_grat"],
        d["dn_prof_raw"],
        "-",
        color="#93C5FD",
        lw=0.6,
        alpha=0.7,
        label="Local $\\Delta N(z)/\\Delta N_{\\mathrm{th}}$ (standing wave)",
    )
    ax3.plot(
        d["u_grat"],
        d["dn_prof_shb"],
        "-",
        color="#0072B2",
        lw=2.0,
        label=f"Period-avg $\\Delta N(z)/\\Delta N_{{\\mathrm{{th}}}}$ (min $= {meta['qws_shb_dn_center_over_th']:.2f}$)",
    )
    ax3.set_xlim(-0.5, 0.5)
    ax3.set_ylim(0.5, 1.9)
    ax3.set_xlabel("Normalized position $z / L$ [-]")
    ax3.set_ylabel("Clamped $\\Delta N(z) / \\Delta N_{\\mathrm{th}}$ [-]")
    ax3.set_title("(c) Carrier hole burning")
    figstyle.place_legend_below(ax3, ncol=1, y_offset=-0.24, fontsize=8.5)

    fig.subplots_adjust(wspace=0.40, bottom=0.33)
    out_path = fig_dir / "08_lasing_from_noise.png"
    figstyle.save(fig, out_path, dpi=220, close=True, auto_legend_below=False)
    return out_path


# ==============================================================================
# 5. Quantitative Summary Table Builder (from data/*.json)
# ==============================================================================


def build_summary_rows(data_dir: str | Path = DATA_DIR) -> list[dict[str, str]]:
    """Build the Quantitative Summary rows directly from ``data/*.json``.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing ``cwt_and_linear_gain.json``, ``convergence.json``,
        ``lasing_dynamics.json``, and ``mutations.json``.

    Returns
    -------
    rows : list of dict
        List of row dictionaries with keys ``'Quantity'``, ``'Theory / CWT'``,
        ``'Wakis FIT'``, and ``'Rel. Error / Status'``.
    """
    data_dir = Path(data_dir)
    cwt = json.loads((data_dir / "cwt_and_linear_gain.json").read_text(encoding="utf-8"))
    conv = json.loads((data_dir / "convergence.json").read_text(encoding="utf-8"))
    las = json.loads((data_dir / "lasing_dynamics.json").read_text(encoding="utf-8"))
    mut = json.loads((data_dir / "mutations.json").read_text(encoding="utf-8"))
    noise = json.loads((data_dir / "lasing_from_noise.json").read_text(encoding="utf-8"))

    return [
        {
            "Quantity": "1D $\\lambda/4$-shifted threshold gain $\\alpha_{\\mathrm{th},0} L$ ($\\kappa L = 2.0$, $N_\\Lambda = 24$)",
            "Theory / CWT": f"{cwt['cwt_qws_kL2_alpha_L']:.4f}",
            "Wakis FIT": f"{cwt['fit_qws_kL2_alpha_L']:.4f}",
            "Rel. Error / Status": f"{cwt['rel_err_qws_kL2_alpha_pct']:.2f}%",
        },
        {
            "Quantity": "1D $\\lambda/4$-shifted linear-gain zero-crossing $\\alpha_{\\mathrm{th},0} L$ ($\\kappa L = 2.0$)",
            "Theory / CWT": f"{cwt['cwt_qws_kL2_alpha_L']:.4f}",
            "Wakis FIT": f"{cwt['fit_qws_zc_alpha_L']:.4f}",
            "Rel. Error / Status": f"{abs(cwt['fit_qws_zc_alpha_L'] - cwt['cwt_qws_kL2_alpha_L']) / cwt['cwt_qws_kL2_alpha_L'] * 100.0:.2f}%",
        },
        {
            "Quantity": "1D $\\lambda/4$-shifted Bragg defect detuning $\\delta_0 L$ [rad] ($\\kappa L = 2.0$)",
            "Theory / CWT": f"{cwt['cwt_qws_kL2_delta_L']:+.4f}",
            "Wakis FIT": f"{cwt['fit_qws_kL2_delta_L']:+.4f}",
            "Rel. Error / Status": f"|Δ(δL)| = {abs(cwt['fit_qws_kL2_delta_L']):.4f} rad",
        },
        {
            "Quantity": "1D $\\lambda/4$-shifted first side-mode $\\alpha_{\\mathrm{th},1} L, |\\delta_{\\mathrm{th},1} L|$ ($\\kappa L = 2.0$, $m = \\pm 1$)",
            "Theory / CWT": f"{mut['cwt_side_m1_alpha_L']:.4f}, {mut['cwt_side_m1_delta_L']:.4f}",
            "Wakis FIT": f"{mut['kappa_neg_center_excited_alpha_L']:.4f}, {mut['kappa_neg_center_excited_delta_L']:.4f}",
            "Rel. Error / Status": f"{abs(mut['kappa_neg_center_excited_alpha_L'] - mut['cwt_side_m1_alpha_L']) / mut['cwt_side_m1_alpha_L'] * 100.0:.2f}%, {abs(mut['kappa_neg_center_excited_delta_L'] - mut['cwt_side_m1_delta_L']) / mut['cwt_side_m1_delta_L'] * 100.0:.2f}%",
        },
        {
            "Quantity": "1D uniform DFB threshold gain $\\alpha_{\\mathrm{th},1} L$ ($\\kappa L = 2.0$, $m = \\pm 1$)",
            "Theory / CWT": f"{cwt['cwt_uni_kL2_alpha_L']:.4f}",
            "Wakis FIT": f"{cwt['fit_uni_kL2_alpha_L']:.4f}",
            "Rel. Error / Status": f"{cwt['rel_err_uni_kL2_alpha_pct']:.2f}%",
        },
        {
            "Quantity": "1D uniform DFB stopband-edge detuning $|\\delta_{\\mathrm{th},1} L|$ [rad] ($\\kappa L = 2.0$)",
            "Theory / CWT": f"{cwt['cwt_uni_kL2_delta_L']:.4f}",
            "Wakis FIT": f"{cwt['fit_uni_kL2_delta_L']:.4f}",
            "Rel. Error / Status": f"{cwt['rel_err_uni_kL2_delta_pct']:.2f}%",
        },
        {
            "Quantity": "2D TE$_0$ slab-waveguide $\\lambda/4$ DFB $\\alpha_{\\mathrm{th},0} L$ ($\\kappa L = 2.0$, $\\Gamma = "
            + f"{cwt['slab_2d_Gamma_core']:.3f}$)",
            "Theory / CWT": f"{cwt['cwt_qws_kL2_alpha_L']:.4f}",
            "Wakis FIT": f"{cwt['slab_2d_alpha_th_L']:.4f}",
            "Rel. Error / Status": f"{cwt['slab_2d_rel_err_alpha_pct']:.2f}%",
        },
        {
            "Quantity": "Finest-grid $\\lambda/4$ DFB $\\alpha_{\\mathrm{th},0} L$ ($N_\\Lambda = 48$, $96\\;\\mathrm{cells}/\\lambda$)",
            "Theory / CWT": f"{conv['cwt_qws_alpha_L']:.5f}",
            "Wakis FIT": f"{conv['qws_alpha_L'][-1]:.5f}",
            "Rel. Error / Status": f"{conv['finest_qws_alpha_rel_err_pct']:.3f}% (order p = {conv['p_obs_qws_alpha']:.2f})",
        },
        {
            "Quantity": "Above-threshold $\\lambda/4$ DFB lasing detuning $\\delta_0 L$ [rad] ($g_0/g_{\\mathrm{th}} = 1.6$)",
            "Theory / CWT": "+0.0000",
            "Wakis FIT": f"{las['qws_peak_delta_L']:+.4f}",
            "Rel. Error / Status": f"Single-mode (SMSR = {las['qws_smsr_db']:.1f} dB)",
        },
        {
            "Quantity": "Above-threshold uniform DFB doublet $|\\delta_{\\pm 1} L|$ [rad] ($g_0/g_{\\mathrm{th}} = 1.6$)",
            "Theory / CWT": f"{las['cwt_uni_delta_L']:.4f}",
            "Wakis FIT": f"{las['uni_half_splitting_delta_L']:.4f}",
            "Rel. Error / Status": f"{abs(las['uni_half_splitting_delta_L'] - las['cwt_uni_delta_L']) / las['cwt_uni_delta_L'] * 100.0:.2f}% (two-mode)",
        },
        {
            "Quantity": "Defect intensity localization $I_{\\mathrm{env}}(0) / I_{\\mathrm{out}}$ ($\\kappa L = 2.0$)",
            "Theory / CWT": f"{las['qws_peak_to_edge_cwt']:.4f}",
            "Wakis FIT": f"{las['qws_peak_to_edge_fit']:.4f}",
            "Rel. Error / Status": f"{las['qws_peak_to_edge_rel_diff_pct']:.2f}% (SHB compression)",
        },
        {
            "Quantity": "L–I slope efficiency $\\mathrm{d}(I_{\\mathrm{out}}/I_{\\mathrm{sat}})/\\mathrm{d}(g_0/g_{\\mathrm{th}})$",
            "Theory / CWT": f"{las['slope_rigrod_sw']:.4f}",
            "Wakis FIT": f"{las['slope_fit']:.4f}",
            "Rel. Error / Status": f"{las['slope_rel_diff_pct']:.2f}%",
        },
        {
            "Quantity": "Lasing from noise (`FourLevelGain`, $s = 100$): clamped $\\langle\\Delta N\\rangle / \\Delta N_{\\mathrm{th}}$",
            "Theory / CWT": "1.0000",
            "Wakis FIT": f"{noise['qws_savg_dn_clamped_over_th']:.4f}",
            "Rel. Error / Status": f"{noise['qws_savg_dn_clamping_err_pct']:.2f}% (self-started, SMSR = {noise['qws_smsr_db']:.1f} dB)",
        },
        {
            "Quantity": "Lasing from noise (`FourLevelGain`, modal avg, $r_p = 2.0$): output $I_{\\mathrm{out}} / I_{\\mathrm{sat}}$",
            "Theory / CWT": f"{noise['qws_savg_theory_I_out_over_Isat']:.4f}",
            "Wakis FIT": f"{noise['qws_savg_I_out_over_Isat']:.4f}",
            "Rel. Error / Status": f"{noise['qws_savg_vs_theory_rel_err_pct']:.2f}% ($s = 50$: {noise['time_scale_I_out_over_Isat'][1]:.4f})",
        },
        {
            "Quantity": "Lasing from noise (`FourLevelGain`, SHB, $r_p = 2.0$): output $I_{\\mathrm{out}} / I_{\\mathrm{sat}}$",
            "Theory / CWT": f"{noise['qws_rigrod_sw_I_out_over_Isat']:.4f}",
            "Wakis FIT": f"{noise['qws_shb_I_out_over_Isat']:.4f}",
            "Rel. Error / Status": f"{noise['qws_shb_vs_rigrod_rel_err_pct']:.2f}% ($s \\in \\{{25,50,100\\}}$ spread {noise['time_scale_I_out_spread_pct']:.2f}%)",
        },
        {
            "Quantity": "Lasing from noise (`FourLevelGain`, uniform DFB): gain-pulled doublet $|\\delta_{\\pm 1} L|$ [rad]",
            "Theory / CWT": f"{noise['cwt_uni_disp_delta_L']:.4f}",
            "Wakis FIT": f"{noise['uni_half_splitting_delta_L']:.4f}",
            "Rel. Error / Status": f"within one bin (2.6 %; pulled from CWT {noise['cwt_uni_delta_L']:.2f})",
        },
        {
            "Quantity": "Mutation: phase shift $\\Phi = \\pi/4$ vs $3\\pi/4$ splitting $\\Delta(\\delta L)$ [rad]",
            "Theory / CWT": f"{mut['phi_45_splitting_cwt_delta_L']:.4f}",
            "Wakis FIT": f"{mut['phi_45_splitting_fit_delta_L']:.4f}",
            "Rel. Error / Status": f"{abs(mut['phi_45_splitting_fit_delta_L'] - mut['phi_45_splitting_cwt_delta_L']) / mut['phi_45_splitting_cwt_delta_L'] * 100.0:.2f}%",
        },
    ]


def format_summary_table(data_dir: str | Path = DATA_DIR) -> str:
    """Format the Quantitative Summary Markdown table from ``data/*.json``."""
    rows = build_summary_rows(data_dir=data_dir)
    lines = [
        "| Quantity | Theory / CWT | Wakis FIT | Rel. Error / Status |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['Quantity']} | {r['Theory / CWT']} | {r['Wakis FIT']} | {r['Rel. Error / Status']} |"
        )
    return "\n".join(lines)


def run_all_tasks_and_figures(force: bool = False) -> dict[str, Any]:
    """Execute all simulation tasks sequentially and generate all setup/result figures and GIF."""
    generate_setup_figures(FIG_DIR)
    s0 = run_task_00_cwt_and_linear_gain(quick=False, force=force)
    s1 = run_task_01_convergence(quick=False, force=force)
    s2 = run_task_02_lasing_dynamics(quick=False, force=force)
    s3 = run_task_03_mutations(quick=False, force=force)
    s4 = run_task_04_lasing_from_noise(quick=False, force=force)
    plot_01_cwt_and_linear_gain(FIG_DIR)
    plot_02_2d_slab_mode(FIG_DIR)
    plot_03_convergence(FIG_DIR)
    plot_04_lasing_spectra(FIG_DIR)
    plot_05_shb_profile_and_li_curve(FIG_DIR)
    plot_06_mutations(FIG_DIR)
    generate_07_buildup_gif(FIG_DIR)
    plot_08_lasing_from_noise(FIG_DIR)
    return {
        "cwt_and_linear_gain": s0,
        "convergence": s1,
        "lasing_dynamics": s2,
        "mutations": s3,
        "lasing_from_noise": s4,
    }
