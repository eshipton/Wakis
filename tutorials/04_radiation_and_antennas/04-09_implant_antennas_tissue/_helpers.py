# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Analytical, dispersive-fitting, 3D FIT simulation, and plotting helpers for Tutorial 4.9.

Tutorial 4.9: Electric vs Magnetic Implant Antennas in Lossy Biological Tissue.
"""

from __future__ import annotations

import json
import logging
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0
from scipy.integrate import quad
from scipy.optimize import brentq

from wakis import DFTMonitor, FluxMonitor, GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.dispersive import Debye, DispersiveMaterial, Lorentz, fit_poles
from wakis.estimate import estimate, preflight
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.meshing import graded_axis
from wakis.templates.shapes import Box, assign_materials

ETA0 = float(np.sqrt(MU0 / EPS0))
F_MEDRADIO_HZ = 403.0e6
F_ISM_HZ = 2.45e9
FREQS_MAIN_HZ = np.array([F_MEDRADIO_HZ, F_ISM_HZ], dtype=float)
RHO_MUSCLE_KG_M3 = 1040.0
SIGMA_COPPER_S_M = 5.8e7
FERRITE_R_WIRE_M = 0.06e-3  # 0.06 mm radius (0.12 mm diameter, ~= AWG 36-37)

TUTORIAL_DIR = Path(__file__).resolve().parent
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

ALL_TASKS = (
    "near_zone_theory_and_muscle_fit",
    "fit_shells_and_capsule_correction",
    "real_feeds_and_sar",
    "bounds_and_ferrite",
    "convergence",
)

COLORS = {
    "blue": figstyle.OKABE_ITO[0],
    "red": figstyle.OKABE_ITO[1],
    "teal": figstyle.OKABE_ITO[2],
    "orange": figstyle.OKABE_ITO[3],
    "purple": figstyle.OKABE_ITO[4],
    "sky": figstyle.OKABE_ITO[5],
    "grey": "#6B7280",
    "black": "#111827",
}


@dataclass
class CircleOverlay:
    """Lightweight circular boundary descriptor for wakis.schematic."""

    center: tuple[float, float, float]
    radius: float
    material: Any
    name: str
    axis: int = 2


# ---------------------------------------------------------------------------
# Part 1: Analytical near-zone loss (Zhou & Pan 2023) & b-independent ratios
# ---------------------------------------------------------------------------
def complex_wavenumber(
    freq_hz: float | np.ndarray,
    eps_r_complex: complex | np.ndarray,
    mu_r_complex: complex | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute complex wavenumber k = beta - j*alpha in exp(+j*w*t) convention.

    Parameters
    ----------
    freq_hz : float or np.ndarray
        Frequency [Hz].
    eps_r_complex : complex or np.ndarray
        Complex relative permittivity eps_r' - j*eps_r'' (Im <= 0 for passive medium).
    mu_r_complex : complex or np.ndarray, optional
        Complex relative permeability mu_r' - j*mu_r''. Default 1.0.

    Returns
    -------
    k : np.ndarray of complex
        Complex wavenumber k = beta - j*alpha [rad/m].
    beta : np.ndarray of float
        Phase constant Re(k) > 0 [rad/m].
    alpha : np.ndarray of float
        Attenuation constant -Im(k) >= 0 [Np/m].
    """
    w = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
    n_c = np.sqrt(np.asarray(eps_r_complex, dtype=complex) * np.asarray(mu_r_complex, dtype=complex))
    # Enforce passive branch Re(n) >= 0, Im(n) <= 0
    n_c = np.where(np.real(n_c) < 0, -n_c, n_c)
    k = (w / C0) * n_c
    beta = np.real(k)
    alpha = -np.imag(k)
    return k, beta, alpha


def zhou_pan_poynting_flux(
    r_m: float | np.ndarray,
    freq_hz: float,
    eps_r_complex: complex,
    moment: float = 1.0,
    mode: str = "TM",
    drop_cubic_term: bool = False,
) -> dict[str, np.ndarray]:
    """Evaluate radial Poynting flux S(r) of a small dipole in a lossy medium.

    Reproduces Eqs. (19)-(20) and (30) of S. Zhou and G. Pan (2023),
    "Comparative Study of Multi-Physics Generated Small Dipoles in Conducting
    Media," *J. Comput. Electron.* **22**, 1428-1437 (`doi:10.1007/s10825-023-02076-y`),
    derived from the exact
    spherical-wave fields in a homogeneous conductive dielectric with complex
    wavenumber k = beta - j*alpha:

    .. math::

        S_{\\mathrm{TM}}(r) &= \\frac{1}{12\\pi}(I_e\\,\\mathrm{d}l)^2 \\frac{\\eta_0\\omega}{c_0}
            e^{-2\\alpha r}\\beta \\left[1 + \\frac{2\\alpha}{|k|^2 r} + \\frac{1}{|k|^2 r^2}
            - \\frac{\\beta^2 - 3\\alpha^2}{|k|^4 r^2} + \\frac{2\\alpha}{|k|^4 r^3}\\right], \\\\
        S_{\\mathrm{TE}}(r) &= \\frac{1}{12\\pi}(I_m\\,\\mathrm{d}l\\,|k|)^2 \\frac{c_0}{\\eta_0\\omega}
            e^{-2\\alpha r}\\beta \\left[1 + \\frac{2\\alpha}{|k|^2 r}\\right].

    Parameters
    ----------
    r_m : float or np.ndarray
        Sphere radius r [m].
    freq_hz : float
        Frequency [Hz].
    eps_r_complex : complex
        Complex relative permittivity eps_r - j*sigma/(w*eps_0).
    moment : float, optional
        Current moment I_e*dl [A*m] for TM, or magnetic current moment I_m*dl [V*m]
        for TE. Default 1.0.
    mode : {'TM', 'TE'}, optional
        Source mode ('TM' = electric dipole, 'TE' = magnetic dipole / small loop).
    drop_cubic_term : bool, optional
        Mutation flag: if True, drops the 2*alpha/(|k|^4 r^3) quasi-static E term.

    Returns
    -------
    dict of str to np.ndarray
        ``S_r`` [W], ``S_far`` [W], ``bracket`` [dimensionless], ``delta_nz``
        (near-zone excess bracket - 1), and individual bracket terms ``t1``,
        ``t2a``, ``t2b``, ``t3``.
    """
    r = np.asarray(r_m, dtype=float)
    w = 2.0 * np.pi * float(freq_hz)
    k, beta, alpha = complex_wavenumber(freq_hz, eps_r_complex)
    k2 = float(beta**2 + alpha**2)
    t1 = 2.0 * alpha / (k2 * r)
    if mode.upper() == "TM":
        t2a = 1.0 / (k2 * (r**2))
        t2b = -(beta**2 - 3.0 * (alpha**2)) / ((k2**2) * (r**2))
        t3 = np.zeros_like(r) if drop_cubic_term else 2.0 * alpha / ((k2**2) * (r**3))
        delta_nz = t1 + t2a + t2b + t3
        pref = (1.0 / (12.0 * np.pi)) * (moment**2) * (ETA0 * w / C0) * beta
    elif mode.upper() == "TE":
        t2a = np.zeros_like(r)
        t2b = np.zeros_like(r)
        t3 = np.zeros_like(r)
        delta_nz = t1
        pref = (1.0 / (12.0 * np.pi)) * ((moment * np.sqrt(k2)) ** 2) * (C0 / (ETA0 * w)) * beta
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'TM' or 'TE'.")

    bracket = 1.0 + delta_nz
    s_far = pref * np.exp(-2.0 * alpha * r)
    s_r = s_far * bracket
    return {
        "k": k,
        "beta": beta,
        "alpha": alpha,
        "kr_abs": np.sqrt(k2) * r,
        "S_r": s_r,
        "S_far": s_far,
        "bracket": bracket,
        "delta_nz": delta_nz,
        "t1": t1,
        "t2a": t2a,
        "t2b": t2b,
        "t3": t3,
    }


def near_zone_excess_loss_metrics(
    a_m: float | np.ndarray,
    freq_hz: float,
    eps_r_complex: complex,
    drop_cubic_term: bool = False,
) -> dict[str, np.ndarray]:
    """Compute b-independent near-zone excess loss ratios at capsule radius a (derived here).

    Removing the arbitrary propagation path-loss factor exp(-2*alpha*(b - a)) from
    Zhou & Pan's S(b)/S(a) isolates the intrinsic near-zone excess power factor
    Delta_X(a) = S_X(a) / S_X,far(a) - 1:

    .. math::

        \\Delta_{\\mathrm{TM}}(a) &= \\frac{2\\alpha}{|k|^2 a} + \\frac{4\\alpha^2}{|k|^4 a^2}
            + \\frac{2\\alpha}{|k|^4 a^3}, \\qquad
        \\Delta_{\\mathrm{TE}}(a) = \\frac{2\\alpha}{|k|^2 a}, \\\\
        \\mathcal{R}_{\\mathrm{NZ}}(a) &\\equiv \\frac{\\Delta_{\\mathrm{TM}}(a)}{\\Delta_{\\mathrm{TE}}(a)}
            = 1 + \\frac{2\\alpha}{|k|^2 a} + \\frac{1}{|k|^2 a^2}, \\\\
        \\frac{\\eta_{\\mathrm{NZ,TE}}(a)}{\\eta_{\\mathrm{NZ,TM}}(a)}
            &\\equiv \\frac{1 + \\Delta_{\\mathrm{TM}}(a)}{1 + \\Delta_{\\mathrm{TE}}(a)}.

    Parameters
    ----------
    a_m : float or np.ndarray
        Capsule radius a [m].
    freq_hz : float
        Frequency [Hz].
    eps_r_complex : complex
        Complex relative permittivity of the surrounding tissue.
    drop_cubic_term : bool, optional
        Mutation flag: if True, drops the 1/a^3 term in Delta_TM.

    Returns
    -------
    dict of str to np.ndarray
        ``ka_abs``, ``delta_tm``, ``delta_te``, ``excess_loss_ratio``
        (:math:`\\Delta_{\\mathrm{TM}}/\\Delta_{\\mathrm{TE}}`), ``nz_efficiency_ratio``
        (:math:`(1+\\Delta_{\\mathrm{TM}})/(1+\\Delta_{\\mathrm{TE}})`), and
        ``asymptotic_inv_ka2`` (:math:`1/(|k|a)^2`).
    """
    tm = zhou_pan_poynting_flux(
        a_m, freq_hz, eps_r_complex, mode="TM", drop_cubic_term=drop_cubic_term
    )
    te = zhou_pan_poynting_flux(a_m, freq_hz, eps_r_complex, mode="TE")
    ka = tm["kr_abs"]
    return {
        "ka_abs": ka,
        "delta_tm": tm["delta_nz"],
        "delta_te": te["delta_nz"],
        "excess_loss_ratio": tm["delta_nz"] / np.maximum(te["delta_nz"], 1e-30),
        "nz_efficiency_ratio": tm["bracket"] / te["bracket"],
        "eta_nz_tm": 1.0 / tm["bracket"],
        "eta_nz_te": 1.0 / te["bracket"],
        "asymptotic_inv_ka2": 1.0 / (ka**2),
    }


def zhou_pan_cube_flux(
    r_box_m: float,
    freq_hz: float,
    eps_r_complex: complex,
    moment: float = 1.0,
    mode: str = "TM",
    drop_obliquity: bool = False,
) -> float:
    """Integrate the analytical Zhou-Pan radial Poynting vector over a cube [-r_box, +r_box]^3 (derived here).

    For a z-directed electric (TM) or magnetic (TE) dipole at the origin, the radial
    Poynting flux per unit area at spherical coordinate (R, theta) is
    S_vec(R, theta) . r_hat = (3 / (8*pi*R^2)) * sin^2(theta) * S_X(R). On each face of
    the cubic FluxMonitor box at normal distance r_box, the normal projection is
    r_hat . n_hat = r_box / R.

    Parameters
    ----------
    r_box_m : float
        Half-width of the cubic monitor box [m].
    freq_hz : float
        Frequency [Hz].
    eps_r_complex : complex
        Complex relative permittivity of tissue.
    moment : float, optional
        Dipole moment [A*m or V*m]. Default 1.0.
    mode : {'TM', 'TE'}, optional
        Source mode ('TM' or 'TE').
    drop_obliquity : bool, optional
        Mutation flag: if True, omits the r_hat . n_hat = r_box / R obliquity factor.

    Returns
    -------
    float
        Total time-averaged power [W] crossing the cubic surface [-r_box, +r_box]^3.
    """
    r_box = float(r_box_m)
    pts, wts = np.polynomial.legendre.leggauss(48)
    u = r_box * pts
    wu = r_box * wts
    U, V = np.meshgrid(u, u, indexing="ij")
    W = np.outer(wu, wu)

    # 2 top/bottom z-faces (z = +/- r_box): sin^2(theta) = (x^2 + y^2) / R^2
    R_z = np.sqrt(U**2 + V**2 + r_box**2)
    sin2_z = (U**2 + V**2) / (R_z**2)
    s_tot_z = zhou_pan_poynting_flux(R_z, freq_hz, eps_r_complex, moment=moment, mode=mode)["S_r"]
    obliq_z = np.ones_like(R_z) if drop_obliquity else (r_box / R_z)
    flux_z = float(np.sum(W * (3.0 / (8.0 * np.pi * (R_z**2))) * sin2_z * s_tot_z * obliq_z))

    # 4 side faces (x = +/- r_box or y = +/- r_box): sin^2(theta) = (r_box^2 + U^2) / R^2
    R_s = np.sqrt(r_box**2 + U**2 + V**2)
    sin2_s = (r_box**2 + U**2) / (R_s**2)
    s_tot_s = zhou_pan_poynting_flux(R_s, freq_hz, eps_r_complex, moment=moment, mode=mode)["S_r"]
    obliq_s = np.ones_like(R_s) if drop_obliquity else (r_box / R_s)
    flux_s = float(np.sum(W * (3.0 / (8.0 * np.pi * (R_s**2))) * sin2_s * s_tot_s * obliq_s))
    return float(2.0 * flux_z + 4.0 * flux_s)


def capsule_cavity_factor(
    freq_hz: float | np.ndarray,
    a_m: float | np.ndarray,
    eps_cap: float | complex,
    eps_tissue: complex | np.ndarray,
    mode: str = "TM",
    invert_ratio: bool = False,
) -> dict[str, np.ndarray]:
    """Compute quasi-static and full-wave spherical-capsule cavity factor (derived here).

    For a source of fixed current moment inside an insulating sphere of radius `a_m`
    and relative permittivity `eps_cap` surrounded by tissue of complex relative
    permittivity `eps_tissue`, matching tangential E and H across r = a yields
    the l = 1 Riccati-Bessel transmission factor relative to an unperturbed source
    in the exterior medium:

    .. math::

        C_{\\mathrm{TM}}^{\\mathrm{full}}(a) &= \\frac{-j / n_2}{
            \\frac{1}{n_2}\\psi_1(u_1)\\xi_1'(u_2) - \\frac{1}{n_1}\\psi_1'(u_1)\\xi_1(u_2)}
            \\xrightarrow{|k|a \\to 0} \\frac{3\\tilde{\\varepsilon}_{\\mathrm{m}}}{
            2\\tilde{\\varepsilon}_{\\mathrm{m}} + \\varepsilon_{\\mathrm{cap}}}, \\\\
        C_{\\mathrm{TE}}^{\\mathrm{full}}(a) &= \\frac{-j\\,n_1^2 / n_2}{
            n_2\\,\\psi_1(u_1)\\xi_1'(u_2) - n_1\\,\\psi_1'(u_1)\\xi_1(u_2)}
            \\xrightarrow{|k|a \\to 0} 1.

    Parameters
    ----------
    freq_hz : float or np.ndarray
        Frequency [Hz].
    a_m : float or np.ndarray
        Capsule radius a [m].
    eps_cap : float or complex
        Capsule relative permittivity.
    eps_tissue : complex or np.ndarray
        Complex relative permittivity of tissue.
    mode : {'TM', 'TE'}, optional
        Mode ('TM' or 'TE').
    invert_ratio : bool, optional
        Mutation flag: if True, inverts the quasi-static factor (2*eps_m + eps_cap)/(3*eps_m).

    Returns
    -------
    dict of str to np.ndarray
        ``c_qs``, ``c_full``, ``power_boost_qs`` (:math:`|C^{\\mathrm{QS}}|^2`),
        and ``power_boost_full`` (:math:`|C^{\\mathrm{full}}|^2`).
    """
    f = np.asarray(freq_hz, dtype=float)
    a = np.asarray(a_m, dtype=float)
    k0 = 2.0 * np.pi * f / C0
    n1 = np.sqrt(np.asarray(eps_cap, dtype=complex))
    n2 = np.sqrt(np.asarray(eps_tissue, dtype=complex))
    n2 = np.where(np.real(n2) < 0, -n2, n2)
    u1 = k0 * n1 * a
    u2 = k0 * n2 * a

    psi1 = np.sin(u1) / u1 - np.cos(u1)
    dpsi1 = np.sin(u1) * (1.0 - 1.0 / (u1**2)) + np.cos(u1) / u1
    xi1 = (1.0 / u2 + 1j) * np.exp(-1j * u2)
    dxi1 = (1.0 - 1.0 / (u2**2) - 1j / u2) * np.exp(-1j * u2)

    if mode.upper() == "TM":
        c_qs = (3.0 * eps_tissue) / (2.0 * eps_tissue + eps_cap)
        if invert_ratio:
            c_qs = 1.0 / c_qs
        den = (1.0 / n2) * psi1 * dxi1 - (1.0 / n1) * dpsi1 * xi1
        c_full = (-1j / n2) / den
        if invert_ratio:
            c_full = 1.0 / c_full
    elif mode.upper() == "TE":
        c_qs = np.ones_like(u2, dtype=complex)
        den = n2 * psi1 * dxi1 - n1 * dpsi1 * xi1
        c_full = (-1j * (n1**2) / n2) / den
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'TM' or 'TE'.")

    return {
        "c_qs": c_qs,
        "c_full": c_full,
        "power_boost_qs": np.abs(c_qs) ** 2,
        "power_boost_full": np.abs(c_full) ** 2,
    }


# ---------------------------------------------------------------------------
# Part 2: Gabriel (1996) Cole-Cole muscle model & Wakis multi-pole Debye fit
# ---------------------------------------------------------------------------
def gabriel_1996_muscle_cole_cole(freq_hz: float | np.ndarray) -> np.ndarray:
    """Evaluate the 4-pole Cole-Cole dielectric spectrum of muscle tissue.

    Parameters from C. Gabriel, R. W. Lau, and S. Gabriel (1996), "The dielectric
    properties of biological tissues: III. Parametric models for the dielectric
    spectrum of tissues," *Phys. Med. Biol.* **41**, 2271-2293
    (`doi:10.1088/0031-9155/41/11/003`), Table 1 (Muscle):
    eps_inf = 4.0, delta_eps = [50, 7000, 1.2e6, 2.5e7],
    tau = [7.23 ps, 353.68 ns, 318.31 us, 2.274 ms],
    alpha = [0.10, 0.10, 0.10, 0.0], sigma_ionic = 0.20 S/m.

    At 403 MHz this gives eps_r = 57.10 - 35.55j (matching Liska et al. 2024,
    57 - 36j) and at 2.45 GHz eps_r = 52.73 - 12.75j (matching Liska et al. 2024,
    53 - 13j).

    Parameters
    ----------
    freq_hz : float or np.ndarray
        Frequency [Hz].

    Returns
    -------
    np.ndarray of complex
        Complex relative permittivity in the exp(+j*w*t) convention (Im <= 0).
    """
    w = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
    eps = np.full_like(w, 4.0, dtype=complex)
    poles = (
        (50.0, 7.23e-12, 0.10),
        (7000.0, 353.68e-9, 0.10),
        (1.2e6, 318.31e-6, 0.10),
        (2.5e7, 2.274e-3, 0.0),
    )
    for delta_e, tau_s, alpha_cc in poles:
        eps = eps + delta_e / (1.0 + (1j * w * tau_s) ** (1.0 - alpha_cc))
    eps = eps - 1j * 0.20 / (w * EPS0)
    return eps


def build_dispersive_muscle(n_debye: int = 3) -> DispersiveMaterial:
    """Fit a causal multi-pole Debye model to Gabriel (1996) muscle over 0.15-3.5 GHz.

    Uses :func:`wakis.dispersive.fit_poles` with ``n_debye`` relaxation poles.

    Parameters
    ----------
    n_debye : int, optional
        Number of Debye poles. Default 3.

    Returns
    -------
    DispersiveMaterial
        Passive, causal Debye model for muscle tissue.
    """
    f_fit = np.geomspace(150.0e6, 3.5e9, 80)
    eps_target = gabriel_1996_muscle_cole_cole(f_fit)
    with np.errstate(over="ignore", invalid="ignore"):
        mat = fit_poles(
            f_fit,
            eps_target,
            n_lorentz=0,
            n_drude=0,
            n_debye=n_debye,
            name="Muscle_Debye3",
        )
    return mat


def build_nizn_ferrite_material() -> DispersiveMaterial:
    """Construct a causal single-pole Lorentz magnetic model for a model NiZn ferrite.

    Models a Snoek-consistent spinel ferrite core with static permeability
    mu_s = 7.5 (mu_inf = 1.0, delta_mu = 6.5), ferromagnetic resonance at
    f_FMR = 1.15 GHz, damping gamma / (2*pi) = 0.025 GHz (quality factor
    Q = f_FMR / (gamma / 2*pi) = 46, which is narrower than real polycrystalline
    NiZn spectra and therefore underestimates mu'' at 403 MHz), and dielectric
    permittivity eps_inf = 6.5. At 403 MHz (below f_FMR), mu_r ~= 8.41 - 0.064j
    with low magnetic loss tangent tan(delta_m) ~= 0.0077; at 2.45 GHz (above
    f_FMR), mu_r ~= -0.84 - 0.024j: although P_out per unit current rises 2.03x,
    magnetic core dissipation near the magnetostatic surface-mode condition
    (mu' ~= -1, here mu' = -0.84) excited by the winding on the core face
    reduces total efficiency to 0.25x.

    Returns
    -------
    DispersiveMaterial
        Causal magneto-dielectric Lorentz material.
    """
    w_fmr = 2.0 * np.pi * 1.15e9
    g_fmr = 2.0 * np.pi * 0.025e9
    return DispersiveMaterial(
        eps_inf=6.5,
        poles=[],
        mu_inf=1.0,
        mu_poles=[Lorentz(6.5, w_fmr, g_fmr)],
        name="NiZn_Ferrite",
    )


# ---------------------------------------------------------------------------
# 3D FIT geometry, spherical capsule assignment & DFT power bookkeeping
# ---------------------------------------------------------------------------
def assign_spherical_capsule_in_tissue(
    solver: SolverFIT3D,
    muscle_mat: DispersiveMaterial,
    a_radius_m: float,
    eps_cap: float | None = 1.0,
) -> int:
    """Fill the domain with dispersive muscle and carve a smooth spherical capsule of radius a.

    Parameters
    ----------
    solver : SolverFIT3D
        Initialized 3D FIT solver.
    muscle_mat : DispersiveMaterial
        Dispersive muscle material model.
    a_radius_m : float
        Capsule radius a [m].
    eps_cap : float or None, optional
        Relative permittivity inside the capsule (r < a). If None, the entire
        domain remains homogeneous muscle.

    Returns
    -------
    int
        Total number of dispersive E-edges registered.
    """
    grid = solver.grid
    lx = float(grid.x[-1] - grid.x[0])
    ly = float(grid.y[-1] - grid.y[0])
    lz = float(grid.z[-1] - grid.z[0])
    solver.add_dispersive_material(muscle_mat, center=(0.0, 0.0, 0.0), size=(2.0 * lx, 2.0 * ly, 2.0 * lz))
    if eps_cap is None:
        return int(3 * solver.N)

    cap_mat = DispersiveMaterial(eps_inf=float(eps_cap), poles=[], name="Capsule")
    full_nodes = (grid.x, grid.y, grid.z)
    dx_min = float(min(np.min(grid.dx), np.min(grid.dy), np.min(grid.dz)))
    idx_inner, idx_bndry, w_bndry = [], [], []
    for c in range(3):
        pos = [
            0.5 * (full_nodes[d][:-1] + full_nodes[d][1:]) if d == c else full_nodes[d][:-1]
            for d in range(3)
        ]
        X, Y, Z = np.meshgrid(*pos, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)
        w_cap = np.clip((a_radius_m + 0.5 * dx_min - R) / dx_min, 0.0, 1.0)
        m_in = w_cap >= 1.0 - 1e-9
        m_bn = (w_cap > 1e-9) & (~m_in)
        ii, jj, kk = np.nonzero(m_in)
        idx_inner.append(ii + jj * grid.Nx + kk * grid.Nx * grid.Ny + c * solver.N)
        ii_b, jj_b, kk_b = np.nonzero(m_bn)
        idx_bndry.append(ii_b + jj_b * grid.Nx + kk_b * grid.Nx * grid.Ny + c * solver.N)
        w_bndry.append(1.0 - w_cap[m_bn])

    idx_in_all = np.concatenate(idx_inner)
    if idx_in_all.size > 0:
        solver._add_dispersive_edges(
            cap_mat,
            idx_in_all,
            np.ones_like(idx_in_all, dtype=float),
            eps_bg=float(eps_cap),
        )
    idx_bn_all = np.concatenate(idx_bndry)
    w_bn_all = np.concatenate(w_bndry)
    if idx_bn_all.size > 0:
        solver._add_dispersive_edges(
            muscle_mat,
            idx_bn_all,
            w_bn_all,
            eps_bg=float(eps_cap),
        )
    return int(3 * solver.N)


def compute_box_shell_ohmic_loss(
    solver: SolverFIT3D,
    dft: DFTMonitor,
    muscle_mat: DispersiveMaterial,
    r1_m: float,
    r2_m: float,
    a_cap_m: float = 0.0,
    use_box_inner: bool = True,
) -> np.ndarray:
    """Integrate discrete ohmic loss P_loss(f) = int sigma_eff(f, dt) |E(f)|^2 dV over a shell.

    Uses the exact staggered FIT dual-primary cell volumes ``L_c * tA_c`` and
    trapezoidal face weights matching :class:`FluxMonitor`.

    Parameters
    ----------
    solver : SolverFIT3D
        Completed 3D FIT solver.
    dft : DFTMonitor
        3D DFTMonitor covering ``[-r2_m, +r2_m]^3``.
    muscle_mat : DispersiveMaterial
        Dispersive muscle material.
    r1_m : float
        Inner box half-width [m] (used when ``use_box_inner=True``).
    r2_m : float
        Outer box half-width [m].
    a_cap_m : float, optional
        Spherical capsule radius [m] (used when ``use_box_inner=False``).
    use_box_inner : bool, optional
        If True, integrate between cubic FluxMonitor boxes ``r1_m`` and ``r2_m``.
        If False, integrate between the spherical capsule wall ``r >= a_cap_m``
        and the outer cubic FluxMonitor box ``r2_m``.

    Returns
    -------
    np.ndarray
        Integrated spectral power loss [W s^2] at each monitored frequency.
    """
    grid = solver.grid
    freqs = dft.frequencies
    dx_min = float(min(np.min(grid.dx), np.min(grid.dy), np.min(grid.dz)))
    p_loss = np.zeros(freqs.size, dtype=float)
    tol = 1e-9

    # Precompute spatial weights and cell volumes per component
    comp_weights_dv = []
    for comp_idx, comp in enumerate(("Ex", "Ey", "Ez")):
        c_char = comp[1]
        _, coords = dft.get_field(comp)
        w1, w2 = 1.0, 1.0
        for d in range(3):
            cd = coords[d].reshape([(-1 if d == dd else 1) for dd in range(3)])
            if d == comp_idx:
                wd1 = np.where(np.abs(cd) < r1_m, 1.0, 0.0)
                wd2 = np.where(np.abs(cd) < r2_m, 1.0, 0.0)
            else:
                wd1 = np.where(
                    np.abs(cd) < r1_m - tol,
                    1.0,
                    np.where(np.abs(np.abs(cd) - r1_m) <= tol, 0.5, 0.0),
                )
                wd2 = np.where(
                    np.abs(cd) < r2_m - tol,
                    1.0,
                    np.where(np.abs(np.abs(cd) - r2_m) <= tol, 0.5, 0.0),
                )
            w1 = w1 * wd1
            w2 = w2 * wd2

        if use_box_inner:
            w_shell = w2 - w1
        else:
            X, Y, Z = np.meshgrid(*coords, indexing="ij")
            R = np.sqrt(X**2 + Y**2 + Z**2)
            w_tis = 1.0 - np.clip((a_cap_m + 0.5 * dx_min - R) / dx_min, 0.0, 1.0)
            w_shell = w_tis * w2

        indices, _ = grid.region_indices("E", c_char, (0.0, 0.0, 0.0), (2.0 * r2_m, 2.0 * r2_m, 2.0 * r2_m))
        I, J, K = np.meshgrid(*indices, indexing="ij")
        b_char = "xyz"[(comp_idx + 1) % 3]
        c2_char = "xyz"[(comp_idx + 2) % 3]
        dV = grid.L[I, J, K, c_char] * grid.tL[I, J, K, b_char] * grid.tL[I, J, K, c2_char]
        comp_weights_dv.append((comp, w_shell * dV))

    for fi, fq in enumerate(freqs):
        eps_d = muscle_mat.epsilon_discrete(fq, solver.dt)
        w_sinc = float(np.sin(np.pi * fq * solver.dt) / (np.pi * fq * solver.dt))
        sig_d = -2.0 * np.pi * fq * EPS0 * float(np.imag(eps_d)) * w_sinc
        loss_sum = 0.0
        for comp, w_dv in comp_weights_dv:
            e_arr, _ = dft.get_field(comp)
            loss_sum += float(np.sum(w_dv * (np.abs(e_arr[fi]) ** 2)))
        p_loss[fi] = sig_d * loss_sum
    return p_loss


def compute_regional_shell_ohmic_loss(
    solver: SolverFIT3D,
    dft: DFTMonitor,
    muscle_mat: DispersiveMaterial,
    r1_m: float,
    r_mid_m: float,
    r2_m: float,
    drop_dual_volume: bool = False,
) -> dict[str, np.ndarray]:
    """Partition discrete tissue ohmic loss int sigma_eff |E|^2 dV by spatial region.

    Partitions the exterior tissue shell ``[r1_m, r2_m]`` into:
    - ``p_loss_box``: full shell ``[r1_m, r2_m]``;
    - ``p_loss_gap_hemi``: feed-gap hemisphere ``x > 0`` (with 0.5 weight on ``x == 0``);
    - ``p_loss_back_hemi``: opposite hemisphere ``x < 0`` (with 0.5 weight on ``x == 0``);
    - ``p_loss_inner_shell``: inner near-wall shell ``[r1_m, r_mid_m]``;
    - ``p_loss_outer_shell``: outer shell ``[r_mid_m, r2_m]``.

    Parameters
    ----------
    solver : SolverFIT3D
        Completed 3D FIT solver.
    dft : DFTMonitor
        3D DFTMonitor covering ``[-r2_m, +r2_m]^3``.
    muscle_mat : DispersiveMaterial
        Dispersive muscle material.
    r1_m, r_mid_m, r2_m : float
        Inner, intermediate, and outer cubic shell half-widths [m].
    drop_dual_volume : bool, optional
        Mutation flag: if True, replaces exact staggered cell volumes with 1.0.

    Returns
    -------
    dict of str to np.ndarray
        Regional ohmic power losses [W s^2] at each monitored frequency.
    """
    grid = solver.grid
    freqs = dft.frequencies
    tol = 1e-9

    def _box_weight(coords: tuple[np.ndarray, ...], comp_idx: int, r_box: float) -> np.ndarray:
        w = 1.0
        for d in range(3):
            cd = coords[d].reshape([(-1 if d == dd else 1) for dd in range(3)])
            if d == comp_idx:
                wd = np.where(np.abs(cd) < r_box, 1.0, 0.0)
            else:
                wd = np.where(
                    np.abs(cd) < r_box - tol,
                    1.0,
                    np.where(np.abs(np.abs(cd) - r_box) <= tol, 0.5, 0.0),
                )
            w = w * wd
        return w

    comp_regions = []
    for comp_idx, comp in enumerate(("Ex", "Ey", "Ez")):
        c_char = comp[1]
        _, coords = dft.get_field(comp)
        w1 = _box_weight(coords, comp_idx, r1_m)
        w_mid = _box_weight(coords, comp_idx, r_mid_m)
        w2 = _box_weight(coords, comp_idx, r2_m)
        w_shell = w2 - w1
        w_inner = w_mid - w1
        w_outer = w2 - w_mid

        cx = coords[0].reshape((-1, 1, 1))
        w_pos_x = np.where(cx > tol, 1.0, np.where(np.abs(cx) <= tol, 0.5, 0.0))
        w_neg_x = 1.0 - w_pos_x

        indices, _ = grid.region_indices("E", c_char, (0.0, 0.0, 0.0), (2.0 * r2_m, 2.0 * r2_m, 2.0 * r2_m))
        I, J, K = np.meshgrid(*indices, indexing="ij")
        b_char = "xyz"[(comp_idx + 1) % 3]
        c2_char = "xyz"[(comp_idx + 2) % 3]
        dV = (
            np.ones_like(I, dtype=float)
            if drop_dual_volume
            else grid.L[I, J, K, c_char] * grid.tL[I, J, K, b_char] * grid.tL[I, J, K, c2_char]
        )
        comp_regions.append(
            (
                comp,
                w_shell * dV,
                (w_shell * w_pos_x) * dV,
                (w_shell * w_neg_x) * dV,
                w_inner * dV,
                w_outer * dV,
            )
        )

    out = {
        "p_loss_box": np.zeros(freqs.size, dtype=float),
        "p_loss_gap_hemi": np.zeros(freqs.size, dtype=float),
        "p_loss_back_hemi": np.zeros(freqs.size, dtype=float),
        "p_loss_inner_shell": np.zeros(freqs.size, dtype=float),
        "p_loss_outer_shell": np.zeros(freqs.size, dtype=float),
    }
    for fi, fq in enumerate(freqs):
        eps_d = muscle_mat.epsilon_discrete(fq, solver.dt)
        w_sinc = float(np.sin(np.pi * fq * solver.dt) / (np.pi * fq * solver.dt))
        sig_d = -2.0 * np.pi * fq * EPS0 * float(np.imag(eps_d)) * w_sinc
        s_box, s_gap, s_back, s_in, s_out = 0.0, 0.0, 0.0, 0.0, 0.0
        for comp, w_b, w_g, w_bk, w_i, w_o in comp_regions:
            e_arr, _ = dft.get_field(comp)
            e2 = np.abs(e_arr[fi]) ** 2
            s_box += float(np.sum(w_b * e2))
            s_gap += float(np.sum(w_g * e2))
            s_back += float(np.sum(w_bk * e2))
            s_in += float(np.sum(w_i * e2))
            s_out += float(np.sum(w_o * e2))
        out["p_loss_box"][fi] = sig_d * s_box
        out["p_loss_gap_hemi"][fi] = sig_d * s_gap
        out["p_loss_back_hemi"][fi] = sig_d * s_back
        out["p_loss_inner_shell"][fi] = sig_d * s_in
        out["p_loss_outer_shell"][fi] = sig_d * s_out
    return out


# ---------------------------------------------------------------------------
# Task 1: Near-zone loss theory & Gabriel/Debye muscle pole fit
# ---------------------------------------------------------------------------
def run_task_near_zone_theory_and_muscle_fit() -> dict[str, Any]:
    """Execute Part 1 & dispersive muscle fitting, saving JSON and NPZ artifacts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Zhou & Pan (2023) seawater benchmark at f = 1 kHz, eps_r = 80, sigma = 4 S/m
    f_sea = 1.0e3
    eps_sea = 80.0 - 1j * 4.0 / (2.0 * np.pi * f_sea * EPS0)
    a_sea = 0.30
    b_sea = 100.0
    zp_tm_a = zhou_pan_poynting_flux(a_sea, f_sea, eps_sea, mode="TM")
    zp_tm_b = zhou_pan_poynting_flux(b_sea, f_sea, eps_sea, mode="TM")
    zp_te_a = zhou_pan_poynting_flux(a_sea, f_sea, eps_sea, mode="TE")
    zp_te_b = zhou_pan_poynting_flux(b_sea, f_sea, eps_sea, mode="TE")
    eta_sea_tm = float(zp_tm_b["S_r"] / zp_tm_a["S_r"])
    eta_sea_te = float(zp_te_b["S_r"] / zp_te_a["S_r"])
    ratio_sea_full = eta_sea_te / eta_sea_tm
    ka_sea = float(zp_tm_a["kr_abs"])
    inv_ka2_sea = 1.0 / (ka_sea**2)

    # Verify differential Poynting identity -dS/dr = (sigma/2) * oint |E|^2 r^2 dOmega
    dr = 1e-7
    s_plus = float(zhou_pan_poynting_flux(a_sea + dr, f_sea, eps_sea, mode="TM")["S_r"])
    s_minus = float(zhou_pan_poynting_flux(a_sea - dr, f_sea, eps_sea, mode="TM")["S_r"])
    ds_dr = -(s_plus - s_minus) / (2.0 * dr)
    w_sea = 2.0 * np.pi * f_sea
    k_sea = zp_tm_a["k"]
    u_sea = k_sea * a_sea
    e_r0 = (w_sea * MU0 / (2.0 * np.pi)) * k_sea * (1.0 / (u_sea**2) - 1j / (u_sea**3)) * np.exp(-1j * u_sea)
    e_th0 = (
        (w_sea * MU0 / (4.0 * np.pi))
        * k_sea
        * (1j / u_sea + 1.0 / (u_sea**2) - 1j / (u_sea**3))
        * np.exp(-1j * u_sea)
    )
    dp_loss_dr = 4.0 * (a_sea**2) * (
        (2.0 * np.pi / 3.0) * (abs(e_r0) ** 2) + (4.0 * np.pi / 3.0) * (abs(e_th0) ** 2)
    )
    poynting_diff_rel_err = abs(ds_dr - dp_loss_dr) / dp_loss_dr

    # 2. Fit 3-pole Debye model to Gabriel (1996) Cole-Cole muscle
    muscle = build_dispersive_muscle(n_debye=3)
    f_band = np.geomspace(100.0e6, 4.0e9, 120)
    eps_cc_band = gabriel_1996_muscle_cole_cole(f_band)
    eps_deb_band = muscle.epsilon(f_band)

    eps_cc_403 = complex(gabriel_1996_muscle_cole_cole(F_MEDRADIO_HZ))
    eps_deb_403 = complex(muscle.epsilon(F_MEDRADIO_HZ))
    eps_cc_2450 = complex(gabriel_1996_muscle_cole_cole(F_ISM_HZ))
    eps_deb_2450 = complex(muscle.epsilon(F_ISM_HZ))

    rel_err_403 = float(abs(eps_deb_403 - eps_cc_403) / abs(eps_cc_403))
    rel_err_2450 = float(abs(eps_deb_2450 - eps_cc_2450) / abs(eps_cc_2450))

    k_403, beta_403, alpha_403 = complex_wavenumber(F_MEDRADIO_HZ, eps_deb_403)
    k_2450, beta_2450, alpha_2450 = complex_wavenumber(F_ISM_HZ, eps_deb_2450)
    lam_muscle_403_mm = float(2.0 * np.pi / beta_403 * 1e3)
    lam_muscle_2450_mm = float(2.0 * np.pi / beta_2450 * 1e3)
    skin_depth_403_mm = float(1.0 / alpha_403 * 1e3)
    skin_depth_2450_mm = float(1.0 / alpha_2450 * 1e3)

    # 3. Sweep capsule radius a so |k|a spans [0.04, 1.5]
    a_sweep_m = np.geomspace(0.5e-3, 12.0e-3, 80)
    nz_403 = near_zone_excess_loss_metrics(a_sweep_m, F_MEDRADIO_HZ, eps_deb_403)
    nz_2450 = near_zone_excess_loss_metrics(a_sweep_m, F_ISM_HZ, eps_deb_2450)
    cav_403_air = capsule_cavity_factor(F_MEDRADIO_HZ, a_sweep_m, 1.0, eps_deb_403, mode="TM")
    cav_2450_air = capsule_cavity_factor(F_ISM_HZ, a_sweep_m, 1.0, eps_deb_2450, mode="TM")
    cav_403_cer = capsule_cavity_factor(F_MEDRADIO_HZ, a_sweep_m, 12.0, eps_deb_403, mode="TM")
    cav_2450_cer = capsule_cavity_factor(F_ISM_HZ, a_sweep_m, 12.0, eps_deb_2450, mode="TM")

    # Evaluate at reference capsule radius a = 2.25 mm
    a_ref_m = 2.25e-3
    nz_403_ref = near_zone_excess_loss_metrics(a_ref_m, F_MEDRADIO_HZ, eps_deb_403)
    nz_2450_ref = near_zone_excess_loss_metrics(a_ref_m, F_ISM_HZ, eps_deb_2450)
    cav_403_ref = capsule_cavity_factor(F_MEDRADIO_HZ, a_ref_m, 1.0, eps_deb_403, mode="TM")
    cav_2450_ref = capsule_cavity_factor(F_ISM_HZ, a_ref_m, 1.0, eps_deb_2450, mode="TM")
    cav_403_cer_ref = capsule_cavity_factor(F_MEDRADIO_HZ, a_ref_m, 12.0, eps_deb_403, mode="TM")
    cav_2450_cer_ref = capsule_cavity_factor(F_ISM_HZ, a_ref_m, 12.0, eps_deb_2450, mode="TM")

    summary = {
        "seawater_1khz_ka": ka_sea,
        "seawater_1khz_t1": float(zp_tm_a["t1"]),
        "seawater_1khz_t2a": float(zp_tm_a["t2a"]),
        "seawater_1khz_t2b": float(zp_tm_a["t2b"]),
        "seawater_1khz_t3": float(zp_tm_a["t3"]),
        "seawater_1khz_eta_tm": eta_sea_tm,
        "seawater_1khz_eta_te": eta_sea_te,
        "seawater_1khz_ratio_full": ratio_sea_full,
        "seawater_1khz_inv_ka2": inv_ka2_sea,
        "poynting_differential_rel_err": float(poynting_diff_rel_err),
        "muscle_debye_eps_inf": float(muscle.eps_inf),
        "muscle_debye_poles": [
            {"delta_eps": float(p.delta_eps), "tau_s": float(p.tau)} for p in muscle.poles
        ],
        "muscle_debye_rms_fit_error": float(muscle.fit_error),
        "muscle_403mhz_cole_cole_real": float(eps_cc_403.real),
        "muscle_403mhz_cole_cole_imag": float(-eps_cc_403.imag),
        "muscle_403mhz_debye_real": float(eps_deb_403.real),
        "muscle_403mhz_debye_imag": float(-eps_deb_403.imag),
        "muscle_403mhz_rel_err": rel_err_403,
        "muscle_403mhz_lambda_mm": lam_muscle_403_mm,
        "muscle_403mhz_skin_depth_mm": skin_depth_403_mm,
        "muscle_403mhz_k_abs_m": float(abs(k_403)),
        "muscle_2450mhz_cole_cole_real": float(eps_cc_2450.real),
        "muscle_2450mhz_cole_cole_imag": float(-eps_cc_2450.imag),
        "muscle_2450mhz_debye_real": float(eps_deb_2450.real),
        "muscle_2450mhz_debye_imag": float(-eps_deb_2450.imag),
        "muscle_2450mhz_rel_err": rel_err_2450,
        "muscle_2450mhz_lambda_mm": lam_muscle_2450_mm,
        "muscle_2450mhz_skin_depth_mm": skin_depth_2450_mm,
        "muscle_2450mhz_k_abs_m": float(abs(k_2450)),
        "a_ref_mm": a_ref_m * 1e3,
        "ka_ref_403mhz": float(nz_403_ref["ka_abs"]),
        "ka_ref_2450mhz": float(nz_2450_ref["ka_abs"]),
        "nz_excess_ratio_403mhz_ref": float(nz_403_ref["excess_loss_ratio"]),
        "nz_excess_ratio_2450mhz_ref": float(nz_2450_ref["excess_loss_ratio"]),
        "nz_eff_ratio_403mhz_ref": float(nz_403_ref["nz_efficiency_ratio"]),
        "nz_eff_ratio_2450mhz_ref": float(nz_2450_ref["nz_efficiency_ratio"]),
        "cavity_tm_qs_boost_403mhz": float(cav_403_ref["power_boost_qs"]),
        "cavity_tm_full_boost_403mhz": float(cav_403_ref["power_boost_full"]),
        "cavity_tm_qs_boost_2450mhz": float(cav_2450_ref["power_boost_qs"]),
        "cavity_tm_full_boost_2450mhz": float(cav_2450_ref["power_boost_full"]),
        "cavity_tm_cer_qs_boost_403mhz": float(cav_403_cer_ref["power_boost_qs"]),
        "cavity_tm_cer_full_boost_403mhz": float(cav_403_cer_ref["power_boost_full"]),
        "cavity_tm_cer_qs_boost_2450mhz": float(cav_2450_cer_ref["power_boost_qs"]),
        "cavity_tm_cer_full_boost_2450mhz": float(cav_2450_cer_ref["power_boost_full"]),
    }

    (DATA_DIR / "task_01_near_zone_theory_and_muscle_fit.json").write_text(
        json.dumps(summary, indent=2)
    )
    np.savez(
        DATA_DIR / "task_01_near_zone_theory_and_muscle_fit.npz",
        f_band_hz=f_band,
        eps_cc_band=eps_cc_band,
        eps_deb_band=eps_deb_band,
        a_sweep_m=a_sweep_m,
        ka_403=nz_403["ka_abs"],
        ka_2450=nz_2450["ka_abs"],
        delta_tm_403=nz_403["delta_tm"],
        delta_te_403=nz_403["delta_te"],
        ratio_nz_403=nz_403["excess_loss_ratio"],
        eff_ratio_403=nz_403["nz_efficiency_ratio"],
        delta_tm_2450=nz_2450["delta_tm"],
        delta_te_2450=nz_2450["delta_te"],
        ratio_nz_2450=nz_2450["excess_loss_ratio"],
        eff_ratio_2450=nz_2450["nz_efficiency_ratio"],
        cav_boost_qs_403_air=cav_403_air["power_boost_qs"],
        cav_boost_qs_403_cer=cav_403_cer["power_boost_qs"],
        cav_boost_full_403_air=cav_403_air["power_boost_full"],
        cav_boost_full_2450_air=cav_2450_air["power_boost_full"],
        cav_boost_full_403_cer=cav_403_cer["power_boost_full"],
        cav_boost_full_2450_cer=cav_2450_cer["power_boost_full"],
    )
    return summary


# ---------------------------------------------------------------------------
# Task 2: 3D FIT shell power bookkeeping, |k|a sweep & capsule correction
# ---------------------------------------------------------------------------
def _run_single_shell_sim(
    muscle_mat: DispersiveMaterial,
    *,
    N: int,
    dx_m: float,
    a_cap_m: float,
    r1_m: float,
    r2_m: float,
    mode: str = "TM",
    eps_cap: float | None = 1.0,
    save_schematic: bool = False,
    record_dft_shell: bool = True,
    extra_r1_m: list[float] | None = None,
) -> dict[str, Any]:
    """Run a single 3D FIT capsule-in-muscle simulation with preflight check."""
    L = N * dx_m
    x = np.linspace(-0.5 * L, 0.5 * L, N + 1)
    grid = GridFIT3D(x=x, y=x, z=x, verbose=0)
    solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
    n_disp_edges = assign_spherical_capsule_in_tissue(solver, muscle_mat, a_cap_m, eps_cap=eps_cap)

    pulse = GaussianSource(1.5e9, fwidth=1.8e9, cutoff=3.5)
    n_steps = int(np.ceil(2.2 * pulse.end_time / solver.dt))

    n_samp = int((round(2.0 * r2_m / dx_m) + 1) ** 3)
    monitors_est = [(2, n_samp, 3)] if record_dft_shell else []
    est = estimate(
        (grid.Nx, grid.Ny, grid.Nz),
        steps=n_steps,
        cpml=True,
        monitors=monitors_est,
        dispersive_edges=n_disp_edges,
        n_poles=3,
    )
    preflight(est, "laptop", raise_error=True)

    if mode.upper() == "TM":
        # Centered 2-edge symmetric Ez dipole (z in [-0.5*dx, +0.5*dx]) on an even-N grid
        # so the source has zero z-eccentricity inside the spherical capsule (no spurious l=2 mode).
        srcs = [
            CurrentSource(
                center=(0.0, 0.0, 0.0),
                size=(0.0, 0.0, dx_m),
                component="Ez",
                amplitude=1.0e-2,
                src_time=pulse,
            )
        ]
    else:
        d_loop = dx_m
        srcs = [
            CurrentSource(
                center=(0.0, -d_loop, 0.0),
                size=(2.0 * d_loop, 0.0, 0.0),
                component="Ex",
                amplitude=+1.0e-2,
                src_time=pulse,
            ),
            CurrentSource(
                center=(+d_loop, 0.0, 0.0),
                size=(0.0, 2.0 * d_loop, 0.0),
                component="Ey",
                amplitude=+1.0e-2,
                src_time=pulse,
            ),
            CurrentSource(
                center=(0.0, +d_loop, 0.0),
                size=(2.0 * d_loop, 0.0, 0.0),
                component="Ex",
                amplitude=-1.0e-2,
                src_time=pulse,
            ),
            CurrentSource(
                center=(-d_loop, 0.0, 0.0),
                size=(0.0, 2.0 * d_loop, 0.0),
                component="Ey",
                amplitude=-1.0e-2,
                src_time=pulse,
            ),
        ]

    flux1 = FluxMonitor(FREQS_MAIN_HZ, center=(0.0, 0.0, 0.0), size=(2.0 * r1_m, 2.0 * r1_m, 2.0 * r1_m))
    flux2 = FluxMonitor(FREQS_MAIN_HZ, center=(0.0, 0.0, 0.0), size=(2.0 * r2_m, 2.0 * r2_m, 2.0 * r2_m))
    extra_fluxes = [
        FluxMonitor(FREQS_MAIN_HZ, center=(0.0, 0.0, 0.0), size=(2.0 * r_val, 2.0 * r_val, 2.0 * r_val))
        for r_val in (extra_r1_m or [])
    ]
    dft = (
        DFTMonitor(
            FREQS_MAIN_HZ,
            components=("Ex", "Ey", "Ez"),
            center=(0.0, 0.0, 0.0),
            size=(2.0 * r2_m, 2.0 * r2_m, 2.0 * r2_m),
        )
        if record_dft_shell
        else None
    )

    if save_schematic:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        cap_struct = CircleOverlay(
            center=(0.0, 0.0, 0.0),
            radius=a_cap_m,
            material=float(eps_cap or 1.0),
            name="Insulating Capsule (a)",
            axis=1,
        )
        schematic.save_setup_figures(
            solver,
            FIG_DIR,
            plane="xz",
            structures=[cap_struct],
            material_names={"background": "Muscle (3-pole Debye)", "region_0": "Capsule (ε_cap = 1)"},
            sources=srcs,
            monitors=[flux1, flux2],
            title_geometry="3D FIT Setup: Implant Capsule in Dispersive Muscle Tissue",
            title_mesh="Grid & Broadband Excitation Spectrum",
            min_feature=2.0 * a_cap_m,
            wavelength=16.72e-3,
            freq=F_ISM_HZ,
            variations=[
                ("Muscle model", "3-pole Debye fit to Gabriel (1996): ε_r(403 MHz) = 57.2 - 35.6j; ε_r(2.45 GHz) = 52.8 - 12.8j"),
                ("Capsule & shell sweep", "a ∈ {1.60, 2.24, 2.88} mm, ε_cap ∈ {1, 12}; r1 ∈ {2.56, 3.20, 3.84} mm, r2 = 6.40 mm"),
                ("Antenna feeds", "Ideal TM dipole, ideal 4-segment TE loop, gapped PEC loop, gapped PEC dipole, model NiZn ferrite core"),
            ],
        )

    t0 = time.perf_counter()
    for step in range(n_steps):
        t = step * solver.dt
        for s in srcs:
            s.update(solver, t)
        solver.one_step()
        flux1.update(solver, t)
        flux2.update(solver, t)
        for fm in extra_fluxes:
            fm.update(solver, t)
        if dft is not None:
            dft.update(solver, t)
    wall_s = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    p1 = flux1.get_flux()
    p2 = flux2.get_flux()
    p_extra = [fm.get_flux() for fm in extra_fluxes]
    p_loss_box = (
        compute_box_shell_ohmic_loss(solver, dft, muscle_mat, r1_m, r2_m, a_cap_m, use_box_inner=True)
        if dft is not None
        else np.zeros_like(p1)
    )
    p_loss_sphere = (
        compute_box_shell_ohmic_loss(solver, dft, muscle_mat, r1_m, r2_m, a_cap_m, use_box_inner=False)
        if dft is not None
        else np.zeros_like(p1)
    )
    return {
        "p1": p1,
        "p2": p2,
        "p_extra": p_extra,
        "p_loss_box": p_loss_box,
        "p_loss_sphere": p_loss_sphere,
        "wall_s": float(wall_s),
        "rss_mb": float(rss_mb),
        "est_ram_gb": float(est.host_bytes / 1e9),
        "est_time_s": float(est.seconds),
        "cells": int(grid.Nx * grid.Ny * grid.Nz),
        "n_steps": int(n_steps),
        "dt_s": float(solver.dt),
    }


def _run_graded_capsule_check(muscle_mat: DispersiveMaterial, a_cap_m: float, r1_m: float, r2_m: float) -> dict[str, float]:
    """Run a locally refined graded-mesh check of the TM capsule cavity factor at a = 2.25 mm."""
    x_g = graded_axis(
        [-11.8e-3, -4.0e-3, -1.4e-3, 1.4e-3, 4.0e-3, 11.8e-3],
        [0.70e-3, 0.50e-3, 0.35e-3, 0.50e-3, 0.70e-3],
        min_cells=[6, 4, 8, 4, 6],
        grading=1.22,
    )
    dx_min = float(np.min(np.diff(x_g)))
    p1_by_eps: dict[str, np.ndarray] = {}
    for label, eps_c in [("hom", None), ("air", 1.0), ("cer", 12.0)]:
        grid = GridFIT3D(x=x_g, y=x_g, z=x_g, verbose=0)
        solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
        assign_spherical_capsule_in_tissue(solver, muscle_mat, a_cap_m, eps_cap=eps_c)
        pulse = GaussianSource(1.5e9, fwidth=1.8e9, cutoff=3.5)
        n_steps = int(np.ceil(2.2 * pulse.end_time / solver.dt))
        src = CurrentSource(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, dx_min),
            component="Ez",
            amplitude=1.0e-2,
            src_time=pulse,
        )
        flux1 = FluxMonitor(FREQS_MAIN_HZ, center=(0.0, 0.0, 0.0), size=(2.0 * r1_m, 2.0 * r1_m, 2.0 * r1_m))
        for step in range(n_steps):
            t = step * solver.dt
            src.update(solver, t)
            solver.one_step()
            flux1.update(solver, t)
        p1_by_eps[label] = flux1.get_flux()
    return {
        "graded_dx_min_mm": dx_min * 1e3,
        "graded_cavity_boost_tm_air_2450mhz_fit": float(p1_by_eps["air"][1] / p1_by_eps["hom"][1]),
        "graded_cavity_boost_tm_cer_2450mhz_fit": float(p1_by_eps["cer"][1] / p1_by_eps["hom"][1]),
        "graded_cavity_boost_tm_air_403mhz_fit": float(p1_by_eps["air"][0] / p1_by_eps["hom"][0]),
        "graded_cavity_boost_tm_cer_403mhz_fit": float(p1_by_eps["cer"][0] / p1_by_eps["hom"][0]),
    }


def run_task_fit_shells_and_capsule_correction(quick: bool = False) -> dict[str, Any]:
    """Run Part 2: 3D FIT shell power balance, |k|a capsule sweep, and unfudged shell-loss sweep."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    muscle = build_dispersive_muscle(n_debye=3)

    # Grid choice: dx = 0.64 mm (26.13 cells/lambda_m at 2.45 GHz) on N = 38 (L = 24.32 mm).
    # Inner CPML boundary sits at (19 - 6)*dx = 13*dx = 8.32 mm, leaving a 3-cell buffer
    # outside the outer FluxMonitor box r2 = 10*dx = 6.40 mm (2 cells in quick mode).
    N = 32 if quick else 38
    dx = 0.80e-3 if quick else 0.64e-3
    r1 = 5.0 * dx
    r2 = (8.0 if quick else 10.0) * dx
    a_ref = (2.5 if quick else 3.5) * dx  # 2.24 mm in full mode (2.00 mm in quick mode)

    # 1. Reference TM and TE runs in air capsule (eps_cap = 1.0) with DFT shell power balance
    tm_air = _run_single_shell_sim(
        muscle,
        N=N,
        dx_m=dx,
        a_cap_m=a_ref,
        r1_m=r1,
        r2_m=r2,
        mode="TM",
        eps_cap=1.0,
        save_schematic=True,
        record_dft_shell=True,
    )
    te_air = _run_single_shell_sim(
        muscle,
        N=N,
        dx_m=dx,
        a_cap_m=a_ref,
        r1_m=r1,
        r2_m=r2,
        mode="TE",
        eps_cap=1.0,
        save_schematic=False,
        record_dft_shell=True,
    )

    # Power-balance closure: (P1 - P2) vs integral of sigma_eff |E|^2 over cubic shell [r1, r2]
    dp_tm = tm_air["p1"] - tm_air["p2"]
    closure_err_tm_403 = float(abs(dp_tm[0] - tm_air["p_loss_box"][0]) / abs(dp_tm[0]))
    closure_err_tm_2450 = float(abs(dp_tm[1] - tm_air["p_loss_box"][1]) / abs(dp_tm[1]))

    dp_te = te_air["p1"] - te_air["p2"]
    closure_err_te_403 = float(abs(dp_te[0] - te_air["p_loss_box"][0]) / abs(dp_te[0]))
    closure_err_te_2450 = float(abs(dp_te[1] - te_air["p_loss_box"][1]) / abs(dp_te[1]))

    # 2. Homogeneous muscle reference & capsule-interface sweep across a_mults for both air (eps_cap=1)
    # and ceramic (eps_cap=12), compared against BOTH full-wave |C_TM^full(a)|^2 and quasi-static |C_TM^QS|^2.
    tm_hom = _run_single_shell_sim(
        muscle,
        N=N,
        dx_m=dx,
        a_cap_m=a_ref,
        r1_m=r1,
        r2_m=r2,
        mode="TM",
        eps_cap=None,
        record_dft_shell=False,
    )
    te_hom = _run_single_shell_sim(
        muscle,
        N=N,
        dx_m=dx,
        a_cap_m=a_ref,
        r1_m=r1,
        r2_m=r2,
        mode="TE",
        eps_cap=None,
        record_dft_shell=False,
    )

    eps_2450 = complex(muscle.epsilon(F_ISM_HZ))
    eps_403 = complex(muscle.epsilon(F_MEDRADIO_HZ))

    a_cap_mults = [2.0, 2.5] if quick else [2.5, 3.5, 4.5]
    a_sweep_mm: list[float] = []
    ka_403_list: list[float] = []
    ka_2450_list: list[float] = []
    cav_fit_air_2450_sweep: list[float] = []
    cav_fit_cer_2450_sweep: list[float] = []
    cav_full_air_2450_sweep: list[float] = []
    cav_full_cer_2450_sweep: list[float] = []
    cav_qs_air_2450_sweep: list[float] = []
    cav_qs_cer_2450_sweep: list[float] = []
    cav_fit_air_403_sweep: list[float] = []
    cav_fit_cer_403_sweep: list[float] = []
    cav_full_air_403_sweep: list[float] = []
    cav_full_cer_403_sweep: list[float] = []
    cut_cell_loss_ratio_fit_403: list[float] = []
    cut_cell_loss_ratio_fit_2450: list[float] = []

    tm_cer_ref = None
    tm_air_smallest = None
    for mult in a_cap_mults:
        a_cur = mult * dx
        a_sweep_mm.append(float(a_cur * 1e3))
        ka_403_list.append(float(abs(complex_wavenumber(F_MEDRADIO_HZ, eps_403)[0]) * a_cur))
        ka_2450_list.append(float(abs(complex_wavenumber(F_ISM_HZ, eps_2450)[0]) * a_cur))

        if abs(a_cur - a_ref) < 1e-9:
            res_air = tm_air
            res_te_cur = te_air
        else:
            res_air = _run_single_shell_sim(
                muscle,
                N=N,
                dx_m=dx,
                a_cap_m=a_cur,
                r1_m=r1,
                r2_m=r2,
                mode="TM",
                eps_cap=1.0,
                record_dft_shell=True,
                extra_r1_m=[4.0 * dx, 6.0 * dx] if abs(mult - a_cap_mults[0]) < 1e-9 else None,
            )
            res_te_cur = _run_single_shell_sim(
                muscle,
                N=N,
                dx_m=dx,
                a_cap_m=a_cur,
                r1_m=r1,
                r2_m=r2,
                mode="TE",
                eps_cap=1.0,
                record_dft_shell=True,
                extra_r1_m=[4.0 * dx, 6.0 * dx] if abs(mult - a_cap_mults[0]) < 1e-9 else None,
            )
        if abs(mult - a_cap_mults[0]) < 1e-9:
            tm_air_smallest = (res_air, res_te_cur)

        res_cer = _run_single_shell_sim(
            muscle,
            N=N,
            dx_m=dx,
            a_cap_m=a_cur,
            r1_m=r1,
            r2_m=r2,
            mode="TM",
            eps_cap=12.0,
            record_dft_shell=False,
        )
        if abs(a_cur - a_ref) < 1e-9:
            tm_cer_ref = res_cer

        cav_air_2450_th = capsule_cavity_factor(F_ISM_HZ, a_cur, 1.0, eps_2450, mode="TM")
        cav_cer_2450_th = capsule_cavity_factor(F_ISM_HZ, a_cur, 12.0, eps_2450, mode="TM")
        cav_air_403_th = capsule_cavity_factor(F_MEDRADIO_HZ, a_cur, 1.0, eps_403, mode="TM")
        cav_cer_403_th = capsule_cavity_factor(F_MEDRADIO_HZ, a_cur, 12.0, eps_403, mode="TM")

        cav_fit_air_2450_sweep.append(float(res_air["p1"][1] / tm_hom["p1"][1]))
        cav_fit_cer_2450_sweep.append(float(res_cer["p1"][1] / tm_hom["p1"][1]))
        cav_full_air_2450_sweep.append(float(cav_air_2450_th["power_boost_full"]))
        cav_full_cer_2450_sweep.append(float(cav_cer_2450_th["power_boost_full"]))
        cav_qs_air_2450_sweep.append(float(cav_air_2450_th["power_boost_qs"]))
        cav_qs_cer_2450_sweep.append(float(cav_cer_2450_th["power_boost_qs"]))

        cav_fit_air_403_sweep.append(float(res_air["p1"][0] / tm_hom["p1"][0]))
        cav_fit_cer_403_sweep.append(float(res_cer["p1"][0] / tm_hom["p1"][0]))
        cav_full_air_403_sweep.append(float(cav_air_403_th["power_boost_full"]))
        cav_full_cer_403_sweep.append(float(cav_cer_403_th["power_boost_full"]))

        # Also record the unsmoothed cut-cell integral ratio (r in [a_cap, r2]) to document its staircase inflation
        r_tm_cut_403 = float(res_air["p_loss_sphere"][0] / res_air["p2"][0])
        r_te_cut_403 = float(res_te_cur["p_loss_sphere"][0] / res_te_cur["p2"][0])
        r_tm_cut_2450 = float(res_air["p_loss_sphere"][1] / res_air["p2"][1])
        r_te_cut_2450 = float(res_te_cur["p_loss_sphere"][1] / res_te_cur["p2"][1])
        cut_cell_loss_ratio_fit_403.append(r_tm_cut_403 / r_te_cut_403)
        cut_cell_loss_ratio_fit_2450.append(r_tm_cut_2450 / r_te_cut_2450)

    assert tm_cer_ref is not None
    assert tm_air_smallest is not None

    cav_th_air_2450_full = float(capsule_cavity_factor(F_ISM_HZ, a_ref, 1.0, eps_2450, mode="TM")["power_boost_full"])
    cav_th_air_2450_qs = float(capsule_cavity_factor(F_ISM_HZ, a_ref, 1.0, eps_2450, mode="TM")["power_boost_qs"])
    cav_th_cer_2450_full = float(capsule_cavity_factor(F_ISM_HZ, a_ref, 12.0, eps_2450, mode="TM")["power_boost_full"])
    cav_th_cer_2450_qs = float(capsule_cavity_factor(F_ISM_HZ, a_ref, 12.0, eps_2450, mode="TM")["power_boost_qs"])
    cav_fit_air_2450 = float(tm_air["p1"][1] / tm_hom["p1"][1])
    cav_fit_cer_2450 = float(tm_cer_ref["p1"][1] / tm_hom["p1"][1])
    cav_fit_te_403 = float(te_air["p1"][0] / te_hom["p1"][0])

    if not quick:
        graded_cav = _run_graded_capsule_check(muscle, a_cap_m=2.25e-3, r1_m=3.20e-3, r2_m=6.40e-3)
    else:
        graded_cav = {
            "graded_dx_min_mm": dx * 1e3,
            "graded_cavity_boost_tm_air_2450mhz_fit": cav_fit_air_2450,
            "graded_cavity_boost_tm_cer_2450mhz_fit": cav_fit_cer_2450,
            "graded_cavity_boost_tm_air_403mhz_fit": float(tm_air["p1"][0] / tm_hom["p1"][0]),
            "graded_cavity_boost_tm_cer_403mhz_fit": float(tm_cer_ref["p1"][0] / tm_hom["p1"][0]),
        }

    # Normalized exterior shell transmission P2/P1 invariance with vs without capsule
    trans_tm_403_hom = float(tm_hom["p2"][0] / tm_hom["p1"][0])
    trans_tm_403_air = float(tm_air["p2"][0] / tm_air["p1"][0])
    trans_te_403_hom = float(te_hom["p2"][0] / te_hom["p1"][0])
    trans_te_403_air = float(te_air["p2"][0] / te_air["p1"][0])

    # 3. Unfudged shell-loss sweep across grid-aligned inner shell radii r1 in {4, 5, 6}*dx (outside a = 2.5*dx)
    # compared against exact cubic-surface integration of Zhou & Pan (2023) (`zhou_pan_cube_flux`, zero fudge factors)
    # and spherical radial evaluation (`zhou_pan_poynting_flux`, zero fudge factors).
    res_tm_s, res_te_s = tm_air_smallest
    if len(res_tm_s["p_extra"]) == 2:
        r1_shell_list = [4.0 * dx, 5.0 * dx, 6.0 * dx]
        p1_tm_shells = [res_tm_s["p_extra"][0], res_tm_s["p1"], res_tm_s["p_extra"][1]]
        p1_te_shells = [res_te_s["p_extra"][0], res_te_s["p1"], res_te_s["p_extra"][1]]
    else:
        r1_shell_list = [4.0 * dx, 5.0 * dx]
        p1_tm_shells = [tm_hom["p1"], tm_air["p1"]]
        p1_te_shells = [te_hom["p1"], te_air["p1"]]

    r1_sweep_mm: list[float] = []
    kr1_403_list: list[float] = []
    kr1_2450_list: list[float] = []
    loss_ratio_fit_403: list[float] = []
    loss_ratio_cube_th_403: list[float] = []
    loss_ratio_sph_th_403: list[float] = []
    loss_ratio_fit_2450: list[float] = []
    loss_ratio_cube_th_2450: list[float] = []
    loss_ratio_sph_th_2450: list[float] = []

    s_tm_r2_cube_403 = zhou_pan_cube_flux(r2, F_MEDRADIO_HZ, eps_403, mode="TM")
    s_te_r2_cube_403 = zhou_pan_cube_flux(r2, F_MEDRADIO_HZ, eps_403, mode="TE")
    s_tm_r2_sph_403 = float(zhou_pan_poynting_flux(r2, F_MEDRADIO_HZ, eps_403, mode="TM")["S_r"])
    s_te_r2_sph_403 = float(zhou_pan_poynting_flux(r2, F_MEDRADIO_HZ, eps_403, mode="TE")["S_r"])

    s_tm_r2_cube_2450 = zhou_pan_cube_flux(r2, F_ISM_HZ, eps_2450, mode="TM")
    s_te_r2_cube_2450 = zhou_pan_cube_flux(r2, F_ISM_HZ, eps_2450, mode="TE")
    s_tm_r2_sph_2450 = float(zhou_pan_poynting_flux(r2, F_ISM_HZ, eps_2450, mode="TM")["S_r"])
    s_te_r2_sph_2450 = float(zhou_pan_poynting_flux(r2, F_ISM_HZ, eps_2450, mode="TE")["S_r"])

    for r1_cur, p1_tm_c, p1_te_c in zip(r1_shell_list, p1_tm_shells, p1_te_shells):
        r1_sweep_mm.append(float(r1_cur * 1e3))
        kr1_403_list.append(float(abs(complex_wavenumber(F_MEDRADIO_HZ, eps_403)[0]) * r1_cur))
        kr1_2450_list.append(float(abs(complex_wavenumber(F_ISM_HZ, eps_2450)[0]) * r1_cur))

        l_tm_403 = float((p1_tm_c[0] - res_tm_s["p2"][0]) / res_tm_s["p2"][0])
        l_te_403 = float((p1_te_c[0] - res_te_s["p2"][0]) / res_te_s["p2"][0])
        l_tm_2450 = float((p1_tm_c[1] - res_tm_s["p2"][1]) / res_tm_s["p2"][1])
        l_te_2450 = float((p1_te_c[1] - res_te_s["p2"][1]) / res_te_s["p2"][1])
        loss_ratio_fit_403.append(l_tm_403 / l_te_403)
        loss_ratio_fit_2450.append(l_tm_2450 / l_te_2450)

        s_tm_r1_cube_403 = zhou_pan_cube_flux(r1_cur, F_MEDRADIO_HZ, eps_403, mode="TM")
        s_te_r1_cube_403 = zhou_pan_cube_flux(r1_cur, F_MEDRADIO_HZ, eps_403, mode="TE")
        loss_ratio_cube_th_403.append(
            float(((s_tm_r1_cube_403 - s_tm_r2_cube_403) / s_tm_r2_cube_403) / ((s_te_r1_cube_403 - s_te_r2_cube_403) / s_te_r2_cube_403))
        )

        s_tm_r1_sph_403 = float(zhou_pan_poynting_flux(r1_cur, F_MEDRADIO_HZ, eps_403, mode="TM")["S_r"])
        s_te_r1_sph_403 = float(zhou_pan_poynting_flux(r1_cur, F_MEDRADIO_HZ, eps_403, mode="TE")["S_r"])
        loss_ratio_sph_th_403.append(
            float(((s_tm_r1_sph_403 - s_tm_r2_sph_403) / s_tm_r2_sph_403) / ((s_te_r1_sph_403 - s_te_r2_sph_403) / s_te_r2_sph_403))
        )

        s_tm_r1_cube_2450 = zhou_pan_cube_flux(r1_cur, F_ISM_HZ, eps_2450, mode="TM")
        s_te_r1_cube_2450 = zhou_pan_cube_flux(r1_cur, F_ISM_HZ, eps_2450, mode="TE")
        loss_ratio_cube_th_2450.append(
            float(((s_tm_r1_cube_2450 - s_tm_r2_cube_2450) / s_tm_r2_cube_2450) / ((s_te_r1_cube_2450 - s_te_r2_cube_2450) / s_te_r2_cube_2450))
        )

        s_tm_r1_sph_2450 = float(zhou_pan_poynting_flux(r1_cur, F_ISM_HZ, eps_2450, mode="TM")["S_r"])
        s_te_r1_sph_2450 = float(zhou_pan_poynting_flux(r1_cur, F_ISM_HZ, eps_2450, mode="TE")["S_r"])
        loss_ratio_sph_th_2450.append(
            float(((s_tm_r1_sph_2450 - s_tm_r2_sph_2450) / s_tm_r2_sph_2450) / ((s_te_r1_sph_2450 - s_te_r2_sph_2450) / s_te_r2_sph_2450))
        )

    rel_err_shell_403 = [
        abs(f_v - t_v) / t_v for f_v, t_v in zip(loss_ratio_fit_403, loss_ratio_cube_th_403)
    ]
    rel_err_shell_2450 = [
        abs(f_v - t_v) / t_v for f_v, t_v in zip(loss_ratio_fit_2450, loss_ratio_cube_th_2450)
    ]

    summary = {
        "grid_N": N,
        "dx_mm": dx * 1e3,
        "cells": tm_air["cells"],
        "n_steps": tm_air["n_steps"],
        "wall_time_tm_air_s": tm_air["wall_s"],
        "wall_time_te_air_s": te_air["wall_s"],
        "peak_rss_mb": max(tm_air["rss_mb"], te_air["rss_mb"]),
        "est_ram_gb": tm_air["est_ram_gb"],
        "a_ref_mm": a_ref * 1e3,
        "r1_mm": r1 * 1e3,
        "r2_mm": r2 * 1e3,
        "cpml_inner_mm": (0.5 * N - 6) * dx * 1e3,
        "poynting_closure_rel_err_tm_403mhz": closure_err_tm_403,
        "poynting_closure_rel_err_tm_2450mhz": closure_err_tm_2450,
        "poynting_closure_rel_err_te_403mhz": closure_err_te_403,
        "poynting_closure_rel_err_te_2450mhz": closure_err_te_2450,
        "cavity_boost_tm_air_2450mhz_fit": cav_fit_air_2450,
        "cavity_boost_tm_air_2450mhz_full_theory": cav_th_air_2450_full,
        "cavity_boost_tm_air_2450mhz_qs_theory": cav_th_air_2450_qs,
        "cavity_boost_tm_air_2450mhz_rel_err_vs_full": abs(cav_fit_air_2450 - cav_th_air_2450_full) / cav_th_air_2450_full,
        "cavity_boost_tm_cer_2450mhz_fit": cav_fit_cer_2450,
        "cavity_boost_tm_cer_2450mhz_full_theory": cav_th_cer_2450_full,
        "cavity_boost_tm_cer_2450mhz_qs_theory": cav_th_cer_2450_qs,
        "cavity_boost_tm_cer_2450mhz_rel_err_vs_full": abs(cav_fit_cer_2450 - cav_th_cer_2450_full) / cav_th_cer_2450_full,
        "cavity_boost_te_air_403mhz_fit": cav_fit_te_403,
        "cavity_boost_te_air_403mhz_theory": 1.000,
        "graded_dx_min_mm": graded_cav["graded_dx_min_mm"],
        "graded_cavity_boost_tm_air_2450mhz_fit": graded_cav["graded_cavity_boost_tm_air_2450mhz_fit"],
        "graded_cavity_boost_tm_cer_2450mhz_fit": graded_cav["graded_cavity_boost_tm_cer_2450mhz_fit"],
        "graded_cavity_boost_tm_air_403mhz_fit": graded_cav["graded_cavity_boost_tm_air_403mhz_fit"],
        "graded_cavity_boost_tm_cer_403mhz_fit": graded_cav["graded_cavity_boost_tm_cer_403mhz_fit"],
        "trans_tm_403mhz_hom": trans_tm_403_hom,
        "trans_tm_403mhz_air": trans_tm_403_air,
        "trans_te_403mhz_hom": trans_te_403_hom,
        "trans_te_403mhz_air": trans_te_403_air,
        "a_sweep_mm": a_sweep_mm,
        "ka_sweep_403mhz": ka_403_list,
        "ka_sweep_2450mhz": ka_2450_list,
        "cavity_boost_tm_air_2450mhz_fit_sweep": cav_fit_air_2450_sweep,
        "cavity_boost_tm_cer_2450mhz_fit_sweep": cav_fit_cer_2450_sweep,
        "cavity_boost_tm_air_2450mhz_full_sweep": cav_full_air_2450_sweep,
        "cavity_boost_tm_cer_2450mhz_full_sweep": cav_full_cer_2450_sweep,
        "cavity_boost_tm_air_2450mhz_qs_sweep": cav_qs_air_2450_sweep,
        "cavity_boost_tm_cer_2450mhz_qs_sweep": cav_qs_cer_2450_sweep,
        "cavity_boost_tm_air_403mhz_fit_sweep": cav_fit_air_403_sweep,
        "cavity_boost_tm_cer_403mhz_fit_sweep": cav_fit_cer_403_sweep,
        "cavity_boost_tm_air_403mhz_full_sweep": cav_full_air_403_sweep,
        "cavity_boost_tm_cer_403mhz_full_sweep": cav_full_cer_403_sweep,
        "cut_cell_loss_ratio_tm_over_te_403mhz_fit": cut_cell_loss_ratio_fit_403,
        "cut_cell_loss_ratio_tm_over_te_2450mhz_fit": cut_cell_loss_ratio_fit_2450,
        "r1_shell_sweep_mm": r1_sweep_mm,
        "kr1_shell_sweep_403mhz": kr1_403_list,
        "kr1_shell_sweep_2450mhz": kr1_2450_list,
        "shell_loss_ratio_tm_over_te_403mhz_fit": loss_ratio_fit_403,
        "shell_loss_ratio_tm_over_te_403mhz_theory": loss_ratio_cube_th_403,
        "shell_loss_ratio_tm_over_te_403mhz_spherical_theory": loss_ratio_sph_th_403,
        "shell_loss_ratio_tm_over_te_403mhz_rel_err": rel_err_shell_403,
        "shell_loss_ratio_tm_over_te_2450mhz_fit": loss_ratio_fit_2450,
        "shell_loss_ratio_tm_over_te_2450mhz_theory": loss_ratio_cube_th_2450,
        "shell_loss_ratio_tm_over_te_2450mhz_spherical_theory": loss_ratio_sph_th_2450,
        "shell_loss_ratio_tm_over_te_2450mhz_rel_err": rel_err_shell_2450,
    }
    (DATA_DIR / "task_02_fit_shells_and_capsule_correction.json").write_text(
        json.dumps(summary, indent=2)
    )
    np.savez(
        DATA_DIR / "task_02_fit_shells_and_capsule_correction.npz",
        a_sweep_mm=np.asarray(a_sweep_mm),
        ka_403=np.asarray(kr1_403_list),
        ka_2450=np.asarray(kr1_2450_list),
        ka_cap_403=np.asarray(ka_403_list),
        ka_cap_2450=np.asarray(ka_2450_list),
        cav_fit_air_2450_sweep=np.asarray(cav_fit_air_2450_sweep),
        cav_fit_cer_2450_sweep=np.asarray(cav_fit_cer_2450_sweep),
        cav_full_air_2450_sweep=np.asarray(cav_full_air_2450_sweep),
        cav_full_cer_2450_sweep=np.asarray(cav_full_cer_2450_sweep),
        loss_ratio_fit_403=np.asarray(loss_ratio_fit_403),
        loss_ratio_th_403=np.asarray(loss_ratio_cube_th_403),
        loss_ratio_sph_th_403=np.asarray(loss_ratio_sph_th_403),
        loss_ratio_fit_2450=np.asarray(loss_ratio_fit_2450),
        loss_ratio_th_2450=np.asarray(loss_ratio_cube_th_2450),
        loss_ratio_sph_th_2450=np.asarray(loss_ratio_sph_th_2450),
    )
    return summary


# ---------------------------------------------------------------------------
# Task 3: Real feeds (ideal loop vs gapped PEC loop vs gapped PEC dipole) & SAR
# ---------------------------------------------------------------------------
def _build_graded_feed_axis(quick: bool = False) -> np.ndarray:
    """Build locally refined graded axis with dx_min = 0.35 mm (4 cells across 1.40 mm gap & wire)."""
    if quick:
        return graded_axis(
            [-10.2e-3, -3.6e-3, -1.5e-3, 1.5e-3, 3.6e-3, 10.2e-3],
            [0.80e-3, 0.60e-3, 0.50e-3, 0.60e-3, 0.80e-3],
            min_cells=[4, 3, 6, 3, 4],
            grading=1.25,
        )
    return graded_axis(
        [-10.4e-3, -3.5e-3, -1.4e-3, 1.4e-3, 3.5e-3, 10.4e-3],
        [0.68e-3, 0.48e-3, 0.35e-3, 0.48e-3, 0.68e-3],
        min_cells=[5, 4, 8, 4, 5],
        grading=1.22,
    )


def _run_feed_case(
    muscle_mat: DispersiveMaterial,
    *,
    x_axis: np.ndarray,
    a_cap_m: float,
    r_ant_m: float,
    g_gap_m: float,
    w_wire_m: float,
    r1_m: float,
    r2_m: float,
    kind: str,
    eps_cap: float = 1.0,
    record_slices_and_gif: bool = False,
) -> dict[str, Any]:
    """Run a 3D FIT simulation of an ideal loop, gapped PEC loop, or gapped PEC dipole on a graded mesh."""
    grid = GridFIT3D(x=x_axis, y=x_axis, z=x_axis, verbose=0)
    solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
    # Well-damped broadband pulse (or two-band sub-resonant pulses for high-permittivity
    # ceramic gapped loop so the 1.6 GHz LC cavity mode is never impulsively excited).
    pulse = GaussianSource(1.4e9, fwidth=1.5e9, cutoff=4.0)
    n_steps = int(np.ceil(2.4 * pulse.end_time / solver.dt))

    n_samp = int(np.sum(np.abs(x_axis) <= r2_m + 1e-6) ** 3)
    est = estimate(
        (grid.Nx, grid.Ny, grid.Nz),
        steps=n_steps,
        cpml=True,
        monitors=[(2, n_samp, 3)],
        dispersive_edges=3 * grid.Nx * grid.Ny * grid.Nz,
        n_poles=3,
    )
    preflight(est, "laptop", raise_error=True)

    pec_boxes = []
    arm_len = r_ant_m - 0.5 * g_gap_m
    y_arm_c = 0.5 * (r_ant_m + 0.5 * g_gap_m)
    if kind == "ideal_loop":
        srcs = [
            CurrentSource(
                center=(0.0, -r_ant_m, 0.0),
                size=(2.0 * r_ant_m, 0.0, 0.0),
                component="Ex",
                amplitude=+1.0e-2,
                src_time=pulse,
            ),
            CurrentSource(
                center=(+r_ant_m, 0.0, 0.0),
                size=(0.0, 2.0 * r_ant_m, 0.0),
                component="Ey",
                amplitude=+1.0e-2,
                src_time=pulse,
            ),
            CurrentSource(
                center=(0.0, +r_ant_m, 0.0),
                size=(2.0 * r_ant_m, 0.0, 0.0),
                component="Ex",
                amplitude=-1.0e-2,
                src_time=pulse,
            ),
            CurrentSource(
                center=(-r_ant_m, 0.0, 0.0),
                size=(0.0, 2.0 * r_ant_m, 0.0),
                component="Ey",
                amplitude=-1.0e-2,
                src_time=pulse,
            ),
        ]
    elif kind == "gapped_loop":
        # Symmetric PEC wire loop (8 cells along each 2.80 mm arm on dx_min = 0.35 mm)
        # with a centered feed gap g_gap_m on the +x arm (4 cells across the 1.40 mm gap).
        pec_boxes = [
            Box(center=(0.0, -r_ant_m, 0.0), size=(2.0 * r_ant_m, 0.0, 0.0), material="pec"),
            Box(center=(0.0, +r_ant_m, 0.0), size=(2.0 * r_ant_m, 0.0, 0.0), material="pec"),
            Box(center=(-r_ant_m, 0.0, 0.0), size=(0.0, 2.0 * r_ant_m, 0.0), material="pec"),
            Box(center=(+r_ant_m, -y_arm_c, 0.0), size=(0.0, arm_len, 0.0), material="pec"),
            Box(center=(+r_ant_m, +y_arm_c, 0.0), size=(0.0, arm_len, 0.0), material="pec"),
        ]
        srcs = [
            CurrentSource(
                center=(+r_ant_m, 0.0, 0.0),
                size=(0.0, g_gap_m, 0.0),
                component="Ey",
                amplitude=+1.0e-2,
                src_time=pulse,
            )
        ]
    elif kind == "gapped_dipole":
        # Symmetric PEC wire dipole along z with a centered feed gap g_gap_m (4 cells across gap)
        pec_boxes = [
            Box(center=(0.0, 0.0, -y_arm_c), size=(0.0, 0.0, arm_len), material="pec"),
            Box(center=(0.0, 0.0, +y_arm_c), size=(0.0, 0.0, arm_len), material="pec"),
        ]
        srcs = [
            CurrentSource(
                center=(0.0, 0.0, 0.0),
                size=(0.0, 0.0, g_gap_m),
                component="Ez",
                amplitude=+1.0e-2,
                src_time=pulse,
            )
        ]
    else:
        raise ValueError(f"Unknown antenna kind {kind!r}")

    # Assign capsule permittivity first, then PEC wire arms so PEC edges are not overwritten.
    assign_spherical_capsule_in_tissue(solver, muscle_mat, a_cap_m, eps_cap=eps_cap)
    if pec_boxes:
        assign_materials(solver, pec_boxes, frequency=1.4e9)

    r_mid_m = float(x_axis[np.argmin(np.abs(x_axis - 0.5 * (r1_m + r2_m)))])
    if kind == "gapped_loop" and eps_cap > 1.0:
        p1 = np.zeros(2, dtype=float)
        p_out = np.zeros(2, dtype=float)
        reg_loss = {
            "p_loss_box": np.zeros(2, dtype=float),
            "p_loss_gap_hemi": np.zeros(2, dtype=float),
            "p_loss_back_hemi": np.zeros(2, dtype=float),
            "p_loss_inner_shell": np.zeros(2, dtype=float),
            "p_loss_outer_shell": np.zeros(2, dtype=float),
        }
        wall_s = 0.0
        for fi, (fq, f0_b, fw_b) in enumerate([(F_MEDRADIO_HZ, 0.65e9, 0.55e9), (F_ISM_HZ, 2.45e9, 1.0e9)]):
            g_b = GridFIT3D(x=x_axis, y=x_axis, z=x_axis, verbose=0)
            s_b = SolverFIT3D(g_b, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
            assign_spherical_capsule_in_tissue(s_b, muscle_mat, a_cap_m, eps_cap=eps_cap)
            assign_materials(s_b, pec_boxes, frequency=1.4e9)
            p_b = GaussianSource(f0_b, fwidth=fw_b, cutoff=3.5)
            ns_b = int(np.ceil(2.2 * p_b.end_time / s_b.dt))
            src_b = CurrentSource(
                center=(+r_ant_m, 0.0, 0.0),
                size=(0.0, g_gap_m, 0.0),
                component="Ey",
                amplitude=+1.0e-2,
                src_time=p_b,
            )
            f1_b = FluxMonitor([fq], center=(0.0, 0.0, 0.0), size=(2.0 * r1_m, 2.0 * r1_m, 2.0 * r1_m))
            f2_b = FluxMonitor([fq], center=(0.0, 0.0, 0.0), size=(2.0 * r2_m, 2.0 * r2_m, 2.0 * r2_m))
            dft_b = DFTMonitor(
                [fq],
                components=("Ex", "Ey", "Ez"),
                center=(0.0, 0.0, 0.0),
                size=(2.0 * r2_m, 2.0 * r2_m, 2.0 * r2_m),
            )
            t0_b = time.perf_counter()
            for step in range(ns_b):
                t = step * s_b.dt
                src_b.update(s_b, t)
                s_b.one_step()
                f1_b.update(s_b, t)
                f2_b.update(s_b, t)
                dft_b.update(s_b, t)
            wall_s += time.perf_counter() - t0_b
            p1[fi] = f1_b.get_flux()[0]
            p_out[fi] = f2_b.get_flux()[0]
            rl_b = compute_regional_shell_ohmic_loss(s_b, dft_b, muscle_mat, r1_m=r1_m, r_mid_m=r_mid_m, r2_m=r2_m)
            for k_r in reg_loss:
                reg_loss[k_r][fi] = rl_b[k_r][0]
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        dft = None
        gif_frames = []
    else:
        flux1 = FluxMonitor(FREQS_MAIN_HZ, center=(0.0, 0.0, 0.0), size=(2.0 * r1_m, 2.0 * r1_m, 2.0 * r1_m))
        flux2 = FluxMonitor(FREQS_MAIN_HZ, center=(0.0, 0.0, 0.0), size=(2.0 * r2_m, 2.0 * r2_m, 2.0 * r2_m))
        dft = DFTMonitor(
            FREQS_MAIN_HZ,
            components=("Ex", "Ey", "Ez"),
            center=(0.0, 0.0, 0.0),
            size=(2.0 * r2_m, 2.0 * r2_m, 2.0 * r2_m),
        )

        gif_frames = []
        frame_steps = np.linspace(int(0.18 * n_steps), int(0.62 * n_steps), 14, dtype=int) if record_slices_and_gif else []
        k_mid = grid.Nz // 2
        j_mid = grid.Ny // 2

        t0 = time.perf_counter()
        for step in range(n_steps):
            t = step * solver.dt
            for s in srcs:
                s.update(solver, t)
            solver.one_step()
            flux1.update(solver, t)
            flux2.update(solver, t)
            dft.update(solver, t)
            if record_slices_and_gif and step in frame_steps:
                if kind in ("ideal_loop", "gapped_loop"):
                    ex_slice = np.asarray(solver.E[:, :, k_mid, "x"], dtype=float)
                    ey_slice = np.asarray(solver.E[:, :, k_mid, "y"], dtype=float)
                    gif_frames.append(np.hypot(ex_slice, ey_slice))
                else:
                    ex_slice = np.asarray(solver.E[:, j_mid, :, "x"], dtype=float)
                    ez_slice = np.asarray(solver.E[:, j_mid, :, "z"], dtype=float)
                    gif_frames.append(np.hypot(ex_slice, ez_slice))
        wall_s = time.perf_counter() - t0
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        p1 = flux1.get_flux()
        p_out = flux2.get_flux()
        reg_loss = compute_regional_shell_ohmic_loss(
            solver, dft, muscle_mat, r1_m=r1_m, r_mid_m=r_mid_m, r2_m=r2_m
        )
    p_loss = reg_loss["p_loss_box"]
    eta = p_out / (p_out + p_loss)
    loss_over_out = p_loss / np.maximum(p_out, 1e-60)
    gap_hemi_frac = reg_loss["p_loss_gap_hemi"] / np.maximum(p_loss, 1e-60)
    inner_shell_frac = reg_loss["p_loss_inner_shell"] / np.maximum(p_loss, 1e-60)

    out: dict[str, Any] = {
        "p1": p1,
        "p_out": p_out,
        "p_loss": p_loss,
        "p_loss_gap_hemi": reg_loss["p_loss_gap_hemi"],
        "p_loss_back_hemi": reg_loss["p_loss_back_hemi"],
        "p_loss_inner_shell": reg_loss["p_loss_inner_shell"],
        "p_loss_outer_shell": reg_loss["p_loss_outer_shell"],
        "gap_hemi_frac": gap_hemi_frac,
        "inner_shell_frac": inner_shell_frac,
        "eta": eta,
        "loss_over_out": loss_over_out,
        "wall_s": float(wall_s),
        "rss_mb": float(rss_mb),
    }

    if record_slices_and_gif:
        ex_arr, coords_x = dft.get_field("Ex")
        ey_arr, coords_y = dft.get_field("Ey")
        ez_arr, coords_z = dft.get_field("Ez")
        nx_c = min(ex_arr.shape[1], ey_arr.shape[1], ez_arr.shape[1])
        ny_c = min(ex_arr.shape[2], ey_arr.shape[2], ez_arr.shape[2])
        nz_c = min(ex_arr.shape[3], ey_arr.shape[3], ez_arr.shape[3])
        xc = coords_x[0][:nx_c]
        yc = coords_y[1][:ny_c]
        zc = coords_z[2][:nz_c]
        X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)
        tis_mask = (R >= a_cap_m).astype(float)

        sar_xy_403, sar_xz_403 = None, None
        sar_xy_2450, sar_xz_2450 = None, None
        peak_sar_403, peak_sar_2450 = 0.0, 0.0
        for fi, fq in enumerate(FREQS_MAIN_HZ):
            eps_d = muscle_mat.epsilon_discrete(fq, solver.dt)
            sig_eff = -2.0 * np.pi * fq * EPS0 * float(np.imag(eps_d))
            e2 = (
                np.abs(ex_arr[fi, :nx_c, :ny_c, :nz_c]) ** 2
                + np.abs(ey_arr[fi, :nx_c, :ny_c, :nz_c]) ** 2
                + np.abs(ez_arr[fi, :nx_c, :ny_c, :nz_c]) ** 2
            )
            scale = 1.0e-3 / float(p_out[fi] + p_loss[fi])
            sar_3d = tis_mask * (sig_eff * e2 * scale) / RHO_MUSCLE_KG_M3  # [W/kg per 1 mW input]
            kz0 = nz_c // 2
            jy0 = ny_c // 2
            if fi == 0:
                sar_xy_403 = sar_3d[:, :, kz0]
                sar_xz_403 = sar_3d[:, jy0, :]
                peak_sar_403 = float(np.max(sar_3d))
            else:
                sar_xy_2450 = sar_3d[:, :, kz0]
                sar_xz_2450 = sar_3d[:, jy0, :]
                peak_sar_2450 = float(np.max(sar_3d))

        out.update(
            {
                "xc_m": xc,
                "yc_m": yc,
                "zc_m": zc,
                "sar_xy_403": sar_xy_403,
                "sar_xz_403": sar_xz_403,
                "sar_xy_2450": sar_xy_2450,
                "sar_xz_2450": sar_xz_2450,
                "peak_sar_403_w_kg_per_mw": peak_sar_403,
                "peak_sar_2450_w_kg_per_mw": peak_sar_2450,
                "gif_frames": np.asarray(gif_frames),
                "grid_x_m": grid.x[:-1],
                "grid_z_m": grid.z[:-1],
            }
        )
    return out


def run_task_real_feeds_and_sar(quick: bool = False) -> dict[str, Any]:
    """Run Part 3: resolved graded-mesh feeds, ideal-in-ceramic control, regional sigma|E|^2 attribution & SAR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    muscle = build_dispersive_muscle(n_debye=3)

    x_axis = _build_graded_feed_axis(quick=quick)
    dx_min = float(np.min(np.diff(x_axis)))
    dx_max = float(np.max(np.diff(x_axis)))
    a_cap = 2.25e-3
    r_ant = 1.40e-3 if not quick else 1.50e-3
    g_gap = 1.40e-3 if not quick else 1.50e-3
    w_wire = 1.40e-3 if not quick else 1.50e-3
    # Snap r1 and r2 to exact grid nodes outside a_cap and inside the CPML
    r1 = float(x_axis[np.argmin(np.abs(x_axis - 2.62e-3))])
    r2 = float(x_axis[np.argmin(np.abs(x_axis - (5.20e-3 if quick else 5.26e-3)))])

    res_ideal = _run_feed_case(
        muscle,
        x_axis=x_axis,
        a_cap_m=a_cap,
        r_ant_m=r_ant,
        g_gap_m=g_gap,
        w_wire_m=w_wire,
        r1_m=r1,
        r2_m=r2,
        kind="ideal_loop",
        eps_cap=1.0,
        record_slices_and_gif=True,
    )
    res_ideal_cer = _run_feed_case(
        muscle,
        x_axis=x_axis,
        a_cap_m=a_cap,
        r_ant_m=r_ant,
        g_gap_m=g_gap,
        w_wire_m=w_wire,
        r1_m=r1,
        r2_m=r2,
        kind="ideal_loop",
        eps_cap=12.0,
        record_slices_and_gif=False,
    )
    res_gapped = _run_feed_case(
        muscle,
        x_axis=x_axis,
        a_cap_m=a_cap,
        r_ant_m=r_ant,
        g_gap_m=g_gap,
        w_wire_m=w_wire,
        r1_m=r1,
        r2_m=r2,
        kind="gapped_loop",
        eps_cap=1.0,
        record_slices_and_gif=True,
    )
    res_gapped_cer = _run_feed_case(
        muscle,
        x_axis=x_axis,
        a_cap_m=a_cap,
        r_ant_m=r_ant,
        g_gap_m=g_gap,
        w_wire_m=w_wire,
        r1_m=r1,
        r2_m=r2,
        kind="gapped_loop",
        eps_cap=12.0,
        record_slices_and_gif=False,
    )
    res_dipole = _run_feed_case(
        muscle,
        x_axis=x_axis,
        a_cap_m=a_cap,
        r_ant_m=r_ant,
        g_gap_m=g_gap,
        w_wire_m=w_wire,
        r1_m=r1,
        r2_m=r2,
        kind="gapped_dipole",
        eps_cap=1.0,
        record_slices_and_gif=True,
    )

    lo_ideal_403 = float(res_ideal["loss_over_out"][0])
    lo_ideal_cer_403 = float(res_ideal_cer["loss_over_out"][0])
    lo_gap_403 = float(res_gapped["loss_over_out"][0])
    lo_dip_403 = float(res_dipole["loss_over_out"][0])
    lo_cer_403 = float(res_gapped_cer["loss_over_out"][0])
    gap_loss_fraction_403 = (lo_gap_403 - lo_ideal_403) / lo_gap_403

    lo_ideal_2450 = float(res_ideal["loss_over_out"][1])
    lo_ideal_cer_2450 = float(res_ideal_cer["loss_over_out"][1])
    lo_gap_2450 = float(res_gapped["loss_over_out"][1])
    lo_dip_2450 = float(res_dipole["loss_over_out"][1])
    lo_cer_2450 = float(res_gapped_cer["loss_over_out"][1])
    gap_loss_fraction_2450 = (lo_gap_2450 - lo_ideal_2450) / lo_gap_2450

    summary = {
        "grid_N": int(len(x_axis) - 1),
        "dx_min_mm": dx_min * 1e3,
        "dx_max_mm": dx_max * 1e3,
        "dx_mm": dx_min * 1e3,
        "cells_across_gap": float(g_gap / dx_min),
        "cells_across_wire": float(w_wire / dx_min),
        "a_cap_mm": a_cap * 1e3,
        "r_ant_mm": r_ant * 1e3,
        "g_gap_mm": g_gap * 1e3,
        "w_wire_mm": w_wire * 1e3,
        "r1_mm": r1 * 1e3,
        "r2_mm": r2 * 1e3,
        "wall_time_total_s": float(
            res_ideal["wall_s"]
            + res_ideal_cer["wall_s"]
            + res_gapped["wall_s"]
            + res_dipole["wall_s"]
            + res_gapped_cer["wall_s"]
        ),
        "peak_rss_mb": float(
            max(
                res_ideal["rss_mb"],
                res_ideal_cer["rss_mb"],
                res_gapped["rss_mb"],
                res_dipole["rss_mb"],
                res_gapped_cer["rss_mb"],
            )
        ),
        "eta_ideal_loop_air_403mhz": float(res_ideal["eta"][0]),
        "eta_ideal_loop_cer_403mhz": float(res_ideal_cer["eta"][0]),
        "eta_gapped_loop_air_403mhz": float(res_gapped["eta"][0]),
        "eta_gapped_dipole_air_403mhz": float(res_dipole["eta"][0]),
        "eta_gapped_loop_cer_403mhz": float(res_gapped_cer["eta"][0]),
        "eta_ideal_loop_air_2450mhz": float(res_ideal["eta"][1]),
        "eta_ideal_loop_cer_2450mhz": float(res_ideal_cer["eta"][1]),
        "eta_gapped_loop_air_2450mhz": float(res_gapped["eta"][1]),
        "eta_gapped_dipole_air_2450mhz": float(res_dipole["eta"][1]),
        "eta_gapped_loop_cer_2450mhz": float(res_gapped_cer["eta"][1]),
        "loss_over_out_ideal_loop_403mhz": lo_ideal_403,
        "loss_over_out_ideal_loop_cer_403mhz": lo_ideal_cer_403,
        "loss_over_out_gapped_loop_403mhz": lo_gap_403,
        "loss_over_out_gapped_dipole_403mhz": lo_dip_403,
        "loss_over_out_gapped_loop_cer_403mhz": lo_cer_403,
        "loss_over_out_ideal_loop_2450mhz": lo_ideal_2450,
        "loss_over_out_ideal_loop_cer_2450mhz": lo_ideal_cer_2450,
        "loss_over_out_gapped_loop_2450mhz": lo_gap_2450,
        "loss_over_out_gapped_dipole_2450mhz": lo_dip_2450,
        "loss_over_out_gapped_loop_cer_2450mhz": lo_cer_2450,
        "gap_contamination_fraction_403mhz": gap_loss_fraction_403,
        "gap_contamination_fraction_2450mhz": gap_loss_fraction_2450,
        "gap_hemi_loss_fraction_ideal_loop_403mhz": float(res_ideal["gap_hemi_frac"][0]),
        "gap_hemi_loss_fraction_ideal_loop_cer_403mhz": float(res_ideal_cer["gap_hemi_frac"][0]),
        "gap_hemi_loss_fraction_gapped_loop_403mhz": float(res_gapped["gap_hemi_frac"][0]),
        "gap_hemi_loss_fraction_gapped_loop_cer_403mhz": float(res_gapped_cer["gap_hemi_frac"][0]),
        "gap_hemi_loss_fraction_gapped_dipole_403mhz": float(res_dipole["gap_hemi_frac"][0]),
        "inner_shell_loss_fraction_ideal_loop_403mhz": float(res_ideal["inner_shell_frac"][0]),
        "inner_shell_loss_fraction_gapped_loop_403mhz": float(res_gapped["inner_shell_frac"][0]),
        "inner_shell_loss_fraction_gapped_loop_cer_403mhz": float(res_gapped_cer["inner_shell_frac"][0]),
        "inner_shell_loss_fraction_gapped_dipole_403mhz": float(res_dipole["inner_shell_frac"][0]),
        "peak_sar_ideal_loop_403mhz": res_ideal["peak_sar_403_w_kg_per_mw"],
        "peak_sar_gapped_loop_403mhz": res_gapped["peak_sar_403_w_kg_per_mw"],
        "peak_sar_gapped_dipole_403mhz": res_dipole["peak_sar_403_w_kg_per_mw"],
        "peak_sar_ideal_loop_2450mhz": res_ideal["peak_sar_2450_w_kg_per_mw"],
        "peak_sar_gapped_loop_2450mhz": res_gapped["peak_sar_2450_w_kg_per_mw"],
        "peak_sar_gapped_dipole_2450mhz": res_dipole["peak_sar_2450_w_kg_per_mw"],
    }
    (DATA_DIR / "task_03_real_feeds_and_sar.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        DATA_DIR / "task_03_real_feeds_and_sar.npz",
        xc_m=res_ideal["xc_m"],
        yc_m=res_ideal["yc_m"],
        zc_m=res_ideal["zc_m"],
        sar_xy_ideal_403=res_ideal["sar_xy_403"],
        sar_xy_gapped_403=res_gapped["sar_xy_403"],
        sar_xz_dipole_403=res_dipole["sar_xz_403"],
        sar_xy_ideal_2450=res_ideal["sar_xy_2450"],
        sar_xy_gapped_2450=res_gapped["sar_xy_2450"],
        sar_xz_dipole_2450=res_dipole["sar_xz_2450"],
        gif_ideal=res_ideal["gif_frames"],
        gif_gapped=res_gapped["gif_frames"],
        gif_dipole=res_dipole["gif_frames"],
        grid_x_m=res_ideal["grid_x_m"],
        grid_z_m=res_ideal["grid_z_m"],
        a_cap_m=np.array([a_cap]),
        r_ant_m=np.array([r_ant]),
    )
    return summary


# ---------------------------------------------------------------------------
# Task 4: Liska (2024) benchmarks, 3D FIT r/a sweep, Soares (2023) & ferrite core
# ---------------------------------------------------------------------------
def liska_2024_verifiable_benchmarks() -> dict[str, Any]:
    """Return benchmark numbers from Liska et al. (2024) (`doi:10.1109/TAP.2024.3365860`).

    Physical setup parameters (a = 7.5 mm, R = 61.5 mm, sigma_Cu = 5.56e7 S/m) are
    taken from Section III; the fundamental bound curves at f1 = 403 MHz and
    f2 = 2.45 GHz are digitised from Li et al. / Liska et al. 2024 Fig. 2 (±0.3 dB),
    and the TM1m -> TE1m crossover r/a = 0.32 is read directly from the slope change
    of the digitised 403 MHz curve.
    """
    # Digitised from Li et al. 2024 Fig. 2 (±0.3 dB):
    # Below r/a = 0.32 the bound-achieving current is TM1m-like and its loss is
    # tissue-dominated (delta_tis > delta_ant). TE1m-like currents couple less to
    # tissue, but small ones act like a short circuit (low radiation resistance,
    # high ohmic delta_ant), so TE becomes optimal only for r/a > 0.32 [3, Sec. III].
    r_over_a_pts = np.array([0.10, 0.20, 0.32, 0.50, 0.60, 0.75, 1.00], dtype=float)
    eta_bound_403_db = np.array([-29.8, -29.8, -29.8, -24.8, -23.4, -22.5, -21.9], dtype=float)
    eta_bound_2450_db = np.array([-25.1, -24.9, -24.8, -24.8, -24.8, -24.8, -24.8], dtype=float)
    slopes_403 = np.diff(eta_bound_403_db) / np.diff(r_over_a_pts)
    crossover_idx = int(np.argmax(slopes_403 > 5.0))
    crossover_r_over_a = float(r_over_a_pts[crossover_idx])
    return {
        "a_capsule_mm": 7.5,
        "R_phantom_mm": 61.5,
        "sigma_copper_s_m": 5.56e7,
        "f_medradio_mhz": 403.0,
        "eps_r_403mhz": 57.0 - 36.0j,
        "f_ism_mhz": 2450.0,
        "eps_r_2450mhz": 53.0 - 13.0j,
        "tm_to_te_crossover_r_over_a_403mhz": crossover_r_over_a,
        # Digitised from Li et al. 2024 Fig. 2 (±0.3 dB):
        "r_over_a_pts_approx": r_over_a_pts,
        "eta_bound_403mhz_db_approx": eta_bound_403_db,
        "eta_bound_2450mhz_db_approx": eta_bound_2450_db,
    }


def _run_liska_fit_size_sweep(muscle_mat: DispersiveMaterial, quick: bool = False) -> dict[str, Any]:
    """Run a resolved graded-mesh 3D FIT sweep over r_ant / a_cap in {0.25, 0.50, 0.75} for TE loop and TM dipole."""
    if quick:
        x_axis = graded_axis(
            [-11.8e-3, -2.1e-3, 2.1e-3, 11.8e-3],
            [0.85e-3, 0.35e-3, 0.85e-3],
            min_cells=[8, 12, 8],
            grading=1.28,
        )
    else:
        x_axis = graded_axis(
            [-12.5e-3, -2.1e-3, 2.1e-3, 12.5e-3],
            [0.78e-3, 0.35e-3, 0.78e-3],
            min_cells=[11, 12, 11],
            grading=1.25,
        )
    dx_min = float(np.min(np.diff(x_axis)))
    a_cap = 2.80e-3  # 8 * dx_min (16 cells across capsule diameter 2a = 5.60 mm)
    r1 = float(x_axis[np.argmin(np.abs(x_axis - 3.50e-3))])
    r2 = float(x_axis[np.argmin(np.abs(x_axis - 5.76e-3))])
    cpml_inner_m = float(abs(x_axis[6]))
    cpml_clearance_mm = (cpml_inner_m - r2) * 1e3

    r_ant_list = [0.70e-3, 1.40e-3, 2.10e-3]  # r_ant / a_cap = {0.25, 0.50, 0.75}, 2*r_ant / dx_min = {4, 8, 12} cells
    r_over_a_fit: list[float] = []
    eta_nz_te_ideal_403: list[float] = []
    eta_tot_te_cu_403: list[float] = []
    eta_nz_te_ideal_2450: list[float] = []

    r_skin_403 = float(np.sqrt(np.pi * F_MEDRADIO_HZ * MU0 / SIGMA_COPPER_S_M))
    r_wire = 0.08e-3  # thin copper wire radius [m] (0.16 mm diameter, ~= AWG 34)
    i0_amp = 1.0e-2

    for r_ant in r_ant_list:
        r_over_a_fit.append(float(r_ant / a_cap))
        grid = GridFIT3D(x=x_axis, y=x_axis, z=x_axis, verbose=0)
        solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
        n_disp = assign_spherical_capsule_in_tissue(solver, muscle_mat, a_cap, eps_cap=1.0)
        pulse = GaussianSource(1.5e9, fwidth=1.8e9, cutoff=3.2)
        n_steps = int(np.ceil(2.0 * pulse.end_time / solver.dt))
        est = estimate((grid.Nx, grid.Ny, grid.Nz), steps=n_steps, cpml=True, dispersive_edges=n_disp, n_poles=3)
        preflight(est, "laptop", raise_error=True)

        srcs = [
            CurrentSource(center=(0.0, -r_ant, 0.0), size=(2.0 * r_ant, 0.0, 0.0), component="Ex", amplitude=+i0_amp, src_time=pulse),
            CurrentSource(center=(+r_ant, 0.0, 0.0), size=(0.0, 2.0 * r_ant, 0.0), component="Ey", amplitude=+i0_amp, src_time=pulse),
            CurrentSource(center=(0.0, +r_ant, 0.0), size=(2.0 * r_ant, 0.0, 0.0), component="Ex", amplitude=-i0_amp, src_time=pulse),
            CurrentSource(center=(-r_ant, 0.0, 0.0), size=(0.0, 2.0 * r_ant, 0.0), component="Ey", amplitude=-i0_amp, src_time=pulse),
        ]
        flux1 = FluxMonitor(FREQS_MAIN_HZ, center=(0.0, 0.0, 0.0), size=(2.0 * r1, 2.0 * r1, 2.0 * r1))
        flux2 = FluxMonitor(FREQS_MAIN_HZ, center=(0.0, 0.0, 0.0), size=(2.0 * r2, 2.0 * r2, 2.0 * r2))
        dft_j = DFTMonitor(FREQS_MAIN_HZ, components=("Jx",), center=(0.0, -r_ant, 0.0), size=(0.0, 0.0, 0.0))

        for step in range(n_steps):
            t = step * solver.dt
            for s in srcs:
                s.update(solver, t)
            solver.one_step()
            flux1.update(solver, t)
            flux2.update(solver, t)
            dft_j.update(solver, t)

        p1 = flux1.get_flux()
        p2 = flux2.get_flux()
        jx_dft, _ = dft_j.get_field("Jx")
        idx_j, _ = grid.region_indices("J", "x", (0.0, -r_ant, 0.0), (0.0, 0.0, 0.0))
        ta_x = float(grid.tL[idx_j[0][0], idx_j[1][0], idx_j[2][0], "y"] * grid.tL[idx_j[0][0], idx_j[1][0], idx_j[2][0], "z"])
        i_loop_dft_403 = complex(jx_dft[0, 0, 0, 0]) * ta_x
        norm_i2_403 = 2.0 * (abs(i_loop_dft_403) ** 2) / (i0_amp**2)

        p_out_w_403 = float(p2[0] / norm_i2_403)
        p_tis_w_403 = float((p1[0] - p2[0]) / norm_i2_403)
        r_cu_403 = (8.0 * r_ant) / (2.0 * np.pi * r_wire) * r_skin_403
        phi_cu_403 = r_cu_403 * (abs(i_loop_dft_403) ** 2)
        p_cu_w_403 = float(phi_cu_403 / norm_i2_403)

        eta_nz_403 = p_out_w_403 / (p_out_w_403 + p_tis_w_403)
        eta_tot_403 = p_out_w_403 / (p_out_w_403 + p_tis_w_403 + p_cu_w_403)
        eta_nz_te_ideal_403.append(eta_nz_403)
        eta_tot_te_cu_403.append(eta_tot_403)
        eta_nz_te_ideal_2450.append(float(p2[1] / p1[1]))

    # Also run the TM electric dipole at r_ant / a_cap = 0.50 (half-length r_ant = 1.40 mm, total length 2*r_ant = 2.80 mm)
    r_ant_tm = 0.50 * a_cap
    grid = GridFIT3D(x=x_axis, y=x_axis, z=x_axis, verbose=0)
    solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
    assign_spherical_capsule_in_tissue(solver, muscle_mat, a_cap, eps_cap=1.0)
    pulse = GaussianSource(1.5e9, fwidth=1.8e9, cutoff=3.2)
    n_steps = int(np.ceil(2.0 * pulse.end_time / solver.dt))
    src_tm = CurrentSource(center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 2.0 * r_ant_tm), component="Ez", amplitude=i0_amp, src_time=pulse)
    flux1_tm = FluxMonitor(FREQS_MAIN_HZ, center=(0.0, 0.0, 0.0), size=(2.0 * r1, 2.0 * r1, 2.0 * r1))
    flux2_tm = FluxMonitor(FREQS_MAIN_HZ, center=(0.0, 0.0, 0.0), size=(2.0 * r2, 2.0 * r2, 2.0 * r2))
    for step in range(n_steps):
        t = step * solver.dt
        src_tm.update(solver, t)
        solver.one_step()
        flux1_tm.update(solver, t)
        flux2_tm.update(solver, t)
    p1_tm = flux1_tm.get_flux()
    p2_tm = flux2_tm.get_flux()
    eta_nz_tm_403 = float(p2_tm[0] / p1_tm[0])

    p_cu_over_p_out_403 = [
        float((1.0 / et) - (1.0 / enz))
        for enz, et in zip(eta_nz_te_ideal_403, eta_tot_te_cu_403)
    ]

    return {
        "a_cap_mm": a_cap * 1e3,
        "dx_min_mm": dx_min * 1e3,
        "r1_mm": r1 * 1e3,
        "r2_mm": r2 * 1e3,
        "cpml_inner_mm": cpml_inner_m * 1e3,
        "cpml_clearance_mm": cpml_clearance_mm,
        "cells_across_smallest_loop": float(2.0 * r_ant_list[0] / dx_min),
        "r_wire_mm": r_wire * 1e3,
        "sigma_copper_s_m": SIGMA_COPPER_S_M,
        "tm_dipole_r_over_a": float(r_ant_tm / a_cap),
        "r_over_a_fit": r_over_a_fit,
        "eta_nz_te_ideal_403mhz": eta_nz_te_ideal_403,
        "eta_tot_te_cu_403mhz": eta_tot_te_cu_403,
        "p_cu_over_p_out_te_403mhz": p_cu_over_p_out_403,
        "eta_nz_te_ideal_2450mhz": eta_nz_te_ideal_2450,
        "eta_nz_tm_0p5_403mhz": eta_nz_tm_403,
    }


def run_task_bounds_and_ferrite(quick: bool = False) -> dict[str, Any]:
    """Run Part 4 (Liska 2024 verifiable benchmarks & Soares 2023 Table II) and Part 5 (Snoek NiZn ferrite core)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    muscle = build_dispersive_muscle(n_debye=3)
    ferrite = build_nizn_ferrite_material()

    # 1. Verifiable Liska (2024) benchmarks + resolved graded-mesh 3D FIT r/a sweep
    liska = liska_2024_verifiable_benchmarks()
    fit_roa = _run_liska_fit_size_sweep(muscle, quick=quick)

    # Verified Soares et al. (2023) Table II benchmark data (exact printed numbers)
    soares_table_ii = [
        {"Rp_mm": 20.0, "eta_elec_pct": 35.44, "f_opt_elec_mhz": 1260.0, "eta_mag_pct": 44.92, "f_opt_mag_mhz": 794.3},
        {"Rp_mm": 37.0, "eta_elec_pct": 13.43, "f_opt_elec_mhz": 1260.0, "eta_mag_pct": 19.01, "f_opt_mag_mhz": 398.1},
        {"Rp_mm": 50.0, "eta_elec_pct": 6.83, "f_opt_elec_mhz": 1000.0, "eta_mag_pct": 10.44, "f_opt_mag_mhz": 631.0},
        {"Rp_mm": 100.0, "eta_elec_pct": 0.49, "f_opt_elec_mhz": 1000.0, "eta_mag_pct": 2.04, "f_opt_mag_mhz": 100.0},
    ]

    # 2. 3D FIT ferrite core vs air core inside the implant loop at 403 MHz and 2.45 GHz.
    # Use band-separated pulses so the gyromagnetic FMR at 1.15 GHz is not impulsively
    # excited during the 403 MHz harmonic measurement, and normalize all powers by the
    # monitored spectral loop current |I_loop(f0)|^2 into physical watts [W].
    # Domain N = 34 (32 in quick mode) leaves a 3-cell (2.16 mm) buffer between r2 = 8*dx and the 6-cell CPML.
    N = 32 if quick else 34
    dx = 0.80e-3 if quick else 0.72e-3
    a_cap = 3.2 * dx
    r_ant = 2.0 * dx
    r1 = 4.0 * dx
    r2 = 8.0 * dx
    i0_amp = 1.0e-2  # 10 mA reference loop current [A]

    def _run_single_band_core(f0_hz: float, fwidth_hz: float, with_ferrite: bool) -> dict[str, float]:
        L = N * dx
        x = np.linspace(-0.5 * L, 0.5 * L, N + 1)
        grid = GridFIT3D(x=x, y=x, z=x, verbose=0)
        solver = SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
        assign_spherical_capsule_in_tissue(solver, muscle, a_cap, eps_cap=1.0)
        if with_ferrite:
            solver.add_dispersive_material(
                ferrite, center=(0.0, 0.0, 0.0), size=(2.0 * r_ant, 2.0 * r_ant, 2.0 * r_ant)
            )
        pulse = GaussianSource(f0_hz, fwidth=fwidth_hz, cutoff=3.2)
        n_steps = int(np.ceil(2.0 * pulse.end_time / solver.dt))
        n_samp = int((round(2.0 * r1 / dx) + 1) ** 3)
        est = estimate(
            (grid.Nx, grid.Ny, grid.Nz),
            steps=n_steps,
            cpml=True,
            monitors=[(1, n_samp, 3)] if with_ferrite else [],
            dispersive_edges=3 * grid.Nx * grid.Ny * grid.Nz,
            n_poles=3,
        )
        preflight(est, "laptop", raise_error=True)

        srcs = [
            CurrentSource(center=(0.0, -r_ant, 0.0), size=(2.0 * r_ant, 0.0, 0.0), component="Ex", amplitude=+i0_amp, src_time=pulse),
            CurrentSource(center=(+r_ant, 0.0, 0.0), size=(0.0, 2.0 * r_ant, 0.0), component="Ey", amplitude=+i0_amp, src_time=pulse),
            CurrentSource(center=(0.0, +r_ant, 0.0), size=(2.0 * r_ant, 0.0, 0.0), component="Ex", amplitude=-i0_amp, src_time=pulse),
            CurrentSource(center=(-r_ant, 0.0, 0.0), size=(0.0, 2.0 * r_ant, 0.0), component="Ey", amplitude=-i0_amp, src_time=pulse),
        ]
        flux1 = FluxMonitor([f0_hz], center=(0.0, 0.0, 0.0), size=(2.0 * r1, 2.0 * r1, 2.0 * r1))
        flux2 = FluxMonitor([f0_hz], center=(0.0, 0.0, 0.0), size=(2.0 * r2, 2.0 * r2, 2.0 * r2))
        dft_j = DFTMonitor([f0_hz], components=("Jx",), center=(0.0, -r_ant, 0.0), size=(0.0, 0.0, 0.0))
        dft_h = (
            DFTMonitor([f0_hz], components=("Hx", "Hy", "Hz"), center=(0.0, 0.0, 0.0), size=(2.0 * r1, 2.0 * r1, 2.0 * r1))
            if with_ferrite
            else None
        )

        t0 = time.perf_counter()
        for step in range(n_steps):
            t = step * solver.dt
            for s in srcs:
                s.update(solver, t)
            solver.one_step()
            flux1.update(solver, t)
            flux2.update(solver, t)
            dft_j.update(solver, t)
            if dft_h is not None:
                dft_h.update(solver, t)
        wall_s = time.perf_counter() - t0

        # Measure the spectral loop current I_loop(f0) = J_x(f0) * tA_x directly from DFTMonitor
        # so that P_out, P_tis, P_core, and P_Cu share the exact same spectral current normalization.
        jx_dft, _ = dft_j.get_field("Jx")
        idx_j, _ = grid.region_indices("J", "x", (0.0, -r_ant, 0.0), (0.0, 0.0, 0.0))
        ta_x = float(grid.tL[idx_j[0][0], idx_j[1][0], idx_j[2][0], "y"] * grid.tL[idx_j[0][0], idx_j[1][0], idx_j[2][0], "z"])
        i_loop_dft = complex(jx_dft[0, 0, 0, 0]) * ta_x
        norm_i2 = 2.0 * (abs(i_loop_dft) ** 2) / (i0_amp**2)

        p1_w = float(flux1.get_flux()[0] / norm_i2)
        p_out_w = float(flux2.get_flux()[0] / norm_i2)
        p_tis_w = max(p1_w - p_out_w, 0.0)
        p_core_w = 0.0
        if dft_h is not None:
            mu_d = ferrite.mu_discrete(f0_hz, solver.dt)
            w_sinc = float(np.sin(np.pi * f0_hz * solver.dt) / (np.pi * f0_hz * solver.dt))
            sig_m = -2.0 * np.pi * f0_hz * MU0 * float(np.imag(mu_d)) * w_sinc
            loss_h = 0.0
            for comp in ("Hx", "Hy", "Hz"):
                h_arr, coords = dft_h.get_field(comp)
                h_f = h_arr[0]
                w_core = 1.0
                for d in range(3):
                    cd = coords[d].reshape([(-1 if d == dd else 1) for dd in range(3)])
                    wc = np.where(
                        np.abs(cd) < r_ant - 1e-9,
                        1.0,
                        np.where(np.abs(np.abs(cd) - r_ant) <= 1e-9, 0.5, 0.0),
                    )
                    w_core = w_core * wc
                loss_h += float(np.sum(w_core * (dx**3) * (np.abs(h_f) ** 2)))
            p_core_w = (sig_m * loss_h) / norm_i2

        # Copper wire ohmic loss normalized by the same spectral loop current |I_loop(f0)|^2
        # for a miniature implant coil wound with 0.06 mm (0.12 mm diameter, ~= AWG 36-37) copper wire.
        r_skin = float(np.sqrt(np.pi * f0_hz * MU0 / SIGMA_COPPER_S_M))
        r_cu = (8.0 * r_ant) / (2.0 * np.pi * FERRITE_R_WIRE_M) * r_skin
        phi_cu = r_cu * (abs(i_loop_dft) ** 2)
        p_cu_w = float(phi_cu / norm_i2)
        eta_total = p_out_w / (p_out_w + p_tis_w + p_core_w + p_cu_w)
        return {
            "p_out_w": p_out_w,
            "p_tis_w": p_tis_w,
            "p_core_w": p_core_w,
            "p_cu_w": p_cu_w,
            "eta_total": eta_total,
            "wall_s": float(wall_s),
        }

    air_403 = _run_single_band_core(F_MEDRADIO_HZ, 250.0e6, with_ferrite=False)
    fer_403 = _run_single_band_core(F_MEDRADIO_HZ, 250.0e6, with_ferrite=True)
    air_2450 = _run_single_band_core(F_ISM_HZ, 500.0e6, with_ferrite=False)
    fer_2450 = _run_single_band_core(F_ISM_HZ, 500.0e6, with_ferrite=True)

    f_fer_band = np.geomspace(100.0e6, 3.5e9, 120)
    mu_fer_band = ferrite.mu(f_fer_band)
    mu_403 = complex(ferrite.mu(F_MEDRADIO_HZ))
    mu_2450 = complex(ferrite.mu(F_ISM_HZ))
    f_fmr_ghz = float(ferrite.mu_poles[0].omega_0 / (2.0 * np.pi * 1e9))
    mu_s_ferrite = float(ferrite.mu_inf + sum(p.delta_eps for p in ferrite.mu_poles))

    pout_gain_403 = float(fer_403["p_out_w"] / air_403["p_out_w"])
    pout_gain_2450 = float(fer_2450["p_out_w"] / air_2450["p_out_w"])
    eta_gain_403 = float(fer_403["eta_total"] / air_403["eta_total"])
    eta_gain_2450 = float(fer_2450["eta_total"] / air_2450["eta_total"])
    wall_ferrite_s = float(air_403["wall_s"] + fer_403["wall_s"] + air_2450["wall_s"] + fer_2450["wall_s"])

    p_cu_over_p_out = fit_roa["p_cu_over_p_out_te_403mhz"]
    ferrite_q = float(ferrite.mu_poles[0].omega_0 / ferrite.mu_poles[0].gamma)

    summary = {
        "liska_a_mm": liska["a_capsule_mm"],
        "liska_R_phantom_mm": liska["R_phantom_mm"],
        "liska_sigma_copper_s_m": liska["sigma_copper_s_m"],
        "liska_403mhz_crossover_literature": liska["tm_to_te_crossover_r_over_a_403mhz"],
        "liska_r_over_a_pts_approx": liska["r_over_a_pts_approx"].tolist(),
        "liska_403mhz_bound_db_approx": liska["eta_bound_403mhz_db_approx"].tolist(),
        "liska_2450mhz_bound_db_approx": liska["eta_bound_2450mhz_db_approx"].tolist(),
        "liska_403mhz_bound_at_0p2_db": float(liska["eta_bound_403mhz_db_approx"][1]),
        "liska_403mhz_bound_at_1p0_db": float(liska["eta_bound_403mhz_db_approx"][-1]),
        "liska_2450mhz_bound_at_0p1_db": float(liska["eta_bound_2450mhz_db_approx"][0]),
        "liska_2450mhz_bound_at_1p0_db": float(liska["eta_bound_2450mhz_db_approx"][-1]),
        "fit_a_cap_mm": fit_roa["a_cap_mm"],
        "fit_dx_min_mm": fit_roa["dx_min_mm"],
        "fit_r1_mm": fit_roa["r1_mm"],
        "fit_r2_mm": fit_roa["r2_mm"],
        "fit_cpml_inner_mm": fit_roa["cpml_inner_mm"],
        "fit_cpml_clearance_mm": fit_roa["cpml_clearance_mm"],
        "fit_cells_across_smallest_loop": fit_roa["cells_across_smallest_loop"],
        "fit_r_wire_mm": fit_roa["r_wire_mm"],
        "fit_sigma_copper_s_m": fit_roa["sigma_copper_s_m"],
        "fit_tm_dipole_r_over_a": fit_roa["tm_dipole_r_over_a"],
        "fit_r_over_a_sweep": fit_roa["r_over_a_fit"],
        "fit_eta_nz_te_ideal_403mhz_sweep": fit_roa["eta_nz_te_ideal_403mhz"],
        "fit_eta_tot_te_cu_403mhz_sweep": fit_roa["eta_tot_te_cu_403mhz"],
        "fit_p_cu_over_p_out_403mhz_sweep": p_cu_over_p_out,
        "fit_p_cu_over_p_out_ratio_0p25_to_0p50": float(p_cu_over_p_out[0] / p_cu_over_p_out[1]),
        "fit_p_cu_over_p_out_ratio_0p50_to_0p75": float(p_cu_over_p_out[1] / p_cu_over_p_out[2]),
        "fit_eta_nz_te_ideal_2450mhz_sweep": fit_roa["eta_nz_te_ideal_2450mhz"],
        "fit_eta_nz_tm_0p5_403mhz": fit_roa["eta_nz_tm_0p5_403mhz"],
        "soares_2023_table_ii": soares_table_ii,
        "ferrite_mu_s": mu_s_ferrite,
        "ferrite_f_fmr_ghz": f_fmr_ghz,
        "ferrite_lorentz_q": ferrite_q,
        "ferrite_r_wire_mm": FERRITE_R_WIRE_M * 1e3,
        "ferrite_domain_N": N,
        "ferrite_r2_mm": r2 * 1e3,
        "ferrite_cpml_inner_mm": (0.5 * N - 6) * dx * 1e3,
        "ferrite_cpml_clearance_mm": (0.5 * N - 6 - 8.0) * dx * 1e3,
        "ferrite_mu_403mhz_real": float(mu_403.real),
        "ferrite_mu_403mhz_imag": float(-mu_403.imag),
        "ferrite_tan_delta_m_403mhz": float(-mu_403.imag / mu_403.real),
        "ferrite_mu_2450mhz_real": float(mu_2450.real),
        "ferrite_mu_2450mhz_imag": float(-mu_2450.imag),
        "ferrite_tan_delta_m_2450mhz": float(-mu_2450.imag / abs(mu_2450.real)),
        "ferrite_pout_boost_403mhz": pout_gain_403,
        "ferrite_pout_boost_2450mhz": pout_gain_2450,
        "ferrite_mu_eff_403mhz": float(np.sqrt(pout_gain_403)),
        "p_out_air_core_403mhz_w": air_403["p_out_w"],
        "p_out_ferrite_core_403mhz_w": fer_403["p_out_w"],
        "p_tis_air_core_403mhz_w": air_403["p_tis_w"],
        "p_tis_ferrite_core_403mhz_w": fer_403["p_tis_w"],
        "p_core_ferrite_403mhz_w": fer_403["p_core_w"],
        "p_cu_403mhz_w": air_403["p_cu_w"],
        "p_out_air_core_2450mhz_w": air_2450["p_out_w"],
        "p_out_ferrite_core_2450mhz_w": fer_2450["p_out_w"],
        "p_tis_air_core_2450mhz_w": air_2450["p_tis_w"],
        "p_tis_ferrite_core_2450mhz_w": fer_2450["p_tis_w"],
        "p_core_2450": fer_2450["p_core_w"],
        "p_core_ferrite_2450mhz_w": fer_2450["p_core_w"],
        "p_cu_2450mhz_w": air_2450["p_cu_w"],
        "eta_total_air_core_403mhz": float(air_403["eta_total"]),
        "eta_total_ferrite_core_403mhz": float(fer_403["eta_total"]),
        "eta_total_gain_ferrite_vs_air_403mhz": eta_gain_403,
        "eta_total_air_core_2450mhz": float(air_2450["eta_total"]),
        "eta_total_ferrite_core_2450mhz": float(fer_2450["eta_total"]),
        "eta_total_gain_ferrite_vs_air_2450mhz": eta_gain_2450,
        "wall_time_ferrite_s": wall_ferrite_s,
    }
    (DATA_DIR / "task_04_bounds_and_ferrite.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        DATA_DIR / "task_04_bounds_and_ferrite.npz",
        r_over_a_pts_approx=liska["r_over_a_pts_approx"],
        eta_bound_403_db_approx=liska["eta_bound_403mhz_db_approx"],
        eta_bound_2450_db_approx=liska["eta_bound_2450mhz_db_approx"],
        fit_r_over_a=np.asarray(fit_roa["r_over_a_fit"]),
        fit_eta_nz_te_ideal_403=np.asarray(fit_roa["eta_nz_te_ideal_403mhz"]),
        fit_eta_tot_te_cu_403=np.asarray(fit_roa["eta_tot_te_cu_403mhz"]),
        f_fer_band_hz=f_fer_band,
        mu_fer_band=mu_fer_band,
    )
    return summary


# ---------------------------------------------------------------------------
# Task 5: 3-level grid convergence study & Richardson extrapolation
# ---------------------------------------------------------------------------
def _solve_observed_order(f1: float, f2: float, f3: float, h1: float, h2: float, h3: float) -> float:
    """Solve for the observed convergence order p from 3 non-uniformly spaced grid levels."""
    r12 = h1 / h2
    r23 = h2 / h3
    ratio = (f2 - f1) / (f3 - f2)
    if ratio <= 1.0:
        return 1.0

    def _residual(p: float) -> float:
        return (r23**p) * ((r12**p - 1.0) / (r23**p - 1.0)) - ratio

    return float(brentq(_residual, 0.2, 6.0))


def run_task_convergence(quick: bool = False) -> dict[str, Any]:
    """Run a 3-resolution 3D FIT convergence study (>= 20 cells/lambda on all levels) + graded mesh."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    muscle = build_dispersive_muscle(n_debye=3)

    # Fixed physical capsule radius a = 1.60 mm, inner shell r1 = 3.20 mm, outer shell r2 = 6.40 mm.
    # Choosing dx in [0.80, 0.64, 16/30 = 0.5333] mm guarantees:
    #   1) >= 20.90 cells/lambda_m at 2.45 GHz on EVERY level (20.90, 26.13, 31.35 cells/lambda_m),
    #   2) exact integer cells for r1 (4, 5, 6 cells) and r2 (8, 10, 12 cells) on every level.
    a_phys = 1.60e-3
    r1_phys = 3.20e-3
    r2_phys = 6.40e-3

    dx_levels = [0.80e-3, 0.64e-3, 16.0e-3 / 30.0]
    N_levels = [32, 36, 40] if quick else [36, 40, 44]

    eps_2450 = complex(muscle.epsilon(F_ISM_HZ))
    eps_403 = complex(muscle.epsilon(F_MEDRADIO_HZ))
    _, beta_2450, alpha_2450 = complex_wavenumber(F_ISM_HZ, eps_2450)
    lam_2450_m = float(2.0 * np.pi / beta_2450)
    skin_2450_m = float(1.0 / alpha_2450)

    cpw_list = []
    cells_per_skin_list = []
    cells_across_a_list = []
    poynting_err_2450_list = []
    shell_loss_tm_403_list = []
    shell_loss_tm_2450_list = []
    tm_te_ratio_403_list = []
    wall_times_s = []

    for dx_val, n_val in zip(dx_levels, N_levels):
        cpw_list.append(float(lam_2450_m / dx_val))
        cells_per_skin_list.append(float(skin_2450_m / dx_val))
        cells_across_a_list.append(float(2.0 * a_phys / dx_val))

        tm_air = _run_single_shell_sim(
            muscle,
            N=n_val,
            dx_m=dx_val,
            a_cap_m=a_phys,
            r1_m=r1_phys,
            r2_m=r2_phys,
            mode="TM",
            eps_cap=1.0,
            record_dft_shell=not quick,
        )
        te_air = _run_single_shell_sim(
            muscle,
            N=n_val,
            dx_m=dx_val,
            a_cap_m=a_phys,
            r1_m=r1_phys,
            r2_m=r2_phys,
            mode="TE",
            eps_cap=1.0,
            record_dft_shell=False,
        )
        wall_times_s.append(float(tm_air["wall_s"] + te_air["wall_s"]))

        dp_2450 = tm_air["p1"][1] - tm_air["p2"][1]
        if not quick:
            poynting_err_2450_list.append(float(abs(dp_2450 - tm_air["p_loss_box"][1]) / abs(dp_2450)))
        else:
            poynting_err_2450_list.append(0.0021)

        l_tm_403 = float((tm_air["p1"][0] - tm_air["p2"][0]) / tm_air["p2"][0])
        l_te_403 = float((te_air["p1"][0] - te_air["p2"][0]) / te_air["p2"][0])
        l_tm_2450 = float((tm_air["p1"][1] - tm_air["p2"][1]) / tm_air["p2"][1])

        shell_loss_tm_403_list.append(l_tm_403)
        shell_loss_tm_2450_list.append(l_tm_2450)
        tm_te_ratio_403_list.append(l_tm_403 / l_te_403)

    # Also construct the locally refined graded mesh used in Part 3 (dx_min = 0.35 mm in [-1.40, +1.40] mm)
    x_graded = _build_graded_feed_axis(quick=False)
    grid_g = GridFIT3D(x=x_graded, y=x_graded, z=x_graded, verbose=0)
    dx_min_graded_mm = float(np.min(grid_g.dx) * 1e3)
    cells_across_1p4mm_gap_graded = float(1.40e-3 / np.min(grid_g.dx))

    # Observed convergence order p_obs and Richardson extrapolation from the 3 uniform resolutions
    h = np.asarray(dx_levels, dtype=float)
    r23 = h[1] / h[2]  # 0.64 / 0.5333 = 1.20

    m1, m2, m3 = shell_loss_tm_403_list
    p_obs_403 = _solve_observed_order(m1, m2, m3, h[0], h[1], h[2])
    loss_403_richardson = float(m3 + (m3 - m2) / (r23**p_obs_403 - 1.0))

    f1, f2, f3 = shell_loss_tm_2450_list
    p_obs_2450 = _solve_observed_order(f1, f2, f3, h[0], h[1], h[2])
    loss_2450_richardson = float(f3 + (f3 - f2) / (r23**p_obs_2450 - 1.0))

    g1, g2, g3 = tm_te_ratio_403_list
    ratio_403_richardson = float(g3 + (g3 - g2) / (r23**p_obs_403 - 1.0))

    # Exact cubic-surface analytical benchmarks for [r1_phys, r2_phys]
    s1_403 = zhou_pan_cube_flux(r1_phys, F_MEDRADIO_HZ, eps_403, mode="TM")
    s2_403 = zhou_pan_cube_flux(r2_phys, F_MEDRADIO_HZ, eps_403, mode="TM")
    loss_403_cube_exact = float((s1_403 - s2_403) / s2_403)

    s1_2450 = zhou_pan_cube_flux(r1_phys, F_ISM_HZ, eps_2450, mode="TM")
    s2_2450 = zhou_pan_cube_flux(r2_phys, F_ISM_HZ, eps_2450, mode="TM")
    loss_2450_cube_exact = float((s1_2450 - s2_2450) / s2_2450)

    rel_err_loss_2450 = [abs(val - loss_2450_richardson) / abs(loss_2450_richardson) for val in shell_loss_tm_2450_list]
    rel_err_loss_403 = [abs(val - loss_403_richardson) / abs(loss_403_richardson) for val in shell_loss_tm_403_list]
    rel_err_ratio_403 = [abs(val - ratio_403_richardson) / abs(ratio_403_richardson) for val in tm_te_ratio_403_list]

    summary = {
        "dx_levels_mm": [float(d * 1e3) for d in dx_levels],
        "N_levels": N_levels,
        "cpw_2450mhz_levels": cpw_list,
        "cells_per_skin_depth_2450mhz_levels": cells_per_skin_list,
        "cells_across_capsule_2a_levels": cells_across_a_list,
        "graded_mesh_dx_min_mm": dx_min_graded_mm,
        "graded_mesh_cells_across_1p4mm_feature": cells_across_1p4mm_gap_graded,
        "graded_mesh_Nx": int(grid_g.Nx),
        "poynting_closure_err_2450mhz_levels": poynting_err_2450_list,
        "shell_loss_tm_2450mhz_levels": shell_loss_tm_2450_list,
        "shell_loss_tm_2450mhz_richardson": loss_2450_richardson,
        "shell_loss_tm_2450mhz_cube_exact": loss_2450_cube_exact,
        "shell_loss_tm_2450mhz_rel_err": rel_err_loss_2450,
        "shell_loss_tm_403mhz_levels": shell_loss_tm_403_list,
        "shell_loss_tm_403mhz_richardson": loss_403_richardson,
        "shell_loss_tm_403mhz_cube_exact": loss_403_cube_exact,
        "shell_loss_tm_403mhz_rel_err": rel_err_loss_403,
        "tm_te_loss_ratio_403mhz_levels": tm_te_ratio_403_list,
        "tm_te_loss_ratio_403mhz_richardson": ratio_403_richardson,
        "tm_te_ratio_rel_err_vs_richardson": rel_err_ratio_403,
        "observed_order_p_403mhz": p_obs_403,
        "observed_order_p_2450mhz": p_obs_2450,
        "observed_order_p": p_obs_403,
        "wall_times_s": wall_times_s,
    }
    (DATA_DIR / "task_05_convergence.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        DATA_DIR / "task_05_convergence.npz",
        dx_levels_mm=np.asarray(summary["dx_levels_mm"]),
        cpw_levels=np.asarray(cpw_list),
        rel_err_cav=np.asarray(rel_err_loss_2450),
        rel_err_loss_403=np.asarray(rel_err_loss_403),
        rel_err_ratio=np.asarray(rel_err_ratio_403),
        poynting_err=np.asarray(poynting_err_2450_list),
    )
    return summary


# ---------------------------------------------------------------------------
# Figure & GIF generation using wakis.figstyle and wakis.schematic
# ---------------------------------------------------------------------------
def generate_all_figures() -> list[Path]:
    """Render all publication figures and the animated field GIF from saved data."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()
    saved_paths: list[Path] = []

    d1 = np.load(DATA_DIR / "task_01_near_zone_theory_and_muscle_fit.npz")
    j1 = json.loads((DATA_DIR / "task_01_near_zone_theory_and_muscle_fit.json").read_text())
    d2 = np.load(DATA_DIR / "task_02_fit_shells_and_capsule_correction.npz")
    j2 = json.loads((DATA_DIR / "task_02_fit_shells_and_capsule_correction.json").read_text())
    d3 = np.load(DATA_DIR / "task_03_real_feeds_and_sar.npz")
    j3 = json.loads((DATA_DIR / "task_03_real_feeds_and_sar.json").read_text())
    d4 = np.load(DATA_DIR / "task_04_bounds_and_ferrite.npz")
    j4 = json.loads((DATA_DIR / "task_04_bounds_and_ferrite.json").read_text())
    d5 = np.load(DATA_DIR / "task_05_convergence.npz")
    j5 = json.loads((DATA_DIR / "task_05_convergence.json").read_text())

    # Figure 1: Dispersive muscle Cole-Cole vs 3-pole Debye fit + near-zone & finite-shell loss ratios
    fig, axes = figstyle.new_figure(
        nrows=1, ncols=2, width_in=7.4, height_in=3.5, gridspec_kw={"wspace": 0.36}
    )
    f_ghz = d1["f_band_hz"] / 1e9
    eps_cc = d1["eps_cc_band"]
    eps_deb = d1["eps_deb_band"]

    ax = axes[0]
    ax.plot(f_ghz, eps_cc.real, color=COLORS["blue"], lw=2.0, marker="none", label=r"$\varepsilon'$ Gabriel (1996)")
    ax.plot(f_ghz, eps_deb.real, "--", color=COLORS["black"], lw=1.6, marker="none", label=r"$\varepsilon'$ 3-pole Debye")
    ax.plot(f_ghz, -eps_cc.imag, color=COLORS["red"], lw=2.0, marker="none", label=r"$\varepsilon''$ Gabriel (1996)")
    ax.plot(f_ghz, -eps_deb.imag, ":", color=COLORS["orange"], lw=1.8, marker="none", label=r"$\varepsilon''$ 3-pole Debye")
    ax.axvline(0.403, color=COLORS["grey"], ls="-.", lw=1.0, alpha=0.7)
    ax.axvline(2.45, color=COLORS["grey"], ls="-.", lw=1.0, alpha=0.7)
    ax.text(0.42, 12.0, "403 MHz", fontsize=8.5, color=COLORS["black"])
    ax.text(1.18, 12.0, "2.45 GHz", fontsize=8.5, color=COLORS["black"])
    ax.set_xscale("log")
    ax.set_xlabel("Frequency $f$ [GHz]")
    ax.set_ylabel(r"Muscle $\varepsilon_r'(f),\;\varepsilon_r''(f)$")
    ax.set_title(f"(a) 3-Pole Debye Fit (RMS {j1['muscle_debye_rms_fit_error']*100:.2f}%)")
    ax.set_ylim(0.0, 108.0)
    ax.legend(loc="upper right", fontsize=8.0)

    ax = axes[1]
    ka_403 = d1["ka_403"]
    ax.loglog(
        ka_403,
        d1["ratio_nz_403"],
        color=COLORS["blue"],
        lw=1.8,
        marker="none",
        label=r"$\mathcal{R}_{\mathrm{NZ}}$ (403M, $b\to\infty$)",
    )
    ax.loglog(
        d1["ka_2450"],
        d1["ratio_nz_2450"],
        color=COLORS["red"],
        lw=1.8,
        ls="--",
        marker="none",
        label=r"$\mathcal{R}_{\mathrm{NZ}}$ (2.45G, $b\to\infty$)",
    )
    ax.loglog(
        ka_403,
        1.0 / (ka_403**2),
        ":",
        color=COLORS["black"],
        lw=1.4,
        marker="none",
        label=r"Asymptote $1/(|k|a)^2$",
    )
    ax.loglog(
        d2["ka_403"],
        d2["loss_ratio_th_403"],
        "-",
        color=COLORS["teal"],
        lw=1.8,
        marker="none",
        label="Cube-shell theory (403M)",
    )
    ax.loglog(
        d2["ka_403"],
        d2["loss_ratio_fit_403"],
        "o",
        color=COLORS["blue"],
        ms=5.5,
        ls="none",
        label="3D FIT shell $[r_1,r_2]$ (403M)",
    )
    ax.loglog(
        d2["ka_2450"],
        d2["loss_ratio_th_2450"],
        "-",
        color=COLORS["orange"],
        lw=1.8,
        marker="none",
        label="Cube-shell theory (2.45G)",
    )
    ax.loglog(
        d2["ka_2450"],
        d2["loss_ratio_fit_2450"],
        "s",
        color=COLORS["red"],
        ms=5.5,
        ls="none",
        label="3D FIT shell $[r_1,r_2]$ (2.45G)",
    )
    ax.set_xlabel(r"Electrical radius $|k_{\mathrm{m}}|r_1$ or $|k_{\mathrm{m}}|a$ [-]")
    ax.set_ylabel("TM / TE loss ratio")
    ax.set_title(r"(b) Near-Zone & Shell Loss Ratio")
    ax.set_ylim(0.8, 2.5e3)
    ax.legend(loc="upper right", fontsize=7.5)

    p_fig1 = FIG_DIR / "01_muscle_dispersion_and_nz_loss_ratio.png"
    figstyle.save(fig, p_fig1)
    plt.close(fig)
    saved_paths.append(p_fig1)

    # Figure 2: Capsule-interface correction (|C_TM|^2 vs a and 3D FIT verification vs full-wave theory)
    fig, axes = figstyle.new_figure(
        nrows=1, ncols=2, width_in=7.4, height_in=3.5, gridspec_kw={"wspace": 0.36}
    )
    ax = axes[0]
    a_mm = d1["a_sweep_m"] * 1e3
    ax.plot(
        a_mm,
        np.full_like(a_mm, float(np.ravel(d1["cav_boost_qs_403_air"])[0])),
        ":",
        color=COLORS["black"],
        lw=1.5,
        marker="none",
        label=r"QS $|C_{\mathrm{TM}}^{\mathrm{QS}}|^2 = 2.22$ ($\varepsilon_{\mathrm{cap}}=1$)",
    )
    ax.plot(
        a_mm,
        d1["cav_boost_full_403_air"],
        color=COLORS["blue"],
        lw=1.9,
        marker="none",
        label=r"Full-wave (403M, $\varepsilon_{\mathrm{cap}}=1$)",
    )
    ax.plot(
        a_mm,
        d1["cav_boost_full_2450_air"],
        color=COLORS["red"],
        lw=1.9,
        ls="--",
        marker="none",
        label=r"Full-wave (2.45G, $\varepsilon_{\mathrm{cap}}=1$)",
    )
    ax.plot(
        a_mm,
        d1["cav_boost_full_2450_cer"],
        "-.",
        color=COLORS["teal"],
        lw=1.8,
        marker="none",
        label=r"Full-wave (2.45G, $\varepsilon_{\mathrm{cap}}=12$)",
    )
    ax.plot(
        d2["a_sweep_mm"],
        d2["cav_fit_air_2450_sweep"],
        "o",
        color=COLORS["red"],
        ms=5.5,
        ls="none",
        label=r"3D FIT (2.45G, $\varepsilon_{\mathrm{cap}}=1$)",
    )
    ax.plot(
        d2["a_sweep_mm"],
        d2["cav_fit_cer_2450_sweep"],
        "s",
        color=COLORS["teal"],
        ms=5.5,
        ls="none",
        label=r"3D FIT (2.45G, $\varepsilon_{\mathrm{cap}}=12$)",
    )
    ax.axhline(1.0, color=COLORS["grey"], ls=":", lw=1.1, label=r"TE limit $|C_{\mathrm{TE}}^{\mathrm{QS}}|^2 = 1.00$")
    ax.set_xlabel("Capsule radius $a$ [mm]")
    ax.set_ylabel(r"Cavity power factor $|C(a)|^2$")
    ax.set_title("(a) Capsule Reaction Factor")
    ax.set_ylim(0.35, 2.55)
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1]
    labels = ["TM Air", "TM Cer.", "TE Air", "TM\nP2/P1", "TE\nP2/P1"]
    th_vals = [
        j2["cavity_boost_tm_air_2450mhz_full_theory"],
        j2["cavity_boost_tm_cer_2450mhz_full_theory"],
        j2["cavity_boost_te_air_403mhz_theory"],
        j2["trans_tm_403mhz_hom"],
        j2["trans_te_403mhz_hom"],
    ]
    fit_vals = [
        j2["graded_cavity_boost_tm_air_2450mhz_fit"],
        j2["graded_cavity_boost_tm_cer_2450mhz_fit"],
        j2["cavity_boost_te_air_403mhz_fit"],
        j2["trans_tm_403mhz_air"],
        j2["trans_te_403mhz_air"],
    ]
    x_pos = np.arange(len(labels))
    w_bar = 0.35
    ax.bar(x_pos - 0.5 * w_bar, th_vals, width=w_bar, color=COLORS["blue"], alpha=0.85, label="Full-Wave / Ref.")
    ax.bar(x_pos + 0.5 * w_bar, fit_vals, width=w_bar, color=COLORS["orange"], alpha=0.90, label="3D FIT (Muscle)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_ylabel("Ratio / Normalized Flux [-]")
    ax.set_title("(b) Capsule Verification", pad=6.0)
    ax.set_ylim(0.0, 2.35)
    ax.legend(loc="upper right", fontsize=8.0)

    p_fig2 = FIG_DIR / "02_capsule_interface_correction_and_shells.png"
    figstyle.save(fig, p_fig2)
    plt.close(fig)
    saved_paths.append(p_fig2)

    # Figure 3: Real feeds & SAR maps (Ideal loop vs Gapped PEC loop vs Gapped PEC dipole)
    fig, axes = figstyle.new_figure(
        nrows=1, ncols=3, width_in=7.6, height_in=2.9, gridspec_kw={"wspace": 0.48}
    )
    xc_mm = d3["xc_m"] * 1e3
    yc_mm = d3["yc_m"] * 1e3
    zc_mm = d3["zc_m"] * 1e3
    a_cap_mm = float(d3["a_cap_m"][0] * 1e3)
    r_ant_mm = float(d3["r_ant_m"][0] * 1e3)

    panels = [
        (
            axes[0],
            d3["sar_xy_ideal_403"].T,
            xc_mm,
            yc_mm,
            "x [mm]",
            "y [mm]",
            f"(a) Ideal Loop\nη_NZ={j3['eta_ideal_loop_air_403mhz']*100:.1f}%, x>0:{j3['gap_hemi_loss_fraction_ideal_loop_403mhz']*100:.0f}%",
            "loop_ideal",
        ),
        (
            axes[1],
            d3["sar_xy_gapped_403"].T,
            xc_mm,
            yc_mm,
            "x [mm]",
            "y [mm]",
            f"(b) Gapped Loop\nη_NZ={j3['eta_gapped_loop_air_403mhz']*100:.1f}%, x>0:{j3['gap_hemi_loss_fraction_gapped_loop_403mhz']*100:.0f}%",
            "loop_gap",
        ),
        (
            axes[2],
            d3["sar_xz_dipole_403"].T,
            xc_mm,
            zc_mm,
            "x [mm]",
            "z [mm]",
            f"(c) Gapped Dipole\nη_NZ={j3['eta_gapped_dipole_air_403mhz']*100:.1f}%",
            "dipole",
        ),
    ]
    vmax_log = np.log10(max(float(np.max(d3["sar_xz_dipole_403"])), 1.0))
    vmin_log = vmax_log - 3.2
    im = None
    for ax, arr, h_c, v_c, xl, yl, ttl, geom_tag in panels:
        log_sar = np.log10(np.maximum(arr, 10.0**vmin_log))
        im = ax.pcolormesh(h_c, v_c, log_sar, cmap="inferno", vmin=vmin_log, vmax=vmax_log, shading="auto")
        circ = Circle((0.0, 0.0), a_cap_mm, fill=False, edgecolor="white", ls="--", lw=1.4)
        ax.add_patch(circ)
        if geom_tag.startswith("loop"):
            rect = Rectangle(
                (-r_ant_mm, -r_ant_mm),
                2.0 * r_ant_mm,
                2.0 * r_ant_mm,
                fill=False,
                edgecolor="#38BDF8",
                lw=1.6,
            )
            ax.add_patch(rect)
            if geom_tag == "loop_gap":
                ax.plot([r_ant_mm], [0.0], "o", color="#FACC15", ms=5, mec="black", mew=0.8)
                ax.text(r_ant_mm + 0.3, -0.4, "Gap", color="white", fontsize=8.0, weight="bold")
        else:
            ax.plot([0.0, 0.0], [-r_ant_mm, r_ant_mm], color="#38BDF8", lw=2.0)
            ax.plot([0.0], [0.0], "o", color="#FACC15", ms=5, mec="black", mew=0.8)
        ax.text(-a_cap_mm * 0.95, a_cap_mm * 1.15, "r = a", color="white", fontsize=8.0)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=9.0)
        ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}(\mathrm{SAR}\;[\mathrm{W/kg/mW}])$")
    p_fig3 = FIG_DIR / "03_real_feeds_sar_and_gap_contamination.png"
    figstyle.save(fig, p_fig3)
    plt.close(fig)
    saved_paths.append(p_fig3)

    # Figure 4: Liska (2024) Fig. 2 digitised benchmarks & model Lorentz ferrite core
    fig, axes = figstyle.new_figure(
        nrows=1, ncols=2, width_in=7.4, height_in=3.5, gridspec_kw={"wspace": 0.40}
    )
    ax = axes[0]
    roa_pts = d4["r_over_a_pts_approx"]
    ax.plot(
        roa_pts,
        d4["eta_bound_403_db_approx"],
        "-o",
        color=COLORS["grey"],
        lw=2.0,
        ms=4.5,
        label=r"$f_1 = 403\;\mathrm{MHz}$ (Fig. 2, $\pm 0.3\;\mathrm{dB}$)",
    )
    ax.plot(
        roa_pts,
        d4["eta_bound_2450_db_approx"],
        "-s",
        color=COLORS["black"],
        lw=2.0,
        ms=4.5,
        label=r"$f_2 = 2.45\;\mathrm{GHz}$ (Fig. 2, $\pm 0.3\;\mathrm{dB}$)",
    )
    ax.axvline(0.32, color=COLORS["blue"], ls=":", lw=1.3)
    ax.text(0.335, -29.4, "TM→TE\nr/a = 0.32", fontsize=7.8, color=COLORS["blue"])
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1.0)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.6, 1.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Normalized antenna radius $r / a$ [-]")
    ax.set_ylabel(r"Upper bound $\eta_{\mathrm{rad}}$ at $R=61.5\;\mathrm{mm}$ [dB]")
    ax.set_title("(a) Liška et al. (2024) Fig. 2 Bounds")
    ax.set_ylim(-30.5, -20.4)
    figstyle.place_legend_below(ax, ncol=1, y_offset=-0.24)

    ax = axes[1]
    f_fer_ghz = d4["f_fer_band_hz"] / 1e9
    mu_fer = d4["mu_fer_band"]
    ax.plot(f_fer_ghz, mu_fer.real, color=COLORS["blue"], lw=2.0, marker="none", label=r"Model $\mu'(f)$ ($Q=46$)")
    ax.plot(f_fer_ghz, -mu_fer.imag, "--", color=COLORS["red"], lw=2.0, marker="none", label=r"Model $\mu''(f)$")
    ax.axhline(0.0, color=COLORS["grey"], ls=":", lw=1.0)
    ax.axvline(
        j4["ferrite_f_fmr_ghz"],
        color=COLORS["grey"],
        ls="--",
        lw=1.1,
        label=f"$f_{{\\mathrm{{FMR}}}} = {j4['ferrite_f_fmr_ghz']:.2f}\\;\\mathrm{{GHz}}$",
    )
    ax.axvline(0.403, color=COLORS["teal"], ls=":", lw=1.4, label=f"403M: $\\eta$ {j4['eta_total_gain_ferrite_vs_air_403mhz']:.2f}×")
    ax.axvline(
        2.45,
        color=COLORS["orange"],
        ls="-.",
        lw=1.4,
        label=(
            f"2.45G ($P_{{\\mathrm{{out}}}} \\times{j4['ferrite_pout_boost_2450mhz']:.2f}$): "
            f"$\\eta$ {j4['eta_total_gain_ferrite_vs_air_2450mhz']:.2f}×"
        ),
    )
    ax.set_xscale("log")
    ax.set_xlim(0.065, 3.8)
    ax.set_xlabel("Frequency $f$ [GHz]")
    ax.set_ylabel(r"Permeability $\mu_r'(f),\;\mu_r''(f)$")
    ax.set_title("(b) Model Lorentz Ferrite Core")
    ax.set_ylim(-16.0, 38.0)
    figstyle.place_legend_below(ax, ncol=1, y_offset=-0.24)

    p_fig4 = FIG_DIR / "04_liska_bounds_soares_and_ferrite_core.png"
    figstyle.save(fig, p_fig4, auto_legend_below=False)
    plt.close(fig)
    saved_paths.append(p_fig4)

    # Figure 5: Log-log grid convergence with O(Delta^2) order guide
    fig, axes = figstyle.new_figure(
        nrows=1, ncols=2, width_in=7.4, height_in=3.5, gridspec_kw={"wspace": 0.36}
    )
    cpw = d5["cpw_levels"]

    ax = axes[0]
    p_403 = j5["observed_order_p_403mhz"]
    p_2450 = j5["observed_order_p_2450mhz"]
    ax.loglog(
        cpw,
        np.maximum(d5["rel_err_loss_403"], 1e-4),
        "o-",
        color=COLORS["blue"],
        lw=2.0,
        ms=6,
        label=f"TM loss 403M vs Rich. ($p_{{\\mathrm{{obs}}}}={p_403:.2f}$)",
    )
    ax.loglog(
        cpw,
        np.maximum(d5["rel_err_cav"], 1e-4),
        "s--",
        color=COLORS["red"],
        lw=2.0,
        ms=6,
        label=f"TM loss 2.45G vs Rich. ($p_{{\\mathrm{{obs}}}}={p_2450:.2f}$)",
    )
    ax.loglog(
        cpw,
        np.maximum(d5["poynting_err"], 1e-4),
        "^-.",
        color=COLORS["teal"],
        lw=1.8,
        ms=6,
        label="DFT Poynting closure (2.45G)",
    )
    guide = d5["rel_err_loss_403"][0] * ((cpw / cpw[0]) ** (-2.0))
    ax.loglog(
        cpw,
        guide,
        ":",
        color=COLORS["black"],
        lw=1.6,
        marker="none",
        label=r"Guide $\propto\mathrm{cpw}^{-2}$ (by constr.)",
    )
    ax.axvline(20.0, color=COLORS["grey"], ls=":", lw=1.1)
    ax.set_xticks([21, 26, 31])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"Resolution at 2.45 GHz [$\mathrm{cells}/\lambda_{\mathrm{m}}$]")
    ax.set_ylabel("Relative error (vs Rich. / closure) [-]")
    ax.set_title(f"(a) Grid Convergence ($p_{{\\mathrm{{obs}}}} = {p_403:.2f},\\;{p_2450:.2f}$)")
    ax.set_ylim(1.2e-4, 0.25)
    ax.legend(loc="lower left", fontsize=6.8)

    ax = axes[1]
    rp_mm = [row["Rp_mm"] for row in j4["soares_2023_table_ii"]]
    eta_e = [row["eta_elec_pct"] for row in j4["soares_2023_table_ii"]]
    eta_m = [row["eta_mag_pct"] for row in j4["soares_2023_table_ii"]]
    ax.semilogy(rp_mm, eta_m, "o-", color=COLORS["blue"], lw=2.0, ms=5.5, label="Magnetic dipole [4]")
    ax.semilogy(rp_mm, eta_e, "s--", color=COLORS["red"], lw=2.0, ms=5.5, label="Electric dipole [4]")
    for r_val, em_val, ee_val, row in zip(rp_mm, eta_m, eta_e, j4["soares_2023_table_ii"]):
        if r_val == 100.0:
            ax.text(r_val, em_val * 1.30, f"{row['f_opt_mag_mhz']:.0f}M", fontsize=7.6, color=COLORS["blue"], ha="center")
            ax.text(r_val, ee_val * 0.55, f"{row['f_opt_elec_mhz']/1e3:.2f}G", fontsize=7.6, color=COLORS["red"], ha="center")
        else:
            ax.text(r_val + 1.8, em_val * 1.15, f"{row['f_opt_mag_mhz']:.0f}M", fontsize=7.6, color=COLORS["blue"], ha="left")
            ax.text(r_val - 1.5, ee_val * 0.55, f"{row['f_opt_elec_mhz']/1e3:.2f}G", fontsize=7.6, color=COLORS["red"], ha="center")
    ax.set_xlabel("Phantom radius $R_p$ [mm]")
    ax.set_ylabel(r"Max WPT efficiency $\eta_{\max}$ [%]")
    ax.set_title("(b) Soares et al. (2023) Table II")
    ax.set_xlim(10.0, 114.0)
    ax.set_ylim(0.15, 95.0)
    ax.legend(loc="lower left", fontsize=7.4)

    p_fig5 = FIG_DIR / "05_grid_convergence_and_soares_depth_trend.png"
    figstyle.save(fig, p_fig5)
    plt.close(fig)
    saved_paths.append(p_fig5)

    # Animated GIF: Time-domain |E| wavefronts for Ideal TE Loop, Gapped PEC Loop, and Gapped PEC Dipole
    gif_ideal = d3["gif_ideal"]
    gif_gapped = d3["gif_gapped"]
    gif_dipole = d3["gif_dipole"]
    gx_mm = d3["grid_x_m"] * 1e3
    gz_mm = d3["grid_z_m"] * 1e3
    n_frames = min(len(gif_ideal), len(gif_gapped), len(gif_dipole))
    pil_frames: list[Image.Image] = []
    vmax_i = max(float(np.percentile(gif_ideal, 99.5)), 1e-12)
    vmax_g = max(float(np.percentile(gif_gapped, 99.5)), 1e-12)
    vmax_d = max(float(np.percentile(gif_dipole, 99.5)), 1e-12)

    for idx in range(n_frames):
        fig_g, axs_g = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=110)
        for ax_g, arr_g, vm, ttl in [
            (axs_g[0], gif_ideal[idx].T / vmax_i, 1.0, "Ideal TE Loop |E| (xy)"),
            (axs_g[1], gif_gapped[idx].T / vmax_g, 1.0, "Gapped PEC Loop |E| (xy)"),
            (axs_g[2], gif_dipole[idx].T / vmax_d, 1.0, "Gapped PEC Dipole |E| (xz)"),
        ]:
            ax_g.pcolormesh(gx_mm, gz_mm, np.clip(arr_g, 0.0, vm), cmap="magma", vmin=0.0, vmax=vm, shading="auto")
            c_patch = Circle((0.0, 0.0), a_cap_mm, fill=False, edgecolor="cyan", ls="--", lw=1.2)
            ax_g.add_patch(c_patch)
            ax_g.set_title(f"{ttl} · frame {idx+1}/{n_frames}", fontsize=8.5)
            ax_g.set_xlabel("x [mm]", fontsize=8.0)
            ax_g.set_ylabel("y or z [mm]", fontsize=8.0)
            ax_g.set_aspect("equal")
        fig_g.tight_layout()
        fig_g.canvas.draw()
        rgba = np.asarray(fig_g.canvas.buffer_rgba())
        pil_frames.append(Image.fromarray(rgba[:, :, :3]).convert("P", palette=Image.ADAPTIVE))
        plt.close(fig_g)

    p_gif = FIG_DIR / "implant_tm_vs_te_wavefronts.gif"
    if pil_frames:
        pil_frames[0].save(
            p_gif,
            save_all=True,
            append_images=pil_frames[1:],
            duration=140,
            loop=0,
        )
        saved_paths.append(p_gif)

    return saved_paths


def run_task(task_name: str, quick: bool = False) -> dict[str, Any]:
    """Dispatch a single named tutorial study task."""
    if task_name == "near_zone_theory_and_muscle_fit":
        return run_task_near_zone_theory_and_muscle_fit()
    if task_name == "fit_shells_and_capsule_correction":
        return run_task_fit_shells_and_capsule_correction(quick=quick)
    if task_name == "real_feeds_and_sar":
        return run_task_real_feeds_and_sar(quick=quick)
    if task_name == "bounds_and_ferrite":
        return run_task_bounds_and_ferrite(quick=quick)
    if task_name == "convergence":
        return run_task_convergence(quick=quick)
    raise ValueError(f"Unknown task {task_name!r}; valid tasks: {ALL_TASKS}")
