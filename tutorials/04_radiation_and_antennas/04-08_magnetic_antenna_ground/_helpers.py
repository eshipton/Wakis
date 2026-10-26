# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Simulation, analytical, and plotting helpers for Tutorial 4.8.

Magnetic antennas on and near a ground plane: magnetic Lorentz/Debye ADE
dispersion, Heaviside magnetic conductivity and hesitivity invariance,
four-orientation PEC image theory (HMD, VMD, VED, HED), free-space to
half-space volume convention, equal-hesitivity Lorentz/Debye families,
horizontal-magnetic / vertical-dielectric PEC duality, conformal metal
strip versus permeable bar efficiency-bandwidth product (EBWP), finite
ground diffraction tilt, and 3-resolution FIT convergence.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0

from wakis import DFTMonitor, FluxMonitor, GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.dispersive import Debye, DispersiveMaterial, Lorentz
from wakis.estimate import estimate, preflight
from wakis.field_monitors import _flat_index
from wakis.near2far import Near2FarMonitor
from wakis.sources import ContinuousSource, CurrentSource, GaussianSource, MagneticCurrentSource
from wakis.templates import Box

ETA0 = float(np.sqrt(MU0 / EPS0))
ETA0_APPROX = float(120.0 * np.pi)
C_YOUSEFI_EXACT = float(6.0 * np.pi * ETA0)  # 6*pi*eta0 ~ 7101.22 Ohm
C_YOUSEFI_120PI = float(720.0 * (np.pi**2))  # 720*pi^2 ~ 7106.12 Ohm (for eta0 = 120*pi Ohm)

# Reference CoZrNb (CZN) ferromagnetic laminate parameters (Sebastian 2013 eq. 8-81)
CZN_CHI_DC = 39.0  # mu_DC = 40
CZN_F0_HZ = 1.5e9
CZN_ALPHA = 4.0 / 3.0  # 1 / alpha = 0.75
CZN_HM_OHM_M = float(2.0 * np.pi * CZN_F0_HZ * MU0 * CZN_CHI_DC / CZN_ALPHA)  # 3.464231e5 Ohm/m

ALL_TASKS = (
    "part0_magnetic_ade",
    "part1_image_theory",
    "part2_volume_convention",
    "part3_equal_hm",
    "part4_dual_pair",
    "part5_ebwp_and_convergence",
)

COLORS = {
    "blue": figstyle.OKABE_ITO[0],
    "red": figstyle.OKABE_ITO[1],
    "teal": figstyle.OKABE_ITO[2],
    "orange": figstyle.OKABE_ITO[3],
    "purple": figstyle.OKABE_ITO[4],
    "sky": figstyle.OKABE_ITO[5],
    "green": figstyle.OKABE_ITO[2],
    "grey": "#6B7280",
}


# ---------------------------------------------------------------------------
# 1. Exact analytical models (derived from first principles)
# ---------------------------------------------------------------------------
def hesitivity_lorentz(chi_dc: float, f0_hz: float, alpha: float) -> float:
    """Return peak Heaviside magnetic conductivity (hesitivity) for a Lorentz pole.

    For a single-pole Lorentz magnetic susceptibility in the ``exp(+j*w*t)``
    convention (Yousefi, Sebastian & Diaz, *IEEE TAP* 2016, eqs. 9b, 11),

    .. math::

        \\chi_{\\mathrm{Lor}}(\\omega)
        = \\frac{\\chi_{\\mathrm{DC}}}{1 + j\\alpha(\\omega/\\omega_0) - (\\omega/\\omega_0)^2},

    the magnetic conductivity :math:`\\sigma_m(\\omega) = \\omega\\mu_0\\mu''(\\omega)`
    attains its global maximum at :math:`\\omega = \\omega_0`:

    .. math::

        h_m = \\sigma_m(\\omega_0) = \\frac{\\omega_0 \\mu_0 \\chi_{\\mathrm{DC}}}{\\alpha}
        = \\frac{\\omega_0^2 \\mu_0 \\chi_{\\mathrm{DC}}}{\\gamma}.

    Parameters
    ----------
    chi_dc : float
        Static magnetic susceptibility :math:`\\chi_{\\mathrm{DC}} = \\mu_{\\mathrm{DC}} - 1` [dimensionless].
    f0_hz : float
        Ferromagnetic resonance frequency :math:`f_0 = \\omega_0 / (2\\pi)` [Hz].
    alpha : float
        Dimensionless Gilbert/Lorentz damping factor :math:`\\alpha = \\gamma / \\omega_0` [dimensionless].

    Returns
    -------
    float
        Hesitivity :math:`h_m` [Ohm/m].
    """
    if chi_dc <= 0.0 or f0_hz <= 0.0 or alpha <= 0.0:
        raise ValueError("chi_dc, f0_hz, and alpha must all be positive.")
    return float((2.0 * np.pi * f0_hz) * MU0 * chi_dc / alpha)


def hesitivity_debye(chi_dc: float, f0_hz: float) -> float:
    """Return asymptotic Heaviside magnetic conductivity (hesitivity) for a Debye pole.

    For a single-pole Debye magnetic susceptibility (Yousefi et al. 2016, eqs. 9a, 10),

    .. math::

        \\chi_{\\mathrm{Debye}}(\\omega) = \\frac{\\chi_{\\mathrm{DC}}}{1 + j(\\omega/\\omega_0)},

    the magnetic conductivity approaches its supremum as :math:`\\omega \\gg \\omega_0`,
    equal to twice its value at the relaxation frequency :math:`\\omega_0 = 1/\\tau`
    (Snoek's product scaled by :math:`\\mu_0`):

    .. math::

        h_m = 2\\,\\sigma_m(\\omega_0) = \\omega_0 \\mu_0 \\chi_{\\mathrm{DC}} = \\frac{\\mu_0 \\chi_{\\mathrm{DC}}}{\\tau}.

    Parameters
    ----------
    chi_dc : float
        Static magnetic susceptibility :math:`\\chi_{\\mathrm{DC}} = \\mu_{\\mathrm{DC}} - 1` [dimensionless].
    f0_hz : float
        Relaxation frequency :math:`f_0 = 1 / (2\\pi\\tau)` [Hz].

    Returns
    -------
    float
        Hesitivity :math:`h_m` [Ohm/m].
    """
    if chi_dc <= 0.0 or f0_hz <= 0.0:
        raise ValueError("chi_dc and f0_hz must be positive.")
    return float((2.0 * np.pi * f0_hz) * MU0 * chi_dc)


def magnetic_loss_function(mu_r: complex | np.ndarray) -> np.ndarray:
    """Evaluate the exact volume-equivalence magnetic loss function :math:`f(\\mu)`.

    Derived from the ratio of volumetric magnetic dissipation
    :math:`\\frac{1}{2}\\omega\\mu_0\\mu''|\\mathbf{H}|^2` to the squared magnitude
    of the induced magnetic polarization current density
    :math:`|\\mathbf{M}_p|^2 = \\omega^2\\mu_0^2|\\mu_r - 1|^2|\\mathbf{H}|^2`:

    .. math::

        f(\\mu) = \\frac{\\mu''(\\omega)}{|\\mu_r(\\omega) - 1|^2}
        = \\frac{-\\mathrm{Im}[\\mu_r(\\omega)]}{(\\mathrm{Re}[\\mu_r(\\omega)] - 1)^2 + (\\mathrm{Im}[\\mu_r(\\omega)])^2}
        = -\\mathrm{Im}\\!\\left(\\frac{1}{\\chi(\\omega)}\\right).

    For both single-pole Debye and single-pole Lorentz susceptibilities,
    :math:`f(\\mu) = \\omega\\mu_0 / h_m` holds identically at all frequencies.

    Parameters
    ----------
    mu_r : complex or np.ndarray
        Complex relative permeability :math:`\\mu_r = \\mu' - j\\mu''` [dimensionless].

    Returns
    -------
    np.ndarray
        Dimensionless magnetic loss function :math:`f(\\mu)`.
    """
    mu_arr = np.asarray(mu_r, dtype=np.complex128)
    chi = mu_arr - 1.0
    return -np.imag(mu_arr) / (np.abs(chi) ** 2)


def closed_form_md_efficiency(
    freq_hz: float | np.ndarray,
    hm_ohm_m: float,
    vol_m3: float,
    use_120pi: bool = False,
) -> np.ndarray:
    """Evaluate closed-form magneto-dielectric dipole radiation efficiency.

    Governing expression derived from exact magnetic dipole radiated power
    :math:`P_{\\mathrm{rad,free}} = k_0^2(I_m\\ell)^2 / (12\\pi\\eta_0)` and volumetric
    core dissipation :math:`P_{\\mathrm{loss}} = \\frac{1}{2}\\omega\\mu_0\\mu''|H_x|^2\\mathrm{Vol}`
    (Yousefi, Sebastian & Diaz, *IEEE TAP* 2016, eq. 19b):

    .. math::

        e_{\\mathrm{MD}}(\\omega)
        = \\frac{1}{1 + \\frac{6\\pi\\eta_0}{h_m \\, \\mathrm{Vol} \\, k_0^2}},

    where the exact wave-impedance constant is :math:`6\\pi\\eta_0 \\approx 7101.2\\,\\Omega`,
    and under the textbook :math:`\\eta_0 \\approx 120\\pi\\,\\Omega` substitution
    (:math:`P_{\\mathrm{rad,free}} \\approx (10\\,\\Omega/\\eta_0^2)(k_0 I_m\\ell)^2`),
    :math:`6\\pi(120\\pi) = 720\\pi^2 \\approx 7106.1\\,\\Omega` (quoted as 7106 in
    Yousefi et al. eq. 19b).

    Parameters
    ----------
    freq_hz : float or np.ndarray
        Operating frequency [Hz].
    hm_ohm_m : float
        Material hesitivity :math:`h_m` [Ohm/m].
    vol_m3 : float
        Effective permeable volume :math:`\\mathrm{Vol}` [m^3]. For a rod in free
        space, :math:`\\mathrm{Vol} = V_{\\mathrm{phys}}`; for a half-rod bisected by
        an infinite PEC ground plane, image symmetry yields
        :math:`\\mathrm{Vol} = 2\\,V_{\\mathrm{phys}} = V_{\\mathrm{full}}`.
    use_120pi : bool, optional
        If True, use :math:`720\\pi^2 \\approx 7106.12\\,\\Omega`; otherwise use
        exact :math:`6\\pi\\eta_0 \\approx 7101.22\\,\\Omega`. Default is False.

    Returns
    -------
    np.ndarray
        Linear radiation efficiency :math:`e_{\\mathrm{MD}} \\in (0, 1)` [dimensionless].
    """
    f = np.asarray(freq_hz, dtype=float)
    k0 = 2.0 * np.pi * f / C0
    c_const = C_YOUSEFI_120PI if use_120pi else C_YOUSEFI_EXACT
    ratio = c_const / (hm_ohm_m * vol_m3 * (k0**2))
    return 1.0 / (1.0 + ratio)


def rumsey_weeks_efficiency(
    freq_hz: float | np.ndarray,
    mu_r: complex | np.ndarray,
    vol_m3: float,
) -> np.ndarray:
    """Evaluate legacy low-loss Rumsey-Weeks / DeVore-Bohley efficiency formula.

    Governing expression (Yousefi et al. 2016, eq. 8), which replaces
    :math:`|\\mu_r - 1|^2` in the denominator of :math:`f(\\mu)` with
    :math:`(\\mu')^2` under the assumption :math:`\\mu' \\gg \\mu''` and :math:`\\mu' \\gg 1`:

    .. math::

        e_{\\mathrm{RW}}(\\omega)
        = \\frac{1}{1 + \\frac{\\mu''(\\omega)}{[\\mu'(\\omega)]^2}
        \\frac{6\\pi\\eta_0}{\\omega\\mu_0 \\, \\mathrm{Vol} \\, k_0^2}}.

    Parameters
    ----------
    freq_hz : float or np.ndarray
        Operating frequency [Hz].
    mu_r : complex or np.ndarray
        Complex relative permeability :math:`\\mu_r = \\mu' - j\\mu''` [dimensionless].
    vol_m3 : float
        Effective permeable volume :math:`\\mathrm{Vol}` [m^3].

    Returns
    -------
    np.ndarray
        Legacy predicted efficiency [dimensionless].
    """
    f = np.asarray(freq_hz, dtype=float)
    omega = 2.0 * np.pi * f
    k0 = omega / C0
    mu_arr = np.asarray(mu_r, dtype=np.complex128)
    f_rw = (-np.imag(mu_arr)) / (np.real(mu_arr) ** 2)
    hm_eff_rw = (omega * MU0) / np.maximum(f_rw, 1e-30)
    return 1.0 / (1.0 + C_YOUSEFI_EXACT / (hm_eff_rw * vol_m3 * (k0**2)))


_PEC_IMAGE_SIGN: dict[str, float] = {
    "HMD": 1.0,
    "HED": -1.0,
    "VED": 1.0,
    "VMD": -1.0,
}


def pec_image_resistance_ratio(
    h_over_lam: float | np.ndarray,
    dipole_type: str,
) -> np.ndarray:
    """Return exact half-space radiation resistance ratio :math:`R_{\\mathrm{rad}}(h)/R_{\\mathrm{free}}` over PEC.

    Derived by integrating the array factor :math:`2[1 + s\\cos(2k_0 h\\cos\\theta)]`
    against the elementary dipole power pattern over the upper hemisphere
    :math:`\\theta \\in [0, \\pi/2]` with :math:`\\xi = 2k_0 h = 4\\pi(h/\\lambda_0)`
    and spherical Bessel functions :math:`j_0(\\xi) = \\sin\\xi/\\xi`,
    :math:`j_1(\\xi)/\\xi = (\\sin\\xi - \\xi\\cos\\xi)/\\xi^3`:

    * **HMD** (:math:`\\mathbf{M} \\parallel \\mathrm{PEC}`, :math:`s = +1`):
      :math:`R/R_{\\mathrm{free}} = 1 + \\frac{3}{2}[j_0(\\xi) - j_1(\\xi)/\\xi] \\to 2 - \\frac{4}{5}(k_0 h)^2`.
    * **HED** (:math:`\\mathbf{J} \\parallel \\mathrm{PEC}`, :math:`s = -1`):
      :math:`R/R_{\\mathrm{free}} = 1 - \\frac{3}{2}[j_0(\\xi) - j_1(\\xi)/\\xi] \\to \\frac{4}{5}(k_0 h)^2`.
    * **VED** (:math:`\\mathbf{J} \\perp \\mathrm{PEC}`, :math:`s = +1`):
      :math:`R/R_{\\mathrm{free}} = 1 + 3\\,j_1(\\xi)/\\xi \\to 2 - \\frac{2}{5}(k_0 h)^2`.
    * **VMD** (:math:`\\mathbf{M} \\perp \\mathrm{PEC}`, :math:`s = -1`):
      :math:`R/R_{\\mathrm{free}} = 1 - 3\\,j_1(\\xi)/\\xi \\to \\frac{2}{5}(k_0 h)^2`.

    Parameters
    ----------
    h_over_lam : float or np.ndarray
        Normalized height :math:`h / \\lambda_0` above the PEC ground plane [dimensionless].
    dipole_type : {"HMD", "VMD", "VED", "HED"}
        Dipole orientation and current type.

    Returns
    -------
    np.ndarray
        Ratio :math:`R_{\\mathrm{rad}}(h) / R_{\\mathrm{free}}` [dimensionless].
    """
    h_lam = np.asarray(h_over_lam, dtype=float)
    xi = 4.0 * np.pi * h_lam
    # Safe series expansion for small xi
    small = np.abs(xi) < 1e-4
    xi_safe = np.where(small, 1.0, xi)
    j0 = np.where(small, 1.0 - (xi**2) / 6.0 + (xi**4) / 120.0, np.sin(xi_safe) / xi_safe)
    j1_over_xi = np.where(
        small,
        1.0 / 3.0 - (xi**2) / 30.0 + (xi**4) / 840.0,
        (np.sin(xi_safe) - xi_safe * np.cos(xi_safe)) / (xi_safe**3),
    )

    kind = dipole_type.upper()
    if kind not in _PEC_IMAGE_SIGN:
        raise ValueError(f"Unknown dipole_type {dipole_type!r}; expected HMD, VMD, VED, or HED.")
    s = float(_PEC_IMAGE_SIGN[kind])
    if kind in ("HMD", "HED"):
        return 1.0 + 1.5 * s * (j0 - j1_over_xi)
    return 1.0 + 3.0 * s * j1_over_xi


def pec_hmd_zenith_directivity(h_over_lam: float | np.ndarray) -> np.ndarray:
    """Return exact broadside zenith directivity :math:`D_\\infty(\\theta = 0^\\circ)` of an HMD over infinite PEC.

    At polar angle :math:`\\theta = 0^\\circ` (zenith), the source at :math:`+h` and
    its in-phase PEC image at :math:`-h` produce array factor :math:`2\\cos(k_0 h)`,
    multiplying the free-space broadside radiation intensity by :math:`4\\cos^2(k_0 h)`
    while total radiated power into the upper hemisphere is multiplied by
    :math:`F_{\\mathrm{HMD}}(k_0 h) = R_{\\mathrm{HMD}}(h)/R_{\\mathrm{free}}` (Balanis 2016,
    ch. 4, sec. 4.7–4.8):

    .. math::

        D_\\infty(\\theta = 0^\\circ; h)
        = \\frac{4\\cos^2(k_0 h)\\,D_{\\mathrm{free}}}{F_{\\mathrm{HMD}}(k_0 h)}
        = \\frac{6\\cos^2(2\\pi h / \\lambda_0)}{F_{\\mathrm{HMD}}(h/\\lambda_0)}
        \\;\\xrightarrow{h \\to 0}\\; 3.0.

    Parameters
    ----------
    h_over_lam : float or np.ndarray
        Normalized height :math:`h / \\lambda_0` above the infinite PEC plane [dimensionless].

    Returns
    -------
    np.ndarray
        Linear zenith directivity :math:`D_\\infty(0^\\circ)` [dimensionless].
    """
    h_lam = np.asarray(h_over_lam, dtype=float)
    k0_h = 2.0 * np.pi * h_lam
    return (6.0 * (np.cos(k0_h) ** 2)) / pec_image_resistance_ratio(h_lam, "HMD")


def slab_transfer_matrix_rt(
    freq_hz: float | np.ndarray,
    thickness_m: float,
    mu_r: complex | np.ndarray,
    eps_r: complex | np.ndarray = 1.0,
) -> dict[str, np.ndarray]:
    """Compute exact 1D normal-incidence reflection, transmission, and absorption of a magnetic slab.

    For a homogeneous slab of thickness :math:`d`, relative permeability
    :math:`\\mu_r(\\omega)`, and relative permittivity :math:`\\varepsilon_r(\\omega)`
    in free space (:math:`\\exp(+j\\omega t)` convention), the refractive index
    and normalized wave impedance are
    :math:`n = \\sqrt{\\varepsilon_r \\mu_r}` (chosen with :math:`\\mathrm{Im}(n) \\le 0`)
    and :math:`z = \\sqrt{\\mu_r / \\varepsilon_r}` (:math:`\\mathrm{Re}(z) \\ge 0`).
    The single-interface Fresnel reflection is :math:`r_{01} = (z - 1)/(z + 1)` and
    the round-trip propagation factor is :math:`p = \\exp(-j k_0 n d)`:

    .. math::

        R(\\omega) = \\frac{r_{01}(1 - p^2)}{1 - r_{01}^2 p^2}, \\qquad
        T(\\omega) = \\frac{(1 - r_{01}^2) p}{1 - r_{01}^2 p^2}, \\qquad
        A(\\omega) = 1 - |R(\\omega)|^2 - |T(\\omega)|^2.

    Parameters
    ----------
    freq_hz : float or np.ndarray
        Frequency [Hz].
    thickness_m : float
        Slab thickness :math:`d` [m].
    mu_r : complex or np.ndarray
        Complex relative permeability :math:`\\mu_r(\\omega)` [dimensionless].
    eps_r : complex or np.ndarray, optional
        Complex relative permittivity :math:`\\varepsilon_r(\\omega)` [dimensionless]. Default is 1.0.

    Returns
    -------
    dict of str to np.ndarray
        Complex ``R`` and ``T``, power coefficients ``R_pow``, ``T_pow``, and
        slab power absorption fraction ``A_pow``.
    """
    f = np.asarray(freq_hz, dtype=float)
    k0 = 2.0 * np.pi * f / C0
    mu_c = np.asarray(mu_r, dtype=np.complex128)
    eps_c = np.asarray(eps_r, dtype=np.complex128)
    n_c = np.sqrt(eps_c * mu_c)
    n_c = np.where(np.imag(n_c) > 0.0, -n_c, n_c)
    z_c = np.sqrt(mu_c / eps_c)
    z_c = np.where(np.real(z_c) < 0.0, -z_c, z_c)

    r01 = (z_c - 1.0) / (z_c + 1.0)
    p = np.exp(-1j * k0 * n_c * thickness_m)
    denom = 1.0 - (r01**2) * (p**2)
    r_tot = r01 * (1.0 - p**2) / denom
    t_tot = (1.0 - r01**2) * p / denom
    r_pow = np.abs(r_tot) ** 2
    t_pow = np.abs(t_tot) ** 2
    a_pow = 1.0 - r_pow - t_pow
    return {
        "R": r_tot,
        "T": t_tot,
        "R_pow": r_pow,
        "T_pow": t_pow,
        "A_pow": a_pow,
    }


def conformal_metal_vs_permeable_ebwp(
    freqs_hz: np.ndarray,
    l_m: float = 0.9906,
    w_m: float = 0.0800,
    t_m: float = 0.0254,
    vswr: float = 2.5,
) -> dict[str, np.ndarray]:
    """Compare conformal CZN permeable bar vs conformal copper strip in the same volume.

    Implements the circuit, Best-Yaghjian bandwidth, and EBWP comparison of
    Yousefi et al. (*IEEE TAP* 2016, Sec. V, eqs. 20-22) for a conformal bounding
    box of length :math:`\\ell = 39\"` (0.9906 m), width :math:`w = 3.15\"` (0.080 m),
    and thickness :math:`t = 1.0\"` (0.0254 m) over a PEC ground plane, alongside
    the Ikonen et al. (*IEEE TAP* 2006) resonant patch Q/efficiency trade-off.

    Parameters
    ----------
    freqs_hz : np.ndarray
        Frequency array [Hz].
    l_m : float, optional
        Antenna length [m]. Default is 0.9906 (39 inches).
    w_m : float, optional
        Antenna width [m]. Default is 0.0800 (3.15 inches).
    t_m : float, optional
        Conformal height/thickness above PEC [m]. Default is 0.0254 (1.0 inch).
    vswr : float, optional
        Target voltage standing-wave ratio for matched bandwidth. Default is 2.5.

    Returns
    -------
    dict of str to np.ndarray
        Impedance, radiation efficiency, Best-Yaghjian FBW, and EBWP arrays.
    """
    f = np.asarray(freqs_hz, dtype=float)
    omega = 2.0 * np.pi * f
    k0 = omega / C0
    lam0 = C0 / f

    gamma_max = (vswr - 1.0) / (vswr + 1.0)
    beta_by = (gamma_max**2) / (1.0 - gamma_max**2)

    # 1. Conformal metal strip:
    # - Air-substrate microstrip characteristic impedance Z0_ms ~ 68 Ohm for w/t = 3.15, eps_r = 1
    #   from Wheeler's wide-strip formula (H. A. Wheeler, IEEE Trans. MTT, vol. 13, no. 2, pp. 172-185,
    #   1965; Yousefi et al., IEEE TAP 2016, Sec. V).
    # - Image-cancelled strip radiation resistance R_rad = 200*(l/lam0)^2 * sin^2(k0*t) [Ohm]
    #   from Yousefi et al. (IEEE TAP 2016, eq. 21) and Balanis (Antenna Theory, 4th ed., Sec. 4.7.2).
    z0_ms = 68.0
    r_rad_strip = 200.0 * ((l_m / lam0) ** 2) * (np.sin(k0 * t_m) ** 2)
    sigma_cu = 5.8e7
    r_surf_cu = np.sqrt(np.pi * f * MU0 / sigma_cu)
    r_ohm_strip = (1.0 / 3.0) * r_surf_cu * (l_m / w_m)
    r_in_strip = r_rad_strip + r_ohm_strip
    eff_strip = r_rad_strip / np.maximum(r_in_strip, 1e-30)

    x_open = -2.0 * z0_ms / np.tan(0.5 * k0 * l_m)
    x_short = 2.0 * z0_ms * np.tan(0.5 * k0 * l_m)
    z_open = r_in_strip + 1j * x_open
    z_short = r_in_strip + 1j * x_short

    dz_open_dw = np.gradient(r_in_strip, omega) + 1j * np.gradient(x_open, omega)
    dz_short_dw = np.gradient(r_in_strip, omega) + 1j * np.gradient(x_short, omega)

    q_open = (omega / (2.0 * r_in_strip)) * np.abs(dz_open_dw)
    q_short = (omega / (2.0 * r_in_strip)) * np.abs(dz_short_dw)
    fbw_open = np.minimum(1.0, (2.0 * np.sqrt(beta_by)) / np.maximum(q_open, 1e-12))
    fbw_short = np.minimum(1.0, (2.0 * np.sqrt(beta_by)) / np.maximum(q_short, 1e-12))
    ebwp_open = eff_strip * fbw_open
    ebwp_short = eff_strip * fbw_short

    # 2. Conformal permeable bar (CZN laminate, chi_dc=39, f0=1.5 GHz, alpha=4/3) in same volume
    # Effective volume on PEC is 2 * (l_m * w_m * t_m); equivalent cylinder radius rho_eq = sqrt(2*w*t/pi)
    vol_phys = l_m * w_m * t_m
    vol_eff = 2.0 * vol_phys
    rho_eq = float(np.sqrt(vol_eff / (np.pi * l_m)))
    mat_czn = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[Lorentz(delta_eps=CZN_CHI_DC, omega_0=2.0 * np.pi * CZN_F0_HZ, gamma=CZN_ALPHA * 2.0 * np.pi * CZN_F0_HZ)],
    )
    mu_czn = mat_czn.mu(f)
    chi_czn = mu_czn - 1.0

    # Shunt magneto-dielectric analytical circuit elements (Yousefi eqs. 5b-5d, 6b-6c, single-turn N=1)
    log_term = float(np.log(l_m / (2.0 * rho_eq)))
    l_md = (MU0 * np.pi * l_m / 2.0) / log_term
    c_md = (EPS0 * np.pi * l_m / (6.0 * np.pi)) * log_term
    g_rad = ((k0 * l_m) ** 2) / C_YOUSEFI_EXACT
    g_mat = (l_m / (omega * MU0 * np.pi * (rho_eq**2))) * (-np.imag(mu_czn) / (np.abs(chi_czn) ** 2))
    b_mat = -(l_m / (omega * MU0 * np.pi * (rho_eq**2))) * (np.real(chi_czn) / (np.abs(chi_czn) ** 2))

    y_in_md = (g_rad + g_mat) + 1j * (b_mat + omega * c_md - 1.0 / (omega * l_md))
    z_in_md = 1.0 / y_in_md
    r_in_md = np.real(z_in_md)
    x_in_md = np.imag(z_in_md)
    r_rad_md = r_in_md * (g_rad / (g_rad + g_mat))
    eff_md = g_rad / (g_rad + g_mat)

    dz_md_dw = np.gradient(r_in_md, omega) + 1j * np.gradient(x_in_md, omega)
    q_md = (omega / (2.0 * np.maximum(r_in_md, 1e-12))) * np.abs(dz_md_dw)
    fbw_md = np.minimum(1.0, (2.0 * np.sqrt(beta_by)) / np.maximum(q_md, 1e-12))
    ebwp_md = eff_md * fbw_md

    # 3. Ikonen et al. (IEEE TAP 2006, Sec. III) resonant cavity patch under a magnetic substrate:
    # Representative unloaded radiation quality factor Q_rad = 35 for a low-profile TM10 microstrip
    # patch resonator on a thin magneto-dielectric substrate, with loaded Q_tot = 1 / (1/Q_rad + tan_delta_m).
    # Notice that in a resonant standing-wave cavity patch, magnetic loss tangent lowers Q_tot ONLY by
    # reducing radiation efficiency e_patch = Q_tot / Q_rad = 1 / (1 + Q_rad * tan_delta_m), leaving
    # the resonant EBWP = e_patch * (2*sqrt(beta)/Q_tot) = 2*sqrt(beta)/Q_rad independent of tan_delta_m!
    tan_d_sweep = np.geomspace(1e-3, 1.0, 41)
    q_rad_patch = 35.0
    q_tot_patch = 1.0 / (1.0 / q_rad_patch + tan_d_sweep)
    eff_patch = q_tot_patch / q_rad_patch
    fbw_patch = (2.0 * np.sqrt(beta_by)) / q_tot_patch
    ebwp_patch = eff_patch * fbw_patch

    return {
        "freqs_mhz": f * 1e-6,
        "r_rad_strip_ohm": r_rad_strip,
        "r_ohm_strip_ohm": r_ohm_strip,
        "r_in_strip_ohm": r_in_strip,
        "x_open_ohm": x_open,
        "x_short_ohm": x_short,
        "eff_strip": eff_strip,
        "fbw_open": fbw_open,
        "fbw_short": fbw_short,
        "ebwp_open": ebwp_open,
        "ebwp_short": ebwp_short,
        "r_in_md_ohm": r_in_md,
        "r_rad_md_ohm": r_rad_md,
        "x_in_md_ohm": x_in_md,
        "eff_md": eff_md,
        "fbw_md": fbw_md,
        "ebwp_md": ebwp_md,
        "ebwp_advantage_db": 10.0 * np.log10(ebwp_md / np.maximum(ebwp_open, 1e-30)),
        "tan_d_patch": tan_d_sweep,
        "eff_patch": eff_patch,
        "fbw_patch": fbw_patch,
        "ebwp_patch": ebwp_patch,
    }


# ---------------------------------------------------------------------------
# 2. Grid construction, memory estimation & preflight helpers
# ---------------------------------------------------------------------------
def _peak_rss_mb() -> float:
    """Return current process peak RSS in megabytes [MB]."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def make_graded_half_axis(
    n_core: int,
    dx_fine: float,
    dx_coarse: float,
    domain_max: float,
    grading: float = 1.30,
) -> np.ndarray:
    """Build a 1D non-negative grid axis with a strictly uniform core and graded outer region.

    Placing exact integer multiples of ``dx_fine`` inside the core region
    :math:`[0, n_{\\mathrm{core}}\\Delta_{\\mathrm{fine}}]` guarantees that all
    primary electric source loops satisfy discrete charge conservation
    (:math:`\\nabla_h \\cdot \\mathbf{J} = 0`) and all dual magnetic edges inside
    the permeable rod share uniform cell volume :math:`\\Delta_{\\mathrm{fine}}^3`.

    Parameters
    ----------
    n_core : int
        Number of uniform cells of width ``dx_fine`` from the origin.
    dx_fine : float
        Uniform cell size inside the core region [m].
    dx_coarse : float
        Maximum cell size in the outer radiation region [m].
    domain_max : float
        Minimum target outer coordinate [m].
    grading : float, optional
        Cell-to-cell growth factor outside the core. Default is 1.30.

    Returns
    -------
    np.ndarray
        Strictly increasing 1D node coordinates starting at 0.0 [m].
    """
    core = np.arange(0, n_core + 1, dtype=float) * dx_fine
    pos = float(core[-1])
    step = float(dx_fine)
    tail: list[float] = []
    while pos < domain_max - 1e-9:
        step = min(float(dx_coarse), step * grading)
        pos += step
        tail.append(pos)
    return np.concatenate([core, np.asarray(tail, dtype=float)])


def build_rod_grid(
    *,
    dx_fine_m: float = 0.0050,
    dx_coarse_m: float = 0.0280,
    n_core_x: int = 7,
    n_core_y: int = 4,
    n_core_z: int = 5,
    domain_x_m: float = 0.21,
    domain_y_m: float = 0.17,
    domain_z_m: float = 0.18,
    half_space: bool = True,
    grading: float = 1.30,
) -> GridFIT3D:
    """Construct a locally refined 3D FIT grid for a permeable rod on PEC or in free space.

    Parameters
    ----------
    dx_fine_m : float, optional
        Uniform cell width inside the permeable rod core [m]. Default is 5.0 mm.
    dx_coarse_m : float, optional
        Outer air cell width [m]. Default is 28.0 mm.
    n_core_x, n_core_y, n_core_z : int, optional
        Number of uniform core cells along +x, +y, and +z.
    domain_x_m, domain_y_m, domain_z_m : float, optional
        Half-domain extents [m].
    half_space : bool, optional
        If True, start ``z`` at ``z = 0`` (for a PEC or PMC ground plane at
        ``bc_low[2]``). If False, mirror ``z`` symmetrically around ``z = 0``.
    grading : float, optional
        Maximum cell growth ratio in the graded air buffer. Default is 1.30.

    Returns
    -------
    GridFIT3D
        Initialized 3D FIT grid.
    """
    x_pos = make_graded_half_axis(n_core_x, dx_fine_m, dx_coarse_m, domain_x_m, grading=grading)
    x_nodes = np.concatenate([-x_pos[:0:-1], x_pos])
    y_pos = make_graded_half_axis(n_core_y, dx_fine_m, dx_coarse_m, domain_y_m, grading=grading)
    y_nodes = np.concatenate([-y_pos[:0:-1], y_pos])
    z_pos = make_graded_half_axis(n_core_z, dx_fine_m, dx_coarse_m, domain_z_m, grading=grading)
    z_nodes = z_pos if half_space else np.concatenate([-z_pos[:0:-1], z_pos])

    return GridFIT3D(
        float(x_nodes[0]),
        float(x_nodes[-1]),
        float(y_nodes[0]),
        float(y_nodes[-1]),
        float(z_nodes[0]),
        float(z_nodes[-1]),
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        verbose=0,
    )


def preflight_simulation_grid(
    grid: GridFIT3D,
    n_steps: int,
    n_freqs: int = 7,
    dispersive_edges: int = 300,
    n_poles: int = 1,
) -> dict[str, Any]:
    """Run :func:`wakis.estimate.estimate` and :func:`wakis.estimate.preflight` on a 3D grid.

    Parameters
    ----------
    grid : GridFIT3D
        Constructed 3D FIT grid.
    n_steps : int
        Number of leapfrog time steps.
    n_freqs : int, optional
        Number of DFT monitor frequencies. Default is 7.
    dispersive_edges : int, optional
        Number of magnetic or electric ADE edges. Default is 300.
    n_poles : int, optional
        Number of dispersive poles per edge. Default is 1.

    Returns
    -------
    dict of str to Any
        Resource estimate dictionary verified against the ``"laptop"`` (2 GB)
        memguard cap.
    """
    nx, ny, nz = int(grid.Nx - 1), int(grid.Ny - 1), int(grid.Nz - 1)
    n_surf = 2 * (nx * ny + nx * nz + ny * nz)
    est = estimate(
        cells=(nx, ny, nz),
        steps=n_steps,
        cpml=True,
        monitors=[(n_freqs, n_surf, 6)],
        dispersive_edges=dispersive_edges,
        n_poles=n_poles,
    )
    preflight(est, "laptop", raise_error=True)
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_cells": int(est.cells),
        "n_steps": int(n_steps),
        "est_ram_mb": float(est.host_bytes / 1e6),
        "est_time_s": float(est.seconds),
    }


def _make_trapezoid_coil_sources(
    pulse: Any,
    *,
    l_coil: float = 0.048,
    w_rod: float = 0.024,
    t_rod: float = 0.024,
    z_center: float = 0.012,
    half_space_arch: bool = True,
) -> list[CurrentSource]:
    """Build a divergence-free solenoid sheet excitation with trapezoidal end-node weights along x.

    Weighting the end nodes at :math:`x = \\pm \\ell_{\\mathrm{coil}}/2` by :math:`1/2`
    ensures that the discrete sheet-current integral over :math:`x \\in [-\\ell_{\\mathrm{coil}}/2, +\\ell_{\\mathrm{coil}}/2]`
    equals :math:`\\ell_{\\mathrm{coil}} / \\Delta_{\\mathrm{fine}}` exactly at every grid resolution.
    """
    half_w = 0.5 * w_rod
    half_t = 0.5 * t_rod
    z_top = z_center + half_t
    z_bot = z_center - half_t

    def _trapz_x(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.where(np.isclose(np.abs(x), 0.5 * l_coil, atol=1e-6), 0.5, 1.0)

    srcs = [
        CurrentSource(
            pulse,
            component="y",
            center=(0.0, 0.0, z_top),
            size=(l_coil, w_rod, 0.0),
            amplitude=1.0,
            amp_func=_trapz_x,
        ),
        CurrentSource(
            pulse,
            component="z",
            center=(0.0, -half_w, z_center),
            size=(l_coil, 0.0, t_rod),
            amplitude=1.0,
            amp_func=_trapz_x,
        ),
        CurrentSource(
            pulse,
            component="z",
            center=(0.0, half_w, z_center),
            size=(l_coil, 0.0, t_rod),
            amplitude=-1.0,
            amp_func=_trapz_x,
        ),
    ]
    if not half_space_arch:
        srcs.insert(
            1,
            CurrentSource(
                pulse,
                component="y",
                center=(0.0, 0.0, z_bot),
                size=(l_coil, w_rod, 0.0),
                amplitude=-1.0,
                amp_func=_trapz_x,
            ),
        )
    return srcs


def _run_rod_simulation(
    *,
    grid: GridFIT3D,
    dx_fine: float,
    rod_center: tuple[float, float, float],
    rod_size: tuple[float, float, float],
    material: DispersiveMaterial | None,
    sources: list[Any],
    pulse: GaussianSource,
    freqs_hz: np.ndarray,
    bc_low: list[str],
    n_pml: int = 4,
    n2f_box_half: tuple[float, float, float] = (0.084, 0.064, 0.074),
    n_periods: float = 5.6,
    record_xz_slice: bool = False,
) -> dict[str, Any]:
    """Execute one 3D FIT permeable or dielectric rod simulation and extract power balance.

    Computes the exact discrete leapfrog source input work :math:`W_{\\mathrm{src}}(f)`,
    closed-surface Poynting flux :math:`P_{\\mathrm{rad,flux}}(f)`, far-field
    hemisphere/sphere integrated power :math:`P_{\\mathrm{rad,n2f}}(f)`, and
    time-centered ADE volumetric dissipation :math:`P_{\\mathrm{loss}}(f)`.
    """
    f0 = float(pulse.frequency)
    omegas = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    solver = SolverFIT3D(
        grid,
        bc_low=bc_low,
        bc_high=["cpml", "cpml", "cpml"],
        n_pml=n_pml,
        verbose=0,
    )
    is_dual_pmc = bc_low[2].lower() == "pmc" and float(grid.z[0]) < 0.0
    if is_dual_pmc:
        solver.imu[:, :, 0, "x"] = 0.0
        solver.imu[:, :, 0, "y"] = 0.0
        solver.ieps[:, :, 0, "z"] = 0.0

    is_mag = material is not None and material.has_magnetic
    is_elec = material is not None and material.has_electric
    if material is not None:
        solver.add_dispersive_material(material, center=rod_center, size=rod_size)
        if is_mag:
            idx_ade, w_ade = solver._box_edges(rod_center, rod_size, dual=True)
            const_d = material.mu_discrete(freqs_hz, solver.dt)
        else:
            idx_ade, w_ade = solver._box_edges(rod_center, rod_size, dual=False)
            const_d = material.epsilon_discrete(freqs_hz, solver.dt)
    else:
        idx_ade = np.zeros(0, dtype=int)
        w_ade = np.zeros(0, dtype=float)
        const_d = np.ones_like(freqs_hz, dtype=np.complex128)

    n_steps = int(np.ceil(n_periods / (f0 * solver.dt)))
    pf_info = preflight_simulation_grid(
        grid,
        n_steps=n_steps,
        n_freqs=len(freqs_hz),
        dispersive_edges=int(idx_ade.size),
    )

    x_ref = 0.5 * (grid.x[:-1] + grid.x[1:]) if is_dual_pmc else grid.x
    y_ref = 0.5 * (grid.y[:-1] + grid.y[1:]) if is_dual_pmc else grid.y
    z_ref = 0.5 * (grid.z[:-1] + grid.z[1:]) if is_dual_pmc else grid.z
    bx = float(x_ref[int(np.argmin(np.abs(x_ref - n2f_box_half[0])))])
    by = float(y_ref[int(np.argmin(np.abs(y_ref - n2f_box_half[1])))])
    bz = float(z_ref[int(np.argmin(np.abs(z_ref - n2f_box_half[2])))])
    has_ground = bc_low[2].lower() in ("pec", "pmc") and (abs(float(grid.z[0])) < 1e-12 or is_dual_pmc)
    if has_ground:
        n2f = Near2FarMonitor(
            center=(0.0, 0.0, 0.5 * bz),
            size=(2.0 * bx, 2.0 * by, bz),
            frequencies=freqs_hz,
            ground_plane="-z",
        )
    else:
        n2f = Near2FarMonitor(
            center=(0.0, 0.0, 0.0),
            size=(2.0 * bx, 2.0 * by, 2.0 * bz),
            frequencies=freqs_hz,
        )

    dual_flux_terms: list[dict[str, Any]] = []
    if is_dual_pmc:
        for a, face_center, sign in n2f.faces:
            face_size = n2f.size.copy()
            face_size[a] = 0.0
            b, c = (a + 1) % 3, (a + 2) % 3
            for e_comp, h_comp, term_sign in ((b, c, 1.0), (c, b, -1.0)):
                e_name, h_name = "xyz"[e_comp], "xyz"[h_comp]
                indices, coords = solver.grid.region_indices("H", h_name, face_center, face_size)
                ii, jj, kk = np.meshgrid(*indices, indexing="ij")
                above = [ii.copy(), jj.copy(), kk.copy()]
                above[a] = above[a] + 1
                da = solver.grid.tL[ii, jj, kk, h_name] * solver.grid.L[ii, jj, kk, e_name]
                lo = face_center - face_size / 2.0
                hi = face_center + face_size / 2.0
                tol = 1e-6 * dx_fine
                for d in (b, c):
                    on_edge = (np.abs(coords[d] - lo[d]) < tol) | (np.abs(coords[d] - hi[d]) < tol)
                    shape = [1, 1, 1]
                    shape[d] = -1
                    da = da * np.where(on_edge, 0.5, 1.0).reshape(shape)
                da_flat = da.ravel()
                dual_flux_terms.append(
                    {
                        "H": _flat_index(solver, h_name, ii, jj, kk).ravel(),
                        "E_below": _flat_index(solver, e_name, ii, jj, kk).ravel(),
                        "E_above": _flat_index(solver, e_name, *above).ravel(),
                        "weight": sign * term_sign * da_flat,
                        "E_acc": np.zeros((len(freqs_hz), da_flat.size), dtype=np.complex128),
                        "H_acc": np.zeros((len(freqs_hz), da_flat.size), dtype=np.complex128),
                    }
                )

    dft_xz = None
    if record_xz_slice:
        z_span = float(grid.z[-1] - grid.z[0])
        z_mid = float(0.5 * (grid.z[-1] + grid.z[0]))
        dft_xz = DFTMonitor(
            frequencies=[f0],
            components=("Ey", "Hx", "Hz"),
            center=(0.0, 0.0, z_mid),
            size=(float(grid.x[-1] - grid.x[0]), 0.0, z_span),
        )

    for s in sources:
        s._set_edges(solver)
        s.is_first_update = False
        if isinstance(s, MagneticCurrentSource) and solver.M is None:
            from wakis.field import Field

            solver.M = Field(solver.Nx, solver.Ny, solver.Nz, use_gpu=solver.use_gpu, dtype=solver.dtype)

    ade_mid_dft = np.zeros((len(freqs_hz), idx_ade.size), dtype=np.complex128)
    ade_uncentered_dft = np.zeros((len(freqs_hz), idx_ade.size), dtype=np.complex128)
    src_field_dft = [np.zeros((len(freqs_hz), s.flat_idx.size), dtype=np.complex128) for s in sources]
    src_curr_dft = [np.zeros((len(freqs_hz), s.flat_idx.size), dtype=np.complex128) for s in sources]

    t0_wall = time.perf_counter()
    for step in range(n_steps):
        t = step * solver.dt
        e_before = [solver.E.array[s.flat_idx].copy() for s in sources]
        h_before_src = [solver.H.array[s.flat_idx].copy() for s in sources]
        ade_before = solver.H.array[idx_ade].copy() if is_mag else (solver.E.array[idx_ade].copy() if is_elec else None)

        for s in sources:
            s.update(solver, t)
        solver.one_step()
        if not is_dual_pmc:
            n2f.update(solver, t)
        if dft_xz is not None:
            dft_xz.update(solver, t)

        ph_t = np.exp(-1j * omegas * t) * solver.dt
        ph_mid = np.exp(-1j * omegas * (t + 0.5 * solver.dt)) * solver.dt
        ph_e = np.exp(-1j * omegas * (t + solver.dt)) * solver.dt

        if is_dual_pmc:
            for term in dual_flux_terms:
                h_val = solver.H.array[term["H"]]
                e_val = 0.5 * (solver.E.array[term["E_below"]] + solver.E.array[term["E_above"]])
                term["H_acc"] += ph_mid[:, None] * h_val[None, :]
                term["E_acc"] += ph_e[:, None] * e_val[None, :]

        if is_mag and ade_before is not None:
            h_after = solver.H.array[idx_ade]
            ade_mid_dft += np.outer(ph_t, 0.5 * (ade_before + h_after))
            ade_uncentered_dft += np.outer(ph_mid, h_after)
        elif is_elec and ade_before is not None:
            e_after = solver.E.array[idx_ade]
            ade_mid_dft += np.outer(ph_mid, 0.5 * (ade_before + e_after))
            ade_uncentered_dft += np.outer(ph_t, e_after)

        for k, s in enumerate(sources):
            if isinstance(s, MagneticCurrentSource):
                a_t = float(pulse(t))
                h_mid_s = 0.5 * (h_before_src[k] + solver.H.array[s.flat_idx])
                src_field_dft[k] += np.outer(ph_t, h_mid_s)
                src_curr_dft[k] += np.outer(ph_t, s.weights * a_t)
            else:
                a_t = float(pulse(t + 0.5 * solver.dt))
                e_mid_s = 0.5 * (e_before[k] + solver.E.array[s.flat_idx])
                src_field_dft[k] += np.outer(ph_mid, e_mid_s)
                src_curr_dft[k] += np.outer(ph_mid, s.weights * a_t)

    wall_s = float(time.perf_counter() - t0_wall)

    if is_dual_pmc:
        p_rad_flux = np.zeros(len(freqs_hz), dtype=float)
        for term in dual_flux_terms:
            p_rad_flux += np.real(term["E_acc"] * np.conj(term["H_acc"])) @ term["weight"]
        p_rad_n2f = p_rad_flux.copy()
    else:
        p_rad_flux = np.asarray(n2f.get_flux(), dtype=float)
        p_rad_n2f = np.asarray(n2f.radiated_power(), dtype=float)
    w_tilde = np.sin(0.5 * omegas * solver.dt) / (0.5 * solver.dt)
    cell_vol = float(dx_fine**3)
    k0_arr = omegas / C0
    gnd_factor = 2.0 if has_ground else 1.0
    imp_factor = (1.0 / ETA0) if is_mag else ETA0

    if idx_ade.size > 0:
        c0_const = MU0 if is_mag else EPS0
        f2_mid = np.sum(w_ade[None, :] * (np.abs(ade_mid_dft) ** 2), axis=1) * cell_vol
        f2_unc = np.sum(w_ade[None, :] * (np.abs(ade_uncentered_dft) ** 2), axis=1) * cell_vol
        p_loss = w_tilde * c0_const * (-np.imag(const_d)) * f2_mid
        p_loss_uncentered = omegas * c0_const * (-np.imag(const_d)) * f2_unc
        comp_ade = idx_ade // solver.N
        ax_comp = 0 if rod_size[0] >= rod_size[2] else 2
        mask_ax = comp_ade == ax_comp
        v_phys_ax = float(np.sum(w_ade[mask_ax])) * cell_vol
        f_mean_abs2 = np.abs(np.sum(w_ade[None, mask_ax] * ade_mid_dft[:, mask_ax], axis=1) * cell_vol) ** 2
        eta_fill = f_mean_abs2 / np.maximum(v_phys_ax * f2_mid, 1e-30)
        # Algebraic identity / consistency check: eff_pol_dft equals closed_form_md_efficiency(Vol * eta_fill)
        # up to discrete w_tilde vs omega and mu_d vs mu, so eff_pol_dft <= e_MD follows from eta_fill <= 1.
        p_rad_pol_dft = (
            gnd_factor
            * (k0_arr**2)
            / (6.0 * np.pi)
            * imp_factor
            * ((w_tilde * c0_const * np.abs(const_d - 1.0)) ** 2)
            * f_mean_abs2
        )
        eff_pol_dft = p_rad_pol_dft / np.maximum(p_rad_pol_dft + p_loss, 1e-30)
    else:
        p_loss = np.zeros_like(freqs_hz, dtype=float)
        p_loss_uncentered = np.zeros_like(freqs_hz, dtype=float)
        eta_fill = np.ones_like(freqs_hz, dtype=float)
        p_rad_pol_dft = np.zeros_like(freqs_hz, dtype=float)
        eff_pol_dft = np.ones_like(freqs_hz, dtype=float)

    w_src = np.zeros_like(freqs_hz, dtype=float)
    w_src_uncentered = np.zeros_like(freqs_hz, dtype=float)
    for k in range(len(sources)):
        w_src += -np.real(np.sum(src_field_dft[k] * np.conj(src_curr_dft[k]), axis=1)) * cell_vol
        # Mutated uncentered source work (shifted by half a leapfrog step dt/2)
        phase_err = np.exp(0.5j * omegas * solver.dt)
        w_src_uncentered += -np.real(np.sum(src_field_dft[k] * phase_err[:, None] * np.conj(src_curr_dft[k]), axis=1)) * cell_vol

    eff_flux = p_rad_flux / np.maximum(p_rad_flux + p_loss, 1e-30)
    eff_n2f = p_rad_n2f / np.maximum(p_rad_n2f + p_loss, 1e-30)
    balance_rel_err = np.abs(w_src - (p_rad_flux + p_loss)) / np.maximum(np.abs(w_src), 1e-30)
    mutated_balance_rel_err = np.abs(w_src_uncentered - (p_rad_flux + p_loss_uncentered)) / np.maximum(
        np.abs(w_src_uncentered), 1e-30
    )

    out: dict[str, Any] = {
        "p_rad_flux": p_rad_flux,
        "p_rad_n2f": p_rad_n2f,
        "p_rad_pol_dft": p_rad_pol_dft,
        "p_loss": p_loss,
        "w_src": w_src,
        "eff_flux": eff_flux,
        "eff_n2f": eff_n2f,
        "eff_pol_dft": eff_pol_dft,
        "eff_flux_db": 10.0 * np.log10(np.maximum(eff_flux, 1e-12)),
        "eff_n2f_db": 10.0 * np.log10(np.maximum(eff_n2f, 1e-12)),
        "eff_pol_dft_db": 10.0 * np.log10(np.maximum(eff_pol_dft, 1e-12)),
        "eta_fill": eta_fill,
        "balance_rel_err": balance_rel_err,
        "mutated_balance_rel_err": mutated_balance_rel_err,
        "const_d": const_d,
        "dt": float(solver.dt),
        "wall_s": wall_s,
        "peak_rss_mb": _peak_rss_mb(),
        **pf_info,
    }
    if dft_xz is not None:
        ey_f, ey_c = dft_xz.get_field("Ey")
        hx_f, hx_c = dft_xz.get_field("Hx")
        hz_f, hz_c = dft_xz.get_field("Hz")
        out["xz_ey_real"] = np.asarray(np.real(ey_f[0, :, 0, :]), dtype=float)
        out["xz_ey_abs"] = np.asarray(np.abs(ey_f[0, :, 0, :]), dtype=float)
        out["xz_x_ey"] = np.asarray(ey_c[0], dtype=float)
        out["xz_z_ey"] = np.asarray(ey_c[2], dtype=float)
        out["xz_hx_real"] = np.asarray(np.real(hx_f[0, :, 0, :]), dtype=float)
        out["xz_hz_real"] = np.asarray(np.real(hz_f[0, :, 0, :]), dtype=float)
        out["xz_x_hx"] = np.asarray(hx_c[0], dtype=float)
        out["xz_z_hx"] = np.asarray(hx_c[2], dtype=float)
    return out


# ---------------------------------------------------------------------------
# 3. Simulation task runners (Parts 0 through 6)
# ---------------------------------------------------------------------------
def _run_1d_magnetic_slab_ade(
    material: DispersiveMaterial,
    freqs_hz: np.ndarray,
    quick: bool = False,
) -> dict[str, np.ndarray]:
    """Run a 3D FIT TEM plane-wave reflection/transmission and absorption test on a magnetic slab.

    Uses a 2 x 2 x Nz TEM waveguide with PEC at x-walls (Ex normal) and PMC at
    y-walls (Hy tangential) so that an exact plane wave (Ex, Hy) propagates
    along +z through a magnetic ADE slab of thickness ``d = n_slab * dz``.
    """
    dz = 0.0035 if quick else 0.0020
    dx = dz
    nz_half = 52 if quick else 70
    n_slab = 6 if quick else 8
    d_slab = n_slab * dz
    x = np.array([-dx, 0.0, dx], dtype=float)
    y = np.array([-dx, 0.0, dx], dtype=float)
    z = np.arange(-nz_half, nz_half + 1, dtype=float) * dz

    f0 = float(np.mean(freqs_hz))
    omegas = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    pulse = GaussianSource(frequency=f0, fwidth=0.50 * f0, cutoff=4.0)
    z_src = -int(0.55 * nz_half) * dz
    z_ref = -int(0.28 * nz_half) * dz
    z_trn = float((n_slab + int(0.22 * nz_half)) * dz)

    # 1) Reference vacuum run
    grid_vac = GridFIT3D(x[0], x[-1], y[0], y[-1], z[0], z[-1], x=x, y=y, z=z, verbose=0)
    solver_vac = SolverFIT3D(
        grid_vac,
        bc_low=["pec", "pmc", "cpml"],
        bc_high=["pec", "pmc", "cpml"],
        n_pml=8,
        verbose=0,
    )
    n_steps = int(np.ceil((pulse.end_time + (6.0 if quick else 8.0) / f0) / solver_vac.dt))
    preflight_simulation_grid(grid_vac, n_steps=n_steps, n_freqs=len(freqs_hz), dispersive_edges=0)

    src_vac = CurrentSource(pulse, component="x", center=(0.0, 0.0, z_src), size=(2.0 * dx, 2.0 * dx, 0.0), amplitude=1.0)
    iz_ref = int(np.argmin(np.abs(z[:-1] - z_ref)))
    iz_trn = int(np.argmin(np.abs(z[:-1] - z_trn)))
    e_inc_ref = np.zeros(len(freqs_hz), dtype=np.complex128)
    e_inc_trn = np.zeros(len(freqs_hz), dtype=np.complex128)

    for step in range(n_steps):
        t = step * solver_vac.dt
        src_vac.update(solver_vac, t)
        solver_vac.one_step()
        ph_e = np.exp(-1j * omegas * (t + solver_vac.dt)) * solver_vac.dt
        ex_mat = solver_vac.E.to_matrix("x")
        e_inc_ref += float(ex_mat[0, 1, iz_ref]) * ph_e
        e_inc_trn += float(ex_mat[0, 1, iz_trn]) * ph_e

    # 2) Magnetic dispersive slab run (slab occupies z in [0, d_slab], comprising n_slab Hy midpoint layers)
    grid_slab = GridFIT3D(x[0], x[-1], y[0], y[-1], z[0], z[-1], x=x, y=y, z=z, verbose=0)
    solver_slab = SolverFIT3D(
        grid_slab,
        bc_low=["pec", "pmc", "cpml"],
        bc_high=["pec", "pmc", "cpml"],
        n_pml=8,
        verbose=0,
    )
    slab_center = (0.0, 0.0, 0.5 * d_slab)
    slab_size = (4.0 * dx, 4.0 * dx, d_slab)
    solver_slab.add_dispersive_material(material, center=slab_center, size=slab_size)
    mu_d = material.mu_discrete(freqs_hz, solver_slab.dt)

    src_slab = CurrentSource(pulse, component="x", center=(0.0, 0.0, z_src), size=(2.0 * dx, 2.0 * dx, 0.0), amplitude=1.0)
    e_tot_ref = np.zeros(len(freqs_hz), dtype=np.complex128)
    e_tot_trn = np.zeros(len(freqs_hz), dtype=np.complex128)
    iz_slab = np.arange(nz_half, nz_half + n_slab)
    hy_mid_dft = np.zeros((len(freqs_hz), n_slab), dtype=np.complex128)

    for step in range(n_steps):
        t = step * solver_slab.dt
        h_bef = solver_slab.H.to_matrix("y")[0, 1, iz_slab].copy()
        src_slab.update(solver_slab, t)
        solver_slab.one_step()
        h_aft = solver_slab.H.to_matrix("y")[0, 1, iz_slab]
        ph_e = np.exp(-1j * omegas * (t + solver_slab.dt)) * solver_slab.dt
        ph_t = np.exp(-1j * omegas * t) * solver_slab.dt
        ex_mat = solver_slab.E.to_matrix("x")
        e_tot_ref += float(ex_mat[0, 1, iz_ref]) * ph_e
        e_tot_trn += float(ex_mat[0, 1, iz_trn]) * ph_e
        hy_mid_dft += np.outer(ph_t, 0.5 * (h_bef + h_aft))

    k0 = omegas / C0
    z_ref_actual = float(z[iz_ref])
    r_fit = ((e_tot_ref - e_inc_ref) / e_inc_ref) * np.exp(-2j * k0 * abs(z_ref_actual))
    t_fit = (e_tot_trn / e_inc_trn) * np.exp(-1j * k0 * d_slab)
    r_pow_fit = np.abs(r_fit) ** 2
    t_pow_fit = np.abs(t_fit) ** 2
    a_pow_fit = 1.0 - r_pow_fit - t_pow_fit

    # Volumetric magnetic ADE loss per unit area divided by incident plane-wave intensity |E_inc|^2 / eta0
    w_tilde = np.sin(0.5 * omegas * solver_slab.dt) / (0.5 * solver_slab.dt)
    s_inc = (np.abs(e_inc_ref) ** 2) / ETA0
    s_loss_ade = w_tilde * MU0 * (-np.imag(mu_d)) * np.sum(np.abs(hy_mid_dft) ** 2, axis=1) * dz
    a_ade_fit = s_loss_ade / np.maximum(s_inc, 1e-30)

    ana_d = slab_transfer_matrix_rt(freqs_hz, d_slab, mu_d)
    ana_c = slab_transfer_matrix_rt(freqs_hz, d_slab, material.mu(freqs_hz))
    return {
        "d_eff_m": np.array(d_slab),
        "r_abs_fit": np.abs(r_fit),
        "t_abs_fit": np.abs(t_fit),
        "a_pow_fit": a_pow_fit,
        "a_ade_fit": a_ade_fit,
        "a_flux_fit": a_pow_fit,
        "r_abs_ana": np.abs(ana_d["R"]),
        "t_abs_ana": np.abs(ana_d["T"]),
        "a_pow_ana": ana_d["A_pow"],
        "r_abs_cont": np.abs(ana_c["R"]),
        "t_abs_cont": np.abs(ana_c["T"]),
    }


def run_task_00_ade_and_hesitivity(quick: bool = False) -> dict[str, Any]:
    """Run Part 0: Magnetic Lorentz/Debye ADE slab benchmark, hesitivity identity, and 3D rod invariance."""
    t0 = time.perf_counter()
    freqs_disp = np.linspace(0.05e9, 3.00e9, 120)
    omegas_disp = 2.0 * np.pi * freqs_disp

    # 1. CZN Lorentz pole (chi_dc=39, f0=1.5 GHz, alpha=4/3 -> hm = 3.464231e5 Ohm/m)
    w0_czn = 2.0 * np.pi * CZN_F0_HZ
    gam_czn = CZN_ALPHA * w0_czn
    mat_lor = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[Lorentz(delta_eps=CZN_CHI_DC, omega_0=w0_czn, gamma=gam_czn)],
        name="CZN Lorentz laminate",
    )
    mu_lor = mat_lor.mu(freqs_disp)
    sigma_m_lor = omegas_disp * MU0 * (-np.imag(mu_lor))
    f_mu_lor = magnetic_loss_function(mu_lor)
    f_mu_lor_exact = (omegas_disp * MU0) / CZN_HM_OHM_M
    rel_err_fmu_lor = float(np.max(np.abs(f_mu_lor - f_mu_lor_exact) / f_mu_lor_exact))

    # 2. Reference Debye pole (a commercially available NiZn ferrite, chi_dc=75, f0=1.6 GHz -> hm = 9.47482e5 Ohm/m)
    debye_chi_dc = 75.0
    debye_f0_hz = 1.6e9
    debye_tau = 1.0 / (2.0 * np.pi * debye_f0_hz)
    debye_hm = hesitivity_debye(debye_chi_dc, debye_f0_hz)
    mat_deb = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[Debye(delta_eps=debye_chi_dc, tau=debye_tau)],
        name="a commercially available NiZn ferrite",
    )
    mu_deb = mat_deb.mu(freqs_disp)
    sigma_m_deb = omegas_disp * MU0 * (-np.imag(mu_deb))
    f_mu_deb = magnetic_loss_function(mu_deb)
    f_mu_deb_exact = (omegas_disp * MU0) / debye_hm
    rel_err_fmu_deb = float(np.max(np.abs(f_mu_deb - f_mu_deb_exact) / f_mu_deb_exact))

    # 3. 1D/quasi-1D FIT slab R/T/A verification for Lorentz-mu and Debye-mu
    freqs_slab = np.linspace(0.60e9, 2.20e9, 9 if quick else 13)
    slab_lor_pole = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[Lorentz(delta_eps=4.5, omega_0=2.0 * np.pi * 1.4e9, gamma=0.45 * 2.0 * np.pi * 1.4e9)],
    )
    slab_deb_pole = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[Debye(delta_eps=4.0, tau=1.0 / (2.0 * np.pi * 1.2e9))],
    )
    rt_lor = _run_1d_magnetic_slab_ade(slab_lor_pole, freqs_slab, quick=quick)
    rt_deb = _run_1d_magnetic_slab_ade(slab_deb_pole, freqs_slab, quick=quick)

    # 4. 3D FIT small permeable rod hesitivity invariance and energy balance test
    # Compare Pole A (CZN: chi=39, f0=1.5 GHz, alpha=4/3, hm=3.464e5),
    # Pole B (chi=26, f0=2.0 GHz, alpha=32/27, equal hm=3.464e5), and
    # Mutated Pole C (chi=39, f0=1.5 GHz, alpha=8/3, halved hm=1.732e5)
    # Unified physical rod geometry: l_rod = 72 mm, w_rod = 24 mm, t_rod = 24 mm on PEC
    f0_rod = 0.28e9
    freqs_rod = np.array([0.22e9, 0.28e9, 0.34e9], dtype=float)
    nh = 2 if quick else 3
    dx_fine = 0.012 / nh  # 6.0 mm (quick) or 4.0 mm (converged Medium level, 6 cells across 24 mm)
    dx_coarse = 0.0300 if quick else 0.0280
    grid_rod = build_rod_grid(
        dx_fine_m=dx_fine,
        dx_coarse_m=dx_coarse,
        n_core_x=3 * nh + 2,
        n_core_y=nh + 2,
        n_core_z=2 * nh + 2,
        domain_x_m=0.18 if quick else 0.21,
        domain_y_m=0.14 if quick else 0.17,
        domain_z_m=0.15 if quick else 0.18,
        half_space=True,
    )
    rod_c = (0.0, 0.0, 0.012)
    rod_s = (0.072, 0.024, 0.024)
    pulse_rod = GaussianSource(frequency=f0_rod, fwidth=0.42 * f0_rod)
    sources_rod = _make_trapezoid_coil_sources(
        pulse_rod, l_coil=0.048, w_rod=0.024, t_rod=0.024, z_center=0.012, half_space_arch=True
    )

    mat_a = mat_lor
    w0_b = 2.0 * np.pi * 2.0e9
    alpha_b = w0_b * MU0 * 26.0 / CZN_HM_OHM_M
    mat_b = DispersiveMaterial(mu_inf=1.0, mu_poles=[Lorentz(delta_eps=26.0, omega_0=w0_b, gamma=alpha_b * w0_b)])
    mat_c_mut = DispersiveMaterial(mu_inf=1.0, mu_poles=[Lorentz(delta_eps=CZN_CHI_DC, omega_0=w0_czn, gamma=2.0 * gam_czn)])

    res_a = _run_rod_simulation(
        grid=grid_rod,
        dx_fine=dx_fine,
        rod_center=rod_c,
        rod_size=rod_s,
        material=mat_a,
        sources=sources_rod,
        pulse=pulse_rod,
        freqs_hz=freqs_rod,
        bc_low=["cpml", "cpml", "pec"],
        n_periods=5.0 if quick else 5.6,
        record_xz_slice=True,
    )
    res_b = _run_rod_simulation(
        grid=grid_rod,
        dx_fine=dx_fine,
        rod_center=rod_c,
        rod_size=rod_s,
        material=mat_b,
        sources=sources_rod,
        pulse=pulse_rod,
        freqs_hz=freqs_rod,
        bc_low=["cpml", "cpml", "pec"],
        n_periods=5.0 if quick else 5.6,
    )
    res_c = _run_rod_simulation(
        grid=grid_rod,
        dx_fine=dx_fine,
        rod_center=rod_c,
        rod_size=rod_s,
        material=mat_c_mut,
        sources=sources_rod,
        pulse=pulse_rod,
        freqs_hz=freqs_rod,
        bc_low=["cpml", "cpml", "pec"],
        n_periods=5.0 if quick else 5.6,
    )

    # Loss-to-radiation ratio (1/e - 1) scales as 1/h_m -> doubling alpha (halving h_m) must double (1/e - 1)
    loss_ratio_a = (1.0 / res_a["eff_flux"]) - 1.0
    loss_ratio_b = (1.0 / res_b["eff_flux"]) - 1.0
    loss_ratio_c = (1.0 / res_c["eff_flux"]) - 1.0

    return {
        "quick": np.array(int(quick)),
        "wall_s": np.array(float(time.perf_counter() - t0)),
        "peak_rss_mb": np.array(_peak_rss_mb()),
        "czn_hm_ohm_m": np.array(CZN_HM_OHM_M),
        "debye_hm_ohm_m": np.array(debye_hm),
        "freqs_disp_ghz": freqs_disp * 1e-9,
        "mu_lor_real": np.real(mu_lor),
        "mu_lor_imag": -np.imag(mu_lor),
        "sigma_m_lor": sigma_m_lor,
        "f_mu_lor": f_mu_lor,
        "f_mu_lor_exact": f_mu_lor_exact,
        "rel_err_fmu_lor": np.array(rel_err_fmu_lor),
        "mu_deb_real": np.real(mu_deb),
        "mu_deb_imag": -np.imag(mu_deb),
        "sigma_m_deb": sigma_m_deb,
        "f_mu_deb": f_mu_deb,
        "f_mu_deb_exact": f_mu_deb_exact,
        "rel_err_fmu_deb": np.array(rel_err_fmu_deb),
        "freqs_slab_ghz": freqs_slab * 1e-9,
        "lor_r_fit": rt_lor["r_abs_fit"],
        "lor_t_fit": rt_lor["t_abs_fit"],
        "lor_a_fit": rt_lor["a_pow_fit"],
        "lor_a_ade": rt_lor["a_ade_fit"],
        "lor_r_ana": rt_lor["r_abs_ana"],
        "lor_t_ana": rt_lor["t_abs_ana"],
        "lor_a_ana": rt_lor["a_pow_ana"],
        "deb_r_fit": rt_deb["r_abs_fit"],
        "deb_t_fit": rt_deb["t_abs_fit"],
        "deb_a_fit": rt_deb["a_pow_fit"],
        "deb_a_ade": rt_deb["a_ade_fit"],
        "deb_r_ana": rt_deb["r_abs_ana"],
        "deb_t_ana": rt_deb["t_abs_ana"],
        "deb_a_ana": rt_deb["a_pow_ana"],
        "slab_lor_max_err": np.array(float(np.max(np.abs(rt_lor["r_abs_fit"] - rt_lor["r_abs_ana"])))),
        "slab_deb_max_err": np.array(float(np.max(np.abs(rt_deb["r_abs_fit"] - rt_deb["r_abs_ana"])))),
        "freqs_rod_ghz": freqs_rod * 1e-9,
        "dx_fine_m": np.array(dx_fine),
        "dx_coarse_m": np.array(dx_coarse),
        "x_nodes": np.asarray(grid_rod.x, dtype=float),
        "y_nodes": np.asarray(grid_rod.y, dtype=float),
        "z_nodes": np.asarray(grid_rod.z, dtype=float),
        "eff_pole_a": res_a["eff_flux"],
        "eff_pole_b": res_b["eff_flux"],
        "eff_pole_c_mut": res_c["eff_flux"],
        "eff_pole_a_db": res_a["eff_flux_db"],
        "eff_pole_b_db": res_b["eff_flux_db"],
        "eff_pole_c_mut_db": res_c["eff_flux_db"],
        "w_src_pole_a": res_a["w_src"],
        "p_rad_pole_a": res_a["p_rad_flux"],
        "p_loss_pole_a": res_a["p_loss"],
        "balance_rel_err_a": res_a["balance_rel_err"],
        "mutated_balance_rel_err_a": res_a["mutated_balance_rel_err"],
        "equal_hm_diff_db": np.abs(res_a["eff_flux_db"] - res_b["eff_flux_db"]),
        "mutated_hm_loss_ratio_factor": loss_ratio_c / loss_ratio_a,
        "xz_ey_real": res_a["xz_ey_real"],
        "xz_ey_abs": res_a["xz_ey_abs"],
        "xz_x_ey": res_a["xz_x_ey"],
        "xz_z_ey": res_a["xz_z_ey"],
        "xz_hx_real": res_a["xz_hx_real"],
        "xz_hz_real": res_a["xz_hz_real"],
        "xz_x_hx": res_a["xz_x_hx"],
        "xz_z_hx": res_a["xz_z_hx"],
    }


def run_task_01_image_theory(quick: bool = False) -> dict[str, Any]:
    """Run Part 1: 4-orientation dipole image theory over PEC (HMD, VMD, VED, HED) and time-domain GIF."""
    t0 = time.perf_counter()
    f0 = 1.0e9
    lam0 = C0 / f0  # 0.30 m
    k0 = 2.0 * np.pi / lam0
    dx = 0.0150 if quick else 0.0125  # 20 to 24 cells/lam0
    nx_half = 11 if quick else 13
    nz_max = 22 if quick else 26

    # Note: MagneticCurrentSource M_x sits at z midpoints (k+0.5)*dx; M_z sits at z nodes k*dx;
    # CurrentSource J_x sits at z nodes k*dx; J_z sits at z midpoints (k+0.5)*dx.
    x = np.arange(-nx_half, nx_half + 1, dtype=float) * dx
    y = np.arange(-nx_half, nx_half + 1, dtype=float) * dx
    z = np.arange(0, nz_max + 1, dtype=float) * dx
    grid = GridFIT3D(x[0], x[-1], y[0], y[-1], z[0], z[-1], x=x, y=y, z=z, verbose=0)

    # Indices along z for midpoint-centered sources (HMD, VED) and node-centered sources (VMD, HED)
    k_mid_list = [1, 2, 3, 5, 7, 10] if not quick else [1, 2, 4, 6, 9]
    k_node_list = [1, 2, 3, 5, 7, 10] if not quick else [1, 2, 4, 6, 9]

    h_mid_lam = (np.asarray(k_mid_list, dtype=float) + 0.5) * dx / lam0
    h_node_lam = np.asarray(k_node_list, dtype=float) * dx / lam0

    pulse = GaussianSource(frequency=f0, fwidth=0.40 * f0)
    sf = complex(pulse.fourier_transform(f0))
    # Free-space reference radiated flux for unit moment:
    # Electric dipole: (eta0 * k0^2 / (6*pi)) * |sf|^2
    # Magnetic dipole: (k0^2 / (6*pi*eta0)) * |sf|^2
    p_free_mag = float((k0**2) / (6.0 * np.pi * ETA0) * (abs(sf) ** 2))
    p_free_elec = float(ETA0 * (k0**2) / (6.0 * np.pi) * (abs(sf) ** 2))

    r_hmd_fit = np.zeros(len(k_mid_list), dtype=float)
    r_ved_fit = np.zeros(len(k_mid_list), dtype=float)
    r_vmd_fit = np.zeros(len(k_node_list), dtype=float)
    r_hed_fit = np.zeros(len(k_node_list), dtype=float)

    mon_w = 2.0 * (nx_half - 5) * dx
    mon_h = (nz_max - 6) * dx
    dt_est = 0.5 * dx / (np.sqrt(3.0) * C0)
    n_steps = int(np.ceil((pulse.end_time + 2.2 / f0) / dt_est))
    preflight_simulation_grid(grid, n_steps=n_steps, n_freqs=1, dispersive_edges=0)

    for idx, k_val in enumerate(k_mid_list):
        h_z = (k_val + 0.5) * dx
        # HMD (M_x || PEC)
        solver = SolverFIT3D(grid, bc_low=["cpml", "cpml", "pec"], bc_high=["cpml"] * 3, n_pml=5, verbose=0)
        src = MagneticCurrentSource(pulse, component="x", center=(0.0, 0.0, h_z), amplitude=1.0)
        n2f = Near2FarMonitor(center=(0.0, 0.0, 0.5 * mon_h), size=(mon_w, mon_w, mon_h), frequencies=[f0], ground_plane="-z")
        for step in range(n_steps):
            t = step * solver.dt
            src.update(solver, t)
            solver.one_step()
            n2f.update(solver, t)
        r_hmd_fit[idx] = float(n2f.radiated_power()[0]) / p_free_mag

        # VED (J_z _|_ PEC)
        solver = SolverFIT3D(grid, bc_low=["cpml", "cpml", "pec"], bc_high=["cpml"] * 3, n_pml=5, verbose=0)
        src_e = CurrentSource(pulse, component="z", center=(0.0, 0.0, h_z), amplitude=1.0)
        n2f_e = Near2FarMonitor(center=(0.0, 0.0, 0.5 * mon_h), size=(mon_w, mon_w, mon_h), frequencies=[f0], ground_plane="-z")
        for step in range(n_steps):
            t = step * solver.dt
            src_e.update(solver, t)
            solver.one_step()
            n2f_e.update(solver, t)
        r_ved_fit[idx] = float(n2f_e.radiated_power()[0]) / p_free_elec

    for idx, k_val in enumerate(k_node_list):
        h_z = k_val * dx
        # VMD (M_z _|_ PEC)
        solver = SolverFIT3D(grid, bc_low=["cpml", "cpml", "pec"], bc_high=["cpml"] * 3, n_pml=5, verbose=0)
        src = MagneticCurrentSource(pulse, component="z", center=(0.0, 0.0, h_z), amplitude=1.0)
        n2f = Near2FarMonitor(center=(0.0, 0.0, 0.5 * mon_h), size=(mon_w, mon_w, mon_h), frequencies=[f0], ground_plane="-z")
        for step in range(n_steps):
            t = step * solver.dt
            src.update(solver, t)
            solver.one_step()
            n2f.update(solver, t)
        r_vmd_fit[idx] = float(n2f.radiated_power()[0]) / p_free_mag

        # HED (J_x || PEC)
        solver = SolverFIT3D(grid, bc_low=["cpml", "cpml", "pec"], bc_high=["cpml"] * 3, n_pml=5, verbose=0)
        src_e = CurrentSource(pulse, component="x", center=(0.0, 0.0, h_z), amplitude=1.0)
        n2f_e = Near2FarMonitor(center=(0.0, 0.0, 0.5 * mon_h), size=(mon_w, mon_w, mon_h), frequencies=[f0], ground_plane="-z")
        for step in range(n_steps):
            t = step * solver.dt
            src_e.update(solver, t)
            solver.one_step()
            n2f_e.update(solver, t)
        r_hed_fit[idx] = float(n2f_e.radiated_power()[0]) / p_free_elec

    # Time-domain CW wave frames in the yz-plane (x = 0) comparing HMD (M_x at h = 0.05*lam0)
    # vs HED (J_x at h = 0.05*lam0) over PEC ground plane for the animated GIF!
    cw = ContinuousSource(frequency=f0, ramp_time=1.0 / f0)
    n_steps_gif = int(np.ceil(3.4 / (f0 * solver.dt)))
    frame_steps = set(np.linspace(int(1.2 / (f0 * solver.dt)), n_steps_gif - 1, 24, dtype=int).tolist())
    ix_mid = int(np.argmin(np.abs(x)))

    solver_hmd_cw = SolverFIT3D(grid, bc_low=["cpml", "cpml", "pec"], bc_high=["cpml"] * 3, n_pml=5, verbose=0)
    src_hmd_cw = MagneticCurrentSource(cw, component="x", center=(0.0, 0.0, 0.5 * dx), amplitude=ETA0)
    solver_hed_cw = SolverFIT3D(grid, bc_low=["cpml", "cpml", "pec"], bc_high=["cpml"] * 3, n_pml=5, verbose=0)
    src_hed_cw = CurrentSource(cw, component="x", center=(0.0, 0.0, 1.0 * dx), amplitude=1.0)

    gif_hmd_frames: list[np.ndarray] = []
    gif_hed_frames: list[np.ndarray] = []
    gif_times_per: list[float] = []
    for step in range(n_steps_gif):
        t = step * solver_hmd_cw.dt
        src_hmd_cw.update(solver_hmd_cw, t)
        solver_hmd_cw.one_step()
        src_hed_cw.update(solver_hed_cw, t)
        solver_hed_cw.one_step()
        if step in frame_steps:
            # In yz-plane (x=0): HMD (M_x) radiates E_y/E_z and H_x; HED (J_x) radiates E_x
            hx_hmd = np.asarray(solver_hmd_cw.H.to_matrix("x")[ix_mid, :-1, :-1], dtype=np.float32) * ETA0
            ex_hed = np.asarray(solver_hed_cw.E.to_matrix("x")[ix_mid, :-1, :-1], dtype=np.float32)
            gif_hmd_frames.append(hx_hmd.copy())
            gif_hed_frames.append(ex_hed.copy())
            gif_times_per.append(float((step + 1) * solver_hmd_cw.dt * f0))

    h_dense = np.linspace(0.0, 0.60, 200)
    r_hmd_ana = pec_image_resistance_ratio(h_mid_lam, "HMD")
    r_ved_ana = pec_image_resistance_ratio(h_mid_lam, "VED")
    r_vmd_ana = pec_image_resistance_ratio(h_node_lam, "VMD")
    r_hed_ana = pec_image_resistance_ratio(h_node_lam, "HED")

    return {
        "quick": np.array(int(quick)),
        "wall_s": np.array(float(time.perf_counter() - t0)),
        "peak_rss_mb": np.array(_peak_rss_mb()),
        "h_mid_lam": h_mid_lam,
        "h_node_lam": h_node_lam,
        "r_hmd_fit": r_hmd_fit,
        "r_ved_fit": r_ved_fit,
        "r_vmd_fit": r_vmd_fit,
        "r_hed_fit": r_hed_fit,
        "r_hmd_ana": r_hmd_ana,
        "r_ved_ana": r_ved_ana,
        "r_vmd_ana": r_vmd_ana,
        "r_hed_ana": r_hed_ana,
        "h_dense": h_dense,
        "r_hmd_dense": pec_image_resistance_ratio(h_dense, "HMD"),
        "r_ved_dense": pec_image_resistance_ratio(h_dense, "VED"),
        "r_vmd_dense": pec_image_resistance_ratio(h_dense, "VMD"),
        "r_hed_dense": pec_image_resistance_ratio(h_dense, "HED"),
        "gif_y_lam": 0.5 * (y[:-1] + y[1:]) / lam0,
        "gif_z_lam": 0.5 * (z[:-1] + z[1:]) / lam0,
        "gif_hmd_frames": np.asarray(gif_hmd_frames, dtype=np.float32),
        "gif_hed_frames": np.asarray(gif_hed_frames, dtype=np.float32),
        "gif_times_per": np.asarray(gif_times_per, dtype=float),
    }


def run_task_02_volume_convention(quick: bool = False) -> dict[str, Any]:
    """Run Part 2: Free-space vs half-space volume convention and height lift transition in 3D FIT."""
    t0 = time.perf_counter()
    f0 = 0.28e9
    freqs = np.linspace(0.16e9, 0.40e9, 5 if quick else 7)
    nh = 2 if quick else 3
    dx_fine = 0.012 / nh  # 6.0 mm (quick) or 4.0 mm (converged Medium level, 6 cells across 24 mm)
    dx_coarse = 0.0300 if quick else 0.0280
    ncx, ncy, ncz = 3 * nh + 2, nh + 2, 2 * nh + 2

    w0_czn = 2.0 * np.pi * CZN_F0_HZ
    gam_czn = CZN_ALPHA * w0_czn
    mat_czn = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[Lorentz(delta_eps=CZN_CHI_DC, omega_0=w0_czn, gamma=gam_czn)],
    )
    pulse = GaussianSource(frequency=f0, fwidth=0.45 * f0)

    # 1. Half-rod flush on PEC (z in [0, 24 mm], width 24 mm, length 72 mm -> V_phys = 4.1472e-5 m^3)
    grid_pec = build_rod_grid(
        dx_fine_m=dx_fine, dx_coarse_m=dx_coarse, n_core_x=ncx, n_core_y=ncy, n_core_z=ncz, half_space=True
    )
    rod_c_pec = (0.0, 0.0, 0.012)
    rod_s_pec = (0.072, 0.024, 0.024)
    sources_pec = _make_trapezoid_coil_sources(
        pulse, l_coil=0.048, w_rod=0.024, t_rod=0.024, z_center=0.012, half_space_arch=True
    )
    res_half_pec = _run_rod_simulation(
        grid=grid_pec,
        dx_fine=dx_fine,
        rod_center=rod_c_pec,
        rod_size=rod_s_pec,
        material=mat_czn,
        sources=sources_pec,
        pulse=pulse,
        freqs_hz=freqs,
        bc_low=["cpml", "cpml", "pec"],
        n_periods=5.0 if quick else 5.6,
    )

    # 2. Full mirrored rod in free space (z in [-24 mm, +24 mm] -> V_full = 2*V_phys = 8.2944e-5 m^3)
    grid_free = build_rod_grid(
        dx_fine_m=dx_fine, dx_coarse_m=dx_coarse, n_core_x=ncx, n_core_y=ncy, n_core_z=ncz, half_space=False
    )
    rod_c_full = (0.0, 0.0, 0.0)
    rod_s_full = (0.072, 0.024, 0.048)
    sources_full = _make_trapezoid_coil_sources(
        pulse, l_coil=0.048, w_rod=0.024, t_rod=0.048, z_center=0.0, half_space_arch=False
    )
    res_full_free = _run_rod_simulation(
        grid=grid_free,
        dx_fine=dx_fine,
        rod_center=rod_c_full,
        rod_size=rod_s_full,
        material=mat_czn,
        sources=sources_full,
        pulse=pulse,
        freqs_hz=freqs,
        bc_low=["cpml", "cpml", "cpml"],
        n_periods=5.0 if quick else 5.6,
    )

    # 3. Same physical rod (V_phys = 4.1472e-5 m^3) in FREE SPACE (no PEC ground plane, h -> infinity)
    sources_phys_free = _make_trapezoid_coil_sources(
        pulse, l_coil=0.048, w_rod=0.024, t_rod=0.024, z_center=0.012, half_space_arch=False
    )
    res_phys_free = _run_rod_simulation(
        grid=grid_free,
        dx_fine=dx_fine,
        rod_center=rod_c_pec,
        rod_size=rod_s_pec,
        material=mat_czn,
        sources=sources_phys_free,
        pulse=pulse,
        freqs_hz=freqs,
        bc_low=["cpml", "cpml", "cpml"],
        n_periods=5.0 if quick else 5.6,
    )

    # 4. Same physical rod lifted to heights h_cells * dx_fine above PEC at f0
    h_lift_cells = [1, 2, 4] if quick else [1, 3, 5, 8]
    eff_lift_f0 = [float(res_half_pec["eff_flux"][len(freqs) // 2])]
    h_center_m = [0.012]
    grid_lift = build_rod_grid(
        dx_fine_m=dx_fine,
        dx_coarse_m=dx_coarse,
        n_core_x=ncx,
        n_core_y=ncy,
        n_core_z=2 * nh + 10,
        domain_z_m=0.22,
        half_space=True,
    )
    for hc in h_lift_cells:
        zc = 0.012 + hc * dx_fine
        srcs_h = _make_trapezoid_coil_sources(
            pulse, l_coil=0.048, w_rod=0.024, t_rod=0.024, z_center=zc, half_space_arch=False
        )
        res_h = _run_rod_simulation(
            grid=grid_lift,
            dx_fine=dx_fine,
            rod_center=(0.0, 0.0, zc),
            rod_size=rod_s_pec,
            material=mat_czn,
            sources=srcs_h,
            pulse=pulse,
            freqs_hz=np.array([f0]),
            bc_low=["cpml", "cpml", "pec"],
            n2f_box_half=(0.085, 0.065, 0.12),
            n_periods=4.8 if quick else 5.4,
        )
        h_center_m.append(float(zc))
        eff_lift_f0.append(float(res_h["eff_flux"][0]))

    vol_phys = float(0.072 * 0.024 * 0.024)
    vol_full = float(2.0 * vol_phys)
    eff_ana_full = closed_form_md_efficiency(freqs, CZN_HM_OHM_M, vol_full)
    eff_ana_phys = closed_form_md_efficiency(freqs, CZN_HM_OHM_M, vol_phys)

    lam0_f0 = C0 / f0
    h_lift_lam = np.asarray(h_center_m, dtype=float) / lam0_f0
    f_hmd_lift = pec_image_resistance_ratio(h_lift_lam, "HMD")
    eff_ana_lift = closed_form_md_efficiency(f0, CZN_HM_OHM_M, vol_phys * f_hmd_lift)

    return {
        "quick": np.array(int(quick)),
        "wall_s": np.array(float(time.perf_counter() - t0)),
        "peak_rss_mb": np.array(_peak_rss_mb()),
        "freqs_ghz": freqs * 1e-9,
        "k0_l": (2.0 * np.pi * freqs / C0) * 0.072,
        "vol_phys_m3": np.array(vol_phys),
        "vol_full_m3": np.array(vol_full),
        "eff_half_pec": res_half_pec["eff_flux"],
        "eff_full_free": res_full_free["eff_flux"],
        "eff_phys_free": res_phys_free["eff_flux"],
        "eff_ana_full": eff_ana_full,
        "eff_ana_phys": eff_ana_phys,
        "p_rad_half_pec": res_half_pec["p_rad_flux"],
        "p_rad_full_free": res_full_free["p_rad_flux"],
        "p_loss_half_pec": res_half_pec["p_loss"],
        "p_loss_full_free": res_full_free["p_loss"],
        "ratio_prad_full_to_half": res_full_free["p_rad_flux"] / res_half_pec["p_rad_flux"],
        "ratio_ploss_full_to_half": res_full_free["p_loss"] / res_half_pec["p_loss"],
        "h_lift_lam": h_lift_lam,
        "eff_lift_fit": np.asarray(eff_lift_f0, dtype=float),
        "eff_lift_ana": np.asarray(eff_ana_lift, dtype=float),
    }


def run_task_03_equal_hm_families(quick: bool = False) -> dict[str, Any]:
    """Run Part 3: Equal-hesitivity Lorentz (4 poles) and Debye (3 poles) families on PEC ground."""
    t0 = time.perf_counter()
    f0 = 0.28e9
    freqs = np.linspace(0.16e9, 0.40e9, 5 if quick else 7)
    nh = 2 if quick else 3
    dx_fine = 0.012 / nh  # 6.0 mm (quick) or 4.0 mm (converged Medium level, 6 cells across 24 mm)
    dx_coarse = 0.0300 if quick else 0.0280

    grid = build_rod_grid(
        dx_fine_m=dx_fine,
        dx_coarse_m=dx_coarse,
        n_core_x=3 * nh + 2,
        n_core_y=nh + 2,
        n_core_z=2 * nh + 2,
        half_space=True,
    )
    rod_c = (0.0, 0.0, 0.012)
    rod_s = (0.072, 0.024, 0.024)
    vol_eff = float(2.0 * 0.072 * 0.024 * 0.024)

    pulse = GaussianSource(frequency=f0, fwidth=0.45 * f0)
    sources = _make_trapezoid_coil_sources(
        pulse, l_coil=0.048, w_rod=0.024, t_rod=0.024, z_center=0.012, half_space_arch=True
    )

    # 4 Lorentz poles with identical hesitivity hm = CZN_HM_OHM_M = 3.464231e5 Ohm/m
    lor_specs = [
        ("Lor-1 (f0=0.80 GHz, chi=54)", 0.80e9, 54.0),
        ("Lor-2 (f0=1.15 GHz, chi=45)", 1.15e9, 45.0),
        ("Lor-3 (f0=1.50 GHz, chi=39)", 1.50e9, 39.0),
        ("Lor-4 (f0=2.10 GHz, chi=32)", 2.10e9, 32.0),
    ]
    lor_eff_db = []
    lor_eff_pol_db = []
    lor_eta_fill = []
    lor_rw_db = []
    lor_tan_d = []
    p_rad_lor3_rod = np.zeros_like(freqs)
    for idx_pole, (_, f_res, chi_dc) in enumerate(lor_specs):
        w0 = 2.0 * np.pi * f_res
        alpha = w0 * MU0 * chi_dc / CZN_HM_OHM_M
        mat = DispersiveMaterial(mu_inf=1.0, mu_poles=[Lorentz(delta_eps=chi_dc, omega_0=w0, gamma=alpha * w0)])
        res = _run_rod_simulation(
            grid=grid,
            dx_fine=dx_fine,
            rod_center=rod_c,
            rod_size=rod_s,
            material=mat,
            sources=sources,
            pulse=pulse,
            freqs_hz=freqs,
            bc_low=["cpml", "cpml", "pec"],
            n_periods=4.8 if quick else 5.6,
        )
        lor_eff_db.append(res["eff_flux_db"])
        lor_eta_fill.append(res["eta_fill"])
        lor_eff_pol_db.append(res["eff_pol_dft_db"])
        if idx_pole == 2:
            p_rad_lor3_rod = res["p_rad_flux"].copy()
        mu_f = mat.mu(freqs)
        lor_tan_d.append(-np.imag(mu_f) / np.real(mu_f))
        lor_rw_db.append(10.0 * np.log10(rumsey_weeks_efficiency(freqs, mu_f, vol_eff)))

    # 3 Debye poles with identical hesitivity hm_deb = 2.357e5 Ohm/m
    hm_deb = 2.357e5
    deb_specs = [
        ("Deb-1 (chi=56)", 56.0),
        ("Deb-2 (chi=42)", 42.0),
        ("Deb-3 (chi=32)", 32.0),
    ]
    deb_eff_db = []
    deb_eff_pol_db = []
    deb_eta_fill = []
    deb_rw_db = []
    deb_tan_d = []
    for _, chi_dc in deb_specs:
        f_rel = hm_deb / (2.0 * np.pi * MU0 * chi_dc)
        tau = 1.0 / (2.0 * np.pi * f_rel)
        mat_d = DispersiveMaterial(mu_inf=1.0, mu_poles=[Debye(delta_eps=chi_dc, tau=tau)])
        res_d = _run_rod_simulation(
            grid=grid,
            dx_fine=dx_fine,
            rod_center=rod_c,
            rod_size=rod_s,
            material=mat_d,
            sources=sources,
            pulse=pulse,
            freqs_hz=freqs,
            bc_low=["cpml", "cpml", "pec"],
            n_periods=4.8 if quick else 5.6,
        )
        deb_eff_db.append(res_d["eff_flux_db"])
        deb_eta_fill.append(res_d["eta_fill"])
        deb_eff_pol_db.append(res_d["eff_pol_dft_db"])
        mu_f = mat_d.mu(freqs)
        deb_tan_d.append(-np.imag(mu_f) / np.real(mu_f))
        deb_rw_db.append(10.0 * np.log10(rumsey_weeks_efficiency(freqs, mu_f, vol_eff)))

    # Air-core vacuum loop reference (material=None) to directly quantify the driving coil's own dipole moment
    res_vac_loop = _run_rod_simulation(
        grid=grid,
        dx_fine=dx_fine,
        rod_center=rod_c,
        rod_size=rod_s,
        material=None,
        sources=sources,
        pulse=pulse,
        freqs_hz=freqs,
        bc_low=["cpml", "cpml", "pec"],
        n_periods=4.8 if quick else 5.6,
    )
    p_rad_vac_loop = res_vac_loop["p_rad_flux"]
    rod_to_loop_prad_ratio = p_rad_lor3_rod / np.maximum(p_rad_vac_loop, 1e-30)

    lor_eff_arr = np.asarray(lor_eff_db, dtype=float)
    deb_eff_arr = np.asarray(deb_eff_db, dtype=float)
    lor_ana_db = 10.0 * np.log10(closed_form_md_efficiency(freqs, CZN_HM_OHM_M, vol_eff))
    deb_ana_db = 10.0 * np.log10(closed_form_md_efficiency(freqs, hm_deb, vol_eff))

    lor_spread_db = float(np.max(np.ptp(lor_eff_arr, axis=0)))
    deb_spread_db = float(np.max(np.ptp(deb_eff_arr, axis=0)))

    return {
        "quick": np.array(int(quick)),
        "wall_s": np.array(float(time.perf_counter() - t0)),
        "peak_rss_mb": np.array(_peak_rss_mb()),
        "freqs_ghz": freqs * 1e-9,
        "k0_l": (2.0 * np.pi * freqs / C0) * 0.072,
        "lor_eff_db": lor_eff_arr,
        "lor_eff_pol_db": np.asarray(lor_eff_pol_db, dtype=float),
        "lor_eta_fill": np.asarray(lor_eta_fill, dtype=float),
        "lor_rw_db": np.asarray(lor_rw_db, dtype=float),
        "lor_tan_d": np.asarray(lor_tan_d, dtype=float),
        "lor_ana_db": lor_ana_db,
        "lor_spread_db": np.array(lor_spread_db),
        "deb_eff_db": deb_eff_arr,
        "deb_eff_pol_db": np.asarray(deb_eff_pol_db, dtype=float),
        "deb_eta_fill": np.asarray(deb_eta_fill, dtype=float),
        "deb_rw_db": np.asarray(deb_rw_db, dtype=float),
        "deb_tan_d": np.asarray(deb_tan_d, dtype=float),
        "deb_ana_db": deb_ana_db,
        "deb_spread_db": np.array(deb_spread_db),
        "p_rad_lor3_rod": p_rad_lor3_rod,
        "p_rad_vac_loop": p_rad_vac_loop,
        "rod_to_loop_prad_ratio": rod_to_loop_prad_ratio,
        "hm_lor_ohm_m": np.array(CZN_HM_OHM_M),
        "hm_deb_ohm_m": np.array(hm_deb),
    }


def _dual_shifted_axis(u: np.ndarray) -> np.ndarray:
    """Return a dual-shifted 1D node axis whose cell midpoints coincide with ``u``."""
    return np.concatenate(
        [
            [u[0] - 0.5 * (u[1] - u[0])],
            0.5 * (u[:-1] + u[1:]),
            [u[-1] + 0.5 * (u[-1] - u[-2])],
        ]
    )


def run_task_04_dual_pair(quick: bool = False) -> dict[str, Any]:
    """Run Part 4: Continuous Maxwell PEC<->PMC dual (discrete dual up to outer grading) and orientational PEC pair."""
    t0 = time.perf_counter()
    f0 = 0.25e9
    freqs = np.linspace(0.18e9, 0.32e9, 5 if quick else 7)
    omega0 = 2.0 * np.pi * f0
    dx = 0.0045 if quick else 0.0034
    dx_coarse = 0.0200 if quick else 0.0170

    # Refined outer grading (g = 1.18) with uniform core (10, 7, 8)*dx enclosing the Huygens box (8, 5, 6)*dx
    grid_m = build_rod_grid(
        dx_fine_m=dx,
        dx_coarse_m=dx_coarse,
        n_core_x=10,
        n_core_y=7,
        n_core_z=8,
        domain_x_m=0.22,
        domain_y_m=0.20,
        domain_z_m=0.20,
        half_space=True,
        grading=1.18,
    )
    xp = _dual_shifted_axis(grid_m.x[:-1])
    yp = _dual_shifted_axis(grid_m.y[:-1])
    zp = _dual_shifted_axis(grid_m.z[:-1])
    grid_pmc = GridFIT3D(xp[0], xp[-1], yp[0], yp[-1], zp[0], zp[-1], x=xp, y=yp, z=zp, verbose=0)

    # Coarser outer grading reference (g = 1.35) to verify that the discrete PEC<->PMC gap O((g-1)^2 * dx) shrinks
    dx_coarse_ref = 0.0280 if quick else 0.0260
    grid_m_coarse = build_rod_grid(
        dx_fine_m=dx,
        dx_coarse_m=dx_coarse_ref,
        n_core_x=8,
        n_core_y=5,
        n_core_z=6,
        domain_x_m=0.22,
        domain_y_m=0.20,
        domain_z_m=0.20,
        half_space=True,
        grading=1.35,
    )
    xp_c = _dual_shifted_axis(grid_m_coarse.x[:-1])
    yp_c = _dual_shifted_axis(grid_m_coarse.y[:-1])
    zp_c = _dual_shifted_axis(grid_m_coarse.z[:-1])
    grid_pmc_coarse = GridFIT3D(xp_c[0], xp_c[-1], yp_c[0], yp_c[-1], zp_c[0], zp_c[-1], x=xp_c, y=yp_c, z=zp_c, verbose=0)

    grid_e = build_rod_grid(
        dx_fine_m=dx,
        dx_coarse_m=dx_coarse,
        n_core_x=6,
        n_core_y=8,
        n_core_z=9,
        domain_x_m=0.20,
        domain_y_m=0.20,
        domain_z_m=0.22,
        half_space=True,
        grading=1.18,
    )

    pulse = GaussianSource(frequency=f0, fwidth=0.42 * f0)
    const_pairs = [(300.0, 15.0), (80.0, 4.0), (20.0, 1.0)]

    eff_horiz_mag_pec = []
    eff_vert_diel_pec = []
    eff_horiz_diel_pmc = []
    eff_closed_form = []
    dual_pmc_coarse_diff_f0_db = []
    vol_eff = float(2.0 * 192.0 * (dx**3))
    rod_c = (0.0, 0.0, 2.0 * dx)
    rod_s = (12.0 * dx, 4.0 * dx, 4.0 * dx)
    box_m = (8.0 * dx, 5.0 * dx, 6.0 * dx)
    l_coil_dual = 8.0 * dx

    def _trapz_x_dual(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.where(np.isclose(np.abs(x), 0.5 * l_coil_dual, atol=1e-6), 0.5, 1.0)

    for re_val, im_val in const_pairs:
        chi_re = re_val - 1.0
        x_wt = im_val / chi_re
        tau = x_wt / omega0
        chi_dc = chi_re * (1.0 + x_wt**2)
        hm_val = hesitivity_debye(chi_dc, 1.0 / (2.0 * np.pi * tau))

        # (A) Horizontal permeable rod on PEC (4 cells across y and z, >= 20 cells/lam_mat even at mu'=300)
        mat_m = DispersiveMaterial(mu_inf=1.0, mu_poles=[Debye(delta_eps=chi_dc, tau=tau)])
        srcs_m = [
            CurrentSource(
                pulse, component="y", center=(0.0, 0.0, 4.0 * dx), size=(l_coil_dual, 4.0 * dx, 0.0), amplitude=1.0, amp_func=_trapz_x_dual
            ),
            CurrentSource(
                pulse, component="z", center=(0.0, -2.0 * dx, 2.0 * dx), size=(l_coil_dual, 0.0, 4.0 * dx), amplitude=1.0, amp_func=_trapz_x_dual
            ),
            CurrentSource(
                pulse, component="z", center=(0.0, 2.0 * dx, 2.0 * dx), size=(l_coil_dual, 0.0, 4.0 * dx), amplitude=-1.0, amp_func=_trapz_x_dual
            ),
        ]
        res_m = _run_rod_simulation(
            grid=grid_m,
            dx_fine=dx,
            rod_center=rod_c,
            rod_size=rod_s,
            material=mat_m,
            sources=srcs_m,
            pulse=pulse,
            freqs_hz=freqs,
            bc_low=["cpml", "cpml", "pec"],
            n_pml=5,
            n2f_box_half=box_m,
            n_periods=4.8 if quick else 5.2,
        )
        eff_horiz_mag_pec.append(res_m["eff_flux_db"])

        # (B) Sebastian orientational PEC pair: Vertical dielectric monopole on PEC (same +1 image sign, 4x8x6 cells = 192*dx^3)
        mat_e = DispersiveMaterial(eps_inf=1.0, poles=[Debye(delta_eps=chi_dc, tau=tau)])
        srcs_e = [
            MagneticCurrentSource(pulse, component="x", center=(0.0, -4.5 * dx, 2.5 * dx), size=(4.0 * dx, 0.0, 5.0 * dx), amplitude=1.0),
            MagneticCurrentSource(pulse, component="x", center=(0.0, 4.5 * dx, 2.5 * dx), size=(4.0 * dx, 0.0, 5.0 * dx), amplitude=-1.0),
            MagneticCurrentSource(pulse, component="y", center=(-2.5 * dx, 0.0, 2.5 * dx), size=(0.0, 8.0 * dx, 5.0 * dx), amplitude=-1.0),
            MagneticCurrentSource(pulse, component="y", center=(2.5 * dx, 0.0, 2.5 * dx), size=(0.0, 8.0 * dx, 5.0 * dx), amplitude=1.0),
        ]
        res_e = _run_rod_simulation(
            grid=grid_e,
            dx_fine=dx,
            rod_center=(0.0, 0.0, 3.0 * dx),
            rod_size=(4.0 * dx, 8.0 * dx, 6.0 * dx),
            material=mat_e,
            sources=srcs_e,
            pulse=pulse,
            freqs_hz=freqs,
            bc_low=["cpml", "cpml", "pec"],
            n_pml=5,
            n2f_box_half=(5.0 * dx, 6.0 * dx, 7.0 * dx),
            n_periods=4.8 if quick else 5.2,
        )
        eff_vert_diel_pec.append(res_e["eff_flux_db"])

        # (C) Continuous Maxwell boundary-value dual: Horizontal dielectric rod on dual-shifted PMC grid
        # Exact to machine precision inside the uniform core; differs by O((g-1)^2 * dx) in the outer graded buffer.
        srcs_pmc = [
            MagneticCurrentSource(
                pulse, component="y", center=(0.0, 0.0, 4.0 * dx), size=(l_coil_dual, 4.0 * dx, 0.0), amplitude=1.0, amp_func=_trapz_x_dual
            ),
            MagneticCurrentSource(
                pulse, component="z", center=(0.0, -2.0 * dx, 2.0 * dx), size=(l_coil_dual, 0.0, 4.0 * dx), amplitude=1.0, amp_func=_trapz_x_dual
            ),
            MagneticCurrentSource(
                pulse, component="z", center=(0.0, 2.0 * dx, 2.0 * dx), size=(l_coil_dual, 0.0, 4.0 * dx), amplitude=-1.0, amp_func=_trapz_x_dual
            ),
        ]
        res_pmc = _run_rod_simulation(
            grid=grid_pmc,
            dx_fine=dx,
            rod_center=rod_c,
            rod_size=rod_s,
            material=mat_e,
            sources=srcs_pmc,
            pulse=pulse,
            freqs_hz=freqs,
            bc_low=["cpml", "cpml", "pmc"],
            n_pml=5,
            n2f_box_half=box_m,
            n_periods=4.8 if quick else 5.2,
        )
        eff_horiz_diel_pmc.append(res_pmc["eff_flux_db"])
        eff_closed_form.append(10.0 * np.log10(closed_form_md_efficiency(freqs, hm_val, vol_eff)))

        # Coarse outer-grading reference at f0 to verify the PEC<->PMC discrepancy shrinks with grading refinement
        res_m_c = _run_rod_simulation(
            grid=grid_m_coarse,
            dx_fine=dx,
            rod_center=rod_c,
            rod_size=rod_s,
            material=mat_m,
            sources=srcs_m,
            pulse=pulse,
            freqs_hz=np.array([f0]),
            bc_low=["cpml", "cpml", "pec"],
            n_pml=5,
            n2f_box_half=box_m,
            n_periods=4.8 if quick else 5.2,
        )
        res_pmc_c = _run_rod_simulation(
            grid=grid_pmc_coarse,
            dx_fine=dx,
            rod_center=rod_c,
            rod_size=rod_s,
            material=mat_e,
            sources=srcs_pmc,
            pulse=pulse,
            freqs_hz=np.array([f0]),
            bc_low=["cpml", "cpml", "pmc"],
            n_pml=5,
            n2f_box_half=box_m,
            n_periods=4.8 if quick else 5.2,
        )
        dual_pmc_coarse_diff_f0_db.append(float(abs(res_m_c["eff_flux_db"][0] - res_pmc_c["eff_flux_db"][0])))

    arr_m = np.asarray(eff_horiz_mag_pec, dtype=float)
    arr_e = np.asarray(eff_vert_diel_pec, dtype=float)
    arr_pmc = np.asarray(eff_horiz_diel_pmc, dtype=float)
    idx_f0 = len(freqs) // 2

    return {
        "quick": np.array(int(quick)),
        "wall_s": np.array(float(time.perf_counter() - t0)),
        "peak_rss_mb": np.array(_peak_rss_mb()),
        "freqs_ghz": freqs * 1e-9,
        "dx_fine_m": np.array(dx),
        "const_labels": np.array(["300-j15", "80-j4", "20-j1"]),
        "eff_horiz_mag_pec_db": arr_m,
        "eff_vert_diel_pec_db": arr_e,
        "eff_horiz_diel_pmc_db": arr_pmc,
        "eff_closed_form_db": np.asarray(eff_closed_form, dtype=float),
        "dual_pmc_diff_f0_db": np.abs(arr_m[:, idx_f0] - arr_pmc[:, idx_f0]),
        "dual_pmc_coarse_diff_f0_db": np.asarray(dual_pmc_coarse_diff_f0_db, dtype=float),
        "dual_pec_diff_f0_db": np.abs(arr_m[:, idx_f0] - arr_e[:, idx_f0]),
    }


def run_task_05_metal_vs_permeable_and_convergence(quick: bool = False) -> dict[str, Any]:
    """Run Parts 5 & 6: Metal strip vs permeable bar EBWP, finite PEC ground tilt, and 3-resolution convergence."""
    t0 = time.perf_counter()

    # 1. Analytic & circuit EBWP comparison (10 MHz to 300 MHz, Yousefi Sec. V)
    freqs_vhf = np.linspace(10.0e6, 300.0e6, 120)
    ebwp_data = conformal_metal_vs_permeable_ebwp(freqs_vhf)

    # Direct 3D FIT verification of the > 10^2x radiation power / resistance suppression of a
    # horizontal electric strip at height h = t above PEC vs a horizontal magnetic bar of thickness t on PEC
    f0_comp = 0.25e9  # lam0 = 1.20 m, t = 0.020 m -> t/lam0 = 1/60 (same electrical thickness as 1" at 315 MHz)
    dx_c = 0.010
    x_c = np.arange(-12, 13, dtype=float) * dx_c
    y_c = np.arange(-10, 11, dtype=float) * dx_c
    z_c = np.arange(0, 14, dtype=float) * dx_c
    grid_c = GridFIT3D(x_c[0], x_c[-1], y_c[0], y_c[-1], z_c[0], z_c[-1], x=x_c, y=y_c, z=z_c, verbose=0)
    pulse_c = GaussianSource(frequency=f0_comp, fwidth=0.45 * f0_comp, cutoff=4.0)
    dt_c = 0.5 * dx_c / (np.sqrt(3.0) * C0)
    n_steps_c = int(np.ceil((pulse_c.end_time + 2.0 / f0_comp) / dt_c))

    # Conformal horizontal electric strip at z = 2*dx_c (t = 20 mm) vs free space
    solver_strip_pec = SolverFIT3D(grid_c, bc_low=["cpml", "cpml", "pec"], bc_high=["cpml"] * 3, n_pml=4, verbose=0)
    src_strip = CurrentSource(pulse_c, component="x", center=(0.0, 0.0, 2.0 * dx_c), size=(10.0 * dx_c, 4.0 * dx_c, 0.0), amplitude=1.0)
    n2f_strip_pec = Near2FarMonitor(center=(0.0, 0.0, 4.0 * dx_c), size=(16.0 * dx_c, 10.0 * dx_c, 8.0 * dx_c), frequencies=[f0_comp], ground_plane="-z")
    for step in range(n_steps_c):
        t = step * solver_strip_pec.dt
        src_strip.update(solver_strip_pec, t)
        solver_strip_pec.one_step()
        n2f_strip_pec.update(solver_strip_pec, t)
    p_strip_pec = float(n2f_strip_pec.radiated_power()[0])

    # Equivalent horizontal magnetic bar at z = 1*dx_c (flush on PEC) with matched free-space moment
    solver_mbar_pec = SolverFIT3D(grid_c, bc_low=["cpml", "cpml", "pec"], bc_high=["cpml"] * 3, n_pml=4, verbose=0)
    src_mbar = MagneticCurrentSource(
        pulse_c, component="x", center=(0.0, 0.0, 1.0 * dx_c), size=(10.0 * dx_c, 4.0 * dx_c, 0.0), amplitude=ETA0
    )
    n2f_mbar_pec = Near2FarMonitor(center=(0.0, 0.0, 4.0 * dx_c), size=(16.0 * dx_c, 10.0 * dx_c, 8.0 * dx_c), frequencies=[f0_comp], ground_plane="-z")
    for step in range(n_steps_c):
        t = step * solver_mbar_pec.dt
        src_mbar.update(solver_mbar_pec, t)
        solver_mbar_pec.one_step()
        n2f_mbar_pec.update(solver_mbar_pec, t)
    p_mbar_pec = float(n2f_mbar_pec.radiated_power()[0])
    fit_mbar_to_strip_ratio = float(p_mbar_pec / max(p_strip_pec, 1e-30))

    # 2. Part 6: Finite PEC ground plane size sweep & elevation pattern tilt
    # Domain & Huygens box sized so even the largest plate (1.10*lam0 = 0.33 m, half-width 11*dx_g = 0.165 m)
    # lies strictly inside the Huygens box (half-span 14*dx_g = 0.21 m, 0.15*lam0 clearance) with >= lam0/4
    # clearance from the plate edge to the CPML boundary (inner CPML at 16*dx_g or 17*dx_g -> 0.25-0.30*lam0 clearance).
    f0_gnd = 1.0e9
    lam0_gnd = C0 / f0_gnd  # 0.30 m
    k0_gnd = 2.0 * np.pi / lam0_gnd
    dx_g = 0.015  # lam0 / 20
    nx_g = 20 if quick else 21
    x_g = np.arange(-nx_g, nx_g + 1, dtype=float) * dx_g
    y_g = np.arange(-nx_g, nx_g + 1, dtype=float) * dx_g
    z_g = np.arange(-11 if quick else -12, 16 if quick else 17, dtype=float) * dx_g
    grid_g = GridFIT3D(x_g[0], x_g[-1], y_g[0], y_g[-1], z_g[0], z_g[-1], x=x_g, y=y_g, z=z_g, verbose=0)
    pulse_g = GaussianSource(frequency=f0_gnd, fwidth=0.45 * f0_gnd, cutoff=4.0)
    sf_g = complex(pulse_g.fourier_transform(f0_gnd))
    p_free_g = float((k0_gnd**2) / (6.0 * np.pi * ETA0) * (abs(sf_g) ** 2))
    dt_g = 0.5 * dx_g / (np.sqrt(3.0) * C0)
    n_steps_g = int(np.ceil((pulse_g.end_time + 2.2 / f0_gnd) / dt_g))
    preflight_simulation_grid(grid_g, n_steps=n_steps_g, n_freqs=1, dispersive_edges=0)

    h_gnd_lam = float(0.5 * dx_g / lam0_gnd)  # 0.025 * lam0
    inf_gnd_r_ratio = float(pec_image_resistance_ratio(h_gnd_lam, "HMD"))
    inf_gnd_zenith_dir = float(pec_hmd_zenith_directivity(h_gnd_lam))

    gnd_widths_lam = [0.30, 0.50, 0.80, 1.10]
    r_ratio_finite_gnd = []
    tilt_deg_finite_gnd = []
    theta_rad = np.linspace(1e-3, np.pi - 1e-3, 181)
    phi_90 = np.full_like(theta_rad, 0.5 * np.pi)
    patterns_finite_gnd = []
    mon_span = 2.0 * 14 * dx_g  # 0.42 m = 1.40 * lam0

    for w_lam in gnd_widths_lam:
        w_m = w_lam * lam0_gnd
        solver_fg = SolverFIT3D(grid_g, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=4, verbose=0)
        solver_fg.add_sheet(center=(0.0, 0.0, 0.0), size=(w_m, w_m, 0.0), sigma_s="pec")
        src_fg = MagneticCurrentSource(pulse_g, component="x", center=(0.0, 0.0, 0.5 * dx_g), amplitude=1.0)
        n2f_fg = Near2FarMonitor(
            center=(0.0, 0.0, 1.5 * dx_g), size=(mon_span, mon_span, 13.0 * dx_g), frequencies=[f0_gnd]
        )
        for step in range(n_steps_g):
            t = step * solver_fg.dt
            src_fg.update(solver_fg, t)
            solver_fg.one_step()
            n2f_fg.update(solver_fg, t)
        r_ratio_finite_gnd.append(float(n2f_fg.radiated_power()[0]) / p_free_g)
        d_pat = np.asarray(n2f_fg.directivity(theta_rad, phi_90)[0], dtype=float)
        patterns_finite_gnd.append(d_pat)
        # Peak elevation angle in upper hemisphere theta in [0, 90 deg]
        upper_mask = theta_rad <= 0.5 * np.pi
        idx_pk = int(np.argmax(d_pat[upper_mask]))
        tilt_deg_finite_gnd.append(float(np.rad2deg(theta_rad[upper_mask][idx_pk])))

    # 3. 3-resolution spatial convergence study on the EXACT Part 2/3 permeable rod geometry
    # Physical rod dimensions fixed: length l = 72 mm, width w = 24 mm, physical height t = 24 mm on PEC
    # Coarse: dx = 6.00 mm (4 cells across 24 mm, 12 cells along 72 mm, >= 28 cells/lam_mat inside mu_r=40)
    # Medium: dx = 4.00 mm (6 cells across 24 mm, 18 cells along 72 mm, >= 42 cells/lam_mat) -> headline resolution
    # Fine:   dx = 3.00 mm (8 cells across 24 mm, 24 cells along 72 mm, >= 56 cells/lam_mat)
    f0_conv = 0.28e9
    half_w = 0.012
    n_half_list = [2, 3, 4]
    n_thick_list = [2 * nh for nh in n_half_list]  # [4, 6, 8] cells across physical thickness & width (24 mm)
    dx_conv_list = [half_w / nh for nh in n_half_list]  # [6.0 mm, 4.0 mm, 3.0 mm]
    conv_eff = []
    conv_bal_err = []
    conv_cells = []
    conv_cpw_mat = []
    conv_cells_per_skin = []

    w0_czn = 2.0 * np.pi * CZN_F0_HZ
    gam_czn = CZN_ALPHA * w0_czn
    mat_conv = DispersiveMaterial(mu_inf=1.0, mu_poles=[Lorentz(delta_eps=CZN_CHI_DC, omega_0=w0_czn, gamma=gam_czn)])
    mu_conv = complex(mat_conv.mu(f0_conv))
    n_mat = np.sqrt(mu_conv)
    n_mat = -n_mat if np.imag(n_mat) > 0 else n_mat
    lam_mat_m = float((C0 / f0_conv) / abs(n_mat))
    skin_depth_m = float(1.0 / ((2.0 * np.pi * f0_conv / C0) * max(-np.imag(n_mat), 1e-6)))

    l_rod = 0.072
    w_rod = 0.024
    t_rod = 0.024
    zc_rod = 0.012
    l_coil = 0.048
    for nh, dx_f in zip(n_half_list, dx_conv_list):
        grid_cv = build_rod_grid(
            dx_fine_m=dx_f,
            dx_coarse_m=0.0300 if quick else 0.0280,
            n_core_x=3 * nh + 2,
            n_core_y=nh + 2,
            n_core_z=2 * nh + 2,
            domain_x_m=0.18 if quick else 0.21,
            domain_y_m=0.14 if quick else 0.17,
            domain_z_m=0.15 if quick else 0.18,
            half_space=True,
        )
        pulse_cv = GaussianSource(frequency=f0_conv, fwidth=0.45 * f0_conv)
        srcs_cv = _make_trapezoid_coil_sources(
            pulse_cv, l_coil=l_coil, w_rod=w_rod, t_rod=t_rod, z_center=zc_rod, half_space_arch=True
        )
        res_cv = _run_rod_simulation(
            grid=grid_cv,
            dx_fine=dx_f,
            rod_center=(0.0, 0.0, zc_rod),
            rod_size=(l_rod, w_rod, t_rod),
            material=mat_conv,
            sources=srcs_cv,
            pulse=pulse_cv,
            freqs_hz=np.array([f0_conv]),
            bc_low=["cpml", "cpml", "pec"],
            n2f_box_half=(0.084, 0.064, 0.074),
            n_periods=4.8 if quick else 5.6,
        )
        conv_eff.append(float(res_cv["eff_flux"][0]))
        conv_bal_err.append(float(res_cv["balance_rel_err"][0]))
        conv_cells.append(int(res_cv["n_cells"]))
        conv_cpw_mat.append(float(lam_mat_m / dx_f))
        conv_cells_per_skin.append(float(skin_depth_m / dx_f))

    # Observed convergence order p_obs from all 3 levels (dx_1=6.0 mm, dx_2=4.0 mm, dx_3=3.0 mm)
    from scipy.optimize import brentq

    diff_ratio = (conv_eff[1] - conv_eff[0]) / (conv_eff[2] - conv_eff[1])
    p_obs = float(
        brentq(
            lambda p: ((dx_conv_list[0] ** p - dx_conv_list[1] ** p) / (dx_conv_list[1] ** p - dx_conv_list[2] ** p))
            - diff_ratio,
            0.1,
            6.0,
        )
    )
    r_grid = dx_conv_list[1] / dx_conv_list[2]  # 4/3
    eff_rich = float(conv_eff[2] + (conv_eff[2] - conv_eff[1]) / (r_grid**p_obs - 1.0))
    conv_rel_err = np.abs(np.asarray(conv_eff, dtype=float) - eff_rich) / eff_rich
    pats_arr = np.asarray(patterns_finite_gnd, dtype=float)

    return {
        "quick": np.array(int(quick)),
        "wall_s": np.array(float(time.perf_counter() - t0)),
        "peak_rss_mb": np.array(_peak_rss_mb()),
        **ebwp_data,
        "fit_mbar_to_strip_ratio": np.array(fit_mbar_to_strip_ratio),
        "h_gnd_lam": np.array(h_gnd_lam),
        "inf_gnd_r_ratio": np.array(inf_gnd_r_ratio),
        "inf_gnd_zenith_dir": np.array(inf_gnd_zenith_dir),
        "gnd_widths_lam": np.asarray(gnd_widths_lam, dtype=float),
        "r_ratio_finite_gnd": np.asarray(r_ratio_finite_gnd, dtype=float),
        "tilt_deg_finite_gnd": np.asarray(tilt_deg_finite_gnd, dtype=float),
        "zenith_dir_finite_gnd": pats_arr[:, 0],
        "nadir_dir_finite_gnd": pats_arr[:, -1],
        "theta_deg": np.rad2deg(theta_rad),
        "patterns_finite_gnd": pats_arr,
        "dx_conv_mm": np.asarray(dx_conv_list, dtype=float) * 1e3,
        "n_thick_conv": np.asarray(n_thick_list, dtype=int),
        "conv_eff": np.asarray(conv_eff, dtype=float),
        "conv_p_obs": np.array(p_obs),
        "conv_eff_rich": np.array(eff_rich),
        "conv_rel_err": conv_rel_err,
        "conv_bal_err": np.asarray(conv_bal_err, dtype=float),
        "conv_cells": np.asarray(conv_cells, dtype=int),
        "conv_cpw_mat": np.asarray(conv_cpw_mat, dtype=float),
        "conv_cells_per_skin": np.asarray(conv_cells_per_skin, dtype=float),
        "lam_mat_mm": np.array(lam_mat_m * 1e3),
        "skin_depth_mm": np.array(skin_depth_m * 1e3),
    }


# ---------------------------------------------------------------------------
# 4. Figure & GIF generators (using wakis.figstyle and wakis.schematic)
# ---------------------------------------------------------------------------
def _build_schematic_objects(d0: Any) -> tuple[SolverFIT3D, list[Any], Near2FarMonitor, Box, dict[int, str]]:
    """Reconstruct the 3D FIT solver, rod structure, sources, and Near2Far box for schematic overlays."""
    x_nodes = np.asarray(d0["x_nodes"], dtype=float)
    y_nodes = np.asarray(d0["y_nodes"], dtype=float)
    z_nodes = np.asarray(d0["z_nodes"], dtype=float)
    grid = GridFIT3D(
        float(x_nodes[0]),
        float(x_nodes[-1]),
        float(y_nodes[0]),
        float(y_nodes[-1]),
        float(z_nodes[0]),
        float(z_nodes[-1]),
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        verbose=0,
    )
    solver = SolverFIT3D(grid, bc_low=["cpml", "cpml", "pec"], bc_high=["cpml"] * 3, n_pml=4, verbose=0)
    f0 = 0.28e9
    pulse = GaussianSource(frequency=f0, fwidth=0.45 * f0)
    w0_czn = 2.0 * np.pi * CZN_F0_HZ
    gam_czn = CZN_ALPHA * w0_czn
    mat_lor = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[Lorentz(delta_eps=CZN_CHI_DC, omega_0=w0_czn, gamma=gam_czn)],
        name="CZN Lorentz laminate",
    )
    rod_box = Box(
        (0.0, 0.0, 0.012),
        (0.072, 0.024, 0.024),
        mat_lor,
    )
    mat_names = {id(rod_box): "Permeable rod (CZN Lorentz mu)"}
    sources = _make_trapezoid_coil_sources(
        pulse, l_coil=0.048, w_rod=0.024, t_rod=0.024, z_center=0.012, half_space_arch=True
    )
    bx = float(x_nodes[int(np.argmin(np.abs(x_nodes - 0.084)))])
    by = float(y_nodes[int(np.argmin(np.abs(y_nodes - 0.064)))])
    bz = float(z_nodes[int(np.argmin(np.abs(z_nodes - 0.074)))])
    n2f = Near2FarMonitor(center=(0.0, 0.0, 0.5 * bz), size=(2.0 * bx, 2.0 * by, bz), frequencies=[f0], ground_plane="-z")
    n2f.faces = [n2f]
    return solver, sources, n2f, rod_box, mat_names


def plot_00_setup_schematic(d0: Any, fig_dir: Path) -> Path:
    """Render Figures 00a/00b and setup table via :func:`wakis.schematic.save_setup_figures`."""
    figstyle.apply_style()
    solver, sources, n2f, rod_box, mat_names = _build_schematic_objects(d0)
    f0 = 0.28e9
    variations = [
        {"name": "1D Magnetic ADE slab", "details": "TEM waveguide ($2\\times 2\\times N_z$), Lorentz & Debye $\\mu(\\omega)$ slabs"},
        {"name": "4-orientation PEC image sweep", "details": "HMD, VMD, VED, HED over $h/\\lambda_0 \\in [0.04, 0.50]$"},
        {"name": "Volume convention & lift", "details": "Half-rod on PEC vs full free-space rod ($2 V_{\\mathrm{phys}}$) vs lifted rod"},
        {"name": "Equal-$h_m$ Lorentz & Debye families", "details": "4 Lorentz poles ($h_m=3.46\\times 10^5\\,\\Omega$/m) & 3 Debye poles ($h_m=2.36\\times 10^5\\,\\Omega$/m)"},
        {"name": "PEC/PMC duality & finite ground", "details": "Horiz permeable rod vs vert dielectric monopole; finite PEC $L_{\\mathrm{gnd}}/\\lambda_0$"},
    ]
    res = schematic.save_setup_figures(
        solver,
        fig_dir,
        plane="xz",
        cut_coord=0.0,
        unit="mm",
        freq=f0,
        min_feature=0.024,
        structures=[rod_box],
        material_names=mat_names,
        sources=sources[:1],
        monitors=[n2f],
        title_geometry="Conformal Permeable Rod on PEC Ground ($xz$ plane, $y = 0$)",
        title_mesh="Locally Refined Graded FIT Mesh and Broadband Excitation",
        variations=variations,
    )
    return res["geometry_figure"]


def plot_01_magnetic_ade_and_hesitivity(d0: Any, fig_dir: Path) -> Path:
    """Render Figure 01: Lorentz/Debye susceptibility, hesitivity identity f(mu)=w*mu0/hm, and 1D FIT slab R/T/A."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=12.2, height_in=4.5, ncols=3, nrows=1)
    fig.subplots_adjust(wspace=0.34)
    ax_mu, ax_hm, ax_slab = axes

    f_ghz = np.asarray(d0["freqs_disp_ghz"], dtype=float)
    ax_mu.plot(f_ghz, d0["mu_lor_real"], color=COLORS["blue"], lw=1.8, label=r"Lorentz $\mu'$ (CZN)")
    ax_mu.plot(f_ghz, d0["mu_lor_imag"], "--", color=COLORS["blue"], lw=1.6, label=r"Lorentz $\mu''$ (CZN)")
    ax_mu.plot(f_ghz, d0["mu_deb_real"], color=COLORS["teal"], lw=1.8, label=r"Debye $\mu'$ (NiZn ferrite)")
    ax_mu.plot(f_ghz, d0["mu_deb_imag"], "--", color=COLORS["teal"], lw=1.6, label=r"Debye $\mu''$ (NiZn ferrite)")
    ax_mu.axhline(0.0, color=COLORS["grey"], ls=":", lw=0.8)
    ax_mu.set_xlabel("Frequency $f$ [GHz]")
    ax_mu.set_ylabel(r"Permeability $\mu_r(f) = \mu' - j\mu''$")
    ax_mu.set_title("(a) Causal magnetic susceptibility", pad=10)
    figstyle.place_legend_below(ax_mu, ncol=1, y_offset=-0.26)

    ax_hm.plot(
        f_ghz,
        np.asarray(d0["sigma_m_lor"]) * 1e-5,
        color=COLORS["blue"],
        lw=1.8,
        label=r"Lorentz $\sigma_m$ ($h_m = 3.46\times 10^5\,\Omega$/m)",
    )
    ax_hm.plot(
        f_ghz,
        np.asarray(d0["sigma_m_deb"]) * 1e-5,
        color=COLORS["teal"],
        lw=1.8,
        label=r"Debye $\sigma_m$ ($h_m = 9.47\times 10^5\,\Omega$/m)",
    )
    ax_hm.axhline(float(d0["czn_hm_ohm_m"]) * 1e-5, color=COLORS["blue"], ls=":", lw=1.1)
    ax_hm.plot(1.5, float(d0["czn_hm_ohm_m"]) * 1e-5, "*", color=COLORS["red"], ms=9.0, label=r"Peak $\sigma_m(\omega_0) = h_m$")
    ax_hm.set_xlabel("Frequency $f$ [GHz]")
    ax_hm.set_ylabel(r"$\sigma_m = \omega\mu_0\mu''$ [$10^5\,\Omega$/m]")
    ax_hm.set_title(r"(b) Hesitivity $h_m$ & $f(\mu) = \omega\mu_0/h_m$", pad=10)
    figstyle.place_legend_below(ax_hm, ncol=1, y_offset=-0.26)

    fs_ghz = np.asarray(d0["freqs_slab_ghz"], dtype=float)
    ax_slab.plot(fs_ghz, d0["lor_r_ana"], color=COLORS["blue"], lw=1.7, label=r"$|R|$ analytic / FIT ADE")
    ax_slab.plot(fs_ghz, d0["lor_r_fit"], "o", color=COLORS["blue"], ms=4.2)
    ax_slab.plot(fs_ghz, d0["lor_t_ana"], color=COLORS["teal"], lw=1.7, label=r"$|T|$ analytic / FIT ADE")
    ax_slab.plot(fs_ghz, d0["lor_t_fit"], "s", color=COLORS["teal"], ms=3.8)
    ax_slab.plot(fs_ghz, d0["lor_a_ana"], "--", color=COLORS["orange"], lw=1.6, label=r"Absorption $1-|R|^2-|T|^2$")
    ax_slab.plot(fs_ghz, d0["lor_a_ade"], "^", color=COLORS["red"], ms=4.2, label=r"ADE loss $\int\omega\mu_0\mu''|H|^2$")
    ax_slab.set_ylim(0.0, 1.05)
    ax_slab.set_xlabel("Frequency $f$ [GHz]")
    ax_slab.set_ylabel("Coefficient / absorption")
    ax_slab.set_title("(c) 1D FIT magnetic ADE slab", pad=10)
    figstyle.place_legend_below(ax_slab, ncol=1, y_offset=-0.26)

    out = fig_dir / "01_magnetic_ade_and_hesitivity.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_02_pec_image_theory_four_dipoles(d0: Any, d1: Any, fig_dir: Path) -> Path:
    """Render Figure 02: 4-orientation radiation resistance R_rad(h)/R_free over PEC and 2D field map."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=12.2, height_in=4.5, ncols=3, nrows=1)
    fig.subplots_adjust(wspace=0.34)
    ax_mag, ax_elec, ax_map = axes

    h_dense = np.asarray(d1["h_dense"], dtype=float)
    h_mid = np.asarray(d1["h_mid_lam"], dtype=float)
    h_node = np.asarray(d1["h_node_lam"], dtype=float)

    ax_mag.plot(h_dense, d1["r_hmd_dense"], color=COLORS["blue"], lw=1.9, label=r"HMD ($M_\parallel$ PEC): $\to 2$ at $h\to 0$")
    ax_mag.plot(h_mid, d1["r_hmd_fit"], "o", color=COLORS["blue"], ms=4.5, label="HMD 3D FIT")
    ax_mag.plot(h_dense, d1["r_vmd_dense"], color=COLORS["red"], lw=1.9, label=r"VMD ($M_\perp$ PEC): $\to 0$ at $h\to 0$")
    ax_mag.plot(h_node, d1["r_vmd_fit"], "s", color=COLORS["red"], ms=4.2, label="VMD 3D FIT")
    ax_mag.axhline(1.0, color=COLORS["grey"], ls=":", lw=1.0)
    ax_mag.axhline(2.0, color=COLORS["grey"], ls="--", lw=0.9)
    ax_mag.set_xlim(0.0, 0.55)
    ax_mag.set_ylim(0.0, 2.15)
    ax_mag.set_xlabel(r"Height above PEC ground $h / \lambda_0$")
    ax_mag.set_ylabel(r"Normalized $R_{\mathrm{rad}}(h) / R_{\mathrm{free}}$")
    ax_mag.set_title("(a) Magnetic dipoles over PEC", pad=10)
    figstyle.place_legend_below(ax_mag, ncol=1, y_offset=-0.26)

    ax_elec.plot(h_dense, d1["r_ved_dense"], color=COLORS["teal"], lw=1.9, label=r"VED ($J_\perp$ PEC): $\to 2$ at $h\to 0$")
    ax_elec.plot(h_mid, d1["r_ved_fit"], "o", color=COLORS["teal"], ms=4.5, label="VED 3D FIT")
    ax_elec.plot(h_dense, d1["r_hed_dense"], color=COLORS["orange"], lw=1.9, label=r"HED ($J_\parallel$ PEC): $\to 0$ at $h\to 0$")
    ax_elec.plot(h_node, d1["r_hed_fit"], "s", color=COLORS["orange"], ms=4.2, label="HED 3D FIT")
    ax_elec.axhline(1.0, color=COLORS["grey"], ls=":", lw=1.0)
    ax_elec.axhline(2.0, color=COLORS["grey"], ls="--", lw=0.9)
    ax_elec.set_xlim(0.0, 0.55)
    ax_elec.set_ylim(0.0, 2.15)
    ax_elec.set_xlabel(r"Height above PEC ground $h / \lambda_0$")
    ax_elec.set_ylabel(r"Normalized $R_{\mathrm{rad}}(h) / R_{\mathrm{free}}$")
    ax_elec.set_title("(b) Electric dipoles over PEC", pad=10)
    figstyle.place_legend_below(ax_elec, ncol=1, y_offset=-0.26)

    solver, sources, n2f, rod_box, _ = _build_schematic_objects(d0)
    short_names = {id(rod_box): "CZN rod"}
    x_mm = np.asarray(d0["xz_x_hx"], dtype=float) * 1e3
    z_mm = np.asarray(d0["xz_z_hx"], dtype=float) * 1e3
    hx_re = np.asarray(d0["xz_hx_real"], dtype=float)
    vmax = float(np.percentile(np.abs(hx_re), 97.0))
    if vmax <= 0.0:
        vmax = 1.0
    hx_norm = np.clip(hx_re / vmax, -1.0, 1.0)
    im = ax_map.pcolormesh(
        x_mm,
        z_mm,
        hx_norm.T,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-1.0,
        vmax=1.0,
        shading="auto",
    )
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.03, label=r"Normalized $\mathrm{Re}[H_x]$ [arb.]")
    schematic.overlay_geometry(
        ax_map,
        solver,
        plane="xz",
        unit="mm",
        structures=[rod_box],
        material_names=short_names,
        sources=sources[:1],
        monitors=[n2f],
        show_sources=True,
        show_monitors=True,
    )
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("$x$ [mm]")
    ax_map.set_ylabel("$z$ [mm]")
    ax_map.set_title(r"(c) Rod $\mathrm{Re}[H_x]$ on PEC", pad=10)

    out = fig_dir / "02_pec_image_theory_four_dipoles.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_03_volume_convention_and_lift(d0: Any, d2: Any, fig_dir: Path, d5: Any = None) -> Path:
    """Render Figure 03: Free-space vs half-space volume convention, 2x power/loss symmetry, and height lift."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=12.2, height_in=4.5, ncols=3, nrows=1)
    fig.subplots_adjust(wspace=0.34)
    ax_vol, ax_sym, ax_lift = axes

    rel_err_grid = float(d5["conv_rel_err"][0 if int(d2["quick"]) else 1]) if d5 is not None else 0.048
    yerr_db = float(10.0 * np.log10(1.0 + rel_err_grid))

    f_ghz = np.asarray(d2["freqs_ghz"], dtype=float)
    ax_vol.plot(
        f_ghz,
        10.0 * np.log10(d2["eff_ana_full"]),
        color=COLORS["blue"],
        lw=1.9,
        label=r"Bound Eq. (19b): $\mathrm{Vol} = 2\,V_{\mathrm{phys}} = V_{\mathrm{full}}$",
    )
    ax_vol.errorbar(
        f_ghz,
        10.0 * np.log10(d2["eff_half_pec"]),
        yerr=yerr_db,
        fmt="o",
        color=COLORS["blue"],
        ms=4.8,
        capsize=2.5,
        label=rf"FIT: Half-rod ($V_{{\mathrm{{phys}}}}$) on PEC ($\pm{yerr_db:.2f}$ dB)",
    )
    ax_vol.plot(
        f_ghz,
        10.0 * np.log10(d2["eff_full_free"]),
        "s",
        color=COLORS["teal"],
        ms=4.2,
        label=r"FIT: Full rod ($2V_{\mathrm{phys}}$) in free space",
    )
    ax_vol.plot(
        f_ghz,
        10.0 * np.log10(d2["eff_ana_phys"]),
        "--",
        color=COLORS["orange"],
        lw=1.8,
        label=r"Bound Eq. (19b): $\mathrm{Vol} = V_{\mathrm{phys}}$",
    )
    ax_vol.errorbar(
        f_ghz,
        10.0 * np.log10(d2["eff_phys_free"]),
        yerr=yerr_db,
        fmt="^",
        color=COLORS["orange"],
        ms=4.5,
        capsize=2.5,
        label=r"FIT: Rod ($V_{\mathrm{phys}}$) in free space",
    )
    ax_vol.set_xlabel("Frequency $f$ [GHz]")
    ax_vol.set_ylabel(r"Efficiency $10\log_{10} e$ [dB]")
    ax_vol.set_title("(a) Volume convention resolution", pad=10)
    figstyle.place_legend_below(ax_vol, ncol=1, y_offset=-0.26)

    ax_sym.plot(
        f_ghz,
        d2["ratio_prad_full_to_half"],
        "o-",
        color=COLORS["blue"],
        lw=1.7,
        ms=4.2,
        label=r"$P_{\mathrm{rad}}^{\mathrm{full,free}} / P_{\mathrm{rad}}^{\mathrm{half,PEC}}$",
    )
    ax_sym.plot(
        f_ghz,
        d2["ratio_ploss_full_to_half"],
        "s--",
        color=COLORS["teal"],
        lw=1.7,
        ms=4.2,
        label=r"$P_{\mathrm{loss}}^{\mathrm{full,free}} / P_{\mathrm{loss}}^{\mathrm{half,PEC}}$",
    )
    ax_sym.axhline(2.0, color=COLORS["red"], ls=":", lw=1.2, label="Image symmetry ratio = 2.0")
    ax_sym.set_ylim(1.75, 2.25)
    ax_sym.set_xlabel("Frequency $f$ [GHz]")
    ax_sym.set_ylabel("Full-space / half-space power")
    ax_sym.set_title("(b) Exact 2x power & loss symmetry", pad=10)
    figstyle.place_legend_below(ax_sym, ncol=1, y_offset=-0.26)

    h_lift = np.asarray(d2["h_lift_lam"], dtype=float)
    ax_lift.plot(
        h_lift,
        10.0 * np.log10(d2["eff_lift_ana"]),
        color=COLORS["blue"],
        lw=1.8,
        label=r"Closed form $\mathrm{Vol}_{\mathrm{eff}}(h) = F_{\mathrm{HMD}}(k_0 h)\,V_{\mathrm{phys}}$",
    )
    ax_lift.errorbar(
        h_lift,
        10.0 * np.log10(d2["eff_lift_fit"]),
        yerr=yerr_db,
        fmt="o",
        color=COLORS["red"],
        ms=4.8,
        capsize=2.5,
        label="3D FIT rod lifted to height $h$",
    )
    ax_lift.axhline(
        float(10.0 * np.log10(d2["eff_phys_free"][len(f_ghz) // 2])),
        color=COLORS["orange"],
        ls="--",
        lw=1.2,
        label=r"Free-space asymptote ($h \to \infty$)",
    )
    ax_lift.set_xlabel(r"Rod center height $h / \lambda_0$")
    ax_lift.set_ylabel(r"Efficiency $10\log_{10} e(h)$ [dB]")
    ax_lift.set_title(r"(c) $2V_{\mathrm{phys}} \to V_{\mathrm{phys}}$ lift transition", pad=10)
    figstyle.place_legend_below(ax_lift, ncol=1, y_offset=-0.26)

    out = fig_dir / "03_volume_convention_and_lift.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_04_equal_hm_collapse_vs_rumsey(d3: Any, fig_dir: Path, d5: Any = None) -> Path:
    """Render Figure 04: Equal-hm Lorentz (4 poles) and Debye (3 poles) efficiency collapse vs Rumsey-Weeks."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=12.2, height_in=4.5, ncols=3, nrows=1)
    fig.subplots_adjust(wspace=0.34)
    ax_td, ax_lor, ax_deb = axes

    rel_err_grid = float(d5["conv_rel_err"][0 if int(d3["quick"]) else 1]) if d5 is not None else 0.048
    yerr_db = float(10.0 * np.log10(1.0 + rel_err_grid))

    f_ghz = np.asarray(d3["freqs_ghz"], dtype=float)
    lor_td = np.asarray(d3["lor_tan_d"], dtype=float)
    lor_eff = np.asarray(d3["lor_eff_db"], dtype=float)
    lor_pol = np.asarray(d3["lor_eff_pol_db"], dtype=float)
    lor_rw = np.asarray(d3["lor_rw_db"], dtype=float)
    deb_eff = np.asarray(d3["deb_eff_db"], dtype=float)
    deb_pol = np.asarray(d3["deb_eff_pol_db"], dtype=float)
    deb_rw = np.asarray(d3["deb_rw_db"], dtype=float)

    palette = [COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["purple"]]
    markers = ["o", "s", "^", "D"]
    for i in range(lor_td.shape[0]):
        ax_td.plot(f_ghz, lor_td[i], marker=markers[i], color=palette[i], lw=1.6, ms=4.0, label=f"Lorentz pole {i+1}")
    ax_td.set_xlabel("Frequency $f$ [GHz]")
    ax_td.set_ylabel(r"Loss tangent $\tan\delta_m = \mu''/\mu'$")
    ax_td.set_title(r"(a) Diverse $\tan\delta_m$ at equal $h_m$", pad=10)
    figstyle.place_legend_below(ax_td, ncol=1, y_offset=-0.26)

    ax_lor.plot(f_ghz, d3["lor_ana_db"], color=figstyle.INK, lw=2.2, label=r"Bound $e_{\mathrm{MD}}$ ($\eta_{\mathrm{fill}}=1$)")
    ax_lor.plot(f_ghz, lor_pol[2], "--", color=COLORS["green"], lw=1.6, label=r"Core DFT $e_{\mathrm{pol}}$ ($\eta_{\mathrm{fill}}\approx 0.81$)")
    for i in range(lor_eff.shape[0]):
        ax_lor.errorbar(
            f_ghz,
            lor_eff[i],
            yerr=yerr_db if i == 2 else None,
            fmt=markers[i],
            color=palette[i],
            ms=4.5,
            capsize=2.2 if i == 2 else 0.0,
            label=f"3D FIT Lor-{i+1}",
        )
    ax_lor.plot(f_ghz, lor_rw[0], ":", color=COLORS["red"], lw=1.6, label="Rumsey-Weeks Eq. (8)")
    ax_lor.set_xlabel("Frequency $f$ [GHz]")
    ax_lor.set_ylabel(r"Efficiency $10\log_{10} e$ [dB]")
    ax_lor.set_title(f"(b) Lorentz collapse ({float(d3['lor_spread_db']):.2f} dB)", pad=10)
    figstyle.place_legend_below(ax_lor, ncol=1, y_offset=-0.26)

    ax_deb.plot(f_ghz, d3["deb_ana_db"], color=figstyle.INK, lw=2.2, label=r"Bound $e_{\mathrm{MD}}$ ($\eta_{\mathrm{fill}}=1$)")
    ax_deb.plot(f_ghz, deb_pol[1], "--", color=COLORS["green"], lw=1.6, label=r"Core DFT $e_{\mathrm{pol}}$ ($\eta_{\mathrm{fill}}\approx 0.81$)")
    for i in range(deb_eff.shape[0]):
        ax_deb.errorbar(
            f_ghz,
            deb_eff[i],
            yerr=yerr_db if i == 1 else None,
            fmt=markers[i],
            color=palette[i],
            ms=4.5,
            capsize=2.2 if i == 1 else 0.0,
            label=f"3D FIT Deb-{i+1}",
        )
    ax_deb.plot(f_ghz, deb_rw[2], ":", color=COLORS["red"], lw=1.6, label="Rumsey-Weeks Eq. (8)")
    ax_deb.set_xlabel("Frequency $f$ [GHz]")
    ax_deb.set_ylabel(r"Efficiency $10\log_{10} e$ [dB]")
    ax_deb.set_title(f"(c) Debye collapse ({float(d3['deb_spread_db']):.2f} dB)", pad=10)
    figstyle.place_legend_below(ax_deb, ncol=1, y_offset=-0.26)

    out = fig_dir / "04_equal_hm_collapse_vs_rumsey.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_05_dual_pair_and_ebwp(d4: Any, d5: Any, fig_dir: Path) -> Path:
    """Render Figure 05: Dual pair (horiz permeable rod vs vert dielectric monopole on PEC) and EBWP comparison."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=12.2, height_in=4.5, ncols=3, nrows=1)
    fig.subplots_adjust(wspace=0.34)
    ax_dual, ax_r, ax_ebwp = axes

    f_ghz = np.asarray(d4["freqs_ghz"], dtype=float)
    eff_m = np.asarray(d4["eff_horiz_mag_pec_db"], dtype=float)
    eff_e = np.asarray(d4["eff_vert_diel_pec_db"], dtype=float)
    eff_pmc = np.asarray(d4["eff_horiz_diel_pmc_db"], dtype=float)
    labels = [str(x) for x in d4["const_labels"]]
    cols = [COLORS["blue"], COLORS["teal"], COLORS["orange"]]

    for i, lbl in enumerate(labels):
        ax_dual.plot(f_ghz, eff_m[i], "-", color=cols[i], lw=1.8, label=f"Horiz $\\mu_r={lbl}$ PEC / $\\varepsilon_r$ PMC ($\\times$)")
        ax_dual.plot(f_ghz, eff_pmc[i], "x", color=cols[i], ms=5.0)
        ax_dual.plot(f_ghz, eff_e[i], "o", color=cols[i], ms=4.0, label=f"Vert $\\varepsilon_r={lbl}$ PEC" if i == 0 else None)
    ax_dual.set_xlabel("Frequency $f$ [GHz]")
    ax_dual.set_ylabel(r"Efficiency $10\log_{10} e$ [dB]")
    ax_dual.set_title("(a) PEC/PMC dual (up to grading) & PEC pair", pad=10)
    figstyle.place_legend_below(ax_dual, ncol=1, y_offset=-0.26)

    f_mhz = np.asarray(d5["freqs_mhz"], dtype=float)
    ax_r.semilogy(f_mhz, d5["r_in_md_ohm"], color=COLORS["blue"], lw=1.9, label="Permeable bar $R_{\\mathrm{in}}$")
    ax_r.semilogy(f_mhz, d5["r_rad_strip_ohm"], color=COLORS["red"], lw=1.9, label="Metal strip $R_{\\mathrm{rad}}$")
    ax_r.semilogy(f_mhz, d5["r_in_strip_ohm"], "--", color=COLORS["orange"], lw=1.6, label="Strip $R_{\\mathrm{rad}} + R_{\\mathrm{ohm,Cu}}$")
    ax_r.set_xlabel("Frequency $f$ [MHz]")
    ax_r.set_ylabel(r"Resistance [$\Omega$]")
    ax_r.set_title(r"(b) $>10^2\times$ resistance degradation", pad=10)
    figstyle.place_legend_below(ax_r, ncol=1, y_offset=-0.26)

    ax_ebwp.semilogy(f_mhz, d5["ebwp_md"], color=COLORS["blue"], lw=1.9, label="Permeable bar EBWP")
    ax_ebwp.semilogy(f_mhz, d5["ebwp_open"], color=COLORS["red"], lw=1.8, label="Open metal strip EBWP")
    ax_ebwp.semilogy(f_mhz, d5["ebwp_short"], "--", color=COLORS["teal"], lw=1.6, label="Shorted metal strip EBWP")
    ax_ebwp.set_xlabel("Frequency $f$ [MHz]")
    ax_ebwp.set_ylabel(r"$\mathrm{EBWP} = e_{\mathrm{rad}} \times \mathrm{FBW}_{2.5:1}$")
    ax_ebwp.set_title("(c) Efficiency-bandwidth product", pad=10)
    figstyle.place_legend_below(ax_ebwp, ncol=1, y_offset=-0.26)

    out = fig_dir / "05_dual_pair_and_ebwp.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_06_finite_ground_and_convergence(d5: Any, fig_dir: Path) -> Path:
    """Render Figure 06: Finite PEC ground pattern tilt, Ikonen resonant patch Q vs flux rod, and log-log convergence."""
    figstyle.apply_style()
    fig, axes = figstyle.new_figure(width_in=12.2, height_in=4.5, ncols=3, nrows=1)
    fig.subplots_adjust(wspace=0.34)
    ax_gnd, ax_iko, ax_conv = axes

    theta_deg = np.asarray(d5["theta_deg"], dtype=float)
    pats = np.asarray(d5["patterns_finite_gnd"], dtype=float)
    w_lams = np.asarray(d5["gnd_widths_lam"], dtype=float)
    d_inf = float(d5["inf_gnd_zenith_dir"]) if "inf_gnd_zenith_dir" in d5 else 2.956
    cols = [COLORS["orange"], COLORS["teal"], COLORS["blue"], COLORS["purple"]]
    for i, wl in enumerate(w_lams):
        ax_gnd.plot(theta_deg, pats[i], color=cols[i], lw=1.6, label=f"$L_{{\\mathrm{{gnd}}}} = {wl:.2f}\\,\\lambda_0$")
    ax_gnd.axhline(d_inf, color=figstyle.INK, ls="--", lw=1.2, label=rf"$\infty$-PEC zenith $D_\infty(0^\circ)={d_inf:.2f}$")
    ax_gnd.axvline(90.0, color=COLORS["grey"], ls=":", lw=1.0, label="Horizon ($90^\\circ$)")
    ax_gnd.set_xlim(0.0, 180.0)
    ax_gnd.set_xlabel(r"Polar angle $\theta$ [deg] ($yz$ plane)")
    ax_gnd.set_ylabel(r"Directivity $D(\theta)$ [linear]")
    ax_gnd.set_title("(a) Finite PEC edge diffraction", pad=10)
    figstyle.place_legend_below(ax_gnd, ncol=1, y_offset=-0.26)

    td_p = np.asarray(d5["tan_d_patch"], dtype=float)
    ax_iko.semilogx(td_p, 10.0 * np.log10(d5["eff_patch"]), color=COLORS["red"], lw=1.8, label=r"Resonant patch $e = Q_{\mathrm{tot}}/Q_{\mathrm{rad}}$")
    ax_iko.semilogx(td_p, 10.0 * np.log10(d5["fbw_patch"]), "--", color=COLORS["orange"], lw=1.6, label=r"Resonant patch $\mathrm{FBW} \propto 1/Q_{\mathrm{tot}}$")
    ax_iko.semilogx(td_p, 10.0 * np.log10(d5["ebwp_patch"]), ":", color=COLORS["blue"], lw=1.8, label=r"Resonant patch $\mathrm{EBWP}$ (invariant)")
    ax_iko.set_xlabel(r"Magnetic loss tangent $\tan\delta_m$")
    ax_iko.set_ylabel("Metric [dB]")
    ax_iko.set_title("(b) Ikonen (2006) cavity patch limit", pad=10)
    figstyle.place_legend_below(ax_iko, ncol=1, y_offset=-0.26)

    dx_mm = np.asarray(d5["dx_conv_mm"], dtype=float)
    rel_err = np.asarray(d5["conv_rel_err"], dtype=float) * 100.0
    bal_err = np.asarray(d5["conv_bal_err"], dtype=float) * 100.0
    p_obs = float(d5["conv_p_obs"]) if "conv_p_obs" in d5 else 2.0
    # Anchor the O(dx^2) slope guide at the single coarse point dx_mm[0] only
    ref_o2 = rel_err[0] * ((dx_mm / dx_mm[0]) ** 2)

    ax_conv.loglog(
        dx_mm,
        rel_err,
        "o-",
        color=COLORS["blue"],
        lw=1.8,
        ms=5.0,
        label=rf"$|e(\Delta) - e_{{\mathrm{{Rich}}}}| / e_{{\mathrm{{Rich}}}}$ ($p_{{\mathrm{{obs}}}}={p_obs:.2f}$)",
    )
    ax_conv.loglog(dx_mm, ref_o2, "--", color=COLORS["grey"], lw=1.4, label=r"$\mathcal{O}(\Delta^2)$ guide (anchored at 6 mm)")
    ax_conv.loglog(dx_mm, np.maximum(bal_err, 1e-4), "s:", color=COLORS["teal"], lw=1.5, ms=4.2, label=r"Power balance error $|W_{\mathrm{src}} - P_{\mathrm{tot}}|$")
    ax_conv.set_xticks([3.0, 4.0, 6.0])
    ax_conv.set_xticklabels(["3.0 (Fine)", "4.0 (Med)", "6.0 (Coarse)"])
    ax_conv.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_conv.set_xlabel(r"Core mesh spacing $\Delta_{\mathrm{fine}}$ [mm]")
    ax_conv.set_ylabel("Relative error [%]")
    ax_conv.set_title("(c) 3-resolution FIT convergence", pad=10)
    figstyle.place_legend_below(ax_conv, ncol=1, y_offset=-0.26)

    out = fig_dir / "06_finite_ground_and_convergence.png"
    return figstyle.save(fig, out, close=True, auto_legend_below=False)


def plot_07_hmd_vs_hed_pec_ground_gif(d1: Any, fig_dir: Path) -> Path:
    """Render Figure 07: Animated time-domain GIF contrasting HMD (+image) vs HED (-image) over PEC."""
    figstyle.apply_style()
    y_lam = np.asarray(d1["gif_y_lam"], dtype=float)
    z_lam = np.asarray(d1["gif_z_lam"], dtype=float)
    f_hmd = np.asarray(d1["gif_hmd_frames"], dtype=float)
    f_hed = np.asarray(d1["gif_hed_frames"], dtype=float)
    t_per = np.asarray(d1["gif_times_per"], dtype=float)
    h_hmd_gif = float(z_lam[0])
    h_hed_gif = float(2.0 * z_lam[0])

    r_grid = np.sqrt(y_lam[:, None] ** 2 + z_lam[None, :] ** 2)
    far_mask = r_grid >= 0.12
    s_hmd = f_hmd * r_grid[None, :, :]
    s_hed = f_hed * r_grid[None, :, :]
    vmax = float(np.percentile(np.abs(s_hmd[:, far_mask]), 98.5))
    if vmax <= 0.0:
        vmax = 1.0

    pil_images: list[Image.Image] = []
    for idx in range(s_hmd.shape[0]):
        fig, axes = figstyle.new_figure(width_in=7.2, height_in=3.4, ncols=2, nrows=1)
        fig.set_dpi(140)
        ax_m, ax_e = axes

        im0 = ax_m.pcolormesh(
            y_lam,
            z_lam,
            s_hmd[idx].T,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
        )
        fig.colorbar(im0, ax=ax_m, fraction=0.046, pad=0.03, label=r"$(r/\lambda_0)\,\eta_0 H_x$")
        ax_m.axhline(0.0, color=figstyle.INK, lw=2.2)
        ax_m.plot(0.0, h_hmd_gif, "*", color=COLORS["blue"], ms=8.0)
        ax_m.set_aspect("equal")
        ax_m.set_xlabel(r"$y / \lambda_0$")
        ax_m.set_ylabel(r"$z / \lambda_0$ (PEC at $z=0$)")
        ax_m.set_title(f"(a) HMD $M_x \\parallel$ PEC ($t={t_per[idx]:.2f}\\,T_0$, $\\times 2$)")

        im1 = ax_e.pcolormesh(
            y_lam,
            z_lam,
            s_hed[idx].T,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
        )
        fig.colorbar(im1, ax=ax_e, fraction=0.046, pad=0.03, label=r"$(r/\lambda_0)\,E_x$")
        ax_e.axhline(0.0, color=figstyle.INK, lw=2.2)
        ax_e.plot(0.0, h_hed_gif, "*", color=COLORS["red"], ms=8.0)
        ax_e.set_aspect("equal")
        ax_e.set_xlabel(r"$y / \lambda_0$")
        ax_e.set_ylabel(r"$z / \lambda_0$ (PEC at $z=0$)")
        ax_e.set_title(f"(b) HED $J_x \\parallel$ PEC ($t={t_per[idx]:.2f}\\,T_0$, cancelled)")

        figstyle.finish(fig, max_panels=2, auto_legend_below=False)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_images.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        plt.close(fig)

    out = fig_dir / "07_hmd_vs_hed_pec_ground.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_images[0].save(
        out,
        save_all=True,
        append_images=pil_images[1:],
        duration=110,
        loop=0,
        optimize=True,
    )
    return out


def generate_all_figures(data_dir: Path, fig_dir: Path) -> list[Path]:
    """Load ``data_dir/part*.npz`` and render all Tutorial 4.8 figures and GIFs into ``fig_dir``."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    d0 = np.load(data_dir / "part0_magnetic_ade.npz")
    d1 = np.load(data_dir / "part1_image_theory.npz")
    d2 = np.load(data_dir / "part2_volume_convention.npz")
    d3 = np.load(data_dir / "part3_equal_hm.npz")
    d4 = np.load(data_dir / "part4_dual_pair.npz")
    d5 = np.load(data_dir / "part5_ebwp_and_convergence.npz")

    with plt.ioff():
        fig00a = plot_00_setup_schematic(d0, fig_dir)
        out = [
            fig00a,
            fig_dir / "00b_setup_mesh_and_excitation.png",
            plot_01_magnetic_ade_and_hesitivity(d0, fig_dir),
            plot_02_pec_image_theory_four_dipoles(d0, d1, fig_dir),
            plot_03_volume_convention_and_lift(d0, d2, fig_dir, d5=d5),
            plot_04_equal_hm_collapse_vs_rumsey(d3, fig_dir, d5=d5),
            plot_05_dual_pair_and_ebwp(d4, d5, fig_dir),
            plot_06_finite_ground_and_convergence(d5, fig_dir),
            plot_07_hmd_vs_hed_pec_ground_gif(d1, fig_dir),
        ]
    plt.close("all")
    return out


def build_summary_json(data_dir: Path) -> dict[str, Any]:
    """Build ``data/summary.json`` from the saved task archives for the Quantitative Summary table."""
    d0 = np.load(data_dir / "part0_magnetic_ade.npz")
    d1 = np.load(data_dir / "part1_image_theory.npz")
    d2 = np.load(data_dir / "part2_volume_convention.npz")
    d3 = np.load(data_dir / "part3_equal_hm.npz")
    d4 = np.load(data_dir / "part4_dual_pair.npz")
    d5 = np.load(data_dir / "part5_ebwp_and_convergence.npz")

    idx_mid_d0 = len(d0["freqs_rod_ghz"]) // 2
    idx_mid_d2 = len(d2["freqs_ghz"]) // 2
    idx_mid_d3 = len(d3["freqs_ghz"]) // 2
    idx_mid_d4 = len(d4["freqs_ghz"]) // 2

    summary = {
        "c_yousefi_exact_ohm": C_YOUSEFI_EXACT,
        "c_yousefi_120pi_ohm": C_YOUSEFI_120PI,
        "czn_hm_ohm_m": float(d0["czn_hm_ohm_m"]),
        "rel_err_fmu_lor": float(d0["rel_err_fmu_lor"]),
        "rel_err_fmu_deb": float(d0["rel_err_fmu_deb"]),
        "slab_lor_max_err": float(d0["slab_lor_max_err"]),
        "slab_deb_max_err": float(d0["slab_deb_max_err"]),
        "rod_energy_balance_rel_err": float(d0["balance_rel_err_a"][idx_mid_d0]),
        "rod_mutated_balance_rel_err": float(d0["mutated_balance_rel_err_a"][idx_mid_d0]),
        "equal_hm_pole_a_eff_db": float(d0["eff_pole_a_db"][idx_mid_d0]),
        "equal_hm_pole_b_eff_db": float(d0["eff_pole_b_db"][idx_mid_d0]),
        "equal_hm_diff_db": float(d0["equal_hm_diff_db"][idx_mid_d0]),
        "mutated_hm_loss_ratio_factor": float(d0["mutated_hm_loss_ratio_factor"][idx_mid_d0]),
        "hmd_low_h_lam": float(d1["h_mid_lam"][0]),
        "hmd_low_h_fit": float(d1["r_hmd_fit"][0]),
        "hmd_low_h_ana": float(d1["r_hmd_ana"][0]),
        "hmd_low_h_rel_err_pct": float(abs(d1["r_hmd_fit"][0] - d1["r_hmd_ana"][0]) / d1["r_hmd_ana"][0] * 100.0),
        "vmd_low_h_lam": float(d1["h_node_lam"][0]),
        "vmd_low_h_fit": float(d1["r_vmd_fit"][0]),
        "vmd_low_h_ana": float(d1["r_vmd_ana"][0]),
        "hed_low_h_fit": float(d1["r_hed_fit"][0]),
        "hed_low_h_ana": float(d1["r_hed_ana"][0]),
        "ved_low_h_fit": float(d1["r_ved_fit"][0]),
        "ved_low_h_ana": float(d1["r_ved_ana"][0]),
        "vol_half_pec_eff": float(d2["eff_half_pec"][idx_mid_d2]),
        "vol_full_free_eff": float(d2["eff_full_free"][idx_mid_d2]),
        "vol_phys_free_eff": float(d2["eff_phys_free"][idx_mid_d2]),
        "vol_ana_full_eff": float(d2["eff_ana_full"][idx_mid_d2]),
        "vol_ana_phys_eff": float(d2["eff_ana_phys"][idx_mid_d2]),
        "ratio_prad_full_to_half": float(d2["ratio_prad_full_to_half"][idx_mid_d2]),
        "ratio_ploss_full_to_half": float(d2["ratio_ploss_full_to_half"][idx_mid_d2]),
        "lor_family_spread_db": float(d3["lor_spread_db"]),
        "deb_family_spread_db": float(d3["deb_spread_db"]),
        "lor_eta_fill_f0": float(d3["lor_eta_fill"][2, idx_mid_d3]),
        "lor_eff_pol_f0_db": float(d3["lor_eff_pol_db"][2, idx_mid_d3]),
        "lor_eff_tot_f0_db": float(d3["lor_eff_db"][2, idx_mid_d3]),
        "lor_eff_ana_f0_db": float(d3["lor_ana_db"][idx_mid_d3]),
        "deb_eta_fill_f0": float(d3["deb_eta_fill"][1, idx_mid_d3]),
        "deb_eff_pol_f0_db": float(d3["deb_eff_pol_db"][1, idx_mid_d3]),
        "deb_eff_tot_f0_db": float(d3["deb_eff_db"][1, idx_mid_d3]),
        "deb_eff_ana_f0_db": float(d3["deb_ana_db"][idx_mid_d3]),
        "rod_to_loop_prad_ratio_f0": float(d3["rod_to_loop_prad_ratio"][idx_mid_d3]),
        "rod_to_loop_moment_ratio_f0": float(np.sqrt(d3["rod_to_loop_prad_ratio"][idx_mid_d3])),
        "rod_only_to_loop_moment_ratio_f0": float(np.sqrt(d3["rod_to_loop_prad_ratio"][idx_mid_d3]) - 1.0),
        "loop_in_phase_prad_boost_db": float(
            20.0 * np.log10(1.0 + 1.0 / np.sqrt(d3["rod_to_loop_prad_ratio"][idx_mid_d3]))
        ),
        "loop_in_phase_eff_boost_db": float(
            20.0
            * np.log10(1.0 + 1.0 / np.sqrt(d3["rod_to_loop_prad_ratio"][idx_mid_d3]))
            * (1.0 - float(d2["eff_half_pec"][idx_mid_d2]))
        ),
        "lor_eff_tot_minus_pol_f0_db": float(d3["lor_eff_db"][2, idx_mid_d3] - d3["lor_eff_pol_db"][2, idx_mid_d3]),
        "lor_max_excess_over_bound_db": float(np.max(d3["lor_eff_db"] - d3["lor_ana_db"][None, :])),
        "deb_max_excess_over_bound_db": float(np.max(d3["deb_eff_db"] - d3["deb_ana_db"][None, :])),
        "dual_300_mag_db": float(d4["eff_horiz_mag_pec_db"][0, idx_mid_d4]),
        "dual_300_pmc_db": float(d4["eff_horiz_diel_pmc_db"][0, idx_mid_d4]),
        "dual_300_diel_db": float(d4["eff_vert_diel_pec_db"][0, idx_mid_d4]),
        "dual_80_mag_db": float(d4["eff_horiz_mag_pec_db"][1, idx_mid_d4]),
        "dual_80_pmc_db": float(d4["eff_horiz_diel_pmc_db"][1, idx_mid_d4]),
        "dual_80_diel_db": float(d4["eff_vert_diel_pec_db"][1, idx_mid_d4]),
        "dual_20_mag_db": float(d4["eff_horiz_mag_pec_db"][2, idx_mid_d4]),
        "dual_20_pmc_db": float(d4["eff_horiz_diel_pmc_db"][2, idx_mid_d4]),
        "dual_20_diel_db": float(d4["eff_vert_diel_pec_db"][2, idx_mid_d4]),
        "dual_pmc_max_diff_f0_db": float(np.max(d4["dual_pmc_diff_f0_db"])),
        "dual_pmc_coarse_max_diff_f0_db": float(np.max(d4["dual_pmc_coarse_diff_f0_db"])),
        "dual_pec_max_diff_f0_db": float(np.max(d4["dual_pec_diff_f0_db"])),
        "fit_mbar_to_strip_ratio": float(d5["fit_mbar_to_strip_ratio"]),
        "ebwp_min_advantage_db": float(np.min(d5["ebwp_advantage_db"])),
        "ebwp_max_advantage_db": float(np.max(d5["ebwp_advantage_db"])),
        "finite_gnd_h_lam": float(d5["h_gnd_lam"]),
        "inf_gnd_r_ratio": float(d5["inf_gnd_r_ratio"]),
        "inf_gnd_zenith_dir": float(d5["inf_gnd_zenith_dir"]),
        "finite_gnd_r_ratio_030": float(d5["r_ratio_finite_gnd"][0]),
        "finite_gnd_r_ratio_110": float(d5["r_ratio_finite_gnd"][-1]),
        "finite_gnd_zenith_dir_030": float(d5["zenith_dir_finite_gnd"][0]),
        "finite_gnd_zenith_dir_110": float(d5["zenith_dir_finite_gnd"][-1]),
        "finite_gnd_tilt_deg_030": float(d5["tilt_deg_finite_gnd"][0]),
        "finite_gnd_tilt_deg_110": float(d5["tilt_deg_finite_gnd"][-1]),
        "conv_eff_coarse": float(d5["conv_eff"][0]),
        "conv_eff_medium": float(d5["conv_eff"][1]),
        "conv_eff_fine": float(d5["conv_eff"][2]),
        "conv_p_obs": float(d5["conv_p_obs"]),
        "conv_eff_rich": float(d5["conv_eff_rich"]),
        "conv_coarse_rel_err_pct": float(d5["conv_rel_err"][0] * 100.0),
        "conv_medium_rel_err_pct": float(d5["conv_rel_err"][1] * 100.0),
        "conv_fine_rel_err_pct": float(d5["conv_rel_err"][2] * 100.0),
        "conv_coarse_cpw_mat": float(d5["conv_cpw_mat"][0]),
        "conv_medium_cpw_mat": float(d5["conv_cpw_mat"][1]),
        "conv_fine_cpw_mat": float(d5["conv_cpw_mat"][2]),
        "conv_medium_cells_per_skin": float(d5["conv_cells_per_skin"][1]),
        "conv_fine_cells_per_skin": float(d5["conv_cells_per_skin"][2]),
        "timings_s": {
            "ade_and_hesitivity": float(d0["wall_s"]),
            "image_theory": float(d1["wall_s"]),
            "volume_convention": float(d2["wall_s"]),
            "equal_hm_families": float(d3["wall_s"]),
            "dual_pair": float(d4["wall_s"]),
            "metal_vs_permeable_and_convergence": float(d5["wall_s"]),
        },
        "peak_rss_mb": {
            "ade_and_hesitivity": float(d0["peak_rss_mb"]),
            "image_theory": float(d1["peak_rss_mb"]),
            "volume_convention": float(d2["peak_rss_mb"]),
            "equal_hm_families": float(d3["peak_rss_mb"]),
            "dual_pair": float(d4["peak_rss_mb"]),
            "metal_vs_permeable_and_convergence": float(d5["peak_rss_mb"]),
        },
    }
    out_json = data_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_all_tasks(data_dir: Path, fig_dir: Path, quick: bool = False, overwrite: bool = False) -> dict[str, Any]:
    """Execute all 6 tasks sequentially (one FIT simulation at a time), save data, and render figures."""
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runners = {
        "part0_magnetic_ade": run_task_00_ade_and_hesitivity,
        "part1_image_theory": run_task_01_image_theory,
        "part2_volume_convention": run_task_02_volume_convention,
        "part3_equal_hm": run_task_03_equal_hm_families,
        "part4_dual_pair": run_task_04_dual_pair,
        "part5_ebwp_and_convergence": run_task_05_metal_vs_permeable_and_convergence,
    }
    for name, fn in runners.items():
        npz_path = data_dir / f"{name}.npz"
        if npz_path.exists() and not overwrite:
            existing = np.load(npz_path)
            if int(existing["quick"]) == int(quick):
                continue
        res = fn(quick=quick)
        np.savez_compressed(npz_path, **res)

    summary = build_summary_json(data_dir)
    if overwrite or not (fig_dir / "07_hmd_vs_hed_pec_ground.gif").exists():
        generate_all_figures(data_dir, fig_dir)
    plt.close("all")
    return summary
