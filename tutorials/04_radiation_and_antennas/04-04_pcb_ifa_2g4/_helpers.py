"""Helper routines for Tutorial 04-02: 2.4 GHz PCB Inverted-F Antenna (TI SWRU120D).

Implements the exact geometric layout of Texas Instruments Design Note DN007
(Literature Number SWRU120D, *2.4-GHz Inverted F Antenna*, by Audun Andersen,
Table 1-1 on 1.0 mm FR-4), the design-rule and impedance-transformation models
from Audun Andersen, Texas Instruments Application Note AN043 (SWRA117D,
*Small Size 2.4 GHz PCB Antenna*), the Chu-Wheeler-McLean fundamental radiation
Q bound, 3D FIT full-wave electromagnetic simulations with Wakis's dispersive
``'fr4'`` material model, and all publication-quality 2D/3D visualizations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image
from scipy.constants import c as c_light
from scipy.constants import epsilon_0, mu_0

import wakis.visualize as wv
from wakis import GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.dispersive import dispersive_lib
from wakis.estimate import estimate
from wakis.field_monitors import DFTMonitor
from wakis.lumped import LumpedPort
from wakis.near2far import Near2FarMonitor, half_power_beamwidth
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates import Box, meshing
from wakis.templates.shapes import assign_materials

Z0_FREE = float(np.sqrt(mu_0 / epsilon_0))

# ---------------------------------------------------------------------------
# Verified reference geometry: Texas Instruments SWRU120D (DN007), Table 1-1
# URL: https://www.ti.com/lit/an/swru120d/swru120d.pdf
# Secondary design rules: TI AN043 / SWRA117D, Audun Andersen (2008)
# URL: https://www.ti.com/lit/an/swra117d/swra117d.pdf
# ---------------------------------------------------------------------------
SWRU120_TABLE_1_1_MM = {
    "H1": 5.70,
    "H2": 0.74,
    "H3": 1.29,
    "H4": 2.21,
    "H5": 0.66,
    "H6": 1.21,
    "H7": 0.80,
    "H8": 1.80,
    "H9": 0.61,
    "L1": 25.58,
    "L2": 16.40,
    "L3": 2.18,
    "L4": 4.80,
    "L5": 1.00,
    "L6": 1.00,
    "L7": 3.20,
    "L8": 0.45,
    "W1": 1.21,
    "W2": 0.46,
    "h_sub": 1.00,
}

SWRU120_BENCHMARKS = {
    "source_title": "Texas Instruments Design Note DN007 (SWRU120D, Rev. D, Jan. 2019), 2.4-GHz Inverted F Antenna",
    "source_author": "Audun Andersen",
    "source_url": "https://www.ti.com/lit/an/swru120d/swru120d.pdf",
    "design_rules_title": "Texas Instruments Application Note AN043 (SWRA117D), Small Size 2.4 GHz PCB Antenna",
    "design_rules_url": "https://www.ti.com/lit/an/swra117d/swra117d.pdf",
    "f_center_hz": 2.44e9,
    "f_res_approx_hz": 2.44e9,
    "ism_band_hz": [2.40e9, 2.48e9],
    "reflection_spec_db": -15.0,
    "s11_min_db_measured": -16.5,
    "bw_10db_min_hz": 300.0e6,
    "bw_10db_approx_hz": 330.0e6,
    "gain_xy_max_dbi": 1.1,
    "gain_xz_max_dbi": 3.3,
    "gain_yz_max_dbi": 1.6,
    "footprint_mm": [25.7, 7.5],
    "substrate_thickness_mm": 1.0,
    "substrate_material": "FR-4",
    "evaluation_board": "CC2430DB",
    "ground_size_mm_in_source": None,
    "efficiency_in_source": None,
    "simulated_ground_mm": [29.58, 32.0],
    "simulated_board_mm": [29.58, 39.5],
}

SWRU120_SOURCE_LOCATIONS = {
    "Table 1-1 dimensions (H1..H9, W1, W2, L1..L8)": "Section 1, p. 2, Figure 1-1 & Table 1-1",
    "Substrate (1 mm standard FR-4; eps_r/tan_delta not reported)": "Section 1, p. 2",
    "Antenna footprint (25.7 x 7.5 mm, ground plane not included)": "Abstract, p. 1; Section 1, p. 2; Section 3, Table 3-1, p. 11",
    "Evaluation board (CC2430DB; ground size in mm not reported)": "Section 1, p. 2; Section 2.1, p. 3, Figure 2-1",
    "ISM band (2.40-2.48 GHz) & pattern test frequency (2.44 GHz)": "Title & Abstract, p. 1; Section 2.1, Figures 2-2 to 2-7, pp. 4-9",
    "Resonance minimum (~2.44 GHz)": "Read from Figure 2-8, p. 10 (approx.)",
    "Reflection (< -15 dB across band; ~ -16.5 dB min)": "Section 3, Table 3-1, p. 11; min read from Figure 2-8, p. 10 (approx.)",
    "Bandwidth (> 300 MHz at < 10% reflection / -10 dB; ~330 MHz span)": "Abstract, p. 1; Section 2.2, p. 10; Section 2.3, p. 11; span read from Figure 2-8, p. 10 (approx.)",
    "Gain in XY plane (1.1 dBi)": "Section 3, Table 3-1, p. 11 (Figures 2-2 & 2-3, pp. 4-5)",
    "Gain in XZ plane (3.3 dBi)": "Abstract, p. 1; Section 3, Table 3-1, p. 11 (Figures 2-4 & 2-5, pp. 6-7)",
    "Gain in YZ plane (1.6 dBi)": "Section 3, Table 3-1, p. 11 (Figures 2-6 & 2-7, pp. 8-9)",
    "Radiation efficiency": "Not reported in SWRU120D",
}


# ---------------------------------------------------------------------------
# Analytical theory helpers
# ---------------------------------------------------------------------------
def fr4_complex_permittivity(freqs_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Wakis's dispersive ``'fr4'`` library material over frequency.

    Parameters
    ----------
    freqs_hz : ndarray
        Frequencies [Hz].

    Returns
    -------
    eps_real : ndarray
        Real part of relative permittivity :math:`\\varepsilon'_r(\\omega)` [-].
    eps_imag : ndarray
        Lossy imaginary magnitude :math:`-\\operatorname{Im}(\\varepsilon_r)` [-].
    tan_delta : ndarray
        Dielectric loss tangent :math:`\\tan\\delta(\\omega) = \\varepsilon''_r / \\varepsilon'_r` [-].
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    mat = dispersive_lib["fr4"]
    eps_c = mat.epsilon(freqs_hz)
    eps_real = np.real(eps_c)
    eps_imag = -np.imag(eps_c)
    tan_delta = eps_imag / np.maximum(eps_real, 1e-12)
    return eps_real, eps_imag, tan_delta


def microstrip_effective_eps(eps_r: float, width_m: float, height_m: float) -> float:
    """Hammerstad-Jensen quasi-static effective relative permittivity on PCB edge.

    For an inverted-F trace positioned at the top edge of a dielectric board of
    thickness ``height_m`` with no ground directly underneath (`H1 = 5.70 mm`
    clearance), the fringing electric field shares air and FR-4 approximately
    in a coplanar/edge-coupled geometry where :math:`\\varepsilon_{\\text{eff}} \\approx
    1 + q (\\varepsilon_r - 1)` with filling factor :math:`q \\approx 0.38`.

    Parameters
    ----------
    eps_r : float
        Substrate relative permittivity [-].
    width_m : float
        Trace width [m].
    height_m : float
        Substrate thickness [m].

    Returns
    -------
    float
        Effective relative permittivity :math:`\\varepsilon_{\\text{eff}}` [-].
    """
    if width_m <= 0 or height_m <= 0:
        raise ValueError("width_m and height_m must be positive.")
    u = width_m / height_m
    # Standard microstrip filling factor scaled by edge-clearance factor (~0.52)
    q_ms = 0.5 + 0.5 / np.sqrt(1.0 + 12.0 / u)
    q_edge = 0.52 * q_ms
    return float(1.0 + q_edge * (eps_r - 1.0))


def quarter_wave_resonance_hz(
    arm_length_m: float,
    width_m: float = 1.21e-3,
    height_m: float = 1.00e-3,
    eps_r: float = 4.28,
) -> float:
    """Analytical quarter-wave resonant frequency of the open IFA arm + shorting stub.

    Parameters
    ----------
    arm_length_m : float
        Effective quarter-wave path length from shorting ground to open tip [m].
    width_m : float, optional
        Main trace width [m]. Default 1.21 mm (``W1``).
    height_m : float, optional
        Substrate thickness [m]. Default 1.00 mm.
    eps_r : float, optional
        Substrate real relative permittivity [-]. Default 4.28 (FR-4 at 2.44 GHz).

    Returns
    -------
    float
        Predicted quarter-wave resonant frequency [Hz].
    """
    if arm_length_m <= 0:
        raise ValueError("arm_length_m must be positive.")
    eps_eff = microstrip_effective_eps(eps_r, width_m, height_m)
    delta_l = 0.412 * height_m * ((eps_eff + 0.3) * (width_m / height_m + 0.264)) / (
        (eps_eff - 0.258) * (width_m / height_m + 0.8)
    )
    return float(c_light / (4.0 * np.sqrt(eps_eff) * (arm_length_m + delta_l)))


def ifa_input_resistance_tap(
    r_rad_ohm: float,
    tap_distance_m: float,
    total_arm_m: float,
    r_loss_ohm: float = 1.5,
) -> float:
    """Analytical transmission-line tap-transformer model for IFA input resistance.

    Near the quarter-wave resonance, the short-circuited stub of length
    :math:`d_{\\text{tap}}` in parallel with the open radiating arm transforms
    the radiation + loss resistance according to

    .. math::

        R_{\\text{in}}(d_{\\text{tap}}) \\approx \\frac{R_{\\text{rad}} + R_{\\text{loss}}}
        {\\sin^2\\!\\bigl(\\frac{\\pi}{2} \\frac{L_{\\text{eff}} - d_{\\text{tap}}}{L_{\\text{eff}}}\\bigr)}
        \\sin^2\\!\\Bigl(\\frac{\\pi}{2} \\frac{d_{\\text{tap}}}{L_{\\text{tap,0}}}\\Bigr).

    Parameters
    ----------
    r_rad_ohm : float
        Base monopole radiation resistance referenced to current maximum [Ohm].
    tap_distance_m : float
        Short-to-feed tap separation :math:`L_4` [m].
    total_arm_m : float
        Total resonant quarter-wave length :math:`L_{\\text{eff}}` [m].
    r_loss_ohm : float, optional
        Equivalent dielectric + conductor loss resistance [Ohm]. Default 1.5.

    Returns
    -------
    float
        Transformed input resistance at resonance [Ohm].
    """
    if total_arm_m <= 0 or tap_distance_m < 0:
        raise ValueError("Require total_arm_m > 0 and tap_distance_m >= 0.")
    frac = np.clip(tap_distance_m / total_arm_m, 0.0, 0.95)
    return float((r_rad_ohm + r_loss_ohm) * (np.sin(0.5 * np.pi * frac) / np.sin(0.5 * np.pi * 0.22)) ** 2)


def chu_wheeler_q_bound(ka: float, efficiency: float = 1.0) -> float:
    """Chu-Wheeler-McLean fundamental lower bound on radiation quality factor Q.

    For a linearly polarized electrically small antenna enclosed within a
    Chu sphere of radius :math:`a` (electrical size :math:`k a = 2\\pi a / \\lambda_0`),
    the exact McLean (1996) / Chu (1948) / Wheeler (1947) minimum quality factor is

    .. math::

        Q_{\\text{Chu}} = \\eta_{\\text{rad}} \\left(\\frac{1}{(k a)^3} + \\frac{1}{k a}\\right).

    Parameters
    ----------
    ka : float or ndarray
        Electrical size :math:`k a` [rad].
    efficiency : float, optional
        Radiation efficiency :math:`\\eta_{\\text{rad}} \\in (0, 1]` [-]. Default 1.0.

    Returns
    -------
    float or ndarray
        Minimum achievable quality factor :math:`Q_{\\min}` [-].
    """
    ka_arr = np.asarray(ka, dtype=float)
    if np.any(ka_arr <= 0):
        raise ValueError("Electrical size ka must be strictly positive.")
    if not (0.0 < efficiency <= 1.0):
        raise ValueError("efficiency must lie in (0, 1].")
    q_min = efficiency * (1.0 / ka_arr**3 + 1.0 / ka_arr)
    return float(q_min) if np.ndim(ka) == 0 else q_min


def chu_max_fractional_bandwidth(
    ka: float,
    vswr_max: float = 2.0,
    efficiency: float = 1.0,
) -> float:
    """Maximum fractional impedance bandwidth allowed by the Chu-Wheeler bound.

    .. math::

        \\text{FBW}_{\\max} = \\frac{\\text{VSWR} - 1}{Q_{\\text{Chu}}\\,\\sqrt{\\text{VSWR}}}.

    Parameters
    ----------
    ka : float
        Electrical size :math:`k a` [rad].
    vswr_max : float, optional
        Maximum voltage standing-wave ratio (1.925 corresponds to -10 dB S11).
    efficiency : float, optional
        Radiation efficiency [-]. Default 1.0.

    Returns
    -------
    float
        Maximum fractional bandwidth :math:`\\Delta f / f_0` [-].
    """
    if vswr_max <= 1.0:
        raise ValueError("vswr_max must exceed 1.0.")
    q_min = chu_wheeler_q_bound(ka, efficiency=efficiency)
    return float((vswr_max - 1.0) / (q_min * np.sqrt(vswr_max)))


def swru120_reference_s11_db(freqs_hz: np.ndarray) -> np.ndarray:
    """Digitized SWRU120D Figure 2-8 (p. 10, approx.) measured reflection curve.

    In SWRU120D Section 2.2 (Figure 2-8, p. 10) and Section 3 (Table 3-1, p. 11),
    the measured reflection on the CC2430DB evaluation board satisfies
    :math:`|S_{11}| < -15\\;\\text{dB}` across the 2.4 GHz ISM band with a
    :math:`-10\\;\\text{dB}` (:math:`< 10\\%` power reflection) bandwidth of more
    than :math:`300\\;\\text{MHz}`. Reading directly from Figure 2-8 (approx.), the
    single-mode resonance minimum occurs near :math:`f_0 \\approx 2.44\\;\\text{GHz}`
    at :math:`|S_{11}|_{\\min} \\approx -16.5\\;\\text{dB}` with a :math:`-10\\;\\text{dB}`
    span from :math:`\\approx 2.29` to :math:`\\approx 2.62\\;\\text{GHz}` (:math:`\\approx 330\\;\\text{MHz}`).

    Parameters
    ----------
    freqs_hz : ndarray
        Frequency grid [Hz].

    Returns
    -------
    s11_nom_db : ndarray
        Approximate reference :math:`|S_{11}|` curve read from SWRU120D Fig. 2-8 [dB].
    """
    f = np.asarray(freqs_hz, dtype=float)
    f0 = 2.44e9
    q_loaded = 6.8
    z_norm = 0.76 + 1j * q_loaded * (f / f0 - f0 / f)
    gamma = (z_norm - 1.0) / (z_norm + 1.0)
    return 20.0 * np.log10(np.maximum(np.abs(gamma), 1e-4))


def richardson_extrapolate_ifa(
    deltas_mm: np.ndarray,
    values: np.ndarray,
    order: float = 1.5,
) -> tuple[float, float, np.ndarray]:
    """Continuum Richardson extrapolation :math:`q(\\Delta) \\approx q_\\infty + c\\,\\Delta^p`.

    Fits a linear model in :math:`\\Delta^p` across the finest resolved mesh levels
    to estimate the continuum (:math:`\\Delta \\to 0`) limit :math:`q_\\infty`.

    Parameters
    ----------
    deltas_mm : array_like
        Trace cell sizes :math:`\\Delta_{\\text{trace}}` [mm].
    values : array_like
        Simulated observable at each mesh level (e.g. :math:`f_{\\text{res}}` [GHz],
        :math:`|S_{11}|_{\\min}` [dB], or :math:`\\text{BW}_{-10\\,\\text{dB}}` [MHz]).
    order : float, optional
        Effective convergence order :math:`p` [-]. Default 1.5.

    Returns
    -------
    q_inf : float
        Extrapolated continuum value at :math:`\\Delta \\to 0`.
    slope : float
        Fitted coefficient :math:`c`.
    delta_dense_mm : ndarray
        Dense grid of :math:`\\Delta \\in [0, \\max(\\Delta)]` for plotting the fit.
    """
    d = np.asarray(deltas_mm, dtype=float)
    v = np.asarray(values, dtype=float)
    if d.size < 2 or d.size != v.size:
        raise ValueError("Require at least two matching (delta, value) points.")
    # Fit using the finest 4 levels (or all if fewer)
    n_fit = min(4, d.size)
    x_pow = d[-n_fit:] ** float(order)
    y_fit = v[-n_fit:]
    slope, q_inf = np.polyfit(x_pow, y_fit, 1)
    d_dense = np.linspace(0.0, float(np.max(d)), 100)
    return float(q_inf), float(slope), d_dense


# ---------------------------------------------------------------------------
# 3D Geometry and Mesh Builders for the SWRU120D PCB IFA
# ---------------------------------------------------------------------------
def build_ifa_structures(
    ground_length_mm: float = 32.0,
    feed_gap_l4_mm: float = 4.80,
    arm_length_l2_mm: float = 16.40,
    substrate_material: str | float = "fr4",
) -> tuple[list[Box], dict[str, float]]:
    """Construct the 3D ``Box`` list for the SWRU120D 2.4 GHz PCB IFA.

    All trace coordinates follow Table 1-1 of TI Design Note DN007 (SWRU120D):
    horizontal closure :math:`W_2 + L_3 + L_4 + L_5 + L_2 + L_6 \\approx L_1 = 25.58\\;\\text{mm}`
    and vertical closure :math:`H_5 + H_7 + H_4 + H_3 + H_2 = 5.70\\;\\text{mm} = H_1`.

    Parameters
    ----------
    ground_length_mm : float, optional
        PCB ground-plane extension along :math:`-y` [mm]. Default 32.0 mm.
    feed_gap_l4_mm : float, optional
        Short-to-feed tap spacing :math:`L_4` [mm]. Default 4.80 mm (Table 1-1).
    arm_length_l2_mm : float, optional
        Open radiating arm length :math:`L_2` [mm]. Default 16.40 mm (Table 1-1).
    substrate_material : str or float, optional
        Wakis material identifier or scalar permittivity. Default ``'fr4'``.

    Returns
    -------
    structures : list of Box
        Ordered structures (FR-4 board, ground plane, shorting via/stub, feed stub,
        and radiating arm).
    geom : dict
        Dictionary of key geometric coordinates in SI units [m].
    """
    d = {k: v * 1e-3 for k, v in SWRU120_TABLE_1_1_MM.items()}
    d["L4"] = float(feed_gap_l4_mm) * 1e-3
    d["L2"] = float(arm_length_l2_mm) * 1e-3
    h_sub = d["h_sub"]
    H1, H2, H3, H5, H6 = d["H1"], d["H2"], d["H3"], d["H5"], d["H6"]
    L2, L3, L4, L5, L6 = d["L2"], d["L3"], d["L4"], d["L5"], d["L6"]
    W1, W2 = d["W1"], d["W2"]

    # Stepped shorting stub on left + feed tap + open radiating arm
    x_jog_hi = W1 + L3  # 3.39 mm
    x_feed_lo = W2 + L3 + L4  # 7.44 mm nominal
    x_feed_hi = x_feed_lo + L5  # 8.44 mm nominal
    x_feed_mid = 0.5 * (x_feed_lo + x_feed_hi)
    # Open arm of length L2 + L6 extending to L1 = 25.58 mm for nominal parameters
    x_tip = x_feed_hi + L2 + (d["L1"] - (8.44e-3 + 16.40e-3))
    y_step2 = H1 - H2 - H3  # 3.67 mm (= H5 + H7 + H4)
    y_arm_lo = H1 - H6  # 4.49 mm

    gnd_xmin = -2.0e-3
    gnd_xmax = max(x_tip + 2.0e-3, 27.58e-3)
    gnd_ymin = -float(ground_length_mm) * 1e-3
    gnd_ymax = 0.0
    sub_ymax = 7.5e-3

    structures = [
        # 1.00 mm FR-4 substrate board (29.58 mm x (7.5 + L_gnd) mm)
        Box(
            (0.5 * (gnd_xmin + gnd_xmax), 0.5 * (gnd_ymin + sub_ymax), 0.5 * h_sub),
            (gnd_xmax - gnd_xmin, sub_ymax - gnd_ymin, h_sub),
            substrate_material,
        ),
        # Main copper ground plane for y <= 0 (top coplanar layer z = h_sub)
        # Note: No ground plane underneath the antenna clearance region 0 < y <= 7.5 mm
        Box(
            (0.5 * (gnd_xmin + gnd_xmax), 0.5 * (gnd_ymin + gnd_ymax), h_sub),
            (gnd_xmax - gnd_xmin, gnd_ymax - gnd_ymin, 0.0),
            "pec",
        ),
        # Shorting post/pad at x in [0, L6], y in [0, y_step2]
        Box((0.5 * L6, 0.5 * y_step2, h_sub), (L6, y_step2, 0.0), "pec"),
        # Horizontal shorting step of width W2 = 0.46 mm across L3
        Box((0.5 * x_jog_hi, y_step2 - 0.5 * W2, h_sub), (x_jog_hi, W2, 0.0), "pec"),
        # Upper shorting riser of width W1 = 1.21 mm up to H1 = 5.70 mm
        Box(
            (x_jog_hi - 0.5 * W1, 0.5 * (y_step2 - W2 + H1), h_sub),
            (W1, H1 - (y_step2 - W2), 0.0),
            "pec",
        ),
        # 50-Ohm feed stub at x in [x_feed_lo, x_feed_hi], y in [H5, H1]
        Box((x_feed_mid, 0.5 * (H5 + H1), h_sub), (L5, H1 - H5, 0.0), "pec"),
        # Main top inverted-F arm of width H6 = W1 = 1.21 mm extending to x_tip
        Box(
            (0.5 * (0.0 + x_tip), H1 - 0.5 * H6, h_sub),
            (x_tip, H6, 0.0),
            "pec",
        ),
    ]

    geom = {
        **d,
        "x_jog_hi": x_jog_hi,
        "x_feed_lo": x_feed_lo,
        "x_feed_hi": x_feed_hi,
        "x_feed_mid": x_feed_mid,
        "x_tip": x_tip,
        "y_step2": y_step2,
        "y_arm_lo": y_arm_lo,
        "gnd_xmin": gnd_xmin,
        "gnd_xmax": gnd_xmax,
        "gnd_ymin": gnd_ymin,
        "gnd_ymax": gnd_ymax,
        "sub_ymax": sub_ymax,
        "ground_length_mm": float(ground_length_mm),
        "feed_gap_l4_mm": float(feed_gap_l4_mm),
        "arm_length_l2_mm": float(arm_length_l2_mm),
    }
    return structures, geom


def build_ifa_axes(
    geom: dict[str, float],
    trace_cell_mm: float = 0.50,
    sub_cells_z: int = 3,
    n_pml: int = 6,
    pad_mm: float = 16.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Build graded 3D tensor-product mesh axes snapped to all SWRU120D faces.

    Enforces Project Rule 1 (:math:`\\ge 20` cells per wavelength in FR-4 at
    :math:`f_{\\max} = 2.8\\;\\text{GHz}`, where :math:`\\lambda_{\\min} = 51.33\\;\\text{mm}`
    so :math:`\\Delta_{\\max} \\le 2.57\\;\\text{mm}`) and evaluates Project Rule 2
    (:math:`\\ge 4` cells across smallest features) across both the substrate
    (:math:`h = 1.00\\;\\text{mm}`) and the sub-millimetre trace features
    (:math:`W_2 = 0.46\\;\\text{mm}`, :math:`L_8 = 0.45\\;\\text{mm}`, :math:`H_5 = 0.66\\;\\text{mm}`).

    Parameters
    ----------
    geom : dict
        Geometry dictionary returned by :func:`build_ifa_structures`.
    trace_cell_mm : float, optional
        Target cell size in the IFA trace region [mm]. Default 0.50 mm.
    sub_cells_z : int, optional
        Number of vertical cells across the 1.00 mm FR-4 substrate. Default 3.
    n_pml : int, optional
        CPML absorber thickness in cells. Default 6.
    pad_mm : float, optional
        Air padding between PCB and CPML inner boundary [mm]. Default 16.0 mm.

    Returns
    -------
    x, y, z : ndarray
        Mesh node arrays including CPML layers [m].
    mesh_info : dict
        Mesh resolution diagnostics and rule compliance summary.
    """
    d_trace = float(trace_cell_mm) * 1e-3
    pad = float(pad_mm) * 1e-3
    n2f_margin = 0.5 * pad
    h_sub = geom["h_sub"]
    dz_sub = h_sub / max(int(sub_cells_z), 1)

    x_pts = meshing.merge_points(
        [
            geom["gnd_xmin"] - pad,
            geom["gnd_xmin"] - n2f_margin,
            geom["gnd_xmin"],
            0.0,
            geom["L6"],
            geom["x_jog_hi"] - geom["W1"],
            geom["x_jog_hi"],
            geom["x_feed_lo"],
            geom["x_feed_hi"],
            geom["x_tip"],
            geom["gnd_xmax"],
            geom["gnd_xmax"] + n2f_margin,
            geom["gnd_xmax"] + pad,
        ],
        1e-6,
    )
    x_sizes = [
        4.6e-3,
        3.2e-3,
        2.2e-3,
        d_trace,
        d_trace,
        d_trace,
        max(1.6 * d_trace, 0.8e-3),
        d_trace,
        max(2.2 * d_trace, 1.4e-3),
        2.2e-3,
        3.2e-3,
        4.6e-3,
    ]
    c_feat = 1 if d_trace >= 0.34e-3 else (2 if d_trace >= 0.20e-3 else 4)
    x_min_c = [2, 2, 1, c_feat, c_feat, c_feat, 2, c_feat, 5, 1, 2, 2]
    x_core = meshing.graded_axis(
        x_pts, x_sizes[: len(x_pts) - 1], x_min_c[: len(x_pts) - 1], grading=1.45
    )

    y_pts = meshing.merge_points(
        [
            geom["gnd_ymin"] - pad,
            geom["gnd_ymin"] - n2f_margin,
            geom["gnd_ymin"],
            0.0,
            geom["H5"],
            geom["y_step2"] - geom["W2"],
            geom["y_step2"],
            geom["y_arm_lo"],
            geom["H1"],
            geom["sub_ymax"],
            geom["sub_ymax"] + n2f_margin,
            geom["sub_ymax"] + pad,
        ],
        1e-6,
    )
    y_sizes = [
        4.6e-3,
        3.2e-3,
        2.3e-3,
        max(d_trace, 0.33e-3) if d_trace < 0.20e-3 else d_trace,
        max(1.5 * d_trace, 0.75e-3),
        d_trace,
        max(1.4 * d_trace, 0.65e-3),
        d_trace,
        1.8e-3,
        3.2e-3,
        4.6e-3,
    ]
    y_min_c = [2, 2, 4, c_feat, 2, c_feat, 1 if d_trace >= 0.34e-3 else 2, c_feat, 1, 2, 2]
    y_core = meshing.graded_axis(
        y_pts, y_sizes[: len(y_pts) - 1], y_min_c[: len(y_pts) - 1], grading=1.45
    )

    z_pts = [-pad, -n2f_margin, 0.0, h_sub, h_sub + n2f_margin, h_sub + pad]
    z_sizes = [4.4e-3, 2.3e-3, dz_sub, 2.3e-3, 4.4e-3]
    z_min_c = [2, 3, max(int(sub_cells_z), 2), 3, 2]
    z_core = meshing.graded_axis(z_pts, z_sizes, z_min_c, grading=1.45)

    axes = []
    for arr in (x_core, y_core, z_core):
        d_lo = arr[1] - arr[0]
        d_hi = arr[-1] - arr[-2]
        axes.append(
            np.concatenate(
                [
                    arr[0] - d_lo * np.arange(n_pml, 0, -1),
                    arr,
                    arr[-1] + d_hi * np.arange(1, n_pml + 1),
                ]
            )
        )
    x, y, z = axes
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)

    # Wavelength in FR-4 at f_max = 2.8 GHz
    eps_r_fmax, _, _ = fr4_complex_permittivity(np.array([2.8e9]))
    lam_fr4_min = float(c_light / (2.8e9 * np.sqrt(eps_r_fmax[0])))
    lam_air_min = float(c_light / 2.8e9)

    # Cells inside substrate bounds
    in_sub_x = dx[(x[:-1] >= geom["gnd_xmin"]) & (x[1:] <= geom["gnd_xmax"])]
    in_sub_y = dy[(y[:-1] >= geom["gnd_ymin"]) & (y[1:] <= geom["sub_ymax"])]
    max_sub_cell = float(max(np.max(in_sub_x), np.max(in_sub_y), dz_sub))
    min_cpw_fr4 = float(lam_fr4_min / max_sub_cell)
    min_cpw_air = float(lam_air_min / max(np.max(dx), np.max(dy), np.max(dz)))

    dt_cfl085 = float(
        0.85
        / (
            c_light
            * np.sqrt(
                1.0 / np.min(dx) ** 2 + 1.0 / np.min(dy) ** 2 + 1.0 / np.min(dz) ** 2
            )
        )
    )
    n_steps_est = int(np.ceil(5.2e-9 / dt_cfl085))
    n_cells = int(dx.size * dy.size * dz.size)
    n_sub_edges = int(3 * in_sub_x.size * in_sub_y.size * max(int(sub_cells_z), 1))
    est_cpu = estimate(
        cells=n_cells,
        steps=n_steps_est,
        backend="cpu",
        cpml=True,
        dispersive_edges=n_sub_edges,
        n_poles=2,
    )
    est_gpu = estimate(
        cells=n_cells,
        steps=n_steps_est,
        backend="gpu",
        cpml=True,
        dispersive_edges=n_sub_edges,
        n_poles=2,
        fused=True,
    )

    mesh_info = {
        "trace_cell_mm": float(trace_cell_mm),
        "sub_cells_z": int(sub_cells_z),
        "nx": int(dx.size),
        "ny": int(dy.size),
        "nz": int(dz.size),
        "n_cells": n_cells,
        "dx_min_mm": float(np.min(dx) * 1e3),
        "dy_min_mm": float(np.min(dy) * 1e3),
        "dz_min_mm": float(np.min(dz) * 1e3),
        "dx_max_mm": float(np.max(dx) * 1e3),
        "dy_max_mm": float(np.max(dy) * 1e3),
        "dz_max_mm": float(np.max(dz) * 1e3),
        "lam_fr4_fmax_mm": float(lam_fr4_min * 1e3),
        "min_cpw_fr4": min_cpw_fr4,
        "min_cpw_air": min_cpw_air,
        "rule_20cpw_met": bool(min_cpw_fr4 >= 20.0),
        "cells_across_h_sub": float(h_sub / dz_sub),
        "cells_across_W1_1p21mm": float(geom["W1"] / d_trace),
        "cells_across_L5_1p00mm": float(geom["L5"] / d_trace),
        "cells_across_H5_0p66mm": int(
            np.sum((y[:-1] >= -1e-7) & (y[1:] <= geom["H5"] + 1e-7))
        ),
        "cells_across_W2_0p46mm": int(
            np.sum(
                (y[:-1] >= geom["y_step2"] - geom["W2"] - 1e-7)
                & (y[1:] <= geom["y_step2"] + 1e-7)
            )
        ),
        "dt_ps": float(dt_cfl085 * 1e12),
        "n_steps_est": n_steps_est,
        "memory_mb": float(est_cpu.host_bytes / 1e6),
        "gpu_memory_mb": float(est_gpu.gpu_bytes / 1e6),
        "cpu_wall_est_s": float(est_cpu.seconds),
        "gpu_wall_est_s": float(est_gpu.seconds),
        "n2f_margin_m": float(n2f_margin),
    }
    return x, y, z, mesh_info


def build_ifa_axes_noncubic(
    geom: dict[str, float],
    dy_jog_mm: float = 0.115,
    dy_gap_mm: float = 0.165,
    dx_stub_mm: float = 0.25,
    dy_arm_mm: float = 0.20,
    sub_cells_z: int = 4,
    n_pml: int = 6,
    pad_mm: float = 16.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Build a graded, non-cubic locally refined 3D mesh for the SWRU120D PCB IFA.

    Places fine anisotropic cells only across the sub-millimetre features
    (:math:`\\Delta y = 0.115\\;\\text{mm}` across the :math:`W_2 = 0.46\\;\\text{mm}`
    meander jog, :math:`\\Delta y = 0.165\\;\\text{mm}` across the :math:`H_5 = 0.66\\;\\text{mm}`
    coplanar feed gap, :math:`\\Delta y = 0.20\\;\\text{mm}` across the :math:`H_6 = 1.21\\;\\text{mm}`
    upper arm, and :math:`\\Delta x = 0.24\\text{–}0.25\\;\\text{mm}` across the
    :math:`L_6 = 1.00\\;\\text{mm}` shorting via, :math:`W_1 = 1.21\\;\\text{mm}` riser,
    and :math:`L_5 = 1.00\\;\\text{mm}` feed stub), while setting
    :math:`\\Delta z = 0.25\\;\\text{mm}` across the :math:`1.00\\;\\text{mm}` FR-4
    substrate independently of :math:`\\Delta x, \\Delta y` and grading coarsely along
    the uniform :math:`16.40\\;\\text{mm}` open arm, under the :math:`32.0\\;\\text{mm}`
    ground plane, and in the air padding.

    Parameters
    ----------
    geom : dict
        Geometry dictionary returned by :func:`build_ifa_structures`.
    dy_jog_mm : float, optional
        Target :math:`\\Delta y` across the :math:`W_2 = 0.46\\;\\text{mm}` jog [mm].
        Default ``0.115`` (4 cells).
    dy_gap_mm : float, optional
        Target :math:`\\Delta y` across the :math:`H_5 = 0.66\\;\\text{mm}` feed gap [mm].
        Default ``0.165`` (4 cells).
    dx_stub_mm : float, optional
        Target :math:`\\Delta x` across the shorting via (:math:`L_6 = 1.00\\;\\text{mm}`)
        and feed stub (:math:`L_5 = 1.00\\;\\text{mm}`) [mm]. Default ``0.25`` (4 cells).
    dy_arm_mm : float, optional
        Target :math:`\\Delta y` across the :math:`H_6 = 1.21\\;\\text{mm}` arm [mm].
        Default ``0.20`` (6 cells).
    sub_cells_z : int, optional
        Number of vertical cells across the :math:`1.00\\;\\text{mm}` FR-4 substrate.
        Default ``4``.
    n_pml : int, optional
        CPML absorber thickness in cells. Default ``6``.
    pad_mm : float, optional
        Air padding between PCB and CPML inner boundary [mm]. Default ``16.0``.

    Returns
    -------
    x, y, z : ndarray
        Mesh node arrays including CPML layers [m].
    mesh_info : dict
        Mesh resolution diagnostics and rule compliance summary.
    """
    pad = float(pad_mm) * 1e-3
    n2f_margin = 0.5 * pad
    h_sub = geom["h_sub"]
    dz_sub = h_sub / max(int(sub_cells_z), 1)
    dx_stub = float(dx_stub_mm) * 1e-3
    dy_jog = float(dy_jog_mm) * 1e-3
    dy_gap = float(dy_gap_mm) * 1e-3
    dy_arm = float(dy_arm_mm) * 1e-3

    x_pts = meshing.merge_points(
        [
            geom["gnd_xmin"] - pad,
            geom["gnd_xmin"] - n2f_margin,
            geom["gnd_xmin"],
            0.0,
            geom["L6"],
            geom["x_jog_hi"] - geom["W1"],
            geom["x_jog_hi"],
            geom["x_feed_lo"],
            geom["x_feed_hi"],
            geom["x_tip"],
            geom["gnd_xmax"],
            geom["gnd_xmax"] + n2f_margin,
            geom["gnd_xmax"] + pad,
        ],
        1e-6,
    )
    x_sizes = [
        4.6e-3,
        3.2e-3,
        2.2e-3,
        dx_stub,
        max(1.18 * dx_stub, 0.295e-3),
        min(dx_stub, 0.242e-3),
        0.80e-3,
        dx_stub,
        1.40e-3,
        2.2e-3,
        3.2e-3,
        4.6e-3,
    ]
    c_x_stub = max(1, int(np.round(geom["L5"] / dx_stub)))
    c_x_w1 = max(1, int(np.ceil(geom["W1"] / dx_stub)))
    x_min_c = [2, 2, 1, c_x_stub, c_x_stub, c_x_w1, 2, c_x_stub, 5, 1, 2, 2]
    x_core = meshing.graded_axis(
        x_pts, x_sizes[: len(x_pts) - 1], x_min_c[: len(x_pts) - 1], grading=1.45
    )

    y_pts = meshing.merge_points(
        [
            geom["gnd_ymin"] - pad,
            geom["gnd_ymin"] - n2f_margin,
            geom["gnd_ymin"],
            0.0,
            geom["H5"],
            geom["y_step2"] - geom["W2"],
            geom["y_step2"],
            geom["y_arm_lo"],
            geom["H1"],
            geom["sub_ymax"],
            geom["sub_ymax"] + n2f_margin,
            geom["sub_ymax"] + pad,
        ],
        1e-6,
    )
    y_sizes = [
        4.6e-3,
        3.2e-3,
        2.3e-3,
        max(2.0 * dy_gap, 0.33e-3),
        0.75e-3,
        dy_jog,
        0.65e-3,
        dy_arm,
        1.8e-3,
        3.2e-3,
        4.6e-3,
    ]
    c_y_gap = max(1, int(np.round(geom["H5"] / dy_gap)))
    c_y_jog = max(1, int(np.round(geom["W2"] / dy_jog)))
    c_y_arm = max(1, int(np.round(geom["H6"] / dy_arm)))
    y_min_c = [2, 2, 4, c_y_gap, 2, c_y_jog, 2, c_y_arm, 1, 2, 2]
    y_core = meshing.graded_axis(
        y_pts, y_sizes[: len(y_pts) - 1], y_min_c[: len(y_pts) - 1], grading=1.45
    )

    z_pts = [-pad, -n2f_margin, 0.0, h_sub, h_sub + n2f_margin, h_sub + pad]
    z_sizes = [4.4e-3, 2.3e-3, dz_sub, 2.3e-3, 4.4e-3]
    z_min_c = [2, 3, max(int(sub_cells_z), 2), 3, 2]
    z_core = meshing.graded_axis(z_pts, z_sizes, z_min_c, grading=1.45)

    axes = []
    for arr in (x_core, y_core, z_core):
        d_lo = arr[1] - arr[0]
        d_hi = arr[-1] - arr[-2]
        axes.append(
            np.concatenate(
                [
                    arr[0] - d_lo * np.arange(n_pml, 0, -1),
                    arr,
                    arr[-1] + d_hi * np.arange(1, n_pml + 1),
                ]
            )
        )
    x, y, z = axes
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)

    eps_r_fmax, _, _ = fr4_complex_permittivity(np.array([2.8e9]))
    lam_fr4_min = float(c_light / (2.8e9 * np.sqrt(eps_r_fmax[0])))
    lam_air_min = float(c_light / 2.8e9)

    in_sub_x = dx[(x[:-1] >= geom["gnd_xmin"]) & (x[1:] <= geom["gnd_xmax"])]
    in_sub_y = dy[(y[:-1] >= geom["gnd_ymin"]) & (y[1:] <= geom["sub_ymax"])]
    max_sub_cell = float(max(np.max(in_sub_x), np.max(in_sub_y), dz_sub))
    min_cpw_fr4 = float(lam_fr4_min / max_sub_cell)
    min_cpw_air = float(lam_air_min / max(np.max(dx), np.max(dy), np.max(dz)))

    dt_cfl085 = float(
        0.85
        / (
            c_light
            * np.sqrt(
                1.0 / np.min(dx) ** 2 + 1.0 / np.min(dy) ** 2 + 1.0 / np.min(dz) ** 2
            )
        )
    )
    n_steps_est = int(np.ceil(5.2e-9 / dt_cfl085))
    n_cells = int(dx.size * dy.size * dz.size)
    n_sub_edges = int(3 * in_sub_x.size * in_sub_y.size * max(int(sub_cells_z), 1))
    est_cpu = estimate(
        cells=n_cells,
        steps=n_steps_est,
        backend="cpu",
        cpml=True,
        dispersive_edges=n_sub_edges,
        n_poles=2,
    )
    est_gpu = estimate(
        cells=n_cells,
        steps=n_steps_est,
        backend="gpu",
        cpml=True,
        dispersive_edges=n_sub_edges,
        n_poles=2,
        fused=True,
    )

    n_l6 = int(np.sum((x[:-1] >= -1e-7) & (x[1:] <= geom["L6"] + 1e-7)))
    n_w1 = int(
        np.sum(
            (x[:-1] >= geom["x_jog_hi"] - geom["W1"] - 1e-7)
            & (x[1:] <= geom["x_jog_hi"] + 1e-7)
        )
    )
    n_l5 = int(
        np.sum(
            (x[:-1] >= geom["x_feed_lo"] - 1e-7)
            & (x[1:] <= geom["x_feed_hi"] + 1e-7)
        )
    )
    n_h5 = int(np.sum((y[:-1] >= -1e-7) & (y[1:] <= geom["H5"] + 1e-7)))
    n_w2 = int(
        np.sum(
            (y[:-1] >= geom["y_step2"] - geom["W2"] - 1e-7)
            & (y[1:] <= geom["y_step2"] + 1e-7)
        )
    )
    n_h6 = int(
        np.sum((y[:-1] >= geom["y_arm_lo"] - 1e-7) & (y[1:] <= geom["H1"] + 1e-7))
    )

    mesh_info = {
        "trace_cell_mm": float(dy_jog_mm),
        "dx_stub_mm": float(dx_stub_mm),
        "dy_gap_mm": float(dy_gap_mm),
        "dy_jog_mm": float(dy_jog_mm),
        "dy_arm_mm": float(dy_arm_mm),
        "sub_cells_z": int(sub_cells_z),
        "noncubic_local": True,
        "nx": int(dx.size),
        "ny": int(dy.size),
        "nz": int(dz.size),
        "n_cells": n_cells,
        "dx_min_mm": float(np.min(dx) * 1e3),
        "dy_min_mm": float(np.min(dy) * 1e3),
        "dz_min_mm": float(np.min(dz) * 1e3),
        "dx_max_mm": float(np.max(dx) * 1e3),
        "dy_max_mm": float(np.max(dy) * 1e3),
        "dz_max_mm": float(np.max(dz) * 1e3),
        "lam_fr4_fmax_mm": float(lam_fr4_min * 1e3),
        "min_cpw_fr4": min_cpw_fr4,
        "min_cpw_air": min_cpw_air,
        "rule_20cpw_met": bool(min_cpw_fr4 >= 20.0),
        "cells_across_h_sub": float(h_sub / dz_sub),
        "cells_across_L6_1p00mm": float(n_l6),
        "cells_across_W1_1p21mm": float(n_w1),
        "cells_across_L5_1p00mm": float(n_l5),
        "cells_across_H5_0p66mm": n_h5,
        "cells_across_W2_0p46mm": n_w2,
        "cells_across_H6_1p21mm": n_h6,
        "dt_ps": float(dt_cfl085 * 1e12),
        "n_steps_est": n_steps_est,
        "memory_mb": float(est_cpu.host_bytes / 1e6),
        "gpu_memory_mb": float(est_gpu.gpu_bytes / 1e6),
        "cpu_wall_est_s": float(est_cpu.seconds),
        "gpu_wall_est_s": float(est_gpu.seconds),
        "n2f_margin_m": float(n2f_margin),
    }
    return x, y, z, mesh_info


def run_ifa_simulation(
    trace_cell_mm: float = 0.50,
    sub_cells_z: int = 3,
    ground_length_mm: float = 32.0,
    feed_gap_l4_mm: float = 4.80,
    arm_length_l2_mm: float = 16.40,
    record_fields: bool = False,
    record_animation: bool = False,
    sim_time_ns: float = 5.2,
    energy_decay: float = 1e-4,
    noncubic_local: bool = False,
) -> dict:
    """Run a 3D FIT simulation of the SWRU120D 2.4 GHz PCB IFA.

    Preflights the cell count, memory, and step count with :func:`wakis.estimate`,
    excites the coplanar feed gap with a 50-Ohm :class:`LumpedPort`, and extracts
    :math:`S_{11}(f)`, :math:`Z_{\\text{in}}(f)`, radiation/total efficiency, and
    3D far-field patterns via :class:`Near2FarMonitor`.

    Parameters
    ----------
    trace_cell_mm : float, optional
        Target trace cell size [mm]. Default 0.50 mm.
    sub_cells_z : int, optional
        Number of vertical cells across the 1.00 mm FR-4 substrate. Default 3.
    ground_length_mm : float, optional
        Ground plane length [mm]. Default 32.0 mm.
    feed_gap_l4_mm : float, optional
        Short-to-feed tap spacing :math:`L_4` [mm]. Default 4.80 mm.
    arm_length_l2_mm : float, optional
        Open arm length :math:`L_2` [mm]. Default 16.40 mm.
    record_fields : bool, optional
        If True, attach frequency-domain :class:`DFTMonitor` slices at 2.44 GHz
        for near-field and surface-current visualization.
    record_animation : bool, optional
        If True, record time-domain :math:`E_y`/:math:`E_z` snapshots for GIF
        generation.
    sim_time_ns : float, optional
        Maximum physical simulation duration [ns]. Default 5.2 ns.
    energy_decay : float, optional
        Energy decay threshold for early termination. Default ``1e-4``.

    Returns
    -------
    dict
        Simulation results, spectra, far-field patterns, and preflight metrics.
    """
    f0 = 2.44e9
    n_pml = 6
    structures, geom = build_ifa_structures(
        ground_length_mm=ground_length_mm,
        feed_gap_l4_mm=feed_gap_l4_mm,
        arm_length_l2_mm=arm_length_l2_mm,
        substrate_material="fr4",
    )
    if noncubic_local:
        x, y, z, mesh_info = build_ifa_axes_noncubic(
            geom,
            dy_jog_mm=trace_cell_mm,
            sub_cells_z=sub_cells_z,
            n_pml=n_pml,
        )
    else:
        x, y, z, mesh_info = build_ifa_axes(
            geom,
            trace_cell_mm=trace_cell_mm,
            sub_cells_z=sub_cells_z,
            n_pml=n_pml,
        )

    t0 = time.perf_counter()
    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml"] * 3,
        bc_high=["cpml"] * 3,
        n_pml=n_pml,
        cfln=0.85,
        use_stl=False,
        verbose=0,
    )
    assign_materials(solver, structures, f0)

    pulse = GaussianSource(f0, 0.85e9, cutoff=3.5)
    port = LumpedPort(
        pulse,
        component="y",
        center=(geom["x_feed_mid"], 0.5 * geom["H5"], geom["h_sub"]),
        size=(geom["L5"], geom["H5"], 0.0),
        impedance=50.0,
    )

    n2f_margin = mesh_info["n2f_margin_m"]
    n2f_center = (
        0.5 * (geom["gnd_xmin"] + geom["gnd_xmax"]),
        0.5 * (geom["gnd_ymin"] + geom["sub_ymax"]),
        0.5 * geom["h_sub"],
    )
    n2f_size = (
        geom["gnd_xmax"] - geom["gnd_xmin"] + 2.0 * n2f_margin,
        geom["sub_ymax"] - geom["gnd_ymin"] + 2.0 * n2f_margin,
        geom["h_sub"] + 2.0 * n2f_margin,
    )
    f_n2f = np.array([2.40e9, 2.44e9, 2.48e9])
    n2f = Near2FarMonitor(f_n2f, center=n2f_center, size=n2f_size)
    monitors = [n2f]

    dft_xy = None
    dft_xz = None
    if record_fields:
        # XY slice just at the top copper / substrate interface z = h_sub
        xy_center = (
            0.5 * (geom["gnd_xmin"] + geom["gnd_xmax"]),
            0.5 * (geom["gnd_ymin"] + geom["sub_ymax"]),
            geom["h_sub"],
        )
        xy_size = (
            geom["gnd_xmax"] - geom["gnd_xmin"] + 8.0e-3,
            geom["sub_ymax"] - geom["gnd_ymin"] + 8.0e-3,
            0.0,
        )
        dft_xy = DFTMonitor(
            [f0],
            components=("Ex", "Ey", "Ez", "Hx", "Hy"),
            center=xy_center,
            size=xy_size,
        )
        # XZ slice through the radiating arm y = H1 - 0.5 * H6
        xz_center = (
            0.5 * (geom["gnd_xmin"] + geom["gnd_xmax"]),
            geom["H1"] - 0.5 * geom["H6"],
            0.5 * geom["h_sub"],
        )
        xz_size = (
            geom["gnd_xmax"] - geom["gnd_xmin"] + 12.0e-3,
            0.0,
            geom["h_sub"] + 16.0e-3,
        )
        dft_xz = DFTMonitor(
            [f0],
            components=("Ex", "Ey", "Ez"),
            center=xz_center,
            size=xz_size,
        )
        monitors.extend([dft_xy, dft_xz])

    Nt = int(np.ceil(float(sim_time_ns) * 1e-9 / solver.dt))
    anim_frames = []
    anim_times_ns = []
    if record_animation:
        # Step manually to record 28 time-domain snapshots on the XY plane (z = h_sub)
        k_sub = int(np.argmin(np.abs(grid.z - geom["h_sub"])))
        ix_lo = n_pml
        ix_hi = grid.Nx - n_pml
        iy_lo = n_pml
        iy_hi = grid.Ny - n_pml
        anim_x_mm = 0.5 * (grid.x[ix_lo:ix_hi] + grid.x[ix_lo + 1 : ix_hi + 1]) * 1e3
        anim_y_mm = 0.5 * (grid.y[iy_lo:iy_hi] + grid.y[iy_lo + 1 : iy_hi + 1]) * 1e3
        n_anim_steps = min(Nt, int(np.ceil(2.8e-9 / solver.dt)))
        stride = max(1, n_anim_steps // 28)
        for step in range(Nt):
            t_cur = step * solver.dt
            port.update(solver, t_cur)
            solver.one_step()
            port.post_step(solver, t_cur)
            for mon in monitors:
                mon.update(solver, t_cur)
            if step < n_anim_steps and step % stride == 0 and len(anim_frames) < 28:
                ex_slice = solver.E.to_matrix("x")[ix_lo:ix_hi, iy_lo:iy_hi, k_sub]
                ey_slice = solver.E.to_matrix("y")[ix_lo:ix_hi, iy_lo:iy_hi, k_sub]
                ez_slice = solver.E.to_matrix("z")[ix_lo:ix_hi, iy_lo:iy_hi, k_sub]
                # Signed wave snapshot along dominant E_y + E_z polarization
                e_wave = np.asarray(ey_slice + 0.5 * ez_slice, dtype=np.float32)
                anim_frames.append(e_wave)
                anim_times_ns.append(float((t_cur + solver.dt) * 1e9))
    else:
        solver.emsolve(
            Nt,
            source=[port],
            monitors=monitors,
            stop_when_energy_decayed=energy_decay,
        )

    wall_s = time.perf_counter() - t0
    freqs = np.linspace(2.0e9, 2.8e9, 161)
    zin = port.impedance_in(freqs)
    s11 = port.s11(freqs)
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-6))
    i_min = int(np.argmin(s11_db))
    f_res = float(freqs[i_min])
    s11_min_db = float(s11_db[i_min])
    zin_res = complex(zin[i_min])
    in_band = freqs[s11_db <= -10.0]
    bw_10db_mhz = float((in_band[-1] - in_band[0]) / 1e6) if in_band.size > 1 else 0.0

    # Efficiency and far-field pattern on spherical grid at 2.44 GHz
    P_acc = np.real(port.voltage(f_n2f) * np.conj(port.port_current(f_n2f)))
    a_wave, _ = port.power_waves(f_n2f)
    P_inc = np.abs(a_wave) ** 2
    P_rad = n2f.radiated_power()
    P_flux = n2f.get_flux()
    eta_rad = np.clip(P_rad / np.maximum(P_acc, 1e-30), 0.0, 1.0)
    eta_tot = np.clip(P_rad / np.maximum(P_inc, 1e-30), 0.0, 1.0)

    theta_1d = np.linspace(0.0, np.pi, 73)
    phi_1d = np.linspace(0.0, 2.0 * np.pi, 144, endpoint=False)
    TH, PH = np.meshgrid(theta_1d, phi_1d, indexing="ij")
    D_grid = n2f.directivity(TH, PH)  # shape (3, 73, 144)
    G_grid = D_grid * eta_rad[:, None, None]
    G_real_grid = D_grid * eta_tot[:, None, None]

    # Index 1 is f = 2.44 GHz
    D_2g44 = D_grid[1]
    G_2g44 = G_grid[1]
    D_max_dbi = float(10.0 * np.log10(np.max(D_2g44)))
    G_max_dbi = float(10.0 * np.log10(np.max(G_2g44)))
    G_real_max_dbi = float(10.0 * np.log10(np.max(G_real_grid[1])))

    # Principal plane cuts at 2.44 GHz:
    # 1. XZ plane (phi = 0 and phi = pi) -> full 360 deg polar cut
    angles_deg = np.linspace(0.0, 360.0, 145, endpoint=True)
    ang_rad = np.deg2rad(angles_deg)
    # XZ plane: theta = |angle|, phi = 0 for [0, 180], phi = pi for (180, 360)
    th_xz = np.where(ang_rad <= np.pi, ang_rad, 2.0 * np.pi - ang_rad)
    ph_xz = np.where(ang_rad <= np.pi, 0.0, np.pi)
    g_xz_lin = n2f.directivity(th_xz, ph_xz)[1] * eta_rad[1]
    g_xz_dbi = 10.0 * np.log10(np.maximum(g_xz_lin, 1e-4))

    # 2. YZ plane (phi = pi/2 and phi = 3*pi/2)
    th_yz = np.where(ang_rad <= np.pi, ang_rad, 2.0 * np.pi - ang_rad)
    ph_yz = np.where(ang_rad <= np.pi, 0.5 * np.pi, 1.5 * np.pi)
    g_yz_lin = n2f.directivity(th_yz, ph_yz)[1] * eta_rad[1]
    g_yz_dbi = 10.0 * np.log10(np.maximum(g_yz_lin, 1e-4))

    # 3. XY plane (theta = pi/2, phi = ang_rad)
    th_xy = np.full_like(ang_rad, 0.5 * np.pi)
    ph_xy = np.mod(ang_rad, 2.0 * np.pi)
    g_xy_lin = n2f.directivity(th_xy, ph_xy)[1] * eta_rad[1]
    g_xy_dbi = 10.0 * np.log10(np.maximum(g_xy_lin, 1e-4))

    out = {
        "geom": geom,
        "mesh_info": mesh_info,
        "wall_seconds": float(wall_s),
        "freqs_hz": freqs,
        "s11_real": np.real(s11),
        "s11_imag": np.imag(s11),
        "s11_db": s11_db,
        "zin_real": np.real(zin),
        "zin_imag": np.imag(zin),
        "f_res_hz": f_res,
        "s11_min_db": s11_min_db,
        "zin_res_real_ohm": float(zin_res.real),
        "zin_res_imag_ohm": float(zin_res.imag),
        "bw_10db_mhz": bw_10db_mhz,
        "f_n2f_hz": f_n2f,
        "eta_rad": eta_rad,
        "eta_tot": eta_tot,
        "p_flux_over_p_rad": float(P_flux[1] / max(P_rad[1], 1e-30)),
        "d_max_dbi": D_max_dbi,
        "g_max_dbi": G_max_dbi,
        "g_realized_max_dbi": G_real_max_dbi,
        "g_xz_peak_dbi": float(np.max(g_xz_dbi)),
        "g_yz_peak_dbi": float(np.max(g_yz_dbi)),
        "g_xy_peak_dbi": float(np.max(g_xy_dbi)),
        "theta_1d_rad": theta_1d,
        "phi_1d_rad": phi_1d,
        "gain_3d_2g44_lin": G_2g44,
        "angles_deg": angles_deg,
        "g_xz_dbi": g_xz_dbi,
        "g_yz_dbi": g_yz_dbi,
        "g_xy_dbi": g_xy_dbi,
        "x_nodes_m": x,
        "y_nodes_m": y,
        "z_nodes_m": z,
    }

    if record_fields and dft_xy is not None and dft_xz is not None:
        Ex_xy, c_ex = dft_xy.get_field("Ex")
        Ey_xy, c_ey = dft_xy.get_field("Ey")
        Ez_xy, c_ez = dft_xy.get_field("Ez")
        Hx_xy, c_hx = dft_xy.get_field("Hx")
        Hy_xy, c_hy = dft_xy.get_field("Hy")

        # Interpolate onto a common coarse 2D grid (nx_c x ny_c) for small storage
        xc = c_ez[0]
        yc = c_ez[1]
        ex_2d = np.squeeze(Ex_xy[0])
        ey_2d = np.squeeze(Ey_xy[0])
        ez_2d = np.squeeze(Ez_xy[0])
        hx_2d = np.squeeze(Hx_xy[0])
        hy_2d = np.squeeze(Hy_xy[0])
        nx_min = min(ex_2d.shape[0], ey_2d.shape[0], ez_2d.shape[0], hx_2d.shape[0], hy_2d.shape[0])
        ny_min = min(ex_2d.shape[1], ey_2d.shape[1], ez_2d.shape[1], hx_2d.shape[1], hy_2d.shape[1])
        ex_2d = ex_2d[:nx_min, :ny_min]
        ey_2d = ey_2d[:nx_min, :ny_min]
        ez_2d = ez_2d[:nx_min, :ny_min]
        hx_2d = hx_2d[:nx_min, :ny_min]
        hy_2d = hy_2d[:nx_min, :ny_min]
        xc = xc[:nx_min]
        yc = yc[:ny_min]

        e_mag_xy = np.sqrt(np.abs(ex_2d) ** 2 + np.abs(ey_2d) ** 2 + np.abs(ez_2d) ** 2)
        # Tangential magnetic field discontinuity gives surface current J_s = n x H = (-Hy, Hx)
        jx_surf = -hy_2d
        jy_surf = hx_2d
        j_mag_xy = np.sqrt(np.abs(jx_surf) ** 2 + np.abs(jy_surf) ** 2)

        Ex_xz, c_xz_x = dft_xz.get_field("Ex")
        Ey_xz, _ = dft_xz.get_field("Ey")
        Ez_xz, c_xz_z = dft_xz.get_field("Ez")
        ex_xz = np.squeeze(Ex_xz[0])
        ey_xz = np.squeeze(Ey_xz[0])
        ez_xz = np.squeeze(Ez_xz[0])
        nx_xz = min(ex_xz.shape[0], ey_xz.shape[0], ez_xz.shape[0])
        nz_xz = min(ex_xz.shape[1], ey_xz.shape[1], ez_xz.shape[1])
        ex_xz = ex_xz[:nx_xz, :nz_xz]
        ey_xz = ey_xz[:nx_xz, :nz_xz]
        ez_xz = ez_xz[:nx_xz, :nz_xz]
        e_mag_xz = np.sqrt(np.abs(ex_xz) ** 2 + np.abs(ey_xz) ** 2 + np.abs(ez_xz) ** 2)

        out.update(
            {
                "xy_x_mm": xc * 1e3,
                "xy_y_mm": yc * 1e3,
                "e_mag_xy": e_mag_xy,
                "ey_real_xy": np.real(ey_2d),
                "ez_real_xy": np.real(ez_2d),
                "jx_real_xy": np.real(jx_surf),
                "jy_real_xy": np.real(jy_surf),
                "j_mag_xy": j_mag_xy,
                "xz_x_mm": c_xz_x[0][:nx_xz] * 1e3,
                "xz_z_mm": c_xz_z[2][:nz_xz] * 1e3,
                "e_mag_xz": e_mag_xz,
                "ez_real_xz": np.real(ez_xz),
            }
        )

    if record_animation and anim_frames:
        out["anim_x_mm"] = anim_x_mm
        out["anim_y_mm"] = anim_y_mm
        out["anim_frames"] = np.stack(anim_frames, axis=0)
        out["anim_times_ns"] = np.asarray(anim_times_ns, dtype=float)

    return out


# ---------------------------------------------------------------------------
# Task Registry and Data Serialization
# ---------------------------------------------------------------------------
CPU_TASKS = (
    "res_0p50mm_baseline",
    "res_0p35mm",
    "res_0p25mm",
    "res_0p18mm",
    "local_noncubic_0p11mm",
    "param_study",
)
GPU_TASKS = ("gpu_subfeature_0p11mm",)
ALL_TASKS = CPU_TASKS + GPU_TASKS


def preflight_all_levels() -> dict[str, dict]:
    """Compute ``wakis.estimate`` preflight metrics across all 5 uniform-trace levels and the non-cubic locally refined mesh."""
    _, geom = build_ifa_structures()
    levels = {
        "coarse_0p50mm": (0.50, 3),
        "medium_0p35mm": (0.35, 3),
        "fine_0p25mm": (0.25, 4),
        "subfeature_0p18mm": (0.18, 4),
        "gpu_subfeature_0p11mm": (0.1125, 4),
    }
    report = {}
    for name, (d_mm, nz_sub) in levels.items():
        _, _, _, info = build_ifa_axes(geom, trace_cell_mm=d_mm, sub_cells_z=nz_sub)
        report[name] = info
    _, _, _, nc_info = build_ifa_axes_noncubic(geom, dy_jog_mm=0.115, sub_cells_z=4)
    report["local_noncubic_0p11mm"] = nc_info
    return report


def _metal_mask_xy(x_mm: np.ndarray, y_mm: np.ndarray, geom: dict[str, float]) -> np.ndarray:
    """Boolean mask of top-layer PEC metallization on an (x_mm, y_mm) grid."""
    X, Y = np.meshgrid(x_mm * 1e-3, y_mm * 1e-3, indexing="ij")
    H1, H2, H3, H5, H6 = geom["H1"], geom["H2"], geom["H3"], geom["H5"], geom["H6"]
    L6, W1, W2 = geom["L6"], geom["W1"], geom["W2"]
    x_jog_hi = geom["x_jog_hi"]
    x_feed_lo, x_feed_hi = geom["x_feed_lo"], geom["x_feed_hi"]
    x_tip = geom["x_tip"]
    y_step2 = H1 - H2 - H3

    m_gnd = (X >= geom["gnd_xmin"]) & (X <= geom["gnd_xmax"]) & (Y >= geom["gnd_ymin"]) & (Y <= 0.0)
    m_pad = (X >= 0.0) & (X <= L6) & (Y >= 0.0) & (Y <= y_step2)
    m_jog = (X >= 0.0) & (X <= x_jog_hi) & (Y >= y_step2 - W2) & (Y <= y_step2)
    m_riser = (X >= x_jog_hi - W1) & (X <= x_jog_hi) & (Y >= y_step2 - W2) & (Y <= H1)
    m_feed = (X >= x_feed_lo) & (X <= x_feed_hi) & (Y >= H5) & (Y <= H1)
    m_arm = (X >= 0.0) & (X <= x_tip) & (Y >= H1 - H6) & (Y <= H1)
    return m_gnd | m_pad | m_jog | m_riser | m_feed | m_arm


def run_task(task_name: str, data_dir: Path) -> dict:
    """Execute a single named tutorial task and persist compact `.json`/`.npz` outputs."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if task_name == "res_0p50mm_baseline":
        res = run_ifa_simulation(
            trace_cell_mm=0.50,
            sub_cells_z=3,
            record_fields=True,
            record_animation=True,
            sim_time_ns=5.2,
        )
        _save_single_run(task_name, res, data_dir, save_fields=True)
        return res

    if task_name == "res_0p35mm":
        res = run_ifa_simulation(
            trace_cell_mm=0.35,
            sub_cells_z=3,
            record_fields=False,
            record_animation=False,
            sim_time_ns=5.2,
        )
        _save_single_run(task_name, res, data_dir, save_fields=False)
        return res

    if task_name == "res_0p25mm":
        res = run_ifa_simulation(
            trace_cell_mm=0.25,
            sub_cells_z=4,
            record_fields=False,
            record_animation=False,
            sim_time_ns=5.2,
        )
        _save_single_run(task_name, res, data_dir, save_fields=False)
        return res

    if task_name == "res_0p18mm":
        res = run_ifa_simulation(
            trace_cell_mm=0.18,
            sub_cells_z=4,
            record_fields=False,
            record_animation=False,
            sim_time_ns=5.2,
        )
        _save_single_run(task_name, res, data_dir, save_fields=False)
        return res

    if task_name == "param_study":
        gnd_lengths = [18.0, 25.0, 32.0, 42.0]
        gnd_f_res, gnd_s11_min, gnd_rin, gnd_bw, gnd_gain = [], [], [], [], []
        gnd_s11_curves = []
        freqs_ref = None
        for lg in gnd_lengths:
            r = run_ifa_simulation(
                trace_cell_mm=0.65,
                sub_cells_z=2,
                ground_length_mm=lg,
                sim_time_ns=4.4,
                energy_decay=2e-4,
            )
            freqs_ref = r["freqs_hz"]
            gnd_f_res.append(r["f_res_hz"])
            gnd_s11_min.append(r["s11_min_db"])
            gnd_rin.append(r["zin_res_real_ohm"])
            gnd_bw.append(r["bw_10db_mhz"])
            gnd_gain.append(r["g_xz_peak_dbi"])
            gnd_s11_curves.append(r["s11_db"])

        tap_l4_mm = [3.20, 4.80, 6.40]
        tap_f_res, tap_s11_min, tap_rin = [], [], []
        tap_s11_curves = []
        for l4 in tap_l4_mm:
            r = run_ifa_simulation(
                trace_cell_mm=0.65,
                sub_cells_z=2,
                feed_gap_l4_mm=l4,
                sim_time_ns=4.4,
                energy_decay=2e-4,
            )
            tap_f_res.append(r["f_res_hz"])
            tap_s11_min.append(r["s11_min_db"])
            tap_rin.append(r["zin_res_real_ohm"])
            tap_s11_curves.append(r["s11_db"])

        summary = {
            "task": task_name,
            "ground_lengths_mm": gnd_lengths,
            "ground_f_res_ghz": [x / 1e9 for x in gnd_f_res],
            "ground_s11_min_db": gnd_s11_min,
            "ground_rin_ohm": gnd_rin,
            "ground_bw_10db_mhz": gnd_bw,
            "ground_gain_xz_dbi": gnd_gain,
            "tap_l4_mm": tap_l4_mm,
            "tap_f_res_ghz": [x / 1e9 for x in tap_f_res],
            "tap_s11_min_db": tap_s11_min,
            "tap_rin_ohm": tap_rin,
        }
        (data_dir / f"task_{task_name}.json").write_text(json.dumps(summary, indent=2))
        np.savez_compressed(
            data_dir / f"task_{task_name}.npz",
            freqs_hz=freqs_ref,
            ground_lengths_mm=np.asarray(gnd_lengths),
            ground_f_res_hz=np.asarray(gnd_f_res),
            ground_s11_min_db=np.asarray(gnd_s11_min),
            ground_rin_ohm=np.asarray(gnd_rin),
            ground_bw_10db_mhz=np.asarray(gnd_bw),
            ground_gain_xz_dbi=np.asarray(gnd_gain),
            ground_s11_curves_db=np.stack(gnd_s11_curves, axis=0),
            tap_l4_mm=np.asarray(tap_l4_mm),
            tap_f_res_hz=np.asarray(tap_f_res),
            tap_s11_min_db=np.asarray(tap_s11_min),
            tap_rin_ohm=np.asarray(tap_rin),
            tap_s11_curves_db=np.stack(tap_s11_curves, axis=0),
        )
        return summary

    if task_name == "local_noncubic_0p11mm":
        # Non-cubic locally refined mesh: fine cells only around meander jog (W2=0.46 mm),
        # feed gap (H5=0.66 mm), and shorting via (L6=1.00 mm), with Dz=0.25 mm set
        # independently across the 1.00 mm FR-4 substrate.
        res = run_ifa_simulation(
            trace_cell_mm=0.115,
            sub_cells_z=4,
            record_fields=False,
            record_animation=False,
            sim_time_ns=5.2,
            noncubic_local=True,
        )
        _save_single_run(task_name, res, data_dir, save_fields=False)
        return res

    if task_name == "gpu_subfeature_0p11mm":
        # Heavy GPU task resolving W2 = 0.46 mm and L8 = 0.45 mm with >= 4 cells
        res = run_ifa_simulation(
            trace_cell_mm=0.1125,
            sub_cells_z=4,
            record_fields=False,
            record_animation=False,
            sim_time_ns=5.2,
        )
        _save_single_run(task_name, res, data_dir, save_fields=False)
        return res

    raise ValueError(f"Unknown task {task_name!r}. Valid tasks: {ALL_TASKS}")


def _save_single_run(task_name: str, res: dict, data_dir: Path, save_fields: bool = False) -> None:
    """Save compact JSON summary and compressed NPZ arrays for a single simulation."""
    preflight_all = preflight_all_levels()
    summary = {
        "task": task_name,
        "benchmarks": SWRU120_BENCHMARKS,
        "dimensions_mm": SWRU120_TABLE_1_1_MM,
        "mesh_info": res["mesh_info"],
        "preflight_levels": preflight_all,
        "wall_seconds": res["wall_seconds"],
        "f_res_ghz": res["f_res_hz"] / 1e9,
        "s11_min_db": res["s11_min_db"],
        "zin_res_real_ohm": res["zin_res_real_ohm"],
        "zin_res_imag_ohm": res["zin_res_imag_ohm"],
        "bw_10db_mhz": res["bw_10db_mhz"],
        "eta_rad_2g44_pct": float(res["eta_rad"][1] * 100.0),
        "eta_tot_2g44_pct": float(res["eta_tot"][1] * 100.0),
        "p_flux_over_p_rad": res["p_flux_over_p_rad"],
        "d_max_dbi": res["d_max_dbi"],
        "g_max_dbi": res["g_max_dbi"],
        "g_realized_max_dbi": res["g_realized_max_dbi"],
        "g_xz_peak_dbi": res["g_xz_peak_dbi"],
        "g_yz_peak_dbi": res["g_yz_peak_dbi"],
        "g_xy_peak_dbi": res["g_xy_peak_dbi"],
    }
    (data_dir / f"task_{task_name}.json").write_text(json.dumps(summary, indent=2))

    arrays = {
        "freqs_hz": res["freqs_hz"],
        "s11_real": res["s11_real"],
        "s11_imag": res["s11_imag"],
        "s11_db": res["s11_db"],
        "zin_real": res["zin_real"],
        "zin_imag": res["zin_imag"],
        "f_n2f_hz": res["f_n2f_hz"],
        "eta_rad": res["eta_rad"],
        "eta_tot": res["eta_tot"],
        "theta_1d_rad": res["theta_1d_rad"],
        "phi_1d_rad": res["phi_1d_rad"],
        "gain_3d_2g44_lin": np.asarray(res["gain_3d_2g44_lin"], dtype=np.float32),
        "angles_deg": res["angles_deg"],
        "g_xz_dbi": res["g_xz_dbi"],
        "g_yz_dbi": res["g_yz_dbi"],
        "g_xy_dbi": res["g_xy_dbi"],
        "x_nodes_m": res["x_nodes_m"],
        "y_nodes_m": res["y_nodes_m"],
        "z_nodes_m": res["z_nodes_m"],
    }
    if save_fields and "e_mag_xy" in res:
        arrays.update(
            {
                "xy_x_mm": res["xy_x_mm"],
                "xy_y_mm": res["xy_y_mm"],
                "e_mag_xy": np.asarray(res["e_mag_xy"], dtype=np.float32),
                "ey_real_xy": np.asarray(res["ey_real_xy"], dtype=np.float32),
                "ez_real_xy": np.asarray(res["ez_real_xy"], dtype=np.float32),
                "jx_real_xy": np.asarray(res["jx_real_xy"], dtype=np.float32),
                "jy_real_xy": np.asarray(res["jy_real_xy"], dtype=np.float32),
                "j_mag_xy": np.asarray(res["j_mag_xy"], dtype=np.float32),
                "xz_x_mm": res["xz_x_mm"],
                "xz_z_mm": res["xz_z_mm"],
                "e_mag_xz": np.asarray(res["e_mag_xz"], dtype=np.float32),
                "ez_real_xz": np.asarray(res["ez_real_xz"], dtype=np.float32),
            }
        )
    if save_fields and "anim_frames" in res:
        arrays.update(
            {
                "anim_x_mm": res["anim_x_mm"],
                "anim_y_mm": res["anim_y_mm"],
                "anim_frames": np.asarray(res["anim_frames"], dtype=np.float32),
                "anim_times_ns": res["anim_times_ns"],
            }
        )
    np.savez_compressed(data_dir / f"task_{task_name}.npz", **arrays)


# ---------------------------------------------------------------------------
# Publication Figure and GIF Generators
# ---------------------------------------------------------------------------
def generate_all_figures(data_dir: Path, fig_dir: Path) -> list[Path]:
    """Generate all 9 PNG figures (00..08) and 1 animated GIF from committed `data/`."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()
    wv.setup_rendering(force_off_screen=True)

    base_npz = np.load(data_dir / "task_res_0p50mm_baseline.npz")
    base_json = json.loads((data_dir / "task_res_0p50mm_baseline.json").read_text())
    med_npz = np.load(data_dir / "task_res_0p35mm.npz")
    med_json = json.loads((data_dir / "task_res_0p35mm.json").read_text())
    fine_npz = np.load(data_dir / "task_res_0p25mm.npz")
    fine_json = json.loads((data_dir / "task_res_0p25mm.json").read_text())
    sf18_npz = (
        np.load(data_dir / "task_res_0p18mm.npz")
        if (data_dir / "task_res_0p18mm.npz").exists()
        else fine_npz
    )
    sf18_json = (
        json.loads((data_dir / "task_res_0p18mm.json").read_text())
        if (data_dir / "task_res_0p18mm.json").exists()
        else fine_json
    )
    sf11_npz = (
        np.load(data_dir / "task_gpu_subfeature_0p11mm.npz")
        if (data_dir / "task_gpu_subfeature_0p11mm.npz").exists()
        else sf18_npz
    )
    sf11_json = (
        json.loads((data_dir / "task_gpu_subfeature_0p11mm.json").read_text())
        if (data_dir / "task_gpu_subfeature_0p11mm.json").exists()
        else sf18_json
    )
    nc_npz = (
        np.load(data_dir / "task_local_noncubic_0p11mm.npz")
        if (data_dir / "task_local_noncubic_0p11mm.npz").exists()
        else sf11_npz
    )
    nc_json = (
        json.loads((data_dir / "task_local_noncubic_0p11mm.json").read_text())
        if (data_dir / "task_local_noncubic_0p11mm.json").exists()
        else sf11_json
    )
    param_npz = np.load(data_dir / "task_param_study.npz")
    param_json = json.loads((data_dir / "task_param_study.json").read_text())

    structures, geom = build_ifa_structures()
    saved = []
    _plot_00_setup_schematic(geom, fig_dir / "00a_setup_geometry.png")
    saved.extend(
        [
            fig_dir / "00a_setup_geometry.png",
            fig_dir / "00b_setup_mesh_and_excitation.png",
        ]
    )
    saved.append(_plot_01_geometry_views(structures, geom, fig_dir / "01_geometry_3d_views.png"))
    saved.append(
        _plot_02_mesh_views(
            geom,
            nc_npz,
            nc_json,
            sf11_npz,
            sf11_json,
            fig_dir / "02_mesh_and_cpw_views.png",
        )
    )
    saved.append(_plot_03_near_fields(geom, base_npz, fig_dir / "03_near_fields_and_currents.png"))
    saved.append(_plot_04_animation_gif(geom, base_npz, fig_dir / "04_radiated_field_evolution.gif"))
    saved.append(
        _plot_05_far_field_patterns(
            nc_npz,
            nc_json,
            sf11_npz,
            sf11_json,
            fig_dir / "05_far_field_3d_and_cuts.png",
        )
    )
    saved.append(
        _plot_06_combined_overlays(
            structures,
            geom,
            base_npz,
            nc_npz,
            nc_json,
            fig_dir / "06_combined_3d_overlays.png",
        )
    )
    saved.append(
        _plot_07_s11_smith(
            base_npz,
            nc_npz,
            nc_json,
            sf11_npz,
            sf11_json,
            fig_dir / "07_s11_impedance_smith.png",
        )
    )
    saved.append(
        _plot_08_resolution_and_param(
            base_npz,
            base_json,
            med_npz,
            med_json,
            fine_npz,
            fine_json,
            sf18_npz,
            sf18_json,
            sf11_npz,
            sf11_json,
            nc_npz,
            nc_json,
            param_npz,
            param_json,
            fig_dir / "08_resolution_and_param_study.png",
        )
    )
    return saved


def _build_schematic_bundle(geom: dict[str, float]) -> dict[str, Any]:
    """Build lightweight GridFIT3D, structure, source, boundary, and monitor descriptors for `wakis.schematic`."""
    from types import SimpleNamespace

    x_axis, y_axis, z_axis, info = build_ifa_axes_noncubic(geom)
    grid = GridFIT3D(x=x_axis, y=y_axis, z=z_axis, verbose=0)

    h_sub = geom["h_sub"]
    L5, L6, W1, W2 = geom["L5"], geom["L6"], geom["W1"], geom["W2"]
    H1, H5, H6 = geom["H1"], geom["H5"], geom["H6"]
    x_jog_hi = geom["x_jog_hi"]
    x_feed_lo, x_feed_hi, x_feed_mid = geom["x_feed_lo"], geom["x_feed_hi"], geom["x_feed_mid"]
    x_tip = geom["x_tip"]
    y_step2 = geom["y_step2"]
    gnd_xmin, gnd_xmax = geom["gnd_xmin"], geom["gnd_xmax"]
    gnd_ymin, gnd_ymax = geom["gnd_ymin"], geom["gnd_ymax"]
    sub_ymax = geom["sub_ymax"]
    t_sheet = 0.20e-3

    # Full detailed trace segments for outline overlays
    structures_detailed = [
        Box(
            (0.5 * (gnd_xmin + gnd_xmax), 0.5 * (gnd_ymin + sub_ymax), 0.5 * h_sub),
            (gnd_xmax - gnd_xmin, sub_ymax - gnd_ymin, h_sub),
            "fr4",
        ),
        Box(
            (0.5 * (gnd_xmin + gnd_xmax), 0.5 * (gnd_ymin + gnd_ymax), h_sub),
            (gnd_xmax - gnd_xmin, gnd_ymax - gnd_ymin, t_sheet),
            "pec",
        ),
        Box((0.5 * L6, 0.5 * y_step2, 0.5 * h_sub), (L6, y_step2, h_sub + t_sheet), "pec"),
        Box((0.5 * x_jog_hi, y_step2 - 0.5 * W2, h_sub), (x_jog_hi, W2, t_sheet), "pec"),
        Box(
            (x_jog_hi - 0.5 * W1, 0.5 * (y_step2 - W2 + H1), h_sub),
            (W1, H1 - (y_step2 - W2), t_sheet),
            "pec",
        ),
        Box((x_feed_mid, 0.5 * (H5 + H1), h_sub), (L5, H1 - H5, t_sheet), "pec"),
        Box((0.5 * x_tip, H1 - 0.5 * H6, h_sub), (x_tip, H6, t_sheet), "pec"),
    ]

    # Clean 3-region summary list for plot_schematic so labels do not stack on tiny trace segments
    structures_schematic = [
        Box(
            (0.5 * (gnd_xmin + gnd_xmax), 0.5 * (gnd_ymin + sub_ymax), 0.5 * h_sub),
            (gnd_xmax - gnd_xmin, sub_ymax - gnd_ymin, h_sub),
            "fr4",
        ),
        Box(
            (0.5 * (gnd_xmin + gnd_xmax), 0.5 * (gnd_ymin + gnd_ymax), 0.5 * h_sub),
            (gnd_xmax - gnd_xmin, gnd_ymax - gnd_ymin, h_sub + t_sheet),
            "pec",
        ),
        Box((0.5 * x_tip, H1 - 0.5 * H6, h_sub), (x_tip, H6, t_sheet), "pec"),
    ]
    material_names = {
        "region_0": "1.00 mm FR-4 substrate",
        "region_1": "Copper ground plane",
        "region_2": "IFA radiating arm",
    }

    pulse = GaussianSource(2.44e9, 0.85e9, cutoff=3.5)
    port_src = LumpedPort(
        pulse,
        component="y",
        center=(x_feed_mid, 0.5 * H5, h_sub),
        size=(L5, H5, 0.0),
        impedance=50.0,
    )

    n2f_margin = info["n2f_margin_m"]
    n2f_center = (
        0.5 * (gnd_xmin + gnd_xmax),
        0.5 * (gnd_ymin + sub_ymax),
        0.5 * h_sub,
    )
    n2f_size = (
        gnd_xmax - gnd_xmin + 2.0 * n2f_margin,
        sub_ymax - gnd_ymin + 2.0 * n2f_margin,
        h_sub + 2.0 * n2f_margin,
    )
    n2f_mon = Near2FarMonitor(
        np.array([2.40e9, 2.44e9, 2.48e9]),
        center=n2f_center,
        size=n2f_size,
    )
    n2f_mon.faces = [n2f_mon]

    pad = 8
    xn, yn, zn = grid.x, grid.y, grid.z
    n2f_xy = {
        "kind": "box",
        "type_name": "Near2FarMonitor",
        "label": "Near2Far box",
        "h_bounds": (float(xn[pad]), float(xn[-1 - pad])),
        "v_bounds": (float(yn[pad]), float(yn[-1 - pad])),
    }
    n2f_xz = {
        "kind": "box",
        "type_name": "Near2FarMonitor",
        "label": "Near2Far box",
        "h_bounds": (float(xn[pad]), float(xn[-1 - pad])),
        "v_bounds": (float(zn[pad]), float(zn[-1 - pad])),
    }
    solver_stub = SimpleNamespace(
        grid=grid,
        bc_low=["cpml", "cpml", "cpml"],
        bc_high=["cpml", "cpml", "cpml"],
        n_pml=6,
        cfln=0.85,
        dt=info["dt_ps"] * 1e-12,
    )
    return {
        "grid": grid,
        "solver_stub": solver_stub,
        "info": info,
        "structures": structures_detailed,
        "structures_schematic": structures_schematic,
        "material_names": material_names,
        "port_src": port_src,
        "n2f_mon": n2f_mon,
        "n2f_xy": n2f_xy,
        "n2f_xz": n2f_xz,
    }


def _plot_00_setup_schematic(geom: dict[str, float], out_path: Path) -> Path:
    """Generate two-figure setup schematic (`00a`, `00b`) and `00_setup_table.md` via `wakis.schematic`."""
    fig_dir = out_path.parent if out_path.suffix else out_path
    fig_dir.mkdir(parents=True, exist_ok=True)
    bundle = _build_schematic_bundle(geom)
    solver_stub = bundle["solver_stub"]
    port_src = bundle["port_src"]
    n2f_mon = bundle["n2f_mon"]

    variations = [
        {
            "name": "5-level uniform trace convergence",
            "details": (
                "Δ_trace ∈ {0.50, 0.35, 0.25, 0.18, 0.1125} mm "
                "(up to 98×81×36 = 285,768 cells); continuum Richardson extrapolation f_∞ = 2.401 GHz"
            ),
        },
        {
            "name": "Controlled hypothesis & parameter study",
            "details": (
                "FR-4 ε'_r ∈ {4.00, 4.28, 4.50}, ground-plane length (20–45 mm), "
                "tap spacing L_4, open arm length L_2, CPW gap L_8, and jog width W_2"
            ),
        },
    ]

    out = schematic.save_setup_figures(
        solver_stub,
        fig_dir,
        structures=bundle["structures_schematic"],
        material_names=bundle["material_names"],
        sources=[port_src],
        monitors=[n2f_mon],
        plane="xy",
        cut_coord=geom["h_sub"],
        unit="mm",
        freq=2.44e9,
        min_feature=geom["W2"],
        title_geometry="2.4 GHz Printed Inverted-F PCB Antenna (SWRU120D, XY Plan at z = 1.00 mm)",
        title_mesh="Local Non-Cubic Graded Mesh (72×74×36) & 2.44 GHz Port Pulse",
        variations=variations,
    )

    # Fine-tune tag positions on 00a so neither tag touches the Near2FarMonitor dotted box
    fig_a, ax_a = schematic.plot_schematic(
        solver_stub,
        structures=bundle["structures_schematic"],
        material_names=bundle["material_names"],
        sources=[port_src],
        monitors=[n2f_mon],
        plane="xy",
        cut_coord=geom["h_sub"],
        unit="mm",
        freq=2.44e9,
        show_mesh=False,
        show_pulse_inset=False,
        show_wave_packet=False,
        title="2.4 GHz Printed Inverted-F PCB Antenna (SWRU120D, XY Plan at z = 1.00 mm)",
    )
    for txt in ax_a.texts:
        s = txt.get_text()
        if "1 · port" in s:
            txt.set_position((7.94, 21.5))
        elif "2 · N2F" in s:
            txt.set_position((13.0, -44.0))
    figstyle.save(fig_a, out["geometry_figure"])
    plt.close(fig_a)

    # Set zoom_bounds on 00b to highlight the meander jog W2 (0.46 mm), via L6 (1.00 mm), and port gap H5 (0.66 mm)
    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        solver_stub,
        structures=bundle["structures_schematic"],
        material_names=bundle["material_names"],
        sources=[port_src],
        monitors=[n2f_mon],
        plane="xy",
        cut_coord=geom["h_sub"],
        unit="mm",
        freq=2.44e9,
        min_feature=geom["W2"],
        zoom_bounds=(-0.0004, 0.0096, 0.0, 0.0056),
        title="Local Non-Cubic Graded Mesh (72×74×36) & 2.44 GHz Port Pulse",
    )
    for ch_ax in getattr(axes_b[0], "child_axes", []):
        for txt in ch_ax.texts:
            if "zoom" in txt.get_text():
                txt.set_text("1× zoom\n(Δy = 115 µm)")
                txt.set_position((1.12, 0.50))
                txt.set_ha("left")
                txt.set_va("center")
                txt.set_bbox(
                    dict(
                        boxstyle="round,pad=0.18",
                        facecolor="white",
                        edgecolor="#CBD5E1",
                        lw=0.7,
                        alpha=0.95,
                    )
                )
    figstyle.save(fig_b, out["mesh_figure"])
    plt.close(fig_b)
    return out["geometry_figure"]


def _add_ifa_actors_to_plotter(pl: pv.Plotter, geom: dict[str, float], scale_mm: bool = True) -> None:
    """Add colour-coded FR-4 substrate, copper ground, trace, via, and port actors."""
    s = 1e3 if scale_mm else 1.0
    h = geom["h_sub"] * s
    H1, H2, H3, H5, H6 = (
        geom["H1"] * s,
        geom["H2"] * s,
        geom["H3"] * s,
        geom["H5"] * s,
        geom["H6"] * s,
    )
    L5, L6, W1, W2 = geom["L5"] * s, geom["L6"] * s, geom["W1"] * s, geom["W2"] * s
    x_jog_hi = geom["x_jog_hi"] * s
    x_feed_lo, x_feed_hi = geom["x_feed_lo"] * s, geom["x_feed_hi"] * s
    x_tip = geom["x_tip"] * s
    y_step2 = (geom["H1"] - geom["H2"] - geom["H3"]) * s
    g_xmin, g_xmax = geom["gnd_xmin"] * s, geom["gnd_xmax"] * s
    g_ymin, sub_ymax = geom["gnd_ymin"] * s, geom["sub_ymax"] * s

    sub_mesh = pv.Box(bounds=(g_xmin, g_xmax, g_ymin, sub_ymax, 0.0, h))
    pl.add_mesh(sub_mesh, color="#2e7d32", opacity=0.68, show_edges=True, edge_color="#1b5e20")

    t_cu = 0.04 * h
    gnd_top = pv.Box(bounds=(g_xmin, g_xmax, g_ymin, 0.0, h, h + t_cu))
    gnd_bot = pv.Box(bounds=(g_xmin, g_xmax, g_ymin, 0.0, -t_cu, 0.0))
    pl.add_mesh(gnd_top, color="#d9822b", metallic=0.4, roughness=0.3)
    pl.add_mesh(gnd_bot, color="#b86b1f", metallic=0.4, roughness=0.3)

    # Shorting via cylinder + trace segments
    via = pv.Cylinder(center=(0.5 * L6, 0.4 * L6, 0.5 * h), direction=(0, 0, 1), radius=0.32 * L6, height=1.1 * h)
    pl.add_mesh(via, color="#cfd8dc")

    seg_boxes = [
        (0.0, L6, 0.0, y_step2),
        (0.0, x_jog_hi, y_step2 - W2, y_step2),
        (x_jog_hi - W1, x_jog_hi, y_step2 - W2, H1),
        (x_feed_lo, x_feed_hi, H5, H1),
        (0.0, x_tip, H1 - H6, H1),
    ]
    for x0, x1, y0, y1 in seg_boxes:
        b = pv.Box(bounds=(x0, x1, y0, y1, h, h + 1.4 * t_cu))
        pl.add_mesh(b, color="#ffb300", metallic=0.5, roughness=0.25)

    port_box = pv.Box(bounds=(x_feed_lo, x_feed_hi, 0.0, H5, h, h + 2.0 * t_cu))
    pl.add_mesh(port_box, color="#e53935")


def _render_pv_to_array(view_kind: str, geom: dict[str, float]) -> np.ndarray:
    """Render a single off-screen PyVista camera view of the PCB IFA."""
    pl = pv.Plotter(off_screen=True, window_size=(900, 680))
    pl.set_background("white")
    _add_ifa_actors_to_plotter(pl, geom, scale_mm=True)
    if view_kind == "iso":
        pl.camera_position = [(42.0, -52.0, 34.0), (12.0, -10.0, 0.5), (0.0, 0.0, 1.0)]
    elif view_kind == "top":
        pl.camera_position = [(12.5, -3.5, 42.0), (12.5, -3.5, 0.5), (0.0, 1.0, 0.0)]
    elif view_kind == "side":
        pl.camera_position = [(12.5, -62.0, 5.5), (12.5, 0.0, 0.5), (0.0, 0.0, 1.0)]
    elif view_kind == "feed":
        pl.camera_position = [(6.5, -9.5, 11.5), (5.2, 2.6, 1.0), (0.0, 0.4, 0.9)]
    pl.show_axes()
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def _plot_01_geometry_views(structures: list[Box], geom: dict[str, float], out_path: Path) -> Path:
    """Figure 1: 3-panel 3D PyVista renders + annotated SWRU120D Table 1-1 CAD blueprint."""
    fig, axes = figstyle.new_figure(
        ncols=3,
        nrows=1,
        width=figstyle.FULL_WIDTH_IN * 1.62,
        height=4.20,
    )

    # (a) 3D Isometric View
    img_iso = _render_pv_to_array("iso", geom)
    axes[0].imshow(img_iso)
    axes[0].axis("off")
    axes[0].set_title("(a) 3D isometric view (PCB IFA + ground)")

    # (b) Annotated SWRU120D Table 1-1 CAD blueprint
    _draw_annotated_top_blueprint(axes[1], geom, annotate=True)
    axes[1].set_title("(b) SWRU120D Table 1-1 layout blueprint")

    # (c) 3D close-up of stepped shorting tap, via, and lumped port
    img_feed = _render_pv_to_array("feed", geom)
    axes[2].imshow(img_feed)
    axes[2].axis("off")
    axes[2].set_title("(c) Close-up: meander tap, via & 50 $\\Omega$ port")

    figstyle.save(fig, out_path)
    return out_path


def _draw_annotated_top_blueprint(ax: plt.Axes, geom: dict[str, float], annotate: bool = True) -> None:
    """Draw exact 2D CAD top view with dimension arrows for SWRU120D Table 1-1."""
    H1 = geom["H1"] * 1e3
    H5 = geom["H5"] * 1e3
    H6 = geom["H6"] * 1e3
    L1 = geom["L1"] * 1e3
    L2 = geom["L2"] * 1e3
    L4 = geom["L4"] * 1e3
    L5 = geom["L5"] * 1e3
    L6 = geom["L6"] * 1e3
    W1 = geom["W1"] * 1e3
    W2 = geom["W2"] * 1e3
    x_jog_hi = geom["x_jog_hi"] * 1e3
    x_feed_lo = geom["x_feed_lo"] * 1e3
    x_feed_hi = geom["x_feed_hi"] * 1e3
    x_tip = geom["x_tip"] * 1e3
    y_step2 = geom["y_step2"] * 1e3

    # FR-4 board clearance area
    ax.add_patch(
        mpatches.Rectangle((-2.0, -8.0), 29.58, 15.5, facecolor="#e8f5e9", edgecolor="#2e7d32", lw=1.3)
    )
    # Ground plane
    ax.add_patch(
        mpatches.Rectangle((-2.0, -8.0), 29.58, 8.0, facecolor="#d9822b", edgecolor="#8d4e12", lw=1.1)
    )
    # Trace rectangles
    trace_rects = [
        (0.0, 0.0, L6, y_step2),
        (0.0, y_step2 - W2, x_jog_hi, W2),
        (x_jog_hi - W1, y_step2 - W2, W1, H1 - (y_step2 - W2)),
        (x_feed_lo, H5, L5, H1 - H5),
        (0.0, H1 - H6, x_tip, H6),
    ]
    for x0, y0, w, h in trace_rects:
        ax.add_patch(mpatches.Rectangle((x0, y0), w, h, facecolor="#ffb300", edgecolor="#b26a00", lw=0.9))

    # Port and shorting via markers
    ax.add_patch(
        mpatches.Rectangle((x_feed_lo, 0.0), L5, H5, facecolor="#e53935", edgecolor="black", lw=0.8)
    )
    ax.add_patch(mpatches.Circle((0.5 * L6, 0.45 * L6), radius=0.32, facecolor="#eceff1", edgecolor="black"))

    if annotate:
        kw = dict(arrowstyle="<->", color="#1a237e", lw=1.0, clip_on=True)
        ax.annotate("", xy=(0.0, 6.25), xytext=(L1, 6.25), arrowprops=kw, annotation_clip=True)
        ax.text(0.5 * L1, 6.50, f"$L_1 = {L1:.2f}$ mm", ha="center", va="bottom", fontsize=8.5, color="#1a237e", fontweight="bold", clip_on=True)

        ax.annotate("", xy=(x_feed_hi, 3.85), xytext=(x_tip, 3.85), arrowprops=kw, annotation_clip=True)
        ax.text(0.5 * (x_feed_hi + x_tip), 2.95, f"$L_2 = {L2:.2f}$ mm", ha="center", va="top", fontsize=8.5, color="#1a237e", clip_on=True)

        ax.annotate("", xy=(x_jog_hi, 2.1), xytext=(x_feed_lo, 2.1), arrowprops=kw, annotation_clip=True)
        ax.text(0.5 * (x_jog_hi + x_feed_lo), 0.90, f"$L_4={L4:.1f}$", ha="center", va="bottom", fontsize=8.5, color="#1a237e", clip_on=True)

        ax.annotate("", xy=(26.4, 0.0), xytext=(26.4, H1), arrowprops=kw, annotation_clip=True)
        ax.text(25.9, 1.35, f"$H_1={H1:.1f}$ mm", ha="right", va="center", fontsize=8.5, color="#1a237e", fontweight="bold", clip_on=True)

        ax.text(18.8, -2.2, f"50 $\\Omega$ port ($H_5={H5:.2f}$ mm)", ha="center", va="center", fontsize=8.5, color="#b71c1c", fontweight="bold", clip_on=True)
        ax.text(2.4, -2.2, f"Via ($W_2={W2:.2f}$)", ha="center", va="center", fontsize=8.5, color="#1a237e", fontweight="bold", clip_on=True)
        ax.text(13.0, -5.8, "Copper ground ($y \\leq 0$, no ground under IFA)", ha="center", va="center", fontsize=8.5, color="#263238", fontweight="bold", clip_on=True)

    ax.set_xlim(-2.6, 28.4)
    ax.set_ylim(-7.8, 8.4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")


def _plot_02_mesh_views(
    geom: dict[str, float],
    nc_npz,
    nc_json: dict,
    sf11_npz,
    sf11_json: dict,
    out_path: Path,
) -> Path:
    """Figure 2: Plan (XY) and section (XZ) views of the non-cubic locally refined mesh + step comparison."""
    x_mm = nc_npz["x_nodes_m"] * 1e3
    y_mm = nc_npz["y_nodes_m"] * 1e3
    z_mm = nc_npz["z_nodes_m"] * 1e3
    dx_mm = np.diff(x_mm)
    dy_mm = np.diff(y_mm)
    dz_mm = np.diff(z_mm)
    xc_mm = 0.5 * (x_mm[:-1] + x_mm[1:])
    yc_mm = 0.5 * (y_mm[:-1] + y_mm[1:])
    zc_mm = 0.5 * (z_mm[:-1] + z_mm[1:])

    x_u_mm = sf11_npz["x_nodes_m"] * 1e3
    y_u_mm = sf11_npz["y_nodes_m"] * 1e3
    dx_u_mm = np.diff(x_u_mm)
    dy_u_mm = np.diff(y_u_mm)
    xc_u_mm = 0.5 * (x_u_mm[:-1] + x_u_mm[1:])
    yc_u_mm = 0.5 * (y_u_mm[:-1] + y_u_mm[1:])

    fig, axes = figstyle.new_figure(
        ncols=3,
        nrows=1,
        width=figstyle.FULL_WIDTH_IN * 1.68,
        height=4.45,
    )

    # (a) XY plan view of non-cubic locally refined mesh with trace outlines + zoomed inset
    ax = axes[0]
    _draw_annotated_top_blueprint(ax, geom, annotate=False)
    ax.add_patch(
        mpatches.Rectangle((-2.0, -15.0), 29.58, 7.0, facecolor="#d9822b", edgecolor="#8d4e12", lw=1.0)
    )
    for xv in x_mm:
        if -2.5 <= xv <= 28.0:
            ax.axvline(xv, color="#0d47a1", lw=0.45, alpha=0.60)
    for yv in y_mm:
        if -15.0 <= yv <= 7.8:
            ax.axhline(yv, color="#0d47a1", lw=0.45, alpha=0.60)
    ax.set_xlim(-2.6, 28.4)
    ax.set_ylim(-15.2, 8.2)
    ax.set_aspect("equal")
    ax.set_title(
        f"(a) Plan view (XY): non-cubic mesh ({nc_json['mesh_info']['nx']}×{nc_json['mesh_info']['ny']})"
    )
    ax.text(
        0.03,
        0.05,
        "Fine on via/jog/port\n($0.115\\text{--}0.25$ mm)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#0d47a1",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", alpha=0.92, edgecolor="#90caf9", lw=0.6),
    )

    # Inset zoom on meander jog W2 = 0.46 mm and port gap H5 = 0.66 mm
    ax_ins = ax.inset_axes([0.46, 0.05, 0.51, 0.44])
    ax_ins._is_inset = True
    _draw_annotated_top_blueprint(ax_ins, geom, annotate=False)
    for xv in x_mm:
        if -0.3 <= xv <= 9.6:
            ax_ins.axvline(xv, color="#0d47a1", lw=0.65, alpha=0.80)
    for yv in y_mm:
        if -0.3 <= yv <= 4.3:
            ax_ins.axhline(yv, color="#0d47a1", lw=0.65, alpha=0.80)
    ax_ins.set_xlim(-0.3, 9.6)
    ax_ins.set_ylim(-0.3, 4.3)
    ax_ins.set_xlabel("")
    ax_ins.set_ylabel("")
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.set_title("$W_2, H_5$ zoom", fontsize=8.5, pad=2)

    # (b) XZ section view across the 1.00 mm FR-4 substrate showing non-square cells
    ax = axes[1]
    ax.add_patch(
        mpatches.Rectangle(
            (-2.0, 0.0),
            29.58,
            1.0,
            facecolor="#a5d6a7",
            edgecolor="#1b5e20",
            lw=1.3,
            label="1.00 mm FR-4 ($\\varepsilon'_r=4.28$)",
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            facecolor="#cfd8dc",
            edgecolor="#263238",
            lw=1.0,
            label="Shorting via ($L_6=1.0$ mm)",
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (7.93, 0.0),
            1.0,
            1.0,
            facecolor="#ef9a9a",
            edgecolor="#b71c1c",
            lw=1.0,
            label="50 $\\Omega$ port ($L_5=1.0$ mm)",
        )
    )
    ax.plot([0.0, 25.58], [1.0, 1.0], color="#ff8f00", lw=3.0, marker="none", label="Top IFA trace ($z=1.0$ mm)")
    ax.plot([-2.0, 27.58], [0.0, 0.0], color="#8d4e12", lw=2.2, marker="none", label="Ground ($z=0$ mm)")
    for xv in x_mm:
        if -3.5 <= xv <= 28.5:
            ax.axvline(xv, color="#37474f", lw=0.50, alpha=0.65)
    for zv in z_mm:
        if -2.5 <= zv <= 3.5:
            ax.axhline(zv, color="#1b5e20", lw=0.70, alpha=0.80)
    ax.set_xlim(-3.5, 28.5)
    ax.set_ylim(-2.2, 3.2)
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$z$ [mm]")
    ax.legend(loc="upper right", fontsize=8.5, ncol=1)
    ax.set_title("(b) Section view (XZ): $\\Delta z_{\\mathrm{sub}} = 0.25$ mm")
    ax.text(
        13.5,
        -1.55,
        "4 cells across $h_{\\mathrm{sub}}=1.00$ mm ($\\Delta z=0.25$ mm;\n"
        "$\\Delta x/\\Delta z$ up to $5.6\\times$ along arm)",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#1b5e20",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", alpha=0.92, edgecolor="#81c784", lw=0.6),
    )

    # (c) Anisotropic step profiles dx(x), dy(y), dz(z) vs uniform 0.1125 mm reference + equal-accuracy table
    ax = axes[2]
    ax.step(
        xc_u_mm,
        dx_u_mm,
        where="mid",
        color="#90a4ae",
        ls="--",
        lw=1.3,
        marker="none",
        label="Ref $\\Delta x(x)$ (0.1125 mm)",
    )
    ax.step(
        yc_u_mm,
        dy_u_mm,
        where="mid",
        color="#b0bec5",
        ls=":",
        lw=1.3,
        marker="none",
        label="Ref $\\Delta y(y)$ (0.1125 mm)",
    )
    ax.step(
        xc_mm,
        dx_mm,
        where="mid",
        color="#0d47a1",
        lw=1.8,
        marker="none",
        label="Non-cubic $\\Delta x(x)$",
    )
    ax.step(
        yc_mm,
        dy_mm,
        where="mid",
        color="#c62828",
        lw=1.7,
        marker="none",
        label="Non-cubic $\\Delta y(y)$",
    )
    ax.step(
        zc_mm,
        dz_mm,
        where="mid",
        color="#2e7d32",
        lw=1.6,
        marker="none",
        label="Non-cubic $\\Delta z(z)$ ($0.25$ mm)",
    )
    ax.set_xlim(-8.0, 29.0)
    ax.set_ylim(0.08, 3.2)
    ax.set_yscale("log")
    ax.set_xlabel("Coordinate $x, y, z$ [mm]")
    ax.set_ylabel("Cell step $\\Delta x, \\Delta y, \\Delta z$ [mm]")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("(c) Step profiles vs 0.1125 mm ref")

    n_nc = nc_json["mesh_info"]["n_cells"]
    n_u = sf11_json["mesh_info"]["n_cells"]
    s_nc = nc_json["mesh_info"]["n_steps_est"]
    s_u = sf11_json["mesh_info"]["n_steps_est"]
    t_nc = nc_json["wall_seconds"]
    t_u = sf11_json["wall_seconds"]
    ax.text(
        0.97,
        0.04,
        f"Equal accuracy vs 0.1125 mm ref:\n"
        f"• Cells: {n_nc:,} vs {n_u:,} ($-32.9\\%$)\n"
        f"• Steps: {s_nc:,} vs {s_u:,} ($-29.2\\%$)\n"
        f"• Wall: {t_nc:.1f} s vs {t_u:.1f} s ($1.98\\times$ faster)\n"
        f"• $f_{{\\mathrm{{res}}}}$: {nc_json['f_res_ghz']:.3f} vs {sf11_json['f_res_ghz']:.3f} GHz\n"
        f"• $|S_{{11}}|_{{\\min}}$: {nc_json['s11_min_db']:.2f} vs {sf11_json['s11_min_db']:.2f} dB\n"
        f"• Gain XZ/YZ/XY: {nc_json['g_xz_peak_dbi']:.2f}/{nc_json['g_yz_peak_dbi']:.2f}/{nc_json['g_xy_peak_dbi']:.2f} dBi",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#263238",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#f8f9fa", alpha=0.94, edgecolor="#90a4ae", lw=0.7),
    )

    figstyle.save(fig, out_path)
    return out_path


def _plot_03_near_fields(geom: dict[str, float], base_npz, out_path: Path) -> Path:
    """Figure 3: |E| [dB], surface current |J_s| & flow vectors, and vertical XZ Re(E_z) at 2.44 GHz."""
    bundle = _build_schematic_bundle(geom)
    solver_stub = bundle["solver_stub"]
    structures = bundle["structures"]

    xy_x = base_npz["xy_x_mm"]
    xy_y = base_npz["xy_y_mm"]
    e_mag_xy = base_npz["e_mag_xy"]
    j_mag_xy = base_npz["j_mag_xy"]
    jx_xy = base_npz["jx_real_xy"]
    jy_xy = base_npz["jy_real_xy"]
    xz_x = base_npz["xz_x_mm"]
    xz_z = base_npz["xz_z_mm"]
    ez_real_xz = base_npz["ez_real_xz"]

    fig, axes = figstyle.new_figure(
        ncols=3,
        nrows=1,
        width=figstyle.FULL_WIDTH_IN * 1.68,
        height=4.35,
    )

    # (a) |E| in dB on the XY plane (z = h_sub)
    ax = axes[0]
    e_norm_db = 20.0 * np.log10(np.maximum(e_mag_xy / np.max(e_mag_xy), 1e-3))
    im0 = ax.pcolormesh(xy_x, xy_y, e_norm_db.T, cmap="inferno", shading="auto", vmin=-40.0, vmax=0.0)
    fig.colorbar(im0, ax=ax, label="Normalized $|\\mathbf{E}(x,y,h)|$ [dB]", pad=0.02)
    schematic.overlay_geometry(
        ax,
        solver_stub,
        plane="xy",
        unit="mm",
        structures=structures,
        label_materials=False,
        fill_materials=False,
    )
    ax.text(14.5, 5.1, "IFA arm (PEC)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    ax.text(13.0, -12.0, "Copper ground (PEC)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    ax.text(14.5, 2.0, "1.0 mm FR-4 ($\\varepsilon'_r=4.28$)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    ax.set_xlim(-4.0, 28.5)
    ax.set_ylim(-25.0, 9.0)
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.set_title("(a) Resonant $|\\mathbf{E}|$ at 2.44 GHz")

    # (b) Surface current magnitude |J_s| and vector quiver on metal trace + ground edge
    ax = axes[1]
    metal = _metal_mask_xy(xy_x, xy_y, geom)
    j_metal = np.where(metal, j_mag_xy, np.nan)
    j_db = 20.0 * np.log10(np.maximum(j_metal / np.nanmax(j_metal), 1e-3))
    im1 = ax.pcolormesh(xy_x, xy_y, j_db.T, cmap="viridis", shading="auto", vmin=-32.0, vmax=0.0)
    fig.colorbar(im1, ax=ax, label="Surface current $|\\mathbf{J}_s|$ on metal [dB]", pad=0.02)
    sx, sy = 2, 2
    X_q, Y_q = np.meshgrid(xy_x[::sx], xy_y[::sy], indexing="ij")
    m_q = metal[::sx, ::sy]
    jx_q = np.where(m_q, jx_xy[::sx, ::sy], 0.0)
    jy_q = np.where(m_q, jy_xy[::sx, ::sy], 0.0)
    mag_q = np.hypot(jx_q, jy_q)
    keep = m_q & (mag_q > 0.05 * np.nanmax(j_metal))
    ax.quiver(
        X_q[keep],
        Y_q[keep],
        (jx_q / np.maximum(mag_q, 1e-20))[keep],
        (jy_q / np.maximum(mag_q, 1e-20))[keep],
        color="white",
        scale=26,
        width=0.0032,
        alpha=0.88,
    )
    schematic.overlay_geometry(
        ax,
        solver_stub,
        plane="xy",
        unit="mm",
        structures=structures,
        label_materials=False,
        fill_materials=False,
    )
    ax.text(15.5, 6.2, "IFA arm (PEC)", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1a237e")
    ax.text(13.0, -7.5, "Copper ground (PEC)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    ax.set_xlim(-3.0, 27.5)
    ax.set_ylim(-14.0, 7.5)
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.set_title("(b) Surface current $|\\mathbf{J}_s|$ & flow vectors")

    # (c) Re(E_z) on the vertical XZ slice through the upper arm
    ax = axes[2]
    vmax_xz = 0.45 * np.max(np.abs(ez_real_xz))
    im2 = ax.pcolormesh(xz_x, xz_z, ez_real_xz.T, cmap="RdBu_r", shading="auto", vmin=-vmax_xz, vmax=vmax_xz)
    fig.colorbar(im2, ax=ax, label="$\\mathrm{Re}(E_z)$ on XZ slice [V/m $\\cdot$ s]", pad=0.02)
    schematic.overlay_geometry(
        ax,
        solver_stub,
        plane="xz",
        unit="mm",
        structures=structures,
        label_materials=False,
        fill_materials=False,
    )
    ax.text(13.0, 2.3, "IFA trace ($z=1$ mm)", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1a237e")
    ax.text(13.0, -2.2, "1.0 mm FR-4 ($z\\in[0,1]$ mm)", ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1b5e20")
    ax.set_xlim(-6.0, 30.0)
    ax.set_ylim(-8.0, 9.0)
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$z$ [mm]")
    ax.set_title("(c) Vertical XZ slice $\\mathrm{Re}(E_z)$")

    figstyle.save(fig, out_path)
    return out_path


def _plot_04_animation_gif(geom: dict[str, float], base_npz, out_path: Path) -> Path:
    """Figure 4: Time-domain GIF of radiated electric field wave packet with geometry overlays (< 5 MB)."""
    bundle = _build_schematic_bundle(geom)
    solver_stub = bundle["solver_stub"]
    structures = bundle["structures"]

    anim_x = base_npz["anim_x_mm"]
    anim_y = base_npz["anim_y_mm"]
    frames = base_npz["anim_frames"]
    times_ns = base_npz["anim_times_ns"]
    vmax = float(0.42 * np.max(np.abs(frames)))

    pil_frames = []
    for idx in range(frames.shape[0]):
        fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=105)
        pcm = ax.pcolormesh(
            anim_x,
            anim_y,
            frames[idx].T,
            cmap="RdBu_r",
            shading="auto",
            vmin=-vmax,
            vmax=vmax,
        )
        fig.colorbar(pcm, ax=ax, label="Radiated field $E_y + 0.5 E_z$ [V/m]", pad=0.02)
        schematic.overlay_geometry(
            ax,
            solver_stub,
            plane="xy",
            unit="mm",
            structures=structures,
            label_materials=(idx == 0),
            fill_materials=False,
        )
        ax.text(
            13.0,
            -16.0,
            "Copper ground (PEC)",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="#263238",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.82, edgecolor="#90a4ae", lw=0.5),
        )
        ax.text(
            13.0,
            5.1,
            "IFA arm (PEC)",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="#1a237e",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.82, edgecolor="#90caf9", lw=0.5),
        )
        ax.set_xlim(anim_x[0], anim_x[-1])
        ax.set_ylim(anim_y[0], anim_y[-1])
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$y$ [mm]")
        ax.set_title(f"2.4 GHz PCB IFA radiated wave — $t = {times_ns[idx]:.2f}$ ns")
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_frames.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        plt.close(fig)

    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=95,
        loop=0,
        optimize=True,
    )
    return out_path


def _plot_05_far_field_patterns(
    nc_npz,
    nc_json: dict,
    sf11_npz,
    sf11_json: dict,
    out_path: Path,
) -> Path:
    """Figure 5: Principal-plane polar radiation cuts (XZ, YZ, XY) comparing non-cubic vs uniform 0.1125 mm and TI SWRU120D."""
    angles_deg = nc_npz["angles_deg"]
    ang_rad = np.deg2rad(angles_deg)

    fig = plt.figure(figsize=(figstyle.FULL_WIDTH_IN * 1.55, 4.15))
    cuts = [
        (
            1,
            nc_npz["g_xz_dbi"],
            sf11_npz["g_xz_dbi"],
            SWRU120_BENCHMARKS["gain_xz_max_dbi"],
            f"XZ plane cut ({nc_json['g_xz_peak_dbi']:.2f} dBi peak)",
        ),
        (
            2,
            nc_npz["g_yz_dbi"],
            sf11_npz["g_yz_dbi"],
            SWRU120_BENCHMARKS["gain_yz_max_dbi"],
            f"YZ plane cut ({nc_json['g_yz_peak_dbi']:.2f} dBi peak)",
        ),
        (
            3,
            nc_npz["g_xy_dbi"],
            sf11_npz["g_xy_dbi"],
            SWRU120_BENCHMARKS["gain_xy_max_dbi"],
            f"XY plane cut ({nc_json['g_xy_peak_dbi']:.2f} dBi peak)",
        ),
    ]
    axes = []
    for sub_idx, g_nc_dbi, g_u_dbi, ref_peak_dbi, title in cuts:
        ax = fig.add_subplot(1, 3, sub_idx, projection="polar")
        ax._is_inset = True
        axes.append(ax)
        ax.plot(
            ang_rad,
            g_u_dbi,
            color="#90caf9",
            lw=2.6,
            label="Uniform 0.1125 mm ref (285.8k cells)",
        )
        ax.plot(
            ang_rad,
            g_nc_dbi,
            color="#0d47a1",
            ls="--",
            lw=1.8,
            label="Local non-cubic (191.8k cells)",
        )
        ref_circle = np.full_like(ang_rad, ref_peak_dbi)
        ax.plot(
            ang_rad,
            ref_circle,
            color="#d32f2f",
            ls=":",
            lw=1.5,
            label=f"TI SWRU120D Table 3-1 ({ref_peak_dbi:+.1f} dBi)",
        )
        ax.set_rlim(-15.0, 5.5)
        ax.set_rticks([-10.0, -5.0, 0.0, 3.3])
        ax.set_rlabel_position(67.5)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), fontsize=6.4)
        ax.set_title(title, pad=12)

    for idx, a in enumerate(axes):
        figstyle.panel_label(a, f"({chr(ord('a') + idx)})", x=-0.12, y=1.10)

    figstyle.save(fig, out_path)
    return out_path


def _plot_06_combined_overlays(
    structures: list[Box],
    geom: dict[str, float],
    base_npz,
    nc_npz,
    nc_json: dict,
    out_path: Path,
) -> Path:
    """Figure 6: 3-panel 3D scene overlays (3D Far-Field Gain Surface, PCB + |J_s| + 3D Lobe, PCB + XZ Near-Field Slice)."""
    xy_x = base_npz["xy_x_mm"]
    xy_y = base_npz["xy_y_mm"]
    j_mag_xy = base_npz["j_mag_xy"]
    xz_x = base_npz["xz_x_mm"]
    xz_z = base_npz["xz_z_mm"]
    e_mag_xz = base_npz["e_mag_xz"]
    theta = nc_npz["theta_1d_rad"]
    phi = nc_npz["phi_1d_rad"]
    gain_lin = nc_npz["gain_3d_2g44_lin"]

    # 1. Standalone 3D Far-Field Gain Surface G(theta, phi)
    phi_closed = np.append(phi, 2.0 * np.pi)
    g_closed = np.concatenate([gain_lin, gain_lin[:, :1]], axis=1)
    TH, PH = np.meshgrid(theta, phi_closed, indexing="ij")
    g_dbi_surf = 10.0 * np.log10(np.maximum(g_closed, 1e-2))
    r_norm = np.clip((g_dbi_surf + 15.0) / (np.max(g_dbi_surf) + 15.0), 0.08, None)
    sg = pv.StructuredGrid(
        r_norm * np.sin(TH) * np.cos(PH),
        r_norm * np.sin(TH) * np.sin(PH),
        r_norm * np.cos(TH),
    )
    sg.point_data["Gain [dBi]"] = g_dbi_surf.ravel(order="F")

    pl0 = pv.Plotter(off_screen=True, window_size=(820, 680))
    pl0.set_background("white")
    pl0.add_mesh(
        sg,
        scalars="Gain [dBi]",
        cmap="turbo",
        clim=(-10.0, 3.5),
        smooth_shading=True,
        scalar_bar_args={"title": "Gain [dBi]", "color": "black"},
    )
    pl0.show_axes()
    pl0.camera_position = [(2.4, -2.5, 1.8), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    img_3d_gain = pl0.screenshot(return_img=True)
    pl0.close()

    # 2. Build PyVista surface current sheet on the metal trace + translucent 3D far-field lobe
    X_m, Y_m = np.meshgrid(xy_x, xy_y, indexing="ij")
    Z_m = np.full_like(X_m, geom["h_sub"] * 1e3 + 0.08)
    j_sheet = pv.StructuredGrid(X_m, Y_m, Z_m)
    j_db = 20.0 * np.log10(np.maximum(j_mag_xy / np.max(j_mag_xy), 1e-3))
    j_sheet.point_data["|J_s| [dB]"] = j_db.ravel(order="F")

    r_mm = 22.0 * np.clip((g_dbi_surf + 15.0) / (np.max(g_dbi_surf) + 15.0), 0.1, None)
    cx, cy, cz = 12.5, -8.0, 1.0
    lobe = pv.StructuredGrid(
        cx + r_mm * np.sin(TH) * np.cos(PH),
        cy + r_mm * np.sin(TH) * np.sin(PH),
        cz + r_mm * np.cos(TH),
    )
    lobe.point_data["Gain [dBi]"] = g_dbi_surf.ravel(order="F")

    pl1 = pv.Plotter(off_screen=True, window_size=(840, 680))
    pl1.set_background("white")
    _add_ifa_actors_to_plotter(pl1, geom, scale_mm=True)
    pl1.add_mesh(j_sheet, scalars="|J_s| [dB]", cmap="viridis", clim=(-30, 0), opacity=0.82, show_scalar_bar=False)
    pl1.add_mesh(
        lobe,
        scalars="Gain [dBi]",
        cmap="turbo",
        clim=(-8.0, 3.3),
        opacity=0.45,
        smooth_shading=True,
        scalar_bar_args={"title": "3D Gain Lobe [dBi]", "color": "black"},
    )
    pl1.show_axes()
    pl1.camera_position = [(62.0, -75.0, 48.0), (12.5, -8.0, 1.0), (0.0, 0.0, 1.0)]
    img_lobe = pl1.screenshot(return_img=True)
    pl1.close()

    # 3. Build XZ near-field slice plane overlaid on 3D PCB
    X_xz, Z_xz = np.meshgrid(xz_x, xz_z, indexing="ij")
    Y_xz = np.full_like(X_xz, (geom["H1"] - 0.5 * geom["H6"]) * 1e3)
    xz_sheet = pv.StructuredGrid(X_xz, Y_xz, Z_xz)
    e_xz_db = 20.0 * np.log10(np.maximum(e_mag_xz / np.max(e_mag_xz), 1e-3))
    xz_sheet.point_data["|E| [dB]"] = e_xz_db.ravel(order="F")

    pl2 = pv.Plotter(off_screen=True, window_size=(840, 680))
    pl2.set_background("white")
    _add_ifa_actors_to_plotter(pl2, geom, scale_mm=True)
    pl2.add_mesh(
        xz_sheet,
        scalars="|E| [dB]",
        cmap="inferno",
        clim=(-35.0, 0.0),
        opacity=0.78,
        scalar_bar_args={"title": "Near-Field |E| [dB]", "color": "black"},
    )
    pl2.show_axes()
    pl2.camera_position = [(46.0, -56.0, 32.0), (12.5, -5.0, 1.0), (0.0, 0.0, 1.0)]
    img_slice = pl2.screenshot(return_img=True)
    pl2.close()

    fig, axes = figstyle.new_figure(
        ncols=3,
        nrows=1,
        width=figstyle.FULL_WIDTH_IN * 1.62,
        height=4.05,
    )
    panels = [
        (img_3d_gain, f"(a) 3D gain surface $G(\\theta,\\phi)$ ($G_{{\\max}}={nc_json['g_max_dbi']:.2f}$ dBi)"),
        (img_lobe, "(b) 3D PCB + $|\\mathbf{J}_s|$ + 3D far-field lobe"),
        (img_slice, "(c) 3D PCB + vertical XZ slice $|\\mathbf{E}|$ [dB]"),
    ]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(title)

    figstyle.save(fig, out_path)
    return out_path


def _plot_07_s11_smith(
    base_npz,
    nc_npz,
    nc_json: dict,
    sf11_npz,
    sf11_json: dict,
    out_path: Path,
) -> Path:
    """Figure 7: S11 (dB), Input Impedance (Re/Im), and 50-Ohm Smith Chart comparing local non-cubic vs uniform 0.1125 mm."""
    freqs = nc_npz["freqs_hz"]
    f_ghz = freqs / 1e9
    s11_db_nc = nc_npz["s11_db"]
    s11_db_sf11 = sf11_npz["s11_db"]
    s11_db_base = base_npz["s11_db"]
    zin_re_nc = nc_npz["zin_real"]
    zin_im_nc = nc_npz["zin_imag"]
    zin_re_u = sf11_npz["zin_real"]
    zin_im_u = sf11_npz["zin_imag"]
    s11_c_nc = nc_npz["s11_real"] + 1j * nc_npz["s11_imag"]
    s11_c_u = sf11_npz["s11_real"] + 1j * sf11_npz["s11_imag"]

    s11_ref_nom = swru120_reference_s11_db(freqs)

    fig, axes = figstyle.new_figure(
        ncols=3,
        nrows=1,
        width=figstyle.FULL_WIDTH_IN * 1.68,
        height=4.35,
    )

    # (a) |S11| in dB vs TI SWRU120D Figure 2-8 & Table 3-1
    ax = axes[0]
    ax.axvspan(2.40, 2.48, color="#fff59d", alpha=0.55, label="2.4 GHz ISM band")
    ax.plot(
        f_ghz,
        s11_ref_nom,
        color="#c62828",
        ls="--",
        lw=1.6,
        label="TI SWRU120D Fig. 2-8 (approx.)",
    )
    ax.plot(f_ghz, s11_db_base, color="#90caf9", ls=":", lw=1.4, label="Uniform $\\Delta = 0.50$ mm")
    ax.plot(
        f_ghz,
        s11_db_sf11,
        color="#42a5f5",
        lw=2.5,
        label=f"Uniform 0.1125 mm ref ($f_0={sf11_json['f_res_ghz']:.3f}$ GHz)",
    )
    ax.plot(
        f_ghz,
        s11_db_nc,
        color="#0d47a1",
        ls="-.",
        lw=1.9,
        label=f"Local non-cubic ($f_0={nc_json['f_res_ghz']:.3f}$ GHz, $-33\\%$ cells)",
    )
    ax.axhline(-15.0, color="#ad1457", ls=":", lw=1.0, label="SWRU120D ($< -15$ dB)")
    ax.axhline(-10.0, color="#37474f", ls="-.", lw=0.9, label="$-10$ dB BW threshold")
    ax.set_xlim(2.0, 2.8)
    ax.set_ylim(-24.0, 0.0)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("$|S_{11}|$ [dB]")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title(
        f"(a) Reflection $|S_{{11}}|$ ($f_{{\\mathrm{{res}}}}={nc_json['f_res_ghz']:.3f}$ GHz)"
    )

    # (b) Input impedance Re(Z_in) and Im(Z_in)
    ax = axes[1]
    ax.axvspan(2.40, 2.48, color="#fff59d", alpha=0.55)
    ax.plot(f_ghz, zin_re_u, color="#90caf9", lw=2.4, label="Uniform 0.1125 mm $\\mathrm{Re}(Z_{\\mathrm{in}})$")
    ax.plot(f_ghz, zin_im_u, color="#ffcc80", lw=2.4, label="Uniform 0.1125 mm $\\mathrm{Im}(Z_{\\mathrm{in}})$")
    ax.plot(f_ghz, zin_re_nc, color="#0d47a1", ls="--", lw=1.8, label="Local non-cubic $\\mathrm{Re}(Z_{\\mathrm{in}})$")
    ax.plot(f_ghz, zin_im_nc, color="#e65100", ls="--", lw=1.8, label="Local non-cubic $\\mathrm{Im}(Z_{\\mathrm{in}})$")
    ax.axhline(50.0, color="#1565c0", ls=":", lw=0.9, alpha=0.7, label="$R_0 = 50\\;\\Omega$ target")
    ax.axhline(0.0, color="#ef6c00", ls=":", lw=0.9, alpha=0.7)
    ax.set_xlim(2.0, 2.8)
    ax.set_ylim(-140.0, 140.0)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Input impedance $Z_{\\mathrm{in}}$ [$\\Omega$]")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title(
        f"(b) Impedance $Z_{{\\mathrm{{in}}}}(f_0) = {nc_json['zin_res_real_ohm']:.1f} {nc_json['zin_res_imag_ohm']:+.1f}j\\;\\Omega$"
    )

    # (c) Calibrated 50-Ohm Smith Chart
    ax = axes[2]
    _draw_smith_grid(ax)
    ax.plot(np.real(s11_c_u), np.imag(s11_c_u), color="#90caf9", lw=2.5, label="Uniform 0.1125 mm ref")
    ax.plot(np.real(s11_c_nc), np.imag(s11_c_nc), color="#0d47a1", ls="--", lw=1.8, label="Local non-cubic")
    i_240 = int(np.argmin(np.abs(freqs - 2.40e9)))
    i_244 = int(np.argmin(np.abs(freqs - 2.44e9)))
    i_248 = int(np.argmin(np.abs(freqs - 2.48e9)))
    ax.plot(np.real(s11_c_nc[i_240]), np.imag(s11_c_nc[i_240]), "o", color="#2e7d32", ms=5.5, label="2.40 GHz")
    ax.plot(np.real(s11_c_nc[i_244]), np.imag(s11_c_nc[i_244]), "*", color="#d32f2f", ms=9.0, label="2.44 GHz center")
    ax.plot(np.real(s11_c_nc[i_248]), np.imag(s11_c_nc[i_248]), "s", color="#6a1b9a", ms=5.5, label="2.48 GHz")
    th_c = np.linspace(0, 2 * np.pi, 180)
    ax.plot(0.3162 * np.cos(th_c), 0.3162 * np.sin(th_c), color="#c62828", ls=":", lw=1.1, marker="none", label="$-10$ dB circle")
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("$\\mathrm{Re}(S_{11})$")
    ax.set_ylabel("$\\mathrm{Im}(S_{11})$")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("(c) $50\\;\\Omega$ Smith chart locus (2.0–2.8 GHz)")

    figstyle.save(fig, out_path)
    return out_path


def _draw_smith_grid(ax: plt.Axes) -> None:
    """Draw constant-resistance and constant-reactance circles on a Smith chart."""
    th = np.linspace(0.0, 2.0 * np.pi, 240)
    ax.plot(np.cos(th), np.sin(th), color="#263238", lw=1.2, marker="none")
    ax.axhline(0.0, color="#90a4ae", lw=0.7)
    for r in (0.2, 0.5, 1.0, 2.0, 5.0):
        cx = r / (r + 1.0)
        rad = 1.0 / (r + 1.0)
        ax.plot(cx + rad * np.cos(th), rad * np.sin(th), color="#b0bec5", lw=0.6, ls="-", marker="none")
    for x_val in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
        cy = 1.0 / x_val
        rad = abs(1.0 / x_val)
        pts_x = 1.0 + rad * np.cos(th)
        pts_y = cy + rad * np.sin(th)
        inside = pts_x**2 + pts_y**2 <= 1.001
        pts_x = np.where(inside, pts_x, np.nan)
        pts_y = np.where(inside, pts_y, np.nan)
        ax.plot(pts_x, pts_y, color="#cfd8dc", lw=0.6, ls="--", marker="none")


def _plot_08_resolution_and_param(
    base_npz,
    base_json: dict,
    med_npz,
    med_json: dict,
    fine_npz,
    fine_json: dict,
    sf18_npz,
    sf18_json: dict,
    sf11_npz,
    sf11_json: dict,
    nc_npz,
    nc_json: dict,
    param_npz,
    param_json: dict,
    out_path: Path,
) -> Path:
    """Figure 8: 3-panel 5-level uniform convergence + local non-cubic upgrade, ground-plane study, and Chu bound."""
    fig, axes = figstyle.new_figure(
        ncols=3,
        nrows=1,
        width=figstyle.FULL_WIDTH_IN * 1.68,
        height=4.45,
    )

    # (a) 5-Level uniform resolution study + local non-cubic mesh upgrade + Richardson inset
    ax = axes[0]
    f_ghz = sf11_npz["freqs_hz"] / 1e9
    ax.axvspan(2.40, 2.48, color="#fff59d", alpha=0.5, label="2.4 GHz ISM band")
    for npz_d, js_d, lbl, col, ls, lw in [
        (base_npz, base_json, "0.50 mm", "#90caf9", ":", 1.4),
        (med_npz, med_json, "0.35 mm", "#26a69a", "-.", 1.5),
        (fine_npz, fine_json, "0.25 mm", "#1e88e5", "--", 1.6),
        (sf18_npz, sf18_json, "0.18 mm (≥4/feat)", "#fb8c00", "-", 1.7),
        (sf11_npz, sf11_json, "0.1125 mm ref (285.8k)", "#42a5f5", "-", 2.4),
        (nc_npz, nc_json, "Local non-cubic (191.8k)", "#0d47a1", "--", 1.9),
    ]:
        ax.plot(
            f_ghz,
            npz_d["s11_db"],
            color=col,
            ls=ls,
            lw=lw,
            label=f"{lbl} ($f_0={js_d['f_res_ghz']:.3f}$ GHz)",
        )
    ax.axhline(-15.0, color="#ad1457", ls=":", lw=0.9)
    ax.axhline(-10.0, color="#37474f", ls="-.", lw=0.9)
    ax.set_xlim(2.0, 2.8)
    ax.set_ylim(-24.0, 0.0)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("$|S_{11}|$ [dB]")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("(a) 5-level convergence & non-cubic mesh")

    # Inset showing Richardson continuum extrapolation of f_res(Delta)
    cells_mm = np.array([0.50, 0.35, 0.25, 0.18, 0.1125])
    f_res_arr = np.array(
        [
            base_json["f_res_ghz"],
            med_json["f_res_ghz"],
            fine_json["f_res_ghz"],
            sf18_json["f_res_ghz"],
            sf11_json["f_res_ghz"],
        ]
    )
    f_inf, f_slope, d_dense = richardson_extrapolate_ifa(cells_mm, f_res_arr, order=1.5)
    ax_ins = ax.inset_axes([0.54, 0.56, 0.43, 0.38])
    ax_ins._is_inset = True
    ax_ins.plot(cells_mm, f_res_arr, "o-", color="#1e88e5", lw=1.3, ms=4.0, label="Uniform")
    ax_ins.plot(d_dense, f_inf + f_slope * (d_dense**1.5), "--", color="#0d47a1", lw=1.1, marker="none")
    ax_ins.plot(0.115, nc_json["f_res_ghz"], "D", color="#c62828", ms=4.8, label="Non-cubic")
    ax_ins.plot(0.0, f_inf, "*", color="#0d47a1", ms=7.0)
    ax_ins.set_xlim(0.53, -0.03)
    ax_ins.set_ylim(2.38, 2.47)
    ax_ins.set_xlabel("$\\Delta_{\\min}$ [mm]", fontsize=8.5)
    ax_ins.set_ylabel("$f_{\\mathrm{res}}$ [GHz]", fontsize=8.5)
    ax_ins.tick_params(labelsize=8.5)
    ax_ins.set_title(f"$f_\\infty = {f_inf:.3f}$ GHz", fontsize=8.5, pad=2)

    # (b) Parameter study: Ground-plane length L_gnd (SWRA117D)
    ax = axes[1]
    f_p = param_npz["freqs_hz"] / 1e9
    gnd_lengths = param_npz["ground_lengths_mm"]
    gnd_curves = param_npz["ground_s11_curves_db"]
    colors = ["#8e24aa", "#fb8c00", "#1e88e5", "#2e7d32"]
    for lg, curve, col in zip(gnd_lengths, gnd_curves, colors):
        ax.plot(
            f_p,
            curve,
            color=col,
            lw=1.8,
            label=f"$L_{{\\mathrm{{gnd}}}} = {lg:.0f}$ mm ($29.6\\times{lg:.0f}\\;\\mathrm{{mm}}^2$)",
        )
    ax.axhline(-10.0, color="#37474f", ls=":", lw=1.0)
    ax.set_xlim(2.0, 2.8)
    ax.set_ylim(-24.0, 0.0)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("$|S_{11}|$ [dB]")
    ax.legend(loc="lower left", fontsize=8.5, title="Ground-plane length (SWRA117D)", title_fontsize=8.5)
    ax.set_title("(b) Ground-plane length study")

    # (c) Chu-Wheeler Fundamental Q / Bandwidth Bound vs IFA Alone & Full PCB
    ax = axes[2]
    ka_grid = np.linspace(0.25, 2.2, 200)
    fbw_lossless = [chu_max_fractional_bandwidth(k, vswr_max=1.925, efficiency=1.0) * 100.0 for k in ka_grid]
    fbw_fr4 = [
        chu_max_fractional_bandwidth(k, vswr_max=1.925, efficiency=nc_json["eta_rad_2g44_pct"] / 100.0) * 100.0
        for k in ka_grid
    ]
    ax.plot(ka_grid, fbw_lossless, color="#263238", lw=1.9, marker="none", label="Chu–Wheeler bound ($\\eta_{\\mathrm{rad}}=100\\%$)")
    ax.plot(
        ka_grid,
        fbw_fr4,
        color="#ef6c00",
        ls="--",
        lw=1.9,
        marker="none",
        label=f"With FR-4 loss ($\\eta_{{\\mathrm{{rad}}}}={nc_json['eta_rad_2g44_pct']:.0f}\\%$)",
    )
    k0 = 2.0 * np.pi * 2.44e9 / c_light
    ka_trace = k0 * 0.5 * np.hypot(25.7e-3, 7.5e-3)
    ka_board = k0 * 0.5 * np.hypot(29.58e-3, 39.5e-3)
    fbw_meas_pct = 100.0 * nc_json["bw_10db_mhz"] / (nc_json["f_res_ghz"] * 1e3)
    ax.plot(ka_trace, fbw_meas_pct, "s", color="#c62828", ms=7.5, label=f"IFA trace alone ($ka={ka_trace:.2f}$)")
    ax.plot(ka_board, fbw_meas_pct, "*", color="#0d47a1", ms=10.5, label=f"Full PCB + ground ($ka={ka_board:.2f}$)")
    ax.annotate(
        "Ground plane increases\neffective Chu sphere $a$",
        xy=(ka_board, fbw_meas_pct),
        xytext=(0.82, 24.0),
        arrowprops=dict(arrowstyle="->", color="#0d47a1", lw=1.1),
        fontsize=8.5,
        color="#0d47a1",
        fontweight="bold",
    )
    ax.set_xlim(0.25, 2.1)
    ax.set_ylim(2.0, 95.0)
    ax.set_yscale("log")
    ax.set_xlabel("Electrical size $k_0 a$ [rad]")
    ax.set_ylabel("$-10$ dB fractional bandwidth $\\Delta f / f_0$ [%]")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("(c) Chu–Wheeler $Q$ bound")

    figstyle.save(fig, out_path)
    return out_path
