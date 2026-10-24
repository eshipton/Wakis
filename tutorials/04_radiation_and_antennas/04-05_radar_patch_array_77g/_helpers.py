"""Helper routines for Tutorial 04-03: 77 GHz Series-Fed Patch Array for Automotive Radar.

Reproduces the 77 GHz series-fed microstrip patch array on Rogers RT/duroid 5880
(h = 254 um, eps_r = 2.20, tan(delta) = 0.0009) from:
  S. Yang, L. Zhang, J. Fu, Z. Zheng, X. Zhang, and A. Liao,
  "Design and Optimization for 77 GHz Series-Fed Patch Array Antenna Based on
  Genetic Algorithm", Sensors 20(11), 3066 (2020), doi:10.3390/s20113066.

Implements:
- Microstrip cavity / transmission-line model (resonant length L1, Hammerstad-Jensen
  fringing extension dL, effective permittivity eps_eff, two-slot element factor);
- Series-feed phase progression, frequency beam squint, and array-factor synthesis
  for uniform ("1111111111"), GA-optimized ("1011111111" & width-tapered), and
  Dolph-Chebyshev / Taylor excitations;
- Analytical 77 GHz copper skin-depth, Hammerstad-Bekkadal surface-roughness, and
  dielectric attenuation model bridging the ~17.0 dBi PEC directivity to the
  15.9 dBi (uniform) / 15.6 dBi (tapered) measured gains;
- 3D FIT full-wave simulations with wakis.estimate preflight, project grid-rule
  verification, and 8 multi-panel figures + 1 animated GIF (< 5 MB).
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
from scipy.signal import argrelmax

import wakis.visualize as wv
from wakis import GridFIT3D, SolverFIT3D
from wakis.conformal import rectangle_sheet
from wakis.estimate import estimate
from wakis.field_monitors import DFTMonitor
from wakis.lumped import LumpedElement, LumpedPort
from wakis.near2far import Near2FarMonitor, half_power_beamwidth
from wakis.sources import GaussianSource
from wakis.templates import Box, meshing
from wakis.templates.shapes import assign_materials

Z0_FREE = float(np.sqrt(mu_0 / epsilon_0))
SIGMA_COPPER = 5.8e7  # Bulk conductivity of annealed copper [S/m]
TUTORIAL_DIR = Path(__file__).resolve().parent
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

# ---------------------------------------------------------------------------
# Verified geometry and benchmarks from Yang et al., Sensors 20(11), 3066 (2020)
# DOI: https://doi.org/10.3390/s20113066
# ---------------------------------------------------------------------------
YANG2020_GEOMETRY_UM = {
    "f0_ghz": 77.0,
    "eps_r": 2.20,
    "tan_delta": 0.0009,  # Standard Rogers RT/duroid 5880 datasheet value (not stated in Yang et al.)
    "h_sub_um": 254.0,
    "L1_patch_um": 1260.0,
    "L1_patch_length_um": 1260.0,
    "W1_patch_um": 1540.0,
    "W1_patch_width_um": 1540.0,
    "L2_line_um": 1260.0,
    "L2_line_length_um": 1260.0,
    "W2_line_um": 320.0,
    "W2_line_width_um": 320.0,
    "L3_qwt_um": 718.0,  # Quarter-wave input transformer length lambda_g(W3)/4 at 77 GHz
    "L3_qwt_length_um": 718.0,
    "W3_qwt_um": 429.0,  # Quarter-wave input transformer width (Z_T = sqrt(50 * 71.8) = 59.9 Ohm)
    "W3_qwt_width_um": 429.0,
    "termination": "open_patch_edge",  # Fig. 5 & Fig. 6a: 10th patch ends in an open-circuit slot
    "L_term_um": 0.0,
    "termination_stub_length_um": 0.0,
    "unit_cell_pitch_um": 2520.0,
    "n_elements": 10,
    "max_total_length_mm": 30.0,
    "uniform_code": "1111111111",
    "tapered_ga_code": "1011111111",
    # In Yang et al. (Sec. 3.2, pp. 5-6; Fig. 5, p. 6; Fig. 6a, p. 7), each bit of the
    # 10-bit GA chromosome '1011111111' is a binary on/off switch for uniform W1=1540 um
    # patches ('1' = patch present, '0' = bare W2=320 um line at element #2).
    # Both 'pure_binary_widths_um' and 'tapered_widths_um' represent the paper's binary
    # '1011111111' design (uniform W1=1540 um on all 9 active patches, element #2 omitted).
    # Our optional raised-cosine width-taper design variation (not in Yang et al.) is kept
    # separately under 'width_taper_variation_widths_um'.
    "uniform_widths_um": [1540.0] * 10,
    "pure_binary_widths_um": [
        1540.0,
        320.0,
        1540.0,
        1540.0,
        1540.0,
        1540.0,
        1540.0,
        1540.0,
        1540.0,
        1540.0,
    ],
    "tapered_widths_um": [
        1540.0,
        320.0,
        1540.0,
        1540.0,
        1540.0,
        1540.0,
        1540.0,
        1540.0,
        1540.0,
        1540.0,
    ],
    "width_taper_variation_widths_um": [
        770.0,
        320.0,
        1155.0,
        1540.0,
        1540.0,
        1540.0,
        1540.0,
        1155.0,
        770.0,
        520.0,
    ],
}

YANG2020_BENCHMARKS = {
    "citation": "S. Yang et al., Sensors 20(11), 3066 (2020), doi:10.3390/s20113066",
    "substrate": "Rogers 5880 (h = 254 um, eps_r = 2.2, Sec. 3.2, p. 5; tan_delta = 0.0009 from Rogers datasheet)",
    "f0_hz": 77.0e9,
    "design_goals": {
        "first_null_beamwidth_deg": 8.0,  # within +-8 deg (Sec. 3.2, p. 6)
        "sll_max_db": -14.0,  # <= -14 dB (Sec. 3.2, p. 6)
    },
    "uniform_1111111111": {
        "gain_sim_dbi": 17.0,
        "gain_meas_dbi": 15.9,
        "sll_e_sim_db": -12.5,
        "sll_e_meas_db": -10.0,
        "hpbw_e_half_sim_deg": 3.7,
        "hpbw_e_half_meas_deg": 3.7,
        "hpbw_e_half_deg": 3.7,
        "hpbw_e_full_deg": 7.4,
        "hpbw_h_half_sim_deg": 33.0,
        "hpbw_h_half_meas_deg": 30.0,
    },
    "tapered_1011111111": {
        "gain_sim_dbi": 17.0,
        "gain_meas_dbi": 15.6,
        "sll_e_sim_db": -16.5,
        "sll_e_meas_db": -14.8,
        "hpbw_e_half_sim_deg": 3.7,
        "hpbw_e_half_meas_deg": 3.8,
        "hpbw_e_full_sim_deg": 7.4,
        "hpbw_e_full_meas_deg": 7.6,
        "hpbw_h_half_sim_deg": 36.0,
        "hpbw_h_half_meas_deg": 30.0,
    },
    "s11_in_source": None,
}

YANG2020_SOURCE_LOCATIONS = {
    "Operating frequency (77 GHz)": "Title & Abstract, p. 1; Section 3.2, p. 5",
    "Substrate (Rogers 5880, eps_r = 2.2, h = 254 um)": "Section 3.2, p. 5 (tan_delta and t_Cu not reported in paper)",
    "Patch & feeder dimensions (L1=1260 um, W1=1540 um, L2=1260 um, W2=320 um)": "Section 3.2, p. 6; Figure 4, p. 6",
    "Inter-element spacing (d = L1 + L2 = 2520 um) & 10-bit gene (<= 3 cm)": "Section 3.2, p. 6; Figure 4, p. 6",
    "Input quarter-wave transformer (L3=718 um, W3=429 um) & open-patch termination": "Figure 5, p. 6 & Figure 6a, p. 7 (10th patch ends in open circuit; lambda_g/4 input transformer between 50 Ohm and Z0(W2)=71.8 Ohm)",
    "Design goals (first-null beamwidth within +-8 deg, SLL <= -14 dB)": "Section 3.2, p. 6",
    "GA chromosome '1011111111' meaning (binary on/off of uniform patches)": "Section 3.2, pp. 5-6; Figure 5, p. 6; Figure 6a, p. 7; Section 4, p. 7",
    "Optimized '1011111111' Gain (17 dBi sim, 15.6 dBi meas)": "Abstract, p. 1; Section 4, p. 7; Section 5, p. 9",
    "Optimized '1011111111' E-plane SLL (-16.5 dB sim, -14.8 dB meas)": "Abstract, p. 1; Section 4, p. 7; Figure 8a, p. 8; Section 5, p. 9",
    "Optimized '1011111111' E-plane HPBW (+-3.7 deg sim, +-3.8 deg meas)": "Abstract, p. 1; Section 4, p. 7; Figure 8a, p. 8; Section 5, p. 9",
    "Optimized '1011111111' H-plane HPBW (+-36 deg sim, +-30 deg meas)": "Abstract, p. 1; Section 4, p. 7; Figure 8b, p. 8; Section 5, p. 9",
    "Traditional '1111111111' Gain (17 dBi sim, 15.9 dBi meas)": "Section 4, p. 8",
    "Traditional '1111111111' E-plane SLL (-12.5 dB sim, -10 dB meas)": "Section 4, p. 8; Figure 9a, p. 9",
    "Traditional '1111111111' E-plane HPBW (+-3.7 deg sim, +-3.7 deg meas)": "Section 4, p. 8; Figure 9a, p. 9",
    "Traditional '1111111111' H-plane HPBW (+-33 deg sim, +-30 deg meas)": "Section 4, p. 8; Figure 9b, p. 9",
    "Return loss |S11| / impedance bandwidth": "Not reported in Yang et al. (2020)",
}


# ---------------------------------------------------------------------------
# Analytical Microstrip Cavity, Array-Factor & 77 GHz Loss Models
# ---------------------------------------------------------------------------
def microstrip_eps_eff_and_z0(
    width_m: float,
    height_m: float = 254e-6,
    eps_r: float = 2.20,
) -> tuple[float, float]:
    """Hammerstad-Jensen effective permittivity and characteristic impedance.

    Parameters
    ----------
    width_m : float
        Microstrip patch or line width :math:`W` [m].
    height_m : float, optional
        Substrate thickness :math:`h` [m]. Default 254 um.
    eps_r : float, optional
        Relative permittivity :math:`\\varepsilon_r` [-]. Default 2.20.

    Returns
    -------
    eps_eff : float
        Effective relative permittivity :math:`\\varepsilon_{\\text{eff}}` [-].
    z0_ohm : float
        Quasi-TEM characteristic impedance :math:`Z_0` [Ohm].
    """
    if width_m <= 0 or height_m <= 0 or eps_r < 1.0:
        raise ValueError("Require width_m > 0, height_m > 0, and eps_r >= 1.")
    u = width_m / height_m
    eps_eff = 0.5 * (eps_r + 1.0) + 0.5 * (eps_r - 1.0) / np.sqrt(1.0 + 12.0 / u)
    if u <= 1.0:
        z0 = (60.0 / np.sqrt(eps_eff)) * np.log(8.0 / u + 0.25 * u)
    else:
        z0 = (120.0 * np.pi / np.sqrt(eps_eff)) / (
            u + 1.393 + 0.667 * np.log(u + 1.444)
        )
    return float(eps_eff), float(z0)


def patch_cavity_resonance_hz(
    length_m: float = 1260e-6,
    width_m: float = 1540e-6,
    height_m: float = 254e-6,
    eps_r: float = 2.20,
) -> dict[str, float]:
    """Evaluate the transmission-line / cavity model for a rectangular patch.

    Parameters
    ----------
    length_m : float, optional
        Physical patch length :math:`L_1` [m]. Default 1260 um.
    width_m : float, optional
        Physical patch width :math:`W_1` [m]. Default 1540 um.
    height_m : float, optional
        Substrate thickness :math:`h` [m]. Default 254 um.
    eps_r : float, optional
        Substrate permittivity [-]. Default 2.20.

    Returns
    -------
    dict
        Cavity-model parameters: ``eps_eff``, ``delta_l_um``, ``l_eff_um``,
        ``f_res_ghz``, and guided wavelength ``lambda_g_um`` at 77 GHz.
    """
    if length_m <= 0 or width_m <= 0 or height_m <= 0:
        raise ValueError("Patch dimensions must be strictly positive.")
    eps_eff, z0_patch = microstrip_eps_eff_and_z0(width_m, height_m, eps_r)
    u = width_m / height_m
    delta_l = (
        0.412
        * height_m
        * ((eps_eff + 0.3) * (u + 0.264))
        / ((eps_eff - 0.258) * (u + 0.8))
    )
    l_eff = length_m + 2.0 * delta_l
    f_res = c_light / (2.0 * np.sqrt(eps_eff) * l_eff)
    lam_g_77 = c_light / (77.0e9 * np.sqrt(eps_eff))
    eps_eff_l, z0_line = microstrip_eps_eff_and_z0(320e-6, height_m, eps_r)
    k0_77 = 2.0 * np.pi * 77.0e9 / c_light
    phase_deg = np.rad2deg(
        k0_77 * (np.sqrt(eps_eff) * l_eff + np.sqrt(eps_eff_l) * (1260e-6 + delta_l))
    )
    return {
        "eps_eff_patch": float(eps_eff),
        "z0_patch_ohm": float(z0_patch),
        "eps_eff_line": float(eps_eff_l),
        "z0_line_ohm": float(z0_line),
        "delta_l_um": float(delta_l * 1e6),
        "delta_L_um": float(delta_l * 1e6),
        "l_eff_um": float(l_eff * 1e6),
        "f_res_hz": float(f_res),
        "f_res_ghz": float(f_res / 1e9),
        "lambda_g_77_um": float(lam_g_77 * 1e6),
        "unit_cell_pitch_um": float((length_m + 1260e-6) * 1e6),
        "unit_cell_phase_77g_deg": float(phase_deg),
    }


def copper_and_dielectric_loss_77ghz(
    freq_hz: float = 77.0e9,
    height_m: float = 254e-6,
    eps_r: float = 2.20,
    tan_delta: float = 0.0009,
    sigma_cu: float = SIGMA_COPPER,
    roughness_rms_um: float = 0.45,
    array_length_m: float = 25.2e-3,
) -> dict[str, float]:
    """Analytical conductor + dielectric + feed-transition loss budget at 77 GHz.

    At 77 GHz the skin depth in copper is :math:`\\delta_s = \\sqrt{2/(\\omega\\mu_0\\sigma)}
    \\approx 0.238\\;\\mu\\text{m}`, which is over :math:`1000\\times` thinner than the
    :math:`254\\;\\mu\\text{m}` substrate and therefore modelled as PEC in the 3D FIT
    volume mesh. This routine computes the analytical Pucel-Hammerstad microstrip
    conductor attenuation (including Hammerstad-Bekkadal RMS copper surface
    roughness :math:`R_q`) and dielectric loss tangent attenuation along the
    10-element series array to relate the lossless 3D FIT directivity
    (:math:`\\approx 17.0\\;\\text{dBi}`) to the measured gain (:math:`15.6\\text{--}15.9\\;\\text{dBi}`).

    Parameters
    ----------
    freq_hz : float, optional
        Operating frequency [Hz]. Default 77.0 GHz.
    height_m : float, optional
        Substrate thickness [m]. Default 254 um.
    eps_r : float, optional
        Relative permittivity [-]. Default 2.20.
    tan_delta : float, optional
        Dielectric loss tangent [-]. Default 0.0009.
    sigma_cu : float, optional
        Copper conductivity [S/m]. Default 5.8e7 S/m.
    roughness_rms_um : float, optional
        Copper foil RMS surface roughness [um]. Default 0.45 um (rolled copper).
    array_length_m : float, optional
        Total physical length of the 10-element series array [m]. Default 25.2 mm.

    Returns
    -------
    dict
        Skin depth, surface resistance, attenuation constants, and total gain loss [dB].
    """
    if freq_hz <= 0 or sigma_cu <= 0 or height_m <= 0:
        raise ValueError("Require freq_hz > 0, sigma_cu > 0, and height_m > 0.")
    omega = 2.0 * np.pi * freq_hz
    skin_depth = float(np.sqrt(2.0 / (omega * mu_0 * sigma_cu)))
    r_s_smooth = float(1.0 / (sigma_cu * skin_depth))
    # Hammerstad-Bekkadal surface-roughness factor at mmWave
    rq_m = max(0.0, float(roughness_rms_um) * 1e-6)
    k_rough = float(1.0 + (2.0 / np.pi) * np.arctan(1.4 * (rq_m / skin_depth) ** 2))
    r_s_eff = r_s_smooth * k_rough

    # Weighted average of patch (W1 = 1540 um) and series line (W2 = 320 um)
    eps_eff_p, z0_p = microstrip_eps_eff_and_z0(1540e-6, height_m, eps_r)
    eps_eff_l, z0_l = microstrip_eps_eff_and_z0(320e-6, height_m, eps_r)
    alpha_c_p = r_s_eff / (z0_p * 1540e-6)  # [Np/m]
    alpha_c_l = r_s_eff / (z0_l * 320e-6)  # [Np/m]
    alpha_c_np_m = 0.5 * (alpha_c_p + alpha_c_l)

    k0 = omega / c_light
    eps_eff_avg = 0.5 * (eps_eff_p + eps_eff_l)
    alpha_d_np_m = (
        k0
        * eps_r
        * (eps_eff_avg - 1.0)
        * tan_delta
        / (2.0 * np.sqrt(eps_eff_avg) * (eps_r - 1.0))
    )

    np_to_db = 8.685889638
    alpha_c_db_m = float(alpha_c_np_m * np_to_db)
    alpha_d_db_m = float(alpha_d_np_m * np_to_db)
    # Effective propagation length in a standing/travelling series array is ~0.55 L_total,
    # plus ~0.45 dB for the microstrip-to-WR12 waveguide transition reported by Yang et al.
    l_eff_prop = 0.55 * array_length_m
    loss_cu_db = float(alpha_c_db_m * l_eff_prop)
    loss_diel_db = float(alpha_d_db_m * l_eff_prop)
    loss_wr12_db = 0.45
    total_loss_db = loss_cu_db + loss_diel_db + loss_wr12_db
    eta_total = float(10.0 ** (-total_loss_db / 10.0))

    return {
        "skin_depth_um": float(skin_depth * 1e6),
        "r_s_smooth_ohm": r_s_smooth,
        "rs_smooth_ohm_sq": r_s_smooth,
        "roughness_factor": k_rough,
        "r_s_eff_ohm": r_s_eff,
        "alpha_c_db_per_cm": float(alpha_c_db_m * 1e-2),
        "alpha_d_db_per_cm": float(alpha_d_db_m * 1e-2),
        "conductor_loss_db": loss_cu_db,
        "dielectric_loss_db": loss_diel_db,
        "wr12_transition_loss_db": loss_wr12_db,
        "total_loss_db": total_loss_db,
        "radiation_efficiency": float(eta_total),
        "estimated_efficiency_pct": float(eta_total * 100.0),
    }


def series_array_factor_e_plane(
    theta_deg: np.ndarray,
    weights: np.ndarray | None = None,
    pitch_m: float = 2520e-6,
    freq_hz: float = 77.0e9,
    patch_length_m: float = 1260e-6,
    progressive_phase_rad: float = 0.0,
    topology: str = "tapered",
    include_element_factor: bool = True,
) -> np.ndarray:
    """Analytical E-plane pattern (Array Factor x Two-Slot Patch Element Factor).

    Parameters
    ----------
    theta_deg : ndarray
        Elevation angle :math:`\\theta \\in [-90^\\circ, 90^\\circ]` in the XZ E-plane [deg].
    weights : array_like, optional
        Complex or real excitation weights :math:`a_n` of the :math:`N` elements [-].
        If ``None``, synthesized from ``topology`` ('uniform', 'tapered'/'binary_ga',
        or 'width_taper_variation').
    pitch_m : float, optional
        Inter-element period :math:`d = L_1 + L_2` [m]. Default 2520 um.
    freq_hz : float, optional
        Frequency [Hz]. Default 77.0 GHz.
    patch_length_m : float, optional
        Radiating slot separation :math:`L_1` [m]. Default 1260 um.
    progressive_phase_rad : float, optional
        Progressive inter-element phase residual :math:`\\Delta\\psi` [rad]. Default 0.0.
    topology : {'uniform', 'tapered', 'binary_ga', 'width_taper_variation'}, optional
        Default weight profile when ``weights`` is ``None``.
    include_element_factor : bool, optional
        Multiply the array factor by the two-slot patch element factor
        :math:`\\cos(k_0 L_1 \\sin\\theta / 2)`. Default ``True``.

    Returns
    -------
    pattern_norm_db : ndarray
        Normalized E-plane power pattern in dB (peak = 0 dB).
    """
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    if weights is None:
        n_idx_1d = np.arange(10, dtype=float)
        if topology == "uniform":
            # Series-fed uniform '1111111111' standing/traveling-wave amplitude profile
            w = np.ones(10, dtype=float)
        elif topology in ("tapered", "binary_ga", "binary"):
            # Paper's binary '1011111111': uniform W1=1540 um patches with element #2 omitted ('0')
            w = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
        elif topology == "width_taper_variation":
            w_um = np.asarray(
                YANG2020_GEOMETRY_UM["width_taper_variation_widths_um"], dtype=float
            )
            mask = np.where(w_um > 320.0, (w_um / 1540.0) ** 1.35, 0.0)
            w = mask * np.exp(-0.08 * n_idx_1d)
        else:
            raise ValueError(f"Unknown topology {topology!r}")
        w = w.astype(complex)
    else:
        w = np.asarray(weights, dtype=complex)
    if w.size == 0 or np.all(np.abs(w) == 0):
        raise ValueError("weights must contain at least one non-zero element.")
    k0 = 2.0 * np.pi * freq_hz / c_light
    n_idx = np.arange(w.size)[:, None]
    psi = k0 * pitch_m * np.sin(th)[None, :] + progressive_phase_rad
    af = np.sum(w[:, None] * np.exp(1j * n_idx * psi), axis=0)
    if include_element_factor:
        ef = np.cos(0.5 * k0 * patch_length_m * np.sin(th))
        pat = np.abs(af * ef) ** 2
    else:
        pat = np.abs(af) ** 2
    pat /= np.maximum(np.max(pat), 1e-30)
    return 10.0 * np.log10(np.maximum(pat, 1e-5))


def compute_sll_db(ang_deg: np.ndarray, pattern_db: np.ndarray) -> float:
    """Compute E-plane sidelobe level (SLL) [dB] relative to the main-beam peak.

    Identifies all local maxima (`argrelmax`) strictly outside the first nulls of
    the main beam (:math:`|\\theta - \\theta_{\\text{peak}}| > 8^\\circ`,
    :math:`|\\theta| < 75^\\circ`) and returns the highest sidelobe peak relative to
    the main-lobe peak.
    """
    ang_deg = np.asarray(ang_deg, dtype=float)
    pattern_db = np.asarray(pattern_db, dtype=float)
    i_pk = int(np.argmax(pattern_db))
    th_pk = ang_deg[i_pk]
    max_idx = argrelmax(pattern_db, order=3)[0]
    side_max = [
        idx
        for idx in max_idx
        if abs(ang_deg[idx] - th_pk) > 8.0 and abs(ang_deg[idx]) < 75.0
    ]
    if not side_max:
        side_mask = (np.abs(ang_deg - th_pk) > 11.5) & (np.abs(ang_deg) < 75.0)
        return float(np.max(pattern_db[side_mask]) - pattern_db[i_pk])
    return float(np.max(pattern_db[side_max]) - pattern_db[i_pk])


def extract_tm100_excitations(
    dft_sub: DFTMonitor,
    x_starts: list[float],
    L1: float = 1260e-6,
    W1: float = 1540e-6,
    code: str = "1011111111",
) -> dict:
    """Extract per-element TM100 complex excitations (amplitude & phase) from a 3D FIT DFTMonitor.

    Projects the 77.0 GHz substrate electric field :math:`E_z(x, y, h/2)` onto the
    fundamental :math:`\\text{TM}_{100}` cavity cosine mode :math:`\\cos(\\pi(x - x_{0,n})/L_1)`
    and the longitudinal surface current :math:`J_x = -H_y(x, y, h/2)` onto the sine
    mode :math:`\\sin(\\pi(x - x_{0,n})/L_1)` over each patch footprint
    :math:`[x_{0,n}, x_{0,n}+L_1] \\times [-W_1/2, +W_1/2]`. Also computes the resulting
    array-factor SLL with and without the two-slot element factor and with analytical
    conductor + dielectric line attenuation.
    """
    Ez_xy, c_ez = dft_sub.get_field("Ez")
    Hy_xy, _ = dft_sub.get_field("Hy")
    ez_2d = np.squeeze(Ez_xy[0])
    hy_2d = np.squeeze(Hy_xy[0])
    nx_m = min(ez_2d.shape[0], hy_2d.shape[0])
    ny_m = min(ez_2d.shape[1], hy_2d.shape[1])
    ez_2d = ez_2d[:nx_m, :ny_m]
    jx_2d = -hy_2d[:nx_m, :ny_m]
    xc = c_ez[0][:nx_m]
    yc = c_ez[1][:ny_m]

    y_mask = np.abs(yc) <= 0.5 * W1 + 1e-6
    v_mode = np.zeros(len(code), dtype=complex)
    i_mode = np.zeros(len(code), dtype=complex)

    xi_rel = np.linspace(0.05, 0.95, 31)
    cos_basis = np.cos(np.pi * xi_rel)
    sin_basis = np.sin(np.pi * xi_rel)
    ez_yavg = np.mean(ez_2d[:, y_mask], axis=1)
    jx_yavg = np.mean(jx_2d[:, y_mask], axis=1)

    for idx, (x0, bit) in enumerate(zip(x_starts, code)):
        if bit == "0":
            v_mode[idx] = 0.0
            i_mode[idx] = 0.0
            continue
        xq = x0 + xi_rel * L1
        ez_q = np.interp(xq, xc, np.real(ez_yavg)) + 1j * np.interp(
            xq, xc, np.imag(ez_yavg)
        )
        jx_q = np.interp(xq, xc, np.real(jx_yavg)) + 1j * np.interp(
            xq, xc, np.imag(jx_yavg)
        )
        v_mode[idx] = 2.0 * np.mean(ez_q * cos_basis)
        i_mode[idx] = 2.0 * np.mean(jx_q * sin_basis)

    ang_deg = np.linspace(-90.0, 90.0, 721)
    k0 = 2.0 * np.pi * 77.0e9 / c_light
    th_rad = np.deg2rad(ang_deg)
    x_centers = np.array([x0 + 0.5 * L1 for x0 in x_starts], dtype=float)
    ef = np.cos(0.5 * k0 * L1 * np.sin(th_rad))
    loss_info = copper_and_dielectric_loss_77ghz()
    alpha_db_m = (
        loss_info["alpha_c_db_per_cm"] + loss_info["alpha_d_db_per_cm"]
    ) * 100.0
    atten_lin = 10.0 ** (-alpha_db_m * x_centers / 20.0)

    out = {}
    for label, raw_w in (("tm100_ez_slot", v_mode), ("tm100_jx_current", i_mode)):
        ref = raw_w[0] if abs(raw_w[0]) > 1e-30 else 1.0
        w_norm = raw_w / abs(ref) * np.exp(-1j * np.angle(ref))
        amps = [round(float(np.abs(v)), 4) for v in w_norm]
        phases_deg = [
            round(float(np.rad2deg(np.angle(v))), 2) if abs(v) > 1e-6 else 0.0
            for v in w_norm
        ]
        af = np.sum(
            w_norm[:, None] * np.exp(1j * k0 * x_centers[:, None] * np.sin(th_rad)[None, :]),
            axis=0,
        )
        pat_af_db = 10.0 * np.log10(
            np.maximum(np.abs(af) ** 2 / max(np.max(np.abs(af) ** 2), 1e-30), 1e-5)
        )
        pat_af_ef_db = 10.0 * np.log10(
            np.maximum(
                np.abs(af * ef) ** 2 / max(np.max(np.abs(af * ef) ** 2), 1e-30), 1e-5
            )
        )
        sll_af = float(compute_sll_db(ang_deg, pat_af_db))
        sll_af_ef = float(compute_sll_db(ang_deg, pat_af_ef_db))
        pk_af_ef = float(ang_deg[int(np.argmax(pat_af_ef_db))])

        w_loss = w_norm * atten_lin
        af_loss = np.sum(
            w_loss[:, None] * np.exp(1j * k0 * x_centers[:, None] * np.sin(th_rad)[None, :]),
            axis=0,
        )
        pat_loss_db = 10.0 * np.log10(
            np.maximum(
                np.abs(af_loss * ef) ** 2 / max(np.max(np.abs(af_loss * ef) ** 2), 1e-30),
                1e-5,
            )
        )
        sll_af_ef_loss = float(compute_sll_db(ang_deg, pat_loss_db))

        out[label] = {
            "amplitudes_norm": amps,
            "phases_deg": phases_deg,
            "sll_af_only_db": round(sll_af, 2),
            "sll_af_times_ef_db": round(sll_af_ef, 2),
            "sll_af_times_ef_with_analytic_loss_db": round(sll_af_ef_loss, 2),
            "theta_peak_deg": round(pk_af_ef, 2),
        }
    return out


# ---------------------------------------------------------------------------
# 3D FIT Geometry, Mesh & Simulation Builders
# ---------------------------------------------------------------------------
def build_array_structures(
    topology: str = "tapered",
) -> tuple[list[Box], dict]:
    """Construct the 3D ``Box`` and conformal PEC ``sheets`` for the 77 GHz series-fed patch array.

    Uses the verbatim geometry from Yang et al. (*Sensors* 2020, 20, 3066, Sec. 3.2,
    p. 6; Figs. 4-6, pp. 6-7):
    - :math:`L_1 = 1260\\;\\mu\\text{m}`, :math:`W_1 = 1540\\;\\mu\\text{m}`,
      :math:`L_2 = 1260\\;\\mu\\text{m}`, :math:`W_2 = 320\\;\\mu\\text{m}`,
      unit-cell pitch :math:`d = L_1 + L_2 = 2520\\;\\mu\\text{m}`;
    - Input quarter-wave impedance transformer :math:`L_3 = 718\\;\\mu\\text{m}`,
      :math:`W_3 = 429\\;\\mu\\text{m}` at :math:`[-L_3, 0]` and open-circuit
      termination at the 10th patch's radiating edge (:math:`L_{\\text{term}} = 0`).
    - ``'uniform'`` (``"1111111111"``): 10 identical patches of width
      :math:`W_1 = 1540\\;\\mu\\text{m}`;
    - ``'tapered'`` or ``'binary_ga'`` (``"1011111111"`` GA-optimized design in Yang et al.):
      9 uniform :math:`W_1 = 1540\\;\\mu\\text{m}` patches with element :math:`n=1`
      omitted (``"0"``, replaced by the bare :math:`W_2 = 320\\;\\mu\\text{m}` line);
    - ``'width_taper_variation'``: optional raised-cosine width-taper variation
      (our own design variation, not in Yang et al.).

    Parameters
    ----------
    topology : {'uniform', 'tapered', 'binary_ga', 'width_taper_variation'}
        Array topology selector.

    Returns
    -------
    structures : list of Box
        Ordered structures (Rogers 5880 substrate, PEC ground, series spine,
        quarter-wave input matching transformer, and radiating patches).
    geom : dict
        Geometry dictionary in SI units [m], micrometres [um], and conformal `sheets`.
    """
    if topology not in ("uniform", "tapered", "binary_ga", "width_taper_variation"):
        raise ValueError(
            f"Unknown topology {topology!r}; expected 'uniform', 'tapered', 'binary_ga', or 'width_taper_variation'."
        )

    eps_r = YANG2020_GEOMETRY_UM["eps_r"]
    h_sub = YANG2020_GEOMETRY_UM["h_sub_um"] * 1e-6
    L1 = YANG2020_GEOMETRY_UM["L1_patch_um"] * 1e-6
    W1 = YANG2020_GEOMETRY_UM["W1_patch_um"] * 1e-6
    L2 = YANG2020_GEOMETRY_UM["L2_line_um"] * 1e-6
    W2 = YANG2020_GEOMETRY_UM["W2_line_um"] * 1e-6
    L3 = YANG2020_GEOMETRY_UM["L3_qwt_um"] * 1e-6
    W3 = YANG2020_GEOMETRY_UM["W3_qwt_um"] * 1e-6
    pitch = L1 + L2
    n_elem = int(YANG2020_GEOMETRY_UM["n_elements"])

    if topology == "uniform":
        widths_um = list(YANG2020_GEOMETRY_UM["uniform_widths_um"])
        code = YANG2020_GEOMETRY_UM["uniform_code"]
    elif topology in ("tapered", "binary_ga"):
        widths_um = list(YANG2020_GEOMETRY_UM["pure_binary_widths_um"])
        code = YANG2020_GEOMETRY_UM["tapered_ga_code"]
    else:
        widths_um = list(YANG2020_GEOMETRY_UM["width_taper_variation_widths_um"])
        code = YANG2020_GEOMETRY_UM["tapered_ga_code"]

    L_feed = L3
    L_term = 0.0
    x_starts = [idx * pitch for idx in range(n_elem)]
    x_min_arr = -L_feed
    x_port = -L_feed + 80e-6
    x_max_arr = x_starts[-1] + L1
    g_xmin = x_min_arr - 750e-6
    g_xmax = x_max_arr + 750e-6
    g_ymin = -(0.5 * W1 + 900e-6)
    g_ymax = 0.5 * W1 + 900e-6

    structures = [
        # Rogers RT/duroid 5880 substrate (h = 254 um, eps_r = 2.20)
        Box(
            (0.5 * (g_xmin + g_xmax), 0.0, 0.5 * h_sub),
            (g_xmax - g_xmin, g_ymax - g_ymin, h_sub),
            eps_r,
        ),
        # Full PEC ground plane at z = 0
        Box(
            (0.5 * (g_xmin + g_xmax), 0.0, 0.0),
            (g_xmax - g_xmin, g_ymax - g_ymin, 0.0),
            "pec",
        ),
        # Continuous series microstrip line of width W2 = 320 um
        Box(
            (0.5 * (x_min_arr + x_max_arr), 0.0, h_sub),
            (x_max_arr - x_min_arr, W2, 0.0),
            "pec",
        ),
        # Input quarter-wave transformer (L3 = 718 um, W3 = 429 um) at [-L3, 0]
        Box(
            (-0.5 * L3, 0.0, h_sub),
            (L3, W3, 0.0),
            "pec",
        ),
    ]

    sheets_dict = {
        "ground": rectangle_sheet(
            (0.5 * (g_xmin + g_xmax), 0.0, 0.0),
            (g_xmax - g_xmin, g_ymax - g_ymin, 0.0),
        ),
        "spine": rectangle_sheet(
            (0.5 * (x_min_arr + x_max_arr), 0.0, h_sub),
            (x_max_arr - x_min_arr, W2, 0.0),
        ),
        "qwt": rectangle_sheet(
            (-0.5 * L3, 0.0, h_sub),
            (L3, W3, 0.0),
        ),
    }

    patch_boxes_um = []
    for idx, (x0, w_um) in enumerate(zip(x_starts, widths_um)):
        x1 = x0 + L1
        w_p = w_um * 1e-6
        patch_boxes_um.append((float(x0 * 1e6), float(x1 * 1e6), float(w_um)))
        if w_um > 320.0:
            structures.append(Box((0.5 * (x0 + x1), 0.0, h_sub), (L1, w_p, 0.0), "pec"))
            sheets_dict[f"patch_{idx}"] = rectangle_sheet(
                (0.5 * (x0 + x1), 0.0, h_sub),
                (L1, w_p, 0.0),
            )

    geom = {
        "topology": topology,
        "code": code,
        "eps_r": eps_r,
        "h_sub": h_sub,
        "t_pec": 17.5e-6,
        "L1": L1,
        "L1_nom": L1,
        "W1": W1,
        "L2": L2,
        "L2_nom": L2,
        "W2": W2,
        "L3": L3,
        "W3": W3,
        "Lp": L1,
        "pitch_eff": pitch,
        "L_feed": L_feed,
        "L_term": L_term,
        "x_port": x_port,
        "w_port": W3,
        "x_starts": x_starts,
        "widths_um": widths_um,
        "patch_centers_x": [x0 + 0.5 * L1 for x0 in x_starts],
        "patch_widths_y": [w * 1e-6 for w in widths_um],
        "active_mask": [w > 320.0 for w in widths_um],
        "patch_boxes_um": patch_boxes_um,
        "x_min_arr": x_min_arr,
        "x_max_arr": x_max_arr,
        "g_xmin": g_xmin,
        "g_xmax": g_xmax,
        "g_ymin": g_ymin,
        "g_ymax": g_ymax,
        "sheets": sheets_dict,
    }
    return structures, geom


def build_array_axes(
    geom: dict,
    level: str = "coarse",
    n_pml: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build graded 3D mesh axes for the 77 GHz series-fed patch array.

    Evaluates Project Rule 1 (:math:`\\ge 20` cells per wavelength in Rogers 5880 at
    :math:`f_{\\max} = 80\\;\\text{GHz}`, where :math:`\\lambda_{\\min,\\text{sub}} =
    2526.5\\;\\mu\\text{m}` requires :math:`\\Delta_{\\max,\\text{sub}} \\le 126.3\\;\\mu\\text{m}`)
    and Project Rule 2 (:math:`\\ge 4` cells across the smallest features:
    :math:`h = 254\\;\\mu\\text{m} \\Rightarrow \\Delta z_{\\text{sub}} \\le 63.5\\;\\mu\\text{m}`
    and :math:`W_2 = 320\\;\\mu\\text{m} \\Rightarrow \\Delta y_{\\text{line}} \\le 80.0\\;\\mu\\text{m}`).
    On the ``'fine'`` CPU mesh, **both Rule 1 and Rule 2 are 100% satisfied**.

    Parameters
    ----------
    geom : dict
        Geometry dictionary returned by :func:`build_array_structures`.
    level : {'coarse', 'medium', 'fine', 'gpu_superfine_32um'}
        Resolution level.
    n_pml : int, optional
        CPML absorber thickness in cells. Default 6.

    Returns
    -------
    x, y, z : ndarray
        Mesh node arrays including CPML layers [m].
    mesh_info : dict
        Mesh resolution diagnostics and `wakis.estimate` preflight metrics.
    """
    presets = {
        # (dx_sub_um, dy_line_um, nz_sub, n_line_y)
        "coarse": (235.0, 160.0, 2, 2),
        "medium": (160.0, 106.7, 3, 3),
        "fine": (122.0, 80.0, 4, 4),
        "gpu_superfine_32um": (62.5, 40.0, 8, 8),
    }
    if level not in presets:
        raise ValueError(f"Unknown level {level!r}; expected one of {list(presets)}.")
    dx_sub_um, dy_line_um, nz_sub, n_line_y = presets[level]
    dx_sub = dx_sub_um * 1e-6
    dy_line = dy_line_um * 1e-6
    h_sub = geom["h_sub"]
    dz_sub = h_sub / nz_sub

    pad = 850e-6
    n2f_m = 420e-6
    W1, W2 = geom["W1"], geom["W2"]

    # Keep a uniform x-spacing across the array so Wakis's conformal sheet
    # knife-edge correction (sheet_edge_correction=True) resolves all patch edges
    # at their exact physical coordinates (L1 = 1260 um, d = 2520 um).
    x_snaps = [geom["g_xmin"], geom["x_min_arr"], 0.0, geom["x_max_arr"], geom["g_xmax"]]

    y_snaps = [
        geom["g_ymin"],
        -0.5 * W1,
        -0.5 * 1155e-6,
        -0.5 * 770e-6,
        -0.5 * 520e-6,
        -0.5 * W2,
        0.5 * W2,
        0.5 * 520e-6,
        0.5 * 770e-6,
        0.5 * 1155e-6,
        0.5 * W1,
        geom["g_ymax"],
    ]

    x_pts = meshing.merge_points(
        [geom["g_xmin"] - pad, geom["g_xmin"] - n2f_m]
        + x_snaps
        + [geom["g_xmax"] + n2f_m, geom["g_xmax"] + pad],
        1e-8,
    )
    y_pts = meshing.merge_points(
        [geom["g_ymin"] - pad, geom["g_ymin"] - n2f_m]
        + y_snaps
        + [geom["g_ymax"] + n2f_m, geom["g_ymax"] + pad],
        1e-8,
    )
    z_pts = [-pad, -n2f_m, 0.0, h_sub, h_sub + n2f_m, h_sub + pad]

    x_sizes = []
    for a, b in zip(x_pts[:-1], x_pts[1:]):
        mid = 0.5 * (a + b)
        in_sub = geom["g_xmin"] <= mid <= geom["g_xmax"]
        x_sizes.append(dx_sub if in_sub else max(dx_sub * 1.4, 210e-6))
    x_core = meshing.graded_axis(x_pts, x_sizes, [1] * len(x_sizes), grading=1.4)

    y_sizes = []
    y_min_c = []
    for a, b in zip(y_pts[:-1], y_pts[1:]):
        mid = 0.5 * (a + b)
        if abs(mid) <= 0.5 * W2 + 1e-8:
            y_sizes.append(dy_line)
            y_min_c.append(n_line_y)
        elif abs(mid) <= 0.5 * W1 + 1e-8:
            y_sizes.append(min(dx_sub, 122e-6 if level in ("fine", "gpu_superfine_32um") else 180e-6))
            y_min_c.append(1)
        elif geom["g_ymin"] <= mid <= geom["g_ymax"]:
            y_sizes.append(dx_sub)
            y_min_c.append(1)
        else:
            y_sizes.append(max(dx_sub * 1.4, 210e-6))
            y_min_c.append(1)
    y_core = meshing.graded_axis(y_pts, y_sizes, y_min_c, grading=1.4)

    z_sizes = [max(dx_sub * 1.4, 200e-6), max(dx_sub, 125e-6), dz_sub, max(dx_sub, 125e-6), max(dx_sub * 1.4, 200e-6)]
    z_min_c = [2, 2, nz_sub, 2, 2]
    z_core = meshing.graded_axis(z_pts, z_sizes, z_min_c, grading=1.4)

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

    # Minimum wavelength in Rogers 5880 (eps_r = 2.20) at f_max = 80 GHz
    lam_sub_min = float(c_light / (80.0e9 * np.sqrt(geom["eps_r"])))
    lam_air_min = float(c_light / 80.0e9)
    in_sub_x = dx[(x[:-1] >= geom["g_xmin"] - 1e-8) & (x[1:] <= geom["g_xmax"] + 1e-8)]
    in_sub_y = dy[(y[:-1] >= geom["g_ymin"] - 1e-8) & (y[1:] <= geom["g_ymax"] + 1e-8)]
    max_sub_cell = float(max(np.max(in_sub_x), np.max(in_sub_y), dz_sub))
    min_cpw_sub = float(lam_sub_min / max_sub_cell)
    min_cpw_air = float(lam_air_min / max(np.max(dx), np.max(dy), np.max(dz)))

    cells_w2 = int(np.sum((y[:-1] >= -0.5 * W2 - 1e-8) & (y[1:] <= 0.5 * W2 + 1e-8)))
    cells_h = int(nz_sub)

    dt_cfl085 = float(
        0.85
        / (
            c_light
            * np.sqrt(
                1.0 / np.min(dx) ** 2 + 1.0 / np.min(dy) ** 2 + 1.0 / np.min(dz) ** 2
            )
        )
    )
    n_steps_est = int(np.ceil(0.29e-9 / dt_cfl085))
    n_cells = int(dx.size * dy.size * dz.size)
    est_cpu = estimate(cells=n_cells, steps=n_steps_est, backend="cpu", cpml=True)
    est_gpu = estimate(cells=n_cells, steps=n_steps_est, backend="gpu", cpml=True, fused=True)

    mesh_info = {
        "level": level,
        "nx": int(dx.size),
        "ny": int(dy.size),
        "nz": int(dz.size),
        "n_cells": n_cells,
        "dx_min_um": float(np.min(dx) * 1e6),
        "dy_min_um": float(np.min(dy) * 1e6),
        "dz_min_um": float(np.min(dz) * 1e6),
        "dx_max_sub_um": float(np.max(in_sub_x) * 1e6),
        "dy_max_sub_um": float(np.max(in_sub_y) * 1e6),
        "lam_sub_fmax_um": float(lam_sub_min * 1e6),
        "min_cpw_sub": min_cpw_sub,
        "min_cpw_air": min_cpw_air,
        "rule_20cpw_met": bool(min_cpw_sub >= 20.0),
        "cells_across_h_254um": cells_h,
        "cells_across_W2_320um": cells_w2,
        "rule_4cells_feature_met": bool(cells_h >= 4 and cells_w2 >= 4),
        "dt_fs": float(dt_cfl085 * 1e15),
        "n_steps_est": n_steps_est,
        "memory_mb": float(est_cpu.host_bytes / 1e6),
        "gpu_memory_mb": float(est_gpu.gpu_bytes / 1e6),
        "cpu_wall_est_s": float(est_cpu.seconds),
        "gpu_wall_est_s": float(est_gpu.seconds),
        "n2f_margin_m": float(n2f_m),
    }
    return x, y, z, mesh_info


def preflight_all_levels() -> dict[str, dict]:
    """Compute `wakis.estimate` preflight metrics across all 4 resolution levels."""
    _, geom = build_array_structures("tapered")
    report = {}
    for lvl in ("coarse", "medium", "fine", "gpu_superfine_32um"):
        _, _, _, info = build_array_axes(geom, level=lvl)
        report[lvl] = info
    return report


def run_array_simulation(
    topology: str = "tapered",
    level: str = "coarse",
    record_fields: bool = False,
    record_animation: bool = False,
    compute_3d_pattern: bool = True,
    sim_time_ps: float = 290.0,
    energy_decay: float = 2e-4,
) -> dict:
    """Run a 3D FIT simulation of the 77 GHz series-fed patch array."""
    f0 = 77.0e9
    n_pml = 6
    structures, geom = build_array_structures(topology=topology)
    x, y, z, mesh_info = build_array_axes(geom, level=level, n_pml=n_pml)

    t0 = time.perf_counter()
    grid = GridFIT3D(
        x=x,
        y=y,
        z=z,
        sheets=geom["sheets"],
        sheet_edge_correction=True,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml"] * 3,
        bc_high=["cpml"] * 3,
        n_pml=n_pml,
        cfln=0.85,
        use_stl=False,
        verbose=0,
    )
    # Substrate is structures[0]; copper ground, spine, QWT, and patches are conformal sheets
    assign_materials(solver, [structures[0]], f0)

    pulse = GaussianSource(f0, 6.0e9, cutoff=3.5)
    port = LumpedPort(
        pulse,
        component="z",
        center=(geom["x_port"], 0.0, 0.5 * geom["h_sub"]),
        size=(0.0, geom["w_port"], geom["h_sub"]),
        impedance=50.0,
    )

    n2f_m = mesh_info["n2f_margin_m"]
    n2f_center = (0.5 * (geom["g_xmin"] + geom["g_xmax"]), 0.0, 0.5 * geom["h_sub"])
    n2f_size = (
        geom["g_xmax"] - geom["g_xmin"] + 2.0 * n2f_m,
        geom["g_ymax"] - geom["g_ymin"] + 2.0 * n2f_m,
        geom["h_sub"] + 2.0 * n2f_m,
    )
    f_eval_ghz = [75.0, 75.5, 76.0, 76.5, 77.0, 77.5, 78.0]
    f_n2f = np.array(f_eval_ghz, dtype=float) * 1e9
    i_77_n2f = f_eval_ghz.index(77.0)
    n2f = Near2FarMonitor(f_n2f, center=n2f_center, size=n2f_size)
    monitors = [n2f]

    dft_xy = None
    dft_xz = None
    dft_sub = None
    if record_fields:
        dft_xy = DFTMonitor(
            [f0],
            components=("Ex", "Ey", "Ez", "Hx", "Hy"),
            center=(0.5 * (geom["g_xmin"] + geom["g_xmax"]), 0.0, geom["h_sub"]),
            size=(
                geom["g_xmax"] - geom["g_xmin"] + 400e-6,
                geom["g_ymax"] - geom["g_ymin"] + 400e-6,
                0.0,
            ),
        )
        dft_xz = DFTMonitor(
            [f0],
            components=("Ex", "Ez"),
            center=(0.5 * (geom["g_xmin"] + geom["g_xmax"]), 0.0, 0.5 * geom["h_sub"]),
            size=(
                geom["g_xmax"] - geom["g_xmin"] + 600e-6,
                0.0,
                geom["h_sub"] + 1100e-6,
            ),
        )
        dft_sub = DFTMonitor(
            [f0],
            components=("Ez", "Hy"),
            center=(0.5 * (geom["g_xmin"] + geom["g_xmax"]), 0.0, 0.5 * geom["h_sub"]),
            size=(
                geom["g_xmax"] - geom["g_xmin"] + 400e-6,
                geom["g_ymax"] - geom["g_ymin"] + 400e-6,
                0.0,
            ),
        )
        monitors.extend([dft_xy, dft_xz, dft_sub])

    Nt = int(np.ceil(float(sim_time_ps) * 1e-12 / solver.dt))
    anim_frames = []
    anim_times_ps = []
    sources = [port]
    if record_animation:
        k_sub = max(1, int(np.argmin(np.abs(grid.z - 0.5 * geom["h_sub"]))))
        ix_lo, ix_hi = n_pml, grid.Nx - n_pml
        iy_lo, iy_hi = n_pml, grid.Ny - n_pml
        anim_x_mm = 0.5 * (grid.x[ix_lo:ix_hi] + grid.x[ix_lo + 1 : ix_hi + 1]) * 1e3
        anim_y_mm = 0.5 * (grid.y[iy_lo:iy_hi] + grid.y[iy_lo + 1 : iy_hi + 1]) * 1e3
        n_anim_steps = min(Nt, int(np.ceil(190.0e-12 / solver.dt)))
        stride = max(1, n_anim_steps // 26)
        for step in range(Nt):
            t_cur = step * solver.dt
            for s in sources:
                s.update(solver, t_cur)
            solver.one_step()
            for s in sources:
                s.post_step(solver, t_cur)
            for mon in monitors:
                mon.update(solver, t_cur)
            if step < n_anim_steps and step % stride == 0 and len(anim_frames) < 26:
                ez_slice = solver.E.to_matrix("z")[ix_lo:ix_hi, iy_lo:iy_hi, k_sub]
                anim_frames.append(np.asarray(ez_slice, dtype=np.float32))
                anim_times_ps.append(float((t_cur + solver.dt) * 1e12))
    else:
        solver.emsolve(
            Nt,
            source=sources,
            monitors=monitors,
            stop_when_energy_decayed=energy_decay,
        )

    wall_s = time.perf_counter() - t0
    freqs = np.linspace(72.0e9, 81.0e9, 181)
    zin = port.impedance_in(freqs)
    i_77 = int(np.argmin(np.abs(freqs - 77.0e9)))
    s11 = (zin - 50.0) / (zin + 50.0)
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-5))
    i_min = int(np.argmin(s11_db))
    f_res = float(freqs[i_min])
    s11_min_db = float(s11_db[i_min])
    zin_77 = complex(zin[i_77])
    in_band = freqs[s11_db <= -10.0]
    bw_10db_ghz = float((in_band[-1] - in_band[0]) / 1e9) if in_band.size > 1 else 0.0

    # Far-field E-plane (XZ) and H-plane (YZ) cuts evaluated at f0 = 77.0 GHz
    ang_deg = np.linspace(-90.0, 90.0, 361)
    th_cut = np.deg2rad(np.abs(ang_deg))
    ph_e = np.where(ang_deg >= 0.0, 0.0, np.pi)
    ph_h = np.where(ang_deg >= 0.0, 0.5 * np.pi, 1.5 * np.pi)

    P_rad_all = n2f.radiated_power()
    D_e_all = 4.0 * np.pi * n2f.radiation_intensity(th_cut, ph_e) / P_rad_all[:, None]
    D_h_all = 4.0 * np.pi * n2f.radiation_intensity(th_cut, ph_h) / P_rad_all[:, None]
    loss_budget = copper_and_dielectric_loss_77ghz()
    loss_db = loss_budget["total_loss_db"]

    freq_scan = {}
    for fi, fg in enumerate(f_eval_ghz):
        de_k = D_e_all[fi]
        de_k_dbi = 10.0 * np.log10(np.maximum(de_k, 1e-4))
        sll_k = float(compute_sll_db(ang_deg, de_k_dbi))
        pk_k = float(ang_deg[int(np.argmax(de_k_dbi))])
        hpbw_k = float(np.rad2deg(half_power_beamwidth(np.deg2rad(ang_deg), de_k)))
        freq_scan[f"{fg:.1f}"] = {
            "f_ghz": fg,
            "theta_peak_deg": round(pk_k, 2),
            "d_max_pec_dbi": round(float(np.max(de_k_dbi)), 2),
            "g_max_lossy_dbi": round(float(np.max(de_k_dbi) - loss_db), 2),
            "sll_e_db": round(sll_k, 2),
            "hpbw_e_half_deg": round(0.5 * hpbw_k, 2),
            "hpbw_e_full_deg": round(hpbw_k, 2),
        }

    D_e = D_e_all[i_77_n2f]
    D_h = D_h_all[i_77_n2f]
    D_e_dbi = 10.0 * np.log10(np.maximum(D_e, 1e-4))
    D_h_dbi = 10.0 * np.log10(np.maximum(D_h, 1e-4))
    G_e_dbi = D_e_dbi - loss_db
    G_h_dbi = D_h_dbi - loss_db

    D_max_dbi = float(np.max(D_e_dbi))
    G_max_dbi = float(D_max_dbi - loss_db)
    sll_e_db = compute_sll_db(ang_deg, D_e_dbi)
    theta_peak_deg = float(ang_deg[int(np.argmax(D_e_dbi))])
    hpbw_e_deg = float(np.rad2deg(half_power_beamwidth(np.deg2rad(ang_deg), D_e)))
    hpbw_h_deg = float(np.rad2deg(half_power_beamwidth(np.deg2rad(ang_deg), D_h)))

    # 3D upper-hemisphere directivity/gain grid for 3D fan-beam visualization at 77.0 GHz
    theta_3d = np.linspace(0.0, 0.55 * np.pi, 31)
    phi_3d = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    if compute_3d_pattern:
        TH3, PH3 = np.meshgrid(theta_3d, phi_3d, indexing="ij")
        U_3d_77 = n2f.radiation_intensity(TH3, PH3)[i_77_n2f]
        D_3d = 4.0 * np.pi * U_3d_77 / P_rad_all[i_77_n2f]
        G_3d_lin = D_3d * (10.0 ** (-loss_db / 10.0))
    else:
        G_3d_lin = np.zeros((theta_3d.size, phi_3d.size), dtype=np.float32)

    out = {
        "topology": topology,
        "level": level,
        "geom": geom,
        "mesh_info": mesh_info,
        "loss_budget": loss_budget,
        "wall_seconds": float(wall_s),
        "freqs_hz": freqs,
        "s11_real": np.real(s11),
        "s11_imag": np.imag(s11),
        "s11_db": s11_db,
        "zin_real": np.real(zin),
        "zin_imag": np.imag(zin),
        "f_res_ghz": float(f_res / 1e9),
        "s11_min_db": s11_min_db,
        "zin_77_real_ohm": float(zin_77.real),
        "zin_77_imag_ohm": float(zin_77.imag),
        "bw_10db_ghz": bw_10db_ghz,
        "d_max_pec_dbi": D_max_dbi,
        "g_max_lossy_dbi": G_max_dbi,
        "sll_e_db": sll_e_db,
        "theta_peak_deg": theta_peak_deg,
        "hpbw_e_full_deg": hpbw_e_deg,
        "hpbw_e_half_deg": 0.5 * hpbw_e_deg,
        "hpbw_h_full_deg": hpbw_h_deg,
        "hpbw_h_half_deg": 0.5 * hpbw_h_deg,
        "freq_scan": freq_scan,
        "ang_deg": ang_deg,
        "d_e_dbi": D_e_dbi,
        "d_h_dbi": D_h_dbi,
        "g_e_dbi": G_e_dbi,
        "g_h_dbi": G_h_dbi,
        "theta_3d_rad": theta_3d,
        "phi_3d_rad": phi_3d,
        "gain_3d_77g_lin": np.asarray(G_3d_lin, dtype=np.float32),
        "x_nodes_m": x,
        "y_nodes_m": y,
        "z_nodes_m": z,
    }

    if record_fields and dft_xy is not None and dft_xz is not None:
        if dft_sub is not None:
            out["excitations_77ghz"] = extract_tm100_excitations(
                dft_sub, geom["x_starts"], geom["L1"], geom["W1"], geom["code"]
            )
        Ex_xy, c_ex = dft_xy.get_field("Ex")
        Ey_xy, _ = dft_xy.get_field("Ey")
        Ez_xy, c_ez = dft_xy.get_field("Ez")
        Hx_xy, _ = dft_xy.get_field("Hx")
        Hy_xy, _ = dft_xy.get_field("Hy")

        ex_2d = np.squeeze(Ex_xy[0])
        ey_2d = np.squeeze(Ey_xy[0])
        ez_2d = np.squeeze(Ez_xy[0])
        hx_2d = np.squeeze(Hx_xy[0])
        hy_2d = np.squeeze(Hy_xy[0])
        nx_m = min(ex_2d.shape[0], ey_2d.shape[0], ez_2d.shape[0], hx_2d.shape[0], hy_2d.shape[0])
        ny_m = min(ex_2d.shape[1], ey_2d.shape[1], ez_2d.shape[1], hx_2d.shape[1], hy_2d.shape[1])
        ex_2d, ey_2d, ez_2d = ex_2d[:nx_m, :ny_m], ey_2d[:nx_m, :ny_m], ez_2d[:nx_m, :ny_m]
        hx_2d, hy_2d = hx_2d[:nx_m, :ny_m], hy_2d[:nx_m, :ny_m]
        xc_mm = c_ez[0][:nx_m] * 1e3
        yc_mm = c_ez[1][:ny_m] * 1e3

        e_mag_xy = np.sqrt(np.abs(ex_2d) ** 2 + np.abs(ey_2d) ** 2 + np.abs(ez_2d) ** 2)
        jx_s = -hy_2d
        jy_s = hx_2d
        j_mag_xy = np.sqrt(np.abs(jx_s) ** 2 + np.abs(jy_s) ** 2)

        Ex_xz, c_xz_x = dft_xz.get_field("Ex")
        Ez_xz, c_xz_z = dft_xz.get_field("Ez")
        ex_xz = np.squeeze(Ex_xz[0])
        ez_xz = np.squeeze(Ez_xz[0])
        nx_xz = min(ex_xz.shape[0], ez_xz.shape[0])
        nz_xz = min(ex_xz.shape[1], ez_xz.shape[1])
        ex_xz, ez_xz = ex_xz[:nx_xz, :nz_xz], ez_xz[:nx_xz, :nz_xz]
        e_mag_xz = np.sqrt(np.abs(ex_xz) ** 2 + np.abs(ez_xz) ** 2)

        out.update(
            {
                "xy_x_mm": xc_mm,
                "xy_y_mm": yc_mm,
                "e_mag_xy": np.asarray(e_mag_xy, dtype=np.float32),
                "ez_real_xy": np.asarray(np.real(ez_2d), dtype=np.float32),
                "jx_real_xy": np.asarray(np.real(jx_s), dtype=np.float32),
                "jy_real_xy": np.asarray(np.real(jy_s), dtype=np.float32),
                "j_mag_xy": np.asarray(j_mag_xy, dtype=np.float32),
                "xz_x_mm": c_xz_x[0][:nx_xz] * 1e3,
                "xz_z_mm": c_xz_z[2][:nz_xz] * 1e3,
                "e_mag_xz": np.asarray(e_mag_xz, dtype=np.float32),
                "ez_real_xz": np.asarray(np.real(ez_xz), dtype=np.float32),
            }
        )

    if record_animation and anim_frames:
        out["anim_x_mm"] = anim_x_mm
        out["anim_y_mm"] = anim_y_mm
        out["anim_frames"] = np.stack(anim_frames, axis=0)
        out["anim_times_ps"] = np.asarray(anim_times_ps, dtype=float)

    return out


CPU_TASKS = (
    "uniform_coarse",
    "tapered_coarse",
    "res_medium",
    "res_fine",
)
GPU_TASKS = ("gpu_superfine_32um",)
ALL_TASKS = CPU_TASKS + GPU_TASKS


def _scalar_summary(res: dict) -> dict:
    """Extract JSON-serializable scalar metrics from a simulation result dict."""
    keys = (
        "topology",
        "level",
        "wall_seconds",
        "f_res_ghz",
        "s11_min_db",
        "zin_77_real_ohm",
        "zin_77_imag_ohm",
        "bw_10db_ghz",
        "d_max_pec_dbi",
        "g_max_lossy_dbi",
        "sll_e_db",
        "theta_peak_deg",
        "hpbw_e_full_deg",
        "hpbw_e_half_deg",
        "hpbw_h_full_deg",
        "hpbw_h_half_deg",
        "freq_scan",
        "excitations_77ghz",
    )
    out = {k: res[k] for k in keys if k in res}
    out["mesh_info"] = res["mesh_info"]
    out["loss_budget"] = res["loss_budget"]
    return out


def save_run_npz(path: Path, res: dict, include_fields: bool = False) -> None:
    """Save compact simulation curves and far-field grids to a compressed `.npz`."""
    arrays = {
        "freqs_hz": res["freqs_hz"],
        "s11_real": res["s11_real"],
        "s11_imag": res["s11_imag"],
        "s11_db": res["s11_db"],
        "zin_real": res["zin_real"],
        "zin_imag": res["zin_imag"],
        "ang_deg": res["ang_deg"],
        "d_e_dbi": res["d_e_dbi"],
        "d_h_dbi": res["d_h_dbi"],
        "g_e_dbi": res["g_e_dbi"],
        "g_h_dbi": res["g_h_dbi"],
        "theta_3d_rad": res["theta_3d_rad"],
        "phi_3d_rad": res["phi_3d_rad"],
        "gain_3d_77g_lin": res["gain_3d_77g_lin"],
        "x_nodes_m": res["x_nodes_m"],
        "y_nodes_m": res["y_nodes_m"],
        "z_nodes_m": res["z_nodes_m"],
    }
    if include_fields and "e_mag_xy" in res:
        for k in (
            "xy_x_mm",
            "xy_y_mm",
            "e_mag_xy",
            "ez_real_xy",
            "jx_real_xy",
            "jy_real_xy",
            "j_mag_xy",
            "xz_x_mm",
            "xz_z_mm",
            "e_mag_xz",
            "ez_real_xz",
        ):
            arrays[k] = res[k]
    np.savez_compressed(path, **arrays)


def update_summary_json(run_key: str, res: dict) -> None:
    """Update `data/summary.json` with preflight, cavity model, and run metrics."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DATA_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}

    summary["geometry_um"] = YANG2020_GEOMETRY_UM
    summary["benchmarks"] = YANG2020_BENCHMARKS
    summary["source_locations"] = YANG2020_SOURCE_LOCATIONS
    summary["cavity_model"] = patch_cavity_resonance_hz()
    summary["loss_budget_77ghz"] = copper_and_dielectric_loss_77ghz()
    summary["preflight"] = preflight_all_levels()
    summary.setdefault("runs", {})[run_key] = _scalar_summary(res)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_task(task_name: str, sim_time_ps: float = 310.0) -> dict:
    """Execute a named simulation task and save its outputs under `data/`."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if task_name == "uniform_coarse":
        res = run_array_simulation(
            topology="uniform",
            level="coarse",
            record_fields=True,
            record_animation=False,
            compute_3d_pattern=False,
            sim_time_ps=sim_time_ps,
        )
        save_run_npz(DATA_DIR / "uniform_coarse.npz", res, include_fields=True)
        update_summary_json("uniform_coarse", res)
        return res

    if task_name == "tapered_coarse":
        res = run_array_simulation(
            topology="tapered",
            level="coarse",
            record_fields=True,
            record_animation=True,
            compute_3d_pattern=True,
            sim_time_ps=sim_time_ps,
        )
        save_run_npz(DATA_DIR / "tapered_coarse.npz", res, include_fields=True)
        if "anim_frames" in res:
            np.savez_compressed(
                DATA_DIR / "tapered_animation.npz",
                anim_x_mm=res["anim_x_mm"],
                anim_y_mm=res["anim_y_mm"],
                anim_frames=res["anim_frames"],
                anim_times_ps=res["anim_times_ps"],
            )
        update_summary_json("tapered_coarse", res)
        return res

    if task_name == "res_medium":
        res = run_array_simulation(
            topology="tapered",
            level="medium",
            record_fields=False,
            record_animation=False,
            compute_3d_pattern=False,
            sim_time_ps=sim_time_ps,
        )
        save_run_npz(DATA_DIR / "res_medium.npz", res, include_fields=False)
        update_summary_json("res_medium", res)
        return res

    if task_name == "res_fine":
        res = run_array_simulation(
            topology="tapered",
            level="fine",
            record_fields=False,
            record_animation=False,
            sim_time_ps=sim_time_ps,
        )
        save_run_npz(DATA_DIR / "res_fine.npz", res, include_fields=False)
        update_summary_json("res_fine", res)
        return res

    if task_name == "gpu_superfine_32um":
        res = run_array_simulation(
            topology="tapered",
            level="gpu_superfine_32um",
            record_fields=False,
            record_animation=False,
            sim_time_ps=sim_time_ps,
        )
        save_run_npz(DATA_DIR / "gpu_superfine_32um.npz", res, include_fields=False)
        update_summary_json("gpu_superfine_32um", res)
        return res

    raise ValueError(f"Unknown task {task_name!r}; expected one of {ALL_TASKS}")


# ---------------------------------------------------------------------------
# Figure generation helpers (PyVista 3D & Matplotlib 2D/Polar/Smith)
# ---------------------------------------------------------------------------
def _make_pv_box(bounds_m: tuple[float, ...], scale: float = 1e3):
    """Create a PyVista Box in mm from meter bounds (xmin, xmax, ymin, ymax, zmin, zmax)."""
    import pyvista as pv

    b = tuple(v * scale for v in bounds_m)
    return pv.Box(bounds=b)


def _add_array_scene_meshes(plotter, geom: dict, z_explode: float = 1.0) -> None:
    """Add colour-coded Rogers 5880 substrate, ground, QWT, patches, and port to a PyVista scene."""
    h_mm = geom["h_sub"] * 1e3 * z_explode
    t_mm = max(geom["t_pec"] * 1e3 * z_explode, 0.025)

    # Ground plane
    gnd = _make_pv_box(
        (geom["g_xmin"], geom["g_xmax"], geom["g_ymin"], geom["g_ymax"], -0.02 / 1e3, 0.0)
    )
    plotter.add_mesh(gnd, color="#b87333", metallic=0.4, roughness=0.4, label="Ground (Cu)")

    # Rogers RT/duroid 5880 substrate
    sub = _make_pv_box(
        (
            geom["g_xmin"],
            geom["g_xmax"],
            geom["g_ymin"],
            geom["g_ymax"],
            0.0,
            h_mm / 1e3,
        )
    )
    plotter.add_mesh(
        sub,
        color="#2a9d8f",
        opacity=0.68,
        show_edges=False,
        label="Rogers 5880 (h=254 um)",
    )

    # Input quarter-wave transformer (L3 = 718 um, W3 = 429 um) and series line (W2 = 320 um)
    if geom.get("include_qwt", True) and geom.get("L3", 0.0) > 0.0:
        qwt_box = _make_pv_box(
            (
                -geom["L3"],
                0.0,
                -0.5 * geom["W3"],
                0.5 * geom["W3"],
                h_mm / 1e3,
                (h_mm + t_mm) / 1e3,
            )
        )
        plotter.add_mesh(qwt_box, color="#eab308", show_edges=True, edge_color="#713f12", line_width=0.6)
        x_spine_start = 0.0
    else:
        x_spine_start = -geom["L_feed"]

    line_box = _make_pv_box(
        (
            x_spine_start,
            geom["x_max_arr"],
            -0.5 * geom["W2"],
            0.5 * geom["W2"],
            h_mm / 1e3,
            (h_mm + t_mm) / 1e3,
        )
    )
    plotter.add_mesh(line_box, color="#e9c46a", smooth_shading=False)

    # Patches
    L1 = geom["L1"]
    for idx, (xc, w_p, active) in enumerate(
        zip(geom["patch_centers_x"], geom["patch_widths_y"], geom["active_mask"])
    ):
        if not active:
            continue
        p_box = _make_pv_box(
            (
                xc - 0.5 * L1,
                xc + 0.5 * L1,
                -0.5 * w_p,
                0.5 * w_p,
                h_mm / 1e3,
                (h_mm + t_mm) / 1e3,
            )
        )
        col = "#f4a261" if idx > 1 else "#e76f51"
        plotter.add_mesh(p_box, color=col, show_edges=True, edge_color="#5c3a21", line_width=0.8)

    # Feed port marker at x_port
    xp = geom["x_port"]
    wp = geom["w_port"]
    port_box = _make_pv_box(
        (
            xp - 0.08e-3,
            xp + 0.08e-3,
            -0.5 * wp,
            0.5 * wp,
            0.0,
            (h_mm + t_mm) / 1e3,
        )
    )
    plotter.add_mesh(port_box, color="#e63946", label="50 Ohm Lumped Port")


def plot_01_geometry_3d_views() -> Path:
    """Generate `01_geometry_3d_views.png`: 4-angle PyVista render + annotated layout."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    _, geom_tap = build_array_structures("tapered")
    _, geom_uni = build_array_structures("uniform")

    # Render (a) Isometric 3D and (d) Feed / "0"-element close-up via PyVista
    pl_iso = pv.Plotter(off_screen=True, window_size=(1100, 620))
    pl_iso.set_background("white")
    _add_array_scene_meshes(pl_iso, geom_tap, z_explode=2.2)
    pl_iso.camera_position = [
        (11.5, -18.5, 11.0),
        (12.0, 0.0, 0.25),
        (0.0, 0.0, 1.0),
    ]
    img_iso = pl_iso.screenshot(return_img=True)
    pl_iso.close()

    pl_zoom = pv.Plotter(off_screen=True, window_size=(1100, 620))
    pl_zoom.set_background("white")
    _add_array_scene_meshes(pl_zoom, geom_tap, z_explode=2.5)
    pl_zoom.camera_position = [
        (1.6, -5.2, 3.6),
        (1.8, 0.0, 0.25),
        (0.0, 0.0, 1.0),
    ]
    img_zoom = pl_zoom.screenshot(return_img=True)
    pl_zoom.close()

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), constrained_layout=True)

    # Panel (a): Isometric 3D view
    ax = axes[0, 0]
    ax.imshow(img_iso)
    ax.axis("off")
    ax.set_title(
        "(a) 3D Isometric View — 10-Element 77 GHz Array (Rogers 5880, h = 254 µm)",
        fontsize=10.2,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.04,
        "Substrate: Rogers 5880 (εr = 2.2, h = 254 µm, Yang et al. Sec. 3.2)\n"
        "Paper Binary '1011111111' (W1 = 1540 µm, L3 = 718 µm QWT, open end)",
        transform=ax.transAxes,
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#cbd5e1"),
    )

    # Panel (b): Annotated Top View comparing Uniform ("1111111111") vs Paper Binary ("1011111111")
    ax = axes[0, 1]
    ax.set_facecolor("#f8fafc")
    for y_off, geom, label, col_p in [
        (1.60, geom_uni, "Uniform '1111111111' (W1 = 1540 µm, SLL = -12.5 dB sim / -10 dB meas)", "#3b82f6"),
        (-1.75, geom_tap, "Paper Binary '1011111111' (W1 = 1540 µm, SLL = -16.5 dB sim / -14.8 dB meas)", "#f97316"),
    ]:
        # Input QWT + Series line
        if geom.get("include_qwt", True) and geom.get("L3", 0.0) > 0.0:
            ax.add_patch(
                plt.Rectangle(
                    (-geom["L3"] * 1e3, y_off - 0.5 * geom["W3"] * 1e3),
                    geom["L3"] * 1e3,
                    geom["W3"] * 1e3,
                    facecolor="#f59e0b",
                    edgecolor="#713f12",
                    lw=0.7,
                )
            )
            x_sp = 0.0
            w_sp = geom["x_max_arr"] * 1e3
        else:
            x_sp = -geom["L_feed"] * 1e3
            w_sp = (geom["x_max_arr"] + geom["L_feed"]) * 1e3
        ax.add_patch(
            plt.Rectangle(
                (x_sp, y_off - 0.5 * geom["W2"] * 1e3),
                w_sp,
                geom["W2"] * 1e3,
                facecolor="#eab308",
                edgecolor="#713f12",
                lw=0.6,
            )
        )
        for idx, (xc, wp, act) in enumerate(
            zip(geom["patch_centers_x"], geom["patch_widths_y"], geom["active_mask"])
        ):
            xc_mm = xc * 1e3
            l1_mm = geom["L1"] * 1e3
            wp_mm = wp * 1e3
            if act:
                ax.add_patch(
                    plt.Rectangle(
                        (xc_mm - 0.5 * l1_mm, y_off - 0.5 * wp_mm),
                        l1_mm,
                        wp_mm,
                        facecolor=col_p,
                        edgecolor="#1e293b",
                        lw=0.8,
                        alpha=0.85,
                    )
                )
                ax.text(
                    xc_mm,
                    y_off + 0.5 * wp_mm + 0.08,
                    "1",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    fontweight="bold",
                )
            else:
                ax.add_patch(
                    plt.Rectangle(
                        (xc_mm - 0.5 * l1_mm, y_off - 0.5 * (1.54)),
                        l1_mm,
                        1.54,
                        facecolor="none",
                        edgecolor="#dc2626",
                        ls="--",
                        lw=1.1,
                    )
                )
                ax.text(
                    xc_mm,
                    y_off + 0.80,
                    "0 (omitted)",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    color="#dc2626",
                    fontweight="bold",
                )
        ax.text(-1.4, y_off + 1.26, label, fontsize=8.5, fontweight="bold", color="#0f172a")

    ax.annotate(
        "Yang et al. Fig. 4: L1 = 1260 µm, L2 = 1260 µm (d = 2520 µm), L3 = 718 µm QWT, open end",
        xy=(10.32, -1.75),
        xytext=(3.2, -3.55),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#334155"),
        fontsize=8.2,
        fontweight="bold",
    )
    ax.set_xlim(-1.8, 26.2)
    ax.set_ylim(-3.95, 3.65)
    ax.set_xlabel("Array axis x [mm]", fontsize=10)
    ax.set_ylabel("Transverse axis y [mm]", fontsize=10)
    ax.set_title(
        "(b) Top View (+z) — Uniform '1111111111' vs Paper Binary '1011111111' Layout",
        fontsize=10.2,
        fontweight="bold",
    )
    ax.grid(True, ls=":", alpha=0.5)

    # Panel (c): Side Cross-Section (xz plane)
    ax = axes[1, 0]
    ax.set_facecolor("#f8fafc")
    ax.add_patch(
        plt.Rectangle(
            (geom_tap["g_xmin"] * 1e3, -25.0),
            (geom_tap["g_xmax"] - geom_tap["g_xmin"]) * 1e3,
            25.0,
            facecolor="#b87333",
            label="Cu Ground (z = 0)",
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (geom_tap["g_xmin"] * 1e3, 0.0),
            (geom_tap["g_xmax"] - geom_tap["g_xmin"]) * 1e3,
            254.0,
            facecolor="#2a9d8f",
            alpha=0.65,
            label="Rogers 5880 (h = 254 µm)",
        )
    )
    for xc, act in zip(geom_tap["patch_centers_x"], geom_tap["active_mask"]):
        if act:
            ax.add_patch(
                plt.Rectangle(
                    ((xc - 0.5 * geom_tap["L1"]) * 1e3, 254.0),
                    geom_tap["L1"] * 1e3,
                    22.0,
                    facecolor="#f97316",
                    edgecolor="#7c2d12",
                )
            )
    ax.add_patch(
        plt.Rectangle(
            (geom_tap["x_port"] * 1e3 - 0.08, 0.0),
            0.16,
            254.0,
            facecolor="#e63946",
            label="50 Ω Port (at QWT input)",
        )
    )
    ax.set_xlim(-2.2, 26.2)
    ax.set_ylim(-60.0, 420.0)
    ax.set_xlabel("Array axis x [mm]", fontsize=10)
    ax.set_ylabel("Height z [µm]", fontsize=10)
    ax.set_title(
        "(c) E-Plane Side View (xz) — Substrate Thickness h = 254 µm & Series Pitch",
        fontsize=10.2,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, ls=":", alpha=0.5)

    # Panel (d): PyVista Feed + "0" Element Close-up
    ax = axes[1, 1]
    ax.imshow(img_zoom)
    ax.axis("off")
    ax.set_title(
        "(d) Close-Up — Input QWT (L3=718 µm), Element #0 ('1'), Omitted #1 ('0'), Element #2 ('1')",
        fontsize=9.8,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.04,
        "Input QWT: L3 = 718 µm, W3 = 429 µm (50 Ω port)\n"
        "Yang et al. Fig. 4: L1 = 1260 µm, W1 = 1540 µm, L2 = 1260 µm, W2 = 320 µm\n"
        "Section #1 ('0'): bare W2 line replacing omitted patch; 10th patch open-ended",
        transform=ax.transAxes,
        fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#cbd5e1"),
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "01_geometry_3d_views.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_02_mesh_and_cpw_views(preflight: dict) -> Path:
    """Generate `02_mesh_and_cpw_views.png`: mesh slices, smallest feature zoom, and CPW map."""
    _, geom = build_array_structures("tapered")
    x_c, y_c, z_c, info_c = build_array_axes(geom, level="coarse")
    x_f, y_f, z_f, info_f = build_array_axes(geom, level="fine")

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.8), constrained_layout=True)

    # (a) xy mesh edges on the first 4 unit cells (fine grid)
    ax = axes[0, 0]
    for xv in x_f * 1e3:
        if -1.2 <= xv <= 9.8:
            ax.axvline(xv, color="#94a3b8", lw=0.45, alpha=0.7)
    for yv in y_f * 1e3:
        if -1.6 <= yv <= 1.6:
            ax.axhline(yv, color="#94a3b8", lw=0.45, alpha=0.7)
    if geom.get("include_qwt", True) and geom.get("L3", 0.0) > 0.0:
        ax.add_patch(
            plt.Rectangle(
                (-geom["L3"] * 1e3, -0.5 * geom["W3"] * 1e3),
                geom["L3"] * 1e3,
                geom["W3"] * 1e3,
                facecolor="#fbbf24",
                edgecolor="#b45309",
                alpha=0.82,
            )
        )
    ax.add_patch(
        plt.Rectangle(
            (0.0, -0.5 * geom["W2"] * 1e3),
            10.2,
            geom["W2"] * 1e3,
            facecolor="#fde047",
            edgecolor="#ca8a04",
            alpha=0.75,
        )
    )
    for xc, wp, act in zip(
        geom["patch_centers_x"][:4], geom["patch_widths_y"][:4], geom["active_mask"][:4]
    ):
        if act:
            ax.add_patch(
                plt.Rectangle(
                    ((xc - 0.5 * geom["L1"]) * 1e3, -0.5 * wp * 1e3),
                    geom["L1"] * 1e3,
                    wp * 1e3,
                    facecolor="#fb923c",
                    edgecolor="#9a3412",
                    alpha=0.8,
                )
            )
    ax.set_xlim(-1.1, 9.5)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        f"(a) xy Graded Mesh on Trace (`fine`: {info_f['nx']}×{info_f['ny']}×{info_f['nz']} = {info_f['n_cells']:,} cells)",
        fontsize=10.5,
        fontweight="bold",
    )

    # (b) xz slice through feed & substrate
    ax = axes[0, 1]
    for xv in x_f * 1e3:
        if -1.2 <= xv <= 6.5:
            ax.axvline(xv, color="#94a3b8", lw=0.45, alpha=0.7)
    for zv in z_f * 1e6:
        if -250.0 <= zv <= 650.0:
            ax.axhline(zv, color="#64748b", lw=0.6, alpha=0.8)
    ax.axhspan(0.0, 254.0, color="#2a9d8f", alpha=0.35, label="Rogers 5880 (h = 254 µm)")
    ax.axhline(0.0, color="#b87333", lw=2.0, label="Ground plane (z = 0)")
    ax.set_xlim(-1.1, 6.2)
    ax.set_ylim(-200.0, 600.0)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [µm]")
    ax.set_title(
        f"(b) xz Mesh Slice Across Substrate (nz_sub = {info_f['cells_across_h_254um']} cells in h = 254 µm)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8.5)

    # (c) Smallest feature zoom: W2 = 320 µm and h = 254 µm
    ax = axes[1, 0]
    levels = ["coarse", "medium", "fine", "gpu_superfine_32um"]
    x_pos = np.arange(len(levels))
    c_h = [preflight[k]["cells_across_h_254um"] for k in levels]
    c_w2 = [preflight[k]["cells_across_W2_320um"] for k in levels]
    w_bar = 0.34
    ax.bar(x_pos - 0.5 * w_bar, c_h, width=w_bar, color="#0ea5e9", label="Substrate h = 254 µm")
    ax.bar(x_pos + 0.5 * w_bar, c_w2, width=w_bar, color="#f97316", label="Feed line W2 = 320 µm")
    ax.axhline(4.0, color="#dc2626", ls="--", lw=1.4, label="Project rule: ≥ 4 cells")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["coarse\n(CPU)", "medium\n(CPU)", "fine\n(CPU)", "superfine_32um\n(GPU task)"])
    ax.set_ylabel("Cells across smallest feature")
    ax.set_title(
        "(c) Smallest-Feature Discretization (h = 254 µm & W2 = 320 µm)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, axis="y", ls=":", alpha=0.5)

    # (d) 2D cells-per-wavelength map in xz plane at f_max = 80 GHz
    ax = axes[1, 1]
    dx_f = np.diff(x_f)
    dz_f = np.diff(z_f)
    xc_f = 0.5 * (x_f[:-1] + x_f[1:])
    zc_f = 0.5 * (z_f[:-1] + z_f[1:])
    XC, ZC = np.meshgrid(xc_f, zc_f, indexing="ij")
    DX, DZ = np.meshgrid(dx_f, dz_f, indexing="ij")
    in_sub = (
        (XC >= geom["g_xmin"])
        & (XC <= geom["g_xmax"])
        & (ZC >= 0.0)
        & (ZC <= geom["h_sub"])
    )
    lam_local = np.where(
        in_sub,
        c_light / (80.0e9 * np.sqrt(geom["eps_r"])),
        c_light / 80.0e9,
    )
    cpw_xz = lam_local / np.maximum(DX, DZ)
    im = ax.pcolormesh(
        xc_f * 1e3,
        zc_f * 1e3,
        cpw_xz.T,
        cmap="viridis",
         shading="auto",
        vmin=15.0,
        vmax=45.0,
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Cells per wavelength at 80 GHz")
    ax.set_xlim(-2.0, 27.5)
    ax.set_ylim(-1.2, 1.5)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(
        f"(d) Cells-per-Wavelength Map at 80 GHz (min in substrate = {info_f['min_cpw_sub']:.1f} ≥ 20)",
        fontsize=10.5,
        fontweight="bold",
    )

    out_path = FIG_DIR / "02_mesh_and_cpw_views.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_03_near_fields_and_currents(tap_npz: dict, uni_npz: dict) -> Path:
    """Generate `03_near_fields_and_currents.png`: |E|, Re(Ez), xz cavity slice, and |Js|."""
    _, geom_tap = build_array_structures("tapered")
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.8), constrained_layout=True)

    x_xy = tap_npz["xy_x_mm"]
    y_xy = tap_npz["xy_y_mm"]
    e_mag = tap_npz["e_mag_xy"]
    e_db = 20.0 * np.log10(np.maximum(e_mag / np.max(e_mag), 1e-3))

    # (a) |E| (dB) on z = h
    ax = axes[0, 0]
    im0 = ax.pcolormesh(x_xy, y_xy, e_db.T, cmap="inferno", shading="auto", vmin=-35.0, vmax=0.0)
    for xc, wp, act in zip(
        geom_tap["patch_centers_x"], geom_tap["patch_widths_y"], geom_tap["active_mask"]
    ):
        if act:
            ax.add_patch(
                plt.Rectangle(
                    ((xc - 0.5 * geom_tap["L1"]) * 1e3, -0.5 * wp * 1e3),
                    geom_tap["L1"] * 1e3,
                    wp * 1e3,
                    fill=False,
                    edgecolor="cyan",
                    lw=0.8,
                )
            )
    fig.colorbar(im0, ax=ax, label="Normalized |E| [dB]")
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        "(a) Normalized |E(x, y, z=h)| at 77.0 GHz — Radiating Slot Maxima at Patch Edges",
        fontsize=10.5,
        fontweight="bold",
    )

    # (b) Re(Ez) on z = h showing 2π in-phase excitation across the series array
    ax = axes[0, 1]
    ez_re = tap_npz["ez_real_xy"]
    vmax = float(np.percentile(np.abs(ez_re), 99.2))
    im1 = ax.pcolormesh(
        x_xy,
        y_xy,
        ez_re.T,
        cmap="RdBu_r",
        shading="auto",
        vmin=-vmax,
        vmax=vmax,
    )
    for xc, wp, act in zip(
        geom_tap["patch_centers_x"], geom_tap["patch_widths_y"], geom_tap["active_mask"]
    ):
        if act:
            ax.add_patch(
                plt.Rectangle(
                    ((xc - 0.5 * geom_tap["L1"]) * 1e3, -0.5 * wp * 1e3),
                    geom_tap["L1"] * 1e3,
                    wp * 1e3,
                    fill=False,
                    edgecolor="black",
                    lw=0.8,
                )
            )
    fig.colorbar(im1, ax=ax, label="Re(Ez) [V/m]")
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        "(b) Re(Ez) on Patch Plane (z = h) — 2π In-Phase Excitation",
        fontsize=10.0,
        fontweight="bold",
    )

    # (c) xz slice of Re(Ez)
    ax = axes[1, 0]
    x_xz = tap_npz["xz_x_mm"]
    z_xz = tap_npz["xz_z_mm"]
    ez_xz = tap_npz["ez_real_xz"]
    vmax_xz = float(np.percentile(np.abs(ez_xz), 98.5))
    im2 = ax.pcolormesh(
        x_xz,
        z_xz,
        ez_xz.T,
        cmap="RdBu_r",
        shading="auto",
        vmin=-vmax_xz,
        vmax=vmax_xz,
    )
    ax.axhline(0.0, color="black", lw=1.2)
    ax.axhline(0.254, color="#0f766e", ls="--", lw=1.0)
    fig.colorbar(im2, ax=ax, label="Re(Ez) [V/m]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(
        "(c) Longitudinal E-Plane Re(Ez(x, 0, z)) — TM100 Cavity & Wavefront",
        fontsize=10.0,
        fontweight="bold",
    )

    # (d) Surface current density |Js| & quiver + 1D envelope comparison (Uniform vs Tapered)
    ax = axes[1, 1]
    j_tap_1d = np.mean(tap_npz["j_mag_xy"], axis=1)
    j_uni_1d = np.mean(uni_npz["j_mag_xy"], axis=1)
    j_tap_1d = j_tap_1d / max(np.max(j_tap_1d), 1e-12)
    j_uni_1d = j_uni_1d / max(np.max(j_uni_1d), 1e-12)
    ax.plot(
        uni_npz["xy_x_mm"],
        j_uni_1d,
        color="#2563eb",
        lw=1.8,
        label="Uniform '1111111111' |Js(x)|",
    )
    ax.plot(
        tap_npz["xy_x_mm"],
        j_tap_1d,
        color="#ea580c",
        lw=1.8,
        label="Paper Binary '1011111111' |Js(x)|",
    )
    for idx, xc in enumerate(geom_tap["patch_centers_x"]):
        ax.axvline(xc * 1e3, color="#94a3b8", ls=":", lw=0.8)
        ax.text(
            xc * 1e3,
            1.02,
            f"P{idx}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#475569",
        )
    ax.set_xlim(-1.0, 26.8)
    ax.set_ylim(0.0, 1.26)
    ax.set_xlabel("Array axis x [mm]")
    ax.set_ylabel("Normalized width-averaged |Js(x)|")
    ax.set_title(
        "(d) Surface Current Along Array Axis — Traveling-Wave Decay & '0' Element",
        fontsize=10.0,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.grid(True, ls=":", alpha=0.5)

    out_path = FIG_DIR / "03_near_fields_and_currents.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_04_radiated_field_gif(anim_npz: dict) -> Path:
    """Generate `04_radiated_field_evolution.gif` (< 5 MB) showing 77 GHz wave propagation."""
    from PIL import Image

    _, geom_tap = build_array_structures("tapered")
    x_mm = anim_npz["anim_x_mm"]
    y_mm = anim_npz["anim_y_mm"]
    frames_arr = anim_npz["anim_frames"]
    times_ps = anim_npz["anim_times_ps"]
    vmax = float(np.percentile(np.abs(frames_arr), 99.3))
    vmax = max(vmax, 1e-6)

    pil_frames = []
    for idx in range(frames_arr.shape[0]):
        fig, ax = plt.subplots(figsize=(8.6, 3.6), constrained_layout=True)
        ez = frames_arr[idx]
        im = ax.pcolormesh(
            x_mm,
            y_mm,
            ez.T,
            cmap="RdBu_r",
            shading="auto",
            vmin=-vmax,
            vmax=vmax,
        )
        for xc, wp, act in zip(
            geom_tap["patch_centers_x"],
            geom_tap["patch_widths_y"],
            geom_tap["active_mask"],
        ):
            if act:
                ax.add_patch(
                    plt.Rectangle(
                        ((xc - 0.5 * geom_tap["L1"]) * 1e3, -0.5 * wp * 1e3),
                        geom_tap["L1"] * 1e3,
                        wp * 1e3,
                        fill=False,
                        edgecolor="black",
                        lw=0.8,
                    )
                )
        fig.colorbar(im, ax=ax, label="Ez(x, y, z=h/2, t) [V/m]")
        ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel("Array axis x [mm]")
        ax.set_ylabel("Transverse y [mm]")
        ax.set_title(
            f"77 GHz Series-Fed Patch Array ('1011111111') — Pulse Propagation t = {times_ps[idx]:.1f} ps",
            fontsize=10,
            fontweight="bold",
        )
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_frames.append(Image.fromarray(rgba[:, :, :3]).quantize(colors=128))
        plt.close(fig)

    out_path = FIG_DIR / "04_radiated_field_evolution.gif"
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=110,
        loop=0,
        optimize=True,
    )
    return out_path


def _build_3d_lobe_mesh(theta_rad: np.ndarray, phi_rad: np.ndarray, gain_lin: np.ndarray, scale: float = 12.0, center=(12.5, 0.0, 0.25)):
    """Build a PyVista structured surface mesh of the 3D far-field radiation lobe."""
    import pyvista as pv

    # Wrap phi to close the surface
    phi_c = np.append(phi_rad, phi_rad[0] + 2.0 * np.pi)
    g_c = np.concatenate([gain_lin, gain_lin[:, :1]], axis=1)
    TH, PH = np.meshgrid(theta_rad, phi_c, indexing="ij")

    g_dbi = 10.0 * np.log10(np.maximum(g_c, 1e-3))
    r_norm = np.clip((g_dbi + 15.0) / (np.max(g_dbi) + 15.0), 0.05, 1.0) * scale

    X = center[0] + r_norm * np.sin(TH) * np.cos(PH)
    Y = center[1] + r_norm * np.sin(TH) * np.sin(PH)
    Z = np.maximum(center[2] + r_norm * np.cos(TH), center[2])
    grid = pv.StructuredGrid(X, Y, Z)
    grid["Gain [dBi]"] = g_dbi.ravel(order="F")
    return grid


def plot_05_far_field_3d_and_cuts(tap_npz: dict, uni_npz: dict, summary: dict) -> Path:
    """Generate `05_far_field_3d_and_cuts.png`: 3D fan-beam surface + E/H cuts vs Yang et al."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    lobe = _build_3d_lobe_mesh(
        tap_npz["theta_3d_rad"],
        tap_npz["phi_3d_rad"],
        tap_npz["gain_3d_77g_lin"],
        scale=12.0,
        center=(0.0, 0.0, 0.0),
    )
    pl = pv.Plotter(off_screen=True, window_size=(960, 680))
    pl.set_background("white")
    pl.add_mesh(
        lobe,
        scalars="Gain [dBi]",
        cmap="turbo",
        smooth_shading=True,
        scalar_bar_args={"title": "Gain [dBi]"},
    )
    pl.add_axes()
    pl.camera_position = [(22.0, -24.0, 18.0), (0.0, 0.0, 4.0), (0.0, 0.0, 1.0)]
    img_3d = pl.screenshot(return_img=True)
    pl.close()

    ang = tap_npz["ang_deg"]
    d_e_uni = uni_npz["d_e_dbi"] - np.max(uni_npz["d_e_dbi"])
    d_e_tap = tap_npz["d_e_dbi"] - np.max(tap_npz["d_e_dbi"])
    d_h_tap = tap_npz["d_h_dbi"] - np.max(tap_npz["d_h_dbi"])

    af_ideal = series_array_factor_e_plane(ang, topology="binary")
    sll_af_ideal = compute_sll_db(ang, af_ideal)

    best_key = (
        "gpu_superfine_32um"
        if "gpu_superfine_32um" in summary.get("runs", {})
        else "res_fine"
    )
    sll_uni = summary["runs"]["uniform_coarse"]["sll_e_db"]
    sll_tap = summary["runs"][best_key]["sll_e_db"]
    hpbw_tap = summary["runs"][best_key]["hpbw_e_half_deg"]

    exc_dict = summary["runs"]["tapered_coarse"].get("excitations_77ghz", {}).get("tm100_ez_slot")
    if exc_dict is not None:
        w_fit = np.asarray(exc_dict["amplitudes_norm"], dtype=float) * np.exp(
            1j * np.deg2rad(np.asarray(exc_dict["phases_deg"], dtype=float))
        )
        af_fit = series_array_factor_e_plane(ang, weights=w_fit)
        sll_af_fit = float(exc_dict["sll_af_times_ef_db"])
    else:
        af_fit = None
        sll_af_fit = np.nan

    fig = plt.figure(figsize=(15.5, 10.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # (a) 3D Fan-beam pattern
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_3d)
    ax0.axis("off")
    ax0.set_title(
        "(a) 3D Far-Field Gain Surface at 77 GHz (Fan Beam)",
        fontsize=10.0,
        fontweight="bold",
    )

    # (b) Rectangular E-plane comparison vs Yang et al. (2020) and Array Factor
    ax1 = fig.add_subplot(gs[0, 1])
    i_pk_uni = int(np.argmax(d_e_uni))
    th_pk_uni = float(ang[i_pk_uni])
    ax1.plot(
        ang,
        d_e_uni,
        color="#2563eb",
        lw=2.0,
        label=f"3D FIT Uniform '1111111111' (SLL = {sll_uni:.2f} dB, squint {th_pk_uni:+.1f}°)",
    )
    ax1.plot(
        ang,
        d_e_tap,
        color="#ea580c",
        lw=2.2,
        label=f"3D FIT Paper Binary '1011111111' (SLL = {sll_tap:.2f} dB)",
    )
    if af_fit is not None:
        ax1.plot(
            ang,
            af_fit,
            color="#7c3aed",
            ls="-.",
            lw=1.7,
            label=f"Extracted TM100 Excitations AF×EF (SLL = {sll_af_fit:.2f} dB)",
        )
    ax1.plot(
        ang,
        af_ideal,
        color="#16a34a",
        ls="--",
        lw=1.5,
        label=f"Ideal Equal-Amp '1011111111' AF×EF (SLL = {sll_af_ideal:.2f} dB)",
    )
    ax1.axhline(
        -10.0,
        color="#2563eb",
        ls=":",
        lw=1.2,
        label="Yang et al. Uniform Meas. SLL (-10.0 dB, Fig. 9a; Sim. -12.5 dB, Sec. 4)",
    )
    ax1.axhline(
        -14.8,
        color="#dc2626",
        ls="-.",
        lw=1.3,
        label="Yang et al. Binary '1011111111' Meas. SLL (-14.8 dB, Fig. 8a; Sim. -16.5 dB)",
    )
    ax1.annotate(
        f"Uniform squint {th_pk_uni:+.1f}°\n(f_broadside = 76.00 GHz)",
        xy=(th_pk_uni, 0.0),
        xytext=(th_pk_uni + 12.0, -3.5),
        fontsize=7.8,
        color="#1d4ed8",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#1d4ed8", "lw": 1.0},
    )
    ax1.set_xlim(-65.0, 65.0)
    ax1.set_ylim(-35.0, 2.2)
    ax1.set_xlabel("Elevation Angle θ in E-Plane (xz) [deg]")
    ax1.set_ylabel("Normalized Radiation Pattern [dB]")
    ax1.set_title(
        "(b) E-Plane (xz) Pattern, Extracted TM100 AF×EF & Sidelobe Comparison",
        fontsize=10.0,
        fontweight="bold",
    )
    ax1.legend(loc="lower center", fontsize=7.3, framealpha=0.92)
    ax1.grid(True, ls=":", alpha=0.5)

    # (c) Polar E-plane and H-plane cuts
    ax2 = fig.add_subplot(gs[1, 0], projection="polar")
    th_rad = np.deg2rad(ang)
    ax2.plot(
        th_rad,
        np.maximum(d_e_tap, -35.0),
        color="#ea580c",
        lw=2.0,
        label=f"E-Plane '1011111111' (±{hpbw_tap:.1f}° vs ±3.8° Yang Fig. 8a)",
    )
    ax2.plot(
        th_rad,
        np.maximum(d_e_uni, -35.0),
        color="#2563eb",
        ls="--",
        lw=1.4,
        label=f"E-Plane Uniform (±{summary['runs']['uniform_coarse']['hpbw_e_half_deg']:.1f}° vs ±3.7° Yang Fig. 9a)",
    )
    ax2.plot(
        th_rad,
        np.maximum(d_h_tap, -35.0),
        color="#0d9488",
        lw=1.8,
        label=f"H-Plane (yz, ±{summary['runs'][best_key]['hpbw_h_half_deg']:.1f}° vs ±36° sim / ±30° meas, Fig. 8b)",
    )
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_thetamin(-90)
    ax2.set_thetamax(90)
    ax2.set_rlim(-35.0, 0.0)
    ax2.set_title(
        "(c) Polar E-Plane (xz) & H-Plane (yz) Fan-Beam Cuts at 77 GHz",
        fontsize=10.0,
        fontweight="bold",
        pad=14,
    )
    ax2.legend(loc="lower left", bbox_to_anchor=(-0.08, -0.12), fontsize=8.0)

    # (d) Absolute Gain / Directivity comparison in E-plane
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(
        ang,
        tap_npz["d_e_dbi"],
        color="#f97316",
        ls="--",
        lw=1.6,
        label=f"3D FIT Directivity (PEC): Dmax = {np.max(tap_npz['d_e_dbi']):.2f} dBi",
    )
    ax3.plot(
        ang,
        tap_npz["g_e_dbi"],
        color="#ea580c",
        lw=2.2,
        label=f"3D FIT Gain (77 GHz Cu + dielectric loss): Gmax = {np.max(tap_npz['g_e_dbi']):.2f} dBi",
    )
    ax3.axhline(
        17.0,
        color="#0f766e",
        ls=":",
        lw=1.4,
        label="Yang et al. (2020) Simulated Gain (17.0 dBi, Sec. 4, p. 7)",
    )
    ax3.axhline(
        15.6,
        color="#15803d",
        ls="-.",
        lw=1.5,
        label="Yang et al. (2020) Measured Gain (15.6 dBi, Abstract & Sec. 4, p. 7)",
    )
    ax3.set_xlim(-60.0, 60.0)
    ax3.set_ylim(-15.0, 18.8)
    ax3.set_xlabel("Elevation Angle θ in E-Plane (xz) [deg]")
    ax3.set_ylabel("Directivity / Gain [dBi]")
    ax3.set_title(
        "(d) Absolute E-Plane Directivity vs Loss-Corrected Gain & Benchmark",
        fontsize=10.0,
        fontweight="bold",
    )
    ax3.legend(loc="lower center", fontsize=7.9, framealpha=0.92)
    ax3.grid(True, ls=":", alpha=0.5)

    out_path = FIG_DIR / "05_far_field_3d_and_cuts.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_06_combined_3d_overlays(tap_npz: dict) -> Path:
    """Generate `06_combined_3d_overlays.png`: 4-panel PyVista geometry + |Js| + 3D lobe + near field."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    _, geom_tap = build_array_structures("tapered")

    x_xy = tap_npz["xy_x_mm"]
    y_xy = tap_npz["xy_y_mm"]
    X_nf, Y_nf = np.meshgrid(x_xy, y_xy, indexing="ij")
    Z_nf = np.full_like(X_nf, 0.32)
    nf_grid = pv.StructuredGrid(X_nf, Y_nf, Z_nf)
    j_norm = tap_npz["j_mag_xy"] / max(np.max(tap_npz["j_mag_xy"]), 1e-12)
    e_db = 20.0 * np.log10(
        np.maximum(tap_npz["e_mag_xy"] / max(np.max(tap_npz["e_mag_xy"]), 1e-12), 1e-3)
    )
    nf_grid["|Js| (norm)"] = j_norm.ravel(order="F")
    nf_grid["|E| [dB]"] = e_db.ravel(order="F")

    lobe = _build_3d_lobe_mesh(
        tap_npz["theta_3d_rad"],
        tap_npz["phi_3d_rad"],
        tap_npz["gain_3d_77g_lin"],
        scale=11.0,
        center=(12.5, 0.0, 0.5),
    )

    views = [
        (
            "(a) Isometric Overlay: Geometry + Surface Current |Js| + 3D Fan-Beam Lobe",
            [(12.5, -24.0, 14.5), (12.5, 0.0, 3.5), (0.0, 0.0, 1.0)],
            True,
            "|Js| (norm)",
        ),
        (
            "(b) E-Plane Side View (-y): Narrow Elevation Beamwidth (±3.9°) over 10-Patch Array",
            [(12.5, -28.0, 4.5), (12.5, 0.0, 4.0), (0.0, 0.0, 1.0)],
            True,
            "|Js| (norm)",
        ),
        (
            "(c) H-Plane Front View (+x): Wide Azimuth Fan-Beam Pattern",
            [(36.0, 0.0, 6.0), (12.5, 0.0, 4.0), (0.0, 0.0, 1.0)],
            True,
            "|Js| (norm)",
        ),
        (
            "(d) Geometry + Near-Field |E| Slice (dB) on Patch Plane z = h",
            [(12.5, -19.0, 12.0), (12.5, 0.0, 0.3), (0.0, 0.0, 1.0)],
            False,
            "|E| [dB]",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), constrained_layout=True)
    for ax, (title, cam, show_lobe, scalar_name) in zip(axes.ravel(), views):
        pl = pv.Plotter(off_screen=True, window_size=(960, 620))
        pl.set_background("white")
        _add_array_scene_meshes(pl, geom_tap, z_explode=1.8)
        sbar_nf = (
            {
                "title": scalar_name,
                "position_x": 0.05,
                "position_y": 0.05,
                "width": 0.38,
                "height": 0.06,
                "title_font_size": 14,
                "label_font_size": 12,
            }
            if show_lobe
            else {
                "title": scalar_name,
                "position_x": 0.25,
                "position_y": 0.05,
                "width": 0.50,
                "height": 0.06,
                "title_font_size": 14,
                "label_font_size": 12,
            }
        )
        pl.add_mesh(
            nf_grid,
            scalars=scalar_name,
            cmap="inferno" if "E" in scalar_name else "magma",
            opacity=0.82 if not show_lobe else 0.72,
            show_scalar_bar=True,
            scalar_bar_args=sbar_nf,
        )
        if show_lobe:
            pl.add_mesh(
                lobe,
                scalars="Gain [dBi]",
                cmap="turbo",
                opacity=0.62,
                smooth_shading=True,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Gain [dBi]",
                    "position_x": 0.55,
                    "position_y": 0.05,
                    "width": 0.38,
                    "height": 0.06,
                    "title_font_size": 14,
                    "label_font_size": 12,
                },
            )
        pl.camera_position = cam
        img = pl.screenshot(return_img=True)
        pl.close()
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=10.0, fontweight="bold")

    out_path = FIG_DIR / "06_combined_3d_overlays.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _draw_smith_grid(ax) -> None:
    """Draw constant-resistance and constant-reactance circles on a unit Smith chart."""
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(np.cos(theta), np.sin(theta), color="#1e293b", lw=1.3)
    ax.axhline(0.0, color="#64748b", lw=0.8)
    for r in (0.2, 0.5, 1.0, 2.0, 5.0):
        cx = r / (r + 1.0)
        rad = 1.0 / (r + 1.0)
        ax.plot(cx + rad * np.cos(theta), rad * np.sin(theta), color="#cbd5e1", lw=0.75, ls="--")
    for x_val in (0.2, 0.5, 1.0, 2.0, -0.2, -0.5, -1.0, -2.0):
        cx = 1.0
        cy = 1.0 / x_val
        rad = abs(1.0 / x_val)
        ang = np.linspace(0.0, 2.0 * np.pi, 721)
        xp = cx + rad * np.cos(ang)
        yp = cy + rad * np.sin(ang)
        mask = xp**2 + yp**2 <= 1.001
        xp = np.where(mask, xp, np.nan)
        yp = np.where(mask, yp, np.nan)
        ax.plot(xp, yp, color="#cbd5e1", lw=0.75, ls=":")
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.axis("off")


def plot_07_s11_impedance_smith(tap_npz: dict, uni_npz: dict) -> Path:
    """Generate `07_s11_impedance_smith.png`: |S11| (dB), Re/Im(Zin), and Smith chart."""
    f_ghz = tap_npz["freqs_hz"] / 1e9
    i_77 = int(np.argmin(np.abs(f_ghz - 77.0)))

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), constrained_layout=True)

    # (a) |S11| in dB
    ax = axes[0]
    ax.axvspan(76.0, 77.0, color="#bbf7d0", alpha=0.45, label="76–77 GHz Automotive LRR Band")
    ax.plot(f_ghz, uni_npz["s11_db"], color="#2563eb", lw=2.0, label="Uniform '1111111111'")
    ax.plot(f_ghz, tap_npz["s11_db"], color="#ea580c", lw=2.2, label="Paper Binary '1011111111'")
    ax.axhline(
        -10.0,
        color="#dc2626",
        ls="--",
        lw=1.2,
        label="|S11| = -10 dB Reference (S11 not in Yang et al.)",
    )
    ax.axvline(77.0, color="#475569", ls=":", lw=1.1)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("|S11| [dB]")
    ax.set_title("(a) Return Loss |S11| Across 72–80 GHz", fontsize=10.2, fontweight="bold")
    ax.set_ylim(-30.0, 0.5)
    ax.legend(loc="lower right", fontsize=7.9, framealpha=0.92)
    ax.grid(True, ls=":", alpha=0.5)

    # (b) Input Impedance Re(Zin), Im(Zin)
    ax = axes[1]
    ax.plot(f_ghz, tap_npz["zin_real"], color="#ea580c", lw=2.0, label="Re(Zin) — '1011111111'")
    ax.plot(f_ghz, tap_npz["zin_imag"], color="#ea580c", ls="--", lw=1.8, label="Im(Zin) — '1011111111'")
    ax.plot(f_ghz, uni_npz["zin_real"], color="#2563eb", lw=1.5, alpha=0.8, label="Re(Zin) — '1111111111'")
    ax.axhline(50.0, color="#16a34a", ls=":", lw=1.2, label="50 Ω Reference")
    ax.axhline(0.0, color="#64748b", ls=":", lw=1.0)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Input Impedance [Ω]")
    ax.set_title("(b) Input Impedance Re(Zin) & Im(Zin)", fontsize=10.2, fontweight="bold")
    y_min = min(-110.0, float(np.min(tap_npz["zin_imag"])) - 15.0)
    y_max = max(180.0, float(np.max(tap_npz["zin_real"])) + 20.0)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper left", fontsize=7.9, ncol=2, framealpha=0.92)
    ax.grid(True, ls=":", alpha=0.5)

    # (c) Smith Chart
    ax = axes[2]
    _draw_smith_grid(ax)
    s11_u = uni_npz["s11_real"] + 1j * uni_npz["s11_imag"]
    s11_t = tap_npz["s11_real"] + 1j * tap_npz["s11_imag"]
    ax.plot(np.real(s11_u), np.imag(s11_u), color="#2563eb", lw=1.8, label="Uniform (72–80 GHz)")
    ax.plot(np.real(s11_t), np.imag(s11_t), color="#ea580c", lw=2.1, label="Binary '1011111111' (72–80 GHz)")
    ax.plot(
        np.real(s11_t[i_77]),
        np.imag(s11_t[i_77]),
        marker="o",
        ms=8,
        color="#dc2626",
        label="77.0 GHz",
    )
    ax.set_title("(c) Smith Chart Trajectory (Z0 = 50 Ω, 72–80 GHz)", fontsize=10.2, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8.2, frameon=True)

    out_path = FIG_DIR / "07_s11_impedance_smith.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_08_resolution_and_loss_study(
    tap_c: dict, tap_m: dict, tap_f: dict, summary: dict, tap_sf: dict | None = None
) -> Path:
    """Generate `08_resolution_and_loss_study.png`: 4-level grid convergence & 77 GHz loss budget."""
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.2), constrained_layout=True)

    pf = summary.get("preflight", {})
    nc_k = round(pf.get("coarse", {}).get("n_cells", 156780) / 1000)
    nm_k = round(pf.get("medium", {}).get("n_cells", 281996) / 1000)
    nf_k = round(pf.get("fine", {}).get("n_cells", 433800) / 1000)
    nsf_m = pf.get("gpu_superfine_32um", {}).get("n_cells", 1388184) / 1e6

    # (a) E-plane pattern across coarse, medium, fine, superfine meshes
    ax = axes[0]
    curves = [
        (tap_c, f"coarse ({nc_k}k, nz=2)", "#64748b", "--"),
        (tap_m, f"medium ({nm_k}k, nz=3)", "#2563eb", "-."),
        (tap_f, f"fine ({nf_k}k, nz=4, ≥4 cells/feat)", "#ea580c", "-"),
    ]
    if tap_sf is not None:
        curves.append((tap_sf, f"superfine ({nsf_m:.2f}M, nz=8, Δz=32 µm)", "#0f766e", ":"))
    for npz_d, lbl, col, ls in curves:
        ang = npz_d["ang_deg"]
        pat = npz_d["d_e_dbi"] - np.max(npz_d["d_e_dbi"])
        ax.plot(ang, pat, color=col, ls=ls, lw=1.9, label=lbl)
    ax.axhline(
        -14.8,
        color="#dc2626",
        ls=":",
        lw=1.3,
        label="Yang et al. Meas. SLL (-14.8 dB, Fig. 8a)",
    )
    ax.set_xlim(-55.0, 55.0)
    ax.set_ylim(-35.0, 1.5)
    ax.set_xlabel("Elevation Angle θ in E-Plane [deg]")
    ax.set_ylabel("Normalized E-Plane Pattern [dB]")
    ax.set_title("(a) 4-Level Resolution Study: E-Plane Pattern", fontsize=10.2, fontweight="bold")
    ax.legend(loc="lower center", fontsize=7.6, framealpha=0.92)
    ax.grid(True, ls=":", alpha=0.5)

    # (b) Gain & SLL convergence vs cell count
    ax = axes[1]
    run_keys = ["tapered_coarse", "res_medium", "res_fine"]
    labels = [f"coarse\n({nc_k}k)", f"medium\n({nm_k}k)", f"fine\n({nf_k}k)"]
    if "gpu_superfine_32um" in summary.get("runs", {}):
        run_keys.append("gpu_superfine_32um")
        labels.append(f"superfine\n({nsf_m:.2f}M)")
    g_vals = [summary["runs"][k]["g_max_lossy_dbi"] for k in run_keys]
    d_vals = [summary["runs"][k]["d_max_pec_dbi"] for k in run_keys]
    sll_vals = [-summary["runs"][k]["sll_e_db"] for k in run_keys]

    x_idx = np.arange(len(run_keys))
    ax.plot(x_idx, d_vals, "o--", color="#f97316", lw=1.8, label="PEC Directivity Dmax [dBi]")
    ax.plot(x_idx, g_vals, "s-", color="#ea580c", lw=2.2, label="Loss-Corrected Gain Gmax [dBi]")
    ax.plot(x_idx, sll_vals, "^-", color="#2563eb", lw=2.0, label="Sidelobe Suppression -SLL [dB]")
    ax.axhline(15.6, color="#15803d", ls=":", lw=1.4, label="Yang et al. Meas. Gain (15.6 dBi, Sec. 4)")
    ax.axhline(14.8, color="#1d4ed8", ls="-.", lw=1.3, label="Yang et al. Meas. -SLL (14.8 dB, Fig. 8a)")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(9.0, 17.2)
    ax.set_ylabel("[dBi] or [dB]")
    ax.set_title("(b) Gain & SLL Convergence Across 4 Mesh Levels", fontsize=10.2, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7.4, framealpha=0.92)
    ax.grid(True, ls=":", alpha=0.5)

    # (c) 77 GHz Conductor & Dielectric Loss Breakdown
    ax = axes[2]
    lb = summary["loss_budget_77ghz"]
    best_key = "gpu_superfine_32um" if "gpu_superfine_32um" in summary.get("runs", {}) else "res_fine"
    d_pec = summary["runs"][best_key]["d_max_pec_dbi"]
    loss_cond_smooth = lb["conductor_loss_db"] / lb["roughness_factor"]

    stages = [
        "PEC FIT\nDirectivity",
        "After Smooth\nCu Skin Loss",
        "After Cu\nRoughness",
        "After Rogers\n5880 + WR-12",
        "Yang et al.\nMeas. (Sec. 4)",
    ]
    vals = [
        d_pec,
        d_pec - loss_cond_smooth,
        d_pec - lb["conductor_loss_db"],
        d_pec - lb["total_loss_db"],
        15.6,
    ]
    colors = ["#f97316", "#fb923c", "#ea580c", "#c2410c", "#16a34a"]
    bars = ax.bar(np.arange(len(stages)), vals, color=colors, width=0.58)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + 0.5 * b.get_width(),
            v + 0.12,
            f"{v:.2f} dBi",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
        )
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stages, fontsize=8.0)
    ax.set_ylim(13.0, 17.5)
    ax.set_ylabel("Peak Broadside Gain [dBi]")
    ax.set_title(
        f"(c) 77 GHz Loss Budget (δs = {lb['skin_depth_um']:.3f} µm, ΔG = -{lb['total_loss_db']:.2f} dB)",
        fontsize=10.2,
        fontweight="bold",
    )
    ax.grid(True, axis="y", ls=":", alpha=0.5)

    out_path = FIG_DIR / "08_resolution_and_loss_study.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def generate_all_figures() -> list[Path]:
    """Generate all 8 publication-quality figures (including the < 5 MB GIF) from `data/`."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DATA_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["geometry_um"] = YANG2020_GEOMETRY_UM
    summary["benchmarks"] = YANG2020_BENCHMARKS
    summary["source_locations"] = YANG2020_SOURCE_LOCATIONS
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    uni_c = dict(np.load(DATA_DIR / "uniform_coarse.npz"))
    tap_c = dict(np.load(DATA_DIR / "tapered_coarse.npz"))
    tap_m = (
        dict(np.load(DATA_DIR / "res_medium.npz"))
        if (DATA_DIR / "res_medium.npz").exists()
        else tap_c
    )
    tap_f = (
        dict(np.load(DATA_DIR / "res_fine.npz"))
        if (DATA_DIR / "res_fine.npz").exists()
        else tap_m
    )
    tap_sf = (
        dict(np.load(DATA_DIR / "gpu_superfine_32um.npz"))
        if (DATA_DIR / "gpu_superfine_32um.npz").exists()
        else None
    )
    anim_d = dict(np.load(DATA_DIR / "tapered_animation.npz"))

    paths = [
        plot_01_geometry_3d_views(),
        plot_02_mesh_and_cpw_views(summary["preflight"]),
        plot_03_near_fields_and_currents(tap_c, uni_c),
        plot_04_radiated_field_gif(anim_d),
        plot_05_far_field_3d_and_cuts(tap_sf if tap_sf is not None else tap_f, uni_c, summary),
        plot_06_combined_3d_overlays(tap_c),
        plot_07_s11_impedance_smith(tap_sf if tap_sf is not None else tap_f, uni_c),
        plot_08_resolution_and_loss_study(tap_c, tap_m, tap_f, summary, tap_sf=tap_sf),
    ]
    return paths
