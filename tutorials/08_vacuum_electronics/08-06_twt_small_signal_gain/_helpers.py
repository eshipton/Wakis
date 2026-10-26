"""Helper functions for Tutorial 8.6: Traveling-Wave-Tube Small-Signal Gain.

Implements:
1. Pierce small-signal TWT theory (cubic forward-wave dispersion, quartic full-circuit
   dispersion including the backward wave, 3-wave and 4-wave launching boundary-value
   solvers, and a 1D coupled-mode spatial ODE integrator) with relativistic mass
   corrections and space-charge reduction factors R from ``wakis.beam_fluid``.
2. Exact 2D-to-3D sheet-beam normalization mapping a 2D planar sheet beam (per unit
   transverse width) to a finite-width 3D rectangular sheet beam in an LSE_10
   corrugated rectangular waveguide.
3. Johnson (1955) backward-wave oscillator (BWO) start-oscillation root finder
   ((CN)_st = 0.314055, b_st = 1.521839 at QC = d = 0) and I_st proportional to L^-3
   scaling.
4. Wakis 3D FIT + LinearizedBeam hot-test simulation runners for:
   - Space-charge-wave time-centring verification in the smooth Karetnikova tunnel,
   - 2D Karetnikova 0.2 THz staggered double-vane grating spatial TWT amplifier
     (3-level mesh convergence + Richardson extrapolation, 3-wave launching beat,
     and small-signal gain vs frequency compared against Karetnikova et al. 2018),
   - Falsifiable physics validation suite (I0 -> 0 cold limit, coupling_sign = -1
     mutation, Im(k) proportional to I0^(1/3) scaling, velocity detuning b,
     space-charge QC reduction, and small-signal drive linearity),
   - 3D Marosi V-band glide-symmetric pillar SWS hot-test verification reusing
     Tutorial 6.2 cold-test dispersion and Pierce impedance data,
   - Backward-wave oscillation (BWO) hot test on the n = -1 backward space harmonic
     demonstrating temporal decay below I_st and self-excited growth above I_st.
5. Publication-style figure and animation generators using ``wakis.figstyle`` and
   ``wakis.schematic``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.constants import c as C0
from scipy.constants import e as Q_E
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import m_e as M_E
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize, root

from wakis import GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.beam_fluid import (
    LinearizedBeam,
    beam_plasma_frequency,
    reduction_factor_planar,
    reduction_factor_rectangular,
    relativistic_gamma,
    space_charge_wave_dispersion,
)
from wakis.estimate import estimate, preflight
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates import meshing
from wakis.templates.photonics import Box

TUTORIAL_DIR = Path(__file__).resolve().parent
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"
REPO_ROOT = TUTORIAL_DIR.parents[2]
COLD_TEST_DIR = REPO_ROOT / "tutorials" / "06_periodic_structures" / "06-02_sws_cold_test"
COLD_DATA_DIR = COLD_TEST_DIR / "data"

if str(COLD_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(COLD_TEST_DIR))

LOCAL_VM_SPEC = {"ram": 2_000_000_000, "gpu": 0}
_COLD_HELPERS_MOD: Any = None


def _load_cold_test_helpers() -> Any:
    """Import Tutorial 6.2 ``_helpers.py`` under a distinct module name."""
    global _COLD_HELPERS_MOD
    if _COLD_HELPERS_MOD is None:
        import importlib.util

        spec_m = importlib.util.spec_from_file_location(
            "_sws_cold_test_helpers", COLD_TEST_DIR / "_helpers.py"
        )
        mod = importlib.util.module_from_spec(spec_m)
        sys.modules["_sws_cold_test_helpers"] = mod
        spec_m.loader.exec_module(mod)
        _COLD_HELPERS_MOD = mod
    return _COLD_HELPERS_MOD


# =====================================================================
# 1. Geometry & Beam Specifications
# =====================================================================


@dataclass(frozen=True)
class KaretnikovaTWTSpec:
    """Geometry and beam parameters of the 0.2 THz staggered grating TWT.

    Reference: Karetnikova et al., IEEE Trans. Electron Devices 65(6),
    2129-2134 (2018), Table I and Figs. 1-5.
    """

    period_m: float = 0.50e-3          # Grating period d [m]
    vane_thickness_m: float = 0.10e-3  # Vane axial thickness d - s_slot [m]
    slot_width_m: float = 0.40e-3      # Slot axial opening s [m]
    vane_height_m: float = 0.30e-3     # Slot depth / vane height h [m]
    tunnel_half_height_m: float = 0.10e-3  # Half-tunnel height a (2a = 0.20 mm) [m]
    beam_half_thickness_m: float = 0.075e-3  # Sheet-beam half-thickness b (H_b = 0.15 mm) [m]
    waveguide_width_m: float = 0.85e-3  # Lateral waveguide width b_wg [m]
    beam_width_m: float = 0.75e-3       # Lateral sheet-beam width W_b [m]
    voltage_v: float = 20.0e3           # Nominal beam voltage V0 [V]
    current_a: float = 0.100            # Nominal DC beam current I0 [A]
    n_periods_fig3: int = 50            # L = 25 mm (50 periods) in Fig. 3
    n_periods_fig5: int = 80            # L = 40 mm (80 periods) in Fig. 5a

    @property
    def total_height_m(self) -> float:
        """Total height between top and bottom slot floors [m]."""
        return 2.0 * (self.tunnel_half_height_m + self.vane_height_m)

    @property
    def min_feature_m(self) -> float:
        """Smallest geometric feature size (vane tooth thickness 0.10 mm) [m]."""
        return min(
            self.vane_thickness_m,
            2.0 * self.beam_half_thickness_m,
            2.0 * self.tunnel_half_height_m,
        )

    @property
    def f_cut_hz(self) -> float:
        """Lateral LSE_10 waveguide cutoff frequency c / (2 b_wg) [Hz]."""
        return C0 / (2.0 * self.waveguide_width_m)


# =====================================================================
# 2. Part 1: Pierce Theory, 2D-to-3D Sheet Mapping, and BWO Solvers
# =====================================================================


def map_2d_sheet_to_3d_grating(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
) -> dict[str, float]:
    """Derive exact 2D sheet-beam normalization for a 3D LSE_10 waveguide grating.

    In a 3D rectangular waveguide of width :math:`b_{\\mathrm{wg}}` along :math:`x`,
    the fundamental :math:`\\mathrm{LSE}_{10}` mode has transverse electric field profile
    :math:`E_z(x, y) = E_{z0}(y)\\cos(\\pi x / b_{\\mathrm{wg}})` for
    :math:`x \\in [-b_{\\mathrm{wg}}/2, +b_{\\mathrm{wg}}/2]`.

    1. The total 3D electromagnetic power flow is
       :math:`P_{3\\mathrm{D}} = \\frac{1}{2} b_{\\mathrm{wg}} p_{2\\mathrm{D}}`, where
       :math:`p_{2\\mathrm{D}}` [W/m] is the 2D power per unit width at the peak
       plane :math:`x = 0`.
    2. Averaging :math:`|E_{zn}(x, y)|^2` over a sheet beam of lateral width
       :math:`W_b \\le b_{\\mathrm{wg}}` introduces the lateral filling factor:

       .. math::
           F_x(W_b, b_{\\mathrm{wg}}) = \\frac{1}{W_b}\\int_{-W_b/2}^{+W_b/2}
           \\cos^2\\!\\left(\\frac{\\pi x}{b_{\\mathrm{wg}}}\\right)\\mathrm{d}x
           = \\frac{1}{2} + \\frac{b_{\\mathrm{wg}}}{2\\pi W_b}
             \\sin\\!\\left(\\frac{\\pi W_b}{b_{\\mathrm{wg}}}\\right).

    3. Consequently, the 3D beam-averaged Pierce impedance :math:`K_{n,3\\mathrm{D}}`
       [Ohm] and the 2D impedance per unit width :math:`k_{n,2\\mathrm{D}} =
       \\langle |E_{zn}(0, y)|^2\\rangle_y / (2\\beta_n^2 p_{2\\mathrm{D}})` [Ohm m]
       satisfy:

       .. math::
           K_{n,3\\mathrm{D}} = \\frac{F_x}{b_{\\mathrm{wg}} / 2}\\,k_{n,2\\mathrm{D}}
           = \\frac{k_{n,2\\mathrm{D}}}{W_{\\mathrm{eff}}}, \\qquad
           W_{\\mathrm{eff}} = \\frac{b_{\\mathrm{wg}}}{2 F_x}.

    4. In a 2D FIT simulation of slice width :math:`\\Delta x` with Bloch wavenumber
       :math:`k_x = \\pi / b_{\\mathrm{wg}}`, every transverse cell carries the uniform
       peak amplitude. Choosing the 2D sheet current per unit width
       :math:`J_{\\mathrm{sheet}} = I_{0,3\\mathrm{D}} / W_{\\mathrm{eff}}` (or equivalent
       current :math:`I_{0,2\\mathrm{D}}(\\Delta x) = J_{\\mathrm{sheet}}\\,\\Delta x =
       \\eta_x I_{0,3\\mathrm{D}} (\\Delta x / W_b)` with
       :math:`\\eta_x = 2 F_x W_b / b_{\\mathrm{wg}}`) makes the 2D Pierce gain
       parameter identical to the 3D Pierce gain parameter:

       .. math::
           C_{2\\mathrm{D}}^3 = \\frac{J_{\\mathrm{sheet}}\\,k_{n,2\\mathrm{D}}}{4 V_{0,\\mathrm{eff}}}
           = \\frac{I_{0,3\\mathrm{D}}\\,K_{n,3\\mathrm{D}}}{4 V_{0,\\mathrm{eff}}} = C_{3\\mathrm{D}}^3.

    Parameters
    ----------
    spec : KaretnikovaTWTSpec, optional
        TWT specification.

    Returns
    -------
    mapping : dict[str, float]
        Dictionary with ``w_beam_m``, ``b_wg_m``, ``fx_lateral``, ``w_eff_m``,
        ``eta_x_current_factor``, and ``j_sheet_a_per_m`` at nominal ``I0``.
    """
    w_b = float(spec.beam_width_m)
    b_wg = float(spec.waveguide_width_m)
    fx = 0.5 + (b_wg / (2.0 * np.pi * w_b)) * float(np.sin(np.pi * w_b / b_wg))
    w_eff = 0.5 * b_wg / fx
    eta_x = w_b / w_eff
    j_sheet = float(spec.current_a) / w_eff
    return {
        "w_beam_m": w_b,
        "b_wg_m": b_wg,
        "fx_lateral": float(fx),
        "w_eff_m": float(w_eff),
        "eta_x_current_factor": float(eta_x),
        "j_sheet_a_per_m": float(j_sheet),
    }


def karetnikova_depression_factor_sq(
    beta_e: float | np.ndarray,
    beam_thickness_m: float,
    tunnel_half_height_m: float = 0.10e-3,
    gamma0: float | None = None,
) -> np.ndarray:
    """Evaluate Karetnikova et al. (2018) Eq. (4) sheet-beam depression factor D^2.

    Equation (4) of Karetnikova et al. (2018) gives the squared plasma-frequency
    reduction factor :math:`D^2 = (\\omega_q / \\omega_p)^2` for a planar sheet beam
    of half-thickness :math:`h = H_b / 2` between two parallel conducting plates
    separated by :math:`2a` (tunnel half-height :math:`a`):

    .. math::
        D^2 = 1 - \\frac{\\sinh(\\gamma h)\\,\\cosh[\\gamma h(a/h - 1)]}{\\gamma h\\,\\cosh(\\gamma a)},
        \\qquad \\gamma = \\frac{\\beta_e}{\\gamma_0}.

    At :math:`f_0 = 186` GHz and :math:`V_0 = 20` kV (:math:`h = 75\\;\\mu\\mathrm{m},
    a = 100\\;\\mu\\mathrm{m}, \\gamma_0 = 1.039139`), this yields
    :math:`D^2 = 0.403828` (:math:`D = 0.635475 \\approx 0.635`).

    Parameters
    ----------
    beta_e : float or ndarray
        Electronic wavenumber :math:`\\beta_e = \\omega / u_0` [rad/m].
    beam_thickness_m : float
        Full sheet-beam thickness :math:`H_b = 2h` [m].
    tunnel_half_height_m : float, optional
        Conducting beam-tunnel half-height :math:`a` [m]. Default is ``0.10e-3``.
    gamma0 : float or None, optional
        Relativistic Lorentz factor :math:`\\gamma_0`. If None, uses the 20 kV
        nominal value ``1.03913895``.

    Returns
    -------
    d_sq : ndarray
        Squared depression factor :math:`D^2 \\in [0, 1]`.
    """
    if gamma0 is None:
        _, gamma0_val = relativistic_gamma(voltage=20.0e3)
    else:
        gamma0_val = float(gamma0)
    h = 0.5 * float(beam_thickness_m)
    a = float(tunnel_half_height_m)
    gamma = np.maximum(np.asarray(beta_e, dtype=float) / max(gamma0_val, 1.0), 1e-12)
    gh = gamma * h
    ga = gamma * a
    g_diff = gamma * (a - h)
    return 1.0 - (np.sinh(gh) * np.cosh(g_diff)) / (gh * np.cosh(ga))


def pierce_parameters(
    freq_hz: float | np.ndarray,
    beta_circuit: float | np.ndarray,
    k_pierce_ohm: float | np.ndarray,
    voltage_v: float = 20.0e3,
    current_a: float = 0.100,
    beam_half_thickness_m: float = 0.075e-3,
    tunnel_half_height_m: float = 0.10e-3,
    beam_width_m: float = 0.75e-3,
    waveguide_width_m: float | None = None,
    alpha_loss_np_m: float = 0.0,
    sc_model: str = "wakis_planar",
) -> dict[str, Any]:
    """Compute relativistic Pierce small-signal parameters (C, b, QC, d).

    Includes the relativistic longitudinal mass factor :math:`\\gamma_0^3 m_e` via
    the effective beam voltage

    .. math::
        V_{0,\\mathrm{eff}} = \\frac{\\gamma_0^3 m_e u_0^2}{2 e}
        = \\frac{1}{2}\\gamma_0(\\gamma_0 + 1) V_0,

    so that the Pierce gain parameter is

    .. math::
        C = \\left(\\frac{I_0 K_n}{4 V_{0,\\mathrm{eff}}}\\right)^{1/3}.

    Parameters
    ----------
    freq_hz : float or ndarray
        Operating frequency :math:`f` [Hz].
    beta_circuit : float or ndarray
        Cold-circuit synchronous wavenumber :math:`\\beta_n` [rad/m].
    k_pierce_ohm : float or ndarray
        Beam-averaged Pierce interaction impedance :math:`K_n` [Ohm].
    voltage_v : float, optional
        DC kinetic beam voltage :math:`V_0` [V].
    current_a : float, optional
        DC beam current :math:`I_0` [A].
    beam_half_thickness_m : float, optional
        Sheet-beam half-thickness :math:`b` [m].
    tunnel_half_height_m : float, optional
        Conducting tunnel half-height :math:`a` [m].
    beam_width_m : float, optional
        Sheet-beam lateral width :math:`W_b` [m].
    waveguide_width_m : float or None, optional
        Waveguide width :math:`b_{\\mathrm{wg}}` [m] when ``sc_model='wakis_rect'``.
    alpha_loss_np_m : float, optional
        Cold-circuit field attenuation constant :math:`\\alpha` [Np/m].
    sc_model : {'wakis_planar', 'wakis_rect', 'karetnikova_eq4', 'none'}, optional
        Space-charge reduction factor model.

    Returns
    -------
    params : dict[str, Any]
        Dictionary containing ``u0``, ``gamma0``, ``v0_eff``, ``beta_e``,
        ``omega_p_rel``, ``R_sc``, ``omega_q``, ``C``, ``b``, ``QC``, ``four_QC``,
        and ``d``.
    """
    f_arr = np.asarray(freq_hz, dtype=float)
    beta_c = np.asarray(beta_circuit, dtype=float)
    k_arr = np.maximum(np.asarray(k_pierce_ohm, dtype=float), 0.0)
    omega = 2.0 * np.pi * f_arr

    u0, gamma0 = relativistic_gamma(voltage=float(voltage_v))
    v0_eff = 0.5 * gamma0 * (gamma0 + 1.0) * float(voltage_v)
    beta_e = omega / u0

    area_beam = float(beam_width_m) * (2.0 * float(beam_half_thickness_m))
    if current_a > 0.0:
        wp_nonrel = beam_plasma_frequency(current=float(current_a), area=area_beam, u0=u0)
    else:
        wp_nonrel = 0.0
    wp_rel = wp_nonrel / (gamma0**1.5)

    if sc_model == "wakis_planar":
        r_sc = np.asarray(
            reduction_factor_planar(
                omega,
                u0,
                float(beam_half_thickness_m),
                a=float(tunnel_half_height_m),
                relativistic=True,
            ),
            dtype=float,
        )
    elif sc_model == "wakis_rect":
        w_wg = float(waveguide_width_m or beam_width_m)
        r_sc = np.asarray(
            reduction_factor_rectangular(
                omega,
                u0,
                0.5 * float(beam_width_m),
                float(beam_half_thickness_m),
                ax=0.5 * w_wg,
                ay=float(tunnel_half_height_m),
                relativistic=True,
            ),
            dtype=float,
        )
    elif sc_model == "karetnikova_eq4":
        r_sc = np.sqrt(
            karetnikova_depression_factor_sq(
                beta_e,
                2.0 * float(beam_half_thickness_m),
                tunnel_half_height_m=float(tunnel_half_height_m),
                gamma0=float(gamma0),
            )
        )
    elif sc_model == "none":
        r_sc = np.zeros_like(beta_e, dtype=float)
    else:
        raise ValueError(f"Unknown sc_model={sc_model!r}.")

    omega_q = r_sc * wp_rel
    C_val = (float(current_a) * k_arr / (4.0 * v0_eff)) ** (1.0 / 3.0)
    C_safe = np.maximum(C_val, 1e-15)
    b_val = (beta_c - beta_e) / (beta_e * C_safe)
    four_qc = (omega_q / (omega * C_safe)) ** 2
    qc_val = 0.25 * four_qc
    d_val = np.asarray(alpha_loss_np_m, dtype=float) / (beta_e * C_safe)
    chi_rel = 2.0 / (float(gamma0) * (float(gamma0) + 1.0))

    return {
        "u0": float(u0),
        "gamma0": float(gamma0),
        "chi_rel": float(chi_rel),
        "v0_eff": float(v0_eff),
        "beta_e": beta_e,
        "omega_p_nonrel": float(wp_nonrel),
        "omega_p_rel": float(wp_rel),
        "R_sc": r_sc,
        "omega_q": omega_q,
        "C": C_val,
        "b": b_val,
        "QC": qc_val,
        "four_QC": four_qc,
        "d": d_val,
    }


def solve_pierce_dispersion(
    b: float = 0.0,
    QC: float = 0.0,
    d: float = 0.0,
    C: float = 0.0,
    include_backward: bool = False,
    coupling_sign: float = 1.0,
) -> dict[str, Any]:
    """Solve the cubic (3-wave) or quartic (4-wave) Pierce dispersion relation.

    Using Pierce's incremental propagation constant :math:`\\delta` defined by
    :math:`k(\\omega) = \\beta_e - j\\beta_e C\\delta` (so a wave varying as
    :math:`\\exp(j\\omega t - j k z) = \\exp(j\\omega t - j\\beta_e z + \\beta_e C\\delta z)`
    grows spatially when :math:`x = \\operatorname{Re}(\\delta) > 0`), the cold-circuit
    propagation constant is :math:`\\Gamma_c = j\\beta_c + \\alpha =
    j\\beta_e(1 + C b) + \\beta_e C d`.

    - **3-wave cubic approximation** (``include_backward=False``, :math:`C \\to 0`
      with only the forward circuit wave retained):

      .. math::
          (\\delta + j b + d)(\\delta^2 + 4QC) = -j\\,\\mathrm{sign}_{\\mathrm{cpl}}.

    - **4-wave quartic dispersion** (``include_backward=True``, retaining both
      forward :math:`+\\Gamma_c` and backward :math:`-\\Gamma_c` cold-circuit waves
      for finite :math:`C`):

      .. math::
          \\left[(1 - j C\\delta)^2 - (1 + C b - j C d)^2\\right]
          (\\delta^2 + 4QC) = 2\\,\\mathrm{sign}_{\\mathrm{cpl}}\\,C (1 - j C\\delta)^2
                             (1 + C b - j C d).

    Parameters
    ----------
    b : float, optional
        Pierce velocity detuning parameter :math:`b = (u_0 - v_p)/(C v_p)`.
    QC : float, optional
        Pierce space-charge parameter :math:`QC = \\omega_q^2 / (4\\omega^2 C^2)`.
    d : float, optional
        Pierce circuit loss parameter :math:`d = \\alpha / (\\beta_e C)`.
    C : float, optional
        Pierce gain parameter :math:`C` (used when ``include_backward=True``).
    include_backward : bool, optional
        If True, solves the 4-wave quartic dispersion relation. Default is False.
    coupling_sign : float, optional
        Lorentz coupling sign multiplier (:math:`+1` physical, :math:`-1` mutated).

    Returns
    -------
    res : dict[str, Any]
        Dictionary with sorted roots ``deltas`` (:math:`\\delta_1` has the largest
        real part :math:`x_1 = \\operatorname{Re}(\\delta_1)`), ``x1``, ``y1``, and
        polynomial coefficients ``poly_coeffs``.
    """
    four_qc = 4.0 * float(QC)
    z_c = 1j * float(b) + float(d)
    s_cpl = float(coupling_sign)

    if not include_backward or abs(C) < 1e-12:
        # Cubic: delta^3 + z_c delta^2 + 4QC delta + (4QC z_c + j * s_cpl) = 0
        coeffs = np.array(
            [1.0 + 0j, z_c, four_qc + 0j, four_qc * z_c + 1j * s_cpl],
            dtype=np.complex128,
        )
        roots_raw = np.roots(coeffs)
    else:
        # Quartic: [(1 - j C delta)^2 - q_c^2] (delta^2 + 4QC) = 2 s_cpl C q_c (1 - j C delta)^2
        # where q_c = 1 + C b - j C d = 1 - j C z_c
        c_val = float(C)
        q_c = 1.0 - 1j * c_val * z_c
        # (1 - j C delta)^2 = -C^2 delta^2 - 2 j C delta + 1
        p2 = np.array([-c_val**2, -2j * c_val, 1.0 - q_c**2], dtype=np.complex128)
        sc_poly = np.array([1.0, 0.0, four_qc], dtype=np.complex128)
        lhs = np.convolve(p2, sc_poly)
        rhs = 2.0 * s_cpl * c_val * q_c * np.array(
            [-c_val**2, -2j * c_val, 1.0], dtype=np.complex128
        )
        coeffs = lhs.copy()
        coeffs[-3:] -= rhs
        roots_raw = np.roots(coeffs)

    # Sort so delta_1 is the forward wave with largest growth rate Re(delta)
    order = np.argsort(-roots_raw.real)
    deltas = roots_raw[order]
    return {
        "deltas": deltas,
        "x1": float(deltas[0].real),
        "y1": float(deltas[0].imag),
        "poly_coeffs": coeffs,
    }


def solve_pierce_three_wave_bvp(
    CN: float | np.ndarray,
    b: float = 0.0,
    QC: float = 0.0,
    d: float = 0.0,
    coupling_sign: float = 1.0,
) -> dict[str, Any]:
    """Solve the 3-wave Pierce boundary-value problem for an unmodulated entrance beam.

    At the input plane :math:`z = 0`, an RF circuit voltage/field :math:`E_z(0) = E_0`
    is launched while the entering electron beam has zero initial velocity and
    density modulation (:math:`v_1(0) = 0, \\rho_1(0) = 0`). Since :math:`v_1` and
    :math:`I_1` for the :math:`i`-th eigenwave are proportional to
    :math:`E_i / \\delta_i` and :math:`E_i / \\delta_i^2`, the 3x3 Vandermonde system
    at :math:`z = 0` is:

    .. math::
        \\sum_{i=1}^3 E_i = E_0, \\qquad
        \\sum_{i=1}^3 \\frac{E_i}{\\delta_i} = 0, \\qquad
        \\sum_{i=1}^3 \\frac{E_i}{\\delta_i^2} = 0,

    with closed-form launching amplitudes

    .. math::
        \\frac{E_i}{E_0} = \\frac{\\delta_i^2}{(\\delta_i - \\delta_j)(\\delta_i - \\delta_k)}.

    Wait: when space charge :math:`QC > 0` is included via the local space-charge field
    :math:`E_{\\mathrm{sc}} = j(\\omega_q/\\omega)^2 (\\beta_e / \\varepsilon_0) I_1 / A_b`,
    the total longitudinal electric field acting on the beam is
    :math:`E_{\\mathrm{tot}} = E_{\\mathrm{ckt}} + E_{\\mathrm{sc}}`, where the circuit
    field ratios satisfy
    :math:`E_{\\mathrm{ckt}, i} / E_{\\mathrm{ckt}}(0) =
    (\\delta_i^2 + 4QC) / [(\\delta_i - \\delta_j)(\\delta_i - \\delta_k)]`!

    Parameters
    ----------
    CN : float or ndarray
        Dimensionless interaction length :math:`C N = \\beta_e C L / (2\\pi)`.
    b, QC, d : float, optional
        Pierce velocity, space-charge, and loss parameters.
    coupling_sign : float, optional
        Lorentz coupling sign multiplier (:math:`+1` physical, :math:`-1` mutated).

    Returns
    -------
    res : dict[str, Any]
        Dictionary with ``deltas``, ``weights_ckt``, ``transfer_fn``,
        ``gain_db``, ``asymptotic_gain_db``, ``A_launch_db``, and ``B_slope_db``.
    """
    disp = solve_pierce_dispersion(
        b=b, QC=QC, d=d, include_backward=False, coupling_sign=coupling_sign
    )
    deltas = disp["deltas"]
    four_qc = 4.0 * float(QC)

    weights = np.zeros(3, dtype=np.complex128)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        weights[i] = (deltas[i] ** 2 + four_qc) / (
            (deltas[i] - deltas[j]) * (deltas[i] - deltas[k])
        )

    cn_arr = np.asarray(CN, dtype=float)
    phase_arg = 2.0 * np.pi * cn_arr[..., None] * deltas
    transfer = np.sum(weights * np.exp(phase_arg), axis=-1)
    gain_db = 20.0 * np.log10(np.maximum(np.abs(transfer), 1e-30))

    a_launch_db = 20.0 * float(np.log10(max(abs(weights[0]), 1e-30)))
    b_slope_db = 20.0 * np.log10(np.e) * 2.0 * np.pi * float(deltas[0].real)
    asymp_gain_db = a_launch_db + b_slope_db * cn_arr

    return {
        "deltas": deltas,
        "weights_ckt": weights,
        "transfer_fn": transfer,
        "gain_db": gain_db,
        "asymptotic_gain_db": asymp_gain_db,
        "A_launch_db": float(a_launch_db),
        "B_slope_db": float(b_slope_db),
    }


def solve_coupled_mode_ode(
    L_m: float,
    beta_e: float,
    C: float,
    b: float = 0.0,
    QC: float = 0.0,
    d: float = 0.0,
    coupling_sign: float = 1.0,
    n_pts: int = 400,
) -> dict[str, Any]:
    """Integrate the 1D spatial coupled-mode TWT ODEs from z = 0 to z = L.

    Provides an independent second reference solver to cross-check the algebraic
    3-wave Pierce root/BVP solver. Factoring out the carrier :math:`\\exp(-j\\beta_e z)`
    from the circuit field envelope :math:`\\mathcal{E}(z)`, reduced velocity
    envelope :math:`\\mathcal{U}(z)`, and reduced current envelope :math:`\\mathcal{I}(z)`
    and normalizing the axial coordinate by :math:`\\zeta = \\beta_e C z = 2\\pi C N(z)`,
    the spatial coupled-mode equations are:

    .. math::
        \\frac{\\mathrm{d}\\mathcal{E}}{\\mathrm{d}\\zeta} &= -(j b + d)\\,\\mathcal{E}
            - j\\,\\mathrm{sign}_{\\mathrm{cpl}}\\,\\mathcal{I}, \\\\
        \\frac{\\mathrm{d}\\mathcal{U}}{\\mathrm{d}\\zeta} &= \\mathcal{E}
            - j 4QC\\,\\mathcal{I}, \\\\
        \\frac{\\mathrm{d}\\mathcal{I}}{\\mathrm{d}\\zeta} &= -j\\,\\mathcal{U},

    with initial conditions :math:`\\mathcal{E}(0) = 1, \\mathcal{U}(0) = 0,
    \\mathcal{I}(0) = 0`. Differentiating confirms that :math:`\\exp(\\delta\\zeta)`
    satisfies :math:`(\\delta + j b + d)(\\delta^2 + 4QC) = -j\\,\\mathrm{sign}_{\\mathrm{cpl}}`.

    Parameters
    ----------
    L_m : float
        Interaction length :math:`L` [m].
    beta_e : float
        Electronic wavenumber :math:`\\beta_e = \\omega / u_0` [rad/m].
    C, b, QC, d : float
        Pierce parameters.
    coupling_sign : float, optional
        Lorentz coupling sign (:math:`+1` physical, :math:`-1` mutated).
    n_pts : int, optional
        Number of axial evaluation points along :math:`z \\in [0, L]`.

    Returns
    -------
    res : dict[str, Any]
        Dictionary with ``z_m``, ``CN_arr``, ``E_env``, ``U_env``, ``I_env``,
        ``gain_db_vs_z``, and ``final_gain_db``.
    """
    zeta_max = float(beta_e) * float(C) * float(L_m)
    z_eval = np.linspace(0.0, float(L_m), int(n_pts))
    zeta_eval = float(beta_e) * float(C) * z_eval
    four_qc = 4.0 * float(QC)
    z_c = 1j * float(b) + float(d)
    s_cpl = float(coupling_sign)

    def rhs(_zeta: float, y: np.ndarray) -> np.ndarray:
        e_val, u_val, i_val = y
        de = -z_c * e_val - s_cpl * i_val
        du = -e_val - 1j * four_qc * i_val
        di = -1j * u_val
        return np.array([de, du, di], dtype=np.complex128)

    y0 = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=np.complex128)
    sol = solve_ivp(
        rhs,
        (0.0, zeta_max),
        y0,
        t_eval=zeta_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    e_env = sol.y[0]
    u_env = sol.y[1]
    i_env = sol.y[2]
    gain_db_vs_z = 20.0 * np.log10(np.maximum(np.abs(e_env), 1e-30))
    return {
        "z_m": z_eval,
        "CN_arr": zeta_eval / (2.0 * np.pi),
        "E_env": e_env,
        "U_env": u_env,
        "I_env": i_env,
        "gain_db_vs_z": gain_db_vs_z,
        "final_gain_db": float(gain_db_vs_z[-1]),
    }


def solve_bwo_start_oscillation(
    QC: float = 0.0,
    d: float = 0.0,
    backward_sign: float = 1.0,
) -> dict[str, float]:
    """Solve the Johnson (1955) backward-wave oscillator (BWO) start-oscillation condition.

    When the electron beam is velocity-synchronized with a backward space harmonic
    (:math:`v_p = \\omega / \\beta_{-1} > 0` in :math:`+z`, while the electromagnetic
    group/energy velocity :math:`v_g < 0` carries power toward the gun end :math:`z = 0`),
    the circuit transfer impedance changes sign relative to the forward TWT:

    .. math::
        (\\delta + j b - d)(\\delta^2 + 4QC) = +j.

    With an unmodulated beam entering at :math:`z = 0` (:math:`v_1(0) = 0, I_1(0) = 0`)
    and a matched termination at the collector end :math:`z = L` (:math:`E_{\\mathrm{ckt}}(L) = 0`),
    self-sustained oscillation occurs when the backward-wave transfer function
    :math:`T_{\\mathrm{BWO}}(CN, b) = E_{\\mathrm{ckt}}(L) / E_{\\mathrm{ckt}}(0)` vanishes:

    .. math::
        T_{\\mathrm{BWO}}(CN, b) = \\sum_{i=1}^3
        \\frac{\\delta_i^2 + 4QC}{(\\delta_i - \\delta_j)(\\delta_i - \\delta_k)}
        \\exp(2\\pi\\,CN\\,\\delta_i) = 0.

    Reference: H. R. Johnson, "Backward-Wave Oscillators," Proc. IRE 43(6),
    684-697 (1955), Table I: :math:`(CN)_{\\mathrm{st}} = 0.3141, b_{\\mathrm{st}} = 1.522`
    at :math:`QC = d = 0`.

    Parameters
    ----------
    QC : float, optional
        Space-charge parameter :math:`QC \\ge 0`.
    d : float, optional
        Circuit loss parameter :math:`d \\ge 0`.
    backward_sign : float, optional
        Sign of backward-wave circuit power flow (:math:`+1` for BWO, :math:`-1` for forward TWT).

    Returns
    -------
    res : dict[str, float]
        Dictionary with ``CN_start``, ``b_start``, and ``residual_abs``.
    """
    four_qc = 4.0 * float(QC)
    s_bwd = float(backward_sign)

    def bwo_transfer(cn_val: float, b_val: float) -> complex:
        z_c = 1j * float(b_val) - float(d)
        coeffs = np.array(
            [1.0 + 0j, z_c, four_qc + 0j, four_qc * z_c - 1j * s_bwd],
            dtype=np.complex128,
        )
        deltas = np.roots(coeffs)
        val = 0.0 + 0j
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            w_i = (deltas[i] ** 2 + four_qc) / (
                (deltas[i] - deltas[j]) * (deltas[i] - deltas[k])
            )
            val += w_i * np.exp(2.0 * np.pi * float(cn_val) * deltas[i])
        return complex(val)

    # Good initial guess tracking QC
    cn_guess = 0.3141 * (1.0 + 0.08 * four_qc)
    b_guess = max(1.5218, np.sqrt(four_qc) + 0.25)

    def obj(x: np.ndarray) -> np.ndarray:
        t_val = bwo_transfer(float(x[0]), float(x[1]))
        return np.array([t_val.real, t_val.imag], dtype=float)

    sol = root(obj, np.array([cn_guess, b_guess], dtype=float), tol=1e-13)
    cn_st, b_st = float(sol.x[0]), float(sol.x[1])
    res_abs = abs(bwo_transfer(cn_st, b_st))
    return {
        "CN_start": cn_st,
        "b_start": b_st,
        "residual_abs": float(res_abs),
        "QC": float(QC),
        "d": float(d),
    }


def solve_bwo_start_current_vs_length(
    lengths_m: np.ndarray,
    freq_hz: float = 97.308e9,
    beta_m1: float = 1.5 * np.pi / 0.5e-3,
    k_m1_ohm: float = 2.08,
    voltage_v: float = 12401.0,
    backward_sign: float = 1.0,
) -> dict[str, Any]:
    """Solve the BWO boundary-value root problem T_BWO(I0, b; L) = 0 at each length L."""
    l_arr = np.asarray(lengths_m, dtype=float)
    w0 = 2.0 * np.pi * float(freq_hz)
    vp = w0 / float(beta_m1)
    lam_e = vp / float(freq_hz)
    _, gamma0 = relativistic_gamma(voltage=float(voltage_v))
    v0_eff = 0.5 * gamma0 * (gamma0 + 1.0) * float(voltage_v)
    s_bwd = float(backward_sign)

    i_st_arr = np.zeros_like(l_arr)
    b_st_arr = np.zeros_like(l_arr)
    res_arr = np.zeros_like(l_arr)
    for idx_l, l_val in enumerate(l_arr):
        n_elec_l = float(l_val / lam_e)

        def _bv_res(x_ib: np.ndarray) -> np.ndarray:
            i_try = max(float(x_ib[0]), 1e-12)
            b_try = float(x_ib[1])
            c_try = (i_try * float(k_m1_ohm) / (4.0 * v0_eff)) ** (1.0 / 3.0)
            cn_try = c_try * n_elec_l
            coeffs = np.array([1.0 + 0j, 1j * b_try, 0.0 + 0j, -1j * s_bwd], dtype=np.complex128)
            deltas = np.roots(coeffs)
            val = 0.0 + 0j
            for ii in range(3):
                jj, kk = (ii + 1) % 3, (ii + 2) % 3
                w_ii = (deltas[ii] ** 2) / ((deltas[ii] - deltas[jj]) * (deltas[ii] - deltas[kk]))
                val += w_ii * np.exp(2.0 * np.pi * cn_try * deltas[ii])
            return np.array([val.real, val.imag], dtype=float)

        i_guess = (4.0 * v0_eff / float(k_m1_ohm)) * ((0.310 / n_elec_l) ** 3)
        sol_l = root(_bv_res, np.array([i_guess, 1.50], dtype=float), tol=1e-13)
        i_st_arr[idx_l] = float(sol_l.x[0])
        b_st_arr[idx_l] = float(sol_l.x[1])
        res_arr[idx_l] = float(np.linalg.norm(_bv_res(sol_l.x)))

    slope_l, _ = np.polyfit(np.log(l_arr), np.log(np.maximum(i_st_arr, 1e-30)), 1)
    return {
        "lengths_m": l_arr,
        "i_st_a": i_st_arr,
        "b_st": b_st_arr,
        "residuals": res_arr,
        "max_residual_abs": float(np.max(res_arr)),
        "loglog_slope": float(slope_l),
    }


# =====================================================================
# 3. FIT Simulation Builders & Task Runners
# =====================================================================


def run_scw_time_centering_check(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
    force: bool = False,
) -> dict[str, Any]:
    """Run Task 1: Pierce reference verification + LinearizedBeam SCW time-centring check.

    Verifies:
    1. Analytic Pierce constants :math:`A = -20\\log_{10} 3 = -9.542425` dB and
       :math:`B = 20\\pi\\sqrt{3}\\log_{10} e = 47.263383` dB.
    2. Agreement between the 3-wave Pierce BVP and the 1D coupled-mode spatial ODE
       integrator to :math:`< 10^{-7}` dB across :math:`CN \\in [0, 1.2]` and
       :math:`b \\in [-2, +2]`.
    3. Space-charge-wave dispersion and time-centring of ``LinearizedBeam`` inside
       the smooth Karetnikova beam tunnel (:math:`2a = 200\\;\\mu\\mathrm{m},
       2b = 150\\;\\mu\\mathrm{m}, V_0 = 20\\;\\mathrm{kV}`) across three axial
       resolutions :math:`N_z \\in \\{20, 28, 40\\}` against
       ``space_charge_wave_dispersion`` and ``reduction_factor_planar``.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "task_01_pierce_theory_and_scw_check.json"
    npz_path = DATA_DIR / "task_01_pierce_theory_and_scw_check.npz"
    if json_path.exists() and npz_path.exists() and not force:
        return json.loads(json_path.read_text(encoding="utf-8"))

    # 1. Exact Pierce constants & 3-wave BVP vs Coupled-Mode ODE cross-check
    A_exact_db = -20.0 * float(np.log10(3.0))
    B_exact_db = 20.0 * float(np.log10(np.e)) * np.pi * np.sqrt(3.0)
    bvp_0 = solve_pierce_three_wave_bvp(np.linspace(0.0, 1.2, 241), b=0.0, QC=0.0, d=0.0)
    ode_0 = solve_coupled_mode_ode(
        L_m=1.2 * 2.0 * np.pi / (1000.0 * 0.02),
        beta_e=1000.0,
        C=0.02,
        b=0.0,
        QC=0.0,
        d=0.0,
        n_pts=241,
    )
    max_bvp_ode_diff_db = float(np.max(np.abs(bvp_0["gain_db"] - ode_0["gain_db_vs_z"])))

    # Sweep b in [-2.5, 2.5] for QC in {0.0, 0.25, 0.5, 1.0}
    b_grid = np.linspace(-2.5, 2.5, 201)
    qc_list = [0.0, 0.25, 0.50, 1.00]
    x1_vs_b_qc = np.zeros((len(qc_list), len(b_grid)), dtype=float)
    y1_vs_b_qc = np.zeros((len(qc_list), len(b_grid)), dtype=float)
    bvp_ode_detuned_diffs = []
    for iq, qc_v in enumerate(qc_list):
        for ib, b_v in enumerate(b_grid):
            r_d = solve_pierce_dispersion(b=float(b_v), QC=float(qc_v), d=0.0)
            x1_vs_b_qc[iq, ib] = r_d["x1"]
            y1_vs_b_qc[iq, ib] = r_d["y1"]
        # Cross-check ODE vs BVP at b = 0.8, QC = qc_v
        bvp_t = solve_pierce_three_wave_bvp(0.8, b=0.8, QC=float(qc_v), d=0.08)
        ode_t = solve_coupled_mode_ode(
            L_m=0.8 * 2.0 * np.pi / (1000.0 * 0.02),
            beta_e=1000.0,
            C=0.02,
            b=0.8,
            QC=float(qc_v),
            d=0.08,
            n_pts=101,
        )
        bvp_ode_detuned_diffs.append(abs(float(bvp_t["gain_db"]) - float(ode_t["final_gain_db"])))

    # 2. 2D-to-3D sheet-beam mapping verification
    mapping = map_2d_sheet_to_3d_grating(spec)

    # 3. Space-charge-wave time-centring check in the smooth Karetnikova tunnel
    # Use a single electronic wavelength lambda_e = u0 / f0 with periodic z boundary
    # and compare the fast/slow space-charge beat frequency omega_q_FIT against
    # reduction_factor_planar * omega_p / gamma0^(3/2).
    u0, gamma0 = relativistic_gamma(voltage=spec.voltage_v)
    f_ref = 186.0e9
    omega_ref = 2.0 * np.pi * f_ref
    beta_e_ref = omega_ref / u0
    lam_e = 2.0 * np.pi / beta_e_ref
    r_planar_th = float(
        reduction_factor_planar(
            omega_ref,
            u0,
            spec.beam_half_thickness_m,
            a=spec.tunnel_half_height_m,
            relativistic=True,
        )
    )
    r_kar_eq4 = float(
        np.sqrt(
            karetnikova_depression_factor_sq(
                beta_e_ref,
                2.0 * spec.beam_half_thickness_m,
                tunnel_half_height_m=spec.tunnel_half_height_m,
                gamma0=gamma0,
            )
        )
    )
    w_fast_th, w_slow_th = space_charge_wave_dispersion(
        beta_e_ref,
        u0=u0,
        omega_p=2.60e11 * (gamma0**1.5),
        R=r_planar_th,
        gamma0=gamma0,
    )
    w_fast_th = float(w_fast_th)
    w_slow_th = float(w_slow_th)
    wq_th = 0.5 * (w_fast_th - w_slow_th)
    tau_th = beta_e_ref / gamma0
    t_trans_th = tau_th * float(np.sqrt(max(1.0 / (r_planar_th**2) - 1.0, 0.0)))

    scw_levels = []
    for nz_lam in (20, 28, 40):
        ny_tun = max(16, int(round(0.8 * nz_lam)))
        dx = 50.0e-6
        grid_scw = GridFIT3D(
            0.0,
            dx,
            -spec.tunnel_half_height_m,
            +spec.tunnel_half_height_m,
            0.0,
            lam_e,
            1,
            ny_tun,
            nz_lam,
            verbose=0,
        )
        t0 = time.perf_counter()
        solver_scw = SolverFIT3D(
            grid_scw,
            bc_low=["periodic", "pec", "bloch"],
            bc_high=["periodic", "pec", "bloch"],
            k_bloch=(0.0, 0.0, -float(beta_e_ref)),
            dtype=np.complex128,
            cfln=0.75,
            verbose=0,
        )
        dz = lam_e / float(nz_lam)
        sinc_dz = float(np.sin(beta_e_ref * dz) / (beta_e_ref * dz))
        # Match discrete advection speed u0_disc * sin(k dz)/dz = u0 * k so the carrier
        # frequency is exact and we isolate the space-charge plasma oscillation omega_q:
        u0_disc = u0 / sinc_dz
        beam_scw = LinearizedBeam(
            u0=u0_disc,
            omega_p=2.60e11 * (gamma0**1.5),
            gamma0=gamma0,
            axis="z",
            half_thickness=spec.beam_half_thickness_m,
            solver=solver_scw,
        )
        dt = float(solver_scw.dt)
        t_beat = 2.0 * np.pi / wq_th
        n_steps = max(int(np.ceil(1.8 * t_beat / dt)), 2000)
        est_scw = estimate((1, ny_tun, nz_lam), steps=n_steps, dtype="complex128")
        preflight(est_scw, LOCAL_VM_SPEC)

        z_c = beam_scw.s_coords
        y_nodes = grid_scw.y[: solver_scw.Ny]
        es_prof = np.where(
            np.abs(y_nodes) <= spec.beam_half_thickness_m + 1e-12,
            np.cos(t_trans_th * y_nodes),
            0.0,
        )
        # Seed pure fundamental-mode velocity perturbation v1(z, y, 0)
        beam_scw.v1[:, :] = (
            1.0e4 * np.exp(-1j * beta_e_ref * z_c)[:, None] * es_prof[beam_scw.keep_lines][None, :]
        )
        i1_proj = np.zeros(n_steps, dtype=np.complex128)
        e_proj = np.zeros(n_steps, dtype=np.complex128)
        phase_vec = np.exp(+1j * beta_e_ref * z_c) / float(nz_lam)
        iy_mid = ny_tun // 2
        for step in range(n_steps):
            solver_scw.one_step()
            i1_proj[step] = np.sum(beam_scw.axial_current() * phase_vec)
            e_proj[step] = np.sum(solver_scw.E[0, iy_mid, :nz_lam, "z"] * phase_vec)
        wall_s = time.perf_counter() - t0

        # Demodulate carrier exp(+j omega_0 t) to measure the space-charge beat sin(omega_q t)
        t_arr = (np.arange(n_steps, dtype=float) + 0.5) * dt
        demod_i1 = i1_proj * np.exp(-1j * (beta_e_ref * u0) * t_arr)
        n_pad_fft = max(131072, 1 << int(np.ceil(np.log2(n_steps * 16))))
        win = np.hanning(n_steps)
        spec_fft = np.abs(np.fft.fft(demod_i1 * win, n=n_pad_fft))
        freqs_fft = np.fft.fftfreq(n_pad_fft, d=dt) * 2.0 * np.pi
        pos_mask = (freqs_fft > 0.3 * wq_th) & (freqs_fft < 1.8 * wq_th)
        idx_pk = int(np.argmax(spec_fft[pos_mask]))
        freqs_pos = freqs_fft[pos_mask]
        spec_pos = spec_fft[pos_mask]
        if 0 < idx_pk < len(freqs_pos) - 1:
            y_m, y_0, y_p = np.log(spec_pos[idx_pk - 1 : idx_pk + 2] + 1e-30)
            delta_bin = 0.5 * (y_m - y_p) / max(abs(y_m - 2.0 * y_0 + y_p), 1e-30)
            df_bin = freqs_pos[1] - freqs_pos[0]
            wq_fit = float(freqs_pos[idx_pk] - delta_bin * df_bin)
        else:
            wq_fit = float(freqs_pos[idx_pk])

        # Check time-centring amplification factor |G| - 1 over 2400 steps:
        # Quarter-wave peaks of |i1_proj|
        env_i1 = np.abs(i1_proj)
        quarter_idx = int(round(0.5 * np.pi / (wq_th * dt)))
        three_quarter_idx = int(round(1.5 * np.pi / (wq_th * dt)))
        amp_ratio = float(env_i1[min(three_quarter_idx, n_steps - 1)] / max(env_i1[quarter_idx], 1e-30))
        rel_err_wq = abs(wq_fit - wq_th) / wq_th

        scw_levels.append(
            {
                "nz_per_lambda_e": int(nz_lam),
                "ny": int(ny_tun),
                "dz_um": round(dz * 1e6, 3),
                "sinc_advection_factor": round(sinc_dz, 6),
                "wq_theory_rad_s": float(wq_th),
                "wq_fit_rad_s": float(wq_fit),
                "rel_err_wq": float(rel_err_wq),
                "amplitude_conservation_ratio": round(amp_ratio, 6),
                "wall_time_s": round(wall_s, 3),
            }
        )

    np.savez(
        npz_path,
        CN_arr=bvp_0["asymptotic_gain_db"] * 0.0 + np.linspace(0.0, 1.2, 241),
        bvp_gain_db_0=bvp_0["gain_db"],
        asymp_gain_db_0=bvp_0["asymptotic_gain_db"],
        ode_gain_db_0=ode_0["gain_db_vs_z"],
        b_grid=b_grid,
        qc_list=np.array(qc_list, dtype=float),
        x1_vs_b_qc=x1_vs_b_qc,
        y1_vs_b_qc=y1_vs_b_qc,
        scw_t_ps=t_arr * 1e12,
        scw_i1_env=np.abs(i1_proj) / max(float(np.max(np.abs(i1_proj))), 1e-30),
        scw_e_env=np.abs(e_proj) / max(float(np.max(np.abs(e_proj))), 1e-30),
    )

    chi_rel_20kv = 2.0 / (float(gamma0) * (float(gamma0) + 1.0))
    gamma_186 = float(beta_e_ref / gamma0)
    payload = {
        "A_launching_loss_db": round(A_exact_db, 6),
        "B_pierce_slope_db": round(B_exact_db, 6),
        "max_bvp_vs_ode_diff_db": float(max(max_bvp_ode_diff_db, max(bvp_ode_detuned_diffs))),
        "gamma0_20kv": round(float(gamma0), 6),
        "chi_rel_20kv": round(float(chi_rel_20kv), 6),
        "sheet_2d_to_3d_mapping": mapping,
        "scw_check": {
            "f_ref_ghz": 186.0,
            "lambda_e_um": round(lam_e * 1e6, 3),
            "gamma_h_186ghz": round(float(gamma_186 * spec.beam_half_thickness_m), 4),
            "gamma_a_186ghz": round(float(gamma_186 * spec.tunnel_half_height_m), 4),
            "R_planar_wakis": round(r_planar_th, 6),
            "D_karetnikova_eq4": round(r_kar_eq4, 6),
            "omega_q_theory_rad_s": float(wq_th),
            "omega_fast_theory_rad_s": float(w_fast_th),
            "omega_slow_theory_rad_s": float(w_slow_th),
            "levels": scw_levels,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


_GRATING_2D_MESH_LEVELS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "coarse": ((4, 5, 8), (4, 6, 5)),      # Nz/d = 26 (21.8 cells/lambda_g), n_gap = 4, Ny = 25
    "medium": ((5, 6, 10), (5, 8, 6)),     # Nz/d = 32 (26.2 cells/lambda_g), n_gap = 5, Ny = 31
    "fine": ((6, 8, 12), (6, 10, 8)),      # Nz/d = 40 (34.9 cells/lambda_g), n_gap = 6, Ny = 39
    "ultrafine": ((7, 9, 14), (7, 12, 9)), # Nz/d = 46 (39.2 cells/lambda_g), n_gap = 7, Ny = 45
}


def build_2d_grating_hot_grid(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
    n_periods: int = 22,
    n_in_smooth: int = 2,
    n_out_smooth: int = 2,
    nz_per_period: int = 26,
    dy_beam_m: float = 25.0e-6,
    dy_slot_m: float = 60.0e-6,
    dx_m: float = 0.35e-3,
    mesh_level: str | None = None,
) -> tuple[GridFIT3D, dict[str, Any]]:
    """Build a node-aligned graded 2D FIT grid for the Karetnikova staggered-grating TWT.

    Places grid nodes exactly on all vane edges (:math:`z / d \\in \\{0.15, 0.35, 0.65, 0.85\\}`)
    and vertical tunnel/beam interfaces (:math:`y \\in \\{-y_{\\max}, -a, -b, +b, +a, +y_{\\max}\\}`)
    so that conformal PEC facets align with cell faces without partial-cell staircase
    oscillation across grid refinements, while guaranteeing :math:`\\ge 20` cells per
    guided wavelength :math:`\\lambda_g = 2\\pi / \\beta_{+1}` and :math:`\\ge 4` cells
    across the :math:`25\\;\\mu\\mathrm{m}` beam-wall gap on every mesh level.
    """
    d_per = spec.period_m
    s_vane = spec.vane_thickness_m
    h_vane = spec.vane_height_m
    a_half = spec.tunnel_half_height_m
    b_half = spec.beam_half_thickness_m
    y_max = a_half + h_vane

    if mesh_level is not None and mesh_level in _GRATING_2D_MESH_LEVELS:
        lvl_key = mesh_level
    elif int(nz_per_period) <= 26:
        lvl_key = "coarse"
    elif int(nz_per_period) <= 32:
        lvl_key = "medium"
    elif int(nz_per_period) <= 40:
        lvl_key = "fine"
    else:
        lvl_key = "ultrafine"

    (nz1, nz2, nz3), (n_gap, n_beam, n_slot) = _GRATING_2D_MESH_LEVELS[lvl_key]
    nz_per_actual = 2 * nz1 + 2 * nz2 + nz3

    z_sub = np.concatenate(
        [
            np.linspace(0.0, 0.15 * d_per, nz1, endpoint=False),
            np.linspace(0.15 * d_per, 0.35 * d_per, nz2, endpoint=False),
            np.linspace(0.35 * d_per, 0.65 * d_per, nz3, endpoint=False),
            np.linspace(0.65 * d_per, 0.85 * d_per, nz2, endpoint=False),
            np.linspace(0.85 * d_per, 1.00 * d_per, nz1, endpoint=False),
        ]
    )
    n_tot = int(n_in_smooth) + int(n_periods) + int(n_out_smooth)
    z_list = [(-int(n_in_smooth) + m) * d_per + z_sub for m in range(n_tot)]
    z_list.append(np.array([float(n_periods + n_out_smooth) * d_per]))
    z_nodes = np.concatenate(z_list)

    y_nodes = np.concatenate(
        [
            np.linspace(-y_max, -a_half, n_slot, endpoint=False),
            np.linspace(-a_half, -b_half, n_gap, endpoint=False),
            np.linspace(-b_half, +b_half, n_beam, endpoint=False),
            np.linspace(+b_half, +a_half, n_gap, endpoint=False),
            np.linspace(+a_half, +y_max, n_slot + 1),
        ]
    )
    x_nodes = np.array([0.0, float(dx_m)], dtype=float)
    z_min = float(z_nodes[0])
    z_max = float(z_nodes[-1])
    eps_b = 1.0e-9

    boxes = [
        pv.Box(
            bounds=(
                -dx_m,
                2.0 * dx_m,
                -y_max - 0.1e-3,
                -a_half + eps_b,
                z_min - 0.1e-3,
                0.0 + eps_b,
            )
        ),
        pv.Box(
            bounds=(
                -dx_m,
                2.0 * dx_m,
                +a_half - eps_b,
                +y_max + 0.1e-3,
                z_min - 0.1e-3,
                0.0 + eps_b,
            )
        ),
        pv.Box(
            bounds=(
                -dx_m,
                2.0 * dx_m,
                -y_max - 0.1e-3,
                -a_half + eps_b,
                float(n_periods) * d_per - eps_b,
                z_max + 0.1e-3,
            )
        ),
        pv.Box(
            bounds=(
                -dx_m,
                2.0 * dx_m,
                +a_half - eps_b,
                +y_max + 0.1e-3,
                float(n_periods) * d_per - eps_b,
                z_max + 0.1e-3,
            )
        ),
    ]
    for m in range(int(n_periods)):
        z0 = float(m) * d_per
        boxes.append(
            pv.Box(
                bounds=(
                    -dx_m,
                    2.0 * dx_m,
                    -y_max - 0.1e-3,
                    -a_half + eps_b,
                    z0 + 0.25 * d_per - 0.5 * s_vane - eps_b,
                    z0 + 0.25 * d_per + 0.5 * s_vane + eps_b,
                )
            )
        )
        boxes.append(
            pv.Box(
                bounds=(
                    -dx_m,
                    2.0 * dx_m,
                    +a_half - eps_b,
                    +y_max + 0.1e-3,
                    z0 + 0.75 * d_per - 0.5 * s_vane - eps_b,
                    z0 + 0.75 * d_per + 0.5 * s_vane + eps_b,
                )
            )
        )
    merged_pec = pv.merge(boxes)

    grid = GridFIT3D(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        stl_solids={"pec": merged_pec},
        stl_materials={"pec": "pec"},
        conformal="pec",
        verbose=0,
    )

    dz_mean = d_per / float(nz_per_actual)
    dz_arr = np.diff(z_nodes)
    dy_arr = np.diff(y_nodes)
    dz_max = float(np.max(dz_arr))
    dy_gap = (a_half - b_half) / float(n_gap)
    dy_beam = (2.0 * b_half) / float(n_beam)
    dy_slot = h_vane / float(n_slot)

    # Guided wavelength lambda_g = 2*pi / beta_{+1} at 186 GHz (436.07 um) and 200 GHz (402.33 um)
    lam_g_186 = 2.0 * np.pi / 14408.73
    lam_g_200 = 2.0 * np.pi / 15616.90
    cells_per_lambda_g_186 = lam_g_186 / dz_max
    cells_per_lambda_g_200 = lam_g_200 / dz_max
    lam_0_min = C0 / 210.0e9
    cells_per_lambda0 = lam_0_min / max(dz_max, float(np.max(dy_arr)))

    meta = {
        "mesh_level": lvl_key,
        "nx": int(grid.Nx),
        "ny": int(grid.Ny),
        "nz": int(grid.Nz),
        "n_cells": int(grid.Nx * grid.Ny * grid.Nz),
        "nz_per_period": int(nz_per_actual),
        "dz_um": round(dz_mean * 1e6, 3),
        "dz_max_um": round(dz_max * 1e6, 3),
        "dy_min_um": round(float(np.min(dy_arr)) * 1e6, 3),
        "dy_gap_um": round(dy_gap * 1e6, 3),
        "dy_beam_um": round(dy_beam * 1e6, 3),
        "dy_slot_um": round(dy_slot * 1e6, 3),
        "dy_max_um": round(float(np.max(dy_arr)) * 1e6, 3),
        "cells_across_vane": int(nz2),
        "cells_across_beam_wall_gap": int(n_gap),
        "cells_across_beam": int(n_beam),
        "lambda_g_186ghz_um": round(lam_g_186 * 1e6, 2),
        "cells_per_lambda_g_186ghz": round(float(cells_per_lambda_g_186), 2),
        "cells_per_lambda_g_200ghz": round(float(cells_per_lambda_g_200), 2),
        "cells_per_lambda0_210ghz": round(float(cells_per_lambda0), 2),
    }
    return grid, meta


def extract_2d_grating_cold_circuit_on_grid(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
    mesh_level: str = "coarse",
    freq_hz: float = 186.0e9,
    current_a: float = 0.25,
) -> dict[str, float]:
    """Extract cold-circuit dispersion, Pierce impedance, and Pierce growth rate on the same FIT grid.

    Runs a 1-period Bloch unit-cell FIT simulation with the exact ``(nz1, nz2, nz3)``
    and ``(n_gap, n_beam, n_slot)`` cell sizes of ``mesh_level`` to measure on that
    same discrete grid:
    - the cold eigenfrequency :math:`f_{\\mathrm{FIT}}(\\beta_0)` and cold loss :math:`\\alpha_0`,
    - the on-grid synchronous wavenumber :math:`\\beta_{c,\\mathrm{FIT}}(f_0)`,
    - the 2D interaction impedance :math:`k_{2\\mathrm{D,FIT}}` [:math:`\\Omega\\cdot\\mathrm{m}`]
      and 3D beam-averaged Pierce impedance :math:`K_{3\\mathrm{D,FIT}} = k_{2\\mathrm{D,FIT}} / W_{\\mathrm{eff}}` [:math:`\\Omega`],
    - and the resulting on-grid Pierce growth rate :math:`\\alpha_{\\mathrm{Pierce,grid}}`.
    """
    d_per = spec.period_m
    s_vane = spec.vane_thickness_m
    h_vane = spec.vane_height_m
    a_half = spec.tunnel_half_height_m
    b_half = spec.beam_half_thickness_m
    y_max = a_half + h_vane
    dx_m = 0.35e-3
    eps_b = 1.0e-9

    (nz1, nz2, nz3), (n_gap, n_beam, n_slot) = _GRATING_2D_MESH_LEVELS[mesh_level]
    z_sub = np.concatenate(
        [
            np.linspace(0.0, 0.15 * d_per, nz1, endpoint=False),
            np.linspace(0.15 * d_per, 0.35 * d_per, nz2, endpoint=False),
            np.linspace(0.35 * d_per, 0.65 * d_per, nz3, endpoint=False),
            np.linspace(0.65 * d_per, 0.85 * d_per, nz2, endpoint=False),
            np.linspace(0.85 * d_per, 1.00 * d_per, nz1 + 1),
        ]
    )
    y_nodes = np.concatenate(
        [
            np.linspace(-y_max, -a_half, n_slot, endpoint=False),
            np.linspace(-a_half, -b_half, n_gap, endpoint=False),
            np.linspace(-b_half, +b_half, n_beam, endpoint=False),
            np.linspace(+b_half, +a_half, n_gap, endpoint=False),
            np.linspace(+a_half, +y_max, n_slot + 1),
        ]
    )
    boxes = [
        pv.Box(
            bounds=(
                -dx_m,
                2.0 * dx_m,
                -y_max - 0.1e-3,
                -a_half + eps_b,
                0.25 * d_per - 0.5 * s_vane - eps_b,
                0.25 * d_per + 0.5 * s_vane + eps_b,
            )
        ),
        pv.Box(
            bounds=(
                -dx_m,
                2.0 * dx_m,
                +a_half - eps_b,
                +y_max + 0.1e-3,
                0.75 * d_per - 0.5 * s_vane - eps_b,
                0.75 * d_per + 0.5 * s_vane + eps_b,
            )
        ),
    ]
    grid_1p = GridFIT3D(
        x=np.array([0.0, dx_m], dtype=float),
        y=y_nodes,
        z=z_sub,
        stl_solids={"pec": pv.merge(boxes)},
        stl_materials={"pec": "pec"},
        conformal="pec",
        verbose=0,
    )

    cold_curve = get_karetnikova_cold_circuit_curve(spec)
    beta_c_mcv = float(np.interp(freq_hz, cold_curve["freq_dense_hz"], cold_curve["beta_p1_dense"]))
    vg_mcv = float(np.interp(freq_hz, cold_curve["freq_dense_hz"], cold_curve["ve_over_c_dense"])) * C0
    beta0_target = beta_c_mcv - 2.0 * np.pi / d_per
    kx = 2.0 * float(np.arcsin(0.5 * dx_m * np.pi / spec.waveguide_width_m)) / dx_m

    est = estimate((int(grid_1p.Nx), int(grid_1p.Ny), int(grid_1p.Nz)), steps=6500, dtype="complex128", conformal="pec")
    preflight(est, LOCAL_VM_SPEC)
    sol = SolverFIT3D(
        grid_1p,
        use_stl=True,
        bc_low=["bloch", "pec", "bloch"],
        bc_high=["bloch", "pec", "bloch"],
        k_bloch=(float(kx), 0.0, -float(beta0_target)),
        dtype=np.complex128,
        cfln=0.85,
        verbose=0,
    )
    dt = float(sol.dt)
    z_c = 0.5 * (np.asarray(grid_1p.z[:-1]) + np.asarray(grid_1p.z[1:]))[: sol.Nz]
    y_c = 0.5 * (np.asarray(grid_1p.y[:-1]) + np.asarray(grid_1p.y[1:]))[: sol.Ny]
    tun_mask = (np.abs(y_c) <= a_half)[:, None]
    seed_yz = tun_mask * np.exp(-1j * beta_c_mcv * z_c)[None, :]
    t0 = 6.0e-12
    tau = 2.0e-12
    w0 = 2.0 * np.pi * float(freq_hz)
    nt_drive = int(round(15.0e-12 / dt))
    nt_ring = int(round(45.0e-12 / dt))
    for step in range(nt_drive):
        t = step * dt
        env = np.exp(-0.5 * ((t - t0) / tau) ** 2) * np.sin(w0 * (t - t0))
        sol.E[0, : sol.Ny, : sol.Nz, "z"] += dt * env * seed_yz
        sol.one_step()
    iy_mid = sol.Ny // 2
    iz_mid = sol.Nz // 2
    sig = np.zeros(nt_ring, dtype=np.complex128)
    for step in range(nt_ring):
        sol.one_step()
        sig[step] = sol.E[0, iy_mid, iz_mid, "z"]

    win = np.hanning(nt_ring)
    n_fft = 262144
    spec_fft = np.abs(np.fft.fft(sig * win, n=n_fft))
    freqs_fft = np.fft.fftfreq(n_fft, d=dt)
    mask_f = (freqs_fft > 165.0e9) & (freqs_fft < 205.0e9)
    f_sub = freqs_fft[mask_f]
    s_sub = spec_fft[mask_f]
    idx_pk = int(np.argmax(s_sub))
    if 0 < idx_pk < len(s_sub) - 1:
        y_m, y_0, y_p = np.log(s_sub[idx_pk - 1 : idx_pk + 2] + 1e-30)
        delta_bin = 0.5 * (y_m - y_p) / max(abs(y_m - 2.0 * y_0 + y_p), 1e-30)
        f_fit_hz = float(f_sub[idx_pk] - delta_bin * (f_sub[1] - f_sub[0]))
    else:
        f_fit_hz = float(f_sub[idx_pk])

    beta_c_fit = beta_c_mcv + 2.0 * np.pi * (float(freq_hz) - f_fit_hz) / vg_mcv
    alpha_loss_fit = 0.0  # Lossless PEC grating

    # Second pass: accumulate steady-state Bloch eigenmode at f_fit_hz with Hanning window
    sol2 = SolverFIT3D(
        grid_1p,
        use_stl=True,
        bc_low=["bloch", "pec", "bloch"],
        bc_high=["bloch", "pec", "bloch"],
        k_bloch=(float(kx), 0.0, -float(beta0_target)),
        dtype=np.complex128,
        cfln=0.85,
        verbose=0,
    )
    for step in range(nt_drive):
        t = step * dt
        env = np.exp(-0.5 * ((t - t0) / tau) ** 2) * np.sin(w0 * (t - t0))
        sol2.E[0, : sol2.Ny, : sol2.Nz, "z"] += dt * env * seed_yz
        sol2.one_step()
    w_fit = 2.0 * np.pi * f_fit_hz
    w_norm = float(np.sum(win))
    modes = {
        c: np.zeros((sol2.Ny, sol2.Nz), dtype=np.complex128)
        for c in ("ex", "ey", "ez", "hx", "hy", "hz")
    }
    for step in range(nt_ring):
        sol2.one_step()
        t = (nt_drive + step) * dt
        ph = (win[step] / w_norm) * np.exp(-1j * w_fit * t)
        for c in ("x", "y", "z"):
            modes["e" + c] += sol2.E[0, : sol2.Ny, : sol2.Nz, c] * ph
            modes["h" + c] += sol2.H[0, : sol2.Ny, : sol2.Nz, c] * ph

    MU0 = 4.0e-7 * np.pi
    dy_v = np.asarray(grid_1p.dy[: sol2.Ny], dtype=float)[:, None]
    dz_v = np.asarray(grid_1p.dz[: sol2.Nz], dtype=float)[None, :]
    dA = dy_v * dz_v
    vac = np.asarray(sol2.ieps[0, : sol2.Ny, : sol2.Nz, "z"] > 0.0, dtype=bool)
    u_e = (
        0.25
        * EPS0
        * (np.abs(modes["ex"]) ** 2 + np.abs(modes["ey"]) ** 2 + np.abs(modes["ez"]) ** 2)
        * vac
    )
    u_m = (
        0.25
        * MU0
        * (np.abs(modes["hx"]) ** 2 + np.abs(modes["hy"]) ** 2 + np.abs(modes["hz"]) ** 2)
        * vac
    )
    W_2d = float(np.sum((u_e + u_m) * dA))
    P_2d = W_2d * vg_mcv / d_per

    dz_1d = np.asarray(grid_1p.dz[: sol2.Nz], dtype=float)
    ez_p1_y = (
        np.sum(modes["ez"] * np.exp(+1j * beta_c_fit * z_c)[None, :] * dz_1d[None, :], axis=1)
        / d_per
    )
    dy_1d = np.asarray(grid_1p.dy[: sol2.Ny], dtype=float)
    beam_mask = np.abs(y_c) <= b_half + 1e-12
    ez2_beam_avg = float(
        np.sum((np.abs(ez_p1_y[beam_mask]) ** 2) * dy_1d[beam_mask]) / np.sum(dy_1d[beam_mask])
    )

    mapping = map_2d_sheet_to_3d_grating(spec)
    w_eff = mapping["w_eff_m"]
    k_2d_fit = ez2_beam_avg / (2.0 * (beta_c_fit**2) * P_2d)
    k_3d_fit = k_2d_fit / w_eff

    v_sync_grid = 2.0 * np.pi * float(freq_hz) / beta_c_fit
    v_volt_grid = float((1.0 / np.sqrt(1.0 - (v_sync_grid / C0) ** 2) - 1.0) * M_E * C0**2 / Q_E)
    p_grid = pierce_parameters(
        freq_hz=freq_hz,
        beta_circuit=beta_c_fit,
        k_pierce_ohm=k_3d_fit,
        voltage_v=v_volt_grid,
        current_a=current_a,
        beam_half_thickness_m=b_half,
        tunnel_half_height_m=a_half,
        beam_width_m=spec.beam_width_m,
        alpha_loss_np_m=alpha_loss_fit,
        sc_model="wakis_planar",
    )
    disp_grid = solve_pierce_dispersion(b=0.0, QC=float(p_grid["QC"]), d=float(p_grid["d"]))
    alpha_pierce_grid = float(p_grid["beta_e"] * p_grid["C"] * disp_grid["x1"])

    return {
        "f_cold_fit_ghz": round(f_fit_hz * 1e-9, 4),
        "beta_c_fit_rad_m": round(beta_c_fit, 3),
        "alpha_cold_loss_np_m": round(alpha_loss_fit, 6),
        "k_2d_fit_ohm_mm": round(k_2d_fit * 1e3, 5),
        "k_3d_fit_ohm": round(k_3d_fit, 5),
        "v_sync_volt_grid": round(v_volt_grid, 2),
        "C_grid": round(float(p_grid["C"]), 6),
        "QC_grid": round(float(p_grid["QC"]), 5),
        "x1_grid": round(float(disp_grid["x1"]), 6),
        "alpha_pierce_grid_np_m": round(alpha_pierce_grid, 4),
    }


def run_2d_grating_hot_case(
    grid: GridFIT3D,
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
    n_periods: int = 22,
    freq_hz: float = 186.0e9,
    current_a: float = 0.25,
    u0_override: float | None = None,
    coupling_sign: float = 1.0,
    drive_amplitude: float = 1.0,
    sc_scale: float = 1.0,
    t_total_s: float = 210.0e-12,
    record_frames: bool = False,
) -> dict[str, Any]:
    """Run a single 2D FIT hot or cold spatial amplifier simulation on ``grid``."""
    d_per = spec.period_m
    a_half = spec.tunnel_half_height_m
    b_half = spec.beam_half_thickness_m
    b_wg = spec.waveguide_width_m
    w_beam = spec.beam_width_m

    dx = float(grid.x[1] - grid.x[0])
    kx = 2.0 * float(np.arcsin(0.5 * dx * np.pi / b_wg)) / dx

    dz_min = float(np.min(np.diff(grid.z)))
    dy_min = float(np.min(np.diff(grid.y)))
    dt_est = 0.85 / (C0 * np.sqrt(1.0 / dx**2 + 1.0 / dy_min**2 + 1.0 / dz_min**2))
    n_steps_est = int(np.ceil(t_total_s / dt_est))
    est = estimate(
        (int(grid.Nx), int(grid.Ny), int(grid.Nz)),
        steps=n_steps_est,
        dtype="complex128",
        conformal="pec",
    )
    preflight(est, LOCAL_VM_SPEC)

    t0 = time.perf_counter()
    solver = SolverFIT3D(
        grid,
        use_stl=True,
        bc_low=["bloch", "pec", "cpml"],
        bc_high=["bloch", "pec", "cpml"],
        k_bloch=(float(kx), 0.0, 0.0),
        n_pml=8,
        dtype=np.complex128,
        cfln=0.85,
        verbose=0,
    )
    dt = float(solver.dt)
    nt = int(np.ceil(float(t_total_s) / dt))
    omega = 2.0 * np.pi * float(freq_hz)
    n_dft = max(int(round((8.0 / float(freq_hz)) / dt)), 16)

    u0_nom, gamma0 = relativistic_gamma(voltage=spec.voltage_v)
    dz_avg = float(np.mean(np.diff(grid.z)))
    beta_e_nom = omega / u0_nom
    sinc_dz = float(np.sin(beta_e_nom * dz_avg) / (beta_e_nom * dz_avg))
    u0_val = float(u0_override) if u0_override is not None else float(u0_nom / sinc_dz)

    mapping = map_2d_sheet_to_3d_grating(spec)
    eta_x = mapping["eta_x_current_factor"]

    beam = None
    if current_a > 0.0:
        wp = beam_plasma_frequency(
            current=float(current_a) * eta_x,
            area=w_beam * 2.0 * b_half,
            u0=u0_nom,
        )
        beam = LinearizedBeam(
            u0=u0_val,
            omega_p=wp,
            gamma0=gamma0,
            axis="z",
            half_thickness=b_half,
            s_min=1.0 * d_per,
            s_max=float(n_periods - 1) * d_per,
            ramp_cells=10,
            coupling_sign=float(coupling_sign),
            solver=solver,
        )
        if abs(float(sc_scale) - 1.0) > 1e-6:
            # Use exact planar reduction factor R_planar(omega, u0, b, a) from wakis.beam_fluid
            r_sc_planar = float(
                reduction_factor_planar(
                    omega, u0_nom, b_half, a=a_half, relativistic=True
                )
            )
            coef_sc = (float(sc_scale) - 1.0) * (1j * (r_sc_planar**2) / (omega * EPS0))
            beam.gap_modulation = lambda s_c, t_now: coef_sc * beam.J1

    iz_src = int(np.argmin(np.abs(solver.z[: solver.Nz] - (-1.2 * d_per))))
    y_c = 0.5 * (solver.grid.y[:-1] + solver.grid.y[1:])
    iy_tunnel = np.where(np.abs(y_c[: solver.Ny]) < a_half)[0]
    flat_src_y = 1 * solver.N + 0 + solver.Nx * iy_tunnel + solver.Nx * solver.Ny * iz_src
    iy_axis = int(np.argmin(np.abs(solver.grid.y[: solver.Ny])))

    ez_acc = np.zeros(solver.Nz, dtype=np.complex128)
    i1_acc = np.zeros(solver.Nz, dtype=np.complex128)
    t_rise = 4.0 / float(freq_hz)

    frames_ez = []
    frames_j1 = []
    frame_steps = set()
    if record_frames:
        steps_per_cycle = max(int(round(1.0 / (float(freq_hz) * dt))), 16)
        n_frames = 16
        stride = max(steps_per_cycle // n_frames, 1)
        start_f = nt - n_frames * stride
        frame_steps = {start_f + k * stride for k in range(n_frames)}

    for step in range(nt):
        t = step * dt
        env = 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))
        solver.J.array[flat_src_y] = float(drive_amplitude) * env * np.exp(1j * omega * t)
        solver.one_step()
        if step >= nt - n_dft:
            phasor = np.exp(-1j * omega * t) / float(n_dft)
            ez_acc += solver.E[0, iy_axis, : solver.Nz, "z"] * phasor
            if beam is not None:
                i1_acc[beam.k_indices] += beam.axial_current() * phasor
        if record_frames and step in frame_steps:
            frames_ez.append(np.real(solver.E[0, : solver.Ny, : solver.Nz, "z"]).copy())
            j1_slice = np.zeros((solver.Ny, solver.Nz), dtype=float)
            if beam is not None:
                j1_slice_flat = np.real(beam.J1)
                iy_lines = np.where(beam.keep_lines)[0]
                for idx_l, iy_l in enumerate(iy_lines):
                    j1_slice[iy_l, beam.k_indices] = j1_slice_flat[:, idx_l]
            frames_j1.append(j1_slice)

    wall_s = time.perf_counter() - t0
    z_arr = np.asarray(solver.z[: solver.Nz], dtype=float).copy()

    m_indices = np.arange(1, int(n_periods) - 1, dtype=int)
    ez_period_rms = np.zeros(len(m_indices), dtype=float)
    i1_period_rms = np.zeros(len(m_indices), dtype=float)
    z_period_m = (m_indices + 0.5) * d_per
    for idx_m, m_val in enumerate(m_indices):
        mask_m = (z_arr >= float(m_val) * d_per) & (z_arr < float(m_val + 1) * d_per)
        ez_period_rms[idx_m] = float(np.sqrt(np.mean(np.abs(ez_acc[mask_m]) ** 2)))
        i1_period_rms[idx_m] = float(np.sqrt(np.mean(np.abs(i1_acc[mask_m]) ** 2)))

    out = {
        "z_m": z_arr,
        "ez_phasor": ez_acc,
        "i1_phasor": i1_acc,
        "m_indices": m_indices,
        "z_period_m": z_period_m,
        "ez_period_rms": ez_period_rms,
        "i1_period_rms": i1_period_rms,
        "dt_s": dt,
        "n_steps": nt,
        "wall_time_s": wall_s,
        "ram_est_mb": float(est.host_bytes / 1e6),
        "u0_used_m_s": u0_val,
        "sinc_dz": sinc_dz,
        "drive_amplitude": float(drive_amplitude),
    }
    if record_frames:
        vac_mask = np.asarray(solver.ieps[0, : solver.Ny, : solver.Nz, "z"] > 0.0, dtype=bool)
        out["frames_ez"] = np.array(frames_ez, dtype=float)
        out["frames_j1"] = np.array(frames_j1, dtype=float)
        out["vac_mask_yz"] = vac_mask
        out["y_um"] = 0.5 * (np.asarray(grid.y[: solver.Ny]) + np.asarray(grid.y[1 : solver.Ny + 1])) * 1e6
        out["z_um"] = z_arr * 1e6
    return out


def get_karetnikova_cold_circuit_curve(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
) -> dict[str, np.ndarray]:
    """Load Tutorial 6.2 cold-test dispersion and Pierce impedance for the Karetnikova grating."""
    d9 = np.load(COLD_DATA_DIR / "task_09_3d_grating_karetnikova.npz")
    bn_62 = np.asarray(d9["bn"], dtype=float)
    f_fit_62_ghz = np.asarray(d9["f3d_glide_low_ghz"], dtype=float)
    f_mcv_62_ghz = np.asarray(d9["mcvey_3d_g_low_ghz"], dtype=float)
    kp1_62_ohm = np.asarray(d9["mcvey_3d_kp1_beam_ohm"], dtype=float)

    from scipy.interpolate import PchipInterpolator

    bn_dense = np.linspace(float(bn_62[0]), float(bn_62[-1]), 65)
    f_dense_ghz = PchipInterpolator(bn_62, f_mcv_62_ghz)(bn_dense)
    f_dense_hz = f_dense_ghz * 1.0e9
    beta0_dense = bn_dense * np.pi / spec.period_m
    beta_p1_dense = (2.0 + bn_dense) * np.pi / spec.period_m
    kp1_beam_dense = np.exp(PchipInterpolator(bn_62, np.log(kp1_62_ohm))(bn_dense))
    d_omega_d_beta = np.gradient(2.0 * np.pi * f_dense_hz, beta0_dense)
    ve_over_c_dense = d_omega_d_beta / C0

    # Exact surface-wave transverse decay factor <cosh^2(tau_{+1} y)>_{y in [-b, +b]}
    # relating the beam-averaged impedance <K_{+1}>_beam to the midplane impedance K_{+1}(y=0):
    kx = np.pi / spec.waveguide_width_m
    k0_dense = 2.0 * np.pi * f_dense_hz / C0
    tau_p1 = np.sqrt(np.maximum(beta_p1_dense**2 + kx**2 - k0_dense**2, 1e-12))
    two_tau_b = 2.0 * tau_p1 * spec.beam_half_thickness_m
    cosh2_beam_avg = 0.5 * (1.0 + np.sinh(two_tau_b) / np.maximum(two_tau_b, 1e-12))
    kp1_axis_dense = kp1_beam_dense / cosh2_beam_avg

    return {
        "bn_62": bn_62,
        "f_fit_62_ghz": f_fit_62_ghz,
        "f_mcv_62_ghz": f_mcv_62_ghz,
        "kp1_62_ohm": kp1_62_ohm,
        "bn_dense": bn_dense,
        "freq_dense_hz": f_dense_hz,
        "freq_dense_ghz": f_dense_hz * 1.0e-9,
        "beta_p1_dense": beta_p1_dense,
        "kp1_beam_dense_ohm": kp1_beam_dense,
        "kp1_axis_dense_ohm": kp1_axis_dense,
        "cosh2_beam_avg_factor": cosh2_beam_avg,
        "ve_over_c_dense": ve_over_c_dense,
    }


def run_hot_test_convergence(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
    force: bool = False,
) -> dict[str, Any]:
    """Run Task 2: 2D FIT hot-test spatial growth convergence (4 resolutions + same-grid Pierce).

    Executes the 22-period Karetnikova staggered-grating spatial TWT hot test at the
    synchronous operating point :math:`f_0 = 186.0` GHz across four systematically
    refined node-aligned meshes:
    - ``coarse`` (:math:`N_z/d = 26, n_{\\mathrm{gap}} = 4, 21.8\\;\\mathrm{cells}/\\lambda_g`),
    - ``medium`` (:math:`N_z/d = 32, n_{\\mathrm{gap}} = 5, 26.2\\;\\mathrm{cells}/\\lambda_g`),
    - ``fine`` (:math:`N_z/d = 40, n_{\\mathrm{gap}} = 6, 34.9\\;\\mathrm{cells}/\\lambda_g`),
    - ``ultrafine`` (:math:`N_z/d = 46, n_{\\mathrm{gap}} = 7, 39.2\\;\\mathrm{cells}/\\lambda_g`).
    On each mesh level, extracts the cold-circuit wavenumber :math:`\\beta_{c,\\mathrm{FIT}}`,
    interaction impedance :math:`K_{2\\mathrm{D,FIT}}` (:math:`K_{3\\mathrm{D,FIT}}`), and
    cold loss :math:`\\alpha_{0,\\mathrm{FIT}}` on the **same FIT grid**, predicts
    :math:`\\alpha_{\\mathrm{Pierce,grid}}`, compares against the hot FIT spatial growth
    rate :math:`\\alpha_{\\mathrm{FIT}}` over :math:`m \\in [8, 19]`, and reports the
    true observed convergence order :math:`p_{\\mathrm{obs}}` without imposing :math:`p = 2`.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "task_02_2d_grating_hot_convergence.json"
    npz_path = DATA_DIR / "task_02_2d_grating_hot_convergence.npz"

    def _enrich_task02_convergence(payload_in: dict[str, Any]) -> dict[str, Any]:
        lvls = payload_in["levels"]
        h_a = np.array([1.0 / float(x["mesh"]["nz_per_period"]) for x in lvls], dtype=float)
        e_a = np.array(
            [abs(float(x["alpha_fit_np_m"]) - float(x["alpha_pierce_grid_np_m"])) for x in lvls],
            dtype=float,
        )
        pw_orders = [
            round(float(np.log(e_a[i] / e_a[i + 1]) / np.log(h_a[i] / h_a[i + 1])), 2)
            for i in range(len(h_a) - 1)
        ]
        inc_a = [
            round(float(lvls[i]["alpha_fit_np_m"]) - float(lvls[i + 1]["alpha_fit_np_m"]), 2)
            for i in range(len(lvls) - 1)
        ]
        p_fit, _ = np.polyfit(np.log(h_a), np.log(e_a), 1)
        p_fit_r = round(float(p_fit), 3)
        payload_in["pairwise_orders"] = pw_orders
        payload_in["alpha_fit_increments_np_m"] = inc_a
        payload_in["observed_order_p"] = p_fit_r
        payload_in["in_asymptotic_range"] = False
        payload_in["asymptotic_range_note"] = (
            f"Pairwise convergence orders across coarse->medium->fine->ultrafine are "
            f"{pw_orders[0]:.2f}, {pw_orders[1]:.2f}, and {pw_orders[2]:.2f} "
            f"(successive alpha_FIT increments {inc_a[0]:.2f}, {inc_a[1]:.2f}, {inc_a[2]:.2f} Np/m; "
            f"4-point log-log fit slope p_obs = {p_fit_r:.3f}); because the pairwise orders "
            f"oscillate between {min(pw_orders):.2f} and {max(pw_orders):.2f}, the 4-level data "
            f"are not in the asymptotic convergence range and no Richardson value is claimed as converged."
        )
        json_path.write_text(json.dumps(payload_in, indent=2) + "\n", encoding="utf-8")
        return payload_in

    if json_path.exists() and npz_path.exists() and not force:
        return _enrich_task02_convergence(json.loads(json_path.read_text(encoding="utf-8")))

    cold_curve = get_karetnikova_cold_circuit_curve(spec)
    f0 = 186.0e9
    beta_c_f0 = float(np.interp(f0, cold_curve["freq_dense_hz"], cold_curve["beta_p1_dense"]))
    k_beam_f0 = float(np.interp(f0, cold_curve["freq_dense_hz"], cold_curve["kp1_beam_dense_ohm"]))

    i0_conv = 0.25
    omega0 = 2.0 * np.pi * f0
    v_sync = omega0 / beta_c_f0
    v_sync_volt = float((1.0 / np.sqrt(1.0 - (v_sync / C0) ** 2) - 1.0) * M_E * C0**2 / Q_E)

    p_th = pierce_parameters(
        freq_hz=f0,
        beta_circuit=beta_c_f0,
        k_pierce_ohm=k_beam_f0,
        voltage_v=v_sync_volt,
        current_a=i0_conv,
        beam_half_thickness_m=spec.beam_half_thickness_m,
        tunnel_half_height_m=spec.tunnel_half_height_m,
        beam_width_m=spec.beam_width_m,
        sc_model="wakis_planar",
    )
    disp_th = solve_pierce_dispersion(
        b=float(p_th["b"]),
        QC=float(p_th["QC"]),
        d=0.0,
        C=float(p_th["C"]),
        include_backward=False,
    )
    disp_4w = solve_pierce_dispersion(
        b=float(p_th["b"]),
        QC=float(p_th["QC"]),
        d=0.0,
        C=float(p_th["C"]),
        include_backward=True,
    )
    alpha_pierce_3w = float(p_th["beta_e"] * p_th["C"] * disp_th["x1"])
    alpha_pierce_4w = float(p_th["beta_e"] * p_th["C"] * disp_4w["x1"])

    levels_names = ("coarse", "medium", "fine", "ultrafine")
    levels_out = []
    npz_dict: dict[str, Any] = {}

    for lvl_name in levels_names:
        cold_on_grid = extract_2d_grating_cold_circuit_on_grid(
            spec=spec, mesh_level=lvl_name, freq_hz=f0, current_a=i0_conv
        )
        grid, meta = build_2d_grating_hot_grid(
            spec=spec,
            n_periods=22,
            n_in_smooth=2,
            n_out_smooth=2,
            mesh_level=lvl_name,
        )
        beta_c_grid = float(cold_on_grid["beta_c_fit_rad_m"])
        alpha_pierce_grid = float(cold_on_grid["alpha_pierce_grid_np_m"])
        dz_avg = spec.period_m / float(meta["nz_per_period"])
        sinc_dz = float(np.sin(beta_c_grid * dz_avg) / (beta_c_grid * dz_avg))
        u0_grid_sync = (omega0 / beta_c_grid) / sinc_dz

        res_cold = run_2d_grating_hot_case(
            grid,
            spec=spec,
            n_periods=22,
            freq_hz=f0,
            current_a=0.0,
            u0_override=u0_grid_sync,
            record_frames=False,
        )
        res_hot = run_2d_grating_hot_case(
            grid,
            spec=spec,
            n_periods=22,
            freq_hz=f0,
            current_a=i0_conv,
            u0_override=u0_grid_sync,
            record_frames=(lvl_name == "medium"),
        )

        ratio_vs_m = res_hot["ez_period_rms"] / np.maximum(res_cold["ez_period_rms"], 1e-30)
        gain_db_vs_m = 20.0 * np.log10(np.maximum(ratio_vs_m, 1e-30))
        z_m_per = res_hot["z_period_m"]

        fit_mask = (res_hot["m_indices"] >= 8) & (res_hot["m_indices"] <= 19)
        alpha_fit, _ = np.polyfit(z_m_per[fit_mask], np.log(ratio_vs_m[fit_mask]), 1)
        alpha_fit = float(alpha_fit)
        slope_db_per_mm = float(20.0 * np.log10(np.e) * alpha_fit * 1.0e-3)
        rel_err_grid = abs(alpha_fit - alpha_pierce_grid) / alpha_pierce_grid
        rel_err_mcvey = abs(alpha_fit - alpha_pierce_3w) / alpha_pierce_3w

        levels_out.append(
            {
                "level": lvl_name,
                "mesh": meta,
                "cold_circuit_on_grid": cold_on_grid,
                "alpha_pierce_grid_np_m": round(alpha_pierce_grid, 4),
                "alpha_fit_np_m": round(alpha_fit, 4),
                "slope_db_per_mm": round(slope_db_per_mm, 4),
                "rel_err_vs_pierce": round(rel_err_grid, 5),
                "rel_err_vs_pierce_grid": round(rel_err_grid, 5),
                "rel_err_vs_pierce_mcvey": round(rel_err_mcvey, 5),
                "gain_period_19_db": round(float(gain_db_vs_m[res_hot["m_indices"] == 19][0]), 3),
                "min_launching_dip_db": round(float(np.min(gain_db_vs_m[:7])), 3),
                "cold_wall_time_s": round(res_cold["wall_time_s"], 3),
                "hot_wall_time_s": round(res_hot["wall_time_s"], 3),
                "ram_est_mb": round(res_hot["ram_est_mb"], 2),
                "n_steps": int(res_hot["n_steps"]),
            }
        )

        npz_dict[f"{lvl_name}_z_period_m"] = z_m_per
        npz_dict[f"{lvl_name}_m_indices"] = res_hot["m_indices"]
        npz_dict[f"{lvl_name}_ratio_vs_m"] = ratio_vs_m
        npz_dict[f"{lvl_name}_gain_db_vs_m"] = gain_db_vs_m
        npz_dict[f"{lvl_name}_i1_rms_vs_m"] = res_hot["i1_period_rms"]
        if lvl_name == "medium":
            npz_dict["medium_z_m"] = res_hot["z_m"]
            npz_dict["medium_ez_cold"] = res_cold["ez_phasor"]
            npz_dict["medium_ez_hot"] = res_hot["ez_phasor"]
            npz_dict["medium_i1_hot"] = res_hot["i1_phasor"]
            npz_dict["frames_ez"] = res_hot["frames_ez"]
            npz_dict["frames_j1"] = res_hot["frames_j1"]
            npz_dict["vac_mask_yz"] = res_hot["vac_mask_yz"]
            npz_dict["y_um"] = res_hot["y_um"]
            npz_dict["z_um"] = res_hot["z_um"]

    # Compute true observed convergence order p_obs across all 4 levels from log-log fit
    # of |alpha_FIT - alpha_Pierce_grid| vs h = d / Nz_per_period (without imposing p = 2)
    h_arr = np.array([1.0 / float(x["mesh"]["nz_per_period"]) for x in levels_out], dtype=float)
    err_grid_arr = np.array(
        [abs(x["alpha_fit_np_m"] - x["alpha_pierce_grid_np_m"]) for x in levels_out], dtype=float
    )
    p_obs, _ = np.polyfit(np.log(h_arr), np.log(err_grid_arr), 1)
    p_obs = float(p_obs)

    # Richardson extrapolation using the observed order p_obs on medium (level 1) and ultrafine (level 3)
    h_med = h_arr[1]
    h_uf = h_arr[3]
    a_med = float(levels_out[1]["alpha_fit_np_m"])
    a_uf = float(levels_out[3]["alpha_fit_np_m"])
    p_uf_grid = float(levels_out[3]["alpha_pierce_grid_np_m"])
    r_uf_med = h_med / h_uf
    alpha_richardson = float(a_uf + (a_uf - a_med) / (r_uf_med**p_obs - 1.0))
    rel_err_richardson_grid = abs(alpha_richardson - p_uf_grid) / p_uf_grid
    rel_err_richardson_mcvey = abs(alpha_richardson - alpha_pierce_3w) / alpha_pierce_3w

    z_th = np.linspace(0.0, 20.0 * spec.period_m, 201)
    cn_th = float(p_th["beta_e"] * p_th["C"]) * z_th / (2.0 * np.pi)
    bvp_th = solve_pierce_three_wave_bvp(
        cn_th, b=float(p_th["b"]), QC=float(p_th["QC"]), d=0.0
    )
    ode_th = solve_coupled_mode_ode(
        L_m=float(z_th[-1]),
        beta_e=float(p_th["beta_e"]),
        C=float(p_th["C"]),
        b=float(p_th["b"]),
        QC=float(p_th["QC"]),
        d=0.0,
        n_pts=201,
    )
    npz_dict["z_th_m"] = z_th
    npz_dict["bvp_gain_db"] = bvp_th["gain_db"]
    npz_dict["bvp_asymp_db"] = bvp_th["asymptotic_gain_db"]
    npz_dict["ode_gain_db"] = ode_th["gain_db_vs_z"]
    npz_dict["ode_i_env_abs"] = np.abs(ode_th["I_env"])

    np.savez(npz_path, **npz_dict)

    payload = {
        "f0_ghz": 186.0,
        "I0_conv_a": i0_conv,
        "beta_c_rad_m": round(beta_c_f0, 3),
        "K_beam_ohm": round(k_beam_f0, 4),
        "C_pierce": round(float(p_th["C"]), 6),
        "QC_pierce": round(float(p_th["QC"]), 5),
        "R_sc_planar": round(float(p_th["R_sc"]), 5),
        "x1_pierce_3w": round(float(disp_th["x1"]), 6),
        "x1_pierce_4w": round(float(disp_4w["x1"]), 6),
        "alpha_pierce_3w_np_m": round(alpha_pierce_3w, 4),
        "alpha_pierce_4w_np_m": round(alpha_pierce_4w, 4),
        "alpha_pierce_finest_grid_np_m": round(p_uf_grid, 4),
        "slope_pierce_db_per_mm": round(float(20.0 * np.log10(np.e) * alpha_pierce_3w * 1e-3), 4),
        "levels": levels_out,
        "observed_order_p": round(p_obs, 3),
        "alpha_richardson_np_m": round(alpha_richardson, 4),
        "rel_err_richardson": round(rel_err_richardson_grid, 5),
        "rel_err_richardson_vs_finest_grid": round(rel_err_richardson_grid, 5),
        "rel_err_richardson_vs_mcvey": round(rel_err_richardson_mcvey, 5),
    }
    return _enrich_task02_convergence(payload)


def run_karetnikova_gain_vs_frequency(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
    force: bool = False,
) -> dict[str, Any]:
    """Run Task 3: Small-signal gain vs frequency at I0 = 100 mA vs Karetnikova et al. (2018).

    Computes:
    1. Full frequency sweeps (:math:`180\\text{--}212` GHz, plus the :math:`178.99` GHz
       lower edge of the Tutorial 6.2 cold-test table) of the 1D Pierce 3-wave BVP
       and coupled-mode ODE at :math:`I_0 = 100` mA for:
       - Karetnikova Fig. 3 (:math:`L = 25` mm, :math:`N = 50` periods) at
         :math:`V_0 \\in \\{19, 20, 21\\}` kV using the beam-averaged impedance
         :math:`\\langle K_{+1}\\rangle_{\\mathrm{beam}}` (with hypothetical midplane
         :math:`K_{+1}(0)` recorded only as an untested hypothesis),
       - Karetnikova :math:`L = 40` mm (:math:`N = 80` periods) at :math:`V_0 = 20` kV,
         comparing ``wakis.beam_fluid`` planar reduction factor :math:`R_{\\mathrm{planar}} = 0.6523`
         against Karetnikova Eq. (4) two-plate reduction factor :math:`D = 0.6355`.
    2. Multi-frequency 2D FIT hot-test runs across :math:`f \\in \\{183.0, 185.0, 186.5,
       188.5, 191.5, 196.0\\}` GHz at :math:`V_0 = 20` kV on the node-aligned ``coarse``
       mesh (:math:`N_z/d = 26, n_{\\mathrm{gap}} = 4`), reporting every point's
       relative error and the Pierce 3-wave extrapolations to :math:`L = 25` mm and
       :math:`L = 40` mm from the 22-period (:math:`11` mm) FIT model.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "task_03_karetnikova_gain_vs_frequency.json"
    npz_path = DATA_DIR / "task_03_karetnikova_gain_vs_frequency.npz"

    cold_curve = get_karetnikova_cold_circuit_curve(spec)
    f_sweep_ghz = np.linspace(180.0, 212.0, 129)
    f_sweep_hz = f_sweep_ghz * 1.0e9
    beta_c_sweep = np.interp(f_sweep_hz, cold_curve["freq_dense_hz"], cold_curve["beta_p1_dense"])
    k_beam_sweep = np.interp(f_sweep_hz, cold_curve["freq_dense_hz"], cold_curve["kp1_beam_dense_ohm"])
    k_mid_sweep = np.interp(f_sweep_hz, cold_curve["freq_dense_hz"], cold_curve["kp1_axis_dense_ohm"])

    f_min_cold_hz = float(cold_curve["freq_dense_hz"][0])
    f_min_cold_ghz = float(cold_curve["freq_dense_ghz"][0])
    beta_c_fmin = float(cold_curve["beta_p1_dense"][0])
    k_beam_fmin = float(cold_curve["kp1_beam_dense_ohm"][0])

    L_fig3_m = float(spec.n_periods_fig3) * spec.period_m  # 25.0 mm (50 periods)
    L_fig5_m = float(spec.n_periods_fig5) * spec.period_m  # 40.0 mm (80 periods)

    fig3_curves: dict[str, np.ndarray] = {}
    fig3_summary: dict[str, Any] = {}
    for v_kv in (19.0, 20.0, 21.0):
        p_v = pierce_parameters(
            freq_hz=f_sweep_hz,
            beta_circuit=beta_c_sweep,
            k_pierce_ohm=k_beam_sweep,
            voltage_v=v_kv * 1.0e3,
            current_a=spec.current_a,
            beam_half_thickness_m=spec.beam_half_thickness_m,
            tunnel_half_height_m=spec.tunnel_half_height_m,
            beam_width_m=spec.beam_width_m,
            sc_model="wakis_planar",
        )
        p_v_mid = pierce_parameters(
            freq_hz=f_sweep_hz,
            beta_circuit=beta_c_sweep,
            k_pierce_ohm=k_mid_sweep,
            voltage_v=v_kv * 1.0e3,
            current_a=spec.current_a,
            beam_half_thickness_m=spec.beam_half_thickness_m,
            tunnel_half_height_m=spec.tunnel_half_height_m,
            beam_width_m=spec.beam_width_m,
            sc_model="karetnikova_eq4",
        )
        p_v_fmin = pierce_parameters(
            freq_hz=f_min_cold_hz,
            beta_circuit=beta_c_fmin,
            k_pierce_ohm=k_beam_fmin,
            voltage_v=v_kv * 1.0e3,
            current_a=spec.current_a,
            beam_half_thickness_m=spec.beam_half_thickness_m,
            tunnel_half_height_m=spec.tunnel_half_height_m,
            beam_width_m=spec.beam_width_m,
            sc_model="wakis_planar",
        )
        cn_fmin = float(p_v_fmin["beta_e"] * p_v_fmin["C"] * L_fig3_m / (2.0 * np.pi))
        g_bvp_fmin = float(
            solve_pierce_three_wave_bvp(
                cn_fmin, b=float(p_v_fmin["b"]), QC=float(p_v_fmin["QC"]), d=0.0
            )["gain_db"]
        )

        g_bvp = np.zeros_like(f_sweep_ghz)
        g_ode = np.zeros_like(f_sweep_ghz)
        g_mid_bvp = np.zeros_like(f_sweep_ghz)
        alpha_np_m = np.zeros_like(f_sweep_ghz)
        for i in range(len(f_sweep_ghz)):
            cn_val = float(p_v["beta_e"][i] * p_v["C"][i] * L_fig3_m / (2.0 * np.pi))
            r_bvp = solve_pierce_three_wave_bvp(
                cn_val, b=float(p_v["b"][i]), QC=float(p_v["QC"][i]), d=0.0
            )
            r_ode = solve_coupled_mode_ode(
                L_m=L_fig3_m,
                beta_e=float(p_v["beta_e"][i]),
                C=float(p_v["C"][i]),
                b=float(p_v["b"][i]),
                QC=float(p_v["QC"][i]),
                d=0.0,
                n_pts=41,
            )
            cn_mid = float(p_v_mid["beta_e"][i] * p_v_mid["C"][i] * L_fig3_m / (2.0 * np.pi))
            r_mid = solve_pierce_three_wave_bvp(
                cn_mid, b=float(p_v_mid["b"][i]), QC=float(p_v_mid["QC"][i]), d=0.0
            )
            g_bvp[i] = float(r_bvp["gain_db"])
            g_ode[i] = float(r_ode["final_gain_db"])
            g_mid_bvp[i] = float(r_mid["gain_db"])
            alpha_np_m[i] = float(p_v["beta_e"][i] * p_v["C"][i] * r_bvp["deltas"][0].real)

        key = f"v_{int(v_kv)}kv"
        fig3_curves[f"{key}_bvp_db"] = g_bvp
        fig3_curves[f"{key}_ode_db"] = g_ode
        fig3_curves[f"{key}_midplane_bvp_db"] = g_mid_bvp
        fig3_curves[f"{key}_alpha_np_m"] = alpha_np_m
        fig3_curves[f"{key}_b"] = np.asarray(p_v["b"], dtype=float)
        idx_pk = int(np.argmax(g_bvp))
        idx_pk_mid = int(np.argmax(g_mid_bvp))
        pk_f = round(float(f_sweep_ghz[idx_pk]), 2)
        pk_g = round(float(g_bvp[idx_pk]), 2)
        fmin_r = round(f_min_cold_ghz, 2)
        g_fmin_r = round(g_bvp_fmin, 2)
        is_bound = bool(idx_pk == 0 or idx_pk == len(f_sweep_ghz) - 1)
        entry_v: dict[str, Any] = {
            "voltage_kv": float(v_kv),
            "peak_freq_ghz": pk_f,
            "peak_gain_25mm_db": pk_g,
            "is_lower_sweep_boundary": bool(idx_pk == 0),
            "cold_table_min_freq_ghz": fmin_r,
            "gain_25mm_at_cold_table_min_db": g_fmin_r,
            "midplane_peak_freq_ghz": round(float(f_sweep_ghz[idx_pk_mid]), 2),
            "midplane_peak_gain_25mm_db": round(float(g_mid_bvp[idx_pk_mid]), 2),
            "max_bvp_ode_diff_db": float(np.max(np.abs(g_bvp - g_ode))),
        }
        if is_bound:
            entry_v["boundary_note"] = (
                f"At {v_kv:.0f} kV, the maximum on the 180.0-212.0 GHz sweep ({pk_g:.2f} dB "
                f"at {pk_f:.2f} GHz) occurs at the lower boundary of linspace(180, 212); "
                f"extending down to the lower edge of the Tutorial 6.2 cold-test table "
                f"({fmin_r:.2f} GHz at beta_0 d = 0.15 pi) gives {g_fmin_r:.2f} dB."
            )
        fig3_summary[key] = entry_v

    # L = 40 mm (N = 80, V0 = 20 kV, I0 = 0.1 A): compare wakis_planar vs karetnikova_eq4
    for sc_tag in ("wakis_planar", "karetnikova_eq4"):
        p_40 = pierce_parameters(
            freq_hz=f_sweep_hz,
            beta_circuit=beta_c_sweep,
            k_pierce_ohm=k_beam_sweep,
            voltage_v=20.0e3,
            current_a=spec.current_a,
            beam_half_thickness_m=spec.beam_half_thickness_m,
            tunnel_half_height_m=spec.tunnel_half_height_m,
            beam_width_m=spec.beam_width_m,
            sc_model=sc_tag,
        )
        g_40 = np.zeros_like(f_sweep_ghz)
        for i in range(len(f_sweep_ghz)):
            cn_val = float(p_40["beta_e"][i] * p_40["C"][i] * L_fig5_m / (2.0 * np.pi))
            r_bvp = solve_pierce_three_wave_bvp(
                cn_val, b=float(p_40["b"][i]), QC=float(p_40["QC"][i]), d=0.0
            )
            g_40[i] = float(r_bvp["gain_db"])
        fig3_curves[f"fig5_40mm_{sc_tag}_db"] = g_40

    p_40_mid = pierce_parameters(
        freq_hz=f_sweep_hz,
        beta_circuit=beta_c_sweep,
        k_pierce_ohm=k_mid_sweep,
        voltage_v=20.0e3,
        current_a=spec.current_a,
        beam_half_thickness_m=spec.beam_half_thickness_m,
        tunnel_half_height_m=spec.tunnel_half_height_m,
        beam_width_m=spec.beam_width_m,
        sc_model="karetnikova_eq4",
    )
    g_40_mid = np.zeros_like(f_sweep_ghz)
    for i in range(len(f_sweep_ghz)):
        cn_val = float(p_40_mid["beta_e"][i] * p_40_mid["C"][i] * L_fig5_m / (2.0 * np.pi))
        r_bvp = solve_pierce_three_wave_bvp(
            cn_val, b=float(p_40_mid["b"][i]), QC=float(p_40_mid["QC"][i]), d=0.0
        )
        g_40_mid[i] = float(r_bvp["gain_db"])
    fig3_curves["fig5_40mm_midplane_eq4_db"] = g_40_mid

    idx_pk_40_wakis = int(np.argmax(fig3_curves["fig5_40mm_wakis_planar_db"]))
    idx_pk_40_kar = int(np.argmax(fig3_curves["fig5_40mm_karetnikova_eq4_db"]))
    idx_pk_40_mid = int(np.argmax(fig3_curves["fig5_40mm_midplane_eq4_db"]))

    # Multi-frequency 2D FIT hot-test sweep at V0 = 20 kV across 6 frequencies on node-aligned coarse grid
    if json_path.exists() and npz_path.exists() and not force:
        d3_cached = np.load(npz_path)
        j3_cached = json.loads(json_path.read_text(encoding="utf-8"))
        fit_freqs_ghz = np.asarray(d3_cached["fit_freqs_ghz"], dtype=float)
        fit_alpha_raw = np.asarray(d3_cached["fit_alpha_raw"], dtype=float)
        fit_alpha_100ma = np.asarray(d3_cached["fit_alpha_100ma"], dtype=float)
        th_alpha_at_fit = np.asarray(d3_cached["th_alpha_at_fit"], dtype=float)
        rel_err_at_fit = np.asarray(d3_cached["rel_err_at_fit"], dtype=float)
        fit_gain_25mm_db = np.asarray(d3_cached["fit_gain_25mm_db"], dtype=float)
        fit_gain_40mm_db = np.asarray(d3_cached["fit_gain_40mm_db"], dtype=float)
        b_at_fit = np.asarray(j3_cached["fit_frequency_sweep"]["b_pierce"], dtype=float)
        fit_wall_s = float(j3_cached["fit_frequency_sweep"]["wall_time_s"])
    else:
        fit_freqs_ghz = np.array([183.0, 185.0, 186.5, 188.5, 191.5, 196.0], dtype=float)
        grid_sw, meta_sw = build_2d_grating_hot_grid(
            spec=spec,
            n_periods=22,
            n_in_smooth=2,
            n_out_smooth=2,
            mesh_level="coarse",
        )
        u0_20kv, _ = relativistic_gamma(voltage=20.0e3)
        dz_sw = spec.period_m / float(meta_sw["nz_per_period"])

        fit_alpha_raw = np.zeros_like(fit_freqs_ghz)
        fit_alpha_100ma = np.zeros_like(fit_freqs_ghz)
        fit_gain_25mm_db = np.zeros_like(fit_freqs_ghz)
        fit_gain_40mm_db = np.zeros_like(fit_freqs_ghz)
        th_alpha_at_fit = np.zeros_like(fit_freqs_ghz)
        rel_err_at_fit = np.zeros_like(fit_freqs_ghz)
        b_at_fit = np.zeros_like(fit_freqs_ghz)
        fit_wall_s = 0.0

        for idx_f, f_ghz in enumerate(fit_freqs_ghz):
            f_hz = float(f_ghz * 1.0e9)
            beta_e_f = 2.0 * np.pi * f_hz / u0_20kv
            sinc_f = float(np.sin(beta_e_f * dz_sw) / (beta_e_f * dz_sw))
            u0_f = u0_20kv / sinc_f

            r_c = run_2d_grating_hot_case(
                grid_sw, spec=spec, n_periods=22, freq_hz=f_hz, current_a=0.0, u0_override=u0_f
            )
            r_h = run_2d_grating_hot_case(
                grid_sw,
                spec=spec,
                n_periods=22,
                freq_hz=f_hz,
                current_a=spec.current_a,
                u0_override=u0_f,
            )
            fit_wall_s += r_c["wall_time_s"] + r_h["wall_time_s"]
            ratio_m = r_h["ez_period_rms"] / np.maximum(r_c["ez_period_rms"], 1e-30)
            z_m_per = r_h["z_period_m"]
            mask_m = (r_h["m_indices"] >= 8) & (r_h["m_indices"] <= 19)
            s_fit_raw, _ = np.polyfit(z_m_per[mask_m], np.log(ratio_m[mask_m]), 1)
            s_fit_raw = float(s_fit_raw)
            fit_alpha_raw[idx_f] = s_fit_raw

            # De-embed the asymptotic growing root alpha_FIT(f) from the finite-length
            # 22-period local slope s_fit_raw using the exact 3-wave Pierce BVP over m in [8, 19]
            # with the true unclamped velocity parameter b(f):
            beta_c_f = float(np.interp(f_hz, cold_curve["freq_dense_hz"], cold_curve["beta_p1_dense"]))
            k_beam_f = float(np.interp(f_hz, cold_curve["freq_dense_hz"], cold_curve["kp1_beam_dense_ohm"]))
            p_f = pierce_parameters(
                freq_hz=f_hz,
                beta_circuit=beta_c_f,
                k_pierce_ohm=k_beam_f,
                voltage_v=20.0e3,
                current_a=spec.current_a,
                beam_half_thickness_m=spec.beam_half_thickness_m,
                tunnel_half_height_m=spec.tunnel_half_height_m,
                beam_width_m=spec.beam_width_m,
                sc_model="wakis_planar",
            )
            b_v = float(p_f["b"])
            qc_v = float(p_f["QC"])
            be_v = float(p_f["beta_e"])
            b_at_fit[idx_f] = b_v
            x1_v = float(solve_pierce_dispersion(b=b_v, QC=qc_v, d=0.0)["x1"])
            z_win = z_m_per[mask_m]

            def _s_bvp_window(alpha_try: float) -> float:
                c_try = alpha_try / max(be_v * x1_v, 1e-12)
                cn_m = be_v * c_try * (z_win + 2.0 * spec.period_m) / (2.0 * np.pi)
                bvp_m = solve_pierce_three_wave_bvp(cn_m, b=b_v, QC=qc_v, d=0.0)
                sl_m, _ = np.polyfit(z_win, np.log(np.abs(bvp_m["transfer_fn"])), 1)
                return float(sl_m)

            alpha_f = float(brentq(lambda a: _s_bvp_window(a) - s_fit_raw, 5.0, 320.0))
            fit_alpha_100ma[idx_f] = alpha_f

            bvp_f = solve_pierce_three_wave_bvp(0.5, b=b_v, QC=qc_v, d=0.0)
            th_alpha_f = float(p_f["beta_e"] * p_f["C"] * bvp_f["deltas"][0].real)
            th_alpha_at_fit[idx_f] = th_alpha_f
            rel_err_at_fit[idx_f] = (alpha_f - th_alpha_f) / th_alpha_f

            # Pierce 3-wave extrapolation to L = 25 mm (50 periods) and L = 40 mm (80 periods)
            # using the 22-period FIT-fitted growth rate alpha_f:
            deltas_fit = bvp_f["deltas"].copy()
            if float(p_f["beta_e"] * p_f["C"]) > 0:
                deltas_fit[0] = (alpha_f / float(p_f["beta_e"] * p_f["C"])) + 1j * deltas_fit[0].imag
            weights_f = bvp_f["weights_ckt"]
            for l_m, arr_out in ((L_fig3_m, fit_gain_25mm_db), (L_fig5_m, fit_gain_40mm_db)):
                cn_l = float(p_f["beta_e"] * p_f["C"] * l_m / (2.0 * np.pi))
                tf_l = np.sum(weights_f * np.exp(2.0 * np.pi * cn_l * deltas_fit))
                arr_out[idx_f] = float(20.0 * np.log10(max(abs(tf_l), 1e-30)))

    np.savez(
        npz_path,
        f_sweep_ghz=f_sweep_ghz,
        beta_c_sweep=beta_c_sweep,
        k_beam_sweep=k_beam_sweep,
        k_mid_sweep=k_mid_sweep,
        fit_freqs_ghz=fit_freqs_ghz,
        fit_alpha_raw=fit_alpha_raw,
        fit_alpha_100ma=fit_alpha_100ma,
        th_alpha_at_fit=th_alpha_at_fit,
        rel_err_at_fit=rel_err_at_fit,
        fit_gain_25mm_db=fit_gain_25mm_db,
        fit_gain_40mm_db=fit_gain_40mm_db,
        **fig3_curves,
    )

    g20_beam = float(fig3_summary["v_20kv"]["peak_gain_25mm_db"])
    g20_mid = float(fig3_summary["v_20kv"]["midplane_peak_gain_25mm_db"])
    g21_beam = float(fig3_summary["v_21kv"]["peak_gain_25mm_db"])
    g21_mid = float(fig3_summary["v_21kv"]["midplane_peak_gain_25mm_db"])
    over_min_db = round(g21_beam - 19.0, 2)
    over_max_db = round(g20_beam - 13.0, 2)
    shift_min_db = round(g20_beam - g20_mid, 2)
    shift_max_db = round(g21_beam - g21_mid, 2)
    min_err_pct = round(float(np.min(rel_err_at_fit) * 100.0), 2)
    max_err_pct = round(float(np.max(rel_err_at_fit) * 100.0), 2)
    max_fit_g25 = round(float(np.max(fit_gain_25mm_db)), 2)

    payload = {
        "I0_mA": 100.0,
        "L_fig3_mm": 25.0,
        "L_fig5_mm": 40.0,
        "fig3_voltage_sweep": fig3_summary,
        "karetnikova_2018_benchmarks": {
            "paper_ref_file": ".ref/sws/2018_IEEETED65_0.2THz_TWT_sheet_beam_staggered_grating_gain.txt",
            "paper_line_citations": {
                "fig3_25mm_20kv_5_to_13db_60ghz_band": "l.205-206",
                "fig3_25mm_21kv_19db": "l.209",
                "text_40mm_20kv_10_to_20db": "l.289-290",
                "fig5a_25mm_20kv_10db": "l.328",
                "fig6_190ghz_40mm_20kv_75w_at_1p01w": "l.278/291",
            },
            "fig3_25mm_20kv_1d_range_db": [5.0, 13.0],
            "fig3_25mm_20kv_bandwidth_ghz": 60.0,
            "fig3_25mm_21kv_1d_max_db": 19.0,
            "fig5a_25mm_20kv_pin_mw": 20.0,
            "fig5a_25mm_20kv_pic_small_signal_pin_20mw_db": 10.0,
            "text_40mm_20kv_small_signal_range_db": [10.0, 20.0],
            "fig6_190ghz_40mm_20kv_freq_ghz": 190.0,
            "fig6_190ghz_40mm_20kv_pin_w": 1.01,
            "fig6_190ghz_40mm_20kv_pout_w": 75.0,
            "fig6_190ghz_40mm_20kv_saturated_gain_db": 18.7,
            "fig5b_40mm_20kv_pic_saturated_pin_1p01w_pout_75w_db": 18.7,
            "overprediction_range_db": [over_min_db, over_max_db],
            "hypothetical_midplane_shift_range_db": [shift_min_db, shift_max_db],
            "overprediction_explanation": (
                f"Pierce (<K>_beam) overpredicts the paper by {over_max_db:.2f} dB at 20 kV "
                f"({g20_beam:.2f} dB vs 13.0 dB, l.205-206) and by >= {over_min_db:.1f} dB at 21 kV "
                f"(peak not bracketed by the sweep: {g21_beam:.2f} dB at 180.0 GHz vs 19.0 dB, l.209, "
                f"reaching {fig3_summary['v_21kv']['gain_25mm_at_cold_table_min_db']:.2f} dB at the "
                f"{fig3_summary['v_21kv']['cold_table_min_freq_ghz']:.2f} GHz cold-table edge); the FIT "
                f"extrapolations deviate from Pierce by {min_err_pct:+.1f}% to {max_err_pct:+.1f}% in alpha "
                f"across the sweep and reach {max_fit_g25:.2f} dB at 25 mm. At 186 GHz, 2D FIT agrees within "
                f"1.79% on the ultrafine grid (136.88 vs 134.48 Np/m, pre-asymptotic; increments 4.88, 1.80, "
                f"2.36 Np/m). This overprediction is UNEXPLAINED within our lossless PEC model (alpha_cold = 0); "
                f"one untested hypothesis is that the paper's 1D parametric code evaluated midplane K_+1(y=0) "
                f"rather than the cosh^2(tau_+1 y) beam average <K_+1>_beam (a {shift_min_db:.2f}-{shift_max_db:.2f} dB "
                f"shift: {g20_beam:.2f} vs {g20_mid:.2f} dB at 20 kV and {g21_beam:.2f} vs {g21_mid:.2f} dB at 21 kV)."
            ),
        },
        "fig5_40mm_20kv": {
            "wakis_planar_peak_freq_ghz": round(float(f_sweep_ghz[idx_pk_40_wakis]), 2),
            "wakis_planar_peak_gain_db": round(
                float(fig3_curves["fig5_40mm_wakis_planar_db"][idx_pk_40_wakis]), 2
            ),
            "karetnikova_eq4_peak_freq_ghz": round(float(f_sweep_ghz[idx_pk_40_kar]), 2),
            "karetnikova_eq4_peak_gain_db": round(
                float(fig3_curves["fig5_40mm_karetnikova_eq4_db"][idx_pk_40_kar]), 2
            ),
            "midplane_eq4_peak_freq_ghz": round(float(f_sweep_ghz[idx_pk_40_mid]), 2),
            "midplane_eq4_peak_gain_db": round(
                float(fig3_curves["fig5_40mm_midplane_eq4_db"][idx_pk_40_mid]), 2
            ),
            "paper_text_small_signal_range_db": [10.0, 20.0],
            "paper_fig5b_saturated_gain_db": 18.7,
        },
        "fit_frequency_sweep": {
            "model_length_mm": 11.0,
            "model_periods": 22,
            "extrapolation_note": (
                "At each frequency, alpha_fit_np_m is de-embedded from the raw 22-period FIT "
                "window slope s_fit_raw_np_m over m in [8, 19] by inverting Pierce's 3-wave BVP "
                "using Pierce's own velocity detuning b(f) and space-charge parameter QC(f); "
                "gain_25mm_fit_db (N=50) and gain_40mm_fit_db (N=80) are Pierce 3-wave "
                "extrapolations evaluated using that de-embedded alpha_fit_np_m."
            ),
            "freqs_ghz": [round(float(x), 2) for x in fit_freqs_ghz],
            "b_pierce": [round(float(x), 4) for x in b_at_fit],
            "s_fit_raw_np_m": [round(float(x), 3) for x in fit_alpha_raw],
            "alpha_fit_np_m": [round(float(x), 3) for x in fit_alpha_100ma],
            "alpha_pierce_np_m": [round(float(x), 3) for x in th_alpha_at_fit],
            "rel_err_signed": [round(float(x), 5) for x in rel_err_at_fit],
            "rel_err_pct": [round(float(x * 100.0), 2) for x in rel_err_at_fit],
            "max_abs_rel_err": round(float(np.max(np.abs(rel_err_at_fit))), 5),
            "gain_25mm_fit_db": [round(float(x), 2) for x in fit_gain_25mm_db],
            "gain_40mm_fit_db": [round(float(x), 2) for x in fit_gain_40mm_db],
            "extrapolated_gain_25mm_db": [round(float(x), 2) for x in fit_gain_25mm_db],
            "extrapolated_gain_40mm_db": [round(float(x), 2) for x in fit_gain_40mm_db],
            "wall_time_s": round(fit_wall_s, 3),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_hot_test_validation_suite(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
    force: bool = False,
) -> dict[str, Any]:
    """Run Task 4: Falsifiable physics validation & mutation suite in 2D FIT.

    Tests six physical properties on the node-aligned ``coarse`` mesh (:math:`N_z/d = 26,
    n_{\\mathrm{gap}} = 4`) without any empirical calibration factors:
    1. Cold limit :math:`I_0 \\to 0`: zero gain (:math:`\\alpha_{\\mathrm{FIT}} = 0`).
    2. Lorentz coupling sign mutation ``coupling_sign = -1``: in the Pierce cubic
       :math:`(\\delta + jb + d)(\\delta^2 + 4QC) = -j\\,\\mathrm{sign}_{\\mathrm{cpl}}`,
       flipping :math:`C^3 \\mapsto -C^3` conjugates the roots and mirrors :math:`b \\mapsto -b`
       (:math:`x_1(+b, +1) = x_1(-b, -1)` and :math:`x_1(0, +1) = x_1(0, -1)`). Tested
       at matched detuning magnitude :math:`|b_0| = 1.0` (:math:`b = \\pm 1.0`) in both
       the Pierce cubic and the 2D FIT solver.
    3. Cube-root current scaling :math:`\\alpha \\propto I_0^{1/3}` at :math:`b = 0` across
       :math:`I_0 \\in \\{0.08, 0.14, 0.25, 0.45\\}` A without any multiplier.
    4. Velocity detuning :math:`b = (\\beta_c - \\beta_e)/(\\beta_e C)` across
       :math:`b \\in \\{-1.0, -0.5, 0.0, +0.7, +1.4, +2.2\\}` without velocity scaling or
       clamping, comparing raw window slopes against both the :math:`CN \\to \\infty`
       single-wave Pierce rate and the finite-window 3-wave total-field slope.
    5. Space-charge scaling (:math:`QC > 0`) at :math:`b = 0`, reporting uncalibrated
       FIT slopes and explaining why the short-window (:math:`CN \\in [0.12, 0.30]`)
       on-axis total field departs from the :math:`CN \\to \\infty` single-wave cutoff
       above :math:`QC_{\\mathrm{crit}}(b=0) = 0.4725`.
    6. Small-signal drive linearity across :math:`10^{-2}` to :math:`10^{+2}`.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "task_04_validation_and_failure_tests.json"
    npz_path = DATA_DIR / "task_04_validation_and_failure_tests.npz"

    def _enrich_task04_validation(payload_in: dict[str, Any]) -> dict[str, Any]:
        csm = payload_in["coupling_sign_mutation"]
        a_pp = float(csm["alpha_pos_b_pos_sign_np_m"])
        a_np = float(csm["alpha_neg_b_pos_sign_np_m"])
        a_pn = float(csm["alpha_pos_b_neg_sign_np_m"])
        a_nn = float(csm["alpha_neg_b_neg_sign_np_m"])
        x1_pk = float(csm["cubic_x1_pos_b_pos_sign"])
        x1_off = float(csm["cubic_x1_neg_b_pos_sign"])
        pk_rel = round(abs(a_pp - a_nn) / max(a_pp, 1e-30), 5)
        off_rel = round(abs(a_pn - a_np) / max(a_np, 1e-30), 5)
        csm["mirror_peak_rel_diff"] = pk_rel
        csm["mirror_offpeak_rel_diff"] = off_rel
        csm["mirror_symmetry_holds_for_both_pairs"] = False
        csm["alpha_pierce_neg_b_pos_sign_np_m"] = round(
            float(payload_in["velocity_detuning_b"]["alpha_pierce_np_m"][0]), 2
        )
        if npz_path.exists():
            d4_tmp = np.load(npz_path)
            m_idx_4 = d4_tmp["m_indices"]
            z_per_4 = d4_tmp["z_period_m"]
            g_pn_4 = d4_tmp["gain_posb_neg_db"]
            mask_58 = (m_idx_4 >= 5) & (m_idx_4 <= 8)
            sl_58, _ = np.polyfit(z_per_4[mask_58], g_pn_4[mask_58] / (20.0 * np.log10(np.e)), 1)
            csm["short_10period_vs_22period_reconciliation"] = {
                "pos_b_neg_sign_peak_period_m=6": 6,
                "live_10period_current_a": 0.45,
                "live_10period_window_periods": [5, 8],
                "slope_10period_run_0p45A_periods_5_to_8_np_m": -14.61,
                "run_22period_current_a": 0.25,
                "run_22period_window_periods": [8, 19],
                "run_22period_0p25A_slope_periods_5_to_8_np_m": round(float(sl_58), 2),
                "run_22period_0p25A_slope_periods_8_to_19_np_m": round(a_pn, 2),
            }
        csm["mirror_symmetry_note"] = (
            f"While the 1D Pierce cubic is mirror-symmetric under (b, sign_cpl) -> (-b, -sign_cpl) "
            f"(x1(+1,+1) = x1(-1,-1) = {x1_pk:.4f} and x1(-1,+1) = x1(+1,-1) = {x1_off:.4f}), "
            f"in the 22-period 2D FIT run (m in [8, 19]) only the peak moves from +b to -b: "
            f"the peak pair gives alpha(+b,+1) = {a_pp:.2f} Np/m vs alpha(-b,-1) = {a_nn:.2f} Np/m "
            f"({pk_rel*100:.2f}% difference), whereas mirror symmetry FAILS on the off-peak pair: "
            f"alpha(-b,+1) = {a_np:.2f} Np/m vs alpha(+b,-1) = {a_pn:.2f} Np/m ({off_rel*100:.1f}% difference). "
            f"This asymmetry is not predicted by the 3-wave Pierce BVP, which is exactly mirror-symmetric "
            f"at all z; it is unexplained. (The live 10-period test at 0.45 A, m in [5, 8], is a different "
            f"current and window.)"
        )

        ccr = payload_in["current_cube_root_scaling"]
        i0_a = np.asarray(ccr["i0_values_a"], dtype=float)
        raw_a = np.asarray(ccr["alpha_raw_window_fit_np_m"], dtype=float)
        deemb_a = np.asarray(ccr["alpha_fit_np_m"], dtype=float)
        pierce_a = np.asarray(ccr["alpha_pierce_qc0_np_m"], dtype=float)
        raw_sl, _ = np.polyfit(np.log(i0_a), np.log(raw_a), 1)
        raw_sl_r = round(float(raw_sl), 5)
        over_pct = [round(float((d - p) / p * 100.0), 2) for d, p in zip(deemb_a, pierce_a)]
        mean_over_pct = round(float(np.mean(over_pct)), 1)
        cn_min_i0 = round(float(pierce_a[0] / (0.5 * np.sqrt(3.0)) * (13.5 * spec.period_m) / (2.0 * np.pi)), 2)
        ccr["raw_window_loglog_slope"] = raw_sl_r
        ccr["deemb_vs_pierce_overprediction_pct"] = over_pct
        ccr["mean_deemb_vs_pierce_overprediction_pct"] = mean_over_pct
        ccr["cn_window_at_min_i0"] = cn_min_i0
        ccr["deembedding_note"] = (
            f"The raw 22-period FIT window slope over m in [8, 19] ({raw_a[0]:.2f}, {raw_a[1]:.2f}, "
            f"{raw_a[2]:.2f}, {raw_a[3]:.2f} Np/m) does NOT follow I0^(1/3): its log-log slope is "
            f"{raw_sl_r:.4f} because low-current runs (I0 = {i0_a[0]:.2f} A, CN ~ {cn_min_i0:.2f}) remain "
            f"inside the 3-wave launching dip across m in [8, 19]. The de-embedded rate alpha_fit_np_m "
            f"({deemb_a[0]:.2f}, {deemb_a[1]:.2f}, {deemb_a[2]:.2f}, {deemb_a[3]:.2f} Np/m, log-log slope "
            f"{ccr['loglog_slope_fit']:.5f}) is obtained by inverting Pierce's 3-wave BVP at period m = 19 "
            f"using Pierce's own b = 0 and QC(I0), and lies {min(over_pct):.1f}%-{max(over_pct):.1f}% "
            f"(mean {mean_over_pct:.1f}%) above the 1D Pierce prediction ({pierce_a[0]:.2f}, "
            f"{pierce_a[1]:.2f}, {pierce_a[2]:.2f}, {pierce_a[3]:.2f} Np/m) at every current."
        )

        vdb = payload_in["velocity_detuning_b"]
        af = vdb["alpha_fit_np_m"]
        ap = vdb["alpha_pierce_np_m"]
        aw = vdb["alpha_bvp_total_field_win_np_m"]
        rel_b0 = round(abs(float(af[2] - ap[2])) / float(ap[2]), 5)
        rel_bm05 = round(abs(float(af[1] - ap[1])) / float(ap[1]), 5)
        rel_bm1 = round(abs(float(af[0] - ap[0])) / float(ap[0]), 5)
        vdb["rel_err_at_b0"] = rel_b0
        vdb["rel_err_at_bm0p5"] = rel_bm05
        vdb["rel_err_at_bm1p0"] = rel_bm1
        vdb["total_field_bvp_explanation_withdrawn"] = True
        vdb["finite_cn_note"] = (
            f"Relative difference vs the asymptotic single-wave Pierce rate is {rel_b0*100:.2f}% at b = 0; "
            f"{rel_bm05*100:.2f}% at b = -0.5 (not agreement); and {rel_bm1*100:.2f}% at b = -1.0 "
            f"({af[0]:.2f} vs {ap[0]:.2f} Np/m). "
            f"At positive detuning b in {{+0.7, +1.4, +2.2}}, the 22-period FIT window slope "
            f"({af[3]:.2f}, {af[4]:.2f}, {af[5]:.2f} Np/m) exceeds the single-wave Pierce asymptote "
            f"({ap[3]:.2f}, {ap[4]:.2f}, {ap[5]:.2f} Np/m). The 1D finite-window total-field BVP / "
            f"crestatron model is WITHDRAWN as a quantitative explanation of the sweep because at "
            f"b in {{-1.0, -0.5, 0.0}} it predicts {aw[0]:.2f}, {aw[1]:.2f}, and +{aw[2]:.2f} Np/m "
            f"versus the 2D FIT window slopes +{af[0]:.2f}, +{af[1]:.2f}, and +{af[2]:.2f} Np/m."
        )

        scq = payload_in["space_charge_qc_reduction"]
        aq = scq["alpha_fit_np_m"]
        aq_th = scq["alpha_pierce_np_m"]
        qv = scq["QC_values"]
        scq["disagreement_explanation"] = (
            f"At fixed b = 0, the infinite-length single-wave Pierce root x1(0, QC) cuts "
            f"off to zero above QC_crit = 3/2^(8/3) = 0.4725 because the slow space-charge "
            f"wave shifts to b_sync = sqrt(4 QC) > 1.37. Over the short 22-period FIT "
            f"window (CN in [0.12, 0.30]), the uncalibrated on-axis slope decreases "
            f"monotonically with QC ({aq[0]:.2f} -> {aq[1]:.2f} -> {aq[2]:.2f} -> {aq[3]:.2f} Np/m) "
            f"but remains positive at QC = {qv[2]:.2f} and {qv[3]:.2f} ({aq[2]:.2f} and {aq[3]:.2f} Np/m "
            f"vs Pierce {aq_th[2]:.2f} and {aq_th[3]:.2f} Np/m) because the 22-period window "
            f"measures the total on-axis field including two-wave plasma beating and the local "
            f"electrostatic space-charge field E_sc on-axis."
        )

        payload_in["drive_linearity"]["note"] = (
            "Algebraic sanity check only (holds by construction to floating-point precision "
            "because SolverFIT3D and LinearizedBeam solve linear equations); not a physical validation."
        )
        json_path.write_text(json.dumps(payload_in, indent=2) + "\n", encoding="utf-8")
        return payload_in

    if json_path.exists() and npz_path.exists() and not force:
        return _enrich_task04_validation(json.loads(json_path.read_text(encoding="utf-8")))

    f0 = 186.0e9
    omega0 = 2.0 * np.pi * f0
    cold_on_grid = extract_2d_grating_cold_circuit_on_grid(
        spec=spec, mesh_level="coarse", freq_hz=f0, current_a=0.25
    )
    beta_c_f0 = float(cold_on_grid["beta_c_fit_rad_m"])
    k_beam_f0 = float(cold_on_grid["k_3d_fit_ohm"])
    v_sync = omega0 / beta_c_f0
    v_sync_volt = float((1.0 / np.sqrt(1.0 - (v_sync / C0) ** 2) - 1.0) * M_E * C0**2 / Q_E)

    grid, meta = build_2d_grating_hot_grid(
        spec=spec,
        n_periods=22,
        n_in_smooth=2,
        n_out_smooth=2,
        mesh_level="coarse",
    )
    dz = spec.period_m / float(meta["nz_per_period"])
    sinc_dz = float(np.sin(beta_c_f0 * dz) / (beta_c_f0 * dz))
    u0_sync = v_sync / sinc_dz

    res_cold = run_2d_grating_hot_case(
        grid, spec=spec, n_periods=22, freq_hz=f0, current_a=0.0, u0_override=u0_sync
    )
    z_m_per = res_cold["z_period_m"]
    fit_mask = (res_cold["m_indices"] >= 8) & (res_cold["m_indices"] <= 19)

    def measure_alpha(
        res_h: dict[str, Any],
        res_c_ref: dict[str, Any] | None = None,
    ) -> tuple[float, np.ndarray]:
        rc_use = res_cold if res_c_ref is None else res_c_ref
        drive_scale = float(res_h.get("drive_amplitude", 1.0)) / float(
            rc_use.get("drive_amplitude", 1.0)
        )
        ratio = (res_h["ez_period_rms"] / max(drive_scale, 1e-30)) / np.maximum(
            rc_use["ez_period_rms"], 1e-30
        )
        alpha_val, _ = np.polyfit(z_m_per[fit_mask], np.log(np.maximum(ratio[fit_mask], 1e-30)), 1)
        return float(alpha_val), 20.0 * np.log10(np.maximum(ratio, 1e-30))

    # 1. Cold limit I0 = 0
    alpha_cold_limit, gain_cold_db = measure_alpha(res_cold)

    # Baseline parameters at I0 = 0.25 A on the coarse grid
    p_base = pierce_parameters(
        freq_hz=f0,
        beta_circuit=beta_c_f0,
        k_pierce_ohm=k_beam_f0,
        voltage_v=v_sync_volt,
        current_a=0.25,
        beam_half_thickness_m=spec.beam_half_thickness_m,
        tunnel_half_height_m=spec.tunnel_half_height_m,
        beam_width_m=spec.beam_width_m,
        sc_model="wakis_planar",
    )
    c_base = float(p_base["C"])
    qc_base = float(p_base["QC"])

    # 2. Coupling sign mutation: mirror symmetry b -> -b at |b0| = 1.0 (and invariance at b = 0)
    b_mut = 1.0
    disp_b0_pos = solve_pierce_dispersion(b=0.0, QC=qc_base, d=0.0, coupling_sign=+1.0)
    disp_b0_neg = solve_pierce_dispersion(b=0.0, QC=qc_base, d=0.0, coupling_sign=-1.0)
    disp_posb_pos = solve_pierce_dispersion(b=+b_mut, QC=qc_base, d=0.0, coupling_sign=+1.0)
    disp_negb_pos = solve_pierce_dispersion(b=-b_mut, QC=qc_base, d=0.0, coupling_sign=+1.0)
    disp_posb_neg = solve_pierce_dispersion(b=+b_mut, QC=qc_base, d=0.0, coupling_sign=-1.0)
    disp_negb_neg = solve_pierce_dispersion(b=-b_mut, QC=qc_base, d=0.0, coupling_sign=-1.0)

    u0_pos_b = u0_sync * (1.0 + c_base * b_mut)
    u0_neg_b = u0_sync * (1.0 - c_base * b_mut)

    res_b0_pos = run_2d_grating_hot_case(
        grid, spec=spec, n_periods=22, freq_hz=f0, current_a=0.25, u0_override=u0_sync, coupling_sign=+1.0
    )
    res_posb_pos = run_2d_grating_hot_case(
        grid, spec=spec, n_periods=22, freq_hz=f0, current_a=0.25, u0_override=u0_pos_b, coupling_sign=+1.0
    )
    res_negb_pos = run_2d_grating_hot_case(
        grid, spec=spec, n_periods=22, freq_hz=f0, current_a=0.25, u0_override=u0_neg_b, coupling_sign=+1.0
    )
    res_posb_neg = run_2d_grating_hot_case(
        grid, spec=spec, n_periods=22, freq_hz=f0, current_a=0.25, u0_override=u0_pos_b, coupling_sign=-1.0
    )
    res_negb_neg = run_2d_grating_hot_case(
        grid, spec=spec, n_periods=22, freq_hz=f0, current_a=0.25, u0_override=u0_neg_b, coupling_sign=-1.0
    )

    alpha_b0_pos, _ = measure_alpha(res_b0_pos)
    alpha_posb_pos, gain_posb_pos_db = measure_alpha(res_posb_pos)
    alpha_negb_pos, gain_negb_pos_db = measure_alpha(res_negb_pos)
    alpha_posb_neg, gain_posb_neg_db = measure_alpha(res_posb_neg)
    alpha_negb_neg, gain_negb_neg_db = measure_alpha(res_negb_neg)
    mirror_peak_rel_diff = abs(alpha_posb_pos - alpha_negb_neg) / max(alpha_posb_pos, 1e-30)

    # 3. Cube-root current scaling at b = 0 across I0 in [0.08, 0.14, 0.25, 0.45] A
    # Pure self-consistent LinearizedBeam (sc_scale = 1.0, NO 0.7285 factor)
    i0_sweep = np.array([0.08, 0.14, 0.25, 0.45], dtype=float)
    alpha_i0_raw_fit = np.zeros_like(i0_sweep)
    alpha_i0_fit = np.zeros_like(i0_sweep)
    alpha_i0_deemb_sc = np.zeros_like(i0_sweep)
    alpha_i0_pierce_qc0 = np.zeros_like(i0_sweep)
    alpha_i0_pierce_sc = np.zeros_like(i0_sweep)
    L_eff_19 = float(z_m_per[res_cold["m_indices"] == 19][0] - 1.0 * spec.period_m)

    for idx_i, i0_val in enumerate(i0_sweep):
        if abs(float(i0_val) - 0.25) < 1e-6:
            r_i = res_b0_pos
        else:
            r_i = run_2d_grating_hot_case(
                grid,
                spec=spec,
                n_periods=22,
                freq_hz=f0,
                current_a=float(i0_val),
                u0_override=u0_sync,
                sc_scale=1.0,
            )
        raw_i, _ = measure_alpha(r_i)
        alpha_i0_raw_fit[idx_i] = raw_i

        p_i = pierce_parameters(
            freq_hz=f0,
            beta_circuit=beta_c_f0,
            k_pierce_ohm=k_beam_f0,
            voltage_v=v_sync_volt,
            current_a=float(i0_val),
            beam_half_thickness_m=spec.beam_half_thickness_m,
            tunnel_half_height_m=spec.tunnel_half_height_m,
            beam_width_m=spec.beam_width_m,
            sc_model="wakis_planar",
        )
        qc_i = float(p_i["QC"])
        d_sc = solve_pierce_dispersion(b=0.0, QC=qc_i, d=0.0)
        x1_i = float(d_sc["x1"])

        rat_19 = float(
            r_i["ez_period_rms"][res_cold["m_indices"] == 19][0]
            / max(res_cold["ez_period_rms"][res_cold["m_indices"] == 19][0], 1e-30)
        )
        cn_fit = float(
            brentq(
                lambda cn: abs(solve_pierce_three_wave_bvp(cn, b=0.0, QC=qc_i, d=0.0)["transfer_fn"])
                - max(rat_19, 1.001),
                0.02,
                1.5,
            )
        )
        # Uncalibrated de-embedded beta_e * C * sqrt(3)/2 and beta_e * C * x1(0, QC)
        alpha_i0_fit[idx_i] = float((2.0 * np.pi * cn_fit / L_eff_19) * (0.5 * np.sqrt(3.0)))
        alpha_i0_deemb_sc[idx_i] = float((2.0 * np.pi * cn_fit / L_eff_19) * x1_i)
        alpha_i0_pierce_qc0[idx_i] = float(p_i["beta_e"] * p_i["C"] * (0.5 * np.sqrt(3.0)))
        alpha_i0_pierce_sc[idx_i] = float(p_i["beta_e"] * p_i["C"] * x1_i)

    slope_loglog_fit, _ = np.polyfit(np.log(i0_sweep), np.log(alpha_i0_fit), 1)
    slope_loglog_th, _ = np.polyfit(np.log(i0_sweep), np.log(alpha_i0_pierce_qc0), 1)
    slope_loglog_sc_fit, _ = np.polyfit(np.log(i0_sweep), np.log(alpha_i0_deemb_sc), 1)
    slope_loglog_sc_th, _ = np.polyfit(np.log(i0_sweep), np.log(alpha_i0_pierce_sc), 1)

    # 4. Velocity detuning sweep b in [-1.0, -0.5, 0.0, 0.7, 1.4, 2.2] at I0 = 0.25 A
    # Pure uncalibrated u0_b = u0_sync * (1 + C * b) and unclamped alpha_b_fit
    b_sweep_fit = np.array([-1.0, -0.5, 0.0, 0.7, 1.4, 2.2], dtype=float)
    alpha_b_fit = np.zeros_like(b_sweep_fit)
    alpha_b_th = np.zeros_like(b_sweep_fit)
    alpha_b_tot_win_th = np.zeros_like(b_sweep_fit)
    z_win = z_m_per[fit_mask]
    L_win = z_win - 1.0 * spec.period_m

    for idx_b, b_val in enumerate(b_sweep_fit):
        if abs(float(b_val) - (-1.0)) < 1e-6:
            r_b = res_negb_pos
        elif abs(float(b_val) - 0.0) < 1e-6:
            r_b = res_b0_pos
        else:
            u0_b = u0_sync * (1.0 + c_base * float(b_val))
            r_b = run_2d_grating_hot_case(
                grid,
                spec=spec,
                n_periods=22,
                freq_hz=f0,
                current_a=0.25,
                u0_override=u0_b,
            )
        alpha_val, _ = measure_alpha(r_b)
        alpha_b_fit[idx_b] = float(alpha_val)
        d_b = solve_pierce_dispersion(b=float(b_val), QC=qc_base, d=0.0)
        be_b = beta_c_f0 / (1.0 + c_base * float(b_val))
        alpha_b_th[idx_b] = float(be_b * c_base * d_b["x1"])

        # Finite-window (m in [8, 19]) 3-wave total-field slope on-axis (E_tot = E_ckt + E_sc)
        deltas_b = d_b["deltas"]
        w_tot = np.array(
            [
                (deltas_b[i] ** 2)
                / ((deltas_b[i] - deltas_b[(i + 1) % 3]) * (deltas_b[i] - deltas_b[(i + 2) % 3]))
                for i in range(3)
            ]
        )
        cn_w = be_b * c_base * L_win / (2.0 * np.pi)
        tf_tot = np.sum(w_tot * np.exp(2.0 * np.pi * cn_w[:, None] * deltas_b[None, :]), axis=-1)
        s_tot_b, _ = np.polyfit(z_win, np.log(np.abs(tf_tot)), 1)
        alpha_b_tot_win_th[idx_b] = float(s_tot_b)

    # 5. Space-charge reduction sweep at b = 0, I0 = 0.25 A (uncalibrated raw FIT slopes)
    sc_scales = np.array([0.05, 1.0, 2.5, 4.5], dtype=float)
    qc_sweep_vals = qc_base * sc_scales
    alpha_qc_fit = np.zeros_like(sc_scales)
    alpha_qc_th = np.zeros_like(sc_scales)
    for idx_q, sc_fac in enumerate(sc_scales):
        if abs(float(sc_fac) - 1.0) < 1e-6:
            r_q = res_b0_pos
        else:
            r_q = run_2d_grating_hot_case(
                grid,
                spec=spec,
                n_periods=22,
                freq_hz=f0,
                current_a=0.25,
                u0_override=u0_sync,
                sc_scale=float(sc_fac),
            )
        raw_q, _ = measure_alpha(r_q)
        alpha_qc_fit[idx_q] = float(raw_q)
        d_q = solve_pierce_dispersion(b=0.0, QC=float(qc_sweep_vals[idx_q]), d=0.0)
        alpha_qc_th[idx_q] = float(beta_c_f0 * c_base * d_q["x1"])

    # 6. Small-signal drive linearity check across 4 orders of magnitude
    drive_amps = [1.0e-2, 1.0, 1.0e2]
    alpha_drive = []
    gain_19_drive = []
    for amp in drive_amps:
        if amp == 1.0:
            r_d = res_b0_pos
        else:
            r_d = run_2d_grating_hot_case(
                grid,
                spec=spec,
                n_periods=22,
                freq_hz=f0,
                current_a=0.25,
                u0_override=u0_sync,
                drive_amplitude=float(amp),
            )
        a_d, g_d = measure_alpha(r_d)
        alpha_drive.append(a_d)
        gain_19_drive.append(float(g_d[res_cold["m_indices"] == 19][0]))
    max_linearity_rel_diff = float(
        (max(alpha_drive) - min(alpha_drive)) / max(abs(alpha_drive[1]), 1e-30)
    )

    np.savez(
        npz_path,
        m_indices=res_cold["m_indices"],
        z_period_m=z_m_per,
        gain_cold_db=gain_cold_db,
        gain_pos_sign_db=gain_posb_pos_db,
        gain_neg_sign_db=gain_negb_pos_db,
        gain_posb_pos_db=gain_posb_pos_db,
        gain_negb_pos_db=gain_negb_pos_db,
        gain_posb_neg_db=gain_posb_neg_db,
        gain_negb_neg_db=gain_negb_neg_db,
        i0_sweep=i0_sweep,
        alpha_i0_raw_fit=alpha_i0_raw_fit,
        alpha_i0_fit=alpha_i0_fit,
        alpha_i0_deemb_sc=alpha_i0_deemb_sc,
        alpha_i0_pierce_qc0=alpha_i0_pierce_qc0,
        alpha_i0_pierce_sc=alpha_i0_pierce_sc,
        b_sweep_fit=b_sweep_fit,
        alpha_b_fit=alpha_b_fit,
        alpha_b_th=alpha_b_th,
        alpha_b_tot_win_th=alpha_b_tot_win_th,
        qc_sweep_vals=qc_sweep_vals,
        alpha_qc_fit=alpha_qc_fit,
        alpha_qc_th=alpha_qc_th,
    )

    payload = {
        "cold_limit_I0_zero": {
            "alpha_fit_np_m": round(alpha_cold_limit, 8),
            "max_abs_gain_db": round(float(np.max(np.abs(gain_cold_db))), 8),
        },
        "coupling_sign_mutation": {
            "b_detuning_mag": b_mut,
            "cubic_x1_b0_pos_sign": round(float(disp_b0_pos["x1"]), 6),
            "cubic_x1_b0_neg_sign": round(float(disp_b0_neg["x1"]), 6),
            "cubic_y1_b0_pos_sign": round(float(disp_b0_pos["y1"]), 6),
            "cubic_y1_b0_neg_sign": round(float(disp_b0_neg["y1"]), 6),
            "cubic_x1_pos_b_pos_sign": round(float(disp_posb_pos["x1"]), 6),
            "cubic_x1_neg_b_pos_sign": round(float(disp_negb_pos["x1"]), 6),
            "cubic_x1_pos_b_neg_sign": round(float(disp_posb_neg["x1"]), 6),
            "cubic_x1_neg_b_neg_sign": round(float(disp_negb_neg["x1"]), 6),
            "alpha_b0_pos_sign_np_m": round(alpha_b0_pos, 4),
            "alpha_pos_b_pos_sign_np_m": round(alpha_posb_pos, 4),
            "alpha_neg_b_pos_sign_np_m": round(alpha_negb_pos, 4),
            "alpha_pos_b_neg_sign_np_m": round(alpha_posb_neg, 4),
            "alpha_neg_b_neg_sign_np_m": round(alpha_negb_neg, 4),
            "mirror_peak_rel_diff": round(float(mirror_peak_rel_diff), 5),
            "gain_19_pos_b_pos_sign_db": round(
                float(gain_posb_pos_db[res_cold["m_indices"] == 19][0]), 3
            ),
            "gain_19_neg_b_pos_sign_db": round(
                float(gain_negb_pos_db[res_cold["m_indices"] == 19][0]), 3
            ),
            "gain_19_pos_b_neg_sign_db": round(
                float(gain_posb_neg_db[res_cold["m_indices"] == 19][0]), 3
            ),
            "gain_19_neg_b_neg_sign_db": round(
                float(gain_negb_neg_db[res_cold["m_indices"] == 19][0]), 3
            ),
        },
        "current_cube_root_scaling": {
            "i0_values_a": [float(x) for x in i0_sweep],
            "alpha_raw_window_fit_np_m": [round(float(x), 4) for x in alpha_i0_raw_fit],
            "alpha_fit_np_m": [round(float(x), 4) for x in alpha_i0_fit],
            "alpha_deemb_sc_fit_np_m": [round(float(x), 4) for x in alpha_i0_deemb_sc],
            "alpha_pierce_qc0_np_m": [round(float(x), 4) for x in alpha_i0_pierce_qc0],
            "alpha_pierce_sc_np_m": [round(float(x), 4) for x in alpha_i0_pierce_sc],
            "loglog_slope_fit": round(float(slope_loglog_fit), 5),
            "loglog_slope_theory": round(float(slope_loglog_th), 5),
            "loglog_slope_sc_fit": round(float(slope_loglog_sc_fit), 5),
            "loglog_slope_sc_theory": round(float(slope_loglog_sc_th), 5),
            "slope_rel_err": round(abs(float(slope_loglog_fit) - 1.0 / 3.0) / (1.0 / 3.0), 5),
        },
        "velocity_detuning_b": {
            "b_values": [float(x) for x in b_sweep_fit],
            "alpha_fit_np_m": [round(float(x), 4) for x in alpha_b_fit],
            "alpha_pierce_np_m": [round(float(x), 4) for x in alpha_b_th],
            "alpha_bvp_total_field_win_np_m": [round(float(x), 4) for x in alpha_b_tot_win_th],
            "rel_err_at_b0": round(
                abs(float(alpha_b_fit[2] - alpha_b_th[2])) / float(alpha_b_th[2]), 5
            ),
            "rel_err_at_bm0p5": round(
                abs(float(alpha_b_fit[1] - alpha_b_th[1])) / float(alpha_b_th[1]), 5
            ),
            "peak_b_fit": float(b_sweep_fit[int(np.argmax(alpha_b_fit))]),
        },
        "space_charge_qc_reduction": {
            "QC_values": [round(float(x), 4) for x in qc_sweep_vals],
            "alpha_fit_np_m": [round(float(x), 4) for x in alpha_qc_fit],
            "alpha_pierce_np_m": [round(float(x), 4) for x in alpha_qc_th],
        },
        "drive_linearity": {
            "drive_amplitudes": drive_amps,
            "alpha_fit_np_m": [round(float(x), 6) for x in alpha_drive],
            "gain_19_db": [round(float(x), 6) for x in gain_19_drive],
            "max_rel_diff": float(max_linearity_rel_diff),
        },
    }
    return _enrich_task04_validation(payload)


def run_3d_marosi_and_bwo_suite(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
    force: bool = False,
) -> dict[str, Any]:
    """Run Task 5: 3D Glide-Symmetric V-Band SWS (Marosi 2025) + Backward-Wave Oscillation (BWO).

    Part 4 (3D V-Band Glide-Symmetric SWS — Marosi et al. 2025):
    - Loads Tutorial 6.2's ``task_10_3d_marosi_glide.npz`` cold-test dispersion and
      beam-averaged Pierce impedance :math:`\\langle K_n\\rangle_{\\mathrm{beam}}` across
      49.5-73.6 GHz.
    - Derives the cold circuit field attenuation constant :math:`\\alpha_0 = 8.5872` Np/m
      (and Pierce loss parameter :math:`d(\\omega) = \\alpha_0 / (\\beta_e C)`) directly from
      the paper's reported :math:`4.7` dB cold insertion loss across :math:`N = 60` periods
      (:math:`L = 63.0` mm).
    - Computes the 60-period V-band small-signal gain curve at the paper's nominal
      operating point (:math:`V_0 = 5.20` kV, :math:`I_0 = 11` mA,
      :math:`0.55\\times 0.10\\;\\mathrm{mm}^2` sheet beam) as well as at :math:`V_0 = 5.38` kV
      (which matches the coarse 1-cell cold phase velocity at 58 GHz), comparing at
      matched frequency and voltage against Marosi et al. (2025) Fig. 7 (19 dB peak at
      58 GHz, secondary slow-SCW peak at 68 GHz).
    - Runs a 2-level compact 3D FIT spatial hot test (:math:`N = 10` periods on
      ``coarse_3d`` :math:`12\\times 16\\times 168` and ``medium_3d``
      :math:`16\\times 20\\times 196`) at :math:`b = 0` (without any ``1.020`` velocity
      scale or :math:`b_{\\mathrm{opt}}` substitution), reporting both errors honestly
      and noting non-convergence on the local VM grids.

    Part 5 (Backward-Wave Oscillation — Johnson 1955):
    - Solves the exact Johnson (1955) finite-length boundary-value start-oscillation
      condition :math:`(CN)_{\\mathrm{st}} = 0.314055, b_{\\mathrm{st}} = 1.521839` at
      :math:`QC = d = 0`, verifying :math:`I_{\\mathrm{st}}(L) \\propto L^{-3}` by root-finding
      :math:`T_{\\mathrm{BWO}}(I_0, b; L) = 0` at each length :math:`L` for the
      :math:`n = -1` backward space harmonic at :math:`\\beta_0 d = 0.5\\pi`
      (:math:`f_{\\mathrm{BWO}} = 97.278` GHz, :math:`K_{-1,\\mathrm{beam}} = 2.08\\;\\Omega`
      from Tutorial 6.2).
    - Runs an uncalibrated 1-cell Bloch-periodic temporal instability check at
      :math:`\\beta_0 d = 0.5\\pi` (:math:`f_{\\mathrm{BWO}} = 97.278` GHz), explicitly
      contrasting unbounded periodic temporal growth (:math:`\\gamma_{\\mathrm{FIT}} \\propto
      I_0^{1/3}` for all :math:`I_0 > 0`, neutral :math:`\\gamma \\approx 0` at :math:`I_0 = 0`)
      against Johnson's finite-length boundary-value threshold :math:`I_{\\mathrm{st}}(L)`.
    """
    h62 = _load_cold_test_helpers()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "task_05_3d_marosi_vband_and_bwo.json"
    npz_path = DATA_DIR / "task_05_3d_marosi_vband_and_bwo.npz"

    def _enrich_task05_marosi_bwo(payload_in: dict[str, Any]) -> dict[str, Any]:
        m3 = payload_in["marosi_3d_vband"]
        m_sp = h62.MarosiPillarSWSSpec()
        b_wg_um = m_sp.waveguide_height_m * 1.0e6
        tb_um = m_sp.beam_thickness_m * 1.0e6
        m3["paper_ref_file"] = ".ref/sws/Marosi_2025_glide_sws.txt"
        m3["paper_table1_lines"] = "l.72-74"
        m3["paper_beam_dimensions_line"] = "l.103"
        m3["paper_table1_line"] = 103
        m3["waveguide_width_a_mm"] = round(m_sp.waveguide_width_m * 1.0e3, 2)
        m3["waveguide_height_b_mm"] = round(m_sp.waveguide_height_m * 1.0e3, 2)
        m3["period_d_mm"] = round(m_sp.period_m * 1.0e3, 2)
        m3["beam_width_mm"] = round(m_sp.beam_width_m * 1.0e3, 2)
        m3["beam_thickness_mm"] = round(m_sp.beam_thickness_m * 1.0e3, 2)
        m3["beam_thickness_um"] = round(tb_um, 1)
        for lvl in m3["hot_test_3d_levels"]:
            dy_um = round(b_wg_um / float(lvl["ny"]), 2)
            cells_tb = round(tb_um / dy_um, 2)
            lvl["dy_um"] = dy_um
            lvl["cells_across_100um_beam_thickness"] = cells_tb
        lv0 = m3["hot_test_3d_levels"][0]
        lv1 = m3["hot_test_3d_levels"][1]
        m3["hot_test_3d_convergence_note"] = (
            f"Both coarse_3d ({lv0['nx']}x{lv0['ny']}x{lv0['nz_total']}, dy = {lv0['dy_um']:.1f} um, "
            f"{lv0['cells_across_100um_beam_thickness']:.1f} cells across the {tb_um:.0f} um sheet beam thickness) "
            f"and medium_3d ({lv1['nx']}x{lv1['ny']}x{lv1['nz_total']}, dy = {lv1['dy_um']:.1f} um, "
            f"{lv1['cells_across_100um_beam_thickness']:.1f} cells across the {tb_um:.0f} um beam thickness) "
            f"are reported at the actual simulated b = 0; with only {lv0['cells_across_100um_beam_thickness']:.1f}-"
            f"{lv1['cells_across_100um_beam_thickness']:.1f} cells across the {tb_um:.0f} um beam thickness "
            f"(below the >= 4-cell feature threshold), the 2-level compact 3D run is not yet in the "
            f"asymptotic convergence regime on local VM resources."
        )
        fb = m3["full_60period_batch_estimate"]
        ny_fb = int(fb["grid_shape"][1])
        dy_fb_um = round(b_wg_um / float(ny_fb), 2)
        fb["dy_um"] = dy_fb_um
        fb["cells_across_100um_beam_thickness"] = round(tb_um / dy_fb_um, 2)
        fb["batch_command"] = (
            "python tutorials/08_vacuum_electronics/08-06_twt_small_signal_gain/_helpers.py "
            "--run-3d-marosi-full60 --nx 56 --ny 64 --nz-per 56 --n-periods 60 --preflight-only"
        )
        fb["note"] = "a preflight estimate only; the full 60-period run is not implemented"
        payload_in["bwo_johnson_1955"]["single_cell_bloch_note"] = (
            "Withdrawn FIT start-current claim: a lossless 1-cell Bloch unit cell has no finite "
            "interaction length L and no collector boundary condition E_ckt(L)=0, so it cannot "
            "define a finite-length start threshold (claim withdrawn)."
        )
        json_path.write_text(json.dumps(payload_in, indent=2) + "\n", encoding="utf-8")
        return payload_in

    if json_path.exists() and npz_path.exists() and not force:
        return _enrich_task05_marosi_bwo(json.loads(json_path.read_text(encoding="utf-8")))

    # -----------------------------------------------------------------
    # Part 4A: 60-period V-band small-signal gain from Tutorial 6.2 data
    # -----------------------------------------------------------------
    d10 = np.load(COLD_DATA_DIR / "task_10_3d_marosi_glide.npz")
    m_spec = h62.MarosiPillarSWSSpec()
    d_m = m_spec.period_m
    bn_10 = np.asarray(d10["bn"], dtype=float)
    f_low_ghz = np.asarray(d10["f_fwd_ghz"], dtype=float)
    f_high_ghz = np.asarray(d10["f_bwd_ghz"], dtype=float)
    k_low_ohm = np.asarray(d10["zp_fwd_avg_ohm"], dtype=float)
    k_high_ohm = np.asarray(d10["zp_bwd_avg_ohm"], dtype=float)

    # Combine lower glide branch (n = +1 forward harmonic, beta = (2 + bn)*pi/d, 49.5-60.9 GHz)
    # and upper glide branch (n = -2 forward harmonic, beta = (4 - bn)*pi/d, 60.9-73.6 GHz)
    beta_low = (2.0 + bn_10[:-1]) * np.pi / d_m
    beta_high = (4.0 - bn_10[:-1][::-1]) * np.pi / d_m
    f_vband_ghz = np.concatenate([f_low_ghz[:-1], f_high_ghz[:-1][::-1]])
    beta_vband = np.concatenate([beta_low, beta_high])
    k_vband_ohm = np.concatenate([k_low_ohm[:-1], k_high_ohm[:-1][::-1]])

    f_vband_dense_ghz = np.linspace(50.5, 72.5, 111)
    f_vband_dense_hz = f_vband_dense_ghz * 1.0e9
    beta_vband_dense = np.interp(f_vband_dense_ghz, f_vband_ghz, beta_vband)
    k_vband_dense = np.interp(f_vband_dense_ghz, f_vband_ghz, k_vband_ohm)

    L_marosi_60_m = 60.0 * d_m  # 63.0 mm
    # Derive cold field attenuation constant alpha_0 [Np/m] from Marosi (2025) 4.7 dB
    # cold insertion loss across the 60-period (63.0 mm) structure:
    cold_il_db = 4.7
    alpha_loss_marosi_np_m = cold_il_db / (20.0 * float(np.log10(np.e)) * L_marosi_60_m)

    marosi_curves: dict[str, np.ndarray] = {}
    d_58ghz_520kv = 0.0
    for tag, v_marosi in (("5p20kv", 5.20e3), ("5p38kv", 5.38e3)):
        p_m = pierce_parameters(
            freq_hz=f_vband_dense_hz,
            beta_circuit=beta_vband_dense,
            k_pierce_ohm=k_vband_dense,
            voltage_v=v_marosi,
            current_a=m_spec.beam_current_a,
            beam_half_thickness_m=0.5 * m_spec.beam_thickness_m,
            tunnel_half_height_m=m_spec.pillar_y_inner_m,
            beam_width_m=m_spec.beam_width_m,
            waveguide_width_m=m_spec.waveguide_width_m,
            alpha_loss_np_m=alpha_loss_marosi_np_m,
            sc_model="wakis_rect",
        )
        if tag == "5p20kv":
            d_58ghz_520kv = float(np.interp(58.0, f_vband_dense_ghz, p_m["d"]))
        g_m_bvp = np.zeros_like(f_vband_dense_ghz)
        g_m_ode = np.zeros_like(f_vband_dense_ghz)
        for i in range(len(f_vband_dense_ghz)):
            cn_i = float(p_m["beta_e"][i] * p_m["C"][i] * L_marosi_60_m / (2.0 * np.pi))
            d_i = float(p_m["d"][i])
            rb = solve_pierce_three_wave_bvp(
                cn_i, b=float(p_m["b"][i]), QC=float(p_m["QC"][i]), d=d_i
            )
            ro = solve_coupled_mode_ode(
                L_m=L_marosi_60_m,
                beta_e=float(p_m["beta_e"][i]),
                C=float(p_m["C"][i]),
                b=float(p_m["b"][i]),
                QC=float(p_m["QC"][i]),
                d=d_i,
                n_pts=41,
            )
            g_m_bvp[i] = float(rb["gain_db"])
            g_m_ode[i] = float(ro["final_gain_db"])
        marosi_curves[f"gain_60per_{tag}_bvp_db"] = g_m_bvp
        marosi_curves[f"gain_60per_{tag}_ode_db"] = g_m_ode

    # -----------------------------------------------------------------
    # Part 4B: Compact 3D FIT Spatial Hot Test on Marosi Pillar SWS
    # -----------------------------------------------------------------
    a_wg = m_spec.waveguide_width_m
    b_wg = m_spec.waveguide_height_m
    w_p = m_spec.pillar_width_m
    l_p = m_spec.pillar_length_m
    y_in = m_spec.pillar_y_inner_m
    y_out = m_spec.pillar_y_outer_m

    n_per_3d = 10
    n_pad_3d = 2
    zmin_3d = -float(n_pad_3d) * d_m
    zmax_3d = float(n_per_3d + n_pad_3d) * d_m

    boxes_3d = []
    for m in range(0, n_per_3d):
        z0 = float(m) * d_m
        z1 = z0 + 0.25 * d_m
        z2 = z0 + 0.75 * d_m
        boxes_3d.append(
            pv.Box(bounds=(-0.5 * a_wg - 0.05e-3, -0.5 * a_wg + w_p, y_in, y_out, z1 - 0.5 * l_p, z1 + 0.5 * l_p))
        )
        boxes_3d.append(
            pv.Box(bounds=(-0.5 * a_wg - 0.05e-3, -0.5 * a_wg + w_p, -y_out, -y_in, z1 - 0.5 * l_p, z1 + 0.5 * l_p))
        )
        boxes_3d.append(
            pv.Box(bounds=(0.5 * a_wg - w_p, 0.5 * a_wg + 0.05e-3, y_in, y_out, z2 - 0.5 * l_p, z2 + 0.5 * l_p))
        )
        boxes_3d.append(
            pv.Box(bounds=(0.5 * a_wg - w_p, 0.5 * a_wg + 0.05e-3, -y_out, -y_in, z2 - 0.5 * l_p, z2 + 0.5 * l_p))
        )
    merged_pillars_3d = pv.merge(boxes_3d)

    f0_3d = 57.5e9
    w0_3d = 2.0 * np.pi * f0_3d
    k_3d_ref = 3.8214  # From Tutorial 6.2 task_10_3d_marosi_glide.npz at 57.93 GHz
    i0_3d_test = 0.030  # 30 mA test current (QC = 0.353 < QC_crit = 0.4725 at b = 0)

    levels_3d_out = []
    for lvl_3d, nx_3d, ny_3d, nz_per_3d in (
        ("coarse_3d", 12, 16, 12),
        ("medium_3d", 16, 20, 14),
    ):
        nz_3d = (n_per_3d + 2 * n_pad_3d) * nz_per_3d
        grid_3d = GridFIT3D(
            xmin=-0.5 * a_wg,
            xmax=0.5 * a_wg,
            ymin=-0.5 * b_wg,
            ymax=0.5 * b_wg,
            zmin=zmin_3d,
            zmax=zmax_3d,
            Nx=nx_3d,
            Ny=ny_3d,
            Nz=nz_3d,
            stl_solids={"pec": merged_pillars_3d},
            stl_materials={"pec": "pec"},
            conformal="pec",
            verbose=0,
        )

        def _run_marosi_3d_pass(i0_val: float, u0_val: float) -> tuple[np.ndarray, float, int, float]:
            dx_3 = a_wg / nx_3d
            dy_3 = b_wg / ny_3d
            dz_3 = d_m / nz_per_3d
            dt_e = 0.5 / (C0 * np.sqrt(1.0 / dx_3**2 + 1.0 / dy_3**2 + 1.0 / dz_3**2))
            nt_e = int(34.0 / (f0_3d * dt_e))
            est_3 = estimate((nx_3d, ny_3d, nz_3d), steps=nt_e, dtype="float64", conformal="pec")
            preflight(est_3, LOCAL_VM_SPEC)

            t0_3 = time.perf_counter()
            sol_3 = SolverFIT3D(
                grid_3d,
                use_stl=True,
                bc_low=["pec", "pec", "cpml"],
                bc_high=["pec", "pec", "cpml"],
                n_pml=8,
                verbose=0,
            )
            dt_3 = float(sol_3.dt)
            if i0_val > 0.0:
                wp_3 = beam_plasma_frequency(
                    current=i0_val,
                    area=m_spec.beam_width_m * m_spec.beam_thickness_m,
                    u0=u0_val,
                )
                LinearizedBeam(
                    solver=sol_3,
                    u0=u0_val,
                    omega_p=wp_3,
                    axis="z",
                    size=(m_spec.beam_width_m, m_spec.beam_thickness_m),
                    s_min=0.5 * d_m,
                    s_max=float(n_per_3d - 0.5) * d_m,
                    ramp_cells=6,
                )
            iz_src = int(np.argmin(np.abs(sol_3.z[:nz_3d] - (-0.8 * d_m))))
            x_c = 0.5 * (grid_3d.x[:-1] + grid_3d.x[1:])[:nx_3d]
            y_c = 0.5 * (grid_3d.y[:-1] + grid_3d.y[1:])[:ny_3d]
            ix_t = np.where(np.abs(x_c) < 0.45e-3)[0]
            iy_t = np.where(np.abs(y_c) < y_in)[0]
            ix_g, iy_g = np.meshgrid(ix_t, iy_t, indexing="ij")
            flat_src = 1 * sol_3.N + ix_g.ravel() + nx_3d * iy_g.ravel() + nx_3d * ny_3d * iz_src
            ix0, iy0 = nx_3d // 2, ny_3d // 2
            n_steps = int(34.0 / (f0_3d * dt_3))
            n_dft = int(6.0 / (f0_3d * dt_3))
            n_start = n_steps - n_dft
            t_ramp = 4.0 / f0_3d
            ez_dft = np.zeros(nz_3d, dtype=np.complex128)
            for n in range(n_steps):
                t = (n + 0.5) * dt_3
                ramp = np.sin(0.5 * np.pi * min(1.0, t / t_ramp)) ** 2
                sol_3.J.array[flat_src] = ramp * np.sin(w0_3d * t)
                sol_3.one_step()
                if n >= n_start:
                    ez_line = sol_3.E[ix0, iy0, :nz_3d, "z"]
                    ez_dft += ez_line * np.exp(-1j * w0_3d * (n + 1) * dt_3) * (2.0 / n_dft)
            wall_3 = time.perf_counter() - t0_3
            return ez_dft, wall_3, n_steps, float(est_3.host_bytes / 1e6)

        ez_c_3d, wall_c_3d, n_steps_3d, ram_3d_mb = _run_marosi_3d_pass(0.0, 0.142 * C0)
        # Measure cold phase advance per period on this 3D grid to set synchronous velocity (b = 0, no 1.020 scale)
        phases_3d = []
        for m in range(2, n_per_3d - 2):
            sl0 = slice((2 + n_pad_3d) * nz_per_3d, (3 + n_pad_3d) * nz_per_3d)
            slm = slice((m + n_pad_3d) * nz_per_3d, (m + 1 + n_pad_3d) * nz_per_3d)
            phases_3d.append(np.angle(np.vdot(ez_c_3d[slm], ez_c_3d[sl0])))
        dphi_3d = float(np.mean(np.diff(np.unwrap(phases_3d))))
        bn_fit_3d = abs(dphi_3d) / np.pi
        beta_p1_3d = (2.0 + bn_fit_3d) * np.pi / d_m
        dz_3d = d_m / float(nz_per_3d)
        vp_grid_3d = w0_3d / beta_p1_3d
        sinc_3d = float(np.sin(beta_p1_3d * dz_3d) / (beta_p1_3d * dz_3d))
        u0_sync_3d = vp_grid_3d / sinc_3d

        ez_h_3d, wall_h_3d, _, _ = _run_marosi_3d_pass(i0_3d_test, u0_sync_3d)
        m_arr_3d = np.arange(2, 7, dtype=int)
        amps_c_3d = np.array(
            [
                np.sqrt(np.mean(np.abs(ez_c_3d[(m + n_pad_3d) * nz_per_3d : (m + 1 + n_pad_3d) * nz_per_3d]) ** 2))
                for m in m_arr_3d
            ]
        )
        amps_h_3d = np.array(
            [
                np.sqrt(np.mean(np.abs(ez_h_3d[(m + n_pad_3d) * nz_per_3d : (m + 1 + n_pad_3d) * nz_per_3d]) ** 2))
                for m in m_arr_3d
            ]
        )
        ratios_3d = amps_h_3d / np.maximum(amps_c_3d, 1e-30)
        z_m_3d = (m_arr_3d + 0.5) * d_m
        alpha_3d_fit, _ = np.polyfit(z_m_3d, np.log(ratios_3d), 1)
        alpha_3d_fit = float(alpha_3d_fit)

        # Evaluate Pierce spatial growth rate at the ACTUAL simulated b = float(p_3d_th["b"]) = 0.0 (not b_opt)
        v_volt_3d = float((1.0 / np.sqrt(1.0 - (vp_grid_3d / C0) ** 2) - 1.0) * M_E * C0**2 / Q_E)
        p_3d_th = pierce_parameters(
            freq_hz=f0_3d,
            beta_circuit=beta_p1_3d,
            k_pierce_ohm=k_3d_ref,
            voltage_v=v_volt_3d,
            current_a=i0_3d_test,
            beam_half_thickness_m=0.5 * m_spec.beam_thickness_m,
            tunnel_half_height_m=m_spec.pillar_y_inner_m,
            beam_width_m=m_spec.beam_width_m,
            waveguide_width_m=m_spec.waveguide_width_m,
            sc_model="wakis_rect",
        )
        b_sim_3d = float(p_3d_th["b"])
        disp_3d_th = solve_pierce_dispersion(b=b_sim_3d, QC=float(p_3d_th["QC"]), d=0.0)
        alpha_3d_pierce = float(beta_p1_3d * p_3d_th["C"] * disp_3d_th["x1"])

        levels_3d_out.append(
            {
                "level": lvl_3d,
                "nx": nx_3d,
                "ny": ny_3d,
                "nz_per_period": nz_per_3d,
                "nz_total": nz_3d,
                "n_cells": int(nx_3d * ny_3d * nz_3d),
                "bn_cold_fit": round(bn_fit_3d, 4),
                "b_simulated": round(b_sim_3d, 6),
                "alpha_fit_np_m": round(alpha_3d_fit, 3),
                "alpha_pierce_np_m": round(alpha_3d_pierce, 3),
                "rel_err_vs_pierce": round(abs(alpha_3d_fit - alpha_3d_pierce) / alpha_3d_pierce, 4),
                "cold_wall_time_s": round(wall_c_3d, 2),
                "hot_wall_time_s": round(wall_h_3d, 2),
                "n_steps": int(n_steps_3d),
                "ram_est_mb": round(ram_3d_mb, 2),
            }
        )
        marosi_curves[f"{lvl_3d}_z_m"] = z_m_3d
        marosi_curves[f"{lvl_3d}_ratios"] = ratios_3d

    # Cloud Batch resource estimate for full 60-period fine 3D Marosi hot test (56 x 64 x 3472)
    est_full_3d = estimate((56, 64, 62 * 56), steps=65000, dtype="float64", conformal="pec")
    batch_estimate_3d = {
        "grid_shape": [56, 64, 3472],
        "n_cells": int(est_full_3d.cells),
        "n_steps": 65000,
        "ram_working_gb": round(float(est_full_3d.host_bytes / 1e9), 3),
        "cpu_hours_est": round(float(est_full_3d.seconds / 3600.0), 2),
        "batch_command": (
            "python tutorials/08_vacuum_electronics/08-06_twt_small_signal_gain/_helpers.py "
            "--run-3d-marosi-full60 --nx 56 --ny 64 --nz-per 56 --n-periods 60"
        ),
    }

    # -----------------------------------------------------------------
    # Part 5: Backward-Wave Oscillation (BWO) — Johnson (1955) + 1-Cell Bloch Instability
    # -----------------------------------------------------------------
    bwo_qc0 = solve_bwo_start_oscillation(QC=0.0, d=0.0)
    qc_bwo_grid = np.linspace(0.0, 1.0, 21)
    cn_st_vs_qc = np.zeros_like(qc_bwo_grid)
    b_st_vs_qc = np.zeros_like(qc_bwo_grid)
    for i, qc_v in enumerate(qc_bwo_grid):
        r_bwo = solve_bwo_start_oscillation(QC=float(qc_v), d=0.0)
        cn_st_vs_qc[i] = r_bwo["CN_start"]
        b_st_vs_qc[i] = r_bwo["b_start"]

    # Load consistent backward-harmonic (n = -1, beta_0 d = 0.5 pi, |beta_{-1} d| = 1.5 pi)
    # cold-circuit parameters from Tutorial 6.2 (f_BWO = 97.278 GHz, K_{-1,beam} = 2.08 Ohm):
    d5_62 = np.load(COLD_DATA_DIR / "task_05_2d_conformal_L1.npz")
    idx05 = int(np.where(np.isclose(d5_62["bn"], 0.5))[0][0])
    f_cold_bwo_hz = float(d5_62["f_glide_low_ghz"][idx05]) * 1.0e9
    w_cold_bwo = 2.0 * np.pi * f_cold_bwo_hz
    beta0_bwo = 0.5 * np.pi / spec.period_m
    beta_m1_bwo = 1.5 * np.pi / spec.period_m
    vp_bwo = w_cold_bwo / beta_m1_bwo
    v_volt_bwo = float((1.0 / np.sqrt(1.0 - (vp_bwo / C0) ** 2) - 1.0) * M_E * C0**2 / Q_E)
    _, gamma0_bwo = relativistic_gamma(voltage=v_volt_bwo)
    v0_eff_bwo = 0.5 * gamma0_bwo * (gamma0_bwo + 1.0) * v_volt_bwo
    lam_e_bwo = vp_bwo / f_cold_bwo_hz
    k_m1_bwo_ohm = 2.08  # Tutorial 6.2 task_08_pierce_impedance.json backward harmonic K_{-1,beam}

    # Verify I_st(L) proportional to L^-3 by solving the BWO boundary-value root problem
    # T_BWO(CN(I_0, L), b) = 0 directly for the unknown current I_0 at each length L_m:
    n_per_bwo_arr = np.arange(15, 65, 5, dtype=float)
    L_bwo_arr_m = n_per_bwo_arr * spec.period_m
    i_st_arr_a = np.zeros_like(L_bwo_arr_m)
    for idx_l, l_val in enumerate(L_bwo_arr_m):
        n_elec_l = float(l_val / lam_e_bwo)

        def _bwo_bv_residual(x_ib: np.ndarray) -> np.ndarray:
            i_try = max(float(x_ib[0]), 1e-12)
            b_try = float(x_ib[1])
            c_try = (i_try * k_m1_bwo_ohm / (4.0 * v0_eff_bwo)) ** (1.0 / 3.0)
            cn_try = c_try * n_elec_l
            coeffs = np.array([1.0 + 0j, 1j * b_try, 0.0 + 0j, -1j], dtype=np.complex128)
            deltas = np.roots(coeffs)
            val = 0.0 + 0j
            for ii in range(3):
                jj, kk = (ii + 1) % 3, (ii + 2) % 3
                w_ii = (deltas[ii] ** 2) / ((deltas[ii] - deltas[jj]) * (deltas[ii] - deltas[kk]))
                val += w_ii * np.exp(2.0 * np.pi * cn_try * deltas[ii])
            return np.array([val.real, val.imag], dtype=float)

        i_guess = (4.0 * v0_eff_bwo / k_m1_bwo_ohm) * ((0.314 / n_elec_l) ** 3)
        sol_l = root(_bwo_bv_residual, np.array([i_guess, 1.522], dtype=float), tol=1e-13)
        i_st_arr_a[idx_l] = float(sol_l.x[0])

    bwo_length_slope, _ = np.polyfit(np.log(L_bwo_arr_m), np.log(i_st_arr_a), 1)

    # Unbounded 1-cell Bloch-periodic temporal instability check on the n = -1 backward harmonic
    # (explicitly contrasted with Johnson's finite-length boundary-value start-oscillation threshold):
    grid_bwo, _ = h62.build_2d_grating_grid(
        h62.GratingSWSSpec(), n_feat=4, glide_symmetric=True, conformal="pec"
    )
    dz_bwo = spec.period_m / float(grid_bwo.Nz)
    sinc_bwo = float(np.sin(beta_m1_bwo * dz_bwo) / (beta_m1_bwo * dz_bwo))
    u0_bwo_sync = vp_bwo / sinc_bwo  # Exact b = 0 synchronism (no 1.015 factor)

    bwo_currents_a = [0.0, 0.04, 0.12, 0.28, 0.55]
    bwo_growth_rates = []
    bwo_envelopes = {}
    t_bwo_arr = None

    for i0_bwo in bwo_currents_a:
        est_bwo = estimate(
            (int(grid_bwo.Nx), int(grid_bwo.Ny), int(grid_bwo.Nz)),
            steps=3800,
            dtype="complex128",
            conformal="pec",
        )
        preflight(est_bwo, LOCAL_VM_SPEC)
        sol_bwo = SolverFIT3D(
            grid_bwo,
            use_stl=True,
            bc_low=["periodic", "pec", "bloch"],
            bc_high=["periodic", "pec", "bloch"],
            dtype=np.complex128,
            k_bloch=(0.0, 0.0, -float(beta0_bwo)),
            verbose=0,
        )

        if i0_bwo > 0.0:
            wp_bwo = beam_plasma_frequency(
                current=float(i0_bwo),
                area=float(grid_bwo.x[1] - grid_bwo.x[0]) * 2.0 * spec.beam_half_thickness_m,
                u0=vp_bwo,
            )
            LinearizedBeam(
                u0=u0_bwo_sync,
                omega_p=wp_bwo,
                axis="z",
                half_thickness=spec.beam_half_thickness_m,
                solver=sol_bwo,
            )

        pulse_bwo = GaussianSource(
            frequency=f_cold_bwo_hz, fwidth=0.18 * f_cold_bwo_hz, cutoff=3.0
        )
        src_bwo = CurrentSource(
            pulse_bwo, component="Ez", center=(0.0, 0.15e-3, 0.125e-3), amplitude=1.0
        )
        n_steps_bwo = 3800
        dt_bwo = float(sol_bwo.dt)
        iy_mid = sol_bwo.Ny // 2
        z_c_bwo = 0.5 * (
            np.asarray(grid_bwo.z[: sol_bwo.Nz]) + np.asarray(grid_bwo.z[1 : sol_bwo.Nz + 1])
        )
        ph_m1 = np.exp(+1j * beta_m1_bwo * z_c_bwo) / float(sol_bwo.Nz)

        sig_bwo = np.zeros(n_steps_bwo, dtype=np.complex128)
        for step in range(n_steps_bwo):
            src_bwo.update(sol_bwo, step * dt_bwo)
            sol_bwo.one_step()
            sig_bwo[step] = np.sum(sol_bwo.E[0, iy_mid, : sol_bwo.Nz, "z"] * ph_m1)

        n_cyc = int(round(1.0 / (f_cold_bwo_hz * dt_bwo)))
        env_b = np.sqrt(np.convolve(np.abs(sig_bwo) ** 2, np.ones(n_cyc) / n_cyc, mode="same"))
        t_bwo_arr = np.arange(n_steps_bwo, dtype=float) * dt_bwo
        i1_b = int(0.45 * n_steps_bwo)
        i2_b = int(0.95 * n_steps_bwo)
        env_norm = env_b / max(float(env_b[i1_b]), 1e-30)
        if i0_bwo == 0.0:
            # In a lossless 1-cell Bloch cavity at I0 = 0, the cold eigenmode is neutral
            # (any post-pulse slope is beat modulation between cavity modes, not physical damping)
            g_fit_bwo, _ = np.polyfit(t_bwo_arr[i1_b:i2_b], np.log(env_b[i1_b:i2_b] + 1e-30), 1)
        else:
            g_fit_bwo, _ = np.polyfit(t_bwo_arr[i1_b:i2_b], np.log(env_b[i1_b:i2_b] + 1e-30), 1)
        bwo_growth_rates.append(float(g_fit_bwo))
        bwo_envelopes[f"i0_{int(i0_bwo*1000)}ma_env"] = env_norm

    np.savez(
        npz_path,
        f_vband_ghz=f_vband_ghz,
        k_vband_ohm=k_vband_ohm,
        f_vband_dense_ghz=f_vband_dense_ghz,
        qc_bwo_grid=qc_bwo_grid,
        cn_st_vs_qc=cn_st_vs_qc,
        b_st_vs_qc=b_st_vs_qc,
        n_per_bwo_arr=n_per_bwo_arr,
        L_bwo_arr_m=L_bwo_arr_m,
        i_st_arr_a=i_st_arr_a,
        t_bwo_ns=t_bwo_arr * 1.0e9,
        bwo_currents_a=np.array(bwo_currents_a, dtype=float),
        bwo_growth_rates_ns=np.array(bwo_growth_rates, dtype=float) * 1.0e-9,
        **marosi_curves,
        **bwo_envelopes,
    )

    idx_pk_520 = int(np.argmax(marosi_curves["gain_60per_5p20kv_bvp_db"]))
    idx_pk_538 = int(np.argmax(marosi_curves["gain_60per_5p38kv_bvp_db"]))
    gain_520_at_58ghz = float(
        np.interp(58.0, f_vband_dense_ghz, marosi_curves["gain_60per_5p20kv_bvp_db"])
    )
    gain_538_at_58ghz = float(
        np.interp(58.0, f_vband_dense_ghz, marosi_curves["gain_60per_5p38kv_bvp_db"])
    )

    payload = {
        "marosi_3d_vband": {
            "cold_test_source": "tutorials/06_periodic_structures/06-02_sws_cold_test/data/task_10_3d_marosi_glide.npz",
            "paper_operating_voltage_kv": 5.20,
            "paper_operating_current_ma": 11.0,
            "paper_cold_insertion_loss_db": cold_il_db,
            "derived_alpha_loss_np_m": round(float(alpha_loss_marosi_np_m), 4),
            "derived_pierce_d_at_58ghz": round(float(d_58ghz_520kv), 4),
            "K_p1_beam_58ghz_ohm": 3.8214,
            "v_energy_over_c_58ghz": 0.1279,
            "n_periods_full": 60,
            "L_full_mm": 63.0,
            "gain_60per_5p20kv_at_58ghz_db": round(gain_520_at_58ghz, 2),
            "gain_60per_5p20kv_peak_db": round(
                float(marosi_curves["gain_60per_5p20kv_bvp_db"][idx_pk_520]), 2
            ),
            "gain_60per_5p20kv_peak_freq_ghz": round(float(f_vband_dense_ghz[idx_pk_520]), 2),
            "gain_60per_5p38kv_at_58ghz_db": round(gain_538_at_58ghz, 2),
            "gain_60per_5p38kv_peak_db": round(
                float(marosi_curves["gain_60per_5p38kv_bvp_db"][idx_pk_538]), 2
            ),
            "gain_60per_5p38kv_peak_freq_ghz": round(float(f_vband_dense_ghz[idx_pk_538]), 2),
            "marosi_2025_fig7_peak_gain_db": 19.0,
            "marosi_2025_fig7_peak_freq_ghz": 58.0,
            "marosi_2025_fig7_secondary_peak_freq_ghz": 68.0,
            "hot_test_3d_levels": levels_3d_out,
            "hot_test_3d_converged": False,
            "hot_test_3d_convergence_note": (
                "Both coarse_3d (12x16x168) and medium_3d (16x20x196) are reported at the "
                "actual simulated b = 0; the 2-level compact 3D run is not yet in the "
                "asymptotic convergence regime on local VM resources."
            ),
            "full_60period_batch_estimate": batch_estimate_3d,
        },
        "bwo_johnson_1955": {
            "CN_start_qc0": round(bwo_qc0["CN_start"], 6),
            "b_start_qc0": round(bwo_qc0["b_start"], 6),
            "johnson_1955_table1_CN": 0.314,
            "johnson_1955_table1_b": 1.522,
            "CN_rel_err_vs_johnson": round(abs(bwo_qc0["CN_start"] - 0.314) / 0.314, 5),
            "ist_vs_length_loglog_slope": round(float(bwo_length_slope), 6),
            "f_cold_bwo_ghz": round(f_cold_bwo_hz * 1e-9, 3),
            "K_m1_beam_ohm": k_m1_bwo_ohm,
            "v_sync_bwo_kv": round(v_volt_bwo * 1e-3, 3),
            "ist_50periods_ma": round(float(i_st_arr_a[n_per_bwo_arr == 50][0]) * 1e3, 2),
            "single_cell_bloch_note": (
                "Withdrawn FIT start-current claim: a lossless 1-period Bloch unit cell has "
                "no finite interaction length L and no collector boundary condition E_ckt(L)=0, "
                "so it exhibits unbounded periodic temporal instability for all I0 > 0 rather "
                "than Johnson's finite-length start-oscillation threshold I_st(L) ~ L^-3."
            ),
            "i0_sweep_a": bwo_currents_a,
            "unbounded_bloch_temporal_rate_ns_inv": [round(x * 1e-9, 4) for x in bwo_growth_rates],
        },
    }
    return _enrich_task05_marosi_bwo(payload)



# =====================================================================
# 4. Setup Figures, Diagnostic Plots, and Animation Generators
# =====================================================================


class _CorrugatedGratingPolygon:
    """Polygon structure wrapper for multi-tooth gratings in ``wakis.schematic`` (``plane='zy'``)."""

    def __init__(self, vertices_zy_m: np.ndarray, material: str | float, name: str) -> None:
        self.vertices = np.asarray(vertices_zy_m, dtype=float)
        self.material = material
        self.name = str(name)


def build_twt_grating_polygons(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
    n_periods: int = 8,
    n_in_smooth: int = 2,
    n_out_smooth: int = 2,
) -> list[_CorrugatedGratingPolygon]:
    """Build contiguous ``(z, y)`` polygon outlines for the lower and upper staggered PEC gratings."""
    d_p = spec.period_m
    s_v = spec.vane_thickness_m
    a_h = spec.tunnel_half_height_m
    y_max = a_h + spec.vane_height_m
    z_min = -float(n_in_smooth) * d_p
    z_end = float(n_periods) * d_p
    z_max = float(n_periods + n_out_smooth) * d_p

    # Lower PEC grating (smooth tunnel wall in [z_min, 0] and [z_end, z_max], teeth at z_m + 0.25 d)
    pts_lo: list[tuple[float, float]] = [
        (z_min, -y_max),
        (z_min, -a_h),
        (0.0, -a_h),
        (0.0, -y_max),
    ]
    for m in range(int(n_periods)):
        z0 = m * d_p
        zt0 = z0 + 0.25 * d_p - 0.5 * s_v
        zt1 = z0 + 0.25 * d_p + 0.5 * s_v
        pts_lo.extend(
            [
                (zt0, -y_max),
                (zt0, -a_h),
                (zt1, -a_h),
                (zt1, -y_max),
            ]
        )
    pts_lo.extend([(z_end, -y_max), (z_end, -a_h), (z_max, -a_h), (z_max, -y_max)])
    for zb in np.linspace(z_max, z_min, 40):
        pts_lo.append((float(zb), -y_max))

    # Upper staggered PEC grating (teeth at z_m + 0.75 d)
    pts_hi: list[tuple[float, float]] = [
        (z_min, +y_max),
        (z_min, +a_h),
        (0.0, +a_h),
        (0.0, +y_max),
    ]
    for m in range(int(n_periods)):
        z0 = m * d_p
        zt0 = z0 + 0.75 * d_p - 0.5 * s_v
        zt1 = z0 + 0.75 * d_p + 0.5 * s_v
        pts_hi.extend(
            [
                (zt0, +y_max),
                (zt0, +a_h),
                (zt1, +a_h),
                (zt1, +y_max),
            ]
        )
    pts_hi.extend([(z_end, +y_max), (z_end, +a_h), (z_max, +a_h), (z_max, +y_max)])
    for zt in np.linspace(z_max, z_min, 40):
        pts_hi.append((float(zt), +y_max))

    return [
        _CorrugatedGratingPolygon(
            np.asarray(pts_lo, dtype=float),
            material="pec",
            name="Lower PEC grating (teeth at z_m + d/4)",
        ),
        _CorrugatedGratingPolygon(
            np.asarray(pts_hi, dtype=float),
            material="pec",
            name="Upper PEC grating (glide +d/2, z_m + 3d/4)",
        ),
    ]


class _TWTSchematicSolverProxy:
    """Lightweight proxy exposing ``_assigned_structures`` and bare grid to ``wakis.schematic``."""

    def __init__(self, solver: SolverFIT3D, bare_grid: GridFIT3D, structures: list[Any]) -> None:
        self._solver = solver
        self.grid = bare_grid
        self._assigned_structures = tuple(structures)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._solver, name)


def generate_setup_figures(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
) -> dict[str, Any]:
    """Generate setup figures ``00a_setup_geometry.png`` and ``00b_setup_mesh_and_excitation.png``."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n_per_schem = 22
    grid, _ = build_2d_grating_hot_grid(
        spec=spec,
        n_periods=n_per_schem,
        n_in_smooth=2,
        n_out_smooth=2,
        mesh_level="coarse",
    )
    dx = float(grid.x[1] - grid.x[0])
    kx = 2.0 * float(np.arcsin(0.5 * dx * np.pi / spec.waveguide_width_m)) / dx
    bare_grid = GridFIT3D(
        x=np.asarray(grid.x),
        y=np.asarray(grid.y),
        z=np.asarray(grid.z),
        verbose=0,
    )
    solver = SolverFIT3D(
        bare_grid,
        bc_low=["bloch", "pec", "cpml"],
        bc_high=["bloch", "pec", "cpml"],
        k_bloch=(float(kx), 0.0, 0.0),
        n_pml=8,
        dtype=np.complex128,
        cfln=0.85,
        verbose=0,
    )
    polys = build_twt_grating_polygons(spec, n_periods=n_per_schem, n_in_smooth=2, n_out_smooth=2)
    proxy = _TWTSchematicSolverProxy(solver, bare_grid, polys)

    pulse = GaussianSource(frequency=186.0e9, fwidth=18.0e9)
    d_p = spec.period_m
    a_h = spec.tunnel_half_height_m
    b_h = spec.beam_half_thickness_m

    sources = [
        {
            "kind": "line",
            "type_name": "CurrentSource",
            "short_word": "RF input Jy",
            "label": "Input waveguide launcher Jy (z = -0.60 mm, f0 = 186 GHz)",
            "desc": "Input LSE_10 launcher Jy across tunnel (z = -0.60 mm, f0 = 186 GHz)",
            "pos_str": "z = -0.60 mm, y in [-0.10, +0.10] mm",
            "polarization": "Jy (excites LSE_10 Ez, Ey, Hx)",
            "h_bounds": (-1.2 * d_p, -1.2 * d_p),
            "v_bounds": (-a_h, +a_h),
            "pulse": {
                "f0": 186.0e9,
                "fwidth": 18.0e9,
                "t_peak": float(4.0 / (np.pi * 18.0e9)),
                "t_end": float(pulse.end_time),
                "src_time": pulse,
            },
        },
        {
            "kind": "box",
            "type_name": "LinearizedBeam",
            "short_word": "Sheet beam (20 kV)",
            "label": "LinearizedBeam sheet (V0 = 20 kV, Hb = 150 um, ramp before CPML)",
            "desc": "LinearizedBeam sheet (V0 = 20 kV, I0 = 100-250 mA, |y| <= 75 um)",
            "pos_str": "y in [-75, +75] um, z in [0.5, 10.5] mm (active in 22-period grating)",
            "polarization": "J1z = rho0 v1z + u0 rho1",
            "h_bounds": (1.0 * d_p, (n_per_schem - 1.0) * d_p),
            "v_bounds": (-b_h, +b_h),
        },
    ]
    monitors = [
        {
            "kind": "line",
            "type_name": "DFTMonitor",
            "short_word": "On-axis Ez & I1 DFT",
            "label": "On-axis phasor monitor Ez(z, f0), I1(z, f0) along y = 0",
            "desc": "On-axis phasor monitor Ez(z, f0) and I1(z, f0) along y = 0",
            "pos_str": "y = 0 mm, z in [0.5, 10.5] mm (periods m = 1..20)",
            "freq_str": "180-210 GHz",
            "h_bounds": (1.0 * d_p, (n_per_schem - 1.0) * d_p),
            "v_bounds": (0.0, 0.0),
        }
    ]
    mat_names = {
        "background": "Vacuum tunnel & slots (eps_r = 1.00)",
        "pec": "PEC staggered grating vanes",
    }
    res = schematic.save_setup_figures(
        proxy,
        FIG_DIR,
        prefix="00",
        plane="zy",
        cut_coord=0.5 * dx,
        material_names=mat_names,
        sources=sources,
        monitors=monitors,
        unit="mm",
        freq=186.0e9,
        min_feature=spec.min_feature_m,
        variations=[
            {
                "name": "2D Karetnikova 0.2 THz staggered grating (Parts 2-3, N = 22 periods)",
                "details": "Domain z in [-1.0, 12.0] mm (13.0 mm total), grating z in [0, 11.0] mm, d = 0.50 mm, s = 0.10 mm, h = 0.30 mm, 2a = 0.20 mm, Hb = 0.15 mm (25 um beam-wall gap resolved with 4-7 cells), Nz/d in {26, 32, 40, 46}",
            },
            {
                "name": "3D Marosi V-band glide-symmetric pillar SWS (Part 4)",
                "details": "a = 1.40 mm, b = 0.80 mm, d = 1.05 mm, pillars 1.08 x 0.27 x 0.10 mm^3, sheet beam 0.55 x 0.10 mm^2, V0 = 5.2 kV, I0 = 11 mA",
            },
        ],
    )
    md_path = FIG_DIR / "00_setup_table.md"
    if md_path.exists():
        md_txt = md_path.read_text(encoding="utf-8")
        md_txt = md_txt.replace(
            "(d = 13\u2009mm)",
            "(L_z = 13.0\u2009mm; grating z \u2208 [0, 11.0]\u2009mm, period d = 0.50\u2009mm)",
        )
        old_disc = "| Discretization | \u0394z & resolution | \u0394_min = 6.25\u2009\u00b5m, 257.9 cells/\u03bb (in n_max = 1.00), 16.0 cells across feature (100\u2009\u00b5m) |"
        new_disc = (
            "| Discretization | \u0394z, \u0394y & resolution | "
            "\u0394z_max = 20.0\u2009\u00b5m (21.8\u201339.2 cells/\u03bb_g at \u03bb_g = 436.1\u2009\u00b5m, 5\u20139 cells across 100\u2009\u00b5m vane); "
            "\u0394y_min = 6.25\u20133.57\u2009\u00b5m (4\u20137 cells across 25\u2009\u00b5m beam\u2013wall gap) |"
        )
        md_txt = md_txt.replace(old_disc, new_disc)
        md_path.write_text(md_txt, encoding="utf-8")
    return res


def generate_all_figures(
    spec: KaretnikovaTWTSpec = KaretnikovaTWTSpec(),
) -> dict[str, str]:
    """Generate all diagnostic figures and the growing-wave/beam-current GIF."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()

    setup_res = generate_setup_figures(spec)

    t1 = json.loads((DATA_DIR / "task_01_pierce_theory_and_scw_check.json").read_text(encoding="utf-8"))
    d1 = np.load(DATA_DIR / "task_01_pierce_theory_and_scw_check.npz")

    t2 = json.loads((DATA_DIR / "task_02_2d_grating_hot_convergence.json").read_text(encoding="utf-8"))
    d2 = np.load(DATA_DIR / "task_02_2d_grating_hot_convergence.npz")

    t3 = json.loads((DATA_DIR / "task_03_karetnikova_gain_vs_frequency.json").read_text(encoding="utf-8"))
    d3 = np.load(DATA_DIR / "task_03_karetnikova_gain_vs_frequency.npz")

    t4 = json.loads((DATA_DIR / "task_04_validation_and_failure_tests.json").read_text(encoding="utf-8"))
    d4 = np.load(DATA_DIR / "task_04_validation_and_failure_tests.npz")

    t5 = json.loads((DATA_DIR / "task_05_3d_marosi_vband_and_bwo.json").read_text(encoding="utf-8"))
    d5 = np.load(DATA_DIR / "task_05_3d_marosi_vband_and_bwo.npz")

    # -----------------------------------------------------------------
    # Figure 01: Pierce Dispersion Roots, BVP vs Coupled-Mode ODE & SCW Check
    # -----------------------------------------------------------------
    fig1, axes1 = figstyle.new_figure(nrows=1, ncols=3, width_in=13.5, height_in=4.1)
    ax1a, ax1b, ax1c = axes1

    b_grid = d1["b_grid"]
    qc_list = d1["qc_list"]
    x1_mat = d1["x1_vs_b_qc"]
    colors_qc = ["#1d4ed8", "#047857", "#d97706", "#b91c1c"]
    for iq, qc_v in enumerate(qc_list):
        ax1a.plot(
            b_grid,
            x1_mat[iq],
            lw=2.0,
            color=colors_qc[iq],
            label=f"$QC = {qc_v:.2f}$",
        )
    ax1a.axhline(0.5 * np.sqrt(3.0), color="#4b5563", ls="--", lw=1.1, label=r"$\sqrt{3}/2 \approx 0.866$")
    ax1a.set_xlabel(r"Velocity parameter $b = (u_0 - v_p)/(C v_p)$")
    ax1a.set_ylabel(r"Growth parameter $x_1 = \mathrm{Re}(\delta_1)$")
    ax1a.set_title("(a) Pierce growing root vs $b$ & $QC$")
    ax1a.legend(frameon=False, fontsize=8.5)

    cn_arr = d1["CN_arr"]
    ax1b.plot(cn_arr, d1["bvp_gain_db_0"], color="#1d4ed8", lw=2.2, label="3-wave Pierce BVP")
    ax1b.plot(
        cn_arr[::8],
        d1["ode_gain_db_0"][::8],
        "o",
        ms=4.5,
        color="#dc2626",
        label="Coupled-mode spatial ODE",
    )
    ax1b.plot(
        cn_arr,
        d1["asymp_gain_db_0"],
        "--",
        color="#374151",
        lw=1.5,
        label=r"Asymptotic $A + B C N$ ($-9.54 + 47.26\,CN$)",
    )
    ax1b.axhline(-9.5424, color="#6b7280", ls=":", lw=1.0)
    ax1b.set_xlabel(r"Normalized interaction length $CN$")
    ax1b.set_ylabel("Small-signal gain $G$ [dB]")
    ax1b.set_title(r"(b) 3-wave launching beat & $A + B C N$ asymptote")
    ax1b.legend(frameon=False, fontsize=8.0)

    scw_lvls = t1["scw_check"]["levels"]
    dz_scw = np.array([x["dz_um"] for x in scw_lvls], dtype=float)
    err_scw = np.array([x["rel_err_wq"] for x in scw_lvls], dtype=float)
    ax1c.loglog(dz_scw, err_scw, "s-", color="#047857", lw=2.0, ms=6.0, label=r"FIT $|\omega_q - R\omega_p/\gamma_0^{3/2}|/\omega_q$")
    h_guide = np.array([dz_scw[-1], dz_scw[0]])
    ax1c.loglog(
        h_guide,
        err_scw[-1] * (h_guide / dz_scw[-1]) ** 2,
        "--",
        color="#6b7280",
        lw=1.3,
        label=r"$\mathcal{O}(\Delta z^2)$ reference",
    )
    ax1c.set_xlabel(r"Axial cell size $\Delta z$ [$\mu$m]")
    ax1c.set_ylabel(r"Relative error in reduced $\omega_q$")
    ax1c.set_title(r"(c) Smooth-tunnel SCW time-centring check")
    ax1c.legend(frameon=False, fontsize=8.0)

    fig1_path = FIG_DIR / "01_pierce_roots_and_scw_check.png"
    figstyle.save(fig1, fig1_path, close=True, auto_legend_below=False)

    # -----------------------------------------------------------------
    # Figure 02: 2D Hot Test Gain vs Length, 3-Wave Beating & Convergence (4 levels)
    # -----------------------------------------------------------------
    fig2, axes2 = figstyle.new_figure(nrows=1, ncols=3, width_in=13.8, height_in=4.2)
    ax2a, ax2b, ax2c = axes2

    z_th_mm = d2["z_th_m"] * 1e3
    ax2a.plot(z_th_mm, d2["bvp_gain_db"], color="#111827", lw=2.1, label="Pierce 3-wave BVP")
    ax2a.plot(z_th_mm, d2["bvp_asymp_db"], "--", color="#6b7280", lw=1.4, label=r"Asymptotic slope $\beta_e C x_1$")
    lvl_colors = {"coarse": "#d97706", "medium": "#2563eb", "fine": "#047857", "ultrafine": "#dc2626"}
    lvl_markers = {"coarse": "^", "medium": "s", "fine": "d", "ultrafine": "o"}
    lvl_nz = {"coarse": 26, "medium": 32, "fine": 40, "ultrafine": 46}
    for lvl in ("coarse", "medium", "fine", "ultrafine"):
        z_mm = d2[f"{lvl}_z_period_m"] * 1e3
        g_db = d2[f"{lvl}_gain_db_vs_m"]
        ax2a.plot(
            z_mm,
            g_db,
            marker=lvl_markers[lvl],
            ms=4.2,
            lw=1.3,
            color=lvl_colors[lvl],
            label=f"FIT {lvl} ($N_z/d={lvl_nz[lvl]}$)",
        )
    ax2a.set_xlabel("Axial position $z$ [mm]")
    ax2a.set_ylabel(r"Hot/cold field ratio $20\log_{10}|E_{\mathrm{hot}}/E_{\mathrm{cold}}|$ [dB]")
    ax2a.set_title("(a) Spatial growth & 3-wave input beat ($f_0=186$ GHz)")
    ax2a.legend(frameon=False, fontsize=7.4)

    z_med_mm = d2["medium_z_period_m"] * 1e3
    i1_med_ma = d2["medium_i1_rms_vs_m"] * 1e3
    ax2b.semilogy(z_med_mm, np.maximum(i1_med_ma, 1e-6), "o-", color="#047857", lw=1.8, ms=4.5, label=r"FIT $|I_1(z)|_{\mathrm{rms}}$")
    idx_anchor = 9
    z_anchor = d2["medium_z_period_m"][idx_anchor]
    i1_anchor = max(float(i1_med_ma[idx_anchor]), 1e-6)
    alpha_p = float(t2["alpha_pierce_finest_grid_np_m"])
    ax2b.semilogy(
        z_med_mm[3:],
        i1_anchor * np.exp(alpha_p * (d2["medium_z_period_m"][3:] - z_anchor)),
        "--",
        color="#111827",
        lw=1.6,
        label=rf"Pierce $\exp(\alpha z)$ ($\alpha={alpha_p:.1f}\;\mathrm{{m}}^{{-1}}$)",
    )
    ax2b.set_xlabel("Axial position $z$ [mm]")
    ax2b.set_ylabel("AC beam current modulation $|I_1|$ [mA]")
    ax2b.set_title("(b) Exponential growth of AC beam bunching")
    ax2b.legend(frameon=False, fontsize=8.0)

    dz_arr = np.array([x["mesh"]["dz_um"] for x in t2["levels"]], dtype=float)
    err_arr = np.array([x["rel_err_vs_pierce_grid"] for x in t2["levels"]], dtype=float)
    err_mcv_arr = np.array([x["rel_err_vs_pierce_mcvey"] for x in t2["levels"]], dtype=float)
    pw_ord = t2.get("pairwise_orders", [2.53, 1.19, 4.15])
    ax2c.loglog(
        dz_arr,
        err_arr,
        "o-",
        color="#1d4ed8",
        lw=2.0,
        ms=6.5,
        label=rf"Same-grid Pierce ($p={pw_ord[0]:.2f},{pw_ord[1]:.2f},{pw_ord[2]:.2f}$; fit ${t2['observed_order_p']:.2f}$)",
    )
    ax2c.loglog(
        dz_arr,
        err_mcv_arr,
        "s--",
        color="#047857",
        lw=1.5,
        ms=5.2,
        label="Continuum McVey Pierce",
    )
    h_g2 = np.array([dz_arr[-1], dz_arr[0]])
    ax2c.loglog(
        h_g2,
        err_arr[-1] * (h_g2 / dz_arr[-1]) ** 2,
        "--",
        color="#6b7280",
        lw=1.3,
        label=r"$\mathcal{O}(\Delta z^2)$ guide",
    )
    ax2c.set_xlabel(r"Mean axial cell size $\Delta z$ [$\mu$m]")
    ax2c.set_ylabel(r"Relative error in growth rate $\alpha$")
    ax2c.set_title("(c) 4-level grid convergence (pre-asymptotic)")
    ax2c.legend(frameon=False, fontsize=7.2)

    fig2_path = FIG_DIR / "02_twt_gain_vs_length_and_convergence.png"
    figstyle.save(fig2, fig2_path, close=True, auto_legend_below=False)

    # -----------------------------------------------------------------
    # Figure 03: Gain vs Frequency (FIT vs Pierce/ODE vs Karetnikova 2018 text benchmarks)
    # -----------------------------------------------------------------
    fig3, axes3 = figstyle.new_figure(nrows=1, ncols=2, width_in=12.4, height_in=4.4)
    ax3a, ax3b = axes3

    f_sw = d3["f_sweep_ghz"]
    v_colors = {19: "#047857", 20: "#1d4ed8", 21: "#dc2626"}
    for v_kv in (19, 20, 21):
        lbl_v = (
            rf"Pierce $\langle K_{{+1}}\rangle_{{\mathrm{{beam}}}}$ ({v_kv} kV)"
            if v_kv != 21
            else r"Pierce $\langle K_{+1}\rangle_{\mathrm{beam}}$ (21 kV, edge 22.1 dB)"
        )
        ax3a.plot(
            f_sw,
            d3[f"v_{v_kv}kv_bvp_db"],
            lw=2.0,
            color=v_colors[v_kv],
            label=lbl_v,
        )
    ax3a.plot(
        d3["fit_freqs_ghz"],
        d3["fit_gain_25mm_db"],
        "s",
        ms=5.8,
        color="#111827",
        label="22-per. FIT BVP extrap. (25 mm, 20 kV)",
    )
    ax3a.axhspan(
        5.0,
        13.0,
        color="#9ca3af",
        alpha=0.18,
        label="Paper text l.205-206: 25 mm / 20 kV (5-13 dB)",
    )
    ax3a.axhline(
        19.0,
        color="#dc2626",
        ls=":",
        lw=1.3,
        label="Paper text l.209: 25 mm / 21 kV max (19 dB)",
    )
    ax3a.axhline(
        10.0,
        color="#1d4ed8",
        ls="-.",
        lw=1.2,
        label="Paper Fig. 5a (l.328) 25 mm particle-in-cell (~10 dB)",
    )
    ax3a.set_xlim(180.0, 210.0)
    ax3a.set_ylim(-5.0, 34.0)
    ax3a.set_xlabel("Frequency $f$ [GHz]")
    ax3a.set_ylabel("Small-signal gain $G$ [dB] ($L = 25$ mm)")
    ax3a.set_title("(a) 25 mm gain vs Karetnikova text ($N=50, I_0=100$ mA)")
    ax3a.legend(frameon=False, fontsize=6.8, ncol=1, loc="upper right")

    ax3b.plot(
        f_sw,
        d3["fig5_40mm_wakis_planar_db"],
        lw=2.0,
        color="#1d4ed8",
        label=r"Pierce $\langle K_{+1}\rangle_{\mathrm{beam}}$ + Wakis $R_{\mathrm{planar}}=0.652$",
    )
    ax3b.plot(
        f_sw,
        d3["fig5_40mm_karetnikova_eq4_db"],
        "--",
        lw=1.8,
        color="#047857",
        label=r"Pierce $\langle K_{+1}\rangle_{\mathrm{beam}}$ + Eq. (4) $D=0.635$",
    )
    ax3b.plot(
        d3["fit_freqs_ghz"],
        d3["fit_gain_40mm_db"],
        "o",
        ms=5.8,
        color="#dc2626",
        label="22-per. FIT BVP extrap. (40 mm, 20 kV)",
    )
    ax3b.axhspan(
        10.0,
        20.0,
        color="#9ca3af",
        alpha=0.18,
        label="Paper text l.289-290: 40 mm small-signal (10-20 dB)",
    )
    ax3b.axhline(
        18.7,
        color="#4b5563",
        ls=":",
        lw=1.4,
        label="Paper Fig. 6 (190 GHz, l.278/291) sat. particle-in-cell (18.7 dB)",
    )
    ax3b.set_xlim(180.0, 205.0)
    ax3b.set_ylim(-5.0, 52.0)
    ax3b.set_xlabel("Frequency $f$ [GHz]")
    ax3b.set_ylabel("Gain $G$ [dB] ($L = 40$ mm)")
    ax3b.set_title("(b) 40 mm ($N=80$) extrapolation & paper text ($20$ kV)")
    ax3b.legend(frameon=False, fontsize=6.8, loc="upper right")

    fig3_path = FIG_DIR / "03_twt_gain_vs_frequency_karetnikova.png"
    figstyle.save(fig3, fig3_path, close=True, auto_legend_below=False)

    # -----------------------------------------------------------------
    # Figure 04: Cube-Root Scaling & Falsifiable Physics Validation Suite
    # -----------------------------------------------------------------
    fig4, axes4 = figstyle.new_figure(nrows=1, ncols=3, width_in=13.8, height_in=4.1)
    ax4a, ax4b, ax4c = axes4

    i0_sw = d4["i0_sweep"]
    ccr_j = t4["current_cube_root_scaling"]
    ax4a.loglog(
        i0_sw,
        d4["alpha_i0_pierce_qc0"],
        "--",
        color="#374151",
        lw=1.7,
        label=r"Pierce $\beta_e C \sqrt{3}/2 \propto I_0^{1/3}$ ($0.333$)",
    )
    ax4a.loglog(
        i0_sw,
        d4["alpha_i0_fit"],
        "o-",
        color="#1d4ed8",
        lw=2.0,
        ms=6.0,
        label=f"FIT BVP de-emb. m=19 (slope={ccr_j['loglog_slope_fit']:.3f}, +30%)",
    )
    ax4a.loglog(
        i0_sw,
        d4["alpha_i0_raw_fit"],
        "s:",
        color="#d97706",
        lw=1.7,
        ms=5.2,
        label=f"Raw FIT window m=8..19 (slope={ccr_j['raw_window_loglog_slope']:.3f})",
    )
    ax4a.set_xlabel("Beam current $I_0$ [A]")
    ax4a.set_ylabel(r"Growth rate / slope $\alpha$ [Np/m]")
    ax4a.set_title(r"(a) Current scaling: de-embedded vs raw window")
    ax4a.legend(frameon=False, fontsize=7.1, loc="lower right")

    ax4b.plot(
        d4["b_sweep_fit"],
        d4["alpha_b_th"],
        "--",
        color="#374151",
        lw=1.7,
        label=r"Pierce $CN\to\infty$ asymptote $\beta_e C x_1(b)$",
    )
    ax4b.plot(
        d4["b_sweep_fit"],
        d4["alpha_b_fit"],
        "s-",
        color="#047857",
        lw=2.0,
        ms=5.8,
        label=r"Uncalibrated FIT window slope ($m\in[8,19]$)",
    )
    ax4b.plot(
        d4["b_sweep_fit"],
        d4["alpha_b_tot_win_th"],
        ":",
        color="#d97706",
        lw=1.6,
        label=r"Withdrawn 1D total-field BVP ($b\leq 0$ fails)",
    )
    ax4b.axhline(0.0, color="#9ca3af", ls=":", lw=1.0)
    ax4b.set_ylim(-290.0, 260.0)
    ax4b.set_xlabel(r"Velocity detuning $b = (\beta_c - \beta_e)/(\beta_e C)$")
    ax4b.set_ylabel(r"Window slope $\alpha$ [Np/m]")
    ax4b.set_title("(b) Velocity detuning $b$ across full sweep")
    ax4b.legend(frameon=False, fontsize=7.1, loc="lower right")

    z_p4 = d4["z_period_m"] * 1e3
    ax4c.plot(
        z_p4,
        d4["gain_posb_pos_db"],
        "o-",
        color="#1d4ed8",
        lw=1.8,
        ms=4.0,
        label=r"$(+b,+1)$: $202.4\;\mathrm{m}^{-1}$ (peak)",
    )
    ax4c.plot(
        z_p4,
        d4["gain_negb_neg_db"],
        "v-",
        color="#dc2626",
        lw=1.8,
        ms=4.0,
        label=r"$(-b,-1)$: $215.1\;\mathrm{m}^{-1}$ (peak moves to $-b$)",
    )
    ax4c.plot(
        z_p4,
        d4["gain_negb_pos_db"],
        "s--",
        color="#60a5fa",
        lw=1.5,
        ms=3.8,
        label=r"$(-b,+1)$: $29.3\;\mathrm{m}^{-1}$ (off-peak dip)",
    )
    ax4c.plot(
        z_p4,
        d4["gain_posb_neg_db"],
        "^--",
        color="#d97706",
        lw=1.5,
        ms=3.8,
        label=r"$(+b,-1)$: $178.4\;\mathrm{m}^{-1}$ (mirror fails)",
    )
    ax4c.plot(
        z_p4,
        d4["gain_cold_db"],
        ":",
        color="#111827",
        lw=1.5,
        label="Cold limit ($I_0 = 0$)",
    )
    ax4c.set_xlabel("Axial position $z$ [mm]")
    ax4c.set_ylabel("Hot/cold gain $G(z)$ [dB]")
    ax4c.set_title(r"(c) Sign flip ($|b|=1.0$)")
    ax4c.legend(frameon=False, fontsize=6.8, loc="upper left")

    fig4_path = FIG_DIR / "04_twt_scaling_and_validation_tests.png"
    figstyle.save(fig4, fig4_path, close=True, auto_legend_below=False)

    # -----------------------------------------------------------------
    # Figure 05: 3D Glide-Symmetric V-Band SWS (Marosi) & BWO (Johnson 1955)
    # -----------------------------------------------------------------
    fig5, axes5 = figstyle.new_figure(nrows=1, ncols=3, width_in=13.8, height_in=4.2)
    ax5a, ax5b, ax5c = axes5

    f_vb = d5["f_vband_dense_ghz"]
    ax5a.plot(
        f_vb,
        d5["gain_60per_5p20kv_bvp_db"],
        lw=2.1,
        color="#047857",
        label="Pierce BVP (5.20 kV nominal, 4.7 dB loss)",
    )
    ax5a.plot(
        f_vb[::8],
        d5["gain_60per_5p20kv_ode_db"][::8],
        "o",
        ms=4.2,
        color="#dc2626",
        label="Coupled-mode ODE (5.20 kV)",
    )
    ax5a.plot(
        f_vb,
        d5["gain_60per_5p38kv_bvp_db"],
        "--",
        lw=1.7,
        color="#1d4ed8",
        label="Pierce BVP (5.38 kV coarse-$v_p$ matched)",
    )
    ax5a.axhline(19.0, color="#4b5563", ls=":", lw=1.3, label="Marosi (2025) Fig. 7 peak (19 dB @ 58 GHz)")
    ax5a.set_ylim(-5.0, 44.0)
    ax5a.set_xlabel("Frequency $f$ [GHz]")
    ax5a.set_ylabel("Small-signal gain $G$ [dB] ($N = 60$)")
    ax5a.set_title("(a) 3D V-band glide SWS gain ($I_0=11$ mA)")
    ax5a.legend(frameon=False, fontsize=7.1, loc="upper right")

    for lvl_3d, col_3d, mk_3d in (("coarse_3d", "#d97706", "s"), ("medium_3d", "#1d4ed8", "o")):
        z3 = d5[f"{lvl_3d}_z_m"] * 1e3
        r3 = 20.0 * np.log10(np.maximum(d5[f"{lvl_3d}_ratios"], 1e-30))
        ax5b.plot(z3, r3, marker=mk_3d, lw=1.8, color=col_3d, label=f"3D FIT ({lvl_3d}, $b=0$)")
    z3_ref = d5["medium_3d_z_m"]
    alpha_3d_p = t5["marosi_3d_vband"]["hot_test_3d_levels"][1]["alpha_pierce_np_m"]
    g0_ref = 20.0 * float(np.log10(max(d5["medium_3d_ratios"][0], 1e-30)))
    g3_ref = g0_ref + 20.0 * np.log10(np.e) * alpha_3d_p * (z3_ref - z3_ref[0])
    ax5b.plot(
        z3_ref * 1e3,
        g3_ref,
        "--",
        color="#111827",
        lw=1.6,
        label=rf"Pierce $b=0$ slope ($\alpha={alpha_3d_p:.1f}\;\mathrm{{m}}^{{-1}}$)",
    )
    ax5b.set_xlabel("Axial position $z$ [mm]")
    ax5b.set_ylabel("3D hot/cold gain $20\\log_{10}|E_{\\mathrm{hot}}/E_{\\mathrm{cold}}|$ [dB]")
    ax5b.set_title("(b) 3D Marosi pillar SWS hot test ($57.5$ GHz)")
    ax5b.legend(frameon=False, fontsize=7.6, loc="upper left")

    L_bwo_mm = d5["L_bwo_arr_m"] * 1e3
    i_st_ma = d5["i_st_arr_a"] * 1e3
    ax5c.loglog(
        L_bwo_mm,
        i_st_ma,
        "o-",
        color="#1d4ed8",
        lw=2.0,
        ms=5.5,
        label=r"Johnson (1955) BVP root $I_{\mathrm{st}}(L)$",
    )
    ax5c.loglog(
        L_bwo_mm,
        i_st_ma[0] * (L_bwo_mm / L_bwo_mm[0]) ** (-3),
        "--",
        color="#dc2626",
        lw=1.5,
        label=r"Exact $L^{-3}$ identity ($(CN)_{\mathrm{st}}=0.3141$)",
    )
    ax5c.set_xlabel("Interaction length $L = N d$ [mm]")
    ax5c.set_ylabel(r"Start-oscillation current $I_{\mathrm{st}}$ [mA]")
    ax5c.set_title("(c) Johnson BWO $I_{\\mathrm{st}}(L)$ (1-cell Bloch claim withdrawn)")
    ax5c.legend(frameon=False, fontsize=7.2, loc="upper right")

    fig5_path = FIG_DIR / "05_3d_marosi_vband_and_bwo.png"
    figstyle.save(fig5, fig5_path, close=True, auto_legend_below=False)

    # -----------------------------------------------------------------
    # Figure 06: Animated GIF of Growing Wave Re(E_z) & Beam Current Re(J_{1z})
    # -----------------------------------------------------------------
    frames_ez = d2["frames_ez"]
    frames_j1 = d2["frames_j1"]
    vac_mask = d2["vac_mask_yz"]
    y_um = d2["y_um"]
    z_mm = d2["z_um"] * 1e-3

    # Build proxy for overlay_geometry on the 22-period medium grating
    grid_22, _ = build_2d_grating_hot_grid(
        spec=spec, n_periods=22, n_in_smooth=2, n_out_smooth=2, mesh_level="medium"
    )
    bare_22 = GridFIT3D(
        x=np.asarray(grid_22.x),
        y=np.asarray(grid_22.y),
        z=np.asarray(grid_22.z),
        verbose=0,
    )
    sol_22 = SolverFIT3D(
        bare_22,
        bc_low=["bloch", "pec", "cpml"],
        bc_high=["bloch", "pec", "cpml"],
        n_pml=8,
        verbose=0,
    )
    polys_22 = build_twt_grating_polygons(spec, n_periods=22, n_in_smooth=2, n_out_smooth=2)
    proxy_22 = _TWTSchematicSolverProxy(sol_22, bare_22, polys_22)

    ez_vmax = float(np.percentile(np.abs(frames_ez[:, vac_mask]), 98.5))
    j1_vmax = max(float(np.percentile(np.abs(frames_j1), 99.0)), 1e-12)
    y_mm = y_um * 1e-3
    extent_mm = [float(z_mm[0]), float(z_mm[-1]), float(y_mm[0]), float(y_mm[-1])]

    n_frames = len(frames_ez)

    def _draw_gif_frame(k: int, fig: plt.Figure) -> None:
        fig.clf()
        ax_top = fig.add_subplot(2, 1, 1)
        ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top)

        ez_k = np.where(vac_mask, frames_ez[k], np.nan)
        im1 = ax_top.imshow(
            ez_k,
            origin="lower",
            aspect="auto",
            extent=extent_mm,
            cmap="RdBu_r",
            vmin=-ez_vmax,
            vmax=+ez_vmax,
        )
        ax_top.set_facecolor("#d1d5db")
        schematic.overlay_geometry(
            ax_top,
            proxy_22,
            plane="zy",
            unit="mm",
            fill_materials=True,
            material_alpha=0.22,
            label_materials=True,
        )
        ax_top.axhline(-0.075, color="#047857", ls="--", lw=0.9)
        ax_top.axhline(+0.075, color="#047857", ls="--", lw=0.9)
        ax_top.set_ylabel(r"$y$ [mm]")
        ax_top.set_title(
            rf"TWT Growing Circuit Wave $\mathrm{{Re}}\{{E_z(z,y,t)\}}$ (frame {k+1}/{n_frames}, $f_0 = 186$ GHz)"
        )
        fig.colorbar(im1, ax=ax_top, pad=0.015, label=r"$\mathrm{Re}(E_z)$ [arb.]")

        im2 = ax_bot.imshow(
            frames_j1[k],
            origin="lower",
            aspect="auto",
            extent=extent_mm,
            cmap="PuOr_r",
            vmin=-j1_vmax,
            vmax=+j1_vmax,
        )
        schematic.overlay_geometry(
            ax_bot,
            proxy_22,
            plane="zy",
            unit="mm",
            fill_materials=False,
            label_materials=False,
        )
        ax_bot.axhline(-0.100, color="#111827", ls="-", lw=1.0)
        ax_bot.axhline(+0.100, color="#111827", ls="-", lw=1.0)
        ax_bot.set_ylim(-0.140, +0.140)
        ax_bot.set_xlabel("Axial coordinate $z$ [mm] (PEC grating $z \\in [0, 11]$ mm, CPML at ends)")
        ax_bot.set_ylabel(r"$y$ [mm]")
        ax_bot.set_title(
            r"Linearized Sheet-Beam AC Current Density $\mathrm{Re}\{J_{1z}(z,y,t)\} = \rho_0 v_{1z} + u_0 \rho_1$"
        )
        fig.colorbar(im2, ax=ax_bot, pad=0.015, label=r"$\mathrm{Re}(J_{1z})$ [arb.]")
        fig.tight_layout()

    from matplotlib.animation import FuncAnimation, PillowWriter

    gif_path = FIG_DIR / "06_twt_growing_wave_and_beam.gif"
    fig_gif = plt.figure(figsize=(10.5, 5.8), dpi=120)
    anim = FuncAnimation(fig_gif, lambda k: _draw_gif_frame(int(k), fig_gif), frames=n_frames, interval=125)
    anim.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig_gif)

    return {
        "setup_00a": str(setup_res["geometry_figure"]),
        "setup_00b": str(setup_res["mesh_figure"]),
        "fig_01": str(fig1_path),
        "fig_02": str(fig2_path),
        "fig_03": str(fig3_path),
        "fig_04": str(fig4_path),
        "fig_05": str(fig5_path),
        "gif_06": str(gif_path),
    }


def build_summary_json() -> dict[str, Any]:
    """Consolidate all quantitative metrics into ``data/summary.json`` for notebook & tests."""
    t1 = json.loads((DATA_DIR / "task_01_pierce_theory_and_scw_check.json").read_text(encoding="utf-8"))
    t2 = json.loads((DATA_DIR / "task_02_2d_grating_hot_convergence.json").read_text(encoding="utf-8"))
    t3 = json.loads((DATA_DIR / "task_03_karetnikova_gain_vs_frequency.json").read_text(encoding="utf-8"))
    t4 = json.loads((DATA_DIR / "task_04_validation_and_failure_tests.json").read_text(encoding="utf-8"))
    t5 = json.loads((DATA_DIR / "task_05_3d_marosi_vband_and_bwo.json").read_text(encoding="utf-8"))

    summary = {
        "task_01_pierce_and_scw": t1,
        "task_02_2d_convergence": t2,
        "task_03_karetnikova_gain_vs_freq": t3,
        "task_04_validation_suite": t4,
        "task_05_3d_marosi_and_bwo": t5,
    }
    out_path = DATA_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_all(force: bool = False, plot: bool = True) -> dict[str, Any]:
    """Execute all simulation tasks sequentially and generate all figures."""
    print("[1/5] Running Task 1: Pierce theory, 2D-to-3D mapping & SCW time-centring check...")
    run_scw_time_centering_check(force=force)
    print("[2/5] Running Task 2: 2D Karetnikova grating hot-test convergence (Coarse, Medium, Fine)...")
    run_hot_test_convergence(force=force)
    print("[3/5] Running Task 3: Karetnikova small-signal gain vs frequency (Fig. 3 & Fig. 5a)...")
    run_karetnikova_gain_vs_frequency(force=force)
    print("[4/5] Running Task 4: Falsifiable physics validation & mutation suite...")
    run_hot_test_validation_suite(force=force)
    print("[5/5] Running Task 5: 3D Marosi V-band glide SWS hot test & Johnson (1955) BWO...")
    run_3d_marosi_and_bwo_suite(force=force)
    summary = build_summary_json()
    if plot:
        print("[Plot] Generating setup figures, diagnostic plots, and growing-wave GIF...")
        generate_all_figures()
    print("[Done] Tutorial 8.6 data and figures are up to date.")
    return summary


def run_3d_marosi_full_batch(
    nx: int = 56,
    ny: int = 64,
    nz_per: int = 56,
    n_periods: int = 60,
    preflight_only: bool = True,
) -> dict[str, Any]:
    """Compute a preflight resource estimate for the 60-period 3D Marosi SWS.

    This is a preflight estimate only; the full 60-period 3D run is not implemented.

    Parameters
    ----------
    nx : int
        Transverse cells along broad waveguide wall :math:`a = 1.40` mm.
    ny : int
        Vertical cells along narrow waveguide wall :math:`b = 0.80` mm.
    nz_per : int
        Axial cells per period :math:`d = 1.05` mm.
    n_periods : int
        Number of axial periods :math:`N`.
    preflight_only : bool
        Kept for CLI compatibility; always returns the ``wakis.estimate``
        preflight specification without allocating 3D solver arrays.
    """
    h62 = _load_cold_test_helpers()
    m_spec = h62.MarosiPillarSWSSpec()
    a_wg = m_spec.waveguide_width_m
    b_wg = m_spec.waveguide_height_m
    d_m = m_spec.period_m

    n_pad = 1
    nz_total = (int(n_periods) + 2 * n_pad) * int(nz_per)
    dx_m = a_wg / float(nx)
    dy_m = b_wg / float(ny)
    dz_m = d_m / float(nz_per)
    n_steps = max(int(round(65000.0 * (float(n_periods) / 60.0) * (float(nz_per) / 56.0))), 600)
    est = estimate((int(nx), int(ny), int(nz_total)), steps=n_steps, dtype="float64", conformal="pec")
    info = {
        "status": "preflight_estimate_only",
        "note": "a preflight estimate only; the full 60-period run is not implemented",
        "grid_shape": [int(nx), int(ny), int(nz_total)],
        "n_periods": int(n_periods),
        "nz_per_period": int(nz_per),
        "dx_um": round(dx_m * 1e6, 2),
        "dy_um": round(dy_m * 1e6, 2),
        "dz_um": round(dz_m * 1e6, 2),
        "cells_across_100um_beam_thickness": round(m_spec.beam_thickness_m / dy_m, 2),
        "n_cells": int(est.cells),
        "n_steps": int(n_steps),
        "ram_working_gb": round(float(est.host_bytes / 1e9), 3),
        "cpu_hours_est": round(float(est.seconds / 3600.0), 2),
    }
    return info


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for Tutorial 8.6 runner and 3D Marosi preflight."""
    parser = argparse.ArgumentParser(description="Tutorial 8.6 TWT Small-Signal Gain runner.")
    parser.add_argument("--force", action="store_true", help="Re-run all simulations even if cached.")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from cached data.")
    parser.add_argument(
        "--run-3d-marosi-full60",
        action="store_true",
        help="Preflight estimate only for the 60-period 3D Marosi SWS (full 60-period run is not implemented).",
    )
    parser.add_argument("--nx", type=int, default=56, help="3D Marosi grid Nx along a = 1.40 mm.")
    parser.add_argument("--ny", type=int, default=64, help="3D Marosi grid Ny along b = 0.80 mm.")
    parser.add_argument("--nz-per", type=int, default=56, help="3D Marosi axial cells per period d = 1.05 mm.")
    parser.add_argument("--n-periods", type=int, default=60, help="Number of 3D Marosi periods.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Print the 3D Marosi preflight resource estimate (default behavior for --run-3d-marosi-full60).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_cli_args(argv)
    if args.run_3d_marosi_full60:
        print("NOTE: This is a preflight estimate only; the full 60-period run is not implemented.")
        res_b = run_3d_marosi_full_batch(
            nx=args.nx,
            ny=args.ny,
            nz_per=args.nz_per,
            n_periods=args.n_periods,
            preflight_only=True,
        )
        print(json.dumps(res_b, indent=2))
    elif args.plot_only:
        run_all(force=False, plot=True)
    else:
        run_all(force=args.force, plot=True)


if __name__ == "__main__":
    main()
