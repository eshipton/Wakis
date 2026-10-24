"""Analytic and numerical helpers for Tutorial 6.2: Slow-Wave Structure Cold Test.

Provides:
- Geometry specifications and ``GridFIT3D`` builders for:
  1. 2D/3D double-corrugated rectangular-waveguide grating SWS
     (Karetnikova et al., IEEE TED 65(6), 2129-2134, 2018; McVey et al.,
     IEEE TMTT 42(6), 995-1003, 1994).
  2. 3D V-band staggered-pillar sheet-beam SWS in glide-symmetric and
     mirror-symmetric configurations (Marosi, Zuboraj & Capolino,
     arXiv:2505.22927, 2025).
- Multimodal and single-mode McVey field-matching dispersion solvers (2D TM
  and 3D LSE_10 with waveguide cutoff mapping k_perp^2 = k_0^2 - (pi/a)^2),
  convergence audit in space harmonics N_harm and slot modes M_slot, and
  analytic space-harmonic Pierce impedance calculation.
- Relativistic electron-beam kinematics (beam velocity u_0(V_0), Lorentz
  factor gamma_0, synchronous phase-velocity point).
- Wakis two-stage Bloch eigenmode solver:
  Stage 1: broadband off-axis Gaussian kick + ``ResonanceMonitor`` (harminv)
           to locate eigenfrequencies f_m(beta_0) and quality factors Q_m.
  Stage 2: narrowband drive at f_m followed by post-pulse free ring-down
           with ``DFTMonitor`` to extract uncontaminated complex Bloch mode
           fields (E_m, H_m), axial Poynting power flow P_z, cell stored
           electromagnetic energy U_cell, energy velocity v_E = P_z d / U_cell,
           Floquet space-harmonic coefficients E_{zn}(x, y), and on-axis or
           beam-averaged Pierce interaction impedance K_n.
- Mesh resolution audit and preflight resource estimator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv
from scipy.constants import c as C0
from scipy.constants import e as QE
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import m_e as ME
from scipy.constants import mu_0 as MU0
from scipy.optimize import brentq

from wakis import GridFIT3D, SolverFIT3D
from wakis.estimate import estimate
from wakis.field_monitors import DFTMonitor
from wakis.resonances import ResonanceMonitor
from wakis.sources import CurrentSource, GaussianSource


# =====================================================================
# 1. Geometry & Beam Specifications
# =====================================================================


@dataclass(frozen=True)
class GratingSWSSpec:
    """Specification for the double-corrugated rectangular waveguide grating SWS.

    Dimensions match Table I ("Optimized dimensions of the double-grating SWS")
    of T. A. Karetnikova et al., IEEE Trans. Electron Devices 65(6), 2129-2134
    (2018), DOI: 10.1109/TED.2017.2787960: period p = 500 um, vane thickness
    s = 100 um, vane height l = 300 um, beam tunnel height 2a = 200 um,
    structure width b = 850 um, sheet-beam thickness H_b = 150 um, sheet-beam
    width W_b = 750 um, staggering w = p/2, and synchronous beam voltage
    V_0 = 20 kV. Analyzed with the LSE mode-matching formulation of McVey et al.,
    IEEE Trans. Microw. Theory Tech. 42(6), 995-1003 (1994),
    DOI: 10.1109/22.293568, extended to double-corrugated mirror/glide gratings.

    Attributes
    ----------
    period_m : float
        Axial grating period d (Table I: p) [m] (default: 0.50e-3 m).
    vane_length_m : float
        Axial vane thickness l (Table I: s) [m] (default: 0.10e-3 m).
    slot_width_m : float
        Axial groove / slot width s = d - l [m] (default: 0.40e-3 m).
    vane_height_m : float
        Groove depth / vane height h (Table I: l) [m] (default: 0.30e-3 m).
    tunnel_half_height_m : float
        Half-height of the beam tunnel g_half (Table I: a) [m], so full clear
        vertical gap between top and bottom vane tips is 2a = 2 * g_half = 0.20e-3 m.
    waveguide_width_m : float
        Transverse waveguide width a (Table I: b) [m] for the 3D rectangular
        housing (default: 0.85e-3 m, giving f_{c,TE10} = c / (2a) = 176.35 GHz).
    beam_voltage_v : float
        DC electron beam accelerating voltage V_0 [V] (default: 20e3 V).
    beam_half_thickness_m : float
        Half-thickness of the sheet beam H_b / 2 along y [m] (default: 0.075e-3 m,
        giving full beam thickness H_b = 0.15e-3 m = 150 um per Table I).
    beam_width_m : float
        Full transverse width of the sheet beam W_b along x [m] (default: 0.75e-3 m
        = 750 um per Table I).
    """

    period_m: float = 0.50e-3
    vane_length_m: float = 0.10e-3
    slot_width_m: float = 0.40e-3
    vane_height_m: float = 0.30e-3
    tunnel_half_height_m: float = 0.10e-3
    waveguide_width_m: float = 0.85e-3
    beam_voltage_v: float = 20.0e3
    beam_half_thickness_m: float = 0.075e-3
    beam_width_m: float = 0.75e-3

    @property
    def total_height_m(self) -> float:
        """Total vertical waveguide height 2 * (g_half + h) [m] (0.80e-3 m)."""
        return 2.0 * (self.tunnel_half_height_m + self.vane_height_m)

    @property
    def min_feature_m(self) -> float:
        """Smallest geometric feature size [m] (0.10e-3 m vane thickness)."""
        return min(
            self.vane_length_m,
            2.0 * self.tunnel_half_height_m,
            self.slot_width_m,
            self.vane_height_m,
        )

    @property
    def cutoff_freq_3d_hz(self) -> float:
        """Transverse TE10/LSE10 cutoff frequency c / (2 a) [Hz]."""
        return float(C0 / (2.0 * self.waveguide_width_m))


@dataclass(frozen=True)
class MarosiPillarSWSSpec:
    """Specification for the 3D V-band staggered-pillar sheet-beam SWS.

    Parameters verified against Table I and Section II of R. Marosi,
    M. Zuboraj, and F. Capolino, "Wideband Glide-Symmetric Slow-Wave Structure
    for Millimeter-Wave Sheet Beam TWTs", arXiv:2505.22927v1 (2025),
    DOI: 10.48550/arXiv.2505.22927.

    Attributes
    ----------
    waveguide_width_m : float
        Broad wall width a along x [m] (1.40e-3 m).
    waveguide_height_m : float
        Narrow wall height b along y [m] (0.80e-3 m).
    period_m : float
        Unit-cell axial period d along z [m] (1.05e-3 m).
    pillar_width_m : float
        Transverse pillar span w along x [m] (1.08e-3 m).
    pillar_length_m : float
        Axial pillar thickness l along z [m] (0.27e-3 m).
    pillar_thickness_m : float
        Vertical pillar thickness t along y [m] (0.10e-3 m).
    beam_width_m : float
        Sheet beam width w_b along x [m] (0.55e-3 m).
    beam_thickness_m : float
        Sheet beam thickness t_b along y [m] (0.10e-3 m).
    pillar_beam_clearance_m : float
        Vertical clearance g between sheet beam surface and pillar inner edge [m]
        (0.10e-3 m), so pillar inner surfaces lie at |y| = t_b/2 + g = 0.15e-3 m.
    beam_voltage_v : float
        DC electron beam voltage V_0 [V] (5.2e3 V).
    beam_current_a : float
        DC electron beam current I_0 [A] (11.0e-3 A).
    """

    waveguide_width_m: float = 1.40e-3
    waveguide_height_m: float = 0.80e-3
    period_m: float = 1.05e-3
    pillar_width_m: float = 1.08e-3
    pillar_length_m: float = 0.27e-3
    pillar_thickness_m: float = 0.10e-3
    beam_width_m: float = 0.55e-3
    beam_thickness_m: float = 0.10e-3
    pillar_beam_clearance_m: float = 0.10e-3
    beam_voltage_v: float = 5.2e3
    beam_current_a: float = 11.0e-3

    @property
    def pillar_y_inner_m(self) -> float:
        """Inner vertical coordinate |y| of each pillar pair [m] (0.15e-3 m)."""
        return 0.5 * self.beam_thickness_m + self.pillar_beam_clearance_m

    @property
    def pillar_y_outer_m(self) -> float:
        """Outer vertical coordinate |y| of each pillar pair [m] (0.25e-3 m)."""
        return self.pillar_y_inner_m + self.pillar_thickness_m

    @property
    def min_feature_m(self) -> float:
        """Smallest geometric feature size [m] (pillar thickness t and clearance g)."""
        return min(
            self.pillar_thickness_m,
            self.pillar_beam_clearance_m,
            self.beam_thickness_m,
            self.pillar_length_m,
            self.waveguide_width_m - self.pillar_width_m,
        )


def relativistic_beam_velocity(voltage_v: float) -> tuple[float, float]:
    """Compute relativistic electron velocity u_0 and Lorentz factor gamma_0.

    Parameters
    ----------
    voltage_v : float
        Accelerating potential V_0 [V].

    Returns
    -------
    u0_m_s : float
        Axial electron beam speed u_0 [m/s].
    gamma0 : float
        Relativistic Lorentz factor gamma_0 = 1 + e V_0 / (m_e c^2) [-].
    """
    if voltage_v <= 0.0:
        raise ValueError("voltage_v must be strictly positive.")
    gamma0 = 1.0 + (QE * float(voltage_v)) / (ME * C0**2)
    beta_rel = np.sqrt(1.0 - 1.0 / (gamma0**2))
    return float(beta_rel * C0), float(gamma0)


# =====================================================================
# 2. Analytic McVey Field-Matching Dispersion & Space Harmonics
# =====================================================================


def overlap_integral_slot_harmonic(
    beta_n: np.ndarray,
    slot_width_m: float,
    m_idx: int,
) -> np.ndarray:
    """Compute normalized slot-to-Floquet overlap integral I_{m,n}.

    Evaluates:
        I_{m,n} = (1 / s) int_0^s cos(m pi z / s) exp(j beta_n (z - s/2)) dz

    Parameters
    ----------
    beta_n : np.ndarray
        Floquet harmonic wavenumbers beta_n = beta_0 + 2 pi n / d [rad/m].
    slot_width_m : float
        Axial groove width s [m].
    m_idx : int
        Slot waveguide mode index m = 0, 1, 2, ...

    Returns
    -------
    overlap : np.ndarray
        Complex overlap array of shape ``beta_n.shape``.
    """
    theta = beta_n * slot_width_m
    if m_idx == 0:
        return np.sinc(theta / (2.0 * np.pi)).astype(np.complex128)
    denom = (m_idx * np.pi) ** 2 - theta**2
    out = np.zeros_like(theta, dtype=np.complex128)
    mask = np.abs(denom) > 1e-8
    th_m = theta[mask]
    out[mask] = (
        1j
        * th_m
        * (((-1) ** m_idx) * np.exp(0.5j * th_m) - np.exp(-0.5j * th_m))
        / denom[mask]
    )
    out[~mask] = 0.5 * np.exp(-0.5j * m_idx * np.pi * np.sign(theta[~mask]))
    return out


def mcvey_dispersion_determinant(
    freq_hz: float,
    beta0_rad_m: float,
    spec: GratingSWSSpec = GratingSWSSpec(),
    symmetry: str = "mirror_odd",
    n_harm: int = 40,
    m_slot: int = 5,
    f_cut_hz: float = 0.0,
) -> float:
    """Evaluate the McVey multimodal field-matching secular determinant.

    Matches tangential electric field E_z and magnetic field H_x across the
    slot mouth plane y = +-g_half between the groove waveguide eigenmodes
    m = 0, 1, ..., M_slot - 1 and the Floquet space harmonics n = -N_harm..N_harm
    in the interaction tunnel |y| <= g_half.

    Parameters
    ----------
    freq_hz : float
        Trial frequency f [Hz].
    beta0_rad_m : float
        Fundamental Floquet wavenumber beta_0 [rad/m].
    spec : GratingSWSSpec
        Grating geometry specification.
    symmetry : {'mirror_odd', 'mirror_even', 'glide_a', 'glide_b'}
        Symmetry class of the grating and mode:
        - 'mirror_odd': mirror-symmetric lower branch (coth channel admittance).
        - 'mirror_even': mirror-symmetric upper branch (tanh channel admittance).
        - 'glide_b': glide-symmetric lower branch (tanh for odd n, coth for even n).
        - 'glide_a': glide-symmetric upper branch (tanh for even n, coth for odd n).
    n_harm : int
        Half-width of the Floquet harmonic sum n in [-n_harm, +n_harm].
    m_slot : int
        Number of slot waveguide modes included in the mouth expansion.
    f_cut_hz : float
        Transverse waveguide cutoff frequency f_c [Hz] (0 for 2D planar;
        c / (2a) for 3D LSE_10 rectangular waveguide).

    Returns
    -------
    det_val : float
        Real secular determinant det(A(f, beta_0)) [-].
    """
    k0 = 2.0 * np.pi * freq_hz / C0
    kc = 2.0 * np.pi * f_cut_hz / C0
    if k0 <= kc:
        return -1.0e9
    k_perp = np.sqrt(k0**2 - kc**2)

    d = spec.period_m
    s = spec.slot_width_m
    h = spec.vane_height_m
    g_half = spec.tunnel_half_height_m

    ns = np.arange(-n_harm, n_harm + 1, dtype=int)
    beta_n = beta0_rad_m + 2.0 * np.pi * ns / d
    tau2 = beta_n**2 - k_perp**2
    y_n = np.zeros_like(beta_n, dtype=float)

    for idx, (n_val, t2) in enumerate(zip(ns, tau2)):
        if symmetry == "mirror_even":
            use_tanh = True
        elif symmetry == "mirror_odd":
            use_tanh = False
        elif symmetry in ("glide_a", "glide_plus"):
            use_tanh = (n_val % 2) == 0
        elif symmetry in ("glide_b", "glide_minus"):
            use_tanh = (n_val % 2) != 0
        else:
            raise ValueError(f"Unsupported symmetry={symmetry!r}")

        if t2 >= 0.0:
            tau = np.sqrt(max(t2, 1e-24))
            ug = tau * g_half
            y_n[idx] = (k_perp / tau) * (
                np.tanh(ug) if use_tanh else 1.0 / np.tanh(max(ug, 1e-12))
            )
        else:
            kap = np.sqrt(-t2)
            ug = kap * g_half
            y_n[idx] = (k_perp / kap) * (
                np.tan(ug) if use_tanh else -1.0 / np.tan(max(ug, 1e-12))
            )

    i_mn = np.array(
        [overlap_integral_slot_harmonic(beta_n, s, m) for m in range(m_slot)],
        dtype=np.complex128,
    )
    a_mat = (s / d) * ((i_mn * y_n[None, :]) @ i_mn.conj().T)

    for m in range(m_slot):
        eps_m = 1.0 if m == 0 else 0.5
        gm2 = (m * np.pi / s) ** 2 - k_perp**2
        if gm2 >= 0.0:
            gm = np.sqrt(max(gm2, 1e-24))
            y_slot_m = -(k_perp / gm) / np.tanh(gm * h)
        else:
            km = np.sqrt(-gm2)
            y_slot_m = (k_perp / km) / np.tan(km * h)
        a_mat[m, m] -= eps_m * y_slot_m

    return float(np.linalg.det(a_mat).real)


def solve_mcvey_band_frequencies(
    beta0_rad_m: float,
    spec: GratingSWSSpec = GratingSWSSpec(),
    symmetry: str = "mirror_odd",
    n_harm: int = 40,
    m_slot: int = 5,
    f_cut_hz: float = 0.0,
    f_min_ghz: float = 5.0,
    f_max_ghz: float = 235.0,
    n_scan: int = 500,
) -> list[float]:
    """Find eigenfrequencies f(beta_0) [Hz] of the McVey dispersion relation.

    Parameters
    ----------
    beta0_rad_m : float
        Floquet wavenumber beta_0 [rad/m].
    spec : GratingSWSSpec
        Grating geometry specification.
    symmetry : {'mirror_odd', 'mirror_even', 'mirror', 'glide_b', 'glide_a', 'glide'}
        Symmetry configuration. Passing ``'mirror'`` solves both ``'mirror_odd'``
        (lower branch) and ``'mirror_even'`` (upper branch); passing ``'glide'``
        solves both ``'glide_b'`` (lower branch) and ``'glide_a'`` (upper branch).
    n_harm : int
        Number of space harmonics n in [-n_harm, +n_harm].
    m_slot : int
        Number of slot waveguide modes.
    f_cut_hz : float
        Transverse waveguide cutoff frequency [Hz] (0 for 2D, c/(2a) for 3D).
    f_min_ghz, f_max_ghz : float
        Frequency search window [GHz].
    n_scan : int
        Number of coarse scan points used to bracket roots while rejecting poles.

    Returns
    -------
    roots_hz : list[float]
        Sorted list of eigenfrequencies [Hz] inside the search interval.
    """
    f_lo_hz = max(f_min_ghz * 1e9, f_cut_hz + 0.5e9)
    f_hi_hz = f_max_ghz * 1e9
    if f_lo_hz >= f_hi_hz:
        return []

    if symmetry == "glide":
        sym_list = ["glide_b", "glide_a"]
    elif symmetry == "mirror":
        sym_list = ["mirror_odd", "mirror_even"]
    else:
        sym_list = [symmetry]

    f_grid = np.linspace(f_lo_hz, f_hi_hz, n_scan)
    f_light_hz = float(np.hypot(f_cut_hz, C0 * abs(beta0_rad_m) / (2.0 * np.pi)))
    roots: list[float] = []

    for sym_key in sym_list:
        vals = np.array(
            [
                mcvey_dispersion_determinant(
                    float(f),
                    beta0_rad_m,
                    spec=spec,
                    symmetry=sym_key,
                    n_harm=n_harm,
                    m_slot=m_slot,
                    f_cut_hz=f_cut_hz,
                )
                for f in f_grid
            ]
        )
        for i in range(len(f_grid) - 1):
            v0, v1 = vals[i], vals[i + 1]
            if not (np.isfinite(v0) and np.isfinite(v1)):
                continue
            if np.sign(v0) != np.sign(v1) and abs(v0 - v1) < 1.0e5:
                try:
                    r_hz = brentq(
                        mcvey_dispersion_determinant,
                        float(f_grid[i]),
                        float(f_grid[i + 1]),
                        args=(beta0_rad_m, spec, sym_key, n_harm, m_slot, f_cut_hz),
                        xtol=1e4,
                    )
                    # Reject poles (where determinant jumps across +-infinity or n=0 light line)
                    if (
                        sym_key in ("mirror_odd", "glide_b")
                        and abs(r_hz - f_light_hz) < 0.15e9
                    ):
                        continue
                    res = abs(
                        mcvey_dispersion_determinant(
                            r_hz,
                            beta0_rad_m,
                            spec=spec,
                            symmetry=sym_key,
                            n_harm=n_harm,
                            m_slot=m_slot,
                            f_cut_hz=f_cut_hz,
                        )
                    )
                    if res < 1.0:
                        roots.append(float(r_hz))
                except ValueError:
                    pass

    roots.sort()
    dedup: list[float] = []
    for r in roots:
        if not dedup or abs(r - dedup[-1]) > 5.0e6:
            dedup.append(r)
    return dedup


def compute_mcvey_harmonic_convergence(
    beta0_rad_m: float,
    spec: GratingSWSSpec = GratingSWSSpec(),
    symmetry: str = "glide",
    n_harm_values: tuple[int, ...] = (1, 2, 3, 5, 8, 12, 20, 35, 50),
    m_slot_values: tuple[int, ...] = (1, 2, 4, 6),
) -> dict[str, Any]:
    """Evaluate convergence of the McVey dispersion in N_harm and M_slot.

    Parameters
    ----------
    beta0_rad_m : float
        Floquet wavenumber beta_0 [rad/m].
    spec : GratingSWSSpec
        Grating geometry specification.
    symmetry : str
        Symmetry class ('glide' or 'mirror').
    n_harm_values : tuple[int, ...]
        Sequence of maximum harmonic indices N_harm to test.
    m_slot_values : tuple[int, ...]
        Sequence of slot-mode truncation orders M_slot to test.

    Returns
    -------
    conv : dict[str, Any]
        Dictionary with frequencies vs N_harm for each M_slot and relative errors
        against the highest (N_harm, M_slot) reference.
    """
    f_fundamental_ghz = np.zeros((len(m_slot_values), len(n_harm_values)), dtype=float)
    f_upper_ghz = np.zeros((len(m_slot_values), len(n_harm_values)), dtype=float)
    km1_axis_ohm = np.zeros((len(m_slot_values), len(n_harm_values)), dtype=float)
    km1_beam_avg_ohm = np.zeros((len(m_slot_values), len(n_harm_values)), dtype=float)
    imp_sym = "glide_b" if symmetry in ("glide", "glide_b") else "mirror_odd"

    for im, m_slot in enumerate(m_slot_values):
        for inh, n_harm in enumerate(n_harm_values):
            roots = solve_mcvey_band_frequencies(
                beta0_rad_m,
                spec=spec,
                symmetry=symmetry,
                n_harm=int(n_harm),
                m_slot=int(m_slot),
                f_min_ghz=10.0,
                f_max_ghz=235.0,
                n_scan=450,
            )
            f_fundamental_ghz[im, inh] = roots[0] * 1e-9 if len(roots) >= 1 else np.nan
            f_upper_ghz[im, inh] = roots[1] * 1e-9 if len(roots) >= 2 else np.nan
            if len(roots) >= 1:
                imp = compute_mcvey_analytic_pierce_impedance(
                    float(roots[0]),
                    beta0_rad_m,
                    spec=spec,
                    symmetry=imp_sym,
                    n_harm=int(n_harm),
                    m_slot=int(m_slot),
                )
                idx_m1 = int(np.where(imp["n_indices"] == -1)[0][0])
                km1_axis_ohm[im, inh] = float(imp["k_n_axis_ohm"][idx_m1])
                km1_beam_avg_ohm[im, inh] = float(imp["k_n_beam_avg_ohm"][idx_m1])
            else:
                km1_axis_ohm[im, inh] = np.nan
                km1_beam_avg_ohm[im, inh] = np.nan

    f_ref_fund = float(f_fundamental_ghz[-1, -1])
    rel_err_vs_nharm = np.abs(f_fundamental_ghz - f_ref_fund) / max(
        abs(f_ref_fund), 1e-12
    )
    return {
        "n_harm_values": np.array(n_harm_values, dtype=int),
        "m_slot_values": np.array(m_slot_values, dtype=int),
        "f_fundamental_ghz": f_fundamental_ghz,
        "f_upper_ghz": f_upper_ghz,
        "f_ref_fundamental_ghz": f_ref_fund,
        "rel_err_vs_nharm": rel_err_vs_nharm,
        "km1_axis_ohm": km1_axis_ohm,
        "km1_beam_avg_ohm": km1_beam_avg_ohm,
    }


def compute_mcvey_analytic_pierce_impedance(
    freq_hz: float,
    beta0_rad_m: float,
    spec: GratingSWSSpec = GratingSWSSpec(),
    symmetry: str = "glide_b",
    n_harm: int = 40,
    f_cut_hz: float = 0.0,
    m_slot: int = 5,
    include_slot_power: bool = True,
) -> dict[str, Any]:
    """Compute first-principles analytic space-harmonic amplitudes and Pierce impedance K_n.

    For a lossless Bloch mode, the time-averaged axial power flow through every
    transverse plane is identical and equals its cell average over one period d:

    .. math::
        P = \\frac{1}{d}\\int_0^d P_z(z)\\,\\mathrm{d}z = P_{\\mathrm{tunnel}} + P_{\\mathrm{slot}},

    where :math:`P_{\\mathrm{tunnel}} = \\sum_n P_{\\mathrm{harm}, n}` is the sum of
    the orthogonal Floquet space-harmonic powers in the interaction tunnel
    :math:`|y| \\le g`, and :math:`P_{\\mathrm{slot}}` is the cell-averaged axial
    Poynting flux :math:`S_z = \\frac{1}{2}\\operatorname{Re}(-E_y H_x^*)` integrated
    over both the top (:math:`y \\in [g, g + h]`) and bottom
    (:math:`y \\in [-g - h, -g]`) rectangular slots of width :math:`s` from the
    same groove waveguide mode expansion :math:`m = 0, \\dots, M_{\\mathrm{slot}} - 1`
    used in ``mcvey_dispersion_determinant``. All transverse and axial integrals for
    :math:`P_{\\mathrm{tunnel}}`, :math:`P_{\\mathrm{slot}}`, and the period-stored
    electromagnetic energy :math:`W = W_{\\mathrm{tunnel}} + W_{\\mathrm{slot}}` are
    evaluated in closed form (matching Gauss--Legendre quadrature to machine
    precision :math:`< 10^{-15}`). Standing slot modes :math:`m` carry zero net axial
    power individually; :math:`P_{\\mathrm{slot}}` arises solely from cross terms
    between even and odd slot modes :math:`(m, m')` with :math:`m + m'` odd, which
    are :math:`90^\\circ` out of phase in time for a traveling Bloch wave.

    Parameters
    ----------
    freq_hz : float
        Eigenfrequency f [Hz].
    beta0_rad_m : float
        Fundamental Floquet wavenumber beta_0 [rad/m].
    spec : GratingSWSSpec
        Grating specification.
    symmetry : {'mirror_odd', 'mirror_even', 'glide_b', 'glide_a'}
        Mode symmetry class. Note: in ``'glide_b'`` (lower glide branch),
        odd space harmonics (n = +-1, +-3, ...) have cosh(tau_n y) profile
        with non-zero on-axis E_z(0), while even space harmonics have sinh(tau_n y).
    n_harm : int
        Half-width of harmonic sum n in [-n_harm, +n_harm].
    f_cut_hz : float
        Transverse cutoff frequency [Hz] (0 for 2D, c/(2a) for 3D LSE_10).
    m_slot : int
        Number of slot waveguide modes included in the mouth and groove expansion.
    include_slot_power : bool
        If True (default), normalizes :math:`K_n` by the full cell-averaged power
        :math:`P_{\\mathrm{total}} = P_{\\mathrm{tunnel}} + P_{\\mathrm{slot}}`.
        If False, sets :math:`P_{\\mathrm{slot}} = 0` (used for mutation testing).

    Returns
    -------
    result : dict[str, Any]
        Dictionary containing ``n_indices``, ``beta_n`` [rad/m], ``ez_n_axis_norm`` [-],
        ``ez_n_mouth_norm`` [-], ``k_n_axis_ohm`` [Ohm], ``k_n_quarter_ohm`` [Ohm],
        ``k_n_beam_avg_ohm`` [Ohm], ``p_tunnel_w`` [W], ``p_slot_w`` [W],
        ``p_slot_true_w`` [W], ``p_total_w`` [W], ``p_total_norm`` [W],
        ``w_tunnel_e_j`` [J], ``w_tunnel_h_j`` [J], ``w_tunnel_j`` [J],
        ``w_slot_e_j`` [J], ``w_slot_h_j`` [J], ``w_slot_j`` [J], and ``w_total_j`` [J].
    """
    d = spec.period_m
    s = spec.slot_width_m
    h_slot = spec.vane_height_m
    g = spec.tunnel_half_height_m
    k0 = 2.0 * np.pi * freq_hz / C0
    kc = 2.0 * np.pi * f_cut_hz / C0
    k_perp = np.sqrt(max(k0**2 - kc**2, 1e-18))
    omega = 2.0 * np.pi * freq_hz

    n_arr = np.arange(-n_harm, n_harm + 1, dtype=int)
    beta_n = beta0_rad_m + 2.0 * np.pi * n_arr / d
    tau2 = beta_n**2 - k_perp**2

    # Which harmonics have cosh(tau_n y) (symmetric E_z -> non-zero on-axis E_z(0))?
    if symmetry == "mirror_even":
        cosh_mask = np.ones_like(n_arr, dtype=bool)
    elif symmetry == "mirror_odd":
        cosh_mask = np.zeros_like(n_arr, dtype=bool)
    elif symmetry in ("glide_a", "glide_plus"):
        cosh_mask = (n_arr % 2) == 0
    elif symmetry in ("glide_b", "glide_minus"):
        cosh_mask = (n_arr % 2) != 0
    else:
        cosh_mask = np.ones_like(n_arr, dtype=bool)

    y_n = np.zeros_like(beta_n, dtype=float)
    int_dy2_over_tau4 = np.zeros_like(beta_n, dtype=float)
    int_ez2 = np.zeros_like(beta_n, dtype=float)
    axis_scale = np.zeros_like(beta_n, dtype=float)
    quarter_scale = np.zeros_like(beta_n, dtype=float)
    beam_y_scale = np.zeros_like(beta_n, dtype=float)
    hb = spec.beam_half_thickness_m

    for idx, (t2, cm) in enumerate(zip(tau2, cosh_mask)):
        if t2 >= 0.0:
            tn = np.sqrt(max(t2, 1e-24))
            ug = tn * g
            uhb = max(tn * hb, 1e-12)
            if cm:
                ch_g = np.cosh(ug)
                y_n[idx] = (k_perp / tn) * np.tanh(ug)
                int_sh2 = (np.sinh(2.0 * ug) / (2.0 * tn) - g) / (ch_g**2)
                int_ch2 = (np.sinh(2.0 * ug) / (2.0 * tn) + g) / (ch_g**2)
                axis_scale[idx] = 1.0 / ch_g
                quarter_scale[idx] = np.cosh(0.5 * ug) / ch_g
                beam_y_scale[idx] = (
                    0.5 * (np.sinh(2.0 * uhb) / (2.0 * uhb) + 1.0) / (ch_g**2)
                )
            else:
                sh_g = np.sinh(max(ug, 1e-12))
                y_n[idx] = (k_perp / tn) / np.tanh(max(ug, 1e-12))
                int_sh2 = (np.sinh(2.0 * ug) / (2.0 * tn) + g) / (sh_g**2)
                int_ch2 = (np.sinh(2.0 * ug) / (2.0 * tn) - g) / (sh_g**2)
                axis_scale[idx] = 0.0
                quarter_scale[idx] = np.sinh(0.5 * ug) / sh_g
                beam_y_scale[idx] = (
                    0.5 * (np.sinh(2.0 * uhb) / (2.0 * uhb) - 1.0) / (sh_g**2)
                )
            int_dy2_over_tau4[idx] = int_sh2 / (tn**2)
            int_ez2[idx] = int_ch2
        else:
            kn = np.sqrt(-t2)
            ug = kn * g
            uhb = max(kn * hb, 1e-12)
            if cm:
                cs_g = np.cos(ug)
                y_n[idx] = (k_perp / kn) * np.tan(ug)
                int_sin2 = (g - np.sin(2.0 * ug) / (2.0 * kn)) / (cs_g**2)
                int_cos2 = (g + np.sin(2.0 * ug) / (2.0 * kn)) / (cs_g**2)
                axis_scale[idx] = 1.0 / abs(cs_g)
                quarter_scale[idx] = abs(np.cos(0.5 * ug) / cs_g)
                beam_y_scale[idx] = (
                    0.5 * (1.0 + np.sin(2.0 * uhb) / (2.0 * uhb)) / (cs_g**2)
                )
                int_dy2_over_tau4[idx] = int_sin2 / (kn**2)
                int_ez2[idx] = int_cos2
            else:
                sn_g = np.sin(max(ug, 1e-12))
                y_n[idx] = -(k_perp / kn) / np.tan(max(ug, 1e-12))
                int_cos2 = (g + np.sin(2.0 * ug) / (2.0 * kn)) / (sn_g**2)
                int_sin2 = (g - np.sin(2.0 * ug) / (2.0 * kn)) / (sn_g**2)
                axis_scale[idx] = 0.0
                quarter_scale[idx] = abs(np.sin(0.5 * ug) / sn_g)
                beam_y_scale[idx] = (
                    0.5 * (1.0 - np.sin(2.0 * uhb) / (2.0 * uhb)) / (sn_g**2)
                )
                int_dy2_over_tau4[idx] = int_cos2 / (kn**2)
                int_ez2[idx] = int_sin2

    m_slot = int(m_slot)
    i_mn = np.array(
        [overlap_integral_slot_harmonic(beta_n, s, m) for m in range(m_slot)],
        dtype=np.complex128,
    )
    a_mat = (s / d) * ((i_mn * y_n[None, :]) @ i_mn.conj().T)
    y_slot = np.zeros(m_slot, dtype=float)
    for m in range(m_slot):
        eps_m = 1.0 if m == 0 else 0.5
        gm2 = (m * np.pi / s) ** 2 - k_perp**2
        if gm2 >= 0.0:
            gm = np.sqrt(max(gm2, 1e-24))
            y_slot_m = -(k_perp / gm) / np.tanh(gm * h_slot)
        else:
            km = np.sqrt(-gm2)
            y_slot_m = (k_perp / km) / np.tan(km * h_slot)
        y_slot[m] = y_slot_m
        a_mat[m, m] -= eps_m * y_slot_m

    _, _, vh = np.linalg.svd(a_mat)
    v_slot = vh[-1].conj()
    # In the exp(j(omega t - beta_n z)) convention, the slot mouth coefficients are
    # V_hat_m = v_slot_m^*, matching E_{zn}(g) = (s / d) * sum_m V_hat_m I_{m,n}.
    v_hat = v_slot.conj()
    e_mouth_n = (s / d) * np.abs(v_slot @ i_mn.conj())

    ez_axis_n = e_mouth_n * axis_scale
    ez_quarter_n = e_mouth_n * quarter_scale

    width_eff = 1.0e-3 if f_cut_hz <= 0.0 else 0.5 * spec.waveguide_width_m
    p_harm = (
        0.5
        * omega
        * EPS0
        * (k_perp / k0) ** 2
        * beta_n
        * (e_mouth_n**2)
        * int_dy2_over_tau4
    )
    p_tunnel = float(np.sum(p_harm) * width_eff)

    # Closed-form cell-averaged axial Poynting power in both slots (top + bottom = 2x single slot).
    # Standing slot modes carry zero power alone; cross terms (m, m') with m + m' odd contribute:
    #   int_0^s (m pi / s) sin(m pi z / s) cos(m' pi z / s) dz = 2 m^2 / (m^2 - m'^2),
    #   int_0^h f_m(eta) f_{m'}(eta) d_eta = (s^2 / (pi^2 (m^2 - m'^2))) * (Y_{slot,m} - Y_{slot,m'}) / k_perp.
    p_slot_single = 0.0
    pref_p = 0.5 * omega * EPS0 * (k_perp / k0) ** 2
    for m in range(1, m_slot):
        for mp in range(m_slot):
            if (m + mp) % 2 == 1:
                dm2 = float(m**2 - mp**2)
                z_int = 2.0 * (m**2) / dm2
                y_int = (
                    (s**2 / (np.pi**2 * dm2))
                    * (y_slot[m] - y_slot[mp])
                    / k_perp
                )
                p_slot_single += (
                    pref_p
                    * float(np.imag(v_hat[m] * np.conj(v_hat[mp])))
                    * z_int
                    * y_int
                )
    p_slot_true = float(2.0 * width_eff * (p_slot_single / d))
    p_slot_used = p_slot_true if include_slot_power else 0.0
    p_total = float(p_tunnel + p_slot_used)

    # Closed-form time-averaged stored EM energy per period W = W_tunnel + W_slot [J]
    e_sq_tun_n = int_ez2 + (beta_n**2) * int_dy2_over_tau4
    we_tunnel = float(
        0.25 * EPS0 * width_eff * d * np.sum((e_mouth_n**2) * e_sq_tun_n)
    )
    wh_tunnel = float(
        0.25
        * EPS0
        * width_eff
        * d
        * np.sum(
            (e_mouth_n**2)
            * (
                (kc**2 / k0**2) * e_sq_tun_n
                + (k_perp**4 / k0**2) * int_dy2_over_tau4
            )
        )
    )
    w_tunnel = we_tunnel + wh_tunnel

    we_slot = 0.0
    wh_slot = 0.0
    for m in range(m_slot):
        eps_m = 1.0 if m == 0 else 0.5
        gm2 = (m * np.pi / s) ** 2 - k_perp**2
        if gm2 >= 0.0:
            gm = np.sqrt(max(gm2, 1e-24))
            sh_gh = np.sinh(gm * h_slot)
            j_zz = (np.sinh(2.0 * gm * h_slot) / (2.0 * gm) - h_slot) / (
                2.0 * (sh_gh**2)
            )
            j_ff = (np.sinh(2.0 * gm * h_slot) / (2.0 * gm) + h_slot) / (
                2.0 * (gm**2) * (sh_gh**2)
            )
        else:
            km = np.sqrt(-gm2)
            sn_kh = np.sin(km * h_slot)
            j_zz = (h_slot - np.sin(2.0 * km * h_slot) / (2.0 * km)) / (
                2.0 * (sn_kh**2)
            )
            j_ff = (h_slot + np.sin(2.0 * km * h_slot) / (2.0 * km)) / (
                2.0 * (km**2) * (sn_kh**2)
            )
        e_sq_slot_m = j_zz + ((m * np.pi / s) ** 2) * j_ff
        vm_sq = float(np.abs(v_hat[m]) ** 2)
        we_slot += 2.0 * 0.25 * EPS0 * width_eff * s * eps_m * vm_sq * e_sq_slot_m
        wh_slot += (
            2.0
            * 0.25
            * EPS0
            * width_eff
            * s
            * eps_m
            * vm_sq
            * ((kc**2 / k0**2) * e_sq_slot_m + (k_perp**4 / k0**2) * j_ff)
        )
    w_slot = float(we_slot + wh_slot)
    w_total = float(w_tunnel + w_slot)

    denom = 2.0 * (beta_n**2) * max(abs(p_total), 1e-30)
    k_n_axis = (ez_axis_n**2) / denom
    k_n_quarter = (ez_quarter_n**2) / denom

    # Beam cross-section averaged Pierce impedance <K_n>_beam over H_b = 2 * beam_half_thickness_m
    ez_sq_y_avg = (e_mouth_n**2) * beam_y_scale
    if f_cut_hz > 0.0:
        wb = getattr(spec, "beam_width_m", 0.75e-3)
        a_wg = spec.waveguide_width_m
        x_arg = np.pi * wb / a_wg
        x_avg_factor = 0.5 * (1.0 + np.sin(x_arg) / max(x_arg, 1e-12))
    else:
        x_avg_factor = 1.0
    k_n_beam_avg = (ez_sq_y_avg * x_avg_factor) / denom

    return {
        "n_indices": n_arr,
        "beta_n": beta_n,
        "ez_n_axis_norm": ez_axis_n / max(float(np.max(ez_axis_n)), 1e-30),
        "ez_n_mouth_norm": e_mouth_n / max(float(np.max(e_mouth_n)), 1e-30),
        "k_n_axis_ohm": k_n_axis,
        "k_n_quarter_ohm": k_n_quarter,
        "k_n_beam_avg_ohm": k_n_beam_avg,
        "p_tunnel_w": p_tunnel,
        "p_slot_w": p_slot_used,
        "p_slot_true_w": p_slot_true,
        "p_total_w": p_total,
        "p_total_norm": p_total,
        "w_tunnel_e_j": we_tunnel,
        "w_tunnel_h_j": wh_tunnel,
        "w_tunnel_j": w_tunnel,
        "w_slot_e_j": float(we_slot),
        "w_slot_h_j": float(wh_slot),
        "w_slot_j": w_slot,
        "w_total_j": w_total,
    }


def compute_mcvey_power_and_energy_check(
    beta0_rad_m: float,
    spec: GratingSWSSpec = GratingSWSSpec(),
    symmetry: str = "glide_b",
    n_harm: int = 40,
    m_slot: int = 5,
    f_cut_hz: float = 0.0,
    delta_beta0_d_over_pi: float = 1.0e-3,
    include_slot_power: bool = True,
    freq_hz: float | None = None,
) -> dict[str, Any]:
    """Compare direct Poynting power P_tunnel + P_slot against v_g * W / d.

    Evaluates the group velocity :math:`v_g = \\mathrm{d}\\omega / \\mathrm{d}\\beta_0`
    by central finite difference of ``solve_mcvey_band_frequencies`` with step
    :math:`\\delta\\beta_0 d = 10^{-3}\\pi` and verifies energy-transport consistency
    :math:`P_{\\mathrm{direct}} = P_{\\mathrm{tunnel}} + P_{\\mathrm{slot}} \\approx v_g W / d`,
    where :math:`W = W_{\\mathrm{tunnel}} + W_{\\mathrm{slot}}` is the time-averaged
    stored electromagnetic energy per period.

    Parameters
    ----------
    beta0_rad_m : float
        Fundamental Floquet wavenumber beta_0 [rad/m].
    spec : GratingSWSSpec
        Grating specification.
    symmetry : {'mirror_odd', 'mirror_even', 'glide_b', 'glide_a'}
        Mode symmetry class.
    n_harm : int
        Half-width of Floquet harmonic sum.
    m_slot : int
        Number of groove waveguide modes.
    f_cut_hz : float
        Transverse waveguide cutoff frequency [Hz] (0 for 2D, c/(2a) for 3D).
    delta_beta0_d_over_pi : float
        Central finite-difference step :math:`\\delta\\beta_0 d / \\pi` [-] (default: 1e-3).
    include_slot_power : bool
        Passed to ``compute_mcvey_analytic_pierce_impedance`` (if False, omits P_slot).
    freq_hz : float or None
        Optional pre-computed eigenfrequency f(beta_0) [Hz]. If None, solved via
        ``solve_mcvey_band_frequencies``.

    Returns
    -------
    check : dict[str, Any]
        Dictionary with all keys of ``compute_mcvey_analytic_pierce_impedance`` plus
        ``freq_hz`` [Hz], ``vg_m_s`` [m/s], ``vg_over_c`` [-], ``v_energy_m_s`` [m/s],
        ``v_energy_over_c`` [-], ``p_direct_w`` [W], ``p_vg_w`` [W], ``p_ratio`` [-],
        and ``p_rel_diff`` [-].
    """
    d = spec.period_m
    f_min_ghz = 10.0 if f_cut_hz <= 0.0 else (f_cut_hz * 1e-9 + 0.5)
    f_max_ghz = 235.0 if f_cut_hz <= 0.0 else 285.0
    if freq_hz is None:
        freq_hz = solve_mcvey_band_frequencies(
            beta0_rad_m,
            spec=spec,
            symmetry=symmetry,
            n_harm=n_harm,
            m_slot=m_slot,
            f_cut_hz=f_cut_hz,
            f_min_ghz=f_min_ghz,
            f_max_ghz=f_max_ghz,
        )[0]

    dbeta = float(delta_beta0_d_over_pi) * np.pi / d
    f_plus_hz = solve_mcvey_band_frequencies(
        beta0_rad_m + dbeta,
        spec=spec,
        symmetry=symmetry,
        n_harm=n_harm,
        m_slot=m_slot,
        f_cut_hz=f_cut_hz,
        f_min_ghz=f_min_ghz,
        f_max_ghz=f_max_ghz,
    )[0]
    f_minus_hz = solve_mcvey_band_frequencies(
        beta0_rad_m - dbeta,
        spec=spec,
        symmetry=symmetry,
        n_harm=n_harm,
        m_slot=m_slot,
        f_cut_hz=f_cut_hz,
        f_min_ghz=f_min_ghz,
        f_max_ghz=f_max_ghz,
    )[0]
    vg_m_s = float(2.0 * np.pi * (f_plus_hz - f_minus_hz) / (2.0 * dbeta))

    imp = compute_mcvey_analytic_pierce_impedance(
        float(freq_hz),
        beta0_rad_m,
        spec=spec,
        symmetry=symmetry,
        n_harm=n_harm,
        f_cut_hz=f_cut_hz,
        m_slot=m_slot,
        include_slot_power=include_slot_power,
    )
    p_direct_w = float(imp["p_total_w"])
    w_total_j = float(imp["w_total_j"])
    p_vg_w = float(vg_m_s * w_total_j / d)
    p_ratio = float(p_direct_w / p_vg_w) if abs(p_vg_w) > 0.0 else np.nan
    p_rel_diff = float(abs(p_direct_w - p_vg_w) / max(abs(p_vg_w), 1e-30))
    v_energy_m_s = float(p_direct_w * d / max(w_total_j, 1e-30))

    return {
        **imp,
        "freq_hz": float(freq_hz),
        "vg_m_s": vg_m_s,
        "vg_over_c": float(vg_m_s / C0),
        "v_energy_m_s": v_energy_m_s,
        "v_energy_over_c": float(v_energy_m_s / C0),
        "p_direct_w": p_direct_w,
        "p_vg_w": p_vg_w,
        "p_ratio": p_ratio,
        "p_rel_diff": p_rel_diff,
        "k_n_axis_vg_w_ohm": imp["k_n_axis_ohm"] * p_ratio,
        "k_n_beam_avg_vg_w_ohm": imp["k_n_beam_avg_ohm"] * p_ratio,
    }


# =====================================================================
# 3. PyVista Geometry & Wakis Grid Builders
# =====================================================================


def build_2d_grating_grid(
    spec: GratingSWSSpec = GratingSWSSpec(),
    n_feat: int = 8,
    glide_symmetric: bool = True,
    conformal: str | bool = "pec",
) -> tuple[GridFIT3D, dict[str, Any]]:
    """Build a 2D quasi-planar ``GridFIT3D`` for the double-corrugated grating SWS.

    Places bottom and top rectangular PEC vanes (`bot_vane`, `top_vane`) in a
    vacuum parallel-plate waveguide of height b = 2(g_half + h) and period d.

    Parameters
    ----------
    spec : GratingSWSSpec
        Grating geometry specification.
    n_feat : int
        Number of grid cells across the smallest feature (0.10 mm).
    glide_symmetric : bool
        If True, shifts the top vane by d/2 relative to the bottom vane
        (z_1 = 0.25 d, z_2 = 0.75 d). If False, aligns both at z_1 = z_2 = 0.25 d.
    conformal : {'pec', False}
        Wakis conformal treatment for PEC boundaries.

    Returns
    -------
    grid : GridFIT3D
        Constructed Wakis FIT grid.
    mesh_meta : dict[str, Any]
        Standardized mesh metadata dictionary (`dx_background`, `dx_min`,
        `cells_across_feature`, `n_cells`, etc.).
    """
    d = spec.period_m
    l_vane = spec.vane_length_m
    g = spec.tunnel_half_height_m
    b = spec.total_height_m

    dy = spec.min_feature_m / float(n_feat)
    dz = spec.min_feature_m / float(n_feat)
    ny = int(round(b / dy))
    nz = int(round(d / dz))
    nx = 1
    lx = 1.0e-3

    z1 = 0.25 * d
    z2 = 0.75 * d if glide_symmetric else 0.25 * d

    bot_vane = pv.Box(
        bounds=(-lx, lx, -0.5 * b - 0.1e-3, -g, z1 - 0.5 * l_vane, z1 + 0.5 * l_vane)
    )
    top_vane = pv.Box(
        bounds=(-lx, lx, g, 0.5 * b + 0.1e-3, z2 - 0.5 * l_vane, z2 + 0.5 * l_vane)
    )

    grid = GridFIT3D(
        xmin=-0.5 * lx,
        xmax=0.5 * lx,
        ymin=-0.5 * b,
        ymax=0.5 * b,
        zmin=0.0,
        zmax=d,
        Nx=nx,
        Ny=ny,
        Nz=nz,
        stl_solids={"bot": bot_vane, "top": top_vane},
        stl_materials={"bot": "pec", "top": "pec"},
        conformal=conformal,
        verbose=0,
    )
    meta = measure_sws_mesh_metrics(
        grid=grid,
        min_feature_m=spec.min_feature_m,
        active_axes=("y", "z"),
    )
    meta["glide_symmetric"] = bool(glide_symmetric)
    meta["conformal"] = str(conformal)
    return grid, meta


def build_3d_grating_grid(
    spec: GratingSWSSpec = GratingSWSSpec(),
    n_feat: int = 4,
    nx: int = 18,
    glide_symmetric: bool = True,
    conformal: str | bool = "pec",
) -> tuple[GridFIT3D, dict[str, Any]]:
    """Build a 3D rectangular-waveguide ``GridFIT3D`` for the Karetnikova grating SWS.

    Parameters
    ----------
    spec : GratingSWSSpec
        Grating geometry specification (with ``waveguide_width_m = 0.85e-3``).
    n_feat : int
        Number of grid cells across the 0.10 mm minimum feature in (y, z).
    nx : int
        Number of cells across the broad waveguide width a along x.
    glide_symmetric : bool
        If True, shifts top vanes by d/2 relative to bottom vanes.
    conformal : {'pec', False}
        Wakis conformal PEC treatment.

    Returns
    -------
    grid : GridFIT3D
        Constructed 3D Wakis FIT grid.
    mesh_meta : dict[str, Any]
        Standardized mesh metadata dictionary.
    """
    a = spec.waveguide_width_m
    b = spec.total_height_m
    d = spec.period_m
    l_vane = spec.vane_length_m
    g = spec.tunnel_half_height_m

    dy = spec.min_feature_m / float(n_feat)
    dz = spec.min_feature_m / float(n_feat)
    ny = int(round(b / dy))
    nz = int(round(d / dz))

    z1 = 0.25 * d
    z2 = 0.75 * d if glide_symmetric else 0.25 * d

    bot_vane = pv.Box(
        bounds=(
            -0.6 * a,
            0.6 * a,
            -0.5 * b - 0.1e-3,
            -g,
            z1 - 0.5 * l_vane,
            z1 + 0.5 * l_vane,
        )
    )
    top_vane = pv.Box(
        bounds=(
            -0.6 * a,
            0.6 * a,
            g,
            0.5 * b + 0.1e-3,
            z2 - 0.5 * l_vane,
            z2 + 0.5 * l_vane,
        )
    )

    grid = GridFIT3D(
        xmin=-0.5 * a,
        xmax=0.5 * a,
        ymin=-0.5 * b,
        ymax=0.5 * b,
        zmin=0.0,
        zmax=d,
        Nx=nx,
        Ny=ny,
        Nz=nz,
        stl_solids={"bot": bot_vane, "top": top_vane},
        stl_materials={"bot": "pec", "top": "pec"},
        conformal=conformal,
        verbose=0,
    )
    meta = measure_sws_mesh_metrics(
        grid=grid,
        min_feature_m=spec.min_feature_m,
        active_axes=("x", "y", "z"),
        feature_axis="y",
    )
    meta["glide_symmetric"] = bool(glide_symmetric)
    meta["conformal"] = str(conformal)
    return grid, meta


def build_3d_marosi_pillar_grid(
    spec: MarosiPillarSWSSpec = MarosiPillarSWSSpec(),
    nx: int = 28,
    ny: int = 32,
    nz: int = 28,
    glide_symmetric: bool = True,
    conformal: str | bool = "pec",
) -> tuple[GridFIT3D, dict[str, Any]]:
    """Build the 3D V-band staggered-pillar SWS grid of Marosi et al. (2025).

    In the glide-symmetric configuration (Marosi et al., Fig. 1 and Table I),
    a vertical pair of pillars at y in [+0.15, +0.25] mm and [-0.25, -0.15] mm
    protrudes from the -x sidewall (x in [-a/2, -a/2 + w]) centered at z_1 = d/4,
    and an identical vertical pair protrudes from the +x sidewall
    (x in [a/2 - w, a/2]) centered at z_2 = 3d/4 = z_1 + d/2.
    In the broken-glide control (`glide_symmetric=False`), the +x pair sits at
    z_2 = 0.62 d instead of z_1 + d/2, so the cell has no glide plane (and no
    x = 0 mirror plane). Both configurations keep the y = 0 mirror plane.

    Parameters
    ----------
    spec : MarosiPillarSWSSpec
        Marosi V-band SWS specification.
    nx, ny, nz : int
        Grid cell counts along (x, y, z). Default (28, 32, 28) gives
        dy = 25.0 um (4.0 cells across the 0.10 mm pillar thickness t and
        pillar-beam clearance g), dz = 37.5 um, dx = 50.0 um.
    glide_symmetric : bool
        If True, staggers the -x and +x pillar pairs by d/2; if False, places
        the +x pair at z_2 = 0.62 d (broken glide) to demonstrate the stopband
        opening at beta_n d = 3 pi.
    conformal : {'pec', False}
        Wakis conformal PEC mode.

    Returns
    -------
    grid : GridFIT3D
        Constructed 3D Wakis FIT grid.
    mesh_meta : dict[str, Any]
        Standardized mesh metadata dictionary.
    """
    a = spec.waveguide_width_m
    b = spec.waveguide_height_m
    d = spec.period_m
    w = spec.pillar_width_m
    l_p = spec.pillar_length_m
    y_in = spec.pillar_y_inner_m
    y_out = spec.pillar_y_outer_m

    z1 = 0.25 * d
    z2 = 0.75 * d if glide_symmetric else 0.62 * d
    p1_top = pv.Box(
        bounds=(-0.5 * a - 0.05e-3, -0.5 * a + w, y_in, y_out, z1 - 0.5 * l_p, z1 + 0.5 * l_p)
    )
    p1_bot = pv.Box(
        bounds=(-0.5 * a - 0.05e-3, -0.5 * a + w, -y_out, -y_in, z1 - 0.5 * l_p, z1 + 0.5 * l_p)
    )
    p2_top = pv.Box(
        bounds=(0.5 * a - w, 0.5 * a + 0.05e-3, y_in, y_out, z2 - 0.5 * l_p, z2 + 0.5 * l_p)
    )
    p2_bot = pv.Box(
        bounds=(0.5 * a - w, 0.5 * a + 0.05e-3, -y_out, -y_in, z2 - 0.5 * l_p, z2 + 0.5 * l_p)
    )
    stl_solids = {"p1t": p1_top, "p1b": p1_bot, "p2t": p2_top, "p2b": p2_bot}
    stl_materials = {"p1t": "pec", "p1b": "pec", "p2t": "pec", "p2b": "pec"}

    grid = GridFIT3D(
        xmin=-0.5 * a,
        xmax=0.5 * a,
        ymin=-0.5 * b,
        ymax=0.5 * b,
        zmin=0.0,
        zmax=d,
        Nx=nx,
        Ny=ny,
        Nz=nz,
        stl_solids=stl_solids,
        stl_materials=stl_materials,
        conformal=conformal,
        verbose=0,
    )
    meta = measure_sws_mesh_metrics(
        grid=grid,
        min_feature_m=spec.min_feature_m,
        active_axes=("x", "y", "z"),
        feature_axis="y",
    )
    meta["glide_symmetric"] = bool(glide_symmetric)
    meta["conformal"] = str(conformal)
    return grid, meta


def measure_sws_mesh_metrics(
    grid: GridFIT3D,
    min_feature_m: float,
    active_axes: tuple[str, ...] = ("y", "z"),
    feature_axis: str = "y",
) -> dict[str, Any]:
    """Compute required mesh resolution metrics from a constructed ``GridFIT3D``.

    Parameters
    ----------
    grid : GridFIT3D
        Constructed Wakis grid.
    min_feature_m : float
        Smallest physical geometric feature [m].
    active_axes : tuple[str, ...]
        Active coordinate axes participating in wave propagation.
    feature_axis : str
        Axis along which the smallest critical feature is resolved.

    Returns
    -------
    metrics : dict[str, Any]
        Dictionary with ``dx_background`` [um], ``dx_min`` [um],
        ``cells_across_feature`` [-], ``n_cells`` [-], and per-axis spacings.
    """
    spacings_m: dict[str, np.ndarray] = {
        "x": np.asarray(grid.dx, dtype=float),
        "y": np.asarray(grid.dy, dtype=float),
        "z": np.asarray(grid.dz, dtype=float),
    }
    active_all = np.concatenate([spacings_m[ax] for ax in active_axes])
    dx_bg_um = float(np.max(active_all) * 1e6)
    dx_min_um = float(np.min(active_all) * 1e6)
    feat_step_m = float(np.min(spacings_m[feature_axis]))
    cells_across_feature = float(min_feature_m / feat_step_m)
    edge_ghosts = getattr(grid, "edge_ghosts", [1, 1, 1])
    nx_cells = max(int(grid.Nx) - int(edge_ghosts[0]), 1)
    ny_cells = max(int(grid.Ny) - int(edge_ghosts[1]), 1)
    nz_cells = max(int(grid.Nz) - int(edge_ghosts[2]), 1)
    n_cells = int(nx_cells * ny_cells * nz_cells)

    return {
        "dx_background": dx_bg_um,
        "dx_min": dx_min_um,
        "cells_across_feature": cells_across_feature,
        "n_cells": n_cells,
        "nx": nx_cells,
        "ny": ny_cells,
        "nz": nz_cells,
        "dx_um": float(np.min(spacings_m["x"]) * 1e6),
        "dy_um": float(np.min(spacings_m["y"]) * 1e6),
        "dz_um": float(np.min(spacings_m["z"]) * 1e6),
        "min_feature_um": float(min_feature_m * 1e6),
    }


# =====================================================================
# 4. Two-Stage Bloch Time-Domain Eigenmode & DFT Extraction
# =====================================================================


class HannDFTMonitor(DFTMonitor):
    """Windowed running DFT monitor over a fixed time span.

    Applies a continuous Hann taper :math:`w(\\tau) = 1 - \\cos(2\\pi \\tau / T_{\\mathrm{win}})`
    (normalized to unit coherent gain) evaluated at each field component's native
    leapfrog time level (`t + dt` for `E`/`J` and `t + dt/2` for `H`). Combined
    with choosing `n_steps_window` to span an integer number of RF cycles at the
    target mode frequency, this suppresses negative-frequency ($-f_m$) and
    out-of-band spectral leakage below $10^{-6}$ while preserving the exact
    electric-to-magnetic phasor ratio $|\\hat{\\mathbf{E}}| / |\\hat{\\mathbf{H}}|$.

    Parameters
    ----------
    frequencies : array_like
        Monitored frequencies [Hz].
    n_steps_window : int
        Total number of time steps in the accumulation window.
    components : sequence of str, optional
        Field components to monitor.
    center, size : tuple of float, optional
        Spatial center and extent of the monitored box [m].
    window : {'hann', 'boxcar'}, optional
        Temporal taper profile. Default is ``'hann'``.
    """

    def __init__(
        self,
        frequencies,
        n_steps_window: int,
        components=("Ex", "Ey", "Ez"),
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
        window: str = "hann",
    ):
        super().__init__(
            frequencies=frequencies,
            components=components,
            center=center,
            size=size,
        )
        self.n_steps_window = max(int(n_steps_window), 1)
        self.window = str(window).lower()
        self.t0_window: float | None = None

    def update(self, solver, t):
        """Accumulate windowed field samples after the step starting at time `t` [s]."""
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False
            self.t0_window = float(t)

        t_span = max(self.n_steps_window * float(solver.dt), float(solver.dt))
        t0 = float(self.t0_window) if self.t0_window is not None else float(t)
        times = {"E": t + solver.dt, "J": t + solver.dt, "H": t + solver.dt / 2}
        kernels = {}
        for f_key in {comp[0] for comp in self.components}:
            if self.window == "hann":
                frac = np.clip((times[f_key] - t0) / t_span, 0.0, 1.0)
                w_val = float(1.0 - np.cos(2.0 * np.pi * frac))
            elif self.window == "boxcar":
                w_val = 1.0
            else:
                raise ValueError(f"Unsupported DFT window={self.window!r}")
            kernels[f_key] = (solver.dt * w_val) * self.xp.exp(
                -2j * np.pi * self._freqs * times[f_key]
            )[:, None]

        for comp in self.components:
            values = getattr(solver, comp[0]).array[self._flat[comp]]
            self._acc[comp] += kernels[comp[0]] * values[None, :]


def project_floquet_space_harmonics(
    ez_field: np.ndarray,
    grid: GridFIT3D,
    beta0_rad_m: float,
    n_harm: int = 6,
    *,
    truncate_off_by_one: bool = False,
) -> dict[str, np.ndarray]:
    """Project a 3D axial electric-field array onto Floquet space harmonics.

    Evaluates the discrete Fourier projection over all :math:`N_z` axial cells
    of one periodic unit cell :math:`z \\in [z_{\\min}, z_{\\max}]` of length
    :math:`d = z_{\\max} - z_{\\min}`:

    .. math::
        E_{zn}(x, y) = \\frac{1}{N_z}\\sum_{k=0}^{N_z - 1}
        E_z(x, y, z_k)\\,e^{+j\\beta_n z_k}, \\qquad
        z_k = \\left(k + \\frac{1}{2}\\right)\\frac{d}{N_z}, \\quad
        \\beta_n = \\beta_0 + \\frac{2\\pi n}{d}.

    When ``bc_z = 'bloch'``, ``SolverFIT3D`` calls ``grid.remove_edge_ghost(2)``,
    setting ``grid.edge_ghosts[2] = 0`` and ``int(grid.Nz) = N_z``. Using
    ``nz = int(grid.Nz) - int(grid.edge_ghosts[2])`` therefore sums over all
    :math:`N_z` cells of the period and preserves exact discrete orthogonality
    to machine precision ($< 10^{-15}$).

    Parameters
    ----------
    ez_field : np.ndarray
        Complex array of shape ``(n_x_ez, n_y_ez, n_z_samples)`` containing
        :math:`E_z` at the staggered axial cell centers.
    grid : GridFIT3D
        Wakis grid defining the unit-cell period and axial discretization.
    beta0_rad_m : float
        Fundamental Floquet wavenumber :math:`\\beta_0` [rad/m].
    n_harm : int, optional
        Half-width of harmonic indices :math:`n \\in [-n_{\\mathrm{harm}}, +n_{\\mathrm{harm}}]`.
    truncate_off_by_one : bool, optional
        If True, sums over :math:`N_z - 1` cells (``nz = int(grid.Nz) - 1``)
        instead of the full :math:`N_z`-cell period, used for mutation testing
        of discrete Floquet orthogonality. Default is False.

    Returns
    -------
    proj : dict[str, np.ndarray]
        Dictionary with ``n_indices``, ``beta_n`` [rad/m], ``z_centers_m`` [m],
        and complex ``ez_harm`` of shape ``(2 * n_harm + 1, n_x_ez, n_y_ez)``.
    """
    edge_ghosts = getattr(grid, "edge_ghosts", [0, 1, 0])
    if truncate_off_by_one:
        nz = max(int(grid.Nz) - 1, 1)
    else:
        nz = max(int(grid.Nz) - int(edge_ghosts[2]), 1)

    d_period = float(grid.zmax - grid.zmin)
    dz = d_period / float(nz)
    z_centers = (np.arange(nz, dtype=float) + 0.5) * dz
    n_indices = np.arange(-int(n_harm), int(n_harm) + 1, dtype=int)
    beta_n = float(beta0_rad_m) + 2.0 * np.pi * n_indices / d_period
    phase_mat = np.exp(1j * beta_n[:, None] * z_centers[None, :]) / float(nz)

    ez_nodes = np.asarray(ez_field, dtype=np.complex128)[:, :, :nz]
    ez_harm = np.einsum("ij,xyj->ixy", phase_mat, ez_nodes)
    return {
        "n_indices": n_indices,
        "beta_n": beta_n,
        "z_centers_m": z_centers,
        "ez_harm": ez_harm,
    }


def run_bloch_harminv_point(
    grid: GridFIT3D,
    beta0_rad_m: float,
    f_center_hz: float,
    f_width_hz: float,
    f_min_hz: float,
    f_max_hz: float,
    probe_Specs: list[tuple[str, tuple[float, float, float]]],
    source_specs: list[tuple[str, tuple[float, float, float], float]],
    n_steps: int = 5000,
    min_q: float = 20.0,
    bc_low: list[str] | None = None,
    bc_high: list[str] | None = None,
) -> dict[str, Any]:
    """Run Stage 1 of the Bloch cold test: broadband kick + harminv extraction.

    Applies Bloch-periodic boundary conditions along z with
    ``k_bloch = (0.0, 0.0, -beta0_rad_m)`` so that positive-frequency DFT/harminv
    phasors e^{+j omega t} have spatial variation e^{-j beta_n z}, excites the
    unit cell with a broadband Gaussian dipole kick at asymmetric positions, and
    analyzes the post-pulse free ring-down with ``ResonanceMonitor``.

    Parameters
    ----------
    grid : GridFIT3D
        Constructed Wakis grid.
    beta0_rad_m : float
        Fundamental Floquet wavenumber beta_0 [rad/m].
    f_center_hz, f_width_hz : float
        Center frequency and bandwidth [Hz] of the Gaussian excitation.
    f_min_hz, f_max_hz : float
        Frequency window [Hz] for harmonic inversion.
    probe_Specs : list[tuple[str, tuple[float, float, float]]]
        List of `(component, (x, y, z))` probes for ``ResonanceMonitor``.
    source_specs : list[tuple[str, tuple[float, float, float], float]]
        List of `(component, (x, y, z), amplitude)` dipoles for ``CurrentSource``.
    n_steps : int
        Number of time steps to evolve.
    min_q : float
        Minimum |Q| quality factor threshold for harminv mode filtering.
    bc_low, bc_high : list[str] or None
        Boundary conditions on (x, y, z). Defaults to ``['periodic', 'pec', 'bloch']``
        for 2D (`grid.Nx <= 2`) or ``['pec', 'pec', 'bloch']`` for 3D.

    Returns
    -------
    res : dict[str, Any]
        Dictionary with extracted ``frequencies_hz``, ``q_factors``,
        ``amplitudes``, ``per_probe_frequencies_hz``, ``n_steps``, ``dt_s``,
        and ``wall_time_s``.
    """
    default_x_bc = "periodic" if grid.Nx <= 2 else "pec"
    if bc_low is None:
        bc_low = [default_x_bc, "pec", "bloch"]
    if bc_high is None:
        bc_high = [default_x_bc, "pec", "bloch"]

    solver = SolverFIT3D(
        grid,
        use_stl=True,
        bc_low=bc_low,
        bc_high=bc_high,
        dtype=np.complex128,
        k_bloch=(0.0, 0.0, -float(beta0_rad_m)),
        verbose=0,
    )
    pulse = GaussianSource(frequency=float(f_center_hz), fwidth=float(f_width_hz))
    sources = [
        CurrentSource(pulse, component=comp, center=center, amplitude=amp)
        for (comp, center, amp) in source_specs
    ]
    monitors = [
        ResonanceMonitor(comp, center=center, t_start=pulse.end_time)
        for (comp, center) in probe_Specs
    ]

    t0 = time.perf_counter()
    solver.emsolve(int(n_steps), source=sources, monitors=monitors, verbose=0)
    wall_s = time.perf_counter() - t0

    all_freqs: list[float] = []
    all_qs: list[float] = []
    all_amps: list[float] = []
    per_probe: list[np.ndarray] = []

    for mon in monitors:
        hres = mon.harminv(float(f_min_hz), float(f_max_hz))
        f_arr = np.asarray(hres.frequency, dtype=float)
        q_arr = np.abs(np.asarray(hres.Q, dtype=float))
        a_arr = np.abs(np.asarray(hres.amplitude))
        mask = q_arr >= float(min_q)
        f_arr = f_arr[mask]
        q_arr = q_arr[mask]
        a_arr = a_arr[mask]
        per_probe.append(f_arr)
        for f_v, q_v, a_v in zip(f_arr, q_arr, a_arr):
            all_freqs.append(float(f_v))
            all_qs.append(float(q_v))
            all_amps.append(float(a_v))

    # Sort and deduplicate modes across probes within 0.25 GHz
    order = np.argsort(all_freqs)
    dedup_f: list[float] = []
    dedup_q: list[float] = []
    dedup_a: list[float] = []
    for idx in order:
        fv = all_freqs[idx]
        qv = all_qs[idx]
        av = all_amps[idx]
        if not dedup_f or abs(fv - dedup_f[-1]) > 0.25e9:
            dedup_f.append(fv)
            dedup_q.append(qv)
            dedup_a.append(av)
        else:
            if av > dedup_a[-1]:
                dedup_f[-1] = fv
                dedup_q[-1] = qv
                dedup_a[-1] = av

    return {
        "frequencies_hz": np.array(dedup_f, dtype=float),
        "q_factors": np.array(dedup_q, dtype=float),
        "amplitudes": np.array(dedup_a, dtype=float),
        "per_probe_frequencies_hz": per_probe,
        "n_steps": int(n_steps),
        "dt_s": float(solver.dt),
        "wall_time_s": float(wall_s),
    }


def extract_bloch_mode_dft(
    grid: GridFIT3D,
    beta0_rad_m: float,
    mode_freq_hz: float,
    source_specs: list[tuple[str, tuple[float, float, float], float]],
    n_harm: int = 6,
    fwidth_hz: float | None = None,
    n_ring_steps: int = 2500,
    beam_y_half_m: float = 0.075e-3,
    beam_x_half_m: float | None = None,
    bc_low: list[str] | None = None,
    bc_high: list[str] | None = None,
    window: str = "hann",
) -> dict[str, Any]:
    """Run Stage 2 of the Bloch cold test: narrowband drive + Hann-windowed ring-down DFT.

    Drives the structure with a narrowband Gaussian pulse centered at the
    harminv-identified eigenfrequency ``mode_freq_hz``, waits until the source
    pulse has completely turned off, and then records the free-oscillation
    complex fields with ``HannDFTMonitor([mode_freq_hz])`` over an integer number
    of RF cycles ``n_ring_cycles``. Windowing over an integer cycle count
    eliminates local dipole self-field contamination and suppresses negative-frequency
    ($-f_m$) and out-of-band spectral leakage below $10^{-6}$, while projecting
    over all :math:`N_z` axial cells of one period yields:
    - Axial Poynting power flow P_z [W],
    - Stored electromagnetic energy U_cell [J],
    - Energy velocity v_E = P_z * d / U_cell [m/s],
    - Space-harmonic amplitudes E_{zn}(x, y) via full-period Fourier projection,
    - On-axis and beam-averaged Pierce interaction impedance K_n [Ohm].

    Parameters
    ----------
    grid : GridFIT3D
        Constructed Wakis grid.
    beta0_rad_m : float
        Fundamental Floquet wavenumber beta_0 [rad/m].
    mode_freq_hz : float
        Target eigenfrequency f_m [Hz].
    source_specs : list[tuple[str, tuple[float, float, float], float]]
        Excitation dipole specifications `(component, (x, y, z), amplitude)`.
        Should be offset slightly from the on-axis measurement line.
    n_harm : int
        Half-width of space harmonics n in [-n_harm, +n_harm] to extract.
    fwidth_hz : float or None
        Bandwidth [Hz] of the narrowband pulse (defaults to 4.5% of ``mode_freq_hz``).
    n_ring_steps : int
        Target number of post-pulse free ring-down time steps (rounded to the
        nearest integer number of RF cycles at ``mode_freq_hz``).
    beam_y_half_m : float
        Half-thickness of the sheet beam along y [m] for beam-averaged K_n.
    beam_x_half_m : float or None
        Half-width of the sheet beam along x [m] for 3D beam-averaged K_n.
    bc_low, bc_high : list[str] or None
        Boundary conditions on (x, y, z).
    window : {'hann', 'boxcar'}, optional
        Temporal window applied to the ring-down DFT. Default is ``'hann'``.

    Returns
    -------
    mode_data : dict[str, Any]
        Dictionary containing Poynting power, stored energy, energy velocity,
        space-harmonic profiles, Pierce impedances, and 2D mid-plane field slices.
    """
    default_x_bc = "periodic" if grid.Nx <= 2 else "pec"
    if bc_low is None:
        bc_low = [default_x_bc, "pec", "bloch"]
    if bc_high is None:
        bc_high = [default_x_bc, "pec", "bloch"]

    solver = SolverFIT3D(
        grid,
        use_stl=True,
        bc_low=bc_low,
        bc_high=bc_high,
        dtype=np.complex128,
        k_bloch=(0.0, 0.0, -float(beta0_rad_m)),
        verbose=0,
    )
    if fwidth_hz is None:
        fwidth_hz = 0.045 * float(mode_freq_hz)

    pulse = GaussianSource(
        frequency=float(mode_freq_hz),
        fwidth=float(fwidth_hz),
        cutoff=4.0,
    )
    sources = [
        CurrentSource(pulse, component=comp, center=center, amplitude=amp)
        for (comp, center, amp) in source_specs
    ]

    n_drive = int(np.ceil(pulse.end_time / solver.dt)) + 10
    n_ring_cycles = max(
        int(round(int(n_ring_steps) * float(solver.dt) * float(mode_freq_hz))), 1
    )
    n_ring_actual = max(
        int(round(n_ring_cycles / (float(mode_freq_hz) * float(solver.dt)))), 1
    )

    lx = float(grid.xmax - grid.xmin)
    ly = float(grid.ymax - grid.ymin)
    lz = float(grid.zmax - grid.zmin)
    dft = HannDFTMonitor(
        [float(mode_freq_hz)],
        n_steps_window=n_ring_actual,
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        center=(0.0, 0.0, 0.5 * lz),
        size=(lx, ly, lz),
        window=window,
    )

    t0 = time.perf_counter()
    solver.emsolve(n_drive, source=sources, verbose=0)
    solver.emsolve(n_ring_actual, monitors=[dft], verbose=0)
    wall_s = time.perf_counter() - t0

    ex_raw, _ = dft.get_field("Ex")
    ey_raw, _ = dft.get_field("Ey")
    ez_raw, _ = dft.get_field("Ez")
    hx_raw, _ = dft.get_field("Hx")
    hy_raw, _ = dft.get_field("Hy")
    hz_raw, _ = dft.get_field("Hz")

    edge_ghosts = getattr(grid, "edge_ghosts", [0, 1, 0])
    nx = max(int(grid.Nx) - int(edge_ghosts[0]), 1)
    ny = max(int(grid.Ny) - int(edge_ghosts[1]), 1)
    nz = max(int(grid.Nz) - int(edge_ghosts[2]), 1)
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    d_period = lz

    # Average fields to cell centers (nx, ny, nz); z-wrap uses Bloch phase exp(-1j*beta0*d)
    phase_z = np.exp(-1j * beta0_rad_m * d_period)

    def _shift_z(arr: np.ndarray) -> np.ndarray:
        out = np.empty_like(arr)
        out[:, :, :-1] = arr[:, :, 1:]
        out[:, :, -1] = arr[:, :, 0] * phase_z
        return out

    ex0 = 0.5 * (ex_raw[0, :nx, :ny, :nz] + ex_raw[0, :nx, 1 : ny + 1, :nz])
    if nx > 1:
        ey0 = 0.5 * (ey_raw[0, :nx, :ny, :nz] + ey_raw[0, 1 : nx + 1, :ny, :nz])
    else:
        ey0 = ey_raw[0, :nx, :ny, :nz]
    ex_c = 0.5 * (ex0 + _shift_z(ex0))
    ey_c = 0.5 * (ey0 + _shift_z(ey0))

    if nx > 1:
        hx_c = 0.5 * (hx_raw[0, :nx, :ny, :nz] + hx_raw[0, 1 : nx + 1, :ny, :nz])
    else:
        hx_c = hx_raw[0, :nx, :ny, :nz]
    hy_c = 0.5 * (hy_raw[0, :nx, :ny, :nz] + hy_raw[0, :nx, 1 : ny + 1, :nz])
    hz_c = hz_raw[0, :nx, :ny, :nz]

    # Axial Poynting vector S_z = 0.5 * Re(E_x H_y* - E_y H_x*)
    sz_3d = 0.5 * np.real(ex_c * np.conj(hy_c) - ey_c * np.conj(hx_c))
    pz_vs_z = np.sum(sz_3d, axis=(0, 1)) * dx * dy
    pz_mean_w = float(np.sum(sz_3d) * dx * dy * dz / d_period)

    # Time-averaged stored energy U_cell = 2 * U_H = 0.5 * mu0 * int |H|^2 dV
    u_cell_j = float(
        0.5
        * MU0
        * np.sum(np.abs(hx_c) ** 2 + np.abs(hy_c) ** 2 + np.abs(hz_c) ** 2)
        * dx
        * dy
        * dz
    )
    v_energy_m_s = float(pz_mean_w * d_period / max(u_cell_j, 1e-30))

    # Space-harmonic projection E_{zn}(x, y) over all Nz axial cells of one period
    n_x_ez = ez_raw.shape[1]
    n_y_ez = ez_raw.shape[2]
    ez_nodes = ez_raw[0, :n_x_ez, :n_y_ez, :nz]
    x_nodes = np.linspace(grid.xmin, grid.xmax, n_x_ez)
    y_nodes = np.linspace(grid.ymin, grid.ymax, n_y_ez)

    proj = project_floquet_space_harmonics(
        ez_nodes,
        grid=grid,
        beta0_rad_m=beta0_rad_m,
        n_harm=n_harm,
    )
    n_indices = proj["n_indices"]
    beta_n = proj["beta_n"]
    z_centers = proj["z_centers_m"]
    ez_harm = proj["ez_harm"]

    ix_mid = 0 if nx == 1 else nx // 2
    iy_mid = ny // 2
    ez_harm_x0 = ez_harm[:, ix_mid, :]
    ez_harm_axis = ez_harm[:, ix_mid, iy_mid]

    p_abs = max(abs(pz_mean_w), 1e-30)
    k_n_axis = (np.abs(ez_harm_axis) ** 2) / (2.0 * (beta_n**2) * p_abs)

    # Beam cross-section averaged Pierce impedance K_{n,avg}
    y_beam_mask = np.abs(y_nodes) <= max(beam_y_half_m, dy)
    if not np.any(y_beam_mask):
        y_beam_mask[iy_mid] = True
    if beam_x_half_m is None or nx == 1:
        x_beam_mask = np.zeros(n_x_ez, dtype=bool)
        x_beam_mask[ix_mid] = True
    else:
        x_beam_mask = np.abs(x_nodes) <= max(beam_x_half_m, dx)
        if not np.any(x_beam_mask):
            x_beam_mask[ix_mid] = True

    ez_sq_beam_avg = np.mean(
        np.abs(ez_harm[:, x_beam_mask, :][:, :, y_beam_mask]) ** 2,
        axis=(1, 2),
    )
    k_n_beam_avg = ez_sq_beam_avg / (2.0 * (beta_n**2) * p_abs)

    # Mid-plane slice E_z(y, z) at x = 0
    ez_yz_slice = ez_nodes[ix_mid, :ny, :nz].copy()
    ref_phasor = ez_yz_slice[iy_mid, nz // 2]
    if abs(ref_phasor) > 0:
        ez_yz_slice = ez_yz_slice * np.exp(-1j * np.angle(ref_phasor))

    return {
        "mode_freq_hz": float(mode_freq_hz),
        "beta0_rad_m": float(beta0_rad_m),
        "pz_mean_w": pz_mean_w,
        "pz_vs_z_w": pz_vs_z,
        "u_cell_j": u_cell_j,
        "v_energy_m_s": v_energy_m_s,
        "v_energy_over_c": float(v_energy_m_s / C0),
        "n_indices": n_indices,
        "beta_n": beta_n,
        "ez_harm_axis": np.abs(ez_harm_axis),
        "ez_harm_y_profile": np.abs(ez_harm_x0),
        "k_n_axis_ohm": k_n_axis,
        "k_n_beam_avg_ohm": k_n_beam_avg,
        "ez_yz_slice": ez_yz_slice,
        "y_nodes_m": y_nodes,
        "z_centers_m": z_centers,
        "n_ring_cycles": int(n_ring_cycles),
        "dft_window": str(window),
        "n_steps": int(n_drive + n_ring_actual),
        "wall_time_s": float(wall_s),
    }


def record_bloch_wave_frames(
    grid: GridFIT3D,
    beta0_rad_m: float,
    mode_freq_hz: float,
    source_specs: list[tuple[str, tuple[float, float, float], float]],
    n_frames: int = 20,
) -> dict[str, Any]:
    """Capture time-domain snapshots of Re{E_z(y, z, t)} over one RF cycle.

    Parameters
    ----------
    grid : GridFIT3D
        2D Wakis grid.
    beta0_rad_m : float
        Floquet wavenumber beta_0 [rad/m].
    mode_freq_hz : float
        Eigenfrequency f_m [Hz].
    source_specs : list[tuple[str, tuple[float, float, float], float]]
        Narrowband excitation dipoles.
    n_frames : int
        Number of frames across one optical/RF period T = 1 / f_m.

    Returns
    -------
    movie_data : dict[str, Any]
        Dictionary containing ``frames_ez`` (shape `(n_frames, Ny, Nz)`),
        ``phase_rad``, ``y_um``, ``z_um``, and ``vac_mask_yz``.
    """
    default_x_bc = "periodic" if grid.Nx <= 2 else "pec"
    solver = SolverFIT3D(
        grid,
        use_stl=True,
        bc_low=[default_x_bc, "pec", "bloch"],
        bc_high=[default_x_bc, "pec", "bloch"],
        dtype=np.complex128,
        k_bloch=(0.0, 0.0, -float(beta0_rad_m)),
        verbose=0,
    )
    pulse = GaussianSource(
        frequency=float(mode_freq_hz),
        fwidth=float(0.05 * mode_freq_hz),
        cutoff=4.0,
    )
    sources = [
        CurrentSource(pulse, component=comp, center=center, amplitude=amp)
        for (comp, center, amp) in source_specs
    ]
    n_drive = int(np.ceil(pulse.end_time / solver.dt)) + 20
    solver.emsolve(n_drive, source=sources, verbose=0)

    steps_per_cycle = max(int(round(1.0 / (mode_freq_hz * solver.dt))), n_frames)
    stride = max(steps_per_cycle // n_frames, 1)

    edge_ghosts = getattr(grid, "edge_ghosts", [0, 1, 0])
    nx = max(int(grid.Nx) - int(edge_ghosts[0]), 1)
    ny = max(int(grid.Ny) - int(edge_ghosts[1]), 1)
    nz = max(int(grid.Nz) - int(edge_ghosts[2]), 1)
    ix_mid = 0 if nx == 1 else nx // 2
    vac_mask_yz = solver.ieps[ix_mid, :ny, :nz, "z"] > 0.0
    frames = []
    phases = []
    for k in range(n_frames):
        ez_arr = np.real(solver.E[ix_mid, :ny, :nz, "z"]) * vac_mask_yz
        frames.append(ez_arr.copy())
        phases.append(float(2.0 * np.pi * k * stride / steps_per_cycle))
        solver.emsolve(stride, verbose=0)

    y_um = 0.5 * (np.asarray(grid.y[:ny]) + np.asarray(grid.y[1 : ny + 1])) * 1e6
    z_um = 0.5 * (np.asarray(grid.z[:nz]) + np.asarray(grid.z[1 : nz + 1])) * 1e6
    return {
        "frames_ez": np.array(frames, dtype=float),
        "phase_rad": np.array(phases, dtype=float),
        "y_um": y_um,
        "z_um": z_um,
        "vac_mask_yz": vac_mask_yz,
    }


def estimate_sws_task_resources(
    nx: int,
    ny: int,
    nz: int,
    n_steps: int,
    n_subruns: int = 1,
    conformal: str | bool = "pec",
) -> dict[str, Any]:
    """Estimate memory and CPU runtime for an SWS task using ``wakis.estimate``.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    n_steps : int
        Time steps per subrun.
    n_subruns : int
        Number of Bloch phase / DFT subruns executed by the task.
    conformal : {'pec', False}
        Conformal PEC mode.

    Returns
    -------
    est : dict[str, Any]
        Resource estimate dictionary with ``n_cells``, ``n_steps_total``,
        ``ram_working_mb``, and ``cpu_seconds_est``.
    """
    est_obj = estimate(
        (int(nx), int(ny), int(nz)),
        steps=int(n_steps * n_subruns),
        dtype="complex128",
        conformal=conformal,
    )
    return {
        "n_cells": int(est_obj.cells),
        "n_steps_per_subrun": int(n_steps),
        "n_subruns": int(n_subruns),
        "n_steps_total": int(n_steps * n_subruns),
        "ram_working_mb": float(est_obj.host_bytes / 1e6),
        "cpu_seconds_est": float(est_obj.seconds),
        "setup_seconds_est": float(est_obj.setup_seconds),
        "step_ms_est": float(est_obj.step_seconds * 1e3),
    }
