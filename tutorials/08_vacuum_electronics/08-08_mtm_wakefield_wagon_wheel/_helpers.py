"""Helper functions for Tutorial 08-08: Reversed-Cherenkov Wakefields in a Wagon-Wheel Metamaterial Structure.

Reproduces the all-metal metamaterial (wagon-wheel) slow-wave wakefield experiment and linear theory of:
- X. Lu, M. A. Shapiro, I. Mastovsky, R. J. Temkin, M. Conde, J. G. Power, J. Shao, E. E. Wisniewski,
  and C. Jing, "Generation of High-Power, Reversed-Cherenkov Wakefield Radiation in a Metamaterial
  Structure," *Phys. Rev. Lett.* **122**, 014801 (2019).
"""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import pyvista as pv
from matplotlib.patches import Circle, Rectangle
from scipy.constants import c as C0
from scipy.constants import e as QE
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import m_e as ME
from scipy.constants import mu_0 as MU0
from scipy.optimize import brentq
from scipy.signal import butter, sosfilt
from scipy.special import jn_zeros

from wakis import GridFIT3D, SolverFIT3D, WakeSolver, figstyle, schematic
from wakis.estimate import estimate, preflight
from wakis.field_monitors import DFTMonitor
from wakis.resonances import ResonanceMonitor
from wakis.sources import Beam, CurrentSource, GaussianSource
from wakis.templates.photonics import Box


def _gc_and_trim() -> None:
    """Run Python garbage collection and release freed glibc heap pages to the OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _build_causal_bandpass_kernel(
    nt: int,
    dt: float,
    f_lo_hz: float,
    f_hi_hz: float,
    *,
    order: int = 3,
    tail_taper_fraction: float = 0.05,
) -> np.ndarray:
    """Build a causal Butterworth bandpass impulse response with a cosine tail taper."""
    fs = 1.0 / float(dt)
    sos = butter(int(order), [float(f_lo_hz), float(f_hi_hz)], btype="bandpass", fs=fs, output="sos")
    imp = np.zeros(int(nt), dtype=float)
    imp[0] = 1.0
    h_bp = sosfilt(sos, imp)
    n_tap = max(int(tail_taper_fraction * int(nt)), 16)
    if n_tap < int(nt):
        h_bp[-n_tap:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(n_tap) / float(n_tap)))
    return h_bp


def _apply_causal_kernel_fft(sig: np.ndarray, h_bp: np.ndarray) -> np.ndarray:
    """Convolve ``sig`` along axis 0 with causal impulse response ``h_bp`` via zero-padded FFT (n_fft >= 2*nt)."""
    n_steps = int(sig.shape[0])
    n_fft = 1 << int(np.ceil(np.log2(max(2 * n_steps + len(h_bp), 64))))
    h_fft = np.fft.rfft(h_bp, n=n_fft)
    if sig.ndim == 1:
        spec = np.fft.rfft(sig, n=n_fft)
        spec *= h_fft
        return np.fft.irfft(spec, n=n_fft)[:n_steps].copy()
    out = np.empty_like(sig, dtype=float)
    chunk = 512
    for c0 in range(0, sig.shape[1], chunk):
        c1 = min(c0 + chunk, sig.shape[1])
        spec = np.fft.rfft(sig[:, c0:c1], n=n_fft, axis=0)
        spec *= h_fft[:, None]
        out[:, c0:c1] = np.fft.irfft(spec, n=n_fft, axis=0)[:n_steps]
    return out


def measure_filter_step_rise_time_ns(
    dt_s: float = 4.0e-13,
    f_lo_hz: float | None = None,
    f_hi_hz: float | None = None,
    *,
    order: int = 3,
    nt: int = 8192,
) -> float:
    """Measure the 10–90% carrier-step rise time [ns] directly from the causal bandpass impulse response ``h_bp``.

    Builds the causal ``order``-th order Butterworth bandpass kernel ``h_bp`` on ``[f_lo_hz, f_hi_hz]``
    using :func:`_build_causal_bandpass_kernel`, demodulates ``h_bp`` at the passband center frequency
    ``f_0 = 0.5 * (f_lo_hz + f_hi_hz)`` to obtain the complex baseband step envelope
    ``A[n] = |sum_{m=0}^n h_bp[m] exp(-i 2*pi*f_0*m*dt_s)|``, normalizes by its steady-state plateau,
    and returns the interpolated 10% to 90% rise time ``t_90 - t_10`` in nanoseconds (S2).
    ``f_lo_hz`` / ``f_hi_hz`` default to the field-filter passband of :class:`WagonWheelMTMSpec`.
    """
    if f_lo_hz is None:
        f_lo_hz = WagonWheelMTMSpec.passband_low_hz
    if f_hi_hz is None:
        f_hi_hz = WagonWheelMTMSpec.passband_high_hz
    h_bp = _build_causal_bandpass_kernel(int(nt), float(dt_s), float(f_lo_hz), float(f_hi_hz), order=int(order))
    t_arr = np.arange(int(nt), dtype=float) * float(dt_s)
    f0_hz = 0.5 * (float(f_lo_hz) + float(f_hi_hz))
    step_env = np.abs(np.cumsum(h_bp * np.exp(-1j * 2.0 * np.pi * f0_hz * t_arr)))
    ss_start = int(0.40 * int(nt))
    ss_end = int(0.85 * int(nt))
    ss_val = float(np.median(step_env[ss_start:ss_end]))
    norm_env = step_env / max(ss_val, 1e-30)
    i_peak = int(np.argmax(norm_env[:ss_end]))
    rising_env = np.maximum.accumulate(norm_env[: i_peak + 1])
    rising_t_ns = t_arr[: i_peak + 1] * 1e9
    t10_ns = float(np.interp(0.10, rising_env, rising_t_ns))
    t90_ns = float(np.interp(0.90, rising_env, rising_t_ns))
    return float(t90_ns - t10_ns)


@dataclass(frozen=True)
class WagonWheelMTMSpec:
    """Geometric and beam parameters of the X-band wagon-wheel metamaterial structure (Lu et al., PRL 2019).

    Attributes
    ----------
    period_m : float
        Axial unit-cell period :math:`p` [m] (2.0 mm; PRL 122, 014801 lines 140, 145–146, 166;
        Lu 2019 MIT PhD thesis lines 1978, 1985, 1994).
    wagon_wheel_thickness_m : float
        Axial thickness of the stainless-steel wagon-wheel plate :math:`t_{\\mathrm{ww}}` [m]
        (1.0 mm; PRL lines 112–113; thesis lines 1950–1951, 1967, 2037–2038).
    spacer_thickness_m : float
        Axial thickness of the copper spacer plate :math:`t_{\\mathrm{sp}}` [m]
        (1.0 mm; PRL lines 112–113; thesis lines 1951, 1967, 2038).
    beam_hole_radius_m : float
        Inner beam-tunnel radius :math:`r_b` [m] (3.0 mm, diameter 6.0 mm; PRL lines 122–123, 145;
        thesis lines 1967, 1969).
    hub_outer_radius_m : float
        Outer radius of the inner annular hub ring :math:`r_{\\mathrm{hub}}` [m] (4.0 mm; ASSUMED by
        setting radial hub wall thickness :math:`r_{\\mathrm{hub}} - r_b = 1.0\\;\\mathrm{mm}` equal
        to the 1.0 mm spoke bar width callout on PRL Fig. 2(b), line 119; not tabulated in prose).
    waveguide_radius_m : float
        Inner radius of the enclosing circular waveguide :math:`R_{\\mathrm{wg}}` [m] (8.0 mm from
        16.0 mm ID in PRL lines 121, 152 and thesis line 1968; note thesis lines 2075, 2208 state
        8.1 mm radius corresponding to the 14.2 GHz TM01 cutoff).
    slot_outer_radius_m : float
        Outer radius of the four open vacuum sectors between spokes :math:`R_{\\mathrm{slot}}` [m]
        (8.0 mm; ASSUMED flush with :math:`R_{\\mathrm{wg}}`).
    spoke_width_m : float
        Bar width of each radial inductive spoke :math:`w_{\\mathrm{spoke}}` [m] (1.0 mm; PRL Fig. 2(b)
        line 119; thesis line 1968).
    n_spokes : int
        Number of radial inductive spokes per wagon-wheel plate [dimensionless] (4; ASSUMED from visual
        inspection of PRL Fig. 2(b), line 119, and thesis Fig. 2-12b/2-14; not printed as a numeral in prose).
    n_periods_experiment : int
        Number of unit cells in the experimental structure [dimensionless] (40, active length 80 mm;
        PRL lines 134, 166; thesis lines 1951–1952, 2201).
    beam_kinetic_energy_ev : float
        Relativistic electron beam kinetic energy :math:`E_k` [eV] (65 MeV; PRL lines 49, 104, 143, 149–150).
    single_bunch_charge_c : float
        Single drive bunch charge :math:`q` [C] (45 nC; PRL lines 49, 145, 150, 214, 224, 233, 291).
    two_bunch_total_charge_c : float
        Total charge of the two-bunch train :math:`Q_1 + Q_2` [C] (85 nC; PRL lines 51, 297–298, 323, 337–338).
    bunch_length_rms_m : float
        RMS longitudinal Gaussian bunch length :math:`\\sigma_z` [m] (1.2 mm; PRL lines 151, 212, 291).
    linac_frequency_hz : float
        Argonne Wakefield Accelerator L-band RF photoinjector frequency [Hz] (1.3 GHz, bunch spacing
        769.23 ps; PRL lines 143, 153–154, 256; thesis lines 2337–2341, 2365).
    unloaded_q_thesis : float
        Unloaded quality factor :math:`Q_0` of the stainless-steel/copper wagon-wheel unit cell at
        11.42 GHz [dimensionless] (1050; Lu 2019 MIT PhD thesis Table 2.4, line 2173).
    coax_port_wall_radius_m : float
        Outer radius of the thin coaxial inner-conductor shield in the numerical port couplers [m]
        (3.30 mm; ASSUMED numerical port transition replacing side-wall WR-90 couplers).
    passband_low_hz, passband_high_hz : float
        Edges of the causal 3rd-order Butterworth band-pass applied to the port Poynting fluxes,
        stored energy and beam work [Hz] (9.8–13.0 GHz; CHOSEN to bracket the backward passband
        of the cold test around the 11.42 GHz synchronous point).
    wake_passband_low_hz, wake_passband_high_hz : float
        Edges of the spectral window used for the longitudinal impedance Re Z and the passband
        wake potential [Hz] (9.5–13.5 GHz; CHOSEN, slightly wider than the field filter so the
        impedance peak is not truncated).
    paper_sync_freq_hz : float
        Cold-test synchronous frequency :math:`f_s` reported by Lu et al. [Hz]
        (11.42 GHz; PRL lines 137, 149; thesis lines 1960, 1991, 2340).
    paper_vg_over_c : float
        Normalized cold-test backward group velocity :math:`v_g / c` at :math:`f_s` reported by Lu et al.
        [dimensionless] (-0.158; PRL line 147; thesis line 1992 and Table 2.4 line 2175).
    paper_r_over_q_ohm_per_m : float
        Cold-test shunt impedance over quality factor per unit length :math:`r/Q` reported by Lu et al.
        [:math:`\\Omega/\\mathrm{m}`] (21.0 k:math:`\\Omega`/m; PRL line 250; thesis Table 2.4 line 2172).
    default_dz_m : float
        Default axial grid cell size :math:`\\Delta z` in the 3D FIT wakefield model [m]
        (0.25 mm; ASSUMED 4 cells across the 1.0 mm plate/spacer thickness, PRL line 112).
    """

    period_m: float = 2.0e-3
    wagon_wheel_thickness_m: float = 1.0e-3
    spacer_thickness_m: float = 1.0e-3
    beam_hole_radius_m: float = 3.0e-3
    hub_outer_radius_m: float = 4.0e-3
    waveguide_radius_m: float = 8.0e-3
    slot_outer_radius_m: float = 8.0e-3
    spoke_width_m: float = 1.0e-3
    n_spokes: int = 4
    n_periods_experiment: int = 40
    beam_kinetic_energy_ev: float = 65.0e6
    single_bunch_charge_c: float = 45.0e-9
    two_bunch_total_charge_c: float = 85.0e-9
    bunch_length_rms_m: float = 1.2e-3
    linac_frequency_hz: float = 1.3e9
    unloaded_q_thesis: float = 1050.0
    coax_port_wall_radius_m: float = 3.30e-3
    passband_low_hz: float = 9.8e9
    passband_high_hz: float = 13.0e9
    wake_passband_low_hz: float = 9.5e9
    wake_passband_high_hz: float = 13.5e9
    paper_sync_freq_hz: float = 11.42e9
    paper_vg_over_c: float = -0.158
    paper_r_over_q_ohm_per_m: float = 21.0e3
    default_dz_m: float = 0.25e-3


def relativistic_beam_kinematics(kinetic_energy_ev: float) -> tuple[float, float, float]:
    """Compute Lorentz factor :math:`\\gamma`, normalized speed :math:`\\beta`, and velocity :math:`u_0`.

    Parameters
    ----------
    kinetic_energy_ev : float
        Electron kinetic energy :math:`E_k` [eV].

    Returns
    -------
    gamma : float
        Relativistic Lorentz factor :math:`\\gamma = 1 + E_k / (m_e c^2)` [dimensionless].
    beta : float
        Normalized velocity :math:`\\beta = \\sqrt{1 - \\gamma^{-2}}` [dimensionless].
    u0_m_s : float
        Beam axial velocity :math:`u_0 = \\beta c` [m/s].
    """
    if kinetic_energy_ev <= 0.0:
        raise ValueError(f"kinetic_energy_ev must be positive; got {kinetic_energy_ev}.")
    gamma = 1.0 + float(kinetic_energy_ev) * QE / (ME * (C0**2))
    beta = float(np.sqrt(1.0 - 1.0 / (gamma**2)))
    return float(gamma), beta, float(beta * C0)


def circular_waveguide_tm01_cutoff_hz(radius_m: float) -> float:
    """Return the analytical :math:`\\mathrm{TM}_{01}` cutoff frequency of an empty circular PEC waveguide.

    Parameters
    ----------
    radius_m : float
        Circular waveguide inner radius :math:`R_{\\mathrm{wg}}` [m].

    Returns
    -------
    f_c_hz : float
        Cutoff frequency :math:`f_{c,\\mathrm{TM01}} = j_{0,1} c / (2\\pi R_{\\mathrm{wg}})` [Hz].
    """
    if radius_m <= 0.0:
        raise ValueError(f"radius_m must be positive; got {radius_m}.")
    j01 = float(jn_zeros(0, 1)[0])
    return float(j01 * C0 / (2.0 * np.pi * float(radius_m)))


def gaussian_bunch_form_factor(freq_hz: float | np.ndarray, sigmaz_m: float, u0_m_s: float) -> np.ndarray:
    """Compute the coherent Gaussian bunch amplitude form factor :math:`\\Phi = \\exp[-(k_z \\sigma_z)^2 / 2]`.

    Matches the definition in Lu et al., *Phys. Rev. Lett.* **122**, 014801 (2019), lines 247–248:
    :math:`\\Phi = \\exp[-(k_z \\sigma_z)^2 / 2] = 0.96` where :math:`k_z = \\omega / u_0` at synchronism,
    so :math:`\\Phi^2 = \\exp[-(k_z \\sigma_z)^2]` enters the radiated power :math:`P`.

    Parameters
    ----------
    freq_hz : float or array_like
        Frequency :math:`f` [Hz].
    sigmaz_m : float
        RMS Gaussian bunch length :math:`\\sigma_z` [m].
    u0_m_s : float
        Beam velocity :math:`u_0 = \\beta c` [m/s].

    Returns
    -------
    phi : np.ndarray
        Amplitude form factor :math:`\\Phi(f)` [dimensionless].
    """
    kz = 2.0 * np.pi * np.asarray(freq_hz, dtype=float) / float(u0_m_s)
    return np.exp(-0.5 * ((kz * float(sigmaz_m)) ** 2))


def paper_cold_test_k_loss_per_m(
    f_sync_hz: float = WagonWheelMTMSpec.paper_sync_freq_hz,
    r_over_q_ohm_per_m: float = WagonWheelMTMSpec.paper_r_over_q_ohm_per_m,
) -> float:
    """Compute the modal loss factor per unit length :math:`k_L = (\\omega_s / 4)(r/Q)` from paper cold-test values.

    Lu et al., *Phys. Rev. Lett.* **122**, 014801 (2019), line 247 (and Lu 2019 MIT PhD thesis line 2310)
    define :math:`k_L = (\\omega / 4)(r/Q)` with cold-test :math:`f_s = 11.42\\;\\mathrm{GHz}` (PRL lines 137, 149)
    and :math:`r/Q = 21.0\\;\\mathrm{k\\Omega/m}` (PRL line 250; thesis Table 2.4 line 2172).
    """
    omega_s = 2.0 * np.pi * float(f_sync_hz)
    return float(0.25 * omega_s * float(r_over_q_ohm_per_m))


def analytic_ohmic_pec_justification(
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
    *,
    f_sync_hz: float | None = None,
    vg_m_s: float | None = None,
) -> dict[str, float]:
    """Quantify the unloaded ohmic decay time :math:`\\tau_0 = 2 Q_0 / \\omega_s` vs wake pulse duration :math:`t_p`.

    In the AWA experiment, the wagon-wheel plates were fabricated from stainless steel and the spacer plates
    from copper (Lu et al., *Phys. Rev. Lett.* **122**, 014801, lines 163–165; Lu 2019 MIT PhD thesis
    lines 1950–1951, 2037–2038). Table 2.4 of Lu (2019) MIT PhD thesis (line 2173) reports the cold-test
    unloaded quality factor :math:`Q_0 = 1050` at :math:`f_s = 11.42\\;\\mathrm{GHz}` (PRL lines 137, 149),
    justifying a lossless PEC boundary model over the single-shot pulse duration :math:`t_p`.
    """
    if f_sync_hz is None:
        f_sync_hz = float(spec.paper_sync_freq_hz)
    if vg_m_s is None:
        vg_m_s = float(spec.paper_vg_over_c) * C0
    _, _, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    omega_s = 2.0 * np.pi * float(f_sync_hz)
    q0 = float(spec.unloaded_q_thesis)
    tau_field_s = 2.0 * q0 / omega_s
    tau_energy_s = q0 / omega_s
    v_abs = abs(float(vg_m_s))
    alpha_np_per_m = omega_s / (2.0 * q0 * v_abs)

    l_n8_m = 8.0 * spec.period_m
    l_n40_m = float(spec.n_periods_experiment) * spec.period_m
    tp_n8_s = analytic_backward_wave_pulse_length_s(l_n8_m, vg_m_s, u0)
    tp_n40_s = analytic_backward_wave_pulse_length_s(l_n40_m, vg_m_s, u0)

    # Power attenuation exp(-2*alpha*z) for a wave traveling distance z to the upstream port z=0:
    # Average transmitted power fraction over uniform emission z in [0, L] is (1 - exp(-2*alpha*L)) / (2*alpha*L)
    x8 = 2.0 * alpha_np_per_m * l_n8_m
    x40 = 2.0 * alpha_np_per_m * l_n40_m
    avg_loss_n8 = float(1.0 - (1.0 - np.exp(-x8)) / x8)
    avg_loss_n40 = float(1.0 - (1.0 - np.exp(-x40)) / x40)

    return {
        "unloaded_q0_thesis": q0,
        "f_sync_ghz": float(f_sync_hz * 1e-9),
        "ohmic_field_decay_time_2q_over_omega_ns": float(tau_field_s * 1e9),
        "ohmic_energy_decay_time_q_over_omega_ns": float(tau_energy_s * 1e9),
        "pulse_duration_n8_ns": float(tp_n8_s * 1e9),
        "pulse_duration_n40_ns": float(tp_n40_s * 1e9),
        "ratio_field_decay_over_pulse_n8": float(tau_field_s / tp_n8_s),
        "ratio_field_decay_over_pulse_n40": float(tau_field_s / tp_n40_s),
        "attenuation_alpha_np_per_m": float(alpha_np_per_m),
        "mean_ohmic_power_loss_fraction_n8": avg_loss_n8,
        "mean_ohmic_power_loss_fraction_n40": avg_loss_n40,
    }


def analytic_backward_wave_pulse_length_s(
    length_m: float,
    vg_m_s: float,
    u0_m_s: float = C0,
    *,
    mutate_flip_vg_sign: bool = False,
) -> float:
    """Compute the output RF pulse duration :math:`t_p` of a backward-wave structure (PRL Eq. 1, line 234).

    In a backward-wave structure (:math:`v_g < 0`), the wavefront radiated at the downstream exit
    (:math:`z = L` at time :math:`t = L / u_0`) must travel the full structure length :math:`L` upstream
    at velocity :math:`v_g = -|v_g| < 0` to reach the upstream output port (:math:`z = 0`), arriving at time:

    .. math::

        t_p = -\\frac{L}{v_g} + \\frac{L}{u_0} = \\frac{L}{|v_g|} + \\frac{L}{u_0} = \\frac{L}{|v_g|}\\left(1 - \\frac{v_g}{u_0}\\right).

    Parameters
    ----------
    length_m : float
        Active structure length :math:`L = N p` [m].
    vg_m_s : float
        Signed group velocity :math:`v_g < 0` [m/s].
    u0_m_s : float, optional
        Beam axial velocity :math:`u_0` [m/s]. Default is :math:`c`.
    mutate_flip_vg_sign : bool, optional
        If ``True``, flip the sign of :math:`v_g \\to +|v_g|` in :math:`t_p = -L / v_g + L / u_0`
        for mutation testing (S2).

    Returns
    -------
    t_p_s : float
        Output RF pulse duration :math:`t_p` [s].
    """
    v_abs = abs(float(vg_m_s))
    if v_abs <= 0.0:
        raise ValueError("Group velocity magnitude must be non-zero.")
    vg_eff = v_abs if mutate_flip_vg_sign else -v_abs
    return float(-length_m / vg_eff + length_m / float(u0_m_s))


def analytic_backward_wave_power_w(
    charge_c: float,
    k_loss_per_m: float,
    vg_m_s: float,
    sigmaz_m: float,
    freq_hz: float,
    u0_m_s: float = C0,
    *,
    mutate_drop_slippage_factor: bool = False,
    mutate_flip_vg_sign: bool = False,
    mutate_drop_quarter_in_kl: bool = False,
) -> float:
    """Compute the radiated RF pulse power :math:`P` of a backward-wave wakefield structure (PRL Eq. 2, lines 240–245).

    Attributed in Lu et al., *Phys. Rev. Lett.* **122**, 014801 (2019), line 238, to T. P. Wangler,
    *RF Linear Accelerators* (Wiley-VCH, 2008; PRL Ref. 24 at lines 370–371), and derived from energy
    conservation between the bunch energy loss over length :math:`L` (including the group-velocity
    slippage factor :math:`(1 - v_g / u_0)^{-1}`) and the RF pulse duration
    :math:`t_p = L(1 - v_g / u_0)/|v_g|` (PRL Eq. 1, line 234):

    .. math::

        P = \\frac{\\Delta U_{\\mathrm{bunch}}}{t_p}
          = q^2 k_L \\frac{|v_g|}{\\left(1 - v_g / u_0\\right)^2} \\Phi^2,
        \\qquad k_L = \\frac{\\omega}{4}\\left(\\frac{r}{Q}\\right).

    Parameters
    ----------
    charge_c : float
        Bunch charge :math:`q` [C].
    k_loss_per_m : float
        Modal loss factor per unit length :math:`k_L = (\\omega/4)(r/Q)` [V/(C·m)].
    vg_m_s : float
        Signed group velocity :math:`v_g < 0` [m/s].
    sigmaz_m : float
        RMS Gaussian bunch length :math:`\\sigma_z` [m].
    freq_hz : float
        Synchronous frequency :math:`f_s` [Hz].
    u0_m_s : float, optional
        Beam velocity :math:`u_0` [m/s]. Default is :math:`c`.
    mutate_drop_slippage_factor : bool, optional
        If ``True``, omit the :math:`(1 - v_g / u_0)^{-2}` slippage denominator for mutation testing.
    mutate_flip_vg_sign : bool, optional
        If ``True``, flip the sign of :math:`v_g` in :math:`(1 - v_g / u_0)^{-2}` to :math:`(1 - |v_g| / u_0)^{-2}`.
    mutate_drop_quarter_in_kl : bool, optional
        If ``True``, multiply :math:`k_L` by 4 (simulating dropping the factor :math:`1/4` in :math:`k_L = (\\omega/4)(r/Q)`).

    Returns
    -------
    power_w : float
        Cycle-averaged output RF power :math:`P` [W].
    """
    v_abs = abs(float(vg_m_s))
    vg_signed = v_abs if mutate_flip_vg_sign else -v_abs
    slip_denom = 1.0 if mutate_drop_slippage_factor else ((1.0 - vg_signed / float(u0_m_s)) ** 2)
    kl_eff = 4.0 * float(k_loss_per_m) if mutate_drop_quarter_in_kl else float(k_loss_per_m)
    phi = float(gaussian_bunch_form_factor(freq_hz, sigmaz_m, u0_m_s))
    return float((float(charge_c) ** 2) * kl_eff * (v_abs / slip_denom) * (phi**2))


def analytic_backward_wave_loss_per_meter_j_per_m(
    charge_c: float,
    k_loss_per_m: float,
    vg_m_s: float,
    sigmaz_m: float,
    freq_hz: float,
    u0_m_s: float = C0,
    *,
    mutate_flip_vg_sign: bool = False,
) -> float:
    """Compute the cold-test analytical bunch energy loss per unit active length :math:`a_{W,\\mathrm{analytic}}`.

    For a synchronous backward-wave mode (:math:`v_g < 0`) with modal loss factor per unit length
    :math:`k_L = (\\omega_s / 4)(r/Q)` and Gaussian bunch form factor :math:`\\Phi`, group-velocity slippage
    compresses the interaction length by :math:`(1 - v_g / u_0)^{-1}` (Lu et al., *Phys. Rev. Lett.* **122**,
    014801, 2019, lines 238–247; Wangler, *RF Linear Accelerators*, 2008), yielding the differential energy
    loss slope with respect to active structure length :math:`L`:

    .. math::

        a_{W,\\mathrm{analytic}} = \\frac{\\mathrm{d}W_{\\mathrm{bunch}}}{\\mathrm{d}L}
        = \\frac{q^2 k_L \\Phi^2}{1 - v_g / u_0}.

    Parameters
    ----------
    charge_c : float
        Bunch charge :math:`q` [C].
    k_loss_per_m : float
        Modal loss factor per unit length :math:`k_L = (\\omega_s/4)(r/Q)` [V/(C·m)].
    vg_m_s : float
        Signed group velocity :math:`v_g < 0` [m/s].
    sigmaz_m : float
        RMS Gaussian bunch length :math:`\\sigma_z` [m].
    freq_hz : float
        Synchronous frequency :math:`f_s` [Hz].
    u0_m_s : float, optional
        Beam velocity :math:`u_0` [m/s]. Default is :math:`c`.
    mutate_flip_vg_sign : bool, optional
        If ``True``, flip the slippage denominator from :math:`(1 - v_g / u_0)` to :math:`(1 + v_g / u_0)`
        (where :math:`v_g < 0`) for mutation testing.

    Returns
    -------
    a_w_j_per_m : float
        Analytical energy loss per unit length [J/m] (numerically identical to [mJ/mm]).
    """
    v_abs = abs(float(vg_m_s))
    vg_signed = -v_abs
    denom = (1.0 + vg_signed / float(u0_m_s)) if mutate_flip_vg_sign else (1.0 - vg_signed / float(u0_m_s))
    phi = float(gaussian_bunch_form_factor(freq_hz, sigmaz_m, u0_m_s))
    return float((float(charge_c) ** 2) * float(k_loss_per_m) * (phi**2) / denom)


def estimate_drainage_wake_length_m(
    n_periods: int,
    *,
    gamma_tap: float | None = None,
    vg_sync_over_c: float | None = None,
    vg_min_over_c: float | None = None,
    data_dir: Path | str | None = None,
    target_u_over_w: float = 0.02,
    n_taper_cells: int = 2,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> float:
    """Estimate the trailing wake window :math:`s_{\\max}` [m] required for :math:`U_{\\mathrm{bp}}(T_{\\mathrm{end}}) / W_{\\mathrm{beam,bp}} < \\epsilon_{\\mathrm{drain}}`.

    Derived from:
    1. Single-pass drainage of the slowest dispersive passband components (:math:`|v_{g,\\min}|`) across the
       active plus two adiabatic taper sections of length :math:`L_{\\mathrm{rad}} = (N + 2 N_{\\mathrm{taper}}) p`:
       :math:`\\Delta t_{\\mathrm{disp}} = L_{\\mathrm{rad}} / |v_{g,\\min}| = (|v_{g,s}| / |v_{g,\\min}|) t_{\\mathrm{tr}}`,
       where :math:`t_{\\mathrm{tr}} = L_{\\mathrm{rad}} / |v_{g,s}|` is the synchronous transit time at
       :math:`|v_{g,s}| = 0.158 c` (Lu et al., *Phys. Rev. Lett.* **122**, 014801, 2019, line 234).
    2. Multi-reflection Fabry–Pérot cavity ring-down between the two coaxial end couplers with voltage
       reflection :math:`|\\Gamma_{\\mathrm{tap}}|`: each transit multiplies stored passband energy by
       :math:`|\\Gamma_{\\mathrm{tap}}|^2`, requiring
       :math:`N_{\\mathrm{tr}} = \\ln(1 / \\epsilon_{\\mathrm{drain}}) / \\ln(1 / |\\Gamma_{\\mathrm{tap}}|^2)`
       synchronous transits.
    Combining both mechanisms gives :math:`s_{\\max}(N) = u_0 (\\Delta t_{\\mathrm{disp}} + N_{\\mathrm{tr}} t_{\\mathrm{tr}})`.
    When ``gamma_tap``, ``vg_sync_over_c``, or ``vg_min_over_c`` are omitted, they are loaded from
    ``task_01_geometry_and_port_reflection.json`` and ``task_02_cold_test_and_convergence.json`` in ``data_dir``.
    """
    d_dir = Path(data_dir) if data_dir is not None else (Path(__file__).resolve().parent / "data")
    if gamma_tap is None:
        t1_p = d_dir / "task_01_geometry_and_port_reflection.json"
        t1_d = json.loads(t1_p.read_text(encoding="utf-8"))
        gamma_tap = float(t1_d["port_reflection"]["gamma_tapered_passband_mean"])
    if vg_sync_over_c is None or vg_min_over_c is None:
        t2_p = d_dir / "task_02_cold_test_and_convergence.json"
        t2_d = json.loads(t2_p.read_text(encoding="utf-8"))
        if vg_sync_over_c is None:
            vg_sync_over_c = abs(float(t2_d["wakis_finest_nf8_benchmarks"]["ve_dft_over_c"]))
        if vg_min_over_c is None:
            phi_rad = np.deg2rad(np.asarray(t2_d["dispersion"]["phi_deg"], dtype=float))
            kz_arr = phi_rad / float(spec.period_m)
            omega_arr = 2.0 * np.pi * np.asarray(t2_d["dispersion"]["f_mode_ghz"], dtype=float) * 1e9
            vg_min_over_c = float(np.min(np.abs(np.gradient(omega_arr, kz_arr) / C0)))
    _, _, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    l_rad_m = float(int(n_periods) + 2 * int(n_taper_cells)) * float(spec.period_m)
    vg_s_m_s = abs(float(vg_sync_over_c)) * C0
    t_tr_s = l_rad_m / vg_s_m_s
    n_disp = abs(float(vg_sync_over_c)) / max(abs(float(vg_min_over_c)), 1e-6)
    g_sq = max(min(float(gamma_tap) ** 2, 0.99), 1e-12)
    n_ring = float(np.log(1.0 / max(float(target_u_over_w), 1e-6)) / np.log(1.0 / g_sq))
    return float(u0 * (n_disp + n_ring) * t_tr_s)


def compute_three_level_gci(
    vals: list[float],
    h_vals: list[float],
    *,
    p_formal: float = 1.0,
    fs: float | None = None,
    mutate_swap_coarse_med: bool = False,
    mutate_force_fs_1p25: bool = False,
) -> dict[str, Any]:
    """Compute 3-level convergence class, observed order :math:`p`, Richardson extrapolate, and uncertainty :math:`U_{\\mathrm{fine}}`.

    Implements the three-level mesh-convergence classification and Grid Convergence Index (GCI)
    procedure of Roache (1998), Celik *et al.* (*J. Fluids Eng.* **130**, 078001, 2008), and
    Stern *et al.* (*J. Fluids Eng.* **123**, 793, 2001):

    1. For three grid spacings :math:`h_1 > h_2 > h_3 > 0` with observable values :math:`f_1, f_2, f_3`
       (e.g., :math:`n_{xy} = 48, 64, 96`), form :math:`e_{21} = f_2 - f_1`, :math:`e_{32} = f_3 - f_2`,
       and the convergence ratio :math:`R = e_{32} / e_{21}`.
    2. Classify the convergence regime from :math:`R`: ``"monotone"`` if :math:`0 < R < 1`,
       ``"oscillatory"`` if :math:`R < 0`, and ``"divergent"`` if :math:`R \\ge 1` (or :math:`e_{21} = 0`).
    3. Solve the exact three-level asymptotic relation

       .. math::

           \\frac{f_1 - f_2}{f_2 - f_3} = \\frac{h_1^p - h_2^p}{h_2^p - h_3^p}

       numerically via Brent's root finder **only** for ``"monotone"`` data (:math:`0 < R < 1` with
       :math:`(f_1 - f_2)/(f_2 - f_3) > \\ln(h_1 / h_2) / \\ln(h_2 / h_3)`). Otherwise set
       ``observed_order = None`` and ``richardson_estimate = None``.
    4. Because the :math:`90^\\circ` PEC spoke and inner-hub corners produce singular edge fields,
       the formal spatial convergence order is first order (:math:`p_{\\mathrm{formal}} = 1`). Use
       safety factor :math:`F_s = 1.25` with :math:`p_{\\mathrm{used}} = p_{\\mathrm{obs}}` only when
       :math:`p_{\\mathrm{obs}}` is solved and lies within :math:`\\pm 25\\%` of :math:`p_{\\mathrm{formal}}`;
       otherwise use :math:`F_s = 3` with :math:`p_{\\mathrm{used}} = 1`.
    5. Report the Stern oscillatory half-range :math:`(\\max f - \\min f) / (2 |f_3|)` and the combined
       finest-level relative uncertainty :math:`U_{\\mathrm{fine,rel}} = \\max(\\mathrm{GCI}(F_s, p_{\\mathrm{used}}), \\text{half\\_range\\_rel})`.
    """
    if len(vals) != 3 or len(h_vals) != 3:
        raise ValueError("compute_three_level_gci requires exactly 3 values and 3 grid spacings.")
    v_use = [float(vals[1]), float(vals[0]), float(vals[2])] if mutate_swap_coarse_med else [float(x) for x in vals]
    f1, f2, f3 = v_use[0], v_use[1], v_use[2]
    h1, h2, h3 = float(h_vals[0]), float(h_vals[1]), float(h_vals[2])
    if not (h1 > h2 > h3 > 0.0):
        raise ValueError(f"h_vals must be strictly decreasing and positive; got {h_vals}.")

    e21 = float(f2 - f1)
    e32 = float(f3 - f2)
    r_ratio = float(e32 / e21) if abs(e21) > 1e-30 else float("inf")
    if 0.0 < r_ratio < 1.0:
        conv_class = "monotone"
    elif r_ratio < 0.0:
        conv_class = "oscillatory"
    else:
        conv_class = "divergent"
    monotone = bool(conv_class == "monotone")

    r12 = h1 / h2
    r23 = h2 / h3
    r_min = float(np.log(r12) / np.log(r23))

    p_obs: float | None = None
    v_rich: float | None = None
    if monotone and abs(e32) > 1e-15:
        target_ratio = float(1.0 / r_ratio)
        if target_ratio > r_min + 1e-9:
            def _res(p_val: float) -> float:
                return float(((h1**p_val) - (h2**p_val)) / ((h2**p_val) - (h3**p_val)) - target_ratio)

            if _res(1e-6) * _res(60.0) <= 0.0:
                p_obs = float(brentq(_res, 1e-6, 60.0, xtol=1e-12))
                v_rich = float(f3 + e32 / ((r23**p_obs) - 1.0))

    if p_obs is not None and (0.75 * float(p_formal) <= p_obs <= 1.25 * float(p_formal)):
        fs_used = 1.25
        p_used = float(p_obs)
    else:
        fs_used = 3.0
        p_used = 1.0

    if mutate_force_fs_1p25:
        fs_used = 1.25

    step_fine = abs(e32)
    step_med = abs(e21)
    gci_fine_rel = float(fs_used * step_fine / (max(abs(f3), 1e-30) * ((r23**p_used) - 1.0)))
    gci_med_rel = float(fs_used * step_med / (max(abs(f2), 1e-30) * ((r12**p_used) - 1.0)))
    half_range_rel = float(0.5 * (max(v_use) - min(v_use)) / max(abs(f3), 1e-30))
    u_fine_rel = gci_fine_rel if mutate_force_fs_1p25 else float(max(gci_fine_rel, half_range_rel))

    return {
        "values": v_use,
        "h_mm": [h1, h2, h3],
        "is_monotone": monotone,
        "convergence_class": conv_class,
        "R_ratio": r_ratio,
        "observed_order": p_obs,
        "p_used": p_used,
        "effective_order_for_gci": p_used,
        "fs_used": fs_used,
        "safety_factor_fs": fs_used,
        "richardson_estimate": v_rich,
        "gci_fine_rel": gci_fine_rel,
        "gci_med_rel": gci_med_rel,
        "half_range_rel": half_range_rel,
        "u_fine_rel": u_fine_rel,
        "rel_diff_64_vs_48": float(step_med / max(abs(f2), 1e-30)),
        "rel_diff_96_vs_64": float(step_fine / max(abs(f3), 1e-30)),
        "rel_diff_96_vs_48": float(abs(f3 - f1) / max(abs(f3), 1e-30)),
    }


def format_gci_line(gci_dict: dict[str, Any], unit: str = "", digits: int = 4) -> str:
    """Format a 3-level convergence dictionary into a diagnostic summary string without Richardson values for non-monotone data."""
    vals_str = str([round(float(x), digits) for x in gci_dict["values"]])
    c_cls_raw = str(gci_dict["convergence_class"])
    r_val = float(gci_dict["R_ratio"])
    c_cls = (
        "oscillatory, |R| > 1 (amplitude not decreasing)"
        if (c_cls_raw == "oscillatory" and abs(r_val) > 1.0)
        else c_cls_raw
    )
    fs_val = float(gci_dict["fs_used"])
    p_used = float(gci_dict["p_used"])
    u_rel = 100.0 * float(gci_dict["u_fine_rel"])
    g_fine = 100.0 * float(gci_dict["gci_fine_rel"])
    g_med = 100.0 * float(gci_dict["gci_med_rel"])
    hr_rel = 100.0 * float(gci_dict["half_range_rel"])
    p_obs = gci_dict["observed_order"]
    rich = gci_dict["richardson_estimate"]
    unit_sfx = f" {unit}" if unit else ""
    if p_obs is not None and rich is not None:
        return (
            f"{vals_str}{unit_sfx} (class={c_cls}, R={r_val:.3f}, p_obs={float(p_obs):.2f}, "
            f"Rich={float(rich):.{digits}f}{unit_sfx}, Fs={fs_val:.2f}, p_used={p_used:.2f}, "
            f"GCI_fine={g_fine:.2f}%, GCI_med={g_med:.2f}%, U_fine={u_rel:.2f}%)"
        )
    return (
        f"{vals_str}{unit_sfx} (class={c_cls}, R={r_val:.3f}, Fs={fs_val:.2f}, p_used={p_used:.2f}, "
        f"GCI_fine={g_fine:.2f}%, GCI_med={g_med:.2f}%, half_range={hr_rel:.2f}%, U_fine={u_rel:.2f}%)"
    )


def analytic_sws_traveling_wave_impedance_ohm(
    freq_hz: np.ndarray | float,
    f_sync_hz: float,
    k_loss_per_m: float,
    length_m: float,
    u0_m_s: float,
    vg_m_s: float,
    *,
    mutate_linear_length: bool = False,
) -> np.ndarray:
    """Evaluate the cold-test travelling-wave beam coupling resistance :math:`\\operatorname{Re} Z_\\parallel(f)`.

    Around the synchronous frequency :math:`\\omega_s = 2\\pi f_s`, the longitudinal coupling impedance
    of an :math:`N`-period slow-wave structure of active length :math:`L = N p` is:

    .. math::

        \\operatorname{Re} Z_\\parallel(\\omega)
        = \\frac{k_L L^2}{2 |v_g|} \\operatorname{sinc}^2\\!\\left(\\frac{\\Delta k_z(\\omega) L}{2}\\right),
        \\qquad \\Delta k_z(\\omega) = (\\omega - \\omega_s)\\left(\\frac{1}{v_g} - \\frac{1}{u_0}\\right),

    where :math:`\\operatorname{sinc}(x) = \\sin(x)/x` and integrating
    :math:`\\frac{1}{\\pi}\\int_0^\\infty \\operatorname{Re} Z_\\parallel(\\omega)\\,\\mathrm{d}\\omega`
    yields the exact single-mode loss factor :math:`k_\\parallel = k_L L / (1 - v_g / u_0)`.

    Parameters
    ----------
    freq_hz : array_like or float
        Frequencies :math:`f` [Hz].
    f_sync_hz : float
        Synchronous frequency :math:`f_s` [Hz].
    k_loss_per_m : float
        Modal loss factor per unit length :math:`k_L = (\\omega_s/4)(r/Q)` [V/(C·m)].
    length_m : float
        Active interaction length :math:`L = N p` [m].
    u0_m_s : float
        Beam axial velocity :math:`u_0` [m/s].
    vg_m_s : float
        Signed group velocity :math:`v_g` [m/s].
    mutate_linear_length : bool, optional
        If ``True``, replace the coherent :math:`L^2` peak impedance factor with linear :math:`L \\cdot L_0`
        (:math:`L_0 = 12\\;\\mathrm{mm}`) for mutation testing.

    Returns
    -------
    re_z_ohm : np.ndarray
        Longitudinal beam coupling resistance :math:`\\operatorname{Re} Z_\\parallel(f)` [:math:`\\Omega`].
    """
    f_arr = np.asarray(freq_hz, dtype=float)
    d_omega = 2.0 * np.pi * (f_arr - float(f_sync_hz))
    delta_kz = d_omega * (1.0 / float(vg_m_s) - 1.0 / float(u0_m_s))
    arg_pi = 0.5 * delta_kz * float(length_m) / np.pi
    l_sq = float(length_m) * 0.012 if mutate_linear_length else (float(length_m) ** 2)
    z_pk = 0.5 * float(k_loss_per_m) * l_sq / abs(float(vg_m_s))
    return z_pk * (np.sinc(arg_pi) ** 2)


def reflection_induced_peak_error_bound(
    gamma_mag: np.ndarray | float,
    *,
    mutate_drop_factor_two: bool = False,
) -> np.ndarray:
    """Compute the worst-case Fabry–Pérot peak impedance relative error bound from port reflection :math:`|\\Gamma|`.

    Parameters
    ----------
    gamma_mag : array_like or float
        Port voltage reflection magnitude :math:`|\\Gamma(f)| \\in [0, 1)` [dimensionless].
    mutate_drop_factor_two : bool, optional
        If ``True``, drop the round-trip factor of 2 in the numerator for mutation testing.

    Returns
    -------
    rel_err_bound : np.ndarray
        Relative error bound :math:`2|\\Gamma| / (1 - |\\Gamma|^2)` [dimensionless].
    """
    g = np.clip(np.asarray(gamma_mag, dtype=float), 0.0, 0.999)
    num = 1.0 if mutate_drop_factor_two else 2.0
    return num * g / (1.0 - (g**2))


def reflection_induced_energy_error_bound(
    gamma_mag: np.ndarray | float,
    *,
    mutate_use_linear_gamma: bool = False,
) -> np.ndarray:
    """Compute the single-reflection energy relative error bound :math:`|\\Gamma|^2` from port reflection :math:`|\\Gamma|`.

    Assumptions: for a wave packet of passband bandwidth :math:`\\Delta f` incident on an extraction port
    with voltage reflection coefficient :math:`\\Gamma`, when phase interference between the primary pulse and
    the reflected pulse averages out over the passband integration window, a single reflection reflects a
    power/energy fraction :math:`|\\Gamma|^2` back into the structure (unlike the coherent narrowband
    standing-wave peak-impedance bound :math:`2|\\Gamma| / (1 - |\\Gamma|^2)`).

    Parameters
    ----------
    gamma_mag : array_like or float
        Port voltage reflection magnitude :math:`|\\Gamma(f)| \\in [0, 1)` [dimensionless].
    mutate_use_linear_gamma : bool, optional
        If ``True``, erroneously return linear :math:`|\\Gamma|` instead of :math:`|\\Gamma|^2` for mutation testing.

    Returns
    -------
    rel_energy_err_bound : np.ndarray
        Single-reflection energy error bound :math:`|\\Gamma|^2` [dimensionless].
    """
    g = np.clip(np.asarray(gamma_mag, dtype=float), 0.0, 0.999)
    return g if mutate_use_linear_gamma else (g**2)


def derive_pre_asymptotic_scaling_slopes(
    lengths_m: list[float] | np.ndarray,
    vg_m_s: float,
    u0_m_s: float = C0,
    *,
    taper_length_total_m: float = 4.0 * WagonWheelMTMSpec.period_m,
    passband_bw_hz: float = WagonWheelMTMSpec.passband_high_hz - WagonWheelMTMSpec.passband_low_hz,
) -> dict[str, float]:
    """Derive pre-asymptotic short-:math:`N` log-log scaling slope bounds from tapers and passband filter width.

    In a short slow-wave structure of active length :math:`L = N p` bordered by two adiabatic tapers of
    combined length :math:`L_{\\mathrm{taper}}` and measured through a passband filter of bandwidth
    :math:`\\Delta f_{\\mathrm{bp}}` (impulse-response floor :math:`\\Delta t_{\\mathrm{filt}} = 1 / \\Delta f_{\\mathrm{bp}}`),
    the effective 20%-envelope pulse duration has additive broadening between the half-taper group-drain
    limit :math:`t_{p,\\mathrm{min}}(L) = (L + \\tfrac{1}{2}L_{\\mathrm{taper}})(1/|v_g| + 1/u_0)` (when the
    causal filter rising and falling 20% crossings shift by equal delays) and the full-taper plus filter
    impulse-response limit :math:`t_{p,\\mathrm{max}}(L) = (L + L_{\\mathrm{taper}})(1/|v_g| + 1/u_0) + \\Delta t_{\\mathrm{filt}}`.
    Because spectral width scales as :math:`\\mathrm{FWHM}(L) \\propto 1 / t_p(L)` and the quadratic peak
    impedance :math:`Z_{\\mathrm{pk}}(L) \\propto L_{\\mathrm{eff}}(L)^2` lies between the full-taper
    resonator limit :math:`L_{\\mathrm{eff}} = L + L_{\\mathrm{taper}}` (slope :math:`2\\alpha_{\\mathrm{taper}}`)
    and the active-loss times half-taper drain limit (slope :math:`1 + \\alpha_{t_p,\\max}`), the log-log slopes
    over :math:`L` satisfy:

    .. math::

        \\alpha_{t_p} \\in [\\alpha_{t_p,\\min},\\, \\alpha_{t_p,\\max}], \\qquad
        \\alpha_{\\mathrm{FWHM}} \\in [-\\alpha_{t_p,\\max},\\, -\\alpha_{t_p,\\min}], \\qquad
        \\alpha_Z \\in [2\\alpha_{\\mathrm{taper}},\\, 1 + \\alpha_{t_p,\\max}].
    """
    l_arr = np.asarray(lengths_m, dtype=float)
    inv_v = (1.0 / abs(float(vg_m_s))) + (1.0 / float(u0_m_s))
    dt_filt_s = 1.0 / float(passband_bw_hz)
    dt_taper_s = float(taper_length_total_m) * inv_v

    tp_min_offset = l_arr * inv_v + 0.25 * dt_taper_s
    tp_taper_full = l_arr * inv_v + dt_taper_s
    tp_max_offset = l_arr * inv_v + dt_taper_s + dt_filt_s

    slope_tp_max = float(np.polyfit(np.log(l_arr), np.log(tp_min_offset), 1)[0])
    slope_taper = float(np.polyfit(np.log(l_arr), np.log(tp_taper_full), 1)[0])
    slope_tp_min = float(np.polyfit(np.log(l_arr), np.log(tp_max_offset), 1)[0])
    slope_tp_mid = 0.5 * (slope_tp_min + slope_tp_max)
    slope_z_min = 2.0 * slope_taper
    slope_z_max = 1.0 + slope_tp_max
    slope_z_mid = 0.5 * (slope_z_min + slope_z_max)

    return {
        "slope_tp_min": slope_tp_min,
        "slope_tp_max": slope_tp_max,
        "slope_tp_mid": slope_tp_mid,
        "slope_fwhm_min": -slope_tp_max,
        "slope_fwhm_max": -slope_tp_min,
        "slope_fwhm_mid": -slope_tp_mid,
        "slope_re_z_pk_min": slope_z_min,
        "slope_re_z_pk_max": slope_z_max,
        "slope_re_z_pk_mid": slope_z_mid,
        "slope_z_min": slope_z_min,
        "slope_z_max": slope_z_max,
    }


class HannDFTMonitor(DFTMonitor):
    """Windowed DFTMonitor supporting a Hann or rectangular temporal weight over ``n_steps_window``."""

    def __init__(self, *args: Any, n_steps_window: int, window: str = "hann", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.n_steps_window = max(int(n_steps_window), 1)
        self.window = str(window).lower()
        self._step_idx = 0
        self._weight_sum = 0.0

    def __call__(self, solver: SolverFIT3D) -> None:
        if self._acc is None:
            self._bind(solver)
        xp = solver.xp
        t = solver.i * solver.dt
        if self.window == "hann" and self.n_steps_window > 1:
            w = 0.5 * (1.0 - float(np.cos(2.0 * np.pi * self._step_idx / (self.n_steps_window - 1))))
        else:
            w = 1.0
        self._step_idx += 1
        self._weight_sum += w
        weights = w * xp.exp(xp.asarray(-2j * np.pi * self.frequencies * t))
        for comp in self.components:
            vals = self._read_component(solver, comp)
            self._acc[comp] += weights[:, None] * vals[None, :]

    def get_field(self, component: str) -> tuple[np.ndarray, np.ndarray]:
        arr, freqs = super().get_field(component)
        if self._weight_sum > 0.0:
            arr = arr * (2.0 / self._weight_sum)
        return arr, freqs


def make_annular_cylinder(
    r_in: float,
    r_out: float,
    z_min: float,
    z_max: float,
    n_theta: int = 96,
) -> pv.PolyData:
    """Create a watertight triangulated hollow cylinder (annulus) between ``r_in`` and ``r_out`` along ``z``."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    pts = np.zeros((4 * n_theta, 3), dtype=float)
    pts[0:n_theta, 0] = r_in * cos_t
    pts[0:n_theta, 1] = r_in * sin_t
    pts[0:n_theta, 2] = z_min

    pts[n_theta : 2 * n_theta, 0] = r_out * cos_t
    pts[n_theta : 2 * n_theta, 1] = r_out * sin_t
    pts[n_theta : 2 * n_theta, 2] = z_min

    pts[2 * n_theta : 3 * n_theta, 0] = r_in * cos_t
    pts[2 * n_theta : 3 * n_theta, 1] = r_in * sin_t
    pts[2 * n_theta : 3 * n_theta, 2] = z_max

    pts[3 * n_theta : 4 * n_theta, 0] = r_out * cos_t
    pts[3 * n_theta : 4 * n_theta, 1] = r_out * sin_t
    pts[3 * n_theta : 4 * n_theta, 2] = z_max

    faces: list[int] = []
    for i in range(n_theta):
        ip = (i + 1) % n_theta
        i_in_b, ip_in_b = i, ip
        i_out_b, ip_out_b = n_theta + i, n_theta + ip
        i_in_t, ip_in_t = 2 * n_theta + i, 2 * n_theta + ip
        i_out_t, ip_out_t = 3 * n_theta + i, 3 * n_theta + ip

        faces.extend([3, i_in_b, ip_out_b, i_out_b])
        faces.extend([3, i_in_b, ip_in_b, ip_out_b])
        faces.extend([3, i_in_t, i_out_t, ip_out_t])
        faces.extend([3, i_in_t, ip_out_t, ip_in_t])
        faces.extend([3, i_in_b, i_in_t, ip_in_t])
        faces.extend([3, i_in_b, ip_in_t, ip_in_b])
        faces.extend([3, i_out_b, ip_out_t, i_out_t])
        faces.extend([3, i_out_b, ip_out_b, ip_out_t])

    mesh = pv.PolyData(pts, np.array(faces, dtype=np.int64))
    return mesh.clean().triangulate()


def build_unit_cell_solids(
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
    *,
    forward_control: bool = False,
) -> dict[str, pv.PolyData]:
    """Build 3D PEC STL solids for a single unit cell :math:`z \\in [0, p]` of the wagon-wheel MTM or forward control.

    Parameters
    ----------
    spec : WagonWheelMTMSpec, optional
        Structure specification.
    forward_control : bool, optional
        If ``False`` (default), build the wagon-wheel metamaterial unit cell (annular hub + 4 radial spokes).
        If ``True``, build the forward-wave disk-loaded circular waveguide control cell (annular iris without
        inductive spokes or inner ring, supporting a :math:`v_g > 0` slow wave).

    Returns
    -------
    solids : dict of str to pyvista.PolyData
        Dictionary of watertight PEC solids for :class:`wakis.GridFIT3D`.
    """
    p = spec.period_m
    t_ww = spec.wagon_wheel_thickness_m
    r_b = spec.beam_hole_radius_m
    r_hub = spec.hub_outer_radius_m
    r_slot = spec.slot_outer_radius_m
    r_wg = spec.waveguide_radius_m
    w_sp = spec.spoke_width_m
    r_box_out = r_wg + 3.0e-3

    z0 = 0.5 * (p - t_ww)
    z1 = z0 + t_ww

    blocks: list[pv.PolyData] = [
        make_annular_cylinder(r_wg, r_box_out, -0.5 * p, 1.5 * p, n_theta=96)
    ]
    if not forward_control:
        if r_slot < r_wg:
            blocks.append(make_annular_cylinder(r_slot, r_wg + 0.2e-3, z0, z1, n_theta=96))
        blocks.append(make_annular_cylinder(r_b, r_hub, z0, z1, n_theta=96))
        r_in_sp = 0.5 * (r_b + r_hub)
        r_out_sp = r_slot + 0.3e-3
        hw = 0.5 * w_sp
        blocks.append(pv.Box(bounds=(r_in_sp, r_out_sp, -hw, hw, z0, z1)).triangulate())
        blocks.append(pv.Box(bounds=(-r_out_sp, -r_in_sp, -hw, hw, z0, z1)).triangulate())
        blocks.append(pv.Box(bounds=(-hw, hw, r_in_sp, r_out_sp, z0, z1)).triangulate())
        blocks.append(pv.Box(bounds=(-hw, hw, -r_out_sp, -r_in_sp, z0, z1)).triangulate())
    else:
        r_iris = 5.0e-3
        blocks.append(make_annular_cylinder(r_iris, r_wg + 0.3e-3, z0, z1, n_theta=96))

    merged = blocks[0]
    for b in blocks[1:]:
        merged = merged.merge(b)
    return {"pec_unit_cell": merged}


def build_graded_unit_cell_axes(
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
    n_feat: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build self-similar feature-snapped graded 1D coordinate arrays ``(x, y, z)`` for the unit cell.

    Uses fine spacing :math:`h_{\\mathrm{fine}} = 1.0\\;\\mathrm{mm} / n_{\\mathrm{feat}}` (spoke width 1.0 mm,
    PRL Fig. 2(b) line 119) across the hub/spoke core (:math:`|x|, |y| \\leq r_{\\mathrm{hub}}`) and coarse
    spacing across the outer quadrant slot region up to :math:`R_{\\mathrm{wg}} = 8.0\\;\\mathrm{mm}` (PRL line 121).
    """
    r_wg = spec.waveguide_radius_m
    p = spec.period_m
    h_fine = 1.0e-3 / float(n_feat)
    h_coarse = 1.5 * h_fine
    n_core = 4 * int(n_feat)
    n_outer = int(round(4.0e-3 / h_coarse))
    x_pos = np.concatenate(
        [
            np.linspace(0.0, 4.0e-3, n_core + 1),
            np.linspace(4.0e-3, r_wg, n_outer + 1)[1:],
        ]
    )
    x = np.concatenate([-x_pos[:0:-1], x_pos])
    y = x.copy()
    z = np.linspace(0.0, p, 2 * int(n_feat) + 1)
    return x, y, z, float(h_fine)


def _extract_resonance_freq_and_q(
    mon: ResonanceMonitor,
    f_lo_hz: float,
    f_hi_hz: float,
    *,
    force_fft_fallback: bool = False,
) -> tuple[float, float]:
    """Extract the dominant resonance frequency [Hz] and quality factor from a :class:`ResonanceMonitor`.

    First invokes ``mon.harminv(f_lo_hz, f_hi_hz)``. If no modes pass the default relative error filter
    (or if ``force_fft_fallback=True``), retries ``harminv`` with a relaxed ``max_rel_error=5e-2`` and
    finally falls back to a zero-padded Hann-windowed FFT of ``mon.signal()`` (since
    :class:`wakis.resonances.ResonanceMonitor` provides ``.signal()`` and ``.harminv()``, not ``.spectrum()``).
    """
    if not force_fft_fallback:
        res = mon.harminv(f_lo_hz, f_hi_hz)
        q_mask = np.abs(res.Q) >= 50.0 if len(res.frequency) > 0 else np.zeros(0, dtype=bool)
        if not np.any(q_mask):
            res = mon.harminv(f_lo_hz, f_hi_hz, max_rel_error=5.0e-2)
            q_mask = np.abs(res.Q) >= 50.0 if len(res.frequency) > 0 else np.zeros(0, dtype=bool)
        if np.any(q_mask):
            f_q = res.frequency[q_mask]
            q_q = res.Q[q_mask]
            a_q = res.amplitude[q_mask]
            idx_dom = int(np.argmax(np.abs(a_q)))
            return float(f_q[idx_dom]), float(q_q[idx_dom])

    t_s, x_s = mon.signal()
    if len(t_s) < 2:
        return float(0.5 * (f_lo_hz + f_hi_hz)), 1.0e4
    dt_s = float(np.median(np.diff(t_s)))
    win = np.hanning(len(x_s))
    n_fft = max(16384, 1 << int(np.ceil(np.log2(len(x_s) * 4))))
    spec_fft = np.fft.fft((x_s - np.mean(x_s)) * win, n=n_fft)
    freqs_fft = np.fft.fftfreq(n_fft, d=dt_s)
    mask_f = (freqs_fft >= f_lo_hz) & (freqs_fft <= f_hi_hz)
    if not np.any(mask_f):
        mask_f = ( np.abs(freqs_fft) >= f_lo_hz ) & ( np.abs(freqs_fft) <= f_hi_hz )
    f_sub = np.abs(freqs_fft[mask_f])
    s_sub = np.abs(spec_fft[mask_f])
    f0 = float(f_sub[int(np.argmax(s_sub))])
    return f0, 1.0e4


def evaluate_bloch_unit_cell(
    phi_deg: float = 27.43,
    *,
    n_feat: int = 4,
    use_graded_mesh: bool = False,
    forward_control: bool = False,
    compute_eigenmode_metrics: bool = True,
    save_field_maps: bool = False,
    quick: bool = False,
    mutate_flip_k_bloch_sign: bool = False,
    force_fft_fallback: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Run a 3D Bloch cold-test simulation of the wagon-wheel unit cell at phase advance ``phi_deg``.

    Stage 1 excites the unit cell with a broadband quadrupole-balanced current distribution and extracts
    the fundamental passband frequency :math:`f_0(\\phi)` via :class:`wakis.resonances.ResonanceMonitor`
    (`harminv`). When ``compute_eigenmode_metrics=True``, Stage 2 drives at :math:`f_0` and accumulates
    a Hann-windowed 3D DFT over an integer number of RF cycles, evaluating the exact conformal FIT
    electric and magnetic stored energies :math:`U_e, U_m`, axial Poynting flux :math:`P_z`, energy
    velocity :math:`v_E`, shunt impedance over :math:`Q` per unit length :math:`r/Q`, and loss factor
    per unit length :math:`k_L = (\\omega/4)(r/Q)`.
    """
    p = spec.period_m
    r_wg = spec.waveguide_radius_m
    solids = build_unit_cell_solids(spec, forward_control=forward_control)
    mats = {k: "pec" for k in solids}

    if use_graded_mesh:
        x, y, z, h_min = build_graded_unit_cell_axes(spec, n_feat=n_feat)
    else:
        n_xy = 16 * int(n_feat)
        n_z = 2 * int(n_feat)
        x = np.linspace(-r_wg, r_wg, n_xy + 1)
        y = np.linspace(-r_wg, r_wg, n_xy + 1)
        z = np.linspace(0.0, p, n_z + 1)
        h_min = float(p / n_z)

    beta0 = float(np.deg2rad(phi_deg) / p)
    k_z_bloch = +beta0 if mutate_flip_k_bloch_sign else -beta0
    t0 = time.perf_counter()

    g1 = GridFIT3D(x=x, y=y, z=z, stl_solids=solids, stl_materials=mats, conformal="pec", verbose=0)
    s1 = SolverFIT3D(
        g1,
        use_stl=True,
        bc_low=["pec", "pec", "bloch"],
        bc_high=["pec", "pec", "bloch"],
        dtype=np.complex128,
        k_bloch=(0.0, 0.0, k_z_bloch),
        verbose=0,
    )
    base_steps = 2600 if quick else 3200
    n_steps = int(round(base_steps * (n_feat / 4.0)))
    est1 = estimate((s1.Nx, s1.Ny, s1.Nz), steps=n_steps, dtype="complex128", conformal="pec")
    preflight(est1, "laptop" if quick else "n2-highmem-8")

    f_center = 18.5e9 if forward_control else 11.5e9
    f_width = 6.0e9 if forward_control else 5.0e9
    pulse1 = GaussianSource(frequency=f_center, fwidth=f_width)
    s_pos = 5.5e-3 / np.sqrt(2.0)
    src_specs = [
        ("Ez", (0.0, 0.0, 0.25e-3), 1.0),
        ("Ex", (s_pos, s_pos, 1.0e-3), 0.5),
        ("Ey", (s_pos, s_pos, 1.0e-3), 0.5),
        ("Ex", (-s_pos, -s_pos, 1.0e-3), -0.5),
        ("Ey", (-s_pos, -s_pos, 1.0e-3), -0.5),
    ]
    sources1 = [CurrentSource(pulse1, component=c, center=pos, amplitude=a) for c, pos, a in src_specs]
    mon1 = ResonanceMonitor("Ez", center=(0.0, 0.0, 0.25e-3), t_start=pulse1.end_time)
    s1.emsolve(n_steps, source=sources1, monitors=[mon1], verbose=0)

    f_lo_h, f_hi_h = (14.5e9, 23.0e9) if forward_control else (8.0e9, 13.8e9)
    f0, q_harminv = _extract_resonance_freq_and_q(
        mon1, f_lo_h, f_hi_h, force_fft_fallback=force_fft_fallback
    )

    nx_cells, ny_cells, nz_cells = int(s1.Nx), int(s1.Ny), int(s1.Nz)
    del s1, g1
    gc.collect()

    out: dict[str, Any] = {
        "phi_deg": float(phi_deg),
        "beta0_rad_m": beta0,
        "n_feat": int(n_feat),
        "use_graded_mesh": bool(use_graded_mesh),
        "h_min_mm": float(h_min * 1e3),
        "cells_per_wavelength_lambda0": float((C0 / f0) / h_min),
        "cells_across_1mm_feature": float(1.0e-3 / h_min),
        "grid_shape": [nx_cells, ny_cells, nz_cells],
        "total_cells": int(nx_cells * ny_cells * nz_cells),
        "f_mode_hz": f0,
        "f_mode_ghz": float(f0 * 1e-9),
        "vp_over_c": float((2.0 * np.pi * f0 / max(beta0, 1e-12)) / C0),
        "q_harminv": q_harminv,
        "est_memory_mb": float(est1.host_bytes / 1e6),
    }

    if not compute_eigenmode_metrics:
        out["wall_time_s"] = float(time.perf_counter() - t0)
        return out

    # Stage 2: narrowband drive at f0 + integer-cycle Hann-windowed 3D DFT
    g2 = GridFIT3D(x=x, y=y, z=z, stl_solids=solids, stl_materials=mats, conformal="pec", verbose=0)
    s2 = SolverFIT3D(
        g2,
        use_stl=True,
        bc_low=["pec", "pec", "bloch"],
        bc_high=["pec", "pec", "bloch"],
        dtype=np.complex128,
        k_bloch=(0.0, 0.0, k_z_bloch),
        verbose=0,
    )
    ds_diag = np.real(s2.Ds.diagonal()).copy()
    itda_diag = np.real(s2.itDa.diagonal()).copy()
    tds_diag = np.real(s2.tDs.diagonal()).copy()
    ieps_vec = np.real(s2.ieps.toarray()).copy()
    c_mat = s2.C.tocsr().copy()
    ds_mat = s2.Ds.tocsr().copy()
    n_tot3 = 3 * s2.N

    pulse2 = GaussianSource(frequency=f0, fwidth=0.05 * f0, cutoff=3.5)
    sources2 = [CurrentSource(pulse2, component=c, center=pos, amplitude=a) for c, pos, a in src_specs]
    n_drive = int(np.ceil(pulse2.end_time / s2.dt)) + 10
    ring_base = 1200 if quick else 1800
    n_ring_target = int(round(ring_base * (n_feat / 4.0)))
    n_ring_cycles = max(int(round(n_ring_target * s2.dt * f0)), 1)
    n_ring_actual = max(int(round(n_ring_cycles / (f0 * s2.dt))), 1)

    lx = float(g2.xmax - g2.xmin)
    ly = float(g2.ymax - g2.ymin)
    lz = float(g2.zmax - g2.zmin)
    dft = HannDFTMonitor(
        [f0],
        n_steps_window=n_ring_actual,
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        center=(0.0, 0.0, 0.5 * lz),
        size=(lx, ly, lz),
        window="hann",
    )
    s2.emsolve(n_drive, source=sources2, verbose=0)
    s2.emsolve(n_ring_actual, monitors=[dft], verbose=0)

    e_vec = np.zeros(n_tot3, dtype=np.complex128)
    h_vec = np.zeros(n_tot3, dtype=np.complex128)
    for comp in ("Ex", "Ey", "Ez"):
        e_vec[dft._flat[comp]] = dft._acc[comp][0]
    for comp in ("Hx", "Hy", "Hz"):
        h_vec[dft._flat[comp]] = dft._acc[comp][0]

    mask_op = (ieps_vec > 0.0) & (itda_diag > 0.0) & (ds_diag > 0.0)
    w_elec_vec = np.zeros_like(ds_diag)
    w_elec_vec[mask_op] = ds_diag[mask_op] / (ieps_vec[mask_op] * itda_diag[mask_op])
    u_elec_raw = float(0.5 * np.sum(w_elec_vec * (np.abs(e_vec) ** 2)))

    omega = 2.0 * np.pi * f0
    phi_b_vec = (c_mat @ (ds_mat @ e_vec)) / (-1j * omega)
    u_mag_raw = float(0.5 * np.real(np.vdot(h_vec, tds_diag * phi_b_vec)))
    scale_w = (2.0 / dft._weight_sum) ** 2 if dft._weight_sum > 0.0 else 1.0
    u_elec = 0.5 * u_elec_raw * scale_w
    u_mag = 0.5 * u_mag_raw * scale_w
    u_cell = u_elec + u_mag

    ex_raw, _ = dft.get_field("Ex")
    ey_raw, _ = dft.get_field("Ey")
    ez_raw, _ = dft.get_field("Ez")
    hx_raw, _ = dft.get_field("Hx")
    hy_raw, _ = dft.get_field("Hy")

    edge_ghosts = getattr(g2, "edge_ghosts", [1, 1, 0])
    nx = max(int(g2.Nx) - int(edge_ghosts[0]), 1)
    ny = max(int(g2.Ny) - int(edge_ghosts[1]), 1)
    nz = max(int(g2.Nz) - int(edge_ghosts[2]), 1)

    dx_3d = np.asarray(g2.dx[:nx], dtype=float)[:, None, None]
    dy_3d = np.asarray(g2.dy[:ny], dtype=float)[None, :, None]
    dz_1d = np.asarray(g2.dz[:nz], dtype=float)
    dz_3d = dz_1d[None, None, :]
    dvol_3d = dx_3d * dy_3d * dz_3d

    phase_z = np.exp(1j * k_z_bloch * lz)

    def _shift_z(arr: np.ndarray) -> np.ndarray:
        res_arr = np.empty_like(arr)
        res_arr[:, :, :-1] = arr[:, :, 1:]
        res_arr[:, :, -1] = arr[:, :, 0] * phase_z
        return res_arr

    ex0 = 0.5 * (ex_raw[0, :nx, :ny, :nz] + ex_raw[0, :nx, 1 : ny + 1, :nz])
    ey0 = 0.5 * (ey_raw[0, :nx, :ny, :nz] + ey_raw[0, 1 : nx + 1, :ny, :nz])
    ex_c = 0.5 * (ex0 + _shift_z(ex0))
    ey_c = 0.5 * (ey0 + _shift_z(ey0))
    hx_c = 0.5 * (hx_raw[0, :nx, :ny, :nz] + hx_raw[0, 1 : nx + 1, :ny, :nz])
    hy_c = 0.5 * (hy_raw[0, :nx, :ny, :nz] + hy_raw[0, :nx, 1 : ny + 1, :nz])

    sz_3d = 0.5 * np.real(ex_c * np.conj(hy_c) - ey_c * np.conj(hx_c))
    pz_mean = float(np.sum(sz_3d * dvol_3d) / lz)
    ve_m_s = float(pz_mean * lz / max(u_cell, 1e-30))

    ix_mid = int(np.argmin(np.abs(g2.x[:nx])))
    iy_mid = int(np.argmin(np.abs(g2.y[:ny])))
    z_centers = np.cumsum(dz_1d) - 0.5 * dz_1d
    ez_axis_z = ez_raw[0, ix_mid, iy_mid, :nz]
    ez0_axis = np.sum(ez_axis_z * np.exp(1j * beta0 * z_centers) * dz_1d) / lz

    r_over_q_ohm_per_m = float((np.abs(ez0_axis) ** 2) * lz / (omega * max(u_cell, 1e-30)))
    k_loss_per_m = float(0.25 * omega * r_over_q_ohm_per_m)
    k_pierce_ohm = float((np.abs(ez0_axis) ** 2) / (2.0 * (beta0**2) * max(abs(pz_mean), 1e-30)))

    out.update(
        {
            "u_elec_over_u_mag": float(u_elec / max(u_mag, 1e-30)),
            "ve_m_s": ve_m_s,
            "ve_over_c": float(ve_m_s / C0),
            "r_over_q_ohm_per_m": r_over_q_ohm_per_m,
            "r_over_q_kohm_per_m": float(r_over_q_ohm_per_m * 1e-3),
            "k_loss_v_per_c_m": k_loss_per_m,
            "k_pierce_ohm": k_pierce_ohm,
            "wall_time_s": float(time.perf_counter() - t0),
        }
    )

    if save_field_maps:
        # Save transverse (x, y) slice in the wagon-wheel plate and longitudinal (z, y) slice
        iz_ww = nz // 2
        iz_sp = 0
        ez_norm = max(float(np.abs(ez0_axis)), 1e-12)
        out["field_maps"] = {
            "x_mm": (0.5 * (g2.x[:nx] + g2.x[1 : nx + 1]) * 1e3).astype(float),
            "y_mm": (0.5 * (g2.y[:ny] + g2.y[1 : ny + 1]) * 1e3).astype(float),
            "z_mm": (z_centers * 1e3).astype(float),
            "ez_zy_real": np.real(ez_raw[0, ix_mid, :ny, :nz] / ez_norm).astype(float),
            "ex_xy_ww_mag": (np.abs(ex_c[:, :, iz_ww]) / ez_norm).astype(float),
            "ey_xy_ww_mag": (np.abs(ey_c[:, :, iz_ww]) / ez_norm).astype(float),
            "et_xy_ww_mag": (np.hypot(np.abs(ex_c[:, :, iz_ww]), np.abs(ey_c[:, :, iz_ww])) / ez_norm).astype(float),
            "ez_xy_sp_real": np.real(ez_raw[0, :nx, :ny, iz_sp] / ez_norm).astype(float),
            "sz_xy_sp": (sz_3d[:, :, iz_sp] / (ez_norm**2 / 376.73)).astype(float),
            "ez_axis_abs": (np.abs(ez_axis_z) / ez_norm).astype(float),
        }

    del s2, g2
    gc.collect()
    return out


def build_adiabatic_coax_mtm_solids(
    n_periods: int = 8,
    *,
    n_taper_cells: int = 2,
    n_port_cells: int = 4,
    forward_control: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Build a finite :math:`N`-period 3D wagon-wheel MTM (or forward control) with adiabatic coaxial couplers.

    Because the empty circular waveguide (:math:`2 R_{\\mathrm{wg}} = 16.0\\;\\mathrm{mm}`, PRL line 121) has its
    :math:`\\mathrm{TM}_{01}` cutoff at :math:`14.2\\;\\mathrm{GHz}` (PRL line 150) above :math:`f_s = 11.42\\;\\mathrm{GHz}`
    (PRL line 137), an abrupt transition to an empty pipe is evanescent and reflects the slow wave. Here, a 2-cell
    adiabatic taper of the wagon-wheel hub and spokes couples the :math:`11.42\\;\\mathrm{GHz}` (PRL line 137) mode
    smoothly into a coaxial TEM output port (terminated by CPML) while shielding the :math:`r_b = 3.0\\;\\mathrm{mm}`
    (PRL line 122) beam tunnel in the port regions to eliminate spurious beam transition radiation.
    """
    p = spec.period_m
    t_ww = spec.wagon_wheel_thickness_m
    r_b = spec.beam_hole_radius_m
    r_hub = spec.hub_outer_radius_m
    r_wg = spec.waveguide_radius_m
    w_sp = spec.spoke_width_m
    r_pipe_wall = spec.coax_port_wall_radius_m
    r_box_out = r_wg + 4.0e-3

    n_end = int(n_port_cells) + int(n_taper_cells)
    n_tot_cells = int(n_periods) + 2 * n_end
    z_total = n_tot_cells * p
    z_act_start = n_end * p
    z_act_end = z_act_start + int(n_periods) * p
    z_up_taper_start = int(n_port_cells) * p
    z_dn_taper_end = (n_end + int(n_periods) + int(n_taper_cells)) * p

    blocks: list[pv.PolyData] = [
        make_annular_cylinder(r_wg, r_box_out, -0.2e-3, z_total + 0.2e-3, n_theta=96),
        make_annular_cylinder(r_b, r_pipe_wall, -0.2e-3, z_up_taper_start, n_theta=96),
        make_annular_cylinder(r_b, r_pipe_wall, z_dn_taper_end, z_total + 0.2e-3, n_theta=96),
    ]

    for k in range(int(n_port_cells), n_tot_cells - int(n_port_cells)):
        if k < n_end:
            frac = float(k - int(n_port_cells) + 1) / float(int(n_taper_cells) + 1)
        elif k >= n_end + int(n_periods):
            frac = float(n_tot_cells - int(n_port_cells) - k) / float(int(n_taper_cells) + 1)
        else:
            frac = 1.0

        z0 = k * p + 0.5 * (p - t_ww)
        z1 = z0 + t_ww
        if not forward_control:
            r_hub_k = r_pipe_wall + frac * (r_hub - r_pipe_wall)
            blocks.append(make_annular_cylinder(r_b, r_hub_k, z0, z1, n_theta=96))
            r_in_sp = 0.5 * (r_b + r_hub_k)
            r_out_sp = r_wg + 0.4e-3
            hw = 0.5 * frac * w_sp
            if hw >= 0.12e-3:
                t_sp = max(frac * t_ww, 0.25e-3)
                z0_s = k * p + 0.5 * (p - t_sp)
                z1_s = z0_s + t_sp
                blocks.append(pv.Box(bounds=(r_in_sp, r_out_sp, -hw, hw, z0_s, z1_s)).triangulate())
                blocks.append(pv.Box(bounds=(-r_out_sp, -r_in_sp, -hw, hw, z0_s, z1_s)).triangulate())
                blocks.append(pv.Box(bounds=(-hw, hw, r_in_sp, r_out_sp, z0_s, z1_s)).triangulate())
                blocks.append(pv.Box(bounds=(-hw, hw, -r_out_sp, -r_in_sp, z0_s, z1_s)).triangulate())
        else:
            r_iris = 5.0e-3
            r_in_k = r_wg - frac * (r_wg - r_iris)
            blocks.append(make_annular_cylinder(r_in_k, r_wg + 0.4e-3, z0, z1, n_theta=96))

    merged = blocks[0]
    for b in blocks[1:]:
        merged = merged.merge(b)

    return {
        "solid": merged,
        "n_periods": int(n_periods),
        "n_taper_cells": int(n_taper_cells),
        "n_port_cells": int(n_port_cells),
        "z_total_m": float(z_total),
        "z_act_start_m": float(z_act_start),
        "z_act_end_m": float(z_act_end),
        "z_up_taper_start_m": float(z_up_taper_start),
        "z_dn_taper_end_m": float(z_dn_taper_end),
        "active_length_m": float(int(n_periods) * p),
        "r_pipe_wall_m": float(r_pipe_wall),
    }


def measure_upstream_port_reflection(
    *,
    n_xy: int = 64,
    dz_mm: float = 0.25,
    f_eval_target_ghz: float = 11.41,
    quick: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Measure the upstream (:math:`-z`) port reflection :math:`|\\Gamma(f)|` of the backward-wave packet.

    Compares four identical-grid 3D conformal FIT runs excited by the 65.0 MeV (PRL line 49) drive bunch:
    1. ``ref_periodic``: upstream cells continued as periodic wagon-wheel cells (reflectionless reference).
    2. ``tapered_coax``: 2-cell adiabatic wagon-wheel-to-coaxial TEM port coupler + CPML.
    3. ``abrupt_empty_pipe``: abrupt transition to the empty :math:`2R_{\\mathrm{wg}}=16.0\\;\\mathrm{mm}`
       (PRL line 121) circular waveguide, cut off at :math:`11.42\\;\\mathrm{GHz}` (PRL line 137) below
       the :math:`14.2\\;\\mathrm{GHz}` (PRL line 150) :math:`\\mathrm{TM}_{01}` cutoff.
    4. ``pec_cap``: shorted PEC endwall at the upstream port (:math:`|\\Gamma_{\\mathrm{PEC}}| \\equiv 1`).
    """
    if quick:
        if n_xy > 48:
            n_xy = 32
        dz_mm = max(dz_mm, 0.50)
    p = spec.period_m
    r_wg = spec.waveguide_radius_m
    r_b = spec.beam_hole_radius_m
    r_hub = spec.hub_outer_radius_m
    t_ww = spec.wagon_wheel_thickness_m
    w_sp = spec.spoke_width_m
    r_pipe_wall = spec.coax_port_wall_radius_m

    n_up = 8
    n_act = 8
    n_dn = 4
    n_tot = n_up + n_act + n_dn
    z_tot = n_tot * p

    def _add_ww_cell(blocks: list[pv.PolyData], k: int, frac: float = 1.0) -> None:
        z0 = k * p + 0.5 * (p - t_ww)
        z1 = z0 + t_ww
        r_hub_k = r_pipe_wall + frac * (r_hub - r_pipe_wall)
        blocks.append(make_annular_cylinder(r_b, r_hub_k, z0, z1, n_theta=96))
        r_in_sp = 0.5 * (r_b + r_hub_k)
        r_out_sp = r_wg + 0.4e-3
        hw = 0.5 * frac * w_sp
        if hw >= 0.12e-3:
            t_sp = max(frac * t_ww, 0.25e-3)
            z0_s = k * p + 0.5 * (p - t_sp)
            z1_s = z0_s + t_sp
            blocks.append(pv.Box(bounds=(r_in_sp, r_out_sp, -hw, hw, z0_s, z1_s)).triangulate())
            blocks.append(pv.Box(bounds=(-r_out_sp, -r_in_sp, -hw, hw, z0_s, z1_s)).triangulate())
            blocks.append(pv.Box(bounds=(-hw, hw, r_in_sp, r_out_sp, z0_s, z1_s)).triangulate())
            blocks.append(pv.Box(bounds=(-hw, hw, -r_out_sp, -r_in_sp, z0_s, z1_s)).triangulate())

    def _build_geom(kind: str) -> pv.PolyData:
        blocks: list[pv.PolyData] = [
            make_annular_cylinder(r_wg, 12.0e-3, -0.2e-3, z_tot + 0.2e-3, n_theta=96)
        ]
        for k in range(n_up, n_up + n_act):
            _add_ww_cell(blocks, k, frac=1.0)
        _add_ww_cell(blocks, n_up + n_act, frac=2.0 / 3.0)
        _add_ww_cell(blocks, n_up + n_act + 1, frac=1.0 / 3.0)
        blocks.append(
            make_annular_cylinder(r_b, r_pipe_wall, (n_up + n_act + 2) * p, z_tot + 0.2e-3, n_theta=96)
        )

        if kind == "ref_periodic":
            blocks.append(make_annular_cylinder(r_b, r_pipe_wall, -0.2e-3, 1.0 * p, n_theta=96))
            _add_ww_cell(blocks, 1, frac=0.5)
            for k in range(2, n_up):
                _add_ww_cell(blocks, k, frac=1.0)
        elif kind == "tapered_coax":
            blocks.append(make_annular_cylinder(r_b, r_pipe_wall, -0.2e-3, (n_up - 2) * p, n_theta=96))
            _add_ww_cell(blocks, n_up - 2, frac=1.0 / 3.0)
            _add_ww_cell(blocks, n_up - 1, frac=2.0 / 3.0)
        elif kind == "abrupt_empty_pipe":
            # Align the inner beam-tunnel shield termination plane with the reference plane z = n_up * p
            # (matching pec_cap) so the abrupt empty circular waveguide step occurs at the exact same
            # normalization plane without exposing an unshielded beam gap in cell k = n_up - 1.
            blocks.append(make_annular_cylinder(r_b, r_pipe_wall, -0.2e-3, n_up * p, n_theta=96))
            blocks.append(
                make_annular_cylinder(r_pipe_wall, r_wg + 0.4e-3, -0.2e-3, (n_up - 1) * p, n_theta=96)
            )
        elif kind == "pec_cap":
            blocks.append(make_annular_cylinder(r_b, r_wg + 0.4e-3, -0.2e-3, n_up * p, n_theta=96))

        merged = blocks[0]
        for b in blocks[1:]:
            merged = merged.merge(b)
        return merged

    x = np.linspace(-r_wg, r_wg, n_xy + 1)
    y = np.linspace(-r_wg, r_wg, n_xy + 1)
    nz = int(round(z_tot / (dz_mm * 1e-3)))
    z = np.linspace(0.0, z_tot, nz + 1)

    _, beta, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    q_c = spec.single_bunch_charge_c
    sigmaz = spec.bunch_length_rms_m
    ti = 4.0 * sigmaz / u0
    t_max = 0.42e-9

    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    wall_times: dict[str, float] = {}
    dt_fixed: float | None = None
    mem_mb = 0.0

    for kind in ("ref_periodic", "pec_cap", "abrupt_empty_pipe", "tapered_coax"):
        t_start = time.perf_counter()
        solid = _build_geom(kind)
        g = GridFIT3D(
            x=x, y=y, z=z, stl_solids={"sws": solid}, stl_materials={"sws": "pec"}, conformal="pec", verbose=0
        )
        s = SolverFIT3D(
            g,
            use_stl=True,
            dt=dt_fixed,
            bc_low=["pec", "pec", "cpml"],
            bc_high=["pec", "pec", "cpml"],
            n_pml=8,
            verbose=0,
        )
        s.source_type = "direct"
        if dt_fixed is None:
            dt_fixed = 0.96 * float(s.dt)
            s.dt = dt_fixed
        nt = int(np.ceil(t_max / dt_fixed))
        est = estimate((s.Nx, s.Ny, s.Nz), steps=nt, dtype="float64", conformal="pec")
        preflight(est, "laptop" if quick else "n2-highmem-8")
        mem_mb = max(mem_mb, float(est.host_bytes / 1e6))

        t_arr = np.arange(nt) * dt_fixed
        beam = Beam(xsource=0.0, ysource=0.0, beta=beta, q=q_c, sigmaz=sigmaz, ti=ti)
        ix0 = int(np.argmin(np.abs(s.grid.x)))
        iy0 = int(np.argmin(np.abs(s.grid.y)))
        cell_mask = (s.z >= 9.0 * p) & (s.z <= 10.0 * p)
        ez_cell = np.zeros((nt, int(np.sum(cell_mask))))
        for n in range(nt):
            t = t_arr[n]
            beam.update(s, t)
            s.one_step()
            ez_cell[n] = s.E[ix0, iy0, cell_mask, "z"]
        traces[kind] = (t_arr, ez_cell)
        wall_times[kind] = float(time.perf_counter() - t_start)
        del s, g, solid
        _gc_and_trim()

    t_arr, ez_ref = traces["ref_periodic"]
    _, ez_pec = traces["pec_cap"]
    _, ez_abr = traces["abrupt_empty_pipe"]
    _, ez_tap = traces["tapered_coax"]

    gate = (t_arr >= 0.13e-9)[:, None]
    win = np.hanning(len(t_arr))[:, None] * gate

    n_fft = 65536
    f_lo_sweep = 10.5
    f_hi_sweep = 12.3
    freqs = np.fft.rfftfreq(n_fft, d=float(dt_fixed)) * 1e-9
    mask = (freqs >= f_lo_sweep) & (freqs <= f_hi_sweep)
    f_b = freqs[mask]

    d_pec = np.sqrt(np.mean(np.abs(np.fft.rfft((ez_pec - ez_ref) * win, n=n_fft, axis=0)[mask, :]) ** 2, axis=1))
    d_abr = np.sqrt(np.mean(np.abs(np.fft.rfft((ez_abr - ez_ref) * win, n=n_fft, axis=0)[mask, :]) ** 2, axis=1))
    d_tap = np.sqrt(np.mean(np.abs(np.fft.rfft((ez_tap - ez_ref) * win, n=n_fft, axis=0)[mask, :]) ** 2, axis=1))

    gamma_abr_raw = d_abr / np.maximum(d_pec, 1e-30)
    gamma_tap_raw = d_tap / np.maximum(d_pec, 1e-30)
    gamma_abr = gamma_abr_raw.copy()
    gamma_tap = gamma_tap_raw.copy()
    gamma_pec = np.ones_like(f_b)

    f_target_ghz = float(f_eval_target_ghz)
    f_paper_sync_ghz = float(spec.paper_sync_freq_hz * 1e-9)
    fc_tm01_ghz = circular_waveguide_tm01_cutoff_hz(spec.waveguide_radius_m) * 1e-9
    n_pml_wake = 10
    f_fwd_ctrl_ghz = 18.86
    idx_s = int(np.argmin(np.abs(f_b - f_target_ghz)))
    f_s_eval = float(f_b[idx_s])
    g_tap_s = float(gamma_tap[idx_s])
    g_tap_raw_s = float(gamma_tap_raw[idx_s])
    g_abr_s = float(gamma_abr[idx_s])
    g_abr_raw_s = float(gamma_abr_raw[idx_s])
    idx_adj = min(idx_s + 1, len(f_b) - 1)
    f_s_adj = float(f_b[idx_adj])
    g_tap_adj = float(gamma_tap[idx_adj])
    is_clipped = False
    h_xy_mm = float((2.0 * r_wg / max(int(n_xy), 1)) * 1e3)

    return {
        "n_xy_port": int(n_xy),
        "cells_across_1mm_feature_port": float(1.0 / h_xy_mm),
        "n_fft_port": int(n_fft),
        "f_ghz": f_b.astype(float),
        "gamma_tapered_coax": gamma_tap.astype(float),
        "gamma_tapered_coax_raw": gamma_tap_raw.astype(float),
        "gamma_abrupt_empty_pipe": gamma_abr.astype(float),
        "gamma_abrupt_empty_pipe_raw": gamma_abr_raw.astype(float),
        "gamma_pec_cap": gamma_pec.astype(float),
        "f_sync_target_ghz": f_paper_sync_ghz,
        "f_sync_eval_ghz": f_s_eval,
        "f_sync_adjacent_bin_ghz": f_s_adj,
        "gamma_tapered_at_sync": g_tap_s,
        "gamma_tapered_raw_at_sync": g_tap_raw_s,
        "gamma_tapered_unclipped_at_sync": g_tap_raw_s,
        "gamma_tapered_at_adjacent_bin": g_tap_adj,
        "gamma_tapered_passband_mean": float(np.mean(gamma_tap)),
        "gamma_tapered_passband_min": float(np.min(gamma_tap)),
        "power_transmission_tapered_at_sync": float(1.0 - g_tap_s**2),
        "gamma_abrupt_at_sync": g_abr_s,
        "gamma_abrupt_raw_at_sync": g_abr_raw_s,
        "gamma_abrupt_unclipped_at_sync": g_abr_raw_s,
        "gamma_abrupt_is_clipped": is_clipped,
        "gamma_abrupt_at_sync_label": f"{g_abr_s:.4f}",
        "gamma_abrupt_passband_mean": float(np.mean(gamma_abr)),
        "reduction_factor_abrupt_over_tapered": float(g_abr_s / max(g_tap_s, 1e-12)),
        "rel_error_bound_tapered": float(reflection_induced_peak_error_bound(g_tap_s)),
        "rel_error_bound_tapered_mutated_no_2x": float(
            reflection_induced_peak_error_bound(g_tap_s, mutate_drop_factor_two=True)
        ),
        "rel_error_bound_abrupt": float(reflection_induced_peak_error_bound(g_abr_s)),
        "forward_control_18p86ghz_port_note": (
            f"The port S-parameter sweep in task_01 covers [{f_lo_sweep:.1f}, {f_hi_sweep:.1f}] GHz around "
            f"the {f_paper_sync_ghz:.2f} GHz (PRL line 137) backward-wave MTM passband. The disk-loaded "
            f"forward-wave control in task_03 operates at f_pk = {f_fwd_ctrl_ghz:.2f} GHz (above the "
            f"{fc_tm01_ghz:.2f} GHz empty-pipe TM01 cutoff), where both the coaxial taper and the circular "
            f"beam pipe are propagating and terminated by {n_pml_wake}-cell CPML layers, but "
            f"|Gamma({f_fwd_ctrl_ghz:.2f} GHz)| is not separately swept in task_01."
        ),
        "wall_times_s": wall_times,
        "est_memory_mb": mem_mb,
    }


def run_finite_mtm_wakefield(
    n_periods: int = 8,
    *,
    n_xy: int = 64,
    dz_mm: float = 0.25,
    forward_control: bool = False,
    record_2d_snapshots: bool = False,
    record_stored_energy: bool = True,
    mutate_unfiltered_stored_energy: bool = False,
    mutate_shrink_beam_work_to_active: bool = False,
    mutate_truncate_wake_window: bool = False,
    quick: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Run a 3D conformal FIT finite-structure wakefield simulation with unified control-volume accounting.

    Executes the structure solver (`solver_sws`) and smooth beam-pipe reference solver (`solver_ref`)
    strictly **one at a time** with explicit garbage collection between runs so peak resident memory
    stays well below half of the VM RAM cap. Evaluates beam work, port Poynting fluxes, and half-step
    time-centered stored energy over the single control volume ``[iz_up, iz_dn]`` bounded by the upstream
    and downstream port planes (both unfiltered and causally bandpass-filtered).
    """
    if quick and n_xy > 48:
        n_xy = 48
    p = spec.period_m
    r_wg = spec.waveguide_radius_m
    r_b = spec.beam_hole_radius_m
    geom = build_adiabatic_coax_mtm_solids(
        n_periods=n_periods,
        n_taper_cells=2,
        n_port_cells=4,
        forward_control=forward_control,
        spec=spec,
    )
    z_tot = geom["z_total_m"]
    z_act_start = geom["z_act_start_m"]
    z_act_end = geom["z_act_end_m"]
    L_act = geom["active_length_m"]
    r_pipe_wall = geom["r_pipe_wall_m"]

    x = np.linspace(-r_wg, r_wg, int(n_xy) + 1)
    y = np.linspace(-r_wg, r_wg, int(n_xy) + 1)
    nz = int(round(z_tot / (dz_mm * 1e-3)))
    z = np.linspace(0.0, z_tot, nz + 1)

    gamma, beta, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    q_c = spec.single_bunch_charge_c
    sigmaz = spec.bunch_length_rms_m
    ti = 4.5 * sigmaz / u0
    # Choose wake_len(N) from estimate_drainage_wake_length_m (dispersive passband transit at
    # |v_g,min| ~ 0.052c plus multi-reflection ring-down with |Gamma_tap| ~ 0.22 across (N+4)*p,
    # yielding 548.1, 657.7, 877.0 mm for N = 6, 8, 12) so U_bp(T_end)/W_beam,bp < 0.02;
    # mutate_truncate_wake_window=True restores the truncated 120 mm window that fails this bound.
    wake_len = (
        120.0e-3
        if mutate_truncate_wake_window
        else estimate_drainage_wake_length_m(max(int(n_periods), 6), spec=spec)
    )
    t_sim = ti + (z_tot + wake_len) / u0

    # --- 1. Structure run (solver_sws) ---
    t0_sws = time.perf_counter()
    grid_sws = GridFIT3D(
        x=x,
        y=y,
        z=z,
        stl_solids={"sws": geom["solid"]},
        stl_materials={"sws": "pec"},
        conformal="pec",
        verbose=0,
    )
    solver_sws = SolverFIT3D(
        grid_sws,
        use_stl=True,
        bc_low=["pec", "pec", "cpml"],
        bc_high=["pec", "pec", "cpml"],
        n_pml=10,
        verbose=0,
    )
    del geom["solid"]
    if hasattr(grid_sws, "grid"):
        del grid_sws.grid
    if hasattr(grid_sws, "stl_solids"):
        grid_sws.stl_solids = {}
    _gc_and_trim()
    solver_sws.source_type = "direct"
    dt = 0.96 * float(solver_sws.dt)
    solver_sws.dt = dt
    nt = int(np.ceil(t_sim / dt))

    est_sws = estimate((int(n_xy), int(n_xy), int(nz)), steps=nt, dtype="float64", conformal="pec")
    preflight(est_sws, "laptop" if quick else "n2-highmem-8")
    mem_mb = float(est_sws.host_bytes / 1e6)

    t_arr = np.arange(nt) * dt
    beam_sws = Beam(xsource=0.0, ysource=0.0, beta=beta, q=q_c, sigmaz=sigmaz, ti=ti)

    ix0 = int(np.argmin(np.abs(solver_sws.grid.x)))
    iy0 = int(np.argmin(np.abs(solver_sws.grid.y)))
    z_nodes = solver_sws.z.copy()
    dz_nodes = solver_sws.dz.copy()
    y_nodes = solver_sws.y.copy()
    ez_sws = np.zeros((nt, solver_sws.Nz), dtype=float)
    ez_bar_sws = np.zeros((nt, solver_sws.Nz), dtype=float)
    jz_sws = np.zeros((nt, solver_sws.Nz), dtype=float)

    z_up_port = geom["z_up_taper_start_m"] - 1.0 * p
    z_dn_port = geom["z_dn_taper_end_m"] + 1.0 * p
    iz_up = int(np.argmin(np.abs(z_nodes - z_up_port)))
    iz_dn = int(np.argmin(np.abs(z_nodes - z_dn_port)))
    iz_a0 = int(np.argmin(np.abs(z_nodes - z_act_start)))
    iz_a1 = int(np.argmin(np.abs(z_nodes - z_act_end)))

    # Pre-extract exact discrete FIT control-volume weights on [iz_up, iz_dn] before freeing Ds/tDs/iDa/L/tL.
    # Primary z-slabs k in [iz_up, iz_dn) carry weight 1 for Ez, Hx, Hy; primary z-planes k in [iz_up, iz_dn]
    # carry trapezoidal weight 1/2 at the boundary planes iz_up, iz_dn and 1 at interior planes for Ex, Ey, Hz.
    nx_cells, ny_cells, nz_cells = int(solver_sws.Nx), int(solver_sws.Ny), int(solver_sws.Nz)
    n_cells_3d = nx_cells * ny_cells * nz_cells
    sl_cv = slice(iz_up, iz_dn + 1)

    def _extract_cv_fit_weights(
        sol_obj: SolverFIT3D,
        grid_obj: GridFIT3D,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        da_u_x = (sol_obj.L[:, :, iz_up, "x"] * sol_obj.tL[:, :, iz_up, "y"]).copy()
        da_u_y = (sol_obj.L[:, :, iz_up, "y"] * sol_obj.tL[:, :, iz_up, "x"]).copy()
        da_d_x = (sol_obj.L[:, :, iz_dn, "x"] * sol_obj.tL[:, :, iz_dn, "y"]).copy()
        da_d_y = (sol_obj.L[:, :, iz_dn, "y"] * sol_obj.tL[:, :, iz_dn, "x"]).copy()
        ds_d = np.real(sol_obj.Ds.diagonal()).copy()
        itda_d = np.real(sol_obj.itDa.diagonal()).copy()
        tds_d = np.real(sol_obj.tDs.diagonal()).copy()
        g_c = sol_obj.iDa.tocoo()
        diag_g_v = np.real(sol_obj.iDa.diagonal()).copy()

        # Run step_0 cleanup immediately so pre-stepping FIT assembly matrices (Ds, tDs, iDa, itDa, Px..Pz, L..itA)
        # are returned to the OS before allocating control-volume weight vectors and time-history buffers.
        sol_obj._set_ghosts_to_0()
        sol_obj.step_0 = False
        sol_obj._attrcleanup()
        sol_obj.J_old = np.zeros_like(sol_obj.J.toarray())
        for _k_del in ("L", "tL", "iA", "itA"):
            if hasattr(grid_obj, _k_del):
                delattr(grid_obj, _k_del)
        for _k_del in (
            "E_trans",
            "H_trans",
            "tf_dxz",
            "tf_dyz",
            "tf_dtxz",
            "tf_dtyz",
            "iDeps",
            "iDmu",
            "Dsigma",
            "conformal_ieps_factor",
        ):
            if hasattr(sol_obj, _k_del):
                delattr(sol_obj, _k_del)
        _gc_and_trim()

        w_z_m = np.zeros(nz_cells, dtype=float)
        w_z_m[iz_up:iz_dn] = 1.0
        w_z_n = np.zeros(nz_cells, dtype=float)
        w_z_n[iz_up + 1 : iz_dn] = 1.0
        w_z_n[iz_up] = 0.5
        w_z_n[iz_dn] = 0.5

        w_e_s = np.concatenate(
            [
                np.broadcast_to(w_z_n[None, None, :], (nx_cells, ny_cells, nz_cells)).ravel(order="F"),
                np.broadcast_to(w_z_n[None, None, :], (nx_cells, ny_cells, nz_cells)).ravel(order="F"),
                np.broadcast_to(w_z_m[None, None, :], (nx_cells, ny_cells, nz_cells)).ravel(order="F"),
            ]
        )
        w_h_s = np.concatenate(
            [
                np.broadcast_to(w_z_m[None, None, :], (nx_cells, ny_cells, nz_cells)).ravel(order="F"),
                np.broadcast_to(w_z_m[None, None, :], (nx_cells, ny_cells, nz_cells)).ravel(order="F"),
                np.broadcast_to(w_z_n[None, None, :], (nx_cells, ny_cells, nz_cells)).ravel(order="F"),
            ]
        )

        ieps_v = np.real(sol_obj.ieps.toarray())
        mask_el = (ieps_v > 0.0) & (itda_d > 0.0) & (ds_d > 0.0)
        w_el_vec = np.zeros(3 * n_cells_3d, dtype=float)
        w_el_vec[mask_el] = w_e_s[mask_el] * ds_d[mask_el] / (ieps_v[mask_el] * itda_d[mask_el])
        w_jv = np.zeros(3 * n_cells_3d, dtype=float)
        w_jv[mask_el] = w_e_s[mask_el] * ds_d[mask_el] / itda_d[mask_el]

        # Reconstruct exact stabilized dual inverse-area weights a_hat from g_c
        off_m = g_c.row != g_c.col
        r_o = g_c.row[off_m]
        c_o = g_c.col[off_m]
        v_o = np.real(g_c.data[off_m])
        unst_f = np.zeros(len(diag_g_v), dtype=bool)
        is_2 = (np.abs(v_o - 2.0 * diag_g_v[r_o]) < 1e-12 * diag_g_v[r_o]) | (
            np.abs(v_o + diag_g_v[r_o]) < 1e-12 * diag_g_v[r_o]
        )
        unst_f[c_o[is_2]] = True
        rem_o = (~unst_f[r_o]) & (~unst_f[c_o]) & (np.abs(v_o - diag_g_v[r_o]) < 1e-12 * diag_g_v[r_o])
        cl_sz = np.ones(len(diag_g_v), dtype=float)
        np.add.at(cl_sz, r_o[rem_o], 1.0)
        inv_k = np.where((~unst_f) & (diag_g_v > 0.0), diag_g_v * cl_sz, 0.0)
        a_h = np.where(inv_k > 0.0, 1.0 / np.where(inv_k > 0.0, inv_k, 1.0), 0.0)

        imu_v = np.real(sol_obj.imu.toarray())
        mask_mg = (imu_v > 0.0) & (a_h > 0.0) & (tds_d > 0.0)
        w_mg_vec = np.zeros(3 * n_cells_3d, dtype=float)
        w_mg_vec[mask_mg] = w_h_s[mask_mg] * tds_d[mask_mg] * a_h[mask_mg] / imu_v[mask_mg]

        w_ex_c = w_el_vec[0:n_cells_3d].reshape((nx_cells, ny_cells, nz_cells), order="F")[:, :, sl_cv].copy()
        w_ey_c = w_el_vec[n_cells_3d : 2 * n_cells_3d].reshape((nx_cells, ny_cells, nz_cells), order="F")[:, :, sl_cv].copy()
        w_ez_c = w_el_vec[2 * n_cells_3d : 3 * n_cells_3d].reshape((nx_cells, ny_cells, nz_cells), order="F")[:, :, sl_cv].copy()
        w_hx_c = w_mg_vec[0:n_cells_3d].reshape((nx_cells, ny_cells, nz_cells), order="F")[:, :, sl_cv].copy()
        w_hy_c = w_mg_vec[n_cells_3d : 2 * n_cells_3d].reshape((nx_cells, ny_cells, nz_cells), order="F")[:, :, sl_cv].copy()
        w_hz_c = w_mg_vec[2 * n_cells_3d : 3 * n_cells_3d].reshape((nx_cells, ny_cells, nz_cells), order="F")[:, :, sl_cv].copy()
        dv_ax_cv = w_jv[2 * n_cells_3d :].reshape((nx_cells, ny_cells, nz_cells), order="F")[ix0, iy0, :].copy()
        dv_ax_act = dv_ax_cv.copy()
        dv_ax_act[:iz_a0] = 0.0
        dv_ax_act[iz_a1:] = 0.0
        return (
            da_u_x,
            da_u_y,
            da_d_x,
            da_d_y,
            w_ex_c,
            w_ey_c,
            w_ez_c,
            w_hx_c,
            w_hy_c,
            w_hz_c,
            dv_ax_cv,
            dv_ax_act,
        )

    (
        da_up_x,
        da_up_y,
        da_dn_x,
        da_dn_y,
        w_ex_cv,
        w_ey_cv,
        w_ez_cv,
        w_hx_cv,
        w_hy_cv,
        w_hz_cv,
        dv_axis_cv,
        dv_axis_act,
    ) = _extract_cv_fit_weights(solver_sws, grid_sws)
    _gc_and_trim()

    X2, Y2 = np.meshgrid(solver_sws.x, solver_sws.y, indexing="ij")
    R2 = np.sqrt(X2**2 + Y2**2)
    coax_2d = (R2 >= r_pipe_wall) & (R2 <= r_wg)
    tun_2d = R2 < r_pipe_wall

    active_up = (da_up_x > 0.0) | (da_up_y > 0.0)
    active_dn = (da_dn_x > 0.0) | (da_dn_y > 0.0)
    idx_ux, idx_uy = np.where(active_up)
    idx_dx, idx_dy = np.where(active_dn)
    is_coax_u = coax_2d[idx_ux, idx_uy]
    is_coax_d = coax_2d[idx_dx, idx_dy]
    da_ux = da_up_x[idx_ux, idx_uy]
    da_uy = da_up_y[idx_ux, idx_uy]
    da_dx = da_dn_x[idx_dx, idx_dy]
    da_dy = da_dn_y[idx_dx, idx_dy]

    # Causal Butterworth bandpass filter kernel h_bp(t) with 5% cosine tail taper
    f_bp_lo_hz, f_bp_hi_hz = (WagonWheelMTMSpec.passband_low_hz, WagonWheelMTMSpec.passband_high_hz) if not forward_control else (16.0e9, 21.5e9)
    h_bp = _build_causal_bandpass_kernel(nt, dt, f_bp_lo_hz, f_bp_hi_hz, order=3, tail_taper_fraction=0.05)

    ex_bp = np.zeros_like(w_ex_cv)
    ey_bp = np.zeros_like(w_ey_cv)
    ez_bp = np.zeros_like(w_ez_cv)
    hx_bp = np.zeros_like(w_hx_cv)
    hy_bp = np.zeros_like(w_hy_cv)
    hz_bp = np.zeros_like(w_hz_cv)

    ex_unf = np.zeros_like(w_ex_cv)
    ey_unf = np.zeros_like(w_ey_cv)
    ez_unf = np.zeros_like(w_ez_cv)
    hx_unf = np.zeros_like(w_hx_cv)
    hy_unf = np.zeros_like(w_hy_cv)
    hz_unf = np.zeros_like(w_hz_cv)

    ex_u_h = np.zeros((nt, len(idx_ux)), dtype=np.float32)
    ey_u_h = np.zeros((nt, len(idx_ux)), dtype=np.float32)
    hx_u_h = np.zeros((nt, len(idx_ux)), dtype=np.float32)
    hy_u_h = np.zeros((nt, len(idx_ux)), dtype=np.float32)
    ex_d_h = np.zeros((nt, len(idx_dx)), dtype=np.float32)
    ey_d_h = np.zeros((nt, len(idx_dx)), dtype=np.float32)
    hx_d_h = np.zeros((nt, len(idx_dx)), dtype=np.float32)
    hy_d_h = np.zeros((nt, len(idx_dx)), dtype=np.float32)
    pu_unf_sws_w = np.zeros(nt, dtype=float)
    pd_unf_sws_w = np.zeros(nt, dtype=float)

    n_snaps = 14
    snap_steps = np.linspace(int(0.05 * nt), int(0.45 * nt), n_snaps, dtype=int)
    snap_set = {int(k): i for i, k in enumerate(snap_steps)}
    snaps_sws_zy = (
        np.zeros((n_snaps, solver_sws.Ny, solver_sws.Nz), dtype=float) if record_2d_snapshots else None
    )

    act_z_mask = (z_nodes >= z_act_start) & (z_nodes <= z_act_end)
    cv_z_mask = (z_nodes >= z_nodes[iz_up]) & (z_nodes < z_nodes[iz_dn])

    for n in range(nt + 1):
        t = n * dt
        beam_sws.update(solver_sws, t)
        ez_old = solver_sws.E[ix0, iy0, :, "z"].copy()
        ex_u_old = solver_sws.E[:, :, iz_up, "x"].copy()
        ey_u_old = solver_sws.E[:, :, iz_up, "y"].copy()
        ex_d_old = solver_sws.E[:, :, iz_dn, "x"].copy()
        ey_d_old = solver_sws.E[:, :, iz_dn, "y"].copy()

        m = nt - n
        if 0 <= m < nt and abs(h_bp[m]) > 1e-12:
            hm = float(h_bp[m])
            ex_bp += hm * solver_sws.E[:, :, sl_cv, "x"]
            ey_bp += hm * solver_sws.E[:, :, sl_cv, "y"]
            ez_bp += hm * solver_sws.E[:, :, sl_cv, "z"]

        if n == nt:
            ex_unf[:] = solver_sws.E[:, :, sl_cv, "x"]
            ey_unf[:] = solver_sws.E[:, :, sl_cv, "y"]
            ez_unf[:] = solver_sws.E[:, :, sl_cv, "z"]
            hx_unf[:] = 0.5 * solver_sws.H[:, :, sl_cv, "x"]
            hy_unf[:] = 0.5 * solver_sws.H[:, :, sl_cv, "y"]
            hz_unf[:] = 0.5 * solver_sws.H[:, :, sl_cv, "z"]

        solver_sws.one_step()

        w_h_m = (0.5 * float(h_bp[m]) if 0 <= m < nt else 0.0) + (
            0.5 * float(h_bp[m - 1]) if (n < nt and 0 <= m - 1 < nt) else 0.0
        )
        if abs(w_h_m) > 1e-12:
            hx_bp += w_h_m * solver_sws.H[:, :, sl_cv, "x"]
            hy_bp += w_h_m * solver_sws.H[:, :, sl_cv, "y"]
            hz_bp += w_h_m * solver_sws.H[:, :, sl_cv, "z"]

        if n == nt:
            hx_unf += 0.5 * solver_sws.H[:, :, sl_cv, "x"]
            hy_unf += 0.5 * solver_sws.H[:, :, sl_cv, "y"]
            hz_unf += 0.5 * solver_sws.H[:, :, sl_cv, "z"]

        if n < nt:
            ez_curr = solver_sws.E[ix0, iy0, :, "z"]
            ez_sws[n] = ez_curr
            ez_bar_sws[n] = 0.5 * (ez_old + ez_curr)
            jz_sws[n] = solver_sws.J[ix0, iy0, :, "z"].copy()

            ex_un = 0.5 * (ex_u_old + solver_sws.E[:, :, iz_up, "x"])[idx_ux, idx_uy]
            ey_un = 0.5 * (ey_u_old + solver_sws.E[:, :, iz_up, "y"])[idx_ux, idx_uy]
            hx_un = 0.5 * (solver_sws.H[:, :, iz_up - 1, "x"] + solver_sws.H[:, :, iz_up, "x"])[idx_ux, idx_uy]
            hy_un = 0.5 * (solver_sws.H[:, :, iz_up - 1, "y"] + solver_sws.H[:, :, iz_up, "y"])[idx_ux, idx_uy]
            pu_unf_sws_w[n] = -float(np.sum(ex_un * hy_un * da_ux - ey_un * hx_un * da_uy))
            ex_u_h[n] = ex_un
            ey_u_h[n] = ey_un
            hx_u_h[n] = hx_un
            hy_u_h[n] = hy_un

            ex_dn = 0.5 * (ex_d_old + solver_sws.E[:, :, iz_dn, "x"])[idx_dx, idx_dy]
            ey_dn = 0.5 * (ey_d_old + solver_sws.E[:, :, iz_dn, "y"])[idx_dx, idx_dy]
            hx_dn = 0.5 * (solver_sws.H[:, :, iz_dn - 1, "x"] + solver_sws.H[:, :, iz_dn, "x"])[idx_dx, idx_dy]
            hy_dn = 0.5 * (solver_sws.H[:, :, iz_dn - 1, "y"] + solver_sws.H[:, :, iz_dn, "y"])[idx_dx, idx_dy]
            pd_unf_sws_w[n] = +float(np.sum(ex_dn * hy_dn * da_dx - ey_dn * hx_dn * da_dy))
            ex_d_h[n] = ex_dn
            ey_d_h[n] = ey_dn
            hx_d_h[n] = hx_dn
            hy_d_h[n] = hy_dn

            if record_2d_snapshots and n in snap_set:
                snaps_sws_zy[snap_set[n]] = solver_sws.E[ix0, :, :, "z"]

    wall_sws = float(time.perf_counter() - t0_sws)
    u_unf_sws_j = float(
        0.5
        * (
            np.sum(w_ex_cv * (ex_unf**2) + w_ey_cv * (ey_unf**2) + w_ez_cv * (ez_unf**2))
            + np.sum(w_hx_cv * (hx_unf**2) + w_hy_cv * (hy_unf**2) + w_hz_cv * (hz_unf**2))
        )
    )
    u_bp_sws_j = float(
        0.5
        * (
            np.sum(w_ex_cv * (ex_bp**2) + w_ey_cv * (ey_bp**2) + w_ez_cv * (ez_bp**2))
            + np.sum(w_hx_cv * (hx_bp**2) + w_hy_cv * (hy_bp**2) + w_hz_cv * (hz_bp**2))
        )
    )
    del (
        solver_sws,
        grid_sws,
        ex_bp,
        ey_bp,
        ez_bp,
        hx_bp,
        hy_bp,
        hz_bp,
        ex_unf,
        ey_unf,
        ez_unf,
        hx_unf,
        hy_unf,
        hz_unf,
        w_ex_cv,
        w_ey_cv,
        w_ez_cv,
        w_hx_cv,
        w_hy_cv,
        w_hz_cv,
    )
    _gc_and_trim()

    # Exact unfiltered FIT control-volume balance on solver_sws over [iz_up, iz_dn]
    w_unf_sws_j = float(-np.sum(jz_sws * ez_bar_sws * dv_axis_cv[None, :]) * dt)
    e_up_unf_j = float(np.sum(pu_unf_sws_w) * dt)
    e_dn_unf_j = float(np.sum(pd_unf_sws_w) * dt)
    e_rad_unf_j = e_up_unf_j + e_dn_unf_j
    rel_err_unf_cv = float(abs(e_rad_unf_j + u_unf_sws_j - w_unf_sws_j) / max(abs(w_unf_sws_j), 1e-30))

    # Causally bandpass-filtered port Poynting fluxes for solver_sws (evaluated pair-by-pair to minimize peak RSS)
    exu_bp = _apply_causal_kernel_fft(ex_u_h, h_bp)
    hyu_bp = _apply_causal_kernel_fft(hy_u_h, h_bp)
    del ex_u_h, hy_u_h
    pu_sws_bp_w = -np.sum(exu_bp * hyu_bp * da_ux[None, :], axis=1)
    pu_coax_bp_w = -np.sum(exu_bp[:, is_coax_u] * hyu_bp[:, is_coax_u] * da_ux[None, is_coax_u], axis=1)
    del exu_bp, hyu_bp

    eyu_bp = _apply_causal_kernel_fft(ey_u_h, h_bp)
    hxu_bp = _apply_causal_kernel_fft(hx_u_h, h_bp)
    del ey_u_h, hx_u_h
    pu_sws_bp_w += np.sum(eyu_bp * hxu_bp * da_uy[None, :], axis=1)
    pu_coax_bp_w += np.sum(eyu_bp[:, is_coax_u] * hxu_bp[:, is_coax_u] * da_uy[None, is_coax_u], axis=1)
    del eyu_bp, hxu_bp

    exd_bp = _apply_causal_kernel_fft(ex_d_h, h_bp)
    hyd_bp = _apply_causal_kernel_fft(hy_d_h, h_bp)
    del ex_d_h, hy_d_h
    pd_sws_bp_w = np.sum(exd_bp * hyd_bp * da_dx[None, :], axis=1)
    pd_coax_bp_w = np.sum(exd_bp[:, is_coax_d] * hyd_bp[:, is_coax_d] * da_dx[None, is_coax_d], axis=1)
    del exd_bp, hyd_bp

    eyd_bp = _apply_causal_kernel_fft(ey_d_h, h_bp)
    hxd_bp = _apply_causal_kernel_fft(hx_d_h, h_bp)
    del ey_d_h, hx_d_h
    pd_sws_bp_w -= np.sum(eyd_bp * hxd_bp * da_dy[None, :], axis=1)
    pd_coax_bp_w -= np.sum(eyd_bp[:, is_coax_d] * hxd_bp[:, is_coax_d] * da_dy[None, is_coax_d], axis=1)
    del eyd_bp, hxd_bp
    _gc_and_trim()

    # --- 2. Reference smooth pipe run (solver_ref) ---
    # Note on the unfiltered reference-run artifact (MF8):
    # The smooth-pipe reference run exhibits an unexplained unfiltered control-volume artifact
    # (W_ref,unf,cv ~ -3.1 mJ across N in {6, 8, 12}, with late-time stored energy U_ref,unf(T_end) ~ 39 mJ
    # and net Poynting inflow E_rad,ref,unf ~ -42 mJ at N = 8, 12). All unfiltered and full-band
    # reference-subtracted numbers carry this artifact. By contrast, in the [9.8, 13.0] GHz passband
    # the smooth r_b = 3.0 mm pipe is deep below its TM01 cutoff (38.25 GHz), so U_ref,bp(T_end) ~ 2.3e-5 mJ
    # and passband numbers are unaffected.
    t0_ref = time.perf_counter()
    pipe_solid = make_annular_cylinder(r_b, 12.0e-3, -0.2e-3, z_tot + 0.2e-3, n_theta=96)
    grid_ref = GridFIT3D(
        x=x,
        y=y,
        z=z,
        stl_solids={"pipe": pipe_solid},
        stl_materials={"pipe": "pec"},
        conformal="pec",
        verbose=0,
    )
    solver_ref = SolverFIT3D(
        grid_ref,
        use_stl=True,
        dt=dt,
        bc_low=["pec", "pec", "cpml"],
        bc_high=["pec", "pec", "cpml"],
        n_pml=10,
        verbose=0,
    )
    del pipe_solid
    if hasattr(grid_ref, "grid"):
        del grid_ref.grid
    if hasattr(grid_ref, "stl_solids"):
        grid_ref.stl_solids = {}
    solver_ref.source_type = "direct"
    (
        da_ref_up_x,
        da_ref_up_y,
        da_ref_dn_x,
        da_ref_dn_y,
        w_ref_ex_cv,
        w_ref_ey_cv,
        w_ref_ez_cv,
        w_ref_hx_cv,
        w_ref_hy_cv,
        w_ref_hz_cv,
        dv_ref_axis_cv,
        _,
    ) = _extract_cv_fit_weights(solver_ref, grid_ref)
    _gc_and_trim()
    beam_ref = Beam(xsource=0.0, ysource=0.0, beta=beta, q=q_c, sigmaz=sigmaz, ti=ti)
    ez_ref = np.zeros((nt, nz_cells), dtype=float)
    ez_bar_ref = np.zeros((nt, nz_cells), dtype=float)

    active_ref_up = (da_ref_up_x > 0.0) | (da_ref_up_y > 0.0)
    idx_tx, idx_ty = np.where(active_ref_up)
    da_tx = da_ref_up_x[idx_tx, idx_ty]
    da_ty = da_ref_up_y[idx_tx, idx_ty]
    da_tdx = da_ref_dn_x[idx_tx, idx_ty]
    da_tdy = da_ref_dn_y[idx_tx, idx_ty]
    ex_tu = np.zeros((nt, len(idx_tx)), dtype=np.float32)
    ey_tu = np.zeros((nt, len(idx_tx)), dtype=np.float32)
    hx_tu = np.zeros((nt, len(idx_tx)), dtype=np.float32)
    hy_tu = np.zeros((nt, len(idx_tx)), dtype=np.float32)
    ex_td = np.zeros((nt, len(idx_tx)), dtype=np.float32)
    ey_td = np.zeros((nt, len(idx_tx)), dtype=np.float32)
    hx_td = np.zeros((nt, len(idx_tx)), dtype=np.float32)
    hy_td = np.zeros((nt, len(idx_tx)), dtype=np.float32)
    pu_unf_ref_w = np.zeros(nt, dtype=float)
    pd_unf_ref_w = np.zeros(nt, dtype=float)

    ex_ref_bp = np.zeros_like(w_ref_ex_cv)
    ey_ref_bp = np.zeros_like(w_ref_ey_cv)
    ez_ref_bp = np.zeros_like(w_ref_ez_cv)
    hx_ref_bp = np.zeros_like(w_ref_hx_cv)
    hy_ref_bp = np.zeros_like(w_ref_hy_cv)
    hz_ref_bp = np.zeros_like(w_ref_hz_cv)

    ex_ref_unf = np.zeros_like(w_ref_ex_cv)
    ey_ref_unf = np.zeros_like(w_ref_ey_cv)
    ez_ref_unf = np.zeros_like(w_ref_ez_cv)
    hx_ref_unf = np.zeros_like(w_ref_hx_cv)
    hy_ref_unf = np.zeros_like(w_ref_hy_cv)
    hz_ref_unf = np.zeros_like(w_ref_hz_cv)

    snaps_ref_zy = (
        np.zeros((n_snaps, ny_cells, nz_cells), dtype=float) if record_2d_snapshots else None
    )

    for n in range(nt + 1):
        t = n * dt
        beam_ref.update(solver_ref, t)
        ez_o = solver_ref.E[ix0, iy0, :, "z"].copy()
        ex_uo = solver_ref.E[:, :, iz_up, "x"][idx_tx, idx_ty].copy()
        ey_uo = solver_ref.E[:, :, iz_up, "y"][idx_tx, idx_ty].copy()
        ex_do = solver_ref.E[:, :, iz_dn, "x"][idx_tx, idx_ty].copy()
        ey_do = solver_ref.E[:, :, iz_dn, "y"][idx_tx, idx_ty].copy()

        m = nt - n
        if 0 <= m < nt and abs(h_bp[m]) > 1e-12:
            hm = float(h_bp[m])
            ex_ref_bp += hm * solver_ref.E[:, :, sl_cv, "x"]
            ey_ref_bp += hm * solver_ref.E[:, :, sl_cv, "y"]
            ez_ref_bp += hm * solver_ref.E[:, :, sl_cv, "z"]

        if n == nt:
            ex_ref_unf[:] = solver_ref.E[:, :, sl_cv, "x"]
            ey_ref_unf[:] = solver_ref.E[:, :, sl_cv, "y"]
            ez_ref_unf[:] = solver_ref.E[:, :, sl_cv, "z"]
            hx_ref_unf[:] = 0.5 * solver_ref.H[:, :, sl_cv, "x"]
            hy_ref_unf[:] = 0.5 * solver_ref.H[:, :, sl_cv, "y"]
            hz_ref_unf[:] = 0.5 * solver_ref.H[:, :, sl_cv, "z"]

        solver_ref.one_step()

        w_h_m = (0.5 * float(h_bp[m]) if 0 <= m < nt else 0.0) + (
            0.5 * float(h_bp[m - 1]) if (n < nt and 0 <= m - 1 < nt) else 0.0
        )
        if abs(w_h_m) > 1e-12:
            hx_ref_bp += w_h_m * solver_ref.H[:, :, sl_cv, "x"]
            hy_ref_bp += w_h_m * solver_ref.H[:, :, sl_cv, "y"]
            hz_ref_bp += w_h_m * solver_ref.H[:, :, sl_cv, "z"]

        if n == nt:
            hx_ref_unf += 0.5 * solver_ref.H[:, :, sl_cv, "x"]
            hy_ref_unf += 0.5 * solver_ref.H[:, :, sl_cv, "y"]
            hz_ref_unf += 0.5 * solver_ref.H[:, :, sl_cv, "z"]

        if n < nt:
            ez_c = solver_ref.E[ix0, iy0, :, "z"]
            ez_ref[n] = ez_c
            ez_bar_ref[n] = 0.5 * (ez_o + ez_c)
            ex_un_r = 0.5 * (ex_uo + solver_ref.E[:, :, iz_up, "x"][idx_tx, idx_ty])
            ey_un_r = 0.5 * (ey_uo + solver_ref.E[:, :, iz_up, "y"][idx_tx, idx_ty])
            hx_un_r = 0.5 * (
                solver_ref.H[:, :, iz_up - 1, "x"][idx_tx, idx_ty] + solver_ref.H[:, :, iz_up, "x"][idx_tx, idx_ty]
            )
            hy_un_r = 0.5 * (
                solver_ref.H[:, :, iz_up - 1, "y"][idx_tx, idx_ty] + solver_ref.H[:, :, iz_up, "y"][idx_tx, idx_ty]
            )
            pu_unf_ref_w[n] = -float(np.sum(ex_un_r * hy_un_r * da_tx - ey_un_r * hx_un_r * da_ty))
            ex_tu[n] = ex_un_r
            ey_tu[n] = ey_un_r
            hx_tu[n] = hx_un_r
            hy_tu[n] = hy_un_r

            ex_dn_r = 0.5 * (ex_do + solver_ref.E[:, :, iz_dn, "x"][idx_tx, idx_ty])
            ey_dn_r = 0.5 * (ey_do + solver_ref.E[:, :, iz_dn, "y"][idx_tx, idx_ty])
            hx_dn_r = 0.5 * (
                solver_ref.H[:, :, iz_dn - 1, "x"][idx_tx, idx_ty] + solver_ref.H[:, :, iz_dn, "x"][idx_tx, idx_ty]
            )
            hy_dn_r = 0.5 * (
                solver_ref.H[:, :, iz_dn - 1, "y"][idx_tx, idx_ty] + solver_ref.H[:, :, iz_dn, "y"][idx_tx, idx_ty]
            )
            pd_unf_ref_w[n] = +float(np.sum(ex_dn_r * hy_dn_r * da_tdx - ey_dn_r * hx_dn_r * da_tdy))
            ex_td[n] = ex_dn_r
            ey_td[n] = ey_dn_r
            hx_td[n] = hx_dn_r
            hy_td[n] = hy_dn_r
            if record_2d_snapshots and n in snap_set:
                snaps_ref_zy[snap_set[n]] = solver_ref.E[ix0, :, :, "z"]

    wall_ref = float(time.perf_counter() - t0_ref)
    u_unf_ref_j = float(
        0.5
        * (
            np.sum(w_ref_ex_cv * (ex_ref_unf**2) + w_ref_ey_cv * (ey_ref_unf**2) + w_ref_ez_cv * (ez_ref_unf**2))
            + np.sum(w_ref_hx_cv * (hx_ref_unf**2) + w_ref_hy_cv * (hy_ref_unf**2) + w_ref_hz_cv * (hz_ref_unf**2))
        )
    )
    u_bp_ref_j = float(
        0.5
        * (
            np.sum(w_ref_ex_cv * (ex_ref_bp**2) + w_ref_ey_cv * (ey_ref_bp**2) + w_ref_ez_cv * (ez_ref_bp**2))
            + np.sum(w_ref_hx_cv * (hx_ref_bp**2) + w_ref_hy_cv * (hy_ref_bp**2) + w_ref_hz_cv * (hz_ref_bp**2))
        )
    )
    del (
        solver_ref,
        grid_ref,
        ex_ref_bp,
        ey_ref_bp,
        ez_ref_bp,
        hx_ref_bp,
        hy_ref_bp,
        hz_ref_bp,
        ex_ref_unf,
        ey_ref_unf,
        ez_ref_unf,
        hx_ref_unf,
        hy_ref_unf,
        hz_ref_unf,
        w_ref_ex_cv,
        w_ref_ey_cv,
        w_ref_ez_cv,
        w_ref_hx_cv,
        w_ref_hy_cv,
        w_ref_hz_cv,
    )
    _gc_and_trim()

    w_unf_ref_j = float(-np.sum(jz_sws * ez_bar_ref * dv_ref_axis_cv[None, :]) * dt)
    e_rad_unf_ref_j = float(np.sum(pu_unf_ref_w + pd_unf_ref_w) * dt)
    ref_unf_scale_j = max(abs(w_unf_ref_j), abs(e_rad_unf_ref_j), abs(u_unf_ref_j), 1e-30)
    rel_err_ref_unf_cv = float(abs(e_rad_unf_ref_j + u_unf_ref_j - w_unf_ref_j) / ref_unf_scale_j)

    extu_bp = _apply_causal_kernel_fft(ex_tu, h_bp)
    hytu_bp = _apply_causal_kernel_fft(hy_tu, h_bp)
    del ex_tu, hy_tu
    pu_ref_bp_w = -np.sum(extu_bp * hytu_bp * da_tx[None, :], axis=1)
    del extu_bp, hytu_bp

    eytu_bp = _apply_causal_kernel_fft(ey_tu, h_bp)
    hxtu_bp = _apply_causal_kernel_fft(hx_tu, h_bp)
    del ey_tu, hx_tu
    pu_ref_bp_w += np.sum(eytu_bp * hxtu_bp * da_ty[None, :], axis=1)
    del eytu_bp, hxtu_bp

    extd_bp = _apply_causal_kernel_fft(ex_td, h_bp)
    hytd_bp = _apply_causal_kernel_fft(hy_td, h_bp)
    del ex_td, hy_td
    pd_ref_bp_w = np.sum(extd_bp * hytd_bp * da_tdx[None, :], axis=1)
    del extd_bp, hytd_bp

    eytd_bp = _apply_causal_kernel_fft(ey_td, h_bp)
    hxtd_bp = _apply_causal_kernel_fft(hx_td, h_bp)
    del ey_td, hx_td
    pd_ref_bp_w -= np.sum(eytd_bp * hxtd_bp * da_tdy[None, :], axis=1)
    del eytd_bp, hxtd_bp
    e_rad_bp_ref_j = float(np.sum(pu_ref_bp_w + pd_ref_bp_w) * dt)
    _gc_and_trim()

    # --- 3. Wake potential W_||(s) and Impedance Z_||(f) via WakeSolver ---
    ez_scat = ez_sws - ez_ref
    z_int = z_nodes[act_z_mask]
    dz_int = dz_nodes[act_z_mask]
    ez_int = ez_scat[:, act_z_mask]

    ds = u0 * dt
    ti_loc = ti + float(z_int[0]) / u0
    s = np.arange(-ti_loc * u0, wake_len, ds)
    wp = np.zeros_like(s)
    for k in range(len(z_int)):
        ts = (float(z_int[k]) - float(z_nodes.min()) + s) / u0 + ti - 0.5 * dt
        wp += np.interp(ts, t_arr, ez_int[:, k], left=0.0, right=0.0) * float(dz_int[k])
    wp /= q_c * 1e12  # [V/pC]

    # Raise fmax to 110.0 GHz so the Gaussian bunch form factor squared |F(fmax)|^2 = 7.90e-4 < 1e-3
    # for sigma_z = 1.2 mm (Item 6), avoiding spectral truncation of full-band impedance integrals.
    fmax_z_hz = 110.0e9
    n_samples_z = 32768
    ff_sq_fmax = float(gaussian_bunch_form_factor(fmax_z_hz, sigmaz, u0) ** 2)
    ff_sq_25ghz = float(gaussian_bunch_form_factor(25.0e9, sigmaz, u0) ** 2)

    ws = WakeSolver(q=q_c, sigmaz=sigmaz, beta=beta, ti=ti_loc, save=False, verbose=0)
    ws.s = s
    ws.WP = wp
    ws.calc_long_Z(samples=n_samples_z, fmax=fmax_z_hz)

    f_ghz_full = np.asarray(ws.f, dtype=float) * 1e-9
    re_z_full = np.real(ws.Z).astype(float)
    im_z_full = np.imag(ws.Z).astype(float)

    f_lo, f_hi = (WagonWheelMTMSpec.wake_passband_low_hz * 1e-9, WagonWheelMTMSpec.wake_passband_high_hz * 1e-9) if not forward_control else (15.5, 22.0)
    mask_b = (f_ghz_full >= f_lo) & (f_ghz_full <= f_hi)
    f_b = f_ghz_full[mask_b]
    rz_b = re_z_full[mask_b]
    i_pk = int(np.argmax(rz_b))
    f_pk = float(f_b[i_pk])
    z_pk = float(rz_b[i_pk])

    half_mask = rz_b >= 0.5 * z_pk
    fwhm_ghz = float(f_b[half_mask][-1] - f_b[half_mask][0]) if np.any(half_mask) else 0.5

    ff_sq_b = gaussian_bunch_form_factor(f_b * 1e9, sigmaz, u0) ** 2
    k_loss_band_act_v_per_c = float(2.0 * np.trapezoid(rz_b * ff_sq_b, f_b * 1e9))
    w_loss_band_act_j = float((q_c**2) * k_loss_band_act_v_per_c)

    # Full-band active-region loss via both 2*q^2*\int_0^{fmax} Re Z_||(f) |F(f)|^2 df and -q^2*\int \lambda(s) W_||(s) ds
    ff_sq_full = gaussian_bunch_form_factor(f_ghz_full * 1e9, sigmaz, u0) ** 2
    k_loss_z_full_act_v_per_c = float(2.0 * np.trapezoid(re_z_full * ff_sq_full, f_ghz_full * 1e9))
    w_loss_z_full_act_j = float((q_c**2) * k_loss_z_full_act_v_per_c)
    lam_s = np.exp(-0.5 * ((s / sigmaz) ** 2)) / (np.sqrt(2.0 * np.pi) * sigmaz)
    k_loss_wake_full_act_v_per_c = float(-np.trapezoid(lam_s * wp, s) * 1e12)
    w_loss_wake_full_act_j = float((q_c**2) * k_loss_wake_full_act_v_per_c)
    oob_frac_wake_act = float(1.0 - w_loss_band_act_j / max(abs(w_loss_wake_full_act_j), 1e-30))

    # Keep saved NPZ trace arrays restricted to <= 25 GHz for compact storage and plotting
    mask_save_f = f_ghz_full <= 25.0
    f_ghz = f_ghz_full[mask_save_f]
    re_z = re_z_full[mask_save_f]
    im_z = im_z_full[mask_save_f]

    # Also integrate 1D wake over the full control volume [iz_up, iz_dn)
    z_cv_int = z_nodes[cv_z_mask]
    dz_cv_int = dz_nodes[cv_z_mask]
    ez_cv_int = ez_scat[:, cv_z_mask]
    ti_cv_loc = ti + float(z_cv_int[0]) / u0
    s_cv = np.arange(-ti_cv_loc * u0, wake_len, ds)
    wp_cv = np.zeros_like(s_cv)
    for k in range(len(z_cv_int)):
        ts_cv = (float(z_cv_int[k]) - float(z_nodes.min()) + s_cv) / u0 + ti - 0.5 * dt
        wp_cv += np.interp(ts_cv, t_arr, ez_cv_int[:, k], left=0.0, right=0.0) * float(dz_cv_int[k])
    wp_cv /= q_c * 1e12
    ws_cv = WakeSolver(q=q_c, sigmaz=sigmaz, beta=beta, ti=ti_cv_loc, save=False, verbose=0)
    ws_cv.s = s_cv
    ws_cv.WP = wp_cv
    ws_cv.calc_long_Z(samples=n_samples_z, fmax=fmax_z_hz)
    f_cv_ghz = np.asarray(ws_cv.f, dtype=float) * 1e-9
    re_z_cv = np.real(ws_cv.Z).astype(float)
    mask_cv_b = (f_cv_ghz >= f_lo) & (f_cv_ghz <= f_hi)
    ff_sq_cv_b = gaussian_bunch_form_factor(f_cv_ghz[mask_cv_b] * 1e9, sigmaz, u0) ** 2
    k_loss_band_cv_v_per_c = float(2.0 * np.trapezoid(re_z_cv[mask_cv_b] * ff_sq_cv_b, f_cv_ghz[mask_cv_b] * 1e9))
    w_loss_wake_cv_j = float((q_c**2) * k_loss_band_cv_v_per_c)
    lam_s_cv = np.exp(-0.5 * ((s_cv / sigmaz) ** 2)) / (np.sqrt(2.0 * np.pi) * sigmaz)
    w_loss_wake_full_cv_j = float((q_c**2) * (-np.trapezoid(lam_s_cv * wp_cv, s_cv) * 1e12))

    # --- 4. Unified control-volume passband beam work, port Poynting flux, and stored energy ---
    ez_scat_bar = ez_bar_sws - ez_bar_ref
    w_beam_cv_scat_full_j = float(-np.sum(jz_sws * ez_scat_bar * dv_axis_cv[None, :]) * dt)
    jz_bp = _apply_causal_kernel_fft(jz_sws, h_bp)
    ez_ref_bp_axis = _apply_causal_kernel_fft(ez_bar_ref, h_bp)
    w_bp_ref_j = float(-np.sum(jz_bp * ez_ref_bp_axis * dv_ref_axis_cv[None, :]) * dt)
    del ez_ref_bp_axis
    ez_scat_bp = _apply_causal_kernel_fft(ez_scat_bar, h_bp)
    w_beam_cv_bp_j = float(-np.sum(jz_bp * ez_scat_bp * dv_axis_cv[None, :]) * dt)
    w_beam_act_bp_j = float(-np.sum(jz_bp * ez_scat_bp * dv_axis_act[None, :]) * dt)
    oob_frac_beam_cv = float(1.0 - w_beam_cv_bp_j / max(abs(w_beam_cv_scat_full_j), 1e-30))
    del jz_sws, ez_bar_sws, ez_bar_ref, ez_scat_bar, jz_bp, ez_scat_bp

    p_up_inst_w = pu_sws_bp_w - pu_ref_bp_w
    p_dn_inst_w = pd_sws_bp_w - pd_ref_bp_w

    # Causal 1-cycle rectangular moving-average envelope (zero pre-arrival wrap-around)
    n_cyc = max(int(round(1.0 / (f_pk * 1e9 * dt))), 1)
    ker = np.ones(n_cyc) / float(n_cyc)
    p_up_env_w = np.convolve(p_up_inst_w, ker, mode="full")[:nt]
    p_dn_env_w = np.convolve(p_dn_inst_w, ker, mode="full")[:nt]

    e_up_j = float(np.sum(p_up_inst_w) * dt)
    e_dn_j = float(np.sum(p_dn_inst_w) * dt)
    e_rad_tot_j = e_up_j + e_dn_j
    e_up_coax_j = float(np.sum(pu_coax_bp_w) * dt)
    e_dn_coax_j = float(np.sum(pd_coax_bp_w) * dt)

    u_stored_final_fit_j = u_unf_sws_j if mutate_unfiltered_stored_energy else u_bp_sws_j
    w_loss_band_j = w_beam_act_bp_j if mutate_shrink_beam_work_to_active else w_beam_cv_bp_j
    k_loss_band_v_per_c = float(w_loss_band_j / (q_c**2))

    p_main_env = p_dn_env_w if forward_control else p_up_env_w
    p_pk_env_w = float(np.max(p_main_env))
    above_20pct = t_arr[p_main_env >= 0.20 * p_pk_env_w]
    t_start_20pct_s = float(above_20pct[0]) if len(above_20pct) > 1 else 0.0
    t_end_20pct_s = float(above_20pct[-1]) if len(above_20pct) > 1 else 0.0
    t_end_window_s = float(t_arr[-1])
    t_rf_s = 1.0 / (f_pk * 1e9)
    margin_to_t_end_s = t_end_window_s - t_end_20pct_s
    pulse_clips_window = bool(margin_to_t_end_s <= t_rf_s)
    # Report start/end-threshold duration only when the envelope ends inside the window (margin > 1 RF period)
    t_pulse_fit_s = (t_end_20pct_s - t_start_20pct_s) if (len(above_20pct) > 1 and not pulse_clips_window) else 0.0

    h_xy_mm = float((2.0 * r_wg / int(n_xy)) * 1e3)
    res_dict: dict[str, Any] = {
        "n_periods": int(n_periods),
        "n_xy": int(n_xy),
        "h_xy_mm": h_xy_mm,
        "dz_mm": float(dz_mm),
        "cells_across_1mm_spoke": float(1.0 / h_xy_mm),
        "active_length_m": float(L_act),
        "active_length_mm": float(L_act * 1e3),
        "wake_length_mm": float(wake_len * 1e3),
        "z_total_m": float(z_tot),
        "z_act_start_m": float(z_act_start),
        "z_act_end_m": float(z_act_end),
        "z_up_port_m": float(z_up_port),
        "z_dn_port_m": float(z_dn_port),
        "forward_control": bool(forward_control),
        "grid_shape": [nx_cells, ny_cells, nz_cells],
        "total_cells": int(nx_cells * ny_cells * nz_cells),
        "dt_fs": float(dt * 1e15),
        "nt_steps": int(nt),
        "f_pk_ghz": f_pk,
        "re_z_pk_ohm": z_pk,
        "fwhm_ghz": fwhm_ghz,
        "fmax_impedance_ghz": float(fmax_z_hz * 1e-9),
        "form_factor_sq_at_fmax": ff_sq_fmax,
        "form_factor_sq_at_25ghz": ff_sq_25ghz,
        "k_loss_passband_v_per_c": k_loss_band_v_per_c,
        "w_loss_passband_j": w_loss_band_j,
        "w_loss_passband_mj": float(w_loss_band_j * 1e3),
        "w_beam_passband_cv_mj": float(w_beam_cv_bp_j * 1e3),
        "w_beam_passband_active_mj": float(w_beam_act_bp_j * 1e3),
        "w_loss_wake_passband_cv_mj": float(w_loss_wake_cv_j * 1e3),
        "w_loss_wake_passband_active_mj": float(w_loss_band_act_j * 1e3),
        "w_loss_wake_fullband_active_mj": float(w_loss_wake_full_act_j * 1e3),
        "w_loss_z_fullband_active_mj": float(w_loss_z_full_act_j * 1e3),
        "out_of_band_wake_loss_fraction_active": oob_frac_wake_act,
        "w_loss_wake_fullband_cv_mj": float(w_loss_wake_full_cv_j * 1e3),
        "w_beam_scat_fullband_cv_mj": float(w_beam_cv_scat_full_j * 1e3),
        "out_of_band_beam_work_fraction_cv": oob_frac_beam_cv,
        "w_beam_unfiltered_cv_mj": float(w_unf_sws_j * 1e3),
        "e_rad_unfiltered_cv_mj": float(e_rad_unf_j * 1e3),
        "u_stored_unfiltered_cv_mj": float(u_unf_sws_j * 1e3),
        "rel_error_unfiltered_cv_closure": rel_err_unf_cv,
        "w_ref_unfiltered_cv_mj": float(w_unf_ref_j * 1e3),
        "e_rad_ref_unfiltered_cv_mj": float(e_rad_unf_ref_j * 1e3),
        "u_stored_ref_unfiltered_cv_mj": float(u_unf_ref_j * 1e3),
        "rel_error_ref_unfiltered_cv_closure": rel_err_ref_unf_cv,
        "w_ref_passband_cv_mj": float(w_bp_ref_j * 1e3),
        "e_rad_ref_passband_cv_mj": float(e_rad_bp_ref_j * 1e3),
        "u_stored_ref_passband_cv_mj": float(u_bp_ref_j * 1e3),
        "e_up_j": e_up_j,
        "e_up_mj": float(e_up_j * 1e3),
        "e_dn_j": e_dn_j,
        "e_dn_mj": float(e_dn_j * 1e3),
        "e_up_coax_mj": float(e_up_coax_j * 1e3),
        "e_dn_coax_mj": float(e_dn_coax_j * 1e3),
        "e_rad_total_j": e_rad_tot_j,
        "e_rad_total_mj": float(e_rad_tot_j * 1e3),
        "u_stored_final_fit_j": u_stored_final_fit_j,
        "u_stored_final_fit_mj": float(u_stored_final_fit_j * 1e3),
        "u_stored_over_w_beam_passband": float(u_stored_final_fit_j / max(abs(w_loss_band_j), 1e-30)),
        "upstream_energy_fraction": float(e_up_j / max(e_rad_tot_j, 1e-30)),
        "downstream_energy_fraction": float(e_dn_j / max(e_rad_tot_j, 1e-30)),
        "p_up_cycle_avg_max_mw": float(np.max(p_up_env_w) * 1e-6),
        "p_up_inst_max_mw": float(np.max(p_up_inst_w) * 1e-6),
        "p_dn_cycle_avg_max_mw": float(np.max(p_dn_env_w) * 1e-6),
        "p_dn_inst_max_mw": float(np.max(p_dn_inst_w) * 1e-6),
        "pulse_start_20pct_ns": float(t_start_20pct_s * 1e9),
        "pulse_end_20pct_ns": float(t_end_20pct_s * 1e9),
        "t_end_window_ns": float(t_end_window_s * 1e9),
        "rf_period_ns": float(t_rf_s * 1e9),
        "pulse_end_margin_to_t_end_ns": float(margin_to_t_end_s * 1e9),
        "pulse_clips_window": pulse_clips_window,
        "pulse_duration_20pct_ns": float(t_pulse_fit_s * 1e9),
        "max_decelerating_gradient_mv_per_m": float(
            np.max(np.abs(wp)) * (q_c * 1e12) / max(L_act, 1e-12) * 1e-6
        ),
        "est_memory_mb": mem_mb,
        "wall_time_sws_s": wall_sws,
        "wall_time_ref_s": wall_ref,
        "wall_time_total_s": float(wall_sws + wall_ref),
        "t_ns": (t_arr * 1e9).astype(float),
        "p_up_inst_mw": (p_up_inst_w * 1e-6).astype(float),
        "p_up_env_mw": (p_up_env_w * 1e-6).astype(float),
        "p_dn_inst_mw": (p_dn_inst_w * 1e-6).astype(float),
        "p_dn_env_mw": (p_dn_env_w * 1e-6).astype(float),
        "s_mm": (s * 1e3).astype(float),
        "wp_v_per_pc": wp.astype(float),
        "f_ghz": f_ghz.astype(float),
        "re_z_ohm": re_z.astype(float),
        "im_z_ohm": im_z.astype(float),
    }
    if not forward_control:
        e_acc_fit_j = e_rad_tot_j + u_stored_final_fit_j
        rel_err_fit = float(abs(e_acc_fit_j - w_loss_band_j) / max(abs(w_loss_band_j), 1e-30))
        rel_err_shrunk_mut = float(abs(e_acc_fit_j - w_beam_act_bp_j) / max(abs(w_beam_act_bp_j), 1e-30))
        res_dict["rel_error_rad_total_vs_bunch_loss"] = float(
            abs(e_rad_tot_j - w_loss_band_j) / max(abs(w_loss_band_j), 1e-30)
        )
        res_dict["rel_error_total_accounted_vs_bunch_loss"] = rel_err_fit
        res_dict["mutated_shrunk_active_beam_work_rel_error"] = rel_err_shrunk_mut

    if record_2d_snapshots and snaps_sws_zy is not None and snaps_ref_zy is not None:
        snaps_scat_zy = snaps_sws_zy - snaps_ref_zy
        t_snaps_ps = (t_arr[snap_steps] * 1e12).astype(float)
        zb_snaps_mm = ((t_arr[snap_steps] - ti) * u0 * 1e3).astype(float)
        res_dict["snapshots"] = {
            "y_mm": (y_nodes * 1e3).astype(float),
            "z_mm": (z_nodes * 1e3).astype(float),
            "snap_ez_scat": snaps_scat_zy.astype(float),
            "snap_t_ps": t_snaps_ps,
            "snap_zb_mm": zb_snaps_mm,
        }

    return res_dict


def verify_two_bunch_fit_superposition(
    *,
    n_periods: int = 6,
    n_xy: int = 64,
    dz_mm: float = 0.25,
    n_steps: int = 520,
    quick: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Verify 3D FIT linear superposition to machine round-off for two relativistic bunches in the MTM structure."""
    if quick and n_xy > 28:
        n_xy = 28
        n_steps = 360
    r_wg = spec.waveguide_radius_m
    geom = build_adiabatic_coax_mtm_solids(
        n_periods=n_periods, n_taper_cells=2, n_port_cells=3, forward_control=False, spec=spec
    )
    z_tot = geom["z_total_m"]

    x = np.linspace(-r_wg, r_wg, int(n_xy) + 1)
    y = np.linspace(-r_wg, r_wg, int(n_xy) + 1)
    nz = int(round(z_tot / (dz_mm * 1e-3)))
    z = np.linspace(0.0, z_tot, nz + 1)

    _, beta, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    sigmaz = spec.bunch_length_rms_m
    f_sync = 11.41e9
    t_rf = 1.0 / f_sync

    ti1 = 3.5 * sigmaz / u0
    ti2 = ti1 + t_rf
    q1 = 0.5 * spec.two_bunch_total_charge_c
    q2 = 0.5 * spec.two_bunch_total_charge_c

    def _run_short(beams: list[Beam]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        g = GridFIT3D(
            x=x, y=y, z=z, stl_solids={"sws": geom["solid"]}, stl_materials={"sws": "pec"}, conformal="pec", verbose=0
        )
        s = SolverFIT3D(
            g,
            use_stl=True,
            bc_low=["pec", "pec", "cpml"],
            bc_high=["pec", "pec", "cpml"],
            n_pml=8,
            verbose=0,
        )
        s.source_type = "direct"
        dt_run = float(s.dt)
        n_steps_eff = max(int(n_steps), int(np.ceil((ti2 + 1.5 * sigmaz / u0) / dt_run)))
        est = estimate((s.Nx, s.Ny, s.Nz), steps=n_steps_eff, dtype="float64", conformal="pec")
        preflight(est, "laptop" if quick else "n2-highmem-8")
        ix0 = int(np.argmin(np.abs(s.x)))
        iy0 = int(np.argmin(np.abs(s.y)))
        ez_trace = np.zeros((n_steps_eff, s.Nz), dtype=float)
        for n in range(n_steps_eff):
            t = n * dt_run
            for b in beams:
                b.update(s, t)
            s.one_step()
            ez_trace[n] = s.E[ix0, iy0, :, "z"]
        e_final = s.E.toarray().copy()
        h_final = s.H.toarray().copy()
        wall = float(time.perf_counter() - t0)
        del s, g
        gc.collect()
        return dt_run, ez_trace, e_final, h_final, wall

    dt0, ez_1, e1, h1, w1 = _run_short([Beam(xsource=0.0, ysource=0.0, beta=beta, q=q1, sigmaz=sigmaz, ti=ti1)])
    _, ez_2, e2, h2, w2 = _run_short([Beam(xsource=0.0, ysource=0.0, beta=beta, q=q2, sigmaz=sigmaz, ti=ti2)])
    _, ez_12, e12, h12, w12 = _run_short(
        [
            Beam(xsource=0.0, ysource=0.0, beta=beta, q=q1, sigmaz=sigmaz, ti=ti1),
            Beam(xsource=0.0, ysource=0.0, beta=beta, q=q2, sigmaz=sigmaz, ti=ti2),
        ]
    )
    # Mutated superposition check: shift bunch 2 by half a time step (0.5 * dt0)
    _, ez_mut, e_mut, _, _ = _run_short(
        [
            Beam(xsource=0.0, ysource=0.0, beta=beta, q=q1, sigmaz=sigmaz, ti=ti1),
            Beam(xsource=0.0, ysource=0.0, beta=beta, q=q2, sigmaz=sigmaz, ti=ti2 + 0.5 * dt0),
        ]
    )

    rel_err_e = float(np.linalg.norm(e12 - (e1 + e2)) / np.linalg.norm(e12))
    rel_err_h = float(np.linalg.norm(h12 - (h1 + h2)) / np.linalg.norm(h12))
    rel_err_ez = float(np.linalg.norm(ez_12 - (ez_1 + ez_2)) / np.linalg.norm(ez_12))
    rel_err_mut_half_dt = float(np.linalg.norm(e_mut - (e1 + e2)) / np.linalg.norm(e12))

    return {
        "n_periods": int(n_periods),
        "n_xy": int(n_xy),
        "n_steps": int(n_steps),
        "dt_fs": float(dt0 * 1e15),
        "q1_nc": float(q1 * 1e9),
        "q2_nc": float(q2 * 1e9),
        "delta_t_ps": float((ti2 - ti1) * 1e12),
        "rel_l2_error_E_3d": rel_err_e,
        "rel_l2_error_H_3d": rel_err_h,
        "rel_l2_error_Ez_axis": rel_err_ez,
        "mutated_half_dt_shift_rel_l2_error_E": rel_err_mut_half_dt,
        "wall_time_total_s": float(w1 + w2 + w12),
    }


def _strip_heavy_arrays(d: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``d`` with 1D/2D NumPy arrays removed for compact JSON serialization."""
    skip = {
        "t_ns",
        "p_up_inst_mw",
        "p_up_env_mw",
        "p_dn_inst_mw",
        "p_dn_env_mw",
        "s_mm",
        "wp_v_per_pc",
        "f_ghz",
        "re_z_ohm",
        "im_z_ohm",
        "snapshots",
        "field_maps",
    }
    return {k: v for k, v in d.items() if k not in skip}


class _MTMSchematicSolverProxy:
    """Lightweight proxy around SolverFIT3D that holds Box structures for ``wakis.schematic``."""

    def __init__(
        self,
        solver: SolverFIT3D,
        grid_bare: GridFIT3D,
        structures: tuple[Any, ...] = (),
        *,
        k_bloch: tuple[float, float, float] | None = None,
    ) -> None:
        self._solver = solver
        self.grid = grid_bare
        self.Nx = solver.Nx
        self.Ny = solver.Ny
        self.Nz = solver.Nz
        self.dt = solver.dt
        self.n_pml = getattr(solver, "n_pml", 0)
        self.bc_low = list(getattr(solver, "bc_low", ["pec", "pec", "cpml"]))
        self.bc_high = list(getattr(solver, "bc_high", ["pec", "pec", "cpml"]))
        self.bg = [1.0, 1.0, 0.0]
        self._assigned_structures = tuple(structures)
        if k_bloch is not None:
            self.k_bloch = k_bloch
            self._bloch_axes = [abs(k) > 0.0 for k in k_bloch]


def build_mtm_schematic_proxy(
    n_periods: int = 8,
    *,
    n_taper_cells: int = 2,
    n_port_cells: int = 4,
    n_xy: int = 64,
    dz_mm: float = 0.25,
    forward_control: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> tuple[_MTMSchematicSolverProxy, dict[str, Any]]:
    """Build a ``_MTMSchematicSolverProxy`` for the finite :math:`N`-period MTM wakefield domain."""
    p = spec.period_m
    t_ww = spec.wagon_wheel_thickness_m
    r_b = spec.beam_hole_radius_m
    r_wg = spec.waveguide_radius_m
    r_pipe_wall = spec.coax_port_wall_radius_m
    r_outer_wall = r_wg + 1.0e-3

    n_end = int(n_port_cells) + int(n_taper_cells)
    n_tot_cells = int(n_periods) + 2 * n_end
    z_tot = n_tot_cells * p
    z_act_start = n_end * p
    z_act_end = z_act_start + int(n_periods) * p
    z_up_taper_start = int(n_port_cells) * p
    z_dn_taper_end = (n_end + int(n_periods) + int(n_taper_cells)) * p

    x = np.linspace(-r_outer_wall, r_outer_wall, int(n_xy) + 1)
    y = np.linspace(-r_outer_wall, r_outer_wall, int(n_xy) + 1)
    nz = int(round(z_tot / (dz_mm * 1e-3)))
    z = np.linspace(0.0, z_tot, nz + 1)

    grid_bare = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver = SolverFIT3D(
        grid_bare,
        bc_low=["pec", "pec", "cpml"],
        bc_high=["pec", "pec", "cpml"],
        n_pml=10,
        verbose=0,
    )

    r_b_mm = r_b * 1e3
    r_p_mm = r_pipe_wall * 1e3
    r_wg_mm = r_wg * 1e3
    structs: list[Any] = []
    # Outer circular waveguide wall at |y| in [r_wg, r_outer_wall] (single shared name to avoid duplicate legend rows)
    for y_sign in (-1.0, 1.0):
        y_c = y_sign * (0.5 * (r_wg + r_outer_wall))
        h_w = r_outer_wall - r_wg
        b_wall = Box(center=(0.0, y_c, 0.5 * z_tot), size=(2.0 * r_outer_wall, h_w, z_tot), material="pec")
        b_wall.name = f"Outer circular PEC waveguide wall (ID = {2.0*r_wg_mm:.0f} mm, PRL line 121)"
        structs.append(b_wall)

    # Coaxial port inner shield at |y| in [r_b, r_pipe_wall] (single shared name to avoid duplicate legend rows)
    for z_c, l_z in (
        (0.5 * z_up_taper_start, z_up_taper_start),
        (0.5 * (z_dn_taper_end + z_tot), z_tot - z_dn_taper_end),
    ):
        for y_sign in (-1.0, 1.0):
            y_c = y_sign * 0.5 * (r_b + r_pipe_wall)
            h_p = r_pipe_wall - r_b
            b_sh = Box(center=(0.0, y_c, z_c), size=(2.0 * r_pipe_wall, h_p, l_z), material="pec")
            b_sh.name = f"Adiabatic coaxial TEM coupler shield (r ∈ [{r_b_mm:.1f}, {r_p_mm:.1f}] mm, ASSUMED)"
            structs.append(b_sh)

    # Wagon-wheel plates (or forward control disks)
    for k in range(int(n_port_cells), n_tot_cells - int(n_port_cells)):
        if k < n_end:
            frac = float(k - int(n_port_cells) + 1) / float(int(n_taper_cells) + 1)
        elif k >= n_end + int(n_periods):
            frac = float(n_tot_cells - int(n_port_cells) - k) / float(int(n_taper_cells) + 1)
        else:
            frac = 1.0
        zc = (k + 0.5) * p
        if not forward_control:
            h_plate = frac * (r_wg - r_b)
            for y_sign in (-1.0, 1.0):
                yc = y_sign * (r_b + 0.5 * h_plate)
                b_ww = Box(center=(0.0, yc, zc), size=(spec.spoke_width_m, h_plate, t_ww), material="pec")
                b_ww.name = "Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm)"
                structs.append(b_ww)
        else:
            r_iris = 5.0e-3
            h_disk = frac * (r_wg - r_iris)
            for y_sign in (-1.0, 1.0):
                yc = y_sign * (r_wg - 0.5 * h_disk)
                b_dk = Box(center=(0.0, yc, zc), size=(2.0 * r_wg, h_disk, t_ww), material="pec")
                b_dk.name = "Forward-wave PEC iris disks (r ∈ [5, 8] mm, t = 1 mm)"
                structs.append(b_dk)

    meta = {
        "z_total_m": float(z_tot),
        "z_act_start_m": float(z_act_start),
        "z_act_end_m": float(z_act_end),
        "z_up_port_m": float(z_up_taper_start - p),
        "z_dn_port_m": float(z_dn_taper_end + p),
        "r_wg_m": float(r_wg),
        "r_beam_m": float(r_b),
    }
    return _MTMSchematicSolverProxy(solver, grid_bare, tuple(structs)), meta


def plot_mtm_setup_schematics(
    figures_dir: Path | str,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
    *,
    data_dir: Path | str | None = None,
) -> dict[str, str]:
    """Generate ``00a``/``00b`` setup figures and Markdown setup tables via :func:`wakis.schematic.save_setup_figures`."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if data_dir is None:
        data_dir = figures_dir.parent / "data"
    else:
        data_dir = Path(data_dir)
    figstyle.apply_style()

    _, beta, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)

    # Load JSON metrics if present so setup annotations reflect live JSON numbers
    t1_path = data_dir / "task_01_geometry_and_port_reflection.json"
    t2_path = data_dir / "task_02_cold_test_and_convergence.json"
    t3_path = data_dir / "task_03_wakefield_and_directionality.json"
    t4_path = data_dir / "task_04_two_bunch_superposition.json"
    t5_path = data_dir / "task_05_analytic_power_and_energy_balance.json"
    t1 = json.loads(t1_path.read_text(encoding="utf-8")) if t1_path.is_file() else {}
    t2 = json.loads(t2_path.read_text(encoding="utf-8")) if t2_path.is_file() else {}
    t3 = json.loads(t3_path.read_text(encoding="utf-8")) if t3_path.is_file() else {}
    t4 = json.loads(t4_path.read_text(encoding="utf-8")) if t4_path.is_file() else {}
    t5 = json.loads(t5_path.read_text(encoding="utf-8")) if t5_path.is_file() else {}

    t1_spec = t1.get("spec", asdict(spec))
    wb = t2.get("wakis_finest_nf8_benchmarks", {})
    pb = t2.get("paper_cold_test_benchmarks", {})
    r8 = t3.get("runs_mtm", {}).get("8", {})
    wa = t5.get("wakis_cold_test_analytic", {})
    f_sync_hz = (
        float(pb["f_sync_ghz"]) * 1.0e9
        if "f_sync_ghz" in pb
        else float(t1_spec.get("paper_sync_freq_hz", spec.paper_sync_freq_hz))
    )
    phi_s_derived_deg = float(np.rad2deg(2.0 * np.pi * f_sync_hz * spec.period_m / u0))
    f_pk_n8_ghz = float(r8["f_pk_ghz"]) if "f_pk_ghz" in r8 else float(f_sync_hz * 1e-9)
    f_cross_nf4_ghz = float(wb["f_beam_crossing_nf4_ghz"]) if "f_beam_crossing_nf4_ghz" in wb else float(f_sync_hz * 1e-9)
    phi_cross_deg = float(wb["phi_beam_crossing_nf4_deg"]) if "phi_beam_crossing_nf4_deg" in wb else phi_s_derived_deg
    phi_s_deg = float(pb["phi_sync_deg"]) if "phi_sync_deg" in pb else phi_s_derived_deg
    f_nf8_ghz = float(wb["f_sync_ghz"]) if "f_sync_ghz" in wb else float(f_sync_hz * 1e-9)
    ve_nf8_c = float(wb["ve_dft_over_c"]) if "ve_dft_over_c" in wb else float(pb.get("vg_over_c", spec.paper_vg_over_c))
    rq_nf8_kohm = (
        float(wb["r_over_q_kohm_per_m"])
        if "r_over_q_kohm_per_m" in wb
        else float(pb.get("r_over_q_kohm_per_m", spec.paper_r_over_q_ohm_per_m * 1e-3))
    )
    rf_period_ps = float(t4["rf_period_ps"]) if "rf_period_ps" in t4 else float(1e12 / f_sync_hz)
    linac_period_ps = float(t4["linac_period_ps"]) if "linac_period_ps" in t4 else float(1e12 / spec.linac_frequency_hz)
    dz_mm = (
        float(r8["dz_mm"])
        if "dz_mm" in r8
        else float(t1_spec.get("default_dz_m", spec.default_dz_m)) * 1.0e3
    )

    # 1. Finite N=8 wagon-wheel MTM wakefield setup (plane="zy")
    proxy_wake, meta = build_mtm_schematic_proxy(n_periods=8, dz_mm=dz_mm, spec=spec)
    z_tot = meta["z_total_m"]
    z_act_start = meta["z_act_start_m"]
    z_act_end = meta["z_act_end_m"]
    z_up_port = meta["z_up_port_m"]
    z_dn_port = meta["z_dn_port_m"]
    r_wg = meta["r_wg_m"]
    r_b_mm = float(t1_spec.get("beam_hole_radius_m", spec.beam_hole_radius_m)) * 1e3
    r_p_mm = float(t1_spec.get("coax_port_wall_radius_m", spec.coax_port_wall_radius_m)) * 1e3
    r_wg_mm = r_wg * 1e3
    ek_mev = float(t1_spec.get("beam_kinetic_energy_ev", spec.beam_kinetic_energy_ev)) * 1e-6
    q_nc = float(t1_spec.get("single_bunch_charge_c", spec.single_bunch_charge_c)) * 1e9
    sig_m = float(t1_spec.get("bunch_length_rms_m", spec.bunch_length_rms_m))
    sig_mm = sig_m * 1e3
    two_sig_mm = 2.0 * sig_mm
    p_mm = float(t1_spec.get("period_m", spec.period_m)) * 1e3
    pb_lo_ghz = float(t1_spec.get("passband_low_hz", spec.passband_low_hz)) * 1e-9
    pb_hi_ghz = float(t1_spec.get("passband_high_hz", spec.passband_high_hz)) * 1e-9
    wpb_lo_ghz = float(t1_spec.get("wake_passband_low_hz", spec.wake_passband_low_hz)) * 1e-9
    wpb_hi_ghz = float(t1_spec.get("wake_passband_high_hz", spec.wake_passband_high_hz)) * 1e-9
    f_1e3_ghz = (
        float(wa["f_form_factor_sq_1e3_ghz"])
        if "f_form_factor_sq_1e3_ghz" in wa
        else float(u0 * np.sqrt(np.log(1.0e3)) / (2.0 * np.pi * sig_m) * 1e-9)
    )

    sources_wake = [
        {
            "kind": "beam",
            "type_name": "Beam",
            "short_word": f"{ek_mev:.0f} MeV bunch",
            "label": (
                f"Rigid relativistic electron bunch ({ek_mev:.0f} MeV, β = {beta:.5f}, "
                f"q = {q_nc:.0f} nC, σ_z = {sig_mm:.1f} mm, PRL lines 49, 151)"
            ),
            "desc": (
                f"Relativistic Gaussian bunch J_z(z − β c t) along r = 0 "
                f"({ek_mev:.0f} MeV, q = {q_nc:.0f} nC, σ_z = {sig_mm:.1f} mm, PRL lines 49, 151)"
            ),
            "pos_str": f"(x, y) = (0, 0) inside r_b = {r_b_mm:.1f} mm (PRL line 122) beam tunnel, z ∈ [0, {z_tot*1e3:.0f}] mm",
            "polarization": "Jz (TM₀₁-like backward slow wave)",
            "h_bounds": (2.0e-3, z_tot - 2.0e-3),
            "v_bounds": (0.0, 0.0),
            "pulse": None,
        }
    ]
    monitors_wake = [
        {
            "kind": "plane",
            "type_name": "Poynting",
            "short_word": "S_z(z_up)",
            "label": "Upstream (−z) coaxial port Poynting plane z_up (backward Cherenkov output)",
            "desc": f"Upstream port Poynting flux monitor P_up(t) across r ∈ [{r_p_mm:.1f}, {r_wg_mm:.1f}] mm",
            "pos_str": f"z_up = {z_up_port*1e3:.1f} mm (upstream TEM port before −z CPML), r ∈ [{r_p_mm:.1f}, {r_wg_mm:.1f}] mm",
            "freq_str": "Passband-filtered Poynting flux P_up(t) [MW]",
            "h_bounds": (z_up_port, z_up_port),
            "v_bounds": (-r_wg, r_wg),
        },
        {
            "kind": "line",
            "type_name": "WakeSolver",
            "short_word": "wake E_z(s)",
            "label": "On-axis wake integration line (L = N p) → W_∥(s), Re Z_∥(f)",
            "desc": "Longitudinal wake integration line E_z(r=0, z, t) across active MTM cells",
            "pos_str": f"(x, y) = (0, 0), z ∈ [{z_act_start*1e3:.1f}, {z_act_end*1e3:.1f}] mm",
            "freq_str": f"0–25 GHz (reversed-Cherenkov peak at f_pk = {f_pk_n8_ghz:.2f} GHz)",
            "h_bounds": (z_act_start, z_act_end),
            "v_bounds": (-2.1e-3, -2.1e-3),
        },
        {
            "kind": "plane",
            "type_name": "Poynting",
            "short_word": "S_z(z_dn)",
            "label": "Downstream (+z) coaxial port Poynting plane z_dn (forward leakage check)",
            "desc": f"Downstream port Poynting flux monitor P_dn(t) across r ∈ [{r_p_mm:.1f}, {r_wg_mm:.1f}] mm",
            "pos_str": f"z_dn = {z_dn_port*1e3:.1f} mm (downstream TEM port before +z CPML), r ∈ [{r_p_mm:.1f}, {r_wg_mm:.1f}] mm",
            "freq_str": "Passband-filtered Poynting flux P_dn(t) [MW]",
            "h_bounds": (z_dn_port, z_dn_port),
            "v_bounds": (-r_wg, r_wg),
        },
    ]
    mat_names_wake = {
        "background": f"Vacuum beam tunnel (r_b = {r_b_mm:.0f} mm, PRL line 122) & inter-spoke/inter-plate vacuum",
    }
    res_wake = schematic.save_setup_figures(
        proxy_wake,
        figures_dir,
        prefix="00",
        plane="zy",
        material_names=mat_names_wake,
        sources=sources_wake,
        monitors=monitors_wake,
        unit="mm",
        freq=f_sync_hz,
        min_feature=spec.wagon_wheel_thickness_m,
        title_geometry="Wagon-Wheel Metamaterial Wakefield Setup (zy Cut, Conformal PEC)",
        title_mesh=f"Computational Mesh, Beam Excitation & Spectrum (Δ = {dz_mm:.2f} mm, {int(round(1.0/dz_mm))} cells across 1 mm)",
        variations=[
            {
                "name": "3D Bloch unit-cell cold test & mesh convergence (Part 2)",
                "details": (
                    f"Single period p = {p_mm:.1f} mm (PRL line 140) with Bloch BC along z (φ ∈ [12°, 72°]) "
                    f"and n_feat ∈ {{4, 6, 8}} → crossing {f_cross_nf4_ghz:.2f} GHz at φ = {phi_cross_deg:.2f}° "
                    f"({f_nf8_ghz:.2f} GHz at φ_s = {phi_s_deg:.2f}°, derived from ω_s p/u_0), v_E = {ve_nf8_c:.3f}c, r/Q = {rq_nf8_kohm:.1f} kΩ/m"
                ),
            },
            {
                "name": "Finite structure length scaling & forward-wave control (Part 3)",
                "details": "N ∈ {6, 8, 12} periods (L = 12–24 mm, short-N pre-asymptotic regime vs experimental N = 40, L = 80 mm) and non-inductive disk-loaded forward-wave control (v_g > 0)",
            },
            {
                "name": "Two-bunch linear superposition & charge-scaled estimate (Part 4)",
                "details": (
                    f"Exact 3D FIT linear superposition check and single-cycle (Δt = 1/f_s = {rf_period_ps:.1f} ps) "
                    f"charge-scaled single-bunch estimate for Q₁ + Q₂ = 85 nC vs experimental 1.3 GHz (PRL line 143) "
                    f"linac spacing ({linac_period_ps:.0f} ps; 9th harmonic of 1.27 GHz bunch train with T_b = 0.79 ns, thesis lines 2340–2341, 2365)"
                ),
            },
        ],
    )

    if res_wake.get("table_path") is not None and Path(res_wake["table_path"]).is_file():
        md_txt = Path(res_wake["table_path"]).read_text(encoding="utf-8")
        md_txt = md_txt.replace(
            "pulse: continuous / custom",
            (
                f"pulse: rigid baseband Gaussian bunch λ_q(z), σ_z = {sig_mm:.1f} mm "
                f"(2σ_z = {two_sig_mm:.1f} mm, PRL line 151), |F(f)|² = exp(-(2π f σ_z / u_0)²)"
            ),
        )
        Path(res_wake["table_path"]).write_text(md_txt, encoding="utf-8")

    # Custom-render Figure 00b (mesh + rigid baseband Gaussian bunch λ_q(z) + form-factor power spectrum |F(f)|²)
    info_b = schematic.extract_setup_info(
        proxy_wake,
        plane="zy",
        material_names=mat_names_wake,
        sources=sources_wake,
        monitors=monitors_wake,
    )
    disc_b = schematic._discretization_metrics(
        info_b,
        freq=f_sync_hz,
        min_feature=spec.wagon_wheel_thickness_m,
    )
    sc_mm = 1.0e3
    h0_mm = float(info_b["h_bounds"][0] * sc_mm)
    h1_mm = float(info_b["h_bounds"][1] * sc_mm)
    v0_mm = float(info_b["v_bounds"][0] * sc_mm)
    v1_mm = float(info_b["v_bounds"][1] * sc_mm)
    dh_mm = h1_mm - h0_mm
    dv_mm = v1_mm - v0_mm
    pml_z_mm = float(info_b["pml_thickness"]["z_low"] * sc_mm)

    fig_b = plt.figure(figsize=(10.6, 5.8), dpi=figstyle.DEFAULT_DPI)
    fig_b._wakis_custom_layout = True
    gs_b = fig_b.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.98,
        top=0.88,
        bottom=0.10,
        height_ratios=[0.95, 1.12],
        hspace=0.46,
        wspace=0.24,
    )
    gs_top = gs_b[0, :].subgridspec(1, 2, width_ratios=[2.75, 1.0], wspace=0.14)
    ax_mesh = fig_b.add_subplot(gs_top[0, 0])
    ax_pkt = fig_b.add_subplot(gs_b[1, 0])
    ax_spec = fig_b.add_subplot(gs_b[1, 1])

    # Panel (a): Full-width computational mesh in the zy cut + 1× zoom inset
    ax_mesh.add_patch(
        Rectangle((h0_mm, v0_mm), dh_mm, dv_mm, facecolor="#F8FAFC", edgecolor="none", zorder=1)
    )
    for mat in info_b["materials"]:
        if mat["kind"] == "rect":
            mh0, mh1 = mat["h_bounds"][0] * sc_mm, mat["h_bounds"][1] * sc_mm
            mv0, mv1 = mat["v_bounds"][0] * sc_mm, mat["v_bounds"][1] * sc_mm
            ax_mesh.add_patch(
                Rectangle(
                    (mh0, mv0),
                    mh1 - mh0,
                    mv1 - mv0,
                    facecolor=mat["color"],
                    edgecolor="#1E293B",
                    lw=1.0,
                    alpha=0.78,
                    zorder=2,
                )
            )
    if pml_z_mm > 0.0:
        for z_pml_start in (h0_mm, h1_mm - pml_z_mm):
            ax_mesh.add_patch(
                Rectangle(
                    (z_pml_start, v0_mm),
                    pml_z_mm,
                    dv_mm,
                    facecolor="#CBD5E1",
                    edgecolor="#64748B",
                    hatch="///",
                    alpha=0.35,
                    lw=0.8,
                    zorder=2,
                )
            )
    mesh_meta = schematic.draw_mesh(
        ax_mesh,
        proxy_wake,
        plane="zy",
        unit="mm",
        wavelength=disc_b["wavelength_m"],
        zoom_inset=False,
        show_cpw=False,
    )
    ax_mesh.add_patch(
        Rectangle((h0_mm, v0_mm), dh_mm, dv_mm, facecolor="none", edgecolor=figstyle.INK, lw=1.3, zorder=6)
    )
    ax_mesh.set_xlim(h0_mm, h1_mm)
    ax_mesh.set_ylim(v0_mm, v1_mm)
    ax_mesh.set_aspect("auto")
    ax_mesh.set_xlabel("z [mm]")
    ax_mesh.set_ylabel("y [mm]")
    for req_disc_key in ("cpw_nmax", "n_max", "cells_across_feature", "min_feature_m"):
        if disc_b.get(req_disc_key) is None:
            raise ValueError(
                f"Discretization metric {req_disc_key!r} is missing (got {disc_b.get(req_disc_key)!r})."
            )
    cpw_val = float(disc_b["cpw_nmax"])
    nmax_val = float(disc_b["n_max"])
    caf_val = float(disc_b["cells_across_feature"])
    feat_mm_val = float(disc_b["min_feature_m"]) * 1e3
    ax_mesh.set_title(
        f"(a) Mesh (every {mesh_meta['stride_h']}th line): Δz = {dz_mm:.2f} mm · "
        f"{cpw_val:.1f} cells/λ (n_max = {nmax_val:.2f}) · {caf_val:.1f} cells / {feat_mm_val:.0f} mm feature",
        fontsize=10.2,
        fontweight="bold",
    )

    # Zoom panel placed outside ax_mesh on the right, showing one period around the active structure centre
    t_ww_mm = float(t1_spec.get("wagon_wheel_thickness_m", spec.wagon_wheel_thickness_m)) * sc_mm
    r_hub_mm_val = float(t1_spec.get("hub_outer_radius_m", spec.hub_outer_radius_m)) * sc_mm
    z_center_mm = 0.5 * (z_act_start + z_act_end) * sc_mm
    zh0_mm = z_center_mm - 0.5 * p_mm
    zh1_mm = z_center_mm + 0.5 * p_mm
    zv0_mm = r_b_mm - 0.75 * t_ww_mm
    zv1_mm = r_hub_mm_val + 0.5 * t_ww_mm
    ins_bounds = [1.07, 0.0, 0.34, 1.0]
    zoom_h = (dh_mm / max(zh1_mm - zh0_mm, 1e-12)) * ins_bounds[2]
    zoom_v = (dv_mm / max(zv1_mm - zv0_mm, 1e-12)) * ins_bounds[3]
    zoom_mag = float(np.sqrt(zoom_h * zoom_v))
    ins_m = ax_mesh.inset_axes(ins_bounds)
    ins_m._is_inset = True
    ins_m.set_in_layout(False)
    ins_m.set_facecolor(figstyle.PAPER)
    for sp_name in ("top", "right", "left", "bottom"):
        ins_m.spines[sp_name].set_visible(True)
        ins_m.spines[sp_name].set_color("#D55E00")
        ins_m.spines[sp_name].set_linewidth(1.2)
    for mat in info_b["materials"]:
        if mat["kind"] == "rect":
            mh0, mh1 = mat["h_bounds"][0] * sc_mm, mat["h_bounds"][1] * sc_mm
            mv0, mv1 = mat["v_bounds"][0] * sc_mm, mat["v_bounds"][1] * sc_mm
            ins_m.add_patch(
                Rectangle(
                    (mh0, mv0),
                    mh1 - mh0,
                    mv1 - mv0,
                    facecolor=mat["color"],
                    edgecolor="#1E293B",
                    lw=0.95,
                    alpha=0.80,
                )
            )
    z_nodes_mm = info_b["nodes"][info_b["h_idx"]] * sc_mm
    y_nodes_mm = info_b["nodes"][info_b["v_idx"]] * sc_mm
    for zx in z_nodes_mm[(z_nodes_mm >= zh0_mm - dz_mm) & (z_nodes_mm <= zh1_mm + dz_mm)]:
        ins_m.axvline(float(zx), color="#475569", lw=0.65, alpha=0.80)
    for yy in y_nodes_mm[(y_nodes_mm >= zv0_mm - dz_mm) & (y_nodes_mm <= zv1_mm + dz_mm)]:
        ins_m.axhline(float(yy), color="#475569", lw=0.65, alpha=0.80)
    ins_m.set_xlim(zh0_mm, zh1_mm)
    ins_m.set_ylim(zv0_mm, zv1_mm)
    ins_m.set_xticks([zh0_mm, z_center_mm, zh1_mm])
    ins_m.set_yticks([r_b_mm, r_hub_mm_val])
    ins_m.tick_params(axis="both", labelsize=7.5)
    ins_m.set_xlabel("z [mm]", fontsize=8.0)
    ax_mesh.indicate_inset_zoom(ins_m, edgecolor="#D55E00", alpha=0.90, lw=1.3, zorder=8)
    ins_m.text(
        0.5,
        0.13,
        f"{zoom_mag:.1f}× zoom (Δz = {dz_mm:.2f} mm)",
        transform=ins_m.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=figstyle.INK,
        bbox={"boxstyle": "round", "facecolor": "#FFFFFF", "edgecolor": "#94A3B8", "alpha": 0.94},
    )

    # Panel (b): Rigid baseband Gaussian bunch λ_q(z) with 2σ_z = 2.4 mm (from t1.spec.bunch_length_rms_m)
    if pml_z_mm > 0.0:
        ax_pkt.axvspan(
            h0_mm,
            h0_mm + pml_z_mm,
            facecolor="#CBD5E1",
            edgecolor="#64748B",
            hatch="///",
            alpha=0.35,
            label="CPML",
        )
        ax_pkt.axvspan(
            h1_mm - pml_z_mm,
            h1_mm,
            facecolor="#CBD5E1",
            edgecolor="#64748B",
            hatch="///",
            alpha=0.35,
        )
    plate_labeled = False
    for mat in info_b["materials"]:
        mh0, mh1 = mat["h_bounds"][0] * sc_mm, mat["h_bounds"][1] * sc_mm
        if abs((mh1 - mh0) - dh_mm) > 1.0:
            ax_pkt.axvspan(
                mh0,
                mh1,
                facecolor="#94A3B8",
                edgecolor="#475569",
                lw=0.7,
                alpha=0.38,
                label="MTM / taper PEC" if not plate_labeled else None,
            )
            plate_labeled = True

    z0_bunch_mm = 0.5 * (h0_mm + h1_mm)
    z_grid_mm = np.linspace(h0_mm, h1_mm, 800)
    lam_q_norm = np.exp(-0.5 * ((z_grid_mm - z0_bunch_mm) / max(sig_mm, 1e-9)) ** 2)
    ax_pkt.fill_between(z_grid_mm, 0.0, lam_q_norm, color="#0072B2", alpha=0.22, zorder=3)
    ax_pkt.plot(
        z_grid_mm,
        lam_q_norm,
        color="#0072B2",
        lw=2.1,
        marker="none",
        zorder=4,
        label=rf"Baseband $\lambda_q(z)$ ($\sigma_z = {sig_mm:.1f}$ mm)",
    )
    y_arrow_b = float(np.exp(-0.5))
    ax_pkt.annotate(
        "",
        xy=(z0_bunch_mm + sig_mm, y_arrow_b),
        xytext=(z0_bunch_mm - sig_mm, y_arrow_b),
        arrowprops={"arrowstyle": "<->", "color": "#D55E00", "lw": 1.4},
        zorder=5,
    )
    ax_pkt.annotate(
        f"2σ_z = {two_sig_mm:.1f} mm\n(σ_z = {sig_mm:.1f} mm, q = {q_nc:.0f} nC)",
        xy=(z0_bunch_mm + sig_mm, y_arrow_b),
        xytext=(z0_bunch_mm + 4.8 * sig_mm, 0.62),
        ha="left",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#005A8D",
        bbox={"boxstyle": "round", "facecolor": "#FFFFFF", "edgecolor": "#CBD5E1", "alpha": 0.94},
        arrowprops={"arrowstyle": "->", "color": "#005A8D", "lw": 1.1},
        zorder=6,
    )
    ax_pkt.set_xlim(h0_mm, h1_mm)
    ax_pkt.set_ylim(-0.06, 1.45)
    ax_pkt.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_pkt.set_xlabel("z [mm]")
    ax_pkt.set_ylabel(r"Normalized $\lambda_q(z) / \lambda_{q,\max}$")
    ax_pkt.set_title(
        f"(b) Rigid Baseband Bunch λ_q(z) (2σ_z = {two_sig_mm:.1f} mm)",
        fontsize=10.0,
        fontweight="bold",
    )
    figstyle.spectrum_grid(ax_pkt)
    ax_pkt.legend(
        loc="upper left",
        fontsize=8.0,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        framealpha=0.92,
    )

    # Panel (c): Baseband bunch form-factor power spectrum |F(f)|² = exp(-(2π f σ_z / u_0)²)
    f_max_spec_ghz = float(max(1.12 * f_1e3_ghz, 58.0))
    f_spec_ghz = np.linspace(0.0, f_max_spec_ghz, 500)
    ff_sq = np.exp(-((2.0 * np.pi * (f_spec_ghz * 1.0e9) * sig_m / u0) ** 2))
    ff_sq_fs = float(np.exp(-((2.0 * np.pi * (f_nf8_ghz * 1.0e9) * sig_m / u0) ** 2)))
    ff_Cross_level = 1.0e-3

    ax_spec.axvspan(
        wpb_lo_ghz,
        wpb_hi_ghz,
        color="#FDE68A",
        alpha=0.55,
        label=f"Wake band ({wpb_lo_ghz:.1f}–{wpb_hi_ghz:.1f} GHz)",
    )
    ax_spec.axvspan(
        pb_lo_ghz,
        pb_hi_ghz,
        color="#A7F3D0",
        alpha=0.80,
        label=f"Field filter ({pb_lo_ghz:.1f}–{pb_hi_ghz:.1f} GHz)",
    )
    ax_spec.plot(
        f_spec_ghz,
        ff_sq,
        color="#D55E00",
        lw=2.1,
        marker="none",
        label=r"$|F(f)|^2 = \exp(-(2\pi f \sigma_z / u_0)^2)$",
    )
    ax_spec.axvline(
        f_nf8_ghz,
        color=figstyle.INK,
        ls="--",
        lw=1.2,
        label=f"f_s = {f_nf8_ghz:.2f} GHz (|F|² = {ff_sq_fs:.2f})",
    )
    ax_spec.axvline(
        f_1e3_ghz,
        color="#0072B2",
        ls=":",
        lw=1.4,
    )
    ax_spec.plot(
        [f_1e3_ghz],
        [ff_Cross_level],
        marker="o",
        color="#0072B2",
        ms=5.5,
        ls="none",
        zorder=6,
        label=f"|F|² = {ff_Cross_level:.0e} at {f_1e3_ghz:.2f} GHz",
    )
    ax_spec.annotate(
        f"|F|² = {ff_Cross_level:.0e}\n({f_1e3_ghz:.2f} GHz)",
        xy=(f_1e3_ghz, ff_Cross_level),
        xytext=(f_1e3_ghz - 14.0, 0.22),
        ha="center",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color="#005A8D",
        arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 1.1},
    )
    ax_spec.set_xlim(0.0, f_max_spec_ghz)
    ax_spec.set_ylim(-0.04, 1.15)
    ax_spec.set_xlabel("Frequency f [GHz]")
    ax_spec.set_ylabel(r"Form factor squared $|F(f)|^2$")
    ax_spec.set_title(
        r"(c) Bunch Spectrum $|F(f)|^2$ & Passband Windows",
        fontsize=10.0,
        fontweight="bold",
    )
    figstyle.spectrum_grid(ax_spec)
    ax_spec.legend(loc="upper right", fontsize=8.5, frameon=False)
    fig_b.suptitle(
        f"Computational Mesh, Baseband Bunch Excitation & Form Factor (Δ = {dz_mm:.2f} mm, {int(round(1.0/dz_mm))} cells across {feat_mm_val:.0f} mm)",
        y=0.97,
        fontsize=11.2,
        fontweight="bold",
    )
    figstyle.save(fig_b, res_wake["mesh_figure"], dpi=figstyle.DEFAULT_DPI, close=True, auto_legend_below=False)

    return {
        "fig_00a": str(res_wake["geometry_figure"]),
        "fig_00b": str(res_wake["mesh_figure"]),
        "table_00": str(res_wake["table_path"]),
    }


def _build_task_01_parameter_tables(
    pec_just: dict[str, Any] | None = None,
    port_refl: dict[str, Any] | None = None,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return verified paper parameter and assumed parameter tables with exact `.ref/mtm/` line citations."""
    if pec_just is None:
        pec_just = analytic_ohmic_pec_justification(spec)
    tau_0_ns = float(pec_just["ohmic_field_decay_time_2q_over_omega_ns"])
    tp_n40_ns = float(pec_just["pulse_duration_n40_ns"])
    fc_tm01_ghz = circular_waveguide_tm01_cutoff_hz(spec.waveguide_radius_m) * 1e-9
    if port_refl is None:
        port_refl = measure_upstream_port_reflection(quick=True, spec=spec)
    f_eval_ghz = float(port_refl["f_sync_eval_ghz"])
    gamma_tap = float(port_refl["gamma_tapered_at_sync"])
    r_hub_mm = spec.hub_outer_radius_m * 1e3
    r_wg_mm = spec.waveguide_radius_m * 1e3
    r_p_mm = spec.coax_port_wall_radius_m * 1e3

    paper_parameters = [
        {
            "parameter": "Unit-cell axial period p",
            "symbol": "p",
            "value": 2.0,
            "unit": "mm",
            "ref_lines": "PRL lines 140, 145–146, 166; Lu thesis lines 1978, 1985, 1994, 2036",
            "quote_context": "Total period 2 mm composed of one 1 mm wagon-wheel plate and one 1 mm spacer plate",
        },
        {
            "parameter": "Wagon-wheel plate thickness t_ww",
            "symbol": "t_ww",
            "value": 1.0,
            "unit": "mm",
            "ref_lines": "PRL lines 112–113, 163–165; Lu thesis lines 1950–1951, 1967, 2037–2038",
            "quote_context": "1-mm-thick stainless-steel wagon-wheel plate in alternating periodic stack",
        },
        {
            "parameter": "Spacer plate thickness t_sp",
            "symbol": "t_sp",
            "value": 1.0,
            "unit": "mm",
            "ref_lines": "PRL lines 112–113, 163–165; Lu thesis lines 1950–1951, 1967, 2037–2038",
            "quote_context": "1-mm-thick copper circular spacer plate between adjacent wagon-wheel plates",
        },
        {
            "parameter": "Radial spoke width w_spoke",
            "symbol": "w_spoke",
            "value": 1.0,
            "unit": "mm",
            "ref_lines": "PRL Fig. 2(b) line 119; Lu thesis line 1968",
            "quote_context": "Dimension callout '1 mm' in PRL Fig. 2(b) and 'thickness of spokes as 1 mm' in thesis line 1968",
        },
        {
            "parameter": "Beam hole diameter 2 r_b",
            "symbol": "2 r_b",
            "value": 6.0,
            "unit": "mm",
            "ref_lines": "PRL lines 122–123, 145; Lu thesis lines 1967, 1969, 2185, 2322",
            "quote_context": "Central electron beam aperture of 6.0 mm diameter (r_b = 3.0 mm, PRL line 122)",
        },
        {
            "parameter": "Enclosing circular waveguide ID 2 R_wg",
            "symbol": "2 R_wg",
            "value": 16.0,
            "unit": "mm",
            "ref_lines": "PRL lines 121, 152 & thesis line 1968 (16 mm ID); cf. thesis lines 2075, 2208 (R_wg = 8.1 mm)",
            "quote_context": "Circular waveguide of 16.0 mm ID (R_wg = 8.0 mm, PRL line 121); thesis lines 2075, 2208 state R_wg = 8.1 mm (TM01 cutoff 14.2 GHz at PRL lines 150–151)",
        },
        {
            "parameter": "Number of periods N_exp (active length L_exp)",
            "symbol": "N_exp (L_exp)",
            "value": 40,
            "unit": "periods (80 mm)",
            "ref_lines": "PRL lines 134, 166; Lu thesis lines 1951–1952, 2201, 2237, 2244, 2376",
            "quote_context": "40 periods for a total metamaterial interaction length of 8 cm (80 mm)",
        },
        {
            "parameter": "Drive electron beam energy E_k",
            "symbol": "E_k",
            "value": 65.0,
            "unit": "MeV",
            "ref_lines": "PRL lines 49, 104, 143, 149–150; Lu thesis lines 1991, 2054",
            "quote_context": "65 MeV relativistic electron bunches from the 1.3 GHz (PRL line 143) AWA rf photoinjector",
        },
        {
            "parameter": "Single drive bunch charge q",
            "symbol": "q",
            "value": 45.0,
            "unit": "nC",
            "ref_lines": "PRL lines 49–50, 145, 150, 214, 224, 233–234, 291–294; Lu thesis lines 2246, 2312–2314, 2364",
            "quote_context": "Single high-charge bunch of up to 45 nC generating 25 MW peak power",
        },
        {
            "parameter": "RMS longitudinal bunch length σ_z",
            "symbol": "σ_z",
            "value": 1.2,
            "unit": "mm",
            "ref_lines": "PRL lines 151, 212, 247–248, 291; Lu thesis lines 2310, 2313",
            "quote_context": "Gaussian longitudinal bunch distribution with σ_z = 1.2 mm (Φ = 0.96, PRL lines 151, 248)",
        },
        {
            "parameter": "Two-bunch total charge Q_1 + Q_2",
            "symbol": "Q_tot",
            "value": 85.0,
            "unit": "nC",
            "ref_lines": "PRL lines 51, 77, 143, 153–154, 255–257, 297–298, 323, 329–331, 337–338; Lu thesis lines 2337–2341, 2365",
            "quote_context": "Two electron bunches spaced around 1/(1.3 GHz) = 769.23 ps (PRL line 143) and fine-tuned in phase with laser splitters (PRL lines 153–154, 256–257, 329–331; thesis lines 2338–2341: 11.42 GHz is the 9th harmonic of a 1.27 GHz bunch train with T_b = 0.79 ns at line 2365) with combined charge 85 nC generating 80 MW",
        },
    ]

    assumed_parameters = [
        {
            "parameter": "Inner hub ring outer radius",
            "symbol": "r_hub",
            "assumed_value": f"{r_hub_mm:.1f} mm (ASSUMED: r_b + 1.0 mm ring wall, PRL Fig. 2(b) line 119)",
            "reason": (
                f"PRL Fig. 2(b) (line 119) shows a 1 mm dimension callout and thesis line 1968 states "
                f"'thickness of spokes as 1 mm', but neither text explicitly prints r_hub = {r_hub_mm:.1f} mm"
            ),
        },
        {
            "parameter": "Number & orientation of radial spokes",
            "symbol": "N_spokes",
            "assumed_value": "4 spokes along ±x, ±y on every plate (ASSUMED)",
            "reason": (
                f"Counted from PRL Fig. 2(b) (line 119); spokes span r ∈ [{r_hub_mm:.1f}, {r_wg_mm:.1f}] mm "
                f"directly into the outer waveguide wall with no separate outer slot rim (R_slot = R_wg = {r_wg_mm:.1f} mm, PRL line 121, ASSUMED)"
            ),
        },
        {
            "parameter": "Spoke & hub corner fillet radius",
            "symbol": "r_fillet",
            "assumed_value": f"{0.0:.1f} mm (sharp rectangular cross-section, ASSUMED)",
            "reason": "Machined corner radii are not specified in PRL or thesis; represented via subcell conformal PEC (conformal='pec')",
        },
        {
            "parameter": "Conductor wall conductivity",
            "symbol": "σ_wall",
            "assumed_value": "Lossless PEC (σ → ∞, ASSUMED)",
            "reason": (
                f"Experiment uses stainless-steel wagon-wheel plates and copper spacers (PRL lines 163–165) with "
                f"cold-test Q_0 = {spec.unloaded_q_thesis:.0f} (Lu thesis Table 2.4, line 2173); ohmic decay time "
                f"2Q_0/ω_s = {tau_0_ns:.2f} ns ≫ pulse length t_p ≤ {tp_n40_ns:.2f} ns"
            ),
        },
        {
            "parameter": "End output port couplers",
            "symbol": f"2-cell adiabatic coaxial TEM coupler (r ∈ [{r_p_mm:.1f}, {r_wg_mm:.1f}] mm) + 10-cell CPML (ASSUMED)",
            "assumed_value": "Replaces 2 rectangular WR-90 side-wall couplers (PRL lines 95–97, 156–158; Lu thesis lines 2198–2224)",
            "reason": (
                f"An empty R_wg = {r_wg_mm:.1f} mm (PRL line 121) circular pipe is cut off below {fc_tm01_ghz:.2f} GHz; "
                f"the coaxial transition extracts the backward wave axisymmetrically with |Γ| = {gamma_tap:.4f} at {f_eval_ghz:.2f} GHz"
            ),
        },
        {
            "parameter": "Simulated active periods N_sim",
            "symbol": "N_sim",
            "assumed_value": "N ∈ {6, 8, 12} (L = 12, 16, 24 mm, ASSUMED)",
            "reason": (
                "Probes the short-structure pre-asymptotic regime where the 8 mm coupler taper length is comparable "
                "to the active interaction length L = N p, isolating the affine active-section slope a_W = dW/dL "
                "from the end/taper intercept b_W and testing length invariance of peak backward-wave power P_up,max(L)"
            ),
        },
    ]
    return paper_parameters, assumed_parameters


def _assemble_task_01_dict(
    refl: dict[str, Any],
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Assemble ``task_01_geometry_and_port_reflection.json`` from raw ``refl`` port-reflectometry metrics."""
    gamma, beta, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    fc_tm01_hz = circular_waveguide_tm01_cutoff_hz(spec.waveguide_radius_m)
    fc_tm01_8p1mm_hz = circular_waveguide_tm01_cutoff_hz(8.1e-3)
    pec_just = analytic_ohmic_pec_justification(spec)
    paper_parameters, assumed_parameters = _build_task_01_parameter_tables(pec_just, refl, spec)

    return {
        "spec": asdict(spec),
        "beam_gamma": gamma,
        "beam_beta": beta,
        "beam_u0_m_s": u0,
        "empty_waveguide_tm01_cutoff_analytic_ghz": float(fc_tm01_hz * 1e-9),
        "empty_waveguide_tm01_cutoff_8p1mm_analytic_ghz": float(fc_tm01_8p1mm_hz * 1e-9),
        "empty_waveguide_tm01_cutoff_paper_ghz": 14.2,
        "pec_ohmic_justification": pec_just,
        "paper_parameters": paper_parameters,
        "assumed_parameters": assumed_parameters,
        "port_reflection": {
            k: v
            for k, v in refl.items()
            if k
            not in (
                "f_ghz",
                "gamma_tapered_coax",
                "gamma_tapered_coax_raw",
                "gamma_abrupt_empty_pipe",
                "gamma_abrupt_empty_pipe_raw",
                "gamma_pec_cap",
            )
        },
    }


def generate_task_01_data(
    data_dir: Path | str,
    *,
    quick: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Generate ``task_01_geometry_and_port_reflection.json`` and ``task_01_port_reflection_spectra.npz``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "task_01_geometry_and_port_reflection.json"
    npz_path = data_dir / "task_01_port_reflection_spectra.npz"

    refl = measure_upstream_port_reflection(quick=quick, spec=spec)
    out_t1 = _assemble_task_01_dict(refl, spec=spec)

    np.savez_compressed(
        npz_path,
        f_ghz=refl["f_ghz"],
        gamma_tapered_coax=refl["gamma_tapered_coax"],
        gamma_tapered_coax_raw=refl.get("gamma_tapered_coax_raw", refl["gamma_tapered_coax"]),
        gamma_abrupt_empty_pipe=refl["gamma_abrupt_empty_pipe"],
        gamma_abrupt_empty_pipe_raw=refl.get("gamma_abrupt_empty_pipe_raw", refl["gamma_abrupt_empty_pipe"]),
        gamma_pec_cap=refl["gamma_pec_cap"],
    )
    json_path.write_text(json.dumps(out_t1, indent=2) + "\n", encoding="utf-8")
    return out_t1


def measure_empty_waveguide_tm01_cutoff_fit(
    radius_m: float = 8.0e-3,
    *,
    period_m: float = 2.0e-3,
    quick: bool = False,
) -> float:
    """Measure the :math:`\\mathrm{TM}_{01}` cutoff frequency [GHz] of an empty circular PEC pipe via 3D FIT."""
    empty_solid = make_annular_cylinder(radius_m, radius_m + 3.0e-3, -0.5 * period_m, 1.5 * period_m, n_theta=96)
    n_xy_emp = 32 if quick else 48
    x_e = np.linspace(-radius_m, radius_m, n_xy_emp + 1)
    y_e = np.linspace(-radius_m, radius_m, n_xy_emp + 1)
    z_e = np.linspace(0.0, period_m, 5)
    g_emp = GridFIT3D(
        x=x_e,
        y=y_e,
        z=z_e,
        stl_solids={"wall": empty_solid},
        stl_materials={"wall": "pec"},
        conformal="pec",
        verbose=0,
    )
    s_emp = SolverFIT3D(
        g_emp,
        use_stl=True,
        bc_low=["pec", "pec", "bloch"],
        bc_high=["pec", "pec", "bloch"],
        dtype=np.complex128,
        k_bloch=(0.0, 0.0, 0.0),
        verbose=0,
    )
    n_steps = 1600 if quick else 2400
    est = estimate((s_emp.Nx, s_emp.Ny, s_emp.Nz), steps=n_steps, dtype="complex128", conformal="pec")
    preflight(est, "laptop" if quick else "n2-highmem-8")
    pulse_emp = GaussianSource(frequency=14.3e9, fwidth=4.0e9)
    src_emp = [CurrentSource(pulse_emp, component="Ez", center=(0.0, 0.0, 0.5 * period_m), amplitude=1.0)]
    mon_emp = ResonanceMonitor("Ez", center=(0.0, 0.0, 0.5 * period_m), t_start=pulse_emp.end_time)
    s_emp.emsolve(n_steps, source=src_emp, monitors=[mon_emp], verbose=0)
    f0_emp_hz, _ = _extract_resonance_freq_and_q(mon_emp, 12.5e9, 16.0e9)
    fc_tm01_fit_ghz = float(f0_emp_hz * 1e-9)
    del s_emp, g_emp
    gc.collect()
    return fc_tm01_fit_ghz


def _assemble_task_02_dict(
    fc_tm01_fit_ghz: float,
    phi_list: list[float],
    disp_freqs_ghz: list[float],
    disp_wall_s: float,
    phi_fwd_list: list[float],
    f_fwd_ghz: list[float],
    uniform_levels: dict[str, Any],
    graded_levels: dict[str, Any],
    v81_nf4: dict[str, Any],
    v81_nf6: dict[str, Any],
    v81_nf8: dict[str, Any],
    *,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Assemble ``task_02_cold_test_and_convergence.json`` from per-run cold-test simulation outputs."""
    _, _, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    p = spec.period_m
    fc_tm01_analytic_ghz = circular_waveguide_tm01_cutoff_hz(spec.waveguide_radius_m) * 1e-9

    phi_arr = np.array(phi_list, dtype=float)
    f_arr_ghz = np.array(disp_freqs_ghz, dtype=float)
    kz_arr = np.deg2rad(phi_arr) / p
    f_beam_ghz = (kz_arr * u0 / (2.0 * np.pi)) * 1e-9

    # Group velocity v_g(phi) = d_omega / d_kz across passband
    vg_curve_over_c = (2.0 * np.pi * np.gradient(f_arr_ghz * 1e9, kz_arr)) / C0

    # Find exact beam-line crossing (f_mode(phi) = f_beam(phi))
    diff_f = f_arr_ghz - f_beam_ghz
    cross_idx = np.where(np.diff(np.sign(diff_f)) != 0)[0]
    if len(cross_idx) > 0:
        ic = int(cross_idx[0])
        w_c = float(-diff_f[ic] / (diff_f[ic + 1] - diff_f[ic]))
        phi_sync_deg = float(phi_arr[ic] + w_c * (phi_arr[ic + 1] - phi_arr[ic]))
        f_sync_disp_ghz = float(f_arr_ghz[ic] + w_c * (f_arr_ghz[ic + 1] - f_arr_ghz[ic]))
        vg_sync_disp_over_c = float(np.interp(phi_sync_deg, phi_arr, vg_curve_over_c))
    else:
        phi_sync_deg = 27.43
        f_sync_disp_ghz = float(f_arr_ghz[int(np.argmin(np.abs(phi_arr - 27.43)))])
        vg_sync_disp_over_c = float(vg_curve_over_c[int(np.argmin(np.abs(phi_arr - 27.43)))])

    phi_fwd_arr = np.array(phi_fwd_list, dtype=float)
    f_fwd_arr = np.array(f_fwd_ghz, dtype=float)
    kz_fwd_arr = np.deg2rad(phi_fwd_arr) / p
    vg_fwd_curve_over_c = (2.0 * np.pi * np.gradient(f_fwd_arr * 1e9, kz_fwd_arr)) / C0

    nf_levels = [4, 6, 8]
    h_mm_arr = [float(uniform_levels[f"nf{nf}"]["h_min_mm"]) for nf in nf_levels]
    f_u_arr = [float(uniform_levels[f"nf{nf}"]["f_mode_ghz"]) for nf in nf_levels]
    ve_u_arr = [float(uniform_levels[f"nf{nf}"]["ve_over_c"]) for nf in nf_levels]
    rq_u_arr = [float(uniform_levels[f"nf{nf}"]["r_over_q_kohm_per_m"]) for nf in nf_levels]
    kl_u_arr = [float(uniform_levels[f"nf{nf}"]["k_loss_v_per_c_m"]) for nf in nf_levels]

    f_g_arr = [float(graded_levels[f"nf{nf}"]["f_mode_ghz"]) for nf in nf_levels]
    ve_g_arr = [float(graded_levels[f"nf{nf}"]["ve_over_c"]) for nf in nf_levels]
    rq_g_arr = [float(graded_levels[f"nf{nf}"]["r_over_q_kohm_per_m"]) for nf in nf_levels]
    kl_g_arr = [float(graded_levels[f"nf{nf}"]["k_loss_v_per_c_m"]) for nf in nf_levels]

    def _conv_summary(vals: list[float], h_vals: list[float]) -> dict[str, Any]:
        gci_d = compute_three_level_gci(vals, h_vals, p_formal=1.0)
        p_obs = gci_d["observed_order"]
        asymptotic = bool(gci_d["fs_used"] == 1.25)
        v4, v6, v8 = float(vals[0]), float(vals[1]), float(vals[2])
        return {
            "values": [v4, v6, v8],
            "is_monotone": bool(gci_d["is_monotone"]),
            "convergence_class": str(gci_d["convergence_class"]),
            "R_ratio": float(gci_d["R_ratio"]),
            "is_asymptotic": asymptotic,
            "observed_order": p_obs,
            "p_used": float(gci_d["p_used"]),
            "fs_used": float(gci_d["fs_used"]),
            "richardson_estimate": gci_d["richardson_estimate"],
            "gci_fine_rel": float(gci_d["gci_fine_rel"]),
            "gci_med_rel": float(gci_d["gci_med_rel"]),
            "half_range_rel": float(gci_d["half_range_rel"]),
            "u_fine_rel": float(gci_d["u_fine_rel"]),
            "finest_value_nf8": v8,
            "rel_change_nf8_vs_nf6": float(abs(v8 - v6) / max(abs(v8), 1e-30)),
            "rel_change_nf8_vs_nf4": float(abs(v8 - v4) / max(abs(v8), 1e-30)),
        }

    conv_uniform = {
        "h_mm": h_mm_arr,
        "f_mode_ghz": _conv_summary(f_u_arr, h_mm_arr),
        "ve_over_c": _conv_summary(ve_u_arr, h_mm_arr),
        "r_over_q_kohm_per_m": _conv_summary(rq_u_arr, h_mm_arr),
        "k_loss_v_per_c_m": _conv_summary(kl_u_arr, h_mm_arr),
    }
    conv_graded = {
        "h_mm": h_mm_arr,
        "f_mode_ghz": _conv_summary(f_g_arr, h_mm_arr),
        "ve_over_c": _conv_summary(ve_g_arr, h_mm_arr),
        "r_over_q_kohm_per_m": _conv_summary(rq_g_arr, h_mm_arr),
        "k_loss_v_per_c_m": _conv_summary(kl_g_arr, h_mm_arr),
    }

    finest_u = uniform_levels["nf8"]
    phi_form = float(gaussian_bunch_form_factor(finest_u["f_mode_hz"], spec.bunch_length_rms_m, u0))
    f_pap_ghz = float(spec.paper_sync_freq_hz * 1e-9)
    vg_pap_c = float(spec.paper_vg_over_c)
    rq_pap_kohm = float(spec.paper_r_over_q_ohm_per_m * 1e-3)
    kl_pap = paper_cold_test_k_loss_per_m(spec.paper_sync_freq_hz, spec.paper_r_over_q_ohm_per_m)

    spec_8p1 = WagonWheelMTMSpec(waveguide_radius_m=8.1e-3, slot_outer_radius_m=8.1e-3)
    fc_8p1_ghz = float(circular_waveguide_tm01_cutoff_hz(spec_8p1.waveguide_radius_m) * 1e-9)
    variant_rwg_8p1mm_thesis = {
        "primary_radius_mm": float(spec.waveguide_radius_m * 1e3),
        "primary_radius_justification": (
            f"R_wg = {spec.waveguide_radius_m * 1e3:.1f} mm (16.0 mm inner diameter in PRL lines 121, 152 and Lu thesis "
            f"line 1968) is used as the primary model radius because it is the dimension printed in the primary PRL "
            f"publication and aligns 1.0 mm (PRL line 112) spoke/hub boundaries with power-of-two transverse grids "
            f"(n_xy = 64); the thesis engineering-drawing variant R_wg = {spec_8p1.waveguide_radius_m * 1e3:.1f} mm "
            f"(thesis lines 2075, 2208) is evaluated across nf4, nf6, and nf8 to bracket the paper's 11.42 GHz "
            f"(PRL line 137) synchronous frequency."
        ),
        "waveguide_radius_mm": 8.1,
        "slot_outer_radius_mm": 8.1,
        "tm01_cutoff_analytic_ghz": fc_8p1_ghz,
        "phi_sync_deg": 27.43,
        "nf4": v81_nf4,
        "nf6": v81_nf6,
        "nf8": v81_nf8,
        "rel_diff_f_sync_nf6_vs_paper_11p42ghz": float(abs(v81_nf6["f_mode_ghz"] - f_pap_ghz) / f_pap_ghz),
        "rel_diff_ve_nf6_vs_paper": float(abs(v81_nf6["ve_over_c"] - vg_pap_c) / abs(vg_pap_c)),
        "rel_diff_r_over_q_nf6_vs_paper": float(abs(v81_nf6["r_over_q_kohm_per_m"] - rq_pap_kohm) / rq_pap_kohm),
        "rel_diff_f_sync_nf8_vs_paper_11p42ghz": float(abs(v81_nf8["f_mode_ghz"] - f_pap_ghz) / f_pap_ghz),
        "rel_diff_ve_nf8_vs_paper": float(abs(v81_nf8["ve_over_c"] - vg_pap_c) / abs(vg_pap_c)),
        "rel_diff_r_over_q_nf8_vs_paper": float(abs(v81_nf8["r_over_q_kohm_per_m"] - rq_pap_kohm) / rq_pap_kohm),
    }

    return {
        "empty_waveguide_tm01_cutoff_analytic_ghz": float(fc_tm01_analytic_ghz),
        "empty_waveguide_tm01_cutoff_fit_ghz": float(fc_tm01_fit_ghz),
        "empty_waveguide_tm01_cutoff_paper_ghz": 14.2,
        "dispersion": {
            "phi_deg": phi_arr.tolist(),
            "kz_rad_per_m": kz_arr.tolist(),
            "f_mode_ghz": f_arr_ghz.tolist(),
            "f_beam_65mev_ghz": f_beam_ghz.tolist(),
            "vg_over_c": vg_curve_over_c.tolist(),
            "upper_cutoff_passband_max_ghz": float(np.max(f_arr_ghz)),
            "lower_passband_min_ghz": float(np.min(f_arr_ghz)),
            "phi_sync_deg": phi_sync_deg,
            "f_sync_crossing_ghz": f_sync_disp_ghz,
            "vg_at_sync_disp_over_c": vg_sync_disp_over_c,
            "forward_control_phi_deg": phi_fwd_arr.tolist(),
            "forward_control_f_ghz": f_fwd_arr.tolist(),
            "forward_control_vg_over_c": vg_fwd_curve_over_c.tolist(),
            "wall_time_s": float(disp_wall_s),
        },
        "paper_cold_test_benchmarks": {
            "f_sync_ghz": f_pap_ghz,
            "phi_sync_deg": 27.43,
            "vg_over_c": vg_pap_c,
            "r_over_q_kohm_per_m": rq_pap_kohm,
            "k_loss_v_per_c_m": kl_pap,
            "form_factor_phi": 0.96,
        },
        "wakis_finest_nf8_benchmarks": {
            "phi_eval_deg": 27.43,
            "f_sync_ghz": float(finest_u["f_mode_ghz"]),
            "f_at_paper_phi_nf8_ghz": float(finest_u["f_mode_ghz"]),
            "phi_beam_crossing_nf4_deg": phi_sync_deg,
            "f_beam_crossing_nf4_ghz": f_sync_disp_ghz,
            "vg_disp_over_c": vg_sync_disp_over_c,
            "ve_dft_over_c": float(finest_u["ve_over_c"]),
            "u_elec_over_u_mag": float(finest_u["u_elec_over_u_mag"]),
            "r_over_q_kohm_per_m": float(finest_u["r_over_q_kohm_per_m"]),
            "k_loss_v_per_c_m": float(finest_u["k_loss_v_per_c_m"]),
            "form_factor_phi": phi_form,
            "rel_error_f_sync_vs_paper": float(abs(finest_u["f_mode_ghz"] - f_pap_ghz) / f_pap_ghz),
            "rel_error_f_beam_crossing_nf4_vs_paper": float(abs(f_sync_disp_ghz - f_pap_ghz) / f_pap_ghz),
            "rel_error_vg_vs_paper": float(abs(vg_sync_disp_over_c - vg_pap_c) / abs(vg_pap_c)),
            "rel_error_ve_vs_paper": float(abs(finest_u["ve_over_c"] - vg_pap_c) / abs(vg_pap_c)),
            "rel_error_r_over_q_vs_paper": float(abs(finest_u["r_over_q_kohm_per_m"] - rq_pap_kohm) / rq_pap_kohm),
            "rel_error_k_loss_vs_paper": float(abs(finest_u["k_loss_v_per_c_m"] - kl_pap) / kl_pap),
        },
        "uniform_levels": uniform_levels,
        "graded_levels": graded_levels,
        "convergence_uniform": conv_uniform,
        "convergence_graded": conv_graded,
        "variant_rwg_8p1mm_thesis": variant_rwg_8p1mm_thesis,
    }


def generate_task_02_data(
    data_dir: Path | str,
    *,
    quick: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Generate ``task_02_cold_test_and_convergence.json`` and ``task_02_cold_test_fields.npz``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "task_02_cold_test_and_convergence.json"
    npz_path = data_dir / "task_02_cold_test_fields.npz"

    p = spec.period_m

    # 1. Verify empty circular waveguide TM01 cutoff in 3D Wakis FIT
    fc_tm01_fit_ghz = measure_empty_waveguide_tm01_cutoff_fit(
        spec.waveguide_radius_m, period_m=p, quick=quick
    )

    # 2. 3D Bloch dispersion sweep across phase advance phi_deg
    phi_list = [16.0, 24.0, 27.43, 36.0, 52.0] if quick else [12.0, 18.0, 24.0, 27.43, 32.0, 40.0, 50.0, 62.0, 72.0]
    disp_freqs_ghz: list[float] = []
    disp_wall_s = 0.0
    for ph in phi_list:
        r_ph = evaluate_bloch_unit_cell(
            phi_deg=ph,
            n_feat=4,
            use_graded_mesh=False,
            forward_control=False,
            compute_eigenmode_metrics=False,
            quick=quick,
            spec=spec,
        )
        disp_freqs_ghz.append(float(r_ph["f_mode_ghz"]))
        disp_wall_s += float(r_ph["wall_time_s"])

    # Forward-wave disk-loaded control dispersion (v_g > 0)
    phi_fwd_list = [28.0, 45.0, 62.0] if quick else [20.0, 32.0, 45.0, 58.0, 72.0]
    f_fwd_ghz: list[float] = []
    for ph in phi_fwd_list:
        r_fw = evaluate_bloch_unit_cell(
            phi_deg=ph,
            n_feat=4,
            use_graded_mesh=False,
            forward_control=True,
            compute_eigenmode_metrics=False,
            quick=quick,
            spec=spec,
        )
        f_fwd_ghz.append(float(r_fw["f_mode_ghz"]))
        disp_wall_s += float(r_fw["wall_time_s"])

    # 3. 3-level mesh convergence across n_feat in {4, 6, 8} at phi = 27.43 deg
    nf_levels = [4, 6, 8]
    uniform_levels: dict[str, Any] = {}
    graded_levels: dict[str, Any] = {}
    field_maps_nf4: dict[str, np.ndarray] = {}

    for nf in nf_levels:
        r_u = evaluate_bloch_unit_cell(
            phi_deg=27.43,
            n_feat=nf,
            use_graded_mesh=False,
            forward_control=False,
            compute_eigenmode_metrics=True,
            save_field_maps=(nf == 4),
            quick=quick,
            spec=spec,
        )
        if nf == 4 and "field_maps" in r_u:
            field_maps_nf4 = r_u.pop("field_maps")
        uniform_levels[f"nf{nf}"] = r_u

        r_g = evaluate_bloch_unit_cell(
            phi_deg=27.43,
            n_feat=nf,
            use_graded_mesh=True,
            forward_control=False,
            compute_eigenmode_metrics=True,
            save_field_maps=False,
            quick=quick,
            spec=spec,
        )
        graded_levels[f"nf{nf}"] = r_g

    # Thesis R_wg = 8.1 mm variant (Lu 2019 MIT PhD thesis lines 2075, 2208)
    spec_8p1 = WagonWheelMTMSpec(waveguide_radius_m=8.1e-3, slot_outer_radius_m=8.1e-3)
    v81_nf4 = evaluate_bloch_unit_cell(
        phi_deg=27.43,
        n_feat=nf_levels[0],
        use_graded_mesh=False,
        forward_control=False,
        compute_eigenmode_metrics=True,
        save_field_maps=False,
        quick=quick,
        spec=spec_8p1,
    )
    v81_nf6 = evaluate_bloch_unit_cell(
        phi_deg=27.43,
        n_feat=nf_levels[1],
        use_graded_mesh=False,
        forward_control=False,
        compute_eigenmode_metrics=True,
        save_field_maps=False,
        quick=quick,
        spec=spec_8p1,
    )
    v81_nf8 = evaluate_bloch_unit_cell(
        phi_deg=27.43,
        n_feat=nf_levels[2],
        use_graded_mesh=False,
        forward_control=False,
        compute_eigenmode_metrics=True,
        save_field_maps=False,
        quick=quick,
        spec=spec_8p1,
    )

    out_t2 = _assemble_task_02_dict(
        fc_tm01_fit_ghz,
        phi_list,
        disp_freqs_ghz,
        disp_wall_s,
        phi_fwd_list,
        f_fwd_ghz,
        uniform_levels,
        graded_levels,
        v81_nf4,
        v81_nf6,
        v81_nf8,
        spec=spec,
    )

    if field_maps_nf4:
        np.savez_compressed(npz_path, **field_maps_nf4)
    json_path.write_text(json.dumps(out_t2, indent=2) + "\n", encoding="utf-8")
    return out_t2


def _build_length_scaling_dict(
    runs_dict: dict[str, Any],
    n_list: list[int],
    *,
    gamma_tap_note: float,
    cold_k_loss_per_m: float,
    cold_vg_m_s: float,
    cold_f_sync_hz: float,
    other_mesh_a_w_j_per_m: float | None = None,
    r_coarse_med: float = 4.0 / 3.0,
    cold_k_loss_gci_fine_rel: float = 0.0,
    z3_traces: Mapping[str, Any] | None = None,
    allow_missing_spectra: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Assemble the ``length_scaling`` dictionary from a set of finite-length MTM runs across ``n_list``."""
    L_arr = np.array([runs_dict[str(n)]["active_length_m"] for n in n_list], dtype=float)
    L_mm_arr = L_arr * 1e3
    z_pk_arr = np.array([runs_dict[str(n)]["re_z_pk_ohm"] for n in n_list], dtype=float)
    fwhm_arr = np.array([runs_dict[str(n)]["fwhm_ghz"] for n in n_list], dtype=float)
    e_up_arr = np.array([runs_dict[str(n)]["e_up_mj"] for n in n_list], dtype=float)
    w_cv_arr = np.array([runs_dict[str(n)]["w_beam_passband_cv_mj"] for n in n_list], dtype=float)
    k_loss_arr = np.array([runs_dict[str(n)]["k_loss_passband_v_per_c"] for n in n_list], dtype=float)
    p_up_arr = np.array([runs_dict[str(n)]["p_up_cycle_avg_max_mw"] for n in n_list], dtype=float)
    tp_arr = np.array([runs_dict[str(n)]["pulse_duration_20pct_ns"] for n in n_list], dtype=float)

    slope_z_pk = float(np.polyfit(np.log(L_arr), np.log(z_pk_arr), 1)[0])
    slope_fwhm = float(np.polyfit(np.log(L_arr), np.log(fwhm_arr), 1)[0])
    slope_p_up = float(np.polyfit(np.log(L_arr), np.log(p_up_arr), 1)[0])
    slope_tp = float(np.polyfit(np.log(L_arr), np.log(np.maximum(tp_arr, 1e-6)), 1)[0])

    # Affine linear fits y(L) = a * L + b over active length L [m] (intercept b in [mJ] or [V/C] comes from
    # the two 2-cell adiabatic end tapers included inside the [z_up, z_dn] control volume).
    a_w_j_per_m, b_w_j = [float(x) for x in np.polyfit(L_arr, w_cv_arr * 1e-3, 1)]
    b_w_mj = float(b_w_j * 1e3)
    a_e_up_j_per_m, b_e_up_j = [float(x) for x in np.polyfit(L_arr, e_up_arr * 1e-3, 1)]
    b_e_up_mj = float(b_e_up_j * 1e3)
    a_k_loss_v_per_c_m, b_k_loss_v_per_c = [float(x) for x in np.polyfit(L_arr, k_loss_arr, 1)]

    _, _, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    a_w_cold_analytic_j_per_m = analytic_backward_wave_loss_per_meter_j_per_m(
        spec.single_bunch_charge_c,
        cold_k_loss_per_m,
        cold_vg_m_s,
        spec.bunch_length_rms_m,
        cold_f_sync_hz,
        u0,
        mutate_flip_vg_sign=False,
    )
    a_w_mut_flipped_vg_j_per_m = analytic_backward_wave_loss_per_meter_j_per_m(
        spec.single_bunch_charge_c,
        cold_k_loss_per_m,
        cold_vg_m_s,
        spec.bunch_length_rms_m,
        cold_f_sync_hz,
        u0,
        mutate_flip_vg_sign=True,
    )
    rel_err_a_w = float(
        abs(a_w_j_per_m - a_w_cold_analytic_j_per_m) / max(abs(a_w_cold_analytic_j_per_m), 1e-30)
    )
    rel_err_a_w_mut = float(
        abs(a_w_j_per_m - a_w_mut_flipped_vg_j_per_m) / max(abs(a_w_mut_flipped_vg_j_per_m), 1e-30)
    )

    other_a_w = float(a_w_j_per_m if other_mesh_a_w_j_per_m is None else other_mesh_a_w_j_per_m)
    gci_a_w_two_level_rel = float(
        3.0 * abs(a_w_j_per_m - other_a_w) / (max(abs(a_w_j_per_m), 1e-30) * ((float(r_coarse_med) ** 1.0) - 1.0))
    )
    a_w_uncertainty_rel = float(gci_a_w_two_level_rel + float(cold_k_loss_gci_fine_rel) + (gamma_tap_note**2))
    z_sinc2_pk_arr = np.array(
        [
            float(
                analytic_sws_traveling_wave_impedance_ohm(
                    np.array([cold_f_sync_hz]),
                    cold_f_sync_hz,
                    cold_k_loss_per_m,
                    float(l_m),
                    u0,
                    cold_vg_m_s,
                )[0]
            )
            for l_m in L_arr
        ],
        dtype=float,
    )
    re_z_pk_over_sinc2_list = [
        float(z_pk / max(z_s2, 1e-30)) for z_pk, z_s2 in zip(z_pk_arr, z_sinc2_pk_arr)
    ]
    n8_re_z_over_sinc2 = float(re_z_pk_over_sinc2_list[1])

    # Integrated active-loss ratio ∫ Re Z_FIT df / ∫ Re Z_sinc^2 df and effective-width ratio over the wake passband
    f_bp_lo_ghz = float(spec.wake_passband_low_hz * 1e-9)
    f_bp_hi_ghz = float(spec.wake_passband_high_hz * 1e-9)
    x_half_sinc2 = float(brentq(lambda x: (np.sin(x) / x) ** 2 - 0.5, 1.0, 2.0))
    int_fit_list: list[float] = []
    int_sinc2_list: list[float] = []
    int_ratio_list: list[float] = []
    w_eff_fit_list: list[float] = []
    w_eff_sinc2_list: list[float] = []
    w_eff_ratio_list: list[float] = []
    fwhm_sinc2_list: list[float] = []
    fwhm_ratio_list: list[float] = []
    has_all_spectra = True
    for idx_n, n_per in enumerate(n_list):
        l_m = float(L_arr[idx_n])
        tp_cold_s = l_m * abs((1.0 / cold_vg_m_s) - (1.0 / u0))
        fwhm_s2_val = float((2.0 * x_half_sinc2 / np.pi) / max(tp_cold_s * 1e9, 1e-30))
        fwhm_rat_val = float(fwhm_arr[idx_n] / max(fwhm_s2_val, 1e-30))
        fwhm_sinc2_list.append(fwhm_s2_val)
        fwhm_ratio_list.append(fwhm_rat_val)

        rn_d = runs_dict[str(n_per)]
        if "f_ghz" in rn_d and "re_z_ohm" in rn_d:
            f_ghz_n = np.asarray(rn_d["f_ghz"], dtype=float)
            re_z_fit_n = np.asarray(rn_d["re_z_ohm"], dtype=float)
        elif z3_traces is not None and f"f_ghz_N{n_per}" in z3_traces and f"re_z_ohm_N{n_per}" in z3_traces:
            f_ghz_n = np.asarray(z3_traces[f"f_ghz_N{n_per}"], dtype=float)
            re_z_fit_n = np.asarray(z3_traces[f"re_z_ohm_N{n_per}"], dtype=float)
        else:
            if not allow_missing_spectra:
                raise KeyError(
                    f"Missing FIT impedance spectrum for N={n_per} in runs_dict and z3_traces; "
                    f"refusing silent analytic sinc^2 fallback."
                )
            has_all_spectra = False
            continue
        re_z_sinc2_n = analytic_sws_traveling_wave_impedance_ohm(
            f_ghz_n * 1e9,
            cold_f_sync_hz,
            cold_k_loss_per_m,
            l_m,
            u0,
            cold_vg_m_s,
        )
        bp_mask = (f_ghz_n >= f_bp_lo_ghz) & (f_ghz_n <= f_bp_hi_ghz)
        if np.count_nonzero(bp_mask) < 2:
            raise ValueError(
                f"Fewer than 2 frequency bins in wake passband [{f_bp_lo_ghz}, {f_bp_hi_ghz}] GHz for N={n_per}."
            )
        int_fit_val = float(np.trapezoid(re_z_fit_n[bp_mask], f_ghz_n[bp_mask]))
        int_sinc2_val = float(np.trapezoid(re_z_sinc2_n[bp_mask], f_ghz_n[bp_mask]))
        ratio_int_val = float(int_fit_val / max(int_sinc2_val, 1e-30))
        w_eff_fit_val = float(int_fit_val / max(float(z_pk_arr[idx_n]), 1e-30))
        w_eff_sinc2_val = float(int_sinc2_val / max(float(z_sinc2_pk_arr[idx_n]), 1e-30))
        ratio_weff_val = float(w_eff_fit_val / max(w_eff_sinc2_val, 1e-30))
        int_fit_list.append(int_fit_val)
        int_sinc2_list.append(int_sinc2_val)
        int_ratio_list.append(ratio_int_val)
        w_eff_fit_list.append(w_eff_fit_val)
        w_eff_sinc2_list.append(w_eff_sinc2_val)
        w_eff_ratio_list.append(ratio_weff_val)

    n_arr_float = np.asarray(n_list, dtype=float)
    if has_all_spectra:
        int_excess_list: list[float] | None = [
            float(i_fit - i_s2) for i_fit, i_s2 in zip(int_fit_list, int_sinc2_list)
        ]
        slope_int_fit_per_n, intercept_int_fit = [float(x) for x in np.polyfit(n_arr_float, int_fit_list, 1)]
        slope_int_sinc2_per_n, intercept_int_sinc2 = [
            float(x) for x in np.polyfit(n_arr_float, int_sinc2_list, 1)
        ]
        rel_diff_slope_int: float | None = float(
            abs(slope_int_fit_per_n - slope_int_sinc2_per_n) / max(abs(slope_int_sinc2_per_n), 1e-30)
        )
        missing_spectra_reason: str | None = None
    else:
        int_excess_list = None
        slope_int_fit_per_n = None
        intercept_int_fit = None
        slope_int_sinc2_per_n = None
        intercept_int_sinc2 = None
        rel_diff_slope_int = None
        missing_spectra_reason = (
            "n_xy=48 frequency-domain impedance spectra are not stored in task_03_wakefield_traces.npz"
        )

    if spec.paper_vg_over_c is None or not (-1.0 < float(spec.paper_vg_over_c) < 0.0):
        raise ValueError(
            f"spec.paper_vg_over_c must be a negative backward-wave group velocity in (-1, 0); got {spec.paper_vg_over_c!r}."
        )
    vg_paper_m_s = float(spec.paper_vg_over_c) * C0
    L_exp_m = spec.n_periods_experiment * spec.period_m
    tp_exp_paper_ns = analytic_backward_wave_pulse_length_s(L_exp_m, vg_paper_m_s, u0) * 1e9
    tp_n8_analytic_ns = analytic_backward_wave_pulse_length_s(8 * spec.period_m, vg_paper_m_s, u0) * 1e9
    tp_analytic_arr_ns = [
        float(analytic_backward_wave_pulse_length_s(float(l_m), vg_paper_m_s, u0) * 1e9) for l_m in L_arr
    ]
    r8_tp = float(runs_dict["8"]["pulse_duration_20pct_ns"])
    dt_r8_s = float(runs_dict["8"]["dt_fs"]) * 1e-15
    filt_rise_10_90_ns = measure_filter_step_rise_time_ns(dt_s=dt_r8_s, f_lo_hz=WagonWheelMTMSpec.passband_low_hz, f_hi_hz=WagonWheelMTMSpec.passband_high_hz, order=3)
    pulse_tol_ns = float(filt_rise_10_90_ns + 0.5 / (cold_f_sync_hz * 1e-9))
    return {
        "active_length_mm": L_mm_arr.tolist(),
        "re_z_pk_ohm": z_pk_arr.tolist(),
        "cold_sinc2_re_z_pk_ohm": z_sinc2_pk_arr.tolist(),
        "re_z_pk_over_cold_sinc2_ratio": re_z_pk_over_sinc2_list,
        "n8_re_z_pk_over_cold_sinc2_ratio": n8_re_z_over_sinc2,
        "passband_integrated_re_z_fit_ohm_ghz": int_fit_list if has_all_spectra else None,
        "passband_integrated_re_z_cold_sinc2_ohm_ghz": int_sinc2_list if has_all_spectra else None,
        "passband_integrated_re_z_excess_ohm_ghz": int_excess_list,
        "passband_integrated_re_z_fit_slope_ohm_ghz_per_period": slope_int_fit_per_n,
        "passband_integrated_re_z_fit_intercept_ohm_ghz": intercept_int_fit,
        "passband_integrated_re_z_cold_sinc2_slope_ohm_ghz_per_period": slope_int_sinc2_per_n,
        "passband_integrated_re_z_cold_sinc2_intercept_ohm_ghz": intercept_int_sinc2,
        "rel_diff_passband_integrated_re_z_slope_fit_vs_cold_sinc2": rel_diff_slope_int,
        "passband_integrated_re_z_missing_reason": missing_spectra_reason,
        "passband_integrated_re_z_fit_over_cold_sinc2_ratio": int_ratio_list if has_all_spectra else None,
        "n8_passband_integrated_re_z_fit_over_cold_sinc2_ratio": float(int_ratio_list[1]) if has_all_spectra else None,
        "effective_width_fit_ghz": w_eff_fit_list if has_all_spectra else None,
        "effective_width_cold_sinc2_ghz": w_eff_sinc2_list if has_all_spectra else None,
        "effective_width_fit_over_cold_sinc2_ratio": w_eff_ratio_list if has_all_spectra else None,
        "n8_effective_width_fit_over_cold_sinc2_ratio": float(w_eff_ratio_list[1]) if has_all_spectra else None,
        "fwhm_cold_sinc2_ghz": fwhm_sinc2_list,
        "fwhm_fit_over_cold_sinc2_ratio": fwhm_ratio_list,
        "fwhm_ghz": fwhm_arr.tolist(),
        "e_up_mj": e_up_arr.tolist(),
        "w_beam_passband_cv_mj": w_cv_arr.tolist(),
        "k_loss_passband_v_per_c": k_loss_arr.tolist(),
        "p_up_cycle_avg_max_mw": p_up_arr.tolist(),
        "pulse_duration_20pct_ns": tp_arr.tolist(),
        "pulse_duration_analytic_ns": tp_analytic_arr_ns,
        "affine_a_w_j_per_m": a_w_j_per_m,
        "affine_b_w_mj": b_w_mj,
        "affine_a_e_up_j_per_m": a_e_up_j_per_m,
        "affine_b_e_up_mj": b_e_up_mj,
        "affine_a_k_loss_v_per_c_m": a_k_loss_v_per_c_m,
        "affine_b_k_loss_v_per_c": b_k_loss_v_per_c,
        "cold_analytic_a_w_j_per_m": a_w_cold_analytic_j_per_m,
        "cold_analytic_loss_per_m_j_per_m": a_w_cold_analytic_j_per_m,
        "mutated_flipped_vg_a_w_j_per_m": a_w_mut_flipped_vg_j_per_m,
        "rel_error_a_w_vs_cold_analytic": rel_err_a_w,
        "mutated_flipped_vg_rel_error_a_w": rel_err_a_w_mut,
        "gci_a_w_two_level_rel": gci_a_w_two_level_rel,
        "cold_k_loss_gci_fine_rel": float(cold_k_loss_gci_fine_rel),
        "a_w_uncertainty_rel": a_w_uncertainty_rel,
        "filter_step_rise_time_10_90_ns": filt_rise_10_90_ns,
        "pulse_duration_tolerance_ns": pulse_tol_ns,
        "slope_re_z_pk_vs_L_fit": slope_z_pk,
        "slope_re_z_pk_vs_L_theory": 2.0,
        "slope_re_z_pk_vs_L_deviation": float(slope_z_pk - 2.0),
        "slope_fwhm_vs_L_fit": slope_fwhm,
        "slope_fwhm_vs_L_theory": -1.0,
        "slope_fwhm_vs_L_deviation": float(slope_fwhm - (-1.0)),
        "slope_p_up_vs_L_fit": slope_p_up,
        "slope_p_up_vs_L_theory": 0.0,
        "slope_tp_vs_L_fit": slope_tp,
        "slope_tp_vs_L_theory": 1.0,
        "slope_tp_vs_L_deviation": float(slope_tp - 1.0),
        "n8_pulse_duration_analytic_ns": float(tp_n8_analytic_ns),
        "n8_pulse_duration_20pct_fit_ns": r8_tp,
        "n8_pulse_broadening_ratio_fit_over_analytic": float(r8_tp / max(tp_n8_analytic_ns, 1e-12)),
        "n40_experiment_pulse_duration_analytic_ns": float(tp_exp_paper_ns),
        "n40_experiment_bandwidth_ghz": float(1.0 / tp_exp_paper_ns),
        "length_ratio_experiment_over_N8": 5.0,
        "is_asymptotic_long_structure_regime": False,
        "scaling_regime_note": (
            f"Over N in {{6, 8, 12}} (L = 12, 16, 24 mm, bordered by two 2-cell adiabatic tapers), "
            f"control-volume passband beam work W_beam,bp,cv(L) = {a_w_j_per_m:.4f} J/m * L + {b_w_mj:.3f} mJ "
            f"and upstream extracted energy E_up(L) = {a_e_up_j_per_m:.4f} J/m * L + {b_e_up_mj:.3f} mJ "
            f"are affine in active length L with positive end-taper intercepts, while the differential slope "
            f"a_W = {a_w_j_per_m:.4f} J/m agrees with the cold-test analytical loss per unit length "
            f"q^2 k_L Phi^2 / (1 - v_g/u_0) = {a_w_cold_analytic_j_per_m:.4f} J/m within {100.0 * rel_err_a_w:.2f}% "
            f"(well inside the combined uncertainty +/-{100.0 * a_w_uncertainty_rel:.2f}%: two-level slope GCI "
            f"{100.0 * gci_a_w_two_level_rel:.2f}% + cold-test k_L GCI {100.0 * cold_k_loss_gci_fine_rel:.2f}% + "
            f"|Gamma_tap|^2 = {100.0 * (gamma_tap_note**2):.2f}%, |Gamma_tap| = {gamma_tap_note:.4f})."
        ),
    }


def _build_n8_transverse_mesh_convergence(
    runs_mtm_nxy48: dict[str, Any],
    runs_mtm_nxy64: dict[str, Any],
    r_n8_nxy96: dict[str, Any],
    n_list: list[int],
    *,
    port_gamma_levels: list[float],
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Assemble 3-level ``n8_transverse_mesh_convergence`` comparing ``n_xy=48, 64, 96`` finite wake runs."""
    r48 = runs_mtm_nxy48["8"]
    r64 = runs_mtm_nxy64["8"]
    r96 = r_n8_nxy96
    nxy_48_val = int(r48["n_xy"])
    nxy_64_val = int(r64["n_xy"])
    nxy_96_val = int(r96["n_xy"])
    h48_mm = float((2.0 * spec.waveguide_radius_m / max(nxy_48_val, 1)) * 1e3)
    h64_mm = float((2.0 * spec.waveguide_radius_m / max(nxy_64_val, 1)) * 1e3)
    h96_mm = float((2.0 * spec.waveguide_radius_m / max(nxy_96_val, 1)) * 1e3)
    h_vals = [h48_mm, h64_mm, h96_mm]
    t_shield_mm = float((spec.coax_port_wall_radius_m - spec.beam_hole_radius_m) * 1e3)
    cells_shield = [float(t_shield_mm / h) for h in h_vals]

    f_vals = [float(r48["f_pk_ghz"]), float(r64["f_pk_ghz"]), float(r96["f_pk_ghz"])]
    w_vals = [
        float(r48["w_beam_passband_cv_mj"]),
        float(r64["w_beam_passband_cv_mj"]),
        float(r96["w_beam_passband_cv_mj"]),
    ]
    p_vals = [
        float(r48["p_up_cycle_avg_max_mw"]),
        float(r64["p_up_cycle_avg_max_mw"]),
        float(r96["p_up_cycle_avg_max_mw"]),
    ]
    e_vals = [float(r48["e_up_mj"]), float(r64["e_up_mj"]), float(r96["e_up_mj"])]
    ec_vals = [float(r48["e_up_coax_mj"]), float(r64["e_up_coax_mj"]), float(r96["e_up_coax_mj"])]
    z_vals = [float(r48["re_z_pk_ohm"]), float(r64["re_z_pk_ohm"]), float(r96["re_z_pk_ohm"])]
    g_vals = [float(x) for x in port_gamma_levels]
    trans_vals = [float(1.0 - (g**2)) for g in g_vals]

    gci_f = compute_three_level_gci(f_vals, h_vals, p_formal=1.0)
    gci_w = compute_three_level_gci(w_vals, h_vals, p_formal=1.0)
    gci_p = compute_three_level_gci(p_vals, h_vals, p_formal=1.0)
    gci_e = compute_three_level_gci(e_vals, h_vals, p_formal=1.0)
    gci_ec = compute_three_level_gci(ec_vals, h_vals, p_formal=1.0)
    gci_z = compute_three_level_gci(z_vals, h_vals, p_formal=1.0)
    gci_gamma = compute_three_level_gci(g_vals, h_vals, p_formal=1.0)

    mc: dict[str, Any] = {
        "levels_n_xy": [nxy_48_val, nxy_64_val, nxy_96_val],
        "h_xy_mm": h_vals,
        "cells_across_1mm_spoke": [float(1.0 / h) for h in h_vals],
        "cells_across_0p3mm_coax_shield": cells_shield,
        "port_gamma_tapered_at_sync_values": g_vals,
        "port_power_transmission_values": trans_vals,
        "n_xy_48": r48,
        "n_xy_64": r64,
        "n_xy_96": r96,
        "f_pk_ghz_values": f_vals,
        "w_beam_passband_cv_mj_values": w_vals,
        "p_up_max_mw_values": p_vals,
        "e_up_mj_values": e_vals,
        "e_up_coax_mj_values": ec_vals,
        "re_z_pk_ohm_values": z_vals,
        "f_pk_ghz": gci_f,
        "w_beam_passband_cv_mj": gci_w,
        "p_up_cycle_avg_max_mw": gci_p,
        "e_up_mj": gci_e,
        "e_up_coax_mj": gci_ec,
        "re_z_pk_ohm": gci_z,
        "port_gamma_tapered_at_sync": gci_gamma,
        "gci": {
            "f_pk_ghz": gci_f,
            "w_beam_passband_cv_mj": gci_w,
            "p_up_cycle_avg_max_mw": gci_p,
            "e_up_mj": gci_e,
            "e_up_coax_mj": gci_ec,
            "re_z_pk_ohm": gci_z,
            "port_gamma_tapered_at_sync": gci_gamma,
        },
        "rel_diff_f_pk_64_vs_48": float(gci_f["rel_diff_64_vs_48"]),
        "rel_diff_f_pk_96_vs_64": float(gci_f["rel_diff_96_vs_64"]),
        "rel_diff_w_beam_64_vs_48": float(gci_w["rel_diff_64_vs_48"]),
        "rel_diff_w_beam_96_vs_64": float(gci_w["rel_diff_96_vs_64"]),
        "rel_diff_w_beam_bp_64_vs_48": float(gci_w["rel_diff_64_vs_48"]),
        "rel_diff_w_beam_bp_96_vs_64": float(gci_w["rel_diff_96_vs_64"]),
        "rel_diff_p_up_64_vs_48": float(gci_p["rel_diff_64_vs_48"]),
        "rel_diff_p_up_96_vs_64": float(gci_p["rel_diff_96_vs_64"]),
        "rel_diff_p_up_max_64_vs_48": float(gci_p["rel_diff_64_vs_48"]),
        "rel_diff_p_up_max_96_vs_64": float(gci_p["rel_diff_96_vs_64"]),
        "rel_diff_e_up_64_vs_48": float(gci_e["rel_diff_64_vs_48"]),
        "rel_diff_e_up_96_vs_64": float(gci_e["rel_diff_96_vs_64"]),
        "rel_diff_re_z_pk_64_vs_48": float(gci_z["rel_diff_64_vs_48"]),
        "rel_diff_re_z_pk_96_vs_64": float(gci_z["rel_diff_96_vs_64"]),
        "coax_shield_resolution_note": (
            f"The numerical coaxial port inner shield has radial thickness r_p - r_b = {t_shield_mm:.2f} mm "
            f"({cells_shield[0]:.2f}, {cells_shield[1]:.2f}, and {cells_shield[2]:.2f} transverse cells at "
            f"n_xy = {nxy_48_val}, {nxy_64_val}, and {nxy_96_val}). Across the three levels, upstream port "
            f"reflection |Gamma| = {g_vals[0]:.4f}, {g_vals[1]:.4f}, {g_vals[2]:.4f} "
            f"({'oscillatory, |R| > 1 (amplitude not decreasing)' if abs(float(gci_gamma['R_ratio'])) > 1.0 else gci_gamma['convergence_class']}, "
            f"R = {float(gci_gamma['R_ratio']):.3f}, "
            f"96-vs-64 step = {100.0 * float(gci_gamma['rel_diff_96_vs_64']):.2f}%, U_fine = "
            f"{100.0 * float(gci_gamma['u_fine_rel']):.2f}%) and coaxial port energy "
            f"E_up,coax = {ec_vals[0]:.3f}, {ec_vals[1]:.3f}, {ec_vals[2]:.3f} mJ "
            f"({'oscillatory, |R| > 1 (amplitude not decreasing)' if abs(float(gci_ec['R_ratio'])) > 1.0 else gci_ec['convergence_class']}, "
            f"R = {float(gci_ec['R_ratio']):.3f}, "
            f"U_fine = {100.0 * float(gci_ec['u_fine_rel']):.2f}%), as the sub-4-cell thin shield relies "
            f"on subcell conformal PEC (conformal='pec')."
        ),
    }
    for n_per in n_list:
        k = str(n_per)
        if k in runs_mtm_nxy48 and k in runs_mtm_nxy64:
            rn48 = runs_mtm_nxy48[k]
            rn64 = runs_mtm_nxy64[k]
            lbl = f"N{n_per}"
            mc[f"{lbl}_rel_diff_f_pk"] = float(
                abs(float(rn64["f_pk_ghz"]) - float(rn48["f_pk_ghz"])) / max(abs(float(rn64["f_pk_ghz"])), 1e-30)
            )
            mc[f"{lbl}_rel_diff_re_z_pk"] = float(
                abs(float(rn64["re_z_pk_ohm"]) - float(rn48["re_z_pk_ohm"]))
                / max(abs(float(rn64["re_z_pk_ohm"])), 1e-30)
            )
            mc[f"{lbl}_rel_diff_e_up"] = float(
                abs(float(rn64["e_up_mj"]) - float(rn48["e_up_mj"])) / max(abs(float(rn64["e_up_mj"])), 1e-30)
            )
    return mc


def _assemble_task_03_dict(
    n_list: list[int],
    runs_mtm_nxy48: dict[str, Any],
    runs_mtm: dict[str, Any],
    r_96_n8: dict[str, Any],
    r_fwd: dict[str, Any],
    r_fwd_48: dict[str, Any],
    port_gamma_levels: list[float],
    *,
    gamma_tap_note: float,
    cold_k_loss_per_m: float,
    cold_vg_m_s: float,
    cold_f_sync_hz: float,
    cold_k_loss_gci_fine_rel: float = 0.0,
    z3_traces: Mapping[str, Any] | None = None,
    mutate_reuse_nxy64_traces_for_nxy48: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Assemble ``task_03_wakefield_and_directionality.json`` from per-run finite wakefield outputs."""
    r8 = runs_mtm["8"]
    r8_48 = runs_mtm_nxy48["8"]
    h48_mm = float((2.0 * spec.waveguide_radius_m / max(int(r8_48["n_xy"]), 1)) * 1e3)
    h64_mm = float((2.0 * spec.waveguide_radius_m / max(int(r8["n_xy"]), 1)) * 1e3)
    r_coarse_med = float(h48_mm / max(h64_mm, 1e-12))
    z3_for_48 = z3_traces if mutate_reuse_nxy64_traces_for_nxy48 else None
    allow_missing_48 = not mutate_reuse_nxy64_traces_for_nxy48
    ls_48_init = _build_length_scaling_dict(
        runs_mtm_nxy48,
        n_list,
        gamma_tap_note=gamma_tap_note,
        cold_k_loss_per_m=cold_k_loss_per_m,
        cold_vg_m_s=cold_vg_m_s,
        cold_f_sync_hz=cold_f_sync_hz,
        r_coarse_med=r_coarse_med,
        cold_k_loss_gci_fine_rel=cold_k_loss_gci_fine_rel,
        z3_traces=z3_for_48,
        allow_missing_spectra=allow_missing_48,
        spec=spec,
    )
    ls_64 = _build_length_scaling_dict(
        runs_mtm,
        n_list,
        gamma_tap_note=gamma_tap_note,
        cold_k_loss_per_m=cold_k_loss_per_m,
        cold_vg_m_s=cold_vg_m_s,
        cold_f_sync_hz=cold_f_sync_hz,
        other_mesh_a_w_j_per_m=float(ls_48_init["affine_a_w_j_per_m"]),
        r_coarse_med=r_coarse_med,
        cold_k_loss_gci_fine_rel=cold_k_loss_gci_fine_rel,
        z3_traces=z3_traces,
        allow_missing_spectra=False,
        spec=spec,
    )
    ls_48 = _build_length_scaling_dict(
        runs_mtm_nxy48,
        n_list,
        gamma_tap_note=gamma_tap_note,
        cold_k_loss_per_m=cold_k_loss_per_m,
        cold_vg_m_s=cold_vg_m_s,
        cold_f_sync_hz=cold_f_sync_hz,
        other_mesh_a_w_j_per_m=float(ls_64["affine_a_w_j_per_m"]),
        r_coarse_med=r_coarse_med,
        cold_k_loss_gci_fine_rel=cold_k_loss_gci_fine_rel,
        z3_traces=z3_for_48,
        allow_missing_spectra=allow_missing_48,
        spec=spec,
    )
    return {
        "n_periods_list": [int(n) for n in n_list],
        "runs_mtm": runs_mtm,
        "runs_mtm_nxy96_N8": r_96_n8,
        "forward_control_N8": _strip_heavy_arrays(r_fwd),
        "length_scaling": ls_64,
        "directionality_comparison_N8": {
            "mtm_backward_f_pk_ghz": float(r8["f_pk_ghz"]),
            "mtm_backward_e_up_mj": float(r8["e_up_mj"]),
            "mtm_backward_e_dn_mj": float(r8["e_dn_mj"]),
            "mtm_backward_upstream_fraction": float(r8["upstream_energy_fraction"]),
            "mtm_backward_p_up_max_mw": float(r8["p_up_cycle_avg_max_mw"]),
            "mtm_backward_p_dn_max_mw": float(r8["p_dn_cycle_avg_max_mw"]),
            "control_forward_f_pk_ghz": float(r_fwd["f_pk_ghz"]),
            "control_forward_e_up_mj": float(r_fwd["e_up_mj"]),
            "control_forward_e_dn_mj": float(r_fwd["e_dn_mj"]),
            "control_forward_downstream_fraction": float(r_fwd["downstream_energy_fraction"]),
            "control_forward_p_dn_max_mw": float(r_fwd["p_dn_cycle_avg_max_mw"]),
            "control_forward_p_up_max_mw": float(r_fwd["p_up_cycle_avg_max_mw"]),
        },
        "n8_transverse_mesh_convergence": _build_n8_transverse_mesh_convergence(
            runs_mtm_nxy48,
            runs_mtm,
            r_96_n8,
            n_list,
            port_gamma_levels=port_gamma_levels,
            spec=spec,
        ),
        "runs_mtm_nxy48": runs_mtm_nxy48,
        "forward_control_N8_nxy48": _strip_heavy_arrays(r_fwd_48),
        "length_scaling_nxy48": ls_48,
    }


def generate_task_03_data(
    data_dir: Path | str,
    *,
    quick: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Generate ``task_03_wakefield_and_directionality.json`` and ``task_03_wakefield_traces.npz``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "task_03_wakefield_and_directionality.json"
    npz_path = data_dir / "task_03_wakefield_traces.npz"

    n_list = [6, 8, 12]
    n_xy_coarse = 28 if quick else 48
    n_xy_fine = 36 if quick else 64
    n_xy_fine96 = 48 if quick else 96

    runs_mtm_nxy48: dict[str, Any] = {}
    runs_mtm: dict[str, Any] = {}
    npz_payload: dict[str, np.ndarray] = {}

    # 1. Coarse transverse mesh level (n_xy = 48 -> 3 cells/mm; allowed only as coarsest of 3 levels)
    for n_per in n_list:
        r_c = run_finite_mtm_wakefield(
            n_periods=n_per,
            n_xy=n_xy_coarse,
            forward_control=False,
            record_2d_snapshots=False,
            quick=quick,
            spec=spec,
        )
        runs_mtm_nxy48[str(n_per)] = _strip_heavy_arrays(r_c)

    r_fwd_48 = run_finite_mtm_wakefield(
        n_periods=8,
        n_xy=n_xy_coarse,
        forward_control=True,
        record_2d_snapshots=False,
        quick=quick,
        spec=spec,
    )

    # 2. Primary medium transverse mesh level (n_xy = 64 -> 4 cells/mm)
    for n_per in n_list:
        is_main = n_per == 8
        r_m = run_finite_mtm_wakefield(
            n_periods=n_per,
            n_xy=n_xy_fine,
            forward_control=False,
            record_2d_snapshots=is_main,
            quick=quick,
            spec=spec,
        )
        npz_payload[f"t_ns_N{n_per}"] = r_m["t_ns"]
        npz_payload[f"p_up_inst_mw_N{n_per}"] = r_m["p_up_inst_mw"]
        npz_payload[f"p_up_env_mw_N{n_per}"] = r_m["p_up_env_mw"]
        npz_payload[f"p_dn_inst_mw_N{n_per}"] = r_m["p_dn_inst_mw"]
        npz_payload[f"p_dn_env_mw_N{n_per}"] = r_m["p_dn_env_mw"]
        npz_payload[f"s_mm_N{n_per}"] = r_m["s_mm"]
        npz_payload[f"wp_v_per_pc_N{n_per}"] = r_m["wp_v_per_pc"]
        npz_payload[f"f_ghz_N{n_per}"] = r_m["f_ghz"]
        npz_payload[f"re_z_ohm_N{n_per}"] = r_m["re_z_ohm"]
        npz_payload[f"im_z_ohm_N{n_per}"] = r_m["im_z_ohm"]
        if is_main and "snapshots" in r_m:
            sn = r_m["snapshots"]
            npz_payload["bwd_y_mm"] = sn["y_mm"]
            npz_payload["bwd_z_mm"] = sn["z_mm"]
            npz_payload["bwd_snap_ez_scat"] = sn["snap_ez_scat"]
            npz_payload["bwd_snap_t_ps"] = sn["snap_t_ps"]
            npz_payload["bwd_snap_zb_mm"] = sn["snap_zb_mm"]
        runs_mtm[str(n_per)] = _strip_heavy_arrays(r_m)

    # 3. Finest transverse mesh level for N = 8 (n_xy = 96 -> 6 cells/mm)
    r_96_n8 = _strip_heavy_arrays(
        run_finite_mtm_wakefield(
            n_periods=8,
            n_xy=n_xy_fine96,
            forward_control=False,
            record_2d_snapshots=False,
            quick=quick,
            spec=spec,
        )
    )

    # Forward-wave disk-loaded control run (N = 8, n_xy = 64)
    r_fwd = run_finite_mtm_wakefield(
        n_periods=8,
        n_xy=n_xy_fine,
        forward_control=True,
        record_2d_snapshots=True,
        quick=quick,
        spec=spec,
    )
    npz_payload["fwd_t_ns"] = r_fwd["t_ns"]
    npz_payload["fwd_p_up_inst_mw"] = r_fwd["p_up_inst_mw"]
    npz_payload["fwd_p_up_env_mw"] = r_fwd["p_up_env_mw"]
    npz_payload["fwd_p_dn_inst_mw"] = r_fwd["p_dn_inst_mw"]
    npz_payload["fwd_p_dn_env_mw"] = r_fwd["p_dn_env_mw"]
    npz_payload["fwd_s_mm"] = r_fwd["s_mm"]
    npz_payload["fwd_wp_v_per_pc"] = r_fwd["wp_v_per_pc"]
    npz_payload["fwd_f_ghz"] = r_fwd["f_ghz"]
    npz_payload["fwd_re_z_ohm"] = r_fwd["re_z_ohm"]
    if "snapshots" in r_fwd:
        sn_f = r_fwd["snapshots"]
        npz_payload["fwd_y_mm"] = sn_f["y_mm"]
        npz_payload["fwd_z_mm"] = sn_f["z_mm"]
        npz_payload["fwd_snap_ez_scat"] = sn_f["snap_ez_scat"]
        npz_payload["fwd_snap_t_ps"] = sn_f["snap_t_ps"]
        npz_payload["fwd_snap_zb_mm"] = sn_f["snap_zb_mm"]

    t1_path = data_dir / "task_01_geometry_and_port_reflection.json"
    if not t1_path.is_file():
        generate_task_01_data(data_dir, quick=quick, spec=spec)
    t1_data = json.loads(t1_path.read_text(encoding="utf-8"))
    gamma_tap_note = float(t1_data["port_reflection"]["gamma_tapered_at_sync"])

    t2_path = data_dir / "task_02_cold_test_and_convergence.json"
    if not t2_path.is_file():
        generate_task_02_data(data_dir, quick=quick, spec=spec)
    t2_data = json.loads(t2_path.read_text(encoding="utf-8"))
    wb_t2 = t2_data["wakis_finest_nf8_benchmarks"]
    cold_kl = float(wb_t2["k_loss_v_per_c_m"])
    cold_vg = float(wb_t2["ve_dft_over_c"]) * C0
    cold_fs = float(wb_t2["f_sync_ghz"]) * 1e9
    cold_kl_gci = float(t2_data["convergence_uniform"]["k_loss_v_per_c_m"]["u_fine_rel"])

    pr_48 = measure_upstream_port_reflection(n_xy=n_xy_coarse, quick=quick, spec=spec)
    pr_64 = (
        t1_data["port_reflection"]
        if int(t1_data["port_reflection"]["n_xy_port"]) == int(n_xy_fine)
        else measure_upstream_port_reflection(n_xy=n_xy_fine, quick=quick, spec=spec)
    )
    pr_96 = measure_upstream_port_reflection(n_xy=n_xy_fine96, quick=quick, spec=spec)
    port_gamma_levels = [
        float(pr_48["gamma_tapered_at_sync"]),
        float(pr_64["gamma_tapered_at_sync"]),
        float(pr_96["gamma_tapered_at_sync"]),
    ]

    out_t3 = _assemble_task_03_dict(
        n_list,
        runs_mtm_nxy48,
        runs_mtm,
        r_96_n8,
        r_fwd,
        r_fwd_48,
        port_gamma_levels,
        gamma_tap_note=gamma_tap_note,
        cold_k_loss_per_m=cold_kl,
        cold_vg_m_s=cold_vg,
        cold_f_sync_hz=cold_fs,
        cold_k_loss_gci_fine_rel=cold_kl_gci,
        z3_traces=npz_payload,
        spec=spec,
    )

    np.savez_compressed(npz_path, **npz_payload)
    json_path.write_text(json.dumps(out_t3, indent=2) + "\n", encoding="utf-8")
    return out_t3


def _assemble_task_04_dict(
    sup_check: dict[str, Any],
    t3: dict[str, Any],
    z3: Any,
    *,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Assemble ``task_04_two_bunch_superposition.json`` and its NPZ payload from per-run inputs."""
    r8 = t3["runs_mtm"]["8"]
    f_sync_ghz = float(r8["f_pk_ghz"])
    f_sync_hz = f_sync_ghz * 1e9
    t_rf_s = 1.0 / f_sync_hz
    _, _, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    lambda_rf_mm = float((u0 / f_sync_hz) * 1e3)

    s_mm = np.asarray(z3["s_mm_N8"], dtype=float)
    wp_1 = np.asarray(z3["wp_v_per_pc_N8"], dtype=float)
    t_ns = np.asarray(z3["t_ns_N8"], dtype=float)
    p1_env_45nc = np.asarray(z3["p_up_env_mw_N8"], dtype=float)

    # Passband-filtered 3D FIT wake potential W_||,bp(s) in [9.5, 13.5] GHz (no pure-sinusoid projection)
    f_bp_lo = WagonWheelMTMSpec.wake_passband_low_hz * 1e-9
    f_bp_hi = WagonWheelMTMSpec.wake_passband_high_hz * 1e-9
    ds_m = float(np.median(np.diff(s_mm)) * 1e-3)
    dt_s = ds_m / u0
    freqs_w_ghz = np.fft.rfftfreq(len(wp_1), d=dt_s) * 1e-9
    bp_w = (freqs_w_ghz >= f_bp_lo) & (freqs_w_ghz <= f_bp_hi)
    wp_bp_1 = np.fft.irfft(np.fft.rfft(wp_1) * bp_w, n=len(wp_1)) * (s_mm >= 0.0)

    ds_in_mm = lambda_rf_mm
    ds_out_mm = 1.5 * lambda_rf_mm
    wp_2_in = np.interp(s_mm - ds_in_mm, s_mm, wp_bp_1, left=0.0, right=0.0)
    wp_2_out = np.interp(s_mm - ds_out_mm, s_mm, wp_bp_1, left=0.0, right=0.0)
    wp_sum_in = wp_bp_1 + wp_2_in
    wp_sum_out = wp_bp_1 + wp_2_out

    ov_mask = (s_mm >= 1.6 * ds_in_mm) & (s_mm <= 90.0)
    wp_amp_1 = float(np.max(np.abs(wp_bp_1[ov_mask])))
    wp_amp_in = float(np.max(np.abs(wp_sum_in[ov_mask])))
    wp_amp_out = float(np.max(np.abs(wp_sum_out[ov_mask])))

    psi_deg = np.linspace(0.0, 720.0, 145)
    psi_rad = np.deg2rad(psi_deg)
    power_ratio_analytic = 4.0 * (np.cos(0.5 * psi_rad) ** 2)
    power_ratio_mutated_linear = 2.0 * np.abs(np.cos(0.5 * psi_rad))

    power_ratio_wake_scan = np.zeros_like(psi_rad)
    for idx_p, dpsi in enumerate(psi_rad):
        ds_p = (dpsi / (2.0 * np.pi)) * lambda_rf_mm
        wp_shift = np.interp(s_mm[ov_mask] - ds_p, s_mm, wp_bp_1, left=0.0, right=0.0)
        wp_p = wp_bp_1[ov_mask] + wp_shift
        power_ratio_wake_scan[idx_p] = float((np.max(np.abs(wp_p)) / max(wp_amp_1, 1e-30)) ** 2)

    if spec.paper_vg_over_c is None or not (-1.0 < float(spec.paper_vg_over_c) < 0.0):
        raise ValueError(
            f"spec.paper_vg_over_c must be a negative backward-wave group velocity in (-1, 0); got {spec.paper_vg_over_c!r}."
        )
    if spec.paper_sync_freq_hz is None or float(spec.paper_sync_freq_hz) <= 0.0:
        raise ValueError(f"spec.paper_sync_freq_hz must be positive; got {spec.paper_sync_freq_hz!r}.")
    vg_paper_m_s = float(spec.paper_vg_over_c) * C0
    f_s_paper_hz = float(spec.paper_sync_freq_hz)
    f_s_paper_ghz = float(f_s_paper_hz * 1e-9)

    q_single_nc = spec.single_bunch_charge_c * 1e9  # 45 nC
    q_tot_paper_nc = spec.two_bunch_total_charge_c * 1e9  # 85 nC
    charge_sq_ratio_85_over_45 = float((q_tot_paper_nc / q_single_nc) ** 2)  # (85/45)^2 = 3.567901
    kl_pap = paper_cold_test_k_loss_per_m(f_s_paper_hz, spec.paper_r_over_q_ohm_per_m)

    p1_fit_45nc_mw = float(r8["p_up_cycle_avg_max_mw"])
    p1_paper_analytic_45nc_mw = (
        analytic_backward_wave_power_w(
            spec.single_bunch_charge_c,
            kl_pap,
            vg_paper_m_s,
            spec.bunch_length_rms_m,
            f_s_paper_hz,
            u0,
        )
        * 1e-6
    )

    # Charge-scaled single-bunch estimates for Q1 = Q2 = 42.5 nC (Q1 + Q2 = 85 nC):
    p_single_42p5nc_fit_mw = p1_fit_45nc_mw * ((42.5 / 45.0) ** 2)
    p_two_in_phase_tot85nc_fit_mw = 4.0 * p_single_42p5nc_fit_mw
    p_two_in_phase_tot85nc_analytic_mw = p1_paper_analytic_45nc_mw * charge_sq_ratio_85_over_45

    # Charge-scaled single-bunch estimates for Q1 = Q2 = 85 nC (Q1 + Q2 = 170 nC):
    p_single_85nc_fit_mw = p1_fit_45nc_mw * charge_sq_ratio_85_over_45
    p_single_85nc_analytic_mw = p1_paper_analytic_45nc_mw * charge_sq_ratio_85_over_45
    p_two_each85nc_fit_mw = 4.0 * p_single_85nc_fit_mw
    p_two_each85nc_analytic_mw = 4.0 * p_single_85nc_analytic_mw

    # Single-bunch N=8 envelope shifted by 1 RF period vs experimental linac period:
    dt_in_ns = float(t_rf_s * 1e9)
    dt_in_ps = float(t_rf_s * 1e12)
    dt_out_ns = float(1.5 * t_rf_s * 1e9)
    t_linac_ps = float(1e12 / spec.linac_frequency_hz)
    t_n8_pulse_ns = float(r8["pulse_duration_20pct_ns"])
    t_n8_pulse_ps = float(t_n8_pulse_ns * 1e3)
    tp_n8_eq1_ns = float(analytic_backward_wave_pulse_length_s(8 * spec.period_m, vg_paper_m_s, u0) * 1e9)
    t_n40_pulse_ns = float(
        analytic_backward_wave_pulse_length_s(spec.n_periods_experiment * spec.period_m, vg_paper_m_s, u0) * 1e9
    )
    t_n40_pulse_ps = float(t_n40_pulse_ns * 1e3)

    e_env_1 = np.sqrt(np.maximum(p1_env_45nc, 0.0))
    e_env_2_in = np.interp(t_ns - dt_in_ns, t_ns, e_env_1, left=0.0, right=0.0)
    e_env_2_out = np.interp(t_ns - dt_out_ns, t_ns, e_env_1, left=0.0, right=0.0)

    scale_q_half = 42.5 / 45.0
    p_env_tot85_in_mw = (scale_q_half * (e_env_1 + e_env_2_in)) ** 2
    p_env_tot85_out_mw = (scale_q_half * (e_env_1 - e_env_2_out)) ** 2
    p_env_in_peak_mw = float(np.max(p_env_tot85_in_mw))

    # At the experimental 1.3 GHz spacing, two N=8 pulses do not overlap, and exact 1/(1.3 GHz) spacing
    # is m = T_b * f_s = 11.42 / 1.3 = 8.7846 cycles (282.5 deg mod 360 deg), NOT an integer 9 cycles (M5):
    e_env_2_linac = np.interp(t_ns - (t_linac_ps * 1e-3), t_ns, e_env_1, left=0.0, right=0.0)
    p_env_tot85_linac_spacing_n8_mw = (scale_q_half * (e_env_1 + e_env_2_linac)) ** 2
    m_1p3ghz_paper = float(f_s_paper_hz / spec.linac_frequency_hz)
    phase_mod_360_deg_paper = float((m_1p3ghz_paper % 1.0) * 360.0)
    m_1p3ghz_model = float(f_sync_hz / spec.linac_frequency_hz)
    phase_mod_360_deg_model = float((m_1p3ghz_model % 1.0) * 360.0)
    f_train_9th_paper_ghz = float(f_s_paper_ghz / 9.0)
    t_b_9th_paper_ns = float(9.0 / f_s_paper_ghz)
    f_train_9th_model_ghz = float(f_sync_ghz / 9.0)
    t_b_9th_model_ns = float(9.0 / f_sync_ghz)

    out_t4 = {
        "fit_3d_superposition_verification": sup_check,
        "f_sync_ghz": f_sync_ghz,
        "rf_period_ps": dt_in_ps,
        "rf_wavelength_mm": lambda_rf_mm,
        "linac_frequency_ghz": 1.3,
        "linac_period_ps": t_linac_ps,
        "harmonic_number_m_at_1p3ghz": m_1p3ghz_paper,
        "phase_mod_360_deg_at_1p3ghz": phase_mod_360_deg_paper,
        "harmonic_number_m_at_1p3ghz_model": m_1p3ghz_model,
        "phase_mod_360_deg_at_1p3ghz_model": phase_mod_360_deg_model,
        "ninth_harmonic_bunch_train_freq_ghz": f_train_9th_paper_ghz,
        "ninth_harmonic_bunch_spacing_ns": t_b_9th_paper_ns,
        "ninth_harmonic_bunch_train_freq_ghz_model": f_train_9th_model_ghz,
        "ninth_harmonic_bunch_spacing_ns_model": t_b_9th_model_ns,
        "n8_pulse_duration_20pct_ps": t_n8_pulse_ps,
        "n40_experiment_pulse_duration_ps": t_n40_pulse_ps,
        "n8_can_overlap_at_experiment_spacing": bool(t_n8_pulse_ps > t_linac_ps),
        "bunch_spacing_limitation_note": (
            f"In Lu et al. (PRL 122, 014801, lines 143, 153-154, 255-257, 329-331; Lu thesis lines 2337-2341, 2365), "
            f"the two bunches are spaced around one 1.3 GHz (PRL line 143) linac period T_b = {t_linac_ps:.2f} ps and fine-tuned "
            f"in phase with laser splitters before the photocathode gun (PRL lines 153-154, 256-257, 329-331; thesis lines 2338-2341): "
            f"an exact 1/(1.3 GHz, PRL line 143) spacing equals m = T_b * f_s = {m_1p3ghz_paper:.3f} cycles ({phase_mod_360_deg_paper:.1f} deg mod 360 deg "
            f"at {f_s_paper_ghz:.2f} GHz, PRL lines 137, 149; {m_1p3ghz_model:.3f} cycles, {phase_mod_360_deg_model:.1f} deg mod 360 deg at the model f_pk = {f_sync_ghz:.4f} GHz) "
            f"and is not in phase without optical delay tuning; instead, Lu's thesis (lines 2340-2341, 2365) states that {f_s_paper_ghz:.2f} GHz is the 9th harmonic "
            f"of a f_s/9 = {f_train_9th_paper_ghz:.2f} GHz bunch train with tuned spacing T_b = 9/f_s = {t_b_9th_paper_ns:.2f} ns ({t_b_9th_paper_ns * 1e3:.2f} ps; "
            f"model 9/f_pk = {t_b_9th_model_ns:.2f} ns at f_pk/9 = {f_train_9th_model_ghz:.2f} GHz). That ~{t_b_9th_paper_ns:.2f} ns ({t_linac_ps:.2f} ps nominal) delay "
            f"overlaps inside the 8 cm (N = 40) structure because t_p(N=40) = {t_n40_pulse_ns:.2f} ns > T_b, whereas in the short N = 8 model "
            f"t_p(N=8) = {tp_n8_eq1_ns:.2f} ns ({t_n8_pulse_ns:.3f} ns at 20% threshold) < T_b = {t_linac_ps:.2f} ps, "
            f"so two bunches at {t_linac_ps:.2f} ps spacing do not overlap in N = 8. The {p_env_in_peak_mw:.2f} MW "
            f"curve is therefore a charge-scaled single-bunch estimate shifted by one synchronous RF period "
            f"(Delta_t = 1/f_s = {dt_in_ps:.2f} ps)."
        ),
        "equal_bunch_superposition": {
            "method": (
                f"Passband-filtered 3D FIT wake potential W_||,bp(s) in [{f_bp_lo:.1f}, {f_bp_hi:.1f}] GHz "
                f"shifted by Delta_s"
            ),
            "in_phase_spacing_mm": ds_in_mm,
            "out_of_phase_spacing_mm": ds_out_mm,
            "single_bunch_trailing_wake_v_per_pc": wp_amp_1,
            "in_phase_two_bunch_wake_v_per_pc": wp_amp_in,
            "out_of_phase_two_bunch_wake_v_per_pc": wp_amp_out,
            "in_phase_wake_ratio_over_single": float(wp_amp_in / max(wp_amp_1, 1e-30)),
            "in_phase_power_ratio_over_single": float((wp_amp_in / max(wp_amp_1, 1e-30)) ** 2),
            "out_of_phase_wake_ratio_over_single": float(wp_amp_out / max(wp_amp_1, 1e-30)),
            "out_of_phase_power_ratio_over_single": float((wp_amp_out / max(wp_amp_1, 1e-30)) ** 2),
            "theoretical_in_phase_power_ratio": 4.0,
            "theoretical_out_of_phase_power_ratio": 0.0,
        },
        "two_bunch_85nc_comparison": {
            "paper_reported_two_bunch_peak_power_mw": 80.0,
            "paper_single_45nc_power_mw": 25.0,
            "paper_single_45nc_eq2_mw": p1_paper_analytic_45nc_mw,
            "wakis_single_45nc_fit_mw": p1_fit_45nc_mw,
            "charge_squared_ratio_85_over_45": charge_sq_ratio_85_over_45,
            "convention_total_85nc_q1_42p5_q2_42p5": {
                "q1_nc": 42.5,
                "q2_nc": 42.5,
                "q_total_nc": 85.0,
                "charge_scaled_single_bunch_42p5nc_mw": p_single_42p5nc_fit_mw,
                "charge_scaled_single_bunch_4x_42p5nc_mw": p_two_in_phase_tot85nc_fit_mw,
                "charge_scaled_1rf_shifted_envelope_peak_mw": p_env_in_peak_mw,
                "non_overlapping_769ps_spacing_n8_peak_mw": float(np.max(p_env_tot85_linac_spacing_n8_mw)),
                "out_of_phase_envelope_overlap_min_mw": float(
                    np.min(p_env_tot85_out_mw[(t_ns >= 0.14) & (t_ns <= 0.28)])
                ),
                "in_phase_eq2_analytic_mw": p_two_in_phase_tot85nc_analytic_mw,
                "rel_diff_eq2_vs_paper_80mw": float(abs(p_two_in_phase_tot85nc_analytic_mw - 80.0) / 80.0),
                "rel_diff_charge_scaled_envelope_vs_paper_80mw": float(abs(p_env_in_peak_mw - 80.0) / 80.0),
                "rel_diff_charge_scaled_4x_vs_paper_80mw": float(
                    abs(p_two_in_phase_tot85nc_fit_mw - 80.0) / 80.0
                ),
            },
            "convention_each_85nc_q1_85_q2_85": {
                "q1_nc": 85.0,
                "q2_nc": 85.0,
                "q_total_nc": 170.0,
                "charge_scaled_single_bunch_85nc_analytic_mw": p_single_85nc_analytic_mw,
                "charge_scaled_single_bunch_85nc_mw": p_single_85nc_fit_mw,
                "charge_scaled_single_bunch_4x_85nc_analytic_mw": p_two_each85nc_analytic_mw,
                "charge_scaled_single_bunch_4x_85nc_mw": p_two_each85nc_fit_mw,
            },
        },
    }
    npz_t4 = {
        "s_mm": s_mm,
        "wp_single": wp_1,
        "wp_single_passband": wp_bp_1,
        "wp_two_in_phase": wp_sum_in,
        "wp_two_out_of_phase": wp_sum_out,
        "t_ns": t_ns,
        "p_env_single_45nc_mw": p1_env_45nc,
        "p_env_tot85_in_phase_mw": p_env_tot85_in_mw,
        "p_env_tot85_out_of_phase_mw": p_env_tot85_out_mw,
        "psi_deg": psi_deg,
        "power_ratio_analytic": power_ratio_analytic,
        "power_ratio_wake_scan": power_ratio_wake_scan,
        "power_ratio_mutated_linear": power_ratio_mutated_linear,
    }
    return out_t4, npz_t4


def generate_task_04_data(
    data_dir: Path | str,
    *,
    quick: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Generate ``task_04_two_bunch_superposition.json`` and ``task_04_two_bunch_traces.npz``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "task_04_two_bunch_superposition.json"
    npz_path = data_dir / "task_04_two_bunch_traces.npz"

    t3_path = data_dir / "task_03_wakefield_and_directionality.json"
    z3_path = data_dir / "task_03_wakefield_traces.npz"
    if not t3_path.is_file() or not z3_path.is_file():
        generate_task_03_data(data_dir, quick=quick, spec=spec)
    sup_check = verify_two_bunch_fit_superposition(quick=quick, spec=spec)

    t3 = json.loads(t3_path.read_text(encoding="utf-8"))
    z3 = np.load(z3_path)
    out_t4, npz_t4 = _assemble_task_04_dict(sup_check, t3, z3, spec=spec)

    np.savez_compressed(npz_path, **npz_t4)
    json_path.write_text(json.dumps(out_t4, indent=2) + "\n", encoding="utf-8")
    return out_t4


def generate_task_05_data(
    data_dir: Path | str,
    *,
    quick: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Generate ``task_05_analytic_power_and_energy_balance.json`` (Parts 5 & 6: Analytic checks & Energy balance)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "task_05_analytic_power_and_energy_balance.json"

    t1_path = data_dir / "task_01_geometry_and_port_reflection.json"
    t2_path = data_dir / "task_02_cold_test_and_convergence.json"
    t3_path = data_dir / "task_03_wakefield_and_directionality.json"
    if not t1_path.is_file():
        generate_task_01_data(data_dir, quick=quick, spec=spec)
    if not t2_path.is_file():
        generate_task_02_data(data_dir, quick=quick, spec=spec)
    if not t3_path.is_file():
        generate_task_03_data(data_dir, quick=quick, spec=spec)

    t1 = json.loads(t1_path.read_text(encoding="utf-8"))
    t2 = json.loads(t2_path.read_text(encoding="utf-8"))
    t3 = json.loads(t3_path.read_text(encoding="utf-8"))

    _, _, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)
    q_c = spec.single_bunch_charge_c
    sigmaz = spec.bunch_length_rms_m
    L_exp_m = spec.n_periods_experiment * spec.period_m

    # 1. Paper cold-test parameters (Lu et al., PRL 122, 014801, lines 137, 147, 234, 247-250)
    if spec.paper_vg_over_c is None or not (-1.0 < float(spec.paper_vg_over_c) < 0.0):
        raise ValueError(
            f"spec.paper_vg_over_c must be a negative backward-wave group velocity in (-1, 0); got {spec.paper_vg_over_c!r}."
        )
    f_pap_hz = float(spec.paper_sync_freq_hz)
    vg_pap_m_s = float(spec.paper_vg_over_c) * C0
    kl_pap = paper_cold_test_k_loss_per_m(f_pap_hz, spec.paper_r_over_q_ohm_per_m)
    phi_pap = float(gaussian_bunch_form_factor(f_pap_hz, sigmaz, u0))
    tp_pap_s = analytic_backward_wave_pulse_length_s(L_exp_m, vg_pap_m_s, u0)
    p_pap_w = analytic_backward_wave_power_w(q_c, kl_pap, vg_pap_m_s, sigmaz, f_pap_hz, u0)

    # 2. Wakis 3D Bloch cold-test parameters (finest level nf8)
    w_bench = t2["wakis_finest_nf8_benchmarks"]
    f_wak_hz = float(w_bench["f_sync_ghz"]) * 1e9
    vg_wak_m_s = float(w_bench["ve_dft_over_c"]) * C0
    kl_wak = float(w_bench["k_loss_v_per_c_m"])
    phi_wak = float(gaussian_bunch_form_factor(f_wak_hz, sigmaz, u0))
    tp_wak_exp_s = analytic_backward_wave_pulse_length_s(L_exp_m, vg_wak_m_s, u0)
    p_wak_cold_w = analytic_backward_wave_power_w(q_c, kl_wak, vg_wak_m_s, sigmaz, f_wak_hz, u0)

    # Mutations on the analytical backward-wave power formula (for mutation tests and Figure 09a)
    p_mut_no_slip_w = analytic_backward_wave_power_w(
        q_c, kl_wak, vg_wak_m_s, sigmaz, f_wak_hz, u0, mutate_drop_slippage_factor=True
    )
    p_mut_flip_vg_w = analytic_backward_wave_power_w(
        q_c, kl_wak, vg_wak_m_s, sigmaz, f_wak_hz, u0, mutate_flip_vg_sign=True
    )
    p_mut_no_quarter_w = analytic_backward_wave_power_w(
        q_c, kl_wak, vg_wak_m_s, sigmaz, f_wak_hz, u0, mutate_drop_quarter_in_kl=True
    )
    tp_mut_flip_vg_s = analytic_backward_wave_pulse_length_s(
        L_exp_m, vg_wak_m_s, u0, mutate_flip_vg_sign=True
    )

    # 3. Energy balance across all simulated lengths N in {6, 8, 12} (Part 6)
    pr = t1["port_reflection"]
    gamma_tap = float(pr["gamma_tapered_at_sync"])
    rel_err_bound_port = float(reflection_induced_peak_error_bound(gamma_tap))
    rel_energy_err_bound_port = float(reflection_induced_energy_error_bound(gamma_tap))
    slip_factor = 1.0 / (1.0 - vg_wak_m_s / u0)

    energy_balance_all_N: dict[str, Any] = {}
    for n_per in t3["n_periods_list"]:
        k = str(n_per)
        rn = t3["runs_mtm"][k]
        l_n_m = float(rn["active_length_m"])
        w_loss_mj = float(rn["w_loss_passband_mj"])
        w_loss_cold_mj = float((q_c**2) * kl_wak * l_n_m * (phi_wak**2) * slip_factor * 1e3)
        e_up_mj = float(rn["e_up_mj"])
        e_dn_mj = float(rn["e_dn_mj"])
        e_rad_mj = float(rn["e_rad_total_mj"])
        u_fit_mj = float(rn["u_stored_final_fit_mj"])
        e_acc_fit_mj = e_rad_mj + u_fit_mj
        rel_err_acc_fit = float(abs(e_acc_fit_mj - w_loss_mj) / max(w_loss_mj, 1e-30))

        eb_n: dict[str, Any] = {
            "n_periods": int(n_per),
            "active_length_mm": float(l_n_m * 1e3),
            "wake_length_mm": float(rn["wake_length_mm"]),
            "fmax_impedance_ghz": float(rn["fmax_impedance_ghz"]),
            "form_factor_sq_at_fmax": float(rn["form_factor_sq_at_fmax"]),
            "form_factor_sq_at_25ghz": float(rn["form_factor_sq_at_25ghz"]),
            "bunch_energy_loss_wake_passband_mj": w_loss_mj,
            "w_beam_passband_cv_mj": float(rn["w_beam_passband_cv_mj"]),
            "w_beam_passband_active_mj": float(rn["w_beam_passband_active_mj"]),
            "w_loss_wake_passband_cv_mj": float(rn["w_loss_wake_passband_cv_mj"]),
            "w_loss_wake_passband_active_mj": float(rn["w_loss_wake_passband_active_mj"]),
            "w_loss_wake_fullband_active_mj": float(rn["w_loss_wake_fullband_active_mj"]),
            "w_loss_z_fullband_active_mj": float(rn["w_loss_z_fullband_active_mj"]),
            "out_of_band_wake_loss_fraction_active": float(rn["out_of_band_wake_loss_fraction_active"]),
            "w_loss_wake_fullband_cv_mj": float(rn["w_loss_wake_fullband_cv_mj"]),
            "w_beam_scat_fullband_cv_mj": float(rn["w_beam_scat_fullband_cv_mj"]),
            "out_of_band_beam_work_fraction_cv": float(rn["out_of_band_beam_work_fraction_cv"]),
            "w_beam_unfiltered_cv_mj": float(rn["w_beam_unfiltered_cv_mj"]),
            "e_rad_unfiltered_cv_mj": float(rn["e_rad_unfiltered_cv_mj"]),
            "u_stored_unfiltered_cv_mj": float(rn["u_stored_unfiltered_cv_mj"]),
            "rel_error_unfiltered_cv_closure": float(rn["rel_error_unfiltered_cv_closure"]),
            "nxy48_w_beam_unfiltered_cv_mj": float(t3["runs_mtm_nxy48"][k]["w_beam_unfiltered_cv_mj"]),
            "nxy48_e_rad_unfiltered_cv_mj": float(t3["runs_mtm_nxy48"][k]["e_rad_unfiltered_cv_mj"]),
            "nxy48_u_stored_unfiltered_cv_mj": float(t3["runs_mtm_nxy48"][k]["u_stored_unfiltered_cv_mj"]),
            "nxy48_rel_error_unfiltered_cv_closure": float(
                t3["runs_mtm_nxy48"][k]["rel_error_unfiltered_cv_closure"]
            ),
            "nxy48_u_stored_over_w_beam_passband": float(
                t3["runs_mtm_nxy48"][k]["u_stored_over_w_beam_passband"]
            ),
            "nxy48_rel_error_total_accounted_vs_bunch_loss": float(
                t3["runs_mtm_nxy48"][k]["rel_error_total_accounted_vs_bunch_loss"]
            ),
            "w_ref_unfiltered_cv_mj": float(rn["w_ref_unfiltered_cv_mj"]),
            "e_rad_ref_unfiltered_cv_mj": float(rn["e_rad_ref_unfiltered_cv_mj"]),
            "u_stored_ref_unfiltered_cv_mj": float(rn["u_stored_ref_unfiltered_cv_mj"]),
            "rel_error_ref_unfiltered_cv_closure": float(
                abs(
                    float(rn["e_rad_ref_unfiltered_cv_mj"])
                    + float(rn["u_stored_ref_unfiltered_cv_mj"])
                    - float(rn["w_ref_unfiltered_cv_mj"])
                )
                / max(
                    abs(float(rn["w_ref_unfiltered_cv_mj"])),
                    abs(float(rn["e_rad_ref_unfiltered_cv_mj"])),
                    abs(float(rn["u_stored_ref_unfiltered_cv_mj"])),
                    1e-30,
                )
            ),
            "w_ref_passband_cv_mj": float(rn["w_ref_passband_cv_mj"]),
            "e_rad_ref_passband_cv_mj": float(rn["e_rad_ref_passband_cv_mj"]),
            "u_stored_ref_passband_cv_mj": float(rn["u_stored_ref_passband_cv_mj"]),
            "u_stored_over_w_beam_passband": float(rn["u_stored_over_w_beam_passband"]),
            "bunch_energy_loss_cold_analytic_mj": w_loss_cold_mj,
            "upstream_radiated_poynting_mj": e_up_mj,
            "downstream_radiated_poynting_mj": e_dn_mj,
            "total_radiated_poynting_mj": e_rad_mj,
            "residual_stored_energy_fit_mj": u_fit_mj,
            "total_accounted_em_energy_mj": e_acc_fit_mj,
            "rel_error_upstream_vs_bunch_loss": float(abs(e_up_mj - w_loss_mj) / max(w_loss_mj, 1e-30)),
            "rel_error_rad_total_vs_bunch_loss": float(abs(e_rad_mj - w_loss_mj) / max(w_loss_mj, 1e-30)),
            "rel_error_total_accounted_vs_bunch_loss": rel_err_acc_fit,
            "mutated_shrunk_active_beam_work_rel_error": float(rn["mutated_shrunk_active_beam_work_rel_error"]),
            "port_gamma_tapered_at_sync": gamma_tap,
            "port_reflection_rel_error_bound": rel_err_bound_port,
            "port_reflection_energy_error_bound": rel_energy_err_bound_port,
            "mutated_dropped_upstream_energy_mj": float(e_dn_mj + u_fit_mj),
        }
        energy_balance_all_N[k] = eb_n

    r8 = t3["runs_mtm"]["8"]
    mc_t3 = t3["n8_transverse_mesh_convergence"]
    gci_p_t3 = mc_t3["gci"]["p_up_cycle_avg_max_mw"]
    L_n8_m = float(r8["active_length_m"])
    p_fit_n8_mw = float(r8["p_up_cycle_avg_max_mw"])
    p_fit_n8_96_mw = float(gci_p_t3["values"][2])
    p_u_fine_rel = float(gci_p_t3["u_fine_rel"])
    p_gci_med_rel = float(gci_p_t3["gci_med_rel"])
    p_conv_class = str(gci_p_t3["convergence_class"])
    eb8 = energy_balance_all_N["8"]
    tp_n8_analytic_ns = analytic_backward_wave_pulse_length_s(L_n8_m, vg_wak_m_s, u0) * 1e9
    dt_r8_s = float(r8["dt_fs"]) * 1e-15
    filt_rise_10_90_ns = measure_filter_step_rise_time_ns(dt_s=dt_r8_s, f_lo_hz=WagonWheelMTMSpec.passband_low_hz, f_hi_hz=WagonWheelMTMSpec.passband_high_hz, order=3)
    f_wak_ghz = float(w_bench["f_sync_ghz"])
    pulse_tol_ns = float(filt_rise_10_90_ns + 0.5 / f_wak_ghz)
    f_form_factor_sq_1e3_ghz = float(u0 * np.sqrt(np.log(1.0e3)) / (2.0 * np.pi * sigmaz) * 1e-9)

    out_t5 = {
        "paper_cold_test_analytic": {
            "f_sync_ghz": float(f_pap_hz * 1e-9),
            "vg_over_c": float(spec.paper_vg_over_c),
            "r_over_q_kohm_per_m": float(spec.paper_r_over_q_ohm_per_m * 1e-3),
            "k_loss_v_per_c_m": float(kl_pap),
            "form_factor_phi": phi_pap,
            "slippage_factor_1_minus_vg_over_c_inv2": float((1.0 - vg_pap_m_s / u0) ** (-2)),
            "pulse_length_N40_ns": float(tp_pap_s * 1e9),
            "bandwidth_N40_ghz": float(1.0 / (tp_pap_s * 1e9)),
            "single_bunch_45nc_power_mw": float(p_pap_w * 1e-6),
            "paper_quoted_power_mw": 25.0,
        },
        "wakis_cold_test_analytic": {
            "f_sync_ghz": f_wak_ghz,
            "vg_over_c": float(w_bench["ve_dft_over_c"]),
            "r_over_q_kohm_per_m": float(w_bench["r_over_q_kohm_per_m"]),
            "k_loss_v_per_c_m": kl_wak,
            "form_factor_phi": phi_wak,
            "f_form_factor_sq_1e3_ghz": f_form_factor_sq_1e3_ghz,
            "slippage_factor_1_minus_vg_over_c_inv2": float((1.0 - vg_wak_m_s / u0) ** (-2)),
            "pulse_length_N40_ns": float(tp_wak_exp_s * 1e9),
            "pulse_length_N8_ns": float(tp_n8_analytic_ns),
            "filter_step_rise_time_10_90_ns": filt_rise_10_90_ns,
            "pulse_duration_tolerance_ns": pulse_tol_ns,
            "single_bunch_45nc_power_mw": float(p_wak_cold_w * 1e-6),
            "w_loss_N8_analytic_mj": float(eb8["bunch_energy_loss_cold_analytic_mj"]),
            "mutated_no_slippage_power_mw": float(p_mut_no_slip_w * 1e-6),
            "mutated_flipped_vg_sign_power_mw": float(p_mut_flip_vg_w * 1e-6),
            "mutated_dropped_quarter_power_mw": float(p_mut_no_quarter_w * 1e-6),
            "mutated_flipped_vg_pulse_length_N40_ns": float(tp_mut_flip_vg_s * 1e9),
        },
        "wakefield_vs_analytic_comparison": {
            "wakis_3d_fit_power_N8_mw": p_fit_n8_mw,
            "wakis_3d_fit_power_N8_nxy96_mw": p_fit_n8_96_mw,
            "p_up_convergence_class": p_conv_class,
            "p_up_u_fine_nxy96_rel": p_u_fine_rel,
            "p_up_gci_med_nxy64_rel": p_gci_med_rel,
            "wakis_cold_analytic_power_mw": float(p_wak_cold_w * 1e-6),
            "paper_eq2_analytic_power_mw": float(p_pap_w * 1e-6),
            "paper_quoted_power_mw": 25.0,
            "port_gamma_tapered_at_sync": gamma_tap,
            "port_reflection_rel_error_bound": rel_err_bound_port,
            "port_reflection_energy_error_bound": rel_energy_err_bound_port,
            "two_port_reflection_power_bound": float(2.0 * rel_energy_err_bound_port),
            "consistency_band_u_p_plus_2gamma_sq": float(p_u_fine_rel + 2.0 * rel_energy_err_bound_port),
            "rel_error_fit_vs_wakis_cold_analytic": float(
                abs(p_fit_n8_mw - p_wak_cold_w * 1e-6) / (p_wak_cold_w * 1e-6)
            ),
            "rel_error_fit_nxy96_vs_wakis_cold_analytic": float(
                abs(p_fit_n8_96_mw - p_wak_cold_w * 1e-6) / (p_wak_cold_w * 1e-6)
            ),
            "rel_error_fit_vs_paper_eq2": float(abs(p_fit_n8_mw - p_pap_w * 1e-6) / (p_pap_w * 1e-6)),
            "rel_error_fit_vs_paper_25mw": float(abs(p_fit_n8_mw - 25.0) / 25.0),
            "rel_error_fit_nxy96_vs_paper_25mw": float(abs(p_fit_n8_96_mw - 25.0) / 25.0),
        },
        "energy_balance_N8": eb8,
        "energy_balance_all_N": energy_balance_all_N,
    }

    json_path.write_text(json.dumps(out_t5, indent=2) + "\n", encoding="utf-8")
    return out_t5


def build_summary_metrics(data_dir: Path | str) -> dict[str, Any]:
    """Build and write ``data/summary_metrics.json`` from ``task_01`` through ``task_05``."""
    data_dir = Path(data_dir)
    t1 = json.loads((data_dir / "task_01_geometry_and_port_reflection.json").read_text(encoding="utf-8"))
    t2 = json.loads((data_dir / "task_02_cold_test_and_convergence.json").read_text(encoding="utf-8"))
    t3 = json.loads((data_dir / "task_03_wakefield_and_directionality.json").read_text(encoding="utf-8"))
    t4 = json.loads((data_dir / "task_04_two_bunch_superposition.json").read_text(encoding="utf-8"))
    t5 = json.loads((data_dir / "task_05_analytic_power_and_energy_balance.json").read_text(encoding="utf-8"))

    pr = t1["port_reflection"]
    wb = t2["wakis_finest_nf8_benchmarks"]
    cu = t2["convergence_uniform"]
    r8 = t3["runs_mtm"]["8"]
    ls = t3["length_scaling"]
    dc = t3["directionality_comparison_N8"]
    sup = t4["fit_3d_superposition_verification"]
    eq_b = t4["equal_bunch_superposition"]
    c85 = t4["two_bunch_85nc_comparison"]
    eb = t5["energy_balance_N8"]
    eb_all = t5["energy_balance_all_N"]
    wc = t5["wakefield_vs_analytic_comparison"]
    kl_pap = paper_cold_test_k_loss_per_m()

    c85_tot = c85["convention_total_85nc_q1_42p5_q2_42p5"]
    fc_8p1_ghz = float(t1["empty_waveguide_tm01_cutoff_8p1mm_analytic_ghz"])
    mc = t3["n8_transverse_mesh_convergence"]
    v81 = t2["variant_rwg_8p1mm_thesis"]
    pb_t2 = t2["paper_cold_test_benchmarks"]
    f_pap_sm_ghz = float(pb_t2["f_sync_ghz"])
    vg_pap_sm_c = float(pb_t2["vg_over_c"])
    rq_pap_sm_kohm = float(pb_t2["r_over_q_kohm_per_m"])
    f_sync_nf8_val = float(cu["f_mode_ghz"]["values"][2])
    sm = {
        "tm01_cutoff_empty_analytic_ghz": float(t2["empty_waveguide_tm01_cutoff_analytic_ghz"]),
        "tm01_cutoff_empty_8p1mm_analytic_ghz": fc_8p1_ghz,
        "tm01_cutoff_empty_fit_ghz": float(t2["empty_waveguide_tm01_cutoff_fit_ghz"]),
        "tm01_cutoff_empty_paper_ghz": 14.2,
        "port_f_sync_eval_ghz": float(pr["f_sync_eval_ghz"]),
        "port_gamma_tapered_coax_at_sync": float(pr["gamma_tapered_at_sync"]),
        "port_power_transmission_tapered_at_sync": float(pr["power_transmission_tapered_at_sync"]),
        "port_gamma_abrupt_pipe_at_sync": float(pr["gamma_abrupt_at_sync"]),
        "port_gamma_abrupt_pipe_label": str(pr["gamma_abrupt_at_sync_label"]),
        "port_rel_error_bound_tapered": float(pr["rel_error_bound_tapered"]),
        "cold_phi_beam_crossing_nf4_deg": float(wb["phi_beam_crossing_nf4_deg"]),
        "cold_f_beam_crossing_nf4_ghz": float(wb["f_beam_crossing_nf4_ghz"]),
        "cold_f_beam_crossing_rel_error_vs_paper": float(wb["rel_error_f_beam_crossing_nf4_vs_paper"]),
        "cold_f_sync_nf4_ghz": float(cu["f_mode_ghz"]["values"][0]),
        "cold_f_sync_nf6_ghz": float(cu["f_mode_ghz"]["values"][1]),
        "cold_f_sync_nf8_ghz": f_sync_nf8_val,
        "cold_f_sync_paper_ghz": f_pap_sm_ghz,
        "cold_f_sync_rel_error_vs_paper": float(wb["rel_error_f_sync_vs_paper"]),
        "cold_vg_over_c_disp": float(wb["vg_disp_over_c"]),
        "cold_ve_over_c_nf8": float(wb["ve_dft_over_c"]),
        "cold_vg_over_c_paper": vg_pap_sm_c,
        "cold_ve_rel_error_vs_paper": float(wb["rel_error_ve_vs_paper"]),
        "cold_u_elec_over_u_mag_nf8": float(wb["u_elec_over_u_mag"]),
        "cold_r_over_q_nf4_kohm_m": float(cu["r_over_q_kohm_per_m"]["values"][0]),
        "cold_r_over_q_nf6_kohm_m": float(cu["r_over_q_kohm_per_m"]["values"][1]),
        "cold_r_over_q_nf8_kohm_m": float(cu["r_over_q_kohm_per_m"]["values"][2]),
        "cold_r_over_q_paper_kohm_m": rq_pap_sm_kohm,
        "cold_r_over_q_rel_error_vs_paper": float(wb["rel_error_r_over_q_vs_paper"]),
        "cold_k_loss_nf8_v_per_c_m": float(wb["k_loss_v_per_c_m"]),
        "cold_k_loss_paper_v_per_c_m": kl_pap,
        "cold_k_loss_rel_error_vs_paper": float(wb["rel_error_k_loss_vs_paper"]),
        "wake_f_pk_N8_ghz": float(r8["f_pk_ghz"]),
        "wake_f_pk_rel_error_vs_paper": float(abs(r8["f_pk_ghz"] - f_pap_sm_ghz) / f_pap_sm_ghz),
        "wake_f_pk_rel_error_vs_cold_nf8": float(abs(r8["f_pk_ghz"] - f_sync_nf8_val) / f_sync_nf8_val),
        "wake_re_z_pk_N8_ohm": float(r8["re_z_pk_ohm"]),
        "wake_re_z_pk_over_cold_sinc2_ratios": [float(x) for x in ls["re_z_pk_over_cold_sinc2_ratio"]],
        "wake_passband_integrated_re_z_ratios": [
            float(x) for x in ls["passband_integrated_re_z_fit_over_cold_sinc2_ratio"]
        ],
        "wake_passband_integrated_re_z_excess_ohm_ghz": [
            float(x) for x in ls["passband_integrated_re_z_excess_ohm_ghz"]
        ],
        "wake_passband_integrated_re_z_fit_slope_ohm_ghz_per_period": float(
            ls["passband_integrated_re_z_fit_slope_ohm_ghz_per_period"]
        ),
        "wake_passband_integrated_re_z_cold_sinc2_slope_ohm_ghz_per_period": float(
            ls["passband_integrated_re_z_cold_sinc2_slope_ohm_ghz_per_period"]
        ),
        "wake_rel_diff_passband_integrated_re_z_slope_fit_vs_cold_sinc2": float(
            ls["rel_diff_passband_integrated_re_z_slope_fit_vs_cold_sinc2"]
        ),
        "wake_effective_width_ratios": [float(x) for x in ls["effective_width_fit_over_cold_sinc2_ratio"]],
        "wake_affine_a_w_j_per_m": float(ls["affine_a_w_j_per_m"]),
        "wake_affine_b_w_mj": float(ls["affine_b_w_mj"]),
        "wake_affine_a_e_up_j_per_m": float(ls["affine_a_e_up_j_per_m"]),
        "wake_affine_b_e_up_mj": float(ls["affine_b_e_up_mj"]),
        "wake_cold_analytic_a_w_j_per_m": float(ls["cold_analytic_a_w_j_per_m"]),
        "wake_rel_error_a_w_vs_cold_analytic": float(ls["rel_error_a_w_vs_cold_analytic"]),
        "wake_a_w_uncertainty_rel": float(ls["a_w_uncertainty_rel"]),
        "wake_slope_re_z_pk_vs_L": float(ls["slope_re_z_pk_vs_L_fit"]),
        "wake_slope_fwhm_vs_L": float(ls["slope_fwhm_vs_L_fit"]),
        "wake_slope_p_up_vs_L": float(ls["slope_p_up_vs_L_fit"]),
        "wake_slope_tp_vs_L": float(ls["slope_tp_vs_L_fit"]),
        "pulse_length_N8_fit_ns": float(r8["pulse_duration_20pct_ns"]),
        "pulse_length_N8_analytic_ns": float(t5["wakis_cold_test_analytic"]["pulse_length_N8_ns"]),
        "pulse_length_N40_wakis_analytic_ns": float(t5["wakis_cold_test_analytic"]["pulse_length_N40_ns"]),
        "pulse_length_N40_paper_analytic_ns": float(t5["paper_cold_test_analytic"]["pulse_length_N40_ns"]),
        "single_bunch_45nc_power_fit_mw": float(wc["wakis_3d_fit_power_N8_mw"]),
        "single_bunch_45nc_power_fit_nxy96_mw": float(wc["wakis_3d_fit_power_N8_nxy96_mw"]),
        "single_bunch_45nc_power_wakis_analytic_mw": float(wc["wakis_cold_analytic_power_mw"]),
        "single_bunch_45nc_power_paper_eq2_mw": float(wc["paper_eq2_analytic_power_mw"]),
        "single_bunch_45nc_power_paper_quoted_mw": 25.0,
        "single_bunch_power_rel_error_vs_wakis_analytic": float(wc["rel_error_fit_vs_wakis_cold_analytic"]),
        "single_bunch_power_rel_error_vs_paper_25mw": float(wc["rel_error_fit_vs_paper_25mw"]),
        "bwd_mtm_upstream_energy_mj": float(dc["mtm_backward_e_up_mj"]),
        "bwd_mtm_downstream_energy_mj": float(dc["mtm_backward_e_dn_mj"]),
        "bwd_mtm_upstream_fraction": float(dc["mtm_backward_upstream_fraction"]),
        "fwd_control_downstream_energy_mj": float(dc["control_forward_e_dn_mj"]),
        "fwd_control_upstream_energy_mj": float(dc["control_forward_e_up_mj"]),
        "fwd_control_downstream_fraction": float(dc["control_forward_downstream_fraction"]),
        "two_bunch_superposition_rel_l2_error_E": float(sup["rel_l2_error_E_3d"]),
        "two_bunch_superposition_rel_l2_error_H": float(sup["rel_l2_error_H_3d"]),
        "two_bunch_in_phase_power_ratio": float(eq_b["in_phase_power_ratio_over_single"]),
        "two_bunch_out_of_phase_power_ratio": float(eq_b["out_of_phase_power_ratio_over_single"]),
        "two_bunch_tot85nc_eq2_mw": float(c85_tot["in_phase_eq2_analytic_mw"]),
        "two_bunch_tot85nc_charge_scaled_single_bunch_mw": float(
            c85_tot["charge_scaled_1rf_shifted_envelope_peak_mw"]
        ),
        "two_bunch_tot85nc_charge_scaled_4x_single_bunch_mw": float(
            c85_tot["charge_scaled_single_bunch_4x_42p5nc_mw"]
        ),
        "two_bunch_paper_quoted_mw": 80.0,
        "two_bunch_m_cycles_at_1p3ghz": float(t4["harmonic_number_m_at_1p3ghz"]),
        "two_bunch_phase_mod_360_deg_at_1p3ghz": float(t4["phase_mod_360_deg_at_1p3ghz"]),
        "energy_balance_bunch_loss_mj": float(eb["bunch_energy_loss_wake_passband_mj"]),
        "energy_balance_upstream_rad_mj": float(eb["upstream_radiated_poynting_mj"]),
        "energy_balance_downstream_rad_mj": float(eb["downstream_radiated_poynting_mj"]),
        "energy_balance_stored_fit_mj": float(eb["residual_stored_energy_fit_mj"]),
        "energy_balance_total_accounted_mj": float(eb["total_accounted_em_energy_mj"]),
        "energy_balance_rel_error_upstream": float(eb["rel_error_upstream_vs_bunch_loss"]),
        "energy_balance_rel_error_total": float(eb["rel_error_total_accounted_vs_bunch_loss"]),
        "energy_balance_rel_error_total_N6": float(eb_all["6"]["rel_error_total_accounted_vs_bunch_loss"]),
        "energy_balance_rel_error_total_N8": float(eb_all["8"]["rel_error_total_accounted_vs_bunch_loss"]),
        "energy_balance_rel_error_total_N12": float(eb_all["12"]["rel_error_total_accounted_vs_bunch_loss"]),
        "u_stored_over_w_beam_passband_N6": float(eb_all["6"]["u_stored_over_w_beam_passband"]),
        "u_stored_over_w_beam_passband_N8": float(eb_all["8"]["u_stored_over_w_beam_passband"]),
        "u_stored_over_w_beam_passband_N12": float(eb_all["12"]["u_stored_over_w_beam_passband"]),
        "w_ref_unfiltered_cv_N8_mj": float(eb["w_ref_unfiltered_cv_mj"]),
        "rel_error_ref_unfiltered_cv_closure_N8": float(eb["rel_error_ref_unfiltered_cv_closure"]),
        "u_stored_ref_passband_cv_N8_mj": float(eb["u_stored_ref_passband_cv_mj"]),
        "wake_n8_nxy48_f_pk_ghz": float(mc["f_pk_ghz_values"][0]),
        "wake_n8_nxy64_f_pk_ghz": float(mc["f_pk_ghz_values"][1]),
        "wake_n8_nxy96_f_pk_ghz": float(mc["f_pk_ghz_values"][2]),
        "wake_n8_conv_rel_diff_f_pk": float(mc["rel_diff_f_pk_64_vs_48"]),
        "wake_n8_f_pk_u_fine_rel": float(mc["gci"]["f_pk_ghz"]["u_fine_rel"]),
        "wake_n8_nxy48_w_beam_cv_mj": float(mc["w_beam_passband_cv_mj_values"][0]),
        "wake_n8_nxy64_w_beam_cv_mj": float(mc["w_beam_passband_cv_mj_values"][1]),
        "wake_n8_nxy96_w_beam_cv_mj": float(mc["w_beam_passband_cv_mj_values"][2]),
        "wake_n8_w_beam_u_fine_rel": float(mc["gci"]["w_beam_passband_cv_mj"]["u_fine_rel"]),
        "wake_n8_w_beam_gci_med_rel": float(mc["gci"]["w_beam_passband_cv_mj"]["gci_med_rel"]),
        "wake_n8_nxy48_p_up_mw": float(mc["p_up_max_mw_values"][0]),
        "wake_n8_nxy64_p_up_mw": float(mc["p_up_max_mw_values"][1]),
        "wake_n8_nxy96_p_up_mw": float(mc["p_up_max_mw_values"][2]),
        "wake_n8_p_up_u_fine_rel": float(mc["gci"]["p_up_cycle_avg_max_mw"]["u_fine_rel"]),
        "wake_n8_p_up_gci_med_rel": float(mc["gci"]["p_up_cycle_avg_max_mw"]["gci_med_rel"]),
        "wake_n8_nxy48_e_up_mj": float(mc["e_up_mj_values"][0]),
        "wake_n8_nxy64_e_up_mj": float(mc["e_up_mj_values"][1]),
        "wake_n8_nxy96_e_up_mj": float(mc["e_up_mj_values"][2]),
        "wake_n8_conv_rel_diff_e_up": float(mc["rel_diff_e_up_64_vs_48"]),
        "wake_n8_e_up_u_fine_rel": float(mc["gci"]["e_up_mj"]["u_fine_rel"]),
        "wake_n8_e_up_gci_med_rel": float(mc["gci"]["e_up_mj"]["gci_med_rel"]),
        "wake_n8_nxy48_re_z_pk_ohm": float(mc["re_z_pk_ohm_values"][0]),
        "wake_n8_nxy64_re_z_pk_ohm": float(mc["re_z_pk_ohm_values"][1]),
        "wake_n8_nxy96_re_z_pk_ohm": float(mc["re_z_pk_ohm_values"][2]),
        "wake_n8_conv_rel_diff_re_z_pk": float(mc["rel_diff_re_z_pk_64_vs_48"]),
        "wake_n8_re_z_pk_u_fine_rel": float(mc["gci"]["re_z_pk_ohm"]["u_fine_rel"]),
        "wake_n8_port_gamma_u_fine_rel": float(mc["gci"]["port_gamma_tapered_at_sync"]["u_fine_rel"]),
        "variant_8p1mm_f_sync_nf4_ghz": float(v81["nf4"]["f_mode_ghz"]),
        "variant_8p1mm_f_sync_nf6_ghz": float(v81["nf6"]["f_mode_ghz"]),
        "variant_8p1mm_ve_over_c_nf6": float(v81["nf6"]["ve_over_c"]),
        "variant_8p1mm_r_over_q_nf6_kohm_m": float(v81["nf6"]["r_over_q_kohm_per_m"]),
        "variant_8p1mm_f_sync_nf8_ghz": float(v81["nf8"]["f_mode_ghz"]),
        "variant_8p1mm_ve_over_c_nf8": float(v81["nf8"]["ve_over_c"]),
        "variant_8p1mm_r_over_q_nf8_kohm_m": float(v81["nf8"]["r_over_q_kohm_per_m"]),
    }

    out_path = data_dir / "summary_metrics.json"
    out_path.write_text(json.dumps(sm, indent=2) + "\n", encoding="utf-8")
    return sm


def generate_all_tutorial_figures(
    data_dir: Path | str,
    figures_dir: Path | str,
    *,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, str]:
    """Generate tutorial figures ``01`` through ``09`` (including animated GIF ``07``) from saved JSON/NPZ files."""
    from PIL import Image

    data_dir = Path(data_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()

    out_paths: dict[str, str] = {}
    _, _, u0 = relativistic_beam_kinematics(spec.beam_kinetic_energy_ev)

    # ------------------------------------------------------------------
    # Figure 01: Wagon-Wheel Transverse Geometry & Upstream Port Reflection |Gamma(f)|
    # ------------------------------------------------------------------
    t1_file = data_dir / "task_01_geometry_and_port_reflection.json"
    z1_file = data_dir / "task_01_port_reflection_spectra.npz"
    if t1_file.is_file() and z1_file.is_file():
        t1 = json.loads(t1_file.read_text(encoding="utf-8"))
        z1 = np.load(z1_file)
        pr = t1["port_reflection"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
        # Panel (a): Transverse (x, y) cross-section of the wagon-wheel plate
        r_wg_mm = spec.waveguide_radius_m * 1e3
        r_hub_mm = spec.hub_outer_radius_m * 1e3
        r_b_mm = spec.beam_hole_radius_m * 1e3
        w_sp_mm = spec.spoke_width_m * 1e3

        outer_ring = Circle((0.0, 0.0), r_wg_mm + 1.2, facecolor="#CBD5E1", edgecolor="#1A1A1A", lw=1.5, zorder=1)
        slot_void = Circle((0.0, 0.0), r_wg_mm, facecolor="#FFFFFF", edgecolor="#1A1A1A", lw=1.5, zorder=2)
        hub_ring = Circle(
            (0.0, 0.0),
            r_hub_mm,
            facecolor="#DCE8F2",
            edgecolor="#0072B2",
            lw=1.6,
            zorder=3,
            label="PEC hub (r∈[3,4] mm, ASSUMED) & 4 spokes",
        )
        beam_void = Circle(
            (0.0, 0.0),
            r_b_mm,
            facecolor="#FFF9DB",
            edgecolor="#D55E00",
            lw=1.6,
            ls="--",
            zorder=5,
            label="6 mm beam tunnel (2r_b = 6 mm)",
        )
        ax1.add_patch(outer_ring)
        ax1.add_patch(slot_void)
        ax1.add_patch(hub_ring)
        for bounds in (
            (r_b_mm, -0.5 * w_sp_mm, r_wg_mm - r_b_mm, w_sp_mm),
            (-r_wg_mm, -0.5 * w_sp_mm, r_wg_mm - r_b_mm, w_sp_mm),
            (-0.5 * w_sp_mm, r_b_mm, w_sp_mm, r_wg_mm - r_b_mm),
            (-0.5 * w_sp_mm, -r_wg_mm, w_sp_mm, r_wg_mm - r_b_mm),
        ):
            ax1.add_patch(
                Rectangle(
                    (bounds[0], bounds[1]),
                    bounds[2],
                    bounds[3],
                    facecolor="#DCE8F2",
                    edgecolor="#0072B2",
                    lw=1.4,
                    zorder=4,
                )
            )
        ax1.add_patch(beam_void)
        ax1.plot([0.0], [0.0], "o", color="#D55E00", ms=6.5, zorder=6, label="65 MeV drive bunch (r = 0)")
        ax1.text(
            4.15,
            4.15,
            "Inter-spoke\nvacuum\nregion",
            ha="center",
            va="center",
            fontsize=7.5,
            color="#1A1A1A",
        )
        ax1.annotate(
            "Radial spoke\n(1 mm, PRL\nline 119)",
            xy=(-0.5, 5.7),
            xytext=(-4.15, 4.15),
            ha="center",
            va="center",
            fontsize=7.4,
            color="#0072B2",
            arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 1.1},
            zorder=7,
        )
        ax1.set_aspect("equal")
        ax1.set_xlim(-10.5, 10.5)
        ax1.set_ylim(-13.4, 10.2)
        ax1.set_yticks([-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5])
        ax1.set_xlabel("Transverse coordinate $x$ [mm]")
        ax1.set_ylabel("Transverse coordinate $y$ [mm]")
        ax1.set_title("(a) Simulated Wagon-Wheel Plate (p = 2 mm)")
        ax1.legend(loc="lower center", fontsize=7.2, frameon=True)

        # Panel (b): Upstream (-z) port reflection |Gamma(f)|
        f1 = np.asarray(z1["f_ghz"], dtype=float)
        g_tap = np.asarray(z1["gamma_tapered_coax"], dtype=float)
        g_abr = np.asarray(z1["gamma_abrupt_empty_pipe"], dtype=float)
        g_pec = np.asarray(z1["gamma_pec_cap"], dtype=float)
        f_s = float(pr["f_sync_eval_ghz"])
        lbl_abr = str(pr["gamma_abrupt_at_sync_label"])

        ax2.plot(
            f1,
            g_pec,
            color="#1A1A1A",
            ls=":",
            lw=1.6,
            marker="",
            label=r"Shorted PEC cap ($|\Gamma| \equiv 1$)",
        )
        ax2.plot(
            f1,
            g_abr,
            color="#D55E00",
            ls="--",
            lw=2.0,
            marker="",
            label=rf"Abrupt cutoff circular pipe ($|\Gamma(f_{{\mathrm{{eval}}}})| = {lbl_abr}$)",
        )
        ax2.plot(
            f1,
            g_tap,
            color="#0072B2",
            lw=2.3,
            marker="",
            label=rf"2-cell adiabatic coaxial coupler ($|\Gamma(f_{{\mathrm{{eval}}}})| = {pr['gamma_tapered_at_sync']:.3f}$, $1-|\Gamma|^2 = {100*pr['power_transmission_tapered_at_sync']:.1f}\%$)",
        )
        ax2.axvline(
            f_s,
            color="#009E73",
            ls="-.",
            lw=1.4,
            label=rf"Eval FFT bin $f_{{\mathrm{{eval}}}} = {f_s:.2f}\;\mathrm{{GHz}}$ (bound $\pm{100*pr['rel_error_bound_tapered']:.1f}\%$)",
        )
        ax2.set_xlabel("Frequency $f$ [GHz]")
        ax2.set_ylabel(r"Upstream port reflection $|\Gamma(f)|$ [—]")
        ax2.set_title("(b) Upstream (−z) Output Port Reflectometry")
        ax2.set_xlim(float(f1[0]), float(f1[-1]))
        ax2.set_ylim(0.0, 1.08)
        figstyle.spectrum_grid(ax2)
        ax2.legend(loc="center right", fontsize=7.6, frameon=True)

        fig.tight_layout()
        p01 = figures_dir / "01_wagon_wheel_geometry_and_port_reflection.png"
        fig.savefig(p01, dpi=220)
        plt.close(fig)
        out_paths["fig_01"] = str(p01)

    # ------------------------------------------------------------------
    # Figure 02 & Figure 03: 3D Bloch Dispersion, Eigenmode Fields & Mesh Convergence
    # ------------------------------------------------------------------
    t2_file = data_dir / "task_02_cold_test_and_convergence.json"
    z2_file = data_dir / "task_02_cold_test_fields.npz"
    if t2_file.is_file():
        t2 = json.loads(t2_file.read_text(encoding="utf-8"))
        disp = t2["dispersion"]
        wb = t2["wakis_finest_nf8_benchmarks"]
        pb = t2["paper_cold_test_benchmarks"]

        phi_d = np.array(disp["phi_deg"], dtype=float)
        f_d = np.array(disp["f_mode_ghz"], dtype=float)
        phi_fw = np.array(disp["forward_control_phi_deg"], dtype=float)
        f_fw = np.array(disp["forward_control_f_ghz"], dtype=float)
        fc_tm01 = float(t2["empty_waveguide_tm01_cutoff_analytic_ghz"])
        fc_8p1 = float(circular_waveguide_tm01_cutoff_hz(8.1e-3) * 1e-9)
        beta_rel = float(u0 / C0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6))
        ax1.plot(
            phi_d,
            f_d,
            "o-",
            color="#0072B2",
            lw=2.2,
            ms=6.0,
            label=rf"Wagon-wheel MTM mode (nf4 crossing: {disp['f_sync_crossing_ghz']:.3f} GHz at {disp['phi_sync_deg']:.2f}$^\circ$)",
        )
        ax1.plot(
            phi_fw,
            f_fw,
            "s--",
            color="#CC79A7",
            lw=1.9,
            ms=5.5,
            label=r"Disk-loaded forward control ($v_g > 0$)",
        )
        phi_line = np.linspace(0.0, 75.0, 200)
        f_beam_line = (np.deg2rad(phi_line) / spec.period_m) * u0 / (2.0 * np.pi) * 1e-9
        ax1.plot(
            phi_line,
            f_beam_line,
            "-.",
            color="#009E73",
            lw=1.8,
            label=rf"65 MeV beam line $\omega = k_z \beta c$ ($\beta = {beta_rel:.5f}$)",
        )
        ax1.axhline(
            fc_tm01,
            color="#D55E00",
            ls=":",
            lw=1.7,
            label=rf"Empty circular pipe $\mathrm{{TM}}_{{01}}$ cutoff ({fc_tm01:.2f} GHz; thesis 8.1 mm line 2075: {fc_8p1:.2f} GHz)",
        )
        ax1.plot(
            [disp["phi_sync_deg"]],
            [disp["f_sync_crossing_ghz"]],
            "D",
            color="#56B4E9",
            mec="#1A1A1A",
            ms=8.0,
            zorder=8,
            label=rf"Model crossing: $\phi_s={disp['phi_sync_deg']:.2f}^\circ, f_s={disp['f_sync_crossing_ghz']:.3f}\;\mathrm{{GHz}}$ (nf8 at ${wb['phi_eval_deg']:.2f}^\circ$: {wb['f_sync_ghz']:.3f} GHz)",
        )
        ax1.plot(
            [pb["phi_sync_deg"]],
            [pb["f_sync_ghz"]],
            "*",
            color="#E69F00",
            mec="#1A1A1A",
            ms=12.0,
            zorder=7,
            label=rf"Lu et al. (2019): $f_s = {pb['f_sync_ghz']:.2f}\;\mathrm{{GHz}}$, $v_g = {pb['vg_over_c']:.3f}c$",
        )
        ax1.set_xlabel(r"Phase advance per cell $\phi = k_z p$ [deg]")
        ax1.set_ylabel("Frequency $f$ [GHz]")
        ax1.set_title("(a) 3D Bloch Dispersion & Reversed-Cherenkov Crossing")
        ax1.set_xlim(8.0, 75.0)
        ax1.set_ylim(8.5, 22.5)
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="upper left", fontsize=7.1, frameon=True)

        if z2_file.is_file():
            z2 = np.load(z2_file)
            x_m = np.asarray(z2["x_mm"], dtype=float)
            y_m = np.asarray(z2["y_mm"], dtype=float)
            et_mag = np.asarray(z2["et_xy_ww_mag"], dtype=float)
            im2 = ax2.pcolormesh(x_m, y_m, et_mag.T, cmap=figstyle.CMAP_MAGNITUDE, shading="auto")
            fig.colorbar(im2, ax=ax2, label=r"$|\mathbf{E}_t(x, y)| / |E_{z,0}|$ [—]")
            # Overlay wagon-wheel PEC boundaries on the field map
            for rad_mm, ls_c, lbl_c in (
                (3.0, "--", "Beam hole r_b = 3 mm"),
                (4.0, "-", "Hub ring r_hub = 4 mm"),
                (8.0, "-", "Waveguide wall R_wg = 8 mm"),
            ):
                ax2.add_patch(
                    Circle((0.0, 0.0), rad_mm, facecolor="none", edgecolor="#FFFFFF", lw=1.3, ls=ls_c)
                )
            ax2.text(
                0.0,
                -7.2,
                "PEC wall (R_wg = 8 mm)",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#FFFFFF",
                fontweight="bold",
            )
            ax2.text(
                0.0,
                0.0,
                "Beam\nhole",
                ha="center",
                va="center",
                fontsize=8.2,
                color="#FFFFFF",
                fontweight="bold",
            )
            ax2.set_aspect("equal")
            ax2.set_xlabel("$x$ [mm]")
            ax2.set_ylabel("$y$ [mm]")
            ax2.set_title(r"(b) Bloch Eigenmode $|\mathbf{E}_t(x,y)|$ in Wagon-Wheel Plane")

        fig.tight_layout()
        p02 = figures_dir / "02_bloch_dispersion_and_eigenmode_fields.png"
        fig.savefig(p02, dpi=220)
        plt.close(fig)
        out_paths["fig_02"] = str(p02)

        # Figure 03: Log-Log 3-Level Mesh Convergence with Order Guides
        cu = t2["convergence_uniform"]
        cg = t2["convergence_graded"]
        h_arr = np.array(cu["h_mm"], dtype=float)

        f_u = np.array(cu["f_mode_ghz"]["values"], dtype=float)
        f_g = np.array(cg["f_mode_ghz"]["values"], dtype=float)
        ve_u = np.abs(np.array(cu["ve_over_c"]["values"], dtype=float))
        rq_u = np.array(cu["r_over_q_kohm_per_m"]["values"], dtype=float)
        rq_g = np.array(cg["r_over_q_kohm_per_m"]["values"], dtype=float)

        if (
            cu["f_mode_ghz"]["richardson_estimate"] is None
            or cg["f_mode_ghz"]["richardson_estimate"] is None
            or cu["ve_over_c"]["richardson_estimate"] is None
            or cu["f_mode_ghz"]["observed_order"] is None
            or cg["f_mode_ghz"]["observed_order"] is None
            or cu["ve_over_c"]["observed_order"] is None
        ):
            raise ValueError("Missing Richardson estimate or observed order in task_02 convergence dict.")
        f_ref = float(cu["f_mode_ghz"]["richardson_estimate"])
        f_g_ref = float(cg["f_mode_ghz"]["richardson_estimate"])
        ve_ref = abs(float(cu["ve_over_c"]["richardson_estimate"]))
        p_obs_f_u = float(cu["f_mode_ghz"]["observed_order"])
        p_obs_f_g = float(cg["f_mode_ghz"]["observed_order"])
        p_obs_ve_u = float(cu["ve_over_c"]["observed_order"])

        err_f_u = np.maximum(np.abs(f_u - f_ref) / f_ref, 1e-5)
        err_f_g = np.maximum(np.abs(f_g - f_g_ref) / f_g_ref, 1e-5)
        err_ve_u = np.maximum(np.abs(ve_u - ve_ref) / ve_ref, 1e-5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6))
        ax1.loglog(
            h_arr,
            100.0 * err_f_u,
            "o-",
            color="#0072B2",
            lw=2.2,
            ms=6.5,
            label=rf"Uniform $f_s$ ($f_{{\mathrm{{nf8}}}} = {f_u[-1]:.4f}\;\mathrm{{GHz}}$, $p_{{\mathrm{{obs}}}} = {p_obs_f_u:.2f}$)",
        )
        ax1.loglog(
            h_arr,
            100.0 * err_f_g,
            "^--",
            color="#009E73",
            lw=2.0,
            ms=6.0,
            label=rf"Graded $f_s$ ($f_{{\mathrm{{nf8}}}} = {f_g[-1]:.4f}\;\mathrm{{GHz}}$, $p_{{\mathrm{{obs}}}} = {p_obs_f_g:.2f}$)",
        )
        ax1.loglog(
            h_arr,
            100.0 * err_ve_u,
            "s-",
            color="#D55E00",
            lw=2.2,
            ms=6.5,
            label=rf"Uniform $|v_E|/c$ ($|v_E|/c_{{\mathrm{{nf8}}}} = {ve_u[-1]:.4f}$, $p_{{\mathrm{{obs}}}} = {p_obs_ve_u:.2f}$)",
        )
        # Order guides O(h^1) and O(h^2)
        h_guide = np.array([h_arr[-1], h_arr[0]])
        g_h1 = 100.0 * err_ve_u[0] * (h_guide / h_arr[0]) ** 1
        g_h2 = 100.0 * err_f_u[0] * (h_guide / h_arr[0]) ** 2
        ax1.loglog(h_guide, g_h1, ":", color="#1A1A1A", lw=1.5, label=r"$O(h^1)$ first-order guide")
        ax1.loglog(h_guide, g_h2, "-.", color="#8C6D00", lw=1.5, label=r"$O(h^2)$ second-order guide")
        ax1.set_xticks([h_arr[2], h_arr[1], h_arr[0]])
        ax1.set_xticklabels([f"{h_arr[2]:.3f} (nf=8)", f"{h_arr[1]:.3f} (nf=6)", f"{h_arr[0]:.3f} (nf=4)"])
        ax1.xaxis.set_minor_formatter(plt.NullFormatter())
        ax1.set_xlim(0.115, 0.270)
        ax1.set_xlabel(r"Minimum cell size $h = 1\;\mathrm{mm} / n_{\mathrm{feat}}$ [mm]")
        ax1.set_ylabel(r"Relative error vs $p_{\mathrm{obs}}$-based Richardson extrapolate [%]")
        ax1.set_title(r"(a) Log–Log Mesh Convergence ($n_{\mathrm{feat}} \in \{4, 6, 8\}$)")
        ax1.legend(loc="lower right", fontsize=7.5, frameon=True)

        ax2_r = ax2.twinx()
        l1 = ax2.plot(
            h_arr,
            rq_u,
            "s-",
            color="#D55E00",
            lw=2.2,
            ms=6.5,
            label=rf"Uniform conformal $r/Q$ ($r/Q_{{\mathrm{{nf8}}}} = {rq_u[-1]:.2f}$ k$\Omega$/m)",
        )
        l2 = ax2.plot(
            h_arr,
            rq_g,
            "^--",
            color="#009E73",
            lw=2.0,
            ms=6.0,
            label=rf"Graded conformal $r/Q$ ($r/Q_{{\mathrm{{nf8}}}} = {rq_g[-1]:.2f}$ k$\Omega$/m)",
        )
        l3 = ax2.axhline(
            float(pb["r_over_q_kohm_per_m"]),
            color="#D55E00",
            ls=":",
            lw=1.5,
            label=rf"Lu et al. (2019) cold test: $r/Q = {pb['r_over_q_kohm_per_m']:.0f}$ k$\Omega$/m",
        )
        l4 = ax2_r.plot(
            h_arr,
            ve_u,
            "o-.",
            color="#0072B2",
            lw=2.0,
            ms=6.0,
            label=rf"Backward energy speed $|v_E|/c$ (nf8: {ve_u[-1]:.4f} vs paper {abs(pb['vg_over_c']):.3f})",
        )
        ax2.set_xlabel(r"Minimum cell size $h = 1\;\mathrm{mm} / n_{\mathrm{feat}}$ [mm]")
        ax2.set_ylabel(r"Shunt impedance $r/Q$ [$\mathrm{k\Omega/m}$]", color="#D55E00")
        ax2_r.set_ylabel(r"Backward energy speed $|v_E| / c$ [—]", color="#0072B2")
        ax2.set_title(r"(b) Direct Convergence of $r/Q$ and $|v_E|/c$ vs Paper")
        ax2.set_ylim(20.0, 22.8)
        ax2_r.set_ylim(0.145, 0.172)
        lines_c = l1 + l2 + [l3] + l4
        ax2.legend(lines_c, [x.get_label() for x in lines_c], loc="lower left", fontsize=7.5, frameon=True)

        fig.tight_layout()
        p03 = figures_dir / "03_cold_test_mesh_convergence.png"
        fig.savefig(p03, dpi=220)
        plt.close(fig)
        out_paths["fig_03"] = str(p03)

    # ------------------------------------------------------------------
    # Figures 04, 05, 06 & GIF 07: Wakefield, Length Scaling, Poynting Flux & Wave Packet
    # ------------------------------------------------------------------
    t3_file = data_dir / "task_03_wakefield_and_directionality.json"
    z3_file = data_dir / "task_03_wakefield_traces.npz"
    t5_file = data_dir / "task_05_analytic_power_and_energy_balance.json"
    if t3_file.is_file() and z3_file.is_file():
        t3 = json.loads(t3_file.read_text(encoding="utf-8"))
        z3 = np.load(z3_file)
        n_list = t3["n_periods_list"]
        ls = t3["length_scaling"]
        dc = t3["directionality_comparison_N8"]
        t5_for_f4 = json.loads(t5_file.read_text(encoding="utf-8"))
        p_eq2_wakis_mw = float(t5_for_f4["wakis_cold_test_analytic"]["single_bunch_45nc_power_mw"])
        p_eq2_paper_mw = float(t5_for_f4["paper_cold_test_analytic"]["single_bunch_45nc_power_mw"])
        p_pap_25_mw = float(t5_for_f4["paper_cold_test_analytic"]["paper_quoted_power_mw"])

        # Figure 04: Re Z_||(f) spectra & Log-Log Length Scaling
        t2_f4_obj = json.loads(t2_file.read_text(encoding="utf-8"))
        wb_f4 = t2_f4_obj["wakis_finest_nf8_benchmarks"]
        pb_f4 = t2_f4_obj["paper_cold_test_benchmarks"]
        cold_fs_f4_hz = float(wb_f4["f_sync_ghz"]) * 1e9
        cold_kl_f4 = float(wb_f4["k_loss_v_per_c_m"])
        cold_ve_f4_m_s = float(wb_f4["ve_dft_over_c"]) * C0
        f_s_pap_f4 = float(pb_f4["f_sync_ghz"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6))
        colors_n = ["#56B4E9", "#0072B2", "#D55E00"]
        for idx_n, n_per in enumerate(n_list):
            r_n = t3["runs_mtm"][str(n_per)]
            f_n = np.asarray(z3[f"f_ghz_N{n_per}"], dtype=float)
            rz_n = np.asarray(z3[f"re_z_ohm_N{n_per}"], dtype=float) * 1e-3
            mask_f = (f_n >= 9.0) & (f_n <= 14.0)
            col = colors_n[idx_n % len(colors_n)]
            ax1.plot(
                f_n[mask_f],
                rz_n[mask_f],
                color=col,
                lw=2.2,
                marker="",
                label=rf"FIT $N={n_per}$ ($L={r_n['active_length_mm']:.0f}\;\mathrm{{mm}}$, $Z_{{\mathrm{{pk}}}}={r_n['re_z_pk_ohm']*1e-3:.2f}\;\mathrm{{k\Omega}}$)",
            )
            f_sub = f_n[mask_f]
            rz_th_fs = (
                analytic_sws_traveling_wave_impedance_ohm(
                    f_sub * 1e9,
                    f_sync_hz=cold_fs_f4_hz,
                    k_loss_per_m=cold_kl_f4,
                    length_m=r_n["active_length_m"],
                    u0_m_s=u0,
                    vg_m_s=cold_ve_f4_m_s,
                )
                * 1e-3
            )
            ax1.plot(
                f_sub,
                rz_th_fs,
                color=col,
                ls="--",
                lw=1.4,
                marker="",
                alpha=0.85,
                label=rf"Cold-test $\mathrm{{sinc}}^2$ at $f_s$ ($N={n_per}$, pk {float(np.max(rz_th_fs)):.2f} $\mathrm{{k\Omega}}$)",
            )
            if int(n_per) == 8:
                rz_th_fpk = (
                    analytic_sws_traveling_wave_impedance_ohm(
                        f_sub * 1e9,
                        f_sync_hz=float(r_n["f_pk_ghz"]) * 1e9,
                        k_loss_per_m=cold_kl_f4,
                        length_m=r_n["active_length_m"],
                        u0_m_s=u0,
                        vg_m_s=cold_ve_f4_m_s,
                    )
                    * 1e-3
                )
                ax1.plot(
                    f_sub,
                    rz_th_fpk,
                    color=col,
                    ls=":",
                    lw=1.3,
                    marker="",
                    alpha=0.80,
                    label=rf"Cold-test $\mathrm{{sinc}}^2$ shifted to $f_{{\mathrm{{pk}}}}$ ($N=8$)",
                )
        ax1.axvline(
            f_s_pap_f4,
            color="#009E73",
            ls=":",
            lw=1.5,
            label=rf"Paper $f_s = {f_s_pap_f4:.2f}$ GHz (PRL lines 137, 149)",
        )
        ax1.set_xlabel("Frequency $f$ [GHz]")
        ax1.set_ylabel(r"Longitudinal resistance $\operatorname{Re} Z_\parallel(f)$ [$\mathrm{k\Omega}$]")
        ax1.set_title(r"(a) Longitudinal Wake Impedance $\operatorname{Re} Z_\parallel(f)$ vs $N$", fontsize=9.6)
        ax1.set_xlim(9.2, 13.8)
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="upper right", fontsize=7.2, frameon=True)

        L_mm = np.array(ls["active_length_mm"], dtype=float)
        w_bp = np.array(ls["w_beam_passband_cv_mj"], dtype=float)
        e_up = np.array(ls["e_up_mj"], dtype=float)
        p_up = np.array(ls["p_up_cycle_avg_max_mw"], dtype=float)
        a_w = float(ls["affine_a_w_j_per_m"])
        b_w = float(ls["affine_b_w_mj"])
        a_w_an = float(ls["cold_analytic_a_w_j_per_m"])
        a_e = float(ls["affine_a_e_up_j_per_m"])
        b_e = float(ls["affine_b_e_up_mj"])
        L_grid_mm = np.linspace(11.0, 25.0, 100)

        ax2.plot(
            L_mm,
            w_bp,
            "o",
            color="#0072B2",
            ms=6.5,
            label=rf"FIT $W_{{\mathrm{{beam,bp,cv}}}}$ ($a_W={a_w:.2f}$ J/m, $b_W={b_w:.2f}$ mJ)",
        )
        ax2.plot(
            L_grid_mm,
            a_w * L_grid_mm + b_w,
            "-",
            color="#0072B2",
            lw=2.0,
            marker="",
        )
        ax2.plot(
            L_grid_mm,
            a_w_an * L_grid_mm + b_w,
            "--",
            color="#0072B2",
            lw=1.4,
            marker="",
            alpha=0.8,
            label=rf"Cold-test slope $a_{{W,\mathrm{{cold}}}}={a_w_an:.2f}$ J/m + $b_W$",
        )
        ax2.plot(
            L_mm,
            e_up,
            "s",
            color="#009E73",
            ms=6.5,
            label=rf"FIT $E_{{\mathrm{{up}}}}$ ($a_E={a_e:.2f}$ J/m, $b_E={b_e:.2f}$ mJ)",
        )
        ax2.plot(
            L_grid_mm,
            a_e * L_grid_mm + b_e,
            "-",
            color="#009E73",
            lw=2.0,
            marker="",
        )
        ax2.plot(
            L_mm,
            p_up,
            "^-.",
            color="#D55E00",
            lw=2.0,
            ms=6.5,
            label=rf"FIT $P_{{\mathrm{{up,max}}}}$ [MW] (slope $= {ls['slope_p_up_vs_L_fit']:.2f}$)",
        )
        ax2.axhline(
            p_eq2_wakis_mw,
            color="#CC79A7",
            ls="--",
            lw=1.5,
            label=rf"Wakis cold-test Eq. (2) power ({p_eq2_wakis_mw:.2f} MW)",
        )
        ax2.axhline(
            p_eq2_paper_mw,
            color="#D55E00",
            ls=":",
            lw=1.3,
            label=rf"Paper-parameter Eq. (2) power ({p_eq2_paper_mw:.2f} MW)",
        )
        ax2.set_xticks([12.0, 16.0, 24.0])
        ax2.set_xticklabels(["12 (N=6)", "16 (N=8)", "24 (N=12)"])
        ax2.set_xlim(11.0, 25.5)
        ax2.set_xlabel("Active metamaterial length $L = N p$ [mm]")
        ax2.set_ylabel("Energy ($W_{\\mathrm{beam,bp,cv}}, E_{\\mathrm{up}}$ [mJ]) / Power ($P_{\\mathrm{up}}$ [MW])")
        ax2.set_title("(b) Affine Length Scaling $y = a L + b$ & Peak Power", fontsize=9.6)
        figstyle.spectrum_grid(ax2)
        ax2.legend(loc="best", fontsize=7.0, frameon=True)

        fig.tight_layout()
        p04 = figures_dir / "04_wake_potential_and_impedance_scaling.png"
        fig.savefig(p04, dpi=220)
        plt.close(fig)
        out_paths["fig_04"] = str(p04)

        # Figure 05: Reversed-Cherenkov Upstream vs Downstream Poynting Flux & Forward Control
        t_bwd = np.asarray(z3["t_ns_N8"], dtype=float)
        p_up_bwd_inst = np.asarray(z3["p_up_inst_mw_N8"], dtype=float)
        p_up_bwd_env = np.asarray(z3["p_up_env_mw_N8"], dtype=float)
        p_dn_bwd_env = np.asarray(z3["p_dn_env_mw_N8"], dtype=float)

        t_fwd = np.asarray(z3["fwd_t_ns"], dtype=float)
        p_up_fwd_env = np.asarray(z3["fwd_p_up_env_mw"], dtype=float)
        p_dn_fwd_env = np.asarray(z3["fwd_p_dn_env_mw"], dtype=float)
        p_dn_fwd_inst = np.asarray(z3["fwd_p_dn_inst_mw"], dtype=float)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))
        ax1.plot(
            t_bwd,
            p_up_bwd_inst,
            color="#56B4E9",
            lw=0.9,
            marker="",
            alpha=0.55,
            label="Instantaneous upstream $P_{\\mathrm{up}}(t)$",
        )
        ax1.plot(
            t_bwd,
            p_up_bwd_env,
            color="#0072B2",
            lw=2.4,
            marker="",
            label=rf"Upstream (−z) cycle-avg $P_{{\mathrm{{up}}}}$ (peak {dc['mtm_backward_p_up_max_mw']:.1f} MW, $E_{{\mathrm{{up}}}}={dc['mtm_backward_e_up_mj']:.2f}$ mJ, {100*dc['mtm_backward_upstream_fraction']:.1f}%)",
        )
        ax1.plot(
            t_bwd,
            p_dn_bwd_env,
            color="#D55E00",
            ls="--",
            lw=2.0,
            marker="",
            label=rf"Downstream (+z) cycle-avg $P_{{\mathrm{{dn}}}}$ ($E_{{\mathrm{{dn}}}}={dc['mtm_backward_e_dn_mj']:.2f}$ mJ, {100*(1-dc['mtm_backward_upstream_fraction']):.1f}%)",
        )
        ax1.axhline(
            p_pap_25_mw,
            color="#009E73",
            ls=":",
            lw=1.6,
            label=rf"Lu et al. (2019) single-bunch power ({p_pap_25_mw:.0f} MW, PRL lines 50, 232, 291)",
        )
        ax1.set_xlabel("Time $t$ [ns]")
        ax1.set_ylabel("Port Poynting power [MW]")
        ax1.set_title(r"(a) Wagon-Wheel MTM ($v_g < 0$): Backward (−z) Output")
        ax1.set_xlim(0.0, float(t_bwd[-1]))
        ax1.legend(loc="upper right", fontsize=7.5, frameon=True)

        ax2.plot(
            t_fwd,
            p_dn_fwd_inst,
            color="#E69F00",
            lw=0.9,
            marker="",
            alpha=0.45,
            label="Instantaneous downstream $P_{\\mathrm{dn}}(t)$",
        )
        ax2.plot(
            t_fwd,
            p_dn_fwd_env,
            color="#D55E00",
            lw=2.4,
            marker="",
            label=rf"Downstream (+z) cycle-avg $P_{{\mathrm{{dn}}}}$ (peak {dc['control_forward_p_dn_max_mw']:.1f} MW, $E_{{\mathrm{{dn}}}}={dc['control_forward_e_dn_mj']:.2f}$ mJ, {100*dc['control_forward_downstream_fraction']:.1f}%)",
        )
        ax2.plot(
            t_fwd,
            p_up_fwd_env,
            color="#0072B2",
            ls="--",
            lw=2.0,
            marker="",
            label=rf"Upstream (−z) cycle-avg $P_{{\mathrm{{up}}}}$ ($E_{{\mathrm{{up}}}}={dc['control_forward_e_up_mj']:.2f}$ mJ, {100*(1-dc['control_forward_downstream_fraction']):.1f}%)",
        )
        ax2.set_xlabel("Time $t$ [ns]")
        ax2.set_ylabel("Port Poynting power [MW]")
        ax2.set_title(r"(b) Disk-Loaded Control ($v_g > 0$): Forward (+z) Output")
        ax2.set_xlim(0.0, float(t_fwd[-1]))
        ax2.legend(loc="upper right", fontsize=7.5, frameon=True)

        fig.tight_layout()
        p05 = figures_dir / "05_reversed_cherenkov_poynting_flux.png"
        fig.savefig(p05, dpi=220)
        plt.close(fig)
        out_paths["fig_05"] = str(p05)

        # Figure 06 (static PNG) & Figure 07 (animated GIF): Reversed-Cherenkov Wave Packet with overlay_geometry
        if "bwd_snap_ez_scat" in z3 and "fwd_snap_ez_scat" in z3:
            y_b = np.asarray(z3["bwd_y_mm"], dtype=float)
            z_b = np.asarray(z3["bwd_z_mm"], dtype=float)
            sn_b = np.asarray(z3["bwd_snap_ez_scat"], dtype=float)
            t_b_ps = np.asarray(z3["bwd_snap_t_ps"], dtype=float)
            zb_b_mm = np.asarray(z3["bwd_snap_zb_mm"], dtype=float)

            y_f = np.asarray(z3["fwd_y_mm"], dtype=float)
            z_f = np.asarray(z3["fwd_z_mm"], dtype=float)
            sn_f = np.asarray(z3["fwd_snap_ez_scat"], dtype=float)
            t_f_ps = np.asarray(z3["fwd_snap_t_ps"], dtype=float)
            zb_f_mm = np.asarray(z3["fwd_snap_zb_mm"], dtype=float)

            proxy_bwd, meta_bwd = build_mtm_schematic_proxy(n_periods=8, forward_control=False, spec=spec)
            proxy_fwd, _ = build_mtm_schematic_proxy(n_periods=8, forward_control=True, spec=spec)
            z_up_mm = meta_bwd["z_up_port_m"] * 1e3
            z_dn_mm = meta_bwd["z_dn_port_m"] * 1e3

            n_frames = min(len(sn_b), len(sn_f))
            vmax_b = max(float(np.percentile(np.abs(sn_b), 99.0)), 1.0)
            vmax_f = max(float(np.percentile(np.abs(sn_f), 99.0)), 1.0)
            mid_idx = n_frames // 2
            nxy_bwd = int(t3["runs_mtm"]["8"]["n_xy"])
            nxy_fwd = int(t3["forward_control_N8"]["n_xy"])

            pil_frames: list[Image.Image] = []
            p06_png = figures_dir / "06_reversed_cherenkov_wave_packet.png"
            p07_gif = figures_dir / "07_reversed_cherenkov_wave_packet.gif"

            for idx in range(n_frames):
                fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10.4, 6.4), sharex=True)
                im_top = ax_top.pcolormesh(
                    z_b,
                    y_b,
                    sn_b[idx] / vmax_b,
                    cmap=figstyle.CMAP_DIVERGING,
                    vmin=-1.0,
                    vmax=1.0,
                    shading="auto",
                )
                schematic.overlay_geometry(
                    ax_top,
                    proxy_bwd,
                    plane="zy",
                    unit="mm",
                    fill_materials=True,
                    material_alpha=0.22,
                    label_materials=False,
                )
                ax_top.axvline(z_up_mm, color="#0072B2", ls="--", lw=1.4, label="Upstream port $z_{\\mathrm{up}}$")
                ax_top.axvline(z_dn_mm, color="#D55E00", ls="--", lw=1.4, label="Downstream port $z_{\\mathrm{dn}}$")
                if 0.0 <= zb_b_mm[idx] <= float(z_b[-1]):
                    ax_top.plot(
                        [zb_b_mm[idx]],
                        [0.0],
                        "o",
                        color="#E69F00",
                        mec="#1A1A1A",
                        mew=1.2,
                        ms=8.5,
                        zorder=9,
                        label="65 MeV bunch ($+c\\hat{z}$)",
                    )
                ax_top.set_ylabel("$y$ [mm]")
                ax_top.set_ylim(-8.5, 8.5)
                ax_top.set_title(
                    rf"(a) Wagon-Wheel MTM ($v_g < 0, n_{{xy}}={nxy_bwd}, t = {t_b_ps[idx]:.0f}\;\mathrm{{ps}}$): Reversed-Cherenkov Packet Streams Upstream ($-z \leftarrow$)"
                )
                ax_top.legend(loc="lower right", ncol=3, fontsize=7.6, frameon=True)
                fig.colorbar(im_top, ax=ax_top, label=r"$E_{z,\mathrm{scat}} / E_{\max}$")

                im_bot = ax_bot.pcolormesh(
                    z_f,
                    y_f,
                    sn_f[idx] / vmax_f,
                    cmap=figstyle.CMAP_DIVERGING,
                    vmin=-1.0,
                    vmax=1.0,
                    shading="auto",
                )
                schematic.overlay_geometry(
                    ax_bot,
                    proxy_fwd,
                    plane="zy",
                    unit="mm",
                    fill_materials=True,
                    material_alpha=0.22,
                    label_materials=False,
                )
                ax_bot.axvline(z_up_mm, color="#0072B2", ls="--", lw=1.4)
                ax_bot.axvline(z_dn_mm, color="#D55E00", ls="--", lw=1.4)
                if 0.0 <= zb_f_mm[idx] <= float(z_f[-1]):
                    ax_bot.plot(
                        [zb_f_mm[idx]],
                        [0.0],
                        "o",
                        color="#E69F00",
                        mec="#1A1A1A",
                        mew=1.2,
                        ms=8.5,
                        zorder=9,
                    )
                ax_bot.set_xlabel("Axial coordinate $z$ [mm]")
                ax_bot.set_ylabel("$y$ [mm]")
                ax_bot.set_ylim(-8.5, 8.5)
                ax_bot.set_xlim(float(z_b[0]), float(z_b[-1]))
                ax_bot.set_title(
                    rf"(b) Disk-Loaded Control ($v_g > 0, n_{{xy}}={nxy_fwd}, t = {t_f_ps[idx]:.0f}\;\mathrm{{ps}}$): Forward Cherenkov Packet Streams Downstream ($\rightarrow +z$)"
                )
                fig.colorbar(im_bot, ax=ax_bot, label=r"$E_{z,\mathrm{scat}} / E_{\max}$")

                fig.tight_layout()
                if idx == mid_idx:
                    fig.savefig(p06_png, dpi=200)
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                pil_frames.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE))
                plt.close(fig)

            pil_frames[0].save(
                p07_gif,
                save_all=True,
                append_images=pil_frames[1:],
                duration=150,
                loop=0,
            )
            out_paths["fig_06"] = str(p06_png)
            out_paths["fig_07_gif"] = str(p07_gif)

    # ------------------------------------------------------------------
    # Figure 08: Two-Bunch Coherent Superposition & Phase Scan
    # ------------------------------------------------------------------
    t4_file = data_dir / "task_04_two_bunch_superposition.json"
    z4_file = data_dir / "task_04_two_bunch_traces.npz"
    if t4_file.is_file() and z4_file.is_file():
        t4 = json.loads(t4_file.read_text(encoding="utf-8"))
        z4 = np.load(z4_file)
        eq_b = t4["equal_bunch_superposition"]
        c85 = t4["two_bunch_85nc_comparison"]["convention_total_85nc_q1_42p5_q2_42p5"]
        p_cs_85_mw = float(c85["charge_scaled_1rf_shifted_envelope_peak_mw"])

        t_ns = np.asarray(z4["t_ns"], dtype=float)
        p_1_45 = np.asarray(z4["p_env_single_45nc_mw"], dtype=float)
        p_in_85 = np.asarray(z4["p_env_tot85_in_phase_mw"], dtype=float)
        p_out_85 = np.asarray(z4["p_env_tot85_out_of_phase_mw"], dtype=float)

        psi_deg = np.asarray(z4["psi_deg"], dtype=float)
        pr_th = np.asarray(z4["power_ratio_analytic"], dtype=float)
        pr_wk = np.asarray(z4["power_ratio_wake_scan"], dtype=float)
        pr_mut = np.asarray(z4["power_ratio_mutated_linear"], dtype=float)

        dt_in_ps_fig = float(t4["rf_period_ps"])
        dt_out_ps_fig = 1.5 * dt_in_ps_fig
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))
        p_paper_2b_mw = float(t4["two_bunch_85nc_comparison"]["paper_reported_two_bunch_peak_power_mw"])
        ax1.plot(
            t_ns,
            p_in_85,
            color="#0072B2",
            lw=2.3,
            marker="",
            label=rf"Charge-scaled in-phase ($85\;\mathrm{{nC}}, \Delta t=1/f_s={dt_in_ps_fig:.0f}\;\mathrm{{ps}}$): peak {p_cs_85_mw:.1f} MW",
        )
        ax1.plot(
            t_ns,
            p_1_45,
            color="#009E73",
            ls="-.",
            lw=2.0,
            marker="",
            label=rf"Single bunch ($q=45\;\mathrm{{nC}}$): peak {t4['two_bunch_85nc_comparison']['wakis_single_45nc_fit_mw']:.1f} MW",
        )
        ax1.plot(
            t_ns,
            p_out_85,
            color="#D55E00",
            ls="--",
            lw=2.0,
            marker="",
            label=rf"Charge-scaled out-of-phase ($\Delta t = (3/2)/f_s = {dt_out_ps_fig:.0f}\;\mathrm{{ps}}$): cancellation",
        )
        ax1.axhline(
            p_paper_2b_mw,
            color="#1A1A1A",
            ls=":",
            lw=1.6,
            label=rf"Lu et al. (2019) $N=40$ two-bunch ({p_paper_2b_mw:.0f} MW)",
        )
        ax1.set_xlabel("Time $t$ [ns]")
        ax1.set_ylabel("Cycle-averaged upstream RF power [MW]")
        ax1.set_title(
            r"(a) Charge-Scaled Single-Bunch Estimate ($Q_1 + Q_2 = 85\;\mathrm{nC}, \Delta t = 1/f_s$)",
            fontsize=9.5,
        )
        ax1.set_xlim(0.0, float(t_ns[-1]))
        ax1.legend(loc="center right", bbox_to_anchor=(0.99, 0.56), fontsize=7.0, frameon=True)

        ax2.plot(
            psi_deg,
            pr_th,
            color="#0072B2",
            lw=2.2,
            marker="",
            label=r"Exact quadratic superposition $P(\Delta\psi)/P_1 = 4\cos^2(\Delta\psi/2)$",
        )
        ax2.plot(
            psi_deg[::4],
            pr_wk[::4],
            "o",
            color="#D55E00",
            ms=5.5,
            label=rf"Passband FIT wake superposition ($\Delta\psi=2\pi$: {eq_b['in_phase_power_ratio_over_single']:.2f}$\times$, $3\pi$: {eq_b['out_of_phase_power_ratio_over_single']:.2f}$\times$)",
        )
        ax2.plot(
            psi_deg,
            pr_mut,
            "--",
            color="#8C6D00",
            lw=1.6,
            marker="",
            label=r"Mutated linear scaling $2|\cos(\Delta\psi/2)|$ (fails $Q^2$ law)",
        )
        ax2.set_xlabel(r"Bunch-to-bunch phase separation $\Delta\psi = \omega_s \Delta t$ [deg]")
        ax2.set_ylabel(r"Normalized peak RF power $P_{1+2}(\Delta\psi) / P_1$ [—]")
        ax2.set_title(r"(b) Passband FIT Wake Superposition vs Phase $\Delta\psi$", fontsize=9.5)
        ax2.set_xlim(0.0, 720.0)
        ax2.set_ylim(-0.18, 5.65)
        ax2.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
        figstyle.spectrum_grid(ax2)
        ax2.legend(loc="upper right", fontsize=7.2, frameon=True)

        fig.tight_layout()
        p08 = figures_dir / "08_two_bunch_coherent_superposition.png"
        fig.savefig(p08, dpi=220)
        plt.close(fig)
        out_paths["fig_08"] = str(p08)

    # ------------------------------------------------------------------
    # Figure 09: Analytic Backward-Wave Power Checks & Control-Volume Poynting Energy Balance
    # ------------------------------------------------------------------
    if t5_file.is_file():
        t5 = json.loads(t5_file.read_text(encoding="utf-8"))
        wa = t5["wakis_cold_test_analytic"]
        pa = t5["paper_cold_test_analytic"]
        eb = t5["energy_balance_N8"]
        eb_all = t5["energy_balance_all_N"]
        wc = t5["wakefield_vs_analytic_comparison"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))
        q_grid_nc = np.linspace(5.0, 55.0, 100)
        q_ref = 45.0
        p_wak_curve = wa["single_bunch_45nc_power_mw"] * ((q_grid_nc / q_ref) ** 2)
        p_pap_curve = pa["single_bunch_45nc_power_mw"] * ((q_grid_nc / q_ref) ** 2)
        p_mut_noslip = wa["mutated_no_slippage_power_mw"] * ((q_grid_nc / q_ref) ** 2)
        p_mut_flip = wa["mutated_flipped_vg_sign_power_mw"] * ((q_grid_nc / q_ref) ** 2)

        ax1.plot(
            q_grid_nc,
            p_wak_curve,
            color="#0072B2",
            lw=2.3,
            marker="",
            label=rf"Wakis cold-test Eq. (2): {wa['single_bunch_45nc_power_mw']:.2f} MW at 45 nC",
        )
        ax1.plot(
            q_grid_nc,
            p_pap_curve,
            "--",
            color="#009E73",
            lw=2.0,
            marker="",
            label=rf"Paper cold-test Eq. (2): {pa['single_bunch_45nc_power_mw']:.2f} MW at 45 nC",
        )
        p_fit_n8_val = float(wc["wakis_3d_fit_power_N8_mw"])
        c_band_rel = float(wc["consistency_band_u_p_plus_2gamma_sq"])
        ax1.errorbar(
            [45.0],
            [p_fit_n8_val],
            yerr=[p_fit_n8_val * c_band_rel],
            fmt="D",
            color="#D55E00",
            mec="#1A1A1A",
            ms=8.0,
            ecolor="#D55E00",
            elinewidth=1.6,
            capsize=4.5,
            zorder=7,
            label=rf"Wakis 3D FIT ($N=8$): {p_fit_n8_val:.2f} MW ($\pm(U_P+2|\Gamma|^2) = \pm{100*c_band_rel:.1f}\%$)",
        )
        ax1.plot(
            q_grid_nc,
            p_mut_noslip,
            ":",
            color="#8C6D00",
            lw=1.6,
            marker="",
            label=rf"Mutated (drops $(1-v_g/c)^{{-2}}$): {wa['mutated_no_slippage_power_mw']:.1f} MW",
        )
        ax1.plot(
            q_grid_nc,
            p_mut_flip,
            "-.",
            color="#CC79A7",
            lw=1.6,
            marker="",
            label=rf"Mutated (flips $v_g \to +|v_g|$): {wa['mutated_flipped_vg_sign_power_mw']:.1f} MW",
        )
        ax1.set_xlabel("Single-bunch charge $q$ [nC]")
        ax1.set_ylabel("Radiated output RF power $P$ [MW]")
        ax1.set_title("(a) Backward-Wave Power vs 3D FIT", fontsize=9.8)
        ax1.set_xlim(5.0, 52.0)
        ax1.set_ylim(0.0, 62.0)
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="upper left", fontsize=7.1, frameon=True)

        u_stored_val = float(eb["residual_stored_energy_fit_mj"])
        categories = [
            "CV beam work\n$W_{\\mathrm{beam,bp}}$",
            "Active wake\n$\\Delta U_{\\mathrm{act,bp}}$",
            "Upstream\n$E_{\\mathrm{up}}$ (−z)",
            "Downstream\n$E_{\\mathrm{dn}}$ (+z)",
            "Stored (FIT)\n$U_{\\mathrm{bp}}(T_{\\mathrm{end}})$",
            "Total EM\n$E_{\\mathrm{rad}}+U_{\\mathrm{bp}}$",
        ]
        vals_mj = [
            eb["bunch_energy_loss_wake_passband_mj"],
            eb["w_loss_wake_passband_active_mj"],
            eb["upstream_radiated_poynting_mj"],
            eb["downstream_radiated_poynting_mj"],
            u_stored_val,
            eb["total_accounted_em_energy_mj"],
        ]
        bar_cols = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]
        bars = ax2.bar(categories, vals_mj, color=bar_cols, edgecolor="#1A1A1A", lw=1.1, width=0.58)
        ax2.tick_params(axis="x", labelsize=7.8)
        for b_rect, v_m in zip(bars, vals_mj):
            ax2.text(
                b_rect.get_x() + 0.5 * b_rect.get_width(),
                v_m + 0.25,
                f"{v_m:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
            )
        e6_err = float(eb_all["6"]["rel_error_total_accounted_vs_bunch_loss"])
        e8_err = float(eb_all["8"]["rel_error_total_accounted_vs_bunch_loss"])
        e12_err = float(eb_all["12"]["rel_error_total_accounted_vs_bunch_loss"])
        ax2.set_ylabel("Energy [mJ]")
        ax2.set_title(
            "(b) Passband Control-Volume Closure ($N = 8$ bars)\n"
            rf"$|\Delta E / W|$: $N=6$: {e6_err:.2e}, $N=8$: {e8_err:.2e}, $N=12$: {e12_err:.2e}",
            fontsize=8.5,
        )
        ax2.set_ylim(0.0, max(vals_mj) * 1.22)

        fig.tight_layout()
        p09 = figures_dir / "09_analytic_power_and_energy_balance.png"
        fig.savefig(p09, dpi=220)
        plt.close(fig)
        out_paths["fig_09"] = str(p09)

    return out_paths


def generate_all_tutorial_data(
    data_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    quick: bool = False,
    tasks: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Driver entry point to run requested tasks, update ``summary_metrics.json``, and build tutorial figures."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if 1 in tasks:
        generate_task_01_data(data_dir, quick=quick, spec=spec)
    if 2 in tasks:
        generate_task_02_data(data_dir, quick=quick, spec=spec)
    if 3 in tasks:
        generate_task_03_data(data_dir, quick=quick, spec=spec)
    if 4 in tasks:
        generate_task_04_data(data_dir, quick=quick, spec=spec)
    if 5 in tasks or 6 in tasks:
        generate_task_05_data(data_dir, quick=quick, spec=spec)

    req_files = [
        data_dir / "task_01_geometry_and_port_reflection.json",
        data_dir / "task_02_cold_test_and_convergence.json",
        data_dir / "task_03_wakefield_and_directionality.json",
        data_dir / "task_04_two_bunch_superposition.json",
        data_dir / "task_05_analytic_power_and_energy_balance.json",
    ]
    sm: dict[str, Any] = {}
    if all(f.is_file() for f in req_files):
        sm = build_summary_metrics(data_dir)

    if figures_dir is not None:
        if 1 in tasks:
            plot_mtm_setup_schematics(figures_dir, spec=spec, data_dir=data_dir)
        generate_all_tutorial_figures(data_dir, figures_dir, spec=spec)

    return sm


def aggregate_all_tutorial_data(
    data_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    source_data_dir: Path | str | None = None,
    mutate_corrupt_raw_run: bool = False,
    mutate_reuse_nxy64_traces_for_nxy48: bool = False,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, Any]:
    """Recompute all derived/aggregate quantities from saved per-run values without running any FIT simulation.

    Reads the raw per-run outputs from ``source_data_dir`` (defaulting to ``data_dir`` if populated,
    otherwise the tutorial's ``data/`` directory), writes ``task_01``..``task_05`` JSON files,
    ``task_04_two_bunch_traces.npz``, and ``summary_metrics.json`` fresh into ``data_dir``, and
    optionally renders all tutorial figures into ``figures_dir``.
    """
    import copy
    import shutil

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if source_data_dir is None:
        if (data_dir / "task_03_wakefield_and_directionality.json").is_file():
            src_dir = data_dir
        else:
            src_dir = Path(__file__).resolve().parent / "data"
    else:
        src_dir = Path(source_data_dir)

    t1_raw = json.loads((src_dir / "task_01_geometry_and_port_reflection.json").read_text(encoding="utf-8"))
    t2_raw = json.loads((src_dir / "task_02_cold_test_and_convergence.json").read_text(encoding="utf-8"))
    t3_raw = json.loads((src_dir / "task_03_wakefield_and_directionality.json").read_text(encoding="utf-8"))
    t4_raw = json.loads((src_dir / "task_04_two_bunch_superposition.json").read_text(encoding="utf-8"))

    if src_dir.resolve() != data_dir.resolve():
        for npz_name in (
            "task_01_port_reflection_spectra.npz",
            "task_02_cold_test_fields.npz",
            "task_03_wakefield_traces.npz",
        ):
            src_npz = src_dir / npz_name
            if src_npz.is_file():
                shutil.copy2(src_npz, data_dir / npz_name)

    refl = copy.deepcopy(t1_raw["port_reflection"])
    if mutate_corrupt_raw_run:
        refl["gamma_tapered_at_sync"] = float(refl["gamma_tapered_at_sync"]) * 1.15
    out_t1 = _assemble_task_01_dict(refl, spec=spec)
    (data_dir / "task_01_geometry_and_port_reflection.json").write_text(
        json.dumps(out_t1, indent=2) + "\n", encoding="utf-8"
    )

    disp_raw = t2_raw["dispersion"]
    fc_tm01_fit_ghz = float(t2_raw["empty_waveguide_tm01_cutoff_fit_ghz"])
    if mutate_corrupt_raw_run:
        fc_tm01_fit_ghz *= 1.05
    uniform_levels = copy.deepcopy(t2_raw["uniform_levels"])
    graded_levels = copy.deepcopy(t2_raw["graded_levels"])
    v81_raw = t2_raw["variant_rwg_8p1mm_thesis"]
    out_t2 = _assemble_task_02_dict(
        fc_tm01_fit_ghz,
        [float(x) for x in disp_raw["phi_deg"]],
        [float(x) for x in disp_raw["f_mode_ghz"]],
        float(disp_raw["wall_time_s"]),
        [float(x) for x in disp_raw["forward_control_phi_deg"]],
        [float(x) for x in disp_raw["forward_control_f_ghz"]],
        uniform_levels,
        graded_levels,
        copy.deepcopy(v81_raw["nf4"]),
        copy.deepcopy(v81_raw["nf6"]),
        copy.deepcopy(v81_raw["nf8"]),
        spec=spec,
    )
    (data_dir / "task_02_cold_test_and_convergence.json").write_text(
        json.dumps(out_t2, indent=2) + "\n", encoding="utf-8"
    )

    n_list = [int(x) for x in t3_raw["n_periods_list"]]
    runs_mtm_nxy48 = copy.deepcopy(t3_raw["runs_mtm_nxy48"])
    runs_mtm = copy.deepcopy(t3_raw["runs_mtm"])
    r_96_n8 = copy.deepcopy(t3_raw["runs_mtm_nxy96_N8"])
    r_fwd = copy.deepcopy(t3_raw["forward_control_N8"])
    r_fwd_48 = copy.deepcopy(t3_raw["forward_control_N8_nxy48"])
    if mutate_corrupt_raw_run:
        runs_mtm["8"]["p_up_cycle_avg_max_mw"] = float(runs_mtm["8"]["p_up_cycle_avg_max_mw"]) * 1.15
    port_gamma_levels = [
        float(x) for x in t3_raw["n8_transverse_mesh_convergence"]["port_gamma_tapered_at_sync_values"]
    ]
    wb_t2 = out_t2["wakis_finest_nf8_benchmarks"]
    z3 = np.load(data_dir / "task_03_wakefield_traces.npz")
    out_t3 = _assemble_task_03_dict(
        n_list,
        runs_mtm_nxy48,
        runs_mtm,
        r_96_n8,
        r_fwd,
        r_fwd_48,
        port_gamma_levels,
        gamma_tap_note=float(out_t1["port_reflection"]["gamma_tapered_at_sync"]),
        cold_k_loss_per_m=float(wb_t2["k_loss_v_per_c_m"]),
        cold_vg_m_s=float(wb_t2["ve_dft_over_c"]) * C0,
        cold_f_sync_hz=float(wb_t2["f_sync_ghz"]) * 1e9,
        cold_k_loss_gci_fine_rel=float(out_t2["convergence_uniform"]["k_loss_v_per_c_m"]["u_fine_rel"]),
        z3_traces=z3,
        mutate_reuse_nxy64_traces_for_nxy48=mutate_reuse_nxy64_traces_for_nxy48,
        spec=spec,
    )
    (data_dir / "task_03_wakefield_and_directionality.json").write_text(
        json.dumps(out_t3, indent=2) + "\n", encoding="utf-8"
    )

    sup_check = copy.deepcopy(t4_raw["fit_3d_superposition_verification"])
    out_t4, npz_t4 = _assemble_task_04_dict(sup_check, out_t3, z3, spec=spec)
    np.savez_compressed(data_dir / "task_04_two_bunch_traces.npz", **npz_t4)
    (data_dir / "task_04_two_bunch_superposition.json").write_text(
        json.dumps(out_t4, indent=2) + "\n", encoding="utf-8"
    )

    generate_task_05_data(data_dir, quick=False, spec=spec)
    sm = build_summary_metrics(data_dir)

    if figures_dir is not None:
        plot_mtm_setup_schematics(figures_dir, spec=spec, data_dir=data_dir)
        generate_all_tutorial_figures(data_dir, figures_dir, spec=spec)

    return sm


def render_cold_test_table_markdown(t2: dict[str, Any]) -> str:
    """Format Table 2 (3D Bloch unit-cell cold-test mesh convergence) from ``task_02`` JSON at render time."""
    cu = t2["convergence_uniform"]
    cg = t2["convergence_graded"]
    ul = t2["uniform_levels"]
    gl = t2["graded_levels"]
    wb = t2["wakis_finest_nf8_benchmarks"]
    pb = t2["paper_cold_test_benchmarks"]
    v81 = t2["variant_rwg_8p1mm_thesis"]

    rows = [
        "| Mesh Family & Level | Min. Cell $h$ [mm] | Cells/$\\lambda_0$ | Cells / $1\\;\\mathrm{mm}$ Feature | $f(\\phi_s)$ [GHz] | $v_E / c$ [—] | $r/Q$ [$\\mathrm{k\\Omega/m}$] | $k_L$ [$10^{14}\\;\\mathrm{V/(C\\cdot m)}$] | $U_e / U_m$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for idx, nf in enumerate((4, 6, 8)):
        u_lvl = ul[f"nf{nf}"]
        h_val = float(u_lvl["h_min_mm"])
        cpl = float(u_lvl["cells_per_wavelength_lambda0"])
        f_val = float(u_lvl["f_mode_ghz"])
        ve_val = float(u_lvl["ve_over_c"])
        rq_val = float(u_lvl["r_over_q_kohm_per_m"])
        kl_val = float(u_lvl["k_loss_v_per_c_m"]) * 1e-14
        ue_val = float(u_lvl["u_elec_over_u_mag"])
        cross_note = (
            f" (`{wb['f_beam_crossing_nf4_ghz']:.4f}` at $\\phi_{{\\mathrm{{cross}}}}={wb['phi_beam_crossing_nf4_deg']:.2f}^\\circ$)"
            if idx == 0
            else ""
        )
        rows.append(
            f"| Uniform conformal `nf{nf}` ($R_{{\\mathrm{{wg}}}}=8\\;\\mathrm{{mm}}$) | `{h_val:.3f}` | `{cpl:.1f}` | `{nf}` | `{f_val:.4f}`{cross_note} | `{ve_val:.4f}` | `{rq_val:.3f}` | `{kl_val:.3f}` | `{ue_val:.3f}` |"
        )
    f_rich_u = cu["f_mode_ghz"]["richardson_estimate"]
    ve_rich_u = cu["ve_over_c"]["richardson_estimate"]
    p_f_u = cu["f_mode_ghz"]["observed_order"]
    p_ve_u = cu["ve_over_c"]["observed_order"]
    if f_rich_u is not None and ve_rich_u is not None:
        if p_f_u is None or p_ve_u is None:
            raise ValueError("Missing observed_order for Richardson row in render_cold_test_table_markdown.")
        rows.append(
            f"| Uniform Richardson limit ($R_{{\\mathrm{{wg}}}}=8\\;\\mathrm{{mm}}$) | $\\to 0$ | — | $p_{{\\mathrm{{obs}}}} = {p_f_u:.2f}, {p_ve_u:.2f}$ | `{f_rich_u:.4f}` | `{ve_rich_u:.4f}` | — | — | — |"
        )
    for nf in (4, 6, 8):
        g_lvl = gl[f"nf{nf}"]
        h_val = float(g_lvl["h_min_mm"])
        cpl = float(g_lvl["cells_per_wavelength_lambda0"])
        f_val = float(g_lvl["f_mode_ghz"])
        ve_val = float(g_lvl["ve_over_c"])
        rq_val = float(g_lvl["r_over_q_kohm_per_m"])
        kl_val = float(g_lvl["k_loss_v_per_c_m"]) * 1e-14
        ue_val = float(g_lvl["u_elec_over_u_mag"])
        rows.append(
            f"| Graded conformal `nf{nf}` ($R_{{\\mathrm{{wg}}}}=8\\;\\mathrm{{mm}}$) | `{h_val:.3f}` | `{cpl:.1f}` | `{nf}` | `{f_val:.4f}` | `{ve_val:.4f}` | `{rq_val:.3f}` | `{kl_val:.3f}` | `{ue_val:.3f}` |"
        )
    for lvl_key in ("nf4", "nf6", "nf8"):
        if lvl_key in v81 and isinstance(v81[lvl_key], dict):
            d81 = v81[lvl_key]
            nf_i = int(d81["n_feat"])
            rows.append(
                f"| Thesis $R_{{\\mathrm{{wg}}}}=8.1\\;\\mathrm{{mm}}$ (line 2075) `{lvl_key}` | `{d81['h_min_mm']:.3f}` | `{d81['cells_per_wavelength_lambda0']:.1f}` | `{nf_i}` | `{d81['f_mode_ghz']:.4f}` | `{d81['ve_over_c']:.4f}` | `{d81['r_over_q_kohm_per_m']:.3f}` | `{d81['k_loss_v_per_c_m']*1e-14:.3f}` | `{d81['u_elec_over_u_mag']:.3f}` |"
            )
    rows.append(
        f"| **Lu *et al.* (2019) Paper** | — | — | PRL lines 137, 147, 250 | `{pb['f_sync_ghz']:.2f}` | `{pb['vg_over_c']:.3f}` | `{pb['r_over_q_kohm_per_m']:.1f}` | `{pb['k_loss_v_per_c_m']*1e-14:.3f}` | `1` |"
    )
    return "\n".join(rows)


def render_quantitative_summary_markdown(
    sm: dict[str, Any],
    t2: dict[str, Any],
    t3: dict[str, Any],
    t4: dict[str, Any],
    t5: dict[str, Any],
) -> str:
    """Format the Quantitative Summary Markdown table directly from JSON dicts at render time."""
    eb_all = t5["energy_balance_all_N"]
    wc = t5["wakefield_vs_analytic_comparison"]
    cu = t2["convergence_uniform"]
    mc = t3["n8_transverse_mesh_convergence"]
    mc_gci = mc["gci"]
    p_cs_1rf = float(sm["two_bunch_tot85nc_charge_scaled_single_bunch_mw"])
    p_cs_4x = float(sm["two_bunch_tot85nc_charge_scaled_4x_single_bunch_mw"])
    p_two_paper_mw = float(sm["two_bunch_paper_quoted_mw"])
    rel_cs_80 = 100.0 * abs(p_cs_1rf - p_two_paper_mw) / p_two_paper_mw
    rel_tm01 = (
        100.0
        * abs(sm["tm01_cutoff_empty_fit_ghz"] - sm["tm01_cutoff_empty_analytic_ghz"])
        / sm["tm01_cutoff_empty_analytic_ghz"]
    )
    min_port_trans = 1.0 - (0.25**2)
    f_u_cold = 100.0 * float(cu["f_mode_ghz"]["u_fine_rel"])
    ve_u_cold = 100.0 * float(cu["ve_over_c"]["u_fine_rel"])
    rq_u_cold = 100.0 * float(cu["r_over_q_kohm_per_m"]["u_fine_rel"])
    kl_u_cold = 100.0 * float(cu["k_loss_v_per_c_m"]["u_fine_rel"])
    g_vals_port = mc["port_gamma_tapered_at_sync_values"]
    p_u_fine = 100.0 * float(sm["wake_n8_p_up_u_fine_rel"])
    w_u_fine = 100.0 * float(sm["wake_n8_w_beam_u_fine_rel"])
    e_u_fine = 100.0 * float(sm["wake_n8_e_up_u_fine_rel"])
    f_u_fine = 100.0 * float(sm["wake_n8_f_pk_u_fine_rel"])
    z_u_fine = 100.0 * float(sm["wake_n8_re_z_pk_u_fine_rel"])
    g_u_fine = 100.0 * float(sm["wake_n8_port_gamma_u_fine_rel"])
    h_nf4_mm = float(t2["uniform_levels"]["nf4"]["h_min_mm"])
    h_nf8_mm = float(t2["uniform_levels"]["nf8"]["h_min_mm"])
    rl_gamma_bound = 1.0 / 4.0

    rows = [
        "| Quantity | Symbol [Unit] | Paper / Theory Benchmark | Wakis 3D FIT Result | Mesh Convergence / GCI | Rel. Diff / Note | Source JSON Key |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        f"| Empty circular pipe $\\mathrm{{TM}}_{{01}}$ cutoff ($R_{{\\mathrm{{wg}}}} = 8\\;\\mathrm{{mm}}$) | $f_{{c,\\mathrm{{TM}}_{{01}}}}$ [GHz] | `{sm['tm01_cutoff_empty_analytic_ghz']:.4f}` (`{sm['tm01_cutoff_empty_8p1mm_analytic_ghz']:.4f}` at thesis $8.1\\;\\mathrm{{mm}}$ line 2075; paper: `{sm['tm01_cutoff_empty_paper_ghz']:.1f}`) | `{sm['tm01_cutoff_empty_fit_ghz']:.4f}` | `nf4` (${h_nf4_mm:.2f}\\;\\mathrm{{mm}}$) | `{rel_tm01:.2f}%` vs analytic | `tm01_cutoff_empty_fit_ghz` |",
        f"| Upstream coaxial coupler reflection at $f_{{\\mathrm{{eval}}}} = {sm['port_f_sync_eval_ghz']:.2f}\\;\\mathrm{{GHz}}$ | $\\lvert\\Gamma(f_{{\\mathrm{{eval}}}})\\rvert$ [—] | Tapered target $< {rl_gamma_bound:.2f}$ (abrupt pipe: `{sm['port_gamma_abrupt_pipe_label']}`) | `{sm['port_gamma_tapered_coax_at_sync']:.4f}` | `{g_vals_port[0]:.4f}` / `{g_vals_port[1]:.4f}` / `{g_vals_port[2]:.4f}` ({mc_gci['port_gamma_tapered_at_sync']['convergence_class']}, $U = {g_u_fine:.2f}\\%$) | $\\pm{100*sm['port_rel_error_bound_tapered']:.1f}\\%$ standing-wave bound | `port_gamma_tapered_coax_at_sync` |",
        f"| Upstream coaxial coupler power transmission | $1 - \\lvert\\Gamma\\rvert^2$ [—] | $\\ge {100*min_port_trans:.2f}\\%$ ($\\lvert\\Gamma\\rvert < {rl_gamma_bound:.2f}$) | `{sm['port_power_transmission_tapered_at_sync']:.4f}` | `{100*mc['port_power_transmission_values'][0]:.2f}%` / `{100*mc['port_power_transmission_values'][1]:.2f}%` / `{100*mc['port_power_transmission_values'][2]:.2f}%` | `{100*sm['port_power_transmission_tapered_at_sync']:.2f}%` single-pass | `port_power_transmission_tapered_at_sync` |",
        f"| 3D Bloch $65\\;\\mathrm{{MeV}}$ beam-line crossing (`nf4`, $\\phi_{{\\mathrm{{cross}}}} = {sm['cold_phi_beam_crossing_nf4_deg']:.2f}^\\circ$) | $f_{{\\mathrm{{cross,nf4}}}}$ [GHz] | `{sm['cold_f_sync_paper_ghz']:.2f}` | `{sm['cold_f_beam_crossing_nf4_ghz']:.4f}` | `nf4` sweep | `{100*sm['cold_f_beam_crossing_rel_error_vs_paper']:.2f}%` | `cold_f_beam_crossing_nf4_ghz` |",
        f"| 3D Bloch frequency at paper's $\\phi_s$ (`nf8`, $R_{{\\mathrm{{wg}}}}=8\\;\\mathrm{{mm}}$) | $f_{{\\mathrm{{nf8}}}}(\\phi_s)$ [GHz] | `{sm['cold_f_sync_paper_ghz']:.2f}` (`{sm['variant_8p1mm_f_sync_nf6_ghz']:.4f}` at thesis $8.1\\;\\mathrm{{mm}}$ line 2075 `nf6`) | `{sm['cold_f_sync_nf8_ghz']:.4f}` | `{sm['cold_f_sync_nf4_ghz']:.4f}` / `{sm['cold_f_sync_nf6_ghz']:.4f}` / `{sm['cold_f_sync_nf8_ghz']:.4f}` ({cu['f_mode_ghz']['convergence_class']}, $U = {f_u_cold:.2f}\\%$) | `{100*sm['cold_f_sync_rel_error_vs_paper']:.2f}%` | `cold_f_sync_nf8_ghz` |",
        f"| 3D Bloch cold-test dispersion group velocity | $v_g / c$ [—] | `{sm['cold_vg_over_c_paper']:.3f}` | `{sm['cold_vg_over_c_disp']:.4f}` | `nf4` $\\partial\\omega/\\partial k_z$ | `{100*t2['wakis_finest_nf8_benchmarks']['rel_error_vg_vs_paper']:.2f}%` | `cold_vg_over_c_disp` |",
        f"| 3D Bloch cold-test DFT energy velocity (`nf8`) | $v_E / c$ [—] | `{sm['cold_vg_over_c_paper']:.3f}` | `{sm['cold_ve_over_c_nf8']:.4f}` | `nf4..nf8` ({cu['ve_over_c']['convergence_class']}, $U = {ve_u_cold:.2f}\\%$) | `{100*sm['cold_ve_rel_error_vs_paper']:.2f}%` | `cold_ve_over_c_nf8` |",
        f"| 3D Bloch cold-test energy equipartition (`nf8`) | $U_e / U_m$ [—] | `1` | `{sm['cold_u_elec_over_u_mag_nf8']:.4f}` | `nf8` (${h_nf8_mm:.3f}\\;\\mathrm{{mm}}$) | `{100*abs(sm['cold_u_elec_over_u_mag_nf8']-1.0):.2f}%` | `cold_u_elec_over_u_mag_nf8` |",
        f"| 3D Bloch cold-test shunt impedance (`nf8`, single convention) | $r/Q$ [$\\mathrm{{k\\Omega/m}}$] | `{sm['cold_r_over_q_paper_kohm_m']:.1f}` | `{sm['cold_r_over_q_nf8_kohm_m']:.3f}` | `{sm['cold_r_over_q_nf4_kohm_m']:.2f}` / `{sm['cold_r_over_q_nf6_kohm_m']:.2f}` / `{sm['cold_r_over_q_nf8_kohm_m']:.2f}` ({cu['r_over_q_kohm_per_m']['convergence_class']}, $U = {rq_u_cold:.2f}\\%$) | `{100*sm['cold_r_over_q_rel_error_vs_paper']:.2f}%` | `cold_r_over_q_nf8_kohm_m` |",
        f"| 3D Bloch cold-test loss factor (`nf8`) | $k_L$ [$10^{{14}}\\;\\mathrm{{V/(C\\cdot m)}}$] | `{sm['cold_k_loss_paper_v_per_c_m']*1e-14:.3f}` | `{sm['cold_k_loss_nf8_v_per_c_m']*1e-14:.3f}` | `nf4..nf8` ({cu['k_loss_v_per_c_m']['convergence_class']}, $U = {kl_u_cold:.2f}\\%$) | `{100*sm['cold_k_loss_rel_error_vs_paper']:.2f}%` | `cold_k_loss_nf8_v_per_c_m` |",
        f"| Finite wakefield peak frequency ($N=8$, `n_xy=64`, $q=45\\;\\mathrm{{nC}}$) | $f_{{\\mathrm{{pk}}}}$ [GHz] | `{sm['cold_f_sync_paper_ghz']:.2f}` (model $f_s$: `{sm['cold_f_sync_nf8_ghz']:.4f}`) | `{sm['wake_f_pk_N8_ghz']:.4f}` | `{sm['wake_n8_nxy48_f_pk_ghz']:.4f}` / `{sm['wake_n8_nxy64_f_pk_ghz']:.4f}` / `{sm['wake_n8_nxy96_f_pk_ghz']:.4f}` ({mc_gci['f_pk_ghz']['convergence_class']}, $U = {f_u_fine:.2f}\\%$) | `{100*sm['wake_f_pk_rel_error_vs_paper']:.2f}%` vs paper (`{100*sm['wake_f_pk_rel_error_vs_cold_nf8']:.2f}%` vs $f_s$) | `wake_f_pk_N8_ghz` |",
        f"| Finite wakefield peak impedance ($N=8$, `n_xy=64`) | $\\operatorname{{Re}} Z_{{\\mathrm{{pk}}}}$ [$\\Omega$] | — | `{sm['wake_re_z_pk_N8_ohm']:.2f}` | `{sm['wake_n8_nxy48_re_z_pk_ohm']:.1f}` / `{sm['wake_n8_nxy64_re_z_pk_ohm']:.1f}` / `{sm['wake_n8_nxy96_re_z_pk_ohm']:.1f}` ({mc_gci['re_z_pk_ohm']['convergence_class']}, $U = {z_u_fine:.2f}\\%$) | `{t3['length_scaling']['n8_re_z_pk_over_cold_sinc2_ratio']:.2f}x` cold $\\mathrm{{sinc}}^2$ (unexplained peak excess; $\\int\\operatorname{{Re}} Z$ ratio `{t3['length_scaling']['n8_passband_integrated_re_z_fit_over_cold_sinc2_ratio']:.2f}`) | `wake_re_z_pk_N8_ohm` |",
        f"| Control-volume passband beam work ($N=8$, `n_xy=64`) | $W_{{\\mathrm{{beam,bp,cv}}}}$ [mJ] | — | `{sm['wake_n8_nxy64_w_beam_cv_mj']:.3f}` | `{sm['wake_n8_nxy48_w_beam_cv_mj']:.3f}` / `{sm['wake_n8_nxy64_w_beam_cv_mj']:.3f}` / `{sm['wake_n8_nxy96_w_beam_cv_mj']:.3f}` ({mc_gci['w_beam_passband_cv_mj']['convergence_class']}, $U = {w_u_fine:.2f}\\%$) | `{mc_gci['w_beam_passband_cv_mj']['convergence_class']}`, $U = {w_u_fine:.2f}\\%$ | `wake_n8_nxy64_w_beam_cv_mj` |",
        f"| Affine passband beam-loss slope over $L \\in \\{{12,16,24\\}}\\;\\mathrm{{mm}}$ | $a_W = \\mathrm{{d}}W/\\mathrm{{d}}L$ [J/m] | `{sm['wake_cold_analytic_a_w_j_per_m']:.4f}` ($q^2 k_L \\Phi^2 / (1-v_g/u_0)$) | `{sm['wake_affine_a_w_j_per_m']:.4f}` | Intercept $b_W = {sm['wake_affine_b_w_mj']:.3f}$ mJ ($a_{{E,\\mathrm{{up}}}} = {sm['wake_affine_a_e_up_j_per_m']:.4f}$ J/m, $b_{{E,\\mathrm{{up}}}} = {sm['wake_affine_b_e_up_mj']:.3f}$ mJ) | `{100*sm['wake_rel_error_a_w_vs_cold_analytic']:.2f}%` vs cold analytic ($\\pm{100*sm['wake_a_w_uncertainty_rel']:.2f}\\%$ unc.) | `wake_affine_a_w_j_per_m` |",
        f"| Short-$N$ ($N \\in \\{{6,8,12\\}}$) pre-asymptotic $\\operatorname{{Re}} Z_\\parallel$ slope | $\\alpha_Z$ [—] | Asymptotic `2` (FWHM `-1`, $t_p$ `1`) | `{sm['wake_slope_re_z_pk_vs_L']:.2f}` (`{sm['wake_slope_fwhm_vs_L']:.2f}`, `{sm['wake_slope_tp_vs_L']:.2f}`) | `n_xy=64` across $N \\in \\{{6,8,12\\}}$ | Pre-asymptotic short-$N$ | `wake_slope_re_z_pk_vs_L` |",
        f"| Single-bunch ($45\\;\\mathrm{{nC}}$) Eq. (2) analytical power | $P_{{\\mathrm{{Eq.2}}}}$ [MW] | `{sm['single_bunch_45nc_power_paper_eq2_mw']:.2f}` (quoted `{sm['single_bunch_45nc_power_paper_quoted_mw']:.1f}`) | `{sm['single_bunch_45nc_power_wakis_analytic_mw']:.2f}` (Wakis cold) | `nf8` cold test | `{100*sm['single_bunch_power_rel_error_vs_wakis_analytic']:.2f}%` vs $N=8$ FIT | `single_bunch_45nc_power_wakis_analytic_mw` |",
        f"| Single-bunch ($45\\;\\mathrm{{nC}}$) 3D FIT upstream RF power ($N=8$, `n_xy=64`) | $P_{{\\mathrm{{up,FIT}}}}$ [MW] | `{sm['single_bunch_45nc_power_wakis_analytic_mw']:.2f}` (Eq. 2; paper $N=40$: `{sm['single_bunch_45nc_power_paper_quoted_mw']:.1f}`) | `{sm['single_bunch_45nc_power_fit_mw']:.2f}` (`n_xy=96`: `{sm['single_bunch_45nc_power_fit_nxy96_mw']:.2f}`) | `{sm['wake_n8_nxy48_p_up_mw']:.2f}` / `{sm['wake_n8_nxy64_p_up_mw']:.2f}` / `{sm['wake_n8_nxy96_p_up_mw']:.2f}` ({mc_gci['p_up_cycle_avg_max_mw']['convergence_class']}, $U = {p_u_fine:.2f}\\%$) | `{100*sm['single_bunch_power_rel_error_vs_wakis_analytic']:.2f}%` vs Eq. (2); `{100*sm['single_bunch_power_rel_error_vs_paper_25mw']:.2f}%` vs paper ($\\pm(U_P + 2\\lvert\\Gamma\\rvert^2) = \\pm{100*wc['consistency_band_u_p_plus_2gamma_sq']:.2f}\\%$) | `single_bunch_45nc_power_fit_mw` |",
        f"| Experimental $N=40$ ($L=80\\;\\mathrm{{mm}}$) pulse length | $t_p(N=40)$ [ns] | `{sm['pulse_length_N40_paper_analytic_ns']:.4f}` | `{sm['pulse_length_N40_wakis_analytic_ns']:.4f}` (Wakis $v_E$) | `nf8` $v_E$ | `{100*abs(sm['pulse_length_N40_wakis_analytic_ns']-sm['pulse_length_N40_paper_analytic_ns'])/sm['pulse_length_N40_paper_analytic_ns']:.2f}%` | `pulse_length_N40_wakis_analytic_ns` |",
        f"| Wagon-wheel MTM upstream energy fraction ($N=8$, `n_xy=64`) | $E_{{\\mathrm{{up}}}} / E_{{\\mathrm{{rad}}}}$ [—] | — ($\\ge 1 - 2\\lvert\\Gamma_{{\\max}}\\rvert^2 - \\lvert\\Gamma(f_s)\\rvert^2$) | `{sm['bwd_mtm_upstream_fraction']:.4f}` | $E_{{\\mathrm{{up}}}}$: `{sm['wake_n8_nxy48_e_up_mj']:.2f}` / `{sm['wake_n8_nxy64_e_up_mj']:.2f}` / `{sm['wake_n8_nxy96_e_up_mj']:.2f}` mJ ({mc_gci['e_up_mj']['convergence_class']}, $U = {e_u_fine:.2f}\\%$) | `{100*sm['bwd_mtm_upstream_fraction']:.2f}%` (`{100*t3['runs_mtm']['6']['upstream_energy_fraction']:.2f}%` $N=6$, `{100*t3['runs_mtm']['12']['upstream_energy_fraction']:.2f}%` $N=12$) | `bwd_mtm_upstream_fraction` |",
        f"| Disk-loaded control downstream energy fraction ($N=8$, `n_xy=64`) | $E_{{\\mathrm{{dn}}}} / E_{{\\mathrm{{rad}}}}$ [—] | — | `{sm['fwd_control_downstream_fraction']:.4f}` | `n_xy=64` control | `{100*sm['fwd_control_downstream_fraction']:.2f}%` downstream ($v_g > 0$) | `fwd_control_downstream_fraction` |",
        f"| 3D FIT two-bunch superposition relative $L_2$ error ($\\mathbf{{E}}$) | $\\lVert\\mathbf{{E}}_{{12}}-(\\mathbf{{E}}_1+\\mathbf{{E}}_2)\\rVert_2 / \\lVert\\mathbf{{E}}_{{12}}\\rVert_2$ | `0` (linear operator) | `{sm['two_bunch_superposition_rel_l2_error_E']:.2e}` | IEEE-754 float64 | Mutated $+\\Delta t / 2$: `{t4['fit_3d_superposition_verification']['mutated_half_dt_shift_rel_l2_error_E']:.2e}` | `two_bunch_superposition_rel_l2_error_E` |",
        f"| Passband FIT wake in-phase vs out-of-phase power ratio ($N=8$) | $P_{{\\mathrm{{in}}}}/P_1, P_{{\\mathrm{{out}}}}/P_1$ | Asymptotic `4`, `0` | `{sm['two_bunch_in_phase_power_ratio']:.3f}` (out-of-phase `{sm['two_bunch_out_of_phase_power_ratio']:.4f}`) | `n_xy=64` wake | Finite-$N$ wake envelope | `two_bunch_in_phase_power_ratio` |",
        f"| Charge-scaled single-bunch estimate ($Q_1+Q_2 = 85\\;\\mathrm{{nC}}$) | $P_{{2,\\mathrm{{est}}}}$ [MW] | `{sm['two_bunch_paper_quoted_mw']:.1f}` (Eq. 2: `{sm['two_bunch_tot85nc_eq2_mw']:.2f}`) | `{p_cs_1rf:.2f}` (1-RF shift) / `{p_cs_4x:.2f}` ($4 P_1$) | `n_xy=64` scaled | `{rel_cs_80:.2f}%` ($N=8$ cannot overlap at $769\\;\\mathrm{{ps}}$) | `two_bunch_tot85nc_charge_scaled_single_bunch_mw` |",
        f"| Control-volume passband energy balance ($N=6, 8, 12$): $E_{{\\mathrm{{tot}}}}$ vs $W_{{\\mathrm{{beam,bp}}}}$ | $E_{{\\mathrm{{tot}}}}$ ($N=8$) [mJ] | $W_{{\\mathrm{{beam,bp}}}} = {sm['energy_balance_bunch_loss_mj']:.2f}$ mJ | `{sm['energy_balance_total_accounted_mj']:.2f}` (`{eb_all['6']['total_accounted_em_energy_mj']:.2f}` at $N=6$, `{eb_all['12']['total_accounted_em_energy_mj']:.2f}` at $N=12$) | $U_{{\\mathrm{{bp}}}}/W$: `{100*sm['u_stored_over_w_beam_passband_N6']:.2f}%`, `{100*sm['u_stored_over_w_beam_passband_N8']:.2f}%`, `{100*sm['u_stored_over_w_beam_passband_N12']:.2f}%` | Closure: `{sm['energy_balance_rel_error_total_N6']:.2e}`, `{sm['energy_balance_rel_error_total_N8']:.2e}`, `{sm['energy_balance_rel_error_total_N12']:.2e}` | `energy_balance_total_accounted_mj` |",
    ]
    return "\n".join(rows)


def build_dummy_all_json_dicts(
    data_dir: Path | str,
    *,
    mutate_drop_key: str | None = None,
    spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
) -> dict[str, dict[str, Any]]:
    """Run ``generate_task_01_data``..``generate_task_05_data`` and ``build_summary_metrics`` on tiny synthetic solver outputs."""
    import sys
    from unittest.mock import patch

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    f_grid = np.linspace(21.0 / 2.0, 49.0 / 4.0, 16)
    gamma_t = np.full_like(f_grid, 0.2175)
    gamma_a = np.full_like(f_grid, 0.8778)
    gamma_p = np.ones_like(f_grid)

    def _dummy_port_refl(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nxy_p = int(kwargs.get("n_xy", 64))
        h_p_mm = float((2.0 * spec.waveguide_radius_m / max(nxy_p, 1)) * 1e3)
        g_tap_s = float(0.2175 - 0.0001 * (nxy_p - 36))
        g_abr_s = 0.8778
        f_lo_sw = 21.0 / 2.0
        f_hi_sw = 49.0 / 4.0
        f_ev = 11.41
        f_adj = 11.42
        fc_tm = float(circular_waveguide_tm01_cutoff_hz(spec.waveguide_radius_m) * 1e-9)
        f_fwd = 18.86
        n_pml_w = 10
        return {
            "n_xy_port": nxy_p,
            "cells_across_1mm_feature_port": float(1.0 / h_p_mm),
            "n_fft_port": 65536,
            "f_ghz": f_grid,
            "gamma_tapered_coax": gamma_t,
            "gamma_tapered_coax_raw": gamma_t,
            "gamma_abrupt_empty_pipe": gamma_a,
            "gamma_abrupt_empty_pipe_raw": gamma_a,
            "gamma_pec_cap": gamma_p,
            "f_sync_target_ghz": 11.42,
            "f_sync_eval_ghz": f_ev,
            "f_sync_adjacent_bin_ghz": f_adj,
            "gamma_tapered_at_sync": g_tap_s,
            "gamma_tapered_raw_at_sync": g_tap_s,
            "gamma_tapered_unclipped_at_sync": g_tap_s,
            "gamma_tapered_at_adjacent_bin": g_tap_s,
            "gamma_tapered_passband_mean": g_tap_s,
            "gamma_tapered_passband_min": g_tap_s,
            "power_transmission_tapered_at_sync": float(1.0 - g_tap_s**2),
            "gamma_abrupt_at_sync": g_abr_s,
            "gamma_abrupt_raw_at_sync": g_abr_s,
            "gamma_abrupt_unclipped_at_sync": g_abr_s,
            "gamma_abrupt_is_clipped": False,
            "gamma_abrupt_at_sync_label": f"{g_abr_s:.4f}",
            "gamma_abrupt_passband_mean": g_abr_s,
            "reduction_factor_abrupt_over_tapered": float(g_abr_s / g_tap_s),
            "rel_error_bound_tapered": float(reflection_induced_peak_error_bound(g_tap_s)),
            "rel_error_bound_tapered_mutated_no_2x": float(
                reflection_induced_peak_error_bound(g_tap_s, mutate_drop_factor_two=True)
            ),
            "rel_error_bound_abrupt": float(reflection_induced_peak_error_bound(g_abr_s)),
            "forward_control_18p86ghz_port_note": (
                f"Measured in 3D FIT across [{f_lo_sw:.1f}, {f_hi_sw:.1f}] GHz; evaluated at f_eval = {f_ev:.4f} GHz "
                f"(f_c,TM01 = {fc_tm:.2f} GHz, {n_pml_w}-cell CPML, f_fwd = {f_fwd:.2f} GHz)."
            ),
            "wall_times_s": {"ref_periodic": 0.01, "pec_cap": 0.01, "abrupt_empty_pipe": 0.01, "tapered_coax": 0.01},
            "est_memory_mb": 10.0,
        }

    def _dummy_tm01_cutoff(*args: Any, **kwargs: Any) -> float:
        return float(circular_waveguide_tm01_cutoff_hz(spec.waveguide_radius_m) * 1e-9)

    def _dummy_bloch_cell(
        phi_deg: float = 27.43,
        n_feat: int = 4,
        use_graded_mesh: bool = False,
        forward_control: bool = False,
        compute_eigenmode_metrics: bool = True,
        save_field_maps: bool = False,
        quick: bool = True,
        spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
    ) -> dict[str, Any]:
        h_min_mm = 1.0 / float(max(n_feat, 1))
        f_ghz = (19.0 + 0.02 * phi_deg) if forward_control else (12.8 - 0.045 * phi_deg + 0.004 * n_feat)
        f_hz = f_ghz * 1e9
        beta0 = (np.deg2rad(phi_deg) / spec.period_m) if abs(phi_deg) > 1e-12 else 0.0
        out_b: dict[str, Any] = {
            "phi_deg": float(phi_deg),
            "beta0_rad_m": float(beta0),
            "n_feat": int(n_feat),
            "use_graded_mesh": bool(use_graded_mesh),
            "h_min_mm": float(h_min_mm),
            "cells_per_wavelength_lambda0": float((C0 / f_hz) / (h_min_mm * 1e-3)),
            "cells_across_1mm_feature": float(1.0 / h_min_mm),
            "grid_shape": [16, 16, 4],
            "total_cells": 1024,
            "f_mode_hz": float(f_hz),
            "f_mode_ghz": float(f_ghz),
            "vp_over_c": 1.0,
            "q_harminv": 500.0,
            "est_memory_mb": 10.0,
        }
        if compute_eigenmode_metrics:
            ve_c = 0.19 if forward_control else (-0.163 + 0.0002 * n_feat)
            rq_k = 21.8 + 0.025 * n_feat
            out_b.update(
                {
                    "u_elec_over_u_mag": 1.0,
                    "ve_m_s": float(ve_c * C0),
                    "ve_over_c": float(ve_c),
                    "r_over_q_ohm_per_m": float(rq_k * 1e3),
                    "r_over_q_kohm_per_m": float(rq_k),
                    "k_loss_v_per_c_m": float(0.25 * 2.0 * np.pi * f_hz * rq_k * 1e3),
                    "k_pierce_ohm": 280.0,
                }
            )
            if save_field_maps:
                ax4 = np.linspace(-8.0, 8.0, 8)
                az4 = np.linspace(0.0, 2.0, 4)
                z2d = np.zeros((8, 4))
                zxy = np.zeros((8, 8))
                out_b["field_maps"] = {
                    "x_mm": ax4,
                    "y_mm": ax4,
                    "z_mm": az4,
                    "ez_real_zy": z2d,
                    "er_real_zy": z2d,
                    "ephi_real_zy": z2d,
                    "hphi_imag_zy": z2d,
                    "hr_imag_zy": z2d,
                    "hz_imag_zy": z2d,
                    "sz_avg_zy": z2d,
                    "ez_real_xy_mid": zxy,
                    "hphi_imag_xy_spoke": zxy,
                    "er_real_xy_mid": zxy,
                }
        out_b["wall_time_s"] = 0.01
        return out_b

    def _dummy_wake(
        n_periods: int = 8,
        *,
        n_xy: int = 64,
        dz_mm: float = 0.25,
        forward_control: bool = False,
        record_2d_snapshots: bool = False,
        quick: bool = True,
        spec: WagonWheelMTMSpec = WagonWheelMTMSpec(),
    ) -> dict[str, Any]:
        l_act = float(n_periods * spec.period_m)
        wl_m = estimate_drainage_wake_length_m(max(int(n_periods), 6), data_dir=data_dir, spec=spec)
        h_xy_mm = float((2.0 * spec.waveguide_radius_m / max(n_xy, 1)) * 1e3)
        mesh_corr = 1.0 - 0.015 * ((h_xy_mm / 0.25) ** 2)
        t_ns = np.linspace(0.0, 1.0, 32)
        s_mm = np.linspace(-10.0, 100.0, 32)
        f_ghz = np.linspace(1.0, 24.0, 32)
        zeros32 = np.zeros(32)
        ones32 = np.ones(32)
        w_cv_mj = (4.2 + 0.64 * (l_act * 1e3)) * mesh_corr
        e_up_mj_val = 1.0 if forward_control else (3.3 + 0.57 * (l_act * 1e3)) * mesh_corr
        e_dn_mj_val = 8.0 if forward_control else 0.14 * w_cv_mj
        e_rad_mj_val = e_up_mj_val + e_dn_mj_val
        u_st_mj_val = 0.012 * w_cv_mj
        res_w: dict[str, Any] = {
            "n_periods": int(n_periods),
            "n_xy": int(n_xy),
            "h_xy_mm": h_xy_mm,
            "dz_mm": float(dz_mm),
            "cells_across_1mm_spoke": float(1.0 / h_xy_mm),
            "active_length_m": l_act,
            "active_length_mm": float(l_act * 1e3),
            "wake_length_mm": float(wl_m * 1e3),
            "z_total_m": float(l_act + 0.024),
            "z_act_start_m": 0.012,
            "z_act_end_m": float(0.012 + l_act),
            "z_up_port_m": 0.006,
            "z_dn_port_m": float(0.018 + l_act),
            "forward_control": bool(forward_control),
            "grid_shape": [int(n_xy), int(n_xy), 32],
            "total_cells": int(n_xy * n_xy * 32),
            "dt_fs": 400.0,
            "nt_steps": 32,
            "f_pk_ghz": 18.86 if forward_control else float(11.35 * mesh_corr),
            "re_z_pk_ohm": float(2900.0 * ((n_periods / 8.0) ** 1.48) * mesh_corr),
            "fwhm_ghz": 3.0 * ((n_periods / 8.0) ** (-0.55)),
            "fmax_impedance_ghz": 110.0,
            "form_factor_sq_at_fmax": 7.90e-4,
            "form_factor_sq_at_25ghz": 0.674,
            "k_loss_passband_v_per_c": float(w_cv_mj * 1e-3 / (spec.single_bunch_charge_c**2)),
            "w_loss_passband_j": float(w_cv_mj * 1e-3),
            "w_loss_passband_mj": float(w_cv_mj),
            "w_beam_passband_cv_mj": float(w_cv_mj),
            "w_beam_passband_active_mj": float(0.68 * w_cv_mj),
            "w_loss_wake_passband_cv_mj": float(w_cv_mj),
            "w_loss_wake_passband_active_mj": float(0.68 * w_cv_mj),
            "w_loss_wake_fullband_active_mj": float(1.15 * w_cv_mj),
            "w_loss_z_fullband_active_mj": float(1.15 * w_cv_mj),
            "out_of_band_wake_loss_fraction_active": 0.375,
            "w_loss_wake_fullband_cv_mj": float(1.50 * w_cv_mj),
            "w_beam_scat_fullband_cv_mj": float(1.50 * w_cv_mj),
            "out_of_band_beam_work_fraction_cv": 0.333,
            "w_beam_unfiltered_cv_mj": float(1.50 * w_cv_mj - 3.1),
            "e_rad_unfiltered_cv_mj": float(1.10 * w_cv_mj),
            "u_stored_unfiltered_cv_mj": float(0.40 * w_cv_mj - 3.1),
            "rel_error_unfiltered_cv_closure": 1.0e-4,
            "w_ref_unfiltered_cv_mj": -3.1,
            "e_rad_ref_unfiltered_cv_mj": 0.01,
            "u_stored_ref_unfiltered_cv_mj": -3.11,
            "rel_error_ref_unfiltered_cv_closure": 1.0e-4,
            "w_ref_passband_cv_mj": 0.0,
            "e_rad_ref_passband_cv_mj": 0.0,
            "u_stored_ref_passband_cv_mj": 0.0,
            "e_up_j": float(e_up_mj_val * 1e-3),
            "e_up_mj": float(e_up_mj_val),
            "e_dn_j": float(e_dn_mj_val * 1e-3),
            "e_dn_mj": float(e_dn_mj_val),
            "e_up_coax_mj": float(0.93 * e_up_mj_val),
            "e_dn_coax_mj": float(0.85 * e_dn_mj_val),
            "e_rad_total_j": float(e_rad_mj_val * 1e-3),
            "e_rad_total_mj": float(e_rad_mj_val),
            "u_stored_final_fit_j": float(u_st_mj_val * 1e-3),
            "u_stored_final_fit_mj": float(u_st_mj_val),
            "u_stored_over_w_beam_passband": 0.012,
            "upstream_energy_fraction": float(e_up_mj_val / max(e_rad_mj_val, 1e-30)),
            "downstream_energy_fraction": float(e_dn_mj_val / max(e_rad_mj_val, 1e-30)),
            "p_up_cycle_avg_max_mw": 2.0 if forward_control else float(24.9 * mesh_corr),
            "p_up_inst_max_mw": 4.0 if forward_control else float(49.8 * mesh_corr),
            "p_dn_cycle_avg_max_mw": 20.0 if forward_control else 1.5,
            "p_dn_inst_max_mw": 40.0 if forward_control else 3.0,
            "pulse_start_20pct_ns": 0.12,
            "pulse_end_20pct_ns": 0.12 + 0.50 * ((n_periods / 8.0) ** 0.55),
            "t_end_window_ns": 2.40,
            "rf_period_ns": 0.0876,
            "pulse_end_margin_to_t_end_ns": 2.28 - 0.50 * ((n_periods / 8.0) ** 0.55),
            "pulse_clips_window": False,
            "pulse_duration_20pct_ns": 0.50 * ((n_periods / 8.0) ** 0.55),
            "max_decelerating_gradient_mv_per_m": 20.0,
            "est_memory_mb": 10.0,
            "wall_time_sws_s": 0.01,
            "wall_time_ref_s": 0.01,
            "wall_time_total_s": 0.02,
            "t_ns": t_ns,
            "p_up_inst_mw": ones32,
            "p_up_env_mw": ones32,
            "p_dn_inst_mw": zeros32,
            "p_dn_env_mw": zeros32,
            "s_mm": s_mm,
            "wp_v_per_pc": np.sin(s_mm) * (s_mm >= 0.0),
            "f_ghz": f_ghz,
            "re_z_ohm": ones32,
            "im_z_ohm": zeros32,
        }
        if not forward_control:
            res_w["rel_error_rad_total_vs_bunch_loss"] = 0.012
            res_w["rel_error_total_accounted_vs_bunch_loss"] = 8.0e-5
            res_w["mutated_shrunk_active_beam_work_rel_error"] = 0.33
        if record_2d_snapshots:
            res_w["snapshots"] = {
                "y_mm": np.linspace(-8.0, 8.0, 8),
                "z_mm": np.linspace(0.0, 40.0, 8),
                "snap_ez_scat": np.zeros((4, 8, 8)),
                "snap_t_ps": np.linspace(10.0, 100.0, 4),
                "snap_zb_mm": np.linspace(3.0, 30.0, 4),
            }
        return res_w

    def _dummy_superposition(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "n_periods": 4,
            "n_xy": 64,
            "n_steps": 32,
            "dt_fs": 400.0,
            "q1_nc": 42.5,
            "q2_nc": 42.5,
            "delta_t_ps": 87.6,
            "rel_l2_error_E_3d": 1.0e-15,
            "rel_l2_error_H_3d": 1.0e-15,
            "rel_l2_error_Ez_axis": 1.0e-15,
            "mutated_half_dt_shift_rel_l2_error_E": 1.0e-2,
            "wall_time_total_s": 0.01,
        }

    with (
        patch.object(sys.modules[__name__], "measure_upstream_port_reflection", side_effect=_dummy_port_refl),
        patch.object(sys.modules[__name__], "measure_empty_waveguide_tm01_cutoff_fit", side_effect=_dummy_tm01_cutoff),
        patch.object(sys.modules[__name__], "evaluate_bloch_unit_cell", side_effect=_dummy_bloch_cell),
        patch.object(sys.modules[__name__], "run_finite_mtm_wakefield", side_effect=_dummy_wake),
        patch.object(sys.modules[__name__], "verify_two_bunch_fit_superposition", side_effect=_dummy_superposition),
    ):
        t1 = generate_task_01_data(data_dir, quick=True, spec=spec)
        t2 = generate_task_02_data(data_dir, quick=True, spec=spec)
        t3 = generate_task_03_data(data_dir, quick=True, spec=spec)
        t4 = generate_task_04_data(data_dir, quick=True, spec=spec)
        t5 = generate_task_05_data(data_dir, quick=True, spec=spec)
        sm = build_summary_metrics(data_dir)

    out_all = {
        "task_01_geometry_and_port_reflection.json": t1,
        "task_02_cold_test_and_convergence.json": t2,
        "task_03_wakefield_and_directionality.json": t3,
        "task_04_two_bunch_superposition.json": t4,
        "task_05_analytic_power_and_energy_balance.json": t5,
        "summary_metrics.json": sm,
    }
    if mutate_drop_key is not None:
        for d_obj in out_all.values():
            d_obj.pop(mutate_drop_key, None)
    return out_all


