"""Helper utilities for Tutorial 8.5: Beam-wave impedance of a slow-wave structure.

Provides:
1. Cold-test dispersion and Pierce impedance benchmark loaders from Tutorial 6.2
   (``tutorials/06_periodic_structures/06-02_sws_cold_test/data/*.npz``).
2. Closed-form analytical models ("derived here") for finite-length SWS beam-driven
   longitudinal impedance Re Z_||(omega), sinc^2 linewidth, infinite-structure
   stopping power dW/dz, Gaussian bunch form factor |F(omega)|^2, loss factor k_||,
   sheet-beam transverse averaging factor, velocity-detuning peak shift, port-reflection
   error bounds, and standing-wave cavity R/Q vs travelling-wave K_n.
3. Wakis 3D FIT simulation runners for:
   - Port termination reflection |Gamma(omega)| of adiabatic vane tapers into CPML,
   - 2D Karetnikova sheet-beam staggered/mirror grating SWS beam-wake impedance,
   - Forward-wave vs backward-wave Poynting flux and time-domain wave-packet snapshots,
   - Velocity detuning and reflecting-end (PEC cavity) mutation checks,
   - 3-level mesh convergence + finest-level converged band & conformal vs staircase ablation,
   - 3D Marosi V-band staggered-pillar glide-symmetric SWS beam-wake impedance.
4. Setup schematic figure generators using ``wakis.schematic.save_setup_figures``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.constants import c, e, epsilon_0, m_e, mu_0

from wakis import GridFIT3D, SolverFIT3D, WakeSolver, figstyle, schematic
from wakis.estimate import estimate, preflight
from wakis.sources import Beam, GaussianSource
from wakis.templates import meshing
from wakis.templates.photonics import Box

SINC2_HALF_MAX_X: float = 1.3915573782515098
Z0_FREE_SPACE: float = float(np.sqrt(mu_0 / epsilon_0))


@dataclass(frozen=True)
class KaretnikovaGratingSpec:
    """2D sheet-beam staggered/mirror grating SWS from Karetnikova et al. (2018)."""

    period_m: float = 0.50e-3
    vane_thickness_m: float = 0.10e-3
    vane_height_m: float = 0.30e-3
    half_tunnel_m: float = 0.10e-3
    beam_thickness_m: float = 0.15e-3
    beam_voltage_v: float = 20.0e3
    transverse_width_m: float = 1.00e-3

    @property
    def total_height_m(self) -> float:
        return 2.0 * (self.half_tunnel_m + self.vane_height_m)

    @property
    def groove_width_m(self) -> float:
        return self.period_m - self.vane_thickness_m


@dataclass(frozen=True)
class MarosiGlideSWSSpec:
    """3D V-band glide-symmetric pillar SWS from Marosi et al. (2025)."""

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


def relativistic_beam_velocity(voltage_v: float) -> tuple[float, float]:
    """Return beam velocity u0 [m/s] and Lorentz factor gamma for accelerating voltage V0 [V]."""
    if voltage_v <= 0.0:
        raise ValueError("Accelerating voltage must be strictly positive.")
    gamma = 1.0 + (e * float(voltage_v)) / (m_e * c**2)
    beta = float(np.sqrt(1.0 - 1.0 / (gamma * gamma)))
    return beta * c, float(gamma)


def beam_voltage_from_velocity(u0_m_s: float) -> float:
    """Return accelerating voltage V0 [V] for a given beam velocity u0 [m/s]."""
    beta = float(u0_m_s) / c
    if not (0.0 < beta < 1.0):
        raise ValueError("Beam velocity u0 must satisfy 0 < u0 < c.")
    gamma = 1.0 / float(np.sqrt(1.0 - beta * beta))
    return float((gamma - 1.0) * m_e * (c**2) / e)


def tutorial_62_data_dir() -> Path:
    """Locate Tutorial 6.2 cold-test data folder relative to the repository."""
    here = Path(__file__).resolve().parent
    cand = (
        here.parent.parent
        / "06_periodic_structures"
        / "06-02_sws_cold_test"
        / "data"
    )
    if not cand.is_dir():
        raise FileNotFoundError(f"Tutorial 6.2 data directory not found at {cand}")
    return cand


def _local_tmp_stl_path(prefix: str = "wakis_sws_") -> Path:
    """Create a temporary STL path inside the tutorial data folder."""
    tmp_dir = Path(__file__).resolve().parent / "data" / "_tmp_stl"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".stl", dir=str(tmp_dir))
    os.close(fd)
    return Path(name)


def transverse_decay_constant(beta_n_rad_m: float, f_hz: float) -> float:
    """Return the transverse slow-wave decay constant tau_n = sqrt(beta_n^2 - (omega/c)^2) [1/m]."""
    k0 = 2.0 * np.pi * float(f_hz) / c
    tau_sq = float(beta_n_rad_m) ** 2 - k0 * k0
    return float(np.sqrt(max(tau_sq, 0.0)))


def sheet_beam_averaging_factor(
    beta_n_rad_m: float,
    f_sync_hz: float,
    beam_thickness_m: float,
) -> float:
    """Return the continuum sheet-beam transverse field-averaging factor eta_sheet >= 1 (derived here).

    For a 2D TM slow space harmonic with symmetric E_{zn}(y) = E_{zn}(0) cosh(tau_n y)
    and tau_n = sqrt(beta_n^2 - (omega/c)^2), averaging |E_{zn}(y)|^2 over a uniform
    sheet beam of thickness H_b gives:
        eta_sheet = (1 / H_b) int_{-H_b/2}^{+H_b/2} cosh^2(tau_n y) dy
                  = 0.5 * (1 + sinh(tau_n H_b) / (tau_n H_b)).
    """
    tau = transverse_decay_constant(beta_n_rad_m, f_sync_hz)
    if tau <= 0.0 or beam_thickness_m <= 0.0:
        return 1.0
    arg = tau * float(beam_thickness_m)
    if arg < 1e-8:
        return 1.0
    return float(0.5 * (1.0 + np.sinh(arg) / arg))


def sheet_beam_node_averaging_factor(
    beta_n_rad_m: float,
    f_sync_hz: float,
    beam_thickness_m: float,
    dy_m: float,
) -> float:
    """Return the discrete grid-node mean of cosh^2(tau_n y) over |y| <= H_b/2 matching Tutorial 6.2."""
    tau = transverse_decay_constant(beta_n_rad_m, f_sync_hz)
    half_hb = 0.5 * float(beam_thickness_m)
    n_half = int(round(half_hb / float(dy_m)))
    y_nodes = np.linspace(-n_half * float(dy_m), n_half * float(dy_m), 2 * n_half + 1)
    return float(np.mean(np.cosh(tau * y_nodes) ** 2))


def mirror_odd_harmonic_k_at_y_from_node_avg(
    k_node_avg_ohm: float | np.ndarray,
    beta_n_rad_m: float | np.ndarray,
    f_hz: float | np.ndarray,
    *,
    y_target_m: float,
    beam_thickness_m: float,
    dy_m: float,
) -> float | np.ndarray:
    """Convert Tutorial 6.2's node-averaged mirror-odd Pierce impedance <K_0>_nodes to K_0(y_target).

    In the mirror-symmetric grating, the fundamental n = 0 passband has odd transverse
    symmetry E_{z,0}(y) = A sinh(tau_0 y), so E_{z,0}(0) = 0 on axis while Tutorial 6.2
    records the discrete node average <K_0>_nodes = mean_{|y_j| <= H_b/2} K_0(y_j).
    Evaluating at off-axis beam position y = y_target gives:
        K_0(y_target) = <K_0>_nodes * sinh^2(tau_0 y_target) / mean_{|y_j| <= H_b/2}(sinh^2(tau_0 y_j)).
    """
    k_arr = np.asarray(k_node_avg_ohm, dtype=float)
    beta_arr = np.asarray(beta_n_rad_m, dtype=float)
    f_arr = np.asarray(f_hz, dtype=float)
    k0_arr = 2.0 * np.pi * f_arr / c
    tau_arr = np.sqrt(np.maximum(beta_arr * beta_arr - k0_arr * k0_arr, 1e-18))

    half_hb = 0.5 * float(beam_thickness_m)
    n_half = int(round(half_hb / float(dy_m)))
    y_nodes = np.linspace(-n_half * float(dy_m), n_half * float(dy_m), 2 * n_half + 1)

    sinh2_nodes = np.mean(
        np.sinh(tau_arr[..., None] * y_nodes) ** 2, axis=-1
    )
    sinh2_target = np.sinh(tau_arr * float(y_target_m)) ** 2
    val = k_arr * sinh2_target / np.maximum(sinh2_nodes, 1e-30)
    if np.ndim(k_node_avg_ohm) == 0 and np.ndim(beta_n_rad_m) == 0 and np.ndim(f_hz) == 0:
        return float(np.asarray(val).item())
    return val


def load_cold_test_benchmarks() -> dict[str, Any]:
    """Load cold-test dispersion and Pierce impedances from Tutorial 6.2 data files."""
    d62 = tutorial_62_data_dir()
    spec_2d = KaretnikovaGratingSpec()
    spec_3d = MarosiGlideSWSSpec()
    u0_2d, gamma_2d = relativistic_beam_velocity(spec_2d.beam_voltage_v)
    u0_3d, gamma_3d = relativistic_beam_velocity(spec_3d.beam_voltage_v)
    d2 = spec_2d.period_m
    d3 = spec_3d.period_m

    # Determine the forward-wave control velocity u0_fwd from L2 mirror-mode dispersion at beta0*d = 0.75*pi
    z_l2_tmp = np.load(d62 / "task_06_2d_conformal_L2.npz")
    bn_l2_tmp = np.asarray(z_l2_tmp["bn"], dtype=float)
    f_mir_l2_tmp = np.asarray(z_l2_tmp["f_mirror_low_ghz"], dtype=float) * 1e9
    idx_075_l2 = int(np.argmin(np.abs(bn_l2_tmp - 0.75)))
    u0_fwd_ctrl = float(
        2.0 * np.pi * f_mir_l2_tmp[idx_075_l2] / (bn_l2_tmp[idx_075_l2] * np.pi / d2)
    )
    v0_fwd_ctrl = beam_voltage_from_velocity(u0_fwd_ctrl)
    y_fwd_ctrl_m = 50.0e-6
    L_fwd_ctrl_m = 10.0 * d2

    levels_2d: dict[str, Any] = {}
    for tag, fname, n_feat_val in [
        ("L1", "task_05_2d_conformal_L1.npz", 4),
        ("L2", "task_06_2d_conformal_L2.npz", 8),
        ("L3", "task_07_2d_conformal_L3.npz", 12),
        ("ref", "task_08_2d_conformal_reference.npz", 16),
    ]:
        z = np.load(d62 / fname)
        bn = np.asarray(z["bn"], dtype=float)
        f_glide_low_hz = np.asarray(
            z["f_glide_low_ghz"] if "f_glide_low_ghz" in z.files else z["f1_ghz"],
            dtype=float,
        ) * 1e9
        ve_over_c = np.asarray(z["v_energy_over_c"], dtype=float)
        km1_ax = np.asarray(z["km1_axis_ohm"], dtype=float)
        km1_av = np.asarray(z["km1_beam_avg_ohm"], dtype=float)
        kp1_ax = np.asarray(z["kp1_axis_ohm"], dtype=float)
        if "f_mirror_low_ghz" in z.files:
            f_mirror_low_hz = np.asarray(z["f_mirror_low_ghz"], dtype=float) * 1e9
            k0_mir_ax = np.asarray(z["k0_mirror_axis_ohm"], dtype=float)
            k0_mir_av = np.asarray(z["km1_mirror_avg_ohm"], dtype=float)
        else:
            f_mirror_low_hz = f_glide_low_hz.copy()
            k0_mir_ax = np.asarray(z["k0_axis_ohm"], dtype=float)
            k0_mir_av = np.asarray(z["km1_beam_avg_ohm"], dtype=float)

        dy_level_m = spec_2d.vane_thickness_m / float(n_feat_val)

        if len(bn) > 1:
            beta_m1_arr = (2.0 - bn) * np.pi / d2
            f_beam_m1_hz = beta_m1_arr * u0_2d / (2.0 * np.pi)
            diff_hz = f_glide_low_hz - f_beam_m1_hz
            idx_cross = int(np.where(np.diff(np.sign(diff_hz)) != 0)[0][0])
            w = float(
                -diff_hz[idx_cross] / (diff_hz[idx_cross + 1] - diff_hz[idx_cross])
            )
            bn_sync = float(bn[idx_cross] + w * (bn[idx_cross + 1] - bn[idx_cross]))
            f_bwd = float(
                f_glide_low_hz[idx_cross]
                + w * (f_glide_low_hz[idx_cross + 1] - f_glide_low_hz[idx_cross])
            )
            beta_m1_d_over_pi = 2.0 - bn_sync
            beta_m1 = beta_m1_d_over_pi * np.pi / d2

            df_dbn = float(
                (f_glide_low_hz[idx_cross + 1] - f_glide_low_hz[idx_cross])
                / (bn[idx_cross + 1] - bn[idx_cross])
            )
            vg_bwd = -2.0 * d2 * df_dbn
            ve_bwd = (
                -abs(
                    float(
                        ve_over_c[idx_cross]
                        + w * (ve_over_c[idx_cross + 1] - ve_over_c[idx_cross])
                    )
                )
                * c
            )
            k_axis = float(
                km1_ax[idx_cross] + w * (km1_ax[idx_cross + 1] - km1_ax[idx_cross])
            )
            k_beam = float(
                km1_av[idx_cross] + w * (km1_av[idx_cross + 1] - km1_av[idx_cross])
            )
            eta_sheet = sheet_beam_averaging_factor(
                beta_m1, f_bwd, spec_2d.beam_thickness_m
            )
            eta_sheet_nodes = sheet_beam_node_averaging_factor(
                beta_m1, f_bwd, spec_2d.beam_thickness_m, dy_level_m
            )

            idx_fwd = int(np.argmin(np.abs(bn - 0.50)))
            # Forward-wave mirror n = 0 control at beam velocity u0_fwd_ctrl (beta0*d = 0.75*pi on L2)
            beta0_mir_arr = bn * np.pi / d2
            f_beam_mir_hz = beta0_mir_arr * u0_fwd_ctrl / (2.0 * np.pi)
            diff_mir_hz = f_mirror_low_hz - f_beam_mir_hz
            sign_changes_mir = np.where(np.diff(np.sign(diff_mir_hz[1:])) != 0)[0] + 1
            if len(sign_changes_mir) > 0:
                idx_m = int(sign_changes_mir[0])
                w_m = float(
                    -diff_mir_hz[idx_m]
                    / (diff_mir_hz[idx_m + 1] - diff_mir_hz[idx_m])
                )
            else:
                idx_m = int(np.argmin(np.abs(bn - 0.75)))
                w_m = 0.0
            bn_fwd_mirror = float(bn[idx_m] + w_m * (bn[idx_m + 1] - bn[idx_m]))
            f_fwd_mirror = float(
                f_mirror_low_hz[idx_m]
                + w_m * (f_mirror_low_hz[idx_m + 1] - f_mirror_low_hz[idx_m])
            )
            beta_fwd_mirror = float(bn_fwd_mirror * np.pi / d2)
            u0_fwd_mirror = float(u0_fwd_ctrl)
            idx_m_near = int(np.argmin(np.abs(bn - bn_fwd_mirror)))
            if 0 < idx_m_near < len(bn) - 1:
                df_mir_dbn = float(
                    (f_mirror_low_hz[idx_m_near + 1] - f_mirror_low_hz[idx_m_near - 1])
                    / (bn[idx_m_near + 1] - bn[idx_m_near - 1])
                )
            else:
                df_mir_dbn = float(
                    (f_mirror_low_hz[idx_m + 1] - f_mirror_low_hz[idx_m])
                    / (bn[idx_m + 1] - bn[idx_m])
                )
            vg_fwd_mirror = 2.0 * d2 * df_mir_dbn
            k_fwd_mirror_axis = float(
                k0_mir_ax[idx_m] + w_m * (k0_mir_ax[idx_m + 1] - k0_mir_ax[idx_m])
            )
            k_fwd_mirror_node_avg = float(
                k0_mir_av[idx_m] + w_m * (k0_mir_av[idx_m + 1] - k0_mir_av[idx_m])
            )
            k_fwd_mirror_y50um = float(
                mirror_odd_harmonic_k_at_y_from_node_avg(
                    k_fwd_mirror_node_avg,
                    beta_fwd_mirror,
                    f_fwd_mirror,
                    y_target_m=y_fwd_ctrl_m,
                    beam_thickness_m=spec_2d.beam_thickness_m,
                    dy_m=dy_level_m,
                )
            )
            eta_fwd_mirror = sheet_beam_averaging_factor(
                beta_fwd_mirror, f_fwd_mirror, spec_2d.beam_thickness_m
            )

            # Full nonlinear mirror-odd passband product Re Z_||^cold(f) at y = 50 um for N = 10
            # Exclude bn = 1.0 (the beta_0 d = pi Bragg stopband edge where v_g -> 0 and K_0 diverges)
            valid_mir = (bn >= 0.15) & (bn <= 0.96)
            bn_v = bn[valid_mir]
            f_mir_v = f_mirror_low_hz[valid_mir]
            beta_mir_v = bn_v * np.pi / d2
            k0_y50_v = np.asarray(
                mirror_odd_harmonic_k_at_y_from_node_avg(
                    k0_mir_av[valid_mir],
                    beta_mir_v,
                    f_mir_v,
                    y_target_m=y_fwd_ctrl_m,
                    beam_thickness_m=spec_2d.beam_thickness_m,
                    dy_m=dy_level_m,
                ),
                dtype=float,
            )
            bn_fine = np.linspace(float(bn_v[0]), float(bn_v[-1]), 401)
            f_mir_fine_hz = np.interp(bn_fine, bn_v, f_mir_v)
            beta_mir_fine = bn_fine * np.pi / d2
            k0_y50_fine = np.interp(bn_fine, bn_v, k0_y50_v)
            delta_beta_fine = beta_mir_fine - (2.0 * np.pi * f_mir_fine_hz / u0_fwd_ctrl)
            sinc2_fine = np.sinc(0.5 * delta_beta_fine * L_fwd_ctrl_m / np.pi) ** 2
            re_z_nl_fine = (
                0.25 * (beta_mir_fine**2) * k0_y50_fine * (L_fwd_ctrl_m**2) * sinc2_fine
            )
            i_nl_pk = int(np.argmax(re_z_nl_fine))
            f_pk_nl_ghz = float(f_mir_fine_hz[i_nl_pk] * 1e-9)
            re_z_pk_nl_ohm = float(re_z_nl_fine[i_nl_pk])
            k0_at_fpk_nl_ohm = float(k0_y50_fine[i_nl_pk])
            half_nl = re_z_nl_fine >= 0.5 * re_z_pk_nl_ohm
            fwhm_nl_ghz = float(
                (f_mir_fine_hz[half_nl][-1] - f_mir_fine_hz[half_nl][0]) * 1e-9
            )
            f_stopband_ghz = float(f_mirror_low_hz[-1] * 1e-9)

            f_fwd_glide = float(f_glide_low_hz[idx_fwd])
            beta_fwd_glide = float((2.0 + bn[idx_fwd]) * np.pi / d2)
            u0_fwd_glide = 2.0 * np.pi * f_fwd_glide / beta_fwd_glide
            df_gld_dbn = float(
                (f_glide_low_hz[idx_fwd + 1] - f_glide_low_hz[idx_fwd - 1])
                / (bn[idx_fwd + 1] - bn[idx_fwd - 1])
            )
            vg_fwd_glide = 2.0 * d2 * df_gld_dbn
            k_fwd_glide_axis = float(kp1_ax[idx_fwd])
            eta_fwd_glide = sheet_beam_averaging_factor(
                beta_fwd_glide, f_fwd_glide, spec_2d.beam_thickness_m
            )
            k_fwd_glide_beam = k_fwd_glide_axis * eta_fwd_glide
        else:
            # Single mid-band convergence point bn = 0.50; scale synchronous values from L2
            l2_ref = levels_2d["L2"]
            f_ratio = float(f_glide_low_hz[0] / l2_ref["glide_fwd_p1"]["f_sync_hz"])
            k_ax_ratio = float(km1_ax[0] / l2_ref["km1_mid_axis_ohm"])
            k_av_ratio = float(km1_av[0] / l2_ref["km1_mid_beam_avg_ohm"])
            bn_sync = float(l2_ref["beta0_d_over_pi"])
            beta_m1_d_over_pi = float(l2_ref["beta_m1_d_over_pi"])
            beta_m1 = float(l2_ref["beta_m1_rad_m"])
            f_bwd = float(l2_ref["f_sync_bwd_hz"] * f_ratio)
            vg_bwd = float(l2_ref["vg_bwd_m_s"] * f_ratio)
            ve_bwd = float(-abs(float(ve_over_c[0])) * c)
            k_axis = float(l2_ref["k_m1_on_axis_ohm"] * k_ax_ratio)
            k_beam = float(l2_ref["k_m1_beam_avg_ohm"] * k_av_ratio)
            eta_sheet = sheet_beam_averaging_factor(
                beta_m1, f_bwd, spec_2d.beam_thickness_m
            )
            eta_sheet_nodes = sheet_beam_node_averaging_factor(
                beta_m1, f_bwd, spec_2d.beam_thickness_m, dy_level_m
            )
            idx_fwd = 0
            bn_fwd_mirror = float(l2_ref["mirror_fwd_n0"]["beta0_d_over_pi"])
            f_fwd_mirror = float(l2_ref["mirror_fwd_n0"]["f_sync_hz"] * f_ratio)
            beta_fwd_mirror = float(l2_ref["mirror_fwd_n0"]["beta_0_rad_m"])
            u0_fwd_mirror = float(l2_ref["mirror_fwd_n0"]["u0_m_s"])
            vg_fwd_mirror = float(l2_ref["mirror_fwd_n0"]["vg_m_s"] * f_ratio)
            k_fwd_mirror_axis = float(
                l2_ref["mirror_fwd_n0"]["k_0_on_axis_ohm"] * k_ax_ratio
            )
            k_fwd_mirror_node_avg = float(
                l2_ref["mirror_fwd_n0"]["k_0_beam_avg_ohm"] * k_av_ratio
            )
            k_fwd_mirror_y50um = float(
                l2_ref["mirror_fwd_n0"]["k_0_at_y50um_ohm"] * k_av_ratio
            )
            eta_fwd_mirror = float(l2_ref["mirror_fwd_n0"]["eta_sheet"])
            f_pk_nl_ghz = float(l2_ref["mirror_fwd_n0"]["f_pk_nonlinear_N10_ghz"] * f_ratio)
            re_z_pk_nl_ohm = float(
                l2_ref["mirror_fwd_n0"]["re_z_pk_nonlinear_N10_ohm"] * k_av_ratio
            )
            k0_at_fpk_nl_ohm = float(
                l2_ref["mirror_fwd_n0"]["k_0_at_fpk_nonlinear_y50um_ohm"] * k_av_ratio
            )
            fwhm_nl_ghz = float(l2_ref["mirror_fwd_n0"]["fwhm_nonlinear_N10_ghz"] * f_ratio)
            f_stopband_ghz = float(l2_ref["mirror_fwd_n0"]["f_upper_stopband_ghz"] * f_ratio)
            f_mir_fine_hz = np.array([f_fwd_mirror])
            k0_y50_fine = np.array([k_fwd_mirror_y50um])
            re_z_nl_fine = np.array([re_z_pk_nl_ohm])

            f_fwd_glide = float(f_glide_low_hz[0])
            beta_fwd_glide = float(l2_ref["glide_fwd_p1"]["beta_p1_rad_m"])
            u0_fwd_glide = 2.0 * np.pi * f_fwd_glide / beta_fwd_glide
            vg_fwd_glide = float(l2_ref["glide_fwd_p1"]["vg_m_s"] * f_ratio)
            k_fwd_glide_axis = float(kp1_ax[0])
            eta_fwd_glide = sheet_beam_averaging_factor(
                beta_fwd_glide, f_fwd_glide, spec_2d.beam_thickness_m
            )
            k_fwd_glide_beam = float(k_fwd_glide_axis * eta_fwd_glide)

        re_z_sync_fwd_y50_ohm = float(
            0.25 * (beta_fwd_mirror**2) * k_fwd_mirror_y50um * (L_fwd_ctrl_m**2)
        )
        fwhm_lin_fwd_ghz = float(
            analytic_fwhm_hz(
                length_m=L_fwd_ctrl_m, u0_m_s=u0_fwd_mirror, vg_m_s=vg_fwd_mirror
            )
            * 1e-9
        )

        levels_2d[tag] = {
            "n_feat": int(n_feat_val),
            "dy_m": float(dy_level_m),
            "f_sync_bwd_hz": f_bwd,
            "f_sync_bwd_ghz": f_bwd * 1e-9,
            "beta0_d_over_pi": bn_sync,
            "beta_m1_d_over_pi": beta_m1_d_over_pi,
            "beta_m1_rad_m": float(beta_m1),
            "vg_bwd_m_s": float(vg_bwd),
            "vg_bwd_over_c": float(vg_bwd / c),
            "ve_bwd_m_s": float(ve_bwd),
            "ve_bwd_over_c": float(ve_bwd / c),
            "k_m1_beam_avg_ohm": k_beam,
            "k_m1_on_axis_ohm": float(k_axis),
            "k_m1_ratio_beam_over_axis": float(k_beam / max(k_axis, 1e-12)),
            "km1_mid_axis_ohm": float(km1_ax[idx_fwd]),
            "km1_mid_beam_avg_ohm": float(km1_av[idx_fwd]),
            "eta_sheet": float(eta_sheet),
            "eta_sheet_nodes": float(eta_sheet_nodes),
            "z_peak_per_n2_on_axis_ohm": float(
                0.25 * (beta_m1**2) * k_axis * (d2**2)
            ),
            "z_peak_per_n2_beam_avg_ohm": float(
                0.25 * (beta_m1**2) * k_beam * (d2**2)
            ),
            "fwhm_times_n_ghz": float(
                analytic_fwhm_hz(
                    length_m=d2, u0_m_s=u0_2d, vg_m_s=vg_bwd
                )
                * 1e-9
            ),
            "bn_array": bn.tolist(),
            "f_glide_low_ghz": (f_glide_low_hz * 1e-9).tolist(),
            "f_mirror_low_ghz": (f_mirror_low_hz * 1e-9).tolist(),
            "km1_axis_ohm_array": km1_ax.tolist(),
            "km1_beam_avg_ohm_array": km1_av.tolist(),
            "k0_mirror_beam_avg_ohm_array": k0_mir_av.tolist(),
            "mirror_fwd_n0": {
                "beta0_d_over_pi": float(bn_fwd_mirror),
                "phase_deg": float(bn_fwd_mirror * 180.0),
                "f_sync_hz": f_fwd_mirror,
                "f_sync_ghz": f_fwd_mirror * 1e-9,
                "beta_0_rad_m": float(beta_fwd_mirror),
                "u0_m_s": float(u0_fwd_mirror),
                "u0_over_c": float(u0_fwd_mirror / c),
                "beam_voltage_v": float(v0_fwd_ctrl),
                "vg_m_s": float(vg_fwd_mirror),
                "vg_over_c": float(vg_fwd_mirror / c),
                "ysource_m": float(y_fwd_ctrl_m),
                "k_0_beam_avg_ohm": float(k_fwd_mirror_node_avg),
                "k_0_on_axis_ohm": float(k_fwd_mirror_axis),
                "k_0_at_y50um_ohm": float(k_fwd_mirror_y50um),
                "re_z_sync_N10_y50um_ohm": re_z_sync_fwd_y50_ohm,
                "fwhm_linear_N10_ghz": fwhm_lin_fwd_ghz,
                "f_pk_nonlinear_N10_ghz": f_pk_nl_ghz,
                "re_z_pk_nonlinear_N10_ohm": re_z_pk_nl_ohm,
                "k_0_at_fpk_nonlinear_y50um_ohm": k0_at_fpk_nl_ohm,
                "fwhm_nonlinear_N10_ghz": fwhm_nl_ghz,
                "f_upper_stopband_ghz": f_stopband_ghz,
                "f_passband_ghz": (f_mir_fine_hz * 1e-9).tolist(),
                "k0_y50um_passband_ohm": k0_y50_fine.tolist(),
                "re_z_nonlinear_N10_passband_ohm": re_z_nl_fine.tolist(),
                "eta_sheet": float(eta_fwd_mirror),
            },
            "glide_fwd_p1": {
                "phase_deg": float(bn[idx_fwd] * 180.0),
                "f_sync_hz": f_fwd_glide,
                "f_sync_ghz": f_fwd_glide * 1e-9,
                "beta_p1_rad_m": float(beta_fwd_glide),
                "u0_m_s": float(u0_fwd_glide),
                "u0_over_c": float(u0_fwd_glide / c),
                "vg_m_s": float(vg_fwd_glide),
                "vg_over_c": float(vg_fwd_glide / c),
                "k_p1_beam_avg_ohm": float(k_fwd_glide_beam),
                "k_p1_on_axis_ohm": float(k_fwd_glide_axis),
                "eta_sheet": float(eta_fwd_glide),
            },
        }

    z3d = np.load(d62 / "task_10_3d_marosi_glide.npz")
    bn3 = np.asarray(z3d["bn"], dtype=float)
    f_fwd_3d_ghz = np.asarray(z3d["f_fwd_ghz"], dtype=float)
    f_bwd_3d_ghz = np.asarray(z3d["f_bwd_ghz"], dtype=float)
    zp_fwd_ax = np.asarray(z3d["zp_fwd_axis_ohm"], dtype=float)
    zp_fwd_av = np.asarray(z3d["zp_fwd_avg_ohm"], dtype=float)
    zp_bwd_ax = np.asarray(z3d["zp_bwd_axis_ohm"], dtype=float)
    zp_bwd_av = np.asarray(z3d["zp_bwd_avg_ohm"], dtype=float)
    ve_fwd_3d = np.asarray(z3d["ve_fwd_over_c"], dtype=float)
    ve_bwd_3d = np.asarray(z3d["ve_bwd_over_c"], dtype=float)

    beta_low_pi = 2.0 + bn3[:-1]
    beta_high_pi = 4.0 - bn3[:-1][::-1]
    beta_unfolded_pi = np.concatenate([beta_low_pi, [3.0], beta_high_pi])
    f_unfolded_ghz = np.concatenate(
        [f_fwd_3d_ghz[:-1], [f_fwd_3d_ghz[-1]], f_bwd_3d_ghz[:-1][::-1]]
    )
    k_ax_unfolded = np.concatenate(
        [
            zp_fwd_ax[:-1],
            [0.5 * (zp_fwd_ax[-2] + zp_bwd_ax[-2])],
            zp_bwd_ax[:-1][::-1],
        ]
    )
    k_av_unfolded = np.concatenate(
        [
            zp_fwd_av[:-1],
            [0.5 * (zp_fwd_av[-2] + zp_bwd_av[-2])],
            zp_bwd_av[:-1][::-1],
        ]
    )
    ve_unfolded_over_c = np.concatenate(
        [
            ve_fwd_3d[:-1],
            [0.5 * (ve_fwd_3d[-2] - ve_bwd_3d[-2])],
            -ve_bwd_3d[:-1][::-1],
        ]
    )

    beta_unfolded_rad_m = beta_unfolded_pi * np.pi / d3
    f_beam_3d_ghz = (beta_unfolded_rad_m * u0_3d / (2.0 * np.pi)) * 1e-9
    diff_3d = f_unfolded_ghz - f_beam_3d_ghz
    idx_3d = int(np.where(np.diff(np.sign(diff_3d)) != 0)[0][0])
    w3 = float(-diff_3d[idx_3d] / (diff_3d[idx_3d + 1] - diff_3d[idx_3d]))
    beta_d_over_pi_3d = float(
        beta_unfolded_pi[idx_3d]
        + w3 * (beta_unfolded_pi[idx_3d + 1] - beta_unfolded_pi[idx_3d])
    )
    f_sync_3d_ghz = float(
        f_unfolded_ghz[idx_3d]
        + w3 * (f_unfolded_ghz[idx_3d + 1] - f_unfolded_ghz[idx_3d])
    )
    f_sync_3d = f_sync_3d_ghz * 1e9
    beta_n_3d = beta_d_over_pi_3d * np.pi / d3
    df_dbpi_3d = float(
        (f_unfolded_ghz[idx_3d + 1] - f_unfolded_ghz[idx_3d])
        * 1e9
        / (beta_unfolded_pi[idx_3d + 1] - beta_unfolded_pi[idx_3d])
    )
    vg_3d = 2.0 * d3 * df_dbpi_3d
    ve_3d = (
        float(
            ve_unfolded_over_c[idx_3d]
            + w3 * (ve_unfolded_over_c[idx_3d + 1] - ve_unfolded_over_c[idx_3d])
        )
        * c
    )
    k_axis_3d = float(
        k_ax_unfolded[idx_3d]
        + w3 * (k_ax_unfolded[idx_3d + 1] - k_ax_unfolded[idx_3d])
    )
    k_beam_3d = float(
        k_av_unfolded[idx_3d]
        + w3 * (k_av_unfolded[idx_3d + 1] - k_av_unfolded[idx_3d])
    )

    with open(d62 / "graded_mesh_comparison.json", encoding="utf-8") as fh:
        graded_62 = json.load(fh)

    return {
        "spec_2d": asdict(spec_2d),
        "u0_2d_m_s": u0_2d,
        "u0_2d_over_c": u0_2d / c,
        "gamma_2d": gamma_2d,
        "u0_fwd_ctrl_m_s": u0_fwd_ctrl,
        "u0_fwd_ctrl_over_c": u0_fwd_ctrl / c,
        "v0_fwd_ctrl_v": v0_fwd_ctrl,
        "levels_2d": levels_2d,
        "graded_62": graded_62,
        "spec_3d": asdict(spec_3d),
        "u0_3d_m_s": u0_3d,
        "u0_3d_over_c": u0_3d / c,
        "gamma_3d": gamma_3d,
        "marosi_3d": {
            "f_sync_hz": f_sync_3d,
            "f_sync_ghz": f_sync_3d_ghz,
            "beta_n_d_over_pi": beta_d_over_pi_3d,
            "beta_n_rad_m": float(beta_n_3d),
            "vg_m_s": float(vg_3d),
            "vg_over_c": float(vg_3d / c),
            "ve_m_s": float(ve_3d),
            "ve_over_c": float(ve_3d / c),
            "k_n_beam_avg_ohm": k_beam_3d,
            "k_n_on_axis_ohm": k_axis_3d,
            "eta_sheet_3d": float(k_beam_3d / max(k_axis_3d, 1e-12)),
            "beta_unfolded_d_over_pi": beta_unfolded_pi.tolist(),
            "f_unfolded_ghz": f_unfolded_ghz.tolist(),
            "k_on_axis_unfolded_ohm": k_ax_unfolded.tolist(),
            "k_beam_avg_unfolded_ohm": k_av_unfolded.tolist(),
            "ve_unfolded_over_c": ve_unfolded_over_c.tolist(),
        },
    }


def analytic_sws_beam_impedance(
    f_hz: np.ndarray | float,
    *,
    f_sync_hz: float,
    beta_n_rad_m: float,
    k_pierce_ohm: float,
    length_m: float,
    u0_m_s: float,
    vg_m_s: float,
) -> np.ndarray:
    """Return the analytical synchronous beam-wave resistance Re Z_||(f) [Ohm] (derived here).

    Re Z_||(omega) = (1 / 4) beta_n^2 K_n L^2 sinc^2(Delta_beta L / 2),
    Delta_beta(omega) = (omega - omega_s) (1/u_0 - 1/v_g).
    """
    f_arr = np.asarray(f_hz, dtype=float)
    domega = 2.0 * np.pi * (f_arr - float(f_sync_hz))
    slowness_diff = (1.0 / float(u0_m_s)) - (1.0 / float(vg_m_s))
    delta_beta = domega * slowness_diff
    half_phase = 0.5 * delta_beta * float(length_m)
    sinc_val = np.sinc(half_phase / np.pi)
    return (
        0.25
        * (float(beta_n_rad_m) ** 2)
        * float(k_pierce_ohm)
        * (float(length_m) ** 2)
        * (sinc_val**2)
    )


def analytic_fwhm_hz(
    *,
    length_m: float,
    u0_m_s: float,
    vg_m_s: float,
) -> float:
    """Return the full-width at half-maximum Delta f_FWHM [Hz] of sinc^2(Delta_beta L / 2) (derived here)."""
    slowness_walkoff = abs((1.0 / float(u0_m_s)) - (1.0 / float(vg_m_s)))
    if slowness_walkoff <= 0.0 or length_m <= 0.0:
        raise ValueError("Length and group/beam slowness mismatch must be positive.")
    return float(2.0 * SINC2_HALF_MAX_X / (np.pi * float(length_m) * slowness_walkoff))


def extract_pierce_k_from_peak_z(
    re_z_peak_ohm: float,
    *,
    beta_n_rad_m: float,
    length_m: float,
) -> float:
    """Invert Re Z_||(omega_s) = (1/4) beta_n^2 K_n L^2 to obtain K_n [Ohm] (derived here)."""
    return float(
        4.0
        * float(re_z_peak_ohm)
        / ((float(beta_n_rad_m) ** 2) * (float(length_m) ** 2))
    )


def gaussian_bunch_form_factor_sq(
    f_hz: np.ndarray | float,
    sigmaz_m: np.ndarray | float,
    u0_m_s: float,
) -> np.ndarray | float:
    """Return the Gaussian bunch power form factor |F(omega)|^2 = exp(-(omega sigma_z / u_0)^2)."""
    omega = 2.0 * np.pi * np.asarray(f_hz, dtype=float)
    sigma_t = np.asarray(sigmaz_m, dtype=float) / float(u0_m_s)
    val = np.exp(-((omega * sigma_t) ** 2))
    return float(val) if (np.ndim(f_hz) == 0 and np.ndim(sigmaz_m) == 0) else val


def compute_loss_factor_v_per_pc(
    f_hz: np.ndarray,
    re_z_ohm: np.ndarray,
    sigmaz_m: float,
    u0_m_s: float,
) -> float:
    """Integrate the longitudinal loss factor k_||(sigma_z) = (1/pi) int_0^inf Re Z_||(omega) |F(omega)|^2 d_omega [V/pC]."""
    f_arr = np.asarray(f_hz, dtype=float)
    z_arr = np.asarray(re_z_ohm, dtype=float)
    ff_sq = gaussian_bunch_form_factor_sq(f_arr, sigmaz_m, u0_m_s)
    k_v_per_c = 2.0 * float(np.trapezoid(z_arr * ff_sq, f_arr))
    return float(k_v_per_c * 1e-12)


def analytic_stopping_power_j_per_m(
    *,
    q_coulomb: float,
    beta_n_rad_m: float,
    k_pierce_ohm: float,
    u0_m_s: float,
    vg_m_s: float,
    form_factor_sq: float = 1.0,
) -> float:
    """Return the asymptotic (L -> inf) coherent energy loss per unit length dW/dz [J/m] (derived here)."""
    slowness_walkoff = abs((1.0 / float(u0_m_s)) - (1.0 / float(vg_m_s)))
    return float(
        0.5
        * (float(q_coulomb) ** 2)
        * (float(beta_n_rad_m) ** 2)
        * float(k_pierce_ohm)
        * float(form_factor_sq)
        / slowness_walkoff
    )


def analytic_detuned_frequency_shift_hz(
    *,
    f_sync_hz: float,
    u0_m_s: float,
    delta_u0_m_s: float,
    vg_m_s: float,
) -> float:
    """Return the linearised synchronous frequency shift Delta f_s [Hz] for beam velocity shift Delta u_0 (derived here)."""
    denom = float(u0_m_s) ** 2 * ((1.0 / float(u0_m_s)) - (1.0 / float(vg_m_s)))
    return float(float(f_sync_hz) * float(delta_u0_m_s) / denom)


def analytic_standing_wave_r_over_q_ohm(
    *,
    beta_n_rad_m: float,
    k_pierce_ohm: float,
    length_m: float,
    f_mode_hz: float,
    vg_m_s: float,
    convention: str = "circuit",
) -> float:
    """Return the standing-wave cavity (R/Q) [Ohm] of an N-period shorted SWS resonator (derived here)."""
    omega_m = 2.0 * np.pi * float(f_mode_hz)
    rq_circuit = (
        0.5
        * (float(beta_n_rad_m) ** 2)
        * float(k_pierce_ohm)
        * float(length_m)
        * abs(float(vg_m_s))
        / omega_m
    )
    if convention == "circuit":
        return float(rq_circuit)
    if convention == "linac":
        return float(2.0 * rq_circuit)
    raise ValueError(f"Unknown R/Q convention: {convention!r}")


def reflection_induced_peak_error_bound(
    gamma_mag: np.ndarray | float,
) -> np.ndarray | float:
    """Return the first-order relative error bound 2 |Gamma(omega_s)| / (1 - |Gamma|^2) on Re Z_||(omega_s) (derived here)."""
    g = np.abs(np.asarray(gamma_mag, dtype=float))
    val = 2.0 * g / np.maximum(1.0 - g * g, 1e-12)
    return float(val) if np.ndim(gamma_mag) == 0 else val


def build_2d_karetnikova_stl(
    spec: KaretnikovaGratingSpec,
    *,
    n_periods: int,
    z_buffer_m: float,
    taper_scales: list[float],
    symmetry: str = "glide",
    x_span_m: float = 2.0e-3,
    stl_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build a closed PyVista STL of an N-period 2D Karetnikova grating with optional end tapers."""
    d = spec.period_m
    l_vane = spec.vane_thickness_m
    h_vane = spec.vane_height_m
    a_tun = spec.half_tunnel_m
    y_wall = a_tun + h_vane
    n_tap = len(taper_scales)
    total_periods = int(n_periods) + 2 * n_tap
    z_total = 2.0 * float(z_buffer_m) + total_periods * d

    eps = 0.05e-3
    blocks: list[pv.PolyData] = []
    for k in range(total_periods):
        if k < n_tap:
            scale = float(taper_scales[k])
        elif k >= n_tap + int(n_periods):
            scale = float(taper_scales[total_periods - 1 - k])
        else:
            scale = 1.0
        hk = scale * h_vane
        if hk <= 1e-9:
            continue
        z0 = float(z_buffer_m) + k * d
        z_bot = z0 + 0.25 * d
        z_top = (z0 + 0.75 * d) if symmetry == "glide" else (z0 + 0.25 * d)

        b_bot = pv.Box(
            bounds=(
                -float(x_span_m),
                float(x_span_m),
                -y_wall - eps,
                -y_wall + hk,
                z_bot - 0.5 * l_vane,
                z_bot + 0.5 * l_vane,
            )
        ).triangulate()
        b_top = pv.Box(
            bounds=(
                -float(x_span_m),
                float(x_span_m),
                y_wall - hk,
                y_wall + eps,
                z_top - 0.5 * l_vane,
                z_top + 0.5 * l_vane,
            )
        ).triangulate()
        blocks.extend([b_bot, b_top])

    merged = blocks[0]
    for blk in blocks[1:]:
        merged = merged.merge(blk)

    stl_path = _local_tmp_stl_path("wakis_2d_sws_") if stl_path is None else Path(stl_path)
    merged.save(str(stl_path))

    z_active_start = float(z_buffer_m) + n_tap * d
    z_active_end = z_active_start + int(n_periods) * d
    return {
        "stl_path": str(stl_path),
        "z_total_m": float(z_total),
        "z_active_start_m": float(z_active_start),
        "z_active_end_m": float(z_active_end),
        "active_length_m": float(int(n_periods) * d),
        "total_periods": int(total_periods),
        "n_taper_each_side": int(n_tap),
    }


def build_2d_karetnikova_axes(
    spec: KaretnikovaGratingSpec,
    *,
    z_total_m: float,
    n_feat: int = 4,
    use_graded_mesh: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 3D FIT axes (1 cell in x, graded or uniform in y and z) for the 2D grating."""
    l_vane = spec.vane_thickness_m
    a_tun = spec.half_tunnel_m
    y_wall = a_tun + spec.vane_height_m

    x = np.linspace(-0.5 * spec.transverse_width_m, 0.5 * spec.transverse_width_m, 2)
    dy_fine = l_vane / float(n_feat)
    dz_fine = l_vane / float(n_feat)

    if use_graded_mesh:
        n_tun_half = max(int(round(a_tun / dy_fine)), 4)
        n_grv = max(int(round(spec.vane_height_m / (1.5 * dy_fine))), 4)
        y_bot = np.linspace(-y_wall, -a_tun, n_grv + 1)[:-1]
        y_mid = np.linspace(-a_tun, a_tun, 2 * n_tun_half + 1)[:-1]
        y_top = np.linspace(a_tun, y_wall, n_grv + 1)
        y = np.concatenate([y_bot, y_mid, y_top])
    else:
        ny = int(round(2.0 * y_wall / dy_fine))
        y = np.linspace(-y_wall, y_wall, ny + 1)

    nz = int(round(float(z_total_m) / dz_fine))
    z = np.linspace(0.0, float(z_total_m), nz + 1)
    return x, y, z


def build_3d_marosi_stl(
    spec: MarosiGlideSWSSpec,
    *,
    n_periods: int,
    z_buffer_m: float,
    taper_scales: list[float],
    stl_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build a closed PyVista STL of the 3D Marosi V-band glide-symmetric pillar SWS."""
    a = spec.waveguide_width_m
    d = spec.period_m
    w_pil = spec.pillar_width_m
    l_pil = spec.pillar_length_m
    t_pil = spec.pillar_thickness_m
    y_in = 0.5 * spec.beam_thickness_m + spec.pillar_beam_clearance_m
    y_out = y_in + t_pil

    n_tap = len(taper_scales)
    total_periods = int(n_periods) + 2 * n_tap
    z_total = 2.0 * float(z_buffer_m) + total_periods * d
    eps = 0.05e-3

    blocks: list[pv.PolyData] = []
    for k in range(total_periods):
        if k < n_tap:
            scale = float(taper_scales[k])
        elif k >= n_tap + int(n_periods):
            scale = float(taper_scales[total_periods - 1 - k])
        else:
            scale = 1.0
        wk = scale * w_pil
        if wk <= 1e-9:
            continue
        z0 = float(z_buffer_m) + k * d
        z1 = z0 + 0.25 * d
        z2 = z0 + 0.75 * d
        for y_lo, y_hi in [(y_in, y_out), (-y_out, -y_in)]:
            b1 = pv.Box(
                bounds=(
                    -0.5 * a - eps,
                    -0.5 * a + wk,
                    y_lo,
                    y_hi,
                    z1 - 0.5 * l_pil,
                    z1 + 0.5 * l_pil,
                )
            ).triangulate()
            b2 = pv.Box(
                bounds=(
                    0.5 * a - wk,
                    0.5 * a + eps,
                    y_lo,
                    y_hi,
                    z2 - 0.5 * l_pil,
                    z2 + 0.5 * l_pil,
                )
            ).triangulate()
            blocks.extend([b1, b2])

    merged = blocks[0]
    for blk in blocks[1:]:
        merged = merged.merge(blk)

    stl_path = _local_tmp_stl_path("wakis_3d_marosi_") if stl_path is None else Path(stl_path)
    merged.save(str(stl_path))

    z_active_start = float(z_buffer_m) + n_tap * d
    z_active_end = z_active_start + int(n_periods) * d
    return {
        "stl_path": str(stl_path),
        "z_total_m": float(z_total),
        "z_active_start_m": float(z_active_start),
        "z_active_end_m": float(z_active_end),
        "active_length_m": float(int(n_periods) * d),
        "total_periods": int(total_periods),
        "n_taper_each_side": int(n_tap),
    }


def measure_sws_termination_reflection(
    spec: KaretnikovaGratingSpec | None = None,
    *,
    n_feat: int = 4,
    n_periods_ref: int = 24,
    n_periods_dut: int = 10,
    taper_scales: list[float] | None = None,
    f_min_ghz: float = 95.0,
    f_max_ghz: float = 135.0,
    quick: bool = False,
) -> dict[str, Any]:
    """Measure SWS port reflection |Gamma(f)| via time-domain wave-packet reflectometry in FIT.

    Uses the exact production 2-stage vane-height taper ``taper_scales = [0.25, 0.55]``
    (which ramps down from the active grating height ``h`` to the smooth waveguide as
    ``[0.55*h, 0.25*h]``, identical to ``build_2d_karetnikova_stl``) and normalizes the
    reflected wave spectrum directly by the PEC-shorted reference ``|S_pec(f)|`` with
    no floor and no clipping.
    """
    if spec is None:
        spec = KaretnikovaGratingSpec()
    if taper_scales is None:
        taper_scales = [0.25, 0.55]
    if quick:
        n_periods_ref = min(int(n_periods_ref), 16)
        n_periods_dut = min(int(n_periods_dut), 6)
    d = spec.period_m
    z_buf = 1.5 * d

    ref_geom = build_2d_karetnikova_stl(
        spec,
        n_periods=n_periods_ref,
        z_buffer_m=z_buf,
        taper_scales=[],
        symmetry="glide",
    )

    bench = load_cold_test_benchmarks()
    level_key = "L2" if int(n_feat) >= 8 else "L1"
    f_sync_target_ghz = float(bench["levels_2d"][level_key]["f_sync_bwd_ghz"])
    f0 = f_sync_target_ghz * 1e9
    bw = 35.0e9
    sigma_t = 1.0 / (np.pi * bw)
    t0_pulse = 4.5 * sigma_t

    def _run_trace(
        stl_file: str,
        z_tot: float,
        bc_high_z: str = "cpml",
        t_max: float = 155e-12,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        t_start = time.perf_counter()
        x_ax, y_ax, z_ax = build_2d_karetnikova_axes(
            spec, z_total_m=z_tot, n_feat=n_feat, use_graded_mesh=quick
        )
        grid = GridFIT3D(
            x=x_ax,
            y=y_ax,
            z=z_ax,
            stl_solids={"sws": stl_file},
            stl_materials={"sws": "pec"},
            conformal="pec",
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            use_stl=True,
            bc_low=["periodic", "pec", "cpml"],
            bc_high=["periodic", "pec", bc_high_z],
            n_pml=10,
            verbose=0,
        )
        est_tr = estimate(
            (solver.Nx, solver.Ny, solver.Nz),
            steps=1000,
            cpml=(bc_high_z == "cpml"),
            conformal="pec",
        )
        preflight(est_tr, "laptop")
        nt = int(np.ceil(t_max / solver.dt))
        ix0 = 0
        iy0 = int(np.argmin(np.abs(solver.grid.y - 0.0)))
        z_src = z_buf + (2.0 if quick else 3.0) * d
        z_prb = z_buf + (4.0 if quick else 6.0) * d
        iz_src = int(np.argmin(np.abs(solver.z - z_src)))
        iz_prb = int(np.argmin(np.abs(solver.z - z_prb)))

        t_arr = np.arange(nt) * solver.dt
        ez_prb = np.zeros(nt)
        for n in range(nt):
            t = t_arr[n]
            env = np.exp(-0.5 * ((t - t0_pulse) / sigma_t) ** 2) * np.sin(
                2.0 * np.pi * f0 * (t - t0_pulse)
            )
            solver.J[ix0, iy0, iz_src, "z"] = 1e6 * env
            solver.one_step()
            ez_prb[n] = solver.E[ix0, iy0, iz_prb, "z"]
        return t_arr, ez_prb, time.perf_counter() - t_start

    def _build_exit_only_stl(
        entrance_taper_scales: list[float], z_tail_m: float
    ) -> tuple[str, float]:
        l_vane = spec.vane_thickness_m
        h_vane = spec.vane_height_m
        a_tun = spec.half_tunnel_m
        y_wall = a_tun + h_vane
        eps = 0.05e-3
        n_tap = len(entrance_taper_scales)
        n_tot = n_periods_dut + n_tap
        z_tot = z_buf + n_tot * d + z_tail_m
        blocks = []
        for k in range(n_tot):
            # Match build_2d_karetnikova_stl exit ramp-down: scale = taper_scales[n_tot - 1 - k]
            scale = (
                1.0
                if k < n_periods_dut
                else float(entrance_taper_scales[n_tot - 1 - k])
            )
            hk = scale * h_vane
            z0 = z_buf + k * d
            z_bot = z0 + 0.25 * d
            z_top = z0 + 0.75 * d
            blocks.append(
                pv.Box(
                    bounds=(
                        -2e-3,
                        2e-3,
                        -y_wall - eps,
                        -y_wall + hk,
                        z_bot - 0.5 * l_vane,
                        z_bot + 0.5 * l_vane,
                    )
                ).triangulate()
            )
            blocks.append(
                pv.Box(
                    bounds=(
                        -2e-3,
                        2e-3,
                        y_wall - hk,
                        y_wall + eps,
                        z_top - 0.5 * l_vane,
                        z_top + 0.5 * l_vane,
                    )
                ).triangulate()
            )
        merged = blocks[0]
        for blk in blocks[1:]:
            merged = merged.merge(blk)
        p_path = _local_tmp_stl_path("wakis_exit_term_")
        merged.save(str(p_path))
        return str(p_path), z_tot

    t_max_sim = 115e-12 if quick else 155e-12
    t_arr, ez_ref, wall_ref = _run_trace(
        ref_geom["stl_path"], ref_geom["z_total_m"], "cpml", t_max_sim
    )
    stl_tap_exit, z_tot_tap = _build_exit_only_stl(list(taper_scales), 2.0 * d)
    stl_abr_exit, z_tot_abr = _build_exit_only_stl([], 2.0 * d)
    stl_pec_exit, z_tot_pec = _build_exit_only_stl([], 0.0)

    t_tap, ez_tap_raw, wall_tap = _run_trace(stl_tap_exit, z_tot_tap, "cpml", t_max_sim)
    t_abr, ez_abr_raw, wall_abr = _run_trace(stl_abr_exit, z_tot_abr, "cpml", t_max_sim)
    t_pec, ez_pec_raw, wall_pec = _run_trace(stl_pec_exit, z_tot_pec, "pec", t_max_sim)

    ez_tap = np.interp(t_arr, t_tap, ez_tap_raw)
    ez_abr = np.interp(t_arr, t_abr, ez_abr_raw)
    ez_pec = np.interp(t_arr, t_pec, ez_pec_raw)

    for p in [ref_geom["stl_path"], stl_tap_exit, stl_abr_exit, stl_pec_exit]:
        Path(p).unlink(missing_ok=True)

    dt = float(t_arr[1] - t_arr[0])
    t_inc_gate = 52e-12 if quick else 68e-12
    t_refl_gate = 102e-12 if quick else 135e-12
    win_inc = np.where(t_arr < t_inc_gate, 1.0, 0.0)
    ez_inc = ez_ref * win_inc
    ez_refl_tap = (ez_tap - ez_ref) * np.where(t_arr < t_refl_gate, 1.0, 0.0)
    ez_refl_abr = (ez_abr - ez_ref) * np.where(t_arr < t_refl_gate, 1.0, 0.0)
    ez_refl_pec = (ez_pec - ez_ref) * np.where(t_arr < t_refl_gate, 1.0, 0.0)

    n_fft = 16384
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    f_ghz = freqs * 1e-9
    mask = (f_ghz >= f_min_ghz) & (f_ghz <= f_max_ghz)
    f_band = f_ghz[mask]

    spec_inc = np.fft.rfft(ez_inc, n=n_fft)[mask]
    spec_pec = np.fft.rfft(ez_refl_pec, n=n_fft)[mask]
    spec_tap = np.fft.rfft(ez_refl_tap, n=n_fft)[mask]
    spec_abr = np.fft.rfft(ez_refl_abr, n=n_fft)[mask]

    # Direct normalization by the PEC-shorted reference spectrum with no floor and no clip
    denom = np.abs(spec_pec)
    if np.any(denom <= 0.0):
        raise RuntimeError("Zero spectral amplitude in PEC reference within passband.")
    gamma_tap = np.abs(spec_tap) / denom
    gamma_abr = np.abs(spec_abr) / denom
    gamma_pec = np.ones_like(f_band)

    idx_sync = int(np.argmin(np.abs(f_band - f_sync_target_ghz)))
    g_tap_sync = float(gamma_tap[idx_sync])
    g_abr_sync = float(gamma_abr[idx_sync])

    return {
        "taper_scales": [float(s) for s in taper_scales],
        "exit_ramp_down_scales": [float(s) for s in reversed(taper_scales)],
        "n_periods_ref": int(n_periods_ref),
        "n_periods_dut": int(n_periods_dut),
        "n_feat": int(n_feat),
        "f_band_ghz": [float(f_min_ghz), float(f_max_ghz)],
        "normalization": "|S_refl(f)| / |S_pec_short(f)| (no floor, no clip)",
        "min_spec_pec_over_max_inc": float(
            np.min(denom) / max(float(np.max(np.abs(spec_inc))), 1e-30)
        ),
        "f_ghz": f_band.tolist(),
        "gamma_tapered": gamma_tap.tolist(),
        "gamma_abrupt": gamma_abr.tolist(),
        "gamma_pec_short": gamma_pec.tolist(),
        "f_sync_target_cold_ghz": f_sync_target_ghz,
        "f_sync_ghz": float(f_band[idx_sync]),
        "gamma_tapered_at_sync": g_tap_sync,
        "gamma_abrupt_at_sync": g_abr_sync,
        "gamma_tapered_passband_mean": float(np.mean(gamma_tap)),
        "gamma_abrupt_passband_mean": float(np.mean(gamma_abr)),
        "gamma_pec_at_sync": 1.0,
        "reduction_factor_abrupt_over_tapered": float(
            g_abr_sync / max(g_tap_sync, 1e-12)
        ),
        "rel_error_bound_tapered": float(reflection_induced_peak_error_bound(g_tap_sync)),
        "rel_error_bound_abrupt": float(reflection_induced_peak_error_bound(g_abr_sync)),
        "wall_time_s": float(wall_ref + wall_tap + wall_abr + wall_pec),
    }


def _integrate_synchronous_wake_potential(
    t_arr: np.ndarray,
    z_int: np.ndarray,
    dz_int: np.ndarray,
    ez_int: np.ndarray,
    *,
    z_min_m: float,
    u0_m_s: float,
    ti_s: float,
    wake_len_m: float,
    dt_s: float,
    q_coulomb: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Integrate the scattered longitudinal field E_z(z, t) along the synchronous trajectory t = (z - z_min + s)/u_0 + t_i - 0.5*dt."""
    ds = float(u0_m_s) * float(dt_s)
    ti_loc = float(ti_s) + float(z_int[0]) / float(u0_m_s)
    s = np.arange(-ti_loc * float(u0_m_s), float(wake_len_m), ds)
    wp = np.zeros_like(s)
    for k in range(len(z_int)):
        ts = (
            (float(z_int[k]) - float(z_min_m) + s) / float(u0_m_s)
            + float(ti_s)
            - 0.5 * float(dt_s)
        )
        wp += np.interp(ts, t_arr, ez_int[:, k], left=0.0, right=0.0) * float(dz_int[k])
    wp /= float(q_coulomb) * 1e12
    return s, wp, ti_loc


def run_2d_karetnikova_beam_wake(
    spec: KaretnikovaGratingSpec | None = None,
    *,
    n_periods: int = 10,
    n_feat: int = 4,
    use_graded_mesh: bool = False,
    conformal_mode: str | bool = "pec",
    termination_mode: str = "tapered",
    symmetry: str = "glide",
    u0_m_s: float | None = None,
    ysource_m: float = 0.0,
    sigmaz_m: float = 0.22e-3,
    q_coulomb: float = 1.0e-12,
    f_max_hz: float = 200.0e9,
    fft_samples: int = 16384,
    f_search_band_ghz: tuple[float, float] = (95.0, 132.0),
    f_sync_target_ghz: float | None = None,
    record_poynting_and_snapshots: bool = False,
    extra_wake_len_m: float = 0.0,
    quick: bool = False,
) -> dict[str, Any]:
    """Run a 2D Karetnikova grating SWS beam-wake simulation in Wakis FIT and extract Re Z_||(f)."""
    if spec is None:
        spec = KaretnikovaGratingSpec()
    if u0_m_s is None:
        u0_m_s, _ = relativistic_beam_velocity(spec.beam_voltage_v)
    beta = float(u0_m_s / c)
    d = spec.period_m

    if termination_mode == "tapered":
        taper_scales = [0.25, 0.55]
        z_buf = 1.5 * d
        bc_z = "cpml"
    elif termination_mode == "abrupt":
        taper_scales = []
        z_buf = 1.5 * d
        bc_z = "cpml"
    elif termination_mode == "pec_cavity":
        taper_scales = []
        z_buf = 0.0
        bc_z = "pec"
    else:
        raise ValueError(f"Unknown termination_mode: {termination_mode!r}")

    geom = build_2d_karetnikova_stl(
        spec,
        n_periods=n_periods,
        z_buffer_m=z_buf,
        taper_scales=taper_scales,
        symmetry=symmetry,
    )
    z_tot = geom["z_total_m"]
    L_act = geom["active_length_m"]

    x, y, z = build_2d_karetnikova_axes(
        spec,
        z_total_m=z_tot,
        n_feat=n_feat,
        use_graded_mesh=use_graded_mesh,
    )

    t0_wall = time.perf_counter()
    grid_sws = GridFIT3D(
        x=x,
        y=y,
        z=z,
        stl_solids={"sws": geom["stl_path"]},
        stl_materials={"sws": "pec"},
        conformal=conformal_mode,
        verbose=0,
    )
    n_pml = 10 if bc_z == "cpml" else 8
    solver_sws = SolverFIT3D(
        grid_sws,
        use_stl=True,
        bc_low=["periodic", "pec", bc_z],
        bc_high=["periodic", "pec", bc_z],
        n_pml=n_pml,
        verbose=0,
    )
    solver_sws.source_type = "direct"
    est = estimate(
        (solver_sws.Nx, solver_sws.Ny, solver_sws.Nz),
        steps=1000,
        cpml=(bc_z == "cpml"),
        conformal=(conformal_mode if conformal_mode else False),
    )
    preflight(est, "laptop")

    grid_ref = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver_ref = SolverFIT3D(
        grid_ref,
        use_stl=False,
        bc_low=["periodic", "pec", "cpml"],
        bc_high=["periodic", "pec", "cpml"],
        n_pml=10,
        verbose=0,
    )
    solver_ref.source_type = "direct"

    ti = 5.0 * float(sigmaz_m) / float(u0_m_s)
    beam_sws = Beam(
        xsource=0.0,
        ysource=float(ysource_m),
        beta=beta,
        q=q_coulomb,
        sigmaz=sigmaz_m,
        ti=ti,
    )
    beam_ref = Beam(
        xsource=0.0,
        ysource=float(ysource_m),
        beta=beta,
        q=q_coulomb,
        sigmaz=sigmaz_m,
        ti=ti,
    )

    if termination_mode == "pec_cavity":
        wake_len = max(3.6 * L_act, 14.0e-3)
    elif quick:
        wake_len = 1.25 * L_act + 2.0e-3
    else:
        wake_len = 1.75 * L_act + 3.5e-3
    wake_len = float(wake_len)

    # For forward-wave Poynting energy balance (u0 > v_g), extend the time-stepping duration
    # so the slower wave packet completely exits the grating and smooth port into the +z CPML,
    # while keeping `wake_len` for the single-pass active-grating wake integration.
    if record_poynting_and_snapshots and float(u0_m_s) > 0.40 * c and extra_wake_len_m <= 0.0:
        extra_wake_len_m = 4.0e-3 if quick else 12.0e-3
    wake_len_sim = float(wake_len + float(extra_wake_len_m))

    t_sim = ti + (z_tot + wake_len_sim) / float(u0_m_s)
    nt = int(np.ceil(t_sim / solver_sws.dt))

    ix0 = 0
    iy0 = int(np.argmin(np.abs(solver_sws.grid.y - float(ysource_m))))
    ez_sws = np.zeros((nt, solver_sws.Nz))
    ez_ref = np.zeros((nt, solver_ref.Nz))
    t_arr = np.arange(nt) * solver_sws.dt

    z_act_start = geom["z_active_start_m"]
    z_act_end = geom["z_active_end_m"]
    # Primary Poynting monitor planes sit in the smooth waveguide buffers (outside all vanes/grooves,
    # just before the 10-cell CPML absorbers) and integrate across the full waveguide height y in [-y_wall, +y_wall].
    dz_cell = float(z[1] - z[0])
    z_in_smooth_target = max(0.50e-3, (n_pml + 2) * dz_cell)
    z_out_smooth_target = z_tot - z_in_smooth_target
    iz_in = int(np.argmin(np.abs(solver_sws.z - z_in_smooth_target)))
    iz_out = int(np.argmin(np.abs(solver_sws.z - z_out_smooth_target)))
    # Also keep groove-adjacent tunnel-only planes for diagnostic comparison
    iz_in_tun = int(np.argmin(np.abs(solver_sws.z - (z_act_start + 0.5 * d))))
    iz_out_tun = int(np.argmin(np.abs(solver_sws.z - (z_act_end - 0.5 * d))))
    tunnel_mask_y = np.abs(solver_sws.y) <= spec.half_tunnel_m
    dy_full = solver_sws.dy
    dy_tun = solver_sws.dy[tunnel_mask_y]
    width_x = float(x[-1] - x[0])

    ey_in_full = np.zeros((nt, solver_sws.Ny)) if record_poynting_and_snapshots else None
    hx_in_full = np.zeros((nt, solver_sws.Ny)) if record_poynting_and_snapshots else None
    ey_out_full = np.zeros((nt, solver_sws.Ny)) if record_poynting_and_snapshots else None
    hx_out_full = np.zeros((nt, solver_sws.Ny)) if record_poynting_and_snapshots else None
    poynting_in_tun = np.zeros(nt) if record_poynting_and_snapshots else None
    poynting_out_tun = np.zeros(nt) if record_poynting_and_snapshots else None
    snapshots: list[dict[str, Any]] = []
    snap_steps: set[int] = set()
    if record_poynting_and_snapshots:
        t_enter = ti + z_act_start / float(u0_m_s)
        t_leave = ti + (z_act_end + 0.6 * L_act) / float(u0_m_s)
        for frac in np.linspace(0.05, 0.95, 18):
            t_snap = t_enter + frac * (t_leave - t_enter)
            snap_steps.add(int(np.clip(round(t_snap / solver_sws.dt), 0, nt - 1)))

    for n in range(nt):
        t = t_arr[n]
        beam_sws.update(solver_sws, t)
        solver_sws.one_step()
        ez_sws[n] = solver_sws.E[ix0, iy0, :, "z"]

        beam_ref.update(solver_ref, t)
        solver_ref.one_step()
        ez_ref[n] = solver_ref.E[ix0, iy0, :, "z"]

        if record_poynting_and_snapshots:
            ey_in_full[n] = (
                solver_sws.E[ix0, :, iz_in, "y"] - solver_ref.E[ix0, :, iz_in, "y"]
            )
            hx_in_full[n] = (
                solver_sws.H[ix0, :, iz_in, "x"] - solver_ref.H[ix0, :, iz_in, "x"]
            )
            ey_out_full[n] = (
                solver_sws.E[ix0, :, iz_out, "y"] - solver_ref.E[ix0, :, iz_out, "y"]
            )
            hx_out_full[n] = (
                solver_sws.H[ix0, :, iz_out, "x"] - solver_ref.H[ix0, :, iz_out, "x"]
            )

            ey_in_t = (
                solver_sws.E[ix0, tunnel_mask_y, iz_in_tun, "y"]
                - solver_ref.E[ix0, tunnel_mask_y, iz_in_tun, "y"]
            )
            hx_in_t = (
                solver_sws.H[ix0, tunnel_mask_y, iz_in_tun, "x"]
                - solver_ref.H[ix0, tunnel_mask_y, iz_in_tun, "x"]
            )
            ey_out_t = (
                solver_sws.E[ix0, tunnel_mask_y, iz_out_tun, "y"]
                - solver_ref.E[ix0, tunnel_mask_y, iz_out_tun, "y"]
            )
            hx_out_t = (
                solver_sws.H[ix0, tunnel_mask_y, iz_out_tun, "x"]
                - solver_ref.H[ix0, tunnel_mask_y, iz_out_tun, "x"]
            )
            poynting_in_tun[n] = float(np.sum((-ey_in_t * hx_in_t) * dy_tun) * width_x)
            poynting_out_tun[n] = float(np.sum((-ey_out_t * hx_out_t) * dy_tun) * width_x)

            if n in snap_steps:
                ez_2d = (
                    solver_sws.E[ix0, :, :, "z"] - solver_ref.E[ix0, :, :, "z"]
                ).copy()
                z_bunch = float(u0_m_s * (t - ti))
                snapshots.append(
                    {
                        "step": int(n),
                        "t_ps": float(t * 1e12),
                        "z_bunch_mm": float(z_bunch * 1e3),
                        "ez_2d": ez_2d,
                    }
                )

    wall_s = time.perf_counter() - t0_wall
    Path(geom["stl_path"]).unlink(missing_ok=True)

    ez_scat = ez_sws - ez_ref
    z_mask = (solver_sws.z >= z_act_start) & (solver_sws.z <= z_act_end)
    z_int = solver_sws.z[z_mask]
    dz_int = solver_sws.dz[z_mask]
    ez_int = ez_scat[:, z_mask]

    dt = float(solver_sws.dt)
    s, wp, ti_loc = _integrate_synchronous_wake_potential(
        t_arr,
        z_int,
        dz_int,
        ez_int,
        z_min_m=float(solver_sws.z.min()),
        u0_m_s=float(u0_m_s),
        ti_s=float(ti),
        wake_len_m=float(wake_len),
        dt_s=dt,
        q_coulomb=float(q_coulomb),
    )

    ws = WakeSolver(
        q=q_coulomb,
        sigmaz=sigmaz_m,
        beta=beta,
        ti=ti_loc,
        save=False,
        verbose=0,
    )
    ws.s = s
    ws.WP = wp
    ws.calc_long_Z(samples=fft_samples, fmax=f_max_hz)

    f_hz = np.asarray(ws.f, dtype=float)
    f_ghz = f_hz * 1e-9
    re_z = np.real(np.asarray(ws.Z, dtype=complex))
    im_z = np.imag(np.asarray(ws.Z, dtype=complex))

    width_scale = spec.transverse_width_m / width_x
    re_z = re_z * width_scale
    im_z = im_z * width_scale

    mask_b = (f_ghz >= f_search_band_ghz[0]) & (f_ghz <= f_search_band_ghz[1])
    f_b = f_ghz[mask_b]
    rz_b = re_z[mask_b]
    i_pk = int(np.argmax(rz_b))
    f_pk_ghz = float(f_b[i_pk])
    z_pk_ohm = float(rz_b[i_pk])

    half_val = 0.5 * z_pk_ohm
    left_idx = i_pk
    while left_idx > 0 and rz_b[left_idx] >= half_val:
        left_idx -= 1
    right_idx = i_pk
    while right_idx < len(rz_b) - 1 and rz_b[right_idx] >= half_val:
        right_idx += 1
    f_left = float(
        np.interp(
            half_val,
            [rz_b[left_idx], rz_b[min(left_idx + 1, i_pk)]],
            [f_b[left_idx], f_b[min(left_idx + 1, i_pk)]],
        )
    )
    f_right = float(
        np.interp(
            half_val,
            [rz_b[right_idx], rz_b[max(right_idx - 1, i_pk)]],
            [f_b[right_idx], f_b[max(right_idx - 1, i_pk)]],
        )
    )
    fwhm_ghz = max(f_right - f_left, 0.0)

    beta_sync = 2.0 * np.pi * (f_pk_ghz * 1e9) / float(u0_m_s)
    k_on_axis_beam = extract_pierce_k_from_peak_z(
        z_pk_ohm, beta_n_rad_m=beta_sync, length_m=L_act
    )
    eta_sheet = sheet_beam_averaging_factor(
        beta_sync, f_pk_ghz * 1e9, spec.beam_thickness_m
    )
    dy_fine = spec.vane_thickness_m / float(n_feat)
    eta_sheet_nodes = sheet_beam_node_averaging_factor(
        beta_sync, f_pk_ghz * 1e9, spec.beam_thickness_m, dy_fine
    )
    k_sheet_avg_beam = float(k_on_axis_beam * eta_sheet)
    k_sheet_avg_nodes_est = float(k_on_axis_beam * eta_sheet_nodes)

    k_loss_v_per_pc = compute_loss_factor_v_per_pc(
        f_b * 1e9, np.maximum(rz_b, 0.0), sigmaz_m, u0_m_s
    )
    ff_sync_sq = float(gaussian_bunch_form_factor_sq(f_pk_ghz * 1e9, sigmaz_m, u0_m_s))
    w_loss_j = float((q_coulomb**2) * (k_loss_v_per_pc * 1e12))
    dw_dz_fit_j_per_m = float(w_loss_j / L_act)

    out: dict[str, Any] = {
        "n_periods": int(n_periods),
        "active_length_m": float(L_act),
        "active_length_mm": float(L_act * 1e3),
        "n_feat": int(n_feat),
        "use_graded_mesh": bool(use_graded_mesh),
        "conformal_mode": str(conformal_mode),
        "termination_mode": termination_mode,
        "taper_scales": [float(s) for s in taper_scales],
        "symmetry": symmetry,
        "transverse_width_m": float(spec.transverse_width_m),
        "width_scale": float(width_scale),
        "u0_m_s": float(u0_m_s),
        "u0_over_c": float(beta),
        "ysource_m": float(ysource_m),
        "sigmaz_m": float(sigmaz_m),
        "q_coulomb": float(q_coulomb),
        "f_search_band_ghz": [float(f_search_band_ghz[0]), float(f_search_band_ghz[1])],
        "grid_shape": [int(solver_sws.Nx), int(solver_sws.Ny), int(solver_sws.Nz)],
        "n_cells": int(solver_sws.N),
        "dt_s": float(solver_sws.dt),
        "nt": int(nt),
        "wall_time_s": float(wall_s),
        "estimated_memory_mb": float(est.host_bytes / 1e6),
        "f_pk_ghz": f_pk_ghz,
        "re_z_pk_ohm": z_pk_ohm,
        "re_z_pk_over_n2_ohm": float(z_pk_ohm / (n_periods**2)),
        "fwhm_ghz": float(fwhm_ghz),
        "f_half_left_ghz": float(f_left),
        "f_half_right_ghz": float(f_right),
        "fwhm_times_n_ghz": float(fwhm_ghz * n_periods),
        "beta_sync_rad_m": float(beta_sync),
        "k_on_axis_beam_ohm": float(k_on_axis_beam),
        "eta_sheet": float(eta_sheet),
        "eta_sheet_nodes": float(eta_sheet_nodes),
        "k_sheet_avg_beam_ohm": float(k_sheet_avg_beam),
        "k_sheet_avg_nodes_est_ohm": float(k_sheet_avg_nodes_est),
        "k_loss_v_per_pc": float(k_loss_v_per_pc),
        "w_loss_active_j": float(w_loss_j),
        "form_factor_sync_sq": float(ff_sync_sq),
        "dw_dz_fit_j_per_m": float(dw_dz_fit_j_per_m),
        "dw_dz_unattenuated_j_per_m": float(
            dw_dz_fit_j_per_m / max(ff_sync_sq, 1e-12)
        ),
        "f_ghz": f_ghz,
        "re_z_ohm": re_z,
        "im_z_ohm": im_z,
        "s_m": s,
        "wp_v_per_pc": wp,
    }
    if f_sync_target_ghz is not None:
        re_z_target = float(np.interp(float(f_sync_target_ghz), f_ghz, re_z))
        beta_target = 2.0 * np.pi * (float(f_sync_target_ghz) * 1e9) / float(u0_m_s)
        k_target = extract_pierce_k_from_peak_z(
            re_z_target, beta_n_rad_m=beta_target, length_m=L_act
        )
        out["f_sync_target_ghz"] = float(f_sync_target_ghz)
        out["re_z_at_sync_target_ohm"] = re_z_target
        out["k_at_sync_target_ohm"] = float(k_target)

    if record_poynting_and_snapshots:
        t_ps = t_arr * 1e12
        z_in_m = float(solver_sws.z[iz_in])
        z_out_m = float(solver_sws.z[iz_out])

        # Bandpass-filter the full-cross-section smooth-guide fields to f_search_band_ghz
        # to separate the radiated passband slow-wave packet from evanescent space-charge transients
        freqs_p = np.fft.rfftfreq(nt, d=dt) * 1e-9
        bp_mask = (
            (freqs_p >= f_search_band_ghz[0]) & (freqs_p <= f_search_band_ghz[1])
        )[:, None]
        ey_in_bp = np.fft.irfft(np.fft.rfft(ey_in_full, axis=0) * bp_mask, n=nt, axis=0)
        hx_in_bp = np.fft.irfft(np.fft.rfft(hx_in_full, axis=0) * bp_mask, n=nt, axis=0)
        ey_out_bp = np.fft.irfft(
            np.fft.rfft(ey_out_full, axis=0) * bp_mask, n=nt, axis=0
        )
        hx_out_bp = np.fft.irfft(
            np.fft.rfft(hx_out_full, axis=0) * bp_mask, n=nt, axis=0
        )

        p_in_wave = (
            np.sum((-ey_in_bp * hx_in_bp) * dy_full[None, :], axis=1)
            * width_x
            * width_scale
        )
        p_out_wave = (
            np.sum((-ey_out_bp * hx_out_bp) * dy_full[None, :], axis=1)
            * width_x
            * width_scale
        )
        e_in = float(np.trapezoid(p_in_wave, t_arr))
        e_out = float(np.trapezoid(p_out_wave, t_arr))
        e_rad_total = float(e_out - e_in)

        # Also compute total bunch energy loss across the entire gridded region between smooth ports
        z_mask_tot = (solver_sws.z >= z_in_m) & (solver_sws.z <= z_out_m)
        z_int_tot = solver_sws.z[z_mask_tot]
        dz_int_tot = solver_sws.dz[z_mask_tot]
        ez_int_tot = ez_scat[:, z_mask_tot]
        s_tot, wp_tot, ti_loc_tot = _integrate_synchronous_wake_potential(
            t_arr,
            z_int_tot,
            dz_int_tot,
            ez_int_tot,
            z_min_m=float(solver_sws.z.min()),
            u0_m_s=float(u0_m_s),
            ti_s=float(ti),
            wake_len_m=float(wake_len_sim),
            dt_s=dt,
            q_coulomb=float(q_coulomb),
        )
        ws_tot = WakeSolver(
            q=q_coulomb,
            sigmaz=sigmaz_m,
            beta=beta,
            ti=ti_loc_tot,
            save=False,
            verbose=0,
        )
        ws_tot.s = s_tot
        ws_tot.WP = wp_tot
        ws_tot.calc_long_Z(samples=fft_samples, fmax=f_max_hz)
        re_z_tot = np.real(np.asarray(ws_tot.Z, dtype=complex)) * width_scale
        k_loss_tot_v_per_pc = compute_loss_factor_v_per_pc(
            f_b * 1e9, np.maximum(re_z_tot[mask_b], 0.0), sigmaz_m, u0_m_s
        )
        w_loss_tot_j = float((q_coulomb**2) * (k_loss_tot_v_per_pc * 1e12))

        # Tunnel-only groove-adjacent diagnostic values
        z_in_tun_m = float(solver_sws.z[iz_in_tun])
        z_out_tun_m = float(solver_sws.z[iz_out_tun])
        mask_in_tun = np.abs(z_in_tun_m - float(u0_m_s) * (t_arr - ti)) > (
            3.5 * float(sigmaz_m)
        )
        mask_out_tun = np.abs(z_out_tun_m - float(u0_m_s) * (t_arr - ti)) > (
            3.5 * float(sigmaz_m)
        )
        e_in_tun = float(
            np.trapezoid(poynting_in_tun * mask_in_tun * width_scale, t_arr)
        )
        e_out_tun = float(
            np.trapezoid(poynting_out_tun * mask_out_tun * width_scale, t_arr)
        )

        out["poynting"] = {
            "t_ps": t_ps,
            "p_in_w": p_in_wave,
            "p_out_w": p_out_wave,
            "integrated_energy_in_j": e_in,
            "integrated_energy_out_j": e_out,
            "total_radiated_cpml_energy_j": e_rad_total,
            "w_loss_active_j": float(w_loss_j),
            "w_loss_between_smooth_ports_j": float(w_loss_tot_j),
            "energy_balance_rel_error_vs_active": float(
                abs(e_rad_total - w_loss_j) / max(w_loss_j, 1e-30)
            ),
            "energy_balance_rel_error_vs_total": float(
                abs(e_rad_total - w_loss_tot_j) / max(w_loss_tot_j, 1e-30)
            ),
            "upstream_fraction": float(abs(e_in) / max(e_rad_total, 1e-30)),
            "downstream_fraction": float(abs(e_out) / max(e_rad_total, 1e-30)),
            "z_in_mm": float(z_in_m * 1e3),
            "z_out_mm": float(z_out_m * 1e3),
            "y_integration_bounds_mm": [
                float(solver_sws.y[0] * 1e3),
                float(solver_sws.y[-1] * 1e3),
            ],
            "tunnel_only_groove_plane_energy_in_j": e_in_tun,
            "tunnel_only_groove_plane_energy_out_j": e_out_tun,
        }
        out["snapshots"] = snapshots
        out["y_mm"] = (solver_sws.y * 1e3).copy()
        out["z_mm"] = (solver_sws.z * 1e3).copy()
        out["z_active_start_mm"] = float(z_act_start * 1e3)
        out["z_active_end_mm"] = float(z_act_end * 1e3)
    return out


def run_2d_sheet_beam_y_sweep(
    spec: KaretnikovaGratingSpec | None = None,
    *,
    n_periods: int = 20,
    n_feat: int = 4,
    use_graded_mesh: bool = False,
    y_positions_um: tuple[float, ...] = (0.0, 25.0, 50.0, 75.0),
    on_axis_run: dict[str, Any] | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Measure the transverse Pierce impedance profile K_{-1}(y_0) across the sheet beam in FIT.

    By even transverse symmetry K_{-1}(-y_0) = K_{-1}(+y_0) of the fundamental glide mode,
    sampling y_0 in {0, 25, 50, 75} um covers all 7 L1 grid nodes
    y_j in {-75, -50, -25, 0, +25, +50, +75} um across the H_b = 150 um sheet beam.
    """
    if spec is None:
        spec = KaretnikovaGratingSpec()
    bench = load_cold_test_benchmarks()
    l1 = bench["levels_2d"]["L1"]
    l2 = bench["levels_2d"]["L2"]

    y_pos_um = [float(y) for y in y_positions_um]
    k_by_y: dict[str, float] = {}
    re_z_by_y: dict[str, float] = {}
    wall_s = 0.0

    for y_um in y_pos_um:
        if abs(y_um) < 1e-9 and on_axis_run is not None:
            r = on_axis_run
        else:
            r = run_2d_karetnikova_beam_wake(
                spec,
                n_periods=n_periods,
                n_feat=n_feat,
                use_graded_mesh=use_graded_mesh,
                ysource_m=y_um * 1e-6,
                quick=quick,
            )
            wall_s += float(r["wall_time_s"])
        key = f"{int(round(y_um))}"
        k_by_y[key] = float(r["k_on_axis_beam_ohm"])
        re_z_by_y[key] = float(r["re_z_pk_ohm"])

    y_half_m = np.array([y * 1e-6 for y in y_pos_um], dtype=float)
    k_half = np.array([k_by_y[f"{int(round(y))}"] for y in y_pos_um], dtype=float)
    k0 = float(k_half[0])

    # Symmetrize across [-H_b/2, +H_b/2]: 7 L1 grid nodes {-75, -50, -25, 0, +25, +50, +75} um
    y_full_um = np.concatenate([-np.array(y_pos_um[:0:-1]), np.array(y_pos_um)])
    k_full = np.concatenate([k_half[:0:-1], k_half])

    # 1. Equal-weight discrete node mean (exact same np.mean definition as Tutorial 6.2)
    k_avg_7node = float(np.mean(k_full))
    ratio_7node = float(k_avg_7node / max(k0, 1e-30))

    # 2. Trapezoidal integral (1 / H_b) int_{-H_b/2}^{+H_b/2} K_{-1}(y) dy
    span_m = float(y_full_um[-1] - y_full_um[0]) * 1e-6
    k_avg_trapz = float(np.trapezoid(k_full, y_full_um * 1e-6) / max(span_m, 1e-30))
    ratio_trapz = float(k_avg_trapz / max(k0, 1e-30))

    # 3. Continuum cosh^2(tau_{-1} y) formula K_{-1}(0) * eta_sheet
    beta_sync = l2["beta_m1_rad_m"]
    f_sync = l2["f_sync_bwd_hz"]
    eta_cont = sheet_beam_averaging_factor(beta_sync, f_sync, spec.beam_thickness_m)
    k_avg_cosh = float(k0 * eta_cont)

    tau_m1 = transverse_decay_constant(beta_sync, f_sync)
    cosh2_half = (np.cosh(tau_m1 * y_half_m) ** 2).tolist()

    return {
        "n_periods": int(n_periods),
        "n_feat": int(n_feat),
        "use_graded_mesh": bool(use_graded_mesh),
        "beam_thickness_um": float(spec.beam_thickness_m * 1e6),
        "y_positions_um": y_pos_um,
        "y_full_nodes_um": y_full_um.tolist(),
        "k_m1_at_y_ohm": [float(v) for v in k_half],
        "k_m1_full_nodes_ohm": [float(v) for v in k_full],
        "re_z_pk_at_y_ohm": [float(re_z_by_y[f"{int(round(y))}"]) for y in y_pos_um],
        "k_ratio_over_axis": [float(v / max(k0, 1e-30)) for v in k_half],
        "analytic_cosh2_at_y": [float(v) for v in cosh2_half],
        "k_on_axis_ohm": k0,
        "k_sheet_avg_7node_mean_ohm": k_avg_7node,
        "ratio_7node_mean_over_axis": ratio_7node,
        "k_sheet_avg_trapz_ohm": k_avg_trapz,
        "ratio_trapz_over_axis": ratio_trapz,
        "k_sheet_avg_analytic_cosh_ohm": k_avg_cosh,
        "eta_sheet_continuum": float(eta_cont),
        "cold_test_L1_k_sheet_avg_ohm": float(l1["k_m1_beam_avg_ohm"]),
        "cold_test_L1_ratio_over_axis": float(l1["k_m1_ratio_beam_over_axis"]),
        "cold_test_L2_k_sheet_avg_ohm": float(l2["k_m1_beam_avg_ohm"]),
        "cold_test_L2_ratio_over_axis": float(l2["k_m1_ratio_beam_over_axis"]),
        "rel_diff_7node_vs_cold_L1": float(
            abs(k_avg_7node - l1["k_m1_beam_avg_ohm"]) / l1["k_m1_beam_avg_ohm"]
        ),
        "rel_diff_ratio_7node_vs_cold_L1": float(
            abs(ratio_7node - l1["k_m1_ratio_beam_over_axis"])
            / l1["k_m1_ratio_beam_over_axis"]
        ),
        "wall_time_s": float(wall_s),
    }


def build_3d_marosi_axes(
    spec: MarosiGlideSWSSpec,
    *,
    z_total_m: float,
    nx: int = 20,
    n_feat_y: int = 4,
    nz_per_period: int = 20,
    use_graded_mesh: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 3D FIT axes for the Marosi V-band glide SWS with >= 4 cells across the 0.10 mm pillar/beam features."""
    a = spec.waveguide_width_m
    b = spec.waveguide_height_m
    d = spec.period_m
    x = np.linspace(-0.5 * a, 0.5 * a, int(nx) + 1)

    dy_fine = spec.pillar_thickness_m / float(n_feat_y)
    y_core = 0.25e-3
    if use_graded_mesh:
        n_core = int(round(2.0 * y_core / dy_fine))
        n_clad = max(int(round((0.5 * b - y_core) / (1.5 * dy_fine))), 4)
        y_bot = np.linspace(-0.5 * b, -y_core, n_clad + 1)[:-1]
        y_mid = np.linspace(-y_core, y_core, n_core + 1)[:-1]
        y_top = np.linspace(y_core, 0.5 * b, n_clad + 1)
        y = np.concatenate([y_bot, y_mid, y_top])
    else:
        ny = int(round(b / dy_fine))
        y = np.linspace(-0.5 * b, 0.5 * b, ny + 1)

    nz = int(round((float(z_total_m) / d) * int(nz_per_period)))
    z = np.linspace(0.0, float(z_total_m), nz + 1)
    return x, y, z


def run_3d_marosi_beam_wake(
    spec: MarosiGlideSWSSpec | None = None,
    *,
    n_periods: int = 6,
    nx: int = 20,
    n_feat_y: int = 4,
    nz_per_period: int = 20,
    use_graded_mesh: bool = True,
    sigmaz_m: float = 0.35e-3,
    q_coulomb: float = 1.0e-12,
    f_max_hz: float = 120.0e9,
    fft_samples: int = 16384,
) -> dict[str, Any]:
    """Run a 3D Marosi V-band staggered-pillar glide SWS beam-wake simulation in Wakis FIT."""
    if spec is None:
        spec = MarosiGlideSWSSpec()
    u0_m_s, _ = relativistic_beam_velocity(spec.beam_voltage_v)
    beta = float(u0_m_s / c)
    d = spec.period_m

    taper_scales = [0.45]
    z_buf = 0.75 * d
    geom = build_3d_marosi_stl(
        spec,
        n_periods=n_periods,
        z_buffer_m=z_buf,
        taper_scales=taper_scales,
    )
    z_tot = geom["z_total_m"]
    L_act = geom["active_length_m"]

    x, y, z = build_3d_marosi_axes(
        spec,
        z_total_m=z_tot,
        nx=nx,
        n_feat_y=n_feat_y,
        nz_per_period=nz_per_period,
        use_graded_mesh=use_graded_mesh,
    )

    t0_wall = time.perf_counter()
    grid_sws = GridFIT3D(
        x=x,
        y=y,
        z=z,
        stl_solids={"pillars": geom["stl_path"]},
        stl_materials={"pillars": "pec"},
        conformal="pec",
        verbose=0,
    )
    solver_sws = SolverFIT3D(
        grid_sws,
        use_stl=True,
        bc_low=["pec", "pec", "cpml"],
        bc_high=["pec", "pec", "cpml"],
        n_pml=8,
        verbose=0,
    )
    solver_sws.source_type = "direct"
    est = estimate(
        (solver_sws.Nx, solver_sws.Ny, solver_sws.Nz),
        steps=1000,
        cpml=True,
        conformal="pec",
    )
    preflight(est, "laptop")

    grid_ref = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver_ref = SolverFIT3D(
        grid_ref,
        use_stl=False,
        bc_low=["pec", "pec", "cpml"],
        bc_high=["pec", "pec", "cpml"],
        n_pml=8,
        verbose=0,
    )
    solver_ref.source_type = "direct"

    ti = 4.2 * float(sigmaz_m) / float(u0_m_s)
    beam_sws = Beam(
        xsource=0.0, ysource=0.0, beta=beta, q=q_coulomb, sigmaz=sigmaz_m, ti=ti
    )
    beam_ref = Beam(
        xsource=0.0, ysource=0.0, beta=beta, q=q_coulomb, sigmaz=sigmaz_m, ti=ti
    )

    wake_len = 2.4e-3
    t_sim = ti + (z_tot + wake_len) / float(u0_m_s)
    nt = int(np.ceil(t_sim / solver_sws.dt))

    ix0 = int(np.argmin(np.abs(solver_sws.grid.x - 0.0)))
    iy0 = int(np.argmin(np.abs(solver_sws.grid.y - 0.0)))
    ez_sws = np.zeros((nt, solver_sws.Nz))
    ez_ref = np.zeros((nt, solver_ref.Nz))
    t_arr = np.arange(nt) * solver_sws.dt

    for n in range(nt):
        t = t_arr[n]
        beam_sws.update(solver_sws, t)
        solver_sws.one_step()
        ez_sws[n] = solver_sws.E[ix0, iy0, :, "z"]

        beam_ref.update(solver_ref, t)
        solver_ref.one_step()
        ez_ref[n] = solver_ref.E[ix0, iy0, :, "z"]

    wall_s = time.perf_counter() - t0_wall
    Path(geom["stl_path"]).unlink(missing_ok=True)

    ez_scat = ez_sws - ez_ref
    z_act_start = geom["z_active_start_m"]
    z_act_end = geom["z_active_end_m"]
    z_mask = (solver_sws.z >= z_act_start) & (solver_sws.z <= z_act_end)
    z_int = solver_sws.z[z_mask]
    dz_int = solver_sws.dz[z_mask]
    ez_int = ez_scat[:, z_mask]

    dt = float(solver_sws.dt)
    s, wp, ti_loc = _integrate_synchronous_wake_potential(
        t_arr,
        z_int,
        dz_int,
        ez_int,
        z_min_m=float(solver_sws.z.min()),
        u0_m_s=float(u0_m_s),
        ti_s=float(ti),
        wake_len_m=float(wake_len),
        dt_s=dt,
        q_coulomb=float(q_coulomb),
    )

    ws = WakeSolver(
        q=q_coulomb,
        sigmaz=sigmaz_m,
        beta=beta,
        ti=ti_loc,
        save=False,
        verbose=0,
    )
    ws.s = s
    ws.WP = wp
    ws.calc_long_Z(samples=fft_samples, fmax=f_max_hz)

    f_ghz = np.asarray(ws.f, dtype=float) * 1e-9
    re_z = np.real(np.asarray(ws.Z, dtype=complex))
    im_z = np.imag(np.asarray(ws.Z, dtype=complex))

    mask_b = (f_ghz >= 52.0) & (f_ghz <= 72.0)
    f_b = f_ghz[mask_b]
    rz_b = re_z[mask_b]
    i_pk = int(np.argmax(rz_b))
    f_pk_ghz = float(f_b[i_pk])
    z_pk_ohm = float(rz_b[i_pk])

    half_val = 0.5 * z_pk_ohm
    half_mask = rz_b >= half_val
    fwhm_ghz = (
        float(f_b[half_mask][-1] - f_b[half_mask][0]) if np.any(half_mask) else 0.0
    )

    beta_pk = 2.0 * np.pi * (f_pk_ghz * 1e9) / float(u0_m_s)
    k_on_axis_at_pk = extract_pierce_k_from_peak_z(
        z_pk_ohm, beta_n_rad_m=beta_pk, length_m=L_act
    )

    # Also evaluate at the exact cold-test phase-synchronism frequency f_s = 64.8442 GHz
    f_sync_cold_ghz = 64.8442
    re_z_at_sync = float(np.interp(f_sync_cold_ghz, f_ghz, re_z))
    beta_sync_cold = 2.0 * np.pi * (f_sync_cold_ghz * 1e9) / float(u0_m_s)
    k_on_axis_at_sync = extract_pierce_k_from_peak_z(
        re_z_at_sync, beta_n_rad_m=beta_sync_cold, length_m=L_act
    )

    return {
        "n_periods": int(n_periods),
        "active_length_m": float(L_act),
        "active_length_mm": float(L_act * 1e3),
        "u0_m_s": float(u0_m_s),
        "u0_over_c": float(beta),
        "sigmaz_m": float(sigmaz_m),
        "grid_shape": [int(solver_sws.Nx), int(solver_sws.Ny), int(solver_sws.Nz)],
        "n_cells": int(solver_sws.N),
        "dt_s": float(solver_sws.dt),
        "nt": int(nt),
        "wall_time_s": float(wall_s),
        "estimated_memory_mb": float(est.host_bytes / 1e6),
        "f_pk_ghz": f_pk_ghz,
        "re_z_pk_ohm": z_pk_ohm,
        "fwhm_ghz": fwhm_ghz,
        "beta_pk_rad_m": float(beta_pk),
        "k_on_axis_at_pk_ohm": float(k_on_axis_at_pk),
        "f_sync_cold_ghz": f_sync_cold_ghz,
        "re_z_at_sync_ohm": re_z_at_sync,
        "k_on_axis_at_sync_ohm": float(k_on_axis_at_sync),
        "f_ghz": f_ghz,
        "re_z_ohm": re_z,
        "im_z_ohm": im_z,
        "s_m": s,
        "wp_v_per_pc": wp,
    }


class _SWSSchematicSolverProxy:
    """Lightweight proxy around SolverFIT3D that holds Box structures for wakis.schematic."""

    def __init__(self, solver: SolverFIT3D, grid_bare: GridFIT3D, structures: tuple[Any, ...] = ()):
        self._solver = solver
        self.grid = grid_bare
        self.Nx = solver.Nx
        self.Ny = solver.Ny
        self.Nz = solver.Nz
        self.dt = solver.dt
        self.n_pml = getattr(solver, "n_pml", 0)
        self.bc_low = list(getattr(solver, "bc_low", ["pmc", "pec", "cpml"]))
        self.bc_high = list(getattr(solver, "bc_high", ["pmc", "pec", "cpml"]))
        self.bg = [1.0, 1.0, 0.0]
        self._assigned_structures = tuple(structures)


def plot_sws_beam_setup_schematics(figures_dir: Path | str) -> dict[str, str]:
    """Generate 00a/00b setup figures and markdown tables for 2D Karetnikova and 3D Marosi SWS."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()

    spec_2d = KaretnikovaGratingSpec()
    spec_3d = MarosiGlideSWSSpec()
    u0_2d, _ = relativistic_beam_velocity(spec_2d.beam_voltage_v)
    u0_3d, _ = relativistic_beam_velocity(spec_3d.beam_voltage_v)

    # 1. 2D Karetnikova sheet-beam grating setup (N=6 periods + 2-stage tapers for schematic clarity)
    d2 = spec_2d.period_m
    l2 = spec_2d.vane_thickness_m
    h2 = spec_2d.vane_height_m
    a2 = spec_2d.half_tunnel_m
    y_wall = a2 + h2
    n_act_2d = 6
    taper_2d = [0.25, 0.55]
    n_tap_2d = len(taper_2d)
    n_tot_2d = n_act_2d + 2 * n_tap_2d
    z_buf_2d = 1.5 * d2

    geom_2d = build_2d_karetnikova_stl(
        spec_2d,
        n_periods=n_act_2d,
        z_buffer_m=z_buf_2d,
        taper_scales=taper_2d,
        symmetry="glide",
    )
    z_tot_2d = geom_2d["z_total_m"]
    z_act_start_2d = geom_2d["z_active_start_m"]
    z_act_end_2d = geom_2d["z_active_end_m"]

    x_ax_2d, y_ax_2d, z_ax_2d = build_2d_karetnikova_axes(
        spec_2d, z_total_m=z_tot_2d, n_feat=4, use_graded_mesh=True
    )
    grid_2d = GridFIT3D(
        x=x_ax_2d,
        y=y_ax_2d,
        z=z_ax_2d,
        stl_solids={"sws": geom_2d["stl_path"]},
        stl_materials={"sws": "pec"},
        conformal="pec",
        verbose=0,
    )
    solver_2d = SolverFIT3D(
        grid_2d,
        use_stl=True,
        bc_low=["pmc", "pec", "cpml"],
        bc_high=["pmc", "pec", "cpml"],
        n_pml=10,
        verbose=0,
    )
    Path(geom_2d["stl_path"]).unlink(missing_ok=True)
    grid_2d_bare = GridFIT3D(x=x_ax_2d, y=y_ax_2d, z=z_ax_2d, verbose=0)

    structs_2d: list[Any] = []
    for k in range(n_tot_2d):
        if k < n_tap_2d:
            sc = taper_2d[k]
        elif k >= n_tap_2d + n_act_2d:
            sc = taper_2d[n_tot_2d - 1 - k]
        else:
            sc = 1.0
        hk = sc * h2
        z0 = z_buf_2d + k * d2
        z_bot = z0 + 0.5 * d2
        b_bot = Box(
            center=(0.0, -y_wall + 0.5 * hk, z_bot),
            size=(1.0e-3, hk, l2),
            material="pec",
        )
        b_bot.name = "Bottom PEC vanes (staggered grating + 2-stage taper)"
        b_top = Box(
            center=(0.0, y_wall - 0.5 * hk, z0 + 0.25 * l2),
            size=(1.0e-3, hk, l2),
            material="pec",
        )
        b_top.name = "Top PEC vanes (glide shift d/2 = 250 µm)"
        structs_2d.extend([b_bot, b_top])

    proxy_2d = _SWSSchematicSolverProxy(solver_2d, grid_2d_bare, tuple(structs_2d))
    bench_2d = load_cold_test_benchmarks()
    f_sync_2d_ghz = float(bench_2d["levels_2d"]["L1"]["f_sync_bwd_ghz"])
    f_sync_2d_hz = f_sync_2d_ghz * 1e9
    pulse_2d = GaussianSource(frequency=f_sync_2d_hz, fwidth=45.0e9)
    sources_2d = [
        {
            "kind": "line",
            "type_name": "Beam",
            "short_word": "20 kV bunch",
            "label": "Rigid electron bunch (V₀ = 20 kV, u₀ = 0.2719c, q = 1 pC, σ_z = 250 µm)",
            "desc": "Rigid electron bunch J_z(z − u₀ t) along y = 0 (20 kV, u₀ = 0.2719c)",
            "pos_str": "y = 0 (sheet-beam channel |y| ≤ 75 µm), z ∈ [0, L_tot]",
            "polarization": "Jz (TM slow space harmonic)",
            "h_bounds": (z_buf_2d, z_tot_2d - z_buf_2d),
            "v_bounds": (0.0, 0.0),
            "pulse": {
                "f0": f_sync_2d_hz,
                "fwidth": 45.0e9,
                "t_peak": float(5.0 * 0.25e-3 / u0_2d),
                "t_end": float(10.0 * 0.25e-3 / u0_2d),
                "src_time": pulse_2d,
            },
        }
    ]
    monitors_2d = [
        {
            "kind": "line",
            "type_name": "WakeSolver",
            "short_word": "wake E_z(s)",
            "label": "Longitudinal wake integration line (L = N d) → W_∥(s), Re Z_∥(f)",
            "desc": "Wake integration line E_z(y=0, z, t) over active SWS length L = N d",
            "pos_str": f"y = 0, z ∈ [{z_act_start_2d*1e3:.2f}, {z_act_end_2d*1e3:.2f}] mm",
            "freq_str": f"0–200 GHz (synchronous crossing f_s = {f_sync_2d_ghz:.2f} GHz)",
            "h_bounds": (z_act_start_2d, z_act_end_2d),
            "v_bounds": (-0.04e-3, 0.04e-3),
        },
        {
            "kind": "plane",
            "type_name": "Poynting",
            "short_word": "S_z(z_in)",
            "label": "Upstream smooth-guide Poynting plane z_in (full height |y| ≤ 400 µm)",
            "desc": "Upstream Poynting monitor S_z(z_in, t) across full smooth guide |y| ≤ 400 µm",
            "pos_str": "z_in = 0.50 mm (smooth guide before −z CPML), y ∈ [−400, +400] µm",
            "freq_str": "Passband-filtered Poynting flux P_z(t)",
            "h_bounds": (0.50e-3, 0.50e-3),
            "v_bounds": (-y_wall, y_wall),
        },
        {
            "kind": "plane",
            "type_name": "Poynting",
            "short_word": "S_z(z_out)",
            "label": "Downstream smooth-guide Poynting plane z_out (full height |y| ≤ 400 µm)",
            "desc": "Downstream Poynting monitor S_z(z_out, t) across full smooth guide |y| ≤ 400 µm",
            "pos_str": f"z_out = {(z_tot_2d - 0.50e-3)*1e3:.2f} mm (smooth guide before +z CPML), y ∈ [−400, +400] µm",
            "freq_str": "Passband-filtered Poynting flux P_z(t)",
            "h_bounds": (z_tot_2d - 0.50e-3, z_tot_2d - 0.50e-3),
            "v_bounds": (-y_wall, y_wall),
        },
    ]
    mat_names_2d = {
        "background": "Vacuum beam tunnel & grooves (ε_r = 1.00, H_b = 150 µm sheet beam, W_x = 1.0 mm)",
    }
    res_2d = schematic.save_setup_figures(
        proxy_2d,
        figures_dir,
        prefix="00",
        plane="zy",
        material_names=mat_names_2d,
        sources=sources_2d,
        monitors=monitors_2d,
        unit="mm",
        freq=f_sync_2d_hz,
        min_feature=l2,
        variations=[
            {
                "name": "Interaction length & off-axis sheet-beam sweep (Task 2)",
                "details": "N ∈ {5, 10, 20, 40} periods (L = 2.5–20.0 mm) + 2-stage vane tapers [0.25h, 0.55h] into CPML (|Γ(f_s)| = 18.0%), plus y₀ ∈ {0, 25, 50, 75} µm sheet-beam profile",
            },
            {
                "name": "Backward-wave vs forward-wave synchronism & energy balance (Task 3)",
                "details": "Glide n = −1 backward wave (20 kV, u₀ = 0.2719c, v_g = −0.544c < 0, f_s = 114.29 GHz) vs mirror n = 0 forward wave (109.9 kV, u₀ = 0.5689c, v_g = +0.325c > 0, f_s = 127.92 GHz, y₀ = 50 µm)",
            },
            {
                "name": "Detuned beam & shorted PEC cavity mutations (Task 4)",
                "details": "Velocity detuning u₀ ± 5% shifts f_s by Δf_s; replacing CPML + tapers with PEC end walls produces standing-wave cavity modes at β₀ d = mπ/N",
            },
        ],
    )

    # 2. 3D Marosi V-band glide-symmetric pillar SWS setup
    d3 = spec_3d.period_m
    a3 = spec_3d.waveguide_width_m
    b3 = spec_3d.waveguide_height_m
    w3 = spec_3d.pillar_width_m
    l3 = spec_3d.pillar_length_m
    t3 = spec_3d.pillar_thickness_m
    y_in_3d = 0.5 * spec_3d.beam_thickness_m + spec_3d.pillar_beam_clearance_m
    y_out_3d = y_in_3d + t3
    yc_3d = 0.5 * (y_in_3d + y_out_3d)
    n_act_3d = 6
    taper_3d = [0.45]
    n_tap_3d = len(taper_3d)
    n_tot_3d = n_act_3d + 2 * n_tap_3d
    z_buf_3d = 0.75 * d3

    geom_3d = build_3d_marosi_stl(
        spec_3d,
        n_periods=n_act_3d,
        z_buffer_m=z_buf_3d,
        taper_scales=taper_3d,
    )
    z_tot_3d = geom_3d["z_total_m"]
    z_act_start_3d = geom_3d["z_active_start_m"]
    z_act_end_3d = geom_3d["z_active_end_m"]

    x_ax_3d, y_ax_3d, z_ax_3d = build_3d_marosi_axes(
        spec_3d,
        z_total_m=z_tot_3d,
        nx=20,
        n_feat_y=4,
        nz_per_period=20,
        use_graded_mesh=True,
    )
    grid_3d = GridFIT3D(
        x=x_ax_3d,
        y=y_ax_3d,
        z=z_ax_3d,
        stl_solids={"pillars": geom_3d["stl_path"]},
        stl_materials={"pillars": "pec"},
        conformal="pec",
        verbose=0,
    )
    solver_3d = SolverFIT3D(
        grid_3d,
        use_stl=True,
        bc_low=["pec", "pec", "cpml"],
        bc_high=["pec", "pec", "cpml"],
        n_pml=8,
        verbose=0,
    )
    Path(geom_3d["stl_path"]).unlink(missing_ok=True)
    grid_3d_bare = GridFIT3D(x=x_ax_3d, y=y_ax_3d, z=z_ax_3d, verbose=0)

    structs_3d: list[Any] = []
    for k in range(n_tot_3d):
        sc = (
            taper_3d[k]
            if k < n_tap_3d
            else (
                taper_3d[n_tot_3d - 1 - k]
                if k >= n_tap_3d + n_act_3d
                else 1.0
            )
        )
        wk = sc * w3
        z0 = z_buf_3d + k * d3
        z1 = z0 + 0.25 * d3
        z2 = z0 + 0.75 * d3
        for y_sign in (-1.0, 1.0):
            p1 = Box(
                center=(-0.5 * a3 + 0.5 * wk, y_sign * yc_3d, z1),
                size=(wk, t3, l3),
                material="pec",
            )
            p1.name = "−x PEC pillar pair (z₁ = d/4)"
            p2 = Box(
                center=(0.5 * a3 - 0.5 * wk, y_sign * yc_3d, z2),
                size=(wk, t3, l3),
                material="pec",
            )
            p2.name = "+x PEC pillar pair (z₂ = 3d/4, glide +d/2)"
            structs_3d.extend([p1, p2])

    proxy_3d = _SWSSchematicSolverProxy(solver_3d, grid_3d_bare, tuple(structs_3d))
    pulse_3d = GaussianSource(frequency=65.17e9, fwidth=28.0e9)
    sources_3d = [
        {
            "kind": "line",
            "type_name": "Beam",
            "short_word": "5.2 kV bunch",
            "label": "Rigid electron bunch (V₀ = 5.2 kV, u₀ = 0.1416c, q = 1 pC, σ_z = 350 µm)",
            "desc": "Rigid electron bunch J_z(z − u₀ t) along (x=0, y=0) (5.2 kV, u₀ = 0.1416c)",
            "pos_str": "(x, y) = (0, 0), z ∈ [0, L_tot]",
            "polarization": "Jz (3D hybrid glide mode)",
            "h_bounds": (z_buf_3d, z_tot_3d - z_buf_3d),
            "v_bounds": (0.0, 0.0),
            "pulse": {
                "f0": 65.17e9,
                "fwidth": 28.0e9,
                "t_peak": float(4.2 * 0.35e-3 / u0_3d),
                "t_end": float(8.4 * 0.35e-3 / u0_3d),
                "src_time": pulse_3d,
            },
        }
    ]
    monitors_3d = [
        {
            "kind": "line",
            "type_name": "WakeSolver",
            "short_word": "wake E_z(s)",
            "label": "3D longitudinal wake integration line (N = 6, L = 6.30 mm) → W_∥(s), Re Z_∥(f)",
            "desc": "Wake integration line E_z(x=0, y=0, z, t) over active 3D Marosi SWS",
            "pos_str": f"(x, y) = (0, 0), z ∈ [{z_act_start_3d*1e3:.2f}, {z_act_end_3d*1e3:.2f}] mm",
            "freq_str": "45–80 GHz (V-band forward passband, f_s = 65.17 GHz)",
            "h_bounds": (z_act_start_3d, z_act_end_3d),
            "v_bounds": (-0.05e-3, 0.05e-3),
        }
    ]
    mat_names_3d = {
        "background": "Vacuum V-band enclosure & sheet-beam tunnel (ε_r = 1.00, 1.40 × 0.80 mm²)",
    }
    res_3d = schematic.save_setup_figures(
        proxy_3d,
        figures_dir,
        prefix="00",
        suffix="_marosi",
        plane="zy",
        cut_coord=0.0,
        material_names=mat_names_3d,
        sources=sources_3d,
        monitors=monitors_3d,
        unit="mm",
        freq=65.17e9,
        min_feature=t3,
        variations=[
            {
                "name": "3D V-Band Staggered-Pillar Glide SWS (Marosi 2025)",
                "details": "Housing a = 1.40 mm × b = 0.80 mm, period d = 1.05 mm, pillars w = 1.08 mm × l = 0.27 mm × t = 0.10 mm, 5.2 kV beam (u₀ = 0.1416c, v_g = +0.1306c)",
            }
        ],
    )

    return {
        "fig_00a_2d": str(res_2d["geometry_figure"]),
        "fig_00b_2d": str(res_2d["mesh_figure"]),
        "table_2d": str(res_2d["table_path"]),
        "fig_00a_3d": str(res_3d["geometry_figure"]),
        "fig_00b_3d": str(res_3d["mesh_figure"]),
        "table_3d": str(res_3d["table_path"]),
    }


def _overlay_2d_grating_geometry(
    ax: plt.Axes,
    spec: KaretnikovaGratingSpec,
    *,
    n_periods: int = 10,
    symmetry: str = "glide",
    z_in_mm: float | None = None,
    z_out_mm: float | None = None,
    show_legend: bool = False,
) -> None:
    """Overlay PEC vanes, CPML boundaries, beam channel, and Poynting port planes on a (z, y) field map."""
    from matplotlib.patches import Rectangle

    d_mm = spec.period_m * 1e3
    l_mm = spec.vane_thickness_m * 1e3
    h_mm = spec.vane_height_m * 1e3
    a_mm = spec.half_tunnel_m * 1e3
    y_wall_mm = a_mm + h_mm
    taper_scales = [0.25, 0.55]
    n_tap = len(taper_scales)
    n_tot = int(n_periods) + 2 * n_tap
    z_buf_mm = 1.5 * d_mm
    z_tot_mm = 2.0 * z_buf_mm + n_tot * d_mm

    for k in range(n_tot):
        if k < n_tap:
            sc = taper_scales[k]
        elif k >= n_tap + int(n_periods):
            sc = taper_scales[n_tot - 1 - k]
        else:
            sc = 1.0
        hk_mm = sc * h_mm
        z0 = z_buf_mm + k * d_mm
        z_bot = z0 + 0.25 * d_mm
        z_top = (z0 + 0.75 * d_mm) if symmetry == "glide" else (z0 + 0.25 * d_mm)
        r_bot = Rectangle(
            (z_bot - 0.5 * l_mm, -y_wall_mm),
            l_mm,
            hk_mm,
            facecolor="#2b2b2b",
            edgecolor="black",
            lw=0.8,
            zorder=4,
            label="PEC vanes (staggered/mirror)" if (k == n_tap and show_legend) else None,
        )
        r_top = Rectangle(
            (z_top - 0.5 * l_mm, y_wall_mm - hk_mm),
            l_mm,
            hk_mm,
            facecolor="#2b2b2b",
            edgecolor="black",
            lw=0.8,
            zorder=4,
        )
        ax.add_patch(r_bot)
        ax.add_patch(r_top)

    # CPML shading (10 cells * 25 um = 0.25 mm at each end)
    pml_w_mm = 0.25
    ax.axvspan(
        0.0,
        pml_w_mm,
        color="#7b2cbf",
        alpha=0.18,
        zorder=3,
        label="CPML absorber" if show_legend else None,
    )
    ax.axvspan(z_tot_mm - pml_w_mm, z_tot_mm, color="#7b2cbf", alpha=0.18, zorder=3)

    # Sheet-beam channel boundaries |y| <= 75 um
    hb_half_mm = 0.5 * spec.beam_thickness_m * 1e3
    ax.axhline(
        hb_half_mm,
        color="#495057",
        ls=":",
        lw=0.9,
        zorder=4,
        label="Sheet-beam envelope (H_b = 150 µm)" if show_legend else None,
    )
    ax.axhline(-hb_half_mm, color="#495057", ls=":", lw=0.9, zorder=4)

    if z_in_mm is not None:
        ax.axvline(
            z_in_mm,
            color="#d9480f",
            ls="--",
            lw=1.3,
            zorder=5,
            label="Upstream port z_in" if show_legend else None,
        )
    if z_out_mm is not None:
        ax.axvline(
            z_out_mm,
            color="#0b7285",
            ls="--",
            lw=1.3,
            zorder=5,
            label="Downstream port z_out" if show_legend else None,
        )


def generate_all_tutorial_figures(
    data_dir: Path | str, figures_dir: Path | str
) -> dict[str, str]:
    """Generate figures 01 through 09 (including GIF 06) from the saved JSON/NPZ datasets."""
    from PIL import Image

    data_dir = Path(data_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()

    spec_2d = KaretnikovaGratingSpec()
    bench = load_cold_test_benchmarks()
    l1 = bench["levels_2d"]["L1"]
    l2 = bench["levels_2d"]["L2"]
    u0_2d = bench["u0_2d_m_s"]

    with open(data_dir / "task_01_termination_reflection.json", encoding="utf-8") as fh:
        t1 = json.load(fh)
    with open(data_dir / "task_02_2d_length_sweep.json", encoding="utf-8") as fh:
        t2 = json.load(fh)
    z2 = np.load(data_dir / "task_02_2d_length_sweep_spectra.npz")
    with open(data_dir / "task_03_fwd_vs_bwd_poynting.json", encoding="utf-8") as fh:
        t3 = json.load(fh)
    z3 = np.load(data_dir / "task_03_fwd_vs_bwd_snapshots.npz")
    with open(data_dir / "task_04_mutations_detuning_and_cavity.json", encoding="utf-8") as fh:
        t4 = json.load(fh)
    z4 = np.load(data_dir / "task_04_mutations_spectra.npz")
    with open(data_dir / "task_05_mesh_convergence.json", encoding="utf-8") as fh:
        t5 = json.load(fh)
    with open(data_dir / "task_06_3d_marosi_beam_wake.json", encoding="utf-8") as fh:
        t6 = json.load(fh)
    z6 = np.load(data_dir / "task_06_3d_marosi_spectrum.npz")

    out_paths: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Figure 01: Port Termination Reflection |Gamma(f)| and Error Bound
    # ------------------------------------------------------------------
    f1 = np.array(t1["f_ghz"])
    g_tap = np.array(t1["gamma_tapered"])
    g_abr = np.array(t1["gamma_abrupt"])
    g_pec = np.array(t1["gamma_pec_short"])
    f_s1 = t1["f_sync_ghz"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1.plot(f1, g_pec, color="#495057", ls=":", lw=1.6, label=r"Shorted PEC end wall ($|\Gamma| = 1.00$)")
    ax1.plot(
        f1,
        g_abr,
        color="#d9480f",
        ls="--",
        lw=1.9,
        label=rf"Abrupt step to smooth guide + CPML ($|\Gamma(f_s)| = {t1['gamma_abrupt_at_sync']:.3f}$)",
    )
    ax1.plot(
        f1,
        g_tap,
        color="#0b7285",
        lw=2.2,
        label=rf"2-stage vane taper $[0.25h, 0.55h]$ + CPML ($|\Gamma(f_s)| = {t1['gamma_tapered_at_sync']:.3f}$)",
    )
    ax1.axvline(f_s1, color="#2b8a3e", ls="-.", lw=1.3, label=rf"Synchronism $f_s = {f_s1:.1f}\;\mathrm{{GHz}}$")
    ax1.scatter([f_s1, f_s1], [t1["gamma_tapered_at_sync"], t1["gamma_abrupt_at_sync"]], color=["#0b7285", "#d9480f"], zorder=5, s=42)
    ax1.set_xlabel("Frequency $f$ [GHz]")
    ax1.set_ylabel(r"Port reflection magnitude $|\Gamma(f)|$ [—]")
    ax1.set_title("(a) SWS Port Reflectometry in Wakis FIT")
    ax1.set_xlim(95.0, 135.0)
    ax1.set_ylim(0.0, 1.08)
    ax1.legend(loc="upper right", fontsize=8.3)

    err_tap_2x = 100.0 * reflection_induced_peak_error_bound(g_tap)
    err_tap_1x = 50.0 * reflection_induced_peak_error_bound(g_tap)
    err_abr_2x = 100.0 * reflection_induced_peak_error_bound(g_abr)
    ax2.plot(
        f1,
        err_abr_2x,
        color="#d9480f",
        ls="--",
        lw=1.8,
        label=rf"Abrupt transition bound $2|\Gamma|/(1-|\Gamma|^2)$ ($f_s$: {100*t1['rel_error_bound_abrupt']:.1f}%)",
    )
    ax2.plot(
        f1,
        err_tap_2x,
        color="#0b7285",
        lw=2.2,
        label=rf"Tapered transition bound $2|\Gamma|/(1-|\Gamma|^2)$ ($f_s$: {100*t1['rel_error_bound_tapered']:.1f}%)",
    )
    ax2.plot(
        f1,
        err_tap_1x,
        color="#868e96",
        ls=":",
        lw=1.6,
        label=r"Mutated bound $|\Gamma|/(1-|\Gamma|^2)$ (drops round-trip factor 2)",
    )
    ax2.axvline(f_s1, color="#2b8a3e", ls="-.", lw=1.3)
    ax2.set_xlabel("Frequency $f$ [GHz]")
    ax2.set_ylabel(r"Peak $\mathrm{Re}\,Z_\parallel(f_s)$ error bound [%]")
    ax2.set_title(r"(b) Fabry–Pérot Impedance Ripple Bound (derived here)")
    ax2.set_xlim(95.0, 135.0)
    ax2.set_ylim(0.0, 105.0)
    ax2.legend(loc="upper right", fontsize=8.3)

    fig.tight_layout()
    p01 = figures_dir / "01_termination_reflection_gamma.png"
    fig.savefig(p01, dpi=220)
    plt.close(fig)
    out_paths["fig_01"] = str(p01)

    # ------------------------------------------------------------------
    # Figure 02: Re Z_||(f) vs Analytic sinc^2 and Log-Log L^2 & 1/L Scaling
    # ------------------------------------------------------------------
    n_keys = list(t2["runs"].keys())
    palette_n = ["#5c7cfa", "#0b7285", "#2b8a3e", "#d9480f"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for idx_n, n_str in enumerate(n_keys):
        r_n = t2["runs"][n_str]
        L_n = r_n["active_length_m"]
        f_sw = z2[f"f_ghz_N{n_str}"]
        mask_sw = (f_sw >= 96.0) & (f_sw <= 132.0)
        f_sub = f_sw[mask_sw]
        rz_fit = z2[f"re_z_ohm_N{n_str}"][mask_sw] * 1e-3
        rz_th = (
            analytic_sws_beam_impedance(
                f_sub * 1e9,
                f_sync_hz=l1["f_sync_bwd_hz"],
                beta_n_rad_m=l1["beta_m1_rad_m"],
                k_pierce_ohm=l1["k_m1_on_axis_ohm"],
                length_m=L_n,
                u0_m_s=u0_2d,
                vg_m_s=l1["vg_bwd_m_s"],
            )
            * 1e-3
        )
        col = palette_n[idx_n % len(palette_n)]
        ax1.plot(
            f_sub,
            rz_fit,
            color=col,
            lw=2.1,
            marker="none",
            label=rf"FIT $N={n_str}$ ($L={L_n*1e3:.1f}\;\mathrm{{mm}}, Z_{{\mathrm{{pk}}}}={r_n['re_z_pk_ohm']*1e-3:.2f}\;\mathrm{{k\Omega}}$)",
        )
        ax1.plot(
            f_sub,
            rz_th,
            color=col,
            ls="--",
            lw=1.5,
            marker="none",
            alpha=0.9,
            label="Cold-test $\\mathrm{sinc}^2$ (no fit)" if idx_n == len(n_keys) - 1 else None,
        )
    ax1.set_xlabel("Frequency $f$ [GHz]")
    ax1.set_ylabel(r"Beam–wave resistance $\mathrm{Re}\,Z_\parallel(f)$ [$\mathrm{k\Omega}$]")
    ax1.set_title(r"(a) $\mathrm{Re}\,Z_\parallel(f)$ vs Cold-Test $\frac{1}{4}\beta_{-1}^2 K_{-1} L^2\,\mathrm{sinc}^2(\Delta\beta L/2)$")
    ax1.set_xlim(98.0, 130.0)
    ax1.legend(loc="upper right", fontsize=8.0)

    L_mm_arr = np.array([t2["runs"][k]["active_length_mm"] for k in n_keys])
    z_pk_arr = np.array([t2["runs"][k]["re_z_pk_ohm"] for k in n_keys])
    fwhm_arr = np.array([t2["runs"][k]["fwhm_ghz"] for k in n_keys])
    z_ref_guide = z_pk_arr[0] * ((L_mm_arr / L_mm_arr[0]) ** 2)
    w_ref_guide = fwhm_arr[0] * ((L_mm_arr / L_mm_arr[0]) ** (-1))

    ax2_r = ax2.twinx()
    p1_l = ax2.loglog(
        L_mm_arr,
        z_pk_arr,
        "o-",
        color="#0b7285",
        lw=2.1,
        ms=6.5,
        label=rf"FIT peak $\mathrm{{Re}}\,Z_\parallel(f_s)$ (slope $= {t2['power_law_slope_peak_vs_L']:.3f}$)",
    )
    p2_l = ax2.loglog(
        L_mm_arr,
        z_ref_guide,
        "--",
        color="#0b7285",
        lw=1.3,
        alpha=0.7,
        label=r"Exact $O(L^2)$ slope $+2.000$ guide",
    )
    p3_l = ax2_r.loglog(
        L_mm_arr,
        fwhm_arr,
        "s-",
        color="#d9480f",
        lw=2.1,
        ms=6.5,
        label=rf"FIT linewidth $\Delta f_{{\mathrm{{FWHM}}}}$ (slope $= {t2['power_law_slope_fwhm_vs_L']:.3f}$)",
    )
    p4_l = ax2_r.loglog(
        L_mm_arr,
        w_ref_guide,
        ":",
        color="#d9480f",
        lw=1.4,
        alpha=0.75,
        label=r"Exact $O(L^{-1})$ slope $-1.000$ guide",
    )
    ax2.set_xlabel("Active interaction length $L = N d$ [mm]")
    ax2.set_ylabel(r"Peak resistance $\mathrm{Re}\,Z_\parallel(f_s)$ [$\Omega$]", color="#0b7285")
    ax2_r.set_ylabel(r"Linewidth $\Delta f_{\mathrm{FWHM}}$ [GHz]", color="#d9480f")
    ax2.set_title(r"(b) Log–Log Scaling of Peak ($\propto L^2$) and Width ($\propto L^{-1}$)")
    lines_all = p1_l + p2_l + p3_l + p4_l
    labels_all = [l.get_label() for l in lines_all]
    ax2.legend(lines_all, labels_all, loc="center left", fontsize=7.8)

    fig.tight_layout()
    p02 = figures_dir / "02_re_z_vs_sinc2_length_sweep.png"
    fig.savefig(p02, dpi=220)
    plt.close(fig)
    out_paths["fig_02"] = str(p02)

    # ------------------------------------------------------------------
    # Figure 03: Extracted Pierce K_{-1} vs Cold Test & Transverse Profile
    # ------------------------------------------------------------------
    n_vals = np.array([int(k) for k in n_keys])
    k_ax_fit = np.array([t2["runs"][str(n)]["k_on_axis_beam_ohm"] for n in n_vals])
    k_av_fit = np.array([t2["runs"][str(n)]["k_sheet_avg_beam_ohm"] for n in n_vals])
    k_ax_mut = 0.25 * k_ax_fit
    y_sweep = t2["sheet_beam_y_sweep"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax1.plot(
        [y_sweep["n_periods"]],
        [y_sweep["k_sheet_avg_7node_mean_ohm"]],
        "D",
        color="#d9480f",
        mec="black",
        ms=8.0,
        zorder=6,
        label=rf"Measured 7-node sheet avg $\langle K_{{-1}}\rangle_{{\mathrm{{nodes}}}} = {y_sweep['k_sheet_avg_7node_mean_ohm']:.2f}\;\Omega$ ($N={y_sweep['n_periods']}$)",
    )
    ax1.plot(
        n_vals,
        k_av_fit,
        "s-",
        color="#d9480f",
        lw=1.9,
        ms=5.8,
        alpha=0.85,
        label=rf"Continuum $K_{{-1}}(0)\eta_{{\mathrm{{sheet}}}}$ ($N={y_sweep['n_periods']}$: {y_sweep['k_sheet_avg_analytic_cosh_ohm']:.2f} $\Omega$, trapz: {y_sweep['k_sheet_avg_trapz_ohm']:.2f} $\Omega$)",
    )
    ax1.axhline(
        l1["k_m1_beam_avg_ohm"],
        color="#d9480f",
        ls="--",
        lw=1.4,
        label=rf"6.2 Cold test $\langle K_{{-1}}\rangle_{{\mathrm{{nodes}}}}$ (L1: {l1['k_m1_beam_avg_ohm']:.2f} $\Omega$, L2: {l2['k_m1_beam_avg_ohm']:.2f} $\Omega$)",
    )
    ax1.axhline(l2["k_m1_beam_avg_ohm"], color="#d9480f", ls=":", lw=1.3)

    ax1.plot(
        n_vals,
        k_ax_fit,
        "o-",
        color="#0b7285",
        lw=2.1,
        ms=6.5,
        label=rf"Beam wake on-axis $K_{{-1}}(0)$ ($N={y_sweep['n_periods']}$: {y_sweep['k_on_axis_ohm']:.2f} $\Omega$; $N={t5['n_periods']}$ finest band: $[{t5['k_on_axis_finest_band_min_ohm']:.2f}, {t5['k_on_axis_finest_band_max_ohm']:.2f}]\;\Omega$)",
    )
    ax1.axhline(
        l1["k_m1_on_axis_ohm"],
        color="#0b7285",
        ls="--",
        lw=1.4,
        label=rf"6.2 Cold test $K_{{-1}}(0)$ (L1: {l1['k_m1_on_axis_ohm']:.2f} $\Omega$, L2: {l2['k_m1_on_axis_ohm']:.2f} $\Omega$)",
    )
    ax1.axhline(l2["k_m1_on_axis_ohm"], color="#0b7285", ls=":", lw=1.3)

    ax1.plot(
        n_vals,
        k_ax_mut,
        "^-.",
        color="#868e96",
        lw=1.5,
        ms=5.5,
        label=r"Mutated inversion $\mathrm{Re}\,Z_\parallel / (\beta_{-1}^2 L^2)$ (drops factor 4)",
    )
    ax1.set_xlabel("Number of grating periods $N$ [—]")
    ax1.set_ylabel(r"Pierce interaction impedance $K_{-1}$ [$\Omega$]")
    ax1.set_title(r"(a) Beam-Extracted $K_{-1}$ vs Tutorial 6.2 Bloch Cold Test")
    ax1.set_ylim(0.0, 4.15)
    ax1.legend(loc="lower right", fontsize=7.3)

    y_um = np.linspace(-100.0, 100.0, 401)
    beta_m1 = l2["beta_m1_rad_m"]
    k0_s = 2.0 * np.pi * l2["f_sync_bwd_hz"] / c
    tau_m1 = np.sqrt(beta_m1**2 - k0_s**2)
    prof_cosh2 = np.cosh(tau_m1 * (y_um * 1e-6)) ** 2
    eta_sh = l2["eta_sheet"]
    y_nodes_um = np.array(y_sweep["y_full_nodes_um"], dtype=float)
    k_nodes_ratio = np.array(y_sweep["k_m1_full_nodes_ohm"], dtype=float) / max(
        float(y_sweep["k_on_axis_ohm"]), 1e-30
    )

    ax2.plot(
        y_um,
        prof_cosh2,
        color="#0b7285",
        lw=2.2,
        label=rf"Evanescent harmonic profile $\cosh^2(\tau_{{-1}} y)$ ($\tau_{{-1}} = {tau_m1*1e-3:.2f}\;\mathrm{{mm^{{-1}}}}$)",
    )
    ax2.plot(
        y_nodes_um,
        k_nodes_ratio,
        "o",
        color="#2b8a3e",
        mec="black",
        mew=1.0,
        ms=6.8,
        zorder=6,
        label=rf"FIT off-axis beam runs $K_{{-1}}(y_0)/K_{{-1}}(0)$ (7-node mean $= {y_sweep['ratio_7node_mean_over_axis']:.3f}$ vs 6.2 L1 ${y_sweep['cold_test_L1_ratio_over_axis']:.3f}$)",
    )
    ax2.axvspan(
        -75.0,
        75.0,
        color="#ffd43b",
        alpha=0.28,
        label=r"Sheet beam thickness $H_b = 150\;\mu\mathrm{m}$ ($|y| \leq 75\;\mu\mathrm{m}$)",
    )
    ax2.axhline(
        y_sweep["ratio_7node_mean_over_axis"],
        color="#2b8a3e",
        ls="-.",
        lw=1.6,
        label=rf"7-node equal-weight mean $= {y_sweep['ratio_7node_mean_over_axis']:.3f}$ (trapz $= {y_sweep['ratio_trapz_over_axis']:.3f}$)",
    )
    ax2.axhline(
        eta_sh,
        color="#d9480f",
        ls="--",
        lw=1.6,
        label=rf"Continuum integral $\eta_{{\mathrm{{sheet}}}} = \frac{{1}}{{2}}[1 + \frac{{\sinh(\tau_{{-1}} H_b)}}{{\tau_{{-1}} H_b}}] = {eta_sh:.3f}$",
    )
    ax2.axhline(1.0, color="#495057", ls=":", lw=1.2)
    ax2.set_xlabel(r"Transverse tunnel coordinate $y$ [$\mu\mathrm{m}$]")
    ax2.set_ylabel(r"Normalized coupling $K_{-1}(y_0) / K_{-1}(0)$ [—]")
    ax2.set_title(r"(b) Measured Off-Axis Beam Profile vs $\cosh^2(\tau_{-1} y)$")
    ax2.set_xlim(-100.0, 100.0)
    ax2.set_ylim(0.95, 1.98)
    ax2.legend(loc="upper center", fontsize=7.4)

    fig.tight_layout()
    p03 = figures_dir / "03_pierce_k_and_sheet_beam_profile.png"
    fig.savefig(p03, dpi=220)
    plt.close(fig)
    out_paths["fig_03"] = str(p03)

    # ------------------------------------------------------------------
    # Figure 04: Asymptotic Stopping Power dW/dz & Bunch Form Factor k_||(sigma_z)
    # ------------------------------------------------------------------
    dw_dz_unatt = np.array([t2["runs"][str(n)]["dw_dz_unattenuated_j_per_m"] * 1e9 for n in n_vals])
    dw_dz_att = np.array([t2["runs"][str(n)]["dw_dz_fit_j_per_m"] * 1e9 for n in n_vals])
    dw_th_unatt = t2["dw_dz_analytic_unattenuated_j_per_m"] * 1e9
    dw_th_mut_2x = 2.0 * dw_th_unatt
    idx_ref_n = min(2, len(n_vals) - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax1.plot(
        L_mm_arr,
        dw_dz_unatt,
        "o-",
        color="#0b7285",
        lw=2.2,
        ms=6.5,
        label=rf"FIT unattenuated $\mathrm{{d}}W/\mathrm{{d}}z / |F(\omega_s)|^2$ ($N={n_vals[idx_ref_n]}$: {dw_dz_unatt[idx_ref_n]:.2f} nJ/m)",
    )
    ax1.axhline(
        dw_th_unatt,
        color="#2b8a3e",
        ls="--",
        lw=1.9,
        label=rf"Asymptotic $\frac{{q^2 \beta_{{-1}}^2 K_{{-1}}}}{{2|1/u_0 - 1/v_g|}} = {dw_th_unatt:.2f}\;\mathrm{{nJ/m}}$ (derived here)",
    )
    ax1.axhline(
        dw_th_mut_2x,
        color="#868e96",
        ls=":",
        lw=1.5,
        label=rf"Mutated formula without $1/2$ factor ({dw_th_mut_2x:.2f} nJ/m)",
    )
    ax1.plot(
        L_mm_arr,
        dw_dz_att * 10.0,
        "s-.",
        color="#d9480f",
        lw=1.8,
        ms=5.5,
        label=rf"Gaussian bunch ($\sigma_z = 220\;\mu\mathrm{{m}}$) $10 \times \mathrm{{d}}W/\mathrm{{d}}z$",
    )
    ax1.set_xlabel("Active structure length $L = N d$ [mm]")
    ax1.set_ylabel("Stopping power $\\mathrm{d}W/\\mathrm{d}z$ [nJ/m] ($q = 1\\;\\mathrm{pC}$)")
    ax1.set_title(r"(a) Infinite-Structure Stopping Power Limit ($L \to \infty$)")
    ax1.set_ylim(0.0, 13.5)
    ax1.legend(loc="center right", fontsize=7.8)

    sig_mm = np.array(t2["sigmaz_sweep_mm"])
    sig_um = sig_mm * 1e3
    k_fit_s = np.array(t2["k_loss_fit_N20_v_per_pc"])
    k_th_s = np.array(t2["k_loss_analytic_N20_v_per_pc"])
    ff_curve = gaussian_bunch_form_factor_sq(l2["f_sync_bwd_hz"], sig_mm * 1e-3, u0_2d)

    ax2_r = ax2.twinx()
    l_k1 = ax2.semilogy(
        sig_um,
        k_fit_s,
        "o-",
        color="#0b7285",
        lw=2.2,
        ms=6.0,
        label=r"FIT loss factor $k_\parallel(\sigma_z)$ ($N = 20$)",
    )
    l_k2 = ax2.semilogy(
        sig_um,
        k_th_s,
        "--",
        color="#2b8a3e",
        lw=1.8,
        label=r"Cold-test $\frac{1}{\pi}\int \mathrm{Re}\,Z_\parallel(\omega)|F(\omega)|^2\,\mathrm{d}\omega$",
    )
    l_k3 = ax2_r.semilogy(
        sig_um,
        ff_curve,
        "s:",
        color="#d9480f",
        lw=1.8,
        ms=5.0,
        label=r"Bunch form factor $|F(\omega_s)|^2 = e^{-(\omega_s \sigma_z / u_0)^2}$",
    )
    ax2.set_xlabel(r"RMS Gaussian bunch length $\sigma_z$ [$\mu\mathrm{m}$]")
    ax2.set_ylabel(r"Loss factor $k_\parallel(\sigma_z)$ [V/pC]", color="#0b7285")
    ax2_r.set_ylabel(r"Form factor $|F(\omega_s)|^2$ [—]", color="#d9480f")
    ax2.set_title(r"(b) Loss Factor $k_\parallel(\sigma_z)$ & Coherent Bunch Form Factor")
    l_all = l_k1 + l_k2 + l_k3
    ax2.legend(l_all, [x.get_label() for x in l_all], loc="lower left", fontsize=7.9)

    fig.tight_layout()
    p04 = figures_dir / "04_stopping_power_and_loss_factor.png"
    fig.savefig(p04, dpi=220)
    plt.close(fig)
    out_paths["fig_04"] = str(p04)

    # ------------------------------------------------------------------
    # Figure 05: Backward vs Forward Wave Poynting Flux & Linewidth
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))
    t_bwd = z3["t_bwd_ps"]
    p_in_bwd = z3["p_in_bwd_w"] * 1e3
    p_out_bwd = z3["p_out_bwd_w"] * 1e3
    t_fwd = z3["t_fwd_ps"]
    p_in_fwd = z3["p_in_fwd_w"] * 1e3
    p_out_fwd = z3["p_out_fwd_w"] * 1e3

    bwd_info = t3["backward_wave_glide_m1"]
    fwd_info = t3["forward_wave_mirror_n0"]
    e_in_bwd_pj = bwd_info["integrated_energy_in_j"] * 1e12
    e_out_bwd_pj = bwd_info["integrated_energy_out_j"] * 1e12
    e_rad_bwd_pj = bwd_info["total_radiated_cpml_energy_j"] * 1e12
    w_bwd_pj = bwd_info["w_loss_between_smooth_ports_j"] * 1e12
    e_in_fwd_pj = fwd_info["integrated_energy_in_j"] * 1e12
    e_out_fwd_pj = fwd_info["integrated_energy_out_j"] * 1e12
    e_rad_fwd_pj = fwd_info["total_radiated_cpml_energy_j"] * 1e12
    w_fwd_pj = fwd_info["w_loss_between_smooth_ports_j"] * 1e12

    ax1.plot(
        t_bwd,
        p_in_bwd,
        color="#d9480f",
        lw=2.0,
        marker="none",
        label=rf"BWD ($v_g<0$) $P_z(z_{{\mathrm{{in}}}})$ ($E_{{\mathrm{{in}}}}={e_in_bwd_pj:.2f}\;\mathrm{{pJ}}, E_{{\mathrm{{rad}}}}={e_rad_bwd_pj:.2f}$ vs $W_{{\mathrm{{loss}}}}={w_bwd_pj:.2f}\;\mathrm{{pJ}}$)",
    )
    ax1.plot(
        t_bwd,
        p_out_bwd,
        color="#d9480f",
        ls=":",
        lw=1.3,
        marker="none",
        alpha=0.75,
        label=rf"BWD ($v_g<0$) $P_z(z_{{\mathrm{{out}}}})$ ($E_{{\mathrm{{out}}}}=+{e_out_bwd_pj:.2f}\;\mathrm{{pJ}}$)",
    )
    ax1.plot(
        t_fwd,
        p_out_fwd * 0.25,
        color="#0b7285",
        lw=2.0,
        marker="none",
        label=rf"FWD ($v_g>0$) $0.25 P_z(z_{{\mathrm{{out}}}})$ ($E_{{\mathrm{{out}}}}=+{e_out_fwd_pj:.2f}\;\mathrm{{pJ}}, E_{{\mathrm{{rad}}}}={e_rad_fwd_pj:.2f}$ vs $W_{{\mathrm{{loss}}}}={w_fwd_pj:.2f}\;\mathrm{{pJ}}$)",
    )
    ax1.plot(
        t_fwd,
        p_in_fwd * 0.25,
        color="#0b7285",
        ls=":",
        lw=1.2,
        marker="none",
        alpha=0.7,
        label=rf"FWD ($v_g>0$) $0.25 P_z(z_{{\mathrm{{in}}}})$ ($E_{{\mathrm{{in}}}}={e_in_fwd_pj:.2f}\;\mathrm{{pJ}}$)",
    )
    ax1.axhline(0.0, color="#495057", lw=0.9)
    ax1.set_xlabel("Simulation time $t$ [ps]")
    ax1.set_ylabel("Full-section smooth-guide Poynting power $P_z(t)$ [mW]")
    ax1.set_title("(a) Smooth-Guide Full-Section Poynting Flux & Energy Balance")
    ax1.set_xlim(15.0, float(max(t_bwd[-1], min(t_fwd[-1], 220.0))))
    ax1.legend(loc="upper right", fontsize=7.2)

    f_b_arr = z3["f_bwd_ghz"]
    rz_b_arr = z3["re_z_bwd_ohm"]
    fpk_b = bwd_info["f_pk_ghz"]
    zpk_b = bwd_info["re_z_pk_ohm"]
    fwhm_b = bwd_info["fwhm_ghz"]

    f_f_arr = z3["f_fwd_ghz"]
    rz_f_arr = z3["re_z_fwd_ohm"]
    fpk_f = fwd_info["f_pk_ghz"]
    zpk_f = fwd_info["re_z_pk_ohm"]
    fwhm_f = fwd_info["fwhm_ghz"]

    df_grid = np.linspace(-25.0, 25.0, 501)
    rz_mut_flipped = analytic_sws_beam_impedance(
        (fpk_b + df_grid) * 1e9,
        f_sync_hz=fpk_b * 1e9,
        beta_n_rad_m=l2["beta_m1_rad_m"],
        k_pierce_ohm=1.0,
        length_m=5.0e-3,
        u0_m_s=u0_2d,
        vg_m_s=-l2["vg_bwd_m_s"],
    )
    rz_mut_flipped /= np.max(rz_mut_flipped)
    fwhm_mut_b = bwd_info["fwhm_mutated_flipped_vg_ghz"]

    mir_cold = l1["mirror_fwd_n0"]
    f_nl_cold = np.array(mir_cold["f_passband_ghz"], dtype=float)
    rz_nl_cold = np.array(mir_cold["re_z_nonlinear_N10_passband_ohm"], dtype=float)

    mb3 = np.abs(f_b_arr - fpk_b) <= 25.0
    mf3 = np.abs(f_f_arr - fpk_f) <= 25.0
    ax2.plot(
        f_b_arr[mb3] - fpk_b,
        rz_b_arr[mb3] / zpk_b,
        color="#d9480f",
        lw=2.2,
        marker="none",
        label=rf"BWD ($v_g = -0.54c < 0$): $|1/u_0 - 1/v_g| = 1/u_0 + 1/|v_g|$, FWHM $= {fwhm_b:.2f}\;\mathrm{{GHz}}$",
    )
    ax2.plot(
        f_f_arr[mf3] - fpk_f,
        rz_f_arr[mf3] / zpk_f,
        color="#0b7285",
        lw=2.2,
        marker="none",
        label=rf"FWD ($v_g = +0.33c > 0, y_0=50\;\mu\mathrm{{m}}$): FIT FWHM $= {fwhm_f:.2f}\;\mathrm{{GHz}}$ ($Z_{{\mathrm{{pk}}}}={zpk_f:.0f}\;\Omega$)",
    )
    if len(f_nl_cold) > 5:
        ax2.plot(
            f_nl_cold - mir_cold["f_pk_nonlinear_N10_ghz"],
            rz_nl_cold / max(float(np.max(rz_nl_cold)), 1e-12),
            color="#2b8a3e",
            ls="-.",
            lw=1.7,
            marker="none",
            label=rf"6.2 Cold-test FWD product ($y_0=50\;\mu\mathrm{{m}}, Z_{{\mathrm{{pk}}}}={mir_cold['re_z_pk_nonlinear_N10_ohm']:.0f}\;\Omega$, stopband cutoff)",
        )
    ax2.plot(
        df_grid,
        rz_mut_flipped,
        color="#868e96",
        ls="--",
        lw=1.5,
        marker="none",
        label=rf"Mutated BWD with $+|v_g|$ sign error: FWHM $= {fwhm_mut_b:.2f}\;\mathrm{{GHz}}$",
    )
    ax2.axhline(0.5, color="#495057", ls=":", lw=1.0)
    ax2.set_xlabel(r"Frequency detuning $f - f_{\mathrm{pk}}$ [GHz]")
    ax2.set_ylabel(r"Normalized resistance $\mathrm{Re}\,Z_\parallel(f) / \mathrm{Re}\,Z_{\mathrm{pk}}$ [—]")
    ax2.set_title(r"(b) Backward vs Forward Linewidth & Bragg Stopband Cutoff")
    ax2.set_xlim(-25.0, 25.0)
    ax2.set_ylim(-0.08, 1.15)
    ax2.legend(loc="upper right", fontsize=7.2)

    fig.tight_layout()
    p05 = figures_dir / "05_forward_vs_backward_poynting_and_width.png"
    fig.savefig(p05, dpi=220)
    plt.close(fig)
    out_paths["fig_05"] = str(p05)

    # ------------------------------------------------------------------
    # Figure 06: Animated GIF (and static PNG) of Bunch Exciting BWD vs FWD Wave
    # ------------------------------------------------------------------
    y_bwd = z3["y_bwd_mm"]
    z_bwd = z3["z_bwd_mm"]
    snaps_bwd = z3["snap_bwd_ez"]
    t_snaps_bwd = z3["snap_bwd_t_ps"]
    zb_snaps_bwd = z3["snap_bwd_z_mm"]

    y_fwd = z3["y_fwd_mm"]
    z_fwd = z3["z_fwd_mm"]
    snaps_fwd = z3["snap_fwd_ez"]
    t_snaps_fwd = z3["snap_fwd_t_ps"]
    zb_snaps_fwd = z3["snap_fwd_z_mm"]

    z_in_mm = t3["backward_wave_glide_m1"]["z_in_mm"]
    z_out_mm = t3["backward_wave_glide_m1"]["z_out_mm"]
    n_frames = min(len(snaps_bwd), len(snaps_fwd))
    vmax_bwd = float(np.percentile(np.abs(snaps_bwd), 98.5))
    vmax_fwd = float(np.percentile(np.abs(snaps_fwd), 98.5))

    pil_frames: list[Image.Image] = []
    p06_png = figures_dir / "06_sws_bunch_excitation_fwd_vs_bwd.png"
    mid_frame_idx = n_frames // 2

    for idx in range(n_frames):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)
        im1 = ax_top.pcolormesh(
            z_bwd,
            y_bwd,
            snaps_bwd[idx] / max(vmax_bwd, 1e-12),
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        _overlay_2d_grating_geometry(
            ax_top,
            spec_2d,
            n_periods=10,
            symmetry="glide",
            z_in_mm=z_in_mm,
            z_out_mm=z_out_mm,
            show_legend=(idx == 0 or idx == mid_frame_idx),
        )
        zb1 = float(zb_snaps_bwd[idx])
        if 0.0 <= zb1 <= float(z_bwd[-1]):
            ax_top.plot(
                [zb1],
                [0.0],
                "o",
                color="#ffd43b",
                mec="black",
                mew=1.2,
                ms=8.5,
                zorder=6,
                label="Electron bunch ($+u_0 \\hat{z}$)",
            )
        ax_top.set_ylabel("$y$ [mm]")
        ax_top.set_ylim(-0.40, 0.40)
        ax_top.set_title(
            rf"(a) Backward-Wave Synchronism ($n = -1, v_g = -0.54c < 0, t = {t_snaps_bwd[idx]:.1f}\;\mathrm{{ps}}$): Wave Packet Streams Upstream ($-z \leftarrow$)"
        )
        if idx == 0 or idx == mid_frame_idx:
            ax_top.legend(loc="upper right", ncol=3, fontsize=7.3)
        fig.colorbar(im1, ax=ax_top, label=r"$E_{z,\mathrm{scat}} / E_{\max}$")

        im2 = ax_bot.pcolormesh(
            z_fwd,
            y_fwd,
            snaps_fwd[idx] / max(vmax_fwd, 1e-12),
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        _overlay_2d_grating_geometry(
            ax_bot,
            spec_2d,
            n_periods=10,
            symmetry="mirror",
            z_in_mm=z_in_mm,
            z_out_mm=z_out_mm,
            show_legend=False,
        )
        zb2 = float(zb_snaps_fwd[idx])
        if 0.0 <= zb2 <= float(z_fwd[-1]):
            ax_bot.plot(
                [zb2],
                [0.05],
                "o",
                color="#ffd43b",
                mec="black",
                mew=1.2,
                ms=8.5,
                zorder=6,
            )
        ax_bot.set_xlabel("Axial coordinate $z$ [mm]")
        ax_bot.set_ylabel("$y$ [mm]")
        ax_bot.set_ylim(-0.40, 0.40)
        ax_bot.set_xlim(float(z_bwd[0]), float(z_bwd[-1]))
        ax_bot.set_title(
            rf"(b) Forward-Wave Synchronism ($n = 0, v_g = +0.33c > 0, t = {t_snaps_fwd[idx]:.1f}\;\mathrm{{ps}}$): Wave Packet Streams Downstream ($\rightarrow +z$)"
        )
        fig.colorbar(im2, ax=ax_bot, label=r"$E_{z,\mathrm{scat}} / E_{\max}$")

        fig.tight_layout()
        if idx == mid_frame_idx:
            fig.savefig(p06_png, dpi=200)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_frames.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)

    p06_gif = figures_dir / "06_sws_bunch_excitation_fwd_vs_bwd.gif"
    pil_frames[0].save(
        p06_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=140,
        loop=0,
    )
    out_paths["fig_06_gif"] = str(p06_gif)
    out_paths["fig_06_png"] = str(p06_png)

    # ------------------------------------------------------------------
    # Figure 07: Mutations — Detuned Beam Velocity u_0 & Shorted PEC Cavity R/Q
    # ------------------------------------------------------------------
    f_u0 = z4["f_u0_ghz"]
    m_u0 = (f_u0 >= 95.0) & (f_u0 <= 132.0)
    fu = f_u0[m_u0]
    rz_low = z4["re_z_u0_low_ohm"][m_u0] * 1e-3
    rz_nom = z4["re_z_u0_nom_ohm"][m_u0] * 1e-3
    rz_high = z4["re_z_u0_high_ohm"][m_u0] * 1e-3

    f_nom_pk = t4["detuning_sweep"]["u0_nominal"]["f_pk_ghz"]
    f_low_pk = t4["detuning_sweep"]["u0_minus_5pct"]["f_pk_ghz"]
    f_high_pk = t4["detuning_sweep"]["u0_plus_5pct"]["f_pk_ghz"]
    df_low_th = t4["detuning_sweep"]["u0_minus_5pct"]["delta_f_linear_analytic_ghz"]
    df_high_th = t4["detuning_sweep"]["u0_plus_5pct"]["delta_f_linear_analytic_ghz"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))
    ax1.plot(
        fu,
        rz_low,
        color="#5c7cfa",
        lw=2.1,
        marker="none",
        label=rf"$0.95\,u_0$: $f_{{\mathrm{{pk}}}} = {f_low_pk:.2f}\;\mathrm{{GHz}}$ ($\Delta f = {f_low_pk - f_nom_pk:+.2f}\;\mathrm{{GHz}}$, th. ${df_low_th:+.2f}$)",
    )
    ax1.plot(
        fu,
        rz_nom,
        color="#0b7285",
        lw=2.2,
        marker="none",
        label=rf"$1.00\,u_0$ (20 kV): $f_{{\mathrm{{pk}}}} = {f_nom_pk:.2f}\;\mathrm{{GHz}}$",
    )
    ax1.plot(
        fu,
        rz_high,
        color="#d9480f",
        lw=2.1,
        marker="none",
        label=rf"$1.05\,u_0$: $f_{{\mathrm{{pk}}}} = {f_high_pk:.2f}\;\mathrm{{GHz}}$ ($\Delta f = {f_high_pk - f_nom_pk:+.2f}\;\mathrm{{GHz}}$, th. ${df_high_th:+.2f}$)",
    )
    ax1.axvline(f_nom_pk + df_low_th, color="#5c7cfa", ls="--", lw=1.3)
    ax1.axvline(f_nom_pk + df_high_th, color="#d9480f", ls="--", lw=1.3)
    ax1.set_xlabel("Frequency $f$ [GHz]")
    ax1.set_ylabel(r"Beam–wave resistance $\mathrm{Re}\,Z_\parallel(f)$ [$\mathrm{k\Omega}$]")
    ax1.set_title(r"(a) Detuned Beam Velocity Shift $\Delta\omega_s = \frac{\omega_s \Delta u_0}{u_0^2(1/u_0 - 1/v_g)}$")
    ax1.set_xlim(96.0, 130.0)
    ax1.set_ylim(-0.08, 1.72)
    ax1.legend(loc="upper right", fontsize=7.7)

    f_cav = z4["f_cav_ghz"]
    m_c = (f_cav >= 95.0) & (f_cav <= 134.0)
    fc = f_cav[m_c]
    rz_c = z4["re_z_cav_ohm"][m_c] * 1e-3

    ax2.plot(
        fc,
        rz_c,
        color="#d9480f",
        lw=2.0,
        marker="none",
        label=r"Unmatched PEC cavity ($|\Gamma|=1$, modes $\beta_0 d = m\pi/N$)",
    )
    ax2.plot(
        fu,
        rz_nom,
        color="#0b7285",
        lw=2.2,
        marker="none",
        label=rf"Matched traveling-wave SWS ($Z_{{\mathrm{{pk}}}} = {t4['pec_cavity_resonances']['matched_traveling_wave_z_pk_ohm']*1e-3:.2f}\;\mathrm{{k\Omega}}$)",
    )
    for m_info in t4["pec_cavity_resonances"]["modes"]:
        m_i = m_info["mode_index_m"]
        f_m = m_info["f_cold_test_ghz"]
        ax2.axvline(
            f_m,
            color="#2b8a3e",
            ls=":",
            lw=1.4,
            label=rf"Cold-test cavity modes $\beta_0 d = m\pi/10$ ($m=5,6,7$)" if m_i == 6 else None,
        )
        if m_i == 6:
            xy_txt = (m_info["f_fit_cavity_ghz"] + 2.6, 7.6)
        elif m_i == 5:
            xy_txt = (m_info["f_fit_cavity_ghz"] - 3.6, 1.55)
        else:
            xy_txt = (m_info["f_fit_cavity_ghz"] - 4.5, 2.25)
        ax2.annotate(
            rf"$m={m_i}$" + "\n" + rf"$(R/Q)_c={m_info['r_over_q_circuit_fit_ohm']:.0f}\;\Omega$",
            xy=(m_info["f_fit_cavity_ghz"], m_info["re_z_peak_ohm"] * 1e-3),
            xytext=xy_txt,
            fontsize=7.6,
            color="#2b2b2b",
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#495057"},
        )
    rec_z_kw = t4["pec_cavity_resonances"]["reconstructed_traveling_wave_z_pk_ohm"] * 1e-3
    ax2.scatter(
        [t4["pec_cavity_resonances"]["modes"][1]["f_fit_cavity_ghz"]],
        [rec_z_kw],
        marker="D",
        s=55,
        color="#2b8a3e",
        zorder=6,
        label=rf"Mode-$6$ area / FSR: $\mathcal{{A}}_6 / \Delta\omega_{{\mathrm{{FSR}}}} = {rec_z_kw:.2f}\;\mathrm{{k\Omega}}$",
    )
    ax2.set_xlabel("Frequency $f$ [GHz]")
    ax2.set_ylabel(r"Longitudinal resistance $\mathrm{Re}\,Z_\parallel(f)$ [$\mathrm{k\Omega}$]")
    ax2.set_title(r"(b) Shorted Cavity Modes & Standing-Wave $(R/Q)_m$ Relation")
    ax2.set_xlim(95.0, 134.0)
    ax2.set_ylim(0.0, 11.4)
    ax2.legend(loc="upper left", fontsize=7.5)

    fig.tight_layout()
    p07 = figures_dir / "07_mutations_detuned_u0_and_cavity_rq.png"
    fig.savefig(p07, dpi=220)
    plt.close(fig)
    out_paths["fig_07"] = str(p07)

    # ------------------------------------------------------------------
    # Figure 08: Mesh Convergence (Log-Log Order Guide + Length/Mesh Separation)
    # ------------------------------------------------------------------
    h_um = np.array(t5["h_um"])
    f_arr = np.array(t5["f_pk_ghz_array"])
    k_ax_arr = np.array(t5["k_on_axis_ohm_array"])
    k_band_min = float(t5["k_on_axis_finest_band_min_ohm"])
    k_band_max = float(t5["k_on_axis_finest_band_max_ohm"])
    k_band_mid = float(t5["k_on_axis_finest_band_mid_ohm"])
    k_band_spread_pct = float(t5["k_on_axis_finest_band_rel_spread"] * 100.0)
    f_band_min = float(t5["f_pk_finest_band_min_ghz"])
    f_band_max = float(t5["f_pk_finest_band_max_ghz"])
    f_band_mid = float(t5["f_pk_finest_band_mid_ghz"])
    k_ref_cold = float(t5["cold_test_ref_k_on_axis_ohm"])

    rel_err_k_band = np.maximum(np.abs(k_ax_arr - k_band_mid) / k_band_mid, 1e-5)
    rel_err_k_ref = np.maximum(np.abs(k_ax_arr - k_ref_cold) / k_ref_cold, 1e-5)
    rel_err_f_band = np.maximum(np.abs(f_arr - f_band_mid) / f_band_mid, 1e-5)
    guide_h2 = rel_err_k_band[0] * ((h_um / h_um[0]) ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))
    ax1.loglog(
        h_um,
        100.0 * rel_err_k_band,
        "o-",
        color="#0b7285",
        lw=2.1,
        ms=6.5,
        label=rf"$|K_{{-1}}(h) - K_{{\mathrm{{band,mid}}}}| / K_{{\mathrm{{band,mid}}}}$ (finest $[{k_band_min:.3f}, {k_band_max:.3f}]\;\Omega$, {k_band_spread_pct:.2f}% spread)",
    )
    ax1.loglog(
        h_um,
        100.0 * rel_err_k_ref,
        "^-.",
        color="#5c7cfa",
        lw=1.8,
        ms=6.0,
        label=rf"$|K_{{-1}}(h) - K_{{\mathrm{{cold,ref}}}}| / K_{{\mathrm{{cold,ref}}}}$ ($K_{{\mathrm{{cold,ref}}}} = {k_ref_cold:.3f}\;\Omega$)",
    )
    ax1.loglog(
        h_um,
        100.0 * rel_err_f_band,
        "s-",
        color="#2b8a3e",
        lw=2.1,
        ms=6.5,
        label=rf"$|f_{{\mathrm{{pk}}}}(h) - f_{{\mathrm{{band,mid}}}}| / f_{{\mathrm{{band,mid}}}}$ ($[{f_band_min:.2f}, {f_band_max:.2f}]\;\mathrm{{GHz}}$)",
    )
    ax1.loglog(
        h_um,
        100.0 * guide_h2,
        "--",
        color="#d9480f",
        lw=1.6,
        label=r"Reference $O(\Delta y_{\min}^2)$ slope $+2$ guide (non-monotone sequence)",
    )
    ax1.set_xlabel(r"Minimum cell size $\Delta y_{\min} = l_{\mathrm{vane}} / n_{\mathrm{feat}}$ [$\mu\mathrm{m}$]")
    ax1.set_ylabel("Relative deviation vs finest-level band / cold ref [%]")
    ax1.set_title(r"(a) Log–Log Spatial Convergence at Fixed $N = 8$ ($n_{\mathrm{feat}} \in \{4, 6, 8\}$)")
    ax1.set_ylim(1e-2, 2e1)
    ax1.legend(loc="lower right", fontsize=7.2)

    n_ref_t2_key = "20" if "20" in t2["runs"] else list(t2["runs"].keys())[-1]
    k_n20_nf4 = float(t2["runs"][n_ref_t2_key]["k_on_axis_beam_ohm"])
    labels_bar = [
        f"Length\n$N={n_ref_t2_key}, n_{{\\mathrm{{feat}}}}=4$",
        "Mesh $N=8$\n$n_{\\mathrm{feat}}=4$\n(uniform)",
        "Mesh $N=8$\n$n_{\\mathrm{feat}}=4$\n(graded)",
        "Mesh $N=8$\n$n_{\\mathrm{feat}}=6$\n(uniform)",
        "Mesh $N=8$\n$n_{\\mathrm{feat}}=8$\n(uniform)",
        "Cold test\nL2 ($n_{\\mathrm{feat}}=8$)",
        "Staircase\n$N=8, n_{\\mathrm{feat}}=4$",
    ]
    k_vals_bar = [
        k_n20_nf4,
        t5["runs"]["uniform_conformal_nf4"]["k_on_axis_beam_ohm"],
        t5["runs"]["graded_conformal_nf4"]["k_on_axis_beam_ohm"],
        t5["runs"]["uniform_conformal_nf6"]["k_on_axis_beam_ohm"],
        t5["runs"]["uniform_conformal_nf8"]["k_on_axis_beam_ohm"],
        l2["k_m1_on_axis_ohm"],
        t5["runs"]["uniform_staircase_nf4"]["k_on_axis_beam_ohm"],
    ]
    colors_bar = ["#5c7cfa", "#0b7285", "#15aabf", "#2b8a3e", "#2b8a3e", "#d9480f", "#868e96"]
    x_pos = np.arange(len(labels_bar))
    ax2.axhspan(
        k_band_min,
        k_band_max,
        color="#2b8a3e",
        alpha=0.22,
        zorder=2,
        label=rf"$N=8$ finest-level band $[{k_band_min:.3f}, {k_band_max:.3f}]\;\Omega$ ({k_band_spread_pct:.2f}% spread)",
    )
    bars = ax2.bar(x_pos, k_vals_bar, color=colors_bar, width=0.62, edgecolor="black", lw=0.8, zorder=3)
    for b_rect, val in zip(bars, k_vals_bar):
        ax2.text(
            b_rect.get_x() + 0.5 * b_rect.get_width(),
            val + 0.12,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
        )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_bar, fontsize=7.1)
    ax2.set_ylabel(r"On-axis Pierce impedance $K_{-1}(0)$ [$\Omega$]")
    ax2.set_title(r"(b) Length ($N$) vs Mesh ($n_{\mathrm{feat}}$) Effects & Finest-Level Band")
    ax2.set_ylim(0.0, 7.15)
    ax2.legend(loc="upper left", fontsize=7.4)

    fig.tight_layout()
    p08 = figures_dir / "08_mesh_convergence.png"
    fig.savefig(p08, dpi=220)
    plt.close(fig)
    out_paths["fig_08"] = str(p08)

    # ------------------------------------------------------------------
    # Figure 09: 3D Marosi V-Band Staggered-Pillar Glide SWS Beam Wake
    # ------------------------------------------------------------------
    f_3d = z6["f_ghz"]
    rz_3d = z6["re_z_ohm"] * 1e-3
    m3_plot = (f_3d >= 46.0) & (f_3d <= 71.5)

    f_u_cold = z6["f_unfolded_cold_ghz"]
    rz_u_cold = z6["re_z_cold_unfolded_ohm"] * 1e-3
    k_u_cold = z6["k_on_axis_cold_unfolded_ohm"]
    s2_u_cold = z6["sinc2_cold_unfolded"]
    f_band_3d = z6["f_band_3d_ghz"]
    rz_lin_sync = z6["re_z_3d_linear_sync_ohm"] * 1e-3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))
    ax1.plot(
        f_3d[m3_plot],
        rz_3d[m3_plot],
        color="#0b7285",
        lw=2.3,
        marker="none",
        label=rf"3D FIT beam wake ($N=6, f_{{\mathrm{{pk}}}}={t6['f_pk_ghz']:.1f}\;\mathrm{{GHz}}, Z_{{\mathrm{{pk}}}}={t6['re_z_pk_ohm']*1e-3:.2f}\;\mathrm{{k\Omega}}$)",
    )
    ax1.plot(
        f_u_cold,
        rz_u_cold,
        "o--",
        color="#d9480f",
        lw=1.9,
        ms=4.8,
        label=r"6.2 3D cold-test $\frac{1}{4}\beta_n(f)^2 K_n(f) L^2\,\mathrm{sinc}^2(\Delta\beta(f) L/2)$",
    )
    ax1.plot(
        f_band_3d,
        rz_lin_sync,
        ":",
        color="#2b8a3e",
        lw=1.7,
        marker="none",
        label=rf"Fixed-$K_n(f_s)$ model at $f_s = {t6['cold_test_marosi_3d']['f_sync_ghz']:.1f}\;\mathrm{{GHz}}$ ($K_n = {t6['cold_test_marosi_3d']['k_n_on_axis_at_sync_ohm']:.2f}\;\Omega$)",
    )
    ax1.axhline(0.0, color="#495057", lw=0.8)
    ax1.set_xlabel("Frequency $f$ [GHz]")
    ax1.set_ylabel(r"3D beam–wave resistance $\mathrm{Re}\,Z_\parallel(f)$ [$\mathrm{k\Omega}$]")
    ax1.set_title("(a) 3D Marosi V-Band Glide SWS (5.2 kV, $u_0 = 0.1416c$)")
    ax1.set_xlim(46.0, 71.5)
    ax1.set_ylim(-2.5, 5.5)
    ax1.legend(loc="upper right", fontsize=7.6)

    ax2_r = ax2.twinx()
    pb_m = (f_u_cold >= 50.0) & (f_u_cold <= 73.5)
    l_m1 = ax2.plot(
        f_u_cold[pb_m],
        k_u_cold[pb_m],
        "o-",
        color="#0b7285",
        lw=2.1,
        ms=5.2,
        label=r"6.2 3D cold-test on-axis $K_n(f)$ [$\Omega$]",
    )
    l_m2 = ax2.plot(
        [t6["f_pk_ghz"]],
        [t6["k_on_axis_at_pk_ohm"]],
        "D",
        color="#d9480f",
        mec="black",
        ms=8.0,
        label=rf"3D beam-wake $K_n(f_{{\mathrm{{pk}}}}) = {t6['k_on_axis_at_pk_ohm']:.2f}\;\Omega$ (cold 55–58 GHz: {t6['cold_test_marosi_3d']['k_cold_lower_branch_55_58ghz_ohm']:.2f} $\Omega$)",
    )
    l_m3 = ax2_r.plot(
        f_u_cold[pb_m],
        s2_u_cold[pb_m],
        "s--",
        color="#2b8a3e",
        lw=1.8,
        ms=4.5,
        label=r"Phase-matching factor $\mathrm{sinc}^2(\Delta\beta(f) L / 2)$ ($u_0 \approx v_g$)",
    )
    ax2.set_xlabel("Frequency $f$ [GHz]")
    ax2.set_ylabel(r"Pierce interaction impedance $K_n(f)$ [$\Omega$]", color="#0b7285")
    ax2_r.set_ylabel(r"Phase factor $\mathrm{sinc}^2(\Delta\beta(f) L/2)$ [—]", color="#2b8a3e")
    ax2.set_title(r"(b) Broadband Group-Velocity Matching ($u_0 = 0.1416c \approx v_g = 0.132c$)")
    ax2.set_xlim(50.0, 73.5)
    ax2.set_ylim(0.0, 25.0)
    ax2_r.set_ylim(0.0, 1.15)
    l_m_all = l_m1 + l_m2 + l_m3
    ax2.legend(l_m_all, [x.get_label() for x in l_m_all], loc="center right", fontsize=7.5)

    fig.tight_layout()
    p09 = figures_dir / "09_3d_marosi_glide_beam_wake.png"
    fig.savefig(p09, dpi=220)
    plt.close(fig)
    out_paths["fig_09"] = str(p09)

    return out_paths


def build_convergence_separation_markdown(data_dir: Path | str) -> str:
    """Build a markdown table separating structure-length (N) from mesh-resolution (n_feat) effects on K_{-1}(0)."""
    data_dir = Path(data_dir)
    t2 = json.loads((data_dir / "task_02_2d_length_sweep.json").read_text())
    t5 = json.loads((data_dir / "task_05_mesh_convergence.json").read_text())
    sm = json.loads((data_dir / "summary_metrics.json").read_text())

    k_l1_cold = float(sm["n20_cold_k_on_axis_L1_ohm"])
    k_l2_cold = float(sm["n20_cold_k_on_axis_L2_ohm"])
    k_ref_cold = float(sm["n20_cold_k_on_axis_ref_ohm"])
    k_n8_nf4 = float(t5["runs"]["uniform_conformal_nf4"]["k_on_axis_beam_ohm"])
    k_n20_nf4 = float(sm["n20_k_on_axis_beam_ohm"])
    len_shift_pct = float(sm["length_effect_k_rel_diff_n20_vs_n8_nf4"] * 100.0)
    gamma_bound_pct = float(sm["termination_rel_error_bound_tapered"] * 100.0)

    k_nf6 = float(t5["runs"]["uniform_conformal_nf6"]["k_on_axis_beam_ohm"])
    k_nf8 = float(t5["runs"]["uniform_conformal_nf8"]["k_on_axis_beam_ohm"])
    k_grad = float(t5["runs"]["graded_conformal_nf4"]["k_on_axis_beam_ohm"])
    k_stair = float(t5["runs"]["uniform_staircase_nf4"]["k_on_axis_beam_ohm"])
    k_bmin = float(t5["k_on_axis_finest_band_min_ohm"])
    k_bmax = float(t5["k_on_axis_finest_band_max_ohm"])
    k_bmid = float(t5["k_on_axis_finest_band_mid_ohm"])
    k_bspr_ohm = float(t5["k_on_axis_finest_band_spread_ohm"])
    k_bspr_pct = float(t5["k_on_axis_finest_band_rel_spread"] * 100.0)
    mesh_nf6_pct = float(t5["rel_diff_nf6_vs_nf4_same_N"] * 100.0)
    mesh_nf8_pct = float(t5["rel_diff_nf8_vs_nf4_same_N"] * 100.0)
    err_b_l2_min = float(t5["rel_diff_finest_band_vs_cold_L2_min"] * 100.0)
    err_b_l2_max = float(t5["rel_diff_finest_band_vs_cold_L2_max"] * 100.0)
    err_b_ref_min = float(t5["rel_diff_finest_band_vs_cold_ref_min"] * 100.0)
    err_b_ref_max = float(t5["rel_diff_finest_band_vs_cold_ref_max"] * 100.0)

    rows = [
        "| Variation Axis | Active Periods $N$ ($L = N d$) | Mesh Resolution ($n_{\\mathrm{feat}}, \\Delta y_{\\min}$) | Extracted $f_{\\mathrm{pk}}$ [GHz] | Extracted $K_{-1}(0)$ [$\\Omega$] | Shift Along Axis (Fixed Complement) | Cold-Test Benchmark Comparison |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    # Part A: Structure-length variation at fixed mesh n_feat = 4
    n_keys_t2 = sorted([int(k) for k in t2["runs"].keys()])
    length_entries: list[tuple[int, dict[str, Any]]] = []
    for n_val in n_keys_t2:
        length_entries.append((n_val, t2["runs"][str(n_val)]))
    if int(t5["n_periods"]) not in n_keys_t2:
        length_entries.append((int(t5["n_periods"]), t5["runs"]["uniform_conformal_nf4"]))
        length_entries.sort(key=lambda item: item[0])

    for n_val, r_item in length_entries:
        k_val = float(r_item["k_on_axis_beam_ohm"])
        f_val = float(r_item["f_pk_ghz"])
        L_mm = float(r_item["active_length_mm"])
        shift_vs_n8 = (k_val - k_n8_nf4) / k_n8_nf4 * 100.0
        err_vs_l1 = abs(k_val - k_l1_cold) / k_l1_cold * 100.0
        shift_str = (
            "baseline ($N=8, n_{\\mathrm{feat}}=4$)"
            if n_val == int(t5["n_periods"])
            else f"${shift_vs_n8:+.2f}\\%$ vs $N={t5['n_periods']}$ (same $n_{{\\mathrm{{feat}}}}=4$)"
        )
        rows.append(
            f"| **Length effect** (fixed $n_{{\\mathrm{{feat}}}}=4$) | $N = {n_val}$ ($L = {L_mm:.1f}\\;\\mathrm{{mm}}$) | Conformal uniform ($n_{{\\mathrm{{feat}}}}=4, 25.0\\;\\mu\\mathrm{{m}}$) | ${f_val:.3f}$ | ${k_val:.3f}$ | {shift_str} | ${err_vs_l1:.2f}\\%$ vs L1 (${k_l1_cold:.3f}\\;\\Omega$; $\\Delta K(N)={len_shift_pct:.2f}\\%, \\Gamma$ bound ${gamma_bound_pct:.1f}\\%$) |"
        )

    # Part B: Mesh variation at fixed length N = 8
    f_nf4 = float(t5["runs"]["uniform_conformal_nf4"]["f_pk_ghz"])
    f_nf6 = float(t5["runs"]["uniform_conformal_nf6"]["f_pk_ghz"])
    f_nf8 = float(t5["runs"]["uniform_conformal_nf8"]["f_pk_ghz"])
    f_grad = float(t5["runs"]["graded_conformal_nf4"]["f_pk_ghz"])
    f_stair = float(t5["runs"]["uniform_staircase_nf4"]["f_pk_ghz"])
    f_bmin = float(t5["f_pk_finest_band_min_ghz"])
    f_bmax = float(t5["f_pk_finest_band_max_ghz"])
    n_conv = int(t5["n_periods"])
    L_conv_mm = float(t5["runs"]["uniform_conformal_nf4"]["active_length_mm"])
    err_nf4_l1 = float(t5["rel_diff_nf4_vs_cold_L1"] * 100.0)
    err_nf6_l2 = float(t5["rel_diff_nf6_vs_cold_L2"] * 100.0)
    err_nf8_l2 = float(t5["rel_diff_nf8_vs_cold_L2"] * 100.0)
    grad_shift_pct = float((k_grad - k_n8_nf4) / k_n8_nf4 * 100.0)
    stair_ratio = float(t5["staircase_over_conformal_k_ratio"])

    rows.extend(
        [
            f"| **Mesh effect** (fixed $N={n_conv}$) | $N = {n_conv}$ ($L = {L_conv_mm:.1f}\\;\\mathrm{{mm}}$) | Conformal uniform ($n_{{\\mathrm{{feat}}}}=4, 25.0\\;\\mu\\mathrm{{m}}$) | ${f_nf4:.3f}$ | ${k_n8_nf4:.3f}$ | baseline ($N={n_conv}, n_{{\\mathrm{{feat}}}}=4$) | ${err_nf4_l1:.2f}\\%$ vs L1 (${k_l1_cold:.3f}\\;\\Omega$) |",
            f"| **Mesh effect** (fixed $N={n_conv}$) | $N = {n_conv}$ ($L = {L_conv_mm:.1f}\\;\\mathrm{{mm}}$) | Conformal uniform ($n_{{\\mathrm{{feat}}}}=6, 16.7\\;\\mu\\mathrm{{m}}$) | ${f_nf6:.3f}$ | ${k_nf6:.3f}$ | ${mesh_nf6_pct:+.2f}\\%$ vs $n_{{\\mathrm{{feat}}}}=4$ (same $N={n_conv}$) | ${err_nf6_l2:.2f}\\%$ vs L2 (${k_l2_cold:.3f}\\;\\Omega$) |",
            f"| **Mesh effect** (fixed $N={n_conv}$) | $N = {n_conv}$ ($L = {L_conv_mm:.1f}\\;\\mathrm{{mm}}$) | Conformal uniform ($n_{{\\mathrm{{feat}}}}=8, 12.5\\;\\mu\\mathrm{{m}}$) | ${f_nf8:.3f}$ | ${k_nf8:.3f}$ | ${mesh_nf8_pct:+.2f}\\%$ vs $n_{{\\mathrm{{feat}}}}=4$ (same $N={n_conv}$) | ${err_nf8_l2:.2f}\\%$ vs L2 (${k_l2_cold:.3f}\\;\\Omega$) |",
            f"| **Finest-level band** ($n_{{\\mathrm{{feat}}}} \\in \\{{6, 8\\}}$) | $N = {n_conv}$ ($L = {L_conv_mm:.1f}\\;\\mathrm{{mm}}$) | Non-monotone finest band ($12.5\\text{{–}}16.7\\;\\mu\\mathrm{{m}}$) | $[{f_bmin:.3f}, {f_bmax:.3f}]$ | $[{k_bmin:.3f}, {k_bmax:.3f}]$ (mid ${k_bmid:.3f}$, spread ${k_bspr_ohm:.3f}\\;\\Omega$ / ${k_bspr_pct:.2f}\\%$) | ${mesh_nf6_pct:+.2f}\\%$ to ${mesh_nf8_pct:+.2f}\\%$ vs $n_{{\\mathrm{{feat}}}}=4$ (no Richardson order justified) | **${err_b_l2_min:.2f}\\text{{–}}{err_b_l2_max:.2f}\\%$** vs L2 (${k_l2_cold:.3f}\\;\\Omega$), **${err_b_ref_min:.2f}\\text{{–}}{err_b_ref_max:.2f}\\%$** vs ref (${k_ref_cold:.3f}\\;\\Omega$) |",
            f"| **Graded mesh ablation** (fixed $N={n_conv}$) | $N = {n_conv}$ ($L = {L_conv_mm:.1f}\\;\\mathrm{{mm}}$) | Conformal graded ($n_{{\\mathrm{{feat}}}}=4, \\Delta y_{{\\min}}=25.0\\;\\mu\\mathrm{{m}}$) | ${f_grad:.3f}$ | ${k_grad:.3f}$ | ${grad_shift_pct:+.2f}\\%$ vs uniform $n_{{\\mathrm{{feat}}}}=4$ | ${abs(k_grad - k_l1_cold)/k_l1_cold*100:.2f}\\%$ vs L1 (${k_l1_cold:.3f}\\;\\Omega$) |",
            f"| **Staircase ablation** (fixed $N={n_conv}$) | $N = {n_conv}$ ($L = {L_conv_mm:.1f}\\;\\mathrm{{mm}}$) | Non-conformal staircase ($n_{{\\mathrm{{feat}}}}=4, 25.0\\;\\mu\\mathrm{{m}}$) | ${f_stair:.3f}$ | ${k_stair:.3f}$ | ${stair_ratio:.2f}\\times$ uniform conformal $n_{{\\mathrm{{feat}}}}=4$ | Over-predicts $K_{{-1}}(0)$ without Dey–Mittra areas |",
        ]
    )
    return "\n".join(rows)


def build_quantitative_summary_markdown(data_dir: Path | str) -> str:
    """Build the Quantitative Summary markdown table directly from data/summary_metrics.json."""
    data_dir = Path(data_dir)
    sm = json.loads((data_dir / "summary_metrics.json").read_text())

    err_slope_pk = (
        abs(sm["length_scaling_peak_slope_fit"] - sm["length_scaling_peak_slope_theory"])
        / abs(sm["length_scaling_peak_slope_theory"])
        * 100.0
    )
    err_slope_fwhm = (
        abs(sm["length_scaling_fwhm_slope_fit"] - sm["length_scaling_fwhm_slope_theory"])
        / abs(sm["length_scaling_fwhm_slope_theory"])
        * 100.0
    )
    err_f_n20 = (
        abs(sm["n20_f_pk_ghz"] - sm["n20_cold_f_sync_L2_ghz"])
        / sm["n20_cold_f_sync_L2_ghz"]
        * 100.0
    )
    err_z_n20_l1 = (
        abs(sm["n20_re_z_pk_ohm"] - sm["n20_cold_re_z_pk_L1_ohm"])
        / sm["n20_cold_re_z_pk_L1_ohm"]
        * 100.0
    )
    err_z_n20_l2 = (
        abs(sm["n20_re_z_pk_ohm"] - sm["n20_cold_re_z_pk_L2_ohm"])
        / sm["n20_cold_re_z_pk_L2_ohm"]
        * 100.0
    )
    err_fwhm_n20 = (
        abs(sm["n20_fwhm_ghz"] - sm["n20_cold_fwhm_L2_ghz"])
        / sm["n20_cold_fwhm_L2_ghz"]
        * 100.0
    )
    err_k_axis_l1 = (
        abs(sm["n20_k_on_axis_beam_ohm"] - sm["n20_cold_k_on_axis_L1_ohm"])
        / sm["n20_cold_k_on_axis_L1_ohm"]
        * 100.0
    )
    err_k_n8_nf4_l1 = (
        abs(sm["conv_k_axis_nf4_ohm"] - sm["n20_cold_k_on_axis_L1_ohm"])
        / sm["n20_cold_k_on_axis_L1_ohm"]
        * 100.0
    )
    len_shift_pct = sm["length_effect_k_rel_diff_n20_vs_n8_nf4"] * 100.0
    gamma_bound_pct = sm["termination_rel_error_bound_tapered"] * 100.0
    k_bmin = sm["conv_k_axis_band_min_ohm"]
    k_bmax = sm["conv_k_axis_band_max_ohm"]
    k_bspr_pct = sm["conv_k_axis_band_rel_spread"] * 100.0
    mesh_nf8_pct = sm["conv_k_axis_nf8_vs_nf4_rel_diff"] * 100.0
    mesh_nf6_pct = sm["conv_k_axis_nf6_vs_nf4_rel_diff"] * 100.0
    err_band_l2_min = sm["conv_k_axis_band_vs_cold_L2_min_rel_diff"] * 100.0
    err_band_l2_max = sm["conv_k_axis_band_vs_cold_L2_max_rel_diff"] * 100.0
    err_band_ref_min = sm["conv_k_axis_band_vs_cold_ref_min_rel_diff"] * 100.0
    err_band_ref_max = sm["conv_k_axis_band_vs_cold_ref_max_rel_diff"] * 100.0
    err_k_7node_l1 = (
        abs(sm["n20_k_sheet_avg_7node_mean_ohm"] - sm["n20_cold_k_sheet_avg_L1_ohm"])
        / sm["n20_cold_k_sheet_avg_L1_ohm"]
        * 100.0
    )
    err_ratio_7node_l1 = (
        abs(sm["n20_sheet_ratio_7node_mean"] - sm["n20_cold_sheet_ratio_L1"])
        / sm["n20_cold_sheet_ratio_L1"]
        * 100.0
    )
    err_dw_dz = (
        abs(sm["n20_dw_dz_unattenuated_j_per_m"] - sm["n20_dw_dz_analytic_j_per_m"])
        / sm["n20_dw_dz_analytic_j_per_m"]
        * 100.0
    )
    err_fwd_k_sync = (
        abs(sm["fwd_k0_at_sync_y50um_fit_ohm"] - sm["fwd_k0_at_sync_y50um_cold_L1_ohm"])
        / sm["fwd_k0_at_sync_y50um_cold_L1_ohm"]
        * 100.0
    )
    err_fwd_z_nl = (
        abs(sm["fwd_re_z_pk_fit_ohm"] - sm["fwd_re_z_pk_nonlinear_cold_L1_ohm"])
        / sm["fwd_re_z_pk_nonlinear_cold_L1_ohm"]
        * 100.0
    )
    err_bwd_ebal = sm["bwd_energy_balance_rel_error_vs_total"] * 100.0
    err_fwd_ebal = sm["fwd_energy_balance_rel_error_vs_total"] * 100.0
    err_detune_m5 = (
        abs(sm["detuning_minus_5pct_df_fit_ghz"] - sm["detuning_minus_5pct_df_theory_ghz"])
        / abs(sm["detuning_minus_5pct_df_theory_ghz"])
        * 100.0
    )
    err_cav_f = (
        abs(sm["cavity_m6_f_fit_ghz"] - sm["cavity_m6_f_cold_ghz"])
        / sm["cavity_m6_f_cold_ghz"]
        * 100.0
    )
    err_cav_rq = (
        abs(sm["cavity_m6_rq_circuit_fit_ohm"] - sm["cavity_m6_rq_circuit_analytic_ohm"])
        / sm["cavity_m6_rq_circuit_analytic_ohm"]
        * 100.0
    )
    err_3d_z = (
        abs(sm["marosi_3d_re_z_pk_ohm"] - sm["marosi_3d_re_z_cold_lower_branch_ohm"])
        / sm["marosi_3d_re_z_cold_lower_branch_ohm"]
        * 100.0
    )
    err_3d_k = (
        abs(sm["marosi_3d_k_at_pk_ohm"] - sm["marosi_3d_k_cold_lower_branch_ohm"])
        / sm["marosi_3d_k_cold_lower_branch_ohm"]
        * 100.0
    )

    rows = [
        "| Quantity / Benchmark | Theory / Tutorial 6.2 Cold Test | Wakis FIT Beam-Wake Result | Relative Difference / Agreement |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Tapered vs abrupt port reflection** $|\\Gamma(f_s)|$ (${sm['termination_f_sync_ghz']:.2f}\\;\\mathrm{{GHz}}$) | Production 2-stage taper $[0.25h, 0.55h]$ vs abrupt end ($|\\Gamma_{{\\mathrm{{abrupt}}}}| = {sm['termination_gamma_abrupt_at_sync']:.4f}$, no floor/clip) | $|\\Gamma_{{\\mathrm{{taper}}}}(f_s)| = {sm['termination_gamma_tapered_at_sync']:.4f}$ (worst-case peak bound ${gamma_bound_pct:.1f}\\%$) | **${sm['termination_reduction_factor']:.2f}\\times$ reflection reduction** |",
        f"| **(a) Peak impedance length scaling** $\\mathrm{{slope}}(\\log\\mathrm{{Re}}\\,Z_{{\\mathrm{{pk}}}}, \\log L)$ | Exact $\\propto L^2$ scaling: slope $= {sm['length_scaling_peak_slope_theory']:.4f}$ | Fitted slope ($N \\in \\{{5,10,20,40\\}}$) $= {sm['length_scaling_peak_slope_fit']:.4f}$ | **${err_slope_pk:.2f}\\%$** relative error |",
        f"| **(a) Linewidth length scaling** $\\mathrm{{slope}}(\\log\\Delta f_{{\\mathrm{{FWHM}}}}, \\log L)$ | Exact $\\propto L^{{-1}}$ scaling: slope $= {sm['length_scaling_fwhm_slope_theory']:.4f}$ | Fitted slope ($N \\in \\{{5,10,20,40\\}}$) $= {sm['length_scaling_fwhm_slope_fit']:.4f}$ | **${err_slope_fwhm:.2f}\\%$** relative error |",
        f"| **(b) Synchronous frequency** $f_s$ ($N=20$, $20\\;\\mathrm{{kV}}$, $u_0 = 0.2719c$) | 6.2 cold-test crossing $f_s = {sm['n20_cold_f_sync_L2_ghz']:.2f}\\;\\mathrm{{GHz}}$ | Beam-wake peak $f_{{\\mathrm{{pk}}}} = {sm['n20_f_pk_ghz']:.2f}\\;\\mathrm{{GHz}}$ | **${err_f_n20:.2f}\\%$** relative error |",
        f"| **(b) Group-velocity walk-off FWHM** $\\Delta f_{{\\mathrm{{FWHM}}}}$ ($N=20$, no fit) | $\\frac{{0.8859}}{{L|1/u_0 - 1/v_g|}} = {sm['n20_cold_fwhm_L2_ghz']:.2f}\\;\\mathrm{{GHz}}$ ($v_g = -0.544c$) | Measured $\\Delta f_{{\\mathrm{{FWHM}}}} = {sm['n20_fwhm_ghz']:.2f}\\;\\mathrm{{GHz}}$ | **${err_fwhm_n20:.2f}\\%$** relative difference |",
        f"| **(c) Peak resistance** $\\mathrm{{Re}}\\,Z_\\parallel(\\omega_s)$ ($N=20, L=10\\;\\mathrm{{mm}}, W_x=1.0\\;\\mathrm{{mm}}$) | $\\frac{{1}}{{4}}\\beta_{{-1}}^2 K_{{-1}}(0) L^2 = {sm['n20_cold_re_z_pk_L1_ohm']:.1f}\\;\\Omega$ (L1) / ${sm['n20_cold_re_z_pk_L2_ohm']:.1f}\\;\\Omega$ (L2) | $\\mathrm{{Re}}\\,Z_{{\\mathrm{{pk}}}} = {sm['n20_re_z_pk_ohm']:.1f}\\;\\Omega$ (per $1.0\\;\\mathrm{{mm}}$ $x$-width) | **${err_z_n20_l1:.2f}\\%$** vs L1 (**${err_z_n20_l2:.2f}\\%$** vs L2) |",
        f"| **(c) On-axis Pierce impedance & length effect** $K_{{-1}}(0)$ (fixed $n_{{\\mathrm{{feat}}}}=4$) | 6.2 cold test L1 ($n_{{\\mathrm{{feat}}}}=4$): ${sm['n20_cold_k_on_axis_L1_ohm']:.3f}\\;\\Omega$ (residual $|\\Gamma|={sm['termination_gamma_tapered_at_sync']:.4f}$ allows $\\le {gamma_bound_pct:.1f}\\%$ bound) | $N=20$: ${sm['n20_k_on_axis_beam_ohm']:.3f}\\;\\Omega$; $N=8$: ${sm['conv_k_axis_nf4_ohm']:.3f}\\;\\Omega$ (${len_shift_pct:+.2f}\\%$ length shift at fixed $n_{{\\mathrm{{feat}}}}=4$) | **${err_k_axis_l1:.2f}\\%$** at $N=20$ (${err_k_n8_nf4_l1:.2f}\\%$ at $N=8$; length shift ${abs(len_shift_pct):.2f}\\%$ & $\\Gamma$ bound ${gamma_bound_pct:.1f}\\%$ exceed ${err_k_axis_l1:.2f}\\%$) |",
        f"| **(c) Mesh convergence of $K_{{-1}}(0)$** (fixed $N=8, n_{{\\mathrm{{feat}}}}\\in\\{{4,6,8\\}}$) | 6.2 cold test: ${sm['n20_cold_k_on_axis_L2_ohm']:.3f}\\;\\Omega$ (L2, $n_{{\\mathrm{{feat}}}}=8$) / ${sm['n20_cold_k_on_axis_ref_ohm']:.3f}\\;\\Omega$ (ref, $n_{{\\mathrm{{feat}}}}=16$) | $n_{{\\mathrm{{feat}}}}=4,6,8$: ${sm['conv_k_axis_nf4_ohm']:.3f}, {sm['conv_k_axis_nf6_ohm']:.3f}, {sm['conv_k_axis_nf8_ohm']:.3f}\\;\\Omega$; finest band $[{k_bmin:.3f}, {k_bmax:.3f}]\\;\\Omega$ (spread ${k_bspr_pct:.2f}\\%$, mesh shift ${mesh_nf6_pct:+.2f}\\%$ to ${mesh_nf8_pct:+.2f}\\%$) | **${err_band_l2_min:.2f}\\text{{–}}{err_band_l2_max:.2f}\\%$** vs L2 (**${err_band_ref_min:.2f}\\text{{–}}{err_band_ref_max:.2f}\\%$** vs ref) |",
        f"| **(c) Measured off-axis sheet average** $\\langle K_{{-1}}\\rangle_{{\\mathrm{{nodes}}}}$ ($H_b = 150\\;\\mu\\mathrm{{m}}$) | 6.2 L1 7-node mean: ${sm['n20_cold_k_sheet_avg_L1_ohm']:.3f}\\;\\Omega$ (ratio ${sm['n20_cold_sheet_ratio_L1']:.3f}$); L2: ${sm['n20_cold_k_sheet_avg_L2_ohm']:.3f}\\;\\Omega$ (ratio ${sm['n20_cold_sheet_ratio_L2']:.3f}$) | 7-node beam sweep: ${sm['n20_k_sheet_avg_7node_mean_ohm']:.3f}\\;\\Omega$ (ratio ${sm['n20_sheet_ratio_7node_mean']:.3f}$; trapz ${sm['n20_k_sheet_avg_trapz_ohm']:.3f}\\;\\Omega$; $K(0)\\eta_{{\\mathrm{{cosh}}}}={sm['n20_k_sheet_avg_beam_ohm']:.3f}\\;\\Omega$) | **${err_k_7node_l1:.2f}\\%$** in $\\langle K_{{-1}}\\rangle$, **${err_ratio_7node_l1:.2f}\\%$** in ratio |",
        f"| **(d) Infinite-structure stopping power** $\\mathrm{{d}}W/\\mathrm{{d}}z$ ($q = 1\\;\\mathrm{{pC}}$) | $\\frac{{q^2 \\beta_{{-1}}^2 K_{{-1}}(0)}}{{2|1/u_0 - 1/v_g|}} = {sm['n20_dw_dz_analytic_j_per_m']*1e9:.3f}\\;\\mathrm{{nJ/m}}$ | $\\frac{{q^2}}{{\\pi L}}\\int \\mathrm{{Re}}\\,Z_\\parallel\\,\\mathrm{{d}}\\omega = {sm['n20_dw_dz_unattenuated_j_per_m']*1e9:.3f}\\;\\mathrm{{nJ/m}}$ ($N=20$) | **${err_dw_dz:.2f}\\%$** relative difference |",
        f"| **(e) Forward-wave control** $K_0(y_0=50\\;\\mu\\mathrm{{m}})$ ($n=0, u_0=0.5689c$) | 6.2 mirror-odd at $y_0=50\\;\\mu\\mathrm{{m}}$: $K_0(f_s) = {sm['fwd_k0_at_sync_y50um_cold_L1_ohm']:.3f}\\;\\Omega$, nonlinear $Z_{{\\mathrm{{pk}}}} = {sm['fwd_re_z_pk_nonlinear_cold_L1_ohm']:.1f}\\;\\Omega$ (${sm['fwd_f_pk_nonlinear_cold_L1_ghz']:.1f}\\;\\mathrm{{GHz}}$) | FIT $K_0(f_s) = {sm['fwd_k0_at_sync_y50um_fit_ohm']:.3f}\\;\\Omega$, $Z_{{\\mathrm{{pk}}}} = {sm['fwd_re_z_pk_fit_ohm']:.1f}\\;\\Omega$ at ${sm['fwd_f_pk_ghz']:.2f}\\;\\mathrm{{GHz}}$, FWHM $= {sm['fwd_fwhm_fit_ghz']:.2f}\\;\\mathrm{{GHz}}$ | **${err_fwd_k_sync:.2f}\\%$** at $f_s$, **${err_fwd_z_nl:.2f}\\%$** at $f_{{\\mathrm{{pk}}}}$ |",
        f"| **(e) Smooth-guide Poynting energy balance** ($E_{{\\mathrm{{rad}}}} = E_{{\\mathrm{{out}}}} - E_{{\\mathrm{{in}}}}$ vs $W_{{\\mathrm{{loss}}}}$ between ports) | Bwd ($v_g < 0$): $W_{{\\mathrm{{loss}}}} = {sm['bwd_w_loss_between_ports_pj']:.3f}\\;\\mathrm{{pJ}}$; Fwd ($v_g > 0$): $W_{{\\mathrm{{loss}}}} = {sm['fwd_w_loss_between_ports_pj']:.3f}\\;\\mathrm{{pJ}}$ | Bwd: $E_{{\\mathrm{{in}}}} = {sm['bwd_poynting_energy_in_pj']:.3f}\\;\\mathrm{{pJ}}, E_{{\\mathrm{{rad}}}} = {sm['bwd_total_radiated_pj']:.3f}\\;\\mathrm{{pJ}}$; Fwd: $E_{{\\mathrm{{out}}}} = +{sm['fwd_poynting_energy_out_pj']:.3f}\\;\\mathrm{{pJ}}, E_{{\\mathrm{{rad}}}} = {sm['fwd_total_radiated_pj']:.3f}\\;\\mathrm{{pJ}}$ | **${err_bwd_ebal:.1f}\\%$** (bwd) & **${err_fwd_ebal:.1f}\\%$** (fwd) closure |",
        f"| **Mutation 1: Detuned beam velocity** ($u_0 \\to 0.95\\,u_0$) | $\\Delta f_s = \\frac{{f_s \\Delta u_0}}{{u_0^2(1/u_0 - 1/v_g)}} = {sm['detuning_minus_5pct_df_theory_ghz']:.2f}\\;\\mathrm{{GHz}}$ | Measured peak shift $\\Delta f_{{\\mathrm{{fit}}}} = {sm['detuning_minus_5pct_df_fit_ghz']:.2f}\\;\\mathrm{{GHz}}$ | **${err_detune_m5:.2f}\\%$** relative error |",
        f"| **Mutation 2: Shorted PEC cavity mode** $m=6$ ($N=10$, $\\beta_0 d = 0.6\\pi$) | $f_6 = {sm['cavity_m6_f_cold_ghz']:.2f}\\;\\mathrm{{GHz}}$, $(R/Q)_{{\\mathrm{{circuit}},6}} = {sm['cavity_m6_rq_circuit_analytic_ohm']:.1f}\\;\\Omega$ | $f_6 = {sm['cavity_m6_f_fit_ghz']:.2f}\\;\\mathrm{{GHz}}$, $(R/Q)_{{\\mathrm{{circuit}},6}} = {sm['cavity_m6_rq_circuit_fit_ohm']:.1f}\\;\\Omega$ | **${err_cav_f:.2f}\\%$** in $f_6$, **${err_cav_rq:.2f}\\%$** in $(R/Q)_6$ |",
        f"| **3D Marosi V-band glide SWS** ($N=6, 5.2\\;\\mathrm{{kV}}, u_0 = 0.1416c$) | Cold-test lower-branch peak ${sm['marosi_3d_re_z_cold_lower_branch_ohm']*1e-3:.2f}\\;\\mathrm{{k\\Omega}}$, $K_n = {sm['marosi_3d_k_cold_lower_branch_ohm']:.2f}\\;\\Omega$ | 3D FIT peak ${sm['marosi_3d_re_z_pk_ohm']*1e-3:.2f}\\;\\mathrm{{k\\Omega}}$ at ${sm['marosi_3d_f_pk_ghz']:.2f}\\;\\mathrm{{GHz}}$, $K_n = {sm['marosi_3d_k_at_pk_ohm']:.2f}\\;\\Omega$ | **${err_3d_z:.2f}\\%$** in $\\mathrm{{Re}}\\,Z_{{\\mathrm{{pk}}}}$, **${err_3d_k:.2f}\\%$** in $K_n$ |",
    ]
    return "\n".join(rows)




def _strip_heavy_arrays(run_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy of a beam-wake run dictionary without large 1D/2D arrays."""
    heavy_keys = {
        "f_ghz",
        "re_z_ohm",
        "im_z_ohm",
        "s_m",
        "wp_v_per_pc",
        "poynting",
        "snapshots",
        "y_mm",
        "z_mm",
    }
    return {k: v for k, v in run_dict.items() if k not in heavy_keys}


def generate_task_01_data(
    data_dir: Path | str,
    *,
    quick: bool = False,
) -> dict[str, Any]:
    """Regenerate ``task_01_termination_reflection.json`` using the production [0.25, 0.55] taper."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    res = measure_sws_termination_reflection(
        taper_scales=[0.25, 0.55],
        quick=quick,
    )
    out_path = data_dir / "task_01_termination_reflection.json"
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


def generate_task_02_data(
    data_dir: Path | str,
    *,
    quick: bool = False,
    reuse_on_axis_length_sweep: bool = False,
) -> dict[str, Any]:
    """Regenerate ``task_02_2d_length_sweep.json``, ``task_02_2d_length_sweep_spectra.npz``, and sheet-beam y-sweep."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    spec = KaretnikovaGratingSpec()
    bench = load_cold_test_benchmarks()
    l1 = bench["levels_2d"]["L1"]
    l2 = bench["levels_2d"]["L2"]
    u0 = bench["u0_2d_m_s"]
    f_band = (95.0, 132.0)

    json_path = data_dir / "task_02_2d_length_sweep.json"
    npz_path = data_dir / "task_02_2d_length_sweep_spectra.npz"

    n_list = [4, 8] if quick else [5, 10, 20, 40]
    n_ref = n_list[-1] if quick else 20

    runs_clean: dict[str, Any] = {}
    npz_dict: dict[str, np.ndarray] = {}

    if reuse_on_axis_length_sweep and not quick and json_path.is_file() and npz_path.is_file():
        old_t2 = json.loads(json_path.read_text(encoding="utf-8"))
        old_z2 = np.load(npz_path)
        for k_file in old_z2.files:
            npz_dict[k_file] = np.asarray(old_z2[k_file])
        for n_val in n_list:
            r_old = dict(old_t2["runs"][str(n_val)])
            r_old.setdefault("taper_scales", [0.25, 0.55])
            r_old.setdefault("transverse_width_m", float(spec.transverse_width_m))
            r_old.setdefault("width_scale", 1.0)
            r_old.setdefault("f_search_band_ghz", [float(f_band[0]), float(f_band[1])])
            runs_clean[str(n_val)] = r_old
    else:
        for n_val in n_list:
            r = run_2d_karetnikova_beam_wake(
                spec,
                n_periods=n_val,
                n_feat=4,
                use_graded_mesh=quick,
                f_search_band_ghz=f_band,
                quick=quick,
            )
            runs_clean[str(n_val)] = _strip_heavy_arrays(r)
            npz_dict[f"f_ghz_N{n_val}"] = np.asarray(r["f_ghz"])
            npz_dict[f"re_z_ohm_N{n_val}"] = np.asarray(r["re_z_ohm"])
            npz_dict[f"im_z_ohm_N{n_val}"] = np.asarray(r["im_z_ohm"])
            npz_dict[f"s_m_N{n_val}"] = np.asarray(r["s_m"])
            npz_dict[f"wp_v_per_pc_N{n_val}"] = np.asarray(r["wp_v_per_pc"])

    L_arr = np.array([runs_clean[str(n)]["active_length_m"] for n in n_list], dtype=float)
    z_pk_arr = np.array([runs_clean[str(n)]["re_z_pk_ohm"] for n in n_list], dtype=float)
    fwhm_arr = np.array([runs_clean[str(n)]["fwhm_ghz"] for n in n_list], dtype=float)

    slope_pk = float(np.polyfit(np.log(L_arr), np.log(z_pk_arr), 1)[0])
    slope_fwhm = float(np.polyfit(np.log(L_arr), np.log(fwhm_arr), 1)[0])

    # Run off-axis sheet-beam y-sweep at N = n_ref across y0 in {0, 25, 50, 75} um
    y_pos_um = (0.0, 50.0, 75.0) if quick else (0.0, 25.0, 50.0, 75.0)
    y_sweep = run_2d_sheet_beam_y_sweep(
        spec,
        n_periods=n_ref,
        n_feat=4,
        use_graded_mesh=quick,
        y_positions_um=y_pos_um,
        on_axis_run=runs_clean[str(n_ref)],
        quick=quick,
    )

    sigmaz_mm_arr = np.linspace(0.05, 0.45, 9)
    f_ref_ghz = np.asarray(npz_dict[f"f_ghz_N{n_ref}"], dtype=float)
    rz_ref_ohm = np.asarray(npz_dict[f"re_z_ohm_N{n_ref}"], dtype=float)
    mask_ref = (f_ref_ghz >= f_band[0]) & (f_ref_ghz <= f_band[1])
    f_ref_b = f_ref_ghz[mask_ref] * 1e9
    rz_ref_b = np.maximum(rz_ref_ohm[mask_ref], 0.0)
    L_ref = float(runs_clean[str(n_ref)]["active_length_m"])
    rz_th_b = analytic_sws_beam_impedance(
        f_ref_b,
        f_sync_hz=l2["f_sync_bwd_hz"],
        beta_n_rad_m=l2["beta_m1_rad_m"],
        k_pierce_ohm=l2["k_m1_on_axis_ohm"],
        length_m=L_ref,
        u0_m_s=u0,
        vg_m_s=l2["vg_bwd_m_s"],
    )
    k_loss_fit_list = [
        compute_loss_factor_v_per_pc(f_ref_b, rz_ref_b, float(sm * 1e-3), u0)
        for sm in sigmaz_mm_arr
    ]
    k_loss_th_list = [
        compute_loss_factor_v_per_pc(f_ref_b, rz_th_b, float(sm * 1e-3), u0)
        for sm in sigmaz_mm_arr
    ]

    dw_dz_th_unatt = analytic_stopping_power_j_per_m(
        q_coulomb=1.0e-12,
        beta_n_rad_m=l2["beta_m1_rad_m"],
        k_pierce_ohm=l2["k_m1_on_axis_ohm"],
        u0_m_s=u0,
        vg_m_s=l2["vg_bwd_m_s"],
        form_factor_sq=1.0,
    )
    ff_220 = gaussian_bunch_form_factor_sq(l2["f_sync_bwd_hz"], 0.22e-3, u0)

    out_t2 = {
        "spec": asdict(spec),
        "n_periods_list": [int(n) for n in n_list],
        "u0_m_s": float(u0),
        "u0_over_c": float(u0 / c),
        "taper_scales": [0.25, 0.55],
        "f_search_band_ghz": [float(f_band[0]), float(f_band[1])],
        "cold_test_L1": {
            k: v
            for k, v in l1.items()
            if not k.endswith("_array") and k not in ("mirror_fwd_n0", "glide_fwd_p1", "f_glide_low_ghz", "f_mirror_low_ghz")
        },
        "cold_test_L2": {
            k: v
            for k, v in l2.items()
            if not k.endswith("_array") and k not in ("mirror_fwd_n0", "glide_fwd_p1", "f_glide_low_ghz", "f_mirror_low_ghz")
        },
        "runs": runs_clean,
        "sheet_beam_y_sweep": y_sweep,
        "power_law_slope_peak_vs_L": slope_pk,
        "power_law_slope_fwhm_vs_L": slope_fwhm,
        "dw_dz_analytic_unattenuated_j_per_m": float(dw_dz_th_unatt),
        "dw_dz_analytic_attenuated_j_per_m": float(dw_dz_th_unatt * ff_220),
        "sigmaz_sweep_mm": [float(x) for x in sigmaz_mm_arr],
        "k_loss_fit_N20_v_per_pc": k_loss_fit_list,
        "k_loss_analytic_N20_v_per_pc": k_loss_th_list,
    }

    np.savez_compressed(npz_path, **npz_dict)
    json_path.write_text(json.dumps(out_t2, indent=2) + "\n", encoding="utf-8")
    return out_t2


def generate_task_03_data(
    data_dir: Path | str,
    *,
    quick: bool = False,
) -> dict[str, Any]:
    """Regenerate ``task_03_fwd_vs_bwd_poynting.json`` and ``task_03_fwd_vs_bwd_snapshots.npz``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    spec = KaretnikovaGratingSpec()
    bench = load_cold_test_benchmarks()
    l1 = bench["levels_2d"]["L1"]
    l2 = bench["levels_2d"]["L2"]
    u0_bwd = bench["u0_2d_m_s"]

    # Derive forward-wave control parameters directly from Tutorial 6.2 mirror-mode dispersion
    mir_l1 = l1["mirror_fwd_n0"]
    mir_l2 = l2["mirror_fwd_n0"]
    u0_fwd = float(mir_l2["u0_m_s"])
    v0_fwd = float(mir_l2["beam_voltage_v"])
    y_fwd_m = float(mir_l2["ysource_m"])
    f_band_bwd = (95.0, 132.0)
    f_band_fwd = (95.0, 145.0)
    n_per = 6 if quick else 10

    r_bwd = run_2d_karetnikova_beam_wake(
        spec,
        n_periods=n_per,
        n_feat=4,
        use_graded_mesh=quick,
        symmetry="glide",
        u0_m_s=u0_bwd,
        ysource_m=0.0,
        sigmaz_m=0.22e-3,
        f_search_band_ghz=f_band_bwd,
        f_sync_target_ghz=float(l1["f_sync_bwd_ghz"]),
        record_poynting_and_snapshots=True,
        quick=quick,
    )
    r_fwd = run_2d_karetnikova_beam_wake(
        spec,
        n_periods=n_per,
        n_feat=4,
        use_graded_mesh=quick,
        symmetry="mirror",
        u0_m_s=u0_fwd,
        ysource_m=y_fwd_m,
        sigmaz_m=0.22e-3,
        f_search_band_ghz=f_band_fwd,
        f_sync_target_ghz=float(mir_l1["f_sync_ghz"]),
        record_poynting_and_snapshots=True,
        quick=quick,
    )

    L_act = float(r_bwd["active_length_m"])
    fwhm_bwd_th = (
        analytic_fwhm_hz(length_m=L_act, u0_m_s=u0_bwd, vg_m_s=l2["vg_bwd_m_s"]) * 1e-9
    )
    fwhm_bwd_mut = (
        analytic_fwhm_hz(length_m=L_act, u0_m_s=u0_bwd, vg_m_s=-l2["vg_bwd_m_s"]) * 1e-9
    )
    fwhm_fwd_lin_l1 = (
        analytic_fwhm_hz(length_m=L_act, u0_m_s=u0_fwd, vg_m_s=mir_l1["vg_m_s"]) * 1e-9
    )
    fwhm_fwd_lin_l2 = (
        analytic_fwhm_hz(length_m=L_act, u0_m_s=u0_fwd, vg_m_s=mir_l2["vg_m_s"]) * 1e-9
    )

    p_bwd = r_bwd["poynting"]
    p_fwd = r_fwd["poynting"]

    out_t3 = {
        "backward_wave_glide_m1": {
            "n_periods": int(n_per),
            "active_length_mm": float(L_act * 1e3),
            "symmetry": "glide",
            "space_harmonic_n": -1,
            "beam_voltage_v": float(spec.beam_voltage_v),
            "u0_over_c": float(u0_bwd / c),
            "vg_over_c": float(l2["vg_bwd_over_c"]),
            "vg_cold_L1_over_c": float(l1["vg_bwd_over_c"]),
            "ysource_um": 0.0,
            "f_search_band_ghz": [float(f_band_bwd[0]), float(f_band_bwd[1])],
            "taper_scales": [0.25, 0.55],
            "f_sync_cold_L1_ghz": float(l1["f_sync_bwd_ghz"]),
            "f_sync_cold_L2_ghz": float(l2["f_sync_bwd_ghz"]),
            "f_pk_ghz": float(r_bwd["f_pk_ghz"]),
            "re_z_pk_ohm": float(r_bwd["re_z_pk_ohm"]),
            "k_on_axis_beam_ohm": float(r_bwd["k_on_axis_beam_ohm"]),
            "k_cold_L1_on_axis_ohm": float(l1["k_m1_on_axis_ohm"]),
            "k_cold_L2_on_axis_ohm": float(l2["k_m1_on_axis_ohm"]),
            "fwhm_ghz": float(r_bwd["fwhm_ghz"]),
            "fwhm_analytic_ghz": float(fwhm_bwd_th),
            "fwhm_mutated_flipped_vg_ghz": float(fwhm_bwd_mut),
            "integrated_energy_in_j": float(p_bwd["integrated_energy_in_j"]),
            "integrated_energy_out_j": float(p_bwd["integrated_energy_out_j"]),
            "total_radiated_cpml_energy_j": float(p_bwd["total_radiated_cpml_energy_j"]),
            "w_loss_active_j": float(p_bwd["w_loss_active_j"]),
            "w_loss_between_smooth_ports_j": float(p_bwd["w_loss_between_smooth_ports_j"]),
            "energy_balance_rel_error_vs_active": float(
                p_bwd["energy_balance_rel_error_vs_active"]
            ),
            "energy_balance_rel_error_vs_total": float(
                p_bwd["energy_balance_rel_error_vs_total"]
            ),
            "upstream_fraction": float(p_bwd["upstream_fraction"]),
            "downstream_fraction": float(p_bwd["downstream_fraction"]),
            "z_in_mm": float(p_bwd["z_in_mm"]),
            "z_out_mm": float(p_bwd["z_out_mm"]),
            "y_integration_bounds_mm": list(p_bwd["y_integration_bounds_mm"]),
            "tunnel_only_groove_plane_energy_in_j": float(
                p_bwd["tunnel_only_groove_plane_energy_in_j"]
            ),
            "tunnel_only_groove_plane_energy_out_j": float(
                p_bwd["tunnel_only_groove_plane_energy_out_j"]
            ),
            "wall_time_s": float(r_bwd["wall_time_s"]),
        },
        "forward_wave_mirror_n0": {
            "n_periods": int(n_per),
            "active_length_mm": float(L_act * 1e3),
            "symmetry": "mirror",
            "space_harmonic_n": 0,
            "beam_voltage_v": float(v0_fwd),
            "u0_over_c": float(u0_fwd / c),
            "u0_derivation": "Tutorial 6.2 L2 mirror-mode dispersion at beta0*d = 0.75*pi",
            "vg_over_c": float(mir_l2["vg_over_c"]),
            "vg_cold_L1_over_c": float(mir_l1["vg_over_c"]),
            "vg_derivation": "Finite difference 2*d*df/d(beta0*d/pi) of Tutorial 6.2 f_mirror_low_ghz",
            "ysource_um": float(y_fwd_m * 1e6),
            "f_search_band_ghz": [float(f_band_fwd[0]), float(f_band_fwd[1])],
            "taper_scales": [0.25, 0.55],
            "f_sync_cold_L1_ghz": float(mir_l1["f_sync_ghz"]),
            "f_sync_cold_L2_ghz": float(mir_l2["f_sync_ghz"]),
            "k0_on_axis_cold_L2_ohm": float(mir_l2["k_0_on_axis_ohm"]),
            "k0_node_avg_cold_L1_ohm": float(mir_l1["k_0_beam_avg_ohm"]),
            "k0_node_avg_cold_L2_ohm": float(mir_l2["k_0_beam_avg_ohm"]),
            "k0_at_sync_y50um_cold_L1_ohm": float(mir_l1["k_0_at_y50um_ohm"]),
            "k0_at_sync_y50um_cold_L2_ohm": float(mir_l2["k_0_at_y50um_ohm"]),
            "re_z_at_sync_y50um_cold_L1_ohm": float(mir_l1["re_z_sync_N10_y50um_ohm"]),
            "re_z_at_sync_y50um_cold_L2_ohm": float(mir_l2["re_z_sync_N10_y50um_ohm"]),
            "re_z_at_sync_L1_fit_ohm": float(r_fwd["re_z_at_sync_target_ohm"]),
            "k0_at_sync_y50um_fit_ohm": float(r_fwd["k_at_sync_target_ohm"]),
            "rel_diff_k0_at_sync_vs_cold_L1": float(
                abs(r_fwd["k_at_sync_target_ohm"] - mir_l1["k_0_at_y50um_ohm"])
                / mir_l1["k_0_at_y50um_ohm"]
            ),
            "f_pk_ghz": float(r_fwd["f_pk_ghz"]),
            "re_z_pk_ohm": float(r_fwd["re_z_pk_ohm"]),
            "k0_at_fpk_y50um_fit_ohm": float(r_fwd["k_on_axis_beam_ohm"]),
            "f_pk_nonlinear_cold_L1_ghz": float(mir_l1["f_pk_nonlinear_N10_ghz"]),
            "re_z_pk_nonlinear_cold_L1_ohm": float(mir_l1["re_z_pk_nonlinear_N10_ohm"]),
            "k0_at_fpk_nonlinear_cold_L1_ohm": float(
                mir_l1["k_0_at_fpk_nonlinear_y50um_ohm"]
            ),
            "rel_diff_re_z_pk_vs_nonlinear_cold_L1": float(
                abs(r_fwd["re_z_pk_ohm"] - mir_l1["re_z_pk_nonlinear_N10_ohm"])
                / mir_l1["re_z_pk_nonlinear_N10_ohm"]
            ),
            "f_upper_stopband_cold_L1_ghz": float(mir_l1["f_upper_stopband_ghz"]),
            "fwhm_ghz": float(r_fwd["fwhm_ghz"]),
            "f_half_left_ghz": float(r_fwd["f_half_left_ghz"]),
            "f_half_right_ghz": float(r_fwd["f_half_right_ghz"]),
            "fwhm_linear_analytic_L1_ghz": float(fwhm_fwd_lin_l1),
            "fwhm_linear_analytic_L2_ghz": float(fwhm_fwd_lin_l2),
            "fwhm_nonlinear_cold_L1_ghz": float(mir_l1["fwhm_nonlinear_N10_ghz"]),
            "integrated_energy_in_j": float(p_fwd["integrated_energy_in_j"]),
            "integrated_energy_out_j": float(p_fwd["integrated_energy_out_j"]),
            "total_radiated_cpml_energy_j": float(p_fwd["total_radiated_cpml_energy_j"]),
            "w_loss_active_j": float(p_fwd["w_loss_active_j"]),
            "w_loss_between_smooth_ports_j": float(p_fwd["w_loss_between_smooth_ports_j"]),
            "energy_balance_rel_error_vs_active": float(
                p_fwd["energy_balance_rel_error_vs_active"]
            ),
            "energy_balance_rel_error_vs_total": float(
                p_fwd["energy_balance_rel_error_vs_total"]
            ),
            "upstream_fraction": float(p_fwd["upstream_fraction"]),
            "downstream_fraction": float(p_fwd["downstream_fraction"]),
            "z_in_mm": float(p_fwd["z_in_mm"]),
            "z_out_mm": float(p_fwd["z_out_mm"]),
            "y_integration_bounds_mm": list(p_fwd["y_integration_bounds_mm"]),
            "tunnel_only_groove_plane_energy_in_j": float(
                p_fwd["tunnel_only_groove_plane_energy_in_j"]
            ),
            "tunnel_only_groove_plane_energy_out_j": float(
                p_fwd["tunnel_only_groove_plane_energy_out_j"]
            ),
            "wall_time_s": float(r_fwd["wall_time_s"]),
        },
    }

    npz_path = data_dir / "task_03_fwd_vs_bwd_snapshots.npz"
    np.savez_compressed(
        npz_path,
        t_bwd_ps=np.asarray(p_bwd["t_ps"]),
        p_in_bwd_w=np.asarray(p_bwd["p_in_w"]),
        p_out_bwd_w=np.asarray(p_bwd["p_out_w"]),
        f_bwd_ghz=np.asarray(r_bwd["f_ghz"]),
        re_z_bwd_ohm=np.asarray(r_bwd["re_z_ohm"]),
        y_bwd_mm=np.asarray(r_bwd["y_mm"]),
        z_bwd_mm=np.asarray(r_bwd["z_mm"]),
        snap_bwd_t_ps=np.array([s["t_ps"] for s in r_bwd["snapshots"]], dtype=float),
        snap_bwd_z_mm=np.array(
            [s["z_bunch_mm"] for s in r_bwd["snapshots"]], dtype=float
        ),
        snap_bwd_ez=np.array([s["ez_2d"] for s in r_bwd["snapshots"]], dtype=float),
        t_fwd_ps=np.asarray(p_fwd["t_ps"]),
        p_in_fwd_w=np.asarray(p_fwd["p_in_w"]),
        p_out_fwd_w=np.asarray(p_fwd["p_out_w"]),
        f_fwd_ghz=np.asarray(r_fwd["f_ghz"]),
        re_z_fwd_ohm=np.asarray(r_fwd["re_z_ohm"]),
        y_fwd_mm=np.asarray(r_fwd["y_mm"]),
        z_fwd_mm=np.asarray(r_fwd["z_mm"]),
        snap_fwd_t_ps=np.array([s["t_ps"] for s in r_fwd["snapshots"]], dtype=float),
        snap_fwd_z_mm=np.array(
            [s["z_bunch_mm"] for s in r_fwd["snapshots"]], dtype=float
        ),
        snap_fwd_ez=np.array([s["ez_2d"] for s in r_fwd["snapshots"]], dtype=float),
    )

    json_path = data_dir / "task_03_fwd_vs_bwd_poynting.json"
    json_path.write_text(json.dumps(out_t3, indent=2) + "\n", encoding="utf-8")
    return out_t3


def generate_task_04_data(
    data_dir: Path | str,
    *,
    quick: bool = False,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Regenerate ``task_04_mutations_detuning_and_cavity.json`` and ``task_04_mutations_spectra.npz``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "task_04_mutations_detuning_and_cavity.json"
    npz_path = data_dir / "task_04_mutations_spectra.npz"
    bench = load_cold_test_benchmarks()
    l2 = bench["levels_2d"]["L2"]
    u0_nom = bench["u0_2d_m_s"]
    f_band = (95.0, 132.0)

    if reuse_existing and not quick and json_path.is_file() and npz_path.is_file():
        t4 = json.loads(json_path.read_text(encoding="utf-8"))
        t4["f_search_band_ghz"] = [float(f_band[0]), float(f_band[1])]
        t4["taper_scales"] = [0.25, 0.55]
        t4["vg_bwd_cold_L2_m_s"] = float(l2["vg_bwd_m_s"])
        t4["vg_bwd_cold_L2_over_c"] = float(l2["vg_bwd_over_c"])
        json_path.write_text(json.dumps(t4, indent=2) + "\n", encoding="utf-8")
        return t4

    spec = KaretnikovaGratingSpec()
    n_per = 6 if quick else 10
    r_nom = run_2d_karetnikova_beam_wake(
        spec, n_periods=n_per, n_feat=4, use_graded_mesh=quick, u0_m_s=u0_nom, f_search_band_ghz=f_band, quick=quick
    )
    r_low = run_2d_karetnikova_beam_wake(
        spec, n_periods=n_per, n_feat=4, use_graded_mesh=quick, u0_m_s=0.95 * u0_nom, f_search_band_ghz=f_band, quick=quick
    )
    r_high = run_2d_karetnikova_beam_wake(
        spec, n_periods=n_per, n_feat=4, use_graded_mesh=quick, u0_m_s=1.05 * u0_nom, f_search_band_ghz=f_band, quick=quick
    )
    r_cav = run_2d_karetnikova_beam_wake(
        spec,
        n_periods=n_per,
        n_feat=4,
        use_graded_mesh=quick,
        termination_mode="pec_cavity",
        f_search_band_ghz=(95.0, 134.0),
        quick=quick,
    )

    f_nom = float(r_nom["f_pk_ghz"])
    df_low_th = (
        analytic_detuned_frequency_shift_hz(
            f_sync_hz=f_nom * 1e9,
            u0_m_s=u0_nom,
            delta_u0_m_s=-0.05 * u0_nom,
            vg_m_s=l2["vg_bwd_m_s"],
        )
        * 1e-9
    )
    df_high_th = (
        analytic_detuned_frequency_shift_hz(
            f_sync_hz=f_nom * 1e9,
            u0_m_s=u0_nom,
            delta_u0_m_s=+0.05 * u0_nom,
            vg_m_s=l2["vg_bwd_m_s"],
        )
        * 1e-9
    )

    # Extract cavity standing-wave modes m around synchronism
    bn_l1 = np.array(bench["levels_2d"]["L1"]["bn_array"], dtype=float)
    f_l1_ghz = np.array(bench["levels_2d"]["L1"]["f_glide_low_ghz"], dtype=float)
    km1_l1 = np.array(bench["levels_2d"]["L1"]["km1_axis_ohm_array"], dtype=float)
    f_cav = np.asarray(r_cav["f_ghz"], dtype=float)
    rz_cav = np.asarray(r_cav["re_z_ohm"], dtype=float)
    L_act = float(r_cav["active_length_m"])
    d = spec.period_m

    modes_list = []
    m_indices = [3, 4, 5] if quick else [5, 6, 7]
    for m_idx in m_indices:
        bn_m = float(m_idx) / float(n_per)
        f_cold_m = float(np.interp(bn_m, bn_l1, f_l1_ghz))
        k_cold_m = float(np.interp(bn_m, bn_l1, km1_l1))
        beta_m1_m = (2.0 - bn_m) * np.pi / d
        win = (f_cav >= f_cold_m - 2.2) & (f_cav <= f_cold_m + 2.2)
        f_w = f_cav[win]
        rz_w = np.maximum(rz_cav[win], 0.0)
        i_m = int(np.argmax(rz_w))
        f_fit_m = float(f_w[i_m])
        z_pk_m = float(rz_w[i_m])
        area_rad_s_ohm = float(2.0 * np.pi * np.trapezoid(rz_w * 1e9, f_w))
        rq_circ_fit = float(area_rad_s_ohm / (np.pi * (2.0 * np.pi * f_fit_m * 1e9)))
        rq_circ_th = analytic_standing_wave_r_over_q_ohm(
            beta_n_rad_m=beta_m1_m,
            k_pierce_ohm=k_cold_m,
            length_m=L_act,
            f_mode_hz=f_cold_m * 1e9,
            vg_m_s=l2["vg_bwd_m_s"],
            convention="circuit",
        )
        modes_list.append(
            {
                "mode_index_m": int(m_idx),
                "beta0_d_over_pi": bn_m,
                "beta_m1_rad_m": float(beta_m1_m),
                "f_cold_test_ghz": f_cold_m,
                "f_fit_cavity_ghz": f_fit_m,
                "re_z_peak_ohm": z_pk_m,
                "integrated_area_rad_s_ohm": area_rad_s_ohm,
                "r_over_q_circuit_fit_ohm": rq_circ_fit,
                "r_over_q_linac_fit_ohm": float(2.0 * rq_circ_fit),
                "r_over_q_circuit_analytic_ohm": float(rq_circ_th),
                "r_over_q_linac_analytic_ohm": float(2.0 * rq_circ_th),
                "r_over_q_mutated_no_half_ohm": float(2.0 * rq_circ_th),
            }
        )

    fsr_hz = abs(float(l2["vg_bwd_m_s"])) / (2.0 * L_act)
    m_sync_info = modes_list[1]
    rec_tw_z_pk = float(m_sync_info["integrated_area_rad_s_ohm"] / (2.0 * np.pi * fsr_hz))

    out_t4 = {
        "f_search_band_ghz": [float(f_band[0]), float(f_band[1])],
        "taper_scales": [0.25, 0.55],
        "vg_bwd_cold_L2_m_s": float(l2["vg_bwd_m_s"]),
        "vg_bwd_cold_L2_over_c": float(l2["vg_bwd_over_c"]),
        "detuning_sweep": {
            "u0_nominal": {
                "u0_over_c": float(u0_nom / c),
                "f_pk_ghz": f_nom,
                "re_z_pk_ohm": float(r_nom["re_z_pk_ohm"]),
            },
            "u0_minus_5pct": {
                "u0_over_c": float(0.95 * u0_nom / c),
                "f_pk_ghz": float(r_low["f_pk_ghz"]),
                "re_z_pk_ohm": float(r_low["re_z_pk_ohm"]),
                "delta_f_fit_ghz": float(r_low["f_pk_ghz"] - f_nom),
                "delta_f_linear_analytic_ghz": float(df_low_th),
                "delta_f_mutated_flipped_sign_ghz": float(-df_low_th),
            },
            "u0_plus_5pct": {
                "u0_over_c": float(1.05 * u0_nom / c),
                "f_pk_ghz": float(r_high["f_pk_ghz"]),
                "re_z_pk_ohm": float(r_high["re_z_pk_ohm"]),
                "delta_f_fit_ghz": float(r_high["f_pk_ghz"] - f_nom),
                "delta_f_linear_analytic_ghz": float(df_high_th),
                "delta_f_mutated_flipped_sign_ghz": float(-df_high_th),
            },
        },
        "pec_cavity_resonances": {
            "n_periods": int(n_per),
            "active_length_m": L_act,
            "fsr_ghz": float(fsr_hz * 1e-9),
            "modes": modes_list,
            "synchronous_mode_m6_r_over_q_circuit_fit_ohm": float(
                m_sync_info["r_over_q_circuit_fit_ohm"]
            ),
            "synchronous_mode_m6_r_over_q_circuit_analytic_ohm": float(
                m_sync_info["r_over_q_circuit_analytic_ohm"]
            ),
            "synchronous_mode_m6_r_over_q_linac_fit_ohm": float(
                m_sync_info["r_over_q_linac_fit_ohm"]
            ),
            "synchronous_mode_m6_r_over_q_linac_analytic_ohm": float(
                m_sync_info["r_over_q_linac_analytic_ohm"]
            ),
            "reconstructed_traveling_wave_z_pk_ohm": rec_tw_z_pk,
            "matched_traveling_wave_z_pk_ohm": float(r_nom["re_z_pk_ohm"]),
            "wall_time_s": float(r_cav["wall_time_s"]),
        },
    }

    np.savez_compressed(
        npz_path,
        f_u0_ghz=np.asarray(r_nom["f_ghz"]),
        re_z_u0_nom_ohm=np.asarray(r_nom["re_z_ohm"]),
        re_z_u0_low_ohm=np.asarray(r_low["re_z_ohm"]),
        re_z_u0_high_ohm=np.asarray(r_high["re_z_ohm"]),
        f_cav_ghz=np.asarray(r_cav["f_ghz"]),
        re_z_cav_ohm=np.asarray(r_cav["re_z_ohm"]),
    )
    json_path.write_text(json.dumps(out_t4, indent=2) + "\n", encoding="utf-8")
    return out_t4


def generate_task_05_data(
    data_dir: Path | str,
    *,
    quick: bool = False,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Regenerate ``task_05_mesh_convergence.json``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "task_05_mesh_convergence.json"
    bench = load_cold_test_benchmarks()
    l1 = bench["levels_2d"]["L1"]
    l2 = bench["levels_2d"]["L2"]
    l_ref = bench["levels_2d"]["ref"]

    spec = KaretnikovaGratingSpec()
    n_per = 4 if quick else 8
    nf_list = [4, 6, 8]
    runs_conv: dict[str, Any] = {}
    if reuse_existing and not quick and json_path.is_file():
        old_t5 = json.loads(json_path.read_text(encoding="utf-8"))
        runs_conv = dict(old_t5["runs"])
    else:
        for nf in nf_list:
            r_u = run_2d_karetnikova_beam_wake(
                spec,
                n_periods=n_per,
                n_feat=nf,
                use_graded_mesh=False,
                conformal_mode="pec",
                quick=quick,
            )
            runs_conv[f"uniform_conformal_nf{nf}"] = _strip_heavy_arrays(r_u)

        r_grad = run_2d_karetnikova_beam_wake(
            spec,
            n_periods=n_per,
            n_feat=4,
            use_graded_mesh=True,
            conformal_mode="pec",
            quick=quick,
        )
        runs_conv["graded_conformal_nf4"] = _strip_heavy_arrays(r_grad)

        r_stair = run_2d_karetnikova_beam_wake(
            spec,
            n_periods=n_per,
            n_feat=4,
            use_graded_mesh=False,
            conformal_mode=False,
            quick=quick,
        )
        runs_conv["uniform_staircase_nf4"] = _strip_heavy_arrays(r_stair)

    h_um = [float(spec.vane_thickness_m * 1e6 / nf) for nf in nf_list]
    f_arr = [float(runs_conv[f"uniform_conformal_nf{nf}"]["f_pk_ghz"]) for nf in nf_list]
    z_arr = [float(runs_conv[f"uniform_conformal_nf{nf}"]["re_z_pk_ohm"]) for nf in nf_list]
    k_ax_arr = [float(runs_conv[f"uniform_conformal_nf{nf}"]["k_on_axis_beam_ohm"]) for nf in nf_list]
    k_sh_arr = [float(runs_conv[f"uniform_conformal_nf{nf}"]["k_sheet_avg_beam_ohm"]) for nf in nf_list]

    is_mono_f = bool((f_arr[0] < f_arr[1] < f_arr[2]) or (f_arr[0] > f_arr[1] > f_arr[2]))
    is_mono_k = bool((k_ax_arr[0] < k_ax_arr[1] < k_ax_arr[2]) or (k_ax_arr[0] > k_ax_arr[1] > k_ax_arr[2]))

    # Because the 3-level sequence across n_feat in {4, 6, 8} is non-monotone, no Richardson
    # extrapolation order is justified; report converged quantities as a band across the finest levels (n_feat in {6, 8}).
    f_bmin = float(min(f_arr[1], f_arr[2]))
    f_bmax = float(max(f_arr[1], f_arr[2]))
    f_bmid = float(0.5 * (f_bmin + f_bmax))
    f_bspr = float(f_bmax - f_bmin)

    z_bmin = float(min(z_arr[1], z_arr[2]))
    z_bmax = float(max(z_arr[1], z_arr[2]))
    z_bmid = float(0.5 * (z_bmin + z_bmax))
    z_bspr = float(z_bmax - z_bmin)

    k_bmin = float(min(k_ax_arr[1], k_ax_arr[2]))
    k_bmax = float(max(k_ax_arr[1], k_ax_arr[2]))
    k_bmid = float(0.5 * (k_bmin + k_bmax))
    k_bspr = float(k_bmax - k_bmin)

    ks_bmin = float(min(k_sh_arr[1], k_sh_arr[2]))
    ks_bmax = float(max(k_sh_arr[1], k_sh_arr[2]))
    ks_bmid = float(0.5 * (ks_bmin + ks_bmax))
    ks_bspr = float(ks_bmax - ks_bmin)

    k_all_min = float(min(k_ax_arr))
    k_all_max = float(max(k_ax_arr))
    k_all_spr = float(k_all_max - k_all_min)

    k_l1 = float(l1["k_m1_on_axis_ohm"])
    k_l2 = float(l2["k_m1_on_axis_ohm"])
    k_ref = float(l_ref["k_m1_on_axis_ohm"])
    r_grad_d = runs_conv["graded_conformal_nf4"]
    r_stair_d = runs_conv["uniform_staircase_nf4"]

    out_t5 = {
        "n_periods": int(n_per),
        "f_search_band_ghz": [95.0, 132.0],
        "taper_scales": [0.25, 0.55],
        "n_feat_levels": nf_list,
        "h_um": h_um,
        "f_pk_ghz_array": f_arr,
        "re_z_pk_ohm_array": z_arr,
        "k_on_axis_ohm_array": k_ax_arr,
        "k_sheet_avg_ohm_array": k_sh_arr,
        "is_monotone_f_pk": is_mono_f,
        "is_monotone_k_on_axis": is_mono_k,
        "richardson_extrapolation_justified": bool(is_mono_k),
        "f_pk_finest_band_ghz": [f_bmin, f_bmax],
        "f_pk_finest_band_min_ghz": f_bmin,
        "f_pk_finest_band_max_ghz": f_bmax,
        "f_pk_finest_band_mid_ghz": f_bmid,
        "f_pk_finest_band_spread_ghz": f_bspr,
        "f_pk_finest_band_rel_spread": float(f_bspr / max(f_bmid, 1e-30)),
        "re_z_pk_finest_band_ohm": [z_bmin, z_bmax],
        "re_z_pk_finest_band_min_ohm": z_bmin,
        "re_z_pk_finest_band_max_ohm": z_bmax,
        "re_z_pk_finest_band_mid_ohm": z_bmid,
        "re_z_pk_finest_band_spread_ohm": z_bspr,
        "re_z_pk_finest_band_rel_spread": float(z_bspr / max(z_bmid, 1e-30)),
        "k_on_axis_finest_band_ohm": [k_bmin, k_bmax],
        "k_on_axis_finest_band_min_ohm": k_bmin,
        "k_on_axis_finest_band_max_ohm": k_bmax,
        "k_on_axis_finest_band_mid_ohm": k_bmid,
        "k_on_axis_finest_band_spread_ohm": k_bspr,
        "k_on_axis_finest_band_rel_spread": float(k_bspr / max(k_bmid, 1e-30)),
        "k_on_axis_all_levels_band_ohm": [k_all_min, k_all_max],
        "k_on_axis_all_levels_spread_ohm": k_all_spr,
        "k_on_axis_all_levels_rel_spread": float(k_all_spr / max(float(np.mean(k_ax_arr)), 1e-30)),
        "k_sheet_avg_finest_band_ohm": [ks_bmin, ks_bmax],
        "k_sheet_avg_finest_band_min_ohm": ks_bmin,
        "k_sheet_avg_finest_band_max_ohm": ks_bmax,
        "k_sheet_avg_finest_band_mid_ohm": ks_bmid,
        "k_sheet_avg_finest_band_spread_ohm": ks_bspr,
        "k_sheet_avg_finest_band_rel_spread": float(ks_bspr / max(ks_bmid, 1e-30)),
        "rel_diff_nf6_vs_nf4_same_N": float((k_ax_arr[1] - k_ax_arr[0]) / k_ax_arr[0]),
        "rel_diff_nf8_vs_nf4_same_N": float((k_ax_arr[2] - k_ax_arr[0]) / k_ax_arr[0]),
        "rel_diff_finest_band_mid_vs_nf4_same_N": float((k_bmid - k_ax_arr[0]) / k_ax_arr[0]),
        "cold_test_L1_f_sync_ghz": float(l1["f_sync_bwd_ghz"]),
        "cold_test_L1_k_on_axis_ohm": k_l1,
        "cold_test_L2_f_sync_ghz": float(l2["f_sync_bwd_ghz"]),
        "cold_test_L2_k_on_axis_ohm": k_l2,
        "cold_test_L2_k_beam_avg_ohm": float(l2["k_m1_beam_avg_ohm"]),
        "cold_test_ref_f_sync_ghz": float(l_ref["f_sync_bwd_ghz"]),
        "cold_test_ref_k_on_axis_ohm": k_ref,
        "cold_test_ref_k_beam_avg_ohm": float(l_ref["k_m1_beam_avg_ohm"]),
        "rel_diff_nf4_vs_cold_L1": float(abs(k_ax_arr[0] - k_l1) / k_l1),
        "rel_diff_nf6_vs_cold_L2": float(abs(k_ax_arr[1] - k_l2) / k_l2),
        "rel_diff_nf8_vs_cold_L2": float(abs(k_ax_arr[2] - k_l2) / k_l2),
        "rel_diff_finest_band_vs_cold_L2_min": float(
            min(abs(k_bmin - k_l2), abs(k_bmax - k_l2)) / k_l2
        ),
        "rel_diff_finest_band_vs_cold_L2_max": float(
            max(abs(k_bmin - k_l2), abs(k_bmax - k_l2)) / k_l2
        ),
        "rel_diff_finest_band_mid_vs_cold_L2": float(abs(k_bmid - k_l2) / k_l2),
        "rel_diff_finest_band_vs_cold_ref_min": float(
            min(abs(k_bmin - k_ref), abs(k_bmax - k_ref)) / k_ref
        ),
        "rel_diff_finest_band_vs_cold_ref_max": float(
            max(abs(k_bmin - k_ref), abs(k_bmax - k_ref)) / k_ref
        ),
        "rel_diff_finest_band_mid_vs_cold_ref": float(abs(k_bmid - k_ref) / k_ref),
        "graded_vs_uniform_rel_diff_f_pk": float(
            abs(float(r_grad_d["f_pk_ghz"]) - f_arr[0]) / f_arr[0]
        ),
        "graded_vs_uniform_rel_diff_k_axis": float(
            abs(float(r_grad_d["k_on_axis_beam_ohm"]) - k_ax_arr[0]) / k_ax_arr[0]
        ),
        "staircase_over_conformal_k_ratio": float(
            float(r_stair_d["k_on_axis_beam_ohm"]) / k_ax_arr[0]
        ),
        "runs": runs_conv,
    }
    json_path.write_text(json.dumps(out_t5, indent=2) + "\n", encoding="utf-8")
    return out_t5


def generate_task_06_data(
    data_dir: Path | str,
    *,
    quick: bool = False,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Regenerate ``task_06_3d_marosi_beam_wake.json`` and ``task_06_3d_marosi_spectrum.npz``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "task_06_3d_marosi_beam_wake.json"
    npz_path = data_dir / "task_06_3d_marosi_spectrum.npz"
    spec_3d = MarosiGlideSWSSpec()
    bench = load_cold_test_benchmarks()
    m3d = bench["marosi_3d"]

    if reuse_existing and not quick and json_path.is_file() and npz_path.is_file():
        t6 = json.loads(json_path.read_text(encoding="utf-8"))
        t6["spec_3d"] = asdict(spec_3d)
        t6["taper_scales"] = [0.45]
        t6["f_search_band_ghz"] = [52.0, 72.0]
        json_path.write_text(json.dumps(t6, indent=2) + "\n", encoding="utf-8")
        return t6

    r3 = run_3d_marosi_beam_wake(
        spec_3d,
        n_periods=4 if quick else 6,
        nx=12 if quick else 20,
        n_feat_y=4,
        nz_per_period=14 if quick else 20,
        use_graded_mesh=True,
    )
    L_act = float(r3["active_length_m"])
    u0_3d = float(r3["u0_m_s"])
    f_u_cold = np.array(m3d["f_unfolded_ghz"], dtype=float)
    beta_u_cold = np.array(m3d["beta_unfolded_d_over_pi"], dtype=float) * np.pi / spec_3d.period_m
    k_u_cold = np.array(m3d["k_on_axis_unfolded_ohm"], dtype=float)
    delta_beta_u = beta_u_cold - (2.0 * np.pi * f_u_cold * 1e9 / u0_3d)
    sinc2_u = np.sinc(0.5 * delta_beta_u * L_act / np.pi) ** 2
    rz_u_cold = 0.25 * (beta_u_cold**2) * k_u_cold * (L_act**2) * sinc2_u

    lb_mask = (f_u_cold >= 54.0) & (f_u_cold <= 60.0)
    k_lb = float(np.max(k_u_cold[lb_mask]))
    rz_lb = float(np.max(rz_u_cold[lb_mask]))

    f_band_3d = np.linspace(46.0, 72.0, 261)
    rz_lin_sync = analytic_sws_beam_impedance(
        f_band_3d * 1e9,
        f_sync_hz=m3d["f_sync_hz"],
        beta_n_rad_m=m3d["beta_n_rad_m"],
        k_pierce_ohm=m3d["k_n_on_axis_ohm"],
        length_m=L_act,
        u0_m_s=u0_3d,
        vg_m_s=m3d["vg_m_s"],
    )

    out_t6 = _strip_heavy_arrays(r3)
    out_t6["spec_3d"] = asdict(spec_3d)
    out_t6["taper_scales"] = [0.45]
    out_t6["f_search_band_ghz"] = [52.0, 72.0]
    out_t6["cold_test_marosi_3d"] = {
        "f_sync_ghz": float(m3d["f_sync_ghz"]),
        "beta_n_d_over_pi": float(m3d["beta_n_d_over_pi"]),
        "beta_n_rad_m": float(m3d["beta_n_rad_m"]),
        "vg_over_c": float(m3d["vg_over_c"]),
        "k_n_on_axis_at_sync_ohm": float(m3d["k_n_on_axis_ohm"]),
        "k_n_beam_avg_at_sync_ohm": float(m3d["k_n_beam_avg_ohm"]),
        "re_z_linear_sync_ohm": float(np.max(rz_lin_sync)),
        "k_cold_lower_branch_55_58ghz_ohm": k_lb,
        "re_z_cold_lower_branch_peak_ohm": rz_lb,
    }

    np.savez_compressed(
        npz_path,
        f_ghz=np.asarray(r3["f_ghz"]),
        re_z_ohm=np.asarray(r3["re_z_ohm"]),
        im_z_ohm=np.asarray(r3["im_z_ohm"]),
        s_m=np.asarray(r3["s_m"]),
        wp_v_per_pc=np.asarray(r3["wp_v_per_pc"]),
        f_unfolded_cold_ghz=f_u_cold,
        k_on_axis_cold_unfolded_ohm=k_u_cold,
        sinc2_cold_unfolded=sinc2_u,
        re_z_cold_unfolded_ohm=rz_u_cold,
        f_band_3d_ghz=f_band_3d,
        re_z_3d_linear_sync_ohm=rz_lin_sync,
    )
    json_path.write_text(json.dumps(out_t6, indent=2) + "\n", encoding="utf-8")
    return out_t6


def build_summary_metrics(data_dir: Path | str) -> dict[str, Any]:
    """Build and write ``data/summary_metrics.json`` from ``task_01``–``task_06`` and Tutorial 6.2 benchmarks."""
    data_dir = Path(data_dir)
    bench = load_cold_test_benchmarks()
    l1 = bench["levels_2d"]["L1"]
    l2 = bench["levels_2d"]["L2"]
    l_ref = bench["levels_2d"]["ref"]

    t1 = json.loads((data_dir / "task_01_termination_reflection.json").read_text(encoding="utf-8"))
    t2 = json.loads((data_dir / "task_02_2d_length_sweep.json").read_text(encoding="utf-8"))
    t3 = json.loads((data_dir / "task_03_fwd_vs_bwd_poynting.json").read_text(encoding="utf-8"))
    t4 = json.loads((data_dir / "task_04_mutations_detuning_and_cavity.json").read_text(encoding="utf-8"))
    t5 = json.loads((data_dir / "task_05_mesh_convergence.json").read_text(encoding="utf-8"))
    t6 = json.loads((data_dir / "task_06_3d_marosi_beam_wake.json").read_text(encoding="utf-8"))

    n_ref_key = "20" if "20" in t2["runs"] else list(t2["runs"].keys())[-1]
    n_ref_int = int(n_ref_key)
    r_n20 = t2["runs"][n_ref_key]
    y_sw = t2["sheet_beam_y_sweep"]
    bwd3 = t3["backward_wave_glide_m1"]
    fwd3 = t3["forward_wave_mirror_n0"]
    m_mid = t4["pec_cavity_resonances"]["modes"][1]

    k_n8_nf4 = float(t5["k_on_axis_ohm_array"][0])
    k_n20_nf4 = float(r_n20["k_on_axis_beam_ohm"])

    sm = {
        "termination_taper_scales": list(t1["taper_scales"]),
        "termination_f_sync_ghz": float(t1["f_sync_ghz"]),
        "termination_gamma_tapered_at_sync": float(t1["gamma_tapered_at_sync"]),
        "termination_gamma_abrupt_at_sync": float(t1["gamma_abrupt_at_sync"]),
        "termination_gamma_tapered_passband_mean": float(
            t1.get("gamma_tapered_passband_mean", float(np.mean(t1["gamma_tapered"])))
        ),
        "termination_reduction_factor": float(t1["reduction_factor_abrupt_over_tapered"]),
        "termination_rel_error_bound_tapered": float(t1["rel_error_bound_tapered"]),
        "termination_rel_error_bound_abrupt": float(t1["rel_error_bound_abrupt"]),
        "length_scaling_peak_slope_fit": float(t2["power_law_slope_peak_vs_L"]),
        "length_scaling_peak_slope_theory": 2.0,
        "length_scaling_fwhm_slope_fit": float(t2["power_law_slope_fwhm_vs_L"]),
        "length_scaling_fwhm_slope_theory": -1.0,
        "n20_f_pk_ghz": float(r_n20["f_pk_ghz"]),
        "n20_cold_f_sync_L1_ghz": float(l1["f_sync_bwd_ghz"]),
        "n20_cold_f_sync_L2_ghz": float(l2["f_sync_bwd_ghz"]),
        "n20_re_z_pk_ohm": float(r_n20["re_z_pk_ohm"]),
        "n20_cold_re_z_pk_L1_ohm": float(l1["z_peak_per_n2_on_axis_ohm"] * (n_ref_int**2)),
        "n20_cold_re_z_pk_L2_ohm": float(l2["z_peak_per_n2_on_axis_ohm"] * (n_ref_int**2)),
        "n20_fwhm_ghz": float(r_n20["fwhm_ghz"]),
        "n20_cold_fwhm_L2_ghz": float(l2["fwhm_times_n_ghz"] / n_ref_int),
        "n20_k_on_axis_beam_ohm": k_n20_nf4,
        "n20_cold_k_on_axis_L1_ohm": float(l1["k_m1_on_axis_ohm"]),
        "n20_cold_k_on_axis_L2_ohm": float(l2["k_m1_on_axis_ohm"]),
        "n20_cold_k_on_axis_ref_ohm": float(l_ref["k_m1_on_axis_ohm"]),
        "n20_k_rel_diff_vs_cold_L1": float(
            abs(k_n20_nf4 - l1["k_m1_on_axis_ohm"]) / l1["k_m1_on_axis_ohm"]
        ),
        "length_effect_k_rel_diff_n20_vs_n8_nf4": float((k_n20_nf4 - k_n8_nf4) / k_n8_nf4),
        "n20_k_sheet_avg_beam_ohm": float(r_n20["k_sheet_avg_beam_ohm"]),
        "n20_k_sheet_avg_7node_mean_ohm": float(y_sw["k_sheet_avg_7node_mean_ohm"]),
        "n20_k_sheet_avg_trapz_ohm": float(y_sw["k_sheet_avg_trapz_ohm"]),
        "n20_sheet_ratio_7node_mean": float(y_sw["ratio_7node_mean_over_axis"]),
        "n20_sheet_ratio_trapz": float(y_sw["ratio_trapz_over_axis"]),
        "n20_sheet_ratio_continuum_cosh": float(y_sw["eta_sheet_continuum"]),
        "n20_cold_k_sheet_avg_L1_ohm": float(l1["k_m1_beam_avg_ohm"]),
        "n20_cold_k_sheet_avg_L2_ohm": float(l2["k_m1_beam_avg_ohm"]),
        "n20_cold_sheet_ratio_L1": float(l1["k_m1_ratio_beam_over_axis"]),
        "n20_cold_sheet_ratio_L2": float(l2["k_m1_ratio_beam_over_axis"]),
        "n20_dw_dz_unattenuated_j_per_m": float(r_n20["dw_dz_unattenuated_j_per_m"]),
        "n20_dw_dz_analytic_j_per_m": float(t2["dw_dz_analytic_unattenuated_j_per_m"]),
        "bwd_poynting_energy_in_pj": float(bwd3["integrated_energy_in_j"] * 1e12),
        "bwd_poynting_energy_out_pj": float(bwd3["integrated_energy_out_j"] * 1e12),
        "bwd_total_radiated_pj": float(bwd3["total_radiated_cpml_energy_j"] * 1e12),
        "bwd_w_loss_active_pj": float(bwd3["w_loss_active_j"] * 1e12),
        "bwd_w_loss_between_ports_pj": float(
            bwd3["w_loss_between_smooth_ports_j"] * 1e12
        ),
        "bwd_energy_balance_rel_error_vs_active": float(
            bwd3["energy_balance_rel_error_vs_active"]
        ),
        "bwd_energy_balance_rel_error_vs_total": float(
            bwd3["energy_balance_rel_error_vs_total"]
        ),
        "fwd_u0_over_c": float(fwd3["u0_over_c"]),
        "fwd_beam_voltage_kv": float(fwd3["beam_voltage_v"] * 1e-3),
        "fwd_vg_cold_L2_over_c": float(fwd3["vg_over_c"]),
        "fwd_vg_cold_L1_over_c": float(fwd3["vg_cold_L1_over_c"]),
        "fwd_f_sync_cold_L1_ghz": float(fwd3["f_sync_cold_L1_ghz"]),
        "fwd_f_sync_cold_L2_ghz": float(fwd3["f_sync_cold_L2_ghz"]),
        "fwd_k0_at_sync_y50um_cold_L1_ohm": float(fwd3["k0_at_sync_y50um_cold_L1_ohm"]),
        "fwd_k0_at_sync_y50um_cold_L2_ohm": float(fwd3["k0_at_sync_y50um_cold_L2_ohm"]),
        "fwd_k0_at_sync_y50um_fit_ohm": float(fwd3["k0_at_sync_y50um_fit_ohm"]),
        "fwd_re_z_at_sync_L1_cold_ohm": float(fwd3["re_z_at_sync_y50um_cold_L1_ohm"]),
        "fwd_re_z_at_sync_L1_fit_ohm": float(fwd3["re_z_at_sync_L1_fit_ohm"]),
        "fwd_f_pk_ghz": float(fwd3["f_pk_ghz"]),
        "fwd_re_z_pk_fit_ohm": float(fwd3["re_z_pk_ohm"]),
        "fwd_k0_at_fpk_y50um_fit_ohm": float(fwd3["k0_at_fpk_y50um_fit_ohm"]),
        "fwd_f_pk_nonlinear_cold_L1_ghz": float(fwd3["f_pk_nonlinear_cold_L1_ghz"]),
        "fwd_re_z_pk_nonlinear_cold_L1_ohm": float(fwd3["re_z_pk_nonlinear_cold_L1_ohm"]),
        "fwd_k0_at_fpk_nonlinear_cold_L1_ohm": float(
            fwd3["k0_at_fpk_nonlinear_cold_L1_ohm"]
        ),
        "fwd_f_upper_stopband_ghz": float(fwd3["f_upper_stopband_cold_L1_ghz"]),
        "fwd_poynting_energy_in_pj": float(fwd3["integrated_energy_in_j"] * 1e12),
        "fwd_poynting_energy_out_pj": float(fwd3["integrated_energy_out_j"] * 1e12),
        "fwd_total_radiated_pj": float(fwd3["total_radiated_cpml_energy_j"] * 1e12),
        "fwd_w_loss_active_pj": float(fwd3["w_loss_active_j"] * 1e12),
        "fwd_w_loss_between_ports_pj": float(
            fwd3["w_loss_between_smooth_ports_j"] * 1e12
        ),
        "fwd_energy_balance_rel_error_vs_active": float(
            fwd3["energy_balance_rel_error_vs_active"]
        ),
        "fwd_energy_balance_rel_error_vs_total": float(
            fwd3["energy_balance_rel_error_vs_total"]
        ),
        "bwd_fwhm_fit_ghz": float(bwd3["fwhm_ghz"]),
        "bwd_fwhm_analytic_ghz": float(bwd3["fwhm_analytic_ghz"]),
        "fwd_fwhm_fit_ghz": float(fwd3["fwhm_ghz"]),
        "fwd_fwhm_linear_L2_ghz": float(fwd3["fwhm_linear_analytic_L2_ghz"]),
        "fwd_fwhm_linear_L1_ghz": float(fwd3["fwhm_linear_analytic_L1_ghz"]),
        "fwd_fwhm_nonlinear_cold_L1_ghz": float(fwd3["fwhm_nonlinear_cold_L1_ghz"]),
        "detuning_minus_5pct_df_fit_ghz": float(
            t4["detuning_sweep"]["u0_minus_5pct"]["delta_f_fit_ghz"]
        ),
        "detuning_minus_5pct_df_theory_ghz": float(
            t4["detuning_sweep"]["u0_minus_5pct"]["delta_f_linear_analytic_ghz"]
        ),
        "detuning_plus_5pct_df_fit_ghz": float(
            t4["detuning_sweep"]["u0_plus_5pct"]["delta_f_fit_ghz"]
        ),
        "detuning_plus_5pct_df_theory_ghz": float(
            t4["detuning_sweep"]["u0_plus_5pct"]["delta_f_linear_analytic_ghz"]
        ),
        "cavity_m6_f_fit_ghz": float(m_mid["f_fit_cavity_ghz"]),
        "cavity_m6_f_cold_ghz": float(m_mid["f_cold_test_ghz"]),
        "cavity_m6_rq_circuit_fit_ohm": float(
            t4["pec_cavity_resonances"].get(
                "synchronous_mode_m6_r_over_q_circuit_fit_ohm",
                m_mid["r_over_q_circuit_fit_ohm"],
            )
        ),
        "cavity_m6_rq_circuit_analytic_ohm": float(
            t4["pec_cavity_resonances"].get(
                "synchronous_mode_m6_r_over_q_circuit_analytic_ohm",
                m_mid.get("r_over_q_circuit_analytic_ohm", 0.0),
            )
        ),
        "cavity_reconstructed_tw_z_pk_ohm": float(
            t4["pec_cavity_resonances"]["reconstructed_traveling_wave_z_pk_ohm"]
        ),
        "conv_k_axis_nf4_ohm": k_n8_nf4,
        "conv_k_axis_nf6_ohm": float(t5["k_on_axis_ohm_array"][1]),
        "conv_k_axis_nf8_ohm": float(t5["k_on_axis_ohm_array"][2]),
        "conv_k_axis_band_min_ohm": float(t5["k_on_axis_finest_band_min_ohm"]),
        "conv_k_axis_band_max_ohm": float(t5["k_on_axis_finest_band_max_ohm"]),
        "conv_k_axis_band_mid_ohm": float(t5["k_on_axis_finest_band_mid_ohm"]),
        "conv_k_axis_band_spread_ohm": float(t5["k_on_axis_finest_band_spread_ohm"]),
        "conv_k_axis_band_rel_spread": float(t5["k_on_axis_finest_band_rel_spread"]),
        "conv_k_axis_nf6_vs_nf4_rel_diff": float(t5["rel_diff_nf6_vs_nf4_same_N"]),
        "conv_k_axis_nf8_vs_nf4_rel_diff": float(t5["rel_diff_nf8_vs_nf4_same_N"]),
        "conv_k_axis_band_mid_vs_nf4_rel_diff": float(
            t5["rel_diff_finest_band_mid_vs_nf4_same_N"]
        ),
        "conv_k_axis_band_vs_cold_L2_min_rel_diff": float(
            t5["rel_diff_finest_band_vs_cold_L2_min"]
        ),
        "conv_k_axis_band_vs_cold_L2_max_rel_diff": float(
            t5["rel_diff_finest_band_vs_cold_L2_max"]
        ),
        "conv_k_axis_band_vs_cold_ref_min_rel_diff": float(
            t5["rel_diff_finest_band_vs_cold_ref_min"]
        ),
        "conv_k_axis_band_vs_cold_ref_max_rel_diff": float(
            t5["rel_diff_finest_band_vs_cold_ref_max"]
        ),
        "conv_k_axis_graded_nf4_ohm": float(
            t5["runs"]["graded_conformal_nf4"]["k_on_axis_beam_ohm"]
        ),
        "conv_k_axis_staircase_nf4_ohm": float(
            t5["runs"]["uniform_staircase_nf4"]["k_on_axis_beam_ohm"]
        ),
        "marosi_3d_f_pk_ghz": float(t6["f_pk_ghz"]),
        "marosi_3d_re_z_pk_ohm": float(t6["re_z_pk_ohm"]),
        "marosi_3d_re_z_cold_lower_branch_ohm": float(
            t6["cold_test_marosi_3d"].get(
                "re_z_cold_lower_branch_54_57ghz_ohm",
                t6["cold_test_marosi_3d"].get("re_z_cold_lower_branch_peak_ohm", 0.0),
            )
        ),
        "marosi_3d_k_at_pk_ohm": float(t6["k_on_axis_at_pk_ohm"]),
        "marosi_3d_k_cold_lower_branch_ohm": float(
            t6["cold_test_marosi_3d"]["k_cold_lower_branch_55_58ghz_ohm"]
        ),
    }
    out_path = data_dir / "summary_metrics.json"
    out_path.write_text(json.dumps(sm, indent=2) + "\n", encoding="utf-8")
    return sm


def generate_all_tutorial_data(
    data_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    quick: bool = False,
    reuse_unchanged_runs: bool = False,
    tasks: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
) -> dict[str, Any]:
    """Driver entry point to regenerate task_01..06, summary_metrics.json, and tutorial figures."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if 1 in tasks:
        generate_task_01_data(data_dir, quick=quick)
    if 2 in tasks:
        generate_task_02_data(
            data_dir, quick=quick, reuse_on_axis_length_sweep=reuse_unchanged_runs
        )
    if 3 in tasks:
        generate_task_03_data(data_dir, quick=quick)
    if 4 in tasks:
        generate_task_04_data(
            data_dir, quick=quick, reuse_existing=reuse_unchanged_runs
        )
    if 5 in tasks:
        generate_task_05_data(
            data_dir, quick=quick, reuse_existing=reuse_unchanged_runs
        )
    if 6 in tasks:
        generate_task_06_data(
            data_dir, quick=quick, reuse_existing=reuse_unchanged_runs
        )

    sm = build_summary_metrics(data_dir)
    if figures_dir is not None:
        plot_sws_beam_setup_schematics(figures_dir)
        generate_all_tutorial_figures(data_dir, figures_dir)
    return sm
