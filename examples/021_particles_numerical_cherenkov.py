# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Example 021: Numerical Cherenkov instability and binomial current filtering in ``wakis.particles``.

Demonstrates the numerical Cherenkov resonance on a 3D FIT/Yee grid in a
periodic-``z`` rectangular PEC pipe and its suppression by the two-pass
compensated 1-2-1 binomial current filter (`filter="binomial"`):

1. **Discrete FIT/Yee dispersion and beam resonance**:
   On a uniform cubic grid of spacing ``Delta`` and time step
   ``dt = 0.9 * Delta / (sqrt(3) * c)`` (90% of the 3D Courant limit, so
   ``sigma = c * dt / Delta = 0.9 / sqrt(3)``), the on-axis discrete phase
   velocity reaches its minimum at the longitudinal Nyquist wavenumber
   ``k_z = pi / Delta``:
       ``v_min / c = (2 / (pi * sigma)) * arcsin(sigma)``.
   A relativistic cold beam with ``beta0 = 0.86 > v_min / c`` intersects the
   discrete electromagnetic branch ``omega_d(k_z) = k_z * beta0 * c`` at a
   fundamental (``m = 0``) resonant longitudinal wavenumber ``k_res``, whereas
   a control beam with ``beta0 = 0.50 < v_min / c`` has no fundamental ``m = 0``
   intersection in the Brillouin zone (though its ``m = -1`` spatial alias
   ``|omega_{-1}(k_z)| = (2 * pi / Delta - k_z) * v0`` intersects ``omega_d``
   near ``k_z * Delta ~ 2.37 rad``, which is invisible in the ``m = 0``
   co-moving frame and isolated when demodulated at ``omega_{-1}``).

2. **Radiated longitudinal spectrum and binomial current filtering**:
   Shot noise from random longitudinal particle placement drives secular
   growth of the radiated transverse field ``E_x`` at ``k_res`` for the
   ``beta0 = 0.86`` beam. Enabling `filter="binomial"` attenuates the resonant
   power by the squared transfer function ``|T_comp(k_res * Delta)|^2`` of the
   two-pass compensated 1-2-1 filter.

Usage
-----
Full run::

    python examples/021_particles_numerical_cherenkov.py

Fast smoke run (< 60 s on CPU)::

    python examples/021_particles_numerical_cherenkov.py --smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from wakis.figstyle import OKABE_ITO, apply_style, save, spectrum_grid

_COMMON_PATH = Path(__file__).resolve().parent / "_selffield_cherenkov_common.py"
_COMMON_SPEC = importlib.util.spec_from_file_location(
    "_selffield_cherenkov_common", _COMMON_PATH
)
_COMMON_MOD = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON_MOD)

binomial_transfer_1d = _COMMON_MOD.binomial_transfer_1d
compute_dispersion_and_resonance = _COMMON_MOD.compute_dispersion_and_resonance
run_cherenkov_case = _COMMON_MOD.run_cherenkov_case

BLUE = OKABE_ITO[0]
RED = OKABE_ITO[1]
GREEN = OKABE_ITO[2]
ORANGE = OKABE_ITO[3]
PURPLE = OKABE_ITO[4]
GREY = "#555555"


def make_figures(disp: dict, res_unfilt: dict, res_filt: dict, res_ctrl: dict, figdir: Path):
    """Generate the two figures for Example 021."""
    apply_style()

    # ------------------------------------------------------------------
    # Figure 1: Discrete dispersion omega_d(k_z) and beam lines (m = 0, +-1)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8.4, 7.4))  # wide enough for the two-column legend below
    u_norm = disp["u_grid"] / np.pi
    norm_w = disp["dz"] / (np.pi * c)

    ax1.plot(
        u_norm,
        disp["omega_vac"] * norm_w,
        linestyle=":",
        color=GREY,
        linewidth=1.5,
        label=r"Continuum light line $\omega = c k_z$",
    )
    ax1.plot(
        u_norm,
        disp["omega_d_1d"] * norm_w,
        color=BLUE,
        linewidth=2.2,
        label=r"Discrete FIT/Yee $\omega_d(0, 0, k_z)$",
    )
    ax1.plot(
        u_norm,
        disp["omega_d_wg"] * norm_w,
        color=PURPLE,
        linestyle="--",
        linewidth=1.6,
        label=r"Waveguide branch $\omega_d(k_{\perp,11}, k_z)$",
    )
    ax1.plot(
        u_norm,
        disp["omega_beam_fast"] * norm_w,
        color=RED,
        linewidth=2.0,
        label=rf"Beam $m=0$: $\omega = k_z v_0$ ($\beta_0 = {disp['beta_fast']:.2f}$)",
    )
    ax1.plot(
        u_norm,
        disp["omega_beam_fast_m_minus1"] * norm_w,
        color=RED,
        linestyle=":",
        linewidth=1.4,
        alpha=0.85,
        label=rf"Beam $m=-1$: $|\omega_{{-1}}| = (2\pi/\Delta - k_z)v_0$ ($\beta_0 = {disp['beta_fast']:.2f}$, no $(1,1)$ root)",
    )
    ax1.plot(
        u_norm,
        disp["omega_beam_ctrl"] * norm_w,
        color=GREEN,
        linestyle="-.",
        linewidth=2.0,
        label=rf"Control $m=0$: $\omega = k_z v_0$ ($\beta_0 = {disp['beta_ctrl']:.2f}$, no root)",
    )
    ax1.plot(
        u_norm,
        disp["omega_beam_ctrl_m_minus1"] * norm_w,
        color=GREEN,
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        label=rf"Control $m=-1$: $|\omega_{{-1}}| = (2\pi/\Delta - k_z)v_0$ ($\beta_0 = {disp['beta_ctrl']:.2f}$)",
    )
    ax1.plot(
        u_norm,
        disp["omega_beam_ctrl_m_plus1"] * norm_w,
        color=GREEN,
        linestyle=":",
        linewidth=1.2,
        alpha=0.75,
        label=rf"Control $m=+1$: $\omega_{{+1}} = (2\pi/\Delta + k_z)v_0$ ($\beta_0 = {disp['beta_ctrl']:.2f}$, no $(1,1)$ root)",
    )

    u_res_norm = disp["u_res_3d_rad"] / np.pi
    w_res_norm = disp["omega_res_3d_rad_s"] * norm_w
    ax1.scatter(
        [u_res_norm],
        [w_res_norm],
        color=ORANGE,
        edgecolor="black",
        s=70,
        zorder=5,
        label=rf"$\beta_0={disp['beta_fast']:.2f}$, $m=0$ resonance $k_{{\mathrm{{res}}}}\Delta/\pi = {u_res_norm:.3f}$",
    )
    ax1.axvline(u_res_norm, color=ORANGE, linestyle=":", linewidth=1.2, alpha=0.8)

    u_ctrl_m1_norm = disp["u_res_ctrl_m1_3d_rad"] / np.pi
    w_ctrl_m1_norm = disp["omega_res_ctrl_m1_3d_rad_s"] * norm_w
    ax1.scatter(
        [u_ctrl_m1_norm],
        [w_ctrl_m1_norm],
        color=GREEN,
        marker="D",
        edgecolor="black",
        s=60,
        zorder=5,
        label=rf"$\beta_0={disp['beta_ctrl']:.2f}$, $m=-1$ alias root $k_{{\mathrm{{res}},-1}}\Delta/\pi = {u_ctrl_m1_norm:.3f}$",
    )
    ax1.axvline(u_ctrl_m1_norm, color=GREEN, linestyle=":", linewidth=1.2, alpha=0.7)

    ax1.set_xlabel(r"Normalized longitudinal wavenumber $k_z \Delta z / \pi$")
    ax1.set_ylabel(r"Normalized frequency $\omega \Delta z / (\pi c)$")
    ax1.set_title(r"3D FIT/Yee Discrete Dispersion and Cherenkov Resonances ($m=0, \pm 1$)")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.25)
    # Nine entries: below the axes so they never cover the dispersion curves.
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=7.2)
    fig1.tight_layout()
    fig1_path = figdir / "particles_numerical_cherenkov_dispersion.png"
    save(fig1, fig1_path)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: Measured k_z spectrum with and without binomial filter
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.4))
    u_bins = res_unfilt["u_arr"][1:] / np.pi
    peak_unfilt = float(np.max(res_unfilt["spec_rad"][1:]))
    spec_u_norm = res_unfilt["spec_rad"][1:] / peak_unfilt
    spec_f_norm = res_filt["spec_rad"][1:] / peak_unfilt
    spec_c_norm = res_ctrl["spec_rad"][1:] / peak_unfilt

    ax2.semilogy(
        u_bins,
        spec_u_norm,
        marker="o",
        color=RED,
        linewidth=1.8,
        markersize=4.5,
        label=rf"Unfiltered ($\beta_0 = {disp['beta_fast']:.2f}$)",
    )
    ax2.semilogy(
        u_bins,
        spec_f_norm,
        marker="s",
        color=BLUE,
        linewidth=1.8,
        markersize=4.5,
        label=rf"Binomial filter ($\beta_0 = {disp['beta_fast']:.2f}$, filter='binomial')",
    )
    ax2.semilogy(
        u_bins,
        spec_c_norm,
        marker="^",
        color=GREEN,
        linestyle="--",
        linewidth=1.5,
        markersize=4.0,
        label=rf"Sub-$v_{{\min}}$ control ($\beta_0 = {disp['beta_ctrl']:.2f}$, unfiltered)",
    )

    ax2.axvline(
        u_res_norm,
        color=ORANGE,
        linestyle=":",
        linewidth=1.8,
        label=rf"Predicted $k_{{\mathrm{{res}}}}\Delta z / \pi = {u_res_norm:.3f}$",
    )

    spectrum_grid(ax2)
    ax2.set_xlabel(r"Normalized longitudinal wavenumber $k_z \Delta z / \pi$")
    ax2.set_ylabel(r"Normalized radiated power $|\Delta \tilde{E}_x(k_z)|^2 / \max$")
    ax2.set_title(r"Radiated Non-Co-Moving $E_x(k_z)$ Spectrum and Binomial Filtering")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(1.0e-5, 2.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5)
    fig2.tight_layout()
    fig2_path = figdir / "particles_numerical_cherenkov_spectrum.png"
    save(fig2, fig2_path)
    plt.close(fig2)

    return fig1_path, fig2_path


def main():
    parser = argparse.ArgumentParser(
        description="Example 021: Numerical Cherenkov resonance and binomial filter."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast smoke configuration (< 60 s on CPU).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for the JSON summary.",
    )
    parser.add_argument(
        "--figdir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "img",
        help="Output directory for the two figures.",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    n_xy = 16
    n_z = 64 if args.smoke else 96
    n_steps = 480 if args.smoke else 720
    n_slices = 64 if args.smoke else 96

    disp = compute_dispersion_and_resonance(
        beta_fast=0.86, beta_ctrl=0.50, cfln=0.9, dz=0.002, n_xy=n_xy
    )
    res_unfilt = run_cherenkov_case(
        beta0=0.86,
        filter_mode=None,
        n_xy=n_xy,
        n_z=n_z,
        n_steps=n_steps,
        n_slices=n_slices,
    )
    res_filt = run_cherenkov_case(
        beta0=0.86,
        filter_mode="binomial",
        n_xy=n_xy,
        n_z=n_z,
        n_steps=n_steps,
        n_slices=n_slices,
    )
    res_ctrl = run_cherenkov_case(
        beta0=0.50,
        filter_mode=None,
        n_xy=n_xy,
        n_z=n_z,
        n_steps=n_steps,
        n_slices=n_slices,
    )

    fig1_path, fig2_path = make_figures(disp, res_unfilt, res_filt, res_ctrl, args.figdir)

    dk_bin = 2.0 * np.pi / (n_z * res_unfilt["dx"])
    pred_bin_1d = disp["k_res_1d_m_inv"] / dk_bin
    pred_bin_3d = disp["k_res_3d_m_inv"] / dk_bin
    pred_bin_ctrl_m1_3d = (disp["u_res_ctrl_m1_3d_rad"] / res_unfilt["dx"]) / dk_bin
    peak_bin_unfilt = int(np.argmax(res_unfilt["spec_rad"][1:]) + 1)
    peak_kz_measured = float(res_unfilt["kz_arr"][peak_bin_unfilt])
    peak_u_measured = float(peak_kz_measured * disp["dz"])
    pred_filter_power_at_bin = float(
        (binomial_transfer_1d(peak_u_measured, compensated=True) ** 2)
        * (disp["t_perp_1d"] ** 4)
    )
    peak_bin_ctrl_m1 = int(np.argmax(res_ctrl["spec_rad_m_minus_1"][1:]) + 1)
    ctrl_m1_peak_over_med = float(
        res_ctrl["spec_rad_m_minus_1"][peak_bin_ctrl_m1]
        / np.median(res_ctrl["spec_rad_m_minus_1"][1:])
    )
    filter_ratio_at_peak = float(
        res_filt["spec_rad"][peak_bin_unfilt] / res_unfilt["spec_rad"][peak_bin_unfilt]
    )
    energy_ratio_fast_vs_ctrl = float(
        np.sum(res_unfilt["spec_rad"][1:]) / np.sum(res_ctrl["spec_rad"][1:])
    )
    elapsed_s = time.perf_counter() - t0

    summary = {
        "example": "021_particles_numerical_cherenkov",
        "smoke": bool(args.smoke),
        "elapsed_s": float(elapsed_s),
        "parameters": {
            "n_xy": n_xy,
            "n_z": n_z,
            "n_steps": n_steps,
            "n_slices": n_slices,
            "dx_m": disp["dz"],
            "dt_s": disp["dt"],
            "cfln": 0.9,
            "sigma_1d": disp["sigma"],
            "beta_fast": disp["beta_fast"],
            "beta_ctrl": disp["beta_ctrl"],
        },
        "predictions": {
            "v_min_over_c": disp["v_min_over_c"],
            "u_res_1d_rad": disp["u_res_1d_rad"],
            "k_res_1d_m_inv": disp["k_res_1d_m_inv"],
            "u_res_3d_rad": disp["u_res_3d_rad"],
            "k_res_3d_m_inv": disp["k_res_3d_m_inv"],
            "predicted_bin_1d": float(pred_bin_1d),
            "predicted_bin_3d": float(pred_bin_3d),
            "u_res_ctrl_m1_1d_rad": disp["u_res_ctrl_m1_1d_rad"],
            "u_res_ctrl_m1_3d_rad": disp["u_res_ctrl_m1_3d_rad"],
            "predicted_bin_ctrl_m1_3d": float(pred_bin_ctrl_m1_3d),
            "cic_alias_weight_ctrl_m1": disp["cic_alias_weight_ctrl_m1"],
            "binomial_t_uncomp_at_res": disp["t_uncomp_at_res"],
            "binomial_t_uncomp_power_at_res": disp["t_uncomp_power_at_res"],
            "binomial_t_comp_at_res": disp["t_comp_at_res"],
            "binomial_t_comp_power_at_res": disp["t_comp_power_at_res"],
            "binomial_3d_power_at_measured_bin": pred_filter_power_at_bin,
        },
        "measurements": {
            "peak_bin_unfiltered": peak_bin_unfilt,
            "peak_kz_m_inv": peak_kz_measured,
            "peak_u_rad": peak_u_measured,
            "radiated_energy_ratio_fast_over_ctrl": energy_ratio_fast_vs_ctrl,
            "filter_power_ratio_at_peak": filter_ratio_at_peak,
            "ctrl_m1_alias_peak_bin": peak_bin_ctrl_m1,
            "ctrl_m1_alias_peak_over_median": ctrl_m1_peak_over_med,
            "kz_bins_m_inv": [float(v) for v in res_unfilt["kz_arr"]],
            "spectrum_unfiltered": [float(v) for v in res_unfilt["spec_rad"]],
            "spectrum_filtered_binomial": [float(v) for v in res_filt["spec_rad"]],
            "spectrum_control_sub_vmin": [float(v) for v in res_ctrl["spec_rad"]],
            "spectrum_control_m_minus_1_demod": [
                float(v) for v in res_ctrl["spec_rad_m_minus_1"]
            ],
        },
        "figures": [str(fig1_path.name), str(fig2_path.name)],
    }

    json_path = args.outdir / "021_particles_numerical_cherenkov.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[Example 021] smoke={args.smoke}, elapsed={elapsed_s:.2f}s, "
        f"v_min/c={disp['v_min_over_c']:.4f}, "
        f"pred_bin_3d={pred_bin_3d:.2f} (1D: {pred_bin_1d:.2f}), "
        f"measured_peak_bin={peak_bin_unfilt}, "
        f"E_rad(0.86)/E_rad(0.50)={energy_ratio_fast_vs_ctrl:.2f}x, "
        f"filter_power_ratio={filter_ratio_at_peak:.6f} "
        f"(pred |T_comp(k_bin)|^2*T_perp^4={pred_filter_power_at_bin:.6f}), "
        f"ctrl_m=-1_bin={peak_bin_ctrl_m1} (pred={pred_bin_ctrl_m1_3d:.2f}, "
        f"peak/med={ctrl_m1_peak_over_med:.1f}x)"
    )


if __name__ == "__main__":
    main()
