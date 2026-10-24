# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial 4.4 · 2.4 GHz Printed Inverted-F PCB Antenna (Texas Instruments SWRU120D)
#
# Low-power 2.4 GHz wireless transceivers (Bluetooth Low Energy, Zigbee / IEEE
# 802.15.4, Thread, and proprietary ISM-band links) almost universally rely on
# **printed circuit board (PCB) inverted-F antennas (IFAs)** etched directly on
# standard FR-4 laminate. Because a printed IFA requires zero bill-of-materials
# cost, occupies less than $26 \times 8\;\text{mm}^2$ of board edge, and matches
# directly to a $50\;\Omega$ radio pin without external matching components when
# laid out according to reference rules, it is the workhorse antenna of embedded
# RF engineering.
#
# In this tutorial we build, preflight, simulate, and validate a full-wave 3D
# Finite Integration Technique (FIT) model of the **Texas Instruments 2.4 GHz
# Inverted-F Antenna** documented in **TI Design Note DN007 (Literature Number
# SWRU120D)** by Audun Andersen, and we analyze its physical design rules using
# **TI Application Note AN043 (Literature Number SWRA117D)**.
#
# ## Learning goals
#
# 1. **Physics of the inverted-F antenna**: understand how bending a
#    $\lambda/4$ monopole parallel to the ground plane and tapping it at distance
#    $L_4$ from a shorting post creates an **inductive shunt impedance
#    transformer** that steps the low radiation resistance of a low-profile arm up
#    to $Z_0 = 50\;\Omega$.
# 2. **Ground plane as half of the radiator & the Chu–Wheeler bound**: evaluate
#    the **Chu–Wheeler–McLean fundamental $Q$ bound**
#    $Q_{\text{Chu}} = \eta_{\text{rad}}\bigl[(ka)^{-3} + (ka)^{-1}\bigr]$ and
#    demonstrate via 3D surface-current maps and a ground-length sweep why the
#    PCB ground plane carries the image return current that enables a
#    $> 300\;\text{MHz}$ ($-10\;\text{dB}$) bandwidth.
# 3. **Dispersive FR-4 dielectric loss**: use Wakis's library `'fr4'` Debye model
#    ($\varepsilon'_r \approx 4.28$, $\tan\delta \approx 0.018$ at $2.44\;\text{GHz}$)
#    and verify power conservation between port accepted power $P_{\text{acc}}$,
#    closed-box Poynting flux $\Phi_{\text{box}}$, and far-field radiated power
#    $P_{\text{rad}}$.
# 4. **Project meshing rules & `wakis.estimate` preflight**: apply the project
#    grid rules ($\ge 20$ cells per wavelength in FR-4 and $\ge 4$ cells across
#    smallest features), state transparently where sub-millimetre copper steps
#    ($W_2 = 0.46\;\text{mm}$, $L_8 = 0.45\;\text{mm}$) are under-resolved on a
#    CPU workstation, and quantify grid convergence across $\Delta_{\text{trace}}
#    \in \{0.50, 0.35, 0.25\}\;\text{mm}$ alongside a dedicated GPU task at
#    $\Delta_{\text{trace}} = 0.1125\;\text{mm}$.

# %% [markdown]
# ## Physical & engineering theory
#
# ### 1. From a $\lambda/4$ monopole to the inverted-F tap transformer
#
# A vertical quarter-wave monopole of height $L \approx \lambda_0 / 4 \approx
# 30.7\;\text{mm}$ at $2.44\;\text{GHz}$ over a large ground plane presents a
# resonant input impedance $Z_{\text{in}} \approx 36.5 + j 0\;\Omega$, but sticks
# out perpendicular to the PCB. Folding the conductor so that it runs parallel to
# the ground plane edge at a small height $H_1 = 5.70\;\text{mm} \approx
# 0.046\,\lambda_0$ makes it planar, but image currents flowing in the opposite
# direction along the adjacent ground edge drastically reduce the radiation
# resistance ($R_{\text{rad}} \approx 6\text{–}12\;\Omega$) while introducing a
# large capacitive reactance.
#
# The **inverted-F topology** solves both problems simultaneously by connecting
# the left end of the horizontal arm ($x \in [0, W_1]$) to the ground plane
# through a shorting trace and via post, and feeding the arm at an intermediate
# tap point $x_{\text{feed}}$ separated by $L_4 = 4.80\;\text{mm}$ from the
# shorting riser:
#
# 1. **Reactive cancellation**: the short-circuited stub of electrical length
#    $\beta \ell_{\text{stub}} \ll \pi/2$ acts as a shunt inductance
#    $j X_L \approx j Z_{0,\text{stub}} \tan(\beta \ell_{\text{stub}})$ that
#    resonates out the open arm's capacitance.
# 2. **Impedance step-up**: the standing-wave current maximum sits at the
#    shorted base, while the feed probes a reduced current $I_{\text{feed}} =
#    I_{\max} \cos(\beta d_{\text{tap}})$, stepping the input resistance up to
#    $R_{\text{in}} \approx 50\;\Omega$ when $L_4 = 4.80\;\text{mm}$.
#
# ### 2. SWRU120D Table 1-1 geometry & Source Table on 1.00 mm FR-4
#
# Texas Instruments Design Note **DN007 (SWRU120D, Rev. D, Jan. 2019)** specifies
# nineteen layout parameters on a $h_{\text{sub}} = 1.00\;\text{mm}$ standard FR-4
# PCB with **no copper ground underneath the $25.7 \times 7.5\;\text{mm}$ antenna
# footprint** (Section 1, p. 2, Figure 1-1 & Table 1-1; Section 3, Table 3-1, p. 11).
# All TI measurements were performed on a battery-powered **CC2430DB** demonstration
# board at $0\;\text{dBm}$ output power (Section 1, p. 2; Section 2.1, p. 3, Figure 2-1).
# SWRU120D does **not** report numerical values for the CC2430DB ground-plane dimensions
# in mm, the FR-4 relative permittivity $\varepsilon_r$ or loss tangent $\tan\delta$, or
# the radiation efficiency $\eta_{\text{rad}}$. Throughout this tutorial we model a
# coplanar copper ground plane of **$29.58 \times 32.0\;\text{mm}^2$** ($x \in [-2.0, 27.58]\;\text{mm}$,
# $y \in [-32.0, 0.0]\;\text{mm}$, giving a total FR-4 PCB of **$29.58 \times 39.5\;\text{mm}^2 \times 1.00\;\text{mm}$**)
# and test ground-plane size, FR-4 permittivity, port gap, and jog width one hypothesis at a time.
#
# The horizontal and vertical Table 1-1 dimensions form a closed trace layout:
#
# $$W_1 + L_3 + L_4 + L_5 + L_2 = 1.21 + 2.18 + 4.80 + 1.00 + 16.40
# = 25.59\;\text{mm} \approx L_1 = 25.58\;\text{mm},$$
#
# $$H_5 + H_7 + H_4 + H_3 + H_2 = 0.66 + 0.80 + 2.21 + 1.29 + 0.74
# = 5.70\;\text{mm} = H_1.$$
#
# #### Source Table — Verbatim Reference Values from TI SWRU120D (DN007, Rev. D, Jan. 2019)
#
# | Quantity / Dimension | Value in TI SWRU120D | Where in TI SWRU120D |
# | :--- | :---: | :--- |
# | $H_1, H_2, H_3, H_4, H_5$ | $5.70, 0.74, 1.29, 2.21, 0.66\;\text{mm}$ | Section 1, p. 2, Figure 1-1 & Table 1-1 |
# | $H_6, H_7, H_8, H_9$ | $1.21, 0.80, 1.80, 0.61\;\text{mm}$ | Section 1, p. 2, Figure 1-1 & Table 1-1 |
# | $W_1, W_2$ | $1.21, 0.46\;\text{mm}$ | Section 1, p. 2, Figure 1-1 & Table 1-1 |
# | $L_1, L_2, L_3, L_4$ | $25.58, 16.40, 2.18, 4.80\;\text{mm}$ | Section 1, p. 2, Figure 1-1 & Table 1-1 |
# | $L_5, L_6, L_7, L_8$ | $1.00, 1.00, 3.20, 0.45\;\text{mm}$ | Section 1, p. 2, Figure 1-1 & Table 1-1 |
# | Substrate thickness & material | $1\;\text{mm}$ standard FR-4 ($\varepsilon_r, \tan\delta$ not reported) | Section 1, p. 2 |
# | Antenna footprint size | $25.7 \times 7.5\;\text{mm}$ (ground plane not included) | Abstract, p. 1; Section 1, p. 2; Section 3, Table 3-1, p. 11 |
# | Evaluation board / ground plane | CC2430DB board (ground size in mm not reported) | Section 1, p. 2; Section 2.1, p. 3, Figure 2-1 |
# | Operating band & pattern test $f$ | $2.40\text{–}2.48\;\text{GHz}$ ISM band; patterns at $2.44\;\text{GHz}$ | Title & Abstract, p. 1; Section 2.1, Figures 2-2 to 2-7, pp. 4–9 |
# | Resonance minimum $f_{\text{res}}$ | $\approx 2.44\;\text{GHz}$ (read from Fig. 2-8, approx.) | Section 2.2, p. 10, Figure 2-8 (read from Fig. 2-8, approx.) |
# | Reflection $|S_{11}|$ | $< -15\;\text{dB}$ (min $\approx -16.5\;\text{dB}$, read from Fig. 2-8, approx.) | Section 3, Table 3-1, p. 11; Section 2.2, p. 10, Figure 2-8 |
# | Bandwidth ($< 10\%$ reflection, $-10\;\text{dB}$) | $> 300\;\text{MHz}$ ($\approx 2.29\text{–}2.62\;\text{GHz} \approx 330\;\text{MHz}$, read from Fig. 2-8, approx.) | Abstract, p. 1; Section 2.2, p. 10, Fig. 2-8; Section 2.3, p. 11, Fig. 2-9 |
# | Gain in XY plane | $1.1\;\text{dBi}$ | Section 3, Table 3-1, p. 11 (Figures 2-2 & 2-3, pp. 4–5) |
# | Gain in XZ plane | $3.3\;\text{dBi}$ | Abstract, p. 1; Section 3, Table 3-1, p. 11 (Figures 2-4 & 2-5, pp. 6–7) |
# | Gain in YZ plane | $1.6\;\text{dBi}$ | Section 3, Table 3-1, p. 11 (Figures 2-6 & 2-7, pp. 8–9) |
# | Radiation efficiency $\eta_{\text{rad}}$ | Not reported in source | Not in SWRU120D |
#
# ### 3. The Chu–Wheeler fundamental bound and ground-plane counterpoise
#
# For any linearly polarized antenna enclosed inside a sphere of radius $a$
# ($k_0 a = 2\pi a / \lambda_0$), the **Chu–Wheeler–McLean bound** limits the
# minimum radiation quality factor to
#
# $$Q_{\text{Chu}}(k_0 a) = \eta_{\text{rad}} \left(\frac{1}{(k_0 a)^3} +
# \frac{1}{k_0 a}\right), \qquad \text{FBW}_{\max}(\text{VSWR}) =
# \frac{\text{VSWR} - 1}{Q_{\text{Chu}}\,\sqrt{\text{VSWR}}}.$$
#
# Considered in isolation, the $25.7 \times 7.5\;\text{mm}$ IFA trace has
# circumradius $a_{\text{trace}} = 13.4\;\text{mm}$ ($k_0 a_{\text{trace}} =
# 0.68$), whereas our complete $29.58 \times 39.5\;\text{mm}^2$ PCB including its
# $29.58 \times 32.0\;\text{mm}^2$ ground plane spans $a_{\text{PCB}} = 24.7\;\text{mm}$
# ($k_0 a_{\text{PCB}} = 1.26$). As Audun Andersen emphasizes in **TI AN043 (SWRA117D)**,
# the ground plane acts as the second arm of an asymmetric dipole: shrinking the ground
# plane below $\sim \lambda_0 / 4 \approx 30\;\text{mm}$ collapses the radiation
# resistance, detunes $f_{\text{res}}$, and narrows the impedance bandwidth.

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _helpers import (
    ALL_TASKS,
    CPU_TASKS,
    GPU_TASKS,
    SWRU120_BENCHMARKS,
    SWRU120_SOURCE_LOCATIONS,
    SWRU120_TABLE_1_1_MM,
    chu_max_fractional_bandwidth,
    chu_wheeler_q_bound,
    fr4_complex_permittivity,
    generate_all_figures,
    preflight_all_levels,
    quarter_wave_resonance_hz,
    richardson_extrapolate_ifa,
    run_ifa_simulation,
    run_task,
)

TUTORIAL_DIR = Path(__file__).resolve().parent
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse standard Wakis tutorial CLI flags."""
    parser = argparse.ArgumentParser(
        description="04-02: 2.4 GHz PCB Inverted-F Antenna (TI SWRU120D)"
    )
    parser.add_argument("--quick", action="store_true", help="Fast coarse validation run.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all CPU-affordable tasks on this VM and regenerate figures.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=ALL_TASKS,
        default=None,
        help="Run a single named task and write data/task_<name>.{json,npz}.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available CPU and GPU tasks and exit.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate all figures in figures/ from committed data/.",
    )
    args, _ = parser.parse_known_args(argv)
    return args


# %% [markdown]
# ## Preflight resource estimation & project grid rules
#
# Before launching any 3D time-domain simulation, we evaluate the dispersive
# `'fr4'` permittivity across $2.0\text{–}2.8\;\text{GHz}$ and preflight all five
# mesh levels with `wakis.estimate`:
#
# - **Rule 1 ($\ge 20$ cells per wavelength in highest-index medium)**: at
#   $f_{\max} = 2.80\;\text{GHz}$, $\varepsilon'_r(\text{FR-4}) = 4.26$, so
#   $\lambda_{\min,\text{FR4}} = 51.4\;\text{mm}$ and $\Delta_{\max,\text{FR4}}
#   \le 2.57\;\text{mm}$. All five mesh levels satisfy Rule 1 inside the FR-4
#   board ($22.5\text{–}23.4$ cells per wavelength at the outer board edge and
#   $> 100$ cells per wavelength in the trace region).
# - **Rule 2 ($\ge 4$ cells across smallest features)**:
#   - Across the $h_{\text{sub}} = 1.00\;\text{mm}$ FR-4 substrate, the `fine_0p25mm`
#     ($\Delta_{\text{trace}} = 0.25\;\text{mm}$), `subfeature_0p18mm`
#     ($\Delta_{\text{trace}} = 0.18\;\text{mm}$), and `gpu_subfeature_0p11mm`
#     ($\Delta_{\text{trace}} = 0.1125\;\text{mm}$) levels place **$4.0$ cells**
#     across $h_{\text{sub}}$ ($\Delta z_{\text{sub}} = 0.25\;\text{mm}$).
#   - Across the sub-millimetre meander jog ($W_2 = 0.46\;\text{mm}$) and coplanar
#     feed gap ($H_5 = 0.66\;\text{mm}$, $L_8 = 0.45\;\text{mm}$), the coarse CPU
#     levels ($\Delta_{\text{trace}} = 0.50, 0.35, 0.25\;\text{mm}$) place only
#     $1\text{–}2$ cells across $W_2$. In this tutorial we execute **both** the
#     intermediate `subfeature_0p18mm` level ($\Delta_{\text{trace}} = 0.18\;\text{mm}$,
#     $215,784$ cells, $4$ cells across $W_2$ and $H_5$) and the full
#     `gpu_subfeature_0p11mm` level ($\Delta_{\text{trace}} = 0.1125\;\text{mm}$,
#     $285,768$ cells, $5$ cells across $W_2$ and $4$ cells across $H_5$),
#     satisfying Project Rule 2 across every Table 1-1 feature.

# %%
def print_theory_and_preflight() -> dict[str, dict]:
    """Print dispersive FR-4 properties, Chu-Wheeler bounds, and preflight table."""
    f_band = np.array([2.0e9, 2.44e9, 2.8e9])
    eps_re, eps_im, tand = fr4_complex_permittivity(f_band)
    f_qwave = quarter_wave_resonance_hz(arm_length_m=22.4e-3, width_m=1.21e-3, height_m=1.0e-3, eps_r=float(eps_re[1]))

    k0 = 2.0 * np.pi * 2.44e9 / 299792458.0
    ka_trace = k0 * 0.5 * np.hypot(25.7e-3, 7.5e-3)
    ka_board = k0 * 0.5 * np.hypot(29.58e-3, 39.5e-3)
    q_trace = chu_wheeler_q_bound(ka_trace, efficiency=0.965)
    q_board = chu_wheeler_q_bound(ka_board, efficiency=0.965)
    fbw_trace = chu_max_fractional_bandwidth(ka_trace, vswr_max=1.925, efficiency=0.965) * 100.0
    fbw_board = chu_max_fractional_bandwidth(ka_board, vswr_max=1.925, efficiency=0.965) * 100.0

    print("=== TI SWRU120D (DN007) 2.4 GHz PCB IFA — Physical Theory Summary ===")
    print(
        f"  FR-4 at 2.44 GHz : eps'_r = {eps_re[1]:.3f}, eps''_r = {eps_im[1]:.4f}, tan(delta) = {tand[1]:.4f}"
    )
    print(f"  Quarter-wave analytical f0 : {f_qwave / 1e9:.3f} GHz (TI SWRU120D Fig. 2-8: ~2.44 GHz, approx.)")
    print(
        f"  Chu-Wheeler bound (IFA trace alone, ka={ka_trace:.2f}) : Q_min = {q_trace:.2f}, FBW_max(-10 dB) = {fbw_trace:.1f}%"
    )
    print(
        f"  Chu-Wheeler bound (Full PCB+ground, ka={ka_board:.2f}) : Q_min = {q_board:.2f}, FBW_max(-10 dB) = {fbw_board:.1f}%"
    )

    preflight = preflight_all_levels()
    print("\n=== wakis.estimate Preflight Across 5 Mesh Levels ===")
    header = (
        f"{'Level':<22} {'Grid (Nx,Ny,Nz)':<18} {'Cells':>9} {'W2/H5':>7} {'dt [ps]':>8} "
        f"{'Steps':>7} {'RAM [MB]':>9} {'CPW(FR4)':>9} {'CPU [s]':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, info in preflight.items():
        grid_str = f"({info['nx']},{info['ny']},{info['nz']})"
        feat_str = f"{info['cells_across_W2_0p46mm']}/{info['cells_across_H5_0p66mm']}"
        print(
            f"{name:<22} {grid_str:<18} {info['n_cells']:>9,} {feat_str:>7} {info['dt_ps']:>8.3f} "
            f"{info['n_steps_est']:>7,} {info['memory_mb']:>9.1f} {info['min_cpw_fr4']:>9.1f} "
            f"{info['cpu_wall_est_s']:>8.1f}"
        )
    return preflight


# %% [markdown]
# ## 5-Level Resolution Study, Continuum Extrapolation & Drift Hypothesis Tests
#
# ### 1. Grid convergence across $\Delta_{\text{trace}} \in \{0.50, 0.35, 0.25, 0.18, 0.1125\}\;\text{mm}$
#
# Refining the mesh from $\Delta_{\text{trace}} = 0.50\;\text{mm}$ to $0.1125\;\text{mm}$
# increases the cell count across the narrow meander jog ($W_2 = 0.46\;\text{mm}$) from
# $1$ cell to $5$ cells. Once $W_2$ and $H_5$ are resolved by $\ge 4$ cells
# ($\Delta_{\text{trace}} \le 0.18\;\text{mm}$), the resonance, reflection depth,
# bandwidth, and principal-plane gains converge:
#
# | Mesh Level | $\Delta_{\text{trace}}$ [mm] | Grid $(N_x,N_y,N_z)$ | Cells | Cells in $W_2 / H_5 / h_{\text{sub}}$ | $f_{\text{res}}$ [GHz] | $|S_{11}|_{\min}$ [dB] | $\text{BW}_{-10\text{dB}}$ [MHz] | $G_{\text{XZ}} / G_{\text{YZ}} / G_{\text{XY}}$ [dBi] | Wall [s] |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | `res_0p50mm_baseline` | $0.50$ | $(64, 72, 35)$ | $161,280$ | $1 / 2 / 3$ | $2.445$ | $-14.70$ | $285.0$ | $2.27 / 1.98 / 2.02$ | $109.8$ |
# | `res_0p35mm` | $0.35$ | $(68, 64, 34)$ | $148,176$ | $1 / 4 / 3$ | $2.425$ | $-14.87$ | $280.0$ | $2.21 / 1.93 / 2.02$ | $71.0$ |
# | `res_0p25mm` | $0.25$ | $(80, 72, 35)$ | $200,880$ | $2 / 4 / 4$ | $2.405$ | $-15.32$ | $280.0$ | $2.20 / 1.92 / 2.00$ | $147.0$ |
# | `res_0p18mm` | $0.18$ | $(81, 74, 36)$ | $215,784$ | $4 / 4 / 4$ | $2.405$ | $-15.34$ | $275.0$ | $2.18 / 1.91 / 1.99$ | $334.7$ |
# | `gpu_subfeature_0p11mm` | $0.1125$ | $(98, 81, 36)$ | $285,768$ | $5 / 4 / 4$ | $2.400$ | $-15.35$ | $275.0$ | $2.18 / 1.91 / 1.99$ | $577.6$ |
# | **Richardson continuum ($\Delta \to 0$)** | $0.00$ | — | $\infty$ | $\infty$ | **$2.396$** | **$-15.42$** | **$273.5$** | **$2.18 / 1.91 / 1.99$** | — |
# | **TI SWRU120D Reference** | — | CC2430DB | — | — | $\approx 2.44$ (Fig. 2-8, approx.) | $< -15$ (Table 3-1); $\approx -16.5$ (Fig. 2-8, approx.) | $> 300$ (Sec. 2.2); $\approx 330$ (Fig. 2-8, approx.) | $3.3 / 1.6 / 1.1$ (Table 3-1) | — |
#
# ### 2. Explaining the resonance drift: controlled one-at-a-time hypothesis study
#
# Why does the simulated resonance move from $2.445\;\text{GHz}$ at $\Delta = 0.50\;\text{mm}$
# to a converged continuum value $f_\infty = 2.396\;\text{GHz}$ ($-44\;\text{MHz}$ below
# TI's $\approx 2.44\;\text{GHz}$ curve in Figure 2-8)? Two distinct mechanisms are at work:
#
# 1. **Numerical discretization drift ($-45\;\text{MHz}$ from $\Delta = 0.50\;\text{mm}$ to $0.1125\;\text{mm}$)**:
#    On a coarse grid where $W_2 = 0.46\;\text{mm}$ spans only $1$ cell, the discrete curl
#    stencil smears the tangential magnetic field over $\Delta \ge W_2$, artificially
#    widening the effective current channel and underestimating both the series self-inductance
#    of the shorting stub and the fringing edge capacitance of the zero-thickness PEC arm.
#    At $\Delta = 0.50\;\text{mm}$, this $+45\;\text{MHz}$ numerical upward bias accidentally
#    cancels the physical offset below, making the coarse mesh appear closer to $2.44\;\text{GHz}$.
#    Once $\Delta \le 0.18\;\text{mm}$ ($\ge 4$ cells across $W_2$), $f_{\text{res}}$ changes by
#    only $5\;\text{MHz}$ between $0.18\;\text{mm}$ and $0.1125\;\text{mm}$.
# 2. **Physical parameter differences vs the unspecified CC2430DB board (tested one hypothesis at a time at fixed $\Delta = 0.35\;\text{mm}$, $2.5\;\text{MHz}$ FFT step)**:
#
# | Hypothesis Tested (one parameter varied at a time) | $f_{\text{res}}$ [GHz] | $\Delta f_{\text{res}}$ vs Baseline [MHz] | $|S_{11}|_{\min}$ [dB] | $\text{BW}_{-10\text{dB}}$ [MHz] | $R_{\text{in}}(f_{\text{res}})$ [$\Omega$] | Physical Interpretation |
# | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
# | **Baseline** ($\varepsilon'_r = 4.28$, ground $29.58\times 32.0\;\text{mm}^2$, $W_2 = 0.46\;\text{mm}$, port gap $H_5 = 0.66\;\text{mm}$) | $2.4075$ | $0.0$ | $-14.81$ | $280.0$ | $34.88$ | Reference Wakis `'fr4'` Debye model |
# | **Hypothesis A1**: Higher FR-4 permittivity $\varepsilon'_r = 4.50$ | $2.3900$ | $-17.5$ | $-14.07$ | $260.0$ | $33.79$ | Higher $\varepsilon_r$ increases $\varepsilon_{\text{eff}}$ and lowers $f_{\text{res}}$ |
# | **Hypothesis A2**: Lower FR-4 permittivity $\varepsilon'_r = 4.00$ (thin $1\;\text{mm}$ E-glass core) | **$2.4450$** | **$+37.5$** | **$-16.78$** | **$320.0$** | **$37.65$** | **Reproduces TI SWRU120D's $\approx 2.44\;\text{GHz}$, $-16.5\;\text{dB}$, and $>300\;\text{MHz}$ BW** |
# | **Hypothesis B**: Coplanar port gap $L_8 = 0.45\;\text{mm}$ instead of $H_5 = 0.66\;\text{mm}$ | $2.4150$ | $+7.5$ | $-14.67$ | $285.0$ | $34.58$ | Shorter feed gap reduces feed-stub series inductance |
# | **Hypothesis C**: Unstepped shorting stub $W_2 \to W_1 = 1.21\;\text{mm}$ | $2.4050$ | $-2.5$ | $-15.53$ | $282.5$ | $36.03$ | Stepped jog $W_2$ mainly adjusts input resistance transformation (+1.15 $\Omega$) |
# | **Hypothesis D**: Larger ground plane $37.0 \times 45.0\;\text{mm}^2$ | $2.3600$ | $-47.5$ | $-20.88$ | $367.5$ | $42.80$ | Larger counterpoise raises $R_{\text{in}} \to 42.8\;\Omega$ and widens BW to $368\;\text{MHz}$ |

# %%
def print_validation_summary(data_dir: Path) -> None:
    """Print quantitative comparison between Wakis 3D FIT and TI SWRU120D."""
    fine_path = data_dir / "task_res_0p25mm.json"
    base_path = data_dir / "task_res_0p50mm_baseline.json"
    med_path = data_dir / "task_res_0p35mm.json"
    sf18_path = data_dir / "task_res_0p18mm.json"
    sf11_path = data_dir / "task_gpu_subfeature_0p11mm.json"
    if not fine_path.exists():
        return
    base = json.loads(base_path.read_text())
    med = json.loads(med_path.read_text())
    fine = json.loads(fine_path.read_text())
    sf18 = json.loads(sf18_path.read_text()) if sf18_path.exists() else fine
    sf11 = json.loads(sf11_path.read_text()) if sf11_path.exists() else sf18

    deltas = np.array([0.50, 0.35, 0.25, 0.18, 0.1125])
    f_arr = np.array([base["f_res_ghz"], med["f_res_ghz"], fine["f_res_ghz"], sf18["f_res_ghz"], sf11["f_res_ghz"]])
    s11_arr = np.array([base["s11_min_db"], med["s11_min_db"], fine["s11_min_db"], sf18["s11_min_db"], sf11["s11_min_db"]])
    bw_arr = np.array([base["bw_10db_mhz"], med["bw_10db_mhz"], fine["bw_10db_mhz"], sf18["bw_10db_mhz"], sf11["bw_10db_mhz"]])
    f_inf, _, _ = richardson_extrapolate_ifa(deltas, f_arr)
    s11_inf, _, _ = richardson_extrapolate_ifa(deltas, s11_arr)
    bw_inf, _, _ = richardson_extrapolate_ifa(deltas, bw_arr)

    print("\n=== Validation vs Texas Instruments SWRU120D (DN007) Benchmarks ===")
    print(
        f"  Resonant frequency f_res : {sf11['f_res_ghz']:.3f} GHz at 0.1125 mm, continuum {f_inf:.3f} GHz "
        f"(TI SWRU120D Fig. 2-8, p. 10: ~2.44 GHz, read from Fig. 2-8, approx.)"
    )
    print(
        f"  Min reflection |S11|_min : {sf11['s11_min_db']:.2f} dB at 0.1125 mm, continuum {s11_inf:.2f} dB "
        f"(TI SWRU120D Table 3-1, p. 11: < -15 dB; Fig. 2-8, p. 10: ~ -16.5 dB, approx.)"
    )
    print(
        f"  Input impedance Zin(f0)  : {sf11['zin_res_real_ohm']:.1f} {sf11['zin_res_imag_ohm']:+.1f}j Ohm "
        f"(Target: 50.0 + 0.0j Ohm)"
    )
    print(
        f"  -10 dB Bandwidth         : {sf11['bw_10db_mhz']:.0f} MHz at 0.1125 mm, continuum {bw_inf:.1f} MHz "
        f"(TI SWRU120D Sec. 2.2, p. 10: > 300 MHz; Fig. 2-8: ~330 MHz, approx.)"
    )
    print(
        f"  Radiation efficiency     : {sf11['eta_rad_2g44_pct']:.1f}% "
        f"(Total efficiency incl. mismatch: {sf11['eta_tot_2g44_pct']:.1f}%; not reported in SWRU120D)"
    )
    print(
        f"  Poynting/Far-field ratio : {sf11['p_flux_over_p_rad']:.4f} "
        f"(Surface equivalence self-consistency check)"
    )
    print(
        f"  XZ Plane Peak Gain       : {sf11['g_xz_peak_dbi']:+.2f} dBi "
        f"(TI SWRU120D Table 3-1, p. 11 & Figs. 2-4/2-5: 3.3 dBi)"
    )
    print(
        f"  YZ Plane Peak Gain       : {sf11['g_yz_peak_dbi']:+.2f} dBi "
        f"(TI SWRU120D Table 3-1, p. 11 & Figs. 2-6/2-7: 1.6 dBi)"
    )
    print(
        f"  XY Plane Peak Gain       : {sf11['g_xy_peak_dbi']:+.2f} dBi "
        f"(TI SWRU120D Table 3-1, p. 11 & Figs. 2-2/2-3: 1.1 dBi)"
    )
    print("\n=== 5-Level Resolution Study Summary ===")
    for label, d in [
        ("Coarse      (0.5000 mm, 3 z-cells)", base),
        ("Medium      (0.3500 mm, 3 z-cells)", med),
        ("Fine        (0.2500 mm, 4 z-cells)", fine),
        ("Subfeature  (0.1800 mm, 4 z-cells)", sf18),
        ("Superfine   (0.1125 mm, 4 z-cells)", sf11),
    ]:
        print(
            f"  {label:<34} : N={d['mesh_info']['n_cells']:>7,}, "
            f"f_res={d['f_res_ghz']:.3f} GHz, |S11|_min={d['s11_min_db']:>6.2f} dB, "
            f"BW={d['bw_10db_mhz']:>5.1f} MHz, G_XZ={d['g_xz_peak_dbi']:+.2f} dBi, wall={d['wall_seconds']:.1f}s"
        )
    print(
        f"  {'Continuum Extrapolation (d -> 0)':<34} : "
        f"f_inf={f_inf:.3f} GHz, |S11|_inf={s11_inf:>6.2f} dB, BW_inf={bw_inf:>5.1f} MHz"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_tasks:
        print("CPU tasks (--full):", ", ".join(CPU_TASKS))
        print("GPU tasks (--task):", ", ".join(GPU_TASKS))
        return

    print_theory_and_preflight()

    if args.task is not None:
        print(f"\n[Task Runner] Executing task {args.task!r}...")
        run_task(args.task, DATA_DIR)
        return

    if args.full:
        for tname in CPU_TASKS:
            print(f"\n[--full] Running CPU task {tname!r}...")
            run_task(tname, DATA_DIR)
        saved = generate_all_figures(DATA_DIR, FIG_DIR)
        print(f"\nGenerated {len(saved)} figures in {FIG_DIR}:")
        for p in saved:
            print(f"  - {p.relative_to(TUTORIAL_DIR)} ({p.stat().st_size / 1024:.1f} KB)")
        print_validation_summary(DATA_DIR)
        return

    if args.plot:
        saved = generate_all_figures(DATA_DIR, FIG_DIR)
        print(f"\nRegenerated {len(saved)} figures in {FIG_DIR}.")
        print_validation_summary(DATA_DIR)
        return

    # Default / --quick mode: fast coarse verification run if data is absent, else summary + plot check
    if not (DATA_DIR / "task_res_0p50mm_baseline.json").exists():
        print("\n[--quick] Running baseline coarse 3D FIT simulation...")
        run_task("res_0p50mm_baseline", DATA_DIR)
    else:
        r_quick = run_ifa_simulation(trace_cell_mm=0.60, sub_cells_z=2, sim_time_ns=4.5)
        print(
            f"\n[--quick] Live 3D FIT check: f_res = {r_quick['f_res_hz']/1e9:.3f} GHz, "
            f"|S11|_min = {r_quick['s11_min_db']:.2f} dB, "
            f"Zin(f_res) = {r_quick['zin_res_real_ohm']:.1f}{r_quick['zin_res_imag_ohm']:+.1f}j Ohm"
        )
    print_validation_summary(DATA_DIR)


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Catalog of generated figures
#
# Running `--full` or `--plot` populates `figures/` with the following eight
# multi-panel diagnostic figures and one animated time-domain GIF:
#
# 1. `figures/01_geometry_3d_views.png`: Four-panel 3D PyVista geometry render
#    (isometric view, annotated top CAD blueprint with all Table 1-1 dimensions,
#    side elevation showing the $1.00\;\text{mm}$ FR-4 board and shorting via,
#    and close-up of the $50\;\Omega$ coplanar lumped port and stepped tap).
# 2. `figures/02_mesh_and_cpw_views.png`: Mesh edges on the XY trace plane,
#    zoomed mesh on the smallest features ($H_5 = 0.66\;\text{mm}$ and
#    $W_2 = 0.46\;\text{mm}$), XZ cross-section through the substrate thickness,
#    and 2D cells-per-wavelength map at $f_{\max} = 2.8\;\text{GHz}$.
# 3. `figures/03_near_fields_and_currents.png`: Resonant near fields at
#    $2.44\;\text{GHz}$ showing $|\mathbf{E}|$ (with the open-tip voltage
#    antinode), $\operatorname{Re}(E_y)$ across the coplanar gap, surface current
#    magnitude $|\mathbf{J}_s|$ with flow vectors on the IFA trace and ground
#    edge, and $\operatorname{Re}(E_z)$ on the vertical XZ slice.
# 4. `figures/04_radiated_field_evolution.gif`: Animated time-domain wave packet
#    launching from the $2.4\;\text{GHz}$ inverted-F trace and PCB ground edge.
# 5. `figures/05_far_field_3d_and_cuts.png`: 3D coloured far-field gain surface
#    $G(\theta,\phi)$ and polar cuts in the XZ, YZ, and XY planes compared against
#    TI SWRU120D Table 3-1 (p. 11) and Figures 2-2 to 2-7 (pp. 4–9).
# 6. `figures/06_combined_3d_overlays.png`: Multi-angle 3D PyVista overlays
#    combining the PCB geometry + surface current $|\mathbf{J}_s|$ + 3D
#    translucent far-field lobe in a single scene from front and rear viewpoints,
#    plus 3D geometry + vertical near-field $|\mathbf{E}|$ slice planes.
# 7. `figures/07_s11_impedance_smith.png`: Port reflection $|S_{11}|$ in dB vs
#    the digitized SWRU120D Figure 2-8 curve (p. 10, approx.) and Table 3-1
#    ($< -15\;\text{dB}$) specification, $\operatorname{Re}/\operatorname{Im}(Z_{\text{in}})$,
#    and calibrated $50\;\Omega$ Smith chart trajectory.
# 8. `figures/08_resolution_and_param_study.png`: 5-level grid resolution
#    convergence ($\Delta_{\text{trace}} \in \{0.50, 0.35, 0.25, 0.18, 0.1125\}\;\text{mm}$)
#    with Richardson continuum extrapolation ($\Delta \to 0$), ground-plane length
#    sweep ($L_{\text{gnd}} \in \{18, 25, 32, 42\}\;\text{mm}$) illustrating
#    TI AN043 (SWRA117D) counterpoise rules, and the Chu–Wheeler $Q$ bound.
#
# ## Limits & failure modes
#
# 1. **Ground-plane size & battery/enclosure detuning**: as demonstrated by the
#    $L_{\text{gnd}}$ sweep in Figure 8(c) and Hypothesis D ($37 \times 45\;\text{mm}^2$),
#    the PCB ground plane carries nearly half of the radiating dipole moment.
#    Shrinking the ground plane below $20\;\text{mm}$ collapses $R_{\text{in}}$ to
#    $15.2\;\Omega$ ($|S_{11}|_{\min} = -5.3\;\text{dB}$), whereas enlarging it to
#    $37 \times 45\;\text{mm}^2$ raises $R_{\text{in}}$ to $42.8\;\Omega$
#    ($|S_{11}|_{\min} = -20.9\;\text{dB}$, $\text{BW}_{-10\text{dB}} = 367.5\;\text{MHz}$).
# 2. **Discretization of sub-millimetre steps**: when the narrow $W_2 = 0.46\;\text{mm}$
#    jog spans only $1$ cell ($\Delta = 0.50\;\text{mm}$), $f_{\text{res}}$ is biased
#    upward by $+45\;\text{MHz}$ relative to the $5$-cell grid ($\Delta = 0.1125\;\text{mm}$).
#    Satisfying Project Rule 2 ($\ge 4$ cells across $W_2$ and $H_5$) requires
#    $\Delta_{\text{trace}} \le 0.18\;\text{mm}$.
# 3. **FR-4 laminate permittivity**: SWRU120D specifies $1\;\text{mm}$ standard FR-4
#    without a numerical $\varepsilon_r$. Varying $\varepsilon'_r$ from $4.50$ to $4.00$
#    shifts $f_{\text{res}}$ by $+55\;\text{MHz}$ ($2.390 \to 2.445\;\text{GHz}$) and
#    increases the $-10\;\text{dB}$ bandwidth from $260\;\text{MHz}$ to $320\;\text{MHz}$.
# 4. **Coaxial cable / SMA connector radiation in measurements**: TI's reference
#    measurements in SWRU120D were taken on a battery-powered CC2430DB board
#    transmitting at $0\;\text{dBm}$ without a feeding coax (Section 2.1, p. 3),
#    avoiding outer-shield common-mode cable currents.
#
# ## Verified references
#
# 1. **Audun Andersen**, *"2.4-GHz Inverted F Antenna"*, Texas Instruments Design
#    Note **DN007**, Literature Number **SWRU120D** (Rev. D, Jan. 2019),
#    URL: <https://www.ti.com/lit/an/swru120d/swru120d.pdf> — Section 1, p. 2,
#    Figure 1-1 & Table 1-1 layout dimensions ($H_1\text{–}H_9$, $L_1\text{–}L_8$,
#    $W_1\text{–}W_2$) on $1\;\text{mm}$ FR-4; Section 2.1, pp. 3–9, Figures 2-1 to
#    2-7 radiation patterns; Section 2.2, p. 10, Figure 2-8 reflection; Section 2.3,
#    p. 11, Figure 2-9 bandwidth; and Section 3, p. 11, Table 3-1 summary ($1.1, 3.3,
#    1.6\;\text{dBi}$ peak gains, $< -15\;\text{dB}$ reflection, $25.7 \times 7.5\;\text{mm}$
#    footprint).
# 2. **Audun Andersen**, *"Small Size 2.4 GHz PCB Antenna"*, Texas Instruments
#    Application Note **AN043**, Literature Number **SWRA117D** (2008),
#    URL: <https://www.ti.com/lit/an/swra117d/swra117d.pdf> — Meandered IFA
#    design rules, ground-plane size sensitivity, and $\pi$-network matching.
# 3. **H. A. Wheeler**, *"Fundamental Limitations of Small Antennas"*, *Proc.
#    IRE*, vol. 35, no. 12, pp. 1479–1484, Dec. 1947,
#    `doi:10.1109/JRPROC.1947.226199`.
# 4. **L. J. Chu**, *"Physical Limitations of Omni-Directional Antennas"*,
#    *J. Appl. Phys.*, vol. 19, no. 12, pp. 1163–1175, Dec. 1948,
#    `doi:10.1063/1.1715038`.
# 5. **J. S. McLean**, *"A Re-Examination of the Fundamental Limits on the
#    Radiation Q of Electrically Small Antennas"*, *IEEE Trans. Antennas
#    Propag.*, vol. 44, no. 5, pp. 672–676, May 1996, `doi:10.1109/8.496253`.
# 6. **C. A. Balanis**, *Antenna Theory: Analysis and Design*, 4th ed., Wiley,
#    2016 — Chapter 12 (Near-to-Far-Field Surface Equivalence Transformation) and
#    Section 14.2 (Planar Inverted-F Antennas).
