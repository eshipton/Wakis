# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# %% [markdown]
# # Tutorial 5.3 · Mode size converters: tapers, Luneburg lens and semi-lens
#
# **Abstract.** Coupling light efficiently between wide multi-mode silicon
# waveguides ($w \sim 9\text{--}10\;\mu\text{m}$, used for low-loss delay lines,
# grating couplers, and star couplers) and submicron single-mode strip waveguides
# ($w \sim 0.45\text{--}0.50\;\mu\text{m}$, used for compact bends and routing)
# is a fundamental challenge in silicon photonic integrated circuits.
# Whereas conventional linear adiabatic tapers require lengths $L_t \gtrsim
# 50\text{--}100\;\mu\text{m}$ to suppress coupling into higher-order symmetric
# slab modes, metamaterial and wavefront-transforming mode-size converters shrink
# the transition footprint by an order of magnitude. In this tutorial we
# benchmark three mode-size converter concepts in Wakis using the **2D
# effective-index method (EIM)** and **Kottke subpixel permittivity smoothing**:
# (1) conventional linear and non-linear tapers ($L_t \in [0, 100]\;\mu\text{m}$),
# (2) a $11.0\;\mu\text{m}$-diameter graded-thickness **Luneburg lens** waveguide
# converter (Badri & Gilarlue 2019 [1]), and (3) a $9.50\;\mu\text{m}$-long
# 6-segment curvature-controlled **semi-lens beam expander** (Abbaslou *et al.*
# 2017 [2]).
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Explain the **Milton–Burns adiabaticity criterion** for waveguide tapers
#    and verify why an abrupt width step ($L_t \to 0$) reduces to the transverse
#    eigenmode overlap integral $|\langle E_{\text{in}}, E_{\text{out}}\rangle|^2$.
# 2. Synthesize a 2D graded-index (GRIN) **Luneburg lens** profile
#    $n_{\text{eff}}(r) = n_{\text{edge}}\sqrt{2 - (r/R)^2}$ by calibrating the
#    1D vertical silicon slab thickness $t_{\text{Si}}(r) \in [0.11, 0.36]\;\mu\text{m}$
#    via a cubic polynomial fit $t_{\text{fit}}(n_{\text{eff}})$ (Badri & Gilarlue 2019 [1]).
# 3. Construct the 6-segment piecewise power-law **semi-lens beam expander** of
#    Abbaslou *et al.* (2017) [2] and decompose the expanded output beam across
#    the first five even modes ($\text{TE}_0, \text{TE}_2, \text{TE}_4, \text{TE}_6, \text{TE}_8$).
# 4. Quantify the convergence rate of **Kottke subpixel smoothing** versus
#    staircasing against an independent fine-grid reference simulation.
#
# ## Contents
#
# 1. [Physical background & device specifications](#1.-Physical-background-&-device-specifications)
# 2. [Effective-Index Method (EIM) & Luneburg slab calibration](#2.-Effective-Index-Method-(EIM)-&-Luneburg-slab-calibration)
# 3. [Parametric geometry construction](#3.-Parametric-geometry-construction)
# 4. [Subpixel permittivity smoothing vs staircasing](#4.-Subpixel-permittivity-smoothing-vs-staircasing)
# 5. [Study tasks & execution modes](#5.-Study-tasks-&-execution-modes)
# 6. [Generating the complete figure suite](#6.-Generating-the-complete-figure-suite)
# 7. [Limits & when 2D EIM breaks down](#7.-Limits-&-when-2D-EIM-breaks-down)
# 8. [Exercises](#8.-Exercises)
# 9. [References](#9.-References)

# %% [markdown]
# ## 1. Physical background & device specifications
#
# ### 1.1 Adiabaticity criterion & abrupt-step overlap limit
#
# Consider a symmetric planar waveguide along $x$ whose width varies from
# $w_{\text{in}}$ to $w_{\text{out}}$ over length $L_t$. Because the structure
# is mirror-symmetric ($y \mapsto -y$), a fundamental symmetric $\text{TE}_0$
# input mode couples only to even higher-order local normal modes
# ($\text{TE}_2, \text{TE}_4, \dots$). According to the **Milton–Burns
# adiabaticity condition** [3], local mode conversion from $\text{TE}_0$ into
# $\text{TE}_2$ remains small only when the local flare angle $\theta(x) =
# \frac{1}{2}|\mathrm{d}w/\mathrm{d}x|$ is much smaller than the ratio of the
# local beat length $L_b(x) = 2\pi / [\beta_0(x) - \beta_2(x)]$ to the local
# waveguide width:
#
# $$\left|\frac{\mathrm{d}w}{\mathrm{d}x}\right| \ll \frac{w(x)\,\left[\beta_0(x) - \beta_2(x)\right]}{\pi} \approx \frac{\lambda_0}{2\,n_{\text{eff}}(x)\,w(x)}.$$
#
# Integrating this condition for a linear taper ($\mathrm{d}w/\mathrm{d}x =
# \text{const}$) shows that the required adiabatic length scales quadratically
# with the wide waveguide width: $L_t \propto n_{\text{eff}}\,w_{\text{in}}^2 /
# \lambda_0$. For $w_{\text{in}} = 10\;\mu\text{m}$ at $\lambda_0 =
# 1.55\;\mu\text{m}$, low-loss linear tapering requires $L_t \gtrsim
# 50\text{--}100\;\mu\text{m}$. In the opposite limit of an **abrupt butt-coupled
# step** ($L_t \to 0$), neglecting small Fresnel reflection, the power coupled
# into the fundamental $\text{TE}_0$ mode of the narrow waveguide is governed
# by the normalized transverse mode-overlap integral:
#
# $$\eta_{\text{step}} \approx \frac{\left|\int_{-\infty}^{+\infty} E_{y,\text{in}}(y)\,E_{y,\text{out}}^*(y)\,\mathrm{d}y\right|^2}{\left(\int |E_{y,\text{in}}|^2\,\mathrm{d}y\right)\left(\int |E_{y,\text{out}}|^2\,\mathrm{d}y\right)}.$$
#
# ### 1.2 Graded-thickness Luneburg lens converter (Badri & Gilarlue 2019)
#
# A classical **Luneburg lens** of radius $R$ is a circularly symmetric GRIN
# medium with refractive-index profile
#
# $$n_{\text{eff}}(r) = n_{\text{edge}}\sqrt{2 - \left(\frac{r}{R}\right)^2}, \qquad 0 \le r \le R,$$
#
# which focuses an incident collimated plane wave at the western rim
# ($x = -R, y = 0$) to a diffraction-limited spot at the opposite eastern rim
# ($x = +R, y = 0$) without spherical aberration. **Badri & Gilarlue (2019)** [1]
# realized this profile on SOI between a $w_{\text{in}} = 10.0\;\mu\text{m}$
# input waveguide and a $w_{\text{out}} = 0.50\;\mu\text{m}$ single-mode output
# waveguide ($t_{\text{wg}} = 0.11\;\mu\text{m}$, $n_{\text{edge}} \approx 2.22$)
# by greyscale-lithography grading of the silicon core thickness $t_{\text{lens}}(r)$
# from $0.11\;\mu\text{m}$ at $r = R = 5.5\;\mu\text{m}$ up to $\approx 0.36\;\mu\text{m}$
# at the lens center $r = 0$.
#
# ### 1.3 Six-segment semi-lens beam expander (Abbaslou *et al.* 2017)
#
# Rather than multi-level greyscale etching, **Abbaslou *et al.* (2017)** [2]
# designed a single-etch ($t_{\text{wg}} = 0.26\;\mu\text{m}$) planar beam
# expander from $w_0 = 0.45\;\mu\text{m}$ to $w_6 = 9.00\;\mu\text{m}$ in just
# $L_{\text{tot}} = 9.50\;\mu\text{m}$ by dividing the transition into six
# consecutive segments $i \in \{1,\dots,6\}$ of length $L_i$, end widths
# $(w_{i-1}, w_i)$, and curvature exponent $m_i$:
#
# $$w(x_i) = (w_{i-1} - w_i)\left|\frac{x_i - L_i}{L_i}\right|^{m_i} + w_i, \qquad x_i \in [0, L_i].$$
#
# Segment 3 ($L_3 = 0.05\;\mu\text{m}$) and Segment 4 ($w = 9.06 \to 10.0\;\mu\text{m}$)
# act as an integrated convex dielectric lens: the input $\text{TE}_0$ mode
# diverges rapidly into a cylindrical wavefront and is then collimated by the
# curved high-contrast boundary of Segment 5 ($w = 10.0 \to 3.19\;\mu\text{m}$,
# $m_5 = 0.32$) and Segment 6 ($w = 3.19 \to 9.00\;\mu\text{m}$, $m_6 = 2.55$)
# so that $> 80\%$ of the output power is recaptured in the fundamental
# $\text{TE}_0$ mode of the $9.0\;\mu\text{m}$ waveguide.
#
# | Design | Reference | Thickness $t_{\text{Si}}$ | Width transition | Converter length |
# |---|---|---|---|---|
# | **1. Linear / Non-linear Tapers** | Milton & Burns (1977) [3] | $0.11\;\mu\text{m}$ | $10.0\;\mu\text{m} \to 0.50\;\mu\text{m}$ | $L_t \in \{0, 5, 10, 20, 50, 100\}\;\mu\text{m}$ |
# | **2. Graded Luneburg Lens** | Badri & Gilarlue (2019) [1] | $0.11 \to 0.36\;\mu\text{m}$ | $10.0\;\mu\text{m} \to 0.50\;\mu\text{m}$ | $D_{\text{lens}} = 11.0\;\mu\text{m}$ ($R = 5.5\;\mu\text{m}$) |
# | **3. 6-Segment Semi-Lens** | Abbaslou *et al.* (2017) [2] | $0.26\;\mu\text{m}$ | $0.45\;\mu\text{m} \to 9.00\;\mu\text{m}$ | $L_{\text{tot}} = 9.50\;\mu\text{m}$ |
#
# ### 1.4 Simulation setup schematics & mesh views (`00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`, `00b_taper_and_luneburg_schematics.png`)
#
# Because this tutorial benchmarks three distinct mode-size converter topologies,
# the figure suite opens with the two-figure setup layout for Design 3
# (`00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`, and
# `00_setup_table.md`, generated via `wakis.schematic.save_setup_figures`)
# together with the companion schematic for Designs 1 & 2
# (`00b_taper_and_luneburg_schematics.png`):
#
# ![Setup geometry](figures/00a_setup_geometry.png)
# ![Setup mesh and excitation](figures/00b_setup_mesh_and_excitation.png)
#
# The setup table below summarizes the computational domain, CPML absorbing
# boundaries, effective-index materials, port excitation and modal monitors for
# the primary Design 3 (6-segment semi-lens beam expander) simulation along with
# the Design 1 (taper), Design 2 (Luneburg lens), and mesh-refinement study
# variations:
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 12.7 µm × 11.6 µm × 25.4 nm (z ∈ [117.3 nm, 142.7 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 511 × 458 × 1 (234038 cells) |
# | Discretization | Δx & resolution | Δ_min = 6.25 nm, 83.9 cells/λ (in n_max = 2.95), 8.0 cells across feature (50 nm) |
# | Boundary | -x face | CPML (10 cells, 253.5 nm) |
# | Boundary | +x face | CPML (10 cells, 253.4 nm) |
# | Boundary | -y face | CPML (10 cells, 253.6 nm) |
# | Boundary | +y face | CPML (10 cells, 253.6 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Air cladding (n = 1.00, ε_r = 1.00; SiO₂ BOX n_sub = 1.44) (isotropic) |
# | Material | Region 1: Si 6-segment semi-lens (t_Si = 260 nm, n_eff = 2.9545, ε_r = 8.73) | ε_r = 8.73, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.40 µm, 11.2 µm] (d = 12.6 µm) |
# | Source | 1 · ModeSource | pos: x = -1.03 µm, y ∈ [-0.90, 0.90] µm, pol: TE₀ (E_y, H_z), pulse: f₀ = 193.6 THz, Δf = 15.61 THz |
# | Monitor | 2 · ModeMonitor | Port 'in' ModeMonitor (S₁₁), pos: x = -0.95 µm, y ∈ [-0.90, 0.90] µm, freqs: 51 freqs in [187.4, 199.9] THz |
# | Monitor | 3 · ModeMonitor | Port 'out' ModeMonitor (TE₀,₂,₄,₆,₈ + FluxMonitor), pos: x = 10.55 µm, y ∈ [-4.72, 4.72] µm, freqs: 51 freqs in [187.4, 199.9] THz |
# | Monitor | 4 · FluxMonitor | 2D Poynting FluxMonitor box, pos: x ∈ [-1.03, 10.80] µm, y ∈ [-5.35, 5.35] µm, freqs: 51 freqs in [187.4, 199.9] THz |
# | Variation | Design 1: Linear & non-linear tapers (tasks 10–14) | param=w_in = 10.0 µm → w_out = 0.50 µm (t_Si = 110 nm, n_eff,slab = 2.2166), L_t ∈ [0, 100] µm, note=Linear, parabolic, exponential, and cosine tapers vs Milton–Burns adiabatic scaling & abrupt-step overlap |
# | Variation | Design 2: Graded-thickness Luneburg lens (task 9) | param=D_lens = 11.0 µm (R = 5.5 µm, 60 CircularPolySlab shells, t(r) ∈ [110, 356.1] nm), note=n_eff(r) = n_edge √(2 − (r/R)²) ∈ [2.2166, 3.1348] calibrated via 1D SOI slab cubic polynomial fit |
# | Variation | Resolution & subpixel ladder (tasks 0–8) | param=(cpw, N_feat) ∈ {(10, 1) demo, (10, 4) L1, (14, 6) L2, (20, 8) L3, (40, 16) R}, note=Kottke subpixel smoothing vs staircasing across the L₃ = 50 nm Segment 3 flare |
# | Variation | Graded non-uniform Cartesian mesh (task 15) | param=NonUniformSemilensCircuit (424 × 407 × 1 = 172,568 cells vs 235,008 at L3), note=8.0 cells across L₃ = 50 nm (Δx_min = 6.25 nm), non-square lead cells (−26.6% cells, −33.2% time) |
#
# ![Taper and Luneburg schematics](figures/00b_taper_and_luneburg_schematics.png)

# %%
import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.constants import c as c_light, epsilon_0

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _helpers import (  # noqa: E402
    D_LENS,
    L_TOT_SEMILENS,
    MIN_FEATURE_LUNEBURG,
    MIN_FEATURE_SEMILENS,
    MIN_FEATURE_TAPER,
    N_AIR,
    N_FREQS,
    N_SI,
    N_SIO2,
    NonUniformSemilensCircuit,
    R_LENS,
    SEMILENS_L_UM,
    SEMILENS_M,
    SEMILENS_SEG3_X_HI,
    SEMILENS_SEG3_X_LO,
    SEMILENS_W_UM,
    T_WG_SEMILENS,
    T_WG_TAPER,
    W_IN_SEMILENS,
    W_IN_TAPER,
    W_OUT_TAPER,
    WL_MAX,
    WL_MIN,
    WL_REF,
    build_luneburg_converter,
    build_schematic_components_converter,
    build_semilens_expander,
    build_taper_converter,
    calibrate_luneburg_profile,
    compute_abrupt_step_overlap,
    draw_polyslab_regions,
    inspect_task_mesh,
    measure_grid_resolution,
    plot_mesh_upgrade_comparison_semilens,
    plot_setup_schematic_semilens,
    plot_taper_and_luneburg_schematics,
    render_movie_with_overlay,
    run_converter_simulation,
    semilens_mesh_overrides,
    semilens_width_profile,
    taper_width_profile,
    waveguide_width_mesh_overrides,
)

from wakis import figstyle, schematic  # noqa: E402
from wakis.templates.photonics import (  # noqa: E402
    PhotonicCircuit,
    make_movie,
    n_to_eps,
    plot_cross_section_mesh,
    plot_eim_stack,
    plot_field_map,
    plot_neff_dispersion,
    plot_port_modes,
    plot_spectra_db,
)

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# %% [markdown]
# ## 2. Effective-Index Method (EIM) & Luneburg slab calibration
#
# In all three converters the silicon waveguide rests on a lower $\text{SiO}_2$
# substrate ($z \le 0$, $n_{\text{SiO}_2} = 1.44$) with an upper air cladding
# ($z > t_{\text{Si}}$, $n_{\text{air}} = 1.0$). Solving the 1D asymmetric
# three-layer vertical $\text{TE}_0$ slab problem at $\lambda_0 = 1550\;\text{nm}$
# gives:
#
# - For Designs 1 & 2 ($t_{\text{wg}} = 0.11\;\mu\text{m}$):
#   $n_{\text{edge}} = n_{\text{eff,slab}}(0.11\;\mu\text{m}) = 2.2166$.
# - At the center of the Luneburg lens ($r = 0$), the required effective index is
#   $n_{\text{eff}}(0) = \sqrt{2}\,n_{\text{edge}} = 3.1348$, which is synthesized
#   by a cubic polynomial fit
#
#   $$t_{\text{fit}}(n_{\text{eff}}) = c_0 n_{\text{eff}}^3 + c_1 n_{\text{eff}}^2 + c_2 n_{\text{eff}} + c_3$$
#
#   across 30 thicknesses $t_{\text{slab}} \in [0.11, 0.40]\;\mu\text{m}$
#   (verified against 1D FIT `solve_modes` in `task_9.npz` with maximum deviation
#   $\max|\Delta n_{\text{eff}}| = 2.13 \times 10^{-3}$), yielding peak thickness
#   $t_{\text{lens}}(0) = 0.3561\;\mu\text{m}$.
# - For Design 3 ($t_{\text{wg}} = 0.26\;\mu\text{m}$):
#   $n_{\text{eff,slab}}(0.26\;\mu\text{m}) = 2.9545$. At the $w_0 = 0.45\;\mu\text{m}$
#   input strip port (`task_6.npz`), the 2D EIM fundamental mode has
#   $n_{\text{eff,in}}(1550\;\text{nm}) = 2.4719$ and group index
#   $n_g(1550\;\text{nm}) = 3.4859$ ($n_{\text{eff,in}} \in [2.4387, 2.5041]$,
#   $n_g \in [3.4580, 3.5149]$ across $1500\text{--}1600\;\text{nm}$), while at the
#   $w_6 = 9.00\;\mu\text{m}$ output port the first three even modes have
#   $n_{\text{eff,out}}(\text{TE}_0) = 2.9647$,
#   $n_{\text{eff,out}}(\text{TE}_2) = 2.9546$, and
#   $n_{\text{eff,out}}(\text{TE}_4) = 2.9342$.
#
# ## 3. Parametric geometry construction
#
# The helper module `_helpers.py` provides three geometry builders that return
# `(structures, ports, domain_2d)`:
#
# - `build_taper_converter(length_taper, profile)` — constructs linear,
#   parabolic, exponential, or cosine tapers ($w_{\text{in}} = 10.0\;\mu\text{m}
#   \to w_{\text{out}} = 0.50\;\mu\text{m}$) as a `SymmetricProfileSlab`.
# - `build_luneburg_converter(n_shells)` — constructs the input/output waveguides
#   plus concentric `CircularPolySlab` shells whose local permittivity matches
#   $n_{\text{eff}}^2(r_i)$.
# - `build_semilens_expander(n_per_seg)` — constructs the 6-segment piecewise
#   power-law beam expander ($0.45\;\mu\text{m} \to 9.00\;\mu\text{m}$ over
#   $9.50\;\mu\text{m}$) of Abbaslou *et al.* (2017) [2].
#
# ## 4. Subpixel permittivity smoothing vs staircasing
#
# Both the curved semi-lens profile $w(x)$ and the circular shells of the
# Luneburg lens intersect the Cartesian Yee grid at oblique angles. Wakis
# applies diagonal **Kottke (2008) subpixel smoothing** [4] using the local dual
# cell fill fraction $w$ and outward normal $\mathbf{n} = (n_x, n_y)$:
#
# $$\varepsilon_{\text{eff}, c}^{-1} = n_c^2 \left(\frac{w}{\varepsilon_s} + \frac{1 - w}{\varepsilon_b}\right) + \left(1 - n_c^2\right)\left(w\varepsilon_s + (1 - w)\varepsilon_b\right)^{-1}, \qquad c \in \{x, y\}.$$
#
# ### 4.1 Resolution: cells per wavelength and cells across the smallest feature
#
# Accurate FDTD/FIT modeling of integrated photonic mode-size converters requires
# satisfying two independent spatial discretization criteria simultaneously:
#
# 1. **Wave-propagation rule ($\text{cpw} \ge 10$)**: The background grid inside
#    the highest-index guided medium ($n_{\text{eff,slab}}$) must provide at
#    least $10$ cells per wavelength ($\Delta x_{\text{bg}} \le \lambda_{\min} /
#    (n_{\text{eff,slab}}\,\text{cpw})$) in every execution mode (`--quick`,
#    `--full`, and parameter sweeps) to bound numerical phase-velocity dispersion.
# 2. **Geometric-feature rule ($N \ge 4$ cells across the smallest feature)**:
#    A grid cell coarser than a structural transition cannot resolve the boundary
#    regardless of subpixel averaging. Every physical feature of width or length
#    $d_{\min}$ must be spanned by at least $N \ge 4$ grid cells
#    ($\Delta x_{\text{feat}} \le d_{\min} / N$).
#
# Across our three devices, the smallest geometric features are:
#
# - **Design 3 (6-segment semi-lens beam expander)**: Segment 3 ($x \in
#   [4.61, 4.66]\;\mu\text{m}$) flares the waveguide width from $w_2 = 3.30\;\mu\text{m}$
#   to $w_3 = 9.06\;\mu\text{m}$ over an axial length of only
#   **$L_3 = 0.05\;\mu\text{m} = 50\;\text{nm}$**, while the single-mode input
#   strip has width $w_0 = 0.45\;\mu\text{m} = 450\;\text{nm}$. At $\text{cpw} = 10$
#   ($n_{\text{eff,slab}} = 2.9545$), the unrefined background cell size is
#   $\Delta x_{\text{bg}} \approx 50.6\;\text{nm}$, which places only **$0.99$
#   cells** across the $50\;\text{nm}$ flare! We use `PhotonicCircuit(..., mesh_overrides=...)`
#   to pass tensor-product refinement boxes over $x \in [4.61, 4.66]\;\mu\text{m}$
#   with $h_x \le 50\;\text{nm}/N$ and over $y \in [-0.225, +0.225]\;\mu\text{m}$
#   with $h_y \le 450\;\text{nm}/N$, which grade smoothly back to $\Delta x_{\text{bg}}$
#   at ratio $\le 1.4$.
# - **Design 1 (tapers) and Design 2 (Luneburg lens)**: The smallest geometric
#   feature is the single-mode output waveguide width
#   **$w_{\text{out}} = 0.50\;\mu\text{m} = 500\;\text{nm}$** ($y \in [-0.25, +0.25]\;\mu\text{m}$),
#   which is refined with a transverse override box $h_y \le 500\;\text{nm}/N$.
#
# ### 4.2 Graded non-uniform Cartesian & body-fitted hybrid mesh upgrades (`NonUniformSemilensCircuit`, `subtri_circles_mesh`)
#
# Even with local `mesh_overrides` across Segment 3 ($x \in [4.61, 4.66]\;\mu\text{m}$),
# the baseline `kottke_L3` (`task_6`) mesh keeps a uniform $25.35\;\text{nm}$
# spacing across the entire $12.2\;\mu\text{m} \times 11.1\;\mu\text{m}$ domain
# ($512 \times 459 \times 1 = 235{,}008$ cells, $895.42\;\text{s}$). Two geometry-adapted
# meshing strategies substantially reduce the cell count or conform to curved rims:
#
# 1. **Graded non-uniform Cartesian mesh (`NonUniformSemilensCircuit`, `kottke_nonuniform`, `task_15`)**:
#    - **Axial grading & non-square lead cells**: Retains $\Delta x_{\text{flare}} = 6.25\;\text{nm}$
#      ($8.0$ cells across $L_3 = 50\;\text{nm}$) across Segment 3 and isotropic
#      $\Delta x_{\text{lens}} = 25.0\;\text{nm}$ across the curved dielectric lens
#      of Segments 4–5 ($x \in [4.66, 8.47]\;\mu\text{m}$), while using non-square
#      axial cells $\Delta x_{\text{lead}} = 36.0\;\text{nm}$ ($\text{AR} \approx 1.37$)
#      along the straight input/output leads and slowly varying Segments 1–2 and 6.
#    - **Transverse core & cladding grading**: Uses $\Delta y_{\text{in}} = 22.5\;\text{nm}$
#      ($20$ cells across $w_0 = 450\;\text{nm}$) on $|y| \le 0.30\;\mu\text{m}$,
#      $\Delta y_{\text{core}} = 26.2\;\text{nm}$ across the expanded lens aperture
#      $|y| \in [0.30, 4.56]\;\mu\text{m}$, and grades smoothly (`grading = 1.22`)
#      to $\Delta y_{\text{clad}} = 70.0\;\text{nm}$ in the exterior air cladding
#      $|y| \in [4.56, 5.55]\;\mu\text{m}$.
#    - **Result (`figures/04b_mesh_upgrade_comparison.png`)**: Cuts the 2D grid
#      from $512 \times 459 = 235{,}008$ cells (`kottke_L3`) to
#      $424 \times 407 = 172{,}568$ cells (**$-26.6\%$ cells**) and reduces wall
#      time from $895.42\;\text{s}$ to $598.51\;\text{s}$ (**$-33.2\%$ runtime**)
#      while matching the $1510\text{--}1590\;\text{nm}$ band-mean guided
#      transmission $\langle T_{\text{even}}\rangle = 94.05\%$ to within $0.01\%$
#      of `kottke_L3` ($94.04\%$) and $0.08\%$ of the $40\;\text{cpw}$
#      `kottke_reference` (`task_8`, $94.13\%$, $896{,}103$ cells, $6{,}830.4\;\text{s}$).
# 2. **Body-fitted quadtree + Delaunay hybrid mesh (`wakis.unstructured.subtri_circles_mesh`)**:
#    For circular GRIN devices such as the $R = 5.5\;\mu\text{m}$ Luneburg lens,
#    `subtri_circles_mesh(circles=[(0, 0, 5.5)], ...)` (`figures/00b_taper_and_luneburg_schematics.png`(c))
#    places body-fitted Delaunay triangles along the circular rim while retaining
#    coarse quadtree squares in the interior and cladding.
#
# ## 5. Study tasks & execution modes
#
# Each resolution level is defined by the pair $(\text{cpw}, N)$:
#
# - **Under-resolved demo** `(10, 1)`: $\text{cpw} = 10$ with no mesh override
#   ($0.988$ cells across the $50\;\text{nm}$ Segment 3 flare, `demo=True`).
# - **`L1 = (10, 4)`**: $\text{cpw} = 10$, $N = 4$ cells across $50\;\text{nm}$ ($\Delta x_{\min} = 12.50\;\text{nm}$).
# - **`L2 = (14, 6)`**: $\text{cpw} = 14$, $N = 6$ cells across $50\;\text{nm}$ ($\Delta x_{\min} = 8.333\;\text{nm}$).
# - **`L3 = (20, 8)`**: $\text{cpw} = 20$, $N = 8$ cells across $50\;\text{nm}$ ($\Delta x_{\min} = 6.250\;\text{nm}$).
# - **`R = (40, 16)`**: $\text{cpw} = 40$, $N = 16$ cells across $50\;\text{nm}$ ($\Delta x_{\min} = 3.125\;\text{nm}$).
#
# The precomputed archive in `data/` holds the complete **`--full` study (16 tasks)**:
# Kottke and staircase run across the full three-level resolved ladder `L1 = (10, 4)`,
# `L2 = (14, 6)`, and `L3 = (20, 8)` (plus the `(10, 1)` under-resolved demo pair),
# `kottke_reference` runs at `R = (40, 16)`, `kottke_nonuniform` (`task_15`) runs the
# graded non-uniform Cartesian mesh on Design 3, and all device sweeps (60-shell Luneburg
# lens, linear tapers at $L_t \in \{10, 20, 50, 100\}\;\mu\text{m}$, and the four-profile
# taper sweep across $L_t \in \{0, 5, 10, 20, 50\}\;\mu\text{m}$) run at `L3 = (20, 8)`.
# Every non-demo task (Tasks 2–15) satisfies $\text{cpw} \ge 10$ in the highest-index
# core material and $\ge 4.0$ cells across the smallest geometric feature ($4.0\text{--}21.0$
# cells, zero violations).
#
# ### 5.1 Measured `--full` resolution & timing table (`data/task_0.json`–`task_15.json`)
#
# | Task ID | Task name | $\text{cpw}$ | $N$ | $d_{\min}$ [nm] | $\Delta x_{\text{bg}}$ [nm] | $\Delta x_{\min}$ [nm] | Cells across feature | $N_{\text{cells}}$ ($N_x \times N_y \times 1$) | $N_{\text{steps}}$ | Time [s] | Peak RSS [MB] |
# |---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | 0 | `kottke_underresolved` *(demo)* | 10 | 1 | 50.0 | 50.622 | 50.622 | 0.988 | 62,880 ($262 \times 240$) | 6,811 | 55.72 | 537.7 |
# | 1 | `staircase_underresolved` *(demo)* | 10 | 1 | 50.0 | 50.622 | 50.622 | 0.988 | 62,880 ($262 \times 240$) | 6,811 | 40.89 | 535.3 |
# | 2 | `kottke_L1` | 10 | 4 | 50.0 | 50.378 | 12.500 | 4.000 | 64,560 ($269 \times 240$) | 16,896 | 118.60 | 538.1 |
# | 3 | `staircase_L1` | 10 | 4 | 50.0 | 50.378 | 12.500 | 4.000 | 64,560 ($269 \times 240$) | 16,896 | 120.06 | 539.4 |
# | 4 | `kottke_L2` | 14 | 6 | 50.0 | 36.141 | 8.333 | 6.000 | 120,376 ($367 \times 328$) | 24,798 | 335.34 | 592.7 |
# | 5 | `staircase_L2` | 14 | 6 | 50.0 | 36.141 | 8.333 | 6.000 | 120,376 ($367 \times 328$) | 24,798 | 341.97 | 595.0 |
# | 6 | `kottke_L3` *(primary + DFT + movie)* | 20 | 8 | 50.0 | 25.345 | 6.250 | 8.000 | 235,008 ($512 \times 459$) | 32,837 | 895.42 | 704.6 |
# | 7 | `staircase_L3` | 20 | 8 | 50.0 | 25.345 | 6.250 | 8.000 | 235,008 ($512 \times 459$) | 32,837 | 824.33 | 708.4 |
# | 8 | `kottke_reference` (`R = (40, 16)`) | 40 | 16 | 50.0 | 12.691 | 3.125 | 16.000 | 896,103 ($999 \times 897$) | 64,713 | 6,830.44 | 1,303.6 |
# | 9 | `luneburg_lens` (`L3`, 60 shells) | 20 | 8 | 500.0 | 23.878 | 23.810 | 21.000 | 307,496 ($578 \times 532$) | 14,925 | 665.14 | 534.7 |
# | 10 | `taper_linear_10um` (`L3`) | 20 | 8 | 500.0 | 33.787 | 33.333 | 15.000 | 138,128 ($388 \times 356$) | 10,151 | 191.81 | 353.2 |
# | 11 | `taper_linear_20um` (`L3`) | 20 | 8 | 500.0 | 33.786 | 33.333 | 15.000 | 243,504 ($684 \times 356$) | 15,946 | 495.91 | 463.2 |
# | 12 | `taper_linear_50um` (`L3`) | 20 | 8 | 500.0 | 33.828 | 33.333 | 15.000 | 558,920 ($1570 \times 356$) | 33,228 | 2,532.55 | 787.4 |
# | 13 | `taper_linear_100um` (`L3`) | 20 | 8 | 500.0 | 33.829 | 33.333 | 15.000 | 1,085,088 ($3048 \times 356$) | 62,041 | 10,015.44 | 1,304.1 |
# | 14 | `taper_profile_sweep` (`L3`, 17 runs) | 20 | 8 | 500.0 | 33.828 | 33.333 | 15.000 | 558,920 ($1570 \times 356$) | 266,702 | 8,165.62 | 1,728.7 |
# | 15 | `kottke_nonuniform` *(graded mesh)* | 20 | 8 | 50.0 | 25.390 | 6.250 | 8.000 | 172,568 ($424 \times 407$) | 33,564 | 598.51 | 634.0 |
#
# ### 5.2 Fast local `--quick` mode (13 tasks)
#
# For rapid local verification on a laptop (`--quick`, 13 tasks), Kottke and
# staircase run at `L1 = (10, 4)` and `L2 = (14, 6)` (plus the `(10, 1)`
# under-resolved demonstration pair), `kottke_reference` runs at `L3 = (20, 8)`,
# and the Luneburg lens ($40$ shells) and taper length/profile sweeps run at
# `L2 = (14, 6)` over compact taper lengths ($L_t \in \{5, 10, 20, 35\}\;\mu\text{m}$
# for single tapers and $L_t \in \{0, 5, 10, 20\}\;\mu\text{m}$ for the profile sweep,
# with $n_{\text{pml}} = 8$ and energy decay threshold $3\text{--}4 \times 10^{-3}$):
#
# | Task ID | Task name (`--quick`) | $\text{cpw}$ | $N$ | $d_{\min}$ [nm] | $\Delta x_{\text{bg}}$ [nm] | $\Delta x_{\min}$ [nm] | Cells across feature | $N_{\text{cells}}$ ($N_x \times N_y \times 1$) | $N_{\text{steps}}$ | Time [s] |
# |---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | 0–1 | `{kottke,staircase}_underresolved` *(demo)* | 10 | 1 | 50.0 | 50.62 | 50.62 | 0.99 | 60,888 ($258 \times 236$) | 6,733 | 55.8 / 55.6 |
# | 2–3 | `{kottke,staircase}_L1` | 10 | 4 | 50.0 | 50.38 | 12.50 | 4.00 | 62,540 ($265 \times 236$) | 16,705 | 137.1 / 133.6 |
# | 4–5 | `{kottke,staircase}_L2` *(4 has DFT)* | 14 | 6 | 50.0 | 36.14 | 8.33 | 6.00 | 117,612 ($363 \times 324$) | 24,593 | 477.3 / 367.8 |
# | 6 | `kottke_reference` (`L3`) | 20 | 8 | 50.0 | 25.34 | 6.25 | 8.00 | 231,140 ($508 \times 455$) | 32,645 | 941.6 |
# | 7 | `luneburg_lens` (`L2`, 40 shells) | 14 | 6 | 500.0 | 34.10 | 33.33 | 15.00 | 153,032 ($407 \times 376$) | 8,901 | 225.5 |
# | 8–11 | `taper_linear_{5,10,20,35}um` (`L2`) | 14 | 6 | 500.0 | 48.05–48.32 | 45.45 | 11.00 | 43,092–199,332 | 4,951–13,601 | 37.8–415.9 |
# | 12 | `taper_profile_sweep` (`L2`, 16 runs) | 14 | 6 | 500.0 | 48.28 | 45.45 | 11.00 | 121,212 ($481 \times 252$) | 53,410 | 519.5 |



# %%
def get_task_table(quick=False):
    """Return the study task table (`--quick`: 13 tasks at L1/L2/L3; `--full`: 15 tasks at L1/L2/L3/R)."""
    tasks = []

    # 1. Under-resolved demonstration pair (cpw=10, no override -> 0.99 cells across 50 nm Segment 3)
    for avg in ("kottke", "staircase"):
        tasks.append(
            {
                "index": len(tasks),
                "name": f"{avg}_underresolved",
                "kind": "semilens",
                "level": "underresolved",
                "averaging": avg,
                "cpw": 10,
                "n_feature": 1,
                "demo": True,
                "n_pml": 8 if quick else 10,
                "energy_decay": 4e-3 if quick else 1e-4,
                "collect_fields": False,
                "record_movie": False,
                "movie_frames": 0,
                "description": (
                    f"Design 3 Semi-lens expander ({avg}, under-resolved demo, cpw=10, N=1, ~1 cell across 50 nm Seg 3)"
                ),
            }
        )

    # 2. Resolved ladder: L1=(10,4), L2=(14,6) in --quick; L1=(10,4), L2=(14,6), L3=(20,8) in --full
    ladder = [("L1", 10, 4), ("L2", 14, 6)] if quick else [("L1", 10, 4), ("L2", 14, 6), ("L3", 20, 8)]
    primary_level = "L2" if quick else "L3"
    for lvl, cpw, n_feat in ladder:
        for avg in ("kottke", "staircase"):
            is_primary = lvl == primary_level and avg == "kottke"
            tasks.append(
                {
                    "index": len(tasks),
                    "name": f"{avg}_{lvl}",
                    "kind": "semilens",
                    "level": lvl,
                    "averaging": avg,
                    "cpw": cpw,
                    "n_feature": n_feat,
                    "demo": False,
                    "n_pml": 8 if quick else 10,
                    "energy_decay": 4e-3 if quick else 1e-4,
                    "collect_fields": is_primary,
                    "record_movie": is_primary,
                    "movie_frames": 30 if quick else 48,
                    "description": (
                        f"Design 3 Semi-lens expander ({avg}, {lvl}=({cpw},{n_feat})"
                        + (
                            ", DFT fields + movie + TE0..TE8 modes)"
                            if is_primary
                            else ")"
                        )
                    ),
                }
            )

    # 3. Independent Kottke reference: L3=(20,8) in --quick, R=(40,16) in --full
    ref_cpw, ref_n = (20, 8) if quick else (40, 16)
    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_reference",
            "kind": "semilens",
            "level": "reference",
            "averaging": "kottke",
            "cpw": ref_cpw,
            "n_feature": ref_n,
            "demo": False,
            "n_pml": 8 if quick else 10,
            "energy_decay": 4e-3 if quick else 1e-4,
            "collect_fields": False,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"Design 3 Semi-lens expander (kottke, reference=({ref_cpw},{ref_n}))"
            ),
        }
    )

    # 4. Device sweeps at L2=(14,6) in --quick and L3=(20,8) in --full
    dev_lvl, dev_cpw, dev_n = ("L2", 14, 6) if quick else ("L3", 20, 8)
    tasks.append(
        {
            "index": len(tasks),
            "name": "luneburg_lens",
            "kind": "luneburg",
            "level": dev_lvl,
            "averaging": "kottke",
            "cpw": dev_cpw,
            "n_feature": dev_n,
            "demo": False,
            "n_shells": 40 if quick else 60,
            "n_pml": 8 if quick else 10,
            "energy_decay": 3e-3 if quick else 1e-4,
            "collect_fields": True,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"Design 2 Graded Luneburg lens D=11 um (kottke, {dev_lvl}=({dev_cpw},{dev_n}), 1D FIT cal + DFT)"
            ),
        }
    )

    taper_lengths = (5, 10, 20, 35) if quick else (10, 20, 50, 100)
    for length_um in taper_lengths:
        tasks.append(
            {
                "index": len(tasks),
                "name": f"taper_linear_{length_um}um",
                "kind": "taper_single",
                "length_um": float(length_um),
                "profile": "linear",
                "level": dev_lvl,
                "averaging": "kottke",
                "cpw": dev_cpw,
                "n_feature": dev_n,
                "demo": False,
                "n_pml": 8 if quick else 10,
                "energy_decay": 3e-3 if quick else 1e-4,
                "collect_fields": True,
                "record_movie": False,
                "movie_frames": 0,
                "description": (
                    f"Design 1 Linear taper L={length_um} um (kottke, {dev_lvl}=({dev_cpw},{dev_n}), DFT fields)"
                ),
            }
        )

    sweep_lengths = [0.0, 5.0, 10.0, 20.0] if quick else [0.0, 5.0, 10.0, 20.0, 50.0]
    tasks.append(
        {
            "index": len(tasks),
            "name": "taper_profile_sweep",
            "kind": "taper_sweep",
            "level": dev_lvl,
            "averaging": "kottke",
            "cpw": dev_cpw,
            "n_feature": dev_n,
            "demo": False,
            "lengths_um": sweep_lengths,
            "n_pml": 8 if quick else 10,
            "energy_decay": 4e-3 if quick else 2e-4,
            "collect_fields": False,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"Design 1 Taper profile & length sweep (linear/parabolic/exp/cosine, {dev_lvl}=({dev_cpw},{dev_n}))"
            ),
        }
    )

    # 5. Graded non-uniform Cartesian mesh upgrade for Design 3 Semi-lens expander
    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_nonuniform",
            "kind": "semilens",
            "level": dev_lvl,
            "averaging": "kottke",
            "cpw": dev_cpw,
            "n_feature": dev_n,
            "demo": False,
            "nonuniform": True,
            "n_pml": 8 if quick else 10,
            "energy_decay": 4e-3 if quick else 1e-4,
            "collect_fields": False,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"Design 3 Semi-lens expander (kottke graded mesh NonUniformSemilensCircuit, {dev_lvl}=({dev_cpw},{dev_n}))"
            ),
        }
    )
    return tasks


def _populate_field_and_port_payload(payload, circuit, solve_ports=False):
    """Store optional DFT field maps, port eigenmodes, movie frames, and flux balance."""
    if solve_ports:
        modes_in = circuit.solve_port_modes("in", frequencies=c_light / WL_REF, n_modes=1)
        modes_out = circuit.solve_port_modes("out", frequencies=c_light / WL_REF, n_modes=9)
        y_in = modes_in[0].coords["Eb"][0]
        ey_in = modes_in[0].e["b"][:, 0]
        m_in = np.abs(y_in) <= 0.9e-6
        y_out = modes_out[0].coords["Eb"][0]
        ey_out0 = modes_out[0].e["b"][:, 0]
        ey_out2 = modes_out[2].e["b"][:, 0]
        ey_out4 = modes_out[4].e["b"][:, 0]
        m_out = np.abs(y_out) <= 5.2e-6
        payload["port_y_in"] = y_in[m_in].astype(np.float32)
        payload["port_ey_in"] = ey_in[m_in].astype(np.complex64)
        payload["port_neff_in"] = np.float64(np.real(modes_in[0].neff))
        payload["port_y_out"] = y_out[m_out].astype(np.float32)
        payload["port_ey_out"] = ey_out0[m_out].astype(np.complex64)
        payload["port_ey_out_te2"] = ey_out2[m_out].astype(np.complex64)
        payload["port_ey_out_te4"] = ey_out4[m_out].astype(np.complex64)
        payload["port_neff_out"] = np.float64(np.real(modes_out[0].neff))
        payload["port_neff_out_te2"] = np.float64(np.real(modes_out[2].neff))
        payload["port_neff_out_te4"] = np.float64(np.real(modes_out[4].neff))

        wl_disp = np.linspace(WL_MIN, WL_MAX, 11)
        f_disp = c_light / wl_disp
        disp_modes = circuit.solve_port_modes("in", frequencies=f_disp, n_modes=1)
        neff_disp = np.array([float(np.real(m[0].neff)) for m in disp_modes])
        dneff_dl = np.gradient(neff_disp, wl_disp)
        ng_disp = neff_disp - wl_disp * dneff_dl
        payload["disp_wavelengths"] = wl_disp.astype(np.float64)
        payload["disp_neff"] = neff_disp.astype(np.float64)
        payload["disp_ng"] = ng_disp.astype(np.float64)

    if WL_REF in circuit.dft_fields:
        fmap = circuit.dft_fields[WL_REF]
        payload["dft_x"] = fmap["x"]
        payload["dft_y"] = fmap["y"]
        payload["dft_Ey"] = fmap["Ey"]
        payload["dft_E2"] = fmap["E2"]

    if circuit.movie_data is not None:
        payload["movie_x"] = circuit.movie_data["x"]
        payload["movie_y"] = circuit.movie_data["y"]
        payload["movie_times"] = circuit.movie_data["times"]
        payload["movie_frames"] = circuit.movie_data["frames"]
        payload["movie_peak"] = circuit.movie_data["peak"]

    if circuit.port_time_trace is not None:
        payload["trace_t"] = circuit.port_time_trace["t"]
        payload["trace_Ey"] = circuit.port_time_trace["Ey"]

    if circuit.flux_balance is not None:
        payload["flux_east"] = circuit.flux_balance["east"].astype(np.float64)
        payload["flux_west"] = circuit.flux_balance["west"].astype(np.float64)
        payload["flux_lateral"] = circuit.flux_balance["lateral"].astype(np.float64)
        payload["flux_total_box"] = circuit.flux_balance["total_box"].astype(np.float64)


def run_single_task(task_cfg, out_dir):
    """
    Execute one 2D EIM mode-size converter study task and write ``task_<I>.npz``
    and ``task_<I>.json`` into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task_cfg["index"]
    kind = task_cfg["kind"]
    n_feat = int(task_cfg.get("n_feature", 4))
    is_demo = bool(task_cfg.get("demo", False))
    t0 = time.perf_counter()
    use_gpu = os.environ.get("WAKIS_USE_GPU", "0") == "1"

    payload = {
        "task_index": np.int32(idx),
        "name": np.array(task_cfg["name"]),
        "kind": np.array(kind),
        "level": np.array(task_cfg["level"]),
        "averaging": np.array(task_cfg["averaging"]),
        "cpw": np.float32(task_cfg["cpw"]),
        "n_feature": np.int32(n_feat),
        "demo": np.bool_(is_demo),
    }

    if kind == "semilens":
        structures, ports, domain_2d = build_semilens_expander()
        is_nu = bool(task_cfg.get("nonuniform", False))
        overrides = [] if is_nu else semilens_mesh_overrides(n_feat)
        res = run_converter_simulation(
            structures,
            ports,
            domain_2d,
            n_clad=N_AIR,
            n_sub=N_SIO2,
            averaging=task_cfg["averaging"],
            cpw=task_cfg["cpw"],
            n_pml=task_cfg["n_pml"],
            energy_decay=task_cfg["energy_decay"],
            mesh_overrides=overrides,
            collect_fields=task_cfg["collect_fields"],
            record_movie=task_cfg["record_movie"],
            movie_frames=task_cfg["movie_frames"],
            multimode_indices=(0, 2, 4, 6, 8),
            use_gpu=use_gpu,
            nonuniform=is_nu,
            n_feature=n_feat,
        )
        circuit = res["circuit"]
        n_steps = int(res["n_steps"])
        res_metrics = measure_grid_resolution(
            circuit.grid,
            min_feature_m=MIN_FEATURE_SEMILENS,
            feature_bounds=(SEMILENS_SEG3_X_LO, SEMILENS_SEG3_X_HI),
            feature_axis=0,
            core_x_bounds=(0.0, 4.0e-6),
        )
        if is_nu:
            res_metrics["dx_background_nm"] = round(
                float(WL_MIN / (circuit.n_eff_slab * float(task_cfg["cpw"]))) * 1e9, 3
            )
        payload["n_eff_slab"] = np.float64(circuit.n_eff_slab)
        payload["frequencies"] = circuit.frequencies.astype(np.float64)
        payload["wavelengths"] = circuit.wavelengths.astype(np.float64)
        payload["S11"] = res["s_matrix"][("in", "in")].astype(np.complex128)
        payload["S21"] = res["s_matrix"][("out", "in")].astype(np.complex128)
        payload["T_flux_port"] = res["t_flux_port"].astype(np.float64)
        for m_idx, s_arr in res["modal_s21"].items():
            payload[f"S21_TE{m_idx}"] = s_arr.astype(np.complex128)
        if task_cfg["collect_fields"]:
            _populate_field_and_port_payload(payload, circuit, solve_ports=True)

    elif kind == "luneburg":
        structures, ports, domain_2d, _ = build_luneburg_converter(
            n_shells=task_cfg.get("n_shells", 40)
        )
        overrides = waveguide_width_mesh_overrides(n_feat, width=W_OUT_TAPER)
        cal = calibrate_luneburg_profile(solve_1d_fit=True, cpw_1d=45)
        res = run_converter_simulation(
            structures,
            ports,
            domain_2d,
            n_clad=N_SIO2,
            n_sub=N_AIR,
            averaging=task_cfg["averaging"],
            cpw=task_cfg["cpw"],
            n_pml=task_cfg["n_pml"],
            energy_decay=task_cfg["energy_decay"],
            mesh_overrides=overrides,
            collect_fields=task_cfg["collect_fields"],
            record_movie=False,
            use_gpu=use_gpu,
        )
        circuit = res["circuit"]
        n_steps = int(res["n_steps"])
        res_metrics = measure_grid_resolution(
            circuit.grid,
            min_feature_m=MIN_FEATURE_LUNEBURG,
            feature_bounds=(-0.5 * W_OUT_TAPER, 0.5 * W_OUT_TAPER),
            feature_axis=1,
            core_x_bounds=(-3.0e-6, 3.0e-6),
        )
        payload["n_eff_slab"] = np.float64(cal["n_edge"])
        payload["frequencies"] = circuit.frequencies.astype(np.float64)
        payload["wavelengths"] = circuit.wavelengths.astype(np.float64)
        payload["S11"] = res["s_matrix"][("in", "in")].astype(np.complex128)
        payload["S21"] = res["s_matrix"][("out", "in")].astype(np.complex128)
        payload["T_flux_port"] = res["t_flux_port"].astype(np.float64)
        for k, v in cal.items():
            payload[f"cal_{k}"] = np.asarray(v, dtype=np.float64)
        if task_cfg["collect_fields"]:
            _populate_field_and_port_payload(payload, circuit, solve_ports=False)

    elif kind == "taper_single":
        length_um = float(task_cfg["length_um"])
        structures, ports, domain_2d = build_taper_converter(
            length_taper=length_um * 1e-6, profile=task_cfg.get("profile", "linear")
        )
        overrides = waveguide_width_mesh_overrides(n_feat, width=W_OUT_TAPER)
        res = run_converter_simulation(
            structures,
            ports,
            domain_2d,
            n_clad=N_SIO2,
            n_sub=N_AIR,
            averaging=task_cfg["averaging"],
            cpw=task_cfg["cpw"],
            n_pml=task_cfg["n_pml"],
            energy_decay=task_cfg["energy_decay"],
            mesh_overrides=overrides,
            collect_fields=task_cfg["collect_fields"],
            record_movie=False,
            use_gpu=use_gpu,
        )
        circuit = res["circuit"]
        n_steps = int(res["n_steps"])
        res_metrics = measure_grid_resolution(
            circuit.grid,
            min_feature_m=MIN_FEATURE_TAPER,
            feature_bounds=(-0.5 * W_OUT_TAPER, 0.5 * W_OUT_TAPER),
            feature_axis=1,
            core_x_bounds=None,
        )
        payload["length_um"] = np.float64(length_um)
        payload["n_eff_slab"] = np.float64(circuit.n_eff_slab)
        payload["frequencies"] = circuit.frequencies.astype(np.float64)
        payload["wavelengths"] = circuit.wavelengths.astype(np.float64)
        payload["S11"] = res["s_matrix"][("in", "in")].astype(np.complex128)
        payload["S21"] = res["s_matrix"][("out", "in")].astype(np.complex128)
        payload["T_flux_port"] = res["t_flux_port"].astype(np.float64)
        if task_cfg["collect_fields"]:
            _populate_field_and_port_payload(payload, circuit, solve_ports=False)

    elif kind == "taper_sweep":
        lengths_um = np.asarray(
            task_cfg.get("lengths_um", [0.0, 5.0, 10.0, 20.0]), dtype=np.float64
        )
        profiles = ("linear", "parabolic", "exponential", "cosine")
        overrides = waveguide_width_mesh_overrides(n_feat, width=W_OUT_TAPER)
        payload["lengths_um"] = lengths_um
        ov = compute_abrupt_step_overlap()
        payload["overlap_y"] = ov["y"].astype(np.float32)
        payload["overlap_e_in"] = ov["e_in"].astype(np.float32)
        payload["overlap_e_out"] = ov["e_out"].astype(np.float32)
        payload["overlap_sq"] = np.float64(ov["overlap_sq"])
        payload["em_overlap"] = np.float64(ov["em_overlap"])
        payload["fresnel_transmission"] = np.float64(ov["fresnel_transmission"])

        # Run L=0 abrupt step once and reuse for all profiles at L=0
        s0, p0, d0 = build_taper_converter(length_taper=0.0, profile="linear")
        r0 = run_converter_simulation(
            s0,
            p0,
            d0,
            n_clad=N_SIO2,
            n_sub=N_AIR,
            averaging=task_cfg["averaging"],
            cpw=task_cfg["cpw"],
            n_pml=task_cfg["n_pml"],
            n_freqs=21,
            energy_decay=task_cfg["energy_decay"],
            mesh_overrides=overrides,
            collect_fields=False,
            use_gpu=use_gpu,
        )
        i_ref0 = int(np.argmin(np.abs(r0["circuit"].wavelengths - WL_REF)))
        t0_1550 = float(np.abs(r0["s_matrix"][("out", "in")][i_ref0]) ** 2)
        r0_1550 = float(np.abs(r0["s_matrix"][("in", "in")][i_ref0]) ** 2)
        n_steps = int(r0["n_steps"])
        res_metrics = measure_grid_resolution(
            r0["circuit"].grid,
            min_feature_m=MIN_FEATURE_TAPER,
            feature_bounds=(-0.5 * W_OUT_TAPER, 0.5 * W_OUT_TAPER),
            feature_axis=1,
            core_x_bounds=None,
        )

        # Check if any linear taper single tasks already wrote results at this cpw
        cached_linear = {}
        for p_existing in out_dir.glob("task_*.npz"):
            try:
                with np.load(p_existing, allow_pickle=False) as d_ex:
                    if (
                        str(d_ex.get("kind", "")) == "taper_single"
                        and abs(float(d_ex["cpw"]) - float(task_cfg["cpw"])) < 1e-3
                    ):
                        l_ex = round(float(d_ex["length_um"]), 3)
                        i_ex = int(np.argmin(np.abs(d_ex["wavelengths"] - WL_REF)))
                        cached_linear[l_ex] = (
                            float(np.abs(d_ex["S21"][i_ex]) ** 2),
                            float(np.abs(d_ex["S11"][i_ex]) ** 2),
                        )
            except Exception:
                pass

        for prof in profiles:
            t_arr = [t0_1550]
            r_arr = [r0_1550]
            for L_u in lengths_um[1:]:
                l_key = round(float(L_u), 3)
                if prof == "linear" and l_key in cached_linear:
                    t_v, r_v = cached_linear[l_key]
                    t_arr.append(t_v)
                    r_arr.append(r_v)
                    continue
                st, pt, dm = build_taper_converter(
                    length_taper=float(L_u) * 1e-6, profile=prof
                )
                rs = run_converter_simulation(
                    st,
                    pt,
                    dm,
                    n_clad=N_SIO2,
                    n_sub=N_AIR,
                    averaging=task_cfg["averaging"],
                    cpw=task_cfg["cpw"],
                    n_pml=task_cfg["n_pml"],
                    n_freqs=21,
                    energy_decay=task_cfg["energy_decay"],
                    mesh_overrides=overrides,
                    collect_fields=False,
                    use_gpu=use_gpu,
                )
                n_steps += int(rs["n_steps"])
                res_metrics = measure_grid_resolution(
                    rs["circuit"].grid,
                    min_feature_m=MIN_FEATURE_TAPER,
                    feature_bounds=(-0.5 * W_OUT_TAPER, 0.5 * W_OUT_TAPER),
                    feature_axis=1,
                    core_x_bounds=None,
                )
                i_c = int(np.argmin(np.abs(rs["circuit"].wavelengths - WL_REF)))
                t_arr.append(float(np.abs(rs["s_matrix"][("out", "in")][i_c]) ** 2))
                r_arr.append(float(np.abs(rs["s_matrix"][("in", "in")][i_c]) ** 2))
            payload[f"T_{prof}"] = np.asarray(t_arr, dtype=np.float64)
            payload[f"R_{prof}"] = np.asarray(r_arr, dtype=np.float64)
    else:
        raise ValueError(f"Unknown task kind {kind!r}")

    payload["min_feature_nm"] = np.float64(res_metrics["min_feature_nm"])
    payload["dx_background_nm"] = np.float64(res_metrics["dx_background_nm"])
    payload["dx_min_nm"] = np.float64(res_metrics["dx_min_nm"])
    payload["cells_across_feature"] = np.float64(res_metrics["cells_across_feature"])
    payload["n_cells"] = np.int64(res_metrics["n_cells"])
    payload["n_steps"] = np.int64(n_steps)

    npz_path = out_dir / f"task_{idx}.npz"
    np.savez_compressed(npz_path, **payload)

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    meta = {
        "task_index": idx,
        "name": task_cfg["name"],
        "kind": kind,
        "level": task_cfg["level"],
        "averaging": task_cfg["averaging"],
        "cpw": task_cfg["cpw"],
        "n_feature": n_feat,
        "demo": is_demo,
        "min_feature_nm": res_metrics["min_feature_nm"],
        "dx_background_nm": res_metrics["dx_background_nm"],
        "dx_min_nm": res_metrics["dx_min_nm"],
        "cells_across_feature": res_metrics["cells_across_feature"],
        "grid_shape": res_metrics["grid_shape"],
        "n_cells": res_metrics["n_cells"],
        "n_steps": n_steps,
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mb": round(rss_mb, 2),
        "npz_bytes": int(npz_path.stat().st_size),
    }
    (out_dir / f"task_{idx}.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[task_{idx} {task_cfg['name']}] cpw={task_cfg['cpw']} N={n_feat} "
        f"min_feature={meta['min_feature_nm']:.1f}nm dx_bg={meta['dx_background_nm']:.2f}nm "
        f"dx_min={meta['dx_min_nm']:.2f}nm cells_across_feature={meta['cells_across_feature']:.2f} "
        f"n_cells={meta['n_cells']} n_steps={meta['n_steps']} "
        f"time={elapsed:.2f}s rss={rss_mb:.1f}MB"
    )
    return meta


# %% [markdown]
# ## 6. Generating the complete figure suite
#
# Running `--plot` (or completing `--quick` / `--full`) reads `data/task_*.npz`
# and generates the complete figure suite in `figures/`, starting with the
# labelled setup schematics (`00a_setup_geometry.png`,
# `00b_setup_mesh_and_excitation.png`, `00_setup_table.md`, and
# `00b_taper_and_luneburg_schematics.png`):
#
# 0a. `00a_setup_geometry.png` — Labelled setup geometry schematic for the 6-segment semi-lens beam expander (Design 3).
# 0b. `00b_setup_mesh_and_excitation.png` — Computational mesh with $1\times$ zoom inset on the $50\;\text{nm}$ Segment 3 flare, spatial wave packet, and excitation pulse/spectrum for Design 3.
# 0c. `00b_taper_and_luneburg_schematics.png` — Labelled setup schematics for Design 1 (adiabatic linear taper) and Design 2 (graded Luneburg lens), plus a body-fitted quadtree + Delaunay hybrid mesh view (`wakis.unstructured.subtri_circles_mesh`) on the circular $R = 5.5\;\mu\text{m}$ Luneburg lens rim.
# 1. `01_converter_layouts_overview.png` — Side-by-side top-view layouts of all three mode-size converter concepts.
# 2. `02_eim_vertical_stack.png` — Vertical SOI slab stack, $\text{TE}_0$ mode profile, and $n_{\text{eff,slab}}(\lambda)$.
# 3. `03_luneburg_calibration_and_profile.png` — 1D FIT vs analytical slab $n_{\text{eff}}(t_{\text{Si}})$ cubic polynomial calibration and radial Luneburg profiles $n_{\text{eff}}(r), t_{\text{lens}}(r)$.
# 4. `04_permittivity_mesh_zoom.png` — Zoomed in-plane permittivity map with Yee mesh lines on the $50\;\text{nm}$ Segment 3 flare: `L1 = (10, 4)` staircase ($4$ cells, $\Delta x_{\min} = 12.50\;\text{nm}$) vs `L1 = (10, 4)` Kottke ($4$ cells) vs `Reference = (40, 16)` Kottke ($16$ cells, $\Delta x_{\min} = 3.125\;\text{nm}$).
# 4b. `04b_mesh_upgrade_comparison.png` — Equal-accuracy comparison of the baseline `kottke_L3` mesh ($512 \times 459 = 235{,}008$ cells, $895.42\;\text{s}$) and the graded non-uniform Cartesian `kottke_nonuniform` mesh ($424 \times 407 = 172{,}568$ cells, $598.51\;\text{s}$, $-26.6\%$ cells, $-33.2\%$ wall time) against `kottke_reference`.
# 5. `05_port_mode_profiles.png` — Single-mode ($0.45\;\mu\text{m}$) input vs wide ($9.0\;\mu\text{m}$) multi-mode output eigenmode profiles.
# 6. `06_port_neff_dispersion.png` — Channel waveguide effective index $n_{\text{eff}}(\lambda)$ and group index $n_g(\lambda)$.
# 7. `07_taper_length_spectra_and_profiles.png` — Broadband insertion loss spectra in dB for $L_t = 10, 20, 50, 100\;\mu\text{m}$ (`--full`) or $5, 10, 20, 35\;\mu\text{m}$ (`--quick`) linear tapers and taper width profiles.
# 8. `08_taper_ey_field_comparison.png` and `08b_taper_ey_long_comparison.png` — Steady-state $\mathrm{Re}(E_y)$ wave-front maps at $1550\;\text{nm}$ with labelled geometry overlays across short ($L_t = 10, 20\;\mu\text{m}$) and long ($L_t = 50, 100\;\mu\text{m}$) linear tapers.
# 9. `09_luneburg_focusing_and_comparison.png` — Graded Luneburg lens $\mathrm{Re}(E_y)$ focusing map (with geometry and boundary overlays) and spectral comparison against $L_t = 10, 20, 50, 100\;\mu\text{m}$ linear tapers.
# 10. `10_semilens_field_maps_1550.png` — 6-segment semi-lens steady-state logarithmic intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ and $\mathrm{Re}(E_y)$ wave-front collimation at $1550\;\text{nm}$ with material and boundary overlays.
# 11. `11_semilens_spectra_and_modal_share.png` — Semi-lens total insertion loss, return loss, and linear modal power decomposition ($\text{TE}_0, \text{TE}_2, \text{TE}_4, \text{TE}_6, \text{TE}_8$) across $1505\text{--}1595\;\text{nm}$.
# 12. `12_taper_profiles_and_overlap.png` — Transmission vs taper length $L_t \in [0, 100]\;\mu\text{m}$ for linear, parabolic, exponential, and cosine tapers compared against the analytical abrupt-step mode overlap $|\langle E_{\text{in}}, E_{\text{out}}\rangle|^2$, Luneburg lens, and semi-lens.
# 13. `13_kottke_vs_staircase_convergence.png` — Subpixel convergence of Kottke vs staircasing (`L1`, `L2`, `L3`, plus under-resolved `(10, 1)` demo) against the independent `kottke_reference` run (`R = (40, 16)`), labeled by both background cpw and cells across the $50\;\text{nm}$ smallest feature.
# 14. `14_resolution_spectra_and_time_trace.png` — Multi-resolution spectral comparison across `underresolved`, `L1`, `L2`, `L3`, and `R = (40, 16)` plus the output port time-domain ring-down trace.
# 15. `15_pulse_propagation.gif` — Animated time-domain pulse expansion and collimation through the 6-segment semi-lens beam expander with labelled geometry and CPML boundary overlays on every frame.
#
# ### 6.1 Summary of quantitative results (`--full` study)
#
# - **Design 3 — 6-segment semi-lens beam expander ($L_{\text{tot}} = 9.50\;\mu\text{m}$, Abbaslou *et al.* 2017 [2])**:
#   - At the fine reference resolution `R = (40, 16)` (`task_8`, $16.0$ cells across the $50\;\text{nm}$ Segment 3 flare, $\Delta x_{\min} = 3.125\;\text{nm}$, $\Delta x_{\text{bg}} = 12.691\;\text{nm}$), the expander achieves total output port flux transmission $T_{\text{flux}}(1550\;\text{nm}) = 95.18\%$ ($-0.215\;\text{dB}$), guided even-mode sum $T_{\text{even}}(1550\;\text{nm}) = 94.22\%$ ($-0.258\;\text{dB}$), and fundamental mode transmission $T_{\text{TE}_0}(1550\;\text{nm}) = 75.88\%$ ($-1.199\;\text{dB}$; $1505\text{--}1595\;\text{nm}$ band mean $76.87\%$), corresponding to an **$80.53\%$ $\text{TE}_0$ share** of guided power ($\text{TE}_2 = 17.64\%$ of input / $18.72\%$ of $T_{\text{even}}$; $\text{TE}_4 = 0.10\%$; $\text{TE}_6 = 0.50\%$; $\text{TE}_8 = 0.11\%$) and return loss $R(1550\;\text{nm}) = 0.0955\%$ ($-30.20\;\text{dB}$).
#   - At the accepted primary resolution `L3 = (20, 8)` (`task_6`, $8.0$ cells across $50\;\text{nm}$, $\Delta x_{\min} = 6.25\;\text{nm}$, $\Delta x_{\text{bg}} = 25.345\;\text{nm}$, $235{,}008$ cells, $895.42\;\text{s}$), Kottke yields $T_{\text{flux}}(1550\;\text{nm}) = 95.12\%$ ($-0.217\;\text{dB}$), $T_{\text{even}} = 94.17\%$ ($-0.261\;\text{dB}$; $1510\text{--}1590\;\text{nm}$ band mean $94.04\%$), $T_{\text{TE}_0} = 76.99\%$ ($-1.136\;\text{dB}$; band mean $77.75\%$), and an **$81.76\%$ $\text{TE}_0$ share** ($\text{TE}_2 = 16.36\%$, $\text{TE}_4 = 0.21\%$, $\text{TE}_6 = 0.55\%$, $\text{TE}_8 = 0.06\%$, $R = 0.0586\% = -32.32\;\text{dB}$). Closed-box Poynting flux accounting at $1550\;\text{nm}$ closes to $99.85\%$ of launched power ($95.14\%$ eastern output, $2.85\%$ lateral radiation into north/south PMLs, $1.86\%$ western backscatter).
#   - The graded non-uniform Cartesian mesh `kottke_nonuniform` (`task_15`, $424 \times 407 \times 1 = 172{,}568$ cells, $-26.6\%$ cells, $598.51\;\text{s}$, $-33.2\%$ wall time vs `kottke_L3`) achieves $T_{\text{flux}}(1550\;\text{nm}) = 94.92\%$ ($-0.226\;\text{dB}$), $T_{\text{even}}(1550\;\text{nm}) = 93.95\%$ ($1510\text{--}1590\;\text{nm}$ band mean $\langle T_{\text{even}}\rangle = 94.05\%$, within $0.01\%$ of `kottke_L3` and $0.08\%$ of `kottke_reference`), and $T_{\text{TE}_0}(1550\;\text{nm}) = 77.48\%$ ($\text{TE}_0$ share $82.47\%$).
#   - Across the resolution ladder (`L1 = (10, 4)` $\to$ `L2 = (14, 6)` $\to$ `L3 = (20, 8)` $\to$ `R = (40, 16)`), the 2D EIM $\text{TE}_0$ modal share at $1550\;\text{nm}$ converges monotonically ($87.29\% \to 83.54\% \to 81.76\% \to 80.53\%$) directly onto the published 3D FDTD value of Abbaslou *et al.* (2017) ($\approx 80\%$ $\text{TE}_0$ share, $\sim 95\%$ total transmission).
# - **Design 2 — Graded Luneburg lens ($D_{\text{lens}} = 11.0\;\mu\text{m}$ at `L3 = (20, 8)`, Badri & Gilarlue 2019 [1])**:
#   With $60$ concentric GRIN shells and $21.0$ cells across the $0.50\;\mu\text{m}$ output waveguide (`task_9`, $\Delta x_{\min} = 23.81\;\text{nm}$, $\Delta x_{\text{bg}} = 23.878\;\text{nm}$), the Luneburg lens achieves **$T_{\text{TE}_0}(1550\;\text{nm}) = 96.48\%$ ($-0.155\;\text{dB}$)** ($1505\text{--}1595\;\text{nm}$ band mean $96.49\%$, min $96.08\%$, max $96.84\%$), $T_{\text{flux}}(1550\;\text{nm}) = 98.15\%$ ($-0.081\;\text{dB}$), and $R(1550\;\text{nm}) = 0.0212\%$ ($-36.75\;\text{dB}$). In an $11.0\;\mu\text{m}$ footprint, it surpasses a $50\;\mu\text{m}$ linear taper ($92.70\%$, $-0.329\;\text{dB}$) and approaches a $100\;\mu\text{m}$ adiabatic linear taper ($99.34\%$ at $1550\;\text{nm}$, $97.07\%$ band mean).
# - **Design 1 — Linear and non-linear tapers (`L3 = (20, 8)`, $15.0$ cells across $w_{\text{out}} = 0.50\;\mu\text{m}$)**:
#   - At $L_t = 0\;\mu\text{m}$ (abrupt $10.0 \to 0.50\;\mu\text{m}$ butt-coupled step, `task_14`), the 2D FIT transmission is $T_{\text{TE}_0}(1550\;\text{nm}) = 19.04\%$ ($-7.20\;\text{dB}$, $R = 5.22\%$), in close agreement with the analytical 2D EIM electromagnetic Poynting overlap $\eta_{\text{EM}} = 21.00\%$ ($-6.78\;\text{dB}$; scalar overlap $|\langle E_y\rangle|^2 = 22.84\%$, Fresnel-weighted scalar overlap $22.69\%$).
#   - Across the dedicated 51-frequency linear taper runs (`tasks 10–13`, energy decay $10^{-4}$), $T_{\text{TE}_0}(1550\;\text{nm})$ rises monotonically from $26.87\%$ ($-5.71\;\text{dB}$, band mean $26.86\%$) at $L_t = 10\;\mu\text{m}$ to $57.76\%$ ($-2.38\;\text{dB}$, band mean $56.40\%$) at $L_t = 20\;\mu\text{m}$, $92.70\%$ ($-0.33\;\text{dB}$, band mean $88.75\%$) at $L_t = 50\;\mu\text{m}$, and **$99.34\%$ ($-0.03\;\text{dB}$, band mean $97.07\%$, $T_{\text{flux}} = 97.25\%$)** at $L_t = 100\;\mu\text{m}$.
#   - Across the 4-profile sweep (`task_14`, $L_t \in \{0, 5, 10, 20, 50\}\;\mu\text{m}$, 21 frequencies, energy decay $2 \times 10^{-4}$), convex profiles that narrow rapidly where the waveguide is wide ($w \sim 10\;\mu\text{m}$, long beat length $L_b \propto w^2$) and flatten near the narrow output ($w \sim 0.5\;\mu\text{m}$) outperform the linear taper at short non-adiabatic lengths ($L_t = 5\text{--}10\;\mu\text{m}$): at $L_t = 5\;\mu\text{m}$, exponential ($27.85\%$) and parabolic ($22.90\%$) exceed cosine ($16.85\%$) and linear ($15.64\%$); at $L_t = 10\;\mu\text{m}$, exponential ($36.16\%$) and parabolic ($35.06\%$) exceed cosine ($27.64\%$) and linear ($26.74\%$). Conversely, at $L_t = 20\text{--}50\;\mu\text{m}$, the steeper initial flare $|\mathrm{d}w/\mathrm{d}x|_{x=0}$ of the exponential and parabolic tapers at the $10\;\mu\text{m}$ entrance excites stronger $\text{TE}_2/\text{TE}_4$ mode beating than the constant-slope linear taper, so the linear taper overtakes them ($58.21\%$ linear vs $50.32\%$ exponential, $49.79\%$ parabolic, and $38.91\%$ cosine at $L_t = 20\;\mu\text{m}$; $87.41\%$ linear vs $80.30\%$ cosine, $78.08\%$ parabolic, and $70.41\%$ exponential at $L_t = 50\;\mu\text{m}$).
#
# ### 6.2 Convergence & resolution verdict (`--full` study)
#
# - **Resolution rule compliance**: Reading `cpw` and `cells_across_feature` from `data/task_0.json`–`task_15.json`, every non-demo run (Tasks 2–15) has $\text{cpw} \in \{10, 14, 20, 40\} \ge 10$ in the highest-index material and $\text{cells\_across\_feature} \in \{4.0, 6.0, 8.0, 15.0, 16.0, 21.0\} \ge 4.0$ across the smallest geometric feature ($50\;\text{nm}$ Segment 3 flare in Design 3; $500\;\text{nm}$ output waveguide in Designs 1–2). Only the intentional under-resolved demonstration pair (`task_0` and `task_1`, marked `"demo": true`) has $0.988 < 4$ cells across Segment 3. Zero violations occur.
# - **Convergence at the accepted level `L3 = (20, 8)`**:
#   - Between `L2 = (14, 6)` and `L3 = (20, 8)` (Kottke), the relative change at $1550\;\text{nm}$ is **$0.089\%$** in $T_{\text{flux}}$ ($95.04\% \to 95.12\%$), **$0.045\%$** in $T_{\text{even}}$ ($94.13\% \to 94.17\%$), **$2.085\%$** in $T_{\text{TE}_0}$ ($78.63\% \to 76.99\%$), and **$2.129\%$** in $\text{TE}_0$ share ($83.54\% \to 81.76\%$).
#   - Between the accepted level `L3 = (20, 8)` and the fine reference `R = (40, 16)` (doubling both background cpw and feature cells), the relative change at $1550\;\text{nm}$ is **$0.064\%$** in $T_{\text{flux}}$ ($95.12\% \to 95.18\%$), **$0.060\%$** in $T_{\text{even}}$ ($94.17\% \to 94.22\%$), **$1.446\%$** in $T_{\text{TE}_0}$ ($76.99\% \to 75.88\%$), and **$1.505\%$** in $\text{TE}_0$ share ($81.76\% \to 80.53\%$). Across the $1510\text{--}1590\;\text{nm}$ band, the RMS deviation between `L3` and `R` is $9.52 \times 10^{-4}$ ($0.095\%$) in $T_{\text{flux}}$, $1.19 \times 10^{-3}$ ($0.119\%$) in $T_{\text{even}}$, and $9.18 \times 10^{-3}$ ($0.918\%$) in $T_{\text{TE}_0}$. Thus total and even-mode transmission are converged to $< 0.12\%$, and the phase-sensitive $\text{TE}_0/\text{TE}_2$ interference split is converged to $\sim 1.0\text{--}1.5\%$ at `L3 = (20, 8)`.
# - **Kottke vs staircasing across `--quick` and `--full`**:
#   In the `--quick` study ($n_{\text{pml}} = 8$, energy decay $4 \times 10^{-3}$, evaluated against `L3 = (20, 8)` as the reference), Kottke yields a $1.2\text{--}1.5\times$ smaller band-RMS error on $T_{\text{even}}(\lambda)$ than staircasing at both `L1` and `L2`. In the `--full` study ($n_{\text{pml}} = 10$, energy decay $10^{-4}$, evaluated against the independent `R = (40, 16)` reference):
#   1. On **total guided even-mode capture $T_{\text{even}}(\lambda)$**, at the coarsest background grid `L1 = (10, 4)` ($\text{cpw} = 10$), weak Fabry–Pérot standing-wave ripples ($\langle |S_{11}|^2\rangle = 4.96 \times 10^{-3}$ for Kottke vs $1.67 \times 10^{-3}$ for staircase; see `14_resolution_spectra_and_time_trace.png`(a)) cause a larger $T_{\text{even}}$ RMS error for Kottke ($3.54 \times 10^{-2}$ at `L1`, $3.63 \times 10^{-2}$ at `(10, 1)`) than staircase ($2.03 \times 10^{-2}$ at `L1`, $2.02 \times 10^{-2}$ at `(10, 1)`). Once the background grid is refined to `L2 = (14, 6)` and `L3 = (20, 8)`, reflection drops below $-33.9\;\text{dB}$ ($\langle |S_{11}|^2\rangle \approx 3.4\text{--}4.0 \times 10^{-4}$) and both discretizations converge steeply (empirical order $p \approx 4.4\text{--}4.9$ from `L1` to `L3`) to virtually identical $T_{\text{even}}$ accuracy ($5.12 \times 10^{-3}$ Kottke vs $5.10 \times 10^{-3}$ staircase at `L2`; $1.19 \times 10^{-3}$ Kottke vs $9.51 \times 10^{-4}$ staircase at `L3`).
#   2. On **fundamental-mode collimation $T_{\text{TE}_0}(\lambda)$** (which is governed by refraction across the oblique curved lens boundary of Segment 5), Kottke subpixel smoothing exhibits a faster asymptotic convergence rate ($p \approx 2.61$: $5.59 \times 10^{-2}$ at `L1` $\to$ $2.18 \times 10^{-2}$ at `L2` $\to$ $9.18 \times 10^{-3}$ at `L3`) than staircasing ($p \approx 2.07$: $5.12 \times 10^{-2}$ at `L1` $\to$ $1.89 \times 10^{-2}$ at `L2` $\to$ $1.22 \times 10^{-2}$ at `L3`), yielding a **$1.33\times$ lower $\text{TE}_0$ RMS error at the accepted `L3 = (20, 8)` level** ($9.18 \times 10^{-3}$ vs $1.22 \times 10^{-2}$).
#
# ### 6.3 Comparison with published 3D FDTD & analytic theory (`--quick` vs `--full` vs Reference)
#
# | Quantity (at $\lambda_0 = 1550\;\text{nm}$ unless noted) | Quick (`--quick`) | Full (`--full`) | Published / Analytic Reference | Notes on Model & Parameter Dependence |
# |---|---:|---:|---:|---|
# | **Design 3 (Semi-lens, $L = 9.50\;\mu\text{m}$)** $T_{\text{flux}}$ | $95.15\%$ (`L2`) / $95.12\%$ (`L3`) | **$95.12\%$** (`L3`) / **$94.92\%$** (`nonuniform`) / **$95.18\%$** (`R`) | $\sim 95\%$ ($-0.22\;\text{dB}$, Abbaslou *et al.* 2017 [2], 3D FDTD) | Matches 3D FDTD total transmission within $0.2\%$ |
# | **Design 3 (Semi-lens)** $T_{\text{even}} = \sum_{m=0,2,4,6,8} T_{\text{TE}_m}$ | $94.24\%$ (`L2`) / $94.17\%$ (`L3`) | **$94.17\%$** (`L3`) / **$93.95\%$** (`nonuniform`) / **$94.22\%$** (`R`) | $\sim 94\text{--}95\%$ guided capture [2] | $> 99\%$ of transmitted port flux is captured in $\text{TE}_{0..8}$ |
# | **Design 3 (Semi-lens)** $T_{\text{TE}_0}$ ($1550\;\text{nm}$ / band mean) | $78.69\%$ (`L2`) / $77.00\%$ (`L3`) | **$76.99\% / 77.75\%$** (`L3`), **$75.88\% / 76.87\%$** (`R`) | $\approx 76\%$ ($-1.2\;\text{dB} = 95\% \times 80\%$ [2], 3D FDTD) | Converges monotonically toward 3D FDTD as mesh is refined |
# | **Design 3 (Semi-lens)** $\text{TE}_0$ share $T_{\text{TE}_0}/T_{\text{even}}$ | $83.50\%$ (`L2`) / $81.77\%$ (`L3`) | **$81.76\%$** (`L3`) / **$80.53\%$** (`R`) | **$\approx 80\%$** (Abbaslou *et al.* 2017 [2], 3D FDTD) | Un-tuned literature geometry `(SEMILENS_L_UM, SEMILENS_W_UM, SEMILENS_M)` |
# | **Design 3 (Semi-lens)** $\text{TE}_2$ power ($\%$ of input / $\%$ of $T_{\text{even}}$) | $14.41\%$ (`L2`) / $16.35\%$ (`L3`) | **$16.36\% / 17.37\%$** (`L3`), **$17.64\% / 18.72\%$** (`R`) | Dominant higher-order even mode ($\sim 15\text{--}19\%$) [2] | Residual wavefront curvature couples primarily into $\text{TE}_2$ |
# | **Design 2 (Luneburg lens, $D = 11.0\;\mu\text{m}$)** $t_{\text{lens}}(0)$ | $0.356\;\mu\text{m}$ ($n_{\text{edge}} = 2.2166$) | **$0.3561\;\mu\text{m}$** ($n_{\text{edge}} = 2.2166$) | $\approx 0.36\;\mu\text{m}$ ($t_{\text{edge}} = 0.11\;\mu\text{m}$, Badri & Gilarlue 2019 [1]) | Cubic polynomial fit to 1D SOI $\text{TE}_0$ slab ($t \in [0.11, 0.40]\;\mu\text{m}$) |
# | **Design 2 (Luneburg lens)** $T_{\text{TE}_0}$ ($1550\;\text{nm}$) | $95.85\%$ ($-0.184\;\text{dB}$, `L2`, 40 shells) | **$96.48\%$** ($-0.155\;\text{dB}$, `L3`, 60 shells) | $\approx 89\text{--}91\%$ ($\text{IL} \approx 0.4\text{--}0.5\;\text{dB}$, 3D FDTD [1]) | 2D EIM neglects 3D vertical step radiation ($\sim 0.3\;\text{dB}$) |
# | **Design 2 (Luneburg lens)** Return loss $R$ ($1550\;\text{nm}$) | $0.034\%$ ($-34.7\;\text{dB}$, `L2`) | **$0.021\%$** ($-36.75\;\text{dB}$, `L3`) | $< -25\;\text{dB}$ ($< 0.3\%$, 3D FDTD [1]) | Smooth GRIN profile suppresses Fresnel back-reflection |
# | **Design 1 (Abrupt step, $L_t = 0\;\mu\text{m}$)** $T_{\text{TE}_0}$ | $19.33\%$ ($-7.14\;\text{dB}$, `L2`) | **$19.04\%$** ($-7.20\;\text{dB}$, `L3`) | $\eta_{\text{EM}} = 21.00\%$ ($-6.78\;\text{dB}$); $\|\langle E_y\rangle\|^2 = 22.84\%$ | 2D FIT includes multi-mode radiation & step impedance mismatch |
# | **Design 1 (Linear taper)** $T_{\text{TE}_0}$ vs $L_t$ | $5\;\mu\text{m}: 16.23\%$; $10\;\mu\text{m}: 26.96\%$; $20\;\mu\text{m}: 57.58\%$; $35\;\mu\text{m}: 81.84\%$ | **$5\;\mu\text{m}: 15.64\%$; $10\;\mu\text{m}: 26.87\%$; $20\;\mu\text{m}: 57.76\%$; $50\;\mu\text{m}: 92.70\%$; $100\;\mu\text{m}: 99.34\%$** | Adiabatic limit $T_{\text{TE}_0} \to 100\%$ for $L_t \gtrsim n_{\text{eff}} w_{\text{in}}^2 / \lambda_0 \sim 100\;\mu\text{m}$ [3] | Follows Milton–Burns quadratic adiabatic length scaling |
# | **Design 1 (Profile sweep, $L_t = 10 / 20 / 50\;\mu\text{m}$)** $T_{\text{TE}_0}$ | $10\;\mu\text{m}$: Lin $26.96\%$, Par $36.13\%$, Exp $36.08\%$, Cos $27.99\%$ (`L2`) | **$10\;\mu\text{m}$: Lin $26.74\%$, Par $35.06\%$, Exp $36.16\%$, Cos $27.64\%$**<br>**$20\;\mu\text{m}$: Lin $58.21\%$, Par $49.79\%$, Exp $50.32\%$, Cos $38.91\%$**<br>**$50\;\mu\text{m}$: Lin $87.41\%$, Par $78.08\%$, Exp $70.41\%$, Cos $80.30\%$** | Non-linear shapes reduce loss at short $L_t$ when local slope tracks beat length $L_b(w)$ [3] | Fixed algebraic shapes without segment optimization; linear wins for $L_t \ge 20\;\mu\text{m}$ |


# %%
def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Read ``data_dir/task_*.npz`` and render all figures into ``fig_dir``."""
    figstyle.apply_style()
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tasks_data = {}
    tasks_meta = {}
    for p in sorted(data_dir.glob("task_*.npz")):
        d = dict(np.load(p, allow_pickle=False))
        name = str(d["name"])
        tasks_data[name] = d
        jp = p.with_suffix(".json")
        if jp.exists():
            tasks_meta[name] = json.loads(jp.read_text())

    primary = None
    for cand in ("kottke_L3", "kottke_L2", "kottke_L1", "kottke_fine"):
        if cand in tasks_data and "dft_Ey" in tasks_data[cand]:
            primary = tasks_data[cand]
            break
    if primary is None:
        for cand in ("kottke_L3", "kottke_L2", "kottke_L1", "kottke_fine"):
            if cand in tasks_data:
                primary = tasks_data[cand]
                break
    if primary is None:
        raise FileNotFoundError(
            f"Missing primary semi-lens task ('kottke_L2' or 'kottke_L3') in {data_dir}. Run --quick or --full first."
        )
    semi_structs, semi_ports, semi_dom = build_semilens_expander()
    cpw_prim = float(primary.get("cpw", 20))
    nf_prim = int(primary.get("n_feature", 8))

    zoom_dom = (4.20e-6, 5.05e-6, -5.20e-6, 5.20e-6)
    circuit_zoom_prim = PhotonicCircuit(
        WL_MIN,
        WL_MAX,
        structures=semi_structs,
        ports=semi_ports,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw_prim,
        domain=zoom_dom,
        n_pml=6,
        mesh_overrides=semilens_mesh_overrides(nf_prim),
    )

    # 0. Setup schematics + mesh views (FIRST figures)
    plot_setup_schematic_semilens(
        semi_structs,
        semi_ports,
        semi_dom,
        primary_data=primary,
        circuit_zoom=circuit_zoom_prim,
        out_path=fig_dir / "00a_setup_geometry.png",
    )
    plot_taper_and_luneburg_schematics(
        out_path=fig_dir / "00b_taper_and_luneburg_schematics.png",
    )

    # 1. Three-panel layout overview of all 3 mode-size converter concepts
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.0)
    tap_s, tap_p, tap_d = build_taper_converter(length_taper=20.0e-6, profile="linear")
    lun_s, lun_p, lun_d, _ = build_luneburg_converter(n_shells=24)

    for ax, (st_list, pt_list, dom, title_str) in zip(
        axes,
        [
            (
                tap_s,
                tap_p,
                tap_d,
                "(a) Design 1: Linear Adiabatic Taper\n(10.0 → 0.50 µm, L = 5–100 µm)",
            ),
            (
                lun_s,
                lun_p,
                lun_d,
                "(b) Design 2: Graded Luneburg Lens\n(D = 11.0 µm, t = 0.11–0.36 µm)",
            ),
            (
                semi_structs,
                semi_ports,
                semi_dom,
                "(c) Design 3: 6-Segment Semi-Lens\n(0.45 → 9.00 µm, L = 9.50 µm)",
            ),
        ],
    ):
        for s_idx, poly in enumerate(st_list):
            v = poly.vertices * 1e6
            if len(st_list) > 3 and s_idx >= 2:
                frac = (s_idx - 2) / max(len(st_list) - 3, 1)
                col = plt.cm.Blues(0.35 + 0.55 * frac)
                ax.fill(v[:, 0], v[:, 1], facecolor=col, edgecolor="none", alpha=0.85)
            else:
                ax.fill(
                    v[:, 0],
                    v[:, 1],
                    facecolor="#3b6fb6",
                    edgecolor="#1c3b6b",
                    lw=1.1,
                    alpha=0.82,
                )
        for pt in pt_list:
            cx, cy = pt.center[0] * 1e6, pt.center[1] * 1e6
            hy = 0.5 * pt.size[1] * 1e6
            ax.plot([cx, cx], [cy - hy, cy + hy], color="#d95f02", lw=2.2)
            ax.text(
                cx,
                cy + hy + 0.35,
                pt.name,
                color="#d95f02",
                fontsize=8.5,
                ha="center",
                weight="bold",
            )
        ax.set(
            xlim=(dom[0] * 1e6, dom[1] * 1e6),
            ylim=(dom[2] * 1e6, dom[3] * 1e6),
            xlabel="x [µm]",
            ylabel="y [µm]",
            title=title_str,
        )
        ax.grid(True, ls=":", alpha=0.4)
    figstyle.save(fig, fig_dir / "01_converter_layouts_overview.png")
    plt.close(fig)

    # 2. Effective-Index Method (EIM) vertical stack (260 nm SOI semi-lens stack)
    plot_eim_stack(
        n_core=N_SI,
        n_clad=N_SIO2,
        thickness=T_WG_SEMILENS,
        wavelength_ref=WL_REF,
        wavelengths=np.linspace(WL_MIN, WL_MAX, 51),
        out_path=fig_dir / "02_eim_vertical_stack.png",
    )

    # 3. Luneburg lens 1D slab calibration & radial thickness/index profiles
    lune_d = tasks_data.get("luneburg_lens")
    if lune_d is not None and "cal_t_sweep_um" in lune_d:
        t_sw = lune_d["cal_t_sweep_um"]
        ne_ana = lune_d["cal_neff_sweep_ana"]
        ne_fit = lune_d["cal_neff_sweep_fit"]
        coeffs = lune_d["cal_poly_coeffs"]
        r_um = lune_d["cal_r_um"]
        t_rad = lune_d["cal_t_luneburg_um"]
        ne_rad = lune_d["cal_neff_luneburg"]
        ne_real = lune_d["cal_neff_realized"]
    else:
        cal = calibrate_luneburg_profile(solve_1d_fit=False)
        t_sw, ne_ana, ne_fit = (
            cal["t_sweep_um"],
            cal["neff_sweep_ana"],
            cal["neff_sweep_fit"],
        )
        coeffs, r_um, t_rad = cal["poly_coeffs"], cal["r_um"], cal["t_luneburg_um"]
        ne_rad, ne_real = cal["neff_luneburg"], cal["neff_realized"]

    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)
    ne_grid = np.linspace(ne_ana[0], ne_ana[-1], 200)
    ax0.plot(
        ne_ana,
        t_sw * 1e3,
        "o",
        color="#2b5c8f",
        ms=4.5,
        label=r"Analytical 1D $\mathrm{TE}_0$ slab $n_{\mathrm{eff}}(t)$",
    )
    ax0.plot(
        ne_fit,
        t_sw * 1e3,
        "x",
        color="#1b9e77",
        ms=5.0,
        label=r"Wakis 1D FIT `solve_modes`",
    )
    ax0.plot(
        ne_grid,
        np.polyval(coeffs, ne_grid) * 1e3,
        "-",
        color="#d95f02",
        lw=2.0,
        label=r"Cubic polynomial fit $t_{\mathrm{fit}}(n_{\mathrm{eff}})$",
    )
    ax0.set(
        xlabel=r"Vertical slab effective index $n_{\mathrm{eff}}$",
        ylabel=r"Silicon core thickness $t_{\mathrm{Si}}$ [nm]",
        title=r"(a) 1D SOI slab calibration ($\lambda_0 = 1550\;\mathrm{nm}$)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper left", fontsize=8.0)

    ax1.plot(
        r_um,
        t_rad * 1e3,
        color="#2b5c8f",
        lw=2.2,
        label=r"Thickness profile $t_{\mathrm{lens}}(r)$ [nm]",
    )
    ax1.set(
        xlabel="Radial distance r from lens center [µm]",
        ylabel=r"Silicon thickness $t_{\mathrm{lens}}(r)$ [nm]",
        title=r"(b) Graded Luneburg lens profile ($R = 5.5\;\mu\mathrm{m}$)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1_r = ax1.twinx()
    ax1_r.plot(
        r_um,
        ne_rad,
        "--",
        color="#d95f02",
        lw=2.0,
        label=r"Target $n_{\mathrm{edge}}\sqrt{2-(r/R)^2}$",
    )
    ax1_r.plot(
        r_um,
        ne_real,
        ":",
        color="#1b9e77",
        lw=2.0,
        label=r"Realized $n_{\mathrm{eff}}(t_{\mathrm{lens}}(r))$",
    )
    ax1_r.set_ylabel(r"Local effective index $n_{\mathrm{eff}}(r)$", color="#d95f02")
    lines0, labels0 = ax1.get_legend_handles_labels()
    lines1, labels1 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines0 + lines1, labels0 + labels1, loc="upper right", fontsize=7.8)
    figstyle.save(fig, fig_dir / "03_luneburg_calibration_and_profile.png")
    plt.close(fig)

    # 4. Permittivity mesh zoom on the 50 nm Segment 3 flare: L1=(10,4) vs Reference
    ref_d = tasks_data.get("kottke_reference", primary)
    cpw_ref = float(ref_d.get("cpw", 20))
    nf_ref = int(ref_d.get("n_feature", 8))
    circuit_l1_stair = PhotonicCircuit(
        WL_MIN,
        WL_MAX,
        structures=semi_structs,
        ports=semi_ports,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        two_d=True,
        averaging="staircase",
        cells_per_wavelength=10,
        domain=zoom_dom,
        n_pml=6,
        mesh_overrides=semilens_mesh_overrides(4),
    )
    circuit_l1_kottke = PhotonicCircuit(
        WL_MIN,
        WL_MAX,
        structures=semi_structs,
        ports=semi_ports,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=10,
        domain=zoom_dom,
        n_pml=6,
        mesh_overrides=semilens_mesh_overrides(4),
    )
    circuit_ref_kottke = PhotonicCircuit(
        WL_MIN,
        WL_MAX,
        structures=semi_structs,
        ports=semi_ports,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw_ref,
        domain=zoom_dom,
        n_pml=6,
        mesh_overrides=semilens_mesh_overrides(nf_ref),
    )

    fig, axes = figstyle.new_figure(
        nrows=1, ncols=3, width=13.8, height=4.4, sharey=True
    )
    xlim_z = (4.53, 4.74)
    ylim_z = (1.45, 4.70)
    v_semi = semi_structs[0].vertices * 1e6
    im_m = None
    for ax, circuit_m, title_m, n_f_m in [
        (
            axes[0],
            circuit_l1_stair,
            "(a) L1=(10, 4) Staircase\n4 cells / 50 nm (dx_min=12.5 nm)",
            4,
        ),
        (
            axes[1],
            circuit_l1_kottke,
            "(b) L1=(10, 4) Kottke\n4 cells / 50 nm (dx_min=12.5 nm)",
            4,
        ),
        (
            axes[2],
            circuit_ref_kottke,
            f"(c) Reference=({int(cpw_ref)}, {nf_ref}) Kottke\n{nf_ref} cells / 50 nm (dx_min={50.0 / nf_ref:.2f} nm)",
            nf_ref,
        ),
    ]:
        slv = circuit_m.solver
        iz = slv.Nz // 2
        ieps_y = slv.ieps.to_matrix("y")[:, :, iz]
        if hasattr(ieps_y, "get"):
            ieps_y = ieps_y.get()
        ieps_arr = np.asarray(ieps_y, dtype=float)
        eps_y = np.where(
            ieps_arr > 0.0,
            1.0 / (np.where(ieps_arr > 0.0, ieps_arr, 1.0) * epsilon_0),
            1.0,
        )
        xe = slv.grid.x * 1e6
        ye = slv.grid.y * 1e6
        im_m = ax.pcolormesh(
            xe,
            ye,
            eps_y[: slv.Nx, : slv.Ny].T,
            cmap="viridis",
            vmin=1.0,
            vmax=float(n_to_eps(circuit_m.n_eff_slab)) * 1.02,
            edgecolors="w",
            linewidth=0.35,
        )
        ax.plot(
            np.append(v_semi[:, 0], v_semi[0, 0]),
            np.append(v_semi[:, 1], v_semi[0, 1]),
            color="#ff3333",
            lw=1.7,
            label="Exact Segment 3 boundary",
        )
        ax.axvline(4.61, color="#ffd92f", ls="--", lw=1.2)
        ax.axvline(4.66, color="#ffd92f", ls="--", lw=1.2)
        ax.annotate(
            "",
            xy=(4.61, 4.30),
            xytext=(4.66, 4.30),
            arrowprops=dict(arrowstyle="<->", color="#ffd92f", lw=1.6),
        )
        ax.text(
            4.635,
            4.38,
            f"L3 = 50 nm\n({n_f_m} cells)",
            color="#ffd92f",
            fontsize=8.2,
            ha="center",
            va="bottom",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="k", ec="none", alpha=0.68),
        )
        ax.set(xlim=xlim_z, ylim=ylim_z, xlabel="x [µm]")
        ax.set_title(title_m, fontsize=9.5)
    axes[0].set_ylabel("y [µm]")
    axes[0].legend(loc="lower right", fontsize=7.8)
    fig.colorbar(
        im_m,
        ax=axes.ravel().tolist(),
        shrink=0.88,
        pad=0.02,
        label=r"Relative permittivity $\varepsilon_{yy}$",
    )
    figstyle.save(fig, fig_dir / "04_permittivity_mesh_zoom.png")
    plt.close(fig)

    # 4b. Mesh upgrade comparison: uniform-background kottke_L3 vs graded kottke_nonuniform
    if "kottke_nonuniform" in tasks_data:
        circuit_u_cmp = PhotonicCircuit(
            WL_MIN,
            WL_MAX,
            structures=semi_structs,
            ports=semi_ports,
            n_core=N_SI,
            n_clad=N_AIR,
            n_sub=N_SIO2,
            two_d=True,
            averaging="kottke",
            cells_per_wavelength=cpw_prim,
            domain=semi_dom,
            n_pml=10,
            mesh_overrides=semilens_mesh_overrides(nf_prim),
        )
        circuit_nu_cmp = NonUniformSemilensCircuit(
            WL_MIN,
            WL_MAX,
            structures=semi_structs,
            ports=semi_ports,
            n_core=N_SI,
            n_clad=N_AIR,
            n_sub=N_SIO2,
            two_d=True,
            averaging="kottke",
            cells_per_wavelength=cpw_prim,
            domain=semi_dom,
            n_pml=10,
            n_feature=nf_prim,
        )
        prim_name = str(primary.get("name", "kottke_L3"))
        plot_mesh_upgrade_comparison_semilens(
            semi_structs,
            d_uniform=primary,
            d_nonuniform=tasks_data["kottke_nonuniform"],
            d_ref=tasks_data.get("kottke_reference", primary),
            meta_uniform=tasks_meta.get(prim_name, {}),
            meta_nonuniform=tasks_meta.get("kottke_nonuniform", {}),
            meta_ref=tasks_meta.get("kottke_reference", tasks_meta.get(prim_name, {})),
            grid_uniform=circuit_u_cmp.grid,
            grid_nonuniform=circuit_nu_cmp.grid,
            out_path=fig_dir / "04b_mesh_upgrade_comparison.png",
        )

    # 5. Port eigenmode profiles (narrow single-mode input vs wide multi-mode output)
    if "port_y_in" in primary:
        plot_port_modes(
            primary["port_y_in"] * 1e6,
            primary["port_ey_in"],
            float(primary["port_neff_in"]),
            primary["port_y_out"] * 1e6,
            primary["port_ey_out"],
            float(primary["port_neff_out"]),
            out_path=fig_dir / "05_port_mode_profiles.png",
        )

    # 6. Input waveguide mode effective index and group index dispersion
    if "disp_wavelengths" in primary:
        plot_neff_dispersion(
            primary["disp_wavelengths"] * 1e9,
            primary["disp_neff"],
            neff_slab=float(primary["n_eff_slab"]),
            ng_2d=primary["disp_ng"],
            out_path=fig_dir / "06_port_neff_dispersion.png",
        )

    # Collect available linear taper tasks sorted by length_um
    avail_tapers = sorted(
        [
            (int(round(float(d["length_um"]))), k, d)
            for k, d in tasks_data.items()
            if str(d.get("kind", "")) == "taper_single"
        ],
        key=lambda item: item[0],
    )
    palette = ["#e41a1c", "#ff7f00", "#377eb8", "#1b9e77", "#7570b3"]

    # 7. Linear taper length spectra & width profiles
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.2, height=4.1)
    s_norm = np.linspace(0.0, 1.0, 300)
    for prof, col, ls in [
        ("linear", "#2b5c8f", "-"),
        ("parabolic", "#d95f02", "--"),
        ("exponential", "#1b9e77", "-."),
        ("cosine", "#7570b3", ":"),
    ]:
        w_p = taper_width_profile(s_norm, 1.0, profile=prof) * 1e6
        ax0.plot(s_norm, w_p, color=col, ls=ls, lw=2.0, label=f"{prof.capitalize()}")
    ax0.set(
        xlabel=r"Normalized axial coordinate $x / L_t$",
        ylabel=r"Waveguide width $w(x)$ [µm]",
        title="(a) Taper width profiles (10.0 → 0.50 µm)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=8.5)

    for idx_t, (L_u, _, td) in enumerate(avail_tapers):
        col = palette[idx_t % len(palette)]
        w_nm = td["wavelengths"] * 1e9
        ord_w = np.argsort(w_nm)
        w_s = w_nm[ord_w]
        m = (w_s >= 1505.0) & (w_s <= 1595.0)
        il_db = 10.0 * np.log10(
            np.maximum(np.abs(td["S21"][ord_w][m]) ** 2, 1e-12)
        )
        ax1.plot(
            w_s[m],
            il_db,
            color=col,
            lw=2.0,
            label=rf"Linear taper $L_t = {L_u}\;\mu\mathrm{{m}}$",
        )
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Fundamental $\mathrm{TE}_0$ transmission $10\log_{10}|S_{21}|^2$ [dB]",
        ylim=(-8.5, 0.3),
        title="(b) Linear taper insertion loss across 1505–1595 nm",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=8.5)
    figstyle.save(fig, fig_dir / "07_taper_length_spectra_and_profiles.png")
    plt.close(fig)

    # 8 & 8b. Re(Ey) field maps for available linear tapers (split into <= 2 panels per figure)
    for fig_name, subset, panel_letters in [
        ("08_taper_ey_field_comparison.png", avail_tapers[:2], ("(a)", "(b)")),
        ("08b_taper_ey_long_comparison.png", avail_tapers[2:4], ("(a)", "(b)")),
    ]:
        if not subset:
            continue
        fig, axes = figstyle.new_figure(
            nrows=1, ncols=len(subset), width=12.4, height=4.4
        )
        axes_arr = np.atleast_1d(axes)
        for ax, p_lbl, (L_u, _, td) in zip(axes_arr, panel_letters, subset):
            if "dft_Ey" not in td:
                continue
            st_t, pt_t, dm_t = build_taper_converter(
                length_taper=float(L_u) * 1e-6, profile="linear"
            )
            ey_re = np.real(td["dft_Ey"]).T
            vmax = max(float(np.percentile(np.abs(ey_re), 99.2)), 1e-12)
            im = ax.pcolormesh(
                td["dft_x"] * 1e6,
                td["dft_y"] * 1e6,
                ey_re,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
            )
            slv_t, _, _, mat_t = build_schematic_components_converter(
                dm_t,
                pt_t,
                n_clad=N_SIO2,
                n_pml=10,
                include_flux_box=False,
            )
            schematic.overlay_geometry(
                ax,
                slv_t,
                plane="xy",
                unit="µm",
                structures=(),
                material_names=mat_t,
                sources=(),
                monitors=(),
                outline_color="#0f172a",
            )
            draw_polyslab_regions(
                ax,
                st_t,
                facecolor=None,
                edgecolor="#0f172a",
                lw=1.25,
                label=f"Si taper (n_eff={float(td.get('n_eff_slab', 2.217)):.3f})",
                label_xy=(0.35 * float(L_u), 0.0),
            )
            i0 = int(np.argmin(np.abs(td["wavelengths"] - WL_REF)))
            t_db0 = 10.0 * np.log10(max(float(np.abs(td["S21"][i0]) ** 2), 1e-12))
            ax.set(
                xlabel="x [µm]",
                ylabel="y [µm]",
                title=rf"{p_lbl} $L_t = {L_u}\;\mu\mathrm{{m}}$: $\mathrm{{Re}}(E_y)$ at $1550\;\mathrm{{nm}}$ ($T_{{\mathrm{{TE}}_0}} = {t_db0:.2f}\;\mathrm{{dB}}$)",
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        figstyle.save(fig, fig_dir / fig_name)
        plt.close(fig)

    # 9. Luneburg lens focusing field maps & spectral comparison against linear tapers
    if lune_d is not None and "dft_Ey" in lune_d:
        fig, (ax0, ax1) = figstyle.new_figure(
            nrows=1, ncols=2, width=11.8, height=4.4
        )
        ey_l = np.real(lune_d["dft_Ey"]).T
        vmax_l = max(float(np.percentile(np.abs(ey_l), 99.2)), 1e-12)
        im0 = ax0.pcolormesh(
            lune_d["dft_x"] * 1e6,
            lune_d["dft_y"] * 1e6,
            ey_l,
            cmap="RdBu_r",
            vmin=-vmax_l,
            vmax=vmax_l,
            shading="auto",
        )
        slv_l, _, _, mat_l = build_schematic_components_converter(
            lun_d,
            lun_p,
            n_clad=N_SIO2,
            n_pml=10,
            include_flux_box=False,
        )
        schematic.overlay_geometry(
            ax0,
            slv_l,
            plane="xy",
            unit="µm",
            structures=(),
            material_names=mat_l,
            sources=(),
            monitors=(),
            outline_color="#0f172a",
        )
        draw_polyslab_regions(
            ax0,
            lun_s[:3],
            facecolor=None,
            edgecolor="#0f172a",
            lw=1.3,
            label="GRIN Luneburg (n=2.22→3.13)",
            label_xy=(0.0, 4.2),
        )
        ax0.set(
            xlabel="x [µm]",
            ylabel="y [µm]",
            title=r"(a) Graded Luneburg lens $\mathrm{Re}(E_y)$ focusing ($1550\;\mathrm{nm}$)",
        )
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)

        w_l = lune_d["wavelengths"] * 1e9
        ord_l = np.argsort(w_l)
        ws_l = w_l[ord_l]
        ml = (ws_l >= 1505.0) & (ws_l <= 1595.0)
        ax1.plot(
            ws_l[ml],
            10.0 * np.log10(np.maximum(lune_d["T_flux_port"][ord_l][ml], 1e-12)),
            color="#1b9e77",
            lw=2.2,
            label=r"Luneburg lens ($D = 11\;\mu\mathrm{m}$) port flux",
        )
        ax1.plot(
            ws_l[ml],
            10.0 * np.log10(np.maximum(np.abs(lune_d["S21"][ord_l][ml]) ** 2, 1e-12)),
            "--",
            color="#1b9e77",
            lw=1.9,
            label=r"Luneburg lens ($D = 11\;\mu\mathrm{m}$) $\mathrm{TE}_0$ $|S_{21}|^2$",
        )
        for idx_t, (L_u, _, td) in enumerate(avail_tapers):
            col = palette[idx_t % len(palette)]
            wt = td["wavelengths"] * 1e9
            ot = np.argsort(wt)
            mt = (wt[ot] >= 1505.0) & (wt[ot] <= 1595.0)
            ax1.plot(
                wt[ot][mt],
                10.0
                * np.log10(np.maximum(np.abs(td["S21"][ot][mt]) ** 2, 1e-12)),
                color=col,
                lw=1.6,
                alpha=0.85,
                label=rf"Linear taper $L_t = {L_u}\;\mu\mathrm{{m}}$",
            )
        ax1.axhline(
            -0.5,
            color="0.35",
            ls=":",
            lw=1.4,
            label="Badri & Gilarlue (2019) 3D FDTD (~−0.5 dB)",
        )
        ax1.set(
            xlabel="Wavelength λ [nm]",
            ylabel="Transmission [dB]",
            ylim=(-8.5, 0.2),
            title="(b) Luneburg lens vs linear tapers (1505–1595 nm)",
        )
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.legend(loc="lower right", fontsize=7.8)
        figstyle.save(fig, fig_dir / "09_luneburg_focusing_and_comparison.png")
        plt.close(fig)

    # 10. Semi-lens steady-state field maps at 1550 nm (intensity_log and Re(Ey))
    if "dft_E2" in primary and "dft_Ey" in primary:
        fig, (ax0, ax1) = figstyle.new_figure(
            nrows=1, ncols=2, width=11.8, height=4.4
        )
        x_um_s = primary["dft_x"] * 1e6
        y_um_s = primary["dft_y"] * 1e6
        e2_s = primary["dft_E2"]
        f_max_s = max(float(np.max(e2_s)), 1e-30)
        e2_db = 10.0 * np.log10(np.maximum(e2_s / f_max_s, 10.0 ** (-35.0 / 10.0)))
        im_a = ax0.pcolormesh(
            x_um_s,
            y_um_s,
            e2_db.T,
            cmap="inferno",
            vmin=-35.0,
            vmax=0.0,
            shading="auto",
        )
        ey_s = np.real(primary["dft_Ey"]).T
        vmax_s = max(float(np.percentile(np.abs(ey_s), 99.2)), 1e-12)
        im_b = ax1.pcolormesh(
            x_um_s,
            y_um_s,
            ey_s,
            cmap="RdBu_r",
            vmin=-vmax_s,
            vmax=vmax_s,
            shading="auto",
        )
        slv_s, _, _, mat_s = build_schematic_components_converter(
            semi_dom,
            semi_ports,
            n_clad=N_AIR,
            n_pml=10,
            include_flux_box=False,
        )
        for ax_i, ec_col in ((ax0, "#facc15"), (ax1, "#0f172a")):
            schematic.overlay_geometry(
                ax_i,
                slv_s,
                plane="xy",
                unit="µm",
                structures=(),
                material_names=mat_s,
                sources=(),
                monitors=(),
                outline_color=ec_col,
            )
            draw_polyslab_regions(
                ax_i,
                semi_structs,
                facecolor=None,
                edgecolor=ec_col,
                lw=1.3,
                label=f"Si semi-lens (n_eff={float(primary.get('n_eff_slab', 2.955)):.3f})",
                label_xy=(6.65, 0.0),
            )
            ax_i.set(xlabel="x [µm]", ylabel="y [µm]")
        ax0.set_title(
            r"(a) Semi-lens $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $1550\;\mathrm{nm}$"
        )
        ax1.set_title(
            r"(b) Semi-lens wave-front collimation $\mathrm{Re}(E_y)$ at $1550\;\mathrm{nm}$"
        )
        fig.colorbar(im_a, ax=ax0, fraction=0.046, pad=0.03, label="dB")
        fig.colorbar(im_b, ax=ax1, fraction=0.046, pad=0.03, label=r"$\mathrm{Re}(E_y)$")
        figstyle.save(fig, fig_dir / "10_semilens_field_maps_1550.png")
        plt.close(fig)

    # 11. Semi-lens broadband transmission & 5-mode decomposition (TE0, TE2, TE4, TE6, TE8)
    wl_nm_all = primary["wavelengths"] * 1e9
    order = np.argsort(wl_nm_all)
    wl_nm = wl_nm_all[order]
    mask = (wl_nm >= 1505.0) & (wl_nm <= 1595.0)
    wl_b = wl_nm[mask]

    t_te0 = np.abs(primary["S21"][order][mask]) ** 2
    t_flux = primary["T_flux_port"][order][mask]
    r_semi = np.abs(primary["S11"][order][mask]) ** 2
    mode_powers = {}
    for m_idx in (0, 2, 4, 6, 8):
        key_m = f"S21_TE{m_idx}"
        if key_m in primary:
            mode_powers[m_idx] = np.abs(primary[key_m][order][mask]) ** 2
    t_even_sum = (
        sum(mode_powers.values()) if mode_powers else t_te0
    )

    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.8, height=4.2)
    ax0.plot(
        wl_b,
        10.0 * np.log10(np.maximum(t_flux, 1e-12)),
        color="#2b5c8f",
        lw=2.2,
        label="Output port total flux transmission",
    )
    ax0.plot(
        wl_b,
        10.0 * np.log10(np.maximum(t_even_sum, 1e-12)),
        "--",
        color="#1b9e77",
        lw=2.0,
        label=r"Even-mode sum $\sum_{m=0,2,4,6,8} |S_{\mathrm{TE}_m, 1}|^2$",
    )
    ax0.plot(
        wl_b,
        10.0 * np.log10(np.maximum(t_te0, 1e-12)),
        "-.",
        color="#d95f02",
        lw=2.0,
        label=r"Fundamental $\mathrm{TE}_0$ $|S_{21}|^2$",
    )
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Transmission [dB]",
        ylim=(-1.5, 0.1),
        title="(a) Semi-lens insertion loss & even-mode capture (dB)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0_r = ax0.twinx()
    ax0_r.plot(
        wl_b,
        10.0 * np.log10(np.maximum(r_semi, 1e-12)),
        ":",
        color="#7570b3",
        lw=1.8,
        label=r"Return loss $10\log_{10}|S_{11}|^2$ (right axis)",
    )
    ax0_r.set_ylabel(r"Return loss $10\log_{10}|S_{11}|^2$ [dB]", color="#7570b3")
    ax0_r.set_ylim(-45.0, -15.0)
    h0, l0 = ax0.get_legend_handles_labels()
    h0r, l0r = ax0_r.get_legend_handles_labels()
    ax0.legend(h0 + h0r, l0 + l0r, loc="lower right", fontsize=7.6)

    m_cols = {0: "#1b9e77", 2: "#d95f02", 4: "#7570b3", 6: "#e7298a", 8: "#66a61e"}
    for m_idx, p_arr in mode_powers.items():
        ax1.plot(
            wl_b,
            p_arr * 100.0,
            color=m_cols[m_idx],
            lw=2.0,
            label=rf"$\mathrm{{TE}}_{m_idx}$ ({np.mean(p_arr) * 100:.1f}% mean)",
        )
    te0_share_arr = (t_te0 / np.maximum(t_even_sum, 1e-30)) * 100.0
    ax1.plot(
        wl_b,
        te0_share_arr,
        "--",
        color="#1b9e77",
        lw=1.6,
        alpha=0.85,
        label=rf"$\mathrm{{TE}}_0 / T_{{\mathrm{{even}}}}$ share ({np.mean(te0_share_arr):.1f}% mean)",
    )
    ax1.axhline(
        80.0,
        color="0.4",
        ls=":",
        lw=1.3,
        label="Abbaslou et al. (2017) 3D TE0 share (~80%)",
    )
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Modal power fraction [%]",
        ylim=(0.0, 100.0),
        title=r"(b) Linear modal power decomposition ($\mathrm{TE}_0\text{--}\mathrm{TE}_8$)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="center right", fontsize=7.6)
    figstyle.save(fig, fig_dir / "11_semilens_spectra_and_modal_share.png")
    plt.close(fig)

    # 12. Taper profile & length sweep vs analytical abrupt-step mode overlap
    sw_d = tasks_data.get("taper_profile_sweep")
    if sw_d is not None:
        fig, (ax0, ax1) = figstyle.new_figure(
            nrows=1, ncols=2, width=11.6, height=4.2
        )
        L_arr = sw_d["lengths_um"]
        for prof, col, mk in [
            ("linear", "#2b5c8f", "o-"),
            ("parabolic", "#d95f02", "s--"),
            ("exponential", "#1b9e77", "^-."),
            ("cosine", "#7570b3", "d:"),
        ]:
            if f"T_{prof}" in sw_d:
                ax0.plot(
                    L_arr,
                    sw_d[f"T_{prof}"],
                    mk,
                    color=col,
                    lw=1.9,
                    ms=5.5,
                    label=f"{prof.capitalize()} taper (sweep)",
                )
        if avail_tapers:
            l_single = [float(L_u) for L_u, _, _ in avail_tapers]
            t_single = [
                float(
                    np.abs(
                        td["S21"][int(np.argmin(np.abs(td["wavelengths"] - WL_REF)))]
                    )
                    ** 2
                )
                for _, _, td in avail_tapers
            ]
            if "T_linear" in sw_d and len(L_arr) > 0 and float(L_arr[0]) == 0.0:
                l_single = [0.0, *l_single]
                t_single = [float(sw_d["T_linear"][0]), *t_single]
            ax0.plot(
                l_single,
                t_single,
                "D:",
                color="#1f446e",
                lw=1.6,
                ms=5.2,
                mfc="none",
                mew=1.5,
                label=r"Linear taper ($10^{-4}$ decay, up to $100\;\mu\mathrm{m}$)",
            )
        ax0.axhline(
            float(sw_d["overlap_sq"]),
            color="#e41a1c",
            ls=":",
            lw=1.6,
            label=rf"Abrupt-step $|\langle E_{{\mathrm{{in}}}}, E_{{\mathrm{{out}}}}\rangle|^2 = {float(sw_d['overlap_sq']):.3f}$",
        )
        ax0.axhline(
            float(sw_d["em_overlap"]),
            color="0.35",
            ls="--",
            lw=1.5,
            label=rf"Poynting overlap $\eta_{{\mathrm{{EM}}}} = {float(sw_d['em_overlap']):.3f}$",
        )
        if lune_d is not None:
            i_l = int(np.argmin(np.abs(lune_d["wavelengths"] - WL_REF)))
            t_l0 = float(np.abs(lune_d["S21"][i_l]) ** 2)
            ax0.plot(
                [D_LENS * 1e6],
                [t_l0],
                "*",
                color="#e7298a",
                ms=11,
                label=rf"Luneburg lens ($D=11\;\mu\mathrm{{m}}, T={t_l0:.2f}$)",
            )
        i_s = int(np.argmin(np.abs(primary["wavelengths"] - WL_REF)))
        t_s0 = float(np.abs(primary["S21"][i_s]) ** 2)
        ax0.plot(
            [L_TOT_SEMILENS * 1e6],
            [t_s0],
            "P",
            color="#66a61e",
            ms=9.5,
            label=rf"Semi-lens ($L=9.5\;\mu\mathrm{{m}}, T_{{\mathrm{{TE}}_0}}={t_s0:.2f}$)",
        )
        ax0.set(
            xlabel=r"Converter / taper length $L$ [µm]",
            ylabel=r"Fundamental $\mathrm{TE}_0$ transmission $|S_{21}(1550\;\mathrm{nm})|^2$",
            ylim=(0.0, 1.05),
            title="(a) Mode conversion efficiency vs converter length",
        )
        ax0.grid(True, ls=":", alpha=0.4)
        ax0.legend(loc="lower right", fontsize=7.0)

        y_ov = sw_d["overlap_y"] * 1e6
        e_in = sw_d["overlap_e_in"] / max(float(np.max(sw_d["overlap_e_in"])), 1e-30)
        e_out = sw_d["overlap_e_out"] / max(
            float(np.max(sw_d["overlap_e_out"])), 1e-30
        )
        ax1.plot(
            y_ov,
            e_in,
            color="#2b5c8f",
            lw=2.1,
            label=r"Wide input $\mathrm{TE}_0$ ($w_{\mathrm{in}} = 10.0\;\mu\mathrm{m}$)",
        )
        ax1.plot(
            y_ov,
            e_out,
            color="#d95f02",
            lw=2.1,
            label=r"Narrow output $\mathrm{TE}_0$ ($w_{\mathrm{out}} = 0.50\;\mu\mathrm{m}$)",
        )
        ax1.fill_between(
            y_ov,
            np.minimum(e_in, e_out),
            color="#1b9e77",
            alpha=0.22,
            label="Transverse overlap region",
        )
        ax1.set(
            xlim=(-6.5, 6.5),
            ylim=(-0.05, 1.12),
            xlabel="Transverse coordinate y [µm]",
            ylabel="Peak-normalized electric field $E_y(y)$",
            title=r"(b) Abrupt-step mode mismatch ($10.0\;\mu\mathrm{m} \to 0.50\;\mu\mathrm{m}$)",
        )
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=8.0)
        figstyle.save(fig, fig_dir / "12_taper_profiles_and_overlap.png")
        plt.close(fig)

    # 13. Multi-resolution convergence: Kottke vs staircase against independent kottke_reference
    levels = [
        lvl
        for lvl in ("L1", "L2", "L3")
        if f"kottke_{lvl}" in tasks_data and f"staircase_{lvl}" in tasks_data
    ]
    if "kottke_reference" in tasks_data and len(levels) >= 2:
        cpws = [float(tasks_data[f"kottke_{lvl}"]["cpw"]) for lvl in levels]
        cells_feat = [
            float(tasks_data[f"kottke_{lvl}"].get("cells_across_feature", 4.0))
            for lvl in levels
        ]

        def _t_even_band(d):
            w = d["wavelengths"] * 1e9
            idx_s = np.argsort(w)
            w_s = w[idx_s]
            m = (w_s >= 1510.0) & (w_s <= 1590.0)
            if "S21_TE0" in d:
                ev = sum(
                    np.abs(d[f"S21_TE{k}"][idx_s][m]) ** 2
                    for k in (0, 2, 4, 6, 8)
                    if f"S21_TE{k}" in d
                )
            else:
                ev = np.abs(d["S21"][idx_s][m]) ** 2
            return w_s[m], ev

        def _t_te0_band(d):
            w = d["wavelengths"] * 1e9
            idx_s = np.argsort(w)
            w_s = w[idx_s]
            m = (w_s >= 1510.0) & (w_s <= 1590.0)
            return w_s[m], np.abs(d["S21"][idx_s][m]) ** 2

        def _r_mean(d):
            w = d["wavelengths"] * 1e9
            m = (w >= 1515.0) & (w <= 1585.0)
            return float(np.mean(np.abs(d["S11"][m]) ** 2))

        _, t_ref_arr = _t_even_band(tasks_data["kottke_reference"])
        _, t_te0_ref_arr = _t_te0_band(tasks_data["kottke_reference"])

        err_k = [
            max(
                float(
                    np.sqrt(
                        np.mean(
                            (_t_even_band(tasks_data[f"kottke_{lvl}"])[1] - t_ref_arr)
                            ** 2
                        )
                    )
                ),
                1e-4,
            )
            for lvl in levels
        ]
        err_s = [
            max(
                float(
                    np.sqrt(
                        np.mean(
                            (
                                _t_even_band(tasks_data[f"staircase_{lvl}"])[1]
                                - t_ref_arr
                            )
                            ** 2
                        )
                    )
                ),
                1e-4,
            )
            for lvl in levels
        ]
        err_te0_k = [
            max(
                float(
                    np.sqrt(
                        np.mean(
                            (
                                _t_te0_band(tasks_data[f"kottke_{lvl}"])[1]
                                - t_te0_ref_arr
                            )
                            ** 2
                        )
                    )
                ),
                1e-4,
            )
            for lvl in levels
        ]
        err_te0_s = [
            max(
                float(
                    np.sqrt(
                        np.mean(
                            (
                                _t_te0_band(tasks_data[f"staircase_{lvl}"])[1]
                                - t_te0_ref_arr
                            )
                            ** 2
                        )
                    )
                ),
                1e-4,
            )
            for lvl in levels
        ]
        rk_vals = [_r_mean(tasks_data[f"kottke_{lvl}"]) for lvl in levels]
        rs_vals = [_r_mean(tasks_data[f"staircase_{lvl}"]) for lvl in levels]

        fig, (ax0, ax1) = figstyle.new_figure(
            nrows=1, ncols=2, width=11.8, height=4.5
        )
        res = np.asarray(cpws, dtype=float)
        ax0.loglog(
            res,
            err_s,
            "s-",
            color="#d95f02",
            lw=2.0,
            ms=6.2,
            label=r"Staircase RMS $|T_{\mathrm{even}} - T_{\mathrm{ref}}|$",
        )
        ax0.loglog(
            res,
            err_k,
            "o-",
            color="#1b9e77",
            lw=2.0,
            ms=6.2,
            label=r"Kottke RMS $|T_{\mathrm{even}} - T_{\mathrm{ref}}|$",
        )
        ax0.loglog(
            res,
            err_te0_s,
            "s--",
            color="#d95f02",
            lw=1.5,
            ms=5.0,
            alpha=0.8,
            label=r"Staircase RMS $|T_{\mathrm{TE}_0} - T_{\mathrm{TE}_0,\mathrm{ref}}|$",
        )
        ax0.loglog(
            res,
            err_te0_k,
            "o--",
            color="#1b9e77",
            lw=1.5,
            ms=5.0,
            alpha=0.8,
            label=r"Kottke RMS $|T_{\mathrm{TE}_0} - T_{\mathrm{TE}_0,\mathrm{ref}}|$",
        )

        # Plot under-resolved demo pair (cpw=10, N=1 -> 0.99 cells) if available
        if (
            "kottke_underresolved" in tasks_data
            and "staircase_underresolved" in tasks_data
        ):
            dk_u = tasks_data["kottke_underresolved"]
            ds_u = tasks_data["staircase_underresolved"]
            cpw_u = float(dk_u["cpw"])
            cf_u = float(dk_u.get("cells_across_feature", 0.99))
            ek_u = float(
                np.sqrt(np.mean((_t_even_band(dk_u)[1] - t_ref_arr) ** 2))
            )
            es_u = float(
                np.sqrt(np.mean((_t_even_band(ds_u)[1] - t_ref_arr) ** 2))
            )
            rk_u = _r_mean(dk_u)
            rs_u = _r_mean(ds_u)
            ax0.loglog(
                [cpw_u],
                [es_u],
                "s",
                mfc="none",
                mec="#d95f02",
                mew=1.8,
                ms=8,
                label=f"Staircase under-resolved ({cf_u:.2f} cells)",
            )
            ax0.loglog(
                [cpw_u],
                [ek_u],
                "o",
                mfc="none",
                mec="#1b9e77",
                mew=1.8,
                ms=8,
                label=f"Kottke under-resolved ({cf_u:.2f} cells)",
            )
            ax0.annotate(
                f"{cf_u:.2f} cells (no override)",
                xy=(cpw_u, ek_u),
                xytext=(10, 4),
                textcoords="offset points",
                fontsize=7.3,
                color="#444444",
            )
            ax1.loglog(
                [cpw_u],
                [rs_u],
                "s",
                mfc="none",
                mec="#d95f02",
                mew=1.8,
                ms=8,
                label=f"Staircase under-resolved ({cf_u:.2f} cells)",
            )
            ax1.loglog(
                [cpw_u],
                [rk_u],
                "o",
                mfc="none",
                mec="#1b9e77",
                mew=1.8,
                ms=8,
                label=f"Kottke under-resolved ({cf_u:.2f} cells)",
            )
            ax1.annotate(
                f"{cf_u:.2f} cells (no override)",
                xy=(cpw_u, rs_u),
                xytext=(10, -2),
                textcoords="offset points",
                fontsize=7.3,
                color="#444444",
            )

        for idx_l, (lvl, x_c, cf_v, ek_v, es_v, rk_v, rs_v) in enumerate(
            zip(levels, res, cells_feat, err_k, err_s, rk_vals, rs_vals)
        ):
            y_off_k = -13 if idx_l != 1 else -15
            y_off_s = -12 if idx_l == 0 else (9 if idx_l == 1 else -12)
            x_off_k = 18 if idx_l == 0 else (0 if idx_l == 1 else -18)
            x_off_s = 18 if idx_l == 0 else (0 if idx_l == 1 else -18)
            ax0.annotate(
                f"{lvl}: {cf_v:.1f} cells",
                xy=(x_c, ek_v),
                xytext=(x_off_k, y_off_k),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color="#1b9e77",
                weight="bold",
            )
            if idx_l == 0:
                ax0.annotate(
                    f"{lvl}: {cf_v:.1f} cells",
                    xy=(x_c, es_v),
                    xytext=(x_off_s, y_off_s),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.3,
                    color="#d95f02",
                )
            ax1.annotate(
                f"{lvl}: {cf_v:.1f} cells",
                xy=(x_c, rk_v),
                xytext=(16 if idx_l == 0 else (0 if idx_l == 1 else -16), 7 if idx_l == 0 else -13),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color="#1b9e77",
            )

        ax0.loglog(
            res,
            err_te0_s[0] * (res[0] / res) ** 1,
            "--",
            color="0.55",
            alpha=0.55,
            lw=1.1,
            label=r"$\mathcal{O}(\Delta x^1)$ guide",
        )
        ax0.loglog(
            res,
            err_te0_k[0] * (res[0] / res) ** 2,
            ":",
            color="0.35",
            alpha=0.65,
            lw=1.3,
            label=r"$\mathcal{O}(\Delta x^2)$ guide",
        )
        ref_cpw = int(tasks_data["kottke_reference"]["cpw"])
        ref_cf = float(
            tasks_data["kottke_reference"].get("cells_across_feature", 8.0)
        )
        ax0.set(
            xlabel="Background resolution & cells across 50 nm Segment 3 flare",
            ylabel=r"Band-RMS error vs reference `R`",
            title=rf"(a) Convergence vs $T_{{\mathrm{{ref}}}}$ (cpw={ref_cpw}, {ref_cf:.0f} cells)",
        )
        ax0.grid(True, which="both", ls=":", alpha=0.4)
        ax0.legend(loc="lower left", fontsize=6.7)

        ax1.loglog(
            res,
            rs_vals,
            "s-",
            color="#d95f02",
            lw=2.0,
            ms=6.5,
            label=r"Staircase mean $|S_{11}|^2$",
        )
        ax1.loglog(
            res,
            rk_vals,
            "o-",
            color="#1b9e77",
            lw=2.0,
            ms=6.5,
            label=r"Kottke mean $|S_{11}|^2$",
        )
        ax1.set(
            xlabel="Background resolution & cells across 50 nm Segment 3 flare",
            ylabel=r"Band-averaged reflection $\langle |S_{11}|^2 \rangle$",
            title="(b) Return loss vs resolution & feature cells",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=7.5)

        for ax_i in (ax0, ax1):
            ax_i.set_xlim(res[0] * 0.90, res[-1] * 1.10)
            ax_i.set_xticks(res)
            ax_i.set_xticklabels(
                [
                    f"{int(c)} cpw\n({cf:.1f} cells / 50 nm)"
                    for c, cf in zip(res, cells_feat)
                ]
            )
            ax_i.xaxis.set_minor_formatter(NullFormatter())

        figstyle.save(fig, fig_dir / "13_kottke_vs_staircase_convergence.png")
        plt.close(fig)

    # 14. Multi-resolution spectra & port time trace / ring-down audit
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.1)
    if "kottke_reference" in tasks_data and len(levels) >= 2:
        cols = {
            "underresolved": "#e7298a",
            "L1": "#7570b3",
            "L2": "#d95f02",
            "L3": "#1b9e77",
        }
        plot_lvls = (
            (["underresolved"] if "kottke_underresolved" in tasks_data else [])
            + levels
        )
        for lvl in plot_lvls:
            for avg, ls in (("kottke", "-"), ("staircase", "--")):
                key_l = f"{avg}_{lvl}"
                if key_l not in tasks_data:
                    continue
                d = tasks_data[key_l]
                w_m, ev_m = _t_even_band(d)
                t_db = 10.0 * np.log10(np.maximum(ev_m, 1e-12))
                cf_d = float(d.get("cells_across_feature", 1.0))
                ax0.plot(
                    w_m,
                    t_db,
                    color=cols.get(lvl, "#333333"),
                    ls=ls,
                    lw=1.6,
                    label=f"{avg.capitalize()} {lvl} (cpw={int(d['cpw'])}, {cf_d:.1f} cells)",
                )
        d_ref = tasks_data["kottke_reference"]
        w_r, ev_r = _t_even_band(d_ref)
        cf_r = float(d_ref.get("cells_across_feature", 8.0))
        ax0.plot(
            w_r,
            10.0 * np.log10(np.maximum(ev_r, 1e-12)),
            color="k",
            ls=":",
            lw=2.1,
            label=f"Kottke reference (cpw={int(d_ref['cpw'])}, {cf_r:.1f} cells)",
        )
        ax0.set(
            xlabel="Wavelength λ [nm]",
            ylabel=r"Guided even-mode transmission $10\log_{10}T_{\mathrm{even}}$ [dB]",
            title="(a) Spectral convergence across resolution ladder",
        )
        ax0.grid(True, ls=":", alpha=0.4)
        ax0.legend(loc="lower right", fontsize=6.9)

    if "trace_t" in primary and primary["trace_t"].size > 0:
        t_fs = primary["trace_t"] * 1e15
        ey_tr = primary["trace_Ey"]
        ey_norm = ey_tr / max(float(np.max(np.abs(ey_tr))), 1e-30)
        env = np.maximum(ey_norm**2, 1e-10)
        ax1.semilogy(
            t_fs,
            env,
            color="#2b5c8f",
            lw=1.4,
            label=r"$|E_y(t)|^2 / |E_y|_{\max}^2$ at output port",
        )
        ax1.axhline(
            1e-4, color="#d95f02", ls="--", lw=1.2, label="1e-4 energy decay threshold"
        )
        ax1.set(
            xlabel="Time t [fs]",
            ylabel="Normalized instantaneous power (log scale)",
            ylim=(1e-8, 2.0),
            title="(b) Output port time trace & ring-down audit",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=8.0)
    figstyle.save(fig, fig_dir / "14_resolution_spectra_and_time_trace.png")
    plt.close(fig)

    # 15. Animated pulse propagation GIF with geometry & boundary overlays
    if "movie_frames" in primary and len(primary["movie_frames"]) > 0:
        render_movie_with_overlay(
            primary["movie_x"],
            primary["movie_y"],
            primary["movie_frames"],
            primary["movie_times"],
            structures=semi_structs,
            domain_2d=semi_dom,
            ports=semi_ports,
            out_path=fig_dir / "15_pulse_propagation.gif",
            fps=12,
        )

    print(f"Rendered all figures in {fig_dir}")


# %% [markdown]
# ## 7. Limits & Failure Modes
#
# ### 7.1 What goes wrong when a geometric feature is resolved by only 1–2 cells
#
# In the 6-segment semi-lens beam expander of Abbaslou *et al.* (2017) [2],
# Segment 3 ($x \in [4.61, 4.66]\;\mu\text{m}$) flares the waveguide width from
# $3.30\;\mu\text{m}$ to $9.06\;\mu\text{m}$ over an axial length of only
# $L_3 = 50\;\text{nm}$. At a background resolution of $\text{cpw} = 10$
# ($\Delta x_{\text{bg}} = 50.62\;\text{nm}$), an unrefined mesh places only
# **$0.99$ cells** across Segment 3 (`kottke_underresolved` and
# `staircase_underresolved`). When a structural transition, coupling gap, or
# grating tooth is resolved by only $1\text{--}2$ cells:
#
# - **Staircasing collapses or shifts the boundary**: The discrete step edge
#   jumps by a full cell width $\Delta x \sim 50\;\text{nm}$ depending on
#   how the grid lines align with the transition, altering the effective
#   launching aperture of the cylindrical wavefront into Segment 4, shifting
#   multi-mode interference fringes, and biasing the $\text{TE}_0 / \text{TE}_2$
#   power split.
# - **Subpixel averaging cannot reconstruct sub-cell curvature**: While Kottke
#   anisotropic smoothing [4] assigns a volume-averaged permittivity tensor to
#   the single cell spanning $x \in [4.61, 4.66]\;\mu\text{m}$, a single Yee
#   cell cannot represent the rapid spatial variation of the tangential and
#   normal field components across a $5.76\;\mu\text{m}$ width flare. Adding
#   local `mesh_overrides` ($N \ge 4$ cells across $L_3 = 50\;\text{nm}$,
#   $\Delta x_{\min} \le 12.5\;\text{nm}$ at `L1 = (10, 4)`) resolves the flare
#   geometry directly while keeping the rest of the $9.5\;\mu\text{m}$ device at
#   the economical background cell size.
#
# ### 7.2 When 2D EIM breaks down vs 3D FDTD
#
# Three physical effects distinguish the 2D effective-index method from full 3D FDTD:
#
# 1. **Vertical step radiation in the graded Luneburg lens**: In 3D SOI, the
#    vertical thickness gradient $\mathrm{d}t_{\text{Si}}/\mathrm{d}r$ across
#    the $0.11 \to 0.3561\;\mu\text{m}$ Luneburg dome breaks vertical translational
#    symmetry ($z \neq 0$) and couples a small fraction of power ($\sim 0.25\text{--}0.35\;\text{dB}$)
#    into substrate radiation and TM-like hybrid modes, explaining why our 2D EIM
#    insertion loss ($0.155\;\text{dB}$, $T_{\text{TE}_0} = 96.48\%$ at `L3 = (20, 8)`)
#    is slightly lower than the 3D FDTD insertion loss ($\approx 0.4\text{--}0.5\;\text{dB}$)
#    reported by Badri & Gilarlue (2019) [1].
# 2. **Lateral index contrast at steep lens interfaces**: In Design 3 (Abbaslou
#    *et al.* 2017 [2]), Segment 3 expands from $3.30\;\mu\text{m}$ to
#    $9.06\;\mu\text{m}$ over $50\;\text{nm}$, and Segment 5 forms a strongly
#    curved silicon–air lens interface. Standard 2D EIM slightly overestimates
#    the lateral Fresnel reflection and shifts the focal length relative to 3D
#    SOI, accounting for the small difference between our 2D EIM $\text{TE}_0$
#    share ($81.76\%$ at `L3 = (20, 8)` and $80.53\%$ at `R = (40, 16)`) and the
#    published 3D FDTD value ($\approx 80\%$).
# 3. **Fixed reference wavelength for $n_{\text{eff,slab}}$**: Evaluating the
#    vertical slab effective index at $\lambda_0 = 1550\;\text{nm}$ neglects
#    vertical waveguide dispersion $\mathrm{d}n_{\text{eff,slab}}/\mathrm{d}\lambda$
#    across the $1500\text{--}1600\;\text{nm}$ band.
#
# ## 8. Exercises
#
# 1. **Parabolic vs linear taper length threshold**: Using `taper_profile_sweep`,
#    find the shortest taper length $L_t$ at which a parabolic taper achieves
#    $< 0.5\;\text{dB}$ insertion loss, and compare it with a linear taper.
# 2. **Luneburg shell quantization**: Vary `n_shells` in `build_luneburg_converter`
#    over $\{5, 10, 20, 40, 80\}$ (modeling multi-step greyscale lithography)
#    and plot the $\text{TE}_0$ insertion loss vs the number of etch levels.
# 3. **Semi-lens lens-curvature sensitivity**: Perturb the curvature exponent
#    $m_5 = 0.32$ of Segment 5 in `SEMILENS_M` over $[0.20, 0.50]$ and observe
#    how defocusing redistributes power from $\text{TE}_0$ into $\text{TE}_2$
#    and $\text{TE}_4$.
#
# ## 9. References
#
# 1. S. H. Badri and M. M. Gilarlue, "Silicon waveguide mode size converter based
#    on Luneburg lens," *J. Opt. Soc. Am. B* **36**(5), 1288–1293 (2019),
#    DOI: [10.1364/JOSAB.36.001288](https://doi.org/10.1364/JOSAB.36.001288).
# 2. S. Abbaslou, R. Gatdula, M. Lu, A. Stein, and W. Jiang, "Ultra-short beam
#    expander with segmented curvature control:udge-free mode conversion in
#    silicon," *Opt. Lett.* **42**(21), 4383–4386 (2017),
#    DOI: [10.1364/OL.42.004383](https://doi.org/10.1364/OL.42.004383).
# 3. A. F. Milton and W. K. Burns, "Mode coupling in optical waveguide horns,"
#    *IEEE J. Quantum Electron.* **13**(10), 828–835 (1977),
#    DOI: [10.1109/JQE.1977.1069240](https://doi.org/10.1109/JQE.1977.1069240).
# 4. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing of
#    discretized numerical methods," *Phys. Rev. E* **77**, 036611 (2008),
#    DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="16. Waveguide Mode Size Converters (2D EIM)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run L1/L2/L3 quick 2D study locally and render all 15 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run L1/L2/L3/R full 2D study across all tasks and render all 15 figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study point I (0-indexed; defaults to --full unless --quick is passed)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output directory for --task (default: data/)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Read data/task_*.npz and generate all figures in figures/",
    )
    ap.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the study task table and exit",
    )
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS")
        argv = shlex.split(env_args) if env_args else sys.argv[1:]
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.list_tasks:
        tasks = get_task_table(quick=bool(args.quick))
        print(f"# Total tasks: {len(tasks)}")
        for t in tasks:
            print(f"{t['index']}\t{t['name']}\t{t['description']}")
        return

    if args.task is not None:
        tasks = get_task_table(quick=bool(args.quick))
        if not (0 <= args.task < len(tasks)):
            raise SystemExit(f"--task must be in 0..{len(tasks) - 1}")
        run_single_task(tasks[args.task], args.out)
        return

    if args.plot:
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)
        return

    if not (args.quick or args.full) and any(DATA_DIR.glob("task_*.npz")):
        generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR)
        return

    # Default when invoked with no flags and no precomputed data: --quick
    quick = not args.full
    tasks = get_task_table(quick=quick)
    for t in tasks:
        run_single_task(t, DATA_DIR)
    generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
