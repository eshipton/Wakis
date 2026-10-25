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
# # Tutorial 5.8 · Directional coupler: supermode beating and sidewall-angle sensitivity
#
# **Abstract.** Evanescent directional couplers are the fundamental $2 \times 2$
# coherent beam splitters of silicon photonic integrated circuits (PICs),
# forming the building blocks of Mach–Zehnder interferometers, optical switches,
# ring-resonator bus couplers, and quantum photonic gates. Power transfer
# between two closely spaced waveguides is governed by the interference of the
# symmetric ($\text{TE}_{0,\text{even}}$) and antisymmetric
# ($\text{TE}_{0,\text{odd}}$) supermodes across the parallel interaction
# length $L_c$ and the S-bend transition regions. Because the supermode index
# splitting $\Delta n_{\text{eff}} = n_{\text{even}} - n_{\text{odd}}$ depends
# exponentially on the sub-wavelength coupling gap, nanometer-scale lithographic
# mask bias (`dilation`) and reactive-ion-etch sidewall slant angles
# ($\theta_{\text{sw}} \in [0^\circ, 30^\circ]$) strongly shift the 50:50
# ($3\;\text{dB}$) splitting wavelength $\lambda_{\text{cross}}$. In this
# tutorial we model a $220\;\text{nm}$ silicon-on-insulator (SOI) directional
# coupler in Wakis using the **2D effective-index method (EIM)** and **Kottke
# subpixel permittivity smoothing**, compare full-wave Finite Integration
# Technique (FIT) S-parameters against **Coupled-Mode Theory (CMT)** supermode
# dispersion across interaction length and wavelength, quantify fabrication
# sensitivity over $\theta_{\text{sw}} \in [0^\circ, 30^\circ]$, and demonstrate
# coherent $100\%$ single-arm routing under dual-port $\pi/2$ phase-quadrature
# excitation.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Reduce an asymmetric 3D air-clad SOI waveguide stack
#    ($\text{SiO}_2$ substrate $n_{\text{sub}} = 1.45$, $220\;\text{nm}$
#    $\text{Si}$ core $n_{\text{wg}} = 3.48$, air cladding $n_{\text{clad}} = 1.0$)
#    to a **2D effective-index (EIM)** FIT model (`Ex`, `Ey`, `Hz`), and map 3D
#    trapezoidal slanted-sidewall cross-sections ($\theta_{\text{sw}} \in [0^\circ, 30^\circ]$)
#    onto area-equivalent 2D waveguide widths and gaps.
# 2. Extract the symmetric ($\text{TE}_{0,\text{even}}$) and antisymmetric
#    ($\text{TE}_{0,\text{odd}}$) supermodes of the coupled waveguide section
#    using `solve_modes`, compute the crossover beat length
#    $L_\pi(\lambda) = \lambda / [2(n_{\text{even}} - n_{\text{odd}})]$, and
#    integrate the non-uniform S-bend coupling contribution $L_{\text{sb,eff}}(\lambda)$.
# 3. Validate broadband 2D FIT cross-port ($|S_{31}|^2$) and bar-port
#    ($|S_{41}|^2$) transmission spectra and a 5-point coupling-length sweep
#    ($L_c \in [0, 8]\;\mu\text{m}$) against the analytical CMT $\sin^2 / \cos^2$
#    transfer law.
# 4. Inject simultaneous phase-locked eigenmodes (`PhasedGaussianSource`,
#    $\Delta\phi = \pm\pi/2$) at both input ports to coherently route $100\%$ of
#    the optical power into either output waveguide, and audit subpixel
#    convergence of Kottke smoothing vs staircasing across the coupled
#    $(\text{cpw}, N)$ resolution ladder ($L_1 = (10, 4)$, $L_2 = (14, 6)$,
#    $L_3 = (20, 8)$, and $R = (40, 16)$) with local `mesh_overrides` boxes.
#
# ## Contents
#
# 1. [Physical background: supermode beating & coupled-mode theory](#1.-Physical-background:-supermode-beating-&-coupled-mode-theory)
# 2. [Device specification & trapezoidal sidewall geometry](#2.-Device-specification-&-trapezoidal-sidewall-geometry)
# 3. [Asymmetric 2D Effective-Index Method (EIM) reduction](#3.-Asymmetric-2D-Effective-Index-Method-(EIM)-reduction)
# 4. [Parametric 4-port coupler construction](#4.-Parametric-4-port-coupler-construction)
# 5. [Study tasks: resolution, coupling length, sidewall angle & quadrature](#5.-Study-tasks:-resolution,-coupling-length,-sidewall-angle-&-quadrature)
# 6. [Generating the 16-figure diagnostic suite](#6.-Generating-the-16-figure-diagnostic-suite)
# 7. [Supermode dispersion & CMT validation](#7.-Supermode-dispersion-&-CMT-validation)
# 8. [Sidewall angle sensitivity & dual-port quadrature routing](#8.-Sidewall-angle-sensitivity-&-dual-port-quadrature-routing)
# 9. [Limits & when 2D EIM breaks down](#9.-Limits-&-when-2D-EIM-breaks-down)
# 10. [Exercises](#10.-Exercises)
# 11. [References](#11.-References)

# %% [markdown]
# ## 1. Physical background: supermode beating & coupled-mode theory
#
# Consider two identical parallel single-mode waveguides centered at
# $y = \pm p/2$ with center-to-center pitch $p$ and edge-to-edge gap
# $g = p - w$. In the weak-to-moderate coupling regime of **Coupled-Mode Theory
# (CMT)** (Yariv 1973 [1]; Haus *et al.* 1987 [2]), the evanescent tail of each
# waveguide perturbs the polarization of its neighbor with coupling coefficient
# $\kappa(\lambda)$, splitting the isolated waveguide propagation constant
# $\beta_0$ into two hybrid **supermodes** of the composite two-waveguide
# cross-section:
#
# 1. **Symmetric (even) fundamental supermode ($\text{TE}_{0,\text{even}}$)**:
#    $E_y^{\text{even}}(-y) = +E_y^{\text{even}}(y)$, with concentrated field in
#    the central gap and higher effective index $n_{\text{even}}(\lambda)$.
# 2. **Antisymmetric (odd) supermode ($\text{TE}_{0,\text{odd}}$)**:
#    $E_y^{\text{odd}}(-y) = -E_y^{\text{odd}}(y)$, with a nodal plane at
#    $y = 0$ and lower effective index $n_{\text{odd}}(\lambda) < n_{\text{even}}(\lambda)$.
#
# Launching unit modal amplitude into the lower waveguide (port 1, `in_bot`) at
# the entrance of the coupling section excites an equal superposition of the
# even and odd supermodes. Over a straight interaction length $L_c$ plus
# symmetric input/output S-bend transitions of axial length $L_{\text{sb}}$, the
# two supermodes accumulate a differential phase $2\theta(\lambda)$:
#
# $$\theta(\lambda) = \frac{\pi}{\lambda}\left[\Delta n_{\text{eff}}(0, \lambda)\,L_c + 2\int_0^{L_{\text{sb}}} \Delta n_{\text{eff}}\!\left(g(x), \lambda\right)\mathrm{d}s\right] = \frac{\pi}{2}\,\frac{L_c + L_{\text{sb,eff}}(\lambda)}{L_\pi(\lambda)},$$
#
# where $\Delta n_{\text{eff}} = n_{\text{even}} - n_{\text{odd}}$ is the
# supermode index splitting and $L_\pi(\lambda)$ is the **crossover beat length**
# (coupling length for $100\%$ power transfer):
#
# $$L_\pi(\lambda) = \frac{\pi}{\beta_{\text{even}} - \beta_{\text{odd}}} = \frac{\lambda}{2\left[n_{\text{even}}(\lambda) - n_{\text{odd}}(\lambda)\right]}, \qquad L_{\text{3dB}}(\lambda) = \frac{L_\pi(\lambda)}{2}.$$
#
# In the Wakis $\exp(+\mathrm{i}\omega t - \mathrm{i}\beta x)$ phasor convention,
# the lossless $2 \times 2$ transmission transfer matrix connecting the input
# amplitudes $(a_{\text{top}}, a_{\text{bot}})^{\mathsf{T}}$ (ports 2 and 1) to
# the output amplitudes $(b_{\text{top}}, b_{\text{bot}})^{\mathsf{T}}$ (ports 3
# and 4) is:
#
# $$\begin{pmatrix} b_{\text{top}} \\ b_{\text{bot}} \end{pmatrix} = \begin{pmatrix} S_{32} & S_{31} \\ S_{42} & S_{41} \end{pmatrix} \begin{pmatrix} a_{\text{top}} \\ a_{\text{bot}} \end{pmatrix} = e^{-\mathrm{i}\bar{\phi}(\lambda)} \begin{pmatrix} \cos\theta(\lambda) & -\mathrm{i}\sin\theta(\lambda) \\ -\mathrm{i}\sin\theta(\lambda) & \cos\theta(\lambda) \end{pmatrix} \begin{pmatrix} a_{\text{top}} \\ a_{\text{bot}} \end{pmatrix}.$$
#
# For single-port excitation of the lower input arm ($a_{\text{bot}} = 1, a_{\text{top}} = 0$),
# the cross-port (`out_top`, port 3) and bar-port (`out_bot`, port 4) power
# transmittances obey:
#
# $$T_{\text{cross}}(L_c, \lambda) = |S_{31}|^2 = \eta(\lambda)\,\sin^2\!\left(\frac{\pi}{2}\,\frac{L_c + L_{\text{sb,eff}}(\lambda)}{L_\pi(\lambda)}\right), \qquad T_{\text{bar}}(L_c, \lambda) = |S_{41}|^2 = \eta(\lambda)\,\cos^2\!\left(\frac{\pi}{2}\,\frac{L_c + L_{\text{sb,eff}}(\lambda)}{L_\pi(\lambda)}\right),$$
#
# where $\eta(\lambda) = T_{\text{cross}} + T_{\text{bar}} \le 1$ is the guided
# transmission efficiency. At the **50:50 ($3\;\text{dB}$) crossing wavelength**
# $\lambda_{\text{cross}}$ where $\theta(\lambda_{\text{cross}}) = \pi/4$, the
# cross-coupled wave $S_{31}$ lags the bar wave $S_{41}$ by exactly $\pi/2$
# ($\arg(S_{31}/S_{41}) = -\pi/2$). Driving both input arms with equal power and
# a $\Delta\phi = \mp\pi/2$ phase offset therefore causes complete constructive
# interference in one output waveguide and destructive cancellation in the other:
#
# $$P_{\text{up}}(\Delta\phi) = \frac{1}{2}\left|e^{\mathrm{i}\Delta\phi}\cos\theta - \mathrm{i}\sin\theta\right|^2, \qquad P_{\text{down}}(\Delta\phi) = \frac{1}{2}\left|-\mathrm{i}e^{\mathrm{i}\Delta\phi}\sin\theta + \cos\theta\right|^2.$$

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
import numpy as np
from scipy.constants import c as c_light, epsilon_0

from wakis import figstyle
from wakis.templates.photonics import (
    PICSimulation,
    PolySlab,
    Port,
    make_movie,
    n_to_eps,
    plot_cross_section_mesh,
    plot_field_map,
    plot_layout,
    plot_neff_dispersion,
    plot_port_modes,
    plot_spectra_db,
    sbend,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _helpers import (  # noqa: E402
    NonUniformCouplerPIC,
    build_coupler_graded_axes,
    build_coupler_mesh_axes,
    coupler_mesh_overrides,
    extract_mesh_metrics_from_grid,
    find_crossing_wavelength,
    overlay_coupler_geometry,
    plot_asymmetric_eim_stack,
    plot_mesh_upgrade_comparison_coupler,
    render_coupler_field_map_with_overlay,
    render_coupler_movie_with_overlay,
    run_phased_dual_drive,
    save_coupler_setup_figures,
    solve_coupler_supermodes,
    trapezoidal_effective_width,
)

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# %% [markdown]
# ## 2. Device specification & trapezoidal sidewall geometry
#
# We model an air-clad silicon-on-insulator directional coupler
# (Chrostowski & Hochberg 2015 [3]):
#
# | Parameter | Symbol | Value |
# |---|---|---|
# | Silicon waveguide refractive index | $n_{\text{wg}}$ | $3.48$ ($\varepsilon_r = 12.1104$) |
# | $\text{SiO}_2$ buried-oxide substrate index ($z < 0$) | $n_{\text{sub}}$ | $1.45$ ($\varepsilon_r = 2.1025$) |
# | Upper / lateral air cladding index ($z > 0$) | $n_{\text{clad}}$ | $1.00$ ($\varepsilon_r = 1.0$) |
# | Silicon core thickness | $h_{\text{wg}}$ | $220\;\text{nm}$ ($z \in [0, 0.22]\;\mu\text{m}$) |
# | Nominal GDS path width / mask dilation | $w_{\text{GDS}}, d_{\text{mask}}$ | $450\;\text{nm}, +20\;\text{nm}$ ($w_{\text{bot}} = 470\;\text{nm}$) |
# | Baseline sidewall slant angle (bottom ref.) | $\theta_{\text{sw}}$ | $30^\circ$ ($\pi/6\;\text{rad}$, $w_{\text{eff}} = 343.0\;\text{nm}$) |
# | Coupling section center-to-center pitch | $p$ | $550\;\text{nm}$ ($y = \pm 0.275\;\mu\text{m}$) |
# | Baseline coupling length | $L_c$ | $4.0\;\mu\text{m}$ ($x \in [-2.0, +2.0]\;\mu\text{m}$) |
# | S-bend transition span | $\Delta x_{\text{sb}}, \Delta y_{\text{sb}}$ | $6.0\;\mu\text{m}, 1.725\;\mu\text{m}$ ($y_{\text{port}} = \pm 2.0\;\mu\text{m}$) |
# | Input / output port axial coordinates | $x_{\text{in}}, x_{\text{out}}$ | $-8.15\;\mu\text{m}, +8.15\;\mu\text{m}$ |
# | Wavelength band ($21$ linear frequencies) | $[\lambda_{\min}, \lambda_{\max}]$ | $[1.40, 1.70]\;\mu\text{m}$ ($\lambda_0 \approx 1.5355\;\mu\text{m}$) |
#
# ### Simulation setup schematics and configuration summary
#
# Before launching the time-domain solver, `save_coupler_setup_figures` calls
# `wakis.schematic.save_setup_figures` on the configured 2D EIM solver to
# generate the labelled setup schematics and Markdown configuration summary:
#
# - `figures/00a_setup_geometry.png` — Domain extents ($17.8\;\mu\text{m} \times 6.49\;\mu\text{m}$
#   including $10$-cell CPML collars on $\pm x, \pm y$ and periodic $\pm z$),
#   air/$\text{SiO}_2$ background ($n_{\text{clad}} = 1.00$, $n_{\text{sub}} = 1.45$)
#   and 2D EIM silicon waveguide arms ($n_{\text{Si}} = 3.48$, $h_{\text{wg}} = 220\;\text{nm} \to n_{\text{eff,slab}} = 2.8424$,
#   $\varepsilon_r = 8.08$, $w_{\text{eff}} = 343.0\;\text{nm}$, $g_{\text{eff}} = 207.02\;\text{nm}$),
#   the fundamental $\text{TE}_0$ (`Ey`, `Hz`) `ModeSource` on port 1 (`in_bot`)
#   with its transverse mode profile $|E_y(y)|$ and $+x$ launch arrow, the four
#   port `ModeMonitor` planes (`in_bot` $S_{11}$, `in_top` $S_{21}$, `out_top`
#   cross $S_{31}$, `out_bot` bar $S_{41}$), and the 4-wall Poynting `FluxMonitor`
#   box.
# - `figures/00b_setup_mesh_and_excitation.png` — Full-domain Yee grid overlay
#   with a $1\times$ zoom inset on the $g_{\text{eff}} = 207.02\;\text{nm}$
#   central coupling slot ($8.55\;\text{cells}$ across the gap at $L_3 = (20, 8)$,
#   $10.00\;\text{cells}$ on the graded non-uniform mesh), propagating wave-packet
#   cartoon, and the broadband Gaussian excitation pulse ($f_0 = 195.2\;\text{THz}$,
#   $\Delta f = 18.89\;\text{THz}$, $\lambda \in [1.40, 1.70]\;\mu\text{m}$) in
#   the time and frequency domains.
# - `figures/00c_quadrature_setup_schematic.png` — Separate setup schematic for
#   the dual-port $\pi/2$ phase-quadrature routing run (`task_9`,
#   `quadrature_dual_drive`), where both `in_bot` ($\phi_1 = 0$) and `in_top`
#   ($\phi_2 = -\pi/2$) are driven simultaneously by phase-locked
#   `PhasedGaussianSource` eigenmodes.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | $L_x \times L_y \times L_z = 17.8\;\mu\text{m} \times 6.49\;\mu\text{m} \times 24.6\;\text{nm}$ ($z \in [97.7\;\text{nm}, 122.3\;\text{nm}]$) |
# | Domain | Grid cells | $N_x \times N_y \times N_z = 725 \times 266 \times 1$ ($192{,}850\;\text{cells}$ baseline $L_3$; $567 \times 211 \times 1 = 119{,}637\;\text{cells}$ graded non-uniform $L_3$) |
# | Discretization | $\Delta x$ & resolution | $\Delta_{\min} = 24.2\;\text{nm}$ ($20.3\;\text{nm}$ graded), $22.2\;\text{cells}/\lambda$ (in $n_{\max} = 2.84$), $8.6\;\text{cells}$ ($10.0\;\text{cells}$ graded) across feature ($207\;\text{nm}$) |
# | Boundary | $-x, +x$ faces | CPML ($10\;\text{cells}$, $245.3\;\text{nm}$) |
# | Boundary | $-y, +y$ faces | CPML ($10\;\text{cells}$, $244.4\;\text{nm}$) |
# | Boundary | $-z, +z$ faces | Periodic ($1\;\text{cell}$, 2D EIM $\text{TE}_0$: $E_x, E_y, H_z$) |
# | Material | Background | Air lateral/upper clad ($n = 1.00, \varepsilon_r = 1.00$; $\text{SiO}_2$ BOX $n_{\text{sub}} = 1.45$, isotropic) |
# | Material | Regions 1 & 2 (upper & lower coupler arms) | Si EIM core ($n_{\text{Si}} = 3.48, h_{\text{wg}} = 220\;\text{nm} \to n_{\text{eff,slab}} = 2.8424, \varepsilon_r = 8.08, \mu_r = 1.00, w_{\text{eff}} = 343\;\text{nm}$) |
# | Source | 1 · `ModeSource` | Port 1 `in_bot` ($x = -8.27\;\mu\text{m}, y = -2.00\;\mu\text{m}$), pol: $\text{TE}_0$ ($E_y, H_z$), pulse: $f_0 = 195.2\;\text{THz}, \Delta f = 18.89\;\text{THz}$ |
# | Monitor | 2–5 · `ModeMonitor` | Port 1 `in_bot` ($S_{11}$), Port 3 `out_top` (Cross $S_{31}$), Port 4 `out_bot` (Bar $S_{41}$), Port 2 `in_top` ($S_{21}$) at $x = \mp 8.15\;\mu\text{m}, y = \pm 2.00\;\mu\text{m}$ |
# | Monitor | 6 · `FluxMonitor` | 4-wall Poynting `FluxMonitor` box ($x \in [-8.85, 8.85]\;\mu\text{m}, y \in [-3.20, 3.20]\;\mu\text{m}$) |
# | Variation | Dual-port quadrature routing (`task_9`) | Simultaneous `ModeSource` on `in_bot` ($\phi = 0$) and `in_top` ($\phi = -\pi/2$) via `PhasedGaussianSource` |
# | Variation | Coupling-length & sidewall sweeps (`task_10`..`task_19`) | $L_c \in \{0, 2, 4, 6, 8\}\;\mu\text{m}$ at $\theta_{\text{sw}} = 30^\circ$; $\theta_{\text{sw}} \in \{0^\circ, 5^\circ, \dots, 30^\circ\}$ ($g_{\text{eff}} \in [80.0, 207.0]\;\text{nm}$) |
# | Variation | Graded non-uniform Cartesian mesh (`task_20`) | Mirror-symmetric graded $\Delta y$ ($10.00\;\text{cells}$ across $g_{\text{eff}}$) & non-square $\Delta x > \Delta y_{\min}$ ($119{,}637\;\text{cells}$ vs $193{,}842$ at $L_3$, $-38.3\%$) |

# %%
# Material refractive indices (non-dispersive benchmark specification)
N_WG = 3.48  # Silicon core (eps_r = 12.1104)
N_SUB = 1.45  # SiO2 buried oxide substrate (eps_r = 2.1025)
N_CLAD = 1.00  # Air upper / trench cladding
H_WG = 0.22e-6  # 220 nm waveguide thickness

# Wavelength band: 1.40 um to 1.70 um (21 linearly spaced frequencies)
WL_MIN = 1.40e-6
WL_MAX = 1.70e-6
F_MIN = c_light / WL_MAX
F_MAX = c_light / WL_MIN
F_0 = 0.5 * (F_MIN + F_MAX)
WL_0 = c_light / F_0  # ~1.53548 um
WL_DESIGN = 1.53e-6  # Baseline 50:50 crossing wavelength target
N_FREQS = 21

# Geometric layout & fabrication parameters
W_GDS = 0.45e-6  # Nominal GDS path width
DILATION = 0.02e-6  # Mask bias (+20 nm -> w_bot = 470 nm)
THETA_SW_BASE_DEG = 30.0  # Baseline sidewall angle (pi/6 rad = 30 deg)
THETA_SW_SWEEP_DEG = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)

PITCH_COUPLED = 0.55e-6  # Center-to-center spacing in coupling region (y = +-0.275 um)
Y_COUPLED = 0.5 * PITCH_COUPLED  # 0.275 um
Y_PORT = 2.00e-6  # Input/output arm centerlines (y = +-2.0 um)
DY_SBEND = Y_PORT - Y_COUPLED  # 1.725 um S-bend lateral offset
DX_SBEND = 6.00e-6  # 6.0 um S-bend axial length
L_COUPLED_BASE = 4.00e-6  # 4.0 um straight interaction section
L_COUPLED_SWEEP = (0.0, 2.0e-6, 4.0e-6, 6.0e-6, 8.0e-6)
L_LEAD = 2.50e-6  # Straight lead extension into CPML

# Baseline 2D simulation domain: Lx = 17.3 um, Ly = 6.0 um
DOMAIN_2D_BASE = (-8.65e-6, 8.65e-6, -3.00e-6, 3.00e-6)

# %% [markdown]
# ## 3. Asymmetric 2D Effective-Index Method (EIM) reduction
#
# In the vertical direction $z$, the etched silicon waveguide forms an
# **asymmetric 3-layer slab** bounded below by the $\text{SiO}_2$ buried oxide
# ($n_{\text{sub}} = 1.45$) and above by air ($n_{\text{clad}} = 1.00$). Solving
# the 1D asymmetric TE slab dispersion relation (`slab_neff` with `n_sub=1.45`)
# at $\lambda_{\text{ref}} = 1.5355\;\mu\text{m}$ yields:
#
# $$n_{\text{eff,slab}}(1.5355\;\mu\text{m}) = 2.8424.$$
#
# For a slanted sidewall angle $\theta_{\text{sw}}$ measured from vertical with
# bottom reference plane (`reference_plane = "bottom"`), the waveguide width
# varies linearly with height $z \in [0, h_{\text{wg}}]$:
#
# $$w(z) = w_{\text{bot}} - 2 z \tan\theta_{\text{sw}}, \qquad w_{\text{bot}} = w_{\text{GDS}} + d_{\text{mask}} = 470\;\text{nm}.$$
#
# Projecting the trapezoidal cross-section onto the 2D mid-plane
# $z = h_{\text{wg}}/2 = 110\;\text{nm}$ gives the area-equivalent effective
# width and coupling gap:
#
# $$w_{\text{eff}}(\theta_{\text{sw}}) = w_{\text{bot}} - h_{\text{wg}}\tan\theta_{\text{sw}}, \qquad g_{\text{eff}}(\theta_{\text{sw}}) = p - w_{\text{eff}}(\theta_{\text{sw}}).$$
#
# At the baseline $\theta_{\text{sw}} = 30^\circ$, the top width shrinks by
# $2 h_{\text{wg}}\tan 30^\circ = 254.0\;\text{nm}$ to $w_{\text{top}} = 216.0\;\text{nm}$,
# giving mid-height width $w_{\text{eff}}(30^\circ) = 343.0\;\text{nm}$ and
# mid-height coupling gap $g_{\text{eff}}(30^\circ) = 207.02\;\text{nm}$.

# %% [markdown]
# ## 4. Parametric 4-port coupler construction
#
# Each coupler arm is constructed as a single continuous counter-clockwise
# `PolySlab` joining the input straight lead, the incoming cosine `sbend`
# ($\Delta x = 6.0\;\mu\text{m}, \Delta y = \mp 1.725\;\mu\text{m}$), the
# straight parallel interaction region of length $L_c$, the outgoing cosine
# `sbend`, and the output straight lead. Mirroring the upper arm across $y = 0$
# (`y -> -y` with reversed vertex winding) guarantees exact $y \mapsto -y$
# symmetry on the Yee grid.


# %%
def build_directional_coupler(
    L_c=L_COUPLED_BASE,
    theta_sw_deg=THETA_SW_BASE_DEG,
    w_gds=W_GDS,
    dilation=DILATION,
    n_sbend=90,
    four_ports=True,
    dx_sbend=DX_SBEND,
    dy_sbend=DY_SBEND,
    l_lead=L_LEAD,
    y_max=3.0e-6,
):
    """
    Build the symmetric 2-arm SOI directional coupler as two seamless
    ``PolySlab`` structures along with its waveguide ``Port`` list and 2D domain.

    Parameters
    ----------
    L_c : float, optional
        Straight parallel coupling length [m]. Default is ``4.0e-6``.
    theta_sw_deg : float, optional
        Sidewall slant angle [deg] from vertical. Default is ``30.0``.
    w_gds : float, optional
        Nominal GDS waveguide path width [m]. Default is ``0.45e-6``.
    dilation : float, optional
        Mask bias / dilation [m]. Default is ``0.02e-6``.
    n_sbend : int, optional
        Number of discretization vertices along each cosine S-bend.
    four_ports : bool, optional
        If True, attach all 4 ports (``in_bot``, ``in_top``, ``out_top``,
        ``out_bot``); if False, attach the 3 ports needed for single-drive
        characterization (``in_bot``, ``out_top``, ``out_bot``).
    dx_sbend, dy_sbend : float, optional
        S-bend axial span and lateral offset [m].
    l_lead : float, optional
        Straight lead extension into CPML [m].
    y_max : float, optional
        Half-width of the transverse domain :math:`[-y_{\\max}, +y_{\\max}]` [m].

    Returns
    -------
    structures : list of PolySlab
        ``[slab_top, slab_bot]`` with 2D mid-height effective width ``w_eff``.
    ports : list of Port
        Named waveguide ports at :math:`x = \\pm (L_c/2 + \\Delta x_{\\mathrm{sb}})`.
    domain : tuple of float
        ``(xmin, xmax, ymin, ymax)`` [m] simulation domain bounds.
    geom_info : dict of float
        Trapezoidal width/gap dictionary from ``trapezoidal_effective_width``.
    """
    geom_info = trapezoidal_effective_width(
        w_gds=w_gds,
        dilation=dilation,
        thickness=H_WG,
        sidewall_angle_deg=theta_sw_deg,
        pitch=PITCH_COUPLED,
    )
    w_eff = geom_info["w_eff"]
    hw = 0.5 * w_eff
    dx_sb = float(dx_sbend)
    dy_sb = float(dy_sbend)
    y_port = Y_COUPLED + dy_sb

    x_c0, x_c1 = -0.5 * float(L_c), 0.5 * float(L_c)
    x_sb0, x_sb1 = x_c0 - dx_sb, x_c1 + dx_sb
    x_in, x_out = x_sb0 - float(l_lead), x_sb1 + float(l_lead)

    sb_in_top = sbend(
        (x_sb0, y_port),
        dx=dx_sb,
        dy=-dy_sb,
        width=w_eff,
        profile="cosine",
        n=n_sbend,
    )
    sb_out_top = sbend(
        (x_c1, Y_COUPLED),
        dx=dx_sb,
        dy=+dy_sb,
        width=w_eff,
        profile="cosine",
        n=n_sbend,
    )

    verts_top = np.vstack(
        [
            [[x_in, y_port - hw]],
            sb_in_top[:n_sbend],
            sb_out_top[:n_sbend] if L_c > 0 else sb_out_top[1:n_sbend],
            [[x_out, y_port - hw], [x_out, y_port + hw]],
            sb_out_top[n_sbend:],
            sb_in_top[n_sbend:] if L_c > 0 else sb_in_top[n_sbend + 1 :],
            [[x_in, y_port + hw]],
        ]
    )
    verts_bot = verts_top[::-1].copy()
    verts_bot[:, 1] *= -1.0

    eps_core = n_to_eps(N_WG)
    slab_top = PolySlab(verts_top, z_min=0.0, z_max=H_WG, material=eps_core)
    slab_bot = PolySlab(verts_bot, z_min=0.0, z_max=H_WG, material=eps_core)

    x_port = 0.5 * float(L_c) + dx_sb + 0.15e-6
    z_mid = 0.5 * H_WG
    port_size = (0.0, min(1.8e-6, 1.6 * dy_sb), 1.0e-6)
    ports = [
        Port("in_bot", center=(-x_port, -y_port, z_mid), size=port_size, direction=+1),
        Port(
            "out_top",
            center=(+x_port, +y_port, z_mid),
            size=port_size,
            direction=+1,
        ),
        Port(
            "out_bot",
            center=(+x_port, -y_port, z_mid),
            size=port_size,
            direction=+1,
        ),
    ]
    if four_ports:
        ports.append(
            Port(
                "in_top",
                center=(-x_port, +y_port, z_mid),
                size=port_size,
                direction=+1,
            )
        )
    domain = (-(x_port + 0.5e-6), +(x_port + 0.5e-6), -float(y_max), +float(y_max))
    return [slab_top, slab_bot], ports, domain, geom_info


# %% [markdown]
# ## 5. Study tasks: resolution, coupling length, sidewall angle & quadrature
#
# ### Resolution: cells per wavelength and cells across the smallest feature
#
# Accurate full-wave modeling of an evanescent directional coupler requires
# satisfying two independent spatial discretization rules in every simulation
# mode (`--quick`, `--full`, parameter sweeps, and automated tests):
#
# 1. **Wavelength sampling rule ($\text{cpw} \ge 10$)**: The background grid
#    must place at least $10$ cells per wavelength inside the highest-index
#    guided medium (`cells_per_wavelength` in `PICSimulation` counts cells per
#    $\lambda_{\min} / n_{\text{eff,slab}} \approx 492\;\text{nm}$, giving a
#    background cell size $\Delta x_{\text{bg}} \approx 49.1\;\text{nm}$ at
#    $10\;\text{cpw}$, $35.1\;\text{nm}$ at $14\;\text{cpw}$, $24.6\;\text{nm}$
#    at $20\;\text{cpw}$, and $12.3\;\text{nm}$ at $40\;\text{cpw}$).
# 2. **Minimum-feature rule ($N \ge 4$ cells across the coupling gap)**: The
#    grid cell size across the device's smallest geometric feature must never
#    exceed $d_{\min} / 4$. In the 2D effective-width directional coupler, the
#    smallest feature of each configuration is its mid-height coupling gap
#    $g_{\text{eff}}(\theta_{\text{sw}}) = p - w_{\text{eff}}(\theta_{\text{sw}})$,
#    which narrows from $207.02\;\text{nm}$ at $\theta_{\text{sw}} = 30^\circ$
#    down to $80.00\;\text{nm}$ for vertical sidewalls ($\theta_{\text{sw}} = 0^\circ$).
#    At $10\text{--}14\;\text{cpw}$, a uniform background grid has
#    $\Delta y_{\text{bg}} \approx 35\text{--}49\;\text{nm}$, leaving only
#    $1.6\text{--}2.3$ cells across an $80\;\text{nm}$ vertical-sidewall gap.
#
# Rather than over-refining the entire $17.3 \times 6.0\;\mu\text{m}^2$ domain,
# `coupler_mesh_overrides` passes a targeted `mesh_overrides=[(lo, hi, (h_max, h_max, 1.0))]`
# box to `PICSimulation` with $h_{\max} = g_{\text{eff}}(\theta_{\text{sw}}) / N$.
# Axially, the box covers the parallel coupling section ($|x| \le L_c/2$) plus
# the S-bend transition ends where the local gap $g(x) = g_{\text{eff}} + \Delta y_{\text{sb}}[1 - \cos(\pi u)]$
# is below $3 g_{\text{eff}}$ ($|x| \le L_c/2 + u_{3\times}\Delta x_{\text{sb}}$).
# Transversely, the box spans $|y| \le 1.5 g_{\text{eff}}$ (symmetric about
# $y = 0$), capping the cell size across the coupling slot to $\le g_{\text{eff}} / N$
# and grading smoothly back out to $\Delta x_{\text{bg}}, \Delta y_{\text{bg}}$
# at a step-to-step growth ratio $\le 1.4$. Setting the $z$ override component
# to $1.0\;\text{m} \gg h_{\text{wg}}$ preserves the single-cell 2D grid
# ($N_z = 1$).
#
# Each resolution level is specified by the pair $(\text{cpw}, N)$:
#
# - **$L_1 = (10, 4)$**: $10\;\text{cpw}$ background with $N = 4$ cells across $g_{\text{eff}}$.
# - **$L_2 = (14, 6)$**: $14\;\text{cpw}$ background with $N = 6$ cells across $g_{\text{eff}}$ (primary level and sweep resolution in `--quick`).
# - **$L_3 = (20, 8)$**: $20\;\text{cpw}$ background with $N = 8$ cells across $g_{\text{eff}}$ (accepted primary level and sweep resolution in `--full`; reference in `--quick`).
# - **$R = (40, 16)$**: $40\;\text{cpw}$ background with $N = 16$ cells across $g_{\text{eff}}$ (independent Kottke reference in `--full`).
# - **Under-resolved demonstration pair (`underresolved_kottke`, `underresolved_staircase`)**:
#   $10\;\text{cpw}$ with `mesh_overrides=None`, retained in both `--quick` and
#   `--full` as an explicit warning example on the spectra and convergence
#   plots (and excluded from the converged trends).
#
# > **Resolution audit (`cpw` in 2D EIM vs 3D bulk core and `cells_across_feature`)**:
# > In `PICSimulation(..., two_d=True)`, `cells_per_wavelength` (`cpw` in `data/task_*.json`)
# > is defined with respect to the highest-index medium on the 2D grid, the EIM slab
# > core $n_{\text{eff,slab}} = 2.8424$ ($\lambda_{\min}/n_{\text{eff,slab}} = 492.54\;\text{nm}$).
# > Every non-demo task in `data/task_*.json` satisfies $\text{cpw} \ge 10$
# > ($10$ at $L_1$, $14$ at $L_2$, $20$ at $L_3$ and all parameter sweeps, $40$ at $R$)
# > and $\text{cells\_across\_feature} \ge 4.0$ ($4.28$ at $L_1$, $6.02$ at $L_2$,
# > $8.00\text{--}8.55$ at $L_3$ across all length and sidewall-angle sweeps, and
# > $16.95$ at $R$). If $\Delta x_{\text{bg}}$ is instead normalized against the
# > 3D bulk silicon index $n_{\text{wg}} = 3.48$ ($\lambda_{\min}/n_{\text{wg}} = 402.30\;\text{nm}$),
# > $L_2$ ($11.46\;\text{cpw}_{3\text{D}}$), the accepted `--full` primary/sweep
# > level $L_3$ ($16.39\;\text{cpw}_{3\text{D}}$), and $R$ ($32.71\;\text{cpw}_{3\text{D}}$)
# > all exceed $10\;\text{cpw}_{3\text{D}}$, while the coarsest convergence-ladder
# > point $L_1$ ($\Delta x_{\text{bg}} = 49.06\;\text{nm}$) corresponds to
# > $8.20\;\text{cpw}_{3\text{D}}$.
#
# ### `--full` task table and measured mesh / runtime metrics (`data/task_0`..`task_20`)
#
# | Task | Name | Averaging | Level | $\text{cpw}$ | $N$ | $g_{\text{eff}}\;[\text{nm}]$ | $\Delta x_{\text{bg}}\;[\text{nm}]$ | $\Delta x_{\min}\;[\text{nm}]$ | Cells across gap | $N_{\text{cells}}$ | $N_{\text{steps}}$ | Time $[\text{s}]$ | $\lambda_{\text{cross}}\;[\text{nm}]$ |
# |---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | `0` | `underresolved_kottke` | Kottke | demo | $10$ | none | $207.02$ | $49.18$ | $49.15$ | $4.21$ | $53{,}339$ | $9{,}851$ | $53.1$ | $1505.55$ |
# | `1` | `underresolved_staircase` | Staircase | demo | $10$ | none | $207.02$ | $49.18$ | $49.15$ | $4.21$ | $53{,}339$ | $13{,}351$ | $78.0$ | $1464.16$ |
# | `2` | `kottke_L1` | Kottke | $L_1$ | $10$ | $4$ | $207.02$ | $49.06$ | $48.41$ | $4.28$ | $53{,}856$ | $9{,}801$ | $52.6$ | $1495.47$ |
# | `3` | `staircase_L1` | Staircase | $L_1$ | $10$ | $4$ | $207.02$ | $49.06$ | $48.41$ | $4.28$ | $53{,}856$ | $12{,}801$ | $82.6$ | $1536.88$ |
# | `4` | `kottke_L2` | Kottke | $L_2$ | $14$ | $6$ | $207.02$ | $35.09$ | $34.37$ | $6.02$ | $99{,}974$ | $9{,}401$ | $111.0$ | $1496.13$ |
# | `5` | `staircase_L2` | Staircase | $L_2$ | $14$ | $6$ | $207.02$ | $35.09$ | $34.37$ | $6.02$ | $99{,}974$ | $16{,}951$ | $189.5$ | $1567.05$ |
# | `6` | `kottke_L3` (primary) | Kottke | $L_3$ | $20$ | $8$ | $207.02$ | $24.55$ | $24.20$ | $8.55$ | $193{,}842$ | $12{,}401$ | $346.5$ | $1497.56$ |
# | `7` | `staircase_L3` | Staircase | $L_3$ | $20$ | $8$ | $207.02$ | $24.55$ | $24.20$ | $8.55$ | $193{,}842$ | $19{,}001$ | $424.1$ | $1456.62$ |
# | `8` | `kottke_reference` | Kottke | $R$ | $40$ | $16$ | $207.02$ | $12.30$ | $12.22$ | $16.95$ | $728{,}280$ | $24{,}501$ | $2011.3$ | $1494.79$ |
# | `9` | `quadrature_dual_drive` | Kottke | $L_3$ | $20$ | $8$ | $207.02$ | $24.55$ | $24.20$ | $8.55$ | $193{,}842$ | $12{,}401$ | $338.0$ | — ($99.65\%$ up) |
# | `10` | `length_L0um` | Kottke | $L_3$ | $20$ | $8$ | $207.02$ | $24.57$ | $24.20$ | $8.55$ | $150{,}321$ | $10{,}101$ | $180.2$ | $>1700$ ($T_{\times}=0.0715$) |
# | `11` | `length_L2um` | Kottke | $L_3$ | $20$ | $8$ | $207.02$ | $24.62$ | $24.20$ | $8.55$ | $171{,}948$ | $11{,}251$ | $235.8$ | $1606.15$ |
# | `12` | `length_L6um` | Kottke | $L_3$ | $20$ | $8$ | $207.02$ | $24.58$ | $24.20$ | $8.55$ | $215{,}469$ | $13{,}551$ | $355.2$ | $1424.95$ |
# | `13` | `length_L8um` | Kottke | $L_3$ | $20$ | $8$ | $207.02$ | $24.60$ | $24.20$ | $8.55$ | $237{,}096$ | $14{,}751$ | $409.9$ | $1649.31$ ($3\pi/4$) |
# | `14` | `sw_angle_00deg` | Kottke | $L_3$ | $20$ | $8$ | $80.00$ | $24.54$ | $10.00$ | $8.00$ | $358{,}231$ | $23{,}451$ | $958.0$ | $1572.66$ |
# | `15` | `sw_angle_05deg` | Kottke | $L_3$ | $20$ | $8$ | $99.25$ | $24.62$ | $12.36$ | $8.03$ | $301{,}126$ | $19{,}551$ | $670.5$ | $1594.59$ |
# | `16` | `sw_angle_10deg` | Kottke | $L_3$ | $20$ | $8$ | $118.79$ | $24.54$ | $14.68$ | $8.09$ | $266{,}100$ | $17{,}101$ | $525.6$ | $1600.09$ |
# | `17` | `sw_angle_15deg` | Kottke | $L_3$ | $20$ | $8$ | $138.95$ | $24.58$ | $17.21$ | $8.07$ | $238{,}210$ | $15{,}251$ | $412.4$ | $1591.67$ |
# | `18` | `sw_angle_20deg` | Kottke | $L_3$ | $20$ | $8$ | $160.07$ | $24.53$ | $19.95$ | $8.00$ | $218{,}508$ | $13{,}801$ | $330.4$ | $1570.60$ |
# | `19` | `sw_angle_25deg` | Kottke | $L_3$ | $20$ | $8$ | $182.59$ | $24.59$ | $22.50$ | $8.11$ | $201{,}420$ | $12{,}851$ | $296.9$ | $1537.51$ |
# | `20` | `kottke_nonuniform` | Kottke | $L_3$ (graded) | $20$ | $8$ | $207.02$ | $51.26$ | $20.31$ | $10.00$ | $119{,}637$ | $13{,}101$ | $162.9$ | $1495.81$ |
#
# **Note on `--quick` mode (`task_0`..`task_18`)**: Passing `--quick` omits
# `kottke_L3` / `staircase_L3` from the resolution ladder, uses `kottke_L2`
# ($L_2 = (14, 6)$, $97{,}146$ cells with $n_{\text{pml}} = 8$,
# $\text{energy\_decay} = 10^{-3}$, $\lambda_{\text{cross}} = 1496.44\;\text{nm}$)
# as the primary run and sweep resolution ($6.02\text{--}6.12$ cells across
# $g_{\text{eff}}$), and uses $L_3 = (20, 8)$ ($189{,}886$ cells,
# $\lambda_{\text{cross}} = 1497.70\;\text{nm}$) as `kottke_reference`.


# %%
def get_task_table(quick=False):
    """
    Return the study task manifest (under-resolved demo pair + resolution ladder
    + reference run + dual-port quadrature routing + coupling-length sweep +
    sidewall-angle sweep).
    """
    ladder = (("L1", 10, 4), ("L2", 14, 6)) if quick else (("L1", 10, 4), ("L2", 14, 6), ("L3", 20, 8))
    primary_lvl = "L2" if quick else "L3"
    ref_lvl, ref_cpw, ref_n_gap = ("L3", 20, 8) if quick else ("R", 40, 16)
    swp_lvl, swp_cpw, swp_n_gap = ("L2", 14, 6) if quick else ("L3", 20, 8)
    n_pml = 8 if quick else 10
    ed = 1e-3 if quick else 1e-4
    tasks = []

    # Under-resolved demonstration pair (10 cpw, no mesh overrides)
    for avg in ("kottke", "staircase"):
        tasks.append(
            {
                "index": len(tasks),
                "name": f"underresolved_{avg}",
                "kind": "underresolved_demo",
                "level": "underresolved",
                "averaging": avg,
                "cpw": 10,
                "n_cells_gap": None,
                "L_c": L_COUPLED_BASE,
                "theta_sw_deg": THETA_SW_BASE_DEG,
                "n_pml": n_pml,
                "energy_decay": ed,
                "collect_fields": False,
                "record_movie": False,
                "movie_frames": 0,
                "description": (
                    f"Under-resolved demo ({avg}, cpw=10, no mesh overrides)"
                ),
            }
        )

    # Resolution ladder: L1=(10,4), L2=(14,6) [and L3=(20,8) in --full]
    for lvl, cpw, n_gap in ladder:
        for avg in ("kottke", "staircase"):
            idx = len(tasks)
            is_primary = lvl == primary_lvl and avg == "kottke"
            tasks.append(
                {
                    "index": idx,
                    "name": f"{avg}_{lvl}",
                    "kind": "resolution",
                    "level": lvl,
                    "averaging": avg,
                    "cpw": cpw,
                    "n_cells_gap": n_gap,
                    "L_c": L_COUPLED_BASE,
                    "theta_sw_deg": THETA_SW_BASE_DEG,
                    "n_pml": n_pml,
                    "energy_decay": ed,
                    "collect_fields": is_primary,
                    "record_movie": is_primary,
                    "movie_frames": 30 if quick else 48,
                    "description": (
                        f"2D EIM directional coupler ({avg}, {lvl}=({cpw},{n_gap})"
                        + (", DFT fields + movie + supermodes)" if is_primary else ")")
                    ),
                }
            )

    # Independent Kottke reference run: L3=(20,8) in --quick, R=(40,16) in --full
    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_reference",
            "kind": "reference",
            "level": ref_lvl,
            "averaging": "kottke",
            "cpw": ref_cpw,
            "n_cells_gap": ref_n_gap,
            "L_c": L_COUPLED_BASE,
            "theta_sw_deg": THETA_SW_BASE_DEG,
            "n_pml": n_pml,
            "energy_decay": ed,
            "collect_fields": False,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"2D EIM directional coupler (kottke, reference {ref_lvl}=({ref_cpw},{ref_n_gap}))"
            ),
        }
    )

    # Dual-port quadrature excitation (phase = -pi/2 on upper input arm)
    tasks.append(
        {
            "index": len(tasks),
            "name": "quadrature_dual_drive",
            "kind": "quadrature",
            "level": swp_lvl,
            "averaging": "kottke",
            "cpw": swp_cpw,
            "n_cells_gap": swp_n_gap,
            "L_c": L_COUPLED_BASE,
            "theta_sw_deg": THETA_SW_BASE_DEG,
            "n_pml": n_pml,
            "energy_decay": ed,
            "collect_fields": True,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"Dual-port quadrature excitation (in_bot=0, in_top=-pi/2, {swp_lvl}=({swp_cpw},{swp_n_gap}))"
            ),
        }
    )

    # Coupling-length sweep L_c in {0, 2, 6, 8} um (L_c=4 um is primary kottke_{primary_lvl})
    for L_um in (0, 2, 6, 8):
        tasks.append(
            {
                "index": len(tasks),
                "name": f"length_L{L_um}um",
                "kind": "length_sweep",
                "level": swp_lvl,
                "averaging": "kottke",
                "cpw": swp_cpw,
                "n_cells_gap": swp_n_gap,
                "L_c": float(L_um) * 1e-6,
                "theta_sw_deg": THETA_SW_BASE_DEG,
                "n_pml": n_pml,
                "energy_decay": ed,
                "collect_fields": False,
                "record_movie": False,
                "movie_frames": 0,
                "description": (
                    f"Coupling-length sweep L_c={L_um:.1f} um (kottke, {swp_lvl}=({swp_cpw},{swp_n_gap}))"
                ),
            }
        )

    # Sidewall-angle sweep theta_sw in {0, 5, 10, 15, 20, 25} deg (30 deg is primary kottke_{primary_lvl})
    for sw_deg in (0, 5, 10, 15, 20, 25):
        tasks.append(
            {
                "index": len(tasks),
                "name": f"sw_angle_{sw_deg:02d}deg",
                "kind": "sw_sweep",
                "level": swp_lvl,
                "averaging": "kottke",
                "cpw": swp_cpw,
                "n_cells_gap": swp_n_gap,
                "L_c": L_COUPLED_BASE,
                "theta_sw_deg": float(sw_deg),
                "n_pml": n_pml,
                "energy_decay": ed,
                "collect_fields": False,
                "record_movie": False,
                "movie_frames": 0,
                "description": (
                    f"Sidewall-angle sweep theta_sw={sw_deg} deg (kottke, {swp_lvl}=({swp_cpw},{swp_n_gap}))"
                ),
            }
        )

    # Upgraded mirror-symmetric graded non-uniform Cartesian mesh (NonUniformCouplerPIC)
    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_nonuniform",
            "kind": "nonuniform",
            "level": swp_lvl,
            "averaging": "kottke",
            "cpw": swp_cpw,
            "n_cells_gap": swp_n_gap,
            "nonuniform": True,
            "L_c": L_COUPLED_BASE,
            "theta_sw_deg": THETA_SW_BASE_DEG,
            "n_pml": n_pml,
            "energy_decay": ed,
            "collect_fields": False,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"Graded non-uniform Cartesian mesh (kottke, {swp_lvl}=({swp_cpw},{swp_n_gap}), non-square Δx > Δy_min)"
            ),
        }
    )

    return tasks


def measure_task_mesh(task_cfg):
    """
    Build only the 1D mesh axes for ``task_cfg`` and return resolution metrics
    (``min_feature_nm``, ``dx_background_nm``, ``dx_min_nm``,
    ``cells_across_feature``, ``n_cells``, ``grid_shape``) without running the
    FDTD/FIT solver.
    """
    structures, _, domain, geom_info = build_directional_coupler(
        L_c=task_cfg["L_c"],
        theta_sw_deg=task_cfg["theta_sw_deg"],
        four_ports=False,
    )
    if task_cfg.get("nonuniform", False):
        _, _, _, metrics = build_coupler_graded_axes(
            domain=domain,
            L_c=task_cfg["L_c"],
            gap_eff=geom_info["gap_eff"],
            w_eff=geom_info["w_eff"],
            y_port=Y_PORT,
            dx_sb=DX_SBEND,
            dy_sb=DY_SBEND,
            cpw=task_cfg["cpw"],
            n_cells_gap=task_cfg.get("n_cells_gap", 8),
            n_pml=task_cfg["n_pml"],
        )
        return metrics
    overrides = coupler_mesh_overrides(
        L_c=task_cfg["L_c"],
        gap_eff=geom_info["gap_eff"],
        w_eff=geom_info["w_eff"],
        n_cells_gap=task_cfg.get("n_cells_gap"),
        dx_sb=DX_SBEND,
        dy_sb=DY_SBEND,
    )
    _, _, metrics = build_coupler_mesh_axes(
        structures=structures,
        domain=domain,
        gap_eff=geom_info["gap_eff"],
        cpw=task_cfg["cpw"],
        mesh_overrides=overrides,
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        wavelength_ref=WL_0,
        n_core=N_WG,
        n_clad=N_CLAD,
        n_sub=N_SUB,
        thickness=H_WG,
        n_pml=task_cfg["n_pml"],
    )
    return metrics


def run_single_task(task_cfg, out_dir):
    """
    Execute one 2D EIM directional-coupler study task and write ``task_<I>.npz``
    and ``task_<I>.json`` into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task_cfg["index"]
    kind = task_cfg["kind"]
    t0 = time.perf_counter()

    four_ports = bool(task_cfg["collect_fields"] or kind == "quadrature")
    structures, ports, domain, geom_info = build_directional_coupler(
        L_c=task_cfg["L_c"],
        theta_sw_deg=task_cfg["theta_sw_deg"],
        four_ports=four_ports,
    )
    n_cells_gap = task_cfg.get("n_cells_gap")
    overrides = coupler_mesh_overrides(
        L_c=task_cfg["L_c"],
        gap_eff=geom_info["gap_eff"],
        w_eff=geom_info["w_eff"],
        n_cells_gap=n_cells_gap,
        dx_sb=DX_SBEND,
        dy_sb=DY_SBEND,
    )
    use_gpu = os.environ.get("WAKIS_USE_GPU", "0") == "1"
    if task_cfg.get("nonuniform", False):
        pic = NonUniformCouplerPIC(
            wavelength_min=WL_MIN,
            wavelength_max=WL_MAX,
            L_c=task_cfg["L_c"],
            gap_eff=geom_info["gap_eff"],
            w_eff=geom_info["w_eff"],
            n_cells_gap=n_cells_gap if n_cells_gap is not None else 8,
            y_port=Y_PORT,
            dx_sb=DX_SBEND,
            dy_sb=DY_SBEND,
            n_freqs=N_FREQS,
            structures=structures,
            ports=ports,
            n_core=N_WG,
            n_clad=N_CLAD,
            n_sub=N_SUB,
            wavelength_ref=WL_0,
            two_d=True,
            averaging=task_cfg["averaging"],
            cells_per_wavelength=task_cfg["cpw"],
            domain=domain,
            n_pml=task_cfg["n_pml"],
            field_wavelengths=(WL_DESIGN,),
            record_movie=task_cfg["record_movie"],
            movie_frames=task_cfg["movie_frames"],
            use_gpu=use_gpu,
        )
    else:
        pic = PICSimulation(
            wavelength_min=WL_MIN,
            wavelength_max=WL_MAX,
            n_freqs=N_FREQS,
            structures=structures,
            ports=ports,
            n_core=N_WG,
            n_clad=N_CLAD,
            n_sub=N_SUB,
            wavelength_ref=WL_0,
            two_d=True,
            averaging=task_cfg["averaging"],
            cells_per_wavelength=task_cfg["cpw"],
            domain=domain,
            n_pml=task_cfg["n_pml"],
            mesh_overrides=overrides,
            field_wavelengths=(WL_DESIGN,),
            record_movie=task_cfg["record_movie"],
            movie_frames=task_cfg["movie_frames"],
            use_gpu=use_gpu,
        )

    mesh_metrics = extract_mesh_metrics_from_grid(
        pic.grid.x,
        pic.grid.y,
        gap_eff=geom_info["gap_eff"],
        Nz=int(pic.grid.Nz),
    )

    payload = {
        "task_index": np.int32(idx),
        "name": np.array(task_cfg["name"]),
        "kind": np.array(kind),
        "level": np.array(task_cfg["level"]),
        "averaging": np.array(task_cfg["averaging"]),
        "cpw": np.float32(task_cfg["cpw"]),
        "n_cells_gap": np.int32(0 if n_cells_gap is None else int(n_cells_gap)),
        "L_c": np.float64(task_cfg["L_c"]),
        "theta_sw_deg": np.float64(task_cfg["theta_sw_deg"]),
        "w_eff": np.float64(geom_info["w_eff"]),
        "gap_eff": np.float64(geom_info["gap_eff"]),
        "w_bot": np.float64(geom_info["w_bot"]),
        "w_top": np.float64(geom_info["w_top"]),
        "n_eff_slab": np.float64(pic.n_eff_slab),
        "min_feature_nm": np.float64(mesh_metrics["min_feature_nm"]),
        "dx_background_nm": np.float64(mesh_metrics["dx_background_nm"]),
        "dx_min_nm": np.float64(mesh_metrics["dx_min_nm"]),
        "cells_across_feature": np.float64(mesh_metrics["cells_across_feature"]),
        "n_cells": np.int64(mesh_metrics["n_cells"]),
        "frequencies": pic.frequencies.astype(np.float64),
        "wavelengths": pic.wavelengths.astype(np.float64),
    }

    if kind == "quadrature":
        res_up = run_phased_dual_drive(
            pic,
            port_phases={"in_bot": 0.0, "in_top": -0.5 * np.pi},
            energy_decay=task_cfg["energy_decay"],
            collect_dft=True,
        )
        payload["P_up_minus_pi2"] = res_up["power"]["out_top"].astype(np.float64)
        payload["P_down_minus_pi2"] = res_up["power"]["out_bot"].astype(np.float64)
        payload["P_refl_bot_minus_pi2"] = res_up["power"]["in_bot"].astype(np.float64)
        payload["P_refl_top_minus_pi2"] = res_up["power"]["in_top"].astype(np.float64)
        if WL_DESIGN in res_up["dft"]:
            fmap = res_up["dft"][WL_DESIGN]
            payload["dft_x"] = fmap["x"]
            payload["dft_y"] = fmap["y"]
            payload["dft_Ey"] = fmap["Ey"]
            payload["dft_E2"] = fmap["E2"]
    else:
        s_mat = pic.run_s_parameters(
            drive=["in_bot"],
            energy_decay=task_cfg["energy_decay"],
            collect_fields=task_cfg["collect_fields"],
        )
        s11 = s_mat[("in_bot", "in_bot")].astype(np.complex128)
        s31 = s_mat[("out_top", "in_bot")].astype(np.complex128)
        s41 = s_mat[("out_bot", "in_bot")].astype(np.complex128)
        s21 = (
            s_mat[("in_top", "in_bot")].astype(np.complex128)
            if ("in_top", "in_bot") in s_mat
            else np.zeros_like(s11)
        )
        payload["S11"] = s11
        payload["S21"] = s21
        payload["S31"] = s31  # Cross port (upper output arm)
        payload["S41"] = s41  # Bar port (lower output arm)

        tc = np.abs(s31) ** 2
        tb = np.abs(s41) ** 2
        wl_cross, t_tot_cross = find_crossing_wavelength(pic.wavelengths, tc, tb)
        payload["wl_cross"] = np.float64(wl_cross)
        payload["t_tot_cross"] = np.float64(t_tot_cross)

        if task_cfg["collect_fields"] or kind == "sw_sweep":
            n_cmt_pts = 13 if task_cfg["collect_fields"] else 7
            wls_cmt = np.linspace(WL_MIN, WL_MAX, n_cmt_pts)
            sm = solve_coupler_supermodes(
                pic,
                wavelengths=wls_cmt,
                x_coupled=0.0,
                x_sb_start=0.5 * task_cfg["L_c"],
                dx_sb=DX_SBEND,
                dy_sb=DY_SBEND,
                L_c=task_cfg["L_c"],
                wavelength_profile=WL_DESIGN,
            )
            for k_sm, v_sm in sm.items():
                payload[k_sm] = v_sm

        if task_cfg["collect_fields"]:
            modes_in = pic.solve_port_modes(
                "in_bot", frequencies=c_light / WL_DESIGN, n_modes=1
            )
            modes_out = pic.solve_port_modes(
                "out_top", frequencies=c_light / WL_DESIGN, n_modes=1
            )
            y_in = modes_in[0].coords["Eb"][0]
            ey_in = modes_in[0].e["b"][:, 0]
            m_in = np.abs(y_in - (-Y_PORT)) <= 0.85e-6
            y_out = modes_out[0].coords["Eb"][0]
            ey_out = modes_out[0].e["b"][:, 0]
            m_out = np.abs(y_out - (+Y_PORT)) <= 0.85e-6
            payload["port_y_in"] = y_in[m_in].astype(np.float32)
            payload["port_ey_in"] = ey_in[m_in].astype(np.complex64)
            payload["port_neff_in"] = np.float64(np.real(modes_in[0].neff))
            payload["port_y_out"] = y_out[m_out].astype(np.float32)
            payload["port_ey_out"] = ey_out[m_out].astype(np.complex64)
            payload["port_neff_out"] = np.float64(np.real(modes_out[0].neff))

            wl_disp = np.linspace(WL_MIN, WL_MAX, 11)
            f_disp = c_light / wl_disp
            disp_modes = pic.solve_port_modes("in_bot", frequencies=f_disp, n_modes=1)
            neff_disp = np.array([float(np.real(m[0].neff)) for m in disp_modes])
            dneff_dl = np.gradient(neff_disp, wl_disp)
            ng_disp = neff_disp - wl_disp * dneff_dl
            payload["disp_wavelengths"] = wl_disp.astype(np.float64)
            payload["disp_neff"] = neff_disp.astype(np.float64)
            payload["disp_ng"] = ng_disp.astype(np.float64)

            if WL_DESIGN in pic.dft_fields:
                fmap = pic.dft_fields[WL_DESIGN]
                payload["dft_x"] = fmap["x"]
                payload["dft_y"] = fmap["y"]
                payload["dft_Ey"] = fmap["Ey"]
                payload["dft_E2"] = fmap["E2"]

            if pic.movie_data is not None:
                payload["movie_x"] = pic.movie_data["x"]
                payload["movie_y"] = pic.movie_data["y"]
                payload["movie_times"] = pic.movie_data["times"]
                payload["movie_frames"] = pic.movie_data["frames"]
                payload["movie_peak"] = pic.movie_data["peak"]

            if pic.port_time_trace is not None:
                payload["trace_t"] = pic.port_time_trace["t"]
                payload["trace_Ey"] = pic.port_time_trace["Ey"]

            if pic.flux_balance is not None:
                payload["flux_east"] = pic.flux_balance["east"].astype(np.float64)
                payload["flux_west"] = pic.flux_balance["west"].astype(np.float64)
                payload["flux_lateral"] = pic.flux_balance["lateral"].astype(np.float64)
                payload["flux_total_box"] = pic.flux_balance["total_box"].astype(
                    np.float64
                )

    n_steps = int(getattr(pic.solver, "Nt", 0))
    elapsed = time.perf_counter() - t0
    payload["n_steps"] = np.int64(n_steps)
    payload["elapsed_s"] = np.float64(elapsed)

    npz_path = out_dir / f"task_{idx}.npz"
    np.savez_compressed(npz_path, **payload)

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    meta = {
        "task_index": idx,
        "name": task_cfg["name"],
        "kind": kind,
        "level": task_cfg["level"],
        "averaging": task_cfg["averaging"],
        "cpw": task_cfg["cpw"],
        "n_cells_gap": None if n_cells_gap is None else int(n_cells_gap),
        "L_c_um": round(task_cfg["L_c"] * 1e6, 2),
        "theta_sw_deg": round(task_cfg["theta_sw_deg"], 1),
        "min_feature_nm": round(mesh_metrics["min_feature_nm"], 2),
        "dx_background_nm": round(mesh_metrics["dx_background_nm"], 2),
        "dx_min_nm": round(mesh_metrics["dx_min_nm"], 2),
        "cells_across_feature": round(mesh_metrics["cells_across_feature"], 2),
        "grid_shape": mesh_metrics["grid_shape"],
        "n_cells": int(mesh_metrics["n_cells"]),
        "n_steps": n_steps,
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mb": round(rss_mb, 2),
        "npz_bytes": int(npz_path.stat().st_size),
    }
    (out_dir / f"task_{idx}.json").write_text(json.dumps(meta, indent=2))
    n_gap_str = "none" if n_cells_gap is None else str(int(n_cells_gap))
    print(
        f"[task_{idx:02d} {task_cfg['name']}] "
        f"cpw={task_cfg['cpw']} N={n_gap_str} "
        f"min_feature={meta['min_feature_nm']:.2f}nm "
        f"dx_bg={meta['dx_background_nm']:.2f}nm "
        f"dx_min={meta['dx_min_nm']:.2f}nm "
        f"cells_across_feature={meta['cells_across_feature']:.2f} "
        f"n_cells={meta['n_cells']} n_steps={meta['n_steps']} "
        f"time={elapsed:.2f}s rss={rss_mb:.1f}MB"
    )
    return meta


# %% [markdown]
# ## 6. Generating the diagnostic figure suite
#
# When invoked with `--plot` (or at the end of `--quick` / `--full`),
# `generate_all_figures` reads `data/task_*.npz` and renders the complete figure
# suite into `figures/`:
#
# 0. `00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`, `00c_quadrature_setup_schematic.png` (plus `00_setup_table.md`) — Labelled setup schematics (domain bounds, CPML boundaries, $\text{Si}/\text{SiO}_2/\text{Air}$ materials, $\text{TE}_0$ mode source with launched profile, 4 port mode monitors and 4-wall Poynting flux box), mesh view with $1\times$ coupling-gap zoom and broadband excitation pulse/spectrum, and companion dual-port $\pi/2$ quadrature setup schematic.
# 1. `01_directional_coupler_layout.png` — Annotated top-view layout of the 4-port SOI directional coupler.
# 2. `02_eim_vertical_stack.png` — Asymmetric vertical SOI stack ($\text{SiO}_2 / \text{Si} / \text{Air}$), $\text{TE}_0$ slab mode, and $n_{\text{eff,slab}}(\lambda)$.
# 3. `03_trapezoidal_sidewall_geometry.png` — Trapezoidal waveguide cross-section ($0^\circ$ vs $30^\circ$) and effective width/gap vs $\theta_{\text{sw}}$.
# 4. `04_permittivity_mesh_zoom.png` — Zoomed in-plane permittivity map with Yee cell edges across the $g_{\text{eff}} = 207.0\;\text{nm}$ coupling gap at $L_1 = (10, 4)$ ($4.28$ cells, $\Delta y_{\min} = 48.41\;\text{nm}$) and the reference level $R = (40, 16)$ ($16.95$ cells, $\Delta y_{\min} = 12.22\;\text{nm}$; $L_3 = (20, 8)$ with $8.55$ cells in `--quick`), plus 1D transverse sidewall subpixel cuts.
# 4b. `04b_mesh_upgrade_comparison.png` — Split-half mesh view, 1D $\Delta y(y)$ and non-square $\Delta x(x)$ cell-spacing profiles, and broadband $T_{\text{cross}}(\lambda), T_{\text{bar}}(\lambda)$ comparison between the baseline `kottke_L3` mesh ($193{,}842\;\text{cells}$), the mirror-symmetric graded non-uniform Cartesian mesh `kottke_nonuniform` ($119{,}637\;\text{cells}$, $-38.3\%$ cells, $-53.0\%$ wall time), and `kottke_reference` $R = (40, 16)$.
# 5. `05_port_mode_profiles.png` — Input (`in_bot`) and output (`out_top`) port eigenmode profiles $|E_y(y)|$ and $\mathrm{Re}(E_y(y))$.
# 6. `06_supermode_profiles_and_dispersion.png` — Symmetric ($\text{TE}_{0,\text{even}}$) and antisymmetric ($\text{TE}_{0,\text{odd}}$) supermode profiles, $\Delta n_{\text{eff}}(\lambda)$, and $L_\pi(\lambda)$.
# 7. `07_field_intensity_single_drive.png` — Steady-state logarithmic intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $\lambda = 1.53\;\mu\text{m}$ with `overlay_geometry` material, boundary, source, and monitor labels.
# 8. `08_field_real_ey_single_drive.png` — Steady-state wave-front map $\mathrm{Re}(E_y)$ at $\lambda = 1.53\;\mu\text{m}$ showing the $\pi/2$ bar/cross phase shift with `overlay_geometry`.
# 9. `09_pulse_propagation.gif` — Time-domain pulse propagation movie through the directional coupler with `overlay_geometry` on every frame.
# 10. `10_sparameter_spectra_linear_and_db.png` — Broadband cross/bar/return-loss spectra in both linear scale (vs CMT and the under-resolved no-override warning pair) and dB scale.
# 11. `11_coupling_length_sweep_cmt_vs_fit.png` — Cross/bar power vs interaction length $L_c \in [0, 8]\;\mu\text{m}$ and 2D $(L_c, \lambda)$ CMT map.
# 12. `12_sidewall_angle_sweep.png` — 50:50 crossing wavelength $\lambda_{\text{cross}}(\theta_{\text{sw}})$ colored by total transmittance, plus $T_{\text{cross}}(\lambda)$ spectra over $0^\circ\text{--}30^\circ$.
# 13. `13_quadrature_dual_drive_routing.png` — Dual-port $\pi/2$ quadrature intensity map ($99.65\%$ routing to `out_top` at $\lambda_{\text{cross}}$) and output power spectra.
# 14. `14_energy_budget_and_quadrature_phase.png` — Insertion loss & 2D Poynting flux-box audit, plus relative bar/cross phase $\arg(S_{31} S_{41}^*)$.
# 15. `15_kottke_vs_staircase_convergence.png` — Subpixel convergence of Kottke vs staircasing across the $(\text{cpw}, N)$ ladder (with top axis showing cells across $g_{\text{eff}}$ and the under-resolved warning pair) relative to the independent Kottke reference run.
# 16. `16_port_time_trace.png` — Input (`in_bot`), cross (`out_top`), and bar (`out_bot`) time-domain pulse waveforms $E_y(t)$ and exponential ring-down / termination audit.


# %%
def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Read ``data_dir/task_*.npz`` and render all diagnostic figures into ``fig_dir``."""
    figstyle.apply_style()
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tasks_data = {}
    for p in sorted(data_dir.glob("task_*.npz")):
        d = dict(np.load(p, allow_pickle=False))
        jp = p.with_suffix(".json")
        if jp.exists():
            meta_j = json.loads(jp.read_text())
            for k_j in ("elapsed_s", "peak_rss_mb", "n_cells", "n_steps"):
                if k_j in meta_j and k_j not in d:
                    d[k_j] = meta_j[k_j]
        tasks_data[str(d["name"])] = d

    primary_name = "kottke_L3" if "kottke_L3" in tasks_data else "kottke_L2"
    if primary_name not in tasks_data:
        raise FileNotFoundError(
            f"Missing primary task ('kottke_L2' or 'kottke_L3') in {data_dir}. Run --quick or --full first."
        )
    primary = tasks_data[primary_name]
    structures, ports, domain, geom_base = build_directional_coupler(four_ports=True)

    # 0. Setup schematics (00a geometry, 00b mesh + excitation, 00c quadrature) + Markdown table
    cpw_prim = float(primary["cpw"])
    n_gap_prim = int(primary.get("n_cells_gap", 8)) or 8
    n_pml_prim = 10 if primary_name == "kottke_L3" else 8
    ov_prim = coupler_mesh_overrides(
        L_COUPLED_BASE, geom_base["gap_eff"], geom_base["w_eff"], n_gap_prim
    )
    st_schem, ports_schem, dom_schem, _ = build_directional_coupler(four_ports=True)
    pic_schem = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=st_schem,
        ports=ports_schem,
        n_core=N_WG,
        n_clad=N_CLAD,
        n_sub=N_SUB,
        wavelength_ref=WL_0,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw_prim,
        domain=dom_schem,
        n_pml=n_pml_prim,
        mesh_overrides=ov_prim,
    )
    save_coupler_setup_figures(
        pic_schem,
        primary,
        fig_dir,
        wavelength=WL_DESIGN,
        min_feature=geom_base["gap_eff"],
    )

    # 1. Top-view device layout with dimensional annotations
    annotations = [
        {
            "text": "Cosine S-bends\nΔx = 6.0 µm, Δy = 1.725 µm",
            "xy": (-5.0, 1.15),
            "xytext": (-5.0, 2.35),
            "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
        },
        {
            "text": "Coupled section\nLc = 4.0 µm, pitch = 550 nm\nweff = 343 nm, geff = 207 nm",
            "xy": (0.0, 0.28),
            "xytext": (0.0, 1.85),
            "arrowprops": dict(arrowstyle="->", color="#d95f02", lw=1.2),
        },
        {
            "text": "Lower input arm\n(y = −2.0 µm)",
            "xy": (-6.8, -2.0),
            "xytext": (-4.5, -2.45),
            "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
        },
    ]
    plot_layout(
        structures,
        ports=ports,
        domain=domain,
        annotations=annotations,
        title="SOI directional coupler layout (Lc = 4.0 µm, θsw = 30°, 2D EIM domain)",
        out_path=fig_dir / "01_directional_coupler_layout.png",
    )

    # 2. Asymmetric Effective-Index Method (EIM) vertical stack
    plot_asymmetric_eim_stack(
        n_core=N_WG,
        n_clad=N_CLAD,
        n_sub=N_SUB,
        thickness=H_WG,
        wavelength_ref=WL_0,
        wavelengths=np.linspace(WL_MIN, WL_MAX, 51),
        out_path=fig_dir / "02_eim_vertical_stack.png",
    )

    # 3. Trapezoidal sidewall cross-section & effective width/gap vs sidewall angle
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 3.9), dpi=150)
    ax0.axhspan(-150, 0, color="#d9e6f2", label="SiO₂ substrate (n = 1.45)")
    ax0.axhline(0, color="#16324f", lw=1.2)
    g0 = trapezoidal_effective_width(W_GDS, DILATION, H_WG, 0.0, PITCH_COUPLED)
    g30 = trapezoidal_effective_width(W_GDS, DILATION, H_WG, 30.0, PITCH_COUPLED)
    h_nm = H_WG * 1e9
    for sgn in (-1.0, +1.0):
        yc_nm = sgn * Y_COUPLED * 1e9
        wb0 = 0.5 * g0["w_bot"] * 1e9
        wt0 = 0.5 * g0["w_top"] * 1e9
        poly0 = np.array(
            [[yc_nm - wb0, 0], [yc_nm + wb0, 0], [yc_nm + wt0, h_nm], [yc_nm - wt0, h_nm]]
        )
        ax0.plot(
            np.append(poly0[:, 0], poly0[0, 0]),
            np.append(poly0[:, 1], poly0[0, 1]),
            "--",
            color="#2b5c8f",
            lw=1.6,
            label="θsw = 0° (w = 470 nm, g = 80 nm)" if sgn < 0 else None,
        )
        wb30 = 0.5 * g30["w_bot"] * 1e9
        wt30 = 0.5 * g30["w_top"] * 1e9
        poly30 = np.array(
            [
                [yc_nm - wb30, 0],
                [yc_nm + wb30, 0],
                [yc_nm + wt30, h_nm],
                [yc_nm - wt30, h_nm],
            ]
        )
        ax0.fill(
            poly30[:, 0],
            poly30[:, 1],
            color="#d95f02",
            alpha=0.55,
            edgecolor="#a63603",
            lw=1.5,
            label="θsw = 30° (weff = 343 nm, geff = 207 nm)" if sgn < 0 else None,
        )
    ax0.axhline(0.5 * h_nm, color="#1b9e77", ls=":", lw=1.5, label="2D EIM mid-plane (z = 110 nm)")
    ax0.set(
        xlim=(-620, 620),
        ylim=(-120, 310),
        xlabel="Transverse coordinate y [nm]",
        ylabel="Vertical height z [nm]",
        title="(a) Coupled waveguide cross-section at x = 0",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=7.8)

    angles_fine = np.linspace(0.0, 30.0, 61)
    weff_nm = [
        trapezoidal_effective_width(W_GDS, DILATION, H_WG, a, PITCH_COUPLED)["w_eff"] * 1e9
        for a in angles_fine
    ]
    wtop_nm = [
        trapezoidal_effective_width(W_GDS, DILATION, H_WG, a, PITCH_COUPLED)["w_top"] * 1e9
        for a in angles_fine
    ]
    geff_nm = [
        trapezoidal_effective_width(W_GDS, DILATION, H_WG, a, PITCH_COUPLED)["gap_eff"] * 1e9
        for a in angles_fine
    ]
    ax1.plot(angles_fine, weff_nm, color="#2b5c8f", lw=2.2, label=r"Mid-height width $w_{\mathrm{eff}}(\theta_{\mathrm{sw}})$")
    ax1.plot(angles_fine, wtop_nm, "--", color="#7570b3", lw=1.7, label=r"Top width $w_{\mathrm{top}}(\theta_{\mathrm{sw}})$")
    ax1.plot(angles_fine, geff_nm, color="#d95f02", lw=2.2, label=r"Mid-height gap $g_{\mathrm{eff}}(\theta_{\mathrm{sw}})$")
    ax1.axvline(30.0, color="0.4", ls=":", lw=1.3, label="Baseline θsw = 30°")
    ax1.set(
        xlabel="Sidewall slant angle θsw [deg]",
        ylabel="Dimension [nm]",
        title="(b) Effective width & gap vs sidewall angle",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="center left", fontsize=8.0)
    figstyle.save(
        fig,
        fig_dir / "03_trapezoidal_sidewall_geometry.png",
        close=True,
        auto_legend_below=False,
    )

    # 4. Permittivity & Yee mesh zoomed on the smallest feature (coupling gap) at L1 and Reference
    ref_task = tasks_data.get("kottke_reference", primary)
    l1_task = tasks_data.get("kottke_L1", primary)
    cpw_l1 = float(l1_task["cpw"])
    n_gap_l1 = int(l1_task.get("n_cells_gap", 4)) or 4
    cpw_ref = float(ref_task["cpw"])
    n_gap_ref = int(ref_task.get("n_cells_gap", 8)) or 8
    ref_lvl_str = str(ref_task.get("level", "L3"))

    ov_l1 = coupler_mesh_overrides(
        L_COUPLED_BASE, geom_base["gap_eff"], geom_base["w_eff"], n_gap_l1
    )
    ov_ref = coupler_mesh_overrides(
        L_COUPLED_BASE, geom_base["gap_eff"], geom_base["w_eff"], n_gap_ref
    )
    # Build zoomed coupling-section PICSimulations with identical override boxes
    zoom_dom = (-1.6e-6, 1.6e-6, -1.8e-6, 1.8e-6)
    st_l1_k, _, _, _ = build_directional_coupler(four_ports=False)
    st_l1_s, _, _, _ = build_directional_coupler(four_ports=False)
    st_ref_k, _, _, _ = build_directional_coupler(four_ports=False)
    pic_l1_k = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=st_l1_k,
        n_core=N_WG,
        n_clad=N_CLAD,
        n_sub=N_SUB,
        wavelength_ref=WL_0,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw_l1,
        domain=zoom_dom,
        n_pml=6,
        mesh_overrides=ov_l1,
    )
    pic_l1_s = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=st_l1_s,
        n_core=N_WG,
        n_clad=N_CLAD,
        n_sub=N_SUB,
        wavelength_ref=WL_0,
        two_d=True,
        averaging="staircase",
        cells_per_wavelength=cpw_l1,
        domain=zoom_dom,
        n_pml=6,
        mesh_overrides=ov_l1,
    )
    pic_ref_k = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=st_ref_k,
        n_core=N_WG,
        n_clad=N_CLAD,
        n_sub=N_SUB,
        wavelength_ref=WL_0,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw_ref,
        domain=zoom_dom,
        n_pml=6,
        mesh_overrides=ov_ref,
    )

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=150)
    g_nm = geom_base["gap_eff"] * 1e9
    w_nm = geom_base["w_eff"] * 1e9
    y_in_nm = 0.5 * g_nm
    y_out_nm = y_in_nm + w_nm

    for ax_m, pic_m, task_m, tag_m in (
        (ax0, pic_l1_k, l1_task, f"(a) L1 (cpw={int(cpw_l1)}, N={n_gap_l1}) Kottke mesh"),
        (
            ax1,
            pic_ref_k,
            ref_task,
            f"(b) Ref {ref_lvl_str} (cpw={int(cpw_ref)}, N={n_gap_ref}) Kottke mesh",
        ),
    ):
        xg = pic_m.grid.x * 1e9
        yg = pic_m.grid.y * 1e9
        nx_m, ny_m = pic_m.grid.Nx, pic_m.grid.Ny
        ieps_m = pic_m.solver.ieps[:nx_m, :ny_m, 0, "y"]
        eps_yy = np.where(
            ieps_m > 0, 1.0 / (np.where(ieps_m > 0, ieps_m, 1.0) * epsilon_0), 1.0
        ).T
        im_m = ax_m.pcolormesh(
            xg,
            yg,
            eps_yy,
            cmap="viridis",
            vmin=1.0,
            vmax=float(pic_m.n_eff_slab**2),
            edgecolors=(1.0, 1.0, 1.0, 0.38),
            linewidth=0.45,
        )
        for sgn in (-1.0, +1.0):
            ax_m.axhline(sgn * y_in_nm, color="#ffae42", ls="--", lw=1.2)
            ax_m.axhline(sgn * y_out_nm, color="#ffae42", ls=":", lw=1.0)
        caf_m = float(task_m.get("cells_across_feature", g_nm / np.min(np.diff(yg))))
        dxmin_m = float(task_m.get("dx_min_nm", np.min(np.diff(yg))))
        ax_m.annotate(
            "",
            xy=(-260.0, +y_in_nm),
            xytext=(-260.0, -y_in_nm),
            arrowprops=dict(arrowstyle="<->", color="white", lw=1.6),
        )
        ax_m.text(
            0.0,
            0.0,
            f"geff = {g_nm:.1f} nm\n({caf_m:.2f} cells, Δymin = {dxmin_m:.1f} nm)",
            color="white",
            fontsize=7.4,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.68, ec="none"),
        )
        ax_m.annotate(
            "",
            xy=(-260.0, +y_out_nm),
            xytext=(-260.0, +y_in_nm),
            arrowprops=dict(arrowstyle="<->", color="#ffae42", lw=1.3),
        )
        ax_m.text(
            -240.0,
            0.5 * (y_in_nm + y_out_nm),
            f"weff = {w_nm:.1f} nm",
            color="#ffae42",
            fontsize=7.3,
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"),
        )
        ax_m.set(
            xlim=(-380.0, 380.0),
            ylim=(-520.0, 520.0),
            aspect="equal",
            xlabel="Axial coordinate x [nm]",
            ylabel="Transverse coordinate y [nm]",
            title=tag_m,
        )
        cb_m = fig.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.03)
        cb_m.set_label(r"Subpixel permittivity $\varepsilon_{yy}$")

    # Panel (c): 1D transverse permittivity cut across gap & mesh override grading
    for pic_c, lbl_c, col_c, ls_c, mk_c in (
        (pic_l1_s, f"L1 Staircase (cpw={int(cpw_l1)}, N={n_gap_l1})", "#d95f02", "--", "s"),
        (pic_l1_k, f"L1 Kottke (cpw={int(cpw_l1)}, N={n_gap_l1})", "#2b5c8f", "-", "o"),
        (
            pic_ref_k,
            f"Ref {ref_lvl_str} Kottke (cpw={int(cpw_ref)}, N={n_gap_ref})",
            "#1b9e77",
            "-",
            ".",
        ),
    ):
        yc_nm = 0.5 * (pic_c.grid.y[:-1] + pic_c.grid.y[1:]) * 1e9
        ix0 = int(np.argmin(np.abs(0.5 * (pic_c.grid.x[:-1] + pic_c.grid.x[1:]))))
        ieps_c = pic_c.solver.ieps[ix0, : pic_c.grid.Ny, 0, "y"]
        eps_cut = np.where(
            ieps_c > 0, 1.0 / (np.where(ieps_c > 0, ieps_c, 1.0) * epsilon_0), 1.0
        )
        m_c = np.abs(yc_nm) <= 520.0
        ax2.step(
            yc_nm[m_c],
            eps_cut[m_c],
            where="mid",
            color=col_c,
            ls=ls_c,
            lw=1.7,
            label=lbl_c,
        )
        ax2.plot(yc_nm[m_c], eps_cut[m_c], mk_c, color=col_c, ms=3.5)
    ax2.axvspan(-y_in_nm, +y_in_nm, color="#d95f02", alpha=0.14, label=f"Gap geff = {g_nm:.1f} nm")
    ax2.set(
        xlim=(-520.0, 520.0),
        xlabel="Transverse coordinate y [nm]",
        ylabel=r"Transverse permittivity $\varepsilon_{yy}(0, y)$",
        title="(c) Sidewall subpixel smoothing across the gap",
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=7.3)
    figstyle.save(
        fig,
        fig_dir / "04_permittivity_mesh_zoom.png",
        close=True,
        auto_legend_below=False,
    )

    # 4b. Mesh upgrade comparison: baseline uniform-override vs graded non-uniform Cartesian mesh
    if "kottke_nonuniform" in tasks_data:
        nu_task = tasks_data["kottke_nonuniform"]
        st_nu, ports_nu, dom_nu, _ = build_directional_coupler(four_ports=True)
        pic_nu = NonUniformCouplerPIC(
            WL_MIN,
            WL_MAX,
            structures=st_nu,
            ports=ports_nu,
            n_core=N_WG,
            n_clad=N_CLAD,
            n_sub=N_SUB,
            wavelength_ref=WL_0,
            two_d=True,
            averaging="kottke",
            cells_per_wavelength=float(nu_task["cpw"]),
            domain=dom_nu,
            n_pml=n_pml_prim,
            L_c=L_COUPLED_BASE,
            gap_eff=geom_base["gap_eff"],
            w_eff=geom_base["w_eff"],
            n_cells_gap=int(nu_task.get("n_cells_gap", n_gap_prim)) or n_gap_prim,
        )
        plot_mesh_upgrade_comparison_coupler(
            pic_schem,
            pic_nu,
            primary,
            nu_task,
            ref_task,
            geom_base,
            out_path=fig_dir / "04b_mesh_upgrade_comparison.png",
        )

    # 5. Port mode profiles at input (in_bot) and cross output (out_top)
    plot_port_modes(
        primary["port_y_in"] * 1e6,
        primary["port_ey_in"],
        float(primary["port_neff_in"]),
        primary["port_y_out"] * 1e6,
        primary["port_ey_out"],
        float(primary["port_neff_out"]),
        out_path=fig_dir / "05_port_mode_profiles.png",
    )

    # 6. Coupled-section supermode profiles (even/odd) & CMT beat-length dispersion
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=150)
    y_sm = primary["supermode_y"] * 1e6
    ey_ev = primary["supermode_ey_even"]
    ey_od = primary["supermode_ey_odd"]
    ey_ev_n = ey_ev / max(float(np.max(np.abs(ey_ev))), 1e-30)
    ey_od_n = ey_od / max(float(np.max(np.abs(ey_od))), 1e-30)
    ne_ref = float(primary["supermode_neff_even_ref"])
    no_ref = float(primary["supermode_neff_odd_ref"])
    w_um_half = 0.5 * float(primary["w_eff"]) * 1e6
    for sgn in (-1.0, +1.0):
        yc_u = sgn * Y_COUPLED * 1e6
        ax0.axvspan(yc_u - w_um_half, yc_u + w_um_half, color="0.85", alpha=0.6)
    ax0.plot(
        y_sm,
        ey_ev_n,
        color="#2b5c8f",
        lw=2.2,
        label=f"Even supermode TE₀,even (neff = {ne_ref:.4f})",
    )
    ax0.plot(
        y_sm,
        ey_od_n,
        "--",
        color="#d95f02",
        lw=2.0,
        label=f"Odd supermode TE₀,odd (neff = {no_ref:.4f})",
    )
    ax0.axhline(0.0, color="0.5", ls=":", lw=1.0)
    ax0.set(
        xlabel="Transverse coordinate y at x = 0 [µm]",
        ylabel=r"Normalized transverse field $E_y(y)$",
        title="(a) Coupled-section even & odd supermodes (λ = 1530 nm)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=8.0)

    wls_cmt_nm = primary["cmt_wavelengths"] * 1e9
    ax1.plot(
        wls_cmt_nm,
        primary["cmt_n_even"],
        "o-",
        color="#2b5c8f",
        lw=2.0,
        ms=4.5,
        label=r"$n_{\mathrm{even}}(\lambda)$",
    )
    ax1.plot(
        wls_cmt_nm,
        primary["cmt_n_odd"],
        "s--",
        color="#d95f02",
        lw=2.0,
        ms=4.5,
        label=r"$n_{\mathrm{odd}}(\lambda)$",
    )
    ax1.set_xlabel("Wavelength λ [nm]")
    ax1.set_ylabel(r"Supermode effective index $n_{\mathrm{eff}}$", color="#2b5c8f")
    ax1.grid(True, ls=":", alpha=0.4)
    ax1_r = ax1.twinx()
    ax1_r._is_inset = True
    ax1_r.plot(
        wls_cmt_nm,
        primary["cmt_L_pi"] * 1e6,
        "^-.",
        color="#1b9e77",
        lw=2.0,
        ms=4.5,
        label=r"Beat length $L_\pi = \lambda / (2\Delta n_{\mathrm{eff}})$",
    )
    ax1_r.plot(
        wls_cmt_nm,
        primary["cmt_L_sb_eff"] * 1e6,
        ":",
        color="#7570b3",
        lw=1.8,
        label=r"S-bend effective length $L_{\mathrm{sb,eff}}$",
    )
    ax1_r.set_ylabel(r"Characteristic coupling length [µm]", color="#1b9e77")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=7.8)
    ax1.set_title("(b) Supermode dispersion & CMT crossover length Lπ(λ)")
    figstyle.save(
        fig,
        fig_dir / "06_supermode_profiles_and_dispersion.png",
        close=True,
        auto_legend_below=False,
    )

    # 7. Steady-state logarithmic intensity map |E|^2 [dB] at 1530 nm (single-port drive)
    render_coupler_field_map_with_overlay(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_E2"],
        pic=pic_schem,
        kind="intensity_log",
        title=r"Single-port drive (`in_bot`): modal intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $\lambda = 1530\;\mathrm{nm}$",
        out_path=fig_dir / "07_field_intensity_single_drive.png",
    )

    # 8. Steady-state real field map Re(Ey) at 1530 nm (single-port drive)
    render_coupler_field_map_with_overlay(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_Ey"],
        pic=pic_schem,
        kind="real",
        title=r"Single-port drive (`in_bot`): wave-front phase $\mathrm{Re}(E_y)$ at $\lambda = 1530\;\mathrm{nm}$ ($\pi/2$ split)",
        out_path=fig_dir / "08_field_real_ey_single_drive.png",
    )

    # 9. Animated pulse propagation GIF
    if "movie_frames" in primary and len(primary["movie_frames"]) > 0:
        render_coupler_movie_with_overlay(
            primary["movie_x"],
            primary["movie_y"],
            primary["movie_frames"],
            primary["movie_times"],
            pic=pic_schem,
            out_path=fig_dir / "09_pulse_propagation.gif",
            fps=12,
        )

    # Sort wavelengths ascending for spectral plots
    wl_nm_all = primary["wavelengths"] * 1e9
    order = np.argsort(wl_nm_all)
    wl_nm = wl_nm_all[order]
    s11 = primary["S11"][order]
    s21 = primary["S21"][order]
    s31 = primary["S31"][order]
    s41 = primary["S41"][order]
    t_cross = np.abs(s31) ** 2
    t_bar = np.abs(s41) ** 2
    r_in = np.abs(s11) ** 2
    c_iso = np.abs(s21) ** 2
    wl_cross_nm = float(primary["wl_cross"]) * 1e9
    wl_cmt_nm = float(primary["cmt_wl_cross"]) * 1e9

    # 10. S-parameter spectra in linear (FIT vs CMT + under-resolved warning) and dB scales
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.8, 4.3), dpi=150)
    p_lvl = str(primary.get("level", "L2"))
    p_cpw = int(primary["cpw"])
    p_ngap = int(primary.get("n_cells_gap", 6))
    ax0.plot(
        wl_nm,
        t_cross,
        "o-",
        color="#1b9e77",
        lw=2.1,
        ms=4.5,
        label=rf"FIT Cross $T_{{\mathrm{{cross}}}} = |S_{{31}}|^2$ ({p_lvl}=({p_cpw},{p_ngap}))",
    )
    ax0.plot(
        wl_nm,
        t_bar,
        "s-",
        color="#d95f02",
        lw=2.1,
        ms=4.5,
        label=rf"FIT Bar $T_{{\mathrm{{bar}}}} = |S_{{41}}|^2$ ({p_lvl}=({p_cpw},{p_ngap}))",
    )
    ax0.plot(
        wl_nm,
        t_cross + t_bar,
        ":",
        color="0.35",
        lw=1.6,
        label=r"FIT Total $T_{\mathrm{cross}} + T_{\mathrm{bar}}$",
    )
    ax0.plot(
        wls_cmt_nm,
        primary["cmt_t_cross"],
        "--",
        color="#1b9e77",
        alpha=0.75,
        lw=1.8,
        label=r"CMT $\sin^2[\pi(L_c + L_{\mathrm{sb,eff}})/(2L_\pi)]$",
    )
    ax0.plot(
        wls_cmt_nm,
        primary["cmt_t_bar"],
        "--",
        color="#d95f02",
        alpha=0.75,
        lw=1.8,
        label=r"CMT $\cos^2[\pi(L_c + L_{\mathrm{sb,eff}})/(2L_\pi)]$",
    )
    for u_key, u_lbl, u_col, u_ls in (
        (
            "underresolved_kottke",
            "Warning: under-resolved Kottke (10 cpw, no override)",
            "#2b5c8f",
            "-.",
        ),
        (
            "underresolved_staircase",
            "Warning: under-resolved Staircase (10 cpw, no override)",
            "#e7298a",
            ":",
        ),
    ):
        if u_key in tasks_data:
            ud = tasks_data[u_key]
            uw = ud["wavelengths"] * 1e9
            uord = np.argsort(uw)
            utc = np.abs(ud["S31"][uord]) ** 2
            ax0.plot(
                uw[uord],
                utc,
                u_ls,
                color=u_col,
                lw=1.7,
                alpha=0.88,
                label=f"{u_lbl} (λc={float(ud['wl_cross']) * 1e9:.1f} nm)",
            )
    ax0.axvline(
        wl_cross_nm,
        color="#1b9e77",
        ls="-.",
        lw=1.4,
        label=f"Converged FIT λcross = {wl_cross_nm:.1f} nm",
    )
    ax0.axhline(0.5, color="0.5", ls=":", lw=1.1)
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Guided modal power fraction",
        ylim=(-0.03, 1.06),
        title="(a) Linear power splitting: 2D FIT vs CMT & under-resolved demo",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="center left", fontsize=6.9)

    ax1.plot(
        wl_nm,
        10.0 * np.log10(np.maximum(t_cross, 1e-12)),
        "o-",
        color="#1b9e77",
        lw=2.0,
        ms=4.0,
        label=rf"Cross $10\log_{{10}}|S_{{31}}|^2$ ({p_lvl})",
    )
    ax1.plot(
        wl_nm,
        10.0 * np.log10(np.maximum(t_bar, 1e-12)),
        "s-",
        color="#d95f02",
        lw=2.0,
        ms=4.0,
        label=rf"Bar $10\log_{{10}}|S_{{41}}|^2$ ({p_lvl})",
    )
    if "underresolved_staircase" in tasks_data:
        uds = tasks_data["underresolved_staircase"]
        uws = uds["wavelengths"] * 1e9
        uords = np.argsort(uws)
        utcs = np.abs(uds["S31"][uords]) ** 2
        ax1.plot(
            uws[uords],
            10.0 * np.log10(np.maximum(utcs, 1e-12)),
            ":",
            color="#e7298a",
            lw=1.7,
            label="Warning: under-resolved Staircase Cross [dB]",
        )
    ax1.plot(
        wl_nm,
        10.0 * np.log10(np.maximum(r_in, 1e-12)),
        "^-.",
        color="#7570b3",
        lw=1.6,
        ms=3.5,
        label=r"Return loss $10\log_{10}|S_{11}|^2$",
    )
    if np.any(c_iso > 0):
        ax1.plot(
            wl_nm,
            10.0 * np.log10(np.maximum(c_iso, 1e-12)),
            "v:",
            color="#2b5c8f",
            lw=1.5,
            ms=3.5,
            label=r"Near-end isolation $10\log_{10}|S_{21}|^2$",
        )
    ax1.axhline(-3.0103, color="0.35", ls=":", lw=1.3, label="−3.01 dB (50:50 split)")
    ax1.axvline(wl_cross_nm, color="#1b9e77", ls="-.", lw=1.3)
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Modal power [dB]",
        ylim=(-45.0, 2.0),
        title="(b) Broadband logarithmic S-parameter spectra [dB]",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=7.2)
    figstyle.save(
        fig,
        fig_dir / "10_sparameter_spectra_linear_and_db.png",
        close=True,
        auto_legend_below=False,
    )

    # 11. Coupling-length sweep: FIT vs CMT sin^2 / cos^2 and 2D (L_c, lambda) map
    len_tasks = {4.0: primary}
    for L_um in (0, 2, 6, 8):
        k_name = f"length_L{L_um}um"
        if k_name in tasks_data:
            len_tasks[float(L_um)] = tasks_data[k_name]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.1), dpi=150)
    L_fine_um = np.linspace(0.0, 9.5, 200)
    wls_cmt = primary["cmt_wavelengths"]
    L_pi_arr = primary["cmt_L_pi"] * 1e6
    L_sb_arr = primary["cmt_L_sb_eff"] * 1e6

    for wl_target_nm, col, mk in (
        (1500.0, "#2b5c8f", "o"),
        (1535.5, "#1b9e77", "s"),
        (1580.0, "#d95f02", "^"),
    ):
        lpi_v = float(np.interp(wl_target_nm * 1e-9, wls_cmt, L_pi_arr))
        lsb_v = float(np.interp(wl_target_nm * 1e-9, wls_cmt, L_sb_arr))
        tc_curve = np.sin(0.5 * np.pi * (L_fine_um + lsb_v) / lpi_v) ** 2
        ax0.plot(
            L_fine_um,
            tc_curve,
            "-",
            color=col,
            lw=1.9,
            label=f"CMT Cross (λ = {wl_target_nm:.0f} nm, Lπ = {lpi_v:.1f} µm)",
        )
        L_pts, tc_pts, tb_pts = [], [], []
        for L_u, d_l in sorted(len_tasks.items()):
            w_l = d_l["wavelengths"] * 1e9
            ord_l = np.argsort(w_l)
            tc_l = np.abs(d_l["S31"][ord_l]) ** 2
            tb_l = np.abs(d_l["S41"][ord_l]) ** 2
            tot_l = np.maximum(tc_l + tb_l, 1e-12)
            L_pts.append(L_u)
            tc_pts.append(float(np.interp(wl_target_nm, w_l[ord_l], tc_l / tot_l)))
            tb_pts.append(float(np.interp(wl_target_nm, w_l[ord_l], tb_l / tot_l)))
        ax0.plot(
            L_pts,
            tc_pts,
            mk,
            color=col,
            ms=6.5,
            label=f"FIT Cross / Total (λ = {wl_target_nm:.0f} nm)",
        )

    ax0.axhline(0.5, color="0.4", ls=":", lw=1.2, label="50:50 splitting")
    ax0.axvline(4.0, color="0.4", ls="--", lw=1.2, label="Baseline Lc = 4.0 µm")
    ax0.set(
        xlabel="Straight coupling length Lc [µm]",
        ylabel=r"Normalized cross-port power $T_{\mathrm{cross}} / (T_{\mathrm{cross}} + T_{\mathrm{bar}})$",
        ylim=(-0.03, 1.05),
        title="(a) Cross-port power vs coupling length: FIT vs CMT",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=7.2)

    L_grid, W_grid = np.meshgrid(L_fine_um, wls_cmt_nm, indexing="ij")
    T_map = (
        np.sin(0.5 * np.pi * (L_grid + L_sb_arr[None, :]) / L_pi_arr[None, :]) ** 2
    )
    pcm = ax1.pcolormesh(
        L_fine_um, wls_cmt_nm, T_map.T, cmap="viridis", vmin=0.0, vmax=1.0, shading="auto"
    )
    cs = ax1.contour(
        L_fine_um,
        wls_cmt_nm,
        T_map.T,
        levels=[0.5],
        colors="white",
        linewidths=2.0,
        linestyles="--",
    )
    ax1.clabel(cs, fmt="CMT 50:50", fontsize=8.5)
    fit_L, fit_wlc = [], []
    for L_u, d_l in sorted(len_tasks.items()):
        ord_l = np.argsort(d_l["wavelengths"])
        diff_l = (np.abs(d_l["S31"][ord_l]) ** 2) - (np.abs(d_l["S41"][ord_l]) ** 2)
        if np.any(diff_l[:-1] * diff_l[1:] <= 0.0):
            fit_L.append(L_u)
            fit_wlc.append(float(d_l["wl_cross"]) * 1e9)
    ax1.plot(
        fit_L,
        fit_wlc,
        "o",
        color="#d95f02",
        mec="white",
        mew=1.2,
        ms=7.5,
        label="2D FIT λcross(Lc)",
    )
    cb = fig.colorbar(pcm, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label(r"CMT cross-port power fraction $T_{\mathrm{cross}}(L_c, \lambda)$")
    ax1.set(
        xlabel="Straight coupling length Lc [µm]",
        ylabel="Wavelength λ [nm]",
        title="(b) CMT splitting map & 50:50 locus",
    )
    ax1.legend(loc="upper right", fontsize=8.0)
    figstyle.save(
        fig,
        fig_dir / "11_coupling_length_sweep_cmt_vs_fit.png",
        close=True,
        auto_legend_below=False,
    )

    # 12. Sidewall-angle sweep (0 deg to 30 deg): 50:50 crossing wavelength & spectra
    sw_records = []
    for sw_deg in THETA_SW_SWEEP_DEG:
        d_sw = (
            primary
            if np.isclose(sw_deg, 30.0)
            else tasks_data.get(f"sw_angle_{int(sw_deg):02d}deg")
        )
        if d_sw is not None:
            sw_records.append((float(sw_deg), d_sw))

    if sw_records:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.1), dpi=150)
        angs = np.array([r[0] for r in sw_records])
        wlc_fit = np.array([float(r[1]["wl_cross"]) * 1e9 for r in sw_records])
        ttot_fit = np.array([float(r[1]["t_tot_cross"]) for r in sw_records])
        wlc_cmt = np.array(
            [
                float(r[1].get("cmt_wl_cross", r[1]["wl_cross"])) * 1e9
                for r in sw_records
            ]
        )

        sc = ax0.scatter(
            angs,
            wlc_fit,
            c=ttot_fit,
            cmap="plasma",
            vmin=min(0.92, float(np.min(ttot_fit))),
            vmax=1.0,
            s=75,
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
            label="2D FIT λcross (colored by Ttot)",
        )
        ax0.plot(angs, wlc_fit, "-", color="#2b5c8f", lw=1.6, alpha=0.7)
        ax0.plot(
            angs,
            wlc_cmt,
            "s--",
            color="#1b9e77",
            lw=1.8,
            ms=5.5,
            label="CMT supermode prediction λcross,CMT",
        )
        cb = fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.03)
        cb.set_label(r"Total guided transmittance $T_{\mathrm{cross}} + T_{\mathrm{bar}}$")
        ax0.set(
            xlabel="Sidewall slant angle θsw [deg]",
            ylabel="50:50 splitting wavelength λcross [nm]",
            title="(a) 50:50 crossing wavelength vs sidewall angle",
        )
        ax0.grid(True, ls=":", alpha=0.4)
        ax0.legend(loc="best", fontsize=7.8)

        cmap_sw = plt.cm.viridis(np.linspace(0.08, 0.95, len(sw_records)))
        for (ang, d_sw), col_sw in zip(sw_records, cmap_sw):
            w_s = d_sw["wavelengths"] * 1e9
            ord_s = np.argsort(w_s)
            tc_s = np.abs(d_sw["S31"][ord_s]) ** 2
            we_nm = float(d_sw["w_eff"]) * 1e9
            ax1.plot(
                w_s[ord_s],
                tc_s,
                "-",
                color=col_sw,
                lw=1.8,
                label=f"θsw = {ang:.0f}° (weff = {we_nm:.0f} nm)",
            )
        ax1.axhline(0.5, color="0.35", ls=":", lw=1.3, label="50:50 split (0.5)")
        ax1.set(
            xlabel="Wavelength λ [nm]",
            ylabel=r"Cross-port power $T_{\mathrm{cross}}(\lambda)$",
            ylim=(0.0, 1.02),
            title="(b) Cross-port transmittance across sidewall angles",
        )
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.legend(loc="upper left", fontsize=7.4, ncol=2)
        figstyle.save(
            fig,
            fig_dir / "12_sidewall_angle_sweep.png",
            close=True,
            auto_legend_below=False,
        )

    # 13. Dual-port pi/2 quadrature excitation: field routing map & output spectra
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.6, 4.1), dpi=150)
    quad = tasks_data.get("quadrature_dual_drive")
    if quad is not None and "dft_E2" in quad:
        qx_um = quad["dft_x"] * 1e6
        qy_um = quad["dft_y"] * 1e6
        q_e2 = quad["dft_E2"]
        q_max = max(float(np.max(q_e2)), 1e-30)
        q_db = 10.0 * np.log10(np.maximum(q_e2 / q_max, 10.0 ** (-35.0 / 10.0)))
        im = ax0.pcolormesh(
            qx_um, qy_um, q_db.T, cmap="inferno", vmin=-35.0, vmax=0.0, shading="auto"
        )
        cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)
        cb.set_label(r"Normalized intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ [dB]")
        ax0.set(
            aspect="equal",
            xlim=(qx_um.min(), qx_um.max()),
            ylim=(qy_um.min(), qy_um.max()),
            xlabel="x [µm]",
            ylabel="y [µm]",
            title="(a) Dual-port quadrature (Δφ = −π/2): upper-arm routing",
        )
        overlay_coupler_geometry(
            ax0, pic_schem, outline_color="white", quadrature=True
        )

    # Superposition prediction from single-port S31, S41 (by mirror symmetry S32=S41, S42=S31)
    p_up_sup_m = 0.5 * np.abs(s31 + (-1j) * s41) ** 2
    p_dn_sup_m = 0.5 * np.abs(s41 + (-1j) * s31) ** 2
    p_up_sup_p = 0.5 * np.abs(s31 + (+1j) * s41) ** 2
    p_dn_sup_p = 0.5 * np.abs(s41 + (+1j) * s31) ** 2
    ax1.plot(
        wl_nm,
        p_up_sup_m,
        "-",
        color="#1b9e77",
        lw=2.0,
        label="S-matrix superposition Pup (Δφ = −π/2)",
    )
    ax1.plot(
        wl_nm,
        p_dn_sup_m,
        "--",
        color="#d95f02",
        lw=2.0,
        label="S-matrix superposition Pdown (Δφ = −π/2)",
    )
    ax1.plot(
        wl_nm,
        p_dn_sup_p,
        ":",
        color="#7570b3",
        lw=1.6,
        label="S-matrix superposition Pdown (Δφ = +π/2 → lower arm)",
    )
    if quad is not None and "P_up_minus_pi2" in quad:
        qw = quad["wavelengths"] * 1e9
        qord = np.argsort(qw)
        ax1.plot(
            qw[qord],
            quad["P_up_minus_pi2"][qord],
            "o",
            color="#1b9e77",
            ms=5.0,
            label="Direct dual-source FIT Pup (Δφ = −π/2)",
        )
        ax1.plot(
            qw[qord],
            quad["P_down_minus_pi2"][qord],
            "s",
            color="#d95f02",
            ms=5.0,
            label="Direct dual-source FIT Pdown (Δφ = −π/2)",
        )
    ax1.axvline(wl_cross_nm, color="#2b5c8f", ls="-.", lw=1.3, label=f"λcross = {wl_cross_nm:.1f} nm")
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Normalized output power fraction",
        ylim=(-0.03, 1.05),
        title="(b) Coherent quadrature routing spectra",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="center right", fontsize=7.4)
    figstyle.save(
        fig,
        fig_dir / "13_quadrature_dual_drive_routing.png",
        close=True,
        auto_legend_below=False,
    )

    # 14. Energy conservation budget & bar/cross quadrature phase audit
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.0), dpi=150)
    mask_in = (wl_nm >= 1420.0) & (wl_nm <= 1680.0)
    wl_b = wl_nm[mask_in]
    t_tot_db = 10.0 * np.log10(np.maximum((t_cross + t_bar)[mask_in], 1e-12))
    ax0.plot(
        wl_b,
        t_tot_db,
        "o-",
        color="#1b9e77",
        lw=2.0,
        ms=4.0,
        label=r"Guided total $10\log_{10}(T_{\mathrm{cross}} + T_{\mathrm{bar}})$ (2D EIM)",
    )
    if "flux_total_box" in primary:
        f_box = primary["flux_total_box"][order][mask_in]
        f_lat = primary["flux_lateral"][order][mask_in]
        ax0.plot(
            wl_b,
            10.0 * np.log10(np.maximum(f_box, 1e-12)),
            color="0.3",
            ls=":",
            lw=1.7,
            label=f"2D Poynting flux box (mean={np.mean(f_box):.3f}, lat={np.mean(f_lat) * 100:.2f}%)",
        )
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Normalized power [dB]",
        ylim=(-0.55, 0.15),
        title="(a) Insertion loss & 2D Poynting flux-box audit",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower left", fontsize=7.8)

    dphi_deg = np.rad2deg(np.angle(s31[mask_in] * np.conj(s41[mask_in])))
    ax1.plot(
        wl_b,
        dphi_deg,
        "o-",
        color="#2b5c8f",
        lw=2.0,
        ms=4.5,
        label=r"FIT relative phase $\arg(S_{31} S_{41}^*)$ [deg]",
    )
    ax1.axhline(
        -90.0,
        color="#d95f02",
        ls="--",
        lw=1.7,
        label="Ideal CMT quadrature (−90°)",
    )
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Cross-to-bar phase difference [deg]",
        ylim=(-105.0, -75.0),
        title="(b) Quadrature phase verification arg(S31 / S41)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.0)
    figstyle.save(
        fig,
        fig_dir / "14_energy_budget_and_quadrature_phase.png",
        close=True,
        auto_legend_below=False,
    )

    # 15. Multi-resolution convergence: Kottke vs Staircase relative to kottke_reference
    levels = [
        lvl
        for lvl in ("L1", "L2", "L3")
        if f"kottke_{lvl}" in tasks_data and f"staircase_{lvl}" in tasks_data
    ]
    if levels and "kottke_reference" in tasks_data:
        ref_d = tasks_data["kottke_reference"]
        cpws = [float(tasks_data[f"kottke_{lvl}"]["cpw"]) for lvl in levels]
        cafs = [
            float(tasks_data[f"kottke_{lvl}"].get("cells_across_feature", 4.0))
            for lvl in levels
        ]
        n_gaps = [
            int(tasks_data[f"kottke_{lvl}"].get("n_cells_gap", 4)) for lvl in levels
        ]

        def _tc_at_design(d):
            w = d["wavelengths"]
            ord_d = np.argsort(w)
            return float(np.interp(WL_DESIGN, w[ord_d], np.abs(d["S31"][ord_d]) ** 2))

        def _wlc_nm(d):
            return float(d["wl_cross"]) * 1e9

        def _r_mean(d):
            w = d["wavelengths"] * 1e9
            m = (w >= 1430.0) & (w <= 1670.0)
            return float(np.mean(np.abs(d["S11"][m]) ** 2))

        tc_ref = _tc_at_design(ref_d)
        wlc_ref = _wlc_nm(ref_d)
        ref_lvl_lbl = str(ref_d.get("level", "L3"))
        ref_cpw_i = int(ref_d["cpw"])
        ref_ngap_i = int(ref_d.get("n_cells_gap", 8))
        ref_caf = float(ref_d.get("cells_across_feature", 8.55))

        err_tc_k = [
            max(abs(_tc_at_design(tasks_data[f"kottke_{lvl}"]) - tc_ref), 1e-4)
            for lvl in levels
        ]
        err_tc_s = [
            max(abs(_tc_at_design(tasks_data[f"staircase_{lvl}"]) - tc_ref), 1e-4)
            for lvl in levels
        ]
        err_wl_k = [
            max(abs(_wlc_nm(tasks_data[f"kottke_{lvl}"]) - wlc_ref), 0.05)
            for lvl in levels
        ]
        err_wl_s = [
            max(abs(_wlc_nm(tasks_data[f"staircase_{lvl}"]) - wlc_ref), 0.05)
            for lvl in levels
        ]
        rk_vals = [_r_mean(tasks_data[f"kottke_{lvl}"]) for lvl in levels]
        rs_vals = [_r_mean(tasks_data[f"staircase_{lvl}"]) for lvl in levels]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.8, 4.4), dpi=150)
        res = np.asarray(cpws, dtype=float)
        ax0.loglog(
            res,
            err_tc_s,
            "s-",
            color="#d95f02",
            lw=2.0,
            ms=6.5,
            label=r"Staircase $|T_{\mathrm{cross}} - T_{\mathrm{ref}}|$ (resolved ladder)",
        )
        ax0.loglog(
            res,
            err_tc_k,
            "o-",
            color="#1b9e77",
            lw=2.0,
            ms=6.5,
            label=r"Kottke $|T_{\mathrm{cross}} - T_{\mathrm{ref}}|$ (resolved ladder)",
        )
        for lvl_i, x_i, yk_i, ys_i, caf_i, ng_i in zip(
            levels, res, err_tc_k, err_tc_s, cafs, n_gaps
        ):
            ax0.annotate(
                f"{lvl_i}: {caf_i:.2f}c (N={ng_i})",
                xy=(x_i, yk_i),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=7.3,
                color="#1b9e77",
                fontweight="bold",
            )
            dy_s = 7 if lvl_i == "L2" else -12
            ax0.annotate(
                f"{lvl_i}: {caf_i:.2f}c",
                xy=(x_i, ys_i),
                xytext=(4, dy_s),
                textcoords="offset points",
                fontsize=7.1,
                color="#d95f02",
            )

        if "kottke_nonuniform" in tasks_data:
            nu_d = tasks_data["kottke_nonuniform"]
            nu_cpw = float(nu_d["cpw"])
            nu_caf = float(nu_d.get("cells_across_feature", 10.0))
            nu_err_tc = max(abs(_tc_at_design(nu_d) - tc_ref), 1e-4)
            nu_err_wl = max(abs(_wlc_nm(nu_d) - wlc_ref), 0.05)
            ax0.loglog(
                [nu_cpw],
                [nu_err_tc],
                "*",
                color="#7570b3",
                ms=10.0,
                zorder=6,
                label=f"Graded non-uniform Kottke ({nu_caf:.1f}c, −38% cells)",
            )
            ax1.loglog(
                [nu_cpw],
                [nu_err_wl],
                "*",
                color="#7570b3",
                ms=10.0,
                zorder=6,
                label=f"Graded non-uniform Kottke ({nu_caf:.1f}c, −38% cells)",
            )

        # Plot under-resolved demonstration pair (excluded from converged trend lines)
        if (
            "underresolved_kottke" in tasks_data
            and "underresolved_staircase" in tasks_data
        ):
            uk = tasks_data["underresolved_kottke"]
            us = tasks_data["underresolved_staircase"]
            u_cpw = float(uk["cpw"])
            u_caf = float(uk.get("cells_across_feature", 4.21))
            u_err_tc_k = max(abs(_tc_at_design(uk) - tc_ref), 1e-4)
            u_err_tc_s = max(abs(_tc_at_design(us) - tc_ref), 1e-4)
            u_err_wl_k = max(abs(_wlc_nm(uk) - wlc_ref), 0.05)
            u_err_wl_s = max(abs(_wlc_nm(us) - wlc_ref), 0.05)
            ax0.loglog(
                [u_cpw],
                [u_err_tc_k],
                "D",
                color="#2b5c8f",
                mfc="none",
                mew=2.0,
                ms=7.5,
                label=f"Warning: under-resolved Kottke (10 cpw, no box, {u_caf:.2f}c)",
            )
            ax0.loglog(
                [u_cpw],
                [u_err_tc_s],
                "X",
                color="#e7298a",
                mfc="none",
                mew=2.0,
                ms=8.0,
                label=f"Warning: under-resolved Staircase (10 cpw, no box, {u_caf:.2f}c)",
            )
            ax0.annotate(
                f"no override ({u_caf:.2f}c)",
                xy=(u_cpw, u_err_tc_k),
                xytext=(6, -9),
                textcoords="offset points",
                fontsize=7.1,
                color="#2b5c8f",
            )

            ax1.loglog(
                [u_cpw],
                [u_err_wl_k],
                "D",
                color="#2b5c8f",
                mfc="none",
                mew=2.0,
                ms=7.5,
                label=f"Warning: under-resolved Kottke (no box, {u_caf:.2f}c)",
            )
            ax1.loglog(
                [u_cpw],
                [u_err_wl_s],
                "X",
                color="#e7298a",
                mfc="none",
                mew=2.0,
                ms=8.0,
                label=f"Warning: under-resolved Staircase (no box, {u_caf:.2f}c)",
            )
            ax1.annotate(
                f"no override ({u_caf:.2f}c)",
                xy=(u_cpw, u_err_wl_k),
                xytext=(6, -9),
                textcoords="offset points",
                fontsize=7.1,
                color="#2b5c8f",
            )

        ax0.set(
            xlim=(8.8, max(res) * 1.32),
            ylim=(min(err_tc_k) * 0.65, max(err_tc_s) * 1.55),
            xlabel="Background resolution [cells per wavelength] (cpw)",
            ylabel=r"Cross-port power error $|T_{\mathrm{cross}}(1530\;\mathrm{nm}) - T_{\mathrm{ref}}|$",
            title=f"(a) Power error vs Kottke ref {ref_lvl_lbl}=({ref_cpw_i},{ref_ngap_i}) ({ref_caf:.2f}c)",
        )
        ax0.set_xticks(res)
        ax0.set_xticklabels([f"{int(r)} cpw ({l})" for r, l in zip(res, levels)], fontsize=8.0)
        ax0.xaxis.set_minor_formatter(plt.NullFormatter())
        ax0.grid(True, which="both", ls=":", alpha=0.4)
        ax0.legend(loc="center right", fontsize=6.8)

        ax0_top = ax0.twiny()
        ax0_top._is_inset = True
        ax0_top.set_xscale("log")
        ax0_top.set_xlim(ax0.get_xlim())
        ax0_top.set_xticks(res)
        ax0_top.set_xticklabels([f"{c:.2f} cells ({l})" for c, l in zip(cafs, levels)], fontsize=7.8)
        ax0_top.xaxis.set_minor_formatter(plt.NullFormatter())
        ax0_top.set_xlabel("Cells across smallest feature (geff = 207.0 nm)", fontsize=8.2)

        ax1.loglog(
            res,
            err_wl_s,
            "s-",
            color="#d95f02",
            lw=2.0,
            ms=6.5,
            label=r"Staircase $|\lambda_{\mathrm{cross}} - \lambda_{\mathrm{ref}}|$ [nm]",
        )
        ax1.loglog(
            res,
            err_wl_k,
            "o-",
            color="#1b9e77",
            lw=2.0,
            ms=6.5,
            label=r"Kottke $|\lambda_{\mathrm{cross}} - \lambda_{\mathrm{ref}}|$ [nm]",
        )
        for lvl_i, x_i, yk_i, ys_i, caf_i, ng_i in zip(
            levels, res, err_wl_k, err_wl_s, cafs, n_gaps
        ):
            dy_k = -11 if lvl_i == "L3" else 5
            ax1.annotate(
                f"{lvl_i}: {caf_i:.2f}c (N={ng_i})",
                xy=(x_i, yk_i),
                xytext=(4, dy_k),
                textcoords="offset points",
                fontsize=7.3,
                color="#1b9e77",
                fontweight="bold",
            )
            dy_s = 7 if lvl_i == "L2" else -12
            ax1.annotate(
                f"{lvl_i}: {caf_i:.2f}c",
                xy=(x_i, ys_i),
                xytext=(4, dy_s),
                textcoords="offset points",
                fontsize=7.1,
                color="#d95f02",
            )

        ax1_r = ax1.twinx()
        ax1_r._is_inset = True
        ax1_r.loglog(
            res,
            rs_vals,
            "^--",
            color="#7570b3",
            lw=1.5,
            ms=5,
            label=r"Staircase mean $|S_{11}|^2$",
        )
        ax1_r.loglog(
            res,
            rk_vals,
            "v:",
            color="#2b5c8f",
            lw=1.5,
            ms=5,
            label=r"Kottke mean $|S_{11}|^2$",
        )
        ax1_r.set_ylabel(r"Band-averaged reflection $\langle |S_{11}|^2 \rangle$", color="#2b5c8f")
        ax1.set(
            xlim=(8.8, max(res) * 1.32),
            ylim=(min(err_wl_k) * 0.65, max(err_wl_s) * 1.55),
            xlabel="Background resolution [cells per wavelength] (cpw)",
            ylabel=r"50:50 wavelength error $|\lambda_{\mathrm{cross}} - \lambda_{\mathrm{ref}}|$ [nm]",
            title="(b) Crossing-wavelength error & spurious reflection",
        )
        ax1.set_xticks(res)
        ax1.set_xticklabels([f"{int(r)} cpw ({l})" for r, l in zip(res, levels)], fontsize=8.0)
        ax1.xaxis.set_minor_formatter(plt.NullFormatter())
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax1_r.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="center right", fontsize=6.5)

        ax1_top = ax1.twiny()
        ax1_top._is_inset = True
        ax1_top.set_xscale("log")
        ax1_top.set_xlim(ax1.get_xlim())
        ax1_top.set_xticks(res)
        ax1_top.set_xticklabels([f"{c:.2f} cells ({l})" for c, l in zip(cafs, levels)], fontsize=7.8)
        ax1_top.xaxis.set_minor_formatter(plt.NullFormatter())
        ax1_top.set_xlabel("Cells across smallest feature (geff = 207.0 nm)", fontsize=8.2)

        figstyle.save(
            fig,
            fig_dir / "15_kottke_vs_staircase_convergence.png",
            close=True,
            auto_legend_below=False,
        )

    # 16. Port time trace and energy decay
    if "trace_t" in primary and primary["trace_t"].size > 0:
        from wakis.sources import GaussianSource

        t_sec = primary["trace_t"].astype(float)
        t_fs = t_sec * 1e15
        ey_tr = primary["trace_Ey"].astype(float)
        m_peak = float(primary.get("movie_peak", np.max(np.abs(ey_tr))))
        ed_thr = 1e-4 if primary_name == "kottke_L3" else 1e-3

        pulse_ref = GaussianSource(frequency=F_0, fwidth=F_MAX - F_MIN)
        src_t = pulse_ref(t_sec)
        f_arr = primary["frequencies"].astype(float)
        ord_f = np.argsort(f_arr)
        f_s = f_arr[ord_f]
        s31_s = primary["S31"][ord_f]
        s41_s = primary["S41"][ord_f]
        f_fine = np.linspace(f_s[0], f_s[-1], 201)
        s31_f = np.interp(f_fine, f_s, np.abs(s31_s)) * np.exp(
            1j * np.interp(f_fine, f_s, np.unwrap(np.angle(s31_s)))
        )
        s41_f = np.interp(f_fine, f_s, np.abs(s41_s)) * np.exp(
            1j * np.interp(f_fine, f_s, np.unwrap(np.angle(s41_s)))
        )
        dt_tr = float(t_sec[1] - t_sec[0]) if t_sec.size > 1 else 1e-17
        df_f = float(f_fine[1] - f_fine[0])
        u_in = (
            np.sum(
                src_t[None, :]
                * np.exp(-1j * 2.0 * np.pi * f_fine[:, None] * t_sec[None, :]),
                axis=1,
            )
            * dt_tr
        )
        win_f = np.hanning(len(f_fine))
        ey_cross_t = np.real(
            np.sum(
                (u_in * win_f * s31_f)[:, None]
                * np.exp(+1j * 2.0 * np.pi * f_fine[:, None] * t_sec[None, :]),
                axis=0,
            )
            * 2.0
            * df_f
        )
        ey_bar_t = np.real(
            np.sum(
                (u_in * win_f * s41_f)[:, None]
                * np.exp(+1j * 2.0 * np.pi * f_fine[:, None] * t_sec[None, :]),
                axis=0,
            )
            * 2.0
            * df_f
        )
        inc_peak = max(float(np.max(np.abs(src_t))), 1e-30)
        ey_cross_n = ey_cross_t / inc_peak
        ey_bar_n = ey_bar_t / inc_peak
        ey_probe_n = ey_tr / max(m_peak, 1e-30)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 3.9), dpi=150)
        ax0.plot(
            t_fs,
            src_t / inc_peak,
            ":",
            color="0.55",
            lw=1.2,
            label="Input pulse (`in_bot`)",
        )
        ax0.plot(
            t_fs,
            ey_cross_n,
            color="#1b9e77",
            lw=1.5,
            label="Cross port (`out_top`, S31)",
        )
        ax0.plot(
            t_fs,
            ey_bar_n,
            "--",
            color="#d95f02",
            lw=1.3,
            alpha=0.85,
            label="Bar port (`out_bot`, S41)",
        )
        ax0.set(
            xlabel="Time t [fs]",
            ylabel=r"Normalized field $E_y(t) / |E_y|_{\mathrm{inc},\max}$",
            title="(a) Input, cross & bar time-domain waveforms",
        )
        ax0.grid(True, ls=":", alpha=0.4)
        ax0.legend(loc="upper right", fontsize=8.0)

        env_cross = np.maximum(ey_cross_n**2 + ey_bar_n**2, 1e-10)
        env_probe = np.maximum(ey_probe_n**2, 1e-10)
        ax1.semilogy(
            t_fs,
            env_cross,
            color="#1b9e77",
            lw=1.4,
            label=r"Output ports $|E_{y,\mathrm{out}}(t)|^2 / |E_y|_{\mathrm{inc}}^2$",
        )
        ax1.semilogy(
            t_fs,
            env_probe,
            color="#2b5c8f",
            lw=1.1,
            alpha=0.8,
            label=r"Isolated input probe $|E_y(t)|^2 / |E_y|_{\max}^2$",
        )
        if "movie_times" in primary and "movie_frames" in primary:
            mt_fs = primary["movie_times"] * 1e15
            mf_pk = np.array(
                [float(np.max(np.abs(fr))) ** 2 for fr in primary["movie_frames"]]
            )
            m_pos = mf_pk > 0
            ax1.semilogy(
                mt_fs[m_pos],
                mf_pk[m_pos],
                "o--",
                color="#d95f02",
                ms=5.0,
                lw=1.4,
                label=r"2D snapshot peak $|E_y|_{\mathrm{dom}}^2 / |E_y|_{\max}^2$",
            )
        ax1.axhline(
            ed_thr,
            color="0.35",
            ls="--",
            lw=1.3,
            label=f"Energy decay threshold ({ed_thr:.0e})",
        )
        ax1.set(
            xlabel="Time t [fs]",
            ylabel="Normalized instantaneous power (log scale)",
            ylim=(1e-9, 2.5),
            title="(b) Ring-down & termination audit",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="lower left", fontsize=7.3)
        figstyle.save(
            fig,
            fig_dir / "16_port_time_trace.png",
            close=True,
            auto_legend_below=False,
        )

    # Also plot port dispersion if available (updates port neff/ng check)
    if "disp_wavelengths" in primary:
        plot_neff_dispersion(
            primary["disp_wavelengths"] * 1e9,
            primary["disp_neff"],
            neff_slab=float(primary["n_eff_slab"]),
            ng_2d=primary["disp_ng"],
            out_path=fig_dir / "05_port_mode_profiles.png"
            if False
            else None,  # kept in memory check
        )
        plt.close("all")

    print(f"Rendered 16 figures in {fig_dir}")


# %% [markdown]
# ## 7. Supermode dispersion, CMT validation & subpixel convergence
#
# In `06_supermode_profiles_and_dispersion.png`, the discrete FIT eigenmode
# solver (`solve_modes` at $x = 0$ on the primary `--full` $L_3 = (20, 8)$ mesh
# with $8.55$ cells across $g_{\text{eff}} = 207.02\;\text{nm}$,
# $\Delta y_{\min} = 24.20\;\text{nm}$) resolves the symmetric
# ($\text{TE}_{0,\text{even}}$) and antisymmetric ($\text{TE}_{0,\text{odd}}$)
# supermodes of the coupled waveguide pair. At $\lambda = 1.530\;\mu\text{m}$,
# the isolated input/output waveguide mode (`05_port_mode_profiles.png`) has
# $n_{\text{eff}} = 2.0032$ and group index $n_g = 3.668 \approx 3.67$
# ($2.0084$ and $3.69$ at $L_2 = (14, 6)$ in `--quick`), while the coupled pair
# splits into $n_{\text{even}}(1530\;\text{nm}) = 2.0417$ and
# $n_{\text{odd}}(1530\;\text{nm}) = 1.9657$ ($\Delta n_{\text{eff}} = 0.07605$,
# crossover beat length $L_\pi(1530\;\text{nm}) = 10.08\;\mu\text{m}$; vs
# $2.0569$, $1.9805$, $\Delta n_{\text{eff}} = 0.07641$, and
# $L_\pi = 10.01\;\mu\text{m}$ in `--quick`). Because longer wavelengths are
# less confined to the silicon cores, their evanescent tails overlap more
# strongly across the central air gap: as $\lambda$ increases from
# $1.40\;\mu\text{m}$ to $1.70\;\mu\text{m}$, $\Delta n_{\text{eff}}(\lambda)$
# grows $3.7\times$ from $0.0411$ to $0.1519$ and $L_\pi(\lambda)$ drops from
# $17.03\;\mu\text{m}$ to $5.60\;\mu\text{m}$.
#
# Accounting for the non-negligible coupling accumulated inside the incoming and
# outgoing cosine S-bends ($L_{\text{sb,eff}} = 1.374\;\mu\text{m}$ at
# $1530\;\text{nm}$, varying from $1.28\;\mu\text{m}$ at $1400\;\text{nm}$ to
# $1.53\;\mu\text{m}$ at $1700\;\text{nm}$), the analytical CMT prediction
# $T_{\text{cross}}^{\text{CMT}}(L_c, \lambda) = \sin^2[\pi(L_c + L_{\text{sb,eff}}) / (2 L_\pi)]$
# gives $\lambda_{\text{cross}}^{\text{CMT}} = 1513.32\;\text{nm}$ ($1512.5\;\text{nm}$
# in `--quick`) and tracks the full-wave 2D FIT S-parameters across both
# wavelength (`10_sparameter_spectra_linear_and_db.png`) and interaction length
# $L_c \in [0, 8]\;\mu\text{m}$ (`11_coupling_length_sweep_cmt_vs_fit.png`).
# Across the 2D Poynting flux box (`14_energy_budget_and_quadrature_phase.png`),
# the band-averaged ($1.42\text{--}1.68\;\mu\text{m}$) guided forward power at
# $L_3 = (20, 8)$ is $99.17\%$ with a total Poynting box balance of $100.00\%$
# ($99.996\%$), $0.31\%$ lateral S-bend radiation, and $0.0444\%$ return loss
# ($\langle |S_{11}|^2 \rangle = -33.53\;\text{dB}$ across $1.43\text{--}1.67\;\mu\text{m}$,
# improving to $0.0414\%$ or $-33.83\;\text{dB}$ at $R = (40, 16)$; vs
# $99.24\%$ guided, $0.44\%$ lateral, and $0.086\%$ or $-30.7\;\text{dB}$ return
# loss in `--quick`).
#
# ### Coupling-length sweep ($L_c \in [0, 8]\;\mu\text{m}$): 2D FIT vs supermode CMT
#
# | $L_c\;[\mu\text{m}]$ | $T_{\text{cross}}(1530)\;\text{Quick } L_2$ | $T_{\text{cross}}(1530)\;\text{Full } L_3$ | $\eta_{\times}(1530)\;\text{Full } L_3$ | $\eta_{\times}^{\text{CMT}}(1530)\;\text{Full } L_3$ | $\lambda_{\text{cross}}\;\text{Full } L_3\;[\text{nm}]$ | $\lambda_{\text{cross}}^{\text{CMT}}\;\text{Full } L_3\;[\text{nm}]$ |
# |---:|---:|---:|---:|---:|---:|---:|
# | $0.0$ | $0.0715$ | $0.0715$ | $0.0715$ | $0.0452$ | $>1700$ (under-coupled) | $>1700$ (under-coupled) |
# | $2.0$ | $0.3022$ | $0.3002$ | $0.3020$ | $0.2521$ | $1606.15$ | $1633.89$ |
# | $4.0$ | $0.6119$ | $0.6069$ | $0.6091$ | $0.5522$ | $1497.56$ | $1513.32$ |
# | $6.0$ | $0.8757$ | $0.8702$ | $0.8721$ | $0.8328$ | $1424.95$ | $1436.14$ |
# | $8.0$ | $0.9923$ | $0.9891$ | $0.9955$ | $0.9881$ | $1649.31$ ($\theta=3\pi/4$) | $1661.20$ ($\theta=3\pi/4$) |
#
# Here $\eta_{\times} = T_{\text{cross}} / (T_{\text{cross}} + T_{\text{bar}})$
# is the normalized cross-port splitting ratio. At $L_c = 8.0\;\mu\text{m}$, the
# coupler reaches near-complete crossover at $1530\;\text{nm}$
# ($\eta_{\times} = 99.55\%$ vs $98.81\%$ CMT) and passes through $50:50$
# splitting a second time at $\lambda_{\text{cross}} = 1649.31\;\text{nm}$ on the
# over-coupled $\theta = 3\pi/4$ branch (`11_coupling_length_sweep_cmt_vs_fit.png`,
# panel b).
#
# ### Resolution & mesh comparison: `--quick`, `--full` baseline, graded non-uniform mesh & CMT
#
# | Quantity ($L_c = 4\;\mu\text{m}, \theta_{\text{sw}} = 30^\circ$) | `--quick` $L_2=(14,6)$ | `--full` Baseline $L_3=(20,8)$ | `--full` Graded Non-Uniform (`task_20`) | `--full` Reference $R=(40,16)$ | Analytic CMT / Theory |
# |---|---:|---:|---:|---:|---|
# | Grid dimensions ($N_x \times N_y \times N_z$) | $492 \times 203 \times 1$ | $726 \times 267 \times 1$ | $567 \times 211 \times 1$ | $1406 \times 518 \times 1$ | 2D EIM ($N_z = 1$) |
# | Total Yee cells $N_{\text{cells}}$ | $99{,}974$ | $193{,}842$ | $119{,}637$ ($-38.3\%$ vs $L_3$) | $728{,}280$ | — |
# | Gap resolution ($\Delta y_{\min}$, cells across $g_{\text{eff}}$) | $34.37\;\text{nm}$ ($6.02\;\text{cells}$) | $24.20\;\text{nm}$ ($8.55\;\text{cells}$) | $20.31\;\text{nm}$ ($10.00\;\text{cells}$) | $12.22\;\text{nm}$ ($16.95\;\text{cells}$) | $\ge 4\;\text{cells}$ across $207.02\;\text{nm}$ |
# | Wall-clock runtime $[\text{s}]$ | $111.0\;\text{s}$ | $346.5\;\text{s}$ | $162.9\;\text{s}$ ($-53.0\%$ vs $L_3$) | $2011.3\;\text{s}$ | — |
# | Port mode index $n_{\text{eff}}(1530\;\text{nm})$ | $2.0084$ | $2.0032$ | $2.0041$ | — | $n_{\text{eff,slab}} = 2.8424$ (1D vertical slab) |
# | Port group index $n_g(1530\;\text{nm})$ | $3.69$ | $3.668$ | $3.671$ | — | $\sim 3.7\text{--}4.1$ (2D EIM vs 3D SOI [3]) |
# | Supermode indices $n_{\text{even}}, n_{\text{odd}}(1530\;\text{nm})$ | $2.0569, 1.9805$ | $2.0417, 1.9657$ | — | — | $\Delta n_{\text{eff}} = 0.07605$ (2D FIT eigenmodes) |
# | Crossover beat length $L_\pi(1530\;\text{nm})$ | $10.01\;\mu\text{m}$ | $10.08\;\mu\text{m}$ | — | — | $\lambda / (2\Delta n_{\text{eff}}) = 10.08\;\mu\text{m}$ (CMT [1, 2]) |
# | S-bend effective length $L_{\text{sb,eff}}(1530\;\text{nm})$ | $1.38\;\mu\text{m}$ | $1.374\;\mu\text{m}$ | — | — | $2\int \Delta n_{\text{eff}}(s)\,\mathrm{d}s / \Delta n_{\text{eff}}(0) = 1.374\;\mu\text{m}$ |
# | 50:50 wavelength $\lambda_{\text{cross}}$ | $1496.13\;\text{nm}$ | $1497.56\;\text{nm}$ | $1495.81\;\text{nm}$ ($+1.03\;\text{nm}$ vs $R$) | $1494.79\;\text{nm}$ | $1513.32\;\text{nm}$ (2D CMT) |
# | Cross transmittance $T_{\text{cross}}(1530\;\text{nm})$ | $0.6097$ | $0.6069$ | $0.6123$ ($-0.67\%$ vs $R$) | $0.6164$ | $0.5522$ (2D CMT) |
# | Bar transmittance $T_{\text{bar}}(1530\;\text{nm})$ | $0.3866$ | $0.3895$ | $0.3845$ | $0.3802$ | $0.4478$ (2D CMT) |
# | Total guided power $T_{\text{tot}}(\lambda_{\text{cross}})$ | $0.9964$ | $0.9968$ | $0.9972$ ($0.012\;\text{dB}$ IL) | $0.9966$ | $1.000$ ($0.00\;\text{dB}$ lossless CMT) |
# | Band-averaged reflection $\langle |S_{11}|^2 \rangle$ | $0.0575\%$ ($-32.4\;\text{dB}$) | $0.0444\%$ ($-33.5\;\text{dB}$) | $0.0457\%$ ($-33.4\;\text{dB}$) | $0.0414\%$ ($-33.8\;\text{dB}$) | $< -30\;\text{dB}$ (adiabatic cosine S-bends [3]) |
# | Cross/bar phase $\arg(S_{31}S_{41}^*)(1530\;\text{nm})$ | $-90.34^\circ$ | $-90.30^\circ$ | $-90.30^\circ$ | $-90.31^\circ$ | $-90.00^\circ$ (lossless CMT transfer matrix [1, 2]) |
# | Quadrature routing $P_{\text{up}}, P_{\text{down}}(\lambda_{\text{cross}})$ | $99.95\%, 0.044\%$ ($33.5\;\text{dB}$) | $99.65\%, 0.050\%$ ($33.0\;\text{dB}$) | — | — | $100\%, 0\%$ ($\infty\;\text{dB}$ ideal lossless CMT) |
#
# ### Graded non-uniform Cartesian mesh upgrade (`04b_mesh_upgrade_comparison.png`)
#
# Because `PolySlab` structures represent extruded 2D polygons whose axis-aligned
# bounding boxes span the entire $17.3 \times 4.34\;\mu\text{m}^2$ waveguide
# footprint, standard `PICSimulation` assigns the core cell size
# $\Delta x_{\text{core}} \approx 24.5\;\text{nm}$ uniformly across the entire
# S-bend and cladding envelope (`kottke_L3`, $726 \times 267 \times 1 = 193{,}842\;\text{cells}$).
# In `04b_mesh_upgrade_comparison.png`, `NonUniformCouplerPIC`
# (`build_coupler_graded_axes`, built with `wakis.templates.meshing.graded_axis`)
# exploits the directional coupler's multi-scale physics while preserving exact
# $x \mapsto -x$ and $y \mapsto -y$ grid mirror symmetry:
#
# 1. **Exact gap pinning & transverse grading ($\Delta y(y)$)**: In $y$, the mesh
#    places $10.00$ exact cells across the $g_{\text{eff}} = 207.02\;\text{nm}$
#    central coupling slot ($\Delta y_{\min} = 20.31\;\text{nm}$ at the slot edges,
#    $\Delta y_{\text{gap}} = 20.70\;\text{nm}$), resolves the coupled cores and
#    the input/output port arms ($|y \mp 2.0\;\mu\text{m}| \le w_{\text{eff}}$) at
#    $\Delta y_{\text{core}} \approx 22\text{--}24\;\text{nm}$, uses an
#    intermediate $35.0\;\text{nm}$ spacing across the S-bend transition corridor
#    ($0.50\;\mu\text{m} \le |y| \le 1.66\;\mu\text{m}$), and grades smoothly out
#    to $\Delta y_{\text{clad}} = 51.3\;\text{nm}$ in the outer air cladding
#    ($|y| > 2.34\;\mu\text{m}$), reducing $N_y$ from $267$ to $211$.
# 2. **Non-square axial grading ($\Delta x(x) > \Delta y_{\min}$)**: Because the
#    guided supermodes vary along the propagation axis $x$ on the characteristic
#    wavelength scale $\lambda_{\min}/n_{\text{eff,slab}} = 492.5\;\text{nm}$
#    rather than the sub-wavelength transverse evanescent scale, `build_coupler_graded_axes`
#    uses non-square cells with $\Delta x = 25.7\;\text{nm}$ inside the central
#    coupling region ($|x| \le 2.5\;\mu\text{m}$) and $\Delta x = 36.9\;\text{nm}$
#    along the adiabatic S-bends and port leads ($|x| > 2.5\;\mu\text{m}$),
#    reducing $N_x$ from $726$ to $567$.
# 3. **Equal-or-better accuracy at $-38.3\%$ cell count and $2.13\times$ speedup**:
#    Compared to the baseline `kottke_L3` mesh ($193{,}842\;\text{cells}$,
#    $346.5\;\text{s}$), `kottke_nonuniform` (`task_20`, $119{,}637\;\text{cells}$,
#    $162.9\;\text{s}$) reduces the total cell count by **$38.3\%$** ($6.09\times$
#    fewer cells than `kottke_reference`) and cuts wall-clock runtime by
#    **$53.0\%$** ($2.13\times$ faster than `kottke_L3` and $12.35\times$ faster
#    than `kottke_reference`). Simultaneously, because its $10.00$ exact gap cells
#    pin the $y = \pm g_{\text{eff}}/2$ slot boundaries directly onto Yee grid
#    nodes, `kottke_nonuniform` achieves $\lambda_{\text{cross}} = 1495.81\;\text{nm}$
#    ($+1.03\;\text{nm}$ or $0.069\%$ vs the $R = (40, 16)$ reference
#    $1494.79\;\text{nm}$, **$2.7\times$ closer to $R$ than `kottke_L3`'s
#    $+2.78\;\text{nm}$ offset**) and $T_{\text{cross}}(1530\;\text{nm}) = 0.6123$
#    ($-0.67\%$ vs $R$'s $0.6164$, vs $-1.54\%$ for `kottke_L3`).
#
# > **2D EIM vs 1D slice-quadrature CMT**: In the 2D EIM reduction, the 3D
# > trapezoidal core ($w_{\text{bot}} = 470\;\text{nm}$,
# > $w_{\text{top}} = 216.0\;\text{nm}$ at $\theta_{\text{sw}} = 30^\circ$) is
# > mapped to its mid-height width $w_{\text{eff}} = 343.0\;\text{nm}$ and gap
# > $g_{\text{eff}} = 207.02\;\text{nm}$ with air cladding ($n_{\text{clad}} = 1.00$)
# > in the coupling slot. The $-15.75\;\text{nm}$ ($1.04\%$) offset between
# > full-wave 2D FIT ($\lambda_{\text{cross}} = 1497.56\;\text{nm}$ at $L_3$,
# > $1494.79\;\text{nm}$ at $R$) and 1D slice-quadrature CMT
# > ($1513.32\;\text{nm}$) arises from weak non-adiabatic wavefront tilt in the
# > incoming/outgoing cosine S-bends, which adds $\sim 0.2\;\mu\text{m}$ of
# > effective coupling beyond the 5-station local-normal supermode integral.
#
# ### Subpixel convergence across the `--full` ladder ($L_1, L_2, L_3, R$)
#
# In `15_kottke_vs_staircase_convergence.png`, comparing Kottke subpixel
# smoothing against staircasing across the $(\text{cpw}, N)$ ladder relative to
# the independent `--full` Kottke reference (`kottke_reference` at
# $R = (40, 16)$ with $16.95$ cells across the gap, $\Delta y_{\min} = 12.22\;\text{nm}$,
# $728{,}280$ cells, $\lambda_{\text{cross}} = 1494.79\;\text{nm}$,
# $T_{\text{cross}}(1530\;\text{nm}) = 0.6164$) establishes two conclusions:
#
# 1. **Kottke with coupling-region mesh overrides reaches the $<0.2\%$ subpixel floor already at $L_1$**:
#    Across the resolved Kottke ladder, $\lambda_{\text{cross}}$ is $1495.47\;\text{nm}$
#    at $L_1 = (10, 4)$ ($+0.69\;\text{nm}$ or $0.046\%$ vs $R$),
#    $1496.13\;\text{nm}$ at $L_2 = (14, 6)$ ($+1.35\;\text{nm}$ or $0.090\%$ vs $R$),
#    and $1497.56\;\text{nm}$ at $L_3 = (20, 8)$ ($+2.78\;\text{nm}$ or $0.186\%$ vs $R$),
#    while $T_{\text{cross}}(1530\;\text{nm})$ is $0.6099$ ($L_1$, $-1.05\%$ vs $R$),
#    $0.6097$ ($L_2$, $-1.09\%$ vs $R$), and $0.6069$ ($L_3$, $-1.54\%$ vs $R$).
#    Between the last two Kottke levels ($L_3 = (20, 8)$ and $R = (40, 16)$),
#    the relative change is $0.186\%$ ($2.78\;\text{nm}$) in $\lambda_{\text{cross}}$,
#    $1.54\%$ ($0.0095$) in $T_{\text{cross}}(1530\;\text{nm})$, and $0.023\%$ in
#    $T_{\text{tot}}(\lambda_{\text{cross}})$, while the band-averaged return
#    loss $\langle |S_{11}|^2 \rangle$ converges strictly monotonically from
#    $0.0987\%$ ($L_1$) $\to 0.0575\%$ ($L_2$) $\to 0.0444\%$ ($L_3$) $\to 0.0414\%$ ($R$).
#    Across the resolved ladder, $L_1, L_2, L_3,$ `kottke_nonuniform`, and $R$
#    all sit within a $\pm 1.4\;\text{nm}$ ($<0.2\%$) subpixel band where small
#    shifts reflect how the grid lines align with the $y = \pm 103.51\;\text{nm}$
#    gap edges and polygonal S-bend segments. By contrast, the unrefined
#    demonstration run `underresolved_kottke` ($10\;\text{cpw}$, no override box,
#    $4.21$ unpinned cells) shifts $\lambda_{\text{cross}}$ to $1505.55\;\text{nm}$
#    ($+10.77\;\text{nm}$ or $0.72\%$ error vs $R$, $|\Delta T_{\text{cross}}| = 3.79 \times 10^{-2}$
#    or $6.15\%$ relative error).
# 2. **Staircasing remains inaccurate even at $L_3 = (20, 8)$ with $8.55$ cells across the gap**:
#    Without subpixel smoothing, snapping the $343.0\;\text{nm}$ waveguide cores
#    and curved S-bend sidewalls to the nearest Yee cell edges quantizes both
#    $w_{\text{eff}}$ and $g_{\text{eff}}$ by $\pm \Delta y_{\min}/2$ and
#    introduces staircase scattering along the $6\;\mu\text{m}$ S-bends. As a
#    result, `underresolved_staircase` ($\lambda_{\text{cross}} = 1464.16\;\text{nm}$,
#    $-30.63\;\text{nm}$ error vs $R$, $\langle |S_{11}|^2 \rangle = 5.25\%$),
#    `staircase_L1` ($\lambda_{\text{cross}} = 1536.88\;\text{nm}$,
#    $+42.09\;\text{nm}$ error, $25.2\%$ error in $T_{\text{cross}}(1530)$,
#    $\langle |S_{11}|^2 \rangle = 2.31\%$), `staircase_L2`
#    ($\lambda_{\text{cross}} = 1567.05\;\text{nm}$, $+72.27\;\text{nm}$ error,
#    $38.2\%$ error in $T_{\text{cross}}(1530)$, $\langle |S_{11}|^2 \rangle = 1.76\%$),
#    and `staircase_L3` ($\lambda_{\text{cross}} = 1456.62\;\text{nm}$,
#    $-38.17\;\text{nm}$ error, $20.8\%$ error in $T_{\text{cross}}(1530)$,
#    $\langle |S_{11}|^2 \rangle = 0.145\%$) exhibit $31\text{--}72\;\text{nm}$
#    crossing-wavelength swings ($14\text{--}53\times$ larger than Kottke at the
#    same mesh resolution) and up to $40\times$ higher spurious back-reflection.
#
# ## 8. Sidewall angle sensitivity & dual-port quadrature routing
#
# In `12_sidewall_angle_sweep.png`, sweeping the reactive-ion-etch sidewall
# angle over $\theta_{\text{sw}} \in [0^\circ, 30^\circ]$ at the full-study
# resolution $L_3 = (20, 8)$ (`task_14`..`task_19` and `task_6`) with per-angle
# mesh overrides ($\Delta y_{\min} = 10.00\;\text{nm}$ at $0^\circ$ up to
# $24.20\;\text{nm}$ at $30^\circ$, maintaining $8.00\text{--}8.55$ cells across
# every angle's $g_{\text{eff}}$; $6.02\text{--}6.12$ cells at $L_2 = (14, 6)$
# in `--quick`) at fixed bottom mask pitch $p = 550\;\text{nm}$ and bottom width
# $w_{\text{bot}} = 470\;\text{nm}$ reveals two competing physical effects:
#
# | $\theta_{\text{sw}}\;[\text{deg}]$ | $w_{\text{eff}}\;[\text{nm}]$ | $g_{\text{eff}}\;[\text{nm}]$ | $n_{\text{even}}(1530)$ | $n_{\text{odd}}(1530)$ | $\Delta n_{\text{eff}}(1530)$ | $L_\pi(1530)\;[\mu\text{m}]$ | $\lambda_{\text{cross}}\;\text{Quick } L_2\;[\text{nm}]$ | $\lambda_{\text{cross}}\;\text{Full } L_3\;[\text{nm}]$ | $\lambda_{\text{cross}}^{\text{CMT}}\;\text{Full } L_3\;[\text{nm}]$ | $T_{\text{tot}}(\lambda_{\text{cross}})\;\text{Full } L_3$ |
# |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | $0^\circ$ | $470.00$ | $80.00$ | $2.4362$ | $2.3737$ | $0.06253$ | $12.23$ | $1570.5$ | $1572.66$ | $1600.87$ | $0.9856$ |
# | $5^\circ$ | $450.75$ | $99.25$ | $2.3933$ | $2.3350$ | $0.05829$ | $13.12$ | $1590.5$ | $1594.59$ | $1619.62$ | $0.9891$ |
# | $10^\circ$ | $431.21$ | $118.79$ | $2.3441$ | $2.2877$ | $0.05636$ | $13.57$ | $1594.0$ | $1600.09$ | $1622.02$ | $0.9948$ |
# | $15^\circ$ | $411.05$ | $138.95$ | $2.2881$ | $2.2314$ | $0.05668$ | $13.50$ | $1586.1$ | $1591.67$ | $1611.75$ | $0.9968$ |
# | $20^\circ$ | $389.93$ | $160.07$ | $2.2251$ | $2.1657$ | $0.05947$ | $12.86$ | $1568.1$ | $1570.60$ | $1589.22$ | $0.9980$ |
# | $25^\circ$ | $367.41$ | $182.59$ | $2.1422$ | $2.0763$ | $0.06589$ | $11.61$ | $1537.8$ | $1537.51$ | $1555.43$ | $0.9980$ |
# | $30^\circ$ | $342.98$ | $207.02$ | $2.0417$ | $1.9657$ | $0.07599$ | $10.07$ | $1496.4$ | $1497.56$ | $1513.32$ | $0.9968$ |
#
# 1. **Gap widening** ($g_{\text{eff}} = g_{\text{bot}} + h_{\text{wg}}\tan\theta_{\text{sw}}$
#    increases from $80.00\;\text{nm}$ at $0^\circ$ to $207.02\;\text{nm}$ at
#    $30^\circ$), which initially reduces evanescent coupling between $0^\circ$
#    ($\Delta n_{\text{eff}} = 0.0625$, $\lambda_{\text{cross}} = 1572.66\;\text{nm}$)
#    and $10^\circ$ ($\Delta n_{\text{eff}} = 0.0564$, $\lambda_{\text{cross}} = 1600.09\;\text{nm}$;
#    $1570.5 \to 1594.0\;\text{nm}$ in `--quick`).
# 2. **Core narrowing & mode delocalization** ($w_{\text{eff}} = w_{\text{bot}} - h_{\text{wg}}\tan\theta_{\text{sw}}$
#    shrinks from $470.0\;\text{nm}$ to $343.0\;\text{nm}$), which reduces the
#    isolated waveguide effective index from $\sim 2.41$ to $2.0032$ and expands
#    the lateral evanescent decay length
#    $1 / \gamma_y = [\,k_0\sqrt{n_{\text{eff}}^2 - n_{\text{clad}}^2}\,]^{-1}$.
#
# For $\theta_{\text{sw}} \ge 15^\circ$, mode delocalization dominates and
# monotonically blue-shifts the 50:50 splitting wavelength from $1591.67\;\text{nm}$
# ($15^\circ$) to $1570.60\;\text{nm}$ ($20^\circ$), $1537.51\;\text{nm}$ ($25^\circ$),
# and $1497.56\;\text{nm}$ ($30^\circ$), closely tracking the CMT supermode
# prediction ($1611.75 \to 1589.22 \to 1555.43 \to 1513.32\;\text{nm}$, within
# $1.04\%\text{--}1.76\%$ across all 7 angles) with high guided transmittance
# ($T_{\text{tot}}(\lambda_{\text{cross}}) = 0.9856\text{--}0.9980$).
#
# In `13_quadrature_dual_drive_routing.png`, driving both input arms simultaneously
# with a $\Delta\phi = -\pi/2$ phase offset (`PhasedGaussianSource`, `task_9` at
# $L_3 = (20, 8)$) exploits the intrinsic $-\pi/2$ phase lag of the
# cross-coupled wave ($\arg(S_{31} S_{41}^*) = -90.30^\circ$ at $1530\;\text{nm}$
# at $L_3$ and $-90.31^\circ$ at $R$, verified in `14_energy_budget_and_quadrature_phase.png`)
# to route $99.65\%$ ($99.645\%$) of the optical power into the upper output
# waveguide (`out_top`) while suppressing the lower output arm (`out_bot`) to
# $0.050\%$ ($0.0498\%$, $33.0\;\text{dB}$ extinction ratio; peaking at
# $35.2\;\text{dB}$ at the nearest frequency grid point $\lambda = 1492.2\;\text{nm}$
# where $P_{\text{up}} = 99.64\%$ and $P_{\text{down}} = 0.030\%$) at
# $\lambda_{\text{cross}} = 1497.56\;\text{nm}$, and $98.36\%$ vs $1.27\%$
# ($18.9\;\text{dB}$ extinction ratio) at $\lambda = 1530\;\text{nm}$ (compared
# to $99.95\%$ vs $0.044\%$ at $\lambda_{\text{cross}}$ and $98.54\%$ vs
# $1.35\%$ at $1530\;\text{nm}$ in `--quick`).
#
# ## 9. Limits, failure modes & when 2D EIM breaks down
#
# 1. **Failure mode — under-resolving the coupling gap ($1\text{--}2$ cells across $g_{\text{eff}}$)**:
#    When a uniform grid is run without coupling-region `mesh_overrides` (such
#    as a bare $6\text{--}10\;\text{cpw}$ mesh where $\Delta y_{\text{bg}} \approx 49\text{--}82\;\text{nm}$
#    leaves only $1\text{--}2$ cells across the $g_{\text{eff}} = 80\text{--}119\;\text{nm}$
#    slot at $\theta_{\text{sw}} = 0^\circ\text{--}10^\circ$), the Yee stencil
#    cannot resolve the exponential decay of the symmetric ($\text{TE}_{0,\text{even}}$)
#    and antisymmetric ($\text{TE}_{0,\text{odd}}$) supermodes inside the air
#    slot. With only one or two E-field nodes between the opposing silicon
#    sidewalls, the computed supermode splitting $\Delta n_{\text{eff}}$ and the
#    50:50 crossing wavelength $\lambda_{\text{cross}}$ become dominated by
#    where the grid lines happen to fall relative to the gap edges. Subpixel
#    permittivity averaging (`averaging="kottke"`) corrects first-order boundary
#    perturbations once the field variation is resolved, but it cannot
#    reconstruct a geometry or an evanescent cusp that spans only $1\text{--}2$
#    cells — hence the mandatory $\text{cpw} \ge 10$ and $N \ge 4$ rules enforced
#    via `coupler_mesh_overrides`.
# 2. **3D trapezoidal hybridity vs 2D mid-plane width**: In 3D SOI, angled
#    sidewalls break vertical mirror symmetry ($z \mapsto -z$) even before
#    considering the asymmetric $\text{SiO}_2 / \text{Air}$ cladding, weakly
#    hybridizing quasi-TE and quasi-TM components near the slanted corners and
#    modifying the 3D overlap integral relative to a uniform rectangular strip of
#    mid-height width $w_{\text{eff}}(\theta_{\text{sw}})$.
# 3. **Etched trench index overestimation**: Standard 2D EIM assigns the lateral
#    cladding index $n_{\text{clad}} = 1.00$ (air) to the etched gap between the
#    waveguides, neglecting the evanescent field penetration into the underlying
#    $\text{SiO}_2$ substrate ($n_{\text{sub}} = 1.45$) beneath the trench. The
#    **variational effective-index method (VEIM)** (Hammer & Ivanova 2009 [4])
#    replaces $n_{\text{clad}}^2$ by the vertical slab-mode-weighted average
#    $\int n_{\text{trench}}^2(z)|\psi(z)|^2\mathrm{d}z \approx 1.15^2$, which
#    slightly increases evanescent coupling at narrow gaps ($0^\circ\text{--}15^\circ$).
# 4. **Slab & material chromatic dispersion**: Freezing $n_{\text{eff,slab}}$ at
#    $\lambda_0 = 1.5355\;\mu\text{m}$ ($2.8424$) across the wide
#    $1.40\text{--}1.70\;\mu\text{m}$ span neglects the vertical slab dispersion
#    $\mathrm{d}n_{\text{eff,slab}}/\mathrm{d}\lambda \approx -0.49\;\mu\text{m}^{-1}$
#    shown in `02_eim_vertical_stack.png`.
#
# ## 10. Exercises
#
# 1. **Complete crossover ($100\%$ cross state)**: Using the CMT beat length
#    $L_\pi(1530\;\text{nm}) = 10.08\;\mu\text{m}$ and S-bend effective length
#    $L_{\text{sb,eff}}(1530\;\text{nm}) = 1.37\;\mu\text{m}$ from `task_6.npz`
#    (`kottke_L3`; or $10.01\;\mu\text{m}$ and $1.38\;\mu\text{m}$ from
#    `kottke_L2` in `--quick`), calculate the straight coupling length
#    $L_c^{\times} = L_\pi - L_{\text{sb,eff}} \approx 8.70\;\mu\text{m}$
#    ($8.63\;\mu\text{m}$ in `--quick`) required for $100\%$ cross-port transfer
#    at $1530\;\text{nm}$, run `build_directional_coupler(L_c=L_c_cross)` at
#    $L_1 = (10, 4)$ or $L_2 = (14, 6)$, and verify that $|S_{31}|^2 > 0.98$ and
#    $|S_{41}|^2 < 0.01$.
# 2. **Quadrature port switching ($\Delta\phi = +\pi/2$ vs $-\pi/2$)**: Call
#    `run_phased_dual_drive` with `port_phases={"in_bot": 0.0, "in_top": +0.5 * np.pi}`
#    and confirm that the output routing flips completely from `out_top` to
#    `out_bot` with identical $> 30\;\text{dB}$ extinction ratio.
# 3. **Mask dilation tolerance**: Sweep `dilation` over
#    $\{-20, -10, 0, +10, +20\}\;\text{nm}$ at fixed $\theta_{\text{sw}} = 30^\circ$
#    using `coupler_mesh_overrides(..., n_cells_gap=6)` and determine the
#    sensitivity $\partial\lambda_{\text{cross}} / \partial d_{\text{mask}}$ in
#    $\text{nm/nm}$.
#
# ## 11. References
#
# 1. A. Yariv, "Coupled-mode theory for guided-wave optics," *IEEE J. Quantum
#    Electron.* **9**(9), 919–933 (1973),
#    DOI: [10.1109/JQE.1973.1077767](https://doi.org/10.1109/JQE.1973.1077767).
# 2. H. A. Haus, W. P. Huang, S. Kawakami, and N. A. Whitaker, "Coupled-mode
#    theory of optical waveguides," *J. Lightwave Technol.* **5**(1), 16–23
#    (1987), DOI: [10.1109/JLT.1987.1075416](https://doi.org/10.1109/JLT.1987.1075416).
# 3. L. Chrostowski and M. Hochberg, *Silicon Photonics Design: From Devices to
#    Systems*, Cambridge University Press (2015), Chapter 4 (Directional Couplers),
#    DOI: [10.1017/CBO9781316084168](https://doi.org/10.1017/CBO9781316084168).
# 4. M. Hammer and O. V. Ivanova, "Effective index approximations of photonic
#    crystal slabs: a 2-to-1-D assessment," *Opt. Quantum Electron.* **41**(4),
#    267–283 (2009),
#    DOI: [10.1007/s11082-009-9349-3](https://doi.org/10.1007/s11082-009-9349-3).
# 5. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing
#    of discretized numerical methods," *Phys. Rev. E* **77**, 036611 (2008),
#    DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="05. SOI Directional Coupler (2D EIM, Wakis FIT)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run resolved L1=(10,4)/L2=(14,6)/L3=(20,8) 2D study locally and render all 16 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run full L1=(10,4)/L2=(14,6)/L3=(20,8)/R=(40,16) 2D study and render all 16 figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study point I (0-indexed) at --full (or --quick) resolution",
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
        help="Read data/task_*.npz and generate all 16 figures in figures/",
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
        tasks = get_task_table(quick=args.quick)
        print(f"# Total tasks: {len(tasks)}")
        for t in tasks:
            print(f"{t['index']}\t{t['name']}\t{t['description']}")
        return

    if args.task is not None:
        tasks = get_task_table(quick=args.quick)
        if not (0 <= args.task < len(tasks)):
            raise SystemExit(f"--task must be in 0..{len(tasks) - 1}")
        run_single_task(tasks[args.task], args.out)
        return

    if args.plot:
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)
        return

    # Default when invoked with no flags (e.g. notebook execution): --quick
    quick = not args.full
    tasks = get_task_table(quick=quick)
    for t in tasks:
        run_single_task(t, DATA_DIR)
    generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
