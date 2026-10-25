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
# # Tutorial 6.1 · Sidewall-corrugated waveguide Bragg grating
#
# **Abstract.** Waveguide Bragg gratings (WBGs) are wavelength-selective
# distributed feedback reflectors used across silicon photonics as add-drop
# filters, dispersion compensators, laser cavity mirrors, and optical sensors.
# In high-index-contrast silicon-on-insulator (SOI) strip waveguides, even the
# smallest lithographically resolvable sidewall corrugations ($\Delta w \sim
# 10\text{--}50\;\text{nm}$) produce strong index perturbations and wide
# stopbands ($\Delta\lambda > 10\;\text{nm}$). **Wang *et al.* (2014)** [1]
# demonstrated that introducing a longitudinal phase offset $\Delta x_{\text{shift}}$
# between the periodic corrugations on the two sidewalls modulates the fundamental
# $\text{TE}_0$ coupling coefficient as $\kappa(\Delta x_{\text{shift}}) =
# \kappa_0 |\cos(\pi\,\Delta x_{\text{shift}} / \Lambda)|$; at a half-period
# shift ($\Delta x_{\text{shift}} = \Lambda / 2$), anti-symmetric teeth on
# opposite sidewalls interfere destructively and completely suppress fundamental
# Bragg reflection. In this tutorial we model the 100-period aligned and
# $\Lambda/2$-misaligned silicon Bragg gratings of Wang *et al.* (2014) in Wakis
# using the **2D effective-index method (EIM)** and **Kottke subpixel
# permittivity smoothing**, benchmark the simulated reflection/transmission
# spectra against analytical **Coupled-Mode Theory (CMT)**, sweep the grating
# period count $N$, corrugation depth $\Delta w$, and Gaussian apodization
# profile, and quantify the convergence advantage of Kottke smoothing over
# staircasing against an independent high-resolution reference simulation.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Reduce a 3D air-clad $220\;\text{nm}$ SOI corrugated strip waveguide on
#    $\text{SiO}_2$ to an equivalent **2D effective-index (EIM)** FIT domain
#    (`Ex`, `Ey`, `Hz`), and predict the Bragg resonance wavelength
#    $\lambda_B = 2 n_{\text{eff}}(w_{\text{avg}})\,\Lambda$ from the 2D port
#    dispersion relation.
# 2. Construct 100-period sidewall-corrugated waveguides (aligned, $\Lambda/2$-shifted,
#    and Gaussian-apodized) as single seamless counter-clockwise polygons in
#    `PolySlab` without internal mesh seams.
# 3. Derive the **Coupled-Mode Theory (CMT)** reflection spectrum
#    $R(\lambda) = |r(\lambda)|^2$, peak reflectivity $R_{\max} = \tanh^2(\kappa L)$,
#    and first-null bandwidth $\Delta\lambda_{\text{null}}$, and explain why a
#    $\Lambda/2$ sidewall phase offset cancels $\text{TE}_0 \leftrightarrow \text{TE}_0$
#    back-reflection.
# 4. Quantify why **Kottke (2008) subpixel smoothing** is essential for resolving
#    $50\;\text{nm}$ sidewall teeth and $324\;\text{nm}$ grating periods whose
#    pitch is incommensurate with the uniform Yee grid.
#
# ## Contents
#
# 1. [Physical background & Coupled-Mode Theory (CMT)](#1.-Physical-background-&-Coupled-Mode-Theory-(CMT))
# 2. [Effective-Index Method (EIM) reduction & Bragg phase matching](#2.-Effective-Index-Method-(EIM)-reduction-&-Bragg-phase-matching)
# 3. [Parametric geometry: aligned, misaligned & apodized gratings](#3.-Parametric-geometry:-aligned,-misaligned-&-apodized-gratings)
# 4. [Subpixel permittivity smoothing & local mesh refinement on 50 nm grating teeth](#4.-Subpixel-permittivity-smoothing-&-local-mesh-refinement-on-50-nm-grating-teeth)
# 5. [Study tasks & execution modes](#5.-Study-tasks-&-execution-modes)
# 6. [Generating the complete diagnostic figure suite](#6.-Generating-the-complete-diagnostic-figure-suite)
# 7. [Limits, failure modes & when 2D EIM breaks down](#7.-Limits,-failure-modes-&-when-2D-EIM-breaks-down)
# 8. [Exercises](#8.-Exercises)
# 9. [References](#9.-References)

# %% [markdown]
# ## 1. Physical background & Coupled-Mode Theory (CMT)
#
# A waveguide Bragg grating modulates the waveguide cross-section periodically
# along the propagation axis $x$ with period $\Lambda$. When a forward-propagating
# guided mode with propagation constant $\beta(\lambda) = 2\pi n_{\text{eff}}(\lambda)/\lambda$
# encounters the periodic permittivity perturbation $\Delta\varepsilon(x, y, z) =
# \sum_m \Delta\varepsilon_m(y, z)\,e^{-i 2\pi m x / \Lambda}$, first-order
# backward phase matching ($\beta - (-\beta) = 2\pi / \Lambda$) occurs at the
# **Bragg wavelength**:
#
# $$\lambda_B = 2\,n_{\text{eff}}(\lambda_B)\,\Lambda.$$
#
# Near $\lambda_B$, the slowly varying envelopes $A(x)$ (forward wave) and
# $B(x)$ (backward wave) obey the contra-directional **coupled-mode equations**
# (Yariv 1973 [2]; Erdogan 1997 [3]):
#
# $$\frac{\mathrm{d}A}{\mathrm{d}x} = -i\delta A - i\kappa B, \qquad
# \frac{\mathrm{d}B}{\mathrm{d}x} = +i\delta B + i\kappa^* A,$$
#
# where $\delta(\lambda) = \beta(\lambda) - \pi/\Lambda \approx
# -\frac{2\pi n_g}{\lambda_B^2}(\lambda - \lambda_B)$ is the phase detuning,
# $n_g = n_{\text{eff}} - \lambda\,\mathrm{d}n_{\text{eff}}/\mathrm{d}\lambda$ is
# the modal group index, and $\kappa$ is the modal overlap integral with the
# first Fourier harmonic of the sidewall perturbation. For a uniform grating of
# length $L = N\Lambda$ with boundary conditions $A(0) = 1$, $B(L) = 0$, the
# complex reflection and transmission coefficients are:
#
# $$r(\lambda) = \frac{-i\kappa\sinh(\gamma L)}{\gamma\cosh(\gamma L) + i\delta\sinh(\gamma L)}, \qquad
# t(\lambda) = \frac{\gamma}{\gamma\cosh(\gamma L) + i\delta\sinh(\gamma L)}, \qquad
# \gamma = \sqrt{|\kappa|^2 - \delta^2}.$$
#
# At the stopband center ($\delta = 0$, $\lambda = \lambda_B$), the peak power
# reflectivity and minimum transmittance are:
#
# $$R_{\max} = |r(\lambda_B)|^2 = \tanh^2(|\kappa| L), \qquad
# T_{\min} = 1 - R_{\max} = \operatorname{sech}^2(|\kappa| L),$$
#
# and the spectral spacing between the first reflection nulls ($\gamma L = i\pi$) is:
#
# $$\Delta\lambda_{\text{null}} = \frac{\lambda_B^2}{\pi n_g L}\sqrt{(|\kappa| L)^2 + \pi^2}.$$
#
# ### Destructive interference via sidewall phase offset (Wang *et al.* 2014)
#
# In a symmetric silicon strip waveguide of inner core width $w_{\text{wg}}$ with
# rectangular teeth of depth $w_{\text{corrug}} = \Delta w$ on both the $+y$ and
# $-y$ sidewalls, shifting the $-y$ teeth longitudinally by $\Delta x_{\text{shift}}$
# relative to the $+y$ teeth multiplies the $-y$ first-order Fourier harmonic by
# $e^{-i 2\pi \Delta x_{\text{shift}} / \Lambda}$. Because the fundamental
# $\text{TE}_0$ mode profile is laterally symmetric ($E_y(-y) = +E_y(+y)$), the
# net coupling coefficient is:
#
# $$\kappa(\Delta x_{\text{shift}}) = \frac{\kappa_0}{2}\left|1 + e^{-i 2\pi \Delta x_{\text{shift}} / \Lambda}\right|
# = \kappa_0 \left|\cos\!\left(\frac{\pi\,\Delta x_{\text{shift}}}{\Lambda}\right)\right|.$$
#
# * **Waveguide 1 (aligned, $\Delta x_{\text{shift}} = 0$)**: Both sidewalls scatter
#   in phase ($\kappa = \kappa_0$), yielding strong Bragg reflection ($R_{\max} > 99\%$).
# * **Waveguide 2 (half-period shifted, $\Delta x_{\text{shift}} = \Lambda / 2$)**:
#   The $+y$ and $-y$ teeth are $180^\circ$ out of phase ($\cos(\pi/2) = 0 \implies \kappa = 0$),
#   canceling fundamental $\text{TE}_0 \leftrightarrow \text{TE}_0$ Bragg reflection
#   so the incident wave transmits straight through the grating.
#
# ![Simulation setup: geometry, sources, monitors, and boundaries](figures/00a_setup_geometry.png)
# ![Simulation setup: graded non-uniform mesh with tooth zoom, wave packet, and excitation spectrum](figures/00b_setup_mesh_and_excitation.png)
#
# The 2D Effective-Index Method (EIM) simulation setup (`00a_setup_geometry.png` and `00b_setup_mesh_and_excitation.png`) models the 100-period sidewall-corrugated silicon Bragg grating ($\Lambda = 324\;\text{nm}$, $\Delta w = 50\;\text{nm}$, Bloch phase advance $\Delta\phi = \beta_B\Lambda = \pi$ at resonance) with $\pm z$ periodic boundaries and 10-cell CPML absorbing layers on $\pm x$ and $\pm y$:
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 34.7 µm × 2.19 µm × 37.8 nm (z ∈ [91.1 nm, 129 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 1264 × 85 × 1 (107440 cells) |
# | Discretization | Δx & resolution | Δ_min = 8.33 nm, 64.9 cells/λ (in n_max = 2.83), 6.0 cells across feature (50 nm) |
# | Boundary | -x face | CPML (10 cells, 366 nm) |
# | Boundary | +x face | CPML (10 cells, 366 nm) |
# | Boundary | -y face | CPML (10 cells, 376 nm) |
# | Boundary | +y face | CPML (10 cells, 376 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Air lateral/upper clad (n = 1.00, ε_r = 1.00; SiO₂ BOX n_sub = 1.44) (isotropic) |
# | Material | Region 1: Si EIM core (n_eff,slab = 2.8335) | ε_r = 8.03, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-19.2 µm, 19.2 µm] (d = 38.4 µm) |
# | Source | 1 · ModeSource | pos: x = -16.67 µm, y ∈ [-0.55, 0.55] µm, pol: TE₀ (in-plane E_y, H_z), pulse: f₀ = 192.5 THz, Δf = 7.4 THz |
# | Monitor | 2 · ModeMonitor | Port 'in' ModeMonitor (S₁₁), pos: x = -16.60 µm, y ∈ [-0.55, 0.55] µm, freqs: 185.1–199.9 THz (λ ∈ [1500, 1620] nm, 81 pts) |
# | Monitor | 3 · ModeMonitor | Port 'out' ModeMonitor (S₂₁), pos: x = 16.60 µm, y ∈ [-0.55, 0.55] µm, freqs: 185.1–199.9 THz (λ ∈ [1500, 1620] nm, 81 pts) |
# | Monitor | 4 · FluxBox | 2D Poynting FluxBox, pos: x ∈ [-17.00, 17.00] µm, y ∈ [-0.72, 0.72] µm, freqs: 185.1–199.9 THz (81 pts) |
# | Variation | Periodic unit cell & Bragg Bloch condition | Λ = 324 nm (D = 50%, l_tooth = 162 nm, w_core = 450 nm, Δw = 50 nm), Bloch phase Δφ = β_B Λ = π at λ_B; ±z faces: Periodic (2D EIM) |
# | Variation | Sidewall phase offset (task_9) | Aligned (Δx_shift = 0 → κ = κ₀, R_max > 99.7%) vs half-period shifted (Δx_shift = Λ/2 = 162 nm → κ = 0, R_max = 0.0003) |
# | Variation | Parametric sweeps (tasks 10–14) | Period count N ∈ {25, 50, 100}, tooth depth Δw ∈ {25, 50, 75} nm, Gaussian apodization (α = 4.0, −10.3 dB sidelobe reduction) |
# | Variation | Resolution & averaging ladder (tasks 0–8) | Kottke vs staircase across L1 (10 cpw, 4/tooth), L2 (14 cpw, 6/tooth), L3 (20 cpw, 8/tooth), and reference R (40 cpw, 16/tooth) |

# %%
import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light, epsilon_0

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _helpers import (  # noqa: E402
    FastPolySlab,
    bragg_grating,
    build_bragg_mesh_overrides,
    build_bragg_setup_objects,
    build_tuned_bragg_mesh_overrides,
    build_uniform_bragg_mesh_overrides,
    cmt_bragg_spectrum,
    cmt_coupling_from_neff_step,
    compare_bragg_mesh_strategies,
    extract_stopband_metrics,
    load_graded_vs_uniform_study_summary,
    measure_grating_mesh_metrics,
    plot_bragg_setup_schematic,
    render_bragg_pulse_gif,
)
from wakis import figstyle, schematic  # noqa: E402
from wakis.templates.photonics import (  # noqa: E402
    PICSimulation,
    Port,
    make_movie,
    n_to_eps,
    plot_eim_stack,
    plot_field_map,
    plot_layout,
    plot_neff_dispersion,
    plot_port_modes,
    plot_spectra_db,
    slab_neff,
)
from wakis.templates.simulation import Simulation  # noqa: E402

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# %% [markdown]
# ## 2. Effective-Index Method (EIM) reduction & Bragg phase matching
#
# Following **Wang *et al.* (2014)** [1], the 3D device consists of a
# $h_{\text{wg}} = 220\;\text{nm}$ silicon strip ($n_{\text{Si}} = 3.47$) on a
# $\text{SiO}_2$ buried-oxide substrate ($n_{\text{SiO}_2} = 1.44$, $z \in [-2.0, 0]\;\mu\text{m}$)
# with air upper and lateral cladding ($n_{\text{air}} = 1.0$, $z > 0$):
#
# | Parameter | Symbol | Value |
# |---|---|---|
# | Silicon core refractive index | $n_{\text{Si}}$ | $3.47$ ($\varepsilon_r = 12.0409$) |
# | $\text{SiO}_2$ substrate refractive index | $n_{\text{SiO}_2}$ | $1.44$ ($\varepsilon_r = 2.0736$) |
# | Upper / lateral cladding refractive index | $n_{\text{air}}$ | $1.00$ |
# | Silicon waveguide thickness | $h_{\text{wg}}$ | $220\;\text{nm}$ |
# | Uncorrugated inner core width | $w_{\text{wg}}$ | $450\;\text{nm}$ |
# | Sidewall tooth protrusion depth (each side) | $w_{\text{corrug}} = \Delta w$ | $50\;\text{nm}$ ($w_{\max} = 550\;\text{nm}$, $w_{\text{avg}} = 500\;\text{nm}$) |
# | Grating period & duty cycle | $\Lambda, D$ | $324\;\text{nm}$, $50\%$ ($l_{\text{corrug}} = 162\;\text{nm}$) |
# | Number of grating periods & total length | $N, L_{\text{wg}}$ | $100$, $32.40\;\mu\text{m}$ |
# | Dual-waveguide lateral positions | $y_1, y_2$ | $\pm \lambda_0 / 2 = \pm 0.766\;\mu\text{m}$ |
#
# In the **2D effective-index method (EIM)**, the asymmetric 3-layer vertical
# stack ($\text{SiO}_2 / 220\;\text{nm}\;\text{Si} / \text{air}$) at the
# reference wavelength $\lambda_0 = 1532\;\text{nm}$ supports a fundamental
# $\text{TE}_0$ slab mode with effective index:
#
# $$n_{\text{eff,slab}}(1532\;\text{nm}) = 2.8335.$$
#
# In the 2D in-plane FIT model (`Ex`, `Ey`, `Hz` with air lateral cladding
# $n_{\text{clad}} = 1.0$), an uncorrugated waveguide of the grating's arithmetic
# mean width $w_{\text{avg}} = 500\;\text{nm}$ has a 1D slab effective index
# $n_{\text{eff,2D}}(w_{\text{avg}}) \approx 2.418$ (group index $n_g = 3.2531$ across
# $1500\text{--}1620\;\text{nm}$, while the $450\;\text{nm}$ feed port mode has
# $n_{\text{eff,port}} = 2.3284$ at $1558\;\text{nm}$). Because the periodic $450/550\;\text{nm}$
# width modulation reduces the Bloch effective index slightly below the arithmetic-mean
# slab index, the 2D corrugated grating exhibits its converged 2D EIM Bragg resonance at:
#
# $$\lambda_{B,\text{2D}} = 1554.81\;\text{nm}\;(\text{L3}) \;\to\; 1552.27\;\text{nm}\;(\text{R})
# \quad \left(n_{\text{eff,grating}} = \frac{\lambda_{B,\text{2D}}}{2\Lambda} = 2.3994 \;\to\; 2.3955\right).$$
#
# Our simulation band $\lambda \in [1500, 1620]\;\text{nm}$
# covers both the 3D design wavelength ($1532\;\text{nm}$) and the complete 2D EIM
# stopband ($\Delta\lambda_{\text{FWHM}} = 30.63\text{--}31.23\;\text{nm}$, first nulls at
# $1535.6\text{--}1569.8\;\text{nm}$ at $\text{R}$ and $1537.0\text{--}1572.8\;\text{nm}$ at $\text{L3}$)
# together with its adjacent Fabry–Pérot sidelobes.

# %%
# Material and vertical stack parameters from Wang et al. (2014)
N_SI = 3.47  # eps_r = 12.0409
N_SIO2 = 1.44  # eps_r = 2.0736 (substrate z in [-2, 0] um)
N_AIR = 1.00  # upper/lateral cladding
H_SI = 220e-9  # 220 nm SOI waveguide height

# Grating geometric parameters
W_CORE = 0.45e-6  # 450 nm inner uncorrugated core width
DW_CORRUG = 0.05e-6  # 50 nm tooth depth on each side -> w_avg = 500 nm, w_max = 550 nm
PITCH = 0.324e-6  # 324 nm grating pitch Lambda
DUTY_CYCLE = 0.50  # 50% duty cycle (l_corrug = 162 nm)
N_PERIODS_DEFAULT = 100  # 100 periods -> L_wg = 32.40 um
Y_SEP_HALF = 0.766e-6  # +-lambda_0 / 2 dual-waveguide positions

# Spectral band covering both 1532 nm (3D design) and ~1555-1565 nm (2D EIM Bragg resonance)
WL_MIN = 1500e-9
WL_MAX = 1620e-9
WL_REF = 1532e-9
WL_FIELD_BRAGG = 1558e-9
N_FREQS = 81


# %% [markdown]
# ## 3. Parametric geometry: aligned, misaligned & apodized gratings
#
# Using `bragg_grating` and `FastPolySlab` from `_helpers.py`, each corrugated
# waveguide is built as a single counter-clockwise polygon without internal
# seams between the core and the 200 sidewall teeth. We provide builders both
# for a single isolated grating (used in the quantitative S-parameter tasks to
# avoid evanescent directional-coupler crosstalk) and for the dual side-by-side
# layout (Waveguide 1 aligned at $y_1 = +0.766\;\mu\text{m}$, Waveguide 2
# $\Lambda/2$-misaligned at $y_2 = -0.766\;\mu\text{m}$).


# %%
def build_bragg_grating(
    n_periods=N_PERIODS_DEFAULT,
    pitch=PITCH,
    w_core=W_CORE,
    dw=DW_CORRUG,
    shift=0.0,
    apodization="uniform",
    y0=0.0,
):
    """
    Build a single corrugated silicon Bragg grating waveguide as a seamless
    `FastPolySlab` together with its input and output `Port` specifications
    and 2D simulation domain `(xmin, xmax, ymin, ymax)` [m].
    """
    length = float(n_periods) * float(pitch)
    pad_x = 3.0e-6
    verts = bragg_grating(
        n_periods=n_periods,
        pitch=pitch,
        w_core=w_core,
        dw=dw,
        duty_cycle=DUTY_CYCLE,
        shift=shift,
        apodization=apodization,
        y0=y0,
        pad_x=pad_x,
    )
    slab = FastPolySlab(
        verts,
        z_min=0.0,
        z_max=H_SI,
        material=n_to_eps(N_SI),
    )
    x_port = 0.5 * length + 0.40e-6
    domain_x = 0.5 * length + 0.80e-6
    domain_y = 0.72e-6
    ports = [
        Port(
            "in",
            center=(-x_port, y0, 0.5 * H_SI),
            size=(0.0, 1.10e-6, 0.8e-6),
            direction=+1,
        ),
        Port(
            "out",
            center=(+x_port, y0, 0.5 * H_SI),
            size=(0.0, 1.10e-6, 0.8e-6),
            direction=+1,
        ),
    ]
    domain = (-domain_x, +domain_x, y0 - domain_y, y0 + domain_y)
    return [slab], ports, domain


def build_dual_bragg_gratings(n_periods=N_PERIODS_DEFAULT):
    """
    Build the side-by-side dual Bragg grating layout from Wang et al. (2014):
    Waveguide 1 (aligned, shift=0) at `y1 = +0.766 um` and Waveguide 2
    (misaligned, shift=Lambda/2) at `y2 = -0.766 um`.
    """
    length = float(n_periods) * PITCH
    verts_aligned = bragg_grating(
        n_periods=n_periods,
        pitch=PITCH,
        w_core=W_CORE,
        dw=DW_CORRUG,
        shift=0.0,
        y0=+Y_SEP_HALF,
        pad_x=2.0e-6,
    )
    verts_misaligned = bragg_grating(
        n_periods=n_periods,
        pitch=PITCH,
        w_core=W_CORE,
        dw=DW_CORRUG,
        shift=0.5 * PITCH,
        y0=-Y_SEP_HALF,
        pad_x=2.0e-6,
    )
    slab1 = FastPolySlab(verts_aligned, z_min=0.0, z_max=H_SI, material=n_to_eps(N_SI))
    slab2 = FastPolySlab(
        verts_misaligned, z_min=0.0, z_max=H_SI, material=n_to_eps(N_SI)
    )
    x_port = 0.5 * length + PITCH
    ports = [
        Port(
            "in_aligned",
            center=(-x_port, +Y_SEP_HALF, 0.5 * H_SI),
            size=(0.0, 0.90e-6, 0.44e-6),
            direction=+1,
        ),
        Port(
            "out_aligned",
            center=(+x_port, +Y_SEP_HALF, 0.5 * H_SI),
            size=(0.0, 1.10e-6, 0.80e-6),
            direction=+1,
        ),
        Port(
            "in_misaligned",
            center=(-x_port, -Y_SEP_HALF, 0.5 * H_SI),
            size=(0.0, 0.90e-6, 0.44e-6),
            direction=+1,
        ),
        Port(
            "out_misaligned",
            center=(+x_port, -Y_SEP_HALF, 0.5 * H_SI),
            size=(0.0, 1.10e-6, 0.80e-6),
            direction=+1,
        ),
    ]
    domain = (-0.5 * length - 1.15e-6, +0.5 * length + 1.15e-6, -1.807e-6, +1.807e-6)
    return [slab1, slab2], ports, domain


def build_task_mesh_metrics(task_cfg):
    """
    Construct the 2D EIM mesh node arrays `(x, y, z)` for `task_cfg` using
    `Simulation._axis` without instantiating `SolverFIT3D` and return the
    measured mesh resolution dictionary from `measure_grating_mesh_metrics`.
    """
    structures, _, domain = build_bragg_grating(
        n_periods=task_cfg["n_periods"],
        pitch=PITCH,
        w_core=W_CORE,
        dw=task_cfg["dw"],
        shift=task_cfg["shift"],
        apodization=task_cfg["apodization"],
    )
    f_min = c_light / WL_MAX
    f_max = c_light / WL_MIN
    neff_slab = slab_neff(N_SI, N_AIR, H_SI, WL_REF, n_sub=N_SIO2, polarization="TE")
    dx_bg = (c_light / (f_max * neff_slab)) / float(task_cfg["cpw"])
    overrides = build_bragg_mesh_overrides(
        n_periods=task_cfg["n_periods"],
        pitch=PITCH,
        w_core=W_CORE,
        dw=task_cfg["dw"],
        duty_cycle=DUTY_CYCLE,
        shift=task_cfg["shift"],
        n_feature=task_cfg["n_feature"],
        dx_background=dx_bg,
    )
    xmin, xmax, ymin, ymax = domain[:4]
    zmin_2d, zmax_2d = 0.5 * H_SI - 0.5 * dx_bg, 0.5 * H_SI + 0.5 * dx_bg
    domain_3d = (xmin, xmax, ymin, ymax, zmin_2d, zmax_2d)

    s_copy = FastPolySlab(
        structures[0].vertices,
        z_min=zmin_2d,
        z_max=zmax_2d,
        material=n_to_eps(neff_slab),
    )
    sim = Simulation(
        f_min,
        f_max,
        structures=[s_copy],
        boundaries=(["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"]),
        domain=domain_3d,
        cells_per_wavelength=task_cfg["cpw"],
        n_pml=task_cfg["n_pml"],
        n_freqs=N_FREQS,
        cfln=0.85,
        mesh_overrides=overrides,
        averaging=task_cfg["averaging"],
    )
    x_nodes, y_nodes, z_nodes = [sim._axis(a) for a in range(3)]
    return measure_grating_mesh_metrics(
        x_nodes,
        y_nodes,
        z_nodes,
        dw=task_cfg["dw"],
        n_periods=task_cfg["n_periods"],
        pitch=PITCH,
        w_core=W_CORE,
        duty_cycle=DUTY_CYCLE,
        shift=task_cfg["shift"],
        n_pml=task_cfg["n_pml"],
    )


# %% [markdown]
# ## 4. Subpixel permittivity smoothing & local mesh refinement on 50 nm grating teeth
#
# ### Resolution: cells per wavelength and cells across the smallest feature
#
# Accurate time-domain modeling of sidewall-corrugated Bragg gratings requires
# satisfying two independent spatial discretization rules simultaneously:
#
# 1. **Waveguide dispersion rule ($\ge 10\;\text{cpw}$ in the highest-index material)**:
#    The background grid inside the silicon slab ($n_{\text{eff,slab}} = 2.8335$)
#    must provide at least $10\;\text{cells}$ per guided wavelength at
#    $\lambda_{\min} = 1500\;\text{nm}$ ($\Delta x_{\text{bg}} \le \lambda_{\min} /
#    (n_{\text{eff,slab}}\,\text{cpw}) \approx 52.9\;\text{nm}$ at $10\;\text{cpw}$)
#    in every execution mode, including `--quick` and parameter sweeps.
# 2. **Geometric feature rule ($N \ge 4\;\text{cells}$ across the smallest feature)**:
#    The smallest feature of this device is the sidewall tooth depth
#    $\Delta w = 50\;\text{nm}$ in $y$ ($25\;\text{nm}$ and $75\;\text{nm}$ in the
#    corrugation-depth sweep), followed by the tooth length $l_{\text{corrug}} =
#    \Lambda / 2 = 162\;\text{nm}$ in $x$. At $10\;\text{cpw}$ without local
#    refinement, $\Delta y \approx 51.4\;\text{nm}$ places only $\sim 0.97\;\text{cells}$
#    across a $50\;\text{nm}$ tooth—making the grid coarser than the feature itself.
#
# To resolve both the transverse tooth depth and longitudinal tooth edges without
# paying the cost of a uniform fine mesh across the entire air cladding and feed
# waveguides, we pass three **per-axis `mesh_overrides` boxes** to `PICSimulation`:
#
# * **Two transverse $y$-strips over the upper and lower tooth bands**: spanning
#   from the inner core edge ($|y| = w_{\text{wg}}/2 = 225\;\text{nm}$) to the
#   tooth tip ($|y| = w_{\text{wg}}/2 + \Delta w = 275\;\text{nm}$) plus one tooth
#   depth of margin ($\pm 0.5\Delta w$ on each side), with per-axis cell cap
#   `(1.0, dw / N, 1.0)` so $h_y \le \Delta w / N$.
# * **One longitudinal $x$-box over the grating extent**: spanning $x \in
#   [-L_{\text{wg}}/2, +L_{\text{wg}}/2 + \Delta x_{\text{shift}}]$ with per-axis
#   cell cap `(min(dx_bg, 162 nm / N), 1.0, 1.0)` so every $162\;\text{nm}$ tooth
#   and trench is spanned by at least $N$ cells along $x$.
#
# ### Kottke subpixel smoothing vs staircasing
#
# Even when a $50\;\text{nm}$ tooth is resolved by $N \ge 4$ cells, **staircase**
# discretization rounds every cut boundary cell ($w \ge 0.5 \to 1$, $w < 0.5 \to 0$),
# altering the effective core width $w_{\text{avg}}$ and shifting $\lambda_B =
# 2 n_{\text{eff}} \Lambda$. **Kottke, Farjadpour & Johnson (2008)** [4]
# anisotropic subpixel smoothing uses harmonic averaging for the normal electric
# field component and arithmetic averaging for the tangential component on each
# cut cell, achieving rapid convergence of both $\lambda_B$ and the stopband
# $-3\;\text{dB}$ width $\Delta\lambda_{\text{FWHM}}$.

# %% [markdown]
# ## 5. Study tasks & execution modes
#
# Each resolution level is specified by a pair $(\text{cpw}, N)$ combining the
# background slab resolution `cells_per_wavelength = cpw` and the minimum number
# of cells $N$ across the smallest geometric feature:
#
# * **Under-resolved warning pair** (`underresolved_kottke`, `underresolved_staircase`):
#   $(10\;\text{cpw}, \text{no overrides})$, placing only $0.97\;\text{cells}$ across
#   the $50\;\text{nm}$ tooth to demonstrate what fails when the feature rule is violated.
# * **Resolution ladder**: $\text{L1} = (10, 4)$, $\text{L2} = (14, 6)$,
#   $\text{L3} = (20, 8)$, and reference $\text{R} = (40, 16)$.
# * In `--quick` mode (13 tasks), Kottke and staircase run at $\text{L1}$ and
#   $\text{L2}$, the independent Kottke reference runs at $\text{L3} = (20, 8)$,
#   and all device sweeps ($\Lambda/2$ phase shift, $N_{\text{periods}} \in \{25, 50\}$,
#   $\Delta w \in \{25, 75\}\;\text{nm}$, and Gaussian apodization) run at
#   $\text{L2} = (14, 6)$ (with $h_y = \Delta w / 6$ scaled from each run's own
#   tooth depth $\Delta w$).
# * In `--full` mode (15 tasks, `n_pml = 10`, `energy_decay = 1e-4`, `max_time = 1.80 ps`),
#   Kottke and staircase run at $\text{L1}$, $\text{L2}$, and $\text{L3}$, the Kottke
#   reference runs at $\text{R} = (40, 16)$ ($3.52\;\text{h}$ wall time on Cloud Batch),
#   and all device sweeps run at $\text{L3} = (20, 8)$.
#
# ### Measured `--full` resolution & results table (15 tasks, reference $\text{R} = (40, 16)$)
#
# | Task | Name | cpw | $N$ | Feature $[\text{nm}]$ | $\Delta x_{\text{bg}}\;[\text{nm}]$ | $\Delta x_{\min}\;[\text{nm}]$ | Cells across feature | $n_{\text{cells}}$ | Steps | $\lambda_B\;[\text{nm}]$ | $\Delta\lambda_{\text{BW}}\;[\text{nm}]$ | $R_{\max}$ | $T_{\min}$ (Ext. $[\text{dB}]$) | Time $[\text{s}]$ | Peak RSS $[\text{MB}]$ |
# |---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | `0` | `underresolved_kottke` *(warning)* | 10 | 0 | 50.0 | 52.88 | 51.43 | 0.97 ($3.06$ in $x$) | 32,536 | 20,985 | 1561.90 | 26.75 | 1.0000 | 0.0069 (21.59) | 82.5 | 291.5 |
# | `1` | `underresolved_staircase` *(warning)* | 10 | 0 | 50.0 | 52.88 | 51.43 | 0.97 ($3.06$ in $x$) | 32,536 | 20,985 | 1570.45 | 30.32 | 1.0000 | 0.0027 (25.72) | 68.5 | 279.6 |
# | `2` | `kottke_L1` | 10 | 4 | 50.0 | 50.27 | 12.50 | 4.00 ($4.00$ in $x$) | 58,004 | 54,489 | 1563.97 | 33.00 | 1.0000 | 0.0006 (31.93) | 339.0 | 332.1 |
# | `3` | `staircase_L1` | 10 | 4 | 50.0 | 50.27 | 12.50 | 4.00 ($4.00$ in $x$) | 58,004 | 54,489 | 1574.68 | 35.83 | 1.0000 | 0.0003 (35.01) | 320.4 | 317.9 |
# | `4` | `kottke_L2` | 14 | 6 | 50.0 | 36.58 | 8.33 | 6.00 ($6.00$ in $x$) | 108,790 | 81,478 | 1557.40 | 31.71 | 1.0000 | 0.0013 (28.71) | 951.2 | 418.8 |
# | `5` | `staircase_L2` | 14 | 6 | 50.0 | 36.58 | 8.33 | 6.00 ($6.00$ in $x$) | 108,790 | 81,478 | 1564.91 | 33.94 | 1.0000 | 0.0003 (35.61) | 1045.4 | 407.6 |
# | `6` | `kottke_L3` (+ fields & movie) | 20 | 8 | 50.0 | 25.88 | 6.25 | 8.00 ($8.00$ in $x$) | 181,764 | 108,978 | 1554.81 | 31.23 | 1.0000 | 0.0025 (26.10) | 3150.0 | 542.2 |
# | `7` | `staircase_L3` | 20 | 8 | 50.0 | 25.88 | 6.25 | 8.00 ($8.00$ in $x$) | 181,764 | 108,978 | 1560.63 | 33.07 | 1.0000 | 0.0006 (32.47) | 2167.8 | 537.6 |
# | `8` | `kottke_reference` ($\text{R}$) | 40 | 16 | 50.0 | 13.13 | 3.12 | 16.00 ($16.00$ in $x$) | 625,141 | 217,955 | 1552.27 | 30.63 | 1.0000 | 0.0031 (25.07) | 12658.4 | 1326.0 |
# | `9` | `misaligned_half_period` (+ fields) | 20 | 8 | 50.0 | 25.88 | 6.25 | 8.00 ($8.00$ in $x$) | 181,980 | 49,751 | 1511.19 | — | 0.0003 | 0.9931 (0.03) | 1496.7 | 539.3 |
# | `10` | `periods_N25` | 20 | 8 | 50.0 | 25.88 | 6.25 | 8.00 ($8.00$ in $x$) | 52,164 | 24,451 | 1555.42 | 50.32 | 0.5087 | 0.4783 (3.20) | 134.6 | 304.9 |
# | `11` | `periods_N50` | 20 | 8 | 50.0 | 25.88 | 6.25 | 8.00 ($8.00$ in $x$) | 95,364 | 54,489 | 1554.94 | 37.30 | 0.8931 | 0.0991 (10.04) | 576.9 | 384.4 |
# | `12` | `depth_dw25nm` | 20 | 8 | 25.0 | 25.88 | 3.12 | 8.00 ($8.00$ in $x$) | 198,594 | 206,953 | 1542.45 | 23.28 | 0.9938 | 0.0254 (15.96) | 4683.6 | 570.8 |
# | `13` | `depth_dw75nm` | 20 | 8 | 75.0 | 25.88 | 9.37 | 8.00 ($8.00$ in $x$) | 171,666 | 78,383 | 1563.23 | 33.07 | 1.0000 | 0.0006 (32.17) | 1400.7 | 522.9 |
# | `14` | `apodized_gaussian` | 20 | 8 | 50.0 | 25.88 | 6.25 | 8.00 ($8.00$ in $x$) | 181,764 | 66,201 | 1559.00 | 31.43 | 0.8945 | 0.1157 (9.37) | 1239.6 | 538.2 |
#
# ### Measured `--quick` resolution & results table (13 tasks, `energy_decay = 2e-2`, `max_time = 0.95 ps`)
#
# | Task | Name | cpw | $N$ | Feature $[\text{nm}]$ | $\Delta x_{\text{bg}}\;[\text{nm}]$ | $\Delta x_{\min}\;[\text{nm}]$ | Cells across feature | $n_{\text{cells}}$ | Steps | $\lambda_B\;[\text{nm}]$ | $\Delta\lambda_{\text{BW}}\;[\text{nm}]$ | Time $[\text{s}]$ |
# |---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | `0` | `underresolved_kottke` *(warning)* | 10 | 0 | 50.0 | 52.88 | 51.43 | 0.97 | 26,896 | 11,076 | 1561.96 | 29.38 | 39.5 |
# | `1` | `underresolved_staircase` *(warning)* | 10 | 0 | 50.0 | 52.88 | 51.43 | 0.97 | 26,896 | 11,076 | 1570.67 | 34.64 | 39.4 |
# | `2` | `kottke_L1` | 10 | 4 | 50.0 | 50.27 | 12.50 | 4.00 | 50,700 | 28,758 | 1564.11 | 38.50 | 183.7 |
# | `3` | `staircase_L1` | 10 | 4 | 50.0 | 50.27 | 12.50 | 4.00 | 50,700 | 28,758 | 1574.78 | 41.15 | 185.0 |
# | `4` | `kottke_L2` | 14 | 6 | 50.0 | 36.58 | 8.33 | 6.00 | 98,046 | 43,002 | 1557.50 | 36.69 | 798.1 |
# | `5` | `staircase_L2` | 14 | 6 | 50.0 | 36.58 | 8.33 | 6.00 | 98,046 | 43,002 | 1565.04 | 39.22 | 521.7 |
# | `6` | `kottke_reference` ($\text{L3}$) | 20 | 8 | 50.0 | 25.88 | 6.25 | 8.00 | 167,500 | 57,516 | 1554.91 | 35.96 | 1123.1 |
# | `7` | `misaligned_half_period` | 14 | 6 | 50.0 | 37.81 | 8.33 | 6.00 | 98,124 | 23,151 | 1565.22 | — | 387.9 |
# | `8` | `periods_N25` | 14 | 6 | 50.0 | 36.58 | 8.33 | 6.00 | 27,846 | 12,701 | 1558.94 | 46.97 | 45.3 |
# | `9` | `periods_N50` | 14 | 6 | 50.0 | 36.58 | 8.33 | 6.00 | 51,246 | 21,501 | 1557.62 | 38.19 | 113.5 |
# | `10` | `depth_dw25nm` | 14 | 6 | 25.0 | 36.58 | 4.17 | 6.00 | 109,359 | 77,501 | 1545.02 | 26.49 | 1019.6 |
# | `11` | `depth_dw75nm` | 14 | 6 | 75.0 | 36.58 | 12.50 | 6.00 | 93,018 | 30,839 | 1566.01 | 39.09 | 343.7 |
# | `12` | `apodized_gaussian` | 14 | 6 | 50.0 | 36.58 | 8.33 | 6.00 | 98,046 | 28,001 | 1561.85 | 37.81 | 323.7 |
#
# ### Convergence & resolution verification (`--full` dataset)
#
# * **Resolution rule audit (`data/task_*.json`)**: Every non-demo task (`task_2` through `task_14`)
#   satisfies $\text{cpw} \ge 10$ in the highest-index material ($10, 14, 20, 40\;\text{cpw}$) and
#   $\text{cells\_across\_feature} \ge 4.0$ ($4.00$ at $\text{L1}$, $6.00$ at $\text{L2}$, $8.00$ at
#   $\text{L3}$, and $16.00$ at $\text{R}$, including the $\Delta w = 25\;\text{nm}$ and $75\;\text{nm}$
#   depth-sweep tasks where $\Delta y_{\min}$ scales to $3.12\;\text{nm}$ and $9.37\;\text{nm}$). Only the
#   two intentional warning tasks (`task_0`, `task_1`) omit `mesh_overrides` ($0.97\;\text{cells}$ across
#   the $50\;\text{nm}$ tooth). No resolution violations exist.
# * **Bragg wavelength $\lambda_B$ convergence**: Against the independent $\text{R} = (40, 16)$ reference
#   ($\lambda_{B,\text{ref}} = 1552.27\;\text{nm}$), Kottke error $|\lambda_B - \lambda_{B,\text{ref}}|$
#   decreases monotonically across $\text{L1} \to \text{L2} \to \text{L3}$ from $11.71\;\text{nm}$ ($0.754\%$)
#   to $5.14\;\text{nm}$ ($0.331\%$) to $2.55\;\text{nm}$ ($0.164\%$), beating staircase ($22.42 \to 12.64 \to 8.36\;\text{nm}$)
#   by $1.91\times$, $2.46\times$, and $3.28\times$. The relative change in $\lambda_B$ between the last two
#   ladder levels ($\text{L2} \to \text{L3}$) is $0.166\%$ ($-2.59\;\text{nm}$), and between $\text{L3}$ and $\text{R}$
#   is $0.164\%$ ($+2.55\;\text{nm}$), well below $1\%$.
# * **Stopband bandwidth $\Delta\lambda_{\text{FWHM}}$ & spectrum convergence**: Against $\text{R} = (40, 16)$
#   ($\Delta\lambda_{\text{BW,ref}} = 30.63\;\text{nm}$, $\Delta\lambda_{\text{null,ref}} = 34.22\;\text{nm}$),
#   Kottke FWHM bandwidth error decreases monotonically from $2.37\;\text{nm}$ ($7.75\%$ at $\text{L1}$) to
#   $1.08\;\text{nm}$ ($3.53\%$ at $\text{L2}$) to $0.61\;\text{nm}$ ($1.98\%$ at $\text{L3}$), beating
#   staircase ($5.20 \to 3.31 \to 2.44\;\text{nm}$) by $2.19\times$, $3.06\times$, and $4.02\times$. The
#   relative change in $\Delta\lambda_{\text{FWHM}}$ is $1.50\%$ between $\text{L2}$ and $\text{L3}$ ($-0.48\;\text{nm}$)
#   and $1.98\%$ between $\text{L3}$ and $\text{R}$ ($+0.61\;\text{nm}$). Full-band transmission RMS error
#   $\|T - T_{\text{ref}}\|_2$ drops monotonically from $0.3549 \to 0.2325 \to 0.1552$ for Kottke vs
#   $0.5310 \to 0.3753 \to 0.2771$ for staircase ($1.50\text{--}1.79\times$ lower).
#
# ### Comparison of mesh strategies: Uniform vs Sidewall-Strip Graded vs Tuned Graded
#
# Benchmarking the 100-period silicon Bragg grating ($L = 32.40\;\mu\text{m}$, domain
# $34.0\;\mu\text{m} \times 1.44\;\mu\text{m}$, $t_{\text{end}} = 1.80\;\text{ps}$) across three
# Cartesian meshing strategies against the multi-arm Richardson continuum limit
# ($\lambda_{B,\text{ref}} = 1551.31 \pm 0.11\;\text{nm}$, $\Delta\lambda_{\text{BW,ref}} =
# 30.41 \pm 0.03\;\text{nm}$; see `studies/subgrid-vs-graded-bragg/`) isolates the
# cost and accuracy trade-offs of local non-uniform grading:
#
# | Mesh strategy | $N_{\text{tooth}}$ | $\Delta y_{\text{tooth}}$ [nm] | $\Delta x_{\text{tooth}}$ [nm] | $\Delta y_{\text{core}}$ [nm] | Grid ($N_x \times N_y$) | $n_{\text{cells}}$ | Steps | Wall time [s] | $\lambda_B$ [nm] ($|\Delta\lambda_B|$ [nm]) | $\Delta\lambda_{\text{BW}}$ [nm] (err [%]) |
# |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | Unrefined uniform (`Task 0`, warning) | $0.97$ | $51.43$ | $52.88$ | $51.43$ | $664 \times 49$ | $32{,}536$ | $20{,}985$ | $82.5$ | $1561.90$ ($10.59$) | $26.75$ ($12.0\%$) |
# | **Uniform fine (`uniform_N4`)** | $4.06$ | $12.31$ | $12.37$ | $12.31$ | $2769 \times 137$ | $379{,}353$ | $68{,}757$ | $1095.1$ | $1551.42$ ($0.11$) | $30.64$ ($0.77\%$) |
# | **Sidewall-strip graded (`Task 2`, `L1`)** | $4.00$ | $12.50$ | $40.50$ | $52.72$ | $853 \times 68$ | $58{,}004$ | $54{,}489$ | $339.0$ ($287.5$) | $1563.97$ ($12.66$) | $33.00$ ($8.52\%$) |
# | **Sidewall-strip graded (`Task 4`, `L2`)** | $6.00$ | $8.33$ | $27.00$ | $35.91$ | $1265 \times 86$ | $108{,}790$ | $81{,}478$ | $951.2$ ($917.0$) | $1557.40$ ($6.10$) | $31.71$ ($4.27\%$) |
# | **Sidewall-strip graded (`Task 6`, `L3`)** | $8.00$ | $6.25$ | $20.25$ | $25.79$ | $1683 \times 108$ | $181{,}764$ | $108{,}978$ | $3150.0$ ($2062.2$) | $1554.81$ ($3.50$) | $31.23$ ($2.71\%$) |
# | **Sidewall-strip graded (`Task 8`, `R`)** | $16.00$ | $3.125$ | $10.125$ | $13.09$ | $3343 \times 187$ | $625{,}141$ | $217{,}955$ | $12658.4$ ($14997.8$) | $1552.27$ ($0.96$) | $30.63$ ($0.71\%$) |
# | **Tuned graded (`graded_tuned_L1`)** | $4.00$ | $12.50$ | $12.50$ | $24.95$ | $2647 \times 74$ | $195{,}878$ | $72{,}820$ | $1516.5$ | $1553.04$ ($1.73$) | $30.29$ ($0.40\%$) |
# | **Tuned graded (`graded_tuned_L2`, equal cells)** | $6.00$ | $8.33$ | $8.33$ | $16.64$ | $3955 \times 96$ | $379{,}680$ | $109{,}039$ | $4617.5$ | $1552.12$ ($0.82$) | $30.31$ ($0.31\%$) |
# | **Tuned graded (`graded_tuned_L3`)** | $8.00$ | $6.25$ | $6.25$ | $12.49$ | $5269 \times 122$ | $642{,}818$ | $145{,}640$ | $8730.8$ | $1551.77$ ($0.46$) | $30.34$ ($0.22\%$) |
# | **Richardson continuum limit ($N_{\text{tooth}} \to \infty$)** | $\infty$ | $0$ | $0$ | $0$ | — | — | — | — | $1551.31$ (tuned) / $1551.42$ (strip) | $30.38$ (tuned) / $30.42$ (strip) |
#
# * **Equal tooth resolution ($N_{\text{tooth}} = 4$, $\Delta y_{\text{tooth}} \approx 12.5\;\text{nm}$)**:
#   `uniform_N4` ($379{,}353$ cells, $1095.1\;\text{s}$) places $12.3\;\text{nm}$ cells across the entire
#   air cladding, whereas `graded_tuned_L1` (`build_tuned_bragg_mesh_overrides`, which refines
#   the tooth strips to $\Delta x = \Delta y = 12.5\;\text{nm}$ and the $400\;\text{nm}$ core to
#   $\Delta y_{\text{core}} = 25.0\;\text{nm}$ while grading at $g = 1.35$ into the cladding)
#   cuts the cell count by **$1.94\times$** ($195{,}878$ cells) and achieves a $1.9\times$ smaller
#   stopband bandwidth error ($0.40\%$ vs. $0.77\%$). The baseline sidewall-strip graded mesh
#   (`Task 2` `kottke_L1`, `build_bragg_mesh_overrides`) cuts the cell count by **$6.54\times$**
#   ($58{,}004$ cells, $287.5\text{--}339.0\;\text{s}$), trading a coarser core and longitudinal
#   tooth step ($\Delta x = 40.5\;\text{nm}$, $\Delta y_{\text{core}} = 52.7\;\text{nm}$) for a
#   $3.8\times$ faster turnaround.
# * **Equal cell count ($\approx 3.80 \times 10^5$ cells)**:
#   At the same total cell budget as `uniform_N4` ($379{,}353$ cells, $N_{\text{tooth}} = 4.06$),
#   `graded_tuned_L2` ($379{,}680$ cells) concentrates $1.5\times$ finer isotropic cells inside
#   every $50\;\text{nm}$ sidewall tooth ($N_{\text{tooth}} = 6.0$, $\Delta x = \Delta y = 8.33\;\text{nm}$),
#   halving the stopband bandwidth error from $0.77\%$ ($30.64\;\text{nm}$) to **$0.31\%$**
#   ($30.31\;\text{nm}$) with $\lambda_B = 1552.12\;\text{nm}$.
# * **Equal accuracy ($\Delta\lambda_{\text{BW}}$ within $0.7\%$, $|\Delta\lambda_B| < 1.0\;\text{nm}$)**:
#   Reaching $N_{\text{tooth}} = 8$ ($\Delta_{\text{tooth}} = 6.25\;\text{nm}$) or $N_{\text{tooth}} = 16$
#   ($\Delta_{\text{tooth}} = 3.125\;\text{nm}$) on a uniform grid would require $1.52\times 10^6$ or
#   $8.9\times 10^6$ cells; the graded meshes reach $N_{\text{tooth}} = 8$ (`graded_tuned_L3`,
#   $|\Delta\lambda_B| = 0.46\;\text{nm}$, $0.22\%$ bandwidth error) in $642{,}818$ cells ($2.36\times$
#   fewer) and $N_{\text{tooth}} = 16$ (`Task 8` `kottke_reference`, $|\Delta\lambda_B| = 0.96\;\text{nm}$,
#   $0.71\%$ bandwidth error) in $625{,}141$ cells (**$14.2\times$ fewer cells**), with both graded arms
#   extrapolating to the same continuum Bragg wavelength within $0.11\;\text{nm}$ ($1551.31\;\text{nm}$
#   vs. $1551.42\;\text{nm}$).
#
# ### Comparison: Compact (`--quick`) vs Production (`--full`) vs Analytical CMT
#
# Extending the time-domain integration from `--quick` (`energy_decay = 2e-2`, $t_{\max} = 0.95\;\text{ps}$,
# sweeps at $\text{L2}$) to `--full` (`energy_decay = 1e-4`, $t_{\max} = 1.80\;\text{ps}$, sweeps at $\text{L3}$,
# reference at $\text{R} = (40, 16)$) allows the slow-group-velocity band-edge Fabry–Pérot modes of the
# 100-period grating ($L = 32.40\;\mu\text{m}$) and the transmitted pulse tail to ring down completely.
# In `--quick`, truncating at $0.95\;\text{ps}$ artificially depresses $T_{\min}$ at $N = 100$ ($0.00012\text{--}0.00026$,
# inflating $\kappa_{T_{\min}}$ to $1490\text{--}1606\;\text{cm}^{-1}$ so it exceeds $\kappa_{\text{BW}} \approx 1163\text{--}1243\;\text{cm}^{-1}$),
# smears the first reflection nulls into the stopband edges (inflating $\Delta\lambda_{\text{FWHM}}$ by $\sim 5\;\text{nm}$
# and damping the first uniform sidelobes to $-7.05\;\text{dB}$), and leaves $9\%$ of the pulse inside the
# $\Lambda/2$-shifted waveguide ($T_{\min} = 0.9095$). In `--full`, all three effects resolve cleanly:
# $\kappa_{T_{\min}}$ ($1141.3\;\text{cm}^{-1}$ at $\text{L3}$, $1104.7\;\text{cm}^{-1}$ at $\text{R}$) and
# $\kappa_{\text{BW}}$ ($1162.8\;\text{cm}^{-1}$ at $\text{L3}$, $1080.2\;\text{cm}^{-1}$ at $\text{R}$) agree to
# within $1.9\text{--}2.2\%$, $\kappa_{T_{\min}}$ is invariant across $N \in \{25, 50, 100\}$ to within $1.4\%$
# ($1126.5, 1125.6, 1141.3\;\text{cm}^{-1}$), and the $\Lambda/2$-shifted grating reaches $T_{\min} = 0.9931$
# ($0.03\;\text{dB}$ maximum insertion loss, $R_{\max} = 0.0003$).
#
# | Metric ($N = 100$, $\Delta w = 50\;\text{nm}$ unless noted) | Compact (`--quick`: primary L2, ref L3) | Production (`--full`: primary L3, ref R) | Analytical CMT (2D EIM, $n_g = 3.2531$) [1–3, 5] |
# |---|---:|---:|---:|
# | Aligned Bragg wavelength $\lambda_B$ (primary / ref) | $1557.50\;\text{nm}$ / $1554.91\;\text{nm}$ | $1554.81\;\text{nm}$ / $1552.27\;\text{nm}$ | $2 n_{\text{eff}}(w_{\text{avg}})\Lambda = 1568.66\;\text{nm}$ (1D slab avg) |
# | Stopband $-3\;\text{dB}$ width $\Delta\lambda_{\text{FWHM}}$ (primary / ref) | $36.69\;\text{nm}$ / $35.96\;\text{nm}$ | $31.23\;\text{nm}$ / $30.63\;\text{nm}$ | $32.19\;\text{nm}$ (at $\kappa_{\text{L3}}$) / $31.39\;\text{nm}$ (at $\kappa_{\text{R}}$) |
# | Stopband first-null width $\Delta\lambda_{\text{null}}$ (primary / ref) | $37.41\;\text{nm}$ / $35.81\;\text{nm}$ | $35.81\;\text{nm}$ / $34.22\;\text{nm}$ | $35.31\;\text{nm}$ (at $\kappa_{\text{L3}}$) / $34.66\;\text{nm}$ (at $\kappa_{\text{R}}$) |
# | Stopband $T_{\min}$ & extinction (primary / ref) | $0.00026$ ($35.91\;\text{dB}$) / $0.00012$ ($39.18\;\text{dB}$) | $0.00245$ ($26.10\;\text{dB}$) / $0.00311$ ($25.07\;\text{dB}$) | $\operatorname{sech}^2(\kappa L) = 0.00245$ ($\kappa L = 3.70$) / $0.00311$ ($\kappa L = 3.58$) |
# | Coupling $\kappa_{T_{\min}}$ vs $\kappa_{\text{BW}}$ (primary / ref) | $1490.0$ vs $1242.7\;\text{cm}^{-1}$ / $1606.0$ vs $1162.6\;\text{cm}^{-1}$ | $1141.3$ vs $1162.8\;\text{cm}^{-1}$ / $1104.7$ vs $1080.2\;\text{cm}^{-1}$ | Self-consistent within $1.9\%$ (L3) & $2.2\%$ (R) |
# | Misaligned ($\Delta x = \Lambda/2$) $R_{\max}$ & $T_{\min}$ | $R_{\max} = 0.0001$, $T_{\min} = 0.9095$ ($0.41\;\text{dB}$) | $R_{\max} = 0.0003$, $T_{\min} = 0.9931$ ($0.03\;\text{dB}$) | $\kappa(\Lambda/2) = \kappa_0|\cos(\pi/2)| = 0 \implies R = 0, T = 1$ [1] |
# | Period sweep $1 - T_{\min}$ ($N = 25, 50, 100$) | $0.4289,\; 0.9396,\; 0.9997$ | $0.5217,\; 0.9009,\; 0.9975$ | $\tanh^2(\kappa_{\text{L3}} N\Lambda) = 0.5297,\; 0.9056,\; 0.9975$ |
# | Period sweep $\kappa_{T_{\min}}$ ($N = 25, 50, 100$) | $967.8,\; 1284.7,\; 1490.0\;\text{cm}^{-1}$ | $1126.5,\; 1125.6,\; 1141.3\;\text{cm}^{-1}$ | Constant $\kappa$ independent of $N$ ($1.4\%$ spread in `--full`) |
# | Depth sweep $\kappa_{T_{\min}}$ ($\Delta w = 25, 50, 75\;\text{nm}$) | $871.2,\; 1490.0,\; 1371.6\;\text{cm}^{-1}$ | $779.0,\; 1141.3,\; 1357.0\;\text{cm}^{-1}$ | $\Gamma\kappa_{1\text{D,step}}(\Delta w)$ ($\Gamma = 0.524$ at $50\;\text{nm}$) |
# | Depth sweep $\Delta\lambda_{\text{FWHM}}$ ($\Delta w = 25, 50, 75\;\text{nm}$) | $26.49,\; 36.69,\; 39.09\;\text{nm}$ | $23.28,\; 31.23,\; 33.07\;\text{nm}$ | $22.77,\; 32.19,\; 37.39\;\text{nm}$ (CMT from $\kappa_{T_{\min}}$) |
# | Uniform vs Gaussian-apodized sidelobe level | $-7.05\;\text{dB}$ vs $-22.24\;\text{dB}$ ($-15.19\;\text{dB}$) | $-3.52\;\text{dB}$ ($-3.74\;\text{dB}$ R) vs $-13.84\;\text{dB}$ ($-10.32\;\text{dB}$) | Uniform CMT first sidelobe $-4.05\;\text{dB}$ (L3) / $-4.23\;\text{dB}$ (R) |
# | Kottke vs staircase $|\lambda_B - \lambda_{B,\text{ref}}|$ | L1: $9.20$ vs $19.87\;\text{nm}$; L2: $2.59$ vs $10.13\;\text{nm}$ | L1: $11.71$ vs $22.42\;\text{nm}$; L2: $5.14$ vs $12.64\;\text{nm}$; L3: $2.55$ vs $8.36\;\text{nm}$ | $1.91\times \to 2.46\times \to 3.28\times$ Kottke advantage [4] |


# %%
def get_task_table(quick=False):
    """Return the Bragg grating study task table for `--quick` (13 tasks) or `--full` (15 tasks)."""
    n_pml = 6 if quick else 10
    energy_decay = 2e-2 if quick else 1e-4
    max_time_100 = 0.95e-12 if quick else 1.80e-12
    cpw_sweep, nf_sweep = (14, 6) if quick else (20, 8)
    lvl_sweep = "L2" if quick else "L3"

    specs = [
        ("underresolved_kottke", "underresolved", "kottke", 10, 0, 100, 50e-9, 0.0, "uniform", False, False, True),
        ("underresolved_staircase", "underresolved", "staircase", 10, 0, 100, 50e-9, 0.0, "uniform", False, False, True),
        ("kottke_L1", "L1", "kottke", 10, 4, 100, 50e-9, 0.0, "uniform", False, False, False),
        ("staircase_L1", "L1", "staircase", 10, 4, 100, 50e-9, 0.0, "uniform", False, False, False),
        ("kottke_L2", "L2", "kottke", 14, 6, 100, 50e-9, 0.0, "uniform", bool(quick), bool(quick), False),
        ("staircase_L2", "L2", "staircase", 14, 6, 100, 50e-9, 0.0, "uniform", False, False, False),
    ]
    if not quick:
        specs.extend(
            [
                ("kottke_L3", "L3", "kottke", 20, 8, 100, 50e-9, 0.0, "uniform", True, True, False),
                ("staircase_L3", "L3", "staircase", 20, 8, 100, 50e-9, 0.0, "uniform", False, False, False),
            ]
        )
    cpw_ref, nf_ref = (20, 8) if quick else (40, 16)
    specs.extend(
        [
            ("kottke_reference", "reference", "kottke", cpw_ref, nf_ref, 100, 50e-9, 0.0, "uniform", False, False, False),
            ("misaligned_half_period", lvl_sweep, "kottke", cpw_sweep, nf_sweep, 100, 50e-9, 0.5 * PITCH, "uniform", True, False, False),
            ("periods_N25", lvl_sweep, "kottke", cpw_sweep, nf_sweep, 25, 50e-9, 0.0, "uniform", False, False, False),
            ("periods_N50", lvl_sweep, "kottke", cpw_sweep, nf_sweep, 50, 50e-9, 0.0, "uniform", False, False, False),
            ("depth_dw25nm", lvl_sweep, "kottke", cpw_sweep, nf_sweep, 100, 25e-9, 0.0, "uniform", False, False, False),
            ("depth_dw75nm", lvl_sweep, "kottke", cpw_sweep, nf_sweep, 100, 75e-9, 0.0, "uniform", False, False, False),
            ("apodized_gaussian", lvl_sweep, "kottke", cpw_sweep, nf_sweep, 100, 50e-9, 0.0, "gaussian", False, False, False),
        ]
    )

    tasks = []
    for idx, (
        name,
        lvl,
        avg,
        cpw,
        n_feat,
        n_per,
        dw,
        shift,
        apod,
        c_fields,
        r_movie,
        is_demo,
    ) in enumerate(specs):
        scale_t = max(0.45, n_per / 100.0)
        tasks.append(
            {
                "index": idx,
                "name": name,
                "level": lvl,
                "averaging": avg,
                "cpw": cpw,
                "n_feature": n_feat,
                "is_demo": is_demo,
                "n_periods": n_per,
                "dw": dw,
                "shift": shift,
                "apodization": apod,
                "n_pml": n_pml,
                "energy_decay": energy_decay,
                "max_time": max_time_100 * scale_t,
                "collect_fields": c_fields,
                "record_movie": r_movie,
                "movie_frames": 28 if quick else 48,
                "description": (
                    f"2D EIM Bragg grating ({name}: {avg}, cpw={cpw}, N_feat={n_feat}, "
                    f"N_per={n_per}, dw={dw * 1e9:.0f}nm, shift={shift * 1e9:.0f}nm, {apod})"
                ),
            }
        )
    return tasks


def run_single_task(task_cfg, out_dir):
    """
    Execute one 2D EIM Bragg grating study task and write ``task_<I>.npz`` and
    ``task_<I>.json`` into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task_cfg["index"]
    t0 = time.perf_counter()

    structures, ports, domain = build_bragg_grating(
        n_periods=task_cfg["n_periods"],
        pitch=PITCH,
        w_core=W_CORE,
        dw=task_cfg["dw"],
        shift=task_cfg["shift"],
        apodization=task_cfg["apodization"],
    )
    neff_slab = slab_neff(N_SI, N_AIR, H_SI, WL_REF, n_sub=N_SIO2, polarization="TE")
    dx_bg = WL_MIN / (neff_slab * float(task_cfg["cpw"]))
    mesh_overrides = build_bragg_mesh_overrides(
        n_periods=task_cfg["n_periods"],
        pitch=PITCH,
        w_core=W_CORE,
        dw=task_cfg["dw"],
        duty_cycle=DUTY_CYCLE,
        shift=task_cfg["shift"],
        n_feature=task_cfg["n_feature"],
        dx_background=dx_bg,
    )
    use_gpu = os.environ.get("WAKIS_USE_GPU", "0") == "1"
    pic = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=N_FREQS,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging=task_cfg["averaging"],
        cells_per_wavelength=task_cfg["cpw"],
        domain=domain,
        n_pml=task_cfg["n_pml"],
        cfln=0.85,
        mesh_overrides=mesh_overrides,
        field_wavelengths=(WL_FIELD_BRAGG,),
        record_movie=task_cfg["record_movie"],
        movie_frames=task_cfg["movie_frames"],
        use_gpu=use_gpu,
    )

    s_mat = pic.run_s_parameters(
        drive=["in"],
        energy_decay=task_cfg["energy_decay"],
        max_time=task_cfg["max_time"],
        collect_fields=task_cfg["collect_fields"],
    )
    n_steps = int(getattr(pic.solver, "Nt", int(np.ceil(task_cfg["max_time"] / pic.solver.dt))))

    mesh_meta = measure_grating_mesh_metrics(
        pic.grid.x,
        pic.grid.y,
        pic.grid.z,
        dw=task_cfg["dw"],
        n_periods=task_cfg["n_periods"],
        pitch=PITCH,
        w_core=W_CORE,
        duty_cycle=DUTY_CYCLE,
        shift=task_cfg["shift"],
        n_pml=task_cfg["n_pml"],
    )

    length = float(task_cfg["n_periods"]) * PITCH
    r_pow = np.abs(s_mat[("in", "in")]) ** 2
    t_pow = np.abs(s_mat[("out", "in")]) ** 2
    metrics = extract_stopband_metrics(pic.wavelengths, r_pow, t_pow, length)

    payload = {
        "task_index": np.int32(idx),
        "name": np.array(task_cfg["name"]),
        "level": np.array(task_cfg["level"]),
        "averaging": np.array(task_cfg["averaging"]),
        "cpw": np.float32(task_cfg["cpw"]),
        "n_feature": np.int32(task_cfg["n_feature"]),
        "min_feature_nm": np.float64(mesh_meta["min_feature_nm"]),
        "dx_background_nm": np.float64(mesh_meta["dx_background_nm"]),
        "dx_min_nm": np.float64(mesh_meta["dx_min_nm"]),
        "dx_min_x_nm": np.float64(mesh_meta["dx_min_x_nm"]),
        "dy_min_nm": np.float64(mesh_meta["dy_min_nm"]),
        "cells_across_feature": np.float64(mesh_meta["cells_across_feature"]),
        "cells_across_tooth_x": np.float64(mesh_meta["cells_across_tooth_x"]),
        "n_cells": np.int32(mesh_meta["n_cells"]),
        "n_steps": np.int32(n_steps),
        "n_periods": np.int32(task_cfg["n_periods"]),
        "dw": np.float64(task_cfg["dw"]),
        "shift": np.float64(task_cfg["shift"]),
        "apodization": np.array(task_cfg["apodization"]),
        "n_eff_slab": np.float64(pic.n_eff_slab),
        "frequencies": pic.frequencies.astype(np.float64),
        "wavelengths": pic.wavelengths.astype(np.float64),
        "S11": s_mat[("in", "in")].astype(np.complex128),
        "S21": s_mat[("out", "in")].astype(np.complex128),
        "lambda_bragg": np.float64(metrics["lambda_bragg"]),
        "stopband_width_nm": np.float64(metrics["fwhm_bandwidth"] * 1e9),
        "R_max": np.float64(metrics["R_max"]),
        "T_min": np.float64(metrics["T_min"]),
    }

    if task_cfg["collect_fields"]:
        if task_cfg["name"] in ("kottke_L2", "kottke_L3"):
            modes_in = pic.solve_port_modes(
                "in", frequencies=c_light / WL_FIELD_BRAGG, n_modes=1
            )
            modes_out = pic.solve_port_modes(
                "out", frequencies=c_light / WL_FIELD_BRAGG, n_modes=1
            )
            y_in = modes_in[0].coords["Eb"][0]
            ey_in = modes_in[0].e["b"][:, 0]
            m_in = np.abs(y_in) <= 0.55e-6
            y_out = modes_out[0].coords["Eb"][0]
            ey_out = modes_out[0].e["b"][:, 0]
            m_out = np.abs(y_out) <= 0.55e-6
            payload["port_y_in"] = y_in[m_in].astype(np.float32)
            payload["port_ey_in"] = ey_in[m_in].astype(np.complex64)
            payload["port_neff_in"] = np.float64(np.real(modes_in[0].neff))
            payload["port_y_out"] = y_out[m_out].astype(np.float32)
            payload["port_ey_out"] = ey_out[m_out].astype(np.complex64)
            payload["port_neff_out"] = np.float64(np.real(modes_out[0].neff))

            # Dispersion of the average-width (500 nm) waveguide across the band
            wl_disp = np.linspace(WL_MIN, WL_MAX, 13)
            neff_avg = np.array(
                [
                    slab_neff(
                        pic.n_eff_slab,
                        N_AIR,
                        W_CORE + DW_CORRUG,
                        w,
                        polarization="TM",
                    )
                    for w in wl_disp
                ]
            )
            neff_min = np.array(
                [
                    slab_neff(pic.n_eff_slab, N_AIR, W_CORE, w, polarization="TM")
                    for w in wl_disp
                ]
            )
            neff_max = np.array(
                [
                    slab_neff(
                        pic.n_eff_slab,
                        N_AIR,
                        W_CORE + 2.0 * DW_CORRUG,
                        w,
                        polarization="TM",
                    )
                    for w in wl_disp
                ]
            )
            dneff_dl = np.gradient(neff_avg, wl_disp)
            ng_avg = neff_avg - wl_disp * dneff_dl
            payload["disp_wavelengths"] = wl_disp.astype(np.float64)
            payload["disp_neff"] = neff_avg.astype(np.float64)
            payload["disp_neff_min"] = neff_min.astype(np.float64)
            payload["disp_neff_max"] = neff_max.astype(np.float64)
            payload["disp_ng"] = ng_avg.astype(np.float64)

        if WL_FIELD_BRAGG in pic.dft_fields:
            fmap = pic.dft_fields[WL_FIELD_BRAGG]
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
            payload["flux_total_box"] = pic.flux_balance["total_box"].astype(np.float64)

    npz_path = out_dir / f"task_{idx}.npz"
    np.savez_compressed(npz_path, **payload)

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    meta = {
        "task_index": idx,
        "name": task_cfg["name"],
        "level": task_cfg["level"],
        "averaging": task_cfg["averaging"],
        "cpw": task_cfg["cpw"],
        "n_feature": task_cfg["n_feature"],
        "min_feature_nm": mesh_meta["min_feature_nm"],
        "dx_background_nm": mesh_meta["dx_background_nm"],
        "dx_min_nm": mesh_meta["dx_min_nm"],
        "dx_min_x_nm": mesh_meta["dx_min_x_nm"],
        "dy_min_nm": mesh_meta["dy_min_nm"],
        "cells_across_feature": mesh_meta["cells_across_feature"],
        "cells_across_tooth_x": mesh_meta["cells_across_tooth_x"],
        "n_periods": task_cfg["n_periods"],
        "dw_nm": round(task_cfg["dw"] * 1e9, 1),
        "shift_nm": round(task_cfg["shift"] * 1e9, 1),
        "apodization": task_cfg["apodization"],
        "grid_shape": mesh_meta["grid_shape"],
        "n_cells": mesh_meta["n_cells"],
        "n_steps": n_steps,
        "lambda_bragg_nm": round(metrics["lambda_bragg"] * 1e9, 2),
        "stopband_width_nm": round(metrics["fwhm_bandwidth"] * 1e9, 2),
        "R_max": round(metrics["R_max"], 4),
        "T_min": round(metrics["T_min"], 4),
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mb": round(rss_mb, 2),
        "npz_bytes": int(npz_path.stat().st_size),
    }
    (out_dir / f"task_{idx}.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[task_{idx} {task_cfg['name']}] cpw={meta['cpw']} N={meta['n_feature']} "
        f"min_feat={meta['min_feature_nm']:.1f}nm dx_bg={meta['dx_background_nm']:.2f}nm "
        f"dx_min={meta['dx_min_nm']:.2f}nm caf={meta['cells_across_feature']:.2f} "
        f"cells={meta['n_cells']} steps={meta['n_steps']} "
        f"lam_B={meta['lambda_bragg_nm']:.2f}nm BW={meta['stopband_width_nm']:.2f}nm "
        f"R_max={meta['R_max']:.4f} T_min={meta['T_min']:.4f} "
        f"time={elapsed:.2f}s rss={rss_mb:.1f}MB",
        flush=True,
    )
    return meta


# %% [markdown]
# ## 6. Generating the complete diagnostic figure suite
#
# When invoked with `--plot` (or at the end of `--quick` / `--full`),
# `generate_all_figures` reads `data/task_*.npz` and renders the setup figures and diagnostic suite into
# `figures/`:
#
# 0. `00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png` & `00_setup_table.md` — Two-figure setup layout and setup specification table (`wakis.schematic.save_setup_figures`): domain extents, CPML and $\pm z$ periodic boundaries, corrugated silicon waveguide $\varepsilon_{r,\text{slab}} = 8.03$ with periodic unit cell ($\Lambda = 324\;\text{nm}$, Bloch phase $\Delta\phi = \beta_B\Lambda = \pi$), `ModeSource`, port `ModeMonitor` planes, 2D Poynting `FluxBox`, non-uniform graded mesh with $1\times$ tooth zoom, spatial wave packet, and excitation pulse/spectrum.
# 1. `01_bragg_grating_layout.png` — Side-by-side dual waveguide layout (Waveguide 1 aligned at $y_1 = +0.766\;\mu\text{m}$, Waveguide 2 $\Lambda/2$-shifted at $y_2 = -0.766\;\mu\text{m}$) with zoomed insets of the in-phase vs half-period-shifted sidewall teeth.
# 2. `02_eim_vertical_stack.png` — Asymmetric $220\;\text{nm}$ SOI stack ($\text{SiO}_2$ substrate, $\text{Si}$ core, air cladding), 1D $\text{TE}_0$ mode profile, and $n_{\text{eff,slab}}(\lambda)$.
# 3. `03_corrugation_and_apodization_profiles.png` — Sidewall width modulation $w(x)$ for uniform, $\Lambda/2$-shifted, and Gaussian-apodized gratings, plus the CMT destructive-interference curve $\kappa(\Delta x_{\text{shift}}/\Lambda)$.
# 4. `04_permittivity_mesh_zoom.png` — Three-panel zoomed in-plane permittivity map with Yee cell edges and exact polygon outlines on the $50\;\text{nm} \times 162\;\text{nm}$ sidewall teeth comparing the unrefined uniform mesh ($0.97\;\text{cells/tooth}$), the baseline sidewall-strip graded mesh (`build_bragg_mesh_overrides`), and the tuned core-and-tooth graded mesh (`build_tuned_bragg_mesh_overrides`, equal cell count to `uniform_N4`).
# 5. `05_port_mode_profiles.png` — Input and output port eigenmode profiles $|E_y(y)|$ and $\mathrm{Re}(E_y(y))$.
# 6. `06_port_neff_dispersion.png` — Channel waveguide effective index $n_{\text{eff}}(\lambda)$, graphical Bragg condition $\lambda / (2\Lambda) = n_{\text{eff}}(\lambda)$, and group index $n_g(\lambda)$.
# 7. `07_field_intensity_bragg.png` — Steady-state logarithmic intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ near $\lambda_B$ with `schematic.overlay_geometry` (aligned stopband decay vs $\Lambda/2$-misaligned transmission).
# 8. `08_field_real_ey_bragg.png` — Steady-state wave-front phase pattern $\mathrm{Re}(E_y)$ near $\lambda_B$ with `schematic.overlay_geometry` (aligned vs $\Lambda/2$-misaligned).
# 9. `09_pulse_propagation.gif` — Animated time-domain pulse reflection and transmission in the Bragg grating with `schematic.overlay_geometry`.
# 10. `10_aligned_vs_misaligned_spectra.png` — Linear and dB transmission ($|S_{21}|^2$) and reflection ($|S_{11}|^2$) spectra of aligned vs $\Lambda/2$-misaligned gratings compared with analytical CMT and the under-resolved warning run.
# 11. `11_periods_sweep_and_cmt.png` — Reflection spectra vs period count $N \in \{25, 50, 100\}$ and peak reflectivity $R_{\max}(N)$ vs CMT $\tanh^2(\kappa N\Lambda)$.
# 12. `12_corrugation_depth_sweep.png` — Stopband transmission spectra, extracted coupling coefficient $\kappa(\Delta w)$, and $-3\;\text{dB}$ bandwidth vs corrugation depth $\Delta w \in \{25, 50, 75\}\;\text{nm}$.
# 13. `13_apodized_vs_uniform_sidelobes.png` — Uniform vs Gaussian-apodized grating reflection spectra in dB demonstrating sidelobe suppression.
# 14. `14_kottke_vs_staircase_convergence.png` — Three-panel convergence and mesh-upgrade comparison: error vs tooth resolution $N_{\text{tooth}}$, error vs total cell count $N_{\text{cells}}$ (Sidewall-Strip Graded vs Tuned Graded vs Uniform Fine `uniform_N4`), and stopband alignment across resolution levels.
# 15. `15_port_time_trace_and_flux_budget.png` — Port time-domain waveform $E_y(t)$ and 2D Poynting flux-box energy conservation audit.


# %%
def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Read ``data_dir/task_*.npz`` and render all setup and diagnostic figures into ``fig_dir``."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tasks_data = {}
    for p in sorted(data_dir.glob("task_*.npz")):
        d = dict(np.load(p, allow_pickle=False))
        name = str(d["name"])
        tasks_data[name] = d

    primary_name = "kottke_L3" if "kottke_L3" in tasks_data else "kottke_L2"
    if primary_name not in tasks_data:
        raise FileNotFoundError(
            f"Missing primary task '{primary_name}' in {data_dir}. Run --quick or --full first."
        )
    primary = tasks_data[primary_name]

    # 0. Setup schematic (00a_setup_geometry.png, 00b_setup_mesh_and_excitation.png, 00_setup_table.md)
    plot_bragg_setup_schematic(
        fig_dir / "00a_setup_geometry.png",
        n_periods=100,
        pitch=PITCH,
        w_core=W_CORE,
        dw=DW_CORRUG,
        duty_cycle=DUTY_CYCLE,
    )

    # 1. Dual side-by-side Bragg grating layout + zoomed view of aligned vs shifted teeth
    dual_structs, dual_ports, dual_domain = build_dual_bragg_gratings()
    with figstyle.apply_style():
        fig, (ax0, ax1) = figstyle.new_figure(
            nrows=2, ncols=1, width=10.6, height=6.2, gridspec_kw={"height_ratios": [1.0, 1.15]}
        )
        plot_layout(
            dual_structs,
            ports=dual_ports,
            domain=dual_domain,
            title="(a) Full 100-period dual waveguide Bragg grating domain (L = 32.40 µm, Λ = 324 nm)",
            ax=ax0,
        )
        ax0.set_aspect("auto")
        plot_layout(
            dual_structs,
            ports=(),
            domain=(0.0, 2.592e-6, -1.35e-6, 1.35e-6),
            annotations=[
                {
                    "text": "WG 1 (aligned, shift = 0)\n±y teeth in phase → max κ",
                    "xy": (1.05, +Y_SEP_HALF * 1e6 + 0.275),
                    "xytext": (0.65, 1.18),
                    "arrowprops": dict(arrowstyle="->", color="#1b9e77", lw=1.2),
                },
                {
                    "text": "WG 2 (misaligned, shift = Λ/2 = 162 nm)\n−y teeth shifted → κ = 0",
                    "xy": (1.21, -Y_SEP_HALF * 1e6 - 0.275),
                    "xytext": (1.75, -1.18),
                    "arrowprops": dict(arrowstyle="->", color="#d95f02", lw=1.2),
                },
            ],
            title="(b) Zoomed top view (8 periods, x ∈ [0, 2.59 µm]): aligned vs Λ/2-shifted sidewall teeth",
            ax=ax1,
        )
        figstyle.finish(fig)
        figstyle.save(fig, fig_dir / "01_bragg_grating_layout.png")

    # 2. Effective-Index Method (EIM) vertical stack (SiO2 / 220 nm Si / air)
    with figstyle.apply_style():
        plot_eim_stack(
            n_core=N_SI,
            n_clad=N_SIO2,
            thickness=H_SI,
            wavelength_ref=WL_REF,
            wavelengths=np.linspace(WL_MIN, WL_MAX, 51),
            out_path=fig_dir / "02_eim_vertical_stack.png",
        )

    # 3. Corrugation profiles, Gaussian apodization envelope, and CMT phase-shift law
    with figstyle.apply_style():
        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=10.8, height=4.0)
        x_per = np.linspace(-0.5 * 100 * PITCH * 1e6, 0.5 * 100 * PITCH * 1e6, 201)
        u_norm = x_per / (0.5 * 100 * PITCH * 1e6)
        env_unif = np.ones_like(x_per) * (DW_CORRUG * 1e9)
        env_gauss = np.exp(-4.0 * u_norm**2) * (DW_CORRUG * 1e9)
        env_rcos = 0.5 * (1.0 + np.cos(np.pi * u_norm)) * (DW_CORRUG * 1e9)
        ax0.plot(
            x_per,
            env_unif,
            label="Uniform grating (Δw = 50 nm)",
            **figstyle.series_style(0, with_marker=False),
        )
        ax0.plot(
            x_per,
            env_gauss,
            label=r"Gaussian apodized ($\alpha = 4.0$)",
            **figstyle.series_style(1, with_marker=False),
        )
        ax0.plot(
            x_per,
            env_rcos,
            label="Raised-cosine apodized",
            **figstyle.series_style(2, with_marker=False),
        )
        ax0.set(
            xlabel="Axial position x along grating [µm]",
            ylabel="Sidewall tooth depth Δw(x) [nm]",
            ylim=(-2, 58),
            title="(a) Longitudinal corrugation apodization profiles",
        )
        ax0.legend(loc="lower center", fontsize=8.0)

        shift_norm = np.linspace(0.0, 1.0, 201)
        kappa_norm = np.abs(np.cos(np.pi * shift_norm))
        r_max_shift = np.tanh(2.15 * kappa_norm) ** 2
        ax1.plot(
            shift_norm,
            kappa_norm,
            label=r"Normalized coupling $\kappa(\Delta x)/\kappa_0 = |\cos(\pi\Delta x/\Lambda)|$",
            **figstyle.series_style(0, with_marker=False),
        )
        ax1.plot(
            shift_norm,
            r_max_shift,
            label=r"CMT peak reflectivity $R_{\max} = \tanh^2(\kappa L)$",
            **figstyle.series_style(1, with_marker=False),
        )
        ax1.plot([0.0], [1.0], "o", color="#059669", ms=7, label="WG 1: aligned (Δx = 0)")
        ax1.plot(
            [0.5],
            [0.0],
            "s",
            color="#dc2626",
            ms=7,
            label="WG 2: misaligned (Δx = Λ/2, κ = 0)",
        )
        ax1.set(
            xlabel="Normalized sidewall phase offset Δx / Λ",
            ylabel="Normalized magnitude",
            title="(b) Destructive interference vs sidewall shift (Wang 2014)",
        )
        ax1.legend(loc="upper right", fontsize=7.8)
        figstyle.finish(fig)
        figstyle.save(fig, fig_dir / "03_corrugation_and_apodization_profiles.png")

    # 4. Three-panel permittivity & mesh zoom comparing:
    #    (a) Unrefined uniform (Task 0, 0.97 cells/tooth)
    #    (b) Sidewall-strip graded Kottke (Task 4 L2, build_bragg_mesh_overrides)
    #    (c) Tuned core+tooth graded Kottke (graded_tuned_L2, build_tuned_bragg_mesh_overrides, equal cells to uniform_N4)
    short_n_per = 6
    short_structs, short_ports, short_dom = build_bragg_grating(n_periods=short_n_per)
    neff_slab_fig4 = float(primary["n_eff_slab"])
    study_sum = load_graded_vs_uniform_study_summary()
    gt_l2 = (
        study_sum["graded_tuned"][1]
        if len(study_sum["graded_tuned"]) > 1
        else {"n_cells": 379680, "lambda_B_nm": 1552.12, "bandwidth_3db_nm": 30.31, "total_wall_time_s": 4617.5}
    )
    unif_n4 = study_sum["uniform_N4"]
    t0_d = tasks_data.get("underresolved_kottke", primary)
    t4_d = tasks_data.get("kottke_L2", primary)

    mesh_cases = [
        (
            "kottke",
            10,
            0,
            False,
            f"(a) Uniform coarse (0.97/tooth, {int(t0_d['n_cells']):,} cells)\n"
            f"λ_B = {float(t0_d['lambda_bragg']) * 1e9:.2f} nm, BW = {float(t0_d['stopband_width_nm']):.2f} nm",
        ),
        (
            "kottke",
            14,
            6,
            False,
            f"(b) Sidewall graded L2 (6/tooth, {int(t4_d['n_cells']):,} cells)\n"
            f"λ_B = {float(t4_d['lambda_bragg']) * 1e9:.2f} nm, BW = {float(t4_d['stopband_width_nm']):.2f} nm",
        ),
        (
            "kottke",
            14,
            6,
            True,
            f"(c) Tuned graded L2 (6/tooth, {int(gt_l2.get('n_cells', 379680)):,} cells)\n"
            f"λ_B = {float(gt_l2.get('lambda_B_nm', 1552.12)):.2f} nm, BW = {float(gt_l2.get('bandwidth_3db_nm', 30.31)):.2f} nm",
        ),
    ]
    xlim_zoom = (-0.34, 0.34)
    ylim_zoom = (0.16, 0.32)
    short_structs[0].name = "Si EIM core"
    verts_short_um = short_structs[0].vertices * 1e6
    with figstyle.apply_style():
        fig, axes_m = figstyle.new_figure(nrows=1, ncols=3, width=15.6, height=4.4, sharey=True)
        for ax, (avg_m, cpw_m, nf_m, use_tuned, ttl_m) in zip(axes_m, mesh_cases):
            dx_bg_m = WL_MIN / (neff_slab_fig4 * float(cpw_m))
            ov_fn = build_tuned_bragg_mesh_overrides if use_tuned else build_bragg_mesh_overrides
            ov_m = ov_fn(
                n_periods=short_n_per,
                pitch=PITCH,
                w_core=W_CORE,
                dw=DW_CORRUG,
                duty_cycle=DUTY_CYCLE,
                shift=0.0,
                n_feature=nf_m,
                dx_background=dx_bg_m,
            )
            pic_m = PICSimulation(
                WL_MIN,
                WL_MAX,
                structures=short_structs,
                ports=short_ports,
                n_core=N_SI,
                n_clad=N_AIR,
                n_sub=N_SIO2,
                wavelength_ref=WL_REF,
                two_d=True,
                averaging=avg_m,
                cells_per_wavelength=cpw_m,
                domain=short_dom,
                n_pml=6,
                mesh_overrides=ov_m,
            )
            g_m = pic_m.solver.grid
            iz_m = pic_m.solver.Nz // 2
            ieps_m = pic_m.solver.ieps[:, :, iz_m, "y"]
            if hasattr(ieps_m, "get"):
                ieps_m = ieps_m.get()
            eps_r_m = np.where(
                ieps_m > 0, 1.0 / (np.where(ieps_m > 0, ieps_m, 1.0) * epsilon_0), 1.0
            )
            x_um_m, y_um_m = g_m.x * 1e6, g_m.y * 1e6
            pcm = ax.pcolormesh(
                x_um_m,
                y_um_m,
                eps_r_m.T,
                cmap="viridis",
                vmin=1.0,
                vmax=neff_slab_fig4**2,
                shading="flat",
            )
            schematic.draw_mesh(
                ax,
                pic_m.solver,
                plane="xy",
                unit="µm",
                every=1,
                color="white",
                alpha=0.65,
                linewidth=0.55,
                show_cpw=False,
            )
            schematic.overlay_geometry(
                ax,
                pic_m.solver,
                plane="xy",
                unit="µm",
                structures=short_structs,
                show_boundaries=False,
                show_materials=True,
                show_sources=False,
                show_monitors=False,
                label_materials=False,
                outline_color="none",
            )
            ax.plot(
                np.append(verts_short_um[:, 0], verts_short_um[0, 0]),
                np.append(verts_short_um[:, 1], verts_short_um[0, 1]),
                color="#ffea00",
                lw=1.4,
                clip_on=True,
                zorder=6,
            )
            ax.annotate(
                "",
                xy=(-0.081, 0.275),
                xytext=(-0.081, 0.225),
                arrowprops=dict(arrowstyle="<->", color="#ff4d4d", lw=1.5),
            )
            ax.text(
                -0.070,
                0.250,
                "Δw = 50 nm",
                color="white",
                fontsize=7.6,
                fontweight="bold",
                va="center",
                bbox=dict(boxstyle="round,pad=0.14", fc="black", ec="none", alpha=0.68),
            )
            ax.set(
                xlim=xlim_zoom,
                ylim=ylim_zoom,
                aspect="auto",
                xlabel="x [µm]",
                title=ttl_m,
            )
            cb = fig.colorbar(pcm, ax=ax, fraction=0.042, pad=0.03)
            cb.set_label(r"$\varepsilon_{r,y}$")
        axes_m[0].set_ylabel("y [µm]")
        figstyle.finish(fig)
        figstyle.save(fig, fig_dir / "04_permittivity_mesh_zoom.png")

    # 5. Port mode profiles at input and output
    if "port_y_in" in primary:
        with figstyle.apply_style():
            plot_port_modes(
                primary["port_y_in"] * 1e6,
                primary["port_ey_in"],
                float(primary["port_neff_in"]),
                primary["port_y_out"] * 1e6,
                primary["port_ey_out"],
                float(primary["port_neff_out"]),
                structures_outline_y=[(-0.5 * W_CORE * 1e6, +0.5 * W_CORE * 1e6)],
                out_path=fig_dir / "05_port_mode_profiles.png",
            )

    # 6. Channel waveguide effective index, Bragg condition, and group index (2 panels, no twinx)
    if "disp_wavelengths" in primary:
        wl_d_nm = np.asarray(primary["disp_wavelengths"], dtype=float) * 1e9
        neff_d = np.asarray(primary["disp_neff"], dtype=float)
        ng_d = np.asarray(primary["disp_ng"], dtype=float)
        neff_slab_d = float(primary["n_eff_slab"])
        with figstyle.apply_style():
            fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=10.8, height=3.9)
            ax0.plot(
                wl_d_nm,
                neff_d,
                label=r"Channel mode $n_{\mathrm{eff}}(\lambda)$ (2D EIM)",
                **figstyle.series_style(0),
            )
            ax0.plot(
                wl_d_nm,
                (wl_d_nm * 1e-9) / (2.0 * PITCH),
                label=r"Bragg line $\lambda / (2\Lambda)$",
                **figstyle.series_style(1, with_marker=False),
            )
            ax0.axhline(
                neff_slab_d,
                color="0.45",
                ls=":",
                lw=1.4,
                label=f"Slab index $n_{{\\mathrm{{eff,slab}}}}$ = {neff_slab_d:.4f}",
            )
            ax0.set(
                xlabel="Wavelength λ [nm]",
                ylabel=r"Effective index $n_{\mathrm{eff}}$",
                title="(a) Channel effective index & Bragg phase matching",
            )
            ax0.legend(loc="best", fontsize=7.8)

            ax1.plot(
                wl_d_nm,
                ng_d,
                label=r"Channel group index $n_g(\lambda)$",
                **figstyle.series_style(2),
            )
            ax1.set(
                xlabel="Wavelength λ [nm]",
                ylabel=r"Group index $n_g$",
                title=f"(b) Modal group index (mean $n_g$ = {float(np.mean(ng_d)):.4f})",
            )
            ax1.legend(loc="best", fontsize=8.0)
            figstyle.finish(fig)
            figstyle.save(fig, fig_dir / "06_port_neff_dispersion.png")

    # 7 & 8. Steady-state field maps at lambda_B with schematic.overlay_geometry
    pic_al_obj, src_al_obj, mons_al_obj, structs_al_schem = build_bragg_setup_objects(
        n_periods=100, shift=0.0, cpw=14, n_feature=6, n_pml=10
    )
    pic_mis_obj, src_mis_obj, mons_mis_obj, structs_mis_schem = build_bragg_setup_objects(
        n_periods=100, shift=0.5 * PITCH, cpw=14, n_feature=6, n_pml=10
    )
    misaligned = tasks_data.get("misaligned_half_period")
    if "dft_x" in primary and misaligned is not None and "dft_x" in misaligned:
        # Figure 7: Logarithmic intensity |E|^2 [dB] with overlay_geometry
        with figstyle.apply_style():
            fig, (ax0, ax1) = figstyle.new_figure(nrows=2, ncols=1, width=10.6, height=5.8, sharex=True)
            for ax, d, pic_o, src_o, mons_o, st_sc, lbl in [
                (
                    ax0,
                    primary,
                    pic_al_obj,
                    src_al_obj,
                    mons_al_obj,
                    structs_al_schem,
                    r"(a) Aligned grating ($\Delta x = 0$): strong Bragg stopband reflection & exponential decay",
                ),
                (
                    ax1,
                    misaligned,
                    pic_mis_obj,
                    src_mis_obj,
                    mons_mis_obj,
                    structs_mis_schem,
                    r"(b) Misaligned grating ($\Delta x = \Lambda/2$): destructive interference ($\kappa = 0$) & full transmission",
                ),
            ]:
                x_um, y_um = d["dft_x"] * 1e6, d["dft_y"] * 1e6
                e2 = d["dft_E2"]
                f_max = float(np.max(e2)) or 1.0
                data_db = 10.0 * np.log10(np.maximum(e2 / f_max, 10.0 ** (-35.0 / 10.0)))
                im = ax.pcolormesh(
                    x_um,
                    y_um,
                    data_db.T,
                    cmap=figstyle.CMAP_MAGNITUDE,
                    vmin=-35.0,
                    vmax=0.0,
                    shading="auto",
                )
                schematic.overlay_geometry(
                    ax,
                    pic_o.solver,
                    plane="xy",
                    unit="µm",
                    structures=st_sc,
                    sources=[src_o],
                    monitors=mons_o,
                    show_boundaries=True,
                    show_materials=True,
                    show_sources=True,
                    show_monitors=True,
                    label_materials=False,
                    outline_color="none",
                )
                v_um_o = st_sc[0].vertices * 1e6
                ax.plot(
                    np.append(v_um_o[:, 0], v_um_o[0, 0]),
                    np.append(v_um_o[:, 1], v_um_o[0, 1]),
                    color="#ffea00",
                    lw=0.9,
                    alpha=0.88,
                    clip_on=True,
                    zorder=6,
                )
                ax.text(
                    -10.8,
                    0.36,
                    "Si EIM core (ε_r = 8.03)",
                    color="white",
                    fontsize=8.0,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.14", fc="black", ec="none", alpha=0.65),
                    zorder=8,
                )
                ax.set(
                    aspect="auto",
                    xlim=(-17.2, 17.2),
                    ylim=(-0.75, 0.75),
                    ylabel="y [µm]",
                    title=lbl,
                )
                cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
                cb.set_label(r"$10\log_{10}(|E|^2/|E|_{\max}^2)$ [dB]")
            ax1.set_xlabel("x [µm]")
            figstyle.finish(fig)
            figstyle.save(fig, fig_dir / "07_field_intensity_bragg.png")

        # Figure 8: Real part Re(Ey) wave-front map with overlay_geometry
        with figstyle.apply_style():
            fig, (ax0, ax1) = figstyle.new_figure(nrows=2, ncols=1, width=10.6, height=5.8, sharex=True)
            for ax, d, pic_o, src_o, mons_o, st_sc, lbl in [
                (
                    ax0,
                    primary,
                    pic_al_obj,
                    src_al_obj,
                    mons_al_obj,
                    structs_al_schem,
                    r"(a) Aligned grating ($\Delta x = 0$): $\mathrm{Re}(E_y)$ standing-wave envelope at $\lambda \approx \lambda_B$",
                ),
                (
                    ax1,
                    misaligned,
                    pic_mis_obj,
                    src_mis_obj,
                    mons_mis_obj,
                    structs_mis_schem,
                    r"(b) Misaligned grating ($\Delta x = \Lambda/2$): $\mathrm{Re}(E_y)$ unattenuated traveling wave",
                ),
            ]:
                x_um, y_um = d["dft_x"] * 1e6, d["dft_y"] * 1e6
                ey = np.real(d["dft_Ey"])
                vmax = float(np.percentile(np.abs(ey), 99.5)) or 1.0
                im = ax.pcolormesh(
                    x_um,
                    y_um,
                    ey.T / vmax,
                    cmap=figstyle.CMAP_DIVERGING,
                    vmin=-1.0,
                    vmax=1.0,
                    shading="auto",
                )
                schematic.overlay_geometry(
                    ax,
                    pic_o.solver,
                    plane="xy",
                    unit="µm",
                    structures=st_sc,
                    sources=[src_o],
                    monitors=mons_o,
                    show_boundaries=True,
                    show_materials=True,
                    show_sources=True,
                    show_monitors=True,
                    label_materials=False,
                    outline_color="none",
                )
                v_um_o = st_sc[0].vertices * 1e6
                ax.plot(
                    np.append(v_um_o[:, 0], v_um_o[0, 0]),
                    np.append(v_um_o[:, 1], v_um_o[0, 1]),
                    color="#0f172a",
                    lw=0.9,
                    alpha=0.88,
                    clip_on=True,
                    zorder=6,
                )
                ax.text(
                    -10.8,
                    0.36,
                    "Si EIM core (ε_r = 8.03)",
                    color="#0f172a",
                    fontsize=8.0,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82),
                    zorder=8,
                )
                ax.set(
                    aspect="auto",
                    xlim=(-17.2, 17.2),
                    ylim=(-0.75, 0.75),
                    ylabel="y [µm]",
                    title=lbl,
                )
                cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
                cb.set_label(r"Normalized $\mathrm{Re}(E_y)$")
            ax1.set_xlabel("x [µm]")
            figstyle.finish(fig)
            figstyle.save(fig, fig_dir / "08_field_real_ey_bragg.png")

    # 9. Animated pulse propagation GIF with overlay_geometry on every frame
    if "movie_frames" in primary and len(primary["movie_frames"]) > 0:
        render_bragg_pulse_gif(
            primary["movie_x"],
            primary["movie_y"],
            primary["movie_frames"],
            primary["movie_times"],
            pic_al_obj.solver,
            src_al_obj,
            mons_al_obj,
            structs_al_schem,
            out_path=fig_dir / "09_pulse_propagation.gif",
            fps=12,
        )

    # Helper to extract sorted interior spectrum (1505-1615 nm)
    def _get_spectrum(d):
        w = d["wavelengths"] * 1e9
        order = np.argsort(w)
        w = w[order]
        m = (w >= 1505.0) & (w <= 1615.0)
        r = np.clip(np.abs(d["S11"][order][m]) ** 2, 0.0, 1.0)
        t = np.clip(np.abs(d["S21"][order][m]) ** 2, 0.0, 1.0)
        return w[m], r, t

    wl_b, r_al, t_al = _get_spectrum(primary)
    L_100 = 100.0 * PITCH
    ng_est = float(np.mean(primary["disp_ng"])) if "disp_ng" in primary else 3.45
    m_al = extract_stopband_metrics(wl_b * 1e-9, r_al, t_al, L_100, n_g=ng_est)
    cmt_fit = cmt_bragg_spectrum(
        wl_b * 1e-9,
        n_eff=m_al["lambda_bragg"] / (2.0 * PITCH),
        pitch=PITCH,
        kappa=m_al["kappa_from_tmin"],
        length=L_100,
        n_g=ng_est,
        wavelength_bragg=m_al["lambda_bragg"],
    )

    # 10. Aligned vs Misaligned spectra (linear & dB) + analytical CMT + under-resolved warning
    under_k = tasks_data.get("underresolved_kottke")
    with figstyle.apply_style():
        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.3)
        ax0.plot(
            wl_b,
            t_al,
            color="#059669",
            lw=2.1,
            label=rf"Aligned $T = |S_{{21}}|^2$ ({primary_name}, N={int(primary['n_feature'])})",
        )
        ax0.plot(
            wl_b,
            r_al,
            color="#1d4ed8",
            lw=2.0,
            ls="--",
            label=rf"Aligned $R = |S_{{11}}|^2$ ({primary_name}, N={int(primary['n_feature'])})",
        )
        ax0.plot(
            wl_b,
            cmt_fit["T"],
            color="0.25",
            lw=1.5,
            ls=":",
            label=rf"Analytical CMT ($\kappa L = {m_al['kappa_from_tmin'] * L_100:.2f}$)",
        )
        if under_k is not None:
            wl_uk, _, t_uk = _get_spectrum(under_k)
            ax0.plot(
                wl_uk,
                t_uk,
                color="#7c3aed",
                lw=1.7,
                ls=":",
                label="Warning: under-resolved T (10 cpw, 0.97 cells/tooth)",
            )
        if misaligned is not None:
            wl_m, r_mis, t_mis = _get_spectrum(misaligned)
            ax0.plot(
                wl_m,
                t_mis,
                color="#d97706",
                lw=2.1,
                label=r"Misaligned ($\Lambda/2$) $T = |S_{21}|^2$",
            )
            ax0.plot(
                wl_m,
                r_mis,
                color="#dc2626",
                lw=1.8,
                ls="-.",
                label=r"Misaligned ($\Lambda/2$) $R = |S_{11}|^2$",
            )
        ax0.set(
            xlabel="Wavelength λ [nm]",
            ylabel="Modal power fraction (linear)",
            ylim=(-0.03, 1.05),
            title=f"(a) Linear spectra (stopband @ {m_al['lambda_bragg'] * 1e9:.1f} nm)",
        )
        ax0.legend(loc="center left", fontsize=7.3)

        ax1.plot(
            wl_b,
            10.0 * np.log10(np.maximum(t_al, 1e-6)),
            color="#059669",
            lw=2.1,
            label=f"Aligned T [dB] ({primary_name})",
        )
        ax1.plot(
            wl_b,
            10.0 * np.log10(np.maximum(r_al, 1e-6)),
            color="#1d4ed8",
            lw=2.0,
            ls="--",
            label=f"Aligned R [dB] ({primary_name})",
        )
        ax1.plot(
            wl_b,
            10.0 * np.log10(np.maximum(cmt_fit["T"], 1e-6)),
            color="0.25",
            lw=1.5,
            ls=":",
            label="CMT T [dB]",
        )
        if under_k is not None:
            wl_uk, _, t_uk = _get_spectrum(under_k)
            ax1.plot(
                wl_uk,
                10.0 * np.log10(np.maximum(t_uk, 1e-6)),
                color="#7c3aed",
                lw=1.7,
                ls=":",
                label="Warning: under-resolved T [dB] (0.97 cells/tooth)",
            )
        if misaligned is not None:
            ax1.plot(
                wl_m,
                10.0 * np.log10(np.maximum(t_mis, 1e-6)),
                color="#d97706",
                lw=2.1,
                label="Misaligned (Λ/2) T [dB]",
            )
            ax1.plot(
                wl_m,
                10.0 * np.log10(np.maximum(r_mis, 1e-6)),
                color="#dc2626",
                lw=1.8,
                ls="-.",
                label="Misaligned (Λ/2) R [dB]",
            )
        ax1.set(
            xlabel="Wavelength λ [nm]",
            ylabel="Modal power [dB]",
            ylim=(-42.0, 2.0),
            title=f"(b) Logarithmic spectra (extinction = {m_al['extinction_db']:.1f} dB)",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="lower left", fontsize=7.3)
        figstyle.finish(fig)
        figstyle.save(fig, fig_dir / "10_aligned_vs_misaligned_spectra.png")

    # 11. Period-count sweep (N = 25, 50, 100) and CMT R_max = tanh^2(kappa L)
    with figstyle.apply_style():
        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)
        period_cases = [
            (25, tasks_data.get("periods_N25"), "#7c3aed", "--"),
            (50, tasks_data.get("periods_N50"), "#d97706", "-."),
            (100, primary, "#059669", "-"),
        ]
        n_vals, r_sim_vals, t_sim_vals = [], [], []
        for n_per, d_p, col, ls in period_cases:
            if d_p is None:
                continue
            w_p, r_p, t_p = _get_spectrum(d_p)
            m_p = extract_stopband_metrics(w_p * 1e-9, r_p, t_p, n_per * PITCH, n_g=ng_est)
            ax0.plot(
                w_p,
                r_p,
                color=col,
                ls=ls,
                lw=2.0,
                label=f"N = {n_per} (L = {n_per * PITCH * 1e6:.2f} µm, 1−T_min = {1.0 - m_p['T_min']:.3f})",
            )
            n_vals.append(n_per)
            r_sim_vals.append(1.0 - m_p["T_min"])
            t_sim_vals.append(m_p["T_min"])

        ax0.set(
            xlabel="Wavelength λ [nm]",
            ylabel=r"Power reflectivity $R = |S_{11}|^2$",
            ylim=(-0.02, 1.05),
            title="(a) Reflection spectrum vs number of periods N",
        )
        ax0.legend(loc="upper right", fontsize=7.8)

        n_cont = np.linspace(1, 120, 200)
        kappa_ref = m_al["kappa_from_tmin"]
        r_cmt_curve = np.tanh(kappa_ref * n_cont * PITCH) ** 2
        t_cmt_curve = 1.0 - r_cmt_curve
        ax1.plot(
            n_cont,
            r_cmt_curve,
            color="#1d4ed8",
            lw=2.0,
            label=rf"CMT $R_{{\max}}(N) = \tanh^2(\kappa N\Lambda)$ ($\kappa = {kappa_ref * 1e-2:.0f}\;\mathrm{{cm}}^{{-1}}$)",
        )
        ax1.plot(
            n_cont,
            t_cmt_curve,
            color="#d97706",
            lw=1.8,
            ls="--",
            label=r"CMT $T_{\min}(N) = \mathrm{sech}^2(\kappa N\Lambda)$",
        )
        if n_vals:
            ax1.plot(
                n_vals,
                r_sim_vals,
                "o",
                color="#059669",
                ms=7,
                label="Wakis 2D EIM stopband rejection (1 − T_min)",
            )
            ax1.plot(
                n_vals,
                t_sim_vals,
                "s",
                color="#dc2626",
                ms=6.5,
                label="Wakis 2D EIM minimum transmission T_min",
            )
        ax1.set(
            xlabel="Number of grating periods N",
            ylabel="Stopband reflection / transmission",
            ylim=(-0.03, 1.05),
            title=r"(b) Stopband saturation vs CMT $\tanh^2(\kappa L)$",
        )
        ax1.legend(loc="center right", fontsize=7.6)
        figstyle.finish(fig)
        figstyle.save(fig, fig_dir / "11_periods_sweep_and_cmt.png")

    # 12. Corrugation depth sweep (dw = 25, 50, 75 nm) — 3 panels, no twinx
    with figstyle.apply_style():
        fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=14.4, height=4.1)
        depth_cases = [
            (25.0, tasks_data.get("depth_dw25nm"), "#7c3aed", "--"),
            (50.0, primary, "#059669", "-"),
            (75.0, tasks_data.get("depth_dw75nm"), "#d97706", "-."),
        ]
        dw_list, kappa_sim_list, bw_sim_list = [], [], []
        for dw_nm, d_dw, col, ls in depth_cases:
            if d_dw is None:
                continue
            w_d, r_d, t_d = _get_spectrum(d_dw)
            m_d = extract_stopband_metrics(w_d * 1e-9, r_d, t_d, L_100, n_g=ng_est)
            ax0.plot(
                w_d,
                10.0 * np.log10(np.maximum(t_d, 1e-6)),
                color=col,
                ls=ls,
                lw=2.0,
                label=f"Δw = {dw_nm:.0f} nm (ext = {m_d['extinction_db']:.1f} dB, BW = {m_d['fwhm_bandwidth'] * 1e9:.1f} nm)",
            )
            dw_list.append(dw_nm)
            kappa_sim_list.append(m_d["kappa_from_tmin"] * 1e-2)  # cm^-1
            bw_sim_list.append(m_d["fwhm_bandwidth"] * 1e9)

        ax0.set(
            xlabel="Wavelength λ [nm]",
            ylabel=r"Transmission $10\log_{10}|S_{21}|^2$ [dB]",
            ylim=(-42.0, 2.0),
            title="(a) Stopband transmission vs Δw",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="lower left", fontsize=7.4)

        if dw_list:
            dw_arr_nm = np.linspace(15.0, 85.0, 29)
            neff_slab_v = float(primary["n_eff_slab"])
            lam_b0 = m_al["lambda_bragg"]
            kap_step_50 = (
                cmt_coupling_from_neff_step(
                    slab_neff(
                        neff_slab_v,
                        N_AIR,
                        W_CORE + 2.0 * DW_CORRUG,
                        lam_b0,
                        polarization="TM",
                    )
                    - slab_neff(neff_slab_v, N_AIR, W_CORE, lam_b0, polarization="TM"),
                    lam_b0,
                    duty_cycle=DUTY_CYCLE,
                )
                * 1e-2
            )
            overlap_ratio = (m_al["kappa_from_tmin"] * 1e-2) / max(kap_step_50, 1e-30)
            kappa_step_cm = [
                cmt_coupling_from_neff_step(
                    slab_neff(
                        neff_slab_v,
                        N_AIR,
                        W_CORE + 2.0 * (d_nm * 1e-9),
                        lam_b0,
                        polarization="TM",
                    )
                    - slab_neff(neff_slab_v, N_AIR, W_CORE, lam_b0, polarization="TM"),
                    lam_b0,
                    duty_cycle=DUTY_CYCLE,
                )
                * 1e-2
                * overlap_ratio
                for d_nm in dw_arr_nm
            ]
            ax1.plot(
                dw_list,
                kappa_sim_list,
                label=r"Wakis 2D EIM $\kappa$ [$\mathrm{cm}^{-1}$]",
                **figstyle.series_style(0),
            )
            ax1.plot(
                dw_arr_nm,
                kappa_step_cm,
                label=rf"1D step scaling $\Gamma\kappa_{{1\mathrm{{D}}}}(\Delta w)$ ($\Gamma={overlap_ratio:.2f}$)",
                **figstyle.series_style(1, with_marker=False),
            )
            ax1.legend(loc="lower right", fontsize=7.4)
            ax2.plot(
                dw_list,
                bw_sim_list,
                label="Wakis 2D EIM −3 dB bandwidth [nm]",
                **figstyle.series_style(2),
            )
            ax2.legend(loc="lower right", fontsize=7.6)
        ax1.set(
            xlabel="Sidewall corrugation depth Δw [nm]",
            ylabel=r"Coupling coefficient $\kappa$ [$\mathrm{cm}^{-1}$]",
            title="(b) Coupling strength κ vs tooth depth Δw",
        )
        ax2.set(
            xlabel="Sidewall corrugation depth Δw [nm]",
            ylabel="Stopband −3 dB bandwidth [nm]",
            title="(c) Stopband −3 dB bandwidth vs Δw",
        )
        figstyle.finish(fig)
        figstyle.save(fig, fig_dir / "12_corrugation_depth_sweep.png")

    # 13. Apodized vs uniform grating sidelobes
    apod = tasks_data.get("apodized_gaussian")
    if apod is not None:
        w_a, r_a, t_a = _get_spectrum(apod)
        m_ap = extract_stopband_metrics(w_a * 1e-9, r_a, t_a, L_100, n_g=ng_est)
        with figstyle.apply_style():
            fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=7.6, height=4.1)
            ax.plot(
                wl_b,
                10.0 * np.log10(np.maximum(r_al, 1e-6)),
                label=f"Uniform grating R (SL = {m_al['sidelobe_db']:.1f} dB)",
                **figstyle.series_style(0, with_marker=False),
            )
            ax.plot(
                w_a,
                10.0 * np.log10(np.maximum(r_a, 1e-6)),
                label=f"Gaussian-apodized R (SL = {m_ap['sidelobe_db']:.1f} dB)",
                **figstyle.series_style(1, with_marker=False),
            )
            ax.plot(
                wl_b,
                10.0 * np.log10(np.maximum(t_al, 1e-6)),
                label="Uniform grating T",
                **figstyle.series_style(2, with_marker=False),
            )
            ax.plot(
                w_a,
                10.0 * np.log10(np.maximum(t_a, 1e-6)),
                label="Gaussian-apodized T",
                **figstyle.series_style(3, with_marker=False),
            )
            ax.set(
                xlabel="Wavelength λ [nm]",
                ylabel="Modal power [dB]",
                ylim=(-42.0, 2.0),
                title="Uniform vs Gaussian-apodized Bragg grating spectra (sidelobe suppression)",
            )
            figstyle.spectrum_grid(ax)
            ax.legend(loc="lower left", fontsize=7.8)
            figstyle.finish(fig)
            figstyle.save(fig, fig_dir / "13_apodized_vs_uniform_sidelobes.png")

    # 14. Three-panel convergence & graded-vs-uniform mesh comparison (no twiny)
    levels = [
        lvl
        for lvl in ("L1", "L2", "L3")
        if f"kottke_{lvl}" in tasks_data and f"staircase_{lvl}" in tasks_data
    ]
    if levels and "kottke_reference" in tasks_data:
        ref_d = tasks_data["kottke_reference"]
        w_ref, r_ref, t_ref = _get_spectrum(ref_d)
        m_ref = extract_stopband_metrics(w_ref * 1e-9, r_ref, t_ref, L_100)
        lam_b_ref_nm = m_ref["lambda_bragg"] * 1e9
        bw_ref_nm = m_ref["fwhm_bandwidth"] * 1e9
        ref_cpw = int(ref_d["cpw"])
        ref_caf = float(ref_d.get("cells_across_feature", ref_d.get("n_feature", 8)))

        cafs = [
            float(
                tasks_data[f"kottke_{lvl}"].get(
                    "cells_across_feature", tasks_data[f"kottke_{lvl}"].get("n_feature", 4)
                )
            )
            for lvl in levels
        ]
        cells_k = [float(tasks_data[f"kottke_{lvl}"]["n_cells"]) for lvl in levels]
        err_lam_k, err_lam_s = [], []
        err_bw_k, err_bw_s = [], []

        for lvl in levels:
            for avg, lk, bwk in [
                ("kottke", err_lam_k, err_bw_k),
                ("staircase", err_lam_s, err_bw_s),
            ]:
                d_cur = tasks_data[f"{avg}_{lvl}"]
                w_c, r_c, t_c = _get_spectrum(d_cur)
                m_c = extract_stopband_metrics(w_c * 1e-9, r_c, t_c, L_100)
                lk.append(max(abs(m_c["lambda_bragg"] * 1e9 - lam_b_ref_nm), 0.05))
                bwk.append(max(abs(m_c["fwhm_bandwidth"] * 1e9 - bw_ref_nm), 0.05))

        study_s = load_graded_vs_uniform_study_summary()
        gt_list = study_s["graded_tuned"]
        unif_d = study_s["uniform_N4"]
        lam_cont_nm = float(study_s["reference"].get("lambda_B_ref_nm", 1551.3084))
        bw_cont_nm = float(study_s["reference"].get("BW_ref_nm", 30.4094))

        with figstyle.apply_style():
            fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=15.6, height=4.5)
            n_arr = np.asarray(cafs, dtype=float)

            # Panel (a): Error vs tooth resolution N_tooth
            ax0.loglog(
                n_arr,
                err_lam_s,
                "s-",
                color="#d97706",
                lw=2.0,
                ms=6.5,
                label=r"Staircase $|\Delta\lambda_B|$ vs R [nm]",
            )
            ax0.loglog(
                n_arr,
                err_lam_k,
                "o-",
                color="#059669",
                lw=2.0,
                ms=6.5,
                label=r"Sidewall-graded Kottke $|\Delta\lambda_B|$ vs R [nm]",
            )
            ax0.loglog(
                n_arr,
                err_bw_s,
                "s--",
                color="#b45309",
                lw=1.6,
                ms=5.8,
                label=r"Staircase $|\Delta\lambda_{\mathrm{BW}}|$ vs R [nm]",
            )
            ax0.loglog(
                n_arr,
                err_bw_k,
                "o--",
                color="#0d9488",
                lw=1.6,
                ms=5.8,
                label=r"Sidewall-graded Kottke $|\Delta\lambda_{\mathrm{BW}}|$ vs R [nm]",
            )
            if gt_list:
                gt_n = [float(g["n_tooth_actual"]) for g in gt_list]
                gt_elam = [abs(float(g["lambda_B_nm"]) - lam_cont_nm) for g in gt_list]
                gt_ebw = [abs(float(g["bandwidth_3db_nm"]) - bw_cont_nm) for g in gt_list]
                ax0.loglog(
                    gt_n,
                    gt_elam,
                    "^-",
                    color="#1d4ed8",
                    lw=2.0,
                    ms=6.8,
                    label=r"Tuned-graded Kottke $|\Delta\lambda_B|$ [nm]",
                )
                ax0.loglog(
                    gt_n,
                    gt_ebw,
                    "^--",
                    color="#3b82f6",
                    lw=1.6,
                    ms=6.0,
                    label=r"Tuned-graded Kottke $|\Delta\lambda_{\mathrm{BW}}|$ [nm]",
                )
            if unif_d:
                ax0.loglog(
                    [float(unif_d["n_tooth_actual"])],
                    [abs(float(unif_d["lambda_B_nm"]) - lam_cont_nm)],
                    "D",
                    color="#dc2626",
                    ms=8.0,
                    label=f"Uniform fine N=4 |Δλ_B| ({int(unif_d['n_cells']):,} cells)",
                )
            ax0.set(
                xlabel="Cells across 50 nm tooth depth N_tooth",
                ylabel="Spectral error [nm]",
                title="(a) Convergence vs tooth resolution N_tooth",
            )
            ax0.legend(loc="lower left", fontsize=6.6)

            # Panel (b): Equal-accuracy / equal-cost trade-off vs total cell count n_cells
            ax1.loglog(
                cells_k + [float(ref_d["n_cells"])],
                [abs(float(tasks_data[f"kottke_{lvl}"]["lambda_bragg"]) * 1e9 - lam_cont_nm) for lvl in levels]
                + [abs(lam_b_ref_nm - lam_cont_nm)],
                "o-",
                color="#059669",
                lw=2.0,
                ms=6.5,
                label="Sidewall-graded Kottke |Δλ_B| [nm]",
            )
            if gt_list:
                gt_cells = [float(g["n_cells"]) for g in gt_list]
                gt_elam = [abs(float(g["lambda_B_nm"]) - lam_cont_nm) for g in gt_list]
                gt_ebw = [abs(float(g["bandwidth_3db_nm"]) - bw_cont_nm) for g in gt_list]
                ax1.loglog(
                    gt_cells,
                    gt_elam,
                    "^-",
                    color="#1d4ed8",
                    lw=2.0,
                    ms=6.8,
                    label="Tuned-graded Kottke |Δλ_B| [nm]",
                )
                ax1.loglog(
                    gt_cells,
                    gt_ebw,
                    "^--",
                    color="#3b82f6",
                    lw=1.6,
                    ms=6.0,
                    label="Tuned-graded Kottke |ΔBW| [nm]",
                )
                for g in gt_list:
                    ax1.annotate(
                        f"N={int(round(g['n_tooth_actual']))}\n({g['total_wall_time_s']:.0f}s)",
                        xy=(float(g["n_cells"]), abs(float(g["lambda_B_nm"]) - lam_cont_nm)),
                        xytext=(0, 7),
                        textcoords="offset points",
                        fontsize=6.7,
                        color="#1d4ed8",
                        ha="center",
                    )
            if unif_d:
                u_c = float(unif_d["n_cells"])
                u_elam = abs(float(unif_d["lambda_B_nm"]) - lam_cont_nm)
                u_ebw = abs(float(unif_d["bandwidth_3db_nm"]) - bw_cont_nm)
                ax1.loglog(
                    [u_c],
                    [u_elam],
                    "D",
                    color="#dc2626",
                    ms=8.0,
                    label=f"Uniform N=4 |Δλ_B| ({int(u_c):,} cells, {unif_d['total_wall_time_s']:.0f}s)",
                )
                ax1.loglog(
                    [u_c],
                    [u_ebw],
                    "s",
                    color="#e11d48",
                    ms=7.0,
                    mfc="none",
                    mew=1.8,
                    label="Uniform N=4 |ΔBW| [nm]",
                )
            ax1.set(
                xlabel="Total 2D mesh cells N_cells",
                ylabel="Error vs continuum limit [nm]",
                title="(b) Graded vs uniform mesh cost-accuracy trade-off",
            )
            ax1.legend(loc="lower left", fontsize=6.6)

            # Panel (c): Stopband transmission spectra overlay across resolution ladder
            ax2.plot(
                w_ref,
                t_ref,
                color="black",
                lw=2.2,
                label=f"Kottke reference ({ref_cpw} cpw, N={ref_caf:.0f}, λ_B={lam_b_ref_nm:.1f} nm)",
            )
            cols_k = {"L1": "#7c3aed", "L2": "#059669", "L3": "#1d4ed8"}
            cols_s = {"L1": "#dc2626", "L2": "#d97706", "L3": "#b45309"}
            for lvl in levels:
                for avg, ls, c_map in (
                    ("kottke", "-", cols_k),
                    ("staircase", "--", cols_s),
                ):
                    d_cur = tasks_data[f"{avg}_{lvl}"]
                    w_c, _, t_c = _get_spectrum(d_cur)
                    caf_c = float(
                        d_cur.get("cells_across_feature", d_cur.get("n_feature", 4))
                    )
                    ax2.plot(
                        w_c,
                        t_c,
                        color=c_map.get(lvl, "#059669"),
                        ls=ls,
                        lw=1.5,
                        alpha=0.88,
                        label=f"{avg.capitalize()} {lvl} ({int(d_cur['cpw'])} cpw, {caf_c:.0f}/tooth)",
                    )
            if "underresolved_kottke" in tasks_data:
                d_uk = tasks_data["underresolved_kottke"]
                w_uk, _, t_uk = _get_spectrum(d_uk)
                ax2.plot(
                    w_uk,
                    t_uk,
                    color="#64748b",
                    ls=":",
                    lw=1.7,
                    label="Warning: under-resolved Kottke (0.97/tooth)",
                )
            ax2.set(
                xlabel="Wavelength λ [nm]",
                ylabel=r"Transmission $T = |S_{21}|^2$",
                xlim=(1515.0, 1605.0),
                ylim=(-0.03, 1.05),
                title="(c) Stopband alignment across resolution ladder",
            )
            ax2.legend(loc="lower left", fontsize=6.5)
            figstyle.finish(fig)
            figstyle.save(fig, fig_dir / "14_kottke_vs_staircase_convergence.png")

    # 15. Port time trace and 2D Poynting flux-box conservation audit
    if "trace_t" in primary and primary["trace_t"].size > 0:
        t_ps = primary["trace_t"] * 1e12
        ey_tr = primary["trace_Ey"]
        ey_norm = ey_tr / max(float(np.max(np.abs(ey_tr))), 1e-30)
        with figstyle.apply_style():
            fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.0)
            ax0.plot(
                t_ps,
                ey_norm,
                color="#1d4ed8",
                lw=1.3,
                label=r"Output port $E_y(t)/\max|E_y|$",
            )
            ax0.plot(
                t_ps,
                np.sqrt(np.maximum(ey_norm**2, 1e-10)),
                color="#d97706",
                lw=1.2,
                alpha=0.75,
                ls=":",
                label="Instantaneous envelope",
            )
            ax0.set(
                xlabel="Time t [ps]",
                ylabel=r"Normalized $E_y(t)$",
                title="(a) Output port time trace & DFB ring-down",
            )
            ax0.legend(loc="upper right", fontsize=8.0)

            ax1.plot(
                wl_b,
                r_al + t_al,
                color="#059669",
                lw=2.0,
                label=r"Guided modal sum $R + T = |S_{11}|^2 + |S_{21}|^2$",
            )
            if "flux_total_box" in primary:
                w_all = primary["wavelengths"] * 1e9
                ord_f = np.argsort(w_all)
                m_f = (w_all[ord_f] >= 1505.0) & (w_all[ord_f] <= 1615.0)
                f_box = primary["flux_total_box"][ord_f][m_f]
                f_lat = primary["flux_lateral"][ord_f][m_f]
                ax1.plot(
                    wl_b,
                    f_box,
                    color="#1d4ed8",
                    ls="--",
                    lw=1.8,
                    label=f"2D Poynting flux box total (mean={np.mean(f_box):.3f})",
                )
                ax1.plot(
                    wl_b,
                    f_lat,
                    color="#d97706",
                    ls=":",
                    lw=1.6,
                    label=f"Lateral radiation flux (mean={np.mean(f_lat) * 100:.1f}%)",
                )
            ax1.axhline(1.0, color="0.4", ls=":", lw=1.2, label="Unit incident power")
            ax1.set(
                xlabel="Wavelength λ [nm]",
                ylabel="Normalized power fraction",
                ylim=(-0.05, 1.15),
                title="(b) Passivity & 2D Poynting flux-box energy conservation",
            )
            ax1.legend(loc="center right", fontsize=7.8)
            figstyle.finish(fig)
            figstyle.save(fig, fig_dir / "15_port_time_trace_and_flux_budget.png")

    print(f"Rendered 16 figures in {fig_dir}")


# %% [markdown]
# ## 7. Limits, failure modes & when 2D EIM breaks down
#
# ### Failure mode: what goes wrong when a 50 nm tooth is resolved by only 1–2 cells
#
# The `underresolved_kottke` and `underresolved_staircase` demonstration pair
# (`cpw = 10` with no `mesh_overrides`, $\Delta y = 51.43\;\text{nm}$, giving
# only $0.97\;\text{cells}$ across each $50\;\text{nm}$ sidewall tooth) shows
# why a grid coarser than the smallest geometric feature fails regardless of the
# permittivity averaging rule:
#
# * **Staircasing without feature resolution (`underresolved_staircase`)**:
#   Because the $50\;\text{nm}$ tooth is comparable to a single $51.43\;\text{nm}$
#   Yee cell, rounding the cell fill fraction ($w \ge 0.5 \to 1$) snaps the
#   entire sidewall corrugation to the background grid lines. This artificially
#   widens the effective waveguide core, shifting the Bragg wavelength by
#   $+18.18\;\text{nm}$ ($\lambda_B = 1570.45\;\text{nm}$ vs $1552.27\;\text{nm}$
#   at the $\text{R} = (40, 16)$ reference; $+15.76\;\text{nm}$ vs $\text{L3}$ in
#   `--quick`) and yielding a $0.4748$ transmission RMS error $\|T - T_{\text{ref}}\|_2$
#   (and in `--quick` at $0.95\;\text{ps}$ truncation, over-predicting stopband
#   extinction at $61.9\;\text{dB}$). If the grid lines happened to fall at fill
#   $w < 0.5$, the teeth would vanish completely ($\kappa \to 0$).
# * **Subpixel smoothing without feature resolution (`underresolved_kottke`)**:
#   When both the inner core edge ($|y| = 225\;\text{nm}$) and the tooth tip
#   ($|y| = 275\;\text{nm}$) lie inside the same $51.43\;\text{nm}$ cell,
#   Kottke subpixel averaging blends the tooth and cladding into a single
#   intermediate permittivity value. Although its Bragg wavelength shift
#   ($+9.63\;\text{nm}$ at $1561.90\;\text{nm}$ vs $\text{R} = 1552.27\;\text{nm}$)
#   is nearly half that of staircase, averaging the entire tooth into one cell
#   washes out the periodic index contrast: the stopband $-3\;\text{dB}$ width
#   collapses from $30.63\;\text{nm}$ ($\text{R}$) and $31.23\;\text{nm}$ ($\text{L3}$)
#   to $26.75\;\text{nm}$ ($-3.88\;\text{nm}$ or $-12.7\%$ vs $\text{R}$), the
#   extracted coupling drops from $1104.7\;\text{cm}^{-1}$ ($\text{R}$) /
#   $1141.3\;\text{cm}^{-1}$ ($\text{L3}$) to $980.4\;\text{cm}^{-1}$, and the
#   stopband extinction drops to $21.59\;\text{dB}$ ($T_{\min} = 0.0069$ vs
#   $25.07\;\text{dB}$ at $\text{R}$ and $26.10\;\text{dB}$ at $\text{L3}$).
#   Subpixel smoothing reduces boundary quantization error, but it cannot recover
#   the field variation across a feature that is narrower than a single grid cell.
# * **Once the tooth is resolved ($N \ge 4$ via `mesh_overrides`)**:
#   Refining the two sidewall tooth strips to $h_y = \Delta w / N$ across
#   $\text{L1} = (10, 4)$ ($\Delta y_{\min} = 12.50\;\text{nm}$),
#   $\text{L2} = (14, 6)$ ($\Delta y_{\min} = 8.33\;\text{nm}$), and
#   $\text{L3} = (20, 8)$ ($\Delta y_{\min} = 6.25\;\text{nm}$) immediately
#   restores the stopband coupling ($31.93, 28.71, 26.10\;\text{dB}$ extinction
#   and $\Delta\lambda_{\text{BW}} = 33.00, 31.71, 31.23\;\text{nm}$, converging
#   monotonically toward $25.07\;\text{dB}$ and $30.63\;\text{nm}$ at $\text{R} = (40, 16)$).
#   Because the tooth depth is now spanned by $4\text{--}16$ cells, staircase no
#   longer loses or doubles whole teeth. Even so, Kottke maintains a
#   $1.91\times \to 2.46\times \to 3.28\times$ smaller error in $\lambda_B$
#   ($11.71 \to 5.14 \to 2.55\;\text{nm}$ vs staircase $22.42 \to 12.64 \to 8.36\;\text{nm}$),
#   a $2.19\times \to 3.06\times \to 4.02\times$ smaller error in $\Delta\lambda_{\text{BW}}$
#   ($2.37 \to 1.08 \to 0.61\;\text{nm}$ vs staircase $5.20 \to 3.31 \to 2.44\;\text{nm}$),
#   and a $1.50\times \to 1.61\times \to 1.79\times$ smaller transmission RMS error
#   $\|T - T_{\text{ref}}\|_2$ ($0.3549 \to 0.2325 \to 0.1552$ vs staircase
#   $0.5310 \to 0.3753 \to 0.2771$) because the longitudinal tooth edges
#   ($\Lambda = 324\;\text{nm}$) and core boundaries remain cut cells.
#
# ### Physical differences between 2D EIM and 3D SOI strip waveguides
#
# While the 2D effective-index method captures the fundamental physics of
# contra-directional Bragg coupling, stopband saturation $R_{\max} = \tanh^2(\kappa L)$,
# sidelobe suppression under Gaussian apodization ($-13.84\;\text{dB}$ vs
# $-3.52\;\text{dB}$ uniform at $\text{L3}$ and $-3.74\;\text{dB}$ at $\text{R}$,
# a $10.32\;\text{dB}$ sidelobe reduction, with short-wavelength sidelobes
# suppressed below $-20\;\text{dB}$), and complete cancellation of fundamental
# reflection in the $\Lambda/2$-shifted grating ($R_{\max} = 0.0003$,
# $T_{\min} = 0.9931$, $0.03\;\text{dB}$ maximum insertion loss, and 2D Poynting
# flux-box closure $\langle \Phi_{\text{box}} \rangle = 1.0000$), three physical
# effects separate 2D EIM from a 3D SOI strip waveguide (Wang *et al.* 2014 [1];
# Chrostowski & Hochberg 2015 [5]):
#
# 1. **Channel effective-index offset ($\lambda_{B,\text{2D}} = 1554.81\;\text{nm}$ at L3 / $1552.27\;\text{nm}$ at R)**:
#    Separating the vertical ($\text{TE}_0$ slab, $n_{\text{eff,slab}} = 2.8335$)
#    and horizontal ($w_{\text{core}} = 450\;\text{nm}$, $\Delta w = 50\;\text{nm}$)
#    confinement problems overestimates the corner field concentration of a
#    rectangular silicon strip waveguide ($n_{\text{eff,2D}} = 2.3955$ at $\text{R}$
#    and $2.3994$ at $\text{L3}$), shifting the 2D EIM stopband center $\lambda_B =
#    2 n_{\text{eff}}\Lambda$ toward longer wavelengths relative to a 3D strip mode.
# 2. **Sidewall coupling coefficient $\kappa$ overestimation**: In 2D EIM, the
#    entire vertical extent experiences the full lateral index contrast between
#    $n_{\text{eff,slab}} = 2.8335$ and air ($n = 1.0$), yielding a stronger
#    modal overlap with the $50\;\text{nm}$ sidewall teeth ($\kappa_{\text{2D}} =
#    1141.3\;\text{cm}^{-1}$ from $T_{\min}$ and $1162.8\;\text{cm}^{-1}$ from
#    $\Delta\lambda_{\text{null}}$ at $\text{L3}$, converging to $1104.7\;\text{cm}^{-1}$
#    and $1080.2\;\text{cm}^{-1}$ at $\text{R}$, and rising across the depth sweep
#    from $779.0\;\text{cm}^{-1}$ at $\Delta w = 25\;\text{nm}$ to
#    $1357.0\;\text{cm}^{-1}$ at $\Delta w = 75\;\text{nm}$, with $-3\;\text{dB}$
#    transmission rejection width $\Delta\lambda_{\text{BW}} = 31.23\;\text{nm}$ at $\text{L3}$
#    and $30.63\;\text{nm}$ at $\text{R}$) than a 3D strip mode,
#    where part of the electric energy resides in the $\text{SiO}_2$ substrate and
#    air cladding above/below the teeth.
# 3. **Out-of-plane radiation at short wavelengths**: In 3D asymmetric SOI
#    ($\text{SiO}_2$ substrate, air top cladding), grating diffraction orders can
#    phase-match to unguided substrate radiation modes below the Bragg wavelength
#    ($\lambda < (n_{\text{eff}} + n_{\text{SiO}_2})\Lambda \approx 1235\;\text{nm}$)
#    and scatter vertically at abrupt tooth corners, whereas in 2D EIM without
#    out-of-plane radiation, in-plane lateral radiation is only $0.12\%$ for the
#    aligned grating and $0.29\%$ for the $\Lambda/2$-shifted grating (with 2D
#    Poynting flux-box totals $\langle \Phi_{\text{box}} \rangle = 0.9982$ and $1.0000$).
#
# ## 8. Exercises
#
# 1. **Continuous $\kappa$ tuning via phase offset**: Simulate a 100-period
#    grating across $\Delta x_{\text{shift}} / \Lambda \in \{0, 0.15, 0.25, 0.35, 0.50\}$
#    and verify that the extracted coupling coefficient follows Wang *et al.*'s
#    cosine law $\kappa(\Delta x_{\text{shift}}) = \kappa_0 \cos(\pi\Delta x_{\text{shift}}/\Lambda)$.
# 2. **Quarter-wave phase-shifted DFB resonator**: Insert a $\Lambda/4 = 81\;\text{nm}$
#    defect spacer at the center ($x = 0$) of the aligned 100-period grating and
#    observe the narrow transmission peak that opens right at $\lambda_B$ inside
#    the stopband.
# 3. **Dual-waveguide evanescent crosstalk**: Run `build_dual_bragg_gratings()`
#    driving only `in_aligned` and measure the cross-coupled power at
#    `out_misaligned` and `in_misaligned` as a function of waveguide center-to-center
#    separation $d_y \in \{1.2, 1.532, 2.0, 2.5\}\;\mu\text{m}$.
#
# ## 9. References
#
# 1. X. Wang, Y. Wang, J. Flueckiger, R. Bojko, A. Liu, A. Reid, J. Pond,
#    N. A. F. Jaeger, and L. Chrostowski, "Precise control of the coupling
#    coefficient through destructive interference in silicon waveguide Bragg
#    gratings," *Opt. Lett.* **39**(19), 5519–5522 (2014),
#    DOI: [10.1364/OL.39.005519](https://doi.org/10.1364/OL.39.005519).
# 2. A. Yariv, "Coupled-mode theory for guided-wave optics," *IEEE J. Quantum
#    Electron.* **9**(9), 919–933 (1973),
#    DOI: [10.1109/JQE.1973.1077767](https://doi.org/10.1109/JQE.1973.1077767).
# 3. T. Erdogan, "Fiber grating spectra," *J. Lightwave Technol.* **15**(8),
#    1277–1294 (1997), DOI: [10.1109/50.618322](https://doi.org/10.1109/50.618322).
# 4. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing
#    of discretized numerical methods," *Phys. Rev. E* **77**, 036611 (2008),
#    DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
# 5. L. Chrostowski and M. Hochberg, *Silicon Photonics Design: From Devices to
#    Systems*, Cambridge University Press (2015), Chapter 4 (Waveguide Bragg
#    Gratings).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="18. Sidewall-Corrugated Silicon Waveguide Bragg Gratings (2D EIM)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run coarse 2D study locally and render all figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run converged 2D study across all tasks and render all figures",
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

    # Default when invoked with no flags (e.g. notebook build/execution):
    # if all task_*.npz files already exist in DATA_DIR and neither --quick nor --full
    # was explicitly forced, render figures directly from cached data; otherwise run --quick.
    quick = not args.full
    tasks = get_task_table(quick=quick)
    existing = list(Path(args.out).glob("task_*.npz"))
    if not (args.quick or args.full) and len(existing) >= len(tasks):
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)
        return

    for t in tasks:
        run_single_task(t, DATA_DIR)
    generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
