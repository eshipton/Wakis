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
# # Tutorial 5.9 · Waveguide-to-ring point coupler
#
# **Abstract.** Microring resonators are essential building blocks of silicon
# photonic integrated circuits (PICs), serving as wavelength-division
# multiplexers, electro-optic modulators, dispersion compensators, and
# Kerr frequency-comb cavities. Their spectral finesse, extinction ratio, and
# quality factor are governed by the single-pass point-coupling coefficients
# $(t, \kappa)$ between a straight bus waveguide and the curved ring waveguide.
# In this tutorial we model a $220\;\text{nm}$ silicon-on-insulator (SOI)
# waveguide-to-ring coupler ($w = 500\;\text{nm}$, radius $r = 5.0\;\mu\text{m}$,
# nominal coupling gap $g = 50\;\text{nm}$) in Wakis using the **2D
# effective-index method (EIM)** and **Kottke subpixel permittivity smoothing**.
# We extract broadband $4\times 4$ scattering parameters across the
# $1500\text{--}1600\;\text{nm}$ telecom window, sweep the coupling gap from
# $50\;\text{nm}$ to $250\;\text{nm}$, compare the numerical coupling against
# first-principles **coupled-mode theory (CMT)** and **5-layer coupled-slab
# supermode dispersion**, verify passivity, Lorentz reciprocity, and CPML
# waveguide extrusion, and synthesize the resulting all-pass microring
# resonator transmission spectrum.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Formulate the **2D effective-index (EIM)** reduction of a $220\;\text{nm}$
#    SOI bus-to-ring directional coupler (`Ex`, `Ey`, `Hz`; one periodic cell
#    along $z$) and construct a single-pass truncated ring coupler (`PolySlab`)
#    with CPML structure extrusion to suppress boundary reflections.
# 2. Extract broadband complex point-coupling coefficients
#    $t(\lambda) = S_{21}(\lambda)$ (`through`) and
#    $\kappa(\lambda) = S_{31}(\lambda)$ (`drop`), return loss $|S_{11}|^2$,
#    and isolated backward ring leakage $|S_{41}|^2$.
# 3. Derive and evaluate **spatial coupled-mode theory (CMT)** for curved
#    waveguide couplers—both from the isolated 2D EIM mode overlap integral and
#    from exact 5-layer even/odd supermode splitting—and fit the exponential
#    evanescent decay constant $\gamma_{\text{fit}}$ against the analytical
#    channel and slab evanescent decay rates.
# 4. Audit multi-port **passivity** ($\sum_j |S_{j1}|^2 \le 1$), **Lorentz
#    reciprocity** ($S_{21} = S_{12}$, $S_{31} = S_{13}$), **Kottke vs
#    staircase** subpixel convergence across narrow $50\;\text{nm}$ slots, and
#    angled/bent waveguide mode profiles (`angle_theta = 45 deg`,
#    `bend_radius = 5.0 um`).
#
# ## Contents
#
# 1. [Physical background & device specification](#1.-Physical-background-&-device-specification)
# 2. [Effective-Index Method (EIM) & Coupled-Mode Theory](#2.-Effective-Index-Method-(EIM)-&-Coupled-Mode-Theory)
# 3. [Parametric geometry & CPML waveguide termination](#3.-Parametric-geometry-&-CPML-waveguide-termination)
# 4. [Subpixel permittivity smoothing & local mesh overrides](#4.-Subpixel-permittivity-smoothing-&-local-mesh-overrides)
# 5. [Study tasks & execution modes](#5.-Study-tasks-&-execution-modes)
# 6. [Generating the complete 15-figure suite](#6.-Generating-the-complete-15-figure-suite)
# 7. [Limits, failure modes & when 2D EIM breaks down](#7.-Limits,-failure-modes-&-when-2D-EIM-breaks-down)
# 8. [Exercises](#8.-Exercises)
# 9. [References](#9.-References)

# %% [markdown]
# ## 1. Physical background & device specification
#
# A full time-domain simulation of a high-$Q$ microring resonator requires many
# round-trip cavity lifetimes for the intracavity field to ring down. Because
# the evanescent interaction between the straight bus waveguide and the ring of
# mean radius $r = 5.0\;\mu\text{m}$ is localized near the point of closest
# approach ($x = 0$), we can isolate the **single-pass directional point
# coupler** by absorbing the curved ring arms in the $+y$ boundary before they
# complete a round trip (Yariv 2000 [1]; Little *et al.* 1997 [2]).
#
# In the lossless point-coupler transfer matrix,
#
# $$\begin{pmatrix} b_{\text{through}} \\ b_{\text{drop}} \end{pmatrix} = \begin{pmatrix} t & \kappa \\ -\kappa^* & t^* \end{pmatrix} \begin{pmatrix} a_{\text{in}} \\ a_{\text{add}} \end{pmatrix}, \qquad |t(\omega)|^2 + |\kappa(\omega)|^2 = 1 - \ell_{\text{rad}}(\omega),$$
#
# driving the bus input port (`in`, $a_{\text{in}} = 1, a_{\text{add}} = 0$)
# yields the complex self-coupling (through) amplitude $t(\omega) = S_{21}(\omega)$
# and cross-coupling (drop) amplitude $\kappa(\omega) = S_{31}(\omega)$ in a
# single non-resonant pass of duration $\approx 0.35\;\text{ps}$.
#
# ### Device parameters
#
# | Parameter | Symbol | Value |
# |---|---|---|
# | Silicon core refractive index ($1550\;\text{nm}$) | $n_{\text{Si}}$ | $3.476$ (Li 1980 [5]) |
# | $\text{SiO}_2$ cladding refractive index ($1550\;\text{nm}$) | $n_{\text{SiO}_2}$ | $1.444$ (Malitson 1965 [6]) |
# | Silicon slab thickness | $h_{\text{Si}}$ | $220\;\text{nm}$ ($z \in [-110, +110]\;\text{nm}$) |
# | Bus and ring waveguide width | $w_{\text{wg}}$ | $500\;\text{nm}$ |
# | Ring mean radius | $r$ | $5.0\;\mu\text{m}$ ($r_{\text{in}} = 4.75\;\mu\text{m}, r_{\text{out}} = 5.25\;\mu\text{m}$) |
# | Nominal edge-to-edge coupling gap | $g_0$ | $50\;\text{nm}$ (swept over $50\text{--}250\;\text{nm}$) |
# | Ring center at $g_0 = 50\;\text{nm}$ | $(x_0, y_0)$ | $(0,\; w_{\text{wg}} + g_0 + r) = (0,\; 5.55\;\mu\text{m})$ |
# | Bus input / through port axial positions | $x_{\text{in}}, x_{\text{through}}$ | $\mp 5.3875\;\mu\text{m}$ |
# | Reference $45^\circ$ ring cross-section | $(r\sin\frac{\pi}{4},\; y_0 - r\cos\frac{\pi}{4})$ | $(3.5355,\; 2.0145)\;\mu\text{m}$ |
#
# ### Simulation setup schematic & specification table
#
# Figures [`00a_setup_geometry.png`](figures/00a_setup_geometry.png) and
# [`00b_setup_mesh_and_excitation.png`](figures/00b_setup_mesh_and_excitation.png)
# (generated via `wakis.schematic.save_setup_figures`) show the complete
# 2D effective-index simulation domain, 10-cell CPML absorbing boundaries on
# $\pm x$ and $\pm y$ (periodic on $\pm z$), $\text{Si}$ bus and curved ring
# cores ($n_{\text{Si}} = 3.476 \to n_{\text{eff,slab}} = 2.8478$,
# $\varepsilon_r = 8.11$) in $\text{SiO}_2$ cladding ($n_{\text{SiO}_2} = 1.444$,
# $\varepsilon_r = 2.09$), the launched $\text{TE}_0$ mode profile at Port 1
# (`in`), the coupled mode profile at Port 3 (`drop`), the four eigenmode port
# monitors (`in`, `through`, `drop`, `add`), and a $1\times$ zoom of the Yee mesh
# across the $g_0 = 50\;\text{nm}$ coupling slot alongside the broadband Gaussian
# excitation pulse and spectrum.
#
# ![SOI Waveguide-to-Ring Point Coupler Setup](figures/00a_setup_geometry.png)
# ![Waveguide-to-Ring Coupler Mesh and Excitation](figures/00b_setup_mesh_and_excitation.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 12.2 µm × 7.47 µm × 26.3 nm (z ∈ [-13.2 nm, 13.2 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 713 × 368 × 1 (262384 cells) |
# | Discretization | Δx & resolution | Δ_min = 6.24 nm, 87.3 cells/λ (in n_max = 2.85), 8.0 cells across feature (50 nm) |
# | Boundary | -x face | CPML (10 cells, 262 nm) |
# | Boundary | +x face | CPML (10 cells, 262 nm) |
# | Boundary | -y face | CPML (10 cells, 256 nm) |
# | Boundary | +y face | CPML (10 cells, 263 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | SiO₂ cladding (n=1.444, ε_r=2.09) (isotropic) |
# | Material | Region 1: Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab=2.8478, w=500 nm) | ε_r = 8.11, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-7.50 µm, 7.50 µm] (d = 15 µm) |
# | Material | Region 2: Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab=2.8478, w=500 nm) | ε_r = 8.11, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-5.25 µm, 5.25 µm] (d = 10.5 µm) |
# | Source | 1 · ModeSource | pos: x = -5.51 µm, y = 0.00 µm, pol: TE₀ (E_y, H_z), pulse: f₀ = 193.6 THz, Δf = 6.2 THz |
# | Monitor | 2 · ModeMonitor | Port 1 'in' (S₁₁), pos: x = -5.39 µm, y = 0.00 µm, freqs: 51 freqs in [187.4, 199.9] THz |
# | Monitor | 3 · ModeMonitor | Port 2 'through' (t = S₂₁), pos: x = 5.39 µm, y = 0.00 µm, freqs: 51 freqs in [187.4, 199.9] THz |
# | Monitor | 4 · ModeMonitor | Port 3 'drop' (κ = S₃₁), pos: x = 5.00 µm, y = 5.80 µm, freqs: 51 freqs in [187.4, 199.9] THz |
# | Monitor | 5 · ModeMonitor | Port 4 'add' (S₄₁), pos: x = -5.00 µm, y = 5.80 µm, freqs: 51 freqs in [187.4, 199.9] THz |
# | Variation | Coupling-gap sweep (task_9..task_12) | g ∈ {50, 100, 150, 200, 250} nm at L3=(20 cpw, 8 cells/gap) |
# | Variation | 3-port Lorentz reciprocity & CPML extrusion audit (task_4) | Drives on 'in', 'through', 'drop' + companion unextruded-CPML run |
# | Variation | Graded non-uniform Cartesian & hybrid unstructured ring mesh (task_13) | Mirror-symmetric graded Δy (8.00 cells across 50 nm slot) & non-square Δx > Δy_min (165,487 cells, −37.2% cells, 650.2 s, −54.8% wall time vs single-drive L3) + wakis.unstructured subtri ring rim |

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
from matplotlib.collections import LineCollection
import numpy as np
from scipy.constants import c as c_light, epsilon_0

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _helpers import (  # noqa: E402
    NonUniformRingCouplerPIC,
    bent_waveguide_radial_mode,
    build_ring_coupler_graded_axes,
    build_ring_coupler_hybrid_unstructured_mesh,
    cmt_coupling_vs_gap,
    cmt_coupling_vs_wavelength,
    fit_exponential_decay,
    isolated_eim_channel_mode,
    overlay_ring_coupler_geometry,
    plot_mesh_upgrade_comparison_ring_coupler,
    render_ring_coupler_field_map_with_overlay,
    render_ring_coupler_movie_with_overlay,
    ring_resonator_metrics,
    sample_angled_drop_slice,
    save_ring_coupler_setup_figures,
)
from wakis import figstyle  # noqa: E402
from wakis.templates import Box, Simulation  # noqa: E402
from wakis.templates.photonics import (  # noqa: E402
    PIC_MATERIALS,
    PICSimulation,
    PolySlab,
    Port,
    make_movie,
    n_to_eps,
    plot_cross_section_mesh,
    plot_eim_stack,
    plot_field_map,
    plot_layout,
    plot_neff_dispersion,
    plot_spectra_db,
    slab_neff,
    straight,
)

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# Material and spectral parameters
N_SI = PIC_MATERIALS["Si"]  # 3.476
N_SIO2 = PIC_MATERIALS["SiO2"]  # 1.444
H_SI = 220e-9  # 220 nm SOI thickness

WL_MIN = 1500e-9
WL_MAX = 1600e-9
WL_REF = 1550e-9
N_FREQS = 51

# Coupler geometric parameters [m]
W_WG = 0.50e-6
R_RING = 5.00e-6
GAP_NOMINAL = 0.05e-6  # 50 nm nominal edge-to-edge gap
GAPS_SWEEP_NM = (50, 100, 150, 200, 250)
X_PORT_BUS = 5.3875e-6
THETA_DROP_REF = 0.25 * np.pi  # 45 deg reference point on the curved ring

# %% [markdown]
# ## 2. Effective-Index Method (EIM) & Coupled-Mode Theory
#
# First we reduce the $220\;\text{nm}$ $\text{Si}/\text{SiO}_2$ vertical stack to
# its fundamental $\text{TE}_0$ slab effective index at $\lambda_0 = 1550\;\text{nm}$:
#
# $$n_{\text{eff,slab}}(1550\;\text{nm}) = 2.8478, \qquad n_{\text{clad}} = 1.444.$$
#
# In the 2D EIM plane (`Ex`, `Ey`, `Hz`), the straight bus waveguide of width
# $w = 500\;\text{nm}$ supports a fundamental in-plane mode with analytical
# 1D effective index $n_{\text{eff,ch}} = 2.4914$ ($n_{\text{eff}} = 2.4984$ and
# group index $n_g = 3.1677$ on the $L_3$ 2D Yee port plane at $1550\;\text{nm}$),
# propagation constant $\beta = k_0 n_{\text{eff,ch}}$, and cladding evanescent
# decay constant
#
# $$\gamma_{\text{evan}}(\lambda) = k_0 \sqrt{n_{\text{eff,ch}}(\lambda)^2 - n_{\text{clad}}^2} = 8.230\;\mu\text{m}^{-1} \quad (\text{vs } \gamma_{\text{slab}} = k_0\sqrt{n_{\text{eff,slab}}^2 - n_{\text{clad}}^2} = 9.950\;\mu\text{m}^{-1}).$$
#
# By spatial coupled-mode theory (Little *et al.* 1997 [2]; Yariv 2000 [1]), the
# local coupling rate $\mu(g)$ between two waveguides separated by edge-to-edge
# distance $g$ decays exponentially as $\mu(g) = \mu_0 \exp(-\gamma_{\text{evan}} g)$.
# Along the circular ring of radius $r = 5.0\;\mu\text{m}$, the local gap varies
# parabolically near $x = 0$:
#
# $$g(x) = g_0 + r - \sqrt{r^2 - x^2} \approx g_0 + \frac{x^2}{2r},$$
#
# so integrating $\mu(g(x))$ along $x$ yields the total coupling phase $\Theta(g_0)$
# and effective interaction length $L_{\text{eff}}$:
#
# $$\Theta_{\text{CMT}}(g_0) = \int_{-r}^{+r} \mu\bigl(g(x)\bigr)\,\mathrm{d}x \approx \mu_0\,\mathrm{e}^{-\gamma_{\text{evan}} g_0}\sqrt{\frac{2\pi r}{\gamma_{\text{evan}}}}, \qquad L_{\text{eff}} = \sqrt{\frac{2\pi r}{\gamma_{\text{evan}}}} = 1.954\;\mu\text{m},$$
#
# and $|\kappa(g_0)|^2 = \sin^2\Theta(g_0)$, $|t(g_0)|^2 = \cos^2\Theta(g_0)$.
# When $g_0 = 50\;\text{nm} < 1/\gamma_{\text{evan}} \approx 121\;\text{nm}$, multiple
# reflections of the normal electric field $E_y$ between the high-contrast silicon
# sidewalls create a **slot-field enhancement** inside the narrow oxide gap, which
# is captured by integrating the exact 5-layer coupled-slab supermode splitting
# $\mu_{\text{super}}(g) = \frac{1}{2}\bigl[\beta_{\text{even}}(g) - \beta_{\text{odd}}(g)\bigr]$.

# %% [markdown]
# ## 3. Parametric geometry & CPML waveguide termination
#
# When a curved dielectric waveguide is truncated inside the domain so that it
# enters a Perfectly Matched Layer (PML) at an oblique angle (such as $\theta \approx 49^\circ$
# at $y_{\max} = 3.825\;\mu\text{m}$), two numerical pathologies arise unless
# special precautions are taken:
#
# 1. **If the waveguide terminates at the PML interface (`extrude=False`)**, the
#    abrupt index step from $n_{\text{eff,slab}} = 2.848$ to $n_{\text{SiO}_2} = 1.444$
#    causes a $\sim 7\%$ Fresnel reflection back into the ring (`add` port $S_{41}$
#    and bus reflection $S_{11}$).
# 2. **If the waveguide enters the PML obliquely without an angled/bent mode
#    solver**, standard Cartesian mode ports cannot project the guided mode
#    orthogonally.
#
# In `build_ring_coupler` we follow the lower semicircle of the ring
# ($\phi \in [-\pi, 0]$ around $(0, w + g + r)$) up to $y = y_0$, where the ring
# tangent is strictly vertical ($\pm\hat{y}$ and completely decoupled from the bus
# waveguide by $> 5.5\;\mu\text{m}$), and **extrude straight vertical waveguide
# leads all the way through the $+y$ CPML** (`extrude_into_pml=True`). All four
# ports (`in`, `through` normal to $x$; `drop`, `add` normal to $y$) are then
# exact Cartesian eigenmode planes entering the CPML at $90^\circ$, reducing
# spurious backward ring leakage at $\lambda_0 = 1550\;\text{nm}$ from
# $|S_{41}|^2 = 1.33 \times 10^{-2}$ ($-18.8\;\text{dB}$ when truncated before the
# CPML) down to $|S_{41}|^2 = 1.02 \times 10^{-5}$ ($-49.9\;\text{dB}$ at $L_3 = (20, 8)$,
# a $1309\times$ / $31.2\;\text{dB}$ suppression) and $6.90 \times 10^{-7}$
# ($-61.6\;\text{dB}$) at the $R = (40, 16)$ reference ($|S_{41}|^2 = 1.89 \times 10^{-5}$
# or $-47.2\;\text{dB}$ at the `--quick` $L_2$ level), while we also sample
# the angled $\theta = 45^\circ$ radial cross-section at $(3.5355, 2.0145)\;\mu\text{m}$
# via `_helpers.sample_angled_drop_slice`.


# %%
def build_ring_coupler(gap=GAP_NOMINAL, n_arc=180, extrude_into_pml=True):
    """
    Build the 2D EIM waveguide-to-ring coupler (`[bus_slab, ring_slab]`), its
    four named `Port` specifications (`in`, `through`, `drop`, `add`), and the
    2D simulation domain `(xmin, xmax, ymin, ymax)` [m].
    """
    gap = float(gap)
    y0 = W_WG + gap + R_RING  # 5.55 um at gap = 50 nm
    dom_ymax = y0 + 0.55e-6
    y_top = (y0 + 1.50e-6) if extrude_into_pml else (dom_ymax - 0.02e-6)

    bus_verts = straight((-7.5e-6, 0.0), (7.5e-6, 0.0), width=W_WG)
    bus_slab = PolySlab(
        bus_verts,
        z_min=-0.5 * H_SI,
        z_max=0.5 * H_SI,
        material=n_to_eps(N_SI),
    )

    theta = np.linspace(-np.pi, 0.0, int(n_arc))
    r_in = R_RING - 0.5 * W_WG
    r_out = R_RING + 0.5 * W_WG
    c, s = np.cos(theta), np.sin(theta)
    outer_arc = np.column_stack([r_out * c, y0 + r_out * s])
    inner_arc = np.column_stack([r_in * c[::-1], y0 + r_in * s[::-1]])
    ring_verts = np.vstack(
        [
            [[-r_out, y_top]],
            outer_arc,
            [[r_out, y_top], [r_in, y_top]],
            inner_arc,
            [[-r_in, y_top]],
        ]
    )
    ring_slab = PolySlab(
        ring_verts,
        z_min=-0.5 * H_SI,
        z_max=0.5 * H_SI,
        material=n_to_eps(N_SI),
    )

    y_port = y0 + 0.25e-6
    ports = [
        Port(
            "in",
            center=(-X_PORT_BUS, 0.0, 0.0),
            size=(0.0, 1.8e-6, 0.8e-6),
            direction=+1,
        ),
        Port(
            "through",
            center=(+X_PORT_BUS, 0.0, 0.0),
            size=(0.0, 1.8e-6, 0.8e-6),
            direction=+1,
        ),
        Port(
            "drop",
            center=(+R_RING, y_port, 0.0),
            size=(1.8e-6, 0.0, 0.8e-6),
            direction=+1,
        ),
        Port(
            "add",
            center=(-R_RING, y_port, 0.0),
            size=(1.8e-6, 0.0, 0.8e-6),
            direction=+1,
        ),
    ]
    domain = (-5.85e-6, 5.85e-6, -0.85e-6, dom_ymax)
    return [bus_slab, ring_slab], ports, domain


# %% [markdown]
# ## 4. Subpixel permittivity smoothing & local mesh overrides
#
# ### Resolution: cells per wavelength and cells across the smallest feature
#
# Accurate time-domain electromagnetic modeling of high-index-contrast photonic
# couplers requires satisfying **two independent resolution conditions** at
# every study level (including `--quick`, parameter sweeps, and tests):
#
# 1. **Waveguide dispersion rule ($\text{cpw} \ge 10$)**: The background grid
#    must place at least $10\;\text{cells}$ per minimum wavelength inside the
#    highest-index material—here the 2D EIM silicon slab core
#    ($n_{\text{eff,slab}} = 2.8478$ at $\lambda_{\min} = 1500\;\text{nm}$,
#    giving $\Delta x_{\text{bg}} \le \lambda_{\min} / (n_{\text{eff,slab}}\,\text{cpw}) \approx 52.6\;\text{nm}$
#    at $10\;\text{cpw}$)—to keep numerical grid dispersion below $\sim 1\%$.
# 2. **Geometric feature rule ($N \ge 4\;\text{cells across the minimum feature}$)**:
#    The grid step across the smallest geometric feature must resolve that
#    feature with at least $4$ cells ($\Delta x_{\min} \le d_{\min} / N$). In our
#    bus-to-ring directional coupler, the smallest feature is the narrow oxide
#    coupling slot ($g_0 = 50\;\text{nm}$ baseline, swept from $50\;\text{nm}$ to
#    $250\;\text{nm}$). At a uniform $10\;\text{cpw}$ ($\Delta y \approx 52.5\;\text{nm}$),
#    only $0.95\;\text{cells}$ span the $50\;\text{nm}$ gap; even at uniform
#    $20\;\text{cpw}$ ($\Delta y \approx 26.3\;\text{nm}$), fewer than $2\;\text{cells}$
#    fit across the slot.
#
# Rather than over-refining the entire $11.7\;\mu\text{m} \times 6.95\;\mu\text{m}$
# domain uniformly, we pass a local **mesh-override box**
# `PICSimulation(..., mesh_overrides=[(lo, hi, (max_cell, max_cell, 1.0))])`
# around the coupling region:
#
# - Along $y$, the override box covers $g_0 \pm 0.30\;\mu\text{m}$, i.e.
#   $y \in [\frac{1}{2}w_{\text{wg}} - 0.30\;\mu\text{m},\; \frac{1}{2}w_{\text{wg}} + g_0 + 0.30\;\mu\text{m}]$.
# - Along $x$, it spans the interaction zone where the local bus-to-ring edge
#   separation $g(x) = g_0 + r - \sqrt{r^2 - x^2}$ is below $3 g_0$, i.e.
#   $|x| \le \sqrt{4 r g_0 - 4 g_0^2}$ ($\pm 0.995\;\mu\text{m}$ at $g_0 = 50\;\text{nm}$).
# - Inside this box, the in-plane step is capped at $\Delta x_{\min} = g_0 / N$
#   whenever $g_0 / N$ is finer than the background core cell size, and the
#   tensor-product mesh grades smoothly back to the background cell size outside
#   the box at an expansion ratio $\le 1.4$. Setting the third tuple component
#   $h_z = 1.0\;\text{m}$ prevents `Simulation._axis(2)` from subdividing the
#   single periodic 2D $z$ layer (`Nz = 1`).
#
# Under **staircasing** (rounding the dual-cell fill fraction $w \in [0, 1]$ to $0$
# or $1$), the curved ring boundary jumps discontinuously by $\Delta y$ steps
# along $x$. Wakis's **Kottke, Farjadpour & Johnson (2008)** [3] diagonal subpixel
# smoothing blends the inverse normal permittivity harmonically
# ($\varepsilon_{yy}^{-1} = w/\varepsilon_{\text{Si}} + (1-w)/\varepsilon_{\text{SiO}_2}$)
# and the tangential permittivity arithmetically across cut cells.
#
# ### Graded non-uniform Cartesian mesh & body-fitted hybrid unstructured ring mesh
#
# While a single `mesh_overrides` box around the coupling slot resolves the gap
# locally, it still uses a uniform $20\;\text{cpw}$ core step ($\Delta x \approx \Delta y \approx 26.3\;\text{nm}$)
# across the entire $11.7\;\mu\text{m} \times 6.95\;\mu\text{m}$ domain, including
# the $\text{SiO}_2$ cladding interior of the semicircle (`263,466` cells in
# `kottke_L3`). In `_helpers.py` we implement two mesh upgrades shown in
# [`04b_mesh_upgrade_comparison.png`](figures/04b_mesh_upgrade_comparison.png):
#
# 1. **Mirror-symmetric graded non-uniform Cartesian mesh (`build_ring_coupler_graded_axes` / `NonUniformRingCouplerPIC`, `task_13` `kottke_nonuniform`)**:
#    Using `wakis.templates.meshing.graded_axis`, we pin exact grid nodes at both
#    walls of the $50\;\text{nm}$ oxide slot ($y \in [250, 300]\;\text{nm}$) with
#    $8.00$ uniform cells of $\Delta y_{\min} = 6.25\;\text{nm}$, resolve the bus
#    core and bottom ring arc with $\Delta y \approx 13.5\text{--}16\;\text{nm}$,
#    use non-square cells along the propagation axis ($\Delta x_{\text{couple}} = 13.5\;\text{nm} > \Delta y_{\min} = 6.25\;\text{nm}$)
#    with exact $x \mapsto -x$ mirror symmetry (`x_axis == -x_axis[::-1]`), and
#    grade smoothly into the $\text{SiO}_2$ cladding ($\Delta_{\max} = 37.43\;\text{nm}$).
#    This cuts the cell count from $714 \times 369 \times 1 = 263,466$ (`kottke_L3`)
#    to $503 \times 329 \times 1 = 165,487$ (**$-37.2\%$ fewer cells**) and reduces
#    single-drive wall time from $1437.3\;\text{s}$ (`staircase_L3`) to $650.2\;\text{s}$
#    (**$-54.8\%$ wall time, $2.21\times$ speedup**) while achieving equal accuracy
#    at $\lambda_0 = 1550\;\text{nm}$: $|t(1550)|^2 = 0.763740$ ($0.060\%$ from
#    `kottke_L3` and $0.261\%$ from `kottke_reference` $R$) and
#    $|\kappa(1550)|^2 = 0.230838$ ($0.305\%$ from `kottke_L3` and $1.059\%$ from $R$,
#    with $|S_{41}(1550)|^2 = 9.69 \times 10^{-6}$ or $-50.1\;\text{dB}$).
# 2. **Body-fitted square+triangle hybrid unstructured mesh (`build_ring_coupler_hybrid_unstructured_mesh`)**:
#    Using `wakis.unstructured.subtri_circles_mesh` on the curved $r = 5.0\;\mu\text{m}$
#    ring boundary, quadtree background squares ($5,588$ quads) are subdivided into
#    body-fitted triangles ($4,212$ interface triangles, $9,800$ total cells) along
#    the circular waveguide rim so that curved dielectric interfaces are represented
#    without staircasing.
#
# ## 5. Study tasks & execution modes
#
# Each resolution level is specified by the pair $(\text{cpw}, N)$, where
# $\text{cpw}$ is the background `cells_per_wavelength` in the slab effective
# index and $N$ is the target number of cells across the coupling gap inside the
# override box ($\Delta x_{\min} = g / N$):
#
# - **Ladder levels**: $L_1 = (10, 4)$, $L_2 = (14, 6)$, $L_3 = (20, 8)$, and
#   full reference $R = (40, 16)$.
# - **`--full` mode (14 tasks, `task_0`..`task_13`, `n_pml = 10`, `energy_decay = 1e-4`)**:
#   Kottke and staircase at $L_1 = (10, 4)$, $L_2 = (14, 6)$, and $L_3 = (20, 8)$
#   (`kottke_L3` is the primary run collecting DFT fields, pulse movie, 3-port
#   reciprocity drives, and the companion unextruded-CPML audit), independent
#   `kottke_reference` at $R = (40, 16)$, one clearly labelled **under-resolved
#   warning pair** (`underresolved_kottke`, `underresolved_staircase` at
#   $10\;\text{cpw}$ with no mesh overrides, giving $0.95\;\text{cells}$ across
#   the $50\;\text{nm}$ gap), the 4 additional gap-sweep points
#   $g \in \{100, 150, 200, 250\}\;\text{nm}$ at $L_3 = (20, 8)$ ($\Delta x_{\min} = g / 8$),
#   and the upgraded graded non-uniform Cartesian mesh task `kottke_nonuniform` (`task_13`).
# - **`--quick` mode (12 tasks, `task_0`..`task_11`, `n_pml = 8`, `energy_decay = 3e-3`)**:
#   Kottke and staircase at $L_1 = (10, 4)$ and $L_2 = (14, 6)$ (`kottke_L2`
#   primary), `kottke_reference` at $L_3 = (20, 8)$, the under-resolved warning
#   pair at $(10, 0)$, the gap sweep $g \in \{100, 150, 200, 250\}\;\text{nm}$
#   at $L_2 = (14, 6)$ ($\Delta x_{\min} = g / 6$), and `kottke_nonuniform` (`task_11`).
#
# ### Measured `--full` resolution, performance & S-parameter table (`data/task_0`..`task_13`)
#
# Every non-demo task in `data/` satisfies $\text{cpw} \ge 10$ in the highest-index
# material ($n_{\text{eff,slab}} = 2.8478$) and $\text{cells\_across\_feature} \ge 4.00$
# across its coupling gap ($N_z = 1$):
#
# | Task ID | Name | Avg | Gap [nm] | $(\text{cpw}, N)$ | $\Delta x_{\text{bg}}$ [nm] | $\Delta x_{\min}$ [nm] | Cells across gap | Grid shape | $N_{\text{cells}}$ | $N_{\text{steps}}$ | Time [s] | Peak RSS [MB] | $\|t(1550)\|^2$ | $\|\kappa(1550)\|^2$ | $\|S_{11}(1550)\|^2$ | $\|S_{41}(1550)\|^2$ |
# |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
# | `0` | `kottke_L1` | `kottke` | $50$ | $(10, 4)$ | $52.61$ | $12.44$ | $4.00$ | $369 \times 197 \times 1$ | $72,693$ | $23,501$ | $204.4$ | $537.4$ | $0.757153$ | $0.237189$ | $1.06 \times 10^{-4}$ | $6.25 \times 10^{-5}$ |
# | `1` | `staircase_L1` | `staircase` | $50$ | $(10, 4)$ | $52.61$ | $12.44$ | $4.00$ | $369 \times 197 \times 1$ | $72,693$ | $29,001$ | $254.6$ | $537.6$ | $0.781097$ | $0.214059$ | $2.38 \times 10^{-4}$ | $4.93 \times 10^{-4}$ |
# | `2` | `kottke_L2` | `kottke` | $50$ | $(14, 6)$ | $37.60$ | $8.33$ | $6.00$ | $522 \times 272 \times 1$ | $141,984$ | $34,701$ | $532.8$ | $678.5$ | $0.764171$ | $0.230141$ | $7.89 \times 10^{-5}$ | $2.70 \times 10^{-5}$ |
# | `3` | `staircase_L2` | `staircase` | $50$ | $(14, 6)$ | $37.60$ | $8.33$ | $6.00$ | $522 \times 272 \times 1$ | $141,984$ | $37,401$ | $622.1$ | $679.7$ | $0.764518$ | $0.231421$ | $1.59 \times 10^{-4}$ | $2.31 \times 10^{-4}$ |
# | `4` | `kottke_L3` *(primary)* | `kottke` | $50$ | $(20, 8)$ | $26.30$ | $6.24$ | $8.00$ | $714 \times 369 \times 1$ | $263,466$ | $50,951$ | $7289.5^*$ | $1198.4$ | $0.764200$ | $0.230135$ | $3.03 \times 10^{-5}$ | $1.02 \times 10^{-5}$ |
# | `5` | `staircase_L3` | `staircase` | $50$ | $(20, 8)$ | $26.30$ | $6.24$ | $8.00$ | $714 \times 369 \times 1$ | $263,466$ | $46,701$ | $1437.3$ | $860.6$ | $0.772212$ | $0.222757$ | $1.20 \times 10^{-5}$ | $3.48 \times 10^{-5}$ |
# | `6` | `kottke_reference` ($R$) | `kottke` | $50$ | $(40, 16)$ | $13.17$ | $3.12$ | $16.00$ | $1400 \times 712 \times 1$ | $996,800$ | $91,551$ | $11317.6$ | $2081.6$ | $0.765742$ | $0.228419$ | $2.99 \times 10^{-6}$ | $6.90 \times 10^{-7}$ |
# | `7` | `underresolved_kottke` *(warning)* | `kottke` | $50$ | $(10, 0)$ | $52.65$ | $52.47$ | $0.95$ | $244 \times 153 \times 1$ | $37,332$ | $6,751$ | $30.9$ | $478.7$ | $0.760200$ | $0.235401$ | $2.53 \times 10^{-6}$ | $4.21 \times 10^{-6}$ |
# | `8` | `underresolved_staircase` *(warning)* | `staircase` | $50$ | $(10, 0)$ | $52.65$ | $52.47$ | $0.95$ | $244 \times 153 \times 1$ | $37,332$ | $9,001$ | $41.7$ | $478.9$ | $0.680790$ | $0.309800$ | $7.06 \times 10^{-4}$ | $1.04 \times 10^{-3}$ |
# | `9` | `gap_100nm` | `kottke` | $100$ | $(20, 8)$ | $26.27$ | $12.50$ | $8.00$ | $585 \times 318 \times 1$ | $186,030$ | $23,651$ | $504.7$ | $754.1$ | $0.908871$ | $0.088533$ | $8.67 \times 10^{-6}$ | $2.32 \times 10^{-6}$ |
# | `10` | `gap_150nm` | `kottke` | $150$ | $(20, 8)$ | $26.34$ | $18.75$ | $8.00$ | $519 \times 301 \times 1$ | $156,219$ | $16,501$ | $309.7$ | $703.8$ | $0.961135$ | $0.036717$ | $6.02 \times 10^{-7}$ | $2.93 \times 10^{-7}$ |
# | `11` | `gap_200nm` | `kottke` | $200$ | $(20, 8)$ | $26.32$ | $24.96$ | $8.00$ | $474 \times 293 \times 1$ | $138,882$ | $13,151$ | $215.2$ | $673.7$ | $0.981894$ | $0.015826$ | $8.03 \times 10^{-7}$ | $1.64 \times 10^{-8}$ |
# | `12` | `gap_250nm` | `kottke` | $250$ | $(20, 8)$ | $26.29$ | $26.29$ | $9.51^\dagger$ | $466 \times 293 \times 1$ | $136,538$ | $12,601$ | $219.6$ | $671.1$ | $0.990773$ | $0.006666$ | $3.15 \times 10^{-8}$ | $3.22 \times 10^{-9}$ |
# | `13` | `kottke_nonuniform` *(graded)* | `kottke` | $50$ | $(20, 8)$ | $37.43$ | $6.25$ | $8.00$ | $503 \times 329 \times 1$ | $165,487$ | $36,301$ | $650.2$ | $706.1$ | $0.763740$ | $0.230838$ | $6.64 \times 10^{-5}$ | $9.69 \times 10^{-6}$ |
#
# $^*$Task `4` (`kottke_L3`) executes 3 port drives (`in`, `through`, `drop` with
# on-the-fly DFT field accumulation and 48-frame movie recording) plus the
# companion unextruded-CPML audit run. $^\dagger$At $g = 250\;\text{nm}$ and
# $20\;\text{cpw}$, the background core cell size ($26.29\;\text{nm}$) is already
# finer than $g / 8 = 31.25\;\text{nm}$, yielding $9.51\;\text{cells}$ across the
# gap without an override.
#
# ### Quick-mode (`--quick`) reference table (`task_0`..`task_11`, `n_pml = 8`, `energy_decay = 3e-3`)
#
# For interactive local testing, `--quick` runs a coarser 12-task table stopping
# at $L_2 = (14, 6)$ with `kottke_reference` at $L_3 = (20, 8)$ and a looser
# ring-down threshold (`energy_decay = 3e-3`):
#
# | Quick Task ID | Name | Avg | Gap [nm] | $(\text{cpw}, N)$ | $\Delta x_{\text{bg}}$ [nm] | $\Delta x_{\min}$ [nm] | Cells across gap | $N_{\text{cells}}$ | $N_{\text{steps}}$ | Time [s] | $\|t(1550)\|^2$ | $\|\kappa(1550)\|^2$ |
# |---|---|---|---|---|---|---|---|---|---|---|---|---|
# | `0` | `kottke_L1` | `kottke` | $50$ | $(10, 4)$ | $52.61$ | $12.44$ | $4.00$ | $70,445$ | $22,051$ | $228.5$ | $0.755176$ | $0.222236$ |
# | `1` | `staircase_L1` | `staircase` | $50$ | $(10, 4)$ | $52.61$ | $12.44$ | $4.00$ | $70,445$ | $22,101$ | $225.1$ | $0.779411$ | $0.199770$ |
# | `2` | `kottke_L2` *(quick primary)* | `kottke` | $50$ | $(14, 6)$ | $37.60$ | $8.33$ | $6.00$ | $138,824$ | $36,401$ | $2032.4$ | $0.761504$ | $0.214254$ |
# | `3` | `staircase_L2` | `staircase` | $50$ | $(14, 6)$ | $37.60$ | $8.33$ | $6.00$ | $138,824$ | $32,751$ | $622.6$ | $0.762375$ | $0.216115$ |
# | `4` | `kottke_reference` | `kottke` | $50$ | $(20, 8)$ | $26.30$ | $6.24$ | $8.00$ | $259,150$ | $43,351$ | $1465.1$ | $0.761478$ | $0.215319$ |
# | `5` | `underresolved_kottke` *(warning)* | `kottke` | $50$ | $(10, 0)$ | $52.65$ | $52.47$ | $0.95$ | $35,760$ | $6,351$ | $38.0$ | $0.758389$ | $0.222793$ |
# | `6` | `underresolved_staircase` *(warning)* | `staircase` | $50$ | $(10, 0)$ | $52.65$ | $52.47$ | $0.95$ | $35,760$ | $6,501$ | $38.0$ | $0.679940$ | $0.301094$ |
# | `7` | `gap_100nm` | `kottke` | $100$ | $(14, 6)$ | $37.60$ | $16.67$ | $6.00$ | $96,444$ | $16,551$ | $225.7$ | $0.902221$ | $0.079064$ |
# | `8` | `gap_150nm` | `kottke` | $150$ | $(14, 6)$ | $37.47$ | $24.90$ | $6.00$ | $81,216$ | $11,501$ | $140.1$ | $0.949296$ | $0.032321$ |
# | `9` | `gap_200nm` | `kottke` | $200$ | $(14, 6)$ | $37.42$ | $33.21$ | $6.00$ | $72,030$ | $9,101$ | $101.0$ | $0.963721$ | $0.013240$ |
# | `10` | `gap_250nm` | `kottke` | $250$ | $(14, 6)$ | $37.62$ | $37.44$ | $6.68$ | $68,224$ | $8,351$ | $83.9$ | $0.972938$ | $0.005433$ |
# | `11` | `kottke_nonuniform` *(graded)* | `kottke` | $50$ | $(14, 6)$ | $52.10$ | $8.33$ | $6.00$ | $85,392$ | $24,201$ | $280.0$ | $0.761100$ | $0.214900$ |



# %%
def coupling_mesh_overrides(gap=GAP_NOMINAL, cells_per_feature=4, cpw=10):
    """
    Build the local mesh-override box around the bus-to-ring point-coupling slot.

    The override box spans ``gap +- 0.30 um`` in ``y`` (``[0.5*W_WG - 0.30 um,
    0.5*W_WG + gap + 0.30 um]``) and the ``x`` range where the local bus-to-ring
    edge distance ``g(x) = gap + R_RING - sqrt(R_RING^2 - x^2)`` is below
    ``3 * gap`` (``|x| <= sqrt(4 * R_RING * gap - 4 * gap^2)``). Inside the box,
    the in-plane cell size is capped at ``max_cell = gap / cells_per_feature``
    whenever that is finer than the background slab cell size
    ``WL_MIN / (n_eff_slab * cpw)``. The periodic 2D ``z`` axis is given a large
    cap (``1.0 m``) so ``Nz`` stays 1.
    """
    gap = float(gap)
    n_feat = int(cells_per_feature) if cells_per_feature is not None else 0
    if n_feat <= 0:
        return []
    neff_slab = slab_neff(N_SI, N_SIO2, H_SI, WL_REF, polarization="TE")
    h_bg_slab = WL_MIN / (neff_slab * float(cpw))
    max_cell = gap / float(n_feat)
    if max_cell >= h_bg_slab * (1.0 - 1e-9):
        return []
    x_half = float(np.sqrt(max(4.0 * R_RING * gap - 4.0 * gap**2, (0.5e-6) ** 2)))
    y_lo = 0.5 * W_WG - 0.30e-6
    y_hi = 0.5 * W_WG + gap + 0.30e-6
    return [
        ((-x_half, y_lo, -np.inf), (x_half, y_hi, np.inf), (max_cell, max_cell, 1.0))
    ]


def measure_mesh_resolution(x_nodes, y_nodes, gap=GAP_NOMINAL):
    """
    Measure background cell size, minimum cell size, total 2D cell count, and the
    actual number of cells across the minimum feature (the coupling gap at x = 0)
    from the 1D mesh node arrays ``x_nodes`` and ``y_nodes`` [m].
    """
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)
    gap = float(gap)
    dx = np.diff(x)
    dy = np.diff(y)
    y_gap_lo = 0.5 * W_WG
    y_gap_hi = 0.5 * W_WG + gap
    in_gap_y = (y[1:] > y_gap_lo + 1e-12) & (y[:-1] < y_gap_hi - 1e-12)
    max_cell_across_gap = float(np.max(dy[in_gap_y]))
    return {
        "min_feature_nm": float(gap * 1e9),
        "dx_background_nm": float(max(np.max(dx), np.max(dy)) * 1e9),
        "dx_min_nm": float(min(np.min(dx), np.min(dy)) * 1e9),
        "cells_across_feature": float(gap / max_cell_across_gap),
        "n_cells": int((len(x) - 1) * (len(y) - 1)),
        "grid_shape": [int(len(x) - 1), int(len(y) - 1), 1],
    }


def build_task_mesh_axes(task_cfg):
    """
    Build the 1D mesh node arrays ``(x, y, z)`` [m] for ``task_cfg`` without
    instantiating ``GridFIT3D`` or ``SolverFIT3D`` (used by preflight & tests).
    """
    gap_m = float(task_cfg["gap_nm"]) * 1e-9
    cpw = float(task_cfg["cpw"])
    n_feat = int(task_cfg.get("cells_per_feature", 0))
    structures, _, dom2d = build_ring_coupler(gap=gap_m, extrude_into_pml=True)
    if task_cfg.get("level") == "nonuniform":
        x, y, z, _ = build_ring_coupler_graded_axes(
            domain=dom2d,
            gap=gap_m,
            width=W_WG,
            radius=R_RING,
            cpw=cpw,
            cells_per_feature=n_feat,
            n_pml=int(task_cfg["n_pml"]),
            wl_min=WL_MIN,
            wl_ref=WL_REF,
            n_core=N_SI,
            n_clad=N_SIO2,
            thickness=H_SI,
        )
        return x, y, z
    overrides = coupling_mesh_overrides(gap=gap_m, cells_per_feature=n_feat, cpw=cpw)
    neff_slab = slab_neff(N_SI, N_SIO2, H_SI, WL_REF, polarization="TE")
    f_min, f_max = c_light / WL_MAX, c_light / WL_MIN
    h_core = WL_MIN / (neff_slab * cpw)
    xmin, xmax, ymin, ymax = dom2d[:4]
    domain = (xmin, xmax, ymin, ymax, -0.5 * h_core, 0.5 * h_core)
    margin = 20.0 * WL_MAX
    bg_box = Box(
        center=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.0),
        size=(xmax - xmin + margin, ymax - ymin + margin, 4.0 * h_core),
        material=n_to_eps(N_SIO2),
    )
    for s in structures:
        s.material = n_to_eps(neff_slab)
    sim = Simulation(
        f_min,
        f_max,
        structures=[bg_box] + structures,
        boundaries=(["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"]),
        domain=domain,
        cells_per_wavelength=cpw,
        n_pml=int(task_cfg["n_pml"]),
        mesh_overrides=overrides,
        averaging=task_cfg["averaging"],
    )
    return tuple(sim._axis(a) for a in range(3))


def get_task_table(quick=False):
    """
    Return the study task table (resolution ladder L1/L2[/L3], independent
    Kottke reference, under-resolved 10-cpw demonstration pair, and gap sweep).

    Resolution ladder levels (cpw, N):
    - L1 = (10, 4), L2 = (14, 6), L3 = (20, 8), R = (40, 16)
    - ``--quick`` (11 tasks): Kottke & staircase at L1 and L2, Kottke reference at L3,
      under-resolved demonstration pair at (10, 0), and gap sweep (100..250 nm) at L2.
    - ``--full`` (13 tasks): Kottke & staircase at L1, L2, L3, Kottke reference at R,
      under-resolved demonstration pair at (10, 0), and gap sweep (100..250 nm) at L3.
    """
    ladder = (
        (("L1", 10, 4), ("L2", 14, 6))
        if quick
        else (("L1", 10, 4), ("L2", 14, 6), ("L3", 20, 8))
    )
    primary_lvl = ladder[-1][0]
    tasks = []
    for lvl, cpw, n_feat in ladder:
        for avg in ("kottke", "staircase"):
            idx = len(tasks)
            is_primary = lvl == primary_lvl and avg == "kottke"
            tasks.append(
                {
                    "index": idx,
                    "name": f"{avg}_{lvl}",
                    "level": lvl,
                    "averaging": avg,
                    "gap_nm": 50,
                    "cpw": cpw,
                    "cells_per_feature": n_feat,
                    "is_demo": False,
                    "n_pml": 8 if quick else 10,
                    "energy_decay": 3e-3 if quick else 1e-4,
                    "collect_fields": is_primary,
                    "record_movie": is_primary,
                    "movie_frames": 32 if quick else 48,
                    "description": (
                        f"2D EIM ring coupler ({avg}, {lvl} grid, gap=50nm, cpw={cpw}, N={n_feat}"
                        + (
                            ", DFT fields + movie + reciprocity + PML audit)"
                            if is_primary
                            else ")"
                        )
                    ),
                }
            )

    ref_cpw, ref_n = (20, 8) if quick else (40, 16)
    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_reference",
            "level": "reference",
            "averaging": "kottke",
            "gap_nm": 50,
            "cpw": ref_cpw,
            "cells_per_feature": ref_n,
            "is_demo": False,
            "n_pml": 8 if quick else 10,
            "energy_decay": 3e-3 if quick else 1e-4,
            "collect_fields": False,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"2D EIM ring coupler (kottke, reference grid, gap=50nm, cpw={ref_cpw}, N={ref_n})"
            ),
        }
    )

    for avg in ("kottke", "staircase"):
        tasks.append(
            {
                "index": len(tasks),
                "name": f"underresolved_{avg}",
                "level": "underresolved",
                "averaging": avg,
                "gap_nm": 50,
                "cpw": 10,
                "cells_per_feature": 0,
                "is_demo": True,
                "n_pml": 8 if quick else 10,
                "energy_decay": 3e-3 if quick else 1e-4,
                "collect_fields": False,
                "record_movie": False,
                "movie_frames": 0,
                "description": (
                    f"2D EIM ring coupler under-resolved warning demo ({avg}, gap=50nm, cpw=10, no override)"
                ),
            }
        )

    sweep_cpw, sweep_n = (14, 6) if quick else (20, 8)
    for gap_nm in GAPS_SWEEP_NM[1:]:
        tasks.append(
            {
                "index": len(tasks),
                "name": f"gap_{gap_nm}nm",
                "level": "gap_sweep",
                "averaging": "kottke",
                "gap_nm": int(gap_nm),
                "cpw": sweep_cpw,
                "cells_per_feature": sweep_n,
                "is_demo": False,
                "n_pml": 8 if quick else 10,
                "energy_decay": 3e-3 if quick else 1e-4,
                "collect_fields": False,
                "record_movie": False,
                "movie_frames": 0,
                "description": (
                    f"2D EIM ring coupler gap sweep (kottke, gap={gap_nm}nm, cpw={sweep_cpw}, N={sweep_n})"
                ),
            }
        )

    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_nonuniform",
            "level": "nonuniform",
            "averaging": "kottke",
            "gap_nm": 50,
            "cpw": sweep_cpw,
            "cells_per_feature": sweep_n,
            "is_demo": False,
            "n_pml": 8 if quick else 10,
            "energy_decay": 3e-3 if quick else 1e-4,
            "collect_fields": False,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"2D EIM ring coupler graded non-uniform Cartesian mesh (kottke, gap=50nm, cpw={sweep_cpw}, N={sweep_n})"
            ),
        }
    )
    return tasks


def _extract_zoom_permittivity(solver, xlim_m=(-1.4e-6, 1.4e-6), ylim_m=(-0.40e-6, 1.15e-6)):
    """Extract a compact zoomed slice of ``eps_{r,y}`` and mesh nodes around the coupling slot."""
    g = solver.grid
    iz = solver.Nz // 2
    ieps = solver.ieps[:, :, iz, "y"]
    if hasattr(ieps, "get"):
        ieps = ieps.get()
    eps_r = np.where(ieps > 0, 1.0 / (np.where(ieps > 0, ieps, 1.0) * epsilon_0), 1.0)
    ix_lo = max(0, int(np.searchsorted(g.x, xlim_m[0])) - 1)
    ix_hi = min(g.Nx, int(np.searchsorted(g.x, xlim_m[1])) + 1)
    iy_lo = max(0, int(np.searchsorted(g.y, ylim_m[0])) - 1)
    iy_hi = min(g.Ny, int(np.searchsorted(g.y, ylim_m[1])) + 1)
    return (
        g.x[ix_lo : ix_hi + 1].astype(np.float32),
        g.y[iy_lo : iy_hi + 1].astype(np.float32),
        eps_r[ix_lo:ix_hi, iy_lo:iy_hi].astype(np.float32),
    )


def run_single_task(task_cfg, out_dir):
    """
    Execute one 2D EIM waveguide-to-ring coupler study task and write
    ``task_<I>.npz`` and ``task_<I>.json`` into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task_cfg["index"]
    t0 = time.perf_counter()

    gap_m = float(task_cfg["gap_nm"]) * 1e-9
    cpw = float(task_cfg["cpw"])
    n_feat = int(task_cfg.get("cells_per_feature", 0))
    structures, ports, domain = build_ring_coupler(gap=gap_m, extrude_into_pml=True)
    overrides = coupling_mesh_overrides(gap=gap_m, cells_per_feature=n_feat, cpw=cpw)
    use_gpu = os.environ.get("WAKIS_USE_GPU", "0") == "1"
    if task_cfg.get("level") == "nonuniform":
        pic = NonUniformRingCouplerPIC(
            wavelength_min=WL_MIN,
            wavelength_max=WL_MAX,
            n_freqs=N_FREQS,
            structures=structures,
            ports=ports,
            n_core=N_SI,
            n_clad=N_SIO2,
            wavelength_ref=WL_REF,
            two_d=True,
            averaging=task_cfg["averaging"],
            cells_per_wavelength=cpw,
            domain=domain,
            n_pml=task_cfg["n_pml"],
            field_wavelengths=(WL_REF,),
            record_movie=task_cfg["record_movie"],
            movie_frames=task_cfg["movie_frames"],
            use_gpu=use_gpu,
            gap=gap_m,
            width=W_WG,
            radius=R_RING,
            cells_per_feature=n_feat,
        )
    else:
        pic = PICSimulation(
            wavelength_min=WL_MIN,
            wavelength_max=WL_MAX,
            n_freqs=N_FREQS,
            structures=structures,
            ports=ports,
            n_core=N_SI,
            n_clad=N_SIO2,
            wavelength_ref=WL_REF,
            two_d=True,
            averaging=task_cfg["averaging"],
            cells_per_wavelength=cpw,
            domain=domain,
            n_pml=task_cfg["n_pml"],
            mesh_overrides=overrides,
            field_wavelengths=(WL_REF,),
            record_movie=task_cfg["record_movie"],
            movie_frames=task_cfg["movie_frames"],
            use_gpu=use_gpu,
        )

    res_metrics = measure_mesh_resolution(pic.grid.x, pic.grid.y, gap=gap_m)
    zoom_x, zoom_y, zoom_eps_y = _extract_zoom_permittivity(pic.solver)

    drive_ports = ["in", "through", "drop"] if task_cfg["collect_fields"] else ["in"]
    s_mat = pic.run_s_parameters(
        drive=drive_ports,
        energy_decay=task_cfg["energy_decay"],
        collect_fields=task_cfg["collect_fields"],
    )
    n_steps = int(pic.solver.Nt)

    payload = {
        "task_index": np.int32(idx),
        "name": np.array(task_cfg["name"]),
        "level": np.array(task_cfg["level"]),
        "averaging": np.array(task_cfg["averaging"]),
        "gap_nm": np.float32(task_cfg["gap_nm"]),
        "cpw": np.float32(task_cfg["cpw"]),
        "cells_per_feature": np.int32(n_feat),
        "min_feature_nm": np.float64(res_metrics["min_feature_nm"]),
        "dx_background_nm": np.float64(res_metrics["dx_background_nm"]),
        "dx_min_nm": np.float64(res_metrics["dx_min_nm"]),
        "cells_across_feature": np.float64(res_metrics["cells_across_feature"]),
        "n_cells": np.int64(res_metrics["n_cells"]),
        "n_steps": np.int64(n_steps),
        "n_eff_slab": np.float64(pic.n_eff_slab),
        "frequencies": pic.frequencies.astype(np.float64),
        "wavelengths": pic.wavelengths.astype(np.float64),
        "S11": s_mat[("in", "in")].astype(np.complex128),
        "S21": s_mat[("through", "in")].astype(np.complex128),
        "S31": s_mat[("drop", "in")].astype(np.complex128),
        "S41": s_mat[("add", "in")].astype(np.complex128),
        "zoom_x_nodes": zoom_x,
        "zoom_y_nodes": zoom_y,
        "zoom_eps_y": zoom_eps_y,
    }

    if task_cfg["collect_fields"]:
        payload["S12"] = s_mat[("in", "through")].astype(np.complex128)
        payload["S22"] = s_mat[("through", "through")].astype(np.complex128)
        payload["S42"] = s_mat[("add", "through")].astype(np.complex128)
        payload["S13"] = s_mat[("in", "drop")].astype(np.complex128)

        # Solve port modes at 1550 nm on 'in' (x-normal, Ey(y)) and 'drop' (y-normal, Ex(x))
        modes_in = pic.solve_port_modes("in", frequencies=c_light / WL_REF, n_modes=1)
        modes_drop = pic.solve_port_modes(
            "drop", frequencies=c_light / WL_REF, n_modes=1
        )
        y_in = modes_in[0].coords["Eb"][0]
        ey_in = modes_in[0].e["b"][:, 0]
        m_in = np.abs(y_in) <= 0.9e-6

        x_drop = modes_drop[0].coords["Ec"][1]
        ex_drop = modes_drop[0].e["c"][0, :]
        m_drop = np.abs(x_drop - R_RING) <= 0.9e-6

        payload["port_y_in"] = y_in[m_in].astype(np.float32)
        payload["port_ey_in"] = ey_in[m_in].astype(np.complex64)
        payload["port_neff_in"] = np.float64(np.real(modes_in[0].neff))
        payload["port_x_drop"] = x_drop[m_drop].astype(np.float32)
        payload["port_ex_drop"] = ex_drop[m_drop].astype(np.complex64)
        payload["port_neff_drop"] = np.float64(np.real(modes_drop[0].neff))

        wl_disp = np.linspace(WL_MIN, WL_MAX, 11)
        f_disp = c_light / wl_disp
        disp_modes = pic.solve_port_modes("in", frequencies=f_disp, n_modes=1)
        neff_disp = np.array([float(np.real(m[0].neff)) for m in disp_modes])
        dneff_dl = np.gradient(neff_disp, wl_disp)
        ng_disp = neff_disp - wl_disp * dneff_dl
        payload["disp_wavelengths"] = wl_disp.astype(np.float64)
        payload["disp_neff"] = neff_disp.astype(np.float64)
        payload["disp_ng"] = ng_disp.astype(np.float64)

        if WL_REF in pic.dft_fields:
            fmap = pic.dft_fields[WL_REF]
            payload["dft_x"] = fmap["x"]
            payload["dft_y"] = fmap["y"]
            payload["dft_Ex"] = fmap["Ex"]
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

        # Companion unextruded-CPML run to quantify PML waveguide truncation reflection
        structs_noext, ports_noext, dom_noext = build_ring_coupler(
            gap=gap_m, extrude_into_pml=False
        )
        pic_noext = PICSimulation(
            wavelength_min=WL_MIN,
            wavelength_max=WL_MAX,
            n_freqs=N_FREQS,
            structures=structs_noext,
            ports=ports_noext,
            n_core=N_SI,
            n_clad=N_SIO2,
            wavelength_ref=WL_REF,
            two_d=True,
            averaging=task_cfg["averaging"],
            cells_per_wavelength=cpw,
            domain=dom_noext,
            n_pml=task_cfg["n_pml"],
            mesh_overrides=overrides,
            use_gpu=use_gpu,
        )
        s_noext = pic_noext.run_s_parameters(
            drive=["in"],
            energy_decay=task_cfg["energy_decay"],
            collect_fields=False,
        )
        payload["noext_S11"] = s_noext[("in", "in")].astype(np.complex128)
        payload["noext_S21"] = s_noext[("through", "in")].astype(np.complex128)
        payload["noext_S31"] = s_noext[("drop", "in")].astype(np.complex128)
        payload["noext_S41"] = s_noext[("add", "in")].astype(np.complex128)

    elapsed = time.perf_counter() - t0
    payload["elapsed_s"] = np.float64(elapsed)

    npz_path = out_dir / f"task_{idx}.npz"
    np.savez_compressed(npz_path, **payload)

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    meta = {
        "task_index": idx,
        "name": task_cfg["name"],
        "level": task_cfg["level"],
        "averaging": task_cfg["averaging"],
        "gap_nm": task_cfg["gap_nm"],
        "cpw": task_cfg["cpw"],
        "cells_per_feature": n_feat,
        "min_feature_nm": round(res_metrics["min_feature_nm"], 3),
        "dx_background_nm": round(res_metrics["dx_background_nm"], 3),
        "dx_min_nm": round(res_metrics["dx_min_nm"], 3),
        "cells_across_feature": round(res_metrics["cells_across_feature"], 3),
        "grid_shape": [int(pic.grid.Nx), int(pic.grid.Ny), int(pic.grid.Nz)],
        "n_cells": int(res_metrics["n_cells"]),
        "n_steps": int(n_steps),
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
# When invoked with `--plot` (or at the end of `--quick` / `--full`),
# `generate_all_figures` reads `data/task_*.npz` and renders the setup and
# physics validation figures to `figures/`:
#
# 0. `00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png` & `00_setup_table.md` — Labelled simulation setup schematic (materials, CPML boundaries, mode source, port monitors, launched & coupled mode profiles) and mesh + broadband excitation view.
# 1. `01_ring_coupler_layout.png` — Annotated top-view layout of the waveguide-to-ring coupler and ports.
# 2. `02_eim_vertical_stack.png` — Vertical $220\;\text{nm}$ SOI stack, 1D $\text{TE}_0$ slab mode profile, and $n_{\text{eff,slab}}(\lambda)$.
# 3. `03_cmt_mode_overlap_and_gap_profile.png` — Isolated 2D EIM mode overlap across the $50\;\text{nm}$ slot and local coupling profile $\mu(g(x))$.
# 4. `04_permittivity_mesh_zoom.png` & `04b_mesh_upgrade_comparison.png` — Zoomed in-plane permittivity map with Yee mesh lines on the $50\;\text{nm}$ bus–ring slot (under-resolved staircase vs $L_1$ Kottke vs Reference Kottke) and mesh upgrade comparison (baseline `kottke_L3` vs graded non-uniform Cartesian `kottke_nonuniform` vs body-fitted `wakis.unstructured` hybrid ring mesh).
# 5. `05_port_mode_profiles.png` — Bus (`in`), ring (`drop`), and $\theta = 45^\circ$ bent-waveguide radial mode profiles.
# 6. `06_port_neff_dispersion.png` — Channel waveguide effective index $n_{\text{eff}}(\lambda)$ and group index $n_g(\lambda)$.
# 7. `07_field_intensity_1550.png` — Steady-state logarithmic intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $1550\;\text{nm}$ with `overlay_geometry`.
# 8. `08_field_real_ey_1550.png` — Steady-state wave-front phase map $\mathrm{Re}(E_y)$ at $1550\;\text{nm}$ with `overlay_geometry`.
# 9. `09_pulse_propagation.gif` — Time-domain pulse propagation movie through the waveguide-to-ring coupler with `overlay_geometry` on every frame.
# 10. `10_sparameter_spectra_linear_and_db.png` — Broadband through ($|t|^2$) and drop ($|\kappa|^2$) spectra in linear and dB scales vs 5-layer supermode CMT.
# 11. `11_coupling_vs_gap_cmt.png` — Coupling $|\kappa|^2, |t|^2$ vs gap ($50\text{--}250\;\text{nm}$) and exponential decay fit vs analytical $\gamma_{\text{evan}}$.
# 12. `12_gap_wavelength_spectra.png` — Broadband $|\kappa(\lambda, g)|^2$ and $|t(\lambda, g)|^2$ across all 5 gaps ($50\text{--}250\;\text{nm}$).
# 13. `13_passivity_reciprocity_and_pml_audit.png` — Multi-port passivity, Lorentz reciprocity ($|S_{21} - S_{12}|, |S_{31} - S_{13}|$), and CPML waveguide extrusion audit.
# 14. `14_kottke_vs_staircase_convergence.png` — Subpixel convergence of Kottke, graded non-uniform Kottke, and staircasing against the independent Kottke reference run.
# 15. `15_ring_resonator_synthesis_and_time_trace.png` — Synthesized all-pass microring resonator transmission spectrum ($\text{FSR}, \mathcal{F}, Q_L$) and port time trace.


# %%
def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Read ``data_dir/task_*.npz`` and render all setup and study figures into ``fig_dir``."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()

    tasks_data = {}
    for p in sorted(data_dir.glob("task_*.npz")):
        d = dict(np.load(p, allow_pickle=False))
        name = str(d["name"])
        tasks_data[name] = d
    for jp in sorted(data_dir.glob("task_*.json")):
        meta_j = json.loads(jp.read_text())
        t_name = str(meta_j.get("name", ""))
        if t_name in tasks_data and "elapsed_s" in meta_j:
            tasks_data[t_name]["elapsed_s"] = float(meta_j["elapsed_s"])

    primary = None
    for cand in ("kottke_L3", "kottke_L2", "kottke_L1", "kottke_fine"):
        if cand in tasks_data and "dft_E2" in tasks_data[cand]:
            primary = tasks_data[cand]
            break
    if primary is None:
        for cand in ("kottke_L3", "kottke_L2", "kottke_L1", "kottke_fine"):
            if cand in tasks_data:
                primary = tasks_data[cand]
                break
    if primary is None:
        raise FileNotFoundError(
            f"Missing primary Kottke task in {data_dir}. Run --quick or --full first."
        )
    structures, ports, domain = build_ring_coupler(gap=GAP_NOMINAL, extrude_into_pml=True)
    y0_um = (W_WG + GAP_NOMINAL + R_RING) * 1e6

    # 0. Setup figures (00a geometry/materials/sources/monitors/boundaries,
    #    00b mesh + excitation pulse/spectrum) and Markdown setup table
    ov_schem = coupling_mesh_overrides(gap=GAP_NOMINAL, cells_per_feature=8, cpw=20)
    pic_schem = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=20,
        domain=domain,
        n_pml=10,
        mesh_overrides=ov_schem,
    )
    save_ring_coupler_setup_figures(
        pic_schem,
        primary,
        fig_dir,
        wavelength=WL_REF,
        min_feature=GAP_NOMINAL,
    )

    # 1. Top-view layout with dimensional and port annotations
    x_45 = R_RING * np.sin(THETA_DROP_REF) * 1e6
    y_45 = y0_um - R_RING * np.cos(THETA_DROP_REF) * 1e6
    annotations = [
        {
            "text": "Bus WG (w = 500 nm)\ny = 0 µm",
            "xy": (-2.6, 0.25),
            "xytext": (-2.8, 1.35),
            "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
        },
        {
            "text": "Coupling slot\ng = 50 nm (x = 0)",
            "xy": (0.0, 0.275),
            "xytext": (0.0, 1.55),
            "arrowprops": dict(arrowstyle="->", color="#d95f02", lw=1.2),
        },
        {
            "text": "Curved ring (r = 5.0 µm)\nrin = 4.75 µm, rout = 5.25 µm",
            "xy": (-3.5355, y_45),
            "xytext": (-2.2, 3.65),
            "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
        },
        {
            "text": "θ = 45° radial cut\n(3.54, 2.01) µm",
            "xy": (x_45, y_45),
            "xytext": (1.85, 3.15),
            "arrowprops": dict(arrowstyle="->", color="#1b9e77", lw=1.2),
        },
    ]
    plot_layout(
        structures,
        ports=ports,
        domain=domain,
        annotations=annotations,
        title="SOI waveguide-to-ring directional point coupler layout (2D EIM domain)",
        out_path=fig_dir / "01_ring_coupler_layout.png",
    )

    # 2. Effective-Index Method (EIM) vertical stack, TE0 slab mode, and dispersion
    plot_eim_stack(
        n_core=N_SI,
        n_clad=N_SIO2,
        thickness=H_SI,
        wavelength_ref=WL_REF,
        wavelengths=np.linspace(WL_MIN, WL_MAX, 51),
        out_path=fig_dir / "02_eim_vertical_stack.png",
    )

    # 3. Coupled-mode theory (CMT) mode overlap across the 50 nm slot & local coupling rate mu(x)
    neff_slab = float(primary["n_eff_slab"])
    iso = isolated_eim_channel_mode(
        neff_slab, N_SIO2, width=W_WG, wavelength=WL_REF, y_span=3.6e-6, n_pts=3601
    )
    cmt_50 = cmt_coupling_vs_gap([GAP_NOMINAL], n_core_2d=neff_slab)
    y_um = iso["y"] * 1e6
    ey_bus = iso["Ey"] / np.max(np.abs(iso["Ey"]))
    hz_bus = iso["Hz"] / np.max(np.abs(iso["Hz"]))
    y_shift_m = W_WG + GAP_NOMINAL
    ey_ring = np.interp(iso["y"] - y_shift_m, iso["y"], ey_bus, left=0.0, right=0.0)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 4.0), dpi=150)
    ax0.axvspan(-0.25, 0.25, color="#2b5c8f", alpha=0.18, label="Bus WG core")
    ax0.axvspan(0.30, 0.80, color="#1b9e77", alpha=0.18, label="Ring WG core (x = 0)")
    ax0.axvspan(0.25, 0.30, color="#d95f02", alpha=0.30, label="50 nm oxide slot")
    ax0.plot(y_um, ey_bus, color="#2b5c8f", lw=2.0, label=r"Bus mode $E_{y,1}(y)$")
    ax0.plot(
        y_um,
        hz_bus,
        color="#2b5c8f",
        ls=":",
        lw=1.4,
        label=r"Bus continuous $H_{z,1}(y)$",
    )
    ax0.plot(
        y_um,
        ey_ring,
        color="#1b9e77",
        ls="--",
        lw=2.0,
        label=r"Ring mode $E_{y,2}(y - w - g_0)$",
    )
    ax0.set(
        xlim=(-0.9, 1.45),
        ylim=(-0.05, 1.08),
        xlabel="Transverse coordinate y at x = 0 [µm]",
        ylabel="Normalized modal amplitude",
        title=f"(a) EIM mode overlap across 50 nm slot ($\\gamma_{{\\mathrm{{evan}}}}$ = {iso['gamma_evan']*1e-6:.2f} µm⁻¹)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=7.8)

    xs_um = np.linspace(-3.5, 3.5, 301)
    xs_m = xs_um * 1e-6
    gx_nm = (GAP_NOMINAL + R_RING - np.sqrt(R_RING**2 - xs_m**2)) * 1e9
    mu_x = (
        cmt_50["mu_0_overlap"]
        * np.exp(-iso["gamma_evan"] * (gx_nm * 1e-9))
        * 1e-6
    )
    ax1.plot(
        xs_um,
        mu_x,
        color="#1b9e77",
        lw=2.2,
        label=r"Local coupling $\mu(g(x))$ [µm$^{-1}$]",
    )
    ax1.fill_between(xs_um, 0.0, mu_x, color="#1b9e77", alpha=0.16)
    L_eff_um = cmt_50["L_eff"] * 1e6
    ax1.axvline(-0.5 * L_eff_um, color="#d95f02", ls="--", lw=1.3)
    ax1.axvline(
        +0.5 * L_eff_um,
        color="#d95f02",
        ls="--",
        lw=1.3,
        label=f"Gaussian $L_{{\\mathrm{{eff}}}} = \\sqrt{{2\\pi r/\\gamma}}$ = {L_eff_um:.2f} µm",
    )
    ax1_r = ax1.twinx()
    ax1_r._is_inset = True
    ax1_r.plot(
        xs_um,
        gx_nm,
        color="#2b5c8f",
        ls="-.",
        lw=1.7,
        label=r"Local gap $g(x)$ [nm]",
    )
    ax1_r.set_ylabel("Local edge-to-edge gap g(x) [nm]", color="#2b5c8f")
    ax1.set(
        xlabel="Axial position x along bus waveguide [µm]",
        ylabel=r"Local coupling rate $\mu(x)$ [µm$^{-1}$]",
        title="(b) Curved-ring gap profile g(x) & coupling window",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper center", fontsize=7.8)
    figstyle.save(
        fig,
        fig_dir / "03_cmt_mode_overlap_and_gap_profile.png",
        close=True,
        auto_legend_below=False,
    )

    # 4. Permittivity & mesh lines zoomed on the 50 nm coupling slot (<= 3 panels):
    #    under-resolved staircase (no override), L1 Kottke, and Reference Kottke
    def _get_zoom_slice(t_name, fallback_cpw, fallback_n, fallback_avg):
        d_t = tasks_data.get(t_name)
        if d_t is not None and "zoom_eps_y" in d_t:
            return (
                d_t["zoom_x_nodes"],
                d_t["zoom_y_nodes"],
                d_t["zoom_eps_y"],
                float(d_t.get("cpw", fallback_cpw)),
                int(d_t.get("cells_per_feature", fallback_n)),
                float(d_t.get("cells_across_feature", max(fallback_n, 1))),
                float(d_t.get("dx_min_nm", 50.0 / max(fallback_n, 1))),
            )
        ov_fb = coupling_mesh_overrides(
            gap=GAP_NOMINAL, cells_per_feature=fallback_n, cpw=fallback_cpw
        )
        pic_fb = PICSimulation(
            WL_MIN,
            WL_MAX,
            structures=structures,
            ports=ports,
            n_core=N_SI,
            n_clad=N_SIO2,
            two_d=True,
            averaging=fallback_avg,
            cells_per_wavelength=fallback_cpw,
            domain=domain,
            n_pml=8,
            mesh_overrides=ov_fb,
        )
        zx, zy, ze = _extract_zoom_permittivity(pic_fb.solver)
        rm = measure_mesh_resolution(pic_fb.grid.x, pic_fb.grid.y, gap=GAP_NOMINAL)
        return (
            zx,
            zy,
            ze,
            float(fallback_cpw),
            int(fallback_n),
            float(rm["cells_across_feature"]),
            float(rm["dx_min_nm"]),
        )

    panel_specs = [
        ("underresolved_staircase", 10, 0, "staircase", "(a) Under-resolved staircase (no override)"),
        ("kottke_L1", 10, 4, "kottke", "(b) L1 Kottke subpixel (override box)"),
        ("kottke_reference", 40, 16, "kottke", "(c) Reference Kottke (override box)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), dpi=150, sharex=True, sharey=True)
    xlim_z = (-0.60, 0.60)
    ylim_z = (0.10, 0.50)
    x_curve_um = np.linspace(xlim_z[0], xlim_z[1], 201)
    r_out_um = (R_RING + 0.5 * W_WG) * 1e6
    y_ring_bot_um = y0_um - np.sqrt(np.maximum(r_out_um**2 - x_curve_um**2, 0.0))

    for ax, (t_name, fb_cpw, fb_n, fb_avg, ttl_prefix) in zip(axes.ravel(), panel_specs):
        zx, zy, ze, cpw_v, n_v, caf_v, dxmin_v = _get_zoom_slice(
            t_name, fb_cpw, fb_n, fb_avg
        )
        x_um, y_um = zx * 1e6, zy * 1e6
        pcm = ax.pcolormesh(
            x_um, y_um, ze.T, cmap="viridis", shading="flat", vmin=2.0, vmax=8.2
        )
        ax.set_xlim(*xlim_z)
        ax.set_ylim(*ylim_z)
        xv = x_um[(x_um >= xlim_z[0]) & (x_um <= xlim_z[1])]
        yv = y_um[(y_um >= ylim_z[0]) & (y_um <= ylim_z[1])]
        vlines = [[(x, ylim_z[0]), (x, ylim_z[1])] for x in xv]
        hlines = [[(xlim_z[0], y), (xlim_z[1], y)] for y in yv]
        ax.add_collection(
            LineCollection(vlines + hlines, colors="white", linewidths=0.45, alpha=0.65)
        )
        # Exact geometric edges of the 50 nm coupling slot
        ax.axhline(0.25, color="#ff7f00", ls="--", lw=1.3, label="Bus top edge (y = 250 nm)")
        ax.plot(
            x_curve_um,
            y_ring_bot_um,
            color="#ff7f00",
            ls="--",
            lw=1.3,
            label="Ring bottom edge",
        )
        # Annotate 50 nm feature width at x = 0
        ax.annotate(
            "",
            xy=(0.0, 0.30),
            xytext=(0.0, 0.25),
            arrowprops=dict(arrowstyle="<->", color="#ffed6f", lw=1.6),
        )
        ax.text(
            0.03,
            0.275,
            f"g₀ = 50 nm\n({caf_v:.2f} cells across gap,\nΔy_min = {dxmin_v:.2f} nm)",
            color="white",
            fontsize=7.6,
            fontweight="bold",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="#ffed6f", alpha=0.72),
        )
        ax.set_title(
            f"{ttl_prefix}\ncpw = {int(cpw_v)}, N = {n_v} → {caf_v:.2f} cells/gap (Δx_min = {dxmin_v:.2f} nm)",
            fontsize=8.8,
        )
        ax.set_xlabel("x [µm]")
        cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
        cb.ax._is_inset = True
        cb.set_label(r"$\varepsilon_{r,y}$")

    axes[0].set_ylabel("y [µm]")
    fig.suptitle(
        "Zoomed Yee mesh & subpixel permittivity across the 50 nm bus–ring coupling slot",
        fontsize=10.8,
        y=1.02,
    )
    figstyle.save(
        fig,
        fig_dir / "04_permittivity_mesh_zoom.png",
        close=True,
        auto_legend_below=False,
    )

    # 4b. Mesh upgrade comparison: (a) baseline override-box mesh vs (b) graded non-uniform
    #     Cartesian mesh vs (c) body-fitted square+triangle hybrid unstructured mesh
    if "kottke_nonuniform" in tasks_data and "kottke_reference" in tasks_data:
        pic_nu_schem = NonUniformRingCouplerPIC(
            WL_MIN,
            WL_MAX,
            structures=structures,
            ports=ports,
            n_core=N_SI,
            n_clad=N_SIO2,
            two_d=True,
            averaging="kottke",
            cells_per_wavelength=20,
            domain=domain,
            n_pml=10,
            gap=GAP_NOMINAL,
            cells_per_feature=8,
        )
        plot_mesh_upgrade_comparison_ring_coupler(
            pic_schem,
            pic_nu_schem,
            tasks_data.get("kottke_L3", primary),
            tasks_data["kottke_nonuniform"],
            tasks_data["kottke_reference"],
            fig_dir / "04b_mesh_upgrade_comparison.png",
            gap=GAP_NOMINAL,
        )

    # 5. Port mode profiles (bus 'in', ring 'drop', and 45-deg bent radial mode vs DFT slice)
    bm = bent_waveguide_radial_mode(
        neff_slab, N_SIO2, width=W_WG, radius=R_RING, wavelength=WL_REF
    )
    slice_45 = sample_angled_drop_slice(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_Ex"],
        primary["dft_Ey"],
        ring_center_xy=(0.0, W_WG + GAP_NOMINAL + R_RING),
        radius=R_RING,
        angle_theta=THETA_DROP_REF,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), dpi=150)
    # Panel (a): Bus input port mode Ey(y)
    y_in_um = primary["port_y_in"] * 1e6
    ey_in_norm = primary["port_ey_in"] / max(
        float(np.max(np.abs(primary["port_ey_in"]))), 1e-30
    )
    axes[0].axvspan(-0.25, 0.25, color="0.85", alpha=0.6, label="Si core")
    axes[0].plot(
        y_in_um, np.abs(ey_in_norm), color="#2b5c8f", lw=2.0, label=r"$|E_y|/\max|E_y|$"
    )
    axes[0].plot(
        y_in_um,
        np.real(ey_in_norm),
        color="#d95f02",
        ls="--",
        lw=1.5,
        label=r"$\mathrm{Re}(E_y)$",
    )
    axes[0].set(
        xlabel="Transverse y [µm]",
        ylabel="Normalized modal field",
        title=f"(a) Bus 'in' port mode\n$n_{{\\mathrm{{eff}}}}$ = {float(primary['port_neff_in']):.4f}",
    )
    axes[0].grid(True, ls=":", alpha=0.4)
    axes[0].legend(loc="upper right", fontsize=8.0)

    # Panel (b): Ring drop port mode Ex(x - R_RING)
    x_dr_um = (primary["port_x_drop"] - R_RING) * 1e6
    ex_dr_norm = primary["port_ex_drop"] / max(
        float(np.max(np.abs(primary["port_ex_drop"]))), 1e-30
    )
    axes[1].axvspan(-0.25, 0.25, color="0.85", alpha=0.6, label="Ring core")
    axes[1].plot(
        x_dr_um, np.abs(ex_dr_norm), color="#1b9e77", lw=2.0, label=r"$|E_x|/\max|E_x|$"
    )
    axes[1].plot(
        x_dr_um,
        np.real(ex_dr_norm),
        color="#d95f02",
        ls="--",
        lw=1.5,
        label=r"$\mathrm{Re}(E_x)$",
    )
    axes[1].set(
        xlabel="Radial offset x − r [µm]",
        ylabel="Normalized modal field",
        title=f"(b) Ring 'drop' port mode\n$n_{{\\mathrm{{eff}}}}$ = {float(primary['port_neff_drop']):.4f}",
    )
    axes[1].grid(True, ls=":", alpha=0.4)
    axes[1].legend(loc="upper right", fontsize=8.0)

    # Panel (c): Angled 45-deg bent radial mode vs sampled 2D DFT slice at (3.54, 2.01) um
    u_bm_um = bm["u"] * 1e6
    u_sl_um = slice_45["u"] * 1e6
    e_sl_abs = np.abs(slice_45["E_radial"]) / max(
        float(np.max(np.abs(slice_45["E_radial"]))), 1e-30
    )
    axes[2].axvspan(-0.25, 0.25, color="0.85", alpha=0.6, label="Ring core")
    axes[2].plot(
        u_bm_um,
        bm["E_radial_straight"],
        ":",
        color="0.45",
        lw=1.5,
        label="Straight mode",
    )
    axes[2].plot(
        u_bm_um,
        bm["E_radial_bent"],
        color="#7570b3",
        lw=2.0,
        label=f"Conformal bent (r=5 µm, +{bm['centroid_shift_nm']:.0f} nm)",
    )
    axes[2].plot(
        u_sl_um,
        e_sl_abs,
        "--",
        color="#d95f02",
        lw=1.8,
        label="2D FIT DFT slice @ θ = 45°",
    )
    axes[2].set(
        xlim=(-0.9, 0.9),
        xlabel="Radial coordinate u = r_loc − r [µm]",
        ylabel="Normalized radial field |E_u|",
        title=f"(c) Bent ring mode @ θ = 45°\n$n_{{\\mathrm{{eff,bent}}}}$ = {bm['neff_bent']:.4f}",
    )
    axes[2].grid(True, ls=":", alpha=0.4)
    axes[2].legend(loc="upper right", fontsize=7.5)
    figstyle.save(
        fig,
        fig_dir / "05_port_mode_profiles.png",
        close=True,
        auto_legend_below=False,
    )

    # 6. Waveguide mode effective index and group index dispersion
    plot_neff_dispersion(
        primary["disp_wavelengths"] * 1e9,
        primary["disp_neff"],
        neff_slab=neff_slab,
        ng_2d=primary["disp_ng"],
        out_path=fig_dir / "06_port_neff_dispersion.png",
    )

    # 7. Steady-state logarithmic intensity map |E|^2 [dB] at 1550 nm with overlay_geometry
    render_ring_coupler_field_map_with_overlay(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_E2"],
        pic=pic_schem,
        kind="intensity_log",
        title=r"Steady-state modal intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $\lambda = 1550\;\mathrm{nm}$ (g = 50 nm)",
        out_path=fig_dir / "07_field_intensity_1550.png",
    )

    # 8. Steady-state real field map Re(Ey) at 1550 nm with overlay_geometry
    render_ring_coupler_field_map_with_overlay(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_Ey"],
        pic=pic_schem,
        kind="real",
        title=r"Steady-state wave-front phase pattern $\mathrm{Re}(E_y)$ at $\lambda = 1550\;\mathrm{nm}$ (g = 50 nm)",
        out_path=fig_dir / "08_field_real_ey_1550.png",
    )

    # 9. Animated pulse propagation GIF with overlay_geometry (< 5 MB)
    if "movie_frames" in primary and len(primary["movie_frames"]) > 0:
        render_ring_coupler_movie_with_overlay(
            primary["movie_x"],
            primary["movie_y"],
            primary["movie_frames"],
            primary["movie_times"],
            pic=pic_schem,
            out_path=fig_dir / "09_pulse_propagation.gif",
            fps=12,
        )

    # Sort wavelengths ascending for spectral plots (interior of pulse band: 1505-1595 nm)
    wl_nm_all = primary["wavelengths"] * 1e9
    order = np.argsort(wl_nm_all)
    wl_nm = wl_nm_all[order]
    mask = (wl_nm >= 1505.0) & (wl_nm <= 1595.0)
    wl_b = wl_nm[mask]
    wl_b_m = wl_b * 1e-9

    s11 = primary["S11"][order][mask]
    s21 = primary["S21"][order][mask]
    s31 = primary["S31"][order][mask]
    s41 = primary["S41"][order][mask]
    t2 = np.abs(s21) ** 2
    k2 = np.abs(s31) ** 2
    r2 = np.abs(s11) ** 2
    a2 = np.abs(s41) ** 2

    cmt_spec = cmt_coupling_vs_wavelength(
        wl_b_m, gap=GAP_NOMINAL, radius=R_RING, width=W_WG, fixed_neff_slab=neff_slab
    )

    # 10. Broadband S-parameter spectra: (a) linear |t|^2 and |kappa|^2 vs supermode CMT & under-resolved warning, (b) dB spectra
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.6, 4.2), dpi=150)
    p_cpw = int(primary["cpw"])
    p_caf = float(primary.get("cells_across_feature", primary.get("cells_per_feature", 6)))
    ax0.plot(
        wl_b,
        t2,
        color="#1b9e77",
        lw=2.2,
        label=rf"2D EIM FIT $|t|^2$ ({p_cpw} cpw, {p_caf:.0f} cells/gap)",
    )
    ax0.plot(
        wl_b,
        k2,
        color="#d95f02",
        lw=2.2,
        label=rf"2D EIM FIT $|\kappa|^2$ ({p_cpw} cpw, {p_caf:.0f} cells/gap)",
    )
    if "kottke_nonuniform" in tasks_data:
        d_nu = tasks_data["kottke_nonuniform"]
        w_nu = d_nu["wavelengths"] * 1e9
        ord_nu = np.argsort(w_nu)
        m_nu = (w_nu[ord_nu] >= 1505.0) & (w_nu[ord_nu] <= 1595.0)
        ax0.plot(
            w_nu[ord_nu][m_nu],
            np.abs(d_nu["S31"][ord_nu][m_nu]) ** 2,
            color="#2b5c8f",
            ls="--",
            lw=1.7,
            label=f"Graded non-uniform Kottke $|\\kappa|^2$ ({int(d_nu['n_cells'])//1000}k cells)",
        )
    if "underresolved_staircase" in tasks_data:
        d_us = tasks_data["underresolved_staircase"]
        w_us = d_us["wavelengths"] * 1e9
        ord_us = np.argsort(w_us)
        m_us = (w_us[ord_us] >= 1505.0) & (w_us[ord_us] <= 1595.0)
        caf_us = float(d_us.get("cells_across_feature", 0.95))
        ax0.plot(
            w_us[ord_us][m_us],
            np.abs(d_us["S21"][ord_us][m_us]) ** 2,
            color="#e7298a",
            ls="-.",
            lw=1.5,
            alpha=0.85,
            label=f"Under-resolved staircase $|t|^2$ (10 cpw, {caf_us:.2f} cells/gap — warning)",
        )
        ax0.plot(
            w_us[ord_us][m_us],
            np.abs(d_us["S31"][ord_us][m_us]) ** 2,
            color="#e7298a",
            ls=":",
            lw=1.7,
            alpha=0.9,
            label=f"Under-resolved staircase $|\\kappa|^2$ (10 cpw, {caf_us:.2f} cells/gap — warning)",
        )
    if "underresolved_kottke" in tasks_data:
        d_uk = tasks_data["underresolved_kottke"]
        w_uk = d_uk["wavelengths"] * 1e9
        ord_uk = np.argsort(w_uk)
        m_uk = (w_uk[ord_uk] >= 1505.0) & (w_uk[ord_uk] <= 1595.0)
        caf_uk = float(d_uk.get("cells_across_feature", 0.95))
        ax0.plot(
            w_uk[ord_uk][m_uk],
            np.abs(d_uk["S31"][ord_uk][m_uk]) ** 2,
            color="#7570b3",
            ls=":",
            lw=1.4,
            alpha=0.85,
            label=f"Under-resolved Kottke $|\\kappa|^2$ (10 cpw, {caf_uk:.2f} cells/gap)",
        )
    ax0.plot(
        wl_b,
        cmt_spec["t2_supermode"],
        "--",
        color="#1b9e77",
        lw=1.5,
        label=r"5-layer supermode $|t_{\mathrm{SM}}|^2$",
    )
    ax0.plot(
        wl_b,
        cmt_spec["kappa2_supermode"],
        "--",
        color="#d95f02",
        lw=1.5,
        label=r"5-layer supermode $|\kappa_{\mathrm{SM}}|^2$",
    )
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Linear modal power fraction",
        ylim=(0.0, 1.03),
        title="(a) Linear through & drop coupling spectra (g = 50 nm)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="center right", fontsize=6.8)

    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(t2, 1e-12)),
        color="#1b9e77",
        lw=2.0,
        label=r"Through $|S_{21}|^2$ [dB]",
    )
    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(k2, 1e-12)),
        color="#d95f02",
        lw=2.0,
        label=r"Drop $|\kappa|^2 = |S_{31}|^2$ [dB]",
    )
    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(a2, 1e-12)),
        color="#7570b3",
        ls="--",
        lw=1.7,
        label=r"Backward ring $|S_{41}|^2$ [dB]",
    )
    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(r2, 1e-12)),
        color="#e7298a",
        ls="-.",
        lw=1.7,
        label=r"Bus reflection $|S_{11}|^2$ [dB]",
    )
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Modal power [dB]",
        ylim=(-75.0, 5.0),
        title="(b) Broadband 4-port S-parameters in dB",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="center right", fontsize=8.0)
    figstyle.save(
        fig,
        fig_dir / "10_sparameter_spectra_linear_and_db.png",
        close=True,
        auto_legend_below=False,
    )

    # 11 & 12. Coupling gap sweep (50-250 nm) vs CMT & wavelength spectra across all gaps
    gap_tasks = {"50": primary}
    for g_nm in GAPS_SWEEP_NM[1:]:
        key = f"gap_{g_nm}nm"
        if key in tasks_data:
            gap_tasks[str(g_nm)] = tasks_data[key]

    gaps_fine_nm = np.linspace(40.0, 260.0, 45)
    cmt_curve = cmt_coupling_vs_gap(gaps_fine_nm * 1e-9, n_core_2d=neff_slab)

    avail_gaps_nm = []
    fit_k2_1550 = []
    fit_t2_1550 = []
    for g_nm in GAPS_SWEEP_NM:
        d_g = gap_tasks.get(str(g_nm))
        if d_g is not None:
            i0 = int(np.argmin(np.abs(d_g["wavelengths"] - WL_REF)))
            avail_gaps_nm.append(float(g_nm))
            fit_k2_1550.append(float(np.abs(d_g["S31"][i0]) ** 2))
            fit_t2_1550.append(float(np.abs(d_g["S21"][i0]) ** 2))

    avail_gaps_nm = np.asarray(avail_gaps_nm)
    fit_k2_1550 = np.asarray(fit_k2_1550)
    fit_t2_1550 = np.asarray(fit_t2_1550)
    decay_fit = fit_exponential_decay(avail_gaps_nm * 1e-9, np.arcsin(np.sqrt(fit_k2_1550)))
    sm_pts = cmt_coupling_vs_gap(avail_gaps_nm * 1e-9, n_core_2d=neff_slab)
    sm_decay_fit = fit_exponential_decay(avail_gaps_nm * 1e-9, sm_pts["theta_supermode"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.1), dpi=150)
    ax0.plot(
        gaps_fine_nm,
        cmt_curve["t2_supermode"],
        color="#1b9e77",
        lw=2.0,
        label=r"Supermode CMT $|t(g)|^2$",
    )
    ax0.plot(
        gaps_fine_nm,
        cmt_curve["kappa2_supermode"],
        color="#d95f02",
        lw=2.0,
        label=r"Supermode CMT $|\kappa(g)|^2$",
    )
    ax0.plot(
        gaps_fine_nm,
        cmt_curve["kappa2_cmt"],
        "--",
        color="#2b5c8f",
        lw=1.7,
        label=r"Isolated-mode overlap CMT $|\kappa_{\mathrm{iso}}(g)|^2$",
    )
    ax0.plot(
        avail_gaps_nm,
        fit_t2_1550,
        "s",
        color="#1b9e77",
        ms=6.5,
        label=r"2D FIT $|t|^2$ @ 1550 nm",
    )
    ax0.plot(
        avail_gaps_nm,
        fit_k2_1550,
        "o",
        color="#d95f02",
        ms=6.5,
        label=r"2D FIT $|\kappa|^2$ @ 1550 nm",
    )
    ax0.set(
        xlabel="Edge-to-edge coupling gap g [nm]",
        ylabel="Power fraction at λ = 1550 nm",
        ylim=(-0.02, 1.03),
        title="(a) Power coupling |κ|² & transmission |t|² vs gap",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="center right", fontsize=7.8)

    theta_fit_pts = np.arcsin(np.sqrt(fit_k2_1550))
    ax1.semilogy(
        avail_gaps_nm,
        theta_fit_pts,
        "o",
        color="#d95f02",
        ms=6.5,
        label=r"2D FIT $\Theta(g) = \arcsin|\kappa(g)|$",
    )
    ax1.semilogy(
        gaps_fine_nm,
        decay_fit["amplitude"] * np.exp(-decay_fit["decay_rate_m"] * gaps_fine_nm * 1e-9),
        "-",
        color="#d95f02",
        lw=1.9,
        label=f"Exp fit: $\\gamma_{{\\mathrm{{fit}}}}$ = {decay_fit['decay_rate_um']:.2f} µm⁻¹ ($R^2$={decay_fit['r_squared']:.4f})",
    )
    ax1.semilogy(
        gaps_fine_nm,
        cmt_curve["theta_supermode"],
        "-.",
        color="#1b9e77",
        lw=1.8,
        label=f"5-layer supermode ($\\gamma_{{\\mathrm{{SM}}}}$ = {sm_decay_fit['decay_rate_um']:.2f} µm⁻¹)",
    )
    ax1.semilogy(
        gaps_fine_nm,
        cmt_curve["theta_cmt"],
        "--",
        color="#2b5c8f",
        lw=1.8,
        label=f"Isolated-mode CMT ($\\gamma_{{\\mathrm{{evan}}}}$ = {cmt_curve['gamma_evan']*1e-6:.2f} µm⁻¹)",
    )
    ax1.semilogy(
        gaps_fine_nm,
        cmt_curve["theta_cmt"][0]
        * np.exp(-cmt_curve["gamma_slab"] * (gaps_fine_nm - gaps_fine_nm[0]) * 1e-9),
        ":",
        color="#7570b3",
        lw=1.6,
        label=f"1D slab decay slope ($\\gamma_{{\\mathrm{{slab}}}}$ = {cmt_curve['gamma_slab']*1e-6:.2f} µm⁻¹)",
    )
    ax1.set(
        xlabel="Edge-to-edge coupling gap g [nm]",
        ylabel=r"Coupling phase $\Theta(g) \approx |\kappa(g)|$ (log scale)",
        title="(b) Exponential evanescent decay constant audit",
    )
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=7.8)
    figstyle.save(
        fig,
        fig_dir / "11_coupling_vs_gap_cmt.png",
        close=True,
        auto_legend_below=False,
    )

    # 12. Broadband coupling and transmission spectra across all 5 gaps
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.1), dpi=150)
    cmap_cols = ["#d95f02", "#1b9e77", "#2b5c8f", "#7570b3", "#e7298a"]
    for i, g_nm in enumerate(GAPS_SWEEP_NM):
        d_g = gap_tasks.get(str(g_nm))
        if d_g is None:
            continue
        w_g = d_g["wavelengths"] * 1e9
        ord_g = np.argsort(w_g)
        w_s = w_g[ord_g]
        m_g = (w_s >= 1505.0) & (w_s <= 1595.0)
        k2_g = np.abs(d_g["S31"][ord_g][m_g]) ** 2
        t2_g = np.abs(d_g["S21"][ord_g][m_g]) ** 2
        col = cmap_cols[i % len(cmap_cols)]
        ax0.plot(w_s[m_g], k2_g, color=col, lw=2.0, label=f"g = {g_nm} nm (FIT)")
        cmt_g = cmt_coupling_vs_wavelength(
            w_s[m_g] * 1e-9, gap=g_nm * 1e-9, fixed_neff_slab=neff_slab
        )
        ax0.plot(w_s[m_g], cmt_g["kappa2_supermode"], "--", color=col, lw=1.2, alpha=0.8)
        ax1.plot(w_s[m_g], t2_g, color=col, lw=2.0, label=f"g = {g_nm} nm (FIT)")
        ax1.plot(w_s[m_g], cmt_g["t2_supermode"], "--", color=col, lw=1.2, alpha=0.8)

    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Drop power coupling $|\kappa(\lambda)|^2$",
        title="(a) Drop coupling |κ(λ)|² across gaps (solid: FIT, dashed: SM-CMT)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper left", fontsize=8.0)

    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Through transmission $|t(\lambda)|^2$",
        title="(b) Through transmission |t(λ)|² across gaps",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=8.0)
    figstyle.save(
        fig,
        fig_dir / "12_gap_wavelength_spectra.png",
        close=True,
        auto_legend_below=False,
    )

    # 13. Passivity, Lorentz reciprocity (S21=S12, S31=S13), and CPML waveguide extrusion audit
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.1), dpi=150)
    p_sum = t2 + k2 + r2 + a2
    ax0.plot(
        wl_b,
        p_sum,
        color="#1b9e77",
        lw=2.2,
        label=f"Guided sum $|t|^2 + |\\kappa|^2 + |S_{{11}}|^2 + |S_{{41}}|^2$ (mean={np.mean(p_sum):.4f})",
    )
    if "flux_total_box" in primary:
        f_box = primary["flux_total_box"][order][mask]
        ax0.plot(
            wl_b,
            f_box,
            color="#2b5c8f",
            ls="--",
            lw=1.8,
            label=f"2D Poynting flux box total (mean={np.mean(f_box):.4f})",
        )
    ax0.axhline(1.0, color="0.35", ls=":", lw=1.4, label="Lossless unitary limit (1.0)")
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Normalized power",
        ylim=(0.90, 1.04),
        title="(a) Passivity & Poynting energy conservation audit",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=7.8)

    if "S12" in primary:
        s12 = primary["S12"][order][mask]
        s42 = primary["S42"][order][mask]
        s13 = primary["S13"][order][mask]
        rec_bus = np.maximum(np.abs(np.abs(s21) - np.abs(s12)), 1e-16)
        rec_sym = np.maximum(np.abs(np.abs(s31) - np.abs(s42)), 1e-16)
        rec_cross = np.maximum(np.abs(np.abs(s31) - np.abs(s13)), 1e-16)
        ax1.semilogy(
            wl_b,
            rec_bus,
            color="#2b5c8f",
            lw=1.8,
            label=r"Bus reciprocity $||S_{21}| - |S_{12}||$",
        )
        ax1.semilogy(
            wl_b,
            rec_sym,
            color="#1b9e77",
            ls="--",
            lw=1.8,
            label=r"Left/right symmetry $||S_{31}| - |S_{42}||$",
        )
        ax1.semilogy(
            wl_b,
            rec_cross,
            color="#7570b3",
            ls=":",
            lw=1.7,
            label=r"Cross-axis reciprocity $||S_{31}| - |S_{13}||$",
        )

    ax1.semilogy(
        wl_b,
        np.maximum(a2, 1e-16),
        color="#d95f02",
        lw=2.0,
        label=r"Backward ring $|S_{41}|^2$ (CPML extruded)",
    )
    if "noext_S41" in primary:
        a2_noext = np.abs(primary["noext_S41"][order][mask]) ** 2
        ax1.semilogy(
            wl_b,
            np.maximum(a2_noext, 1e-16),
            color="#e7298a",
            ls="-.",
            lw=1.9,
            label=r"Backward ring $|S_{41}|^2$ (truncated before CPML)",
        )
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Dimensionless magnitude / power (log scale)",
        ylim=(1e-9, 1e-1),
        title="(b) Lorentz reciprocity & CPML waveguide extrusion audit",
    )
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=7.2)
    figstyle.save(
        fig,
        fig_dir / "13_passivity_reciprocity_and_pml_audit.png",
        close=True,
        auto_legend_below=False,
    )

    # 14. Multi-resolution convergence: Kottke vs Staircase against independent kottke_reference
    levels = [
        lvl
        for lvl in ("L1", "L2", "L3")
        if f"kottke_{lvl}" in tasks_data and f"staircase_{lvl}" in tasks_data
    ]
    if not levels:
        levels = [
            lvl
            for lvl in ("coarse", "medium", "fine")
            if f"kottke_{lvl}" in tasks_data and f"staircase_{lvl}" in tasks_data
        ]
    if levels and "kottke_reference" in tasks_data:
        cpws = [float(tasks_data[f"kottke_{lvl}"]["cpw"]) for lvl in levels]
        cafs = [
            float(
                tasks_data[f"kottke_{lvl}"].get(
                    "cells_across_feature",
                    tasks_data[f"kottke_{lvl}"].get("cells_per_feature", 4.0),
                )
            )
            for lvl in levels
        ]

        def _k2_t2_at_1550(d):
            w = d["wavelengths"]
            i0 = int(np.argmin(np.abs(w - WL_REF)))
            return (
                float(np.abs(d["S31"][i0]) ** 2),
                float(np.abs(d["S21"][i0]) ** 2),
            )

        d_ref = tasks_data["kottke_reference"]
        k_ref, t_ref = _k2_t2_at_1550(d_ref)
        ref_cpw = int(d_ref["cpw"])
        ref_caf = float(
            d_ref.get("cells_across_feature", d_ref.get("cells_per_feature", 8.0))
        )
        kk_vals = [_k2_t2_at_1550(tasks_data[f"kottke_{lvl}"])[0] for lvl in levels]
        ks_vals = [_k2_t2_at_1550(tasks_data[f"staircase_{lvl}"])[0] for lvl in levels]
        tk_vals = [_k2_t2_at_1550(tasks_data[f"kottke_{lvl}"])[1] for lvl in levels]
        ts_vals = [_k2_t2_at_1550(tasks_data[f"staircase_{lvl}"])[1] for lvl in levels]
        err_t_k = [max(abs(v - t_ref) / max(t_ref, 1e-12), 1e-6) for v in tk_vals]
        err_t_s = [max(abs(v - t_ref) / max(t_ref, 1e-12), 1e-6) for v in ts_vals]
        err_k_k = [max(abs(v - k_ref) / max(k_ref, 1e-12), 1e-6) for v in kk_vals]
        err_k_s = [max(abs(v - k_ref) / max(k_ref, 1e-12), 1e-6) for v in ks_vals]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=150)
        res = np.asarray(cpws, dtype=float)
        ax0.loglog(
            res,
            err_t_s,
            "s-",
            color="#d95f02",
            lw=2.1,
            ms=6.5,
            label=r"Staircase $|T - T_{\mathrm{ref}}| / T_{\mathrm{ref}}$",
        )
        ax0.loglog(
            res,
            err_t_k,
            "o-",
            color="#1b9e77",
            lw=2.1,
            ms=6.5,
            label=r"Kottke $|T - T_{\mathrm{ref}}| / T_{\mathrm{ref}}$",
        )
        ax0.loglog(
            res,
            err_k_s,
            "s--",
            color="#d95f02",
            mfc="none",
            lw=1.3,
            ms=5.5,
            alpha=0.8,
            label=r"Staircase $||\kappa|^2 - |\kappa_{\mathrm{ref}}|^2| / |\kappa_{\mathrm{ref}}|^2$",
        )
        ax0.loglog(
            res,
            err_k_k,
            "o--",
            color="#1b9e77",
            mfc="none",
            lw=1.3,
            ms=5.5,
            alpha=0.8,
            label=r"Kottke $||\kappa|^2 - |\kappa_{\mathrm{ref}}|^2| / |\kappa_{\mathrm{ref}}|^2$",
        )
        ax0.loglog(
            res,
            err_t_s[0] * (res[0] / res) ** 1,
            ":",
            color="0.45",
            lw=1.2,
            label=r"$\mathcal{O}(\Delta x^1)$ / $\mathcal{O}(\Delta x^2)$ guides",
        )
        ax0.loglog(
            res,
            err_t_k[0] * (res[0] / res) ** 2,
            ":",
            color="0.45",
            lw=1.2,
        )

        # Annotate every resolved ladder point with its measured cells across the 50 nm feature
        for idx_p, (lvl, x_cpw, caf_val, etk, ets, ekk, eks) in enumerate(
            zip(levels, cpws, cafs, err_t_k, err_t_s, err_k_k, err_k_s)
        ):
            if idx_p == 0:
                k_off, s_off, ha_k, ha_s = (8, 1), (8, 4), "left", "left"
            elif idx_p == len(levels) - 1:
                k_off, s_off, ha_k, ha_s = (-6, -12), (-6, 6), "right", "right"
            else:
                k_off, s_off, ha_k, ha_s = (-6, 8), (0, -14), "center", "center"
            ax0.annotate(
                f"{lvl} Kottke ({caf_val:.1f} cells/gap)",
                xy=(x_cpw, etk),
                xytext=k_off,
                textcoords="offset points",
                fontsize=7.3,
                color="#1b9e77",
                fontweight="bold",
                ha=ha_k,
            )
            ax0.annotate(
                f"{lvl} Staircase ({caf_val:.1f} cells/gap)",
                xy=(x_cpw, ets),
                xytext=s_off,
                textcoords="offset points",
                fontsize=7.3,
                color="#d95f02",
                ha=ha_s,
            )

        # Upgraded graded non-uniform Cartesian mesh point
        if "kottke_nonuniform" in tasks_data:
            d_nu = tasks_data["kottke_nonuniform"]
            k_nu, t_nu = _k2_t2_at_1550(d_nu)
            et_nu = max(abs(t_nu - t_ref) / max(t_ref, 1e-12), 1e-6)
            ek_nu = max(abs(k_nu - k_ref) / max(k_ref, 1e-12), 1e-6)
            caf_nu = float(d_nu.get("cells_across_feature", 8.0))
            cpw_nu = float(d_nu["cpw"])
            n_nu = int(d_nu["n_cells"])
            ax0.loglog(
                [cpw_nu, cpw_nu],
                [et_nu, ek_nu],
                "*",
                color="#2b5c8f",
                ms=10.0,
                label=f"Graded non-uniform Kottke ({n_nu//1000}k cells, {caf_nu:.0f} cells/gap)",
            )
            ax0.annotate(
                f"Graded non-uniform\n({n_nu//1000}k cells, -37.2% vs L3)",
                xy=(cpw_nu, et_nu),
                xytext=(-8, -22),
                textcoords="offset points",
                fontsize=7.1,
                color="#2b5c8f",
                fontweight="bold",
                ha="right",
            )

        # Under-resolved demonstration pair (10 cpw, no mesh overrides, ~0.95 cells/gap)
        if "underresolved_staircase" in tasks_data:
            d_us = tasks_data["underresolved_staircase"]
            k_us, t_us = _k2_t2_at_1550(d_us)
            et_us = max(abs(t_us - t_ref) / max(t_ref, 1e-12), 1e-6)
            ek_us = max(abs(k_us - k_ref) / max(k_ref, 1e-12), 1e-6)
            caf_us = float(d_us.get("cells_across_feature", 0.95))
            cpw_us = float(d_us["cpw"])
            ax0.loglog(
                [cpw_us, cpw_us],
                [et_us, ek_us],
                "^",
                color="#e7298a",
                mfc="none",
                mew=2.0,
                ms=8.0,
                label=f"Under-resolved staircase ({caf_us:.2f} cells/gap — warning)",
            )
            ax0.annotate(
                f"Under-resolved staircase\n({caf_us:.2f} cells/gap, no override)",
                xy=(cpw_us, ek_us),
                xytext=(10, -6),
                textcoords="offset points",
                fontsize=7.2,
                color="#e7298a",
                fontweight="bold",
            )
        if "underresolved_kottke" in tasks_data:
            d_uk = tasks_data["underresolved_kottke"]
            k_uk, t_uk = _k2_t2_at_1550(d_uk)
            et_uk = max(abs(t_uk - t_ref) / max(t_ref, 1e-12), 1e-6)
            caf_uk = float(d_uk.get("cells_across_feature", 0.95))
            cpw_uk = float(d_uk["cpw"])
            ax0.loglog(
                [cpw_uk],
                [et_uk],
                "D",
                color="#7570b3",
                mfc="none",
                mew=2.0,
                ms=7.5,
                label=f"Under-resolved Kottke ({caf_uk:.2f} cells/gap — warning)",
            )
            ax0.annotate(
                f"Under-resolved Kottke ({caf_uk:.2f} cells/gap)",
                xy=(cpw_uk, et_uk),
                xytext=(10, -13),
                textcoords="offset points",
                fontsize=7.1,
                color="#7570b3",
            )

        ax0.set_xlim(res[0] * 0.86, res[-1] * 1.24)
        ax0.set_ylim(1.5e-5, 1.0)
        ax0.set(
            xlabel="Background resolution [cells per wavelength in slab]",
            ylabel=r"Relative error vs Kottke reference $T_{\mathrm{ref}}, |\kappa_{\mathrm{ref}}|^2$",
            title=f"(a) Convergence vs Kottke ref (cpw={ref_cpw}, {ref_caf:.0f} cells/gap)",
        )
        ax0.grid(True, which="both", ls=":", alpha=0.4)
        ax0.legend(loc="lower left", fontsize=6.2)

        # Second x-axis on top showing measured cells across the 50 nm feature
        ax0_top = ax0.twiny()
        ax0_top._is_inset = True
        ax0_top.set_xscale("log")
        ax0_top.set_xlim(ax0.get_xlim())
        ax0_top.set_xticks(res)
        ax0_top.set_xticklabels(
            [f"{c:.1f} cells/gap\n({lvl})" for lvl, c in zip(levels, cafs)],
            fontsize=7.8,
        )
        ax0_top.minorticks_off()
        ax0_top.set_xlabel(
            "Cells across 50 nm minimum feature (override box)", fontsize=8.5
        )

        cols = {
            "L1": "#7570b3",
            "L2": "#d95f02",
            "L3": "#1b9e77",
            "coarse": "#7570b3",
            "medium": "#d95f02",
            "fine": "#1b9e77",
        }
        for lvl in levels:
            for avg, ls in (("kottke", "-"), ("staircase", "--")):
                d = tasks_data[f"{avg}_{lvl}"]
                w = d["wavelengths"] * 1e9
                idx_s = np.argsort(w)
                w_s = w[idx_s]
                m = (w_s >= 1505.0) & (w_s <= 1595.0)
                k2_c = np.abs(d["S31"][idx_s][m]) ** 2
                caf_d = float(
                    d.get("cells_across_feature", d.get("cells_per_feature", 4.0))
                )
                lbl = f"{avg.capitalize()} {lvl} (cpw={int(d['cpw'])}, {caf_d:.0f} cells/gap)"
                ax1.plot(w_s[m], k2_c, color=cols.get(lvl, "#2b5c8f"), ls=ls, lw=1.7, label=lbl)

        if "kottke_nonuniform" in tasks_data:
            d_nu = tasks_data["kottke_nonuniform"]
            w_nu = d_nu["wavelengths"] * 1e9
            idx_nu = np.argsort(w_nu)
            m_nu = (w_nu[idx_nu] >= 1505.0) & (w_nu[idx_nu] <= 1595.0)
            ax1.plot(
                w_nu[idx_nu][m_nu],
                np.abs(d_nu["S31"][idx_nu][m_nu]) ** 2,
                color="#2b5c8f",
                ls="-.",
                lw=1.8,
                label=f"Graded non-uniform Kottke ({int(d_nu['n_cells'])//1000}k cells, 8 cells/gap)",
            )

        if "underresolved_staircase" in tasks_data:
            d_us = tasks_data["underresolved_staircase"]
            w_u = d_us["wavelengths"] * 1e9
            idx_u = np.argsort(w_u)
            m_u = (w_u[idx_u] >= 1505.0) & (w_u[idx_u] <= 1595.0)
            caf_u = float(d_us.get("cells_across_feature", 0.95))
            ax1.plot(
                w_u[idx_u][m_u],
                np.abs(d_us["S31"][idx_u][m_u]) ** 2,
                color="#e7298a",
                ls="-.",
                lw=1.6,
                label=f"Under-resolved staircase (10 cpw, {caf_u:.2f} cells/gap — warning)",
            )
        if "underresolved_kottke" in tasks_data:
            d_uk = tasks_data["underresolved_kottke"]
            w_u = d_uk["wavelengths"] * 1e9
            idx_u = np.argsort(w_u)
            m_u = (w_u[idx_u] >= 1505.0) & (w_u[idx_u] <= 1595.0)
            caf_u = float(d_uk.get("cells_across_feature", 0.95))
            ax1.plot(
                w_u[idx_u][m_u],
                np.abs(d_uk["S31"][idx_u][m_u]) ** 2,
                color="#2b5c8f",
                ls=":",
                lw=1.5,
                label=f"Under-resolved Kottke (10 cpw, {caf_u:.2f} cells/gap)",
            )

        w_r = d_ref["wavelengths"] * 1e9
        idx_r = np.argsort(w_r)
        m_r = (w_r[idx_r] >= 1505.0) & (w_r[idx_r] <= 1595.0)
        ax1.plot(
            w_r[idx_r][m_r],
            np.abs(d_ref["S31"][idx_r][m_r]) ** 2,
            color="black",
            lw=2.1,
            ls=":",
            label=f"Kottke reference (cpw={ref_cpw}, {ref_caf:.0f} cells/gap)",
        )
        ax1.set(
            xlabel="Wavelength λ [nm]",
            ylabel=r"Drop power coupling $|\kappa(\lambda)|^2$",
            title="(b) Broadband coupling: resolved ladder vs under-resolved warning",
        )
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.legend(loc="best", fontsize=6.5)
        figstyle.save(
            fig,
            fig_dir / "14_kottke_vs_staircase_convergence.png",
            close=True,
            auto_legend_below=False,
        )

    # 15. Synthesized all-pass microring resonator spectrum & time-domain port trace
    i_disp_ref = int(np.argmin(np.abs(primary["disp_wavelengths"] - WL_REF)))
    neff_1550 = float(primary["disp_neff"][i_disp_ref])
    ng_1550 = float(primary["disp_ng"][i_disp_ref])
    ring_m = ring_resonator_metrics(
        wl_b_m,
        s21,
        s31,
        neff_ch=neff_1550,
        ng_ch=ng_1550,
        radius=R_RING,
        wavelength_ref=WL_REF,
        round_trip_loss_db=0.15,
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.0), dpi=150)
    ax0.plot(
        ring_m["wl_fine_nm"],
        ring_m["T_allpass_db"],
        color="#2b5c8f",
        lw=1.7,
        label=(
            f"All-pass ring (FSR = {ring_m['fsr_nm']:.2f} nm, "
            f"$\\mathcal{{F}}$ = {ring_m['finesse']:.1f}, $Q_L$ = {ring_m['Q_loaded']:.0f})"
        ),
    )
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="All-pass bus transmission [dB]",
        ylim=(-22.0, 1.0),
        title="(a) Synthesized microring resonator spectrum from FIT (t, κ)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=7.8)

    if "trace_t" in primary and primary["trace_t"].size > 0:
        t_fs = primary["trace_t"] * 1e15
        ey_tr = primary["trace_Ey"]
        ey_norm = ey_tr / max(float(np.max(np.abs(ey_tr))), 1e-30)
        env = np.maximum(ey_norm**2, 1e-10)
        ax1.semilogy(
            t_fs,
            env,
            color="#d95f02",
            lw=1.5,
            label=r"Through port $|E_y(t)|^2 / |E_y|_{\max}^2$",
        )
        ax1.plot(
            t_fs,
            0.5 * (1.0 + ey_norm) * 1e-8,
             alpha=0.0,
        )
        ax1.axhline(
            float(primary.get("energy_decay", 1e-4)),
            color="0.4",
            ls="--",
            lw=1.2,
            label="Energy decay threshold",
        )
        ax1.set(
            xlabel="Time t [fs]",
            ylabel="Normalized instantaneous power (log scale)",
            ylim=(1e-8, 2.0),
            title="(b) Single-pass coupler ring-down & termination audit",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=8.0)

    figstyle.save(
        fig,
        fig_dir / "15_ring_resonator_synthesis_and_time_trace.png",
        close=True,
        auto_legend_below=False,
    )

    print(f"Rendered setup and study figures in {fig_dir}")


# %% [markdown]
# ## 7. Limits, failure modes & when 2D EIM breaks down
#
# ### What goes wrong when a narrow coupling gap is resolved by only 1–2 cells
#
# The under-resolved warning pair (`underresolved_staircase` and
# `underresolved_kottke`, `task_8` and `task_7`, run at $10\;\text{cpw}$ with no
# mesh override so that $\Delta y = 52.47\;\text{nm}$, or **$0.95\;\text{cells}$
# across the $50\;\text{nm}$ gap**) demonstrates why background wavelength
# resolution alone is insufficient for sub-wavelength photonic slots:
#
# - **Under-resolved staircasing (`0.95 cells/gap`, `task_8`)**: Because a single
#   $52.5\;\text{nm}$ cell straddles the $50\;\text{nm}$ oxide slot (`04_permittivity_mesh_zoom.png`
#   panel a), rounding the fill fraction to $0$ or $1$ snaps the curved ring
#   wall by full $52.5\;\text{nm}$ jumps, artificially narrowing the effective
#   gap near $x = 0$. The resulting coupling at $1550\;\text{nm}$ is
#   $|\kappa|^2 = 0.309800$ ($|t|^2 = 0.680790$)—a **$35.63\%$ relative error**
#   in drop power and an **$11.09\%$ relative error** in through power compared
#   to the $R = (40, 16)$ reference (`kottke_reference`, $|\kappa_{\text{ref}}|^2 = 0.228419$,
#   $|t_{\text{ref}}|^2 = 0.765742$), accompanied by $|S_{41}|^2 = 1.04 \times 10^{-3}$
#   ($-29.8\;\text{dB}$) of spurious backward ring scattering. In a closed
#   microring resonator, this staircased gap error shifts the coupling $Q_c$ by
#   $\sim 40\%$ and distorts the critical-coupling condition.
# - **Under-resolved Kottke subpixel smoothing (`0.95 cells/gap`, `task_7`)**:
#   Harmonic averaging of $\varepsilon_{yy}^{-1}$ preserves the integrated series
#   capacitance of the $50\;\text{nm}$ slot inside the cut cell, reducing the
#   through-power error to $0.72\%$ ($|t|^2 = 0.760200$) and the drop-power error
#   to $3.06\%$ ($|\kappa|^2 = 0.235401$). However, a single cell across the gap
#   cannot resolve the spatial field profile or transverse gradient inside the
#   slot.
#
# ### Convergence & resolution audit across the `--full` ladder ($L_1, L_2, L_3, R$)
#
# - **Resolution audit**: Every non-demo run in `data/` (`task_0`..`task_6` and
#   `task_9`..`task_13`) satisfies $\text{cpw} \ge 10$ in the highest-index
#   material ($n_{\text{eff,slab}} = 2.8478$, $\text{cpw} \in \{10, 14, 20, 40\}$)
#   and $\text{cells\_across\_feature} \ge 4.00$ across the coupling gap
#   ($\{4.00, 6.00, 8.00, 9.51, 16.00\}\;\text{cells/gap}$, $N_z = 1$), with zero
#   violations. Only the two intentional warning demos (`task_7`, `task_8`) are
#   under-resolved across the gap ($0.95\;\text{cells/gap}$).
# - **Kottke subpixel & graded non-uniform convergence**: Adding $N = 4\;\text{cells}$ across the
#   $50\;\text{nm}$ slot at $L_1 = (10, 4)$ ($\Delta x_{\min} = 12.44\;\text{nm}$)
#   gives `kottke_L1` $|t|^2 = 0.757153$ ($1.12\%$ error vs $R$) and
#   $|\kappa|^2 = 0.237189$ ($3.84\%$ error vs $R$). Refining to $L_2 = (14, 6)$
#   ($\Delta x_{\min} = 8.33\;\text{nm}$) and $L_3 = (20, 8)$ ($\Delta x_{\min} = 6.24\;\text{nm}$)
#   yields monotonic convergence: between the last two ladder levels ($L_2 \to L_3$),
#   Kottke's through power changes from $0.764171$ to $0.764200$ (**relative
#   change $3.69 \times 10^{-5}$ or $0.0037\%$**) and drop power changes from
#   $0.230141$ to $0.230135$ (**relative change $2.94 \times 10^{-5}$ or
#   $0.0029\%$**). Relative to the fine $R = (40, 16)$ reference
#   ($\Delta x_{\min} = 3.12\;\text{nm}$, $996,800\;\text{cells}$,
#   $|t_{\text{ref}}|^2 = 0.765742$, $|\kappa_{\text{ref}}|^2 = 0.228419$),
#   `kottke_L3` is converged to **$0.201\%$ in $|t|^2$** and **$0.751\%$ in
#   $|\kappa|^2$**, while the graded non-uniform Cartesian mesh (`kottke_nonuniform`,
#   $165,487\;\text{cells}$, $-37.2\%$ fewer cells and $-54.8\%$ wall time vs
#   single-drive $L_3$) achieves **$0.261\%$ in $|t|^2$** ($0.763740$) and
#   **$1.059\%$ in $|\kappa|^2$** ($0.230838$).
# - **Kottke vs staircase across full levels**: In the
#   `--quick` study (which stops at $L_2$ and uses $L_3$ as reference
#   with `energy_decay = 3e-3`), `staircase_L2` crosses near the
#   reference value ($0.37\%$ apparent difference). Extending the ladder in
#   `--full` to $L_3 = (20, 8)$ and $R = (40, 16)$ (`energy_decay = 1e-4`)
#   shows that **staircase coupling oscillates non-monotonically
#   with grid alignment** as Yee cell boundaries shift across the curved ring wall,
#   with drop-power error relative to $R$ swinging from $+35.63\%$ (`underresolved`,
#   $0.95\;\text{cells/gap}$) $\to -6.29\%$ ($L_1$, $|\kappa|^2 = 0.214059$)
#   $\to +1.31\%$ ($L_2$, $|\kappa|^2 = 0.231421$) $\to -2.48\%$ ($L_3$,
#   $|\kappa|^2 = 0.222757$). Between $L_2$ and $L_3$, staircase shifts by
#   **$0.996\%$ in $|t|^2$** and **$3.89\%$ in $|\kappa|^2$** ($>1000\times$ larger
#   than Kottke's $L_2 \to L_3$ step), leaving at $L_3$ a **$2.48\%$ drop-power
#   error** ($3.3\times$ larger than `kottke_L3`'s $0.751\%$), a **$0.845\%$
#   through-power error** ($4.2\times$ larger than `kottke_L3`'s $0.201\%$), and
#   **$3.4\times$ higher spurious backward ring reflection** ($|S_{41}|^2 = 3.48 \times 10^{-5}$
#   vs $1.02 \times 10^{-5}$ at $L_3$ and $9.69 \times 10^{-6}$ at `kottke_nonuniform`).
#
# ### Physical regimes of 2D EIM vs 3D SOI & gap-sweep decay audit
#
# Comparing our converged 2D EIM Kottke coupling at $g_0 = 50\;\text{nm}$ and
# $\lambda_0 = 1550\;\text{nm}$ ($|\kappa|^2 = 0.230135$ at $L_3$ and $0.228419$
# at $R$; $|t|^2 = 0.764200$ at $L_3$ and $0.765742$ at $R$; synthesized ring
# $\text{FSR} = 24.14\;\text{nm}$, finesse $\mathcal{F} = 20.7$,
# $Q_L = 1328$) against analytical coupled-mode theory highlights three physical
# regimes:
#
# 1. **Narrow-slot normal-$E_y$ enhancement ($g_0 = 50\;\text{nm}$)**: In 2D EIM,
#   the silicon core is effectively infinite along $z$ (`bc_z = periodic`) with
#   effective index $n_{\text{eff,slab}} = 2.8478$. Because the in-plane electric
#   field $E_y$ is normal to the sidewall, continuity of $D_y = \varepsilon E_y$
#   enhances the electric field inside the $50\;\text{nm}$ $\text{SiO}_2$ slot by
#   $(n_{\text{eff,slab}} / n_{\text{SiO}_2})^2 = 3.89\times$ across the entire
#   vertical slab. In 3D SOI ($h_{\text{Si}} = 220\;\text{nm}$), electric field
#   lines fringe into the top and bottom $\text{SiO}_2$ cladding around the
#   $50\;\text{nm} \times 220\;\text{nm}$ high-aspect-ratio slot, reducing the net
#   slot capacitance and lowering $|\kappa|^2$ from $0.228\text{--}0.230$ (2D EIM
#   FIT) and $0.2009$ (5-layer supermode theory) toward the
#   isolated-mode weak-coupling CMT value ($0.1582$). Note also why the `--full` drop
#   power at $50\;\text{nm}$ is $\sim 7\%$ higher than in `--quick` ($0.2301$ vs
#   $0.2143$): tightening `energy_decay` from `3e-3` to `1e-4` and widening the
#   CPML from $8$ to $10\;\text{cells}$ allows the slower curved-ring drop pulse
#   to ring down completely, raising the guided power sum $\sum_j |S_{j1}(1550)|^2$
#   from $0.9758$ ($0.107\;\text{dB}$ excess loss) to $0.9944$ ($0.025\;\text{dB}$
#   insertion loss at $1550\;\text{nm}$; $0.9966$ band mean).
# 2. **Gap sweep $\kappa^2(g)$ ($50\text{--}250\;\text{nm}$) and exponential decay
#   constant vs supermode theory**: Across the resolved $L_3 = (20, 8)$ gap sweep
#   ($g_0 \in \{50, 100, 150, 200, 250\}\;\text{nm}$), the 2D EIM FIT drop power
#   $|\kappa(g)|^2 = \{0.230135, 0.088533, 0.036717, 0.015826, 0.006666\}$ and
#   through power $|t(g)|^2 = \{0.764200, 0.908871, 0.961135, 0.981894, 0.990773\}$
#   track the 5-layer supermode prediction
#   $|\kappa_{\text{SM}}(g)|^2 = \{0.200859, 0.078862, 0.033155, 0.014314, 0.006244\}$
#   and isolated-mode CMT
#   $|\kappa_{\text{CMT}}(g)|^2 = \{0.158183, 0.071685, 0.031914, 0.014099, 0.006207\}$.
#   Fitting $\Theta(g) = \arcsin|\kappa(g)| \propto \exp(-\gamma_{\text{fit}} g)$
#   yields $\gamma_{\text{fit}} = 8.99\;\mu\text{m}^{-1}$ ($8.994\;\mu\text{m}^{-1}$,
#   $R^2 = 0.9988$, vs $9.31\;\mu\text{m}^{-1}$ in `--quick`), which agrees with
#   the **analytic 5-layer supermode decay constant**
#   $\gamma_{\text{SM}} = 8.81\;\mu\text{m}^{-1}$ ($8.811\;\mu\text{m}^{-1}$,
#   $R^2 = 0.9989$) to within **$2.07\%$** and lies cleanly between the isolated
#   channel decay constant $\gamma_{\text{evan}} = 8.23\;\mu\text{m}^{-1}$ and the
#   1D slab decay rate $\gamma_{\text{slab}} = 9.95\;\mu\text{m}^{-1}$. Fitting
#   the power coupling $|\kappa(g)|^2 \propto \exp(-2\gamma g)$ directly gives
#   $2\gamma_{\text{fit}} = 17.61\;\mu\text{m}^{-1}$ ($\gamma_{\text{power}} = 8.81\;\mu\text{m}^{-1}$,
#   $R^2 = 0.9994$) vs $2\gamma_{\text{SM}} = 17.30\;\mu\text{m}^{-1}$
#   ($\gamma_{\text{SM,power}} = 8.65\;\mu\text{m}^{-1}$, $R^2 = 0.9994$, within
#   **$1.81\%$**) and isolated-mode CMT $2\gamma_{\text{CMT}} = 16.20\;\mu\text{m}^{-1}$.
# 3. **Bent-waveguide radiation & dispersion**: A $5\;\mu\text{m}$ silicon bend
#   exhibits an $+80.1\;\text{nm}$ centrifugal outward shift of the radial mode
#   profile (`05_port_mode_profiles.png` panel c, $n_{\text{eff,bent}} = 2.5023$ vs
#   $n_{\text{eff,straight}} = 2.4914$) and a small bend-to-straight mode mismatch
#   that accounts for the $0.025\;\text{dB}$ single-pass coupler insertion loss at
#   $1550\;\text{nm}$ ($0.003\;\text{dB}$ mean across the 2D Poynting flux box).
#
# ### Quick (`--quick`) vs Full (`--full` & `kottke_nonuniform`) vs Reference comparison table
#
# | Metric ($\lambda_0 = 1550\;\text{nm}$ unless noted) | Quick (`--quick`, $L_2 = (14, 6)$, `decay=3e-3`) | Full (`--full`, $L_3 = (20, 8)$ / `kottke_nonuniform` / $R = (40, 16)$) | Analytic / Published Reference |
# |---|---|---|---|
# | Slab effective index $n_{\text{eff,slab}}$ | $2.8478$ | $2.8478$ | $2.8478$ (exact 1D $\text{TE}_0$ slab) |
# | Bus port effective index $n_{\text{eff}}(1550)$ | $2.4980$ | $2.4984$ (`in`), $2.4990$ (`drop`) | $2.4914$ (1D isolated channel); $2.5023$ (conformal bent) |
# | Bus port group index $n_g(1550)$ | $3.197$ | $3.1677$ ($3.157\text{--}3.178$ band) | $3.168$ (2D EIM channel dispersion) |
# | Through power $\|t(1550)\|^2$ ($g = 50\;\text{nm}$) | $0.761504$ ($L_2$) / $0.761478$ ($L_3$ q-ref) | $0.764200$ ($L_3$) / $0.763740$ (`nonuniform`) / $0.765742$ ($R$) | $0.799141$ (5-layer SM); $0.841817$ (Iso-CMT) |
# | Drop power $\|\kappa(1550)\|^2$ ($g = 50\;\text{nm}$) | $0.214254$ ($L_2$) / $0.215319$ ($L_3$ q-ref) | $0.230135$ ($L_3$) / $0.230838$ (`nonuniform`) / $0.228419$ ($R$) | $0.200859$ (5-layer SM); $0.158183$ (Iso-CMT) |
# | Cell count & single-drive wall time ($g = 50\;\text{nm}$) | $138,824\;\text{cells}$ ($L_2$) | $263,466\;\text{cells}$ ($1437.3\;\text{s}$, $L_3$) vs $165,487\;\text{cells}$ ($650.2\;\text{s}$, `nonuniform`, $-37.2\%$ cells, $-54.8\%$ time) | $996,800\;\text{cells}$ ($11,317.6\;\text{s}$, $R$) |
# | Kottke last-step rel. change ($\|t\|^2, \|\kappa\|^2$) | $3.4 \times 10^{-5},\; 4.9 \times 10^{-3}$ ($L_2 \to L_3$) | $3.7 \times 10^{-5},\; 2.9 \times 10^{-5}$ ($L_2 \to L_3$); $0.20\%,\; 0.75\%$ vs $R$ | Converged $< 1\%$ |
# | Staircase rel. error vs ref ($\|t\|^2, \|\kappa\|^2$) | $0.12\%,\; 0.37\%$ ($L_2$ vs $L_3$ q-ref) | $0.85\%,\; 2.48\%$ ($L_3$ vs $R$); $3.89\%$ jump $L_2 \to L_3$ | Grid-alignment oscillation across curved wall |
# | Under-resolved staircase ($0.95\;\text{cells/gap}$) | $\|t\|^2 = 0.679940,\; \|\kappa\|^2 = 0.301094$ ($39.8\%$ err) | $\|t\|^2 = 0.680790,\; \|\kappa\|^2 = 0.309800$ ($35.6\%$ err vs $R$) | Fails $< 4\;\text{cells/feature}$ rule |
# | Guided power sum $\sum_j \|S_{j1}(1550)\|^2$ | $0.9758$ ($0.107\;\text{dB}$ loss) | $0.9944$ at $1550\;\text{nm}$ ($0.9966$ band mean; flux box $0.9993$) | $\le 1.0$ (passive unitary limit) |
# | Backward ring $\|S_{41}(1550)\|^2$ (CPML extruded) | $1.89 \times 10^{-5}$ ($-47.2\;\text{dB}$) | $1.02 \times 10^{-5}$ ($-49.9\;\text{dB}$ at $L_3$); $6.90 \times 10^{-7}$ ($-61.6\;\text{dB}$ at $R$) | $0$ (ideal reflectionless termination) |
# | Backward ring $\|S_{41}(1550)\|^2$ (unextruded CPML) | $\sim 1.3 \times 10^{-2}$ ($-18.8\;\text{dB}$) | $1.33 \times 10^{-2}$ ($-18.8\;\text{dB}$; band max $1.62 \times 10^{-2}$) | Fresnel reflection at truncated core termination |
# | Bus reciprocity $\max \|\|S_{21}\| - \|S_{12}\|\|$ | $< 10^{-6}$ | $2.19 \times 10^{-7}$ (mirror $\|\|S_{31}\| - \|S_{42}\|\| \le 1.04 \times 10^{-7}$) | $0$ (exact Lorentz reciprocity) |
# | Cross-axis reciprocity $\max \|\|S_{31}\| - \|S_{13}\|\|$ | $\sim 3.5 \times 10^{-3}$ | $3.51 \times 10^{-3}$ ($2.66 \times 10^{-3}$ at $1550\;\text{nm}$) | $0$ (continuous); $\mathcal{O}(10^{-3})$ $x$- vs $y$-port discretization |
# | Gap sweep $\|\kappa(g)\|^2$, $g = 100\;\text{nm}$ | $0.079064$ ($L_2$) | $0.088533$ ($L_3$) | $0.078862$ (5-layer SM); $0.071685$ (Iso-CMT) |
# | Gap sweep $\|\kappa(g)\|^2$, $g = 150\;\text{nm}$ | $0.032321$ ($L_2$) | $0.036717$ ($L_3$) | $0.033155$ (5-layer SM); $0.031914$ (Iso-CMT) |
# | Gap sweep $\|\kappa(g)\|^2$, $g = 200\;\text{nm}$ | $0.013240$ ($L_2$) | $0.015826$ ($L_3$) | $0.014314$ (5-layer SM); $0.014099$ (Iso-CMT) |
# | Gap sweep $\|\kappa(g)\|^2$, $g = 250\;\text{nm}$ | $0.005433$ ($L_2$) | $0.006666$ ($L_3$) | $0.006244$ (5-layer SM); $0.006207$ (Iso-CMT) |
# | Amplitude decay constant $\gamma_{\text{fit}}$ ($\arcsin\|\kappa\|$) | $9.31\;\mu\text{m}^{-1}$ ($R^2 = 0.9989$) | $8.99\;\mu\text{m}^{-1}$ ($8.994\;\mu\text{m}^{-1}$, $R^2 = 0.9988$) | $8.81\;\mu\text{m}^{-1}$ (5-layer SM, $2.1\%$ diff); $8.23\;\mu\text{m}^{-1}$ ($\gamma_{\text{evan}}$); $9.95\;\mu\text{m}^{-1}$ ($\gamma_{\text{slab}}$) |
# | Power decay constant $2\gamma_{\text{fit}}$ ($\|\kappa(g)\|^2$) | $18.35\;\mu\text{m}^{-1}$ | $17.61\;\mu\text{m}^{-1}$ ($\gamma = 8.81\;\mu\text{m}^{-1}$, $R^2 = 0.9994$) | $17.30\;\mu\text{m}^{-1}$ (5-layer SM, $1.8\%$ diff); $16.20\;\mu\text{m}^{-1}$ (Iso-CMT) |
# | Synthesized all-pass ring $\text{FSR}, \mathcal{F}, Q_L^\ddagger$ | $23.91\;\text{nm},\; 20.4,\; 1325$ | $24.14\;\text{nm},\; 20.7,\; 1328$ ($Q_c = 1531, Q_i = 10030$) | $\text{FSR} = \lambda_0^2 / (n_g L_{\text{rt}})$ [1, 7] |
#
# $^\ddagger$In `ring_resonator_metrics`, the intrinsic quality factor $Q_i$,
# finesse $\mathcal{F}$, and loaded $Q_L$ depend on the user-chosen closed-ring
# round-trip propagation loss parameter (`round_trip_loss_db = 0.15 dB` default),
# whereas $\text{FSR}$ and $Q_c$ depend only on the simulated waveguide group
# index $n_g$ and unitary point-coupler split ratio.
#
# ## 8. Exercises
#
# 1. **Critical coupling gap**: Using `ring_resonator_metrics` in `_helpers.py`,
#    find the coupling gap $g_{\text{crit}}$ at which the single-pass through
#    transmission $|t(g_{\text{crit}})|$ equals the round-trip cavity attenuation
#    $a_{\text{rt}}$ for a closed ring with $0.5\;\text{dB}$ round-trip loss, and
#    verify that the on-resonance transmission drops below $-40\;\text{dB}$.
# 2. **Ring radius scaling ($L_{\text{eff}} \propto \sqrt{r}$)**: Modify `R_RING`
#    over $r \in \{3.0, 5.0, 10.0, 20.0\}\;\mu\text{m}$ at a fixed gap
#    $g_0 = 150\;\text{nm}$ and verify the coupled-mode scaling law
#    $\arcsin|\kappa(r)| \propto \sqrt{r}$.
# 3. **Angled drop monitor vs polar angle $\theta$**: Use
#    `sample_angled_drop_slice` at $\theta \in \{30^\circ, 45^\circ, 60^\circ, 75^\circ\}$
#    along the ring arc and plot the radial mode centroid shift vs $\theta$.
#
# ## 9. References
#
# 1. A. Yariv, "Universal relations for coupling of optical power between
#    microresonators and dielectric waveguides," *Electron. Lett.* **36**(4),
#    321–322 (2000), DOI: [10.1049/el:20000340](https://doi.org/10.1049/el:20000340).
# 2. B. E. Little, S. T. Chu, H. A. Haus, J. Foresi, and J.-P. Laine, "Microring
#    resonator channel dropping filters," *J. Lightwave Technol.* **15**(6),
#    998–1005 (1997), DOI: [10.1109/50.588673](https://doi.org/10.1109/50.588673).
# 3. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing of
#    discretized numerical methods," *Phys. Rev. E* **77**, 036611 (2008),
#    DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
# 4. M. Heiblum and J. H. Harris, "Analysis of curved optical waveguides by
#    conformal transformation," *IEEE J. Quantum Electron.* **11**(2), 75–83
#    (1975), DOI: [10.1109/JQE.1975.1068563](https://doi.org/10.1109/JQE.1975.1068563).
# 5. H. H. Li, "Refractive index of silicon and germanium and its wavelength and
#    temperature derivatives," *J. Phys. Chem. Ref. Data* **9**, 561–658 (1980),
#    DOI: [10.1063/1.555624](https://doi.org/10.1063/1.555624).
# 6. I. H. Malitson, "Interspecimen comparison of the refractive index of fused
#    silica," *J. Opt. Soc. Am.* **55**, 1205–1209 (1965),
#    DOI: [10.1364/JOSA.55.001205](https://doi.org/10.1364/JOSA.55.001205).
# 7. W. Bogaerts *et al.*, "Silicon microring resonators," *Laser Photonics Rev.*
#    **6**(1), 47–73 (2012),
#    DOI: [10.1002/lpor.201100017](https://doi.org/10.1002/lpor.201100017).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="02. Silicon Waveguide-to-Ring Directional Point Coupler (2D EIM)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run coarse 2D study locally and render all 15 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run converged 2D study across all tasks and render all 15 figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study point I (0-indexed) at --full resolution (or --quick if combined)",
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
        help="Read data/task_*.npz and generate all 15 figures in figures/",
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
