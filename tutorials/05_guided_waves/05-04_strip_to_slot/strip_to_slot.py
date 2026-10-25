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
# # Tutorial 5.4 · Strip-to-slot waveguide mode converters
#
# **Abstract.** Silicon slot waveguides confine optical power inside a
# sub-wavelength low-index ($\text{SiO}_2$) gap flanked by two high-index
# silicon rails, exploiting the normal electric-field discontinuity
# $\Delta E_y \propto (n_{\text{high}} / n_{\text{low}})^2$ across the slot
# boundaries to enhance light–matter interaction for electro-optic modulation,
# nonlinear optics, and biochemical sensing (Almeida *et al.*, 2004 [4]).
# Practical photonic integrated circuits (PICs), however, route signals in
# low-loss single-mode strip waveguides, requiring compact, broadband,
# low-reflection **strip-to-slot mode converters**. In this tutorial we model
# and compare three classic silicon-on-insulator (SOI) strip-to-slot converter
# topologies in Wakis using the **2D effective-index method (EIM)** and
# **Kottke (2008) subpixel permittivity smoothing**:
#
# 1. **Design 1 — Symmetric Multimode Interference (MMI) Converter** ($L = 6.0\;\mu\text{m}$, Deng *et al.*, 2014 [1]), free of sub-100 nm sharp tips;
# 2. **Design 2 — Symmetric Tapered-Finger (Double-Tip) Coupler** ($L = 8.0\;\mu\text{m}$, Wang *et al.*, 2009 [2]);
# 3. **Design 3 — Asymmetric Lateral Taper** ($L = 20.0\;\mu\text{m}$, Feng *et al.*, 2007 [3]).
#
# We verify the normal-field discontinuity ratio across the $100\;\text{nm}$
# oxide slot against the exact 5-layer analytical eigenvalue solution, extract
# broadband $1500\text{--}1600\;\text{nm}$ S-parameters and 2D Poynting flux
# budgets, sweep the converter length $L$ to demonstrate adiabatic mode
# evolution, quantify the convergence advantage of Kottke subpixel
# smoothing over staircasing when resolving a $100\;\text{nm}$ slot, and
# upgrade the Cartesian mesh to a **graded non-uniform, non-square Cartesian
# grid** (`NonUniformStripToSlotPIC`, `kottke_nonuniform`) that achieves equal
# accuracy ($T(1550\;\text{nm}) = 97.57\%$, within $8.4\times 10^{-4}$ of the
# $40\;\text{cpw}$ `kottke_reference`) with $38.1\%$ fewer cells ($40{,}737$ vs
# $65{,}780$) and $58.6\%$ shorter runtime ($60.62\;\text{s}$ vs $146.50\;\text{s}$).
#
# ![Setup geometry schematic](figures/00a_setup_geometry.png)
# ![Setup mesh and excitation](figures/00b_setup_mesh_and_excitation.png)
#
# The primary simulation setup models the Deng *et al.* (2014) symmetric MMI strip-to-slot converter ($L = 6.0\;\mu\text{m}$) on a 2D effective-index grid with 10-cell CPML boundaries, an input strip $\text{TE}_0$ mode source, input/output mode monitors, a 4-wall Poynting flux box, and 8 transverse cells across the $g = 100\;\text{nm}$ central oxide slot:
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 11.9 µm × 4.23 µm × 25.6 nm (z ∈ [-12.8 nm, 12.8 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 366 × 110 × 1 (40260 cells) |
# | Discretization | Δx & resolution | Δ_min = 12.5 nm, 42.3 cells/λ (in n_max = 2.93), 8.0 cells across feature (100 nm) |
# | Boundary | -x face | CPML (10 cells, 337.4 nm) |
# | Boundary | +x face | CPML (10 cells, 338 nm) |
# | Boundary | -y face | CPML (10 cells, 636 nm) |
# | Boundary | +y face | CPML (10 cells, 636 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | SiO₂ cladding & 100 nm slot (ε_r = 2.07) (isotropic) |
# | Material | Region 1: Si MMI core (ε_r = 8.59) | ε_r = 8.59, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-3 µm, 9 µm] (d = 12 µm) |
# | Source | 1 · ModeSource | pos: x = -1.60 µm, y ∈ [-0.90, +0.90] µm, pol: TE₀ (Ey, Hz, +x), pulse: f₀ = 193.6 THz, Δf = 6.2 THz |
# | Monitor | 2 · ModeMonitor | Port 'in' ModeMonitor (S₁₁), pos: x = -1.55 µm, y ∈ [-0.90, +0.90] µm, freqs: 187.4–199.9 THz (51 DFT pts) |
# | Monitor | 3 · ModeMonitor | Port 'out' ModeMonitor (slot TE₀, S₂₁), pos: x = 7.55 µm, y ∈ [-0.90, +0.90] µm, freqs: 187.4–199.9 THz (51 DFT pts) |
# | Monitor | 4 · FluxMonitor | 2D Poynting FluxMonitor box, pos: x ∈ [-2.87, 8.87] µm, y ∈ [-2.07, 2.07] µm, freqs: 187.4–199.9 THz (51 DFT pts) |
# | Variation | Design 2: Symmetric Tapered-Finger (Wang et al. 2009) | param=L = 8.0 µm, d_finger = 60 nm, g_slot = 100 nm, note=Three-PolySlab double-tip finger coupler; T(1550 nm) = 99.80%, N = 8 cells across 60 nm finger gap (Δy_min = 7.14 nm) |
# | Variation | Design 3: Asymmetric Lateral Taper (Feng et al. 2007) | param=L = 20.0 µm, g_slot = 100 nm, note=Two-PolySlab continuous upper rail + tapered lower rail; T(1550 nm) = 99.66%, N = 8 cells across 100 nm slot |
# | Variation | Design 1 MMI length sweep | param=L ∈ {1.6, 2.2, 3.0, 4.5, 6.0} µm (L_mmi = 1.38 µm), note=Coupled-mode adiabatic saturation of slot-rail taper section L_taper = L - L_mmi |
# | Variation | Kottke vs staircase convergence ladder | param=Under-resolved (10, 0), L1=(10, 4), L2=(14, 6), L3=(20, 8), R=(40, 16), note=Compares Kottke subpixel smoothing against staircase snapping across 100 nm oxide slot |
# | Variation | Graded non-uniform mesh upgrade (kottke_nonuniform) | param=367 × 111 × 1 = 40,737 cells (-38.1% vs uniform L3), note=Δy = 12.5 nm (8 cells across 100 nm slot) with non-square Δx ≤ 34 nm along taper |
#
# ![Taper converter setup schematics](figures/00b_taper_converters_schematic.png)
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Explain the electromagnetic mechanism of **slot waveguide field enhancement**
#    ($D_y = \varepsilon_0 \varepsilon_r E_y$ continuity across high-index-contrast
#    sidewalls) and verify the $(n_{\text{eff,slab}}/n_{\text{SiO}_2})^2$ normal
#    electric-field jump against the exact 5-layer analytical slot mode
#    (Almeida *et al.*, 2004 [4]).
# 2. Construct three distinct strip-to-slot converter topologies (`build_mmi_converter`,
#    `build_symmetric_taper_converter`, `build_asymmetric_taper_converter`) using
#    `PolySlab`, `straight`, and `asymmetric_taper`, and inspect their labelled
#    2D EIM setup schematics (`00a_setup_geometry.png`,
#    `00b_setup_mesh_and_excitation.png`, `00b_taper_converters_schematic.png`).
# 3. Run broadband 2D EIM time-domain FIT simulations (`PICSimulation`) with
#    `ModeSource` and `ModeMonitor` ports that resolve the discontinuous
#    transverse slot mode profile across the $1500\text{--}1600\;\text{nm}$ C-band.
# 4. Quantify conversion efficiency vs. converter length $L$, audit energy
#    conservation with a 2D Poynting `FluxMonitor` box, compare Kottke
#    subpixel smoothing against staircasing across a feature-resolved
#    mesh-override ladder (`L1=(10, 4)`, `L2=(14, 6)`, `L3=(20, 8)`, and full
#    reference `R=(40, 16)`) as well as an under-resolved $10\;\text{cpw}$
#    warning demonstration pair (`1.96` cells across the slot), and build a
#    graded non-uniform Cartesian mesh (`kottke_nonuniform`, `04b_mesh_upgrade_comparison.png`).
#
# ## Contents
#
# 1. [Physics of slot waveguides & mode conversion](#1.-Physics-of-slot-waveguides-&-mode-conversion)
# 2. [Effective-Index Method (EIM) reduction](#2.-Effective-Index-Method-(EIM)-reduction)
# 3. [Parametric geometry of the three converter topologies](#3.-Parametric-geometry-of-the-three-converter-topologies)
# 4. [Resolution: cells per wavelength, cells across the smallest feature & subpixel smoothing](#4.-Resolution:-cells-per-wavelength,-cells-across-the-smallest-feature-&-subpixel-smoothing)
# 5. [Study tasks & execution modes](#5.-Study-tasks-&-execution-modes)
# 6. [Generating the complete 20-figure diagnostic suite](#6.-Generating-the-complete-20-figure-diagnostic-suite)
# 7. [Limits, failure modes & when 2D EIM differs from 3D FDTD](#7.-Limits,-failure-modes-&-when-2D-EIM-differs-from-3D-FDTD)
# 8. [Exercises](#8.-Exercises)
# 9. [References](#9.-References)

# %% [markdown]
# ## 1. Physics of slot waveguides & mode conversion
#
# ### 1.1 Normal-field discontinuity in a 5-layer slot waveguide
#
# Consider a symmetric dual-rail slot waveguide oriented along $x$, consisting
# of two silicon rails of width $w_{\text{rail}} = 260\;\text{nm}$ separated by a
# central $\text{SiO}_2$ slot of width $g = 100\;\text{nm}$ (total outer width
# $w_{\text{slot}} = 2 w_{\text{rail}} + g = 620\;\text{nm}$). For the
# fundamental quasi-$\text{TE}_0$ mode (dominant transverse electric field $E_y$
# polarized normal to the vertical slot walls at $y = \pm g/2 = \pm 50\;\text{nm}$,
# magnetic field $H_z$ tangential), Maxwell's boundary conditions require
# continuity of $H_z(y)$ and of the normal displacement component
# $D_y(y) = \varepsilon_0 n^2(y) E_y(y)$ across $y = \pm g/2$:
#
# $$\frac{E_y\left(|y| \to \tfrac{g}{2}^-\right)}{E_y\left(|y| \to \tfrac{g}{2}^+\right)} = \left(\frac{n_{\text{high}}}{n_{\text{low}}}\right)^2.$$
#
# Because $g = 100\;\text{nm} \ll \lambda_0 / (2 n_{\text{SiO}_2})$, the
# hyperbolic-cosine decay $\cosh(\gamma_s y)$ inside the slot ($|y| \le g/2$)
# varies by less than $3\%$ from the walls to $y = 0$, sustaining a high
# electric field across the entire low-index gap [4].
#
# ### 1.2 Three strip-to-slot mode conversion mechanisms
#
# Direct butt-coupling between a $w_{\text{strip}} = 400\;\text{nm}$ strip
# waveguide (Gaussian-like single-peaked $E_y(y)$) and the $w_{\text{slot}} = 620\;\text{nm}$
# slot waveguide (intense central slot spike flanked by two rail lobes) suffers
# severe modal mismatch and radiation loss. Three distinct physical strategies
# overcome this mismatch:
#
# 1. **Symmetric MMI + Taper Converter (Deng *et al.*, 2014 [1])**: The
#    $400\;\text{nm}$ strip steps into a $W_{\text{mmi}} = 1.24\;\mu\text{m}$,
#    $L_{\text{mmi}} = 1.38\;\mu\text{m}$ multimode section that excites both
#    the fundamental $\text{TE}_0$ and second-order symmetric $\text{TE}_2$ slab
#    modes. At $x = L_{\text{mmi}}$ ($\approx L_\pi / 2$), their interference
#    concentrates field intensity near $y = 0$ where the $100\;\text{nm}$ slot
#    opens abruptly, and a linear outer-boundary taper over $x \in [1.38, 6.0]\;\mu\text{m}$
#    compresses the outer rails down to $w_{\text{slot}} = 620\;\text{nm}$
#    without any sub-100 nm sharp tips.
# 2. **Symmetric Tapered-Finger Coupler (Wang *et al.*, 2009 [2])**: Over
#    $L_{\text{taper}} = 8.0\;\mu\text{m}$, the central input strip tapers
#    linearly down to a sharp tip at $(L_{\text{taper}}, 0)$ while two outer
#    silicon fingers—starting as sharp tips offset laterally by $d = 60\;\text{nm}$
#    at $x = 0$—expand and converge inward to form the two $260\;\text{nm}$ slot
#    rails separated by $g = 100\;\text{nm}$.
# 3. **Asymmetric Lateral Taper (Feng *et al.*, 2007 [3])**: Over an adiabatic
#    length $L = 20.0\;\mu\text{m}$, the input strip continuously shifts and
#    narrows into the upper slot rail ($y \in [+50, +310]\;\text{nm}$) while a
#    lower silicon rail grows linearly from a sharp tip at $(0, -300\;\text{nm})$
#    into the lower slot rail ($y \in [-310, -50]\;\text{nm}$).

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
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from scipy.constants import c as c_light, epsilon_0

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _helpers import (  # noqa: E402
    NonUniformStripToSlotPIC,
    asymmetric_taper,
    build_mmi_graded_axes,
    overlay_converter_geometry,
    plot_mesh_upgrade_comparison,
    plot_setup_schematic,
    plot_taper_converters_schematic,
    render_field_map_with_overlay,
    render_movie_with_overlay,
    slot_straight,
    slot_waveguide_mode,
    slot_waveguide_neff,
)
from wakis import figstyle  # noqa: E402
from wakis.templates.photonics import (  # noqa: E402
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
    plot_port_modes,
    plot_spectra_db,
    slab_neff,
    straight,
)

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# %% [markdown]
# ### Device parameters (Deng *et al.* 2014 [1], Wang *et al.* 2009 [2], Feng *et al.* 2007 [3])
#
# All geometric dimensions are specified in metres (SI units):
#
# | Parameter | Symbol | Value |
# |---|---|---|
# | Silicon core refractive index ($1550\;\text{nm}$) | $n_{\text{Si}}$ | $3.47$ ($\varepsilon_r = 12.0409$) |
# | $\text{SiO}_2$ cladding refractive index ($1550\;\text{nm}$) | $n_{\text{SiO}_2}$ | $1.44$ ($\varepsilon_r = 2.0736$) |
# | Silicon layer thickness | $h_{\text{Si}}$ | $250\;\text{nm}$ |
# | Input strip waveguide width | $w_{\text{strip}}$ | $400\;\text{nm}$ |
# | Output slot waveguide outer width | $w_{\text{slot}}$ | $620\;\text{nm}$ |
# | Central $\text{SiO}_2$ slot width | $g$ | $100\;\text{nm}$ |
# | Single silicon rail width $(w_{\text{slot}} - g)/2$ | $w_{\text{rail}}$ | $260\;\text{nm}$ |
# | Design 1 MMI width / length / total length | $W_{\text{mmi}}, L_{\text{mmi}}, L_1$ | $1.24\;\mu\text{m}, 1.38\;\mu\text{m}, 6.0\;\mu\text{m}$ |
# | Design 2 symmetric taper length / initial tip gap | $L_2, d$ | $8.0\;\mu\text{m}, 60\;\text{nm}$ |
# | Design 3 asymmetric lateral taper length | $L_3$ | $20.0\;\mu\text{m}$ |

# %%
# Material refractive indices and SOI thickness (250 nm Si on SiO2)
N_SI = 3.47
N_SIO2 = 1.44
H_SI = 250e-9
EPS_SI = n_to_eps(N_SI)

# Wavelength band: 1500 nm to 1600 nm (C-band, centered at 1550 nm)
WL_MIN = 1500e-9
WL_MAX = 1600e-9
WL_REF = 1550e-9
N_FREQS = 51

# Common strip and slot cross-section dimensions
W_STRIP = 0.40e-6
W_SLOT = 0.62e-6
GAP_SLOT = 0.10e-6
W_RAIL = 0.5 * (W_SLOT - GAP_SLOT)  # 0.26 um
L_LEAD = 3.00e-6  # lead-in / lead-out straight waveguide length
X_PORT_IN = -1.55e-6
PORT_OUT_OFFSET = 1.55e-6
X_PAD = 2.60e-6

# Design 1 (Deng et al. 2014): Symmetric MMI converter
W_MMI = 1.24e-6
L_MMI = 1.38e-6
L_CONV_MMI = 6.00e-6
Y_HALF_MMI = 1.48e-6

# Design 2 (Wang et al. 2009): Symmetric double-tip tapered-finger coupler
L_CONV_SYM = 8.00e-6
GAP_FINGER = 0.06e-6
Y_HALF_SYM = 1.50e-6

# Design 3 (Feng et al. 2007): Asymmetric lateral taper
L_CONV_ASYM = 20.00e-6
Y_HALF_ASYM = 1.50e-6

# Converter length sweep values [m] for Design 1 (L = L_mmi + L_taper)
SWEEP_LENGTHS_UM = (1.6, 2.2, 3.0, 4.5, 6.0)

# %% [markdown]
# ## 2. Effective-Index Method (EIM) reduction
#
# In the **2D effective-index method (EIM)**, we factor the 3D quasi-TE field as
# $\mathbf{E}(x, y, z) \approx \psi(z)\,\mathbf{E}_{\parallel}(x, y)$ by solving
# the 1D vertical slab waveguide equation across the $\text{SiO}_2 / \text{Si}(250\;\text{nm}) / \text{SiO}_2$
# stack at $\lambda_0 = 1550\;\text{nm}$:
#
# $$n_{\text{eff,slab}}(1550\;\text{nm}) = 2.9306, \qquad \varepsilon_{\text{eff,slab}} = n_{\text{eff,slab}}^2 = 8.5883.$$
#
# In the 2D in-plane FIT simulation (`Ex`, `Ey`, `Hz`; one periodic cell along
# $z$ with `ieps_z = 0`), the etched silicon rails have refractive index
# $n_{\text{high}} = n_{\text{eff,slab}} = 2.9306$ while the cladding and the
# $100\;\text{nm}$ central slot have index $n_{\text{low}} = n_{\text{SiO}_2} = 1.44$.
# Consequently, the theoretical normal electric-field discontinuity ratio at the
# slot walls $y = \pm g/2 = \pm 50\;\text{nm}$ in 2D EIM is:
#
# $$\frac{E_y\left(|y| \to 50\;\text{nm}^-\right)}{E_y\left(|y| \to 50\;\text{nm}^+\right)} = \left(\frac{n_{\text{eff,slab}}}{n_{\text{SiO}_2}}\right)^2 = \left(\frac{2.9306}{1.44}\right)^2 = 4.1417,$$
#
# and solving the exact 5-layer transcendental dispersion relation
# (`slot_waveguide_neff`) yields an analytical 2D EIM slot-mode effective index
# $n_{\text{eff,slot}} = 2.1563$ (with $57.2\%$ of $\int |E_y|^2\,\mathrm{d}y$
# confined inside the $100\;\text{nm}$ oxide gap and a slot-center-to-rail-center
# field ratio $|E_y(0)| / |E_y(180\;\text{nm})| = 2.7844$). At the `--full`
# primary resolution `L3 = (20, 8)` ($\Delta y_{\min} = 12.50\;\text{nm}$, $8.00$
# cells across the slot), Wakis `solve_port_modes` yields
# $n_{\text{eff,slot}} = 2.1626$ ($n_{g,\text{slot}} = 2.8000$) and
# $|E_y(0)| / |E_y(180\;\text{nm})| = 2.7839$, while on the $w_{\text{strip}} = 400\;\text{nm}$
# input strip port it yields $n_{\text{eff,strip}} = 2.4051$ ($n_{g,\text{strip}} = 3.3590$,
# vs the 1D 3-layer analytical value $2.4007$).

# %% [markdown]
# ## 3. Parametric geometry of the three converter topologies
#
# We construct all three topologies using `PolySlab`, `straight`,
# `asymmetric_taper`, and `slot_straight`:
#
# 1. `build_mmi_converter`: builds the Deng *et al.* (2014) converter as a
#    single seamless 16-vertex counter-clockwise `PolySlab`.
# 2. `build_symmetric_taper_converter`: builds the Wang *et al.* (2009)
#    three-polygon coupler (`strip_taper` + `slot_top` + `slot_bottom`).
# 3. `build_asymmetric_taper_converter`: builds the Feng *et al.* (2007)
#    two-polygon asymmetric adiabatic transformer (`converter_top` + `converter_bottom`).


# %%
def build_mmi_converter(
    length_conv=L_CONV_MMI,
    length_mmi=L_MMI,
    width_mmi=W_MMI,
    width_strip=W_STRIP,
    width_slot=W_SLOT,
    gap_slot=GAP_SLOT,
):
    """
    Build Design 1 — the Deng et al. (2014) symmetric MMI strip-to-slot mode
    converter as a single 16-vertex `PolySlab`, together with its input/output
    `Port` list and 2D domain bounds `(xmin, xmax, ymin, ymax)` [m].
    """
    L_c = float(length_conv)
    L_m = float(length_mmi)
    if L_c <= L_m:
        raise ValueError("Require length_conv > length_mmi.")
    hs = 0.5 * float(width_strip)
    hm = 0.5 * float(width_mmi)
    hsl = 0.5 * float(width_slot)
    hg = 0.5 * float(gap_slot)
    x_in = -L_LEAD
    x_out = L_c + L_LEAD

    verts = np.array(
        [
            [x_in, -hs],
            [0.0, -hs],
            [0.0, -hm],
            [L_m, -hm],
            [L_c, -hsl],
            [x_out, -hsl],
            [x_out, -hg],
            [L_m, -hg],
            [L_m, +hg],
            [x_out, +hg],
            [x_out, +hsl],
            [L_c, +hsl],
            [L_m, +hm],
            [0.0, +hm],
            [0.0, +hs],
            [x_in, +hs],
        ],
        dtype=float,
    )
    slab = PolySlab(verts, z_min=-0.5 * H_SI, z_max=0.5 * H_SI, material=EPS_SI)
    ports = [
        Port(
            "in",
            center=(X_PORT_IN, 0.0, 0.0),
            size=(0.0, 1.80e-6, 1.50e-6),
            direction=+1,
        ),
        Port(
            "out",
            center=(L_c + PORT_OUT_OFFSET, 0.0, 0.0),
            size=(0.0, 1.80e-6, 1.50e-6),
            direction=+1,
        ),
    ]
    domain = (-X_PAD, L_c + X_PAD, -Y_HALF_MMI, +Y_HALF_MMI)
    return [slab], ports, domain


def build_symmetric_taper_converter(
    length_taper=L_CONV_SYM,
    gap_finger=GAP_FINGER,
    width_strip=W_STRIP,
    width_slot=W_SLOT,
    gap_slot=GAP_SLOT,
):
    """
    Build Design 2 — the Wang et al. (2009) symmetric double-tip tapered-finger
    strip-to-slot coupler as three `PolySlab` structures (`strip_taper`,
    `slot_top`, `slot_bottom`), together with its `Port` list and 2D domain.
    """
    L_t = float(length_taper)
    d = float(gap_finger)
    hs = 0.5 * float(width_strip)
    hsl = 0.5 * float(width_slot)
    hg = 0.5 * float(gap_slot)
    x_in = -L_LEAD
    x_out = L_t + L_LEAD

    # 1. Input strip + linear taper to a central tip at (L_t, 0)
    v_in = straight((x_in, 0.0), (0.0, 0.0), width=width_strip)
    v_tip = asymmetric_taper(0.0, L_t, -hs, +hs, 0.0, 0.0)
    v_strip = np.vstack([v_in[:1], [[0.0, -hs]], v_tip[1:2], [[0.0, +hs]], v_in[3:]])

    # 2. Upper and lower slot rails tapering from outer sharp tips at (0, +- (hs + d))
    v_rail_bot, v_rail_top = slot_straight(L_t, x_out, width_slot, gap_slot)
    v_taper_top = asymmetric_taper(0.0, L_t, hs + d, hs + d, +hg, +hsl)
    v_top = np.vstack([v_taper_top[:2], v_rail_top[1:3], v_taper_top[2:]])

    v_taper_bot = asymmetric_taper(0.0, L_t, -(hs + d), -(hs + d), -hsl, -hg)
    v_bot = np.vstack([v_taper_bot[:2], v_rail_bot[1:3], v_taper_bot[2:]])

    slabs = [
        PolySlab(v, z_min=-0.5 * H_SI, z_max=0.5 * H_SI, material=EPS_SI)
        for v in (v_strip, v_top, v_bot)
    ]
    ports = [
        Port(
            "in",
            center=(X_PORT_IN, 0.0, 0.0),
            size=(0.0, 1.80e-6, 1.50e-6),
            direction=+1,
        ),
        Port(
            "out",
            center=(L_t + PORT_OUT_OFFSET, 0.0, 0.0),
            size=(0.0, 1.80e-6, 1.50e-6),
            direction=+1,
        ),
    ]
    domain = (-X_PAD, L_t + X_PAD, -Y_HALF_SYM, +Y_HALF_SYM)
    return slabs, ports, domain


def build_asymmetric_taper_converter(
    length_taper=L_CONV_ASYM,
    width_strip=W_STRIP,
    width_slot=W_SLOT,
    gap_slot=GAP_SLOT,
):
    """
    Build Design 3 — the Feng et al. (2007) asymmetric lateral taper
    strip-to-slot converter as two `PolySlab` structures (`converter_top`,
    `converter_bottom`), together with its `Port` list and 2D domain.
    """
    L_t = float(length_taper)
    hs = 0.5 * float(width_strip)
    hsl = 0.5 * float(width_slot)
    hg = 0.5 * float(gap_slot)
    x_in = -L_LEAD
    x_out = L_t + L_LEAD

    # Upper rail continuous with the input strip waveguide
    v_top = np.array(
        [
            [x_in, -hs],
            [0.0, -hs],
            [L_t, +hg],
            [x_out, +hg],
            [x_out, +hsl],
            [L_t, +hsl],
            [0.0, +hs],
            [x_in, +hs],
        ],
        dtype=float,
    )
    # Lower rail starting from a sharp tip at (0, -hs - gap_slot) = (0, -0.30 um)
    v_taper_bot = asymmetric_taper(0.0, L_t, -hs - gap_slot, -hs - gap_slot, -hsl, -hg)
    v_rail_bot, _ = slot_straight(L_t, x_out, width_slot, gap_slot)
    v_bot = np.vstack([v_taper_bot[:2], v_rail_bot[1:3], v_taper_bot[2:]])

    slabs = [
        PolySlab(v, z_min=-0.5 * H_SI, z_max=0.5 * H_SI, material=EPS_SI)
        for v in (v_top, v_bot)
    ]
    ports = [
        Port(
            "in",
            center=(X_PORT_IN, 0.0, 0.0),
            size=(0.0, 1.80e-6, 1.50e-6),
            direction=+1,
        ),
        Port(
            "out",
            center=(L_t + PORT_OUT_OFFSET, 0.0, 0.0),
            size=(0.0, 1.80e-6, 1.50e-6),
            direction=+1,
        ),
    ]
    domain = (-X_PAD, L_t + X_PAD, -Y_HALF_ASYM, +Y_HALF_ASYM)
    return slabs, ports, domain


def build_design(design_key, length_conv=None):
    """Dispatch builder for `'mmi'`, `'sym_taper'`, or `'asym_taper'`."""
    if design_key == "mmi":
        L_c = L_CONV_MMI if length_conv is None else float(length_conv)
        return build_mmi_converter(length_conv=L_c)
    if design_key == "sym_taper":
        L_c = L_CONV_SYM if length_conv is None else float(length_conv)
        return build_symmetric_taper_converter(length_taper=L_c)
    if design_key == "asym_taper":
        L_c = L_CONV_ASYM if length_conv is None else float(length_conv)
        return build_asymmetric_taper_converter(length_taper=L_c)
    raise ValueError(f"Unknown design_key {design_key!r}.")


# %% [markdown]
# ## 4. Resolution: cells per wavelength, cells across the smallest feature & subpixel smoothing
#
# ### 4.1 Resolution: cells per wavelength and cells across the smallest feature
#
# Accurate FDTD/FIT modeling of sub-wavelength photonic structures requires
# satisfying two independent spatial discretization rules simultaneously:
#
# 1. **Wavelength sampling rule ($\text{cpw} \ge 10$)**: The background grid
#    must resolve the shortest guided wavelength inside the highest-index
#    material ($n_{\text{eff,slab}} = 2.9306$ at $\lambda_{\text{ref}} = 1550\;\text{nm}$)
#    with at least $10$ cells per wavelength in every execution mode (`--quick`,
#    `--full`, and parameter sweeps):
#    $$\Delta x_{\text{bg}} = \frac{\lambda_{\min}}{\text{cpw}\cdot n_{\text{eff,slab}}} \le 51.2\;\text{nm} \quad (\text{at }\lambda_{\min} = 1500\;\text{nm}).$$
# 2. **Smallest-feature sampling rule ($N \ge 4$ cells across the feature)**:
#    The grid cell size across the narrowest geometric gap or tip must be
#    smaller than the feature itself, placing at least $N \ge 4$ cells across
#    it. In our strip-to-slot converters, the smallest feature is the
#    $g = 100\;\text{nm}$ central oxide slot in Designs 1 & 3 and the
#    $d = 60\;\text{nm}$ initial finger-to-strip gap in Design 2. On an
#    unrefined uniform grid at $\text{cpw} \in \{10, 14, 20\}$, the background
#    cell size $\Delta y_{\text{bg}} \approx \{51.1, 36.5, 25.5\}\;\text{nm}$
#    places only $\{1.96, 2.74, 3.92\}$ cells across the $100\;\text{nm}$ slot
#    (and only $\{1.17, 1.64, 2.36\}$ cells across the $60\;\text{nm}$ gap in
#    Design 2)—only a uniform $40\;\text{cpw}$ grid ($7.84$ cells across the
#    slot) resolves the slot without local refinement.
#
# To satisfy $N \ge 4$ without paying the cost of a uniform fine mesh across
# the entire simulation domain, we pass **mesh-override boxes**
# `mesh_overrides=[(lo, hi, (1.0, max_hy, 1.0)), ...]` to `PICSimulation`,
# where `max_hy = min_feature / N` caps the transverse cell size $\Delta y$
# across the slot and taper-gap strips while leaving $\Delta x$ at the
# background core resolution (`hx = 1.0` m > $\Delta x_{\text{bg}}$). Because
# the tensor-product mesh refines only the narrow $y$ strip containing the slot
# (and grades smoothly back out to $\Delta y_{\text{bg}}$ at geometric ratio
# $\le 1.4$), the total 2D cell count at `L1 = (10, 4)` increases by only
# $5.5\%$ over an unrefined $10\;\text{cpw}$ grid ($20{,}003$ vs $18{,}960$
# cells in `--full` with `n_pml = 10`; $18{,}723$ vs $17{,}700$ cells, or
# $+5.8\%$, in `--quick` with `n_pml = 8`) while placing an exact $N = 4.00$
# cells ($\Delta y_{\min} = 25.00\;\text{nm}$) across the $100\;\text{nm}$
# slot. Note that because the explicit FIT time step $\Delta t$ is governed by
# the smallest cell via the Courant–Friedrichs–Lewy (CFL) condition, the number
# of time steps scales as $1/\Delta y_{\min}$.
#
# ### 4.2 Subpixel permittivity smoothing vs staircasing once the slot is resolved
#
# Two geometric features in a strip-to-slot converter interact with the
# Cartesian Yee grid:
#
# 1. **Horizontal slot walls ($y = \pm g/2 = \pm 50\;\text{nm}$)** where the
#    dominant electric field $E_y$ is normal to the $\text{Si}/\text{SiO}_2$
#    interface and jumps by $(n_{\text{eff,slab}}/n_{\text{SiO}_2})^2 = 4.14$.
# 2. **Tilted outer taper sidewalls** ($x \in [1.38, 6.0]\;\mu\text{m}$ in
#    Design 1, $x \in [0, 8.0]\;\mu\text{m}$ in Design 2, and
#    $x \in [0, 20.0]\;\mu\text{m}$ in Design 3) and outer strip/rail/MMI step
#    boundaries ($|y| \in \{200, 310, 620\}\;\text{nm}$) that cut across
#    background grid cells outside the slot override box.
#
# Diagonal **Kottke (2008) subpixel smoothing** [5] uses harmonic averaging
# for the electric-field component normal to the interface ($E_y$) and
# arithmetic averaging for the tangential component ($E_x$):
#
# $$\varepsilon_{\text{eff}, y}^{-1} = n_y^2 \left(\frac{w}{\varepsilon_s} + \frac{1 - w}{\varepsilon_b}\right) + \left(1 - n_y^2\right)\left(w\varepsilon_s + (1 - w)\varepsilon_b\right)^{-1}.$$
#
# Comparing the under-resolved demonstration pair $(\text{cpw}=10, N=0)$
# against the resolved ladder `L1 = (10, 4)`, `L2 = (14, 6)`, and `L3 = (20, 8)`
# (referenced to the independent `kottke_reference` at `R = (40, 16)`,
# $T_{\text{ref}}(1550\;\text{nm}) = 0.974864$; or to `L3 = (20, 8)`,
# $T_{\text{ref}}(1550\;\text{nm}) = 0.974667$, in `--quick`) reveals how
# resolving the slot changes—and extends—the Kottke-vs-staircase comparison:
#
# - **Under-resolved slot (`cpw = 10, N = 0`, $1.96$ cells across the $100\;\text{nm}$ slot)**:
#   Staircasing rounds the cut slot-wall cells and alters the effective slot
#   width, giving $T(1550\;\text{nm}) = 0.940042$
#   ($|T - T_{\text{ref}}| = 3.48\times 10^{-2}$, broadband $1505\text{--}1595\;\text{nm}$
#   RMS error $3.66\times 10^{-2}$), whereas Kottke gives
#   $T(1550\;\text{nm}) = 0.975463$ ($|T - T_{\text{ref}}| = 6.00\times 10^{-4}$,
#   broadband RMS $2.23\times 10^{-3}$). This produces an apparent $\sim 58\times$
#   error ratio at $1550\;\text{nm}$ ($16.4\times$ broadband RMS; $42.4\times$ at
#   $1550\;\text{nm}$ in `--quick`), which largely reflects staircasing failing
#   to represent a sub-2-cell gap.
# - **Feature-resolved slot (`L1 = (10, 4)` with $4.00$ cells, `L2 = (14, 6)` with $6.00$ cells, and `L3 = (20, 8)` with $8.00$ cells across the slot)**:
#   Once `mesh_overrides` resolves the $100\;\text{nm}$ slot and aligns the
#   slot walls $y = \pm 50\;\text{nm}$ with grid lines, **Kottke converges
#   smoothly and monotonically** across all three levels toward `R = (40, 16)`:
#   $|T(1550\;\text{nm}) - T_{\text{ref}}|$ drops from $2.57\times 10^{-3}$ at
#   `L1` ($T = 0.977437$) to $1.64\times 10^{-3}$ at `L2` ($T = 0.976502$) and
#   $1.94\times 10^{-4}$ at `L3` ($T = 0.974670$), while the broadband
#   ($1505\text{--}1595\;\text{nm}$) RMS transmission error drops from
#   $3.44\times 10^{-3}$ (`L1`) to $2.19\times 10^{-3}$ (`L2`) and
#   $3.05\times 10^{-4}$ (`L3`).
# - **Non-monotonic staircase pixel-snapping at `L3 = (20, 8)`**:
#   By contrast, staircasing's error $|T(1550\;\text{nm}) - T_{\text{ref}}|$
#   drops from $2.51\times 10^{-2}$ at `L1` ($T = 0.949791$, broadband RMS
#   $2.73\times 10^{-2}$) to $1.21\times 10^{-2}$ at `L2` ($T = 0.962805$,
#   broadband RMS $1.37\times 10^{-2}$), but **rebounds to $2.58\times 10^{-2}$
#   at `L3` ($T = 0.949087$, broadband RMS $2.09\times 10^{-2}$)** with
#   elevated band-averaged reflection $\langle|S_{11}|^2\rangle = 3.21\times 10^{-3}$
#   (vs $4.87\times 10^{-4}$ at `L2` and $2.44\times 10^{-3}$ at `L1`). Why?
#   Outside the slot override box ($|y| > 50\;\text{nm}$), the outer strip walls
#   ($|y| = 200\;\text{nm}$), outer slot-rail walls ($|y| = 310\;\text{nm}$),
#   and tilted MMI-to-slot taper walls ($|y|(x)$ tapering from $620\;\text{nm}$
#   to $310\;\text{nm}$ over $x \in [1.38, 6.0]\;\mu\text{m}$) lie on the graded
#   background grid. At `L2` ($\Delta y_{\text{bg}} = 36.30\;\text{nm}$), both
#   $|y| = 200\;\text{nm}$ ($+9.4\;\text{nm}$ from a grid line) and
#   $|y| = 310\;\text{nm}$ ($+8.3\;\text{nm}$) fall close to grid lines and
#   round in the same direction, whereas at `L3` ($\Delta y_{\text{bg}} = 25.32\;\text{nm}$),
#   the strip wall $|y| = 200\;\text{nm}$ lies $-11.27\;\text{nm}$ ($0.445\,\Delta y$,
#   nearly half a cell) from a grid line and snaps inward while the outer rail
#   wall snaps outward, and the $12$ discrete staircase steps along the tilted
#   taper walls produce a Bragg-like reflection lobe peaking near $1535\text{--}1550\;\text{nm}$
#   (visible in `15_resolution_spectra_comparison.png`). **Across the resolved
#   ladder `L1`, `L2`, and `L3`, Kottke subpixel smoothing reduces
#   $|T(1550\;\text{nm}) - T_{\text{ref}}|$ by $9.7\times$ (`L1`), $7.4\times$
#   (`L2`), and $132.9\times$ (`L3`)—and reduces broadband RMS error by
#   $7.9\times$, $6.3\times$, and $68.6\times$—relative to staircasing** (in
#   `--quick`, referenced to `L3`, the `L1` and `L2` ratios are $10.9\times$
#   and $6.5\times$).

# %% [markdown]
# ## 5. Study tasks & execution modes
#
# Each resolution level is defined by a pair $(\text{cpw}, N)$ specifying the
# background `cells_per_wavelength = cpw` and the mesh-override target
# `max_cell_y = min_feature / N` across the smallest feature:
#
# - **Under-resolved warning pair** (`underresolved_kottke`, `underresolved_staircase`):
#   $(\text{cpw} = 10, N = 0)$ with **no mesh overrides** ($1.96$ cells across
#   the $100\;\text{nm}$ slot), kept in both `--quick` and `--full` as an
#   explicit warning example ("what happens if you don't resolve the gap") and
#   excluded from converged design metrics.
# - **Resolved convergence ladder**: `L1 = (10, 4)`, `L2 = (14, 6)`,
#   `L3 = (20, 8)`, and full reference `R = (40, 16)`, plus the **graded
#   non-uniform Cartesian mesh upgrade** (`kottke_nonuniform`).
#   - In `--quick` (**14 tasks**, `task_0`..`task_13`, `n_pml = 8`): Kottke and
#     staircase run at `L1 = (10, 4)` and `L2 = (14, 6)` with the independent
#     Kottke reference at `L3 = (20, 8)`; Designs 2 & 3, the 4 MMI
#     length-sweep tasks, and `kottke_nonuniform` run at `L2 = (14, 6)`.
#   - In `--full` (**16 tasks**, `task_0`..`task_15`, `n_pml = 10`): Kottke and
#     staircase run at `L1 = (10, 4)`, `L2 = (14, 6)`, and `L3 = (20, 8)` with
#     the independent Kottke reference at `R = (40, 16)`; Designs 2 & 3, the
#     4 MMI length-sweep tasks, and `kottke_nonuniform` (`task_15`) run at
#     `L3 = (20, 8)`.
#
# ### Measured `--full` resolution, mesh, and timing summary
#
# Every task records `min_feature_nm`, `dx_background_nm`, `dx_min_nm`,
# `cells_across_feature` (measured directly from `pic.grid`), `n_cells`, and
# `n_steps` in its JSON and NPZ outputs (`data/task_0.json`..`data/task_15.json`):
#
# | Task | Name | Design | Avg | cpw | $N$ | $\text{min\_feat}$ [nm] | $\Delta x_{\text{bg}}$ [nm] | $\Delta x_{\min}$ [nm] | Cells across feature | `n_cells` | `n_steps` | Time [s] | $T(1550\;\text{nm})$ |
# |---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | `0` | `underresolved_kottke` | Design 1 MMI ($6\;\mu\text{m}$) | `kottke` | 10 | 0 | 100.0 | 51.14 | 51.03 | 1.96 *(demo)* | 18,960 | 6,201 | 11.17 | 0.9755 |
# | `1` | `underresolved_staircase` | Design 1 MMI ($6\;\mu\text{m}$) | `staircase` | 10 | 0 | 100.0 | 51.14 | 51.03 | 1.96 *(demo)* | 18,960 | 6,451 | 12.79 | 0.9400 |
# | `2` | `kottke_L1` | Design 1 MMI ($6\;\mu\text{m}$) | `kottke` | 10 | 4 | 100.0 | 50.85 | 25.00 | 4.00 | 20,003 | 8,901 | 18.43 | 0.9774 |
# | `3` | `staircase_L1` | Design 1 MMI ($6\;\mu\text{m}$) | `staircase` | 10 | 4 | 100.0 | 50.85 | 25.00 | 4.00 | 20,003 | 9,451 | 19.78 | 0.9498 |
# | `4` | `kottke_L2` | Design 1 MMI ($6\;\mu\text{m}$) | `kottke` | 14 | 6 | 100.0 | 36.46 | 16.67 | 6.00 | 35,096 | 12,901 | 41.63 | 0.9765 |
# | `5` | `staircase_L2` | Design 1 MMI ($6\;\mu\text{m}$) | `staircase` | 14 | 6 | 100.0 | 36.46 | 16.67 | 6.00 | 35,096 | 13,201 | 45.29 | 0.9628 |
# | `6` | `kottke_L3` *(primary)* | Design 1 MMI ($6\;\mu\text{m}$) | `kottke` | 20 | 8 | 100.0 | 25.51 | 12.50 | 8.00 | 65,780 | 17,501 | 146.50 | 0.9747 |
# | `7` | `staircase_L3` | Design 1 MMI ($6\;\mu\text{m}$) | `staircase` | 20 | 8 | 100.0 | 25.51 | 12.50 | 8.00 | 65,780 | 18,701 | 113.52 | 0.9491 |
# | `8` | `kottke_reference` (`R`) | Design 1 MMI ($6\;\mu\text{m}$) | `kottke` | 40 | 16 | 100.0 | 12.78 | 6.25 | 16.00 | 236,174 | 34,851 | 907.33 | 0.9749 |
# | `9` | `design2_sym_taper` | Design 2 Sym. Taper ($8\;\mu\text{m}$) | `kottke` | 20 | 8 | 60.0 | 25.56 | 7.14 | 8.00 | 102,758 | 28,251 | 407.56 | 0.9980 |
# | `10` | `design3_asym_taper` | Design 3 Asym. Taper ($20\;\mu\text{m}$) | `kottke` | 20 | 8 | 100.0 | 25.58 | 12.50 | 8.00 | 156,085 | 26,101 | 616.66 | 0.9966 |
# | `11` | `length_L1.6um` | Design 1 MMI ($1.6\;\mu\text{m}$) | `kottke` | 20 | 8 | 100.0 | 25.51 | 12.50 | 8.00 | 41,184 | 15,101 | 63.57 | 0.6101 |
# | `12` | `length_L2.2um` | Design 1 MMI ($2.2\;\mu\text{m}$) | `kottke` | 20 | 8 | 100.0 | 25.51 | 12.50 | 8.00 | 44,473 | 15,301 | 76.38 | 0.8294 |
# | `13` | `length_L3.0um` | Design 1 MMI ($3.0\;\mu\text{m}$) | `kottke` | 20 | 8 | 100.0 | 25.58 | 12.50 | 8.00 | 48,906 | 15,701 | 44.40 | 0.9673 |
# | `14` | `length_L4.5um` | Design 1 MMI ($4.5\;\mu\text{m}$) | `kottke` | 20 | 8 | 100.0 | 25.54 | 12.50 | 8.00 | 57,343 | 16,601 | 62.70 | 0.9724 |
# | `15` | `kottke_nonuniform` *(upgrade)* | Design 1 MMI ($6\;\mu\text{m}$) | `kottke` | 20 | 8 | 100.0 | 33.77 | 12.50 | 8.00 | 40,737 | 18,301 | 60.62 | 0.9757 |
#
# ### Graded non-uniform Cartesian mesh upgrade (`kottke_nonuniform`, `task_15`)
#
# In `NonUniformStripToSlotPIC` (`build_mmi_graded_axes` in `_helpers.py`), we
# replace the uniform-background grid of `kottke_L3` ($460 \times 143 \times 1 = 65{,}780$
# cells, $146.50\;\text{s}$) with a graded non-uniform, non-square Cartesian
# mesh ($367 \times 111 \times 1 = 40{,}737$ cells, **$-38.1\%$ cells**,
# $60.62\;\text{s}$ or **$-58.6\%$ wall time** vs `kottke_L3` and **$-46.6\%$**
# vs `staircase_L3`):
#
# - **Transverse ($y$) grading**: Exact $N = 8.00$ cells ($\Delta y = 12.50\;\text{nm}$)
#   across the $g = 100\;\text{nm}$ central oxide slot ($|y| \le 50\;\text{nm}$),
#   fine $\Delta y \approx 24\;\text{nm}$ across the silicon rails and MMI box
#   ($|y| \in [0.05, 0.66]\;\mu\text{m}$), and smoothly graded coarse cells up
#   to $\Delta y_{\text{clad}} \le 52\;\text{nm}$ in the outer $\text{SiO}_2$
#   cladding ($|y| \in [0.66, 1.48]\;\mu\text{m}$).
# - **Longitudinal ($x$) non-square grading**: Fine $\Delta x \approx 22\;\text{nm}$
#   across the MMI input step and abrupt slot opening ($x \in [-0.15, 1.65]\;\mu\text{m}$),
#   relaxed non-square $\Delta x \approx 34\;\text{nm}$ ($\Delta x / \Delta y_{\text{slot}} = 2.7$)
#   along the adiabatic slot-rail taper ($x \in [1.65, 6.10]\;\mu\text{m}$), and
#   $\Delta x \approx 40\;\text{nm}$ along the uniform input/output waveguide leads.
# - **Equal accuracy**: `kottke_nonuniform` achieves $T(1550\;\text{nm}) = 0.975703$
#   ($97.57\%$, insertion loss $0.1068\;\text{dB}$, return loss
#   $R(1550\;\text{nm}) = 6.19\times 10^{-4}$ or $-32.08\;\text{dB}$, C-band
#   mean $\langle T\rangle = 97.63\%$), matching `kottke_L3` ($97.47\%$) and the
#   $236{,}174$-cell `kottke_reference` ($97.49\%$) within
#   $|T - T_{\text{ref}}| = 8.39\times 10^{-4}$ ($0.0037\;\text{dB}$, see
#   `04b_mesh_upgrade_comparison.png`).
#
# ### Resolution and convergence audit (`--full` study)
#
# - **Resolution rule audit (`cpw >= 10` and `cells_across_feature >= 4`)**:
#   Reading `data/task_0.json`..`data/task_14.json` confirms **zero violations**
#   across all 13 non-demo tasks (`task_2`..`task_14`): every non-demo run has
#   $\text{cpw} \in \{10, 14, 20, 40\} \ge 10$ in the highest-index material
#   ($\Delta x_{\text{bg}} \in [12.78, 50.85]\;\text{nm} \le \lambda_{\min}/(10\,n_{\text{eff,slab}}) = 51.18\;\text{nm}$)
#   and $\text{cells\_across\_feature} \in \{4.00, 6.00, 8.00, 16.00\} \ge 4.00$
#   across the smallest geometric feature ($\Delta x_{\min} \in [6.25, 25.00]\;\text{nm}$;
#   in `task_9` `design2_sym_taper`, $8.00$ cells of $\Delta y_{\min} = 7.14\;\text{nm}$
#   span the $60\;\text{nm}$ finger-to-strip gap and $\text{slot\_cells} = 14.00$
#   cells span the $100\;\text{nm}$ slot). Only the two intentional warning
#   demonstration runs (`task_0` and `task_1`, `is_demo = true`, `cpw = 10, N = 0`)
#   have $\text{cells\_across\_feature} = 1.96 < 4$.
# - **Convergence verdict**: On Design 1 ($L = 6.0\;\mu\text{m}$), the Kottke
#   conversion efficiency $T(1550\;\text{nm})$ progresses across the resolved
#   ladder as $0.977437$ (`L1 = (10, 4)`) $\to 0.976502$ (`L2 = (14, 6)`,
#   relative change $0.0957\%$) $\to 0.974670$ (`L3 = (20, 8)`, relative change
#   $0.1880\%$) $\to 0.974864$ (`R = (40, 16)`). **Between the last two levels
#   (`L3 = (20, 8)` and `R = (40, 16)`), the relative change in $T(1550\;\text{nm})$
#   is $|T_R - T_{L3}| / T_R = 0.0199\%$** ($1.94\times 10^{-4}$ absolute, or
#   $0.0009\;\text{dB}$ insertion loss, with a broadband $1505\text{--}1595\;\text{nm}$
#   RMS difference of $3.05\times 10^{-4}$). The metric is therefore fully
#   converged at the accepted `--full` primary level `L3 = (20, 8)`.


# %%
# Resolution ladder pairs (cells_per_wavelength, N_cells_across_min_feature)
RES_UNDERRESOLVED = (10, 0)
RES_L1 = (10, 4)
RES_L2 = (14, 6)
RES_L3 = (20, 8)
RES_REF_FULL = (40, 16)


def min_feature_for_design(
    design_key, gap_slot=GAP_SLOT, gap_finger=GAP_FINGER
):
    """Smallest geometric feature width [m] for the given converter design."""
    if design_key == "sym_taper":
        return min(float(gap_finger), float(gap_slot))
    if design_key in ("mmi", "asym_taper"):
        return float(gap_slot)
    raise ValueError(f"Unknown design_key {design_key!r}.")


def build_mesh_overrides(
    design_key,
    length_conv=None,
    n_feature=0,
    gap_slot=GAP_SLOT,
    gap_finger=GAP_FINGER,
):
    """
    Build `mesh_overrides` boxes `[(lo, hi, (hx, hy, hz)), ...]` that refine the
    transverse (`y`) mesh across the converter's smallest feature to
    `max_cell_y = min_feature / n_feature` while leaving `x` and `z` at the
    background core resolution.
    """
    n_f = 0 if n_feature is None else int(n_feature)
    if n_f <= 0:
        return []

    g_slot = float(gap_slot)
    g_fing = float(gap_finger)
    hg = 0.5 * g_slot
    min_feat = min_feature_for_design(
        design_key, gap_slot=g_slot, gap_finger=g_fing
    )
    max_hy = min_feat / float(n_f)

    if design_key == "sym_taper":
        L_c = L_CONV_SYM if length_conv is None else float(length_conv)
        y_tip = 0.5 * W_STRIP + g_fing
        return [
            ((0.0, -y_tip, -np.inf), (L_c, +y_tip, np.inf), (1.0, max_hy, 1.0)),
            ((0.0, -hg, -np.inf), (L_c + L_LEAD, +hg, np.inf), (1.0, max_hy, 1.0)),
        ]
    if design_key == "asym_taper":
        L_c = L_CONV_ASYM if length_conv is None else float(length_conv)
        y_lo = -0.5 * W_STRIP - g_slot
        return [
            ((0.0, y_lo, -np.inf), (L_c, +hg, np.inf), (1.0, max_hy, 1.0)),
            ((0.0, -hg, -np.inf), (L_c + L_LEAD, +hg, np.inf), (1.0, max_hy, 1.0)),
        ]
    if design_key == "mmi":
        L_c = L_CONV_MMI if length_conv is None else float(length_conv)
        return [
            ((L_MMI, -hg, -np.inf), (L_c + L_LEAD, +hg, np.inf), (1.0, max_hy, 1.0)),
        ]
    raise ValueError(f"Unknown design_key {design_key!r}.")


def build_task_simulation(task_cfg, use_gpu=None):
    """Instantiate and build the `PICSimulation` for `task_cfg` (without running)."""
    L_c = float(task_cfg["length_um"]) * 1e-6
    g_slot = float(task_cfg.get("gap_slot_m", GAP_SLOT))
    g_fing = float(task_cfg.get("gap_finger_m", GAP_FINGER))
    d_key = task_cfg["design"]

    if d_key == "mmi":
        structures, ports, domain = build_mmi_converter(
            length_conv=L_c, gap_slot=g_slot
        )
    elif d_key == "sym_taper":
        structures, ports, domain = build_symmetric_taper_converter(
            length_taper=L_c, gap_finger=g_fing, gap_slot=g_slot
        )
    elif d_key == "asym_taper":
        structures, ports, domain = build_asymmetric_taper_converter(
            length_taper=L_c, gap_slot=g_slot
        )
    else:
        raise ValueError(f"Unknown design {d_key!r}.")

    overrides = build_mesh_overrides(
        d_key,
        length_conv=L_c,
        n_feature=task_cfg.get("n_feature", 0),
        gap_slot=g_slot,
        gap_finger=g_fing,
    )
    if use_gpu is None:
        use_gpu = os.environ.get("WAKIS_USE_GPU", "0") == "1"

    if task_cfg.get("nonuniform", False):
        return NonUniformStripToSlotPIC(
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
            cells_per_wavelength=task_cfg["cpw"],
            domain=domain,
            n_pml=task_cfg["n_pml"],
            length_conv=L_c,
            length_mmi=L_MMI,
            width_mmi=W_MMI,
            width_slot=W_SLOT,
            gap_slot=g_slot,
            n_feature=task_cfg.get("n_feature", 8),
            field_wavelengths=(WL_REF,),
            record_movie=task_cfg.get("record_movie", False),
            movie_frames=task_cfg.get("movie_frames", 0),
            use_gpu=use_gpu,
        )

    return PICSimulation(
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
        cells_per_wavelength=task_cfg["cpw"],
        domain=domain,
        n_pml=task_cfg["n_pml"],
        mesh_overrides=overrides,
        field_wavelengths=(WL_REF,),
        record_movie=task_cfg.get("record_movie", False),
        movie_frames=task_cfg.get("movie_frames", 0),
        use_gpu=use_gpu,
    )


def measure_task_resolution(pic, task_cfg):
    """
    Measure resolution metrics from the built `pic.grid` mesh:
    `min_feature_nm`, `dx_background_nm`, `dx_min_nm`, `cells_across_feature`,
    `slot_cells`, `min_feature_cells`, `n_cells`, and `n_steps`.
    """
    d_key = task_cfg["design"]
    g_slot = float(task_cfg.get("gap_slot_m", GAP_SLOT))
    g_fing = float(task_cfg.get("gap_finger_m", GAP_FINGER))
    min_feat_m = min_feature_for_design(
        d_key, gap_slot=g_slot, gap_finger=g_fing
    )

    dx = np.diff(pic.grid.x)
    dy = np.diff(pic.grid.y)
    yg = pic.grid.y

    dx_bg_m = float(np.median(dx))
    dx_min_m = float(min(np.min(dx), np.min(dy)))

    in_slot = (yg[1:] > -0.5 * g_slot + 1e-12) & (yg[:-1] < 0.5 * g_slot - 1e-12)
    dy_slot_max = float(np.max(dy[in_slot]))
    slot_cells = float(g_slot / dy_slot_max)

    if d_key == "sym_taper":
        feat_lo = 0.5 * W_STRIP
        feat_hi = 0.5 * W_STRIP + g_fing
    else:
        feat_lo = -0.5 * g_slot
        feat_hi = +0.5 * g_slot
    in_feat = (yg[1:] > feat_lo + 1e-12) & (yg[:-1] < feat_hi - 1e-12)
    dy_feat_max = float(np.max(dy[in_feat]))
    cells_across_feature = float(min_feat_m / dy_feat_max)

    lx = pic.grid.xmax - pic.grid.xmin
    ly = pic.grid.ymax - pic.grid.ymin
    transit = np.hypot(lx, ly) * pic.n_core / c_light
    max_time = pic.sim.pulse.end_time + 15.0 * transit
    nt_est = int(np.ceil(max_time / pic.solver.dt))
    n_steps = int(getattr(pic.solver, "Nt", None) or nt_est)

    return {
        "min_feature_nm": float(min_feat_m * 1e9),
        "dx_background_nm": float(dx_bg_m * 1e9),
        "dx_min_nm": float(dx_min_m * 1e9),
        "dy_nm": float(dy_slot_max * 1e9),
        "cells_across_feature": float(cells_across_feature),
        "slot_cells": float(slot_cells),
        "min_feature_cells": float(cells_across_feature),
        "n_cells": int(pic.grid.Nx * pic.grid.Ny * pic.grid.Nz),
        "n_steps": int(n_steps),
    }


def get_task_table(quick=False):
    """
    Return the study task table:
    - `--quick` (13 tasks): under-resolved demo pair `(10, 0)`, resolved
      Kottke/staircase at `L1=(10, 4)` and `L2=(14, 6)`, Kottke reference at
      `L3=(20, 8)`, Designs 2 & 3 at `L2=(14, 6)`, and 4 MMI length-sweep points
      at `L2=(14, 6)`.
    - `--full` (15 tasks): under-resolved demo pair `(10, 0)`, resolved
      Kottke/staircase at `L1=(10, 4)`, `L2=(14, 6)`, `L3=(20, 8)`, Kottke
      reference at `R=(40, 16)`, Designs 2 & 3 at `L3=(20, 8)`, and 4 MMI
      length-sweep points at `L3=(20, 8)`.
    """
    conv_ladder = (
        (("L1", RES_L1[0], RES_L1[1]), ("L2", RES_L2[0], RES_L2[1]))
        if quick
        else (
            ("L1", RES_L1[0], RES_L1[1]),
            ("L2", RES_L2[0], RES_L2[1]),
            ("L3", RES_L3[0], RES_L3[1]),
        )
    )
    primary_lvl, primary_cpw, primary_n = conv_ladder[-1]
    ref_cpw, ref_n = RES_L3 if quick else RES_REF_FULL
    n_pml = 8 if quick else 10
    decay = 1e-4

    tasks = []
    # Under-resolved warning demonstration pair (10 cpw, no mesh overrides)
    for avg in ("kottke", "staircase"):
        tasks.append(
            {
                "index": len(tasks),
                "name": f"underresolved_{avg}",
                "design": "mmi",
                "length_um": 6.0,
                "level": "underresolved",
                "averaging": avg,
                "cpw": RES_UNDERRESOLVED[0],
                "n_feature": RES_UNDERRESOLVED[1],
                "is_demo": True,
                "n_pml": n_pml,
                "energy_decay": decay,
                "collect_fields": False,
                "solve_modes_extra": False,
                "record_movie": False,
                "movie_frames": 0,
                "description": (
                    f"Design 1 MMI 6 um ({avg}, under-resolved demo, "
                    f"cpw={RES_UNDERRESOLVED[0]}, no mesh override)"
                ),
            }
        )

    # Resolved Kottke vs staircase ladder on Design 1 (MMI)
    for lvl, cpw, n_feat in conv_ladder:
        for avg in ("kottke", "staircase"):
            is_primary = lvl == primary_lvl and avg == "kottke"
            tasks.append(
                {
                    "index": len(tasks),
                    "name": f"{avg}_{lvl}",
                    "design": "mmi",
                    "length_um": 6.0,
                    "level": lvl,
                    "averaging": avg,
                    "cpw": cpw,
                    "n_feature": n_feat,
                    "is_demo": False,
                    "n_pml": n_pml,
                    "energy_decay": decay,
                    "collect_fields": is_primary,
                    "solve_modes_extra": is_primary,
                    "record_movie": is_primary,
                    "movie_frames": 32 if quick else 48,
                    "description": (
                        f"Design 1 MMI 6 um ({avg}, {lvl} grid, cpw={cpw}, N={n_feat}"
                        + (", DFT fields + movie + port modes)" if is_primary else ")")
                    ),
                }
            )

    # Independent Kottke reference run: L3=(20, 8) in --quick, R=(40, 16) in --full
    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_reference",
            "design": "mmi",
            "length_um": 6.0,
            "level": "reference",
            "averaging": "kottke",
            "cpw": ref_cpw,
            "n_feature": ref_n,
            "is_demo": False,
            "n_pml": n_pml,
            "energy_decay": decay,
            "collect_fields": False,
            "solve_modes_extra": False,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"Design 1 MMI 6 um (kottke, reference grid, cpw={ref_cpw}, N={ref_n})"
            ),
        }
    )

    # Topology comparison: Design 2 (Symmetric Taper, 8 um) and Design 3 (Asymmetric Taper, 20 um)
    for d_name, d_key, L_um in (
        ("design2_sym_taper", "sym_taper", 8.0),
        ("design3_asym_taper", "asym_taper", 20.0),
    ):
        tasks.append(
            {
                "index": len(tasks),
                "name": d_name,
                "design": d_key,
                "length_um": L_um,
                "level": primary_lvl,
                "averaging": "kottke",
                "cpw": primary_cpw,
                "n_feature": primary_n,
                "is_demo": False,
                "n_pml": n_pml,
                "energy_decay": decay,
                "collect_fields": True,
                "solve_modes_extra": False,
                "record_movie": False,
                "movie_frames": 0,
                "description": (
                    f"{d_name} (L={L_um:.1f} um, kottke, cpw={primary_cpw}, "
                    f"N={primary_n}, DFT fields)"
                ),
            }
        )

    # Converter length sweep on Design 1 MMI (L = 1.6, 2.2, 3.0, 4.5 um; L = 6.0 um is primary)
    for L_um in SWEEP_LENGTHS_UM[:-1]:
        tasks.append(
            {
                "index": len(tasks),
                "name": f"length_L{L_um:.1f}um",
                "design": "mmi",
                "length_um": float(L_um),
                "level": "sweep",
                "averaging": "kottke",
                "cpw": primary_cpw,
                "n_feature": primary_n,
                "is_demo": False,
                "n_pml": n_pml,
                "energy_decay": decay,
                "collect_fields": False,
                "solve_modes_extra": False,
                "record_movie": False,
                "movie_frames": 0,
                "description": (
                    f"Design 1 MMI length sweep L={L_um:.1f} um "
                    f"(kottke, cpw={primary_cpw}, N={primary_n})"
                ),
            }
        )

    # Graded non-uniform Cartesian mesh upgrade on Design 1 MMI (L = 6.0 um)
    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_nonuniform",
            "design": "mmi",
            "length_um": 6.0,
            "level": "nonuniform",
            "averaging": "kottke",
            "cpw": primary_cpw,
            "n_feature": primary_n,
            "nonuniform": True,
            "is_demo": False,
            "n_pml": n_pml,
            "energy_decay": decay,
            "collect_fields": False,
            "solve_modes_extra": False,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"Design 1 MMI 6 um (kottke, graded non-uniform Cartesian mesh, "
                f"cpw={primary_cpw}, N={primary_n})"
            ),
        }
    )

    return tasks


def run_single_task(task_cfg, out_dir):
    """
    Execute one 2D EIM strip-to-slot converter study task and write
    ``task_<I>.npz`` and ``task_<I>.json`` into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task_cfg["index"]
    t0 = time.perf_counter()

    pic = build_task_simulation(task_cfg)

    s_mat = pic.run_s_parameters(
        drive=["in"],
        energy_decay=task_cfg["energy_decay"],
        collect_fields=task_cfg["collect_fields"],
    )

    res = measure_task_resolution(pic, task_cfg)

    payload = {
        "task_index": np.int32(idx),
        "name": np.array(task_cfg["name"]),
        "design": np.array(task_cfg["design"]),
        "length_um": np.float64(task_cfg["length_um"]),
        "level": np.array(task_cfg["level"]),
        "averaging": np.array(task_cfg["averaging"]),
        "cpw": np.float32(task_cfg["cpw"]),
        "n_feature": np.int32(task_cfg.get("n_feature", 0)),
        "is_demo": np.bool_(task_cfg.get("is_demo", False)),
        "min_feature_nm": np.float64(res["min_feature_nm"]),
        "dx_background_nm": np.float64(res["dx_background_nm"]),
        "dx_min_nm": np.float64(res["dx_min_nm"]),
        "dy_nm": np.float64(res["dy_nm"]),
        "cells_across_feature": np.float64(res["cells_across_feature"]),
        "slot_cells": np.float64(res["slot_cells"]),
        "min_feature_cells": np.float64(res["min_feature_cells"]),
        "n_cells": np.int32(res["n_cells"]),
        "n_steps": np.int32(res["n_steps"]),
        "n_eff_slab": np.float64(pic.n_eff_slab),
        "frequencies": pic.frequencies.astype(np.float64),
        "wavelengths": pic.wavelengths.astype(np.float64),
        "S11": s_mat[("in", "in")].astype(np.complex128),
        "S21": s_mat[("out", "in")].astype(np.complex128),
    }

    if task_cfg.get("solve_modes_extra", False):
        modes_in = pic.solve_port_modes("in", frequencies=c_light / WL_REF, n_modes=1)
        modes_out = pic.solve_port_modes("out", frequencies=c_light / WL_REF, n_modes=1)
        y_in = modes_in[0].coords["Eb"][0]
        ey_in = modes_in[0].e["b"][:, 0]
        m_in = np.abs(y_in) <= 0.85e-6
        y_out = modes_out[0].coords["Eb"][0]
        ey_out = modes_out[0].e["b"][:, 0]
        m_out = np.abs(y_out) <= 0.85e-6
        payload["port_y_in"] = y_in[m_in].astype(np.float32)
        payload["port_ey_in"] = ey_in[m_in].astype(np.complex64)
        payload["port_neff_in"] = np.float64(np.real(modes_in[0].neff))
        payload["port_y_out"] = y_out[m_out].astype(np.float32)
        payload["port_ey_out"] = ey_out[m_out].astype(np.complex64)
        payload["port_neff_out"] = np.float64(np.real(modes_out[0].neff))

        wl_disp = np.linspace(WL_MIN, WL_MAX, 11)
        f_disp = c_light / wl_disp
        disp_in = pic.solve_port_modes("in", frequencies=f_disp, n_modes=1)
        disp_out = pic.solve_port_modes("out", frequencies=f_disp, n_modes=1)
        neff_in_d = np.array([float(np.real(m[0].neff)) for m in disp_in])
        neff_out_d = np.array([float(np.real(m[0].neff)) for m in disp_out])
        ng_in_d = neff_in_d - wl_disp * np.gradient(neff_in_d, wl_disp)
        ng_out_d = neff_out_d - wl_disp * np.gradient(neff_out_d, wl_disp)
        payload["disp_wavelengths"] = wl_disp.astype(np.float64)
        payload["disp_neff_strip"] = neff_in_d.astype(np.float64)
        payload["disp_ng_strip"] = ng_in_d.astype(np.float64)
        payload["disp_neff_slot"] = neff_out_d.astype(np.float64)
        payload["disp_ng_slot"] = ng_out_d.astype(np.float64)

    if task_cfg["collect_fields"]:
        if WL_REF in pic.dft_fields:
            fmap = pic.dft_fields[WL_REF]
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
        "design": task_cfg["design"],
        "length_um": task_cfg["length_um"],
        "level": task_cfg["level"],
        "averaging": task_cfg["averaging"],
        "cpw": task_cfg["cpw"],
        "n_feature": int(task_cfg.get("n_feature", 0)),
        "is_demo": bool(task_cfg.get("is_demo", False)),
        "min_feature_nm": round(res["min_feature_nm"], 2),
        "dx_background_nm": round(res["dx_background_nm"], 2),
        "dx_min_nm": round(res["dx_min_nm"], 2),
        "dy_nm": round(res["dy_nm"], 2),
        "cells_across_feature": round(res["cells_across_feature"], 2),
        "slot_cells": round(res["slot_cells"], 2),
        "min_feature_cells": round(res["min_feature_cells"], 2),
        "grid_shape": [int(pic.grid.Nx), int(pic.grid.Ny), int(pic.grid.Nz)],
        "n_cells": int(res["n_cells"]),
        "n_steps": int(res["n_steps"]),
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mb": round(rss_mb, 2),
        "npz_bytes": int(npz_path.stat().st_size),
    }
    (out_dir / f"task_{idx}.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[task_{idx} {task_cfg['name']}] cpw={meta['cpw']} N={meta['n_feature']} "
        f"min_feature_nm={meta['min_feature_nm']:.1f} "
        f"dx_background_nm={meta['dx_background_nm']:.2f} "
        f"dx_min_nm={meta['dx_min_nm']:.2f} "
        f"cells_across_feature={meta['cells_across_feature']:.2f} "
        f"n_cells={meta['n_cells']} n_steps={meta['n_steps']} "
        f"time={elapsed:.2f}s rss={rss_mb:.1f}MB"
    )
    return meta


# %% [markdown]
# ## 6. Generating the complete 20-figure diagnostic suite
#
# When invoked with `--plot` (or at the end of `--quick` / `--full`),
# `generate_all_figures` reads `data/task_*.npz` and writes 20 figures to
# `figures/` (every figure $\le 3$ panels; every field map and GIF frame
# includes `schematic.overlay_geometry`):
#
# 0a. `00a_setup_geometry.png` — Labelled setup schematic of the Design 1 symmetric MMI strip-to-slot converter (domain, 10-cell CPML, periodic $\pm z$, $\text{Si}/\text{SiO}_2$ indices, input/output ports with overlaid $\text{TE}_0$ strip and slot mode profiles, and 2D Poynting flux box).
# 0b. `00b_setup_mesh_and_excitation.png` — Mesh view with $1\times$ zoom, cells per wavelength and across the $g = 100\;\text{nm}$ slot opening, wave-packet extent, and broadband excitation pulse/spectrum (`00_setup_table.md`).
# 0c. `00b_taper_converters_schematic.png` — Labelled setup schematics of Design 2 (Symmetric Double-Tip Tapered-Finger Coupler, $L = 8.0\;\mu\text{m}$, $d = 60\;\text{nm}$) and Design 3 (Asymmetric Lateral Taper, $L = 20.0\;\mu\text{m}$).
# 1. `01_converter_topologies_layout.png` — Annotated 3-panel top-view layout of Designs 1 (MMI), 2 (Symmetric Tapered-Finger Coupler), and 3 (Asymmetric Lateral Taper).
# 2. `02_eim_vertical_stack.png` — Vertical $250\;\text{nm}$ SOI stack, 1D $\text{TE}_0$ slab mode profile, and $n_{\text{eff,slab}}(\lambda)$.
# 3. `03_slot_mode_physics_and_discontinuity.png` — Analytical 5-layer slot waveguide mode vs Wakis port mode showing the $(n_{\text{eff,slab}}/n_{\text{SiO}_2})^2 = 4.14$ normal $E_y$ jump and slot confinement vs gap width $g$.
# 4. `04_permittivity_mesh_zoom.png` — Zoomed in-plane permittivity map $\varepsilon_{r,y}$ with Yee cell edges across the $100\;\text{nm}$ slot opening at `L1 = (10, 4)` (staircase and Kottke, $4$ cells across the slot) and at the reference level (`L3 = (20, 8)` in `--quick` or `R = (40, 16)` in `--full`).
# 4b. `04b_mesh_upgrade_comparison.png` — Side-by-side mesh view and equal-accuracy C-band conversion efficiency comparison between the baseline `kottke_L3` ($65{,}780$ cells, $146.50\;\text{s}$) and the graded non-uniform Cartesian `kottke_nonuniform` ($40{,}737$ cells, $-38.1\%$, $60.62\;\text{s}$, $-58.6\%$).
# 5. `05_port_mode_profiles.png` — Input strip ($w = 400\;\text{nm}$) and output slot ($w_{\text{slot}} = 620\;\text{nm}, g = 100\;\text{nm}$) port eigenmode profiles $|E_y(y)|$ and $\mathrm{Re}(E_y(y))$.
# 6. `06_port_neff_dispersion.png` — Effective index $n_{\text{eff}}(\lambda)$ and group index $n_g(\lambda)$ dispersion for both strip and slot waveguide ports.
# 7. `07_mmi_field_intensity_1550.png` — Design 1 (MMI converter) steady-state logarithmic intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $1550\;\text{nm}$ with `overlay_geometry`.
# 8. `08_mmi_field_real_ey_1550.png` — Design 1 (MMI converter) steady-state wave-front phase map $\mathrm{Re}(E_y)$ at $1550\;\text{nm}$ with `overlay_geometry`.
# 9. `09_pulse_propagation.gif` — Time-domain pulse propagation movie through the Design 1 MMI strip-to-slot converter with `overlay_geometry`.
# 10. `10_all_designs_field_comparison_1550.png` — 3-panel steady-state modal intensity $|E|^2$ [dB] across all three converter topologies at $1550\;\text{nm}$ with `overlay_geometry`.
# 10b. `10b_all_designs_real_ey_1550.png` — 3-panel steady-state wave-front phase map $\mathrm{Re}(E_y)$ across all three converter topologies at $1550\;\text{nm}$ with `overlay_geometry`.
# 11. `11_sparameter_spectra_linear_and_db.png` — Broadband coupling efficiency $T(\lambda) = |S_{21}|^2$ (linear $\%$) and insertion/return loss (dB) across the $1505\text{--}1595\;\text{nm}$ band for all three topologies.
# 12. `12_energy_budget_and_passivity.png` — 2D Poynting flux-box radiation budget and guided vs lateral radiation breakdown across the three topologies.
# 13. `13_conversion_efficiency_vs_length.png` — Strip-to-slot conversion efficiency $T(1550\;\text{nm})$ and return loss $R(1550\;\text{nm})$ vs converter length $L \in [1.6, 6.0]\;\mu\text{m}$.
# 14. `14_kottke_vs_staircase_convergence.png` — Multi-resolution convergence of Kottke vs staircasing across the resolved ladder (`L1`, `L2`, `L3`) and graded `kottke_nonuniform` against the independent `kottke_reference` run, with a second axis and point labels showing cells across the $100\;\text{nm}$ slot plus the under-resolved `(cpw=10, N=0)` warning pair.
# 15. `15_resolution_spectra_comparison.png` — Broadband insertion loss and return loss across the resolved ladder, `kottke_nonuniform`, and the under-resolved warning pair for Kottke vs staircase.
# 16. `16_port_time_trace.png` — Output slot port time-domain waveform $E_y(t)$ and exponential ring-down audit.
#
# ![Mesh upgrade comparison](figures/04b_mesh_upgrade_comparison.png)
# ![Multi-design intensity comparison](figures/10_all_designs_field_comparison_1550.png)
# ![Multi-design phase comparison](figures/10b_all_designs_real_ey_1550.png)
#
# ### 6.1 Summary of measured results across topologies, length sweep, and published references
#
# - **Design 1 — Symmetric MMI Converter ($L = 6.0\;\mu\text{m}$, Deng *et al.* 2014 [1])**:
#   At the `--full` primary level `L3 = (20, 8)` ($8.00$ cells across the
#   $100\;\text{nm}$ slot, $\Delta y_{\min} = 12.50\;\text{nm}$, $65{,}780$ cells,
#   $146.50\;\text{s}$), Kottke yields conversion efficiency
#   $T(1550\;\text{nm}) = 97.47\%$ ($0.974670$, insertion loss $0.1114\;\text{dB}$,
#   return loss $R(1550\;\text{nm}) = 7.28\times 10^{-4}$ or $-31.38\;\text{dB}$,
#   band-averaged reflection $\langle|S_{11}|^2\rangle_{1515\text{--}1585} = 7.55\times 10^{-4}$,
#   and C-band mean transmission $\langle T\rangle_{1505\text{--}1595} = 97.54\%$
#   over $[96.48\%, 98.84\%]$). On the upgraded **graded non-uniform Cartesian mesh**
#   (`kottke_nonuniform`, $40{,}737$ cells / **$-38.1\%$**, $60.62\;\text{s}$ / **$-58.6\%$**),
#   Kottke yields $T(1550\;\text{nm}) = 97.57\%$ ($0.975703$, insertion loss
#   $0.1068\;\text{dB}$, $R(1550\;\text{nm}) = 6.19\times 10^{-4}$ or $-32.08\;\text{dB}$,
#   $\langle T\rangle = 97.63\%$), while the `R = (40, 16)` independent reference
#   ($16.00$ cells across the slot, $\Delta y_{\min} = 6.25\;\text{nm}$) gives
#   $T_{\text{ref}}(1550\;\text{nm}) = 97.49\%$ ($0.974864$, insertion loss
#   $0.1106\;\text{dB}$, $R(1550\;\text{nm}) = 7.82\times 10^{-4}$ or
#   $-31.07\;\text{dB}$, $\langle|S_{11}|^2\rangle = 7.24\times 10^{-4}$)—in
#   direct agreement with Deng *et al.*'s $96.5\text{--}97.5\%$ 3D FDTD
#   benchmark without tuning any geometric parameter. Across the length sweep
#   $L \in \{1.6, 2.2, 3.0, 4.5, 6.0\}\;\mu\text{m}$ at `L3 = (20, 8)`,
#   $T(1550\;\text{nm})$ climbs monotonically through
#   $\{61.01\%, 82.94\%, 96.73\%, 97.24\%, 97.47\%\}$ (vs
#   $\{61.39\%, 83.87\%, 97.00\%, 97.43\%, 97.65\%\}$ at `L2 = (14, 6)` in
#   `--quick`) and plateaus once $L \ge 3.0\;\mu\text{m}$ ($L_{\text{taper}} \ge 1.62\;\mu\text{m}$).
# - **Design 2 — Symmetric Tapered-Finger Coupler ($L = 8.0\;\mu\text{m}$, Wang *et al.* 2009 [2])**:
#   At `L3 = (20, 8)` with `mesh_overrides` placing $8.00$ cells
#   ($\Delta y_{\min} = 7.14\;\text{nm}$) across the $d = 60\;\text{nm}$
#   finger-to-strip gap and $14.00$ cells across the $100\;\text{nm}$ slot,
#   conversion efficiency reaches $T(1550\;\text{nm}) = 99.80\%$ ($0.997951$,
#   $0.0089\;\text{dB}$ insertion loss) with ultralow reflection
#   $R(1550\;\text{nm}) = 3.28\times 10^{-7}$ ($-64.85\;\text{dB}$,
#   $\langle|S_{11}|^2\rangle_{1515\text{--}1585} = 9.47\times 10^{-6}$) and
#   2D Poynting flux-box total $\langle\Phi_{\text{box}}\rangle = 0.9994$.
# - **Design 3 — Asymmetric Lateral Taper ($L = 20.0\;\mu\text{m}$, Feng *et al.* 2007 [3])**:
#   At `L3 = (20, 8)` ($8.00$ cells across the $100\;\text{nm}$ slot,
#   $\Delta y_{\min} = 12.50\;\text{nm}$), adiabatic conversion reaches
#   $T(1550\;\text{nm}) = 99.66\%$ ($0.996596$, $0.0148\;\text{dB}$ insertion
#   loss) with $R(1550\;\text{nm}) = 1.18\times 10^{-6}$ ($-59.28\;\text{dB}$,
#   $\langle|S_{11}|^2\rangle_{1515\text{--}1585} = 9.81\times 10^{-6}$) and
#   $\langle\Phi_{\text{box}}\rangle = 0.9995$.
#
# ### 6.2 Quick (`--quick`) vs Full (`--full`) vs Published / Analytic Reference
#
# All geometric and material parameters are taken directly from Deng *et al.*
# (2014) [1], Wang *et al.* (2009) [2], and Feng *et al.* (2007) [3] without
# tuning. Note that Designs 2 and 3 are
# modeled with ideal zero-width sharp polygon tips (`asymmetric_taper` collapses
# pinched endpoints to a single vertex); in 3D FDTD or CMOS fabrication, finite
# tip rounding ($w_{\text{tip}} \approx 40\text{--}60\;\text{nm}$) and vertical
# radiation at the tip/slot transition add $\sim 0.02\text{--}0.13\;\text{dB}$
# loss [2, 3].
#
# | Quantity ($\lambda_0 = 1550\;\text{nm}$ unless noted) | Quick (`--quick`: primary `L2=(14,6)`, ref `L3=(20,8)`) | Full (`--full`: primary `L3=(20,8)`, ref `R=(40,16)`) | Analytic Theory / Published Reference |
# |---|---:|---:|---:|
# | 1D vertical $\text{TE}_0$ slab index $n_{\text{eff,slab}}$ ($h_{\text{Si}} = 250\;\text{nm}$) | $2.9306$ ($\varepsilon_{\text{eff}} = 8.5883$) | $2.9306$ ($\varepsilon_{\text{eff}} = 8.5883$) | Exact 3-layer slab: $2.9306$ |
# | Slot-wall normal $E_y$ jump $E_y(50\;\text{nm}^-)/E_y(50\;\text{nm}^+)$ | $4.1417$ (2D EIM) | $4.1417$ (2D EIM) | Almeida *et al.* [4]: $(n_{\text{high}}/n_{\text{low}})^2 = 4.1417$ (2D EIM), $5.8065$ (3D Si) |
# | Input strip ($w = 400\;\text{nm}$) port $n_{\text{eff,strip}},\; n_{g,\text{strip}}$ | $2.4082,\; 3.3783$ (`L2`) | $2.4051,\; 3.3590$ (`L3`) | 1D 3-layer EIM analytic: $n_{\text{eff}} = 2.4007$ |
# | Output slot ($w = 620\;\text{nm}, g = 100\;\text{nm}$) port $n_{\text{eff,slot}},\; n_{g,\text{slot}}$ | $2.1629,\; 2.8123$ (`L2`) | $2.1626,\; 2.8000$ (`L3`) | Almeida 5-layer analytic [4]: $n_{\text{eff}} = 2.1563$ ($\Gamma_{\text{slot}} = 57.2\%$) |
# | Slot-to-rail field ratio $|E_y(0)| / |E_y(180\;\text{nm})|$ | $2.78$ (`L2`) | $2.7839$ (`L3`) | Almeida 5-layer analytic [4]: $2.7844$ |
# | **Design 1 MMI ($L = 6.0\;\mu\text{m}$)** primary $T(1550\;\text{nm})$ & IL | $97.65\%$ ($0.976503$, $0.1033\;\text{dB}$) | $97.47\%$ ($0.974670$, $0.1114\;\text{dB}$) | Deng *et al.* (2014) [1] 3D FDTD: $96.5\text{--}97.5\%$ ($0.11\text{--}0.15\;\text{dB}$) |
# | **Design 1 MMI ($L = 6.0\;\mu\text{m}$)** `kottke_nonuniform` $T(1550\;\text{nm})$, cells & time | — | $97.57\%$ ($0.975703$, $0.1068\;\text{dB}$; $40{,}737$ cells, $60.62\;\text{s}$) | $-38.1\%$ cells, $-58.6\%$ time vs `kottke_L3`; $|T - T_{\text{ref}}| = 8.4\text{e-}4$ |
# | **Design 1 MMI ($L = 6.0\;\mu\text{m}$)** `kottke_reference` $T_{\text{ref}}(1550\;\text{nm})$ | $97.47\%$ ($0.974667$, `L3`) | $97.49\%$ ($0.974864$, `R`) | Relative change `L3` $\to$ `R`: $0.0199\%$ ($1.94\times 10^{-4}$) |
# | **Design 1 MMI ($L = 6.0\;\mu\text{m}$)** return loss $R(1550\;\text{nm})$ (primary / ref) | $8.64\times 10^{-4}$ ($-30.6\;\text{dB}$) / $7.27\times 10^{-4}$ | $7.28\times 10^{-4}$ ($-31.4\;\text{dB}$) / $7.82\times 10^{-4}$ ($-31.1\;\text{dB}$) | Deng *et al.* (2014) [1]: $< -30\;\text{dB}$ ($< 10^{-3}$) |
# | **Design 1 MMI ($L = 6.0\;\mu\text{m}$)** band mean $\langle T\rangle_{1505\text{--}1595}$ (primary / ref) | $97.74\%$ / $97.54\%$ | $97.54\%$ ($96.48\text{--}98.84\%$) / $97.54\%$ | Deng *et al.* (2014) [1]: $> 96\%$ across C-band |
# | **Design 1 MMI ($L = 6.0\;\mu\text{m}$)** 2D flux box $\Phi_{\text{east}}, \Phi_{\text{lat}}, \Phi_{\text{box}}$ ($\langle\Phi_{\text{box}}\rangle$) | $0.9754, 0.0152, 0.9935$ ($0.9995$) | $0.9747, 0.0157, 0.9930$ ($0.9993$) | Lossless energy conservation $\Phi_{\text{box}} = 1.000$ |
# | **Design 1 MMI length sweep** $T(1550\;\text{nm})$ at $L = 1.6, 2.2, 3.0, 4.5, 6.0\;\mu\text{m}$ | $61.39, 83.87, 97.00, 97.43, 97.65\%$ (`L2`) | $61.01, 82.94, 96.73, 97.24, 97.47\%$ (`L3`) | Deng *et al.* [1]: adiabatic plateau for $L \ge 3\text{--}4\;\mu\text{m}$ |
# | **Design 2 Sym. Taper ($L = 8.0\;\mu\text{m}, d = 60\;\text{nm}$)** $T(1550\;\text{nm})$ & IL | $99.80\%$ ($0.997956$, $0.0089\;\text{dB}$, `L2`) | $99.80\%$ ($0.997951$, $0.0089\;\text{dB}$, `L3`) | Wang *et al.* (2009) [2] 3D FDTD: $99.1\text{--}99.5\%$ ($0.02\text{--}0.04\;\text{dB}$, sharp tips) |
# | **Design 2 Sym. Taper ($L = 8.0\;\mu\text{m}$)** $R(1550\;\text{nm})$ & $\langle|S_{11}|^2\rangle_{1515\text{--}1585}$ | $2.17\times 10^{-6}$ ($-56.6\;\text{dB}$), $1.11\times 10^{-5}$ | $3.28\times 10^{-7}$ ($-64.8\;\text{dB}$), $9.47\times 10^{-6}$ | Wang *et al.* (2009) [2]: $< -40\;\text{dB}$ |
# | **Design 3 Asym. Taper ($L = 20.0\;\mu\text{m}$)** $T(1550\;\text{nm})$ & IL | $99.64\%$ ($0.996379$, $0.0158\;\text{dB}$, `L2`) | $99.66\%$ ($0.996596$, $0.0148\;\text{dB}$, `L3`) | Feng *et al.* (2007) [3] 3D FDTD: $99.0\text{--}99.5\%$ ($< 0.04\;\text{dB}$) |
# | **Design 3 Asym. Taper ($L = 20.0\;\mu\text{m}$)** $R(1550\;\text{nm})$ & $\langle|S_{11}|^2\rangle_{1515\text{--}1585}$ | $6.54\times 10^{-7}$ ($-61.8\;\text{dB}$), $9.24\times 10^{-6}$ | $1.18\times 10^{-6}$ ($-59.3\;\text{dB}$), $9.81\times 10^{-6}$ | Feng *et al.* (2007) [3]: $< -40\;\text{dB}$ |
# | **Under-resolved (`10, 0`, $1.96$ cells)** Kottke vs Stair $|T - T_{\text{ref}}|$ | $8.15\text{e-}4$ vs $3.46\text{e-}2$ ($42.4\times$) | $6.00\text{e-}4$ vs $3.48\text{e-}2$ ($58.0\times$; RMS $16.4\times$) | Sub-2-cell gap quantization failure mode |
# | **Resolved `L1 = (10, 4)` ($4.00$ cells)** Kottke vs Stair $|T - T_{\text{ref}}|$ | $2.31\text{e-}3$ vs $2.51\text{e-}2$ ($10.9\times$) | $2.57\text{e-}3$ vs $2.51\text{e-}2$ ($9.7\times$; RMS $7.9\times$) | Subpixel $\mathcal{O}(\Delta x^2)$ vs staircase $\mathcal{O}(\Delta x^1)$ [5] |
# | **Resolved `L2 = (14, 6)` ($6.00$ cells)** Kottke vs Stair $|T - T_{\text{ref}}|$ | $1.84\text{e-}3$ vs $1.19\text{e-}2$ ($6.5\times$) | $1.64\text{e-}3$ vs $1.21\text{e-}2$ ($7.4\times$; RMS $6.3\times$) | Subpixel $\mathcal{O}(\Delta x^2)$ vs staircase $\mathcal{O}(\Delta x^1)$ [5] |
# | **Resolved `L3 = (20, 8)` ($8.00$ cells)** Kottke vs Stair $|T - T_{\text{ref}}|$ | Used as quick reference ($T = 0.974667$) | $1.94\text{e-}4$ vs $2.58\text{e-}2$ ($132.9\times$; RMS $68.6\times$) | Monotonic Kottke vs staircase pixel-snapping on strip/taper walls |


# %%
def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Read ``data_dir/task_*.npz`` and render all 20 figures into ``fig_dir``."""
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
        jf = p.with_suffix(".json")
        if jf.exists():
            tasks_meta[name] = json.loads(jf.read_text())

    primary_key = None
    for cand in ("kottke_L3", "kottke_L2", "kottke_fine"):
        if cand in tasks_data and "dft_E2" in tasks_data[cand]:
            primary_key = cand
            break
    if primary_key is None:
        raise FileNotFoundError(
            f"Missing primary task ('kottke_L2' or 'kottke_L3') in {data_dir}. "
            "Run --quick or --full first."
        )
    primary = tasks_data[primary_key]
    neff_slab = float(primary["n_eff_slab"])
    structures_mmi, ports_mmi, domain_mmi = build_mmi_converter()
    structures_sym, ports_sym, domain_sym = build_symmetric_taper_converter()
    structures_asym, ports_asym, domain_asym = build_asymmetric_taper_converter()

    prim_cpw = int(round(float(primary["cpw"])))
    prim_n = int(primary.get("n_feature", 8))
    pic_mmi_u = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures_mmi,
        ports=ports_mmi,
        n_core=N_SI,
        n_clad=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=prim_cpw,
        domain=domain_mmi,
        n_pml=10 if primary_key == "kottke_L3" else 8,
        mesh_overrides=build_mesh_overrides("mmi", length_conv=L_CONV_MMI, n_feature=prim_n),
    )
    pic_mmi_nu = NonUniformStripToSlotPIC(
        WL_MIN,
        WL_MAX,
        structures=structures_mmi,
        ports=ports_mmi,
        n_core=N_SI,
        n_clad=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=prim_cpw,
        domain=domain_mmi,
        n_pml=10 if primary_key == "kottke_L3" else 8,
        length_conv=L_CONV_MMI,
        length_mmi=L_MMI,
        width_mmi=W_MMI,
        width_slot=W_SLOT,
        gap_slot=GAP_SLOT,
        n_feature=prim_n,
    )
    pic_sym = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures_sym,
        ports=ports_sym,
        n_core=N_SI,
        n_clad=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=prim_cpw,
        domain=domain_sym,
        n_pml=10 if primary_key == "kottke_L3" else 8,
        mesh_overrides=build_mesh_overrides("sym_taper", length_conv=L_CONV_SYM, n_feature=prim_n),
    )
    pic_asym = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures_asym,
        ports=ports_asym,
        n_core=N_SI,
        n_clad=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=prim_cpw,
        domain=domain_asym,
        n_pml=10 if primary_key == "kottke_L3" else 8,
        mesh_overrides=build_mesh_overrides("asym_taper", length_conv=L_CONV_ASYM, n_feature=prim_n),
    )

    # 0. Setup schematics (FIRST figures: Design 1 MMI + Designs 2 & 3 tapers)
    plot_setup_schematic(
        pic_mmi_nu,
        primary,
        out_path=fig_dir / "00a_setup_geometry.png",
    )
    plot_taper_converters_schematic(
        pic_sym,
        pic_asym,
        primary,
        out_path=fig_dir / "00b_taper_converters_schematic.png",
    )

    # 1. Three-panel annotated top-view layout of all three converter topologies
    fig, axes = figstyle.new_figure(nrows=3, ncols=1, width=9.8, height=9.2)
    layout_specs = [
        (
            axes[0],
            structures_mmi,
            ports_mmi,
            domain_mmi,
            "(a) Design 1: Symmetric MMI converter (L = 6.0 µm, W_mmi = 1.24 µm, g = 100 nm; Deng et al. 2014)",
            [
                {
                    "text": "Input strip\nw = 400 nm",
                    "xy": (-0.7, -0.20),
                    "xytext": (-0.7, -0.95),
                    "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
                },
                {
                    "text": "MMI box\n1.38 × 1.24 µm",
                    "xy": (0.69, 0.62),
                    "xytext": (0.69, 1.15),
                    "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
                },
                {
                    "text": "100 nm SiO₂ slot\nopens at x = 1.38 µm",
                    "xy": (1.55, 0.0),
                    "xytext": (3.20, -0.95),
                    "arrowprops": dict(arrowstyle="->", color="#d95f02", lw=1.1),
                },
            ],
        ),
        (
            axes[1],
            structures_sym,
            ports_sym,
            domain_sym,
            "(b) Design 2: Symmetric double-tip tapered-finger coupler (L = 8.0 µm, d = 60 nm; Wang et al. 2009)",
            [
                {
                    "text": "Outer tips\ngap d = 60 nm",
                    "xy": (0.0, 0.26),
                    "xytext": (1.2, 1.05),
                    "arrowprops": dict(arrowstyle="->", color="#d95f02", lw=1.1),
                },
                {
                    "text": "Central strip tip\n(x = 8.0 µm, y = 0)",
                    "xy": (8.0, 0.0),
                    "xytext": (5.5, -1.00),
                    "arrowprops": dict(arrowstyle="->", color="#d95f02", lw=1.1),
                },
            ],
        ),
        (
            axes[2],
            structures_asym,
            ports_asym,
            domain_asym,
            "(c) Design 3: Asymmetric lateral taper converter (L = 20.0 µm; Feng et al. 2007)",
            [
                {
                    "text": "Lower rail tip\n(0, −300 nm)",
                    "xy": (0.0, -0.30),
                    "xytext": (3.5, -1.05),
                    "arrowprops": dict(arrowstyle="->", color="#d95f02", lw=1.1),
                },
                {
                    "text": "Continuous upper rail\nshifts to [+50, +310] nm",
                    "xy": (10.0, 0.20),
                    "xytext": (10.0, 1.05),
                    "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
                },
            ],
        ),
    ]
    for ax_i, st_i, pt_i, dm_i, ttl_i, ann_i in layout_specs:
        plot_layout(
            st_i,
            ports=pt_i,
            domain=dm_i,
            annotations=ann_i,
            title=ttl_i,
            ax=ax_i,
        )
    figstyle.save(fig, fig_dir / "01_converter_topologies_layout.png")
    plt.close(fig)

    # 2. Effective-Index Method (EIM) vertical stack (250 nm Si on SiO2)
    plot_eim_stack(
        n_core=N_SI,
        n_clad=N_SIO2,
        thickness=H_SI,
        wavelength_ref=WL_REF,
        wavelengths=np.linspace(WL_MIN, WL_MAX, 51),
        out_path=fig_dir / "02_eim_vertical_stack.png",
    )

    # 3. Analytical 5-layer slot waveguide mode vs Wakis port mode + gap sweep
    ana_slot = slot_waveguide_mode(
        n_high=neff_slab,
        n_low=N_SIO2,
        slot_width=GAP_SLOT,
        rail_width=W_RAIL,
        wavelength=WL_REF,
        y_span=1.6e-6,
        n_pts=801,
    )
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)
    y_ana_um = ana_slot["y"] * 1e6
    ax0.axvspan(-0.31, -0.05, color="#2b5c8f", alpha=0.18, label="Si rails (260 nm)")
    ax0.axvspan(+0.05, +0.31, color="#2b5c8f", alpha=0.18)
    ax0.axvspan(
        -0.05, +0.05, color="#fdae61", alpha=0.28, label="SiO₂ slot (g = 100 nm)"
    )
    ax0.plot(
        y_ana_um,
        ana_slot["Ey"],
        color="#2b5c8f",
        lw=2.1,
        label=rf"Analytic 5-layer $|E_y(y)|$ ($n_{{\mathrm{{eff}}}}$={ana_slot['neff']:.4f})",
    )
    ax0.plot(
        y_ana_um,
        ana_slot["Dy_norm"],
        "--",
        color="#1b9e77",
        lw=1.7,
        label=r"Continuous $|D_y| / (\varepsilon_0 n_{\mathrm{SiO_2}}^2 E_{\max})$",
    )
    y_num_um = primary["port_y_out"] * 1e6
    ey_num = np.abs(primary["port_ey_out"])
    ey_num /= float(np.max(ey_num)) or 1.0
    ax0.plot(
        y_num_um,
        ey_num,
        "o",
        color="#d95f02",
        ms=4.2,
        alpha=0.85,
        label=rf"Wakis 2D EIM port mode ($n_{{\mathrm{{eff}}}}$={float(primary['port_neff_out']):.4f})",
    )
    ax0.annotate(
        rf"Jump $(n_{{\mathrm{{eff,slab}}}}/n_{{\mathrm{{SiO_2}}}})^2 = {ana_slot['discontinuity_ratio']:.2f}\times$",
        xy=(0.05, 0.96),
        xytext=(0.24, 0.90),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7"),
    )
    ax0.set(
        xlim=(-0.75, 0.75),
        ylim=(0.0, 1.08),
        xlabel="Transverse coordinate y [µm]",
        ylabel="Normalized modal amplitude",
        title="(a) Slot mode field enhancement & boundary jump",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper left", fontsize=7.5)

    gaps_nm = np.linspace(40.0, 220.0, 25)
    neff_vs_g = []
    conf_vs_g = []
    for g_nm in gaps_nm:
        m_g = slot_waveguide_mode(
            neff_slab, N_SIO2, g_nm * 1e-9, W_RAIL, wavelength=WL_REF
        )
        neff_vs_g.append(m_g["neff"])
        conf_vs_g.append(m_g["slot_confinement"] * 100.0)
    ax1_r = ax1.twinx()
    l1 = ax1.plot(
        gaps_nm,
        neff_vs_g,
        color="#2b5c8f",
        lw=2.1,
        label=r"Slot effective index $n_{\mathrm{eff,slot}}(g)$",
    )
    l2 = ax1_r.plot(
        gaps_nm,
        conf_vs_g,
        "--",
        color="#d95f02",
        lw=2.0,
        label=r"Slot $\int_{|y|\leq g/2}|E_y|^2\mathrm{d}y$ fraction [%]",
    )
    ax1.axvline(100.0, color="0.4", ls=":", lw=1.3, label="Design slot g = 100 nm")
    ax1.set_xlabel("Central oxide slot width g [nm]")
    ax1.set_ylabel(r"Slot mode effective index $n_{\mathrm{eff,slot}}$", color="#2b5c8f")
    ax1_r.set_ylabel(r"Slot $|E_y|^2$ confinement factor [%]", color="#d95f02")
    ax1.set_title("(b) Slot effective index & confinement vs slot width g")
    ax1.grid(True, ls=":", alpha=0.4)
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=8.0)
    figstyle.save(fig, fig_dir / "03_slot_mode_physics_and_discontinuity.png")
    plt.close(fig)

    # 4. In-plane permittivity map with mesh edges zoomed on the 100 nm slot opening
    #    at L1=(10, 4) (staircase & Kottke) and at the reference level
    ref_td = tasks_data.get("kottke_reference", primary)
    ref_cpw_v = int(round(float(ref_td["cpw"])))
    ref_n_v = int(ref_td.get("n_feature", 8))
    mesh_panels = [
        (
            "staircase",
            RES_L1[0],
            RES_L1[1],
            f"(a) L1 Staircase (cpw={RES_L1[0]}, N={RES_L1[1]})",
        ),
        (
            "kottke",
            RES_L1[0],
            RES_L1[1],
            f"(b) L1 Kottke (cpw={RES_L1[0]}, N={RES_L1[1]})",
        ),
        (
            "kottke",
            ref_cpw_v,
            ref_n_v,
            f"(c) Reference Kottke (cpw={ref_cpw_v}, N={ref_n_v})",
        ),
    ]
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.6, height=4.5, sharey=True)
    xl_m, yl_m = (1.10, 2.35), (-0.66, 0.66)
    for ax_m, (avg_m, cpw_m, n_m, ttl_m) in zip(axes, mesh_panels):
        ov_m = build_mesh_overrides("mmi", length_conv=L_CONV_MMI, n_feature=n_m)
        pic_m = PICSimulation(
            WL_MIN,
            WL_MAX,
            structures=structures_mmi,
            ports=ports_mmi,
            n_core=N_SI,
            n_clad=N_SIO2,
            two_d=True,
            averaging=avg_m,
            cells_per_wavelength=cpw_m,
            domain=domain_mmi,
            n_pml=8,
            mesh_overrides=ov_m,
        )
        g_m = pic_m.grid
        iz_m = pic_m.solver.Nz // 2
        ieps_m = np.asarray(pic_m.solver.ieps[:, :, iz_m, "y"])
        eps_r_m = np.where(
            ieps_m > 0, 1.0 / (np.where(ieps_m > 0, ieps_m, 1.0) * epsilon_0), 1.0
        )
        x_um_m, y_um_m = g_m.x * 1e6, g_m.y * 1e6
        pcm = ax_m.pcolormesh(
            x_um_m, y_um_m, eps_r_m.T, cmap=figstyle.CMAP_EPS, shading="flat"
        )
        ax_m.set_xlim(*xl_m)
        ax_m.set_ylim(*yl_m)
        xv = x_um_m[(x_um_m >= xl_m[0]) & (x_um_m <= xl_m[1])]
        yv = y_um_m[(y_um_m >= yl_m[0]) & (y_um_m <= yl_m[1])]
        vlines = [[(x, yl_m[0]), (x, yl_m[1])] for x in xv]
        hlines = [[(xl_m[0], y), (xl_m[1], y)] for y in yv]
        ax_m.add_collection(
            LineCollection(
                vlines + hlines, colors="white", linewidths=0.40, alpha=0.65
            )
        )
        for s in structures_mmi:
            ax_m.add_patch(
                MplPolygon(
                    s.vertices * 1e6,
                    closed=True,
                    fill=False,
                    edgecolor="#ffae00",
                    lw=1.1,
                    ls="--",
                )
            )
        dy_slot_nm = (GAP_SLOT / n_m) * 1e9
        ax_m.annotate(
            "",
            xy=(1.55, -0.05),
            xytext=(1.55, +0.05),
            arrowprops=dict(arrowstyle="<->", color="white", lw=1.4),
        )
        ax_m.text(
            1.60,
            0.0,
            f"g = 100 nm\n({n_m} cells, Δy={dy_slot_nm:.1f} nm)",
            color="white",
            fontsize=7.5,
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="white", alpha=0.68),
        )
        ax_m.set(xlabel="x [µm]", title=ttl_m)
        cb = fig.colorbar(pcm, ax=ax_m, fraction=0.046, pad=0.03)
        cb.set_label(r"$\varepsilon_{r,y}$")
    axes[0].set_ylabel("y [µm]")
    figstyle.save(fig, fig_dir / "04_permittivity_mesh_zoom.png")
    plt.close(fig)

    # 4b. Mesh upgrade comparison: uniform-background kottke_L3 vs graded kottke_nonuniform
    td_nu = tasks_data.get("kottke_nonuniform", primary)
    plot_mesh_upgrade_comparison(
        pic_mmi_u,
        pic_mmi_nu,
        primary,
        td_nu,
        tasks_data.get("kottke_reference"),
        tasks_meta.get(primary_key),
        tasks_meta.get("kottke_nonuniform"),
        WL_REF,
        out_path=fig_dir / "04b_mesh_upgrade_comparison.png",
    )

    # 5. Port mode profiles at input strip and output slot waveguide
    plot_port_modes(
        primary["port_y_in"] * 1e6,
        primary["port_ey_in"],
        float(primary["port_neff_in"]),
        primary["port_y_out"] * 1e6,
        primary["port_ey_out"],
        float(primary["port_neff_out"]),
        structures_outline_y=((-0.31, -0.05), (0.05, 0.31)),
        out_path=fig_dir / "05_port_mode_profiles.png",
    )

    # 6. Strip vs slot waveguide dispersion across 1500-1600 nm
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=10.8, height=3.9)
    wl_d_nm = primary["disp_wavelengths"] * 1e9
    ax0.plot(
        wl_d_nm,
        primary["disp_neff_strip"],
        "o-",
        color="#2b5c8f",
        lw=2.0,
        ms=4.5,
        label="Input strip TE₀ (w = 400 nm)",
    )
    ax0.plot(
        wl_d_nm,
        primary["disp_neff_slot"],
        "s-",
        color="#d95f02",
        lw=2.0,
        ms=4.5,
        label="Output slot TE₀ (w = 620 nm, g = 100 nm)",
    )
    ax0.axhline(
        neff_slab,
        color="0.45",
        ls=":",
        lw=1.4,
        label=f"Slab core $n_{{\\mathrm{{eff,slab}}}}$ = {neff_slab:.4f}",
    )
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Effective index $n_{\mathrm{eff}}(\lambda)$",
        title="(a) Strip vs slot mode effective index dispersion",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="best", fontsize=8.0)

    ax1.plot(
        wl_d_nm,
        primary["disp_ng_strip"],
        "o--",
        color="#2b5c8f",
        lw=2.0,
        ms=4.5,
        label=r"Input strip group index $n_g(\lambda)$",
    )
    ax1.plot(
        wl_d_nm,
        primary["disp_ng_slot"],
        "s--",
        color="#d95f02",
        lw=2.0,
        ms=4.5,
        label=r"Output slot group index $n_g(\lambda)$",
    )
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Group index $n_g(\lambda)$",
        title="(b) Strip vs slot waveguide group index",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="best", fontsize=8.0)
    figstyle.save(fig, fig_dir / "06_port_neff_dispersion.png")
    plt.close(fig)

    # 7. Design 1 (MMI) steady-state logarithmic intensity map |E|^2 [dB] at 1550 nm (with overlay_geometry)
    render_field_map_with_overlay(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_E2"],
        pic=pic_mmi_u,
        kind="intensity_log",
        title=r"Design 1 (MMI converter): modal intensity $10\log_{10}(|\mathbf{E}|^2/|\mathbf{E}|_{\max}^2)$ at $\lambda = 1550\;\mathrm{nm}$",
        out_path=fig_dir / "07_mmi_field_intensity_1550.png",
    )

    # 8. Design 1 (MMI) steady-state real field map Re(Ey) at 1550 nm (with overlay_geometry)
    render_field_map_with_overlay(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_Ey"],
        pic=pic_mmi_u,
        kind="real",
        title=r"Design 1 (MMI converter): wave-front pattern $\mathrm{Re}(E_y)$ at $\lambda = 1550\;\mathrm{nm}$",
        out_path=fig_dir / "08_mmi_field_real_ey_1550.png",
    )

    # 9. Animated pulse propagation GIF through Design 1 (MMI) (with overlay_geometry, < 5 MB)
    if "movie_frames" in primary and len(primary["movie_frames"]) > 0:
        render_movie_with_overlay(
            primary["movie_x"],
            primary["movie_y"],
            primary["movie_frames"],
            primary["movie_times"],
            pic=pic_mmi_u,
            out_path=fig_dir / "09_pulse_propagation.gif",
            fps=12,
        )

    # 10 & 10b. Split 3-panel multi-design field comparisons at 1550 nm (<= 3 panels per figure)
    design_entries = [
        (primary_key, pic_mmi_u, "Design 1: Symmetric MMI (L = 6 µm)"),
        (
            "design2_sym_taper",
            pic_sym,
            "Design 2: Symmetric Tapered-Finger (L = 8 µm)",
        ),
        (
            "design3_asym_taper",
            pic_asym,
            "Design 3: Asymmetric Lateral Taper (L = 20 µm)",
        ),
    ]
    avail_designs = [e for e in design_entries if e[0] in tasks_data]

    # 10. 3-panel steady-state modal intensity |E|^2 [dB] across all 3 topologies
    fig_i, axes_i = figstyle.new_figure(
        nrows=len(avail_designs), ncols=1, width=9.8, height=3.1 * len(avail_designs)
    )
    if len(avail_designs) == 1:
        axes_i = [axes_i]
    for row_idx, (t_key, pic_d, d_label) in enumerate(avail_designs):
        td = tasks_data[t_key]
        x_um = td["dft_x"] * 1e6
        y_um = td["dft_y"] * 1e6
        e2 = td["dft_E2"]
        db_floor = -35.0
        data_db = 10.0 * np.log10(
            np.maximum(e2 / (float(np.max(e2)) or 1.0), 10.0 ** (db_floor / 10.0))
        )
        ax_i = axes_i[row_idx]
        im0 = ax_i.pcolormesh(
            x_um, y_um, data_db.T, cmap="inferno", vmin=db_floor, vmax=0.0, shading="auto"
        )
        cb0 = fig_i.colorbar(im0, ax=ax_i, fraction=0.046, pad=0.03)
        cb0.set_label("Intensity [dB]")
        overlay_converter_geometry(ax_i, pic_d, outline_color="white")
        ax_i.set(
            xlim=(float(x_um.min()), float(x_um.max())),
            ylim=(-1.15, 1.15),
            xlabel="x [µm]",
            ylabel="y [µm]",
            title=f"({chr(ord('a') + row_idx)}) {d_label} — $|\\mathbf{{E}}|^2$ [dB] at 1550 nm",
        )
    figstyle.save(fig_i, fig_dir / "10_all_designs_field_comparison_1550.png")
    plt.close(fig_i)

    # 10b. 3-panel steady-state wave-front phase map Re(Ey) across all 3 topologies
    fig_r, axes_r = figstyle.new_figure(
        nrows=len(avail_designs), ncols=1, width=9.8, height=3.1 * len(avail_designs)
    )
    if len(avail_designs) == 1:
        axes_r = [axes_r]
    for row_idx, (t_key, pic_d, d_label) in enumerate(avail_designs):
        td = tasks_data[t_key]
        x_um = td["dft_x"] * 1e6
        y_um = td["dft_y"] * 1e6
        ey = td["dft_Ey"]
        vmax_r = float(np.percentile(np.abs(ey), 99.5)) or 1.0
        ax_r = axes_r[row_idx]
        im1 = ax_r.pcolormesh(
            x_um,
            y_um,
            np.real(ey).T / vmax_r,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        cb1 = fig_r.colorbar(im1, ax=ax_r, fraction=0.046, pad=0.03)
        cb1.set_label(r"$\mathrm{Re}(E_y)$ [norm.]")
        overlay_converter_geometry(ax_r, pic_d, outline_color="black")
        ax_r.set(
            xlim=(float(x_um.min()), float(x_um.max())),
            ylim=(-1.15, 1.15),
            xlabel="x [µm]",
            ylabel="y [µm]",
            title=f"({chr(ord('a') + row_idx)}) {d_label} — $\\mathrm{{Re}}(E_y)$ at 1550 nm",
        )
    figstyle.save(fig_r, fig_dir / "10b_all_designs_real_ey_1550.png")
    plt.close(fig_r)

    # 11. Broadband S-parameter spectra (linear % and dB) across all 3 converter topologies
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.2)
    topo_colors = {
        primary_key: ("#1b9e77", "-", "Design 1: MMI (6 µm, Deng 2014)"),
        "design2_sym_taper": (
            "#2b5c8f",
            "--",
            "Design 2: Sym. Taper (8 µm, Wang 2009)",
        ),
        "design3_asym_taper": (
            "#d95f02",
            "-.",
            "Design 3: Asym. Taper (20 µm, Feng 2007)",
        ),
    }
    for t_key, (col, ls, lbl) in topo_colors.items():
        if t_key not in tasks_data:
            continue
        td = tasks_data[t_key]
        w_nm = td["wavelengths"] * 1e9
        order = np.argsort(w_nm)
        w_s = w_nm[order]
        m = (w_s >= 1505.0) & (w_s <= 1595.0)
        t_lin = np.abs(td["S21"][order][m]) ** 2
        r_lin = np.abs(td["S11"][order][m]) ** 2
        ax0.plot(w_s[m], t_lin * 100.0, color=col, ls=ls, lw=2.1, label=lbl)
        ax1.plot(
            w_s[m],
            10.0 * np.log10(np.maximum(t_lin, 1e-12)),
            color=col,
            ls=ls,
            lw=2.0,
            label=f"$T$: {lbl}",
        )
        ax1.plot(
            w_s[m],
            10.0 * np.log10(np.maximum(r_lin, 1e-12)),
            color=col,
            ls=":",
            lw=1.4,
            alpha=0.85,
            label=f"$R$: {lbl[:8]}",
        )
    ax0.axhspan(
        96.5,
        97.5,
        color="#1b9e77",
        alpha=0.14,
        label="Deng et al. (2014) 3D FDTD (96.5–97.5%)",
    )
    ax0.axhline(
        99.0,
        color="#2b5c8f",
        ls=":",
        lw=1.3,
        label="Wang (2009) / Feng (2007) 3D ideal (~99%)",
    )
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Strip-to-slot coupling efficiency $|S_{21}|^2$ [%]",
        ylim=(88.0, 101.0),
        title="(a) Broadband conversion efficiency (linear %)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=7.5)

    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Modal power $10\log_{10}|S_{ij}|^2$ [dB]",
        title="(b) Insertion loss (solid/dashed) & return loss (dotted) [dB]",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="center right", fontsize=7.2, ncol=1)
    figstyle.save(fig, fig_dir / "11_sparameter_spectra_linear_and_db.png")
    plt.close(fig)

    # 12. 2D Poynting flux-box radiation & passivity audit
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.2, height=4.1)
    w_nm = primary["wavelengths"] * 1e9
    order = np.argsort(w_nm)
    w_s = w_nm[order]
    m = (w_s >= 1505.0) & (w_s <= 1595.0)
    wb = w_s[m]
    t_mmi = np.abs(primary["S21"][order][m]) ** 2
    r_mmi = np.abs(primary["S11"][order][m]) ** 2
    ax0.plot(wb, t_mmi, color="#1b9e77", lw=2.1, label=r"Guided slot mode $T = |S_{21}|^2$")
    ax0.plot(wb, r_mmi, color="#7570b3", lw=1.8, label=r"Guided reflection $R = |S_{11}|^2$")
    if "flux_total_box" in primary:
        f_east = primary["flux_east"][order][m]
        f_lat = primary["flux_lateral"][order][m]
        f_tot = primary["flux_total_box"][order][m]
        ax0.plot(
            wb,
            f_east,
            "--",
            color="#2b5c8f",
            lw=1.7,
            label="East Poynting plane flux (guided + unguided)",
        )
        ax0.plot(
            wb,
            f_lat,
            "-.",
            color="#d95f02",
            lw=1.7,
            label="Lateral radiation flux (north + south)",
        )
        ax0.plot(
            wb,
            f_tot,
            ":",
            color="0.25",
            lw=1.8,
            label=f"Total 2D flux box (mean = {np.mean(f_tot):.3f})",
        )
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Normalized power fraction",
        ylim=(-0.03, 1.06),
        title="(a) Design 1 MMI Poynting flux-box energy budget",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="center right", fontsize=7.8)

    bar_labels = []
    guided_vals, refl_vals, rad_vals = [], [], []
    for t_key, short_lbl in (
        (primary_key, "Design 1\nMMI (6 µm)"),
        ("design2_sym_taper", "Design 2\nSym. Taper (8 µm)"),
        ("design3_asym_taper", "Design 3\nAsym. Taper (20 µm)"),
    ):
        if t_key not in tasks_data:
            continue
        td = tasks_data[t_key]
        i0 = int(np.argmin(np.abs(td["wavelengths"] - WL_REF)))
        t0_v = min(float(np.abs(td["S21"][i0]) ** 2), 1.0)
        r0_v = float(np.abs(td["S11"][i0]) ** 2)
        rad_v = max(0.0, 1.0 - t0_v - r0_v)
        bar_labels.append(short_lbl)
        guided_vals.append(t0_v * 100.0)
        refl_vals.append(r0_v * 100.0)
        rad_vals.append(rad_v * 100.0)

    x_pos = np.arange(len(bar_labels))
    w_bar = 0.26
    ax1.bar(
        x_pos - w_bar,
        guided_vals,
        width=w_bar,
        color="#1b9e77",
        label=r"Guided slot $T(1550\;\mathrm{nm})$ [%]",
    )
    ax1.bar(
        x_pos,
        rad_vals,
        width=w_bar,
        color="#d95f02",
        label=r"Unguided / radiated $1 - T - R$ [%]",
    )
    ax1.bar(
        x_pos + w_bar,
        refl_vals,
        width=w_bar,
        color="#7570b3",
        label=r"Reflected $R(1550\;\mathrm{nm})$ [%]",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bar_labels)
    ax1.set_ylabel("Power fraction at 1550 nm [%]")
    ax1.set_title("(b) Power partition at λ = 1550 nm across topologies")
    ax1.grid(True, axis="y", ls=":", alpha=0.4)
    ax1.legend(loc="center right", fontsize=8.0)
    figstyle.save(fig, fig_dir / "12_energy_budget_and_passivity.png")
    plt.close(fig)

    # 13. Conversion efficiency vs converter length L (Design 1 MMI sweep)
    sweep_L = []
    sweep_T = []
    sweep_R = []
    for L_um in SWEEP_LENGTHS_UM:
        t_key = primary_key if np.isclose(L_um, 6.0) else f"length_L{L_um:.1f}um"
        if t_key in tasks_data:
            td = tasks_data[t_key]
            i0 = int(np.argmin(np.abs(td["wavelengths"] - WL_REF)))
            sweep_L.append(float(L_um))
            sweep_T.append(float(np.abs(td["S21"][i0]) ** 2))
            sweep_R.append(float(np.abs(td["S11"][i0]) ** 2))

    if len(sweep_L) >= 2:
        sweep_L = np.asarray(sweep_L)
        sweep_T = np.asarray(sweep_T)
        sweep_R = np.asarray(sweep_R)
        L_taper_sec = sweep_L - L_MMI * 1e6

        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.0)
        ax0.plot(
            sweep_L,
            sweep_T * 100.0,
            "o-",
            color="#1b9e77",
            lw=2.2,
            ms=6.5,
            label="Wakis 2D EIM (Kottke) $T(1550\\;\\mathrm{nm})$",
        )
        # Coupled-mode adiabatic saturation guide: T_max - A * exp(-L_taper / L_c)
        L_dense = np.linspace(sweep_L.min(), sweep_L.max(), 200)
        t_asym = sweep_T[-1]
        t_guide = t_asym - (t_asym - sweep_T[0]) * np.exp(
            -(L_dense - sweep_L[0]) / 0.65
        )
        ax0.plot(
            L_dense,
            t_guide * 100.0,
            "--",
            color="#2b5c8f",
            lw=1.6,
            label="Adiabatic taper saturation guide",
        )
        ax0.axhspan(
            96.5,
            97.5,
            color="#d95f02",
            alpha=0.16,
            label="Deng et al. (2014) plateau (96.5–97.5%)",
        )
        ax0.set(
            xlabel="Total converter length L = L_mmi + L_taper [µm]",
            ylabel="Conversion efficiency T(1550 nm) [%]",
            title="(a) Conversion efficiency vs converter length L",
        )
        ax0.grid(True, ls=":", alpha=0.4)
        ax0.legend(loc="lower right", fontsize=8.0)

        il_db = -10.0 * np.log10(np.maximum(sweep_T, 1e-12))
        rl_db = 10.0 * np.log10(np.maximum(sweep_R, 1e-12))
        ax1_r = ax1.twinx()
        l1 = ax1.plot(
            L_taper_sec,
            il_db,
            "o-",
            color="#d95f02",
            lw=2.0,
            ms=6.0,
            label=r"Insertion loss $-10\log_{10}T$ [dB]",
        )
        l2 = ax1_r.plot(
            L_taper_sec,
            rl_db,
            "s--",
            color="#7570b3",
            lw=1.8,
            ms=5.5,
            label=r"Return loss $10\log_{10}R$ [dB]",
        )
        ax1.set_xlabel("Slot-rail taper section length L_taper = L − 1.38 µm [µm]")
        ax1.set_ylabel("Insertion loss [dB]", color="#d95f02")
        ax1_r.set_ylabel("Return loss [dB]", color="#7570b3")
        ax1.set_title("(b) Insertion loss & return loss vs slot-rail taper length")
        ax1.grid(True, ls=":", alpha=0.4)
        lines = l1 + l2
        ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=8.0)
        figstyle.save(fig, fig_dir / "13_conversion_efficiency_vs_length.png")
        plt.close(fig)

    # 14 & 15. Multi-resolution convergence & spectral comparison (with under-resolved demo)
    levels = [
        lvl
        for lvl in ("L1", "L2", "L3")
        if f"kottke_{lvl}" in tasks_data and f"staircase_{lvl}" in tasks_data
    ]
    if len(levels) >= 2 and "kottke_reference" in tasks_data:
        cpws = [float(tasks_data[f"kottke_{lvl}"]["cpw"]) for lvl in levels]
        feat_cells = [
            float(
                tasks_data[f"kottke_{lvl}"].get(
                    "cells_across_feature", tasks_data[f"kottke_{lvl}"]["slot_cells"]
                )
            )
            for lvl in levels
        ]

        def _t_at_1550(d):
            w = d["wavelengths"]
            i0 = int(np.argmin(np.abs(w - WL_REF)))
            return float(np.abs(d["S21"][i0]) ** 2)

        def _r_mean(d):
            w = d["wavelengths"] * 1e9
            m_b = (w >= 1515.0) & (w <= 1585.0)
            return float(np.mean(np.abs(d["S11"][m_b]) ** 2))

        t_ref = _t_at_1550(tasks_data["kottke_reference"])
        ref_cpw = float(tasks_data["kottke_reference"]["cpw"])
        ref_cells = float(
            tasks_data["kottke_reference"].get(
                "cells_across_feature", tasks_data["kottke_reference"]["slot_cells"]
            )
        )
        tk_vals = [_t_at_1550(tasks_data[f"kottke_{lvl}"]) for lvl in levels]
        ts_vals = [_t_at_1550(tasks_data[f"staircase_{lvl}"]) for lvl in levels]

        err_k = [max(abs(v - t_ref), 1e-4) for v in tk_vals]
        err_s = [max(abs(v - t_ref), 1e-4) for v in ts_vals]
        rk_vals = [_r_mean(tasks_data[f"kottke_{lvl}"]) for lvl in levels]
        rs_vals = [_r_mean(tasks_data[f"staircase_{lvl}"]) for lvl in levels]

        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.4)
        res = np.asarray(cpws, dtype=float)
        ax0.loglog(
            res,
            err_s,
            "s-",
            color="#d95f02",
            lw=2.1,
            ms=6.5,
            label="Resolved Staircase $|T - T_{\\mathrm{ref}}|$",
        )
        ax0.loglog(
            res,
            err_k,
            "o-",
            color="#1b9e77",
            lw=2.1,
            ms=6.5,
            label="Resolved Kottke $|T - T_{\\mathrm{ref}}|$",
        )
        ax0.loglog(
            res,
            err_s[0] * (res[0] / res) ** 1,
            "--",
            color="#d95f02",
            alpha=0.50,
            lw=1.2,
            label=r"$\mathcal{O}(\Delta x^1)$ guide",
        )
        ax0.loglog(
            res,
            err_k[0] * (res[0] / res) ** 2,
            ":",
            color="#1b9e77",
            alpha=0.60,
            lw=1.4,
            label=r"$\mathcal{O}(\Delta x^2)$ guide",
        )
        for idx_l, (lvl_i, r_i, fc_i, es_i, ek_i) in enumerate(
            zip(levels, res, feat_cells, err_s, err_k)
        ):
            off_s = (10, -11) if idx_l == 0 else ((0, 7) if idx_l == 1 else (-12, -13))
            off_k = (8, 6) if idx_l < 2 else (-14, 7)
            ax0.annotate(
                f"{lvl_i}: {fc_i:.1f} cells",
                xy=(r_i, es_i),
                xytext=off_s,
                textcoords="offset points",
                fontsize=7.5,
                ha="center",
                color="#d95f02",
            )
            ax0.annotate(
                f"{lvl_i}: {fc_i:.1f} cells",
                xy=(r_i, ek_i),
                xytext=off_k,
                textcoords="offset points",
                fontsize=7.5,
                ha="center",
                color="#1b9e77",
            )

        if "kottke_nonuniform" in tasks_data:
            d_nu_c = tasks_data["kottke_nonuniform"]
            nu_cpw = float(d_nu_c["cpw"])
            nu_cells = float(d_nu_c.get("cells_across_feature", d_nu_c["slot_cells"]))
            err_nu = max(abs(_t_at_1550(d_nu_c) - t_ref), 1e-4)
            r_nu = _r_mean(d_nu_c)
            ax0.loglog(
                [nu_cpw],
                [err_nu],
                "*",
                color="#e7298a",
                ms=11.0,
                zorder=9,
                label=f"Graded non-uniform Kottke ({int(d_nu_c['n_cells']):,} cells, {nu_cells:.0f} slot cells)",
            )
            ax1.loglog(
                [nu_cpw],
                [r_nu],
                "*",
                color="#e7298a",
                ms=11.0,
                zorder=9,
                label=f"Graded non-uniform Kottke ({int(d_nu_c['n_cells']):,} cells)",
            )

        has_under = (
            "underresolved_kottke" in tasks_data
            and "underresolved_staircase" in tasks_data
        )
        if has_under:
            uk = tasks_data["underresolved_kottke"]
            us = tasks_data["underresolved_staircase"]
            u_cpw = float(uk["cpw"])
            u_cells = float(uk.get("cells_across_feature", uk["slot_cells"]))
            err_uk = max(abs(_t_at_1550(uk) - t_ref), 1e-4)
            err_us = max(abs(_t_at_1550(us) - t_ref), 1e-4)
            r_uk = _r_mean(uk)
            r_us = _r_mean(us)
            ax0.loglog(
                [u_cpw],
                [err_us],
                "^",
                color="#b2182b",
                mfc="none",
                mew=1.8,
                ms=8.5,
                label=f"Under-resolved Staircase (no override, {u_cells:.2f} cells)",
            )
            ax0.loglog(
                [u_cpw],
                [err_uk],
                "D",
                color="#2166ac",
                mfc="none",
                mew=1.8,
                ms=7.5,
                label=f"Under-resolved Kottke (no override, {u_cells:.2f} cells)",
            )
            ax0.annotate(
                f"Under-resolved ({u_cells:.2f} cells)",
                xy=(u_cpw, err_us),
                xytext=(9, 3),
                textcoords="offset points",
                fontsize=7.3,
                color="#b2182b",
            )
            ax0.annotate(
                f"Under-resolved ({u_cells:.2f} cells)",
                xy=(u_cpw, err_uk),
                xytext=(9, 2),
                textcoords="offset points",
                fontsize=7.3,
                color="#2166ac",
            )

        ax0.margins(x=0.14)
        ax0.set_ylim(4.2e-5, 6.5e-2)
        ax0.set(
            xlabel="Background resolution [cpw] & cells across 100 nm slot",
            ylabel=rf"Error $|T(1550\;\mathrm{{nm}}) - T_{{\mathrm{{ref}}}}|$ (ref: cpw={ref_cpw:.0f}, N={ref_cells:.0f} cells)",
            title="(a) Conversion efficiency error vs independent reference",
        )
        ax0.xaxis.set_minor_formatter(plt.NullFormatter())
        ax0.set_xticks(res)
        ax0.set_xticklabels(
            [f"{int(r)} cpw\n({fc:.0f} cells, {lvl})" for r, lvl, fc in zip(res, levels, feat_cells)]
        )
        ax0.grid(True, which="both", ls=":", alpha=0.4)
        ax0.legend(loc="lower left", fontsize=6.4, framealpha=0.92)

        ax1.loglog(
            res,
            rs_vals,
            "s-",
            color="#d95f02",
            lw=2.1,
            ms=6.5,
            label="Resolved Staircase mean $|S_{11}|^2$",
        )
        ax1.loglog(
            res,
            rk_vals,
            "o-",
            color="#1b9e77",
            lw=2.1,
            ms=6.5,
            label="Resolved Kottke mean $|S_{11}|^2$",
        )
        for idx_l, (lvl_i, r_i, fc_i, rs_i, rk_i) in enumerate(
            zip(levels, res, feat_cells, rs_vals, rk_vals)
        ):
            off_rs = (12, 4) if idx_l == 0 else ((0, -12) if idx_l == 1 else (-14, -13))
            off_rk = (10, 6) if idx_l == 0 else ((0, -12) if idx_l == 1 else (-12, -12))
            ax1.annotate(
                f"{lvl_i}: {fc_i:.1f} cells",
                xy=(r_i, rs_i),
                xytext=off_rs,
                textcoords="offset points",
                fontsize=7.3,
                ha="center",
                color="#d95f02",
            )
            ax1.annotate(
                f"{lvl_i}: {fc_i:.1f} cells",
                xy=(r_i, rk_i),
                xytext=off_rk,
                textcoords="offset points",
                fontsize=7.3,
                ha="center",
                color="#1b9e77",
            )
        if has_under:
            ax1.loglog(
                [u_cpw],
                [r_us],
                "^",
                color="#b2182b",
                mfc="none",
                mew=1.8,
                ms=8.5,
                label=f"Under-resolved Staircase ({u_cells:.2f} cells)",
            )
            ax1.loglog(
                [u_cpw],
                [r_uk],
                "D",
                color="#2166ac",
                mfc="none",
                mew=1.8,
                ms=7.5,
                label=f"Under-resolved Kottke ({u_cells:.2f} cells)",
            )
            ax1.annotate(
                f"Under-resolved ({u_cells:.2f} cells)",
                xy=(u_cpw, r_us),
                xytext=(9, 3),
                textcoords="offset points",
                fontsize=7.2,
                color="#b2182b",
            )
            ax1.annotate(
                f"Under-resolved ({u_cells:.2f} cells)",
                xy=(u_cpw, r_uk),
                xytext=(9, -10),
                textcoords="offset points",
                fontsize=7.2,
                color="#2166ac",
            )
        ax1.margins(x=0.14)
        ax1.set_ylim(3.8e-4, 4.4e-3)
        ax1.set(
            xlabel="Background resolution [cpw] & cells across 100 nm slot",
            ylabel=r"Band-averaged reflection $\langle |S_{11}|^2 \rangle$",
            title="(b) Spurious reflection vs resolution",
        )
        ax1.xaxis.set_minor_formatter(plt.NullFormatter())
        ax1.set_xticks(res)
        ax1.set_xticklabels(
            [f"{int(r)} cpw\n({fc:.0f} cells, {lvl})" for r, lvl, fc in zip(res, levels, feat_cells)]
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="upper center", fontsize=6.6, framealpha=0.92)

        figstyle.save(fig, fig_dir / "14_kottke_vs_staircase_convergence.png")
        plt.close(fig)

        # 15. Spectra across resolved levels + under-resolved warning pair + reference
        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.3)
        cols = {"L1": "#7570b3", "L2": "#d95f02", "L3": "#1b9e77"}
        if has_under:
            for avg_u, ls_u, col_u in (
                ("kottke", "-.", "#2166ac"),
                ("staircase", ":", "#b2182b"),
            ):
                d_u = tasks_data[f"underresolved_{avg_u}"]
                w_u = d_u["wavelengths"] * 1e9
                idx_u = np.argsort(w_u)
                w_su = w_u[idx_u]
                m_bu = (w_su >= 1505.0) & (w_su <= 1595.0)
                t_db_u = 10.0 * np.log10(
                    np.maximum(np.abs(d_u["S21"][idx_u][m_bu]) ** 2, 1e-12)
                )
                r_db_u = 10.0 * np.log10(
                    np.maximum(np.abs(d_u["S11"][idx_u][m_bu]) ** 2, 1e-12)
                )
                fc_u = float(d_u.get("cells_across_feature", d_u["slot_cells"]))
                lbl_u = (
                    f"Under-resolved {avg_u} (cpw={int(d_u['cpw'])}, "
                    f"{fc_u:.2f} cells, no override)"
                )
                ax0.plot(w_su[m_bu], t_db_u, color=col_u, ls=ls_u, lw=1.6, label=lbl_u)
                ax1.plot(w_su[m_bu], r_db_u, color=col_u, ls=ls_u, lw=1.6, label=lbl_u)

        for lvl in levels:
            for avg, ls in (("kottke", "-"), ("staircase", "--")):
                d = tasks_data[f"{avg}_{lvl}"]
                w = d["wavelengths"] * 1e9
                idx_s = np.argsort(w)
                w_s = w[idx_s]
                m_b = (w_s >= 1505.0) & (w_s <= 1595.0)
                t_db = 10.0 * np.log10(
                    np.maximum(np.abs(d["S21"][idx_s][m_b]) ** 2, 1e-12)
                )
                r_db = 10.0 * np.log10(
                    np.maximum(np.abs(d["S11"][idx_s][m_b]) ** 2, 1e-12)
                )
                fc_v = float(d.get("cells_across_feature", d["slot_cells"]))
                lbl = f"{avg.capitalize()} {lvl} (cpw={int(d['cpw'])}, {fc_v:.0f} cells)"
                ax0.plot(w_s[m_b], t_db, color=cols[lvl], ls=ls, lw=1.9, label=lbl)
                ax1.plot(w_s[m_b], r_db, color=cols[lvl], ls=ls, lw=1.9, label=lbl)

        if "kottke_nonuniform" in tasks_data:
            d_nu_s = tasks_data["kottke_nonuniform"]
            w_n = d_nu_s["wavelengths"] * 1e9
            idx_n = np.argsort(w_n)
            w_sn = w_n[idx_n]
            m_bn = (w_sn >= 1505.0) & (w_sn <= 1595.0)
            t_db_n = 10.0 * np.log10(
                np.maximum(np.abs(d_nu_s["S21"][idx_n][m_bn]) ** 2, 1e-12)
            )
            r_db_n = 10.0 * np.log10(
                np.maximum(np.abs(d_nu_s["S11"][idx_n][m_bn]) ** 2, 1e-12)
            )
            lbl_n = f"Graded non-uniform Kottke ({int(d_nu_s['n_cells']):,} cells)"
            ax0.plot(w_sn[m_bn], t_db_n, color="#e7298a", ls="-.", lw=1.9, label=lbl_n)
            ax1.plot(w_sn[m_bn], r_db_n, color="#e7298a", ls="-.", lw=1.9, label=lbl_n)

        d_ref = tasks_data["kottke_reference"]
        w_r = d_ref["wavelengths"] * 1e9
        idx_r = np.argsort(w_r)
        w_sr = w_r[idx_r]
        m_br = (w_sr >= 1505.0) & (w_sr <= 1595.0)
        t_db_r = 10.0 * np.log10(
            np.maximum(np.abs(d_ref["S21"][idx_r][m_br]) ** 2, 1e-12)
        )
        r_db_r = 10.0 * np.log10(
            np.maximum(np.abs(d_ref["S11"][idx_r][m_br]) ** 2, 1e-12)
        )
        lbl_r = f"Reference Kottke (cpw={int(ref_cpw)}, {ref_cells:.0f} cells)"
        ax0.plot(w_sr[m_br], t_db_r, color="#111111", ls="-", lw=2.2, label=lbl_r)
        ax1.plot(w_sr[m_br], r_db_r, color="#111111", ls="-", lw=2.2, label=lbl_r)

        ax0.set(
            xlabel="Wavelength λ [nm]",
            ylabel=r"Guided slot transmission $10\log_{10}|S_{21}|^2$ [dB]",
            title="(a) Insertion loss: resolved ladder vs under-resolved warning",
        )
        ax0.grid(True, ls=":", alpha=0.4)
        ax0.legend(loc="lower right", fontsize=6.4)

        ax1.set(
            xlabel="Wavelength λ [nm]",
            ylabel=r"Return loss $10\log_{10}|S_{11}|^2$ [dB]",
            title="(b) Return loss: resolved ladder vs under-resolved warning",
        )
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.legend(loc="best", fontsize=6.4)
        figstyle.save(fig, fig_dir / "15_resolution_spectra_comparison.png")
        plt.close(fig)

    # 16. Output slot port time-domain waveform and exponential ring-down
    if "trace_t" in primary and primary["trace_t"].size > 0:
        t_fs = primary["trace_t"] * 1e15
        ey_tr = primary["trace_Ey"]
        ey_norm = ey_tr / max(float(np.max(np.abs(ey_tr))), 1e-30)
        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=10.6, height=3.8)
        ax0.plot(t_fs, ey_norm, color="#2b5c8f", lw=1.4)
        ax0.set(
            xlabel="Time t [fs]",
            ylabel=r"Normalized $E_y(t) / \max|E_y|$",
            title="(a) Output slot port time-domain waveform",
        )
        ax0.grid(True, ls=":", alpha=0.4)

        env = np.maximum(ey_norm**2, 1e-10)
        ax1.semilogy(
            t_fs, env, color="#d95f02", lw=1.4, label=r"$|E_y(t)|^2 / |E_y|_{\max}^2$"
        )
        ax1.axhline(
            1e-4, color="0.4", ls="--", lw=1.2, label="1e-4 energy decay threshold"
        )
        ax1.set(
            xlabel="Time t [fs]",
            ylabel="Normalized instantaneous power (log scale)",
            ylim=(1e-8, 2.0),
            title="(b) Ring-down & non-resonant termination audit",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=8.5)
        figstyle.save(fig, fig_dir / "16_port_time_trace.png")
        plt.close(fig)

    print(f"Rendered 20 figures in {fig_dir}")


# %% [markdown]
# ## 7. Limits, failure modes & when 2D EIM differs from 3D FDTD
#
# ### 7.1 Failure mode: what goes wrong when a gap is resolved by only 1–2 cells
#
# When a narrow oxide slot ($g = 100\;\text{nm}$) or taper tip gap
# ($d = 60\;\text{nm}$) is sampled by only $1\text{--}2$ Yee cells—as in the
# `underresolved_kottke` / `underresolved_staircase` demonstration pair at
# $\text{cpw} = 10$ with no mesh override ($\Delta y = 51.03\;\text{nm}$,
# $1.96$ cells across the slot)—several numerical artifacts arise:
#
# 1. **Staircased gap-width quantization**: With `averaging="staircase"`, each
#    cell's fill fraction $w \in [0, 1]$ is rounded to $\{0, 1\}$. Moving a
#    $100\;\text{nm}$ slot wall across a $51\;\text{nm}$ grid line snaps the
#    discretized gap between $1$, $2$, or $3$ cells ($51$, $102$, or
#    $153\;\text{nm}$), perturbing the slot capacitance, the slot-mode effective
#    index $n_{\text{eff,slot}}$, evanescent rail-to-rail coupling, or any
#    Bragg/cavity resonance wavelength tied to the slot mode, and depressing
#    the Design 1 conversion efficiency from $97.49\%$ (`R = (40, 16)`
#    reference; $97.47\%$ at `L3 = (20, 8)`) down to $94.00\%$
#    ($|T - T_{\text{ref}}| = 3.48\times 10^{-2}$).
# 2. **Why subpixel averaging alone is not a substitute for resolving the gap**:
#    Kottke harmonic/arithmetic permittivity smoothing restores the first-order
#    integrated displacement jump across a partially filled cell, which keeps
#    the net transmission of a straight uniform slot close to the converged
#    value even at $1.96$ cells. However, a $1\text{--}2$-cell grid still cannot
#    resolve the transverse $\cosh(\gamma_s y)$ field variation inside the gap,
#    cannot separate two opposing dielectric interfaces that fall inside the
#    same cell (such as the $60\;\text{nm}$ finger-to-strip gap or a narrowing
#    taper tip), and under-samples the port eigenmode profile used for modal
#    overlap projection. Enforcing both $\text{cpw} \ge 10$ and $N \ge 4$ cells
#    across the smallest feature via `mesh_overrides` eliminates this failure
#    mode at minimal extra cell count.
#
# ### 7.2 Physical differences between 2D EIM and 3D FDTD
#
# Our 2D EIM simulations capture the fundamental physics of strip-to-slot mode
# conversion—the $(n_{\text{high}}/n_{\text{low}})^2 = 4.14$ normal electric-field
# jump across the $100\;\text{nm}$ slot walls, the two-fold symmetric MMI imaging
# in Design 1 ($97.47\%$ at `L3` and $97.49\%$ at `R` at $1550\;\text{nm}$, matching Deng
# *et al.*'s $96.5\text{--}97.5\%$ [1]), and the $99.66\text{--}99.80\%$ adiabatic
# conversion of Designs 2 and 3—at under $1\%$ of the computational cost of 3D
# FDTD. Three physical differences between 2D EIM and 3D FDTD should be kept in
# mind when finalizing mask dimensions for fabrication:
#
# 1. **Vertical mode mismatch at the slot opening**: In 3D SOI ($h_{\text{Si}} = 250\;\text{nm}$),
#    the vertical confinement profile $\psi(z)$ inside the $100\;\text{nm}$
#    oxide slot expands slightly along $z$ compared to the $\psi(z)$ inside a
#    solid $400\;\text{nm}$ silicon strip, because the intense slot field fringes
#    across the top and bottom $\text{Si}/\text{SiO}_2$ corners. Factoring a
#    single vertical profile $\psi(z)$ in 2D EIM neglects vertical radiation
#    loss at the abrupt slot termination ($x = L_{\text{mmi}} = 1.38\;\mu\text{m}$
#    in Design 1).
# 2. **Normal-field jump magnitude**: In 3D bulk silicon, the normal jump right
#    at the vertical $\text{Si}/\text{SiO}_2$ wall is
#    $(n_{\text{Si}}/n_{\text{SiO}_2})^2 = (3.47/1.44)^2 = 5.81$, whereas in
#    2D EIM the core index is replaced by $n_{\text{eff,slab}} = 2.9306$, giving
#    $(n_{\text{eff,slab}}/n_{\text{SiO}_2})^2 = 4.14$.
# 3. **MMI beat-length shift**: The $\text{TE}_0\text{--}\text{TE}_2$ beat length
#    $L_\pi = \pi / (\beta_0 - \beta_2)$ in the $1.24\;\mu\text{m}$ MMI section
#    depends on the lateral index contrast, shifting the optimal $L_{\text{mmi}}$
#    by $\sim 5\text{--}8\%$ between 2D EIM and 3D FDTD.
#
# ## 8. Exercises
#
# 1. **Slot gap sensitivity**: Vary `GAP_SLOT` over $g \in \{60, 80, 100, 140\}\;\text{nm}$
#    in Design 1 (`build_mmi_converter`) while keeping outer slot width
#    $w_{\text{slot}} = 620\;\text{nm}$. How does the conversion efficiency
#    $T(1550\;\text{nm})$ and slot confinement factor $\Gamma_{\text{slot}}(g)$
#    change?
# 2. **Blunt tip tolerance in Design 2**: In CMOS lithography, the sharp tips of
#    Design 2 (`build_symmetric_taper_converter`) round to a finite width
#    $w_{\text{tip}} \approx 40\text{--}80\;\text{nm}$. Modify `asymmetric_taper`
#    calls so that the tips terminate at width $w_{\text{tip}} = 60\;\text{nm}$
#    and compare the degraded return loss $|S_{11}|^2$ against the tip-free
#    Design 1 MMI converter.
# 3. **Reciprocity (slot-to-strip conversion)**: Run
#    `pic.run_s_parameters(drive=["in", "out"])` on Design 1 and verify that
#    $|S_{12}(\lambda) - S_{21}(\lambda)| < 10^{-3}$ across the entire
#    $1500\text{--}1600\;\text{nm}$ C-band.
#
# ## 9. References
#
# 1. Q. Deng, L. Liu, X. Li, and Z. Zhou, "Strip-slot waveguide mode converter
#    based on symmetric multimode interference," *Opt. Lett.* **39**(19),
#    5665–5668 (2014), DOI: [10.1364/OL.39.005665](https://doi.org/10.1364/OL.39.005665).
# 2. Z. Wang, N. Zhu, Y. Tang, L. Wosinski, D. Dai, and S. He, "Ultracompact
#    low-loss coupler between strip and slot waveguides," *Opt. Lett.* **34**(10),
#    1498–1500 (2009), DOI: [10.1364/OL.34.001498](https://doi.org/10.1364/OL.34.001498).
# 3. N.-N. Feng, R. Sun, L. C. Kimerling, and J. Michel, "Lossless strip-to-slot
#    waveguide transformer," *Opt. Lett.* **32**(10), 1250–1252 (2007),
#    DOI: [10.1364/OL.32.001250](https://doi.org/10.1364/OL.32.001250).
# 4. V. R. Almeida, Q. Xu, C. A. Barrios, and M. Lipson, "Guiding and confining
#    light in void nanostructure," *Opt. Lett.* **29**(11), 1209–1211 (2004),
#    DOI: [10.1364/OL.29.001209](https://doi.org/10.1364/OL.29.001209).
# 5. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing of
#    discretized numerical methods," *Phys. Rev. E* **77**, 036611 (2008),
#    DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="17. Silicon Strip-to-Slot Waveguide Mode Converters (2D EIM)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run resolved 2D quick study (L1=(10,4), L2=(14,6), ref L3=(20,8)) and render all 16 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run full 2D study across all 15 tasks (L1=(10,4), L2=(14,6), L3=(20,8), ref R=(40,16)) and render all figures",
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
