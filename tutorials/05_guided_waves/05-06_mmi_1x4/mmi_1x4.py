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
# # Tutorial 5.6 · 1×4 multimode-interference power splitter
#
# **Abstract.** Multimode interference (MMI) couplers exploit the self-imaging
# property of wide multimode waveguides to split or combine optical signals with
# broad optical bandwidth and relaxed lithographic tolerances. By embedding a
# nanoscale low-index gallium nitride ($\text{GaN}$, $n = 2.305$) slot layer
# symmetrically between two high-index silicon ($\text{Si}$, $n = 3.48$) rails,
# **Malka *et al.* (2016)** [1] demonstrated a compact $1 \times 4$ power
# splitter operating in the fundamental vertical-slot TM mode, where the normal
# electric field $E_z$ is strongly enhanced inside the $100\;\text{nm}$
# $\text{GaN}$ slot by dielectric continuity ($E_{z,\text{GaN}} / E_{z,\text{Si}} = \varepsilon_{\text{Si}} / \varepsilon_{\text{GaN}} = 2.28$).
# In this tutorial we model the 34-vertex $1 \times 4$ $\text{Si–GaN–Si}$ MMI
# splitter in Wakis using a **5-layer effective-index reduction (2D EIM,
# $n_{\text{eff,slot}} = 2.9842$)** and **Kottke subpixel permittivity
# smoothing**, compare the self-imaging length against Soldano & Pennings
# (1995) [2] theory and a 1D eigenmode expansion (EME), extract broadband
# $5 \times 5$ S-parameters across the $1500\text{--}1600\;\text{nm}$ C/L
# telecom window, audit pairwise mirror symmetry ($< 2 \times 10^{-7}$) and 2D
# Poynting flux conservation ($1.0000$), evaluate the $3 \times 3$ and extended
# $5 \times 6$ $(W_{\text{MMI}}, L_{\text{MMI}})$ design sweeps (bracketing the
# 2D EIM minimum-excess-loss optimum $0.325\;\text{dB}$ at $(5.00, 11.20)\;\mu\text{m}$
# and the global minimum inner/outer imbalance $0.0027$ at $(5.00, 11.30)\;\mu\text{m}$
# strictly in the interior of the grid), and analyze Kottke vs staircase
# convergence against an independent $40\;\text{cpw}$ reference run.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Reduce a 5-layer symmetric vertical slot stack
#    ($\text{SiO}_2 / \text{Si}(300\;\text{nm}) / \text{GaN}(100\;\text{nm}) / \text{Si}(300\;\text{nm}) / \text{SiO}_2$)
#    to its fundamental vertical-slot TM effective index
#    $n_{\text{eff,slot}}(1550\;\text{nm}) = 2.9842$ ($\Gamma_{\text{slot}} = 38.3\%$,
#    $\Gamma_{\text{stack}} = 93.7\%$) for 2D in-plane FIT simulation.
# 2. Construct the 34-vertex $1 \times 4$ MMI polygon (input waveguide, input
#    linear taper, multimode interference box, four output tapers, and four
#    output waveguides) as a single seamless `PolySlab` with Kottke (2008) [3]
#    subpixel smoothing.
# 3. Predict the beat length $L_\pi = \frac{4 n_r W_e^2}{3 \lambda_0}$ and first
#    4-fold symmetric self-imaging length $L_{\text{MMI}} = \frac{3 L_\pi}{16}$
#    analytically and via 1D eigenmode expansion (EME), and explain why
#    non-paraxial modal dispersion and the 2D lateral-TM Goos–Hänchen shift
#    offset the 2D EIM optimum from the 3D lateral-TE dimensions.
# 4. Quantify output uniformity, excess insertion loss, pairwise mirror
#    symmetry ($T_1 = T_4$, $T_2 = T_3$), higher-order Mode 1 excitation at the
#    outer ports, and subpixel vs staircase mesh convergence (both single-wavelength
#    and band-averaged RMS) across the full-resolution study.
#
# ## Contents
#
# 1. [Physical background & self-imaging theory](#1.-Physical-background-&-self-imaging-theory)
# 2. [5-layer vertical-slot Effective-Index Method (EIM)](#2.-5-layer-vertical-slot-Effective-Index-Method-(EIM))
# 3. [Parametric 34-vertex MMI geometry construction](#3.-Parametric-34-vertex-MMI-geometry-construction)
# 4. [Subpixel permittivity smoothing vs staircasing](#4.-Subpixel-permittivity-smoothing-vs-staircasing)
# 5. [Study tasks, resolution ladder & extended design sweep](#5.-Study-tasks,-resolution-ladder-&-extended-design-sweep)
# 6. [Generating the complete 15-figure diagnostic suite](#6.-Generating-the-complete-15-figure-diagnostic-suite)
# 7. [Full-resolution results: spectra, symmetry, mode purity & parameter sweep](#7.-Full-resolution-results:-spectra,-symmetry,-mode-purity-&-parameter-sweep)
# 8. [Resolution table & subpixel convergence analysis](#8.-Resolution-table-&-subpixel-convergence-analysis)
# 9. [Limits, failure modes & when 2D EIM breaks down](#9.-Limits,-failure-modes-&-when-2D-EIM-breaks-down)
# 10. [Exercises](#10.-Exercises)
# 11. [References](#11.-References)

# %% [markdown]
# ## 1. Physical background & self-imaging theory
#
# In a multimode waveguide of physical width $W_{\text{MMI}}$, slab effective
# index $n_r$, and cladding index $n_{\text{clad}}$, the lateral guided modes
# $m = 0, 1, 2, \dots$ have approximate transverse wavenumbers
# $k_{y,m} \approx (m + 1)\pi / W_e$, where the **Goos–Hänchen effective width**
# $W_e$ accounts for evanescent penetration into the lateral cladding [2]:
#
# $$W_e = W_{\text{MMI}} + \frac{\lambda_0}{\pi}\left(\frac{n_{\text{clad}}}{n_r}\right)^{2\sigma}\left(n_r^2 - n_{\text{clad}}^2\right)^{-1/2},$$
#
# with $\sigma = 0$ for lateral TE boundary conditions (tangential electric
# field at the vertical sidewalls $y = \pm W_{\text{MMI}}/2$, as for the
# $E_z$-polarized 3D vertical-slot mode) and $\sigma = 1$ for lateral TM
# boundary conditions (normal electric field $E_y$ at the vertical sidewalls, as
# in the 2D `Ex, Ey, Hz` FIT reduction).
#
# In the paraxial approximation, the longitudinal propagation constants
# $\beta_m = \sqrt{k_0^2 n_r^2 - k_{y,m}^2}$ obey a quadratic spacing law:
#
# $$\beta_0 - \beta_m \approx \frac{m(m + 2)\pi}{3 L_\pi}, \qquad L_\pi = \frac{\pi}{\beta_0 - \beta_1} \approx \frac{4 n_r W_e^2}{3 \lambda_0}.$$
#
# When the MMI box is excited **symmetrically** on axis ($y = 0$), only the even
# lateral modes ($m = 0, 2, 4, 6, \dots$) are excited (`restricted symmetric
# interference`), reducing the device length by a factor of $4$. The first
# $N$-fold symmetric self-image forms at axial distance [1, 2]:
#
# $$L_{\text{MMI}}^{(N)} = \frac{3 L_\pi}{4 N} = \frac{n_r W_e^2}{N \lambda_0}.$$
#
# For a $1 \times 4$ splitter ($N = 4$), $L_{\text{MMI}}^{(4)} = \frac{3}{16} L_\pi = \frac{n_r W_e^2}{4 \lambda_0}$,
# producing four equal-intensity images spaced laterally by $W_e / 4$. Mirror
# symmetry $y \mapsto -y$ guarantees pairwise identical output transmissions:
#
# $$T_1(\omega) = |S_{21}(\omega)|^2 = T_4(\omega) = |S_{51}(\omega)|^2 \;\text{(outer pair)}, \qquad T_2(\omega) = |S_{31}(\omega)|^2 = T_3(\omega) = |S_{41}(\omega)|^2 \;\text{(inner pair)},$$
#
# while passivity requires $\sum_{i=1}^{4} T_i(\omega) + |S_{11}(\omega)|^2 \le 1$.

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
import matplotlib.ticker as mticker
import numpy as np
from scipy.constants import c as c_light

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
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _helpers import (  # noqa: E402
    equivalent_3layer_core_index,
    mmi_analytic_estimates,
    mmi_discretization_diagnostics,
    mmi_eme_length_sweep,
    plot_slot_eim_stack,
    slot_slab_neff,
)

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# %% [markdown]
# ### Device parameters from Malka *et al.* (2016)
#
# In keeping with the Wakis PIC convention, propagation is oriented along $+x$,
# transverse waveguide width along $y$, and the vertical sandwich slot stack
# along $z$:
#
# | Parameter | Symbol | Value |
# |---|---|---|
# | Silicon rail refractive index ($1550\;\text{nm}$) | $n_{\text{Si}}$ | $3.48$ ($\varepsilon_r = 12.1104$) |
# | $\text{GaN}$ slot refractive index ($1550\;\text{nm}$) | $n_{\text{GaN}}$ | $2.305$ ($\varepsilon_r = 5.313025$) |
# | $\text{SiO}_2$ cladding refractive index ($1550\;\text{nm}$) | $n_{\text{SiO}_2}$ | $1.444$ ($\varepsilon_r = 2.085136$) |
# | Upper / lower $\text{Si}$ rail thickness | $H_{\text{Si}}$ | $300\;\text{nm}$ ($z \in [-350, -50]\;\text{nm} \cup [50, 350]\;\text{nm}$) |
# | Central $\text{GaN}$ slot thickness | $H_{\text{GaN}}$ | $100\;\text{nm}$ ($z \in [-50, +50]\;\text{nm}$, total stack $700\;\text{nm}$) |
# | Fundamental vertical-slot TM effective index ($1550\;\text{nm}$) | $n_{\text{eff,slot}}$ | $2.9842$ (equivalent 3-layer core $n_{\text{core,eq}} = 3.1108$) |
# | Input / output waveguide width | $W_1$ | $0.40\;\mu\text{m}$ ($n_{\text{eff,in}} = 2.4660$, $n_g = 3.4221$ at $1550\;\text{nm}$) |
# | Taper width at MMI interfaces | $W_2$ | $0.80\;\mu\text{m}$ (taper-to-taper gap $g_2 = 0.50\;\mu\text{m}$ at MMI face) |
# | Input linear taper length | $L_1$ | $2.00\;\mu\text{m}$ |
# | Output linear taper length | $L_2$ | $5.00\;\mu\text{m}$ |
# | Output edge-to-edge gap / center pitch | $g_1, \; p = W_1 + g_1$ | $0.90\;\mu\text{m}, \; 1.30\;\mu\text{m}$ ($y_c = \pm 0.65, \pm 1.95\;\mu\text{m}$) |
# | Baseline MMI lengths & widths | $(L_{\text{MMI}}, W_{\text{MMI}})$ | $(11.20, 5.00)\;\mu\text{m}$ (initial / $3\times 3$ sweep corner best), $(11.15, 4.90)\;\mu\text{m}$ (primary task) |
# | Extended 2D sweep grid (`--full`) | $W_{\text{MMI}} \times L_{\text{MMI}}$ | $\{4.80, 4.90, 5.00, 5.10, 5.20\}\;\mu\text{m} \times \{11.10, 11.15, 11.20, 11.25, 11.30, 11.35\}\;\mu\text{m}$ |

# %%
# Material refractive indices (Malka et al., Materials 9, 516, 2016)
N_SI = 3.48
N_GAN = 2.305
N_SIO2 = 1.444

# Vertical Si-GaN-Si sandwich slot stack dimensions [m]
H_SI = 0.30e-6
H_GAN = 0.10e-6
H_STACK = 2.0 * H_SI + H_GAN  # 0.70 um total stack height

# Wavelength band: 1500 nm to 1600 nm (centered at 1550 nm)
WL_MIN = 1.50e-6
WL_MAX = 1.60e-6
WL_REF = 1.55e-6
N_FREQS = 51

# Exact 5-layer vertical-slot TM mode effective index at 1550 nm (2.984168)
# and the equivalent 3-layer core index passed to PICSimulation (3.110774)
N_EFF_SLOT = slot_slab_neff(
    n_rail=N_SI,
    n_slot=N_GAN,
    n_clad=N_SIO2,
    h_rail=H_SI,
    h_slot=H_GAN,
    wavelength=WL_REF,
    polarization="TM",
)
N_CORE_EQ = equivalent_3layer_core_index(
    N_EFF_SLOT, n_clad=N_SIO2, h_total=H_STACK, wavelength=WL_REF
)

# Horizontal xy polygon dimensions [m] (propagation along +x, transverse along y)
W1 = 0.40e-6  # Input and output straight waveguide width
W2 = 0.80e-6  # Taper width at MMI input and output interfaces
L1 = 2.00e-6  # Input linear taper length
L2 = 5.00e-6  # Output linear taper length
G1 = 0.90e-6  # Edge-to-edge gap between adjacent output waveguides
PITCH = W1 + G1  # 1.30 um center-to-center pitch
Y_OUT_CENTERS = np.array([-1.5 * PITCH, -0.5 * PITCH, 0.5 * PITCH, 1.5 * PITCH])

# Initial and optimal MMI box dimensions [m], initial 3x3 sweep grid (tasks 7..15),
# and extended 5x6 sweep grid (tasks 16..36) placing the 2D EIM optimum in the interior
L_MMI_INIT = 11.20e-6
W_MMI_INIT = 5.00e-6
L_MMI_OPT = 11.15e-6
W_MMI_OPT = 4.90e-6

SWEEP_L_UM_INIT = (11.10, 11.15, 11.20)
SWEEP_W_UM_INIT = (4.80, 4.90, 5.00)
SWEEP_L_UM = (11.10, 11.15, 11.20, 11.25, 11.30, 11.35)
SWEEP_W_UM = (4.80, 4.90, 5.00, 5.10, 5.20)

# Input/output straight lead extensions and port x-positions
L_LEAD_IN = 2.50e-6
L_LEAD_OUT = 2.50e-6
X_PORT_IN = -0.75e-6

# %% [markdown]
# ## 2. 5-layer vertical-slot Effective-Index Method (EIM)
#
# Unlike a standard 3-layer SOI strip waveguide, the Malka *et al.* (2016)
# splitter uses a **5-layer symmetric vertical slot stack** along $z$:
#
# $$\text{SiO}_2\;\text{cladding} \;\big|\; \text{Si}\;(300\;\text{nm}) \;\big|\; \text{GaN}\;(100\;\text{nm}) \;\big|\; \text{Si}\;(300\;\text{nm}) \;\big|\; \text{SiO}_2\;\text{cladding}.$$
#
# The device is driven in the **fundamental vertical-slot TM mode** (even
# transverse magnetic field $H_y(z) = H_y(-z)$ and normal electric field
# $E_z(z) \propto n^{-2}(z)\,H_y(z)$), which satisfies the 1D Sturm–Liouville
# equation:
#
# $$n^2(z)\frac{\mathrm{d}}{\mathrm{d}z}\left(\frac{1}{n^2(z)}\frac{\mathrm{d} H_y}{\mathrm{d} z}\right) + k_0^2\left[n^2(z) - n_{\text{eff,slot}}^2\right]H_y(z) = 0.$$
#
# Because $n_{\text{GaN}} = 2.305 < n_{\text{eff,slot}} < n_{\text{Si}} = 3.48$,
# the field is oscillatory inside the two $300\;\text{nm}$ silicon rails and
# hyperbolic-cosine ($\cosh(\gamma_{\text{slot}} z)$) inside the $100\;\text{nm}$
# $\text{GaN}$ slot, while continuity of $D_z = \varepsilon_0 n^2(z) E_z(z)$
# forces a discontinuity jump of $(n_{\text{Si}} / n_{\text{GaN}})^2 = 2.279$
# in $|E_z|$ at $z = \pm 50\;\text{nm}$. Solving the 5-layer transfer-matrix
# dispersion relation in `_helpers.slot_slab_neff` yields:
#
# $$n_{\text{eff,slot}}(1550\;\text{nm}) = 2.9842.$$
#
# Passing `N_CORE_EQ = 3.1108` on the $H_{\text{stack}} = 0.70\;\mu\text{m}$
# `PolySlab` ensures that `PICSimulation` sets `pic.n_eff_slab = 2.9842` in the
# 2D FIT domain.

# %% [markdown]
# ## 3. Parametric 34-vertex MMI geometry construction
#
# Following the device specification, the horizontal $xy$ footprint is a single
# counter-clockwise **34-vertex polygon** comprising:
#
# 1. **Input straight waveguide** ($x \in [-L_{\text{in}}, 0]$, width $W_1 = 0.40\;\mu\text{m}$).
# 2. **Input linear taper** ($x \in [0, L_1]$ with $L_1 = 2.0\;\mu\text{m}$, expanding from $W_1 = 0.40\;\mu\text{m}$ to $W_2 = 0.80\;\mu\text{m}$).
# 3. **MMI box** ($x \in [L_1, L_1 + L_{\text{MMI}}]$, width $W_{\text{MMI}}$).
# 4. **Four output linear tapers** ($x \in [L_1 + L_{\text{MMI}}, L_1 + L_{\text{MMI}} + L_2]$ with $L_2 = 5.0\;\mu\text{m}$, tapering from $W_2 = 0.80\;\mu\text{m}$ to $W_1 = 0.40\;\mu\text{m}$ at centers $y_c \in \{\pm 0.65, \pm 1.95\}\;\mu\text{m}$) and **four straight output waveguides** ($x > L_1 + L_{\text{MMI}} + L_2$).
#
# In 3D (`two_d=False`), `build_mmi_1x4` can emit the two-layer `PolySlab` pair
# (`mmi_layer1` spanning $z \in [-0.35, +0.35]\;\mu\text{m}$ in $\text{Si}$
# overridden by `mmi_layer2` on $z \in [-0.05, +0.05]\;\mu\text{m}$ in
# $\text{GaN}$). In 2D EIM (`two_d=True`), the single 34-vertex `PolySlab`
# carries the 5-layer vertical-slot effective index $n_{\text{eff,slot}} = 2.9842$.


# %%
def build_mmi_1x4(
    l_mmi=L_MMI_OPT,
    w_mmi=W_MMI_OPT,
    include_mode1_port=False,
    three_d_stack=False,
):
    """
    Construct the 34-vertex 1x4 MMI power splitter `PolySlab`, its waveguide
    `Port` list (`in`, `out1`..`out4`, and optional `out4_m1`), and the 2D
    simulation `domain` tuple.

    Parameters
    ----------
    l_mmi, w_mmi : float, optional
        Length and width of the multimode interference section [m].
    include_mode1_port : bool, optional
        If True, append `Port('out4_m1', ..., mode_index=1)` at the outer output
        waveguide to decompose fundamental Mode 0 vs higher-order Mode 1 power.
    three_d_stack : bool, optional
        If True, return the two-layer 3D `[mmi_layer1 (Si), mmi_layer2 (GaN)]`
        `PolySlab` list; otherwise return the single 2D EIM `PolySlab`.

    Returns
    -------
    structures : list of PolySlab
        Extruded 34-vertex MMI polygon slab(s).
    ports : list of Port
        Input port (`in`), four fundamental output ports (`out1`..`out4` from
        bottom `y = -1.95 um` to top `y = +1.95 um`), and optional `out4_m1`.
    domain : tuple of 4 float
        `(xmin, xmax, ymin, ymax)` [m] matching `(L_1 + L_MMI + L_2 + 3.0 um)`
        by `(W_MMI + 2.0 um)`.
    """
    x0 = -L_LEAD_IN
    x1 = 0.0
    x2 = L1
    x3 = L1 + float(l_mmi)
    x4 = x3 + L2
    x5 = x4 + L_LEAD_OUT

    # Counter-clockwise 34-vertex polygon in the (x, y) plane
    pts = [
        [x0, -0.5 * W1],
        [x1, -0.5 * W1],
        [x2, -0.5 * W2],
        [x2, -0.5 * w_mmi],
        [x3, -0.5 * w_mmi],
    ]
    for y_arm in Y_OUT_CENTERS:
        pts.extend(
            [
                [x3, y_arm - 0.5 * W2],
                [x4, y_arm - 0.5 * W1],
                [x5, y_arm - 0.5 * W1],
                [x5, y_arm + 0.5 * W1],
                [x4, y_arm + 0.5 * W1],
                [x3, y_arm + 0.5 * W2],
            ]
        )
    pts.extend(
        [
            [x3, +0.5 * w_mmi],
            [x2, +0.5 * w_mmi],
            [x2, +0.5 * W2],
            [x1, +0.5 * W1],
            [x0, +0.5 * W1],
        ]
    )
    verts = np.asarray(pts, dtype=float)
    assert verts.shape == (34, 2), f"Expected 34 vertices, got {verts.shape}"

    if three_d_stack:
        mmi_layer1 = PolySlab(
            verts,
            z_min=-0.5 * H_STACK,
            z_max=0.5 * H_STACK,
            material=n_to_eps(N_SI),
        )
        mmi_layer2 = PolySlab(
            verts,
            z_min=-0.5 * H_GAN,
            z_max=0.5 * H_GAN,
            material=n_to_eps(N_GAN),
        )
        structures = [mmi_layer1, mmi_layer2]
    else:
        slab = PolySlab(
            verts,
            z_min=-0.5 * H_STACK,
            z_max=0.5 * H_STACK,
            material=n_to_eps(N_CORE_EQ),
        )
        structures = [slab]

    x_out = x4 + 0.80e-6  # 18.95 um for L_MMI = 11.15 um (19.00 um for 11.20 um)
    ports = [
        Port(
            "in",
            center=(X_PORT_IN, 0.0, 0.0),
            size=(0.0, 2.0e-6, 2.5e-6),
            direction=+1,
        )
    ]
    for i, y_arm in enumerate(Y_OUT_CENTERS):
        ports.append(
            Port(
                f"out{i + 1}",
                center=(x_out, float(y_arm), 0.0),
                size=(0.0, 1.20e-6, 2.5e-6),
                direction=+1,
                mode_index=0,
            )
        )
    if include_mode1_port:
        ports.append(
            Port(
                "out4_m1",
                center=(x_out, float(Y_OUT_CENTERS[3]), 0.0),
                size=(0.0, 1.20e-6, 2.5e-6),
                direction=+1,
                mode_index=1,
            )
        )

    domain = (
        -1.50e-6,
        x4 + 1.50e-6,
        -(0.5 * float(w_mmi) + 1.00e-6),
        +(0.5 * float(w_mmi) + 1.00e-6),
    )
    return structures, ports, domain


# %% [markdown]
# ## 4. Subpixel permittivity smoothing vs staircasing
#
# In a $1 \times 4$ MMI splitter, two geometric features are acutely sensitive
# to Cartesian grid discretization:
#
# 1. **The MMI box width $W_{\text{MMI}}$**: Because the self-imaging length
#    scales quadratically with effective width ($L_{\text{MMI}} \propto W_e^2$),
#    a staircase rounding error of $\Delta W_y$ on the horizontal MMI walls
#    ($y = \pm W_{\text{MMI}}/2$) shifts the beat length by
#    $\Delta L_{\text{MMI}} / L_{\text{MMI}} \approx 2 \Delta W_y / W_e$,
#    causing first-order $\mathcal{O}(\Delta y)$ phase errors that accumulate
#    over the $11.15\;\mu\text{m}$ interference region and converge toward a
#    slightly different device width unless subpixel smoothing is used.
# 2. **The tilted sidewalls of the input ($L_1 = 2\;\mu\text{m}$) and four
#    output ($L_2 = 5\;\mu\text{m}$) linear tapers**: Staircasing turns the
#    $0.80\;\mu\text{m} \to 0.40\;\mu\text{m}$ taper sidewalls into discrete
#    steps that scatter guided power and perturb the $0.50\;\mu\text{m}$
#    taper-to-taper gaps at the MMI exit face.
#
# Wakis applies the diagonal **Kottke, Farjadpour & Johnson (2008)** [3]
# subpixel permittivity tensor on every cut cell, using harmonic averaging for
# the field component normal to the local polygon edge and arithmetic averaging
# for tangential components. As analyzed in detail in Section 8 (Fig. 14), at
# $20\;\text{cpw}$ ($\Delta y = 25.09\;\text{nm}$) staircasing truncates the
# $E_y$ core width to $195\Delta y = 4.8927\;\mu\text{m}$ ($\Delta W_y = -7.3\;\text{nm}$)
# and plateaus at $|T(1550\;\text{nm}) - T_{\text{ref}}| = 4.06 \times 10^{-3}$
# (band RMS $6.80 \times 10^{-3}$), whereas Kottke restores the effective MMI
# width to $4.8990\;\mu\text{m}$ and reaches $2.24 \times 10^{-3}$ (band RMS
# $4.68 \times 10^{-3}$).

# %% [markdown]
# ## 5. Study tasks, resolution ladder & extended design sweep
#
# The full-resolution study (`--full`, completed on Cloud Batch across all 37
# tasks `task_0` through `task_36`) comprises:
#
# - **Tasks `0`–`5` (Resolution & subpixel study on the $(11.15, 4.90)\;\mu\text{m}$ MMI)**:
#   `kottke` vs `staircase` at `coarse` ($10\;\text{cpw}$ in `--full`, $5\;\text{cpw}$ in `--quick`),
#   `medium` ($14\;\text{cpw}$ in `--full`, $7\;\text{cpw}$ in `--quick`), and
#   `fine` ($20\;\text{cpw}$ in `--full`, $10\;\text{cpw}$ in `--quick`). Every
#   full-mode resolution satisfies $\ge 10\;\text{cpw}$ in the 2D slab core and
#   $\ge 8$ cells across the smallest features ($W_1 = 0.40\;\mu\text{m}$
#   waveguide and $g_2 = 0.50\;\mu\text{m}$ MMI-face taper-to-taper gap).
# - **Task `6` (`kottke_reference`)**: Independent Kottke reference run at
#   $40\;\text{cpw}$ in `--full` ($20\;\text{cpw}$ in `--quick`) used as
#   $T_{\text{ref}}$ in the convergence analysis (never Richardson extrapolation
#   from the compared runs).
# - **Tasks `7`–`15` (Initial $3 \times 3$ $(L_{\text{MMI}}, W_{\text{MMI}})$ parameter sweep)**:
#   $W_{\text{MMI}} \in \{4.80, 4.90, 5.00\}\;\mu\text{m} \times L_{\text{MMI}} \in \{11.10, 11.15, 11.20\}\;\mu\text{m}$
#   at $20\;\text{cpw}$ ($7\;\text{cpw}$ in `--quick`), including `task_15`
#   (`sweep_W5.00_L11.20`, which also collects DFT fields, 2D flux-box budget,
#   and Mode 0/1 power share).
# - **Tasks `16`–`36` (Extended $5 \times 6$ parameter sweep in `--full`)**:
#   Because the optimum of the initial $3 \times 3$ sweep (`task_15` at
#   $W_{\text{MMI}} = 5.00\;\mu\text{m}$, $L_{\text{MMI}} = 11.20\;\mu\text{m}$:
#   imbalance $0.0075$, excess loss $0.325\;\text{dB}$) lay at the upper-right
#   corner of $[4.80, 5.00]\;\mu\text{m} \times [11.10, 11.20]\;\mu\text{m}$, we
#   kept `task_7`–`task_15` at the exact same indices and names and appended 21
#   tasks (`task_16`–`task_36`) spanning
#   $W_{\text{MMI}} \in \{4.80, 4.90, 5.00, 5.10, 5.20\}\;\mu\text{m} \times L_{\text{MMI}} \in \{11.10, 11.15, 11.20, 11.25, 11.30, 11.35\}\;\mu\text{m}$.
#   In the completed $5 \times 6$ grid, both the minimum-excess-loss point
#   (`task_15` at $(5.00, 11.20)\;\mu\text{m}$, $\text{EL}(1550\;\text{nm}) = 0.325\;\text{dB}$,
#   $T_{\text{tot}} = 0.9279$) and the global minimum-imbalance point (`task_23`
#   at $(5.00, 11.30)\;\mu\text{m}$, $|T_{\text{in}} - T_{\text{out}}| = 0.0027$,
#   uniformity $0.050\;\text{dB}$, $\text{EL} = 0.366\;\text{dB}$, band
#   $\langle \text{EL}\rangle = 0.401\;\text{dB}$) are strictly interior.


# %%
def get_task_table(quick=False):
    """
    Return the study task table: 6 resolution runs (`task_0`..`task_5`),
    1 independent reference (`task_6`), the initial 9-point 3x3 sweep
    (`task_7`..`task_15`), and in `--full` (`quick=False`) 21 appended sweep
    points (`task_16`..`task_36`) extending the sweep grid to 5x6
    (`W_MMI` in 4.80..5.20 um x `L_MMI` in 11.10..11.35 um).
    """
    cpws = (5, 7, 10) if quick else (10, 14, 20)
    levels = ("coarse", "medium", "fine")
    tasks = []

    for lvl, cpw in zip(levels, cpws):
        for avg in ("kottke", "staircase"):
            idx = len(tasks)
            is_primary = lvl == "fine" and avg == "kottke"
            tasks.append(
                {
                    "index": idx,
                    "name": f"{avg}_{lvl}",
                    "kind": "resolution",
                    "level": lvl,
                    "averaging": avg,
                    "cpw": cpw,
                    "l_mmi": L_MMI_OPT,
                    "w_mmi": W_MMI_OPT,
                    "n_pml": 8 if quick else 10,
                    "energy_decay": 5e-3 if quick else 1e-4,
                    "collect_fields": is_primary,
                    "record_movie": is_primary,
                    "movie_frames": 32 if quick else 48,
                    "include_mode1": is_primary,
                    "description": (
                        f"2D EIM 1x4 MMI ({avg}, {lvl} grid, cpw={cpw}, "
                        f"L={L_MMI_OPT * 1e6:.2f}um, W={W_MMI_OPT * 1e6:.2f}um"
                        + (", DFT + movie + flux + Mode 0/1)" if is_primary else ")")
                    ),
                }
            )

    ref_cpw = 20 if quick else 40
    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_reference",
            "kind": "reference",
            "level": "reference",
            "averaging": "kottke",
            "cpw": ref_cpw,
            "l_mmi": L_MMI_OPT,
            "w_mmi": W_MMI_OPT,
            "n_pml": 8 if quick else 10,
            "energy_decay": 5e-3 if quick else 1e-4,
            "collect_fields": False,
            "record_movie": False,
            "movie_frames": 0,
            "include_mode1": False,
            "description": (
                f"2D EIM 1x4 MMI (kottke, reference grid, cpw={ref_cpw}, "
                f"L={L_MMI_OPT * 1e6:.2f}um, W={W_MMI_OPT * 1e6:.2f}um)"
            ),
        }
    )

    sweep_cpw = 7 if quick else 20
    # Keep tasks 7..15 identical in name and index to the original 3x3 sweep
    for w_um in SWEEP_W_UM_INIT:
        for l_um in SWEEP_L_UM_INIT:
            idx = len(tasks)
            is_init = np.isclose(l_um, 11.20) and np.isclose(w_um, 5.00)
            tasks.append(
                {
                    "index": idx,
                    "name": f"sweep_W{w_um:.2f}_L{l_um:.2f}",
                    "kind": "sweep",
                    "level": "sweep",
                    "averaging": "kottke",
                    "cpw": sweep_cpw,
                    "l_mmi": l_um * 1e-6,
                    "w_mmi": w_um * 1e-6,
                    "n_pml": 8 if quick else 10,
                    "energy_decay": 5e-3 if quick else 1e-4,
                    "collect_fields": is_init,
                    "record_movie": False,
                    "movie_frames": 0,
                    "include_mode1": is_init,
                    "description": (
                        f"2D EIM 1x4 MMI 3x3 sweep (kottke, cpw={sweep_cpw}, "
                        f"L={l_um:.2f}um, W={w_um:.2f}um"
                        + (", initial design + DFT + Mode 0/1)" if is_init else ")")
                    ),
                }
            )

    if not quick:
        for w_um in SWEEP_W_UM:
            for l_um in SWEEP_L_UM:
                if any(np.isclose(w_um, w0) for w0 in SWEEP_W_UM_INIT) and any(
                    np.isclose(l_um, l0) for l0 in SWEEP_L_UM_INIT
                ):
                    continue
                idx = len(tasks)
                tasks.append(
                    {
                        "index": idx,
                        "name": f"sweep_W{w_um:.2f}_L{l_um:.2f}",
                        "kind": "sweep",
                        "level": "sweep_ext",
                        "averaging": "kottke",
                        "cpw": sweep_cpw,
                        "l_mmi": l_um * 1e-6,
                        "w_mmi": w_um * 1e-6,
                        "n_pml": 10,
                        "energy_decay": 1e-4,
                        "collect_fields": False,
                        "record_movie": False,
                        "movie_frames": 0,
                        "include_mode1": False,
                        "description": (
                            f"2D EIM 1x4 MMI extended 5x6 sweep (kottke, cpw={sweep_cpw}, "
                            f"L={l_um:.2f}um, W={w_um:.2f}um)"
                        ),
                    }
                )
    return tasks


def run_single_task(task_cfg, out_dir):
    """
    Execute one 2D EIM 1x4 MMI study task and write ``task_<I>.npz`` and
    ``task_<I>.json`` into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task_cfg["index"]
    t0 = time.perf_counter()

    structures, ports, domain = build_mmi_1x4(
        l_mmi=task_cfg["l_mmi"],
        w_mmi=task_cfg["w_mmi"],
        include_mode1_port=task_cfg.get("include_mode1", False),
    )
    use_gpu = os.environ.get("WAKIS_USE_GPU", "0") == "1"
    pic = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=N_FREQS,
        structures=structures,
        ports=ports,
        n_core=N_CORE_EQ,
        n_clad=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging=task_cfg["averaging"],
        cells_per_wavelength=task_cfg["cpw"],
        domain=domain,
        n_pml=task_cfg["n_pml"],
        cfln=0.85,
        field_wavelengths=(WL_REF,),
        record_movie=task_cfg["record_movie"],
        movie_frames=task_cfg["movie_frames"],
        use_gpu=use_gpu,
    )

    s_mat = pic.run_s_parameters(
        drive=["in"],
        energy_decay=task_cfg["energy_decay"],
        collect_fields=task_cfg["collect_fields"],
    )

    payload = {
        "task_index": np.int32(idx),
        "name": np.array(task_cfg["name"]),
        "kind": np.array(task_cfg["kind"]),
        "level": np.array(task_cfg["level"]),
        "averaging": np.array(task_cfg["averaging"]),
        "cpw": np.float32(task_cfg["cpw"]),
        "l_mmi": np.float64(task_cfg["l_mmi"]),
        "w_mmi": np.float64(task_cfg["w_mmi"]),
        "n_eff_slab": np.float64(pic.n_eff_slab),
        "frequencies": pic.frequencies.astype(np.float64),
        "wavelengths": pic.wavelengths.astype(np.float64),
        "S11": s_mat[("in", "in")].astype(np.complex128),
        "S21": s_mat[("out1", "in")].astype(np.complex128),
        "S31": s_mat[("out2", "in")].astype(np.complex128),
        "S41": s_mat[("out3", "in")].astype(np.complex128),
        "S51": s_mat[("out4", "in")].astype(np.complex128),
    }

    if ("out4_m1", "in") in s_mat:
        payload["S51_m1"] = s_mat[("out4_m1", "in")].astype(np.complex128)

    if task_cfg["collect_fields"]:
        modes_in = pic.solve_port_modes("in", frequencies=c_light / WL_REF, n_modes=1)
        modes_out = pic.solve_port_modes(
            "out3", frequencies=c_light / WL_REF, n_modes=1
        )
        y_in = modes_in[0].coords["Eb"][0]
        ey_in = modes_in[0].e["b"][:, 0]
        m_in = np.abs(y_in) <= 0.95e-6
        y_out = modes_out[0].coords["Eb"][0]
        ey_out = modes_out[0].e["b"][:, 0]
        m_out = np.abs(y_out - Y_OUT_CENTERS[2]) <= 0.60e-6
        payload["port_y_in"] = y_in[m_in].astype(np.float32)
        payload["port_ey_in"] = ey_in[m_in].astype(np.complex64)
        payload["port_neff_in"] = np.float64(np.real(modes_in[0].neff))
        payload["port_y_out"] = y_out[m_out].astype(np.float32)
        payload["port_ey_out"] = ey_out[m_out].astype(np.complex64)
        payload["port_neff_out"] = np.float64(np.real(modes_out[0].neff))

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
        "kind": task_cfg["kind"],
        "averaging": task_cfg["averaging"],
        "cpw": task_cfg["cpw"],
        "l_mmi_um": round(task_cfg["l_mmi"] * 1e6, 3),
        "w_mmi_um": round(task_cfg["w_mmi"] * 1e6, 3),
        "grid_shape": [int(pic.grid.Nx), int(pic.grid.Ny), int(pic.grid.Nz)],
        "n_cells": int(pic.grid.Nx * pic.grid.Ny * pic.grid.Nz),
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mb": round(rss_mb, 2),
        "npz_bytes": int(npz_path.stat().st_size),
    }
    (out_dir / f"task_{idx}.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[task_{idx} {task_cfg['name']}] grid={meta['grid_shape']} "
        f"time={elapsed:.2f}s rss={rss_mb:.1f}MB size={meta['npz_bytes'] / 1024:.1f}KB"
    )
    return meta


# %% [markdown]
# ## 6. Generating the complete 15-figure diagnostic suite
#
# Invoking `--plot` (or completing `--quick` / `--full`) reads `data/task_*.npz`
# and generates 15 figures in `figures/`:
#
# 1. `01_mmi_1x4_layout.png` — Annotated top-view layout of the 34-vertex $1 \times 4$ MMI splitter.
# 2. `02_slot_eim_vertical_stack.png` — 5-layer $\text{Si–GaN–Si}$ vertical slot stack, 1D TM slot mode profile $|E_z(z)|$, and $n_{\text{eff}}(\lambda)$.
# 3. `03_self_imaging_and_eme_length_sweep.png` — Analytical self-imaging beat length $L_\pi$, 1D EME interference carpet $I(x, y)$, and length sweep $T_{\text{tot}}(L_{\text{MMI}})$.
# 4. `04_permittivity_mesh_zoom.png` — Zoomed in-plane permittivity map with Yee mesh lines on the output tapers: staircase vs Kottke side by side.
# 5. `05_port_mode_profiles.png` — Input ($y = 0$) and inner output ($y = +0.65\;\mu\text{m}$) port eigenmode profiles $|E_y(y)|$ and $\mathrm{Re}(E_y(y))$.
# 6. `06_port_neff_dispersion.png` — Channel waveguide effective index $n_{\text{eff}}(\lambda)$ and group index $n_g(\lambda)$.
# 7. `07_field_intensity_1550.png` — Steady-state logarithmic intensity map $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $1550\;\text{nm}$ showing the 4-fold self-images.
# 8. `08_field_real_ey_1550.png` — Steady-state wave-front phase pattern $\mathrm{Re}(E_y)$ at $1550\;\text{nm}$.
# 9. `09_pulse_propagation.gif` — Time-domain pulse propagation movie through the $1 \times 4$ MMI splitter.
# 10. `10_sparameter_spectra_linear_and_db.png` — Broadband linear and dB S-parameter spectra ($T_1..T_4$, $T_{\text{tot}}$, $R$).
# 11. `11_uniformity_symmetry_and_energy_budget.png` — Output uniformity, excess insertion loss, pairwise mirror-symmetry audit ($|T_1 - T_4|$, $|T_2 - T_3|$), and 2D flux-box budget.
# 12. `12_outer_port_mode_power_share.png` — Fundamental Mode 0 vs odd Mode 1 power share (%) at the outer output waveguide for $(5.00, 11.20)\;\mu\text{m}$ ($98.0\%$ Mode 0 at $1550\;\text{nm}$) and $(4.90, 11.15)\;\mu\text{m}$ ($96.7\%$ Mode 0 at $1550\;\text{nm}$).
# 13. `13_parameter_sweep_3x3_heatmaps.png` — Parameter-sweep heatmaps of inner/outer imbalance $|T_{\text{inner}} - T_{\text{outer}}|$ and excess insertion loss over $(W_{\text{MMI}}, L_{\text{MMI}})$ (automatically expanding from the $3 \times 3$ grid to $5 \times 6$ as extended sweep data is added).
# 14. `14_kottke_vs_staircase_convergence.png` — 3-panel multi-resolution convergence of Kottke vs staircasing against the $40\;\text{cpw}$ reference: (a) single-wavelength $|T(1550\;\text{nm}) - T_{\text{ref}}|$ and band-averaged $\mathrm{RMS}_\lambda(|T(\lambda) - T_{\text{ref}}(\lambda)|)$, (b) band-averaged excess-loss error $|\langle \text{EL}\rangle - \langle \text{EL}_{\text{ref}}\rangle|$ and $\mathrm{RMS}_\lambda(\text{EL} - \text{EL}_{\text{ref}})$, and (c) multi-resolution spectra.
# 15. `15_port_time_trace.png` — Output port time-domain waveform $E_y(t)$ and exponential ring-down audit.


# %%
def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Read ``data_dir/task_*.npz`` and render all 15 figures into ``fig_dir``."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tasks_data = {}
    for p in sorted(data_dir.glob("task_*.npz")):
        d = dict(np.load(p, allow_pickle=False))
        tasks_data[str(d["name"])] = d

    if "kottke_fine" not in tasks_data:
        raise FileNotFoundError(
            f"Missing primary task 'kottke_fine' in {data_dir}. Run --quick or --full first."
        )
    primary = tasks_data["kottke_fine"]
    init_task = tasks_data.get("sweep_W5.00_L11.20", primary)
    structures, ports, domain = build_mmi_1x4(L_MMI_OPT, W_MMI_OPT)

    # 1. Top-view 34-vertex 1x4 MMI layout with dimensional annotations
    annotations = [
        {
            "text": "Input WG & taper\nW₁=0.4→W₂=0.8 µm\nL₁ = 2.0 µm",
            "xy": (1.0, 0.25),
            "xytext": (1.2, 2.25),
            "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
        },
        {
            "text": "Si–GaN–Si MMI box\nL_MMI = 11.15 µm\nW_MMI = 4.90 µm",
            "xy": (7.5, 0.0),
            "xytext": (7.5, -2.55),
            "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
        },
        {
            "text": "4 output tapers\nL₂ = 5.0 µm, pitch = 1.30 µm\n(y = ±0.65, ±1.95 µm)",
            "xy": (15.5, 1.30),
            "xytext": (14.2, 2.65),
            "arrowprops": dict(arrowstyle="->", color="#d95f02", lw=1.1),
        },
    ]
    plot_layout(
        structures,
        ports=ports,
        domain=domain,
        annotations=annotations,
        title="Malka et al. (2016) 1×4 Si–GaN–Si vertical-slot MMI splitter layout (34 vertices)",
        out_path=fig_dir / "01_mmi_1x4_layout.png",
    )

    # 2. 5-layer SiO2 / Si(300nm) / GaN(100nm) / Si(300nm) / SiO2 vertical slot EIM stack
    plot_slot_eim_stack(
        n_rail=N_SI,
        n_slot=N_GAN,
        n_clad=N_SIO2,
        h_rail=H_SI,
        h_slot=H_GAN,
        wavelength_ref=WL_REF,
        wavelengths=np.linspace(WL_MIN, WL_MAX, 51),
        out_path=fig_dir / "02_slot_eim_vertical_stack.png",
    )

    # 3. Self-imaging theory: EME interference carpet I(x, y) & length sweep vs Soldano-Pennings
    est = mmi_analytic_estimates(
        w_mmi=W_MMI_OPT, n_r=N_EFF_SLOT, n_clad=N_SIO2, wavelength=WL_REF, n_splits=4
    )
    eme = mmi_eme_length_sweep(
        w_mmi=W_MMI_OPT,
        w_taper=W2,
        pitch=PITCH,
        n_r=N_EFF_SLOT,
        n_clad=N_SIO2,
        wavelength=WL_REF,
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.1), dpi=150)
    im = ax0.pcolormesh(
        eme["carpet_x"] * 1e6,
        eme["carpet_y"] * 1e6,
        eme["carpet_I"].T,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )
    ax0.axhline(-0.5 * W_MMI_OPT * 1e6, color="white", ls="--", lw=1.0, alpha=0.8)
    ax0.axhline(+0.5 * W_MMI_OPT * 1e6, color="white", ls="--", lw=1.0, alpha=0.8)
    ax0.axvline(
        L_MMI_OPT * 1e6,
        color="#1b9e77",
        ls="-",
        lw=1.6,
        label=f"Design L_MMI = {L_MMI_OPT * 1e6:.2f} µm",
    )
    for y_c in Y_OUT_CENTERS * 1e6:
        ax0.plot(L_MMI_OPT * 1e6, y_c, "o", color="#1b9e77", ms=5, mec="white")
    cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)
    cb.set_label("Normalized EME intensity $I(x, y)$")
    ax0.set(
        xlabel="Axial distance along MMI box x - L₁ [µm]",
        ylabel="Transverse coordinate y [µm]",
        ylim=(-2.8, 2.8),
        title="(a) Multi-mode self-imaging carpet (1→2→4 images)",
    )
    ax0.legend(loc="upper left", fontsize=8.0)

    l_um_arr = eme["lengths"] * 1e6
    ax1.plot(
        l_um_arr,
        eme["t_tot"],
        color="#2b5c8f",
        lw=2.1,
        label=r"EME total $T_{\mathrm{tot}} = 2(T_{\mathrm{in}} + T_{\mathrm{out}})$",
    )
    ax1.plot(
        l_um_arr,
        2.0 * eme["t_inner"],
        "--",
        color="#1b9e77",
        lw=1.6,
        label=r"EME inner pair $2 T_{\mathrm{inner}}$",
    )
    ax1.plot(
        l_um_arr,
        2.0 * eme["t_outer"],
        "-.",
        color="#d95f02",
        lw=1.6,
        label=r"EME outer pair $2 T_{\mathrm{outer}}$",
    )
    ax1.axvline(
        est["l_mmi_tm"] * 1e6,
        color="#7570b3",
        ls=":",
        lw=1.6,
        label=f"Analytic 3L_π/16 (2D TM: {est['l_mmi_tm'] * 1e6:.2f} µm)",
    )
    ax1.axvline(
        L_MMI_OPT * 1e6,
        color="0.25",
        ls="--",
        lw=1.4,
        label=f"Optimal L_MMI = {L_MMI_OPT * 1e6:.2f} µm (EME peak {eme['l_opt_eme'] * 1e6:.2f} µm)",
    )
    ax1.set(
        xlabel="MMI box length L_MMI [µm]",
        ylabel="Guided power fraction",
        ylim=(0.0, 1.05),
        title=f"(b) Self-imaging length sweep (L_π = {est['l_pi_tm'] * 1e6:.1f} µm)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=7.6)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "03_self_imaging_and_eme_length_sweep.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4. In-plane permittivity map with mesh lines zoomed on the MMI-to-output-taper junction
    cpw_mesh = float(tasks_data.get("kottke_medium", primary)["cpw"])
    pic_stair = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures,
        ports=ports,
        n_core=N_CORE_EQ,
        n_clad=N_SIO2,
        two_d=True,
        averaging="staircase",
        cells_per_wavelength=cpw_mesh,
        domain=domain,
        n_pml=8,
    )
    pic_kottke = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures,
        ports=ports,
        n_core=N_CORE_EQ,
        n_clad=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw_mesh,
        domain=domain,
        n_pml=8,
    )
    plot_cross_section_mesh(
        pic_stair.solver,
        pic_kottke.solver,
        xlim_um=(12.6, 15.2),
        ylim_um=(-1.35, 1.35),
        component="y",
        out_path=fig_dir / "04_permittivity_mesh_zoom.png",
    )

    # 5. Port mode profiles at input (y = 0) and inner output arm (y = +0.65 um)
    plot_port_modes(
        primary["port_y_in"] * 1e6,
        primary["port_ey_in"],
        float(primary["port_neff_in"]),
        primary["port_y_out"] * 1e6,
        primary["port_ey_out"],
        float(primary["port_neff_out"]),
        structures_outline_y=[(-0.20, 0.20)],
        out_path=fig_dir / "05_port_mode_profiles.png",
    )

    # 6. Waveguide mode effective index and group index dispersion across C-band
    plot_neff_dispersion(
        primary["disp_wavelengths"] * 1e9,
        primary["disp_neff"],
        neff_slab=float(primary["n_eff_slab"]),
        ng_2d=primary["disp_ng"],
        out_path=fig_dir / "06_port_neff_dispersion.png",
    )

    # 7. Steady-state logarithmic intensity map |E|^2 [dB] at 1550 nm
    plot_field_map(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_E2"],
        structures=structures,
        kind="intensity_log",
        title=r"Steady-state modal intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $\lambda = 1550\;\mathrm{nm}$ (1×4 self-imaging)",
        out_path=fig_dir / "07_field_intensity_1550.png",
    )

    # 8. Steady-state real field map Re(Ey) at 1550 nm
    plot_field_map(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_Ey"],
        structures=structures,
        kind="real",
        title=r"Steady-state wave-front phase pattern $\mathrm{Re}(E_y)$ at $\lambda = 1550\;\mathrm{nm}$",
        out_path=fig_dir / "08_field_real_ey_1550.png",
    )

    # 9. Animated pulse propagation GIF
    if "movie_frames" in primary and len(primary["movie_frames"]) > 0:
        make_movie(
            primary["movie_x"],
            primary["movie_y"],
            primary["movie_frames"],
            primary["movie_times"],
            structures=structures,
            out_path=fig_dir / "09_pulse_propagation.gif",
            fps=12,
        )

    # Sort wavelengths ascending for spectral plots (interior of pulse band: 1505-1595 nm)
    wl_nm_all = primary["wavelengths"] * 1e9
    order = np.argsort(wl_nm_all)
    wl_nm = wl_nm_all[order]
    mask = (wl_nm >= 1505.0) & (wl_nm <= 1595.0)
    wl_b = wl_nm[mask]

    s11 = primary["S11"][order][mask]
    s21 = primary["S21"][order][mask]  # out1 (outer bottom, y = -1.95 um)
    s31 = primary["S31"][order][mask]  # out2 (inner bottom, y = -0.65 um)
    s41 = primary["S41"][order][mask]  # out3 (inner top, y = +0.65 um)
    s51 = primary["S51"][order][mask]  # out4 (outer top, y = +1.95 um)
    t1, t2, t3, t4 = (
        np.abs(s21) ** 2,
        np.abs(s31) ** 2,
        np.abs(s41) ** 2,
        np.abs(s51) ** 2,
    )
    t_tot = t1 + t2 + t3 + t4
    r = np.abs(s11) ** 2

    # 10. Broadband S-parameter spectra: (a) linear transmissions, (b) dB spectra
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.1), dpi=150)
    ax0.plot(
        wl_b, t2, "-", color="#1b9e77", lw=2.1, label=r"$T_2 = |S_{31}|^2$ (inner −)"
    )
    ax0.plot(
        wl_b, t3, "--", color="#2b5c8f", lw=1.9, label=r"$T_3 = |S_{41}|^2$ (inner +)"
    )
    ax0.plot(
        wl_b, t1, "-.", color="#d95f02", lw=2.1, label=r"$T_1 = |S_{21}|^2$ (outer −)"
    )
    ax0.plot(
        wl_b, t4, ":", color="#7570b3", lw=2.1, label=r"$T_4 = |S_{51}|^2$ (outer +)"
    )
    ax0.axhline(
        0.25, color="0.35", ls=":", lw=1.3, label="Ideal lossless 1×4 split (0.25)"
    )
    if init_task is not primary:
        w_i = init_task["wavelengths"] * 1e9
        ord_i = np.argsort(w_i)
        m_i = (w_i[ord_i] >= 1505.0) & (w_i[ord_i] <= 1595.0)
        ti_in = np.abs(init_task["S31"][ord_i][m_i]) ** 2
        ti_out = np.abs(init_task["S21"][ord_i][m_i]) ** 2
        ax0.plot(
            w_i[ord_i][m_i],
            ti_in,
            "-",
            color="#1b9e77",
            alpha=0.35,
            lw=1.4,
            label="Initial (11.20, 5.00) inner",
        )
        ax0.plot(
            w_i[ord_i][m_i],
            ti_out,
            "-.",
            color="#d95f02",
            alpha=0.35,
            lw=1.4,
            label="Initial (11.20, 5.00) outer",
        )
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Per-port modal power transmission T_i",
        ylim=(0.08, 0.28),
        title="(a) Linear per-port transmission spectra (T₁..T₄)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower center", ncol=2, fontsize=7.5)

    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(t_tot, 1e-12)),
        "-",
        color="#1b9e77",
        lw=2.1,
        label=r"Total $10\log_{10}(\sum T_i)$",
    )
    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(t2, 1e-12)),
        "--",
        color="#2b5c8f",
        lw=1.8,
        label=r"Inner arms $T_2, T_3$ [dB]",
    )
    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(t1, 1e-12)),
        "-.",
        color="#d95f02",
        lw=1.8,
        label=r"Outer arms $T_1, T_4$ [dB]",
    )
    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(r, 1e-12)),
        ":",
        color="#7570b3",
        lw=1.8,
        label=r"Return loss $R = |S_{11}|^2$ [dB]",
    )
    ax1.axhline(
        -6.0206, color="0.35", ls=":", lw=1.2, label="Ideal 25% split (−6.02 dB)"
    )
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Modal power [dB]",
        title="(b) Broadband S-parameter spectra in dB",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="center right", fontsize=7.8)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "10_sparameter_spectra_linear_and_db.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 11. Output uniformity, excess loss vs Malka (2016), pairwise symmetry & flux budget
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.1), dpi=150)
    el_db = -10.0 * np.log10(np.maximum(t_tot, 1e-12))
    t_stack = np.vstack([t1, t2, t3, t4])
    unif_db = 10.0 * np.log10(
        np.maximum(np.max(t_stack, axis=0), 1e-12)
        / np.maximum(np.min(t_stack, axis=0), 1e-12)
    )
    ax0.plot(
        wl_b,
        el_db,
        color="#1b9e77",
        lw=2.1,
        label=r"Excess loss $-10\log_{10}(\sum T_i)$ (11.15, 4.90)",
    )
    ax0.plot(
        wl_b,
        unif_db,
        color="#2b5c8f",
        ls="--",
        lw=1.9,
        label=r"Uniformity $10\log_{10}(\max T_i / \min T_i)$",
    )
    if init_task is not primary:
        w_i = init_task["wavelengths"] * 1e9
        ord_i = np.argsort(w_i)
        m_i = (w_i[ord_i] >= 1505.0) & (w_i[ord_i] <= 1595.0)
        t_tot_init = (
            np.abs(init_task["S21"][ord_i][m_i]) ** 2
            + np.abs(init_task["S31"][ord_i][m_i]) ** 2
            + np.abs(init_task["S41"][ord_i][m_i]) ** 2
            + np.abs(init_task["S51"][ord_i][m_i]) ** 2
        )
        ax0.plot(
            w_i[ord_i][m_i],
            -10.0 * np.log10(np.maximum(t_tot_init, 1e-12)),
            color="#d95f02",
            ls="-.",
            lw=1.8,
            label="Excess loss (11.20, 5.00 min-EL)",
        )
    if "sweep_W5.00_L11.30" in tasks_data:
        opt_imb_task = tasks_data["sweep_W5.00_L11.30"]
        w_o = opt_imb_task["wavelengths"] * 1e9
        ord_o = np.argsort(w_o)
        m_o = (w_o[ord_o] >= 1505.0) & (w_o[ord_o] <= 1595.0)
        t_tot_opt_imb = (
            np.abs(opt_imb_task["S21"][ord_o][m_o]) ** 2
            + np.abs(opt_imb_task["S31"][ord_o][m_o]) ** 2
            + np.abs(opt_imb_task["S41"][ord_o][m_o]) ** 2
            + np.abs(opt_imb_task["S51"][ord_o][m_o]) ** 2
        )
        ax0.plot(
            w_o[ord_o][m_o],
            -10.0 * np.log10(np.maximum(t_tot_opt_imb, 1e-12)),
            color="#7570b3",
            ls=":",
            lw=1.7,
            label="Excess loss (11.30, 5.00 min-imb)",
        )
    if "flux_total_box" in primary:
        f_box = primary["flux_total_box"][order][mask]
        f_lat = primary["flux_lateral"][order][mask]
        ax0.plot(
            wl_b,
            -10.0 * np.log10(np.maximum(f_box, 1e-12)),
            color="0.4",
            ls=":",
            lw=1.4,
            label=f"Flux-box residual (mean box={np.mean(f_box):.3f}, lat={np.mean(f_lat) * 100:.1f}%)",
        )
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Loss / Imbalance [dB]",
        title="(a) Excess insertion loss, uniformity & flux budget",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=7.2)

    sym_outer = np.maximum(np.abs(t1 - t4), 1e-16)
    sym_inner = np.maximum(np.abs(t2 - t3), 1e-16)
    ax1.semilogy(
        wl_b,
        sym_outer,
        color="#d95f02",
        lw=2.0,
        label=r"Outer-pair asymmetry $|T_1 - T_4|$",
    )
    ax1.semilogy(
        wl_b,
        sym_inner,
        color="#2b5c8f",
        ls="--",
        lw=1.8,
        label=r"Inner-pair asymmetry $|T_2 - T_3|$",
    )
    ax1.semilogy(
        wl_b,
        r,
        color="#7570b3",
        ls="-.",
        lw=1.8,
        label=r"Return loss $R = |S_{11}|^2$",
    )
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Dimensionless magnitude (log scale)",
        title="(b) Pairwise mirror-symmetry audit & return loss",
    )
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(loc="best", fontsize=8.0)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "11_uniformity_symmetry_and_energy_budget.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 12. Mode 0 vs Mode 1 power share (%) at the outer output waveguide
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=150, sharey=True)
    for ax, d_task, ttl in (
        (ax0, init_task, "(a) W=5.00 µm, L=11.20 µm (initial / 2D min-EL)"),
        (ax1, primary, "(b) W=4.90 µm, L=11.15 µm (primary task)"),
    ):
        w_t = d_task["wavelengths"] * 1e9
        ord_t = np.argsort(w_t)
        m_t = (w_t[ord_t] >= 1505.0) & (w_t[ord_t] <= 1595.0)
        wl_t = w_t[ord_t][m_t]
        p_m0 = np.abs(d_task["S51"][ord_t][m_t]) ** 2
        p_m1 = (
            np.abs(d_task["S51_m1"][ord_t][m_t]) ** 2
            if "S51_m1" in d_task
            else np.zeros_like(p_m0)
        )
        p_sum = np.maximum(p_m0 + p_m1, 1e-12)
        sh_m0 = 100.0 * p_m0 / p_sum
        sh_m1 = 100.0 * p_m1 / p_sum
        i_1550 = int(np.argmin(np.abs(wl_t - 1550.0)))
        ax.plot(
            wl_t,
            sh_m0,
            color="#1b9e77",
            lw=2.2,
            label=f"Fundamental Mode 0 ({sh_m0[i_1550]:.1f}% @ 1550 nm)",
        )
        ax.plot(
            wl_t,
            sh_m1,
            color="#d95f02",
            ls="--",
            lw=2.0,
            label=f"First odd Mode 1 ({sh_m1[i_1550]:.1f}% @ 1550 nm)",
        )
        ax.axhline(98.0, color="0.4", ls=":", lw=1.2, label="98% Mode 0 reference")
        ax.set(
            xlabel="Wavelength λ [nm]",
            ylim=(-2.0, 103.0),
            title=ttl,
        )
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="center right", fontsize=8.0)
    ax0.set_ylabel("Outer port modal power share [%]")
    fig.tight_layout()
    fig.savefig(
        fig_dir / "12_outer_port_mode_power_share.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # 13. Parameter sweep heatmaps over available points in SWEEP_W_UM x SWEEP_L_UM
    imb_full = np.full((len(SWEEP_W_UM), len(SWEEP_L_UM)), np.nan)
    el_full = np.full((len(SWEEP_W_UM), len(SWEEP_L_UM)), np.nan)
    for iw, w_um in enumerate(SWEEP_W_UM):
        for il, l_um in enumerate(SWEEP_L_UM):
            key = f"sweep_W{w_um:.2f}_L{l_um:.2f}"
            d_sw = tasks_data.get(key)
            if d_sw is None and np.isclose(w_um, 4.90) and np.isclose(l_um, 11.15):
                d_sw = primary
            if d_sw is not None:
                i0 = int(np.argmin(np.abs(d_sw["wavelengths"] - WL_REF)))
                t_out = float(np.abs(d_sw["S21"][i0]) ** 2)
                t_in = float(np.abs(d_sw["S31"][i0]) ** 2)
                imb_full[iw, il] = abs(t_in - t_out)
                el_full[iw, il] = -10.0 * np.log10(
                    max(2.0 * (t_in + t_out), 1e-12)
                )

    active_w = [
        iw for iw in range(len(SWEEP_W_UM)) if np.any(np.isfinite(imb_full[iw, :]))
    ] or list(range(len(SWEEP_W_UM_INIT)))
    active_l = [
        il for il in range(len(SWEEP_L_UM)) if np.any(np.isfinite(imb_full[:, il]))
    ] or list(range(len(SWEEP_L_UM_INIT)))
    w_vals = [SWEEP_W_UM[iw] for iw in active_w]
    l_vals = [SWEEP_L_UM[il] for il in active_l]
    imb_grid = imb_full[np.ix_(active_w, active_l)]
    el_grid = el_full[np.ix_(active_w, active_l)]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=150)
    for ax, mat, cmap, ttl, cbl in (
        (
            ax0,
            imb_grid,
            "YlOrRd",
            r"(a) Output power imbalance $|T_{\mathrm{inner}} - T_{\mathrm{outer}}|$ @ 1550 nm",
            r"$|T_{\mathrm{inner}} - T_{\mathrm{outer}}|$",
        ),
        (
            ax1,
            el_grid,
            "viridis_r",
            r"(b) Excess insertion loss $-10\log_{10}(\sum T_i)$ [dB] @ 1550 nm",
            "Excess loss [dB]",
        ),
    ):
        im = ax.imshow(mat, origin="lower", cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(l_vals)), [f"{v:.2f}" for v in l_vals])
        ax.set_yticks(range(len(w_vals)), [f"{v:.2f}" for v in w_vals])
        ax.set(
            xlabel="MMI length L_MMI [µm]",
            ylabel="MMI width W_MMI [µm]",
            title=ttl,
        )
        mid_val = 0.5 * (float(np.nanmin(mat)) + float(np.nanmax(mat)))
        for iw in range(len(w_vals)):
            for il in range(len(l_vals)):
                val = mat[iw, il]
                if np.isfinite(val):
                    ax.text(
                        il,
                        iw,
                        f"{val:.4f}" if ax is ax0 else f"{val:.2f} dB",
                        ha="center",
                        va="center",
                        color="black" if val < mid_val else "white",
                        fontsize=7.6 if len(l_vals) > 3 else 8.5,
                        fontweight="bold",
                    )
        if np.any(np.isfinite(mat)):
            iw_min, il_min = np.unravel_index(int(np.nanargmin(mat)), mat.shape)
            rect = plt.Rectangle(
                (il_min - 0.48, iw_min - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor="#0055aa" if ax is ax0 else "#d95f02",
                lw=2.4,
            )
            ax.add_patch(rect)
        if len(w_vals) > 3 and len(l_vals) > 3:
            ax.axhline(2.5, color="0.3", ls="--", lw=1.1, alpha=0.8)
            ax.axvline(2.5, color="0.3", ls="--", lw=1.1, alpha=0.8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbl)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "13_parameter_sweep_3x3_heatmaps.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # 14. Multi-resolution convergence (single-wavelength, band RMS, excess-loss error & spectra)
    levels = ["coarse", "medium", "fine"]
    avail = all(
        f"{avg}_{lvl}" in tasks_data
        for avg in ("kottke", "staircase")
        for lvl in levels
    )
    if avail:
        if "kottke_reference" not in tasks_data:
            raise FileNotFoundError(
                "Missing independent 'kottke_reference' task for convergence plot."
            )
        d_ref = tasks_data["kottke_reference"]
        ref_cpw_val = int(d_ref["cpw"])
        w_r_nm = d_ref["wavelengths"] * 1e9
        ord_r = np.argsort(w_r_nm)
        wl_r_s = w_r_nm[ord_r]
        m_band = (wl_r_s >= 1505.0) & (wl_r_s <= 1595.0)
        i_1550 = int(np.argmin(np.abs(wl_r_s - 1550.0)))

        def _t_tot_sorted(d):
            w_d = d["wavelengths"] * 1e9
            idx_d = np.argsort(w_d)
            return (
                np.abs(d["S21"][idx_d]) ** 2
                + np.abs(d["S31"][idx_d]) ** 2
                + np.abs(d["S41"][idx_d]) ** 2
                + np.abs(d["S51"][idx_d]) ** 2
            )

        t_ref_s = _t_tot_sorted(d_ref)
        t_ref_1550 = float(t_ref_s[i_1550])
        t_ref_b = t_ref_s[m_band]
        el_ref_b = -10.0 * np.log10(np.maximum(t_ref_b, 1e-12))
        mean_el_ref = float(np.mean(el_ref_b))

        cpws = [float(tasks_data[f"kottke_{lvl}"]["cpw"]) for lvl in levels]
        res = np.asarray(cpws, dtype=float)

        err_k_1550, err_s_1550 = [], []
        rms_k_band, rms_s_band = [], []
        el_err_k, el_err_s = [], []
        el_rms_k, el_rms_s = [], []

        for lvl in levels:
            tk_s = _t_tot_sorted(tasks_data[f"kottke_{lvl}"])
            ts_s = _t_tot_sorted(tasks_data[f"staircase_{lvl}"])
            err_k_1550.append(max(abs(float(tk_s[i_1550]) - t_ref_1550), 1e-5))
            err_s_1550.append(max(abs(float(ts_s[i_1550]) - t_ref_1550), 1e-5))
            rms_k_band.append(
                max(float(np.sqrt(np.mean((tk_s[m_band] - t_ref_b) ** 2))), 1e-5)
            )
            rms_s_band.append(
                max(float(np.sqrt(np.mean((ts_s[m_band] - t_ref_b) ** 2))), 1e-5)
            )
            elk_b = -10.0 * np.log10(np.maximum(tk_s[m_band], 1e-12))
            els_b = -10.0 * np.log10(np.maximum(ts_s[m_band], 1e-12))
            el_err_k.append(max(abs(float(np.mean(elk_b)) - mean_el_ref), 1e-4))
            el_err_s.append(max(abs(float(np.mean(els_b)) - mean_el_ref), 1e-4))
            el_rms_k.append(
                max(float(np.sqrt(np.mean((elk_b - el_ref_b) ** 2))), 1e-4)
            )
            el_rms_s.append(
                max(float(np.sqrt(np.mean((els_b - el_ref_b) ** 2))), 1e-4)
            )

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14.4, 4.2), dpi=150)
        ax0.loglog(
            res,
            err_s_1550,
            "s-",
            color="#d95f02",
            lw=2.0,
            ms=6,
            label="Staircase $|T(1550) - T_{\\mathrm{ref}}|$",
        )
        ax0.loglog(
            res,
            err_k_1550,
            "o-",
            color="#1b9e77",
            lw=2.0,
            ms=6,
            label="Kottke $|T(1550) - T_{\\mathrm{ref}}|$",
        )
        ax0.loglog(
            res,
            rms_s_band,
            "s--",
            color="#d95f02",
            lw=1.7,
            ms=5,
            mfc="white",
            label="Staircase band $\\mathrm{RMS}(|T - T_{\\mathrm{ref}}|)$",
        )
        ax0.loglog(
            res,
            rms_k_band,
            "o--",
            color="#1b9e77",
            lw=1.7,
            ms=5,
            mfc="white",
            label="Kottke band $\\mathrm{RMS}(|T - T_{\\mathrm{ref}}|)$",
        )
        ax0.loglog(
            res,
            err_s_1550[0] * (res[0] / res) ** 1,
            ":",
            color="#d95f02",
            alpha=0.55,
            lw=1.2,
            label=r"$\mathcal{O}(\Delta x^1)$ guide",
        )
        ax0.loglog(
            res,
            err_k_1550[0] * (res[0] / res) ** 2,
            ":",
            color="#1b9e77",
            alpha=0.65,
            lw=1.4,
            label=r"$\mathcal{O}(\Delta x^2)$ guide",
        )
        ax0.set(
            xlabel="Resolution [cells per wavelength]",
            ylabel=r"Transmission error vs 40 cpw ref",
            title=f"(a) 1550 nm & band-RMS error (ref={ref_cpw_val} cpw)",
        )
        ax0.grid(True, which="both", ls=":", alpha=0.4)
        ax0.legend(loc="best", fontsize=7.0)

        ax1.loglog(
            res,
            el_err_s,
            "s-",
            color="#d95f02",
            lw=2.0,
            ms=6,
            label=r"Staircase $|\langle \mathrm{EL}\rangle - \langle \mathrm{EL}_{\mathrm{ref}}\rangle|$",
        )
        ax1.loglog(
            res,
            el_err_k,
            "o-",
            color="#1b9e77",
            lw=2.0,
            ms=6,
            label=r"Kottke $|\langle \mathrm{EL}\rangle - \langle \mathrm{EL}_{\mathrm{ref}}\rangle|$",
        )
        ax1.loglog(
            res,
            el_rms_s,
            "s--",
            color="#d95f02",
            lw=1.7,
            ms=5,
            mfc="white",
            label=r"Staircase band $\mathrm{RMS}(\mathrm{EL} - \mathrm{EL}_{\mathrm{ref}})$",
        )
        ax1.loglog(
            res,
            el_rms_k,
            "o--",
            color="#1b9e77",
            lw=1.7,
            ms=5,
            mfc="white",
            label=r"Kottke band $\mathrm{RMS}(\mathrm{EL} - \mathrm{EL}_{\mathrm{ref}})$",
        )
        ax1.set(
            xlabel="Resolution [cells per wavelength]",
            ylabel="Excess-loss error [dB]",
            title="(b) Band-averaged excess-loss convergence",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="best", fontsize=7.2)

        for ax in (ax0, ax1):
            ax.set_xticks(res)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        cols = {"coarse": "#7570b3", "medium": "#d95f02", "fine": "#1b9e77"}
        for lvl in levels:
            for avg, ls in (("kottke", "-"), ("staircase", "--")):
                d = tasks_data[f"{avg}_{lvl}"]
                t_db = 10.0 * np.log10(np.maximum(_t_tot_sorted(d)[m_band], 1e-12))
                lbl = f"{avg.capitalize()} {lvl} ({int(d['cpw'])} cpw)"
                ax2.plot(wl_r_s[m_band], t_db, color=cols[lvl], ls=ls, lw=1.6, label=lbl)
        t_ref_db = 10.0 * np.log10(np.maximum(t_ref_b, 1e-12))
        ax2.plot(
            wl_r_s[m_band],
            t_ref_db,
            color="black",
            ls=":",
            lw=2.0,
            label=f"Kottke ref ({ref_cpw_val} cpw)",
        )
        ax2.set(
            xlabel="Wavelength λ [nm]",
            ylabel=r"Total guided transmission $10\log_{10}(\sum T_i)$ [dB]",
            title="(c) Broadband transmission across resolutions",
        )
        ax2.grid(True, ls=":", alpha=0.4)
        ax2.legend(loc="best", fontsize=7.0)
        fig.tight_layout()
        fig.savefig(
            fig_dir / "14_kottke_vs_staircase_convergence.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 15. Port time trace and energy decay
    if "trace_t" in primary and primary["trace_t"].size > 0:
        t_fs = primary["trace_t"] * 1e15
        ey_tr = primary["trace_Ey"]
        ey_norm = ey_tr / max(float(np.max(np.abs(ey_tr))), 1e-30)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.6, 3.8), dpi=150)
        ax0.plot(t_fs, ey_norm, color="#2b5c8f", lw=1.4)
        ax0.set(
            xlabel="Time t [fs]",
            ylabel=r"Normalized $E_y(t) / \max|E_y|$",
            title="(a) Output arm port time-domain waveform",
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
            title="(b) Ring-down & termination audit",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(fig_dir / "15_port_time_trace.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Rendered 15 figures in {fig_dir}")


# %% [markdown]
# ## 7. Full-resolution results: spectra, symmetry, mode purity & parameter sweep
#
# ### 7.1 Vertical-slot EIM & channel waveguide dispersion (`task_4`, Figs. 2, 5, 6)
#
# At $\lambda_0 = 1550\;\text{nm}$, the 5-layer transfer-matrix solver gives a
# fundamental vertical-slot TM effective index of $n_{\text{eff,slot}} = 2.9842$
# (compared to $n_{\text{eff,TE}} = 3.1877$ for the rail-confined TE mode),
# with $\Gamma_{\text{slot}} = 38.3\%$ of $\int |E_z|^2\,\mathrm{d}z$ confined
# inside the $100\;\text{nm}$ $\text{GaN}$ slot and $\Gamma_{\text{stack}} = 93.7\%$
# inside the $700\;\text{nm}$ $\text{Si–GaN–Si}$ sandwich (Fig. 2b). Across
# $1500\text{--}1600\;\text{nm}$, $n_{\text{eff,slot}}(\lambda)$ decreases
# monotonically from $3.0120$ to $2.9562$ ($\Delta n_{\text{eff,slot}} = -0.0558$,
# Fig. 2c). Mapping $n_{\text{eff,slot}}(1550\;\text{nm})$ onto the
# $H_{\text{stack}} = 0.70\;\mu\text{m}$ `PolySlab` yields an equivalent
# 3-layer core index $n_{\text{core,eq}} = 3.1108$.
#
# In the 2D FIT domain at $20\;\text{cpw}$ (`task_4`, `kottke_fine`), the
# $W_1 = 0.40\;\mu\text{m}$ input channel waveguide supports a single guided
# lateral mode with $n_{\text{eff,in}}(1550\;\text{nm}) = 2.4660$
# ($n_{\text{eff,out}} = 2.4616$ on the $1.20\;\mu\text{m}$ output port window,
# Fig. 5). Across $1500\text{--}1600\;\text{nm}$, the channel effective index
# varies from $2.4966$ to $2.4350$ and the group index increases from
# $n_g = 3.4104$ to $3.4328$, with $n_g(1550\;\text{nm}) = 3.4221$ (Fig. 6).
#
# ### 7.2 Broadband S-parameters, mirror symmetry, flux conservation, mode purity & comparison with theory (`task_4`, `task_6`, `task_15`, `task_23`, Figs. 7–12)
#
# Full-resolution results across the $1505\text{--}1595\;\text{nm}$ interior
# telecom band compare the baseline geometry
# $(W_{\text{MMI}}, L_{\text{MMI}}) = (4.90, 11.15)\;\mu\text{m}$ (`task_4` at
# $20\;\text{cpw}$ and `task_6` at $40\;\text{cpw}$), the initial geometry
# $(5.00, 11.20)\;\mu\text{m}$ (`task_15` at $20\;\text{cpw}$, which is the
# minimum-excess-loss point of the 2D sweep), the 2D minimum-imbalance interior
# optimum $(5.00, 11.30)\;\mu\text{m}$ (`task_23` at $20\;\text{cpw}$), the
# earlier `--quick` runs, and the analytical Soldano & Pennings [2] / 1D EME
# references in a single table:
#
# | Metric | Old (`--quick`, $10\;\text{cpw}$ `task_4` / $7\;\text{cpw}$ `task_15`) | New `--full` $(4.90, 11.15)\;\mu\text{m}$ (`task_4`, $20\;\text{cpw}$ / `task_6`, $40\;\text{cpw}$) | New `--full` $(5.00, 11.20)\;\mu\text{m}$ (`task_15`, $20\;\text{cpw}$ min-EL) | New `--full` $(5.00, 11.30)\;\mu\text{m}$ (`task_23`, $20\;\text{cpw}$ min-imb) | Analytic / 1D EME Reference (Soldano & Pennings [2]) |
# |---|---|---|---|---|---|
# | 5-layer vertical-slot $n_{\text{eff,slot}}(1550\;\text{nm})$ | $2.9842$ ($\Gamma_{\text{slot}} = 38.3\%$) | $2.9842$ ($n_{\text{core,eq}} = 3.1108$) | $2.9842$ | $2.9842$ | Exact 5-layer TM transfer matrix: $2.9842$ |
# | Analytic $L_\pi = \frac{4 n_r W_e^2}{3\lambda_0}$ / $L_{\text{MMI}} = \frac{3 L_\pi}{16}$ | $62.75 / 11.77\;\mu\text{m}$ ($W=4.90$) | $62.75 / 11.77\;\mu\text{m}$ (2D TM); $66.48 / 12.46\;\mu\text{m}$ (3D TE) | $65.32 / 12.25\;\mu\text{m}$ (2D TM) | $65.32 / 12.25\;\mu\text{m}$ (2D TM) | Soldano & Pennings [2]; 1D EME $L_{\text{opt}} = 11.09\;\mu\text{m}$ ($4.90$), $11.51\;\mu\text{m}$ ($5.00$) |
# | Inner port $T_2 = T_3$ ($1550\;\text{nm}$) | $0.2334$ / $0.2253$ | $0.2471$ ($-6.07\;\text{dB}$) / $0.2468$ ($-6.08\;\text{dB}$) | $0.2282$ ($-6.42\;\text{dB}$) | $0.2311$ ($-6.36\;\text{dB}$) | Ideal lossless $1\times 4$: $0.2500$ ($-6.02\;\text{dB}$) |
# | Outer port $T_1 = T_4$ ($1550\;\text{nm}$) | $0.1918$ / $0.2014$ | $0.2019$ ($-6.95\;\text{dB}$) / $0.2033$ ($-6.92\;\text{dB}$) | $0.2357$ ($-6.28\;\text{dB}$) | $0.2284$ ($-6.41\;\text{dB}$) | Ideal lossless $1\times 4$: $0.2500$ ($-6.02\;\text{dB}$) |
# | Total guided $T_{\text{tot}}(1550\;\text{nm})$ | $0.8503$ / $0.8533$ | $0.8980$ / $0.9002$ | $0.9279$ | $0.9191$ | 1D EME: $0.965$ ($4.90$), $0.978$ ($5.00$) |
# | Excess insertion loss $\text{EL}(1550\;\text{nm})$ | $0.704\;\text{dB}$ / $0.689\;\text{dB}$ | $0.467\;\text{dB}$ / $0.456\;\text{dB}$ | $0.325\;\text{dB}$ | $0.366\;\text{dB}$ | 1D EME: $0.155\;\text{dB}$ ($4.90$), $0.097\;\text{dB}$ ($5.00$) |
# | Imbalance $|T_{\text{in}} - T_{\text{out}}|$ ($1550\;\text{nm}$) | $0.0416$ / $0.0239$ | $0.0451$ / $0.0435$ | $0.0075$ | $0.0027$ | Ideal symmetric self-image: $0$ |
# | Output uniformity $10\log_{10}(\max T_i / \min T_i)$ ($1550\;\text{nm}$) | $0.853\;\text{dB}$ / $0.487\;\text{dB}$ | $0.876\;\text{dB}$ / $0.841\;\text{dB}$ | $0.141\;\text{dB}$ | $0.050\;\text{dB}$ | Ideal symmetric self-image: $0.00\;\text{dB}$ |
# | Return loss $R = |S_{11}|^2$ ($1550\;\text{nm}$) | $-55.7\;\text{dB}$ / $-46.1\;\text{dB}$ | $-46.6\;\text{dB}$ / $-45.4\;\text{dB}$ | $-43.7\;\text{dB}$ | $-51.5\;\text{dB}$ | Adiabatic input taper ($R \ll -30\;\text{dB}$) |
# | Band-averaged $\langle T_{\text{tot}}\rangle$ ($1505\text{--}1595\;\text{nm}$) | $0.8524$ / $0.8541$ | $0.8887$ / $0.8908$ | $0.9103$ | $0.9120$ | Passivity $\le 1.0$ |
# | Band-averaged $\langle \text{EL}\rangle$ [min, max] ($1505\text{--}1595\;\text{nm}$) | $0.696\;\text{dB}$ / $0.687\;\text{dB}$ | $0.513\;\text{dB}$ [$0.403, 0.659$] / $0.503\;\text{dB}$ | $0.408\;\text{dB}$ [$0.308, 0.550$] | $0.401\;\text{dB}$ [$0.294, 0.519$] | Passivity $\ge 0.0\;\text{dB}$ |
# | Max mirror asymmetry $\max(|T_1 - T_4|, |T_2 - T_3|)$ | $(8.1 \times 10^{-8}, 8.3 \times 10^{-8})$ | $(6.0 \times 10^{-8}, 1.0 \times 10^{-7})$ | $(4.6 \times 10^{-8}, 1.8 \times 10^{-7})$ | $(5.1 \times 10^{-8}, 2.1 \times 10^{-7})$ | Exact $y \mapsto -y$ symmetry: $0$ |
# | Outer port Mode 0 / Mode 1 share ($1550\;\text{nm}$) | $97.09\% / 2.91\%$ ($99.15\%$ init) | $96.71\% \; / \; 3.29\%$ | $97.98\% \; / \; 2.02\%$ | — | Fundamental mode operation ($\sim 98\%$ Mode 0) |
# | Outer port Mode 0 share band mean [min, max] | $96.05\%$ ($98.37\%$ init) | $95.58\%$ [$93.31\%, 96.86\%$] | $98.29\%$ [$96.12\%, 99.39\%$] | — | Fundamental mode operation |
# | 2D flux-box total $\langle \Phi_{\text{box}}\rangle$ (east / lateral / west) | $0.9999$ ($90.50\% / 9.15\% / 0.34\%$) | $1.0000$ ($94.06\% / 5.69\% / 0.25\%$) | $0.9999$ ($95.09\% / 4.62\% / 0.28\%$) | — | Lossless 2D Poynting conservation: $1.0000$ |
#
# Key physical takeaways and **conclusions that changed from `--quick` to `--full`**:
#
# 1. **Pairwise mirror symmetry and energy conservation**: Across the entire
#    $1505\text{--}1595\;\text{nm}$ band, outer-pair ($|T_1 - T_4|$) and
#    inner-pair ($|T_2 - T_3|$) asymmetries remain below $2.1 \times 10^{-7}$
#    (Fig. 11b), confirming that the symmetric Yee grid and Kottke subpixel
#    averaging preserve $y \mapsto -y$ parity to floating-point precision. The
#    closed 2D Poynting flux box accounts for $100.00\%$ of the injected power
#    ($\langle \Phi_{\text{box}}\rangle = 1.0000$ in `task_4` and $0.9999$ in
#    `task_15`, Fig. 11a): the power not coupled into the four fundamental
#    output modes ($\approx 11.1\%$ in `task_4`, $\approx 9.0\%$ in `task_15`)
#    is quantitatively accounted for by lateral cladding radiation ($5.69\%$ vs
#    $4.62\%$, down from $9.15\%$ in the under-resolved $10\;\text{cpw}$
#    `--quick` run) and unguided/higher-order forward power crossing the east
#    plane ($5.19\%$ vs $4.06\%$).
# 2. **Why $(5.00, 11.20\text{--}11.30)\;\mu\text{m}$ outperforms $(4.90, 11.15)\;\mu\text{m}$ in 2D EIM (lateral-TM vs lateral-TE Goos–Hänchen shift)**:
#    In the 3D vertical-slot geometry, the $E_z$-polarized slot mode is laterally
#    TE at the vertical sidewalls ($\Delta W_{\text{TE}} = 0.189\;\mu\text{m}$,
#    giving $W_{e,\text{TE}} = 5.089\;\mu\text{m}$ at $W_{\text{MMI}} = 4.90\;\mu\text{m}$
#    and image spacing $W_{e,\text{TE}}/4 = 1.272\;\mu\text{m}$, closely matching
#    the output pitch $p = 1.30\;\mu\text{m}$). In 2D EIM (`Ex, Ey, Hz`), the lateral boundary
#    condition at $y = \pm W_{\text{MMI}}/2$ is TM ($\Delta W_{\text{TM}} = 0.044\;\mu\text{m}$).
#    At $W_{\text{MMI}} = 4.90\;\mu\text{m}$, $W_{e,\text{TM}} = 4.944\;\mu\text{m}$
#    places the outer self-images at $y \approx \pm \frac{3}{8}W_{e,\text{TM}} = \pm 1.854\;\mu\text{m}$—about
#    $96\;\text{nm}$ inboard of the outer taper centers at $y_c = \pm 1.95\;\mu\text{m}$.
#    Consequently, the outer ports capture less fundamental power ($T_{\text{out}} = 0.2019$
#    vs $T_{\text{in}} = 0.2471$) and excite $3.29\%$ antisymmetric Mode 1 at
#    $1550\;\text{nm}$ ($96.71\%$ Mode 0, Fig. 12b). Widening the box to
#    $W_{\text{MMI}} = 5.00\;\mu\text{m}$ (`task_15` and `task_23`) compensates
#    the $\Delta W_{\text{TE}} - \Delta W_{\text{TM}} = 0.145\;\mu\text{m}$
#    Goos–Hänchen difference, raising outer-port Mode 0 purity to $97.98\%$ at
#    $1550\;\text{nm}$ ($98.29\%$ band average, Fig. 12a), lowering excess loss
#    to $0.325\;\text{dB}$ at $(5.00, 11.20)\;\mu\text{m}$ (`task_15`), and
#    reducing inner/outer imbalance seventeen-fold from $0.0451$ to $0.0027$
#    ($0.050\;\text{dB}$ uniformity) at $(5.00, 11.30)\;\mu\text{m}$ (`task_23`).
# 3. **How conclusions changed from `--quick` to `--full`**:
#    - In the `--quick` $3 \times 3$ sweep (run at $7\;\text{cpw}$), numerical
#      dispersion ($\Delta L_{\text{disp}} \approx 0.35\;\mu\text{m}$) shifted
#      the apparent self-image upstream so that transmission at
#      $W_{\text{MMI}} = 5.00\;\mu\text{m}$ appeared to *decrease* with length
#      from $L_{\text{MMI}} = 11.10\;\mu\text{m}$ ($T_{\text{tot}} = 0.8722$,
#      $\text{EL} = 0.594\;\text{dB}$, `task_13`) to $11.20\;\mu\text{m}$
#      ($0.8533$, $0.689\;\text{dB}$, `task_15`), and the lowest imbalance in
#      `--quick` occurred at $(4.80, 11.10)\;\mu\text{m}$ (`task_7`,
#      $|T_{\text{in}} - T_{\text{out}}| = 0.0097$ but $\text{EL} = 1.620\;\text{dB}$).
#      In `--full` ($20\;\text{cpw}$), suppressing numerical dispersion reverses
#      that trend: along $W_{\text{MMI}} = 5.00\;\mu\text{m}$, $T_{\text{tot}}(1550\;\text{nm})$
#      *increases* from $0.9058$ ($11.10\;\mu\text{m}$) to $0.9279$
#      ($11.20\;\mu\text{m}$, `task_15`) and imbalance drops monotonically from
#      $0.0212$ ($11.10\;\mu\text{m}$) through $0.0075$ ($11.20\;\mu\text{m}$) to
#      $0.0027$ at $11.30\;\mu\text{m}$ (`task_23`).
#    - In the `--quick` resolution ladder ($5, 7, 10\;\text{cpw}$ vs $20\;\text{cpw}$
#      ref), `staircase_fine` at $10\;\text{cpw}$ ($T_{\text{tot}} = 0.8843$)
#      appeared closer to the reference than `kottke_fine` at $10\;\text{cpw}$
#      ($0.8503$). In `--full` ($10, 14, 20\;\text{cpw}$ vs $40\;\text{cpw}$ ref),
#      extending the ladder to $14$ and $20\;\text{cpw}$ reverses that coarse-grid
#      cancellation: Kottke converges monotonically at $\mathcal{O}(\Delta x^2)$
#      to $|T(1550) - T_{\text{ref}}| = 2.24 \times 10^{-3}$ (band RMS
#      $4.68 \times 10^{-3}$), beating staircase ($4.06 \times 10^{-3}$, band
#      RMS $6.80 \times 10^{-3}$).
#
# ### 7.3 Extended $5 \times 6$ design sweep (`task_7`–`task_36`, Fig. 13), interior optima & analytic comparison
#
# The complete 30-point $5 \times 6$ parameter sweep at $20\;\text{cpw}$
# (`task_7`–`task_15` from the initial $3 \times 3$ grid plus `task_16`–`task_36`
# from the extended grid) yields the following performance at
# $\lambda_0 = 1550\;\text{nm}$ and across the $1505\text{--}1595\;\text{nm}$ band:
#
# | Task | Grid | $W_{\text{MMI}}$ [$\mu\text{m}$] | $L_{\text{MMI}}$ [$\mu\text{m}$] | $T_{\text{in}}(1550)$ | $T_{\text{out}}(1550)$ | $T_{\text{tot}}(1550)$ | Imbalance $|T_{\text{in}} - T_{\text{out}}|$ | $\text{EL}(1550)$ [dB] | Uniformity [dB] | Band $\langle \text{EL}\rangle$ [dB] |
# |---|---|---|---|---|---|---|---|---|---|---|
# | `task_7` | $3\times 3$ | $4.80$ | $11.10$ | $0.2443$ | $0.1851$ | $0.8588$ | $0.0593$ | $0.661$ | $1.207$ | $0.670$ |
# | `task_8` | $3\times 3$ | $4.80$ | $11.15$ | $0.2333$ | $0.1888$ | $0.8442$ | $0.0445$ | $0.736$ | $0.919$ | $0.699$ |
# | `task_9` | $3\times 3$ | $4.80$ | $11.20$ | $0.2249$ | $0.1946$ | $0.8390$ | $0.0303$ | $0.762$ | $0.627$ | $0.731$ |
# | `task_16` | $5\times 6$ | $4.80$ | $11.25$ | $0.2259$ | $0.1965$ | $0.8446$ | $0.0294$ | $0.733$ | $0.606$ | $0.762$ |
# | `task_17` | $5\times 6$ | $4.80$ | $11.30$ | $0.2304$ | $0.1941$ | $0.8490$ | $0.0363$ | $0.711$ | $0.745$ | $0.795$ |
# | `task_18` | $5\times 6$ | $4.80$ | $11.35$ | $0.2295$ | $0.1917$ | $0.8424$ | $0.0378$ | $0.745$ | $0.781$ | $0.831$ |
# | `task_10` | $3\times 3$ | $4.90$ | $11.10$ | $0.2462$ | $0.2031$ | $0.8986$ | $0.0431$ | $0.464$ | $0.836$ | $0.504$ |
# | `task_11` | $3\times 3$ | $4.90$ | $11.15$ | $0.2471$ | $0.2019$ | $0.8980$ | $0.0451$ | $0.467$ | $0.876$ | $0.513$ |
# | `task_12` | $3\times 3$ | $4.90$ | $11.20$ | $0.2442$ | $0.2009$ | $0.8903$ | $0.0433$ | $0.505$ | $0.848$ | $0.526$ |
# | `task_19` | $5\times 6$ | $4.90$ | $11.25$ | $0.2423$ | $0.1974$ | $0.8793$ | $0.0449$ | $0.559$ | $0.890$ | $0.540$ |
# | `task_20` | $5\times 6$ | $4.90$ | $11.30$ | $0.2431$ | $0.1927$ | $0.8716$ | $0.0504$ | $0.597$ | $1.010$ | $0.558$ |
# | `task_21` | $5\times 6$ | $4.90$ | $11.35$ | $0.2442$ | $0.1915$ | $0.8714$ | $0.0527$ | $0.598$ | $1.057$ | $0.579$ |
# | `task_13` | $3\times 3$ | $5.00$ | $11.10$ | $0.2158$ | $0.2371$ | $0.9058$ | $0.0212$ | $0.430$ | $0.407$ | $0.422$ |
# | `task_14` | $3\times 3$ | $5.00$ | $11.15$ | $0.2239$ | $0.2363$ | $0.9204$ | $0.0123$ | $0.360$ | $0.233$ | $0.414$ |
# | **`task_15`** | $3\times 3$ (**min EL**) | **$5.00$** | **$11.20$** | **$0.2282$** | **$0.2357$** | **$0.9279$** | **$0.0075$** | **$0.325$** | **$0.141$** | **$0.408$** |
# | `task_22` | $5\times 6$ | $5.00$ | $11.25$ | $0.2298$ | $0.2336$ | $0.9267$ | $0.0038$ | $0.330$ | $0.072$ | $0.404$ |
# | **`task_23`** | $5\times 6$ (**min imb**) | **$5.00$** | **$11.30$** | **$0.2311$** | **$0.2284$** | **$0.9191$** | **$0.0027$** | **$0.366$** | **$0.050$** | **$0.401$** |
# | `task_24` | $5\times 6$ | $5.00$ | $11.35$ | $0.2339$ | $0.2236$ | $0.9151$ | $0.0103$ | $0.385$ | $0.196$ | $0.400$ |
# | `task_25` | $5\times 6$ | $5.10$ | $11.10$ | $0.2059$ | $0.2306$ | $0.8730$ | $0.0247$ | $0.590$ | $0.491$ | $0.688$ |
# | `task_26` | $5\times 6$ | $5.10$ | $11.15$ | $0.2097$ | $0.2261$ | $0.8715$ | $0.0164$ | $0.597$ | $0.328$ | $0.624$ |
# | `task_27` | $5\times 6$ | $5.10$ | $11.20$ | $0.2015$ | $0.2326$ | $0.8682$ | $0.0311$ | $0.614$ | $0.623$ | $0.572$ |
# | `task_28` | $5\times 6$ | $5.10$ | $11.25$ | $0.1940$ | $0.2454$ | $0.8789$ | $0.0514$ | $0.560$ | $1.020$ | $0.534$ |
# | `task_29` | $5\times 6$ | $5.10$ | $11.30$ | $0.2012$ | $0.2509$ | $0.9041$ | $0.0497$ | $0.438$ | $0.960$ | $0.504$ |
# | `task_30` | $5\times 6$ | $5.10$ | $11.35$ | $0.2165$ | $0.2452$ | $0.9235$ | $0.0287$ | $0.346$ | $0.541$ | $0.475$ |
# | `task_31` | $5\times 6$ | $5.20$ | $11.10$ | $0.1592$ | $0.1846$ | $0.6876$ | $0.0254$ | $1.627$ | $0.642$ | $1.604$ |
# | `task_32` | $5\times 6$ | $5.20$ | $11.15$ | $0.1737$ | $0.1903$ | $0.7281$ | $0.0166$ | $1.378$ | $0.397$ | $1.477$ |
# | `task_33` | $5\times 6$ | $5.20$ | $11.20$ | $0.1786$ | $0.2018$ | $0.7608$ | $0.0231$ | $1.187$ | $0.529$ | $1.365$ |
# | `task_34` | $5\times 6$ | $5.20$ | $11.25$ | $0.1771$ | $0.2120$ | $0.7783$ | $0.0349$ | $1.089$ | $0.781$ | $1.262$ |
# | `task_35` | $5\times 6$ | $5.20$ | $11.30$ | $0.1754$ | $0.2150$ | $0.7807$ | $0.0396$ | $1.075$ | $0.884$ | $1.158$ |
# | `task_36` | $5\times 6$ | $5.20$ | $11.35$ | $0.1803$ | $0.2132$ | $0.7869$ | $0.0329$ | $1.041$ | $0.729$ | $1.057$ |
#
# **Both 2D EIM optima are strictly interior to the extended $5 \times 6$ grid**
# ($W_{\text{MMI}} \in [4.80, 5.20]\;\mu\text{m} \times L_{\text{MMI}} \in [11.10, 11.35]\;\mu\text{m}$, Fig. 13):
#
# 1. **Global minimum inner/outer imbalance (and best uniformity) at $1550\;\text{nm}$**:
#    Occurs at **`task_23` ($(W_{\text{MMI}}, L_{\text{MMI}}) = (5.00, 11.30)\;\mu\text{m}$)**,
#    strictly interior to both the width ($4.80 < 5.00 < 5.20\;\mu\text{m}$) and
#    length ($11.10 < 11.30 < 11.35\;\mu\text{m}$) ranges, where
#    $|T_{\text{in}} - T_{\text{out}}| = 0.0027$ and uniformity reaches
#    $0.050\;\text{dB}$ ($T_{\text{in}} = 0.2311$, $T_{\text{out}} = 0.2284$,
#    $T_{\text{tot}} = 0.9191$, $\text{EL}(1550\;\text{nm}) = 0.366\;\text{dB}$,
#    band-averaged $\langle \text{EL}\rangle = 0.401\;\text{dB}$). Along
#    $W_{\text{MMI}} = 5.00\;\mu\text{m}$, the signed difference
#    $T_{\text{in}} - T_{\text{out}}$ increases monotonically from $-0.0212$
#    ($11.10\;\mu\text{m}$) $\to -0.0123$ ($11.15\;\mu\text{m}$) $\to -0.0075$
#    ($11.20\;\mu\text{m}$) $\to -0.0038$ ($11.25\;\mu\text{m}$, `task_22`) and
#    crosses zero to $+0.0027$ ($11.30\;\mu\text{m}$, `task_23`) before growing to
#    $+0.0103$ ($11.35\;\mu\text{m}$, `task_24`), placing the exact zero-imbalance
#    crossing at $L_{\text{MMI}} \approx 11.28\;\mu\text{m}$.
# 2. **Global minimum excess insertion loss (maximum total transmission) at $1550\;\text{nm}$**:
#    Occurs at **`task_15` ($(W_{\text{MMI}}, L_{\text{MMI}}) = (5.00, 11.20)\;\mu\text{m}$)**,
#    which lay at the upper-right corner of the initial $3 \times 3$ subgrid but
#    is now strictly interior to the $5 \times 6$ sweep, achieving
#    $T_{\text{tot}}(1550\;\text{nm}) = 0.9279$ and $\text{EL}(1550\;\text{nm}) = 0.325\;\text{dB}$
#    (followed immediately by `task_22` at $(5.00, 11.25)\;\mu\text{m}$ with
#    $T_{\text{tot}} = 0.9267$, $\text{EL} = 0.330\;\text{dB}$, and imbalance
#    $0.0038$), before turning over to $0.9191$ at $11.30\;\mu\text{m}$ and
#    $0.9151$ at $11.35\;\mu\text{m}$ (and dropping to $0.8682$ at
#    $W_{\text{MMI}} = 5.10\;\mu\text{m}$ and $0.7608$ at $5.20\;\mu\text{m}$).
#
# Comparing all five swept widths against the Soldano & Pennings (1995) analytic
# self-imaging length $L_{\text{MMI}} = \frac{3 L_\pi}{4 N} = \frac{3 L_\pi}{16}$
# ($N = 4$) with $W_e$ evaluated per $W_{\text{MMI}}$ and the non-paraxial 1D EME
# length sweep at $\lambda_0 = 1550\;\text{nm}$ ($n_r = 2.9842$,
# $n_{\text{clad}} = 1.444$) confirms how the self-imaging length scales with
# $W_e^2$:
#
# | $W_{\text{MMI}}$ [$\mu\text{m}$] | 2D TM $W_{e,\text{TM}}$ [$\mu\text{m}$] | Paraxial $L_{\pi,\text{TM}}$ [$\mu\text{m}$] | Exact $L_{\pi,\text{exact}}$ [$\mu\text{m}$] | Analytic $L_{\text{MMI}} = \frac{3 L_{\pi,\text{TM}}}{16}$ [$\mu\text{m}$] | 3D TE $W_{e,\text{TE}}$ [$\mu\text{m}$] | 3D Analytic $\frac{3 L_{\pi,\text{TE}}}{16}$ [$\mu\text{m}$] | 1D EME $L_{\text{opt}}$ ($\max T_{\text{tot}}$) [$\mu\text{m}$] | 2D FIT $5\times 6$ best $L$ ($\max T_{\text{tot}}$) [$\mu\text{m}$] | 2D FIT $5\times 6$ best $L$ ($\min |T_{\text{in}}-T_{\text{out}}|$) [$\mu\text{m}$] |
# |---|---|---|---|---|---|---|---|---|---|
# | $4.80$ | $4.8442$ | $60.24$ | $60.03$ | $11.29$ | $4.9889$ | $11.98$ | $10.74$ ($T = 0.945$) | $11.10$ ($T = 0.8588$, `task_7`) | $11.25$ ($\text{imb} = 0.0294$, `task_16`) |
# | $4.90$ | $4.9442$ | $62.75$ | $62.55$ | $11.77$ | $5.0889$ | $12.46$ | $11.09$ ($T = 0.965$) | $11.10\text{--}11.15$ ($T = 0.8986$, `task_10`) | $11.10$ ($\text{imb} = 0.0431$, `task_10`) |
# | $5.00$ | $5.0442$ | $65.32$ | $65.11$ | $12.25$ | $5.1889$ | $12.96$ | $11.51$ ($T = 0.978$) | **$11.20$ ($T = 0.9279$, `task_15`)** | **$11.30$ ($\text{imb} = 0.0027$, `task_23`)** |
# | $5.10$ | $5.1442$ | $67.93$ | $67.72$ | $12.74$ | $5.2889$ | $13.46$ | $11.96$ ($T = 0.981$) | $11.35$ ($T = 0.9235$, `task_30`) | $11.15$ ($\text{imb} = 0.0164$, `task_26`) |
# | $5.20$ | $5.2442$ | $70.60$ | $70.39$ | $13.24$ | $5.3889$ | $13.98$ | $12.42$ ($T = 0.970$) | $11.35$ ($T = 0.7869$, `task_36`) | $11.15$ ($\text{imb} = 0.0166$, `task_32`) |
#
# Notice that at every width $W_{\text{MMI}}$, the non-paraxial EME and 2D FIT
# optimal lengths are $\approx 5.5\text{--}8\%$ shorter than the paraxial formula
# $L_{\text{MMI}} = \frac{3 L_\pi}{16}$ (e.g. $11.10\text{--}11.15\;\mu\text{m}$
# vs $11.77\;\mu\text{m}$ at $W_{\text{MMI}} = 4.90\;\mu\text{m}$, and
# $11.20\text{--}11.30\;\mu\text{m}$ in 2D FIT / $11.51\;\mu\text{m}$ in 1D EME
# vs $12.25\;\mu\text{m}$ at $W_{\text{MMI}} = 5.00\;\mu\text{m}$). This occurs
# for two physical reasons: first, in a high-index-contrast slab ($n_r = 2.9842$
# vs $n_{\text{clad}} = 1.444$), the exact propagation constants
# $\beta_m = \sqrt{k_0^2 n_r^2 - k_{y,m}^2}$ expand beyond the paraxial quadratic
# law as $\beta_0 - \beta_m \approx \frac{k_{y,m}^2 - k_{y,0}^2}{2 k_0 n_r}\left[1 + \frac{k_{y,m}^2 + k_{y,0}^2}{4 k_0^2 n_r^2}\right]$,
# so the quartic non-paraxial correction makes higher-order even modes
# ($m = 2, 4, 6$) beat faster than $m(m+2)\pi / (3 L_\pi)$ and focuses the 4-fold
# image $\approx 0.65\text{--}0.80\;\mu\text{m}$ upstream of $3 L_\pi / 16$
# (Fig. 3b). Second, at $W_{\text{MMI}} = 5.10$ and $5.20\;\mu\text{m}$, the
# self-imaging length shifts upward as $W_e^2$ ($L_{\text{opt,EME}} = 11.96$ and
# $12.42\;\mu\text{m}$, so $T_{\text{tot}}$ in the $[11.10, 11.35]\;\mu\text{m}$
# window increases toward the upper $11.35\;\mu\text{m}$ boundary), while the
# lateral image spacing $W_{e,\text{TM}} / 4 = 1.286\text{--}1.311\;\mu\text{m}$
# overshoots the fixed $1.30\;\mu\text{m}$ output waveguide pitch and degrades
# coupling at $W_{\text{MMI}} = 5.20\;\mu\text{m}$.
#
# ---
#
# ## 8. Resolution table & subpixel convergence analysis
#
# ### 8.1 Resolution per run (`task_0`–`task_15` and extended sweep `task_16`–`task_36`)
#
# Across the full study, `cells_per_wavelength` (`cpw`) is defined inside the
# high-index core at $\lambda_0 = 1550\;\text{nm}$. The smallest geometric
# features are the straight input/output waveguides ($W_1 = 0.40\;\mu\text{m}$)
# and the three oxide gaps between adjacent output tapers at the MMI exit face
# ($g_2 = p - W_2 = 1.30 - 0.80 = 0.50\;\mu\text{m}$):
#
# | Task(s) | Name | Averaging | `cpw` | $\Delta x$ [nm] | $\Delta y$ [nm] | Cells / $0.40\;\mu\text{m}$ WG | Cells / $0.50\;\mu\text{m}$ gap | Grid $N_x \times N_y \times N_z$ | `n_cells` | Batch time [s] |
# |---|---|---|---|---|---|---|---|---|---|---|
# | `task_0` | `kottke_coarse` | Kottke | $10$ | $50.24$ | $50.00$ | $8.00$ | $10.00$ | $442 \times 159 \times 1$ | $70{,}278$ | $60.4$ |
# | `task_1` | `staircase_coarse` | Staircase | $10$ | $50.24$ | $50.00$ | $8.00$ | $10.00$ | $442 \times 159 \times 1$ | $70{,}278$ | $116.3$ |
# | `task_2` | `kottke_medium` | Kottke | $14$ | $35.85$ | $35.75$ | $11.19$ | $13.99$ | $611 \times 214 \times 1$ | $130{,}754$ | $235.5$ |
# | `task_3` | `staircase_medium` | Staircase | $14$ | $35.85$ | $35.75$ | $11.19$ | $13.99$ | $611 \times 214 \times 1$ | $130{,}754$ | $243.0$ |
# | `task_4` | `kottke_fine` | Kottke | $20$ | $25.12$ | $25.09$ | $15.94$ | $19.93$ | $863 \times 296 \times 1$ | $255{,}448$ | $845.4$ |
# | `task_5` | `staircase_fine` | Staircase | $20$ | $25.12$ | $25.09$ | $15.94$ | $19.93$ | $863 \times 296 \times 1$ | $255{,}448$ | $718.2$ |
# | `task_6` | `kottke_reference` | Kottke | $40$ | $12.56$ | $12.55$ | $31.88$ | $39.86$ | $1705 \times 571 \times 1$ | $973{,}555$ | $5709.3$ |
# | `task_7`–`9` | `sweep_W4.80_L11.10..20` | Kottke | $20$ | $25.12$ | $25.09$ | $15.94$ | $19.93$ | $(861\text{--}865) \times 292 \times 1$ | $251{,}412\text{--}252{,}580$ | $707.2\text{--}743.2$ |
# | `task_10`–`12` | `sweep_W4.90_L11.10..20` | Kottke | $20$ | $25.12$ | $25.09$ | $15.94$ | $19.93$ | $(861\text{--}865) \times 296 \times 1$ | $254{,}856\text{--}256{,}040$ | $636.7\text{--}740.1$ |
# | `task_13`–`15` | `sweep_W5.00_L11.10..20` | Kottke | $20$ | $25.12$ | $25.09$ | $15.94$ | $19.93$ | $(861\text{--}865) \times 300 \times 1$ | $258{,}300\text{--}259{,}500$ | $647.3\text{--}867.8$ |
# | `task_16`–`18` | `sweep_W4.80_L11.25..35` | Kottke | $20$ | $25.12$ | $25.09$ | $15.94$ | $19.93$ | $(867\text{--}871) \times 292 \times 1$ | $253{,}164\text{--}254{,}332$ | $700.1\text{--}727.7$ |
# | `task_19`–`21` | `sweep_W4.90_L11.25..35` | Kottke | $20$ | $25.12$ | $25.09$ | $15.94$ | $19.93$ | $(867\text{--}871) \times 296 \times 1$ | $256{,}632\text{--}257{,}816$ | $537.5\text{--}683.2$ |
# | `task_22`–`24` | `sweep_W5.00_L11.25..35` | Kottke | $20$ | $25.12$ | $25.09$ | $15.94$ | $19.93$ | $(867\text{--}871) \times 300 \times 1$ | $260{,}100\text{--}261{,}300$ | $729.8\text{--}773.2$ |
# | `task_25`–`30` | `sweep_W5.10_L11.10..35` | Kottke | $20$ | $25.12$ | $25.09$ | $15.94$ | $19.93$ | $(861\text{--}871) \times 304 \times 1$ | $261{,}744\text{--}264{,}784$ | $722.4\text{--}793.7$ |
# | `task_31`–`36` | `sweep_W5.20_L11.10..35` | Kottke | $20$ | $25.12$ | $25.09$ | $15.94$ | $19.93$ | $(861\text{--}871) \times 308 \times 1$ | $265{,}188\text{--}268{,}268$ | $943.0\text{--}1012.2$ |
#
# ### 8.2 Explaining the convergence behavior (Fig. 14)
#
# Comparing `task_0`–`task_5` against the independent $40\;\text{cpw}$ reference
# (`task_6`, where $T_{\text{ref}}(1550\;\text{nm}) = 0.90023$,
# $\langle T_{\text{ref}}\rangle_{\text{band}} = 0.89081$, and
# $\langle \text{EL}_{\text{ref}}\rangle_{\text{band}} = 0.5026\;\text{dB}$)
# confirms that the accepted $20\;\text{cpw}$ Kottke resolution (`task_4`) is
# well converged: the relative change in $T_{\text{tot}}(1550\;\text{nm})$
# between the last two resolution study levels (`medium` $14\;\text{cpw} \to$
# `fine` $20\;\text{cpw}$) is **$1.037\%$** ($0.88869 \to 0.89800$, and
# **$0.479\%$** in band-averaged $\langle T_{\text{tot}}\rangle$:
# $0.88444 \to 0.88870$), while the relative change from `fine` ($20\;\text{cpw}$)
# to the independent $40\;\text{cpw}$ reference (`task_6`) is **$0.248\%$** at
# $1550\;\text{nm}$ ($0.89800 \to 0.90023$) and **$0.237\%$** in band-averaged
# $\langle T_{\text{tot}}\rangle$ ($0.88870 \to 0.89081$). Inspection of the
# single-wavelength curve (Fig. 14a, solid lines) also reveals two striking
# features:
#
# - At **$10\;\text{cpw}$**, Kottke has a larger single-wavelength error
#   ($|T(1550) - T_{\text{ref}}| = 5.00 \times 10^{-2}$) than staircase
#   ($2.83 \times 10^{-2}$).
# - Between **$14\;\text{cpw}$ and $20\;\text{cpw}$**, staircase plateaus and
#   slightly worsens ($3.46 \times 10^{-3} \to 4.06 \times 10^{-3}$ at
#   $1550\;\text{nm}$), whereas Kottke converges monotonically ($5.00 \times 10^{-2} \to 1.15 \times 10^{-2} \to 2.24 \times 10^{-3}$).
#
# Combining **band-averaged metrics**, **staircase width quantization**, and
# **Yee numerical-dispersion analysis** (`_helpers.mmi_discretization_diagnostics`)
# explains both observations quantitatively:
#
# | Resolution (`cpw`) | Scheme | $T_{\text{tot}}(1550)$ | $|T(1550) - T_{\text{ref}}|$ | Band $\mathrm{RMS}_\lambda(|T - T_{\text{ref}}|)$ | Band $\langle \text{EL}\rangle$ [dB] | $|\langle \text{EL}\rangle - \langle \text{EL}_{\text{ref}}\rangle|$ [dB] | Band $\mathrm{RMS}_\lambda(\text{EL} - \text{EL}_{\text{ref}})$ [dB] |
# |---|---|---|---|---|---|---|---|
# | $10$ (`coarse`) | Kottke | $0.85023$ | $5.00 \times 10^{-2}$ | $4.13 \times 10^{-2}$ | $0.6916$ | $0.1890$ | $0.2087$ |
# | $10$ (`coarse`) | Staircase | $0.87197$ | $2.83 \times 10^{-2}$ | $1.78 \times 10^{-2}$ | $0.5576$ | $0.0550$ | $0.0871$ |
# | $14$ (`medium`) | Kottke | $0.88869$ | $1.15 \times 10^{-2}$ | $1.04 \times 10^{-2}$ | $0.5342$ | $0.0316$ | $0.0516$ |
# | $14$ (`medium`) | Staircase | $0.89678$ | $3.46 \times 10^{-3}$ | $1.03 \times 10^{-2}$ | $0.4805$ | $0.0221$ | $0.0502$ |
# | $20$ (`fine`) | Kottke | $0.89800$ | $2.24 \times 10^{-3}$ | $4.68 \times 10^{-3}$ | $0.5131$ | $0.0105$ | $0.0230$ |
# | $20$ (`fine`) | Staircase | $0.89618$ | $4.06 \times 10^{-3}$ | $6.80 \times 10^{-3}$ | $0.5283$ | $0.0257$ | $0.0336$ |
# | $40$ (`reference`) | Kottke | $0.90023$ | — | — | $0.5026$ | — | — |
#
# | Resolution (`cpw`) | $\Delta x, \Delta y$ [nm] | Staircase $W_{\text{MMI}}(E_y)$ [$\mu\text{m}$] ($\Delta W_y$) | Staircase $W_{\text{MMI}}(E_x)$ [$\mu\text{m}$] ($\Delta W_x$) | $\Delta L_{\pi}(E_y), \Delta L_{\text{MMI}}(E_y)$ [$\mu\text{m}$] | $\Delta L_{\pi}(E_x), \Delta L_{\text{MMI}}(E_x)$ [$\mu\text{m}$] | $k\Delta x$ [rad] | Dispersion $\Delta\phi_{\text{disp}}$ [rad] | Axial shift $\Delta L_{\text{disp}}$ [$\mu\text{m}$] |
# |---|---|---|---|---|---|---|---|---|
# | $10$ | $50.24, 50.00$ | $4.9000$ ($0.0\;\text{nm}$, $98\Delta y$) | $4.9500$ ($+50.0\;\text{nm}$, $99\Delta y$) | $0.000, \; 0.000$ | $+1.28, \; +0.239$ | $0.608$ | $2.08\;\text{rad}$ ($119^\circ$) | $0.172$ ($1.54\%$) |
# | $14$ | $35.85, 35.75$ | $4.8979$ ($-2.1\;\text{nm}$, $137\Delta y$) | $4.9337$ ($+33.7\;\text{nm}$, $138\Delta y$) | $-0.053, \; -0.010$ | $+0.86, \; +0.161$ | $0.434$ | $1.06\;\text{rad}$ ($61^\circ$) | $0.087$ ($0.78\%$) |
# | $20$ | $25.12, 25.09$ | $4.8927$ ($-7.3\;\text{nm}$, $195\Delta y$) | $4.9178$ ($+17.8\;\text{nm}$, $196\Delta y$) | $-0.184, \; -0.035$ | $+0.45, \; +0.085$ | $0.304$ | $0.52\;\text{rad}$ ($30^\circ$) | $0.043$ ($0.39\%$) |
# | $40$ | $12.56, 12.55$ | $4.8927$ ($-7.3\;\text{nm}$, $390\Delta y$) | $4.9053$ ($+5.3\;\text{nm}$, $391\Delta y$) | $-0.184, \; -0.035$ | $+0.13, \; +0.025$ | $0.152$ | $0.13\;\text{rad}$ ($7.4^\circ$) | $0.011$ ($0.10\%$) |
#
# 1. **(a) Spectral ripples vs band-averaged RMS and excess-loss metrics (Fig. 14a–c)**:
#    Weak multimode reflections between the input taper junction ($x = L_1$) and
#    the three $0.50\;\mu\text{m}$ oxide gaps at the MMI exit face
#    ($x = L_1 + L_{\text{MMI}}$) superimpose a $\sim 0.02\text{--}0.05\;\text{dB}$
#    spectral ripple onto $T_{\text{tot}}(\lambda)$ (Fig. 14c). At $14\;\text{cpw}$,
#    the `staircase_medium` spectrum happens to cross the $40\;\text{cpw}$
#    reference curve near $\lambda = 1550\;\text{nm}$ ($T = 0.8968$ vs $0.9002$,
#    giving $|T(1550) - T_{\text{ref}}| = 3.46 \times 10^{-3}$), while sitting
#    above $T_{\text{ref}}$ across $1515\text{--}1545\;\text{nm}$. Evaluating the
#    **band-averaged RMS error** $\mathrm{RMS}_\lambda(|T(\lambda) - T_{\text{ref}}(\lambda)|)$
#    across $1505\text{--}1595\;\text{nm}$ eliminates this single-wavelength
#    crossing artifact: at $14\;\text{cpw}$, Kottke ($1.04 \times 10^{-2}$) and
#    staircase ($1.03 \times 10^{-2}$) have virtually identical band RMS error
#    ($0.052\;\text{dB}$ vs $0.050\;\text{dB}$ in RMS excess loss). Moving from
#    $14\;\text{cpw}$ to $20\;\text{cpw}$, Kottke's band RMS error drops more
#    than two-fold to $4.68 \times 10^{-3}$ and its band-averaged excess-loss
#    error $|\langle \text{EL}\rangle - \langle \text{EL}_{\text{ref}}\rangle|$
#    falls from $0.0316\;\text{dB}$ to $0.0105\;\text{dB}$ (Fig. 14b), whereas
#    staircase stalls at $6.80 \times 10^{-3}$ band RMS and its excess-loss
#    error actually *increases* from $0.0221\;\text{dB}$ ($14\;\text{cpw}$) to
#    $0.0257\;\text{dB}$ ($20\;\text{cpw}$).
# 2. **(b) Staircase MMI width quantization explains the $14\text{--}20\;\text{cpw}$ plateau**:
#    In a 2D TM simulation, the dominant transverse electric field $E_y$ sits at
#    half-cell centers $y_{j+1/2}$, while the longitudinal field $E_x$ sits on
#    grid nodes $y_j$. Because `PICSimulation` rounds the total cell count $N_y$
#    across the $[-3.45, +3.45]\;\mu\text{m}$ domain plus PMLs, $\Delta y$ at
#    $14\;\text{cpw}$ ($35.75\;\text{nm}$) and $20\;\text{cpw}$ ($25.09\;\text{nm}$)
#    does not divide $W_{\text{MMI}} = 4.9000\;\mu\text{m}$ evenly. Crucially,
#    the staircase $E_y$ MMI width error is **larger at $20\;\text{cpw}$
#    ($195\Delta y = 4.8927\;\mu\text{m}$, $\Delta W_y = -7.3\;\text{nm}$) than
#    at $14\;\text{cpw}$ ($137\Delta y = 4.8979\;\mu\text{m}$,
#    $\Delta W_y = -2.1\;\text{nm}$)**! Because $L_\pi \propto W_e^2$, a
#    $-7.3\;\text{nm}$ width truncation at $20\;\text{cpw}$ shifts $L_\pi$ by
#    $-0.184\;\mu\text{m}$ and $L_{\text{MMI}}$ by $-0.035\;\mu\text{m}$ (and
#    simultaneously rounds the input waveguide from $0.4000\;\mu\text{m}$ to
#    $17\Delta y = 0.4265\;\mu\text{m}$ and the central MMI-face taper gap from
#    $0.5000\;\mu\text{m}$ to $19\Delta y = 0.4767\;\mu\text{m}$). Evaluating
#    `_helpers.mmi_eme_length_sweep` at $W_{\text{stair}}(E_y) = 4.8927\;\mu\text{m}$
#    predicts a transmission drop of $\Delta T_{\text{EME}} = -2.30 \times 10^{-3}$
#    relative to $4.9000\;\mu\text{m}$ (and multiplying $\Delta W_y = -7.3\;\text{nm}$
#    by the 2D FIT sweep slope $\partial T_{\text{tot}}/\partial W_{\text{MMI}} \approx 0.38\;\mu\text{m}^{-1}$
#    between `task_8` and `task_14` gives $\Delta T_{\text{FIT}} \approx -2.8 \times 10^{-3}$).
#    This width-quantization offset directly accounts for the observed gap
#    between `staircase_fine` ($T = 0.89618$, error $4.06 \times 10^{-3}$) and
#    `kottke_fine` ($T = 0.89800$, error $2.24 \times 10^{-3}$, where Kottke
#    subpixel smoothing restores the effective MMI width to $4.8990\;\mu\text{m}$).
# 3. **(c) Numerical-dispersion phase error at $10\;\text{cpw}$ and accidental cancellation in `staircase_coarse`**:
#    At $10\;\text{cpw}$ ($\Delta x = 50.24\;\text{nm}$), the core wavenumber is
#    $k = 2\pi n_{\text{eff,slot}} / \lambda_0 = 12.10\;\mu\text{m}^{-1}$, so
#    $k\Delta x = 0.608\;\text{rad}$. Over the $L_{\text{MMI}} = 11.15\;\mu\text{m}$
#    cavity ($\approx 21.5$ core wavelengths), the 2nd-order Yee/FIT scheme
#    accumulates a numerical-dispersion phase error of
#    $$\Delta\phi_{\text{disp}} \approx L_{\text{MMI}}\, k\, \frac{(k\Delta x)^2}{24} = 2.08\;\text{rad}\;(119^\circ), \qquad \Delta L_{\text{disp}} \approx L_{\text{MMI}}\,\frac{(k\Delta x)^2}{24} = 0.172\;\mu\text{m}\;(1.54\%\;\text{of}\;L_{\text{MMI}}).$$
#    At $10\;\text{cpw}$, this bulk $\mathcal{O}(\Delta x^2)$ phase error shifts
#    the 4-fold image plane axially and dominates over boundary smoothing;
#    Kottke preserves the exact $4.9000\;\mu\text{m}$ width and therefore
#    exhibits the uncompensated dispersion drop ($T_{\text{tot}} = 0.8502$,
#    error $5.00 \times 10^{-2}$). Meanwhile, at $10\;\text{cpw}$ the transverse
#    step happens to be **$\Delta y = 50.00\;\text{nm}$ exactly**, which divides
#    every nominal transverse dimension ($W_1 = 0.40\;\mu\text{m} = 8\Delta y$,
#    $W_2 = 0.80\;\mu\text{m} = 16\Delta y$, $g_2 = 0.50\;\mu\text{m} = 10\Delta y$,
#    $W_{\text{MMI}} = 4.90\;\mu\text{m} = 98\Delta y$) with zero $E_y$ rounding
#    error while point-sampling both boundary $E_x$ nodes at
#    $y = \pm 2.45\;\mu\text{m}$ as core ($W_{\text{MMI}}(E_x) = 99\Delta y = 4.9500\;\mu\text{m}$,
#    $\Delta W_x = +50\;\text{nm}$, shifting $\Delta L_{\text{MMI}}(E_x) = +0.239\;\mu\text{m}$).
#    Because $(4.90, 11.15)\;\mu\text{m}$ is under-wide in 2D EIM, this
#    $+50\;\text{nm}$ staircase $E_x$ widening increases outer-arm coupling
#    ($T_{\text{out}} = 0.2029$ vs $0.1902$ in Kottke) and partially cancels
#    the coarse-grid dispersion loss at $10\;\text{cpw}$. As $\Delta x$ refines
#    to $14$ and $20\;\text{cpw}$, $\Delta\phi_{\text{disp}}$ falls as
#    $\Delta x^2$ ($2.08 \to 1.06 \to 0.52\;\text{rad}$) and Kottke converges at
#    second order, whereas staircase is left with its non-monotonic width
#    quantization plateau.
#
# ---
#
# ## 9. Limits, failure modes & when 2D EIM breaks down
#
# Reducing the 3D $\text{Si–GaN–Si}$ vertical-slot MMI splitter to a 2D
# effective-index model reproduces the 4-fold self-imaging carpet, fundamental
# mode purity of $96.71\%$ at $(4.90, 11.15)\;\mu\text{m}$ and $97.98\%$ at
# $(5.00, 11.20)\;\mu\text{m}$ at $1550\;\text{nm}$ ($98.29\%$ band mean), and
# sub-dB excess insertion loss ($0.467\;\text{dB}$ at $(4.90, 11.15)\;\mu\text{m}$,
# $0.325\;\text{dB}$ at $(5.00, 11.20)\;\mu\text{m}$, and $0.366\;\text{dB}$
# with $0.050\;\text{dB}$ uniformity at $(5.00, 11.30)\;\mu\text{m}$) on a
# compact 2D mesh ($0.255\;\text{Mcells}$ at $20\;\text{cpw}$). Both the
# physical 2D EIM reduction and the numerical FIT discretization have
# well-defined limits and failure modes [1, 3, 4]:
#
# 1. **Polarization swap at the vertical etched sidewalls (Goos–Hänchen shift)**:
#    In 3D, the fundamental vertical-slot mode is **vertically TM** ($E_z$
#    normal to the horizontal $\text{Si/GaN}$ interfaces, giving
#    $n_{\text{eff,slot}} = 2.9842$) and **laterally TE** ($E_z$ tangential to
#    the vertical sidewalls at $y = \pm W_{\text{MMI}}/2$), with lateral
#    Goos–Hänchen penetration depth
#    $\Delta W_{\text{TE}} = \frac{\lambda_0}{\pi}(n_{\text{eff,slot}}^2 - n_{\text{SiO}_2}^2)^{-1/2} = 0.189\;\mu\text{m}$.
#    In Wakis's 2D in-plane FIT reduction (`Ex`, `Ey`, `Hz`), the transverse
#    electric field $E_y$ is **normal** to the vertical sidewalls
#    ($y = \pm W_{\text{MMI}}/2$), obeying lateral TM boundary conditions with a
#    smaller Goos–Hänchen penetration depth
#    $\Delta W_{\text{TM}} = (n_{\text{SiO}_2}/n_{\text{eff,slot}})^2 \Delta W_{\text{TE}} = 0.044\;\mu\text{m}$.
#    The resulting $\Delta W_{\text{TE}} - \Delta W_{\text{TM}} = 0.145\;\mu\text{m}$
#    reduction in effective width $W_e$ shifts the minimum-imbalance MMI width
#    in 2D EIM from $W_{\text{MMI}} = 4.90\;\mu\text{m}$ (3D lateral-TE) to
#    $W_{\text{MMI}} = 5.00\;\mu\text{m}$ (`task_23` at $L_{\text{MMI}} = 11.30\;\mu\text{m}$
#    with $|T_{\text{in}} - T_{\text{out}}| = 0.0027$, strictly in the interior of
#    the $5 \times 6$ sweep in Fig. 13).
# 2. **Out-of-plane radiative scattering at the MMI-to-taper junctions**:
#    At $x = L_1 + L_{\text{MMI}}$, the abrupt step from the $W_{\text{MMI}}$
#    slab into four $W_2 = 0.80\;\mu\text{m}$ tapers separated by
#    $g_2 = 0.50\;\mu\text{m}$ oxide gaps scatters unguided light into both
#    in-plane cladding waves ($4.6\text{--}5.7\%$ lateral flux in Fig. 11a) and
#    3D vertical radiation modes that a 2D slab model cannot capture.
# 3. **Chromatic dispersion of the 5-layer slot stack**: Across
#    $1500\text{--}1600\;\text{nm}$, the vertical-slot effective index varies
#    from $n_{\text{eff,slot}}(1500\;\text{nm}) = 3.0120$ to
#    $n_{\text{eff,slot}}(1600\;\text{nm}) = 2.9562$ ($\Delta n_{\text{eff,slot}} = -0.0558$,
#    shown in `02_slot_eim_vertical_stack.png`), slightly steepening the
#    wavelength roll-off of $L_\pi(\lambda)$ relative to a fixed-index 2D slab.
# 4. **Numerical failure modes in multimode interference convergence**:
#    - *Coarse-grid dispersion regime ($\le 10\;\text{cpw}$)*: Because an MMI
#      splitter relies on relative phase beating over tens of core wavelengths,
#      running at $10\;\text{cpw}$ ($k\Delta x = 0.608\;\text{rad}$) incurs a
#      bulk Yee phase error $\Delta\phi_{\text{disp}} \approx 2.08\;\text{rad}$
#      ($\Delta L_{\text{disp}} \approx 0.172\;\mu\text{m}$) that swamps
#      subpixel boundary accuracy and can accidentally cancel opposite-sign
#      staircase width biases. Reliable MMI phase predictions require
#      $\ge 14\text{--}20\;\text{cpw}$ ($\Delta L_{\text{disp}} \le 0.04\text{--}0.09\;\mu\text{m}$).
#    - *Staircase width-locking on non-commensurate grids*: Because
#      $L_{\text{MMI}} \propto W_e^2$, staircasing locks the device onto
#      $W_{\text{stair}} = N_{\text{core}}\Delta y$, which jumps non-monotonically
#      with resolution ($\Delta W_y = 0.0\;\text{nm}$ at $10\;\text{cpw}$,
#      $-2.1\;\text{nm}$ at $14\;\text{cpw}$, $-7.3\;\text{nm}$ at $20\;\text{cpw}$)
#      and causes staircase convergence to plateau near $4 \times 10^{-3}$.
#    - *Single-wavelength vs broadband metrics and reference bias*: Spectral
#      Fabry–Pérot ripples between the MMI faces can make a coarse or staircase
#      curve cross $T_{\text{ref}}(\lambda)$ at an isolated wavelength (as at
#      $14\;\text{cpw}$ staircase at $1550\;\text{nm}$); band-averaged RMS error
#      $\mathrm{RMS}_\lambda(|T - T_{\text{ref}}|)$ and $|\langle \text{EL}\rangle - \langle \text{EL}_{\text{ref}}\rangle|$
#      must therefore be inspected alongside single-wavelength error. Finally,
#      note that `kottke_reference` at $40\;\text{cpw}$ ($\Delta y = 12.55\;\text{nm}$)
#      uses Kottke smoothing itself, though by $40\;\text{cpw}$ the subpixel
#      correction is at the $\sim 10^{-3}$ level.
#
# ## 10. Exercises
#
# 1. **Fine interpolation around the zero-imbalance crossing**: Using the
#    completed $5 \times 6$ sweep data along $W_{\text{MMI}} = 5.00\;\mu\text{m}$
#    (`task_13`–`task_15` and `task_22`–`task_24`), interpolate
#    $T_{\text{in}}(L_{\text{MMI}}) - T_{\text{out}}(L_{\text{MMI}})$ between
#    $11.25\;\mu\text{m}$ ($-0.0038$) and $11.30\;\mu\text{m}$ ($+0.0027$) to
#    verify the zero-imbalance length $L_{\text{MMI}} \approx 11.279\;\mu\text{m}$.
# 2. **Slot thickness sensitivity**: Use `slot_slab_neff` in `_helpers.py` to
#    sweep the $\text{GaN}$ slot thickness over
#    $H_{\text{GaN}} \in \{60, 80, 100, 120, 140\}\;\text{nm}$ and plot how the
#    optimal 4-fold imaging length $L_{\text{MMI}}^{(4)} = \frac{3}{16}L_\pi$
#    shifts with $H_{\text{GaN}}$.
# 3. **1×2 self-imaging length**: Shorten `l_mmi` to half the 4-fold symmetric
#    self-imaging length ($\frac{1}{2} L_{\text{MMI}}^{(4)} \approx 5.58\;\mu\text{m}$)
#    and inspect `07_field_intensity_1550.png` to observe the 2-fold
#    interference maxima at $y = \pm W_e / 4 \approx \pm 1.24\;\mu\text{m}$
#    landing in the oxide gaps between the inner and outer output tapers.
#
# ## 11. References
#
# 1. D. Malka, Y. Danan, Y. Ramon, and Z. Zalevsky, "A Photonic $1 \times 4$
#    Power Splitter Based on Multimode Interference in Silicon–Gallium-Nitride
#    Slot Waveguide Structures," *Materials* **9**(7), 516 (2016),
#    DOI: [10.3390/ma9070516](https://doi.org/10.3390/ma9070516).
# 2. L. B. Soldano and E. C. M. Pennings, "Optical multi-mode interference
#    devices based on self-imaging: principles and applications," *J. Lightwave
#    Technol.* **13**(4), 615–627 (1995),
#    DOI: [10.1109/50.372474](https://doi.org/10.1109/50.372474).
# 3. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing
#    of discretized numerical methods," *Phys. Rev. E* **77**, 036611 (2008),
#    DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
# 4. V. R. Almeida, Q. Xu, C. A. Barrios, and M. Lipson, "Guiding and confining
#    light in void nanostructure," *Opt. Lett.* **29**(11), 1209–1211 (2004),
#    DOI: [10.1364/OL.29.001209](https://doi.org/10.1364/OL.29.001209).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="12. 1x4 MMI Power Splitter in a Si-GaN-Si Slot Waveguide (2D EIM)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run coarse 2D study locally (2x scaled-down resolutions, 16 tasks) and render all 15 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run converged 2D study across all 37 tasks (including extended 5x6 sweep) and render all 15 figures",
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
