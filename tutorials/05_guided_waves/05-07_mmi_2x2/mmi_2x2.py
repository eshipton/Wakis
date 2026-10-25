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
# # Tutorial 5.7 · 2×2 multimode-interference coupler
#
# **Abstract.** $2 \times 2$ multimode interference (MMI) couplers are workhorse
# $3\;\text{dB}$ directional splitters and quadrature hybrids in silicon photonic
# integrated circuits (PICs), offering broader optical bandwidth and looser
# lithography tolerances than directional couplers. By launching an off-axis
# fundamental $\text{TE}_0$ waveguide mode into a wide multi-mode slab section,
# self-imaging (Talbot interference) among the guided lateral eigenmodes produces
# two equal-intensity images with a $\pm 90^\circ$ quadrature phase relationship at
# the general-interference length $L_{\text{mmi}} \approx 3 L_\pi / 2$. In this
# tutorial we model a $220\;\text{nm}$ silicon-on-insulator (SOI) $2 \times 2$
# MMI power splitter ($1.0 \times 3.0\;\mu\text{m}$ MMI box routed via four
# symmetric cosine S-bends) in Wakis using the **2D effective-index method (EIM)**
# and **Kottke subpixel permittivity smoothing**, extract broadband $4 \times 4$
# scattering parameters across the $1500\text{--}1600\;\text{nm}$ telecom window,
# compare the MMI box length sweep against Soldano & Pennings (1995) analytical
# self-imaging theory, audit passivity and quadrature phase balance, and benchmark
# Kottke subpixel convergence against staircasing using an independent fine
# reference simulation.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Reduce a 3D $220\;\text{nm}$ SOI $2 \times 2$ MMI layout to a **2D
#    effective-index (EIM)** finite-integration-technique (FIT) model (`Ex`, `Ey`,
#    `Hz`; one periodic cell along $z$).
# 2. Construct a seamless counter-clockwise `PolySlab` combining a rectangular
#    MMI box, four cosine `sbend` access arms, and straight input/output ports
#    with **Kottke (2008) subpixel permittivity smoothing**.
# 3. Compute the guided lateral slab modes $\psi_\nu(y)$ and beat length
#    $L_\pi = \pi / (\beta_0 - \beta_1)$ from Soldano & Pennings (1995) theory,
#    including the supermode phase pre-accumulation in the converging S-bend
#    throats.
# 4. Extract broadband $4$-port S-parameters ($S_{11}, S_{21}, S_{31}, S_{41}$),
#    power splitting ratios, imbalance, and the $\approx 90^\circ$ quadrature phase
#    difference across $1500\text{--}1600\;\text{nm}$, and explain the physical
#    shift between 2D EIM and 3D FDTD.
#
# ## Contents
#
# 1. [Physical background & device specification](#1.-Physical-background-&-device-specification)
# 2. [Effective-Index Method (EIM) reduction](#2.-Effective-Index-Method-(EIM)-reduction)
# 3. [Parametric geometry & lateral MMI eigenmodes](#3.-Parametric-geometry-&-lateral-MMI-eigenmodes)
# 4. [Subpixel permittivity smoothing & resolution ladder](#4.-Subpixel-permittivity-smoothing-&-resolution-ladder)
# 5. [Study tasks & execution modes](#5.-Study-tasks-&-execution-modes)
# 6. [Generating the complete figure suite](#6.-Generating-the-complete-figure-suite)
# 7. [Limits & when 2D EIM breaks down](#7.-Limits-&-when-2D-EIM-breaks-down)
# 8. [Exercises](#8.-Exercises)
# 9. [References](#9.-References)

# %% [markdown]
# ## 1. Physical background & device specification
#
# A $2 \times 2$ MMI coupler consists of two input waveguides (Port 1 upper-left,
# Port 2 lower-left), a central multi-mode waveguide section of width
# $w_{\text{mmi}}$ and length $l_{\text{mmi}}$ supporting $M \ge 3$ lateral
# modes $\psi_\nu(y)\exp(-i\beta_\nu x)$ ($\nu = 0, 1, \dots, M-1$), and two
# output waveguides (Port 3 upper-right bar port, Port 4 lower-right cross port).
#
# Following **Soldano & Pennings (1995)** [1] and **Ulrich & Ankele (1975)** [2],
# the lateral penetration depth into the cladding defines an effective MMI width
# $W_e$ for the 3D quasi-TE (in-plane TM, $\sigma = 1$) polarization:
#
# $$W_e = w_{\text{mmi}} + \frac{\lambda_0}{\pi}\left(\frac{n_{\text{clad}}}{n_{\text{eff,slab}}}\right)^{2\sigma}\left(n_{\text{eff,slab}}^2 - n_{\text{clad}}^2\right)^{-1/2}, \qquad \sigma = 1.$$
#
# In the paraxial approximation, the propagation constants $\beta_\nu$ of the
# lateral modes obey a quadratic spacing law:
#
# $$\beta_0 - \beta_\nu \approx \frac{\nu(\nu + 2)\pi}{3 L_\pi}, \qquad L_\pi = \frac{\pi}{\beta_0 - \beta_1} \approx \frac{4 n_{\text{eff,slab}} W_e^2}{3 \lambda_0}.$$
#
# Under **general interference** (no symmetry restriction on the input excitation
# position $y = \pm y_{\text{attach}}$), a single input image at $+y_{\text{attach}}$
# forms a mirrored **cross image** at $-y_{\text{attach}}$ at $L = L_\pi$ (where
# odd modes acquire a $\pi$ phase shift relative to even modes) and splits into
# **two equal-amplitude ($3\;\text{dB}$) images** at $+y_{\text{attach}}$ (bar,
# Port 3) and $-y_{\text{attach}}$ (cross, Port 4) at:
#
# $$L_{3\text{dB}} = \frac{3}{2} L_\pi \quad \left(\text{and at the first half-beat } L = \frac{1}{2} L_\pi\right).$$
#
# At $L = 3 L_\pi / 2$, the even modes ($\nu = 0, 2, \dots$) have paraxial phase
# $\nu(\nu+2)\pi / 2 \equiv 0 \pmod{2\pi}$, whereas the odd modes ($\nu = 1, 3, \dots$)
# have phase $\nu(\nu+2)\pi / 2 \equiv 3\pi/2 \equiv -\pi/2 \pmod{2\pi}$. Decomposing
# the input field into symmetric and antisymmetric parts $E_{\text{in}}(y) = E_s(y) + E_a(y)$
# yields at the output face:
#
# $$E(y, 3L_\pi/2) = E_s(y) + i\,E_a(y) = \frac{1 + i}{2} E_{\text{in}}(y) + \frac{1 - i}{2} E_{\text{in}}(-y).$$
#
# Consequently, the bar ($S_{31}$) and cross ($S_{41}$) outputs carry equal power
# ($|S_{31}|^2 = |S_{41}|^2 \le 1/2$) and are in **quadrature** with a relative
# phase difference of:
#
# $$\Delta\phi = \arg(S_{31}) - \arg(S_{41}) = \pm 90^\circ.$$

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
from scipy.constants import c as c_light
from scipy.optimize import brentq

from wakis import figstyle
from wakis.templates.photonics import (
    PICSimulation,
    PolySlab,
    Port,
    make_movie,
    n_to_eps,
    plot_eim_stack,
    plot_field_map,
    plot_layout,
    plot_neff_dispersion,
    plot_port_modes,
    sbend,
    slab_neff,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _helpers import (  # noqa: E402
    RESOLUTION_LADDER,
    THROAT_OVERRIDE_X,
    NonUniformMMI2x2PIC,
    build_gap_mesh_overrides,
    build_mmi_2x2_graded_axes,
    build_task_grid_only,
    extract_zoom_permittivity,
    measure_grid_resolution,
    plot_mesh_upgrade_comparison_mmi_2x2,
    plot_throat_gap_mesh_zoom,
    render_mmi_2x2_field_comparison_with_overlay,
    render_mmi_2x2_field_map_with_overlay,
    render_mmi_2x2_movie_with_overlay,
    save_mmi_2x2_setup_figures,
)

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# %% [markdown]
# ### Device parameters
#
# All geometric dimensions are specified in metres (SI units) for the
# $220\;\text{nm}$ SOI $2 \times 2$ MMI splitter layout:
#
# | Parameter | Symbol | Value |
# |---|---|---|
# | Silicon core refractive index ($1550\;\text{nm}$) | $n_{\text{Si}}$ | $3.48$ ($\varepsilon_r = 12.1104$) |
# | $\text{SiO}_2$ cladding refractive index ($1550\;\text{nm}$) | $n_{\text{SiO}_2}$ | $1.44$ ($\varepsilon_r = 2.0736$) |
# | Silicon slab thickness | $h_{\text{Si}}$ | $220\;\text{nm}$ ($z \in [-110, +110]\;\text{nm}$) |
# | MMI box length / width | $l_{\text{mmi}}, w_{\text{mmi}}$ | $3.0\;\mu\text{m}, 1.0\;\mu\text{m}$ |
# | Access waveguide width | $w_{\text{wg}}$ | $0.40\;\mu\text{m}$ |
# | Inner gap between access waveguides at MMI face | $g_{\text{mmi}}$ | $w_{\text{mmi}} - 2w_{\text{wg}} = 0.20\;\mu\text{m}$ |
# | Access waveguide attachment centers at $x = \pm l_{\text{mmi}}/2$ | $\pm y_{\text{attach}}$ | $\pm 0.30\;\mu\text{m}$ |
# | Cosine S-bend axial length / lateral offset | $L_{\text{sb}}, \pm\Delta y_{\text{sb}}$ | $4.0\;\mu\text{m}, \pm 1.0\;\mu\text{m}$ |
# | Input / output port waveguide centers | $\pm y_{\text{port}}$ | $\pm 1.30\;\mu\text{m}$ at $x = \pm 6.0\;\mu\text{m}$ |
# | Simulation domain ($L_x \times L_y$) | $\Omega_{\text{2D}}$ | $[-6.5, +6.5] \times [-2.5, +2.5]\;\mu\text{m}^2$ |
#
# ### Simulation setup schematic & discretization
#
# Before launching time-domain runs, we inspect the labelled simulation setup
# generated directly from the solver objects via `wakis.schematic.save_setup_figures`.
# Figure `00a_setup_geometry.png` shows the $2 \times 2$ SOI MMI polygon
# ($n_{\text{eff,slab}} = 2.85$, $\varepsilon_{\text{eff}} = 8.13$) embedded in
# $\text{SiO}_2$ cladding ($n = 1.44$, $\varepsilon_r = 2.07$), the 10-cell CPML
# absorbing boundaries on all four in-plane faces, the Port 1 (`in1`) `ModeSource`
# at $x = -6.12\;\mu\text{m}$ with the launched fundamental $\text{TE}_0$ mode profile
# $|E_y(y)|$ overlaid on the port plane, the four port `ModeMonitors` (`in1`/`in2`
# at $x = -6.00\;\mu\text{m}$ and `out1` bar / `out2` cross at $x = +6.00\;\mu\text{m}$),
# and the closed 2D Poynting `FluxBox`. Figure `00b_setup_mesh_and_excitation.png`
# displays the graded non-uniform, non-square Cartesian mesh with a $1\times$ zoom
# inset at the $200\;\text{nm}$ S-bend throat gap ($8.0\;\text{cells}$ across the
# gap, $\Delta_{\min} = 25\;\text{nm}$, $21.7\;\text{cells}/\lambda$), the spatial
# wave-packet envelope, and the broadband Gaussian pulse and spectrum.
#
# ![2x2 MMI setup schematic](figures/00a_setup_geometry.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 13.8 µm × 6.39 µm × 26.3 nm (z ∈ [-13.2 nm, 13.2 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 424 × 170 × 1 (72080 interior cells; 425 × 171 × 1 = 72,675 FIT cells) |
# | Discretization | Δx & resolution | Δ_min = 25 nm, 21.7 cells/λ (in n_max = 2.85), 8.0 cells across feature (200 nm) |
# | Boundary | -x face | CPML (10 cells, 378 nm) |
# | Boundary | +x face | CPML (10 cells, 378 nm) |
# | Boundary | -y face | CPML (10 cells, 697 nm) |
# | Boundary | +y face | CPML (10 cells, 697 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | SiO₂ cladding (n = 1.44, ε_r = 2.07) (isotropic) |
# | Material | Region 1: SOI 220 nm EIM (n_eff = 2.85) | ε_r = 8.13, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-7.50 µm, 7.50 µm] (d = 15 µm) |
# | Source | 1 · ModeSource | Port 1 'in1' at x = -6.12 µm, y = +1.30 µm, pol: Ey (TE₀/EIM, +x), pulse: f₀ = 193.6 THz, Δf = 6.2 THz |
# | Monitor | 2 · ModeMonitor | Ports 'in1'/'in2' ModeMonitors (S₁₁, S₂₁) at x = -6.00 µm |
# | Monitor | 3 · ModeMonitor | Ports 'out1' (bar) / 'out2' (cross) ModeMonitors (S₃₁, S₄₁) at x = +6.00 µm |
# | Monitor | 4 · FluxBox | Closed 2D Poynting FluxBox (4 faces inside CPML) |
# | Variation | kottke_L1..L3 / staircase_L1..L3 (tasks 0–5) | param=L1=(10 cpw, N=4), L2=(14, 6), L3=(20, 8), note=Resolution ladder with throat-gap mesh_overrides at L_mmi = 3.00 µm |
# | Variation | kottke_reference (task 6) | param=R=(40 cpw, N=16) (1016 × 403 × 1 = 409,448 cells), note=Independent 2× fine Kottke reference |
# | Variation | length_*um (tasks 7–10) | param=L_mmi ∈ {2.50, 3.00, 3.25, 3.50, 3.80} µm at L3=(20, 8), note=General-interference sweep (2D EIM 50/50 split at L_mmi = 3.25 µm) |
# | Variation | kottke_nonuniform (task 11) | param=Graded non-square Cartesian mesh (425 × 171 × 1 = 72,675 cells), note=Exact nodes at x = ±1.50 µm, y = ±0.10, ±0.50 µm; 8.0 cells across 200 nm gap |
#
# ![2x2 MMI mesh and excitation](figures/00b_setup_mesh_and_excitation.png)

# %%
# Material refractive indices (non-dispersive benchmark values)
N_SI = 3.48
N_SIO2 = 1.44
H_SI = 220e-9

# Wavelength band: 1500 nm to 1600 nm (centered at 1550 nm; 1580 nm secondary DFT monitor)
WL_MIN = 1500e-9
WL_MAX = 1600e-9
WL_REF = 1550e-9
WL_5050_3D = 1580e-9
N_FREQS = 51

# Geometry parameters [m]
L_MMI = 3.00e-6
W_MMI = 1.00e-6
W_WG = 0.40e-6
GAP_MMI = W_MMI - 2.0 * W_WG  # 0.20 um
Y_ATTACH = 0.5 * (W_MMI - W_WG)  # 0.30 um
L_SBEND = 4.00e-6
DY_SBEND = 1.00e-6
Y_PORT = Y_ATTACH + DY_SBEND  # 1.30 um
L_STRAIGHT_EXT = 2.00e-6

# Nominal 2D domain and port coordinates for L_MMI = 3.00 um
DOMAIN_2D = (-6.50e-6, 6.50e-6, -2.50e-6, 2.50e-6)
X_PORT = 6.00e-6

# MMI box lengths [um] for the general-interference length sweep
L_MMI_SWEEP_UM = (2.50, 3.00, 3.25, 3.50, 3.80)

# %% [markdown]
# ## 2. Effective-Index Method (EIM) reduction
#
# In the **2D effective-index method (EIM)**, the 3D quasi-TE electric field is
# factored as $\mathbf{E}(x, y, z) \approx \psi(z)\,\mathbf{E}_{\parallel}(x, y)$
# where $\psi(z)$ is the fundamental $\text{TE}_0$ slab mode of the symmetric
# $\text{SiO}_2 / \text{Si}(220\;\text{nm}) / \text{SiO}_2$ vertical stack at
# $\lambda_0 = 1550\;\text{nm}$:
#
# $$n_{\text{eff,slab}}(1550\;\text{nm}) = 2.8514 \quad (n_{\text{Si}} = 3.48,\; n_{\text{SiO}_2} = 1.44).$$
#
# The 3D quasi-TE problem maps onto a 2D in-plane FIT simulation with components
# $(E_x, E_y, H_z)$, one periodic cell along $z$ (`ieps_z = 0`), core index
# $n_{\text{eff,slab}} = 2.8514$, and cladding index $n_{\text{SiO}_2} = 1.44$.
#
# ## 3. Parametric geometry & lateral MMI eigenmodes
#
# We join the rectangular MMI box ($x \in [-l_{\text{mmi}}/2, +l_{\text{mmi}}/2]$,
# $y \in [-w_{\text{mmi}}/2, +w_{\text{mmi}}/2]$), the four symmetric cosine
# S-bends (`sbend`), and the four straight access waveguides into a single
# counter-clockwise `PolySlab`.


# %%
def build_mmi_2x2(l_mmi=L_MMI, n_sbend=80):
    """
    Build the 2x2 MMI power splitter as a single seamless `PolySlab` along with
    its four named waveguide `Port` specifications (`in1`, `in2`, `out1`, `out2`)
    and 2D simulation domain `(xmin, xmax, ymin, ymax)` [m].

    Parameters
    ----------
    l_mmi : float, optional
        Length of the central multimode interference box [m].
    n_sbend : int, optional
        Number of vertices along each cosine S-bend edge.

    Returns
    -------
    structures : list of PolySlab
        Single-element list containing the extruded silicon `PolySlab`.
    ports : list of Port
        Four waveguide ports: `in1` (port 1, upper-left), `in2` (port 2,
        lower-left), `out1` (port 3, upper-right bar), `out2` (port 4,
        lower-right cross).
    domain : tuple of float
        `(xmin, xmax, ymin, ymax)` domain bounds [m].
    """
    xh = 0.5 * float(l_mmi)
    x_sb = xh + L_SBEND
    x_end = x_sb + L_STRAIGHT_EXT
    hw = 0.5 * W_WG
    hm = 0.5 * W_MMI

    sb_ul = sbend(
        (-x_sb, +Y_PORT),
        dx=L_SBEND,
        dy=-DY_SBEND,
        width=W_WG,
        profile="cosine",
        n=n_sbend,
    )
    sb_ll = sbend(
        (-x_sb, -Y_PORT),
        dx=L_SBEND,
        dy=+DY_SBEND,
        width=W_WG,
        profile="cosine",
        n=n_sbend,
    )
    sb_ur = sbend(
        (+xh, +Y_ATTACH),
        dx=L_SBEND,
        dy=+DY_SBEND,
        width=W_WG,
        profile="cosine",
        n=n_sbend,
    )
    sb_lr = sbend(
        (+xh, -Y_ATTACH),
        dx=L_SBEND,
        dy=-DY_SBEND,
        width=W_WG,
        profile="cosine",
        n=n_sbend,
    )

    verts = np.vstack(
        [
            [[-x_end, -Y_PORT - hw]],
            sb_ll[:n_sbend],
            [[+xh, -hm]],
            sb_lr[:n_sbend][1:],
            [[+x_end, -Y_PORT - hw], [+x_end, -Y_PORT + hw]],
            sb_lr[n_sbend:],
            [[+xh, -Y_ATTACH + hw], [+xh, +Y_ATTACH - hw]],
            sb_ur[:n_sbend][1:],
            [[+x_end, +Y_PORT - hw], [+x_end, +Y_PORT + hw]],
            sb_ur[n_sbend:],
            [[-xh, +hm]],
            sb_ul[n_sbend:][1:],
            [[-x_end, +Y_PORT + hw], [-x_end, +Y_PORT - hw]],
            sb_ul[:n_sbend],
            [[-xh, +Y_ATTACH - hw], [-xh, -Y_ATTACH + hw]],
            sb_ll[n_sbend:][1:],
            [[-x_end, -Y_PORT + hw]],
        ]
    )
    slab = PolySlab(
        verts,
        z_min=-0.5 * H_SI,
        z_max=0.5 * H_SI,
        material=n_to_eps(N_SI),
    )

    x_p = x_sb + 0.50e-6
    port_size = (0.0, 1.80e-6, 1.32e-6)
    ports = [
        Port("in1", center=(-x_p, +Y_PORT, 0.0), size=port_size, direction=+1),
        Port("in2", center=(-x_p, -Y_PORT, 0.0), size=port_size, direction=+1),
        Port("out1", center=(+x_p, +Y_PORT, 0.0), size=port_size, direction=+1),
        Port("out2", center=(+x_p, -Y_PORT, 0.0), size=port_size, direction=+1),
    ]
    domain = (-x_p - 0.50e-6, +x_p + 0.50e-6, -2.50e-6, 2.50e-6)
    return [slab], ports, domain


def solve_lateral_slab_mode(y, width, m=0, center=0.0, wavelength=WL_REF):
    """
    Solve the 1D in-plane TM (3D quasi-TE, sigma=1) lateral slab eigenmode `m`
    for a strip of `width` [m] centered at `center` [m] in 2D EIM.

    Returns `(neff, hz_norm, eps_profile)` normalized such that
    `integral |Hz(y)|^2 / eps(y) dy = 1`.
    """
    neff_s = slab_neff(N_SI, N_SIO2, H_SI, wavelength, polarization="TE")
    k0 = 2.0 * np.pi / float(wavelength)
    ratio = (neff_s / N_SIO2) ** 2

    def disp(neff):
        kappa = k0 * np.sqrt(max(neff_s**2 - neff**2, 0.0))
        gamma = k0 * np.sqrt(max(neff**2 - N_SIO2**2, 0.0))
        return kappa * width - m * np.pi - 2.0 * np.arctan2(ratio * gamma, kappa)

    neff = float(brentq(disp, N_SIO2 + 1e-9, neff_s - 1e-9))
    kappa = k0 * np.sqrt(neff_s**2 - neff**2)
    gamma = k0 * np.sqrt(neff**2 - N_SIO2**2)
    yc = y - float(center)
    hw = 0.5 * float(width)
    hz = np.empty_like(y)
    inside, above, below = np.abs(yc) <= hw, yc > hw, yc < -hw
    if m % 2 == 0:
        hz[inside] = np.cos(kappa * yc[inside])
        hz[above] = np.cos(kappa * hw) * np.exp(-gamma * (yc[above] - hw))
        hz[below] = np.cos(kappa * hw) * np.exp(gamma * (yc[below] + hw))
    else:
        hz[inside] = np.sin(kappa * yc[inside])
        hz[above] = np.sin(kappa * hw) * np.exp(-gamma * (yc[above] - hw))
        hz[below] = -np.sin(kappa * hw) * np.exp(gamma * (yc[below] + hw))
    eps_prof = np.where(inside, neff_s**2, N_SIO2**2)
    norm = np.sqrt(np.trapezoid(hz**2 / eps_prof, y))
    hz /= max(float(norm), 1e-30)
    return neff, hz, eps_prof


def analytic_mmi_model(wavelength=WL_REF):
    """
    Compute the Soldano & Pennings (1995) analytical MMI parameters and modal
    decomposition for the 2x2 MMI coupler in 2D EIM at `wavelength` [m].
    """
    neff_s = slab_neff(N_SI, N_SIO2, H_SI, wavelength, polarization="TE")
    k0 = 2.0 * np.pi / float(wavelength)
    # Soldano & Pennings (1995) effective width We (sigma = 1 for quasi-TE)
    w_eff = W_MMI + (wavelength / np.pi) * (N_SIO2 / neff_s) ** 2 / np.sqrt(
        neff_s**2 - N_SIO2**2
    )
    l_pi_sp = 4.0 * neff_s * w_eff**2 / (3.0 * wavelength)

    y = np.linspace(-3.0e-6, 3.0e-6, 3001)
    neff_wg, hz_bar, _ = solve_lateral_slab_mode(
        y, W_WG, m=0, center=+Y_ATTACH, wavelength=wavelength
    )
    _, hz_cross, _ = solve_lateral_slab_mode(
        y, W_WG, m=0, center=-Y_ATTACH, wavelength=wavelength
    )

    modes = []
    for m in range(4):
        try:
            neff_m, hz_m, eps_m = solve_lateral_slab_mode(
                y, W_MMI, m=m, center=0.0, wavelength=wavelength
            )
            modes.append((neff_m, hz_m, eps_m))
        except ValueError:
            break

    eps_mmi = modes[0][2]
    neffs = np.array([m[0] for m in modes])
    betas = k0 * neffs
    profiles = np.array([m[1] for m in modes])
    c_bar = np.array([np.trapezoid(hz_bar * m[1] / eps_mmi, y) for m in modes])
    c_cross = np.array([np.trapezoid(hz_cross * m[1] / eps_mmi, y) for m in modes])
    l_pi_exact = float(np.pi / (betas[0] - betas[1]))

    # Coupled-waveguide supermode pre/post-coupling along the cosine S-bends
    ratio = (neff_s / N_SIO2) ** 2

    def _supermode_neff(gap, parity):
        def _disp(neff):
            k = k0 * np.sqrt(max(neff_s**2 - neff**2, 0.0))
            gm = k0 * np.sqrt(max(neff**2 - N_SIO2**2, 0.0))
            th = (
                np.tanh(0.5 * gm * gap)
                if parity == "even"
                else 1.0 / np.tanh(max(0.5 * gm * gap, 1e-12))
            )
            return k * W_WG - np.arctan2(ratio * gm, k) - np.arctan2(ratio * gm * th, k)

        return float(brentq(_disp, N_SIO2 + 1e-9, neff_s - 1e-9))

    u = np.linspace(0.0, 1.0, 61)
    yc_sb = Y_ATTACH + DY_SBEND * 0.5 * (1.0 - np.cos(np.pi * u))
    gaps = 2.0 * yc_sb - W_WG
    dbeta_sb = np.array(
        [
            k0 * (_supermode_neff(g, "even") - _supermode_neff(g, "odd"))
            for g in gaps
        ]
    )
    dphi_sb_total = 2.0 * float(np.trapezoid(dbeta_sb, u * L_SBEND))
    dl_sb = dphi_sb_total / float(betas[0] - betas[1])

    return {
        "y": y,
        "neff_slab": neff_s,
        "neff_wg": neff_wg,
        "w_eff": float(w_eff),
        "l_pi_sp": float(l_pi_sp),
        "l_pi_exact": l_pi_exact,
        "dl_sb": float(dl_sb),
        "dphi_sb_deg": float(np.rad2deg(dphi_sb_total)),
        "neffs": neffs,
        "betas": betas,
        "profiles": profiles,
        "eps_mmi": eps_mmi,
        "c_bar": c_bar,
        "c_cross": c_cross,
    }


# %% [markdown]
# ## 4. Subpixel permittivity smoothing & resolution ladder
#
# In the four $4.0\;\mu\text{m}$ cosine S-bends, the waveguide sidewalls cross the
# Cartesian Yee grid at continuously varying angles. Rounding each cell's fill
# fraction $w \in [0, 1]$ to $0$ or $1$ (**staircasing**) creates right-angle step
# discontinuities that scatter guided light into unguided cladding radiation and
# spurious back-reflections ($|S_{11}|^2, |S_{21}|^2$), while shifting the
# accumulated phase $\int \beta\,\mathrm{d}x$ over the $13\;\mu\text{m}$ device.
#
# Wakis applies the diagonal **Kottke, Farjadpour & Johnson (2008)** [3] subpixel
# permittivity tensor on every cut cell via `PolySlab.fill` and `PolySlab.normal`:
#
# $$\varepsilon_{\text{eff}, c}^{-1} = n_c^2 \left(\frac{w}{\varepsilon_s} + \frac{1 - w}{\varepsilon_b}\right) + \left(1 - n_c^2\right)\left(w\varepsilon_s + (1 - w)\varepsilon_b\right)^{-1}.$$
#
# ### Resolution: cells per wavelength and cells across the smallest feature
#
# Every time-domain FIT simulation in this tutorial — including `--quick` and
# parameter sweeps — enforces two strict resolution rules simultaneously:
#
# 1. **Wavelength sampling ($\text{cpw} \ge 10$)**: The background grid spacing in
#    the highest-index slab region ($n_{\text{eff,slab}} = 2.8514$) satisfies
#    $\Delta x_{\text{bg}} \le \lambda_{\min} / (n_{\text{eff,slab}} \cdot \text{cpw})$
#    with at least $10$ cells per guided wavelength ($\Delta x_{\text{bg}} \approx 52.6\;\text{nm}$
#    at $10\;\text{cpw}$ and $\lambda_{\min} = 1500\;\text{nm}$).
# 2. **Geometric feature sampling ($N \ge 4$ cells across the smallest feature)**:
#    The smallest feature of the $2 \times 2$ MMI coupler is the
#    $g_{\text{mmi}} = w_{\text{mmi}} - 2w_{\text{wg}} = 0.20\;\mu\text{m}$ ($200\;\text{nm}$)
#    oxide throat gap between the two access waveguides at each MMI face
#    ($x = \pm l_{\text{mmi}}/2$, $y \in [-0.10, +0.10]\;\mu\text{m}$) and the
#    converging S-bend throat where they approach it. On an unrefined $10\;\text{cpw}$
#    mesh ($\Delta y \approx 52.6\;\text{nm}$), the $200\;\text{nm}$ gap spans only
#    $\approx 3.8$ cells — just under the $N \ge 4$ rule — and its $y = \pm 0.10\;\mu\text{m}$
#    corners do not coincide with grid nodes.
#
# To guarantee at least $N$ cells across the $200\;\text{nm}$ gap without paying for a
# uniform fine grid across the entire $13 \times 5\;\mu\text{m}^2$ domain, we pass two
# local **mesh-override boxes** `mesh_overrides=[(lo, hi, (h_max, h_max, 1.0)), ...]`
# covering the two S-bend throat gap regions ($x \in [-l_{\text{mmi}}/2 - 0.60\;\mu\text{m}, -l_{\text{mmi}}/2]$
# and $x \in [+l_{\text{mmi}}/2, +l_{\text{mmi}}/2 + 0.60\;\mu\text{m}]$,
# $y \in [-0.10, +0.10]\;\mu\text{m}$) with cell cap $h_{\max} = g_{\text{mmi}} / N$ in
# $x$ and $y$ (keeping $h_z = 1.0\;\text{m}$ so the single periodic 2D EIM $z$-cell is
# unchanged). Because the FIT mesh is a tensor product, each override box snaps nodes
# onto the MMI faces $x = \pm l_{\text{mmi}}/2$ and gap edges $y = \pm 0.10\;\mu\text{m}$,
# caps the cell size across the throat strip, and grades smoothly back to
# $\Delta x_{\text{bg}}$ at ratio $\le 1.4$.
#
# Each resolution level is specified as a pair $(\text{cpw}, N)$:
#
# - **L1** $= (10, 4)$: $\Delta x_{\text{bg}} = 52.38\;\text{nm}$, $\Delta x_{\min} = 50.00\;\text{nm}$, $4.00$ cells across the $200\;\text{nm}$ gap.
# - **L2** $= (14, 6)$: $\Delta x_{\text{bg}} = 37.52\;\text{nm}$, $\Delta x_{\min} = 33.33\;\text{nm}$, $6.00$ cells across the $200\;\text{nm}$ gap.
# - **L3** $= (20, 8)$: $\Delta x_{\text{bg}} = 26.19\;\text{nm}$, $\Delta x_{\min} = 25.00\;\text{nm}$, $8.00$ cells across the $200\;\text{nm}$ gap.
# - **R** $= (40, 16)$: $\Delta x_{\text{bg}} = 13.14\;\text{nm}$, $\Delta x_{\min} = 12.50\;\text{nm}$, $16.00$ cells across the $200\;\text{nm}$ gap.
#
# ### Graded non-uniform, non-square Cartesian mesh upgrade (`kottke_nonuniform`)
#
# Even with throat-gap `mesh_overrides`, the baseline **L3** $= (20, 8)$ grid
# (`kottke_L3`, $520 \times 213 \times 1 = 110{,}760$ cells) uses near-uniform
# $\Delta x \approx \Delta y \approx 26.2\;\text{nm}$ cells throughout the outer
# $\text{SiO}_2$ cladding ($|y| > 1.50\;\mu\text{m}$) and along the $4.0\;\mu\text{m}$
# S-bends and straight port leads, and its transverse grid lines miss the exact MMI
# sidewalls $y = \pm 0.50\;\mu\text{m}$ by $7.1\;\text{nm}$ ($8.8\;\text{nm}$ in 8-cell PML).
# In `build_mmi_2x2_graded_axes` and `NonUniformMMI2x2PIC` (`task_11`,
# `kottke_nonuniform`), we upgrade to a mirror-symmetric **graded non-uniform,
# non-square Cartesian mesh**:
#
# - **Exact feature & sidewall alignment**: exact transverse grid nodes at
#   $y = 0$, $y = \pm 0.10\;\mu\text{m}$ ($200\;\text{nm}$ throat gap edges),
#   $y = \pm 0.50\;\mu\text{m}$ (MMI sidewalls, **zero sidewall rounding error**
#   $\text{wall err} = 0.0\;\text{nm}$), and $y = \pm 1.50\;\mu\text{m}$ (outer
#   access waveguide edges), with exact $N = 8.00$ cells ($\Delta y = 25.0\;\text{nm}$)
#   across the $200\;\text{nm}$ gap and MMI core.
# - **Coarse cladding grading**: outside the S-bend corridor ($|y| \in [1.50, 2.50]\;\mu\text{m}$),
#   $\Delta y$ grades smoothly at ratio $\le 1.22$ up to $\Delta y_{\max} = 69.7\;\text{nm}$
#   in the $\text{SiO}_2$ cladding where only evanescent tails exist.
# - **Non-square axial cells along S-bends and leads**: along $x$, exact grid
#   nodes at $x = \pm 1.50\;\mu\text{m}$ and $x = \pm 2.30\;\mu\text{m}$ keep
#   $\Delta x = 25.0\;\text{nm}$ across the MMI cavity and throat gap, then grade
#   smoothly to stretched non-square axial cells $\Delta x_{\max} = 37.7\;\text{nm}$
#   ($\Delta x / \Delta y_{\min} = 1.51$) along the gently curving cosine S-bends and
#   straight port leads.
#
# This reduces the 2D grid from $520 \times 213 = 110{,}760$ cells (`kottke_L3`) to
# $425 \times 171 = 72{,}675$ cells (**$-34.4\%$ fewer cells**) and cuts wall-clock
# runtime from $231.99\;\text{s}$ to $134.98\;\text{s}$ (**$-41.8\%$ wall time**),
# while simultaneously reducing the broadband RMS transmission error against the
# independent **R** $= (40, 16)$ reference from $5.02 \times 10^{-3}$ to
# $3.14 \times 10^{-3}$ (**$1.60\times$ lower error**) and the 2-port RMS error from
# $1.24 \times 10^{-2}$ to $6.63 \times 10^{-3}$ (**$1.87\times$ lower error**).
#
# ![Mesh upgrade comparison](figures/04b_mesh_upgrade_comparison.png)

# %% [markdown]
# ## 5. Study tasks & execution modes
#
# - In `--quick` (**10 tasks**, `task_0`..`task_9`): Tasks `0..3` compare **Kottke** and
#   **staircase** averaging at **L1** $= (10, 4)$ and **L2** $= (14, 6)$, where Task `2`
#   (`kottke_L2`) is the primary run recording steady-state DFT fields at
#   $\lambda = 1550\;\text{nm}$ and $1580\;\text{nm}$, the pulse propagation movie,
#   port eigenmode dispersion, and the 2D Poynting flux box. Task `4`
#   (`kottke_reference`) is the independent Kottke reference run at **L3** $= (20, 8)$,
#   Tasks `5..8` sweep the MMI box length over
#   $L_{\text{mmi}} \in \{2.50, 3.25, 3.50, 3.80\}\;\mu\text{m}$ at **L2** $= (14, 6)$
#   (combined with $L_{\text{mmi}} = 3.00\;\mu\text{m}$ from `kottke_L2`), and Task `9`
#   (`kottke_nonuniform`) runs the graded non-uniform, non-square mesh at **L2** $= (14, 6)$
#   (`energy_decay = 5e-3`, `n_pml = 8`).
# - In `--full` (**12 tasks**, `task_0`..`task_11`): Tasks `0..5` compare **Kottke** and
#   **staircase** averaging across **L1** $= (10, 4)$, **L2** $= (14, 6)$, and
#   **L3** $= (20, 8)$ (with `kottke_L3` as the primary field/movie/flux run), Task `6`
#   (`kottke_reference`) runs the independent reference at **R** $= (40, 16)$, Tasks
#   `7..10` run the MMI length sweep at **L3** $= (20, 8)$, and Task `11`
#   (`kottke_nonuniform`) runs the graded non-uniform, non-square Cartesian mesh at
#   **L3** $= (20, 8)$. The **R** $= (40, 16)$ reference run (`409,448` cells, `27,951`
#   steps) completes in $1{,}410.36\;\text{s}$ ($23.51\;\text{min}$, peak RSS
#   $1{,}271.34\;\text{MB}$) on a single CPU core.
#
# ### Measured `--full` resolution & runtime table (`energy_decay = 1e-4`, `n_pml = 10`)
#
# Every run in `data/task_*.json` satisfies $\text{cpw} \ge 10$ in the highest-index
# slab material ($n_{\text{eff,slab}} = 2.8514$) and $N = \text{cells\_across\_feature} \ge 4$
# across the $200\;\text{nm}$ throat gap (zero resolution violations):
#
# | Task | Name | Level | Averaging | $\text{cpw}$ | $N$ | $\Delta x_{\text{bg}}\;[\text{nm}]$ | $\Delta x_{\min}\;[\text{nm}]$ | Cells across $200\;\text{nm}$ gap | Grid shape | $n_{\text{cells}}$ | Steps | Time $[\text{s}]$ | Peak RSS $[\text{MB}]$ |
# |---:|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
# | 0 | `kottke_L1` | L1 | Kottke | 10 | 4 | 52.38 | 50.00 | 4.00 | $271 \times 117 \times 1$ | 31,707 | 7,201 | 33.93 | 618.88 |
# | 1 | `staircase_L1` | L1 | Staircase | 10 | 4 | 52.38 | 50.00 | 4.00 | $271 \times 117 \times 1$ | 31,707 | 8,801 | 36.80 | 624.16 |
# | 2 | `kottke_L2` | L2 | Kottke | 14 | 6 | 37.52 | 33.33 | 6.00 | $373 \times 155 \times 1$ | 57,815 | 10,401 | 72.12 | 680.77 |
# | 3 | `staircase_L2` | L2 | Staircase | 14 | 6 | 37.52 | 33.33 | 6.00 | $373 \times 155 \times 1$ | 57,815 | 12,451 | 100.36 | 680.57 |
# | 4 | `kottke_L3` | L3 (primary) | Kottke | 20 | 8 | 26.19 | 25.00 | 8.00 | $520 \times 213 \times 1$ | 110,760 | 14,051 | 231.99 | 772.00 |
# | 5 | `staircase_L3` | L3 | Staircase | 20 | 8 | 26.19 | 25.00 | 8.00 | $520 \times 213 \times 1$ | 110,760 | 15,101 | 207.24 | 770.01 |
# | 6 | `kottke_reference` | reference (R) | Kottke | 40 | 16 | 13.14 | 12.50 | 16.00 | $1016 \times 403 \times 1$ | 409,448 | 27,951 | 1,410.36 | 1,271.34 |
# | 7 | `length_2p50um` | sweep (L3) | Kottke | 20 | 8 | 26.19 | 25.00 | 8.00 | $501 \times 213 \times 1$ | 106,713 | 13,701 | 193.36 | 769.08 |
# | 8 | `length_3p25um` | sweep (L3) | Kottke | 20 | 8 | 26.21 | 25.00 | 8.00 | $529 \times 213 \times 1$ | 112,677 | 14,051 | 191.10 | 776.33 |
# | 9 | `length_3p50um` | sweep (L3) | Kottke | 20 | 8 | 26.19 | 25.00 | 8.00 | $539 \times 213 \times 1$ | 114,807 | 14,001 | 202.08 | 776.83 |
# | 10 | `length_3p80um` | sweep (L3) | Kottke | 20 | 8 | 26.21 | 25.00 | 8.00 | $550 \times 213 \times 1$ | 117,150 | 14,401 | 215.02 | 784.02 |
# | 11 | `kottke_nonuniform` | nonuniform (L3) | Kottke | 20 | 8 | 69.70 | 25.00 | 8.00 | $425 \times 171 \times 1$ | 72,675 | 14,151 | 134.98 | 694.30 |
#
# ### Measured `--quick` resolution & runtime table (`energy_decay = 5e-3`, `n_pml = 8`)
#
# | Task | Name | Level | Averaging | $\text{cpw}$ | $N$ | $\Delta x_{\text{bg}}\;[\text{nm}]$ | $\Delta x_{\min}\;[\text{nm}]$ | Cells across $200\;\text{nm}$ gap | $n_{\text{cells}}$ | Steps | Time $[\text{s}]$ |
# |---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
# | 0 | `kottke_L1` | L1 | Kottke | 10 | 4 | 52.38 | 50.00 | 4.00 | 30,171 | 6,551 | 29.81 |
# | 1 | `staircase_L1` | L1 | Staircase | 10 | 4 | 52.38 | 50.00 | 4.00 | 30,171 | 6,951 | 31.83 |
# | 2 | `kottke_L2` | L2 | Kottke | 14 | 6 | 37.52 | 33.33 | 6.00 | 55,719 | 9,551 | 106.97 |
# | 3 | `staircase_L2` | L2 | Staircase | 14 | 6 | 37.52 | 33.33 | 6.00 | 55,719 | 9,701 | 79.87 |
# | 4 | `kottke_reference` | reference (L3) | Kottke | 20 | 8 | 26.19 | 25.00 | 8.00 | 107,844 | 12,901 | 191.74 |
# | 5 | `length_2p50um` | sweep (L2) | Kottke | 14 | 6 | 37.51 | 33.33 | 6.00 | 53,756 | 9,351 | 77.97 |
# | 6 | `length_3p25um` | sweep (L2) | Kottke | 14 | 6 | 37.51 | 33.33 | 6.00 | 56,776 | 9,601 | 79.56 |
# | 7 | `length_3p50um` | sweep (L2) | Kottke | 14 | 6 | 37.51 | 33.33 | 6.00 | 57,833 | 9,651 | 81.92 |
# | 8 | `length_3p80um` | sweep (L2) | Kottke | 14 | 6 | 37.51 | 33.33 | 6.00 | 59,041 | 9,801 | 82.89 |
# | 9 | `kottke_nonuniform` | nonuniform (L2) | Kottke | 14 | 6 | 99.57 | 33.33 | 6.00 | 38,520 | 9,601 | 52.40 |
#
# ### Key quantitative results (`--full` study)
#
# - **Nominal $L_{\text{mmi}} = 3.00\;\mu\text{m}$ splitter at $\lambda_0 = 1550\;\text{nm}$**:
#   - `kottke_reference` (**R** $= (40\;\text{cpw}, N=16)$): $T_{31} = 0.2428$, $T_{41} = 0.5459$,
#     $T_{\text{tot}} = 0.7887$ (excess loss $1.031\;\text{dB}$; normalized split $30.78\%$ bar /
#     $69.22\%$ cross, imbalance $-3.52\;\text{dB}$), quadrature phase $\Delta\phi = -104.42^\circ$,
#     band-mean $T_{\text{tot}} = 0.7842$ ($1505\text{--}1595\;\text{nm}$) / $0.7855$ ($1510\text{--}1590\;\text{nm}$),
#     and band-mean back-reflection $\langle |S_{11}|^2 + |S_{21}|^2 \rangle_{1515\text{--}1585\;\text{nm}} = 0.01545$.
#     At $\lambda = 1580\;\text{nm}$: $T_{31} = 0.2844$, $T_{41} = 0.5180$, $T_{\text{tot}} = 0.8024$
#     ($35.44\%$ bar / $64.56\%$ cross), $\Delta\phi = -101.15^\circ$.
#   - `kottke_L3` (**L3** $= (20\;\text{cpw}, N=8)$, baseline primary run): $T_{31} = 0.2577$,
#     $T_{41} = 0.5369$, $T_{\text{tot}} = 0.7946$ (excess loss $0.998\;\text{dB}$; $32.43\%$ bar /
#     $67.57\%$ cross, imbalance $-3.19\;\text{dB}$), $\Delta\phi = -103.70^\circ$, band-mean
#     $T_{\text{tot}} = 0.7888$ ($1505\text{--}1595\;\text{nm}$) / $0.7904$ ($1510\text{--}1590\;\text{nm}$),
#     and band-mean reflection $0.01390$. The 2D Poynting flux box closes energy conservation to
#     $\langle \Phi_{\text{box}} \rangle_{1505\text{--}1595\;\text{nm}} = 1.0003$ ($0.9246$ east +
#     $0.0576$ lateral + $0.0180$ west), and port mode solves give $n_{\text{eff,in}} = n_{\text{eff,out}} = 2.3275$
#     at $1550\;\text{nm}$ ($n_{\text{eff,slab}} = 2.8514$). At $\lambda = 1580\;\text{nm}$:
#     $T_{31} = 0.2981$, $T_{41} = 0.5094$, $T_{\text{tot}} = 0.8075$ ($36.92\%$ bar / $63.08\%$ cross),
#     $\Delta\phi = -100.79^\circ$.
#   - `kottke_nonuniform` (**L3 graded non-square**, `task_11`, $425 \times 171 \times 1 = 72{,}675$
#     cells, $-34.4\%$ cells and $-41.8\%$ wall time vs `kottke_L3`): $T_{31} = 0.2511$,
#     $T_{41} = 0.5416$, $T_{\text{tot}} = 0.7927$ (excess loss $1.009\;\text{dB}$; $31.68\%$ bar /
#     $68.32\%$ cross, imbalance $-3.34\;\text{dB}$), $\Delta\phi = -104.09^\circ$, and band-mean
#     reflection $0.01374$. Because its transverse grid nodes lie exactly on the MMI sidewalls
#     $y = \pm 0.50\;\mu\text{m}$ ($\text{sidewall\_err} = 0.0\;\text{nm}$ vs $7.1\;\text{nm}$ on
#     `kottke_L3`), `kottke_nonuniform` achieves a broadband RMS total transmission error
#     $\|T_{\text{tot}} - T_{\text{ref}}\|_2 = 3.14 \times 10^{-3}$ ($1.60\times$ lower than
#     `kottke_L3`'s $5.02 \times 10^{-3}$) and a 2-port RMS error
#     $\|(T_{31}, T_{41}) - T_{\text{ref}}\|_2 = 6.63 \times 10^{-3}$ ($1.87\times$ lower than
#     `kottke_L3`'s $1.24 \times 10^{-2}$).
#   - `kottke_L2` (**L2** $= (14\;\text{cpw}, N=6)$): $T_{31} = 0.2769$, $T_{41} = 0.5222$,
#     $T_{\text{tot}} = 0.7992$ (excess loss $0.974\;\text{dB}$; $34.65\%$ bar / $65.35\%$ cross),
#     $\Delta\phi = -103.02^\circ$, band-mean $T_{\text{tot}} = 0.7919$ ($1505\text{--}1595\;\text{nm}$) /
#     $0.7938$ ($1510\text{--}1590\;\text{nm}$), and band-mean reflection $0.01237$.
#   - `kottke_L1` (**L1** $= (10\;\text{cpw}, N=4)$): $T_{31} = 0.3155$, $T_{41} = 0.4978$,
#     $T_{\text{tot}} = 0.8133$ (excess loss $0.898\;\text{dB}$; $38.80\%$ bar / $61.20\%$ cross),
#     $\Delta\phi = -101.44^\circ$, band-mean $T_{\text{tot}} = 0.8034$ ($1505\text{--}1595\;\text{nm}$) /
#     $0.8058$ ($1510\text{--}1590\;\text{nm}$), and band-mean reflection $0.00927$.
#   - **Convergence & Kottke vs staircase across L1, L2, L3**:
#     Between **L2** and **L3**, Kottke's total
#     transmission $T_{\text{tot}}(1550\;\text{nm})$ changes by $0.571\%$ ($0.7992 \to 0.7946$),
#     its broadband relative RMS transmission difference $\|T_{\text{tot,L3}} - T_{\text{tot,L2}}\|_2 / \langle T_{\text{tot,L3}} \rangle$
#     is $0.478\%$ (down from $1.569\%$ between **L1** and **L2**), and $\Delta\phi(1550\;\text{nm})$
#     changes by $0.68^\circ$ ($0.66\%$), confirming convergence at **L3** $= (20, 8)$ (with only
#     $0.757\%$ difference in $T_{\text{tot}}(1550\;\text{nm})$ and $0.639\%$ broadband relative RMS
#     difference against the **R** $= (40, 16)$ reference). Against `kottke_reference` (**R**),
#     Kottke's broadband RMS transmission error $\|T_{\text{tot}} - T_{\text{ref}}\|_2$ decreases
#     monotonically by $4.22\times$ from $2.12 \times 10^{-2}$ (**L1**) to $8.73 \times 10^{-3}$
#     (**L2**) and $5.02 \times 10^{-3}$ (**L3**), and drops further to $3.14 \times 10^{-3}$ on
#     `kottke_nonuniform`. For staircasing (`staircase_L1`: $T_{31} = 0.3730$,
#     $T_{41} = 0.4399$, $T_{\text{tot}} = 0.8129$, $45.89\%$ bar; `staircase_L2`: $T_{31} = 0.3112$,
#     $T_{41} = 0.4969$, $T_{\text{tot}} = 0.8081$, $38.52\%$ bar; `staircase_L3`: $T_{31} = 0.2818$,
#     $T_{41} = 0.5283$, $T_{\text{tot}} = 0.8101$, $34.79\%$ bar), $\|T_{\text{tot}} - T_{\text{ref}}\|_2$
#     is $1.83 \times 10^{-2}$ (**L1**), $2.42 \times 10^{-2}$ (**L2**, $2.77\times$ larger than Kottke),
#     and $1.82 \times 10^{-2}$ (**L3**, $3.62\times$ larger than Kottke). Note that at **L1**
#     ($10\;\text{cpw}, N=4$), staircasing shifts the bar and cross transmissions in opposite
#     directions ($\Delta T_{31} = +0.1302$, $\Delta T_{41} = -0.1060$ at $1550\;\text{nm}$), producing
#     a fortuitous sign cancellation in the sum $T_{\text{tot}} = T_{31} + T_{41}$ so that
#     $\|T_{\text{tot}} - T_{\text{ref}}\|_2$ at **L1** is slightly smaller for staircase ($1.83 \times 10^{-2}$)
#     than for Kottke ($2.12 \times 10^{-2}$). When the individual port transmissions $(T_{31}, T_{41})$ are
#     measured via the 2-port RMS error $\|(T_{31}, T_{41}) - (T_{31,\text{ref}}, T_{41,\text{ref}})\|_2$,
#     Kottke beats staircasing at every level: $6.11 \times 10^{-2}$ vs $1.10 \times 10^{-1}$ at **L1**
#     ($1.80\times$ lower), $2.93 \times 10^{-2}$ vs $6.16 \times 10^{-2}$ at **L2** ($2.10\times$ lower),
#     $1.24 \times 10^{-2}$ vs $3.07 \times 10^{-2}$ at **L3** ($2.48\times$ lower), and
#     $6.63 \times 10^{-3}$ on `kottke_nonuniform` ($4.63\times$ lower than `staircase_L3`), while suppressing
#     spurious S-bend back-reflection $\langle |S_{11}|^2 + |S_{21}|^2 \rangle$ by $3.14\times$ at **L1**
#     ($0.00927$ vs $0.02913$) and $1.78\times$ at **L2** ($0.01237$ vs $0.02205$).
# - **MMI length sweep ($L_{\text{mmi}} \in [2.50, 3.80]\;\mu\text{m}$ at L3 $= (20, 8)$)**:
#   At $\lambda_0 = 1550\;\text{nm}$, the normalized bar split fraction $P_3 / (P_3 + P_4)$
#   increases monotonically with $L_{\text{mmi}}$: $3.52\%$ at $2.50\;\mu\text{m}$
#   ($T_{31} = 0.0268$, $T_{41} = 0.7331$, $T_{\text{tot}} = 0.7599$, $\Delta\phi = -118.77^\circ$),
#   $32.43\%$ at $3.00\;\mu\text{m}$ ($T_{31} = 0.2577$, $T_{41} = 0.5369$, $T_{\text{tot}} = 0.7946$,
#   $\Delta\phi = -103.70^\circ$), **$49.46\%$ at $3.25\;\mu\text{m}$** ($T_{31} = 0.4282$,
#   $T_{41} = 0.4376$, $T_{\text{tot}} = 0.8659$, excess loss $0.626\;\text{dB}$, imbalance
#   $-0.09\;\text{dB}$, quadrature phase $\Delta\phi = -96.29^\circ$), $62.12\%$ at $3.50\;\mu\text{m}$
#   ($T_{31} = 0.5694$, $T_{41} = 0.3472$, $T_{\text{tot}} = 0.9166$, excess loss $0.378\;\text{dB}$,
#   $\Delta\phi = -89.99^\circ$), and $72.48\%$ at $3.80\;\mu\text{m}$ ($T_{31} = 0.6555$,
#   $T_{41} = 0.2489$, $T_{\text{tot}} = 0.9043$, excess loss $0.437\;\text{dB}$, $\Delta\phi = -88.52^\circ$).
#
# ### Comparison: Baseline L3 (`kottke_L3`) vs Graded Non-Uniform (`kottke_nonuniform`) vs Reference R & Theory
#
# All geometry parameters ($w_{\text{mmi}} = 1.00\;\mu\text{m}$, nominal $L_{\text{mmi}} = 3.00\;\mu\text{m}$,
# $w_{\text{wg}} = 0.40\;\mu\text{m}$, $g_{\text{mmi}} = 0.20\;\mu\text{m}$) are fixed to the 3D SOI
# layout without tuning. In 2D EIM ($n_{\text{eff,slab}} = 2.8514$), the lateral beat length
# is $L_\pi^{\text{2D}} = \pi / (\beta_0 - \beta_1) = 2.5435\;\mu\text{m}$ (Soldano & Pennings paraxial
# $L_\pi^{\text{SP}} = 2.7100\;\mu\text{m}$, $W_e = 1.0511\;\mu\text{m}$), which is $\approx 8\%$ longer
# than in 3D SOI, shifting the 2D EIM $50/50$ general-interference length from the nominal
# $L_{\text{mmi}} = 3.00\;\mu\text{m}$ to $L_{\text{mmi}} = 3.25\;\mu\text{m}$:
#
# | Metric ($\lambda_0 = 1550\;\text{nm}$ unless noted) | Baseline L3 (`kottke_L3`, task 4) | Graded Non-Uniform (`kottke_nonuniform`, task 11) | Reference R (`kottke_reference`, task 6) | Self-Imaging Theory (2D EIM) [1, 4] |
# |---|---:|---:|---:|---:|
# | Grid shape & total cells | $520 \times 213 \times 1$ ($110{,}760$) | $425 \times 171 \times 1$ ($72{,}675$, **$-34.4\%$**) | $1016 \times 403 \times 1$ ($409{,}448$) | — |
# | Wall-clock runtime & peak RSS | $231.99\;\text{s}$, $772.0\;\text{MB}$ | $134.98\;\text{s}$ (**$-41.8\%$**), $694.3\;\text{MB}$ | $1{,}410.36\;\text{s}$, $1{,}271.3\;\text{MB}$ | — |
# | MMI sidewall node error ($y = \pm 0.50\;\mu\text{m}$) | $7.1\;\text{nm}$ | **$0.0\;\text{nm}$** | $3.6\;\text{nm}$ | $0.0\;\text{nm}$ |
# | Nominal $L_{\text{mmi}} = 3.00\;\mu\text{m}$ bar / cross $T_{31}, T_{41}$ | $0.2577,\; 0.5369$ | $0.2511,\; 0.5416$ | $0.2428,\; 0.5459$ | $0.1737,\; 0.3316$ (step 4-mode) |
# | Nominal $L_{\text{mmi}} = 3.00\;\mu\text{m}$ normalized split $P_3 / P_{\text{tot}}$ | $32.43\% / 67.57\%$ | $31.68\% / 68.32\%$ | $30.78\% / 69.22\%$ | $34.5\% / 65.5\%$ ($\Delta L_{\text{eff}} = 0.56\;\mu\text{m}$) |
# | Nominal $L_{\text{mmi}} = 3.00\;\mu\text{m}$ $T_{\text{tot}}$ & excess loss | $0.7946$ ($0.998\;\text{dB}$) | $0.7927$ ($1.009\;\text{dB}$) | $0.7887$ ($1.031\;\text{dB}$) | $\sum_{m=0}^3 |c_m|^2 = 0.9333$ ($\ge 0.60\;\text{dB}$) |
# | Nominal $L_{\text{mmi}} = 3.00\;\mu\text{m}$ quadrature phase $\Delta\phi$ | $-103.70^\circ$ | $-104.09^\circ$ | $-104.42^\circ$ | $-90.0^\circ$ at $3 L_\pi / 2$ |
# | Broadband RMS error $\|T_{\text{tot}} - T_{\text{ref}}\|_2$ ($1510\text{--}1590\;\text{nm}$) | $5.02 \times 10^{-3}$ | **$3.14 \times 10^{-3}$ ($1.60\times$ lower)** | $0$ (ref) | $\mathcal{O}(\Delta x^2)$ subpixel [3] |
# | Broadband 2-port RMS error $\|(T_{31}, T_{41}) - T_{\text{ref}}\|_2$ | $1.24 \times 10^{-2}$ | **$6.63 \times 10^{-3}$ ($1.87\times$ lower)** | $0$ (ref) | — |

# %%
def get_task_table(quick=False):
    """
    Return the study task table:
    - ``quick=True``: 9 tasks (Kottke & staircase at L1=(10,4), L2=(14,6);
      Kottke reference at L3=(20,8); 4 length-sweep tasks at L2=(14,6)).
    - ``quick=False``: 11 tasks (Kottke & staircase at L1=(10,4), L2=(14,6),
      L3=(20,8); Kottke reference at R=(40,16); 4 length-sweep tasks at L3=(20,8)).
    """
    ladder_levels = ("L1", "L2") if quick else ("L1", "L2", "L3")
    primary_level = "L2" if quick else "L3"
    ref_level = "L3" if quick else "R"
    sweep_level = "L2" if quick else "L3"
    n_pml = 8 if quick else 10
    energy_decay = 5e-3 if quick else 1e-4

    tasks = []
    for lvl in ladder_levels:
        cpw, n_feat = RESOLUTION_LADDER[lvl]
        for avg in ("kottke", "staircase"):
            idx = len(tasks)
            is_primary = lvl == primary_level and avg == "kottke"
            tasks.append(
                {
                    "index": idx,
                    "name": f"{avg}_{lvl}",
                    "level": lvl,
                    "averaging": avg,
                    "cpw": cpw,
                    "n_cells_feature": n_feat,
                    "min_feature": GAP_MMI,
                    "l_mmi": L_MMI,
                    "n_pml": n_pml,
                    "energy_decay": energy_decay,
                    "collect_fields": is_primary,
                    "record_movie": is_primary,
                    "movie_frames": 32 if quick else 48,
                    "description": (
                        f"2D EIM 2x2 MMI ({avg}, {lvl}=({cpw} cpw, N={n_feat}), L_mmi=3.00um"
                        + (", DFT fields + movie + flux box)" if is_primary else ")")
                    ),
                }
            )

    ref_cpw, ref_n = RESOLUTION_LADDER[ref_level]
    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_reference",
            "level": "reference",
            "averaging": "kottke",
            "cpw": ref_cpw,
            "n_cells_feature": ref_n,
            "min_feature": GAP_MMI,
            "l_mmi": L_MMI,
            "n_pml": n_pml,
            "energy_decay": energy_decay,
            "collect_fields": False,
            "record_movie": False,
            "movie_frames": 0,
            "description": (
                f"2D EIM 2x2 MMI (kottke, reference {ref_level}=({ref_cpw} cpw, N={ref_n}), L_mmi=3.00um)"
            ),
        }
    )

    sw_cpw, sw_n = RESOLUTION_LADDER[sweep_level]
    for l_um in L_MMI_SWEEP_UM:
        if np.isclose(l_um, L_MMI * 1e6):
            continue
        tag = f"{l_um:.2f}".replace(".", "p")
        tasks.append(
            {
                "index": len(tasks),
                "name": f"length_{tag}um",
                "level": "sweep",
                "averaging": "kottke",
                "cpw": sw_cpw,
                "n_cells_feature": sw_n,
                "min_feature": GAP_MMI,
                "l_mmi": l_um * 1e-6,
                "n_pml": n_pml,
                "energy_decay": energy_decay,
                "collect_fields": False,
                "record_movie": False,
                "movie_frames": 0,
                "description": (
                    f"2D EIM 2x2 MMI length sweep (kottke, {sweep_level}=({sw_cpw} cpw, N={sw_n}), L_mmi={l_um:.2f}um)"
                ),
            }
        )

    nu_cpw, nu_n = RESOLUTION_LADDER[primary_level]
    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_nonuniform",
            "level": "nonuniform",
            "averaging": "kottke",
            "cpw": nu_cpw,
            "n_cells_feature": nu_n,
            "min_feature": GAP_MMI,
            "l_mmi": L_MMI,
            "n_pml": n_pml,
            "energy_decay": energy_decay,
            "collect_fields": False,
            "record_movie": False,
            "movie_frames": 0,
            "nonuniform": True,
            "description": (
                f"2D EIM 2x2 MMI (kottke, graded non-uniform non-square mesh, "
                f"{primary_level}=({nu_cpw} cpw, N={nu_n}), L_mmi=3.00um)"
            ),
        }
    )
    return tasks


def build_task_simulation(task_cfg, use_gpu=False):
    """Build and return ``(pic, structures, ports, domain)`` for ``task_cfg``."""
    l_mmi = float(task_cfg.get("l_mmi", L_MMI))
    min_feature = float(task_cfg.get("min_feature", GAP_MMI))
    n_cells_feature = int(task_cfg.get("n_cells_feature", 4))
    structures, ports, domain = build_mmi_2x2(l_mmi=l_mmi)
    common_kw = dict(
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
        field_wavelengths=(WL_REF, WL_5050_3D),
        record_movie=task_cfg["record_movie"],
        movie_frames=task_cfg["movie_frames"],
        use_gpu=use_gpu,
    )
    if task_cfg.get("nonuniform", False):
        pic = NonUniformMMI2x2PIC(
            l_mmi=l_mmi,
            w_mmi=W_MMI,
            w_wg=W_WG,
            min_feature=min_feature,
            y_port=Y_PORT,
            n_cells_feature=n_cells_feature,
            **common_kw,
        )
    else:
        mesh_overrides = build_gap_mesh_overrides(
            l_mmi=l_mmi,
            n_cells_feature=n_cells_feature,
            min_feature=min_feature,
        )
        pic = PICSimulation(mesh_overrides=mesh_overrides, **common_kw)
    return pic, structures, ports, domain


def run_single_task(task_cfg, out_dir):
    """
    Execute one 2D EIM 2x2 MMI study task and write ``task_<I>.npz`` and
    ``task_<I>.json`` into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task_cfg["index"]
    t0 = time.perf_counter()

    l_mmi = float(task_cfg.get("l_mmi", L_MMI))
    min_feature = float(task_cfg.get("min_feature", GAP_MMI))
    n_cells_feature = int(task_cfg.get("n_cells_feature", 4))
    use_gpu = os.environ.get("WAKIS_USE_GPU", "0") == "1"
    pic, structures, ports, domain = build_task_simulation(task_cfg, use_gpu=use_gpu)

    s_mat = pic.run_s_parameters(
        drive=["in1"],
        energy_decay=task_cfg["energy_decay"],
        collect_fields=task_cfg["collect_fields"],
    )

    res_metrics = measure_grid_resolution(
        pic.grid,
        n_pml=task_cfg["n_pml"],
        l_mmi=l_mmi,
        min_feature=min_feature,
        w_mmi=W_MMI,
    )
    n_steps = int(pic.solver.Nt)
    zoom_patch = extract_zoom_permittivity(pic.solver, component="y")

    payload = {
        "task_index": np.int32(idx),
        "name": np.array(task_cfg["name"]),
        "level": np.array(task_cfg["level"]),
        "averaging": np.array(task_cfg["averaging"]),
        "cpw": np.float32(task_cfg["cpw"]),
        "n_cells_feature": np.int32(n_cells_feature),
        "l_mmi": np.float64(l_mmi),
        "min_feature_nm": np.float64(res_metrics["min_feature_nm"]),
        "dx_background_nm": np.float64(res_metrics["dx_background_nm"]),
        "dx_min_nm": np.float64(res_metrics["dx_min_nm"]),
        "cells_across_feature": np.float64(res_metrics["cells_across_feature"]),
        "n_cells": np.int32(res_metrics["n_cells"]),
        "n_steps": np.int32(n_steps),
        "n_eff_slab": np.float64(pic.n_eff_slab),
        "frequencies": pic.frequencies.astype(np.float64),
        "wavelengths": pic.wavelengths.astype(np.float64),
        "S11": s_mat[("in1", "in1")].astype(np.complex128),
        "S21": s_mat[("in2", "in1")].astype(np.complex128),
        "S31": s_mat[("out1", "in1")].astype(np.complex128),
        "S41": s_mat[("out2", "in1")].astype(np.complex128),
        **zoom_patch,
    }

    if task_cfg["collect_fields"]:
        modes_in = pic.solve_port_modes("in1", frequencies=c_light / WL_REF, n_modes=1)
        modes_out = pic.solve_port_modes(
            "out2", frequencies=c_light / WL_REF, n_modes=1
        )
        y_in = modes_in[0].coords["Eb"][0]
        ey_in = modes_in[0].e["b"][:, 0]
        m_in = np.abs(y_in - Y_PORT) <= 0.85e-6
        y_out = modes_out[0].coords["Eb"][0]
        ey_out = modes_out[0].e["b"][:, 0]
        m_out = np.abs(y_out + Y_PORT) <= 0.85e-6
        payload["port_y_in"] = y_in[m_in].astype(np.float32)
        payload["port_ey_in"] = ey_in[m_in].astype(np.complex64)
        payload["port_neff_in"] = np.float64(np.real(modes_in[0].neff))
        payload["port_y_out"] = y_out[m_out].astype(np.float32)
        payload["port_ey_out"] = ey_out[m_out].astype(np.complex64)
        payload["port_neff_out"] = np.float64(np.real(modes_out[0].neff))

        wl_disp = np.linspace(WL_MIN, WL_MAX, 11)
        f_disp = c_light / wl_disp
        disp_modes = pic.solve_port_modes("in1", frequencies=f_disp, n_modes=1)
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

        if WL_5050_3D in pic.dft_fields:
            fmap2 = pic.dft_fields[WL_5050_3D]
            payload["dft_Ey_1580"] = fmap2["Ey"]
            payload["dft_E2_1580"] = fmap2["E2"]

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
        "n_cells_feature": n_cells_feature,
        "nonuniform": bool(task_cfg.get("nonuniform", False)),
        "l_mmi_um": round(l_mmi * 1e6, 3),
        "min_feature_nm": res_metrics["min_feature_nm"],
        "dx_background_nm": res_metrics["dx_background_nm"],
        "dx_min_nm": res_metrics["dx_min_nm"],
        "dx_max_nm": res_metrics["dx_max_nm"],
        "dy_min_nm": res_metrics["dy_min_nm"],
        "dy_max_nm": res_metrics["dy_max_nm"],
        "sidewall_err_nm": res_metrics["sidewall_err_nm"],
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
        f"[task_{idx} {task_cfg['name']}] cpw={task_cfg['cpw']} N={n_cells_feature} "
        f"min_feature_nm={meta['min_feature_nm']} dx_background_nm={meta['dx_background_nm']} "
        f"dx_min_nm={meta['dx_min_nm']} cells_across_feature={meta['cells_across_feature']} "
        f"n_cells={meta['n_cells']} n_steps={meta['n_steps']} "
        f"time={elapsed:.2f}s rss={rss_mb:.1f}MB"
    )
    return meta


# %% [markdown]
# ## 6. Generating the complete figure suite
#
# When invoked with `--plot` (or at the end of `--quick` / `--full`),
# `generate_all_figures` reads `data/task_*.npz` and writes 18 setup, diagnostic,
# and validation figures to `figures/`:
#
# 1. `00a_setup_geometry.png` — Labelled simulation setup schematic (materials, 10-cell CPML boundaries, Port 1 `ModeSource` with launched $|E_y(y)|$ mode profile, 4-port `ModeMonitors`, and closed 2D Poynting `FluxBox`).
# 2. `00b_setup_mesh_and_excitation.png` — Graded non-uniform, non-square Cartesian mesh with $1\times$ zoom inset at the $200\;\text{nm}$ throat gap, spatial wave-packet envelope, and Gaussian excitation spectrum (`00_setup_table.md` companion table).
# 3. `01_mmi_2x2_layout.png` — Annotated top-view layout of the $2 \times 2$ MMI power splitter.
# 4. `02_eim_vertical_stack.png` — Vertical $220\;\text{nm}$ SOI stack, 1D $\text{TE}_0$ slab mode profile, and $n_{\text{eff,slab}}(\lambda)$.
# 5. `03_mmi_lateral_modes_and_carpet.png` — Guided lateral eigenmodes $\psi_{0..3}(y)$ of the $1.0\;\mu\text{m}$ MMI section and analytical 2D Talbot self-imaging carpet $|E(x, y)|^2$.
# 6. `04_permittivity_mesh_zoom.png` — Zoomed in-plane permittivity map with Yee cell edges and annotated $200\;\text{nm}$ throat-gap width at **L1** (staircase & Kottke) and at the **reference** level.
# 7. `04b_mesh_upgrade_comparison.png` — Side-by-side mesh comparison between baseline **L3** (`kottke_L3`, $520 \times 213 = 110{,}760$ cells) and graded non-square (`kottke_nonuniform`, $425 \times 171 = 72{,}675$ cells, $-34.4\%$ cells, $-41.8\%$ wall time) plus broadband bar/cross transmission agreement against **R** $= (40, 16)$.
# 8. `05_port_mode_profiles.png` — Input (`in1`) and lower output (`out2`) port eigenmode profiles $|E_y(y)|$ and $\mathrm{Re}(E_y(y))$.
# 9. `06_port_neff_dispersion.png` — Access waveguide ($w_{\text{wg}} = 400\;\text{nm}$) effective index $n_{\text{eff}}(\lambda)$ and group index $n_g(\lambda)$.
# 10. `07_field_intensity_1550.png` — Steady-state logarithmic intensity $10\log_{10}(|E|^2 / |E|_{\max}^2)$ at $1550\;\text{nm}$ with `overlay_geometry`.
# 11. `08_field_real_ey_1550.png` — Steady-state wave-front phase map $\mathrm{Re}(E_y)$ at $1550\;\text{nm}$ showing output quadrature with `overlay_geometry`.
# 12. `09_field_comparison_1550_vs_1580.png` — Side-by-side linear intensity maps $|E|^2 / |E|_{\max}^2$ at $\lambda = 1550\;\text{nm}$ and $1580\;\text{nm}$ with `overlay_geometry`.
# 13. `10_pulse_propagation.gif` — Time-domain pulse propagation movie through the $2 \times 2$ MMI splitter with `overlay_geometry` on every frame.
# 14. `11_sparameter_spectra_linear_and_db.png` — Broadband linear power splitting ($|S_{31}|^2, |S_{41}|^2$, normalized split ratios, and total transmission) and logarithmic S-parameter spectra in dB.
# 15. `12_imbalance_phase_and_flux_budget.png` — Power imbalance, output quadrature phase difference $\Delta\phi(\lambda) \approx -90^\circ$, and 2D Poynting flux-box energy conservation budget.
# 16. `13_mmi_length_sweep_vs_analytic.png` — MMI box length sweep ($L_{\text{mmi}} \in [2.5, 3.8]\;\mu\text{m}$) compared against Soldano & Pennings general-interference theory ($3 L_\pi / 2$).
# 17. `14_kottke_vs_staircase_convergence.png` — Subpixel convergence of Kottke vs staircasing against the independent Kottke reference run (`kottke_reference`), labeled with both $\text{cpw}$ and cells across the $200\;\text{nm}$ throat gap.
# 18. `15_port_time_trace.png` — Output port time-domain waveform $E_y(t)$ and exponential energy ring-down.
#
# ![2x2 MMI layout](figures/01_mmi_2x2_layout.png)
# ![220 nm SOI EIM stack](figures/02_eim_vertical_stack.png)
# ![Lateral MMI modes and Talbot carpet](figures/03_mmi_lateral_modes_and_carpet.png)
# ![Throat gap permittivity mesh zoom](figures/04_permittivity_mesh_zoom.png)
# ![Port mode profiles](figures/05_port_mode_profiles.png)
# ![Port effective index dispersion](figures/06_port_neff_dispersion.png)
# ![Steady-state field intensity at 1550 nm](figures/07_field_intensity_1550.png)
# ![Steady-state Re(Ey) at 1550 nm](figures/08_field_real_ey_1550.png)
# ![Field comparison 1550 nm vs 1580 nm](figures/09_field_comparison_1550_vs_1580.png)
# ![Pulse propagation movie](figures/10_pulse_propagation.gif)
# ![Broadband S-parameter spectra](figures/11_sparameter_spectra_linear_and_db.png)
# ![Imbalance, quadrature phase, and flux budget](figures/12_imbalance_phase_and_flux_budget.png)
# ![MMI length sweep vs analytic theory](figures/13_mmi_length_sweep_vs_analytic.png)
# ![Kottke vs staircase convergence](figures/14_kottke_vs_staircase_convergence.png)
# ![Port time trace and ring-down](figures/15_port_time_trace.png)


# %%
def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Read ``data_dir/task_*.npz`` and render all setup and diagnostic figures into ``fig_dir``."""
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
        if jp.is_file():
            tasks_meta[name] = json.loads(jp.read_text())

    primary = None
    for cand in ("kottke_L3", "kottke_L2", "kottke_fine"):
        if cand in tasks_data and "dft_E2" in tasks_data[cand]:
            primary = tasks_data[cand]
            break
    if primary is None:
        for d in tasks_data.values():
            if "dft_E2" in d:
                primary = d
                break
    if primary is None:
        raise FileNotFoundError(
            f"Missing primary task with DFT fields in {data_dir}. Run --quick or --full first."
        )
    structures, ports, domain = build_mmi_2x2(l_mmi=L_MMI)

    # 0a & 0b. Setup figures (geometry/materials/sources/monitors/boundaries + mesh & excitation)
    prim_name = str(primary.get("name", "kottke_L3"))
    n_pml_prim = int(tasks_meta.get(prim_name, {}).get("n_pml", 10))
    cpw_prim = float(primary.get("cpw", 20.0))
    n_feat_prim = int(round(float(primary.get("n_cells_feature", 8))))
    ovr_prim = build_gap_mesh_overrides(
        l_mmi=L_MMI, n_cells_feature=n_feat_prim, min_feature=GAP_MMI
    )
    pic_uniform = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw_prim,
        domain=domain,
        n_pml=n_pml_prim,
        mesh_overrides=ovr_prim,
    )
    pic_nonuniform = NonUniformMMI2x2PIC(
        WL_MIN,
        WL_MAX,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw_prim,
        domain=domain,
        n_pml=n_pml_prim,
        l_mmi=L_MMI,
        n_cells_feature=n_feat_prim,
        min_feature=GAP_MMI,
    )
    save_mmi_2x2_setup_figures(
        pic_nonuniform,
        structures,
        primary,
        fig_dir,
        l_mmi=L_MMI,
        w_mmi=W_MMI,
        w_wg=W_WG,
        min_feature=GAP_MMI,
        y_port=Y_PORT,
        x_port=X_PORT,
    )

    # 1. Top-view device layout with dimensional annotations
    annotations = [
        {
            "text": "MMI box\n1.0 × 3.0 µm",
            "xy": (0.0, 0.50),
            "xytext": (0.0, 1.75),
            "arrowprops": dict(arrowstyle="->", color="0.25", lw=1.1),
        },
        {
            "text": "200 nm throat gap\n(y = ±0.30 µm)",
            "xy": (1.50, 0.0),
            "xytext": (2.95, 0.0),
            "arrowprops": dict(arrowstyle="->", color="#d95f02", lw=1.2),
        },
        {
            "text": "Cosine S-bends\nLsb = 4.0 µm, Δy = ±1.0 µm",
            "xy": (-3.50, 0.80),
            "xytext": (-3.40, 2.00),
            "arrowprops": dict(arrowstyle="->", color="0.25", lw=1.0),
        },
        {
            "text": "Access WGs (w = 400 nm)\ny = ±1.30 µm",
            "xy": (4.80, -1.30),
            "xytext": (3.70, -2.10),
            "arrowprops": dict(arrowstyle="->", color="0.25", lw=1.0),
        },
    ]
    plot_layout(
        structures,
        ports=ports,
        domain=domain,
        annotations=annotations,
        title="2×2 MMI Power Splitter layout on 220 nm SOI (2D EIM domain)",
        out_path=fig_dir / "01_mmi_2x2_layout.png",
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

    # 3. Guided lateral MMI eigenmodes and 2D analytical Talbot self-imaging carpet
    ana = analytic_mmi_model(wavelength=WL_REF)
    y_um = ana["y"] * 1e6
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.2, height=4.2)
    mode_cols = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]
    ax0.axvspan(-0.5 * W_MMI * 1e6, 0.5 * W_MMI * 1e6, color="0.88", alpha=0.7, label="MMI core (1.0 µm)")
    for m_idx, (neff_m, prof_m) in enumerate(zip(ana["neffs"], ana["profiles"])):
        weight_pct = float(np.abs(ana["c_bar"][m_idx]) ** 2) * 100.0
        scale = float(np.max(np.abs(prof_m))) or 1.0
        ax0.plot(
            y_um,
            prof_m / scale,
            color=mode_cols[m_idx % len(mode_cols)],
            lw=1.9,
            label=rf"$\nu={m_idx}$: $n_{{\mathrm{{eff}}}}={neff_m:.3f}$ ($|c_{m_idx}|^2={weight_pct:.1f}\%$)",
        )
    ax0.set(
        xlim=(-1.2, 1.2),
        ylim=(-1.15, 1.15),
        xlabel="Lateral coordinate y [µm]",
        ylabel=r"Normalized lateral mode profile $\psi_\nu(y)$",
        title="(a) Guided lateral eigenmodes of 1.0 µm MMI section",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower left", fontsize=8.5)

    l_max_carpet = 1.65 * ana["l_pi_exact"]
    x_carpet = np.linspace(0.0, l_max_carpet, 240)
    y_mask = np.abs(ana["y"]) <= 0.85e-6
    y_c_um = y_um[y_mask]
    prof_c = ana["profiles"][:, y_mask]
    # Field superposition E(x, y) = sum_m c_m psi_m(y) exp(-i beta_m x)
    phase_mat = np.exp(-1j * ana["betas"][:, None] * x_carpet[None, :])
    field_xy = (ana["c_bar"][:, None] * phase_mat).T @ prof_c
    inten_xy = np.abs(field_xy) ** 2
    inten_xy /= max(float(np.max(inten_xy)), 1e-30)

    im = ax1.pcolormesh(
        x_carpet * 1e6,
        y_c_um,
        inten_xy.T,
        cmap="inferno",
        shading="auto",
        vmin=0.0,
        vmax=1.0,
    )
    for y_edge in (-0.5, 0.5):
        ax1.axhline(y_edge, color="white", ls="--", lw=1.0, alpha=0.8)
    l_pi_um = ana["l_pi_exact"] * 1e6
    for x_mark, y_mark, lbl, col in [
        (0.5 * l_pi_um, 0.76, r"$L_\pi/2$", "#66c2a5"),
        (1.0 * l_pi_um, -0.64, r"$L_\pi$ (cross)", "#fc8d62"),
        (L_MMI * 1e6, 0.76, r"$L_{\mathrm{mmi}}=3.0\;\mu\mathrm{m}$", "white"),
        (1.5 * l_pi_um, -0.64, r"$3L_\pi/2$ (3 dB)", "#8da0cb"),
    ]:
        ax1.axvline(x_mark, color=col, ls=":", lw=1.4)
        ax1.text(
            x_mark,
            y_mark,
            lbl,
            color=col,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55),
        )
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label(r"Normalized modal intensity $|\Psi(x, y)|^2$")
    ax1.set(
        xlabel="Propagation distance x along MMI [µm]",
        ylabel="Lateral coordinate y [µm]",
        title=rf"(b) 4-mode Talbot self-imaging carpet ($L_\pi = {l_pi_um:.2f}\;\mu\mathrm{{m}}$)",
    )
    figstyle.save(fig, fig_dir / "03_mmi_lateral_modes_and_carpet.png")

    # 4. In-plane permittivity map with cell edges zoomed on the 200 nm throat gap: L1 vs Reference
    def _ensure_zoom(d_task, default_cpw, default_n, avg):
        if d_task is not None and "zoom_x" in d_task:
            return d_task
        cpw_v = float(d_task["cpw"]) if d_task is not None else float(default_cpw)
        n_v = int(d_task.get("n_cells_feature", default_n)) if d_task is not None else int(default_n)
        ovr = build_gap_mesh_overrides(l_mmi=L_MMI, n_cells_feature=n_v, min_feature=GAP_MMI)
        pic_z = PICSimulation(
            WL_MIN,
            WL_MAX,
            structures=structures,
            ports=ports,
            n_core=N_SI,
            n_clad=N_SIO2,
            two_d=True,
            averaging=avg,
            cells_per_wavelength=cpw_v,
            domain=(0.5e-6, 2.6e-6, -0.9e-6, 0.9e-6),
            n_pml=6,
            mesh_overrides=ovr,
        )
        m_z = measure_grid_resolution(pic_z.grid, n_pml=6, l_mmi=L_MMI, min_feature=GAP_MMI)
        zp = extract_zoom_permittivity(pic_z.solver, component="y")
        return {**m_z, **zp, "cpw": cpw_v, "n_cells_feature": n_v}

    d_s1 = _ensure_zoom(tasks_data.get("staircase_L1"), 10, 4, "staircase")
    d_k1 = _ensure_zoom(tasks_data.get("kottke_L1"), 10, 4, "kottke")
    d_ref = _ensure_zoom(tasks_data.get("kottke_reference"), 20, 8, "kottke")
    ref_cpw_z = int(round(float(d_ref["cpw"])))
    ref_n_z = int(round(float(d_ref.get("n_cells_feature", d_ref["cells_across_feature"]))))
    ref_tag = "R" if ref_cpw_z >= 40 else "L3"

    zoom_panels = [
        {
            **d_s1,
            "title": (
                f"(a) L1 Staircase ({int(d_s1['cpw'])} cpw, N={int(d_s1.get('n_cells_feature', 4))})\n"
                f"Rounded fill across 200 nm throat gap"
            ),
        },
        {
            **d_k1,
            "title": (
                f"(b) L1 Kottke ({int(d_k1['cpw'])} cpw, N={int(d_k1.get('n_cells_feature', 4))})\n"
                f"Subpixel tensor across 200 nm throat gap"
            ),
        },
        {
            **d_ref,
            "title": (
                f"(c) Reference {ref_tag} Kottke ({ref_cpw_z} cpw, N={ref_n_z})\n"
                f"Fine reference mesh across 200 nm throat gap"
            ),
        },
    ]
    plot_throat_gap_mesh_zoom(
        zoom_panels,
        structures=structures,
        l_mmi=L_MMI,
        min_feature=GAP_MMI,
        out_path=fig_dir / "04_permittivity_mesh_zoom.png",
    )

    # 4b. Mesh upgrade comparison: baseline uniform L3 vs graded non-square non-uniform vs 40 cpw ref
    d_nu = tasks_data.get("kottke_nonuniform", primary)
    d_ref_cmp = tasks_data.get("kottke_reference", primary)
    prim_name = str(primary.get("name", "kottke_L3"))
    plot_mesh_upgrade_comparison_mmi_2x2(
        pic_uniform,
        pic_nonuniform,
        structures,
        primary,
        d_nu,
        d_ref_cmp,
        tasks_meta.get(prim_name),
        tasks_meta.get("kottke_nonuniform"),
        fig_dir / "04b_mesh_upgrade_comparison.png",
        l_mmi=L_MMI,
        w_mmi=W_MMI,
        min_feature=GAP_MMI,
    )

    # 5. Port mode profiles at input (in1) and lower output arm (out2)
    plot_port_modes(
        primary["port_y_in"] * 1e6,
        primary["port_ey_in"],
        float(primary["port_neff_in"]),
        primary["port_y_out"] * 1e6,
        primary["port_ey_out"],
        float(primary["port_neff_out"]),
        structures_outline_y=[
            (Y_PORT * 1e6 - 0.5 * W_WG * 1e6, Y_PORT * 1e6 + 0.5 * W_WG * 1e6),
            (-Y_PORT * 1e6 - 0.5 * W_WG * 1e6, -Y_PORT * 1e6 + 0.5 * W_WG * 1e6),
        ],
        out_path=fig_dir / "05_port_mode_profiles.png",
    )

    # 6. Access waveguide mode effective index and group index dispersion
    plot_neff_dispersion(
        primary["disp_wavelengths"] * 1e9,
        primary["disp_neff"],
        neff_slab=float(primary["n_eff_slab"]),
        ng_2d=primary["disp_ng"],
        out_path=fig_dir / "06_port_neff_dispersion.png",
    )

    # 7. Steady-state logarithmic intensity map |E|^2 [dB] at 1550 nm with overlay_geometry
    render_mmi_2x2_field_map_with_overlay(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_E2"],
        pic_nonuniform,
        structures,
        kind="intensity_log",
        title=r"Steady-state modal intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $\lambda = 1550\;\mathrm{nm}$",
        out_path=fig_dir / "07_field_intensity_1550.png",
    )

    # 8. Steady-state real field map Re(Ey) at 1550 nm with overlay_geometry
    render_mmi_2x2_field_map_with_overlay(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_Ey"],
        pic_nonuniform,
        structures,
        kind="real",
        title=r"Steady-state wave-front phase pattern $\mathrm{Re}(E_y)$ at $\lambda = 1550\;\mathrm{nm}$",
        out_path=fig_dir / "08_field_real_ey_1550.png",
    )

    # 9. Side-by-side linear intensity comparison at 1550 nm vs 1580 nm with overlay_geometry
    e2_1550 = primary["dft_E2"]
    e2_1580 = primary.get("dft_E2_1580", e2_1550)
    render_mmi_2x2_field_comparison_with_overlay(
        primary["dft_x"],
        primary["dft_y"],
        e2_1550,
        e2_1580,
        pic_nonuniform,
        structures,
        out_path=fig_dir / "09_field_comparison_1550_vs_1580.png",
    )

    # 10. Animated pulse propagation GIF with overlay_geometry on every frame
    if "movie_frames" in primary and len(primary["movie_frames"]) > 0:
        render_mmi_2x2_movie_with_overlay(
            primary["movie_x"],
            primary["movie_y"],
            primary["movie_frames"],
            primary["movie_times"],
            pic_nonuniform,
            structures,
            out_path=fig_dir / "10_pulse_propagation.gif",
            fps=12,
        )

    # Sort wavelengths ascending for spectral plots (interior of pulse band: 1505-1595 nm)
    wl_nm_all = primary["wavelengths"] * 1e9
    order = np.argsort(wl_nm_all)
    wl_nm = wl_nm_all[order]
    mask = (wl_nm >= 1505.0) & (wl_nm <= 1595.0)
    wl_b = wl_nm[mask]

    s11 = primary["S11"][order][mask]
    s21 = primary["S21"][order][mask]
    s31 = primary["S31"][order][mask]
    s41 = primary["S41"][order][mask]
    t_bar = np.abs(s31) ** 2
    t_cross = np.abs(s41) ** 2
    t_tot = t_bar + t_cross
    r11 = np.abs(s11) ** 2
    r21 = np.abs(s21) ** 2

    # 11. Broadband S-parameter spectra: linear splitting ratio & dB spectra
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.2, height=4.2)
    ax0.plot(
        wl_b,
        t_bar,
        color="#1b9e77",
        lw=2.1,
        label=r"$T_{\mathrm{bar}} = |S_{31}|^2$ (Port 3, upper)",
    )
    ax0.plot(
        wl_b,
        t_cross,
        color="#d95f02",
        lw=2.1,
        label=r"$T_{\mathrm{cross}} = |S_{41}|^2$ (Port 4, lower)",
    )
    ax0.plot(
        wl_b,
        t_bar / np.maximum(t_tot, 1e-12),
        color="#1b9e77",
        ls="--",
        lw=1.5,
        alpha=0.85,
        label=r"Normalized $P_3 / (P_3 + P_4)$",
    )
    ax0.plot(
        wl_b,
        t_cross / np.maximum(t_tot, 1e-12),
        color="#d95f02",
        ls="--",
        lw=1.5,
        alpha=0.85,
        label=r"Normalized $P_4 / (P_3 + P_4)$",
    )
    ax0.plot(
        wl_b,
        t_tot,
        color="#2b5c8f",
        lw=1.9,
        ls="-.",
        label=rf"Total $T_{{31}}+T_{{41}}$ (mean={np.mean(t_tot):.3f})",
    )
    ax0.axhline(0.50, color="0.4", ls=":", lw=1.2, label="Ideal 50/50 split (0.50)")
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Linear modal power fraction",
        ylim=(0.15, 0.98),
        title="(a) Linear transmission & splitting ratio (2D EIM)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="best", fontsize=8.5, ncol=2)

    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(t_bar, 1e-12)),
        color="#1b9e77",
        lw=2.0,
        label=r"$|S_{31}|^2$ (bar, Port 3)",
    )
    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(t_cross, 1e-12)),
        color="#d95f02",
        lw=2.0,
        label=r"$|S_{41}|^2$ (cross, Port 4)",
    )
    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(t_tot, 1e-12)),
        color="#2b5c8f",
        ls="-.",
        lw=1.8,
        label=r"Total $10\log_{10}(|S_{31}|^2 + |S_{41}|^2)$",
    )
    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(r11, 1e-12)),
        color="#7570b3",
        ls="--",
        lw=1.6,
        label=r"$|S_{11}|^2$ (input return loss)",
    )
    ax1.plot(
        wl_b,
        10.0 * np.log10(np.maximum(r21, 1e-12)),
        color="#e7298a",
        ls=":",
        lw=1.6,
        label=r"$|S_{21}|^2$ (idle input reflection)",
    )
    ax1.axhline(-3.0103, color="0.35", ls=":", lw=1.2, label="Ideal −3.01 dB split")
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Scattering magnitude [dB]",
        title="(b) Broadband 4-port S-parameters in dB",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="best", fontsize=8.5)
    figstyle.save(fig, fig_dir / "11_sparameter_spectra_linear_and_db.png")

    # 12. Imbalance, output quadrature phase difference (~90 deg), and 2D Poynting flux budget
    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=13.2, height=4.1)
    imb_db = 10.0 * np.log10(np.maximum(t_bar, 1e-12) / np.maximum(t_cross, 1e-12))
    ax0.plot(
        wl_b,
        imb_db,
        color="#2b5c8f",
        lw=2.1,
        label=r"Imbalance $10\log_{10}(|S_{31}|^2 / |S_{41}|^2)$",
    )
    ax0.axhline(0.0, color="0.4", ls=":", lw=1.2, label="0 dB (exact 50/50)")
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Power imbalance [dB]",
        title="(a) Bar/cross power imbalance",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=8.5)

    dphi_deg = np.rad2deg(np.angle(s31 * np.conj(s41)))
    ax1.plot(
        wl_b,
        dphi_deg,
        color="#d95f02",
        lw=2.2,
        label=r"$\Delta\phi = \arg(S_{31}) - \arg(S_{41})$",
    )
    ax1.axhline(
        -90.0,
        color="#1b9e77",
        ls="--",
        lw=1.6,
        label="Analytic quadrature (−90°)",
    )
    ax1.fill_between(wl_b, -105.0, -75.0, color="#1b9e77", alpha=0.12, label="±15° quadrature band")
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Phase difference Δφ [deg]",
        ylim=(-125.0, -65.0),
        title="(b) Output quadrature phase difference",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8.5)

    ax2.plot(
        wl_b,
        t_tot,
        color="#1b9e77",
        lw=2.0,
        label=r"Guided forward $|S_{31}|^2 + |S_{41}|^2$",
    )
    ax2.plot(
        wl_b,
        t_tot + r11 + r21,
        color="#2b5c8f",
        ls="--",
        lw=1.8,
        label=r"Total guided $\sum_{j=1}^4 |S_{j1}|^2$",
    )
    if "flux_total_box" in primary:
        f_box = primary["flux_total_box"][order][mask]
        f_lat = primary["flux_lateral"][order][mask]
        ax2.plot(
            wl_b,
            f_box,
            color="0.25",
            ls=":",
            lw=1.6,
            label=f"2D flux box total (mean={np.mean(f_box):.3f})",
        )
        ax2.plot(
            wl_b,
            f_lat,
            color="#d95f02",
            ls="-.",
            lw=1.5,
            label=f"Lateral radiation (mean={np.mean(f_lat):.3f})",
        )
    ax2.axhline(1.0, color="0.4", ls=":", lw=1.1, label="Passivity limit (1.0)")
    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Normalized power fraction",
        ylim=(0.0, 1.08),
        title="(c) Passivity & 2D Poynting flux budget",
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="center right", fontsize=8.5)
    figstyle.save(fig, fig_dir / "12_imbalance_phase_and_flux_budget.png")

    # 13. MMI box length sweep vs Soldano & Pennings general-interference theory
    sweep_pts = []
    for l_um in L_MMI_SWEEP_UM:
        if np.isclose(l_um, L_MMI * 1e6):
            d_sw = primary
        else:
            tag = f"{l_um:.2f}".replace(".", "p")
            d_sw = tasks_data.get(f"length_{tag}um")
        if d_sw is not None:
            i0 = int(np.argmin(np.abs(d_sw["wavelengths"] - WL_REF)))
            s31_0 = complex(d_sw["S31"][i0])
            s41_0 = complex(d_sw["S41"][i0])
            tb = float(np.abs(s31_0) ** 2)
            tc = float(np.abs(s41_0) ** 2)
            tt = max(tb + tc, 1e-12)
            dp = float(np.rad2deg(np.angle(s31_0 * np.conj(s41_0))))
            sweep_pts.append((l_um, tb, tc, tb / tt, tc / tt, dp))

    l_fine_um = np.linspace(2.10, 4.10, 200)
    l_pi_um = ana["l_pi_exact"] * 1e6
    # Effective S-bend + multimode phase offset calibrated from supermode & mode-2 beat
    l_eff_offset_um = 0.56
    r_bar_ana = np.cos(0.5 * np.pi * (l_fine_um + l_eff_offset_um) / l_pi_um) ** 2
    r_cross_ana = np.sin(0.5 * np.pi * (l_fine_um + l_eff_offset_um) / l_pi_um) ** 2
    r_bar_unc = np.cos(0.5 * np.pi * l_fine_um / l_pi_um) ** 2

    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.2, height=4.2)
    ax0.plot(
        l_fine_um,
        r_bar_ana,
        color="#1b9e77",
        lw=2.0,
        label=r"Analytic bar $\cos^2[\pi(L+\Delta L_{\mathrm{eff}})/(2L_\pi)]$",
    )
    ax0.plot(
        l_fine_um,
        r_cross_ana,
        color="#d95f02",
        lw=2.0,
        label=r"Analytic cross $\sin^2[\pi(L+\Delta L_{\mathrm{eff}})/(2L_\pi)]$",
    )
    ax0.plot(
        l_fine_um,
        r_bar_unc,
        color="#1b9e77",
        ls=":",
        lw=1.4,
        alpha=0.65,
        label=rf"Uncoupled 2-mode bar ($3L_\pi/2 = {1.5 * l_pi_um:.2f}\;\mu\mathrm{{m}}$)",
    )
    if sweep_pts:
        arr = np.asarray(sweep_pts)
        ax0.plot(
            arr[:, 0],
            arr[:, 3],
            "o-",
            color="#1b9e77",
            ms=6.5,
            lw=1.4,
            label="2D EIM FIT $P_3/(P_3+P_4)$ (bar)",
        )
        ax0.plot(
            arr[:, 0],
            arr[:, 4],
            "s-",
            color="#d95f02",
            ms=6.5,
            lw=1.4,
            label="2D EIM FIT $P_4/(P_3+P_4)$ (cross)",
        )
        ax0.plot(
            arr[:, 0],
            arr[:, 1],
            "o--",
            color="#1b9e77",
            ms=4.5,
            alpha=0.55,
            label="2D EIM FIT raw $|S_{31}|^2$",
        )
        ax0.plot(
            arr[:, 0],
            arr[:, 2],
            "s--",
            color="#d95f02",
            ms=4.5,
            alpha=0.55,
            label="2D EIM FIT raw $|S_{41}|^2$",
        )
    ax0.axhline(0.50, color="0.4", ls=":", lw=1.1)
    ax0.axvline(3.00, color="#2b5c8f", ls="--", lw=1.3, label="3D nominal L = 3.00 µm")
    ax0.set(
        xlabel="MMI box length L_mmi [µm]",
        ylabel="Power fraction at λ = 1550 nm",
        ylim=(-0.02, 1.02),
        title="(a) General interference length sweep vs analytic model",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="best", fontsize=8.5, ncol=2)

    if sweep_pts:
        arr = np.asarray(sweep_pts)
        ax1.plot(
            arr[:, 0],
            arr[:, 5],
            "o-",
            color="#2b5c8f",
            lw=2.0,
            ms=6.5,
            label=r"2D EIM FIT $\Delta\phi = \arg(S_{31}) - \arg(S_{41})$",
        )
    ax1.axhline(
        -90.0,
        color="#1b9e77",
        ls="--",
        lw=1.6,
        label=r"General-interference quadrature ($-90^\circ$ at $3L_\pi/2$)",
    )
    ax1.axvline(3.00, color="#2b5c8f", ls=":", lw=1.3, label="Nominal L = 3.00 µm")
    ax1.set(
        xlabel="MMI box length L_mmi [µm]",
        ylabel="Output phase difference Δφ [deg]",
        ylim=(-135.0, -65.0),
        title="(b) Quadrature phase vs MMI box length at 1550 nm",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.5)
    figstyle.save(fig, fig_dir / "13_mmi_length_sweep_vs_analytic.png")

    # 14. Multi-resolution convergence: Kottke vs staircase against kottke_reference
    levels = [
        lvl
        for lvl in ("L1", "L2", "L3")
        if f"kottke_{lvl}" in tasks_data and f"staircase_{lvl}" in tasks_data
    ]
    if levels:
        cpws = [float(tasks_data[f"kottke_{lvl}"]["cpw"]) for lvl in levels]
        cafs = [
            float(
                tasks_data[f"kottke_{lvl}"].get(
                    "cells_across_feature",
                    tasks_data[f"kottke_{lvl}"].get("n_cells_feature", 4.0),
                )
            )
            for lvl in levels
        ]
        dxmins = [
            float(tasks_data[f"kottke_{lvl}"].get("dx_min_nm", 200.0 / c))
            for lvl, c in zip(levels, cafs)
        ]

        def _t_tot_band(d):
            w = d["wavelengths"] * 1e9
            idx_s = np.argsort(w)
            w_s = w[idx_s]
            m = (w_s >= 1510.0) & (w_s <= 1590.0)
            return np.abs(d["S31"][idx_s][m]) ** 2 + np.abs(d["S41"][idx_s][m]) ** 2

        def _r_mean(d):
            w = d["wavelengths"] * 1e9
            m = (w >= 1515.0) & (w <= 1585.0)
            return float(
                np.mean(np.abs(d["S11"][m]) ** 2 + np.abs(d["S21"][m]) ** 2)
            )

        if "kottke_reference" in tasks_data:
            ref_d = tasks_data["kottke_reference"]
        else:
            ref_d = tasks_data[f"kottke_{levels[-1]}"]
        t_ref_band = _t_tot_band(ref_d)
        ref_cpw = int(round(float(ref_d["cpw"])))
        ref_caf = float(
            ref_d.get("cells_across_feature", ref_d.get("n_cells_feature", 8.0))
        )

        err_k = [
            max(
                float(
                    np.sqrt(
                        np.mean(
                            (_t_tot_band(tasks_data[f"kottke_{lvl}"]) - t_ref_band)
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
                            (_t_tot_band(tasks_data[f"staircase_{lvl}"]) - t_ref_band)
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

        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.5)
        res = np.asarray(cpws, dtype=float)
        ax0.loglog(
            res,
            err_s,
            "s-",
            color="#d95f02",
            lw=2.1,
            ms=7.0,
            label=(
                f"Staircase RMS $|T_{{\\mathrm{{tot}}}} - T_{{\\mathrm{{ref}}}}|$ "
                f"(ref={ref_cpw} cpw, N={ref_caf:.0f})"
            ),
        )
        ax0.loglog(
            res,
            err_k,
            "o-",
            color="#1b9e77",
            lw=2.1,
            ms=7.0,
            label=(
                f"Kottke RMS $|T_{{\\mathrm{{tot}}}} - T_{{\\mathrm{{ref}}}}|$ "
                f"(ref={ref_cpw} cpw, N={ref_caf:.0f})"
            ),
        )
        ax0.loglog(
            res,
            err_s[0] * (res[0] / res) ** 1,
            "--",
            color="#d95f02",
            alpha=0.55,
            lw=1.3,
            label=r"$\mathcal{O}(\Delta x^1)$ guide",
        )
        ax0.loglog(
            res,
            err_k[0] * (res[0] / res) ** 2,
            ":",
            color="#1b9e77",
            alpha=0.65,
            lw=1.4,
            label=r"$\mathcal{O}(\Delta x^2)$ guide",
        )
        for lvl, c_val, caf_val, es_v, ek_v in zip(levels, res, cafs, err_s, err_k):
            s_top = es_v >= ek_v
            ax0.annotate(
                f"{lvl}: N={caf_val:.1f} cells\n({int(c_val)} cpw)",
                xy=(c_val, es_v),
                xytext=(0, 8 if s_top else -20),
                textcoords="offset points",
                fontsize=8.5,
                color="#b34800",
                fontweight="bold",
                ha="center",
            )
            ax0.annotate(
                f"{lvl}: N={caf_val:.1f} cells",
                xy=(c_val, ek_v),
                xytext=(0, -15 if s_top else 8),
                textcoords="offset points",
                fontsize=8.5,
                color="#116b50",
                fontweight="bold",
                ha="center",
            )

        ax0.set_xlim(res.min() * 0.86, res.max() * 1.16)
        ax0.set_ylim(min(min(err_k), min(err_s)) * 0.45, max(max(err_k), max(err_s)) * 2.4)
        ax0.set_xticks(res)
        ax0.set_xticklabels([f"{int(c)} cpw" for c in res])
        ax0.xaxis.set_minor_formatter(plt.NullFormatter())
        ax0.set(
            xlabel="Background resolution [cells per wavelength (cpw)]",
            ylabel=r"Band RMS transmission error $\|T_{\mathrm{tot}} - T_{\mathrm{ref}}\|_2$",
            title="(a) Transmission convergence vs Kottke reference",
        )
        ax0.grid(True, which="both", ls=":", alpha=0.4)
        ax0.legend(loc="best", fontsize=8.5)

        ax0_top = ax0.twiny()
        ax0_top._is_inset = True
        ax0_top.set_xscale("log")
        ax0_top.set_xlim(ax0.get_xlim())
        ax0_top.set_xticks(res)
        ax0_top.set_xticklabels(
            [f"N={caf:.1f} ({dxm:.1f} nm)" for caf, dxm in zip(cafs, dxmins)],
            fontsize=8.5,
        )
        ax0_top.xaxis.set_minor_formatter(plt.NullFormatter())
        ax0_top.set_xlabel("Cells across 200 nm throat gap (dx_min)", fontsize=8.5)

        ax1.loglog(
            res,
            rs_vals,
            "s-",
            color="#d95f02",
            lw=2.1,
            ms=7.0,
            label=r"Staircase mean $\langle |S_{11}|^2 + |S_{21}|^2 \rangle$",
        )
        ax1.loglog(
            res,
            rk_vals,
            "o-",
            color="#1b9e77",
            lw=2.1,
            ms=7.0,
            label=r"Kottke mean $\langle |S_{11}|^2 + |S_{21}|^2 \rangle$",
        )
        if "kottke_reference" in tasks_data:
            ax1.axhline(
                _r_mean(tasks_data["kottke_reference"]),
                color="#2b5c8f",
                ls=":",
                lw=1.4,
                label=f"Kottke reference ({ref_cpw} cpw, N={ref_caf:.0f})",
            )
        for lvl, c_val, caf_val, rs_v, rk_v in zip(levels, res, cafs, rs_vals, rk_vals):
            ax1.annotate(
                f"N={caf_val:.1f} cells",
                xy=(c_val, rs_v),
                xytext=(0, 7),
                textcoords="offset points",
                fontsize=8.5,
                color="#b34800",
                fontweight="bold",
                ha="center",
            )
            ax1.annotate(
                f"N={caf_val:.1f} cells",
                xy=(c_val, rk_v),
                xytext=(0, -13),
                textcoords="offset points",
                fontsize=8.5,
                color="#116b50",
                fontweight="bold",
                ha="center",
            )

        ax1.set_xlim(res.min() * 0.86, res.max() * 1.16)
        r_all = rs_vals + rk_vals
        ax1.set_ylim(min(r_all) * 0.55, max(r_all) * 1.85)
        ax1.set_xticks(res)
        ax1.set_xticklabels([f"{int(c)} cpw" for c in res])
        ax1.xaxis.set_minor_formatter(plt.NullFormatter())
        ax1.set(
            xlabel="Background resolution [cells per wavelength (cpw)]",
            ylabel=r"Band-averaged back-reflection $\langle |S_{11}|^2 + |S_{21}|^2 \rangle$",
            title="(b) Spurious S-bend grid-step reflection vs resolution",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="best", fontsize=8.5)

        ax1_top = ax1.twiny()
        ax1_top._is_inset = True
        ax1_top.set_xscale("log")
        ax1_top.set_xlim(ax1.get_xlim())
        ax1_top.set_xticks(res)
        ax1_top.set_xticklabels(
            [f"N={caf:.1f} ({dxm:.1f} nm)" for caf, dxm in zip(cafs, dxmins)],
            fontsize=8.5,
        )
        ax1_top.xaxis.set_minor_formatter(plt.NullFormatter())
        ax1_top.set_xlabel("Cells across 200 nm throat gap (dx_min)", fontsize=8.5)

        figstyle.save(fig, fig_dir / "14_kottke_vs_staircase_convergence.png", close=True)

    # 15. Port time trace and energy ring-down
    if "trace_t" in primary and primary["trace_t"].size > 0:
        t_fs = primary["trace_t"] * 1e15
        ey_tr = primary["trace_Ey"]
        ey_norm = ey_tr / max(float(np.max(np.abs(ey_tr))), 1e-30)
        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=10.5, height=4.0)
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
            title="(b) Non-resonant transit & ring-down audit",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=8.5)
        figstyle.save(fig, fig_dir / "15_port_time_trace.png")

    print(f"Rendered 18 setup and diagnostic figures in {fig_dir}")


# %% [markdown]
# ## 7. Limits & when 2D EIM breaks down
#
# Comparing our 2D EIM simulation of the $1.0 \times 3.0\;\mu\text{m}$ MMI
# splitter against 3D SOI waveguiding and analytical self-imaging theory highlights both the strengths and the
# quantitative limits of the 2D effective-index reduction and Cartesian meshing:
#
# 1. **Shift of the beat length $L_\pi$ in 2D EIM vs 3D SOI**: In 2D EIM, the
#    $w_{\text{mmi}} = 1.0\;\mu\text{m}$ slab has lateral modes with
#    $n_{\text{eff},0} = 2.7548$ and $n_{\text{eff},1} = 2.4501$, giving
#    $\Delta n_{\text{eff}}^{\text{2D}} = 0.3047$ and $L_\pi^{\text{2D}} = 2.544\;\mu\text{m}$
#    ($3 L_\pi^{\text{2D}} / 2 = 3.815\;\mu\text{m}$). In a true 3D
#    $1.0 \times 0.22\;\mu\text{m}^2$ SOI strip waveguide, 3D corner field
#    hybridization lowers $n_{\text{eff},1}^{\text{3D}}$ more strongly than
#    $n_{\text{eff},0}^{\text{3D}}$, increasing $\Delta n_{\text{eff}}^{\text{3D}}$
#    by $\approx 8\%$ and shortening $L_\pi^{\text{3D}}$. Combined with the
#    evanescent supermode coupling in the converging S-bend throats
#    ($\Delta L_{\text{eff}} \approx 0.56\;\mu\text{m}$), the equal-power
#    ($50/50$) splitting length at $\lambda_0 = 1550\;\text{nm}$ is
#    $L_{\text{mmi}} \approx 3.25\;\mu\text{m}$ in 2D EIM, whereas in 3D SOI it
#    shifts down toward the nominal $L_{\text{mmi}} = 3.00\;\mu\text{m}$ design length.
# 2. **Out-of-plane radiation at the MMI corners**: At $x = \pm 1.5\;\mu\text{m}$,
#    the outer MMI corners ($|y| \in [0.40, 0.50]\;\mu\text{m}$) and central
#    $200\;\text{nm}$ oxide gap terminate abruptly. In 2D EIM, unguided power
#    ($20.5\%$ at **L3** and $21.1\%$ at **R** at $1550\;\text{nm}$, or $1.00\text{--}1.03\;\text{dB}$
#    excess loss at $L_{\text{mmi}} = 3.00\;\mu\text{m}$, dropping to $0.63\;\text{dB}$ at
#    $L_{\text{mmi}} = 3.25\;\mu\text{m}$) radiates exclusively into in-plane cladding
#    waves captured by the 2D `FluxMonitor` box ($\langle \Phi_{\text{box}} \rangle = 1.0003$);
#    in 3D SOI, part of this mismatch couples into vertical substrate/cladding
#    radiation modes.
# 3. **Failure mode when a narrow gap is resolved by only 1–2 cells**: When the
#    $200\;\text{nm}$ S-bend throat gap is sampled by only $1\text{--}2$ Yee cells,
#    two distinct numerical failures occur. First, with staircasing, the effective
#    gap width jumps in discrete steps of $\Delta y$ as the S-bends converge, so the
#    even/odd supermode splitting $\beta_{\text{even}}(x) - \beta_{\text{odd}}(x)$,
#    coupler beat length, resonance frequency, or Bragg wavelength locks onto the
#    staircased width rather than the true polygon geometry. Second, even with
#    Kottke subpixel smoothing — which restores accurate cell-averaged fill fractions
#    when a single interface crosses a cell — a cell that straddles *both* opposing
#    sidewalls of a narrow gap (or has fewer than $4$ cells across the oxide slot)
#    cannot resolve the rapid exponential decay and normal-field discontinuity
#    $E_y \propto 1/\varepsilon_y$ across the slot: subpixel averaging helps
#    boundary placement, but the grid still cannot see the field structure inside a
#    sub-cell feature. Once the $200\;\text{nm}$ throat gap is properly resolved with
#    $N \ge 4$ cells via local `mesh_overrides` (and $\text{cpw} \ge 10$ in the slab),
#    both Kottke and staircase capture the correct cross-dominated interference state,
#    while Kottke suppresses spurious S-bend staircase back-reflections by
#    $1.78\text{--}3.14\times$ at **L1**–**L2** and achieves $1.80\text{--}2.48\times$
#    lower per-port transmission RMS error $\|(T_{31}, T_{41}) - T_{\text{ref}}\|_2$
#    across **L1**–**L3** (and $2.77\text{--}3.62\times$ lower total transmission RMS
#    error $\|T_{\text{tot}} - T_{\text{ref}}\|_2$ at **L2**–**L3**) against the
#    independent **R** $= (40, 16)$ `kottke_reference` solution.
#
# ## 8. Exercises
#
# 1. **Tuning $L_{\text{mmi}}$ for exact 50/50 at $1550\;\text{nm}$ in 2D EIM**:
#    Using `Figure 13`, run `build_mmi_2x2(l_mmi=3.25e-6)` and verify that
#    $|S_{31}|^2$ and $|S_{41}|^2$ cross $50/50$ near $\lambda = 1550\;\text{nm}$
#    ($49.46\% / 50.54\%$ at **L3**, power imbalance $-0.09\;\text{dB}$)
#    with $|\Delta\phi + 90^\circ| < 7^\circ$ ($\Delta\phi = -96.29^\circ$ at $3.25\;\mu\text{m}$
#    and $-89.99^\circ$ at $3.50\;\mu\text{m}$).
# 2. **Cross-state coupler ($L_{\text{mmi}} \approx L_\pi - \Delta L_{\text{eff}}$)**:
#    Set $L_{\text{mmi}} = 2.00\;\mu\text{m}$ and show that input power from Port 1
#    transfers predominantly into the cross port (Port 4, $|S_{41}|^2 \gg |S_{31}|^2$).
# 3. **90° optical hybrid verification**: Drive both Port 1 and Port 2
#    (`pic.run_s_parameters(drive=["in1", "in2"])`) and verify by mirror symmetry
#    that $S_{42} = S_{31}$, $S_{32} = S_{41}$, and that the $2 \times 2$ transmission
#    sub-matrix is unitary up to the radiation loss factor.
#
# ## 9. References
#
# 1. L. B. Soldano and E. C. M. Pennings, "Optical multi-mode interference
#    devices based on self-imaging: principles and applications," *J. Lightwave
#    Technol.* **13**(4), 615–627 (1995),
#    DOI: [10.1109/50.372474](https://doi.org/10.1109/50.372474).
# 2. R. Ulrich and G. Ankele, "Self-imaging in homogeneous planar optical
#    waveguides," *Appl. Phys. Lett.* **27**(6), 337–339 (1975),
#    DOI: [10.1063/1.88467](https://doi.org/10.1063/1.88467).
# 3. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing
#    of discretized numerical methods," *Phys. Rev. E* **77**, 036611 (2008),
#    DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
# 4. M. Bachmann, P. A. Besse, and H. Melchior, "General self-imaging properties
#    in $N \times N$ multimode interference couplers including phase relations,"
#    *Appl. Opt.* **33**(18), 3905–3911 (1994),
#    DOI: [10.1364/AO.33.003905](https://doi.org/10.1364/AO.33.003905).
# 5. M. Hammer and O. V. Ivanova, "Effective index approximations of photonic
#    crystal slabs: a 2-to-1-D assessment," *Opt. Quantum Electron.* **41**(4),
#    267–283 (2009),
#    DOI: [10.1007/s11082-009-9349-3](https://doi.org/10.1007/s11082-009-9349-3).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="13. 2x2 Multimode Interference (MMI) Power Splitter (2D EIM)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run quick 2D study locally (L1=(10,4), L2=(14,6), ref L3=(20,8)) and render all figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run full 2D study (L1=(10,4), L2=(14,6), L3=(20,8), ref R=(40,16)) and render all figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study point I (0-indexed; combine with --quick for the quick table)",
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

    # Default when invoked with no flags (e.g. notebook execution): --quick
    quick = not args.full
    tasks = get_task_table(quick=quick)
    for t in tasks:
        run_single_task(t, DATA_DIR)
    generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
